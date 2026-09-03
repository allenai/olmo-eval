"""Rubric judge for GDP.pdf, mirroring the official Surge AI harness.

Faithful to ``surge-ai/gdp-pdf`` ``src/gdp_pdf/scorer.py``: one judge call per
rubric criterion, the judge seeing only the response and that single criterion
(never the prompt or the document), a JSON verdict of ``score`` (the string
``"0"`` or ``"1"``) plus ``rationale``, strict parsing (fences stripped, strict
JSON, schema validation, decimal coercion), a criterion passing iff
``score.round(4) >= 1``, and five attempts before a criterion is left
*unscored*. Unscored criteria leave the ``mean_criteria`` denominator and drop
their whole response from ``all_pass``, so a partly graded response is reported
as missing rather than as a failure.

Two deviations from the official harness, both deliberate:

* The judge model is configurable (``GDP_PDF_JUDGE_MODEL``) and defaults to an
  OpenAI model rather than the leaderboard's Gemini, so results are not
  directly leaderboard-comparable unless the official judge is selected.
* Attempts wait a randomized exponential backoff. Without one, a transient API
  condition exhausts every attempt at once and silently drops criteria.

Validated replies are cached on disk so re-runs are free; the cache dir comes
from ``GDP_PDF_JUDGE_CACHE_DIR`` or a fresh per-process temp dir. Requires
``OPENAI_API_KEY`` on cache misses.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import tempfile
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gpt-4o-2024-08-06"

#: Attempts after the first when a judge reply fails to parse (official: 4).
MAX_RETRIES = 4

_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 30.0

_PROCESS_CACHE_DIR: list[str] = []
_ASYNC_CLIENTS: dict[str, Any] = {}

#: Verbatim from the official harness; the placeholders are substituted raw.
JUDGE_PROMPT_TEMPLATE = """Read this json object:
{
  response: {{ rubric.response_to_grade }},
  criteria: {{ rubric.criterion }},
}

Score the response based on the criteria.
If the response perfectly satisfies the criteria, give it a score of "1".
If it does not satisfy the criteria at all, give it a score of "0".
Return the score and rationale in a JSON object with two keys, 'score' and 'rationale'.
'score' should be a string that's either "0" or "1".
'rationale' should be a description of why you gave the response that score.

Here are some examples of how to score a response:
{ score: "1", rationale: "The response is a perfect match for the criteria." }
{ score: "0", rationale: "The response does not satisfy the criteria at all." }
Unlike these examples, when you create a rationale, be specific and detailed. Give reasons specific to this response and criteria. Don't be generic."""  # noqa: E501

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "individual_criteria_score",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["rationale", "score"],
            "properties": {
                "rationale": {"type": "string"},
                "score": {"type": "string", "enum": ["0", "1"]},
            },
            "additionalProperties": False,
        },
    },
}


def default_judge_model() -> str:
    return os.environ.get("GDP_PDF_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL


def default_judge_cache_dir() -> str:
    """Judge-response cache dir: env override or a fresh process-local temp dir."""
    env_dir = os.environ.get("GDP_PDF_JUDGE_CACHE_DIR")
    if env_dir:
        return env_dir
    if not _PROCESS_CACHE_DIR:
        _PROCESS_CACHE_DIR.append(tempfile.mkdtemp(prefix="gdp-pdf-judge-cache-"))
    return _PROCESS_CACHE_DIR[0]


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()


def _get_client(model: str) -> Any:
    if model not in _ASYNC_CLIENTS:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai") from None
        _ASYNC_CLIENTS[model] = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _ASYNC_CLIENTS[model]


def _retry_delay(attempt: int) -> float:
    """Randomized exponential backoff, in seconds, for judge attempt ``attempt``."""
    return min(_RETRY_MAX_DELAY, _RETRY_BASE_DELAY * 2 ** (attempt - 1)) * (0.5 + random.random())


class ParseError(ValueError):
    """Judge output failed strict parsing/validation."""


def build_judge_prompt(response_text: str, criterion: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.replace("{{ rubric.response_to_grade }}", response_text).replace(
        "{{ rubric.criterion }}", criterion
    )


def parse_verdict(raw_text: str) -> dict[str, Any]:
    """Strict parse of one judge reply into ``{"score": Decimal, "rationale": str}``."""
    cleaned = raw_text.strip() if raw_text else ""
    cleaned = re.sub(r"\A```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\Z", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e

    if not isinstance(payload, dict):
        raise ParseError("Judge output must be a JSON object")
    if "score" not in payload or "rationale" not in payload:
        raise ParseError("Judge output must contain 'score' and 'rationale'")
    raw_score = payload["score"]
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float, str)):
        raise ParseError("'score' must be a number or string")
    if not isinstance(payload["rationale"], str):
        raise ParseError("'rationale' must be a string")
    try:
        score = Decimal(str(raw_score).strip())
    except InvalidOperation as e:
        raise ParseError(f"'score' is not a valid decimal: {raw_score!r}") from e
    return {"score": score, "rationale": payload["rationale"]}


def criterion_passes(score: Decimal) -> bool:
    """Official pass condition: ``score.round(4) >= 1`` (half-up)."""
    return score.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) >= 1


async def grade_criterion(
    response_text: str,
    criterion: str,
    *,
    model: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, Any] | None:
    """Grade one criterion. Returns ``None`` when every attempt failed to parse."""
    model = model or default_judge_model()
    cache_dir = cache_dir or default_judge_cache_dir()
    prompt = build_judge_prompt(response_text, criterion)
    key = _cache_key(model, prompt)
    cache_file = Path(cache_dir) / f"{key}-v1.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        return {"score": Decimal(cached["score"]), "rationale": cached["rationale"]}

    client = _get_client(model)
    raw: str | None = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                response_format=_RESPONSE_FORMAT,
                n=1,
                temperature=0,
                top_p=1,
                seed=42,
            )
            raw = completion.choices[0].message.content
            verdict = parse_verdict(raw or "")
        except Exception as e:
            logger.warning("GDP.pdf grading error: %s", e)
            if attempt <= MAX_RETRIES:
                await asyncio.sleep(_retry_delay(attempt))
            continue

        os.makedirs(cache_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(".tmp", prefix=f"{key}-v1.json", text=True, dir=cache_dir)
        os.close(fd)
        with open(tmp, "w") as f:
            json.dump({"score": str(verdict["score"]), "rationale": verdict["rationale"]}, f)
        os.rename(tmp, str(cache_file))
        return verdict

    logger.warning(
        "GDP.pdf grading failed after %d attempts; last reply: %.200s", MAX_RETRIES + 1, raw
    )
    return None


@dataclass(frozen=True)
class GdpPdfRubricScorer(Scorer):
    """Score channel for the GDP.pdf rubric judge.

    Grading happens task-level (one call per criterion, gathered under the
    runner's scoring concurrency) and stores ``score:gdp_pdf`` per output; this
    scorer exposes the stored value so metric plumbing stays uniform.
    """

    name: str = "gdp_pdf"
    model: str = field(default_factory=default_judge_model)
    cache_dir: str = field(default_factory=default_judge_cache_dir)

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get("score:gdp_pdf", 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0
