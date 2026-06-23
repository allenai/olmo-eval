"""GPT-judge scorer for pixmo-cap dense-caption evaluation.

Ports the scoring logic from mm_olmo/scripts/gpt_dense_caption_eval.py into
the olmo-eval-internal ContextScorer abstraction.  The cache-key scheme is
byte-identical to the legacy Gpt4WithCache so the existing gpt4-cache/ files
are reused for offline/reproducible runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.scorers.execution import ContextScorer
from olmo_eval.common.types import Instance, LMOutput

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = "/weka/oe-training-default/mm-olmo/dense_caption_eval/gpt4-cache"

# Labels that GPT returns instead of Consistent/Inconsistent; skip them silently.
_UNKNOWN_CONSISTENCY_LABELS = [
    "not specified",
    "cannot determine",
    "not determinable",
    "no verification",
    "n/a",
    "not confirmed",
    "neither",
    "not stated",
    "no judgement",
    "unable to determine",
    "inconclusive",
    "undetermined",
    "insufficient information",
    "no relevant information",
    "no conclusion",
    "not clear",
    "unknown",
    "uncertain",
    "ambiguous",
    "not addressed",
    "not enough information",
    "not mentioned",
    "not enough info",
    "no information",
    "not verifiable",
    "not applicable",
]
_UNKNOWN_PATTERN = re.compile(
    r".*\b(" + "|".join(re.escape(s) for s in _UNKNOWN_CONSISTENCY_LABELS) + r").*$",
    re.IGNORECASE,
)

# Module-level lazy async clients, keyed by model name.
_ASYNC_CLIENTS: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Cache helpers (identical semantics to legacy Gpt4WithCache)
# ---------------------------------------------------------------------------


def _compute_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_key(model: str, prompt: str) -> str:
    kwargs = {"temperature": 0}
    return _compute_hash(
        model + "::::" + json.dumps(prompt) + "::::" + json.dumps(kwargs, sort_keys=True)
    )


async def _cached_gpt_call(
    prompt: str,
    *,
    model: str,
    cache_dir: str,
    cache_only: bool,
    recompute: bool = False,
) -> str:
    """Async GPT call with file-based caching compatible with legacy gpt4-cache/.

    When ``recompute=True`` an existing cache entry is ignored and a fresh API
    call is made; the new result overwrites the old cache file.
    """
    key = _cache_key(model, prompt)
    cache_file = Path(cache_dir) / f"{key}-v1.json"

    if not recompute and cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        return data["choices"][0]["message"]["content"]

    if cache_only:
        raise ValueError(f"Cache miss (cache_only=True) for key {key[:16]}…")

    if model not in _ASYNC_CLIENTS:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for DenseCaptionJudgeScorer on a cache miss."
            )
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai") from None
        _ASYNC_CLIENTS[model] = AsyncOpenAI(api_key=api_key)

    client = _ASYNC_CLIENTS[model]
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    completion = response.model_dump()

    # Atomic write: tmp → rename, identical to legacy Gpt4WithCache.
    # Ensure the cache dir exists so a user-supplied (e.g. MATHVISTA_GPT_CACHE_DIR) path that
    # hasn't been created yet doesn't fail every GPT call with "No such file or directory".
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(".tmp", prefix=f"{key}-v1.json", text=True, dir=cache_dir)
    os.close(fd)
    with open(tmp, "w") as f:
        json.dump(completion, f)
    os.rename(tmp, str(cache_file))

    return completion["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# GPT prompt builders (verbatim from gpt_dense_caption_eval.py)
# ---------------------------------------------------------------------------


def _recall_prompt(mturk_statements: str, caption: str) -> str:
    return (
        "Here are statements that annotators gave for an image.\n\n"
        + mturk_statements.strip()
        + (
            "\n\nNext, consider the following caption of the image. For each statement above,"
            ' state whether the fact is "Stated" or "Not Stated" in the caption.'
            " The output should be in the form\n\n1. Stated\n2. Not Stated\n3. Stated\n\n"
            "Do not output anything other than an ordered list of Stated and Not Stated.\n\n"
            " Here is the caption: "
        )
        + (caption.strip() if caption else "No caption provided.")
    )


def _canonical_prompt(caption: str) -> str:
    return (
        "Based on the description of the image, come up with a list of the MOST canonical"
        " statements that are mentioned in it. Each statement should be broken down as much"
        " as possible. The statements should be an ordered list, where each item is separated"
        " a newline. For instance, the rseponse may look like:\n\n"
        "1. Statement A\n2. Statement B\n3. Statement C\n\n\n"
        f"\n\n\nHere is the image description: {caption}"
    )


def _consistency_prompt(num_transcripts: int, transcripts_str: str, statements_str: str) -> str:
    return (
        f"Here are {num_transcripts} captions people gave for an image using their voice.\n\n"
        + transcripts_str
        + (
            "\n\nHere are statements that a captioning model made about the image."
            ' For each statement, state whether it\'s "Consistent" or "Inconsistent"'
            " with the statements provided above. The output should be in the form\n\n"
            "1. Consistent\n2. Inconsistent\n3. Consistent\n\n"
            "Do not output anything other than an ordered list of Consistent and Inconsistent.\n\n"
        )
        + statements_str
    )


# ---------------------------------------------------------------------------
# Parse helpers (verbatim logic from gpt_dense_caption_eval.py)
# ---------------------------------------------------------------------------


def parse_recall_output(text: str) -> tuple[int, int]:
    """Parse GPT stated/not-stated output.

    Returns (num_covered, num_statements) counting only unambiguous lines.
    Mirrors eval_recall() lines 323–346 in gpt_dense_caption_eval.py.
    """
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    valid_scores: list[bool] = []
    for line in lines:
        if re.fullmatch(r".*\bnot st[a-z]+$", line, flags=re.IGNORECASE):
            valid_scores.append(False)
        elif " stated" in line.lower():
            valid_scores.append(True)
        # else: ambiguous line — skip (like legacy code)
    return int(sum(valid_scores)), len(valid_scores)


def parse_consistency_output(text: str) -> tuple[int, int]:
    """Parse GPT consistent/inconsistent output.

    Returns (num_consistent, num_valid) counting only unambiguous lines.
    Mirrors eval_consistency() lines 403–461 in gpt_dense_caption_eval.py.
    """
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    valid_scores: list[bool] = []
    for line in lines:
        inconsistent: bool | None = None
        if re.fullmatch(
            r".*[^a-z]((i?inconsis?ten(t|cy)?)|incorrect|inconsistence|iconsistent"
            r"|inconsisent|incomplete|contradictory).*",
            line,
            flags=re.IGNORECASE,
        ):
            inconsistent = True
        if re.fullmatch(
            r".*[^a-z](consistent(ly)?|constistent|correct).*$",
            line,
            flags=re.IGNORECASE,
        ):
            # both matched — treat as ambiguous (None); otherwise consistent (False)
            inconsistent = None if inconsistent else False
        if inconsistent is None:
            if not _UNKNOWN_PATTERN.match(line):
                logger.warning("Unexpected consistency label: %r", line)
            continue
        valid_scores.append(inconsistent)
    num_consistent = sum(not x for x in valid_scores)
    return num_consistent, len(valid_scores)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseCaptionJudgeScorer(ContextScorer):
    """GPT-as-judge scorer for pixmo-cap dense-caption evaluation.

    Runs up to three GPT calls per example (recall stated-check, canonical
    statements, consistency check) and stashes all per-example results in
    ``output.metadata["dense_caption_result"]``.  The primary ``float``
    return value is the raw recall ratio (0–1) for that output, or 0.0 if
    the example is invalid.

    Cache keys are byte-identical to the legacy ``Gpt4WithCache`` in
    ``mm_olmo/scripts/gpt_dense_caption_eval.py``, so existing ``gpt4-cache/``
    entries are reused automatically.

    ``instance.metadata`` must contain:
        - ``mturk_statements`` (str): canonical_statements string from
          ``mturk-eval-statements/{sha256(url)}.json``.
        - ``transcripts`` (list[dict]): dicts with a ``"whisperTranscript"``
          key, from ``final-data.json``.
    """

    name: str = "dense_caption_judge"

    model: str = "gpt-4o-2024-05-13"
    cache_dir: str = _DEFAULT_CACHE_DIR
    cache_only: bool = False
    recompute: bool = False
    target_metrics: tuple[str, ...] = ("recall", "consistency")

    async def ascore_with_context(
        self,
        instance: Instance,
        output: LMOutput,
        context: ScoringContext,
    ) -> float:
        caption = (output.extracted_answer or output.text or "").strip()
        mturk_statements: str = instance.metadata.get("mturk_statements", "")
        transcripts: list[dict] = instance.metadata.get("transcripts", [])
        transcripts_str = "\n\n".join(
            t["whisperTranscript"] for t in transcripts if "whisperTranscript" in t
        )

        result: dict = {}

        if "recall" in self.target_metrics:
            try:
                raw = await _cached_gpt_call(
                    _recall_prompt(mturk_statements, caption),
                    model=self.model,
                    cache_dir=self.cache_dir,
                    cache_only=self.cache_only,
                    recompute=self.recompute,
                )
                num_covered, num_statements = parse_recall_output(raw)
                recall_valid = num_statements > 0
                result["recall"] = num_covered / num_statements if recall_valid else 0.0
                result["recall_at_10"] = (
                    min(num_covered, 10) / min(num_statements, 10) if recall_valid else 0.0
                )
                result["num_statements"] = num_statements
                result["num_covered"] = num_covered
                result["recall_valid"] = recall_valid
            except Exception as exc:
                logger.warning(
                    "Recall scoring failed for %s: %s",
                    instance.metadata.get("url", "?"),
                    exc,
                )
                result.update(
                    recall=0.0,
                    recall_at_10=0.0,
                    num_statements=0,
                    num_covered=0,
                    recall_valid=False,
                )

        if "consistency" in self.target_metrics:
            try:
                statements_str = await _cached_gpt_call(
                    _canonical_prompt(caption),
                    model=self.model,
                    cache_dir=self.cache_dir,
                    cache_only=self.cache_only,
                    recompute=self.recompute,
                )
                cons_raw = await _cached_gpt_call(
                    _consistency_prompt(len(transcripts), transcripts_str, statements_str),
                    model=self.model,
                    cache_dir=self.cache_dir,
                    cache_only=self.cache_only,
                    recompute=self.recompute,
                )
                num_consistent, num_valid = parse_consistency_output(cons_raw)
                consistency_valid = num_valid > 0
                result["consistency"] = num_consistent / num_valid if consistency_valid else 0.0
                result["num_consistent"] = num_consistent
                result["consistency_valid"] = consistency_valid
            except Exception as exc:
                logger.warning(
                    "Consistency scoring failed for %s: %s",
                    instance.metadata.get("url", "?"),
                    exc,
                )
                result.update(consistency=0.0, num_consistent=0, consistency_valid=False)

        if output.metadata is None:
            output.metadata = {}
        output.metadata["dense_caption_result"] = result

        return result.get("recall", 0.0) if result.get("recall_valid", False) else 0.0
