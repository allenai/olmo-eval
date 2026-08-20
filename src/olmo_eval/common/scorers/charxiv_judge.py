"""GPT-4o judge for CharXiv, mirroring the official graders exactly.

Faithful to ``princeton-nlp/CharXiv`` ``src/descriptive_utils.py::get_descriptive_result_gpt``
and ``src/reasoning_utils.py::get_reasoning_result_gpt``: model ``gpt-4o-2024-05-13``,
``response_format={"type": "json_object"}``, ``n=1, temperature=0, top_p=1, seed=42``,
``max_tokens`` starting at 256 and doubling (cap 1024) when the JSON reply is truncated
("Unterminated string"), at most 10 retries, and the official dummy fallback (score ``-1``)
when grading ultimately fails. Downstream stats count ``-1`` as 0, as in the official
``get_stats.py``.

Successful (validated) grading replies are cached on disk so re-runs are free; the cache dir
comes from ``CHARXIV_GPT_CACHE_DIR`` or a fresh per-process temp dir (never a shared cache).
Requires ``OPENAI_API_KEY`` on cache misses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olmo_eval.common.image_qa.charxiv import build_dummy_output, verify_grading_output
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput

logger = logging.getLogger(__name__)

CHARXIV_JUDGE_MODEL = "gpt-4o-2024-05-13"

_ASYNC_CLIENTS: dict[str, Any] = {}
_PROCESS_CACHE_DIR: list[str] = []


def default_charxiv_cache_dir() -> str:
    """Judge-response cache dir: env override or a fresh process-local temp dir."""
    env_dir = os.environ.get("CHARXIV_GPT_CACHE_DIR")
    if env_dir:
        return env_dir
    if not _PROCESS_CACHE_DIR:
        _PROCESS_CACHE_DIR.append(tempfile.mkdtemp(prefix="charxiv-gpt-cache-"))
    return _PROCESS_CACHE_DIR[0]


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()


def _get_client(model: str):
    if model not in _ASYNC_CLIENTS:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for CharXiv grading on a cache miss.")
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai") from None
        _ASYNC_CLIENTS[model] = AsyncOpenAI(api_key=api_key)
    return _ASYNC_CLIENTS[model]


async def _charxiv_chat_json(
    prompt: str,
    *,
    validate: Callable[[dict], None],
    dummy: dict,
    cache_dir: str,
    cache_only: bool = False,
    recompute: bool = False,
    max_retries: int = 10,
) -> dict:
    """One official grading call: cached, JSON-mode, with the official retry ladder."""
    key = _cache_key(CHARXIV_JUDGE_MODEL, prompt)
    cache_file = Path(cache_dir) / f"{key}-v1.json"
    if not recompute and cache_file.exists():
        with open(cache_file) as f:
            content = json.load(f)
        return content
    if cache_only:
        raise ValueError(f"Cache miss (cache_only=True) for key {key[:16]}…")

    client = _get_client(CHARXIV_JUDGE_MODEL)
    curr_retries = 0
    max_tokens = 256
    content: dict | None = None
    while curr_retries < max_retries:
        try:
            response = (
                (
                    await client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=CHARXIV_JUDGE_MODEL,
                        response_format={"type": "json_object"},
                        n=1,
                        max_tokens=max_tokens,
                        temperature=0,
                        top_p=1,
                        seed=42,
                    )
                )
                .choices[0]
                .message.content
            )
            content = json.loads(response)
            validate(content)
            break
        except Exception as e:  # official: retry on any error; grow tokens on truncation
            logger.warning("CharXiv grading error: %s", e)
            content = None
            if "Unterminated string starting at" in str(e):
                if max_tokens >= 1024:
                    logger.warning("CharXiv grading failed (truncated at max tokens)")
                    return dummy
                max_tokens = min(1024, max_tokens * 2)
            else:
                curr_retries += 1
    if content is None:
        logger.warning("CharXiv grading failed after %d retries", max_retries)
        return dummy

    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(".tmp", prefix=f"{key}-v1.json", text=True, dir=cache_dir)
    os.close(fd)
    with open(tmp, "w") as f:
        json.dump(content, f)
    os.rename(tmp, str(cache_file))
    return content


async def grade_descriptive_batch(
    grading_query: str,
    length: int,
    *,
    cache_dir: str,
    cache_only: bool = False,
    recompute: bool = False,
) -> dict:
    """Grade one batched descriptive query (official ``get_descriptive_result_gpt``)."""

    def _validate(content: dict) -> None:
        verify_grading_output(content, length)

    return await _charxiv_chat_json(
        grading_query,
        validate=_validate,
        dummy=build_dummy_output(length),
        cache_dir=cache_dir,
        cache_only=cache_only,
        recompute=recompute,
    )


async def grade_reasoning(
    grading_query: str,
    *,
    cache_dir: str,
    cache_only: bool = False,
    recompute: bool = False,
) -> tuple[Any, int]:
    """Grade one reasoning query (official ``get_reasoning_result_gpt``)."""

    def _validate(content: dict) -> None:
        # official accesses these keys directly; a KeyError triggers a retry
        content["extracted_answer"]
        content["score"]

    content = await _charxiv_chat_json(
        grading_query,
        validate=_validate,
        dummy={"extracted_answer": "Failed to parse response", "score": -1},
        cache_dir=cache_dir,
        cache_only=cache_only,
        recompute=recompute,
    )
    return content["extracted_answer"], content["score"]


@dataclass(frozen=True)
class CharxivJudgeScorer(Scorer):
    """Score channel for the CharXiv GPT judge.

    Actual grading happens task-level (the official protocol batches descriptive triplets
    across instances), which stores per-output ``score:charxiv``; this scorer exposes that
    stored value so the metric/scorer plumbing stays uniform.
    """

    name: str = "charxiv"
    cache_dir: str = field(default_factory=default_charxiv_cache_dir)
    cache_only: bool = False
    recompute: bool = False

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get("score:charxiv", 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0
