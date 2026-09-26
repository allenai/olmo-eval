"""Per-task counts of generations that did not end cleanly.

A score computed over truncated, empty or never-concluded generations reads the
same as one over complete answers. These counts sit beside every score so that
a budget or template defect is visible where the score is read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from olmo_eval.common.types import RequestType, Response

GENERATION_COUNT_KEYS = (
    "generations",
    "cap_hit",
    "empty",
    "unclosed_think",
    "finish_reason_unknown",
)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def count_generations(responses: Sequence[Response]) -> dict[str, int] | None:
    """Count how the generated outputs of a task ended.

    Call after thinking traces are stripped, so ``empty`` means no answer was
    left to score. Keys:

    - ``generations``: generated outputs counted (the denominator).
    - ``cap_hit``: the provider stopped at the token limit
      (``finish_reason == "length"``).
    - ``empty``: no non-whitespace text left to score.
    - ``unclosed_think``: the generation opened a ``<think>`` block and never
      closed it. A block opened by the chat template instead leaves no opening
      tag in the output; when such a trace is cut off it shows up as ``cap_hit``.
    - ``finish_reason_unknown``: the provider did not report why generation
      stopped, so ``cap_hit`` is a lower bound.

    Returns:
        The counts, or None when the task produced no generated outputs
        (e.g. loglikelihood tasks).
    """
    counts = dict.fromkeys(GENERATION_COUNT_KEYS, 0)
    for response in responses:
        if response.request.request_type == RequestType.LOGLIKELIHOOD:
            continue
        for output in response.outputs:
            metadata = output.metadata or {}
            text = output.text or ""
            raw = metadata.get("original_text", text)
            finish_reason = metadata.get("finish_reason")

            counts["generations"] += 1
            if finish_reason is None:
                counts["finish_reason_unknown"] += 1
            elif finish_reason == "length":
                counts["cap_hit"] += 1
            if not text.strip():
                counts["empty"] += 1
            if raw.rfind(_THINK_OPEN) > raw.rfind(_THINK_CLOSE):
                counts["unclosed_think"] += 1

    if counts["generations"] == 0:
        return None
    return counts


def sum_generation_counts(
    task_results: Mapping[str, Mapping[str, Any]], task_specs: Sequence[str]
) -> dict[str, int] | None:
    """Sum the counts of the given tasks, e.g. the members of a suite average.

    Returns:
        The summed counts, or None when none of the tasks carries counts.
    """
    total = dict.fromkeys(GENERATION_COUNT_KEYS, 0)
    found = False
    for spec in task_specs:
        counts: Mapping[str, int] | None = (task_results.get(spec) or {}).get("generation_counts")
        if not counts:
            continue
        found = True
        for key in GENERATION_COUNT_KEYS:
            total[key] += counts.get(key, 0)
    return total if found else None


def format_generation_counts(counts: Mapping[str, int] | None) -> str:
    """Render counts compactly for log lines and the summary table."""
    if not counts or not counts.get("generations"):
        return "-"
    total = counts["generations"]
    parts = [
        f"cap-hit {counts.get('cap_hit', 0)}/{total}",
        f"empty {counts.get('empty', 0)}",
        f"unclosed-think {counts.get('unclosed_think', 0)}",
    ]
    unknown = counts.get("finish_reason_unknown", 0)
    if unknown:
        parts.append(f"finish-reason unknown {unknown}")
    return ", ".join(parts)


__all__ = [
    "GENERATION_COUNT_KEYS",
    "count_generations",
    "format_generation_counts",
    "sum_generation_counts",
]
