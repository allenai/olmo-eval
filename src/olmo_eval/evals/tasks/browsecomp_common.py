"""Scoring shared by the BrowseComp family of tasks.

BrowseComp and BrowseComp-Plus pose the same kind of question, ask for the same
answer format, and are graded by the same judge prompt; they differ in where the
agent searches. The grader prompt and verdict parsing come from the pinned
BrowseComp-Plus reference, which reuses the original benchmark's grader.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.inference.retry import retry_with_backoff

logger = logging.getLogger(__name__)


def decrypt_with_password(value: str, password: str) -> str:
    """Decode an obfuscated benchmark string with the key derived from its password."""
    encrypted = base64.b64decode(value)
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    key = digest * (len(encrypted) // len(digest)) + digest[: len(encrypted) % len(digest)]
    return bytes(a ^ b for a, b in zip(encrypted, key, strict=True)).decode("utf-8")


@dataclass(frozen=True)
class BrowseCompScorer(Scorer):
    """Read a score produced by the benchmark's trajectory and judge evaluation."""

    name: str = "accuracy"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return float(output.metadata.get(f"score:{self.name}", 0.0))


@dataclass(frozen=True)
class BrowseCompMetric(Metric):
    """Aggregate the reference benchmark's per-question measurements."""

    def compute(self, responses: Sequence[Response]) -> float:
        eligible = responses
        if self.name in {"citation_precision", "citation_recall", "citations_per_response"}:
            eligible = [r for r in responses if r.scores.get("citation_coverage", 0.0)]
        return (
            sum(r.scores.get(self.name, 0.0) for r in eligible) / len(eligible) if eligible else 0.0
        )

    def supports_pairwise_scorer_fallback(self) -> bool:
        return self.name not in {"citation_precision", "citation_recall", "citations_per_response"}

    def compute_instance(self, response: Response) -> float | None:
        if not self.supports_pairwise_scorer_fallback() and not response.scores.get(
            "citation_coverage", 0.0
        ):
            return None
        return response.scores.get(self.name)

    def pairwise_display_format(self) -> str:
        return (
            "raw"
            if self.name.endswith("_calls")
            or self.name in {"citations_per_response", "tool_errors"}
            else "percentage"
        )


@dataclass(frozen=True)
class BrowseCompCalibration(Metric):
    """Reference calibration error, including the upstream binning convention."""

    name: str = "calibration_error_official"
    scorer: type[Scorer] | Scorer = BrowseCompScorer(name="calibration_error_official")

    def compute(self, responses: Sequence[Response]) -> float:
        from olmo_eval.evals.tasks.browsecomp_plus.reference import calculate_calibration_error

        verdicts = [r.outputs[0].metadata.get("judge_result", {}) for r in responses if r.outputs]
        verdicts = [
            v
            for v in verdicts
            if not v.get("parse_error", True)
            and v.get("correct") is not None
            and v.get("confidence") is not None
        ]
        if len(verdicts) < 100:
            logger.warning("BrowseComp calibration needs 100 valid confidences; returning 0")
            return 0.0
        return (
            calculate_calibration_error(
                [v["confidence"] for v in verdicts], [v["correct"] for v in verdicts]
            )
            / 100.0
        )

    def compute_instance(self, response: Response) -> None:
        return None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False

    def pairwise_higher_is_better(self) -> bool:
        return False

    def pairwise_display_format(self) -> str:
        return "percentage"


def trajectory_statistics(response: Response) -> tuple[set[str], dict[str, int]]:
    """Collect unique searched documents and tool-call counts from the trajectory."""
    trajectory = response.trajectory
    if trajectory is None:
        return set(), {}
    calls = {call.id: call.function.name for call in trajectory.tool_call_sequence}
    counts = dict(Counter(call.function.name for call in trajectory.tool_call_sequence))
    retrieved: set[str] = set()
    for result in trajectory.tool_result_sequence:
        if calls.get(result.tool_call_id) != "search" or result.is_error:
            continue
        try:
            hits = json.loads(result.content)
        except (TypeError, ValueError):
            continue
        if isinstance(hits, list):
            retrieved.update(
                str(hit["docid"]) for hit in hits if isinstance(hit, dict) and "docid" in hit
            )
    return retrieved, counts


def run_completed(response: Response, output: LMOutput | None) -> bool:
    """Whether the agent run ended with a model-authored final answer.

    Runs cut off at the turn limit, and runs the scaffold abandoned with a
    placeholder message and an empty trajectory, are not graded.
    """
    if output is None or not output.text.strip() or output.metadata.get("max_turns_reached"):
        return False
    if output.metadata.get("num_turns") == 0:
        return False
    return response.trajectory is None or response.trajectory.num_turns > 0


async def grade_with_judge(
    client: Any,
    *,
    question: str,
    response_text: str,
    correct_answer: str,
    model: str,
    max_output_tokens: int,
    query_id: str,
) -> tuple[dict[str, Any], str]:
    """Grade a final answer with the reference judge prompt, retrying transient API errors."""
    import openai

    from olmo_eval.evals.tasks.browsecomp_plus.reference import (
        GRADER_TEMPLATE,
        parse_judge_response,
    )

    prompt = GRADER_TEMPLATE.format(
        question=question, response=response_text, correct_answer=correct_answer
    )
    try:
        result = await retry_with_backoff(
            lambda: client.responses.create(
                model=model, input=prompt, max_output_tokens=max_output_tokens
            ),
            context="browsecomp judge",
            sdk_module=openai,
        )
    except Exception:
        logger.exception("BrowseComp judge failed for query %s", query_id)
        raise
    raw = result.output_text
    verdict = parse_judge_response(raw)
    if verdict["parse_error"]:
        logger.warning("BrowseComp judge verdict could not be parsed for query %s", query_id)
    return verdict, raw
