"""Opt-in answer-region policy for audited text correctness metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from olmo_eval.common.metrics import (
    AccuracyMetric,
    F1Metric,
    PassAtKMetric,
    PassPowKMetric,
    SQuADF1Metric,
)
from olmo_eval.common.types import RequestType, Response
from olmo_eval.common.types.base import TURNS_KEY
from olmo_eval.evals.tasks.common import base as task_base

if TYPE_CHECKING:
    from olmo_eval.common.execution import ScoringContext


GATE_KEY = "final_answer_gate"


def validate_final_answer_only(task: task_base.Task, *, force: bool = False) -> None:
    """Reject combinations whose zero or answer-region semantics have not been audited."""
    if not (force or task.config.reward_final_answer_only):
        return
    prefix = f"{task.config.name}: reward_final_answer_only is unsupported"
    if task.request_type not in (RequestType.CHAT, RequestType.COMPLETION):
        raise ValueError(f"{prefix} for {task.request_type} requests")
    for method in (
        "score_responses",
        "compute_metrics",
        "_get_scorers",
        "_apply_scorers",
        "_apply_scorers_async",
        "_aggregate_output_scores",
        "_expand_multi_output_responses",
    ):
        if getattr(type(task), method) is not getattr(task_base.Task, method):
            raise ValueError(f"{prefix} with a custom {method} implementation")
    if not task.config.metrics:
        raise ValueError(f"{prefix} without correctness metrics")
    for metric in task.config.metrics:
        if type(metric) not in (
            AccuracyMetric,
            F1Metric,
            SQuADF1Metric,
            PassAtKMetric,
            PassPowKMetric,
        ):
            raise ValueError(f"{prefix} for metric {type(metric).__name__}")
        scorer = metric.scorer()
        # Subclasses must explicitly re-audit their own behavior; an inherited
        # capability could silently authorize a refusal or inverted scorer.
        if not type(scorer).__dict__.get("supports_final_answer_only", False):
            raise ValueError(f"{prefix} for scorer {type(scorer).__name__}")
        if scorer.requires_async or scorer.reads_logprobs:
            raise ValueError(f"{prefix} for non-text/synchronous scorer {type(scorer).__name__}")


def prepare_final_answers(task: task_base.Task, responses: Sequence[Response]) -> None:
    """Select the final region from the original response, preserving whitespace."""
    validate_final_answer_only(task)
    for response in responses:
        if response.request.request_type not in (RequestType.CHAT, RequestType.COMPLETION):
            raise ValueError("reward_final_answer_only requires text generation responses")
        if not response.outputs:
            raise ValueError(
                "reward_final_answer_only requires an output, even for an empty answer"
            )
        if (
            response.trajectory is not None
            or response.request.tools
            or any(o.has_tool_calls or TURNS_KEY in o.metadata for o in response.outputs)
        ):
            raise ValueError("reward_final_answer_only does not support agent/tool responses")
        for output in response.outputs:
            raw = output.metadata.setdefault("original_text", output.text)
            if not isinstance(raw, str):
                raise ValueError("reward_final_answer_only requires original response text")
            _, delimiter, final = raw.rpartition("</think>")
            reason = (
                "missing_delimiter"
                if not delimiter
                else "empty_final_answer"
                if not final.strip()
                else "final_answer"
            )
            output.text = final if delimiter else ""
            output.extracted_answer = None
            output.metadata[GATE_KEY] = {
                "applied": True,
                "reason": reason,
                "zeroed": reason != "final_answer",
            }


def score_final_answers(task: task_base.Task, responses: Sequence[Response]) -> Sequence[Response]:
    """Score eligible answers and retain missing answers as explicit zero observations."""
    prepare_final_answers(task, responses)
    for response in responses:
        response.scores.clear()
        response.scoring_error = None
        for output in response.outputs:
            for key in list(output.metadata):
                if key.startswith("score:") or key in ("scoring_errors", "answer_format_correct"):
                    del output.metadata[key]
        eligible = [o for o in response.outputs if not o.metadata[GATE_KEY]["zeroed"]]
        if eligible:
            task._extract_answers([replace(response, outputs=eligible, scores={})])
        for scorer in task._get_scorers().values():
            scores = {}
            for index, output in enumerate(response.outputs):
                score = (
                    0.0
                    if output.metadata[GATE_KEY]["zeroed"]
                    else scorer.score(response.instance, output)
                )
                output.metadata[f"score:{scorer.name}"] = score
                scores[index] = score
            response.scores[scorer.name] = task._aggregate_output_scores(scores)
    return responses


async def score_with_answer_region(
    task: task_base.Task, responses: Sequence[Response], context: ScoringContext | None = None
) -> Sequence[Response]:
    """Runner entry point that validates policy before dispatching task overrides."""
    validate_final_answer_only(task)
    if not task.config.reward_final_answer_only:
        task.strip_thinking_traces(responses)
    if context is None:
        return await task.score_responses(responses)
    return await task.score_responses(responses, context=context)
