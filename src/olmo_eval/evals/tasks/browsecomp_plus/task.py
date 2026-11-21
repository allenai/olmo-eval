"""BrowseComp-Plus with reference retrieval and GPT-4.1 grading.

The search-only task and get_document variant use the corresponding upstream
prompts and the retriever exposed by the configured search server.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, Response
from olmo_eval.data import DataSource, SourceType
from olmo_eval.evals.tasks.browsecomp_plus.data import (
    DATASET_FILENAME,
    DATASET_REPO,
    artifact_home,
)
from olmo_eval.evals.tasks.browsecomp_plus.reference import (
    GRADER_TEMPLATE,
    QUERY_TEMPLATE,
    QUERY_TEMPLATE_NO_GET_DOCUMENT,
    calculate_calibration_error,
    compute_citation_metrics,
    extract_citations_from_response,
    parse_judge_response,
)
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.inference.retry import retry_with_backoff

logger = logging.getLogger(__name__)


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
            if self.name.endswith("_calls") or self.name == "citations_per_response"
            else "percentage"
        )


@dataclass(frozen=True)
class BrowseCompCalibration(Metric):
    """Reference calibration error, including the upstream binning convention."""

    name: str = "calibration_error_official"
    scorer: type[Scorer] | Scorer = BrowseCompScorer(name="calibration_error_official")

    def compute(self, responses: Sequence[Response]) -> float:
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


MEAN_METRICS = tuple(
    BrowseCompMetric(name=name, scorer=BrowseCompScorer(name=name))
    for name in (
        "accuracy",
        "retrieval_recall",
        "search_calls",
        "get_document_calls",
        "citation_coverage",
        "citation_precision",
        "citation_recall",
        "citations_per_response",
    )
)


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


def run_completed(output: LMOutput | None) -> bool:
    """Whether the agent run ended with a final answer and was not cut off at the turn limit."""
    return (
        output is not None
        and bool(output.text.strip())
        and not output.metadata.get("max_turns_reached")
    )


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


@register("browsecomp_plus")
class BrowseCompPlus(Task):
    """Fixed-corpus research questions scored by the official GPT-4.1 evaluator."""

    data_source = DataSource(path=DATASET_REPO, split="test")
    formatter = ChatFormatter(
        user_template=QUERY_TEMPLATE_NO_GET_DOCUMENT.replace("{Question}", "{question}")
    )
    metrics = (*MEAN_METRICS, BrowseCompCalibration())
    primary_metric = MEAN_METRICS[0]
    judge_model = "gpt-4.1"
    judge_max_tokens = 1024
    strip_thinking = True
    required_secrets = ("OPENAI_API_KEY",)
    dependencies = ["httpx~=0.28.1"]

    def _load_instances(self, split: str | None = None) -> Iterator[Instance]:
        """Load the prepared local copy of the dataset the task is configured with."""
        from olmo_eval.data import DataLoader

        source = self.config.get_data_source(split=split)
        if source.path == DATASET_REPO:
            prepared = (artifact_home() / DATASET_FILENAME).resolve()
            if not prepared.is_file():
                raise FileNotFoundError(
                    "BrowseComp-Plus data is not prepared. Run "
                    "uv run python -m olmo_eval.evals.tasks.browsecomp_plus.prepare "
                    "and set BROWSECOMP_PLUS_HOME to its output directory."
                )
            source = replace(source, path=str(prepared), source_type=SourceType.LOCAL)
        for index, doc in enumerate(DataLoader().load(source)):
            yield self.process_doc(doc, index)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        query_id = str(doc["query_id"])
        if not doc.get("query") or not doc.get("answer") or not doc.get("evidence_docs"):
            raise ValueError(
                f"Incomplete BrowseComp-Plus record {query_id}; use the preparation script"
            )
        return Instance(
            question=doc["query"],
            gold_answer=doc["answer"],
            metadata={
                "id": query_id,
                "query_id": query_id,
                "evidence_docids": [str(evidence["docid"]) for evidence in doc["evidence_docs"]],
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is None:
            raise ValueError("BrowseComp-Plus requires an answering prompt")
        return self.config.formatter.format(
            replace(instance, question=instance.question.replace("\t", " ").strip())
        )

    async def score_responses(
        self, responses: Sequence[Response], context: Any = None
    ) -> Sequence[Response]:
        """Grade final responses and measure retrieval across their full trajectories."""
        import openai

        if self.config.sampling_params and self.config.sampling_params.num_samples != 1:
            raise ValueError("BrowseComp-Plus evaluates one agent trajectory per question")
        async with openai.AsyncOpenAI() as client:
            for response in responses:
                if len(response.outputs) > 1:
                    raise ValueError("BrowseComp-Plus evaluates one agent trajectory per question")
                retrieved, counts = trajectory_statistics(response)
                gold_docids = response.instance.metadata["evidence_docids"]
                output = response.outputs[0] if response.outputs else None
                text = output.text if output is not None else ""
                completed = run_completed(output)
                scores = {metric.name: 0.0 for metric in MEAN_METRICS}
                scores.update(
                    retrieval_recall=len(retrieved.intersection(gold_docids)) / len(gold_docids),
                    search_calls=float(counts.get("search", 0)),
                    get_document_calls=float(counts.get("get_document", 0)),
                )
                verdict: dict[str, Any] = {
                    "parse_error": True,
                    "error": "Response incomplete or cannot be parsed",
                }
                raw = None
                if completed:
                    verdict, raw = await grade_with_judge(
                        client,
                        question=response.instance.question,
                        response_text=text,
                        correct_answer=response.instance.gold_answer or "",
                        model=self.config.judge_model or "gpt-4.1",
                        max_output_tokens=self.config.judge_max_tokens or 1024,
                        query_id=response.instance.metadata["id"],
                    )
                    scores["accuracy"] = float(verdict["correct"] is True)
                    cited = extract_citations_from_response(text)
                    citation = compute_citation_metrics(cited, gold_docids)
                    scores.update(
                        citation_coverage=float(bool(cited)),
                        citation_precision=citation["precision"],
                        citation_recall=citation["recall"],
                        citations_per_response=float(len(cited)),
                    )
                response.scores.update(scores)
                if output is not None:
                    output.metadata["judge_result"] = {
                        **verdict,
                        "raw_response": raw,
                        "judge_model": self.config.judge_model,
                        "is_completed": completed,
                        "retrieved_docids": sorted(retrieved),
                        "tool_call_counts": counts,
                    }
                    for name, score in scores.items():
                        output.metadata[f"score:{name}"] = score
        return responses


register_variant(
    "browsecomp_plus",
    "get_document",
    formatter=ChatFormatter(user_template=QUERY_TEMPLATE.replace("{Question}", "{question}")),
)
register_variant("browsecomp_plus", "mini", limit=10)
