"""BrowseComp-Plus with reference retrieval and GPT-4.1 grading.

The search-only task and get_document variant use the corresponding upstream
prompts and the retriever exposed by the configured search server.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.types import Instance, LMRequest, Response
from olmo_eval.data import DataSource, SourceType
from olmo_eval.evals.tasks.browsecomp_common import (
    BrowseCompCalibration,
    BrowseCompMetric,
    BrowseCompScorer,
    grade_with_judge,
    run_completed,
    trajectory_statistics,
)
from olmo_eval.evals.tasks.browsecomp_plus.data import (
    DATASET_FILENAME,
    DATASET_REPO,
    artifact_home,
)
from olmo_eval.evals.tasks.browsecomp_plus.reference import (
    QUERY_TEMPLATE,
    QUERY_TEMPLATE_NO_GET_DOCUMENT,
    compute_citation_metrics,
    extract_citations_from_response,
)
from olmo_eval.evals.tasks.common import Task, register, register_variant

logger = logging.getLogger(__name__)


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
        "completed",
        "tool_errors",
    )
)


def tool_error_count(response: Response) -> int:
    """Count tool results that are not the JSON a retrieval endpoint returns.

    A tool that raises is reported to the model as plain text, so a retrieval
    outage shows up here instead of failing the run.
    """
    if response.trajectory is None:
        return 0
    errors = 0
    for result in response.trajectory.tool_result_sequence:
        try:
            json.loads(result.content)
        except (TypeError, ValueError):
            errors += 1
        else:
            errors += int(result.is_error)
    return errors


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
                completed = run_completed(response, output)
                scores = {metric.name: 0.0 for metric in MEAN_METRICS}
                scores.update(
                    retrieval_recall=len(retrieved.intersection(gold_docids)) / len(gold_docids),
                    search_calls=float(counts.get("search", 0)),
                    get_document_calls=float(counts.get("get_document", 0)),
                    completed=float(completed),
                    tool_errors=float(tool_error_count(response)),
                )
                if completed:
                    cited = extract_citations_from_response(text)
                    citation = compute_citation_metrics(cited, gold_docids)
                    scores.update(
                        citation_coverage=float(bool(cited)),
                        citation_precision=citation["precision"],
                        citation_recall=citation["recall"],
                        citations_per_response=float(len(cited)),
                    )
                # These measurements do not depend on the judge, so record them
                # before grading can fail.
                response.scores.update(scores)
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
