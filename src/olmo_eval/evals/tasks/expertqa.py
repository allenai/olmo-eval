"""ExpertQA: multi-domain expert-curated attributed long-form QA.

2,177 expert-written questions across 32 fields (Malaviya et al., NAACL 2024,
arXiv 2309.07852). The model generates a well-cited long-form answer, which is
graded for attribution and on-topic precision.

ExpertQA's human annotations grade the dataset's own pre-generated answers, so
they cannot grade a fresh model answer directly. We therefore score a model
under test on the axes the shared judges support without a reference:

- citation_precision / citation_recall: claim-level attribution, via the shared
  citation scorer (olmo_eval.common.scorers.citation).
- answer_precision: fraction of paragraphs the judge finds on-topic, via the
  irrelevant-paragraph judge reused from astabench_sqa.

The generation prompt, response parsing, precision judge, and the score-reading
metric classes are reused from astabench_sqa; ExpertQA drops the SQA rubric
(ingredient_recall), which has no per-question analogue here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.common.scorers.citation import (
    extract_json_from_response,
    score_citations_for_sections,
)
from olmo_eval.common.scorers.llm_judge import JudgeFn, build_default_judge_fn
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.astabench_sqa import (
    PRECISION_EVAL_PROMPT,
    SQA_GENERATION_PROMPT,
    AnswerPrecisionMetric,
    CitationPrecisionMetric,
    CitationRecallMetric,
    GlobalAvgMetric,
    compute_precision_score,
    format_report,
    normalize_agent_response_dict,
)
from olmo_eval.evals.tasks.common import Task, register

logger = logging.getLogger(__name__)

EXPERTQA_REPO = "cmalaviya/expertqa"

EXPERTQA_METRICS = (
    GlobalAvgMetric(),
    CitationPrecisionMetric(),
    CitationRecallMetric(),
    AnswerPrecisionMetric(),
)

# Metrics averaged into global_avg (ingredient_recall excluded, no rubric here).
EXPERTQA_METRIC_LABELS = ["citation_precision", "citation_recall", "answer_precision"]


@register("expertqa")
class ExpertQA(Task):
    """ExpertQA attributed long-form QA, graded for attribution and precision."""

    split = Split.TRAIN  # `main` config exposes a single train split
    data_source = DataSource(path=EXPERTQA_REPO, subset="main", split="train")
    metrics = EXPERTQA_METRICS
    primary_metric = GlobalAvgMetric()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc.get("question", "")
        if not question:
            return None

        metadata = doc.get("metadata") or {}
        return Instance(
            question=question,
            metadata={
                "case_id": f"expertqa_{index}",
                "field": metadata.get("field", ""),
                "specific_field": metadata.get("specific_field", ""),
                "question_type": metadata.get("question_type", ""),
                "index": index,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        prompt_text = SQA_GENERATION_PROMPT + instance.question
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": prompt_text},),
        )

    def extract_answer(self, output: LMOutput) -> Any:
        """Parse JSON from model output and store the structured response."""
        text = output.text
        # Strip <think>...</think> blocks so JSON extraction ignores reasoning braces.
        think_end = text.find("</think>")
        if think_end >= 0:
            text = text[think_end + len("</think>") :]
        parsed = extract_json_from_response(text)
        if parsed is not None:
            parsed = normalize_agent_response_dict(parsed)
        output.metadata["parsed_response"] = parsed
        return parsed

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Score responses for citation precision/recall and answer precision."""
        self._extract_answers(responses)

        judge_fn = build_default_judge_fn(scorer_name="ExpertQA")

        for response in responses:
            response.scores.update(await self._score_single(response, judge_fn))

        return responses

    async def _score_single(self, response: Response, judge_fn: JudgeFn) -> dict[str, float]:
        zeros = {k: 0.0 for k in EXPERTQA_METRIC_LABELS + ["global_avg"]}

        output = response.outputs[0] if response.outputs else None
        if output is None:
            return zeros

        parsed = output.metadata.get("parsed_response")
        if not parsed or not parsed.get("sections"):
            return zeros

        answer_text = format_report(parsed)
        precision_score = await self._score_precision(
            judge_fn, response.instance.question, answer_text
        )
        citation_scores = await score_citations_for_sections(judge_fn, parsed)

        scores = {
            "citation_precision": citation_scores.get("citation_precision", 0.0),
            "citation_recall": citation_scores.get("citation_recall", 0.0),
            "answer_precision": precision_score,
        }
        scores["global_avg"] = sum(scores[k] for k in EXPERTQA_METRIC_LABELS) / len(
            EXPERTQA_METRIC_LABELS
        )
        return scores

    async def _score_precision(self, judge_fn: JudgeFn, question: str, answer: str) -> float:
        """Answer precision via the irrelevant-paragraph judge (astabench precision_eval)."""
        prompt = PRECISION_EVAL_PROMPT.format(query=question, answer=answer)
        raw = await judge_fn(prompt)
        parsed = extract_json_from_response(raw)
        if not parsed:
            return 1.0  # No irrelevant paragraphs identified = perfect precision

        score, _ = compute_precision_score(parsed, answer)
        return score
