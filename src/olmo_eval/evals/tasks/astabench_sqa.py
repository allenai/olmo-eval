"""AstaBench ScholArQA-CS2 scientific question answering task.

Evaluates models on generating well-cited research reports with inline citations.
Scoring uses 4 LLM-as-judge metrics (each 25% weight):
ingredient_recall, answer_precision, citation_precision, citation_recall.

Reference implementation: https://github.com/allenai/asta-bench
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from abc import ABC
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.llm_judge import JudgeFn, build_openai_judge_fn
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
from olmo_eval.evals.tasks.common import Task, register, register_variant

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

ASTA_BENCH_REPO = "allenai/asta-bench"
ASTA_BENCH_REVISION = "a600dc767f850385f4664772e3ba7a7f8be17d5e"

JUST_HAS_A_TITLE = (
    "Paper content unavailable. The paper's title is: "  # from astabench citation_eval.py
)

# =============================================================================
# Prompt Templates (from astabench evals/sqa/task.py, rubric.py, precision_eval.py, citation_eval.py)
# =============================================================================

# from astabench evals/sqa/task.py:json_to_sample
SQA_GENERATION_PROMPT = """Generate a report answering the following research question. Be sure to include inline citations for each claim. Return your result as valid JSON with a single key `sections` which is a list of sections, each having keys `title`, `text`, and `citations`. Each entry in `citations` should have a JSON list of `snippets` extracted from the reference document and an `id`, each of which appears exactly in the text. Each `id` should be an inline citation as it appears in the text (with wrapping parentheses or square brackets if appropriate). Each citation should have a `title` if one is available. Any additional information about the citation should go under `metadata`. Do not create a References section.

Here is an example `section` to help you with formatting:

        {
          "title": "Background",
          "text": "Convolutional neural networks (CNNs) have achieved state-of-the-art results in image classification [1][2].",
          "citations": [
            {
              "id": "[1]",
              "snippets": [
                "CNNs have become the standard for many visual tasks."
              ],
              "title": "ImageNet Classification with Deep Convolutional Neural Networks",
              "metadata": {
                "authors": "Krizhevsky, A. et al.",
                "year": 2012,
                "arxiv": "1207.0580"
              }
            },
            {
              "id": "[2]",
              "snippets": [
                "Significant improvements in image recognition have been observed with CNNs."
              ],
              "title": "Very Deep Convolutional Networks for Large-Scale Image Recognition",
              "metadata": {
                "authors": "Simonyan, K. & Zisserman, A.",
                "year": 2014,
                "arxiv": "1409.1556"
              }
            }
          ]
        }

Question: """

# from astabench rubric.py:_assess_properties_jointly
INGREDIENT_SYSTEM_PROMPT = """You will be given a question someone asked (in <question></question> tags) and the corresponding response (in <response></response> tags) given to them by an assistant.
        You will then be given an enumerated list of criteria by which to evaluate the response. Each criterion specifies requirements that the answer must satisfy. You will assign a score accordingly (see below).
        You will also be given a list of examples (in <examples></examples> tags, below each criterion) that illustrate the type of details that would satisfy the criterion. We do NOT expect any of the specified details to necessarily appear in the answer. These are strictly to be used as guidance for locating the answers that satisfy the set requirement.

        For each criterion, return a score of 0, 1 or 2 indicating how appropriate the response is based on the given criterion. 0 means the response does not meet the criterion, 1 means the response somewhat meets the criterion, 2 means the response perfectly meets the criterion. Judge only the specified aspect(s) delimited by the criterion, not any other qualities of the answer.

        Scoring Example 1:
        <question>Common medical NLP papers on clinical text benchmarks</question>
        <response>The application of natural language processing (NLP) and machine learning to medical text presents tremendous opportunities for healthcare tasks such as prediction ... [TRUNCATED]</response>
        Criteria:
        <criterion>
        1. Detail the well-known medical NLP datasets
        <examples>
        i2b2 includes datasets focused on temporal relations in clinical narratives, CRAFT Corpus is a collection of 97 full-length, open-access biomedical journal articles with semantic and syntactic annotations.]
        </examples>
        </criterion>
        <criterion>
        2. ... [TRUNCATED]
        <examples>
        ...[TRUNCATED]
        </examples>
        </criterion>

        A 2 point answer would fully satisfy the criterion #1. For example, it would include specific names with some details of well-known medical datasets for ML like those mentioned in the examples.
        A 1 point answer would only partially satisfy the criterion #1. For example, a dataset (like those in examples) may be mentioned, but no detail would be provided. Or datasets may be simply listed without further discussion.
        A 0 point answer would not mention datasets at all.

        Scoring Example 2:
        <question>What are some of the documentation methods used in Linguistics fieldwork.</question>
        <response>Language documentation, also called documentary linguistics, is a specialized subfield of linguistics ... [TRUNCATED]</response>
        Criteria:
        <criterion>
        1. ... [TRUNCATED]
        <examples>
        ...[TRUNCATED]
        </examples>
        </criterion>
        <criterion>
        2. Cover elicitation techniques for capturing specific linguistic data.
        <examples>
        structured interviews, elicitations based on standard word lists, prompted speech tasks
        </examples>
        </criterion>

        A 2 point answer to criterion #2 would contain common elicitation techniques like (but not limited to) those mentioned in the examples. The answer specifics don't have to match exactly with the examples, but examples show the types of instances that would count towards satisfying the criterion.
        A 1 point answer to criterion #2 be incomplete in some way. For example, the answer might mention "elicitation sessions" during a discussion on audio recording, but it fails to specifically address the requirement. Or the answer gives a list of standard word lists in the answer as resources, but fails to tie this information to elicitation.
        A 0 point answer to criterion #2 would simply not include the discussion in any way. For example, if an answer focuses only on data handling (post elicitation) techniques, it would miss out on techniques for documentation interview itself.

        Scoring Example 3:

        <question>How do transformer models differ from recurrent neural networks (RNNs)?</question>
        <response>Transformer models use self-attention mechanisms to process input, while RNNs process input sequentially. Transformers are better at handling long-range dependencies in data because they don't rely on previous time steps to pass information. RNNs may suffer from vanishing gradients and have trouble with long-term dependencies.</response>
        Criteria:
        <criterion>
        1. Must compare how the architecture and data processing flow differ between transformers and RNNs. <examples>
        Transformers use parallel processing and self-attention; RNNs process input tokens one at a time in sequence. Transformers can look at the entire input sequence at once, while RNNs have to pass information step by step.
        </examples>
        </criterion>

        A 2 point answer would accurately and distinctly contrast both architecture and sequence-processing style of both model families (e.g., parallelism vs. sequential processing, use of self-attention vs. recurrence).

        A 1 point answer would provide a partial or imprecise comparison, perhaps only mentioning one difference, or being vague (e.g., "Transformers work differently from RNNs in how they process text" without further elaboration).

        A 0 point answer would explain only one architecture (e.g., only transformers), or describe both but fail to contrast them on the asked criteria.

        Return your result as a JSON object with a single key `scores` whose value is a list of objects, each having keys `criteria_idx`, `reasoning`, `score` and `evidence` from the text supporting the claim."""

# from astabench precision_eval.py:PrecisionEval
PRECISION_EVAL_PROMPT = """You are given a query and a corresponding long answer.

Goal: find irrelevant paragraphs in the answer. These are paragraphs that don't directly answer the query and shouldn't be in the answer.

For instance, if the query is about datasets for scientific question answering, a paragraph about multilingual question answering datasets that don't contain scientific text would be considered irrelevant.

Explicitly consider whether something may be indirectly relevant. For example, if the question is about the conditions of horses in South Africa, a paragraph about general animal welfare in South Africa is potentially relevant while not being precisely about horses. On the other hand, a paragraph about pig welfare in South Africa is irrelevant.

Note that subtle differences can make the text irrelevant to the query. For instance, text about scientific survey paper generation is not relevant to a query about automatic paper review generation. Even though they seem related, they are about very different tasks.

Also, useful background in general is relevant. If the question is about an approach to creating liver-related proteins, some information about liver-related proteins could contextualize other parts of the answer. If a paragraph contextualizes another part of the answer, then it is relevant.

Go through the answer and output a list of irrelevant paragraphs. Every single paragraph needs to be considered, one by one. Our goal is to catch all the irrelevant paragraphs, so please be thorough.
Return your result as a JSON object with a single key `irrelevant_paragraphs` whose value is a list of objects, each having keys `reason`, and `answer_text` as follows:
{{"irrelevant_paragraphs":[
{{
"reason": "discuss why something is irrelevant (not indirectly relevant)",
"answer_text": "exact ENTIRE paragraph (not just a part of it) from the answer that is irrelevant"
}},
...
]
}}
Make sure all the irrelevant paragraphs are included.

Question: {query}
Answer: {answer}
"""

# from astabench citation_eval.py:CitationEval.score_citation_group
CITATION_GROUP_PROMPT = """You are a claim validator. For each claim made in the following text you will determine if it is supported by the quote from it's corresponding inline citations. As is typically done in academic writing, assume that consecutive sentences can share citations. Make sure to also include claims presented in table format. For references with only the title available (ie no quotes from the reference are included), judge them as `supporting` if the title indicates that the paper is likely relevant to the claim being considered. Return a JSON object with a single key `claims` which is a list of `claim` objects, one for each sentence in the text. Each `claim` object contains the claim itself (`text`), a list of `supporting` inline citations and `non_supporting` inline citations and finally a boolean `is_fully_supported` which indicates if the claim is entirely supported by the quotations in the associated citations. Each inline citation corresponding to that claim should appear in either `supporting` or `non_supporting`, but not both. Each claim made in the text should appear in your output, but you should skip sentences covering high level introductory information.

Text:
{}

References:
{}"""


# =============================================================================
# JSON / Response Parsing Helpers (from astabench evals/sqa/task.py)
# =============================================================================


def extract_json_from_response(response: str) -> dict[str, Any] | None:
    """Extract JSON object from a model response string."""
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    if json_start == -1 or json_end == 0:
        return None
    try:
        return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        try:
            return json.loads(response[json_start + 1 : json_end - 1])
        except json.JSONDecodeError:
            logger.warning("Could not decode JSON from response")
            return None


def normalize_agent_response_dict(response_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize response format to always have top-level `sections` key."""
    if not response_dict.get("sections") and response_dict.get("response"):
        return response_dict["response"]
    return response_dict


def format_report(response_dict: dict[str, Any]) -> str:
    """Extract plain text from structured response sections."""
    report = ""
    for section in response_dict.get("sections", []):
        text = section.get("text", "")
        if text:
            report += f"{text}\n\n"
        table = section.get("table")
        if table and isinstance(table, dict):
            table_text = table.get("text", "")
            if table_text:
                report += f"{table_text}\n\n"
    return report.strip()


# from astabench citation_eval.py:clean_sentence
def clean_sentence(sentence: str) -> str:
    """Clean XML tags from model-generated sentences."""
    pattern = r'<Paper [^>]*paperTitle="\W*([ _a-zA-Z0-9,.;]+)\W*"[^>]*/?>\s*</Paper>'
    sentence = re.sub(pattern, r"(\1)", sentence)
    pattern = r'<Model name="Anthropic" version="[^"]+">'
    return re.sub(pattern, "", sentence)


# =============================================================================
# Scoring Helpers (ported from astabench precision_eval.py, rubric.py, citation_eval.py)
# =============================================================================


# from astabench precision_eval.py:PrecisionEval.score
def compute_precision_score(
    parsed_json: dict[str, Any], answer: str
) -> tuple[float, dict[str, Any]]:
    """Compute answer precision from judge output identifying irrelevant paragraphs."""
    metadata: dict[str, Any] = {}
    irrelevant_texts = [
        item["answer_text"] for item in parsed_json.get("irrelevant_paragraphs", [])
    ]
    metadata["irrelevant_texts"] = irrelevant_texts

    paragraphs = re.split(r"\n\s*\n", answer.strip())
    metadata["answer_paragraphs"] = paragraphs

    if not paragraphs:
        return 1.0, metadata

    matching_paragraph_indices: set[int] = set()
    for sentence in irrelevant_texts:
        for idx, para in enumerate(paragraphs):
            if difflib.SequenceMatcher(None, sentence.strip(), para).ratio() >= 0.85:
                matching_paragraph_indices.add(idx)
                break

    score = 1 - (len(matching_paragraph_indices) / len(paragraphs))
    return score, metadata


# from astabench rubric.py:_assess_properties_jointly
def compute_ingredient_score(
    parsed_json: dict[str, Any],
    ingredients: list[dict[str, Any]],
) -> float:
    """Compute ingredient recall from joint assessment judge output."""
    has_criterion = [ing for ing in ingredients if ing.get("criterion")]
    if not has_criterion:
        return 0.0

    # Normalize weights so they sum to 1
    raw_weights = {ing["name"]: ing["weight"] for ing in has_criterion}
    total_weight = sum(raw_weights.values())
    if total_weight == 0:
        return 0.0
    weights = {k: v / total_weight for k, v in raw_weights.items()}

    scores_list = parsed_json.get("scores", [])
    score_components: dict[str, float] = {}
    for item in scores_list:
        idx = item.get("criteria_idx", 0) - 1  # 1-indexed in prompt
        if 0 <= idx < len(has_criterion):
            name = has_criterion[idx]["name"]
            score_components[name] = item.get("score", 0) / 2  # Normalize 0-2 to 0-1

    return sum(weights.get(k, 0) * score_components.get(k, 0) for k in weights)


# from astabench citation_eval.py
def _clean_citation_id(c: str) -> str:
    """Remove brackets/parens from citation ID for comparison."""
    for char in "[]()":
        c = c.replace(char, "")
    return c


# from astabench citation_eval.py
def _citation_intersection(supporting: list[str], half_credit_ids: list[str]) -> int:
    """Count supporting citations that are title-only (half credit)."""
    supporting_clean = {_clean_citation_id(str(c)) for c in supporting}
    half_clean = {_clean_citation_id(str(c)) for c in half_credit_ids}
    return len(supporting_clean.intersection(half_clean))


# from astabench citation_eval.py
def _filter_citation(citation: dict[str, Any], sec_text: str) -> bool:
    """Check if citation snippets are present and usable."""
    sec_text_alpha = re.sub(r"[^a-zA-Z]", "", sec_text).lower()
    raw_snippets = citation.get("snippets", [])
    if isinstance(raw_snippets, str):
        raw_snippets = [raw_snippets]
    snippets_alpha = [re.sub(r"[^a-zA-Z]", "", s).lower() for s in raw_snippets]
    return bool(
        citation.get("snippets")
        and not any(s in sec_text_alpha for s in snippets_alpha)
        and not (
            citation.get("title")
            and any(
                re.sub(r"[^a-zA-Z]", "", citation["title"]).lower() == s for s in snippets_alpha
            )
        )
    )


# from astabench citation_eval.py:CitationEval
def compute_citation_scores_from_groups(
    group_results: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate citation precision and recall from per-group scoring results."""
    n_attributable = 0
    n_extrapolatory = 0
    n_half_credit = 0
    precisions: list[float] = []

    for result in group_results:
        n_attributable += result["n_attributable"]
        n_extrapolatory += result["n_extrapolatory"]
        n_half_credit += result["n_half_credit_claims"]
        for s, e, p in zip(
            result["supporting_counts"],
            result["non_supporting_counts"],
            result["n_half_credit_citations"],
            strict=True,
        ):
            if s + e:
                precisions.append((s - 0.5 * p) / (s + e))

    total = n_attributable + n_extrapolatory
    recall = ((n_attributable - 0.5 * n_half_credit) / total) if total else 0.0
    precision = (sum(precisions) / len(precisions)) if precisions else 0.0

    return {
        "citation_recall": recall,
        "citation_precision": precision,
    }


# from astabench citation_eval.py:CitationEval.score_citation_group
def score_citation_group(
    judge_fn: JudgeFn,
    citation_group: str,
    citations: list[dict[str, str]],
) -> dict[str, Any]:
    """Score citations for a single section using the judge.

    Returns counts for aggregation by compute_citation_scores_from_groups.
    """
    if not citations:
        # Count sentences heuristically (split on period + space)
        n_sentences = len([s for s in re.split(r"(?<=[.!?])\s+", citation_group) if s.strip()])
        return {
            "n_attributable": 0,
            "n_extrapolatory": max(n_sentences, 1),
            "n_half_credit_claims": 0,
            "supporting_counts": [],
            "non_supporting_counts": [],
            "n_half_credit_citations": [],
        }

    prompt = CITATION_GROUP_PROMPT.format(
        citation_group,
        "\n\n".join(c["id"] + ": " + c["snippets"] for c in citations),
    )
    raw = judge_fn(prompt)
    parsed = extract_json_from_response(raw)
    if not parsed or "claims" not in parsed:
        n_sentences = len([s for s in re.split(r"(?<=[.!?])\s+", citation_group) if s.strip()])
        return {
            "n_attributable": 0,
            "n_extrapolatory": max(n_sentences, 1),
            "n_half_credit_claims": 0,
            "supporting_counts": [],
            "non_supporting_counts": [],
            "n_half_credit_citations": [],
        }

    claims = parsed["claims"]
    half_credit_ids = [c["id"] for c in citations if c["snippets"].startswith(JUST_HAS_A_TITLE)]

    n_attributable = 0
    n_extrapolatory = 0
    n_half_credit_claims = 0
    supporting_counts: list[int] = []
    non_supporting_counts: list[int] = []
    n_half_credit_citations: list[int] = []

    for claim in claims:
        supported = claim.get("is_fully_supported", False) and claim.get("supporting", [])
        n_attributable += 1 if supported else 0
        n_extrapolatory += 0 if supported else 1
        supporting_cits = claim.get("supporting", [])
        hc = _citation_intersection(supporting_cits, half_credit_ids)
        n_half_credit_citations.append(hc)
        supporting_counts.append(len(supporting_cits))
        n_half_credit_claims += (
            1 if supported and supporting_counts[-1] == n_half_credit_citations[-1] else 0
        )
        non_supporting_counts.append(len(claim.get("non_supporting", [])))

    return {
        "n_attributable": n_attributable,
        "n_extrapolatory": n_extrapolatory,
        "n_half_credit_claims": n_half_credit_claims,
        "supporting_counts": supporting_counts,
        "non_supporting_counts": non_supporting_counts,
        "n_half_credit_citations": n_half_credit_citations,
    }


# =============================================================================
# Scorer + Metrics
# =============================================================================


@dataclass(frozen=True)
class SQAJudgeScorer(Scorer):
    """Placeholder scorer for SQA metrics.

    Scoring is done in AstaBenchSQA.score_responses, not via this scorer.
    The score() method reads pre-computed scores from output metadata.
    """

    name: ClassVar[str] = "sqa_judge"
    score_key: str = "global_avg"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return (output.metadata or {}).get(self.score_key, 0.0)


class _SQAMetricBase(Metric, ABC):
    """Base for SQA metrics that read from response.scores."""

    scorer: ClassVar[type[Scorer]] = SQAJudgeScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        total = sum(r.scores.get(self.name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True)
class IngredientRecallMetric(_SQAMetricBase):
    name: str = "ingredient_recall"
    scorer: type[Scorer] = SQAJudgeScorer


@dataclass(frozen=True)
class AnswerPrecisionMetric(_SQAMetricBase):
    name: str = "answer_precision"
    scorer: type[Scorer] = SQAJudgeScorer


@dataclass(frozen=True)
class CitationPrecisionMetric(_SQAMetricBase):
    name: str = "citation_precision"
    scorer: type[Scorer] = SQAJudgeScorer


@dataclass(frozen=True)
class CitationRecallMetric(_SQAMetricBase):
    name: str = "citation_recall"
    scorer: type[Scorer] = SQAJudgeScorer


@dataclass(frozen=True)
class GlobalAvgMetric(_SQAMetricBase):
    name: str = "global_avg"
    scorer: type[Scorer] = SQAJudgeScorer


SQA_METRICS = (
    GlobalAvgMetric(),
    IngredientRecallMetric(),
    AnswerPrecisionMetric(),
    CitationPrecisionMetric(),
    CitationRecallMetric(),
)

SQA_METRIC_LABELS = [
    "ingredient_recall",
    "answer_precision",
    "citation_precision",
    "citation_recall",
]

# =============================================================================
# Task
# =============================================================================


@register("astabench_scholarqa")
class AstaBenchSQA(Task):
    """AstaBench ScholArQA-CS2 scientific question answering task."""

    split = Split.TRAIN  # HF JSON files load as a single "train" split
    data_source = DataSource(
        path=ASTA_BENCH_REPO,
        data_files="tasks/sqa/rubrics_v1_recomputed.json",
        revision=ASTA_BENCH_REVISION,
        split="train",  # JSON files load as a single "train" split in HF
    )
    metrics = SQA_METRICS
    primary_metric = GlobalAvgMetric()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc.get("question", "")
        if not question:
            return None

        # The HF dataset stores ingredients as a top-level list of dicts,
        # each with keys: criterion, examples, name, weight.
        ingredients = doc.get("ingredients", [])

        return Instance(
            question=question,
            metadata={
                "case_id": doc.get("case_id", f"sqa_{index}"),
                "annotator": doc.get("annotator", ""),
                "rubric_config": {"ingredients": ingredients},
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
        """Parse JSON from model output and store structured response."""
        text = output.text
        # Strip <think>...</think> blocks (e.g. Qwen3) so that JSON
        # extraction doesn't grab braces from the reasoning trace.
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
        """Score all responses using LLM-as-judge for 4 metrics.

        Overrides Task.score_responses because SQA needs multiple judge calls
        per instance producing 4+ scores (doesn't fit single-float Scorer.score).
        Scoring logic ported from astabench evals/sqa/task.py:SQATask.score.
        """
        self._extract_answers(responses)

        judge_fn = build_openai_judge_fn(
            model="gpt-4o-mini",
            scorer_name="AstaBenchSQA",
            max_tokens=4096,
            temperature=0.5,
        )

        for response in responses:
            scores = self._score_single(response, judge_fn)
            response.scores.update(scores)

        return responses

    def _score_single(self, response: Response, judge_fn: JudgeFn) -> dict[str, float]:
        """Score a single response across all 4 metrics + global_avg."""
        output = response.outputs[0] if response.outputs else None
        if output is None:
            return {k: 0.0 for k in SQA_METRIC_LABELS + ["global_avg"]}

        parsed = output.metadata.get("parsed_response")
        if not parsed or not parsed.get("sections"):
            return {k: 0.0 for k in SQA_METRIC_LABELS + ["global_avg"]}

        rubric_config = response.instance.metadata.get("rubric_config", {})
        question = response.instance.question
        answer_text = format_report(parsed)

        # 1. Ingredient recall
        ingredient_score = self._score_ingredients(judge_fn, question, answer_text, rubric_config)

        # 2. Answer precision
        precision_score = self._score_precision(judge_fn, question, answer_text)

        # 3. Citation precision & recall
        citation_scores = self._score_citations(judge_fn, parsed)

        scores = {
            "ingredient_recall": ingredient_score,
            "answer_precision": precision_score,
            "citation_precision": citation_scores.get("citation_precision", 0.0),
            "citation_recall": citation_scores.get("citation_recall", 0.0),
        }
        scores["global_avg"] = sum(scores[k] for k in SQA_METRIC_LABELS) / len(SQA_METRIC_LABELS)
        return scores

    def _score_ingredients(
        self,
        judge_fn: JudgeFn,
        question: str,
        answer: str,
        rubric_config: dict[str, Any],
    ) -> float:
        """Score ingredient recall. From astabench rubric.py:_assess_properties_jointly."""
        ingredients = rubric_config.get("ingredients", [])
        has_criterion = [ing for ing in ingredients if ing.get("criterion")]
        if not has_criterion:
            return 0.0

        # Build criteria text
        criteria_parts = []
        for i, ing in enumerate(has_criterion, 1):
            examples_text = "\n".join(ing.get("examples", []))
            criteria_parts.append(
                f"<criterion>\n{i}. {ing['criterion']}\n<examples>\n{examples_text}\n</examples>\n</criterion>"
            )
        criteria = "\n".join(criteria_parts)

        user_prompt = (
            f"<question>{question}</question>\n<response>{answer}</response>\nCriteria:\n{criteria}"
        )
        raw = judge_fn(user_prompt, system_prompt=INGREDIENT_SYSTEM_PROMPT)
        parsed = extract_json_from_response(raw)
        if not parsed:
            return 0.0

        return compute_ingredient_score(parsed, ingredients)

    def _score_precision(
        self,
        judge_fn: JudgeFn,
        question: str,
        answer: str,
    ) -> float:
        """Score answer precision. From astabench precision_eval.py:PrecisionEval."""
        prompt = PRECISION_EVAL_PROMPT.format(query=question, answer=answer)
        raw = judge_fn(prompt)
        parsed = extract_json_from_response(raw)
        if not parsed:
            return 1.0  # No irrelevant paragraphs identified = perfect precision

        score, _ = compute_precision_score(parsed, answer)
        return score

    def _score_citations(
        self,
        judge_fn: JudgeFn,
        parsed_response: dict[str, Any],
    ) -> dict[str, float]:
        """Score citation precision and recall. From astabench citation_eval.py:CitationEval."""
        bad_snippet = "Please click on the paper title to read the abstract on Semantic Scholar."
        group_results: list[dict[str, Any]] = []

        for section in parsed_response.get("sections", []):
            sec_iter = [section]
            if section.get("table") and isinstance(section["table"], dict):
                sec_iter.append(section["table"])

            for curr_sec in sec_iter:
                sec_text = curr_sec.get("text", "")
                raw_citations = curr_sec.get("citations", [])

                citations: list[dict[str, str]] = []
                for c in raw_citations:
                    cit_id = c.get("id")
                    if not cit_id:
                        continue
                    snippets = c.get("snippets", [])
                    if isinstance(snippets, list):
                        snippet_text = "... ".join(str(s) for s in snippets)
                    else:
                        snippet_text = str(snippets)

                    if _filter_citation(c, sec_text) and bad_snippet not in snippet_text:
                        citations.append({"id": cit_id, "snippets": snippet_text})
                    else:
                        title = c.get("title", "")
                        if title:
                            citations.append(
                                {
                                    "id": cit_id,
                                    "snippets": f"{JUST_HAS_A_TITLE}{title}",
                                }
                            )
                        else:
                            citations.append({"id": cit_id, "snippets": ""})

                clean_text = clean_sentence(sec_text)
                result = score_citation_group(judge_fn, clean_text, citations)
                group_results.append(result)

        if not group_results:
            return {"citation_precision": 0.0, "citation_recall": 0.0}

        return compute_citation_scores_from_groups(group_results)


# =============================================================================
# Variant: test split
# =============================================================================

register_variant(
    "astabench_scholarqa",
    "test",
    data_source=DataSource(
        path=ASTA_BENCH_REPO,
        data_files="tasks/sqa/rubrics_v2_recomputed.json",
        revision=ASTA_BENCH_REVISION,
        split="train",
    ),
)
