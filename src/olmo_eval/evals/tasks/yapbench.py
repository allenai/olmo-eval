"""
YapBench Evaluation Task

YapBench measures how much longer a chat model's answers are than they need to be. Each
prompt comes with a minimal baseline answer: a clarification request or a short
acknowledgment for vague, empty or conversational input (category A), a short fact for
closed-form questions (B), or a one-line command, snippet or yes/no for coding requests (C).

Paper: https://arxiv.org/abs/2601.00624
Data: https://huggingface.co/datasets/tabularisai/yapbench_dataset
Leaderboard: https://huggingface.co/spaces/tabularisai/YapBench

Scoring follows the paper. A response's YapScore is its length in characters beyond its
prompt's baseline answer, floored at zero. Each category's score is the median YapScore of
its prompts, and the YapIndex, the headline, is the mean of the three category medians.
Lower is better for every metric. The authors' scoring code is not published; where the
leaderboard Space's code and sample responses show what they measured, this port follows
them, and the rest are choices of this port:

- The paper counts characters without markdown but does not define the stripping. The
  ``yap_stripped`` scorer removes it with :func:`strip_markdown`, the authors' function
  from the leaderboard Space, which labels its lengths "markdown stripped"; the
  ``yap_raw`` scorer counts every character, as the lengths stored with the Space's sample
  responses do. Both report every metric, and the primary metric is the stripped YapIndex.
- Characters are Unicode code points, and whitespace in the answer counts, as in the
  Space. Baselines are counted as given.
- Only the final answer after a reasoning trace is measured, without the whitespace that
  separates it from the trace, as in the Space's sample answers from OLMo 3.1 Think, Qwen3
  and DeepSeek reasoning runs; some other providers return that whitespace, and the Space
  counts it there. A response whose trace never closed, or that hit the token limit with
  no closing ``</think>`` (a trace opened by the chat template), has no answer to measure
  and is left unscored. The Space's samples instead measure an answer cut at the token
  limit as it stands, and store an output that spent its budget on reasoning as an empty
  answer, which scores zero.
- A category with no scored responses reports 0.0 with a warning, and the YapIndex then
  averages the other categories.
- Bootstrap confidence intervals, the tail statistics and the YapTax cost estimate are not
  reported, and reasoning is not run both on and off as the paper's paired runs are.

The prompts are sent as the only user turn, with no system prompt, as in the paper, at
temperature 0 and with no token limit. The paper uses temperature 0 where the provider
allows it and the provider's default otherwise, and gives no token limit, but the Space's
sample responses stop at exactly 1,000 output tokens for several models and at 3,000 or
10,000 for a few others. Category A includes empty and whitespace-only prompts, which some
providers alter or reject; they are sent as the dataset gives them, although the paper
recommends a placeholder such as ``<EMPTY>`` for them. The pinned revision has no dataset
card; the card it deletes declared the MIT license.

Example command to run:

olmo-eval run -m allenai/Olmo-3-7B-Instruct -t yapbench
"""

import logging
import re
import statistics
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers import Scorer, ScoringIncompleteError
from olmo_eval.common.scorers.execution import ContextScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, Response, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import OutputScoreAggregation, Task, register
from olmo_eval.evals.tasks.common.judge import is_truncated
from olmo_eval.evals.tasks.omniscience import extract_final_answer

logger = logging.getLogger(__name__)

YAPBENCH_REPO = "tabularisai/yapbench_dataset"
YAPBENCH_REVISION = "be8427ddf7780201b73676c1563bc3ea6d0a71ca"

# Temperature 0 and otherwise the provider's defaults, as in the paper; no token limit,
# so that long answers are measured whole
DEFAULT_SAMPLING = SamplingParams(max_tokens=None, temperature=0.0)

CATEGORIES = ("A", "B", "C")

# =============================================================================
# Answer Length
# =============================================================================


def strip_markdown(text: str) -> str:
    """Remove markdown formatting from text, as the YapBench authors' leaderboard does.

    Copied from ``strip_markdown`` in ``src/side_by_side.py`` of the tabularisai/YapBench
    Space at 38546db374592b757b01cc1471afcddf5097a279 (Apache-2.0), with its steps kept in
    order: code fences, inline code, headings, bold, italics, strikethrough, links, images,
    blockquotes, horizontal rules, bullets and numbered-list markers. Whitespace is kept.
    """
    if not text:
        return ""
    text = re.sub(r"```[\s\S]*?```", lambda m: re.sub(r"```\w*\n?", "", m.group()), text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
    return text


def visible_answer(text: str) -> str | None:
    """Return the final answer after any reasoning trace, or None if a trace never closed.

    The whitespace between a closed trace and the answer is dropped; the answer's own
    whitespace is kept, since the authors count it.
    """
    answer = text.rpartition("</think>")[2].lstrip() if "</think>" in text else text
    return None if "<think>" in answer else answer


# =============================================================================
# Scoring and Metrics
# =============================================================================


@dataclass(frozen=True)
class YapScorer(ContextScorer):
    """Characters a response's final answer adds beyond its baseline answer, floored at zero.

    With ``strip``, markdown is removed from the answer before it is counted.
    """

    name: str = "yap_stripped"
    strip: bool = True

    async def ascore_with_context(
        self, instance: Instance, output: LMOutput, context: ScoringContext
    ) -> float:
        """Return the output's YapScore, raising when it has no final answer to measure.

        An output that hit the token limit with no ``</think>`` may be a reasoning trace
        that the chat template opened, so it has no answer either.
        """
        text = output.text or ""
        answer = visible_answer(text)
        if answer is None or (is_truncated(output) and "</think>" not in text):
            raise ScoringIncompleteError(
                "the response's reasoning trace never closed, so it has no final answer"
            )
        if self.strip:
            answer = strip_markdown(answer)
        return float(max(0, len(answer) - len(instance.metadata["baseline"])))


@dataclass(frozen=True, slots=True)
class _YapMetric(Metric):
    """A YapBench metric: lower is better, in characters."""

    scorer: type[Scorer] | Scorer = YapScorer

    def _yap_scores(self, responses: Sequence[Response], category: str) -> list[float]:
        """The YapScores of the responses in a category that were scored."""
        return [
            value
            for response in responses
            if response.instance.metadata.get("category") == category
            and (value := self._yap_score(response)) is not None
        ]

    def _yap_score(self, response: Response) -> float | None:
        value = response.scores.get(self.scorer().name)
        return float(value) if isinstance(value, (int, float)) else None

    def supports_scorer_fallback(self) -> bool:
        return False

    def higher_is_better(self) -> bool:
        return False

    def display_format(self) -> str:
        return "raw"


@dataclass(frozen=True, slots=True)
class CategoryMedianYapMetric(_YapMetric):
    """Median YapScore of the scored responses in one category; 0.0 if it has none.

    The name takes the form ``category__<category>__median_yap``.
    """

    name: str = "category__A__median_yap"

    def __post_init__(self) -> None:
        parts = self.name.split("__")
        if len(parts) != 3 or parts[0] != "category" or parts[1] not in CATEGORIES:
            raise ValueError(
                f"{type(self).__name__} name must take the form 'category__<A|B|C>__<metric>', "
                f"got {self.name!r}"
            )

    @property
    def category(self) -> str:
        return self.name.split("__")[1]

    def compute(self, responses: Sequence[Response]) -> float:
        scores = self._yap_scores(responses, self.category)
        if not scores:
            logger.warning(
                "%s has no scored responses for %s; its value of 0.0 is not a measured score.",
                self.name,
                self.scorer().name,
            )
            return 0.0
        return statistics.median(scores)

    def compute_instance(self, response: Response) -> float | None:
        if response.instance.metadata.get("category") != self.category:
            return None
        return self._yap_score(response)


@dataclass(frozen=True, slots=True)
class YapIndexMetric(_YapMetric):
    """The YapIndex: the mean over categories of their median YapScore.

    A category with no scored responses is left out, with a warning; with none at all the
    index is 0.0.
    """

    name: str = "yap_index"

    def compute(self, responses: Sequence[Response]) -> float:
        medians = []
        for category in CATEGORIES:
            scores = self._yap_scores(responses, category)
            if scores:
                medians.append(statistics.median(scores))
            else:
                logger.warning(
                    "%s for %s has no scored responses in category %s; it averages the "
                    "other categories.",
                    self.name,
                    self.scorer().name,
                    category,
                )
        return sum(medians) / len(medians) if medians else 0.0

    def compute_instance(self, response: Response) -> float | None:
        """A mean of medians has no per-instance equivalent."""
        return None


YAP_STRIPPED = YapScorer()
YAP_RAW = YapScorer(name="yap_raw", strip=False)

# =============================================================================
# Task Definition
# =============================================================================


@register("yapbench")
class YapBench(Task):
    """Prompts whose ideal answer is short, measured by how far responses run over."""

    data_source = DataSource(path=YAPBENCH_REPO, revision=YAPBENCH_REVISION, split="train")
    split = Split.TRAIN
    formatter = ChatFormatter()
    sampling_params = DEFAULT_SAMPLING
    answer_extractor = extract_final_answer
    output_score_aggregation = OutputScoreAggregation.MEAN
    metrics = tuple(
        metric
        for scorer in (YAP_STRIPPED, YAP_RAW)
        for metric in (
            YapIndexMetric(scorer=scorer),
            *(
                CategoryMedianYapMetric(name=f"category__{category}__median_yap", scorer=scorer)
                for category in CATEGORIES
            ),
        )
    )
    primary_metric = YapIndexMetric(scorer=YAP_STRIPPED)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance; the prompt is sent verbatim."""
        if doc["category"] not in CATEGORIES:
            raise ValueError(
                f"YapBench row {doc['id']} has an unknown category {doc['category']!r}"
            )
        return Instance(
            question=doc["prompt"],
            gold_answer=doc["baseline"],
            metadata={
                "id": doc["id"],
                "category": doc["category"],
                "baseline": doc["baseline"],
                "baseline_type": doc["baseline_type"],
                "domain": doc["domain"],
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Send the prompt as the only user turn, with no system prompt."""
        formatter = self.config.formatter or ChatFormatter()
        return formatter.format(instance, self.get_fewshot())
