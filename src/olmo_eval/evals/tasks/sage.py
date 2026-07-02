"""SAGE short-form scientific paper retrieval task.

SAGE asks a model to identify a target paper from a reasoning-intensive query.
This task scores the model's final output, not the agent trajectory.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register

SAGE_REPO = "yilunzhao/sage-retrieval"  # HF repo set (OT-1)

SAGE_SHORT_FORM_PROMPT = (
    "You are helping a researcher identify a scientific paper from a detailed query. "
    "Use any available search tools to find the paper that matches the query. "
    "After searching, give your single best answer: state the most likely paper's "
    "title, or explicitly say no match was found.\n\n"
    "Query: "
)

SAGE_OPEN_ENDED_PROMPT = (
    "You are helping a researcher answer a scientific literature question. "
    "Find the relevant papers that support the answer. "
    "In your final answer, list the titles of the most relevant papers you found, "
    "up to about 10, ordered from most to least relevant.\n\n"
    "Question: "
)


class RequiredGoldPaper(TypedDict):
    paperId: str
    title: str
    abstract: str


class GoldPaper(RequiredGoldPaper, total=False):
    arxiv_id: str
    doi: str
    corpus_id: str


def make_gold(
    paper_id: str,
    title: str,
    abstract: str = "",
    *,
    arxiv_id: str = "",
    doi: str = "",
    corpus_id: str = "",
) -> GoldPaper:
    """Build a gold paper record with optional external IDs defaulted."""
    return {
        "paperId": paper_id,
        "title": title,
        "abstract": abstract,
        "arxiv_id": arxiv_id,
        "doi": doi,
        "corpus_id": corpus_id,
    }


class Matcher(Protocol):
    """Async predicate for whether an output identifies a gold paper."""

    @property
    def name(self) -> str:
        """Stable matcher name, including config where relevant."""
        ...

    async def matched(self, gold: GoldPaper, output: str) -> bool:
        """Return whether the output identifies the gold paper."""
        ...


def normalize_title(s: str) -> str:
    """Normalize a paper title for substring matching.

    Article-stripping (a/an/the) is intentionally not done here. This matches the
    task packet and remains pending confirmation of SAGE's original EM
    normalization (open thread OT-2).
    """
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def strip_think(text: str) -> str:
    """Drop balanced and truncated think regions, preserving visible text."""
    open_tag = "<think>"
    close_tag = "</think>"
    output: list[str] = []
    index = 0
    depth = 0

    while index < len(text):
        next_open = text.find(open_tag, index)
        next_close = text.find(close_tag, index)

        if depth == 0:
            if next_open == -1 and next_close == -1:
                output.append(text[index:])
                break
            if next_open != -1 and (next_close == -1 or next_open < next_close):
                output.append(text[index:next_open])
                index = next_open + len(open_tag)
                depth = 1
            else:
                end = next_close + len(close_tag)
                output.append(text[index:end])
                index = end
            continue

        if next_open == -1 and next_close == -1:
            break
        if next_open != -1 and (next_close == -1 or next_open < next_close):
            depth += 1
            index = next_open + len(open_tag)
        else:
            depth -= 1
            index = next_close + len(close_tag)

    return "".join(output)


@dataclass(frozen=True, slots=True)
class NormalizedStringMatcher:
    """SAGE's normalized title substring baseline."""

    name: str = "normalized_string"

    async def matched(self, gold: GoldPaper, output: str) -> bool:
        gold_title = normalize_title(gold["title"])
        if not gold_title:
            return False
        return gold_title in normalize_title(output)


async def exact_match(matcher: Matcher, gold: GoldPaper, output: str) -> float:
    """Return 1.0 iff the matcher finds the gold paper in the output."""
    return 1.0 if await matcher.matched(gold, strip_think(output)) else 0.0


async def weighted_recall(
    matcher: Matcher, golds: list[tuple[GoldPaper, int]], output: str
) -> float:
    """Compute relevance-weighted recall over SAGE gold papers."""
    total = sum(relevance for _, relevance in golds)
    if total == 0:
        return 0.0

    stripped = strip_think(output)
    matched_weight = 0
    for gold, relevance in golds:
        if await matcher.matched(gold, stripped):
            matched_weight += relevance
    return matched_weight / total


@dataclass(frozen=True)
class SageExactMatchScorer(Scorer):
    """Placeholder scorer; SAGE scores are computed in score_responses."""

    name: str = "exact_match"
    score_key: str = "exact_match"

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get(self.score_key, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass(frozen=True)
class SageExactMatchMetric(Metric):
    """Mean exact-match over precomputed response scores."""

    name: str = "exact_match"
    scorer: type[Scorer] = SageExactMatchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(r.scores.get(self.name, 0.0) for r in responses) / len(responses)


@dataclass(frozen=True)
class SageWeightedRecallScorer(Scorer):
    """Placeholder scorer; SAGE weighted recall is computed in score_responses."""

    name: str = "weighted_recall"
    score_key: str = "weighted_recall"

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get(self.score_key, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass(frozen=True)
class SageWeightedRecallMetric(Metric):
    """Mean weighted recall over precomputed response scores."""

    name: str = "weighted_recall"
    scorer: type[Scorer] = SageWeightedRecallScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(r.scores.get(self.name, 0.0) for r in responses) / len(responses)


class _SageRetrieval(Task):
    """Shared SAGE retrieval task behavior."""

    # Deterministic normalized-title substring matching is the sole SAGE matcher.
    matcher: Matcher = NormalizedStringMatcher()
    prompt: str = ""

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached("train")

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=(
                {
                    "role": "user",
                    "content": self.prompt + instance.question,
                },
            ),
        )


@register("sage_short_form")
class SageShortForm(_SageRetrieval):
    """SAGE short-form paper identification."""

    data_source = DataSource(path=SAGE_REPO, subset="short_form", split="train")
    prompt = SAGE_SHORT_FORM_PROMPT
    metrics = (SageExactMatchMetric(),)
    primary_metric = SageExactMatchMetric()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        query = str(doc.get("complete_query") or "").strip()
        ground_truth = doc.get("ground_truth") or {}
        if not isinstance(ground_truth, dict):
            return None

        title = str(ground_truth.get("title") or "").strip()
        if not query or not title:
            return None

        corpus_id = ground_truth.get("corpus_id", ground_truth.get("corpusId", ""))
        gold: GoldPaper = make_gold(
            paper_id=str(ground_truth.get("paperId") or doc.get("paper_id") or ""),
            title=title,
            abstract=str(ground_truth.get("abstract") or ""),
            arxiv_id=str(ground_truth.get("arxiv_id") or ground_truth.get("arxivId") or ""),
            doi=str(ground_truth.get("doi") or ground_truth.get("DOI") or ""),
            corpus_id=str(corpus_id or ""),
        )

        return Instance(
            question=query,
            metadata={
                "gold": gold,
                "case_id": (
                    doc.get("case_id")
                    or doc.get("paper_id")
                    or ground_truth.get("paperId")
                    or f"sage_short_form_{index}"
                ),
                "domain": doc.get("domain", ""),
                "index": index,
            },
        )

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Score each response by matching the final output against the gold paper."""
        for response in responses:
            output = response.outputs[0] if response.outputs else None
            output_text = output.text if output is not None else ""
            em = await exact_match(self.matcher, response.instance.metadata["gold"], output_text)
            response.scores["exact_match"] = em

            if output is not None:
                output.metadata = output.metadata or {}
                output.metadata["sage_matched"] = bool(em)
                output.metadata["exact_match"] = em

        return responses


@register("sage_open_ended")
class SageOpenEnded(_SageRetrieval):
    """SAGE open-ended paper retrieval scored by relevance-weighted recall."""

    data_source = DataSource(path=SAGE_REPO, subset="open_ended", split="train")
    prompt = SAGE_OPEN_ENDED_PROMPT
    metrics = (SageWeightedRecallMetric(),)
    primary_metric = SageWeightedRecallMetric()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = str(doc.get("question") or "").strip()
        ground_truth = doc.get("ground_truth") or {}
        if not question or not isinstance(ground_truth, dict):
            return None

        golds: list[tuple[GoldPaper, int]] = []
        for key, relevance in (("most_relevant", 2), ("relevant", 1)):
            papers = ground_truth.get(key) or []
            if not isinstance(papers, list):
                continue
            for paper in papers:
                if not isinstance(paper, dict):
                    continue
                title = str(paper.get("title") or "").strip()
                if not title:
                    continue
                corpus_id = paper.get("corpus_id", paper.get("corpusId", ""))
                golds.append(
                    (
                        make_gold(
                            paper_id=str(paper.get("paperId") or paper.get("paper_id") or ""),
                            title=title,
                            abstract=str(paper.get("abstract") or ""),
                            arxiv_id=str(paper.get("arxiv_id") or paper.get("arxivId") or ""),
                            doi=str(paper.get("doi") or paper.get("DOI") or ""),
                            corpus_id=str(corpus_id or ""),
                        ),
                        relevance,
                    )
                )

        if not golds:
            return None

        return Instance(
            question=question,
            metadata={
                "golds": golds,
                "case_id": doc.get("case_id") or f"sage_open_ended_{index}",
                "domain": doc.get("domain", ""),
                "index": index,
            },
        )

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Score each response by relevance-weighted recall over gold papers."""
        for response in responses:
            output = response.outputs[0] if response.outputs else None
            output_text = output.text if output is not None else ""
            wr = await weighted_recall(
                self.matcher, response.instance.metadata["golds"], output_text
            )
            response.scores["weighted_recall"] = wr

            if output is not None:
                output.metadata = output.metadata or {}
                output.metadata["weighted_recall"] = wr

        return responses
