"""LitSearch: agentic literature-search retrieval.

597 literature-search queries over ML/NLP papers, each with gold Semantic
Scholar corpus IDs (Ajith et al., EMNLP 2024, arXiv 2407.18940).

This is an AGENTIC adaptation, not the published Recall@5/@20. The model is
given the `semantic_scholar_snippet_search` tool and may search the live
Semantic Scholar API; the task succeeds for a query if any gold paper's corpus
ID surfaces in the tool results across the trajectory. This measures the live S2
API plus the model's agent loop, not retrieval over LitSearch's fixed corpus, so
the numbers are NOT comparable to published LitSearch results.

Requirements: this task only produces signal when run through a tool-providing
agentic harness (scaffold that executes tool calls, with the
`semantic_scholar_snippet_search` tool available, e.g. the `dr_tulu` preset).
Run without tools, the trajectory is empty and every query scores zero.

Scoring reads `response.trajectory`: corpus IDs are parsed from the tool results
(the search tool emits a `Corpus ID:` line per result) and intersected with the
query's gold corpus IDs.
"""

from __future__ import annotations

import logging
import re
from abc import ABC
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

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

logger = logging.getLogger(__name__)

LITSEARCH_REPO = "princeton-nlp/LitSearch"
SEARCH_TOOL_NAME = "semantic_scholar_snippet_search"

# The search tool emits "Corpus ID: <int>" per result (see harness/tools/search.py).
_CORPUS_ID_RE = re.compile(r"Corpus ID:\s*(\d+)")

LITSEARCH_PROMPT = (
    "You are helping a researcher find relevant scientific papers. Use the "
    f"{SEARCH_TOOL_NAME} tool to search for papers that answer the question below. "
    "Issue as many searches as needed, refining your queries, then list the titles "
    "of the most relevant papers you found.\n\nQuestion: "
)

LITSEARCH_METRIC_LABELS = ["found_rate", "gold_recall"]


@dataclass(frozen=True)
class LitSearchScorer(Scorer):
    """Placeholder scorer; LitSearch scores are computed in score_responses."""

    name: str = "litsearch"
    score_key: str = "found_rate"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return (output.metadata or {}).get(self.score_key, 0.0)


class _LitSearchMetricBase(Metric, ABC):
    """Base for LitSearch metrics that read precomputed values from response.scores."""

    scorer: type[Scorer] = LitSearchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(r.scores.get(self.name, 0.0) for r in responses) / len(responses)


@dataclass(frozen=True)
class FoundRateMetric(_LitSearchMetricBase):
    """Fraction of queries where at least one gold paper surfaced in search results."""

    name: str = "found_rate"


@dataclass(frozen=True)
class GoldRecallMetric(_LitSearchMetricBase):
    """Mean fraction of a query's gold papers that surfaced (for multi-gold queries)."""

    name: str = "gold_recall"


LITSEARCH_METRICS = (FoundRateMetric(), GoldRecallMetric())


def extract_corpus_ids(response: Response) -> set[int]:
    """Collect every corpus ID returned by search tools across the trajectory."""
    trajectory = response.trajectory
    if trajectory is None:
        return set()
    seen: set[int] = set()
    for result in trajectory.tool_result_sequence:
        for match in _CORPUS_ID_RE.finditer(result.content or ""):
            seen.add(int(match.group(1)))
    return seen


def score_litsearch(gold_ids: set[int], seen_ids: set[int]) -> dict[str, float]:
    """Found-rate (any gold surfaced) and recall (fraction of gold surfaced)."""
    if not gold_ids:
        return {"found_rate": 0.0, "gold_recall": 0.0}
    hits = gold_ids & seen_ids
    return {
        "found_rate": 1.0 if hits else 0.0,
        "gold_recall": len(hits) / len(gold_ids),
    }


@register("litsearch")
class LitSearch(Task):
    """LitSearch agentic literature-search retrieval over Semantic Scholar."""

    data_source = DataSource(path=LITSEARCH_REPO, subset="query", split="full")
    metrics = LITSEARCH_METRICS
    primary_metric = FoundRateMetric()
    # Advisory only: the openai_agents scaffold this task runs under does not yet
    # thread sampling_params into the Agents SDK model settings, so the agent uses
    # server defaults. Scoring depends only on which corpus IDs surface, not on
    # the generated text, so this does not affect the metric.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached("full")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        query = doc.get("query", "")
        gold_ids = [int(c) for c in doc.get("corpusids", []) if c is not None]
        if not query or not gold_ids:
            return None

        return Instance(
            question=query,
            metadata={
                "case_id": f"litsearch_{index}",
                "gold_corpusids": gold_ids,
                "query_set": doc.get("query_set", ""),
                "specificity": doc.get("specificity"),
                "quality": doc.get("quality"),
                "index": index,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": LITSEARCH_PROMPT + instance.question},),
        )

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Score each query by whether its gold corpus IDs surfaced in search results."""
        missing_trajectory = 0
        for response in responses:
            gold_ids = set(response.instance.metadata.get("gold_corpusids", []))
            seen_ids = extract_corpus_ids(response)
            if response.trajectory is None:
                missing_trajectory += 1

            scores = score_litsearch(gold_ids, seen_ids)
            response.scores.update(scores)

            if response.outputs:
                meta = response.outputs[0].metadata
                meta["litsearch_seen_corpusids"] = sorted(seen_ids)
                meta["litsearch_num_searches"] = (
                    len(response.trajectory.tool_calls_by_name(SEARCH_TOOL_NAME))
                    if response.trajectory is not None
                    else 0
                )

        if missing_trajectory:
            logger.warning(
                "LitSearch scored %d/%d responses with no trajectory; this task needs an "
                "agentic harness exposing the %s tool, else every query scores zero.",
                missing_trajectory,
                len(responses),
                SEARCH_TOOL_NAME,
            )
        return responses
