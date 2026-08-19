"""DeepScholar query generation: does live search surface the right references?

Gives the model DeepScholar-Bench's own "write a Related Works section" prompt
(no separate system prompt, matching the `paper_search_agent` preset) and asks
it to search live via Semantic Scholar before writing. Scoring never looks at
the written section -- it only checks which of the paper's real arXiv
citations (extracted from its LaTeX \\cite commands by upstream DeepScholar-Bench,
see dataset/important_citations.csv) were ever surfaced by one of the model's
own search queries.

This isolates search-query quality from write-up quality: a fast, judge-free
proxy for the `reference_coverage` metric that DeepScholar-Bench's own judge
computes over the full generated section. Recomputing this proxy against
Yilun Zhao's existing gpt-5.6-sol/single_agent traces landed within 0.001 of
the real reference_coverage (0.266 vs 0.267 mean over 63 papers), so title
matching against the search-returned corpus is a validated stand-in for the
real (expensive, judge-based) metric.

Dataset: 63 papers from DeepScholar-Bench (arXiv 2508.20033), bundled locally
in data/deepscholar_queries_papers.jsonl (built from
https://github.com/guestrin-lab/deepscholar at c95413b3b2f3255b461b90d0ce650f685ae2d1ff,
the commit pinned by both this repo's deepscholar_bench external eval and by
Yilun Zhao's lit-agents runs).

Requirements: needs a tool-providing agentic harness exposing the
`semantic_scholar_snippet_search` tool (e.g. the `paper_search_agent` preset).
Run without tools, the trajectory is empty and every instance scores zero.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
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

PAPERS_PATH = Path(__file__).parent / "data" / "deepscholar_queries_papers.jsonl"

SEARCH_TOOL_NAME = "semantic_scholar_snippet_search"

# Verbatim from lit-agents' shared/deepscholar.py QUERY_TEMPLATE (which itself
# mirrors DeepScholar-Bench's own generation prompt), so this task's prompt is
# not a re-derivation but the one actually used to produce reference_coverage
# elsewhere in both this repo and Yilun Zhao's runs.
QUERY_TEMPLATE = (
    "Your task is to write a Related Works section for an academic paper given the "
    "paper's abstract. Your response should provide the Related Works section and "
    "references. Only include references from arXiv that are published before "
    "{cutoff_date}. Mention them in a separate, numbered reference list at the end "
    "and use the reference numbers to provide in-line citations in the Related Works "
    "section for all claims referring to a source (e.g., description of source [3]. "
    "Further details [6][7][8][9][10].) Each in-line citation must consist of a single "
    "reference number within a pair of brackets. Do not use any other citation format. "
    "Do not exceed 600 words for the related works section. Here is the paper "
    "abstract: {abstract}"
)

# The search tool bolds each result's title as "**Title**" (see
# harness/tools/search.py's semantic_scholar_search formatting).
_TITLE_RE = re.compile(r"\*\*(.+?)\*\*")


def normalize_title(s: str | None) -> str:
    """Normalize a paper title for exact-set matching, dropping punctuation/case."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


@dataclass(frozen=True)
class GoldReferenceCoverageScorer(Scorer):
    """Placeholder scorer; scores are computed in score_responses."""

    name: str = "gold_reference_coverage"
    score_key: str = "gold_reference_coverage"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return (output.metadata or {}).get(self.score_key, 0.0)


@dataclass(frozen=True)
class GoldReferenceCoverageMetric(Metric):
    """Mean fraction of a paper's arXiv gold references surfaced by any search query."""

    name: str = "gold_reference_coverage"
    scorer: type[Scorer] = GoldReferenceCoverageScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(r.scores.get(self.name, 0.0) for r in responses) / len(responses)


GOLD_REFERENCE_COVERAGE_METRIC = GoldReferenceCoverageMetric()


def extract_query_results(response: Response) -> list[dict[str, Any]]:
    """Pair each search-tool call with its query string and the titles it returned."""
    trajectory = response.trajectory
    if trajectory is None:
        return []

    results_by_id = {r.tool_call_id: r for r in trajectory.tool_result_sequence}
    out: list[dict[str, Any]] = []
    for call in trajectory.tool_calls_by_name(SEARCH_TOOL_NAME):
        try:
            args = json.loads(call.function.arguments)
        except (TypeError, ValueError):
            args = {}
        result = results_by_id.get(call.id)
        content = result.content if result else ""
        titles = [m.strip() for m in _TITLE_RE.findall(content or "")]
        out.append({"query": args.get("query", ""), "titles": titles})
    return out


@register("deepscholar_queries")
class DeepScholarQueries(Task):
    """Does live search surface the paper's real references? (query-quality proxy)."""

    data_source = DataSource(path=str(PAPERS_PATH))
    metrics = (GOLD_REFERENCE_COVERAGE_METRIC,)
    primary_metric = GOLD_REFERENCE_COVERAGE_METRIC
    sampling_params = SamplingParams(temperature=1.0, max_tokens=4096)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        abstract = doc.get("abstract", "")
        arxiv_id = doc.get("arxiv_id", "")
        if not abstract or not arxiv_id:
            return None

        return Instance(
            question=abstract,
            metadata={
                "case_id": f"deepscholar_queries_{index}",
                "arxiv_id": arxiv_id,
                "title": doc.get("title", ""),
                "cutoff_date": doc.get("cutoff_date", ""),
                "gold_arxiv_titles": doc.get("gold_arxiv_titles", []),
                "index": index,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        prompt = QUERY_TEMPLATE.format(
            cutoff_date=instance.metadata.get("cutoff_date", ""),
            abstract=instance.question,
        )
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": prompt},),
        )

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Score each paper by whether its arXiv gold references surfaced in search."""
        missing_trajectory = 0
        for response in responses:
            gold_titles = {
                normalize_title(t) for t in response.instance.metadata.get("gold_arxiv_titles", [])
            }
            if response.trajectory is None:
                missing_trajectory += 1

            query_results = extract_query_results(response)
            surfaced_titles = {normalize_title(t) for qr in query_results for t in qr["titles"]}
            surfaced_titles.discard("")

            hit = gold_titles & surfaced_titles
            coverage = (len(hit) / len(gold_titles)) if gold_titles else 0.0
            response.scores["gold_reference_coverage"] = coverage

            if response.outputs:
                meta = response.outputs[0].metadata
                meta["gold_reference_coverage"] = coverage
                meta["num_search_queries"] = len(query_results)
                meta["search_queries"] = [qr["query"] for qr in query_results]
                meta["num_gold_arxiv_references"] = len(gold_titles)
                meta["gold_arxiv_hit_titles"] = sorted(hit)

        if missing_trajectory:
            logger.warning(
                "deepscholar_queries scored %d/%d responses with no trajectory; this "
                "task needs an agentic harness exposing the %s tool, else every "
                "instance scores zero.",
                missing_trajectory,
                len(responses),
                SEARCH_TOOL_NAME,
            )
        return responses
