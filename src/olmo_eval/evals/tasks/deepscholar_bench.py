"""DeepScholar-Bench related-works generation (github.com/guestrin-lab/deepscholar).

63 recent arXiv papers, each supplying its abstract and its own publication
date. The model writes that paper's Related Works section and may cite only
arXiv papers published before that date, which is what makes this a live
literature-search task rather than a recall-from-memory one.

The generation prompt is VERBATIM the one lit-agents runs
(``integrated/shared/deepscholar.py::QUERY_TEMPLATE``). A single changed
character makes the two systems' outputs incomparable, so the template is
pinned by a golden-string test.

Scoring here is a placeholder. DeepScholar-Bench's published metrics (nugget
coverage, citation precision, reference coverage) are a second pass run outside
this repo: :mod:`olmo_eval.evals.tasks.deepscholar_export` converts saved
predictions into the per-query ``{intro.md, final_report.md, paper.csv}``
folders the upstream scorer reads. The metric computed here reports only
whether a response is exportable at all -- it has answer text and at least one
arXiv source in its trajectory -- so a run that produced nothing is visible
without this task pretending to measure quality.

Requirements: this task only produces signal under a tool-providing agentic
harness exposing ``arxiv_paper_search`` (the ``arxiv_paper_search_agent``
preset). The per-instance cutoff reaches that tool through
``Instance.metadata["retrieval_date_cutoff"]``, which the runner forwards into
the scaffold's ``search_date_cutoff`` block.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx

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
from olmo_eval.evals.tasks.common import Task, register
from olmo_eval.harness.tools.search import normalize_arxiv_id

logger = logging.getLogger(__name__)

DEEPSCHOLAR_REPO = "guestrin-lab/deepscholar"
# Pinned rather than tracking a branch: this file is the query set, so a silent
# upstream edit would change what the reported numbers mean.
DEEPSCHOLAR_COMMIT = "c95413b3b2f3255b461b90d0ce650f685ae2d1ff"
DEEPSCHOLAR_DATASET_PATH = "dataset/papers_with_related_works.csv"
DEEPSCHOLAR_DATASET_URL = (
    f"https://raw.githubusercontent.com/{DEEPSCHOLAR_REPO}/"
    f"{DEEPSCHOLAR_COMMIT}/{DEEPSCHOLAR_DATASET_PATH}"
)

SEARCH_TOOL_NAME = "arxiv_paper_search"

# VERBATIM from lit-agents' shared/deepscholar.py::QUERY_TEMPLATE, which is what
# the systems this task is compared against were prompted with.
DEEPSCHOLAR_QUERY_TEMPLATE = """Your task is to write a Related Works section for an academic paper given the paper's abstract. Your response should provide the Related Works section and references. Only include references from arXiv that are published before {cutoff_date}. Mention them in a separate, numbered reference list at the end and use the reference numbers to provide in-line citations in the Related Works section for all claims referring to a source (e.g., description of source [3]. Further details [6][7][8][9][10].) Each in-line citation must consist of a single reference number within a pair of brackets. Do not use any other citation format. Do not exceed 600 words for the related works section. Here is the paper abstract: {abstract}"""

# The blocks and labels arxiv_paper_search renders (see harness/tools/search.py).
_RESULT_SEPARATOR = "\n\n---\n\n"
_RESULT_FIELD_RE = re.compile(r"^(Authors|Year|Abstract|arXiv|URL):[ \t]*(.*)$")
_RESULT_TITLE_RE = re.compile(r"^\*\*(.+?)\*\*(?:\s*\[context only.*\])?$")


def _first_nonempty(row: Mapping[str, Any], *keys: str) -> str:
    """First key holding a non-blank value, stripped; "" when none does.

    Mirrors lit-agents' loader so an instance built here carries exactly the
    text lit-agents would interpolate into the same prompt.
    """
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _parse_iso_date(value: str) -> date | None:
    """Parse an ISO timestamp or date into a date, or None when unparseable."""
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def _dataset_cache_dir() -> Path:
    """Cache root, following the same env-var precedence as the RULER loader."""
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    if not cache_dir:
        hf_home = os.environ.get("HF_HOME") or os.path.join(
            os.environ.get("XDG_CACHE_HOME", "~/.cache"), "huggingface"
        )
        cache_dir = os.path.join(hf_home, "datasets")
    return Path(os.path.expanduser(cache_dir)) / "guestrin-lab--deepscholar"


def download_deepscholar_dataset() -> Path:
    """Fetch the pinned dataset CSV once, returning its cached path."""
    target = _dataset_cache_dir() / f"{DEEPSCHOLAR_COMMIT}-papers_with_related_works.csv"
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading DeepScholar-Bench dataset to %s", target)
    response = httpx.get(DEEPSCHOLAR_DATASET_URL, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    # Written via a temporary so a killed download cannot leave a truncated CSV
    # that later runs would treat as the pinned dataset.
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(response.content)
    os.replace(temporary, target)
    return target


def build_deepscholar_prompt(cutoff_date: str, abstract: str) -> str:
    """Fill the verbatim DeepScholar-Bench template for one paper."""
    return DEEPSCHOLAR_QUERY_TEMPLATE.format(cutoff_date=cutoff_date, abstract=abstract)


def parse_arxiv_tool_results(content: str) -> list[dict[str, str]]:
    """Parse ``arxiv_paper_search`` output back into its per-result fields.

    The rendered tool output is the only record of what the agent was shown, so
    the export path reads it rather than re-querying Semantic Scholar; the same
    query months later would not return the same page. Results marked context
    only carry no arXiv ID and are skipped, because nothing can cite them.
    """
    parsed: list[dict[str, str]] = []
    for block in (content or "").split(_RESULT_SEPARATOR):
        lines = block.strip().splitlines()
        if not lines:
            continue
        title_match = _RESULT_TITLE_RE.match(lines[0].strip())
        if title_match is None:
            continue

        fields: dict[str, str] = {"title": title_match.group(1).strip()}
        current: str | None = None
        for line in lines[1:]:
            field_match = _RESULT_FIELD_RE.match(line)
            if field_match is not None:
                current = field_match.group(1).lower()
                fields[current] = field_match.group(2)
            elif current is not None:
                # An abstract can wrap across lines; keep it whole.
                fields[current] = f"{fields[current]}\n{line}"

        arxiv_id = normalize_arxiv_id(fields.get("arxiv", ""))
        if not arxiv_id:
            continue
        parsed.append(
            {
                "arxiv_id": arxiv_id,
                "title": fields["title"],
                "authors": fields.get("authors", "").strip(),
                "year": fields.get("year", "").strip(),
                "abstract": fields.get("abstract", "").strip(),
                "url": fields.get("url", "").strip() or f"https://arxiv.org/abs/{arxiv_id}",
            }
        )
    return parsed


def sources_from_trajectory(trajectory: Any) -> list[dict[str, str]]:
    """Collect every arXiv source the search tool showed, first mention winning."""
    if trajectory is None:
        return []
    by_id: dict[str, dict[str, str]] = {}
    for result in trajectory.tool_result_sequence:
        for source in parse_arxiv_tool_results(result.content or ""):
            by_id.setdefault(source["arxiv_id"], source)
    return list(by_id.values())


@dataclass(frozen=True)
class DeepScholarBenchScorer(Scorer):
    """Placeholder scorer; DeepScholar-Bench scores are an external second pass."""

    name: str = "deepscholar_bench"
    score_key: str = "exportable_rate"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return (output.metadata or {}).get(self.score_key, 0.0)


class _DeepScholarMetricBase(Metric, ABC):
    """Base for metrics reading values precomputed in ``score_responses``."""

    scorer: type[Scorer] = DeepScholarBenchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(r.scores.get(self.name, 0.0) for r in responses) / len(responses)


@dataclass(frozen=True)
class ExportableRateMetric(_DeepScholarMetricBase):
    """Fraction of responses that can be scored externally at all.

    Not a quality measure: it is 1.0 for any response holding answer text and at
    least one retrieved arXiv source, which is exactly the condition
    ``deepscholar_export`` needs to write a query folder. Its job is to make a
    run that generated nothing impossible to mistake for a run that scored zero.
    """

    name: str = "exportable_rate"


@dataclass(frozen=True)
class ArxivCitationRateMetric(_DeepScholarMetricBase):
    """Fraction of responses whose answer cites at least one arxiv.org URL.

    The benchmark's citation parser credits arxiv.org URLs alone, so an answer
    without one scores zero downstream however good its prose is.
    """

    name: str = "arxiv_citation_rate"


EXPORTABLE_RATE_METRIC = ExportableRateMetric()
DEEPSCHOLAR_METRICS = (EXPORTABLE_RATE_METRIC, ArxivCitationRateMetric())

_ANSWER_ARXIV_URL_RE = re.compile(
    r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/([^)\s?#\]]+)",
    re.IGNORECASE,
)


@register("deepscholar_bench")
class DeepScholarBench(Task):
    """DeepScholar-Bench related-works generation over live arXiv search."""

    metrics = DEEPSCHOLAR_METRICS
    primary_metric = EXPORTABLE_RATE_METRIC
    # A related-works section plus its reference list is long-form output; this
    # matches deepresearch_bench rather than the short-answer tasks'. Whether it
    # takes effect depends on the harness threading sampling_params through.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16_384)

    @property
    def instances(self) -> Iterator[Instance]:
        """Load the pinned CSV, keeping upstream's positional query IDs."""
        if self._instances_cache is None:
            from olmo_eval.data import DataLoader, DataSource

            loader = DataLoader()
            source = DataSource(path=str(download_deepscholar_dataset()))
            self._instances_cache = []
            for index, doc in enumerate(loader.load(source)):
                instance = self.process_doc(doc, index)
                if instance is not None:
                    self._instances_cache.append(instance)
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        abstract = _first_nonempty(doc, "abstract")
        cutoff = _parse_iso_date(_first_nonempty(doc, "published_date", "cutoff_date"))
        if not abstract or cutoff is None:
            return None

        return Instance(
            question=abstract,
            metadata={
                # Upstream identifies a query by its row position, and the export
                # folders are named for it, so the ID has to stay positional.
                "id": str(index),
                "case_id": f"deepscholar_{index}",
                "index": index,
                "title": _first_nonempty(doc, "title") or f"DeepScholar query {index}",
                "arxiv_id": normalize_arxiv_id(_first_nonempty(doc, "arxiv_id")),
                "cutoff_date": cutoff.isoformat(),
                # Read by the runner and applied to the search tools as this
                # instance's date cutoff.
                "retrieval_date_cutoff": cutoff.isoformat(),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=(
                {
                    "role": "user",
                    "content": build_deepscholar_prompt(
                        cutoff_date=instance.metadata["cutoff_date"],
                        abstract=instance.question,
                    ),
                },
            ),
        )

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Record export readiness and stash the sources the export pass needs."""
        missing_trajectory = 0
        for response in responses:
            if response.trajectory is None:
                missing_trajectory += 1
            sources = sources_from_trajectory(response.trajectory)
            answer = response.outputs[0].text if response.outputs else ""
            cited = {
                normalize_arxiv_id(match.group(1))
                for match in _ANSWER_ARXIV_URL_RE.finditer(answer or "")
            }

            response.scores.update(
                {
                    "exportable_rate": 1.0 if (answer.strip() and sources) else 0.0,
                    "arxiv_citation_rate": 1.0 if cited else 0.0,
                }
            )

            if response.outputs:
                meta = response.outputs[0].metadata
                meta["deepscholar_source_arxiv_ids"] = [s["arxiv_id"] for s in sources]
                meta["deepscholar_cited_arxiv_ids"] = sorted(cited)
                meta["deepscholar_num_searches"] = (
                    len(response.trajectory.tool_calls_by_name(SEARCH_TOOL_NAME))
                    if response.trajectory is not None
                    else 0
                )

        if missing_trajectory:
            logger.warning(
                "DeepScholar-Bench scored %d/%d responses with no trajectory; this task needs "
                "an agentic harness exposing the %s tool, else nothing is exportable.",
                missing_trajectory,
                len(responses),
                SEARCH_TOOL_NAME,
            )
        return responses
