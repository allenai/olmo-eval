"""DeepScholar-Bench related-works generation (github.com/guestrin-lab/deepscholar).

63 recent arXiv papers, each supplying its abstract and its own publication
date. The model writes that paper's Related Works section and may cite only
arXiv papers published before that date, which is what makes this a live
literature-search task rather than a recall-from-memory one.

The generation prompt is VERBATIM the one lit-agents runs
(``integrated/shared/deepscholar.py::QUERY_TEMPLATE``). A single changed
character makes the two systems' outputs incomparable, so the template is
pinned by a golden-string test.

Scoring here is a placeholder. DeepScholar-Bench's published metrics run as a
second pass outside this repo, on the per-query folders
:mod:`olmo_eval.evals.tasks.deepscholar_export` writes. What this task computes
is whether that second pass will find anything at all: the prompt mandates
numbered citations while the upstream parser only credits markdown links to
arxiv.org/abs, so :mod:`olmo_eval.evals.tasks.deepscholar_citations` has to
bridge the two, and ``exportable_rate`` runs that same bridge and counts the
answers it succeeds on. A run whose citations never resolve is therefore
visible here rather than only after external scoring returns zeros.

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
from olmo_eval.evals.tasks.deepscholar_citations import (
    reference_list,
    resolve_numbering,
    rewrite_intro,
    split_final_report,
    strip_references,
    unresolved_citation_forms,
)
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
# The search tool marks a date it could only resolve to a month.
_MONTH_PRECISION_SUFFIX = " (month precision)"
# Fields the search tool renders before the abstract. Matched only while still
# in a block's header: once the abstract starts, every remaining line is text.
_RESULT_HEADER_FIELD_RE = re.compile(r"^(Authors|Year|Published|arXiv|URL):[ \t]*(.*)$")
_ABSTRACT_FIELD_PREFIX = "Abstract:"
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


def _parse_published_field(value: str) -> tuple[str, str]:
    """Split a rendered ``Published`` line into an ISO date and its precision.

    Returns ``("", "")`` for anything unparseable, which sends the caller to the
    arXiv-ID fallback rather than writing a date the contract would reject.
    """
    text = value.strip()
    if not text:
        return "", ""
    if text.endswith(_MONTH_PRECISION_SUFFIX):
        month = text[: -len(_MONTH_PRECISION_SUFFIX)].strip()
        parsed = _parse_iso_date(f"{month}-01")
        return ("", "") if parsed is None else (parsed.isoformat(), "month")
    parsed = _parse_iso_date(text)
    return ("", "") if parsed is None else (parsed.isoformat(), "day")


def parse_arxiv_tool_results(content: str) -> list[dict[str, str]]:
    """Parse ``arxiv_paper_search`` output back into its per-result fields.

    A block's identity comes from its header -- the lines between the title and
    the abstract -- and from nowhere else. The abstract is the one field a paper
    controls the text of, so once it starts, every remaining line in the block is
    read as text even when it is shaped exactly like a field. Without that, a
    paper whose abstract contains "arXiv: 1234.56789" on its own line would
    rename itself, and a context-only block quoting one would invent a citable
    source out of a result the tool explicitly refused to make citable.

    A block with no arXiv ID in its header is not a citable result: that covers
    the context-only blocks and any prose the tool emitted around them.
    """
    parsed: list[dict[str, str]] = []
    for block in (content or "").split(_RESULT_SEPARATOR):
        lines = block.strip().splitlines()
        if not lines:
            continue

        title_match = _RESULT_TITLE_RE.match(lines[0].strip())
        # A title the renderer mangled still leaves a readable header, and the
        # export re-fetches titles anyway, so this does not reject the block.
        fields: dict[str, str] = {
            "title": title_match.group(1).strip() if title_match else lines[0].strip("* ").strip()
        }
        abstract_lines: list[str] = []
        in_abstract = False
        for line in lines[1:]:
            if in_abstract:
                abstract_lines.append(line)
                continue
            if line.startswith(_ABSTRACT_FIELD_PREFIX):
                in_abstract = True
                abstract_lines.append(line[len(_ABSTRACT_FIELD_PREFIX) :].strip())
                continue
            field_match = _RESULT_HEADER_FIELD_RE.match(line)
            if field_match is not None:
                fields[field_match.group(1).lower()] = field_match.group(2)

        arxiv_id = normalize_arxiv_id(fields.get("arxiv", "").strip())
        if not arxiv_id:
            continue
        published_date, date_precision = _parse_published_field(fields.get("published", ""))
        parsed.append(
            {
                "arxiv_id": arxiv_id,
                "title": fields["title"],
                "authors": fields.get("authors", "").strip(),
                "year": fields.get("year", "").strip(),
                "published_date": published_date,
                "date_precision": date_precision,
                "abstract": "\n".join(abstract_lines).strip(),
                "url": fields.get("url", "").strip() or f"https://arxiv.org/abs/{arxiv_id}",
            }
        )
    return parsed


def sources_from_trajectory(trajectory: Any) -> list[dict[str, str]]:
    """Collect every arXiv source the search tool showed, first mention winning.

    Only results of ``SEARCH_TOOL_NAME`` calls are read. Another tool's output
    can mention an arXiv ID without the agent ever having been shown that paper
    as a citable result, and crediting it would let a citation resolve against a
    source the search never returned. Within those results, identity comes from
    each block's header; see :func:`parse_arxiv_tool_results`.
    """
    if trajectory is None:
        return []
    search_call_ids = {
        call.id for call in trajectory.tool_call_sequence if call.function.name == SEARCH_TOOL_NAME
    }
    by_id: dict[str, dict[str, str]] = {}
    for result in trajectory.tool_result_sequence:
        if result.tool_call_id not in search_call_ids:
            continue
        for source in parse_arxiv_tool_results(result.content or ""):
            by_id.setdefault(source["arxiv_id"], source)
    return list(by_id.values())


def score_answer(answer: str, sources: Sequence[Mapping[str, str]]) -> dict[str, float]:
    """Run the citation bridge over one answer and report what it achieved.

    ``exportable_rate`` is whether the rewritten intro holds at least one
    resolvable arxiv.org citation, which is exactly the condition under which
    the upstream parser returns any documents for this query.
    ``citation_resolution_rate`` is the share of the reference list the answer
    published that resolved to a retrieved source -- the diagnostic that
    separates "the model cited nothing" from "the model cited papers it never
    retrieved" from "the bridge failed".

    ``marker_compliance_rate`` is whether the answer delivered its report under
    the marker line the preset's system prompt mandates. Bookkeeping, not
    quality: an answer without the marker is scored on its full text exactly as
    before, and this only records how often that happened, because the same
    prompt run on a different backbone can comply at a very different rate.

    ``unresolved_citation_forms`` counts citation shapes no pass can turn into a
    link -- superscripts, footnote markers, author-year brackets. An answer that
    cited carefully in a style this bridge does not read scores zero on the other
    two, and without this it would look identical to one that never cited.
    """
    # Everything below the marker, or the whole answer when there is none --
    # the same split the export applies, so a number reported during the run
    # and a number reported after export describe the same text.
    report, marker_found = split_final_report(answer)
    _, cited = rewrite_intro(report, list(sources))
    published_refs = reference_list(report)
    resolved = resolve_numbering(report, list(sources))
    return {
        "exportable_rate": 1.0 if cited else 0.0,
        "citation_resolution_rate": (
            len(resolved) / len(published_refs) if published_refs else 0.0
        ),
        "unresolved_citation_forms": float(unresolved_citation_forms(strip_references(report))),
        "marker_compliance_rate": 1.0 if marker_found else 0.0,
    }


@dataclass(frozen=True)
class DeepScholarBenchScorer(Scorer):
    """Interface stub; both metrics read values ``score_responses`` precomputes.

    ``Task.score_responses`` is overridden here, so the scorer pipeline never
    runs and this method is never called. It reads the same key
    ``score_responses`` writes so that if it ever is called it cannot report a
    silent zero.
    """

    name: str = "deepscholar_bench"
    score_key: str = "score:exportable_rate"

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
    """Fraction of answers the external scoring pass will find citations in.

    Not a quality measure. It is 1.0 when the citation bridge turns the answer
    into an intro holding at least one resolvable arxiv.org link, which is the
    condition for the upstream parser to return any documents at all. Its job is
    to make a run whose citations never resolve impossible to mistake for a run
    that was scored and did badly.
    """

    name: str = "exportable_rate"


@dataclass(frozen=True)
class CitationResolutionRateMetric(_DeepScholarMetricBase):
    """Share of each answer's published reference list that named a retrieved paper."""

    name: str = "citation_resolution_rate"


@dataclass(frozen=True)
class UnresolvedCitationFormsMetric(_DeepScholarMetricBase):
    """Mean count per answer of citation shapes the bridge cannot read.

    A count, not a rate. Non-zero means answers are citing in a style -- footnote
    markers, superscripts, author-year -- that scores zero here for a reason that
    has nothing to do with the papers they found.
    """

    name: str = "unresolved_citation_forms"


@dataclass(frozen=True)
class MarkerComplianceRateMetric(_DeepScholarMetricBase):
    """Share of answers that delivered under the mandated final-report marker.

    Not a quality measure and never a penalty: an answer that skips the marker
    is scored on its whole text, as every answer was before the contract
    existed. It is here because compliance is a property of the backbone rather
    than of the prompt -- one model puts the line where it was asked to and the
    next buries it in prose -- and scored deliberation looks like a bad report
    unless this number says otherwise.
    """

    name: str = "marker_compliance_rate"


EXPORTABLE_RATE_METRIC = ExportableRateMetric()
DEEPSCHOLAR_METRICS = (
    EXPORTABLE_RATE_METRIC,
    CitationResolutionRateMetric(),
    UnresolvedCitationFormsMetric(),
    MarkerComplianceRateMetric(),
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
        """Run the citation bridge over each answer and record what it achieved."""
        missing_trajectory = 0
        for response in responses:
            if response.trajectory is None:
                missing_trajectory += 1
            sources = sources_from_trajectory(response.trajectory)
            answer = response.outputs[0].text if response.outputs else ""
            scores = score_answer(answer, sources)
            response.scores.update(scores)

            if response.outputs:
                meta = response.outputs[0].metadata
                meta["deepscholar_source_arxiv_ids"] = [s["arxiv_id"] for s in sources]
                meta["deepscholar_num_searches"] = (
                    len(response.trajectory.tool_calls_by_name(SEARCH_TOOL_NAME))
                    if response.trajectory is not None
                    else 0
                )
                for name, value in scores.items():
                    meta[f"score:{name}"] = value

        if missing_trajectory:
            logger.warning(
                "DeepScholar-Bench scored %d/%d responses with no trajectory; this task needs "
                "an agentic harness exposing the %s tool, else nothing is exportable.",
                missing_trajectory,
                len(responses),
                SEARCH_TOOL_NAME,
            )
        return responses
