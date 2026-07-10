"""Citation-grounding scorer.

Evaluates whether claims in a generated, inline-cited report are supported by
the snippets attached to their citations, producing citation precision and
recall via an LLM judge. Reusable across attributed-QA benchmarks (ScholarQA,
ExpertQA).

Ported from astabench citation_eval.py (https://github.com/allenai/asta-bench).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, TypedDict

from olmo_eval.common.scorers.llm_judge import JudgeFn

logger = logging.getLogger(__name__)


class Citation(TypedDict):
    """A citation normalized for judging: an id and its snippet text."""

    id: str
    snippets: str


class GroupedCitationResult(TypedDict):
    """Per-section citation counts aggregated by compute_citation_scores_from_groups."""

    n_attributable: int
    n_extrapolatory: int
    n_half_credit_claims: int
    supporting_counts: list[int]
    non_supporting_counts: list[int]
    n_half_credit_citations: list[int]


# from astabench citation_eval.py: marker for citations where only the title
# (no quoted snippet) is available, which earns half credit.
JUST_HAS_A_TITLE = "Paper content unavailable. The paper's title is: "

# Snippet placeholder that Semantic Scholar returns when no abstract is exposed.
SEMANTIC_SCHOLAR_BAD_SNIPPET = (
    "Please click on the paper title to read the abstract on Semantic Scholar."
)

# from astabench citation_eval.py:CitationEval.score_citation_group
CITATION_GROUP_PROMPT = """You are a claim validator. For each claim made in the following text you will determine if it is supported by the quote from it's corresponding inline citations. As is typically done in academic writing, assume that consecutive sentences can share citations. Make sure to also include claims presented in table format. For references with only the title available (ie no quotes from the reference are included), judge them as `supporting` if the title indicates that the paper is likely relevant to the claim being considered. Return a JSON object with a single key `claims` which is a list of `claim` objects, one for each sentence in the text. Each `claim` object contains the claim itself (`text`), a list of `supporting` inline citations and `non_supporting` inline citations and finally a boolean `is_fully_supported` which indicates if the claim is entirely supported by the quotations in the associated citations. Each inline citation corresponding to that claim should appear in either `supporting` or `non_supporting`, but not both. Each claim made in the text should appear in your output, but you should skip sentences covering high level introductory information.

Text:
{}

References:
{}"""


def extract_json_from_response(response: str) -> dict[str, Any] | None:
    """Extract a JSON object from a model or judge response string."""
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


# from astabench citation_eval.py:clean_sentence
def clean_sentence(sentence: str) -> str:
    """Clean XML tags from model-generated sentences."""
    pattern = r'<Paper [^>]*paperTitle="\W*([ _a-zA-Z0-9,.;]+)\W*"[^>]*/?>\s*</Paper>'
    sentence = re.sub(pattern, r"(\1)", sentence)
    pattern = r'<Model name="Anthropic" version="[^"]+">'
    return re.sub(pattern, "", sentence)


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
    group_results: list[GroupedCitationResult],
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


def _empty_group_result(citation_group: str) -> GroupedCitationResult:
    """Fallback group result counting sentences as unsupported claims."""
    n_sentences = len([s for s in re.split(r"(?<=[.!?])\s+", citation_group) if s.strip()])
    return {
        "n_attributable": 0,
        "n_extrapolatory": max(n_sentences, 1),
        "n_half_credit_claims": 0,
        "supporting_counts": [],
        "non_supporting_counts": [],
        "n_half_credit_citations": [],
    }


# from astabench citation_eval.py:CitationEval.score_citation_group
async def score_citation_group(
    judge_fn: JudgeFn,
    citation_group: str,
    citations: list[Citation],
) -> GroupedCitationResult:
    """Score citations for a single section using the judge.

    Returns counts for aggregation by compute_citation_scores_from_groups.
    """
    if not citations:
        return _empty_group_result(citation_group)

    prompt = CITATION_GROUP_PROMPT.format(
        citation_group,
        "\n\n".join(c["id"] + ": " + c["snippets"] for c in citations),
    )
    raw = await judge_fn(prompt)
    parsed = extract_json_from_response(raw)
    if not parsed or "claims" not in parsed:
        return _empty_group_result(citation_group)

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


def _build_section_citations(raw_citations: list[dict[str, Any]], sec_text: str) -> list[Citation]:
    """Normalize a section's raw citations into id/snippets pairs for judging.

    Citations whose snippets are unusable (absent, echoing the section text, or
    the Semantic Scholar placeholder) fall back to a title-only half-credit
    marker, or empty snippets when no title is available.
    """
    citations: list[Citation] = []
    for c in raw_citations:
        cit_id = c.get("id")
        if not cit_id:
            continue
        snippets = c.get("snippets", [])
        if isinstance(snippets, list):
            snippet_text = "... ".join(str(s) for s in snippets)
        else:
            snippet_text = str(snippets)

        if _filter_citation(c, sec_text) and SEMANTIC_SCHOLAR_BAD_SNIPPET not in snippet_text:
            citations.append({"id": cit_id, "snippets": snippet_text})
        else:
            title = c.get("title", "")
            if title:
                citations.append({"id": cit_id, "snippets": f"{JUST_HAS_A_TITLE}{title}"})
            else:
                citations.append({"id": cit_id, "snippets": ""})
    return citations


# from astabench citation_eval.py:CitationEval
async def score_citations_for_sections(
    judge_fn: JudgeFn,
    parsed_response: dict[str, Any],
) -> dict[str, float]:
    """Score citation precision and recall over a structured, sectioned report.

    Args:
        judge_fn: Async LLM-judge callable used for per-section claim validation.
        parsed_response: Report dict with a ``sections`` list; each section has
            ``text`` and ``citations``, and optionally a ``table`` subsection.

    Returns:
        Mapping with ``citation_precision`` and ``citation_recall`` in [0, 1].
    """
    sections_to_judge: list[tuple[str, list[Citation]]] = []
    for section in parsed_response.get("sections", []):
        sec_iter = [section]
        if section.get("table") and isinstance(section["table"], dict):
            sec_iter.append(section["table"])

        for curr_sec in sec_iter:
            sec_text = curr_sec.get("text", "")
            citations = _build_section_citations(curr_sec.get("citations", []), sec_text)
            sections_to_judge.append((clean_sentence(sec_text), citations))

    if not sections_to_judge:
        return {"citation_precision": 0.0, "citation_recall": 0.0}

    group_results = await asyncio.gather(
        *(score_citation_group(judge_fn, text, citations) for text, citations in sections_to_judge)
    )
    return compute_citation_scores_from_groups(list(group_results))
