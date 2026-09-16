"""Rewrite a DeepScholar-Bench answer's citations into the form the scorer reads.

The benchmark prompt mandates numbered inline citations ("[3]") plus a numbered
reference list. The upstream scorer disagrees: ``DeepScholarBaseParser`` (at the
pinned commit c95413b3b2f3255b461b90d0ce650f685ae2d1ff,
``eval/parsers/deepscholar_base.py``) recognises a citation *only* as a markdown
link whose URL matches ``arxiv.org/abs/``, and a query whose ``intro.md`` holds
none of those yields no documents at all. A prompt-compliant answer therefore
scores zero unless something bridges the two forms.

lit-agents bridges it in ``shared/deepscholar.py::render_intro`` (read-only
reference), and this module mirrors that function's semantics: strip the
reference list, resolve every inline citation against the retrieved sources,
rewrite each resolved one as ``[Title](https://arxiv.org/abs/<id>)``, and delete
the ones that resolve to nothing rather than leave a marker the parser would
misread.

The one thing it cannot mirror is where the numbering comes from. lit-agents'
graphs publish a ``citation_order`` in their state, recording which paper each
number means. Here the model writes its own reference list and nothing else
records the mapping, so :func:`resolve_numbering` parses that list back out of
the answer and matches each entry to a retrieved source by arXiv ID first, then
by title.

Version suffixes are normalised everywhere. The upstream parser keys its
reference map on the raw text between ``arxiv.org/abs/`` and the closing paren,
and compares it against ``paper.csv``'s ``id`` column, so a generated URL
ending ``v2`` against a normalised CSV row silently resolves to an empty title
and abstract.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from olmo_eval.harness.tools.search import normalize_arxiv_id

logger = logging.getLogger(__name__)

# Mirrors of the patterns in lit-agents' shared/deepscholar.py. The two link
# patterns capture the whitespace introducing the citation so that a citation
# resolving to nothing can be removed without leaving a gap in the sentence.
_ARXIV_URL_RE = re.compile(
    r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/([^)\s?#]+)",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(r"([ \t]*)\[([^\]]*)\]\((https?://[^)\s]+)\)")
_SUPERSCRIPT_LINK_RE = re.compile(r"([ \t]*)\[<sup>\[([1-9][0-9]*)\]</sup>\]\((https?://[^)\s]+)\)")
# Deviation from lit-agents' bracket pattern: the negative lookahead. The
# link pass runs first and its output is `[Title](url)`, whose `[Title]` this
# pattern would otherwise match -- and, since titles are aliases, resolve --
# rewriting it to `[Title](url)(url)`.
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\](?!\()")
# A bracket holding only numbers, not followed by "(" so a rendered link label
# can never match. Captures the space that introduces it.
_LEFTOVER_NUMBER_RE = re.compile(
    r"([ \t]*)\[[ \t]*\d{1,3}(?:[ \t]*[,;][ \t]*\d{1,3})*[ \t]*\](?!\()"
)
# A citation label that is only a number is a marker rather than prose.
_NUMERIC_LABEL_RE = re.compile(r"\d{1,3}(?:[ \t]*[,;][ \t]*\d{1,3})*")
_ORPHANED_SPACE_RE = re.compile(r"[ \t]+([.,;:)])")
_REPEATED_SPACE_RE = re.compile(r"[ \t]{2,}")
# lit-agents' _strip_references matches "references" alone (case-insensitive,
# optional heading markup, optional colon). Deliberate deviation: models also
# title this section Bibliography, Works Cited or Sources, and a tail left
# standing is worse than a tail wrongly stripped -- its numbered entries are
# read as prose and every "[1]" in them becomes a fabricated inline citation.
# A strip under any heading but "references" is logged, so the deviation shows
# up in a run rather than only in this comment.
_REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s{0,3}(?:#{1,6}\s*)?(references|bibliography|works\s+cited|sources)\s*:?\s*$"
)
_CANONICAL_REFERENCE_HEADING = "references"

# Which rule supplied the scored text. Reported per query so a run can say
# how it recovered its reports, not merely that it did.
SPLIT_TIER_SENTINEL = "sentinel"
SPLIT_TIER_HEADING = "heading"
SPLIT_TIER_PROSE = "prose"
SPLIT_TIER_NONE = "none"
SPLIT_TIERS = (SPLIT_TIER_SENTINEL, SPLIT_TIER_HEADING, SPLIT_TIER_PROSE, SPLIT_TIER_NONE)

# The delimiter the arxiv_paper_search_agent preset's system prompt asks for.
# Deliberation is not banned -- an agent that plans in prose writes a better
# related-works section, and the benchmark's own user prompt is verbatim and
# cannot be edited to forbid it -- so the contract is a marker instead: think
# above the line, deliver below it.
FINAL_REPORT_MARKER = "=== FINAL REPORT ==="
# Line-exact by construction. A model that names the sentinel inside a sentence
# ("I will write === FINAL REPORT === once I have searched") must not split its
# own answer, so only a whole line can match: MULTILINE anchors with nothing but
# whitespace either side. The run of '=' is loosened to two-or-more because a
# model that pads the rule still means the marker.
_FINAL_REPORT_MARKER_RE = re.compile(
    r"^\s*={2,}\s*FINAL REPORT\s*={2,}\s*$",
    re.MULTILINE,
)

# Citation shapes neither this module nor lit-agents' render_intro resolves.
# Counted rather than ignored: an answer full of them scores zero for a reason
# worth seeing, and silence would make that look like a model that cited nothing.
_UNRESOLVED_FORM_RES = (
    re.compile(r"(?i)<sup>.*?</sup>"),
    re.compile(r"[\u00b9\u00b2\u00b3\u2070\u2074-\u2079]+"),
    re.compile(r"\[\^[^\]]{1,32}\]"),
    re.compile(r"\[[^\]\d]{2,60},\s*(?:19|20)\d{2}[a-z]?\]"),
)
# "[1-3]" and "[1--3]" mean three citations; lit-agents never sees one because
# its graphs emit a citation per paper.
_NUMERIC_RANGE_RE = re.compile(r"^(\d{1,3})\s*[-\u2010-\u2015]\s*(\d{1,3})$")
_MAX_RANGE_SPAN = 50

# One entry of the numbered reference list the prompt asks for: "[3] ..." or
# "3. ..." or "3) ...".
_REFERENCE_ENTRY_RE = re.compile(r"^\s{0,3}(?:\[(\d{1,3})\]|(\d{1,3})[.)])\s+(.*)$")
# An arXiv ID written as prose rather than as a URL, e.g. "arXiv:2401.01234v2".
_ARXIV_PROSE_ID_RE = re.compile(
    r"(?i)arxiv[:\s]\s*(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+/\d{7}(?:v\d+)?)",
)
_TITLE_NOISE_RE = re.compile(r"[^a-z0-9]+")
# Below this, a "title" is too generic to identify a paper by containment.
_MIN_TITLE_MATCH_CHARS = 12


# Exactly what DeepScholarBaseParser credits, copied from its own patterns: a
# markdown link whose URL contains arxiv.org/abs. A bare URL sitting in prose is
# invisible to it, so it must be invisible here too -- counting one would report
# a citation the scorer will never see.
_PARSER_CITATION_RE = re.compile(r"\[([^\]]+?)\]\((https?://[^\)]+)\)")
_PARSER_ARXIV_URL_RE = re.compile(r"arxiv\.org/abs/([^)\s]+)")


def cited_arxiv_ids(body: str) -> list[str]:
    """The arXiv IDs the upstream parser would credit this text with citing."""
    found = []
    for _, url in _PARSER_CITATION_RE.findall(body):
        match = _PARSER_ARXIV_URL_RE.search(url)
        if match is not None:
            arxiv_id = normalize_arxiv_id(match.group(1))
            if arxiv_id:
                found.append(arxiv_id)
    return found


def abs_url(arxiv_id: str) -> str:
    """The only citation URL form the benchmark's parser credits."""
    return f"https://arxiv.org/abs/{normalize_arxiv_id(arxiv_id)}"


def _normalize_alias(value: str) -> str:
    return value.strip().casefold().rstrip("/")


def _normalize_title(value: str) -> str:
    return _TITLE_NOISE_RE.sub(" ", value.casefold()).strip()


def arxiv_ids_in_text(text: str) -> list[str]:
    """Every arXiv ID the text names, as a URL or in prose, normalised."""
    found = [normalize_arxiv_id(match.group(1)) for match in _ARXIV_URL_RE.finditer(text)]
    found.extend(normalize_arxiv_id(match.group(1)) for match in _ARXIV_PROSE_ID_RE.finditer(text))
    return [arxiv_id for arxiv_id in found if arxiv_id]


def split_final_report(answer: str) -> tuple[str, bool]:
    """Return the deliverable half of an answer, and whether a marker chose it.

    The LAST line-exact marker wins. A model that starts its report, thinks
    better of it and starts again writes the marker twice, and the attempt it
    stood behind is the last one; taking the first would score the draft it
    abandoned. Everything above the winning marker is deliberation the
    benchmark never asked for and is dropped along with the marker line itself,
    so the retained tail is what the reference-list and citation passes read.

    An answer with no line-exact marker comes back whole, which is exactly what
    this module did before the contract existed -- a backbone that ignores the
    instruction is scored no worse than it was. That fallback is deliberate but
    not silent: the second return value is what ``marker_compliance_rate``
    counts, so non-compliance surfaces as a number per backbone rather than as
    unexplained deliberation sitting in the scored text.
    """
    matches = list(_FINAL_REPORT_MARKER_RE.finditer(answer))
    if not matches:
        return answer, False
    return answer[matches[-1].end() :].lstrip("\n"), True


def strip_references(report: str) -> str:
    """Drop the reference list; the scorer reads citations from the prose only."""
    match = _REFERENCE_HEADING_RE.search(report)
    if match is None:
        return report.rstrip()
    heading = match.group(1).strip().casefold()
    if heading != _CANONICAL_REFERENCE_HEADING:
        logger.warning(
            "Stripped a reference tail under the heading %r; lit-agents strips "
            "only 'References', so this answer is treated differently there.",
            match.group(1).strip(),
        )
    return report[: match.start()].rstrip()


def unresolved_citation_forms(body: str) -> int:
    """Count citation-shaped fragments no pass here can turn into a link.

    Superscripts, footnote markers and author-year brackets are real citation
    styles that neither this module nor render_intro resolves. They are reported
    so an answer that cited diligently in an unsupported style is distinguishable
    from one that did not cite at all.
    """
    return sum(len(pattern.findall(body)) for pattern in _UNRESOLVED_FORM_RES)


def reference_list(report: str) -> dict[str, str]:
    """Parse the answer's own numbered reference list into number -> entry text.

    Continuation lines belong to the entry above them, because a reference that
    wraps is still one reference.
    """
    match = _REFERENCE_HEADING_RE.search(report)
    tail = report[match.end() :] if match else report
    entries: dict[str, list[str]] = {}
    current: str | None = None
    for line in tail.splitlines():
        entry_match = _REFERENCE_ENTRY_RE.match(line)
        if entry_match is not None:
            number = entry_match.group(1) or entry_match.group(2)
            current = str(int(number))
            entries.setdefault(current, []).append(entry_match.group(3).strip())
        elif current is not None and line.strip():
            entries[current].append(line.strip())
        elif not line.strip():
            current = None
    return {number: " ".join(parts).strip() for number, parts in entries.items()}


def _match_reference(
    entry: str,
    by_arxiv_id: Mapping[str, Mapping[str, Any]],
    by_title: Sequence[tuple[str, Mapping[str, Any]]],
) -> Mapping[str, Any] | None:
    """Find the retrieved source a reference-list entry names.

    ID first because it is exact; title only as a fallback, longest first so a
    short title cannot claim a match that belongs to a longer one containing it.
    """
    for arxiv_id in arxiv_ids_in_text(entry):
        source = by_arxiv_id.get(arxiv_id)
        if source is not None:
            return source

    normalized_entry = _normalize_title(entry)
    for title, source in by_title:
        if len(title) >= _MIN_TITLE_MATCH_CHARS and title in normalized_entry:
            return source
    return None


def resolve_numbering(
    report: str,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Map each reference number the answer published to a retrieved source.

    This stands in for the ``citation_order`` lit-agents' graphs publish as
    state. A number whose entry names no retrieved source stays unmapped, and
    its inline citations are dropped rather than pointed at the wrong paper.
    """
    by_arxiv_id = {normalize_arxiv_id(str(s.get("arxiv_id") or "")): s for s in sources}
    by_title = sorted(
        (
            (_normalize_title(str(s.get("title") or "")), s)
            for s in sources
            if str(s.get("title") or "").strip()
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    resolved: dict[str, Mapping[str, Any]] = {}
    for number, entry in reference_list(report).items():
        source = _match_reference(entry, by_arxiv_id, by_title)
        if source is not None:
            resolved[number] = source
    return resolved


def _citation_tokens(content: str) -> list[str]:
    """Split a bracket's contents into the citation markers it stands for.

    Expands "1-3" into 1, 2, 3 and drops the caret of a "^1" footnote marker, so
    both resolve through the same numbering as a plain "[1]". A range wider than
    _MAX_RANGE_SPAN is left alone: at that width it is far likelier to be a page
    range or a year span than a citation.
    """
    tokens: list[str] = []
    for raw in re.split(r"[\s,;]+", content):
        token = raw.strip().lstrip("^")
        if not token:
            continue
        span = _NUMERIC_RANGE_RE.match(token)
        if span is None:
            tokens.append(token)
            continue
        first, last = int(span.group(1)), int(span.group(2))
        if first > last or last - first > _MAX_RANGE_SPAN:
            tokens.append(token)
            continue
        tokens.extend(str(number) for number in range(first, last + 1))
    return tokens


def _markdown_citation(source: Mapping[str, Any]) -> str:
    arxiv_id = normalize_arxiv_id(str(source.get("arxiv_id") or ""))
    visible = str(source.get("title") or arxiv_id).replace("[", "").replace("]", "").strip()
    return f"[{visible or arxiv_id}]({abs_url(arxiv_id)})"


def _deduplicate(sources: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for source in sources:
        arxiv_id = normalize_arxiv_id(str(source.get("arxiv_id") or ""))
        if arxiv_id:
            by_id.setdefault(arxiv_id, source)
    return list(by_id.values())


def rewrite_intro(
    report: str,
    sources: Sequence[Mapping[str, Any]],
) -> tuple[str, list[Mapping[str, Any]]]:
    """Rewrite one answer's citations into resolvable arxiv.org links.

    Mirrors lit-agents' ``render_intro``. Returns the rewritten body and the
    sources it ends up citing, in retrieval order. An empty source list is the
    caller's signal that this answer yields nothing the upstream parser can
    score -- which is a result to record, not an error to raise, because a model
    that cited only papers it never retrieved has genuinely produced nothing
    citable.
    """
    if not report.strip():
        return "", []
    ordered = _deduplicate(sources)
    if not ordered:
        return "", []

    alias_map: dict[str, Mapping[str, Any]] = {}
    for source in ordered:
        arxiv_id = normalize_arxiv_id(str(source.get("arxiv_id") or ""))
        alias_map[_normalize_alias(arxiv_id)] = source
        alias_map[_normalize_alias(f"arXiv:{arxiv_id}")] = source
        alias_map[_normalize_alias(abs_url(arxiv_id))] = source
        title = str(source.get("title") or "").strip()
        if title:
            alias_map[_normalize_alias(title)] = source
    alias_map.update(resolve_numbering(report, ordered))

    body = strip_references(report)
    removals = 0

    def strip_citation(leading: str, label: str) -> str:
        """Drop a citation that resolves to nothing, keeping any prose label.

        The visible label survives, which is what the upstream parser does with
        a link it cannot resolve. A label that is only a number carries no
        prose: it is a citation marker, and leaving one behind is what makes the
        parser read citation 8 of a six-document list.
        """
        nonlocal removals
        removals += 1
        if _NUMERIC_LABEL_RE.fullmatch(label.strip()):
            return ""
        return f"{leading}{label}"

    def replace_superscript(match: re.Match[str]) -> str:
        source = alias_map.get(_normalize_alias(match.group(3)))
        if source is not None:
            return f"{match.group(1)}{_markdown_citation(source)}"
        return strip_citation(match.group(1), match.group(2))

    body = _SUPERSCRIPT_LINK_RE.sub(replace_superscript, body)

    def replace_link(match: re.Match[str]) -> str:
        source = alias_map.get(_normalize_alias(match.group(3)))
        if source is None:
            ids = arxiv_ids_in_text(match.group(3))
            source = alias_map.get(_normalize_alias(ids[0])) if ids else None
        if source is not None:
            return f"{match.group(1)}{_markdown_citation(source)}"
        return strip_citation(match.group(1), match.group(2))

    body = _MARKDOWN_LINK_RE.sub(replace_link, body)

    def replace_bracket(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        direct = alias_map.get(_normalize_alias(content))
        if direct is not None:
            return _markdown_citation(direct)
        tokens = _citation_tokens(content)
        resolved = [alias_map.get(_normalize_alias(token)) for token in tokens]
        if tokens and all(source is not None for source in resolved):
            seen: dict[str, Mapping[str, Any]] = {}
            for source in resolved:
                if source is not None:
                    seen.setdefault(normalize_arxiv_id(str(source.get("arxiv_id") or "")), source)
            return "".join(_markdown_citation(source) for source in seen.values())
        return match.group(0)

    body = _BRACKET_RE.sub(replace_bracket, body)

    # A bracket number no pass above resolved is not a citation this benchmark
    # can score, and leaving it in makes the parser number a document that has
    # no row in paper.csv.
    leftovers = _LEFTOVER_NUMBER_RE.findall(body)
    if leftovers:
        removals += len(leftovers)
        body = _LEFTOVER_NUMBER_RE.sub("", body)
    # Tidy only when something was removed, so a report needing no deletion
    # comes back exactly as the passes above rendered it.
    if removals:
        body = _REPEATED_SPACE_RE.sub(" ", body)
        body = _ORPHANED_SPACE_RE.sub(r"\1", body)
    body = body.strip()

    cited = list(dict.fromkeys(cited_arxiv_ids(body)))
    exported = [
        source
        for source in ordered
        if normalize_arxiv_id(str(source.get("arxiv_id") or "")) in cited
    ]
    if not exported:
        return "", []
    return body + "\n", exported


# The model's own Related Works heading, in the four shapes the archived runs
# actually produced: markdown hashes, bold, a numbered section, or a bare line.
# This is the fallback that does the real work -- a marker is only present on 55%
# of 9B answers, while a heading is present on 60-95% depending on the backbone.
_RELATED_WORKS_TITLE = r"related[ \t]+works?(?:[ \t]+(?:and|/)[ \t]+background)?"
_HEADING_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in (
        r"^[ \t]{0,3}#{1,6}[ \t]*" + _RELATED_WORKS_TITLE + r"[ \t]*:?[ \t]*$",
        r"^[ \t]{0,3}\*\*[ \t]*" + _RELATED_WORKS_TITLE + r"[ \t]*\*\*[ \t]*:?[ \t]*$",
        r"^[ \t]{0,3}\d{1,2}[.)]?[ \t]+" + _RELATED_WORKS_TITLE + r"[ \t]*:?[ \t]*$",
        r"^[ \t]{0,3}" + _RELATED_WORKS_TITLE + r"[ \t]*:?[ \t]*$",
    )
)
# A line that opens a numbered reference entry, and the share of such lines above
# which a paragraph is a reference list rather than prose.
_REFERENCE_ENTRY_LINE_RE = re.compile(r"^[ \t]{0,3}(?:\[\d{1,3}\]|\d{1,3}[.)])[ \t]+")
_REFERENCE_BLOCK_RATIO = 0.5
_INLINE_NUMERIC_CITATION_RE = re.compile(r"\[[ \t]*\d{1,3}(?:[ \t]*[-,;][ \t]*\d{1,3})*[ \t]*\]")
_ARXIV_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*arxiv\.org/abs[^)]*\)", re.IGNORECASE)
# Below this a paragraph is too short to be the opening of a related-works
# section; it is a caption, a fragment, or a note to self.
_MIN_PROSE_WORDS = 30


def _reference_heading_start(report: str) -> int | None:
    match = _REFERENCE_HEADING_RE.search(report)
    return match.start() if match else None


def heading_split(report: str) -> str | None:
    """Everything from the model's FIRST Related Works heading, or None.

    First rather than last, which is the opposite of the sentinel's rule and was
    settled by measurement (U0009) rather than by analogy. Last-wins loses to a
    late skeleton draft -- 27B writes `Related Works ... References 1. ... Need
    maybe reference list not counted? fine.` near the end of some answers, and
    that outranks the real report written earlier. First-wins never lands after
    the sentinel on any answer where both exist, so it cannot cut into a report.

    The heading is retained, not stripped: it is part of the report and is clean
    markdown for the organisation judge.
    """
    limit = _reference_heading_start(report)
    starts = sorted(
        match.start()
        for pattern in _HEADING_PATTERNS
        for match in pattern.finditer(report)
        if limit is None or match.start() < limit
    )
    return report[starts[0] :] if starts else None


def _is_reference_block(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    if not lines:
        return False
    entries = sum(1 for line in lines if _REFERENCE_ENTRY_LINE_RE.match(line))
    return entries / len(lines) >= _REFERENCE_BLOCK_RATIO


def prose_split(report: str) -> str | None:
    """Everything from the first citation-bearing prose paragraph, or None.

    The last resort, for an answer that marks its report neither with the
    sentinel nor with a heading. It only ever drops LEADING paragraphs, so it
    cannot damage a reference list or truncate a report that began earlier; its
    worst case is trimming too little, which the measurements confirm is how it
    fails.
    """
    limit = _reference_heading_start(report)
    position = 0
    for separator in re.finditer(r"\n[ \t]*\n", report):
        block = report[position : separator.start()]
        if limit is not None and position >= limit:
            return None
        if _qualifies_as_report_opening(block):
            return report[position:]
        position = separator.end()
    if (limit is None or position < limit) and _qualifies_as_report_opening(report[position:]):
        return report[position:]
    return None


def _qualifies_as_report_opening(block: str) -> bool:
    if _is_reference_block(block):
        return False
    if len(block.split()) < _MIN_PROSE_WORDS:
        return False
    return bool(_INLINE_NUMERIC_CITATION_RE.search(block) or _ARXIV_MARKDOWN_LINK_RE.search(block))


def select_scored_text(
    report: str,
    sources: Sequence[Mapping[str, Any]],
) -> tuple[str, str, bool, bool]:
    """Choose the text to score, and say which rule chose it.

    Returns ``(text, split_tier, marker_found, marker_misused)`` where
    ``split_tier`` is one of ``sentinel``, ``heading``, ``prose`` or ``none``.

    Three rules are tried in descending order of how explicitly the model marked
    its report: the sentinel it was asked to write, its own Related Works
    heading, then the first citation-bearing paragraph. Each candidate must pass
    the same fuse before it is accepted, so a tier that would cost the answer its
    citations is skipped rather than trusted, and ``none`` means every rule was
    refused and the full text was kept.

    :func:`split_final_report` decides where the deliverable starts; this decides
    whether trusting it would cost the answer its export. A model can honour the
    marker and still misplace the report -- Qwen3.5-9B wrote its whole Related
    Works section while deliberating and put only the numbered list below the
    line -- and the split would then hand the scorer a reference list with
    nothing citing into it, turning an exportable answer into an unscoreable one.

    So the split is applied only when it keeps something scoreable. If the tail
    resolves no citation but the full answer does, the full answer is kept: the
    marker-absent path exactly, which makes the split lossless by construction --
    it can never lower ``exportable_rate`` relative to not splitting at all.
    ``marker_misused`` counts those saves, and is deliberately distinct from
    ``marker_compliance_rate``: the instruction was followed, the report was put
    in the wrong place, and a run needs to tell those apart to know whether the
    prompt or the backbone is at fault.

    Note the test is "resolves no citation", not "is empty after
    :func:`strip_references`". A bare numbered list carrying no ``References``
    heading is not stripped, so it survives as prose and the emptiness test never
    fires -- which is exactly the shape both real 9B failures took.
    """
    sources = list(sources)
    _, cited_from_full = rewrite_intro(report, sources)
    full_is_scoreable = bool(cited_from_full)

    sentinel_tail, marker_found = split_final_report(report)
    marker_misused = False
    if marker_found:
        _, cited_from_sentinel = rewrite_intro(sentinel_tail, sources)
        # The marker was written and the report was not put under it. Recorded
        # whatever any later tier manages to recover, because it measures the
        # prompt being misread, not the outcome.
        marker_misused = not cited_from_sentinel and full_is_scoreable

    candidates: list[tuple[str, str]] = []
    if marker_found:
        candidates.append((SPLIT_TIER_SENTINEL, sentinel_tail))
    heading_tail = heading_split(report)
    if heading_tail is not None:
        candidates.append((SPLIT_TIER_HEADING, heading_tail))
    prose_tail = prose_split(report)
    if prose_tail is not None:
        candidates.append((SPLIT_TIER_PROSE, prose_tail))

    for tier, tail in candidates:
        _, cited = rewrite_intro(tail, sources)
        if cited:
            return tail, tier, marker_found, marker_misused

    if full_is_scoreable:
        # Every tier would have cost the answer its citations, so keep all of it.
        return report, SPLIT_TIER_NONE, marker_found, marker_misused
    if candidates:
        # Nothing is scoreable anywhere; prefer the most specific candidate,
        # which is at least the text the model nominated.
        tier, tail = candidates[0]
        return tail, tier, marker_found, marker_misused
    return report, SPLIT_TIER_NONE, marker_found, marker_misused
