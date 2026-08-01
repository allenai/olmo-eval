"""SAGE short-form scientific paper retrieval task.

SAGE asks a model to identify a target paper from a reasoning-intensive query.
This task scores the model's final output, not the agent trajectory.

Requirements: this task only measures retrieval when run through a
tool-providing agentic harness (scaffold that executes tool calls, e.g. the
`paper_search_agent` preset). Run without tools, the model answers from
parametric memory and scores can look plausible but do not measure retrieval.
"""

from __future__ import annotations

import logging
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
from olmo_eval.evals.tasks.common.base import _store_output_score

logger = logging.getLogger(__name__)

SAGE_REPO = "allenai/sage-retrieval"  # HF dataset repo in the allenai org

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

    SAGE scores short-form EM as whether the gold paper is "included in the output
    text or citations" (SAGE paper, arXiv:2602.05975); we implement that as a
    normalized-substring inclusion check over the model's single output string (any
    citations the model emits are part of that string, not a separate parsed
    field). The normalization follows the SAGE authors' reference EM
    implementation, confirmed with them directly (the public release ships data
    only, no scoring code): the title is lowercased, every run of characters
    outside [a-z0-9] is collapsed to a single space, and the result is trimmed, so
    punctuation and any non-ASCII letters or digits (accented or non-Latin) are
    dropped rather than transliterated. Article words (a/an/the) are kept.
    """
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def strip_think(text: str) -> str:
    """Drop balanced, unopened and truncated think regions, preserving visible text.

    An unopened ``</think>`` closes a reasoning region that began before the text
    starts, which is what a modern thinking template produces: it writes the
    opening ``<think>`` into the generation prompt, so the completion carries only
    the closing tag. Everything ahead of such a tag is monologue and is dropped.

    Keeping it would be worse here than elsewhere, because SAGE matches a gold
    title by substring: a title the model merely weighed and then rejected while
    reasoning would score as a hit, inflating exact match. Dropping it matches the
    ResearchQA and DeepResearch Bench extractors, which likewise cut at the first
    ``</think>``.
    """
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
                # Unopened close tag: discard this region rather than emitting it.
                output.clear()
                index = next_close + len(close_tag)
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


# ---------------------------------------------------------------------------
# Answer-only scoring for SAGE short-form
# ---------------------------------------------------------------------------
#
# SAGE short-form's prompt states a contract: "After searching, give your single
# best answer: state the most likely paper's title, or explicitly say no match
# was found."  ``exact_match`` above deliberately does not use that contract --
# it is SAGE's published metric (normalized-title substring over the whole
# output) and must stay byte-for-byte comparable with external numbers.
#
# The metrics below read the contract instead, so that a run can be inspected
# for two failure modes ``exact_match`` cannot distinguish:
#
#   * volume artifacts -- a reference list, a rejected-candidate list, or simply
#     a longer report raises ``exact_match`` without the system ever answering;
#   * abstention -- declining and answering wrongly both score 0, so guessing is
#     free and abstaining is punished.
#
# Extraction is deliberately conservative: it never searches the whole output
# for a title, only the one statement in which the system states its answer.
# Anything it cannot resolve is reported as unparsed rather than guessed at, and
# ``answer_unparsed_rate`` is exported so a reader can tell when
# ``answer_only_match`` is not trustworthy for a given system.
#
# Outputs that both decline and name a paper are the hard case, and they are not
# rare: measured over five 599-row runs they are 52.9% of AgentDisCo's instances
# (of which 41.9% decline first and then name a paper), 9.7% Arman's, 5.3% dp1,
# 3.7% the single-agent baseline and 1.3% Allyson's. Asking "did it commit?" of
# such an output has no answer, so the rate metrics do not ask. They partition
# the run four ways -- ``commit_rate``, ``decline_rate``, ``hedge_rate`` and
# ``answer_unparsed_rate``, which sum to 1 -- and every one of the four is
# invariant to how a hedge is resolved. Whatever tie-break rule a reader prefers,
# its commit rate is bounded by ``commit_rate`` and ``commit_rate + hedge_rate``;
# on AgentDisCo that band is 0.404 to 0.933, which is why no single commit number
# was reportable before.
#
# ``answer_only_match`` still needs one title per output, so it keeps the
# first-in-reading-order rule; it is stable under that choice (AgentDisCo 0.080
# to 0.109 across first, last, prefer-commit and prefer-decline).
#
# What the five runs show, on those same measurements: reading the stated answer
# instead of the whole output moves AgentDisCo 0.2805 -> 0.1002 and Allyson's
# 0.1503 -> 0.0000, while dp1 barely moves, 0.0968 -> 0.0818. Of each system's
# official ``exact_match`` score, the share earned only outside the answer region
# is AgentDisCo 0.6%, baseline 4.0%, dp1 0.0%, Arman's 23.8%, Allyson's 86.7%.
#
# These metrics do not all re-rank the systems in different ways: across the five
# runs ``exact_match``, ``answer_only_match``, ``commit_rate`` and
# ``accuracy_given_commit`` produce three distinct orderings, not four, because
# ``answer_only_match`` and ``commit_rate`` order the systems identically
# (AgentDisCo > baseline > dp1 > Arman's > Allyson's).

_THINK_CLOSE = "</think>"
_THINK_OPEN = "<think>"

#: A bibliography heading on a line of its own. Everything from the last such
#: heading to the end of the output is a reference list, not an answer.
_REFERENCE_HEADING = re.compile(
    r"(?im)^[ \t]{0,3}(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*"
    r"(?:references?|bibliography|sources?|citations?|works\ cited|reference\ list|"
    r"cited\ works|(?:candidate\ )?papers?\ (?:found|retrieved|consulted|reviewed|considered))"
    r"[ \t]*(?:\*\*|__)?[ \t]*:?[ \t]*$"
)

_HEADING_MARK = re.compile(r"^#{1,6}[ \t]*")
_BULLET_MARK = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])[ \t]+(?=[A-Z0-9\"“*\[(])")

#: Upper bound on the statement string recorded in output metadata, so a
#: predictions file does not carry a multi-kilobyte blob on every row. It bounds
#: what is *written down*; extraction reads the whole statement.
#:
#: It used to be applied to the statement before matching, on the rationale that
#: a longer statement is prose rather than an answer. Truncating does not enforce
#: that rationale: it does not reject a long paragraph, it searches its first 600
#: characters anyway. Measured over the five 599-row runs, 3.7% of the 187,668
#: answer statements exceed 600 characters, and truncating them destroyed the
#: committed title on 13 AgentDisCo instances, 3 of which named the gold paper --
#: so as a matching bound 600 was not merely arbitrary, it was wrong. What keeps
#: prose out of the metric is the commit predicate plus a delimited title, not a
#: character count.
_MAX_RECORDED_STATEMENT_CHARS = 600


def answer_region(text: str) -> str:
    """Return the part of an output in which the stated answer can live.

    Three things are removed, in order:

    1. ``<think>`` regions, via :func:`strip_think`.
    2. A reasoning region that was closed but never opened. Some scaffolds emit
       the chain of thought followed by a bare ``</think>`` and then the visible
       answer; :func:`strip_think` keeps that leading text (it has no opening
       tag to pair with), so everything up to the last lone ``</think>`` is
       dropped here. This only affects the answer metrics -- ``exact_match``
       still scores the string SAGE says it scores.
    3. A trailing bibliography, from the last reference heading onwards, when
       that heading sits in the back 60% of the text.
    """
    region = strip_think(text)
    if _THINK_CLOSE in region and _THINK_OPEN not in region:
        region = region[region.rindex(_THINK_CLOSE) + len(_THINK_CLOSE) :]
    headings = list(_REFERENCE_HEADING.finditer(region))
    if headings:
        last = headings[-1]
        if last.start() >= 0.4 * len(region):
            region = region[: last.start()]
    return region.strip()


def answer_statements(region: str) -> list[str]:
    """Split an answer region into answer-sized statements.

    Markdown line structure is honoured before sentence structure: heading and
    bullet markers are dropped, and a line ending in a colon is joined to the
    next line, because "the most likely paper is:" puts its answer on the
    following line. Statements are returned whole; see
    :data:`_MAX_RECORDED_STATEMENT_CHARS` for why they are no longer truncated.
    """
    lines = []
    for raw in region.split("\n"):
        line = _BULLET_MARK.sub("", _HEADING_MARK.sub("", raw.strip())).strip()
        if line:
            lines.append(line)

    joined: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.rstrip().rstrip("*_ ").endswith(":") and index + 1 < len(lines):
            line = f"{line} {lines[index + 1]}"
            index += 2
        else:
            index += 1
        joined.append(line)

    statements: list[str] = []
    for line in joined:
        for piece in _SENTENCE_BREAK.split(line):
            piece = piece.strip()
            if piece:
                statements.append(piece)
    return statements


#: Ways the five systems we have run actually word "no match was found". Kept as
#: a list so a new phrasing can be added without re-reading one large regex.
_DECLINE_PATTERNS = (
    r"no\s+match(?:ing)?\b(?:\s+\w+){0,3}?\s+(?:was\s+|were\s+|is\s+|could\s+be\s+)?found",
    r"no\s+match\s+found",
    r"^no\s+match\b\W*$",
    r"answer\s*:\s*\**\s*no\s+match",
    r"no\s+(?:single\s+)?(?:paper|study|article|publication|candidate|document)\b"
    r"(?:\s+\S+){0,12}?\s+(?:match(?:es|ing|ed)?|satisf(?:ies|y)|meet(?:s)?|corresponds?|fulfil?l?s)",
    r"no\s+(?:single\s+)?(?:paper|study|article|publication|candidate|document)\b"
    r"(?:\s+\S+){0,12}?\s+(?:was|were|is|are|could\s+be)\s+(?:found|identified|located)",
    r"(?:do(?:es)?|did)\s+not\s+contain\s+(?:a|an|any|the)\b(?:\s+\S+){0,6}?\s*"
    r"(?:paper|stud|article|publication|document|match)",
    r"cannot\s+be\s+(?:definitively\s+|conclusively\s+|reliably\s+)?"
    r"(?:determined|identified|located|found|confirmed|established)",
    r"(?:could|can)\s*not\s+be\s+(?:definitively\s+|conclusively\s+|reliably\s+)?identified",
    r"(?:could|can)\s*not\s+(?:definitively\s+|conclusively\s+|reliably\s+)?"
    r"(?:find|identify|determine|locate|pinpoint)\b",
    r"(?:un(?:able|successful)|failed)\s+to\s+(?:definitively\s+|conclusively\s+)?"
    r"(?:find|identify|determine|locate|pinpoint)\b",
    r"does\s+not\s+appear\s+to\s+exist",
    r"remains?\s+unidentified",
    r"not\s+(?:been\s+)?identified\s+(?:in|among|from|within)\b",
    r"none\s+of\s+the\s+(?:provided\s+|retrieved\s+|available\s+|listed\s+)?"
    r"(?:candidates?|papers?|studies|articles|documents|summaries)\b"
    r"(?:\s+\S+){0,10}?\s*(?:match|satisf|meet|fulfil|correspond)",
    r"no\s+(?:exact\s+|definitive\s+|clear\s+)?(?:match|answer)\b(?:\s+\S+){0,6}?\s+"
    r"(?:was|is|could\s+be)\s+(?:found|identified|determined)",
    r"target\s+paper\s+(?:was\s+)?not\s+identified",
    r"not\s+found\s+(?:in|among|within)\s+the\s+(?:provided|available|retrieved|search)",
    r"(?:fails?|failed)\s+to\s+contain\s+(?:a|an|any|the)\b",
    r"there\s+is\s+no\s+(?:paper|study|article|match)\b",
)
DECLINE_RE = re.compile("(?ix)" + "|".join(f"(?:{p})" for p in _DECLINE_PATTERNS))

#: A title the system has set apart: quoted, bolded, or a markdown link label.
#: Requiring a delimited title is what stops a whole sentence of prose from
#: being read as "the answer".
_TITLE_OBJECT = re.compile(
    r"\"(?P<double>[^\"\n]{10,300})\""
    r"|“(?P<curly>[^”\n]{10,300})”"
    r"|\*\*(?P<bold>[^*\n]{10,300})\*\*"
    r"|\[(?P<link>[^\]\n]{10,300})\]"
    r"|'(?P<single>[^'\n]{10,300})'"
)

_ANSWER_NOUN = r"(?:paper|study|article|publication|work|match|answer|candidate|title|manuscript)"

#: A delimited span that describes the answer instead of naming it. Two ways this
#: arises, both measured on real outputs:
#:
#: * the system writes the description first --
#:   ``The best match is **the 2024 ACL paper** "<Title>"`` used to extract
#:   ``the 2024 ACL paper``;
#: * markdown bold pairing manufactures one -- in
#:   ``**Primary Candidate:** The paper **"<Title>"**`` the closing ``**`` of the
#:   label and the opening ``**`` of the title bracket the words between them, so
#:   the first bold span is ``The paper``. That happens on AgentDisCo docs 31,
#:   321 and 326, and on 326 the span it displaced is the gold title.
_DESCRIPTOR_SPAN = re.compile(
    r"(?i)^(?:the|an?|this|that|these|those|his|her|its|their|our|said|above|following)\s+"
    r"[^\n]{0,80}?" + _ANSWER_NOUN + r"s?$"
)
_COMMIT_PATTERNS = (
    r"\bthe\s+(?:single\s+|most\s+likely\s+|best\s+|target\s+|identified\s+|correct\s+"
    r"|matching\s+|closest\s+|strongest\s+|only\s+)*" + _ANSWER_NOUN + r"\b[^.\n]{0,140}?"
    r"\b(?:is|are|appears\s+to\s+be|seems\s+to\s+be|would\s+be|must\s+be)\b\s*:?",
    r"\b" + _ANSWER_NOUN + r"\s+(?:that|which)\s+(?:best\s+)?"
    r"(?:match(?:es)?|satisf(?:ies|y)|meet(?:s)?|fit(?:s)?|correspond(?:s)?|align(?:s)?)\b"
    r"[^.\n]{0,140}?\bis\b\s*:?",
    r"\b(?:i|we)\s+(?:have\s+)?identif(?:y|ied)\b",
    r"\bthis\s+is\s+the\s+" + _ANSWER_NOUN + r"\b",
    r"\bidentified\s+(?:paper|title|work|study)\s*:",
    r"\b(?:final\s+|most\s+likely\s+|best\s+|direct\s+|my\s+)?answer\s*(?:\*\*|__)?\s*:",
    r"\bprimary\s+candidate\s*:",
    r"\bbest\s+(?:match|candidate|guess)\s*:",
    r"\b(?:most\s+likely|best)\s+(?:paper|title|match|candidate)\s*:",
)
COMMIT_RE = re.compile("(?ix)" + "|".join(f"(?:{p})" for p in _COMMIT_PATTERNS))

#: An explicit answer label opening a statement. Only after one of these is an
#: undelimited title accepted, because "Direct Answer: Target Paper Identified"
#: style headings would otherwise be read as titles.
_ANSWER_LABEL = re.compile(
    r"(?i)^[\W_]{0,4}(?:final\s+answer|most\s+likely\s+answer|best\s+answer|my\s+answer|answer)"
    r"[\W_]{0,4}:\s*(?P<value>.+)$"
)

#: A line that is nothing but a delimited title -- a bare answer with no
#: surrounding sentence, which the prompt's contract also permits.
_STANDALONE_TITLE = re.compile(
    r"^\W{0,3}(?:\*\*(?P<bold>[^*\n]{10,300})\*\*|\"(?P<double>[^\"\n]{10,300})\")\W{0,3}$"
)

#: Longest a bold standalone line may be and still read as a section heading
#: rather than a paper title.
#:
#: Bolded section headings look exactly like bare answers: ``**Claims with Strong
#: Consensus**`` opens a section on Arman's docs 22 and 83 and used to be
#: extracted as the committed title, and ``**Final Answer**`` above the real
#: answer has the same shape. Three properties separate them from titles -- a
#: heading is short, carries no subtitle colon, and has no digits -- and a fourth
#: separates a heading from a bare answer: body text follows it.
#:
#: The bound is measured rather than guessed. SAGE short-form's 599 gold titles
#: have a median of 12 words, and only 20 of them (3.3%) are five words or fewer
#: with no colon and no digit, so this rule can cost at most 3.3% of bare-title
#: answers -- and only for a title that is bolded rather than quoted and has text
#: after it. On the five runs it costs nothing: the bare answers all survive it,
#: and the only two extractions it removes are the two headings.
_MAX_HEADING_WORDS = 5


def _is_bold_section_heading(value: str) -> bool:
    """Whether a bold standalone line reads as a section heading, not a title."""
    return (
        len(value.split()) <= _MAX_HEADING_WORDS
        and ":" not in value
        and re.search(r"\d", value) is None
    )


@dataclass(frozen=True, slots=True)
class SageAnswer:
    """What a short-form output states, per the prompt's contract.

    Two independent readings are kept, because they answer different questions.

    ``states_commit`` / ``states_decline`` say what the output contains: whether
    any statement in the answer region names a paper, and whether any declines.
    Both can be true. They are the basis of every rate metric, and they do not
    depend on which statement is picked as *the* answer.

    ``kind``, ``statement`` and ``title`` are the single answer, resolved
    first-in-reading-order, and are what ``answer_only_match`` scores. ``kind``
    is one of:

    ``"commit"``
        The output names a paper. ``title`` holds it.
    ``"decline"``
        The output says no match was found. ``title`` is ``None``.
    ``"unparsed"``
        Neither. The output did not follow the contract (typically it is a
        literature survey that never answers), or it worded its answer in a way
        this extractor does not recognise. These are reported, never guessed at.
    """

    kind: str
    statement: str
    title: str | None
    states_commit: bool
    states_decline: bool

    @property
    def hedged(self) -> bool:
        """Whether the output both declines and names a paper."""
        return self.states_commit and self.states_decline

    @property
    def committed(self) -> bool:
        return self.kind == "commit"

    @property
    def declined(self) -> bool:
        return self.kind == "decline"

    @property
    def unparsed(self) -> bool:
        return self.kind == "unparsed"

    @property
    def commits_without_hedging(self) -> bool:
        """Names a paper and declines nowhere -- an unambiguous commit."""
        return self.states_commit and not self.states_decline

    @property
    def declines_without_hedging(self) -> bool:
        """Declines and names no paper -- an unambiguous decline."""
        return self.states_decline and not self.states_commit


def _title_after(tail: str) -> str | None:
    """Return the first delimited span in ``tail`` that names a paper.

    Spans that describe the answer rather than naming it are skipped; see
    :data:`_DESCRIPTOR_SPAN`.
    """
    for found in _TITLE_OBJECT.finditer(tail):
        value = next(group for group in found.groups() if group)
        value = value.strip().strip("*_ \t").strip("\"'“”")
        if not value or _DESCRIPTOR_SPAN.match(value):
            continue
        return value
    return None


def _labelled_title(statement: str) -> str | None:
    """Return an undelimited title stated after an explicit answer label."""
    label = _ANSWER_LABEL.match(statement)
    if not label:
        return None
    value = re.split(r"(?<=[.!?])\s", label.group("value"))[0]
    value = value.strip().strip("*_ \t.")
    return value if 10 <= len(value) <= 300 else None


def committed_title(statement: str) -> str | None:
    """Return the paper title a statement commits to, or ``None``.

    A commit needs both an identification predicate ("the paper that matches the
    query is", "Answer:") and a title the statement sets apart. Requiring both
    is what keeps incidental prose -- "the identified paper confirms that ..." --
    out of the metric.

    Those two requirements are also what rejects a section heading that quotes
    the prompt back ("Direct Answer: Identification of the Most Likely Paper's
    Title"): it delimits no title, and :data:`_ANSWER_LABEL` is anchored at the
    start of the statement so it cannot skip the leading "Direct ". A separate
    guard used to match the phrase "most likely paper's title" anywhere in a
    statement and suppress the commit. It rejected no heading the two
    requirements did not already reject, and instead suppressed the single most
    common way these systems phrase a real answer -- "The most likely paper title
    is **"<Title>"**" -- on 25 statements over 19 instances across the five runs,
    6 of those statements naming the gold paper. It is gone.
    """
    predicate = COMMIT_RE.search(statement)
    if not predicate:
        return None
    return _title_after(statement[predicate.end() :]) or _labelled_title(statement)


def extract_answer(text: str) -> SageAnswer:
    """Extract what a short-form output states about its answer.

    Two things are read out of the answer region.

    ``states_commit`` and ``states_decline`` record whether *any* statement names
    a paper and whether *any* declines. Nothing about them depends on reading
    order, which is why the rate metrics are built on them: on the five runs we
    have, resolving a hedge by first, last, prefer-commit or prefer-decline moves
    AgentDisCo's commit rate between 0.404 and 0.933, so a metric that depends on
    the choice is reporting the choice.

    ``kind``/``title`` are the single answer ``answer_only_match`` scores, and
    that does need one statement, so it takes the *first* statement that declines
    or commits. First, not best and not last: the prompt asks for one answer, so
    scanning further would let a system raise its score by writing more, which is
    the artifact these metrics exist to measure.
    """
    region = answer_region(text)
    statements = answer_statements(region)

    first: tuple[str, str, str | None] | None = None
    states_commit = False
    states_decline = False
    for statement in statements:
        if DECLINE_RE.search(statement):
            candidate: tuple[str, str, str | None] = ("decline", statement, None)
            states_decline = True
        else:
            title = committed_title(statement)
            if title is None:
                continue
            candidate = ("commit", statement, title)
            states_commit = True
        if first is None:
            first = candidate

    if first is not None:
        kind, statement, title = first
        return SageAnswer(
            kind,
            statement[:_MAX_RECORDED_STATEMENT_CHARS],
            title,
            states_commit,
            states_decline,
        )

    for index, statement in enumerate(statements):
        bare = _STANDALONE_TITLE.match(statement)
        if not bare:
            continue
        bold = bare.group("bold")
        value = (bold or bare.group("double")).strip()
        # A bolded heading with body text under it announces a section, not an
        # answer; a quoted line is always a title.
        if bold is not None and index + 1 < len(statements) and _is_bold_section_heading(value):
            continue
        return SageAnswer("commit", statement[:_MAX_RECORDED_STATEMENT_CHARS], value, True, False)

    return SageAnswer("unparsed", region[:_MAX_RECORDED_STATEMENT_CHARS], None, False, False)


async def answer_only_match(matcher: Matcher, gold: GoldPaper, output: str) -> float:
    """Return 1.0 iff the output's stated answer names the gold paper.

    The same normalized-title substring test :func:`exact_match` uses, applied to
    the extracted title instead of the whole output. Declines and unparsed
    outputs score 0.0: neither names a paper.
    """
    answer = extract_answer(output)
    if not answer.committed or not answer.title:
        return 0.0
    return 1.0 if await matcher.matched(gold, answer.title) else 0.0


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
class SageAnswerOnlyMatchScorer(Scorer):
    """Placeholder scorer; SAGE answer-only match is computed in score_responses."""

    name: str = "answer_only_match"
    score_key: str = "answer_only_match"

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get(self.score_key, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass(frozen=True)
class SageAnswerOnlyMatchMetric(Metric):
    """Mean normalized-title match against the stated answer only.

    Companion to ``exact_match``, not a replacement: ``exact_match`` asks whether
    the gold title appears anywhere in the output, this asks whether the system
    answered with it. Read it next to ``answer_unparsed_rate`` -- a system whose
    outputs never state an answer will score 0 here for that reason, not because
    its retrieval is worse.
    """

    name: str = "answer_only_match"
    scorer: type[Scorer] = SageAnswerOnlyMatchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(r.scores.get(self.name, 0.0) for r in responses) / len(responses)


@dataclass(frozen=True)
class SageCommitScorer(Scorer):
    """Placeholder scorer; SAGE commit detection is computed in score_responses."""

    name: str = "commit_rate"
    score_key: str = "sage_committed"

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get(self.score_key, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass(frozen=True)
class SageCommitRateMetric(Metric):
    """Fraction of instances naming a paper without also declining.

    Read with ``hedge_rate``. Together they are the whole of what the run
    supports: an output that both declines and names a candidate has no commit
    value, so ``commit_rate`` counts only unambiguous commits and ``hedge_rate``
    reports how many were set aside. Any tie-break rule for hedges yields a
    commit rate somewhere in ``[commit_rate, commit_rate + hedge_rate]``, and
    that band is why a single number was not reportable: on AgentDisCo it runs
    0.404 to 0.933, while no other system we have run has a band wider than 0.10.

    ``commit_rate``, ``decline_rate``, ``hedge_rate`` and
    ``answer_unparsed_rate`` partition the run and sum to 1.
    """

    name: str = "commit_rate"
    scorer: type[Scorer] = SageCommitScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(r.scores.get(self.name, 0.0) for r in responses) / len(responses)


@dataclass(frozen=True)
class SageDeclineScorer(Scorer):
    """Placeholder scorer; SAGE decline detection is computed in score_responses."""

    name: str = "decline_rate"
    score_key: str = "sage_declined"

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get(self.score_key, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass(frozen=True)
class SageDeclineRateMetric(Metric):
    """Fraction of instances on which the system declines and names no paper.

    Abstention, counted on its own terms. ``exact_match`` scores a decline and a
    wrong guess identically at zero, so without this a system cannot be credited
    for knowing that it did not find the paper.
    """

    name: str = "decline_rate"
    scorer: type[Scorer] = SageDeclineScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(r.scores.get(self.name, 0.0) for r in responses) / len(responses)


@dataclass(frozen=True)
class SageHedgeScorer(Scorer):
    """Placeholder scorer; SAGE hedge detection is computed in score_responses."""

    name: str = "hedge_rate"
    score_key: str = "sage_answer_hedged"

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get(self.score_key, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass(frozen=True)
class SageHedgeRateMetric(Metric):
    """Fraction of instances that both decline and name a paper.

    This is the second trust bound on the answer metrics, alongside
    ``answer_unparsed_rate``, and on the runs we have it is the larger one by two
    orders of magnitude: AgentDisCo's ``answer_unparsed_rate`` is 0.2% while more
    than half its outputs hedge. Without it exported, a reader sees a low unparsed
    rate next to a commit rate and concludes the commit rate is settled.

    It is also exactly the width of the band that ``commit_rate`` cannot resolve;
    see :class:`SageCommitRateMetric`.
    """

    name: str = "hedge_rate"
    scorer: type[Scorer] = SageHedgeScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(r.scores.get(self.name, 0.0) for r in responses) / len(responses)


@dataclass(frozen=True)
class SageAnswerUnparsedScorer(Scorer):
    """Placeholder scorer; SAGE parse failures are computed in score_responses."""

    name: str = "answer_unparsed_rate"
    score_key: str = "sage_answer_unparsed"

    def score(self, instance: Instance, output: LMOutput) -> float:
        value = (output.metadata or {}).get(self.score_key, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass(frozen=True)
class SageAnswerUnparsedRateMetric(Metric):
    """Fraction of instances whose output states no answer this task can read.

    One of two trust bounds on ``answer_only_match``, ``commit_rate`` and
    ``accuracy_given_commit`` -- the other, usually much larger, is
    ``hedge_rate``. A high value here means the system's outputs do not follow
    the prompt's "state one title or say no match" contract, and those three
    metrics should not be compared across systems without saying so.
    """

    name: str = "answer_unparsed_rate"
    scorer: type[Scorer] = SageAnswerUnparsedScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(r.scores.get(self.name, 0.0) for r in responses) / len(responses)


#: Below this many committed instances, ``accuracy_given_commit`` is a ratio of
#: small integers and is logged as such. It is not suppressed -- ``commit_count``
#: is exported next to it so the denominator is always visible -- but the warning
#: exists because the failure it guards against was real: Allyson's run committed
#: on 3 instances and scored 0.3333, landing tied with AgentDisCo's 0.3333 from
#: 309.
_MIN_COMMITS_FOR_ACCURACY = 30


@dataclass(frozen=True)
class SageAccuracyGivenCommitMetric(Metric):
    """``answer_only_match`` restricted to instances with an unambiguous commit.

    Separates declining from being wrong: ``exact_match`` scores both 0, so a
    system that guesses on every instance cannot be told apart from one that
    answers rarely and well.

    Numerator and denominator read the same text. The denominator is the set of
    outputs that name a paper and do not also decline -- ``commit_rate``'s
    numerator -- and the numerator asks whether the paper they named was the gold
    one. Conditioning ``exact_match`` on committing instead, as this metric first
    did, mixes two texts: it credits a system for "answering correctly" when its
    stated answer was wrong and the gold title merely turned up in its
    bibliography. Measured, that was 42% of AgentDisCo's numerator (43 of 103)
    and 100% of Allyson's (its one hit was a reference-list match) -- exactly the
    artifact these metrics exist to remove.

    Two things this metric cannot do, both of which need ``commit_count`` and
    ``commit_rate`` read alongside it:

    * It is not comparable across systems whose commit counts differ by an order
      of magnitude. ``commit_count`` is exported for that reason, and a run below
      :data:`_MIN_COMMITS_FOR_ACCURACY` commits is logged.
    * It is trivially maximised by abstaining. Every system we have run reaches
      1.000 by committing only on the instances it already gets right --
      AgentDisCo would do so at ``commit_rate`` 0.084, baseline at 0.089, dp1 at
      0.082 and Arman's at 0.052. Moving the numerator to ``answer_only_match``
      does not change that and cannot: any metric conditioned on answering is
      gameable by answering less, which is why ``commit_rate`` and
      ``commit_count`` are exported beside it rather than folded into it, and why
      ``exact_match`` stays the primary metric.

    Returns 0.0 when nothing was committed to, in which case ``commit_count`` is
    0 and the value carries no information.
    """

    name: str = "accuracy_given_commit"
    scorer: type[Scorer] = SageAnswerOnlyMatchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        committed = [r for r in responses if r.scores.get("commit_rate", 0.0) > 0.0]
        if responses and len(committed) < _MIN_COMMITS_FOR_ACCURACY:
            logger.warning(
                "SAGE accuracy_given_commit is computed over %d committed instance(s) of "
                "%d, below the %d needed to read it as a rate; report it with "
                "commit_count or not at all.",
                len(committed),
                len(responses),
                _MIN_COMMITS_FOR_ACCURACY,
            )
        if not committed:
            return 0.0
        return sum(r.scores.get("answer_only_match", 0.0) for r in committed) / len(committed)

    def compute_instance(self, response: Response) -> float | None:
        """No per-instance value: this metric is conditional, not an average."""
        return None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class SageCommitCountMetric(Metric):
    """Number of instances in ``accuracy_given_commit``'s denominator.

    A count, not a rate, and exported for one reason: ``accuracy_given_commit``
    is a ratio whose denominator varies by two orders of magnitude between the
    systems it is used to compare, and without the denominator beside it 0.3333
    from 3 instances reads the same as 0.3333 from 309.
    """

    name: str = "commit_count"
    scorer: type[Scorer] = SageCommitScorer

    def compute(self, responses: Sequence[Response]) -> float:
        return float(sum(1 for r in responses if r.scores.get("commit_rate", 0.0) > 0.0))

    def compute_instance(self, response: Response) -> float | None:
        """No per-instance value: this metric is a corpus-level count."""
        return None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


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
    metrics = (
        SageExactMatchMetric(),
        SageAnswerOnlyMatchMetric(),
        SageCommitRateMetric(),
        SageDeclineRateMetric(),
        SageHedgeRateMetric(),
        SageAnswerUnparsedRateMetric(),
        SageAccuracyGivenCommitMetric(),
        SageCommitCountMetric(),
    )
    # exact_match stays primary: it is SAGE's published metric and the only one
    # comparable with numbers reported outside this harness.
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
                    or doc.get("query_id")
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
        missing_trajectory = 0
        for response in responses:
            if response.trajectory is None:
                missing_trajectory += 1

            gold = response.instance.metadata["gold"]
            scores: list[float] = []
            answer_scores: list[float] = []
            commit_scores: list[float] = []
            decline_scores: list[float] = []
            hedge_scores: list[float] = []
            unparsed_scores: list[float] = []
            for output in response.outputs:
                em = await exact_match(self.matcher, gold, output.text)
                scores.append(em)
                output.metadata = output.metadata or {}
                output.metadata["sage_matched"] = bool(em)
                output.metadata["exact_match"] = em
                _store_output_score(output, scorer_name="exact_match", score=em)

                # The stated answer, scored alongside -- never replacing -- SAGE's
                # own whole-output match above.
                answer = extract_answer(output.text)
                aom = await answer_only_match(self.matcher, gold, output.text)
                answer_scores.append(aom)
                # The rate metrics read the tie-break-invariant partition, not
                # the first-in-reading-order ``kind`` that answer_only_match uses.
                commit_scores.append(1.0 if answer.commits_without_hedging else 0.0)
                decline_scores.append(1.0 if answer.declines_without_hedging else 0.0)
                hedge_scores.append(1.0 if answer.hedged else 0.0)
                unparsed_scores.append(1.0 if answer.unparsed else 0.0)
                output.metadata["answer_only_match"] = aom
                output.metadata["sage_answer_kind"] = answer.kind
                output.metadata["sage_answer_title"] = answer.title
                output.metadata["sage_answer_statement"] = answer.statement
                output.metadata["sage_answer_hedged"] = answer.hedged
                output.metadata["sage_states_commit"] = answer.states_commit
                output.metadata["sage_states_decline"] = answer.states_decline
                output.metadata["sage_committed"] = 1.0 if answer.commits_without_hedging else 0.0
                output.metadata["sage_declined"] = 1.0 if answer.declines_without_hedging else 0.0
                output.metadata["sage_answer_unparsed"] = 1.0 if answer.unparsed else 0.0
                _store_output_score(output, scorer_name="answer_only_match", score=aom)

            response.scores["exact_match"] = self._aggregate_output_scores(dict(enumerate(scores)))
            response.scores["answer_only_match"] = self._aggregate_output_scores(
                dict(enumerate(answer_scores))
            )
            response.scores["commit_rate"] = self._aggregate_output_scores(
                dict(enumerate(commit_scores))
            )
            response.scores["decline_rate"] = self._aggregate_output_scores(
                dict(enumerate(decline_scores))
            )
            response.scores["hedge_rate"] = self._aggregate_output_scores(
                dict(enumerate(hedge_scores))
            )
            response.scores["answer_unparsed_rate"] = self._aggregate_output_scores(
                dict(enumerate(unparsed_scores))
            )

        if missing_trajectory:
            logger.warning(
                "SAGE short-form scored %d/%d responses with no trajectory; scores likely "
                "reflect parametric memory, not agentic retrieval. Run through a "
                "tool-providing agentic harness such as paper_search_agent.",
                missing_trajectory,
                len(responses),
            )
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
                "case_id": doc.get("case_id") or doc.get("query_id") or f"sage_open_ended_{index}",
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
        missing_trajectory = 0
        for response in responses:
            if response.trajectory is None:
                missing_trajectory += 1

            scores: list[float] = []
            for output in response.outputs:
                wr = await weighted_recall(
                    self.matcher, response.instance.metadata["golds"], output.text
                )
                scores.append(wr)
                output.metadata = output.metadata or {}
                output.metadata["weighted_recall"] = wr
                _store_output_score(output, scorer_name="weighted_recall", score=wr)

            response.scores["weighted_recall"] = self._aggregate_output_scores(
                dict(enumerate(scores))
            )

        if missing_trajectory:
            logger.warning(
                "SAGE open-ended scored %d/%d responses with no trajectory; scores likely "
                "reflect parametric memory, not agentic retrieval. Run through a "
                "tool-providing agentic harness such as paper_search_agent.",
                missing_trajectory,
                len(responses),
            )
        return responses
