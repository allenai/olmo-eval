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

#: Upper bound on a single answer statement. A statement longer than this is
#: prose, not an answer, and matching a title inside it would reintroduce the
#: very volume artifact these metrics exist to expose.
_MAX_STATEMENT_CHARS = 600


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
    following line. Statements are capped at :data:`_MAX_STATEMENT_CHARS`.
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
                statements.append(piece[:_MAX_STATEMENT_CHARS])
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

#: Section headings that quote the prompt back ("Direct Answer: Identification
#: of the Most Likely Paper or Explicit Statement of No Match") announce an
#: answer without giving one.
_PROMPT_ECHO = re.compile(
    r"(?i)(?:most\s+likely\s+paper'?s?\s+title"
    r"|explicit(?:ly)?\s+statements?\s+of\s+no\s+match"
    r"|or\s+explicitly\s+say\s+no\s+match"
    r"|state\s+the\s+most\s+likely)"
)

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


@dataclass(frozen=True, slots=True)
class SageAnswer:
    """The single answer a short-form output states, per the prompt's contract.

    ``kind`` is one of:

    ``"commit"``
        The output names a paper. ``title`` holds it.
    ``"decline"``
        The output says no match was found. ``title`` is ``None``.
    ``"unparsed"``
        Neither. The output did not follow the contract (typically it is a
        literature survey that never answers), or it worded its answer in a way
        this extractor does not recognise. These are reported, never guessed at.

    ``hedged`` marks outputs that contain both a decline and a named candidate.
    The classification is the first of the two in reading order; ``hedged``
    exists so the size of that ambiguous population can be reported.
    """

    kind: str
    statement: str
    title: str | None
    hedged: bool

    @property
    def committed(self) -> bool:
        return self.kind == "commit"

    @property
    def declined(self) -> bool:
        return self.kind == "decline"

    @property
    def unparsed(self) -> bool:
        return self.kind == "unparsed"


def _title_after(tail: str) -> str | None:
    """Return the first delimited title in ``tail``."""
    found = _TITLE_OBJECT.search(tail)
    if not found:
        return None
    value = next(group for group in found.groups() if group)
    return value.strip().strip("*_ \t").strip("\"'“”") or None


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
    """
    predicate = COMMIT_RE.search(statement)
    if not predicate or _PROMPT_ECHO.search(statement):
        return None
    return _title_after(statement[predicate.end() :]) or _labelled_title(statement)


def extract_answer(text: str) -> SageAnswer:
    """Extract the single answer a short-form output states.

    The answer is the *first* statement in the answer region that either
    declines or commits to a named title. First, not best and not last: the
    prompt asks for one answer, so scanning further would let a system raise its
    score by writing more, which is the artifact these metrics exist to measure.
    """
    region = answer_region(text)
    statements = answer_statements(region)

    first: SageAnswer | None = None
    kinds: set[str] = set()
    for statement in statements:
        if DECLINE_RE.search(statement):
            candidate = SageAnswer("decline", statement, None, False)
        else:
            title = committed_title(statement)
            if title is None:
                continue
            candidate = SageAnswer("commit", statement, title, False)
        kinds.add(candidate.kind)
        if first is None:
            first = candidate

    if first is not None:
        return SageAnswer(first.kind, first.statement, first.title, len(kinds) > 1)

    for statement in statements:
        bare = _STANDALONE_TITLE.match(statement)
        if bare:
            title = (bare.group("bold") or bare.group("double")).strip()
            return SageAnswer("commit", statement, title, False)

    return SageAnswer("unparsed", region[:_MAX_STATEMENT_CHARS], None, False)


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
    """Fraction of instances on which the system names a paper.

    The complement is split between explicit declines and outputs that state no
    answer at all; see ``answer_unparsed_rate``.
    """

    name: str = "commit_rate"
    scorer: type[Scorer] = SageCommitScorer

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

    This is the trust bound on ``answer_only_match``, ``commit_rate`` and
    ``accuracy_given_commit``. A high value means the system's outputs do not
    follow the prompt's "state one title or say no match" contract, and those
    three metrics should not be compared across systems without saying so.
    """

    name: str = "answer_unparsed_rate"
    scorer: type[Scorer] = SageAnswerUnparsedScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(r.scores.get(self.name, 0.0) for r in responses) / len(responses)


@dataclass(frozen=True)
class SageAccuracyGivenCommitMetric(Metric):
    """``exact_match`` restricted to instances where the system named a paper.

    Separates declining from being wrong: ``exact_match`` scores both 0, so a
    system that guesses on every instance cannot be told apart from one that
    answers rarely and well. Returns 0.0 when nothing was committed to.

    The numerator is ``exact_match``, not ``answer_only_match``, so that this
    reads as "of the times it answered, how often was the gold paper in the
    output" on exactly SAGE's own matching rule.
    """

    name: str = "accuracy_given_commit"
    scorer: type[Scorer] = SageExactMatchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        committed = [r for r in responses if r.scores.get("commit_rate", 0.0) > 0.0]
        if not committed:
            return 0.0
        return sum(r.scores.get("exact_match", 0.0) for r in committed) / len(committed)

    def compute_instance(self, response: Response) -> float | None:
        """No per-instance value: this metric is conditional, not an average."""
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
        SageAccuracyGivenCommitMetric(),
        SageAnswerUnparsedRateMetric(),
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
                commit_scores.append(1.0 if answer.committed else 0.0)
                unparsed_scores.append(1.0 if answer.unparsed else 0.0)
                output.metadata["answer_only_match"] = aom
                output.metadata["sage_answer_kind"] = answer.kind
                output.metadata["sage_answer_title"] = answer.title
                output.metadata["sage_answer_statement"] = answer.statement
                output.metadata["sage_answer_hedged"] = answer.hedged
                output.metadata["sage_committed"] = 1.0 if answer.committed else 0.0
                output.metadata["sage_answer_unparsed"] = 1.0 if answer.unparsed else 0.0
                _store_output_score(output, scorer_name="answer_only_match", score=aom)

            response.scores["exact_match"] = self._aggregate_output_scores(dict(enumerate(scores)))
            response.scores["answer_only_match"] = self._aggregate_output_scores(
                dict(enumerate(answer_scores))
            )
            response.scores["commit_rate"] = self._aggregate_output_scores(
                dict(enumerate(commit_scores))
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
