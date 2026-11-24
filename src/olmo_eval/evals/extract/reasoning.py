"""Extraction functions for handling thinking/reasoning traces"""

import re
from collections.abc import Mapping
from typing import Any, Literal

from olmo_eval.common.types import PROMPT_OPENS_THINK_KEY, THINK_CLOSED_KEY

_THINK_START = "<think>"
_THINK_END = "</think>"

_ANSWER_OPEN = re.compile(r"(?ms)^\s*<answer>\s*")
_ANSWER_CLOSE = re.compile(r"(?ms)</answer>\s*$")


def _after_last_think(text: str) -> str:
    """Text after the last ``</think>``, with ``<answer>`` tags removed.

    Equivalent to the reference harness's ``r1_style`` regexes
    (``re.sub("(?ms).*</think>", "", text)`` then the tag strips). The trace
    is dropped in linear time, where the greedy regex is quadratic in the text
    after the last ``</think>``. Each tag strip is skipped when its literal tag
    is absent. When ``<answer>`` is present, a long run of blank lines that
    does not lead to it can still make the opening-tag strip quadratic.
    """
    answer = text.rpartition(_THINK_END)[2] if _THINK_END in text else text
    # A multiline whitespace prefix can rescan long runs of blank lines when
    # the tag is absent. Literal checks make those negative scans linear.
    if "<answer>" in answer:
        answer = _ANSWER_OPEN.sub("", answer)
    if "</answer>" in answer:
        answer = _ANSWER_CLOSE.sub("", answer)
    return answer


def extract_think_answer(text: str) -> str | None:
    """Deepseek-R1 style answer: drop the ``<think>`` trace, keep the rest.

    Text without a closing tag is returned unchanged (the trace is treated as
    the answer), matching the reference implementation.
    """
    return _after_last_think(text)


def extract_think_answer_only(text: str) -> str | None:
    """Like :func:`extract_think_answer`, but an empty string when no trace closed."""
    if _THINK_END not in text:
        return ""
    return _after_last_think(text)


#: Whether the reasoning trace of a generation finished. ``closed`` covers text with no
#: trace at all: either way the text after any ``</think>`` is the answer. ``unclosed``
#: text is a trace that never finished and holds no answer. ``unknown`` means the
#: generation started inside a trace the prompt opened, or may have, and nothing shows
#: whether that trace closed.
TraceState = Literal["closed", "unclosed", "unknown"]


def trace_state(
    text: str,
    metadata: Mapping[str, Any] | None = None,
    *,
    prefilled: bool | None = None,
) -> TraceState:
    """Say whether the reasoning trace in ``text`` finished, from tags and evidence.

    Tags decide first: a ``<think>`` after the last ``</think>`` (or with none) is
    unclosed, and any other text holding ``</think>`` is closed. Text with neither tag
    depends on whether the generation started inside a trace:

    - ``prefilled`` given: True means it did, so the text is an unclosed trace; False
      means it did not, so the text is an answer.
    - ``prefilled`` left as None: the provider's record in ``metadata`` decides. Without
      ``PROMPT_OPENS_THINK_KEY`` the text is an answer, as it always was. With it set
      to True, ``THINK_CLOSED_KEY`` gives the state, and its absence is ``unknown``.
      With it set to None (the provider could not tell what the prompt ends with), the
      state is ``unknown``.

    Nothing is inferred from why generation stopped: a trace can end at the end-of-turn
    token without closing, and a closed trace can be followed by an answer that runs to
    the token limit.
    """
    closed = _THINK_END in text
    suffix = text.rpartition(_THINK_END)[2] if closed else text
    if _THINK_START in suffix:
        return "unclosed"
    if closed:
        return "closed"
    if prefilled is not None:
        return "unclosed" if prefilled else "closed"
    recorded = metadata or {}
    if PROMPT_OPENS_THINK_KEY not in recorded or recorded[PROMPT_OPENS_THINK_KEY] is False:
        return "closed"
    if recorded[PROMPT_OPENS_THINK_KEY] is None:
        return "unknown"
    evidence = recorded.get(THINK_CLOSED_KEY)
    if evidence is None:
        return "unknown"
    return "closed" if evidence else "unclosed"


def trace_unfinished(
    text: str,
    metadata: Mapping[str, Any] | None = None,
    *,
    prefilled: bool | None = None,
) -> bool:
    """Whether ``text`` is known to be a reasoning trace that never closed.

    False for an ``unknown`` state as well as a closed one; callers that must not
    score an unknown trace as an answer use :func:`trace_state`.
    """
    return trace_state(text, metadata, prefilled=prefilled) == "unclosed"


def visible_answer(
    text: str,
    metadata: Mapping[str, Any] | None = None,
    *,
    prefilled: bool | None = None,
) -> str:
    """The text after the last closed reasoning trace, or "" when the trace never closed.

    Text with no trace is returned whole, and so is text whose state is unknown.
    Nothing is trimmed. See :func:`trace_state` for ``metadata`` and ``prefilled``.
    """
    if trace_unfinished(text, metadata, prefilled=prefilled):
        return ""
    return text.rpartition(_THINK_END)[2] if _THINK_END in text else text


# OpenAI Harmony's documented token names, as in the gpt-oss tokenizer.
# https://github.com/openai/harmony/blob/main/docs/format.md
_HARMONY_HEADER = re.compile(
    r"(?P<start><\|start\|>assistant(?: to=[^\s<]+)?)?"
    r"<\|channel\|>(?P<channel>\w+)(?: to=[^\s<]+)?"
    r"<\|message\|>"
)
_HARMONY_END = re.compile(r"<\|(?:end|return|endoftext|call|start)\|>")


def extract_harmony_answer(text: str) -> str:
    """Keep a Harmony final-channel body, preserving its whitespace.

    Plain responses already parsed by a provider pass through. A recognized
    Harmony transcript with no final channel has no answer. Special token
    delimiters must be preserved by the provider's decoder.
    """
    first = _HARMONY_HEADER.match(text)
    if first is None:
        if text.startswith(("<|channel|>", "<|start|>assistant")):
            return ""
        return text
    for header in _HARMONY_HEADER.finditer(text):
        if header.start() and header.group("start") is None:
            continue
        if header.group("channel") == "final" and " to=" not in header.group():
            end = _HARMONY_END.search(text, header.end())
            return text[header.end() : end.start() if end else len(text)]
    return ""
