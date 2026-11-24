"""Extraction functions for handling thinking/reasoning traces"""

import re
from collections.abc import Mapping
from typing import Any

from olmo_eval.common.types import (
    PROMPT_OPENS_THINK_KEY,
    REASONING_SEPARATED_KEY,
    THINK_CLOSE_HIDDEN_KEY,
)

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


def trace_unfinished(
    text: str,
    metadata: Mapping[str, Any] | None = None,
    *,
    prefilled: bool | None = None,
) -> bool:
    """Whether ``text`` is a reasoning trace that never closed, so it holds no answer.

    A ``<think>`` after the last ``</think>`` (or with none) is always unfinished. Text
    with neither tag is unfinished only when the generation started inside a trace.
    ``prefilled`` says whether it did; left as None, the answer comes from what the
    provider recorded in ``metadata``:

    - without ``PROMPT_OPENS_THINK_KEY`` the text is taken as an answer, as it always
      was;
    - with ``REASONING_SEPARATED_KEY`` the server already removed a closed trace;
    - with ``THINK_CLOSE_HIDDEN_KEY`` a closed trace may leave no ``</think>``, so only
      a generation that stopped at the token limit is unfinished;
    - otherwise a missing ``</think>`` means the trace never closed.
    """
    closed = _THINK_END in text
    suffix = text.rpartition(_THINK_END)[2] if closed else text
    if _THINK_START in suffix:
        return True
    if closed:
        return False
    if prefilled is not None:
        return prefilled
    recorded = metadata or {}
    if not recorded.get(PROMPT_OPENS_THINK_KEY) or recorded.get(REASONING_SEPARATED_KEY):
        return False
    if recorded.get(THINK_CLOSE_HIDDEN_KEY):
        return recorded.get("finish_reason") == "length"
    return True


def visible_answer(
    text: str,
    metadata: Mapping[str, Any] | None = None,
    *,
    prefilled: bool | None = None,
) -> str:
    """The text after the last closed reasoning trace, or "" when the trace never closed.

    Text with no trace is returned whole. Nothing is trimmed. See
    :func:`trace_unfinished` for ``metadata`` and ``prefilled``.
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
