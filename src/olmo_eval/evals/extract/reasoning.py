"""Extraction functions for handling thinking/reasoning traces"""

import re

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


# OpenAI Harmony's documented token names and gpt-oss tokenizer spellings.
# https://github.com/openai/harmony/blob/main/docs/format.md
_HARMONY_HEADER = re.compile(
    r"(?P<start><\|(?:start|im_start)\|>assistant(?: to=[^\s<]+)?)?"
    r"<\|(?:channel|meta_sep)\|>(?P<channel>\w+)(?: to=[^\s<]+)?"
    r"<\|(?:message|im_sep)\|>"
)
_HARMONY_END = re.compile(r"<\|(?:end|im_end|return|fim_suffix|call|ghissue|start|im_start)\|>")


def extract_harmony_answer(text: str) -> str:
    """Keep a Harmony final-channel body, preserving its whitespace.

    Plain responses already parsed by a provider pass through. A recognized
    Harmony transcript with no final channel has no answer. Special token
    delimiters must be preserved by the provider's decoder.
    """
    first = _HARMONY_HEADER.match(text)
    if first is None:
        if text.startswith(
            ("<|channel|>", "<|meta_sep|>", "<|start|>assistant", "<|im_start|>assistant")
        ):
            return ""
        return text
    for header in _HARMONY_HEADER.finditer(text):
        if header.start() and header.group("start") is None:
            continue
        if header.group("channel") == "final" and " to=" not in header.group():
            end = _HARMONY_END.search(text, header.end())
            return text[header.end() : end.start() if end else len(text)]
    return ""
