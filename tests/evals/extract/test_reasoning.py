"""Tests for the reasoning-trace extractors."""

import re
import subprocess
import sys
import time

import pytest

from olmo_eval.common.types import (
    PROMPT_OPENS_THINK_KEY,
    REASONING_SEPARATED_KEY,
    THINK_CLOSE_HIDDEN_KEY,
)
from olmo_eval.evals.extract import (
    extract_think_answer,
    extract_think_answer_only,
    trace_unfinished,
    visible_answer,
)

# The reference harness's r1_style regexes, kept here as the oracle.


def _reference(text: str) -> str:
    answer = re.sub("(?ms).*</think>", "", text)
    answer = re.sub("(?ms)^\\s*<answer>\\s*", "", answer)
    return re.sub("(?ms)</answer>\\s*$", "", answer)


@pytest.mark.parametrize(
    "text",
    [
        "<think>plan</think>\n\nFirst line.\n\nSecond line.",
        "<think>a</think> draft <think>b</think> 4",
        "<think>unterminated",
        "no trace at all",
        "",
        "<think>plan</think>\n<answer>4</answer>",
        "<think>plan</think>  <answer>\n4\n</answer>\n",
        "<think>x</think>\n\n<<title>>\n\nbody with `</answer>` mid-text </answer>",
    ],
)
def test_matches_reference_regexes(text: str) -> None:
    assert extract_think_answer(text) == _reference(text)


def test_keeps_text_after_trace_verbatim() -> None:
    assert extract_think_answer("<think>p</think>\n\n  4 ") == "\n\n  4 "


def test_only_variant_is_empty_without_closed_trace() -> None:
    assert extract_think_answer_only("<think>still going") == ""
    assert extract_think_answer_only("plain") == ""
    assert extract_think_answer_only("<think>p</think>\n4") == "\n4"


def test_linear_on_long_answers() -> None:
    text = "<think>" + "x" * 1000 + "</think>" + ("answer text. " * 8000)
    start = time.perf_counter()
    extract_think_answer(text)
    assert time.perf_counter() - start < 0.05


def test_long_blank_lines_without_answer_tags_complete() -> None:
    # Multiline whitespace matching previously revisited each blank line.
    # Bound a regression without making ordinary test runs wait indefinitely.
    code = """
from olmo_eval.common.types import (
    PROMPT_OPENS_THINK_KEY,
    REASONING_SEPARATED_KEY,
    THINK_CLOSE_HIDDEN_KEY,
)
from olmo_eval.evals.extract import (
    extract_think_answer,
    extract_think_answer_only,
    trace_unfinished,
    visible_answer,
)
import time
text = '\\n' * 100000
started = time.perf_counter()
assert extract_think_answer(text) == text
assert extract_think_answer('<think>x</think>' + text) == text
assert extract_think_answer_only('<think>x</think>' + text) == text
assert time.perf_counter() - started < 2
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


_PREFILLED = {PROMPT_OPENS_THINK_KEY: True}


@pytest.mark.parametrize(
    ("text", "metadata", "unfinished"),
    [
        # Tags alone decide when the provider recorded nothing, as before.
        ("<think>still thinking", None, True),
        ("<think>a</think> 4 <think>again", None, True),
        ("<think>a</think> 4", None, False),
        ("step 1, step 2", None, False),
        ("step 1, step 2", {"finish_reason": "length"}, False),
        # The prompt opened the trace: no closing tag means it never closed.
        ("step 1, step 2", _PREFILLED, True),
        ("", _PREFILLED, True),
        ("step 1</think> 4", _PREFILLED, False),
        ("step 1</think> 4 <think>again", _PREFILLED, True),
        # A server that returned the trace apart left only the answer in the text.
        ("4", {**_PREFILLED, REASONING_SEPARATED_KEY: True, "finish_reason": "length"}, False),
        # A closed trace may leave no tag: only a length stop is unfinished.
        ("4", {**_PREFILLED, THINK_CLOSE_HIDDEN_KEY: True, "finish_reason": "stop"}, False),
        ("step 1", {**_PREFILLED, THINK_CLOSE_HIDDEN_KEY: True, "finish_reason": "length"}, True),
    ],
)
def test_trace_unfinished_reads_tags_and_the_provider_record(text, metadata, unfinished):
    assert trace_unfinished(text, metadata) is unfinished


def test_an_explicit_prefilled_setting_overrides_the_provider_record():
    assert trace_unfinished("step 1", None, prefilled=True) is True
    assert trace_unfinished("step 1", _PREFILLED, prefilled=False) is False
    # A trace the model opened itself is unfinished whatever the setting.
    assert trace_unfinished("<think>step 1", None, prefilled=False) is True


def test_visible_answer_keeps_the_text_after_a_closed_trace_untrimmed():
    assert visible_answer("<think>plan</think>\n\n 4 \n") == "\n\n 4 \n"
    assert visible_answer("plan</think>4", _PREFILLED) == "4"
    assert visible_answer("no trace at all") == "no trace at all"
    assert visible_answer("<think>unterminated") == ""
    assert visible_answer("step 1, step 2", _PREFILLED) == ""
    assert visible_answer("step 1, step 2") == "step 1, step 2"
