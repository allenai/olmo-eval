"""Tests for the reasoning-trace extractors."""

import re
import subprocess
import sys
import time

import pytest

from olmo_eval.common.types import PROMPT_OPENS_THINK_KEY, THINK_CLOSED_KEY
from olmo_eval.evals.extract import (
    extract_think_answer,
    extract_think_answer_only,
    trace_state,
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
from olmo_eval.evals.extract import extract_think_answer, extract_think_answer_only
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
_OPEN = {**_PREFILLED, THINK_CLOSED_KEY: False}
_CLOSED = {**_PREFILLED, THINK_CLOSED_KEY: True}


@pytest.mark.parametrize(
    ("text", "metadata", "state"),
    [
        # Tags alone decide when the provider recorded nothing, as before.
        ("<think>still thinking", None, "unclosed"),
        ("<think>a</think> 4 <think>again", None, "unclosed"),
        ("<think>a</think> 4", None, "closed"),
        ("step 1, step 2", None, "closed"),
        ("step 1, step 2", {"finish_reason": "length"}, "closed"),
        ("step 1, step 2", {PROMPT_OPENS_THINK_KEY: False}, "closed"),
        # The prompt opened the trace and the provider has evidence of how it ended.
        ("step 1, step 2", _OPEN, "unclosed"),
        ("", _OPEN, "unclosed"),
        ("the answer is 4", _CLOSED, "closed"),
        # Tags in the text outrank the record.
        ("step 1</think> 4", _OPEN, "closed"),
        ("step 1</think> 4 <think>again", _CLOSED, "unclosed"),
        # No evidence: unknown, whatever stopped the generation.
        ("step 1, step 2", {**_PREFILLED, "finish_reason": "length"}, "unknown"),
        ("the answer is 4", {**_PREFILLED, "finish_reason": "stop"}, "unknown"),
        # The provider could not tell what the prompt ends with.
        ("the answer is 4", {PROMPT_OPENS_THINK_KEY: None}, "unknown"),
        ("step 1</think> 4", {PROMPT_OPENS_THINK_KEY: None}, "closed"),
    ],
)
def test_trace_state_reads_tags_and_the_provider_record(text, metadata, state):
    assert trace_state(text, metadata) == state
    assert trace_unfinished(text, metadata) is (state == "unclosed")


@pytest.mark.parametrize("finish_reason", ["stop", "length", None])
def test_the_finish_reason_is_never_evidence(finish_reason):
    # An unclosed trace can end at end of turn, and a closed one can be followed by an
    # answer that runs to the token limit.
    assert trace_state("step 1", {**_OPEN, "finish_reason": finish_reason}) == "unclosed"
    assert trace_state("4 beca", {**_CLOSED, "finish_reason": finish_reason}) == "closed"
    assert trace_state("4", {**_PREFILLED, "finish_reason": finish_reason}) == "unknown"


def test_an_explicit_prefilled_setting_overrides_the_provider_record():
    assert trace_state("step 1", None, prefilled=True) == "unclosed"
    assert trace_state("step 1", _PREFILLED, prefilled=False) == "closed"
    # A trace the model opened itself is unclosed whatever the setting.
    assert trace_state("<think>step 1", None, prefilled=False) == "unclosed"


def test_visible_answer_keeps_the_text_after_a_closed_trace_untrimmed():
    assert visible_answer("<think>plan</think>\n\n 4 \n") == "\n\n 4 \n"
    assert visible_answer("plan</think>4", _OPEN) == "4"
    assert visible_answer("no trace at all") == "no trace at all"
    assert visible_answer("<think>unterminated") == ""
    assert visible_answer("step 1, step 2", _OPEN) == ""
    assert visible_answer("step 1, step 2") == "step 1, step 2"
    # Unknown text is returned whole; callers that must not score it check the state.
    assert visible_answer("step 1, step 2", _PREFILLED) == "step 1, step 2"
