"""A misspelled tool name must not end the run, and must not be repaired into a different request.

Nine of two hundred instances in a DeepSeek smoke ended as the string "[Tool error: Tool
semantic_scholarly_snippet_search not found in agent openai_agents]" -- the SDK raises on an
unknown name and the scaffold makes that the final answer. Every one was a misspelling of the one
tool the harness offers.

The repair has to hold two lines at once, so both are pinned here: every observed misspelling gets
routed, and a name the model invented for a capability that does not exist does not. The observed
typos score 0.873-0.984 against the real name and the invented ones 0.108-0.390, so the cutoff has
room on both sides -- but the margin is the thing that could rot, which is why the values are
asserted rather than described.
"""

from __future__ import annotations

import difflib
import logging
import types

import pytest

from olmo_eval.harness.scaffolds.openai_agents import (
    TOOL_NAME_REPAIR_CUTOFF,
    _repair_tool_names,
)

REAL = "semantic_scholar_snippet_search"
TOOLS = [REAL]
LOGGER = logging.getLogger(__name__)

MISSPELLINGS = [
    "semantic_scholarly_snippet_search",
    "semantic_schol_snippet_search",
    "semantic_scholarlar_snippet_search",
    "semantic_semantic_scholar_snippet_search",
    "semantic_schollar_snippet_search",
    "semantic_schololar_snippet_search",
]

INVENTED = ["web_search", "python", "browse", "fetch_page", "calculator"]


@pytest.mark.parametrize("name", MISSPELLINGS)
def test_a_misspelling_is_routed_to_the_tool_it_meant(name):
    item = types.SimpleNamespace(name=name)
    assert _repair_tool_names([item], TOOLS, LOGGER) == 1
    assert item.name == REAL


@pytest.mark.parametrize("name", INVENTED)
def test_an_invented_tool_is_left_to_fail(name):
    """Routing these would answer a question nobody asked and hide the hallucination."""
    item = types.SimpleNamespace(name=name)
    assert _repair_tool_names([item], TOOLS, LOGGER) == 0
    assert item.name == name


def test_the_cutoff_sits_in_the_gap_between_the_two_populations():
    worst_typo = min(difflib.SequenceMatcher(None, n, REAL).ratio() for n in MISSPELLINGS)
    best_invented = max(difflib.SequenceMatcher(None, n, REAL).ratio() for n in INVENTED)
    assert best_invented < TOOL_NAME_REPAIR_CUTOFF < worst_typo


def test_a_correct_name_is_neither_touched_nor_counted():
    item = types.SimpleNamespace(name=REAL)
    assert _repair_tool_names([item], TOOLS, LOGGER) == 0
    assert item.name == REAL


def test_output_items_that_are_not_tool_calls_pass_through():
    item = types.SimpleNamespace(content="an ordinary assistant message")
    assert _repair_tool_names([item], TOOLS, LOGGER) == 0


def test_no_tools_configured_is_not_an_error():
    item = types.SimpleNamespace(name="anything")
    assert _repair_tool_names([item], [], LOGGER) == 0
    assert item.name == "anything"
