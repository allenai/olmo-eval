"""Tests for HardReasoning task logic."""

from olmo_eval.evals.tasks.hard_reasoning import _extract_last_complete_json


class TestExtractLastCompleteJson:
    def test_simple_json(self):
        assert _extract_last_complete_json('{"a": 1}') == {"a": 1}

    def test_json_after_text(self):
        assert _extract_last_complete_json('Some text {"key": "value"}') == {"key": "value"}

    def test_returns_last_json(self):
        result = _extract_last_complete_json('{"first": 1} some text {"second": 2}')
        assert result == {"second": 2}

    def test_nested_json(self):
        assert _extract_last_complete_json('{"outer": {"inner": 42}}') == {"outer": {"inner": 42}}

    def test_json_with_newlines(self):
        assert _extract_last_complete_json('{"a":\n1}') == {"a": 1}

    def test_no_json_returns_none(self):
        assert _extract_last_complete_json("no json here") is None

    def test_incomplete_json_returns_none(self):
        assert _extract_last_complete_json('{"incomplete": ') is None

    def test_empty_string_returns_none(self):
        assert _extract_last_complete_json("") is None
