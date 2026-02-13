"""Tests for olmo_eval.core.tool_scorers module."""

from olmo_eval.common.scorers import (
    ToolArgumentScorer,
    ToolCallScorer,
    ToolSequenceScorer,
)
from olmo_eval.common.types import Instance, LMOutput, ToolCall


class TestToolCallScorer:
    """Tests for ToolCallScorer."""

    def test_correct_tool_called(self):
        """Test score when correct tool is called."""
        scorer = ToolCallScorer()
        instance = Instance(
            question="What's the weather?",
            expected_tool_calls=({"name": "get_weather", "arguments": {}},),
        )
        output = LMOutput(text="", tool_calls=[ToolCall.create("1", "get_weather", {})])

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_wrong_tool_called(self):
        """Test score when wrong tool is called."""
        scorer = ToolCallScorer()
        instance = Instance(
            question="What's the weather?",
            expected_tool_calls=({"name": "get_weather", "arguments": {}},),
        )
        output = LMOutput(text="", tool_calls=[ToolCall.create("1", "search", {})])

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_no_tool_called(self):
        """Test score when no tool is called."""
        scorer = ToolCallScorer()
        instance = Instance(
            question="What's the weather?",
            expected_tool_calls=({"name": "get_weather", "arguments": {}},),
        )
        output = LMOutput(text="I don't know")

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_multiple_expected_one_match(self):
        """Test with multiple expected tools, one matches."""
        scorer = ToolCallScorer()
        instance = Instance(
            question="Complex query",
            expected_tool_calls=(
                {"name": "search", "arguments": {}},
                {"name": "calculate", "arguments": {}},
            ),
        )
        output = LMOutput(text="", tool_calls=[ToolCall.create("1", "search", {})])

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_openai_format_expected(self):
        """Test with OpenAI format expected calls."""
        scorer = ToolCallScorer()
        instance = Instance(
            question="Test",
            expected_tool_calls=({"function": {"name": "test_func", "arguments": "{}"}},),
        )
        output = LMOutput(text="", tool_calls=[ToolCall.create("1", "test_func", {})])

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_scorer_name(self):
        """Test scorer name."""
        scorer = ToolCallScorer()
        assert scorer.name == "tool_call"


class TestToolArgumentScorer:
    """Tests for ToolArgumentScorer."""

    def test_exact_argument_match(self):
        """Test exact argument matching."""
        scorer = ToolArgumentScorer()
        instance = Instance(
            question="Weather in NYC",
            expected_tool_calls=({"name": "get_weather", "arguments": {"location": "NYC"}},),
        )
        output = LMOutput(
            text="", tool_calls=[ToolCall.create("1", "get_weather", {"location": "NYC"})]
        )

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_wrong_argument_value(self):
        """Test with wrong argument value."""
        scorer = ToolArgumentScorer()
        instance = Instance(
            question="Weather in NYC",
            expected_tool_calls=({"name": "get_weather", "arguments": {"location": "NYC"}},),
        )
        output = LMOutput(
            text="",
            tool_calls=[ToolCall.create("1", "get_weather", {"location": "LA"})],
        )

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        scorer = ToolArgumentScorer(case_sensitive=False)
        instance = Instance(
            question="Test",
            expected_tool_calls=({"name": "test", "arguments": {"text": "HELLO"}},),
        )
        output = LMOutput(text="", tool_calls=[ToolCall.create("1", "test", {"text": "hello"})])

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_case_sensitive(self):
        """Test case sensitive matching."""
        scorer = ToolArgumentScorer(case_sensitive=True)
        instance = Instance(
            question="Test",
            expected_tool_calls=({"name": "test", "arguments": {"text": "HELLO"}},),
        )
        output = LMOutput(text="", tool_calls=[ToolCall.create("1", "test", {"text": "hello"})])

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_missing_argument(self):
        """Test with missing argument."""
        scorer = ToolArgumentScorer()
        instance = Instance(
            question="Test",
            expected_tool_calls=({"name": "test", "arguments": {"a": 1, "b": 2}},),
        )
        output = LMOutput(text="", tool_calls=[ToolCall.create("1", "test", {"a": 1})])

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_extra_arguments_ok(self):
        """Test that extra arguments don't affect score."""
        scorer = ToolArgumentScorer()
        instance = Instance(
            question="Test",
            expected_tool_calls=({"name": "test", "arguments": {"a": 1}},),
        )
        output = LMOutput(
            text="",
            tool_calls=[ToolCall.create("1", "test", {"a": 1, "extra": "value"})],
        )

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_numeric_coercion(self):
        """Test numeric type coercion."""
        scorer = ToolArgumentScorer()
        instance = Instance(
            question="Test",
            expected_tool_calls=({"name": "calc", "arguments": {"n": 5}},),
        )
        output = LMOutput(text="", tool_calls=[ToolCall.create("1", "calc", {"n": 5.0})])

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_wrong_tool_name(self):
        """Test with wrong tool name."""
        scorer = ToolArgumentScorer()
        instance = Instance(
            question="Test",
            expected_tool_calls=({"name": "expected", "arguments": {"a": 1}},),
        )
        output = LMOutput(text="", tool_calls=[ToolCall.create("1", "different", {"a": 1})])

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_scorer_name(self):
        """Test scorer name."""
        scorer = ToolArgumentScorer()
        assert scorer.name == "tool_argument"


class TestToolSequenceScorer:
    """Tests for ToolSequenceScorer."""

    def test_exact_sequence_match(self):
        """Test exact sequence matching."""
        scorer = ToolSequenceScorer()
        instance = Instance(
            question="Multi-step task",
            required_trajectory=(
                {"name": "step1"},
                {"name": "step2"},
                {"name": "step3"},
            ),
        )
        output = LMOutput(
            text="",
            tool_calls=[
                ToolCall.create("1", "step1", {}),
                ToolCall.create("2", "step2", {}),
                ToolCall.create("3", "step3", {}),
            ],
        )

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_subsequence_match(self):
        """Test subsequence matching with extra calls."""
        scorer = ToolSequenceScorer()
        instance = Instance(
            question="Task",
            required_trajectory=({"name": "a"}, {"name": "c"}),
        )
        output = LMOutput(
            text="",
            tool_calls=[
                ToolCall.create("1", "a", {}),
                ToolCall.create("2", "b", {}),  # Extra
                ToolCall.create("3", "c", {}),
            ],
        )

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_wrong_order(self):
        """Test wrong order fails."""
        scorer = ToolSequenceScorer(strict_order=True)
        instance = Instance(
            question="Task",
            required_trajectory=({"name": "a"}, {"name": "b"}),
        )
        output = LMOutput(
            text="",
            tool_calls=[
                ToolCall.create("1", "b", {}),
                ToolCall.create("2", "a", {}),
            ],
        )

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_non_strict_order(self):
        """Test non-strict order (set membership)."""
        scorer = ToolSequenceScorer(strict_order=False)
        instance = Instance(
            question="Task",
            required_trajectory=({"name": "a"}, {"name": "b"}),
        )
        output = LMOutput(
            text="",
            tool_calls=[
                ToolCall.create("1", "b", {}),
                ToolCall.create("2", "a", {}),
            ],
        )

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_missing_required_tool(self):
        """Test when required tool is missing."""
        scorer = ToolSequenceScorer()
        instance = Instance(
            question="Task",
            required_trajectory=({"name": "a"}, {"name": "b"}, {"name": "c"}),
        )
        output = LMOutput(
            text="",
            tool_calls=[
                ToolCall.create("1", "a", {}),
                ToolCall.create("2", "c", {}),  # Missing b
            ],
        )

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_scorer_name(self):
        """Test scorer name."""
        scorer = ToolSequenceScorer()
        assert scorer.name == "tool_sequence"
