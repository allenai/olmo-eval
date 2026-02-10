"""Tests for olmo_eval.core.formatters module."""

import pytest

from olmo_eval.common.formatters import (
    ChatFormatter,
    CompletionFormatter,
    MultipleChoiceFormatter,
    PPLFormatter,
)
from olmo_eval.common.types import Instance, RequestType


class TestChatFormatter:
    """Tests for ChatFormatter."""

    def test_format_basic(self):
        """Test basic chat formatting."""
        formatter = ChatFormatter()
        instance = Instance(question="What is 2+2?", gold_answer="4")

        request = formatter.format(instance)

        assert request.request_type == RequestType.CHAT
        assert len(request.messages) == 1
        assert request.messages[0]["role"] == "user"
        assert request.messages[0]["content"] == "What is 2+2?"

    def test_format_with_system_prompt(self):
        """Test chat formatting with system prompt."""
        formatter = ChatFormatter(system_prompt="You are a helpful assistant.")
        instance = Instance(question="What is 2+2?", gold_answer="4")

        request = formatter.format(instance)

        assert len(request.messages) == 2
        assert request.messages[0]["role"] == "system"
        assert request.messages[0]["content"] == "You are a helpful assistant."
        assert request.messages[1]["role"] == "user"

    def test_format_with_fewshot(self):
        """Test chat formatting with few-shot examples."""
        formatter = ChatFormatter()
        instance = Instance(question="What is 3+3?", gold_answer="6")
        fewshot = [
            Instance(question="What is 1+1?", gold_answer="2"),
            Instance(question="What is 2+2?", gold_answer="4"),
        ]

        request = formatter.format(instance, fewshot)

        # 2 fewshot * 2 messages each + 1 final user message = 5
        assert len(request.messages) == 5
        assert request.messages[0]["role"] == "user"
        assert request.messages[0]["content"] == "What is 1+1?"
        assert request.messages[1]["role"] == "assistant"
        assert request.messages[1]["content"] == "2"
        assert request.messages[4]["role"] == "user"
        assert request.messages[4]["content"] == "What is 3+3?"

    def test_format_with_custom_templates(self):
        """Test chat formatting with custom templates."""
        formatter = ChatFormatter(
            user_template="Q: {question}",
            assistant_template="A: {answer}",
        )
        instance = Instance(question="Capital of France?", gold_answer="Paris")

        request = formatter.format(instance)

        assert request.messages[0]["content"] == "Q: Capital of France?"

    def test_format_fewshot_with_none_gold_answer(self):
        """Test that None gold_answer becomes empty string."""
        formatter = ChatFormatter()
        instance = Instance(question="Test?", gold_answer="yes")
        fewshot = [Instance(question="Example?", gold_answer=None)]

        request = formatter.format(instance, fewshot)

        assert request.messages[1]["content"] == ""


class TestCompletionFormatter:
    """Tests for CompletionFormatter."""

    def test_format_basic(self):
        """Test basic completion formatting."""
        formatter = CompletionFormatter()
        instance = Instance(question="What is 2+2?", gold_answer="4")

        request = formatter.format(instance)

        assert request.request_type == RequestType.COMPLETION
        assert request.prompt == "What is 2+2?"

    def test_format_with_template(self):
        """Test completion formatting with custom template."""
        formatter = CompletionFormatter(template="Question: {question}\nAnswer:")
        instance = Instance(question="What is 2+2?", gold_answer="4")

        request = formatter.format(instance)

        assert request.prompt == "Question: What is 2+2?\nAnswer:"

    def test_format_with_fewshot(self):
        """Test completion formatting with few-shot examples."""
        formatter = CompletionFormatter(
            template="Q: {question}",
            answer_prefix=" A: ",
        )
        instance = Instance(question="3+3?", gold_answer="6")
        fewshot = [
            Instance(question="1+1?", gold_answer="2"),
            Instance(question="2+2?", gold_answer="4"),
        ]

        request = formatter.format(instance, fewshot)

        expected = "Q: 1+1? A: 2\n\nQ: 2+2? A: 4\n\nQ: 3+3? A: "
        assert request.prompt == expected

    def test_format_with_custom_separator(self):
        """Test completion formatting with custom separator."""
        formatter = CompletionFormatter(
            template="{question}",
            fewshot_separator="---",
        )
        instance = Instance(question="C", gold_answer="3")
        fewshot = [
            Instance(question="A", gold_answer="1"),
            Instance(question="B", gold_answer="2"),
        ]

        request = formatter.format(instance, fewshot)

        assert request.prompt == "A1---B2---C"

    def test_format_no_fewshot(self):
        """Test completion formatting without few-shot."""
        formatter = CompletionFormatter(template="{question}")
        instance = Instance(question="Test", gold_answer="yes")

        request = formatter.format(instance, None)

        assert request.prompt == "Test"


class TestMultipleChoiceFormatter:
    """Tests for MultipleChoiceFormatter."""

    def test_format_basic_with_choices_in_prompt(self):
        """Test basic multiple choice formatting (default includes choices in prompt)."""
        formatter = MultipleChoiceFormatter()
        instance = Instance(
            question="What color is the sky?",
            gold_answer="B",
            choices=("Red", "Blue", "Green"),
        )

        request = formatter.format(instance)

        assert request.request_type == RequestType.COMPLETION
        # Default behavior includes labeled choices in prompt
        assert "What color is the sky?" in request.prompt
        assert "A. Red" in request.prompt
        assert "B. Blue" in request.prompt
        assert "C. Green" in request.prompt
        assert request.continuations == ("Red", "Blue", "Green")

    def test_format_without_choices_in_prompt(self):
        """Test multiple choice formatting without choices in prompt."""
        formatter = MultipleChoiceFormatter(include_choices_in_prompt=False)
        instance = Instance(
            question="What color is the sky?",
            gold_answer="B",
            choices=("Red", "Blue", "Green"),
        )

        request = formatter.format(instance)

        assert request.request_type == RequestType.COMPLETION
        assert request.prompt == "What color is the sky?"
        assert request.continuations == ("Red", "Blue", "Green")

    def test_format_with_templates(self):
        """Test multiple choice formatting with custom templates."""
        formatter = MultipleChoiceFormatter(
            template="Q: {question}",
            choice_template=" {choice}",
            include_choices_in_prompt=False,
        )
        instance = Instance(
            question="Capital?",
            gold_answer="A",
            choices=("Paris", "London"),
        )

        request = formatter.format(instance)

        assert request.prompt == "Q: Capital?"
        assert request.continuations == (" Paris", " London")

    def test_format_no_choices(self):
        """Test multiple choice formatting without choices."""
        formatter = MultipleChoiceFormatter()
        instance = Instance(question="Test?", gold_answer="yes", choices=None)

        request = formatter.format(instance)

        assert request.prompt == "Test?"
        assert request.continuations == ()

    def test_format_empty_choices(self):
        """Test multiple choice formatting with empty choices."""
        formatter = MultipleChoiceFormatter()
        instance = Instance(question="Test?", gold_answer="yes", choices=())

        request = formatter.format(instance)

        assert request.continuations == ()

    def test_format_ignores_fewshot(self):
        """Test that MultipleChoiceFormatter ignores fewshot (by design)."""
        formatter = MultipleChoiceFormatter(include_choices_in_prompt=False)
        instance = Instance(
            question="Test?",
            gold_answer="A",
            choices=("Yes", "No"),
        )
        fewshot = [Instance(question="Example?", gold_answer="A")]

        request = formatter.format(instance, fewshot)

        # Fewshot is ignored for MC format
        assert request.prompt == "Test?"
        assert request.continuations == ("Yes", "No")


class TestPPLFormatter:
    """Tests for PPLFormatter."""

    def test_ppl_formatter_uses_question_as_context(self):
        """PPLFormatter uses question as context to avoid first-token issue."""
        formatter = PPLFormatter()

        # With question - uses question as prompt
        instance = Instance(question="What is 2+2?", gold_answer="4")
        request = formatter.format(instance)
        assert request.prompt == "What is 2+2?"
        assert request.request_type == RequestType.LOGLIKELIHOOD

        # Without question - empty prompt
        instance = Instance(question="", gold_answer="text")
        request = formatter.format(instance)
        assert request.prompt == ""

    def test_ppl_formatter_leading_space_with_context(self):
        """PPLFormatter adds leading space to continuation when there's context."""
        formatter = PPLFormatter()

        # With question - adds leading space
        instance = Instance(question="Context", gold_answer="answer")
        request = formatter.format(instance)
        assert request.continuations[0] == " answer"
        assert request.continuations[0][0] == " "

        # Without question - no leading space
        instance = Instance(question="", gold_answer="answer")
        request = formatter.format(instance)
        assert request.continuations[0] == "answer"

    def test_ppl_formatter_mc_uses_choice_text(self):
        """MC tasks use actual choice text via gold_idx."""
        formatter = PPLFormatter()
        instance = Instance(
            question="Question?",
            gold_answer="B",
            choices=("Option A", "Option B", "Option C"),
            metadata={"gold_idx": 1},
        )

        request = formatter.format(instance)

        # Leading space added because there's a question
        assert request.continuations == (" Option B",)
        assert request.prompt == "Question?"

    def test_ppl_formatter_gold_text_metadata(self):
        """PPLFormatter uses gold_text from metadata when available."""
        formatter = PPLFormatter()
        instance = Instance(
            question="Question?",
            gold_answer="B",
            metadata={"gold_text": "The actual gold text"},
        )

        request = formatter.format(instance)

        # Leading space added because there's a question
        assert request.continuations == (" The actual gold text",)

    def test_ppl_formatter_fallback_to_gold_answer(self):
        """PPLFormatter falls back to gold_answer when no other source."""
        formatter = PPLFormatter()
        instance = Instance(question="Question?", gold_answer="fallback answer")

        request = formatter.format(instance)

        # Leading space added because there's a question
        assert request.continuations == (" fallback answer",)

    def test_ppl_formatter_requires_gold_answer(self):
        """PPLFormatter raises ValueError when no gold answer is available."""
        formatter = PPLFormatter()
        instance = Instance(question="Question?", gold_answer=None)

        with pytest.raises(ValueError, match="PPLFormatter requires a gold answer"):
            formatter.format(instance)

    def test_ppl_formatter_includes_fewshot(self):
        """PPLFormatter includes fewshot examples in the prompt."""
        formatter = PPLFormatter()
        instance = Instance(question="Test?", gold_answer="answer")
        fewshot = [Instance(question="Example?", gold_answer="example")]

        request = formatter.format(instance, fewshot)

        # Fewshot examples are included in prompt, joined by separator
        assert request.prompt == "Example?example\n\nTest?"
        assert request.continuations == (" answer",)
