"""Tests for olmo_eval.core.formatters module."""

from olmo_eval.core.formatters import (
    ChatFormatter,
    CompletionFormatter,
    MultipleChoiceFormatter,
)
from olmo_eval.core.types import Instance, RequestType


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

    def test_format_basic(self):
        """Test basic multiple choice formatting."""
        formatter = MultipleChoiceFormatter()
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
        formatter = MultipleChoiceFormatter()
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
