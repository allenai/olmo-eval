"""Tests for olmo_eval.tasks.arc module."""

import pytest

from olmo_eval.core import Instance, LMOutput, RequestType
from olmo_eval.evals.tasks.arc import (
    ARCChallenge,
    ARCEasy,
    _arc_challenge_config,
    _arc_easy_config,
    _extract_mcqa_answer,
)
from olmo_eval.evals.tasks import TaskConfig, get_task, list_tasks


class TestExtractMCQAAnswer:
    """Tests for _extract_mcqa_answer function."""

    def test_extract_uppercase_letter(self):
        """Test extracting uppercase letters A-D."""
        assert _extract_mcqa_answer("A") == "A"
        assert _extract_mcqa_answer("B") == "B"
        assert _extract_mcqa_answer("C") == "C"
        assert _extract_mcqa_answer("D") == "D"

    def test_extract_lowercase_letter(self):
        """Test extracting lowercase letters and converting to uppercase."""
        assert _extract_mcqa_answer("a") == "A"
        assert _extract_mcqa_answer("b") == "B"
        assert _extract_mcqa_answer("c") == "C"
        assert _extract_mcqa_answer("d") == "D"

    def test_extract_with_parentheses(self):
        """Test extracting letters with parentheses."""
        assert _extract_mcqa_answer("(A)") == "A"
        assert _extract_mcqa_answer("(B)") == "B"
        assert _extract_mcqa_answer("(c)") == "C"
        assert _extract_mcqa_answer("(d)") == "D"

    def test_extract_from_text(self):
        """Test extracting from longer text."""
        assert _extract_mcqa_answer("The answer is A.") == "A"
        assert _extract_mcqa_answer("I think (B) is correct") == "B"
        assert _extract_mcqa_answer("Option C seems right") == "C"

    def test_extract_first_match(self):
        """Test that first match is returned when multiple present."""
        assert _extract_mcqa_answer("A or B") == "A"
        assert _extract_mcqa_answer("Not C, but D") == "C"

    def test_no_match_returns_none(self):
        """Test that None is returned when no match found."""
        assert _extract_mcqa_answer("") is None
        assert _extract_mcqa_answer("I think so") is None  # No a-d letters
        assert _extract_mcqa_answer("123") is None
        assert _extract_mcqa_answer("E") is None  # E not in A-D range
        assert _extract_mcqa_answer("xyz") is None

    def test_extract_with_whitespace(self):
        """Test extraction with surrounding whitespace."""
        assert _extract_mcqa_answer("  A  ") == "A"
        assert _extract_mcqa_answer("\n(B)\n") == "B"


class TestARCTaskProcessDoc:
    """Tests for ARCTask._process_doc method."""

    @pytest.fixture
    def arc_task(self):
        """Create an ARC task for testing."""
        config = _arc_challenge_config()
        return ARCChallenge(config)

    def test_process_doc_letter_answer(self, arc_task):
        """Test processing doc with letter answer key."""
        doc = {
            "id": "test_001",
            "question": "What is the capital of France?",
            "answerKey": "B",
            "choices": {
                "text": ["London", "Paris", "Berlin", "Madrid"],
                "label": ["A", "B", "C", "D"],
            },
        }

        instance = arc_task._process_doc(doc)

        assert isinstance(instance, Instance)
        assert instance.question == "What is the capital of France?"
        assert instance.gold_answer == "B"
        assert instance.choices == ("London", "Paris", "Berlin", "Madrid")
        assert instance.metadata["id"] == "test_001"
        assert instance.metadata["gold_idx"] == 1
        assert instance.metadata["gold_text"] == "Paris"

    def test_process_doc_numeric_answer(self, arc_task):
        """Test processing doc with numeric answer key (1-based index)."""
        doc = {
            "id": "test_002",
            "question": "What is 2+2?",
            "answerKey": "3",  # 1-based, so this is C
            "choices": {
                "text": ["2", "3", "4", "5"],
                "label": ["A", "B", "C", "D"],
            },
        }

        instance = arc_task._process_doc(doc)

        assert instance.gold_answer == "C"  # Converted from "3"
        assert instance.metadata["gold_idx"] == 2
        assert instance.metadata["gold_text"] == "4"

    def test_process_doc_three_choices(self, arc_task):
        """Test processing doc with only 3 choices."""
        doc = {
            "id": "test_003",
            "question": "Primary colors include?",
            "answerKey": "A",
            "choices": {
                "text": ["Red", "Purple", "Orange"],
                "label": ["A", "B", "C"],
            },
        }

        instance = arc_task._process_doc(doc)

        assert len(instance.choices) == 3
        assert instance.gold_answer == "A"
        assert instance.metadata["gold_idx"] == 0


class TestARCTaskFormatRequest:
    """Tests for ARCTask.format_request method."""

    @pytest.fixture
    def arc_task_with_formatter(self):
        """Create an ARC task with the default MultipleChoiceFormatter."""
        config = _arc_challenge_config()
        return ARCChallenge(config)

    @pytest.fixture
    def arc_task_no_formatter(self):
        """Create an ARC task without formatter (uses default behavior)."""
        config = TaskConfig(
            name="arc_no_formatter",
            hf_dataset="allenai/ai2_arc",
            hf_subsets=("ARC-Challenge",),
            formatter=None,  # No formatter, use default
        )
        return ARCChallenge(config)

    def test_format_request_with_formatter(self, arc_task_with_formatter):
        """Test request formatting when using MultipleChoiceFormatter."""
        instance = Instance(
            question="What is the boiling point of water?",
            gold_answer="B",
            choices=("50C", "100C", "150C", "200C"),
        )

        request = arc_task_with_formatter.format_request(instance)

        assert request.request_type == RequestType.COMPLETION
        assert "What is the boiling point of water?" in request.prompt
        # MultipleChoiceFormatter puts choices as continuations
        assert request.continuations == ("50C", "100C", "150C", "200C")

    def test_format_request_default_behavior(self, arc_task_no_formatter):
        """Test default request formatting without formatter."""
        instance = Instance(
            question="What is the boiling point of water?",
            gold_answer="B",
            choices=("50C", "100C", "150C", "200C"),
        )

        request = arc_task_no_formatter.format_request(instance)

        assert request.request_type == RequestType.COMPLETION
        assert "What is the boiling point of water?" in request.prompt
        # Default behavior includes labeled choices in prompt
        assert "A. 50C" in request.prompt
        assert "B. 100C" in request.prompt
        assert "C. 150C" in request.prompt
        assert "D. 200C" in request.prompt
        assert request.prompt.endswith("Answer:")
        # Default behavior uses letter continuations
        assert request.continuations == (" A", " B", " C", " D")

    def test_format_request_three_choices(self, arc_task_no_formatter):
        """Test formatting with only 3 choices (default behavior)."""
        instance = Instance(
            question="Test question?",
            gold_answer="A",
            choices=("Choice 1", "Choice 2", "Choice 3"),
        )

        request = arc_task_no_formatter.format_request(instance)

        assert request.continuations == (" A", " B", " C")
        assert "D." not in request.prompt


class TestARCTaskExtractAnswer:
    """Tests for ARCTask.extract_answer method."""

    @pytest.fixture
    def arc_task(self):
        """Create an ARC task for testing."""
        config = _arc_challenge_config()
        return ARCChallenge(config)

    def test_extract_answer_simple(self, arc_task):
        """Test extracting simple letter answer."""
        output = LMOutput(text="A")
        assert arc_task.extract_answer(output) == "A"

    def test_extract_answer_with_text(self, arc_task):
        """Test extracting answer from longer response."""
        output = LMOutput(text="I think (B) is right...")
        assert arc_task.extract_answer(output) == "B"

    def test_extract_answer_none(self, arc_task):
        """Test extracting when no valid answer present."""
        output = LMOutput(text="I think so")  # No a-d letters
        assert arc_task.extract_answer(output) is None


class TestARCConfigs:
    """Tests for ARC task config factories."""

    def test_arc_challenge_config(self):
        """Test ARC Challenge config factory."""
        config = _arc_challenge_config()

        assert config.name == "arc_challenge"
        assert config.hf_dataset == "allenai/ai2_arc"
        assert config.hf_subsets == ("ARC-Challenge",)
        assert config.formatter is not None
        assert len(config.scorers) == 1
        assert len(config.metrics) == 1

    def test_arc_easy_config(self):
        """Test ARC Easy config factory."""
        config = _arc_easy_config()

        assert config.name == "arc_easy"
        assert config.hf_dataset == "allenai/ai2_arc"
        assert config.hf_subsets == ("ARC-Easy",)
        assert config.formatter is not None
        assert len(config.scorers) == 1
        assert len(config.metrics) == 1


class TestARCTaskClasses:
    """Tests for ARCChallenge and ARCEasy classes."""

    def test_arc_challenge_dataset_name(self):
        """Test ARCChallenge uses correct dataset name."""
        config = _arc_challenge_config()
        task = ARCChallenge(config)
        assert task.dataset_name == "ARC-Challenge"

    def test_arc_easy_dataset_name(self):
        """Test ARCEasy uses correct dataset name."""
        config = _arc_easy_config()
        task = ARCEasy(config)
        assert task.dataset_name == "ARC-Easy"

    def test_arc_challenge_hf_path(self):
        """Test ARCChallenge has correct HF path."""
        assert ARCChallenge.hf_path == "allenai/ai2_arc"

    def test_arc_easy_hf_path(self):
        """Test ARCEasy has correct HF path."""
        assert ARCEasy.hf_path == "allenai/ai2_arc"


class TestARCTaskRegistration:
    """Tests for ARC task registration."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Ensure tasks are registered by importing module."""
        # Import to trigger registration
        import olmo_eval.evals.tasks.arc  # noqa: F401

        yield

    def test_arc_challenge_registered(self):
        """Test that arc_challenge is registered."""
        tasks = list_tasks()
        assert "arc_challenge" in tasks

    def test_arc_easy_registered(self):
        """Test that arc_easy is registered."""
        tasks = list_tasks()
        assert "arc_easy" in tasks

    def test_get_arc_challenge(self):
        """Test getting arc_challenge task."""
        task = get_task("arc_challenge")
        assert isinstance(task, ARCChallenge)
        assert task.config.name == "arc_challenge"

    def test_get_arc_easy(self):
        """Test getting arc_easy task."""
        task = get_task("arc_easy")
        assert isinstance(task, ARCEasy)
        assert task.config.name == "arc_easy"
