"""Tests for all task module registrations and basic functionality."""

import pytest

from olmo_eval.common.types import Instance, LMOutput, RequestType
from olmo_eval.evals.tasks.common import get_task, list_tasks
from olmo_eval.evals.tasks.gsm8k import _clean_short_answer, _extract_last_number


class TestGSMNumberExtraction:
    """Tests for GSM8K/GSM-Symbolic number extraction helpers."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("The answer is 42.", "42"),
            ("So the answer is 6.", "6"),
            ("5 + 4 = 9. So the answer is 9.", "9"),
            # Negative integers
            ("The temperature dropped to -3 degrees.", "-3"),
            ("The answer is -15.", "-15"),
            # Positive sign
            ("The result is +7.", "+7"),
            # Floats
            ("The answer is 3.14.", "3.14"),
            ("She has -2.5 dollars.", "-2.5"),
            ("A gain of +0.75 points.", "+0.75"),
            # Comma-separated numbers
            ("The population is 1,000,000.", "1000000"),
            ("He earned $2,500.", "2500"),
            # Last number wins
            ("First 10, then 20, finally 30.", "30"),
            ("Step 1: 100 - 60 = 40. Step 2: 40 - 8 = 32.", "32"),
            # No numbers
            ("There is no number here.", None),
            ("", None),
            # Edge: bare decimal
            (".5 liters", ".5"),
        ],
    )
    def test_extract_last_number(self, text: str, expected: str | None):
        assert _extract_last_number(text) == expected

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("6", "6"),
            ("-3", "-3"),
            ("3.14", "3.14"),
            ("1,200", "1200"),
            # Falls back to original text when no number is found
            ("no number", "no number"),
        ],
    )
    def test_clean_short_answer(self, text: str, expected: str):
        assert _clean_short_answer(text) == expected


class TestTaskRegistration:
    """Tests for task registration across all modules."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Ensure tasks are registered by importing modules."""
        import olmo_eval.evals.tasks  # noqa: F401

        yield

    # HumanEval
    def test_humaneval_registered(self):
        """Test that humaneval is registered."""
        assert "humaneval" in list_tasks()

    def test_humaneval_plus_registered(self):
        """Test that humaneval_plus is registered."""
        assert "humaneval_plus" in list_tasks()

    def test_get_humaneval(self):
        """Test getting humaneval task."""
        task = get_task("humaneval")
        assert task.config.name == "humaneval"

    def test_get_humaneval_plus(self):
        """Test getting humaneval_plus task."""
        task = get_task("humaneval_plus")
        assert task.config.name == "humaneval_plus"

    # MBPP
    def test_mbpp_registered(self):
        """Test that mbpp is registered."""
        assert "mbpp" in list_tasks()

    def test_mbpp_plus_registered(self):
        """Test that mbpp_plus is registered."""
        assert "mbpp_plus" in list_tasks()

    def test_get_mbpp(self):
        """Test getting mbpp task."""
        task = get_task("mbpp")
        assert task.config.name == "mbpp"

    def test_get_mbpp_plus(self):
        """Test getting mbpp_plus task."""
        task = get_task("mbpp_plus")
        assert task.config.name == "mbpp_plus"

    # Multilingual MBPP (sample languages)
    def test_mt_mbpp_python_registered(self):
        """Test that mt_mbpp_python is registered."""
        assert "mt_mbpp_python" in list_tasks()

    def test_mt_mbpp_javascript_registered(self):
        """Test that mt_mbpp_javascript is registered."""
        assert "mt_mbpp_javascript" in list_tasks()

    def test_mt_mbpp_v2fix_python_registered(self):
        """Test that mt_mbpp_v2fix_python is registered."""
        assert "mt_mbpp_v2fix_python" in list_tasks()

    def test_get_mt_mbpp_python(self):
        """Test getting mt_mbpp_python task."""
        task = get_task("mt_mbpp_python")
        assert task.config.name == "mt_mbpp_python"


class TestHumanEvalTask:
    """Tests for HumanEval task functionality."""

    @pytest.fixture
    def task(self):
        """Create a HumanEval task."""
        return get_task("humaneval")

    def test_format_request(self, task):
        """Test request formatting."""
        instance = Instance(
            question="```python\ndef add(a, b):",
            gold_answer="    return a + b",
            metadata={
                "id": "test_0",
                "entry_point": "add",
                "answer_prefix": "def add(a, b):",
                "test": "assert add(1, 2) == 3",
            },
        )

        request = task.format_request(instance)
        assert request.request_type == RequestType.COMPLETION
        assert "def add" in request.prompt


class TestMBPPTask:
    """Tests for MBPP task functionality."""

    @pytest.fixture
    def task(self):
        """Create a MBPP task."""
        return get_task("mbpp")

    def test_format_request(self, task):
        """Test request formatting."""
        instance = Instance(
            question="Write a function to add two numbers.\ndef add(a, b):",
            gold_answer="    return a + b",
            metadata={
                "id": 1,
                "answer_prefix": "def add(a, b):",
                "test": "assert add(1, 2) == 3",
            },
        )

        request = task.format_request(instance)
        assert request.request_type == RequestType.COMPLETION
        assert "add" in request.prompt


class TestMultilingualMBPPTask:
    """Tests for Multilingual MBPP task functionality."""

    @pytest.fixture
    def task(self):
        """Create a Multilingual MBPP Python task."""
        return get_task("mt_mbpp_python")

    def test_format_request(self, task):
        """Test request formatting."""
        instance = Instance(
            question="Write a function to add two numbers.\n```python\n",
            gold_answer="def add(a, b):\n    return a + b\n```",
            metadata={
                "id": 1,
                "language": "python",
                "text": "Write a function to add two numbers.",
                "code": "def add(a, b):\n    return a + b",
            },
        )

        request = task.format_request(instance)
        assert request.request_type == RequestType.COMPLETION
        assert "python" in request.prompt

    def test_extract_answer(self, task):
        """Test answer extraction."""
        output = LMOutput(text="def add(a, b):\n    return a + b\n```")
        answer = task.extract_answer(output)
        assert "def add" in answer
        assert "return a + b" in answer
