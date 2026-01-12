"""Tests for olmo_eval.tasks.deepmind_math module."""

import pytest

from olmo_eval.core import Instance, LMOutput, RequestType
from olmo_eval.evals.constants.benchmarks import DEEPMIND_MATH_CATEGORIES
from olmo_eval.tasks.deepmind_math import (
    DeepMindMathTask,
    _check_sympy_equal,
    _clean_deepmind_string,
    _clean_prediction,
    _compare_math_answers,
    _make_deepmind_math_config,
)
from olmo_eval.tasks.base import TaskConfig
from olmo_eval.tasks.registry import get_task, list_tasks


# =============================================================================
# String Cleaning Tests
# =============================================================================


class TestCleanDeepMindString:
    """Tests for _clean_deepmind_string function."""

    def test_clean_binary_wrapper(self):
        """Test removing b'...\\n' wrapper."""
        assert _clean_deepmind_string("b'hello\\n'") == "hello"

    def test_clean_normal_string(self):
        """Test that normal strings are unchanged."""
        assert _clean_deepmind_string("hello world") == "hello world"

    def test_clean_with_whitespace(self):
        """Test stripping whitespace."""
        assert _clean_deepmind_string("  hello  ") == "hello"


class TestCleanPrediction:
    """Tests for _clean_prediction function."""

    def test_remove_hope_correct(self):
        """Test removing 'I hope it is correct' suffix."""
        text = "42 I hope it is correct"
        assert _clean_prediction(text) == "42"

    def test_remove_special_tokens(self):
        """Test removing special tokens."""
        text = "42<|eot_id|>"
        assert _clean_prediction(text) == "42"

    def test_strip_trailing_period(self):
        """Test stripping trailing period."""
        text = "42."
        assert _clean_prediction(text) == "42"

    def test_strip_latex_delimiters(self):
        """Test stripping LaTeX delimiters."""
        assert _clean_prediction("$42$") == "42"
        assert _clean_prediction("**42**") == "42"


# =============================================================================
# Math Comparison Tests
# =============================================================================


class TestCheckSympyEqual:
    """Tests for _check_sympy_equal function."""

    def test_equal_numbers(self):
        """Test equal numbers."""
        import sympy

        assert _check_sympy_equal(sympy.Integer(5), sympy.Integer(5))

    def test_unequal_numbers(self):
        """Test unequal numbers."""
        import sympy

        assert not _check_sympy_equal(sympy.Integer(5), sympy.Integer(6))

    def test_equivalent_expressions(self):
        """Test equivalent expressions."""
        import sympy

        x = sympy.Symbol("x")
        assert _check_sympy_equal(x + x, 2 * x)

    def test_non_sympy_values(self):
        """Test non-sympy values fall back to ==."""
        assert _check_sympy_equal(5, 5)
        assert not _check_sympy_equal(5, 6)
        assert _check_sympy_equal("hello", "hello")


class TestCompareMathAnswers:
    """Tests for _compare_math_answers function."""

    def test_exact_match(self):
        """Test exact string match."""
        assert _compare_math_answers("42", "42")

    def test_case_insensitive(self):
        """Test case insensitive match."""
        assert _compare_math_answers("Hello", "hello")

    def test_boolean_true_yes(self):
        """Test True matches yes."""
        assert _compare_math_answers("True", "yes")
        assert _compare_math_answers("true", "Yes")

    def test_boolean_false_no(self):
        """Test False matches no."""
        assert _compare_math_answers("False", "no")
        assert _compare_math_answers("false", "No")

    def test_boolean_exact(self):
        """Test boolean exact match."""
        assert _compare_math_answers("True", "True")
        assert _compare_math_answers("False", "false")

    def test_sympy_expression(self):
        """Test sympy expression comparison."""
        assert _compare_math_answers("5", "5.0")
        assert _compare_math_answers("2*3", "6")

    def test_sympy_mismatch(self):
        """Test sympy mismatch."""
        assert not _compare_math_answers("5", "6")


# =============================================================================
# DeepMindMathTask Tests
# =============================================================================


class TestDeepMindMathTask:
    """Tests for DeepMindMathTask."""

    @pytest.fixture
    def algebra_task(self):
        """Create an algebra task for testing."""
        return get_task("deepmind_math_algebra__linear_1d")

    def test_process_doc(self, algebra_task):
        """Test processing a document."""
        doc = {
            "question": "Solve 2*x = 4.",
            "answer": "2",
        }

        instance = algebra_task._process_doc(doc, index=0)

        assert isinstance(instance, Instance)
        assert instance.question == "Solve 2*x = 4."
        assert instance.gold_answer == "2"
        assert instance.metadata["index"] == 0
        assert instance.metadata["category"] == "algebra__linear_1d"

    def test_process_doc_with_binary_encoding(self, algebra_task):
        """Test processing doc with binary encoding artifacts."""
        doc = {
            "question": "b'Solve x = 1.\\n'",
            "answer": "b'1\\n'",
        }

        instance = algebra_task._process_doc(doc, index=0)

        assert instance.question == "Solve x = 1."
        assert instance.gold_answer == "1"

    def test_format_request(self, algebra_task):
        """Test request formatting."""
        instance = Instance(
            question="What is 2 + 2?",
            gold_answer="4",
        )

        request = algebra_task.format_request(instance)

        assert request.request_type == RequestType.COMPLETION
        assert "Problem:" in request.prompt
        assert "What is 2 + 2?" in request.prompt
        assert "Answer:" in request.prompt

    def test_extract_answer_with_pattern(self, algebra_task):
        """Test extracting answer with 'answer is' pattern."""
        output = LMOutput(text="Let me calculate. The answer is 42.")

        answer = algebra_task.extract_answer(output)

        assert answer == "42"

    def test_extract_answer_final_answer_pattern(self, algebra_task):
        """Test extracting with 'final answer is' pattern."""
        output = LMOutput(text="After much work, the final answer is 100.")

        answer = algebra_task.extract_answer(output)

        assert answer == "100"

    def test_extract_answer_raw(self, algebra_task):
        """Test extracting raw answer when no pattern matches."""
        output = LMOutput(text="42")

        answer = algebra_task.extract_answer(output)

        assert answer == "42"

    def test_score_answer_correct(self, algebra_task):
        """Test scoring correct answer."""
        assert algebra_task.score_answer("42", "42") is True
        assert algebra_task.score_answer("2*21", "42") is True

    def test_score_answer_incorrect(self, algebra_task):
        """Test scoring incorrect answer."""
        assert algebra_task.score_answer("41", "42") is False

    def test_score_answer_none(self, algebra_task):
        """Test scoring None answer."""
        assert algebra_task.score_answer(None, "42") is False

    def test_config_name(self):
        """Test config naming."""
        config = _make_deepmind_math_config("algebra__linear_1d")

        assert config.name == "deepmind_math_algebra__linear_1d"
        assert config.hf_dataset == "deepmind/math_dataset"
        assert config.hf_subsets == ("algebra__linear_1d",)


# =============================================================================
# Task Registration Tests
# =============================================================================


class TestDeepMindMathRegistration:
    """Tests for DeepMind Math task registration."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Ensure tasks are registered by importing module."""
        import olmo_eval.tasks.deepmind_math  # noqa: F401

        yield

    def test_all_56_tasks_registered(self):
        """Test that all 56 categories are registered."""
        all_tasks = list_tasks()
        for category in DEEPMIND_MATH_CATEGORIES:
            full_name = f"deepmind_math_{category}"
            assert full_name in all_tasks, f"{full_name} not registered"

    def test_get_algebra_linear_1d(self):
        """Test getting algebra__linear_1d task."""
        task = get_task("deepmind_math_algebra__linear_1d")
        assert task is not None
        assert task.category == "algebra__linear_1d"

    def test_get_arithmetic_add_or_sub(self):
        """Test getting arithmetic__add_or_sub task."""
        task = get_task("deepmind_math_arithmetic__add_or_sub")
        assert task is not None
        assert task.category == "arithmetic__add_or_sub"

    def test_get_calculus_differentiate(self):
        """Test getting calculus__differentiate task."""
        task = get_task("deepmind_math_calculus__differentiate")
        assert task is not None
        assert task.category == "calculus__differentiate"

    def test_task_has_correct_hf_path(self):
        """Test that tasks have correct HuggingFace path."""
        task = get_task("deepmind_math_numbers__is_prime")
        assert task.hf_path == "deepmind/math_dataset"


# =============================================================================
# Category-Specific Tests
# =============================================================================


class TestDeepMindMathCategories:
    """Tests for specific category behaviors."""

    def test_algebra_polynomial_roots(self):
        """Test algebra polynomial roots task."""
        task = get_task("deepmind_math_algebra__polynomial_roots")
        assert task.category == "algebra__polynomial_roots"

    def test_probability_task(self):
        """Test probability task."""
        task = get_task("deepmind_math_probability__swr_p_sequence")
        assert task.category == "probability__swr_p_sequence"

    def test_polynomials_expand(self):
        """Test polynomials expand task."""
        task = get_task("deepmind_math_polynomials__expand")
        assert task.category == "polynomials__expand"


# =============================================================================
# Answer Extraction Edge Cases
# =============================================================================


class TestAnswerExtractionEdgeCases:
    """Tests for edge cases in answer extraction."""

    @pytest.fixture
    def task(self):
        return get_task("deepmind_math_arithmetic__add_or_sub")

    def test_extract_negative_number(self, task):
        """Test extracting negative number."""
        output = LMOutput(text="The answer is -5.")
        assert task.extract_answer(output) == "-5"

    def test_extract_with_trailing_text(self, task):
        """Test extracting with trailing text."""
        output = LMOutput(text="The answer is 42, which makes sense.")
        # Should get "42, which makes sense" due to greedy match
        answer = task.extract_answer(output)
        assert "42" in answer

    def test_extract_decimal(self, task):
        """Test extracting decimal number (no trailing period)."""
        output = LMOutput(text="The answer is 3.14")
        assert task.extract_answer(output) == "3.14"
