"""Tests for olmo_eval.tasks.bbh module."""

import pytest

from olmo_eval.core import Instance, LMOutput, RequestType
from olmo_eval.evals.tasks import get_task, list_tasks
from olmo_eval.evals.tasks.bbh import BBH_ANSWER_REGEX, _make_bbh_config

# =============================================================================
# BBH Task Tests
# =============================================================================


class TestBBHTask:
    """Tests for BBH base task functionality."""

    @pytest.fixture
    def bbh_boolean_task(self):
        """Create a BBH boolean_expressions task for testing."""
        return get_task("bbh_boolean_expressions")

    def test_process_doc_with_answer_pattern(self, bbh_boolean_task):
        """Test processing doc with 'answer is X' pattern."""
        doc = {
            "input": "not ( ( not not True ) ) is",
            "target": (
                "Let's think step by step. not not True is True. "
                "not True is False. So the answer is False."
            ),
        }

        instance = bbh_boolean_task._process_doc(doc, index=0)

        assert isinstance(instance, Instance)
        assert instance.question == doc["input"]
        assert instance.gold_answer == "False"
        assert instance.metadata["index"] == 0
        assert instance.metadata["solution"] == doc["target"]
        assert instance.metadata["subset"] == "boolean_expressions"

    def test_process_doc_without_answer_pattern(self, bbh_boolean_task):
        """Test processing doc without standard answer pattern."""
        doc = {
            "input": "Test input",
            "target": "Some answer without the pattern",
        }

        instance = bbh_boolean_task._process_doc(doc, index=1)

        assert instance.gold_answer == "Some answer without the pattern"

    def test_format_request(self, bbh_boolean_task):
        """Test request formatting with CoT prompt."""
        instance = Instance(
            question="not True is",
            gold_answer="False",
        )

        request = bbh_boolean_task.format_request(instance)

        assert request.request_type == RequestType.COMPLETION
        assert "Q: not True is" in request.prompt
        assert "A: Let's think step by step." in request.prompt

    def test_extract_answer_with_answer_is_pattern(self, bbh_boolean_task):
        """Test answer extraction with 'the answer is' pattern."""
        output = LMOutput(
            text="Let me think. not True is False. The answer is False.",
        )

        answer = bbh_boolean_task.extract_answer(output)

        assert answer == "False"

    def test_extract_answer_with_fallback_regex(self, bbh_boolean_task):
        """Test answer extraction falls back to task regex."""
        output = LMOutput(
            text="I believe the result is True.",
        )

        answer = bbh_boolean_task.extract_answer(output)

        assert answer == "True"

    def test_config_name(self):
        """Test BBH config naming."""
        config = _make_bbh_config("boolean_expressions")

        assert config.name == "bbh_boolean_expressions"
        assert config.hf_dataset == "lukaemon/bbh"
        assert config.hf_subsets == ("boolean_expressions",)


# =============================================================================
# Answer Regex Pattern Tests
# =============================================================================


class TestBBHAnswerRegex:
    """Tests for BBH answer extraction regex patterns."""

    def test_boolean_expressions_pattern(self):
        """Test boolean expressions regex."""
        import re

        pattern = BBH_ANSWER_REGEX["boolean_expressions"]
        assert re.search(pattern, "True")
        assert re.search(pattern, "false")
        assert not re.search(pattern, "(A)")

    def test_yes_no_patterns(self):
        """Test yes/no patterns for causal judgement."""
        import re

        pattern = BBH_ANSWER_REGEX["causal_judgement"]
        assert re.search(pattern, "Yes")
        assert re.search(pattern, "no")
        assert not re.search(pattern, "True")

    def test_multiple_choice_patterns(self):
        """Test multiple choice letter patterns."""
        import re

        # Date understanding: A-F
        pattern = BBH_ANSWER_REGEX["date_understanding"]
        assert re.search(pattern, "(A)")
        assert re.search(pattern, "(F)")
        assert not re.search(pattern, "(G)")

        # Disambiguation: A-C
        pattern = BBH_ANSWER_REGEX["disambiguation_qa"]
        assert re.search(pattern, "(B)")
        assert not re.search(pattern, "(D)")

    def test_numeric_patterns(self):
        """Test numeric answer patterns."""
        import re

        # Multistep arithmetic
        pattern = BBH_ANSWER_REGEX["multistep_arithmetic_two"]
        assert re.search(pattern, "42")
        assert re.search(pattern, "-123")

        # Object counting
        pattern = BBH_ANSWER_REGEX["object_counting"]
        assert re.search(pattern, "7")

    def test_dyck_languages_pattern(self):
        """Test Dyck languages bracket pattern."""
        import re

        pattern = BBH_ANSWER_REGEX["dyck_languages"]
        assert re.search(pattern, "] ) }")
        assert re.search(pattern, ">>")

    def test_word_sorting_pattern(self):
        """Test word sorting pattern."""
        import re

        pattern = BBH_ANSWER_REGEX["word_sorting"]
        assert re.search(pattern, "apple banana cherry")

    def test_all_27_tasks_have_regex(self):
        """Test that all 27 BBH tasks have regex patterns."""
        from olmo_eval.evals.constants.benchmarks import BBH_TASKS

        for task in BBH_TASKS:
            assert task in BBH_ANSWER_REGEX, f"Missing regex for {task}"


# =============================================================================
# Task Registration Tests
# =============================================================================


class TestBBHRegistration:
    """Tests for BBH task registration."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Ensure tasks are registered by importing module."""
        import olmo_eval.evals.tasks.bbh  # noqa: F401

        yield

    def test_all_27_tasks_registered(self):
        """Test that all 27 BBH tasks are registered."""
        from olmo_eval.evals.constants.benchmarks import BBH_TASKS

        all_tasks = list_tasks()
        for task_name in BBH_TASKS:
            full_name = f"bbh_{task_name}"
            assert full_name in all_tasks, f"{full_name} not registered"

    def test_get_bbh_boolean_expressions(self):
        """Test getting boolean_expressions task."""
        task = get_task("bbh_boolean_expressions")
        assert task is not None
        assert task.subset == "boolean_expressions"

    def test_get_bbh_date_understanding(self):
        """Test getting date_understanding task."""
        task = get_task("bbh_date_understanding")
        assert task is not None
        assert task.subset == "date_understanding"

    def test_get_bbh_word_sorting(self):
        """Test getting word_sorting task."""
        task = get_task("bbh_word_sorting")
        assert task is not None
        assert task.subset == "word_sorting"

    def test_task_has_correct_hf_path(self):
        """Test that tasks have correct HuggingFace path."""
        task = get_task("bbh_navigate")
        assert task.hf_path == "lukaemon/bbh"


# =============================================================================
# Specific BBH Task Type Tests
# =============================================================================


class TestBBHMultipleChoiceTasks:
    """Tests for BBH multiple choice tasks."""

    def test_geometric_shapes_answer_extraction(self):
        """Test geometric shapes (A-K choices)."""
        task = get_task("bbh_geometric_shapes")
        output = LMOutput(
            text="Looking at the shape, the answer is (C).",
        )
        assert task.extract_answer(output) == "(C)"

    def test_movie_recommendation_answer_extraction(self):
        """Test movie recommendation (A-E choices)."""
        task = get_task("bbh_movie_recommendation")
        output = LMOutput(
            text="Based on the preferences, the answer is (D).",
        )
        assert task.extract_answer(output) == "(D)"


class TestBBHBooleanTasks:
    """Tests for BBH boolean/yes-no tasks."""

    def test_formal_fallacies_valid_invalid(self):
        """Test formal fallacies (valid/invalid)."""
        task = get_task("bbh_formal_fallacies")
        output = LMOutput(
            text="The argument is invalid because the premise is flawed.",
        )
        assert task.extract_answer(output) == "invalid"

    def test_navigate_yes_no(self):
        """Test navigate (yes/no)."""
        task = get_task("bbh_navigate")
        output = LMOutput(
            text="Following the instructions, you return to the start. Yes.",
        )
        assert task.extract_answer(output) == "Yes"


class TestBBHNumericTasks:
    """Tests for BBH numeric answer tasks."""

    def test_multistep_arithmetic(self):
        """Test multistep arithmetic task."""
        task = get_task("bbh_multistep_arithmetic_two")
        output = LMOutput(
            text="Let me calculate: 5 + 3 = 8, then 8 * 2 = 16. The answer is 16.",
        )
        assert task.extract_answer(output) == "16"

    def test_object_counting(self):
        """Test object counting task."""
        task = get_task("bbh_object_counting")
        output = LMOutput(
            text="Counting all the objects: 1, 2, 3, 4, 5. The answer is 5.",
        )
        assert task.extract_answer(output) == "5"
