"""Tests for olmo_eval.tasks.basic_skills module."""

import pytest

from olmo_eval.core import Instance, LMOutput, RequestType
from olmo_eval.evals.tasks import get_task, list_tasks
from olmo_eval.evals.tasks.basic_skills import (
    BASIC_SKILLS_SUBSETS,
    _make_basic_skills_config,
    _shuffle_and_insert,
)

# =============================================================================
# Shuffle and Insert Tests
# =============================================================================


class TestShuffleAndInsert:
    """Tests for the shuffle_and_insert function."""

    def test_inserts_correct_answer(self):
        """Test that correct answer is inserted into choices."""
        wrong_answers = ["wrong1", "wrong2", "wrong3"]
        correct = "correct"

        result, idx = _shuffle_and_insert(wrong_answers, correct, seed=42)

        assert len(result) == 4
        assert correct in result
        assert result[idx] == correct

    def test_preserves_all_wrong_answers(self):
        """Test that all wrong answers are preserved."""
        wrong_answers = ["a", "b", "c"]
        correct = "d"

        result, _ = _shuffle_and_insert(wrong_answers, correct, seed=123)

        for wrong in wrong_answers:
            assert wrong in result

    def test_deterministic_with_same_seed(self):
        """Test that results are deterministic with same seed."""
        wrong_answers = ["x", "y", "z"]
        correct = "w"

        result1, idx1 = _shuffle_and_insert(wrong_answers, correct, seed=999)
        result2, idx2 = _shuffle_and_insert(wrong_answers, correct, seed=999)

        assert result1 == result2
        assert idx1 == idx2

    def test_different_seeds_may_differ(self):
        """Test that different seeds produce different results."""
        wrong_answers = ["a", "b", "c", "d", "e"]
        correct = "f"

        # With enough different seeds, we should get different orderings
        results = set()
        for seed in range(100):
            result, _ = _shuffle_and_insert(wrong_answers, correct, seed=seed)
            results.add(tuple(result))

        # Should have more than one unique ordering
        assert len(results) > 1


# =============================================================================
# BasicSkillsTask Tests
# =============================================================================


class TestBasicSkillsTask:
    """Tests for BasicSkillsTask."""

    @pytest.fixture
    def arithmetic_task(self):
        """Create a basic skills arithmetic task for testing."""
        return get_task("basic_skills_arithmetic")

    def test_process_doc(self, arithmetic_task):
        """Test processing a basic skills document."""
        doc = {
            "id": "test_001",
            "question": "What is 2 + 2?",
            "answer": "4",
            "wrong_answers": ["3", "5", "6"],
        }

        instance = arithmetic_task.process_doc(doc)

        assert isinstance(instance, Instance)
        assert instance.question == "What is 2 + 2?"
        assert len(instance.choices) == 4
        assert "4" in instance.choices
        # Gold answer should be the letter at gold_idx
        gold_idx = instance.metadata["gold_idx"]
        assert instance.gold_answer == chr(ord("A") + gold_idx)
        assert instance.choices[gold_idx] == "4"

    def test_format_request(self, arithmetic_task):
        """Test request formatting."""
        instance = Instance(
            question="What is 3 + 3?",
            gold_answer="B",
            choices=("5", "6", "7", "8"),
        )

        request = arithmetic_task.format_request(instance)

        assert request.request_type == RequestType.COMPLETION
        assert "What is 3 + 3?" in request.prompt
        assert "A." in request.prompt
        assert "B." in request.prompt
        assert len(request.continuations) == 4

    def test_extract_answer(self, arithmetic_task):
        """Test answer extraction."""
        output = LMOutput(text="The answer is B.")

        answer = arithmetic_task.extract_answer(output)

        assert answer == "B"

    def test_config_name(self):
        """Test config naming."""
        config = _make_basic_skills_config("arithmetic")

        assert config.name == "basic_skills_arithmetic"
        assert config.data_source.path == "allenai/basic-skills"
        assert config.data_source.subset == "arithmetic"


# =============================================================================
# Task Registration Tests
# =============================================================================


class TestBasicSkillsRegistration:
    """Tests for basic skills task registration."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Ensure tasks are registered by importing module."""
        import olmo_eval.evals.tasks.basic_skills  # noqa: F401

        yield

    def test_all_6_tasks_registered(self):
        """Test that all 6 basic skills tasks are registered."""
        all_tasks = list_tasks()
        for subset in BASIC_SKILLS_SUBSETS:
            full_name = f"basic_skills_{subset}"
            assert full_name in all_tasks, f"{full_name} not registered"

    def test_get_arithmetic(self):
        """Test getting arithmetic task."""
        task = get_task("basic_skills_arithmetic")
        assert task is not None
        assert task.subset == "arithmetic"

    def test_get_coding(self):
        """Test getting coding task."""
        task = get_task("basic_skills_coding")
        assert task is not None
        assert task.subset == "coding"

    def test_get_common_knowledge(self):
        """Test getting common_knowledge task."""
        task = get_task("basic_skills_common_knowledge")
        assert task is not None
        assert task.subset == "common_knowledge"

    def test_get_logical_reasoning(self):
        """Test getting logical_reasoning task."""
        task = get_task("basic_skills_logical_reasoning")
        assert task is not None
        assert task.subset == "logical_reasoning"

    def test_get_string_operations(self):
        """Test getting string_operations task."""
        task = get_task("basic_skills_string_operations")
        assert task is not None
        assert task.subset == "string_operations"

    def test_get_pattern(self):
        """Test getting pattern task."""
        task = get_task("basic_skills_pattern")
        assert task is not None
        assert task.subset == "pattern"

    def test_task_has_correct_hf_path(self):
        """Test that tasks have correct HuggingFace path."""
        task = get_task("basic_skills_arithmetic")
        assert task.default_hf_path == "allenai/basic-skills"


# =============================================================================
# Answer Extraction Tests
# =============================================================================


class TestBasicSkillsAnswerExtraction:
    """Tests for answer extraction across different formats."""

    @pytest.fixture
    def task(self):
        """Create a basic skills task for testing."""
        return get_task("basic_skills_arithmetic")

    def test_extract_single_letter(self, task):
        """Test extracting single letter answer."""
        output = LMOutput(text="A")
        assert task.extract_answer(output) == "A"

    def test_extract_lowercase_letter(self, task):
        """Test extracting lowercase letter answer (normalized to uppercase)."""
        output = LMOutput(text="b")
        assert task.extract_answer(output) == "B"

    def test_extract_from_sentence(self, task):
        """Test extracting answer from sentence."""
        output = LMOutput(text="I think the answer is C because...")
        assert task.extract_answer(output) == "C"

    def test_extract_with_period(self, task):
        """Test extracting answer with period."""
        output = LMOutput(text="D.")
        assert task.extract_answer(output) == "D"
