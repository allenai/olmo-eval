"""Tests for MedQA task logic."""

import pytest

from olmo_eval.common.types import Instance, LMOutput, RequestType
from olmo_eval.evals.tasks.common import get_task, list_tasks


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


class TestMedQARegistration:
    """Tests for MedQA task registration."""

    @pytest.mark.parametrize("task_name", ["medqa", "medqa:mc", "medqa:bpb"])
    def test_task_registered(self, task_name):
        base = task_name.split(":")[0]
        assert base in list_tasks()

    @pytest.mark.parametrize("task_name", ["medqa", "medqa:mc", "medqa:bpb"])
    def test_get_task(self, task_name):
        task = get_task(task_name)
        assert task is not None


class TestProcessDoc:
    """Tests for MedQA.process_doc."""

    @pytest.fixture
    def task(self):
        return get_task("medqa")

    def test_basic_conversion(self, task):
        doc = {
            "question": "What is the most likely diagnosis?",
            "choices": ["Diabetes", "Hypertension", "Asthma", "COPD"],
            "answer_idx": 0,
        }
        instance = task.process_doc(doc, index=0)

        assert instance is not None
        assert instance.question == "What is the most likely diagnosis?"
        assert len(instance.choices) == 4
        assert "Diabetes" in instance.choices

    def test_gold_letter_points_to_answer(self, task):
        doc = {
            "question": "What is the answer?",
            "choices": ["A-ans", "B-ans", "C-ans", "D-ans"],
            "answer_idx": 2,
        }
        instance = task.process_doc(doc, index=42)

        gold_idx = instance.metadata["gold_idx"]
        assert instance.choices[gold_idx] == "C-ans"
        assert instance.gold_answer == chr(ord("A") + gold_idx)

    def test_shuffle_determinism(self, task):
        doc = {
            "question": "Q?",
            "choices": ["A", "B", "C", "D"],
            "answer_idx": 0,
        }
        inst1 = task.process_doc(doc, index=7)
        inst2 = task.process_doc(doc, index=7)
        assert inst1.choices == inst2.choices
        assert inst1.gold_answer == inst2.gold_answer

    def test_skip_missing_question(self, task):
        doc = {"question": "", "choices": ["A", "B"], "answer_idx": 0}
        assert task.process_doc(doc, index=0) is None

    def test_skip_missing_choices(self, task):
        doc = {"question": "Q?", "choices": [], "answer_idx": 0}
        assert task.process_doc(doc, index=0) is None

    def test_skip_missing_answer_idx(self, task):
        doc = {"question": "Q?", "choices": ["A", "B"]}
        assert task.process_doc(doc, index=0) is None

    def test_skip_negative_answer_idx(self, task):
        doc = {"question": "Q?", "choices": ["A", "B"], "answer_idx": -1}
        assert task.process_doc(doc, index=0) is None

    def test_skip_out_of_range_answer_idx(self, task):
        doc = {"question": "Q?", "choices": ["A", "B"], "answer_idx": 5}
        assert task.process_doc(doc, index=0) is None

    def test_skip_non_int_answer_idx(self, task):
        doc = {"question": "Q?", "choices": ["A", "B"], "answer_idx": "0"}
        assert task.process_doc(doc, index=0) is None

    def test_duplicate_choices_tracks_correct_gold(self, task):
        import random

        doc = {
            "question": "Q?",
            "choices": ["Same", "Same", "Different"],
            "answer_idx": 1,
        }
        instance = task.process_doc(doc, index=0)
        assert instance is not None

        # Reproduce shuffle to verify gold tracks original position, not first text match
        paired = list(zip(doc["choices"], range(len(doc["choices"])), strict=True))
        rng = random.Random(f"{task.config.seed}:0")
        rng.shuffle(paired)
        expected_gold_idx = next(
            i for i, (_, orig) in enumerate(paired) if orig == doc["answer_idx"]
        )
        assert instance.metadata["gold_idx"] == expected_gold_idx

    def test_metadata_contains_gold_idx_and_text(self, task):
        doc = {
            "question": "Q?",
            "choices": ["Wrong", "Right"],
            "answer_idx": 1,
        }
        instance = task.process_doc(doc, index=0)
        assert "gold_idx" in instance.metadata
        assert instance.metadata["gold_text"] == "Right"


class TestExtractAnswer:
    """Tests for MedQA.extract_answer."""

    @pytest.fixture
    def task(self):
        return get_task("medqa")

    def test_answer_pattern(self, task):
        output = LMOutput(text="The answer is clearly B because...\nANSWER: B")
        assert task.extract_answer(output) == "B"

    def test_answer_pattern_lowercase(self, task):
        output = LMOutput(text="answer: c")
        assert task.extract_answer(output) == "C"

    def test_answer_pattern_no_space(self, task):
        output = LMOutput(text="ANSWER:A")
        assert task.extract_answer(output) == "A"

    def test_last_answer_pattern_wins(self, task):
        output = LMOutput(text="ANSWER: A\nWait, actually ANSWER: C")
        assert task.extract_answer(output) == "C"

    def test_boxed_letter(self, task):
        output = LMOutput(text="Therefore...\n$$\\boxed{B}$$")
        assert task.extract_answer(output) == "B"

    def test_boxed_text(self, task):
        output = LMOutput(text="$$\\boxed{\\text{A}}$$")
        assert task.extract_answer(output) == "A"

    def test_paren_letter(self, task):
        output = LMOutput(text="**Final Answer:**\n**(C) Diabetes**")
        assert task.extract_answer(output) == "C"

    def test_last_paren_letter_wins(self, task):
        output = LMOutput(text="Option (A) is wrong.\n\n**Answer: (D) Correct**")
        assert task.extract_answer(output) == "D"

    def test_answer_pattern_preferred_over_boxed(self, task):
        output = LMOutput(text="\\boxed{A}\nANSWER: B")
        assert task.extract_answer(output) == "B"

    def test_boxed_preferred_over_paren(self, task):
        output = LMOutput(text="(A) is likely\n$$\\boxed{C}$$")
        assert task.extract_answer(output) == "C"

    def test_no_pattern_returns_none(self, task):
        output = LMOutput(text="I think the answer is B")
        assert task.extract_answer(output) is None

    def test_empty_output(self, task):
        output = LMOutput(text="")
        assert task.extract_answer(output) is None


class TestFormatRequest:
    """Tests for format_request delegation."""

    def test_chat_format(self):
        task = get_task("medqa")
        instance = Instance(
            question="What is the diagnosis?",
            gold_answer="A",
            choices=("Diabetes", "Hypertension"),
            metadata={"gold_idx": 0, "gold_text": "Diabetes"},
        )
        request = task.format_request(instance)
        assert request.request_type == RequestType.CHAT
        assert any("Diabetes" in m["content"] for m in request.messages if m["role"] == "user")

    def test_mc_format(self):
        task = get_task("medqa:mc")
        instance = Instance(
            question="What is the diagnosis?",
            gold_answer="A",
            choices=("Diabetes", "Hypertension"),
            metadata={"gold_idx": 0, "gold_text": "Diabetes"},
        )
        request = task.format_request(instance)
        assert request.request_type == RequestType.LOGLIKELIHOOD
        assert request.continuations is not None
        assert len(request.continuations) == 2
