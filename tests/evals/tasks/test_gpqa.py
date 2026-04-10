"""Tests for GPQA task logic."""

import pytest

from olmo_eval.common.types import Instance, LMOutput, RequestType
from olmo_eval.evals.tasks.common import get_task, list_tasks


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


_ALL_TASKS = (
    "gpqa_diamond",
    "gpqa_main",
    "gpqa_extended",
)

_SUBJECT_TASKS = (
    "gpqa_diamond_biology",
    "gpqa_diamond_chemistry",
    "gpqa_diamond_physics",
    "gpqa_main_biology",
    "gpqa_main_chemistry",
    "gpqa_main_physics",
    "gpqa_extended_biology",
    "gpqa_extended_chemistry",
    "gpqa_extended_physics",
)


class TestGPQARegistration:
    """Tests for GPQA task registration."""

    @pytest.mark.parametrize("task_name", _ALL_TASKS + _SUBJECT_TASKS)
    def test_task_registered(self, task_name):
        assert task_name in list_tasks()

    @pytest.mark.parametrize("task_name", _ALL_TASKS + _SUBJECT_TASKS)
    def test_get_task(self, task_name):
        task = get_task(task_name)
        assert task.config.name == task_name

    @pytest.mark.parametrize("task_name", _ALL_TASKS + _SUBJECT_TASKS)
    def test_mc_variant(self, task_name):
        task = get_task(f"{task_name}:mc")
        assert task is not None

    @pytest.mark.parametrize("task_name", _ALL_TASKS + _SUBJECT_TASKS)
    def test_bpb_variant(self, task_name):
        task = get_task(f"{task_name}:bpb")
        assert task is not None


class TestProcessDoc:
    """Tests for GPQATask.process_doc."""

    @pytest.fixture
    def task(self):
        return get_task("gpqa_diamond")

    def test_basic_conversion(self, task):
        doc = {
            "Question": "What is the ground state of hydrogen?",
            "Correct Answer": "1s",
            "Incorrect Answer 1": "2s",
            "Incorrect Answer 2": "2p",
            "Incorrect Answer 3": "3s",
        }
        instance = task.process_doc(doc, index=0)

        assert instance is not None
        assert "ground state of hydrogen" in instance.question
        assert len(instance.choices) == 4
        assert "1s" in instance.choices
        assert "2s" in instance.choices
        assert "2p" in instance.choices
        assert "3s" in instance.choices

    def test_gold_letter_points_to_correct(self, task):
        doc = {
            "Question": "What is the answer?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong 1",
            "Incorrect Answer 2": "Wrong 2",
            "Incorrect Answer 3": "Wrong 3",
        }
        instance = task.process_doc(doc, index=42)

        gold_idx = instance.metadata["gold_idx"]
        assert instance.choices[gold_idx] == "Correct"
        assert instance.gold_answer == chr(ord("A") + gold_idx)

    def test_shuffle_determinism(self, task):
        doc = {
            "Question": "Q?",
            "Correct Answer": "Right",
            "Incorrect Answer 1": "W1",
            "Incorrect Answer 2": "W2",
            "Incorrect Answer 3": "W3",
        }
        inst1 = task.process_doc(doc, index=7)
        inst2 = task.process_doc(doc, index=7)
        assert inst1.choices == inst2.choices
        assert inst1.gold_answer == inst2.gold_answer

    def test_different_indices_produce_different_shuffles(self, task):
        doc = {
            "Question": "Q?",
            "Correct Answer": "Right",
            "Incorrect Answer 1": "W1",
            "Incorrect Answer 2": "W2",
            "Incorrect Answer 3": "W3",
        }
        inst0 = task.process_doc(doc, index=0)
        inst1 = task.process_doc(doc, index=1)
        assert inst0.choices != inst1.choices

    def test_metadata_contains_gold_idx_and_text(self, task):
        doc = {
            "Question": "Q?",
            "Correct Answer": "The right answer",
            "Incorrect Answer 1": "Wrong",
            "Incorrect Answer 2": "Also wrong",
            "Incorrect Answer 3": "Still wrong",
        }
        instance = task.process_doc(doc, index=0)
        assert "gold_idx" in instance.metadata
        assert instance.metadata["gold_text"] == "The right answer"

    def test_explanation_in_metadata(self, task):
        doc = {
            "Question": "Q?",
            "Correct Answer": "A",
            "Incorrect Answer 1": "B",
            "Incorrect Answer 2": "C",
            "Incorrect Answer 3": "D",
            "Explanation": "Because physics.",
        }
        instance = task.process_doc(doc, index=0)
        assert instance.metadata["explanation"] == "Because physics."

    def test_no_explanation_omitted(self, task):
        doc = {
            "Question": "Q?",
            "Correct Answer": "A",
            "Incorrect Answer 1": "B",
            "Incorrect Answer 2": "C",
            "Incorrect Answer 3": "D",
        }
        instance = task.process_doc(doc, index=0)
        assert "explanation" not in instance.metadata

    def test_skip_missing_question(self, task):
        doc = {
            "Question": "",
            "Correct Answer": "A",
            "Incorrect Answer 1": "B",
            "Incorrect Answer 2": "C",
            "Incorrect Answer 3": "D",
        }
        assert task.process_doc(doc, index=0) is None

    def test_skip_missing_correct_answer(self, task):
        doc = {
            "Question": "Q?",
            "Correct Answer": "",
            "Incorrect Answer 1": "B",
            "Incorrect Answer 2": "C",
            "Incorrect Answer 3": "D",
        }
        assert task.process_doc(doc, index=0) is None

    def test_text_preprocessing_only_strips_title_marker(self, task):
        doc = {
            "Question": "According to [title] some paper [2023], what is X?",
            "Correct Answer": "Answer [ref]",
            "Incorrect Answer 1": "Wrong",
            "Incorrect Answer 2": "Also wrong",
            "Incorrect Answer 3": "Still wrong",
        }
        instance = task.process_doc(doc, index=0)
        assert "[title]" not in instance.question
        assert "[2023]" in instance.question
        assert instance.metadata["gold_text"] == "Answer [ref]"

    def test_text_preprocessing_double_spaces(self, task):
        doc = {
            "Question": "What  is  the  answer?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong",
            "Incorrect Answer 2": "Also wrong",
            "Incorrect Answer 3": "Still wrong",
        }
        instance = task.process_doc(doc, index=0)
        assert "  " not in instance.question

    @pytest.mark.parametrize(
        ("task_name", "subdomain"),
        (
            ("gpqa_diamond_biology", "Molecular Biology"),
            ("gpqa_diamond_biology", "Genetics"),
            ("gpqa_diamond_chemistry", "Chemistry (general)"),
            ("gpqa_diamond_chemistry", "Analytical Chemistry"),
            ("gpqa_diamond_chemistry", "Organic Chemistry"),
            ("gpqa_diamond_physics", "Physics (general)"),
            ("gpqa_diamond_physics", "Quantum Mechanics"),
            ("gpqa_diamond_physics", "Statistical Mechanics"),
        ),
    )
    def test_subject_task_accepts_mapped_subdomains(self, task_name, subdomain):
        task = get_task(task_name)
        doc = {
            "Question": "Q?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong 1",
            "Incorrect Answer 2": "Wrong 2",
            "Incorrect Answer 3": "Wrong 3",
            "Subdomain": subdomain,
        }
        instance = task.process_doc(doc, index=0)
        assert instance is not None
        assert instance.metadata["subdomain"] == subdomain

    def test_subject_task_rejects_non_matching_subdomain(self):
        task = get_task("gpqa_diamond_physics")
        doc = {
            "Question": "Q?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong 1",
            "Incorrect Answer 2": "Wrong 2",
            "Incorrect Answer 3": "Wrong 3",
            "Subdomain": "Molecular Biology",
        }
        assert task.process_doc(doc, index=0) is None

    def test_subject_task_warns_for_unmapped_subdomain(self, caplog):
        task = get_task("gpqa_diamond_physics")
        doc = {
            "Question": "Q?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong 1",
            "Incorrect Answer 2": "Wrong 2",
            "Incorrect Answer 3": "Wrong 3",
            "Subdomain": "Unknown Subdomain",
        }
        with caplog.at_level("WARNING"):
            assert task.process_doc(doc, index=0) is None
        assert "unmapped subdomain" in caplog.text


class TestExtractAnswer:
    """Tests for GPQATask.extract_answer."""

    @pytest.fixture
    def task(self):
        return get_task("gpqa_diamond")

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

    def test_fallback_parenthesized(self, task):
        output = LMOutput(text="I think the answer is (B)")
        assert task.extract_answer(output) == "B"

    def test_fallback_parenthesized_last_wins(self, task):
        output = LMOutput(text="Maybe (A) but actually (D)")
        assert task.extract_answer(output) == "D"

    def test_fallback_standalone_letter(self, task):
        output = LMOutput(text="The correct choice is D")
        assert task.extract_answer(output) == "D"

    def test_answer_pattern_takes_priority_over_paren(self, task):
        output = LMOutput(text="I think (A) but ANSWER: C")
        assert task.extract_answer(output) == "C"

    def test_no_match_returns_none(self, task):
        output = LMOutput(text="I have no idea what the answer is")
        assert task.extract_answer(output) is None

    def test_empty_output(self, task):
        output = LMOutput(text="")
        assert task.extract_answer(output) is None

    def test_whitespace_only(self, task):
        output = LMOutput(text="   ")
        assert task.extract_answer(output) is None


class TestFormatRequest:
    """Tests for format_request delegation."""

    def test_chat_format(self):
        task = get_task("gpqa_diamond")
        instance = Instance(
            question="What is the ground state?",
            gold_answer="A",
            choices=("1s", "2s", "2p", "3s"),
            metadata={"gold_idx": 0, "gold_text": "1s"},
        )
        request = task.format_request(instance)
        assert request.request_type == RequestType.CHAT
        assert any("1s" in m["content"] for m in request.messages if m["role"] == "user")

    def test_mc_format(self):
        task = get_task("gpqa_diamond:mc")
        instance = Instance(
            question="What is the ground state?",
            gold_answer="A",
            choices=("1s", "2s", "2p", "3s"),
            metadata={"gold_idx": 0, "gold_text": "1s"},
        )
        request = task.format_request(instance)
        assert request.request_type == RequestType.LOGLIKELIHOOD
        assert request.continuations is not None
        assert len(request.continuations) == 4
