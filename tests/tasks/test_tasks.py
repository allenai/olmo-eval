"""Tests for all task module registrations and basic functionality."""

import pytest

from olmo_eval.core import Instance, LMOutput, RequestType
from olmo_eval.tasks import get_task, list_tasks


class TestTaskRegistration:
    """Tests for task registration across all modules."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Ensure tasks are registered by importing modules."""
        import olmo_eval.tasks  # noqa: F401

        yield

    # HellaSwag
    def test_hellaswag_registered(self):
        """Test that hellaswag is registered."""
        assert "hellaswag" in list_tasks()

    def test_get_hellaswag(self):
        """Test getting hellaswag task."""
        task = get_task("hellaswag")
        assert task.config.name == "hellaswag"

    # MedMCQA
    def test_medmcqa_registered(self):
        """Test that medmcqa is registered."""
        assert "medmcqa" in list_tasks()

    def test_get_medmcqa(self):
        """Test getting medmcqa task."""
        task = get_task("medmcqa")
        assert task.config.name == "medmcqa"

    # MMLU (sample subsets)
    def test_mmlu_abstract_algebra_registered(self):
        """Test that mmlu_abstract_algebra is registered."""
        assert "mmlu_abstract_algebra" in list_tasks()

    def test_mmlu_anatomy_registered(self):
        """Test that mmlu_anatomy is registered."""
        assert "mmlu_anatomy" in list_tasks()

    def test_get_mmlu_abstract_algebra(self):
        """Test getting mmlu_abstract_algebra task."""
        task = get_task("mmlu_abstract_algebra")
        assert task.config.name == "mmlu_abstract_algebra"

    # MMLU-Pro (sample subsets)
    def test_mmlu_pro_math_registered(self):
        """Test that mmlu_pro_math is registered."""
        assert "mmlu_pro_math" in list_tasks()

    def test_get_mmlu_pro_math(self):
        """Test getting mmlu_pro_math task."""
        task = get_task("mmlu_pro_math")
        assert task.config.name == "mmlu_pro_math"

    # GSM tasks
    def test_gsm8k_registered(self):
        """Test that gsm8k is registered."""
        assert "gsm8k" in list_tasks()

    def test_gsm_plus_registered(self):
        """Test that gsm_plus is registered."""
        assert "gsm_plus" in list_tasks()

    def test_gsm_symbolic_registered(self):
        """Test that gsm_symbolic is registered."""
        assert "gsm_symbolic" in list_tasks()

    def test_get_gsm8k(self):
        """Test getting gsm8k task."""
        task = get_task("gsm8k")
        assert task.config.name == "gsm8k"

    # Minerva Math
    def test_minerva_math_algebra_registered(self):
        """Test that minerva_math_algebra is registered."""
        assert "minerva_math_algebra" in list_tasks()

    def test_math500_registered(self):
        """Test that math500 is registered."""
        assert "math500" in list_tasks()

    def test_get_minerva_math_algebra(self):
        """Test getting minerva_math_algebra task."""
        task = get_task("minerva_math_algebra")
        assert task.config.name == "minerva_math_algebra"

    # AIME
    def test_aime_registered(self):
        """Test that aime is registered."""
        assert "aime" in list_tasks()

    def test_aime_2024_registered(self):
        """Test that aime_2024 is registered."""
        assert "aime_2024" in list_tasks()

    def test_get_aime(self):
        """Test getting aime task."""
        task = get_task("aime")
        assert task.config.name == "aime"

    # AGI-Eval (sample subsets)
    def test_agi_eval_lsat_ar_registered(self):
        """Test that agi_eval_lsat_ar is registered."""
        assert "agi_eval_lsat_ar" in list_tasks()

    def test_agi_eval_sat_math_registered(self):
        """Test that agi_eval_sat_math is registered."""
        assert "agi_eval_sat_math" in list_tasks()

    def test_get_agi_eval_lsat_ar(self):
        """Test getting agi_eval_lsat_ar task."""
        task = get_task("agi_eval_lsat_ar")
        assert task.config.name == "agi_eval_lsat_ar"

    # GPQA
    def test_gpqa_registered(self):
        """Test that gpqa is registered."""
        assert "gpqa" in list_tasks()

    def test_gpqa_diamond_registered(self):
        """Test that gpqa_diamond is registered."""
        assert "gpqa_diamond" in list_tasks()

    def test_super_gpqa_registered(self):
        """Test that super_gpqa is registered."""
        assert "super_gpqa" in list_tasks()

    def test_get_gpqa(self):
        """Test getting gpqa task."""
        task = get_task("gpqa")
        assert task.config.name == "gpqa"

    # QA Tasks
    def test_drop_registered(self):
        """Test that drop is registered."""
        assert "drop" in list_tasks()

    def test_coqa_registered(self):
        """Test that coqa is registered."""
        assert "coqa" in list_tasks()

    def test_squad_registered(self):
        """Test that squad is registered."""
        assert "squad" in list_tasks()

    def test_naturalqs_registered(self):
        """Test that naturalqs is registered."""
        assert "naturalqs" in list_tasks()

    def test_jeopardy_registered(self):
        """Test that jeopardy is registered."""
        assert "jeopardy" in list_tasks()

    def test_get_drop(self):
        """Test getting drop task."""
        task = get_task("drop")
        assert task.config.name == "drop"

    # Code Tasks
    def test_humaneval_registered(self):
        """Test that humaneval is registered."""
        assert "humaneval" in list_tasks()

    def test_humaneval_plus_registered(self):
        """Test that humaneval_plus is registered."""
        assert "humaneval_plus" in list_tasks()

    def test_mbpp_registered(self):
        """Test that mbpp is registered."""
        assert "mbpp" in list_tasks()

    def test_mbpp_plus_registered(self):
        """Test that mbpp_plus is registered."""
        assert "mbpp_plus" in list_tasks()

    def test_get_humaneval(self):
        """Test getting humaneval task."""
        task = get_task("humaneval")
        assert task.config.name == "humaneval"

    # WikiText
    def test_wikitext_registered(self):
        """Test that wikitext is registered."""
        assert "wikitext" in list_tasks()

    def test_wikitext2_registered(self):
        """Test that wikitext2 is registered."""
        assert "wikitext2" in list_tasks()

    def test_get_wikitext(self):
        """Test getting wikitext task."""
        task = get_task("wikitext")
        assert task.config.name == "wikitext"


class TestHellaSwagTask:
    """Tests for HellaSwag task functionality."""

    @pytest.fixture
    def task(self):
        """Create a HellaSwag task."""
        return get_task("hellaswag")

    def test_format_request(self, task):
        """Test request formatting."""
        instance = Instance(
            question="A person is walking down the street.",
            gold_answer="0",
            choices=(
                "They continue walking.",
                "They start running.",
                "They sit down.",
                "They fly away.",
            ),
        )

        request = task.format_request(instance)
        assert request.request_type == RequestType.COMPLETION
        assert "walking down the street" in request.prompt
        assert request.continuations is not None
        assert len(request.continuations) == 4


class TestGSMTask:
    """Tests for GSM8K task functionality."""

    @pytest.fixture
    def task(self):
        """Create a GSM8K task."""
        return get_task("gsm8k")

    def test_format_request(self, task):
        """Test request formatting."""
        instance = Instance(
            question="What is 2 + 2?",
            gold_answer="4",
        )

        request = task.format_request(instance)
        assert request.request_type == RequestType.COMPLETION
        assert "2 + 2" in request.prompt

    def test_extract_answer(self, task):
        """Test answer extraction."""
        output = LMOutput(text="Let me solve this step by step. 2 + 2 = 4")
        answer = task.extract_answer(output)
        assert answer == "4"


class TestMMLUTask:
    """Tests for MMLU task functionality."""

    @pytest.fixture
    def task(self):
        """Create an MMLU task."""
        return get_task("mmlu_abstract_algebra")

    def test_format_request(self, task):
        """Test request formatting."""
        instance = Instance(
            question="What is a group in abstract algebra?",
            gold_answer="A",
            choices=(
                "A set with a binary operation",
                "A set of numbers",
                "A vector space",
                "A matrix",
            ),
        )

        request = task.format_request(instance)
        assert request.request_type == RequestType.COMPLETION
        assert "group" in request.prompt.lower()

    def test_extract_answer(self, task):
        """Test answer extraction."""
        output = LMOutput(text="The answer is A")
        answer = task.extract_answer(output)
        assert answer == "A"


class TestQATask:
    """Tests for QA task functionality."""

    @pytest.fixture
    def drop_task(self):
        """Create a DROP task."""
        return get_task("drop")

    def test_format_request(self, drop_task):
        """Test request formatting."""
        instance = Instance(
            question="Passage: The sky is blue.\nWhat color is the sky?",
            gold_answer="blue",
        )

        request = drop_task.format_request(instance)
        assert request.request_type == RequestType.COMPLETION
        assert "sky" in request.prompt

    def test_extract_answer(self, drop_task):
        """Test answer extraction."""
        output = LMOutput(text="blue")
        answer = drop_task.extract_answer(output)
        assert answer == "blue"


class TestGPQATask:
    """Tests for GPQA task functionality."""

    @pytest.fixture
    def task(self):
        """Create a GPQA task."""
        return get_task("gpqa")

    def test_format_request(self, task):
        """Test request formatting."""
        instance = Instance(
            question="What is the mass of a proton?",
            gold_answer="A",
            choices=(
                "1.67 × 10^-27 kg",
                "9.11 × 10^-31 kg",
                "1.99 × 10^-26 kg",
                "6.63 × 10^-34 kg",
            ),
        )

        request = task.format_request(instance)
        assert request.request_type == RequestType.COMPLETION
        assert "proton" in request.prompt

    def test_extract_answer(self, task):
        """Test answer extraction."""
        output = LMOutput(text="(A)")
        answer = task.extract_answer(output)
        assert answer == "A"


class TestCodeTask:
    """Tests for code generation task functionality."""

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


class TestWikiTextTask:
    """Tests for WikiText task functionality."""

    @pytest.fixture
    def task(self):
        """Create a WikiText task."""
        return get_task("wikitext")

    def test_format_request(self, task):
        """Test request formatting for perplexity."""
        instance = Instance(
            question="",
            gold_answer="The quick brown fox jumps over the lazy dog.",
        )

        request = task.format_request(instance)
        assert request.request_type == RequestType.LOGLIKELIHOOD
        assert "fox" in request.prompt
