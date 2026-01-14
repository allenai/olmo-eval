"""Tests for olmo_eval.tasks.core_tasks module."""

import pytest

from olmo_eval.core import Instance, RequestType
from olmo_eval.evals.tasks.core_tasks import (
    CSQA,
    PIQA,
    BoolQ,
    OpenBookQA,
    SocialIQA,
    WinoGrande,
    _boolq_config,
    _csqa_config,
    _openbookqa_config,
    _piqa_config,
    _socialiqa_config,
    _winogrande_config,
)
from olmo_eval.evals.tasks.registry import get_task, list_tasks

# =============================================================================
# BoolQ Tests
# =============================================================================


class TestBoolQTask:
    """Tests for BoolQ task."""

    @pytest.fixture
    def boolq_task(self):
        """Create a BoolQ task for testing."""
        config = _boolq_config()
        return BoolQ(config)

    def test_process_doc_yes_answer(self, boolq_task):
        """Test processing doc with yes (True) answer."""
        doc = {
            "idx": 1,
            "question": "is this a test",
            "passage": "This is a test passage.",
            "label": True,
        }

        instance = boolq_task._process_doc(doc)

        assert isinstance(instance, Instance)
        assert "This is a test passage." in instance.question
        assert "is this a test?" in instance.question
        assert instance.gold_answer == "A"  # yes = A
        assert instance.choices == ("yes", "no")
        assert instance.metadata["gold_idx"] == 0

    def test_process_doc_no_answer(self, boolq_task):
        """Test processing doc with no (False) answer."""
        doc = {
            "idx": 2,
            "question": "is the sky green",
            "passage": "The sky is blue.",
            "label": False,
        }

        instance = boolq_task._process_doc(doc)

        assert instance.gold_answer == "B"  # no = B
        assert instance.metadata["gold_idx"] == 1

    def test_format_request(self, boolq_task):
        """Test request formatting."""
        instance = Instance(
            question="Test passage.\nQuestion: Is this correct?",
            gold_answer="A",
            choices=("yes", "no"),
        )

        request = boolq_task.format_request(instance)

        assert request.request_type == RequestType.COMPLETION
        assert "Is this correct?" in request.prompt

    def test_config(self):
        """Test BoolQ config."""
        config = _boolq_config()
        assert config.name == "boolq"
        assert config.hf_dataset == "super_glue"


# =============================================================================
# CSQA Tests
# =============================================================================


class TestCSQATask:
    """Tests for CommonsenseQA task."""

    @pytest.fixture
    def csqa_task(self):
        """Create a CSQA task for testing."""
        config = _csqa_config()
        return CSQA(config)

    def test_process_doc(self, csqa_task):
        """Test processing a CSQA document."""
        doc = {
            "id": "test_001",
            "question": "Where would you find a dog?",
            "answerKey": "B",
            "choices": {
                "text": ["ocean", "park", "space", "volcano", "desert"],
                "label": ["A", "B", "C", "D", "E"],
            },
        }

        instance = csqa_task._process_doc(doc)

        assert isinstance(instance, Instance)
        assert instance.question == "Where would you find a dog?"
        assert instance.gold_answer == "B"
        assert len(instance.choices) == 5
        assert instance.metadata["gold_idx"] == 1
        assert instance.metadata["gold_text"] == "park"

    def test_format_request_five_choices(self, csqa_task):
        """Test formatting with 5 choices."""
        instance = Instance(
            question="Test question?",
            gold_answer="C",
            choices=("A1", "B2", "C3", "D4", "E5"),
        )

        request = csqa_task.format_request(instance)

        assert request.request_type == RequestType.COMPLETION
        assert len(request.continuations) == 5

    def test_config(self):
        """Test CSQA config."""
        config = _csqa_config()
        assert config.name == "csqa"
        assert config.hf_dataset == "commonsense_qa"


# =============================================================================
# OpenBookQA Tests
# =============================================================================


class TestOpenBookQATask:
    """Tests for OpenBookQA task."""

    @pytest.fixture
    def openbookqa_task(self):
        """Create an OpenBookQA task for testing."""
        config = _openbookqa_config()
        return OpenBookQA(config)

    def test_process_doc(self, openbookqa_task):
        """Test processing an OpenBookQA document."""
        doc = {
            "id": "test_001",
            "question_stem": "What happens when ice melts?",
            "answerKey": "C",
            "choices": {
                "text": ["It freezes", "It evaporates", "It becomes water", "It disappears"],
                "label": ["A", "B", "C", "D"],
            },
        }

        instance = openbookqa_task._process_doc(doc)

        assert isinstance(instance, Instance)
        assert instance.question == "What happens when ice melts?"
        assert instance.gold_answer == "C"
        assert len(instance.choices) == 4
        assert instance.metadata["gold_idx"] == 2

    def test_config(self):
        """Test OpenBookQA config."""
        config = _openbookqa_config()
        assert config.name == "openbookqa"
        assert config.hf_dataset == "openbookqa"


# =============================================================================
# PIQA Tests
# =============================================================================


class TestPIQATask:
    """Tests for PIQA task."""

    @pytest.fixture
    def piqa_task(self):
        """Create a PIQA task for testing."""
        config = _piqa_config()
        return PIQA(config)

    def test_process_doc(self, piqa_task):
        """Test processing a PIQA document."""
        doc = {
            "goal": "To make coffee",
            "sol1": "Add water to coffee grounds",
            "sol2": "Add sand to coffee grounds",
            "label": 0,
        }

        instance = piqa_task._process_doc(doc, index=0)

        assert isinstance(instance, Instance)
        assert instance.question == "To make coffee"
        assert instance.gold_answer == "A"  # label 0 = A
        assert instance.choices == ("Add water to coffee grounds", "Add sand to coffee grounds")
        assert instance.metadata["gold_idx"] == 0

    def test_process_doc_second_choice(self, piqa_task):
        """Test processing doc where second choice is correct."""
        doc = {
            "goal": "To open a jar",
            "sol1": "Push the lid",
            "sol2": "Twist the lid",
            "label": 1,
        }

        instance = piqa_task._process_doc(doc, index=0)

        assert instance.gold_answer == "B"  # label 1 = B
        assert instance.metadata["gold_idx"] == 1

    def test_config(self):
        """Test PIQA config."""
        config = _piqa_config()
        assert config.name == "piqa"
        assert config.hf_dataset == "piqa"


# =============================================================================
# SocialIQA Tests
# =============================================================================


class TestSocialIQATask:
    """Tests for SocialIQA task."""

    @pytest.fixture
    def socialiqa_task(self):
        """Create a SocialIQA task for testing."""
        config = _socialiqa_config()
        return SocialIQA(config)

    def test_process_doc(self, socialiqa_task):
        """Test processing a SocialIQA document."""
        doc = {
            "context": "Alex helped their friend move.",
            "question": "How would Alex feel?",
            "answerA": "Tired",
            "answerB": "Helpful",
            "answerC": "Angry",
            "label": "2",  # 1-indexed, so 2 = B
        }

        instance = socialiqa_task._process_doc(doc, index=0)

        assert isinstance(instance, Instance)
        assert "Alex helped their friend move." in instance.question
        assert "How would Alex feel?" in instance.question
        assert instance.gold_answer == "B"  # label "2" -> index 1 -> B
        assert len(instance.choices) == 3
        assert instance.metadata["gold_idx"] == 1

    def test_process_doc_first_choice(self, socialiqa_task):
        """Test processing doc with first choice correct."""
        doc = {
            "context": "Pat worked all day.",
            "question": "How would Pat feel?",
            "answerA": "Tired",
            "answerB": "Energetic",
            "answerC": "Happy",
            "label": "1",
        }

        instance = socialiqa_task._process_doc(doc, index=0)

        assert instance.gold_answer == "A"
        assert instance.metadata["gold_idx"] == 0

    def test_config(self):
        """Test SocialIQA config."""
        config = _socialiqa_config()
        assert config.name == "socialiqa"
        assert config.hf_dataset == "social_i_qa"


# =============================================================================
# WinoGrande Tests
# =============================================================================


class TestWinoGrandeTask:
    """Tests for WinoGrande task."""

    @pytest.fixture
    def winogrande_task(self):
        """Create a WinoGrande task for testing."""
        config = _winogrande_config()
        return WinoGrande(config)

    def test_process_doc(self, winogrande_task):
        """Test processing a WinoGrande document."""
        doc = {
            "sentence": "John couldn't lift his son because _ was so weak.",
            "option1": "John",
            "option2": "his son",
            "answer": "1",  # 1-indexed
        }

        instance = winogrande_task._process_doc(doc, index=0)

        assert isinstance(instance, Instance)
        assert "___" in instance.question  # underscore replaced with ___
        assert instance.gold_answer == "A"  # answer "1" -> A
        assert instance.choices == ("John", "his son")
        assert instance.metadata["gold_idx"] == 0

    def test_process_doc_second_option(self, winogrande_task):
        """Test processing doc with second option correct."""
        doc = {
            "sentence": "The trophy doesn't fit in the suitcase because _ is too big.",
            "option1": "the trophy",
            "option2": "the suitcase",
            "answer": "1",  # trophy is too big
        }

        instance = winogrande_task._process_doc(doc, index=0)

        assert instance.gold_answer == "A"

    def test_format_request(self, winogrande_task):
        """Test request formatting includes 'Fill in the blank'."""
        instance = Instance(
            question="John couldn't lift ___ because he was weak.",
            gold_answer="A",
            choices=("John", "the weight"),
        )

        request = winogrande_task.format_request(instance)

        assert "Fill in the blank:" in request.prompt

    def test_config(self):
        """Test WinoGrande config."""
        config = _winogrande_config()
        assert config.name == "winogrande"
        assert config.hf_dataset == "winogrande"
        assert config.hf_subsets == ("winogrande_xl",)


# =============================================================================
# Task Registration Tests
# =============================================================================


class TestCoreTasksRegistration:
    """Tests for core task registration."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Ensure tasks are registered by importing module."""
        import olmo_eval.evals.tasks.core_tasks  # noqa: F401

        yield

    def test_boolq_registered(self):
        """Test that boolq is registered."""
        assert "boolq" in list_tasks()

    def test_csqa_registered(self):
        """Test that csqa is registered."""
        assert "csqa" in list_tasks()

    def test_openbookqa_registered(self):
        """Test that openbookqa is registered."""
        assert "openbookqa" in list_tasks()

    def test_piqa_registered(self):
        """Test that piqa is registered."""
        assert "piqa" in list_tasks()

    def test_socialiqa_registered(self):
        """Test that socialiqa is registered."""
        assert "socialiqa" in list_tasks()

    def test_winogrande_registered(self):
        """Test that winogrande is registered."""
        assert "winogrande" in list_tasks()

    def test_get_boolq(self):
        """Test getting boolq task."""
        task = get_task("boolq")
        assert isinstance(task, BoolQ)

    def test_get_csqa(self):
        """Test getting csqa task."""
        task = get_task("csqa")
        assert isinstance(task, CSQA)

    def test_get_openbookqa(self):
        """Test getting openbookqa task."""
        task = get_task("openbookqa")
        assert isinstance(task, OpenBookQA)

    def test_get_piqa(self):
        """Test getting piqa task."""
        task = get_task("piqa")
        assert isinstance(task, PIQA)

    def test_get_socialiqa(self):
        """Test getting socialiqa task."""
        task = get_task("socialiqa")
        assert isinstance(task, SocialIQA)

    def test_get_winogrande(self):
        """Test getting winogrande task."""
        task = get_task("winogrande")
        assert isinstance(task, WinoGrande)
