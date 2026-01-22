"""Tests for olmo_eval.tasks.fim module."""

import pytest

from olmo_eval.core import Instance, LMOutput, RequestType
from olmo_eval.evals.constants.code import (
    DEEPSEEK_CODER_FIM,
    FIM_CONFIGS,
    OLMO_FIM,
    SANTACODER_FIM,
    STARCODER_FIM,
)
from olmo_eval.evals.tasks import get_task, list_tasks
from olmo_eval.evals.tasks.fim import (
    HumanEvalFIMMulti,
    HumanEvalFIMRandom,
    HumanEvalFIMSingle,
    HumanEvalFIMTask,
    _fim_multi_config,
    _fim_random_config,
    _fim_single_config,
)

# =============================================================================
# FIM Config Tests
# =============================================================================


class TestFIMConfig:
    """Tests for FIM token configurations."""

    def test_santacoder_config(self):
        """Test SantaCoder FIM configuration."""
        assert SANTACODER_FIM.lead_token == "<fim-prefix>"
        assert SANTACODER_FIM.center_token == "<fim-suffix>"
        assert SANTACODER_FIM.end_token == "<fim-middle>"
        assert "<|endoftext|>" in SANTACODER_FIM.stop_sequences

    def test_starcoder_config(self):
        """Test StarCoder FIM configuration."""
        assert STARCODER_FIM.lead_token == "<fim_prefix>"
        assert STARCODER_FIM.center_token == "<fim_suffix>"
        assert STARCODER_FIM.end_token == "<fim_middle>"

    def test_deepseek_config(self):
        """Test DeepSeek Coder FIM configuration."""
        assert DEEPSEEK_CODER_FIM.lead_token == "<｜fim▁begin｜>"
        assert DEEPSEEK_CODER_FIM.center_token == "<｜fim▁hole｜>"
        assert DEEPSEEK_CODER_FIM.end_token == "<｜fim▁end｜>"

    def test_olmo_config(self):
        """Test OLMo FIM configuration."""
        assert OLMO_FIM.lead_token == "<|fim_prefix|>"
        assert OLMO_FIM.center_token == "<|fim_suffix|>"
        assert OLMO_FIM.end_token == "<|fim_middle|>"

    def test_fim_configs_mapping(self):
        """Test FIM configs mapping has all expected entries."""
        assert "santacoder" in FIM_CONFIGS
        assert "starcoder" in FIM_CONFIGS
        assert "deepseek" in FIM_CONFIGS
        assert "olmo" in FIM_CONFIGS
        assert len(FIM_CONFIGS) == 4

    def test_to_context_kwargs(self):
        """Test FIMConfig.to_context_kwargs method."""
        kwargs = SANTACODER_FIM.to_context_kwargs()
        assert kwargs["lead_token"] == "<fim-prefix>"
        assert kwargs["center_token"] == "<fim-suffix>"
        assert kwargs["end_token"] == "<fim-middle>"

    def test_to_generation_kwargs(self):
        """Test FIMConfig.to_generation_kwargs method."""
        kwargs = SANTACODER_FIM.to_generation_kwargs()
        assert "stop_sequences" in kwargs
        assert "<|endoftext|>" in kwargs["stop_sequences"]


# =============================================================================
# FIM Task Tests
# =============================================================================


class TestHumanEvalFIMTask:
    """Tests for HumanEvalFIMTask base class."""

    @pytest.fixture
    def fim_single_task(self):
        """Create a FIM single task for testing."""
        return get_task("codex_humanevalfim_single")

    def test_process_doc(self, fim_single_task):
        """Test processing a FIM document."""
        doc = {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n    ",
            "suffix": "\n    return result",
            "canonical_solution": "result = a + b",
            "entry_point": "add",
            "test": "def check(add):\n    assert add(1, 2) == 3",
        }

        instance = fim_single_task.process_doc(doc, index=0)

        assert isinstance(instance, Instance)
        # Check that FIM tokens are in the prompt
        assert SANTACODER_FIM.lead_token in instance.question
        assert SANTACODER_FIM.center_token in instance.question
        assert SANTACODER_FIM.end_token in instance.question
        # Check metadata
        assert instance.metadata["prefix"] == doc["prompt"]
        assert instance.metadata["suffix"] == doc["suffix"]
        assert instance.metadata["task_id"] == "HumanEval/0"

    def test_format_request(self, fim_single_task):
        """Test request formatting."""
        instance = Instance(
            question="<fim-prefix>def add(a, b):<fim-suffix>return result<fim-middle>",
            gold_answer="result = a + b",
            metadata={"stop_sequences": list(SANTACODER_FIM.stop_sequences)},
        )

        request = fim_single_task.format_request(instance)

        assert request.request_type == RequestType.COMPLETION
        assert "<fim-prefix>" in request.prompt
        # Stop sequences are in instance metadata, not in request
        assert instance.metadata["stop_sequences"] is not None
        assert len(instance.metadata["stop_sequences"]) > 0

    def test_extract_answer(self, fim_single_task):
        """Test answer extraction."""
        output = LMOutput(text="    result = a + b")
        answer = fim_single_task.extract_answer(output)
        assert answer == "result = a + b"

    def test_extract_answer_removes_stop_tokens(self, fim_single_task):
        """Test that stop tokens are removed from extracted answer."""
        output = LMOutput(text="result = a + b<|endoftext|>")
        answer = fim_single_task.extract_answer(output)
        assert answer == "result = a + b"

    def test_extract_answer_empty(self, fim_single_task):
        """Test extraction from empty output."""
        output = LMOutput(text="")
        answer = fim_single_task.extract_answer(output)
        assert answer is None

    def test_assemble_code(self, fim_single_task):
        """Test code assembly from prefix + middle + suffix."""
        instance = Instance(
            question="<fim-prefix>def add(a, b):\n    <fim-suffix>\n    return result<fim-middle>",
            gold_answer="result = a + b",
            metadata={
                "prefix": "def add(a, b):\n    ",
                "suffix": "\n    return result",
            },
        )

        code = fim_single_task.assemble_code(instance, "result = a + b")
        assert code == "def add(a, b):\n    result = a + b\n    return result"


# =============================================================================
# Task Configuration Tests
# =============================================================================


class TestFIMTaskConfigs:
    """Tests for FIM task configurations."""

    def test_fim_single_config(self):
        """Test FIM single task config."""
        config = _fim_single_config()
        assert config.name == "codex_humanevalfim_single"
        assert config.data_source.path == "loubnabnl/humaneval_infilling"
        assert config.data_source.subset == "HumanEval-SingleLineInfilling"

    def test_fim_multi_config(self):
        """Test FIM multi task config."""
        config = _fim_multi_config()
        assert config.name == "codex_humanevalfim_multi"
        assert config.data_source.subset == "HumanEval-MultiLineInfilling"

    def test_fim_random_config(self):
        """Test FIM random task config."""
        config = _fim_random_config()
        assert config.name == "codex_humanevalfim_random"
        assert config.data_source.subset == "HumanEval-RandomSpanInfilling"


# =============================================================================
# Task Registration Tests
# =============================================================================


class TestFIMTaskRegistration:
    """Tests for FIM task registration."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Ensure tasks are registered by importing module."""
        import olmo_eval.evals.tasks.fim  # noqa: F401

        yield

    def test_all_fim_tasks_registered(self):
        """Test that all 3 FIM tasks are registered."""
        all_tasks = list_tasks()
        assert "codex_humanevalfim_single" in all_tasks
        assert "codex_humanevalfim_multi" in all_tasks
        assert "codex_humanevalfim_random" in all_tasks

    def test_get_fim_single(self):
        """Test getting FIM single task."""
        task = get_task("codex_humanevalfim_single")
        assert task is not None
        assert isinstance(task, HumanEvalFIMSingle)
        assert task.default_subset == "HumanEval-SingleLineInfilling"

    def test_get_fim_multi(self):
        """Test getting FIM multi task."""
        task = get_task("codex_humanevalfim_multi")
        assert task is not None
        assert isinstance(task, HumanEvalFIMMulti)
        assert task.default_subset == "HumanEval-MultiLineInfilling"

    def test_get_fim_random(self):
        """Test getting FIM random task."""
        task = get_task("codex_humanevalfim_random")
        assert task is not None
        assert isinstance(task, HumanEvalFIMRandom)
        assert task.default_subset == "HumanEval-RandomSpanInfilling"

    def test_task_has_correct_hf_path(self):
        """Test that tasks have correct HuggingFace path."""
        task = get_task("codex_humanevalfim_single")
        assert task.default_hf_path == "loubnabnl/humaneval_infilling"


# =============================================================================
# Custom FIM Config Tests
# =============================================================================


class TestCustomFIMConfig:
    """Tests for using custom FIM configurations."""

    def test_task_with_starcoder_config(self):
        """Test creating task with StarCoder FIM tokens."""
        config = _fim_single_config()

        class StarCoderFIMTask(HumanEvalFIMTask):
            hf_subset = "HumanEval-SingleLineInfilling"

            def __init__(self, cfg):
                super().__init__(cfg, fim_config=STARCODER_FIM)

        task = StarCoderFIMTask(config)
        assert task.fim_config == STARCODER_FIM
        assert task.fim_config.lead_token == "<fim_prefix>"

    def test_task_with_olmo_config(self):
        """Test creating task with OLMo FIM tokens."""
        config = _fim_single_config()

        class OLMoFIMTask(HumanEvalFIMTask):
            hf_subset = "HumanEval-SingleLineInfilling"

            def __init__(self, cfg):
                super().__init__(cfg, fim_config=OLMO_FIM)

        task = OLMoFIMTask(config)
        assert task.fim_config == OLMO_FIM
        assert task.fim_config.lead_token == "<|fim_prefix|>"


# =============================================================================
# FIM Prompt Formatting Tests
# =============================================================================


class TestFIMPromptFormatting:
    """Tests for FIM prompt formatting with different configs."""

    def test_santacoder_prompt_format(self):
        """Test prompt formatting with SantaCoder tokens."""
        task = get_task("codex_humanevalfim_single")
        doc = {
            "task_id": "test",
            "prompt": "PREFIX",
            "suffix": "SUFFIX",
            "canonical_solution": "MIDDLE",
            "entry_point": "test",
            "test": "",
        }

        instance = task.process_doc(doc, 0)

        expected = "<fim-prefix>PREFIX<fim-suffix>SUFFIX<fim-middle>"
        assert instance.question == expected

    def test_prompt_contains_all_fim_tokens(self):
        """Test that generated prompt contains all FIM tokens."""
        task = get_task("codex_humanevalfim_single")
        doc = {
            "task_id": "test",
            "prompt": "def foo():\n    ",
            "suffix": "\n    return x",
            "canonical_solution": "x = 1",
            "entry_point": "foo",
            "test": "",
        }

        instance = task.process_doc(doc, 0)
        prompt = instance.question

        # Verify structure
        assert prompt.startswith(SANTACODER_FIM.lead_token)
        assert SANTACODER_FIM.center_token in prompt
        assert prompt.endswith(SANTACODER_FIM.end_token)

        # Verify content order
        lead_idx = prompt.index(SANTACODER_FIM.lead_token)
        center_idx = prompt.index(SANTACODER_FIM.center_token)
        end_idx = prompt.index(SANTACODER_FIM.end_token)
        assert lead_idx < center_idx < end_idx
