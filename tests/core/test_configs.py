"""Tests for olmo_eval.core.configs module."""

# Import to ensure suites are registered
import olmo_eval.evals  # noqa: F401
from olmo_eval.core.configs import ModelConfig, expand_tasks, get_model_config


class TestExpandTasks:
    """Tests for expand_tasks function."""

    def test_expand_single_task(self):
        """Test expanding a single task (no expansion needed)."""
        result = expand_tasks(["arc_challenge"])

        assert result == ["arc_challenge"]

    def test_expand_multiple_tasks(self):
        """Test expanding multiple tasks (no expansion needed)."""
        result = expand_tasks(["arc_challenge", "arc_easy"])

        assert result == ["arc_challenge", "arc_easy"]

    def test_expand_suite(self):
        """Test expanding a suite to its tasks."""
        result = expand_tasks(["core:mc"])

        # core:mc should expand to multiple tasks
        assert len(result) > 1
        assert all(isinstance(t, str) for t in result)

    def test_expand_mixed_tasks_and_suites(self):
        """Test expanding mix of tasks and suites."""
        result = expand_tasks(["arc_challenge", "core:mc"])

        # Should have arc_challenge plus all core:mc tasks
        assert "arc_challenge" in result
        assert len(result) > 2

    def test_expand_empty_list(self):
        """Test expanding empty list."""
        result = expand_tasks([])

        assert result == []

    def test_expand_preserves_task_order(self):
        """Test that task order is preserved."""
        result = expand_tasks(["arc_easy", "arc_challenge"])

        assert result[0] == "arc_easy"
        assert result[1] == "arc_challenge"


class TestGetModelConfig:
    """Tests for get_model_config function."""

    def test_get_preset_model(self):
        """Test getting a preset model config."""
        config = get_model_config("llama3.1-8b")

        assert isinstance(config, ModelConfig)
        assert config.model == "meta-llama/Meta-Llama-3.1-8B"
        assert config.backend == "hf"

    def test_get_preset_with_trust_remote_code(self):
        """Test preset that requires trust_remote_code."""
        config = get_model_config("olmo-2-7b")

        assert config.model == "allenai/OLMo-2-1124-7B"
        assert config.trust_remote_code is True

    def test_get_unknown_model_as_hf_path(self):
        """Test that unknown model name is treated as HF path."""
        config = get_model_config("some-org/custom-model")

        assert config.model == "some-org/custom-model"
        assert config.backend == "hf"  # Default

    def test_get_model_with_override(self):
        """Test getting model with field override."""
        config = get_model_config("llama3.1-8b", backend="vllm")

        assert config.model == "meta-llama/Meta-Llama-3.1-8B"
        assert config.backend == "vllm"

    def test_get_model_with_multiple_overrides(self):
        """Test getting model with multiple overrides."""
        config = get_model_config(
            "llama3.1-8b",
            backend="vllm",
            dtype="float16",
            revision="main",
        )

        assert config.backend == "vllm"
        assert config.dtype == "float16"
        assert config.revision == "main"

    def test_get_unknown_model_with_overrides(self):
        """Test unknown model with overrides."""
        config = get_model_config(
            "custom/model",
            backend="vllm",
            trust_remote_code=True,
        )

        assert config.model == "custom/model"
        assert config.backend == "vllm"
        assert config.trust_remote_code is True

    def test_get_model_extra_args_merged(self):
        """Test that extra_args are merged for presets."""
        # Override with additional extra_args
        config = get_model_config(
            "llama3.1-8b",
            extra_args={"custom_arg": "value"},
        )

        assert "custom_arg" in config.extra_args
        assert config.extra_args["custom_arg"] == "value"

    def test_preset_not_mutated(self):
        """Test that getting with overrides doesn't mutate preset."""
        original = get_model_config("llama3.1-8b")
        _ = get_model_config("llama3.1-8b", backend="vllm")
        after = get_model_config("llama3.1-8b")

        assert original.backend == after.backend == "hf"
