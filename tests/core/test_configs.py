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
        result = expand_tasks(["mt_mbpp_v2fix"])

        # mt_mbpp_v2fix should expand to multiple tasks (17 languages)
        assert len(result) > 1
        assert all(isinstance(t, str) for t in result)

    def test_expand_mixed_tasks_and_suites(self):
        """Test expanding mix of tasks and suites."""
        result = expand_tasks(["humaneval", "mt_mbpp_v2fix"])

        # Should have humaneval plus all mt_mbpp_v2fix tasks
        assert "humaneval" in result
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

    def test_expand_suite_with_overrides(self):
        """Test expanding a suite with inline overrides."""
        result = expand_tasks(["mt_mbpp_v2fix::temperature=0.6"])

        # All expanded tasks should have the override suffix
        assert len(result) > 1
        assert all("::temperature=0.6" in t for t in result)

    def test_expand_suite_with_priority(self):
        """Test expanding a suite with priority suffix."""
        result = expand_tasks(["mt_mbpp_v2fix@high"])

        # All expanded tasks should have the priority suffix
        assert len(result) > 1
        assert all(t.endswith("@high") for t in result)

    def test_expand_suite_with_overrides_and_priority(self):
        """Test expanding a suite with both overrides and priority."""
        result = expand_tasks(["mt_mbpp_v2fix::temperature=0@urgent"])

        # All expanded tasks should have both suffixes in correct order
        assert len(result) > 1
        for task in result:
            assert "::temperature=0" in task
            assert task.endswith("@urgent")

    def test_expand_task_with_overrides_unchanged(self):
        """Test that non-suite tasks with overrides are unchanged."""
        result = expand_tasks(["arc_challenge::temperature=0.5@high"])

        assert result == ["arc_challenge::temperature=0.5@high"]

    def test_expand_mixed_with_overrides(self):
        """Test mix of tasks and suites with various suffixes."""
        result = expand_tasks(
            [
                "humaneval::limit=10",
                "mt_mbpp_v2fix::temperature=0@high",
            ]
        )

        # First task should be unchanged
        assert result[0] == "humaneval::limit=10"
        # Rest should be expanded suite tasks with overrides and priority
        for task in result[1:]:
            assert "::temperature=0" in task
            assert task.endswith("@high")


class TestGetModelConfig:
    """Tests for get_model_config function."""

    def test_get_preset_model(self):
        """Test getting a preset model config."""
        config = get_model_config("llama3.1-8b")

        assert isinstance(config, ModelConfig)
        assert config.model == "meta-llama/Meta-Llama-3.1-8B"
        assert config.backend == "vllm"

    def test_get_preset_with_trust_remote_code(self):
        """Test preset that requires trust_remote_code."""
        config = get_model_config("olmo-2-7b")

        assert config.model == "allenai/OLMo-2-1124-7B"
        assert config.trust_remote_code is True

    def test_get_unknown_model_as_hf_path(self):
        """Test that unknown model name is treated as HF path."""
        config = get_model_config("some-org/custom-model")

        assert config.model == "some-org/custom-model"
        assert config.backend == "vllm"  # Default

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
        _ = get_model_config("llama3.1-8b", backend="hf")
        after = get_model_config("llama3.1-8b")

        assert original.backend == after.backend == "vllm"

    def test_tokenizer_override_preset(self):
        """Test tokenizer override on a preset model."""
        config = get_model_config("llama3.1-8b", tokenizer="allenai/dolma2-tokenizer")

        assert config.model == "meta-llama/Meta-Llama-3.1-8B"
        assert config.tokenizer == "allenai/dolma2-tokenizer"

    def test_tokenizer_override_custom_model(self):
        """Test tokenizer override on a custom (non-preset) model."""
        config = get_model_config(
            "custom/my-model",
            tokenizer="custom/my-tokenizer",
            trust_remote_code=True,
        )

        assert config.model == "custom/my-model"
        assert config.tokenizer == "custom/my-tokenizer"
        assert config.trust_remote_code is True

    def test_tokenizer_default_is_none(self):
        """Test that tokenizer defaults to None for models without preset tokenizer."""
        config = get_model_config("llama3.1-8b")

        # llama3.1 doesn't have a preset tokenizer - defaults to None
        assert config.tokenizer is None

    def test_preset_with_tokenizer_preserved(self):
        """Test that presets with tokenizer keep their tokenizer."""
        config = get_model_config("olmo-2-7b")

        # OLMo-2 has a default tokenizer set in the preset
        assert config.tokenizer is not None
