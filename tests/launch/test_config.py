"""Tests for olmo_eval.launch.config module."""

import tempfile

import pytest

from olmo_eval.launch.config import (
    EvalConfig,
    ModelConfig,
    get_model_short_name,
    get_tasks_short_name,
    get_template,
    parse_model_config,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test creating a ModelConfig with name_or_path only."""
        config = ModelConfig(name_or_path="llama3.1-8b")
        assert config.name_or_path == "llama3.1-8b"
        assert config.alias is None
        assert config.gpus is None
        assert config.cluster is None
        assert config.preemptible is None
        assert config.timeout is None
        assert config.shared_memory is None

    def test_model_config_with_alias(self):
        """Test creating a ModelConfig with an alias."""
        config = ModelConfig(
            name_or_path="/weka/checkpoints/my-model/step1000-hf",
            alias="my-model-1k",
            gpus=4,
        )
        assert config.name_or_path == "/weka/checkpoints/my-model/step1000-hf"
        assert config.alias == "my-model-1k"
        assert config.gpus == 4

    def test_model_config_with_overrides(self):
        """Test creating a ModelConfig with resource overrides."""
        config = ModelConfig(
            name_or_path="llama3.1-70b",
            gpus=4,
            cluster="h100",
            preemptible=False,
            timeout="48h",
            shared_memory="20GiB",
        )
        assert config.name_or_path == "llama3.1-70b"
        assert config.gpus == 4
        assert config.cluster == "h100"
        assert config.preemptible is False
        assert config.timeout == "48h"
        assert config.shared_memory == "20GiB"


class TestParseModelConfig:
    """Tests for parse_model_config function."""

    def test_parse_string_model(self):
        """Test parsing a simple string model name."""
        config = parse_model_config("llama3.1-8b")
        assert isinstance(config, ModelConfig)
        assert config.name_or_path == "llama3.1-8b"
        assert config.gpus is None

    def test_parse_dict_model(self):
        """Test parsing a dict model config."""
        config = parse_model_config({"name_or_path": "llama3.1-70b", "gpus": 4})
        assert isinstance(config, ModelConfig)
        assert config.name_or_path == "llama3.1-70b"
        assert config.gpus == 4

    def test_parse_dict_with_all_fields(self):
        """Test parsing a dict with all fields."""
        config = parse_model_config(
            {
                "name_or_path": "llama3.1-70b",
                "gpus": 4,
                "cluster": "h100",
                "preemptible": False,
                "timeout": "48h",
                "shared_memory": "20GiB",
            }
        )
        assert config.name_or_path == "llama3.1-70b"
        assert config.gpus == 4
        assert config.cluster == "h100"
        assert config.preemptible is False
        assert config.timeout == "48h"
        assert config.shared_memory == "20GiB"

    def test_parse_model_config_passthrough(self):
        """Test that ModelConfig passes through unchanged."""
        original = ModelConfig(name_or_path="test", gpus=2)
        parsed = parse_model_config(original)
        assert parsed is original

    def test_parse_invalid_type_raises(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Invalid model specification"):
            parse_model_config(123)  # type: ignore[arg-type]

    def test_parse_dict_with_alias(self):
        """Test parsing a dict model config with alias."""
        config = parse_model_config(
            {
                "name_or_path": "/weka/checkpoints/my-model/step1000-hf",
                "alias": "my-model-1k",
                "gpus": 4,
            }
        )
        assert config.name_or_path == "/weka/checkpoints/my-model/step1000-hf"
        assert config.alias == "my-model-1k"
        assert config.gpus == 4


class TestGetModelShortName:
    """Tests for get_model_short_name function."""

    def test_simple_model_name(self):
        """Test simple model name returns as-is (lowercased)."""
        config = ModelConfig(name_or_path="llama3.1-8b")
        assert get_model_short_name(config) == "llama3.1-8b"

    def test_huggingface_path(self):
        """Test HuggingFace path returns last component."""
        config = ModelConfig(name_or_path="meta-llama/Llama-3.1-8B")
        assert get_model_short_name(config) == "llama-3.1-8b"

    def test_local_path(self):
        """Test local path returns last component."""
        config = ModelConfig(name_or_path="/weka/checkpoints/model/step1000-hf")
        assert get_model_short_name(config) == "step1000-hf"

    def test_local_path_with_trailing_slash(self):
        """Test local path with trailing slash returns last non-empty component."""
        config = ModelConfig(name_or_path="/weka/checkpoints/model/step1000-hf/")
        assert get_model_short_name(config) == "step1000-hf"

    def test_alias_overrides_name(self):
        """Test alias is used when provided."""
        config = ModelConfig(
            name_or_path="/weka/checkpoints/model/step1000-hf/",
            alias="my-model-1k",
        )
        assert get_model_short_name(config) == "my-model-1k"

    def test_alias_is_lowercased(self):
        """Test alias is lowercased."""
        config = ModelConfig(
            name_or_path="some-model",
            alias="My-Model-Name",
        )
        assert get_model_short_name(config) == "my-model-name"

    def test_long_path_uses_last_16_chars(self):
        """Test very long last component uses last 16 chars of full path."""
        # Create a path where the last component is > 32 chars
        long_component = "a" * 40
        config = ModelConfig(name_or_path=f"/weka/checkpoints/{long_component}")
        result = get_model_short_name(config)
        assert len(result) == 16
        assert result == "a" * 16

    def test_empty_last_component_uses_last_16_chars(self):
        """Test path ending with just slashes uses last 16 chars."""
        config = ModelConfig(name_or_path="/weka/checkpoints/my-model-name")
        # Last component is "my-model-name" which is fine
        assert get_model_short_name(config) == "my-model-name"


class TestGetTasksShortName:
    """Tests for get_tasks_short_name function."""

    def test_single_task(self):
        """Test single task returns task name."""
        assert get_tasks_short_name(["mmlu"]) == "mmlu"

    def test_single_task_with_priority(self):
        """Test single task strips @priority suffix."""
        assert get_tasks_short_name(["mmlu@high"]) == "mmlu"

    def test_single_task_with_variant(self):
        """Test single task strips :variant suffix."""
        assert get_tasks_short_name(["arc:mc"]) == "arc"

    def test_single_task_with_regime(self):
        """Test single task strips ::regime suffix."""
        assert get_tasks_short_name(["mmlu::olmes"]) == "mmlu"

    def test_two_tasks(self):
        """Test two tasks joined with underscore."""
        assert get_tasks_short_name(["gsm8k", "arc_challenge"]) == "gsm8k_arc"

    def test_three_tasks(self):
        """Test three tasks joined with underscore."""
        assert get_tasks_short_name(["mmlu", "gsm8k", "hellaswag"]) == "mmlu_gsm8k_hellaswa"

    def test_four_or_more_tasks(self):
        """Test 4+ tasks uses first task and count."""
        result = get_tasks_short_name(["mmlu", "gsm8k", "hellaswag", "arc_challenge"])
        assert result == "mmlu_3more"

    def test_many_tasks(self):
        """Test many tasks uses first task and count."""
        tasks = ["mmlu", "gsm8k", "hellaswag", "arc_challenge", "winogrande", "truthfulqa"]
        result = get_tasks_short_name(tasks)
        assert result == "mmlu_5more"

    def test_empty_list(self):
        """Test empty task list returns placeholder."""
        assert get_tasks_short_name([]) == "notasks"

    def test_strips_challenge_suffix(self):
        """Test _challenge suffix is removed."""
        assert get_tasks_short_name(["arc_challenge"]) == "arc"

    def test_long_task_name_truncated(self):
        """Test long task names are truncated."""
        result = get_tasks_short_name(["verylongtasknamethatshouldbetruncated"])
        assert len(result) <= 24

    def test_mixed_priorities_and_variants(self):
        """Test tasks with mixed priorities and variants."""
        tasks = ["mmlu@high", "gsm8k::olmes", "arc:mc@low"]
        result = get_tasks_short_name(tasks)
        assert result == "mmlu_gsm8k_arc"


class TestEvalConfigModelConfigs:
    """Tests for EvalConfig model configuration features."""

    def test_get_model_configs_from_strings(self):
        """Test get_model_configs with simple string models."""
        config = EvalConfig(
            name="test",
            models=["llama3.1-8b", "olmo-2-7b"],
            tasks=["mmlu"],
        )
        model_configs = config.get_model_configs()

        assert len(model_configs) == 2
        assert model_configs[0].name_or_path == "llama3.1-8b"
        assert model_configs[0].gpus is None
        assert model_configs[1].name_or_path == "olmo-2-7b"

    def test_get_model_configs_from_dicts(self):
        """Test get_model_configs with dict model configs."""
        config = EvalConfig(
            name="test",
            models=[
                {"name_or_path": "llama3.1-8b", "gpus": 1},
                {"name_or_path": "llama3.1-70b", "gpus": 4, "timeout": "48h"},
            ],
            tasks=["mmlu"],
        )
        model_configs = config.get_model_configs()

        assert len(model_configs) == 2
        assert model_configs[0].name_or_path == "llama3.1-8b"
        assert model_configs[0].gpus == 1
        assert model_configs[1].name_or_path == "llama3.1-70b"
        assert model_configs[1].gpus == 4
        assert model_configs[1].timeout == "48h"

    def test_get_model_configs_mixed(self):
        """Test get_model_configs with mixed string and dict models."""
        config = EvalConfig(
            name="test",
            models=[
                "llama3.1-8b",  # Simple string
                {"name_or_path": "llama3.1-70b", "gpus": 4},  # Dict with override
            ],
            tasks=["mmlu"],
        )
        model_configs = config.get_model_configs()

        assert len(model_configs) == 2
        assert model_configs[0].name_or_path == "llama3.1-8b"
        assert model_configs[0].gpus is None
        assert model_configs[1].name_or_path == "llama3.1-70b"
        assert model_configs[1].gpus == 4


class TestEvalConfigGetModelResources:
    """Tests for EvalConfig.get_model_resources method."""

    def test_get_model_resources_no_overrides(self):
        """Test get_model_resources returns defaults when no model overrides."""
        config = EvalConfig(
            name="test",
            models=["llama3.1-8b"],
            tasks=["mmlu"],
            gpus=2,
            cluster="a100",
            timeout="12h",
        )
        model = ModelConfig(name_or_path="llama3.1-8b")
        resources = config.get_model_resources(model)

        assert resources["gpus"] == 2
        assert resources["cluster"] == "a100"
        assert resources["timeout"] == "12h"

    def test_get_model_resources_with_overrides(self):
        """Test get_model_resources applies model overrides."""
        config = EvalConfig(
            name="test",
            models=["llama3.1-8b"],
            tasks=["mmlu"],
            gpus=1,
            cluster="h100",
            timeout="24h",
        )
        model = ModelConfig(
            name_or_path="llama3.1-70b",
            gpus=4,
            timeout="48h",
        )
        resources = config.get_model_resources(model)

        assert resources["gpus"] == 4  # Model override
        assert resources["cluster"] == "h100"  # Default (no override)
        assert resources["timeout"] == "48h"  # Model override

    def test_get_model_resources_partial_overrides(self):
        """Test get_model_resources with only some overrides."""
        config = EvalConfig(
            name="test",
            models=["llama3.1-8b"],
            tasks=["mmlu"],
            gpus=1,
            cluster="h100",
            preemptible=True,
        )
        model = ModelConfig(
            name_or_path="llama3.1-13b",
            gpus=2,
            # No cluster, preemptible overrides
        )
        resources = config.get_model_resources(model)

        assert resources["gpus"] == 2  # Model override
        assert resources["cluster"] == "h100"  # Default
        assert resources["preemptible"] is True  # Default

    def test_get_model_resources_shared_memory(self):
        """Test get_model_resources handles shared_memory."""
        config = EvalConfig(
            name="test",
            models=["llama3.1-8b"],
            tasks=["mmlu"],
        )
        model = ModelConfig(
            name_or_path="llama3.1-8b",
            shared_memory="10GiB",
        )
        resources = config.get_model_resources(model)

        assert resources["shared_memory"] == "10GiB"

    def test_get_model_resources_parallelism_default(self):
        """Test get_model_resources returns default parallelism."""
        config = EvalConfig(
            name="test",
            models=["llama3.1-8b"],
            tasks=["mmlu"],
            parallelism=4,
        )
        model = ModelConfig(name_or_path="llama3.1-8b")
        resources = config.get_model_resources(model)

        assert resources["parallelism"] == 4

    def test_get_model_resources_parallelism_override(self):
        """Test get_model_resources applies model parallelism override."""
        config = EvalConfig(
            name="test",
            models=["llama3.1-8b"],
            tasks=["mmlu"],
            parallelism=2,
        )
        model = ModelConfig(
            name_or_path="llama3.1-8b",
            parallelism=8,
        )
        resources = config.get_model_resources(model)

        assert resources["parallelism"] == 8  # Model override wins


class TestEvalConfigFromYaml:
    """Tests for EvalConfig.from_yaml with per-model configs."""

    def test_from_yaml_simple_models(self):
        """Test loading YAML with simple string models."""
        yaml_content = """
name: test-eval
models:
  - llama3.1-8b
  - olmo-2-7b
tasks:
  - mmlu
  - gsm8k
cluster: h100
gpus: 1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = EvalConfig.from_yaml(f.name)

            assert config.name == "test-eval"
            assert len(config.models) == 2
            assert config.models[0] == "llama3.1-8b"
            assert config.models[1] == "olmo-2-7b"

            model_configs = config.get_model_configs()
            assert model_configs[0].name_or_path == "llama3.1-8b"
            assert model_configs[0].gpus is None

    def test_from_yaml_per_model_resources(self):
        """Test loading YAML with per-model resource overrides."""
        yaml_content = """
name: test-eval
models:
  - name_or_path: llama3.1-8b
    gpus: 1
  - name_or_path: llama3.1-70b
    gpus: 4
    timeout: 48h
    preemptible: false
tasks:
  - mmlu@high
  - gsm8k@normal
cluster: h100
gpus: 1
priority: normal
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = EvalConfig.from_yaml(f.name)

            model_configs = config.get_model_configs()
            assert len(model_configs) == 2

            # First model
            assert model_configs[0].name_or_path == "llama3.1-8b"
            assert model_configs[0].gpus == 1

            # Second model with overrides
            assert model_configs[1].name_or_path == "llama3.1-70b"
            assert model_configs[1].gpus == 4
            assert model_configs[1].timeout == "48h"
            assert model_configs[1].preemptible is False

    def test_from_yaml_mixed_models(self):
        """Test loading YAML with mixed string and dict models."""
        yaml_content = """
name: test-eval
models:
  - llama3.1-8b
  - name_or_path: llama3.1-70b
    gpus: 4
tasks:
  - mmlu
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = EvalConfig.from_yaml(f.name)
            model_configs = config.get_model_configs()

            assert model_configs[0].name_or_path == "llama3.1-8b"
            assert model_configs[0].gpus is None
            assert model_configs[1].name_or_path == "llama3.1-70b"
            assert model_configs[1].gpus == 4

    def test_from_yaml_with_cli_overrides(self):
        """Test YAML loading with CLI-style overrides."""
        yaml_content = """
name: test-eval
models:
  - llama3.1-8b
tasks:
  - mmlu
gpus: 1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = EvalConfig.from_yaml(f.name, overrides=["gpus=4", "priority=high"])

            assert config.gpus == 4
            assert config.priority == "high"


class TestGetTemplate:
    """Tests for get_template function."""

    def test_get_quick_template(self):
        """Test getting quick template."""
        template = get_template("quick")
        assert template["timeout"] == "4h"
        assert template["preemptible"] is True

    def test_get_standard_template(self):
        """Test getting standard template."""
        template = get_template("standard")
        assert template["timeout"] == "24h"

    def test_get_large_model_template(self):
        """Test getting large-model template."""
        template = get_template("large-model")
        assert template["gpus"] == 4
        assert template["priority"] == "high"
        assert template["timeout"] == "48h"
        assert template["preemptible"] is False

    def test_get_urgent_template(self):
        """Test getting urgent template."""
        template = get_template("urgent")
        assert template["priority"] == "urgent"
        assert template["preemptible"] is False

    def test_get_unknown_template_raises(self):
        """Test that unknown template name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown template"):
            get_template("nonexistent")

    def test_template_is_copy(self):
        """Test that returned template is a copy (not mutable)."""
        template1 = get_template("quick")
        template1["gpus"] = 999

        template2 = get_template("quick")
        assert template2["gpus"] == 1  # Original value
