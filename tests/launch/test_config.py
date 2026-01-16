"""Tests for olmo_eval.launch.config module."""

import tempfile

import pytest

from olmo_eval.launch.config import (
    LaunchConfig,
    ModelConfig,
    get_template,
    parse_model_config,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test creating a ModelConfig with name_or_path only."""
        config = ModelConfig(name_or_path="llama3.1-8b")
        assert config.name_or_path == "llama3.1-8b"
        assert config.gpus is None
        assert config.cluster is None
        assert config.preemptible is None
        assert config.timeout is None
        assert config.shared_memory is None

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


class TestLaunchConfigModelConfigs:
    """Tests for LaunchConfig model configuration features."""

    def test_get_model_configs_from_strings(self):
        """Test get_model_configs with simple string models."""
        config = LaunchConfig(
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
        config = LaunchConfig(
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
        config = LaunchConfig(
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


class TestLaunchConfigGetModelResources:
    """Tests for LaunchConfig.get_model_resources method."""

    def test_get_model_resources_no_overrides(self):
        """Test get_model_resources returns defaults when no model overrides."""
        config = LaunchConfig(
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
        config = LaunchConfig(
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
        config = LaunchConfig(
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
        config = LaunchConfig(
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
        config = LaunchConfig(
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
        config = LaunchConfig(
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


class TestLaunchConfigFromYaml:
    """Tests for LaunchConfig.from_yaml with per-model configs."""

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

            config = LaunchConfig.from_yaml(f.name)

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

            config = LaunchConfig.from_yaml(f.name)

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

            config = LaunchConfig.from_yaml(f.name)
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

            config = LaunchConfig.from_yaml(f.name, overrides=["gpus=4", "priority=high"])

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
