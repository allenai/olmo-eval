"""Tests for olmo_eval.launch.config module."""

import tempfile

import pytest

from olmo_eval.launch.config import (
    BeakerModelSpec,
    EvalConfig,
    ProviderConfig,
    apply_overrides_to_model,
    get_model_short_name,
    get_tasks_short_name,
    parse_model_config,
)


class TestBeakerModelSpec:
    """Tests for BeakerModelSpec dataclass."""

    def test_model_config_creation(self):
        """Test creating a BeakerModelSpec with name_or_path only."""
        config = BeakerModelSpec(name_or_path="llama3.1-8b")
        assert config.name_or_path == "llama3.1-8b"
        assert config.alias is None
        assert config.gpus == 1  # Default
        assert config.parallelism == 1  # Default
        assert config.cluster is None
        assert config.preemptible is None
        assert config.timeout is None
        assert config.shared_memory is None

    def test_model_config_with_alias(self):
        """Test creating a BeakerModelSpec with an alias."""
        config = BeakerModelSpec(
            name_or_path="/weka/checkpoints/my-model/step1000-hf",
            alias="my-model-1k",
            gpus=4,
        )
        assert config.name_or_path == "/weka/checkpoints/my-model/step1000-hf"
        assert config.alias == "my-model-1k"
        assert config.gpus == 4

    def test_model_config_with_overrides(self):
        """Test creating a BeakerModelSpec with resource overrides."""
        config = BeakerModelSpec(
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
        assert isinstance(config, BeakerModelSpec)
        assert config.name_or_path == "llama3.1-8b"
        assert config.gpus == 1  # Default

    def test_parse_dict_model(self):
        """Test parsing a dict model config."""
        config = parse_model_config({"name_or_path": "llama3.1-70b", "gpus": 4})
        assert isinstance(config, BeakerModelSpec)
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
        """Test that BeakerModelSpec passes through unchanged."""
        original = BeakerModelSpec(name_or_path="test", gpus=2)
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
        config = BeakerModelSpec(name_or_path="llama3.1-8b")
        assert get_model_short_name(config) == "llama3.1-8b"

    def test_huggingface_path(self):
        """Test HuggingFace path returns last component."""
        config = BeakerModelSpec(name_or_path="meta-llama/Llama-3.1-8B")
        assert get_model_short_name(config) == "llama-3.1-8b"

    def test_local_path(self):
        """Test local path returns last component."""
        config = BeakerModelSpec(name_or_path="/weka/checkpoints/model/step1000-hf")
        assert get_model_short_name(config) == "step1000-hf"

    def test_local_path_with_trailing_slash(self):
        """Test local path with trailing slash returns last non-empty component."""
        config = BeakerModelSpec(name_or_path="/weka/checkpoints/model/step1000-hf/")
        assert get_model_short_name(config) == "step1000-hf"

    def test_alias_overrides_name(self):
        """Test alias is used when provided."""
        config = BeakerModelSpec(
            name_or_path="/weka/checkpoints/model/step1000-hf/",
            alias="my-model-1k",
        )
        assert get_model_short_name(config) == "my-model-1k"

    def test_alias_is_lowercased(self):
        """Test alias is lowercased."""
        config = BeakerModelSpec(
            name_or_path="some-model",
            alias="My-Model-Name",
        )
        assert get_model_short_name(config) == "my-model-name"

    def test_long_path_uses_last_16_chars(self):
        """Test very long last component uses last 16 chars of full path."""
        # Create a path where the last component is > 32 chars
        long_component = "a" * 40
        config = BeakerModelSpec(name_or_path=f"/weka/checkpoints/{long_component}")
        result = get_model_short_name(config)
        assert len(result) == 16
        assert result == "a" * 16

    def test_empty_last_component_uses_last_16_chars(self):
        """Test path ending with just slashes uses last 16 chars."""
        config = BeakerModelSpec(name_or_path="/weka/checkpoints/my-model-name")
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
        """Test single task strips :regime suffix."""
        assert get_tasks_short_name(["mmlu:olmes"]) == "mmlu"

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
        tasks = ["mmlu@high", "gsm8k:olmes", "arc:mc@low"]
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
        assert model_configs[0].gpus == 1  # Default
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
        assert model_configs[0].gpus == 1  # Default
        assert model_configs[1].name_or_path == "llama3.1-70b"
        assert model_configs[1].gpus == 4


class TestEvalConfigGetModelResources:
    """Tests for EvalConfig.get_model_resources method."""

    def test_get_model_resources_defaults(self):
        """Test get_model_resources returns model defaults (gpus/parallelism are per-model)."""
        config = EvalConfig(
            name="test",
            models=["llama3.1-8b"],
            tasks=["mmlu"],
            cluster="a100",
            timeout="12h",
        )
        model = BeakerModelSpec(name_or_path="llama3.1-8b")
        resources = config.get_model_resources(model)

        assert resources["gpus"] == 1  # Model default
        assert resources["parallelism"] == 1  # Model default
        assert resources["cluster"] == "a100"
        assert resources["timeout"] == "12h"

    def test_get_model_resources_with_overrides(self):
        """Test get_model_resources applies model overrides."""
        config = EvalConfig(
            name="test",
            models=["llama3.1-8b"],
            tasks=["mmlu"],
            cluster="h100",
            timeout="24h",
        )
        model = BeakerModelSpec(
            name_or_path="llama3.1-70b",
            gpus=4,
            timeout="48h",
        )
        resources = config.get_model_resources(model)

        assert resources["gpus"] == 4  # Model value
        assert resources["cluster"] == "h100"  # Config default
        assert resources["timeout"] == "48h"  # Model override

    def test_get_model_resources_partial_overrides(self):
        """Test get_model_resources with only some overrides."""
        config = EvalConfig(
            name="test",
            models=["llama3.1-8b"],
            tasks=["mmlu"],
            cluster="h100",
            preemptible=True,
        )
        model = BeakerModelSpec(
            name_or_path="llama3.1-13b",
            gpus=2,
            # No cluster, preemptible overrides
        )
        resources = config.get_model_resources(model)

        assert resources["gpus"] == 2  # Model value
        assert resources["cluster"] == "h100"  # Default
        assert resources["preemptible"] is True  # Default

    def test_get_model_resources_shared_memory(self):
        """Test get_model_resources handles shared_memory."""
        config = EvalConfig(
            name="test",
            models=["llama3.1-8b"],
            tasks=["mmlu"],
        )
        model = BeakerModelSpec(
            name_or_path="llama3.1-8b",
            shared_memory="10GiB",
        )
        resources = config.get_model_resources(model)

        assert resources["shared_memory"] == "10GiB"

    def test_get_model_resources_parallelism_default(self):
        """Test get_model_resources returns model's parallelism (per-model setting)."""
        config = EvalConfig(
            name="test",
            models=["llama3.1-8b"],
            tasks=["mmlu"],
        )
        model = BeakerModelSpec(name_or_path="llama3.1-8b")
        resources = config.get_model_resources(model)

        assert resources["parallelism"] == 1  # Model default

    def test_get_model_resources_parallelism_override(self):
        """Test get_model_resources uses model's parallelism value."""
        config = EvalConfig(
            name="test",
            models=["llama3.1-8b"],
            tasks=["mmlu"],
        )
        model = BeakerModelSpec(
            name_or_path="llama3.1-8b",
            parallelism=8,
        )
        resources = config.get_model_resources(model)

        assert resources["parallelism"] == 8  # Model value


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
            assert model_configs[0].gpus == 1  # Default

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
            assert model_configs[0].gpus == 1  # Default
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
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = EvalConfig.from_yaml(f.name, overrides=["priority=high"])

            assert config.priority == "high"

    def test_from_yaml_rejects_top_level_gpus(self):
        """Test that top-level gpus field raises error."""
        yaml_content = """
name: test-eval
models:
  - llama3.1-8b
tasks:
  - mmlu
gpus: 4
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            import pytest

            with pytest.raises(ValueError, match="Top-level 'gpus' is no longer supported"):
                EvalConfig.from_yaml(f.name)

    def test_from_yaml_rejects_top_level_parallelism(self):
        """Test that top-level parallelism field raises error."""
        yaml_content = """
name: test-eval
models:
  - llama3.1-8b
tasks:
  - mmlu
parallelism: 4
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            import pytest

            with pytest.raises(ValueError, match="Top-level 'parallelism' is no longer supported"):
                EvalConfig.from_yaml(f.name)


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_default_values(self):
        """Test ProviderConfig default values."""
        config = ProviderConfig()
        assert config.name == "vllm"
        assert config.package is None

    def test_custom_name_and_package(self):
        """Test ProviderConfig with custom name and package."""
        config = ProviderConfig(name="litellm", package="litellm==1.0.0")
        assert config.name == "litellm"
        assert config.package == "litellm==1.0.0"


class TestApplyOverridesToModel:
    """Tests for apply_overrides_to_model function."""

    def test_no_overrides(self):
        """Test with no overrides."""
        config = apply_overrides_to_model("llama3.1-8b", [])
        assert config.name_or_path == "llama3.1-8b"
        assert config.gpus == 1  # Default
        assert config.provider is None

    def test_simple_overrides(self):
        """Test with simple key=value overrides."""
        config = apply_overrides_to_model("llama3.1-8b", ["gpus=4", "timeout=48h"])
        assert config.name_or_path == "llama3.1-8b"
        assert config.gpus == 4
        assert config.timeout == "48h"

    def test_nested_provider_overrides(self):
        """Test with nested provider overrides."""
        config = apply_overrides_to_model(
            "llama3.1-8b",
            ["provider.name=vllm", "provider.package=vllm==0.14.0"],
        )
        assert config.name_or_path == "llama3.1-8b"
        assert config.provider is not None
        assert config.provider.name == "vllm"
        assert config.provider.package == "vllm==0.14.0"

    def test_github_url_package(self):
        """Test with GitHub URL as provider package."""
        config = apply_overrides_to_model(
            "llama3.1-8b",
            ["provider.name=vllm", "provider.package=https://github.com/user/vllm@branch"],
        )
        assert config.provider is not None
        assert config.provider.package == "https://github.com/user/vllm@branch"

    def test_mixed_overrides(self):
        """Test with mix of simple and nested overrides."""
        config = apply_overrides_to_model(
            "llama3.1-8b",
            ["gpus=4", "provider.name=vllm", "load_format=auto"],
        )
        assert config.gpus == 4
        assert config.provider is not None
        assert config.provider.name == "vllm"
        assert config.load_format == "auto"


class TestParseModelConfigWithOverridesParam:
    """Tests for parse_model_config with overrides parameter (new -o syntax)."""

    def test_string_with_overrides_param(self):
        """Test parsing string model with overrides parameter."""
        config = parse_model_config(
            "llama3.1-8b",
            overrides=["provider.name=vllm", "gpus=4"],
        )
        assert config.name_or_path == "llama3.1-8b"
        assert config.gpus == 4
        assert config.provider is not None
        assert config.provider.name == "vllm"

    def test_dict_with_overrides_param(self):
        """Test parsing dict model with overrides parameter."""
        config = parse_model_config(
            {"name_or_path": "llama3.1-8b", "gpus": 2},
            overrides=["gpus=4", "timeout=48h"],
        )
        assert config.name_or_path == "llama3.1-8b"
        assert config.gpus == 4  # Override wins
        assert config.timeout == "48h"

    def test_model_config_with_overrides_param(self):
        """Test that BeakerModelSpec with overrides param gets new values applied."""
        original = BeakerModelSpec(name_or_path="llama3.1-8b", gpus=2)
        config = parse_model_config(original, overrides=["gpus=4"])
        assert config.gpus == 4
        # Original should be unchanged
        assert original.gpus == 2


class TestParseModelConfigWithProvider:
    """Tests for parse_model_config with ProviderConfig."""

    def test_parse_string_with_nested_provider(self):
        """Test parsing model string with nested provider config via overrides."""
        config = parse_model_config(
            "llama3.1-8b",
            overrides=["provider.name=vllm", "provider.package=vllm==0.14.0"],
        )
        assert config.name_or_path == "llama3.1-8b"
        assert config.provider is not None
        assert config.provider.name == "vllm"
        assert config.provider.package == "vllm==0.14.0"

    def test_parse_string_with_provider_github_url(self):
        """Test parsing model string with GitHub URL as provider package."""
        config = parse_model_config(
            "llama3.1-8b",
            overrides=[
                "provider.name=vllm",
                "provider.package=https://github.com/user/vllm@branch",
            ],
        )
        assert config.provider is not None
        assert config.provider.name == "vllm"
        assert config.provider.package == "https://github.com/user/vllm@branch"

    def test_parse_string_with_provider_name_only(self):
        """Test parsing model string with only provider name (no package)."""
        config = parse_model_config("llama3.1-8b", overrides=["provider.name=litellm"])
        assert config.provider is not None
        assert config.provider.name == "litellm"
        assert config.provider.package is None

    def test_parse_dict_with_nested_provider(self):
        """Test parsing dict with nested provider config."""
        config = parse_model_config(
            {
                "name_or_path": "llama3.1-8b",
                "provider": {
                    "name": "vllm",
                    "package": "vllm==0.14.0",
                },
            }
        )
        assert config.name_or_path == "llama3.1-8b"
        assert config.provider is not None
        assert config.provider.name == "vllm"
        assert config.provider.package == "vllm==0.14.0"

    def test_parse_dict_with_provider_name_only(self):
        """Test parsing dict with provider name only."""
        config = parse_model_config(
            {
                "name_or_path": "llama3.1-8b",
                "provider": {
                    "name": "litellm",
                },
            }
        )
        assert config.provider is not None
        assert config.provider.name == "litellm"
        assert config.provider.package is None

    def test_parse_string_with_mixed_overrides(self):
        """Test parsing model string with provider and other overrides."""
        config = parse_model_config(
            "llama3.1-8b",
            overrides=["provider.name=vllm", "load_format=auto"],
        )
        assert config.name_or_path == "llama3.1-8b"
        assert config.provider is not None
        assert config.provider.name == "vllm"
        assert config.load_format == "auto"


class TestEvalConfigWithProvider:
    """Tests for EvalConfig with ProviderConfig."""

    def test_get_model_resources_with_provider(self):
        """Test get_model_resources extracts provider name and package."""
        config = EvalConfig(
            name="test",
            models=["llama3.1-8b"],
            tasks=["mmlu"],
        )
        model = BeakerModelSpec(
            name_or_path="llama3.1-8b",
            provider=ProviderConfig(name="vllm", package="vllm==0.14.0"),
        )
        resources = config.get_model_resources(model)

        assert resources["provider"] == "vllm"
        assert resources["provider_package"] == "vllm==0.14.0"

    def test_get_model_resources_without_provider(self):
        """Test get_model_resources with no provider config."""
        config = EvalConfig(
            name="test",
            models=["llama3.1-8b"],
            tasks=["mmlu"],
        )
        model = BeakerModelSpec(name_or_path="llama3.1-8b")
        resources = config.get_model_resources(model)

        assert resources["provider"] is None
        assert resources["provider_package"] is None

    def test_from_yaml_with_nested_provider(self):
        """Test loading YAML with nested provider config."""
        yaml_content = """
name: test-eval
models:
  - name_or_path: llama3.1-8b
    provider:
      name: vllm
      package: vllm==0.14.0
tasks:
  - mmlu
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = EvalConfig.from_yaml(f.name)
            model_configs = config.get_model_configs()

            assert len(model_configs) == 1
            assert model_configs[0].provider is not None
            assert model_configs[0].provider.name == "vllm"
            assert model_configs[0].provider.package == "vllm==0.14.0"

    def test_from_yaml_with_github_provider(self):
        """Test loading YAML with GitHub URL provider package."""
        yaml_content = """
name: test-eval
models:
  - name_or_path: llama3.1-8b
    provider:
      name: vllm
      package: https://github.com/davidheineman/vllm@my-branch
tasks:
  - mmlu
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = EvalConfig.from_yaml(f.name)
            model_configs = config.get_model_configs()

            assert model_configs[0].provider is not None
            assert (
                model_configs[0].provider.package
                == "https://github.com/davidheineman/vllm@my-branch"
            )

    def test_from_yaml_with_provider_name_only(self):
        """Test loading YAML with provider name only (no custom package)."""
        yaml_content = """
name: test-eval
models:
  - name_or_path: gpt-4o
    provider:
      name: litellm
tasks:
  - mmlu
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = EvalConfig.from_yaml(f.name)
            model_configs = config.get_model_configs()

            assert model_configs[0].provider is not None
            assert model_configs[0].provider.name == "litellm"
            assert model_configs[0].provider.package is None
