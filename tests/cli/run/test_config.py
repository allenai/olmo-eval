"""Tests for run command configuration assembly."""

from olmo_eval.cli.run.config import RunConfigBuilder
from olmo_eval.common.types import ProviderKind


class TestRunConfigBuilder:
    """Tests for RunConfigBuilder provider resolution."""

    def test_named_harness_preset_provider_kind_overrides_model_preset(self):
        """The default harness should run vLLM model presets through vllm_server."""
        builder = RunConfigBuilder(
            model="olmo-3-1025-7b",
            task=("humaneval",),
            output_dir="/tmp/results",
            harness_preset="default",
        )

        config = builder.build()

        assert config.provider_config.kind == ProviderKind.VLLM_SERVER
        assert config.provider_config.model == "allenai/Olmo-3-1025-7B"

    def test_model_preset_provider_kind_is_used_without_harness_preset(self):
        """Without a harness preset, model presets should keep their provider kind."""
        builder = RunConfigBuilder(
            model="olmo-3-1025-7b",
            task=("humaneval",),
            output_dir="/tmp/results",
        )

        config = builder.build()

        assert config.provider_config.kind == ProviderKind.VLLM
        assert config.provider_config.model == "allenai/Olmo-3-1025-7B"
