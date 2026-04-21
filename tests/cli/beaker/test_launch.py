"""Tests for beaker launch functionality."""

from __future__ import annotations

import pytest

from olmo_eval.inference.providers.config import ProviderConfig


class TestProviderConfigDependenciesValidation:
    """Tests for ProviderConfig.from_dict dependencies validation."""

    def test_dependencies_as_list_accepted(self):
        """Test that dependencies as a list is accepted."""
        config = ProviderConfig.from_dict(
            {"dependencies": ["https://github.com/user/repo@v1.0", "some-package==1.0"]}
        )
        assert config.dependencies == (
            "https://github.com/user/repo@v1.0",
            "some-package==1.0",
        )

    def test_dependencies_as_empty_list_accepted(self):
        """Test that empty dependencies list is accepted."""
        config = ProviderConfig.from_dict({"dependencies": []})
        assert config.dependencies == ()

    def test_dependencies_default_to_empty(self):
        """Test that dependencies default to empty tuple."""
        config = ProviderConfig.from_dict({})
        assert config.dependencies == ()

    def test_dependencies_as_string_rejected(self):
        """Test that dependencies as a string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ProviderConfig.from_dict({"dependencies": "https://github.com/user/repo@v1.0"})

        assert "must be a list, not a string" in str(exc_info.value)
        assert "https://github.com/user/repo@v1.0" in str(exc_info.value)
        assert "provider.dependencies=[url1,url2]" in str(exc_info.value)

    def test_dependencies_string_does_not_iterate_chars(self):
        """Test that string dependencies don't silently iterate to characters.

        This is the bug we're preventing - tuple("abc") gives ('a', 'b', 'c').
        """
        with pytest.raises(ValueError):
            ProviderConfig.from_dict({"dependencies": "abc"})


class TestHarnessOverridesProviderDependencies:
    """Tests for harness overrides with provider.dependencies."""

    def test_apply_harness_overrides_with_list_dependencies(self):
        """Test that harness overrides with list dependencies work."""
        from olmo_eval.cli.beaker.launch import _apply_harness_overrides
        from olmo_eval.harness import get_harness_preset

        preset = get_harness_preset("default")

        # Apply override with list syntax (as OmegaConf parses it)
        overrides = ["provider.dependencies=[https://github.com/user/repo@v1.0]"]
        result = _apply_harness_overrides(preset, overrides)

        assert result.provider.dependencies == ("https://github.com/user/repo@v1.0",)

    def test_apply_harness_overrides_with_string_dependencies_raises(self):
        """Test that harness overrides with string dependencies raises error."""
        from olmo_eval.cli.beaker.launch import _apply_harness_overrides
        from olmo_eval.harness import get_harness_preset

        preset = get_harness_preset("default")

        # This is what happens when user forgets brackets
        overrides = ["provider.dependencies=https://github.com/user/repo@v1.0"]

        with pytest.raises(ValueError, match="must be a list, not a string"):
            _apply_harness_overrides(preset, overrides)

    def test_harness_provider_dependencies_in_job_config(self):
        """Test that harness provider.dependencies end up in BeakerJobConfig."""
        from unittest.mock import patch

        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler

        # Create a minimal launch config with harness overrides
        launch_config = LaunchConfig(
            name="test",
            model_specs=["test-model"],
            task_specs=["humaneval"],  # Use real task
            cluster="h100",
            workspace="ai2/test",
            budget="ai2/test",
            harness="default",
            harness_overrides=["provider.dependencies=[https://github.com/user/repo@v1.0]"],
        )

        # Create a minimal experiment plan
        exp = ExperimentPlan(
            name="test",
            model_spec="test-model",
            priority="normal",
            tasks=["humaneval"],  # Use real task
            original_task_specs=["humaneval"],
            total_expanded_tasks=1,
            num_gpus=1,
        )

        # Create job assembler with minimal required params
        assembler = JobConfigAssembler(
            config=launch_config,
            effective_image="test-image",
            effective_groups=[],
            beaker_username="test-user",
            common_secrets=[],
            store_secrets=[],
            task_secrets=[],
            inject_aws_credentials=False,
            inject_gcs_credentials=False,
        )

        # Mock cluster_has_weka to avoid cluster lookups
        with patch("olmo_eval.cli.beaker.job_assembler.cluster_has_weka", return_value=False):
            job_config = assembler.assemble(exp)

        # The harness override should have been extracted
        assert job_config.provider_packages is not None
        assert "https://github.com/user/repo@v1.0" in job_config.provider_packages

    def test_provider_package_overrides_vllm_extra(self):
        """Test that provider.package overrides the default vllm extra."""
        from olmo_eval.cli.beaker.launch import _apply_harness_overrides
        from olmo_eval.harness import get_harness_preset

        preset = get_harness_preset("default")

        # Apply provider.package override
        overrides = ["provider.package=https://github.com/user/vllm@custom"]
        result = _apply_harness_overrides(preset, overrides)

        assert result.provider.package == "https://github.com/user/vllm@custom"

    def test_provider_package_skips_vllm_extra_in_job_config(self):
        """Test that provider.package causes vllm extra to be skipped."""
        from unittest.mock import patch

        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler

        # Create config with provider.package override
        launch_config = LaunchConfig(
            name="test",
            model_specs=["test-model"],
            task_specs=["humaneval"],
            cluster="h100",
            workspace="ai2/test",
            budget="ai2/test",
            harness="default",
            harness_overrides=["provider.package=https://github.com/user/vllm@custom"],
        )

        exp = ExperimentPlan(
            name="test",
            model_spec="test-model",
            priority="normal",
            tasks=["humaneval"],
            original_task_specs=["humaneval"],
            total_expanded_tasks=1,
            num_gpus=1,
        )

        assembler = JobConfigAssembler(
            config=launch_config,
            effective_image="test-image",
            effective_groups=[],
            beaker_username="test-user",
            common_secrets=[],
            store_secrets=[],
            task_secrets=[],
            inject_aws_credentials=False,
            inject_gcs_credentials=False,
        )

        with patch("olmo_eval.cli.beaker.job_assembler.cluster_has_weka", return_value=False):
            job_config = assembler.assemble(exp)

        # vllm should NOT be in extras (provider.package overrides it)
        assert "vllm" not in job_config.extras
        # But provider.package should be in provider_packages
        assert job_config.provider_packages is not None
        assert "https://github.com/user/vllm@custom" in job_config.provider_packages

    def test_apply_harness_overrides_with_global_sandbox_override(self):
        """Test that sandboxes={...} sets the shared pool and common sandbox fields."""
        from olmo_eval.cli.beaker.launch import _apply_harness_overrides
        from olmo_eval.harness import get_harness_preset
        from olmo_eval.harness.sandbox import SandboxMode

        preset = get_harness_preset("codex_universal")

        result = _apply_harness_overrides(
            preset,
            ['sandboxes={"mode":"modal","instances":64,"registry_auth":{"provider":"gcp"}}'],
        )

        assert result.sandbox_pool_instances == 64
        assert all(sandbox.mode == SandboxMode.MODAL for sandbox in result.sandboxes)
        assert all(sandbox.registry_auth is not None for sandbox in result.sandboxes)
        assert all(sandbox.registry_auth.provider == "gcp" for sandbox in result.sandboxes)


class TestTaskExpansionInExperimentSummary:
    """Tests for task expansion in _build_experiment_summary."""

    def test_expanded_tasks_match_task_configs(self):
        """Test that expanded tasks are used to lookup task configs."""
        from olmo_eval.common.configs import expand_tasks

        # Verify that minerva_math:olmo3base expands to multiple tasks
        expanded = expand_tasks(["minerva_math:olmo3base"])
        assert len(expanded) > 1

        # The expanded tasks should have different names than the input
        assert "minerva_math:olmo3base" not in expanded
        # But should contain minerva_math variants
        assert any("minerva_math" in t for t in expanded)

    def test_build_experiment_summary_uses_expanded_tasks(self):
        """Test that _build_experiment_summary expands tasks before lookup."""
        from unittest.mock import MagicMock, patch

        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.launch import _build_experiment_summary
        from olmo_eval.common.configs import expand_tasks

        # Create experiment with unexpanded task spec
        exp = ExperimentPlan(
            name="test",
            model_spec="allenai/test-model",
            priority="normal",
            tasks=["minerva_math:olmo3base"],  # Unexpanded suite name
            original_task_specs=["minerva_math:olmo3base"],
            total_expanded_tasks=7,
            num_gpus=1,
        )

        # Create mock job config
        mock_job_config = MagicMock()

        # Get the actual expanded task names
        expanded = expand_tasks(["minerva_math:olmo3base"])

        # Create task_configs_by_spec keyed by EXPANDED task names
        # (This simulates how _get_task_configs works)
        task_configs_by_spec = {}
        for task_name in expanded:
            mock_config = MagicMock()
            mock_config.name = task_name
            task_configs_by_spec[task_name] = mock_config

        # Patch the harness and provider lookups that happen inside the function
        with patch("olmo_eval.harness.get_harness_preset") as mock_harness:
            mock_harness_config = MagicMock()
            mock_harness_config.backend = None
            mock_harness_config.sandboxes = ()
            mock_harness_config.merge_provider = MagicMock(return_value=mock_harness_config)
            mock_harness.return_value = mock_harness_config

            with patch("olmo_eval.common.configs.get_provider_config") as mock_provider:
                mock_provider.return_value = MagicMock()

                summary = _build_experiment_summary(exp, mock_job_config, task_configs_by_spec)

        # The summary should have found all expanded tasks
        # because _build_experiment_summary now expands the task specs
        assert len(summary.tasks) == len(expanded)

    def test_unexpanded_task_spec_would_miss_lookup(self):
        """Test that without expansion, task lookup would fail.

        This documents the bug we fixed - unexpanded task specs
        don't match keys in task_configs_by_spec.
        """
        from olmo_eval.common.configs import expand_tasks

        # The suite name
        suite_spec = "minerva_math:olmo3base"

        # The expanded tasks have different names
        expanded = expand_tasks([suite_spec])

        # Create dict keyed by expanded names
        task_configs_by_spec = {task: f"config_for_{task}" for task in expanded}

        # The unexpanded suite name is NOT a key
        assert suite_spec not in task_configs_by_spec

        # But expanded tasks ARE keys
        for task in expanded:
            assert task in task_configs_by_spec
