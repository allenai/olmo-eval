"""Tests for olmo_eval.launch.beaker module."""

import pytest

from olmo_eval.launch.beaker import (
    BeakerEnvSecret,
    BeakerJobConfig,
    BeakerWekaBucket,
    _parse_timeout,
    calculate_experiment_splits,
    normalize_provider_package,
    parse_task_with_priority,
    resolve_clusters,
    validate_priority_configuration,
)


class TestResolveClustors:
    """Tests for cluster resolution."""

    def test_resolve_h100_alias(self):
        """Test resolving h100 alias."""
        clusters = resolve_clusters("h100")
        assert "ai2/jupiter" in clusters
        assert "ai2/ceres" in clusters

    def test_resolve_a100_alias(self):
        """Test resolving a100 alias."""
        clusters = resolve_clusters("a100")
        assert clusters == ["ai2/saturn"]

    def test_resolve_aus_alias(self):
        """Test resolving aus alias."""
        clusters = resolve_clusters("aus")
        assert "ai2/jupiter" in clusters
        assert "ai2/neptune" in clusters
        assert "ai2/saturn" in clusters
        assert "ai2/ceres" in clusters

    def test_resolve_full_name(self):
        """Test that full cluster names pass through."""
        clusters = resolve_clusters("ai2/jupiter")
        assert clusters == ["ai2/jupiter"]

    def test_resolve_list_of_clusters(self):
        """Test resolving a list of clusters."""
        clusters = resolve_clusters(["ai2/jupiter", "ai2/saturn"])
        assert "ai2/jupiter" in clusters
        assert "ai2/saturn" in clusters

    def test_resolve_mixed_aliases_and_names(self):
        """Test resolving mixed aliases and full names."""
        clusters = resolve_clusters(["h100", "ai2/saturn"])
        assert "ai2/jupiter" in clusters
        assert "ai2/ceres" in clusters
        assert "ai2/saturn" in clusters

    def test_resolve_legacy_cluster_name(self):
        """Test resolving legacy cluster names."""
        clusters = resolve_clusters("ai2/jupiter-cirrascale-2")
        assert clusters == ["ai2/jupiter"]

    def test_resolve_deduplicates(self):
        """Test that duplicate clusters are removed."""
        clusters = resolve_clusters(["h100", "ai2/jupiter"])
        assert clusters.count("ai2/jupiter") == 1


class TestParseTimeout:
    """Tests for timeout parsing."""

    def test_parse_hours(self):
        """Test parsing hours."""
        ns = _parse_timeout("24h")
        assert ns == 24 * 3600_000_000_000

    def test_parse_minutes(self):
        """Test parsing minutes."""
        ns = _parse_timeout("30m")
        assert ns == 30 * 60_000_000_000

    def test_parse_seconds(self):
        """Test parsing seconds."""
        ns = _parse_timeout("90s")
        assert ns == 90 * 1_000_000_000

    def test_parse_combined(self):
        """Test parsing combined time units."""
        ns = _parse_timeout("1h30m")
        expected = 1 * 3600_000_000_000 + 30 * 60_000_000_000
        assert ns == expected

    def test_parse_invalid_returns_default(self):
        """Test that invalid timeout returns 24h default."""
        ns = _parse_timeout("invalid")
        assert ns == 86400_000_000_000  # 24h in ns


class TestBeakerEnvSecret:
    """Tests for BeakerEnvSecret."""

    def test_creation(self):
        """Test creating a secret."""
        secret = BeakerEnvSecret(name="HF_TOKEN", secret="my_hf_token")
        assert secret.name == "HF_TOKEN"
        assert secret.secret == "my_hf_token"


class TestBeakerWekaBucket:
    """Tests for BeakerWekaBucket."""

    def test_default_mount(self):
        """Test that mount path is auto-generated."""
        bucket = BeakerWekaBucket(bucket="oe-eval-default")
        assert bucket.bucket == "oe-eval-default"
        assert bucket.mount == "/weka/oe-eval-default"

    def test_custom_mount(self):
        """Test custom mount path."""
        bucket = BeakerWekaBucket(bucket="oe-eval-default", mount="/custom/path")
        assert bucket.mount == "/custom/path"


class TestBeakerJobConfig:
    """Tests for BeakerJobConfig."""

    def test_minimal_config(self):
        """Test creating minimal config with required fields."""
        config = BeakerJobConfig(
            name="test-job",
            command=["echo", "hello"],
            cluster="h100",
            workspace="ai2/oe-data",
            budget="ai2/oe-base",
        )
        assert config.name == "test-job"
        assert config.command == ["echo", "hello"]
        assert config.num_gpus == 1
        assert config.cluster == "h100"
        assert config.workspace == "ai2/oe-data"
        assert config.budget == "ai2/oe-base"
        assert config.priority == "normal"
        assert config.preemptible is True
        assert config.timeout == "24h"

    def test_full_config(self):
        """Test creating full config with all options."""
        config = BeakerJobConfig(
            name="test-job",
            command=["olmo-eval", "run", "-m", "llama3.1-8b", "-t", "mmlu"],
            num_gpus=4,
            shared_memory="20GiB",
            cluster=["ai2/jupiter", "ai2/saturn"],
            priority="high",
            preemptible=False,
            timeout="48h",
            retries=2,
            workspace="ai2/custom-workspace",
            budget="ai2/custom-budget",
            beaker_image="custom-image",
            description="Test description",
            weka_buckets=[BeakerWekaBucket("custom-bucket")],
            nfs=True,
            env_vars={"CUSTOM_VAR": "value"},
            env_secrets=[BeakerEnvSecret("CUSTOM_SECRET", "secret_name")],
        )
        assert config.num_gpus == 4
        assert config.cluster == ["ai2/jupiter", "ai2/saturn"]
        assert config.priority == "high"
        assert config.preemptible is False
        assert config.retries == 2
        assert config.nfs is True
        assert len(config.weka_buckets) == 1

    def test_default_weka_buckets(self):
        """Test default Weka buckets are set and properly configured."""
        config = BeakerJobConfig(
            name="test",
            command=["echo"],
            cluster="h100",
            workspace="ai2/oe-data",
            budget="ai2/oe-base",
        )
        # Just verify defaults exist and have valid mount paths
        assert len(config.weka_buckets) >= 1
        for bucket in config.weka_buckets:
            assert bucket.bucket  # Non-empty bucket name
            assert bucket.mount and bucket.mount.startswith("/weka/")

    def test_default_secrets(self):
        """Test default env_secrets is empty (secrets are added during launch)."""
        config = BeakerJobConfig(
            name="test",
            command=["echo"],
            cluster="h100",
            workspace="ai2/oe-data",
            budget="ai2/oe-base",
        )
        # env_secrets defaults to empty list; secrets are injected during launch
        assert len(config.env_secrets) == 0


class TestBeakerLauncherImport:
    """Tests for BeakerLauncher import behavior."""

    def test_launcher_imports_without_beaker(self):
        """Test that BeakerLauncher can be imported without beaker-py installed.

        The actual beaker import should be lazy (only when using the launcher).
        """
        from olmo_eval.launch import BeakerLauncher

        # Should be able to instantiate without error
        launcher = BeakerLauncher()
        assert launcher._beaker is None

    def test_config_imports_work(self):
        """Test that config classes can be imported."""
        from olmo_eval.launch import (
            BeakerEnvSecret,
            BeakerJobConfig,
            BeakerWekaBucket,
        )

        # All should be importable
        assert BeakerEnvSecret is not None
        assert BeakerJobConfig is not None
        assert BeakerWekaBucket is not None


class TestParseTaskWithPriority:
    """Tests for task priority parsing."""

    def test_task_only_uses_default(self):
        """Test task without priority uses default."""
        task, priority = parse_task_with_priority("mmlu")
        assert task == "mmlu"
        assert priority == "normal"

    def test_task_with_priority(self):
        """Test task with @priority suffix."""
        task, priority = parse_task_with_priority("mmlu@high")
        assert task == "mmlu"
        assert priority == "high"

    def test_task_with_regime_and_priority(self):
        """Test task with regime and priority."""
        task, priority = parse_task_with_priority("mmlu:olmes@high")
        assert task == "mmlu:olmes"
        assert priority == "high"

    def test_custom_default_priority(self):
        """Test using custom default priority."""
        task, priority = parse_task_with_priority("mmlu", default_priority="high")
        assert task == "mmlu"
        assert priority == "high"

    def test_explicit_priority_overrides_default(self):
        """Test that explicit @priority overrides default."""
        task, priority = parse_task_with_priority("mmlu@low", default_priority="high")
        assert task == "mmlu"
        assert priority == "low"

    def test_all_valid_priorities(self):
        """Test all valid priority values."""
        for p in ("low", "normal", "high", "urgent"):
            task, priority = parse_task_with_priority(f"mmlu@{p}")
            assert priority == p

    def test_invalid_priority_raises(self):
        """Test that invalid priority raises ValueError."""
        with pytest.raises(ValueError, match="Invalid priority"):
            parse_task_with_priority("mmlu@invalid")

    def test_invalid_priority_error_message(self):
        """Test error message includes valid options."""
        with pytest.raises(ValueError, match="low, normal, high, urgent"):
            parse_task_with_priority("mmlu@bad")


class TestValidatePriorityConfiguration:
    """Tests for priority configuration validation."""

    def test_tasks_without_priority_no_cli_flag(self):
        """Test tasks without @priority suffix and no CLI flag use default."""
        result = validate_priority_configuration(["mmlu", "gsm8k"], None)
        assert result == {"normal": ["mmlu", "gsm8k"]}

    def test_tasks_without_priority_with_cli_flag(self):
        """Test tasks without @priority suffix use CLI priority."""
        result = validate_priority_configuration(["mmlu", "gsm8k"], "high")
        assert result == {"high": ["mmlu", "gsm8k"]}

    def test_tasks_with_priority_no_cli_flag(self):
        """Test tasks with @priority suffixes are grouped correctly."""
        result = validate_priority_configuration(["mmlu@high", "gsm8k@normal", "arc@high"], None)
        assert result == {"high": ["mmlu", "arc"], "normal": ["gsm8k"]}

    def test_tasks_with_priority_and_cli_flag_raises(self):
        """Test that using CLI --priority with @priority suffixes raises error."""
        with pytest.raises(ValueError, match="Conflicting priority specification"):
            validate_priority_configuration(["mmlu@high", "gsm8k"], "normal")

    def test_conflict_error_message_shows_tasks(self):
        """Test that conflict error message lists tasks with @priority suffixes."""
        with pytest.raises(ValueError, match="mmlu@high"):
            validate_priority_configuration(["mmlu@high"], "normal")

    def test_mixed_tasks_no_cli_flag(self):
        """Test mixed tasks (some with, some without @priority) and no CLI flag."""
        result = validate_priority_configuration(["mmlu@high", "gsm8k", "arc@low"], None)
        # gsm8k should use default "normal"
        assert result == {"high": ["mmlu"], "normal": ["gsm8k"], "low": ["arc"]}

    def test_custom_default_priority(self):
        """Test custom default priority for tasks without @priority suffix."""
        result = validate_priority_configuration(["mmlu", "gsm8k"], None, default_priority="high")
        assert result == {"high": ["mmlu", "gsm8k"]}

    def test_all_priority_levels(self):
        """Test all valid priority levels work."""
        result = validate_priority_configuration(["a@low", "b@normal", "c@high", "d@urgent"], None)
        assert result == {"low": ["a"], "normal": ["b"], "high": ["c"], "urgent": ["d"]}

    def test_empty_tasks_list(self):
        """Test empty tasks list returns empty dict."""
        result = validate_priority_configuration([], None)
        assert result == {}

    def test_tuple_input(self):
        """Test that tuple input works (from CLI)."""
        result = validate_priority_configuration(("mmlu", "gsm8k"), None)
        assert result == {"normal": ["mmlu", "gsm8k"]}

    def test_task_with_regime_and_priority(self):
        """Test task with regime (:) and @priority suffix."""
        result = validate_priority_configuration(["mmlu:olmes@high"], None)
        assert result == {"high": ["mmlu:olmes"]}


class TestCalculateExperimentSplits:
    """Tests for calculate_experiment_splits function."""

    def test_single_node_no_split(self):
        """Test case where total GPUs fit on single node."""
        # 4 instances × 2 GPUs = 8 GPUs (fits on 8-GPU node)
        result = calculate_experiment_splits(
            tasks=["a", "b", "c", "d"],
            gpus_per_model=2,
            parallelism=4,
            max_gpus_per_node=8,
        )
        assert len(result) == 1
        assert result[0]["tasks"] == ["a", "b", "c", "d"]
        assert result[0]["num_gpus"] == 8
        assert result[0]["parallelism"] == 4

    def test_split_into_two_experiments(self):
        """Test case requiring split into 2 experiments."""
        # 4 instances × 4 GPUs = 16 GPUs, max 8 per node = 2 experiments
        result = calculate_experiment_splits(
            tasks=["a", "b", "c", "d"],
            gpus_per_model=4,
            parallelism=4,
            max_gpus_per_node=8,
        )
        assert len(result) == 2
        # Each experiment gets 2 instances (8 GPUs) and 2 tasks
        assert result[0]["tasks"] == ["a", "b"]
        assert result[0]["num_gpus"] == 8
        assert result[0]["parallelism"] == 2
        assert result[1]["tasks"] == ["c", "d"]
        assert result[1]["num_gpus"] == 8
        assert result[1]["parallelism"] == 2

    def test_split_into_multiple_experiments(self):
        """Test case requiring split into many experiments."""
        # 8 instances × 4 GPUs = 32 GPUs, max 8 per node = 4 experiments
        result = calculate_experiment_splits(
            tasks=["a", "b", "c", "d", "e", "f", "g", "h"],
            gpus_per_model=4,
            parallelism=8,
            max_gpus_per_node=8,
        )
        assert len(result) == 4
        for split in result:
            assert split["num_gpus"] == 8
            assert split["parallelism"] == 2

    def test_single_task_no_split(self):
        """Test with single task, no split needed."""
        result = calculate_experiment_splits(
            tasks=["mmlu"],
            gpus_per_model=2,
            parallelism=4,
            max_gpus_per_node=8,
        )
        assert len(result) == 1
        assert result[0]["tasks"] == ["mmlu"]
        assert result[0]["num_gpus"] == 8
        assert result[0]["parallelism"] == 4

    def test_single_task_with_split(self):
        """Test single task distributed across splits."""
        # With 1 task and 2 required experiments, task goes to first experiment
        result = calculate_experiment_splits(
            tasks=["mmlu"],
            gpus_per_model=4,
            parallelism=4,
            max_gpus_per_node=8,
        )
        assert len(result) == 1
        assert result[0]["tasks"] == ["mmlu"]
        assert result[0]["num_gpus"] == 8
        assert result[0]["parallelism"] == 2

    def test_parallelism_one_no_split(self):
        """Test with parallelism=1, should never split."""
        result = calculate_experiment_splits(
            tasks=["a", "b", "c"],
            gpus_per_model=4,
            parallelism=1,
            max_gpus_per_node=8,
        )
        assert len(result) == 1
        assert result[0]["tasks"] == ["a", "b", "c"]
        assert result[0]["num_gpus"] == 4
        assert result[0]["parallelism"] == 1

    def test_exactly_fits_node(self):
        """Test when total GPUs exactly equal max per node."""
        # 2 instances × 4 GPUs = 8 GPUs = max
        result = calculate_experiment_splits(
            tasks=["a", "b"],
            gpus_per_model=4,
            parallelism=2,
            max_gpus_per_node=8,
        )
        assert len(result) == 1
        assert result[0]["num_gpus"] == 8
        assert result[0]["parallelism"] == 2

    def test_uneven_task_distribution(self):
        """Test with odd number of tasks split unevenly."""
        # 3 tasks split across 2 experiments
        result = calculate_experiment_splits(
            tasks=["a", "b", "c"],
            gpus_per_model=4,
            parallelism=4,
            max_gpus_per_node=8,
        )
        assert len(result) == 2
        assert result[0]["tasks"] == ["a", "b"]
        assert result[1]["tasks"] == ["c"]

    def test_more_experiments_than_tasks(self):
        """Test when splitting would create more experiments than tasks."""
        # 4 experiments needed, but only 2 tasks
        result = calculate_experiment_splits(
            tasks=["a", "b"],
            gpus_per_model=4,
            parallelism=8,
            max_gpus_per_node=8,
        )
        # Should create experiments for all tasks, even if some splits are empty
        assert len(result) == 2
        assert result[0]["tasks"] == ["a"]
        assert result[1]["tasks"] == ["b"]

    def test_gpu_calculation(self):
        """Test that GPU calculations are correct."""
        # 3 instances × 2 GPUs = 6 GPUs (fits on 8-GPU node)
        result = calculate_experiment_splits(
            tasks=["a"],
            gpus_per_model=2,
            parallelism=3,
            max_gpus_per_node=8,
        )
        assert result[0]["num_gpus"] == 6
        assert result[0]["parallelism"] == 3

    def test_large_model_exceeds_node(self):
        """Test edge case where single model instance exceeds node GPUs."""
        # Model needs 16 GPUs but max is 8 - falls back to 1 instance
        result = calculate_experiment_splits(
            tasks=["a", "b"],
            gpus_per_model=16,
            parallelism=2,
            max_gpus_per_node=8,
        )
        # Should still work, using 1 instance per experiment
        assert len(result) == 2
        assert result[0]["num_gpus"] == 16
        assert result[0]["parallelism"] == 1


class TestBeakerJobConfigTaskPackages:
    """Tests for BeakerJobConfig task_packages field."""

    def test_task_packages_default_none(self):
        """Test that task_packages defaults to None."""
        config = BeakerJobConfig(
            name="test",
            command=["echo"],
            cluster="h100",
            workspace="ai2/oe-data",
            budget="ai2/oe-base",
        )
        assert config.task_packages is None

    def test_task_packages_can_be_set(self):
        """Test that task_packages can be set."""
        config = BeakerJobConfig(
            name="test",
            command=["echo"],
            cluster="h100",
            workspace="ai2/oe-data",
            budget="ai2/oe-base",
            task_packages=["special-lib==1.0", "git+https://github.com/user/repo"],
        )
        assert config.task_packages == ["special-lib==1.0", "git+https://github.com/user/repo"]


class TestBuildCommandWithTaskPackages:
    """Tests for command building with task packages."""

    def test_command_includes_task_packages(self):
        """Test that task packages are installed in the generated command."""
        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher()
        command = launcher._build_command_with_extras(
            command=["olmo-eval", "run"],
            extras=[],
            env_exports=None,
            provider_package=None,
            task_packages=["special-lib==1.0", "another-pkg"],
        )

        # The command should be a bash -c with the full script
        assert command[0] == "bash"
        assert command[1] == "-c"
        script = command[2]

        # Check that task packages are being installed
        assert "uv pip install 'special-lib==1.0'" in script
        assert "uv pip install 'another-pkg'" in script

    def test_task_packages_installed_after_provider(self):
        """Test that task packages are installed after provider package."""
        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher()
        command = launcher._build_command_with_extras(
            command=["olmo-eval", "run"],
            extras=[],
            env_exports=None,
            provider_package="vllm==0.14.0",
            task_packages=["task-dep==1.0"],
        )

        script = command[2]

        # Provider should be installed before task packages
        provider_pos = script.find("uv pip install 'vllm==0.14.0'")
        task_pos = script.find("uv pip install 'task-dep==1.0'")
        assert provider_pos < task_pos

    def test_no_task_packages_if_none(self):
        """Test that no extra install steps if task_packages is None."""
        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher()
        command = launcher._build_command_with_extras(
            command=["olmo-eval", "run"],
            extras=[],
            env_exports=None,
            provider_package=None,
            task_packages=None,
        )

        script = command[2]
        # Should only have the base install, not any extra pip install for task packages
        # Count occurrences of 'uv pip install' - should only be the base install
        install_count = script.count("uv pip install")
        assert install_count == 1  # Only the base olmo-eval install

    def test_task_packages_git_url_normalized(self):
        """Test that git URLs in task packages are normalized."""
        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher()
        command = launcher._build_command_with_extras(
            command=["olmo-eval", "run"],
            extras=[],
            env_exports=None,
            provider_package=None,
            task_packages=["https://github.com/user/repo@v1.0"],
        )

        script = command[2]
        # GitHub URL should get git+ prefix
        assert "uv pip install 'git+https://github.com/user/repo@v1.0'" in script


class TestNormalizeProviderPackage:
    """Tests for normalize_provider_package function."""

    def test_github_url_gets_git_prefix(self):
        """Test GitHub URL gets git+ prefix."""
        result = normalize_provider_package("https://github.com/user/vllm")
        assert result == "git+https://github.com/user/vllm"

    def test_github_url_with_branch_gets_git_prefix(self):
        """Test GitHub URL with branch gets git+ prefix."""
        result = normalize_provider_package("https://github.com/user/vllm@my-branch")
        assert result == "git+https://github.com/user/vllm@my-branch"

    def test_gitlab_url_gets_git_prefix(self):
        """Test GitLab URL gets git+ prefix."""
        result = normalize_provider_package("https://gitlab.com/user/package")
        assert result == "git+https://gitlab.com/user/package"

    def test_git_plus_url_unchanged(self):
        """Test git+ URL passes through unchanged."""
        result = normalize_provider_package("git+https://github.com/user/vllm@v0.14.0")
        assert result == "git+https://github.com/user/vllm@v0.14.0"

    def test_pypi_version_unchanged(self):
        """Test PyPI version spec passes through unchanged."""
        result = normalize_provider_package("vllm==0.14.0")
        assert result == "vllm==0.14.0"

    def test_pypi_version_range_unchanged(self):
        """Test PyPI version range passes through unchanged."""
        result = normalize_provider_package("vllm>=0.13.0,<0.15.0")
        assert result == "vllm>=0.13.0,<0.15.0"

    def test_pypi_with_extras_unchanged(self):
        """Test PyPI with extras passes through unchanged."""
        result = normalize_provider_package("vllm[runai]==0.14.0")
        assert result == "vllm[runai]==0.14.0"

    def test_local_path_unchanged(self):
        """Test local path passes through unchanged."""
        result = normalize_provider_package("/path/to/local/vllm")
        assert result == "/path/to/local/vllm"

    def test_package_name_only_unchanged(self):
        """Test package name only passes through unchanged."""
        result = normalize_provider_package("vllm")
        assert result == "vllm"


class TestModelPackingAndGrouping:
    """Tests for model packing and GPU grouping behavior."""

    def test_runtime_signature_excludes_gpus(self):
        """Models with different gpus but same provider/cluster are grouped together."""
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.model_grouper import ModelGrouper
        from olmo_eval.launch import BeakerModelSpec

        # Create models with different GPUs but same cluster/provider
        model1 = BeakerModelSpec(name_or_path="model1", gpus=1)
        model2 = BeakerModelSpec(name_or_path="model2", gpus=4)
        model3 = BeakerModelSpec(name_or_path="model3", gpus=2)

        config = LaunchConfig(
            name="test",
            model_configs=[model1, model2, model3],
            model_specs=["model1", "model2", "model3"],
            task_specs=["mmlu"],
            cluster="h100",
            workspace="test",
            budget="test",
        )

        grouper = ModelGrouper(config, eval_config=None)

        # All models should be in the same group (same signature)
        groups = grouper.group()
        assert len(groups) == 1
        assert len(groups[0]) == 3  # All 3 models in one group

    def test_runtime_signature_splits_by_cluster(self):
        """Models with different clusters are in separate groups."""
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.model_grouper import ModelGrouper
        from olmo_eval.launch import BeakerModelSpec

        model1 = BeakerModelSpec(name_or_path="model1", gpus=1, cluster="h100")
        model2 = BeakerModelSpec(name_or_path="model2", gpus=1, cluster="a100")

        config = LaunchConfig(
            name="test",
            model_configs=[model1, model2],
            model_specs=["model1", "model2"],
            task_specs=["mmlu"],
            cluster="h100",  # Default
            workspace="test",
            budget="test",
        )

        grouper = ModelGrouper(config, eval_config=None)
        groups = grouper.group()

        # Should be 2 groups (different clusters)
        assert len(groups) == 2

    def test_no_pack_creates_separate_experiments(self):
        """Default (pack_models=False) creates separate experiment per model."""
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_builder import ExperimentPlanBuilder
        from olmo_eval.cli.beaker.model_grouper import ModelGrouper
        from olmo_eval.launch import BeakerModelSpec

        model1 = BeakerModelSpec(name_or_path="model1", gpus=1)
        model2 = BeakerModelSpec(name_or_path="model2", gpus=1)

        config = LaunchConfig(
            name="test",
            model_configs=[model1, model2],
            model_specs=["model1", "model2"],
            task_specs=["mmlu"],
            cluster="h100",
            workspace="test",
            budget="test",
            pack_models=False,  # Default
        )

        grouper = ModelGrouper(config, eval_config=None)
        builder = ExperimentPlanBuilder(
            config, grouper, tasks_by_priority={"normal": ["mmlu"]}, agent_task_specs=set()
        )

        experiments, _ = builder.build()

        # Should have 2 experiments, one per model
        assert len(experiments) == 2
        assert len(experiments[0].model_cfgs) == 1
        assert len(experiments[1].model_cfgs) == 1
        assert experiments[0].num_gpus == 1
        assert experiments[1].num_gpus == 1

    def test_pack_groups_models_together(self):
        """With pack_models=True, compatible models are grouped together."""
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_builder import ExperimentPlanBuilder
        from olmo_eval.cli.beaker.model_grouper import ModelGrouper
        from olmo_eval.launch import BeakerModelSpec

        model1 = BeakerModelSpec(name_or_path="model1", gpus=1)
        model2 = BeakerModelSpec(name_or_path="model2", gpus=1)

        config = LaunchConfig(
            name="test",
            model_configs=[model1, model2],
            model_specs=["model1", "model2"],
            task_specs=["mmlu"],
            cluster="h100",
            workspace="test",
            budget="test",
            pack_models=True,
            max_gpus_per_node=8,
        )

        grouper = ModelGrouper(config, eval_config=None)
        builder = ExperimentPlanBuilder(
            config, grouper, tasks_by_priority={"normal": ["mmlu"]}, agent_task_specs=set()
        )

        experiments, _ = builder.build()

        # Should have 1 experiment with both models
        assert len(experiments) == 1
        assert len(experiments[0].model_cfgs) == 2
        assert experiments[0].num_gpus == 2  # 1 + 1

    def test_pack_mixed_gpu_counts(self):
        """Packed models with different GPU counts in single experiment."""
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_builder import ExperimentPlanBuilder
        from olmo_eval.cli.beaker.model_grouper import ModelGrouper
        from olmo_eval.launch import BeakerModelSpec

        # 1 + 1 + 6 = 8 GPUs total
        model1 = BeakerModelSpec(name_or_path="model1", gpus=1)
        model2 = BeakerModelSpec(name_or_path="model2", gpus=1)
        model3 = BeakerModelSpec(name_or_path="model3", gpus=6)

        config = LaunchConfig(
            name="test",
            model_configs=[model1, model2, model3],
            model_specs=["model1", "model2", "model3"],
            task_specs=["mmlu"],
            cluster="h100",
            workspace="test",
            budget="test",
            pack_models=True,
            max_gpus_per_node=8,
        )

        grouper = ModelGrouper(config, eval_config=None)
        builder = ExperimentPlanBuilder(
            config, grouper, tasks_by_priority={"normal": ["mmlu"]}, agent_task_specs=set()
        )

        experiments, _ = builder.build()

        # Should fit in 1 experiment (8 GPUs total)
        assert len(experiments) == 1
        assert experiments[0].num_gpus == 8
        assert experiments[0].model_gpu_counts == [1, 1, 6]

    def test_pack_splits_when_exceeds_max(self):
        """Packed models split when total exceeds max_gpus_per_node."""
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_builder import ExperimentPlanBuilder
        from olmo_eval.cli.beaker.model_grouper import ModelGrouper
        from olmo_eval.launch import BeakerModelSpec

        # 4 + 6 = 10 GPUs, exceeds max of 8
        model1 = BeakerModelSpec(name_or_path="model1", gpus=4)
        model2 = BeakerModelSpec(name_or_path="model2", gpus=6)

        config = LaunchConfig(
            name="test",
            model_configs=[model1, model2],
            model_specs=["model1", "model2"],
            task_specs=["mmlu"],
            cluster="h100",
            workspace="test",
            budget="test",
            pack_models=True,
            max_gpus_per_node=8,
        )

        grouper = ModelGrouper(config, eval_config=None)
        builder = ExperimentPlanBuilder(
            config, grouper, tasks_by_priority={"normal": ["mmlu"]}, agent_task_specs=set()
        )

        experiments, split_models = builder.build()

        # Should split into 2 experiments
        assert len(experiments) == 2
        assert experiments[0].num_gpus == 4
        assert experiments[1].num_gpus == 6
        assert len(split_models) > 0  # At least one model was split

    def test_model_gpu_counts_in_experiment(self):
        """Experiment plan includes model_gpu_counts for each model."""
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_builder import ExperimentPlanBuilder
        from olmo_eval.cli.beaker.model_grouper import ModelGrouper
        from olmo_eval.launch import BeakerModelSpec

        model1 = BeakerModelSpec(name_or_path="model1", gpus=2)
        model2 = BeakerModelSpec(name_or_path="model2", gpus=4)

        config = LaunchConfig(
            name="test",
            model_configs=[model1, model2],
            model_specs=["model1", "model2"],
            task_specs=["mmlu"],
            cluster="h100",
            workspace="test",
            budget="test",
            pack_models=True,
            max_gpus_per_node=8,
        )

        grouper = ModelGrouper(config, eval_config=None)
        builder = ExperimentPlanBuilder(
            config, grouper, tasks_by_priority={"normal": ["mmlu"]}, agent_task_specs=set()
        )

        experiments, _ = builder.build()

        assert len(experiments) == 1
        assert experiments[0].model_gpu_counts == [2, 4]
        assert experiments[0].num_gpus == 6


@pytest.fixture
def mock_tasks():
    """Register mock tasks for testing JobConfigAssembler."""
    from collections.abc import Iterator

    from olmo_eval.core.types import Instance, LMOutput, LMRequest, RequestType
    from olmo_eval.evals.tasks import Task, TaskConfig, clear_registry, register
    from olmo_eval.evals.tasks.core.registry import _configs, _regimes, _tasks, _variants

    # Save original state
    original_tasks = _tasks.copy()
    original_configs = _configs.copy()
    original_regimes = {k: v.copy() for k, v in _regimes.items()}
    original_variants = {k: v.copy() for k, v in _variants.items()}

    class MockTask(Task):
        @property
        def instances(self) -> Iterator[Instance]:
            yield Instance(question="test", gold_answer="test")

        def format_request(self, instance: Instance) -> LMRequest:
            return LMRequest(request_type=RequestType.COMPLETION, prompt="test")

        def extract_answer(self, output: LMOutput) -> str:
            return output.text.strip()

    # Register mock tasks used in tests
    @register("mmlu", lambda: TaskConfig(name="mmlu", data_source="test/dataset"))
    class MMLUTask(MockTask):
        pass

    @register("gsm8k", lambda: TaskConfig(name="gsm8k", data_source="test/dataset"))
    class GSM8KTask(MockTask):
        pass

    @register(
        "task_with_deps",
        lambda: TaskConfig(
            name="task_with_deps", data_source="test/dataset", dependencies=["base-pkg"]
        ),
    )
    class TaskWithDeps(MockTask):
        pass

    yield

    # Restore original state
    clear_registry()
    _tasks.update(original_tasks)
    _configs.update(original_configs)
    _regimes.update(original_regimes)
    _variants.update(original_variants)


class TestTaskOverrideHandling:
    """Tests for task override extraction and command generation."""

    def test_priority_extracted_from_task_overrides(self):
        """Priority in task overrides sets experiment priority."""
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_builder import ExperimentPlanBuilder
        from olmo_eval.cli.beaker.model_grouper import ModelGrouper
        from olmo_eval.cli.utils import extract_priority_from_overrides
        from olmo_eval.launch import BeakerModelSpec

        # Raw task overrides include priority
        raw_task_overrides = {"mmlu": ["priority=urgent", "limit=10"]}

        # Extract priority before creating builder (as done in launch.py)
        override_priority, filtered_task_overrides = extract_priority_from_overrides(
            raw_task_overrides
        )

        config = LaunchConfig(
            name="test",
            model_configs=[BeakerModelSpec(name_or_path="model1")],
            model_specs=["model1"],
            task_specs=["mmlu"],
            cluster="h100",
            workspace="test",
            budget="test",
            task_overrides=filtered_task_overrides,  # Already filtered
        )

        grouper = ModelGrouper(config, eval_config=None)
        builder = ExperimentPlanBuilder(
            config,
            grouper,
            tasks_by_priority={"normal": ["mmlu"]},
            agent_task_specs=set(),
            override_priority=override_priority,  # Pass extracted priority
        )

        experiments, _ = builder.build()

        # Priority should be extracted and used for experiment
        assert experiments[0].priority == "urgent"
        # Task overrides should NOT include priority (it's not a task config field)
        assert "priority=urgent" not in experiments[0].task_overrides.get("mmlu", [])
        # Other overrides should be preserved
        assert "limit=10" in experiments[0].task_overrides.get("mmlu", [])

    def test_task_overrides_passed_to_experiment(self):
        """Non-priority task overrides are passed through to experiment."""
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_builder import ExperimentPlanBuilder
        from olmo_eval.cli.beaker.model_grouper import ModelGrouper
        from olmo_eval.launch import BeakerModelSpec

        config = LaunchConfig(
            name="test",
            model_configs=[BeakerModelSpec(name_or_path="model1")],
            model_specs=["model1"],
            task_specs=["mmlu"],
            cluster="h100",
            workspace="test",
            budget="test",
            task_overrides={"mmlu": ["limit=100", "num_fewshot=5"]},
        )

        grouper = ModelGrouper(config, eval_config=None)
        builder = ExperimentPlanBuilder(
            config, grouper, tasks_by_priority={"normal": ["mmlu"]}, agent_task_specs=set()
        )

        experiments, _ = builder.build()

        # Both overrides should be in task_overrides
        assert experiments[0].task_overrides == {"mmlu": ["limit=100", "num_fewshot=5"]}

    def test_task_overrides_in_command(self, mock_tasks):
        """Task overrides are included in the generated command."""
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler
        from olmo_eval.launch import BeakerModelSpec

        model = BeakerModelSpec(name_or_path="model1")
        exp = ExperimentPlan(
            name="test-exp",
            model_cfgs=[model],
            model_specs=["model1"],
            priority="normal",
            tasks=["mmlu", "gsm8k"],
            original_task_specs=["mmlu", "gsm8k"],
            total_expanded_tasks=2,
            model_gpu_counts=[1],
            num_gpus=1,
            task_overrides={"mmlu": ["limit=10"], "gsm8k": ["limit=20", "num_fewshot=3"]},
        )

        config = LaunchConfig(
            name="test",
            model_configs=[model],
            model_specs=["model1"],
            task_specs=["mmlu", "gsm8k"],
            cluster="h100",
            workspace="test",
            budget="test",
        )

        assembler = JobConfigAssembler(
            config=config,
            eval_config=None,
            effective_image="test-image",
            effective_groups=[],
            beaker_username="test-user",
            common_secrets=[],
            store_secrets=[],
            task_secrets=[],
            inject_aws_credentials=False,
            inject_gcs_credentials=False,
        )

        job_config = assembler.assemble(exp)

        # Command should include task overrides
        cmd = job_config.command
        # Find position of -t mmlu and check -o limit=10 follows
        mmlu_idx = cmd.index("mmlu")
        assert cmd[mmlu_idx + 1] == "-o"
        assert cmd[mmlu_idx + 2] == "limit=10"

        # Find position of -t gsm8k and check both overrides follow
        gsm8k_idx = cmd.index("gsm8k")
        assert cmd[gsm8k_idx + 1] == "-o"
        assert cmd[gsm8k_idx + 2] == "limit=20"
        assert cmd[gsm8k_idx + 3] == "-o"
        assert cmd[gsm8k_idx + 4] == "num_fewshot=3"

    def test_priority_override_not_in_command(self, mock_tasks):
        """Priority override should not appear in the run command."""
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler
        from olmo_eval.launch import BeakerModelSpec

        model = BeakerModelSpec(name_or_path="model1")
        # Simulate already-filtered overrides (priority removed)
        exp = ExperimentPlan(
            name="test-exp",
            model_cfgs=[model],
            model_specs=["model1"],
            priority="urgent",  # Priority was extracted to here
            tasks=["mmlu"],
            original_task_specs=["mmlu"],
            total_expanded_tasks=1,
            model_gpu_counts=[1],
            num_gpus=1,
            task_overrides={"mmlu": ["limit=10"]},  # No priority here
        )

        config = LaunchConfig(
            name="test",
            model_configs=[model],
            model_specs=["model1"],
            task_specs=["mmlu"],
            cluster="h100",
            workspace="test",
            budget="test",
        )

        assembler = JobConfigAssembler(
            config=config,
            eval_config=None,
            effective_image="test-image",
            effective_groups=[],
            beaker_username="test-user",
            common_secrets=[],
            store_secrets=[],
            task_secrets=[],
            inject_aws_credentials=False,
            inject_gcs_credentials=False,
        )

        job_config = assembler.assemble(exp)

        # Command should NOT include priority=urgent as a task override
        cmd = job_config.command
        assert "priority=urgent" not in cmd
        # But experiment priority should be urgent (used for Beaker job)
        assert job_config.priority == "urgent"

    def test_task_packages_from_cli_overrides(self, mock_tasks):
        """Test that task_packages are extracted from CLI dependency overrides."""
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler
        from olmo_eval.launch import BeakerModelSpec

        model = BeakerModelSpec(name_or_path="model1")
        exp = ExperimentPlan(
            name="test-exp",
            model_cfgs=[model],
            model_specs=["model1"],
            priority="normal",
            tasks=["mmlu"],
            original_task_specs=["mmlu"],
            total_expanded_tasks=1,
            model_gpu_counts=[1],
            num_gpus=1,
            task_overrides={"mmlu": ['dependencies=["ai2-olmo-eval", "special-pkg==1.0"]']},
        )

        config = LaunchConfig(
            name="test",
            model_configs=[model],
            model_specs=["model1"],
            task_specs=["mmlu"],
            cluster="h100",
            workspace="test",
            budget="test",
        )

        assembler = JobConfigAssembler(
            config=config,
            eval_config=None,
            effective_image="test-image",
            effective_groups=[],
            beaker_username="test-user",
            common_secrets=[],
            store_secrets=[],
            task_secrets=[],
            inject_aws_credentials=False,
            inject_gcs_credentials=False,
        )

        job_config = assembler.assemble(exp)

        # Dependencies from CLI overrides should be in task_packages
        assert job_config.task_packages == ["ai2-olmo-eval", "special-pkg==1.0"]

    def test_task_packages_merged_from_config_and_cli(self, mock_tasks):
        """Test that task_packages from config and CLI are merged."""
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler
        from olmo_eval.launch import BeakerModelSpec

        model = BeakerModelSpec(name_or_path="model1")
        # task_with_deps has dependencies=["base-pkg"] registered
        exp = ExperimentPlan(
            name="test-exp",
            model_cfgs=[model],
            model_specs=["model1"],
            priority="normal",
            tasks=["task_with_deps"],
            original_task_specs=["task_with_deps"],
            total_expanded_tasks=1,
            model_gpu_counts=[1],
            num_gpus=1,
            task_overrides={"task_with_deps": ['dependencies=["cli-pkg==2.0"]']},
        )

        config = LaunchConfig(
            name="test",
            model_configs=[model],
            model_specs=["model1"],
            task_specs=["task_with_deps"],
            cluster="h100",
            workspace="test",
            budget="test",
        )

        assembler = JobConfigAssembler(
            config=config,
            eval_config=None,
            effective_image="test-image",
            effective_groups=[],
            beaker_username="test-user",
            common_secrets=[],
            store_secrets=[],
            task_secrets=[],
            inject_aws_credentials=False,
            inject_gcs_credentials=False,
        )

        job_config = assembler.assemble(exp)

        # Both registered deps (base-pkg) and CLI deps (cli-pkg==2.0) should be present
        assert "base-pkg" in job_config.task_packages
        assert "cli-pkg==2.0" in job_config.task_packages
