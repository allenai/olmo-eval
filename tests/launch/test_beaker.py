"""Tests for olmo_eval.launch.beaker module."""

import pytest

from olmo_eval.launch.beaker import (
    BeakerEnvSecret,
    BeakerJobConfig,
    BeakerWekaBucket,
    _parse_timeout,
    parse_task_with_priority,
    resolve_clusters,
)


class TestResolveClustors:
    """Tests for cluster resolution."""

    def test_resolve_h100_alias(self):
        """Test resolving h100 alias."""
        clusters = resolve_clusters("h100")
        assert "ai2/augusta" in clusters
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
        assert "ai2/augusta" in clusters
        assert "ai2/jupiter" in clusters
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
        """Test creating minimal config."""
        config = BeakerJobConfig(
            name="test-job",
            command=["echo", "hello"],
        )
        assert config.name == "test-job"
        assert config.command == ["echo", "hello"]
        assert config.num_gpus == 1
        assert config.cluster == "h100"
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
        """Test default Weka buckets are set."""
        config = BeakerJobConfig(name="test", command=["echo"])
        assert len(config.weka_buckets) == 2
        bucket_names = [b.bucket for b in config.weka_buckets]
        assert "oe-eval-default" in bucket_names
        assert "oe-data-default" in bucket_names

    def test_default_secrets(self):
        """Test default secrets are set."""
        config = BeakerJobConfig(name="test", command=["echo"])
        assert len(config.env_secrets) == 2
        secret_names = [s.name for s in config.env_secrets]
        assert "HF_TOKEN" in secret_names
        assert "WANDB_API_KEY" in secret_names


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
        task, priority = parse_task_with_priority("mmlu::olmes@high")
        assert task == "mmlu::olmes"
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
