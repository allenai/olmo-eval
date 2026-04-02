"""Tests for provider registry and GPU planning."""

from __future__ import annotations

import pytest

from olmo_eval.common.types import ProviderKind
from olmo_eval.inference.gpu_planner import GPUAllocation, GPUPlan, GPUPlanner, validate_gpu_plan
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.inference.registry import ProviderRegistry, ReplicaSet


class TestProviderConfigRoundtrip:
    """Test that provider kind and fields survive serialization/deserialization."""

    def test_kind_survives_roundtrip(self):
        """Provider kind is preserved through to_dict() -> from_dict()."""
        config = ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            max_concurrency=10,
        )

        serialized = config.to_dict()
        restored = ProviderConfig.from_dict(serialized)

        assert restored.kind == config.kind
        assert restored.model == config.model
        assert restored.base_url == config.base_url
        assert restored.max_concurrency == config.max_concurrency

    def test_vllm_server_kind_survives_roundtrip(self):
        """vllm_server kind is preserved through serialization."""
        config = ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="meta-llama/Llama-3.1-8B-Instruct",
            base_url="http://localhost:8000/v1",
            num_instances=1,
            kwargs={"tensor_parallel_size": 2},
        )

        serialized = config.to_dict()
        restored = ProviderConfig.from_dict(serialized)

        assert restored.kind == ProviderKind.VLLM_SERVER
        assert restored.model == config.model
        assert restored.base_url == config.base_url
        assert restored.kwargs.get("tensor_parallel_size") == 2

    def test_num_instances_survives_roundtrip(self):
        """num_instances field is preserved through serialization."""
        config = ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="test-model",
            num_instances=4,
        )

        serialized = config.to_dict()
        restored = ProviderConfig.from_dict(serialized)

        assert restored.num_instances == 4

    def test_num_instances_default_not_serialized(self):
        """num_instances=1 is not included in serialization (default value)."""
        config = ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="test-model",
            num_instances=1,  # default value
        )

        serialized = config.to_dict()

        assert "num_instances" not in serialized

    def test_kwargs_survive_roundtrip(self):
        """kwargs are preserved through serialization."""
        config = ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="test-model",
            base_url="http://localhost:8000/v1",
            kwargs={
                "tensor_parallel_size": 4,
                "enable_prefix_caching": True,
                "custom_field": "value",
            },
        )

        serialized = config.to_dict()
        restored = ProviderConfig.from_dict(serialized)

        assert restored.kwargs.get("tensor_parallel_size") == 4
        assert restored.kwargs.get("enable_prefix_caching") is True
        assert restored.kwargs.get("custom_field") == "value"


class TestReplicaSetRoundRobin:
    """Test ReplicaSet round-robin behavior."""

    def test_single_replica_returns_same_provider(self):
        """Single replica always returns the same provider."""
        config = ProviderConfig(
            kind=ProviderKind.MOCK,
            model="mock-model",
        )
        replica_set = ReplicaSet("test", [config])

        provider1 = replica_set.next()
        provider2 = replica_set.next()
        provider3 = replica_set.next()

        # All should be the same instance (lazy creation)
        assert provider1 is provider2
        assert provider2 is provider3

    def test_multiple_replicas_round_robin(self):
        """Multiple replicas are accessed in round-robin order."""
        configs = [
            ProviderConfig(kind=ProviderKind.MOCK, model=f"mock-model-{i}") for i in range(3)
        ]
        replica_set = ReplicaSet("test", configs)

        # Get providers - should cycle through configs
        providers = [replica_set.next() for _ in range(6)]

        # Check model names cycle: 0, 1, 2, 0, 1, 2
        model_names = [p.model_name for p in providers]
        assert model_names == [
            "mock-model-0",
            "mock-model-1",
            "mock-model-2",
            "mock-model-0",
            "mock-model-1",
            "mock-model-2",
        ]

    def test_empty_configs_raises(self):
        """ReplicaSet requires at least one config."""
        with pytest.raises(ValueError, match="requires at least one config"):
            ReplicaSet("test", [])


class TestProviderRegistry:
    """Test ProviderRegistry creation and lookup."""

    def test_from_serialized_preserves_kind(self):
        """Registry creation preserves provider kind from serialized configs."""
        serialized = {
            "judge": [
                {
                    "kind": ProviderKind.LITELLM,
                    "model": "gpt-4",
                    "base_url": "https://api.openai.com/v1",
                }
            ]
        }

        registry = ProviderRegistry.from_serialized(serialized)

        # Get the underlying config to verify kind
        replica_set = registry.get_replica_set("judge")
        config = replica_set.get_config(0)
        assert config.kind == ProviderKind.LITELLM

    def test_multi_replica_round_robin(self):
        """Registry returns providers in round-robin for multi-replica sets."""
        serialized = {
            "judge": [
                {"kind": ProviderKind.MOCK, "model": "mock-0"},
                {"kind": ProviderKind.MOCK, "model": "mock-1"},
            ]
        }

        registry = ProviderRegistry.from_serialized(serialized)

        # Get providers - should round-robin
        p1 = registry.get("judge")
        p2 = registry.get("judge")
        p3 = registry.get("judge")

        assert p1.model_name == "mock-0"
        assert p2.model_name == "mock-1"
        assert p3.model_name == "mock-0"  # wraps around

    def test_to_serialized_roundtrip(self):
        """to_serialized() output can be used with from_serialized()."""
        original_configs = {
            "judge": [
                ProviderConfig(kind=ProviderKind.MOCK, model="mock-judge"),
            ],
            "embedder": [
                ProviderConfig(kind=ProviderKind.MOCK, model="mock-embed-0"),
                ProviderConfig(kind=ProviderKind.MOCK, model="mock-embed-1"),
            ],
        }

        registry1 = ProviderRegistry.from_resolved_configs(original_configs)
        serialized = registry1.to_serialized()

        registry2 = ProviderRegistry.from_serialized(serialized)

        assert registry2.names == registry1.names
        assert registry2.models == registry1.models


class TestGPUPlanner:
    """Test GPU allocation planning."""

    def test_main_workers_only(self):
        """Plan with only main workers allocates correctly."""
        planner = GPUPlanner(
            total_gpus=4,
            num_main_workers=2,
            main_tensor_parallel=2,
        )

        plan = planner.plan()

        assert len(plan.main_workers) == 2
        assert plan.main_workers[0].gpu_ids == [0, 1]
        assert plan.main_workers[1].gpu_ids == [2, 3]
        assert plan.auxiliary == {}
        assert plan.total_gpus_used == 4

    def test_main_and_auxiliary(self):
        """Plan with main workers and auxiliary providers."""
        aux_config = ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="judge-model",
            num_instances=2,
            kwargs={"tensor_parallel_size": 2},
        )

        planner = GPUPlanner(
            total_gpus=8,
            num_main_workers=2,
            main_tensor_parallel=1,
            auxiliary_configs={"judge": aux_config},
        )

        plan = planner.plan()

        # Main workers: 2 × 1 = 2 GPUs (0, 1)
        assert len(plan.main_workers) == 2
        assert plan.main_workers[0].gpu_ids == [0]
        assert plan.main_workers[1].gpu_ids == [1]

        # Auxiliary: 2 instances × 2 TP = 4 GPUs (2, 3, 4, 5)
        assert "judge" in plan.auxiliary
        assert plan.auxiliary["judge"].gpu_ids == [2, 3, 4, 5]

        assert plan.total_gpus_used == 6

    def test_insufficient_gpus_raises(self):
        """Plan raises error when not enough GPUs."""
        planner = GPUPlanner(
            total_gpus=4,
            num_main_workers=2,
            main_tensor_parallel=2,  # 4 GPUs for main
            auxiliary_configs={
                "judge": ProviderConfig(
                    kind=ProviderKind.VLLM_SERVER,
                    model="judge",
                    num_instances=1,
                    kwargs={"tensor_parallel_size": 2},  # 2 more GPUs needed
                )
            },
        )

        with pytest.raises(RuntimeError, match="Not enough GPUs"):
            planner.plan()

    def test_no_overlap_between_main_and_auxiliary(self):
        """GPU assignments don't overlap between main and auxiliary."""
        aux_config = ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="judge",
            num_instances=2,
            kwargs={"tensor_parallel_size": 1},
        )

        planner = GPUPlanner(
            total_gpus=8,
            num_main_workers=4,
            main_tensor_parallel=1,
            auxiliary_configs={"judge": aux_config},
        )

        plan = planner.plan()

        main_gpus = set()
        for alloc in plan.main_workers:
            for gpu in alloc.gpu_ids:
                assert gpu not in main_gpus, f"GPU {gpu} duplicated in main"
                main_gpus.add(gpu)

        aux_gpus = set()
        for alloc in plan.auxiliary.values():
            for gpu in alloc.gpu_ids:
                assert gpu not in aux_gpus, f"GPU {gpu} duplicated in aux"
                assert gpu not in main_gpus, f"GPU {gpu} in both main and aux"
                aux_gpus.add(gpu)

    def test_external_server_no_gpus(self):
        """External servers (with base_url) don't consume GPUs."""
        external_config = ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="external-model",
            base_url="http://external:8000/v1",
            num_instances=4,  # Should be ignored
        )

        planner = GPUPlanner(
            total_gpus=2,
            num_main_workers=2,
            main_tensor_parallel=1,
            auxiliary_configs={"external": external_config},
        )

        plan = planner.plan()

        # External provider should not appear in auxiliary allocations
        assert "external" not in plan.auxiliary
        assert plan.total_gpus_used == 2  # Only main workers


class TestProviderConfigRequiresLocalGpu:
    """Test requires_local_gpu property."""

    def test_vllm_server_without_base_url_requires_local_gpu(self):
        """vllm_server without base_url requires local GPU."""
        config = ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="test-model",
        )
        assert config.requires_gpu is True
        assert config.requires_local_gpu is True

    def test_vllm_server_with_base_url_no_local_gpu(self):
        """vllm_server with base_url doesn't require local GPU."""
        config = ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="test-model",
            base_url="http://external:8000/v1",
        )
        assert config.requires_gpu is True
        assert config.requires_local_gpu is False

    def test_litellm_no_local_gpu(self):
        """litellm provider doesn't require local GPU."""
        config = ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="gpt-4",
        )
        assert config.requires_gpu is False
        assert config.requires_local_gpu is False

    def test_mock_no_local_gpu(self):
        """mock provider doesn't require local GPU."""
        config = ProviderConfig(
            kind=ProviderKind.MOCK,
            model="mock-model",
        )
        assert config.requires_gpu is False
        assert config.requires_local_gpu is False

    def test_vllm_requires_local_gpu(self):
        """vllm provider requires local GPU."""
        config = ProviderConfig(
            kind=ProviderKind.VLLM,
            model="test-model",
        )
        assert config.requires_gpu is True
        assert config.requires_local_gpu is True

    def test_hf_requires_local_gpu(self):
        """hf provider requires local GPU."""
        config = ProviderConfig(
            kind=ProviderKind.HF,
            model="test-model",
        )
        assert config.requires_gpu is True
        assert config.requires_local_gpu is True


class TestGPUPlannerFromHarnessConfig:
    """Test GPUPlanner.from_harness_config behavior."""

    def test_uses_provider_num_instances(self):
        """from_harness_config uses provider.num_instances for worker count."""
        provider = ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="test-model",
            num_instances=4,
            kwargs={"tensor_parallel_size": 2},
        )

        class MockHarnessConfig:
            auxiliary_providers = {}

        mock_config = MockHarnessConfig()
        mock_config.provider = provider

        planner = GPUPlanner.from_harness_config(mock_config, total_gpus=8)

        assert planner.num_main_workers == 4
        assert planner.main_tensor_parallel == 2

        # Plan should allocate 4 × 2 = 8 GPUs
        plan = planner.plan()
        assert plan.total_gpus_used == 8

    def test_api_backed_provider_no_main_gpus(self):
        """API-backed providers don't consume main GPUs."""
        provider = ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="gpt-4",
            num_instances=4,  # Ignored for API-backed providers
        )

        class MockHarnessConfig:
            auxiliary_providers = {}

        mock_config = MockHarnessConfig()
        mock_config.provider = provider

        planner = GPUPlanner.from_harness_config(mock_config, total_gpus=8)

        # API-backed provider should not consume GPUs
        assert planner.num_main_workers == 0

        plan = planner.plan()
        assert plan.total_gpus_used == 0

    def test_external_vllm_server_no_main_gpus(self):
        """vllm_server with base_url doesn't consume main GPUs."""
        provider = ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="remote-model",
            base_url="http://external:8000/v1",
            num_instances=4,  # Ignored for external servers
        )

        class MockHarnessConfig:
            auxiliary_providers = {}

        mock_config = MockHarnessConfig()
        mock_config.provider = provider

        planner = GPUPlanner.from_harness_config(mock_config, total_gpus=8)

        # External server should not consume GPUs
        assert planner.num_main_workers == 0

        plan = planner.plan()
        assert plan.total_gpus_used == 0

    def test_main_and_auxiliary_gpu_allocation(self):
        """Both main and auxiliary providers are allocated GPUs correctly."""
        main_provider = ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="main-model",
            num_instances=2,
            kwargs={"tensor_parallel_size": 2},
        )

        aux_provider = ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="aux-model",
            num_instances=1,
            kwargs={"tensor_parallel_size": 1},
        )

        class MockHarnessConfig:
            auxiliary_providers = {"judge": aux_provider}

        mock_config = MockHarnessConfig()
        mock_config.provider = main_provider

        planner = GPUPlanner.from_harness_config(mock_config, total_gpus=8)

        # Main: 2 instances × 2 TP = 4 GPUs
        # Aux: 1 instance × 1 TP = 1 GPU
        # Total: 5 GPUs
        plan = planner.plan()
        assert plan.total_gpus_used == 5


class TestValidateGPUPlan:
    """Test GPU plan validation."""

    def test_valid_plan_passes(self):
        """Valid plan passes validation."""
        plan = GPUPlan(
            main_workers=[
                GPUAllocation("main-0", [0, 1]),
                GPUAllocation("main-1", [2, 3]),
            ],
            auxiliary={"judge": GPUAllocation("judge", [4, 5])},
        )

        # Should not raise
        validate_gpu_plan(plan, total_gpus=6)

    def test_overlapping_gpus_fails(self):
        """Overlapping GPU assignments fail validation."""
        plan = GPUPlan(
            main_workers=[
                GPUAllocation("main-0", [0, 1]),
            ],
            auxiliary={"judge": GPUAllocation("judge", [1, 2])},  # Overlaps on GPU 1
        )

        with pytest.raises(ValueError, match="GPU 1"):
            validate_gpu_plan(plan, total_gpus=4)

    def test_out_of_range_gpu_fails(self):
        """GPU IDs beyond total_gpus fail validation."""
        plan = GPUPlan(
            main_workers=[
                GPUAllocation("main-0", [0, 10]),  # GPU 10 out of range
            ],
            auxiliary={},
        )

        with pytest.raises(ValueError, match="out of range"):
            validate_gpu_plan(plan, total_gpus=4)
