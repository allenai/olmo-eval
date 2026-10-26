"""Resolved worker settings must reach metrics, manifests and model identity."""

import json
from dataclasses import replace
from queue import Queue
from unittest.mock import Mock

import pytest

from olmo_eval.common.types import compute_model_hash
from olmo_eval.harness.config import HarnessConfig, ProviderConfig
from olmo_eval.inference.metrics.core.config import MetricsConfig
from olmo_eval.runners.asynq.batching import BatchConfig
from olmo_eval.runners.asynq.monitoring import wait_for_init_times
from olmo_eval.runners.asynq.runner import AsyncEvalRunner
from olmo_eval.runners.asynq.runtime import apply_worker_configs, resolved_worker_config
from olmo_eval.runners.asynq.types import WorkerReady


def requested(kind="sglang"):
    return HarnessConfig(
        name="test", provider=ProviderConfig(kind=kind, model="model"), batching=BatchConfig()
    )


def resolved(profile="olmo35-h100-v1", cache=False, free=81000, commit="abc"):
    return resolved_worker_config(
        requested(),
        BatchConfig(chunk_size=512),
        profile={
            "selected": profile,
            "effective_options": {"attention_backend": "triton", "disable_radix_cache": not cache},
            "devices": [["H100", 81559, free]],
        },
        packages={"olmo-sglang": f"0.1.0@{commit}"},
    )


def test_profile_cache_and_package_revision_change_identity_but_free_memory_does_not():
    def identity(config):
        return compute_model_hash(config.provider.to_dict())

    baseline = identity(resolved())
    assert baseline != identity(requested())
    assert baseline != identity(resolved(profile="generic"))
    assert baseline != identity(resolved(cache=True))
    assert baseline != identity(resolved(commit="def"))
    assert baseline == identity(resolved(free=80500))


def test_runtime_config_roundtrip_does_not_forward_metadata_to_engine(monkeypatch):
    provider = resolved().provider
    restored = ProviderConfig.from_dict(provider.to_dict())
    assert restored == provider
    factory = Mock()
    monkeypatch.setattr("olmo_eval.inference.create_provider", factory)
    restored.create_provider()
    assert "resolved_runtime" not in factory.call_args.kwargs


@pytest.mark.parametrize(
    "other",
    [
        resolved(profile="generic"),
        resolved(cache=True),
        replace(resolved(), batching=BatchConfig(chunk_size=64)),
    ],
)
def test_mixed_replica_settings_are_rejected_before_dispatch(other):
    with pytest.raises(RuntimeError, match="different runtime"):
        apply_worker_configs(requested(), {"0": resolved().to_dict(), "1": other.to_dict()})


def test_worker_handshake_carries_runtime_into_final_manifest(tmp_path):
    queue = Queue()
    queue.put(WorkerReady("0", 1.25, resolved().to_dict()))
    reports = {}
    assert wait_for_init_times(queue, 1, worker_configs=reports) == {"0": 1.25}
    config = apply_worker_configs(requested(), reports)
    runner = AsyncEvalRunner(harness_config=config, output_dir=str(tmp_path))
    runner._started_at = "2026-10-26T00:00:00Z"
    results = runner._aggregate_results({}, [])
    expected = compute_model_hash(config.provider.to_dict())
    runner._write_final_manifest(results, "test", expected, 2.0, {"0": 1.25})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["model"]["model_hash"] == expected
    assert manifest["model"]["provider_config"]["resolved_runtime"] == dict(
        config.provider.resolved_runtime
    )
    assert manifest["run"]["harness_config"]["batching"]["chunk_size"] == 512


def test_other_provider_hash_stays_unchanged_and_effective_batch_is_saved():
    config = requested("vllm")
    worker = resolved_worker_config(config, BatchConfig(chunk_size=64))
    parent = apply_worker_configs(config, {"0": worker.to_dict()})
    assert parent.to_dict()["batching"]["chunk_size"] == 64
    assert parent.provider.to_dict() == config.provider.to_dict()


def test_replayed_runtime_is_refreshed():
    config = resolved_worker_config(
        resolved(),
        BatchConfig(chunk_size=64),
        profile={"selected": "generic", "effective_options": {}},
        packages={},
    )
    assert config.provider.resolved_runtime["profile"] == "generic"
    assert config.provider.resolved_runtime["packages"] == {}


def test_resolved_batch_roundtrip_retains_expanded_sample_budget():
    config = replace(resolved(), batching=BatchConfig(chunk_size=512, max_batch_sequences=512))
    restored = HarnessConfig.from_dict(config.to_dict())
    assert restored.batching == config.batching
    assert restored.batching.max_batch_sequences == 512


def test_worker_metrics_and_parent_results_share_resolved_hash():
    config = replace(requested(), metrics=MetricsConfig(model_hash="provisional"))
    worker = resolved_worker_config(
        config,
        BatchConfig(chunk_size=512),
        profile={
            "selected": "olmo35-h100-v1",
            "effective_options": {"attention_backend": "triton"},
        },
    )
    parent = apply_worker_configs(config, {"0": worker.to_dict()})
    runner = AsyncEvalRunner(harness_config=parent)
    runner._update_metrics_config("test")
    expected = compute_model_hash(runner._aggregate_results({}, [])["model_config"])
    assert worker.metrics.model_hash == expected
    assert runner.harness_config.metrics.model_hash == expected
    assert expected != "provisional"
