"""Hardware/shape gating and explicit-override contracts, without a CUDA runtime."""

import asyncio
import copy
import json
import queue
from types import SimpleNamespace

import pytest

from olmo_eval.common.types import SamplingParams
from olmo_eval.inference.providers.sglang_profiles import (
    MODEL_SHAPE,
    prepare_profile,
    resolve_profile,
)
from olmo_eval.runners.asynq.batching import BatchConfig, get_strategy


def model():
    return dict(
        copy.deepcopy(MODEL_SHAPE),
        dtype="bfloat16",
        architectures=["Olmo3MoeForCausalLM"],
        max_position_embeddings=65536,
    )


@pytest.mark.parametrize(
    "gpu,total,cache,expected",
    [
        ("NVIDIA B300", 288000, None, (512, 17825792, 3072, False)),
        ("NVIDIA B300", 288000, True, (512, 17825792, 1024, True)),
        ("NVIDIA H100 80GB HBM3", 81559, None, (512, 6684672, 512, True)),
        ("NVIDIA H100 80GB HBM3", 81559, False, (320, 4456448, 1920, False)),
    ],
)
def test_selected_pools(gpu, total, cache, expected):
    options = {"context_length": 8192}
    if cache is not None:
        options["disable_radix_cache"] = cache
    report = resolve_profile("auto", model(), options, [(gpu, total, total)])
    effective = report["defaults"] | options
    assert report["selected"] != "generic"
    assert (
        tuple(
            effective[k]
            for k in (
                "max_running_requests",
                "max_total_tokens",
                "max_mamba_cache_size",
                "disable_radix_cache",
            )
        )
        == expected
    )
    assert effective["cuda_graph_max_bs_decode"] == expected[0]
    assert effective["cuda_graph_backend_prefill"] == "disabled"


@pytest.mark.parametrize(
    "change",
    [
        {"hidden_size": 2048},
        {"dtype": "float16"},
        {"architectures": ["Olmo3MoeVLMForConditionalGeneration"]},
        {"quantization_config": {"method": "fp8"}},
        {"linear_value_head_dim": 128},
    ],
)
def test_other_models_fall_back(change):
    report = resolve_profile(
        "auto", model() | change, {"context_length": 8192}, [("NVIDIA H100", 81559, 81559)]
    )
    assert report["selected"] == "generic" and not report["defaults"]


@pytest.mark.parametrize(
    "options,devices",
    [
        ({}, [("NVIDIA H100", 81559, 81559)]),
        ({"context_length": 65536}, [("NVIDIA H100", 81559, 81559)]),
        ({"context_length": 8192, "tp_size": 2}, [("NVIDIA H100", 81559, 81559)]),
        ({"context_length": 8192, "dtype": "float32"}, [("NVIDIA H100", 81559, 81559)]),
        ({"context_length": 8192, "enable_multimodal": True}, [("NVIDIA H100", 81559, 81559)]),
        ({"context_length": 8192}, [("NVIDIA A100", 81559, 81559)]),
        ({"context_length": 8192}, [("NVIDIA H100", 81559, 60000)]),
        ({"context_length": 8192}, [("NVIDIA H100 MIG", 40000, 40000)]),
    ],
)
def test_unqualified_context_topology_or_hardware_falls_back(options, devices):
    assert resolve_profile("auto", model(), options, devices)["selected"] == "generic"


def test_explicit_overrides_and_generic():
    options = {
        "context_length": 34816,
        "max_running_requests": 128,
        "max_total_tokens": 12345,
        "mem_fraction_static": 0.8,
        "cuda_graph_backend_decode": "disabled",
    }
    report = resolve_profile("auto", model(), options, [("NVIDIA H100", 81559, 81559)])
    assert not set(options) & set(report["defaults"])
    assert report["defaults"]["max_mamba_cache_size"] == 128
    assert resolve_profile("generic", model(), options, [])["defaults"] == {}
    with pytest.raises(ValueError, match="profile"):
        resolve_profile("bogus", model(), options, [])


def test_hardware_is_selected_after_visible_device_mapping(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(model()))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-assigned,GPU-other")
    commands = []

    def query(command, **kwargs):
        commands.append(command)
        return "NVIDIA H100 80GB HBM3, 81559, 81000\n"

    monkeypatch.setattr(
        "olmo_eval.inference.providers.sglang_profiles.subprocess.check_output", query
    )
    report = prepare_profile(
        "auto", {"model_path": str(tmp_path), "context_length": 8192}, "olmo_sglang:register"
    )
    assert report["selected"] == "olmo35-h100-v1"
    assert "--id=GPU-assigned" in commands[0]


def test_generic_does_not_probe_hardware(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Hardware must not be queried")

    monkeypatch.setattr(
        "olmo_eval.inference.providers.sglang_profiles.subprocess.check_output", fail
    )
    assert (
        prepare_profile("generic", {"model_path": "unused"}, "olmo_sglang:register")["selected"]
        == "generic"
    )


def test_auto_batching_counts_samples_without_losing_or_reordering_requests():
    q = queue.Queue()
    items = [
        SimpleNamespace(sampling_params=SamplingParams(num_samples=n), idx=i)
        for i, n in enumerate([16, 16, 1, 16, 64, 1])
    ]
    for item in items:
        q.put(item)
    q.put(None)
    strategy = get_strategy(BatchConfig(chunk_size=512), max_batch_sequences=32)

    async def collect():
        batches = []
        while True:
            batch, done = await strategy.collect_batch(q)
            batches.append([x.idx for x in batch])
            if done:
                return batches

    assert asyncio.run(collect()) == [[0, 1], [2, 3], [4], [5]]


def test_explicit_batch_size_and_serialized_auto():
    assert BatchConfig.from_dict(BatchConfig().to_dict()).chunk_size == "auto"
    assert BatchConfig().resolved_chunk_size == 64
    assert BatchConfig.from_dict({"chunk_size": 64}).chunk_size == 64
    q = queue.Queue()
    for i in range(3):
        q.put(SimpleNamespace(sampling_params=SamplingParams(num_samples=16), idx=i))
    strategy = get_strategy(BatchConfig(chunk_size=2))
    batch, done = asyncio.run(strategy.collect_batch(q))
    assert [x.idx for x in batch] == [0, 1] and not done
