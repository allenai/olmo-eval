"""Hardware/shape gating and profile-conflict contracts, without a CUDA runtime."""

import asyncio
import copy
import json
import queue
from types import SimpleNamespace

import pytest

from olmo_eval.common.types import SamplingParams
from olmo_eval.inference.model_profiles import MODEL_SHAPES
from olmo_eval.inference.providers.sglang_profiles import (
    ENGINE_PROFILES,
    EngineProfile,
    prepare_profile,
    resolve_profile,
)
from olmo_eval.runners.asynq.batching import BatchConfig, get_strategy


def model():
    return dict(
        copy.deepcopy(MODEL_SHAPES["olmo3.5-12.5b"]),
        dtype="bfloat16",
        architectures=["Olmo3MoeForCausalLM"],
        max_position_embeddings=65536,
    )


@pytest.mark.parametrize(
    "gpu,total,cache,expected",
    [
        ("NVIDIA B300", 288000, None, (512, 17825792, 3072, False)),
        ("NVIDIA B300", 288000, False, (512, 17825792, 3072, False)),
        ("NVIDIA H100 80GB HBM3", 81559, None, (512, 6684672, 512, True)),
        ("NVIDIA A100-SXM4-80GB", 81251, None, (512, 6684672, 512, True)),
        ("NVIDIA H100 80GB HBM3", 81559, True, (512, 6684672, 512, True)),
    ],
)
@pytest.mark.parametrize("selector", ["auto", "olmo3.5-12.5b"])
def test_selected_pools(gpu, total, cache, expected, selector):
    options = {"context_length": 8192}
    if cache is not None:
        options["disable_radix_cache"] = cache
    report = resolve_profile(selector, model(), options, [(gpu, total, total)])
    effective = report["defaults"] | options
    assert report["selected"].startswith("olmo3.5-12.5b-sglang-")
    assert report["model_profile"] == "olmo3.5-12.5b"
    assert report["requested"] == selector
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
    with pytest.raises(ValueError, match="not qualified"):
        resolve_profile(
            "olmo3.5-12.5b",
            model() | change,
            {"context_length": 8192},
            [("NVIDIA H100", 81559, 81559)],
        )


@pytest.mark.parametrize(
    "options,devices",
    [
        ({}, [("NVIDIA H100", 81559, 81559)]),
        ({"context_length": 65536}, [("NVIDIA H100", 81559, 81559)]),
        ({"context_length": 8192, "tp_size": 2}, [("NVIDIA H100", 81559, 81559)]),
        ({"context_length": 8192, "dtype": "float32"}, [("NVIDIA H100", 81559, 81559)]),
        ({"context_length": 8192, "enable_multimodal": True}, [("NVIDIA H100", 81559, 81559)]),
        ({"context_length": 8192}, [("NVIDIA A100", 40960, 40960)]),
        ({"context_length": 8192}, [("NVIDIA H100", 81559, 60000)]),
        ({"context_length": 8192}, [("NVIDIA H100 MIG", 40000, 40000)]),
    ],
)
def test_unqualified_context_topology_or_hardware_falls_back(options, devices):
    assert resolve_profile("auto", model(), options, devices)["selected"] == "generic"
    with pytest.raises(ValueError, match="not qualified"):
        resolve_profile("olmo3.5-12.5b", model(), options, devices)


@pytest.mark.parametrize("selector", ["auto", "olmo3.5-12.5b"])
@pytest.mark.parametrize(
    "option",
    [
        {"max_running_requests": 128},
        {"max_running_requests": 1024},
        {"max_total_tokens": 12345},
        {"mem_fraction_static": 0.8},
        {"disable_radix_cache": False},
        {"cuda_graph_backend_decode": "disabled"},
        {"disable_cuda_graph": True},
        {"disable_decode_cuda_graph": True},
        {"cuda_graph_config": {"decode": {"backend": "disabled"}}},
        {"cuda_graph_config": {"prefill": {"backend": "full"}}},
        {"cuda_graph_config": {"decode": {"max_bs": 128}}},
        {"cuda_graph_bs_decode": [1, 64, 128]},
        {"cuda_graph_config": {"decode": {"bs": [1, 64, 128]}}},
    ],
)
def test_conflicting_options_require_manual_profile(selector, option):
    options = {"context_length": 8192} | option
    with pytest.raises(ValueError, match="Settings conflict.*profile=generic"):
        resolve_profile(selector, model(), options, [("NVIDIA H100", 81559, 81559)])
    assert resolve_profile("generic", model(), options, [])["defaults"] == {}


@pytest.mark.parametrize("selector", ["auto", "olmo3.5-12.5b"])
def test_matching_and_unmanaged_options_are_preserved(selector):
    options = {
        "context_length": 8192,
        "max_running_requests": 512,
        "disable_prefill_cuda_graph": True,
        "random_seed": 42,
        "cuda_graph_config": {"decode": {"backend": "full", "max_bs": 512}},
        "cuda_graph_bs_decode": [1, 128, 512],
    }
    report = resolve_profile(selector, model(), options, [("NVIDIA H100", 81559, 81559)])
    assert not set(options) & set(report["defaults"])
    assert report["defaults"]["max_mamba_cache_size"] == 512
    assert (report["defaults"] | options)["random_seed"] == 42
    with pytest.raises(ValueError, match="Unknown SGLang profile"):
        resolve_profile("bogus", model(), options, [])


@pytest.mark.parametrize("selector", ["auto", "olmo3.5-12.5b"])
def test_hardware_is_selected_after_visible_device_mapping(tmp_path, monkeypatch, selector):
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
        selector, {"model_path": str(tmp_path), "context_length": 8192}, "olmo_sglang:register"
    )
    assert report["selected"] == "olmo3.5-12.5b-sglang-h100-v1"
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


def test_named_profile_requires_registration_without_probing(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Hardware must not be queried")

    monkeypatch.setattr(
        "olmo_eval.inference.providers.sglang_profiles.subprocess.check_output", fail
    )
    options = {"model_path": "unused"}
    assert prepare_profile("auto", options, None)["selected"] == "generic"
    with pytest.raises(ValueError, match="model registration"):
        prepare_profile("olmo3.5-12.5b", options, None)
    with pytest.raises(ValueError, match="Unknown SGLang profile"):
        prepare_profile("olmo3.5-typo", options, None)


@pytest.mark.parametrize("failure", ["config", "hardware"])
def test_named_profile_cannot_silently_fall_back_when_probe_fails(tmp_path, monkeypatch, failure):
    (tmp_path / "config.json").write_text("invalid" if failure == "config" else json.dumps(model()))

    def query(*args, **kwargs):
        raise OSError("GPU telemetry unavailable")

    monkeypatch.setattr(
        "olmo_eval.inference.providers.sglang_profiles.subprocess.check_output", query
    )
    options = {"model_path": str(tmp_path), "context_length": 8192}
    assert prepare_profile("auto", options, "olmo_sglang:register")["selected"] == "generic"
    with pytest.raises(ValueError, match="metadata/hardware unavailable"):
        prepare_profile("olmo3.5-12.5b", options, "olmo_sglang:register")


def test_shared_geometry_requires_engine_registration(monkeypatch):
    # A model family entry alone must not borrow another shape's engine defaults.
    monkeypatch.setitem(MODEL_SHAPES, "another-size", model() | {"hidden_size": 2048})
    options = {"context_length": 8192}
    devices = [("NVIDIA H100", 81559, 81559)]
    assert (
        resolve_profile("auto", model() | {"hidden_size": 2048}, options, devices)["selected"]
        == "generic"
    )
    with pytest.raises(ValueError, match="Unknown SGLang profile"):
        resolve_profile("another-size", model() | {"hidden_size": 2048}, options, devices)


def test_registered_shape_dispatches_its_own_defaults_and_requires_its_hook(tmp_path, monkeypatch):
    shape = model() | {"hidden_size": 2048}
    monkeypatch.setitem(MODEL_SHAPES, "another-size", shape)
    monkeypatch.setitem(
        ENGINE_PROFILES,
        "another-size",
        EngineProfile(
            "another_plugin:register",
            lambda model, options, devices: {
                "selected": "another-size-sglang-test-v1",
                "defaults": {"max_running_requests": 7},
            },
        ),
    )
    (tmp_path / "config.json").write_text(json.dumps(shape))
    monkeypatch.setattr(
        "olmo_eval.inference.providers.sglang_profiles.subprocess.check_output",
        lambda *args, **kwargs: "NVIDIA H100, 81559, 81000",
    )
    options = {"model_path": str(tmp_path)}
    for selector in ("auto", "another-size"):
        report = prepare_profile(selector, options, "another_plugin:register")
        assert report["selected"] == "another-size-sglang-test-v1"
        assert report["defaults"] == {"max_running_requests": 7}
    assert prepare_profile("auto", options, "olmo_sglang:register")["selected"] == "generic"
    with pytest.raises(ValueError, match="model registration"):
        prepare_profile("another-size", options, "olmo_sglang:register")


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
