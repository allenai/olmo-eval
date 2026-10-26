"""Contract tests for the inline SGLang generation provider."""

import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from olmo_eval.common.types import (
    Instance,
    LMRequest,
    ProviderKind,
    RequestType,
    Response,
    SamplingParams,
    ToolSchema,
)
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.inference import create_provider
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.inference.providers.sglang import SGLangProvider
from olmo_eval.runners.asynq.batching.config import BatchConfig, BatchStrategy
from olmo_eval.runners.asynq.workers import inference_worker
from olmo_eval.runners.io.builders import build_predictions
from olmo_eval.runners.processing.generation_counts import count_generations


@pytest.fixture
def setup_engine(monkeypatch):
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "1")
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = "<user>question<assistant><think>"
    tokenizer.encode.return_value = [10, 20]
    tokenizer.decode.side_effect = lambda ids: f"token{ids[0]}"
    engine = Mock(tokenizer_manager=SimpleNamespace(tokenizer=tokenizer))

    def generate(**kwargs):
        return [
            {
                "text": f"response{index}",
                "meta_info": {
                    "output_token_logprobs": [[-0.3, 17, None], [-0.7, 18, None]],
                    "output_top_logprobs": [[[-0.1, 99, None]], [[-0.2, 98, None]]],
                    "completion_tokens": 2,
                    "prompt_tokens": 5,
                    "finish_reason": {"type": "stop", "matched": 18},
                },
            }
            for index, _ in enumerate(kwargs.get("prompt", kwargs.get("input_ids")))
        ]

    engine.generate.side_effect = generate
    constructor = Mock(return_value=engine)
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=constructor))
    return engine, constructor, tokenizer


def test_provider_factory_and_config(setup_engine, monkeypatch):
    engine, constructor, _ = setup_engine
    register = Mock()
    monkeypatch.setitem(sys.modules, "model_plugin", SimpleNamespace(register=register))
    config = ProviderConfig(
        kind=ProviderKind.SGLANG,
        model="model",
        dtype="bfloat16",
        max_model_len=65536,
        max_concurrency=32,
        kwargs={"model_registration": "model_plugin:register", "mem_fraction_static": 0.75},
    )
    assert config.requires_local_gpu
    provider = ProviderConfig.from_dict(config.to_dict()).create_provider()
    assert isinstance(provider, SGLangProvider)
    register.assert_called_once_with()
    constructor.assert_called_once_with(
        model_path="model",
        dtype="bfloat16",
        trust_remote_code=False,
        context_length=65536,
        max_running_requests=32,
        mem_fraction_static=0.75,
    )
    provider.close()
    provider.close()
    provider.shutdown()
    engine.shutdown.assert_called_once_with()
    with pytest.raises(RuntimeError, match="closed"):
        provider.generate([LMRequest(RequestType.COMPLETION, prompt="abc")])
    with pytest.raises(ValueError, match="only supports batched"):
        BatchConfig(strategy=BatchStrategy.STREAMING).validate_for_provider("sglang")


def test_chat_sampling_and_sampled_token_logprobs(setup_engine):
    engine, _, tokenizer = setup_engine
    provider = create_provider("sglang", "model")
    request = LMRequest(RequestType.CHAT, messages=({"role": "user", "content": "question"},))
    params = SamplingParams(
        max_tokens=32768,
        temperature=0.8,
        top_p=0.95,
        top_k=-1,
        stop_sequences=("<|im_end|>", "<|endoftext|>"),
    )
    output = provider.generate([request], params)[0][0]
    tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "question"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    engine.generate.assert_called_once_with(
        prompt=["<user>question<assistant><think>"],
        sampling_params={
            "max_new_tokens": 32768,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": -1,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "n": 1,
        },
        return_logprob=True,
        top_logprobs_num=1,
    )
    assert output.text == "response0"
    assert output.logprobs is not None
    assert output.logprobs[0]["logprob"] == -0.3  # sampled token, not top-ranked alternative
    assert output.logprobs[0]["token"] == "token17"
    assert output.logprobs[0]["token_id"] == 17
    assert output.logprobs[0]["top_logprobs"] == [{"token": "token99", "logprob": -0.1}]
    assert output.metadata["output_token_ids"] == [17, 18]
    assert output.metadata["sum_logits"] == -1.0
    assert output.metadata["finish_reason"] == "stop"
    assert output.metadata["sglang_finish_reason"] == {"type": "stop", "matched": 18}
    trace = provider.describe_request(request, params)
    assert trace is not None
    assert trace["generation_kwargs"]["top_p"] == 0.95


@pytest.mark.parametrize("reason", [{"type": "length", "length": 2}, "length", None])
def test_finish_reason_reaches_harness_counts_and_predictions(setup_engine, reason):
    engine, _, _ = setup_engine
    provider = SGLangProvider("model")
    raw = engine.generate.side_effect(prompt=["abc"])[0]
    raw["meta_info"]["finish_reason"] = reason
    engine.generate.side_effect = None
    engine.generate.return_value = [raw]
    request = LMRequest(RequestType.COMPLETION, prompt="abc")
    output = provider.generate([request])[0][0]
    response = Response(Instance(question="abc"), request, outputs=[output])
    counts = count_generations([response])
    assert counts is not None
    assert counts["cap_hit"] == int(reason is not None)
    assert counts["finish_reason_unknown"] == int(reason is None)
    assert build_predictions([response])[0]["model_output"][0]["finish_reason"] == (
        "length" if reason is not None else None
    )
    assert output.metadata["sglang_finish_reason"] == reason
    assert raw["meta_info"]["finish_reason"] == reason


def test_bos_policy_multiple_samples_and_async(setup_engine):
    engine, _, tokenizer = setup_engine
    provider = SGLangProvider("model", add_bos_token=False)
    request = LMRequest(RequestType.COMPLETION, prompt="abc")
    outputs = asyncio.run(provider.agenerate([request, request], SamplingParams(num_samples=2)))
    assert [[o.text for o in group] for group in outputs] == [
        ["response0", "response1"],
        ["response2", "response3"],
    ]
    assert engine.generate.call_args.kwargs["input_ids"] == [[10, 20]] * 4
    tokenizer.encode.assert_called_with("abc", add_special_tokens=False)


def test_greedy_uncapped_and_empty(setup_engine):
    engine, _, _ = setup_engine
    provider = SGLangProvider("model")
    assert provider._build_sampling_params(
        SamplingParams(max_tokens=None, do_sample=False, temperature=0.8, top_p=0.95)
    ) == {"max_new_tokens": None, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "stop": [], "n": 1}
    assert provider.generate([]) == []
    engine.generate.assert_not_called()


def test_unsupported_and_incomplete_results_fail_closed(setup_engine):
    engine, _, _ = setup_engine
    provider = SGLangProvider("model")
    request = LMRequest(RequestType.COMPLETION, prompt="abc")
    with pytest.raises(NotImplementedError, match="truncation"):
        provider.generate([request], SamplingParams(truncate_prompt_tokens=5))
    with pytest.raises(NotImplementedError, match="continuation"):
        provider.logprobs([request])
    with pytest.raises(ValueError, match="tokenizer initialization"):
        SGLangProvider("model", skip_tokenizer_init=True)
    engine.generate.return_value = []
    engine.generate.side_effect = None
    with pytest.raises(RuntimeError, match="number of completions"):
        provider.generate([request])


def test_cache_refresh_uses_shared_helper(setup_engine, monkeypatch):
    _, constructor, _ = setup_engine
    refresh = Mock()
    monkeypatch.setattr("olmo_eval.inference.providers.sglang.refresh_hf_cache", refresh)
    provider = ProviderConfig(
        kind="sglang",
        model="model",
        tokenizer="tokenizer",
        revision="pinned",
        force_download=True,
        kwargs={"download_dir": "/cache"},
    ).create_provider()
    assert isinstance(provider, SGLangProvider)
    assert [call.args for call in refresh.call_args_list] == [("model",), ("tokenizer",)]
    assert all(
        call.kwargs == {"revision": "pinned", "cache_dir": "/cache", "force_download": True}
        for call in refresh.call_args_list
    )
    assert "force_download" not in constructor.call_args.kwargs
    assert constructor.call_args.kwargs["tokenizer_path"] == "tokenizer"
    assert constructor.call_args.kwargs["revision"] == "pinned"


def test_tensor_parallel_setting_maps_to_engine(setup_engine):
    _, constructor, _ = setup_engine
    ProviderConfig(
        kind="sglang", model="model", kwargs={"tensor_parallel_size": 2}
    ).create_provider()
    assert constructor.call_args.kwargs["tp_size"] == 2
    assert "tensor_parallel_size" not in constructor.call_args.kwargs
    with pytest.raises(ValueError, match="must agree"):
        SGLangProvider("model", tensor_parallel_size=2, tp_size=4)


def test_worker_maps_allocated_gpus_and_closes_provider(setup_engine, monkeypatch):
    engine, constructor, _ = setup_engine
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("olmo_eval.common.logging.configure_logging", Mock())
    monkeypatch.setattr(
        "olmo_eval.common.logging.configure_worker_logging", Mock(return_value=Mock())
    )
    strategy = SimpleNamespace(run=AsyncMock())
    monkeypatch.setattr(
        "olmo_eval.runners.asynq.batching.get_strategy", Mock(return_value=strategy)
    )

    def make_harness(config):
        return SimpleNamespace(
            provider=config.provider.create_provider(),
            initialize_reporters=Mock(),
            cleanup=AsyncMock(),
        )

    monkeypatch.setattr("olmo_eval.harness.Harness", make_harness)
    config = HarnessConfig(
        name="test",
        provider=ProviderConfig(
            kind="sglang",
            model="model",
            kwargs={"load_format": "safetensors", "model_loader_extra_config": '{"threads": 2}'},
        ),
    )
    results = Mock()
    inference_worker("test", [0, 1], Mock(), results, config.to_dict(), 0)
    assert constructor.call_args.kwargs["tp_size"] == 2
    assert constructor.call_args.kwargs["load_format"] == "safetensors"
    assert constructor.call_args.kwargs["model_loader_extra_config"] == '{"threads": 2}'
    strategy.run.assert_awaited_once()
    engine.shutdown.assert_called_once_with()
    results.put.assert_not_called()


@pytest.mark.parametrize(
    "lm_request",
    [
        LMRequest(RequestType.CHAT, messages=()),
        LMRequest(RequestType.COMPLETION, prompt="abc", images=("image",)),
        LMRequest(
            RequestType.CHAT,
            messages=({"role": "user", "content": "abc"},),
            tools=(ToolSchema(name="tool", description="test"),),
        ),
        LMRequest(RequestType.LOGLIKELIHOOD, prompt="abc", continuations=("def",)),
    ],
)
def test_unsupported_request_fails_before_engine(setup_engine, lm_request):
    engine, _, _ = setup_engine
    provider = SGLangProvider("model")
    with pytest.raises((NotImplementedError, ValueError)):
        provider.generate([lm_request])
    engine.generate.assert_not_called()


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(fit_max_tokens_to_context=True),
        SamplingParams(truncation_side="left"),
        SamplingParams(num_samples=0),
        SamplingParams(logprobs=-1),
    ],
)
def test_unsupported_sampling_fails_before_engine(setup_engine, params):
    engine, _, _ = setup_engine
    provider = SGLangProvider("model")
    with pytest.raises((NotImplementedError, ValueError)):
        provider.generate([LMRequest(RequestType.COMPLETION, prompt="abc")], params)
    engine.generate.assert_not_called()


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"finish_reason": {"type": "abort", "message": "context overflow"}}, "aborted"),
        ({"completion_tokens": 3}, "every generated token"),
        ({"output_top_logprobs": [[[-0.1, 99, None]]]}, "align"),
        ({"output_token_logprobs": [[float("nan"), 17, None], [-0.7, 18, None]]}, "non-finite"),
    ],
)
def test_invalid_native_output_is_failure_not_a_score(setup_engine, metadata, message):
    engine, _, _ = setup_engine
    provider = SGLangProvider("model")
    output = engine.generate(prompt=["abc"])[0]
    output["meta_info"].update(metadata)
    engine.generate.side_effect = None
    engine.generate.return_value = [output]
    with pytest.raises(RuntimeError, match=message):
        provider.generate([LMRequest(RequestType.COMPLETION, prompt="abc")])


def test_zero_top_logprobs_and_runtime_trace(setup_engine, monkeypatch):
    engine, _, _ = setup_engine
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "1")
    provider = SGLangProvider("model")
    request = LMRequest(RequestType.COMPLETION, prompt="abc")
    params = SamplingParams(logprobs=0)
    output = engine.generate(prompt=["abc"])[0]
    output["meta_info"].pop("output_top_logprobs")
    engine.generate.side_effect = None
    engine.generate.return_value = [output]
    result = provider.generate([request], params)[0][0]
    assert engine.generate.call_args.kwargs["top_logprobs_num"] == 0
    assert result.logprobs is not None and len(result.logprobs) == 2
    assert "top_logprobs" not in result.logprobs[0]
    assert result.metadata["sglang_return_original_logprob"] == "1"
    assert provider.describe_request(request, params)["sglang_return_original_logprob"] == "1"


def test_raw_logprobs_default_precedes_model_registration(setup_engine, monkeypatch):
    _, constructor, _ = setup_engine
    monkeypatch.delenv("SGLANG_RETURN_ORIGINAL_LOGPROB")

    def register():
        assert os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] == "1"
        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.layers.sampler",
            SimpleNamespace(SGLANG_RETURN_ORIGINAL_LOGPROB=True),
        )

    monkeypatch.setitem(sys.modules, "plugin", SimpleNamespace(register=register))
    provider = SGLangProvider("model", model_registration="plugin:register")
    constructor.assert_called_once()
    request = LMRequest(RequestType.COMPLETION, prompt="test")
    assert provider.generate([request])[0][0].metadata["sglang_return_original_logprob"] == "1"


@pytest.mark.parametrize("value", ["0", "false", "invalid"])
def test_conflicting_logprob_environment_rejected(setup_engine, monkeypatch, value):
    _, constructor, _ = setup_engine
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", value)
    with pytest.raises(ValueError, match="requires raw model logprobs"):
        SGLangProvider("model")
    constructor.assert_not_called()


@pytest.mark.parametrize(
    "module", ["sglang.srt.layers.sampler", "sglang.srt.speculative.standalone_worker_v2"]
)
def test_preimported_sampler_convention_rejected(setup_engine, monkeypatch, module):
    _, constructor, _ = setup_engine
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(SGLANG_RETURN_ORIGINAL_LOGPROB=False))
    with pytest.raises(RuntimeError, match="Restart the worker"):
        SGLangProvider("model")
    constructor.assert_not_called()


def test_plugin_cannot_change_logprob_convention(setup_engine, monkeypatch):
    _, constructor, _ = setup_engine

    def register():
        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.layers.sampler",
            SimpleNamespace(SGLANG_RETURN_ORIGINAL_LOGPROB=False),
        )

    monkeypatch.setitem(sys.modules, "plugin", SimpleNamespace(register=register))
    with pytest.raises(RuntimeError, match="Restart the worker"):
        SGLangProvider("model", model_registration="plugin:register")
    constructor.assert_not_called()


def test_environment_change_does_not_mislabel_outputs(setup_engine, monkeypatch):
    engine, _, _ = setup_engine
    provider = SGLangProvider("model")
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "0")
    with pytest.raises(ValueError, match="requires raw model logprobs"):
        provider.generate([LMRequest(RequestType.COMPLETION, prompt="test")])
    engine.generate.assert_not_called()


def capacity_info(capacity=512, tokens=1000000):
    return {
        "dp_size": 1,
        "internal_states": [
            {
                "effective_max_running_requests_per_dp": capacity,
                "memory_usage": {"token_capacity": tokens},
            }
        ],
    }


def test_concurrency_hint_once_without_changing_generation(setup_engine, caplog):
    engine, _, _ = setup_engine
    engine.get_server_info.return_value = capacity_info()
    provider = SGLangProvider("model")
    requests = [LMRequest(RequestType.COMPLETION, prompt="test")] * 64
    params = SamplingParams(max_tokens=100)
    first = provider.generate(requests, params)
    assert "Consider increasing concurrency" not in caplog.text
    engine.get_server_info.assert_not_called()
    assert provider.generate(requests, params) == first
    assert provider.generate(requests, params) == first
    assert caplog.text.count("Consider increasing concurrency") == 1
    assert "effective engine limit is 512" in caplog.text
    assert "-o batching.chunk_size=128" in caplog.text
    engine.get_server_info.assert_called_once_with()
    assert all(call == engine.generate.call_args_list[0] for call in engine.generate.call_args_list)


@pytest.mark.parametrize(
    "info",
    [
        {},
        None,
        {"max_running_requests": 512},  # Requested capacity is insufficient evidence.
        capacity_info(capacity=100),  # The engine reduced the requested limit.
        capacity_info(tokens=26000),  # Full output budget, not the two-token fixture output.
        {**capacity_info(), "dp_size": 2},
        {"internal_states": [None]},
        {"internal_states": [capacity_info()["internal_states"][0]] * 2},
    ],
)
def test_concurrency_hint_requires_resolved_capacity(setup_engine, caplog, info):
    engine, _, _ = setup_engine
    engine.get_server_info.return_value = info
    provider = SGLangProvider("model")
    requests = [LMRequest(RequestType.COMPLETION, prompt="test")] * 64
    for _ in range(2):
        assert len(provider.generate(requests, SamplingParams(max_tokens=100))) == 64
    assert "Consider increasing concurrency" not in caplog.text


def test_concurrency_hint_counts_samples_and_ignores_small_or_unequal_batches(setup_engine, caplog):
    engine, _, _ = setup_engine
    engine.get_server_info.return_value = capacity_info()
    provider = SGLangProvider("model")
    request = LMRequest(RequestType.COMPLETION, prompt="test")
    params = SamplingParams(max_tokens=100, num_samples=4)
    for count in (2, 2, 16, 32):
        provider.generate([request] * count, params)
    engine.get_server_info.assert_not_called()
    provider.generate([request] * 32, params)
    assert "supplied 128 sequences" in caplog.text
    assert "-o batching.chunk_size=64" in caplog.text
    assert "up to 256 sequences at 4 sample(s)" in caplog.text


def test_concurrency_diagnostic_failure_preserves_outputs(setup_engine, caplog):
    engine, _, _ = setup_engine
    engine.get_server_info.side_effect = RuntimeError("telemetry unavailable")
    provider = SGLangProvider("model")
    requests = [LMRequest(RequestType.COMPLETION, prompt="test")] * 64
    first = provider.generate(requests, SamplingParams(max_tokens=100))
    assert provider.generate(requests, SamplingParams(max_tokens=100)) == first
    assert "Consider increasing concurrency" not in caplog.text


def test_concurrency_hint_suppressed_after_retractions(setup_engine, caplog):
    engine, _, _ = setup_engine
    generate = engine.generate.side_effect

    def retracted(**kwargs):
        outputs = generate(**kwargs)
        outputs[0]["meta_info"]["num_retractions"] = 1
        return outputs

    engine.generate.side_effect = retracted
    engine.get_server_info.return_value = capacity_info()
    provider = SGLangProvider("model")
    requests = [LMRequest(RequestType.COMPLETION, prompt="test")] * 64
    for _ in range(2):
        provider.generate(requests, SamplingParams(max_tokens=100))
    engine.get_server_info.assert_not_called()
    assert "Consider increasing concurrency" not in caplog.text


def test_concurrency_hint_uses_longest_prompt_across_both_calls(setup_engine, caplog):
    engine, _, _ = setup_engine
    generate = engine.generate.side_effect
    prompt_length = 5000

    def with_prompt_length(**kwargs):
        outputs = generate(**kwargs)
        outputs[0]["meta_info"]["prompt_tokens"] = prompt_length
        return outputs

    engine.generate.side_effect = with_prompt_length
    engine.get_server_info.return_value = capacity_info(tokens=1000000)
    provider = SGLangProvider("model")
    requests = [LMRequest(RequestType.COMPLETION, prompt="test")] * 64
    provider.generate(requests, SamplingParams(max_tokens=100))
    prompt_length = 5
    provider.generate(requests, SamplingParams(max_tokens=100))
    assert "Consider increasing concurrency" not in caplog.text


def test_small_batch_breaks_consecutive_capacity_observations(setup_engine, caplog):
    engine, _, _ = setup_engine
    engine.get_server_info.return_value = capacity_info()
    provider = SGLangProvider("model")
    request = LMRequest(RequestType.COMPLETION, prompt="test")
    for count in (64, 1, 64):
        provider.generate([request] * count, SamplingParams(max_tokens=100))
    engine.get_server_info.assert_not_called()
    assert "Consider increasing concurrency" not in caplog.text


def test_uncapped_generation_abstains_from_capacity_estimate(setup_engine, caplog):
    engine, _, _ = setup_engine
    engine.get_server_info.return_value = capacity_info()
    provider = SGLangProvider("model")
    requests = [LMRequest(RequestType.COMPLETION, prompt="test")] * 64
    # Include a bounded call first to ensure an uncapped call resets history.
    provider.generate(requests, SamplingParams(max_tokens=100))
    for _ in range(2):
        outputs = provider.generate(requests, SamplingParams(max_tokens=None))
        assert len(outputs) == 64
        assert engine.generate.call_args.kwargs["sampling_params"]["max_new_tokens"] is None
    provider.generate(requests, SamplingParams(max_tokens=100))
    engine.get_server_info.assert_not_called()
    assert "Consider increasing concurrency" not in caplog.text


@pytest.mark.parametrize("chunk_size,expected,sequences", [("auto", 128, 128), (64, 64, None)])
def test_worker_applies_profile_and_preserves_explicit_batch(
    setup_engine, monkeypatch, tmp_path, chunk_size, expected, sequences
):
    engine, constructor, _ = setup_engine
    engine.get_server_info.return_value = {
        "internal_states": [
            {"effective_max_running_requests_per_dp": 128, "memory_usage": {"token_capacity": 1234}}
        ]
    }
    monkeypatch.setattr(
        "olmo_eval.inference.providers.sglang.prepare_profile",
        lambda *args: {
            "selected": "olmo35-h100-v1",
            "defaults": {"max_running_requests": 512, "max_total_tokens": 6684672},
            "reason": "test fixture",
        },
    )
    monkeypatch.setattr("olmo_eval.common.logging.configure_logging", Mock())
    monkeypatch.setattr(
        "olmo_eval.common.logging.configure_worker_logging", Mock(return_value=Mock())
    )
    strategy = SimpleNamespace(run=AsyncMock())
    factory = Mock(return_value=strategy)
    monkeypatch.setattr("olmo_eval.runners.asynq.batching.get_strategy", factory)
    monkeypatch.setattr(
        "olmo_eval.harness.Harness",
        lambda config: SimpleNamespace(
            provider=config.provider.create_provider(),
            initialize_reporters=Mock(),
            cleanup=AsyncMock(),
        ),
    )
    config = HarnessConfig(
        name="test",
        provider=ProviderConfig(kind="sglang", model="model"),
        batching=BatchConfig(chunk_size=chunk_size),
    )
    results = Mock()
    ready = Mock()
    inference_worker(
        "profile",
        [0],
        Mock(),
        results,
        config.to_dict(),
        0,
        init_queue=ready,
        output_dir=str(tmp_path),
    )
    assert constructor.call_args.kwargs["max_running_requests"] == 512
    assert factory.call_args.args[0].chunk_size == expected
    assert factory.call_args.kwargs["max_batch_sequences"] == sequences

    report = json.loads((tmp_path / "runtime/sglang-profile-profile.json").read_text())
    assert report["batch_size"] == expected
    assert report["effective_max_running_requests"] == 128
    assert report["engine_memory"]["token_capacity"] == 1234
    message = ready.put.call_args.args[0]
    assert message.harness_config["batching"]["chunk_size"] == expected
    runtime = message.harness_config["provider"]["resolved_runtime"]
    assert runtime["profile"] == "olmo35-h100-v1"
    assert runtime["effective_max_running_requests"] == 128
    assert "devices" not in runtime
    results.put.assert_not_called()
