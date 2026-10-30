"""Contract tests for the inline SGLang generation provider."""

import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from jinja2 import TemplateError

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
from olmo_eval.inference.errors import REQUEST_ERROR_KEY, is_context_overflow_output, request_error
from olmo_eval.inference.model_profiles import MODEL_SHAPES
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.inference.providers.sglang import SGLangProvider
from olmo_eval.runners.asynq.batching.config import BatchConfig, BatchStrategy
from olmo_eval.runners.asynq.workers import inference_worker
from olmo_eval.runners.io.builders import build_predictions
from olmo_eval.runners.processing.generation_counts import count_generations


@pytest.fixture
def setup_engine(monkeypatch):
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "1")
    tokenizer = Mock(bos_token_id=99, eos_token_id=98)
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


@pytest.mark.parametrize(
    "name,value",
    [
        ("OLMO_SGLANG_CORE_COMPAT", "rounding"),
        ("OLMO_SGLANG_ROUNDING_KERNELS", "torch"),
        ("OLMO_SGLANG_ROUTING_KERNELS", "fused"),
        ("OLMO_SGLANG_ROUTING_TAIL", "fused"),
        ("OLMO_SGLANG_ROUTER_INPUT", "fused"),
        ("OLMO_SGLANG_SHARED_ADD_NORM", "fused"),
        ("OLMO_SGLANG_KDA_PROJECTIONS", "packed"),
        ("OLMO_SGLANG_KDA_GATE_PROJECTIONS", "packed"),
        ("OLMO_SGLANG_MOE_PROJECTIONS", "packed"),
        ("OLMO_SGLANG_SAMPLER", "fused"),
    ],
)
def test_arithmetic_environment_is_captured_at_engine_startup(
    setup_engine, monkeypatch, name, value
):
    _, constructor, _ = setup_engine
    monkeypatch.setenv(name, value)
    monkeypatch.setenv("OLMO_SGLANG_UNRELATED_SECRET", "must-not-be-saved")
    provider = SGLangProvider("model")
    snapshot = provider.profile_report["arithmetic_environment"]
    assert snapshot[name] == value
    assert "OLMO_SGLANG_UNRELATED_SECRET" not in snapshot
    assert "arithmetic_environment" not in constructor.call_args.kwargs
    # Changing the parent environment later cannot rewrite engine-startup provenance.
    monkeypatch.setenv(name, "changed-after-startup")
    assert snapshot[name] == value


def test_unset_arithmetic_option_is_preserved_as_null(setup_engine, monkeypatch):
    monkeypatch.delenv("OLMO_SGLANG_ROUTING_KERNELS", raising=False)
    provider = SGLangProvider("model")
    report = json.loads(json.dumps(provider.profile_report))
    assert "OLMO_SGLANG_ROUTING_KERNELS" in report["arithmetic_environment"]
    assert report["arithmetic_environment"]["OLMO_SGLANG_ROUTING_KERNELS"] is None


@pytest.mark.parametrize(
    "device",
    ["NVIDIA H100, 81559, 81000", "NVIDIA B300, 288000, 287000", "NVIDIA A100, 81251, 81000"],
)
def test_named_and_auto_profiles_produce_identical_engine_options(
    setup_engine, monkeypatch, tmp_path, device
):
    _, constructor, _ = setup_engine
    (tmp_path / "config.json").write_text(
        json.dumps(MODEL_SHAPES["olmo3.5-12.5b"] | {"dtype": "bfloat16"})
    )
    monkeypatch.setitem(sys.modules, "olmo_sglang", SimpleNamespace(register=Mock()))
    monkeypatch.setattr(
        "olmo_eval.inference.providers.sglang_profiles.subprocess.check_output",
        lambda *args, **kwargs: device,
    )
    calls = []
    for selector in ("auto", "olmo3.5-12.5b"):
        config = ProviderConfig(
            kind="sglang",
            model=str(tmp_path),
            max_model_len=8192,
            max_concurrency=512,
            kwargs={"model_registration": "olmo_sglang:register", "profile": selector},
        )
        provider = ProviderConfig.from_dict(config.to_dict()).create_provider()
        assert provider.profile_report["requested"] == selector
        assert provider.profile_report["model_profile"] == "olmo3.5-12.5b"
        assert provider.recommended_batch_sequences == 512
        calls.append(constructor.call_args.kwargs)
        provider.close()
    assert calls[0] == calls[1]
    assert calls[0]["max_running_requests"] == 512
    assert "profile" not in calls[0]


def test_named_profile_mismatch_fails_before_engine_construction(
    setup_engine, monkeypatch, tmp_path
):
    _, constructor, _ = setup_engine
    (tmp_path / "config.json").write_text(
        json.dumps(MODEL_SHAPES["olmo3.5-12.5b"] | {"hidden_size": 2048, "dtype": "bfloat16"})
    )
    monkeypatch.setitem(sys.modules, "olmo_sglang", SimpleNamespace(register=Mock()))
    monkeypatch.setattr(
        "olmo_eval.inference.providers.sglang_profiles.subprocess.check_output",
        lambda *args, **kwargs: "NVIDIA H100, 81559, 81000",
    )
    with pytest.raises(ValueError, match="geometry/architecture"):
        SGLangProvider(
            str(tmp_path),
            profile="olmo3.5-12.5b",
            model_registration="olmo_sglang:register",
            max_model_len=8192,
        )
    constructor.assert_not_called()


@pytest.mark.parametrize("selector", ["auto", "olmo3.5-12.5b"])
@pytest.mark.parametrize("conflict", [{"max_concurrency": 128}, {"disable_cuda_graph": True}])
def test_profile_conflict_fails_before_engine_construction(
    setup_engine, monkeypatch, tmp_path, selector, conflict
):
    _, constructor, _ = setup_engine
    (tmp_path / "config.json").write_text(
        json.dumps(MODEL_SHAPES["olmo3.5-12.5b"] | {"dtype": "bfloat16"})
    )
    monkeypatch.setitem(sys.modules, "olmo_sglang", SimpleNamespace(register=Mock()))
    monkeypatch.setattr(
        "olmo_eval.inference.providers.sglang_profiles.subprocess.check_output",
        lambda *args, **kwargs: "NVIDIA H100, 81559, 81000",
    )
    with pytest.raises(ValueError, match="Settings conflict.*profile=generic"):
        SGLangProvider(
            str(tmp_path),
            profile=selector,
            model_registration="olmo_sglang:register",
            max_model_len=8192,
            **conflict,
        )
    constructor.assert_not_called()


def test_generic_profile_allows_manual_engine_settings(setup_engine, monkeypatch):
    _, constructor, _ = setup_engine
    monkeypatch.setitem(sys.modules, "olmo_sglang", SimpleNamespace(register=Mock()))
    SGLangProvider(
        "model",
        profile="generic",
        model_registration="olmo_sglang:register",
        max_concurrency=128,
        disable_cuda_graph=True,
    )
    assert constructor.call_args.kwargs["max_running_requests"] == 128
    assert constructor.call_args.kwargs["disable_cuda_graph"] is True
    assert "cuda_graph_backend_decode" not in constructor.call_args.kwargs


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


@pytest.mark.parametrize("partial_logprobs", [False, True])
def test_abort_isolates_request_and_marks_incomplete_sample_group(setup_engine, partial_logprobs):
    engine, _, _ = setup_engine
    provider = SGLangProvider("model")
    outputs = engine.generate(prompt=["x"] * 4)
    outputs[1] = {
        "meta_info": {
            "finish_reason": {"type": "abort", "message": "input length exceeds context length"}
        }
    }
    if partial_logprobs:
        outputs[1]["meta_info"].update(
            output_token_logprobs=[(-0.3, 10, "partial")],
            output_top_logprobs=[[(-0.1, 20, "alternative")]],
        )
    engine.generate.side_effect = None
    engine.generate.return_value = outputs
    results = provider.generate(
        [LMRequest(RequestType.COMPLETION, prompt=p) for p in ("a", "b")],
        SamplingParams(num_samples=2),
    )
    assert request_error(results[0]) is not None
    assert all(REQUEST_ERROR_KEY in output.metadata for output in results[0])
    assert is_context_overflow_output(results[0])
    aborted = results[0][1]
    assert aborted.metadata["finish_reason"] == "abort"
    assert aborted.metadata["sglang_finish_reason"] == outputs[1]["meta_info"]["finish_reason"]
    assert "output_token_logprobs" not in aborted.metadata
    assert "output_top_logprobs" not in aborted.metadata
    assert ("output_token_logprobs" in outputs[1]["meta_info"]) is partial_logprobs
    assert request_error(results[1]) is None
    assert [output.text for output in results[1]] == ["response2", "response3"]


def test_template_failure_is_request_local_and_rendered_only_once(setup_engine):
    engine, _, tokenizer = setup_engine
    provider = SGLangProvider("model")
    requests = [
        LMRequest(RequestType.CHAT, messages=({"role": "user", "content": p},))
        for p in ("bad", "good")
    ]

    def render(messages, **kwargs):
        if messages[0]["content"] == "bad":
            raise TemplateError("unsupported turn")
        return "rendered"

    tokenizer.apply_chat_template.side_effect = render
    for request in requests:
        provider.describe_request(request)
    tokenizer.apply_chat_template.assert_not_called()
    results = provider.generate(requests, SamplingParams(num_samples=2))
    assert request_error(results[0]) == "unsupported turn"
    assert request_error(results[1]) is None
    assert tokenizer.apply_chat_template.call_count == 2
    assert engine.generate.call_args.kwargs["prompt"] == ["rendered", "rendered"]


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


@pytest.fixture
def multimodal_engine(setup_engine):
    engine, constructor, tokenizer = setup_engine
    processor = SimpleNamespace(
        document_prompt_ids=Mock(
            side_effect=lambda text, num_images: [
                99,
                *([70] * num_images),
                *tokenizer.encode(text, add_special_tokens=False),
            ]
        ),
        apply_chat_template=Mock(return_value="<image>question<assistant>"),
    )
    engine.tokenizer_manager.processor = processor
    return engine, constructor, tokenizer, processor


def test_document_images_samples_and_mixed_requests(multimodal_engine, tmp_path):
    from PIL import Image

    engine, constructor, tokenizer, processor = multimodal_engine
    path = tmp_path / "page.png"
    Image.new("RGB", (2, 3), "red").save(path)
    second = Image.new("RGB", (4, 5), "blue")
    lazy = Mock(return_value=[path, second, None])
    requests = [
        LMRequest(
            RequestType.CHAT, messages=({"role": "user", "content": "olmocr:"},), images=(lazy,)
        ),
        LMRequest(RequestType.COMPLETION, prompt="text control"),
    ]
    config = ProviderConfig(
        kind="sglang",
        model="model",
        kwargs={"enable_multimodal": True, "prompt_format": "document"},
    )
    provider = ProviderConfig.from_dict(config.to_dict()).create_provider()
    assert provider.supports_images
    result = provider.generate(requests, SamplingParams(num_samples=2, max_tokens=32))
    assert [[o.text for o in group] for group in result] == [
        ["response0", "response1"],
        ["response2", "response3"],
    ]
    lazy.assert_called_once_with()
    kwargs = engine.generate.call_args.kwargs
    assert kwargs["input_ids"] == [[99, 70, 70, 10, 20]] * 2 + [[99, 10, 20]] * 2
    assert [image.size for image in kwargs["image_data"][0]] == [(2, 3), (4, 5)]
    assert kwargs["image_data"][0][1] is second
    assert kwargs["image_data"][2:] == [None, None]
    assert result[0][0].metadata["input_token_ids"] == [99, 70, 70, 10, 20]
    assert result[0][0].metadata["num_images"] == 2
    assert result[1][0].metadata["num_images"] == 0
    assert result[0][0].metadata["prompt_format"] == "document"
    processor.apply_chat_template.assert_not_called()
    tokenizer.apply_chat_template.assert_not_called()
    assert constructor.call_args.kwargs["enable_multimodal"] is True
    assert "prompt_format" not in constructor.call_args.kwargs


def test_document_turn_boundaries_match_core(multimodal_engine):
    engine, _, tokenizer, processor = multimodal_engine
    provider = SGLangProvider("model", enable_multimodal=True, prompt_format="document")
    request = LMRequest(
        RequestType.COMPLETION,
        prompt="question",
        messages=({"role": "system", "content": "instructions"},),
    )
    provider.generate([request])
    processor.document_prompt_ids.assert_called_once_with("instructions", num_images=0)
    assert [call.args for call in tokenizer.encode.call_args_list] == [
        ("instructions",),
        (" question",),
    ]
    assert engine.generate.call_args.kwargs["input_ids"] == [[99, 10, 20, 10, 20]]
    assert "image_data" not in engine.generate.call_args.kwargs


def test_image_chat_uses_processor_and_preserves_system_turn(multimodal_engine):
    from PIL import Image

    engine, _, tokenizer, processor = multimodal_engine
    image = Image.new("RGB", (2, 2))
    request = LMRequest(
        RequestType.CHAT,
        messages=(
            {"role": "system", "content": "instruction"},
            {"role": "user", "content": "question"},
        ),
        images=(image,),
    )
    provider = SGLangProvider("model", enable_multimodal=True)
    provider.generate([request])
    processor.apply_chat_template.assert_called_once_with(
        [
            {"role": "system", "content": [{"type": "text", "text": "instruction"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "question"},
                ],
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenizer.encode.assert_called_once_with("<image>question<assistant>", add_special_tokens=False)
    tokenizer.apply_chat_template.assert_not_called()
    assert engine.generate.call_args.kwargs["image_data"] == [[image]]


def test_multimodal_trace_does_not_decode_images(multimodal_engine):
    loader = Mock(side_effect=AssertionError("trace must not resolve pixels"))
    provider = SGLangProvider("model", enable_multimodal=True, prompt_format="document")
    request = LMRequest(
        RequestType.CHAT, messages=({"role": "user", "content": "olmocr:"},), images=(loader,)
    )
    trace = provider.describe_request(request)
    assert trace["input_mode"] == "input_ids"
    assert trace["prompt_format"] == "document" and trace["images_attached"]
    loader.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt_format": "unknown"},
        {"prompt_format": "document"},
        {"prompt_format": "document", "enable_multimodal": True, "add_bos_token": False},
        {"enable_multimodal": True, "language_model_only": True},
    ],
)
def test_invalid_image_configuration_fails_before_engine(setup_engine, kwargs):
    _, constructor, _ = setup_engine
    with pytest.raises(ValueError):
        SGLangProvider("model", **kwargs)
    constructor.assert_not_called()


def test_empty_lazy_image_is_not_silently_dropped(multimodal_engine):
    engine, _, _, _ = multimodal_engine
    provider = SGLangProvider("model", enable_multimodal=True, prompt_format="document")
    with pytest.raises(ValueError, match="no images"):
        provider.generate(
            [LMRequest(RequestType.COMPLETION, prompt="question", images=(lambda: None,))]
        )
    engine.generate.assert_not_called()


def test_document_requires_processor_protocol(multimodal_engine):
    engine, _, _, _ = multimodal_engine
    engine.tokenizer_manager.processor = SimpleNamespace()
    provider = SGLangProvider("model", enable_multimodal=True, prompt_format="document")
    with pytest.raises(NotImplementedError, match="document_prompt_ids"):
        provider.generate([LMRequest(RequestType.COMPLETION, prompt="question")])
    engine.generate.assert_not_called()


def test_multimodal_rejects_tools_without_discarding_other_requests(multimodal_engine):
    engine, _, _, _ = multimodal_engine
    from PIL import Image

    loader = Mock(return_value=Image.new("RGB", (2, 2)))
    provider = SGLangProvider("model", enable_multimodal=True, prompt_format="document")
    valid = LMRequest(RequestType.COMPLETION, prompt="question", images=(loader,))
    invalid = LMRequest(
        RequestType.COMPLETION, prompt="tool", tools=(ToolSchema(name="f", description="test"),)
    )
    results = provider.generate([valid, invalid])
    assert request_error(results[0]) is None
    assert request_error(results[1]) is not None
    loader.assert_called_once()
    assert len(engine.generate.call_args.kwargs["input_ids"]) == 1


def test_image_request_rejects_embedded_media_content(multimodal_engine):
    engine, _, _, _ = multimodal_engine
    provider = SGLangProvider("model", enable_multimodal=True, prompt_format="document")
    request = LMRequest(
        RequestType.CHAT,
        messages=({"role": "user", "content": [{"type": "image_url", "image_url": "unbound"}]},),
    )
    with pytest.raises(NotImplementedError, match="plain text"):
        provider.generate([request])
    engine.generate.assert_not_called()


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
            "selected": "olmo3.5-12.5b-sglang-h100-v1",
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
    assert runtime["profile"] == "olmo3.5-12.5b-sglang-h100-v1"
    assert runtime["effective_max_running_requests"] == 128
    assert "devices" not in runtime
    results.put.assert_not_called()


@pytest.mark.parametrize(
    "prefix,bos,eos,expected",
    [
        ([], 99, 98, [99, 10]),
        ([99], 99, 98, [99, 10]),
        ([], None, 98, [98, 10]),
        ([], 0, 98, [0, 10]),
    ],
)
def test_multimodal_chat_has_one_boundary_with_or_without_images(
    multimodal_engine, prefix, bos, eos, expected
):
    from PIL import Image

    engine, _, tokenizer, processor = multimodal_engine
    tokenizer.bos_token_id, tokenizer.eos_token_id = bos, eos
    tokenizer.encode.side_effect = lambda *args, **kwargs: [*prefix, 10]
    provider = SGLangProvider("model", enable_multimodal=True)
    requests = [
        LMRequest(
            RequestType.CHAT, messages=({"role": "user", "content": "question"},), images=images
        )
        for images in ((Image.new("RGB", (2, 2)),), ())
    ]
    outputs = provider.generate(requests)
    assert engine.generate.call_args.kwargs["input_ids"] == [expected, expected]
    assert [o[0].metadata["input_token_ids"] for o in outputs] == [expected, expected]
    assert processor.apply_chat_template.call_count == 2
    tokenizer.apply_chat_template.assert_not_called()
    assert all(
        call.kwargs == {"add_special_tokens": False} for call in tokenizer.encode.call_args_list
    )


def test_bad_image_is_request_local(multimodal_engine):
    from PIL import Image

    engine, _, _, _ = multimodal_engine
    provider = SGLangProvider("model", enable_multimodal=True, prompt_format="document")
    broken = Mock(side_effect=OSError("corrupt image"))
    requests = [
        LMRequest(RequestType.COMPLETION, prompt="question", images=(image,))
        for image in (broken, Image.new("RGB", (2, 2)))
    ]
    outputs = provider.generate(requests, SamplingParams(num_samples=2))
    assert request_error(outputs[0]) == "corrupt image"
    assert request_error(outputs[1]) is None
    assert len(engine.generate.call_args.kwargs["input_ids"]) == 2


def test_document_system_turn_with_image_matches_core(multimodal_engine):
    from PIL import Image

    engine, _, tokenizer, processor = multimodal_engine
    provider = SGLangProvider("model", enable_multimodal=True, prompt_format="document")
    request = LMRequest(
        RequestType.CHAT,
        messages=({"role": "system", "content": "SYS"}, {"role": "user", "content": "Q"}),
        images=(Image.new("RGB", (2, 2)),),
    )
    provider.generate([request])
    processor.document_prompt_ids.assert_called_once_with("SYS", num_images=1)
    assert [call.args for call in tokenizer.encode.call_args_list] == [("SYS",), (" Q",)]
    assert engine.generate.call_args.kwargs["input_ids"] == [[99, 70, 10, 20, 10, 20]]


def test_request_trace_keeps_only_profile_identity_and_options(setup_engine):
    provider = SGLangProvider("model")
    provider.profile_report.update(
        devices=[{"free_memory": 123}], engine_memory={"token_capacity": 456}
    )
    trace = provider.describe_request(LMRequest(RequestType.COMPLETION, prompt="test"))
    assert trace["serving_profile"] == {
        "selected": provider.profile_report["selected"],
        "effective_options": provider.profile_report["effective_options"],
    }
    assert "devices" in provider.profile_report


def test_output_does_not_duplicate_raw_logprob_arrays(setup_engine):
    provider = SGLangProvider("model")
    output = provider.generate([LMRequest(RequestType.COMPLETION, prompt="test")])[0][0]
    assert [entry["logprob"] for entry in output.logprobs] == [-0.3, -0.7]
    assert output.logprobs[0]["top_logprobs"] == [{"token": "token99", "logprob": -0.1}]
    assert "output_token_logprobs" not in output.metadata
    assert "output_top_logprobs" not in output.metadata
    assert output.metadata["sum_logits"] == -1.0


def test_request_length_caps_reach_the_profile_report(setup_engine):
    engine, _, _ = setup_engine
    engine.scheduler_info = {"max_total_num_tokens": 4096, "max_req_input_len": 4000}
    provider = SGLangProvider("model")
    assert provider.profile_report["max_total_num_tokens"] == 4096
    assert provider.profile_report["max_req_input_len"] == 4000
    engine.get_server_info.assert_not_called()


@pytest.mark.parametrize(
    "scheduler_info",
    [
        {"max_total_num_tokens": 0, "max_req_input_len": 0},
        {"max_total_num_tokens": -5, "max_req_input_len": -5},
        {"max_total_num_tokens": 4096.0, "max_req_input_len": "4000"},
        {"max_total_num_tokens": True, "max_req_input_len": None},
        {},
    ],
)
def test_unusable_request_length_caps_are_left_out_of_the_profile_report(
    setup_engine, scheduler_info
):
    engine, _, _ = setup_engine
    engine.scheduler_info = scheduler_info
    provider = SGLangProvider("model")
    assert "max_total_num_tokens" not in provider.profile_report
    assert "max_req_input_len" not in provider.profile_report


def test_each_request_length_cap_is_kept_on_its_own(setup_engine):
    engine, _, _ = setup_engine
    engine.scheduler_info = {"max_total_num_tokens": 4096, "max_req_input_len": 0}
    provider = SGLangProvider("model")
    assert provider.profile_report["max_total_num_tokens"] == 4096
    assert "max_req_input_len" not in provider.profile_report
