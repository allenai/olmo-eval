"""Contract tests for the inline SGLang generation provider."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from olmo_eval.common.types import LMRequest, ProviderKind, RequestType, SamplingParams
from olmo_eval.inference import create_provider
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.inference.providers.sglang import SGLangProvider
from olmo_eval.runners.asynq.batching.config import BatchConfig, BatchStrategy


@pytest.fixture
def setup_engine(monkeypatch):
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
    provider.shutdown()
    engine.shutdown.assert_called_once_with()
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
    assert output.logprobs[0]["top_logprobs"] == [{"token": "token99", "logprob": -0.1}]
    assert output.metadata["output_token_ids"] == [17, 18]
    assert output.metadata["sum_logits"] == -1.0
    assert output.metadata["finish_reason"] == {"type": "stop", "matched": 18}
    trace = provider.describe_request(request, params)
    assert trace is not None
    assert trace["generation_kwargs"]["top_p"] == 0.95


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
