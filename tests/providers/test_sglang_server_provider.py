"""Unit tests for SGLangServerProvider and the api_key_env it shares with vllm_server."""

from unittest.mock import MagicMock, patch

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.inference.providers.sglang_server import SGLangServerProvider
from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

BASE_URL = "http://127.0.0.1:30000/v1"


@pytest.fixture(autouse=True)
def no_beaker_status():
    with patch("olmo_eval.inference.providers.vllm_server.BeakerStatusReporter"):
        yield


@pytest.fixture
def tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text, add_special_tokens=True: text.split()
    tokenizer.apply_chat_template.side_effect = (
        lambda messages, tools=None, add_generation_prompt=False, tokenize=True, **kwargs: (
            " ".join(m["content"] for m in messages) + (" thinking" if kwargs else "")
        )
    )
    return tokenizer


@pytest.fixture
def provider(tokenizer):
    provider = SGLangServerProvider("test-model", base_url=BASE_URL, max_model_len=4096)
    provider._tokenizer = tokenizer
    return provider


def test_config_creates_the_provider_with_its_fields():
    config = ProviderConfig(
        kind="sglang_server",
        model="test-model",
        base_url=BASE_URL,
        max_model_len=4096,
        max_concurrency=8,
        kwargs={"chat_template_kwargs": {"enable_thinking": False}},
    )

    provider = config.create_provider()

    assert isinstance(provider, SGLangServerProvider)
    assert provider.base_url == BASE_URL
    assert provider.max_length == 4096
    assert provider.max_concurrency == 8
    assert provider.chat_template_kwargs == {"enable_thinking": False}
    assert not config.requires_local_gpu


def test_refuses_to_start_without_a_server():
    with pytest.raises(ValueError, match="base_url"):
        SGLangServerProvider("test-model")


def test_api_key_env_authenticates_every_client(monkeypatch):
    monkeypatch.setenv("ROUTER_KEY", "secret")

    provider = SGLangServerProvider("test-model", base_url=BASE_URL, api_key_env="ROUTER_KEY")

    assert provider.get_openai_client().api_key == "secret"
    assert provider._get_raw_http_client().headers["Authorization"] == "Bearer secret"


def test_api_key_env_must_be_set(monkeypatch):
    monkeypatch.delenv("ROUTER_KEY", raising=False)

    with pytest.raises(ValueError, match="ROUTER_KEY"):
        VLLMServerProvider("test-model", base_url=BASE_URL, api_key_env="ROUTER_KEY")


def test_without_api_key_env_sends_no_authorization():
    provider = VLLMServerProvider("test-model", base_url=BASE_URL)

    assert "Authorization" not in provider._get_raw_http_client().headers


def test_tokenizer_is_always_local(provider, tokenizer):
    assert provider.get_tokenizer() is tokenizer


@pytest.mark.anyio
async def test_counts_chat_prompts_with_the_chat_template(provider):
    count = await provider._count_tokens(
        {"model": "test-model", "messages": [{"role": "user", "content": "one two three"}]}
    )

    assert count == 3


@pytest.mark.anyio
async def test_counts_chat_prompts_with_chat_template_kwargs(provider):
    count = await provider._count_tokens(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "one two three"}],
            "chat_template_kwargs": {"enable_thinking": True},
        }
    )

    assert count == 4


@pytest.mark.anyio
async def test_counts_completion_prompts_with_the_tokenizer(provider):
    count = await provider._count_tokens(
        {"model": "test-model", "prompt": "one two", "add_special_tokens": False}
    )

    assert count == 2


def test_refuses_prompt_truncation(provider):
    with pytest.raises(ValueError, match="truncate_prompt_tokens"):
        provider._get_prompt_truncation(SamplingParams(truncate_prompt_tokens=100))


def test_describes_requests_as_itself(provider):
    request = LMRequest(prompt="Hi", request_type=RequestType.CHAT)

    assert provider.describe_request(request)["provider"] == "SGLangServerProvider"


@pytest.mark.anyio
async def test_refuses_loglikelihood(provider):
    request = LMRequest(prompt="Q", continuations=("A",), request_type=RequestType.LOGLIKELIHOOD)

    with pytest.raises(NotImplementedError, match="prompt_logprobs"):
        await provider.alogprobs([request])
