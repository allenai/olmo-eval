"""Unit tests for the OLMo-core provider."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from olmo_eval.common.constants.infrastructure import BACKEND_OPTIONAL_GROUPS
from olmo_eval.common.types import LMRequest, ProviderKind, RequestType, SamplingParams
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.inference.providers.olmo_core import (
    OlmoCoreProvider,
    _validate_olmo_core_checkpoint,
)
from olmo_eval.runners.asynq.batching.config import BatchConfig, BatchStrategy


class FakeTransformerConfig:
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> object:
        if data.get("invalid"):
            raise ValueError("bad model config")
        return object()


class FakeTokenizerConfig:
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            identifier=data.get("identifier"),
            pad_token_id=data.get("pad_token_id"),
            eos_token_id=data.get("eos_token_id"),
        )


def _write_olmo_config(
    checkpoint_dir,
    *,
    model: dict[str, Any] | None = None,
    tokenizer: dict[str, Any] | None = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text(
        json.dumps(
            {
                "model": model if model is not None else {"d_model": 8},
                "dataset": {
                    "tokenizer": tokenizer
                    if tokenizer is not None
                    else {
                        "identifier": "fake-tokenizer",
                        "vocab_size": 16,
                        "pad_token_id": 0,
                        "eos_token_id": 2,
                    }
                },
            }
        )
    )


def _metadata_reader(path: str) -> SimpleNamespace:
    if (path if hasattr(path, "exists") else None) is not None:
        path_obj = path
    else:
        from pathlib import Path

        path_obj = Path(path)
    if (path_obj / ".metadata").exists():
        return SimpleNamespace(state_dict_metadata={"model.transformer.wte.weight": object()})
    raise FileNotFoundError(path)


def test_validates_raw_olmo_core_checkpoint(tmp_path) -> None:
    checkpoint_dir = tmp_path / "step1000"
    _write_olmo_config(checkpoint_dir)
    (checkpoint_dir / "model_and_optim").mkdir()
    (checkpoint_dir / "model_and_optim" / ".metadata").write_text("fake")

    info = _validate_olmo_core_checkpoint(
        str(checkpoint_dir),
        TransformerConfig=FakeTransformerConfig,
        TokenizerConfig=FakeTokenizerConfig,
        get_checkpoint_metadata=_metadata_reader,
    )

    assert info.tokenizer_config.identifier == "fake-tokenizer"
    assert info.metadata_dir == str(checkpoint_dir / "model_and_optim")


def test_rejects_hf_style_checkpoint_config(tmp_path) -> None:
    checkpoint_dir = tmp_path / "hf"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (checkpoint_dir / ".metadata").write_text("fake")

    with pytest.raises(ValueError, match="HF-format checkpoints should use"):
        _validate_olmo_core_checkpoint(
            str(checkpoint_dir),
            TransformerConfig=FakeTransformerConfig,
            TokenizerConfig=FakeTokenizerConfig,
            get_checkpoint_metadata=_metadata_reader,
        )


def test_rejects_checkpoint_without_distributed_metadata(tmp_path) -> None:
    checkpoint_dir = tmp_path / "missing-metadata"
    _write_olmo_config(checkpoint_dir)

    with pytest.raises(ValueError, match="missing distributed checkpoint metadata"):
        _validate_olmo_core_checkpoint(
            str(checkpoint_dir),
            TransformerConfig=FakeTransformerConfig,
            TokenizerConfig=FakeTokenizerConfig,
            get_checkpoint_metadata=_metadata_reader,
        )


def test_rejects_invalid_checkpoint_token_ids(tmp_path) -> None:
    checkpoint_dir = tmp_path / "bad-tokenizer"
    _write_olmo_config(
        checkpoint_dir,
        tokenizer={
            "identifier": "fake-tokenizer",
            "vocab_size": 16,
            "pad_token_id": 2,
            "eos_token_id": 2,
        },
    )
    (checkpoint_dir / ".metadata").write_text("fake")

    with pytest.raises(ValueError, match="must be different"):
        _validate_olmo_core_checkpoint(
            str(checkpoint_dir),
            TransformerConfig=FakeTransformerConfig,
            TokenizerConfig=FakeTokenizerConfig,
            get_checkpoint_metadata=_metadata_reader,
        )


def test_provider_registration_and_config_round_trip(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_create_provider(kind: str, model: str, **kwargs: Any) -> object:
        captured.update({"kind": kind, "model": model, "kwargs": kwargs})
        return object()

    import olmo_eval.inference as inference_module

    monkeypatch.setattr(inference_module, "create_provider", fake_create_provider)
    config = ProviderConfig(
        kind=ProviderKind.OLMO_CORE,
        model="/weka/ckpts/step1000",
        tokenizer="allenai/dolma2-tokenizer",
        dtype="bfloat16",
        max_model_len=4096,
        kwargs={"attention_backend": "torch", "validate_checkpoint": False},
    )

    restored = ProviderConfig.from_dict(config.to_dict())
    restored.create_provider()

    assert restored.kind == ProviderKind.OLMO_CORE
    assert restored.requires_gpu is True
    assert restored.requires_local_gpu is True
    assert BACKEND_OPTIONAL_GROUPS["olmo_core"] == "olmo_core"
    assert captured == {
        "kind": ProviderKind.OLMO_CORE,
        "model": "/weka/ckpts/step1000",
        "kwargs": {
            "attention_backend": "torch",
            "validate_checkpoint": False,
            "tokenizer": "allenai/dolma2-tokenizer",
            "dtype": "bfloat16",
            "max_model_len": 4096,
        },
    }


def test_olmo_core_is_sequential_only() -> None:
    with pytest.raises(ValueError, match="only supports batched processing"):
        BatchConfig(strategy=BatchStrategy.STREAMING).validate_for_provider("olmo_core")


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    model_max_length = 32

    def __init__(self) -> None:
        self.template_calls: list[dict[str, Any]] = []
        self.vocab = {
            "": [],
            "Prompt": [10, 11],
            "Other": [12],
            "!": [6],
            " STOP": [4],
            "a": [1],
            "ab": [1, 2],
            "abc": [1, 2, 3],
            "bc": [2, 3],
        }
        self.id_to_text = {
            0: "<pad>",
            1: "a",
            2: "b",
            3: "c",
            4: " STOP",
            5: "hello",
            6: "!",
            7: "x",
            10: "P",
            11: "rompt",
            12: "Other",
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in self.vocab:
            return self.vocab[text]
        return [ord(char) % 13 + 3 for char in text]

    def decode(self, token_ids: int | list[int], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        pieces = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            pieces.append(self.id_to_text.get(token_id, str(token_id)))
        return "".join(pieces)

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.template_calls.append({"messages": messages, **kwargs})
        return "<chat>" + "|".join(message["content"] for message in messages)


class FakeGenerationModule:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.forward_calls: list[Any] = []
        self.free_calls = 0

    def generate_batch(self, **kwargs: Any):
        torch = pytest.importorskip("torch")

        self.generate_calls.append(kwargs)
        batch_size = kwargs["input_ids"].shape[0]
        rows = [[5, 4, 0] if idx % 2 == 0 else [5, 6, 0] for idx in range(batch_size)]
        generated = torch.tensor(rows, dtype=torch.long)
        logprobs = torch.tensor(
            [
                [-0.1, -0.2, -9.0] if idx % 2 == 0 else [-0.3, -0.4, -9.0]
                for idx in range(batch_size)
            ],
            dtype=torch.float32,
        )
        return generated, None, logprobs

    def model_forward(self, *, input_ids):
        torch = pytest.importorskip("torch")

        self.forward_calls.append(input_ids)
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, 8), dtype=torch.float32)
        logits[:, :, :] = -5.0
        logits[:, 0, 2] = 5.0
        logits[:, 1, 4] = 5.0
        logits[:, 1, 3] = 2.0
        return logits

    def free_inference_cache(self) -> None:
        self.free_calls += 1


@pytest.fixture
def fake_provider() -> tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer]:
    torch = pytest.importorskip("torch")

    provider = OlmoCoreProvider.__new__(OlmoCoreProvider)
    tokenizer = FakeTokenizer()
    module = FakeGenerationModule()
    provider.model_name = "fake-model"
    provider.tokenizer = tokenizer
    provider.generation_module = module
    provider.pad_token_id = tokenizer.pad_token_id
    provider.eos_token_id = tokenizer.eos_token_id
    provider.use_cache = True
    provider.batch_size = None
    provider.chat_template = None
    provider.device = torch.device("cpu")
    provider.max_length = 32
    return provider, module, tokenizer


def test_generate_repeats_prompts_for_num_samples_and_trims_stops(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, module, _ = fake_provider
    requests = [
        LMRequest(request_type=RequestType.COMPLETION, prompt="Prompt"),
        LMRequest(request_type=RequestType.COMPLETION, prompt="Other"),
    ]

    outputs = provider.generate(
        requests,
        SamplingParams(
            max_tokens=3,
            num_samples=2,
            temperature=0.7,
            top_p=None,
            top_k=None,
            stop_sequences=(" STOP", "!"),
        ),
    )

    call = module.generate_calls[0]
    assert call["input_ids"].tolist() == [
        [10, 11],
        [10, 11],
        [0, 12],
        [0, 12],
    ]
    assert call["attention_mask"].tolist() == [
        [1, 1],
        [1, 1],
        [0, 1],
        [0, 1],
    ]
    assert call["max_new_tokens"] == 3
    assert call["top_k"] == -1
    assert call["top_p"] == 1.0
    assert call["stop_token_ids"] == [4, 6]
    assert [output.text for output in outputs[0]] == ["hello", "hello"]
    assert outputs[0][0].metadata["sum_logits"] == pytest.approx(-0.3)
    assert outputs[1][1].metadata["num_tokens"] == 2
    assert module.free_calls == 1


def test_generate_formats_chat_with_optional_template(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, module, tokenizer = fake_provider
    provider.chat_template = "custom-template"

    provider.generate(
        [
            LMRequest(
                request_type=RequestType.CHAT,
                messages=({"role": "user", "content": "Hello"},),
            )
        ],
        SamplingParams(max_tokens=1),
    )

    assert tokenizer.template_calls == [
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "tokenize": False,
            "add_generation_prompt": True,
            "chat_template": "custom-template",
        }
    ]
    assert module.generate_calls[0]["input_ids"].shape[0] == 1


def test_logprobs_uses_model_forward_and_computes_greedy_metadata(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, module, _ = fake_provider
    request = LMRequest(
        request_type=RequestType.LOGLIKELIHOOD,
        prompt="a",
        continuations=("bc",),
    )

    outputs = provider.logprobs([request])

    assert module.forward_calls[0].tolist() == [[1, 2]]
    output = outputs[0][0]
    assert output.text == "bc"
    assert [entry["token"] for entry in output.logprobs or []] == ["b", "c"]
    assert output.metadata["num_tokens"] == 2
    assert output.metadata["num_tokens_all"] == 3
    assert output.metadata["is_greedy"] is False
    assert output.metadata["total_logprob"] < 0
    assert module.free_calls == 1
