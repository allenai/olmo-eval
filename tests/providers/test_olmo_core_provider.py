"""Hotspot tests for the OLMo-core provider."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import olmo_eval.inference.providers.olmo_core_utils as olmo_core_utils
from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.olmo_core import OlmoCoreProvider
from olmo_eval.inference.providers.olmo_core_utils import _TRANSFORMERS_UNSET_MODEL_MAX_LENGTH


class FakeTokenizerConfig:
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            identifier=data.get("identifier"),
            pad_token_id=data.get("pad_token_id"),
            eos_token_id=data.get("eos_token_id"),
        )


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    model_max_length = 32

    def __init__(self) -> None:
        self.add_bos_token = False
        self.encode_calls: list[dict[str, Any]] = []
        self.decode_calls: list[list[int]] = []
        self.vocab = {
            "": [],
            "Prompt": [10, 11],
            "Prompt!": [10, 11, 6],
            "Other": [12],
            "!": [6],
            " !": [13, 6],
            " STOP": [4],
            "STOP": [8, 9],
        }
        self.id_to_text = {
            0: "<pad>",
            1: "<bos>",
            2: "<eos>",
            4: " STOP",
            5: "hello",
            6: "!",
            7: "x",
            8: "ST",
            9: "OP",
            10: "P",
            11: "rompt",
            12: "Other",
            13: " ",
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encode_calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
            }
        )
        if text in self.vocab:
            token_ids = self.vocab[text]
        else:
            token_ids = [ord(char) % 13 + 3 for char in text]
        if add_special_tokens:
            return [self.bos_token_id, *token_ids]
        return token_ids

    def decode(self, token_ids: int | list[int], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        self.decode_calls.append(list(token_ids))
        pieces = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            pieces.append(self.id_to_text.get(token_id, str(token_id)))
        return "".join(pieces)


class FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, tokenizer_path: str, **kwargs: Any) -> FakeTokenizer:
        del cls, kwargs
        assert tokenizer_path == "fake-tokenizer"
        tokenizer = FakeTokenizer()
        tokenizer.model_max_length = _TRANSFORMERS_UNSET_MODEL_MAX_LENGTH
        return tokenizer


class MissingSpecialTokenAutoTokenizer:
    @classmethod
    def from_pretrained(cls, tokenizer_path: str, **kwargs: Any) -> FakeTokenizer:
        del cls
        tokenizer = FakeAutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = None
        return tokenizer


class FakeTensorRows:
    def __init__(self, rows: list[list[int]] | list[list[float]]) -> None:
        self._rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def __getitem__(self, idx: int) -> Any:
        return SimpleNamespace(tolist=lambda: self._rows[idx])

    def tolist(self) -> list[list[int]] | list[list[float]]:
        return self._rows


class FakeGenerationModule:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.checkpoint_kwargs: dict[str, Any] = {}
        self.prepare_calls: list[tuple[int, int]] = []
        self.cache_allocated = False
        self.free_calls = 0

    @classmethod
    def from_checkpoint(cls, **kwargs: Any) -> FakeGenerationModule:
        module = cls()
        module.checkpoint_kwargs = kwargs
        return module

    def generate_batch(self, **kwargs: Any):
        self.generate_calls.append(kwargs)
        batch_size = kwargs["input_ids"].shape[0]
        if kwargs["use_cache"]:
            self.prepare_inference_cache(batch_size, kwargs["max_length"])

        completion_rows = [[5, 4, 0] if idx % 2 == 0 else [5, 6, 0] for idx in range(batch_size)]
        generated_rows = (
            completion_rows
            if kwargs["completions_only"]
            else [
                [*kwargs["input_ids"][idx].tolist(), *completion_rows[idx]]
                for idx in range(batch_size)
            ]
        )
        logprob_rows = [
            [-0.1, -0.2, -9.0] if idx % 2 == 0 else [-0.3, -0.4, -9.0] for idx in range(batch_size)
        ]
        return FakeTensorRows(generated_rows), None, FakeTensorRows(logprob_rows)

    def prepare_inference_cache(self, batch_size: int, max_seq_len: int) -> None:
        self.prepare_calls.append((batch_size, max_seq_len))
        self.cache_allocated = True

    def free_inference_cache(self) -> None:
        self.cache_allocated = False
        self.free_calls += 1


def _write_raw_checkpoint(
    checkpoint_dir: Path,
    *,
    model: dict[str, Any] | None = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text(
        json.dumps(
            {
                "model": model if model is not None else {"d_model": 8},
                "dataset": {
                    "tokenizer": {
                        "identifier": "fake-tokenizer",
                        "vocab_size": 16,
                        "pad_token_id": 0,
                        "eos_token_id": 2,
                    }
                },
            }
        )
    )
    (checkpoint_dir / ".metadata").write_text("fake")


def _metadata_reader(path: str | Path) -> SimpleNamespace:
    path_obj = path if isinstance(path, Path) else Path(path)
    if (path_obj / ".metadata").exists():
        return SimpleNamespace(state_dict_metadata={"model.transformer.wte.weight": object()})
    raise FileNotFoundError(path)


def _fake_olmo_core_imports(
    *,
    cuda_available: bool = False,
    auto_tokenizer: type[Any] = FakeAutoTokenizer,
) -> SimpleNamespace:
    return SimpleNamespace(
        AutoTokenizer=auto_tokenizer,
        AttentionBackendName=str,
        GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        TokenizerConfig=FakeTokenizerConfig,
        TransformerGenerationModule=FakeGenerationModule,
        cached_path=None,
        get_checkpoint_metadata=_metadata_reader,
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_capability=lambda: (9, 0),
                is_available=lambda: cuda_available,
            ),
            device=lambda device: device,
        ),
    )


@pytest.fixture
def fake_provider() -> tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer]:
    provider = OlmoCoreProvider.__new__(OlmoCoreProvider)
    tokenizer = FakeTokenizer()
    module = FakeGenerationModule()
    provider.model_name = "fake-model"
    provider.tokenizer = tokenizer
    provider.generation_module = module
    provider.pad_token_id = tokenizer.pad_token_id
    provider.eos_token_id = tokenizer.eos_token_id
    provider.use_cache = True
    provider.add_bos_token = False
    provider.batch_size = None
    provider.chat_template = None
    provider.max_length = 32

    def left_pad(sequences: list[list[int]]) -> tuple[FakeTensorRows, FakeTensorRows]:
        max_len = max(max((len(seq) for seq in sequences), default=0), 1)
        rows = []
        masks = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            rows.append([tokenizer.pad_token_id] * pad_len + seq)
            masks.append([0] * pad_len + [1] * len(seq))
        return FakeTensorRows(rows), FakeTensorRows(masks)

    provider._left_pad = left_pad
    return provider, module, tokenizer


def test_provider_loads_checkpoint_with_olmes_defaults(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "step1000"
    _write_raw_checkpoint(checkpoint_dir, model={"d_model": 8, "max_sequence_length": 4096})
    monkeypatch.setattr(
        olmo_core_utils,
        "_import_olmo_core",
        lambda: _fake_olmo_core_imports(auto_tokenizer=MissingSpecialTokenAutoTokenizer),
    )

    provider = OlmoCoreProvider(str(checkpoint_dir))

    checkpoint_kwargs = provider.generation_module.checkpoint_kwargs
    generation_config = checkpoint_kwargs["generation_config"]
    assert provider.max_length == 4096
    assert provider.add_bos_token is False
    assert provider.pad_token_id == 0
    assert provider.eos_token_id == 2
    assert provider.tokenizer.pad_token_id == 0
    assert provider.tokenizer.eos_token_id == 2
    assert checkpoint_kwargs["dtype"] == "bfloat16"
    assert "attention_backend" not in checkpoint_kwargs
    assert generation_config.pad_token_id == 0
    assert generation_config.eos_token_id == 2
    assert generation_config.use_cache is True


def test_provider_passes_explicit_attention_backend(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "step1000"
    _write_raw_checkpoint(checkpoint_dir, model={"d_model": 8, "max_sequence_length": 4096})
    monkeypatch.setattr(
        olmo_core_utils,
        "_import_olmo_core",
        lambda: _fake_olmo_core_imports(cuda_available=True),
    )

    provider = OlmoCoreProvider(
        str(checkpoint_dir),
        attention_backend="torch",
    )

    assert provider.generation_module.checkpoint_kwargs["attention_backend"] == "torch"


def test_generate_uses_olmes_batch_contract(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, module, _ = fake_provider

    outputs = provider.generate(
        [
            LMRequest(request_type=RequestType.COMPLETION, prompt="Prompt"),
            LMRequest(request_type=RequestType.COMPLETION, prompt="Other"),
        ],
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
    assert call["return_logprobs"] is True
    assert call["completions_only"] is False
    assert call["max_length"] == 5
    assert "max_new_tokens" not in call
    assert "stop_token_ids" not in call
    assert module.prepare_calls == [(4, 5)]
    assert [output.text for output in outputs[0]] == ["hello", "hello"]
    assert outputs[0][0].metadata["sum_logits"] == pytest.approx(-0.3)
    assert outputs[1][1].metadata["num_tokens"] == 2
    assert module.cache_allocated is False
    assert module.free_calls == 1


def test_generate_left_truncates_to_leave_completion_room(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, module, _ = fake_provider
    provider.max_length = 4

    provider.generate(
        [LMRequest(request_type=RequestType.COMPLETION, prompt="Prompt")],
        SamplingParams(max_tokens=3),
    )

    call = module.generate_calls[0]
    assert call["input_ids"].tolist() == [[11]]
    assert call["attention_mask"].tolist() == [[1]]
    assert call["max_length"] == 4

    module.generate_calls.clear()
    provider.max_length = 3
    with pytest.raises(ValueError, match=r"max_tokens \(3\) is greater than or equal"):
        provider.generate(
            [LMRequest(request_type=RequestType.COMPLETION, prompt="Prompt")],
            SamplingParams(max_tokens=3),
        )
    assert module.generate_calls == []


def test_generation_encoding_uses_provider_bos_flag(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, _, tokenizer = fake_provider
    tokenizer.add_bos_token = True

    assert provider._encode_prompt("Prompt") == [10, 11]
    assert tokenizer.encode_calls[-1] == {
        "text": "Prompt",
        "add_special_tokens": False,
    }

    provider.add_bos_token = True
    assert provider._encode_prompt("Prompt") == [1, 10, 11]
    assert tokenizer.encode_calls[-1] == {
        "text": "Prompt",
        "add_special_tokens": False,
    }


def test_logprob_encoding_uses_provider_bos_flag(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, _, tokenizer = fake_provider
    tokenizer.add_bos_token = True

    rows = provider._logprob_inputs_for_request(
        LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="Prompt",
            continuations=("!",),
        )
    )
    assert rows[0].input_ids == [10, 11]
    assert rows[0].continuation_token_ids == [6]

    provider.add_bos_token = True
    rows = provider._logprob_inputs_for_request(
        LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="Prompt",
            continuations=("!",),
        )
    )
    assert rows[0].input_ids == [1, 10, 11]
    assert rows[0].continuation_token_ids == [6]


def test_logprob_encoding_adds_prefix_when_context_tokenizes_empty(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, _, _ = fake_provider

    rows = provider._logprob_inputs_for_request(
        LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=" ",
            continuations=("!",),
        )
    )

    assert rows[0].input_ids == [1, 13]
    assert rows[0].continuation_token_ids == [13, 6]
    assert rows[0].input_length == 2
    assert rows[0].num_tokens_all == 3


def test_stop_text_postprocessing_matches_olmes(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, _, tokenizer = fake_provider

    token_ids = [7, 7, 8, 9, 7]
    token_logprobs = [-0.01] * len(token_ids)
    normalized_ids, normalized_logprobs, text = provider._normalize_generation_output(
        token_ids,
        token_logprobs,
        ("STOP",),
    )
    assert normalized_ids == token_ids
    assert normalized_logprobs == token_logprobs
    assert text == "xx"

    tokenizer.id_to_text[13] = tokenizer.decode(
        [tokenizer.eos_token_id],
        skip_special_tokens=False,
    )
    normalized_ids, normalized_logprobs, text = provider._normalize_generation_output(
        [5, 13, 7],
        [-0.1, -0.2, -0.3],
        provider._stop_sequences_with_eos(None),
    )
    assert normalized_ids == [5, 13, 7]
    assert normalized_logprobs == [-0.1, -0.2, -0.3]
    assert text == "hello"


def test_logprobs_clears_generation_cache_before_forward(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, module, _ = fake_provider
    module.cache_allocated = True
    cache_states: list[bool] = []

    def logprobs_chunk(requests: list[LMRequest]) -> list[list[LMOutput]]:
        cache_states.append(module.cache_allocated)
        return [[] for _ in requests]

    provider._logprobs_chunk = logprobs_chunk
    provider.logprobs(
        [
            LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt="Prompt",
                continuations=("!",),
            )
        ]
    )

    assert cache_states == [False]
    assert module.cache_allocated is False
    assert module.free_calls == 1
