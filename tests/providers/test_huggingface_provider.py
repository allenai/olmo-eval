"""Unit tests for HuggingFaceProvider batching behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams


class FakeBatchEncoding(dict):
    """Minimal BatchEncoding replacement for tests."""

    def to(self, device):
        return {key: value.to(device) for key, value in self.items()}


class FakeTokenizer:
    """Simple tokenizer stub with configurable padding side."""

    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.eos_token_id = 99
        self.padding_side = "right"
        self.prompt_tokens = {
            "short": [11, 12],
            "longer": [21, 22, 23],
        }
        self.id_to_text = {
            31: "A",
            32: "B",
            41: "C",
            42: "D",
            51: "E",
            52: "F",
            61: "G",
            62: "H",
        }

    def __call__(self, prompts, return_tensors="pt", padding=True):
        if isinstance(prompts, str):
            prompts = [prompts]

        rows = [list(self.prompt_tokens[prompt]) for prompt in prompts]
        max_len = max(len(row) for row in rows)

        input_ids = []
        attention_mask = []
        for row in rows:
            pad_len = max_len - len(row)
            if self.padding_side == "left":
                input_ids.append(([self.pad_token_id] * pad_len) + row)
                attention_mask.append(([0] * pad_len) + ([1] * len(row)))
            else:
                input_ids.append(row + ([self.pad_token_id] * pad_len))
                attention_mask.append(([1] * len(row)) + ([0] * pad_len))

        return FakeBatchEncoding(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )

    def decode(self, token_or_tokens, skip_special_tokens=True):
        if isinstance(token_or_tokens, torch.Tensor):
            if token_or_tokens.dim() == 0:
                token_ids = [int(token_or_tokens.item())]
            else:
                token_ids = [int(token_id) for token_id in token_or_tokens.tolist()]
        elif isinstance(token_or_tokens, (list, tuple)):
            token_ids = [int(token_id) for token_id in token_or_tokens]
        else:
            token_ids = [int(token_or_tokens)]

        pieces = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            pieces.append(self.id_to_text.get(token_id, f"tok{token_id}"))
        return "".join(pieces)


class FakeGenerateModel:
    """Model stub for exercising batched generate plus batched scoring paths."""

    def __init__(self) -> None:
        self.generate_calls = []
        self.forward_calls = []
        self._call_idx = 0
        self._forward_idx = 0
        self._sequences = [
            torch.tensor(
                [
                    [0, 11, 12, 31, 32],
                    [21, 22, 23, 41, 42],
                ],
                dtype=torch.long,
            ),
            torch.tensor(
                [
                    [0, 11, 12, 51, 52],
                    [21, 22, 23, 61, 62],
                ],
                dtype=torch.long,
            ),
        ]
        self._forward_logits = [
            self._make_logits(
                targets=[
                    [(1, 31, 2.0, 1), (2, 32, 1.5, 2)],
                    [(2, 41, 1.8, 3), (3, 42, 1.2, 4)],
                ]
            ),
            self._make_logits(
                targets=[
                    [(1, 51, 1.7, 5), (2, 52, 1.1, 6)],
                    [(2, 61, 1.9, 7), (3, 62, 1.4, 8)],
                ]
            ),
        ]

    def _make_logits(self, targets):
        logits = torch.full((2, 5, 80), -100.0, dtype=torch.float32)
        for row_idx, row_targets in enumerate(targets):
            for position, token_id, target_logit, competitor_id in row_targets:
                logits[row_idx, position, token_id] = target_logit
                logits[row_idx, position, competitor_id] = target_logit - 1.0
        return logits

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        call_idx = self._call_idx
        self._call_idx += 1
        return SimpleNamespace(sequences=self._sequences[call_idx])

    def __call__(self, input_ids=None, attention_mask=None):
        self.forward_calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
            }
        )
        logits = self._forward_logits[self._forward_idx]
        self._forward_idx += 1
        return SimpleNamespace(logits=logits)


class FakeForwardModel:
    """Model stub for exercising batched logprob scoring."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.forward_calls = []

    def __call__(self, input_ids=None, attention_mask=None):
        self.forward_calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
            }
        )
        return SimpleNamespace(logits=self.logits)


def _make_provider(model):
    from olmo_eval.inference.providers.huggingface import HuggingFaceProvider

    with patch.object(HuggingFaceProvider, "__init__", lambda self, *a, **kw: None):
        provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
        provider.model_name = "test-model"
        provider.device = torch.device("cpu")
        provider.model = model
        provider.tokenizer = FakeTokenizer()
        return provider


class TestHuggingFaceProviderBatching:
    """Tests for batched Hugging Face inference."""

    def test_generate_batches_requests_per_sample(self):
        """Generation should batch prompts together instead of looping per request."""
        provider = _make_provider(FakeGenerateModel())
        requests = [
            LMRequest(request_type=RequestType.COMPLETION, prompt="short"),
            LMRequest(request_type=RequestType.COMPLETION, prompt="longer"),
        ]
        params = SamplingParams(max_tokens=2, num_samples=2)

        outputs = provider.generate(requests, sampling_params=params)

        assert len(provider.model.generate_calls) == 2
        assert len(provider.model.forward_calls) == 2

        first_call = provider.model.generate_calls[0]
        assert first_call["return_dict_in_generate"] is True
        assert first_call["input_ids"].tolist() == [
            [0, 11, 12],
            [21, 22, 23],
        ]
        assert first_call["attention_mask"].tolist() == [
            [0, 1, 1],
            [1, 1, 1],
        ]

        first_forward = provider.model.forward_calls[0]
        assert first_forward["input_ids"].tolist() == [
            [11, 12, 31, 32, 0],
            [21, 22, 23, 41, 42],
        ]
        assert first_forward["attention_mask"].tolist() == [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ]

        assert [[output.text for output in request_outputs] for request_outputs in outputs] == [
            ["AB", "EF"],
            ["CD", "GH"],
        ]
        first_log_probs = torch.log_softmax(provider.model._forward_logits[0], dim=-1)
        second_log_probs = torch.log_softmax(provider.model._forward_logits[1], dim=-1)
        expected_first = first_log_probs[0, 1, 31].item() + first_log_probs[0, 2, 32].item()
        expected_last = second_log_probs[1, 2, 61].item() + second_log_probs[1, 3, 62].item()
        assert outputs[0][0].metadata["sum_logits"] == pytest.approx(expected_first)
        assert outputs[1][1].metadata["sum_logits"] == pytest.approx(expected_last)
        assert [entry["token"] for entry in outputs[1][1].logprobs] == ["G", "H"]

    def test_logprobs_batches_all_continuations_in_one_forward(self):
        """Logprob scoring should flatten all continuations into one model forward pass."""
        logits = torch.full((3, 4, 64), -100.0, dtype=torch.float32)
        logits[0, 1, 12] = 2.0
        logits[0, 1, 1] = 1.0
        logits[0, 2, 13] = 1.5
        logits[0, 2, 2] = 0.5
        logits[1, 0, 21] = 3.0
        logits[1, 0, 0] = 1.0
        logits[2, 2, 33] = 0.7
        logits[2, 2, 3] = 0.2

        provider = _make_provider(FakeForwardModel(logits))
        requests = [
            LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt="prompt-1",
                continuations=("cont-1", "cont-2"),
            ),
            LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt="prompt-2",
                continuations=("cont-3",),
            ),
        ]

        with patch(
            "olmo_eval.inference.providers.huggingface.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.side_effect = [
                ([10, 11], [12, 13]),
                ([20], [21]),
                ([30, 31, 32], [33]),
            ]
            outputs = provider.logprobs(requests)

        assert len(provider.model.forward_calls) == 1

        call = provider.model.forward_calls[0]
        assert call["input_ids"].tolist() == [
            [10, 11, 12, 13],
            [20, 21, 0, 0],
            [30, 31, 32, 33],
        ]
        assert call["attention_mask"].tolist() == [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ]

        log_probs = torch.log_softmax(logits, dim=-1)
        expected_first = log_probs[0, 1, 12].item() + log_probs[0, 2, 13].item()
        expected_second = log_probs[1, 0, 21].item()
        expected_third = log_probs[2, 2, 33].item()

        assert [[output.text for output in request_outputs] for request_outputs in outputs] == [
            ["cont-1", "cont-2"],
            ["cont-3"],
        ]
        assert outputs[0][0].metadata["total_logprob"] == pytest.approx(expected_first)
        assert outputs[0][1].metadata["total_logprob"] == pytest.approx(expected_second)
        assert outputs[1][0].metadata["total_logprob"] == pytest.approx(expected_third)
