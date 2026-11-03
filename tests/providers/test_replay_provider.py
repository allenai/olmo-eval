"""Tests for the replay provider."""

from pathlib import Path

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.errors import request_error
from olmo_eval.inference.providers import ProviderConfig
from olmo_eval.inference.providers.replay import ReplayProvider
from olmo_eval.runners.io.writers import write_predictions_jsonl, write_requests_jsonl

CHAT = ({"role": "user", "content": "What is 6 x 7?"},)


def _save(output_dir: Path, spec: str, saved: list[tuple[object, list[dict]]]) -> None:
    """Write a run's files for (context, model_output) pairs, as the runner does."""
    requests = [{"doc_id": i, "request": {"context": c}} for i, (c, _) in enumerate(saved)]
    predictions = [{"doc_id": i, "model_output": out} for i, (_, out) in enumerate(saved)]
    write_requests_jsonl(str(output_dir), spec, requests, "model")
    write_predictions_jsonl(str(output_dir), spec, predictions, "model")


def _completion(prompt: str) -> LMRequest:
    return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)


def _chat(messages: tuple[dict, ...]) -> LMRequest:
    return LMRequest(request_type=RequestType.CHAT, messages=messages)


def test_replays_saved_generations_by_prompt(tmp_path: Path) -> None:
    _save(
        tmp_path,
        "task",
        [
            ("Q: one", [{"text": "1", "finish_reason": "stop", "extracted_answer": "x"}]),
            (list(CHAT), [{"text": "42", "original_text": "<think>hm</think>42"}]),
        ],
    )
    provider = ReplayProvider(predictions=str(tmp_path))

    chat, completion = provider.generate([_chat(CHAT), _completion("Q: one")])

    # The raw generation comes back, so the task strips and extracts again.
    assert [o.text for o in chat] == ["<think>hm</think>42"]
    assert completion[0].text == "1"
    assert completion[0].metadata == {"finish_reason": "stop"}
    assert completion[0].extracted_answer is None


def test_reads_a_single_predictions_file(tmp_path: Path) -> None:
    _save(tmp_path, "a", [("Q: a", [{"text": "A"}])])
    _save(tmp_path, "b", [("Q: b", [{"text": "B"}])])
    path = next(tmp_path.rglob("a*-predictions.jsonl"))

    provider = ReplayProvider(predictions=str(path))

    assert request_error(provider.generate([_completion("Q: b")])[0]) is not None
    assert provider.generate([_completion("Q: a")])[0][0].text == "A"


def test_unsaved_prompt_and_sample_count_mismatch_fail_the_request(tmp_path: Path) -> None:
    _save(tmp_path, "task", [("Q: one", [{"text": "1"}, {"text": "2"}])])
    provider = ReplayProvider(predictions=str(tmp_path))

    unsaved, wrong_count = provider.generate(
        [_completion("Q: other"), _completion("Q: one")], SamplingParams(num_samples=1)
    )

    assert "no saved generation" in str(request_error(unsaved))
    assert "saved 2 samples" in str(request_error(wrong_count))
    two = provider.generate([_completion("Q: one")], SamplingParams(num_samples=2))
    assert [o.text for o in two[0]] == ["1", "2"]


def test_repeated_prompts_are_served_in_order(tmp_path: Path) -> None:
    _save(tmp_path, "task", [("Q", [{"text": "first"}]), ("Q", [{"text": "second"}])])
    provider = ReplayProvider(predictions=str(tmp_path))

    outputs = provider.generate([_completion("Q"), _completion("Q"), _completion("Q")])

    assert [o[0].text for o in outputs[:2]] == ["first", "second"]
    assert request_error(outputs[2]) is not None


def test_provider_config_passes_the_predictions_path(tmp_path: Path) -> None:
    _save(tmp_path, "task", [("Q", [{"text": "A"}])])
    config = ProviderConfig.from_dict(
        {"kind": "replay", "model": "replay", "predictions": str(tmp_path)}
    )

    provider = config.create_provider()

    assert isinstance(provider, ReplayProvider)
    assert not config.requires_gpu


def test_missing_predictions_are_reported(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provider.predictions"):
        ReplayProvider()
    with pytest.raises(FileNotFoundError):
        ReplayProvider(predictions=str(tmp_path))
