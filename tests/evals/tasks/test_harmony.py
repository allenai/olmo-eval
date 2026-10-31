"""Opt-in Harmony parsing is part of the stored task identity."""

from dataclasses import replace

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.common.types.base import compute_task_hash
from olmo_eval.evals.tasks.common import get_task


def test_harmony_setting_changes_only_opted_in_identity() -> None:
    task = get_task("ifeval")
    old = task.config.to_dict()
    parsed = get_task("ifeval", {"response_parser": "harmony"}).config.to_dict()
    assert "response_parser" not in old
    assert parsed.pop("response_parser") == "harmony"
    assert parsed == old
    assert compute_task_hash(replace(task.config, response_parser="harmony").to_dict()) != (
        compute_task_hash(old)
    )


def test_unknown_response_parser_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported response_parser"):
        get_task("ifeval", {"response_parser": "guess"})


@pytest.mark.parametrize("parser", [None, "harmony"])
def test_harmony_is_opt_in_and_idempotent(parser: str | None) -> None:
    task = get_task("ifeval", {"response_parser": parser})
    text = "<|meta_sep|>analysis<|im_sep|>wrong<|im_end|>"
    text += "<|im_start|>assistant<|meta_sep|>final<|im_sep|>\n  good <|fim_suffix|>"
    output = LMOutput(text=text)
    response = Response(
        instance=Instance(question="Q"),
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[output],
    )
    task.strip_thinking_traces([response])
    task.strip_thinking_traces([response])
    assert output.text == ("\n  good " if parser else text)
    if parser:
        assert output.metadata["original_text"] == text
    else:
        assert "original_text" not in output.metadata


def test_harmony_retention_preserves_logprob_byte_normalization() -> None:
    import math

    from olmo_eval.common.scorers import BitsPerByteScorer

    task = get_task("ifeval", {"response_parser": "harmony"})
    decoded = "analysiswrongassistantfinalgood"
    raw = (
        "<|channel|>analysis<|message|>wrong<|end|>"
        "<|start|>assistant<|channel|>final<|message|>good<|return|>"
    )
    output = LMOutput(
        text=decoded,
        logprobs=[{"token": "good", "logprob": -1.0}],
        metadata={"text_with_special_tokens": raw},
    )
    instance = Instance(question="Q")
    response = Response(instance=instance, request=task.format_request(instance), outputs=[output])
    task.strip_thinking_traces([response])
    assert output.text == "good"
    assert BitsPerByteScorer().score(instance, output) == 1 / (len(decoded) * math.log(2))


@pytest.mark.parametrize("text", ["already parsed answer", "analysiswrongassistantfinalgood"])
def test_harmony_noop_is_visible_in_predictions(text: str) -> None:
    from olmo_eval.runners.io.builders import build_predictions

    task = get_task("ifeval", {"response_parser": "harmony"})
    output = LMOutput(text=text)
    instance = Instance(question="Q")
    response = Response(instance=instance, request=task.format_request(instance), outputs=[output])
    task.strip_thinking_traces([response])
    row = build_predictions([response])[0]["model_output"][0]
    assert row["text"] == text
    assert row["response_parser_found"] is False


def test_unrecognized_raw_decode_does_not_replace_provider_text() -> None:
    from olmo_eval.runners.io.builders import build_predictions

    task = get_task("ifeval", {"response_parser": "harmony"})
    decoded = "userplain"
    raw = "<|im_start|>user<|im_sep|>plain<|im_end|>"
    output = LMOutput(text=decoded, metadata={"text_with_special_tokens": raw})
    instance = Instance(question="Q")
    response = Response(instance=instance, request=task.format_request(instance), outputs=[output])
    task.strip_thinking_traces([response])
    row = build_predictions([response])[0]["model_output"][0]
    assert row["text"] == decoded
    assert row["response_parser_found"] is False
    assert row["text_with_special_tokens"] == raw
    assert "original_text" not in row
