"""Tests for persisted executed request rows."""

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.runners.io.builders import build_requests_from_responses


def test_build_requests_from_responses_uses_executed_request_trace():
    """Executed request rows should preserve full continuations and provider trace."""
    response = Response(
        instance=Instance(
            question="Pick the right answer",
            choices=("A", "B"),
            metadata={"id": "doc-1", "gold_idx": 1},
        ),
        request=LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="Question: Pick one",
            continuations=(" A", " B"),
            continuation_prompts=("Question: Pick one", "Question: Pick one"),
            max_length=256,
        ),
        outputs=[LMOutput(text=" A"), LMOutput(text=" B")],
        request_trace={
            "provider": "VLLMServerProvider",
            "endpoint": "/completions",
            "generation_kwargs": {
                "max_gen_toks": 1,
                "do_sample": False,
                "temperature": 1.0,
                "prompt_logprobs": 1,
                "add_special_tokens": False,
            },
            "stop_sequences": [],
            "input_mode": "prompt_token_ids",
        },
    )

    rows = build_requests_from_responses([response], "basic_skills:rc:olmo3base")
    assert len(rows) == 1

    row = rows[0]
    assert row["request_type"] == "loglikelihood"
    assert row["request"]["continuation"] == " A"
    assert row["request"]["continuations"] == [" A", " B"]
    assert row["request"]["continuation_prompts"] == ["Question: Pick one", "Question: Pick one"]
    assert row["request"]["max_length"] == 256
    assert row["request"]["generation_kwargs"] == {
        "max_gen_toks": 1,
        "do_sample": False,
        "temperature": 1.0,
        "prompt_logprobs": 1,
        "add_special_tokens": False,
    }
    assert row["request"]["stop_sequences"] == []
    assert row["request"]["provider_request"]["input_mode"] == "prompt_token_ids"
