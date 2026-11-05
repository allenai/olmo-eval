"""RULER's `:v2` variant."""

import pytest

from olmo_eval.common.types import CONTEXT_CUT_TOKENS_KEY, ContextFit, Instance, LMOutput, Response
from olmo_eval.evals.suites.registry import get_suite
from olmo_eval.evals.tasks.common.registry import get_task
from olmo_eval.runners.processing.utils import get_metric_metadata, get_primary_metric


def _ruler_doc(context: str = "HAYSTACK WITH THE NEEDLE") -> dict:
    return {
        "input": f"Read the text.\n{context}\nWhat is the needle? Answer:",
        "context": context,
        "outputs": ["needle"],
        "index": 0,
    }


def _scored_text(spec: str, text: str, cut_tokens: int = 0) -> str:
    task = get_task(spec)
    instance = Instance(question="q", gold_answer=["needle"], metadata={})
    metadata = {CONTEXT_CUT_TOKENS_KEY: cut_tokens} if cut_tokens else {}
    response = Response(
        instance=instance,
        request=task.format_request(instance),
        outputs=[LMOutput(text=text, metadata=metadata)],
    )
    task.strip_thinking_traces([response])
    task.strip_thinking_traces([response])  # the runner may prepare a response twice
    return response.outputs[0].text


@pytest.mark.parametrize("spec", ["ruler_niah_s_1__4096:v2", "ruler_niah_s_1__4096:chat:v2"])
def test_v2_marks_only_the_haystack_for_the_window_cut(spec):
    task = get_task(spec)
    request = task.format_request(task.process_doc(_ruler_doc()))

    assert request.context_fit == ContextFit(text="HAYSTACK WITH THE NEEDLE")
    assert "What is the needle?" in (request.prompt or str(request.messages))


def test_the_original_spec_is_unchanged():
    task = get_task("ruler_niah_s_1__4096")

    assert task.format_request(task.process_doc(_ruler_doc())).context_fit is None
    assert task.config.task_settings is None
    assert _scored_text("ruler_vt__4096", "a Question: needle", cut_tokens=12) == (
        "a Question: needle"
    )


def test_v2_raises_only_the_vt_budget():
    assert get_task("ruler_vt__4096").config.sampling_params.max_tokens == 50
    assert get_task("ruler_vt__4096:v2").config.sampling_params.max_tokens == 100
    assert get_task("ruler_niah_s_1__4096:v2").config.sampling_params.max_tokens == 50


@pytest.mark.parametrize(
    ("spec", "text", "scored"),
    [
        ("ruler_vt__4096:v2", " A, B.\n\nQuestion: What is VAR C? Answer: 1", " A, B.\n\n"),
        ("ruler_vt__4096:v2", "\nVAR A\n \nVAR B", "\nVAR A\n \nVAR B"),
        ("ruler_niah_s_1__4096:v2", "a Question: b", "a Question: b"),
    ],
)
def test_only_vt_v2_is_scored_up_to_the_first_question(spec, text, scored):
    assert _scored_text(spec, text) == scored


@pytest.mark.parametrize("spec", ["ruler_niah_s_1__4096:v2", "ruler_vt__4096:v2"])
def test_v2_scores_an_overflowing_instance_zero(spec):
    assert _scored_text(spec, "the needle", cut_tokens=12) == ""
    assert _scored_text(spec, "the needle") == "the needle"


def test_the_v2_suite_reports_recall_over_every_task_as_v2():
    task = get_task("ruler_qa_1__4096:v2")
    preferred = get_metric_metadata(task)

    assert get_primary_metric({"recall": {"substring_recall": 0.75}}, preferred)[1] == 0.75
    assert get_suite("ruler_all_v2__4096").tasks == tuple(
        f"{t}:v2" for t in get_suite("ruler_all__4096").tasks
    )
