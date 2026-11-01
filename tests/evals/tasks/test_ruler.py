"""RULER's `:v2` variant: the haystack is cut to the window, and VT's budget is raised."""

from olmo_eval.common.types import ContextFit
from olmo_eval.evals.suites.registry import get_suite
from olmo_eval.evals.tasks.common.registry import get_task


def _ruler_doc(context: str = "HAYSTACK WITH THE NEEDLE") -> dict:
    return {
        "input": f"Read the text.\n{context}\nWhat is the needle? Answer:",
        "context": context,
        "outputs": ["needle"],
        "index": 0,
    }


def test_v2_marks_the_haystack_for_a_window_only_cut():
    """The haystack is the ContextFit text, with no prompt budget: the provider cuts
    its end only as far as the model's window requires, never the question."""
    task = get_task("ruler_niah_s_1__4096:v2")
    request = task.format_request(task.process_doc(_ruler_doc()))

    assert request.context_fit == ContextFit(text="HAYSTACK WITH THE NEEDLE")
    assert request.context_fit.max_prompt_tokens is None
    before, after = request.prompt.split("HAYSTACK WITH THE NEEDLE")
    assert before == "Read the text.\n"
    assert after.startswith("\nWhat is the needle?")
    assert task.config.task_settings == {"context_cut": "model_window"}


def test_the_original_spec_is_unchanged():
    task = get_task("ruler_niah_s_1__4096")
    request = task.format_request(task.process_doc(_ruler_doc()))

    assert request.context_fit is None
    assert task.config.task_settings is None


def test_chat_and_v2_compose():
    task = get_task("ruler_niah_s_1__4096:chat:v2")
    request = task.format_request(task.process_doc(_ruler_doc()))

    assert request.context_fit is not None
    assert request.context_fit.text == "HAYSTACK WITH THE NEEDLE"


def test_a_haystack_that_does_not_occur_exactly_once_is_not_marked():
    task = get_task("ruler_niah_s_1__4096:v2")
    request = task.format_request(task.process_doc(_ruler_doc("x")))  # "x" is also in "text"

    assert request.context_fit is None


def test_v2_raises_only_the_vt_budget():
    """An answer that restates the chain needs about 60 tokens for five names."""
    vt, vt_v2 = get_task("ruler_vt__4096"), get_task("ruler_vt__4096:v2")

    assert vt.config.sampling_params.max_tokens == 50
    assert vt_v2.config.sampling_params.max_tokens == 100
    assert vt_v2.config.sampling_params.temperature == vt.config.sampling_params.temperature
    assert get_task("ruler_cwe__4096:v2").config.sampling_params.max_tokens == 100
    assert get_task("ruler_niah_s_1__4096:v2").config.sampling_params.max_tokens == 50


def test_v2_suites_run_every_task_as_v2():
    suite = get_suite("ruler_all_v2__4096")
    original = get_suite("ruler_all__4096")

    assert suite.tasks == tuple(f"{t}:v2" for t in original.tasks)
    assert len(suite.tasks) == 13
    assert get_suite("ruler_all_chat_v2__4096").tasks == tuple(
        f"{t}:chat:v2" for t in original.tasks
    )
