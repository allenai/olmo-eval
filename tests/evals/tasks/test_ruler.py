"""RULER's haystack cut: the haystack is marked so a provider cuts it to the window."""

from olmo_eval.evals.tasks.common.registry import get_task


def _ruler_doc(context: str = "HAYSTACK WITH THE NEEDLE") -> dict:
    return {
        "input": f"Read the text.\n{context}\nWhat is the needle? Answer:",
        "context": context,
        "outputs": ["needle"],
        "index": 0,
    }


def test_plain_ruler_marks_the_haystack_for_a_window_only_cut():
    """The haystack is the ContextFit text, with no prompt budget: the provider cuts
    its end only as far as the model's window requires, never the question."""
    from olmo_eval.common.types import ContextFit

    task = get_task("ruler_niah_s_1__4096")
    instance = task.process_doc(_ruler_doc())
    request = task.format_request(instance)

    assert request.context_fit == ContextFit(text="HAYSTACK WITH THE NEEDLE")
    assert request.context_fit.max_prompt_tokens is None
    before, after = request.prompt.split("HAYSTACK WITH THE NEEDLE")
    assert before == "Read the text.\n"
    assert after.startswith("\nWhat is the needle?")
    assert task.task_settings == {"context_cut": "model_window"}


def test_the_chat_variant_carries_the_same_cut():
    task = get_task("ruler_niah_s_1__4096:chat")
    request = task.format_request(task.process_doc(_ruler_doc()))

    assert request.context_fit is not None
    assert request.context_fit.text == "HAYSTACK WITH THE NEEDLE"


def test_a_haystack_that_does_not_occur_exactly_once_is_not_marked():
    task = get_task("ruler_niah_s_1__4096")
    doc = _ruler_doc("x")  # "x" also occurs in "text"
    request = task.format_request(task.process_doc(doc))

    assert request.context_fit is None


def test_tasks_built_on_ruler_do_not_cut_unless_they_ask():
    from olmo_eval.evals.tasks.ruler import RulerTask

    assert RulerTask.cut_haystack is False
