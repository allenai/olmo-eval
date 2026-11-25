"""RULER's `:v2` variant."""

import re

import pytest

from olmo_eval.common.types import CONTEXT_CUT_TOKENS_KEY, ContextFit, Instance, LMOutput, Response
from olmo_eval.evals.suites.registry import get_suite
from olmo_eval.evals.tasks.common.registry import get_task
from olmo_eval.evals.tasks.ruler import randomize_vt_example
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


@pytest.mark.parametrize(
    ("spec", "text", "cut_tokens", "scored"),
    [
        (
            "ruler_vt__4096:v2",
            " VAR A, VAR B.\nQuestion: What is VAR C? Answer: 1",
            0,
            " VAR A, VAR B.\n",
        ),
        ("ruler_niah_s_1__4096:v2", "the needle", 12, ""),
    ],
)
def test_v2_keeps_the_original_text_of_an_output_it_changes(spec, text, cut_tokens, scored):
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

    assert response.outputs[0].text == scored
    assert response.outputs[0].metadata["original_text"] == text


def test_v2_leaves_an_unchanged_output_without_original_text():
    task = get_task("ruler_vt__4096:v2")
    instance = Instance(question="q", gold_answer=["needle"], metadata={})
    response = Response(
        instance=instance, request=task.format_request(instance), outputs=[LMOutput(text=" VAR A.")]
    )

    task.strip_thinking_traces([response])

    assert "original_text" not in response.outputs[0].metadata


_EXAMPLE = (
    "Track it. VAR QPE = 64886 x\n VAR SEJ = VAR QPE  x\n VAR ZQO = VAR SEJ  x\n"
    " Question: Find all variables that are assigned the value 64886 in the text Answer: "
    "5 variables are assgined the value 64886, they are:  QPE SEJ ZQO\n\n"
)


def _vt_doc(context: str = "HAYSTACK VAR AAAAA = 1", index: int = 0) -> dict:
    return {
        **_ruler_doc(context),
        "example": _EXAMPLE,
        "outputs": ["AAAAA"],
        "query": "1",
        "index": index,
    }


def test_vt_v3_puts_a_randomized_worked_example_before_the_input():
    task = get_task("ruler_vt__4096:v3")
    doc = _vt_doc()

    instance = task.process_doc(doc)

    example, sep, rest = instance.question.partition("\n\n")
    assert sep and rest == doc["input"]
    assert example == randomize_vt_example(_EXAMPLE.strip(), 0, ["AAAAA"], "1")
    request = task.format_request(instance)
    assert request.prompt == instance.question
    assert request.context_fit == ContextFit(text=doc["context"])


def test_the_example_keeps_its_chain_with_new_names_and_value():
    out = randomize_vt_example(_EXAMPLE.strip(), seed=3)

    names = out.rpartition("they are:")[2].split()
    value = re.search(r"VAR ([A-Z]+) = (\d+)", out)
    assert len(names) == 3 and len(set(names)) == 3
    assert all(len(n) == 3 and n.isupper() for n in names)
    assert not {"QPE", "SEJ", "ZQO"} & set(re.findall(r"[A-Z]{3}", out.replace("VAR", "")))
    assert value.group(1) == names[0] and value.group(2) != "64886"
    assert out.count(value.group(2)) == 3  # the assignment, the question and the answer
    assert f"VAR {names[1]} = VAR {names[0]}" in out and f"VAR {names[2]} = VAR {names[1]}" in out
    assert out.replace(value.group(2), "#") == re.sub(
        r"\b(QPE|SEJ|ZQO)\b",
        lambda m: names[["QPE", "SEJ", "ZQO"].index(m.group(1))],
        _EXAMPLE.strip(),
    ).replace("64886", "#")


def test_the_example_differs_by_instance_and_is_deterministic():
    examples = [randomize_vt_example(_EXAMPLE.strip(), seed=i) for i in range(20)]

    assert len(set(examples)) == 20
    assert examples == [randomize_vt_example(_EXAMPLE.strip(), seed=i) for i in range(20)]


def test_the_example_never_takes_the_instances_answers_or_value():
    for seed in range(200):
        out = randomize_vt_example(_EXAMPLE.strip(), seed, avoid_names=["QPE"], avoid_value="64886")
        assert "ABC" not in out.rpartition("they are:")[2]
    out = randomize_vt_example("VAR A = 1 they are:  A", 0, avoid_names=[], avoid_value="")
    assert out != "VAR A = 1 they are:  A"
    assert randomize_vt_example("no answer here", 0) == "no answer here"


@pytest.mark.parametrize("spec", ["ruler_vt__4096", "ruler_vt__4096:v2"])
def test_earlier_vt_specs_keep_the_input_without_the_example(spec):
    doc = _vt_doc()

    assert get_task(spec).process_doc(doc).question == doc["input"]


def test_v3_is_v2_plus_the_example_and_exists_only_for_vt():
    v2, v3 = get_task("ruler_vt__4096:v2").config, get_task("ruler_vt__4096:v3").config

    assert v3.task_settings == {**v2.task_settings, "prepend_example": "random"}
    assert v3.sampling_params == v2.sampling_params
    assert _scored_text("ruler_vt__4096:v3", " A.\nQuestion: B") == " A.\n"
    assert _scored_text("ruler_vt__4096:v3", "VAR A", cut_tokens=3) == ""
    with pytest.raises(KeyError, match="Unknown variant 'v3'"):
        get_task("ruler_niah_s_1__4096:v3")


def test_v3_fixed_keeps_the_datas_example_for_every_instance():
    task = get_task("ruler_vt__4096:v3_fixed")

    for index in (0, 7):
        doc = _vt_doc(index=index)
        assert task.process_doc(doc).question == _EXAMPLE.strip() + "\n\n" + doc["input"]
    assert get_task("ruler_vt__4096:v3_fixed").config.task_settings == {
        **get_task("ruler_vt__4096:v2").config.task_settings,
        "prepend_example": "fixed",
    }
