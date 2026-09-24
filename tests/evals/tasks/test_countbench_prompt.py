"""The CountBenchQA question follows the checkpoint's prompt family, like ``pixmo_count``.

Stage-1 checkpoints train on ``point_count: <object name>``. CountBenchQA ships no name, but
nearly every question is "How many <objects> are there in the image?", so ``-o
prompt_templates=none`` sends the name taken from it. The default (instruction-tuned
checkpoints) keeps the question verbatim, as pinned by ``test_image_qa_dump_parity``.
"""

from __future__ import annotations

import pytest

from olmo_eval.evals.tasks.common.registry import get_task
from olmo_eval.evals.tasks.countbench_qa import countbench_label

STAGE1 = {"prompt_templates": "none", "system_prompt_style": "style_and_length_v2"}
QUESTION = "How many headsets are there in the image?"


@pytest.mark.parametrize(
    "question, label",
    [
        (QUESTION, "headsets"),
        ("How many light bulbs are there in the image?", "light bulbs"),
        ("How many CDs are there in the image?", "cds"),
        ("How many people in the foreground are there in the image?", "people in the foreground"),
        ("How many petals does each flower have in this image?", None),
        ("How many stories does this cottage have?", None),
    ],
)
def test_countbench_label(question: str, label: str | None):
    assert countbench_label(question) == label


def test_stage1_family_sends_the_object_name():
    task = get_task("countbench_qa", STAGE1)
    assert task.apply_family_prefix(task._question(QUESTION)) == "point_count: headsets"


def test_stage1_family_keeps_a_question_with_no_name():
    task = get_task("countbench_qa", STAGE1)
    other = "How many stories does this cottage have?"
    assert task.apply_family_prefix(task._question(other)) == f"point_count: {other}"


def test_default_family_sends_the_question_verbatim():
    task = get_task("countbench_qa")
    assert task.apply_family_prefix(task._question(QUESTION)) == QUESTION


def test_unknown_family_is_refused():
    with pytest.raises(ValueError, match="prompt_templates"):
        get_task("countbench_qa", {"prompt_templates": "nope"})._question(QUESTION)
