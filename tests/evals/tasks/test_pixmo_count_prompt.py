"""The PixMo-Count question follows the checkpoint's prompt family, like the ``_mp`` pointing tasks.

Stage-1 checkpoints train on ``point_count: <label>``, so ``-o prompt_templates=none`` must send
exactly that; the default (instruction-tuned checkpoints) keeps the templated question that
reproduces the released Molmo2-4B prompts (pinned by ``test_image_qa_dump_parity``).
"""

from __future__ import annotations

import pytest

from olmo_eval.common.image_qa import pixmo_count_question
from olmo_eval.evals.tasks.common.registry import get_task

STAGE1 = {"prompt_templates": "none", "system_prompt_style": "style_and_length_v2"}


def test_stage1_family_sends_the_bare_label():
    task = get_task("pixmo_count", STAGE1)
    assert task._question("Cows", 0) == "cows"
    assert task.apply_family_prefix(task._question("Cows", 0)) == "point_count: cows"


@pytest.mark.parametrize("idx", [0, 1, 7, 539])
def test_default_family_is_the_released_templated_question(idx: int):
    task = get_task("pixmo_count")
    assert task._question("cows", idx) == pixmo_count_question("cows", idx)
    templated = pixmo_count_question("cows", idx)
    assert task.apply_family_prefix(task._question("cows", idx)) == templated


def test_bare_label_does_not_depend_on_the_example_index():
    """No template is drawn under ``none``, so nothing varies with the arrow position."""
    task = get_task("pixmo_count", STAGE1)
    assert {task._question("cows", i) for i in range(50)} == {"cows"}


def test_test_split_variant_follows_the_family_too():
    assert get_task("pixmo_count:test", STAGE1)._question("cows", 3) == "cows"


def test_unknown_family_is_refused():
    with pytest.raises(ValueError, match="prompt_templates"):
        get_task("pixmo_count", {"prompt_templates": "nope"})._question("cows", 0)
