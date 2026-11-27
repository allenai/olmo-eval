"""Tests for resolving a task spec into its registered task and ordered variants."""

import pytest

import olmo_eval.evals.tasks  # noqa: F401  (registers every task)
from olmo_eval.evals.tasks.common import get_task, resolve_task_spec


def test_longest_registered_prefix_is_the_task() -> None:
    # The same segments in a different order pick different task classes.
    by_class = resolve_task_spec("mbpp:olmo3base:bpb")
    by_variant = resolve_task_spec("mbpp:bpb:olmo3base")

    assert by_class.task_name == "mbpp:olmo3base"
    assert by_class.variants == ("bpb",)
    assert by_variant.task_name == "mbpp:bpb"
    assert by_variant.variants == ("olmo3base",)
    assert type(get_task("mbpp:olmo3base:bpb")) is not type(get_task("mbpp:bpb:olmo3base"))


def test_overrides_follow_variant_order() -> None:
    resolved = resolve_task_spec("arc_easy:bpb:olmo3base")

    assert resolved.variants == ("bpb", "olmo3base")
    assert [sorted(o) for o in resolved.overrides] == [
        ["formatter", "metrics"],
        ["fewshot_source", "metrics", "num_fewshot", "split"],
    ]
    # Both set metrics, and get_task applies the later one.
    assert type(get_task("arc_easy:bpb:olmo3base").config.metrics[0]) is type(
        resolved.overrides[1]["metrics"][0]
    )


def test_base_task_has_no_variants() -> None:
    resolved = resolve_task_spec("arc_easy")

    assert resolved.task_name == "arc_easy"
    assert resolved.variants == ()
    assert resolved.overrides == ()


def test_unknown_task_and_variant_raise() -> None:
    with pytest.raises(KeyError, match="Unknown task 'no_such_task'"):
        resolve_task_spec("no_such_task:bpb")
    with pytest.raises(KeyError, match="Unknown variant 'nope' for task 'arc_easy'"):
        resolve_task_spec("arc_easy:nope")
    with pytest.raises(KeyError, match="Unknown variant ''"):
        resolve_task_spec("arc_easy::mc")
