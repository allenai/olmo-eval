"""Tests for the Olmo 3.5 post-training dev and test suites."""

import pytest

from olmo_eval.common.configs import expand_tasks, validate_task_metrics, validate_tasks
from olmo_eval.evals.suites import AggregationStrategy, Suite, get_suite
from olmo_eval.evals.tasks.common import get_task

DEV_SHARED = (
    "gpqa_diamond:cot",
    "gpqa_main:cot",
    "mmlu:cot",
    "popqa:chat",
    "omega:dev",
    "hmmt_feb_2025:pass_at_32",
    "hmmt_nov_2025:pass_at_32",
    "hmmt_feb_2026:pass_at_32",
    "livecodebench:lite",
    "bfcl",
    "ifeval",
    "ifeval_ood",
    "ruler_plus_all__65536",
    "ruler_plus_all__131072",
    "ruler_plus_all__262144",
    "helmet_all__65536",
    "ctc:r64k",
)
TEST_SHARED = (
    "mmlu_pro:cot",
    "omniscience:judge",
    "zebralogic:chat",
    "aime_2025:pass_at_32",
    "aime_2026:pass_at_32",
    "livecodebench_hidden",
    "do_anything_now:wg_judge_thinking",
    "wildguardtest:wg_judge_thinking",
    "wildjailbreak:wg_judge_thinking",
    "wmdp:mcq",
    "bbq:mcq",
    "strongreject:sr_judge_thinking",
)
MEMBERS = {
    "posttrain:tiny:dev": DEV_SHARED,
    "posttrain:tiny:test": TEST_SHARED,
    "posttrain:small:dev": (
        *DEV_SHARED,
        "mt_bench",
        "multi_if",
        "multichallenge",
        "harmbench:wg_judge_thinking",
        "xstest:wg_judge_thinking",
    ),
    "posttrain:small:test": (
        *TEST_SHARED,
        "hle:text:verified",
        "arena_hard_v2",
        "frontierscience",
    ),
}


def _names(suite: Suite) -> tuple[str, ...]:
    return tuple(t.name if isinstance(t, Suite) else t for t in suite.tasks)


@pytest.mark.parametrize("name", MEMBERS)
def test_members_and_no_pass_average(name):
    suite = get_suite(name)

    assert suite.aggregation == AggregationStrategy.NONE
    assert _names(suite) == MEMBERS[name]
    assert tuple(expand_tasks([name])) == suite.expand()


@pytest.mark.parametrize("size", ["tiny", "small"])
def test_dev_and_test_share_no_task(size):
    dev = set(get_suite(f"posttrain:{size}:dev").expand())
    test = set(get_suite(f"posttrain:{size}:test").expand())

    assert not dev & test


@pytest.mark.parametrize("size", ["tiny", "small"])
def test_gpqa_diamond_and_its_superset_stay_in_dev(size):
    # Diamond's questions are a subset of main's, so splitting them leaks test items.
    dev = get_suite(f"posttrain:{size}:dev").expand()

    assert {"gpqa_diamond:cot", "gpqa_main:cot"} <= set(dev)


# Exempt by design: long-context retrieval runs in completion mode with direct-answer
# budgets, and BFCL keeps the 4,096-token budget that existing post-training reads use.
OWN_BUDGET_SUITES = (
    "ruler_plus_all__65536",
    "ruler_plus_all__131072",
    "ruler_plus_all__262144",
    "helmet_all__65536",
    "ctc:r64k",
    "bfcl",
)
MIN_GENERATION_BUDGET = 32768


@pytest.mark.parametrize("name", MEMBERS)
def test_no_task_cuts_off_a_reasoning_trace(name):
    exempt = {task for suite in OWN_BUDGET_SUITES for task in get_suite(suite).expand()}
    for task in get_suite(name).expand():
        if task in exempt:
            continue
        params = get_task(task).config.sampling_params
        assert params is not None, task
        assert not params.stop_sequences, task
        assert params.max_tokens is None or params.max_tokens >= MIN_GENERATION_BUDGET, task


@pytest.mark.parametrize("name", MEMBERS)
def test_every_task_resolves_and_has_metrics(name):
    # The checks `olmo-eval beaker launch` runs before it builds a spec.
    _valid, invalid = validate_tasks(list(get_suite(name).expand()))
    _with_metrics, without_metrics = validate_task_metrics(list(get_suite(name).expand()))

    assert not invalid
    assert not without_metrics
