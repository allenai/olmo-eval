"""Tests for the Olmo 3.5 post-training dev and test suites."""

from dataclasses import replace

import pytest

from olmo_eval.common.configs import expand_tasks, validate_task_metrics, validate_tasks
from olmo_eval.evals.suites import AggregationStrategy, Suite, get_suite
from olmo_eval.evals.suites.posttrain import MODEL_SAMPLING, OWN_SAMPLING, SAMPLED_FROM
from olmo_eval.evals.tasks.common import get_task

DEV_SHARED = (
    "gpqa_main:cot:t10_32k",
    "mmlu:cot:hillclimb:t10_32k",
    "popqa:chat_hillclimb_t10_32k",
    "omega:dev:t10_32k",
    "hmmt_feb_2025:pass_at_8_t10_32k",
    "hmmt_nov_2025:pass_at_8_t10_32k",
    "hmmt_feb_2026:pass_at_8_t10_32k",
    "livecodebench:lite_t10_32k",
    "bfcl",
    "ifeval:t10_32k",
    "ifeval_ood:t10_32k",
    "ruler_plus_all__65536",
    "ruler_plus_all__131072:hillclimb",
    "ruler_plus_all__262144:hillclimb",
)


TEST_SHARED = (
    "mmlu_pro:cot:t10_32k",
    "omniscience:judge_t10_32k",
    "zebralogic:chat_t10_32k",
    "aime_2025:pass_at_32_t10_32k",
    "aime_2026:pass_at_32_t10_32k",
    "livecodebench_hidden:t10_32k",
    "do_anything_now:wg_judge_thinking_t10_32k",
    "wildguardtest:wg_judge_thinking_t10_32k",
    "wildjailbreak:wg_judge_thinking_t10_32k",
    "wmdp:mcq_t10_32k",
    "bbq:mcq_t10_32k",
    "strongreject:sr_judge_thinking_t10_32k",
)
MEMBERS = {
    "posttrain:tiny:dev": DEV_SHARED,
    "posttrain:tiny:test": TEST_SHARED,
    "posttrain:small:dev": (
        *DEV_SHARED,
        "mt_bench:t10_32k",
        "multi_if:t10_32k",
        "multichallenge:t10_32k",
        "harmbench:wg_judge_thinking_t10_32k",
        "xstest:wg_judge_thinking_t10_32k",
    ),
    "posttrain:small:test": (
        *TEST_SHARED,
        "hle:text:verified:t10_32k",
        "arena_hard_v2:t10_32k",
        "frontierscience:t10_32k",
    ),
}
# RULER+ runs greedy in completion mode with direct-answer budgets, and BFCL
# keeps the settings that existing post-training reads use.
OWN_SAMPLING_TASKS = frozenset(task for suite in OWN_SAMPLING for task in get_suite(suite).expand())


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
def test_dev_reads_gpqa_main_alone_and_test_no_gpqa(size):
    # Diamond's questions are a subset of main's: in dev it adds nothing, in test it leaks.
    dev = get_suite(f"posttrain:{size}:dev").expand()
    test = get_suite(f"posttrain:{size}:test").expand()

    assert [task for task in dev if task.startswith("gpqa")] == ["gpqa_main:cot:t10_32k"]
    assert not [task for task in test if task.startswith("gpqa")]


MIN_GENERATION_BUDGET = 32768


@pytest.mark.parametrize("name", MEMBERS)
def test_no_task_cuts_off_a_reasoning_trace(name):
    for task in get_suite(name).expand():
        if task in OWN_SAMPLING_TASKS:
            continue
        params = get_task(task).config.sampling_params
        assert params is not None, task
        assert not params.stop_sequences, task
        assert params.max_tokens is None or params.max_tokens >= MIN_GENERATION_BUDGET, task


@pytest.mark.parametrize("name", MEMBERS)
def test_sampled_tasks_run_at_the_models_suggested_sampling(name):
    # The ledger's sampling, not olmo-eval's defaults (0.6, no budget); nothing else changes.
    for task in get_suite(name).expand():
        if task in OWN_SAMPLING_TASKS:
            continue
        config = get_task(task).config
        source = get_task(SAMPLED_FROM[task]).config
        assert config.sampling_params == replace(source.sampling_params, **MODEL_SAMPLING), task
        assert replace(config, sampling_params=source.sampling_params) == source, task


@pytest.mark.parametrize("name", MEMBERS)
def test_every_task_resolves_and_has_metrics(name):
    # The checks `olmo-eval beaker launch` runs before it builds a spec.
    _valid, invalid = validate_tasks(list(get_suite(name).expand()))
    _with_metrics, without_metrics = validate_task_metrics(list(get_suite(name).expand()))

    assert not invalid
    assert not without_metrics


def test_tiny_dev_needs_no_secret():
    # Hill-climbing on Tiny should not need an API key.
    needs = {
        task: get_task(task).config.required_secrets
        for task in get_suite("posttrain:tiny:dev").expand()
        if get_task(task).config.required_secrets
    }

    assert not needs


def test_dev_reads_fixed_samples_of_popqa_and_mmlu():
    # Dev has to run fast: about 14,000 items each, at ~11k reasoning tokens per item, is too slow.
    popqa = get_task("popqa:chat:hillclimb").config
    mmlu = [get_task(task).config for task in get_suite("mmlu:cot:hillclimb").expand()]

    assert popqa.limit == 1000
    assert popqa.sampling_params == get_task("popqa:chat").config.sampling_params
    assert len(mmlu) == 57
    assert {config.limit for config in mmlu} == {18}
    assert get_suite("mmlu:cot:hillclimb").aggregation == get_suite("mmlu:cot").aggregation


@pytest.mark.parametrize("size", [131072, 262144])
def test_dev_reads_fixed_ruler_plus_samples_at_long_contexts(size):
    full = get_suite(f"ruler_plus_all__{size}")
    sample = get_suite(f"ruler_plus_all__{size}:hillclimb")

    assert sample.expand() == tuple(f"{task}:hillclimb" for task in full.expand())
    assert {get_task(task).config.limit for task in sample.expand()} == {128}
    assert sample.aggregation == full.aggregation


@pytest.mark.parametrize("task", ["hmmt_feb_2025", "hmmt_nov_2025", "hmmt_feb_2026"])
def test_dev_hmmt_draws_8_samples_with_the_pass_at_32_setup(task):
    dev = get_task(f"{task}:pass_at_8").config
    full = get_task(f"{task}:pass_at_32").config

    assert dev.sampling_params.num_samples == 8
    assert replace(dev.sampling_params, num_samples=32) == full.sampling_params
    assert dev.primary_metric == full.primary_metric
