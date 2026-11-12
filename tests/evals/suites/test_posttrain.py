"""Tests for the Olmo 3.5 post-training dev and test suites."""

from dataclasses import replace

import pytest

from olmo_eval.common.configs import expand_tasks, validate_task_metrics, validate_tasks
from olmo_eval.evals.suites import AggregationStrategy, Suite, get_suite
from olmo_eval.evals.suites.posttrain import (
    BFCL_MULTI_TURN_SAMPLING,
    MODEL_SAMPLING,
    OWN_SAMPLING,
    SAMPLED_FROM,
)
from olmo_eval.evals.tasks.common import get_task

DEV_SHARED = (
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
    "harmbench:wg_judge_thinking_t10_32k",
    "xstest:wg_judge_thinking_t10_32k",
    "wildguardtest:wg_judge_thinking_t10_32k",
)


TEST_SHARED = (
    "mmlu_pro:cot:t10_32k",
    "omniscience:judge_t10_32k",
    "zebralogic:chat_t10_32k",
    "aime_2025:pass_at_32_t10_32k",
    "aime_2026:pass_at_32_t10_32k",
    "livecodebench_hidden:t10_32k",
    "do_anything_now:wg_judge_thinking_t10_32k",
    "wildjailbreak:wg_judge_thinking_t10_32k",
    "wmdp:mcq_t10_32k",
    "bbq:mcq_t10_32k",
    "strongreject:sr_judge_thinking_t10_32k",
)
MEMBERS = {
    "posttrain:tiny:dev": DEV_SHARED,
    "posttrain:tiny:dev:bfcl_multi_turn": ("bfcl:multi_turn:t10_16k",),
    "posttrain:tiny:test": TEST_SHARED,
    "posttrain:small:dev": (*DEV_SHARED, "multichallenge:t10_32k"),
    "posttrain:small:dev:multi_turn": ("mt_bench:t10_32k", "multi_if:t10_32k"),
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
# BFCL multi-turn runs at the ledger's setting for it: 16,384 tokens per model turn.
BFCL_MULTI_TURN_TASKS = frozenset(get_suite("posttrain:tiny:dev:bfcl_multi_turn").expand())


def _names(suite: Suite) -> tuple[str, ...]:
    return tuple(t.name if isinstance(t, Suite) else t for t in suite.tasks)


@pytest.mark.parametrize("name", MEMBERS)
def test_members_and_no_pass_average(name):
    suite = get_suite(name)

    assert suite.aggregation == AggregationStrategy.NONE
    assert _names(suite) == MEMBERS[name]
    assert tuple(expand_tasks([name])) == suite.expand()


DEV_SUITES = {
    "tiny": ("posttrain:tiny:dev", "posttrain:tiny:dev:bfcl_multi_turn"),
    "small": ("posttrain:small:dev", "posttrain:small:dev:multi_turn"),
}


@pytest.mark.parametrize("size", ["tiny", "small"])
def test_dev_and_test_share_no_task(size):
    dev = {task for name in DEV_SUITES[size] for task in get_suite(name).expand()}
    test = set(get_suite(f"posttrain:{size}:test").expand())

    assert not dev & test


@pytest.mark.parametrize("name", MEMBERS)
def test_each_suite_runs_under_one_scaffold(name):
    # The multi_turn_chat scaffold drops tool calls and has no sandbox, so its tasks need a
    # suite and a job of their own.
    expected = "multi_turn_chat" if name.endswith(":multi_turn") else None
    for task in get_suite(name).expand():
        assert get_task(task).required_scaffold == expected, task


@pytest.mark.parametrize("name", MEMBERS)
def test_bfcl_multi_turn_has_a_suite_of_its_own(name):
    # Its tasks declare no scaffold but score only under --harness bfcl_multi_turn.
    tasks = get_suite(name).expand()
    multi_turn = [task for task in tasks if task.startswith("bfcl_multi_turn")]
    if name.endswith(":bfcl_multi_turn"):
        assert multi_turn == list(tasks)
    else:
        assert not multi_turn


@pytest.mark.parametrize("size", ["tiny", "small"])
def test_no_suite_reads_gpqa(size):
    # Main truncates at chance for a reasoning model at 32,768 tokens; Diamond is a subset of it.
    dev = [task for name in DEV_SUITES[size] for task in get_suite(name).expand()]
    test = get_suite(f"posttrain:{size}:test").expand()

    assert not [task for task in dev if task.startswith("gpqa")]
    assert not [task for task in test if task.startswith("gpqa")]


MIN_GENERATION_BUDGET = 32768


@pytest.mark.parametrize("name", MEMBERS)
def test_no_task_cuts_off_a_reasoning_trace(name):
    for task in get_suite(name).expand():
        if task in OWN_SAMPLING_TASKS | BFCL_MULTI_TURN_TASKS:
            continue
        params = get_task(task).config.sampling_params
        assert params is not None, task
        assert not params.stop_sequences, task
        assert params.max_tokens is None or params.max_tokens >= MIN_GENERATION_BUDGET, task


@pytest.mark.parametrize("name", MEMBERS)
def test_sampled_tasks_run_at_the_models_suggested_sampling(name):
    # The ledger's sampling, not olmo-eval's defaults (0.6, no budget); nothing else changes.
    for task in get_suite(name).expand():
        if task in OWN_SAMPLING_TASKS | BFCL_MULTI_TURN_TASKS:
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


def test_bfcl_multi_turn_runs_at_the_ledgers_setting():
    # The ledger's BFCL multi-turn reads: temperature 1.0, top_p 0.95, 16,384 tokens per turn.
    suite = get_suite("posttrain:tiny:dev:bfcl_multi_turn")
    (summary,) = suite.tasks
    assert isinstance(summary, Suite)
    assert summary.aggregation == get_suite("bfcl:multi_turn").aggregation

    for task, source in zip(summary.tasks, get_suite("bfcl:multi_turn").tasks, strict=True):
        config = get_task(task).config
        source_config = get_task(source).config
        assert config.sampling_params == replace(
            source_config.sampling_params, **BFCL_MULTI_TURN_SAMPLING
        ), task
        assert replace(config, sampling_params=source_config.sampling_params) == source_config


def test_tiny_dev_needs_no_secret():
    # Hill-climbing on Tiny should not need an API key.
    needs = {
        task: get_task(task).config.required_secrets
        for name in DEV_SUITES["tiny"]
        for task in get_suite(name).expand()
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
