from pathlib import Path

import pytest

from olmo_eval.cli.run.config import RunConfigBuilder
from olmo_eval.cli.utils import process_ordered_args, reconstruct_ordered_args
from olmo_eval.common.configs import expand_tasks
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.launch.peers import CORE_TASKS, PEERS, build_command
from olmo_eval.runners.asynq.runner import AsyncEvalRunner


def resolved(command):
    ordered = reconstruct_ordered_args(command[2:])
    task_overrides, harness_overrides = process_ordered_args(ordered)
    tasks = tuple(command[i + 1] for i, arg in enumerate(command[:-1]) if arg == "-t")
    return RunConfigBuilder(
        model=command[3],
        task=tasks,
        output_dir="/tmp/peer-test",
        harness_preset="default",
        cli_task_overrides=task_overrides,
        cli_harness_overrides=harness_overrides,
    ).build()


def resolved_tasks(config):
    from dataclasses import replace

    runner = AsyncEvalRunner(task_overrides=config.task_overrides)
    for spec in expand_tasks(config.task_specs):
        overrides, sampling = runner._build_task_overrides(spec)
        task = get_task(spec, config_overrides=overrides)
        task.config = replace(
            task.config, sampling_params=replace(task.config.sampling_params, **sampling)
        )
        yield task


@pytest.mark.parametrize("peer", PEERS)
def test_core_contract_resolves_for_every_peer(peer):
    command = build_command(peer, "core", Path("/tmp/peers"), "http://localhost:30000/v1")
    config = resolved(command)
    assert config.task_specs == list(CORE_TASKS)
    assert config.provider_config.revision == PEERS[peer].revision
    assert config.provider_config.base_url == "http://localhost:30000/v1"
    assert config.harness_config.max_hard_failure_rate == 0.0
    for task in resolved_tasks(config):
        assert task.config.sampling_params.max_tokens == 32768
        assert task.config.sampling_params.temperature == 1.0
        assert task.config.sampling_params.top_p == 0.95
        assert task.config.sampling_params.top_k == 20
        assert task.config.sampling_params.num_samples == 1
        assert task.config.strip_thinking


def test_smoke_limits_every_task_and_generation():
    config = resolved(build_command("lfm", "smoke", Path("/tmp/peers")))
    for task in resolved_tasks(config):
        assert task.config.limit == 2
        assert task.config.sampling_params.max_tokens == 128


def test_knowledge_pass_expands_to_cot_categories():
    config = resolved(build_command("minicpm", "knowledge", Path("/tmp/peers")))
    tasks = list(resolved_tasks(config))
    assert len(tasks) == 14
    assert all(task.config.sampling_params.max_tokens == 32768 for task in tasks)


def test_ling_requires_supported_external_runtime():
    with pytest.raises(ValueError, match="BailingMoeV3"):
        build_command("ling-tiny", "core", Path("/tmp/peers"))
