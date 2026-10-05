import json
import sys
from pathlib import Path

import pytest

from olmo_eval.cli.run.config import RunConfigBuilder
from olmo_eval.cli.utils import process_ordered_args, reconstruct_ordered_args
from olmo_eval.common.configs import expand_tasks
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.launch import peers
from olmo_eval.launch.peers import CORE_TASKS, PEERS, build_command, validate_smoke
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


def smoke_metrics():
    return {
        "errors": [],
        "tasks": [
            {
                "task": task,
                "instances_saved": 2,
                "instances_processed": 2,
                "instances_failed": 0,
                "generation_counts": {
                    "generations": 2,
                    "cap_hit": 2,
                    "empty": 0,
                    "unclosed_think": 0,
                    "finish_reason_unknown": 0,
                },
            }
            for task in CORE_TASKS
        ],
    }


def test_smoke_accepts_deliberate_truncation(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps(smoke_metrics()))
    counts = validate_smoke(tmp_path)
    assert counts["generations"] == 12
    assert counts["cap_hit"] == 12
    assert json.loads((tmp_path / "smoke-validation.json").read_text())["accepted"]


@pytest.mark.parametrize(
    "problem", ("missing", "duplicate", "errored", "unknown", "partial", "empty")
)
def test_smoke_rejects_incomplete_or_unaccounted_outputs(tmp_path, problem):
    metrics = smoke_metrics()
    if problem == "missing":
        metrics["tasks"].pop()
    elif problem == "duplicate":
        metrics["tasks"][-1] = metrics["tasks"][0]
    elif problem == "errored":
        metrics["errors"].append({"error": "provider failed"})
    elif problem == "unknown":
        metrics["tasks"][0]["generation_counts"]["finish_reason_unknown"] = 1
    elif problem == "empty":
        metrics["tasks"][0]["generation_counts"]["empty"] = 2
    else:
        metrics["tasks"][0]["instances_saved"] = 1
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    (tmp_path / "smoke-validation.json").write_text('{"accepted": true}')
    with pytest.raises(ValueError):
        validate_smoke(tmp_path)
    assert not (tmp_path / "smoke-validation.json").exists()


def test_manifest_records_clean_checkout_and_job_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["peers", "--peer", "lfm", "--phase", "core", "--output-dir", str(tmp_path)],
    )
    monkeypatch.setenv("BEAKER_JOB_ID", "smoke-job")
    monkeypatch.setattr(
        peers.subprocess,
        "check_output",
        lambda command, **kwargs: "abc123\n" if "rev-parse" in command else "",
    )
    launched = []
    monkeypatch.setattr(peers.subprocess, "run", lambda command, **kwargs: launched.append(command))
    peers.main()
    manifest = json.loads((tmp_path / "peer-manifest.json").read_text())
    assert manifest["commit"] == "abc123"
    assert manifest["dirty_tree"] is False
    assert manifest["beaker_job_id"] == "smoke-job"
    assert manifest["model_revision"] == PEERS["lfm"].revision
    assert manifest["sampling"]["max_tokens"] == 32768
    assert manifest["command"] == launched[0]
    assert not manifest["smoke_only"]


@pytest.mark.parametrize("phase", ("core", "knowledge"))
def test_full_runs_reject_dirty_source(tmp_path, monkeypatch, phase):
    monkeypatch.setattr(
        sys, "argv", ["peers", "--peer", "lfm", "--phase", phase, "--output-dir", str(tmp_path)]
    )
    monkeypatch.setattr(
        peers.subprocess,
        "check_output",
        lambda command, **kwargs: "abc123\n" if "rev-parse" in command else " M dirty\n",
    )
    with pytest.raises(SystemExit, match="2"):
        peers.main()
    assert not (tmp_path / "peer-manifest.json").exists()


def test_failed_launch_clears_previous_smoke_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["peers", "--peer", "lfm", "--phase", "smoke", "--output-dir", str(tmp_path)]
    )
    monkeypatch.setattr(peers.subprocess, "check_output", lambda *args, **kwargs: "")

    def fail_launch(*args, **kwargs):
        raise peers.subprocess.CalledProcessError(1, "olmo-eval")

    monkeypatch.setattr(peers.subprocess, "run", fail_launch)
    (tmp_path / "smoke-validation.json").write_text('{"accepted": true}')
    with pytest.raises(peers.subprocess.CalledProcessError):
        peers.main()
    assert not (tmp_path / "smoke-validation.json").exists()
