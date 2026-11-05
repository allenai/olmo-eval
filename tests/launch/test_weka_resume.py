"""Native mounts and shared-run ownership through public launch/lifecycle paths."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest

from olmo_eval.launch.beaker.resume import RETENTION_SECONDS, refresh_files, run_shared
from olmo_eval.launch.beaker.volume import launch_with_volume
from olmo_eval.runners.asynq.progress import ProgressMismatchError, ProgressStore


@pytest.fixture
def shared(tmp_path):
    root = tmp_path / "volume"
    root.mkdir()
    run_uuid = str(uuid.uuid4())
    directory = root / "tmp-7d" / run_uuid
    return root, run_uuid, tmp_path / "results", directory


def child(code):
    # The wrapper appends the normal runner's output option.
    return [sys.executable, "-c", "import pathlib,sys; out=pathlib.Path(sys.argv[-1]); " + code]


def test_independent_uuids_do_not_overwrite_and_completed_run_is_not_repeated(shared):
    root, run_uuid, results, directory = shared
    command = child("p=out/'count'; p.write_text(str(int(p.read_text())+1) if p.exists() else '1')")
    assert run_shared(root, run_uuid, results, command) == 0
    assert run_shared(root, run_uuid, results, command) == 0
    assert (directory / "output/count").read_text() == "1"
    another = str(uuid.uuid4())
    assert run_shared(root, another, results.parent / "other-results", command) == 0
    assert (root / "tmp-7d" / another / "output/count").read_text() == "1"
    assert sorted(p.name for p in results.iterdir()) == ["weka-resume.json"]
    assert not (results / "weka-resume.json").is_symlink()


def test_a_live_writer_cannot_be_overwritten(shared):
    root, run_uuid, results, directory = shared
    directory.mkdir(parents=True)
    with (directory / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="Another attempt owns"):
            run_shared(root, run_uuid, results, child("raise AssertionError('must not run')"))
    assert not (directory / "session.json").exists()
    assert not (results / "weka-resume.json").exists()


def test_failed_child_retains_files_and_resumes_after_stale_progress_lock(shared):
    root, run_uuid, results, directory = shared
    command = child(
        "p=out/'attempt'; prior=p.exists(); p.write_text('saved'); "
        "lock=out/'progress/model/.lock'; lock.parent.mkdir(parents=True, exist_ok=True); "
        "assert not lock.exists(); lock.write_text('old-node'); sys.exit(0 if prior else 143)"
    )
    assert run_shared(root, run_uuid, results, command) == 143
    assert not (directory / "complete.json").exists()
    assert run_shared(root, run_uuid, results, command) == 0
    assert not (directory / "output/progress").exists()


@pytest.mark.parametrize(
    "damage", ["expired", "session", "artifact", "command", "changed", "completion"]
)
def test_expired_missing_or_changed_state_is_refused(shared, damage):
    root, run_uuid, results, directory = shared
    command = child("(out/'prediction').write_text('result')")
    run_shared(root, run_uuid, results, command)
    if damage == "expired":
        old = time.time() - RETENTION_SECONDS - 1
        os.utime(directory / "session.json", (old, old))
    elif damage == "session":
        (directory / "session.json").unlink()
    elif damage == "artifact":
        (directory / "output/prediction").unlink()
    elif damage == "changed":
        (directory / "output/prediction").write_text("RESULT")
    elif damage == "completion":
        (directory / "complete.json").unlink()
    else:
        command = child("raise AssertionError('changed')")
    with pytest.raises(RuntimeError):
        run_shared(root, run_uuid, results, command)


def test_refresh_extends_each_file_without_following_links(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    old_segment = root / "old.jsonl"
    outside = tmp_path / "outside"
    for path in (old_segment, outside):
        path.write_text("preserved")
        os.utime(path, (1, 1))
    (root / "linked-file").symlink_to(outside)
    (root / "linked-dir").symlink_to(tmp_path, target_is_directory=True)
    refresh_files(root)
    assert old_segment.stat().st_mtime > 1
    assert outside.stat().st_mtime == 1


def test_missing_mount_and_invalid_uuid_are_refused(shared):
    root, run_uuid, results, _ = shared
    with pytest.raises(RuntimeError, match="not mounted"):
        run_shared(root / "missing", run_uuid, results, child("pass"))
    with pytest.raises(ValueError):
        run_shared(root, "../escape", results, child("pass"))


@pytest.mark.parametrize("damage", ["deleted", "changed", "inventory"])
def test_committed_segments_cannot_be_silently_lost_or_changed(tmp_path, damage):
    store = ProgressStore(tmp_path / "segments", verify_integrity=True)
    store.append({"header": {"spec": "test"}, "t": "test"})
    store.close()
    store.append({"t": "test", "score": 1})
    store.close()
    if damage == "deleted":
        (store.path / "000000.jsonl").unlink()
    elif damage == "changed":
        path = store.path / "000001.jsonl"
        path.write_text(path.read_text().replace('"score":1', '"score":0'))
    else:
        (store.path / "index.json").unlink()
    with pytest.raises(ProgressMismatchError):
        list(ProgressStore(store.path, verify_integrity=True).read())


def test_integrity_checked_segments_resume_and_append(tmp_path):
    store = ProgressStore(tmp_path / "segments", verify_integrity=True)
    store.append({"t": "a", "score": 1})
    store.close()
    resumed = ProgressStore(store.path, verify_integrity=True)
    assert list(resumed.read()) == [{"t": "a", "score": 1}]
    resumed.append({"t": "a", "score": 2})
    resumed.close()
    assert list(ProgressStore(store.path, verify_integrity=True).read()) == [
        {"t": "a", "score": 1},
        {"t": "a", "score": 2},
    ]


def test_native_submission_preserves_gantry_spec_and_adds_volume(monkeypatch):
    seen = {}

    def gantry(**kwargs):
        assert kwargs["dry_run"] is True
        Path(kwargs["save_spec"]).write_text(
            json.dumps(
                {
                    "version": "v2",
                    "groups": ["test-group"],
                    "retry": {"allowedTaskRetries": 2},
                    "tasks": [
                        {
                            "context": {"minRuntime": 300_000_000_000},
                            "timeout": 600_000_000_000,
                            "datasets": [{"mountPath": "/gantry", "source": {"beaker": "setup"}}],
                            "envVars": [{"name": "HF_TOKEN", "secret": "hf-secret"}],
                        }
                    ],
                }
            )
        )

    def submit(command, **kwargs):
        seen.update(json.loads(Path(command[3]).read_text()))
        assert command[4:] == ["--workspace", "ai2/test", "--name", "test", "--format", "json"]
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(command, 0, '[{"id":"experiment-id"}]')

    monkeypatch.setattr("olmo_eval.launch.beaker.volume.subprocess.run", submit)
    result = launch_with_volume(
        gantry, {"name": "test", "workspace": "ai2/test"}, "ai2/cache", dry_run=False
    )
    assert result == "experiment-id"
    assert seen["tasks"][0]["datasets"][-1] == {
        "mountPath": "/eval-cache",
        "source": {"volume": "ai2/cache"},
        "mode": "readwrite",
    }
    assert seen["tasks"][0]["context"]["minRuntime"] == "300000000000ns"
    assert seen["tasks"][0]["timeout"] == "600000000000ns"
    assert seen["tasks"][0]["envVars"][0]["secret"] == "hf-secret"
    assert seen["retry"]["allowedTaskRetries"] == 2
    assert seen["groups"] == ["test-group"]


def test_native_submission_does_not_retry_failure(monkeypatch):
    def gantry(**kwargs):
        Path(kwargs["save_spec"]).write_text('{"version":"v2","tasks":[{}]}')

    submit = Mock(
        side_effect=subprocess.CalledProcessError(1, "beaker", stderr="workspace has no claim")
    )
    monkeypatch.setattr("olmo_eval.launch.beaker.volume.subprocess.run", submit)
    with pytest.raises(RuntimeError, match="has no claim"):
        launch_with_volume(
            gantry, {"name": "test", "workspace": "ai2/test"}, "ai2/cache", dry_run=False
        )
    assert submit.call_count == 1


def test_sigterm_is_forwarded_and_releases_ownership(shared):
    root, run_uuid, results, directory = shared
    command = child("(out/'ready').touch(); import time; time.sleep(30)")
    script = (
        "import sys,json; from pathlib import Path; "
        "from olmo_eval.launch.beaker.resume import run_shared; "
        "sys.exit(run_shared(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),json.loads(sys.argv[4])))"
    )
    supervisor = subprocess.Popen(
        [sys.executable, "-c", script, str(root), run_uuid, str(results), json.dumps(command)]
    )
    try:
        deadline = time.monotonic() + 15
        while not (directory / "output/ready").exists():
            assert supervisor.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        supervisor.send_signal(signal.SIGTERM)
        assert supervisor.wait(timeout=15) == 143
        with (directory / "run.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()


def test_killing_supervisor_does_not_release_a_live_childs_lock(shared):
    root, run_uuid, results, directory = shared
    command = child("import os,time; (out/'pid').write_text(str(os.getpid())); time.sleep(30)")
    script = (
        "import sys,json; from pathlib import Path; "
        "from olmo_eval.launch.beaker.resume import run_shared; "
        "run_shared(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),json.loads(sys.argv[4]))"
    )
    supervisor = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(root),
            run_uuid,
            str(results),
            json.dumps(command),
        ]
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 15
        while not (directory / "output/pid").exists():
            assert supervisor.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        child_pid = int((directory / "output/pid").read_text())
        supervisor.kill()
        supervisor.wait(timeout=10)
        with pytest.raises(RuntimeError, match="Another attempt owns"):
            run_shared(root, run_uuid, results, command)
    finally:
        if child_pid is not None:
            os.killpg(child_pid, signal.SIGKILL)
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()


def test_yaml_volume_config_reaches_unique_native_job_commands(tmp_path, monkeypatch):
    import importlib

    from olmo_eval.cli.beaker.config_loader import LaunchConfigLoader
    from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
    from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler

    path = tmp_path / "eval.yaml"
    path.write_text(
        "name: test\nmodels: [mock]\ntasks: [gsm8k]\ngpus: 0\n"
        "cluster: h100\nworkspace: ai2/test\nbudget: ai2/test\nresume_volume: ai2/cache\n"
    )
    config = LaunchConfigLoader(str(path), {"upload": False}).load()
    assert config.resume_volume == "ai2/cache"
    assembler = JobConfigAssembler(
        config=config,
        effective_image="image",
        effective_groups=[],
        beaker_username="tester",
        common_secrets=[],
        task_secrets=[],
        inject_aws_credentials=False,
        inject_gcs_credentials=False,
    )
    monkeypatch.setattr(
        importlib.import_module("olmo_eval.cli.beaker.job_assembler"),
        "cluster_has_weka",
        lambda _: False,
    )
    plan = ExperimentPlan(
        name="test",
        model_spec="mock",
        priority="normal",
        tasks=["gsm8k"],
        original_task_specs=["gsm8k"],
        total_expanded_tasks=1,
        num_gpus=0,
    )
    first, second = assembler.assemble(plan), assembler.assemble(plan)
    assert first.resume_volume == "ai2/cache"
    assert first.command[:3] == ["python", "-m", "olmo_eval.launch.beaker.resume"]
    one = first.command[first.command.index("--run-uuid") + 1]
    two = second.command[second.command.index("--run-uuid") + 1]
    assert str(uuid.UUID(one)) == one
    assert one != two
    assert first.command[first.command.index("--") + 1 :][:2] == ["olmo-eval", "run"]
    assert "--resume" in first.command
    assert "-O" not in first.command
    overridden = LaunchConfigLoader(str(path), {"resume_volume": "ai2/other"}).load()
    assert overridden.resume_volume == "ai2/other"
