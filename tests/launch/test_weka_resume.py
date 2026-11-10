"""Native mounts and shared-run ownership through public launch/lifecycle paths."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from olmo_eval.common.constants.infrastructure import SHARED_RUN_ENV
from olmo_eval.launch.beaker.resume import RETENTION_SECONDS, refresh_files, run_shared
from olmo_eval.launch.beaker.volume import launch_with_volume
from olmo_eval.runners.asynq.progress import ProgressMismatchError, ProgressStore


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("published", [False, True])
def test_interruption_during_inventory_commit_replays_only_committed_segments(
    tmp_path, monkeypatch, prior, published
):
    from olmo_eval.runners.asynq import progress as module

    store = ProgressStore(tmp_path / "segments", verify_integrity=True)
    expected = []
    if prior:
        store.append({"score": 1})
        store.close()
        expected.append({"score": 1})
    store.append({"score": 2})
    write = module._write_json_atomic

    def interrupted(path, data):
        if path.name == "index.json" and data:
            if published:
                write(path, data)
            raise OSError("interrupted inventory publication")
        write(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(module, "_write_json_atomic", interrupted)
        with pytest.raises(OSError, match="inventory publication"):
            store.close()
    if published:
        expected.append({"score": 2})
    restored = ProgressStore(store.path, verify_integrity=True)
    assert list(restored.read()) == expected
    restored.append({"score": 3})
    restored.close()
    assert list(ProgressStore(store.path, verify_integrity=True).read()) == [
        *expected,
        {"score": 3},
    ]


def test_retry_after_uncertain_commit_never_overwrites_a_referenced_segment(tmp_path, monkeypatch):
    from olmo_eval.runners.asynq import progress as module

    store = ProgressStore(tmp_path / "segments", verify_integrity=True)
    store.append({"score": 1})
    store.close()
    write = module._write_json_atomic

    def after_replace(path, data):
        write(path, data)
        raise OSError("index directory fsync failed after replace")

    store.append({"score": 2})
    with monkeypatch.context() as patch:
        patch.setattr(module, "_write_json_atomic", after_replace)
        with pytest.raises(OSError):
            store.close()
    committed = (store.path / "000001.jsonl").read_bytes()
    store.append({"score": 3})
    with monkeypatch.context() as patch:
        patch.setattr(module, "_write_json_atomic", Mock(side_effect=OSError("next commit failed")))
        with pytest.raises(OSError):
            store.close()
    assert (store.path / "000001.jsonl").read_bytes() == committed
    assert list(ProgressStore(store.path, verify_integrity=True).read()) == [
        {"score": 1},
        {"score": 2},
    ]


def test_progress_directory_io_failure_cannot_reset_segment_numbering(tmp_path, monkeypatch):
    store = ProgressStore(tmp_path / "segments", verify_integrity=True)
    store.append({"score": 1})
    store.close()
    original = (store.path / "000000.jsonl").read_bytes()
    scandir = os.scandir

    def fail(path):
        if Path(path) == store.path:
            raise OSError(errno.EIO, "segment directory unavailable")
        return scandir(path)

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", fail)
        with pytest.raises(OSError, match="segment directory unavailable"):
            list(ProgressStore(store.path, verify_integrity=True).read())
    resumed = ProgressStore(store.path, verify_integrity=True)
    assert list(resumed.read()) == [{"score": 1}]
    resumed.append({"score": 2})
    resumed.close()
    assert (store.path / "000000.jsonl").read_bytes() == original
    assert list(ProgressStore(store.path, verify_integrity=True).read()) == [
        {"score": 1},
        {"score": 2},
    ]


def test_inventory_also_establishes_next_segment_id(tmp_path, monkeypatch):
    import contextlib

    store = ProgressStore(tmp_path / "segments", verify_integrity=True)
    store.append({"score": 1})
    store.close()
    original = (store.path / "000000.jsonl").read_bytes()
    scandir = os.scandir

    @contextlib.contextmanager
    def incomplete_listing(path):
        if Path(path) == store.path:
            yield iter(())
        else:
            with scandir(path) as entries:
                yield entries

    resumed = ProgressStore(store.path, verify_integrity=True)
    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", incomplete_listing)
        assert list(resumed.read()) == [{"score": 1}]
    resumed.append({"score": 2})
    resumed.close()
    assert (store.path / "000000.jsonl").read_bytes() == original
    assert list(ProgressStore(store.path, verify_integrity=True).read()) == [
        {"score": 1},
        {"score": 2},
    ]


@pytest.mark.parametrize("stage", ["before_index", "after_index", "delete_old"])
def test_compaction_interruption_preserves_the_committed_generation(tmp_path, monkeypatch, stage):
    from olmo_eval.runners.asynq import progress as module

    store = ProgressStore(tmp_path / "segments", verify_integrity=True)
    store.append({"t": "a", "score": 1})
    store.append({"t": "b", "score": 2})
    store.close()
    write = module._write_json_atomic
    unlink = Path.unlink

    def fail_index(path, data):
        if stage == "after_index":
            write(path, data)
        raise OSError("compaction interrupted")

    def fail_delete(path, *args, **kwargs):
        if path.name == "000000.jsonl":
            raise OSError("compaction interrupted")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        if stage == "delete_old":
            patch.setattr(Path, "unlink", fail_delete)
        else:
            patch.setattr(module, "_write_json_atomic", fail_index)
        with pytest.raises(OSError):
            store.drop(keys=["a"])
    expected = [{"t": "b", "score": 2}]
    if stage == "before_index":
        expected.insert(0, {"t": "a", "score": 1})
    assert list(ProgressStore(store.path, verify_integrity=True).read()) == expected


def test_shared_discard_updates_inventory_and_keeps_empty_stream_initialized(tmp_path, monkeypatch):
    from olmo_eval.common.constants.infrastructure import SHARED_RUN_ENV
    from olmo_eval.runners.asynq.progress import discard_progress, stream_dir

    out = tmp_path / "results"
    weka = tmp_path / "weka"
    weka.mkdir()
    (weka / "session.json").write_text(json.dumps({"uuid": "u", "command": []}))
    monkeypatch.setenv(SHARED_RUN_ENV, str(weka))
    store = ProgressStore(stream_dir(out, "mock"), verify_integrity=True)
    assert store.path == weka / "progress/mock/run.segments"
    for spec in ("a", "b"):
        store.append({"t": spec, "header": {"spec": spec}})
    store.close()
    discard_progress(out, "mock", ["a"])
    assert list(ProgressStore(store.path, verify_integrity=True).read()) == [
        {"t": "b", "header": {"spec": "b"}},
    ]
    discard_progress(out, "mock", ["b"])
    assert (store.path / "index.json").is_file()
    assert list(ProgressStore(store.path, verify_integrity=True).read()) == []


@pytest.fixture
def shared(tmp_path):
    root = tmp_path / "volume"
    root.mkdir()
    run_uuid = str(uuid.uuid4())
    directory = root / "tmp-7d" / run_uuid
    return root, run_uuid, tmp_path / "results", directory


def child(code):
    # The wrapper appends the normal runner's output option (Beaker results) and points
    # SHARED_RUN_ENV at the WEKA recovery directory.
    return [
        sys.executable,
        "-c",
        "import os,pathlib,sys; out=pathlib.Path(sys.argv[-1]); "
        f"weka=pathlib.Path(os.environ['{SHARED_RUN_ENV}']); " + code,
    ]


RECORD = (
    "from olmo_eval.runners.asynq.progress import ProgressStore; "
    "store=ProgressStore(weka/'progress/model/run.segments',verify_integrity=True); "
    "store.append({'score':1}); store.close(); "
)


def test_outputs_go_to_beaker_results_and_only_recovery_state_to_weka(shared):
    root, run_uuid, results, directory = shared
    command = child(
        "(out/'predictions').mkdir(); (out/'predictions/gsm8k.jsonl').write_text('{}'); "
        "(out/'metrics.json').write_text('{}'); (out/'seen').write_text(str(weka)); "
        + RECORD
        + "sys.exit(143)"
    )
    assert run_shared(root, run_uuid, results, command) == 143
    assert (results / "seen").read_text() == str(directory)
    assert sorted(str(p.relative_to(results)) for p in results.rglob("*") if p.is_file()) == [
        "metrics.json",
        "predictions/gsm8k.jsonl",
        "seen",
    ]
    assert sorted(p.name for p in directory.iterdir()) == ["progress", "run.lock", "session.json"]
    assert json.loads((directory / "session.json").read_text())["progress_streams"] == [
        "progress/model/run.segments"
    ]


def test_completion_removes_recovery_state_and_keeps_beaker_results(shared):
    root, run_uuid, results, directory = shared
    command = child(
        "(out/'metrics.json').write_text('{}'); "
        "(weka/'identity').mkdir(); (weka/'identity/manifest.json').write_text('{}'); " + RECORD
    )
    assert run_shared(root, run_uuid, results, command) == 0
    assert (results / "metrics.json").read_text() == "{}"
    assert sorted(p.name for p in directory.iterdir()) == [
        "complete.json",
        "identity",
        "run.lock",
        "session.json",
    ]


def test_identity_names_match_the_upload_module():
    from olmo_eval.common.constants.infrastructure import SHARED_RUN_IDENTITY_FILES
    from olmo_eval.upload.manifest import MANIFEST_NAME, RUN_SECRET_NAME

    assert SHARED_RUN_IDENTITY_FILES == (MANIFEST_NAME, RUN_SECRET_NAME)


@pytest.mark.parametrize("code", [0, 1, 143])
def test_identity_created_late_is_saved_when_the_attempt_exits(shared, code):
    root, run_uuid, results, directory = shared
    command = child(
        '(out/\'manifest.json\').write_text(\'{"run": {"run_id": "r1"}}\'); '
        f"(out/'.upload-secret').write_text('s3cret'); sys.exit({code})"
    )
    assert run_shared(root, run_uuid, results, command) == code
    assert (directory / "identity/.upload-secret").read_text() == "s3cret"
    assert json.loads((directory / "identity/manifest.json").read_text())["run"]["run_id"] == "r1"


FINISH = (
    "import json; p=pathlib.Path(os.environ['RUNS']); "
    "p.write_text(str(int(p.read_text())+1) if p.exists() else '1'); "
    "(out/'metrics.json').write_text('{}'); "
    "(out/'manifest.json').write_text(json.dumps({'run': {'run_id': 'r1', 'status': 'complete'}}))"
)


@pytest.mark.parametrize("carried_over", [True, False])
def test_rerun_of_a_completed_job_runs_again_only_without_its_results(
    shared, tmp_path, monkeypatch, carried_over
):
    root, run_uuid, results, directory = shared
    runs = tmp_path / "runs"
    monkeypatch.setenv("RUNS", str(runs))
    command = child(FINISH)
    assert run_shared(root, run_uuid, results, command) == 0
    rerun = tmp_path / "rerun-results"
    if carried_over:
        shutil.copytree(results, rerun)
    assert run_shared(root, run_uuid, rerun, command) == 0
    assert runs.read_text() == ("1" if carried_over else "2")
    assert (rerun / "metrics.json").is_file()
    assert (directory / "complete.json").is_file()
    assert (directory / "identity/manifest.json").is_file()


def test_rerun_of_a_completed_job_ignores_a_stale_failed_snapshot(shared, tmp_path, monkeypatch):
    root, run_uuid, results, directory = shared
    runs = tmp_path / "runs"
    monkeypatch.setenv("RUNS", str(runs))
    command = child(FINISH)
    assert run_shared(root, run_uuid, results, command) == 0
    # Beaker carried over a snapshot from an earlier attempt that tripped the budget.
    stale = tmp_path / "stale-results"
    stale.mkdir()
    (stale / "metrics.json").write_text("{}")
    (stale / "manifest.json").write_text(json.dumps({"run": {"run_id": "r1", "status": "failed"}}))
    assert run_shared(root, run_uuid, stale, command) == 0
    assert runs.read_text() == "2"


@pytest.mark.parametrize("missing", ["progress", "stream"])
def test_retry_refuses_disappearance_of_initialized_recovery_directories(shared, missing):
    root, run_uuid, results, directory = shared
    command = child(RECORD + "sys.exit(143)")
    assert run_shared(root, run_uuid, results, command) == 143
    paths = {
        "progress": directory / "progress",
        "stream": directory / "progress/model/run.segments",
    }
    shutil.rmtree(paths[missing])
    with pytest.raises(RuntimeError, match="recovery .*missing"):
        run_shared(root, run_uuid, results, command)


def test_retention_refresh_raises_when_a_subtree_cannot_be_scanned(tmp_path, monkeypatch):
    directory = tmp_path / "progress"
    directory.mkdir()
    segment = directory / "old.jsonl"
    segment.write_text("checkpoint")
    os.utime(segment, (1, 1))
    scandir = os.scandir

    def fail(path):
        if Path(path) == directory:
            raise OSError(errno.EIO, "volume directory unavailable")
        return scandir(path)

    monkeypatch.setattr(os, "scandir", fail)
    with pytest.raises(OSError, match="volume directory unavailable"):
        refresh_files(tmp_path)
    assert segment.stat().st_mtime == 1


def test_retention_refresh_tolerates_atomic_publication_after_enumeration(tmp_path, monkeypatch):
    temporary = tmp_path / "index.tmp"
    published = tmp_path / "index.json"
    temporary.write_text("inventory")
    original_stat = Path.stat

    def rename_before_stat(path, *args, **kwargs):
        if path == temporary:
            temporary.replace(published)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", rename_before_stat)
    refresh_files(tmp_path)
    assert published.read_text() == "inventory"


def test_retention_traversal_failure_stops_the_live_child(shared, monkeypatch):
    root, run_uuid, results, directory = shared
    command = child("(weka/'progress/model').mkdir(parents=True); import time; time.sleep(0.5)")
    scandir = os.scandir

    def fail(path):
        if Path(path) == directory / "progress/model":
            raise OSError(errno.EIO, "volume directory unavailable")
        return scandir(path)

    monkeypatch.setattr(os, "scandir", fail)
    with pytest.raises(RuntimeError, match="retention refresh failed"):
        run_shared(root, run_uuid, results, command, refresh_seconds=0.01)
    assert not (directory / "complete.json").exists()


def test_independent_uuids_do_not_overwrite_and_completed_run_is_not_repeated(shared):
    root, run_uuid, results, directory = shared
    command = child(
        "import json; p=weka/'count'; "
        "p.write_text(str(int(p.read_text())+1) if p.exists() else '1'); "
        "(out/'metrics.json').write_text('{}'); "
        "run={'run_id': 'r', 'status': 'complete'}; "
        "(out/'manifest.json').write_text(json.dumps({'run': run}))"
    )
    assert run_shared(root, run_uuid, results, command) == 0
    replacement = results.parent / "replacement-results"
    shutil.copytree(results, replacement)
    assert run_shared(root, run_uuid, replacement, command) == 0
    assert (directory / "count").read_text() == "1"
    another = str(uuid.uuid4())
    assert run_shared(root, another, results.parent / "other-results", command) == 0
    assert (root / "tmp-7d" / another / "count").read_text() == "1"


def test_a_live_writer_cannot_be_overwritten(shared):
    root, run_uuid, results, directory = shared
    directory.mkdir(parents=True)
    with (directory / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="Another attempt owns"):
            run_shared(root, run_uuid, results, child("raise AssertionError('must not run')"))
    assert not (directory / "session.json").exists()
    assert not results.exists() or not any(results.iterdir())


def test_failed_child_retains_files_and_resumes_after_stale_progress_lock(shared):
    root, run_uuid, results, directory = shared
    command = child(
        "p=weka/'attempt'; prior=p.exists(); p.write_text('saved'); "
        "lock=weka/'progress/model/.lock'; lock.parent.mkdir(parents=True, exist_ok=True); "
        "assert not lock.exists(); lock.write_text('old-node'); sys.exit(0 if prior else 143)"
    )
    assert run_shared(root, run_uuid, results, command) == 143
    assert not (directory / "complete.json").exists()
    assert run_shared(root, run_uuid, results, command) == 0
    assert not (directory / "progress").exists()


@pytest.mark.parametrize("damage", ["expired", "session", "command", "completion"])
def test_expired_missing_or_changed_state_is_refused(shared, damage):
    root, run_uuid, results, directory = shared
    command = child("(out/'prediction').write_text('result')")
    run_shared(root, run_uuid, results, command)
    if damage == "expired":
        old = time.time() - RETENTION_SECONDS - 1
        os.utime(directory / "session.json", (old, old))
    elif damage == "session":
        (directory / "session.json").unlink()
    elif damage == "completion":
        (directory / "complete.json").unlink()
    else:
        command = child("raise AssertionError('changed')")
    with pytest.raises(RuntimeError):
        run_shared(root, run_uuid, results, command)


def test_an_interrupted_run_whose_session_was_lost_is_refused(shared):
    root, run_uuid, results, directory = shared
    command = child(RECORD + "sys.exit(143)")
    assert run_shared(root, run_uuid, results, command) == 143
    (directory / "session.json").unlink()
    with pytest.raises(RuntimeError, match="session is missing"):
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
            yaml.safe_dump(
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
        while not (results / "ready").exists():
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
        while not (results / "pid").exists():
            assert supervisor.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        child_pid = int((results / "pid").read_text())
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


@pytest.mark.parametrize("days", [1, 14])
def test_custom_retention_selects_directory_and_expiry(shared, days):
    root, run_uuid, results, _ = shared
    command = child("(out/'prediction').write_text('result')")
    assert run_shared(root, run_uuid, results, command, retention_days=days) == 0
    directory = root / f"tmp-{days}d" / run_uuid
    assert (directory / "complete.json").is_file()
    assert (results / "prediction").read_text() == "result"
    # A replacement of the completed run does not run the eval again.
    assert run_shared(root, run_uuid, results, command, retention_days=days) == 0
    old = time.time() - days * 86400 - 1
    os.utime(directory / "session.json", (old, old))
    with pytest.raises(RuntimeError, match="expired"):
        run_shared(root, run_uuid, results, command, retention_days=days)


@pytest.mark.parametrize("days", [0, -1, True, 1.5])
def test_invalid_retention_cannot_start_child(shared, days):
    root, run_uuid, results, _ = shared
    with pytest.raises(ValueError, match="positive integer"):
        run_shared(
            root, run_uuid, results, child("raise AssertionError('started')"), retention_days=days
        )


@pytest.mark.parametrize(
    ("yaml_options", "cli_options", "volume", "days"),
    [
        ("", {}, "ai2/olmo-eval", 7),
        ("resume_volume: ai2/custom\nresume_retention_days: 14\n", {}, "ai2/custom", 14),
        (
            "resume_volume: ai2/custom\nresume_retention_days: 14\n",
            {"resume_volume": "ai2/other", "resume_retention_days": 3},
            "ai2/other",
            3,
        ),
    ],
)
def test_automatic_volume_and_retention_yaml_cli_precedence(
    tmp_path, monkeypatch, yaml_options, cli_options, volume, days
):
    import importlib

    from olmo_eval.cli.beaker.config_loader import LaunchConfigLoader
    from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
    from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler

    path = tmp_path / "eval.yaml"
    path.write_text(
        "name: defaults\nmodels: [mock]\ntasks: [gsm8k]\ngpus: 0\n"
        "cluster: h100\nworkspace: ai2/test\nbudget: ai2/test\n" + yaml_options
    )
    config = LaunchConfigLoader(str(path), {"upload": False, **cli_options}).load()
    assert config.resume_volume == volume
    assert config.resume_retention_days == days
    monkeypatch.setattr(
        importlib.import_module("olmo_eval.cli.beaker.job_assembler"),
        "cluster_has_weka",
        lambda _: False,
    )
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
    plan = ExperimentPlan(
        name="defaults",
        model_spec="mock",
        priority="normal",
        tasks=["gsm8k"],
        original_task_specs=["gsm8k"],
        total_expanded_tasks=1,
        num_gpus=0,
    )
    job = assembler.assemble(plan)
    assert job.resume_volume == volume
    assert job.command[job.command.index("--volume") + 1] == volume
    assert job.command[job.command.index("--retention-days") + 1] == str(days)
    assert "--resume" in job.command


@pytest.mark.parametrize("value", ["0", "-2", "true", "1.5"])
def test_invalid_yaml_retention_rejected_before_launch(tmp_path, value):
    from olmo_eval.cli.beaker.config_loader import LaunchConfigLoader

    path = tmp_path / "eval.yaml"
    path.write_text(
        "models: [mock]\ntasks: [gsm8k]\ncluster: h100\nworkspace: ai2/test\n"
        "budget: ai2/test\nresume_retention_days: " + value + "\n"
    )
    with pytest.raises((SystemExit, ValueError)):
        LaunchConfigLoader(str(path), {}).load()
