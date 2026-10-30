"""Exercise the container-side supervisor on compatible Linux hosts."""

import ast
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from olmo_eval.harness.sandbox._local_process import COMMAND_SUPERVISOR

linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision")


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 3
    while not path.exists():
        assert time.monotonic() < deadline, f"Child did not create {path}"
        time.sleep(0.025)


@linux_only
@pytest.mark.parametrize("interrupt", ["timeout", "cancel"])
@pytest.mark.parametrize("missing_children", [False, True])
def test_supervisor_reaps_escaped_descendants(tmp_path, interrupt, missing_children):
    record = tmp_path / "supervisor"
    child_file = tmp_path / "child"
    code = (
        "import os,signal,time\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "if os.fork() == 0:\n"
        "    os.setsid()\n"
        "    if os.fork(): os._exit(0)\n"
        f"    open({str(child_file)!r},'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            _without_children_file(COMMAND_SUPERVISOR) if missing_children else COMMAND_SUPERVISOR,
            str(record),
            "__OLMO_STARTED_test__",
            "0.2" if interrupt == "timeout" else "60",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process.stdin.write(json.dumps(["python3", "-c", code]))
    process.stdin.close()
    process.stdin = None
    try:
        _wait_for(child_file)
        pid = int(child_file.read_text())
        if interrupt == "cancel":
            os.kill(process.pid, signal.SIGUSR1)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode in (124, 137), stderr
        assert Path(str(record) + ".clean").exists()
        assert not Path(f"/proc/{pid}").exists()
        report = json.loads(stdout.rsplit("__OLMO_FINISHED_test__:", 1)[1])
        assert report["exit_code"] == process.returncode
        assert report["timed_out"] is (interrupt == "timeout")
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        if child_file.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(child_file.read_text()), signal.SIGKILL)


@linux_only
def test_supervisor_isolates_imports_and_reports_ordinary_exit(tmp_path):
    (tmp_path / "subprocess.py").write_text("raise RuntimeError('workspace import')")
    record = tmp_path / "supervisor"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            COMMAND_SUPERVISOR,
            str(record),
            "__OLMO_STARTED_test__",
            "5",
        ],
        input=json.dumps(["python3", "-c", "print('ok'); raise SystemExit(124)"]),
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 124, result.stderr
    assert result.stdout.startswith("__OLMO_STARTED_test__\nok\n")
    assert Path(str(record) + ".clean").exists()
    report = json.loads(result.stdout.rsplit("__OLMO_FINISHED_test__:", 1)[1])
    assert report == {"exit_code": 124, "timed_out": False}


def _without_children_file(supervisor: str) -> str:
    return (
        "from pathlib import Path\n"
        "_read_text = Path.read_text\n"
        "def read_text(self, *args, **kwargs):\n"
        "    if self.name == 'children': raise FileNotFoundError(str(self))\n"
        "    return _read_text(self, *args, **kwargs)\n"
        "Path.read_text = read_text\n" + supervisor
    )


@pytest.fixture
def proc_children(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    tree = ast.parse(COMMAND_SUPERVISOR)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "direct_children"
    )
    namespace = {
        "os": SimpleNamespace(getpid=lambda: 900),
        "Path": lambda path: proc if path == "/proc" else tmp_path / "children",
        "time": time,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<supervisor>", "exec"), namespace)
    return proc, namespace["direct_children"]


def test_child_enumeration_preserves_the_children_file_fast_path(proc_children):
    proc, children = proc_children
    (proc.parent / "children").write_text("101 202 ")
    assert children(time.monotonic() + 2) == [101, 202]


def test_child_enumeration_without_children_file_selects_only_owned_children(proc_children):
    proc, children = proc_children
    for pid, parent, name in ((101, 900, "a ) tricky ( name)"), (202, 901, "unrelated")):
        entry = proc / str(pid)
        entry.mkdir()
        (entry / "stat").write_text(f"{pid} ({name}) S {parent} 0 0 0")
    (proc / "303").mkdir()  # A process exits between listing /proc and reading its stat.
    (proc / "self").mkdir()
    assert children(time.monotonic() + 2) == [101]


@pytest.mark.parametrize("error", [PermissionError, ProcessLookupError])
def test_child_enumeration_handles_proc_read_errors(proc_children, monkeypatch, error):
    proc, children = proc_children
    entry = proc / "101"
    entry.mkdir()
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == entry / "stat":
            raise error(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    if error is ProcessLookupError:
        assert children(time.monotonic() + 2) == []
    else:
        with pytest.raises(PermissionError):
            children(time.monotonic() + 2)


def test_child_enumeration_rejects_malformed_stat(proc_children):
    proc, children = proc_children
    entry = proc / "101"
    entry.mkdir()
    (entry / "stat").write_text("unparseable")
    with pytest.raises(IndexError):
        children(time.monotonic() + 2)


def test_child_enumeration_has_a_deadline(proc_children):
    proc, children = proc_children
    (proc / "101").mkdir()
    with pytest.raises(RuntimeError, match="enumeration exceeded cleanup deadline"):
        children(time.monotonic() - 1)
