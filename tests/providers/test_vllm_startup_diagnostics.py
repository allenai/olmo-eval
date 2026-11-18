"""Exercise startup reporting and exit detection with real local child processes."""

import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib import error

import pytest

from olmo_eval.inference import manager
from olmo_eval.inference.providers import config
from olmo_eval.inference.providers import vllm_server_utils as servers


@pytest.fixture
def launch(monkeypatch, tmp_path):
    children = []

    def start(script, *, timeout=3, log_dir=True):
        monkeypatch.setattr(
            servers, "_build_server_command", lambda **kwargs: [sys.executable, "-u", "-c", script]
        )
        server = servers.VLLMServerProcess(
            "test-model", startup_timeout=timeout, log_dir=str(tmp_path) if log_dir else None
        )
        children.append(server)
        return server

    yield start
    for server in children:
        server.stop()


@pytest.mark.parametrize("model", ["allenai/wildguard", "google/gemma-2b"])
def test_shared_default_and_override_reach_lifecycle_only(model):
    for kwargs, expected in [({}, 1200), ({"startup_timeout": 42}, 42)]:
        provider = config.ProviderConfig(kind="vllm_server", model=model, kwargs=kwargs)
        server = manager._create_server(provider, gpu_ids=[0])
        assert server.startup_timeout == expected
        assert "startup_timeout" not in server.server_kwargs
        command = servers._build_server_command(model, server.port, **server.server_kwargs)
        assert not any("startup-timeout" in part or "startup_timeout" in part for part in command)


def test_reports_observed_phases_and_models_fallback(launch, monkeypatch, caplog):
    server = launch("""
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
print('compilation_config={"mode": 3}')
time.sleep(.2)
print('Downloading model.safetensors: 1%')
time.sleep(.2)
print('Loading safetensors checkpoint shards: 0%')
time.sleep(.2)
print('Compiling graph for batch size 1')
time.sleep(.2)
print('Graph capturing finished')
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == '/v1/models' else 503)
        self.end_headers()
HTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
""")
    command = servers._build_server_command()
    command[-1] = command[-1].replace("PORT", str(server.port))
    monkeypatch.setattr(servers, "_build_server_command", lambda **kwargs: command)
    messages = []
    with caplog.at_level("INFO", logger=servers.__name__):
        assert server.start(messages.append) == server.base_url
    text = "\n".join(messages)
    for phase in [
        "no phase reported",
        "downloading",
        "loading weights",
        "compiling /",
        "waiting for readiness",
        "ready after",
    ]:
        assert phase in text
    assert "[vllm-startup]" in caplog.text
    assert "log:" in text
    assert "Graph capturing finished" in server._read_log_output()


@pytest.mark.parametrize("log_dir", [True, False])
def test_crash_interrupts_blocked_probe_and_preserves_verbose_output(launch, monkeypatch, log_dir):
    blocked = threading.Event()
    release = threading.Event()

    def probe(*args, **kwargs):
        blocked.set()
        release.wait(5)
        raise error.URLError("blocked readiness probe")

    monkeypatch.setattr(servers.request, "urlopen", probe)
    server = launch(
        """
import sys, time
print('x' * 1000000)
print('Autotuning kernels')
time.sleep(.25)
print('fatal: initialization failed')
sys.exit(7)
""",
        log_dir=log_dir,
    )
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="Process exited with code 7") as exc:
            server.start()
        assert blocked.is_set()
        assert time.monotonic() - started < 1.5
        assert "fatal: initialization failed" in str(exc.value)
        assert "compiling /" in str(exc.value)
        assert "Startup deadline exceeded" not in str(exc.value)
        assert server._log_path.stat().st_size > 1000000
        assert server._process is None
    finally:
        release.set()


def test_deadline_remains_bounded_and_does_not_invent_phase(launch, monkeypatch):
    release = threading.Event()

    def probe(*args, **kwargs):
        release.wait(5)
        raise error.URLError("not ready")

    monkeypatch.setattr(servers.request, "urlopen", probe)
    server = launch("import time; print('initialization'); time.sleep(30)", timeout=0.25)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="Startup deadline exceeded") as exc:
            server.start()
        assert time.monotonic() - started < 1.5
        assert "no phase reported" in str(exc.value)
        assert server._process is None
        assert "initialization" in server._log_path.read_text()
    finally:
        release.set()


def test_dead_process_never_probes_http(monkeypatch):
    def probe(*args, **kwargs):
        pytest.fail("a dead child must be detected before probing")

    monkeypatch.setattr(servers.request, "urlopen", probe)
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(9)"])
    process.wait(timeout=5)
    success, failure, _ = servers._wait_for_server("http://localhost:1/v1", process=process)
    assert not success
    assert "exited with code 9" in str(failure)


def test_split_log_lines_and_unrelated_config(tmp_path):
    path = tmp_path / "judge.log"
    path.write_text('compilation_config={"mode": 3}\nDownloa')
    status = servers._StartupLog(path)
    assert "no phase reported" in status.update()
    with path.open("a") as stream:
        stream.write("ding model: 1%\r")
    assert status.update() == "downloading"
    with path.open("a") as stream:
        stream.write("Loading safetensors checkpoint shards: 1%\r")
    assert status.update() == "loading weights"


def test_error_tail_does_not_read_another_servers_log(tmp_path):
    server = servers.VLLMServerProcess("test", log_dir=str(tmp_path))
    (tmp_path / "vllm_server_other.log").write_text("unrelated OOM")
    server._log_path = Path(tmp_path / "vllm_server_current.log")
    server._log_path.write_text("current failure")
    assert server._read_log_output() == "current failure"
