"""Start the retrieval server alongside an evaluation run.

Setting ``BROWSECOMP_PLUS_RETRIEVER`` asks the evaluation process to launch the
bundled server from the prepared retrieval environment the first time a tool
needs it, so a single job can serve both the model and the retriever.
"""

from __future__ import annotations

import asyncio
import atexit
import fcntl
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from olmo_eval.evals.tasks.browsecomp_plus.data import RETRIEVERS, artifact_home

logger = logging.getLogger(__name__)

DEFAULT_SERVER_URL = "http://127.0.0.1:8081"
RETRIEVER_ENV = "BROWSECOMP_PLUS_RETRIEVER"
STARTUP_TIMEOUT_SECONDS = 3600
POLL_SECONDS = 5

_ready = False
_startup_lock: asyncio.Lock | None = None
_process: subprocess.Popen[bytes] | None = None
_lock_fd: int | None = None


def server_url() -> str:
    """Return the base URL of the retrieval server used by the tools."""
    return os.environ.get("BROWSECOMP_PLUS_SEARCH_SERVER_URL", DEFAULT_SERVER_URL).rstrip("/")


def server_command(retriever: str, home: Path, port: int) -> list[str]:
    """Build the command that serves a prepared index from the retrieval environment."""
    python = home / "retrieval-venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(
            f"No retrieval environment at {python}; run retrieval/install.sh with "
            "BROWSECOMP_PLUS_HOME set to this directory"
        )
    return [
        str(python),
        str(Path(__file__).resolve().parent / "server.py"),
        "--home",
        str(home),
        "--retriever",
        retriever,
        "--get-document",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


async def _is_serving(url: str) -> bool:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            return (await client.get(f"{url}/openapi.json")).status_code == 200
    except httpx.HTTPError:
        return False


def _stop() -> None:
    if _process is not None and _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _process.kill()


def _log_tail(path: Path, lines: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


async def ensure_retriever() -> None:
    """Make sure the requested retrieval server is up, starting it if nobody has.

    Does nothing unless ``BROWSECOMP_PLUS_RETRIEVER`` names a retriever. A server
    that already answers at the configured URL is reused, and concurrent
    processes on one machine agree on a single owner for the port.
    """
    global _ready, _startup_lock, _process, _lock_fd

    retriever = os.environ.get(RETRIEVER_ENV)
    if not retriever or _ready:
        return
    if retriever not in RETRIEVERS:
        raise ValueError(f"{RETRIEVER_ENV} must be one of {RETRIEVERS}, got {retriever!r}")

    if _startup_lock is None:
        _startup_lock = asyncio.Lock()
    async with _startup_lock:
        if _ready:
            return
        url = server_url()
        if await _is_serving(url):
            _ready = True
            return

        port = urlsplit(url).port or 80
        log_path: Path | None = None
        lock_path = Path(tempfile.gettempdir()) / f"browsecomp-plus-retriever-{port}.lock"
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(lock_fd)
            logger.info(
                "Waiting for the %s retrieval server another process is starting", retriever
            )
        else:
            _lock_fd = lock_fd
            home = artifact_home().resolve()
            log_dir = Path(os.environ.get("OLMO_RESULT_DIR") or tempfile.gettempdir())
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"browsecomp_plus_retriever_{retriever}.log"
            command = server_command(retriever, home, port)
            logger.info("Starting the %s retrieval server; logs at %s", retriever, log_path)
            with log_path.open("ab") as log:
                _process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            atexit.register(_stop)

        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while not await _is_serving(url):
            if _process is not None and _process.poll() is not None:
                tail = _log_tail(log_path) if log_path else ""
                raise RuntimeError(
                    f"The {retriever} retrieval server exited with code "
                    f"{_process.returncode} during startup:\n{tail}"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(f"The {retriever} retrieval server did not start at {url}")
            await asyncio.sleep(POLL_SECONDS)
        logger.info("The %s retrieval server is ready at %s", retriever, url)
        _ready = True
