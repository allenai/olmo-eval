"""TheAgentCompany service checks and optional Beaker startup."""

from __future__ import annotations

import fcntl
import logging
import os
import socket
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

OAS_START_TAC_ENV = "OAS_START_TAC"
OAS_TAC_IMAGE_STORE_ENV = "OAS_TAC_IMAGE_STORE"
OAS_TAC_IMAGE_STORE = "/weka/oe-eval-default/olmo-eval/openagentsafety-tac"
TAC_SETUP_URL = (
    "https://github.com/TheAgentCompany/the-agent-company-backup-data/releases/download/"
    "setup-script-20241208/setup.sh"
)
TAC_PORTS = (3000, 8091, 8092, 8929)
_READY_NAME = ".oas-tac-ready"
_SERVICES_DOWN = (
    "TheAgentCompany services are not reachable on ports 3000 (Rocket.Chat), "
    "8091 (Plane), 8092 (ownCloud), and 8929 (GitLab). Start them on the host "
    "before OpenAgentSafety."
)
_COMPOSE_MISSING = "docker compose or podman compose is required before starting TheAgentCompany."


def tac_services_ready(host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    """Return whether every TheAgentCompany service port accepts a connection."""
    for port in TAC_PORTS:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                pass
        except OSError:
            return False
    return True


def compose_available() -> bool:
    """Return whether ``docker compose`` or ``podman compose`` can run."""
    for command in (["docker", "compose", "version"], ["podman", "compose", "version"]):
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return True
    return False


def ensure_tac_services() -> None:
    """Start TheAgentCompany when requested and the service ports are closed.

    Healthy ports are left alone, including a host that already runs the stack.
    ``OAS_START_TAC=1`` runs the upstream setup script. A shared image store, when
    set, is locked so only one job populates it; later jobs read that store.
    """
    if tac_services_ready():
        logger.info("TheAgentCompany services are already reachable")
        return
    if os.environ.get(OAS_START_TAC_ENV) != "1":
        raise RuntimeError(_SERVICES_DOWN)
    if not compose_available():
        raise RuntimeError(_COMPOSE_MISSING)

    store = os.environ.get(OAS_TAC_IMAGE_STORE_ENV)
    if store:
        _start_with_image_store(Path(store))
        return
    _run_tac_setup()


def _image_graphroot(store: Path) -> Path:
    """Image layers live beside the lock file so Podman owns an empty directory."""
    graphroot = store / "images"
    graphroot.mkdir(parents=True, exist_ok=True)
    return graphroot


def _start_with_image_store(store: Path) -> None:
    store.mkdir(parents=True, exist_ok=True)
    ready = store / _READY_NAME
    graphroot = _image_graphroot(store)
    with _store_lock(store):
        if ready.is_file():
            _use_additional_image_store(graphroot)
            run_setup = True
        else:
            _use_store_as_graphroot(graphroot)
            _run_tac_setup()
            ready.write_text("ok\n")
            run_setup = False
    if run_setup:
        _run_tac_setup()


def _use_store_as_graphroot(graphroot: Path) -> None:
    conf = _storage_conf_path()
    _write_storage_conf(conf, graphroot=graphroot, additional=None)
    os.environ["CONTAINERS_STORAGE_CONF"] = str(conf)
    logger.info("TheAgentCompany image store graphroot is %s", graphroot)


def _use_additional_image_store(graphroot: Path) -> None:
    local_root = Path(tempfile.gettempdir()) / "olmo-eval-oas-graphroot"
    local_root.mkdir(parents=True, exist_ok=True)
    conf = _storage_conf_path()
    _write_storage_conf(conf, graphroot=local_root, additional=graphroot)
    os.environ["CONTAINERS_STORAGE_CONF"] = str(conf)
    logger.info("Reusing TheAgentCompany images from %s", graphroot)


def _storage_conf_path() -> Path:
    conf_dir = Path(tempfile.gettempdir()) / "olmo-eval-oas-storage"
    conf_dir.mkdir(parents=True, exist_ok=True)
    return conf_dir / "containers-storage.conf"


def _write_storage_conf(path: Path, graphroot: Path, additional: Path | None) -> None:
    lines = [
        "[storage]",
        'driver = "overlay"',
        f'graphroot = "{graphroot}"',
        f'runroot = "{graphroot}/runroot"',
        "",
    ]
    if additional is not None:
        lines.extend(
            [
                "[storage.options]",
                "additionalimagestores = [",
                f'  "{additional}"',
                "]",
                "",
            ]
        )
    path.write_text("\n".join(lines))


@contextmanager
def _store_lock(store: Path) -> Iterator[None]:
    lock_path = store / ".lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_tac_setup() -> None:
    logger.info("Starting TheAgentCompany from %s", TAC_SETUP_URL)
    result = subprocess.run(
        ["bash", "-lc", f"curl -fsSL {TAC_SETUP_URL} | sh"],
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"TheAgentCompany setup script exited with {result.returncode}")
