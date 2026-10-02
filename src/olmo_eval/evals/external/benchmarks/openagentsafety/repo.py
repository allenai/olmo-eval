"""Clone helper for the OpenHands/benchmarks OpenAgentSafety runner."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from olmo_eval.evals.external.network import PODMAN_DNS, podman_pasta_network

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/OpenHands/benchmarks.git"
# Pin the OpenHands/benchmarks revision that this wrapper was developed against.
DEFAULT_REF = "405bae7140d7e961a75f4910a0b2e7069731db96"
CACHE_PARENT_NAME = "olmo-eval-openhands-benchmarks"
# OAS Dockerfile COPYs benchmarks/vendor/software-agent-sdk and sets the
# Docker context to the parent of the git checkout, so the clone must be
# named "benchmarks".
REPO_DIR_NAME = "benchmarks"
DEFAULT_AGENT_IMAGE = "openagentsafety-agent-server"


def default_cache_dir() -> Path:
    """Checkout cache under ``TMPDIR`` (or the platform temp dir)."""
    return Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / CACHE_PARENT_NAME / REPO_DIR_NAME


def ensure_repo(target_dir: Path, ref: str | None = None) -> Path:
    """Clone or update OpenHands/benchmarks at ``ref``, including the SDK submodule."""
    ref = ref or DEFAULT_REF
    target_dir = target_dir.expanduser().resolve()
    _migrate_flat_checkout(target_dir)

    if (target_dir / ".git").exists():
        logger.info("Updating OpenHands/benchmarks at %s", target_dir)
        _run(["git", "-C", str(target_dir), "fetch", "origin"])
        _run(["git", "-C", str(target_dir), "checkout", ref])
    elif target_dir.exists() and any(target_dir.iterdir()):
        logger.info("Using existing OpenHands/benchmarks checkout at %s", target_dir)
    else:
        logger.info("Cloning OpenHands/benchmarks to %s", target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", REPO_URL, str(target_dir)])
        _run(["git", "-C", str(target_dir), "checkout", ref])

    if (target_dir / ".gitmodules").exists():
        logger.info("Initializing OpenHands Agent SDK submodule in %s", target_dir)
        _run(["git", "-C", str(target_dir), "submodule", "update", "--init", "--recursive"])
    _write_build_dockerignore(docker_build_context(target_dir))
    return target_dir


def sync_repo_env(repo_dir: Path) -> None:
    """Install OpenHands/benchmarks into its own uv environment."""
    logger.info("Running uv sync in %s", repo_dir)
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    _run(["uv", "sync"], cwd=repo_dir, env=env)


def docker_build_context(repo_dir: Path) -> Path:
    """Parent directory OAS uses as the Docker build context."""
    repo_dir = repo_dir.expanduser().resolve()
    if repo_dir.name != REPO_DIR_NAME:
        raise RuntimeError(
            "OpenAgentSafety image builds require the OpenHands/benchmarks "
            f"checkout to be named '{REPO_DIR_NAME}' (got {repo_dir}). "
            f"Clone into .../{REPO_DIR_NAME} or omit repo_path."
        )
    return repo_dir.parent


_GATEWAY_IP_ASSIGNMENT = re.compile(r'(gateway_ip\s*=\s*)(["\'])([^"\']*)(\2)')
_PASTA_NETWORK_ARG = re.compile(r'network="pasta:--map-guest-addr,[^"]+"')
_WORKSPACE_FORWARD_ENV = "forward_env=forward_env or [],\n        )"
_NETWORK_FLAG_BLOCK = """        if self.network:
            flags += ["--network", self.network]
"""
_PASTA_DNS_BLOCK = f"""        if self.network:
            flags += ["--network", self.network]
            if str(self.network).startswith("pasta:"):
                flags += ["--dns", "{PODMAN_DNS}"]
"""
RUN_INFER_RELATIVE = Path("benchmarks") / "openagentsafety" / "run_infer.py"
_RUN_CONTAINER_ANCHOR = "        # Run container\n"
_HOOK_MOUNT_BLOCK = """        hook_dir = os.environ.get("OAS_AGENT_HOOK_DIR")
        if hook_dir:
            flags += [
                "-v",
                f"{hook_dir}:/opt/oas-hooks:ro",
                "-e",
                "PYTHONPATH=/opt/oas-hooks:/utils:",
            ]

"""
WORKSPACE_RELATIVE = (
    Path("vendor")
    / "software-agent-sdk"
    / "openhands-workspace"
    / "openhands"
    / "workspace"
    / "docker"
    / "workspace.py"
)


def patch_gateway_host_mapping(repo_dir: Path, gateway_ip: str) -> Path:
    """Point ``setup_host_mapping`` at a workspace-reachable gateway IP.

    Upstream hardcodes ``172.17.0.1`` (docker0). Beaker pasta jobs need
    ``169.254.1.2``. The patch is idempotent and only touches the working tree.
    """
    path = repo_dir.expanduser().resolve() / RUN_INFER_RELATIVE
    if not path.is_file():
        raise RuntimeError(f"OAS run_infer.py not found at {path}")
    text = path.read_text()
    already = re.search(
        rf'gateway_ip\s*=\s*["\']{re.escape(gateway_ip)}["\']',
        text,
    )
    if already:
        logger.info("OAS gateway mapping already set to %s in %s", gateway_ip, path)
        return path
    new_text, count = _GATEWAY_IP_ASSIGNMENT.subn(
        rf"\g<1>\g<2>{gateway_ip}\g<4>",
        text,
        count=1,
    )
    if count == 0:
        raise RuntimeError(f"Could not find gateway_ip assignment in {path}")
    path.write_text(new_text)
    logger.info("Patched OAS gateway_ip to %s in %s", gateway_ip, path)
    return path


def patch_agent_hook_mount(repo_dir: Path) -> None:
    """Mount the OAS agent hooks into each workspace container.

    The agent server imports ``sitecustomize`` from that directory. The mount
    is read-only and does not change the image tag.
    """
    path = repo_dir.expanduser().resolve() / WORKSPACE_RELATIVE
    if not path.is_file():
        raise RuntimeError(f"DockerWorkspace source not found at {path}")
    text = path.read_text()
    if "OAS_AGENT_HOOK_DIR" in text:
        logger.info("OAS agent hook mount already present in %s", path)
        return
    if _RUN_CONTAINER_ANCHOR not in text:
        raise RuntimeError(f"Could not find Docker run command in {path}")
    path.write_text(
        text.replace(_RUN_CONTAINER_ANCHOR, _HOOK_MOUNT_BLOCK + _RUN_CONTAINER_ANCHOR, 1)
    )
    logger.info("Patched OAS agent hook mount in %s", path)


def patch_podman_workspace_network(repo_dir: Path, gateway_ip: str) -> None:
    """Give OAS workspaces a pasta network so published ports survive host netns.

    Beaker's Podman config uses ``netns=host``, which TheAgentCompany needs in
    order to bind the job's service ports. ``DockerWorkspace`` publishes a host
    port and then connects to it, so the workspace itself must use pasta.
    TheAgentCompany containers are left on the default host network.
    """
    repo_dir = repo_dir.expanduser().resolve()
    network = podman_pasta_network(gateway_ip)
    _patch_workspace_network_arg(repo_dir / RUN_INFER_RELATIVE, network)
    _patch_workspace_dns(repo_dir / WORKSPACE_RELATIVE)


def _patch_workspace_network_arg(path: Path, network: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"OAS run_infer.py not found at {path}")
    desired = f'network="{network}",'
    text = path.read_text()
    if desired in text:
        logger.info("OAS workspace network already set to %s", network)
        return
    if _PASTA_NETWORK_ARG.search(text):
        path.write_text(_PASTA_NETWORK_ARG.sub(desired, text, count=1))
        logger.info("Updated OAS workspace network to %s", network)
        return
    if _WORKSPACE_FORWARD_ENV not in text:
        raise RuntimeError(f"Could not find DockerWorkspace constructor in {path}")
    replacement = f"forward_env=forward_env or [],\n            {desired}\n        )"
    path.write_text(text.replace(_WORKSPACE_FORWARD_ENV, replacement, 1))
    logger.info("Patched OAS workspace network to %s", network)


def _patch_workspace_dns(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"DockerWorkspace source not found at {path}")
    text = path.read_text()
    if _PASTA_DNS_BLOCK in text:
        logger.info("OAS workspace DNS patch already present in %s", path)
        return
    if _NETWORK_FLAG_BLOCK not in text:
        raise RuntimeError(f"Could not find Docker network flags in {path}")
    path.write_text(text.replace(_NETWORK_FLAG_BLOCK, _PASTA_DNS_BLOCK, 1))
    logger.info("Patched OAS workspace DNS flags in %s", path)


def ensure_workspace_image(repo_dir: Path, container_runtime: str = "podman") -> str:
    """Build ``openagentsafety-agent-server:<sdk-sha>`` if it is not already local."""
    image_name = workspace_image_name(repo_dir)
    if _docker_image_exists(image_name):
        logger.info("Using existing OAS workspace image %s", image_name)
        return image_name

    dockerfile = repo_dir / "benchmarks" / "openagentsafety" / "Dockerfile"
    context = docker_build_context(repo_dir)
    vendor = context / REPO_DIR_NAME / "vendor" / "software-agent-sdk"
    if not vendor.exists():
        raise RuntimeError(f"Vendor SDK not found at {vendor}")
    if not dockerfile.is_file():
        raise RuntimeError(f"OAS Dockerfile not found at {dockerfile}")

    build_cmd = _workspace_build_command(
        dockerfile, image_name, context, container_runtime=container_runtime
    )
    logger.info(
        "Building OAS workspace image %s from %s (first build takes several minutes)",
        image_name,
        context,
    )
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    _run(build_cmd, env=env, capture=False)
    if not _docker_image_exists(image_name):
        raise RuntimeError(f"Built image {image_name} is not present in local Docker/Podman")
    logger.info("Built OAS workspace image %s", image_name)
    return image_name


def _has_buildx(cli: str) -> bool:
    try:
        result = subprocess.run(
            [cli, "buildx", "version"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _workspace_build_command(
    dockerfile: Path,
    image_name: str,
    context: Path,
    container_runtime: str = "podman",
) -> list[str]:
    """Prefer ``docker buildx``; fall back to ``docker``/``podman`` ``build``."""
    docker = shutil.which("docker")
    buildx_cli = docker or container_runtime
    if _has_buildx(buildx_cli):
        return [
            buildx_cli,
            "buildx",
            "build",
            "--file",
            str(dockerfile),
            "--tag",
            image_name,
            "--platform",
            "linux/amd64",
            "--load",
            str(context),
        ]
    if container_runtime == "podman" and shutil.which("podman"):
        runtime_cli = "podman"
    else:
        runtime_cli = docker or container_runtime
    return [
        runtime_cli,
        "build",
        "--file",
        str(dockerfile),
        "--tag",
        image_name,
        "--platform",
        "linux/amd64",
        str(context),
    ]


def workspace_image_name(repo_dir: Path) -> str:
    image = os.environ.get("EVAL_AGENT_SERVER_IMAGE", DEFAULT_AGENT_IMAGE)
    prefix = os.environ.get("IMAGE_TAG_PREFIX")
    if prefix:
        return f"{image}:{prefix}-openagentsafety"
    return f"{image}:{_sdk_short_hash(repo_dir)}"


def _sdk_short_hash(repo_dir: Path) -> str:
    sdk_dir = repo_dir / "vendor" / "software-agent-sdk"
    if not sdk_dir.exists():
        raise RuntimeError(f"Vendor SDK not found at {sdk_dir}")
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(sdk_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to read SDK commit: {(result.stderr or '').strip()}")
    return result.stdout.strip()


def _docker_image_exists(image_name: str) -> bool:
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _migrate_flat_checkout(target_dir: Path) -> None:
    """Move a git checkout that lived at the cache parent into ``.../benchmarks``."""
    if target_dir.name != REPO_DIR_NAME:
        return
    parent = target_dir.parent
    if (target_dir / ".git").exists() or not (parent / ".git").exists():
        return
    logger.info("Moving OpenHands/benchmarks checkout into %s for OAS image builds", target_dir)
    staging = parent.with_name(f"{parent.name}.migrating")
    if staging.exists():
        raise RuntimeError(f"Refusing to migrate checkout; leftover path exists: {staging}")
    parent.rename(staging)
    parent.mkdir(parents=True)
    staging.rename(target_dir)


def _write_build_dockerignore(context_dir: Path) -> None:
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / ".dockerignore").write_text(
        "\n".join(
            [
                "benchmarks/.venv",
                "benchmarks/.git",
                "benchmarks/**/__pycache__",
                "benchmarks/.pytest_cache",
                "benchmarks/.ruff_cache",
                "",
            ]
        )
    )


def _run(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = True,
) -> None:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        capture_output=capture,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Command failed ({' '.join(command)}): {stderr}")
