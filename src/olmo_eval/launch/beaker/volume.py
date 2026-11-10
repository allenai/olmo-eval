"""Native volume submission while Gantry's mount API supports legacy WEKA only."""

from __future__ import annotations

import contextlib
import io
import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import yaml

if TYPE_CHECKING:
    from collections.abc import Callable

VOLUME_MOUNT = "/eval-cache"


class VolumeClaimError(RuntimeError):
    """A workspace cannot mount the resume volume with read-write access."""


class GantryLaunchKwargs(TypedDict):
    """The Gantry options shared by native-volume and legacy submissions."""

    args: list[str]
    name: str
    description: str | None
    workspace: str
    group_names: list[str] | None
    clusters: list[str]
    gpus: int
    shared_memory: str
    priority: str
    preemptible: bool | None
    min_runtime: str | None
    task_timeout: str | None
    retries: int | None
    budget: str | None
    beaker_image: str
    weka: list[tuple[str, str]] | None
    gh_token_secret: str
    env_vars: list[tuple[str, str]] | None
    env_secrets: list[tuple[str, str]] | None
    google_credentials_secret: str | None
    mounts: list[tuple[str, str]] | None
    results: str
    install: str
    no_python: bool
    dry_run: bool
    timeout: int
    yes: bool


def validate_volume(volume: str) -> str:
    """Accept a qualified volume name or a Volume Service ID."""
    import re

    if not re.fullmatch(r"(?:[\w-]+/[\w.-]+|v_[A-Za-z0-9]+)", volume):
        raise ValueError("resume_volume must be <org>/<name> or a volume ID (v_...)")
    return volume


def _cli_durations(value: Any) -> None:
    # beaker-py emits durations as integer nanoseconds; the native CLI parses strings.
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"timeout", "minRuntime", "synchronizedStartTimeout"} and isinstance(
                item, int
            ):
                value[key] = f"{item}ns"
            else:
                _cli_durations(item)
    elif isinstance(value, list):
        for item in value:
            _cli_durations(item)


def launch_with_volume(
    launch: Callable[..., Any],
    kwargs: GantryLaunchKwargs,
    volume: str,
    *,
    dry_run: bool,
) -> str | None:
    """Prepare with Gantry, then submit the native spec with the Beaker CLI.

    Returns the experiment ID. The saved spec stays private and is removed afterward;
    credentials remain secret references. Submission failures are never retried here.
    """
    validate_volume(volume)
    with tempfile.TemporaryDirectory(prefix="olmo-eval-volume-") as directory:
        path = Path(directory) / "spec.json"
        with contextlib.redirect_stdout(io.StringIO()):
            launch(**{**kwargs, "dry_run": True, "save_spec": str(path)})
        spec = yaml.safe_load(path.read_text())
        _cli_durations(spec)
        for task in spec["tasks"]:
            mounts = task.setdefault("datasets", [])
            if any(m["mountPath"] == VOLUME_MOUNT for m in mounts):
                raise ValueError(f"Mount path {VOLUME_MOUNT} is already in use")
            mounts.append(
                {
                    "mountPath": VOLUME_MOUNT,
                    "source": {"volume": volume},
                    "mode": "readwrite",
                }
            )
        path.write_text(json.dumps(spec))
        if dry_run:
            from olmo_eval.launch.beaker.launcher import print_experiment_config

            print_experiment_config(spec, name=kwargs["name"])
            return None
        try:
            result = subprocess.run(
                [
                    "beaker",
                    "experiment",
                    "create",
                    str(path),
                    "--workspace",
                    kwargs["workspace"],
                    "--name",
                    kwargs["name"],
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError("Native volumes require a Beaker CLI with volume support") from e
        except subprocess.CalledProcessError as e:
            error = e.stderr.strip()
            if (
                "has no claim on volume" in error
                or "claim is read-only; cannot mount with read-write mode" in error
            ):
                claim = shlex.join(
                    [
                        "beaker",
                        "volume",
                        "claim",
                        "create",
                        "--workspace",
                        kwargs["workspace"],
                        "--volume",
                        volume,
                        "--mode",
                        "readwrite",
                    ]
                )
                raise VolumeClaimError(
                    f"Evaluation was not submitted: workspace {kwargs['workspace']!r} needs a "
                    f"read-write claim on resume volume {volume!r}.\n\n"
                    "Ask someone with write access to the workspace and the volume's parent "
                    "workspace to grant access:\n"
                    f"  {claim}\n"
                    "For an existing read-only claim, an administrator must replace it with "
                    "a read-write claim.\n\n"
                    "Alternatively, rerun your launch with --no-resume to run without a "
                    "resume-volume claim. Interrupted or retried attempts will start over.\n\n"
                    f"Beaker reported: {error}"
                ) from e
            raise RuntimeError(f"Native volume submission failed: {error}") from e
        experiments = json.loads(result.stdout)
        return experiments[0]["id"]
