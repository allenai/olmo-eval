"""Credentialed BFCL v4 web-search external evaluation."""

from __future__ import annotations

import os
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olmo_eval.evals.external.benchmarks.bfcl.eval import (
    BFCLV4Args,
    BFCLV4NonWebExternalEval,
)

_SERPAPI_SECRET_CONTAINER_PATH = "/run/secrets/serpapi_api_key"
_DEFAULT_MINIMUM_SERPAPI_CREDITS = 628


@dataclass(frozen=True)
class BFCLV4WebArgs(BFCLV4Args):
    """Runtime arguments for the credentialed official web-search suite."""

    num_threads: int = 4
    minimum_serpapi_credits: int = _DEFAULT_MINIMUM_SERPAPI_CREDITS

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BFCLV4WebArgs:
        common = BFCLV4Args.from_dict(
            {
                "temperature": data.get("temperature", 0.001),
                "num_threads": data.get("num_threads", 4),
            }
        )
        minimum_credits = int(data.get("minimum_serpapi_credits", _DEFAULT_MINIMUM_SERPAPI_CREDITS))
        if minimum_credits < 0:
            raise ValueError("minimum_serpapi_credits must be non-negative")
        return cls(
            temperature=common.temperature,
            num_threads=common.num_threads,
            minimum_serpapi_credits=minimum_credits,
        )


class BFCLV4WebExternalEval(BFCLV4NonWebExternalEval):
    """Official BFCL v4 web-search categories with audited SerpAPI requests.

    This is intentionally separate from the non-web suite: it consumes a paid,
    live external service and can fail for reasons unrelated to the evaluated
    model. Its fixed-weight contribution has a maximum of 0.2.
    """

    @property
    def name(self) -> str:
        return "bfcl_v4_web"

    @property
    def description(self) -> str:
        return (
            "Runs the official BFCL v4 web-search and no-snippet categories against "
            "an OpenAI-compatible endpoint, with quota preflight and audited SerpAPI calls."
        )

    @property
    def required_secrets(self) -> tuple[str, ...]:
        return ("SERPAPI_API_KEY",)

    @property
    def setup_command(self) -> tuple[str, ...]:
        # Web search does not use the sentence encoder required by memory_vector.
        return super().setup_command[:-1]

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "temperature": ("Sampling temperature used by the official BFCL generator", 0.001),
            "num_threads": ("Maximum concurrent model requests", 4),
            "minimum_serpapi_credits": (
                "Minimum account balance required before a complete web run",
                _DEFAULT_MINIMUM_SERPAPI_CREDITS,
            ),
        }

    @property
    def run_command(self) -> str:
        return "bfcl_runner.py run --provider-model MODEL --provider-url URL --suite web"

    @property
    def _suite(self) -> str:
        return "web"

    def _parse_args(self, args: dict[str, Any]) -> BFCLV4WebArgs:
        return BFCLV4WebArgs.from_dict(args)

    def _build_env_vars(self, secrets: tuple[str, ...] | None = None) -> dict[str, str]:
        # required_secrets makes orchestration inject the key into the outer
        # process. It must not become a Docker `-e KEY=value` argument, which is
        # observable in process listings and sandbox debug logs.
        return {}

    def _create_sensitive_volumes(self, stack: ExitStack) -> tuple[tuple[str, str], ...]:
        secret = os.environ.get("SERPAPI_API_KEY")
        if not secret:
            raise ValueError("Missing required secret: SERPAPI_API_KEY")

        secret_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="bfcl_secret_")))
        secret_dir.chmod(0o700)
        secret_path = secret_dir / "serpapi_api_key"
        descriptor = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as file_handle:
            file_handle.write(secret)
        if secret_path.stat().st_mode & 0o077:
            raise ValueError("SerpAPI secret file permissions are too broad")
        return ((str(secret_path), _SERPAPI_SECRET_CONTAINER_PATH),)

    def _extra_run_command_parts(self, bfcl_args: BFCLV4Args) -> list[str]:
        if not isinstance(bfcl_args, BFCLV4WebArgs):
            raise TypeError("BFCL web suite requires BFCLV4WebArgs")
        return [
            "--serpapi-secret-file",
            _SERPAPI_SECRET_CONTAINER_PATH,
            "--minimum-serpapi-credits",
            str(bfcl_args.minimum_serpapi_credits),
        ]
