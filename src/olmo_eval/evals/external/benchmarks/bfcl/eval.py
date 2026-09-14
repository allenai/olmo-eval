"""BFCL v4 non-web external evaluation."""

from __future__ import annotations

import json
import logging
import shlex
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from olmo_eval.evals.external.base import SandboxedExternalEval
from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

BFCL_REPOSITORY = "https://github.com/ShishirPatil/gorilla.git"
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_ENCODER_REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"
BFCL_ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BFCL_SCORER_PATCH_SHA256 = "5ff242c9273854b9c5ca70b039f7d0bbdc58126e9ce4cb5caf23c65279ee1ccf"

_RUNNER_CONTAINER_PATH = "/opt/olmo-eval/bfcl_runner.py"
_SCORER_PATCH_CONTAINER_PATH = "/opt/olmo-eval/0001-check-multi-turn-irrelevance.patch"
_ARTIFACT_CONTAINER_PATH = "/output"


@dataclass(frozen=True)
class BFCLV4Args:
    """Runtime arguments for the official BFCL generator."""

    temperature: float = 0.001
    num_threads: int = 100

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BFCLV4Args:
        temperature = float(data.get("temperature", 0.001))
        num_threads = int(data.get("num_threads", 100))
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if num_threads < 1:
            raise ValueError("num_threads must be positive")
        return cls(temperature=temperature, num_threads=num_threads)


class BFCLV4NonWebExternalEval(SandboxedExternalEval):
    """Official BFCL v4 excluding the credentialed web-search slice.

    The returned primary metric is the contribution of these categories to the
    official fixed-weight BFCL v4 score. Its maximum is 0.8 because web search,
    which contributes the remaining 0.2, is deliberately not run here.
    """

    @property
    def name(self) -> str:
        return "bfcl_v4_non_web"

    @property
    def description(self) -> str:
        return (
            "Runs the official BFCL v4 non-live, live, multi-turn, and memory suites "
            "against an OpenAI-compatible endpoint. Web search is excluded."
        )

    @property
    def sandbox_image(self) -> str:
        return "ghcr.io/astral-sh/uv:python3.11-bookworm"

    @property
    def working_dir(self) -> str:
        return "/workspace"

    @property
    def timeout_seconds(self) -> float:
        return 8 * 60 * 60

    @property
    def setup_command(self) -> tuple[str, ...]:
        gorilla_dir = f"{self.working_dir}/gorilla"
        bfcl_dir = f"{gorilla_dir}/berkeley-function-call-leaderboard"
        python = f"{bfcl_dir}/.venv/bin/python"
        return (
            f"git clone --filter=blob:none {BFCL_REPOSITORY} {gorilla_dir}",
            (
                f"cd {gorilla_dir} && git checkout --detach {BFCL_COMMIT} && "
                f'test "$(git rev-parse HEAD)" = {BFCL_COMMIT}'
            ),
            (
                f"echo '{BFCL_SCORER_PATCH_SHA256}  {_SCORER_PATCH_CONTAINER_PATH}' | "
                f"sha256sum -c - && cd {gorilla_dir} && "
                f"git apply --check {_SCORER_PATCH_CONTAINER_PATH} && "
                f"git apply {_SCORER_PATCH_CONTAINER_PATH}"
            ),
            (
                f"cd {bfcl_dir} && uv venv --python 3.11 && "
                "uv pip install --python .venv/bin/python --torch-backend cpu -e . soundfile"
            ),
            (
                f"{python} {_RUNNER_CONTAINER_PATH} prepare "
                f"--cache-dir {self.working_dir}/hf-cache "
                f"--encoder-repository {BFCL_ENCODER_REPOSITORY} "
                f"--encoder-revision {BFCL_ENCODER_REVISION}"
            ),
        )

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "temperature": ("Sampling temperature used by the official BFCL generator", 0.001),
            "num_threads": ("Maximum concurrent model requests", 100),
        }

    @property
    def run_command(self) -> str:
        return "bfcl_runner.py run --provider-model MODEL --provider-url URL --suite non_web"

    @property
    def _suite(self) -> str:
        return "non_web"

    def _parse_args(self, args: dict[str, Any]) -> BFCLV4Args:
        return BFCLV4Args.from_dict(args)

    def _create_sensitive_volumes(self, stack: ExitStack) -> tuple[tuple[str, str], ...]:
        return ()

    def _extra_run_command_parts(self, bfcl_args: BFCLV4Args) -> list[str]:
        return ["--encoder-revision", BFCL_ENCODER_REVISION]

    def _create_bfcl_sandbox_config(
        self,
        container_runtime: str,
        output_dir: str | None,
        artifact_dir: Path,
        extra_volumes: tuple[tuple[str, str], ...] = (),
    ) -> Any:
        config = self._create_sandbox_config(container_runtime, output_dir)
        env = dict(config.environment)
        env["HF_HOME"] = f"{self.working_dir}/hf-cache"
        env["HF_HUB_CACHE"] = f"{self.working_dir}/hf-cache/hub"

        # This adapter targets the local OpenAI-compatible model endpoint that
        # olmo-eval launches. The OpenAI client requires a non-empty value even
        # though the endpoint does not authenticate. Never forward an ambient
        # credential into the benchmark container.
        env["OPENAI_API_KEY"] = "EMPTY"

        runner_path = Path(__file__).with_name("runner.py")
        scorer_patch_path = (
            Path(__file__).with_name("patches") / "0001-check-multi-turn-irrelevance.patch"
        )
        volumes = (
            config.volumes
            + (
                (str(runner_path), _RUNNER_CONTAINER_PATH),
                (str(scorer_patch_path), _SCORER_PATCH_CONTAINER_PATH),
                (str(artifact_dir), _ARTIFACT_CONTAINER_PATH),
            )
            + extra_volumes
        )
        return replace(
            config,
            environment=tuple(env.items()),
            volumes=volumes,
            inject_swerex=True,
            # Docker's json-file logger does not support the `path` log option
            # used by the shared sandbox logger. The BFCL runner's raw output
            # and artifacts are persisted separately.
            log_dir=None if container_runtime == "docker" else config.log_dir,
        )

    def _build_run_command(
        self,
        model_name: str,
        provider_url: str,
        bfcl_args: BFCLV4Args,
    ) -> str:
        bfcl_dir = f"{self.working_dir}/gorilla/berkeley-function-call-leaderboard"
        python = f"{bfcl_dir}/.venv/bin/python"
        parts = [
            python,
            _RUNNER_CONTAINER_PATH,
            "run",
            "--bfcl-root",
            bfcl_dir,
            "--output-dir",
            _ARTIFACT_CONTAINER_PATH,
            "--provider-model",
            model_name,
            "--provider-url",
            provider_url,
            "--temperature",
            str(bfcl_args.temperature),
            "--num-threads",
            str(bfcl_args.num_threads),
            "--suite",
            self._suite,
            "--bfcl-commit",
            BFCL_COMMIT,
            "--scorer-patch-sha256",
            BFCL_SCORER_PATCH_SHA256,
            *self._extra_run_command_parts(bfcl_args),
        ]
        return " ".join(shlex.quote(part) for part in parts)

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        start_time = time.time()
        all_output: list[str] = []

        try:
            bfcl_args = self._parse_args(args)
        except (TypeError, ValueError) as error:
            return self._error_result(str(error), start_time)

        provider_url = getattr(provider, "base_url", None)
        if not provider_url:
            return self._error_result(
                "BFCL v4 requires a provider exposing an OpenAI-compatible base_url",
                start_time,
            )
        model_name = provider.model_name
        is_local = self._is_local_provider(provider, provider_url)

        try:
            from olmo_eval.harness.sandbox.executor import SandboxExecutor
        except ImportError as error:
            return self._error_result(f"SWE-ReX not installed: {error}", start_time)

        output_root = Path(output_dir) if output_dir else None
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            artifact_dir = Path(tempfile.mkdtemp(prefix=f"{self.name}_artifacts_", dir=output_root))
            temporary_dir = None
        else:
            temporary_dir = tempfile.TemporaryDirectory(prefix=f"{self.name}_")
            artifact_dir = Path(temporary_dir.name)

        result: ExternalEvalResult
        try:
            with ExitStack() as stack:
                extra_volumes = self._create_sensitive_volumes(stack)
                sandbox_config = self._create_bfcl_sandbox_config(
                    container_runtime,
                    output_dir,
                    artifact_dir,
                    extra_volumes=extra_volumes,
                )
                async with SandboxExecutor(sandbox_config, name=self.name) as executor:
                    if setup_error := await self._run_setup(executor, all_output, start_time):
                        return setup_error

                    sandbox_url = self._get_provider_url_for_sandbox(provider_url)
                    if is_local and not await self._check_provider_health(executor, sandbox_url):
                        return self._error_result(
                            f"Provider not reachable at {sandbox_url}",
                            start_time,
                            "\n".join(all_output),
                        )

                    run_command = self._build_run_command(model_name, sandbox_url, bfcl_args)
                    logger.info(f"[{self.name}] Running BFCL v4 {self._suite} suite")
                    run_result = await executor.execute_command(
                        run_command,
                        timeout=self.timeout_seconds,
                        stream=True,
                        log_prefix=self.name,
                    )
                    all_output.append(f"$ {run_command}\n{run_result.output}")

                    summary_path = artifact_dir / "summary.json"
                    if not run_result.success:
                        return self._error_result(
                            f"BFCL runner exited with code {run_result.exit_code}",
                            start_time,
                            "\n".join(all_output),
                        )
                    if not summary_path.is_file():
                        return self._error_result(
                            "BFCL runner completed without summary.json",
                            start_time,
                            "\n".join(all_output),
                        )

                    summary = json.loads(summary_path.read_text())
                    result = ExternalEvalResult(
                        name=self.name,
                        metrics=summary["metrics"],
                        metadata={
                            **summary["metadata"],
                            "artifacts_dir": str(artifact_dir) if output_dir else None,
                        },
                        predictions=summary.get("predictions"),
                        raw_output="\n".join(all_output),
                    )
        except Exception as error:
            logger.exception(f"[{self.name}] Execution failed")
            return self._error_result(str(error), start_time, "\n".join(all_output))
        finally:
            if temporary_dir is not None:
                temporary_dir.cleanup()

        result.duration_seconds = time.time() - start_time
        if output_dir:
            self._save_results(result, output_dir)
        return result
