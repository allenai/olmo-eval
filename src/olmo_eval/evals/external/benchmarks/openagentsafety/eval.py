"""OpenAgentSafety external evaluation.

OpenAgentSafety evaluates agent safety in workplace scenarios with NPC
interactions on TheAgentCompany infrastructure. The upstream runner owns
Docker workspace creation, NPC config, and the OpenHands agent loop, so this
eval invokes that CLI on the host rather than nesting a second sandbox.

Source: https://github.com/OpenHands/benchmarks/tree/main/benchmarks/openagentsafety
Dataset: https://huggingface.co/datasets/mgulavani/openagentsafety_full_updated_v3
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.benchmarks.openagentsafety.args import OpenAgentSafetyArgs
from olmo_eval.evals.external.benchmarks.openagentsafety.repo import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REF,
    ensure_repo,
)
from olmo_eval.evals.external.benchmarks.openagentsafety.result_parser import (
    find_output_jsonl,
    parse_output_jsonl,
)
from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)


class OpenAgentSafetyExternalEval(ExternalEval):
    """OpenAgentSafety evaluation via the OpenHands/benchmarks infer CLI."""

    @property
    def name(self) -> str:
        return "openagentsafety"

    @property
    def description(self) -> str:
        return (
            "Evaluates AI agent safety in workplace scenarios with NPC interactions. "
            "Requires Docker, TheAgentCompany services (~30GB), NPC_API_KEY, and "
            "optionally NPC_BASE_URL / NPC_MODEL."
        )

    @property
    def timeout_seconds(self) -> float:
        return 43200.0

    @property
    def required_secrets(self) -> tuple[str, ...]:
        return ("NPC_API_KEY",)

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "dataset": ("HuggingFace dataset identifier", OpenAgentSafetyArgs.dataset),
            "split": ("Dataset split", OpenAgentSafetyArgs.split),
            "n_limit": ("Limit number of instances (omit for all)", None),
            "num_workers": ("Parallel OAS workers", 1),
            "critic": ("OAS critic: pass, finish_with_patch, empty_patch_critic", "pass"),
            "select": ("Instance-ID file path or comma-separated IDs", None),
            "prompt_path": ("Custom Jinja2 prompt template path", None),
            "repo_path": ("Local OpenHands/benchmarks checkout (default: clone)", None),
            "repo_ref": ("Git ref of OpenHands/benchmarks to checkout", DEFAULT_REF),
            "max_iterations": ("Max agent iterations per instance", 500),
            "n_critic_runs": ("Critic retry attempts", 1),
            "max_retries": ("Retries for instances that throw", 3),
            "tool_preset": ("OpenHands tool preset", "default"),
            "enable_delegation": ("Enable OpenHands sub-agent delegation", False),
            "note": ("OAS eval note used in the output directory name", "olmo-eval"),
        }

    @property
    def run_command(self) -> str:
        return (
            "uv run olmo-eval run-external -e openagentsafety --model <model> "
            "-a dataset=mgulavani/openagentsafety_full_updated_v3 -a n_limit=1 -a critic=pass"
        )

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        logger.debug(
            "[%s] OAS uses host Docker via OpenHands DockerWorkspace; "
            "ignoring container_runtime=%s",
            self.name,
            container_runtime,
        )
        start_time = time.time()
        oas_args = OpenAgentSafetyArgs.from_dict(args)
        run_dir = Path(output_dir) if output_dir else Path("/tmp") / "olmo-eval-openagentsafety"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._build_env_vars()
        except ValueError as exc:
            return self._error_result(str(exc), start_time)

        docker_error = self._docker_error()
        if docker_error:
            return self._error_result(docker_error, start_time)

        try:
            repo_dir = ensure_repo(
                Path(oas_args.repo_path) if oas_args.repo_path else DEFAULT_CACHE_DIR,
                oas_args.repo_ref,
            )
        except Exception as exc:
            logger.exception("[%s] Failed to prepare OpenHands/benchmarks", self.name)
            return self._error_result(f"Failed to prepare OpenHands/benchmarks: {exc}", start_time)

        llm_config_path = run_dir / "oas_llm_config.json"
        llm_config = self._build_llm_config(provider)
        llm_config_path.write_text(json.dumps(llm_config, indent=2))

        select_path = self._materialize_select(oas_args.select, run_dir)
        try:
            prompt_restore = self._apply_prompt_override(repo_dir, oas_args.prompt_path)
        except FileNotFoundError as exc:
            return self._error_result(str(exc), start_time)

        infer_cmd = self._build_infer_command(
            repo_dir=repo_dir,
            llm_config_path=llm_config_path,
            oas_args=oas_args,
            output_dir=run_dir,
            select_path=select_path,
        )
        logger.info("[%s] Running: %s", self.name, " ".join(infer_cmd))

        try:
            returncode, raw_output = await self._run_infer(infer_cmd, repo_dir)
        except TimeoutError:
            return self._error_result(
                f"OpenAgentSafety exceeded timeout of {self.timeout_seconds:.0f}s",
                start_time,
            )
        except Exception as exc:
            logger.exception("[%s] Infer process failed", self.name)
            return self._error_result(str(exc), start_time)
        finally:
            if prompt_restore is not None:
                prompt_path, original = prompt_restore
                prompt_path.write_text(original)

        jsonl_path = find_output_jsonl(run_dir)
        if jsonl_path is None:
            return self._error_result(
                "No output.jsonl produced by OpenAgentSafety infer",
                start_time,
                raw_output,
            )

        parsed = parse_output_jsonl(jsonl_path)
        result = ExternalEvalResult(
            name=self.name,
            success=returncode == 0 and bool(parsed["metrics"].get("num_instances", 0)),
            metrics=parsed["metrics"],
            metadata={
                **parsed["metadata"],
                "model_name": provider.model_name,
                "dataset": oas_args.dataset,
                "split": oas_args.split,
                "critic": oas_args.critic,
                "llm_config": {k: v for k, v in llm_config.items() if k != "api_key"},
                "output_jsonl": str(jsonl_path),
                "repo_dir": str(repo_dir),
                "exit_code": returncode,
            },
            predictions=parsed["predictions"],
            raw_output=raw_output,
            error=None if returncode == 0 else f"openagentsafety-infer exited with {returncode}",
        )
        result.duration_seconds = time.time() - start_time
        if output_dir:
            self._save_results(result, output_dir)
        return result

    def _build_llm_config(self, provider: InferenceProvider) -> dict[str, str]:
        """Build the JSON LLM config consumed by ``openagentsafety-infer``."""
        inner = getattr(provider, "base_provider", provider)
        base_url = getattr(inner, "base_url", None) or getattr(provider, "base_url", None)
        model_name = getattr(inner, "model_name", None) or provider.model_name
        api_key = "dummy"
        is_local = self._is_local_provider(inner, str(base_url or ""))

        try:
            client = inner.get_openai_client()
        except Exception:
            client = None
        if client is not None:
            client_key = getattr(client, "api_key", None)
            if client_key:
                api_key = str(client_key)
            client_url = getattr(client, "base_url", None)
            if client_url:
                base_url = str(client_url)

        if not is_local:
            env_key = (
                os.environ.get("OPENAI_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("LITELLM_PROXY_API_KEY")
            )
            if env_key:
                api_key = env_key

        config: dict[str, str] = {
            "model": f"hosted_vllm/{model_name}" if is_local else str(model_name),
            "api_key": api_key,
        }
        if base_url:
            config["base_url"] = str(base_url)
        return config

    def _build_infer_command(
        self,
        repo_dir: Path,
        llm_config_path: Path,
        oas_args: OpenAgentSafetyArgs,
        output_dir: Path,
        select_path: str | None,
    ) -> list[str]:
        """Build the ``uv run openagentsafety-infer`` command."""
        del repo_dir
        cmd = [
            "uv",
            "run",
            "openagentsafety-infer",
            str(llm_config_path),
            "--dataset",
            oas_args.dataset,
            "--split",
            oas_args.split,
            "--output-dir",
            str(output_dir),
            "--num-workers",
            str(oas_args.num_workers),
            "--critic",
            oas_args.critic,
            "--max-iterations",
            str(oas_args.max_iterations),
            "--n-critic-runs",
            str(oas_args.n_critic_runs),
            "--max-retries",
            str(oas_args.max_retries),
            "--tool-preset",
            oas_args.tool_preset,
            "--workspace",
            "docker",
            "--note",
            oas_args.note,
        ]
        if oas_args.n_limit:
            cmd.extend(["--n-limit", str(oas_args.n_limit)])
        if select_path:
            cmd.extend(["--select", select_path])
        if oas_args.enable_delegation:
            cmd.append("--enable-delegation")
        return cmd

    def _materialize_select(self, select: str | list[str] | None, run_dir: Path) -> str | None:
        if select is None:
            return None
        if isinstance(select, list):
            select_file = run_dir / "select_instances.txt"
            select_file.write_text("\n".join(select) + "\n")
            return str(select_file)
        return select

    def _apply_prompt_override(
        self, repo_dir: Path, prompt_path: str | None
    ) -> tuple[Path, str] | None:
        """Copy a custom prompt over OAS default.j2, returning restore data."""
        if not prompt_path:
            return None
        source = Path(prompt_path)
        if not source.is_file():
            raise FileNotFoundError(f"prompt_path does not exist: {prompt_path}")
        destination = repo_dir / "benchmarks" / "openagentsafety" / "prompts" / "default.j2"
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = destination.read_text() if destination.exists() else ""
        shutil.copyfile(source, destination)
        logger.info("[%s] Using custom prompt %s", self.name, source)
        return destination, original

    def _docker_error(self) -> str | None:
        docker = shutil.which("docker")
        if docker is None:
            return (
                "OpenAgentSafety requires Docker on the host (TheAgentCompany + "
                "per-task OpenHands workspaces). Install Docker and retry."
            )
        try:
            result = subprocess.run(
                [docker, "info"],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"Could not query Docker: {exc}"
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            return f"Docker is installed but not usable: {detail}"
        return None

    async def _run_infer(self, command: list[str], repo_dir: Path) -> tuple[int, str]:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(repo_dir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        chunks: list[str] = []
        try:
            assert process.stdout is not None
            async with asyncio.timeout(self.timeout_seconds):
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    text = line.decode(errors="replace")
                    chunks.append(text)
                    logger.info("[%s] %s", self.name, text.rstrip())
                await process.wait()
        except TimeoutError:
            process.kill()
            await process.wait()
            raise
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        return process.returncode or 0, "".join(chunks)
