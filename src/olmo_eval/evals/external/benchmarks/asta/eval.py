"""ASTA-bench external evaluation implementation."""

from __future__ import annotations

import hashlib
import json
import logging
import shlex
import subprocess
import time
from typing import TYPE_CHECKING, Any

from olmo_eval.common.config import get_infra_config
from olmo_eval.evals.external.base import SandboxedExternalEval
from olmo_eval.evals.external.benchmarks.asta.args import ASTA_TASKS, AstaArgs
from olmo_eval.evals.external.benchmarks.asta.result_parser import (
    aggregate_metrics,
    parse_inspect_log,
)
from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox.executor import SandboxExecutor
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

ASTA_IMAGE_VERSION = "20260223.2"
ASTA_BENCH_VERSION = "v0.3.1"


def _get_asta_image(container_runtime: str = "docker") -> str:
    """Get or build the ASTA-bench container image.

    Checks local cache first, then registry, then builds locally.
    """
    config = get_infra_config()
    registry = config.swerex_registry

    hash_input = f"asta-bench:{ASTA_BENCH_VERSION}:{ASTA_IMAGE_VERSION}"
    tag_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    local_image = f"asta-bench-{tag_hash}:latest"

    result = subprocess.run(
        [container_runtime, "image", "inspect", local_image],
        capture_output=True,
    )
    if result.returncode == 0:
        logger.info(f"Using cached ASTA image: {local_image}")
        return local_image

    logger.debug(f"Local image {local_image} not found, checking registry...")

    if registry:
        registry_image = f"{registry}/asta-bench-{tag_hash}:latest"
        result = subprocess.run(
            [container_runtime, "pull", registry_image],
            capture_output=True,
        )
        if result.returncode == 0:
            subprocess.run(
                [container_runtime, "tag", registry_image, local_image],
                capture_output=True,
            )
            logger.info(f"Pulled ASTA image from registry: {registry_image}")
            return local_image
        logger.debug(f"Registry pull failed for {registry_image}")

    logger.info("Building ASTA image locally...")

    dockerfile = f"""\
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \\
    git curl ca-certificates build-essential && \\
    rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
WORKDIR /workspace
RUN git clone --recursive --branch {ASTA_BENCH_VERSION} \\
    https://github.com/allenai/asta-bench.git
WORKDIR /workspace/asta-bench
RUN uv sync
RUN uv pip install swe-rex
RUN mkdir -p /workspace/asta-bench/results
ENV PATH="/workspace/asta-bench/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV INSPECT_SANDBOX=local
ENV INSPECT_EVAL_SANDBOX=local
"""

    result = subprocess.run(
        [container_runtime, "build", "-t", local_image, "-"],
        input=dockerfile.encode(),
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode() if result.stderr else ""
        raise RuntimeError(f"Failed to build ASTA image: {stderr}")

    logger.info(f"Built ASTA image: {local_image}")

    if registry:
        registry_image = f"{registry}/asta-bench-{tag_hash}:latest"
        logger.info(f"Pushing ASTA image to registry: {registry_image}")
        tag_cmd = [container_runtime, "tag", local_image, registry_image]
        subprocess.run(tag_cmd, capture_output=True)
        push_cmd = [container_runtime, "push", registry_image]
        push_result = subprocess.run(push_cmd, capture_output=True)
        if push_result.returncode == 0:
            logger.info(f"Pushed ASTA image to registry: {registry_image}")
        else:
            stderr = push_result.stderr.decode() if push_result.stderr else ""
            logger.warning(f"Failed to push to registry (using local image): {stderr}")

    return local_image


class AstaExternalEval(SandboxedExternalEval):
    """ASTA-bench evaluation for AI scientist tasks."""

    @property
    def name(self) -> str:
        return "asta_bench"

    @property
    def description(self) -> str:
        return (
            "Evaluates LLM agents on AI scientist tasks including literature search, "
            "code execution, data analysis, and end-to-end discovery. Uses Inspect AI harness."
        )

    @property
    def sandbox_image(self) -> str:
        return "(built locally)"

    @property
    def working_dir(self) -> str:
        return "/workspace/asta-bench"

    @property
    def timeout_seconds(self) -> float:
        return 14400.0  # 4 hours

    @property
    def setup_command(self) -> tuple[str, ...]:
        return (
            f"cd {self.working_dir} && uv sync",
            f"mkdir -p {self.results_dir}",
        )

    @property
    def required_secrets(self) -> tuple[str, ...]:
        # GOOGLE_API_KEY is required because asta-bench scorers hardcode Google models
        return (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "ASTA_TOOL_KEY",
            "HF_TOKEN",
        )

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        # Build task list description from ASTA_TASKS
        task_descriptions = []
        for category, tasks in ASTA_TASKS.items():
            task_descriptions.append(f"{category}: {', '.join(tasks)}")
        tasks_help = (
            "Comma-separated task names to run (default: all). "
            f"Available: {'; '.join(task_descriptions)}"
        )
        return {
            "split": ("Dataset split: 'validation' or 'test'", "validation"),
            "tasks": (tasks_help, None),
            "limit": ("Maximum problems per task", None),
            "solver": ("Agent solver type: 'react' or 'basic'", "react"),
            "max_samples": ("Max concurrent problems", 1),
            "max_sandboxes": ("Max parallel sandboxes", 1),
            "max_connections": ("Max model API connections", 8),
            "sandbox_type": ("Sandbox type: 'local' (Beaker) or 'docker'", "local"),
            "temperature": ("Temperature for agent responses", None),
            "max_tokens": ("Max tokens for agent responses", None),
            "extra_args": ("Extra args to pass to inspect eval", None),
        }

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        start_time = time.time()
        asta_args = AstaArgs.from_dict(args)
        all_output: list[str] = []

        # Extract URL and model name from provider
        provider_url = getattr(provider, "base_url", None) or "http://localhost:8000/v1"
        model_name = provider.model_name
        # Detect if this is a locally-deployed server (vLLM) vs external API
        is_local = self._is_local_provider(provider, provider_url)

        try:
            from olmo_eval.harness.sandbox.executor import SandboxExecutor
        except ImportError as e:
            return self._error_result(f"SWE-ReX not installed: {e}", start_time)

        sandbox_config = self._create_sandbox_config_with_env(
            container_runtime, output_dir, asta_args
        )

        try:
            async with SandboxExecutor(sandbox_config, name=self.name) as executor:
                if err := await self._run_setup(executor, all_output, start_time):
                    return err

                if is_local:
                    provider_url = self._get_provider_url_for_sandbox(provider_url)
                    if not await self._check_provider_health(executor, provider_url):
                        return self._error_result(
                            f"Provider not reachable at {provider_url}",
                            start_time,
                            "\n".join(all_output),
                        )

                run_cmd = self._build_run_command(model_name, provider_url, is_local, asta_args)
                logger.info(f"[{self.name}] Running: {run_cmd}")

                run_result = await executor.execute_command(
                    run_cmd, timeout=self.timeout_seconds, stream=True, log_prefix=self.name
                )
                all_output.append(f"$ {run_cmd}\n{run_result.output}")
                logger.info(f"[{self.name}] Run exit code: {run_result.exit_code}")

                # Run scoring to compile aggregate results
                # First check if eval_config.json exists (may not for single-task runs)
                config_check = await executor.execute_command(
                    f"test -f {self.results_dir}/eval_config.json && echo exists",
                    timeout=30.0,
                )
                if "exists" not in config_check.output:
                    # Generate config for single-task scoring
                    config_cmd = self._build_config_only_command(asta_args)
                    logger.info(f"[{self.name}] Generating eval config: {config_cmd}")
                    config_result = await executor.execute_command(
                        config_cmd, timeout=120.0, stream=True, log_prefix=f"{self.name}-config"
                    )
                    all_output.append(f"$ {config_cmd}\n{config_result.output}")

                score_cmd = self._build_score_command()
                logger.info(f"[{self.name}] Running scoring: {score_cmd}")

                score_result = await executor.execute_command(
                    score_cmd, timeout=300.0, stream=True, log_prefix=f"{self.name}-score"
                )
                all_output.append(f"$ {score_cmd}\n{score_result.output}")
                logger.info(f"[{self.name}] Score exit code: {score_result.exit_code}")

                result = await self._extract_results(
                    executor,
                    "\n".join(all_output),
                    run_result.exit_code,
                    output_dir,
                )

        except Exception as e:
            logger.exception(f"[{self.name}] Execution failed")
            return self._error_result(str(e), start_time, "\n".join(all_output))

        result.duration_seconds = time.time() - start_time
        if output_dir:
            self._save_results(result, output_dir)

        return result

    def _create_sandbox_config_with_env(
        self,
        container_runtime: str,
        output_dir: str | None,
        asta_args: AstaArgs,
    ) -> Any:
        """Create sandbox configuration with ASTA-specific environment variables."""
        import os
        from typing import cast

        from olmo_eval.evals.external.network import get_docker_network_args
        from olmo_eval.harness.sandbox.config import (
            ContainerRuntime,
            SandboxConfig,
            SandboxMode,
        )

        env_vars = self._build_env_vars()

        if asta_args.sandbox_type == "local":
            env_vars["INSPECT_SANDBOX"] = "local"
            env_vars["INSPECT_EVAL_SANDBOX"] = "local"

        log_dir = None
        if output_dir:
            log_dir = os.path.join(output_dir, "logs")

        runtime = cast(ContainerRuntime, container_runtime)
        image = _get_asta_image(container_runtime)

        return SandboxConfig(
            image=image,
            mode=SandboxMode.DOCKER,
            container_runtime=runtime,
            command_timeout=self.timeout_seconds,
            working_dir=self.working_dir,
            environment=tuple(env_vars.items()),
            docker_args=tuple(get_docker_network_args(runtime=container_runtime)),
            log_dir=log_dir,
        )

    def _build_run_command(
        self,
        model_name: str,
        provider_url: str,
        is_local: bool,
        asta_args: AstaArgs,
    ) -> str:
        """Build the astabench run command."""
        model_spec = f"openai-api/vllm/{model_name}" if is_local else model_name

        args = [
            "uv",
            "run",
            "astabench",
            "eval",
            "--model",
            model_spec,
            "--solver",
            asta_args.solver,
            "--max-samples",
            str(asta_args.max_samples),
            "--max-sandboxes",
            str(asta_args.max_sandboxes),
            "--max-connections",
            str(asta_args.max_connections),
            "--display",
            "plain",
            "--log-dir",
            self.results_dir,
            "--split",
            asta_args.split,
        ]

        if asta_args.sandbox_type == "local":
            args.extend(["--sandbox", "local"])

        if asta_args.limit is not None:
            args.extend(["--limit", str(asta_args.limit)])

        if asta_args.temperature is not None:
            args.extend(["-T", f"temperature={asta_args.temperature}"])

        if asta_args.max_tokens is not None:
            args.extend(["-T", f"max_tokens={asta_args.max_tokens}"])

        # Extra args (for task-specific flags like -T with_search_tools=1)
        args.extend(asta_args.extra_args)

        # Task specifications
        for task in asta_args.tasks or []:
            args.append(f"astabench/{task}")

        # Build command with cd and optional env prefix
        # For local vLLM providers, set VLLM_BASE_URL and VLLM_API_KEY
        # This avoids polluting OPENAI_BASE_URL which would affect scorer models
        env_prefix = (
            f"VLLM_BASE_URL={shlex.quote(provider_url)} VLLM_API_KEY=local " if is_local else ""
        )
        return f"cd {self.working_dir} && {env_prefix}{shlex.join(args)}"

    def _build_score_command(self) -> str:
        """Build the astabench score command."""
        return f"cd {self.working_dir} && uv run astabench score {self.results_dir}"

    def _build_config_only_command(self, asta_args: AstaArgs) -> str:
        """Build command to generate eval_config.json for single-task runs."""
        args = [
            "uv",
            "run",
            "astabench",
            "eval",
            "--config-only",
            "--log-dir",
            self.results_dir,
            "--split",
            asta_args.split,
        ]
        # Task specifications
        for task in asta_args.tasks or []:
            args.append(f"astabench/{task}")

        return f"cd {self.working_dir} && {shlex.join(args)}"

    async def _extract_results(
        self,
        executor: SandboxExecutor,
        raw_output: str,
        exit_code: int,
        output_dir: str | None = None,
    ) -> ExternalEvalResult:
        """Extract metrics from scores.json and Inspect AI log files."""
        all_predictions: list[dict[str, Any]] = []
        parsed_logs: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {"log_files": []}
        score_metrics: dict[str, float] = {}

        # Try to read scores.json (produced by astabench score)
        scores_path = f"{self.results_dir}/scores.json"
        scores_result = await executor.execute_command(
            f"cat {shlex.quote(scores_path)}", timeout=60.0
        )

        if scores_result.success and scores_result.output.strip():
            try:
                scores_content = json.loads(scores_result.output)
                # scores.json contains task-level scores directly
                for task_name, task_data in scores_content.items():
                    if isinstance(task_data, dict):
                        for metric_name, value in task_data.items():
                            if isinstance(value, (int, float)):
                                score_metrics[f"{task_name}_{metric_name}"] = float(value)
                    elif isinstance(task_data, (int, float)):
                        score_metrics[task_name] = float(task_data)
                logger.info(f"[{self.name}] Parsed scores.json with {len(score_metrics)} metrics")

                # Save scores.json to output directory
                if output_dir:
                    from pathlib import Path

                    local_path = Path(output_dir) / "scores.json"
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(local_path, "w") as f:
                        json.dump(scores_content, f, indent=2)

            except json.JSONDecodeError as e:
                logger.warning(f"[{self.name}] Failed to parse scores.json: {e}")

        # Also try summary_stats.json for aggregate metrics
        summary_path = f"{self.results_dir}/summary_stats.json"
        summary_result = await executor.execute_command(
            f"cat {shlex.quote(summary_path)}", timeout=60.0
        )

        if summary_result.success and summary_result.output.strip():
            try:
                summary_content = json.loads(summary_result.output)
                metadata["summary_stats"] = summary_content

                # Save summary_stats.json to output directory
                if output_dir:
                    from pathlib import Path

                    local_path = Path(output_dir) / "summary_stats.json"
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(local_path, "w") as f:
                        json.dump(summary_content, f, indent=2)

            except json.JSONDecodeError as e:
                logger.warning(f"[{self.name}] Failed to parse summary_stats.json: {e}")

        # Also parse individual .eval files for per-sample predictions
        ls_result = await executor.execute_command(
            f"ls {self.results_dir}/*.eval 2>/dev/null",
            timeout=30.0,
        )

        if ls_result.success and ls_result.output.strip():
            for log_file in ls_result.output.strip().split("\n"):
                log_file = log_file.strip()
                if not log_file:
                    continue

                # .eval files may be gzipped - try zcat first, then cat
                cat_result = await executor.execute_command(
                    f"zcat {shlex.quote(log_file)} 2>/dev/null || cat {shlex.quote(log_file)}",
                    timeout=60.0,
                )

                if not cat_result.success or not cat_result.output.strip():
                    logger.warning(f"[{self.name}] Failed to read {log_file}")
                    continue

                try:
                    log_content = json.loads(cat_result.output)
                    parsed = parse_inspect_log(log_content)
                    parsed_logs.append(parsed)
                    all_predictions.extend(parsed.get("predictions", []))
                    metadata["log_files"].append(log_file)

                    # Copy log file to output directory
                    if output_dir:
                        from pathlib import Path

                        log_basename = Path(log_file).name
                        local_path = Path(output_dir) / "inspect_logs" / log_basename
                        local_path.parent.mkdir(parents=True, exist_ok=True)

                        # Save the parsed content
                        with open(local_path.with_suffix(".json"), "w") as f:
                            json.dump(log_content, f, indent=2)

                except json.JSONDecodeError as e:
                    logger.warning(f"[{self.name}] Failed to parse {log_file}: {e}")

        # Build final metrics: prefer scores.json, supplement with parsed logs
        if score_metrics:
            all_metrics = score_metrics
            # Add sample counts from parsed logs
            if parsed_logs:
                all_metrics["num_tasks"] = float(len(parsed_logs))
                total_samples = sum(
                    log.get("metadata", {}).get("total_samples", 0) for log in parsed_logs
                )
                all_metrics["total_samples"] = float(total_samples)
        elif parsed_logs:
            # Fall back to aggregating from individual logs
            all_metrics = aggregate_metrics(parsed_logs)
        else:
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error="No scores.json or Inspect log files found",
                raw_output=raw_output,
            )

        # Merge per-task metadata from parsed logs
        for log_data in parsed_logs:
            task_name = log_data.get("metadata", {}).get("task", "")
            if task_name:
                metadata[f"task_{task_name}"] = log_data.get("metadata", {})

        return ExternalEvalResult(
            name=self.name,
            success=exit_code == 0 and bool(all_metrics),
            metrics=all_metrics,
            metadata=metadata,
            raw_output=raw_output,
            predictions=all_predictions if all_predictions else None,
        )
