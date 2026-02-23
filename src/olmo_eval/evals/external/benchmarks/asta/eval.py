"""ASTA-bench external evaluation implementation."""

from __future__ import annotations

import json
import logging
import shlex
import time
from typing import TYPE_CHECKING, Any

from olmo_eval.evals.external.base import SandboxedExternalEval
from olmo_eval.evals.external.benchmarks.asta.args import AstaArgs
from olmo_eval.evals.external.benchmarks.asta.result_parser import (
    aggregate_metrics,
    parse_inspect_log,
)
from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox.executor import SandboxExecutor
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

# ASTA-bench task categories
ASTA_TASKS = {
    "literature": [
        "paper_finder",
        "sqa",
        "litqa2",
        "paper_finder_litqa2",
        "arxivdigestables",
    ],
    "code": ["core_bench", "ds1000", "super"],
    "data_analysis": ["discoverybench"],
    "discovery": ["e2e_discovery", "e2e_discovery_hard"],
}


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
        return "ghcr.io/allenai/olmo-eval-asta:latest"

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
        return ("OPENAI_API_KEY", "ASTA_TOOL_KEY", "HF_TOKEN")

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            # Dataset selection
            "split": ("Dataset split: 'validation' or 'test'", "validation"),
            "tasks": ("Comma-separated task names to run (default: all)", None),
            "limit": ("Maximum problems per task", None),
            # Agent configuration
            "solver": ("Agent solver type: 'react' or 'basic'", "react"),
            # Parallelism
            "max_samples": ("Max concurrent problems", 1),
            "max_sandboxes": ("Max parallel sandboxes", 1),
            "max_connections": ("Max model API connections", 8),
            # Sandbox mode
            "sandbox_type": ("Sandbox type: 'local' (Beaker) or 'docker'", "local"),
            # Tool flags
            "with_asta_tools": ("Enable literature search tools", True),
            "with_stateful_python": ("Enable persistent code execution", True),
            "with_report_editor": ("Enable SQA report editor", False),
            "with_table_editor": ("Enable ArxivDigestables editor", False),
            "with_thinking_tool": ("Enable extended reasoning tool", False),
            # Model overrides
            "temperature": ("Temperature for agent responses", None),
            "max_tokens": ("Max tokens for agent responses", None),
            # Scoring
            "scorer_model": ("Model for scoring (requires API key)", None),
            # Extra args
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
        provider_url = getattr(provider, "base_url", "http://localhost:8000/v1")
        model_name = provider.model_name
        is_local = hasattr(provider, "_server") or "localhost" in provider_url

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

                sandbox_url = self._get_provider_url_for_sandbox(provider_url)

                if not await self._check_provider_health(executor, sandbox_url):
                    return self._error_result(
                        f"Provider not reachable at {sandbox_url}",
                        start_time,
                        "\n".join(all_output),
                    )

                run_cmd = self._build_run_command(model_name, sandbox_url, is_local, asta_args)
                logger.info(f"[{self.name}] Running: {run_cmd}")

                run_result = await executor.execute_command(
                    run_cmd, timeout=self.timeout_seconds, stream=True, log_prefix=self.name
                )
                all_output.append(f"$ {run_cmd}\n{run_result.output}")
                logger.info(f"[{self.name}] Run exit code: {run_result.exit_code}")

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

        # Build base environment from required secrets
        env_vars = self._build_env_vars()

        # Add ASTA-specific environment variables
        if asta_args.sandbox_type == "local":
            # Tell Inspect to run code locally (no nested containers)
            env_vars["INSPECT_SANDBOX"] = "local"
            env_vars["INSPECT_EVAL_SANDBOX"] = "local"

        # Forward optional secrets if available
        for optional_secret in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
            if value := os.environ.get(optional_secret):
                env_vars[optional_secret] = value

        # Set log_dir for sandbox container logs
        log_dir = None
        if output_dir:
            log_dir = os.path.join(output_dir, "logs")

        runtime = cast(ContainerRuntime, container_runtime)
        return SandboxConfig(
            image=self.sandbox_image,
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
        # For local providers, use openai-compatible model spec
        if is_local:
            model_spec = f"openai/{model_name}"
            # Set base URL via environment in the command
            env_prefix = f"OPENAI_BASE_URL={shlex.quote(provider_url)}"
        else:
            model_spec = model_name
            env_prefix = ""

        parts = [f"cd {self.working_dir} &&"]
        if env_prefix:
            parts.append(env_prefix)

        parts.extend(
            [
                "uv run astabench eval",
                f"--model {shlex.quote(model_spec)}",
                f"--solver {shlex.quote(asta_args.solver)}",
                f"--max-samples {asta_args.max_samples}",
                f"--max-sandboxes {asta_args.max_sandboxes}",
                f"--max-connections {asta_args.max_connections}",
                "--display=plain",
                f"--log-dir {self.results_dir}",
            ]
        )

        # Add sandbox mode
        if asta_args.sandbox_type == "local":
            parts.append("--sandbox local")

        # Add tool flags as solver args
        tool_args = []
        if asta_args.with_asta_tools:
            tool_args.append("with_asta_tools=1")
        if asta_args.with_stateful_python:
            tool_args.append("with_stateful_python=1")
        if asta_args.with_report_editor:
            tool_args.append("with_report_editor=1")
        if asta_args.with_table_editor:
            tool_args.append("with_table_editor=1")
        if asta_args.with_thinking_tool:
            tool_args.append("with_thinking_tool=1")

        for arg in tool_args:
            parts.append(f"-S {arg}")

        # Model overrides
        if asta_args.temperature is not None:
            parts.append(f"-T temperature={asta_args.temperature}")
        if asta_args.max_tokens is not None:
            parts.append(f"-T max_tokens={asta_args.max_tokens}")

        # Scorer model (for tasks that need separate scoring model)
        if asta_args.scorer_model:
            parts.append(f"--scorer-model {shlex.quote(asta_args.scorer_model)}")

        # Limit number of problems
        if asta_args.limit is not None:
            parts.append(f"--limit {asta_args.limit}")

        # Extra args
        for extra_arg in asta_args.extra_args:
            parts.append(shlex.quote(extra_arg))

        # Add task specifications
        if asta_args.tasks:
            for task in asta_args.tasks:
                task_spec = f"astabench/{task}_{asta_args.split}"
                parts.append(shlex.quote(task_spec))
        else:
            # Run all tasks for the split
            parts.append(f"astabench/{asta_args.split}")

        return " ".join(parts)

    async def _extract_results(
        self,
        executor: SandboxExecutor,
        raw_output: str,
        exit_code: int,
        output_dir: str | None = None,
    ) -> ExternalEvalResult:
        """Extract metrics from Inspect AI log files."""
        # List log files in results directory
        ls_result = await executor.execute_command(
            f"ls {self.results_dir}/*.json 2>/dev/null || ls {self.results_dir}/*.eval 2>/dev/null",
            timeout=30.0,
        )

        if not ls_result.success or not ls_result.output.strip():
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error="No Inspect log files found",
                raw_output=raw_output,
            )

        all_predictions: list[dict[str, Any]] = []
        parsed_logs: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {"log_files": []}

        for log_file in ls_result.output.strip().split("\n"):
            log_file = log_file.strip()
            if not log_file:
                continue

            # Read log file content
            if log_file.endswith(".eval"):
                # .eval files are gzipped - use zcat
                cat_result = await executor.execute_command(
                    f"zcat {shlex.quote(log_file)}", timeout=60.0
                )
            else:
                cat_result = await executor.execute_command(
                    f"cat {shlex.quote(log_file)}", timeout=60.0
                )

            if not cat_result.success:
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

        if not parsed_logs:
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error="Failed to parse any Inspect log files",
                raw_output=raw_output,
            )

        # Aggregate metrics across all logs
        all_metrics = aggregate_metrics(parsed_logs)

        # Merge per-task metadata
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
