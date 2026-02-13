"""Default implementation of external evaluations."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox.config import ContainerRuntime

from olmo_eval.evals.external.base import ExternalEval, SandboxExecutor
from olmo_eval.evals.external.config import ExternalEvalConfig
from olmo_eval.evals.external.network import get_docker_network_args
from olmo_eval.evals.external.result import ExternalEvalResult

logger = logging.getLogger(__name__)


class BaseExternalEval(ExternalEval):
    """Default implementation that executes external evaluations via SWE-ReX.

    This implementation:
    1. Starts a sandbox container
    2. Runs setup commands (e.g., clone repo, pip install)
    3. Executes the evaluation command
    4. Calls extract_results() to parse the output

    The default extract_results() reads a JSON file at /workspace/results.json.
    Subclass and override extract_results() for custom result handling.
    """

    # Default path for JSON results file
    results_path: str = "/workspace/results.json"

    def __init__(self, config: ExternalEvalConfig) -> None:
        """Initialize the default external evaluation.

        Args:
            config: Configuration for the evaluation.
        """
        super().__init__(config)

    async def extract_results(
        self,
        executor: SandboxExecutor,
        run_output: str,
        exit_code: int,
    ) -> ExternalEvalResult:
        """Extract results by reading a JSON file from the sandbox.

        Override this method for custom result extraction logic.

        Args:
            executor: Sandbox executor for reading files.
            run_output: Stdout/stderr from the run command.
            exit_code: Exit code from the run command.

        Returns:
            Evaluation result.
        """
        # Read JSON results file
        results_data = await self._read_json_file(executor, self.results_path)

        if results_data is None:
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error=f"Results file not found at {self.results_path}",
                raw_output=run_output,
            )

        # Extract metrics from the results
        metrics = self._extract_metrics(results_data)
        metadata = self._extract_metadata(results_data)
        success = self._determine_success(results_data, exit_code, metrics)

        return ExternalEvalResult(
            name=self.name,
            metrics=metrics,
            metadata=metadata,
            success=success,
            raw_output=run_output,
        )

    async def execute(
        self,
        provider_url: str,
        model_name: str,
        output_dir: str | None = None,
        container_runtime: str = "podman",
        use_network_host: bool = False,
        extra_env: dict[str, str] | None = None,
    ) -> ExternalEvalResult:
        """Execute the external evaluation.

        Args:
            provider_url: URL of the inference provider.
            model_name: Name/identifier of the model being evaluated.
            output_dir: Optional directory to write results.
            container_runtime: Container runtime to use.
            use_network_host: Whether to use --network=host for the container.
            extra_env: Additional environment variables to set.

        Returns:
            Result of the evaluation.
        """
        start_time = time.time()

        try:
            from olmo_eval.harness.sandbox.config import SandboxConfig, SandboxMode
            from olmo_eval.harness.sandbox.executor import SandboxExecutor
        except ImportError as e:
            return ExternalEvalResult.from_error(
                self.name,
                f"SWE-ReX not installed. Install with: pip install swe-rex. Error: {e}",
            )

        # Build environment variables
        env_vars: dict[str, str] = {
            self.config.api_base_env_var: provider_url,
            self.config.model_env_var: model_name,
        }

        # Add configured environment variables
        for name, value in self.config.environment:
            env_vars[name] = value

        # Add extra environment variables
        if extra_env:
            env_vars.update(extra_env)

        # Forward required secrets from host environment
        for secret in self.config.required_secrets:
            secret_value = os.environ.get(secret)
            if secret_value:
                env_vars[secret] = secret_value
            else:
                logger.warning(f"Required secret '{secret}' not found in environment")

        # Build docker args for networking
        docker_args = list(
            get_docker_network_args(
                runtime=container_runtime,
                use_network_host=use_network_host,
            )
        )

        # Create sandbox config
        runtime: ContainerRuntime = cast("ContainerRuntime", container_runtime)
        sandbox_config = SandboxConfig(
            image=self.config.sandbox_image,
            mode=SandboxMode.DOCKER,
            container_runtime=runtime,
            command_timeout=self.config.timeout,
            working_dir=self.config.working_dir,
            environment=tuple((k, v) for k, v in env_vars.items()),
            docker_args=tuple(docker_args),
        )

        # Execute in sandbox
        all_output: list[str] = []
        exit_code: int = 0

        try:
            async with SandboxExecutor(sandbox_config, name=self.name) as executor:
                # Run setup commands
                for cmd in self.config.setup_commands:
                    logger.info(f"[{self.name}] Running setup: {cmd}")
                    output = await executor.execute(cmd, timeout=self.config.timeout)
                    all_output.append(f"$ {cmd}\n{output}")

                    cmd_exit_code = self._parse_exit_code(output)
                    if cmd_exit_code != 0:
                        return ExternalEvalResult(
                            name=self.name,
                            success=False,
                            error=f"Setup command failed: {cmd}",
                            raw_output="\n".join(all_output),
                            duration_seconds=time.time() - start_time,
                        )

                # Run the evaluation command
                run_cmd = self._expand_command(
                    self.config.run_command,
                    provider_url=provider_url,
                    model_name=model_name,
                )
                logger.info(f"[{self.name}] Running evaluation: {run_cmd}")
                run_output = await executor.execute(run_cmd, timeout=self.config.timeout)
                all_output.append(f"$ {run_cmd}\n{run_output}")
                exit_code = self._parse_exit_code(run_output)

                # Extract results using the subclass method
                result = await self.extract_results(
                    executor=executor,
                    run_output="\n".join(all_output),
                    exit_code=exit_code,
                )

        except Exception as e:
            logger.exception(f"[{self.name}] Execution failed")
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error=str(e),
                raw_output="\n".join(all_output),
                duration_seconds=time.time() - start_time,
            )

        # Add duration to result
        result.duration_seconds = time.time() - start_time

        # Save results to output directory
        if output_dir:
            self._save_results(result, output_dir)

        return result

    def _expand_command(
        self,
        command: str,
        provider_url: str,
        model_name: str,
    ) -> str:
        """Expand environment variable references in the command."""
        result = command
        result = result.replace(f"${self.config.api_base_env_var}", provider_url)
        result = result.replace(f"${self.config.model_env_var}", model_name)
        return result

    def _parse_exit_code(self, output: str) -> int:
        """Parse exit code from command output."""
        # SWE-ReX appends "[Exit code: N]" to output
        if "[Exit code:" in output:
            try:
                code_str = output.split("[Exit code:")[1].split("]")[0].strip()
                return int(code_str)
            except (IndexError, ValueError):
                pass
        return 0

    async def _read_json_file(self, executor: SandboxExecutor, path: str) -> dict[str, Any] | None:
        """Read and parse a JSON file from the sandbox."""
        try:
            output = await executor.execute(f"cat {path}", timeout=30.0)

            # Check for file not found
            exit_code = self._parse_exit_code(output)
            if exit_code != 0 or "No such file" in output:
                return None

            # Parse JSON (remove exit code suffix)
            json_text = output.split("[Exit code:")[0].strip()
            return json.loads(json_text)

        except json.JSONDecodeError as e:
            logger.warning(f"[{self.name}] Failed to parse JSON from {path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to read {path}: {e}")
            return None

    def _extract_metrics(self, results_data: dict[str, Any]) -> dict[str, float]:
        """Extract metrics from results data.

        Override for custom metric extraction.
        """
        metrics: dict[str, float] = {}

        # Look for common metric keys
        for key in ("metrics", "scores", "results"):
            if key in results_data and isinstance(results_data[key], dict):
                for metric_name, value in results_data[key].items():
                    if isinstance(value, (int, float)):
                        metrics[metric_name] = float(value)

        # Also check top-level numeric values
        for key, value in results_data.items():
            if isinstance(value, (int, float)) and key not in ("timestamp", "duration"):
                metrics[key] = float(value)

        return metrics

    def _extract_metadata(self, results_data: dict[str, Any]) -> dict[str, Any]:
        """Extract metadata from results data.

        Override for custom metadata extraction.
        """
        metadata: dict[str, Any] = {}

        for key, value in results_data.items():
            if key in ("metrics", "scores", "results"):
                continue
            if isinstance(value, (int, float)):
                continue
            metadata[key] = value

        return metadata

    def _determine_success(
        self,
        results_data: dict[str, Any],
        exit_code: int,
        metrics: dict[str, float],
    ) -> bool:
        """Determine if the evaluation was successful.

        Override for custom success logic.
        """
        # Check for explicit success field
        for key in ("success", "passed", "ok"):
            if key in results_data:
                return bool(results_data[key])

        # Default: success if exit code is 0 and we have metrics
        return exit_code == 0 and bool(metrics)

    def _save_results(self, result: ExternalEvalResult, output_dir: str) -> None:
        """Save results to the output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / f"{self.name}_results.json"
        with open(results_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"[{self.name}] Results saved to {results_file}")
