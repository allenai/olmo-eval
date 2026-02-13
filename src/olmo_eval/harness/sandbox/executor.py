"""Sandbox executor for isolated command execution via SWE-ReX."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from olmo_eval.common.execution.environment import ExecutionResult

from .config import SandboxConfig, SandboxMode

logger = logging.getLogger(__name__)


def _get_log_docker_args(log_dir: str, name: str) -> tuple[str, ...]:
    """Get docker args for logging to a named file.

    Args:
        log_dir: Directory to write log files.
        name: Sandbox name for the log file.

    Returns:
        Docker args tuple for json-file logging.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    return ("--log-driver=json-file", "--log-opt", f"path={log_path}")


async def _run_with_progress(
    coro: Any,
    message: str,
    interval: float = 5.0,
) -> Any:
    """Run a coroutine while logging progress at regular intervals.

    Args:
        coro: The coroutine to run.
        message: Base message to log (elapsed time will be appended).
        interval: Seconds between progress logs.

    Returns:
        The result of the coroutine.
    """
    task = asyncio.create_task(coro)
    start = time.time()

    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=interval)
        except TimeoutError:
            elapsed = time.time() - start
            logger.info(f"{message} ({elapsed:.0f}s elapsed)")

    return task.result()


class SandboxExecutor:
    """Executor for sandboxed command execution via SWE-ReX.

    This class manages the lifecycle of a SWE-ReX deployment for executing
    commands in an isolated container environment.

    Usage:
        async with SandboxExecutor(config) as executor:
            result = await executor.execute("python --version")
            print(result)
    """

    def __init__(self, config: SandboxConfig, name: str | None = None) -> None:
        """Initialize the sandbox executor.

        Args:
            config: Sandbox configuration.
            name: Optional name for logging (e.g., "sandbox-0").
        """
        self.config = config
        self.name = name
        self._deployment: Any = None
        self._runtime: Any = None

    def _log(self, level: int, msg: str) -> None:
        """Log a message with optional name prefix."""
        if self.name:
            logger.log(level, f"[{self.name}] {msg}")
        else:
            logger.log(level, msg)

    async def __aenter__(self) -> SandboxExecutor:
        """Start the sandbox environment."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Stop the sandbox environment."""
        await self.stop()

    async def start(self) -> None:
        """Start the sandbox deployment.

        Raises:
            ImportError: If swe-rex is not installed.
            RuntimeError: If container runtime is not available.
        """
        self._log(logging.INFO, "Creating sandbox deployment...")
        deployment = self.get_deployment()

        self._log(logging.INFO, "Starting sandbox deployment...")
        prefix = f"[{self.name}] " if self.name else ""
        await _run_with_progress(
            deployment.start(),
            f"{prefix}Waiting for sandbox runtime",
            interval=5.0,
        )

        self._deployment = deployment
        self._runtime = deployment.runtime
        self._log(logging.INFO, "Sandbox deployment ready!")

    def get_deployment(self) -> Any:
        """Create the appropriate deployment based on configuration.

        Returns:
            A deployment instance.

        Raises:
            ImportError: If swe-rex or required extras are not installed.
            RuntimeError: If the requested container runtime is not available.
        """
        match self.config.mode:
            case SandboxMode.DOCKER:
                try:
                    from swerex.deployment.docker import DockerDeployment
                except ImportError as e:
                    raise ImportError(
                        "swe-rex not installed. Install with: pip install swe-rex"
                    ) from e

                # Build docker args, adding log args if log_dir is configured
                docker_args = list(self.config.docker_args) if self.config.docker_args else []
                if self.config.log_dir and self.name:
                    docker_args.extend(_get_log_docker_args(self.config.log_dir, self.name))

                return DockerDeployment(
                    image=self.config.image,
                    container_runtime=self.config.container_runtime,
                    startup_timeout=self.config.startup_timeout,
                    docker_args=docker_args or None,
                )

            case SandboxMode.LOCAL:
                try:
                    from swerex.deployment.local import LocalDeployment
                except ImportError as e:
                    raise ImportError(
                        "swe-rex not installed. Install with: pip install swe-rex"
                    ) from e

                self._log(
                    logging.WARNING,
                    "Using local deployment (unsandboxed). Commands will run on host system.",
                )
                return LocalDeployment()

            case SandboxMode.MODAL:
                try:
                    from swerex.deployment.modal import ModalDeployment
                except ImportError as e:
                    raise ImportError(
                        "swe-rex modal support not installed. "
                        "Install with: pip install 'swe-rex[modal]'"
                    ) from e

                return ModalDeployment(
                    image=self.config.image,
                    startup_timeout=self.config.startup_timeout,
                    runtime_timeout=self.config.runtime_timeout,
                    modal_sandbox_kwargs=self.config.modal_sandbox_kwargs,
                )

    async def stop(self) -> None:
        """Stop the sandbox deployment and clean up resources."""
        if self._deployment is not None:
            try:
                await self._deployment.stop()
            except Exception as e:
                self._log(logging.WARNING, f"Failed to stop deployment: {e}")
            self._deployment = None
            self._runtime = None

        self._log(logging.INFO, "Sandbox stopped")

    async def execute(self, command: str, timeout: float | None = None) -> str:
        """Execute a command in the sandbox.

        Args:
            command: The bash command to execute.
            timeout: Optional timeout override in seconds.

        Returns:
            The command output (stdout + stderr).

        Raises:
            RuntimeError: If the sandbox is not started.
        """
        if self._runtime is None:
            raise RuntimeError("Sandbox not started. Call start() first or use async context.")

        from swerex.runtime.abstract import Command

        effective_timeout = timeout if timeout is not None else self.config.command_timeout

        response = await self._runtime.execute(
            Command(
                command=["bash", "-c", command],
                timeout=effective_timeout,
            )
        )

        # Combine stdout and stderr, include exit code information
        output_parts = []
        if response.stdout:
            output_parts.append(response.stdout)
        if response.stderr:
            output_parts.append(response.stderr)
        if response.exit_code != 0:
            output_parts.append(f"\n[Exit code: {response.exit_code}]")

        return "".join(output_parts) if output_parts else ""

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute code in the specified language.

        Args:
            code: Source code to execute.
            language: Programming language (default: "python").
            timeout: Optional timeout in seconds.

        Returns:
            ExecutionResult with success status and output.
        """
        if self._runtime is None:
            return ExecutionResult(
                success=False,
                error="Sandbox not started. Call start() first or use async context.",
            )

        interpreters = {
            "python": "python",
            "python3": "python3",
            "bash": "bash",
            "sh": "sh",
        }

        interpreter = interpreters.get(language.lower())
        if interpreter is None:
            return ExecutionResult(
                success=False,
                error=f"Unsupported language: {language}",
            )

        try:
            from swerex.runtime.abstract import Command

            effective_timeout = timeout if timeout is not None else self.config.command_timeout

            response = await self._runtime.execute(
                Command(
                    command=[interpreter, "-c", code],
                    timeout=effective_timeout,
                )
            )

            output = response.stdout or ""
            if response.stderr:
                output += response.stderr

            return ExecutionResult(
                success=response.exit_code == 0,
                output=output,
                exit_code=response.exit_code,
            )

        except Exception as e:
            self._log(logging.WARNING, f"Code execution failed: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
            )

    @property
    def is_running(self) -> bool:
        """Check if the sandbox is running."""
        return self._deployment is not None and self._runtime is not None
