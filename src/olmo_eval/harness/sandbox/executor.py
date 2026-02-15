"""Sandbox executor for isolated command execution via SWE-ReX."""

from __future__ import annotations

import asyncio
import contextlib
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

                # Add environment variables as docker args
                for key, value in self.config.environment:
                    docker_args.extend(["-e", f"{key}={value}"])

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
        result = await self.execute_command(command, timeout)
        output = result.output
        if result.exit_code != 0:
            output += f"\n[Exit code: {result.exit_code}]"
        return output

    async def execute_command(
        self,
        command: str,
        timeout: float | None = None,
        stream: bool = False,
        log_prefix: str | None = None,
    ) -> ExecutionResult:
        """Execute a command in the sandbox and return structured result.

        Args:
            command: The bash command to execute.
            timeout: Optional timeout override in seconds.
            stream: If True, stream output to logs as the command runs.
            log_prefix: Prefix for streamed log lines (defaults to self.name).

        Returns:
            ExecutionResult with success status, output, and exit code.

        Raises:
            RuntimeError: If the sandbox is not started.
        """
        if self._runtime is None:
            raise RuntimeError("Sandbox not started. Call start() first or use async context.")

        from swerex.runtime.abstract import Command

        effective_timeout = timeout if timeout is not None else self.config.command_timeout
        prefix = log_prefix or self.name or "sandbox"

        if stream:
            return await self._execute_streaming(command, effective_timeout, prefix)

        response = await self._runtime.execute(
            Command(
                command=["bash", "-c", command],
                timeout=effective_timeout,
            )
        )

        # Combine stdout and stderr
        output_parts = []
        if response.stdout:
            output_parts.append(response.stdout)
        if response.stderr:
            output_parts.append(response.stderr)

        return ExecutionResult(
            success=response.exit_code == 0,
            output="".join(output_parts) if output_parts else "",
            exit_code=response.exit_code,
        )

    async def _execute_streaming(
        self, command: str, timeout: float, prefix: str
    ) -> ExecutionResult:
        """Execute a command with streaming output to logs.

        Uses background execution to avoid HTTP timeout issues with long-running commands.
        The command runs in a background process while we poll for output and completion.
        """
        from swerex.runtime.abstract import Command

        # File paths for background execution
        output_file = "/tmp/_sandbox_output.log"
        exit_code_file = "/tmp/_sandbox_exit_code"
        pid_file = "/tmp/_sandbox_pid"
        script_file = "/tmp/_sandbox_script.sh"

        # Write the command to a script file to avoid quoting issues
        # Use base64 encoding to safely transfer the command
        import base64

        script_content = f"""#!/bin/bash
export PYTHONUNBUFFERED=1
{command}
"""
        encoded_script = base64.b64encode(script_content.encode()).decode()

        # Create script and start it in background
        # The script's stdout/stderr goes to output_file, exit code captured separately
        setup_cmd = (
            f"rm -f {output_file} {exit_code_file} {pid_file} && "
            f"echo '{encoded_script}' | base64 -d > {script_file} && "
            f"chmod +x {script_file} && "
            f"( {script_file}; echo $? > {exit_code_file} ) > {output_file} 2>&1 & "
            f"echo $! > {pid_file}"
        )

        # Start the background process (quick HTTP call)
        try:
            setup_result = await self._runtime.execute(
                Command(command=["bash", "-c", setup_cmd], timeout=30.0)
            )
            self._log(
                logging.INFO,
                f"Background setup: exit_code={setup_result.exit_code}",
            )
            if setup_result.stdout:
                self._log(logging.INFO, f"Setup stdout: {setup_result.stdout.strip()}")
            if setup_result.stderr:
                self._log(logging.INFO, f"Setup stderr: {setup_result.stderr.strip()}")
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=f"Failed to start background command: {e}",
                exit_code=-1,
            )

        # Give the script a moment to start
        await asyncio.sleep(0.5)

        # Log initial state for debugging
        try:
            of = output_file
            debug_cmd = (
                f"echo 'PID:' $(cat {pid_file} 2>/dev/null || echo 'missing'); "
                f"echo 'Script:' $(test -f {script_file} && echo yes || echo no); "
                f"echo 'Output:' $(test -f {of} && wc -c < {of} || echo 'no'); "
                f"echo 'Exit:' $(cat {exit_code_file} 2>/dev/null || echo 'pending'); "
                f"echo 'Proc:'; ps aux | grep _sandbox_script | grep -v grep || echo '(none)'"
            )
            debug_result = await self._runtime.execute(
                Command(command=["bash", "-c", debug_cmd], timeout=10.0)
            )
            for line in (debug_result.stdout or "").strip().split("\n"):
                if line:
                    self._log(logging.INFO, f"Debug: {line}")
        except Exception as e:
            self._log(logging.INFO, f"Debug check failed: {e}")

        # Poll for output and completion
        last_pos = 0
        poll_interval = 1.0
        streamed_output: list[str] = []
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self._log(logging.WARNING, f"Command timed out after {timeout}s")
                # Try to kill the background process
                with contextlib.suppress(Exception):
                    await self._runtime.execute(
                        Command(
                            command=[
                                "bash",
                                "-c",
                                f"kill $(cat {pid_file} 2>/dev/null) 2>/dev/null || true",
                            ],
                            timeout=5.0,
                        )
                    )
                break

            await asyncio.sleep(poll_interval)

            # Read new output
            try:
                tail_cmd = f"tail -c +{last_pos + 1} {output_file} 2>/dev/null"
                tail_resp = await self._runtime.execute(
                    Command(command=["bash", "-c", tail_cmd], timeout=10.0)
                )
                new_output = tail_resp.stdout or ""
                if new_output:
                    last_pos += len(new_output)
                    streamed_output.append(new_output)
                    for line in new_output.rstrip("\n").split("\n"):
                        if line:
                            logger.info(f"[{prefix}] {line}")
            except Exception:
                pass

            # Check if process completed (exit code file exists)
            try:
                check_resp = await self._runtime.execute(
                    Command(
                        command=["bash", "-c", f"cat {exit_code_file} 2>/dev/null"],
                        timeout=5.0,
                    )
                )
                if check_resp.stdout and check_resp.stdout.strip():
                    # Process completed
                    self._log(
                        logging.INFO,
                        f"Process completed after {elapsed:.1f}s, "
                        f"exit_code_file={check_resp.stdout.strip()!r}",
                    )
                    break
            except Exception:
                pass

        # Read final output
        full_output = ""
        try:
            # Give a moment for final writes
            await asyncio.sleep(0.5)
            cat_resp = await self._runtime.execute(
                Command(command=["bash", "-c", f"cat {output_file} 2>/dev/null"], timeout=30.0)
            )
            full_output = cat_resp.stdout or ""
            # Log any remaining output not yet streamed
            if len(full_output) > last_pos:
                remaining = full_output[last_pos:]
                for line in remaining.rstrip("\n").split("\n"):
                    if line:
                        logger.info(f"[{prefix}] {line}")
        except Exception:
            full_output = "".join(streamed_output)

        # Get exit code
        exit_code = -1
        timed_out = time.time() - start_time > timeout
        if not timed_out:
            try:
                code_resp = await self._runtime.execute(
                    Command(
                        command=["bash", "-c", f"cat {exit_code_file} 2>/dev/null"],
                        timeout=5.0,
                    )
                )
                if code_resp.stdout and code_resp.stdout.strip():
                    exit_code = int(code_resp.stdout.strip())
            except Exception:
                pass

        if timed_out:
            if full_output:
                full_output += "\n[Command timed out]"
            else:
                full_output = "[Command timed out]"

        return ExecutionResult(
            success=exit_code == 0,
            output=full_output,
            exit_code=exit_code,
        )

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
