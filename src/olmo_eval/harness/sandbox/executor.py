"""Sandbox executor for isolated command execution via SWE-ReX."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
import uuid
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
        self._session_created: bool = False

    def _log(self, level: int, msg: str) -> None:
        """Log a message with optional name prefix."""
        if self.name:
            logger.log(level, f"[{self.name}] {msg}")
        else:
            logger.log(level, msg)

    async def _get_container_diagnostics(self) -> str:
        """Gather diagnostic info when sandbox becomes unresponsive."""
        import subprocess

        diag_lines = ["--- Container Diagnostics ---"]

        # Check if deployment reports alive
        if self._deployment is not None:
            try:
                is_alive = await self._deployment.is_alive()
                diag_lines.append(f"Deployment is_alive: {is_alive}")
            except Exception as e:
                diag_lines.append(f"Deployment is_alive check failed: {e}")

            # Get container name and fetch logs from host
            container_name = getattr(self._deployment, "container_name", None)
            if container_name:
                diag_lines.append(f"Container name: {container_name}")

                # Try to get container status and logs via subprocess
                runtime = self.config.container_runtime or "docker"
                try:
                    # Get container state
                    inspect_result = subprocess.run(
                        [runtime, "inspect", "--format", "{{.State.Status}}", container_name],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if inspect_result.returncode == 0:
                        diag_lines.append(f"Container status: {inspect_result.stdout.strip()}")
                    else:
                        err = inspect_result.stderr.strip()
                        diag_lines.append(f"Container inspect failed: {err}")
                except Exception as e:
                    diag_lines.append(f"Container inspect error: {e}")

                try:
                    # Get last 50 lines of container logs
                    logs_result = subprocess.run(
                        [runtime, "logs", "--tail", "50", container_name],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if logs_result.returncode == 0:
                        logs = logs_result.stdout.strip() or logs_result.stderr.strip()
                        if logs:
                            diag_lines.append(f"Container logs (last 50 lines):\n{logs}")
                        else:
                            diag_lines.append("Container logs: (empty)")
                    else:
                        diag_lines.append(f"Container logs failed: {logs_result.stderr.strip()}")
                except Exception as e:
                    diag_lines.append(f"Container logs error: {e}")
            else:
                diag_lines.append("Container name: not available")
        else:
            diag_lines.append("Deployment: None")

        diag = "\n".join(diag_lines)
        self._log(logging.ERROR, diag)
        return diag

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
        # Close session before stopping deployment
        if self._session_created and self._runtime is not None:
            try:
                from swerex.runtime.abstract import CloseBashSessionRequest

                await self._runtime.close_session(CloseBashSessionRequest(session="default"))
            except Exception as e:
                self._log(logging.DEBUG, f"Failed to close session: {e}")
            self._session_created = False

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

        Uses background execution to avoid swerex HTTP timeout issues.
        """
        from swerex.runtime.abstract import Command

        # Use unique temp paths to avoid conflicts with concurrent executions
        cmd_id = uuid.uuid4().hex[:12]
        output_file = f"/tmp/_sandbox_output_{cmd_id}.log"
        exit_code_file = f"/tmp/_sandbox_exit_code_{cmd_id}"
        script_file = f"/tmp/_sandbox_script_{cmd_id}.sh"
        pid_file = f"/tmp/_sandbox_pid_{cmd_id}"

        # Create script via base64 to avoid quoting issues
        env_prefix = "PYTHONUNBUFFERED=1 NO_COLOR=1 TERM=dumb TTY_COMPATIBLE=0 TTY_INTERACTIVE=0"
        script = f"#!/bin/bash\n{env_prefix} {command}\n"
        encoded = base64.b64encode(script.encode()).decode()

        # Setup: create script file
        setup = (
            f"rm -f {output_file} {exit_code_file} {pid_file} && "
            f"echo '{encoded}' | base64 -d > {script_file} && "
            f"chmod +x {script_file}"
        )
        try:
            await self._runtime.execute(Command(command=["bash", "-c", setup], timeout=30.0))
        except Exception as e:
            return ExecutionResult(False, f"Failed to create script: {e}", -1)

        # Start script in detached background process (setsid creates new session)
        # Store the PID so we can kill the process group on timeout
        start = (
            f"setsid bash -c '{script_file} > {output_file} 2>&1; "
            f"echo $? > {exit_code_file}' < /dev/null > /dev/null 2>&1 & "
            f"echo $! > {pid_file}"
        )
        try:
            await self._runtime.execute(Command(command=["bash", "-c", start], timeout=10.0))
        except Exception as e:
            return ExecutionResult(False, f"Failed to start command: {e}", -1)

        # Poll for output and completion
        last_pos = 0
        start_time = time.time()
        timed_out = False
        consecutive_failures = 0
        max_consecutive_failures = 3

        async def kill_process_group() -> None:
            """Kill the background process group."""
            try:
                # Read the PID and kill its process group (negative PID kills group)
                kill_cmd = (
                    f"pid=$(cat {pid_file} 2>/dev/null) && "
                    f'[ -n "$pid" ] && kill -TERM -$pid 2>/dev/null; '
                    f"sleep 0.5; "
                    f'[ -n "$pid" ] && kill -KILL -$pid 2>/dev/null; '
                    "true"
                )
                await self._runtime.execute(Command(command=["bash", "-c", kill_cmd], timeout=5.0))
            except Exception as e:
                self._log(logging.DEBUG, f"Process group kill (may be already exited): {e}")

        async def cleanup_temp_files() -> None:
            """Remove temporary files."""
            try:
                cleanup_cmd = f"rm -f {output_file} {exit_code_file} {script_file} {pid_file}"
                await self._runtime.execute(
                    Command(command=["bash", "-c", cleanup_cmd], timeout=5.0)
                )
            except Exception:
                pass  # Best effort cleanup

        while True:
            await asyncio.sleep(1.0)

            if time.time() - start_time > timeout:
                self._log(logging.WARNING, f"Command timed out after {timeout}s")
                timed_out = True
                await kill_process_group()
                break

            # Stream new output and check for completion in one pass
            try:
                resp = await self._runtime.execute(
                    Command(
                        command=[
                            "bash",
                            "-c",
                            f"tail -c +{last_pos + 1} {output_file} 2>/dev/null; "
                            f"echo '---EXIT_CODE---'; "
                            f"cat {exit_code_file} 2>/dev/null",
                        ],
                        timeout=10.0,
                    )
                )
                consecutive_failures = 0  # Reset on success

                parts = (resp.stdout or "").split("---EXIT_CODE---")
                new_output = parts[0] if parts else ""
                exit_marker = parts[1].strip() if len(parts) > 1 else ""

                if new_output:
                    last_pos += len(new_output)
                    for line in new_output.rstrip("\n").split("\n"):
                        if line:
                            logger.info(f"[{prefix}] {line}")

                if exit_marker:
                    break

            except Exception as e:
                consecutive_failures += 1
                self._log(
                    logging.WARNING,
                    f"Poll failed ({consecutive_failures}/{max_consecutive_failures}): {e}",
                )
                if consecutive_failures >= max_consecutive_failures:
                    self._log(logging.ERROR, "Sandbox unresponsive, aborting")
                    await kill_process_group()
                    await cleanup_temp_files()
                    diag = await self._get_container_diagnostics()
                    msg = f"Sandbox unresponsive after {consecutive_failures} polls\n{diag}"
                    return ExecutionResult(False, msg, -1)

        # Read final output and exit code
        full_output = ""
        exit_code = -1

        try:
            await asyncio.sleep(0.2)
            resp = await self._runtime.execute(
                Command(
                    command=[
                        "bash",
                        "-c",
                        f"cat {output_file} 2>/dev/null; "
                        f"echo '---EXIT_CODE---'; "
                        f"cat {exit_code_file} 2>/dev/null",
                    ],
                    timeout=30.0,
                )
            )
            parts = (resp.stdout or "").split("---EXIT_CODE---")
            full_output = parts[0] if parts else ""
            if len(parts) > 1 and parts[1].strip():
                exit_code = int(parts[1].strip())
        except Exception as e:
            self._log(logging.WARNING, f"Failed to read final output: {e}")

            # Log any output we missed during streaming
            if len(full_output) > last_pos:
                for line in full_output[last_pos:].rstrip("\n").split("\n"):
                    if line:
                        logger.info(f"[{prefix}] {line}")

        # Clean up temp files
        await cleanup_temp_files()

        if timed_out:
            full_output = (full_output + "\n[Command timed out]") if full_output else "[Timed out]"

        return ExecutionResult(exit_code == 0, full_output, exit_code)

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

    async def _ensure_session(self) -> None:
        """Create the session if it doesn't exist."""
        if self._session_created:
            return
        from swerex.runtime.abstract import CreateBashSessionRequest

        await self._runtime.create_session(CreateBashSessionRequest(session="default"))
        self._session_created = True

    async def execute_in_session(
        self,
        command: str,
        timeout: float | None = None,
        stream: bool = False,
        log_prefix: str | None = None,
    ) -> ExecutionResult:
        """Execute a command in the persistent bash session.

        Shell state (cd, export, aliases) persists between calls.
        Session is auto-created on first use.

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
            raise RuntimeError("Sandbox not started.")

        await self._ensure_session()

        from swerex.runtime.abstract import BashAction

        effective_timeout = timeout if timeout is not None else self.config.command_timeout
        prefix = log_prefix or self.name or "sandbox"

        observation = await self._runtime.run_in_session(
            BashAction(
                command=command,
                session="default",
                timeout=effective_timeout,
                check="silent",
            )
        )

        output = observation.output or ""

        if stream and output:
            for line in output.rstrip("\n").split("\n"):
                if line:
                    logger.info(f"[{prefix}] {line}")

        return ExecutionResult(
            success=observation.exit_code == 0,
            output=output,
            exit_code=observation.exit_code or 0,
            error=observation.failure_reason or None,
        )

    @property
    def is_running(self) -> bool:
        """Check if the sandbox is running."""
        return self._deployment is not None and self._runtime is not None
