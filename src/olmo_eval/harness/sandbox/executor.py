"""Sandbox executor for isolated command execution via SWE-ReX."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
import uuid
from typing import Any

import aiohttp

from olmo_eval.common.execution.environment import ExecutionResult

from .config import SandboxConfig, SandboxMode
from .diagnostics import start_internal_monitor

logger = logging.getLogger(__name__)


def _get_log_docker_args(log_dir: str, name: str) -> tuple[str, ...]:
    """Get docker args for logging to a named file.

    Args:
        log_dir: Directory to write log files.
        name: Sandbox name for the log file.

    Returns:
        Docker args tuple for json-file logging.
    """
    sandbox_log_dir = os.path.join(log_dir, "sandboxes", name)
    os.makedirs(sandbox_log_dir, exist_ok=True)
    log_path = os.path.join(sandbox_log_dir, "container.log")
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
            logger.debug(f"{message} ({elapsed:.0f}s elapsed)")

    return task.result()


def _get_modal_app_name() -> str:
    """Generate unique Modal app name to avoid conflicts between concurrent runs."""
    return f"swerex-{uuid.uuid4().hex[:12]}"


class SandboxExecutor:
    """Executor for sandboxed command execution via SWE-ReX.

    This class manages the lifecycle of a SWE-ReX deployment for executing
    commands in an isolated container environment.

    Usage:
        async with SandboxExecutor(config) as executor:
            result = await executor.execute("python --version")
            print(result)
    """

    def __init__(
        self,
        config: SandboxConfig,
        name: str | None = None,
        modal_app_name: str | None = None,
    ) -> None:
        """Initialize the sandbox executor.

        Args:
            config: Sandbox configuration.
            name: Optional name for logging (e.g., "sandbox-0").
            modal_app_name: Optional shared Modal app name for this run.
        """
        self.config = config
        self.name = name
        self._modal_app_name = modal_app_name
        self._deployment: Any = None
        self._runtime: Any = None
        self._session_created: bool = False
        self._session_lock: asyncio.Lock = asyncio.Lock()

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
        self._log(logging.DEBUG, "Starting sandbox deployment...")
        prefix = f"[{self.name}] " if self.name else ""

        # For Modal deployments, patch app lookup during deployment creation
        # to use unique app names and avoid conflicts between concurrent runs
        if self.config.mode == SandboxMode.MODAL:
            from unittest.mock import patch

            import modal

            app_name = self._modal_app_name or _get_modal_app_name()
            if not self._modal_app_name:
                # Only log if we generated it (manager logs its own)
                self._log(logging.INFO, f"Using Modal app: {app_name}")
            original_lookup = modal.App.lookup

            def patched_lookup(name: str, *args, **kwargs):
                if name == "swe-rex":
                    name = app_name
                return original_lookup(name, *args, **kwargs)

            with patch.object(modal.App, "lookup", patched_lookup):
                deployment = self.get_deployment()
        else:
            deployment = self.get_deployment()

        await _run_with_progress(
            deployment.start(),
            f"{prefix}Waiting for sandbox runtime",
            interval=5.0,
        )

        self._deployment = deployment
        self._runtime = deployment.runtime
        self._log(logging.DEBUG, "Sandbox deployment ready!")

        if (
            self.config.enable_diagnostics
            and self.config.log_dir
            and self.config.mode == SandboxMode.DOCKER
        ):
            await start_internal_monitor(self._runtime, self.name)

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

                # Get image, optionally building a derived image with swe-rex
                image = self.config.image
                if self.config.inject_swerex:
                    from .image import get_swerex_image

                    image = get_swerex_image(
                        image, self.config.container_runtime, self.config.dockerfile_extra
                    )

                # Build docker args, adding log args if log_dir is configured
                docker_args = list(self.config.docker_args) if self.config.docker_args else []
                if self.config.log_dir and self.name:
                    docker_args.extend(_get_log_docker_args(self.config.log_dir, self.name))
                    # Create container-specific subdirectory for system logs
                    # Mount only this container's subdirectory, not parent
                    # Use :Z for SELinux relabeling (safe for single-container access)
                    container_log_dir = os.path.join(self.config.log_dir, "sandboxes", self.name)
                    os.makedirs(container_log_dir, exist_ok=True, mode=0o777)
                    # Ensure writable even if directory already existed
                    os.chmod(container_log_dir, 0o777)
                    docker_args.extend(["-v", f"{container_log_dir}:/sandbox_logs:Z"])

                # Add environment variables as docker args
                for key, value in self.config.environment:
                    docker_args.extend(["-e", f"{key}={value}"])

                # Add volume mounts from config
                for host_path, container_path in self.config.volumes:
                    if self.config.container_runtime == "podman":
                        # Use --mount with bind-propagation=rslave for nested mounts
                        # This ensures mount propagation flows inward for DinD scenarios
                        docker_args.extend(
                            [
                                "--mount",
                                f"type=bind,src={host_path},dst={container_path},bind-propagation=rslave",
                            ]
                        )
                    else:
                        docker_args.extend(["-v", f"{host_path}:{container_path}"])

                # Build kwargs, omitting None values (swerex doesn't accept None for some fields)
                deployment_kwargs: dict[str, Any] = {
                    "image": image,
                    "container_runtime": self.config.container_runtime,
                    "startup_timeout": self.config.startup_timeout,
                }
                if docker_args:
                    deployment_kwargs["docker_args"] = docker_args
                if self.config.exec_shell:
                    deployment_kwargs["exec_shell"] = list(self.config.exec_shell)

                # Set image pull policy - default to "never" when we build the image ourselves
                if self.config.image_pull:
                    deployment_kwargs["pull"] = self.config.image_pull
                elif self.config.inject_swerex:
                    deployment_kwargs["pull"] = "never"

                return DockerDeployment(**deployment_kwargs)

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

                import modal

                # Build image locally and push to registry (same as Docker/Podman mode)
                # Modal will pull the pre-built image from the registry
                if self.config.inject_swerex:
                    from .image import get_swerex_image

                    # Build locally with swe-rex + dockerfile_extra, push to registry
                    image = get_swerex_image(
                        self.config.image,
                        self.config.container_runtime,
                        self.config.dockerfile_extra,
                        require_registry=True,  # Must push to registry for Modal
                    )
                    self._log(logging.DEBUG, f"Using pre-built swerex image: {image}")
                else:
                    image = self.config.image

                # Modal pulls pre-built image from registry (no pip_install/run_commands)
                if self.config.registry_auth:
                    provider = self.config.registry_auth.provider
                    match provider:
                        case "gcp":
                            # Default to gcp-service-account-json for GCP
                            secret_name = (
                                self.config.registry_auth.secret_name or "gcp-service-account-json"
                            )
                            self._log(
                                logging.DEBUG,
                                f"Modal pulling from GCP AR: {image}, secret={secret_name}",
                            )
                            secret = modal.Secret.from_name(secret_name)
                            modal_image = modal.Image.from_gcp_artifact_registry(
                                image, secret=secret
                            )
                        case "ecr":
                            secret_name = self.config.registry_auth.secret_name
                            if not secret_name:
                                raise ValueError("ECR registry_auth requires secret_name")
                            secret = modal.Secret.from_name(secret_name)
                            modal_image = modal.Image.from_aws_ecr(image, secret=secret)
                        case "docker":
                            secret_name = self.config.registry_auth.secret_name
                            if not secret_name:
                                raise ValueError("Docker registry_auth requires secret_name")
                            secret = modal.Secret.from_name(secret_name)
                            modal_image = modal.Image.from_registry(image, secret=secret)
                        case _:
                            modal_image = modal.Image.from_registry(image)
                else:
                    modal_image = modal.Image.from_registry(image)

                return ModalDeployment(
                    image=modal_image,
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

        self._log(logging.DEBUG, "Sandbox stopped")

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

        try:
            response = await self._runtime.execute(
                Command(
                    command=["bash", "-c", command],
                    timeout=effective_timeout,
                )
            )
        except Exception as e:
            # Check for timeout errors (swerex.exceptions.CommandTimeoutError)
            if "CommandTimeoutError" in type(e).__name__ or "timed out" in str(e).lower():
                return ExecutionResult(
                    success=False,
                    output=f"Command timed out after {effective_timeout}s",
                    exit_code=-1,
                    error="timeout",
                )
            raise

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
        output_truncated = False
        max_output_size = 5 * 1024 * 1024  # 5MB
        keep_after_truncate = 1 * 1024 * 1024  # Keep last 1MB after truncation

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
            # Also check if output file is too large and truncate if needed
            poll_start = time.time()
            try:
                poll_cmd = (
                    f"size=$(stat -c%s {output_file} 2>/dev/null || echo 0); "
                    f'if [ "$size" -gt {max_output_size} ]; then '
                    f"  tail -c {keep_after_truncate} {output_file} > {output_file}.tmp && "
                    f"  mv {output_file}.tmp {output_file}; "
                    f"  echo '---TRUNCATED---'; "
                    f"fi; "
                    f"tail -c +{last_pos + 1} {output_file} 2>/dev/null; "
                    f"echo '---EXIT_CODE---'; "
                    f"cat {exit_code_file} 2>/dev/null"
                )
                resp = await self._runtime.execute(
                    Command(command=["bash", "-c", poll_cmd], timeout=10.0)
                )
                poll_duration = time.time() - poll_start
                consecutive_failures = 0  # Reset on success

                if poll_duration > 5.0:
                    self._log(logging.WARNING, f"Poll slow ({poll_duration:.1f}s)")

                stdout = resp.stdout or ""

                # Check if output was truncated
                if "---TRUNCATED---" in stdout:
                    if not output_truncated:
                        self._log(
                            logging.WARNING,
                            f"Output exceeded {max_output_size // (1024 * 1024)}MB, "
                            f"truncating to last {keep_after_truncate // 1024}KB",
                        )
                        output_truncated = True
                    # Reset position since file was truncated
                    last_pos = 0
                    stdout = stdout.replace("---TRUNCATED---\n", "").replace("---TRUNCATED---", "")

                parts = stdout.split("---EXIT_CODE---")
                new_output = parts[0] if parts else ""
                exit_marker = parts[1].strip() if len(parts) > 1 else ""

                if new_output:
                    last_pos += len(new_output)
                    for line in new_output.rstrip("\n").split("\n"):
                        if line:
                            logger.info(f"[{prefix}] {line}")

                if exit_marker:
                    break

            except TimeoutError:
                poll_duration = time.time() - poll_start
                consecutive_failures += 1
                self._log(
                    logging.WARNING,
                    f"Poll timed out after {poll_duration:.1f}s "
                    f"({consecutive_failures}/{max_consecutive_failures}) "
                    "- sandbox may be under resource pressure",
                )
            except aiohttp.ClientConnectionError as e:
                poll_duration = time.time() - poll_start
                consecutive_failures += 1
                self._log(
                    logging.WARNING,
                    f"Poll connection failed after {poll_duration:.1f}s "
                    f"({consecutive_failures}/{max_consecutive_failures}): {e} "
                    "- swerex server may be down",
                )
            except aiohttp.ClientError as e:
                poll_duration = time.time() - poll_start
                consecutive_failures += 1
                self._log(
                    logging.WARNING,
                    f"Poll network error after {poll_duration:.1f}s "
                    f"({consecutive_failures}/{max_consecutive_failures}): {e}",
                )
            except Exception as e:
                poll_duration = time.time() - poll_start
                consecutive_failures += 1
                self._log(
                    logging.WARNING,
                    f"Poll failed after {poll_duration:.1f}s "
                    f"({consecutive_failures}/{max_consecutive_failures}): "
                    f"{type(e).__name__}: {e}",
                )

            # Check if we've exceeded max consecutive failures (after any exception)
            if consecutive_failures >= max_consecutive_failures:
                self._log(logging.ERROR, "Sandbox unresponsive, aborting")
                await kill_process_group()
                await cleanup_temp_files()
                diagnostics_path = None
                if self.config.log_dir and self.name:
                    diagnostics_path = os.path.join(
                        self.config.log_dir, "sandboxes", self.name, "stats.log"
                    )
                msg = f"Sandbox unresponsive after {consecutive_failures} polls"
                if diagnostics_path:
                    msg += f"\nDiagnostics available at: {diagnostics_path}"
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

        if output_truncated:
            truncate_msg = (
                f"[Output truncated: exceeded {max_output_size // (1024 * 1024)}MB, "
                f"showing last {keep_after_truncate // (1024 * 1024)}MB]"
            )
            full_output = truncate_msg + "\n" + full_output if full_output else truncate_msg

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

        async with self._session_lock:
            # Double-check after acquiring lock
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
