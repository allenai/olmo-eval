"""Sandbox executor for isolated command execution via SWE-ReX."""

from __future__ import annotations

import logging
import shutil
import uuid
from typing import TYPE_CHECKING, Any

from olmo_eval.common.execution.environment import ExecutionResult

from .config import SandboxConfig, SandboxMode

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SandboxExecutor:
    """Executor for sandboxed command execution via SWE-ReX.

    This class manages the lifecycle of a SWE-ReX deployment for executing
    commands in an isolated container environment.

    Usage:
        async with SandboxExecutor(config) as executor:
            result = await executor.execute("python --version")
            print(result)
    """

    def __init__(self, config: SandboxConfig) -> None:
        """Initialize the sandbox executor.

        Args:
            config: Sandbox configuration.
        """
        self.config = config
        self._deployment: Any = None
        self._runtime: Any = None
        self._session: Any = None

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
        """Start the sandbox deployment and create a session.

        Raises:
            ImportError: If swe-rex is not installed.
            RuntimeError: If container runtime is not available.
        """
        deployment = self._create_deployment()

        logger.info(f"Starting sandbox deployment: {type(deployment).__name__}")
        await deployment.start(timeout=self.config.startup_timeout)

        self._deployment = deployment
        self._runtime = await deployment.get_runtime()

        # Create a persistent bash session for command execution
        from swerex.runtime.abstract import CreateBashSessionRequest

        session_response = await self._runtime.create_session(
            CreateBashSessionRequest(startup_timeout=30.0)
        )
        self._session = session_response.session_id
        logger.info(f"Sandbox session created: {self._session}")

    def _create_deployment(self) -> Any:
        """Create the appropriate deployment based on configuration.

        Returns:
            A deployment instance.

        Raises:
            ImportError: If swe-rex is not installed.
            RuntimeError: If the requested container runtime is not available.
        """
        match self.config.mode:
            case SandboxMode.DOCKER:
                from swerex.deployment.docker import DockerDeployment

                if not shutil.which("docker"):
                    raise RuntimeError("Docker not available but mode=SandboxMode.DOCKER")
                return DockerDeployment(
                    image=self.config.image,
                    container_runtime="docker",
                )

            case SandboxMode.LOCAL:
                from swerex.deployment.local import LocalDeployment

                logger.warning(
                    "Using local deployment (unsandboxed). Commands will run on host system."
                )
                return LocalDeployment()

            case SandboxMode.MODAL:
                from swerex.deployment.modal import ModalDeployment

                return ModalDeployment(
                    image=self.config.image,
                    runtime_timeout=self.config.runtime_timeout,
                    modal_sandbox_kwargs=self.config.modal_sandbox_kwargs,
                )

    async def stop(self) -> None:
        """Stop the sandbox deployment and clean up resources."""
        if self._session is not None and self._runtime is not None:
            try:
                from swerex.runtime.abstract import CloseBashSessionRequest

                await self._runtime.close_session(CloseBashSessionRequest(session_id=self._session))
            except Exception as e:
                logger.warning(f"Failed to close session: {e}")
            self._session = None

        if self._deployment is not None:
            try:
                await self._deployment.stop()
            except Exception as e:
                logger.warning(f"Failed to stop deployment: {e}")
            self._deployment = None
            self._runtime = None

        logger.info("Sandbox stopped")

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
        if self._runtime is None or self._session is None:
            raise RuntimeError("Sandbox not started. Call start() first or use async context.")

        from swerex.runtime.abstract import BashAction

        effective_timeout = timeout if timeout is not None else self.config.command_timeout

        response = await self._runtime.run_in_session(
            BashAction(
                session_id=self._session,
                command=command,
                timeout=effective_timeout,
            )
        )

        # Combine stdout and stderr, include exit code information
        output_parts = []
        if response.output:
            output_parts.append(response.output)
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

        This method writes the code to a temporary file in the sandbox,
        executes it, and returns the result.

        Args:
            code: Source code to execute.
            language: Programming language (default: "python").
            timeout: Optional timeout in seconds.

        Returns:
            ExecutionResult with success status and output.
        """
        if self._runtime is None or self._session is None:
            return ExecutionResult(
                success=False,
                error="Sandbox not started. Call start() first or use async context.",
            )

        # Map language to file extension and interpreter
        lang_config = {
            "python": {"ext": ".py", "cmd": "python"},
            "python3": {"ext": ".py", "cmd": "python3"},
            "bash": {"ext": ".sh", "cmd": "bash"},
            "sh": {"ext": ".sh", "cmd": "sh"},
        }

        config = lang_config.get(language.lower())
        if config is None:
            return ExecutionResult(
                success=False,
                error=f"Unsupported language: {language}",
            )

        # Generate unique filename
        file_id = uuid.uuid4().hex[:8]
        filename = f"/tmp/code_{file_id}{config['ext']}"

        try:
            from swerex.runtime.abstract import BashAction

            effective_timeout = timeout if timeout is not None else self.config.command_timeout

            # Write code to file using heredoc (no escaping needed)
            write_cmd = f"cat > {filename} << 'CODEEOF'\n{code}\nCODEEOF"
            await self._runtime.run_in_session(
                BashAction(
                    session_id=self._session,
                    command=write_cmd,
                    timeout=10.0,
                )
            )

            # Execute the code
            exec_cmd = f"{config['cmd']} {filename}"
            response = await self._runtime.run_in_session(
                BashAction(
                    session_id=self._session,
                    command=exec_cmd,
                    timeout=effective_timeout,
                )
            )

            # Cleanup
            await self._runtime.run_in_session(
                BashAction(
                    session_id=self._session,
                    command=f"rm -f {filename}",
                    timeout=5.0,
                )
            )

            return ExecutionResult(
                success=response.exit_code == 0,
                output=response.output or "",
                exit_code=response.exit_code,
            )

        except Exception as e:
            logger.warning(f"Code execution failed: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
            )

    @property
    def is_running(self) -> bool:
        """Check if the sandbox is running."""
        return self._deployment is not None and self._session is not None
