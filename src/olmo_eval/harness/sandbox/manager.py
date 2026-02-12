"""Manager for multiple sandbox executors with capability-based routing."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from olmo_eval.common.execution.environment import ExecutionResult

from .config import SandboxConfig
from .executor import SandboxExecutor

logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages multiple sandbox executors with capability-based routing.

    Executors are selected using round-robin among those that support the
    required capabilities.

    Usage:
        configs = [SandboxConfig(...), SandboxConfig(...)]
        manager = SandboxManager(configs)
        await manager.start()
        try:
            result = await manager.execute("echo hello", frozenset({"bash"}))
        finally:
            await manager.stop()
    """

    def __init__(self, configs: Sequence[SandboxConfig]) -> None:
        """Initialize the sandbox manager.

        Args:
            configs: Sequence of sandbox configurations to manage.
        """
        self._configs = list(configs)
        self._executors: list[SandboxExecutor] = []
        self._round_robin_indices: dict[frozenset[str], int] = {}

    async def start(self) -> None:
        """Start all sandbox executors."""
        for config in self._configs:
            for i in range(config.instances):
                executor = SandboxExecutor(config)
                await executor.start()
                self._executors.append(executor)
                if config.instances > 1:
                    logger.info(
                        f"Started sandbox {i + 1}/{config.instances} "
                        f"with capabilities: {config.capabilities}"
                    )
                else:
                    logger.info(f"Started sandbox with capabilities: {config.capabilities}")

    async def stop(self) -> None:
        """Stop all sandbox executors."""
        for executor in self._executors:
            await executor.stop()
        self._executors.clear()
        self._round_robin_indices.clear()
        logger.info("All sandboxes stopped")

    def get_executor(self, required_capabilities: frozenset[str]) -> SandboxExecutor:
        """Get an executor that supports the required capabilities.

        Uses round-robin selection among matching executors.

        Args:
            required_capabilities: Set of capabilities the executor must support.

        Returns:
            A SandboxExecutor that supports all required capabilities.

        Raises:
            ValueError: If no executor supports the required capabilities.
        """
        matching = [
            (i, e)
            for i, e in enumerate(self._executors)
            if required_capabilities <= e.config.capabilities
        ]

        if not matching:
            available = [e.config.capabilities for e in self._executors]
            raise ValueError(
                f"No sandbox supports capabilities {required_capabilities}. Available: {available}"
            )

        # Round-robin selection
        key = required_capabilities
        idx = self._round_robin_indices.get(key, 0)
        selected_idx = idx % len(matching)
        self._round_robin_indices[key] = idx + 1

        return matching[selected_idx][1]

    async def execute(
        self,
        command: str,
        timeout: float | None = None,
    ) -> str:
        """Execute a command on the first available sandbox.

        This method implements the ExecutionEnvironment protocol.
        For capability-based routing, use execute_with_capabilities.

        Args:
            command: The command to execute.
            timeout: Optional timeout override in seconds.

        Returns:
            The command output.
        """
        executor = self.get_executor(frozenset())
        return await executor.execute(command, timeout)

    async def execute_with_capabilities(
        self,
        command: str,
        required_capabilities: frozenset[str],
        timeout: float | None = None,
    ) -> str:
        """Execute a command on a sandbox with required capabilities.

        Uses round-robin selection among executors that support the
        required capabilities.

        Args:
            command: The command to execute.
            required_capabilities: Capabilities needed for execution.
            timeout: Optional timeout override in seconds.

        Returns:
            The command output.
        """
        executor = self.get_executor(required_capabilities)
        return await executor.execute(command, timeout)

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute code in the specified language.

        Implements the ExecutionEnvironment protocol by delegating to the
        first available executor.

        Args:
            code: Source code to execute.
            language: Programming language (default: "python").
            timeout: Optional timeout in seconds.

        Returns:
            ExecutionResult with success status and output.
        """
        executor = self.get_executor(frozenset())
        return await executor.execute_code(code, language, timeout)

    @property
    def is_running(self) -> bool:
        """Check if any executors are running."""
        return len(self._executors) > 0 and all(e.is_running for e in self._executors)

    @property
    def executor_count(self) -> int:
        """Number of active executors."""
        return len(self._executors)
