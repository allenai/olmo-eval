"""Manager for multiple sandbox executors with capability-based routing."""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextlib
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from olmo_eval.common.execution.environment import ExecutionResult
from olmo_eval.common.logging import configure_worker_logging

from .config import Capability, SandboxConfig, SandboxMode
from .executor import SandboxExecutor


@dataclass
class ExecutorBinding:
    """Pins a caller to a specific executor for session state continuity."""

    id: str
    executor: SandboxExecutor
    capabilities: frozenset[str]
    _manager: SandboxManager
    _released: bool = field(default=False, repr=False)

    async def execute_in_session(
        self, command: str, timeout: float | None = None
    ) -> ExecutionResult:
        """Execute in the bound executor's bash session."""
        if self._released:
            raise RuntimeError("Binding has been released")
        return await self.executor.execute_in_session(command, timeout)

    async def release(self) -> None:
        if not self._released:
            self._released = True
            await self._manager._release_binding(self)

    async def __aenter__(self) -> ExecutorBinding:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.release()


class SandboxManager:
    """Manages multiple sandbox executors with capability-based routing.

    Executors are selected using round-robin among those that support the
    required capabilities. For session-based execution requiring state
    continuity, use acquire_binding() or the binding() context manager.

    Usage:
        from olmo_eval.harness.sandbox import Capability

        configs = [SandboxConfig(...), SandboxConfig(...)]
        manager = SandboxManager(configs, owner="scorer")
        await manager.start()
        try:
            result = await manager.execute("echo hello", capabilities=Capability.BASH)
        finally:
            await manager.stop()
    """

    def __init__(self, configs: Sequence[SandboxConfig], owner: str = "default") -> None:
        """Initialize the sandbox manager.

        Args:
            configs: Sequence of sandbox configurations to manage.
            owner: Identifier for the owner of these sandboxes (e.g., "agent", "scorer").
                Used in log messages to distinguish sandbox instances.
        """
        self._configs = list(configs)
        self._owner = owner
        self._logger = configure_worker_logging("sb-manager")
        self._executors: list[SandboxExecutor] = []
        self._round_robin_indices: dict[frozenset[str], int] = {}
        self._execution_semaphores: dict[frozenset[str], asyncio.Semaphore] = {}
        self._bindings: dict[str, ExecutorBinding] = {}
        self._bound_executors: set[int] = set()
        self._binding_lock: asyncio.Lock = asyncio.Lock()
        self._binding_counter: int = 0
        self._modal_app_name: str | None = None

        # Generate shared Modal app name if any config uses Modal
        if any(c.mode == SandboxMode.MODAL for c in self._configs):
            self._modal_app_name = f"swerex-{uuid.uuid4().hex[:12]}"
            self._logger.info(f"Using Modal app: {self._modal_app_name}")

    async def start(self) -> None:
        """Start all sandbox executors.

        Uses thread pool to avoid event loop blocking from swe-rex subprocess calls.
        Allows partial failures if min_instances is configured on the sandbox config.
        """
        # Track per-type instance indices for naming
        type_indices: dict[str, int] = {}
        # Track which config each executor belongs to: executor index -> config index
        executor_to_config: dict[int, int] = {}

        # Create all executors first
        executor_idx = 0
        for config_idx, config in enumerate(self._configs):
            # Derive type name from capabilities, replacing ':' to avoid
            # breaking podman volume mount paths (host:container separator)
            type_name = "+".join(sorted(config.capabilities)) or str(config_idx)
            safe_type_name = type_name.replace(":", "_")

            for _ in range(config.resolved_instances):
                idx = type_indices.get(type_name, 0)
                name = f"sb-{safe_type_name}-{self._owner}-{idx}"
                type_indices[type_name] = idx + 1

                executor = SandboxExecutor(config, name=name, modal_app_name=self._modal_app_name)
                self._executors.append(executor)
                executor_to_config[executor_idx] = config_idx
                executor_idx += 1

        # Start all executors in thread pool to avoid blocking event loop
        # swe-rex's DockerDeployment.start() has blocking subprocess calls
        start_time = time.time()
        self._logger.info(f"Starting {len(self._executors)} sandbox executors...")

        def start_in_thread(executor: SandboxExecutor) -> None:
            """Run executor.start() in a dedicated thread with its own event loop."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(executor.start())
            finally:
                loop.close()

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._executors)) as pool:
            futures = [loop.run_in_executor(pool, start_in_thread, e) for e in self._executors]
            results = await asyncio.gather(*futures, return_exceptions=True)

        # Track successes and failures per config
        num_configs = len(self._configs)
        config_successes: dict[int, list[SandboxExecutor]] = {i: [] for i in range(num_configs)}
        config_failures: dict[int, list[tuple[SandboxExecutor, BaseException]]] = {
            i: [] for i in range(num_configs)
        }

        for exec_idx, (executor, result) in enumerate(zip(self._executors, results, strict=True)):
            config_idx = executor_to_config[exec_idx]
            if isinstance(result, BaseException):
                config_failures[config_idx].append((executor, result))
                self._logger.warning(f"Executor {executor.name} failed to start: {result}")
            else:
                config_successes[config_idx].append(executor)

        # Check minimum requirements per config
        for config_idx, config in enumerate(self._configs):
            min_required = (
                config.min_instances
                if config.min_instances is not None
                else config.resolved_instances
            )
            started_count = len(config_successes[config_idx])
            failed_count = len(config_failures[config_idx])

            if started_count < min_required:
                raise RuntimeError(
                    f"Sandbox config {config_idx} ({config.image}): "
                    f"only {started_count}/{min_required} required instances started "
                    f"({failed_count} failed)"
                )

            if failed_count > 0:
                self._logger.warning(
                    f"Sandbox config {config_idx} ({config.image}): "
                    f"{started_count}/{config.resolved_instances} instances started "
                    f"({failed_count} failed, min_required={min_required})"
                )

        # Keep only successfully started executors
        self._executors = [e for successes in config_successes.values() for e in successes]

        # Build per-capability execution semaphores from running executors
        cap_counts: dict[frozenset[str], int] = {}
        cap_mc: dict[frozenset[str], int] = {}
        for e in self._executors:
            cap = e.config.capabilities
            cap_counts[cap] = cap_counts.get(cap, 0) + 1
            cap_mc[cap] = e.config.max_concurrency
        for cap, count in cap_counts.items():
            limit = cap_mc[cap] * count
            self._execution_semaphores[cap] = asyncio.Semaphore(limit)
            self._logger.info(
                f"Execution semaphore for {sorted(cap)}: "
                f"{limit} ({cap_mc[cap]} x {count} instances)"
            )

        elapsed = time.time() - start_time
        total_attempted = sum(c.resolved_instances for c in self._configs)
        self._logger.info(
            f"Started {len(self._executors)}/{total_attempted} sandbox executors in {elapsed:.1f}s"
        )

        atexit.register(self._atexit_cleanup)

    async def stop(self) -> None:
        """Stop all sandbox executors."""
        async with self._binding_lock:
            for binding in self._bindings.values():
                binding._released = True
            self._bindings.clear()
            self._bound_executors.clear()

        await asyncio.gather(*[e.stop() for e in self._executors])
        self._executors.clear()
        self._round_robin_indices.clear()
        self._logger.info("All sandboxes stopped")

        with contextlib.suppress(Exception):
            atexit.unregister(self._atexit_cleanup)

    def _atexit_cleanup(self) -> None:
        """Synchronous cleanup for atexit. Runs stop() if executors are still active."""
        if not self._executors:
            return
        self._logger.info("Cleaning up sandboxes on exit")
        try:
            asyncio.run(self.stop())
        except Exception as e:
            self._logger.error(f"Sandbox cleanup failed: {e}")

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
            if required_capabilities <= e.config.capabilities and i not in self._bound_executors
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

    def get_execution_semaphore(
        self, required_capabilities: frozenset[str]
    ) -> asyncio.Semaphore | None:
        """Get the execution semaphore for the given capabilities.

        Returns a shared semaphore sized to max_concurrency * running_instances
        for the matching capability set. Returns None if no match.
        """
        for cap, sem in self._execution_semaphores.items():
            if required_capabilities <= cap:
                return sem
        return None

    async def execute(
        self,
        command: str,
        timeout: float | None = None,
        capabilities: frozenset[str] | None = None,
    ) -> str:
        """Execute a command on a sandbox.

        Args:
            command: The command to execute.
            timeout: Optional timeout override in seconds.
            capabilities: Optional required capabilities. If None, uses default.

        Returns:
            The command output.
        """
        executor = self.get_executor(capabilities or Capability.DEFAULT)
        return await executor.execute(command, timeout)

    async def execute_command(
        self,
        command: str,
        timeout: float | None = None,
        capabilities: frozenset[str] | None = None,
    ) -> ExecutionResult:
        """Execute a command and return structured result.

        Args:
            command: The command to execute.
            timeout: Optional timeout override in seconds.
            capabilities: Optional required capabilities. If None, uses default.

        Returns:
            ExecutionResult with success status, output, and exit code.
        """
        executor = self.get_executor(capabilities or Capability.DEFAULT)
        return await executor.execute_command(command, timeout)

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
        capabilities: frozenset[str] | None = None,
    ) -> ExecutionResult:
        """Execute code in the specified language.

        Implements the ExecutionEnvironment protocol by delegating to the
        first available executor.

        Args:
            code: Source code to execute.
            language: Programming language (default: "python").
            timeout: Optional timeout in seconds.
            capabilities: Optional required capabilities. If None, uses default.

        Returns:
            ExecutionResult with success status and output.
        """
        executor = self.get_executor(capabilities or Capability.DEFAULT)
        return await executor.execute_code(code, language, timeout)

    async def acquire_binding(
        self,
        capabilities: frozenset[str] | None = None,
    ) -> ExecutorBinding:
        """Acquire exclusive binding to an executor for session execution."""
        required = capabilities or frozenset()
        async with self._binding_lock:
            available = [
                (i, e)
                for i, e in enumerate(self._executors)
                if required <= e.config.capabilities and i not in self._bound_executors
            ]
            if not available:
                raise ValueError(f"No available executor for {required}")

            executor_idx, executor = available[0]
            self._binding_counter += 1
            binding = ExecutorBinding(
                id=f"binding-{self._binding_counter}",
                executor=executor,
                capabilities=required,
                _manager=self,
            )
            self._bindings[binding.id] = binding
            self._bound_executors.add(executor_idx)
            return binding

    async def _release_binding(self, binding: ExecutorBinding) -> None:
        async with self._binding_lock:
            if binding.id not in self._bindings:
                return
            for idx, e in enumerate(self._executors):
                if e is binding.executor:
                    self._bound_executors.discard(idx)
                    break
            del self._bindings[binding.id]

    @asynccontextmanager
    async def binding(
        self,
        capabilities: frozenset[str] | None = None,
    ) -> AsyncIterator[ExecutorBinding]:
        """Context manager for executor binding."""
        b = await self.acquire_binding(capabilities)
        try:
            yield b
        finally:
            await b.release()

    @property
    def is_running(self) -> bool:
        """Check if any executors are running."""
        return len(self._executors) > 0 and all(e.is_running for e in self._executors)

    @property
    def executor_count(self) -> int:
        """Number of active executors."""
        return len(self._executors)
