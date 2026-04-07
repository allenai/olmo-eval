"""Terminal-Bench 2.0 external evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from olmo_eval.common.types import LMRequest, RequestType
from olmo_eval.common.types.trajectory import AgentTrajectory
from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.network import get_docker_network_args
from olmo_eval.evals.external.result import ExternalEvalResult
from olmo_eval.harness.sandbox.config import ContainerRuntime, SandboxConfig, SandboxMode
from olmo_eval.harness.sandbox.image import get_swerex_image

from .loader import TerminalBenchLoader
from .task import TerminalBenchTask
from .verifier import TerminalBenchVerifier

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox import SandboxManager
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are an AI assistant helping complete tasks in a Linux terminal.
You have access to the following tools:
- execute_bash_session(command): Execute a bash command in a persistent shell session
- submit(): Call when you have completed the task

Focus on completing the task described in the instructions. Work step by step,
checking your progress as you go. When you believe the task is complete, call
the submit() tool.
"""


@dataclass
class TerminalBenchArgs:
    """Arguments for Terminal-Bench 2.0 evaluation."""

    task_ids: list[str] | None = None
    repo_path: str | None = None
    repo_ref: str = TerminalBenchLoader.DEFAULT_REF
    max_concurrency: int = 1
    max_turns: int = 50
    oracle: bool = False
    sandbox_mode: str = "docker"
    enable_compaction: bool = True
    backend: str = "openai_agents"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TerminalBenchArgs:
        task_ids = data.get("task_ids")
        if isinstance(task_ids, str):
            task_ids = [t.strip() for t in task_ids.split(",") if t.strip()]
        return cls(
            task_ids=task_ids,
            repo_path=data.get("repo_path"),
            repo_ref=data.get("repo_ref", TerminalBenchLoader.DEFAULT_REF),
            max_concurrency=int(data.get("max_concurrency", 1)),
            max_turns=int(data.get("max_turns", 50)),
            oracle=data.get("oracle", False) in (True, "true", "True", "1"),
            sandbox_mode=data.get("sandbox_mode", "docker"),
            enable_compaction=data.get("enable_compaction", True) in (True, "true", "True", "1", 1),
            backend=data.get("backend", "openai_agents"),
        )


@dataclass
class TaskResult:
    """Result of executing a single Terminal-Bench task."""

    task_id: str
    reward: float
    trajectory: AgentTrajectory
    completion_reason: str
    agent_duration: float
    verification_output: str
    verification_exit_code: int
    error: str | None = None
    difficulty: str = "unknown"
    category: str = "unknown"


class TerminalBenchExternalEval(ExternalEval):
    """Terminal-Bench 2.0 evaluation with per-task container orchestration."""

    @property
    def name(self) -> str:
        return "terminal_bench_2"

    @property
    def description(self) -> str:
        return "Evaluates LLM agents on 89 diverse terminal tasks"

    @property
    def timeout_seconds(self) -> float:
        return 12000.0  # Max timeout across all tasks

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "task_ids": ("Comma-separated task IDs to run (default: all)", None),
            "repo_path": ("Local repo path (default: clone fresh)", None),
            "repo_ref": ("Git ref to checkout", TerminalBenchLoader.DEFAULT_REF),
            "max_concurrency": ("Max parallel containers", 1),
            "max_turns": ("Max agent turns per task", 50),
            "oracle": ("Run solve.sh instead of LLM agent", False),
            "sandbox_mode": ("Sandbox mode: docker, modal", "docker"),
            "backend": ("Backend to use for agent execution", "openai_agents"),
        }

    @property
    def backend(self) -> str | None:
        return "openai_agents"

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        """Execute Terminal-Bench evaluation.

        Args:
            provider: Inference provider for LLM calls.
            args: Evaluation-specific arguments.
            output_dir: Directory to write results.
            container_runtime: Container runtime (docker or podman).

        Returns:
            ExternalEvalResult with metrics and per-task results.
        """
        start_time = time.time()
        tb_args = TerminalBenchArgs.from_dict(args)

        # Validate backend early to fail fast before spinning up sandboxes
        from olmo_eval.harness.backends import validate_backend

        validate_backend(tb_args.backend)

        # Load tasks
        loader = TerminalBenchLoader()
        if tb_args.repo_path:
            repo_dir = Path(tb_args.repo_path)
        else:
            # Clone to a cache directory (not output_dir to avoid copying repo to results)
            repo_dir = Path("/tmp") / "terminal-bench-2-cache"
            loader.ensure_repo(repo_dir, tb_args.repo_ref)

        tasks = loader.load_tasks(repo_dir, tb_args.task_ids)
        if not tasks:
            return self._error_result(
                "No tasks found", start_time, f"repo_dir={repo_dir}, task_ids={tb_args.task_ids}"
            )

        # Execute tasks with concurrency limit
        semaphore = asyncio.Semaphore(tb_args.max_concurrency)

        async def run_task(task: TerminalBenchTask) -> TaskResult:
            async with semaphore:
                return await self._execute_task(
                    task=task,
                    provider=provider,
                    container_runtime=container_runtime,
                    max_turns=tb_args.max_turns,
                    oracle_mode=tb_args.oracle,
                    sandbox_mode=tb_args.sandbox_mode,
                    enable_compaction=tb_args.enable_compaction,
                    backend_name=tb_args.backend,
                )

        results = await asyncio.gather(*[run_task(t) for t in tasks], return_exceptions=True)

        # Process results
        task_results: list[TaskResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                task = tasks[i]
                logger.error(f"Task {task.task_id} failed with exception: {result}")
                task_results.append(
                    TaskResult(
                        task_id=task.task_id,
                        reward=0.0,
                        trajectory=AgentTrajectory(turns=()),
                        completion_reason="error",
                        agent_duration=0.0,
                        verification_output="",
                        verification_exit_code=-1,
                        error=str(result),
                        difficulty=task.difficulty,
                        category=task.category,
                    )
                )
            else:
                task_results.append(result)

        # Compute metrics
        rewards = [r.reward for r in task_results]
        pass_rate = sum(rewards) / len(rewards) if rewards else 0.0

        metrics: dict[str, float] = {
            "pass_rate": pass_rate,
            "num_tasks": len(task_results),
            "num_passed": sum(1 for r in task_results if r.reward == 1.0),
        }

        # Group by difficulty
        by_difficulty: dict[str, list[float]] = {}
        by_category: dict[str, list[float]] = {}
        for r in task_results:
            by_difficulty.setdefault(r.difficulty, []).append(r.reward)
            by_category.setdefault(r.category, []).append(r.reward)

        for difficulty, rewards_list in by_difficulty.items():
            metrics[f"pass_rate_{difficulty}"] = sum(rewards_list) / len(rewards_list)

        for category, rewards_list in by_category.items():
            safe_name = category.replace(" ", "_").lower()
            metrics[f"pass_rate_{safe_name}"] = sum(rewards_list) / len(rewards_list)

        predictions = self._build_predictions(task_results)

        result = ExternalEvalResult(
            name=self.name,
            success=True,
            metrics=metrics,
            metadata={
                "model_name": provider.model_name,
                "oracle_mode": tb_args.oracle,
                "max_turns": tb_args.max_turns,
                "repo_ref": tb_args.repo_ref,
            },
            duration_seconds=time.time() - start_time,
            predictions=predictions,
        )

        # Save results
        if output_dir:
            self._save_results(result, output_dir)
            self._save_task_results(task_results, output_dir)

        return result

    async def _execute_task(
        self,
        task: TerminalBenchTask,
        provider: InferenceProvider,
        container_runtime: str,
        max_turns: int,
        oracle_mode: bool,
        sandbox_mode: str,
        enable_compaction: bool = True,
        backend_name: str = "openai_agents",
    ) -> TaskResult:
        """Execute a single Terminal-Bench task.

        Args:
            task: The task to execute.
            provider: Inference provider for LLM calls.
            container_runtime: Container runtime.
            max_turns: Maximum agent turns.
            oracle_mode: Whether to run the solution script instead of agent.
            sandbox_mode: Sandbox mode (docker, modal).
            enable_compaction: Enable context compaction for long conversations.

        Returns:
            TaskResult with reward and trajectory.
        """
        from olmo_eval.harness.sandbox import SandboxManager

        logger.info(f"Executing task: {task.task_id}")
        task_start = time.time()

        # Create sandbox config for this task
        if sandbox_mode == "docker":
            mode = SandboxMode.DOCKER
        elif sandbox_mode == "modal":
            mode = SandboxMode.MODAL
        else:
            raise ValueError(
                f"Invalid sandbox_mode: {sandbox_mode!r}. Must be 'docker' or 'modal'."
            )
        runtime = cast(ContainerRuntime, container_runtime)

        docker_args: tuple[str, ...] = ()
        if mode == SandboxMode.DOCKER:
            docker_args = tuple(get_docker_network_args(runtime))

        image = get_swerex_image(task.image, runtime)

        sandbox_config = SandboxConfig(
            image=image,
            mode=mode,
            container_runtime=runtime if mode == SandboxMode.DOCKER else "docker",
            working_dir=task.working_dir,
            command_timeout=task.agent_timeout,
            docker_args=docker_args,
        )

        sandbox_manager = SandboxManager([sandbox_config], owner=f"tb2-{task.task_id}")

        try:
            await sandbox_manager.start()

            if oracle_mode:
                trajectory, completion_reason = await self._run_oracle(sandbox_manager, task)
            else:
                trajectory, completion_reason = await self._run_agent(
                    sandbox_manager, task, provider, max_turns, enable_compaction, backend_name
                )

            agent_duration = time.time() - task_start

            executor = sandbox_manager.get_executor(frozenset())
            verifier = TerminalBenchVerifier()
            await verifier.inject_tests(executor, task.test_files)
            verification = await verifier.run_verification(
                executor, task.verifier_timeout, task.working_dir, task.task_id
            )

            return TaskResult(
                task_id=task.task_id,
                reward=verification.reward,
                trajectory=trajectory,
                completion_reason=completion_reason,
                agent_duration=agent_duration,
                verification_output=verification.test_output,
                verification_exit_code=verification.test_exit_code,
                difficulty=task.difficulty,
                category=task.category,
            )

        except Exception as e:
            logger.exception(f"Task {task.task_id} failed")
            return TaskResult(
                task_id=task.task_id,
                reward=0.0,
                trajectory=AgentTrajectory(turns=()),
                completion_reason="error",
                agent_duration=time.time() - task_start,
                verification_output="",
                verification_exit_code=-1,
                error=str(e),
                difficulty=task.difficulty,
                category=task.category,
            )
        finally:
            await sandbox_manager.stop()

    async def _run_oracle(
        self,
        sandbox_manager: SandboxManager,
        task: TerminalBenchTask,
    ) -> tuple[AgentTrajectory, str]:
        """Run the oracle (solution script).

        Args:
            sandbox_manager: The sandbox manager.
            task: The task to run.

        Returns:
            Tuple of (trajectory, completion_reason).
        """
        executor = sandbox_manager.get_executor(frozenset())

        result = await executor.execute_in_session(
            f"bash << 'SOLVEEOF'\n{task.solution_script}\nSOLVEOF",
            timeout=task.agent_timeout,
            stream=True,
            log_prefix=f"tb2-{task.task_id}-oracle",
        )

        logger.info(f"Oracle exit code: {result.exit_code}")
        return AgentTrajectory(turns=()), "oracle"

    async def _run_agent(
        self,
        sandbox_manager: SandboxManager,
        task: TerminalBenchTask,
        provider: InferenceProvider,
        max_turns: int,
        enable_compaction: bool = True,
        backend_name: str = "openai_agents",
    ) -> tuple[AgentTrajectory, str]:
        """Run the LLM agent.

        Args:
            sandbox_manager: The sandbox manager.
            task: The task to run.
            provider: Inference provider for LLM calls.
            max_turns: Maximum turns.
            enable_compaction: Enable context compaction for long conversations.
            backend_name: Name of the backend to use.

        Returns:
            Tuple of (trajectory, completion_reason).
        """
        from olmo_eval.harness.backends import get_backend, validate_backend
        from olmo_eval.harness.config import HarnessConfig
        from olmo_eval.harness.tools import get_tools

        validate_backend(backend_name)
        tools = get_tools(("execute_bash_session", "submit"))
        harness_config = HarnessConfig(
            name=f"terminal_bench_{task.task_id}",
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            max_turns=max_turns,
        )

        backend = get_backend(backend_name)
        backend.set_sandbox_manager(sandbox_manager)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": task.instruction},),
        )

        run_config = {
            "backend": backend.name,
            "task_id": task.task_id,
            "max_turns": max_turns,
            "enable_compaction": enable_compaction,
            "tools": [t.name for t in tools],
        }
        logger.info(f"Starting agent: {run_config}")
        harness_result = await backend.run(
            provider,
            harness_config,
            request,
            trace_metadata={"task_id": task.task_id},
            enable_compaction=enable_compaction,
        )

        completion_reason = "max_turns" if harness_result.max_turns_reached else "complete"
        logger.info(f"Agent completed task {task.task_id}: {completion_reason}")
        trajectory = harness_result.trajectory or AgentTrajectory(turns=())
        return trajectory, completion_reason

    def _build_predictions(self, task_results: list[TaskResult]) -> list[dict[str, Any]]:
        """Build predictions list from task results."""
        predictions = []
        for r in task_results:
            predictions.append(
                {
                    "native_id": r.task_id,
                    "instance_metrics": {
                        "reward": {"external": r.reward},
                        "agent_duration": {"external": r.agent_duration},
                    },
                    "completion_reason": r.completion_reason,
                    "verification_exit_code": r.verification_exit_code,
                    "difficulty": r.difficulty,
                    "category": r.category,
                    "error": r.error,
                    "trajectory": r.trajectory.to_dict() if r.trajectory else None,
                }
            )
        return predictions

    def _save_task_results(self, task_results: list[TaskResult], output_dir: str) -> None:
        """Save detailed task results and trajectories to files."""
        import hashlib
        import re

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / f"{self.name}_tasks.json"
        data = []
        for result in task_results:
            output = result.verification_output[:10000] if result.verification_output else ""
            data.append(
                {
                    "task_id": result.task_id,
                    "reward": result.reward,
                    "completion_reason": result.completion_reason,
                    "agent_duration": result.agent_duration,
                    "verification_exit_code": result.verification_exit_code,
                    "verification_output": output,
                    "difficulty": result.difficulty,
                    "category": result.category,
                    "error": result.error,
                }
            )

        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Task results saved to {results_file}")

        traces_dir = output_path / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        for result in task_results:
            if not result.trajectory:
                continue

            sanitized_id = re.sub(r"[^\w\-]", "_", result.task_id)
            sanitized_id = re.sub(r"_+", "_", sanitized_id).strip("_").lower()

            hash_input = f"{self.name}:{result.task_id}"
            short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:6]

            trace_file = traces_dir / f"{self.name}_{sanitized_id}_{short_hash}.jsonl"

            trajectory_data = {
                "task_id": result.task_id,
                "trajectory": result.trajectory.to_dict(),
            }
            with open(trace_file, "w") as f:
                f.write(json.dumps(trajectory_data) + "\n")

        logger.info(f"Trajectories saved to {traces_dir}/")
