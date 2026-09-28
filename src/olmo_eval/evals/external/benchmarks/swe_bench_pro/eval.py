"""SWE-Bench Pro external evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from olmo_eval.common.types import LMRequest, RequestType
from olmo_eval.common.types.trajectory import AgentTrajectory
from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.network import get_docker_network_args
from olmo_eval.evals.external.result import ExternalEvalResult
from olmo_eval.harness.sandbox.config import ContainerRuntime, SandboxConfig, SandboxMode

from .loader import SWEBenchProLoader
from .task import SWEBenchProTask
from .verifier import (
    SWEBenchProVerifier,
    VerificationResult,
    capture_patch_command,
    parse_captured_patch,
    parse_snapshot,
    snapshot_command,
)

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox import SandboxManager
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

Version = Literal["v1", "v2"]

SYSTEM_PROMPT = """\
You are an autonomous software engineer working in a Linux container that holds a \
code repository.
You have access to the following tools:
- execute_bash_session(command): Execute a bash command in a persistent shell session
- submit(): Call when you have finished making changes

Read the task, explore the repository, and edit its source files to implement the \
requested change. Make minimal changes to non-test files. You can create scripts to \
reproduce the problem and check your fix, but remove them before you finish. The \
repository's current state is graded when you stop, so leave your final changes in \
place. When the change is complete, call submit().
"""

_TRUE_VALUES = (True, 1, "true", "True", "1")
_FALSE_VALUES = (False, 0, "false", "False", "0")


def _parse_bool(value: Any, name: str) -> bool:
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean for {name}: {value!r}")


def _parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


@dataclass
class SWEBenchProArgs:
    """Arguments for SWE-Bench Pro evaluation."""

    instance_ids: list[str] | None = None
    subset: str = "all"
    limit: int | None = None
    repo_path: str | None = None
    repo_ref: str = SWEBenchProLoader.DEFAULT_REF
    max_concurrency: int = 1
    max_turns: int = 250
    agent_timeout: float | None = None
    command_timeout: float = 900.0
    oracle: bool = False
    enable_compaction: bool = True
    scaffold: str = "openai_agents"
    remove_images: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SWEBenchProArgs:
        instance_ids = data.get("instance_ids")
        if isinstance(instance_ids, str):
            instance_ids = [t.strip() for t in instance_ids.split(",") if t.strip()]
        subset = data.get("subset", "all")
        if subset not in ("all", "hard"):
            raise ValueError(f"Invalid subset: {subset!r}. Must be 'all' or 'hard'.")
        agent_timeout = data.get("agent_timeout")
        return cls(
            instance_ids=instance_ids or None,
            subset=subset,
            limit=_parse_optional_int(data.get("limit")),
            repo_path=data.get("repo_path"),
            repo_ref=data.get("repo_ref", SWEBenchProLoader.DEFAULT_REF),
            max_concurrency=int(data.get("max_concurrency", 1)),
            max_turns=int(data.get("max_turns", 250)),
            agent_timeout=float(agent_timeout) if agent_timeout not in (None, "") else None,
            command_timeout=float(data.get("command_timeout", 900.0)),
            oracle=_parse_bool(data.get("oracle", False), "oracle"),
            enable_compaction=_parse_bool(data.get("enable_compaction", True), "enable_compaction"),
            scaffold=data.get("scaffold", "openai_agents"),
            remove_images=_parse_bool(data.get("remove_images", True), "remove_images"),
        )


@dataclass
class TaskResult:
    """Result of executing a single SWE-Bench Pro task."""

    instance_id: str
    repo: str
    hard: bool
    resolved: bool
    patch: str
    trajectory: AgentTrajectory
    completion_reason: str
    agent_duration: float
    patch_applied: bool = False
    verification_output: str = ""
    verification_exit_code: int = -1
    error: str | None = None
    runner_logs: dict[str, str] = field(default_factory=dict)

    @property
    def reward(self) -> float:
        return 1.0 if self.resolved else 0.0


class SWEBenchProExternalEval(ExternalEval):
    """SWE-Bench Pro evaluation with per-task container orchestration.

    Each task runs the agent in the task's image, captures the repository diff,
    and grades it in a fresh container of the same image.
    """

    def __init__(self, version: Version = "v2") -> None:
        self._version: Version = version

    @property
    def name(self) -> str:
        return "swe_bench_pro" if self._version == "v2" else f"swe_bench_pro_{self._version}"

    @property
    def version(self) -> Version:
        return self._version

    @property
    def description(self) -> str:
        if self._version == "v2":
            return "Evaluates LLM agents on 642 validated SWE-Bench Pro v2 tasks"
        return "Evaluates LLM agents on the original 731 SWE-Bench Pro tasks"

    @property
    def timeout_seconds(self) -> float:
        # Per-task budget: the agent phase plus the verifier
        return 6000.0

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "instance_ids": ("Comma-separated instance IDs to run (default: all)", None),
            "subset": ("Task subset: 'all' or 'hard'", "all"),
            "limit": ("Run only the first N tasks after filtering", None),
            "repo_path": ("Local harness repo path (default: fetch fresh)", None),
            "repo_ref": ("Harness repo commit to fetch", SWEBenchProLoader.DEFAULT_REF),
            "max_concurrency": ("Max tasks in flight", 1),
            "max_turns": ("Max agent turns per task", 250),
            "agent_timeout": ("Agent wall-clock budget per task in seconds", None),
            "command_timeout": ("Timeout for each agent shell command in seconds", 900.0),
            "oracle": ("Grade the reference patch instead of running the agent", False),
            "scaffold": ("Scaffold to use for agent execution", "openai_agents"),
            "remove_images": ("Remove each task's images after grading", True),
        }

    @property
    def scaffold(self) -> str | None:
        return "openai_agents"

    def load_tasks(self, repo_dir: Path, args: SWEBenchProArgs) -> list[SWEBenchProTask]:
        """Load and filter the tasks selected by ``args``."""
        loader = SWEBenchProLoader()
        if self._version == "v1":
            if args.subset != "all":
                raise ValueError("SWE-Bench Pro v1 has no task subsets")
            tasks = loader.load_v1_tasks(repo_dir, instance_ids=args.instance_ids)
        else:
            tasks = loader.load_v2_tasks(
                repo_dir, instance_ids=args.instance_ids, hard_only=args.subset == "hard"
            )
        if args.limit is not None:
            tasks = tasks[: args.limit]
        return tasks

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        """Execute SWE-Bench Pro evaluation.

        Args:
            provider: Inference provider for LLM calls.
            args: Evaluation-specific arguments.
            output_dir: Directory to write results.
            container_runtime: Container runtime (docker or podman).

        Returns:
            ExternalEvalResult with metrics and per-task results.
        """
        start_time = time.time()
        sbp_args = SWEBenchProArgs.from_dict(args)

        if not sbp_args.oracle:
            from olmo_eval.harness.scaffolds import validate_scaffold

            validate_scaffold(sbp_args.scaffold)

        if sbp_args.repo_path:
            repo_dir = Path(sbp_args.repo_path)
        else:
            repo_dir = Path("/tmp") / "swe-bench-pro-cache"
            await asyncio.to_thread(SWEBenchProLoader().ensure_repo, repo_dir, sbp_args.repo_ref)

        tasks = await asyncio.to_thread(self.load_tasks, repo_dir, sbp_args)
        if not tasks:
            return self._error_result(
                "No tasks found",
                start_time,
                f"repo_dir={repo_dir}, instance_ids={sbp_args.instance_ids}",
            )

        runtime = cast(ContainerRuntime, container_runtime)
        semaphore = asyncio.Semaphore(sbp_args.max_concurrency)

        async def run_task(task: SWEBenchProTask) -> TaskResult:
            async with semaphore:
                return await self._execute_task(task, provider, runtime, sbp_args)

        results = await asyncio.gather(*[run_task(t) for t in tasks], return_exceptions=True)

        task_results: list[TaskResult] = []
        for task, result in zip(tasks, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(f"Task {task.instance_id} failed with exception: {result}")
                task_results.append(self._failed_result(task, str(result), 0.0))
            else:
                task_results.append(result)

        metrics = self.compute_metrics(task_results)
        result = ExternalEvalResult(
            name=self.name,
            success=True,
            metrics=metrics,
            metadata={
                "model_name": provider.model_name,
                "version": self._version,
                "subset": sbp_args.subset,
                "oracle_mode": sbp_args.oracle,
                "max_turns": sbp_args.max_turns,
                "repo_ref": sbp_args.repo_ref,
                "num_tasks": len(task_results),
            },
            duration_seconds=time.time() - start_time,
            predictions=self._build_predictions(task_results),
        )

        if output_dir:
            self._save_results(result, output_dir)
            self._save_task_results(task_results, output_dir)

        return result

    @staticmethod
    def compute_metrics(task_results: list[TaskResult]) -> dict[str, float]:
        """Aggregate per-task results into resolve-rate metrics."""
        num_tasks = len(task_results)
        num_resolved = sum(1 for r in task_results if r.resolved)
        metrics: dict[str, float] = {
            "pass_rate": num_resolved / num_tasks if num_tasks else 0.0,
            "num_tasks": num_tasks,
            "num_resolved": num_resolved,
            "num_errors": sum(1 for r in task_results if r.error is not None),
            "num_empty_patches": sum(
                1 for r in task_results if r.error is None and not r.patch.strip()
            ),
        }

        hard = [r.reward for r in task_results if r.hard]
        if hard:
            metrics["pass_rate_hard"] = sum(hard) / len(hard)

        by_repo: dict[str, list[float]] = {}
        for r in task_results:
            by_repo.setdefault(r.repo, []).append(r.reward)
        for repo, rewards in sorted(by_repo.items()):
            safe_name = re.sub(r"[^a-z0-9]+", "_", repo.lower()).strip("_")
            metrics[f"pass_rate_{safe_name}"] = sum(rewards) / len(rewards)

        return metrics

    def _sandbox_config(
        self, image: str, task: SWEBenchProTask, runtime: ContainerRuntime, command_timeout: float
    ) -> SandboxConfig:
        return SandboxConfig(
            image=image,
            mode=SandboxMode.DOCKER,
            container_runtime=runtime,
            working_dir=task.working_dir,
            startup_timeout=300.0,
            command_timeout=command_timeout,
            docker_args=tuple(get_docker_network_args(runtime)),
            image_pull="never",
        )

    async def _execute_task(
        self,
        task: SWEBenchProTask,
        provider: InferenceProvider,
        runtime: ContainerRuntime,
        args: SWEBenchProArgs,
    ) -> TaskResult:
        """Run the agent on one task and grade its patch in a fresh sandbox."""
        from olmo_eval.harness.sandbox import SandboxManager

        logger.info(f"Executing task: {task.instance_id}")
        task_start = time.time()
        agent_timeout = args.agent_timeout or task.agent_timeout
        owner = f"sbp-{hashlib.sha256(task.instance_id.encode()).hexdigest()[:10]}"
        image: str | None = None

        try:
            image = await asyncio.to_thread(_prepare_image, task, runtime)

            trajectory = AgentTrajectory(turns=())
            if args.oracle:
                patch, completion_reason = task.gold_patch, "oracle"
            else:
                agent_config = self._sandbox_config(
                    image, task, runtime, min(args.command_timeout, agent_timeout)
                )
                agent_sandbox = SandboxManager([agent_config], owner=f"{owner}-agent")
                try:
                    await agent_sandbox.start()
                    await _pin_localhost_to_ipv4(agent_sandbox)
                    baseline = await self._snapshot(agent_sandbox, task)
                    trajectory, completion_reason = await self._run_agent(
                        agent_sandbox, task, provider, args, agent_timeout
                    )
                    patch = await self._capture_patch(agent_sandbox, task, baseline)
                finally:
                    await agent_sandbox.stop()
            agent_duration = time.time() - task_start

            grade_config = self._sandbox_config(image, task, runtime, task.verifier_timeout)
            verification = await self._grade(grade_config, task, patch, owner)
            return TaskResult(
                instance_id=task.instance_id,
                repo=task.repo,
                hard=task.hard,
                resolved=verification.resolved,
                patch=patch,
                trajectory=trajectory,
                completion_reason=completion_reason,
                agent_duration=agent_duration,
                patch_applied=verification.patch_applied,
                verification_output=verification.test_output,
                verification_exit_code=verification.test_exit_code,
                runner_logs=verification.runner_logs,
            )
        except Exception as e:
            logger.exception(f"Task {task.instance_id} failed")
            return self._failed_result(task, str(e), time.time() - task_start)
        finally:
            if args.remove_images:
                await asyncio.to_thread(_remove_images, runtime, [image, task.image])

    async def _grade(
        self, config: SandboxConfig, task: SWEBenchProTask, patch: str, owner: str
    ) -> VerificationResult:
        """Grade ``patch`` in a fresh sandbox, retrying once after an infrastructure failure."""
        from olmo_eval.harness.sandbox import SandboxManager

        verification: VerificationResult | None = None
        for attempt in range(1 + GRADE_RETRIES):
            grade_sandbox = SandboxManager([config], owner=f"{owner}-grade-{attempt}")
            try:
                await grade_sandbox.start()
                await _pin_localhost_to_ipv4(grade_sandbox)
                verification = await SWEBenchProVerifier().verify(
                    grade_sandbox.get_executor(frozenset()),
                    patch,
                    task.test_files,
                    task.working_dir,
                    task.verifier_timeout,
                    log_prefix=f"{task.instance_id[:60]}_verifier",
                )
            finally:
                await grade_sandbox.stop()
            if not _is_infrastructure_failure(verification) or attempt == GRADE_RETRIES:
                break
            logger.warning(f"Grading {task.instance_id} hit an infrastructure failure; retrying")
        assert verification is not None
        return verification

    async def _run_agent(
        self,
        sandbox_manager: SandboxManager,
        task: SWEBenchProTask,
        provider: InferenceProvider,
        args: SWEBenchProArgs,
        agent_timeout: float,
    ) -> tuple[AgentTrajectory, str]:
        """Run the LLM agent until it stops, runs out of turns, or times out."""
        from olmo_eval.harness.config import HarnessConfig
        from olmo_eval.harness.scaffolds import get_scaffold
        from olmo_eval.harness.tools import get_tools

        tools = get_tools(("execute_bash_session", "submit"))
        harness_config = HarnessConfig(
            name=f"{self.name}_{task.instance_id}",
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            max_turns=args.max_turns,
            scaffold=args.scaffold,
        )

        scaffold = get_scaffold(args.scaffold)
        scaffold.set_sandbox_manager(sandbox_manager)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": task.instruction},),
        )

        try:
            harness_result = await asyncio.wait_for(
                scaffold.run(
                    provider,
                    harness_config,
                    request,
                    trace_metadata={"task_id": task.instance_id},
                    enable_compaction=args.enable_compaction,
                ),
                timeout=agent_timeout,
            )
        except TimeoutError:
            logger.info(f"Agent timed out on {task.instance_id} after {agent_timeout}s")
            return AgentTrajectory(turns=()), "timeout"

        completion_reason = "max_turns" if harness_result.max_turns_reached else "complete"
        logger.info(f"Agent finished {task.instance_id}: {completion_reason}")
        return harness_result.trajectory or AgentTrajectory(turns=()), completion_reason

    async def _snapshot(self, sandbox_manager: SandboxManager, task: SWEBenchProTask) -> str:
        """Record the pristine repository, including files the image leaves untracked."""
        executor = sandbox_manager.get_executor(frozenset())
        result = await executor.execute_command(snapshot_command(task.working_dir), timeout=300.0)
        tree = parse_snapshot(result.output)
        if tree is None:
            raise RuntimeError(f"Failed to snapshot repository: {result.output[-2000:]}")
        return tree

    async def _capture_patch(
        self, sandbox_manager: SandboxManager, task: SWEBenchProTask, baseline: str
    ) -> str:
        """Return the agent's changes to the repository, relative to ``baseline``, as a diff."""
        executor = sandbox_manager.get_executor(frozenset())
        # A command the agent left running can keep the swe-rex server busy,
        # so capture through the container runtime instead.
        result = await executor.execute_control(
            capture_patch_command(task.working_dir, baseline), timeout=300.0
        )
        patch = parse_captured_patch(result.output)
        if patch is None:
            raise RuntimeError(f"Failed to capture patch: {result.output[-2000:]}")
        logger.info(f"Captured {len(patch)} byte patch for {task.instance_id}")
        return patch

    @staticmethod
    def _failed_result(task: SWEBenchProTask, error: str, duration: float) -> TaskResult:
        return TaskResult(
            instance_id=task.instance_id,
            repo=task.repo,
            hard=task.hard,
            resolved=False,
            patch="",
            trajectory=AgentTrajectory(turns=()),
            completion_reason="error",
            agent_duration=duration,
            error=error,
        )

    def _build_predictions(self, task_results: list[TaskResult]) -> list[dict[str, Any]]:
        """Build predictions list from task results."""
        return [
            {
                "native_id": r.instance_id,
                "instance_metrics": {
                    "reward": {"external": r.reward},
                    "agent_duration": {"external": r.agent_duration},
                },
                "repo": r.repo,
                "hard": r.hard,
                "completion_reason": r.completion_reason,
                "patch_applied": r.patch_applied,
                "verification_exit_code": r.verification_exit_code,
                "patch": r.patch,
                "error": r.error,
                "trajectory": r.trajectory.to_dict() if r.trajectory else None,
            }
            for r in task_results
        ]

    def _save_task_results(self, task_results: list[TaskResult], output_dir: str) -> None:
        """Save per-task results, patches, and trajectories."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "instance_id": r.instance_id,
                "repo": r.repo,
                "hard": r.hard,
                "resolved": r.resolved,
                "completion_reason": r.completion_reason,
                "agent_duration": r.agent_duration,
                "patch_applied": r.patch_applied,
                "verification_exit_code": r.verification_exit_code,
                "verification_output": r.verification_output[-10000:],
                "error": r.error,
            }
            for r in task_results
        ]
        results_file = output_path / f"{self.name}_tasks.json"
        results_file.write_text(json.dumps(data, indent=2))
        logger.info(f"Task results saved to {results_file}")

        patches_file = output_path / f"{self.name}_patches.jsonl"
        with open(patches_file, "w") as f:
            for r in task_results:
                f.write(json.dumps({"instance_id": r.instance_id, "model_patch": r.patch}) + "\n")

        for r in task_results:
            if not r.runner_logs:
                continue
            logs_dir = output_path / "verifier_logs" / r.instance_id
            logs_dir.mkdir(parents=True, exist_ok=True)
            for name, content in r.runner_logs.items():
                (logs_dir / name).write_text(content)

        traces_dir = output_path / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        for r in task_results:
            if not r.trajectory or not r.trajectory.turns:
                continue
            short_hash = hashlib.sha256(f"{self.name}:{r.instance_id}".encode()).hexdigest()[:6]
            trace_file = traces_dir / f"{self.name}_{r.instance_id[:80]}_{short_hash}.jsonl"
            trace_file.write_text(
                json.dumps({"task_id": r.instance_id, "trajectory": r.trajectory.to_dict()}) + "\n"
            )


# Image pulls, builds, and removals share the container runtime's storage.
# Running them one at a time avoids removing layers that a concurrent build is
# using, and keeps storage-lock contention from stalling running sandboxes.
_IMAGE_LOCK = threading.Lock()

GRADE_RETRIES = 1
_INFRASTRUCTURE_ERRORS = ("Sandbox unresponsive",)


# When the container has IPv6, localhost resolves to ::1 first, but some test
# suites only listen on 127.0.0.1 and then fail to connect. Docker containers
# have IPv6 disabled by default and resolve localhost to 127.0.0.1; keeping
# localhost off the IPv6 loopback entry gives the same result under Podman.
_PIN_LOCALHOST_COMMAND = (
    'hosts=$(sed -E "/^[[:space:]]*::1[[:space:]]/ s/[[:space:]]localhost([[:space:]]|$)/\\1/g" '
    '/etc/hosts) && printf "%s\\n" "$hosts" > /etc/hosts'
)


async def _pin_localhost_to_ipv4(sandbox_manager: SandboxManager) -> None:
    """Make ``localhost`` resolve only to 127.0.0.1 inside the sandbox."""
    executor = sandbox_manager.get_executor(frozenset())
    result = await executor.execute_command(_PIN_LOCALHOST_COMMAND, timeout=30.0)
    if not result.success:
        logger.warning(f"Could not update /etc/hosts: {result.output[-500:]}")


def _is_infrastructure_failure(verification: VerificationResult) -> bool:
    """Whether a grading run failed because of the sandbox rather than the patch."""
    return (
        not verification.resolved
        and verification.test_exit_code == -1
        and any(marker in verification.test_output for marker in _INFRASTRUCTURE_ERRORS)
    )


def _prepare_image(task: SWEBenchProTask, runtime: str) -> str:
    """Build the task's sandbox image with swe-rex installed alongside its toolchain."""
    from olmo_eval.harness.sandbox.image import get_swerex_image

    with _IMAGE_LOCK:
        return get_swerex_image(
            task.image, runtime, task.dockerfile_extra, isolated=True, use_registry=False
        )


def _remove_images(runtime: str, images: list[str | None]) -> None:
    """Best-effort removal of task images to bound disk usage."""
    names = [image for image in images if image]
    if not names:
        return
    with _IMAGE_LOCK:
        result = subprocess.run([runtime, "rmi", *names], capture_output=True, text=True)
    if result.returncode != 0:
        logger.debug(f"Image removal failed for {names}: {result.stderr.strip()}")
