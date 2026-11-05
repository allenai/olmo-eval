"""BixBench-Verified-50 external evaluation."""

from __future__ import annotations

import asyncio
import logging
import shutil
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from olmo_eval.common.types import SamplingParams
from olmo_eval.common.types.trajectory import AgentTrajectory
from olmo_eval.evals.external.args import as_bool, as_list, as_optional_float
from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.result import ExternalEvalResult
from olmo_eval.harness.sandbox.config import ContainerRuntime, SandboxConfig, SandboxMode
from olmo_eval.inference.base import require_tool_support

from . import notebook as nb
from .agent import (
    CONTAINER_WORKDIR,
    DEFAULT_EXEC_TIMEOUT,
    DEFAULT_MAX_STEPS,
    NotebookEpisode,
)
from .grader import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TEMPERATURE,
    Grade,
    build_judge_fn,
    grade_answer,
)
from .loader import (
    DATASET_REVISION,
    EVAL_MODES,
    BixBenchQuestion,
    download_questions,
    load_questions,
    prepare_capsule,
    select_questions,
    stage_workdir,
)

if TYPE_CHECKING:
    from olmo_eval.common.scorers.llm_judge import ConfiguredJudgeFn
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

#: The BixBench-env image: upstream's arm64-only ``futurehouse/bixbench:aviary-notebook-env``
#: rebuilt for linux/amd64 from ``sandbox/Dockerfile``.
SANDBOX_IMAGE = (
    "ghcr.io/allenai/olmo-eval-bixbench-env:2026.11.03"
    "@sha256:624403103a4cfbcb9a33091cf56d8f164901f91e5fabe9a5c5fe3b4864421227"
)
SANDBOX_STARTUP_TIMEOUT = 600.0
SANDBOX_START_ATTEMPTS = 3
SANDBOX_START_BACKOFF = 10.0
#: Extra time past the step and notebook budgets before an episode is cut off from
#: outside, bounding a model call or command that never returns.
EPISODE_GRACE = 1800.0

COMPLETION_REASONS = ("submitted", "max_steps", "exec_timeout", "context_overflow")


@dataclass
class BixBenchArgs:
    """Arguments of a BixBench-Verified-50 run."""

    question_ids: list[str] | None = None
    revision: str = DATASET_REVISION
    n_attempts: int = 3
    max_concurrency: int = 8
    max_steps: int = DEFAULT_MAX_STEPS
    exec_timeout: float = DEFAULT_EXEC_TIMEOUT
    temperature: float = 1.0
    top_p: float | None = None
    max_tokens: int = 16384
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_temperature: float = DEFAULT_JUDGE_TEMPERATURE
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
    judge_concurrency: int = 8
    image: str = SANDBOX_IMAGE
    image_timeout: float = 3600.0
    system_first: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BixBenchArgs:
        defaults = cls()
        return cls(
            question_ids=as_list(data.get("question_ids")),
            revision=str(data.get("revision", defaults.revision)),
            n_attempts=int(data.get("n_attempts", defaults.n_attempts)),
            max_concurrency=int(data.get("max_concurrency", defaults.max_concurrency)),
            max_steps=int(data.get("max_steps", defaults.max_steps)),
            exec_timeout=float(data.get("exec_timeout", defaults.exec_timeout)),
            temperature=float(data.get("temperature", defaults.temperature)),
            top_p=as_optional_float(data.get("top_p")),
            max_tokens=int(data.get("max_tokens", defaults.max_tokens)),
            judge_model=str(data.get("judge_model", defaults.judge_model)),
            judge_temperature=float(data.get("judge_temperature", defaults.judge_temperature)),
            judge_max_tokens=int(data.get("judge_max_tokens", defaults.judge_max_tokens)),
            judge_concurrency=int(data.get("judge_concurrency", defaults.judge_concurrency)),
            image=str(data.get("image", defaults.image)),
            image_timeout=float(data.get("image_timeout", defaults.image_timeout)),
            system_first=as_bool(data.get("system_first", defaults.system_first)),
        )

    @property
    def sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            fit_max_tokens_to_context=True,
        )


@dataclass
class TrialResult:
    """One agent episode on one question, and the grading of its answer."""

    question: BixBenchQuestion
    attempt: int
    completion_reason: str = "error"
    answer: Any = None
    grade: Grade | None = None
    steps: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    trajectory: AgentTrajectory = field(default_factory=lambda: AgentTrajectory(turns=()))
    agent_duration: float = 0.0
    #: A failure of the harness, sandbox, model server or judge. A trial with an
    #: error is left out of the metrics.
    error: str | None = None

    @property
    def native_id(self) -> str:
        return f"{self.question.question_id}:{self.attempt}"

    @property
    def correct(self) -> bool | None:
        return None if self.grade is None else self.grade.correct


def _accuracy(trials: list[TrialResult]) -> float:
    return sum(bool(t.correct) for t in trials) / len(trials) if trials else 0.0


def compute_metrics(trials: list[TrialResult]) -> dict[str, float]:
    """Accuracy over the trials that ran and were graded, and how the episodes ended.

    An episode that never submitted, ran out of steps or context, or timed out in
    the notebook counts as incorrect. ``accuracy_std`` is the standard deviation of
    the per-attempt accuracies.
    """
    scored = [t for t in trials if t.error is None]
    metrics: dict[str, float] = {"accuracy": _accuracy(scored)}
    for mode in EVAL_MODES:
        of_mode = [t for t in scored if t.question.eval_mode == mode]
        if of_mode:
            metrics[f"accuracy_{mode}"] = _accuracy(of_mode)

    attempts = sorted({t.attempt for t in scored})
    per_attempt = [_accuracy([t for t in scored if t.attempt == a]) for a in attempts]
    for attempt, value in zip(attempts, per_attempt, strict=True):
        metrics[f"accuracy_attempt_{attempt}"] = value
    if len(per_attempt) > 1:
        metrics["accuracy_std"] = statistics.stdev(per_attempt)
        metrics["accuracy_stderr"] = metrics["accuracy_std"] / len(per_attempt) ** 0.5

    n = len(scored)
    for reason in COMPLETION_REASONS:
        count = sum(t.completion_reason == reason for t in scored)
        metrics[f"{reason}_rate"] = count / n if n else 0.0
    metrics["mean_steps"] = statistics.fmean(t.steps for t in scored) if scored else 0.0
    metrics["num_trials"] = float(len(trials))
    metrics["num_scored_trials"] = float(n)
    metrics["num_failed_trials"] = float(len(trials) - n)
    return metrics


class BixBenchVerifiedExternalEval(ExternalEval):
    """BixBench-Verified-50: notebook agent episodes, graded by string match or a judge."""

    @property
    def name(self) -> str:
        return "bixbench_verified"

    @property
    def description(self) -> str:
        return (
            "Evaluates LLM agents on BixBench-Verified-50: open-answer computational biology "
            "questions answered by analysing capsule data in a Jupyter notebook"
        )

    @property
    def timeout_seconds(self) -> float:
        return 24 * 3600.0

    @property
    def required_secrets(self) -> tuple[str, ...]:
        # The dataset is gated; Beaker launches inject the user's HF token on their own.
        return ("OPENAI_API_KEY",)

    @property
    def extras(self) -> tuple[str, ...]:
        # The judge calls the OpenAI API.
        return ("clients",)

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        defaults = BixBenchArgs()
        return {
            "question_ids": ("Comma-separated question IDs, e.g. bix-12-q2 (default: all)", None),
            "revision": ("Dataset revision", defaults.revision),
            "n_attempts": ("Episodes per question", defaults.n_attempts),
            "max_concurrency": ("Max episodes in flight at once", defaults.max_concurrency),
            "max_steps": ("Max model calls per episode", defaults.max_steps),
            "exec_timeout": (
                "Timeout in seconds for one run of the notebook",
                defaults.exec_timeout,
            ),
            "temperature": ("Sampling temperature", defaults.temperature),
            "top_p": ("Nucleus sampling threshold", None),
            "max_tokens": ("Max tokens per model reply", defaults.max_tokens),
            "judge_model": ("Grading judge model", defaults.judge_model),
            "judge_temperature": ("Grading judge temperature", defaults.judge_temperature),
            "judge_max_tokens": ("Max tokens per judge reply", defaults.judge_max_tokens),
            "judge_concurrency": ("Max judge calls in flight", defaults.judge_concurrency),
            "image": ("BixBench-env sandbox image", defaults.image),
            "image_timeout": (
                "Timeout in seconds for preparing the sandbox image",
                defaults.image_timeout,
            ),
            "system_first": (
                "Send the system prompt first instead of after the problem and the "
                "first notebook render, for chat templates that require it",
                defaults.system_first,
            ),
        }

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        """Run the agent on each selected question, then grade what it submitted."""
        require_tool_support(provider)
        start_time = time.time()
        bb_args = BixBenchArgs.from_dict(args)
        runtime = cast(ContainerRuntime, container_runtime)
        if not bb_args.image:
            return self._error_result("no sandbox image: pass -A image=<image>", start_time)

        questions_path = await asyncio.to_thread(download_questions, bb_args.revision)
        questions = select_questions(load_questions(questions_path), bb_args.question_ids)
        # Fail before any agent runs if the judge cannot be reached.
        self._build_env_vars()

        capsules: dict[str, Path] = {}
        for folder in sorted({q.data_folder for q in questions}):
            capsules[folder] = await asyncio.to_thread(prepare_capsule, folder, bb_args.revision)

        try:
            image = await asyncio.to_thread(
                self._sandbox_image, bb_args.image, runtime, bb_args.image_timeout
            )
        except Exception as e:
            logger.exception("Could not prepare the BixBench sandbox image")
            return self._error_result(f"image_error: {e}", start_time)

        judge_fn = build_judge_fn(
            bb_args.judge_model, bb_args.judge_temperature, bb_args.judge_max_tokens
        )
        judge_limiter = asyncio.Semaphore(max(1, bb_args.judge_concurrency))
        trial_limiter = asyncio.Semaphore(max(1, bb_args.max_concurrency))

        with tempfile.TemporaryDirectory(prefix="bixbench-") as scratch:
            artifacts = Path(output_dir) / self.name if output_dir else Path(scratch) / "out"

            async def run(question: BixBenchQuestion, attempt: int) -> TrialResult:
                async with trial_limiter:
                    return await self._run_trial(
                        question,
                        attempt,
                        capsules[question.data_folder],
                        provider,
                        image,
                        runtime,
                        bb_args,
                        judge_fn,
                        judge_limiter,
                        Path(scratch),
                        artifacts / f"{question.question_id}-{attempt}",
                    )

            trials = await asyncio.gather(
                *[run(q, a) for q in questions for a in range(bb_args.n_attempts)]
            )

        failed = [t for t in trials if t.error is not None]
        error = None
        if failed:
            details = "; ".join(f"{t.native_id}: {t.error}" for t in failed)
            error = f"{len(failed)} of {len(trials)} trials failed: {details}"
        result = ExternalEvalResult(
            name=self.name,
            # A failed trial is not a score, so a run with one does not stand as a result.
            success=not failed,
            error=error,
            metrics=compute_metrics(trials),
            metadata={
                "model_name": provider.model_name,
                "num_questions": len(questions),
                "num_trials": len(trials),
                "sandbox_image": image,
                "judge_config": dict(judge_fn.judge_config),
                **asdict(bb_args),
            },
            duration_seconds=time.time() - start_time,
            predictions=[self._prediction(t) for t in trials],
        )
        if output_dir:
            self._save_results(result, output_dir)
        return result

    def _sandbox_image(self, base: str, runtime: str, timeout: float) -> str:
        from olmo_eval.harness.sandbox.image import get_swerex_image

        return get_swerex_image(base, runtime, pristine=True, timeout=timeout)

    def _sandbox_config(
        self, image: str, runtime: ContainerRuntime, workdir: Path, command_timeout: float
    ) -> SandboxConfig:
        return SandboxConfig(
            image=image,
            mode=SandboxMode.DOCKER,
            container_runtime=runtime,
            working_dir=CONTAINER_WORKDIR,
            startup_timeout=SANDBOX_STARTUP_TIMEOUT,
            command_timeout=command_timeout,
            docker_args=("--mount", f"type=bind,src={workdir},dst={CONTAINER_WORKDIR}"),
            pristine_image=True,
        )

    async def _run_trial(
        self,
        question: BixBenchQuestion,
        attempt: int,
        capsule_dir: Path,
        provider: InferenceProvider,
        image: str,
        runtime: ContainerRuntime,
        bb_args: BixBenchArgs,
        judge_fn: ConfiguredJudgeFn,
        judge_limiter: asyncio.Semaphore,
        scratch: Path,
        artifacts: Path,
    ) -> TrialResult:
        """Run one episode in a fresh copy of the capsule, keep its notebook, and grade it."""
        trial = TrialResult(question=question, attempt=attempt)
        workdir = scratch / f"{question.question_id}-{attempt}"
        try:
            await asyncio.to_thread(stage_workdir, capsule_dir, workdir)
            await self._run_episode(question, provider, image, runtime, bb_args, workdir, trial)
        except Exception as e:
            logger.exception(f"[{trial.native_id}] Episode failed")
            trial.error = trial.error or f"sandbox_error: {e}"
        finally:
            self._keep_notebook(workdir, artifacts)
            shutil.rmtree(workdir, ignore_errors=True)
        if trial.error is not None:
            return trial

        trial.grade = await grade_answer(question, trial.answer, judge_fn, judge_limiter)
        if trial.grade.correct is None:
            trial.error = trial.grade.error or "ungraded"
        logger.info(
            f"[{trial.native_id}] {trial.completion_reason} after {trial.steps} steps, "
            f"correct={trial.correct}, error={trial.error}"
        )
        return trial

    async def _run_episode(
        self,
        question: BixBenchQuestion,
        provider: InferenceProvider,
        image: str,
        runtime: ContainerRuntime,
        bb_args: BixBenchArgs,
        workdir: Path,
        trial: TrialResult,
    ) -> None:
        from olmo_eval.harness.sandbox import Capability, SandboxManager

        config = self._sandbox_config(image, runtime, workdir, bb_args.exec_timeout)
        # Containers sometimes fail to come up when many start at once on a cold image.
        # Nothing has reached the model yet, so starting again leaves the episode as is.
        for attempt in range(1, SANDBOX_START_ATTEMPTS + 1):
            manager = SandboxManager([config], owner=f"bixbench-{trial.native_id}")
            try:
                await manager.start()
                break
            except RuntimeError as e:
                await manager.stop()
                if attempt == SANDBOX_START_ATTEMPTS:
                    raise
                logger.warning(f"[{trial.native_id}] Sandbox start {attempt} failed: {e}")
                await asyncio.sleep(SANDBOX_START_BACKOFF * attempt)
        try:
            binding = await manager.acquire_binding(Capability.BASH)
            try:
                episode = NotebookEpisode(
                    question.question,
                    workdir,
                    binding.executor,
                    max_steps=bb_args.max_steps,
                    exec_timeout=bb_args.exec_timeout,
                    label=trial.native_id,
                    system_first=bb_args.system_first,
                )
                budget = bb_args.max_steps * bb_args.exec_timeout + EPISODE_GRACE
                try:
                    result = await asyncio.wait_for(
                        episode.run(provider, bb_args.sampling_params), timeout=budget
                    )
                except TimeoutError:
                    # Keep the steps taken so the prediction shows where it stalled.
                    result = episode.snapshot()
                    result.completion_reason = "error"
                    result.duration_s = budget
                    result.error = "agent_stuck: the episode outlasted its time budget"
            finally:
                await binding.release()
        finally:
            await manager.stop()
        trial.completion_reason = result.completion_reason
        trial.answer = result.answer
        trial.steps = result.steps
        trial.counts = dict(result.counts)
        trial.usage = dict(result.usage)
        trial.trajectory = result.trajectory
        trial.agent_duration = result.duration_s
        if result.error is not None:
            trial.error = result.error

    @staticmethod
    def _keep_notebook(workdir: Path, artifacts: Path) -> None:
        """Save the episode's notebook and its markdown render beside the results."""
        source = workdir / nb.NOTEBOOK_NAME
        if not source.is_file():
            return
        # The notebook is a diagnostic: failing to keep it must not cost the run
        # its other trials, which an exception escaping the trial would.
        try:
            artifacts.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, artifacts / nb.NOTEBOOK_NAME)
            markdown, _ = nb.view_notebook(nb.read_notebook(source))
            (artifacts / "notebook.md").write_text(markdown)
        except (OSError, ValueError, TypeError, KeyError) as e:
            logger.warning(f"Could not keep the notebook of {workdir.name}: {e!r}")

    def _prediction(self, trial: TrialResult) -> dict[str, Any]:
        instance_metrics: dict[str, Any] = {}
        if trial.error is None and trial.correct is not None:
            instance_metrics["accuracy"] = {"external": float(trial.correct)}
        return {
            "native_id": trial.native_id,
            "question_id": trial.question.question_id,
            "attempt": trial.attempt,
            "eval_mode": trial.question.eval_mode,
            "question": trial.question.question,
            "ideal": trial.question.ideal,
            "instance_metrics": instance_metrics,
            "completion_reason": trial.completion_reason,
            "answer": trial.answer,
            "grade": trial.grade.to_dict() if trial.grade is not None else None,
            "steps": trial.steps,
            "counts": trial.counts,
            "usage": trial.usage,
            "agent_duration": trial.agent_duration,
            "error": trial.error,
            "trajectory": trial.trajectory.to_dict(),
        }
