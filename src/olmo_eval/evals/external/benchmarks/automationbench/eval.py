"""AutomationBench-AA external evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from olmo_eval.evals.external.args import as_list, as_optional_float, as_optional_int
from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/zapier/AutomationBench.git"
#: The commit of the AutomationBench 1.0.6 release, the version Artificial
#: Analysis runs. Its tasks, tools and assertions are code in this repository.
REPO_COMMIT = "6d210543b7a046f0f451c828cd2dadef774276eb"
BENCHMARK_VERSION = "1.0.6"
#: AutomationBench requires Python 3.13; its environment is separate from olmo-eval's.
PYTHON_VERSION = "3.13"

#: The six scored domains, 100 public tasks each. AutomationBench's ``simple``
#: domain is a tool-use warm-up that its own score leaves out.
DOMAINS = ("sales", "marketing", "operations", "support", "finance", "hr")

DRIVER = Path(__file__).parent / "driver" / "run_tasks.py"

#: Rollout outcomes recorded by the driver.
MODEL_ERROR = "model"
INFRA_ERROR = "infra"

#: Environment variables that would point uv at olmo-eval's own environment or
#: constrain the benchmark's locked install.
_UV_PASSTHROUGH = ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_LINK_MODE")
_STRIPPED_ENV = ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX")

#: The variable the driver's client reads its API key from.
API_KEY_VAR = "AUTOMATIONBENCH_API_KEY"


@dataclass
class AutomationBenchArgs:
    """Arguments for the AutomationBench-AA evaluation."""

    domains: list[str] = field(default_factory=lambda: list(DOMAINS))
    task_names: list[str] | None = None
    limit: int | None = None
    max_turns: int = 50
    max_concurrency: int = 32
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_failure_rate: float = 0.05
    setup_timeout: float = 1800.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutomationBenchArgs:
        defaults = cls()
        domains = as_list(data.get("domains")) or list(defaults.domains)
        unknown = sorted(set(domains) - set(DOMAINS))
        if unknown:
            raise ValueError(
                f"Unknown AutomationBench domains: {', '.join(unknown)} "
                f"(choose from {', '.join(DOMAINS)})"
            )
        args = cls(
            domains=domains,
            task_names=as_list(data.get("task_names")),
            limit=as_optional_int(data.get("limit")),
            max_turns=int(data.get("max_turns", defaults.max_turns)),
            max_concurrency=int(data.get("max_concurrency", defaults.max_concurrency)),
            temperature=as_optional_float(data.get("temperature")),
            top_p=as_optional_float(data.get("top_p")),
            max_tokens=as_optional_int(data.get("max_tokens")),
            max_failure_rate=float(data.get("max_failure_rate", defaults.max_failure_rate)),
            setup_timeout=float(data.get("setup_timeout", defaults.setup_timeout)),
        )
        if args.max_turns < 1 or args.max_concurrency < 1:
            raise ValueError("max_turns and max_concurrency must be at least 1")
        if args.limit is not None and args.limit < 1:
            raise ValueError("limit must be at least 1")
        if not 0.0 <= args.max_failure_rate <= 1.0:
            raise ValueError("max_failure_rate must be between 0 and 1")
        return args

    def sampling_args(self, chat_template_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Request parameters to send; an unset one is left to the model server.

        AutomationBench's runner sends none, so by default each reply is sampled
        with the server's defaults (vLLM applies the model's generation config) and
        may use the rest of the context window.
        """
        sampling: dict[str, Any] = {}
        if self.temperature is not None:
            sampling["temperature"] = self.temperature
        if self.top_p is not None:
            sampling["top_p"] = self.top_p
        if self.max_tokens is not None:
            sampling["max_tokens"] = self.max_tokens
        if chat_template_kwargs:
            sampling["extra_body"] = {"chat_template_kwargs": dict(chat_template_kwargs)}
        return sampling


@dataclass
class TaskResult:
    """One task's rollout and its scores, as the driver wrote them."""

    task_name: str
    domain: str
    stop_condition: str | None = None
    error_kind: str | None = None
    error: str | None = None
    scoring_error: str | None = None
    objectives: int = 0
    objectives_met: int = 0
    guardrails: int = 0
    guardrail_violations: int = 0
    aa_score: float = 0.0
    partial_credit: float = 0.0
    task_completed_correctly: float = 0.0
    turns: int = 0
    tool_calls: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> TaskResult:
        return cls(
            task_name=str(record["task_name"]),
            domain=str(record.get("domain") or str(record["task_name"]).split(".", 1)[0]),
            stop_condition=record.get("stop_condition"),
            error_kind=record.get("error_kind"),
            error=record.get("error"),
            scoring_error=record.get("scoring_error"),
            objectives=int(record.get("objectives", 0)),
            objectives_met=int(record.get("objectives_met", 0)),
            guardrails=int(record.get("guardrails", 0)),
            guardrail_violations=int(record.get("guardrail_violations", 0)),
            aa_score=float(record.get("aa_score", 0.0)),
            partial_credit=float(record.get("partial_credit", 0.0)),
            task_completed_correctly=float(record.get("task_completed_correctly", 0.0)),
            turns=int(record.get("turns", 0)),
            tool_calls=int(record.get("tool_calls", 0)),
            raw=record,
        )

    @property
    def failure(self) -> str | None:
        """Why this task has no score, or None when it is scored."""
        if self.scoring_error:
            return f"scoring_error: {self.scoring_error}"
        if self.error_kind == INFRA_ERROR:
            return f"rollout_error: {self.error}"
        return None

    @property
    def score(self) -> float:
        """The AutomationBench-AA score: a model error scores 0, as AA scores it."""
        return 0.0 if self.error_kind == MODEL_ERROR else self.aa_score

    @property
    def completed(self) -> bool:
        """Every objective met and no guardrail broken."""
        return (
            self.error_kind != MODEL_ERROR
            and self.objectives > 0
            and self.objectives_met == self.objectives
            and self.guardrail_violations == 0
        )


def compute_metrics(tasks: list[TaskResult]) -> dict[str, float]:
    """Metrics over the scored tasks; the first is the primary metric."""
    scored = [t for t in tasks if t.failure is None]
    n = len(scored)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    objectives = sum(t.objectives for t in scored)
    met = sum(t.objectives_met for t in scored)
    violations = sum(t.guardrail_violations for t in scored)
    metrics: dict[str, float] = {
        "aa_score": mean([t.score for t in scored]),
        "objectives_completed": met / objectives if objectives else 0.0,
        "tasks_completed": mean([float(t.completed) for t in scored]),
        "violations_per_task": violations / n if n else 0.0,
    }
    if violations:
        metrics["objectives_per_violation"] = met / violations
    metrics["zapier_pass_rate"] = mean([t.task_completed_correctly for t in scored])
    metrics["zapier_partial_credit"] = mean([t.partial_credit for t in scored])
    for domain in sorted({t.domain for t in scored}):
        metrics[f"aa_score_{domain}"] = mean([t.score for t in scored if t.domain == domain])

    stops = Counter(t.stop_condition for t in scored)
    metrics["max_turns_rate"] = stops["max_turns_reached"] / n if n else 0.0
    metrics["context_overflow_rate"] = stops["prompt_too_long"] / n if n else 0.0
    metrics["model_error_rate"] = mean([float(t.error_kind == MODEL_ERROR) for t in scored])
    metrics["mean_turns"] = mean([float(t.turns) for t in scored])
    metrics["mean_tool_calls"] = mean([float(t.tool_calls) for t in scored])
    metrics["num_tasks"] = float(len(tasks))
    metrics["num_scored_tasks"] = float(n)
    metrics["num_failed_tasks"] = float(len(tasks) - n)
    return metrics


def _subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """olmo-eval's environment without what would redirect the benchmark's uv or Python."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _STRIPPED_ENV and (not k.startswith("UV_") or k in _UV_PASSTHROUGH)
    }
    env.update(extra or {})
    return env


def default_cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "olmo-eval" / "automationbench"


def prepare_environment(cache_root: Path, timeout: float) -> Path:
    """Check out the pinned AutomationBench and sync its locked environment.

    Returns the environment's Python. A checkout that finished syncing before is
    reused; a partial one is replaced.
    """
    git, uv = shutil.which("git"), shutil.which("uv")
    if git is None or uv is None:
        raise RuntimeError("AutomationBench-AA needs git and uv on PATH to install the benchmark")
    repo = cache_root / REPO_COMMIT
    python = repo / ".venv" / "bin" / "python"
    ready = repo / ".olmo-eval-ready"
    if ready.exists() and python.exists():
        return python

    shutil.rmtree(repo, ignore_errors=True)
    repo.parent.mkdir(parents=True, exist_ok=True)
    env = _subprocess_env()

    def run(cmd: list[str], cwd: Path | None = None) -> str:
        logger.info(f"[automationbench] $ {' '.join(cmd)}")
        done = subprocess.run(
            cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False
        )
        if done.returncode != 0:
            output = (done.stderr or done.stdout)[-2000:]
            raise RuntimeError(f"{' '.join(cmd)} failed ({done.returncode}): {output}")
        return done.stdout

    run([git, "init", "--quiet", str(repo)])
    run([git, "-C", str(repo), "fetch", "--quiet", "--depth", "1", REPO_URL, REPO_COMMIT])
    run([git, "-C", str(repo), "checkout", "--quiet", "FETCH_HEAD"])
    head = run([git, "-C", str(repo), "rev-parse", "HEAD"]).strip()
    if head != REPO_COMMIT:
        raise RuntimeError(f"AutomationBench checkout is at {head}, expected {REPO_COMMIT}")
    run([uv, "sync", "--frozen", "--no-dev", "--python", PYTHON_VERSION], cwd=repo)
    ready.write_text(REPO_COMMIT + "\n")
    return python


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class AutomationBenchAAExternalEval(ExternalEval):
    """AutomationBench's public tasks, scored as Artificial Analysis scores AutomationBench-AA."""

    @property
    def name(self) -> str:
        return "automationbench_aa"

    @property
    def description(self) -> str:
        return (
            "Evaluates LLM agents on Zapier AutomationBench's 600 public business-workflow "
            "tasks, scored as AutomationBench-AA: objectives completed, 0 for a broken guardrail"
        )

    @property
    def timeout_seconds(self) -> float:
        return 24 * 3600.0

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        defaults = AutomationBenchArgs()
        return {
            "domains": ("Comma-separated domains (default: all six)", ",".join(DOMAINS)),
            "task_names": ("Comma-separated task names, e.g. sales.multi_hop_lookup", None),
            "limit": ("Run only the first N selected tasks", None),
            "max_turns": ("Max model replies per task", defaults.max_turns),
            "max_concurrency": ("Max tasks in flight at once", defaults.max_concurrency),
            "temperature": ("Sampling temperature (default: the server's)", None),
            "top_p": ("Nucleus sampling threshold (default: the server's)", None),
            "max_tokens": ("Max tokens per reply (default: the rest of the context)", None),
            "max_failure_rate": (
                "Largest share of unscored tasks for the run to succeed",
                defaults.max_failure_rate,
            ),
            "setup_timeout": (
                "Timeout in seconds for each benchmark install step",
                defaults.setup_timeout,
            ),
        }

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        """Run the selected tasks in AutomationBench's environment, then aggregate."""
        start_time = time.time()
        ab_args = AutomationBenchArgs.from_dict(args)

        try:
            python = await asyncio.to_thread(
                prepare_environment, default_cache_root(), ab_args.setup_timeout
            )
        except Exception as e:
            logger.exception("Could not install AutomationBench")
            return self._error_result(f"setup_error: {e}", start_time)

        base_url = getattr(provider, "base_url", None)
        if not base_url:
            return self._error_result(
                "AutomationBench-AA needs a provider with an OpenAI-compatible base_url",
                start_time,
            )
        api_key = (
            "EMPTY"
            if self._is_local_provider(provider, base_url)
            else os.environ.get("OPENAI_API_KEY", "EMPTY")
        )
        chat_template_kwargs = getattr(provider, "chat_template_kwargs", None)
        sampling = ab_args.sampling_args(chat_template_kwargs)

        with tempfile.TemporaryDirectory(prefix="automationbench-") as scratch:
            run_dir = Path(output_dir) / self.name if output_dir else Path(scratch)
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / "config.json"
            records_path = run_dir / "tasks.jsonl"
            config = {
                "model": provider.model_name,
                "base_url": base_url,
                "api_key_var": API_KEY_VAR,
                "domains": ab_args.domains,
                "task_names": ab_args.task_names,
                "limit": ab_args.limit,
                "max_turns": ab_args.max_turns,
                "max_concurrent": ab_args.max_concurrency,
                "sampling_args": sampling,
            }
            config_path.write_text(json.dumps(config, indent=2))

            exit_code, output_tail = await self._run_driver(
                python, config_path, records_path, run_dir, api_key
            )
            records = read_records(records_path)

        tasks = [TaskResult.from_record(r) for r in records]
        failed = [t for t in tasks if t.failure is not None]
        problems: list[str] = []
        if exit_code != 0:
            problems.append(f"the AutomationBench run exited with {exit_code}: {output_tail}")
        if not tasks:
            problems.append("no task finished")
        elif len(failed) > ab_args.max_failure_rate * len(tasks):
            details = "; ".join(f"{t.task_name}: {t.failure}" for t in failed[:5])
            problems.append(
                f"{len(failed)} of {len(tasks)} tasks failed, over the "
                f"{ab_args.max_failure_rate:.0%} budget: {details}"
            )
        if failed:
            logger.warning(f"[{self.name}] {len(failed)} of {len(tasks)} tasks are unscored")

        result = ExternalEvalResult(
            name=self.name,
            success=not problems,
            error="; ".join(problems) or None,
            metrics=compute_metrics(tasks),
            metadata={
                "model_name": provider.model_name,
                "num_tasks": len(tasks),
                "benchmark_repo": REPO_URL,
                "benchmark_commit": REPO_COMMIT,
                "benchmark_version": BENCHMARK_VERSION,
                "toolset": "api",
                "sampling_args": sampling,
                "failed_tasks": {t.task_name: t.failure for t in failed},
                "stop_conditions": dict(Counter(t.stop_condition or "none" for t in tasks)),
                **asdict(ab_args),
            },
            duration_seconds=time.time() - start_time,
            predictions=[self._prediction(t) for t in tasks],
        )
        if output_dir:
            self._save_results(result, output_dir)
        return result

    async def _run_driver(
        self, python: Path, config_path: Path, records_path: Path, run_dir: Path, api_key: str
    ) -> tuple[int, str]:
        """Run the driver in the benchmark's environment, streaming its output to the log."""
        proc = await asyncio.create_subprocess_exec(
            str(python),
            "-I",
            str(DRIVER),
            str(config_path),
            str(records_path),
            cwd=run_dir,
            env=_subprocess_env({API_KEY_VAR: api_key}),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        tail: list[str] = []
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            logger.info(f"[{self.name}] {line}")
            tail = (tail + [line])[-20:]
        exit_code = await proc.wait()
        return exit_code, "\n".join(tail)

    def _prediction(self, task: TaskResult) -> dict[str, Any]:
        instance_metrics: dict[str, Any] = {}
        if task.failure is None:
            instance_metrics = {
                "aa_score": {"external": task.score},
                "zapier_partial_credit": {"external": task.partial_credit},
                "zapier_pass": {"external": task.task_completed_correctly},
            }
        return {
            "native_id": task.task_name,
            "instance_metrics": instance_metrics,
            "failure": task.failure,
            **{k: v for k, v in task.raw.items() if k != "task_name"},
        }
