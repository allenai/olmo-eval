"""Tau2-bench external evaluation.

tau2_bench is a benchmark for evaluating language model agents on realistic
customer service tasks. It measures both task completion and constraint
satisfaction.

Repository: https://github.com/sierra-research/tau2-bench
"""

from __future__ import annotations

import base64
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.registry import register_external_eval
from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox.executor import SandboxExecutor

logger = logging.getLogger(__name__)

Tau2Domain = Literal["airline", "retail", "telecom"]

# Default max tokens if we can't query the server
DEFAULT_MAX_TOKENS = 32768


def _parse_optional(data: dict, key: str, type_fn: type) -> Any:
    """Parse an optional value from a dict with type conversion."""
    value = data.get(key)
    return type_fn(value) if value is not None else None


@dataclass
class Tau2Args:
    """Arguments for tau2_bench evaluation."""

    # Core settings
    domain: Tau2Domain = "airline"
    num_trials: int = 1
    max_steps: int = 30
    max_concurrency: int = 3

    # Agent LLM settings
    max_tokens: int | None = None
    temperature: float | None = None

    # User LLM settings
    user_llm: str = "gpt-4o-mini"
    user_temperature: float | None = None

    # Task filtering
    task_split_name: str | None = None
    task_ids: list[str] | None = None
    num_tasks: int | None = None

    # Execution settings
    max_errors: int | None = None
    seed: int | None = None
    log_level: str | None = None
    enforce_communication_protocol: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tau2Args:
        # Handle task_ids which can be comma-separated string or list
        task_ids = data.get("task_ids")
        if isinstance(task_ids, str):
            task_ids = [t.strip() for t in task_ids.split(",") if t.strip()]

        return cls(
            domain=data.get("domain", "airline"),
            num_trials=int(data.get("num_trials", 1)),
            max_steps=int(data.get("max_steps", 30)),
            max_concurrency=int(data.get("max_concurrency", 3)),
            max_tokens=_parse_optional(data, "max_tokens", int),
            temperature=_parse_optional(data, "temperature", float),
            user_llm=data.get("user_llm", "gpt-4o-mini"),
            user_temperature=_parse_optional(data, "user_temperature", float),
            task_split_name=data.get("task_split_name"),
            task_ids=task_ids,
            num_tasks=_parse_optional(data, "num_tasks", int),
            max_errors=_parse_optional(data, "max_errors", int),
            seed=_parse_optional(data, "seed", int),
            log_level=data.get("log_level"),
            enforce_communication_protocol=bool(data.get("enforce_communication_protocol", False)),
        )


class Tau2ExternalEval(ExternalEval):
    """Tau2-bench evaluation for customer service agent tasks."""

    @property
    def name(self) -> str:
        return "tau2_bench"

    @property
    def description(self) -> str:
        return (
            "Evaluates LLM agents on customer service tasks across airline and retail domains. "
            "Measures task completion rate and constraint satisfaction."
        )

    @property
    def sandbox_image(self) -> str:
        return "python:3.11"

    @property
    def working_dir(self) -> str:
        return "/workspace"

    @property
    def timeout_seconds(self) -> float:
        return 7200.0

    @property
    def setup_command(self) -> tuple[str, ...]:
        repo = f"{self.working_dir}/tau2-bench"
        return (
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            f"git clone --depth 1 https://github.com/sierra-research/tau2-bench.git {repo}",
            f"cd {repo} && ~/.local/bin/uv sync",
            f"mkdir -p {self.results_dir}",
        )

    @property
    def run_command(self) -> str:
        return self._build_run_command(
            model_name="{model}",
            provider_url="{provider_url}",
            provider_kind="vllm_server",
            tau2_args=Tau2Args(),
        )

    @property
    def required_secrets(self) -> tuple[str, ...]:
        return ("OPENAI_API_KEY",)

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "domain": ("Task domain: 'airline', 'retail', or 'telecom'", "airline"),
            "num_trials": ("Number of trials per task", 1),
            "max_steps": ("Max agent steps per trial", 30),
            "max_concurrency": ("Max concurrent requests", 3),
            "max_tokens": ("Max tokens for agent LLM responses", None),
            "temperature": ("Temperature for agent LLM responses", None),
            "user_llm": ("LLM for simulated user (requires API key)", "gpt-4o-mini"),
            "user_temperature": ("Temperature for user LLM", None),
            "task_split_name": ("Task split to run (default: 'base')", None),
            "task_ids": ("Comma-separated task IDs to run", None),
            "num_tasks": ("Number of tasks to run (default: all)", None),
            "max_errors": ("Max consecutive tool errors allowed", None),
            "seed": ("Random seed for reproducibility", None),
            "log_level": ("Log level (DEBUG, INFO, WARNING, ERROR)", None),
            "enforce_communication_protocol": ("Enforce communication protocol rules", False),
        }

    async def execute(
        self,
        provider_url: str,
        model_name: str,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
        provider_kind: str | None = None,
    ) -> ExternalEvalResult:
        start_time = time.time()
        tau2_args = Tau2Args.from_dict(args)
        all_output: list[str] = []
        is_local = provider_kind in ("vllm", "vllm_server")

        try:
            from olmo_eval.harness.sandbox.executor import SandboxExecutor
        except ImportError as e:
            return self._error_result(f"SWE-ReX not installed: {e}", start_time)

        sandbox_config = self._create_sandbox_config(container_runtime, output_dir)

        try:
            async with SandboxExecutor(sandbox_config, name=self.name) as executor:
                if err := await self._run_setup(executor, all_output, start_time):
                    return err

                sandbox_url = self._get_provider_url_for_sandbox(provider_url)

                if not await self._check_provider_health(executor, sandbox_url):
                    return self._error_result(
                        f"Provider not reachable at {sandbox_url}",
                        start_time,
                        "\n".join(all_output),
                    )

                if is_local:
                    await self._setup_litellm_wrapper(executor, model_name, sandbox_url)

                run_cmd = self._build_run_command(model_name, sandbox_url, provider_kind, tau2_args)
                logger.info(f"[{self.name}] Running: {run_cmd}")

                run_result = await executor.execute_command(
                    run_cmd, timeout=self.timeout_seconds, stream=True, log_prefix=self.name
                )
                all_output.append(f"$ {run_cmd}\n{run_result.output}")
                logger.info(f"[{self.name}] Run exit code: {run_result.exit_code}")

                result = await self._extract_results(
                    executor, "\n".join(all_output), run_result.exit_code, tau2_args.num_trials
                )

        except Exception as e:
            logger.exception(f"[{self.name}] Execution failed")
            return self._error_result(str(e), start_time, "\n".join(all_output))

        result.duration_seconds = time.time() - start_time
        if output_dir:
            self._save_results(result, output_dir)

        return result

    def _build_run_command(
        self,
        model_name: str,
        provider_url: str,
        provider_kind: str | None,
        tau2_args: Tau2Args,
    ) -> str:
        """Build the tau2 run command."""
        is_local = provider_kind in ("vllm", "vllm_server")
        agent_model = f"hosted_vllm/{model_name}" if is_local else model_name
        repo = f"{self.working_dir}/tau2-bench"

        # Use wrapper for local providers (registers model with litellm)
        if is_local:
            tau2_cmd = f"{repo}/.venv/bin/python {repo}/tau2_wrapper.py run"
        else:
            tau2_cmd = f"{repo}/.venv/bin/tau2 run"

        parts = [f"cd {repo} &&", tau2_cmd, f"--agent-llm '{agent_model}'"]

        # Agent LLM args
        agent_llm_args: dict[str, Any] = {}
        if is_local:
            agent_llm_args["api_base"] = provider_url
        if tau2_args.max_tokens:
            agent_llm_args["max_tokens"] = tau2_args.max_tokens
        if tau2_args.temperature is not None:
            agent_llm_args["temperature"] = tau2_args.temperature
        if agent_llm_args:
            parts.append(f"--agent-llm-args '{json.dumps(agent_llm_args)}'")

        # User LLM
        parts.append(f"--user-llm '{tau2_args.user_llm}'")
        if tau2_args.user_temperature is not None:
            user_args = json.dumps({"temperature": tau2_args.user_temperature})
            parts.append(f"--user-llm-args '{user_args}'")

        # Core settings
        parts.extend(
            [
                f"--domain '{tau2_args.domain}'",
                f"--num-trials {tau2_args.num_trials}",
                f"--max-steps {tau2_args.max_steps}",
                f"--max-concurrency {tau2_args.max_concurrency}",
                "--save-to results",  # Saves to {repo}/data/simulations/results.json
            ]
        )

        # Optional args
        if tau2_args.task_split_name:
            parts.append(f"--task-split-name '{tau2_args.task_split_name}'")
        if tau2_args.task_ids:
            parts.append(f"--task-ids {' '.join(tau2_args.task_ids)}")
        if tau2_args.num_tasks is not None:
            parts.append(f"--num-tasks {tau2_args.num_tasks}")
        if tau2_args.max_errors is not None:
            parts.append(f"--max-errors {tau2_args.max_errors}")
        if tau2_args.seed is not None:
            parts.append(f"--seed {tau2_args.seed}")
        if tau2_args.log_level:
            parts.append(f"--log-level {tau2_args.log_level}")
        if tau2_args.enforce_communication_protocol:
            parts.append("--enforce-communication-protocol")

        return " ".join(parts)

    async def _setup_litellm_wrapper(
        self, executor: SandboxExecutor, model_name: str, provider_url: str
    ) -> None:
        """Create wrapper script that registers the model with litellm."""
        max_tokens = await self._query_max_tokens(executor, provider_url)
        repo = f"{self.working_dir}/tau2-bench"

        script = f'''\
#!/usr/bin/env python
"""Wrapper to register local vLLM model with litellm."""
import litellm
import sys

litellm.register_model({{
    "hosted_vllm/{model_name}": {{
        "max_tokens": {max_tokens},
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }}
}})

from tau2.cli import main
sys.exit(main())
'''
        encoded = base64.b64encode(script.encode()).decode()
        await executor.execute_command(
            f"echo '{encoded}' | base64 -d > {repo}/tau2_wrapper.py", timeout=30.0
        )
        logger.info(f"[{self.name}] Created litellm wrapper (max_tokens={max_tokens})")

    async def _query_max_tokens(self, executor: SandboxExecutor, provider_url: str) -> int:
        """Query vLLM server for max_model_len."""
        result = await executor.execute_command(
            f"curl -s {provider_url.rstrip('/')}/v1/models", timeout=30.0
        )
        if result.success and result.output:
            try:
                data = json.loads(result.output)
                if models := data.get("data"):
                    return models[0].get("max_model_len", DEFAULT_MAX_TOKENS)
            except json.JSONDecodeError:
                pass
        return DEFAULT_MAX_TOKENS

    async def _extract_results(
        self, executor: SandboxExecutor, raw_output: str, exit_code: int, num_trials: int
    ) -> ExternalEvalResult:
        """Extract metrics from tau2-bench results."""
        # tau2 saves results to {repo}/data/simulations/*.json
        results_path = f"{self.working_dir}/tau2-bench/data/simulations"
        ls_result = await executor.execute_command(
            f"ls {results_path}/*.json 2>/dev/null", timeout=30.0
        )
        if not ls_result.success:
            return ExternalEvalResult(
                name=self.name, success=False, error="No results files found", raw_output=raw_output
            )

        all_metrics: dict[str, float] = {}
        metadata: dict[str, Any] = {}

        for json_file in ls_result.output.strip().split("\n"):
            if not (json_file := json_file.strip()):
                continue

            cat_result = await executor.execute_command(f"cat {json_file}", timeout=30.0)
            if not cat_result.success:
                continue

            try:
                data = json.loads(cat_result.output)
                if "simulations" in data and "tasks" in data:
                    all_metrics.update(self._compute_pass_k_metrics(data, num_trials))
                    metadata["simulations_file"] = json_file
            except json.JSONDecodeError as e:
                logger.warning(f"[{self.name}] Failed to parse {json_file}: {e}")

        return ExternalEvalResult(
            name=self.name,
            success=exit_code == 0 and bool(all_metrics),
            metrics=all_metrics,
            metadata=metadata,
            raw_output=raw_output,
        )

    def _compute_pass_k_metrics(self, data: dict[str, Any], num_trials: int) -> dict[str, float]:
        """Compute pass^k metrics from tau2-bench simulations.

        See: https://arxiv.org/abs/2406.12045
        """
        task_ids = {task["id"] for task in data["tasks"]}
        simulation_ids = {sim["task_id"] for sim in data["simulations"]}

        if task_ids != simulation_ids:
            logger.warning(f"[{self.name}] Missing simulations: {task_ids - simulation_ids}")
            return {}

        # Group rewards by task
        rewards_by_task: dict[str, list[float]] = defaultdict(list)
        for sim in data["simulations"]:
            rewards_by_task[sim["task_id"]].append(sim["reward_info"]["reward"])

        # Compute pass^k for each k
        metrics: dict[str, float] = {}
        for k in range(1, num_trials + 1):
            pass_k_values = []
            for rewards in rewards_by_task.values():
                c = int(sum(rewards))
                pass_k_values.append(math.comb(c, k) / math.comb(num_trials, k))
            if pass_k_values:
                metrics[f"pass^{k}"] = sum(pass_k_values) / len(pass_k_values)

        return metrics


register_external_eval(Tau2ExternalEval())
