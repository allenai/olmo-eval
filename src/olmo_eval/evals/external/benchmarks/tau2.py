"""Tau2-bench external evaluation.

tau2_bench is a benchmark for evaluating language model agents on realistic
customer service tasks. It measures both task completion and constraint
satisfaction.

Repository: https://github.com/sierra-research/tau2-bench
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Any, Literal

from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.registry import register_external_eval
from olmo_eval.evals.external.result import ExternalEvalResult

logger = logging.getLogger(__name__)

Tau2Domain = Literal["airline", "retail", "telecom"]


@dataclass
class Tau2Args:
    """Arguments for tau2_bench evaluation."""

    domain: Tau2Domain = "airline"
    user_llm: str = "gpt-4o-mini"
    num_trials: int = 5
    max_steps: int = 200
    max_concurrency: int = 8
    max_tokens: int | None = None
    temperature: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tau2Args:
        # Get defaults from dataclass fields
        defaults = {f.name: f.default for f in fields(cls)}
        return cls(
            domain=data.get("domain", defaults["domain"]),
            user_llm=data.get("user_llm", defaults["user_llm"]),
            num_trials=int(data.get("num_trials", defaults["num_trials"])),
            max_steps=int(data.get("max_steps", defaults["max_steps"])),
            max_concurrency=int(data.get("max_concurrency", defaults["max_concurrency"])),
            max_tokens=int(data["max_tokens"]) if data.get("max_tokens") else None,
            temperature=float(data["temperature"]) if data.get("temperature") else None,
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
        """Command template for display"""
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
        defaults = {f.name: f.default for f in fields(Tau2Args)}
        return {
            "domain": ("Task domain: 'airline', 'retail', or 'telecom'", defaults["domain"]),
            "user_llm": ("LLM for simulated user (requires API key)", defaults["user_llm"]),
            "num_trials": ("Number of trials per task", defaults["num_trials"]),
            "max_steps": ("Max agent steps per trial", defaults["max_steps"]),
            "max_concurrency": ("Max concurrent requests", defaults["max_concurrency"]),
            "max_tokens": ("Max tokens for agent LLM responses", defaults["max_tokens"]),
            "temperature": ("Temperature for agent LLM responses", defaults["temperature"]),
        }

    async def execute(
        self,
        provider_url: str,
        model_name: str,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
        use_host_network: bool = False,
        provider_kind: str | None = None,
    ) -> ExternalEvalResult:
        start_time = time.time()
        tau2_args = Tau2Args.from_dict(args)
        all_output: list[str] = []

        try:
            from olmo_eval.harness.sandbox.executor import SandboxExecutor
        except ImportError as e:
            return self._error_result(f"SWE-ReX not installed: {e}", start_time)

        sandbox_config = self._create_sandbox_config(container_runtime, use_host_network)

        try:
            async with SandboxExecutor(sandbox_config, name=self.name) as executor:
                setup_result = await self._run_setup(executor, all_output, start_time)
                if setup_result:
                    return setup_result

                # Run evaluation
                run_cmd = self._build_run_command(
                    model_name, provider_url, provider_kind, tau2_args
                )
                logger.info(f"[{self.name}] Running: {run_cmd}")

                run_result = await executor.execute_command(run_cmd, timeout=self.timeout_seconds)
                all_output.append(f"$ {run_cmd}\n{run_result.output}")

                # Debug logging
                logger.info(f"[{self.name}] Run exit code: {run_result.exit_code}")
                output_preview = run_result.output[-3000:] if run_result.output else "(empty)"
                logger.info(f"[{self.name}] Run output (last 3000 chars):\n{output_preview}")

                result = await self._extract_results(
                    executor,
                    "\n".join(all_output),
                    run_result.exit_code,
                    tau2_args.num_trials,
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
        """Build the full run command with all arguments."""
        is_local = provider_kind in ("vllm", "vllm_server")
        agent_model = f"hosted_vllm/{model_name}" if is_local else model_name

        parts = [
            f"cd {self.working_dir}/tau2-bench &&",
            "~/.local/bin/uv run python -m tau2.run",
            f"--agent-llm '{agent_model}'",
        ]

        llm_args = {
            k: v
            for k, v in [
                ("api_base", provider_url if is_local else None),
                ("max_tokens", tau2_args.max_tokens),
                ("temperature", tau2_args.temperature),
            ]
            if v is not None
        }
        if llm_args:
            parts.append(f"--agent-llm-args '{json.dumps(llm_args)}'")

        parts.extend(
            [
                f"--user-llm '{tau2_args.user_llm}'",
                f"--domain '{tau2_args.domain}'",
                f"--num-trials {tau2_args.num_trials}",
                f"--max-steps {tau2_args.max_steps}",
                f"--max-concurrency {tau2_args.max_concurrency}",
                f"--save-to {self.results_dir}",
            ]
        )

        return " ".join(parts)

    async def _extract_results(
        self, executor: Any, raw_output: str, exit_code: int, num_trials: int
    ) -> ExternalEvalResult:
        """Extract metrics from tau2-bench simulations file."""
        # Debug: list results directory
        ls_dir = await executor.execute_command(f"ls -la {self.results_dir} 2>&1", timeout=30.0)
        logger.info(f"[{self.name}] Results dir contents:\n{ls_dir.output}")

        ls_result = await executor.execute_command(
            f"ls {self.results_dir}/*.json 2>/dev/null", timeout=30.0
        )
        json_files = ls_result.output if ls_result.success else "(none)"
        logger.info(f"[{self.name}] JSON files: {json_files}")
        if not ls_result.success:
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error="No results files found",
                raw_output=raw_output,
            )

        all_metrics: dict[str, float] = {}
        metadata: dict[str, Any] = {}

        for json_file in ls_result.output.strip().split("\n"):
            json_file = json_file.strip()
            if not json_file:
                continue

            try:
                cat_result = await executor.execute_command(f"cat {json_file}", timeout=30.0)
                if not cat_result.success:
                    continue

                data = json.loads(cat_result.output)

                if "simulations" in data and "tasks" in data:
                    domain_metrics = self._compute_pass_k_metrics(data, num_trials)
                    all_metrics.update(domain_metrics)
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

        Tau-bench reports pass^k metrics. See for details: https://arxiv.org/abs/2406.12045
        """
        task_ids = {task["id"] for task in data["tasks"]}
        simulation_ids = {sim["task_id"] for sim in data["simulations"]}

        if task_ids != simulation_ids:
            missing = task_ids - simulation_ids
            logger.warning(f"[{self.name}] Missing simulations for tasks: {missing}")
            return {}

        metrics: dict[str, float] = {}

        if num_trials == 1:
            rewards = [sim["reward_info"]["reward"] for sim in data["simulations"]]
            metrics["pass^1"] = sum(rewards) / len(rewards) if rewards else 0.0
        else:
            rewards_by_task: dict[str, list[float]] = defaultdict(list)
            for sim in data["simulations"]:
                rewards_by_task[sim["task_id"]].append(sim["reward_info"]["reward"])

            instance_pass_k: list[dict[int, float]] = []
            for instance_rewards in rewards_by_task.values():
                c = int(sum(instance_rewards))
                instance_pass_k.append(
                    {
                        k: math.comb(c, k) / math.comb(num_trials, k)
                        for k in range(1, num_trials + 1)
                    }
                )

            if instance_pass_k:
                for k in range(1, num_trials + 1):
                    metrics[f"pass^{k}"] = sum(inst[k] for inst in instance_pass_k) / len(
                        instance_pass_k
                    )

        return metrics


register_external_eval(Tau2ExternalEval())
