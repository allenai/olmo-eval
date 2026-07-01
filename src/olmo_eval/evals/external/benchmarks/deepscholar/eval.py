"""DeepScholar-Bench external evaluation implementation.

DeepScholar-Bench (arXiv 2508.20033) evaluates generative research synthesis:
given a paper's context, a system retrieves prior work and writes the
related-work section, scored on organization, nugget coverage, reference
coverage, and citation precision. We run it in two phases inside one sandbox:
generation (the model under test, driven through LOTUS) then eval (an external
judge model scoring the generated sections).

Modeled on the tau2 external eval: a stock uv image, repo cloned at setup, model
under test wired in via a generated LOTUS config rather than a CLI flag.

Repository: https://github.com/guestrin-lab/deepscholar-bench
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from olmo_eval.evals.external.base import SandboxedExternalEval
from olmo_eval.evals.external.benchmarks.deepscholar.args import DeepScholarArgs
from olmo_eval.evals.external.benchmarks.deepscholar.result_parser import (
    compute_geomean,
    flatten_numeric,
    parse_aggregate_csv,
)
from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox.executor import SandboxExecutor
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

# Pin once a validation run succeeds (see plans/006_deepscholar_bench.md).
DEEPSCHOLAR_REF = "main"

# Mirrors configs/deepscholar_base.yaml; the `lm` block is filled in per run.
_BASE_CONFIG: dict[str, Any] = {
    "queries_file": "dataset/queries.csv",
    "max_search_retries": 3,
    "use_structured_output": True,
    "enable_web_search": True,
    "per_query_max_search_results_count": 10,
    "use_responses_model": None,
    "num_search_steps": 3,
    "num_search_queries_per_step_per_corpus": 2,
    "use_sem_filter": True,
    "use_sem_topk": True,
    "final_max_results_count": 30,
    "categorize_references": True,
    "generate_category_summary": True,
    "generate_insights": True,
}


class DeepScholarExternalEval(SandboxedExternalEval):
    """DeepScholar-Bench evaluation for generative research synthesis."""

    @property
    def name(self) -> str:
        return "deepscholar_bench"

    @property
    def description(self) -> str:
        return (
            "Evaluates generative research synthesis: given a paper's context, the model "
            "retrieves prior work and writes a related-work section, scored on organization, "
            "nugget/reference coverage, and citation precision (geometric mean)."
        )

    @property
    def sandbox_image(self) -> str:
        return "ghcr.io/astral-sh/uv:python3.10-bookworm"

    @property
    def working_dir(self) -> str:
        return "/workspace"

    @property
    def timeout_seconds(self) -> float:
        return 14400.0  # 4 hours; generation + judge over many queries

    @property
    def _repo(self) -> str:
        return f"{self.working_dir}/deepscholar-bench"

    @property
    def _venv_python(self) -> str:
        return f"{self._repo}/.venv/bin/python"

    @property
    def _gen_dir(self) -> str:
        return f"{self._repo}/outputs/generation"

    @property
    def _eval_dir(self) -> str:
        return f"{self._repo}/outputs/evaluation"

    @property
    def _config_path(self) -> str:
        return f"{self._repo}/olmo_eval_config.yaml"

    @property
    def setup_command(self) -> tuple[str, ...]:
        repo_url = "https://github.com/guestrin-lab/deepscholar-bench.git"
        # The repo is ~1.3GB (dataset CSVs + baseline outputs), so a full clone is
        # slow. Shallow-fetch just the target ref (works for a branch name or a
        # commit SHA); no submodules exist. checkout FETCH_HEAD lands the snapshot.
        return (
            f"git init {self._repo}",
            f"cd {self._repo} && git remote add origin {repo_url}",
            f"cd {self._repo} && git fetch --depth 1 origin {DEEPSCHOLAR_REF}",
            f"cd {self._repo} && git checkout FETCH_HEAD",
            f"cd {self._repo} && uv venv --python 3.10",
            # Target our .venv explicitly: the swe-rex derived image ships an active
            # /root/venv (3.12), which uv would otherwise install into by default.
            f"cd {self._repo} && uv pip install --python {self._venv_python} -r requirements.txt",
            # The eval phase's cite_p scorer calls nltk.sent_tokenize, which needs the
            # punkt_tab tokenizer data (not bundled with the pip install).
            f"{self._venv_python} -m nltk.downloader punkt_tab",
            f"mkdir -p {self._gen_dir} {self._eval_dir}",
        )

    @property
    def required_secrets(self) -> tuple[str, ...]:
        # Only OPENAI_API_KEY (the gpt-4o judge) is always required. Web search
        # defaults to the keyless ARXIV corpus; TAVILY_API_KEY is forwarded only
        # if set (see _build_env_vars), for users who opt into the TAVILY corpus.
        return ("OPENAI_API_KEY",)

    def _build_env_vars(self, secrets: tuple[str, ...] | None = None) -> dict[str, str]:
        env = super()._build_env_vars(secrets)
        # Optional: forward web-search keys when present so a non-default corpus
        # (e.g. -a web_corpuses=TAVILY) works without making the key mandatory.
        for optional in ("TAVILY_API_KEY", "S2_API_KEY", "SERPAPI_API_KEY"):
            value = os.environ.get(optional)
            if value:
                env[optional] = value
        return env

    def _create_sandbox_config(self, container_runtime: str, output_dir: str | None = None) -> Any:
        # The bare uv base image has no swe-rex; inject_swerex builds a derived
        # image with swe-rex (plus git/curl) preinstalled, avoiding the failing
        # runtime bootstrap. Same approach as the scicode external eval.
        from dataclasses import replace

        config = super()._create_sandbox_config(container_runtime, output_dir)
        return replace(config, inject_swerex=True)

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "limit": ("Number of queries to run (maps to generation --end-idx)", None),
            "start_idx": ("Starting query index", 0),
            "search_mode": ("Search mode: 'agentic' or 'recursive' (default: config value)", None),
            "web_corpuses": (
                "Search corpus: ARXIV (keyless), TAVILY, GOOGLE, GOOGLE_SCHOLAR, BING",
                "ARXIV",
            ),
            "temperature": ("Generation temperature for the model under test", None),
            "max_tokens": ("Max tokens for the model under test", 10000),
            "local_model_prefix": (
                "litellm prefix for local vLLM ('openai' or 'hosted_vllm')",
                "openai",
            ),
            "judge_model": ("Judge model for the eval phase", "gpt-4o"),
            "evals": (
                "Comma-separated eval metrics, or 'all' for the full upstream set",
                "organization,nugget_coverage,reference_coverage,cite_p",
            ),
            "allow_partial_generation": (
                "Score even if some generation queries failed (default: require all)",
                False,
            ),
            "extra_gen_args": ("Extra args appended to the generation command", None),
            "extra_eval_args": ("Extra args appended to the eval command", None),
        }

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        start_time = time.time()
        ds_args = DeepScholarArgs.from_dict(args)
        all_output: list[str] = []

        provider_url = getattr(provider, "base_url", None) or "http://localhost:8000/v1"
        model_name = provider.model_name
        is_local = self._is_local_provider(provider, provider_url)

        # Upstream's agentic search builds a raw OpenAI Agents SDK client and sends
        # `lm.model` verbatim, so the litellm-style "openai/<model>" prefix the LOTUS
        # sem-ops require would be rejected by a vLLM server. Recursive search keeps
        # every model call on the LOTUS/litellm path, where the prefix is consistent.
        # External API models keep the upstream default (agentic).
        if is_local and ds_args.search_mode is None:
            ds_args.search_mode = "recursive"

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
                if is_local and not await self._check_provider_health(executor, sandbox_url):
                    return self._error_result(
                        f"Provider not reachable at {sandbox_url}",
                        start_time,
                        "\n".join(all_output),
                    )

                await self._write_config(executor, model_name, sandbox_url, is_local, ds_args)

                gen_cmd = self._build_generation_command(ds_args)
                logger.info(f"[{self.name}] Generation: {gen_cmd}")
                gen_result = await executor.execute_command(
                    gen_cmd,
                    timeout=self.timeout_seconds,
                    stream=True,
                    log_prefix=f"{self.name}-gen",
                )
                all_output.append(f"$ {gen_cmd}\n{gen_result.output}")
                logger.info(f"[{self.name}] Generation exit code: {gen_result.exit_code}")

                if not gen_result.success:
                    return self._error_result(
                        "Generation phase failed", start_time, "\n".join(all_output)
                    )

                # Generation catches per-query exceptions and still exits 0. The eval
                # then scores only the folders that succeeded, so partial generation
                # would silently report metrics over a smaller subset than requested.
                # Require every query to succeed unless partial runs are opted into.
                n_success, n_total = await self._generation_counts(executor)
                logger.info(f"[{self.name}] Generation: {n_success}/{n_total} queries succeeded")
                incomplete = n_success != n_total and not ds_args.allow_partial_generation
                if n_success == 0 or incomplete:
                    if output_dir:
                        await self._copy_back(executor, output_dir)
                    return self._error_result(
                        f"Generation incomplete: {n_success}/{n_total} queries succeeded; "
                        "skipping eval to avoid scoring a partial subset "
                        "(pass -a allow_partial_generation=true to score anyway; "
                        "see copied summary.json)",
                        start_time,
                        "\n".join(all_output),
                    )

                eval_cmd = self._build_eval_command(ds_args)
                logger.info(f"[{self.name}] Eval: {eval_cmd}")
                eval_result = await executor.execute_command(
                    eval_cmd,
                    timeout=self.timeout_seconds,
                    stream=True,
                    log_prefix=f"{self.name}-eval",
                )
                all_output.append(f"$ {eval_cmd}\n{eval_result.output}")
                logger.info(f"[{self.name}] Eval exit code: {eval_result.exit_code}")

                result = await self._extract_results(
                    executor,
                    "\n".join(all_output),
                    eval_result.exit_code,
                    output_dir,
                    n_success=n_success,
                    n_total=n_total,
                )

        except Exception as e:
            logger.exception(f"[{self.name}] Execution failed")
            return self._error_result(str(e), start_time, "\n".join(all_output))

        result.duration_seconds = time.time() - start_time
        if output_dir:
            self._save_results(result, output_dir)
        return result

    def _build_lm_config(
        self, model_name: str, sandbox_url: str, is_local: bool, ds_args: DeepScholarArgs
    ) -> dict[str, Any]:
        """Build the LOTUS `lm` block pointing at the model under test."""
        lm: dict[str, Any] = {
            "temperature": ds_args.temperature if ds_args.temperature is not None else 1.0,
            "max_tokens": ds_args.max_tokens,
        }
        if is_local:
            # litellm routes "<prefix>/<model>" to the OpenAI-compatible vLLM server at api_base.
            lm["model"] = f"{ds_args.local_model_prefix}/{model_name}"
            lm["api_base"] = sandbox_url
            lm["api_key"] = "EMPTY"
        else:
            lm["model"] = model_name
        return lm

    async def _write_config(
        self,
        executor: SandboxExecutor,
        model_name: str,
        sandbox_url: str,
        is_local: bool,
        ds_args: DeepScholarArgs,
    ) -> None:
        """Write the LOTUS config into the sandbox (JSON is valid YAML)."""
        config = dict(_BASE_CONFIG)
        config["lm"] = self._build_lm_config(model_name, sandbox_url, is_local, ds_args)
        config["web_corpuses"] = ds_args.web_corpuses
        if ds_args.search_mode:
            config["search_mode"] = ds_args.search_mode
        content = json.dumps(config, indent=2)
        encoded = base64.b64encode(content.encode()).decode()
        await executor.execute_command(
            f"echo '{encoded}' | base64 -d > {shlex.quote(self._config_path)}", timeout=30.0
        )
        logger.info(f"[{self.name}] Wrote LOTUS config (model={config['lm']['model']})")

    def _build_generation_command(self, ds_args: DeepScholarArgs) -> str:
        parts = [
            f"cd {self._repo} &&",
            self._venv_python,
            "-m",
            "deepscholar_base.main",
            "--output-folder",
            shlex.quote(self._gen_dir),
            "--config-yaml",
            shlex.quote(self._config_path),
            "--start-idx",
            str(ds_args.start_idx),
        ]
        if ds_args.limit is not None:
            parts.extend(["--end-idx", str(ds_args.start_idx + ds_args.limit)])
        # search_mode is carried in the generated config (see _write_config).
        parts.extend(ds_args.extra_gen_args)
        return " ".join(parts)

    def _build_eval_command(self, ds_args: DeepScholarArgs) -> str:
        parts = [
            f"cd {self._repo} &&",
            self._venv_python,
            "-m",
            "eval.main",
            "--modes",
            "deepscholar_base",
            "--evals",
            *ds_args.evals,
            "--input-folder",
            shlex.quote(self._gen_dir),
            "--output-folder",
            shlex.quote(self._eval_dir),
            "--model-name",
            shlex.quote(ds_args.judge_model),
        ]
        parts.extend(ds_args.extra_eval_args)
        return " ".join(parts)

    async def _generation_counts(self, executor: SandboxExecutor) -> tuple[int, int]:
        """Return (successful_queries, total_queries) from generation summary.json."""
        cat = await executor.execute_command(
            f"cat {shlex.quote(self._gen_dir)}/summary.json", timeout=60.0
        )
        if not (cat.success and cat.output.strip()):
            return (0, 0)
        try:
            summary = json.loads(cat.output)
        except json.JSONDecodeError:
            return (0, 0)
        if not isinstance(summary, list):
            return (0, 0)
        n_success = sum(1 for r in summary if isinstance(r, dict) and r.get("status") == "success")
        return (n_success, len(summary))

    async def _read_dir(
        self, executor: SandboxExecutor, remote_dir: str, pattern: str
    ) -> dict[str, str]:
        """Return {relative_path: file_text} for files matching pattern under remote_dir."""
        find = await executor.execute_command(
            f"find {shlex.quote(remote_dir)} -type f -name {shlex.quote(pattern)} 2>/dev/null",
            timeout=60.0,
        )
        files: dict[str, str] = {}
        if not find.success or not find.output.strip():
            return files
        for remote_path in (p.strip() for p in find.output.strip().split("\n") if p.strip()):
            cat = await executor.execute_command(f"cat {shlex.quote(remote_path)}", timeout=60.0)
            if cat.success:
                rel = remote_path.replace(remote_dir.rstrip("/") + "/", "")
                files[rel] = cat.output
        return files

    async def _extract_results(
        self,
        executor: SandboxExecutor,
        raw_output: str,
        exit_code: int,
        output_dir: str | None = None,
        n_success: int = 0,
        n_total: int = 0,
    ) -> ExternalEvalResult:
        # Canonical layout: evaluation/<metric>/aggregated_results.csv holds the
        # aggregate for each metric on a single `deepscholar_base` row. The metric
        # name is the parent directory.
        agg_files = await self._read_dir(executor, self._eval_dir, "aggregated_results.csv")
        all_metrics: dict[str, float] = {}
        for rel, text in sorted(agg_files.items()):
            metric = rel.split("/")[0]
            value = parse_aggregate_csv(text, metric)
            if value is not None:
                all_metrics[metric] = value

        # Fallback if the upstream layout changes: scan any JSON for numeric leaves.
        if not all_metrics:
            for rel, text in (await self._read_dir(executor, self._eval_dir, "*.json")).items():
                try:
                    parsed = flatten_numeric(json.loads(text))
                except json.JSONDecodeError:
                    continue
                for key, value in parsed.items():
                    all_metrics[f"{Path(rel).stem}.{key}"] = value

        # Headline geomean over the primary metrics (None if any is missing).
        geomean = compute_geomean(all_metrics)
        if geomean is not None:
            all_metrics["geomean"] = geomean

        if output_dir:
            await self._copy_back(executor, output_dir)

        if not all_metrics:
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error="No metrics parsed from eval outputs (check raw_output / copied files)",
                raw_output=raw_output,
            )

        success = exit_code == 0 and bool(all_metrics)
        error = None if success else f"Eval phase exited {exit_code} (metrics may be partial)"
        return ExternalEvalResult(
            name=self.name,
            success=success,
            error=error,
            metrics=all_metrics,
            metadata={
                "eval_dir": self._eval_dir,
                "ref": DEEPSCHOLAR_REF,
                "queries_succeeded": n_success,
                "queries_total": n_total,
            },
            raw_output=raw_output,
        )

    async def _copy_back(self, executor: SandboxExecutor, output_dir: str) -> None:
        """Copy eval outputs and generation summary back for inspection."""
        dest = Path(output_dir) / "deepscholar_results"
        for remote_dir, pattern, subdir in (
            (self._eval_dir, "*", "evaluation"),
            (self._gen_dir, "summary.json", "generation"),
        ):
            find = await executor.execute_command(
                f"find {shlex.quote(remote_dir)} -type f -name {shlex.quote(pattern)} 2>/dev/null",
                timeout=60.0,
            )
            if not find.success or not find.output.strip():
                continue
            for remote_path in (p.strip() for p in find.output.strip().split("\n") if p.strip()):
                read = await executor.execute_command(
                    f"base64 {shlex.quote(remote_path)}", timeout=120.0
                )
                if not (read.success and read.output.strip()):
                    continue
                rel = remote_path.replace(remote_dir.rstrip("/") + "/", "")
                local_path = dest / subdir / rel
                local_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    local_path.write_bytes(base64.b64decode(read.output.strip()))
                except Exception as e:
                    logger.warning(f"[{self.name}] Failed to copy {rel}: {e}")
