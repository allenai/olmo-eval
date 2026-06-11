"""DeepScholar-Bench external evaluation (EXPERIMENTAL / UNVALIDATED).

DeepScholar-Bench (Sky Computing Lab, arXiv 2508.20033) evaluates generative
research synthesis: given a paper, generate its related-work section by
retrieving, synthesizing, and citing prior work. Scoring spans knowledge
synthesis, retrieval quality, and verifiability, aggregated as a geometric mean.

This is a SKELETON. It follows the SandboxedExternalEval pattern (modeled on
tau2/eval.py) and stakes out the integration, but NOTHING here has been run:
there is no Docker/GPU/API-key environment available where it was authored. It
must be validated by a real run (e.g. on beaker) and will need fixes. The two
load-bearing unknowns are marked TODO inline and enumerated in
plans/003_deepscholar_bench.md:

1. Model integration: the model under test is configured via a LOTUS config
   YAML (`configs/deepscholar_base.yaml`). Pointing LOTUS's `LM` at a local
   vLLM endpoint (model string + api_base) is unverified.
2. Result schema: `eval.main` writes `results.csv`; its exact columns and the
   geomean computation are not confirmed (see result_parser.py).

Design choices:
- Pin to the repo's shipped dataset snapshot (dataset/*.csv) rather than the
  live arXiv pipeline, so runs are reproducible. The live data_pipeline is out
  of scope for this skeleton.
- The eval metrics use a gpt-4o judge (OPENAI_API_KEY); generation uses Tavily
  web search (TAVILY_API_KEY). Both are required.
"""

from __future__ import annotations

import base64
import logging
import shlex
import time
from typing import TYPE_CHECKING, Any

from olmo_eval.evals.external.base import SandboxedExternalEval
from olmo_eval.evals.external.benchmarks.deepscholar.args import DeepScholarArgs
from olmo_eval.evals.external.benchmarks.deepscholar.result_parser import parse_results_csv
from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox.executor import SandboxExecutor
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

# TODO(unvalidated): pin to a specific upstream commit for reproducibility.
DEEPSCHOLAR_REPO = "https://github.com/guestrin-lab/deepscholar-bench.git"
DEEPSCHOLAR_REF = "main"

DEFAULT_MAX_TOKENS = 8192


class DeepScholarExternalEval(SandboxedExternalEval):
    """DeepScholar-Bench generative research-synthesis eval (experimental skeleton)."""

    @property
    def name(self) -> str:
        return "deepscholar_bench"

    @property
    def description(self) -> str:
        return (
            "EXPERIMENTAL. Generative research synthesis: generate a paper's related-work "
            "section by retrieving, synthesizing, and citing prior work. Scored on synthesis, "
            "retrieval, and verifiability (geomean). Unvalidated; needs a real run."
        )

    @property
    def sandbox_image(self) -> str:
        # DeepScholar requires Python 3.10. uv image ships git + a 3.10 interpreter.
        return "ghcr.io/astral-sh/uv:python3.10-bookworm"

    @property
    def working_dir(self) -> str:
        return "/workspace"

    @property
    def timeout_seconds(self) -> float:
        return 14400.0  # 4h: generation + judge-based eval over many queries is slow

    @property
    def required_secrets(self) -> tuple[str, ...]:
        # OPENAI_API_KEY: eval judge models. TAVILY_API_KEY: generation web search.
        return ("OPENAI_API_KEY", "TAVILY_API_KEY")

    @property
    def setup_command(self) -> tuple[str, ...]:
        repo = f"{self.working_dir}/deepscholar-bench"
        return (
            f"git clone {DEEPSCHOLAR_REPO} {repo}",
            f"cd {repo} && git checkout {DEEPSCHOLAR_REF}",
            # DeepScholar installs via pip + requirements.txt (no pyproject upstream).
            f"cd {repo} && uv venv && uv pip install -r requirements.txt",
            f"mkdir -p {self.results_dir}",
        )

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "modes": ("System(s) to evaluate", "deepscholar_base"),
            "evals": ("Eval metrics to run, or 'all'", "all"),
            "dataset_path": ("Dataset CSV (shipped snapshot)", DeepScholarArgs.dataset_path),
            "config_yaml": ("LOTUS generation config path", DeepScholarArgs.config_yaml),
            "eval_model": ("Judge model used by eval metrics (not the model under test)", "gpt-4o"),
            "max_tokens": ("Max tokens for the model under test", None),
            "temperature": ("Temperature for the model under test", None),
            "max_model_len": ("Model context length for litellm registration", None),
            "num_queries": ("Limit number of queries (smoke run)", None),
            "file_id": ("Run a single query by id", None),
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

                # TODO(unvalidated): the LOTUS config schema is a best-effort guess.
                await self._write_model_config(executor, ds_args, model_name, sandbox_url, is_local)

                gen_cmd = self._build_generation_command(ds_args)
                gen = await executor.execute_command(
                    gen_cmd, timeout=self.timeout_seconds, stream=True, log_prefix=self.name
                )
                all_output.append(f"$ {gen_cmd}\n{gen.output}")

                eval_cmd = self._build_eval_command(ds_args)
                ev = await executor.execute_command(
                    eval_cmd, timeout=self.timeout_seconds, stream=True, log_prefix=self.name
                )
                all_output.append(f"$ {eval_cmd}\n{ev.output}")

                result = await self._extract_results(executor, "\n".join(all_output), ev.exit_code)
        except Exception as e:
            logger.exception(f"[{self.name}] Execution failed")
            return self._error_result(str(e), start_time, "\n".join(all_output))

        result.duration_seconds = time.time() - start_time
        if output_dir:
            self._save_results(result, output_dir)
        return result

    async def _write_model_config(
        self,
        executor: SandboxExecutor,
        ds_args: DeepScholarArgs,
        model_name: str,
        sandbox_url: str,
        is_local: bool,
    ) -> None:
        """Write the LOTUS generation config pointing at the model under test.

        TODO(unvalidated): the exact LOTUS `LM` config schema is unconfirmed. LOTUS
        uses litellm, so a local vLLM endpoint should be reachable via a
        `hosted_vllm/<model>` model string plus an `api_base`. This writer is a
        best-effort guess and must be reconciled with the upstream config format.
        """
        repo = f"{self.working_dir}/deepscholar-bench"
        model_str = f"hosted_vllm/{model_name}" if is_local else model_name
        max_tokens = ds_args.max_tokens or ds_args.max_model_len or DEFAULT_MAX_TOKENS

        lines = [
            "# Auto-generated by olmo-eval (EXPERIMENTAL, unvalidated).",
            "lm:",
            f"  model: {model_str}",
            f"  max_tokens: {max_tokens}",
        ]
        if ds_args.temperature is not None:
            lines.append(f"  temperature: {ds_args.temperature}")
        if is_local:
            lines.append(f"  api_base: {sandbox_url}")
            lines.append("  api_key: local")
        config_text = "\n".join(lines) + "\n"

        encoded = base64.b64encode(config_text.encode()).decode()
        path = shlex.quote(f"{repo}/{ds_args.config_yaml}")
        await executor.execute_command(f"echo '{encoded}' | base64 -d > {path}", timeout=30.0)
        logger.warning(
            "[%s] Wrote best-effort LOTUS config to %s; schema is UNVALIDATED.",
            self.name,
            ds_args.config_yaml,
        )

    def _build_generation_command(self, ds_args: DeepScholarArgs) -> str:
        """Generation: run DeepScholar-ref to produce related-work sections."""
        repo = f"{self.working_dir}/deepscholar-bench"
        parts = [
            f"cd {repo} &&",
            f"{repo}/.venv/bin/python -m deepscholar_base.main",
            "--queries-file",
            shlex.quote("dataset/queries.csv"),
            "--output-folder",
            shlex.quote(f"{self.results_dir}/generated"),
            "--config-yaml",
            shlex.quote(ds_args.config_yaml),
        ]
        if ds_args.num_queries is not None:
            # TODO(unvalidated): confirm upstream supports a limit flag; name is a guess.
            parts.extend(["--num-queries", str(ds_args.num_queries)])
        if ds_args.file_id:
            parts.extend(["--file-id", shlex.quote(ds_args.file_id)])
        return " ".join(parts)

    def _build_eval_command(self, ds_args: DeepScholarArgs) -> str:
        """Eval: score generated sections with the judge metrics."""
        repo = f"{self.working_dir}/deepscholar-bench"
        parts = [
            f"cd {repo} &&",
            f"{repo}/.venv/bin/python -m eval.main",
            "--modes",
            shlex.quote(ds_args.modes),
            "--evals",
            shlex.quote(ds_args.evals),
            "--input-folder",
            shlex.quote(f"{self.results_dir}/generated"),
            "--output-folder",
            shlex.quote(self.results_dir),
            "--dataset-path",
            shlex.quote(ds_args.dataset_path),
            "--model-name",
            shlex.quote(ds_args.eval_model),
        ]
        return " ".join(parts)

    async def _extract_results(
        self,
        executor: SandboxExecutor,
        raw_output: str,
        exit_code: int,
    ) -> ExternalEvalResult:
        """Read results.csv from the sandbox and parse into metrics."""
        results_csv = f"{self.results_dir}/results.csv"
        cat = await executor.execute_command(f"cat {shlex.quote(results_csv)}", timeout=60.0)
        if not cat.success:
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error=f"No results.csv at {results_csv}",
                raw_output=raw_output,
            )

        parsed = parse_results_csv(cat.output)
        metrics = parsed["metrics"]
        return ExternalEvalResult(
            name=self.name,
            success=exit_code == 0 and bool(metrics),
            metrics=metrics,
            metadata=parsed["metadata"],
            raw_output=raw_output,
        )
