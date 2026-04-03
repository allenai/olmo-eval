"""SWE-bench external evaluation implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

# Maps short aliases to HuggingFace dataset paths used by the SWE-bench harness.
# mini-swe-agent has its own alias table; we pass aliases through to it directly.
_DATASET_HF_PATHS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}


@dataclass
class SWEBenchArgs:
    """Arguments for the swe_bench evaluation."""

    dataset: str = "lite"
    split: str = "test"
    instance_filter: str = ""
    instance_slice: str = ""
    workers: int = 4
    max_workers_eval: int = 4
    temperature: float = 0.0
    max_turns: int = 30
    use_modal: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SWEBenchArgs:
        return cls(
            dataset=data.get("dataset", "lite"),
            split=data.get("split", "test"),
            instance_filter=data.get("instance_filter", ""),
            instance_slice=data.get("instance_slice", ""),
            workers=int(data.get("workers", 4)),
            max_workers_eval=int(data.get("max_workers_eval", 4)),
            temperature=float(data.get("temperature", 0.0)),
            max_turns=int(data.get("max_turns", 30)),
            use_modal=data.get("use_modal", False) in (True, "true", "True", "1", 1),
        )


class SWEBenchExternalEval(ExternalEval):
    """SWE-bench evaluation.

    Uses mini-swe-agent for patch generation and the official SWE-bench harness
    for scoring. Both run on the host (not inside a sandbox container), as the
    SWE-bench harness manages its own per-instance Docker containers.
    """

    @property
    def name(self) -> str:
        return "swe_bench"

    @property
    def description(self) -> str:
        return (
            "Evaluates LLM coding agents on real GitHub issues. "
            "Agents produce git patches verified by each repository's test suite."
        )

    @property
    def timeout_seconds(self) -> float:
        return 14400.0  # 4 hours

    @property
    def required_extras(self) -> tuple[str, ...]:
        return ("swebench",)

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "dataset": ("Dataset: 'lite', 'verified', 'full', or a HuggingFace path", "lite"),
            "split": ("Dataset split", "test"),
            "instance_filter": ("Regex to filter instance IDs (e.g. 'django')", None),
            "instance_slice": ("Slice spec to run a subset (e.g. '0:50')", None),
            "workers": ("Parallel Docker workers for patch generation", 4),
            "max_workers_eval": ("Parallel workers for the SWE-bench scoring harness", 4),
            "temperature": ("Sampling temperature", 0.0),
            "max_turns": ("Max agent turns per instance", 30),
            "use_modal": ("Run scoring harness on Modal cloud", False),
        }

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        start_time = time.time()
        swe_args = SWEBenchArgs.from_dict(args)

        if swe_args.use_modal:
            missing = []
            if not os.environ.get("MODAL_TOKEN_ID"):
                missing.append("MODAL_TOKEN_ID")
            if not os.environ.get("MODAL_TOKEN_SECRET"):
                missing.append("MODAL_TOKEN_SECRET")
            if missing:
                return self._error_result(
                    f"Modal credentials missing: {', '.join(missing)}",
                    start_time,
                )

        tmp_dir = None
        if output_dir is None:
            tmp_dir = tempfile.mkdtemp()
            work_dir = Path(tmp_dir)
        else:
            work_dir = Path(output_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        preds_path = work_dir / "preds.json"
        run_id = f"swe_bench_{uuid.uuid4().hex[:8]}"

        provider_url = getattr(provider, "base_url", None) or "http://localhost:8000/v1"
        model_name = provider.model_name
        is_local = self._is_local_provider(provider, provider_url)

        try:
            # Phase 1: generate patches with mini-swe-agent
            gen_ok, gen_output = await self._run_agent(
                provider_url, model_name, is_local, swe_args, work_dir, container_runtime
            )
            logger.info(f"[{self.name}] Patch generation exit: {'ok' if gen_ok else 'failed'}")

            if not gen_ok or not preds_path.exists():
                return self._error_result(
                    "mini-swe-agent produced no preds.json", start_time, gen_output
                )

            # Phase 2: score patches with the SWE-bench harness
            score_ok, score_output = await self._run_scoring(
                swe_args, preds_path, run_id, work_dir, container_runtime
            )
            logger.info(f"[{self.name}] Scoring exit: {'ok' if score_ok else 'failed'}")

            all_output = gen_output + "\n" + score_output
            result = self._parse_results(work_dir, run_id, score_ok, all_output, start_time)

        except Exception as e:
            logger.exception(f"[{self.name}] Execution failed")
            return self._error_result(str(e), start_time)
        finally:
            if tmp_dir is not None:
                import shutil

                shutil.rmtree(tmp_dir, ignore_errors=True)

        result.duration_seconds = time.time() - start_time
        if output_dir:
            self._save_results(result, output_dir)
        return result

    async def _run_agent(
        self,
        provider_url: str,
        model_name: str,
        is_local: bool,
        swe_args: SWEBenchArgs,
        work_dir: Path,
        container_runtime: str,
    ) -> tuple[bool, str]:
        """Run mini-swe-agent to generate patches. Returns (success, output)."""
        # For local vLLM servers, use the hosted_vllm/ litellm prefix with api_base config.
        # For external APIs (OpenAI etc.), pass the model name as-is.
        litellm_model = f"hosted_vllm/{model_name}" if is_local else model_name

        cmd = [
            sys.executable,
            "-m",
            "minisweagent.run.benchmarks.swebench",
            "--subset",
            swe_args.dataset,
            "--split",
            swe_args.split,
            "--output",
            str(work_dir),
            "--workers",
            str(swe_args.workers),
            "--model",
            litellm_model,
            # Explicitly include the default swebench.yaml so our overrides are merged on top
            "-c",
            "swebench.yaml",
            "-c",
            f"agent.max_iterations={swe_args.max_turns}",
            "-c",
            f"model.model_kwargs.temperature={swe_args.temperature}",
        ]
        if is_local:
            cmd += ["-c", f"model.api_base={provider_url}"]
            cmd += ["-c", "model.api_key=local"]
        if swe_args.instance_filter:
            cmd += ["--filter", swe_args.instance_filter]
        if swe_args.instance_slice:
            cmd += ["--slice", swe_args.instance_slice]

        # mini-swe-agent respects MSWEA_DOCKER_EXECUTABLE to switch container runtimes
        env = os.environ.copy()
        env["MSWEA_DOCKER_EXECUTABLE"] = container_runtime

        logger.info(f"[{self.name}] Running mini-swe-agent: {shlex.join(cmd)}")
        # Reserve 20% of total timeout for the scoring phase
        return await self._run_subprocess(cmd, timeout=self.timeout_seconds * 0.8, env=env)

    async def _run_scoring(
        self,
        swe_args: SWEBenchArgs,
        preds_path: Path,
        run_id: str,
        work_dir: Path,
        container_runtime: str,
    ) -> tuple[bool, str]:
        """Run the SWE-bench evaluation harness to score patches."""
        dataset_hf = _DATASET_HF_PATHS.get(swe_args.dataset, swe_args.dataset)

        cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset_hf,
            "--predictions_path",
            str(preds_path),
            "--run_id",
            run_id,
        ]

        env = os.environ.copy()

        cmd.extend(["--max_workers", str(swe_args.max_workers_eval)])

        if swe_args.use_modal:
            # Modal handles containers in the cloud
            cmd.extend(["--modal", "true"])
        elif "DOCKER_HOST" not in env:
            # Use podman service socket (started by Beaker launcher)
            # DOCKER_HOST should already be set by _build_install_cmd
            logger.warning("DOCKER_HOST not set. Scoring may fail without podman service or Modal.")

        logger.info(f"[{self.name}] Running SWE-bench harness: {shlex.join(cmd)}")
        ok, output = await self._run_subprocess(
            cmd, timeout=self.timeout_seconds * 0.2, cwd=str(work_dir), env=env
        )
        if not ok:
            logger.warning(f"[{self.name}] Scoring subprocess output:\n{output}")
        return ok, output

    async def _run_subprocess(
        self,
        cmd: list[str],
        timeout: float,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        """Run a subprocess and return (success, combined stdout+stderr)."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=env if env is not None else os.environ.copy(),
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return False, f"[Subprocess timed out after {timeout:.0f}s]"

        output = stdout.decode(errors="replace")
        return proc.returncode == 0, output

    def _parse_results(
        self,
        work_dir: Path,
        run_id: str,
        score_ok: bool,
        raw_output: str,
        start_time: float,
    ) -> ExternalEvalResult:
        """Parse the SWE-bench harness results JSON into an ExternalEvalResult."""
        # SWE-bench writes <run_id>.json to the working directory; some versions
        # use an evaluation_results/ subdirectory instead.
        candidates = [
            work_dir / f"{run_id}.json",
            work_dir / "evaluation_results" / f"{run_id}.json",
        ]
        results_file = next((p for p in candidates if p.exists()), None)

        if results_file is None:
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error="No evaluation results file found after scoring",
                raw_output=raw_output,
                duration_seconds=time.time() - start_time,
            )

        try:
            data = json.loads(results_file.read_text())
        except json.JSONDecodeError as e:
            return ExternalEvalResult(
                name=self.name,
                success=False,
                error=f"Failed to parse results JSON: {e}",
                raw_output=raw_output,
                duration_seconds=time.time() - start_time,
            )

        resolved = data.get("resolved_instances", data.get("resolved", []))
        unresolved = data.get("unresolved_instances", data.get("unresolved", []))
        total = len(resolved) + len(unresolved)
        resolve_rate = len(resolved) / total if total > 0 else 0.0

        predictions = [
            {"native_id": iid, "instance_metrics": {"resolved": {"external": 1.0}}}
            for iid in resolved
        ] + [
            {"native_id": iid, "instance_metrics": {"resolved": {"external": 0.0}}}
            for iid in unresolved
        ]

        return ExternalEvalResult(
            name=self.name,
            success=score_ok and total > 0,
            metrics={
                "resolve_rate": resolve_rate,
                "resolved": float(len(resolved)),
                "total": float(total),
            },
            metadata={"run_id": run_id, "resolved_instances": resolved},
            raw_output=raw_output,
            predictions=predictions if predictions else None,
        )
