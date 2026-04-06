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

_INSTANCE_REPORT_KEYS = frozenset(
    {
        "patch_is_None",
        "patch_exists",
        "patch_successfully_applied",
        "resolved",
        "tests_status",
    }
)

# Maps short aliases to HuggingFace dataset paths used by the SWE-bench harness.
# mini-swe-agent has its own alias table; we pass aliases through to it directly.
_DATASET_HF_PATHS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}


def _ensure_modal_config_exists() -> None:
    """Create empty ~/.modal.toml if missing.

    SWE-bench's validate_modal_credentials() checks for this file's existence,
    but Modal SDK uses MODAL_TOKEN_ID/MODAL_TOKEN_SECRET env vars for auth
    (which take precedence over the file). We create an empty file to satisfy
    the validation while keeping credentials in env vars only.
    """
    modal_toml = Path.home() / ".modal.toml"
    if not modal_toml.exists():
        modal_toml.touch()
        logger.info("Created empty ~/.modal.toml to satisfy SWE-bench validation")


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

        # Set up real-time log files if output_dir is provided
        logs_dir = work_dir / "logs" if output_dir else None
        agent_log = logs_dir / "mini_swe_agent.log" if logs_dir else None
        scoring_log = logs_dir / "swebench_scoring.log" if logs_dir else None

        try:
            # Phase 1: generate patches with mini-swe-agent
            gen_ok, gen_output = await self._run_agent(
                provider_url,
                model_name,
                is_local,
                swe_args,
                work_dir,
                container_runtime,
                log_file=agent_log,
            )
            logger.info(f"[{self.name}] Patch generation exit: {'ok' if gen_ok else 'failed'}")

            if not gen_ok or not preds_path.exists():
                return self._error_result(
                    "mini-swe-agent produced no preds.json", start_time, gen_output
                )

            # Phase 2: score patches with the SWE-bench harness
            score_ok, score_output = await self._run_scoring(
                swe_args,
                preds_path,
                run_id,
                work_dir,
                container_runtime,
                log_file=scoring_log,
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
        log_file: Path | None = None,
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
            f"agent.step_limit={swe_args.max_turns}",
            "-c",
            f"model.model_kwargs.temperature={swe_args.temperature}",
        ]
        if is_local:
            cmd += ["-c", f"model.api_base={provider_url}"]
        if swe_args.instance_filter:
            cmd += ["--filter", swe_args.instance_filter]
        if swe_args.instance_slice:
            cmd += ["--slice", swe_args.instance_slice]

        # mini-swe-agent respects MSWEA_DOCKER_EXECUTABLE to switch container runtimes
        env = os.environ.copy()
        env["MSWEA_DOCKER_EXECUTABLE"] = container_runtime
        if is_local:
            env["HOSTED_VLLM_API_KEY"] = "local"
            env["HOSTED_VLLM_API_BASE"] = provider_url
            env["MSWEA_COST_TRACKING"] = "ignore_errors"

        logger.info(f"[{self.name}] Running mini-swe-agent: {shlex.join(cmd)}")
        # Reserve 20% of total timeout for the scoring phase
        return await self._run_subprocess(
            cmd, timeout=self.timeout_seconds * 0.8, env=env, log_file=log_file
        )

    async def _run_scoring(
        self,
        swe_args: SWEBenchArgs,
        preds_path: Path,
        run_id: str,
        work_dir: Path,
        container_runtime: str,
        log_file: Path | None = None,
    ) -> tuple[bool, str]:
        """Run the SWE-bench evaluation harness to score patches."""
        if swe_args.use_modal:
            _ensure_modal_config_exists()

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
            cmd.extend(["--modal", "true"])
        elif "DOCKER_HOST" not in env:
            logger.warning("DOCKER_HOST not set. Scoring may fail without podman service or Modal.")

        logger.info(f"[{self.name}] Running SWE-bench harness: {shlex.join(cmd)}")
        ok, output = await self._run_subprocess(
            cmd,
            timeout=self.timeout_seconds * 0.2,
            cwd=str(work_dir),
            env=env,
            log_file=log_file,
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
        log_file: Path | None = None,
    ) -> tuple[bool, str]:
        """Run a subprocess and return (success, combined stdout+stderr).

        If log_file is provided, output is streamed to the file in real-time.
        """
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=env if env is not None else os.environ.copy(),
        )

        output_chunks: list[str] = []
        log_handle = None

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_file.open("w")

        try:

            async def stream_output() -> None:
                assert proc.stdout is not None
                while True:
                    chunk = await proc.stdout.read(4096)
                    if not chunk:
                        break
                    decoded = chunk.decode(errors="replace")
                    output_chunks.append(decoded)
                    if log_handle:
                        log_handle.write(decoded)
                        log_handle.flush()

            await asyncio.wait_for(stream_output(), timeout=timeout)
            await proc.wait()
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            if log_handle:
                log_handle.write(f"\n[Subprocess timed out after {timeout:.0f}s]\n")
            return False, f"[Subprocess timed out after {timeout:.0f}s]"
        finally:
            if log_handle:
                log_handle.close()

        return proc.returncode == 0, "".join(output_chunks)

    def _parse_results(
        self,
        work_dir: Path,
        run_id: str,
        score_ok: bool,
        raw_output: str,
        start_time: float,
    ) -> ExternalEvalResult:
        """Parse the SWE-bench harness results JSON into an ExternalEvalResult."""
        # SWE-bench writes results with pattern {model_name}.{run_id}.json where
        # model slashes become double underscores. Search for any file ending
        # in the run_id pattern.
        pattern = f"*{run_id}.json"
        matches = list(work_dir.glob(pattern))
        # Also check evaluation_results/ subdirectory (older harness versions)
        matches.extend(work_dir.glob(f"evaluation_results/{pattern}"))
        results_file = matches[0] if matches else None

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

        resolved_ids = self._coerce_instance_ids(data.get("resolved_ids", []))
        unresolved_ids = self._coerce_instance_ids(data.get("unresolved_ids", []))
        submitted_ids = self._coerce_instance_ids(data.get("submitted_ids", []))
        instance_reports = self._find_instance_reports(work_dir, results_file)
        exit_statuses = self._find_exit_statuses(work_dir)
        predictions = self._build_predictions(
            submitted_ids=submitted_ids,
            resolved_ids=resolved_ids,
            unresolved_ids=unresolved_ids,
            instance_reports=instance_reports,
            exit_statuses=exit_statuses,
        )

        if predictions:
            resolved_instances = [
                pred["native_id"]
                for pred in predictions
                if pred["instance_metrics"]["resolved"]["external"] == 1.0
            ]
            resolved_count = len(resolved_instances)
            total = len(predictions)
        else:
            resolved_count = self._read_int(
                data.get("resolved"),
                data.get("resolved_instances"),
                len(resolved_ids),
            )
            unresolved_count = self._read_int(
                data.get("unresolved"),
                data.get("unresolved_instances"),
                len(unresolved_ids),
            )
            submitted_count = self._read_int(
                data.get("submitted_instances"),
                len(submitted_ids),
            )
            total = submitted_count if submitted_count > 0 else resolved_count + unresolved_count
            resolved_instances = resolved_ids

        resolve_rate = resolved_count / total if total > 0 else 0.0
        metadata: dict[str, Any] = {
            "run_id": run_id,
            "resolved_instances": resolved_instances,
        }
        if exit_statuses:
            metadata["instance_exit_statuses"] = exit_statuses

        return ExternalEvalResult(
            name=self.name,
            success=score_ok and total > 0,
            metrics={
                "resolve_rate": resolve_rate,
                "resolved": float(resolved_count),
                "total": float(total),
            },
            metadata=metadata,
            raw_output=raw_output,
            predictions=predictions if predictions else None,
        )

    def _build_predictions(
        self,
        submitted_ids: list[str],
        resolved_ids: list[str],
        unresolved_ids: list[str],
        instance_reports: dict[str, dict[str, Any]],
        exit_statuses: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Build per-instance predictions from any available SWE-bench artifacts."""
        instance_ids = self._ordered_unique_ids(
            resolved_ids,
            unresolved_ids,
            submitted_ids,
            instance_reports.keys(),
            exit_statuses.keys(),
        )
        return [
            {
                "native_id": instance_id,
                "instance_metrics": {
                    "resolved": {
                        "external": self._resolve_prediction_score(
                            instance_id,
                            resolved_ids=resolved_ids,
                            unresolved_ids=unresolved_ids,
                            instance_reports=instance_reports,
                        )
                    }
                },
            }
            for instance_id in instance_ids
        ]

    def _coerce_instance_ids(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if item is not None]

    def _ordered_unique_ids(self, *groups: Any) -> list[str]:
        ordered_ids: list[str] = []
        seen: set[str] = set()

        for group in groups:
            for instance_id in group:
                if instance_id in seen:
                    continue
                seen.add(instance_id)
                ordered_ids.append(str(instance_id))

        return ordered_ids

    def _resolve_prediction_score(
        self,
        instance_id: str,
        resolved_ids: list[str],
        unresolved_ids: list[str],
        instance_reports: dict[str, dict[str, Any]],
    ) -> float:
        report_resolved = instance_reports.get(instance_id, {}).get("resolved")
        if isinstance(report_resolved, bool):
            return 1.0 if report_resolved else 0.0

        if instance_id in resolved_ids:
            return 1.0

        if instance_id in unresolved_ids:
            return 0.0

        return 0.0

    def _read_int(self, *values: Any) -> int:
        for value in values:
            try:
                if value is None:
                    continue
                return int(value)
            except (TypeError, ValueError):
                continue
        return 0

    def _find_instance_reports(
        self, work_dir: Path, results_file: Path
    ) -> dict[str, dict[str, Any]]:
        """Locate a per-instance report JSON if the harness emitted one."""
        for candidate in sorted(work_dir.rglob("*.json")):
            if candidate == results_file:
                continue
            payload = self._load_json(candidate)
            if payload is None:
                continue
            reports = self._extract_instance_reports(payload)
            if reports:
                return reports
        return {}

    def _load_json(self, path: Path) -> Any | None:
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _extract_instance_reports(self, payload: Any) -> dict[str, dict[str, Any]]:
        if isinstance(payload, dict):
            if (
                payload
                and all(isinstance(v, dict) for v in payload.values())
                and any(self._looks_like_instance_report(v) for v in payload.values())
            ):
                return {str(k): v for k, v in payload.items() if isinstance(v, dict)}

            for value in payload.values():
                nested = self._extract_instance_reports(value)
                if nested:
                    return nested

        if isinstance(payload, list):
            for value in payload:
                nested = self._extract_instance_reports(value)
                if nested:
                    return nested

        return {}

    def _looks_like_instance_report(self, value: dict[str, Any]) -> bool:
        return bool(_INSTANCE_REPORT_KEYS.intersection(value))

    def _find_exit_statuses(self, work_dir: Path) -> dict[str, str]:
        """Locate and parse instances_by_exit_status if present."""
        for candidate in sorted(work_dir.rglob("*")):
            if not candidate.is_file():
                continue
            try:
                if candidate.suffix == ".json":
                    payload = self._load_json(candidate)
                    statuses = self._extract_exit_statuses_from_json(payload)
                    if statuses:
                        return statuses

                text = candidate.read_text(errors="ignore")
            except OSError:
                continue

            statuses = self._parse_instances_by_exit_status(text)
            if statuses:
                return statuses

        return {}

    def _extract_exit_statuses_from_json(self, payload: Any) -> dict[str, str]:
        if isinstance(payload, dict):
            grouped = payload.get("instances_by_exit_status")
            if isinstance(grouped, dict):
                return self._coerce_exit_status_mapping(grouped)

            for value in payload.values():
                nested = self._extract_exit_statuses_from_json(value)
                if nested:
                    return nested

        if isinstance(payload, list):
            for value in payload:
                nested = self._extract_exit_statuses_from_json(value)
                if nested:
                    return nested

        return {}

    def _coerce_exit_status_mapping(self, grouped: dict[str, Any]) -> dict[str, str]:
        statuses: dict[str, str] = {}
        for status, instance_ids in grouped.items():
            if not isinstance(instance_ids, list):
                continue
            for instance_id in instance_ids:
                if instance_id is not None:
                    statuses[str(instance_id)] = str(status)
        return statuses

    def _parse_instances_by_exit_status(self, text: str) -> dict[str, str]:
        marker = "instances_by_exit_status:"
        if marker not in text:
            return {}

        statuses: dict[str, str] = {}
        in_section = False
        section_indent = 0
        current_status: str | None = None

        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not in_section:
                if stripped == marker:
                    in_section = True
                    section_indent = len(raw_line) - len(raw_line.lstrip())
                continue

            if not stripped:
                continue

            indent = len(raw_line) - len(raw_line.lstrip())
            if indent <= section_indent and not stripped.startswith("-"):
                break

            if stripped.endswith(":") and not stripped.startswith("-"):
                current_status = stripped[:-1]
                continue

            if stripped.startswith("-") and current_status is not None:
                instance_id = stripped[1:].strip()
                if instance_id:
                    statuses[instance_id] = current_status

        return statuses
