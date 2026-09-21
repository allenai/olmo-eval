"""Official OmniDocBench end-to-end evaluation, run out of process.

OmniDocBench (https://github.com/opendatalab/OmniDocBench) grades a page by parsing the
predicted markdown into text blocks, display formulas and tables, matching them against the
annotated blocks, and measuring text edit distance, table TEDS, formula CDM and
reading-order edit distance. The matching alone is several thousand lines and changes
between benchmark versions, so rather than port it this module runs the pinned official
evaluator: its scores are the leaderboard's by construction.

The evaluator is not importable as a library — it installs a top-level package named
``src``, pins old numpy/scipy/pandas and writes to ``./result`` — so it runs as a
subprocess in a virtualenv of its own, provisioned on first use under
``$OMNIDOCBENCH_EVAL_DIR`` (default ``~/.cache/olmo_eval/omnidocbench``). Set
``OMNIDOCBENCH_EVAL_REPO`` and ``OMNIDOCBENCH_EVAL_PYTHON`` to use an existing checkout and
interpreter instead.

Evaluation is a dataset-level step: every metric is a mean over pages, and tables/formulas
are zero-filled for annotated pages that produced no scored sample. One call to
:func:`run_official_evaluation` scores all pages and returns both the evaluator's own
summary and the per-page values, which the task stores on each response.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, Response

logger = logging.getLogger(__name__)

EVAL_REPO_URL = "https://github.com/opendatalab/OmniDocBench.git"
#: Scoring code of the v1.6 release (later commits on ``main`` touch only docs and tools).
EVAL_REPO_COMMIT = "f133a71e9e91c3621c7ce8994200a7b394a06eb3"
EVAL_PYTHON_VERSION = "3.11"
#: The evaluator forks scoring workers from threads, which newer ``filelock`` releases
#: break; the failure is swallowed and the sample scored zero.
EVAL_EXTRA_REQUIREMENTS = ("filelock==3.16.1",)

RESULT_KEY = "omnidocbench_result"

#: Binaries the CDM formula metric shells out to (LaTeX, ImageMagick 7, Ghostscript).
CDM_BINARIES = ("pdflatex", "magick", "gs")

_PRED_DIR_NAME = "pred"
_MATCH_METHOD = "quick_match"
_SAMPLE_KEY_RE = re.compile(r"^(?P<image>.*)_\[[^\]]*\]$")


def cdm_available() -> bool:
    return all(shutil.which(binary) for binary in CDM_BINARIES)


def missing_cdm_binaries() -> list[str]:
    return [binary for binary in CDM_BINARIES if not shutil.which(binary)]


# ---------------------------------------------------------------------------
# Evaluator provisioning
# ---------------------------------------------------------------------------


def _run(cmd: Sequence[str], **kwargs: Any) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(list(cmd), check=True, **kwargs)


def ensure_evaluator() -> tuple[Path, Path]:
    """Return ``(repo_dir, python)`` for the official evaluator, provisioning it if needed."""
    repo_env, python_env = (
        os.environ.get("OMNIDOCBENCH_EVAL_REPO"),
        os.environ.get("OMNIDOCBENCH_EVAL_PYTHON"),
    )
    if repo_env and python_env:
        return Path(repo_env), Path(python_env)

    root = Path(
        os.environ.get("OMNIDOCBENCH_EVAL_DIR")
        or Path.home() / ".cache" / "olmo_eval" / "omnidocbench"
    )
    home = root / EVAL_REPO_COMMIT[:12]
    repo, venv, ready = home / "repo", home / "venv", home / ".ready"
    python = venv / "bin" / "python"
    if ready.exists():
        return repo, python

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "Provisioning the OmniDocBench evaluator needs `uv` on PATH (or point "
            "OMNIDOCBENCH_EVAL_REPO / OMNIDOCBENCH_EVAL_PYTHON at an existing install)."
        )
    from filelock import FileLock

    home.mkdir(parents=True, exist_ok=True)
    with FileLock(str(home / ".lock")):
        if ready.exists():
            return repo, python
        shutil.rmtree(repo, ignore_errors=True)
        shutil.rmtree(venv, ignore_errors=True)
        _run(["git", "clone", "--quiet", EVAL_REPO_URL, str(repo)])
        _run(["git", "-C", str(repo), "checkout", "--quiet", EVAL_REPO_COMMIT])
        _run([uv, "venv", "--quiet", "--python", EVAL_PYTHON_VERSION, str(venv)])
        _run(
            [uv, "pip", "install", "--quiet", "--python", str(python), "-e", str(repo)]
            + list(EVAL_EXTRA_REQUIREMENTS)
        )
        ready.touch()
    return repo, python


# ---------------------------------------------------------------------------
# Running the evaluator
# ---------------------------------------------------------------------------


@dataclass
class OfficialEvaluation:
    """Outcome of one official end-to-end evaluation."""

    #: ``image_name -> {text_edit, read_order_edit, formula_edit, table_teds,
    #: table_teds_s, formula_cdm}``; a key is absent when the page has no such sample.
    per_page: dict[str, dict[str, float]] = field(default_factory=dict)
    #: The evaluator's own leaderboard summary (``notebook_metric_summary``).
    summary: dict[str, Any] = field(default_factory=dict)
    #: Samples the evaluator scored zero because its scoring code crashed or timed out.
    num_scorer_failures: int = 0


def _config(gt_path: Path, pred_dir: Path, *, with_cdm: bool, workers: int) -> dict[str, Any]:
    formula_metrics = ["Edit_dist", "CDM"] if with_cdm else ["Edit_dist"]
    return {
        "end2end_eval": {
            "metrics": {
                "text_block": {"metric": ["Edit_dist"]},
                "display_formula": {"metric": formula_metrics, "cdm_workers": workers},
                "table": {"metric": ["TEDS", "Edit_dist"], "teds_workers": workers},
                "reading_order": {"metric": ["Edit_dist"]},
            },
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": str(gt_path)},
                "prediction": {"data_path": str(pred_dir)},
                "match_method": _MATCH_METHOD,
                "match_workers": workers,
                "quick_match_truncated_timeout_sec": 300,
                "match_timeout_sec": 420,
                "timeout_fallback_max_chunk_span": 10,
                "timeout_fallback_order_penalty": 0.10,
            },
        }
    }


def _load(result_dir: Path, name: str) -> Any:
    path = result_dir / f"{_PRED_DIR_NAME}_{_MATCH_METHOD}_{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _page_means(per_sample: Mapping[str, Any] | None, field_name: str | None) -> dict[str, float]:
    """Mean per page of a ``"<image>_[gt idx]" -> score`` map."""
    by_page: dict[str, list[float]] = {}
    for key, value in (per_sample or {}).items():
        match = _SAMPLE_KEY_RE.match(key)
        score = value[field_name] if field_name is not None else value
        if match is not None and isinstance(score, (int, float)):
            by_page.setdefault(match.group("image"), []).append(float(score))
    return {image: sum(scores) / len(scores) for image, scores in by_page.items()}


def _count_failures(metric_result: Mapping[str, Any]) -> int:
    count = 0
    for element in ("table", "display_formula"):
        for debug in ((metric_result.get(element) or {}).get("metric_debug") or {}).values():
            for counter in ("error_case_count", "timeout_case_count", "exception_case_count"):
                count += int(debug.get(counter) or 0)
    fallbacks = (metric_result.get("match_debug") or {}).get("text_match_fallback_counts") or {}
    return count + sum(int(v or 0) for v in fallbacks.values())


def run_official_evaluation(
    predictions: Mapping[str, str],
    gt_pages: Sequence[Mapping[str, Any]],
    *,
    with_cdm: bool,
    workers: int | None = None,
) -> OfficialEvaluation:
    """Score ``predictions`` (``image_name -> markdown``) against their annotated pages."""
    import yaml

    repo, python = ensure_evaluator()
    if workers is None:
        workers = int(os.environ.get("OMNIDOCBENCH_EVAL_WORKERS") or 0) or max(
            1, min(16, (os.cpu_count() or 4) // 4)
        )

    with tempfile.TemporaryDirectory(prefix="omnidocbench_eval_") as tmp:
        work = Path(tmp)
        pred_dir = work / _PRED_DIR_NAME
        pred_dir.mkdir()
        for image_name, markdown in predictions.items():
            # The evaluator's first lookup for image "X.ext" is "X.md".
            (pred_dir / f"{image_name[:-4]}.md").write_text(markdown, encoding="utf-8")
        gt_path = work / "gt.json"
        gt_path.write_text(json.dumps(list(gt_pages), ensure_ascii=False), encoding="utf-8")
        config_path = work / "end2end.yaml"
        config_path.write_text(
            yaml.safe_dump(_config(gt_path, pred_dir, with_cdm=with_cdm, workers=workers))
        )

        log_path = work / "evaluator.log"
        with open(log_path, "w") as log:
            proc = subprocess.run(
                [str(python), str(repo / "pdf_validation.py"), "--config", str(config_path)],
                cwd=work,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        result_dir = work / "result"
        summary = (_load(result_dir, "run_summary") or {}).get("notebook_metric_summary")
        if proc.returncode != 0 or not summary:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
            raise RuntimeError(
                f"OmniDocBench evaluator failed (exit {proc.returncode}). Log tail:\n{tail}"
            )

        per_page: dict[str, dict[str, float]] = {}
        page_values = {
            "text_edit": _load(result_dir, "text_block_per_page_edit") or {},
            "read_order_edit": _load(result_dir, "reading_order_per_page_edit") or {},
            "formula_edit": _load(result_dir, "display_formula_per_page_edit") or {},
            "table_teds": _page_means(_load(result_dir, "table_per_table_TEDS"), "TEDS"),
            "table_teds_s": _page_means(
                _load(result_dir, "table_per_table_TEDS"), "TEDS_structure_only"
            ),
            "formula_cdm": _page_means(_load(result_dir, "display_formula_per_sample_CDM"), None),
        }
        for name, values in page_values.items():
            for image_name, value in values.items():
                if isinstance(value, (int, float)):
                    per_page.setdefault(image_name, {})[name] = float(value)

        failures = _count_failures(_load(result_dir, "metric_result") or {})
        if failures:
            logger.error(
                "OmniDocBench evaluator scored %d sample(s) zero after an internal crash, "
                "timeout or matching fallback; see n_scorer_failures.",
                failures,
            )
        return OfficialEvaluation(per_page=per_page, summary=summary, num_scorer_failures=failures)


# ---------------------------------------------------------------------------
# Scorer + metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OmniDocBenchScorer(Scorer):
    """Names the OmniDocBench score channel.

    Pages are graded together by :func:`run_official_evaluation`, which stores each page's
    results on ``output.metadata["omnidocbench_result"]``; the per-response score is the
    page's text accuracy (``1 - text edit distance``).
    """

    name: str = "omnidocbench"

    def score(self, instance: Instance, output: LMOutput) -> float:
        result = (output.metadata or {}).get(RESULT_KEY) or {}
        text_edit = result.get("text_edit")
        return 1.0 - text_edit if text_edit is not None else 0.0


def _page_results(responses: Sequence[Response]) -> Iterator[tuple[Response, dict[str, Any]]]:
    for response in responses:
        if response.outputs:
            result = (response.outputs[0].metadata or {}).get(RESULT_KEY)
            if result is not None:
                yield response, result


@dataclass(frozen=True)
class OmniDocBenchPageMetric(Metric):
    """Page average of one evaluator component, optionally within a page-attribute slice.

    ``zero_fill`` names the instance-metadata flag marking pages annotated with the element
    (``has_table`` / ``has_formula``): such a page counts as zero when no sample was scored
    for it, as the evaluator does. Without it, only pages that have a value are averaged.
    ``scale`` converts to the leaderboard's unit (x100 for TEDS and CDM).
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    component: str = ""
    zero_fill: str | None = None
    scale: float = 1.0
    attribute: str | None = None
    attribute_value: str | None = None

    def values(self, responses: Sequence[Response]) -> list[float]:
        vals: list[float] = []
        for response, result in _page_results(responses):
            meta = response.instance.metadata
            if self.attribute is not None and meta.get(self.attribute) != self.attribute_value:
                continue
            value = result.get(self.component)
            if value is None and self.zero_fill is not None and meta.get(self.zero_fill):
                value = 0.0
            if value is not None:
                vals.append(float(value))
        return vals

    def compute(self, responses: Sequence[Response]) -> float:
        vals = self.values(responses)
        return sum(vals) / len(vals) * self.scale if vals else 0.0


@dataclass(frozen=True)
class OmniDocBenchOverallMetric(Metric):
    """The leaderboard Overall: ``((1 - TextEdit) * 100 + TableTEDS + FormulaCDM) / 3``."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        text = OmniDocBenchPageMetric("", self.scorer, "text_edit").compute(responses)
        teds = OmniDocBenchPageMetric(
            "", self.scorer, "table_teds", zero_fill="has_table", scale=100.0
        ).compute(responses)
        cdm = OmniDocBenchPageMetric(
            "", self.scorer, "formula_cdm", zero_fill="has_formula", scale=100.0
        ).compute(responses)
        return ((1.0 - text) * 100.0 + teds + cdm) / 3.0


@dataclass(frozen=True)
class OmniDocBenchTextScoreMetric(Metric):
    """``(1 - TextEdit) * 100`` — the text term of the leaderboard Overall."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        text = OmniDocBenchPageMetric("", self.scorer, "text_edit").compute(responses)
        return (1.0 - text) * 100.0


@dataclass(frozen=True)
class OmniDocBenchScorerFailuresMetric(Metric):
    """Samples the official evaluator zeroed after an internal crash, timeout or fallback."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        for _, result in _page_results(responses):
            return float(result.get("num_scorer_failures", 0))
        return 0.0
