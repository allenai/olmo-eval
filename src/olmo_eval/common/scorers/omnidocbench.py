"""Official OmniDocBench end-to-end evaluation, run out of process.

OmniDocBench (https://github.com/opendatalab/OmniDocBench) grades a page by parsing the
predicted markdown into text blocks, display formulas and tables, matching them against the
annotated blocks, and measuring text edit distance, table TEDS, formula CDM and
reading-order edit distance. The matching alone is several thousand lines and changes
between benchmark versions, so rather than port it this module runs the pinned official
evaluator of each version: its scores are that leaderboard's by construction.

The evaluator is not importable as a library — it installs a top-level package named
``src`` (v1.6) or none at all (v1.5), pins old numpy/scipy/pandas and writes to
``./result`` — so each version runs as a subprocess in a virtualenv of its own,
provisioned on first use under ``$OMNIDOCBENCH_EVAL_DIR`` (default
``~/.cache/olmo_eval/omnidocbench``). Set ``OMNIDOCBENCH_EVAL_REPO`` and
``OMNIDOCBENCH_EVAL_PYTHON`` to use an existing checkout and interpreter instead.

Evaluation is a dataset-level step: every metric is a mean over pages. One call to
:func:`run_official_evaluation` scores all pages and returns both the evaluator's own
leaderboard numbers and the per-page values, which the task stores on each response.
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

RESULT_KEY = "omnidocbench_result"

#: Binaries the CDM formula metric shells out to (LaTeX, ImageMagick 7, Ghostscript).
CDM_BINARIES = ("pdflatex", "magick", "gs")

_PRED_DIR_NAME = "pred"
_MATCH_METHOD = "quick_match"
_SAMPLE_KEY_RE = re.compile(r"^(?P<image>.*)_\[[^\]]*\]$")


@dataclass(frozen=True)
class EvaluatorVersion:
    """One release of the official evaluator and how to stand it up."""

    name: str
    repo_commit: str
    python_version: str
    #: ``uv pip install`` arguments; ``{repo}`` is the checkout.
    install: tuple[str, ...]
    #: Whether pages annotated with a table / display formula but left without a scored
    #: sample count as zero in the page average (v1.6 and later).
    zero_fills_missing_pages: bool
    #: Extra binaries CDM needs beyond :data:`CDM_BINARIES`.
    cdm_binaries: tuple[str, ...] = ()
    #: ``(installed, replacement)`` pairs swapped after installing, for dependencies that
    #: pull a variant the evaluation host cannot load.
    replacements: tuple[tuple[str, str], ...] = ()
    #: Modules the evaluator imports at startup, checked right after provisioning so a
    #: broken environment fails before inference rather than after it.
    startup_imports: tuple[str, ...] = ()


EVALUATORS: dict[str, EvaluatorVersion] = {
    # Scoring code of the v1.6 release (later commits on ``main`` touch only docs and tools).
    # The evaluator forks scoring workers from threads, which newer ``filelock`` releases
    # break; the failure is swallowed and the sample scored zero, hence the pin.
    "v1.6": EvaluatorVersion(
        name="v1.6",
        repo_commit="f133a71e9e91c3621c7ce8994200a7b394a06eb3",
        python_version="3.11",
        install=("-e", "{repo}", "filelock==3.16.1"),
        zero_fills_missing_pages=True,
        startup_imports=("src.cli",),
    ),
    # Head of the ``v1_5`` branch. Its ``requirements.txt`` pins a whole notebook
    # environment; these are the packages the evaluator imports, at those pins.
    "v1.5": EvaluatorVersion(
        name="v1.5",
        repo_commit="59b103c4b47d3a01fada83491585d6512a40c0bc",
        python_version="3.10",
        install=(
            "apted==1.0.3",
            "beautifulsoup4==4.11.1",
            "datasets==3.1.0",
            "evaluate==0.4.3",
            "filelock==3.16.1",
            "func-timeout==4.3.5",
            "Levenshtein==0.25.1",
            "loguru==0.7.2",
            "lxml==4.9.1",
            "matplotlib==3.7.5",
            "mmeval==0.2.1",
            "nltk==3.9.1",
            "numpy==1.24.4",
            "pandas==2.0.3",
            "pillow==10.4.0",
            "pycocotools==2.0.7",
            "pylatexenc==3.0a30",
            "PyYAML==6.0.2",
            "rapidfuzz==3.9.7",
            # CDM's RANSAC; imported lazily inside a bare ``except`` that turns a missing
            # package into a silent zero, and pinned by ``metrics/cdm/requirements.txt``.
            "scikit-image==0.20.0",
            "scipy==1.10.1",
            "tabulate==0.9.0",
            "tqdm==4.67.1",
        ),
        zero_fills_missing_pages=False,
        # v1.5 tokenizes formulas with the KaTeX parser through Node.js.
        cdm_binaries=("node",),
        # ``mmeval`` requires the GUI OpenCV build, which needs ``libGL`` at import time;
        # the headless build provides the same ``cv2`` without it.
        replacements=(("opencv-python", "opencv-python-headless==4.11.0.86"),),
        startup_imports=("dataset", "metrics", "task", "metrics.cdm_metric"),
    ),
}


def missing_cdm_binaries(version: str = "v1.6") -> list[str]:
    binaries = (*CDM_BINARIES, *EVALUATORS[version].cdm_binaries)
    return [binary for binary in binaries if not shutil.which(binary)]


# ---------------------------------------------------------------------------
# Evaluator provisioning
# ---------------------------------------------------------------------------


def _run(cmd: Sequence[str], **kwargs: Any) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(list(cmd), check=True, **kwargs)


def ensure_evaluator(version: str = "v1.6") -> tuple[Path, Path]:
    """Return ``(repo_dir, python)`` for the official evaluator, provisioning it if needed."""
    spec = EVALUATORS[version]
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
    home = root / spec.repo_commit[:12]
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
        _run(["git", "-C", str(repo), "checkout", "--quiet", spec.repo_commit])
        _run([uv, "venv", "--quiet", "--python", spec.python_version, str(venv)])
        packages = [arg.format(repo=repo) for arg in spec.install]
        _run([uv, "pip", "install", "--quiet", "--python", str(python), *packages])
        for installed, replacement in spec.replacements:
            _run([uv, "pip", "uninstall", "--quiet", "--python", str(python), installed])
            _run([uv, "pip", "install", "--quiet", "--python", str(python), replacement])
        _check_startup_imports(spec, repo, python)
        ready.touch()
    return repo, python


def _check_startup_imports(spec: EvaluatorVersion, repo: Path, python: Path) -> None:
    if not spec.startup_imports:
        return
    script = "import importlib\n" + "".join(
        f"importlib.import_module({module!r})\n" for module in spec.startup_imports
    )
    proc = subprocess.run(
        [str(python), "-c", script], cwd=repo, capture_output=True, text=True, timeout=600
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-15:])
        raise RuntimeError(
            f"The OmniDocBench {spec.name} evaluator was installed but cannot start:\n{tail}"
        )


# ---------------------------------------------------------------------------
# Running the evaluator
# ---------------------------------------------------------------------------


@dataclass
class OfficialEvaluation:
    """Outcome of one official end-to-end evaluation."""

    #: ``image_name -> {text_edit, read_order_edit, formula_edit, table_teds,
    #: table_teds_s, formula_cdm}``; a key is absent when the page has no such sample.
    per_page: dict[str, dict[str, float]] = field(default_factory=dict)
    #: The evaluator's own page-averaged numbers, in its raw 0-1 units, under the same keys
    #: (plus ``overall`` on the leaderboard's 0-100 scale when CDM ran).
    summary: dict[str, float] = field(default_factory=dict)
    #: Samples the evaluator scored zero because its scoring code crashed or timed out.
    num_scorer_failures: int = 0


def _config(
    version: str, gt_path: Path, pred_dir: Path, *, with_cdm: bool, workers: int
) -> dict[str, Any]:
    formula_metrics = ["Edit_dist", "CDM"] if with_cdm else ["Edit_dist"]
    metrics: dict[str, dict[str, Any]] = {
        "text_block": {"metric": ["Edit_dist"]},
        "display_formula": {"metric": formula_metrics},
        "table": {"metric": ["TEDS", "Edit_dist"]},
        "reading_order": {"metric": ["Edit_dist"]},
    }
    dataset: dict[str, Any] = {
        "dataset_name": "end2end_dataset",
        "ground_truth": {"data_path": str(gt_path)},
        "prediction": {"data_path": str(pred_dir)},
        "match_method": _MATCH_METHOD,
    }
    if version == "v1.6":
        metrics["display_formula"]["cdm_workers"] = workers
        metrics["table"]["teds_workers"] = workers
        dataset.update(
            {
                "match_workers": workers,
                "quick_match_truncated_timeout_sec": 300,
                "match_timeout_sec": 420,
                "timeout_fallback_max_chunk_span": 10,
                "timeout_fallback_order_penalty": 0.10,
            }
        )
    return {"end2end_eval": {"metrics": metrics, "dataset": dataset}}


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


def _summary(metric_result: Mapping[str, Any]) -> dict[str, float]:
    """The leaderboard numbers as ``tools/generate_result_tables.ipynb`` reads them."""

    def page_avg(element: str, metric: str) -> float | None:
        value = ((metric_result.get(element) or {}).get("page") or {}).get(metric)
        return float(value["ALL"]) if value and "ALL" in value else None

    def edit(element: str) -> float | None:
        value = ((metric_result.get(element) or {}).get("all") or {}).get("Edit_dist") or {}
        return float(value["ALL_page_avg"]) if "ALL_page_avg" in value else None

    values = {
        "text_edit": edit("text_block"),
        "read_order_edit": edit("reading_order"),
        "table_teds": page_avg("table", "TEDS"),
        "table_teds_s": page_avg("table", "TEDS_structure_only"),
        "formula_cdm": page_avg("display_formula", "CDM"),
    }
    summary = {k: v for k, v in values.items() if v is not None}
    if all(k in summary for k in ("text_edit", "table_teds", "formula_cdm")):
        summary["overall"] = (
            (1.0 - summary["text_edit"]) * 100
            + summary["table_teds"] * 100
            + summary["formula_cdm"] * 100
        ) / 3.0
    return summary


def _count_failures(metric_result: Mapping[str, Any], log_text: str) -> int:
    count = 0
    for element in ("table", "display_formula"):
        for debug in ((metric_result.get(element) or {}).get("metric_debug") or {}).values():
            for counter in ("error_case_count", "timeout_case_count", "exception_case_count"):
                count += int(debug.get(counter) or 0)
    fallbacks = (metric_result.get("match_debug") or {}).get("text_match_fallback_counts") or {}
    count += sum(int(v or 0) for v in fallbacks.values())
    if count == 0:
        # Older evaluators only report a zeroed sample on stdout.
        count = len(re.findall(r"score is set to 0", log_text))
    return count


#: Runs ``pdf_validation.py`` with a deeper recursion limit. The v1.6 display-formula
#: matcher recurses once per predicted formula on a page, so a page with a thousand formulas
#: (a model stuck repeating one) crashes the whole evaluation at Python's default limit of
#: 1,000. Its candidate search is capped, so a deeper limit changes no score. Matching runs in
#: worker threads, which get a larger stack to match.
_LAUNCH_SCRIPT = """
import runpy, sys, threading
sys.setrecursionlimit(100_000)
threading.stack_size(512 * 1024 * 1024)
repo, script = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
sys.argv = [script, *sys.argv[3:]]
runpy.run_path(script, run_name="__main__")
"""


def run_official_evaluation(
    predictions: Mapping[str, str],
    gt_pages: Sequence[Mapping[str, Any]],
    *,
    with_cdm: bool,
    version: str = "v1.6",
    workers: int | None = None,
) -> OfficialEvaluation:
    """Score ``predictions`` (``image_name -> markdown``) against their annotated pages."""
    import yaml

    repo, python = ensure_evaluator(version)
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
            yaml.safe_dump(_config(version, gt_path, pred_dir, with_cdm=with_cdm, workers=workers))
        )
        result_dir = work / "result"
        result_dir.mkdir()

        log_path = work / "evaluator.log"
        with open(log_path, "w") as log:
            proc = subprocess.run(
                [
                    str(python),
                    "-c",
                    _LAUNCH_SCRIPT,
                    str(repo),
                    str(repo / "pdf_validation.py"),
                    "--config",
                    str(config_path),
                ],
                cwd=work,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        log_text = log_path.read_text(errors="replace")
        metric_result = _load(result_dir, "metric_result")
        if proc.returncode != 0 or not metric_result:
            tail = "\n".join(log_text.splitlines()[-40:])
            raise RuntimeError(
                f"OmniDocBench {version} evaluator failed (exit {proc.returncode}). "
                f"Log tail:\n{tail}"
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

        failures = _count_failures(metric_result, log_text)
        if failures:
            logger.error(
                "OmniDocBench evaluator scored %d sample(s) zero after an internal crash, "
                "timeout or matching fallback; see n_scorer_failures.",
                failures,
            )
        return OfficialEvaluation(
            per_page=per_page, summary=_summary(metric_result), num_scorer_failures=failures
        )


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
    for it, as the v1.6 evaluator does. Without it, only pages that have a value are
    averaged. ``scale`` converts to the leaderboard's unit (x100 for TEDS and CDM).
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
    #: Whether missing table / formula pages count as zero (see the page metric).
    zero_fill: bool = True

    def compute(self, responses: Sequence[Response]) -> float:
        text = OmniDocBenchPageMetric("", self.scorer, "text_edit").compute(responses)
        teds = OmniDocBenchPageMetric(
            "",
            self.scorer,
            "table_teds",
            zero_fill="has_table" if self.zero_fill else None,
            scale=100.0,
        ).compute(responses)
        cdm = OmniDocBenchPageMetric(
            "",
            self.scorer,
            "formula_cdm",
            zero_fill="has_formula" if self.zero_fill else None,
            scale=100.0,
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
