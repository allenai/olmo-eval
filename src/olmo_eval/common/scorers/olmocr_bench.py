"""Scorer for olmOCR-bench: run a page's unit tests against the model's markdown.

olmOCR-bench (https://huggingface.co/datasets/allenai/olmOCR-bench) grades a page by a set
of binary unit tests — text presence/absence, reading order, table cell relations, rendered
math equivalence, and a degenerate-output ``baseline`` check. The tests are executed by the
official implementation (``olmocr.bench.tests``), so a pass here means exactly what it means
on the published leaderboard.

Math tests compare KaTeX renderings, which the official code produces in a headless Chromium
through Playwright. :func:`ensure_olmocr_bench_runtime` verifies that stack up front and
installs the browser when it is missing, so a run fails before inference rather than scoring
every math test as a failure afterwards.

Required ``instance.metadata``:

==========  ==========================================================================
``tests``   list of ``{"category": str, "test": dict}``; ``test`` is one verbatim JSONL
            row of the benchmark (or a synthesized ``baseline`` row)
==========  ==========================================================================

The per-test outcomes are stored on ``output.metadata["olmocr_bench_result"]`` for
:class:`OlmocrBenchCategoryMetric` / :class:`OlmocrBenchOverallMetric` to aggregate; the
scorer itself returns the fraction of the page's tests that passed.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, Response

logger = logging.getLogger(__name__)

RESULT_KEY = "olmocr_bench_result"

#: Pseudo-category the official aggregation assigns the synthesized per-PDF baseline tests.
BASELINE_CATEGORY = "baseline"

_RUNTIME_LOCK = threading.Lock()
_runtime_ready = False

#: A reference equation that must render for the math tests to be runnable at all.
_PROBE_EQUATION = r"e^{i\pi} + 1 = 0"


def _probe_render() -> bool:
    from olmocr.bench.katex.render import render_equation

    return render_equation(_PROBE_EQUATION, use_cache=False) is not None


def ensure_olmocr_bench_runtime() -> None:
    """Verify the official scorer can run, installing the headless browser if needed.

    Raises ``RuntimeError`` when the scorer package is missing or equations still cannot be
    rendered after installing Chromium.
    """
    global _runtime_ready
    with _RUNTIME_LOCK:
        if _runtime_ready:
            return
        try:
            import olmocr.bench.tests  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "olmOCR-bench scoring needs the official scorer: pip install 'olmocr[bench]' numpy"
            ) from exc

        try:
            ready = _probe_render()
        except Exception as exc:
            logger.info("KaTeX render probe failed (%s); installing Chromium", exc)
            ready = False
        if not ready:
            for extra in ([], ["--with-deps"]):
                cmd = [sys.executable, "-m", "playwright", "install", *extra, "chromium"]
                logger.info("Running: %s", " ".join(cmd))
                subprocess.run(cmd, check=False)
                try:
                    ready = _probe_render()
                except Exception as exc:
                    logger.warning("KaTeX render probe still failing: %s", exc)
                    ready = False
                if ready:
                    break
        if not ready:
            raise RuntimeError(
                "olmOCR-bench math tests need a headless Chromium for KaTeX rendering and it "
                "could not be started. Install it with "
                "`python -m playwright install --with-deps chromium`."
            )
        _runtime_ready = True


def run_page_tests(tests: Sequence[dict[str, Any]], markdown: str) -> list[dict[str, Any]]:
    """Run every unit test of one page; an exception inside a test counts as a failure."""
    from olmocr.bench.tests import load_single_test

    results: list[dict[str, Any]] = []
    for entry in tests:
        row = entry["test"]
        try:
            passed, explanation = load_single_test(dict(row)).run(markdown)
        except Exception as exc:
            passed, explanation = False, f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "id": row["id"],
                "type": row["type"],
                "category": entry["category"],
                "passed": bool(passed),
                "explanation": explanation,
            }
        )
    return results


def _response_text(output: LMOutput) -> str:
    answer = output.extracted_answer
    if isinstance(answer, str) and answer:
        return answer
    return output.text or ""


@dataclass(frozen=True, slots=True)
class OlmocrBenchScorer(Scorer):
    """Fraction of a page's olmOCR-bench unit tests that the output passes."""

    name: str = "olmocr_bench"

    def score(self, instance: Instance, output: LMOutput) -> float:
        ensure_olmocr_bench_runtime()
        results = run_page_tests(instance.metadata["tests"], _response_text(output))
        if output.metadata is None:
            output.metadata = {}
        passed = sum(r["passed"] for r in results)
        output.metadata[RESULT_KEY] = {"tests": results, "passed": passed, "total": len(results)}
        return passed / len(results) if results else 0.0


def _test_results(responses: Sequence[Response]) -> Iterator[dict[str, Any]]:
    for response in responses:
        if not response.outputs:
            continue
        result = (response.outputs[0].metadata or {}).get(RESULT_KEY)
        if result:
            yield from result["tests"]


def _category_pass_rates(responses: Sequence[Response]) -> dict[str, float]:
    passed: dict[str, int] = {}
    total: dict[str, int] = {}
    for test in _test_results(responses):
        category = test["category"]
        total[category] = total.get(category, 0) + 1
        passed[category] = passed.get(category, 0) + int(test["passed"])
    return {category: passed[category] / total[category] for category in total}


@dataclass(frozen=True)
class OlmocrBenchCategoryMetric(Metric):
    """Pass rate over the unit tests of one category (one JSONL file, or ``baseline``)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    category: str = ""

    def compute(self, responses: Sequence[Response]) -> float:
        return _category_pass_rates(responses).get(self.category, 0.0)


@dataclass(frozen=True)
class OlmocrBenchOverallMetric(Metric):
    """The leaderboard number: unweighted mean of the per-category pass rates."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        rates = _category_pass_rates(responses)
        return sum(rates.values()) / len(rates) if rates else 0.0


@dataclass(frozen=True)
class OlmocrBenchTestTypeMetric(Metric):
    """Pass rate over every unit test of one type, across categories."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    test_type: str = ""

    def compute(self, responses: Sequence[Response]) -> float:
        vals = [t["passed"] for t in _test_results(responses) if t["type"] == self.test_type]
        return sum(vals) / len(vals) if vals else 0.0
