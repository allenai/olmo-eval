"""GDP.pdf — long-document question answering graded against per-task rubrics.

The benchmark (https://surgehq.ai/benchmarks/gdp-pdf, harness ``surge-ai/gdp-pdf``,
data ``surgeai/GDP.pdf``) pairs 100 held-out professional documents with a
question and a hand-written rubric: 1,275 criteria in all, 3-30 per task, over
domains such as healthcare, insurance, construction and real estate. A response
is graded criterion by criterion, and the leaderboard reports

* ``all_pass`` — the response satisfied *every* criterion (the headline number),
* ``mean_criteria`` — the fraction of criteria satisfied.

The official harness attaches the PDF itself, which suits models with native
document input. Vision-language models take images, so each document is
rendered to page images here and attached like any other multi-image task.
``max_pages`` caps how many pages are attached: the documents run 8-192 pages
(mean 62), well past what a short-context model can hold, so the capped task
measures a model on a prefix of each document and is not comparable to the
leaderboard. ``gdp_pdf`` attaches every page; ``gdp_pdf_first16`` attaches the
first 16.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.gdp_pdf_judge import (
    GdpPdfRubricScorer,
    criterion_passes,
    grade_criterion,
)
from olmo_eval.common.types import Instance, Response, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.common.multi_image_base import MultiImageQATask

if TYPE_CHECKING:
    from olmo_eval.common.execution import ScoringContext

HF_REPO = "surgeai/GDP.pdf"

#: Matches the rubric columns of the HF dataset, e.g. "rubric - 12. criterion".
_RUBRIC_RE = re.compile(r"rubric\s*-\s*(\d+)\.\s*criterion", re.IGNORECASE)


def _render_pdf_pages(pdf_path: str, dpi: int, max_pages: int | None) -> list:
    """Render a PDF's pages to PIL images (module-level so it stays picklable)."""
    import io

    import pymupdf
    from PIL import Image

    with pymupdf.open(pdf_path) as doc:
        pages = doc.page_count if max_pages is None else min(max_pages, doc.page_count)
        images = []
        for index in range(pages):
            pixmap = doc[index].get_pixmap(dpi=dpi)
            images.append(Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB"))
    return images


def _lazy_pdf_pages(pdf_path: str, dpi: int, max_pages: int | None):
    """A picklable zero-arg callable rendering one document's pages on demand."""
    return functools.partial(_render_pdf_pages, pdf_path, dpi, max_pages)


def _rubrics_from_row(row: dict) -> list[dict[str, str]]:
    """Collect the non-empty ``rubric - N. criterion`` columns, ordered by N."""
    found: list[tuple[int, str]] = []
    for column, value in row.items():
        match = _RUBRIC_RE.fullmatch(str(column).strip())
        if match and value not in (None, ""):
            found.append((int(match.group(1)), str(value)))
    found.sort(key=lambda pair: pair[0])
    return [{"id": f"r{number}", "criterion": text} for number, text in found]


def _store_result(response: Response, grades: list[dict], n_criteria: int) -> None:
    scored = [grade for grade in grades if grade["result"] != "unscored"]
    passed = sum(grade["result"] == "pass" for grade in grades)
    fully_scored = bool(n_criteria) and len(scored) == n_criteria

    all_pass = (1.0 if passed == n_criteria else 0.0) if fully_scored else None
    mean_criteria = passed / len(scored) if scored else None

    response.scores["gdp_pdf"] = all_pass or 0.0
    if not response.outputs:
        return
    output = response.outputs[0]
    if output.metadata is None:
        output.metadata = {}
    output.metadata["gdp_pdf_result"] = {
        "all_pass": all_pass,
        "mean_criteria": mean_criteria,
        "n_criteria": n_criteria,
        "n_scored": len(scored),
        "grades": grades,
    }
    output.metadata["score:gdp_pdf"] = all_pass or 0.0


def _results(responses: Sequence[Response]) -> Iterator[dict]:
    for response in responses:
        for output in response.outputs:
            if output.metadata and "gdp_pdf_result" in output.metadata:
                yield output.metadata["gdp_pdf_result"]


@dataclass(frozen=True)
class GdpPdfMeanMetric(Metric):
    """Mean of one per-response value, over the responses where it was scored.

    A response whose grading did not complete is missing rather than zero, so
    an outage shows up as coverage loss instead of a depressed score.
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    key: str = "all_pass"

    def compute(self, responses: Sequence[Response]) -> float:
        values = [
            float(result[self.key])
            for result in _results(responses)
            if result[self.key] is not None
        ]
        return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class GdpPdfUnscoredMetric(Metric):
    """Number of rubric criteria the judge never returned a usable verdict for."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        return float(
            sum(result["n_criteria"] - result["n_scored"] for result in _results(responses))
        )


_SCORER = GdpPdfRubricScorer()
_METRICS: tuple[Metric, ...] = (
    GdpPdfMeanMetric(name="all_pass", scorer=_SCORER, key="all_pass"),
    GdpPdfMeanMetric(name="mean_criteria", scorer=_SCORER, key="mean_criteria"),
    GdpPdfUnscoredMetric(name="n_unscored", scorer=_SCORER),
)


@register("gdp_pdf")
class GdpPdfTask(MultiImageQATask):
    """Every page of each document, for models with the context to hold them."""

    #: Rendering needs pymupdf; the rubric judge calls OpenAI.
    dependencies = ["pillow", "pymupdf", "openai", "datasets", "huggingface-hub"]
    #: Rubric answers are long-form prose, unlike the short-answer image tasks.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    metrics = _METRICS
    primary_metric = _METRICS[0]  # all_pass
    split = Split.TEST

    #: Pages attached per document; None attaches all of them.
    max_pages: int | None = None
    #: Render resolution. Page images are downscaled again by each model's
    #: own preprocessing, so this only needs to preserve small print.
    dpi: int = 150

    @property
    def max_images(self) -> int:  # type: ignore[override]
        return self.max_pages if self.max_pages is not None else 10_000

    def _build_instances(self) -> Iterator[Instance]:
        import datasets
        from huggingface_hub import hf_hub_download

        rows = datasets.load_dataset(HF_REPO, split=self.config.split.value)
        for index in range(len(rows)):
            row = rows[index]
            pdf_path = hf_hub_download(HF_REPO, row["pdf_path"], repo_type="dataset")
            rubrics = _rubrics_from_row(row)
            yield Instance(
                question=row["prompt"],
                gold_answer=None,
                metadata={
                    "task_id": str(row.get("task_id") or index),
                    "example_id": str(row.get("task_id") or index),
                    "domain": row.get("domain"),
                    "pdf_path": row["pdf_path"],
                    "rubrics": rubrics,
                    "images": _lazy_pdf_pages(pdf_path, self.dpi, self.max_pages),
                },
            )

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: ScoringContext | None = None,
    ) -> Sequence[Response]:
        import asyncio

        limit = context.scoring_concurrency if context is not None else 8
        semaphore = asyncio.Semaphore(limit)

        async def grade(text: str, criterion: str):
            async with semaphore:
                return await grade_criterion(
                    text, criterion, model=_SCORER.model, cache_dir=_SCORER.cache_dir
                )

        for response in responses:
            rubrics: list[dict[str, str]] = response.instance.metadata.get("rubrics", [])
            text = _response_text(response)
            verdicts = await asyncio.gather(
                *(grade(text, rubric["criterion"]) for rubric in rubrics)
            )
            grades = [
                {
                    "rubric_id": rubric["id"],
                    "result": (
                        "unscored"
                        if verdict is None
                        else "pass"
                        if criterion_passes(verdict["score"])
                        else "fail"
                    ),
                    "score": None if verdict is None else str(verdict["score"]),
                    "rationale": None if verdict is None else verdict["rationale"],
                }
                for rubric, verdict in zip(rubrics, verdicts, strict=True)
            ]
            _store_result(response, grades, len(rubrics))
        return responses


def _response_text(response: Response) -> str:
    for output in response.outputs:
        text = output.text
        if text:
            return text
    return ""


@register("gdp_pdf_first16")
class GdpPdfFirst16Task(GdpPdfTask):
    """The first 16 pages only, for models that cannot hold a whole document."""

    max_pages: int | None = 16
