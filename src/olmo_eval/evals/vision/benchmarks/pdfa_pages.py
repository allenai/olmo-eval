"""PDFA pages — held-out long-document tasks over runs of PDF pages.

The examples are drawn by OLMo-core's ``PdfaPagesDataset`` from documents no run trains on,
so every prompt, page label and answer has exactly the training form. Five tasks:

``pdfa_pages_ocr``
    One page, transcribed.
``pdfa_pages_transcribe``
    A run of consecutive pages, each after its label (``"Page 12:"``); transcribe one of them.
``pdfa_pages_missing``
    The same run with one page left out; write that page's text.
``pdfa_pages_find``
    The same run and a line quoted from one page; answer with that page's number. A quote
    that itself contains the page number (a running header) gives the answer away, so
    ``clean_quote_accuracy`` leaves those examples out.
``pdfa_pages_order``
    A run of consecutive pages in shuffled order, labelled ``"Image k"``; list the image
    numbers in reading order.

Transcriptions are scored on their first 300 words (see
:mod:`olmo_eval.evals.vision.scoring.pdfa_pages`), which bounds the generation a page needs.
The multi-page tasks also report ``short`` and ``long`` runs separately.

Data is read from ``$PDFA_PAGES_DIR``: ``heldout/examples.jsonl`` lists the examples, and each
names the tar under ``data/pdfa/`` and the byte range that hold its PDF. Pages are rasterized
with pypdfium2 at the example's size, as OLMo-core renders them. The page labels travel on
``LMRequest.image_labels``, which needs a provider with the document prompt layout.
"""

from __future__ import annotations

import functools
import json
import os
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.scoring.pdfa_pages import (
    PageNumberScorer,
    PageOrderScorer,
    PdfaPagesMetric,
    TranscriptionScorer,
)
from olmo_eval.evals.vision.tasks.base import VisionTask

DEFAULT_PDFA_PAGES_DIR = "/weka/oe-training-default/rustin/v7-recipe/long-docs"

# pypdfium2 is not thread-safe.
_PDFIUM_LOCK = threading.Lock()


def pdfa_pages_dir() -> Path:
    """The PDFA pages build, from ``$PDFA_PAGES_DIR``."""
    root = Path(os.environ.get("PDFA_PAGES_DIR", DEFAULT_PDFA_PAGES_DIR))
    if not (root / "heldout" / "examples.jsonl").is_file():
        raise FileNotFoundError(
            f"{root} has no heldout/examples.jsonl; set PDFA_PAGES_DIR to the PDFA pages build."
        )
    return root


def _render_pages(tar_path: str, offset: int, size: int, pages: tuple[int, ...], longest_dim: int):
    """Rasterize pages of a PDF stored in a tar (module-level so it stays picklable).

    :returns: One RGB PIL image per requested page, in the order given.
    """
    import pypdfium2  # type: ignore[ty:unresolved-import]

    with open(tar_path, "rb") as f:
        f.seek(offset)
        pdf_bytes = f.read(size)
    with _PDFIUM_LOCK:
        pdf = pypdfium2.PdfDocument(pdf_bytes)
        try:
            images = []
            for index in pages:
                page = pdf[index]
                scale = longest_dim / max(page.get_size())
                images.append(page.render(scale=scale).to_pil().convert("RGB"))
            return images
        finally:
            pdf.close()


def _quote_has_number(prompt: str, number: int) -> bool:
    """Whether the line quoted in a find-the-page prompt contains ``number`` as a whole number."""
    quote = prompt[prompt.find('"') + 1 : prompt.rfind('"')]
    return re.search(rf"(?<!\d){number}(?!\d)", quote) is not None


def _run_metrics(scorer: Scorer, key: str, name: str) -> tuple[Metric, ...]:
    """``name`` over all examples, then over the short and the long runs of pages."""
    return (
        PdfaPagesMetric(name=name, scorer=scorer, key=key),
        *(
            PdfaPagesMetric(name=f"{run}_{name}", scorer=scorer, key=key, field="run", value=run)
            for run in ("short", "long")
        ),
    )


class PdfaPagesTask(VisionTask):
    """Base class of the PDFA pages tasks; a subclass is one task of the examples file."""

    dependencies = ["pillow", "pypdfium2"]
    split = Split.TEST
    #: The ``task`` value of this task's rows in ``examples.jsonl``.
    pdfa_task: ClassVar[str]

    def _build_instances(self) -> Iterator[Instance]:
        root = pdfa_pages_dir()
        with open(root / "heldout" / "examples.jsonl") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for row in rows:
            if row["task"] != self.pdfa_task:
                continue
            tar_path = root / "data" / "pdfa" / f"pdfa-{row['shard']:04d}.tar"
            metadata: dict[str, Any] = {
                "example_id": row["id"],
                "pdfa_key": row["key"],
                "num_images": len(row["pages"]),
                "run": "long" if row["source"].endswith("_long") else "short",
                "image_labels": row["image_labels"],
                "images": functools.partial(
                    _render_pages,
                    str(tar_path),
                    row["pdf_off"],
                    row["pdf_size"],
                    tuple(row["pages"]),
                    row["render_longest_dim"],
                ),
            }
            metadata.update(self._answer_metadata(row))
            yield Instance(question=row["prompt"], gold_answer=row["answer"], metadata=metadata)

    def _answer_metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        """Task-specific metadata the scorer and the subset metrics read."""
        return {}

    def _attach_images(self, instance: Instance) -> tuple[Any, ...] | None:
        # One lazy entry for the whole run: pages render in the provider.
        return (instance.metadata["images"],)

    def format_request(self, instance: Instance) -> LMRequest:
        labels = instance.metadata["image_labels"]
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
            images=self._attach_images(instance),
            image_labels=tuple(labels) if labels else None,
        )


_TRANSCRIPTION = TranscriptionScorer()
_TRANSCRIPTION_SAMPLING = SamplingParams(temperature=0.0, max_tokens=768)


def _transcription_metrics(runs: bool) -> tuple[Metric, ...]:
    if runs:
        return (
            *_run_metrics(_TRANSCRIPTION, "word_similarity", "word_similarity"),
            *_run_metrics(_TRANSCRIPTION, "word_f1", "word_f1"),
        )
    return (
        PdfaPagesMetric(name="word_similarity", scorer=_TRANSCRIPTION, key="word_similarity"),
        PdfaPagesMetric(name="word_f1", scorer=_TRANSCRIPTION, key="word_f1"),
    )


@register("pdfa_pages_ocr")
class PdfaPagesOcrTask(PdfaPagesTask):
    pdfa_task = "page_ocr"
    sampling_params = _TRANSCRIPTION_SAMPLING
    metrics = _transcription_metrics(runs=False)
    primary_metric = metrics[0]  # word_similarity


@register("pdfa_pages_transcribe")
class PdfaPagesTranscribeTask(PdfaPagesTask):
    pdfa_task = "transcribe_page"
    sampling_params = _TRANSCRIPTION_SAMPLING
    metrics = _transcription_metrics(runs=True)
    primary_metric = metrics[0]  # word_similarity


@register("pdfa_pages_missing")
class PdfaPagesMissingTask(PdfaPagesTask):
    pdfa_task = "missing_page"
    sampling_params = _TRANSCRIPTION_SAMPLING
    metrics = _transcription_metrics(runs=True)
    primary_metric = metrics[0]  # word_similarity


_PAGE_NUMBER = PageNumberScorer()


@register("pdfa_pages_find")
class PdfaPagesFindTask(PdfaPagesTask):
    pdfa_task = "find_page"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8)
    metrics = (
        *_run_metrics(_PAGE_NUMBER, "accuracy", "accuracy"),
        PdfaPagesMetric(
            name="clean_quote_accuracy", scorer=_PAGE_NUMBER, field="quote", value="clean"
        ),
        PdfaPagesMetric(name="chance_accuracy", scorer=_PAGE_NUMBER, key="chance"),
    )
    primary_metric = metrics[0]  # accuracy

    def _answer_metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        number = int(row["target_page"]) + 1
        leaky = _quote_has_number(row["prompt"], number)
        return {"page_number": number, "quote": "leaky" if leaky else "clean"}


_PAGE_ORDER = PageOrderScorer()


@register("pdfa_pages_order")
class PdfaPagesOrderTask(PdfaPagesTask):
    pdfa_task = "page_order"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=96)
    metrics = (
        PdfaPagesMetric(name="accuracy", scorer=_PAGE_ORDER),
        PdfaPagesMetric(name="pair_accuracy", scorer=_PAGE_ORDER, key="pairs"),
    )
    primary_metric = metrics[0]  # accuracy

    def _answer_metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        return {"order": [int(n) for n in re.findall(r"\d+", row["answer"])]}
