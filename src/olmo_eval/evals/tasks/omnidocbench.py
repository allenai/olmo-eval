"""OmniDocBench v1.6 — end-to-end document parsing (https://github.com/opendatalab/OmniDocBench).

1,651 annotated page images — books, papers, slides, exam sheets, newspapers, notes, in
English and Chinese — including the 296 hard pages v1.6 adds to v1.5's 1,355. The model
converts each page to markdown; the official evaluator matches its text blocks, display
formulas and tables against the annotations and reports, averaged over pages:

* ``text_edit`` — normalized edit distance of text blocks (lower is better),
* ``table_teds`` / ``table_teds_s`` — table TEDS and structure-only TEDS (x100),
* ``formula_cdm`` — display-formula CDM (x100),
* ``read_order_edit`` — reading-order edit distance (lower is better),
* ``overall`` = ``((1 - text_edit) * 100 + table_teds + formula_cdm) / 3``, the leaderboard
  number.

Scoring is the pinned official evaluator run out of process
(:mod:`olmo_eval.common.scorers.omnidocbench`), once per task after every page is
generated. v1.6 numbers are not comparable to the v1.5 leaderboard: the matching and the
page averaging both changed.

CDM renders formulas with LaTeX and needs ``pdflatex``, ImageMagick 7 (``magick``) and
Ghostscript (``gs``) on ``PATH``. ``omnidocbench`` requires them and fails before inference
without them; ``omnidocbench_no_cdm`` skips CDM, so it has no ``overall`` and reports the
text term ``text_score = (1 - text_edit) * 100`` as its primary metric instead.

The prompt is the one the official inference scripts share across general VLMs
(``tools/model_infer``), sent in a single user turn with the page image as shipped.
Decoding is greedy. Data is fetched from the Hub at the pinned v1.6 revision; set
``OMNIDOCBENCH_DIR`` to a local copy (``OmniDocBench.json`` + ``images/``) to read from disk.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.omnidocbench import (
    RESULT_KEY,
    OmniDocBenchOverallMetric,
    OmniDocBenchPageMetric,
    OmniDocBenchScorer,
    OmniDocBenchScorerFailuresMetric,
    OmniDocBenchTextScoreMetric,
    ensure_evaluator,
    missing_cdm_binaries,
    run_official_evaluation,
)
from olmo_eval.common.types import Instance, Response, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.common.ocr_base import OcrTask

if TYPE_CHECKING:
    from olmo_eval.common.execution import ScoringContext

logger = logging.getLogger(__name__)

HF_REPO = "opendatalab/OmniDocBench"
#: The "add v1.6" commit; the repo has no version tags and ``main`` keeps moving.
HF_REVISION = "d386947f7fc3bafdcd756c8485845a2f43a19875"

#: ``tools/model_infer/Qwen3-VL-235B_img2md.py`` of the official repo, verbatim (the copies
#: in the other scripts differ only by string-escaping accidents).
PROMPT = (
    "You are an AI assistant specialized in converting PDF images to Markdown format. "
    "Please follow these instructions for the conversion:\n\n"
    "1. Text Processing:\n"
    "- Accurately recognize all text content in the PDF image without guessing or inferring.\n"
    "- Convert the recognized text into Markdown format.\n"
    "- Maintain the original document structure, including headings, paragraphs, lists, etc.\n\n"
    "2. Mathematical Formula Processing:\n"
    "- Convert all mathematical formulas to LaTeX format.\n"
    "- Enclose inline formulas with \\( \\). For example: This is an inline formula "
    "\\( E = mc^2 \\)\n"
    "- Enclose block formulas with \\[ \\]. For example: "
    "\\[ \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\]\n\n"
    "3. Table Processing:\n"
    "- Convert tables to HTML format.\n"
    "- Wrap the entire table with <table> and </table>.\n\n"
    "4. Figure Handling:\n"
    "- Ignore figures content in the PDF image. Do not attempt to describe or convert "
    "images.\n\n"
    "5. Output Format:\n"
    "- Ensure the output Markdown document has a clear structure with appropriate line "
    "breaks between elements.\n"
    "- For complex layouts, try to maintain the original document's structure and format "
    "as closely as possible.\n\n"
    "Please strictly follow these guidelines to ensure accuracy and consistency in the "
    "conversion. Your task is to accurately convert the content of the PDF image into "
    "Markdown format without adding any extra explanations or comments."
)

_SCORER = OmniDocBenchScorer()
_OVERALL = OmniDocBenchOverallMetric(name="overall", scorer=_SCORER)
_TEXT_SCORE = OmniDocBenchTextScoreMetric(name="text_score", scorer=_SCORER)

_LANGUAGES = {"english": "english", "chinese": "simplified_chinese", "mixed": "en_ch_mixed"}
#: ``page_attribute.subset``: the v1.5 pages and the three hard sets v1.6 added.
_SUBSETS = {
    "v15": "v1.5",
    "equation_hard": "equation_hard",
    "layout_hard": "layout_hard",
    "table_hard": "table_hard",
}


def _page(name: str, component: str, **kwargs) -> OmniDocBenchPageMetric:
    return OmniDocBenchPageMetric(name=name, scorer=_SCORER, component=component, **kwargs)


_COMPONENT_METRICS: tuple[Metric, ...] = (
    _page("text_edit", "text_edit"),
    _page("table_teds", "table_teds", zero_fill="has_table", scale=100.0),
    _page("table_teds_s", "table_teds_s", zero_fill="has_table", scale=100.0),
    _page("formula_edit", "formula_edit"),
    _page("read_order_edit", "read_order_edit"),
    *(
        _page(f"text_edit_{key}", "text_edit", attribute="language", attribute_value=value)
        for key, value in _LANGUAGES.items()
    ),
    *(
        _page(f"text_edit_{key}", "text_edit", attribute="subset", attribute_value=value)
        for key, value in _SUBSETS.items()
    ),
    OmniDocBenchScorerFailuresMetric(name="n_scorer_failures", scorer=_SCORER),
)
_CDM_METRIC = _page("formula_cdm", "formula_cdm", zero_fill="has_formula", scale=100.0)

#: Evaluator summary key -> (our metric name, factor from the evaluator's raw value).
_SUMMARY_CHECKS = {
    "text_block_Edit_dist": ("text_edit", 1.0),
    "table_TEDS": ("table_teds", 100.0),
    "table_TEDS_structure_only": ("table_teds_s", 100.0),
    "reading_order_Edit_dist": ("read_order_edit", 1.0),
    "display_formula_CDM": ("formula_cdm", 100.0),
}


def _data_dir() -> Path:
    local = os.environ.get("OMNIDOCBENCH_DIR")
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            revision=HF_REVISION,
            allow_patterns=["OmniDocBench.json", "images/*"],
        )
    )


def _load_pages(data_dir: Path) -> list[dict]:
    with open(data_dir / "OmniDocBench.json", encoding="utf-8") as f:
        return json.load(f)


def _has_block(page: dict, category_type: str) -> bool:
    return any(
        block.get("category_type") == category_type and not block.get("ignore", False)
        for block in page.get("layout_dets", ())
    )


@register("omnidocbench")
class OmniDocBenchTask(OcrTask):
    #: Provisioning the evaluator also needs `git` and `uv` on PATH.
    dependencies = ["pillow", "huggingface-hub", "pyyaml", "filelock"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8192)
    metrics = (_OVERALL, _CDM_METRIC, *_COMPONENT_METRICS)
    primary_metric = _OVERALL
    split = Split.TEST
    #: Whether formulas are graded with CDM, which needs a LaTeX toolchain.
    with_cdm: bool = True

    def _build_instances(self) -> Iterator[Instance]:
        if self.with_cdm and missing_cdm_binaries():
            raise RuntimeError(
                "omnidocbench grades formulas with CDM, which needs "
                f"{', '.join(missing_cdm_binaries())} on PATH (a LaTeX distribution with CJK "
                "support, ImageMagick 7 and Ghostscript). Install them, or run "
                "omnidocbench_no_cdm, which reports every metric except CDM and overall."
            )
        ensure_evaluator()
        data_dir = _data_dir()
        question = self._question(PROMPT)
        for page in _load_pages(data_dir):
            info = page["page_info"]
            image_name = os.path.basename(info["image_path"])
            attributes = info.get("page_attribute") or {}
            yield Instance(
                question=question,
                gold_answer=None,
                metadata={
                    "example_id": image_name,
                    "image_name": image_name,
                    "image_path": str(data_dir / "images" / image_name),
                    "data_source": attributes.get("data_source"),
                    "language": attributes.get("language"),
                    "layout": attributes.get("layout"),
                    "subset": attributes.get("subset"),
                    "has_table": _has_block(page, "table"),
                    "has_formula": _has_block(page, "equation_isolated"),
                },
            )

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: ScoringContext | None = None,
    ) -> Sequence[Response]:
        # Pages are graded together in `compute_metrics`; nothing is scored per response.
        self._extract_answers(responses)
        return responses

    def compute_metrics(self, responses: Sequence[Response]) -> dict[str, dict[str, float]]:
        predictions = {
            r.instance.metadata["image_name"]: (r.outputs[0].text or "") if r.outputs else ""
            for r in responses
        }
        gt_pages = [
            page
            for page in _load_pages(_data_dir())
            if os.path.basename(page["page_info"]["image_path"]) in predictions
        ]
        evaluation = run_official_evaluation(predictions, gt_pages, with_cdm=self.with_cdm)

        for response in responses:
            if not response.outputs:
                continue
            output = response.outputs[0]
            if output.metadata is None:
                output.metadata = {}
            output.metadata[RESULT_KEY] = {
                **evaluation.per_page.get(response.instance.metadata["image_name"], {}),
                "num_scorer_failures": evaluation.num_scorer_failures,
            }
            score = _SCORER.score(response.instance, output)
            output.metadata[f"score:{_SCORER.name}"] = score
            response.scores[_SCORER.name] = score

        metrics = super().compute_metrics(responses)
        self._check_against_evaluator(metrics, evaluation.summary)
        return metrics

    @staticmethod
    def _check_against_evaluator(metrics: dict[str, dict[str, float]], summary: dict) -> None:
        """Warn when a page average disagrees with the evaluator's own leaderboard summary."""
        for key, (name, factor) in _SUMMARY_CHECKS.items():
            raw = ((summary.get("metrics") or {}).get(key) or {}).get("raw")
            if raw is None or name not in metrics:
                continue
            ours = metrics[name][_SCORER.name]
            if abs(ours - raw * factor) > 1e-6 * max(1.0, factor):
                logger.warning(
                    "OmniDocBench %s: page average %.6f differs from the evaluator's %.6f",
                    name,
                    ours,
                    raw * factor,
                )


@register("omnidocbench_no_cdm")
class OmniDocBenchNoCdmTask(OmniDocBenchTask):
    """OmniDocBench without the CDM formula metric, for hosts without a LaTeX toolchain."""

    metrics = (_TEXT_SCORE, *_COMPONENT_METRICS)
    primary_metric = _TEXT_SCORE
    with_cdm = False
