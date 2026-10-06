"""RULER: What's the Real Context Size of Your Long-Context Language Models?

This task implements the RULER benchmark for evaluating long-context language models.
RULER generates synthetic examples to evaluate models across 4 task categories:
- NIAH (Needle in a Haystack): Single/multi-key/multi-value/multi-query variants
- Multi-hop tracing: Variable tracking (VT)
- Aggregation: Common word extraction (CWE), Frequency word extraction (FWE)
- Question Answering: QA with long context

Paper: https://arxiv.org/abs/2404.06654
Original implementation: https://github.com/hsiehjackson/RULER
"""

import os
import re
import string
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics import RecallMetric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput
from olmo_eval.data.ruler_loader import download_ruler_data, load_ruler_dataset
from olmo_eval.data.ruler_tasks import RULER_TASKS
from olmo_eval.evals.tasks.common import register_configured
from olmo_eval.evals.tasks.common.long_context import (
    LongContextTask,
    long_context_sampling_params,
)


def _normalize_answer(s: str) -> str:
    """Normalize answer text for QA scoring (matches HELMET/old framework)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


@dataclass(frozen=True, slots=True)
class RulerQAScorer(Scorer):
    """Substring scorer for RULER QA tasks with HELMET-style normalization.

    Matches old framework behavior: normalizes both gold and prediction
    (lowercase, remove punctuation, remove articles) then checks if any
    normalized gold answer is a substring of the normalized prediction.
    Takes max over all gold answers.
    """

    name: str = "substring_recall"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None or output.text is None:
            return 0.0

        gold_answers = (
            instance.gold_answer
            if isinstance(instance.gold_answer, list)
            else [str(instance.gold_answer)]
        )
        if not gold_answers:
            return 0.0

        pred_norm = _normalize_answer(output.text)

        # max over ground truths (1.0 if any gold answer is found)
        for answer in gold_answers:
            if _normalize_answer(str(answer)) in pred_norm:
                return 1.0
        return 0.0


class RulerTask(LongContextTask):
    """RULER task, one per task variant and context size (e.g. niah_s_1__4096).

    RULER data is pre-generated offline at specific context lengths and
    stored as task-specific JSONL files, so it is loaded with a custom loader
    rather than the standard HuggingFace pipeline.
    """

    name_prefix = "ruler_"
    task_table = RULER_TASKS

    def _load_dataset(self) -> dict[str, Any]:
        return load_ruler_dataset(
            task_name=self.task_name,
            data_path=os.path.join(download_ruler_data(), self.task_config["data"]),
            max_samples=self.config.limit,
            seed=42,
        )

    def render_prompt(self, doc: dict[str, Any]) -> tuple[str, str]:
        """Prefer the released prompt, which already ends in the answer prefix."""
        prebuilt = doc.get("input")
        if isinstance(prebuilt, str) and prebuilt:
            return prebuilt, ""
        return super().render_prompt({"context": "", **doc})

    def instance_id(self, doc: dict[str, Any], index: int) -> Any:
        return doc.get("index", index)


# RULER stops on a newline in any of the forms tokenizers render it.
_NEWLINE_STOPS = ("\n", "Ċ", "ĊĊ", "<0x0A>")

for _task_name, _task_config in RULER_TASKS.items():
    register_configured(
        f"ruler_{_task_name}",
        RulerTask,
        # QA tasks normalize answers HELMET-style; the rest match substrings
        metrics=(RecallMetric(scorer=RulerQAScorer),)
        if _task_config["tag"] == "qa"
        else (RecallMetric(),),
        primary_metric="recall",
        sampling_params=long_context_sampling_params(
            _task_config.get("max_gen_toks", 50),
            _NEWLINE_STOPS if _task_config.get("stop_new_line") else None,
        ),
        limit=100,
    )
