from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric, PassAtKMetric
from olmo_eval.common.scorers import MinervaMathScorer
from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.tasks.minerva_math import MinervaMathTask

_PASS_AT_32_METRICS = {
    "acc": AccuracyMetric(scorer=MinervaMathScorer),
    "k1": PassAtKMetric(k=1, scorer=MinervaMathScorer),
    "k4": PassAtKMetric(k=4, scorer=MinervaMathScorer),
    "k8": PassAtKMetric(k=8, scorer=MinervaMathScorer),
    "k16": PassAtKMetric(k=16, scorer=MinervaMathScorer),
    "k32": PassAtKMetric(k=32, scorer=MinervaMathScorer),
}

_PASS_AT_32_SAMPLING = SamplingParams(
    max_tokens=32768,
    temperature=0.6,
    top_p=0.95,
    num_samples=32,
)
_PASS_AT_32_16K_SAMPLING = replace(_PASS_AT_32_SAMPLING, max_tokens=16384)

_COT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."

_PASS_AT_32_FORMATTER = ChatFormatter(
    user_template="{question}" + _COT_SUFFIX,
)


def _normalize_gold_answer(answer: Any) -> str:
    gold_answer = str(answer)
    # AIME data stores some answers with leading zeros, strip them,
    # fall back to 0 if the answer was just 0
    return gold_answer.lstrip("0") or "0"


def _build_aime_instance(
    *,
    question: str,
    answer: Any,
    identifier: Any,
    year: int,
    problem_number: Any,
) -> Instance:
    gold_normalized = _normalize_gold_answer(answer)

    return Instance(
        question=question,
        gold_answer=gold_normalized,
        metadata={
            "id": identifier,
            "year": year,
            "problem_number": problem_number,
            "all_gold_answers": [gold_normalized],
        },
    )


class AIMETask(MinervaMathTask):
    data_source = DataSource(path="allenai/aime-2022-2025")
    split = Split.TRAIN  # HF dataset only has a train split
    formatter = ChatFormatter(user_template="{question}")
    metrics = (AccuracyMetric(scorer=MinervaMathScorer),)
    num_fewshot = 0
    sampling_params = SamplingParams(max_tokens=32768, temperature=0.0)

    years: tuple[int, ...]

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        year = doc.get("year")
        if year not in self.years:
            return None

        return _build_aime_instance(
            question=doc["problem"],
            answer=doc["answer"],
            identifier=doc.get("id", index),
            year=year,
            problem_number=doc.get("problem_number"),
        )


@register("aime_2022")
class AIME2022Task(AIMETask):
    years = (2022,)


@register("aime_2023")
class AIME2023Task(AIMETask):
    years = (2023,)


@register("aime_2024")
class AIME2024Task(AIMETask):
    years = (2024,)


@register("aime_2025")
class AIME2025Task(AIMETask):
    years = (2025,)


@register("aime_2026")
class AIME2026Task(AIMETask):
    data_source = DataSource(path="MathArena/aime_2026")
    years = (2026,)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        problem_idx = doc.get("problem_idx")
        return _build_aime_instance(
            question=doc["problem"],
            answer=doc["answer"],
            identifier=problem_idx if problem_idx is not None else index,
            year=2026,
            problem_number=problem_idx,
        )


# RL Zero uses slightly different defaults.
_RLZERO_FORMATTER = ChatFormatter(user_template="{question}")

_RLZERO_SAMPLING = SamplingParams(
    max_tokens=16384,
    temperature=1.0,
    top_p=0.95,
    num_samples=32,
)

for _year in (2022, 2023, 2024, 2025, 2026):
    register_variant(
        f"aime_{_year}",
        "pass_at_32",
        formatter=_PASS_AT_32_FORMATTER,
        metrics=tuple(_PASS_AT_32_METRICS.values()),
        primary_metric=_PASS_AT_32_METRICS["k1"],
        sampling_params=_PASS_AT_32_SAMPLING,
    )

    register_variant(
        f"aime_{_year}",
        "pass_at_32_rlzero",
        formatter=_RLZERO_FORMATTER,
        metrics=tuple(_PASS_AT_32_METRICS.values()),
        primary_metric=_PASS_AT_32_METRICS["k1"],
        sampling_params=_RLZERO_SAMPLING,
    )

for _year in (2022, 2023, 2024, 2025, 2026):
    register_variant(
        f"aime_{_year}",
        "16k",
        sampling_params=_PASS_AT_32_16K_SAMPLING,
    )
