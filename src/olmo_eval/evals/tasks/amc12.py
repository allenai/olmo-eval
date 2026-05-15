import re
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric, PassAtKMetric
from olmo_eval.common.scorers import MinervaMathScorer
from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.tasks.minerva_math import MinervaMathTask

_COT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer letter"
    " (A, B, C, D, or E) within \\boxed{{}}."
)

AMC12_YEARS: tuple[int, ...] = (
    2000,
    2001,
    2002,
    2003,
    2004,
    2005,
    2006,
    2007,
    2008,
    2009,
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2020,
    2022,
    2023,
    2024,
    2025,
)

_PROBLEM_ID_YEAR_RE = re.compile(r"^(\d{4})")


class AMC12Task(MinervaMathTask):
    data_source = DataSource(path="edev2000/amc12-full")
    split = Split.TRAIN
    formatter = ChatFormatter(user_template="{question}" + _COT_SUFFIX)
    metrics = (AccuracyMetric(scorer=MinervaMathScorer),)
    num_fewshot = 0
    sampling_params = SamplingParams(max_tokens=16384, temperature=0.0)

    years: tuple[int, ...] = AMC12_YEARS

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        problem_id = doc.get("problem_id", "")
        match = _PROBLEM_ID_YEAR_RE.match(str(problem_id))
        year = int(match.group(1)) if match else None
        if year not in self.years:
            return None

        question = doc["question"]
        gold_answer = str(doc["answer"]).strip()

        return Instance(
            question=question,
            gold_answer=gold_answer,
            metadata={
                "id": problem_id or index,
                "year": year,
                "difficulty": doc.get("difficulty"),
                "all_gold_answers": [gold_answer],
            },
        )


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


for _year in AMC12_YEARS:
    _task_name = f"amc12_{_year}"
    _class_name = f"AMC12_{_year}Task"

    _task_cls = type(
        _class_name,
        (AMC12Task,),
        {
            "__module__": __name__,
            "__qualname__": _class_name,
            "years": (_year,),
        },
    )
    globals()[_class_name] = _task_cls
    register(_task_name)(_task_cls)

    register_variant(
        _task_name,
        "pass_at_32",
        metrics=tuple(_PASS_AT_32_METRICS.values()),
        primary_metric=_PASS_AT_32_METRICS["k1"],
        sampling_params=_PASS_AT_32_SAMPLING,
    )
