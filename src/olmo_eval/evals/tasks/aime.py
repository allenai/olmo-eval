from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric, PassAtKMetric
from olmo_eval.common.scorers import MinervaMathScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, SamplingParams
from olmo_eval.data import DataSource
from olmo_eval.evals.extract import MathExtractor
from olmo_eval.evals.tasks.common import Task, register, register_variant

ZS_COT_R1_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

_PASS_AT_32_METRICS = (
    AccuracyMetric(scorer=MinervaMathScorer),
    PassAtKMetric(k=1, scorer=MinervaMathScorer),
    PassAtKMetric(k=4, scorer=MinervaMathScorer),
    PassAtKMetric(k=8, scorer=MinervaMathScorer),
    PassAtKMetric(k=16, scorer=MinervaMathScorer),
    PassAtKMetric(k=32, scorer=MinervaMathScorer),
)

_PASS_AT_32_SAMPLING = SamplingParams(
    max_tokens=16384,
    temperature=0.6,
    top_p=0.95,
    num_samples=32,
)

_PASS_AT_32_R1_FORMATTER = ChatFormatter(
    system_prompt=ZS_COT_R1_SYSTEM_PROMPT,
    user_template="{question}",
)


class AIMETask(Task):
    data_source = DataSource(path="allenai/aime-2021-2025")
    formatter = ChatFormatter(user_template="{question}")
    metrics = (AccuracyMetric(scorer=MinervaMathScorer),)
    num_fewshot = 0
    sampling_params = SamplingParams(max_tokens=16384, temperature=0.0)
    dependencies = ["lark>=1.0"]

    years: list[int]

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        year = doc.get("year")
        if year not in self.years:
            return None

        question = doc["problem"]
        gold_answer = str(doc["answer"])

        # AIME answers are integers 0-999; normalize by stripping leading zeros
        gold_normalized = gold_answer.lstrip("0") or "0"

        return Instance(
            question=question,
            gold_answer=gold_normalized,
            metadata={
                "id": doc.get("id", index),
                "year": year,
                "problem_number": doc.get("problem_number"),
                "all_gold_answers": [gold_normalized],
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        assert self.config.formatter is not None
        return self.config.formatter.format(instance, self.get_fewshot())

    def extract_answer(self, output: LMOutput) -> str | None:
        answers = MathExtractor.extract_answer(output.text)
        output.metadata["all_extracted_answers"] = answers if answers else []
        return answers[0] if answers else None


@register("aime_2024")
class AIME2024Task(AIMETask):
    years = [2024]


@register("aime_2025")
class AIME2025Task(AIMETask):
    years = [2025]


for _year in ("2024", "2025"):
    _task = f"aime_{_year}"

    register_variant(
        _task,
        "pass_at_32_rlzero",
        formatter=_PASS_AT_32_R1_FORMATTER,
        metrics=_PASS_AT_32_METRICS,
        primary_metric=PassAtKMetric(k=1, scorer=MinervaMathScorer),
        sampling_params=_PASS_AT_32_SAMPLING,
    )

    register_variant(
        _task,
        "pass_at_32_deepseek",
        formatter=_PASS_AT_32_R1_FORMATTER,
        metrics=_PASS_AT_32_METRICS,
        primary_metric=PassAtKMetric(k=1, scorer=MinervaMathScorer),
        sampling_params=_PASS_AT_32_SAMPLING,
    )
