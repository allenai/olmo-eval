import re
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import SQuADF1Metric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.jeopardy import JEOPARDY_FIXED_FEWSHOT


def _parse_context(context: str) -> tuple[str, str]:
    match = re.findall(r"(.*?):\s*(.*)", context)
    if match:
        return match[0][0], match[0][1]
    return "", context


def _format_query(category: str, question: str) -> str:
    return f"Category: {category}\nQuestion: {question}\nAnswer:"


class JeopardyBase(Task):
    fewshot_split: str = "train"
    metrics = (SQuADF1Metric(),)
    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0,
        stop_sequences=("\n\n", "Question:", "Category:"),
    )
    _fewshot_source_name = "jeopardy_fixed"

    def _build_fewshot(self) -> list[Instance]:
        if getattr(self.config, "fewshot_source", None) == self._fewshot_source_name:
            return self._build_fixed_fewshot()
        return super()._build_fewshot()

    def _build_fixed_fewshot(self) -> list[Instance]:
        instances = []
        for doc in JEOPARDY_FIXED_FEWSHOT:
            category, question = _parse_context(doc["context"])
            instances.append(
                Instance(
                    question=_format_query(category, question),
                    gold_answer=doc["continuation"],
                    metadata={"category": doc["category"]},
                )
            )
        if self.config.num_fewshot and self.config.num_fewshot < len(instances):
            instances = instances[: self.config.num_fewshot]
        return instances

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        context = doc.get("context", "")
        continuation = doc.get("continuation", "")
        if not context or not continuation:
            return None

        category, question = _parse_context(context)

        return Instance(
            question=_format_query(category, question),
            gold_answer=continuation,
            metadata={
                "id": index,
                "category": doc.get("category", ""),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        parts: list[str] = []
        for ex in self.get_fewshot():
            parts.append(f"{ex.question} {ex.gold_answer}")
        parts.append(instance.question)
        prompt = "\n\n".join(parts)
        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)


@register("jeopardy")
class Jeopardy(JeopardyBase):
    data_source = DataSource(path="soldni/jeopardy", subset="mosaicml_gauntlet", split="train")
    split = Split.TRAIN
    num_fewshot = 5
    fewshot_source = "jeopardy_fixed"


register_variant(
    "jeopardy",
    "gen",
    # no-op variant, for naming parity
)

register_variant(
    "jeopardy",
    "olmo3base",
    data_source=DataSource(path="soldni/jeopardy", subset="all_questions", split="train"),
    limit=10_000,
    seed=1234,
    fewshot_source="jeopardy_fixed",
)
