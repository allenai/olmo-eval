from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import LogprobMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.jeopardy_mc import JEOPARDY_MC_FIXED_FEWSHOT


def _parse_context(context: str) -> tuple[str, str]:
    match = re.findall(r"(.*?):\s*(.*)", context)
    if match:
        return match[0][0], match[0][1]
    return "", context


def _format_jeopardy_mc(
    category: str, question: str, choices: tuple[str, ...], answer: str | None = None
) -> str:
    choices_text = "\n".join(f" {chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    prompt = f"Category: {category}\nQuestion: {question}\n{choices_text}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


def _build_jeopardy_mc_fixed_fewshot(
    raw_docs: list[dict[str, Any]], num_fewshot: int
) -> list[Instance]:
    instances = []
    for doc in raw_docs:
        context = doc["context_original"]
        category, question = _parse_context(context)
        choices = tuple(doc["choices"]["text"])
        answer_key = doc["answerKey"]
        gold_idx = ord(answer_key) - ord("A")
        gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

        instances.append(
            Instance(
                question=question,
                choices=choices,
                gold_answer=gold_text,
                metadata={
                    "category": category,
                    "gold_idx": gold_idx,
                    "gold_text": gold_text,
                    "mc_answer": answer_key,
                },
            )
        )

    if num_fewshot and num_fewshot < len(instances):
        instances = instances[:num_fewshot]
    return instances


@register("jeopardy:mc")
class JeopardyMC(Task):
    data_source = DataSource(path="allenai/jeopardy_mc", split="test")
    split = Split.TEST
    metrics = (LogprobMCAccuracyMetric(),)
    num_fewshot = 5
    fewshot_source = "jeopardy_mc_fixed"
    sampling_params = SamplingParams(temperature=0.0)

    _fewshot_source_name = "jeopardy_mc_fixed"

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        context = doc.get("context_original", "")
        if not context:
            return None

        choices_data = doc.get("choices", {})
        choices = choices_data.get("text", [])
        if not choices:
            return None

        category, question = _parse_context(context)
        answer_key = doc.get("answerKey", "")
        gold_idx = ord(answer_key) - ord("A") if answer_key else 0
        gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

        return Instance(
            question=question,
            choices=tuple(choices),
            gold_answer=answer_key,
            metadata={
                "id": doc.get("id", f"jeopardy_mc_{index}"),
                "index": index,
                "dataset": "jeopardy_mc",
                "category": category,
                "gold_idx": gold_idx,
                "gold_text": gold_text,
                "mc_answer": answer_key,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        if getattr(self.config, "fewshot_source", None) == self._fewshot_source_name:
            return _build_jeopardy_mc_fixed_fewshot(
                JEOPARDY_MC_FIXED_FEWSHOT, self.config.num_fewshot
            )
        return super()._build_fewshot()

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()
        category = instance.metadata.get("category", "")
        choices = instance.choices or ()

        parts: list[str] = []
        for ex in fewshot:
            ex_category = ex.metadata.get("category", "")
            ex_answer = ex.metadata.get("mc_answer", "")
            parts.append(_format_jeopardy_mc(ex_category, ex.question, ex.choices or (), ex_answer))

        parts.append(_format_jeopardy_mc(category, instance.question, choices))
        prompt = "\n\n".join(parts)

        continuations = tuple(f" {chr(ord('A') + i)}" for i in range(len(choices)))

        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )


register_variant("jeopardy:mc", "olmo3base")
