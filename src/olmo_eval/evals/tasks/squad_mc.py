from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import LogprobMCAccuracyMetric, LogprobPerCharMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.squad_mc import SQUAD_MC_FIXED_FEWSHOT


def _process_squad_mc_doc(doc: dict[str, Any], index: int) -> Instance | None:
    title = doc.get("title_original", "")
    passage = doc.get("context_original", "").strip()
    question = doc.get("question_original", "")
    choices_data = doc.get("choices", {})
    choices = choices_data.get("text", [])
    answer_key = doc.get("answerKey", "")

    if not question or not choices:
        return None

    gold_idx = ord(answer_key) - ord("A") if answer_key else 0
    gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

    return Instance(
        question=f"Title: {title}\nPassage: {passage}\nQuestion: {question}",
        choices=tuple(choices),
        gold_answer=answer_key,
        metadata={
            "id": doc.get("id", f"squad_mc_{index}"),
            "index": index,
            "dataset": "squad_mc",
            "gold_idx": gold_idx,
            "gold_text": gold_text,
        },
    )


def _build_squad_mc_fixed_fewshot(
    raw_docs: list[dict[str, Any]], num_fewshot: int, seed: int
) -> list[Instance]:
    instances = []
    for doc in raw_docs:
        title = doc["title_original"]
        passage = doc["context_original"].strip()
        question = doc["question_original"]
        choices = tuple(doc["choices"]["text"])
        answer_key = doc["answerKey"]
        gold_idx = ord(answer_key) - ord("A")
        gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

        instances.append(
            Instance(
                question=f"Title: {title}\nPassage: {passage}\nQuestion: {question}",
                choices=choices,
                gold_answer=gold_text,
                metadata={
                    "gold_idx": gold_idx,
                    "gold_text": gold_text,
                    "mc_answer": answer_key,
                },
            )
        )

    if num_fewshot and num_fewshot < len(instances):
        instances = instances[:num_fewshot]
    return instances


def _format_mc(question: str, choices: tuple[str, ...], answer: str | None = None) -> str:
    choices_text = "\n".join(f" {chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    prompt = f"{question}\n{choices_text}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


def _format_rc(question: str, answer: str | None = None) -> str:
    prompt = f"{question}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


class _SquadMCBase(Task):
    metrics = (LogprobMCAccuracyMetric(),)
    num_fewshot = 5
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)
    _fewshot_source_name = "squad_mc_fixed"

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        return _process_squad_mc_doc(doc, index)

    def _build_fewshot(self) -> list[Instance]:
        if getattr(self.config, "fewshot_source", None) == self._fewshot_source_name:
            return _build_squad_mc_fixed_fewshot(
                SQUAD_MC_FIXED_FEWSHOT, self.config.num_fewshot, self.config.fewshot_seed
            )
        return super()._build_fewshot()

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()
        is_mc = self.config.formatter is not None

        parts: list[str] = []
        for ex in fewshot:
            if is_mc:
                answer = ex.metadata.get("mc_answer", "")
                parts.append(_format_mc(ex.question, ex.choices or (), answer))
            else:
                answer = ex.gold_answer or ex.metadata.get("gold_text", "")
                parts.append(_format_rc(ex.question, answer))

        if is_mc:
            parts.append(_format_mc(instance.question, instance.choices or ()))
            continuations = tuple(
                f" {chr(ord('A') + i)}" for i in range(len(instance.choices or ()))
            )
        else:
            parts.append(_format_rc(instance.question))
            continuations = tuple(f" {c}" for c in (instance.choices or ()))

        prompt = "\n\n".join(parts)
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )


@register("squad:mc")
class SquadMC(_SquadMCBase):
    data_source = DataSource(path="allenai/squad_mc", split="validation")
    split = Split.VALIDATION
    formatter = MultipleChoiceFormatter()
    fewshot_source = "squad_mc_fixed"


@register("squad:rc")
class SquadRC(_SquadMCBase):
    data_source = DataSource(path="allenai/squad_mc", split="validation")
    split = Split.VALIDATION
    metrics = (LogprobPerCharMCAccuracyMetric(),)
    fewshot_source = "squad_mc_fixed"


register_variant(
    "squad:mc",
    "olmo3base",
    limit=10_000,
    seed=1234,
    fewshot_source="squad_mc_fixed",
)

register_variant(
    "squad:rc",
    "olmo3base",
    limit=10_000,
    seed=1234,
    fewshot_source="squad_mc_fixed",
)
