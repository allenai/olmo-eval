from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import LogprobMCAccuracyMetric, LogprobPerCharMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

# fmt: off
JEOPARDY_MC_FIXED_FEWSHOT = [
    {
        "id": "jeopardy_mc_format_fewshot_0",
        "choices": {
            "text": ["Iceland", "Denmark", "Finland", "Netherlands"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
        "context_original": "HISTORY: Under the 1814 Treaty of Kiel, this country gave Norway to Sweden but kept Greenland & other islands",
    },
    {
        "id": "jeopardy_mc_format_fewshot_1",
        "choices": {
            "text": ["George Wallace", "Barry Goldwater", "Strom Thurmond", "Hubert Humphrey"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
        "context_original": "U.S. HISTORY: In the 1968 election, he won 13 1/2 percent of the popular vote & carried 5 southern states",
    },
    {
        "id": "jeopardy_mc_format_fewshot_2",
        "choices": {
            "text": ["Antonio", "Bassanio", "Gratiano", "Shylock"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "D",
        "context_original": 'SHAKESPEARE: In "The Merchant of Venice" he tells his friend Tubal, "Meet me at our synagogue"',
    },
    {
        "id": "jeopardy_mc_format_fewshot_3",
        "choices": {
            "text": ["Fluorine", "Sulfur", "Chlorine", "Radon"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
        "context_original": 'SCIENCE & NATURE: Sir Humphry Davy named this yellowish-green gas from a Greek word meaning "greenish-yellow"',
    },
    {
        "id": "jeopardy_mc_format_fewshot_5",
        "choices": {
            "text": ["Syngman Rhee", "Kim Il-sung", "Moon Jae-in", "Park Chung-hee"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
        "context_original": "HISTORY: In the midst of the Korean War, this South Korean president was elected to his second of 4 terms",
    },
    {
        "id": "jeopardy_mc_format_fewshot_6",
        "choices": {
            "text": [
                "Income tax reform",
                "Civil rights for African Americans",
                "Prohibition",
                "Women's suffrage",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "D",
        "context_original": "U.S. HISTORY: In 1878 an amendment for this was introduced in Congress; its adoption didn't occur until 1920",
    },
    {
        "id": "jeopardy_mc_format_fewshot_4",
        "choices": {
            "text": ["Coupe", "Jeep", "SUV", "Sedan"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
        "context_original": 'IN THE DICTIONARY: This car name may come from an abbreviation of "general purpose vehicle"',
    },
    {
        "id": "jeopardy_mc_format_fewshot_7",
        "choices": {
            "text": ["Malta", "Crete", "Cyprus", "Sicily"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
        "context_original": "SHAKESPEARE: Othello kills himself on this island",
    },
    {
        "id": "jeopardy_mc_format_fewshot_8",
        "choices": {
            "text": ["Serum", "Hemoglobin", "Platelets", "Plasma"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
        "context_original": "SCIENCE & NATURE: Take the fibrinogen out of blood plasma & you're left with a fluid called this",
    },
    {
        "id": "jeopardy_mc_format_fewshot_9",
        "choices": {
            "text": ["Shakespearean", "Austenian", "Dickensian", "Hemingwayesque"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
        "context_original": 'ADDICTED TO ADJECTIVES: This adjective derives from the name of the author of "Martin Chuzzlewit"',
    },
]
# fmt: on


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


def _format_jeopardy_rc(
    category: str, question: str, answer: str | None = None
) -> str:
    prompt = f"Category: {category}\nQuestion: {question}\nAnswer:"
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


class _JeopardyMCBase(Task):
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
        is_mc = self.config.formatter is not None
        category = instance.metadata.get("category", "")
        choices = instance.choices or ()

        parts: list[str] = []
        for ex in fewshot:
            ex_category = ex.metadata.get("category", "")
            if is_mc:
                ex_answer = ex.metadata.get("mc_answer", "")
                parts.append(
                    _format_jeopardy_mc(ex_category, ex.question, ex.choices or (), ex_answer)
                )
            else:
                ex_answer = ex.metadata.get("gold_text", "")
                parts.append(_format_jeopardy_rc(ex_category, ex.question, ex_answer))

        if is_mc:
            parts.append(_format_jeopardy_mc(category, instance.question, choices))
            continuations = tuple(f" {chr(ord('A') + i)}" for i in range(len(choices)))
        else:
            parts.append(_format_jeopardy_rc(category, instance.question))
            continuations = tuple(f" {c}" for c in choices)

        prompt = "\n\n".join(parts)

        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )


@register("jeopardy:mc")
class JeopardyMC(_JeopardyMCBase):
    data_source = DataSource(path="allenai/jeopardy_mc", split="test")
    split = Split.TEST
    formatter = MultipleChoiceFormatter()
    fewshot_source = "jeopardy_mc_fixed"


@register("jeopardy:rc")
class JeopardyRC(_JeopardyMCBase):
    data_source = DataSource(path="allenai/jeopardy_mc", split="test")
    split = Split.TEST
    metrics = (LogprobPerCharMCAccuracyMetric(),)
    fewshot_source = "jeopardy_mc_fixed"


register_variant("jeopardy:mc", "olmo3base")
register_variant("jeopardy:rc", "olmo3base")
