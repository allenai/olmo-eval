from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import PPLFormatter
from olmo_eval.common.metrics import BPBMetric, LogprobMCAccuracyMetric, LogprobPerCharMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

# fmt: off
NQ_MC_FIXED_FEWSHOT = [
    {
        "id": "nq_open_mc_format_fewshot_0",
        "question": "Which side of the White House is the front?",
        "choices": {"text": ["East", "North", "South", "West"], "label": ["A", "B", "C", "D"]},
        "answerKey": "B",
    },
    {
        "id": "nq_open_mc_format_fewshot_1",
        "question": "Where was the Super Bowl hosted in 2019?",
        "choices": {
            "text": [
                "Atlanta, Georgia",
                "Houston, Texas",
                "Los Angeles, California",
                "Miami, Florida",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
    },
    {
        "id": "nq_open_mc_format_fewshot_2",
        "question": "What is the origin of the name Cynthia?",
        "choices": {"text": ["Latin", "Hebrew", "Norse", "Greek"], "label": ["A", "B", "C", "D"]},
        "answerKey": "D",
    },
    {
        "id": "nq_open_mc_format_fewshot_7",
        "question": "Who composed and performed the theme song for Miami Vice?",
        "choices": {
            "text": ["Danny Elfman", "Hans Zimmer", "Jan Hammer", "John Williams"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
    },
    {
        "id": "nq_open_mc_format_fewshot_4",
        "question": "What is the size of the angles of an equilateral triangle?",
        "choices": {
            "text": ["90\u00b0", "60\u00b0", "120\u00b0", "45\u00b0"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
    },
    {
        "id": "nq_open_mc_format_fewshot_5",
        "question": "Who plays Mavis in the movie Hotel Transylvania?",
        "choices": {
            "text": ["Selena Gomez", "Emma Stone", "Kristen Bell", "Anne Hathaway"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
    },
    {
        "id": "nq_open_mc_format_fewshot_6",
        "question": "Which West Ham players participated in the 1966 World Cup?",
        "choices": {
            "text": ["Wayne Rooney", "David Beckham", "Alan Shearer", "Bobby Moore"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "D",
    },
    {
        "id": "nq_open_mc_format_fewshot_8",
        "question": "Ice sheets and tundra are typical of which K\u00f6ppen climate category?",
        "choices": {
            "text": ["Arid", "Temperate", "Polar", "Tropical"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
    },
    {
        "id": "nq_open_mc_format_fewshot_9",
        "question": "What is the legal marriage age in New York?",
        "choices": {"text": ["18", "21", "16", "25"], "label": ["A", "B", "C", "D"]},
        "answerKey": "A",
    },
]
# fmt: on


def _process_nq_mc_doc(doc: dict[str, Any], index: int) -> Instance | None:
    question = doc.get("question", "")
    choices_data = doc.get("choices", {})
    choices = choices_data.get("text", [])
    answer_key = doc.get("answerKey", "")

    if not question or not choices:
        return None

    gold_idx = ord(answer_key) - ord("A") if answer_key else 0
    gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

    return Instance(
        question=question,
        choices=tuple(choices),
        gold_answer=answer_key,
        metadata={
            "id": doc.get("id", f"nq_mc_{index}"),
            "index": index,
            "dataset": "naturalqs_mc",
            "gold_idx": gold_idx,
            "gold_text": gold_text,
        },
    )


def _build_nq_mc_fixed_fewshot(
    raw_docs: list[dict[str, Any]], num_fewshot: int
) -> list[Instance]:
    instances = []
    for doc in raw_docs:
        question = doc["question"]
        choices = tuple(doc["choices"]["text"])
        answer_key = doc["answerKey"]
        gold_idx = ord(answer_key) - ord("A")
        gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

        instances.append(
            Instance(
                question=question,
                choices=choices,
                gold_answer=answer_key,
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
    prompt = f"Question: {question}\n{choices_text}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


def _format_rc(question: str, choices: tuple[str, ...], answer: str | None = None) -> str:
    prompt = f"Question: {question}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


class _NaturalQsMCBase(Task):
    metrics = (LogprobMCAccuracyMetric(),)
    num_fewshot = 5
    sampling_params = SamplingParams(temperature=0.0)
    _fewshot_source_name = "nq_mc_fixed"

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        return _process_nq_mc_doc(doc, index)

    def _build_fewshot(self) -> list[Instance]:
        if getattr(self.config, "fewshot_source", None) == self._fewshot_source_name:
            return _build_nq_mc_fixed_fewshot(
                NQ_MC_FIXED_FEWSHOT, self.config.num_fewshot
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
                answer = ex.metadata.get("gold_text", "") or ex.gold_answer
                parts.append(_format_rc(ex.question, ex.choices or (), answer))

        if is_mc:
            parts.append(_format_mc(instance.question, instance.choices or ()))
            continuations = tuple(
                f" {chr(ord('A') + i)}" for i in range(len(instance.choices or ()))
            )
        else:
            parts.append(_format_rc(instance.question, instance.choices or ()))
            continuations = tuple(f" {c}" for c in (instance.choices or ()))

        prompt = "\n\n".join(parts)
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )


@register("naturalqs:mc")
class NaturalQsMC(_NaturalQsMCBase):
    data_source = DataSource(path="allenai/nq_open_mc", split="validation")
    split = Split.VALIDATION
    from olmo_eval.common.formatters import MultipleChoiceFormatter

    formatter = MultipleChoiceFormatter()
    fewshot_source = "nq_mc_fixed"


@register("naturalqs:rc")
class NaturalQsRC(_NaturalQsMCBase):
    data_source = DataSource(path="allenai/nq_open_mc", split="validation")
    split = Split.VALIDATION
    metrics = (LogprobPerCharMCAccuracyMetric(),)
    fewshot_source = "nq_mc_fixed"


register_variant(
    "naturalqs:mc",
    "olmo3base",
    data_source=DataSource(path="allenai/nq-gen2mc", split="validation"),
    limit=10_000,
    seed=1234,
    fewshot_source="nq_mc_fixed",
)

register_variant(
    "naturalqs:rc",
    "olmo3base",
    data_source=DataSource(path="allenai/nq-gen2mc", split="validation"),
    limit=10_000,
    seed=1234,
    fewshot_source="nq_mc_fixed",
)


@register("naturalqs:bpb")
class NaturalQsBPB(_NaturalQsMCBase):
    data_source = DataSource(path="allenai/nq_open_mc", split="validation")
    split = Split.VALIDATION
    formatter = PPLFormatter()
    metrics = (BPBMetric(),)
    primary_metric = BPBMetric()
    num_fewshot = 5
    fewshot_source = "nq_mc_fixed"

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc.get("question", "")
        choices_data = doc.get("choices", {})
        choices = choices_data.get("text", [])
        answer_key = doc.get("answerKey", "")

        if not question or not choices:
            return None

        gold_idx = ord(answer_key) - ord("A") if answer_key else 0
        gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

        formatted_question = f"Question: {question}\nAnswer:"

        return Instance(
            question=formatted_question,
            choices=tuple(choices),
            gold_answer=" " + gold_text,
            metadata={
                "id": doc.get("id", f"nq_bpb_{index}"),
                "index": index,
                "gold_idx": gold_idx,
                "gold_text": gold_text,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        if self.config.fewshot_source == "nq_mc_fixed":
            return self._build_bpb_fixed_fewshot()
        return super()._build_fewshot()

    def _build_bpb_fixed_fewshot(self) -> list[Instance]:
        instances = []
        for doc in NQ_MC_FIXED_FEWSHOT:
            question = doc["question"]
            choices = tuple(doc["choices"]["text"])
            answer_key = doc["answerKey"]
            gold_idx = ord(answer_key) - ord("A")
            gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

            formatted_question = f"Question: {question}\nAnswer:"

            instances.append(
                Instance(
                    question=formatted_question,
                    choices=choices,
                    gold_answer=" " + gold_text,
                    metadata={
                        "gold_idx": gold_idx,
                        "gold_text": gold_text,
                    },
                )
            )

        num = self.config.num_fewshot
        if num and num < len(instances):
            instances = instances[:num]
        return instances

    def format_request(self, instance: Instance) -> LMRequest:
        assert self.config.formatter is not None
        return self.config.formatter.format(instance, self.get_fewshot())


register_variant(
    "naturalqs:bpb",
    "olmo3base",
    data_source=DataSource(path="allenai/nq-gen2mc", split="validation"),
    limit=10_000,
    seed=1234,
    fewshot_source="nq_mc_fixed",
)
