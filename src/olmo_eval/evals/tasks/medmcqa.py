from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import LogprobMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

logger = logging.getLogger(__name__)


@register("medmcqa")
class MedMCQA(Task):
    data_source = DataSource(path="openlifescienceai/medmcqa", split="validation")
    split = Split.VALIDATION
    formatter = MultipleChoiceFormatter(
        template="{question}",
        choice_template=" {choice}",
        include_choices_in_prompt=True,
        prompt_suffix="\n\nAnswer:",
    )
    metrics = (LogprobMCAccuracyMetric(),)
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc.get("question", "")
        if not question:
            return None

        choices = [
            doc.get("opa", ""),
            doc.get("opb", ""),
            doc.get("opc", ""),
            doc.get("opd", "")
        ]

        # Remove any empty choices
        choices = [c for c in choices if c.strip()]
        if not choices:
            return None

        gold_idx = int(doc.get("cop", 0))

        # Validate gold index
        if not (0 <= gold_idx < len(choices)):
            return None

        # Get gold text for BPB/PPL evaluation
        gold_text = choices[gold_idx] if gold_idx < len(choices) else ""

        return Instance(
            question=question,
            choices=choices,
            gold_answer=chr(ord('A') + gold_idx),  # Convert to letter (A, B, C, D)
            metadata={
                "id": doc.get("id", f"medmcqa_{index}"),
                "index": index,
                "dataset": "medmcqa",
                "gold_idx": gold_idx,
                "gold_text": gold_text,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        # Fallback: create a simple multiple choice request
        prompt = instance.question
        if instance.choices:
            choices_text = "\n".join(
                f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(instance.choices)
            )
            prompt = f"{prompt}\n\n{choices_text}"

        continuations = tuple(f" {chr(ord('A') + i)}" for i in range(len(instance.choices)))

        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )


register_variant(
    "medmcqa",
    "mc",
    formatter=MultipleChoiceFormatter(
        template="{question}",
        choice_template=" {choice}",
        include_choices_in_prompt=True,
        prompt_suffix="\n\nAnswer:",
    ),
    metrics=(LogprobMCAccuracyMetric(),),
)

register_variant(
    "medmcqa",
    "none",
    formatter=MultipleChoiceFormatter(
        template="{question}",
        choice_template=" {choice}",
        include_choices_in_prompt=True,
        prompt_suffix="\n\nAnswer:",
    ),
    metrics=(LogprobMCAccuracyMetric(),),
)