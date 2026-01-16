"""HellaSwag task implementation."""

import re
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

from olmo_eval.core import (
    AccuracyMetric,
    Instance,
    LMOutput,
    LMRequest,
    MultipleChoiceFormatter,
    MultipleChoiceScorer,
    RequestType,
)
from olmo_eval.evals.tasks.core import Task, TaskConfig, register


def _preprocess(text: str) -> str:
    """Preprocess HellaSwag text by cleaning up formatting artifacts."""
    text = text.strip()
    # Remove [title] markers
    text = re.sub(r"\.? \[title\]", ". ", text)
    # Remove other bracketed content
    text = re.sub(r"\[.*?\]", "", text)
    # Normalize whitespace
    text = text.replace("  ", " ")
    return text


class HellaSwagTask(Task):
    """HellaSwag continuation prediction task."""

    hf_path: str = "Rowan/hellaswag"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._instances_cache: list[Instance] | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the validation split."""
        if self._instances_cache is None:
            self._instances_cache = []
            dataset = load_dataset(
                self.hf_path,
                split="validation",
                trust_remote_code=True,
            )
            for doc in dataset:
                self._instances_cache.append(self._process_doc(doc))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        # Build context from ctx_a and ctx_b
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()

        # Preprocess choices
        choices = tuple(_preprocess(ending) for ending in doc["endings"])
        gold_idx = int(doc["label"])

        return Instance(
            question=_preprocess(ctx),
            gold_answer=str(gold_idx),
            choices=choices,
            metadata={
                "id": doc["ind"],
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx] if gold_idx < len(choices) else None,
                "activity_label": doc.get("activity_label", ""),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        # Default: format as continuation task with context + choices
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=instance.question,
            continuations=instance.choices,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer index from model output."""
        # For logprob-based evaluation, the answer is determined by the scorer
        # For generation, try to extract a number
        text = output.text.strip()
        if text.isdigit() and 0 <= int(text) < 4:
            return text
        return None


def _hellaswag_config() -> TaskConfig:
    return TaskConfig(
        name="hellaswag",
        hf_dataset="Rowan/hellaswag",
        formatter=MultipleChoiceFormatter(
            template="{question}",
            include_choices_in_prompt=False,
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer_name="multiple_choice"),),
    )


@register("hellaswag", _hellaswag_config)
class HellaSwag(HellaSwagTask):
    """HellaSwag task (continuation prediction)."""

    pass
