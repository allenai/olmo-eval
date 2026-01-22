"""HellaSwag task implementation."""

import re
from collections.abc import Iterator
from typing import Any

from olmo_eval.core import (
    AccuracyMetric,
    Instance,
    LMOutput,
    LMRequest,
    MultipleChoiceFormatter,
    MultipleChoiceScorer,
    RequestType,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.core import Task, TaskConfig, register


def _preprocess(text: str) -> str:
    """Preprocess HellaSwag text by cleaning up formatting artifacts."""
    text = text.strip()
    text = re.sub(r"\.? \[title\]", ". ", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


class HellaSwagTask(Task):
    """HellaSwag continuation prediction task."""

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the validation split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("validation")
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(
                path="Rowan/hellaswag",
                split=split,
            )

    def process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
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

        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=instance.question,
            continuations=instance.choices,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer index from model output."""
        text = output.text.strip()
        if text.isdigit() and 0 <= int(text) < 4:
            return text
        return None


def _hellaswag_config() -> TaskConfig:
    return TaskConfig(
        name="hellaswag",
        data_source=DataSource(path="Rowan/hellaswag"),
        formatter=MultipleChoiceFormatter(
            template="{question}",
            include_choices_in_prompt=False,
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


@register("hellaswag", _hellaswag_config)
class HellaSwag(HellaSwagTask):
    """HellaSwag task (continuation prediction)."""

    pass
