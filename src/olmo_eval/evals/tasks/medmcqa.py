"""MedMCQA task implementation."""

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
from olmo_eval.evals.extract import extract_mcqa_answer
from olmo_eval.evals.tasks.core import Task, TaskConfig, register


class MedMCQATask(Task):
    """MedMCQA medical multiple choice question answering task."""

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from all splits."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            for split in ("train", "validation", "test"):
                source = self._get_source_for_split(split)
                for doc in loader.load(source):
                    self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(
                path="openlifescienceai/medmcqa",
                split=split,
            )

    def process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        choices_raw = [doc["opa"], doc["opb"], doc.get("opc"), doc.get("opd")]
        choices = tuple(c for c in choices_raw if c is not None)

        gold_idx = int(doc["cop"])
        answer_key = chr(ord("A") + gold_idx)

        return Instance(
            question=doc["question"],
            gold_answer=answer_key,
            choices=choices,
            metadata={
                "id": doc["id"],
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx] if gold_idx < len(choices) else None,
                "subject_name": doc.get("subject_name", ""),
                "topic_name": doc.get("topic_name", ""),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        choices_text = "\n".join(
            f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(instance.choices or [])
        )
        prompt = f"Question: {instance.question}\n\n{choices_text}\n\nAnswer:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
            continuations=tuple(
                f" {chr(ord('A') + i)}" for i in range(len(instance.choices or []))
            ),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer letter from model output."""
        return extract_mcqa_answer(output.text, [r"[A-D]"])


def _medmcqa_config() -> TaskConfig:
    return TaskConfig(
        name="medmcqa",
        data_source=DataSource(path="openlifescienceai/medmcqa"),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


@register("medmcqa", _medmcqa_config)
class MedMCQA(MedMCQATask):
    """MedMCQA task (medical multiple choice)."""

    pass
