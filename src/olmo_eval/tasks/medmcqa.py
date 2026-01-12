"""MedMCQA task implementation."""

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
from olmo_eval.extract import extract_mcqa_answer
from olmo_eval.tasks import Task, TaskConfig, register


class MedMCQATask(Task):
    """MedMCQA medical multiple choice question answering task."""

    hf_path: str = "openlifescienceai/medmcqa"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._instances_cache: list[Instance] | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from all splits."""
        if self._instances_cache is None:
            self._instances_cache = []
            for split in ("train", "validation", "test"):
                dataset = load_dataset(
                    self.hf_path,
                    split=split,
                    trust_remote_code=True,
                )
                for doc in dataset:
                    self._instances_cache.append(self._process_doc(doc))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        # Build choices from opa, opb, opc, opd
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

        # Default: format as multiple choice question with labeled choices
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
        hf_dataset="openlifescienceai/medmcqa",
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer_name="multiple_choice"),),
    )


@register("medmcqa", _medmcqa_config)
class MedMCQA(MedMCQATask):
    """MedMCQA task (medical multiple choice)."""

    pass
