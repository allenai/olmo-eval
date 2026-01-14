"""ARC (AI2 Reasoning Challenge) task implementations."""

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
from olmo_eval.evals.tasks import Task, TaskConfig, register

# Answer extraction regex for multiple choice (A, B, C, D with optional parens)
_ANSWER_PATTERN = re.compile(r"\(?([A-Da-d])\)?")


def _extract_mcqa_answer(text: str) -> str | None:
    """Extract a multiple choice answer (A-D) from text."""
    match = _ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    return None


class ARCTask(Task):
    """Base class for ARC tasks."""

    hf_path: str = "allenai/ai2_arc"

    def __init__(self, config: TaskConfig, dataset_name: str) -> None:
        super().__init__(config)
        self.dataset_name = dataset_name
        self._instances_cache: list[Instance] | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from all dataset splits."""
        if self._instances_cache is None:
            self._instances_cache = []
            for split in ("train", "validation", "test"):
                dataset = load_dataset(
                    self.hf_path,
                    name=self.dataset_name,
                    split=split,
                    trust_remote_code=True,
                )
                for doc in dataset:
                    self._instances_cache.append(self._process_doc(doc))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        answer_key = doc["answerKey"]
        # Handle numeric answer keys (convert 1-based to letter)
        if answer_key.isdigit():
            answer_key = chr(ord("A") + int(answer_key) - 1)

        choices = tuple(doc["choices"]["text"])
        gold_idx = ["A", "B", "C", "D", "E"].index(answer_key)

        return Instance(
            question=doc["question"],
            gold_answer=answer_key,
            choices=choices,
            metadata={
                "id": doc["id"],
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx] if gold_idx < len(choices) else None,
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
            continuations=(" A", " B", " C", " D")[: len(instance.choices or [])],
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer letter from model output."""
        return _extract_mcqa_answer(output.text)


def _arc_challenge_config() -> TaskConfig:
    return TaskConfig(
        name="arc_challenge",
        hf_dataset="allenai/ai2_arc",
        hf_subsets=("ARC-Challenge",),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer_name="multiple_choice"),),
    )


def _arc_easy_config() -> TaskConfig:
    return TaskConfig(
        name="arc_easy",
        hf_dataset="allenai/ai2_arc",
        hf_subsets=("ARC-Easy",),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer_name="multiple_choice"),),
    )


@register("arc_challenge", _arc_challenge_config)
class ARCChallenge(ARCTask):
    """ARC Challenge task (harder questions)."""

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config, dataset_name="ARC-Challenge")


@register("arc_easy", _arc_easy_config)
class ARCEasy(ARCTask):
    """ARC Easy task (easier questions)."""

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config, dataset_name="ARC-Easy")
