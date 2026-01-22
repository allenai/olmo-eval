"""ARC (AI2 Reasoning Challenge) task implementations."""

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

# Answer extraction regex for multiple choice (A, B, C, D with optional parens)
_ANSWER_PATTERN = re.compile(r"\(?([A-Da-d])\)?")


def _extract_mcqa_answer(text: str) -> str | None:
    """Extract a multiple choice answer (A-D) from text."""
    match = _ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    return None


class ARCTask(Task):
    """Base class for ARC tasks.

    Uses the unified DataLoader to load from HuggingFace, local files, S3, or GCS.
    By default loads from all splits (train, validation, test) for the full dataset.
    """

    def __init__(self, config: TaskConfig, dataset_name: str) -> None:
        super().__init__(config)
        self.dataset_name = dataset_name

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from all dataset splits.

        Uses the unified DataLoader to load from the configured data_source.
        Loads from train, validation, and test splits for full dataset coverage.
        """
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()

            # Load from all splits
            for split in ("train", "validation", "test"):
                source = self._get_source_for_split(split)
                for doc in loader.load(source):
                    instance = self.process_doc(doc)
                    if instance is not None:
                        self._instances_cache.append(instance)

        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get the data source for a specific split.

        If config has a data_source configured, uses that with the specified split.
        Otherwise falls back to the default HuggingFace path with dataset_name.
        """
        try:
            # Try to use the configured data source
            return self.config.get_data_source(split=split).with_subset(self.dataset_name)
        except ValueError:
            # Fall back to default HuggingFace source
            return DataSource(
                path="allenai/ai2_arc",
                subset=self.dataset_name,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
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
        data_source=DataSource(
            path="allenai/ai2_arc",
            subset="ARC-Challenge",
        ),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


def _arc_easy_config() -> TaskConfig:
    return TaskConfig(
        name="arc_easy",
        data_source=DataSource(
            path="allenai/ai2_arc",
            subset="ARC-Easy",
        ),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
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
