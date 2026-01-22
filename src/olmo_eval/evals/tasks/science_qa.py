"""Science QA task implementations (SciQ)."""

import random
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

# =============================================================================
# SciQ
# =============================================================================


class SciQTask(Task):
    """SciQ multiple choice science questions task.

    Crowdsourcing Multiple Choice Science Questions
    https://aclanthology.org/W17-4413.pdf

    The SciQ dataset contains 13,679 crowdsourced science exam questions about Physics,
    Chemistry and Biology, among others. The questions are in multiple-choice format
    with 4 answer options each. For the majority of the questions, an additional paragraph
    with supporting evidence for the correct answer is provided.

    Homepage: https://allenai.org/data/sciq
    """

    default_hf_path: str = "sciq"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the validation split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("validation")
            for idx, doc in enumerate(loader.load(source)):
                self._instances_cache.append(self.process_doc(doc, idx))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(
                path=self.default_hf_path,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int) -> Instance:
        """Convert a dataset document to an Instance.

        Note: The correct answer is always at index 3 (last position) in the
        base format. The MC variant shuffles choices.
        """
        choices = (
            doc["distractor1"],
            doc["distractor2"],
            doc["distractor3"],
            doc["correct_answer"],
        )
        # Correct answer is always the last one (index 3)
        gold_idx = 3
        gold_answer = "D"

        # Include support paragraph if available
        support = doc.get("support", "")
        question = doc["question"]
        if support:
            question = f"{support}\n\nQuestion: {question}"

        return Instance(
            question=question,
            gold_answer=gold_answer,
            choices=choices,
            metadata={
                "index": index,
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx],
                "support": support,
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
            continuations=(" A", " B", " C", " D"),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer letter from model output."""
        return extract_mcqa_answer(output.text, [r"[A-Da-d]"])


class SciQMCTask(SciQTask):
    """SciQ with shuffled multiple choice answers.

    This variant shuffles the answer choices using a deterministic seed
    based on the question index, so that the correct answer isn't always D.
    """

    def _process_doc(self, doc: dict[str, Any], index: int) -> Instance:
        """Convert a dataset document to an Instance with shuffled choices."""
        # Create deterministic RNG for this document
        rng = random.Random(index)

        choices = [
            doc["distractor1"],
            doc["distractor2"],
            doc["distractor3"],
            doc["correct_answer"],
        ]

        # Shuffle choices
        num_choices = len(choices)
        positions = list(range(num_choices))
        rng.shuffle(positions)
        shuffled_choices = tuple(choices[i] for i in positions)

        # Find where the correct answer ended up
        gold_idx = positions.index(3)  # Original correct answer was at index 3
        gold_answer = chr(ord("A") + gold_idx)

        # Include support paragraph if available
        support = doc.get("support", "")
        question = doc["question"]
        if support:
            question = f"{support}\n\nQuestion: {question}"

        return Instance(
            question=question,
            gold_answer=gold_answer,
            choices=shuffled_choices,
            metadata={
                "index": index,
                "gold_idx": gold_idx,
                "gold_text": shuffled_choices[gold_idx],
                "support": support,
            },
        )


def _sciq_config() -> TaskConfig:
    return TaskConfig(
        name="sciq",
        data_source=DataSource(path="sciq"),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


@register("sciq", _sciq_config)
class SciQ(SciQTask):
    """SciQ task."""

    pass
