"""LAB-Bench task implementations."""

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

# LAB-Bench subsets that don't require images or long context
LAB_BENCH_SUBSETS = ("ProtocolQA", "DbQA")

# =============================================================================
# LAB-Bench Base Classes
# =============================================================================


class LabBenchTask(Task):
    """LAB-Bench: The Language Agent Biology Benchmark.

    A benchmark for foundational scientific reasoning in biology, with 8 broad categories.
    https://huggingface.co/datasets/futurehouse/lab-bench

    Citation:
    @article{futurehouse2024labbench,
      title={LAB-Bench: The Language Agent Biology Benchmark},
      author={Future House},
      journal={arXiv preprint arXiv:2407.10362},
      year={2024}
    }
    """

    default_hf_path: str = "futurehouse/lab-bench"
    default_subset: str = ""  # Set by subclasses

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the train split (only split available)."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("train")
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(
                path=self.default_hf_path,
                subset=self.default_subset if self.default_subset else None,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance.

        The correct answer (ideal) is appended to distractors, so gold_idx
        is always len(distractors).
        """
        distractors = doc.get("distractors", [])
        ideal = doc.get("ideal", "")
        choices = tuple(distractors) + (ideal,)
        gold_idx = len(distractors)
        gold_answer = chr(ord("A") + gold_idx)

        return Instance(
            question=doc["question"],
            gold_answer=gold_answer,
            choices=choices,
            metadata={
                "id": doc["id"],
                "gold_idx": gold_idx,
                "gold_text": ideal,
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
        num_choices = len(instance.choices or [])
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
            continuations=tuple(f" {chr(ord('A') + i)}" for i in range(num_choices)),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer letter from model output."""
        return extract_mcqa_answer(output.text, [r"[A-Ea-e]"])


class LabBenchMCTask(LabBenchTask):
    """LAB-Bench with shuffled multiple choice answers.

    This variant shuffles the answer choices using a deterministic seed
    based on the document ID.
    """

    def _process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance with shuffled choices."""
        # Create deterministic RNG for this document
        rng = random.Random(doc["id"])

        distractors = doc.get("distractors", [])
        ideal = doc.get("ideal", "")
        choices = list(distractors) + [ideal]
        original_gold_idx = len(distractors)

        # Shuffle choices
        num_choices = len(choices)
        positions = list(range(num_choices))
        rng.shuffle(positions)
        shuffled_choices = tuple(choices[i] for i in positions)

        # Find where the correct answer ended up
        gold_idx = positions.index(original_gold_idx)
        gold_answer = chr(ord("A") + gold_idx)

        return Instance(
            question=doc["question"],
            gold_answer=gold_answer,
            choices=shuffled_choices,
            metadata={
                "id": doc["id"],
                "gold_idx": gold_idx,
                "gold_text": shuffled_choices[gold_idx],
            },
        )


# =============================================================================
# ProtocolQA
# =============================================================================


def _lab_bench_protocolqa_config() -> TaskConfig:
    return TaskConfig(
        name="lab_bench_protocolqa",
        data_source=DataSource(path="futurehouse/lab-bench", subset="ProtocolQA"),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


@register("lab_bench_protocolqa", _lab_bench_protocolqa_config)
class LabBenchProtocolQA(LabBenchTask):
    """LAB-Bench ProtocolQA task."""

    default_subset: str = "ProtocolQA"


# =============================================================================
# DbQA
# =============================================================================


def _lab_bench_dbqa_config() -> TaskConfig:
    return TaskConfig(
        name="lab_bench_dbqa",
        data_source=DataSource(path="futurehouse/lab-bench", subset="DbQA"),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


@register("lab_bench_dbqa", _lab_bench_dbqa_config)
class LabBenchDbQA(LabBenchTask):
    """LAB-Bench DbQA task."""

    default_subset: str = "DbQA"
