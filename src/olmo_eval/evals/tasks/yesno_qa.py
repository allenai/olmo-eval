"""Yes/No QA task implementations (QASPER, SciRIFF)."""

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
from olmo_eval.evals.extract import extract_mcqa_answer
from olmo_eval.evals.tasks.core import Task, TaskConfig, register

# =============================================================================
# QASPER Yes/No
# =============================================================================


class QasperYesNoTask(Task):
    """QASPER Yes/No question answering task.

    Subset of yes/no questions from the QASPER dataset.
    See: A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers
    https://arxiv.org/abs/2105.03011

    QASPER is a dataset for question answering on scientific research papers.
    It consists of 5,049 questions over 1,585 Natural Language Processing papers.
    """

    hf_path: str = "allenai/qasper-yesno"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._instances_cache: list[Instance] | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the train split (default for this dataset)."""
        if self._instances_cache is None:
            self._instances_cache = []
            dataset = load_dataset(
                self.hf_path,
                split="train",
                trust_remote_code=True,
            )
            for doc in dataset:
                self._instances_cache.append(self._process_doc(doc))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        choices = ("Yes", "No")
        # Join evidence passages
        source = " ".join(doc["evidence"]) if doc["evidence"] else ""
        # gold_idx: 0 = Yes, 1 = No
        gold_idx = 0 if doc["answer"] == "Yes" else 1
        gold_answer = "A" if gold_idx == 0 else "B"

        return Instance(
            question=f"{source}\nQuestion: {doc['question']}",
            gold_answer=gold_answer,
            choices=choices,
            metadata={
                "id": doc["id"],
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx],
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        choices_text = "\n".join(
            f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(instance.choices or [])
        )
        prompt = f"{instance.question}\n\n{choices_text}\n\nAnswer:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
            continuations=(" A", " B"),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer letter from model output."""
        return extract_mcqa_answer(output.text, [r"[A-Ba-b]"])


def _qasper_yesno_config() -> TaskConfig:
    return TaskConfig(
        name="qasper_yesno",
        hf_dataset="allenai/qasper-yesno",
        formatter=MultipleChoiceFormatter(
            template="{question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer_name="multiple_choice"),),
    )


@register("qasper_yesno", _qasper_yesno_config)
class QasperYesNo(QasperYesNoTask):
    """QASPER Yes/No task."""

    pass


# =============================================================================
# SciRIFF Yes/No
# =============================================================================


class SciriffYesNoTask(Task):
    """SciRIFF Yes/No question answering task.

    Subset of yes/no questions from the SciRIFF dataset.
    See: SciRIFF: A Resource to Enhance Language Model Instruction-Following over Scientific Literature
    https://arxiv.org/abs/2406.07835

    SciRIFF is a dataset of 137K instruction-following demonstrations for 54 tasks
    covering five essential scientific literature understanding capabilities.
    """

    hf_path: str = "allenai/sciriff-yesno"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._instances_cache: list[Instance] | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the train split (default for this dataset)."""
        if self._instances_cache is None:
            self._instances_cache = []
            dataset = load_dataset(
                self.hf_path,
                split="train",
                trust_remote_code=True,
            )
            for doc in dataset:
                self._instances_cache.append(self._process_doc(doc))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        choices = ("Yes", "No")
        # gold_idx: 0 = Yes, 1 = No
        gold_idx = 0 if doc["answer"] == "Yes" else 1
        gold_answer = "A" if gold_idx == 0 else "B"

        return Instance(
            question=f"{doc['context']}\nQuestion: {doc['question']}",
            gold_answer=gold_answer,
            choices=choices,
            metadata={
                "id": doc["id"],
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx],
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        choices_text = "\n".join(
            f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(instance.choices or [])
        )
        prompt = f"{instance.question}\n\n{choices_text}\n\nAnswer:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
            continuations=(" A", " B"),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer letter from model output."""
        return extract_mcqa_answer(output.text, [r"[A-Ba-b]"])


def _sciriff_yesno_config() -> TaskConfig:
    return TaskConfig(
        name="sciriff_yesno",
        hf_dataset="allenai/sciriff-yesno",
        formatter=MultipleChoiceFormatter(
            template="{question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer_name="multiple_choice"),),
    )


@register("sciriff_yesno", _sciriff_yesno_config)
class SciriffYesNo(SciriffYesNoTask):
    """SciRIFF Yes/No task."""

    pass
