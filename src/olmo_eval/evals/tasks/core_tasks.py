"""Core evaluation task implementations."""

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
# BoolQ
# =============================================================================


class BoolQTask(Task):
    """BoolQ yes/no question answering task."""

    hf_path: str = "super_glue"
    hf_subset: str = "boolq"

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
                name=self.hf_subset,
                split="validation",
                trust_remote_code=True,
            )
            for doc in dataset:
                self._instances_cache.append(self._process_doc(doc))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        choices = ("yes", "no")
        # BoolQ label: 1 = yes (True), 0 = no (False)
        # gold_idx: 0 = yes, 1 = no
        gold_idx = 0 if doc["label"] else 1
        gold_answer = "A" if gold_idx == 0 else "B"

        return Instance(
            question=f"{doc['passage']}\nQuestion: {doc['question']}?",
            gold_answer=gold_answer,
            choices=choices,
            metadata={
                "idx": doc["idx"],
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
        prompt = f"Question: {instance.question}\n\n{choices_text}\n\nAnswer:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
            continuations=(" A", " B"),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer letter from model output."""
        return extract_mcqa_answer(output.text, [r"[A-Ba-b]"])


def _boolq_config() -> TaskConfig:
    return TaskConfig(
        name="boolq",
        hf_dataset="super_glue",
        hf_subsets=("boolq",),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


@register("boolq", _boolq_config)
class BoolQ(BoolQTask):
    """BoolQ task."""

    pass


# =============================================================================
# CommonsenseQA
# =============================================================================


class CSQATask(Task):
    """CommonsenseQA multiple choice task."""

    hf_path: str = "commonsense_qa"

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
        choices = tuple(doc["choices"]["text"])
        gold_idx = ["A", "B", "C", "D", "E"].index(doc["answerKey"])

        return Instance(
            question=doc["question"],
            gold_answer=doc["answerKey"],
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


def _csqa_config() -> TaskConfig:
    return TaskConfig(
        name="csqa",
        hf_dataset="commonsense_qa",
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


@register("csqa", _csqa_config)
class CSQA(CSQATask):
    """CommonsenseQA task."""

    pass


# =============================================================================
# OpenBookQA
# =============================================================================


class OpenBookQATask(Task):
    """OpenBookQA multiple choice task."""

    hf_path: str = "openbookqa"
    hf_subset: str = "main"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._instances_cache: list[Instance] | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the test split."""
        if self._instances_cache is None:
            self._instances_cache = []
            dataset = load_dataset(
                self.hf_path,
                name=self.hf_subset,
                split="test",
                trust_remote_code=True,
            )
            for doc in dataset:
                self._instances_cache.append(self._process_doc(doc))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        choices = tuple(doc["choices"]["text"])
        gold_idx = ["A", "B", "C", "D"].index(doc["answerKey"].strip())

        return Instance(
            question=doc["question_stem"],
            gold_answer=doc["answerKey"].strip(),
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


def _openbookqa_config() -> TaskConfig:
    return TaskConfig(
        name="openbookqa",
        hf_dataset="openbookqa",
        hf_subsets=("main",),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


@register("openbookqa", _openbookqa_config)
class OpenBookQA(OpenBookQATask):
    """OpenBookQA task."""

    pass


# =============================================================================
# PIQA (Physical IQA)
# =============================================================================


class PIQATask(Task):
    """PIQA (Physical Interaction Question Answering) task."""

    hf_path: str = "piqa"

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
            for idx, doc in enumerate(dataset):
                self._instances_cache.append(self._process_doc(doc, idx))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any], index: int) -> Instance:
        """Convert a dataset document to an Instance."""
        choices = (doc["sol1"], doc["sol2"])
        gold_idx = doc["label"]
        gold_answer = "A" if gold_idx == 0 else "B"

        return Instance(
            question=doc["goal"],
            gold_answer=gold_answer,
            choices=choices,
            metadata={
                "index": index,
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx] if gold_idx < len(choices) else None,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        choices_text = "\n".join(
            f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(instance.choices or [])
        )
        prompt = f"Goal: {instance.question}\n\n{choices_text}\n\nAnswer:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
            continuations=(" A", " B"),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer letter from model output."""
        return extract_mcqa_answer(output.text, [r"[A-Ba-b]"])


def _piqa_config() -> TaskConfig:
    return TaskConfig(
        name="piqa",
        hf_dataset="piqa",
        formatter=MultipleChoiceFormatter(
            template="Goal: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


@register("piqa", _piqa_config)
class PIQA(PIQATask):
    """PIQA task."""

    pass


# =============================================================================
# SocialIQA
# =============================================================================


class SocialIQATask(Task):
    """SocialIQA (Social Interaction Question Answering) task."""

    hf_path: str = "social_i_qa"

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
            for idx, doc in enumerate(dataset):
                self._instances_cache.append(self._process_doc(doc, idx))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any], index: int) -> Instance:
        """Convert a dataset document to an Instance."""
        choices = (doc["answerA"], doc["answerB"], doc["answerC"])
        # Label is 1-indexed in the dataset
        gold_idx = int(doc["label"]) - 1
        gold_answer = chr(ord("A") + gold_idx)

        return Instance(
            question=f"{doc['context']} {doc['question']}",
            gold_answer=gold_answer,
            choices=choices,
            metadata={
                "index": index,
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx] if gold_idx < len(choices) else None,
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
            continuations=(" A", " B", " C"),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer letter from model output."""
        return extract_mcqa_answer(output.text, [r"[A-Ca-c]"])


def _socialiqa_config() -> TaskConfig:
    return TaskConfig(
        name="socialiqa",
        hf_dataset="social_i_qa",
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


@register("socialiqa", _socialiqa_config)
class SocialIQA(SocialIQATask):
    """SocialIQA task."""

    pass


# =============================================================================
# WinoGrande
# =============================================================================


class WinoGrandeTask(Task):
    """WinoGrande coreference resolution task."""

    hf_path: str = "winogrande"
    hf_subset: str = "winogrande_xl"

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
                name=self.hf_subset,
                split="validation",
                trust_remote_code=True,
            )
            for idx, doc in enumerate(dataset):
                self._instances_cache.append(self._process_doc(doc, idx))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any], index: int) -> Instance:
        """Convert a dataset document to an Instance."""
        choices = (doc["option1"], doc["option2"])
        # Answer is 1-indexed ("1" or "2")
        gold_idx = int(doc["answer"]) - 1 if doc["answer"] else -1
        gold_answer = "A" if gold_idx == 0 else "B"

        # Replace underscore with blank for display
        question = doc["sentence"].replace("_", "___")

        return Instance(
            question=question,
            gold_answer=gold_answer,
            choices=choices,
            metadata={
                "index": index,
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx] if 0 <= gold_idx < len(choices) else None,
                "sentence": doc["sentence"],
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        choices_text = "\n".join(
            f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(instance.choices or [])
        )
        prompt = f"Fill in the blank: {instance.question}\n\n{choices_text}\n\nAnswer:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
            continuations=(" A", " B"),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer letter from model output."""
        return extract_mcqa_answer(output.text, [r"[A-Ba-b]"])


def _winogrande_config() -> TaskConfig:
    return TaskConfig(
        name="winogrande",
        hf_dataset="winogrande",
        hf_subsets=("winogrande_xl",),
        formatter=MultipleChoiceFormatter(
            template="Fill in the blank: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


@register("winogrande", _winogrande_config)
class WinoGrande(WinoGrandeTask):
    """WinoGrande task."""

    pass
