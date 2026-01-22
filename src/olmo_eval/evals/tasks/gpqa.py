"""GPQA task implementations."""

import random
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
from olmo_eval.evals.extract import extract_mcqa_answer
from olmo_eval.evals.tasks.core import Task, TaskConfig, register


def _preprocess(text: str | None) -> str:
    """Preprocess GPQA text by cleaning up formatting artifacts."""
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


class GPQATask(Task):
    """GPQA (Graduate-level Google-Proof Q&A) task."""

    default_hf_path: str = "Idavidrein/gpqa"
    default_subset: str = "gpqa_main"
    seed: int = 42

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the train split."""
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
                subset=self.default_subset,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        gold_answer = _preprocess(doc["Correct Answer"])
        choices = [
            _preprocess(doc["Incorrect Answer 1"]),
            _preprocess(doc["Incorrect Answer 2"]),
            _preprocess(doc["Incorrect Answer 3"]),
            gold_answer,
        ]

        # Shuffle choices with deterministic seed based on record ID
        random.Random(self.seed + hash(doc["Record ID"])).shuffle(choices)
        correct_answer_index = choices.index(gold_answer)
        answer_key = chr(ord("A") + correct_answer_index)

        return Instance(
            question=doc["Question"],
            gold_answer=answer_key,
            choices=tuple(choices),
            metadata={
                "id": doc["Record ID"],
                "gold_idx": correct_answer_index,
                "gold_text": gold_answer,
                "canary_string": doc.get("Canary String", ""),
                "explanation": doc.get("Explanation", ""),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        # Default: format as multiple choice question
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
        answer = extract_mcqa_answer(output.text, [r"\(?([A-D])\)?"])
        if answer in ["A", "B", "C", "D"]:
            return answer
        return None


class GPQADiamondTask(GPQATask):
    """GPQA Diamond task (harder subset)."""

    default_subset: str = "gpqa_diamond"


class SuperGPQATask(Task):
    """SuperGPQA task with broader coverage."""

    default_hf_path: str = "m-a-p/SuperGPQA"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the train split."""
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
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        gold_answer = doc["answer"]
        choices = doc["options"]
        correct_answer_index = choices.index(gold_answer)
        answer_key = chr(ord("A") + correct_answer_index)

        return Instance(
            question=doc["question"],
            gold_answer=answer_key,
            choices=tuple(choices),
            metadata={
                "id": doc["uuid"],
                "gold_idx": correct_answer_index,
                "gold_text": gold_answer,
                "discipline": doc.get("discipline", ""),
                "field": doc.get("field", ""),
                "subfield": doc.get("subfield", ""),
                "difficulty": doc.get("difficulty", ""),
                "is_calculation": doc.get("is_calculation", False),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        # Default: format as multiple choice question
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
        answer = extract_mcqa_answer(output.text, [r"\(?([A-D])\)?"])
        if answer in ["A", "B", "C", "D"]:
            return answer
        return None


def _gpqa_config() -> TaskConfig:
    return TaskConfig(
        name="gpqa",
        data_source=DataSource(path="Idavidrein/gpqa", subset="gpqa_main"),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


def _gpqa_diamond_config() -> TaskConfig:
    return TaskConfig(
        name="gpqa_diamond",
        data_source=DataSource(path="Idavidrein/gpqa", subset="gpqa_diamond"),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


def _super_gpqa_config() -> TaskConfig:
    return TaskConfig(
        name="super_gpqa",
        data_source=DataSource(path="m-a-p/SuperGPQA"),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


@register("gpqa", _gpqa_config)
class GPQA(GPQATask):
    """GPQA task (Graduate-level Google-Proof Q&A)."""

    pass


@register("gpqa_diamond", _gpqa_diamond_config)
class GPQADiamond(GPQADiamondTask):
    """GPQA Diamond task (harder subset)."""

    pass


@register("super_gpqa", _super_gpqa_config)
class SuperGPQA(SuperGPQATask):
    """SuperGPQA task."""

    pass
