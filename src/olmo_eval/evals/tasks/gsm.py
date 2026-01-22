"""GSM (Grade School Math) task implementations."""

import re
from collections.abc import Iterator
from typing import Any

from olmo_eval.core import (
    AccuracyMetric,
    ExactMatchScorer,
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.extract import extract_math_answer
from olmo_eval.evals.tasks.core import Task, TaskConfig, register


def _clean_answer(answer: str) -> str:
    """Clean and normalize a numeric answer."""
    output = re.sub(r"(\d),(\d)", r"\1\2", answer)
    numbers = re.findall(r"[-+]?\d*\.?\d+", output)
    if numbers:
        return numbers[-1]
    return output


class GSM8KTask(Task):
    """GSM8K (Grade School Math 8K) task."""

    default_hf_path: str = "gsm8k"
    default_subset: str = "main"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from train and test splits."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            for split in ("train", "test"):
                source = self._get_source_for_split(split)
                for doc in loader.load(source):
                    self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split).with_subset(self.default_subset)
        except ValueError:
            return DataSource(
                path=self.default_hf_path,
                subset=self.default_subset,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        answer_text = doc["answer"]
        short_answer = answer_text.split("####")[-1].strip()

        return Instance(
            question=doc["question"],
            gold_answer=short_answer,
            metadata={
                "short_answer": short_answer,
                "full_answer": answer_text,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        prompt = f"Question: {instance.question}\n\nAnswer:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the numeric answer from model output."""
        answers = extract_math_answer(output.text)
        if answers:
            return _clean_answer(answers[0])
        return None


class GSMPlusTask(GSM8KTask):
    """GSM-Plus task (extended GSM8K)."""

    default_hf_path: str = "qintongli/GSM-Plus"

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the test split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache


class GSMSymbolicTask(GSM8KTask):
    """GSM-Symbolic task."""

    default_hf_path: str = "apple/GSM-Symbolic"

    def __init__(self, config: TaskConfig, split: str = "main") -> None:
        super().__init__(config)
        self.split = split

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the specified split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split(self.split)
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache


def _gsm8k_config() -> TaskConfig:
    return TaskConfig(
        name="gsm8k",
        data_source=DataSource(path="gsm8k", subset="main"),
        scorers=(ExactMatchScorer(),),
        metrics=(AccuracyMetric(scorer=ExactMatchScorer),),
    )


def _gsm_plus_config() -> TaskConfig:
    return TaskConfig(
        name="gsm_plus",
        data_source=DataSource(path="qintongli/GSM-Plus", subset="main"),
        scorers=(ExactMatchScorer(),),
        metrics=(AccuracyMetric(scorer=ExactMatchScorer),),
    )


def _gsm_symbolic_main_config() -> TaskConfig:
    return TaskConfig(
        name="gsm_symbolic",
        data_source=DataSource(path="apple/GSM-Symbolic", subset="main"),
        scorers=(ExactMatchScorer(),),
        metrics=(AccuracyMetric(scorer=ExactMatchScorer),),
    )


def _gsm_symbolic_p1_config() -> TaskConfig:
    return TaskConfig(
        name="gsm_symbolic_p1",
        data_source=DataSource(path="apple/GSM-Symbolic", subset="main"),
        scorers=(ExactMatchScorer(),),
        metrics=(AccuracyMetric(scorer=ExactMatchScorer),),
    )


def _gsm_symbolic_p2_config() -> TaskConfig:
    return TaskConfig(
        name="gsm_symbolic_p2",
        data_source=DataSource(path="apple/GSM-Symbolic", subset="main"),
        scorers=(ExactMatchScorer(),),
        metrics=(AccuracyMetric(scorer=ExactMatchScorer),),
    )


@register("gsm8k", _gsm8k_config)
class GSM8K(GSM8KTask):
    """GSM8K task (Grade School Math 8K)."""

    pass


@register("gsm_plus", _gsm_plus_config)
class GSMPlus(GSMPlusTask):
    """GSM-Plus task."""

    pass


@register("gsm_symbolic", _gsm_symbolic_main_config)
class GSMSymbolicMain(GSMSymbolicTask):
    """GSM-Symbolic main split."""

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config, split="main")


@register("gsm_symbolic_p1", _gsm_symbolic_p1_config)
class GSMSymbolicP1(GSMSymbolicTask):
    """GSM-Symbolic p1 split."""

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config, split="p1")


@register("gsm_symbolic_p2", _gsm_symbolic_p2_config)
class GSMSymbolicP2(GSMSymbolicTask):
    """GSM-Symbolic p2 split."""

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config, split="p2")
