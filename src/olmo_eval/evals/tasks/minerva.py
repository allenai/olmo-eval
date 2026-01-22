"""Minerva Math task implementations."""

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

MINERVA_SUBSETS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


class MinervaMathTask(Task):
    """Minerva Math task (Hendrycks MATH dataset)."""

    default_hf_path: str = "EleutherAI/hendrycks_math"

    def __init__(self, config: TaskConfig, subset: str) -> None:
        super().__init__(config)
        self.subset = subset

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

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split).with_subset(self.subset)
        except ValueError:
            return DataSource(
                path=self.default_hf_path,
                subset=self.subset,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        # Extract the answer from the solution
        answers = extract_math_answer(doc["solution"])
        solution = answers[0] if answers else doc["solution"]

        return Instance(
            question=doc["problem"],
            gold_answer=solution,
            metadata={
                "level": doc.get("level"),
                "type": doc.get("type"),
                "full_solution": doc["solution"],
                "subset": self.subset,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        # Default: simple problem format for chain-of-thought
        prompt = f"Problem: {instance.question}\n\nSolution:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the math answer from model output."""
        answers = extract_math_answer(output.text)
        return answers[0] if answers else None


class Math500Task(MinervaMathTask):
    """MATH-500 task (subset of MATH dataset)."""

    default_hf_path: str = "HuggingFaceH4/MATH-500"

    def __init__(self, config: TaskConfig) -> None:
        # Math500 doesn't have subsets
        Task.__init__(self, config)

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
        answers = extract_math_answer(doc["solution"])
        solution = answers[0] if answers else doc["solution"]

        return Instance(
            question=doc["problem"],
            gold_answer=solution,
            metadata={
                "level": doc.get("level"),
                "type": doc.get("type"),
                "full_solution": doc["solution"],
            },
        )


def _make_minerva_config(subset: str) -> TaskConfig:
    return TaskConfig(
        name=f"minerva_math_{subset}",
        data_source=DataSource(path="EleutherAI/hendrycks_math", subset=subset),
        scorers=(ExactMatchScorer(),),
        metrics=(AccuracyMetric(scorer=ExactMatchScorer),),
    )


def _math500_config() -> TaskConfig:
    return TaskConfig(
        name="math500",
        data_source=DataSource(path="HuggingFaceH4/MATH-500"),
        scorers=(ExactMatchScorer(),),
        metrics=(AccuracyMetric(scorer=ExactMatchScorer),),
    )


# Register all Minerva subsets
for subset in MINERVA_SUBSETS:

    def make_config_factory(s: str):
        return lambda: _make_minerva_config(s)

    def make_class_factory(s: str):
        class _MinervaSubset(MinervaMathTask):
            def __init__(self, config: TaskConfig) -> None:
                super().__init__(config, subset=s)

        _MinervaSubset.__name__ = f"MinervaMath_{s.title().replace('_', '')}"
        return _MinervaSubset

    register(f"minerva_math_{subset}", make_config_factory(subset))(make_class_factory(subset))


@register("math500", _math500_config)
class Math500(Math500Task):
    """MATH-500 task."""

    pass


def _minerva_math_500_config() -> TaskConfig:
    """Alias config for minerva_math_500 (same as math500)."""
    return TaskConfig(
        name="minerva_math_500",
        data_source=DataSource(path="HuggingFaceH4/MATH-500"),
        scorers=(ExactMatchScorer(),),
        metrics=(AccuracyMetric(scorer=ExactMatchScorer),),
    )


@register("minerva_math_500", _minerva_math_500_config)
class MinervaMath500(Math500Task):
    """Minerva MATH-500 task (alias for math500)."""

    pass
