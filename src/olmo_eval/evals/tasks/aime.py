"""AIME (American Invitational Mathematics Examination) task implementation."""

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

AIME_YEARS = ("2022", "2023", "2024", "2025")


class AIMETask(Task):
    """AIME (American Invitational Mathematics Examination) task."""

    default_hf_path: str = "allenai/aime-2021-2025"

    def __init__(self, config: TaskConfig, year: str | None = None) -> None:
        super().__init__(config)
        self.year = year

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the train split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("train")
            for doc in loader.load(source):
                instance = self.process_doc(doc)
                # Filter by year if specified
                if self.year is not None and instance.metadata.get("year") != self.year:
                    continue
                self._instances_cache.append(instance)
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
        # Extract year and AIME number from URL
        problem_from = doc.get("url", "").split("/")[-2]
        parts = problem_from.split("_")
        year = parts[0] if parts else ""
        aime_number = f"AIME_{parts[2]}" if len(parts) > 2 else ""

        return Instance(
            question=doc["problem"],
            gold_answer=str(doc["answer"]),
            metadata={
                "id": aime_number,
                "year": year,
                "full_solution": doc.get("solution", ""),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        # Default: problem format for chain-of-thought
        prompt = f"Problem: {instance.question}\n\nSolution:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the math answer from model output."""
        answers = extract_math_answer(output.text)
        return answers[0] if answers else None


def _aime_config() -> TaskConfig:
    return TaskConfig(
        name="aime",
        data_source=DataSource(path="allenai/aime-2021-2025"),
        scorers=(ExactMatchScorer(),),
        metrics=(AccuracyMetric(scorer=ExactMatchScorer),),
    )


def _make_aime_year_config(year: str) -> TaskConfig:
    return TaskConfig(
        name=f"aime_{year}",
        data_source=DataSource(path="allenai/aime-2021-2025"),
        scorers=(ExactMatchScorer(),),
        metrics=(AccuracyMetric(scorer=ExactMatchScorer),),
    )


@register("aime", _aime_config)
class AIME(AIMETask):
    """AIME task (all years)."""

    pass


# Register year-specific AIME tasks
for year in AIME_YEARS:

    def make_config_factory(y: str):
        return lambda: _make_aime_year_config(y)

    def make_class_factory(y: str):
        class _AIMEYear(AIMETask):
            def __init__(self, config: TaskConfig) -> None:
                super().__init__(config, year=y)

        _AIMEYear.__name__ = f"AIME_{y}"
        return _AIMEYear

    register(f"aime_{year}", make_config_factory(year))(make_class_factory(year))
