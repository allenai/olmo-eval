"""WikiText perplexity task implementation."""

from collections.abc import Iterator
from typing import Any

from olmo_eval.core import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.core import Task, TaskConfig, register


class WikiTextTask(Task):
    """WikiText perplexity evaluation task."""

    dataset_name: str = "wikitext-103-v1"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the validation split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("validation")
            for doc in loader.load(source):
                instance = self.process_doc(doc)
                if instance is not None:
                    self._instances_cache.append(instance)
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split).with_subset(self.dataset_name)
        except ValueError:
            return DataSource(
                path="Salesforce/wikitext",
                subset=self.dataset_name,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any]) -> Instance | None:
        """Convert a dataset document to an Instance."""
        text = doc["text"].strip()
        if not text:
            return None

        return Instance(
            question="",
            gold_answer=text,
            metadata={},
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request for perplexity computation."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=instance.gold_answer or "",
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """No answer extraction needed for perplexity tasks."""
        return None


class WikiText2Task(WikiTextTask):
    """WikiText-2 perplexity evaluation task."""

    dataset_name: str = "wikitext-2-v1"


def _wikitext_config() -> TaskConfig:
    return TaskConfig(
        name="wikitext",
        data_source=DataSource(
            path="Salesforce/wikitext",
            subset="wikitext-103-v1",
        ),
        scorers=(),
        metrics=(),
    )


def _wikitext2_config() -> TaskConfig:
    return TaskConfig(
        name="wikitext2",
        data_source=DataSource(
            path="Salesforce/wikitext",
            subset="wikitext-2-v1",
        ),
        scorers=(),
        metrics=(),
    )


@register("wikitext", _wikitext_config)
class WikiText(WikiTextTask):
    """WikiText-103 perplexity task."""

    pass


@register("wikitext2", _wikitext2_config)
class WikiText2(WikiText2Task):
    """WikiText-2 perplexity task."""

    pass
