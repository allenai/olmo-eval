"""C4 perplexity task implementations."""

from collections.abc import Iterator
from typing import Any

from olmo_eval.core.metrics import CorpusPerplexityMetric
from olmo_eval.core.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.core import Task, TaskConfig, register, register_variant


class C4Task(Task):
    """C4 perplexity task."""

    # This is a version of C4 where the 5% longest documents have been removed
    default_source: str = "valentinhofmann/c4_short"
    default_split = "validation"
    default_subset = "full"  # Options are "full", "1k", "10k", "100k"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the test split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split(self.default_split)
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        return DataSource(
            path=self.default_source,
            split=split,
            subset=self.default_subset,
        )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        text = doc["text"]

        return Instance(
            question="",  # Context
            gold_answer=text,  # The text we score as the "continuation"
            metadata={
                "id": index,
                "num_chars": len(text),
                "num_words": len(text.strip().split()),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        gold = instance.gold_answer
        continuations = (gold,) if gold is not None else None
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=instance.question,
            continuations=continuations,
        )

    def extract_answer(self, output: LMOutput) -> str:
        # Not used for scoring
        return output.text


class C41KTask(C4Task):
    """C4 perplexity task on 1,000 randomly sampled documents."""

    default_subset: str = "1k"


class C410KTask(C4Task):
    """C4 perplexity task on 10,000 randomly sampled documents."""

    default_subset: str = "10k"


class C4100KTask(C4Task):
    """C4 perplexity task on 100,000 randomly sampled documents."""

    default_subset: str = "100k"


# =============================================================================
# Task Configs
# =============================================================================


def _c4_config() -> TaskConfig:
    return TaskConfig(
        name="c4",
        data_source=DataSource(path="valentinhofmann/c4_short", subset="full", split="validation"),
        metrics=(),
    )


def _c4_1k_config() -> TaskConfig:
    return TaskConfig(
        name="c4_1k",
        data_source=DataSource(path="valentinhofmann/c4_short", subset="1k", split="validation"),
        metrics=(),
    )


def _c4_10k_config() -> TaskConfig:
    return TaskConfig(
        name="c4_10k",
        data_source=DataSource(path="valentinhofmann/c4_short", subset="10k", split="validation"),
        metrics=(),
    )


def _c4_100k_config() -> TaskConfig:
    return TaskConfig(
        name="c4_100k",
        data_source=DataSource(path="valentinhofmann/c4_short", subset="100k", split="validation"),
        metrics=(),
    )


# =============================================================================
# Task Registrations
# =============================================================================


@register("c4", _c4_config)
class C4(C4Task):
    """C4 perplexity task."""

    pass


@register("c4_1k", _c4_1k_config)
class C41K(C41KTask):
    """C4 perplexity task on 1,000 randomly sampled documents."""

    pass


@register("c4_10k", _c4_10k_config)
class C410K(C410KTask):
    """C4 perplexity task on 10,000 randomly sampled documents."""

    pass


@register("c4_100k", _c4_100k_config)
class C4100K(C4100KTask):
    """C4 perplexity task on 100,000 randomly sampled documents."""

    pass


# =============================================================================
# Variant Registrations
# =============================================================================


register_variant(
    "c4",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)


register_variant(
    "c4_1k",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)


register_variant(
    "c4_10k",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)


register_variant(
    "c4_100k",
    "ppl",
    metrics=(CorpusPerplexityMetric(),),
)
