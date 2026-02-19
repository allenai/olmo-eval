"""C4 perplexity task implementations."""

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import CorpusPerplexityMetric
from olmo_eval.common.types import (
    Instance,
    LMRequest,
    RequestType,
    Split,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant


class C4Base(Task):
    """Base class for C4 perplexity tasks."""

    split = Split.VALIDATION

    @property
    def request_type(self) -> RequestType:
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.LOGLIKELIHOOD

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the dataset."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self.config.get_data_source()
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

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
            request_type=self.request_type,
            prompt=instance.question,
            continuations=continuations,
        )


@register("c4")
class C4(C4Base):
    """C4 perplexity task."""

    data_source = DataSource(path="valentinhofmann/c4_short", subset="full")


@register("c4_1k")
class C41K(C4Base):
    """C4 perplexity task on 1,000 randomly sampled documents."""

    data_source = DataSource(path="valentinhofmann/c4_short", subset="1k")


@register("c4_10k")
class C410K(C4Base):
    """C4 perplexity task on 10,000 randomly sampled documents."""

    data_source = DataSource(path="valentinhofmann/c4_short", subset="10k")


@register("c4_100k")
class C4100K(C4Base):
    """C4 perplexity task on 100,000 randomly sampled documents."""

    data_source = DataSource(path="valentinhofmann/c4_short", subset="100k")


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
