"""C4 perplexity task implementations."""

from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.core import (
    BitsPerByteScorer,
    BPBMetric,
    CorpusPerplexityMetric,
    Instance,
    LMOutput,
    LMRequest,
    PerplexityScorer,
    RequestType,
    Response,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.core import Task, TaskConfig, register, register_variant


class C4Task(Task):
    """C4 perplexity task."""

    default_source: str = "allenai/c4"
    default_data_files = {"validation": "en/c4-validation.*.json.gz"}

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the test split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_from_data_files(self.default_data_files)
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def _get_source_from_data_files(self, data_files: dict) -> DataSource:
        """Get data source from data files."""
        return DataSource(
            path=self.default_source,
            data_files=data_files,
        )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        text = (doc.get("text") or "").strip()
        if not text:
            return None

        return Instance(
            question="",          # context
            gold_answer=text,     # the text we score as the "continuation"
            metadata={
                "id": index,
                "timestamp": doc.get("timestamp"),
                "url": doc.get("url"),
                "num_chars": len(text),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=instance.question,
            continuations=(instance.gold_answer,),
        )


    def extract_answer(self, output: LMOutput) -> str:
        # Not used for scoring PPL/BPB; this is just the input text again.
        return output.text


# =============================================================================
# Task Configs
# =============================================================================


def _c4_config() -> TaskConfig:
    return TaskConfig(
        name="c4",
        data_source=DataSource(path="allenai/c4", subset="en", split="validation"),
        scorers=(),
        metrics=(),
    )


# =============================================================================
# Task Registrations
# =============================================================================


@register("c4", _c4_config)
class C4(C4Task):
    """C4 perplexity task."""
    pass


# =============================================================================
# Variant Registrations
# =============================================================================


register_variant(
    "c4",
    "ppl",
    scorers=(PerplexityScorer(),),
    metrics=(CorpusPerplexityMetric(),),
    primary_metric=CorpusPerplexityMetric(),
)
