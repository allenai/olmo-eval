"""WikiText perplexity task implementation."""

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

from olmo_eval.core import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
)
from olmo_eval.evals.tasks import Task, TaskConfig, register


class WikiTextTask(Task):
    """WikiText perplexity evaluation task."""

    hf_path: str = "Salesforce/wikitext"
    dataset_name: str = "wikitext-103-v1"

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
                name=self.dataset_name,
                split="validation",
                trust_remote_code=True,
            )
            for doc in dataset:
                instance = self._process_doc(doc)
                if instance is not None:
                    self._instances_cache.append(instance)
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any]) -> Instance | None:
        """Convert a dataset document to an Instance."""
        text = doc["text"].strip()
        if not text:
            return None

        return Instance(
            question="",  # No question for perplexity tasks
            gold_answer=text,
            metadata={},
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request for perplexity computation."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        # For perplexity, we want to compute logprobs over the text
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


# Task configurations
# Note: Perplexity tasks require special handling for loglikelihood computation
# The configs below use empty scorers/metrics as perplexity is computed differently


def _wikitext_config() -> TaskConfig:
    return TaskConfig(
        name="wikitext",
        hf_dataset="Salesforce/wikitext",
        hf_subsets=("wikitext-103-v1",),
        scorers=(),  # Perplexity computed separately
        metrics=(),  # Perplexity metric to be added
    )


def _wikitext2_config() -> TaskConfig:
    return TaskConfig(
        name="wikitext2",
        hf_dataset="Salesforce/wikitext",
        hf_subsets=("wikitext-2-v1",),
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
