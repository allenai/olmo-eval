"""LAMBADA task implementation."""

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

from olmo_eval.core import (
    AccuracyMetric,
    ExactMatchScorer,
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
)
from olmo_eval.evals.tasks.core import Task, TaskConfig, register

# =============================================================================
# LAMBADA
# =============================================================================


class LambadaTask(Task):
    """LAMBADA word prediction task.

    The LAMBADA dataset tests the ability of language models to predict the
    final word of a passage, requiring understanding of broad context.

    The task uses loglikelihood scoring - given the context (all words except
    the last), the model computes the probability of the target word.

    Citation:
    @misc{
        author={Paperno, Denis and Kruszewski, Germán and Lazaridou, Angeliki and
                Pham, Quan Ngoc and Bernardi, Raffaella and Pezzelle, Sandro and
                Baroni, Marco and Boleda, Gemma and Fernández, Raquel},
        title={The LAMBADA dataset},
        DOI={10.5281/zenodo.2630551},
        publisher={Zenodo},
        year={2016},
        month={Aug}
    }
    """

    hf_path: str = "EleutherAI/lambada_openai"

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
                split="test",
                trust_remote_code=True,
            )
            for idx, doc in enumerate(dataset):
                self._instances_cache.append(self._process_doc(doc, idx))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any], index: int) -> Instance:
        """Convert a dataset document to an Instance.

        Splits the passage into context (all but last word) and target (last word).
        """
        text = doc["text"]
        words = text.split()
        context = " ".join(words[:-1])
        target = words[-1]

        return Instance(
            question=context,
            gold_answer=target,
            choices=(target,),  # Single choice for loglikelihood
            metadata={
                "id": index,
                "full_text": text,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request.

        Uses loglikelihood request type - computes probability of target
        continuation given context.
        """
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        # For loglikelihood, we provide the context and the target continuation
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=instance.question,
            continuations=(f" {instance.gold_answer}",),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer from model output.

        For loglikelihood tasks, we return the generated text stripped.
        """
        if output.text:
            return output.text.strip()
        return None


def _lambada_config() -> TaskConfig:
    return TaskConfig(
        name="lambada",
        hf_dataset="EleutherAI/lambada_openai",
        scorers=(ExactMatchScorer(),),
        metrics=(AccuracyMetric(scorer=ExactMatchScorer),),
    )


@register("lambada", _lambada_config)
class Lambada(LambadaTask):
    """LAMBADA task."""

    pass
