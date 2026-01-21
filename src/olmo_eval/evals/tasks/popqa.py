"""PopQA task implementation."""

import json
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

# Template mappings for PopQA questions
POPQA_TEMPLATES = {
    22: "What is {}'s occupation?",
    218: "In what city was {} born?",
    91: "What genre is {}?",
    257: "Who is the father of {}?",
    182: "In what country is {}?",
    164: "Who was the producer of {}?",
    526: "Who was the director of {}?",
    97: "What is {} the capital of?",
    533: "Who was the screenwriter for {}?",
    639: "Who was the composer of {}?",
    472: "What color is {}?",
    106: "What is the religion of {}?",
    560: "What sport does {} play?",
    484: "Who is the author of {}?",
    292: "Who is the mother of {}?",
    422: "What is the capital of {}?",
}


class PopQAScorer(ExactMatchScorer):
    """Custom scorer for PopQA that checks against aliases."""

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score by checking if any alias matches the prediction."""
        if output.extracted_answer is None:
            return 0.0

        pred = str(output.extracted_answer).strip()
        aliases = instance.metadata.get("aliases", [])

        # Check if any alias appears in the prediction
        for alias in aliases:
            if alias in pred or alias.lower() in pred.lower() or alias.capitalize() in pred:
                return 1.0
        return 0.0


class PopQATask(Task):
    """PopQA question answering task.

    PopQA is a large-scale open-domain QA dataset for probing the factual knowledge
    of language models. Questions are derived from knowledge base triples.

    Citation:
    @article{mallen2023llm_memorization,
      title={When Not to Trust Language Models: Investigating Effectiveness and
             Limitations of Parametric and Non-Parametric Memories},
      author={Mallen, Alex and Asai, Akari and Zhong, Victor and Das, Rajarshi and
              Hajishirzi, Hannaneh and Khashabi, Daniel},
      journal={arXiv preprint},
      year={2022}
    }
    """

    hf_path: str = "akariasai/PopQA"

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
            for doc in dataset:
                self._instances_cache.append(self._process_doc(doc))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        # Parse aliases from JSON string
        try:
            aliases = json.loads(doc["possible_answers"])
        except (json.JSONDecodeError, TypeError):
            aliases = [doc["obj"]]

        return Instance(
            question=doc["question"],
            gold_answer=doc["obj"],
            metadata={
                "id": doc["id"],
                "template_id": doc["prop_id"],
                "aliases": aliases,
                "subject": doc.get("subj", ""),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        prompt = f"Q: {instance.question} A:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer from model output."""
        if output.text:
            # Take first line, strip whitespace
            answer = output.text.split("\n")[0].strip()
            return answer
        return None


def _popqa_config() -> TaskConfig:
    return TaskConfig(
        name="popqa",
        hf_dataset="akariasai/PopQA",
        scorers=(PopQAScorer(),),
        metrics=(AccuracyMetric(scorer=ExactMatchScorer),),
    )


@register("popqa", _popqa_config)
class PopQA(PopQATask):
    """PopQA task."""

    pass
