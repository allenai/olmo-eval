"""Medical QA task implementations (MedQA)."""

from collections.abc import Iterator
from typing import Any

from olmo_eval.core import (
    AccuracyMetric,
    Instance,
    LMOutput,
    LMRequest,
    MultipleChoiceFormatter,
    MultipleChoiceScorer,
    RequestType,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.extract import extract_mcqa_answer
from olmo_eval.evals.tasks.core import Task, TaskConfig, register

# =============================================================================
# MedQA (English)
# =============================================================================


class MedQAEnTask(Task):
    """MedQA English medical question answering task.

    A large-scale open domain medical question answering dataset from medical exams.
    See: https://huggingface.co/datasets/davidheineman/medqa-en

    Citation:
    @article{jin2021disease,
      title={What disease does this patient have?
             A large-scale open domain question answering dataset from medical exams},
      author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung
              and Fang, Hanyi and Szolovits, Peter},
      journal={Applied Sciences},
      volume={11},
      number={14},
      pages={6421},
      year={2021},
      publisher={MDPI}
    }
    """

    default_hf_path: str = "davidheineman/medqa-en"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the test split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            for idx, doc in enumerate(loader.load(source)):
                self._instances_cache.append(self.process_doc(doc, idx))
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

    def process_doc(self, doc: dict[str, Any], index: int) -> Instance:
        """Convert a dataset document to an Instance."""
        # Dataset fields: question (str), choices (list of str), answer_idx (int), answer (str)
        choices = tuple(doc["choices"])
        gold_idx = doc["answer_idx"]
        gold_answer = chr(ord("A") + gold_idx)

        return Instance(
            question=doc["question"],
            gold_answer=gold_answer,
            choices=choices,
            metadata={
                "index": index,
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx] if gold_idx < len(choices) else None,
                "answer": doc.get("answer", ""),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        choices_text = "\n".join(
            f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(instance.choices or [])
        )
        prompt = f"Question: {instance.question}\n\n{choices_text}\n\nAnswer:"
        num_choices = len(instance.choices or [])
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
            continuations=tuple(f" {chr(ord('A') + i)}" for i in range(num_choices)),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer letter from model output."""
        # MedQA can have up to 5 choices (A-E)
        return extract_mcqa_answer(output.text, [r"[A-Ea-e]"])


def _medqa_en_config() -> TaskConfig:
    return TaskConfig(
        name="medqa_en",
        data_source=DataSource(path="davidheineman/medqa-en"),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


@register("medqa_en", _medqa_en_config)
class MedQAEn(MedQAEnTask):
    """MedQA English task."""

    pass
