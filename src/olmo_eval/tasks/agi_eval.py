"""AGI-Eval task implementation."""

import re
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

from olmo_eval.core import (
    AccuracyMetric,
    Instance,
    LMOutput,
    LMRequest,
    MultipleChoiceFormatter,
    MultipleChoiceScorer,
    RequestType,
)
from olmo_eval.evals.constants.benchmarks import AGI_EVAL_ENGLISH_TASKS
from olmo_eval.extract import extract_mcqa_answer
from olmo_eval.tasks import Task, TaskConfig, register


class AGIEvalTask(Task):
    """AGI-Eval standardized test task."""

    hf_base: str = "https://raw.githubusercontent.com/ruixiangcui/AGIEval/main/data/v1"

    def __init__(self, config: TaskConfig, subset: str) -> None:
        super().__init__(config)
        self.subset = subset
        self._instances_cache: list[Instance] | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the dataset."""
        if self._instances_cache is None:
            self._instances_cache = []
            path = f"{self.hf_base}/{self.subset}.jsonl"
            dataset = load_dataset("json", data_files=path, split="train")
            for doc in dataset:
                self._instances_cache.append(self._process_doc(doc))
        yield from self._instances_cache

    def _strip_choice_prefix(self, choice: str) -> str:
        """Remove leading choice labels like (A), A., etc."""
        return re.sub(r"^\s*\([A-E]\)\s*|^\s*[A-E][.?]?\s*", "", choice)

    def _get_gold_completion(self, doc: dict[str, Any]) -> str:
        """Build the gold completion with explanation if available."""
        explanation = (doc.get("other") or {}).get("solution", "")
        if explanation:
            explanation = re.sub(r"\s+", " ", explanation).strip()
            return f"{explanation} Therefore, the answer is ({doc['label']})."
        return f"({doc['label']})"

    def _process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        passage = doc.get("passage")
        choices = tuple(self._strip_choice_prefix(c) for c in doc.get("options", []))

        question = f"{passage}\n\n{doc['question']}" if passage else doc["question"]

        label_idx = list("ABCDE").index(doc["label"])
        answer_key = doc["label"]

        return Instance(
            question=question,
            gold_answer=answer_key,
            choices=choices,
            metadata={
                "gold_idx": label_idx,
                "gold_text": choices[label_idx] if label_idx < len(choices) else None,
                "gold_completion": self._get_gold_completion(doc),
                "subset": self.subset,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        # Default: format as multiple choice question
        num_choices = len(instance.choices or [])
        choices_text = "\n".join(
            f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(instance.choices or [])
        )
        prompt = f"Question: {instance.question}\n\n{choices_text}\n\nAnswer:"
        continuations = tuple(f" {chr(ord('A') + i)}" for i in range(num_choices))
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
            continuations=continuations,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer letter from model output."""
        ans = extract_mcqa_answer(output.text, [r"[A-E]"])
        if ans and ans in list("ABCDE"):
            return ans
        # Fallback: find first A-E in output
        match = re.search(r"[A-E]", output.text)
        return match.group(0) if match else None


def _make_agi_eval_config(subset: str) -> TaskConfig:
    return TaskConfig(
        name=f"agi_eval_{subset.replace('-', '_')}",
        hf_dataset="AGIEval",
        hf_subsets=(subset,),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer_name="multiple_choice"),),
    )


# Register all AGI-Eval subsets
for subset in AGI_EVAL_ENGLISH_TASKS:
    subset_key = subset.replace("-", "_")

    def make_config_factory(s: str):
        return lambda: _make_agi_eval_config(s)

    def make_class_factory(s: str):
        class _AGIEvalSubset(AGIEvalTask):
            def __init__(self, config: TaskConfig) -> None:
                super().__init__(config, subset=s)

        _AGIEvalSubset.__name__ = f"AGIEval_{s.replace('-', '_').title()}"
        return _AGIEvalSubset

    register(f"agi_eval_{subset_key}", make_config_factory(subset))(make_class_factory(subset))
