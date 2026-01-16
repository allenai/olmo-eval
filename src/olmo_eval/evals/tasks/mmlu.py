"""MMLU task implementations."""

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
from olmo_eval.evals.extract import extract_mcqa_answer
from olmo_eval.evals.tasks.core import Task, TaskConfig, register

# fmt: off
MMLU_SUBSETS = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management",
    "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "virology", "world_religions",
)

MMLU_PRO_SUBSETS = (
    "math", "health", "physics", "business", "biology", "chemistry", "computer science",
    "economics", "engineering", "philosophy", "other", "history", "psychology", "law",
)
# fmt: on


class MMLUTask(Task):
    """MMLU (Massive Multitask Language Understanding) task."""

    hf_path: str = "cais/mmlu"

    def __init__(self, config: TaskConfig, subset: str) -> None:
        super().__init__(config)
        self.subset = subset
        self._instances_cache: list[Instance] | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the test split."""
        if self._instances_cache is None:
            self._instances_cache = []
            dataset = load_dataset(
                self.hf_path,
                name=self.subset,
                split="test",
                trust_remote_code=True,
            )
            for doc in dataset:
                self._instances_cache.append(self._process_doc(doc))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        gold_idx = doc["answer"]
        choices = tuple(doc["choices"])
        answer_key = chr(ord("A") + gold_idx)

        return Instance(
            question=doc["question"],
            gold_answer=answer_key,
            choices=choices,
            metadata={
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx] if gold_idx < len(choices) else None,
                "subset": self.subset,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        # Default: format as multiple choice question
        choices_text = "\n".join(
            f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(instance.choices or [])
        )
        prompt = f"Question: {instance.question}\n\n{choices_text}\n\nAnswer:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
            continuations=(" A", " B", " C", " D"),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer letter from model output."""
        return extract_mcqa_answer(output.text, [r"[A-D]"])

    def _build_fewshot(self) -> list[Instance]:
        """Build few-shot examples from the dev split."""
        dataset = load_dataset(
            self.hf_path,
            name=self.subset,
            split="dev",
            trust_remote_code=True,
        )
        return [self._process_doc(doc) for doc in dataset]


class MMLUProTask(Task):
    """MMLU-Pro task with 10 answer choices."""

    hf_path: str = "TIGER-Lab/MMLU-Pro"

    def __init__(self, config: TaskConfig, subset: str | None = None) -> None:
        super().__init__(config)
        self.subset = subset
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
                # Filter by subset if specified
                if self.subset is not None and doc.get("category") != self.subset:
                    continue
                self._instances_cache.append(self._process_doc(doc))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        gold_idx = doc["answer_index"]
        choices = tuple(doc["options"])
        answer_key = chr(ord("A") + gold_idx)

        return Instance(
            question=doc["question"],
            gold_answer=answer_key,
            choices=choices,
            metadata={
                "id": doc["question_id"],
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx] if gold_idx < len(choices) else None,
                "src": doc.get("src", ""),
                "subset": doc.get("category", ""),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        # MMLU-Pro has up to 10 choices (A-J)
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
        return extract_mcqa_answer(output.text, [r"[A-J]"])

    def _build_fewshot(self) -> list[Instance]:
        """Build few-shot examples from the validation split."""
        dataset = load_dataset(
            self.hf_path,
            split="validation",
            trust_remote_code=True,
        )
        instances = []
        for doc in dataset:
            if self.subset is not None and doc.get("category") != self.subset:
                continue
            instances.append(self._process_doc(doc))
        return instances


def _make_mmlu_config(subset: str) -> TaskConfig:
    return TaskConfig(
        name=f"mmlu_{subset}",
        hf_dataset="cais/mmlu",
        hf_subsets=(subset,),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer_name="multiple_choice"),),
    )


def _make_mmlu_pro_config(subset: str | None = None) -> TaskConfig:
    name = f"mmlu_pro_{subset}" if subset else "mmlu_pro"
    return TaskConfig(
        name=name,
        hf_dataset="TIGER-Lab/MMLU-Pro",
        hf_subsets=(subset,) if subset else None,
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer_name="multiple_choice"),),
    )


# Register all MMLU subsets
for subset in MMLU_SUBSETS:
    # Create a factory function that captures the subset
    def make_config_factory(s: str):
        return lambda: _make_mmlu_config(s)

    def make_class_factory(s: str):
        class _MMLUSubset(MMLUTask):
            def __init__(self, config: TaskConfig) -> None:
                super().__init__(config, subset=s)

        _MMLUSubset.__name__ = f"MMLU_{s.title().replace('_', '')}"
        return _MMLUSubset

    register(f"mmlu_{subset}", make_config_factory(subset))(make_class_factory(subset))


# Register all MMLU-Pro subsets
for subset in MMLU_PRO_SUBSETS:
    subset_key = subset.replace(" ", "_")

    def make_pro_config_factory(s: str):
        return lambda: _make_mmlu_pro_config(s)

    def make_pro_class_factory(s: str):
        class _MMLUProSubset(MMLUProTask):
            def __init__(self, config: TaskConfig) -> None:
                super().__init__(config, subset=s)

        _MMLUProSubset.__name__ = f"MMLUPro_{s.title().replace(' ', '')}"
        return _MMLUProSubset

    register(f"mmlu_pro_{subset_key}", make_pro_config_factory(subset))(
        make_pro_class_factory(subset)
    )
