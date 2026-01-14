"""Basic Skills evaluation task implementations."""

import random
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
from olmo_eval.evals.tasks import Task, TaskConfig, register

# Subsets for basic skills tasks
BASIC_SKILLS_SUBSETS: tuple[str, ...] = (
    "arithmetic",
    "coding",
    "common_knowledge",
    "logical_reasoning",
    "string_operations",
    "pattern",
)


def _shuffle_and_insert(
    wrong_answers: list[str], correct_answer: str, seed: int
) -> tuple[list[str], int]:
    """Shuffle wrong answers and insert correct answer at random position.

    Uses deterministic randomness based on seed for reproducibility.
    """
    rnd = random.Random(seed)
    shuffled = wrong_answers.copy()
    rnd.shuffle(shuffled)
    insert_idx = rnd.randint(0, len(shuffled))
    shuffled.insert(insert_idx, correct_answer)
    return shuffled, insert_idx


class BasicSkillsTask(Task):
    """Base class for Basic Skills evaluation tasks."""

    hf_path: str = "allenai/basic-skills"

    def __init__(self, config: TaskConfig, subset: str) -> None:
        super().__init__(config)
        self.subset = subset
        self._instances_cache: list[Instance] | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the validation split."""
        if self._instances_cache is None:
            self._instances_cache = []
            dataset = load_dataset(
                self.hf_path,
                name=self.subset,
                split="validation",
                trust_remote_code=True,
            )
            for doc in dataset:
                self._instances_cache.append(self._process_doc(doc))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        question = doc["question"]
        answer = doc["answer"]
        wrong_answers = doc["wrong_answers"]

        # Shuffle and insert answer at random position (deterministic based on id)
        choices, gold_idx = _shuffle_and_insert(wrong_answers, answer, hash(doc["id"]))
        gold_answer = chr(ord("A") + gold_idx)

        return Instance(
            question=question,
            gold_answer=gold_answer,
            choices=tuple(choices),
            metadata={
                "id": doc["id"],
                "gold_idx": gold_idx,
                "gold_text": answer,
                "subset": self.subset,
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
        num_choices = 4  # Default to 4 choices (A-D)
        pattern = f"[A-{chr(ord('A') + num_choices - 1)}a-{chr(ord('a') + num_choices - 1)}]"
        return extract_mcqa_answer(output.text, [pattern])


def _make_basic_skills_config(subset: str) -> TaskConfig:
    """Create a Basic Skills task config for a specific subset."""
    return TaskConfig(
        name=f"basic_skills_{subset}",
        hf_dataset="allenai/basic-skills",
        hf_subsets=(subset,),
        formatter=MultipleChoiceFormatter(
            template="Question: {question}\n\nAnswer:",
        ),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer_name="multiple_choice"),),
    )


# Register all Basic Skills tasks dynamically
for subset in BASIC_SKILLS_SUBSETS:

    def make_config_factory(s: str):
        return lambda: _make_basic_skills_config(s)

    def make_class_factory(s: str):
        class _BasicSkillsSubset(BasicSkillsTask):
            def __init__(self, config: TaskConfig) -> None:
                super().__init__(config, subset=s)

        _BasicSkillsSubset.__name__ = f"BasicSkills_{s.title().replace('_', '')}"
        return _BasicSkillsSubset

    register(f"basic_skills_{subset}", make_config_factory(subset))(make_class_factory(subset))
