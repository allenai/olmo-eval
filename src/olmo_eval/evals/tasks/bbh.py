"""BIG-Bench Hard task implementations."""

import re
from collections.abc import Iterator
from typing import Any

from olmo_eval.core import (
    AccuracyMetric,
    ExactMatchScorer,
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.benchmarks import BBH_TASKS
from olmo_eval.evals.tasks.core import Task, TaskConfig, register

# Answer extraction regex patterns for each BBH task type
# "MC" indicates multiple choice tasks where answer is in format "(A)", "(B)", etc.
BBH_ANSWER_REGEX: dict[str, str] = {
    "boolean_expressions": r"[tT]rue|[fF]alse",
    "causal_judgement": r"[yY]es|[nN]o",
    "date_understanding": r"\([A-F]\)",
    "disambiguation_qa": r"\([A-C]\)",
    "dyck_languages": r"[\]\)\}\> ]+",
    "formal_fallacies": r"[iI]nvalid|[vV]alid",
    "geometric_shapes": r"\([A-K]\)",
    "hyperbaton": r"\([A-B]\)",
    "logical_deduction_five_objects": r"\([A-E]\)",
    "logical_deduction_seven_objects": r"\([A-G]\)",
    "logical_deduction_three_objects": r"\([A-C]\)",
    "movie_recommendation": r"\([A-E]\)",
    "multistep_arithmetic_two": r"-?\d+",
    "navigate": r"[nN]o|[yY]es",
    "object_counting": r"\d+",
    "penguins_in_a_table": r"\([A-E]\)",
    "reasoning_about_colored_objects": r"\([A-R]\)",
    "ruin_names": r"\([A-D]\)",
    "salient_translation_error_detection": r"\([A-F]\)",
    "snarks": r"\([A-B]\)",
    "sports_understanding": r"[yY]es|[nN]o",
    "temporal_sequences": r"\([A-D]\)",
    "tracking_shuffled_objects_five_objects": r"\([A-E]\)",
    "tracking_shuffled_objects_seven_objects": r"\([A-G]\)",
    "tracking_shuffled_objects_three_objects": r"\([A-C]\)",
    "web_of_lies": r"[yY]es|[nN]o",
    "word_sorting": r"[a-z ]+",
}


class BBHTask(Task):
    """Base class for BIG-Bench Hard tasks."""

    default_hf_path: str = "lukaemon/bbh"

    def __init__(self, config: TaskConfig, subset: str) -> None:
        super().__init__(config)
        self.subset = subset

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
            return self.config.get_data_source(split=split).with_subset(self.subset)
        except ValueError:
            return DataSource(
                path=self.default_hf_path,
                subset=self.subset,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int) -> Instance:
        """Convert a dataset document to an Instance."""
        # Extract the answer from the target field
        # Format: "Let's think step by step. ... So the answer is X."
        answer_match = re.search(r"(?<=answer is )(.*)(?=\.)", doc["target"])
        answer = answer_match.group(0).strip() if answer_match else doc["target"]

        return Instance(
            question=doc["input"],
            gold_answer=answer,
            metadata={
                "index": index,
                "solution": doc["target"],
                "subset": self.subset,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        # Default: Chain-of-thought prompting style
        prompt = f"Q: {instance.question}\nA: Let's think step by step."
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer from model output using task-specific regex."""
        # Try to find "the answer is X" pattern first
        match = re.search(r"(?i)(?:the )?answer is[:\s]*(.+?)(?:\.|$)", output.text)
        if match:
            answer = match.group(1).strip()
            # Clean up common artifacts
            answer = re.sub(r"^[\s\"\']|[\s\"\']$", "", answer)
            return answer

        # Fall back to task-specific regex
        answer_regex = BBH_ANSWER_REGEX.get(self.subset, r"\([A-Z]\)")
        matches = re.findall(answer_regex, output.text, re.IGNORECASE)
        if matches:
            return matches[-1]  # Return last match

        return None


def _make_bbh_config(subset: str) -> TaskConfig:
    """Create a BBH task config for a specific subset."""
    return TaskConfig(
        name=f"bbh_{subset}",
        data_source=DataSource(path="lukaemon/bbh", subset=subset),
        scorers=(ExactMatchScorer(case_sensitive=False),),
        metrics=(AccuracyMetric(scorer=ExactMatchScorer),),
    )


# Register all BBH tasks dynamically
for subset in BBH_TASKS:

    def make_config_factory(s: str):
        return lambda: _make_bbh_config(s)

    def make_class_factory(s: str):
        class _BBHSubset(BBHTask):
            def __init__(self, config: TaskConfig) -> None:
                super().__init__(config, subset=s)

        _BBHSubset.__name__ = f"BBH_{s.title().replace('_', '')}"
        return _BBHSubset

    register(f"bbh_{subset}", make_config_factory(subset))(make_class_factory(subset))
