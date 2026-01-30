"""Minerva Math evaluation task implementations.

Implements the MATH benchmark (hendrycks_math) with Minerva-style evaluation,
as well as the MATH-500 subset.

References:
    - Hendrycks et al., "Measuring Mathematical Problem Solving With the MATH Dataset"
    - Lewkowycz et al., "Solving Quantitative Reasoning Problems with Language Models" (Minerva)
"""

from collections.abc import Iterator
from typing import Any

from olmo_eval.core import (
    AccuracyMetric,
    BitsPerByteScorer,
    BPBMetric,
    ExactMatchScorer,
    Instance,
    LMOutput,
    LMRequest,
    PPLFormatter,
    RequestType,
    SamplingParams,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.extract import MathExtractor
from olmo_eval.evals.tasks.core import Task, TaskConfig, register, register_variant


# =============================================================================
# MATH Subsets
# =============================================================================

MATH_SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


# =============================================================================
# Task Classes
# =============================================================================


class MinervaMathTask(Task):
    """MATH benchmark task with Minerva-style evaluation.

    The MATH dataset consists of 12,500 problems from mathematics competitions.
    This implementation follows the Minerva paper's evaluation methodology,
    extracting the final answer from \\boxed{} notation.

    Attributes:
        default_source: HuggingFace path to the MATH dataset.
        fewshot_split: Split to use for few-shot examples.
    """

    default_source: str = "EleutherAI/hendrycks_math"
    fewshot_split: str = "train"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._subset = self._get_subset_from_config()

    def _get_subset_from_config(self) -> str | None:
        """Extract subset from config data source."""
        if isinstance(self.config.data_source, DataSource):
            return self.config.data_source.subset
        elif isinstance(self.config.data_source, str):
            # Parse from URI like "hf://EleutherAI/hendrycks_math?subset=algebra"
            if "subset=" in self.config.data_source:
                import re

                match = re.search(r"subset=([^&]+)", self.config.data_source)
                if match:
                    return match.group(1)
        return None

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the test split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            for doc in loader.load(source):
                instance = self.process_doc(doc)
                if instance is not None:
                    self._instances_cache.append(instance)
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(
                path=self.default_source,
                subset=self._subset,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a dataset document to an Instance.

        Extracts the final answer from the solution using the MathExtractor.

        Args:
            doc: A document with 'problem', 'solution', 'level', and 'type' fields.
            index: Document index (unused).

        Returns:
            An Instance with the problem and extracted solution.
        """
        # Extract the final answer from the boxed solution
        solution_text = doc.get("solution", "")
        extracted_answers = MathExtractor.extract_answer(solution_text)
        extracted_answer = extracted_answers[0] if extracted_answers else None

        return Instance(
            question=doc["problem"],
            gold_answer=extracted_answer,
            metadata={
                "level": doc.get("level"),
                "type": doc.get("type"),
                "full_solution": solution_text,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request.

        Uses the configured formatter if available, otherwise creates
        a completion request with few-shot examples included.

        For few-shot examples, uses the full solution (with chain-of-thought
        reasoning and \\boxed{} notation) so the model learns the expected format.
        """
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        # Build prompt with few-shot examples
        fewshot = self.get_fewshot()
        parts: list[str] = []

        for ex in fewshot:
            # Use full solution (with chain-of-thought and \boxed{}) from metadata
            # so the model learns the expected format
            full_solution = ex.metadata.get("full_solution", ex.gold_answer)
            example = f"Problem: {ex.question}\nSolution: {full_solution}"
            parts.append(example)

        # Add current instance with solution prefix to prompt generation
        parts.append(f"Problem: {instance.question}\nSolution:")

        prompt = "\n\n".join(parts)

        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the mathematical answer from model output.

        Uses the MathExtractor to find boxed answers or other patterns
        in the model's response.

        Args:
            output: The model's output.

        Returns:
            The extracted answer string, or None if not found.
        """
        answers = MathExtractor.extract_answer(output.text)
        return answers[0] if answers else None


class Math500Task(MinervaMathTask):
    """MATH-500 benchmark task.

    A curated subset of 500 problems from the MATH dataset,
    provided by HuggingFace for more efficient evaluation.

    Note: MATH-500 only has a test split, so few-shot examples are drawn
    from the full MATH dataset's train split.
    """

    default_source: str = "HuggingFaceH4/MATH-500"

    def _get_subset_from_config(self) -> str | None:
        """MATH-500 doesn't use subsets."""
        return None

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split.

        For test split, uses MATH-500. For other splits (train/dev),
        uses the full MATH dataset since MATH-500 only has test.
        """
        # MATH-500 only has test split; use full MATH for train/dev
        if split != "test":
            return DataSource(
                path="EleutherAI/hendrycks_math",
                subset="algebra",  # Use algebra subset for few-shot
                split=split,
            )

        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(
                path=self.default_source,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a MATH-500 document to an Instance.

        MATH-500 uses 'answer' field directly instead of extracting from solution.

        Args:
            doc: A document with 'problem', 'solution', 'answer', etc.
            index: Document index (unused).

        Returns:
            An Instance with the problem and gold answer.
        """
        # MATH-500 provides the answer directly
        gold_answer = doc.get("answer")
        if gold_answer is None:
            # Fall back to extraction from solution
            solution_text = doc.get("solution", "")
            extracted_answers = MathExtractor.extract_answer(solution_text)
            gold_answer = extracted_answers[0] if extracted_answers else None

        return Instance(
            question=doc["problem"],
            gold_answer=gold_answer,
            metadata={
                "level": doc.get("level"),
                "type": doc.get("type", doc.get("subject")),
                "full_solution": doc.get("solution", ""),
            },
        )


# =============================================================================
# Task Configs
# =============================================================================


def _minerva_math_config(subset: str | None = None) -> TaskConfig:
    """Create config for Minerva MATH task.

    Args:
        subset: Optional subset name (e.g., "algebra", "geometry").

    Returns:
        TaskConfig for the MATH task.
    """
    data_source = DataSource(
        path="EleutherAI/hendrycks_math",
        subset=subset,
    )

    return TaskConfig(
        name=f"minerva_math_{subset}" if subset else "minerva_math",
        data_source=data_source,
        scorers=(ExactMatchScorer(case_sensitive=False),),
        metrics=(AccuracyMetric(),),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
        ),
    )


def _math500_config() -> TaskConfig:
    """Create config for MATH-500 task."""
    return TaskConfig(
        name="math500",
        data_source=DataSource(path="HuggingFaceH4/MATH-500"),
        scorers=(ExactMatchScorer(case_sensitive=False),),
        metrics=(AccuracyMetric(),),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
        ),
    )


# =============================================================================
# Task Registrations
# =============================================================================


# Register base minerva_math task (uses all subsets when no subset specified)
@register("minerva_math", lambda: _minerva_math_config(None))
class MinervaMath(MinervaMathTask):
    """Minerva MATH benchmark task - all subsets."""

    pass


# Register each MATH subset as a separate task
for _subset in MATH_SUBSETS:
    # Create a factory function that captures the subset value
    def _make_config(s: str = _subset) -> TaskConfig:
        return _minerva_math_config(s)

    # Dynamically create and register the task class
    _task_name = f"minerva_math_{_subset}"
    _task_class = type(
        f"MinervaMath_{_subset.title().replace('_', '')}",
        (MinervaMathTask,),
        {"default_source": "EleutherAI/hendrycks_math"},
    )
    register(_task_name, _make_config)(_task_class)


# Register MATH-500 task
@register("math500", _math500_config)
class Math500(Math500Task):
    """MATH-500 benchmark task."""

    pass


# =============================================================================
# Variant Registrations
# =============================================================================

# 4-shot variants (common for math tasks)
register_variant(
    "minerva_math",
    "4shot",
    num_fewshot=4,
    fewshot_seed=42,
)

register_variant(
    "math500",
    "4shot",
    num_fewshot=4,
    fewshot_seed=42,
)

# Register 4-shot variants for all subsets
for _subset in MATH_SUBSETS:
    _task_name = f"minerva_math_{_subset}"
    register_variant(
        _task_name,
        "4shot",
        num_fewshot=4,
        fewshot_seed=42,
    )

# =============================================================================
# BPB (Bits-Per-Byte) Variants - PPL-based evaluation
# =============================================================================
# These variants measure how well the model predicts the gold answer,
# rather than generating and extracting answers.
# Usage: math500:bpb, math500:4shot:bpb, minerva_math:bpb, etc.

register_variant(
    "minerva_math",
    "bpb",
    formatter=PPLFormatter(),
    scorers=(BitsPerByteScorer(),),
    metrics=(BPBMetric(),),
    primary_metric=BPBMetric(),
)

register_variant(
    "math500",
    "bpb",
    formatter=PPLFormatter(),
    scorers=(BitsPerByteScorer(),),
    metrics=(BPBMetric(),),
    primary_metric=BPBMetric(),
)

# Register BPB variants for all subsets
for _subset in MATH_SUBSETS:
    _task_name = f"minerva_math_{_subset}"
    register_variant(
        _task_name,
        "bpb",
        formatter=PPLFormatter(),
        scorers=(BitsPerByteScorer(),),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
    )
