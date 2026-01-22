"""Fill-in-the-Middle (FIM) code completion task implementations.

Based on: https://arxiv.org/abs/2207.14255
"Efficient Training of Language Models to Fill in the Middle"

The FIM paper created benchmarks for infilling by adapting the HumanEval dataset
to mask portions of code; solutions should infill the masked parts.

There are three primary subsets:
- Single: Single line masking
- Multi: Multiple line masking
- Random: Random span masking

Original dataset: https://github.com/openai/human-eval
Infilling dataset: https://github.com/openai/human-eval-infilling
HuggingFace: loubnabnl/humaneval_infilling
"""

from collections.abc import Iterator
from typing import Any

from olmo_eval.core import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.code import FIM_CONFIGS, SANTACODER_FIM, FIMConfig
from olmo_eval.evals.tasks.core import Task, TaskConfig, register


class HumanEvalFIMTask(Task):
    """Base class for HumanEval Fill-in-the-Middle tasks.

    FIM tasks present prompts following the pattern:
    <lead_token>prefix<center_token>suffix<end_token>

    The model should generate the middle content that fills the gap between
    prefix and suffix.

    Attributes:
        default_hf_path: HuggingFace dataset path.
        default_subset: Dataset subset name.
        fim_config: FIM token configuration to use.
    """

    default_hf_path: str = "loubnabnl/humaneval_infilling"
    default_subset: str = "HumanEval-SingleLineInfilling"

    def __init__(
        self,
        config: TaskConfig,
        fim_config: FIMConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.fim_config = fim_config or SANTACODER_FIM

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
                subset=self.default_subset,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance.

        The prompt is formatted as:
        <lead_token>prefix<center_token>suffix<end_token>
        """
        prefix = doc["prompt"]
        suffix = doc["suffix"]

        # Format FIM prompt
        query = (
            self.fim_config.lead_token
            + prefix
            + self.fim_config.center_token
            + suffix
            + self.fim_config.end_token
        )

        # Build test code
        unit_tests = doc["test"] + f"\ncheck({doc['entry_point']})"

        return Instance(
            question=query,
            gold_answer=doc["canonical_solution"],
            metadata={
                "index": index,
                "task_id": doc["task_id"],
                "prefix": prefix,
                "suffix": suffix,
                "entry_point": doc["entry_point"],
                "test": unit_tests,
                "fim_config": self.fim_config.lead_token,  # For identification
                "stop_sequences": list(self.fim_config.stop_sequences),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request.

        Note: Stop sequences are stored in instance metadata and should be
        applied by the runner/backend. Access via instance.metadata["stop_sequences"].
        """
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the generated code from model output.

        For FIM tasks, the model output is the middle portion that fills
        the gap between prefix and suffix.
        """
        if not output.text:
            return None

        # Clean up the output - remove any trailing special tokens
        text = output.text.strip()

        # Remove common stop sequences that might be included
        for stop in self.fim_config.stop_sequences:
            if text.endswith(stop):
                text = text[: -len(stop)].strip()

        return text

    def assemble_code(self, instance: Instance, generated_middle: str) -> str:
        """Assemble the complete code from prefix + generated middle + suffix.

        This is useful for code execution testing.
        """
        prefix = instance.metadata.get("prefix", "")
        suffix = instance.metadata.get("suffix", "")
        return prefix + generated_middle + suffix


class HumanEvalFIMSingleTask(HumanEvalFIMTask):
    """HumanEval FIM task with single-line masking.

    1k rows derived by masking single lines from 164 HumanEval problems.
    """

    default_subset: str = "HumanEval-SingleLineInfilling"


class HumanEvalFIMMultiTask(HumanEvalFIMTask):
    """HumanEval FIM task with multi-line masking.

    5.8k rows derived by masking multiple lines from 164 HumanEval problems.
    """

    default_subset: str = "HumanEval-MultiLineInfilling"


class HumanEvalFIMRandomTask(HumanEvalFIMTask):
    """HumanEval FIM task with random span masking.

    1.6k rows derived by masking random spans from 164 HumanEval problems.
    """

    default_subset: str = "HumanEval-RandomSpanInfilling"


# =============================================================================
# Task Configurations
# =============================================================================


def _fim_single_config() -> TaskConfig:
    return TaskConfig(
        name="humanevalfim_single",
        data_source=DataSource(
            path="loubnabnl/humaneval_infilling",
            subset="HumanEval-SingleLineInfilling",
        ),
        scorers=(),
        metrics=(),
    )


def _fim_multi_config() -> TaskConfig:
    return TaskConfig(
        name="humanevalfim_multi",
        data_source=DataSource(
            path="loubnabnl/humaneval_infilling",
            subset="HumanEval-MultiLineInfilling",
        ),
        scorers=(),
        metrics=(),
    )


def _fim_random_config() -> TaskConfig:
    return TaskConfig(
        name="humanevalfim_random",
        data_source=DataSource(
            path="loubnabnl/humaneval_infilling",
            subset="HumanEval-RandomSpanInfilling",
        ),
        scorers=(),
        metrics=(),
    )


# =============================================================================
# Task Registration
# =============================================================================


@register("humanevalfim_single", _fim_single_config)
class HumanEvalFIMSingle(HumanEvalFIMSingleTask):
    """HumanEval FIM single-line infilling task."""

    pass


@register("humanevalfim_multi", _fim_multi_config)
class HumanEvalFIMMulti(HumanEvalFIMMultiTask):
    """HumanEval FIM multi-line infilling task."""

    pass


@register("humanevalfim_random", _fim_random_config)
class HumanEvalFIMRandom(HumanEvalFIMRandomTask):
    """HumanEval FIM random span infilling task."""

    pass


# =============================================================================
# Convenience aliases for backwards compatibility
# =============================================================================

# These match the task names used in oe-eval-internal and olmo-cookbook
FIM_SINGLE = HumanEvalFIMSingle
FIM_MULTI = HumanEvalFIMMulti
FIM_RANDOM = HumanEvalFIMRandom

# Export FIM configurations for external use
__all__ = [
    "HumanEvalFIMTask",
    "HumanEvalFIMSingle",
    "HumanEvalFIMMulti",
    "HumanEvalFIMRandom",
    "FIM_SINGLE",
    "FIM_MULTI",
    "FIM_RANDOM",
    "FIM_CONFIGS",
    "FIMConfig",
    "SANTACODER_FIM",
]
