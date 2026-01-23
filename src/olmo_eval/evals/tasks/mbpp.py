"""MBPP code generation task implementations."""

from collections.abc import Iterator
from typing import Any

from olmo_eval.core import (
    BitsPerByteScorer,
    BPBMetric,
    Instance,
    LMOutput,
    LMRequest,
    PPLFormatter,
    RequestType,
    SamplingParams,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.code import MBPP_STOP_SEQUENCES
from olmo_eval.evals.extract import extract_code
from olmo_eval.evals.tasks.core import Task, TaskConfig, register, register_variant


class MBPPTask(Task):
    """MBPP (Mostly Basic Python Problems) task."""

    default_source: str = "google-research-datasets/mbpp"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the test split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(
                path=self.default_source,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        # Build prompt from text and function signature
        question = doc["text"].strip() + "\n" + doc["code"].split(":")[0] + ":"

        # Build test code
        tests = doc.get("test_setup_code", "") or ""
        if tests:
            tests += "\n"
        tests += "\n".join(doc["test_list"])

        return Instance(
            question=question,
            gold_answer=doc["code"],
            metadata={
                "id": doc["task_id"],
                "answer_prefix": question,
                "test": tests,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract code from model output."""
        code = extract_code(output.text)
        if code and "answer_prefix" in (output.metadata or {}):
            return output.metadata["answer_prefix"] + code
        return code

    def _build_fewshot(self) -> list[Instance]:
        """Build few-shot examples from the prompt split.

        MBPP has a dedicated 'prompt' split with 10 examples for few-shot prompting.
        Falls back to 'train' split if 'prompt' is not available.
        """
        return self._build_fewshot_from_source(
            split="prompt",
            sample=True,
            fallback_splits=["train"],
        )


class MBPPPlusTask(Task):
    """MBPP+ task with additional test cases."""

    default_source: str = "evalplus/mbppplus"
    fewshot_split: str = "test"  # MBPP+ doesn't have a dedicated prompt split

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the test split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(
                path=self.default_source,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        # Build prompt from text and function signature
        question = doc["prompt"].strip() + doc["code"].split(":")[0] + ":"

        # Build test code
        tests = doc.get("test_setup_code", "") or ""
        if tests:
            tests += "\n"
        tests += doc["test"]

        return Instance(
            question=question,
            gold_answer=doc["code"],
            metadata={
                "id": doc["task_id"],
                "answer_prefix": question,
                "test": tests,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract code from model output."""
        code = extract_code(output.text)
        if code and "answer_prefix" in (output.metadata or {}):
            return output.metadata["answer_prefix"] + code
        return code


# =============================================================================
# Task Configs
# =============================================================================


def _mbpp_config() -> TaskConfig:
    return TaskConfig(
        name="mbpp",
        data_source=DataSource(path="google-research-datasets/mbpp"),
        scorers=(),
        metrics=(),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            stop_sequences=MBPP_STOP_SEQUENCES,
        ),
    )


def _mbpp_plus_config() -> TaskConfig:
    return TaskConfig(
        name="mbpp_plus",
        data_source=DataSource(path="evalplus/mbppplus"),
        scorers=(),
        metrics=(),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            stop_sequences=MBPP_STOP_SEQUENCES,
        ),
    )


# =============================================================================
# Task Registrations
# =============================================================================


@register("mbpp", _mbpp_config)
class MBPP(MBPPTask):
    """MBPP code generation task."""

    pass


@register("mbpp_plus", _mbpp_plus_config)
class MBPPPlus(MBPPPlusTask):
    """MBPP+ code generation task."""

    pass


# =============================================================================
# Variant Registrations
# =============================================================================

# BPB variant - use mbpp:bpb or mbpp_plus:bpb
register_variant(
    "mbpp",
    "bpb",
    formatter=PPLFormatter(leading_space=False),
    scorers=(BitsPerByteScorer(),),
    metrics=(BPBMetric(),),
    primary_metric=BPBMetric(),
)

register_variant(
    "mbpp_plus",
    "bpb",
    formatter=PPLFormatter(leading_space=False),
    scorers=(BitsPerByteScorer(),),
    metrics=(BPBMetric(),),
    primary_metric=BPBMetric(),
)

# 3shot variant - composable with bpb (e.g., mbpp:3shot:bpb)
register_variant(
    "mbpp",
    "3shot",
    num_fewshot=3,
)

register_variant(
    "mbpp_plus",
    "3shot",
    num_fewshot=3,
)
