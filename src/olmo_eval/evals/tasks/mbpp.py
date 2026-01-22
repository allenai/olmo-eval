"""MBPP code generation task implementations."""

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

from olmo_eval.core import (
    BPBMetric,
    BitsPerByteScorer,
    Instance,
    LMOutput,
    LMRequest,
    PPLFormatter,
    RequestType,
    SamplingParams,
)
from olmo_eval.evals.constants.code import MBPP_STOP_SEQUENCES
from olmo_eval.evals.extract import extract_code
from olmo_eval.evals.tasks.core import Task, TaskConfig, register


class MBPPTask(Task):
    """MBPP (Mostly Basic Python Problems) task."""

    hf_path: str = "google-research-datasets/mbpp"

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


class MBPPPlusTask(Task):
    """MBPP+ task with additional test cases."""

    hf_path: str = "evalplus/mbppplus"

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
# Generative Task Configs (with sampling_params)
# =============================================================================


def _mbpp_config() -> TaskConfig:
    return TaskConfig(
        name="mbpp",
        hf_dataset="google-research-datasets/mbpp",
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
        hf_dataset="evalplus/mbppplus",
        scorers=(),
        metrics=(),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            stop_sequences=MBPP_STOP_SEQUENCES,
        ),
    )


def _mbppplus_config() -> TaskConfig:
    return TaskConfig(
        name="mbppplus",
        hf_dataset="evalplus/mbppplus",
        scorers=(),
        metrics=(),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            stop_sequences=MBPP_STOP_SEQUENCES,
        ),
    )


# =============================================================================
# BPB Task Configs (no sampling_params)
# =============================================================================


def _mbpp_bpb_config() -> TaskConfig:
    return TaskConfig(
        name="mbpp:bpb",
        hf_dataset="google-research-datasets/mbpp",
        formatter=PPLFormatter(),
        scorers=(BitsPerByteScorer(),),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
    )


def _mbpp_plus_bpb_config() -> TaskConfig:
    return TaskConfig(
        name="mbpp_plus:bpb",
        hf_dataset="evalplus/mbppplus",
        formatter=PPLFormatter(),
        scorers=(BitsPerByteScorer(),),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
    )


def _mbppplus_bpb_config() -> TaskConfig:
    return TaskConfig(
        name="mbppplus:bpb",
        hf_dataset="evalplus/mbppplus",
        formatter=PPLFormatter(),
        scorers=(BitsPerByteScorer(),),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
    )


# =============================================================================
# Task Registrations - Generative
# =============================================================================


@register("mbpp", _mbpp_config)
class MBPP(MBPPTask):
    """MBPP code generation task."""

    pass


@register("mbpp_plus", _mbpp_plus_config)
class MBPPPlus(MBPPPlusTask):
    """MBPP+ code generation task."""

    pass


@register("mbppplus", _mbppplus_config)
class MBPPPlusAlias(MBPPPlusTask):
    """MBPP+ code generation task."""

    pass


# =============================================================================
# Task Registrations - BPB
# =============================================================================


@register("mbpp:bpb", _mbpp_bpb_config)
class MBPPBPB(MBPPTask):
    """MBPP BPB evaluation task."""

    pass


@register("mbpp_plus:bpb", _mbpp_plus_bpb_config)
class MBPPPlusBPB(MBPPPlusTask):
    """MBPP+ BPB evaluation task."""

    pass


@register("mbppplus:bpb", _mbppplus_bpb_config)
class MBPPPlusAliasBPB(MBPPPlusTask):
    """MBPPPlus BPB evaluation task."""

    pass
