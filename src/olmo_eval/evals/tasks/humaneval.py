"""HumanEval code generation task implementations."""

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
from olmo_eval.evals.constants.code import HUMANEVAL_STOP_SEQUENCES
from olmo_eval.evals.extract import extract_code
from olmo_eval.evals.tasks.core import Task, TaskConfig, register


class HumanEvalTask(Task):
    """HumanEval code generation task."""

    hf_path: str = "openai_humaneval"

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
        # Format prompt with code fence
        prompt = "```python\n" + doc["prompt"]

        # Build test code
        unit_tests = doc["test"] + f"\ncheck({doc['entry_point']})"

        return Instance(
            question=prompt,
            gold_answer=doc["canonical_solution"],
            metadata={
                "id": doc["task_id"],
                "entry_point": doc["entry_point"],
                "answer_prefix": doc["prompt"],
                "test": unit_tests,
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
            # Prepend the prompt to the generated code
            return output.metadata["answer_prefix"] + code
        return code


class HumanEvalPlusTask(HumanEvalTask):
    """HumanEval+ task with additional test cases."""

    hf_path: str = "evalplus/humanevalplus"


# =============================================================================
# Generative Task Configs (with sampling_params)
# =============================================================================


def _humaneval_config() -> TaskConfig:
    return TaskConfig(
        name="humaneval",
        hf_dataset="openai_humaneval",
        scorers=(),
        metrics=(),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            stop_sequences=HUMANEVAL_STOP_SEQUENCES,
        ),
    )


def _humaneval_plus_config() -> TaskConfig:
    return TaskConfig(
        name="humaneval_plus",
        hf_dataset="evalplus/humanevalplus",
        scorers=(),
        metrics=(),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            stop_sequences=HUMANEVAL_STOP_SEQUENCES,
        ),
    )


def _codex_humaneval_config() -> TaskConfig:
    return TaskConfig(
        name="codex_humaneval",
        hf_dataset="openai_humaneval",
        scorers=(),
        metrics=(),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            stop_sequences=HUMANEVAL_STOP_SEQUENCES,
        ),
    )


def _codex_humanevalplus_config() -> TaskConfig:
    return TaskConfig(
        name="codex_humanevalplus",
        hf_dataset="evalplus/humanevalplus",
        scorers=(),
        metrics=(),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            stop_sequences=HUMANEVAL_STOP_SEQUENCES,
        ),
    )


# =============================================================================
# BPB Task Configs (no sampling_params)
# =============================================================================


def _humaneval_bpb_config() -> TaskConfig:
    return TaskConfig(
        name="humaneval:bpb",
        hf_dataset="openai_humaneval",
        formatter=PPLFormatter(),
        scorers=(BitsPerByteScorer(),),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
    )


def _humaneval_plus_bpb_config() -> TaskConfig:
    return TaskConfig(
        name="humaneval_plus:bpb",
        hf_dataset="evalplus/humanevalplus",
        formatter=PPLFormatter(),
        scorers=(BitsPerByteScorer(),),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
    )


def _codex_humaneval_bpb_config() -> TaskConfig:
    return TaskConfig(
        name="codex_humaneval:bpb",
        hf_dataset="openai_humaneval",
        formatter=PPLFormatter(),
        scorers=(BitsPerByteScorer(),),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
    )


def _codex_humanevalplus_bpb_config() -> TaskConfig:
    return TaskConfig(
        name="codex_humanevalplus:bpb",
        hf_dataset="evalplus/humanevalplus",
        formatter=PPLFormatter(),
        scorers=(BitsPerByteScorer(),),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
    )


# =============================================================================
# Task Registrations - Generative
# =============================================================================


@register("humaneval", _humaneval_config)
class HumanEval(HumanEvalTask):
    """HumanEval code generation task."""

    pass


@register("humaneval_plus", _humaneval_plus_config)
class HumanEvalPlus(HumanEvalPlusTask):
    """HumanEval+ code generation task."""

    pass


@register("codex_humaneval", _codex_humaneval_config)
class CodexHumanEval(HumanEvalTask):
    """HumanEval code generation task."""

    pass


@register("codex_humanevalplus", _codex_humanevalplus_config)
class CodexHumanEvalPlus(HumanEvalPlusTask):
    """HumanEval+ code generation task."""

    pass


# =============================================================================
# Task Registrations - BPB
# =============================================================================


@register("humaneval:bpb", _humaneval_bpb_config)
class HumanEvalBPB(HumanEvalTask):
    """HumanEval BPB evaluation task."""

    pass


@register("humaneval_plus:bpb", _humaneval_plus_bpb_config)
class HumanEvalPlusBPB(HumanEvalPlusTask):
    """HumanEval+ BPB evaluation task."""

    pass


@register("codex_humaneval:bpb", _codex_humaneval_bpb_config)
class CodexHumanEvalBPB(HumanEvalTask):
    """CodexHumanEval BPB evaluation task."""

    pass


@register("codex_humanevalplus:bpb", _codex_humanevalplus_bpb_config)
class CodexHumanEvalPlusBPB(HumanEvalPlusTask):
    """CodexHumanEvalPlus BPB evaluation task."""

    pass
