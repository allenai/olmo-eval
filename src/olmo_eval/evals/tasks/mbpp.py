"""MBPP code generation task implementations."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import CompletionFormatter, PPLFormatter
from olmo_eval.common.metrics import BPBMetricByteAvg, BPBMetricInstanceAvg, PassAtKMetric
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.code import MBPP_STOP_SEQUENCES, OLMO3_MBPP_STOP_SEQUENCES
from olmo_eval.evals.extract import extract_code, extract_code_before_fence
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.mbpp import MBPP_FEWSHOT_SOURCES


@dataclass(frozen=True, slots=True)
class CodeExecutionScorer3s(CodeExecutionScorer):
    separator: str = "\n"


class MBPPBase(Task):
    """Base class for MBPP (Mostly Basic Python Problems) tasks."""

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

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        # Build prompt from text and function signature
        func_sig = doc["code"].split(":")[0] + ":"
        question = doc["text"].strip() + "\n" + func_sig

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
                "answer_prefix": func_sig,
                "test": tests,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        return LMRequest(
            request_type=self.request_type,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract code from model output."""
        return extract_code(output.text)

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        """Extract code and prepend the function signature.

        The answer_prefix contains the function signature (e.g. ``def func(x):``).
        Prepending it ensures the generated body is wrapped in a valid function
        definition so that test assertions can call the function.
        """
        for response in responses:
            for output in response.outputs:
                code = self.extract_answer(output)
                if code:
                    output.extracted_answer = response.instance.metadata["answer_prefix"] + code
                else:
                    output.extracted_answer = None

    def _build_fewshot(self) -> list[Instance]:
        """Build few-shot examples from the prompt split.

        MBPP has a dedicated 'prompt' split with 10 examples for few-shot prompting.
        Falls back to 'train' split if 'prompt' is not available.

        Uses shuffle+slice (not sample) to match the legacy oe-eval-internal behavior.
        """
        import random

        if self.config.num_fewshot == 0:
            return []

        loader = DataLoader()
        all_instances: list[Instance] = []

        for split in ["prompt", "train"]:
            try:
                source = self._get_source_for_split(split)
                all_instances = [
                    inst
                    for doc in loader.load(source)
                    if (inst := self.process_doc(doc)) is not None
                ]
                if all_instances:
                    break
            except Exception:
                continue

        if not all_instances:
            return []

        rng = random.Random(self.config.fewshot_seed)
        rng.shuffle(all_instances)
        return all_instances[: self.config.num_fewshot]


@register("mbpp")
class MBPP(MBPPBase):
    """MBPP code generation task."""

    data_source = DataSource(path="google-research-datasets/mbpp")
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.0,
        stop_sequences=MBPP_STOP_SEQUENCES,
    )


class MBPPPlusBase(Task):
    """Base class for MBPP+ tasks with additional test cases."""

    fewshot_split: str = "test"  # MBPP+ doesn't have a dedicated prompt split

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
            request_type=self.request_type,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract code from model output."""
        code = extract_code(output.text)
        if code and "answer_prefix" in (output.metadata or {}):
            return output.metadata["answer_prefix"] + code
        return code


@register("mbpp_plus")
class MBPPPlus(MBPPPlusBase):
    """MBPP+ code generation task."""

    data_source = DataSource(path="evalplus/mbppplus")
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.0,
        stop_sequences=MBPP_STOP_SEQUENCES,
    )


@register("mbpp:bpb")
class MBPPBPB(MBPPBase):
    data_source = DataSource(path="google-research-datasets/mbpp")
    formatter = PPLFormatter(leading_space=False)
    metrics = (BPBMetricInstanceAvg(),)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        question = doc["text"].strip() + "\n```python\n"
        gold_answer = doc["code"].rstrip("\n").rstrip().replace("\r", "") + "\n```"

        tests = doc.get("test_setup_code", "") or ""
        if tests:
            tests += "\n"
        tests += "\n".join(doc["test_list"])

        return Instance(
            question=question,
            gold_answer=gold_answer,
            metadata={
                "id": doc["task_id"],
                "test": tests,
            },
        )


register_variant(
    "mbpp:bpb",
    "olmo3base",
    num_fewshot=3,
    limit=500,
    fewshot_seed=1234,
)


# =============================================================================
# Variant Registrations
# =============================================================================

# BPB variant - use mbpp:bpb or mbpp_plus:bpb
register_variant(
    "mbpp",
    "bpb",
    formatter=PPLFormatter(leading_space=False),
    metrics=(BPBMetricInstanceAvg(),),
)

register_variant(
    "mbpp_plus",
    "bpb",
    formatter=PPLFormatter(leading_space=False),
    metrics=(BPBMetricInstanceAvg(),),
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

# =============================================================================
# Pass@K Execution Variants (require sandbox)
# =============================================================================
# These variants execute generated code against test cases.
# Requires HarnessConfig with sandboxes configured:
#   sandboxes=(SandboxConfig(image="..."),)

register_variant(
    "mbpp",
    "pass_at_1",
    metrics=(PassAtKMetric(k=1, scorer=CodeExecutionScorer),),
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.2,
        stop_sequences=MBPP_STOP_SEQUENCES,
    ),
)

register_variant(
    "mbpp",
    "pass_at_10",
    metrics=(PassAtKMetric(k=10, scorer=CodeExecutionScorer),),
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.8,
        num_samples=10,
        stop_sequences=MBPP_STOP_SEQUENCES,
    ),
)

register_variant(
    "mbpp_plus",
    "pass_at_1",
    metrics=(PassAtKMetric(k=1, scorer=CodeExecutionScorer),),
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.2,
        stop_sequences=MBPP_STOP_SEQUENCES,
    ),
)

register_variant(
    "mbpp_plus",
    "pass_at_10",
    metrics=(PassAtKMetric(k=10, scorer=CodeExecutionScorer),),
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.8,
        num_samples=10,
        stop_sequences=MBPP_STOP_SEQUENCES,
    ),
)


# =============================================================================
# EvalPlus Variant (different prompt format)
# =============================================================================


@register("mbpp:olmo3base")
class MBPPOlmo3Base(MBPPBase):
    """MBPP with EvalPlus-style prompt format for OLMo3 base evaluation.

    Wraps the problem in an instruction + markdown code block with a sample test case.
    Matches the old oe-eval-internal ``mbpp:3shot::olmo3:n32:v2`` configuration:
    - Fewshot examples use ``question + code + "\\n"`` (no answer prefix).
    - The answer prefix ``Here is the completed function:\\n\\n```python\\n`` is
      appended only to the final (target) prompt.
    - Fewshot examples are taken in dataset order (no shuffle).
    """

    data_source = DataSource(path="google-research-datasets/mbpp")
    num_fewshot: int = 3
    fewshot_seed: int = 1234
    # We override format_request so the formatter is unused for this task, but
    # keep it for the bpb variant which overrides it via register_variant.
    formatter = CompletionFormatter(
        answer_prefix="Here is the completed function:\n\n```python\n",
    )
    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.6,
        top_p=0.6,
        do_sample=True,
        num_samples=32,
        stop_sequences=OLMO3_MBPP_STOP_SEQUENCES,
    )
    metrics = (
        PassAtKMetric(k=1, scorer=CodeExecutionScorer3s),
        PassAtKMetric(k=2, scorer=CodeExecutionScorer3s),
        PassAtKMetric(k=4, scorer=CodeExecutionScorer3s),
        PassAtKMetric(k=8, scorer=CodeExecutionScorer3s),
        PassAtKMetric(k=16, scorer=CodeExecutionScorer3s),
    )

    _ANSWER_PREFIX = "Here is the completed function:\n\n```python\n"

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        random_test = doc["test_list"][0] if doc.get("test_list") else ""
        question = (
            "Please provide a self-contained Python script that solves the "
            "following problem in a markdown code block:\n```\n"
            + doc["text"].strip()
            + "\n"
            + random_test
            + "\n```\n"
        )

        # Match old oe-eval-internal: only use test_list, no test_setup_code
        tests = "\n".join(doc["test_list"])

        return Instance(
            question=question,
            gold_answer=doc["code"] + "\n```",
            metadata={
                "id": doc["task_id"],
                "answer_prefix": "",
                "test": tests,
                "code": doc["code"],
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Build prompt matching old oe-eval-internal format.

        Fewshot examples: ``question + code + "\\n"`` (no answer prefix).
        Target: ``question + answer_prefix``.
        Separator between parts: ``"\\n\\n"``.
        """
        fewshot = self.get_fewshot()
        parts: list[str] = []
        for ex in fewshot:
            parts.append(ex.question + ex.metadata["code"] + "\n")
        # Target gets the answer prefix
        parts.append(instance.question + self._ANSWER_PREFIX)
        prompt = "\n\n".join(parts)
        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)

    def _build_fewshot(self) -> list[Instance]:
        if self.config.num_fewshot == 0:
            return []

        instances = [self.process_doc(doc) for doc in MBPP_FEWSHOT_SOURCES]
        return instances[: self.config.num_fewshot]

    def extract_answer(self, output: LMOutput) -> str | None:
        return extract_code_before_fence(output.text)


register_variant(
    "mbpp:olmo3base",
    "bpb",
    formatter=PPLFormatter(leading_space=False),
    metrics=(BPBMetricByteAvg(),),
)
