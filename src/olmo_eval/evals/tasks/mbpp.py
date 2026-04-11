"""MBPP code generation task implementations."""

from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.common.formatters import CompletionFormatter, PPLFormatter
from olmo_eval.common.metrics import BPBMetric, BPBMetricByteAvg, PassAtKMetric
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, Response, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.code import MBPP_STOP_SEQUENCES, OLMO3_MBPP_STOP_SEQUENCES
from olmo_eval.evals.extract import extract_code
from olmo_eval.evals.tasks.common import Task, register, register_variant


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

        # For fewshot: the function body without the signature line (which is
        # already at the end of `question`).  gold_answer keeps the full code
        # so that BPB variants are unaffected.
        code_lines = doc["code"].split("\n", 1)
        fewshot_body = "\n" + code_lines[1] if len(code_lines) > 1 else ""

        return Instance(
            question=question,
            gold_answer=doc["code"],
            metadata={
                "id": doc["task_id"],
                "answer_prefix": func_sig,
                "fewshot_body": fewshot_body,
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
    metrics = (BPBMetric(),)

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
    metrics=(BPBMetricByteAvg(),),
)

register_variant(
    "mbpp_plus",
    "bpb",
    formatter=PPLFormatter(leading_space=False),
    metrics=(BPBMetricByteAvg(),),
)

# 3shot variant - composable with bpb (e.g., mbpp:3shot:bpb)
register_variant(
    "mbpp",
    "3shot",
    num_fewshot=3,
    formatter=CompletionFormatter(fewshot_answer_key="fewshot_body"),
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


@register("mbpp_olmo3base")
class MBPPOlmo3Base(MBPPBase):
    """MBPP with EvalPlus-style prompt format for OLMo3 base evaluation.

    Wraps the problem in an instruction + markdown code block with a sample test case.
    """

    data_source = DataSource(path="google-research-datasets/mbpp")
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.6,
        top_p=0.6,
        do_sample=True,
        stop_sequences=OLMO3_MBPP_STOP_SEQUENCES,
    )

    # Assistant prefix matching oe-eval-internal's evalplus variant default.
    ASSISTANT_PREFIX = "Here is the completed function:\n\n```python\n"

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

        tests = doc.get("test_setup_code", "") or ""
        if tests:
            tests += "\n"
        tests += "\n".join(doc["test_list"])

        return Instance(
            question=question,
            gold_answer=doc["code"] + "\n",
            metadata={
                "id": doc["task_id"],
                "answer_prefix": "",
                "test": tests,
            },
        )

    # Legacy fewshot examples from oe-eval-internal's ``Original:MBPP`` source.
    # The hardcoded source has single-line code (all on one line) and a specific
    # ordering (task_id=2 first).  The HF ``prompt`` split has multi-line code
    # with ``\r\n`` endings, which changes the prompt the model sees and breaks
    # score parity.  We hardcode the legacy data to guarantee identical prompts.
    _LEGACY_FEWSHOT: tuple[dict[str, Any], ...] = (
        {
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2): res = tuple(set(test_tup1) & set(test_tup2)) return (res)",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "task_id": 2,
            "test_setup_code": "",
        },
        {
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math def is_not_prime(n): result = False for i in range(2,int(math.sqrt(n)) + 1): if n % i == 0: result = True return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "task_id": 3,
            "test_setup_code": "",
        },
        {
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq def heap_queue_largest(nums,n): largest_nums = hq.nlargest(n, nums) return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "task_id": 4,
            "test_setup_code": "",
        },
    )

    def _build_fewshot(self) -> list[Instance]:
        """Build few-shot examples matching oe-eval-internal exactly.

        Uses the hardcoded legacy fewshot data (``Original:MBPP`` source) with
        single-line code to produce identical prompts to oe-eval-internal.
        Falls back to HF data if more examples are needed than hardcoded.
        """
        if self.config.num_fewshot == 0:
            return []

        # Use hardcoded legacy data when possible
        if self.config.num_fewshot <= len(self._LEGACY_FEWSHOT):
            return [
                self.process_doc(doc) for doc in self._LEGACY_FEWSHOT[: self.config.num_fewshot]
            ]

        # Fallback for > 3 fewshot: load from HF and strip \r
        from olmo_eval.data import DataLoader

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

        # Rotate so that task_id=1 (HF index 0) goes to the end, matching
        # the oe-eval-internal hardcoded fewshot source order.
        if len(all_instances) > 1:
            all_instances = all_instances[1:] + all_instances[:1]

        return all_instances[: self.config.num_fewshot]

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request.

        When a formatter is configured (e.g. PPLFormatter for BPB variants),
        delegates to that formatter.  Otherwise builds a completion prompt with
        optional few-shot examples and the assistant prefix, matching
        oe-eval-internal's fewshot_context behaviour:

        - Few-shot examples: question + gold_answer (no assistant prefix)
        - Eval example: question + assistant prefix
        - Separator: ``\\n\\n``
        """
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        fewshot = self.get_fewshot()
        if not fewshot:
            prompt = instance.question + self.ASSISTANT_PREFIX
        else:
            parts: list[str] = []
            for ex in fewshot:
                parts.append(ex.question + (ex.gold_answer or ""))
            parts.append(instance.question + self.ASSISTANT_PREFIX)
            prompt = "\n\n".join(parts)
        return LMRequest(request_type=self.request_type, prompt=prompt)


register_variant(
    "mbpp_olmo3base",
    "3shot",
    num_fewshot=3,
    fewshot_seed=1234,
)

register_variant(
    "mbpp_olmo3base",
    "n32",
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.6,
        top_p=0.6,
        do_sample=True,
        num_samples=32,
        stop_sequences=OLMO3_MBPP_STOP_SEQUENCES,
    ),
    metrics=(
        PassAtKMetric(k=1, scorer=CodeExecutionScorer),
        PassAtKMetric(k=2, scorer=CodeExecutionScorer),
        PassAtKMetric(k=4, scorer=CodeExecutionScorer),
        PassAtKMetric(k=8, scorer=CodeExecutionScorer),
        PassAtKMetric(k=16, scorer=CodeExecutionScorer),
    ),
)

register_variant(
    "mbpp_olmo3base",
    "bpb",
    formatter=PPLFormatter(leading_space=False),
    metrics=(BPBMetricByteAvg(),),
)

# Parity variant matching oe-eval-internal's mbpp:3shot::olmo3:n32:v2 exactly.
# Uses EvalPlus prompt format, 3-shot from "prompt" split in legacy order,
# max_tokens=512, and pass@k metrics with 32 samples.
# Timeout=3.0 matches oe-eval-internal's code_eval_local default.
_olmo3base_scorer = CodeExecutionScorer(timeout=3.0)
register_variant(
    "mbpp_olmo3base",
    "olmo3base",
    num_fewshot=3,
    fewshot_seed=1234,
    sampling_params=SamplingParams(
        max_tokens=512,
        temperature=0.6,
        top_p=0.6,
        do_sample=True,
        num_samples=32,
        stop_sequences=OLMO3_MBPP_STOP_SEQUENCES,
    ),
    metrics=(
        PassAtKMetric(k=1, scorer=_olmo3base_scorer),
        PassAtKMetric(k=2, scorer=_olmo3base_scorer),
        PassAtKMetric(k=4, scorer=_olmo3base_scorer),
        PassAtKMetric(k=8, scorer=_olmo3base_scorer),
        PassAtKMetric(k=16, scorer=_olmo3base_scorer),
    ),
)

register_variant(
    "mbpp",
    "olmo3base",
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.6,
        top_p=0.6,
        do_sample=True,
        num_samples=32,
        stop_sequences=OLMO3_MBPP_STOP_SEQUENCES,
    ),
    metrics=(
        PassAtKMetric(k=1, scorer=CodeExecutionScorer),
        PassAtKMetric(k=2, scorer=CodeExecutionScorer),
        PassAtKMetric(k=4, scorer=CodeExecutionScorer),
        PassAtKMetric(k=8, scorer=CodeExecutionScorer),
        PassAtKMetric(k=16, scorer=CodeExecutionScorer),
    ),
)
