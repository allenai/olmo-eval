"""MBPP code generation task implementations."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import CompletionFormatter, PPLFormatter
from olmo_eval.common.metrics import BPBMetricByteAvg, BPBMetricInstanceAvg, PassAtKMetric
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.code import MBPP_STOP_SEQUENCES, OLMO3_MBPP_STOP_SEQUENCES
from olmo_eval.evals.extract import extract_code, extract_code_before_fence
from olmo_eval.evals.tasks.common import Task, register, register_variant

_logger = __import__("logging").getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _CodeExecScorer3s(CodeExecutionScorer):
    """CodeExecutionScorer with 3s timeout and \\n separator matching old oe-eval-internal."""

    timeout: float = 3.0

    async def ascore(self, instance, output, execution_env):  # type: ignore[override]
        if output.extracted_answer is None:
            return 0.0
        test_code = instance.metadata.get("test", "")
        if not test_code:
            return 0.0
        # Old system used single \n separator: completion + "\n" + test
        full_code = f"{output.extracted_answer}\n{test_code}"
        result = await execution_env.execute_code(
            full_code, language=self.language, timeout=self.timeout,
        )
        if not result.success and result.error:
            instance_id = instance.metadata.get("id", "?")
            _logger.warning(f"Code execution failed [{instance_id}]: {result.error}")
        return 1.0 if result.success else 0.0


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
        PassAtKMetric(k=1, scorer=_CodeExecScorer3s),
        PassAtKMetric(k=2, scorer=_CodeExecScorer3s),
        PassAtKMetric(k=4, scorer=_CodeExecScorer3s),
        PassAtKMetric(k=8, scorer=_CodeExecScorer3s),
        PassAtKMetric(k=16, scorer=_CodeExecScorer3s),
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

        tests = doc.get("test_setup_code", "") or ""
        if tests:
            tests += "\n"
        tests += "\n".join(doc["test_list"])

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
            # Old format: question + code + "\n" (no answer prefix in examples)
            parts.append(ex.question + ex.metadata["code"] + "\n")
        # Target gets the answer prefix
        parts.append(instance.question + self._ANSWER_PREFIX)
        prompt = "\n\n".join(parts)
        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)

    # Hardcoded fewshot examples matching old oe-eval-internal FEWSHOT_SOURCES["Original:MBPP"].
    # The code strings are single-line (no newlines) exactly as in the old system.
    # Order matches the old hardcoded order: task_id [2,3,4,...,10,1].
    _FEWSHOT_SOURCES: list[dict[str, Any]] = [
        {
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": (
                "def similar_elements(test_tup1, test_tup2):"
                " res = tuple(set(test_tup1) & set(test_tup2))"
                " return (res)"
            ),
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "task_id": 2,
        },
        {
            "text": "Write a python function to identify non-prime numbers.",
            "code": (
                "import math"
                " def is_not_prime(n):"
                " result = False"
                " for i in range(2,int(math.sqrt(n)) + 1):"
                " if n % i == 0:"
                " result = True"
                " return result"
            ),
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "task_id": 3,
        },
        {
            "text": (
                "Write a function to find the largest integers from a given list"
                " of numbers using heap queue algorithm."
            ),
            "code": (
                "import heapq as hq"
                " def heap_queue_largest(nums,n):"
                " largest_nums = hq.nlargest(n, nums)"
                " return largest_nums"
            ),
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                (
                    "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)"
                    "==[85, 75, 65, 58, 35]"
                ),
            ],
            "task_id": 4,
        },
        {
            "text": (
                "Write a function to find the number of ways to fill it"
                " with 2 x 1 dominoes for the given 3 x n board."
            ),
            "code": (
                "def count_ways(n):"
                " A = [0] * (n + 1)"
                " B = [0] * (n + 1)"
                " A[0] = 1"
                " A[1] = 0"
                " B[0] = 0"
                " B[1] = 1"
                " for i in range(2, n+1):"
                " A[i] = A[i - 2] + 2 * B[i - 1]"
                " B[i] = A[i - 1] + B[i - 2]"
                " return A[n]"
            ),
            "test_list": [
                "assert count_ways(2) == 3",
                "assert count_ways(8) == 153",
                "assert count_ways(12) == 2131",
            ],
            "task_id": 5,
        },
        {
            "text": (
                "Write a python function to check whether the two numbers"
                " differ at one bit position only or not."
            ),
            "code": (
                "def is_Power_Of_Two (x):"
                " return x and (not(x & (x - 1)))"
                " def differ_At_One_Bit_Pos(a,b):"
                " return is_Power_Of_Two(a ^ b)"
            ),
            "test_list": [
                "assert differ_At_One_Bit_Pos(13,9) == True",
                "assert differ_At_One_Bit_Pos(15,8) == False",
                "assert differ_At_One_Bit_Pos(2,4) == False",
            ],
            "task_id": 6,
        },
        {
            "text": (
                "Write a function to find all words which are at least 4 characters"
                " long in a string by using regex."
            ),
            "code": (
                "import re"
                " def find_char_long(text):"
                " return (re.findall(r'\\b\\w{4,}\\b', text))"
            ),
            "test_list": [
                "assert find_char_long('Please move back to stream') == ['Please', 'move', 'back', 'stream']",
                "assert find_char_long('Jing Eco and Tech') == ['Jing', 'Tech']",
                "assert find_char_long('Jhingai wulu road Zone 3') == ['Jhingai', 'wulu', 'road', 'Zone']",
            ],
            "task_id": 7,
        },
        {
            "text": (
                "Write a function to find squares of individual elements"
                " in a list using lambda function."
            ),
            "code": (
                "def square_nums(nums):"
                " square_nums = list(map(lambda x: x ** 2, nums))"
                " return square_nums"
            ),
            "test_list": [
                "assert square_nums([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])==[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]",
                "assert square_nums([10,20,30])==([100,400,900])",
                "assert square_nums([12,15])==([144,225])",
            ],
            "task_id": 8,
        },
        {
            "text": (
                "Write a python function to find the minimum number of rotations"
                " required to get the same string."
            ),
            "code": (
                "def find_Rotations(str):"
                " tmp = str + str"
                " n = len(str)"
                " for i in range(1,n + 1):"
                " substring = tmp[i: i+n]"
                " if (str == substring):"
                " return i"
                " return n"
            ),
            "test_list": [
                "assert find_Rotations('aaaa') == 1",
                "assert find_Rotations('ab') == 2",
                "assert find_Rotations('abc') == 3",
            ],
            "task_id": 9,
        },
        {
            "text": "Write a function to get the n smallest items from a dataset.",
            "code": (
                "import heapq"
                " def small_nnum(list1,n):"
                " smallest=heapq.nsmallest(n,list1)"
                " return smallest"
            ),
            "test_list": [
                "assert small_nnum([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100],2)==[10,20]",
                "assert small_nnum([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100],5)==[10,20,20,40,50]",
                "assert small_nnum([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100],3)==[10,20,20]",
            ],
            "task_id": 10,
        },
        {
            "text": (
                "Write a function to find the minimum cost path to reach (m, n)"
                " from (0, 0) for the given cost matrix cost[][] and a position"
                " (m, n) in cost[][]."
            ),
            "code": (
                "R = 3"
                " C = 3"
                " def min_cost(cost, m, n):"
                " tc = [[0 for x in range(C)] for x in range(R)]"
                " tc[0][0] = cost[0][0]"
                " for i in range(1, m+1):"
                " tc[i][0] = tc[i-1][0] + cost[i][0]"
                " for j in range(1, n+1):"
                " tc[0][j] = tc[0][j-1] + cost[0][j]"
                " for i in range(1, m+1):"
                " for j in range(1, n+1):"
                " tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j]"
                " return tc[m][n]"
            ),
            "test_list": [
                "assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8",
                "assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12",
                "assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16",
            ],
            "task_id": 1,
        },
    ]

    def _build_fewshot(self) -> list[Instance]:
        """Build few-shot examples from hardcoded sources matching old oe-eval-internal.

        Uses the same single-line code format as FEWSHOT_SOURCES["Original:MBPP"]
        in the old system.  Examples are in the old hardcoded order (task_id
        [2,3,4,...,10,1]) and taken without shuffling.
        """
        if self.config.num_fewshot == 0:
            return []

        # process_doc stores doc["code"] in metadata["code"], so the hardcoded
        # single-line code flows through automatically.
        instances = [self.process_doc(doc) for doc in self._FEWSHOT_SOURCES]
        return instances[: self.config.num_fewshot]

    def extract_answer(self, output: LMOutput) -> str | None:
        return extract_code_before_fence(output.text)


register_variant(
    "mbpp:olmo3base",
    "bpb",
    formatter=PPLFormatter(leading_space=False),
    metrics=(BPBMetricByteAvg(),),
)
