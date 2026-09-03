"""LiveCodeBench code generation task.

LiveCodeBench collects programming contest problems tagged by contest date, so
a model can be scored on problems published after its training data was
collected. Solutions are graded by running them against the contest's own test
cases inside a sandbox.

Paper: https://arxiv.org/abs/2403.07974
Dataset: livecodebench/code_generation_lite
"""

from __future__ import annotations

import json
import logging
import shlex
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from olmo_eval.common.execution.environment import StagingExecutionEnvironment
from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import PassAtKMetric
from olmo_eval.common.scorers import ExecutionScorer, SandboxRequiredError
from olmo_eval.common.scorers.code_execution.scripts import get_script
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.extract import extract_code
from olmo_eval.evals.tasks.common import Task, register, register_variant

if TYPE_CHECKING:
    from olmo_eval.common.execution import ExecutionEnvironment

logger = logging.getLogger(__name__)

LIVECODEBENCH_REPO = "livecodebench/code_generation_lite"

# Each release adds one data file; a window of files is one contest date range.
RELEASE_V3_FILES: tuple[str, ...] = ("test.jsonl", "test2.jsonl", "test3.jsonl")
RELEASE_V4_V6_FILES: tuple[str, ...] = ("test4.jsonl", "test5.jsonl", "test6.jsonl")

SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests."
)

# The reasoning line sits between the problem statement and the per-problem
# format instruction, so both parts are substituted here rather than folded
# into the instance question.
USER_TEMPLATE = (
    "### Question:\n{question}\n\n"
    "### Format:\n{reasoning_instruction}{format_instruction}\n\n"
    "### Answer: (use the provided format with backticks)\n\n"
)

THINK_INSTRUCTION = (
    "Provide CONCISE reasoning on how to arrive at the answer in the <think> </think> tag.\n"
)

STARTER_CODE_INSTRUCTION = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters."
)

STDIN_INSTRUCTION = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters "
    "as follows. Ensure that when the python program runs, it reads the inputs, runs "
    "the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```"
)


@lru_cache(maxsize=4)
def _test_case_rows(repo: str, files: tuple[str, ...]) -> Any:
    """Open a release's data files for random access by row.

    Test payloads run to gigabytes per release, so they are read here when a
    solution is graded rather than carried on every instance, which would put
    them in the request records written for the run.
    """
    from datasets import load_dataset

    return load_dataset(
        "json",
        data_files={"train": [f"hf://datasets/{repo}/{name}" for name in files]},
        split="train",
    )


def _parse_verdict(output: str) -> dict[str, Any] | None:
    """Read the grader's JSON verdict, which is the last line it prints."""
    for line in reversed((output or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


@dataclass(slots=True)
class LiveCodeBenchFormatter(ChatFormatter):
    """Chat formatter that fills in each problem's answer format instruction.

    Whether the model is asked to reason in ``<think>`` tags is part of the
    prompt, so regimes that differ only in that line differ only by formatter.
    """

    reasoning_instruction: str = ""

    def format(
        self,
        instance: Instance,
        fewshot: list[Instance] | None = None,
    ) -> LMRequest:
        content = USER_TEMPLATE.format(
            question=instance.question,
            reasoning_instruction=self.reasoning_instruction,
            format_instruction=instance.metadata.get("format_instruction", ""),
        )
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        return LMRequest(
            request_type=self.request_type,
            messages=tuple(messages),
            system_prompt=self.system_prompt or None,
        )


@dataclass(frozen=True, slots=True)
class LiveCodeBenchScorer(ExecutionScorer):
    """Run one solution against a problem's contest test cases in a sandbox.

    The test cases are staged as a file rather than passed in the command,
    because a single problem's cases can run to tens of megabytes and command
    text is bounded by the operating system's argument size limit.
    """

    name: str = "code_exec"
    #: Per test case, matching the reference harness.
    timeout: float = 6.0
    #: Ceiling for one solution across all of its test cases. Grading stops at
    #: the first failure, so only a solution that passes runs the full set.
    overall_timeout: float = 900.0
    max_output_len: int = 4000

    async def ascore(
        self,
        instance: Instance,
        output: LMOutput,
        execution_env: ExecutionEnvironment,
    ) -> float:
        if output.extracted_answer is None:
            output.metadata["execution_result"] = {"success": False, "error": "No extracted answer"}
            return 0.0

        if not isinstance(execution_env, StagingExecutionEnvironment):
            raise SandboxRequiredError(
                f"{type(self).__name__} needs an execution environment that can stage "
                "files; LiveCodeBench test cases are too large to pass as a command."
            )

        metadata = instance.metadata
        row = _test_case_rows(metadata["test_repo"], tuple(metadata["test_files"]))[metadata["row"]]
        if row["question_id"] != metadata["id"]:
            # Grading against another problem's tests would score every
            # solution wrong while still looking like a plausible result.
            raise RuntimeError(
                f"Test cases for problem {metadata['id']} are not at row "
                f"{metadata['row']}; the dataset's row order changed."
            )
        problem = {
            "public_test_cases": row["public_test_cases"],
            "private_test_cases": row["private_test_cases"],
            "fn_name": metadata.get("fn_name"),
            "timeout": self.timeout,
        }

        work_dir = f"/tmp/lcb-{uuid.uuid4().hex}"
        quoted = shlex.quote(work_dir)
        result = await execution_env.execute_with_files(
            f"cd {quoted} && python3 grade.py; status=$?; rm -rf {quoted}; exit $status",
            {
                f"{work_dir}/grade.py": get_script("livecodebench_grader"),
                f"{work_dir}/problem.json": json.dumps(problem),
                f"{work_dir}/solution.py": output.extracted_answer,
            },
            timeout=self.overall_timeout,
        )

        verdict = _parse_verdict(result.output)
        passed = bool(verdict and verdict.get("passed"))
        output.metadata["execution_result"] = {
            "success": passed,
            "exit_code": result.exit_code,
            "error": result.error or (verdict or {}).get("error_message", ""),
            "output": result.output[: self.max_output_len] if result.output else "",
        }
        if verdict is None:
            logger.warning(
                "No grader verdict for instance %s: %s",
                metadata.get("id", "?"),
                (result.error or result.output or "")[:200],
            )
        return 1.0 if passed else 0.0


PASS_AT_KS = (1, 5, 10)
METRICS = tuple(PassAtKMetric(k=k, scorer=LiveCodeBenchScorer) for k in PASS_AT_KS)
PASS_AT_1 = PassAtKMetric(k=1, scorer=LiveCodeBenchScorer)

THINK_FORMATTER = LiveCodeBenchFormatter(
    system_prompt=SYSTEM_PROMPT,
    reasoning_instruction=THINK_INSTRUCTION,
)
PLAIN_FORMATTER = LiveCodeBenchFormatter(system_prompt=SYSTEM_PROMPT)

# Defaults mirror oe-eval's ``livecodebench_codegeneration::olmo3:adapt``, the
# OLMo 3 post-training regime. max_tokens=None generates to the model's context
# limit, matching the reference regime's effective behavior on any context size.
ADAPT_SAMPLING = SamplingParams(
    max_tokens=None,
    temperature=0.6,
    top_p=0.95,
    num_samples=10,
)


@register("livecodebench")
class LiveCodeBench(Task):
    """LiveCodeBench release_v3: contests through the v3 cutoff (612 problems)."""

    release_files: tuple[str, ...] = RELEASE_V3_FILES

    data_source = DataSource(
        path=LIVECODEBENCH_REPO,
        data_files=RELEASE_V3_FILES,
        split="train",
    )
    split = Split.TRAIN
    formatter = THINK_FORMATTER
    metrics = METRICS
    primary_metric = PASS_AT_1
    sampling_params = ADAPT_SAMPLING
    # Test execution is slow and long-tailed; keep a share of shared sandbox
    # pools comparable to the other heavyweight code tasks.
    sandbox_allocation_weight = 6.0

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        starter_code = (doc.get("starter_code") or "").strip()
        if starter_code:
            format_instruction = (
                f"{STARTER_CODE_INSTRUCTION}\n```python\n{doc['starter_code']}\n```"
            )
        else:
            format_instruction = STDIN_INSTRUCTION

        problem_metadata = json.loads(doc["metadata"]) if doc.get("metadata") else {}
        contest_date = doc.get("contest_date", "unknown")
        if hasattr(contest_date, "isoformat"):
            contest_date = contest_date.isoformat()

        return Instance(
            question=doc["question_content"],
            metadata={
                "id": doc["question_id"],
                # Located by row so the scorer can read this problem's test
                # cases without them travelling on the instance.
                "row": index,
                "test_repo": LIVECODEBENCH_REPO,
                "test_files": self.release_files,
                "format_instruction": format_instruction,
                "fn_name": problem_metadata.get("func_name"),
                "platform": doc.get("platform", "unknown"),
                "difficulty": doc.get("difficulty", "unknown"),
                "contest_date": contest_date,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        assert self.config.formatter is not None
        return self.config.formatter.format(instance, self.get_fewshot())

    def extract_answer(self, output: LMOutput) -> str | None:
        return extract_code(output.text)


@register("livecodebench_hidden")
class LiveCodeBenchHidden(LiveCodeBench):
    """LiveCodeBench v4-v6: contests after the v3 cutoff (443 problems).

    Held out from the main task so that problems postdating a model's training
    data can be scored separately.
    """

    release_files: tuple[str, ...] = RELEASE_V4_V6_FILES

    data_source = DataSource(
        path=LIVECODEBENCH_REPO,
        data_files=RELEASE_V4_V6_FILES,
        split="train",
    )


for _task_name in ("livecodebench", "livecodebench_hidden"):
    # Mirrors oe-eval's ``::tulu``: no reasoning instruction, shorter budget.
    register_variant(
        _task_name,
        "tulu",
        formatter=PLAIN_FORMATTER,
        sampling_params=SamplingParams(
            max_tokens=2048,
            temperature=0.2,
            top_p=0.95,
            num_samples=10,
        ),
    )

    # Mirrors oe-eval's ``::tulu-thinker_deepseek_lite``: the default regime
    # scored on a single sample, for cheaper iteration.
    register_variant(
        _task_name,
        "lite",
        metrics=(PASS_AT_1,),
        primary_metric=PASS_AT_1,
        sampling_params=replace(ADAPT_SAMPLING, num_samples=1),
    )

    # Mirrors oe-eval's ``:grpo::tulu-thinker``, used for RL runs.
    register_variant(
        _task_name,
        "grpo",
        metrics=tuple(PassAtKMetric(k=k, scorer=LiveCodeBenchScorer) for k in (1, 2, 4, 8, 10)),
        primary_metric=PassAtKMetric(k=10, scorer=LiveCodeBenchScorer),
        sampling_params=SamplingParams(
            max_tokens=16384,
            temperature=1.0,
            top_p=1.0,
            num_samples=10,
        ),
    )
