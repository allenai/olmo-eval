"""
Lost in Conversation: Multi-Turn Conversation Reliability Tasks

LLMs Get Lost In Multi-Turn Conversation shards a fully specified instruction into pieces
that a user reveals one per turn, and measures how much worse models do when the task is
underspecified at first than when it is given whole. This module implements five of the
reference's six task types, one task each: ``math`` (GSM8K), ``code`` (HumanEval and
LiveCodeBench), ``actions`` (BFCL), ``data2text`` (ToTTo) and ``summary`` (Summary of a
Haystack).

Paper: https://arxiv.org/abs/2505.06120 (https://openreview.net/forum?id=VKGTGGcwl6)
Reference: https://github.com/microsoft/lost_in_conversation/tree/c865793fe34a929d316119b0451d01bd9183bcfd
Dataset: microsoft/lost_in_conversation

Each task runs a conversation under the ``multi_turn_chat`` scaffold and scores it as the
reference's ``simulator_sharded.py`` does. A judge, the reference's "system model",
classifies every assistant reply (answer attempt, clarification, hedge, ...); each answer
attempt is extracted and scored by the task's own evaluator, and the conversation ends at
its first correct answer. A conversation scores its last answer attempt, or 0 when it
made none. The ``full`` variants ask the fully specified instruction in one turn, as the
reference's ``simulator_full.py`` does, so the drop from ``full`` to the default sharded
setting is the paper's multi-turn degradation.

Scoring per task, as in the reference's ``tasks/<task>/task_<task>.py``:

- math: the judge extracts the answer verbatim; it is compared with the GSM8K gold
  number after lm-evaluation-harness's flexible-extract normalization.
- code: the last top-level function in the reply is renamed to the problem's function
  and run against its public and private test cases in a sandbox, 6 seconds per case.
  The task also reports ``last_reply_pass_rate``, the judge-free pass rate of the
  conversation's last reply, whose execution scorer is what starts the sandbox.
- actions: the whole reply is parsed as Python function calls and checked with BFCL's AST
  checker against the reference answer.
- data2text: the whole reply is scored by sacrebleu's corpus BLEU against the ToTTo
  references, from 0 to 1. A description matching the references can score a hair above
  1 in floating point; as in the reference, it then does not count as correct, and the
  conversation goes on.
- summary: every reply counts as an answer attempt and only the last turn's is evaluated:
  the judge checks which bullet covers each reference insight, and the score is the mean
  of coverage times citation F1 over the insights.

Everything the judge does goes through one ``LostInConversationJudge``, so ``-o
judge_model=<model>`` swaps the judge, ``-o 'judge_models=[...]'`` or the
``judge_panel`` variant grades every conversation with each judge of a panel (see
``olmo_eval.evals.tasks.common.judge_panel``), and a subclass can replace the judge's
prompts.

Deviations from the reference:

- User turns are scripted. The reference's ``user_agent.py`` has an LLM reveal one shard
  per turn, rephrasing it and choosing which comes next, for math, code and actions; here
  turn k sends shard k verbatim, in the dataset's order, so every conversation has one
  turn per shard. The first turn, and every data2text and summary turn, are scripted in
  the reference too. The reference's summary turns are kept as written, including its
  sending the first shard's documents twice, in the first two turns.
- The scaffold asks every turn, and the scorer ignores the replies after the first correct
  answer; the reference stops the conversation there, so the earlier replies, and the
  score, are the same.
- The database task is not included: it executes SQL on Spider's test-suite databases,
  which are not published in a pinnable location.
- The judge defaults to ``JUDGE_MODEL``, the reference's default system model, for the
  summary coverage check too, which the reference makes with gpt-4o at temperature 1.
- A judge reply that is not JSON, or names no known response type, coverage or bullet,
  is asked for again up to ``JUDGE_ATTEMPTS`` times, then the conversation is left
  unscored; the reference stops with an error. As in the reference, an answer the judge
  cannot extract verbatim in that many attempts is scored as empty.
- Reasoning traces are removed from a reply before the judge or an evaluator sees it,
  and a trace that never closed leaves an empty reply.
- Each instruction is simulated once. The paper's aptitude and unreliability metrics,
  the 90th percentile and the 90th-10th percentile spread over repeated simulations, need
  several conversations per instruction, which a judge panel's per-instance verdicts do
  not allow.
- The reference samples at temperature 1.0 with 1000 new tokens, or 10000 for a reasoning
  model; pass ``-o max_tokens=10000`` for a reasoning model.

Example commands to run:

olmo-eval run -m allenai/Olmo-3-7B-Instruct --harness multi_turn_chat \
  -t lost_in_conversation_math -t lost_in_conversation_math:full --no-upload

olmo-eval run -m allenai/Olmo-3-7B-Instruct --harness multi_turn_chat \
  -t lost_in_conversation_math -o 'judge_models=["gpt-4o-mini-2024-07-18","gpt-4.1-2025-04-14"]'
"""

from __future__ import annotations

import ast
import json
import re
import shlex
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric, Metric
from olmo_eval.common.multi_turn import missing_replies_error, scripted_chat_request, turn_replies
from olmo_eval.common.scorers import (
    ExecutionScorer,
    SandboxRequiredError,
    Scorer,
    ScoringIncompleteError,
)
from olmo_eval.common.scorers.bfcl.checker import ast_checker
from olmo_eval.common.scorers.bfcl.constants import Language
from olmo_eval.common.scorers.bfcl.decoding import resolve_ast_call
from olmo_eval.common.scorers.code_execution.scripts import get_script
from olmo_eval.common.types import Instance, LMOutput, LMRequest, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.tasks.common.judge import JudgeParseError, visible_response
from olmo_eval.evals.tasks.common.judge_panel import (
    JudgePanelTask,
    JudgeVerdictMetric,
    PanelJudgeScorer,
    register_judge_panel_variant,
)
from olmo_eval.evals.tasks.constants.lost_in_conversation import (
    ACTIONS_ANSWER_DESCRIPTION,
    ACTIONS_FULL_PROMPT,
    ACTIONS_SYSTEM_PROMPT,
    ANSWER_EXTRACTION_PROMPT,
    CODE_ANSWER_DESCRIPTION,
    CODE_SYSTEM_PROMPT,
    DATA2TEXT_ANSWER_DESCRIPTION,
    DATA2TEXT_FULL_PROMPT,
    DATA2TEXT_SYSTEM_PROMPT,
    HUMANEVAL_FULL_PROMPT,
    LCB_FORMATTING_PREAMBLE,
    LCB_FULL_PROMPT,
    MATH_ANSWER_DESCRIPTION,
    MATH_FULL_PROMPT,
    MATH_SYSTEM_PROMPT,
    SUMMARY_ANSWER_DESCRIPTION,
    SUMMARY_COVERAGE_PROMPT,
    SUMMARY_FOLLOW_UP_PROMPT,
    SUMMARY_FULL_PROMPT_CONV,
    SUMMARY_FULL_PROMPT_NEWS,
    SUMMARY_SYSTEM_PROMPT,
    TURN_CATEGORIZATION_PROMPT,
)
from olmo_eval.evals.tasks.livecodebench import GRADER_ERROR_CODE, VERDICT_PREFIX

if TYPE_CHECKING:
    from olmo_eval.common.execution import ExecutionEnvironment

DATASET_REPO = "microsoft/lost_in_conversation"
DATASET_REVISION = "b6b71f418423c61cef0f439c610a4a655cc0e2d5"
DATASET_FILE = "lost_in_conversation.json"

# The reference's default system model (run_simulations.py's --system_model)
JUDGE_MODEL = "gpt-4o-mini-2024-07-18"
JUDGE_MAX_TOKENS = 1024
# The reference's max_extraction_attempts (system_agent.py)
JUDGE_ATTEMPTS = 3

# The reference's assistant settings for a model that does not reason
DEFAULT_SAMPLING = SamplingParams(max_tokens=1000, temperature=1.0)

SHARDED = "sharded"
FULL = "full"
SETTINGS = (SHARDED, FULL)

ANSWER_ATTEMPT = "answer_attempt"
RESPONSE_TYPES = (
    "refuse",
    "missing",
    ANSWER_ATTEMPT,
    "hedge",
    "clarification",
    "interrogation",
    "discussion",
)

# Per test case, as the reference's tasks/code/task_code.py passes to check_correctness
CODE_TEST_TIMEOUT = 6.0
# Ceiling for one grading run, well above the grader's own budget for the largest problem
CODE_GRADING_TIMEOUT = 900.0

COVERAGES = ("NO_COVERAGE", "PARTIAL_COVERAGE", "FULL_COVERAGE")
# The reference's partial-coverage credit and summary length (tasks/summary/eval_summhay.py)
PARTIAL_COVERAGE_SCORE = 0.5
MAX_SUMMARY_WORDS = 300

# =============================================================================
# Dataset
# =============================================================================


@cache
def _load_dataset(path: str, filename: str, revision: str | None) -> tuple[dict[str, Any], ...]:
    """Read the dataset's JSON file, a single list of samples.

    The file is read with Python's JSON parser because ``datasets`` cannot build one
    table from it: ``fully_specified_question`` is a string for some tasks and a list
    of messages for others.
    """
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=path.removeprefix("hf://"),
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    with open(local, encoding="utf-8") as handle:
        return tuple(json.load(handle))


def load_samples(source: DataSource, task: str) -> list[dict[str, Any]]:
    """Return the dataset's samples of one task type, in file order."""
    samples = _load_dataset(source.path, str(source.data_files), source.revision)
    return [sample for sample in samples if sample["task"] == task]


def sample_for(instance: Instance) -> dict[str, Any]:
    """Return the dataset sample an instance was built from.

    Instances carry only their prompts; evaluators read references, test cases and
    insights from the sample, so request records stay small.
    """
    metadata = instance.metadata
    source = DataSource(
        path=metadata["source_path"],
        data_files=metadata["source_file"],
        revision=metadata["source_revision"],
    )
    for sample in load_samples(source, metadata["task"]):
        if sample["task_id"] == metadata["id"]:
            return sample
    raise ScoringIncompleteError(f"Sample {metadata['id']!r} is not in the dataset")


# =============================================================================
# Answer Evaluation
# =============================================================================

_MATH_IGNORED = (",", "\\$", "(?s).*#### ", "\\.$")


def evaluate_math(answer: str, sample: dict[str, Any]) -> dict[str, Any]:
    """Compare an extracted answer with the GSM8K gold number, as the reference does.

    Takes the last number in the answer (lm-evaluation-harness's flexible extract),
    drops a ``.00`` after a dollar amount by ``rstrip``, which the reference also uses,
    and removes commas, dollar signs and a final period from both sides.
    """
    gold = sample["answer"].split("####")[1].strip().lower()
    matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", answer.strip())
    if not matches:
        return {"score": 0.0, "error": f"Answer could not be extracted: {answer!r}"}
    predicted = next(group for group in matches[-1] if group)
    if re.search(r"\$", predicted) and predicted.endswith(".00"):
        # The reference strips trailing "." and "0" characters, so "$100.00" reads as "$1"
        predicted = predicted.rstrip(".00")  # noqa: B005
    for pattern in _MATH_IGNORED:
        predicted = re.sub(pattern, "", predicted)
        gold = re.sub(pattern, "", gold)
    return {"score": 1.0 if predicted == gold else 0.0}


def _function_from_code(code: str) -> str:
    """Return the last top-level function in code, with every import before it, or ""."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    imports = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    # The last function whose parent is the module
    last = next((node for node in reversed(functions) if node in tree.body), None)
    if last is None:
        return ""
    lines = code.splitlines()
    start = last.decorator_list[0].lineno - 1 if last.decorator_list else last.lineno - 1
    function = "\n".join(lines[start : last.end_lineno])
    return "\n".join(imports) + "\n\n" + function if imports else function


def extract_code(text: str) -> str:
    """Return the function a reply answers with, as the reference's ``extract_answer``.

    Takes the last fenced block holding a top-level function, or the last ``class
    Solution`` block; without fences, the text from its last ``import`` (or ``def``).
    Returns "" when there is no function.
    """
    blocks = re.findall(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if "class Solution" in text:
        solutions = [block for block in blocks if "class Solution" in block]
        if solutions:
            return solutions[-1]
        # The reference indexes an empty list here and stops with an error
        raise ScoringIncompleteError("A reply names class Solution outside a code block")
    if blocks:
        for block in reversed(blocks):
            function = _function_from_code(block)
            if function:
                return function
        return ""
    text = text.strip()
    if text.startswith("```") or text.startswith("`"):
        text = text[text.find("\n") :].strip()
    import_index, def_index = text.rfind("import"), text.rfind("def")
    start = import_index if import_index >= 0 else def_index
    if start >= 0:
        text = text[start:]
    return _function_from_code(text)


def prepare_solution(code: str, sample: dict[str, Any]) -> str | None:
    """Prepare an extracted function for the tests, or None if it defines no function.

    As in the reference, a HumanEval problem's imports are prepended, and the first
    function name is replaced everywhere by the problem's function name.
    """
    code = code.replace("```python", "").replace("```", "")
    if "def " not in code:
        return None
    if "prompt" in sample:
        imports = [
            ast.unparse(node)
            for node in ast.parse(sample["prompt"]).body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        if imports:
            code = "\n".join(imports) + "\n\n" + code
    old_name = code.split("def ")[1].split("(")[0].strip()
    return code.replace(old_name, sample["metadata"]["func_name"])


def _code_executor(context: ScoringContext) -> ExecutionEnvironment:
    """Return the default-capability sandbox of the scoring context."""
    if not context.has_execution_env:
        raise SandboxRequiredError(
            "Lost in Conversation code answers run in a sandbox; none is running."
        )
    environment = context.execution_env
    if hasattr(environment, "for_capabilities"):
        from olmo_eval.harness.sandbox.config import Capability

        return environment.for_capabilities(Capability.DEFAULT)  # ty: ignore[call-non-callable]
    assert environment is not None
    return environment


async def run_code_tests(
    execution_env: ExecutionEnvironment, code: str, sample: dict[str, Any]
) -> dict[str, Any]:
    """Run a solution against a problem's test cases with the LiveCodeBench grader.

    The reference grades with LiveCodeBench's ``check_correctness``, which the grader
    ports. A wrong, crashing or slow solution scores 0; a run that gives no verdict,
    grades no test, or fails inside the grader raises.
    """
    from olmo_eval.common.execution.environment import StagingExecutionEnvironment

    solution = prepare_solution(code, sample)
    if solution is None:
        return {"is_correct": False, "score": 0.0, "error": "No function definition"}
    if not isinstance(execution_env, StagingExecutionEnvironment):
        raise SandboxRequiredError("Code tests need a sandbox that can stage files")

    problem = json.dumps(
        {
            "public_test_cases": sample["public_test_cases"],
            # HumanEval problems have public test cases only
            "private_test_cases": sample.get("private_test_cases", ""),
            "fn_name": sample["metadata"]["func_name"],
            "timeout": CODE_TEST_TIMEOUT,
        }
    )
    work_dir = f"/tmp/lic-{uuid.uuid4().hex}"
    quoted = shlex.quote(work_dir)
    result = await execution_env.execute_with_files(
        f"cd {quoted} && timeout {int(CODE_GRADING_TIMEOUT)} python3 grade.py; "
        f"status=$?; rm -rf {quoted}; exit $status",
        {
            f"{work_dir}/grade.py": get_script("livecodebench_grader"),
            f"{work_dir}/problem.json": problem,
            f"{work_dir}/solution.py": solution,
        },
        timeout=CODE_GRADING_TIMEOUT + 60.0,
    )
    verdict = _grader_verdict(result.output)
    if verdict is None or not verdict.get("num_tests"):
        raise ScoringIncompleteError(
            f"No grader verdict for {sample['task_id']} (exit code {result.exit_code}): "
            f"{(result.error or result.output or '')[:500]}"
        )
    if verdict.get("error_code") == GRADER_ERROR_CODE:
        raise ScoringIncompleteError(
            f"The grader failed on {sample['task_id']}: {verdict.get('error_message', '')}"
        )
    passed = bool(verdict.get("passed"))
    return {"is_correct": passed, "score": 1.0 if passed else 0.0}


def _grader_verdict(output: str | None) -> dict[str, Any] | None:
    """Read the grader's JSON verdict line from the command output."""
    for line in reversed((output or "").strip().splitlines()):
        line = line.strip()
        if line.startswith(VERDICT_PREFIX):
            try:
                return json.loads(line[len(VERDICT_PREFIX) :])
            except json.JSONDecodeError:
                continue
    return None


_BAD_STARTS: tuple[str, ...] = ("```tool_code", "```python", "```json", "```", "`")
_BAD_ENDS: tuple[str, ...] = ("```", "`")


def parse_function_calls(text: str) -> list[dict[str, Any]]:
    """Parse a reply into function calls, as the reference's ``ast_parse`` for Python.

    Removes a code fence, joins the lines, strips brackets and quotes from both ends, and
    reads a call or a list of calls.

    Raises:
        Exception: Any error parsing the reply.
    """
    text = text.strip()
    for start in _BAD_STARTS:
        if text.startswith(start):
            text = text[len(start) :]
    for end in _BAD_ENDS:
        if text.endswith(end):
            text = text[: -len(end)]
    text = text.strip().replace("\n", " ")
    parsed = ast.parse(text.strip("[]'"), mode="eval")
    if isinstance(parsed.body, ast.Call):
        return [resolve_ast_call(parsed.body)]
    elements = getattr(parsed.body, "elts", None)
    if elements is None:
        raise ValueError("The reply is not a call or a list of calls")
    calls = []
    for element in elements:
        if not isinstance(element, ast.Call):
            raise ValueError("A list element is not a function call")
        calls.append(resolve_ast_call(element))
    return calls


def evaluate_actions(answer: str, sample: dict[str, Any]) -> dict[str, Any]:
    """Check a reply's function calls against the reference answer with BFCL's checker."""
    try:
        calls = parse_function_calls(answer)
    except Exception:
        return {"is_correct": False, "error": "Failing to parse the predicted answer as an AST"}
    result = ast_checker(
        sample["function"],
        calls,
        sample["reference_answer"],
        Language.PYTHON,
        sample["test_category"],
    )
    valid = bool(result["valid"])
    return {"is_correct": valid, "score": 1.0 if valid else 0.0, "error": result["error"]}


def evaluate_data2text(answer: str, sample: dict[str, Any]) -> dict[str, Any]:
    """Score a description by sacrebleu's corpus BLEU against the references, from 0 to 1."""
    import sacrebleu

    references = [[reference.strip()] for reference in sample["references"]]
    return {"score": sacrebleu.corpus_bleu([answer.strip()], references).score / 100.0}


def summary_bullets(summary: str) -> list[str]:
    """Split a summary into lines, trimming each by the same share past the word limit."""
    bullets = summary.split("\n")
    counts = [len(bullet.split()) for bullet in bullets]
    total = sum(counts)
    if total <= MAX_SUMMARY_WORDS:
        return bullets
    excess = (total - MAX_SUMMARY_WORDS) / total
    trimmed = []
    for bullet, count in zip(bullets, counts, strict=True):
        if count == 0:
            trimmed.append(bullet)
            continue
        keep = max(int(count * (1 - excess)), 1)
        trimmed.append(" ".join(bullet.split()[:keep]))
    return trimmed


def bullets_json(bullets: Sequence[str]) -> str:
    """The bullets as the coverage prompt shows them."""
    return json.dumps(
        {"bullets": [{"bullet_id": i + 1, "text": bullet} for i, bullet in enumerate(bullets)]},
        indent=1,
    )


def _bullet_index(bullet_id: Any) -> int:
    """The list index the reference reads a coverage verdict's bullet from.

    An unreadable bullet id is read as -1, so the reference looks at the second to last
    bullet; that is kept.
    """
    try:
        return int(bullet_id) - 1
    except (TypeError, ValueError):
        return -2


def citations(bullet: str) -> list[int]:
    """The document numbers a bullet cites, such as 1 and 4 in "[1][4]" or "[1, 4]"."""
    numbers = []
    for match in re.findall(r"\[([\d, ]+)\]", bullet):
        numbers += [int(part.strip()) for part in match.split(",") if part.strip()]
    return numbers


def summary_scores(
    bullets: Sequence[str], verdicts: Sequence[dict[str, Any]], references: dict[str, list[int]]
) -> dict[str, Any]:
    """Score a summary's coverage verdicts, as the reference's ``compute_single_sample_results``.

    Each insight scores 1 if fully covered, ``PARTIAL_COVERAGE_SCORE`` if partially, and
    the citation F1 of the covering bullet against the insight's documents; the score is
    the mean of their products.
    """
    coverage, joint, f1s = [], [], []
    for verdict in verdicts:
        covered = verdict["coverage"]
        coverage_score = citation_score = 0.0
        if covered in ("PARTIAL_COVERAGE", "FULL_COVERAGE"):
            coverage_score = 1.0 if covered == "FULL_COVERAGE" else PARTIAL_COVERAGE_SCORE
            bullet = bullets[_bullet_index(verdict.get("bullet_id"))]
            generated = set(citations(bullet))
            reference = set(references[verdict["insight_id"]])
            shared = len(generated & reference)
            precision = shared / len(generated) if generated else 0.0
            recall = shared / len(reference) if reference else 0.0
            citation_score = (
                2 * precision * recall / (precision + recall) if precision + recall else 0.0
            )
            f1s.append(citation_score)
        coverage.append(coverage_score)
        joint.append(coverage_score * citation_score)

    def mean(values: list[float]) -> float | None:
        # The reference's np.mean of no values is NaN; None keeps the record valid JSON
        return sum(values) / len(values) if values else None

    return {
        "coverage_score": mean(coverage),
        "citation_score": mean(f1s),
        "score": mean(joint),
    }


# =============================================================================
# Judge
# =============================================================================


def _json_object(reply: str) -> dict[str, Any] | None:
    """The JSON object a judge reply holds, or None."""
    try:
        parsed = json.loads(reply.strip().removeprefix("```json").removesuffix("```"))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def last_turn_text(reply: str) -> str:
    """The conversation's last turn as the reference shows it to the judge."""
    return f"[assistant] {reply}"


@dataclass(frozen=True)
class LostInConversationJudge(PanelJudgeScorer):
    """The reference's system model: classifies, extracts and evaluates every reply.

    Subclasses set the task's ``answer_description`` and ``evaluate``, and override
    ``extract_answer`` when the reference extracts answers other than whole replies.
    The judge's prompts are class attributes, so a subclass can replace them.

    Scores the conversation's last answer attempt, from 0 to 1, and records each judged
    turn under ``turns``.
    """

    name: str = "lic_judge"
    default_model: ClassVar[str] = JUDGE_MODEL
    default_max_tokens: ClassVar[int] = JUDGE_MAX_TOKENS
    attempts: ClassVar[int] = JUDGE_ATTEMPTS
    judge_response_format: ClassVar[dict[str, Any] | None] = {"type": "json_object"}

    categorization_prompt: ClassVar[str] = TURN_CATEGORIZATION_PROMPT
    extraction_prompt: ClassVar[str] = ANSWER_EXTRACTION_PROMPT
    answer_description: ClassVar[str]
    #: Whether every reply is an answer attempt, without asking the judge.
    every_reply_answers: ClassVar[bool] = False
    #: Whether only the last turn's answer is evaluated; earlier attempts score 0.
    evaluates_last_turn_only: ClassVar[bool] = False

    def format_judge_prompt(self, instance: Instance, output: LMOutput) -> str:
        """The categorization prompt for the output's last reply."""
        replies = turn_replies(output) or [output.text]
        return self.categorization_prompt_for(sample_for(instance), visible_response(replies[-1]))

    def parse_judge_response(self, response: str) -> float:
        """Return 1.0 if a categorization reply calls the turn an answer attempt.

        Raises:
            ValueError: If the reply names no known response type.
        """
        response_type = self.parse_response_type(response)
        if response_type is None:
            raise ValueError(f"no response type in {response!r}")
        return float(response_type == ANSWER_ATTEMPT)

    def categorization_prompt_for(self, sample: dict[str, Any], reply: str) -> str:
        """The categorization prompt, filled in the reference's order."""
        shards = sample["shards"]
        return (
            self.categorization_prompt.replace("[[CONVERSATION_SO_FAR]]", last_turn_text(reply))
            .replace("[[INITIAL_SHARD]]", shards[0]["shard"])
            .replace("[[SHARDS]]", json.dumps(shards[1:]))
            .replace("[[ANSWER_DESCRIPTION]]", self.answer_description)
        )

    @staticmethod
    def parse_response_type(reply: str) -> str | None:
        """The response type a categorization reply names, or None."""
        parsed = _json_object(reply)
        response_type = None if parsed is None else parsed.get("response_type")
        return response_type if response_type in RESPONSE_TYPES else None

    async def extract_answer(
        self,
        reply: str,
        sample: dict[str, Any],
        context: ScoringContext,
        log: list[dict[str, Any]],
        turn: int,
    ) -> str:
        """The answer of an answer attempt: the whole reply, the reference's ``full_response``."""
        return reply

    async def extract_with_judge(
        self, reply: str, context: ScoringContext, log: list[dict], turn: int
    ) -> str:
        """Ask the judge to extract the answer verbatim, the reference's ``gen`` strategy.

        An extraction that is not a substring of the reply is asked for again; after
        ``attempts`` the answer is empty, as in the reference.
        """
        prompt = self.extraction_prompt.replace(
            "[[ASSISTANT_RESPONSE]]", last_turn_text(reply)
        ).replace("[[ANSWER_DESCRIPTION]]", self.answer_description)

        def parse(response: str) -> str | None:
            parsed = _json_object(response)
            answer = None if parsed is None else parsed.get("answer")
            return answer if isinstance(answer, str) and answer in reply else None

        answer, raw = await self._judge(prompt, parse, context)
        log.append({"turn": turn, "step": "extraction", "response": raw})
        return answer if answer is not None else ""

    async def evaluate(
        self,
        answer: str,
        sample: dict[str, Any],
        context: ScoringContext,
        log: list[dict],
        turn: int,
    ) -> dict[str, Any]:
        """Score an extracted answer: ``score`` from 0 to 1, and optionally ``is_correct``."""
        raise NotImplementedError

    async def _categorize(
        self,
        sample: dict[str, Any],
        reply: str,
        context: ScoringContext,
        log: list[dict],
        turn: int,
    ) -> str:
        response_type, raw = await self._judge(
            self.categorization_prompt_for(sample, reply), self.parse_response_type, context
        )
        log.append({"turn": turn, "step": "categorization", "response": raw})
        if response_type is None:
            raise JudgeParseError(
                f"no response type for turn {turn + 1} after {self.attempts} judge attempts"
            )
        return response_type

    async def judge_conversation(
        self, instance: Instance, replies: Sequence[str], context: ScoringContext, log: list[dict]
    ) -> dict[str, Any]:
        """Judge each reply in turn until one answers correctly, as ``simulator_sharded.py``."""
        sample = sample_for(instance)
        setting = instance.metadata["setting"]
        last_turn = len(instance.metadata["turns"]) - 1
        turns: list[dict[str, Any]] = []
        score: float | None = None
        is_correct = False
        for turn, reply in enumerate(replies):
            if setting == FULL or self.every_reply_answers:
                response_type = ANSWER_ATTEMPT
            else:
                response_type = await self._categorize(sample, reply, context, log, turn)
            record: dict[str, Any] = {"turn": turn, "response_type": response_type}
            turns.append(record)
            if response_type != ANSWER_ATTEMPT:
                continue

            answer = await self.extract_answer(reply, sample, context, log, turn)
            if self.evaluates_last_turn_only and turn != last_turn:
                evaluation: dict[str, Any] = {"score": 0.0}
                correct = None
            else:
                evaluation = await self.evaluate(answer, sample, context, log, turn)
                correct = evaluation.get("is_correct")
            score = evaluation.get("score")
            if score == 1.0 and not correct:
                correct = True
            is_correct = bool(correct)
            record.update(answer=answer, score=score, is_correct=correct, evaluation=evaluation)
            if is_correct:
                break

        attempts = [record["turn"] for record in turns if record["response_type"] == ANSWER_ATTEMPT]
        return {
            "setting": setting,
            "score": float(score) if score is not None else float(is_correct),
            "is_correct": is_correct,
            "first_answer_turn": attempts[0] if attempts else None,
            "num_turns": last_turn + 1,
            "turns_answered": len(replies),
            "turns": turns,
        }

    async def ascore_with_context(
        self, instance: Instance, output: LMOutput, context: ScoringContext
    ) -> float:
        """Judge a conversation, recording its turns and judge replies on the instance."""
        replies = turn_replies(output)
        if replies is None:
            raise missing_replies_error("Lost in Conversation")
        log: list[dict[str, Any]] = []
        try:
            self._validate_provider(context)
            result = await self.judge_conversation(
                instance, [visible_response(reply) for reply in replies], context, log
            )
        except Exception:
            self.record(instance, None, raw_response=log or None)
            raise
        self.record(instance, result, raw_response=log)
        return result["score"]


@dataclass(frozen=True)
class MathJudge(LostInConversationJudge):
    """Judge for GSM8K conversations: the judge extracts the number, which is compared."""

    answer_description: ClassVar[str] = MATH_ANSWER_DESCRIPTION

    async def extract_answer(self, reply, sample, context, log, turn):  # noqa: D102
        return await self.extract_with_judge(reply, context, log, turn)

    async def evaluate(self, answer, sample, context, log, turn):  # noqa: D102
        return evaluate_math(answer, sample)


@dataclass(frozen=True)
class CodeJudge(LostInConversationJudge):
    """Judge for coding conversations: the function in the reply is run against the tests."""

    answer_description: ClassVar[str] = CODE_ANSWER_DESCRIPTION

    async def extract_answer(self, reply, sample, context, log, turn):  # noqa: D102
        return extract_code(reply)

    async def evaluate(self, answer, sample, context, log, turn):  # noqa: D102
        return await run_code_tests(_code_executor(context), answer, sample)


@dataclass(frozen=True)
class ActionsJudge(LostInConversationJudge):
    """Judge for function-calling conversations: the reply is checked with BFCL's checker."""

    answer_description: ClassVar[str] = ACTIONS_ANSWER_DESCRIPTION

    async def evaluate(self, answer, sample, context, log, turn):  # noqa: D102
        return evaluate_actions(answer, sample)


@dataclass(frozen=True)
class Data2TextJudge(LostInConversationJudge):
    """Judge for table-description conversations: the reply is scored by BLEU."""

    answer_description: ClassVar[str] = DATA2TEXT_ANSWER_DESCRIPTION

    async def evaluate(self, answer, sample, context, log, turn):  # noqa: D102
        return evaluate_data2text(answer, sample)


@dataclass(frozen=True)
class SummaryJudge(LostInConversationJudge):
    """Judge for summary conversations: the judge checks each insight's coverage.

    Every reply is an answer attempt and only the last turn's is evaluated, as in the
    reference.
    """

    answer_description: ClassVar[str] = SUMMARY_ANSWER_DESCRIPTION
    coverage_prompt: ClassVar[str] = SUMMARY_COVERAGE_PROMPT
    every_reply_answers: ClassVar[bool] = True
    evaluates_last_turn_only: ClassVar[bool] = True

    async def evaluate(self, answer, sample, context, log, turn):  # noqa: D102
        bullets = summary_bullets(answer)
        listed = bullets_json(bullets)

        def parse(response: str) -> dict[str, Any] | None:
            parsed = _json_object(response)
            if parsed is None or parsed.get("coverage") not in COVERAGES:
                return None
            if parsed["coverage"] != "NO_COVERAGE":
                index = _bullet_index(parsed.get("bullet_id"))
                if not -len(bullets) <= index < len(bullets):
                    return None
            return parsed

        verdicts = []
        for insight in sample["insights"]:
            prompt = self.coverage_prompt.replace("[[BULLETS]]", listed).replace(
                "[[INSIGHT]]", insight["insight"]
            )
            verdict, raw = await self._judge(prompt, parse, context)
            log.append({"turn": turn, "step": "coverage", "response": raw})
            if verdict is None:
                raise JudgeParseError(
                    f"no coverage verdict for insight {insight['insight_id']} after "
                    f"{self.attempts} judge attempts"
                )
            verdicts.append({**verdict, "insight_id": insight["insight_id"]})
        return {
            **summary_scores(bullets, verdicts, sample["insightid2ref_citations"]),
            "verdicts": verdicts,
        }


@dataclass(frozen=True, slots=True)
class LastReplyCodeScorer(ExecutionScorer):
    """Run the function in a conversation's last reply against the tests, without a judge."""

    name: str = "lic_last_reply_code_exec"

    async def ascore(
        self, instance: Instance, output: LMOutput, execution_env: ExecutionEnvironment
    ) -> float:
        replies = turn_replies(output)
        if replies is None:
            raise missing_replies_error("Lost in Conversation")
        code = extract_code(visible_response(replies[-1]))
        result = await run_code_tests(execution_env, code, sample_for(instance))
        output.metadata["execution_result"] = result
        return result["score"]


# =============================================================================
# Metrics
# =============================================================================


@dataclass(frozen=True, slots=True)
class ConversationScoreMetric(JudgeVerdictMetric):
    """Mean conversation score: the score of each conversation's last answer attempt."""

    name: str = "any__any__score"
    scorer: type[Scorer] | Scorer = LostInConversationJudge

    def judged_value(self, judge_result: dict[str, Any]) -> float | None:
        return judge_result["score"]


@dataclass(frozen=True, slots=True)
class PrematureAnswerRateMetric(JudgeVerdictMetric):
    """Share of sharded conversations whose first answer attempt came before the last shard."""

    name: str = "any__any__premature_answer_rate"
    scorer: type[Scorer] | Scorer = LostInConversationJudge

    def judged_value(self, judge_result: dict[str, Any]) -> float | None:
        if judge_result["setting"] != SHARDED or judge_result["first_answer_turn"] is None:
            return None
        return float(judge_result["first_answer_turn"] < judge_result["num_turns"] - 1)

    def higher_is_better(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class NoAnswerRateMetric(JudgeVerdictMetric):
    """Share of conversations with no answer attempt at all."""

    name: str = "any__any__no_answer_rate"
    scorer: type[Scorer] | Scorer = LostInConversationJudge

    def judged_value(self, judge_result: dict[str, Any]) -> float | None:
        return float(judge_result["first_answer_turn"] is None)

    def higher_is_better(self) -> bool:
        return False


# =============================================================================
# Task Definitions
# =============================================================================


@dataclass(slots=True)
class LostInConversationFormatter(ChatFormatter):
    """Asks an instance's scripted turns after its system prompt.

    ``setting`` picks the turns: one per shard (``sharded``) or the fully specified
    instruction in one turn (``full``).
    """

    setting: str = SHARDED

    def format(self, instance: Instance, fewshot: list[Instance] | None = None) -> LMRequest:
        if fewshot:
            raise ValueError("Lost in Conversation is asked without examples; use num_fewshot=0")
        system_prompt = instance.metadata["system_prompt"]
        request = scripted_chat_request(instance.metadata["turns"])
        # Providers read the messages, so the system prompt goes in them, as ChatFormatter
        # sends it; the scaffold carries it into every later turn
        return replace(
            request,
            messages=({"role": "system", "content": system_prompt}, *request.messages),
            system_prompt=system_prompt,
        )


SHARDED_FORMATTER = LostInConversationFormatter(setting=SHARDED)
FULL_FORMATTER = LostInConversationFormatter(setting=FULL)

_DATA_SOURCE = DataSource(
    path=DATASET_REPO, data_files=DATASET_FILE, revision=DATASET_REVISION, split="train"
)


def _shard_texts(sample: dict[str, Any]) -> list[str]:
    return [shard["shard"] for shard in sample["shards"]]


class _LostInConversation(JudgePanelTask):
    """Base for the Lost in Conversation tasks, one per task type."""

    data_source = _DATA_SOURCE
    split = Split.TRAIN
    sampling_params = DEFAULT_SAMPLING
    formatter = SHARDED_FORMATTER
    required_scaffold = "multi_turn_chat"
    required_secrets = ("OPENAI_API_KEY",)

    #: The dataset's ``task`` value of the samples this task asks.
    task_type: ClassVar[str]

    def _setting(self) -> str:
        formatter = self.config.formatter
        setting = getattr(formatter, "setting", None)
        if setting not in SETTINGS:
            raise ValueError(f"{self.config.name} needs a LostInConversationFormatter")
        return setting

    def _load_instances(self, split: str | None = None) -> Iterator[Instance]:
        """Yield the dataset's samples of this task type."""
        source = self.config.get_data_source(split=split)
        for index, sample in enumerate(load_samples(source, self.task_type)):
            yield self.process_doc(sample, index)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Build an instance asking the sample's turns for the configured setting."""
        setting = self._setting()
        turns = [self.full_prompt(doc)] if setting == FULL else self.sharded_turns(doc)
        source = self.config.get_data_source()
        return Instance(
            question=turns[0],
            metadata={
                "id": doc["task_id"],
                "index": index,
                "task": doc["task"],
                "setting": setting,
                "system_prompt": self.system_prompt(doc),
                "turns": turns,
                "source_path": source.path,
                "source_file": str(source.data_files),
                "source_revision": source.revision,
                **self.subset_metadata(doc),
            },
        )

    def get_fewshot(self) -> list[Instance]:
        """Return no examples, rejecting a config that asks for some."""
        if self.config.num_fewshot:
            raise ValueError("Lost in Conversation is asked without examples; use num_fewshot=0")
        return []

    def format_request(self, instance: Instance) -> LMRequest:
        """Ask the instance's turns in order, each after the model's earlier replies."""
        assert self.config.formatter is not None
        return self.config.formatter.format(instance, self.get_fewshot())

    def system_prompt(self, sample: dict[str, Any]) -> str:
        raise NotImplementedError

    def full_prompt(self, sample: dict[str, Any]) -> str:
        """The fully specified instruction, the reference's ``populate_fully_specific_prompt``."""
        raise NotImplementedError

    def sharded_turns(self, sample: dict[str, Any]) -> list[str]:
        """The user turns of a sharded conversation: each shard verbatim, in order."""
        return _shard_texts(sample)

    def subset_metadata(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Metadata that subset metrics select on."""
        return {}


def _metrics(judge: LostInConversationJudge, *extra: Metric) -> tuple[Metric, ...]:
    """The conversation score and answer-timing rates, then any task-specific metrics.

    The rates are left out when every reply counts as an answer attempt.
    """
    rates = (
        ()
        if judge.every_reply_answers
        else (PrematureAnswerRateMetric(scorer=judge), NoAnswerRateMetric(scorer=judge))
    )
    return (ConversationScoreMetric(scorer=judge), *rates, *extra)


_MATH_JUDGE = MathJudge()


@register("lost_in_conversation_math")
class LostInConversationMath(_LostInConversation):
    """GSM8K problems revealed a shard at a time."""

    task_type = "math"
    judge = _MATH_JUDGE
    metrics = _metrics(_MATH_JUDGE)
    primary_metric = ConversationScoreMetric(scorer=_MATH_JUDGE)

    def system_prompt(self, sample: dict[str, Any]) -> str:
        return MATH_SYSTEM_PROMPT

    def full_prompt(self, sample: dict[str, Any]) -> str:
        return MATH_FULL_PROMPT.replace("[[QUESTION]]", sample["question"])


_CODE_JUDGE = CodeJudge()
CODE_SOURCES = ("humaneval", "lcb_easy", "lcb_medium")


@register("lost_in_conversation_code")
class LostInConversationCode(_LostInConversation):
    """HumanEval and LiveCodeBench problems revealed a shard at a time, graded in a sandbox."""

    task_type = "code"
    judge = _CODE_JUDGE
    metrics = _metrics(
        _CODE_JUDGE,
        *(
            ConversationScoreMetric(name=f"source__{source}__score", scorer=_CODE_JUDGE)
            for source in CODE_SOURCES
        ),
        AccuracyMetric(name="last_reply_pass_rate", scorer=LastReplyCodeScorer),
    )
    primary_metric = ConversationScoreMetric(scorer=_CODE_JUDGE)
    sandbox_allocation_weight = 6.0

    def system_prompt(self, sample: dict[str, Any]) -> str:
        # The reference asks every code problem with LiveCodeBench's system prompt
        return CODE_SYSTEM_PROMPT

    def full_prompt(self, sample: dict[str, Any]) -> str:
        if sample["source"] == "humaneval":
            instruction = (
                "Complete the following incomplete function signature:\n"
                f"```python\n{sample['prompt']}\n```"
            )
            return HUMANEVAL_FULL_PROMPT.replace("[[INSTRUCTION]]", instruction)
        starter = sample["starter_code"] or "# YOUR CODE HERE"
        return (
            LCB_FULL_PROMPT.replace("[[QUESTION]]", sample["question_content"])
            .replace("[[FORMATTING_PREAMBLE]]", LCB_FORMATTING_PREAMBLE)
            .replace("[[FORMATTING]]", f"```python\n{starter}\n```")
        )

    def subset_metadata(self, sample: dict[str, Any]) -> dict[str, Any]:
        return {"source": sample["source"]}


_ACTIONS_JUDGE = ActionsJudge()


@register("lost_in_conversation_actions")
class LostInConversationActions(_LostInConversation):
    """BFCL parallel function-calling requests revealed a shard at a time."""

    task_type = "actions"
    judge = _ACTIONS_JUDGE
    metrics = _metrics(_ACTIONS_JUDGE)
    primary_metric = ConversationScoreMetric(scorer=_ACTIONS_JUDGE)

    def system_prompt(self, sample: dict[str, Any]) -> str:
        # The function list is written as a Python literal, as the reference formats it
        return ACTIONS_SYSTEM_PROMPT.replace("[[FUNCTIONS]]", f"{sample['function']}")

    def full_prompt(self, sample: dict[str, Any]) -> str:
        return ACTIONS_FULL_PROMPT.replace(
            "[[QUESTION]]", sample["fully_specified_question"][0][0]["content"]
        )


_DATA2TEXT_JUDGE = Data2TextJudge()


@register("lost_in_conversation_data2text")
class LostInConversationData2Text(_LostInConversation):
    """ToTTo table descriptions, refined as the table and its context arrive turn by turn."""

    task_type = "data2text"
    judge = _DATA2TEXT_JUDGE
    metrics = _metrics(_DATA2TEXT_JUDGE)
    primary_metric = ConversationScoreMetric(scorer=_DATA2TEXT_JUDGE)

    def system_prompt(self, sample: dict[str, Any]) -> str:
        return DATA2TEXT_SYSTEM_PROMPT

    def full_prompt(self, sample: dict[str, Any]) -> str:
        context = "".join(f"{key}: {value}\n" for key, value in sample["metadata"].items())
        return (
            DATA2TEXT_FULL_PROMPT.replace("[[TABLE_HTML]]", sample["table_highlighted_html"])
            .replace("[[FEWSHOT_DESCRIPTIONS]]", sample["fewshot_descriptions"])
            .replace("[[CONTEXT]]", context)
        )


_SUMMARY_JUDGE = SummaryJudge()


def _documents(sample: dict[str, Any], indices: Sequence[int]) -> str:
    texts = {doc["document_index"]: doc["document_text"] for doc in sample["documents"]}
    return "".join(f"Document {index}:\n{texts[index]}\n\n" for index in indices)


@register("lost_in_conversation_summary")
class LostInConversationSummary(_LostInConversation):
    """Summary of a Haystack summaries, rewritten as documents arrive turn by turn."""

    task_type = "summary"
    judge = _SUMMARY_JUDGE
    metrics = _metrics(
        _SUMMARY_JUDGE,
        *(
            ConversationScoreMetric(name=f"domain__{domain}__score", scorer=_SUMMARY_JUDGE)
            for domain in ("conv", "news")
        ),
    )
    primary_metric = ConversationScoreMetric(scorer=_SUMMARY_JUDGE)

    def system_prompt(self, sample: dict[str, Any]) -> str:
        return SUMMARY_SYSTEM_PROMPT

    def _summary_prompt(self, sample: dict[str, Any], documents: str) -> str:
        template = (
            SUMMARY_FULL_PROMPT_CONV if sample["domain"] == "conv" else SUMMARY_FULL_PROMPT_NEWS
        )
        return (
            template.replace("[[TOPIC]]", sample["topic"])
            .replace("[[DOCUMENTS]]", documents)
            .replace("[[QUERY]]", sample["query"])
            .replace("[[N_DOCS]]", str(len(sample["documents"])))
            .replace("[[N_INSIGHTS]]", str(len(sample["insights"])))
        )

    def full_prompt(self, sample: dict[str, Any]) -> str:
        indices = [doc["document_index"] for doc in sample["documents"]]
        return self._summary_prompt(sample, _documents(sample, indices))

    def sharded_turns(self, sample: dict[str, Any]) -> list[str]:
        """The first shard's documents in the full prompt, then each shard's, from the first.

        As in the reference's ``populate_sharded_prompt``, the first shard's documents are
        sent again in the second turn, so the conversation has one more turn than shards.
        """
        shards = sample["shards"]
        first = self._summary_prompt(sample, _documents(sample, shards[0]["doc_idxs"]))
        follow_ups = [
            SUMMARY_FOLLOW_UP_PROMPT + _documents(sample, shard["doc_idxs"]) for shard in shards
        ]
        return [first, *follow_ups]

    def subset_metadata(self, sample: dict[str, Any]) -> dict[str, Any]:
        return {"domain": sample["domain"]}


# =============================================================================
# Variant Registrations
# =============================================================================

TASK_NAMES = (
    "lost_in_conversation_math",
    "lost_in_conversation_code",
    "lost_in_conversation_actions",
    "lost_in_conversation_data2text",
    "lost_in_conversation_summary",
)

for _task_name in TASK_NAMES:
    register_variant(_task_name, FULL, formatter=FULL_FORMATTER)
    register_judge_panel_variant(_task_name)
