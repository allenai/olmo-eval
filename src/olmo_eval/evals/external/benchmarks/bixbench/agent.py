"""The BixBench notebook agent: upstream's tools and rollout over this repo's providers.

Follows the v1.5 open-answer runs (``bixbench/run_configuration/*_no_image.yaml`` of
Future-House/BixBench at 49311180): ldp 0.26's ``SimpleAgent`` with
``hide_old_env_states``, fhaviary 0.19's tool execution, and data-analysis-crow
v1.5.0's ``DataAnalysisEnv`` in Docker mode.

Each step is one model call and the environment's reply to it:

- The model sees the problem, the notebook as markdown and the system prompt, in
  that order as upstream's environment sends them, then each of its earlier replies
  with their tool results and the notebook render that followed. Only the newest
  render is shown; older ones are replaced by a placeholder. A chat template that
  requires the system message first cannot run this eval.
- The calls of a reply run one after another. ``edit_cell`` appends or replaces a
  code cell, saves the notebook and reruns all of it with ``jupyter nbconvert
  --execute --allow-errors``, so cell errors land in outputs. ``list_workdir``
  returns the working directory as nested JSON. ``submit_answer`` records the
  answer and ends the episode after the reply's calls have run.
- A tool that raises is answered with the exception's message, as aviary does with
  ``handle_tool_exc``; a notebook run that times out ends the episode without an
  answer.
- The episode ends after ``max_steps`` model calls.

Differences from upstream:

- Upstream asks for a tool call on every reply (``tool_choice="required"``); the
  providers here do not send ``tool_choice``, so a reply may have no call. Such a
  reply uses its step and gets the notebook render, as an empty action would.
- Images in cell outputs are not sent to the model: the markdown keeps upstream's
  numbered placeholders. The prompt asks the agent to avoid plots.
"""

from __future__ import annotations

import json
import logging
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.common.types.tools import Function, ToolCall, ToolResult, ToolSchema
from olmo_eval.common.types.trajectory import AgentTrajectory, AgentTurn
from olmo_eval.harness.scaffolds.stirrup import run_command
from olmo_eval.inference.errors import is_context_overflow_output, request_error

from . import notebook as nb
from .prompts import CAPSULE_SYSTEM_PROMPT_OPEN, LANGUAGE, build_problem

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox import SandboxExecutor
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

#: Where the working directory is mounted in the sandbox, as upstream mounts it.
CONTAINER_WORKDIR = "/workspace"
#: Upstream's ``max_steps`` for its v1.5 runs.
DEFAULT_MAX_STEPS = 20
#: Upstream's ``NBEnvironment.EXEC_TIMEOUT``, in seconds, for one run of the notebook.
DEFAULT_EXEC_TIMEOUT = 1200.0

HIDDEN_ENV_STATE = "[Previous environment state - hidden]"
ENV_STATE_HEADER = "Markdown representation of notebook contents ({path}):\n\n"

EDIT_CELL = "edit_cell"
LIST_WORKDIR = "list_workdir"
SUBMIT_ANSWER = "submit_answer"

#: The tool schemas fhaviary 0.19's ``Tool.from_function`` builds from upstream's
#: ``edit_cell``, ``list_workdir`` and ``DataAnalysisEnv.submit_answer``.
TOOL_SCHEMAS = (
    ToolSchema(
        name=EDIT_CELL,
        description=(
            "Edit the notebook by modifying a specific code cell.\n\n"
            "ONLY CODE CELLS ARE SUPPORTED. Do no attempt to write Markdown or raw text,\n"
            "though you are permitted (and encouraged) to write comments in the code cells.\n"
            "The notebook will be automatically rerun if a successful edit is made."
        ),
        parameters={
            "type": "object",
            "properties": {
                "contents": {
                    "description": "Cell contents to insert. We assume the cell is a code block.",
                    "title": "Contents",
                    "type": "string",
                },
                "idx": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "description": (
                        "Index of the cell to edit. If not provided (None default),\n"
                        "then appends a new cell."
                    ),
                    "title": "Idx",
                },
            },
            "required": ["contents"],
        },
    ),
    ToolSchema(
        name=LIST_WORKDIR,
        description=(
            "Recursively lists the contents of the working directory.\n\n"
            "The contents is represented as a nested JSON dictionary."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSchema(
        name=SUBMIT_ANSWER,
        description=(
            "Submit an answer to the problem.\n\n"
            "Note that this tool may only be called once and ends the episode."
        ),
        parameters={
            "type": "object",
            "properties": {
                "answer": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"},
                        {"additionalProperties": True, "type": "object"},
                        {"type": "null"},
                    ],
                    "description": "The answer to the problem",
                    "title": "Answer",
                }
            },
            "required": ["answer"],
        },
    ),
)

#: Counts kept in the result to show how an episode spent its steps.
DIAGNOSTIC_COUNTS = (
    "no_tool_call_steps",
    "invalid_tool_calls",
    "tool_exceptions",
    "notebook_runs",
    "notebook_run_failures",
    "length_truncated_replies",
)

DEFAULT_SAMPLING_PARAMS = SamplingParams(
    temperature=1.0, max_tokens=16384, fit_max_tokens_to_context=True
)


class NotebookTimeoutError(TimeoutError):
    """A run of the notebook outlasted its timeout, which ends the episode."""


# Signatures of upstream's tools, called with the model's arguments so that a bad
# call fails with the TypeError Python gives upstream's methods.
def _edit_cell_args(contents: str, idx: int | None = None) -> tuple[str, Any]:
    return contents, idx


def _list_workdir_args() -> None:
    return None


def _submit_answer_args(answer: Any) -> Any:
    return answer


@dataclass
class EpisodeResult:
    """How one agent episode went and what it submitted."""

    #: ``submitted``, ``max_steps``, ``exec_timeout``, ``context_overflow`` or ``error``.
    completion_reason: str = "error"
    submitted: bool = False
    answer: Any = None
    steps: int = 0
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(DIAGNOSTIC_COUNTS, 0))
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0}
    )
    trajectory: AgentTrajectory = field(default_factory=lambda: AgentTrajectory(turns=()))
    duration_s: float = 0.0
    #: A failure of the model server or the sandbox, which leaves the episode unscored.
    error: str | None = None


class NotebookEpisode:
    """One agent episode on one question, with its notebook in ``workdir``.

    ``workdir`` is on this host and is mounted at ``/workspace`` in the sandbox
    that ``executor`` runs commands in.
    """

    def __init__(
        self,
        question: str,
        workdir: Path,
        executor: SandboxExecutor,
        max_steps: int = DEFAULT_MAX_STEPS,
        exec_timeout: float = DEFAULT_EXEC_TIMEOUT,
        label: str = "bixbench",
    ) -> None:
        self.question = question
        self.workdir = workdir
        self.executor = executor
        self.max_steps = max_steps
        self.exec_timeout = exec_timeout
        self.label = label
        self.nb_path = workdir / nb.NOTEBOOK_NAME
        self.notebook = nb.new_notebook()
        self.messages: list[dict[str, Any]] = []
        #: Indices of the notebook-render messages in ``messages``.
        self._env_state_indices: list[int] = []
        self.turns: list[AgentTurn] = []
        self.done = False
        self.result = EpisodeResult()

    async def run(
        self, provider: InferenceProvider, sampling_params: SamplingParams | None = None
    ) -> EpisodeResult:
        """Run the episode to its end. A model or sandbox failure is recorded, not raised."""
        started = time.monotonic()
        params = sampling_params or DEFAULT_SAMPLING_PARAMS
        result = self.result
        problem = build_problem(self.question)
        nb.write_notebook(self.notebook, self.nb_path)
        self._add_user(problem)
        self._add_env_state()
        self.messages.append({"role": "system", "content": CAPSULE_SYSTEM_PROMPT_OPEN})
        result.completion_reason = "max_steps"
        try:
            while result.steps < self.max_steps:
                replies = await self._generate(provider, params)
                if is_context_overflow_output(replies):
                    result.completion_reason = "context_overflow"
                    break
                failure = request_error(replies)
                if failure is not None or not replies:
                    result.completion_reason = "error"
                    result.error = f"agent_error: {failure or 'the provider returned no output'}"
                    break
                result.steps += 1
                output = replies[0]
                self._accumulate_usage(output)
                await self._step(output)
                if self.done:
                    if result.completion_reason not in ("submitted", "exec_timeout"):
                        result.completion_reason = "submitted"
                    break
        except Exception as e:
            logger.exception(f"[{self.label}] Episode failed")
            result.completion_reason = "error"
            result.error = f"sandbox_error: {type(e).__name__}: {e}"
        result.duration_s = round(time.monotonic() - started, 1)
        result.trajectory = AgentTrajectory(
            turns=tuple(self.turns),
            metadata={
                "completion_reason": result.completion_reason,
                "steps": result.steps,
                **result.counts,
            },
        )
        logger.info(
            f"[{self.label}] {result.completion_reason} after {result.steps} steps "
            f"({', '.join(f'{k}={v}' for k, v in result.counts.items() if v)})"
        )
        return result

    def _add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.turns.append(AgentTurn.user(content=content))

    def _add_env_state(self) -> None:
        """Show the current notebook, hiding every earlier render, as ldp does."""
        for index in self._env_state_indices:
            self.messages[index] = {"role": "user", "content": HIDDEN_ENV_STATE}
        markdown, _images = nb.view_notebook(self.notebook, LANGUAGE)
        path = f"{CONTAINER_WORKDIR}/{nb.NOTEBOOK_NAME}"
        self._env_state_indices.append(len(self.messages))
        self._add_user(ENV_STATE_HEADER.format(path=path) + markdown)

    async def _generate(
        self, provider: InferenceProvider, params: SamplingParams
    ) -> list[LMOutput]:
        outputs = await provider.agenerate(
            [
                LMRequest(
                    request_type=RequestType.CHAT,
                    messages=tuple(self.messages),
                    tools=TOOL_SCHEMAS,
                )
            ],
            params,
        )
        return list(outputs[0]) if outputs else []

    def _accumulate_usage(self, output: LMOutput) -> None:
        metadata = output.metadata or {}
        for key in self.result.usage:
            value = metadata.get(key)
            if isinstance(value, int | float):
                self.result.usage[key] += int(value)
        if metadata.get("finish_reason") == "length":
            self.result.counts["length_truncated_replies"] += 1

    async def _step(self, output: LMOutput) -> None:
        """Record a reply, run its tool calls in order, and show the notebook again."""
        text = output.text or ""
        calls = list(output.tool_calls or ())
        self.turns.append(AgentTurn.assistant(content=text, tool_calls=calls))
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = [_replayable(call).to_openai() for call in calls]
        self.messages.append(message)
        if not calls:
            self.result.counts["no_tool_call_steps"] += 1

        results = [await self._call_tool(call) for call in calls]
        for tool_result in results:
            self.messages.append(tool_result.to_openai())
        if results:
            self.turns.append(AgentTurn.tool(results))
        self._add_env_state()

    async def _call_tool(self, call: ToolCall) -> ToolResult:
        name = call.function.name
        if name not in (EDIT_CELL, LIST_WORKDIR, SUBMIT_ANSWER):
            self.result.counts["invalid_tool_calls"] += 1
            return ToolResult(
                tool_call_id=call.id, content=f"Invalid tool call: {name}", is_error=True
            )
        try:
            arguments = _parse_arguments(call)
            if name == EDIT_CELL:
                contents, idx = _edit_cell_args(**arguments)
                content = await self._edit_cell(contents, idx)
            elif name == LIST_WORKDIR:
                _list_workdir_args(**arguments)
                content = json.dumps(nb.list_dir(self.workdir), indent=2)
            else:
                content = self._submit_answer(_submit_answer_args(**arguments))
        except NotebookTimeoutError as e:
            self.result.counts["tool_exceptions"] += 1
            self.done = True
            self.result.completion_reason = "exec_timeout"
            return _exception_result(call, e)
        except (TypeError, ValueError) as e:
            self.result.counts["tool_exceptions"] += 1
            return _exception_result(call, e)
        return ToolResult(tool_call_id=call.id, content=content)

    async def _edit_cell(self, contents: str, idx: Any) -> str:
        try:
            return nb.edit_cell(self.notebook, contents, idx)
        finally:
            # As upstream, the notebook is saved and rerun even when the edit failed,
            # and a failed run replaces the edit's reply with its exception.
            nb.write_notebook(self.notebook, self.nb_path)
            await self._run_notebook()

    async def _run_notebook(self) -> None:
        """Run every cell with nbconvert in the sandbox and load the executed notebook.

        Raises:
            NotebookTimeoutError: If the run outlasts the timeout.
            ValueError: If nbconvert exits with an error, as upstream raises it.
        """
        self.result.counts["notebook_runs"] += 1
        path = f"{CONTAINER_WORKDIR}/{nb.NOTEBOOK_NAME}"
        command = (
            f"cd {CONTAINER_WORKDIR} && jupyter nbconvert --to notebook --execute --inplace "
            f"{shlex.quote(path)} --allow-errors"
        )
        run = await run_command(self.executor, command, self.exec_timeout)
        if run.timed_out:
            self.result.counts["notebook_run_failures"] += 1
            raise NotebookTimeoutError(
                f"Notebook execution timed out after {self.exec_timeout} seconds"
            )
        if run.exit_code != 0:
            self.result.counts["notebook_run_failures"] += 1
            logger.warning(f"[{self.label}] nbconvert exited {run.exit_code}: {run.stderr[-2000:]}")
            raise ValueError(f"Error executing the notebook in Docker (exit code: {run.exit_code})")
        self.notebook = nb.read_notebook(self.nb_path)

    def _submit_answer(self, answer: Any) -> str:
        self.result.answer = answer
        self.result.submitted = True
        self.done = True
        return f"Submitted answer: {answer}"


def _parse_arguments(call: ToolCall) -> dict[str, Any]:
    """A call's arguments as a dict.

    Raises:
        ValueError: If the arguments are not a JSON object.
    """
    raw = call.function.arguments
    if isinstance(raw, dict):
        return raw
    try:
        arguments = json.loads(raw) if raw else {}
    except (TypeError, ValueError) as e:
        raise ValueError(f"Tool call arguments are not valid JSON: {e}") from e
    if not isinstance(arguments, dict):
        raise ValueError("Tool call arguments must be a JSON object")
    return arguments


def _exception_result(call: ToolCall, error: Exception) -> ToolResult:
    """The reply aviary gives a tool call that raised."""
    return ToolResult(
        tool_call_id=call.id,
        content=f"Encountered exception during tool call: {error}",
        is_error=True,
    )


def _replayable(call: ToolCall) -> ToolCall:
    """The call with arguments an OpenAI-compatible server accepts back in the history."""
    try:
        _parse_arguments(call)
    except ValueError:
        return ToolCall(id=call.id, function=Function(name=call.function.name, arguments="{}"))
    if not call.function.arguments:
        return ToolCall(id=call.id, function=Function(name=call.function.name, arguments="{}"))
    return call
