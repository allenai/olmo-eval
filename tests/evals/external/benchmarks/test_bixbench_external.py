"""Tests for the BixBench-Verified-50 external evaluation."""

from __future__ import annotations

import asyncio
import json
import re
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from olmo_eval.common.execution import ExecutionResult
from olmo_eval.common.types import LMOutput
from olmo_eval.common.types.tools import ToolCall
from olmo_eval.evals.external.benchmarks.bixbench import notebook as nb
from olmo_eval.evals.external.benchmarks.bixbench.agent import (
    HIDDEN_ENV_STATE,
    TOOL_SCHEMAS,
    NotebookEpisode,
)
from olmo_eval.evals.external.benchmarks.bixbench.eval import (
    BixBenchArgs,
    TrialResult,
    compute_metrics,
)
from olmo_eval.evals.external.benchmarks.bixbench.grader import (
    Grade,
    answer_text,
    grade_answer,
    parse_grade,
    string_match,
)
from olmo_eval.evals.external.benchmarks.bixbench.loader import (
    BixBenchQuestion,
    extract_capsule,
    load_questions,
    select_questions,
    stage_workdir,
)
from olmo_eval.evals.external.benchmarks.bixbench.prompts import (
    CAPSULE_SYSTEM_PROMPT_OPEN,
    build_problem,
)
from olmo_eval.evals.external.registry import get_external_eval
from olmo_eval.evals.external.result import ExternalEvalResult
from olmo_eval.harness.sandbox import DetachedRun
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.errors import REQUEST_ERROR_KEY

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox import SandboxExecutor


def question(
    eval_mode: str = "llm_verifier", ideal: str = "CD14", question_id: str = "bix-1-q1"
) -> BixBenchQuestion:
    return BixBenchQuestion(
        question_id=question_id,
        question="Which gene is most upregulated?",
        ideal=ideal,
        eval_mode=eval_mode,
        capsule_uuid="uuid",
        data_folder="CapsuleFolder-uuid.zip",
    )


class TestLoader:
    def test_rows_load_and_unknown_eval_modes_are_rejected(self, tmp_path: Path) -> None:
        row = {
            "question_id": "bix-1-q1",
            "question": "Q?",
            "ideal": 0.5,
            "eval_mode": "str_verifier",
            "capsule_uuid": "u",
            "data_folder": "CapsuleFolder-u.zip",
        }
        path = tmp_path / "q.jsonl"
        path.write_text(json.dumps(row) + "\n")
        (loaded,) = load_questions(path)
        assert loaded.ideal == "0.5"
        path.write_text(json.dumps({**row, "eval_mode": "mcq"}) + "\n")
        with pytest.raises(ValueError, match="unknown eval_mode"):
            load_questions(path)

    def test_selection_keeps_dataset_order_and_rejects_unknown_ids(self) -> None:
        questions = [question(question_id=f"bix-1-q{i}") for i in range(3)]
        selected = select_questions(questions, ["bix-1-q2", "bix-1-q0"])
        assert [q.question_id for q in selected] == ["bix-1-q0", "bix-1-q2"]
        assert select_questions(questions, None) == questions
        with pytest.raises(ValueError, match="bix-9-q9"):
            select_questions(questions, ["bix-9-q9"])

    def test_staged_workdir_is_writable_by_any_user(self, tmp_path: Path) -> None:
        capsule = tmp_path / "capsule"
        (capsule / "sub").mkdir(parents=True)
        (capsule / "sub").chmod(0o755)
        (capsule / "counts.csv").write_text("a\n")
        (capsule / "counts.csv").chmod(0o644)
        (capsule / "run.sh").write_text("true\n")
        (capsule / "run.sh").chmod(0o755)
        workdir = tmp_path / "work"
        stage_workdir(capsule, workdir)
        assert workdir.stat().st_mode & 0o777 == 0o777
        assert (workdir / "sub").stat().st_mode & 0o777 == 0o777
        assert (workdir / "counts.csv").stat().st_mode & 0o777 == 0o666
        assert (workdir / "run.sh").stat().st_mode & 0o777 == 0o777

    def test_extraction_keeps_only_the_input_data(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "CapsuleFolder-u.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("CapsuleData-u/counts.csv", "a,b\n")
            archive.writestr("CapsuleData-u/sub/meta.txt", "x")
            archive.writestr("CapsuleNotebook-u/CapsuleNotebook-u_executed.ipynb", "{}")
        out = tmp_path / "out"
        extract_capsule(zip_path, out)
        assert sorted(str(p.relative_to(out)) for p in out.rglob("*")) == [
            "counts.csv",
            "sub",
            "sub/meta.txt",
        ]


class TestPrompts:
    def test_problem_fills_question_and_language(self) -> None:
        problem = build_problem("What is {x}?")
        assert "<question>\nWhat is {x}?\n</question>" in problem
        assert "{language}" not in problem
        assert "All cells are by default python cells." in problem
        assert problem.rstrip().endswith(
            "AVOID USING PLOTS/IMAGES. USE TABLES AND PRINT OUTPUTS INSTEAD AS MUCH AS POSSIBLE."
        )
        # The R guidelines' doubled braces are format escapes upstream.
        assert 'if (!requireNamespace("package_name", quietly = TRUE)) {\n' in problem


class TestNotebook:
    def test_edit_cell_appends_edits_and_falls_back_to_append(self) -> None:
        notebook = nb.new_notebook()
        assert nb.edit_cell(notebook, "a = 1") == "Appended new cell (#0)."
        assert nb.edit_cell(notebook, "a = 2", 0) == "Edited cell #0."
        assert nb.edit_cell(notebook, "b = 1", "zero") == "Appended new cell (#1)."
        assert nb.edit_cell(notebook, "c = 1", 7) == "Appended new cell (#2)."
        assert [c["source"] for c in notebook["cells"]] == ["a = 2", "b = 1", "c = 1"]

    def test_render_matches_upstream_layout(self) -> None:
        notebook = nb.new_notebook()
        notebook["cells"] = [
            {
                "cell_type": "code",
                "source": ["import pandas\n", "print(1)"],
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": ["1\n"]},
                    {"output_type": "stream", "name": "stderr", "text": "warn\n"},
                    {"output_type": "execute_result", "data": {"text/plain": "42"}},
                    {"output_type": "display_data", "data": {"text/html": "<table/>"}},
                    {"output_type": "display_data", "data": {"image/png": "aGk="}},
                ],
            },
            {
                "cell_type": "code",
                "source": "1/0",
                "outputs": [
                    {"output_type": "error", "traceback": ["Traceback", "ZeroDivisionError"]},
                ],
            },
            {"cell_type": "code", "source": "x = 1", "outputs": []},
        ]
        markdown, images = nb.view_notebook(notebook)
        assert markdown == "\n".join(
            [
                "### Cell 0:",
                "```python",
                "import pandas\nprint(1)",
                "```",
                "### Output 0:",
                "```",
                "42",
                "<1>",
                "1\n\nwarn\n",
                "```",
                "### Cell 1:",
                "```python",
                "1/0",
                "```",
                "### Output 1:",
                "```",
                "Traceback\nZeroDivisionError",
                "```",
                "### Cell 2:",
                "```python",
                "x = 1",
                "```",
            ]
        )
        assert images == ["data:image/png;base64,aGk="]

    def test_long_outputs_keep_both_ends(self) -> None:
        text = "a" * 2000 + "b" * 2000
        limited = nb.limit_notebook_output(text)
        assert limited == "a" * 1500 + "\n<...output limited...>\n" + "b" * 1500
        assert nb.limit_notebook_output("short") == "short"

    def test_listing_has_upstream_shape(self, tmp_path: Path) -> None:
        (tmp_path / "b.csv").write_text("")
        (tmp_path / "a.csv").write_text("")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.txt").write_text("")
        assert nb.list_dir(tmp_path) == {
            "files": ["a.csv", "b.csv"],
            "directories": {},
            "sub": {"files": ["c.txt"]},
        }


class FakeJudge:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[Any] = []
        self.judge_config = {"model": "fake"}

    async def __call__(self, prompt: Any, *, system_prompt: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)


def grade(q: BixBenchQuestion, answer: Any, judge: FakeJudge) -> Grade:
    return asyncio.run(grade_answer(q, answer, judge))


class TestGrader:
    def test_string_match_is_upstreams_cleaned_exact_then_substring(self) -> None:
        assert string_match("CD14", " cd-14 ") == "exact"
        # Upstream's partial match checks the answer inside the target.
        assert string_match("11", "1") == "substring"
        assert string_match("CD14", "<answer>CD14</answer>") is None

    def test_str_verifier_match_needs_no_judge(self) -> None:
        judge = FakeJudge([])
        result = grade(question("str_verifier", "CD14"), "CD14", judge)
        assert result.correct is True and result.method == "exact"
        assert judge.prompts == []

    def test_str_verifier_falls_back_to_the_judge(self) -> None:
        judge = FakeJudge(["<grade> correct </grade>"])
        result = grade(question("str_verifier", "CD14"), "<answer>CD14</answer>", judge)
        assert result.correct is True and result.method == "judge"
        assert "Predicted Answer: <answer>CD14</answer>" in judge.prompts[0]
        assert "Target Answer: CD14" in judge.prompts[0]

    def test_range_verifier_uses_the_generic_prompt_with_the_range(self) -> None:
        judge = FakeJudge(["<grade>incorrect</grade>"])
        result = grade(question("range_verifier", "(0.7, 0.8)"), "0.9", judge)
        assert result.correct is False and result.judge_grade == "incorrect"
        assert "Target Answer: (0.7, 0.8)" in judge.prompts[0]

    def test_refusal_is_incorrect(self) -> None:
        result = grade(question(), "I cannot tell", FakeJudge(["<grade>refused</grade>"]))
        assert result.correct is False and result.judge_grade == "refused"

    def test_no_submission_is_incorrect_without_a_call(self) -> None:
        judge = FakeJudge([])
        result = grade(question(), None, judge)
        assert result.correct is False and result.method == "no_answer"
        assert judge.prompts == []

    def test_reply_without_a_grade_is_asked_again_then_left_ungraded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def no_sleep(_: float) -> None:
            return None

        monkeypatch.setattr(asyncio, "sleep", no_sleep)
        judge = FakeJudge(["no tags", "<grade>correct</grade>"])
        assert grade(question(), "x", judge).correct is True
        judge = FakeJudge(["no tags"] * 10)
        result = grade(question(), "x", judge)
        assert result.correct is None
        assert result.error is not None and result.error.startswith("judge_parse_error")

    def test_answer_text_unwraps_single_key_dicts(self) -> None:
        assert answer_text({"answer": "CD14"}) == "CD14"
        assert answer_text(3.5) == "3.5"
        assert answer_text({"a": 1, "b": 2}) == '{"a": 1, "b": 2}'
        assert parse_grade("<grade>\n Correct \n</grade>") == "correct"


class ScriptedProvider(InferenceProvider):
    """Returns prepared replies in order, recording the messages it was sent."""

    supports_tools = True

    def __init__(self, replies: list[LMOutput]) -> None:
        super().__init__("scripted")
        self.replies = list(replies)
        self.requests: list[Any] = []

    def generate(self, requests, sampling_params=None):  # pragma: no cover - unused
        raise NotImplementedError

    def logprobs(self, requests, sampling_params=None):  # pragma: no cover - unused
        raise NotImplementedError

    async def agenerate(self, requests, sampling_params=None):
        self.requests.append(requests[0])
        reply = self.replies.pop(0) if self.replies else LMOutput(text="thinking")
        return [[reply]]


class FakeNotebookExecutor:
    """Plays nbconvert: each code cell gets one stream output naming its source."""

    def __init__(self, workdir: Path, outcomes: list[str] | None = None) -> None:
        self.workdir = workdir
        self.outcomes = list(outcomes or [])
        self.scripts: list[str] = []

    async def run_detached(self, script: str, timeout: float) -> DetachedRun:
        self.scripts.append(script)
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        run_dir = f"/tmp/.detached/{len(self.scripts)}"
        if outcome == "timeout":
            return DetachedRun(run_dir, None, timed_out=True)
        if outcome == "crash":
            return DetachedRun(run_dir, 1)
        path = self.workdir / nb.NOTEBOOK_NAME
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            cell["outputs"] = [
                {"output_type": "stream", "name": "stdout", "text": f"ran {cell['source']}\n"}
            ]
        path.write_text(json.dumps(notebook))
        return DetachedRun(run_dir, 0)

    async def execute_command(self, command, timeout=None, **_):
        return ExecutionResult(True, "")


def call(name: str, arguments: dict[str, Any] | str, call_id: str = "c1") -> ToolCall:
    return ToolCall.create(call_id, name, arguments)


def reply(*calls: ToolCall, text: str = "") -> LMOutput:
    return LMOutput(text=text, tool_calls=list(calls), metadata={"prompt_tokens": 5})


def run_episode(
    tmp_path: Path, replies: list[LMOutput], outcomes: list[str] | None = None, **kwargs: Any
):
    (tmp_path / "data.csv").write_text("x\n1\n")
    executor = FakeNotebookExecutor(tmp_path, outcomes)
    provider = ScriptedProvider(replies)
    episode = NotebookEpisode("Which gene?", tmp_path, cast("SandboxExecutor", executor), **kwargs)
    result = asyncio.run(episode.run(provider))
    return result, provider, executor


def contents(message: dict[str, Any]) -> str:
    return str(message.get("content", ""))


class TestEpisode:
    def test_tools_are_upstreams(self) -> None:
        assert [t.name for t in TOOL_SCHEMAS] == ["edit_cell", "list_workdir", "submit_answer"]
        assert TOOL_SCHEMAS[0].parameters["required"] == ["contents"]

    def test_every_request_requires_a_tool_call(self, tmp_path: Path) -> None:
        _, provider, _ = run_episode(tmp_path, [reply(call("list_workdir", {}))], max_steps=2)
        assert [r.tool_choice for r in provider.requests] == ["required", "required"]

    def test_edit_runs_the_notebook_and_submit_ends_the_episode(self, tmp_path: Path) -> None:
        result, provider, executor = run_episode(
            tmp_path,
            [
                reply(call("edit_cell", {"contents": "print('hi')"})),
                reply(call("submit_answer", {"answer": "<answer>CD14</answer>"}, "c2")),
            ],
        )
        assert result.completion_reason == "submitted"
        assert result.answer == "<answer>CD14</answer>"
        assert result.steps == 2
        assert "jupyter nbconvert --to notebook --execute --inplace" in executor.scripts[0]
        assert "--allow-errors" in executor.scripts[0]

        first = provider.requests[0].messages
        # Upstream's order: the problem, the first notebook render, then the system prompt.
        assert [m["role"] for m in first] == ["user", "user", "system"]
        assert contents(first[2]) == CAPSULE_SYSTEM_PROMPT_OPEN
        assert contents(first[1]) == (
            "Markdown representation of notebook contents (/workspace/notebook.ipynb):\n\n"
        )
        second = provider.requests[1].messages
        assert contents(second[1]) == HIDDEN_ENV_STATE
        assert [contents(m) for m in second if m["role"] == "tool"] == ["Appended new cell (#0)."]
        assert "ran print('hi')" in contents(second[-1])
        assert json.loads((tmp_path / nb.NOTEBOOK_NAME).read_text())["cells"][0]["outputs"]
        # The sandbox rewrites the notebook, possibly as another user.
        assert (tmp_path / nb.NOTEBOOK_NAME).stat().st_mode & 0o777 == 0o666

    def test_system_first_moves_only_the_system_prompt(self, tmp_path: Path) -> None:
        _, provider, _ = run_episode(
            tmp_path,
            [
                reply(call("edit_cell", {"contents": "print('hi')"})),
                reply(call("submit_answer", {"answer": "<answer>CD14</answer>"}, "c2")),
            ],
            system_first=True,
        )
        first = provider.requests[0].messages
        assert [m["role"] for m in first] == ["system", "user", "user"]
        assert contents(first[0]) == CAPSULE_SYSTEM_PROMPT_OPEN
        assert contents(first[2]).startswith("Markdown representation of notebook contents")
        second = provider.requests[1].messages
        assert contents(second[0]) == CAPSULE_SYSTEM_PROMPT_OPEN
        assert contents(second[2]) == HIDDEN_ENV_STATE
        assert "ran print('hi')" in contents(second[-1])

    def test_listing_reports_the_workdir(self, tmp_path: Path) -> None:
        _, provider, _ = run_episode(tmp_path, [reply(call("list_workdir", {}))], max_steps=2)
        listing = json.loads(
            contents(next(m for m in provider.requests[1].messages if m["role"] == "tool"))
        )
        assert listing == {"files": ["data.csv", "notebook.ipynb"]}

    def test_step_limit_without_a_submission(self, tmp_path: Path) -> None:
        result, _, _ = run_episode(tmp_path, [reply(), reply()], max_steps=2)
        assert result.completion_reason == "max_steps"
        assert result.submitted is False and result.answer is None
        assert result.counts["no_tool_call_steps"] == 2

    def test_bad_calls_get_upstreams_messages(self, tmp_path: Path) -> None:
        _, provider, _ = run_episode(
            tmp_path,
            [
                reply(
                    call("run_shell", {"cmd": "ls"}, "c1"),
                    call("edit_cell", {"idx": 0}, "c2"),
                    call("edit_cell", "{not json", "c3"),
                )
            ],
            max_steps=2,
        )
        tools = [contents(m) for m in provider.requests[1].messages if m["role"] == "tool"]
        assert tools[0] == "Invalid tool call: run_shell"
        assert re.match(
            r"Encountered exception during tool call: .*missing 1 required positional "
            r"argument: 'contents'",
            tools[1],
        )
        assert tools[2].startswith("Encountered exception during tool call: Tool call arguments")

    def test_edit_before_the_first_cell_gets_upstreams_message(self, tmp_path: Path) -> None:
        result, provider, _ = run_episode(
            tmp_path, [reply(call("edit_cell", {"contents": "x", "idx": -3}))], max_steps=2
        )
        tool = next(m for m in provider.requests[1].messages if m["role"] == "tool")
        assert contents(tool) == "Encountered exception during tool call: list index out of range"
        assert result.counts["tool_exceptions"] == 1
        assert result.completion_reason == "max_steps"

    def test_failed_run_replaces_the_edit_reply(self, tmp_path: Path) -> None:
        result, provider, _ = run_episode(
            tmp_path, [reply(call("edit_cell", {"contents": "x"}))], ["crash"], max_steps=2
        )
        tool = next(m for m in provider.requests[1].messages if m["role"] == "tool")
        assert contents(tool) == (
            "Encountered exception during tool call: "
            "Error executing the notebook in Docker (exit code: 1)"
        )
        assert result.counts["notebook_run_failures"] == 1
        assert result.completion_reason == "max_steps"

    def test_notebook_timeout_ends_the_episode(self, tmp_path: Path) -> None:
        result, _, _ = run_episode(
            tmp_path, [reply(call("edit_cell", {"contents": "x"}))], ["timeout"]
        )
        assert result.completion_reason == "exec_timeout"
        assert result.answer is None and result.steps == 1

    def test_provider_failure_is_an_error_not_a_score(self, tmp_path: Path) -> None:
        failed = LMOutput(text="", metadata={REQUEST_ERROR_KEY: "server down"})
        result, _, _ = run_episode(tmp_path, [failed])
        assert result.completion_reason == "error"
        assert result.error == "agent_error: server down"


def trial(attempt: int, correct: bool | None, reason: str = "submitted", **kw: Any) -> TrialResult:
    t = TrialResult(question=question(**kw), attempt=attempt, completion_reason=reason)
    t.grade = Grade(correct=correct, method="judge")
    return t


class TestMetrics:
    def test_accuracy_spread_and_endings(self) -> None:
        trials = [
            trial(0, True),
            trial(0, False, "max_steps", eval_mode="str_verifier"),
            trial(1, True),
            trial(1, True, eval_mode="str_verifier"),
        ]
        failed = trial(1, None)
        failed.error = "sandbox_error: boom"
        metrics = compute_metrics([*trials, failed])
        assert metrics["accuracy"] == 0.75
        assert metrics["accuracy_attempt_0"] == 0.5
        assert metrics["accuracy_attempt_1"] == 1.0
        assert metrics["accuracy_std"] == pytest.approx(0.3535533)
        assert metrics["accuracy_str_verifier"] == 0.5
        assert "accuracy_range_verifier" not in metrics
        assert metrics["max_steps_rate"] == 0.25
        assert metrics["num_failed_trials"] == 1.0

    def test_args_defaults_and_registration(self) -> None:
        args = BixBenchArgs.from_dict({"question_ids": "bix-1-q1,bix-2-q2", "n_attempts": "5"})
        assert args.question_ids == ["bix-1-q1", "bix-2-q2"]
        assert args.n_attempts == 5 and args.max_steps == 20
        assert args.judge_model == "gpt-6-luna"
        assert args.system_first is False
        assert BixBenchArgs.from_dict({"system_first": "true"}).system_first is True
        assert get_external_eval("bixbench_verified").name == "bixbench_verified"


class FakeBinding:
    def __init__(self, executor: FakeNotebookExecutor) -> None:
        self.executor = executor

    async def release(self) -> None:
        return None


class FlakySandboxManager:
    """Fails to start a set number of times across all instances, then works."""

    failures_left = 0
    starts = 0

    def __init__(self, configs: list[Any], owner: str = "") -> None:
        self.workdir = Path(configs[0].docker_args[1].split("src=")[1].split(",")[0])

    async def start(self) -> None:
        FlakySandboxManager.starts += 1
        if FlakySandboxManager.failures_left > 0:
            FlakySandboxManager.failures_left -= 1
            raise RuntimeError("only 0/1 required instances started (1 failed)")

    async def acquire_binding(self, capability: Any) -> FakeBinding:
        return FakeBinding(FakeNotebookExecutor(self.workdir))

    async def stop(self) -> None:
        return None


class TestExecute:
    def run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        start_failures: int = 0,
        replies: list[LMOutput] | None = None,
    ) -> ExternalEvalResult:
        import olmo_eval.harness.sandbox as sandbox_module
        from olmo_eval.evals.external.benchmarks.bixbench import eval as eval_module

        rows = tmp_path / "questions.jsonl"
        rows.write_text(
            json.dumps(
                {
                    "question_id": "bix-1-q1",
                    "question": "Which gene?",
                    "ideal": "CD14",
                    "eval_mode": "str_verifier",
                    "capsule_uuid": "u",
                    "data_folder": "CapsuleFolder-u.zip",
                }
            )
            + "\n"
        )
        capsule = tmp_path / "capsule"
        capsule.mkdir()
        (capsule / "counts.csv").write_text("gene\nCD14\n")

        async def no_sleep(_: float) -> None:
            return None

        FlakySandboxManager.failures_left = start_failures
        FlakySandboxManager.starts = 0
        monkeypatch.setattr(sandbox_module, "SandboxManager", FlakySandboxManager)
        monkeypatch.setattr(eval_module, "download_questions", lambda revision: rows)
        monkeypatch.setattr(eval_module, "prepare_capsule", lambda folder, revision: capsule)
        monkeypatch.setattr(eval_module, "build_judge_fn", lambda *a: FakeJudge([]))
        monkeypatch.setattr(
            eval_module.BixBenchVerifiedExternalEval, "_sandbox_image", lambda *a: "img"
        )
        monkeypatch.setattr(eval_module.asyncio, "sleep", no_sleep)
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        provider = ScriptedProvider(
            [*(replies or []), reply(call("submit_answer", {"answer": "CD14"}))],
        )
        return asyncio.run(
            get_external_eval("bixbench_verified").execute(
                provider, {"n_attempts": 1}, output_dir=str(tmp_path / "out")
            )
        )

    def test_a_sandbox_that_fails_to_start_is_started_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self.run(tmp_path, monkeypatch, start_failures=2)
        assert result.success, result.error
        assert FlakySandboxManager.starts == 3
        assert result.metrics["accuracy"] == 1.0

    def test_a_sandbox_that_never_starts_fails_the_trial(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self.run(tmp_path, monkeypatch, start_failures=10)
        assert not result.success
        assert result.error is not None and "sandbox_error" in result.error
        assert result.metrics["num_failed_trials"] == 1.0

    @pytest.mark.parametrize("contents", [None, 7, {"code": "x"}])
    def test_a_cell_that_is_not_text_is_refused_and_the_run_completes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: Any
    ) -> None:
        result = self.run(
            tmp_path, monkeypatch, replies=[reply(call("edit_cell", {"contents": contents}))]
        )
        assert result.success, result.error
        assert result.metrics["accuracy"] == 1.0
        (prediction,) = result.predictions or []
        assert prediction["counts"]["tool_exceptions"] == 1

    def test_a_notebook_that_cannot_be_kept_does_not_fail_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from olmo_eval.evals.external.benchmarks.bixbench import eval as eval_module

        def refuse(*_: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(eval_module.shutil, "copy", refuse)
        result = self.run(
            tmp_path, monkeypatch, replies=[reply(call("edit_cell", {"contents": "x = 1"}))]
        )
        assert result.success, result.error
        assert result.metrics["accuracy"] == 1.0
