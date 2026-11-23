"""Tests for the AutomationBench-AA external evaluation."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from olmo_eval.evals.external.benchmarks.automationbench import eval as ab_eval
from olmo_eval.evals.external.benchmarks.automationbench.eval import (
    DOMAINS,
    AutomationBenchAAExternalEval,
    AutomationBenchArgs,
    TaskResult,
    compute_metrics,
    prepare_environment,
)
from olmo_eval.evals.external.registry import get_external_eval


def load_driver() -> ModuleType:
    """Import the driver script, which lives outside olmo-eval's package tree."""
    spec = importlib.util.spec_from_file_location("ab_driver", ab_eval.DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load_driver()


def classify(assertions: list[dict], initially: set[str], finally_: set[str]) -> dict[str, Any]:
    """Classify assertions whose pass state is given by their ``id``."""
    return driver.classify_assertions(
        assertions, lambda a: a["id"] in initially, lambda a: a["id"] in finally_
    )


def zapier_partial_credit(assertions: list[dict], initially: set[str], finally_: set[str]):
    """AutomationBench's partial_credit branches, restated for the cross-check."""
    passed = total = 0
    for a in assertions:
        result = a["id"] in finally_
        if a.get("scored") is False or a.get("excluded") is True:
            continue
        if a["id"] in initially and a.get("excluded") is not False:
            if not result:
                total += 1
        else:
            total += 1
            passed += int(result)
    return passed / total if total else 0.0


class TestClassifyAssertions:
    def test_objectives_only_scores_share_met(self):
        assertions = [{"type": "t", "id": "a"}, {"type": "t", "id": "b"}]
        out = classify(assertions, initially=set(), finally_={"a"})
        assert (out["objectives"], out["objectives_met"]) == (2, 1)
        assert out["guardrails"] == 0
        assert out["aa_score"] == pytest.approx(0.5)

    def test_broken_guardrail_zeroes_the_task(self):
        assertions = [{"type": "t", "id": "obj"}, {"type": "t", "id": "guard"}]
        out = classify(assertions, initially={"guard"}, finally_={"obj"})
        assert out["guardrails"] == 1
        assert out["guardrail_violations"] == 1
        assert out["objectives_met"] == 1
        assert out["aa_score"] == 0.0
        # partial_credit counts the broken guardrail as a failure: 1 of 2.
        assert out["implied_partial_credit"] == pytest.approx(0.5)

    def test_intact_guardrail_is_left_out_of_the_score(self):
        assertions = [{"type": "t", "id": "obj"}, {"type": "t", "id": "guard"}]
        out = classify(assertions, initially={"guard"}, finally_={"obj", "guard"})
        assert out["guardrail_violations"] == 0
        assert out["aa_score"] == 1.0
        assert out["implied_partial_credit"] == 1.0

    def test_unscored_and_excluded_assertions_are_left_out(self):
        assertions = [
            {"type": "t", "id": "a"},
            {"type": "t", "id": "b", "scored": False},
            {"type": "t", "id": "c", "excluded": True},
        ]
        out = classify(assertions, initially={"c"}, finally_={"a"})
        assert out["excluded"] == 2
        assert out["objectives"] == 1
        assert out["guardrails"] == 0
        assert [r["class"] for r in out["assertions"]] == ["objective", "excluded", "excluded"]

    def test_excluded_false_is_an_objective_even_when_it_passes_initially(self):
        # A "do nothing" task: negative assertions pass at the start and must stay so.
        assertions = [{"type": "t", "id": "a", "excluded": False}]
        kept = classify(assertions, initially={"a"}, finally_={"a"})
        assert (kept["objectives"], kept["guardrails"], kept["aa_score"]) == (1, 0, 1.0)
        broken = classify(assertions, initially={"a"}, finally_=set())
        assert broken["aa_score"] == 0.0

    def test_no_scored_assertions_scores_zero(self):
        out = classify([{"type": "t", "id": "g"}], initially={"g"}, finally_={"g"})
        assert out["objectives"] == 0
        assert out["aa_score"] == 0.0
        assert out["implied_partial_credit"] == 0.0

    @pytest.mark.parametrize(
        ("initially", "finally_"),
        [
            (set(), set()),
            ({"g1"}, {"g1", "o1"}),
            ({"g1", "g2"}, {"o1", "o2", "g2"}),
            ({"g1", "f"}, {"f"}),
            ({"g1", "f"}, {"g1", "o1", "o2"}),
        ],
    )
    def test_implied_partial_credit_matches_automationbench(self, initially, finally_):
        assertions = [
            {"type": "t", "id": "o1"},
            {"type": "t", "id": "o2"},
            {"type": "t", "id": "g1"},
            {"type": "t", "id": "g2"},
            {"type": "t", "id": "f", "excluded": False},
            {"type": "t", "id": "x", "scored": False},
        ]
        out = classify(assertions, initially, finally_)
        assert out["implied_partial_credit"] == pytest.approx(
            zapier_partial_credit(assertions, initially, finally_)
        )

    def test_params_are_kept_without_the_type(self):
        out = classify([{"type": "gmail_sent", "id": "a", "to": "x"}], set(), {"a"})
        assert out["assertions"][0] == {
            "type": "gmail_sent",
            "passed": True,
            "class": "objective",
            "params": {"id": "a", "to": "x"},
        }


class TestDriverRecords:
    def test_describe_error_follows_the_cause_chain(self):
        try:
            try:
                raise ConnectionError("refused")
            except ConnectionError as e:
                raise RuntimeError("request failed") from e
        except RuntimeError as e:
            text = driver.describe_error(e)
        assert text == "RuntimeError: request failed <- ConnectionError: refused"
        assert driver.describe_error(None) is None
        assert driver.describe_error({"error": "x"}) == "{'error': 'x'}"

    def test_task_record_from_a_scored_output(self):
        output = {
            "example_id": 3,
            "info": json.dumps({"task_name": "finance.grant_expense_tracking"}),
            "prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            "completion": [
                {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "content": "done"},
            ],
            "stop_condition": "no_tools_called",
            "metrics": {"partial_credit": 0.5, "task_completed_correctly": 0.0},
            "_perf": {"tool_calls": 1},
            "_task_contract_sha256": "abc",
            "_olmo_eval": {
                "error_kind": None,
                "error": None,
                "scoring_error": None,
                "objectives": 2,
                "objectives_met": 1,
                "aa_score": 0.5,
            },
        }
        record = driver.task_record(output)
        assert record["task_name"] == "finance.grant_expense_tracking"
        assert record["domain"] == "finance"
        assert record["turns"] == 2
        assert record["tool_calls"] == 1
        assert record["partial_credit"] == 0.5
        assert record["aa_score"] == 0.5
        assert record["task_contract_sha256"] == "abc"
        assert len(record["messages"]) == 5

    def test_progress_line_reports_how_tasks_ended(self):
        from collections import Counter

        line = driver.progress_line(
            4, 600, 61.2, 10, Counter({"no_tools_called": 3, "max_turns_reached": 1})
        )
        assert line == (
            "[automationbench] 4/600 tasks done (61s elapsed; mean turns 2.5; "
            "max_turns_reached=1, no_tools_called=3)"
        )

    def test_task_record_without_a_score_is_an_infra_failure(self):
        record = driver.task_record(
            {"info": {"task_name": "hr.x"}, "error": "boom", "completion": []}
        )
        assert record["error_kind"] == driver.INFRA_ERROR
        assert record["error"] == "boom"
        assert record["scoring_error"]
        assert TaskResult.from_record(record).failure is not None


class StubWorld:
    """Stands in for AutomationBench's WorldState: holds the ids that pass in it."""

    def __init__(self, passing: list[str] | None = None, **_: Any):
        self.passing = set(passing or [])


class StubRegistry:
    @staticmethod
    def check(world: StubWorld, assertion: dict) -> bool:
        return assertion["id"] in world.passing


class StubInvalidResponse(Exception):
    pass


@pytest.fixture
def stub_benchmark(monkeypatch):
    """Stub the verifiers and AutomationBench modules olmo_eval_record imports."""
    modules = {
        "verifiers": SimpleNamespace(InvalidModelResponseError=StubInvalidResponse),
        "automationbench": SimpleNamespace(),
        "automationbench.rubric": SimpleNamespace(),
        "automationbench.rubric.registry": SimpleNamespace(AssertionRegistry=StubRegistry),
        "automationbench.schema": SimpleNamespace(),
        "automationbench.schema.world": SimpleNamespace(WorldState=StubWorld),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def rollout_state(**overrides: Any) -> dict[str, Any]:
    state = {
        "info": json.dumps(
            {"assertions": [{"type": "t", "id": "obj"}, {"type": "t", "id": "guard"}]}
        ),
        "initial_state": {"passing": ["guard"]},
        "world": StubWorld(["obj", "guard"]),
        "partial_credit": 1.0,
        "error": None,
    }
    return {**state, **overrides}


@pytest.mark.usefixtures("stub_benchmark")
class TestOlmoEvalRecord:
    def test_records_the_classification(self):
        state = rollout_state()
        assert driver.olmo_eval_record(state) == 1.0
        record = state["_olmo_eval"]
        assert record["scoring_error"] is None
        assert (record["objectives"], record["guardrails"]) == (1, 1)
        assert record["error_kind"] is None

    def test_mismatch_with_partial_credit_is_a_scoring_error(self):
        state = rollout_state(partial_credit=0.25)
        assert driver.olmo_eval_record(state) == 0.0
        assert "imply partial_credit" in state["_olmo_eval"]["scoring_error"]
        assert TaskResult.from_record(state["_olmo_eval"] | {"task_name": "a.b"}).failure

    def test_missing_partial_credit_is_a_scoring_error(self):
        state = rollout_state()
        del state["partial_credit"]
        driver.olmo_eval_record(state)
        assert "did not run" in state["_olmo_eval"]["scoring_error"]

    def test_missing_world_is_a_scoring_error(self):
        state = rollout_state(world=None)
        driver.olmo_eval_record(state)
        assert "no world state" in state["_olmo_eval"]["scoring_error"]

    def test_empty_initial_state_makes_every_assertion_an_objective(self):
        # partial_credit evaluates every assertion normally without an initial world.
        state = rollout_state(initial_state={}, partial_credit=1.0)
        driver.olmo_eval_record(state)
        record = state["_olmo_eval"]
        assert record["scoring_error"] is None
        assert (record["objectives"], record["guardrails"]) == (2, 0)

    def test_error_kinds(self):
        state = rollout_state(error=StubInvalidResponse("empty reply"))
        driver.olmo_eval_record(state)
        assert state["_olmo_eval"]["error_kind"] == driver.MODEL_ERROR
        state = rollout_state(error=ConnectionError("server down"))
        driver.olmo_eval_record(state)
        assert state["_olmo_eval"]["error_kind"] == driver.INFRA_ERROR
        assert "server down" in state["_olmo_eval"]["error"]

    def test_rejected_history_with_a_malformed_tool_call_is_the_models_error(self):
        class BadRequestError(Exception):
            pass

        def rejected() -> Exception:
            try:
                try:
                    raise BadRequestError("Error code: 400 - Unterminated string")
                except BadRequestError as e:
                    raise RuntimeError("model call failed") from e
            except RuntimeError as e:
                return e

        bad_call = {"role": "assistant", "tool_calls": [{"id": "1", "arguments": '{"q": "x'}]}
        good_call = {"role": "assistant", "tool_calls": [{"id": "1", "arguments": '{"q": "x"}'}]}
        state = rollout_state(error=rejected(), completion=[bad_call])
        driver.olmo_eval_record(state)
        assert state["_olmo_eval"]["error_kind"] == driver.MODEL_ERROR
        # A 400 with only well-formed calls is not explained by the model.
        state = rollout_state(error=rejected(), completion=[good_call])
        driver.olmo_eval_record(state)
        assert state["_olmo_eval"]["error_kind"] == driver.INFRA_ERROR
        # A connection error is the server's, whatever the history holds.
        state = rollout_state(error=ConnectionError("down"), completion=[bad_call])
        driver.olmo_eval_record(state)
        assert state["_olmo_eval"]["error_kind"] == driver.INFRA_ERROR

    def test_malformed_tool_call_detection_reads_objects_and_dicts(self):
        call = SimpleNamespace(id="1", name="api_fetch", arguments="{not json")
        message = SimpleNamespace(role="assistant", tool_calls=[call])
        assert driver.has_malformed_tool_call([message])
        serialized = {"role": "assistant", "tool_calls": [json.dumps({"arguments": '{"a": '})]}
        assert driver.has_malformed_tool_call([serialized])
        nested = {"role": "assistant", "tool_calls": [{"function": {"arguments": "{}"}}]}
        assert not driver.has_malformed_tool_call([nested, {"role": "user", "content": "x"}])


def task(name: str, **kwargs: Any) -> TaskResult:
    return TaskResult(task_name=name, domain=name.split(".")[0], **kwargs)


class TestComputeMetrics:
    def test_primary_metric_is_first_and_averages_task_scores(self):
        tasks = [
            task("sales.a", objectives=2, objectives_met=2, aa_score=1.0),
            task("sales.b", objectives=4, objectives_met=1, aa_score=0.25),
            task(
                "hr.c",
                objectives=2,
                objectives_met=2,
                guardrails=1,
                guardrail_violations=1,
                aa_score=0.0,
            ),
        ]
        metrics = compute_metrics(tasks)
        assert next(iter(metrics)) == "aa_score"
        assert metrics["aa_score"] == pytest.approx(1.25 / 3)
        # Pooled over tasks, regardless of the violation.
        assert metrics["objectives_completed"] == pytest.approx(5 / 8)
        assert metrics["tasks_completed"] == pytest.approx(1 / 3)
        assert metrics["violations_per_task"] == pytest.approx(1 / 3)
        assert metrics["objectives_per_violation"] == pytest.approx(5.0)
        assert metrics["aa_score_sales"] == pytest.approx(0.625)
        assert metrics["aa_score_hr"] == 0.0

    def test_unscored_tasks_are_left_out_and_counted(self):
        tasks = [
            task("sales.a", objectives=1, objectives_met=1, aa_score=1.0),
            task("sales.b", error_kind="infra", error="server down"),
            task("sales.c", scoring_error="KeyError"),
        ]
        metrics = compute_metrics(tasks)
        assert metrics["aa_score"] == 1.0
        assert metrics["num_tasks"] == 3
        assert metrics["num_scored_tasks"] == 1
        assert metrics["num_failed_tasks"] == 2
        assert "objectives_per_violation" not in metrics

    def test_model_error_scores_zero_but_is_scored(self):
        # The driver records the state the agent reached, but AA scores an errored task 0.
        errored = task(
            "support.a", objectives=1, objectives_met=1, aa_score=1.0, error_kind="model"
        )
        assert errored.failure is None
        assert errored.score == 0.0
        assert not errored.completed
        metrics = compute_metrics([errored])
        assert metrics["aa_score"] == 0.0
        assert metrics["model_error_rate"] == 1.0
        assert metrics["num_scored_tasks"] == 1
        # Its objectives do not count as met in the pooled metric either.
        assert metrics["objectives_completed"] == 0.0

    def test_stop_conditions(self):
        tasks = [
            task("ops.a", stop_condition="max_turns_reached"),
            task("ops.b", stop_condition="prompt_too_long"),
            task("ops.c", stop_condition="no_tools_called"),
            task("ops.d", stop_condition="no_tools_called"),
        ]
        metrics = compute_metrics(tasks)
        assert metrics["max_turns_rate"] == 0.25
        assert metrics["context_overflow_rate"] == 0.25

    def test_no_tasks(self):
        metrics = compute_metrics([])
        assert metrics["aa_score"] == 0.0
        assert metrics["num_tasks"] == 0


class TestArgs:
    def test_defaults_match_the_aa_harness(self):
        args = AutomationBenchArgs.from_dict({})
        assert args.domains == list(DOMAINS)
        assert args.max_turns == 50
        assert args.sampling_args() == {}

    def test_parses_strings_from_the_command_line(self):
        args = AutomationBenchArgs.from_dict(
            {
                "domains": "finance,hr",
                "task_names": "hr.a, hr.b",
                "limit": "10",
                "temperature": "0.6",
                "top_p": "0.95",
                "max_tokens": "16384",
            }
        )
        assert args.domains == ["finance", "hr"]
        assert args.task_names == ["hr.a", "hr.b"]
        assert args.limit == 10
        assert args.sampling_args({"enable_thinking": True}) == {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 16384,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        }

    def test_temperature_zero_is_sent(self):
        assert AutomationBenchArgs.from_dict({"temperature": "0"}).sampling_args() == {
            "temperature": 0.0
        }

    @pytest.mark.parametrize(
        "data",
        [
            {"domains": "simple"},
            {"max_turns": "0"},
            {"run_timeout": "0"},
            {"limit": "0"},
            {"max_failure_rate": "1.5"},
        ],
    )
    def test_rejects_bad_values(self, data):
        with pytest.raises(ValueError):
            AutomationBenchArgs.from_dict(data)


class TestEnvironment:
    def test_subprocess_env_drops_what_would_redirect_uv(self, monkeypatch):
        monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/opt/venv")
        monkeypatch.setenv("UV_CONSTRAINT", "/tmp/pins.txt")
        monkeypatch.setenv("UV_CACHE_DIR", "/tmp/uv-cache")
        monkeypatch.setenv("VIRTUAL_ENV", "/opt/venv")
        monkeypatch.setenv("PYTHONPATH", "/somewhere")
        env = ab_eval._subprocess_env({"X": "1"})
        assert "UV_PROJECT_ENVIRONMENT" not in env
        assert "UV_CONSTRAINT" not in env
        assert "VIRTUAL_ENV" not in env
        assert "PYTHONPATH" not in env
        assert env["UV_CACHE_DIR"] == "/tmp/uv-cache"
        assert env["X"] == "1"

    def test_reuses_a_finished_checkout(self, tmp_path, monkeypatch):
        repo = tmp_path / ab_eval.REPO_COMMIT
        python = repo / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.touch()
        (repo / ".olmo-eval-ready").write_text(ab_eval.REPO_COMMIT)
        monkeypatch.setattr(ab_eval.shutil, "which", lambda name: f"/usr/bin/{name}")

        def no_run(*args, **kwargs):
            raise AssertionError("a finished checkout must not be reinstalled")

        monkeypatch.setattr(ab_eval.subprocess, "run", no_run)
        assert prepare_environment(tmp_path, timeout=10) == python

    def test_installs_the_pinned_commit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ab_eval.shutil, "which", lambda name: name)
        commands: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            if cmd[1] == "init":
                Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
            stdout = ab_eval.REPO_COMMIT + "\n" if "rev-parse" in cmd else ""
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

        monkeypatch.setattr(ab_eval.subprocess, "run", fake_run)
        python = prepare_environment(tmp_path, timeout=10)
        assert python == tmp_path / ab_eval.REPO_COMMIT / ".venv" / "bin" / "python"
        assert any(ab_eval.REPO_COMMIT in cmd and "fetch" in cmd for cmd in commands)
        assert commands[-1] == ["uv", "sync", "--frozen", "--no-dev", "--python", "3.13"]
        assert (tmp_path / ab_eval.REPO_COMMIT / ".olmo-eval-ready").exists()

    def test_wrong_checkout_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ab_eval.shutil, "which", lambda name: name)
        monkeypatch.setattr(
            ab_eval.subprocess,
            "run",
            lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="deadbeef\n", stderr=""),
        )
        with pytest.raises(RuntimeError, match="expected"):
            prepare_environment(tmp_path, timeout=10)

    def test_missing_tools_fail(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ab_eval.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="git and uv"):
            prepare_environment(tmp_path, timeout=10)


#: A stand-in for the driver: echoes its config and writes the records it is given.
FAKE_DRIVER = textwrap.dedent(
    """
    import json, os, sys, time
    config = json.load(open(sys.argv[1]))
    records = json.load(open(os.path.join(os.path.dirname(sys.argv[1]), "fake_records.json")))
    with open(sys.argv[2], "w") as out:
        for record in records:
            out.write(json.dumps(record) + "\\n")
    print("model=" + config["model"] + " key=" + os.environ["AUTOMATIONBENCH_API_KEY"])
    sys.stdout.flush()
    time.sleep(float(os.environ.get("FAKE_DRIVER_SLEEP", "0")))
    sys.exit(int(os.environ.get("FAKE_DRIVER_EXIT", "0")))
    """
)


def record(name: str, **kwargs: Any) -> dict[str, Any]:
    base = {
        "task_name": name,
        "domain": name.split(".")[0],
        "stop_condition": "no_tools_called",
        "error_kind": None,
        "error": None,
        "scoring_error": None,
        "objectives": 2,
        "objectives_met": 2,
        "guardrails": 1,
        "guardrail_violations": 0,
        "aa_score": 1.0,
        "partial_credit": 1.0,
        "task_completed_correctly": 1.0,
        "messages": [],
    }
    return {**base, **kwargs}


class FakeProvider:
    model_name = "org/model"
    base_url = "http://localhost:8000/v1"
    chat_template_kwargs = None


@pytest.fixture
def run_eval(tmp_path, monkeypatch):
    """Run execute() with the fake driver and the given records."""
    script = tmp_path / "fake_driver.py"
    script.write_text(FAKE_DRIVER)
    monkeypatch.setattr(ab_eval, "DRIVER", script)
    monkeypatch.setattr(ab_eval, "prepare_environment", lambda root, timeout: Path(sys.executable))

    def run(records: list[dict[str, Any]], args: dict[str, Any] | None = None):
        out = tmp_path / "out"
        run_dir = out / "automationbench_aa"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "fake_records.json").write_text(json.dumps(records))
        return asyncio.run(
            AutomationBenchAAExternalEval().execute(FakeProvider(), args or {}, str(out))
        )

    return run


class TestExecute:
    def test_scores_and_saves_a_run(self, run_eval, tmp_path):
        result = run_eval(
            [
                record("sales.a"),
                record("hr.b", guardrail_violations=1, aa_score=0.0, partial_credit=2 / 3),
            ],
            {"limit": "2", "temperature": "1.0"},
        )
        assert result.success, result.error
        assert result.metrics["aa_score"] == 0.5
        assert result.metadata["benchmark_commit"] == ab_eval.REPO_COMMIT
        assert result.metadata["sampling_args"] == {"temperature": 1.0}
        assert [p["native_id"] for p in result.predictions or []] == ["sales.a", "hr.b"]
        assert result.predictions[0]["instance_metrics"]["aa_score"] == {"external": 1.0}
        config = json.loads((tmp_path / "out/automationbench_aa/config.json").read_text())
        assert config["model"] == "org/model"
        assert config["limit"] == 2
        assert config["max_turns"] == 50
        assert (tmp_path / "out/automationbench_aa_results.json").exists()

    def test_failures_within_budget_succeed(self, run_eval):
        records = [record(f"sales.t{i}") for i in range(20)]
        records[0] = record("sales.t0", error_kind="infra", error="503")
        result = run_eval(records)
        assert result.success, result.error
        assert result.metrics["num_failed_tasks"] == 1
        assert result.metadata["failed_tasks"] == {"sales.t0": "rollout_error: 503"}
        assert result.predictions[0]["instance_metrics"] == {}

    def test_failures_over_budget_fail_the_run(self, run_eval):
        records = [record("sales.a"), record("sales.b", scoring_error="KeyError: 'x'")]
        result = run_eval(records)
        assert not result.success
        assert "1 of 2 tasks failed" in (result.error or "")
        # The partial metrics are still reported.
        assert result.metrics["aa_score"] == 1.0

    def test_driver_exit_code_fails_the_run(self, run_eval, monkeypatch):
        monkeypatch.setenv("FAKE_DRIVER_EXIT", "1")
        result = run_eval([record("sales.a")])
        assert not result.success
        assert "exited with 1" in (result.error or "")
        assert "key=EMPTY" in (result.error or "")

    def test_truncated_last_line_is_skipped(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        path.write_text(json.dumps(record("sales.a")) + "\n" + '{"task_name": "sal')
        assert [r["task_name"] for r in ab_eval.read_records(path)] == ["sales.a"]
        path.write_text('{"broken\n' + json.dumps(record("sales.a")) + "\n")
        with pytest.raises(json.JSONDecodeError):
            ab_eval.read_records(path)

    def test_run_timeout_stops_the_driver(self, run_eval, monkeypatch):
        monkeypatch.setenv("FAKE_DRIVER_SLEEP", "60")
        result = run_eval([record("sales.a")], {"run_timeout": "1"})
        assert not result.success
        assert "outlasted run_timeout" in (result.error or "")
        # The tasks written before the timeout are still scored.
        assert result.metrics["num_tasks"] == 1

    def test_prediction_score_is_zero_after_a_model_error(self, run_eval):
        result = run_eval([record("sales.a", error_kind="model")])
        prediction = (result.predictions or [])[0]
        assert prediction["aa_score"] == 0.0
        assert prediction["state_aa_score"] == 1.0
        assert prediction["instance_metrics"]["aa_score"] == {"external": 0.0}

    def test_no_tasks_fails_the_run(self, run_eval):
        result = run_eval([])
        assert not result.success
        assert "no task finished" in (result.error or "")

    def test_setup_failure(self, monkeypatch):
        def broken(root, timeout):
            raise RuntimeError("uv sync failed")

        monkeypatch.setattr(ab_eval, "prepare_environment", broken)
        result = asyncio.run(AutomationBenchAAExternalEval().execute(FakeProvider(), {}, None))
        assert not result.success
        assert "setup_error" in (result.error or "")


def test_registered():
    ext = get_external_eval("automationbench_aa")
    assert isinstance(ext, AutomationBenchAAExternalEval)
    assert ext.required_secrets == ()
    assert set(ext.arguments) >= {"domains", "task_names", "limit", "max_turns", "temperature"}
