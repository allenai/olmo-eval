"""Run AutomationBench tasks and score them the way AutomationBench-AA does.

This script runs inside AutomationBench's own environment (a checkout of the
pinned release, synced from its lockfile), not inside olmo-eval's:

    python -I run_tasks.py CONFIG_JSON OUTPUT_JSONL

It drives AutomationBench's verifiers environment with the API toolset, as the
AutomationBench CLI does, and writes one JSON line per task as each finishes.
Each line carries AutomationBench's own scores (``partial_credit`` and
``task_completed_correctly``) and the objective and guardrail counts that
Artificial Analysis scores by:

- an assertion that fails in the task's initial state is an objective, which
  the agent must make true;
- an assertion that already passes in the initial state is a guardrail, which
  the agent must not break.

Assertions a task marks as unscored are left out, and an assertion marked
``"excluded": false`` is an objective even when it passes initially, as in
AutomationBench's ``partial_credit``. The script checks that its counts give back
AutomationBench's ``partial_credit`` for every task, and records a scoring error
for a task where they do not.

This file must not import olmo_eval: only AutomationBench and its dependencies
are installed where it runs.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from typing import Any

#: Model-side errors: the model's reply could not be used. AutomationBench-AA
#: scores a task that ends this way as 0.
MODEL_ERROR = "model"
#: Everything else that stops a rollout with an error: the model server, a tool,
#: or the environment failed. olmo-eval leaves these tasks unscored.
INFRA_ERROR = "infra"

#: Tolerance when checking our counts against AutomationBench's partial_credit.
PARTIAL_CREDIT_TOLERANCE = 1e-9


def classify_assertions(
    assertions: Iterable[dict[str, Any]],
    passes_initially: Callable[[dict[str, Any]], bool],
    passes_finally: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    """Sort a task's assertions into objectives and guardrails and count outcomes.

    Mirrors the branches of AutomationBench's ``rubric.partial_credit``.

    Args:
        assertions: The task's assertion specs.
        passes_initially: Checks an assertion against the initial world.
        passes_finally: Checks an assertion against the world the agent left.

    Returns:
        Counts of objectives met, guardrails broken and excluded assertions, the
        AutomationBench-AA task score, the ``partial_credit`` those counts imply,
        and one entry per assertion.
    """
    objectives = objectives_met = guardrails = violations = excluded = 0
    results: list[dict[str, Any]] = []
    for assertion in assertions:
        final = bool(passes_finally(assertion))
        entry: dict[str, Any] = {"type": assertion.get("type"), "passed": final}
        if assertion.get("scored") is False or assertion.get("excluded") is True:
            excluded += 1
            entry["class"] = "excluded"
        elif bool(passes_initially(assertion)) and assertion.get("excluded") is not False:
            guardrails += 1
            violations += int(not final)
            entry["class"] = "guardrail"
        else:
            objectives += 1
            objectives_met += int(final)
            entry["class"] = "objective"
        entry["params"] = {k: v for k, v in assertion.items() if k != "type"}
        results.append(entry)

    # partial_credit counts a broken guardrail as a failed assertion and leaves an
    # intact one out; with nothing to score it is 0.
    denominator = objectives + violations
    partial_credit = objectives_met / denominator if denominator else 0.0
    # A broken guardrail zeroes the task; otherwise it scores the objectives met.
    met_share = objectives_met / objectives if objectives else 0.0
    aa_score = 0.0 if violations else met_share
    return {
        "objectives": objectives,
        "objectives_met": objectives_met,
        "guardrails": guardrails,
        "guardrail_violations": violations,
        "excluded": excluded,
        "aa_score": aa_score,
        "implied_partial_credit": partial_credit,
        "assertions": results,
    }


def error_kind(error: BaseException | None) -> str | None:
    """Whether a rollout error is the model's (scored) or the harness's (unscored)."""
    if error is None:
        return None
    import verifiers as vf

    if isinstance(error, vf.InvalidModelResponseError):
        return MODEL_ERROR
    return INFRA_ERROR


def describe_error(error: Any) -> str | None:
    """A readable error message, from an exception or an already serialized error."""
    if error is None:
        return None
    if isinstance(error, BaseException):
        chain = []
        current: BaseException | None = error
        while current is not None and len(chain) < 5:
            chain.append(f"{type(current).__name__}: {current}")
            current = current.__cause__ or current.__context__
        return " <- ".join(chain)
    return str(error)


def olmo_eval_record(state: Any, **kwargs: Any) -> float:
    """Rubric function that records AA scoring in ``state["_olmo_eval"]``.

    It runs after AutomationBench's ``partial_credit`` in the same rubric, so the
    world it checks is the one the agent left. verifiers turns an exception in a
    reward function into a score of 0, so this function catches its own errors
    and records them instead.
    """
    record: dict[str, Any] = {
        "error_kind": None,
        "error": None,
        "scoring_error": None,
    }
    state["_olmo_eval"] = record
    try:
        error = state.get("error")
        record["error_kind"] = error_kind(error)
        record["error"] = describe_error(error)

        from automationbench.rubric.registry import AssertionRegistry
        from automationbench.schema.world import WorldState

        info = state.get("info") or {}
        if isinstance(info, str):
            info = json.loads(info)
        world = state.get("world")
        initial = state.get("initial_state")
        if world is None or initial is None:
            raise RuntimeError("the rollout has no world state to score")
        if "partial_credit" not in state:
            raise RuntimeError("AutomationBench's partial_credit did not run to completion")
        initial_world = WorldState(**initial)
        aa = classify_assertions(
            info.get("assertions", []),
            lambda a: AssertionRegistry.check(initial_world, a),
            lambda a: AssertionRegistry.check(world, a),
        )
        expected = float(state["partial_credit"])
        if abs(aa["implied_partial_credit"] - expected) > PARTIAL_CREDIT_TOLERANCE:
            raise RuntimeError(
                f"objective and guardrail counts imply partial_credit "
                f"{aa['implied_partial_credit']}, AutomationBench computed {expected}"
            )
        record.update(aa)
        return float(aa["aa_score"])
    except Exception as e:
        record["scoring_error"] = "".join(traceback.format_exception_only(e)).strip()
        return 0.0


def _serialize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_serialize(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def task_record(output: dict[str, Any]) -> dict[str, Any]:
    """One output line for a finished rollout."""
    info = output.get("info") or {}
    if isinstance(info, str):
        info = json.loads(info)
    name = info.get("task_name") or output.get("task") or "unknown"
    prompt = output.get("prompt") or []
    completion = output.get("completion") or []
    messages = _serialize(list(prompt) + list(completion))
    record = output.get("_olmo_eval") or {
        "error_kind": INFRA_ERROR,
        "error": describe_error(output.get("error")),
        "scoring_error": "the rollout was never scored",
    }
    if output.get("error") is not None and record.get("error") is None:
        record = {**record, "error": describe_error(output.get("error"))}
        record.setdefault("error_kind", INFRA_ERROR)
    metrics = output.get("metrics") or {}
    perf = output.get("_perf") or {}
    return {
        "task_name": name,
        "domain": name.split(".", 1)[0],
        "example_id": output.get("example_id"),
        "task_contract_sha256": output.get("_task_contract_sha256"),
        "stop_condition": output.get("stop_condition"),
        "is_truncated": bool(output.get("is_truncated")),
        "partial_credit": float(metrics.get("partial_credit", output.get("reward", 0.0))),
        "task_completed_correctly": float(metrics.get("task_completed_correctly", 0.0)),
        "turns": sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant"),
        "tool_calls": int(perf.get("tool_calls", 0) or 0),
        "usage": _serialize(output.get("_usage")),
        "debug": _serialize(output.get("_debug")),
        **_serialize(record),
        "messages": messages,
    }


async def run(config: dict[str, Any], output_path: str) -> int:
    import verifiers as vf
    from automationbench.clients import RetryingOpenAIChatCompletionsClient
    from automationbench.domains import get_combined_dataset
    from automationbench.rubric import partial_credit, task_completed_correctly
    from automationbench.runner import AutomationBenchEnv
    from verifiers.types import ClientConfig

    dataset = get_combined_dataset(list(config["domains"]))
    wanted = config.get("task_names")
    if wanted:
        names = [
            (json.loads(r["info"]) if isinstance(r["info"], str) else r["info"]).get("task_name")
            for r in dataset
        ]
        missing = sorted(set(wanted) - set(names))
        if missing:
            raise SystemExit(f"Unknown task names: {', '.join(missing)}")
        dataset = dataset.select([i for i, n in enumerate(names) if n in set(wanted)])
    limit = config.get("limit")
    if limit is not None:
        dataset = dataset.select(range(min(int(limit), len(dataset))))
    total = len(dataset)
    print(f"[automationbench] running {total} tasks", flush=True)

    # AutomationBench's rubric, with the AA record added after partial_credit.
    rubric = vf.Rubric(
        funcs=[partial_credit, task_completed_correctly, olmo_eval_record],
        weights=[1.0, 0.0, 0.0],
    )
    env = AutomationBenchEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=int(config["max_turns"]),
        toolset="api",
    )
    client = RetryingOpenAIChatCompletionsClient(
        ClientConfig(api_key_var=config["api_key_var"], api_base_url=config["base_url"])
    )

    done = 0
    started = time.time()
    with open(output_path, "w") as out:

        def on_progress(all_outputs: Any, new_outputs: Any, new_metadata: Any) -> None:
            nonlocal done
            for output in new_outputs or []:
                out.write(json.dumps(task_record(output)) + "\n")
                done += 1
            out.flush()
            print(
                f"[automationbench] {done}/{total} tasks done "
                f"({time.time() - started:.0f}s elapsed)",
                flush=True,
            )

        await env.evaluate(
            client=client,
            model=config["model"],
            sampling_args=config.get("sampling_args") or None,
            num_examples=-1,
            rollouts_per_example=1,
            max_concurrent=int(config["max_concurrent"]),
            state_columns=["_usage", "_debug", "_perf", "_olmo_eval", "_task_contract_sha256"],
            on_progress=on_progress,
        )
    if done != total:
        print(f"[automationbench] wrote {done} of {total} tasks", file=sys.stderr, flush=True)
        return 1
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[1]) as f:
        config = json.load(f)
    return asyncio.run(run(config, argv[2]))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
