"""Pin the task hash of every registered spec.

A task hash identifies stored results, so a change that moves one makes earlier
results incomparable. The hashes are of the config the runner stores, built by
``configure_task``, for every task, every single variant, and every spec a suite
expands to. This test fails when a spec's hash differs from ``task_hashes.json``, or
when a spec is added or removed. Regenerate the file with
``uv run pytest tests/evals/tasks/test_task_hashes.py --update-task-hashes``; AGENTS.md
says when a moved or removed hash also needs an entry in docs/result-compatibility.md.

``with_judge_env`` holds the specs whose hash differs when ``OLMO_EVAL_JUDGE`` is
set, so changes to how the environment override reaches the hash show up too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import olmo_eval.evals.tasks  # noqa: F401  (registers every task)
import olmo_eval.evals.vision.benchmarks  # noqa: F401  (registers the vision tasks)
from olmo_eval.common.configs import expand_tasks
from olmo_eval.common.types.base import compute_task_hash
from olmo_eval.evals.suites import list_suites
from olmo_eval.evals.tasks.common import list_tasks, list_variants
from olmo_eval.evals.tasks.ctc_suite import DATA_ROOT_ENV
from olmo_eval.runners.asynq.preparation import configure_task

HASHES_FILE = Path(__file__).with_name("task_hashes.json")

#: Judge spec set for the environment-override hashes.
JUDGE_ENV_SPEC = "pinned-judge-model:high:12345"

#: Environment variables a task reads into its config when it is built. They are
#: cleared so a developer's local settings cannot reach the pinned hashes.
CONFIG_ENV_VARS = ("OLMO_EVAL_JUDGE", DATA_ROOT_ENV)


def _specs() -> list[str]:
    """Every registered task, each of its single variants, and every suite's specs."""
    variants = list_variants()
    specs = set()
    for name in list_tasks():
        specs.add(name)
        specs.update(f"{name}:{variant}" for variant in variants.get(name, ()))
    for suite in list_suites():
        specs.update(expand_tasks([suite]))
    return sorted(specs)


def _hashes(monkeypatch: pytest.MonkeyPatch, judge_env: str | None) -> dict[str, str | None]:
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    if judge_env is not None:
        monkeypatch.setenv("OLMO_EVAL_JUDGE", judge_env)
    return {spec: compute_task_hash(configure_task(spec).config.to_dict()) for spec in _specs()}


def _differences(expected: dict[str, str | None], current: dict[str, str | None]) -> str:
    """Describe the specs added, removed, and changed between two hash maps."""
    added = sorted(current.keys() - expected.keys())
    removed = sorted(expected.keys() - current.keys())
    changed = sorted(
        spec for spec in expected.keys() & current.keys() if expected[spec] != current[spec]
    )
    lines = []
    if added:
        lines.append(f"{len(added)} added: {', '.join(added[:10])}")
    if removed:
        lines.append(f"{len(removed)} removed: {', '.join(removed[:10])}")
    if changed:
        shown = ", ".join(f"{spec} {expected[spec]} -> {current[spec]}" for spec in changed[:10])
        lines.append(f"{len(changed)} changed: {shown}")
    return "; ".join(lines)


def test_task_hashes_are_unchanged(request: pytest.FixtureRequest, monkeypatch) -> None:
    default = _hashes(monkeypatch, None)
    with_env = _hashes(monkeypatch, JUDGE_ENV_SPEC)
    current = {
        "default": default,
        "with_judge_env": {
            spec: value for spec, value in with_env.items() if value != default[spec]
        },
    }

    if request.config.getoption("--update-task-hashes"):
        HASHES_FILE.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        return

    expected = json.loads(HASHES_FILE.read_text())
    problems = {
        section: difference
        for section in ("default", "with_judge_env")
        if (difference := _differences(expected[section], current[section]))
    }
    assert not problems, (
        f"Task hashes differ from {HASHES_FILE.name}: {problems}. Regenerate it with "
        "--update-task-hashes. Added specs need nothing more; for a changed or removed "
        "hash, follow the 'Pinned hashes' rule in AGENTS.md."
    )
