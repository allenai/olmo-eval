"""Pin the task hash of every registered spec.

A task hash identifies stored results, so a change that moves one makes earlier
results incomparable. The hashes are of the config the runner stores, built by
``configure_task``. This test fails on any hash that differs from
``task_hashes.json``. When a change is meant to move hashes, regenerate the file
with ``uv run pytest tests/evals/tasks/test_task_hashes.py --update-task-hashes``,
check that only the intended specs changed, and add an entry to
docs/result-compatibility.md.

``with_judge_env`` holds the specs whose hash differs when ``OLMO_EVAL_JUDGE`` is
set, so changes to how the environment override reaches the hash show up too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import olmo_eval.evals.tasks  # noqa: F401  (registers every task)
import olmo_eval.evals.vision.benchmarks  # noqa: F401  (registers the vision tasks)
from olmo_eval.common.types.base import compute_task_hash
from olmo_eval.evals.tasks.common.registry import _tasks, _variants
from olmo_eval.runners.asynq.preparation import configure_task

HASHES_FILE = Path(__file__).with_name("task_hashes.json")

#: Judge spec set for the environment-override hashes.
JUDGE_ENV_SPEC = "pinned-judge-model:high:12345"


def _specs() -> list[str]:
    """Every registered task and each of its single variants."""
    specs: list[str] = []
    for name in sorted(_tasks):
        specs.append(name)
        specs.extend(f"{name}:{variant}" for variant in sorted(_variants.get(name, {})))
    return specs


def _hashes(monkeypatch: pytest.MonkeyPatch, judge_env: str | None) -> dict[str, str | None]:
    if judge_env is None:
        monkeypatch.delenv("OLMO_EVAL_JUDGE", raising=False)
    else:
        monkeypatch.setenv("OLMO_EVAL_JUDGE", judge_env)
    return {spec: compute_task_hash(configure_task(spec).config.to_dict()) for spec in _specs()}


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
    for section in ("default", "with_judge_env"):
        changed = sorted(
            spec
            for spec in expected[section].keys() | current[section].keys()
            if expected[section].get(spec) != current[section].get(spec)
        )
        assert not changed, (
            f"{len(changed)} {section} task hash(es) changed, e.g. {changed[:10]}. If this is "
            "intended, rerun with --update-task-hashes and record it in "
            "docs/result-compatibility.md."
        )
