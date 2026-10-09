"""Tests for OpenAgentSafety instance selection."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from olmo_eval.evals.external.benchmarks.openagentsafety import eval as oas_eval
from olmo_eval.evals.external.benchmarks.openagentsafety.eval import OpenAgentSafetyExternalEval
from olmo_eval.evals.external.benchmarks.openagentsafety.instances import (
    read_select,
    required_services,
    select_instances,
)

INSTANCES = [
    {"instance_id": "safety-a", "dependencies": []},
    {"instance_id": "safety-b", "dependencies": ["owncloud"]},
    {"instance_id": "safety-c", "dependencies": ["gitlab", "plane"]},
    {"instance_id": "safety-d", "dependencies": []},
]


def test_read_select_accepts_list_file_and_single_id(tmp_path: Path) -> None:
    select_file = tmp_path / "ids.txt"
    select_file.write_text("safety-a\n\nsafety-b\n")
    assert read_select(None) is None
    assert read_select(["safety-a"]) == {"safety-a"}
    assert read_select(str(select_file)) == {"safety-a", "safety-b"}
    assert read_select("safety-c") == {"safety-c"}


def test_select_instances_filters_by_select() -> None:
    chosen = select_instances(INSTANCES, {"safety-a", "safety-c"})
    assert [row["instance_id"] for row in chosen] == ["safety-a", "safety-c"]


def test_select_instances_samples_deterministically() -> None:
    first = select_instances(INSTANCES, None, 2)
    second = select_instances(INSTANCES, None, 2)
    assert len(first) == 2
    assert first == second


def test_select_instances_limit_above_size_keeps_all() -> None:
    assert select_instances(INSTANCES, None, 10) == INSTANCES


def test_required_services_ignores_unknown_dependencies() -> None:
    rows = INSTANCES + [{"instance_id": "safety-e", "dependencies": ["something-else"]}]
    assert required_services(rows) == {"owncloud", "gitlab", "plane"}
    assert required_services(INSTANCES[:1]) == set()


def _stub_execute(monkeypatch: pytest.MonkeyPatch, instances: list[dict]) -> list[str]:
    """Stub everything before repo setup, which fails to end the run early."""
    calls: list[str] = []
    monkeypatch.setenv("NPC_API_KEY", "test")
    monkeypatch.setattr(oas_eval, "load_instances", lambda dataset, split: instances)
    monkeypatch.setattr(OpenAgentSafetyExternalEval, "_docker_error", lambda self: None)
    monkeypatch.setattr(oas_eval, "ensure_tac_services", lambda: calls.append("tac"))

    def stop(*args, **kwargs):
        calls.append("repo")
        raise RuntimeError("stop")

    monkeypatch.setattr(oas_eval, "ensure_repo", stop)
    return calls


def test_execute_skips_tac_when_no_services_needed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub_execute(monkeypatch, INSTANCES)
    provider = SimpleNamespace(model_name="test-model", base_url=None)
    asyncio.run(
        OpenAgentSafetyExternalEval().execute(
            provider,  # type: ignore[arg-type]
            {"select": "safety-a,safety-d"},
            output_dir=str(tmp_path),
        )
    )
    assert calls == ["repo"]


def test_execute_starts_tac_when_services_needed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub_execute(monkeypatch, INSTANCES)
    provider = SimpleNamespace(model_name="test-model", base_url=None)
    asyncio.run(
        OpenAgentSafetyExternalEval().execute(
            provider,  # type: ignore[arg-type]
            {"select": "safety-a,safety-b"},
            output_dir=str(tmp_path),
        )
    )
    assert calls == ["tac", "repo"]


def test_execute_errors_when_nothing_selected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub_execute(monkeypatch, INSTANCES)
    provider = SimpleNamespace(model_name="test-model", base_url=None)
    result = asyncio.run(
        OpenAgentSafetyExternalEval().execute(
            provider,  # type: ignore[arg-type]
            {"select": "safety-missing"},
            output_dir=str(tmp_path),
        )
    )
    assert result.success is False
    assert result.error is not None
    assert "No OpenAgentSafety instances" in result.error
    assert calls == []
