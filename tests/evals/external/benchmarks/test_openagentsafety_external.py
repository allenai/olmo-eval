"""Tests for the OpenAgentSafety external evaluation wrapper."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from olmo_eval.evals.external.benchmarks.openagentsafety.args import (
    DEFAULT_DATASET,
    OpenAgentSafetyArgs,
)
from olmo_eval.evals.external.benchmarks.openagentsafety.eval import OpenAgentSafetyExternalEval
from olmo_eval.evals.external.benchmarks.openagentsafety.result_parser import (
    find_output_jsonl,
    parse_output_jsonl,
    parse_output_records,
)
from olmo_eval.evals.external.registry import get_external_eval, list_external_evals


def test_openagentsafety_is_registered() -> None:
    assert "openagentsafety" in list_external_evals()
    evaluation = get_external_eval("openagentsafety")
    assert evaluation.name == "openagentsafety"
    assert "dataset" in evaluation.arguments
    assert "n_limit" in evaluation.arguments
    assert "critic" in evaluation.arguments
    assert evaluation.required_secrets == ("NPC_API_KEY",)
    assert "openagentsafety" in evaluation.run_command


def test_args_defaults() -> None:
    args = OpenAgentSafetyArgs.from_dict({})
    assert args.dataset == DEFAULT_DATASET
    assert args.split == "train"
    assert args.n_limit is None
    assert args.num_workers == 1
    assert args.critic == "pass"
    assert args.select is None


def test_args_parses_limit_and_select_ids() -> None:
    args = OpenAgentSafetyArgs.from_dict(
        {
            "n_limit": "2",
            "num_workers": "4",
            "select": "safety-audit,safety-compliance",
            "critic": "finish_with_patch",
            "enable_delegation": "true",
        }
    )
    assert args.n_limit == 2
    assert args.num_workers == 4
    assert args.select == ["safety-audit", "safety-compliance"]
    assert args.critic == "finish_with_patch"
    assert args.enable_delegation is True


def test_args_keeps_select_file_path(tmp_path: Path) -> None:
    select_file = tmp_path / "instances.txt"
    select_file.write_text("safety-audit\n")
    args = OpenAgentSafetyArgs.from_dict({"select": str(select_file)})
    assert args.select == str(select_file)


def test_args_rejects_unknown_critic() -> None:
    with pytest.raises(ValueError, match="Unknown critic"):
        OpenAgentSafetyArgs.from_dict({"critic": "not-a-critic"})


def test_parse_output_records_matches_eval_infer_resolution() -> None:
    parsed = parse_output_records(
        [
            {
                "instance_id": "safety-resolved",
                "test_result": {"final_score": {"result": 2, "total": 2}},
            },
            {
                "instance_id": "safety-unresolved",
                "test_result": {"final_score": {"result": 1, "total": 2}},
            },
            {
                "instance_id": "safety-zero",
                "test_result": {"final_score": {"result": 0, "total": 1}},
            },
            {
                "instance_id": "safety-error",
                "error": "boom",
                "test_result": {"error": "evaluator crashed"},
            },
        ]
    )
    assert parsed["metrics"]["num_instances"] == 4.0
    assert parsed["metrics"]["num_completed"] == 3.0
    assert parsed["metrics"]["num_resolved"] == 1.0
    assert parsed["metrics"]["num_unresolved"] == 2.0
    assert parsed["metrics"]["num_errors"] == 1.0
    assert parsed["metrics"]["resolve_rate"] == pytest.approx(1 / 3)
    assert parsed["metrics"]["checkpoint_score"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert parsed["metadata"]["resolved_ids"] == ["safety-resolved"]
    by_id = {item["native_id"]: item for item in parsed["predictions"]}
    assert by_id["safety-resolved"]["instance_metrics"]["resolved"]["external"] == 1.0
    assert by_id["safety-unresolved"]["instance_metrics"]["checkpoint_score"]["external"] == 0.5


def test_parse_output_jsonl_and_find(tmp_path: Path) -> None:
    nested = tmp_path / "run" / "nested"
    nested.mkdir(parents=True)
    jsonl = nested / "output.jsonl"
    jsonl.write_text(
        '{"instance_id": "safety-audit", '
        '"test_result": {"final_score": {"result": 1, "total": 1}}}\n'
    )
    found = find_output_jsonl(tmp_path)
    assert found == jsonl
    parsed = parse_output_jsonl(jsonl)
    assert parsed["metrics"]["num_resolved"] == 1.0
    assert parsed["metrics"]["resolve_rate"] == 1.0


def test_parse_empty_records() -> None:
    parsed = parse_output_records([])
    assert parsed["metrics"]["resolve_rate"] == 0.0
    assert parsed["metrics"]["num_instances"] == 0.0
    assert parsed["success"] is False


def test_build_infer_command_includes_core_flags(tmp_path: Path) -> None:
    evaluation = OpenAgentSafetyExternalEval()
    args = OpenAgentSafetyArgs.from_dict(
        {"n_limit": 1, "select": ["safety-audit"], "critic": "pass"}
    )
    select_path = evaluation._materialize_select(args.select, tmp_path)
    command = evaluation._build_infer_command(
        repo_dir=tmp_path,
        llm_config_path=tmp_path / "llm.json",
        oas_args=args,
        output_dir=tmp_path,
        select_path=select_path,
    )
    assert command[:3] == ["uv", "run", "openagentsafety-infer"]
    assert "--n-limit" in command
    assert command[command.index("--n-limit") + 1] == "1"
    assert "--critic" in command
    assert command[command.index("--critic") + 1] == "pass"
    assert "--select" in command
    assert Path(command[command.index("--select") + 1]).read_text() == "safety-audit\n"
    assert "--workspace" in command
    assert command[command.index("--workspace") + 1] == "docker"


def test_build_llm_config_prefixes_local_vllm() -> None:
    evaluation = OpenAgentSafetyExternalEval()
    provider = SimpleNamespace(
        model_name="olmo-2",
        base_url="http://localhost:8000/v1",
        _server=object(),
        get_openai_client=lambda: SimpleNamespace(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        ),
    )
    config = evaluation._build_llm_config(provider)  # type: ignore[arg-type]
    assert config["model"] == "hosted_vllm/olmo-2"
    assert config["base_url"] == "http://localhost:8000/v1"
    assert config["api_key"] == "EMPTY"


def test_build_llm_config_keeps_remote_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluation = OpenAgentSafetyExternalEval()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = SimpleNamespace(
        model_name="gpt-4o-mini",
        base_url=None,
        get_openai_client=lambda: None,
    )
    config = evaluation._build_llm_config(provider)  # type: ignore[arg-type]
    assert config["model"] == "gpt-4o-mini"
    assert config["api_key"] == "sk-test"
    assert "base_url" not in config


def test_execute_requires_npc_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPC_API_KEY", raising=False)
    evaluation = OpenAgentSafetyExternalEval()
    provider = SimpleNamespace(model_name="test-model", base_url=None)
    result = asyncio.run(evaluation.execute(provider, {}))  # type: ignore[arg-type]
    assert result.success is False
    assert result.error is not None
    assert "NPC_API_KEY" in result.error


def test_prompt_override_restores_default(tmp_path: Path) -> None:
    evaluation = OpenAgentSafetyExternalEval()
    prompts = tmp_path / "benchmarks" / "openagentsafety" / "prompts"
    prompts.mkdir(parents=True)
    default = prompts / "default.j2"
    default.write_text("ORIGINAL")
    custom = tmp_path / "custom.j2"
    custom.write_text("CUSTOM")
    restore = evaluation._apply_prompt_override(tmp_path, str(custom))
    assert restore is not None
    assert default.read_text() == "CUSTOM"
    path, original = restore
    path.write_text(original)
    assert default.read_text() == "ORIGINAL"
