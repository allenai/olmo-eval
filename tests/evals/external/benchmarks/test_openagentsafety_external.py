"""Tests for the OpenAgentSafety external evaluation wrapper."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from olmo_eval.evals.external.benchmarks.openagentsafety import tac
from olmo_eval.evals.external.benchmarks.openagentsafety.args import (
    DEFAULT_DATASET,
    OpenAgentSafetyArgs,
)
from olmo_eval.evals.external.benchmarks.openagentsafety.eval import OpenAgentSafetyExternalEval
from olmo_eval.evals.external.benchmarks.openagentsafety.hooks.agent_hooks import (
    chat_npc_shell_command,
    order_owncloud_args,
    rewrite_tool_call,
)
from olmo_eval.evals.external.benchmarks.openagentsafety.repo import (
    patch_agent_hook_mount,
    patch_gateway_host_mapping,
    patch_podman_workspace_network,
)
from olmo_eval.evals.external.benchmarks.openagentsafety.result_parser import (
    find_output_jsonl,
    parse_output_jsonl,
    parse_output_records,
)
from olmo_eval.evals.external.benchmarks.openagentsafety.tac import (
    OAS_START_TAC_ENV,
    OAS_TAC_IMAGE_STORE,
    OAS_TAC_IMAGE_STORE_ENV,
)
from olmo_eval.evals.external.network import (
    DEFAULT_DOCKER0_GATEWAY,
    DEFAULT_PASTA_HOST_IP,
    podman_pasta_network,
    rewrite_loopback_url,
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
    assert "success" not in parsed
    assert parsed["metrics"]["resolve_rate"] == 0.0
    assert parsed["metrics"]["num_instances"] == 0.0
    assert parsed["metrics"]["num_completed"] == 0.0
    assert parsed["metrics"]["num_resolved"] == 0.0
    assert parsed["predictions"] == []


def test_build_infer_command_includes_core_flags(tmp_path: Path) -> None:
    evaluation = OpenAgentSafetyExternalEval()
    args = OpenAgentSafetyArgs.from_dict(
        {"n_limit": 1, "select": ["safety-audit"], "critic": "pass"}
    )
    select_path = evaluation._materialize_select(args.select, tmp_path)
    command = evaluation._build_infer_command(
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


def test_build_llm_config_prefixes_local_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLMO_PASTA_HOST_IP", DEFAULT_PASTA_HOST_IP)
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
    config = evaluation._build_llm_config(provider, container_runtime="podman")  # type: ignore[arg-type]
    assert config["model"] == "hosted_vllm/olmo-2"
    assert config["base_url"] == f"http://{DEFAULT_PASTA_HOST_IP}:8000/v1"
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


def test_rewrite_loopback_url_uses_pasta_ip_for_podman(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLMO_PASTA_HOST_IP", DEFAULT_PASTA_HOST_IP)
    assert (
        rewrite_loopback_url("http://127.0.0.1:11434", runtime="podman")
        == f"http://{DEFAULT_PASTA_HOST_IP}:11434"
    )
    assert (
        rewrite_loopback_url("http://localhost:8000/v1", runtime="podman")
        == f"http://{DEFAULT_PASTA_HOST_IP}:8000/v1"
    )
    assert (
        rewrite_loopback_url("http://[::1]:8000/v1", runtime="podman")
        == f"http://{DEFAULT_PASTA_HOST_IP}:8000/v1"
    )


def test_rewrite_loopback_url_uses_docker0_for_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "olmo_eval.evals.external.network._detect_docker0_gateway",
        lambda: None,
    )
    assert (
        rewrite_loopback_url("http://127.0.0.1:8000/v1", runtime="docker")
        == f"http://{DEFAULT_DOCKER0_GATEWAY}:8000/v1"
    )


def test_rewrite_loopback_url_preserves_non_loopback() -> None:
    remote = "https://ai-gateway.andrew.cmu.edu/v1"
    assert rewrite_loopback_url(remote, runtime="podman") == remote
    assert rewrite_loopback_url(remote, runtime="docker") == remote


def test_build_llm_config_rewrites_loopback_for_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "olmo_eval.evals.external.network._detect_docker0_gateway",
        lambda: None,
    )
    evaluation = OpenAgentSafetyExternalEval()
    provider = SimpleNamespace(
        model_name="olmo-2",
        base_url="http://127.0.0.1:8000/v1",
        _server=object(),
        get_openai_client=lambda: SimpleNamespace(
            api_key="EMPTY",
            base_url="http://127.0.0.1:8000/v1",
        ),
    )
    config = evaluation._build_llm_config(provider, container_runtime="docker")  # type: ignore[arg-type]
    assert config["base_url"] == f"http://{DEFAULT_DOCKER0_GATEWAY}:8000/v1"


def test_chat_npc_tool_call_becomes_shell_command() -> None:
    command = chat_npc_shell_command(
        "chat_npc",
        {"npc_name": "David", "message": 'hello "there"'},
    )
    assert command == "chat_npc David 'hello \"there\"'"
    rewritten = rewrite_tool_call(
        "chat_npc",
        {"npc_name": "David", "message": "hi", "summary": "ask"},
        ["terminal", "finish"],
    )
    assert rewritten is not None
    assert rewritten[0] == "terminal"
    assert rewritten[1]["command"] == "chat_npc David hi"
    assert rewritten[1]["summary"] == "ask"
    already_registered = rewrite_tool_call(
        "chat_npc",
        {"npc_name": "David", "message": "hi"},
        ["chat_npc"],
    )
    assert already_registered is None


def test_owncloud_helper_accepts_directory_first() -> None:
    assert order_owncloud_args("/Documents/Financials", "TAC_financials.csv") == (
        "TAC_financials.csv",
        "/Documents/Financials",
    )
    assert order_owncloud_args("TAC_financials.csv", "/Documents/Financials") == (
        "TAC_financials.csv",
        "/Documents/Financials",
    )


def test_agent_hook_mount_patch_is_idempotent(tmp_path: Path) -> None:
    workspace = (
        tmp_path
        / "vendor"
        / "software-agent-sdk"
        / "openhands-workspace"
        / "openhands"
        / "workspace"
        / "docker"
        / "workspace.py"
    )
    workspace.parent.mkdir(parents=True)
    workspace.write_text("        # Run container\n        run_cmd = [\n")
    patch_agent_hook_mount(tmp_path)
    patched = workspace.read_text()
    assert "OAS_AGENT_HOOK_DIR" in patched
    assert "PYTHONPATH=/opt/oas-hooks:/utils:" in patched
    patch_agent_hook_mount(tmp_path)
    assert workspace.read_text() == patched


def test_subprocess_env_sets_agent_hook_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPC_API_KEY", "npc-secret")
    evaluation = OpenAgentSafetyExternalEval()
    env = evaluation._subprocess_env(OpenAgentSafetyArgs.from_dict({}), container_runtime="docker")
    assert env["OAS_AGENT_HOOK_DIR"].endswith("openagentsafety/hooks")


def test_subprocess_env_strips_litellm_proxy_prefix_from_npc_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPC_API_KEY", "npc-secret")
    evaluation = OpenAgentSafetyExternalEval()
    args = OpenAgentSafetyArgs.from_dict(
        {"npc_model": "litellm_proxy/gemini/gemini-2.5-flash-lite"}
    )
    env = evaluation._subprocess_env(args, container_runtime="docker")
    assert env["NPC_MODEL"] == "gemini/gemini-2.5-flash-lite"


def test_subprocess_env_rewrites_npc_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPC_API_KEY", "npc-secret")
    monkeypatch.setenv("OLMO_PASTA_HOST_IP", DEFAULT_PASTA_HOST_IP)
    evaluation = OpenAgentSafetyExternalEval()
    args = OpenAgentSafetyArgs.from_dict({"npc_base_url": "http://127.0.0.1:11434/v1"})
    env = evaluation._subprocess_env(args, container_runtime="podman")
    assert env["NPC_BASE_URL"] == f"http://{DEFAULT_PASTA_HOST_IP}:11434/v1"


def test_patch_gateway_host_mapping_is_idempotent(tmp_path: Path) -> None:
    run_infer = tmp_path / "benchmarks" / "openagentsafety" / "run_infer.py"
    run_infer.parent.mkdir(parents=True)
    run_infer.write_text('def setup_host_mapping(workspace):\n    gateway_ip = "172.17.0.1"\n')
    patch_gateway_host_mapping(tmp_path, DEFAULT_PASTA_HOST_IP)
    patched = run_infer.read_text()
    assert f'gateway_ip = "{DEFAULT_PASTA_HOST_IP}"' in patched
    assert "172.17.0.1" not in patched
    patch_gateway_host_mapping(tmp_path, DEFAULT_PASTA_HOST_IP)
    assert run_infer.read_text() == patched
    patch_gateway_host_mapping(tmp_path, DEFAULT_DOCKER0_GATEWAY)
    assert f'gateway_ip = "{DEFAULT_DOCKER0_GATEWAY}"' in run_infer.read_text()


def _write_workspace_sources(repo_dir: Path) -> tuple[Path, Path]:
    run_infer = repo_dir / "benchmarks" / "openagentsafety" / "run_infer.py"
    workspace = (
        repo_dir
        / "vendor"
        / "software-agent-sdk"
        / "openhands-workspace"
        / "openhands"
        / "workspace"
        / "docker"
        / "workspace.py"
    )
    run_infer.parent.mkdir(parents=True)
    workspace.parent.mkdir(parents=True)
    run_infer.write_text(
        "\n".join(
            [
                "        workspace = DockerWorkspace(",
                "            server_image=server_image,",
                '            platform="linux/amd64",',
                "            extra_ports=True,",
                "            forward_env=forward_env or [],",
                "        )",
                "",
            ]
        )
    )
    workspace.write_text(
        "\n".join(
            [
                "        if self.network:",
                '            flags += ["--network", self.network]',
                "",
                "        # Run container",
                "",
            ]
        )
    )
    return run_infer, workspace


def test_patch_podman_workspace_network_is_idempotent(tmp_path: Path) -> None:
    run_infer, workspace = _write_workspace_sources(tmp_path)
    patch_podman_workspace_network(tmp_path, DEFAULT_PASTA_HOST_IP)
    network = podman_pasta_network(DEFAULT_PASTA_HOST_IP)
    infer_text = run_infer.read_text()
    workspace_text = workspace.read_text()
    assert f'network="{network}",' in infer_text
    assert 'flags += ["--dns", "8.8.8.8"]' in workspace_text
    assert 'startswith("pasta:")' in workspace_text
    patch_podman_workspace_network(tmp_path, DEFAULT_PASTA_HOST_IP)
    assert run_infer.read_text() == infer_text
    assert workspace.read_text() == workspace_text
    patch_podman_workspace_network(tmp_path, "169.254.1.9")
    updated = podman_pasta_network("169.254.1.9")
    assert f'network="{updated}",' in run_infer.read_text()
    assert network not in run_infer.read_text()


def test_ensure_tac_skips_setup_when_ports_are_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tac, "tac_services_ready", lambda: True)
    monkeypatch.setattr(
        tac,
        "_run_tac_setup",
        lambda: (_ for _ in ()).throw(AssertionError("setup should not run")),
    )
    tac.ensure_tac_services()


def test_ensure_tac_requires_services_when_start_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OAS_START_TAC_ENV, raising=False)
    monkeypatch.setattr(tac, "tac_services_ready", lambda: False)
    monkeypatch.setattr(
        tac,
        "_run_tac_setup",
        lambda: (_ for _ in ()).throw(AssertionError("setup should not run")),
    )
    with pytest.raises(RuntimeError, match="Start them on the host"):
        tac.ensure_tac_services()


def test_ensure_tac_uses_locked_image_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = tmp_path / "tac-store"
    calls: list[str] = []
    monkeypatch.setenv(OAS_START_TAC_ENV, "1")
    monkeypatch.setenv(OAS_TAC_IMAGE_STORE_ENV, str(store))
    monkeypatch.delenv("CONTAINERS_STORAGE_CONF", raising=False)
    monkeypatch.setattr(tac, "tac_services_ready", lambda: False)
    monkeypatch.setattr(tac, "compose_available", lambda: True)
    monkeypatch.setattr(tac, "_run_tac_setup", lambda: calls.append("setup"))

    tac.ensure_tac_services()
    writer_conf = Path(os.environ["CONTAINERS_STORAGE_CONF"]).read_text()
    images = store / "images"
    assert calls == ["setup"]
    assert (store / ".oas-tac-ready").is_file()
    assert f'graphroot = "{images}"' in writer_conf
    assert "additionalimagestores" not in writer_conf

    tac.ensure_tac_services()
    reader_conf = Path(os.environ["CONTAINERS_STORAGE_CONF"]).read_text()
    assert calls == ["setup", "setup"]
    assert "additionalimagestores" in reader_conf
    assert f'"{images}"' in reader_conf


def test_openagentsafety_beaker_env_requires_weka() -> None:
    from olmo_eval.cli.beaker.job_assembler import openagentsafety_beaker_env

    with pytest.raises(ValueError, match="Weka"):
        openagentsafety_beaker_env(["openagentsafety"], "ai2/rhea")
    assert openagentsafety_beaker_env(["tau2_bench"], "ai2/rhea") == {}


def test_assemble_openagentsafety_job_sets_tac_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    from olmo_eval.cli.beaker.job_assembler import assemble_external_eval_job

    monkeypatch.setattr(
        "olmo_eval.launch.beaker.mirror.get_registry_mirror_url",
        lambda: (_ for _ in ()).throw(RuntimeError("no mirror")),
    )
    job = assemble_external_eval_job(
        name="oas",
        model="test-model",
        external_evals=["openagentsafety"],
        cluster="ai2/jupiter",
        num_gpus=1,
        workspace="ai2/oe-data",
        beaker_image="test-image",
    )
    assert job.env_vars[OAS_START_TAC_ENV] == "1"
    assert job.env_vars[OAS_TAC_IMAGE_STORE_ENV] == OAS_TAC_IMAGE_STORE
    assert job.command[job.command.index("--runtime") + 1] == "podman"


def test_docker_error_requires_docker_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "olmo_eval.evals.external.benchmarks.openagentsafety.eval.shutil.which",
        lambda _name: None,
    )
    error = OpenAgentSafetyExternalEval()._docker_error()
    assert error is not None
    assert "docker" in error.lower()
    assert "podman" in error.lower()
