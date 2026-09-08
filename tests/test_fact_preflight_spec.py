"""The CPU preflight may inherit dependency setup, never serving or scoring credentials."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize("has_browser_setup", [False, True])
def test_cpu_preflight_preserves_setup_and_removes_model_runtime(tmp_path, has_browser_setup):
    setup = (
        "uv venv --python 3.12 /opt/venv &&\n"
        "uv venv --python 3.12 /opt/oe-venv &&\n"
        "uv pip install --python /opt/venv/bin/python -e /opt/lit-agents"
    )
    if has_browser_setup:
        setup += " &&\n/opt/venv/bin/python -m playwright install --with-deps chromium"
    values = {
        "GANTRY_INSTALL_CMD": setup,
        "GITHUB_REPO": "allenai/olmo-eval",
        "GANTRY_VERSION": "3.7.0",
        "GANTRY_NO_PYTHON": "1",
        "GANTRY_EXEC_METHOD": "exec",
        "GANTRY_RUNTIME_DIR": "/gantry-runtime",
        "LIT_AGENTS_GIT_REF": "b" * 40,
        "VLLM_PYTHON": "/usr/bin/python3",
    }
    source = {
        "version": "v2",
        "budget": "ai2/oe-omai",
        "tasks": [
            {
                "image": {"docker": "vllm/vllm-openai:target-image"},
                "datasets": [
                    {"mountPath": "/gantry", "source": {"beaker": "fixture"}},
                    {"mountPath": "/weights", "source": {"beaker": "model-weights"}},
                ],
                "envVars": [{"name": k, "value": v} for k, v in values.items()]
                + [
                    {"name": "GITHUB_TOKEN", "secret": "code-access"},
                    {"name": "OPENAI_API_KEY", "secret": "must-not-inherit"},
                    {"name": "HF_TOKEN", "secret": "must-not-inherit"},
                ],
                "command": ["must-not-serve-models"],
                "resources": {"gpuCount": 8},
            }
        ],
    }
    source_path, output = tmp_path / "glm.yaml", tmp_path / "cpu.yaml"
    source_path.write_text(yaml.safe_dump(source))
    renderer = Path(__file__).resolve().parents[1] / "scripts/internal/fact_preflight/render.py"
    subprocess.run(
        [sys.executable, str(renderer), str(source_path), "a" * 40, str(output)], check=True
    )
    spec = yaml.safe_load(output.read_text())
    task = spec["tasks"][0]
    env = {r["name"]: r for r in task["envVars"]}
    assert task["image"] == source["tasks"][0]["image"]
    assert env["GANTRY_INSTALL_CMD"]["value"].startswith(setup)
    assert task["resources"]["gpuCount"] == 0
    assert task["timeout"] == "15m"
    assert spec["retry"]["allowedTaskRetries"] == 0
    assert task["datasets"] == [source["tasks"][0]["datasets"][0]]
    assert [r["name"] for r in task["envVars"] if "secret" in r] == ["GITHUB_TOKEN"]
    assert "VLLM_PYTHON" not in env
    assert env["DEEPRESEARCH_SKIP_FACT"]["value"] == "0"
    assert env["GIT_REF"]["value"] == "a" * 40
    assert (
        "/opt/venv/bin/python scripts/internal/fact_preflight/supervise.py" in task["arguments"][-1]
    )
    assert "must-not" not in output.read_text()
