"""Derive a CPU-only browser preflight from the exact GLM generation spec, without submission."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("glm_spec", type=Path)
parser.add_argument("scorer_ref")
parser.add_argument("output", type=Path)
args = parser.parse_args()
assert re.fullmatch(r"[0-9a-f]{40}", args.scorer_ref)
source = yaml.safe_load(args.glm_spec.read_text())
target = source["tasks"][0]
env = {entry["name"]: entry for entry in target["envVars"]}
setup = env["GANTRY_INSTALL_CMD"]["value"].rstrip()
assert "uv venv --python 3.12 /opt/venv" in setup
assert "uv venv --python 3.12 /opt/oe-venv" in setup
assert "uv pip install --python /opt/venv/bin/python -e /opt/lit-agents" in setup
# The target RACE-only route omits the FACT optional setup; append make_dev_spec.py's
# original dependency/browser install chain, then the process inspection dependency.
for interpreter in ("/opt/venv/bin/python", "/opt/oe-venv/bin/python"):
    setup += f" &&\nuv pip install --python {interpreter} 'crawl4ai>=0.8'"
for interpreter in ("/opt/venv/bin/python", "/opt/oe-venv/bin/python"):
    setup += (
        f" &&\n{{ {interpreter} -m playwright install --with-deps chromium ||"
        f" {interpreter} -m playwright install chromium; }}"
    )
for interpreter in ("/opt/venv/bin/python", "/opt/oe-venv/bin/python"):
    verify = (
        "from playwright.sync_api import sync_playwright; "
        "p = sync_playwright().start(); b = p.chromium.launch(); "
        "b.close(); p.stop(); print('playwright chromium launches')"
    )
    setup += f' &&\n{interpreter} -c "{verify}"'
for interpreter in ("/opt/venv/bin/python", "/opt/oe-venv/bin/python"):
    setup += f" &&\nuv pip install --python {interpreter} psutil"
kept = [
    "GITHUB_REPO",
    "GANTRY_VERSION",
    "GANTRY_NO_PYTHON",
    "GANTRY_EXEC_METHOD",
    "GANTRY_RUNTIME_DIR",
    "LIT_AGENTS_GIT_REF",
    "GITHUB_TOKEN",
]
env_vars = [env[key] for key in kept]
env_vars += [
    {"name": key, "value": value}
    for key, value in {
        "GIT_REF": args.scorer_ref,
        "GIT_BRANCH": "yilun/w0184-fact-recovery",
        "GANTRY_TASK_NAME": "w0184-fact-chromium-cpu",
        "GANTRY_INSTALL_CMD": setup,
        "RESULTS_DIR": "/results",
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "HF_HUB_OFFLINE": "1",
        "DEEPRESEARCH_SKIP_FACT": "0",
        "DEEPRESEARCH_FACT_PAGE_TIMEOUT_MS": "1000",
        "DEEPRESEARCH_FACT_JUDGE_TIMEOUT_S": "10",
        "DEEPRESEARCH_FACT_CRAWLER_RECYCLE_AFTER": "200",
        "PLAYWRIGHT_BROWSERS_PATH": "/opt/w0184-playwright",
        "W0184_IMAGE": target["image"]["docker"],
    }.items()
]
# Setup needs no model download. The GitHub credential only retrieves the two pinned source trees.
command = [
    "bash",
    "-c",
    "set -e; apt-get update -qq; apt-get install -y -qq --no-install-recommends git gh; "
    'exec timeout --signal=TERM --kill-after=10s 850s bash /gantry/entrypoint.sh "$@"',
    "--",
]
script = """set -Eeuo pipefail
mkdir -p /results/w0184
cd /gantry-runtime
test "$(git rev-parse HEAD)" = "$GIT_REF"
printf '%s\\n' "$GIT_REF" > /results/w0184/scorer-ref.txt
printf '%s\\n' "$W0184_IMAGE" > /results/w0184/image.txt
for label in planner single; do
  if [ "$label" = planner ]; then py=/opt/venv/bin/python; else py=/opt/oe-venv/bin/python; fi
  uv pip freeze --python "$py" > "/results/w0184/$label.freeze.txt"
done
unset GITHUB_TOKEN
exec /opt/venv/bin/python scripts/internal/fact_preflight/supervise.py
"""
task = {
    "name": "w0184-fact-chromium-cpu",
    "image": target["image"],
    "datasets": [entry for entry in target["datasets"] if entry["mountPath"] == "/gantry"],
    "command": command,
    "arguments": ["bash", "-lc", script],
    "envVars": env_vars,
    "result": {"path": "/results"},
    "resources": {"gpuCount": 0, "cpuCount": 8, "memory": "24GiB", "sharedMemory": "4GiB"},
    "context": {"priority": "normal", "autoResume": False},
    "timeout": "15m",
}
# CPU scheduling is left to Beaker; no GPU-cluster constraint or weight storage mounts.
spec = {
    "version": "v2",
    "budget": source["budget"],
    "description": (
        "W0184 real Chromium FACT recovery preflight, two scoring interpreters, "
        "local fixtures and judges, zero GPUs"
    ),
    "retry": {"allowedTaskRetries": 0},
    "tasks": [task],
}
args.output.write_text(yaml.safe_dump(spec, sort_keys=False))
args.output.with_suffix(".provenance.json").write_text(
    json.dumps(
        {
            "target_spec_sha256": hashlib.sha256(args.glm_spec.read_bytes()).hexdigest(),
            "scorer_ref": args.scorer_ref,
            "image": target["image"],
            "lit_agents_ref": env["LIT_AGENTS_GIT_REF"]["value"],
            "source_install_chain_preserved": True,
            "cpu_only": True,
            "submitted": False,
            "image_digest": None,
            "image_digest_note": (
                "The target spec records a commit-tagged image, not a registry digest."
            ),
        },
        indent=2,
    )
    + "\n"
)
