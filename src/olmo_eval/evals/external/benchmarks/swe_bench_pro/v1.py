"""Task construction for the original SWE-Bench Pro release (v1).

v1 ships its data as a Hugging Face dataset and its per-task run scripts and
parsers in the harness repository. This module turns one dataset row into a
task whose verifier follows the same ``/tests/test.sh`` contract as v2.
"""

from __future__ import annotations

import ast
import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .task import SWEBenchProTask

DATASET_NAME = "ScaleAI/SWE-bench_Pro"
# Tag v1.0 of the dataset
DATASET_REVISION = "307bd006a0a3807424d3c5525eaf98ca8e63aed3"
IMAGE_REPOSITORY = "jefzda/sweap-images"
WORKING_DIR = "/app"
TIMEOUT_SECONDS = 3000.0

# The v1 images start /bin/bash as their entrypoint, which would swallow the
# sandbox server's start command.
DOCKERFILE_EXTRA = ("ENTRYPOINT []",)

INSTRUCTION_TEMPLATE = """\
<uploaded_files>
{working_dir}
</uploaded_files>
I've uploaded a code repository in the directory {working_dir}. Consider the following PR \
description:

<pr_description>
{problem_statement}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements \
specified in the <pr_description> are met?
I've already taken care of all changes to any of the test files described in the \
<pr_description>. This means you DON'T have to modify the testing logic or any of the tests \
in any way!
Your task is to make the minimal changes to non-tests files in the {working_dir} directory to \
ensure the <pr_description> is satisfied.
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to find and read code relevant to the \
<pr_description>
2. Create a script to reproduce the error and execute it using the bash tool, to confirm the \
error
3. Edit the source code of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
Your thinking should be thorough and so it's fine if it's very long."""

_TEST_SCRIPT_TEMPLATE = """\
#!/bin/bash
# Verifier for a SWE-Bench Pro v1 task. Writes 1 to the reward file when every
# fail-to-pass and pass-to-pass test passes.
set -uo pipefail
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

cd {working_dir} || {{ echo "ERROR: {working_dir} does not exist"; exit 1; }}

# Restore the task's hidden test files.
{restore_command}

bash /tests/run_script.sh {selected_tests} > /tmp/sbp_stdout.log 2> /tmp/sbp_stderr.log
cp /tmp/sbp_stdout.log /logs/verifier/run-script-stdout.txt 2>/dev/null || true
cp /tmp/sbp_stderr.log /logs/verifier/run-script-stderr.txt 2>/dev/null || true
PYTHON=$(command -v python3 || command -v python)
"$PYTHON" /tests/parser.py /tmp/sbp_stdout.log /tmp/sbp_stderr.log /tmp/sbp_output.json \\
    || {{ echo "ERROR: parser failed"; cat /tmp/sbp_stderr.log; exit 1; }}
cp /tmp/sbp_output.json /logs/verifier/output.json 2>/dev/null || true

"$PYTHON" - <<'EVAL_EOF'
import json
import sys

with open("/tmp/sbp_output.json") as f:
    results = json.load(f)
with open("/tests/config.json") as f:
    config = json.load(f)

passed = {{t.get("name", "") for t in results.get("tests", []) if t.get("status") == "PASSED"}}
required = set(config["fail_to_pass"]) | set(config["pass_to_pass"])
missing = sorted(required - passed)
print(f"Required tests: {{len(required)}}, passed: {{len(required) - len(missing)}}")
if missing:
    print(f"Missing tests: {{missing[:50]}}")
    print("RESULT: FAILED")
    sys.exit(1)
print("RESULT: PASSED")
with open("/logs/verifier/reward.txt", "w") as f:
    f.write("1\\n")
EVAL_EOF
"""


def parse_list_field(value: Any) -> list[str]:
    """Parse a v1 list column, which may be a JSON or a Python-literal string."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list, got {type(parsed).__name__}")
    return [str(v) for v in parsed]


def parse_text_field(value: Any) -> str:
    """Parse a v1 text column, decoding values that were stored JSON-encoded."""
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    if len(stripped) >= 2 and stripped.startswith('"') and stripped.endswith('"'):
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return value
        if isinstance(decoded, str):
            return decoded
    return value


def build_instruction(row: Mapping[str, Any], working_dir: str = WORKING_DIR) -> str:
    """Build the agent prompt used by the benchmark's reference agent setup."""
    problem_statement = (
        f"{parse_text_field(row.get('problem_statement'))}\n\n"
        f"Requirements:\n{parse_text_field(row.get('requirements'))}\n\n"
        f"New interfaces introduced:\n{parse_text_field(row.get('interface'))}"
    )
    return INSTRUCTION_TEMPLATE.format(working_dir=working_dir, problem_statement=problem_statement)


def build_test_script(row: Mapping[str, Any], working_dir: str = WORKING_DIR) -> str:
    """Build the ``test.sh`` verifier for one v1 task.

    Like the official harness, only the last line of ``before_repo_set_cmd`` is
    run: it checks the hidden test files out of the fixing commit.
    """
    before_lines = [
        line.strip()
        for line in str(row.get("before_repo_set_cmd", "")).strip().splitlines()
        if line.strip()
    ]
    restore_command = before_lines[-1] if before_lines else "true"
    selected = ",".join(parse_list_field(row.get("selected_test_files_to_run")))
    return _TEST_SCRIPT_TEMPLATE.format(
        working_dir=shlex.quote(working_dir),
        restore_command=restore_command,
        selected_tests=shlex.quote(selected) if selected else "",
    )


def image_for_row(row: Mapping[str, Any]) -> str:
    """Return the Docker Hub image for a v1 task."""
    return f"{IMAGE_REPOSITORY}:{row['dockerhub_tag']}"


def task_from_row(row: Mapping[str, Any], run_scripts_dir: Path) -> SWEBenchProTask:
    """Build a task from a v1 dataset row and its run script directory.

    Args:
        row: One row of the v1 dataset.
        run_scripts_dir: The harness repository's ``run_scripts`` directory.

    Returns:
        The task.

    Raises:
        FileNotFoundError: If the task's run script or parser is missing.
    """
    instance_id = row["instance_id"]
    scripts_dir = run_scripts_dir / instance_id
    run_script = scripts_dir / "run_script.sh"
    parser = scripts_dir / "parser.py"
    for path in (run_script, parser):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path.name} for {instance_id}")

    config = {
        "fail_to_pass": parse_list_field(row.get("fail_to_pass")),
        "pass_to_pass": parse_list_field(row.get("pass_to_pass")),
        "selected_test_files_to_run": parse_list_field(row.get("selected_test_files_to_run")),
    }
    test_files = {
        "test.sh": build_test_script(row).encode(),
        "run_script.sh": run_script.read_bytes(),
        "parser.py": parser.read_bytes(),
        "config.json": json.dumps(config, indent=1).encode(),
    }
    return SWEBenchProTask(
        instance_id=instance_id,
        repo=str(row["repo"]),
        image=image_for_row(row),
        working_dir=WORKING_DIR,
        instruction=build_instruction(row),
        test_files=test_files,
        gold_patch=str(row.get("patch") or ""),
        agent_timeout=TIMEOUT_SECONDS,
        verifier_timeout=TIMEOUT_SECONDS,
        dockerfile_extra=DOCKERFILE_EXTRA,
    )
