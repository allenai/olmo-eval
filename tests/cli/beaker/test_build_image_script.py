"""Check image-builder flags without invoking Docker or downloading anything."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("devel", [False, True])
@pytest.mark.parametrize("sandbox", [False, True])
def test_build_script_forwards_cuda_flavor(tmp_path, devel, sandbox):
    script = Path(__file__).parents[3] / "scripts" / "build_image.sh"
    commands = tmp_path / "docker.jsonl"
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['DOCKER_TEST_COMMANDS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    docker.chmod(0o755)
    args = [
        "bash",
        str(script),
        "--platform",
        "linux/amd64",
        "--cuda-version",
        "13.0.2",
        "--torch-version",
        "2.13.0",
    ]
    if devel:
        args.append("--cuda-devel")
    if sandbox:
        args.append("--sandbox")
    subprocess.run(
        args,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "DOCKER_TEST_COMMANDS": str(commands),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    build = json.loads(commands.read_text().splitlines()[0])
    assert build[0] == "build"
    assert f"CUDA_IMAGE_FLAVOR={'devel' if devel else 'runtime'}" in build
    tag = "olmo-eval:cu1302-trc2130-amd64"
    if devel:
        tag += "-devel"
    if sandbox:
        tag += "-sandbox"
    assert build[build.index("-t") + 1] == tag
    assert build[build.index("--target") + 1] == ("runtime-sandbox" if sandbox else "runtime")
