"""Bundled vLLM installs consume lock constraints in both Beaker environments."""

import os
import subprocess

import pytest

from olmo_eval.launch.beaker.launcher import BeakerLauncher


def rendered_install(tmp_path, *, isolated=False, export_fails=False):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        'case "$1:$2" in\n'
        "  pip:freeze) echo 'torch==2.10.0'; echo 'nvidia-cublas-cu12==12.8.4.1';;\n"
        "  export:*)\n"
        "    echo 'torch==2.11.0'; echo 'nvidia-cublas-cu12==12.9.0.13'\n"
        "    echo 'transformers==5.8.1'; echo 'xgrammar==0.1.34'\n"
        "    echo 'ifbench @ git+https://example.invalid/ifbench@abc'\n"
        + ("    echo 'export failed' >&2; exit 23\n" if export_fails else "")
        + "    ;;\n"
        '  pip:install) printf "%s\\n" "$*" >> "$INSTALL_LOG";;\n'
        "  *) exit 99;;\n"
        "esac\n"
    )
    uv.chmod(0o755)
    launcher = object.__new__(BeakerLauncher)
    command = launcher._build_install_cmd(
        extras=["vllm"], task_packages=["task-dep==1"], vllm_isolated_venv=isolated
    )
    command = command.replace("/tmp/", str(tmp_path) + "/").replace(
        "/gantry-runtime", str(tmp_path)
    )
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    env["INSTALL_LOG"] = str(tmp_path / "installs.txt")
    return subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True)


def test_inline_install_and_task_packages_use_image_and_lock_constraints(tmp_path):
    result = rendered_install(tmp_path)
    assert result.returncode == 0, result.stderr
    constraints = (tmp_path / "inline-vllm-constraints.txt").read_text().splitlines()
    assert constraints == [
        "torch==2.10.0",
        "nvidia-cublas-cu12==12.8.4.1",
        "transformers==5.8.1",
        "xgrammar==0.1.34",
    ]
    installs = (tmp_path / "installs.txt").read_text().splitlines()
    assert len(installs) == 2
    assert all(f"-c {tmp_path}/inline-vllm-constraints.txt" in line for line in installs)


@pytest.mark.parametrize("isolated", [False, True])
def test_failed_lock_export_stops_installation_even_with_partial_output(tmp_path, isolated):
    result = rendered_install(tmp_path, isolated=isolated, export_fails=True)
    assert result.returncode == 23
    assert "export failed" in result.stderr
    assert not (tmp_path / "installs.txt").exists()


@pytest.mark.parametrize("isolated", [False, True])
def test_custom_vllm_package_keeps_its_own_dependency_resolution(isolated):
    launcher = object.__new__(BeakerLauncher)
    command = launcher._build_install_cmd(
        extras=["clients"],
        provider_packages=["vllm==0.19.1"],
        vllm_isolated_venv=isolated,
    )
    assert "uv export" not in command
    assert "inline-vllm-constraints" not in command
    assert "'vllm==0.19.1' -c /tmp/cuda-constraints.txt" in command
