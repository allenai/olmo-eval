from __future__ import annotations

import os
from pathlib import Path

import sitecustomize


def test_expose_packaged_cuda_toolkit(monkeypatch, tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    cuda_root = site_packages / "nvidia" / "cu13"
    nvcc = cuda_root / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    (cuda_root / "lib").mkdir()
    (cuda_root / "lib" / "libcudart.so.13").touch()
    (cuda_root / "lib" / "libnvrtc.so.13").touch()

    monkeypatch.setattr(sitecustomize.site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "missing-cuda"))
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("CUDACXX", raising=False)

    sitecustomize._expose_packaged_cuda_toolkit()

    assert os.environ["CUDA_HOME"] == str(cuda_root)
    assert os.environ["CUDACXX"] == str(nvcc)
    assert (cuda_root / "lib64").is_symlink()
    assert (cuda_root / "lib64").resolve() == cuda_root / "lib"
    assert (cuda_root / "lib" / "libcudart.so").resolve() == (
        cuda_root / "lib" / "libcudart.so.13"
    )
    assert (cuda_root / "lib" / "libnvrtc.so").resolve() == (
        cuda_root / "lib" / "libnvrtc.so.13"
    )
    assert os.environ["PATH"].split(os.pathsep)[0] == str(cuda_root / "bin")


def test_expose_packaged_cuda_toolkit_preserves_working_cuda_home(
    monkeypatch, tmp_path: Path
) -> None:
    cuda_root = tmp_path / "cuda"
    nvcc = cuda_root / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    monkeypatch.setenv("CUDA_HOME", str(cuda_root))
    monkeypatch.delenv("CUDACXX", raising=False)

    sitecustomize._expose_packaged_cuda_toolkit()

    assert os.environ["CUDA_HOME"] == str(cuda_root)
    assert "CUDACXX" not in os.environ
