"""Opt-in Python startup hooks used by isolated inference environments."""

from __future__ import annotations

import contextlib
import os
import site
import sys
from pathlib import Path

source_root = str(Path(__file__).resolve().parent)
if source_root not in sys.path:
    sys.path.insert(0, source_root)


def _expose_current_environment_binaries() -> None:
    """Make console scripts installed beside this Python visible to JITs.

    olmo-eval runs vLLM in an isolated environment. FlashInfer launches
    ``ninja`` with ``subprocess`` rather than importing it, while the isolated
    environment's bin directory is not necessarily on PATH. Updating PATH
    here is scoped to each Python process: the harness sees /opt/venv/bin and
    the vLLM process sees /opt/vllm-venv/bin.
    """

    # Do not resolve the executable symlink: in a venv it commonly points to
    # /usr/local/bin/python, while its sibling console scripts live in the
    # venv's own bin directory.
    bin_dir = str(Path(sys.executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in path_entries:
        os.environ["PATH"] = os.pathsep.join([bin_dir, *path_entries])


_expose_current_environment_binaries()


def _expose_packaged_cuda_toolkit() -> None:
    """Point CUDA JITs at the toolkit installed in the Python environment.

    CUDA 13's ``nvidia-cuda-nvcc`` wheel installs the compiler under
    ``site-packages/nvidia/cu13``.  Runtime-oriented CUDA images can therefore
    have a complete compiler toolchain without ``/usr/local/cuda/bin/nvcc``.
    FlashInfer and PyTorch extensions honor ``CUDA_HOME``, so use the packaged
    toolkit whenever the configured system toolkit has no compiler.
    """

    configured_home = os.environ.get("CUDA_HOME")
    if configured_home and (Path(configured_home) / "bin" / "nvcc").is_file():
        return

    for site_packages in site.getsitepackages():
        cuda_root = Path(site_packages) / "nvidia" / "cu13"
        nvcc = cuda_root / "bin" / "nvcc"
        if not nvcc.is_file():
            continue

        # NVIDIA's CUDA 13 Python wheels install runtime libraries in ``lib``,
        # while PyTorch's extension helper (and therefore FlashInfer's JIT)
        # unconditionally links against ``CUDA_HOME/lib64``. Preserve the
        # packaged layout and provide the conventional toolkit alias.
        cuda_lib = cuda_root / "lib"
        cuda_lib64 = cuda_root / "lib64"
        if cuda_lib.is_dir() and not cuda_lib64.exists():
            # Tensor-parallel workers may import this hook concurrently.
            with contextlib.suppress(FileExistsError):
                cuda_lib64.symlink_to("lib", target_is_directory=True)
        # Runtime wheels omit the unversioned linker names that a full toolkit
        # ships, but JIT build systems pass ``-lcudart`` and ``-lnvrtc``.
        for library in ("cudart", "nvrtc"):
            library_link = cuda_lib / f"lib{library}.so"
            if cuda_lib.is_dir() and not library_link.exists():
                library_versions = sorted(cuda_lib.glob(f"lib{library}.so.*"))
                if library_versions:
                    with contextlib.suppress(FileExistsError):
                        library_link.symlink_to(library_versions[-1].name)

        os.environ["CUDA_HOME"] = str(cuda_root)
        os.environ["CUDACXX"] = str(nvcc)
        cuda_bin = str(cuda_root / "bin")
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if cuda_bin not in path_entries:
            os.environ["PATH"] = os.pathsep.join([cuda_bin, *path_entries])
        break


_expose_packaged_cuda_toolkit()


if (
    os.environ.get("OLMO_EVAL_VLLM_GPTOSS_NONPOW2_TOPK_FALLBACK") == "1"
    and "vllm-venv" in sys.executable
):
    from olmo_eval.compat.vllm_gptoss_topk import (
        apply_gptoss_non_power_of_two_topk_patch,
    )

    if apply_gptoss_non_power_of_two_topk_patch():
        print(
            "olmo-eval: enabled GPT-OSS non-power-of-two top-k compatibility fallback",
            file=sys.stderr,
        )


if os.environ.get("OLMO_EVAL_VLLM_GPTOSS_REFERENCE_SCALE") == "1" and "vllm-venv" in sys.executable:
    from olmo_eval.compat.vllm_gptoss_topk import (
        apply_gptoss_reference_scale_patch,
    )

    gptoss_reference_k = int(os.environ.get("OLMO_EVAL_VLLM_GPTOSS_REFERENCE_K", "4"))
    if apply_gptoss_reference_scale_patch(reference_k=gptoss_reference_k):
        print(
            "olmo-eval: enabled GPT-OSS "
            f"reference-scaled routing (reference_k={gptoss_reference_k})",
            file=sys.stderr,
        )


if (
    os.environ.get("OLMO_EVAL_VLLM_GLM_REFERENCE_TRUNCATION") == "1"
    and "vllm-venv" in sys.executable
):
    from olmo_eval.compat.vllm_glm_reference_truncation import (
        apply_glm_reference_truncation_patch,
    )

    glm_keep_k = int(os.environ.get("OLMO_EVAL_VLLM_GLM_KEEP_K", "4"))
    glm_reference_k = int(os.environ.get("OLMO_EVAL_VLLM_GLM_REFERENCE_K", "8"))
    glm_num_experts = int(os.environ.get("OLMO_EVAL_VLLM_GLM_NUM_EXPERTS", "256"))
    if apply_glm_reference_truncation_patch(
        keep_k=glm_keep_k,
        reference_k=glm_reference_k,
        num_experts=glm_num_experts,
    ):
        print(
            "olmo-eval: enabled GLM native-reference truncation "
            f"(kernel_k={glm_keep_k}, reference_k={glm_reference_k}, "
            f"global_num_experts={glm_num_experts})",
            file=sys.stderr,
        )


qwen_expert_weight_mode = (
    os.environ.get("OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_MODE", "").strip().lower()
)
if qwen_expert_weight_mode and "vllm-venv" in sys.executable:
    from olmo_eval.compat.qwen_expert_weight_policies import (
        qwen_expert_weight_policy_from_env,
    )
    from olmo_eval.compat.vllm_qwen_expert_weights import (
        apply_qwen_expert_weight_patch,
    )

    qwen_expert_weight_policy = qwen_expert_weight_policy_from_env(os.environ)
    qwen_record_realized_k = os.environ.get("OLMO_EVAL_VLLM_QWEN_RECORD_REALIZED_K", "0") == "1"
    qwen_num_experts = int(os.environ.get("OLMO_EVAL_VLLM_QWEN_NUM_EXPERTS", "128"))
    if apply_qwen_expert_weight_patch(
        policy=qwen_expert_weight_policy,
        record_realized_k=qwen_record_realized_k,
        num_experts=qwen_num_experts,
    ):
        print(
            "olmo-eval: enabled Qwen expert-weight policy "
            f"{qwen_expert_weight_policy.describe()} "
            f"for {qwen_num_experts} routed experts",
            file=sys.stderr,
        )
