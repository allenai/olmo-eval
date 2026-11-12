# B300 vLLM runtime compatibility

[Hero B300 grouped-expert profiling (#390)](https://github.com/allenai/olmo-eval/issues/390)
reports a BF16 grouped-expert failure before predictions with PyTorch
`2.10.0+cu128` and vLLM `0.19.1`. The failing kernel has not been isolated:
CUDA errors are asynchronous, so the reported `F.silu` frame after grouped
matmul does not identify the cause.

## Preparing a CUDA 13 environment

PyTorch and vLLM version numbers alone do not identify their CUDA builds.
Public CUDA 13 artifacts include PyTorch `2.10.0+cu130` from the
[PyTorch CUDA 13 index](https://download.pytorch.org/whl/cu130) and the
[vLLM 0.19.1 CUDA 13 wheel](https://github.com/vllm-project/vllm/releases/download/v0.19.1/vllm-0.19.1+cu130-cp38-abi3-manylinux_2_35_x86_64.whl).
These are candidates for a separately validated environment; their availability
does not establish that installing them resolves the issue.

Keep the complete package inventory, base image and model-plugin source
revisions with each trial. Record whether the plugin enables torch grouped MoE
and FLA KDA, since those settings select different execution paths. Do not infer
plugin source identity from a filesystem-built distribution's version alone.

The current vLLM extra pins Transformers `5.8.1`. Transformers 5.13+ nests
`rope_parameters` per layer type, while vLLM 0.19.1 reads it flat; Olmo-3 mixed
sliding/full attention can therefore fail to load with a `rope_theta` KeyError.
See the compatibility comment beside the pin in `pyproject.toml`. Check the
intended model before reusing an environment across architectures.

Validate startup, finite outputs, and numerical agreement with a reference on
the intended checkpoint before running a full evaluation. Preserve the supported
runtime separately while testing a candidate; a startup-only check does not
establish numerical or benchmark parity. The repository's default image and
dependency pins remain unchanged.
