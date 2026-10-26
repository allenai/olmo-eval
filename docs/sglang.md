# SGLang generation provider

Select `-o provider.kind=sglang` to use a local SGLang engine with the existing
olmo-eval tasks, generation harness, scoring, prediction saving and metrics.
The `sglang` dependency extra installs the pinned SGLang engine, olmo-sglang
model plugin and FLA. Beaker jobs select this extra automatically when the
resolved provider is `sglang`, just as vLLM jobs select the `vllm` extra.
Use separate environments for these backends; their dependency pins conflict.

## Installation and Beaker jobs

For a local Linux CUDA 13 / PyTorch 2.13 environment:

```bash
uv sync --frozen --no-group vllm --extra sglang
```

For remote evaluation, only the launcher is needed locally. The image contains
Python, PyTorch and CUDA; Gantry installs the provider extra at job startup.
Build a compatible base through the existing image workflow:

```bash
./scripts/build_image.sh --cuda-version 13.0.2 --torch-version 2.13.0 \
    --platform linux/amd64
./scripts/beaker/push_beaker_image.sh \
    --source olmo-eval:cu1302-trc2130-amd64 --workspace YOUR_WORKSPACE
```

Select that image with `--image` when launching. The repository's default
CUDA 12.8 / PyTorch 2.10 image is not compatible with this SGLang pin. A normal
launcher invocation for an exported Olmo KDA MoE Hugging Face checkpoint is:

```bash
uv run --frozen --no-group vllm olmo-eval beaker launch \
    --image YOUR_CUDA13_TORCH213_IMAGE \
    --harness default -o provider.kind=sglang \
    -o provider.kwargs.model_registration=olmo_sglang:register \
    -o provider.kwargs.tensor_parallel_size=1 \
    -o provider.kwargs.sampling_backend=flashinfer \
    -o provider.max_model_len=8192 \
    -m /weka/MODEL_BUCKET/path/to/hf-checkpoint --weka-bucket MODEL_BUCKET \
    -t gsm8k -o limit=5 -o max_tokens=256 \
    -w YOUR_WORKSPACE -B YOUR_BUDGET \
    --cluster ai2/jupiter --gpus 1 --priority high \
    --min-runtime 20m --timeout 20m --no-upload --dry-run
```

Replace the uppercase placeholders and remove `--dry-run` to submit. The limit
and token cap make this an installation check, not an accuracy measurement;
remove them and choose a suitable timeout for full evaluation. Configure normal
Beaker/GitHub access and checkpoint mounts as described in the
[launcher guide](../README.md#launching-on-beaker). Omit `--no-upload` for normal
evaluations after configuring Google Application Default Credentials.

The extra pins SGLang source `3145136dcd1238754e0ea2b2ffd546532119c71c`,
olmo-sglang `94ba0d5036c8ea7794636009d4ef233b0f3db32f`, and FLA 0.5.2.
SGLang requires Transformers 5.12.1 and PyTorch 2.13.0; the tested CUDA 13 build
uses Triton 3.7.1. The extra also pins tqdm 4.67.3 to avoid a dataset-download
regression. These are a qualified combination, not interchangeable latest
versions. The optional fused routing arithmetic (#10) remains off by default.

The uv build configuration skips SGLang's optional Rust HTTP/gRPC servers and
multimodal transport: this provider uses the inline Python Engine. No Rust
compiler is needed. This extra does not qualify Rust-server modes or image
inference. `provider.package` can replace the bundled engine extra for an
explicitly managed runtime; compatibility then belongs to that configuration.

## Provider settings

`provider.max_model_len` maps to `context_length`; `provider.max_concurrency`
maps to `max_running_requests`. Use `provider.kwargs.tensor_parallel_size` for
tensor parallelism so olmo-eval's GPU planner sees the allocation; the provider
maps it to SGLang's `tp_size`. The worker also forwards its allocated GPU count.
Additional engine arguments go under `provider.kwargs`, including
`mem_fraction_static` and `random_seed`. `provider.force_download` uses the
shared Hugging Face cache-refresh helper.
For model plugins, `provider.kwargs.model_registration=olmo_sglang:register`
calls that registration function before constructing the engine. Engine workers
must also be able to import the model package.

Chat requests use the model tokenizer's chat template with a generation prompt.
The provider forwards temperature, top-p, top-k, generation length and string
stops; EOS handling remains with SGLang. It retains sampled-token logprobs,
engine metadata, token IDs and finish reasons. The provider defaults
`SGLANG_RETURN_ORIGINAL_LOGPROB=1` before model registration or engine imports,
so sampled-token logprobs and `sum_logits` use raw model probabilities rather
than temperature-scaled probabilities. An explicitly conflicting flag or an
already-imported sampler with the other convention fails with a restart message.
Set the flag before importing SGLang in applications that initialize it elsewhere.
The enforced convention is recorded in request traces and output metadata;
it does not imply cross-engine numerical parity. Generation samples and their token IDs
are preserved separately from the top-ranked alternatives. Finish reasons use
the harness's `stop`/`length` strings; `sglang_finish_reason` in returned metadata
retains the engine's original matched-token or length detail.

Only the batched generation strategy is supported. Continuation likelihood
scoring, prompt truncation, automatic fitting of generation length to the context
window, image requests and tool requests fail explicitly. Aborted generation or
missing sampled-token probabilities raise errors rather than becoming scored
answers. The extra includes olmo-sglang; select its registration hook for Olmo
checkpoints. Model-specific optimization defaults come from the pinned plugin.

For top-p evaluations, select `provider.kwargs.sampling_backend=flashinfer`
in a compatible pinned runtime. This uses SGLang's existing nucleus sampler.
The olmo-sglang simple fused sampler applies only to unrestricted sampling;
its narrower coverage does not prevent efficient top-p generation. Preserve
the task's sampling settings when comparing engines.

For a small generation trial in an environment where compatible SGLang and
olmo-eval are already installed:

```bash
uv run --no-sync olmo-eval run \
    --harness default -o provider.kind=sglang \
    -o provider.kwargs.tensor_parallel_size=1 \
    -m /path/to/model -t gsm8k -o limit=5 --no-upload
```

Remote jobs install the SGLang extra into the selected compatible base image.
The launcher's CUDA/PyTorch constraints prevent silently replacing the base GPU
stack. Use `provider.package` / `provider.dependencies` for intentional overrides,
and retain the exact image, package pins and effective settings with results.

The provider delegates sampling to SGLang. It does not implement a sampler or
promise identical random draws to another backend for the same seed. Keep task
definitions and scorer revisions fixed when comparing providers, and record the
runtime and model-plugin revisions alongside the olmo-eval commit.

Equal weights and seeds also do not fix every numerical choice within an engine.
Autotuned kernels can select different launch configurations across processes
and produce different floating-point results. When investigating repeatability,
record the chosen kernel configurations and cache policy where available, along
with the runtime settings. Compare token IDs and logprobs on identical input
histories: after generated tokens differ, later logprobs describe different
conditional distributions. Repeated accuracy measurements and controlled
numerical checks answer different questions; neither substitutes for the other.

## Concurrency advice

The harness defaults to `batching.chunk_size=64` examples per call. The SGLang
provider leaves `provider.max_concurrency` unset unless configured; SGLang chooses
its effective request limit from its configuration and memory pools. A requested
limit can be reduced by the engine, including by KDA recurrent-state capacity.
Multiple samples expand each example into separate engine requests.

After two consecutive successful calls of the same size and generation budget
(at least 16 sequences), the provider checks the engine's resolved capacity once.
When the effective request limit allows twice that many sequences, and their
estimated token budget fits within half the token pool, it logs:

```text
Consider increasing concurrency: ... Try adding '-o batching.chunk_size=128'
after your existing --harness argument.
```

The estimate uses the longest observed expanded prompt in the two calls plus the
full requested output cap, including repeated samples. Calls reporting request
retractions do not trigger the hint. Uncapped generation (`max_tokens=None`),
missing/unsupported telemetry and multiple data-parallel workers suppress it.
It does not infer capacity from free VRAM or the requested concurrency limit,
and does not change any settings.
This is advice to benchmark a larger batch, not an automatic throughput tuner or
a guarantee that transient allocations (including image processing) will fit.
If the engine limit itself is the bottleneck, this diagnostic makes no memory
claim and does not suggest raising it without a separate capacity check.
