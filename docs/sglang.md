# SGLang generation provider

Select `-o provider.kind=sglang` to use a local SGLang engine with the existing
olmo-eval tasks, generation harness, scoring, prediction saving and metrics.
The `sglang` dependency extra installs the pinned SGLang engine, olmo-sglang
model plugin and FLA. Beaker jobs select this extra automatically when the
resolved provider is `sglang`, just as vLLM jobs select the `vllm` extra.
Use separate environments for these backends; their dependency pins conflict.

The root lock resolves these mutually exclusive extras separately. Compatibility
pins in the vLLM extra and dependency group preserve its existing package versions
instead of inheriting SGLang's newer dependencies. When upgrading vLLM, update both
sets of pins together and check exported dependency profiles for unintended changes.

## Installation and Beaker jobs

For a local Linux CUDA 13 / PyTorch 2.13 environment:

```bash
uv sync --frozen --no-group vllm --extra sglang
```

For remote evaluation, only the launcher is needed locally. Selecting SGLang
also selects the shared CUDA 13.0.2 / PyTorch 2.13 compiler-equipped base image
`01M5VBMKPG8BX4BVJBGVBMF64V`, owned by Ai2 in `open-instruct-dev`
(name: `robertb/olmo-eval-cu1302-trc2130-amd64-devel`). The author namespace
records who published it; the immutable ID selects the shared image. Gantry installs
the extra at job startup; users do not build an image or install GPU packages on
their laptop. The launcher selects images per model, so other providers keep
their existing default. Explicit `--image` or YAML `beaker_image` settings win.

For an exported Olmo KDA MoE Hugging Face checkpoint:

```bash
uv run --frozen --no-group vllm olmo-eval beaker launch \
    --harness default -o provider.kind=sglang \
    -o provider.kwargs.model_registration=olmo_sglang:register \
    -o provider.trust_remote_code=true \
    -o provider.kwargs.tensor_parallel_size=1 \
    -o provider.dtype=bfloat16 -o provider.max_model_len=8192 \
    -m /weka/MODEL_BUCKET/path/to/hf-checkpoint --weka-bucket MODEL_BUCKET \
    -t gsm8k -o limit=5 -o max_tokens=256 \
    -w YOUR_WORKSPACE -B YOUR_BUDGET \
    --cluster ai2/jupiter --gpus 1 --priority high \
    --min-runtime 20m --timeout 20m --no-upload --dry-run
```

Replace the uppercase placeholders and remove `--dry-run` to submit. The example
trusts the checkpoint's custom Hugging Face configuration code; use that flag only
for checkpoints whose code you trust. The limit
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
versions. The optional [fused routing arithmetic (olmo-sglang #10)](https://github.com/allenai/olmo-sglang/pull/10)
remains off by default.
For this KDA plugin pin, keep `cuda_graph_backend_prefill=disabled`: the
upstream default prefill graph backend pads tensors incompatibly with the KDA
prefill path. Decode CUDA graphs remain enabled.

The uv build configuration skips SGLang's optional Rust HTTP/gRPC servers and
multimodal transport: this provider uses the inline Python Engine. No Rust
compiler is needed. This extra does not qualify Rust-server modes or image
inference. `provider.package` can replace the bundled engine extra for an
explicitly managed runtime; compatibility then belongs to that configuration.

### Rebuilding the base image

The shared base uses the existing repository build and publication scripts:

```bash
./scripts/build_image.sh --cuda-devel --cuda-version 13.0.2 --torch-version 2.13.0 \
    --platform linux/amd64
./scripts/beaker/push_beaker_image.sh \
    --source olmo-eval:cu1302-trc2130-amd64-devel --workspace YOUR_WORKSPACE
```

SGLang compiles CUDA kernels during startup, so `--cuda-devel` is required even
when all Python packages install successfully. Select a replacement with `--image`.
The ordinary CUDA 12.8 / PyTorch 2.10 base
is incompatible with this SGLang pin; installation constraints reject it. The
launcher checks `CUDA_VERSION` inside SGLang images, including explicit images
and aliases. For CUDA 13 images it selects `/usr/local/cuda/compat` on hosts with
drivers older than R580, and native libraries on R580 or newer, before Python.
This follows the [CUDA 13 driver requirements](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/).
CUDA 12 images and images without `CUDA_VERSION` retain their own library
configuration; custom images must supply compatible driver libraries. An explicit
`--env LD_LIBRARY_PATH=...` always takes precedence. Qualify a new base
with a small GPU evaluation before changing the shared default.

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

Returned `LMOutput` objects contain sampled-token logprobs and token IDs. The
standard predictions JSONL saves generation text, finish reasons and logprob
summaries, not the full per-token arrays or arbitrary provider metadata.
Numerical replay tools must save those returned fields explicitly.

Only the batched generation strategy is supported. Continuation likelihood
scoring, prompt truncation, automatic fitting of generation length to the context
window, image requests and tool requests fail explicitly. Request-specific
preparation errors and engine aborts leave the affected request unscored without
discarding other requests in the batch. If any sample aborts, its entire logical
request fails: pass@k never scores a reduced sample budget. All-output aborts,
engine failures and malformed or missing logprobs still raise errors.
The extra includes olmo-sglang; select its registration hook for Olmo
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

## Automatic serving profiles

`provider.kwargs.profile=auto` is the default. With `olmo_sglang:register`,
the provider reads checkpoint geometry and the assigned GPU's name and memory
before constructing the engine. It does not execute checkpoint configuration
code to select a profile or guess hardware from the cluster name.

The initial profiles cover the measured BF16, 16-layer Olmo 3.5 KDA MoE shape
(hidden size 1024, 512 routed experts, top-16, latent width 512), dedicated TP1,
nonquantized text inference and an explicitly configured context up to 34,816.
All cache, attention and recurrent-state dimensions must match, not just the
model name. The GPU must have at least 95% of its memory free at startup.

| Hardware / cache choice | Running sequences | KV token pool | Recurrent slots | Memory fraction |
| --- | ---: | ---: | ---: | ---: |
| B300 288 GB, cache on (default) | 512 | 17,825,792 | 3,072 | 0.7 |
| B300, explicit cache off | 512 | 17,825,792 | 1,024 | 0.7 |
| H100 80 GB, cache off (default) | 512 | 6,684,672 | 512 | 0.9 |
| H100, explicit cache on | 320 | 4,456,448 | 1,920 | 0.9 |
| A100 80 GB, cache off | 512 | 6,684,672 | 512 | 0.9 |

The H100/B300 configurations come from the RL study; A100 comes from a bounded
olmo-eval admission comparison. These are starting configurations, not universal
throughput optima. A100 validation used short capped generations; long-response
throughput has not been qualified there. Profiles also select 8,192-token prefill chunks,
full decode graphs through admission, disabled prefill graphs, Triton attention
and FlashInfer sampling. H100's KV pool cannot hold all admitted requests at the
maximum context simultaneously; actual lengths and prefix reuse determine
occupancy. The engine may reduce admission further.

Explicit engine settings always win. For example, `provider.max_concurrency=128`
reduces default pool and graph sizing to that admission; explicit pool or graph
settings remain unchanged. Use `provider.kwargs.disable_radix_cache=false` to
request caching, or `provider.kwargs.profile=generic` to opt out of all profile
defaults. Unknown GPUs (including A100 40 GB), A100 prefix caching, model shapes, vision,
tensor parallelism, quantization, larger contexts and occupied GPUs retain
engine defaults with a logged fallback reason. No context limit or task sampling
setting is silently changed.

`batching.chunk_size=auto` is the harness default. With a selected SGLang profile,
it feeds the resolved engine admission and counts expanded samples: pass@16
with 512 slots normally submits 32 examples per call. A single example requesting
more samples than the limit is submitted intact and queued by the engine.
Explicit integer batch sizes, including 64, are preserved exactly. Other
providers and generic SGLang retain the historical 64-example batch behavior.

Requested provider options remain in the manifest. Before generation starts,
workers return their selected profile, effective engine settings and installed
engine/plugin revisions to the runner. These are stored in
`provider.resolved_runtime` and participate in the final model hash and uploaded
configuration. Available memory and device ordinals do not affect the hash.
Replicas with different resolved profiles/settings or batch sizes fail before
evaluation; use homogeneous GPUs or evaluate those configurations separately.
The saved harness batching configuration contains the resolved integer size
(including 64 for other providers) and `max_batch_sequences` when expanded
samples count against admission. Stored runtime metadata is refreshed on replay
and is never passed to engine constructors.

Each worker additionally
saves `runtime/sglang-profile-<worker>.json` with hardware, model-shape hash,
selection/fallback reason, applied defaults, effective engine options and resolved
batch limit. Request traces include the provider profile. These records distinguish
automatic decisions across hardware; they do not promise identical random draws
or numerical results across batch sizes. Existing saved integer batch settings
retain their behavior on replay.

### Validation scope

These profiles are starting configurations, not guarantees of optimal throughput
or numerical equivalence across hardware and batch sizes. Requalify changes to
the checkpoint shape, context, runtime or image on representative workloads.
The [provider PR's validation record](https://github.com/allenai/olmo-eval/pull/506)
contains the measured hardware panel and its limitations.

## Concurrency advice

For generic configurations, the harness uses 64 examples per call and SGLang
chooses its effective request limit from its configuration and memory pools.
A requested limit can be reduced by the engine, including by KDA recurrent-state
capacity. Multiple samples expand each example into separate engine requests.

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
