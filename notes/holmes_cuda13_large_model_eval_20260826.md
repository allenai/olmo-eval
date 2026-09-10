# Holmes CUDA 13 large-model evaluation bring-up

Status timestamp: 2026-08-26 16:40 UTC. This note records the operational state of
the B300 bring-up for AIME 2026. The broader research handoff remains in
[`adaptive_compute_handoff_20260824.md`](adaptive_compute_handoff_20260824.md),
and the report plan remains in
[`adaptive_compute_three_phase_plan_20260825.md`](adaptive_compute_three_phase_plan_20260825.md).

## Outcome so far

- `Qwen/Qwen3.6-35B-A3B` has completed a one-problem `aime_2026:pass_at_32`
  smoke and the full 30-problem run on one full Holmes node.
- `zai-org/GLM-5.2` and `deepseek-ai/DeepSeek-V4-Flash-0731` both completed
  one-problem, 32-sample AIME smokes on one TP=8 Holmes node. This validates
  the corrected CUDA 13 / vLLM 0.26 runtime end to end for both architectures.
- Native-K and unnormalized half-K full AIME 2026 runs have now been submitted
  for both DeepSeek and GLM: DeepSeek K=6/K=3 and GLM K=8/K=4. Each experiment
  requests one full eight-B300 node and uses the model-recommended reasoning,
  sampling, and generation limits described below.
- The whole-node `Qwen/Qwen3.5-397B-A17B` one-problem AIME smoke also completed
  successfully. This extends the bring-up to a large member of the Qwen family.
- All active jobs are allocation-backed Beaker jobs in
  `ai2/OLMo-3-moe-experiments`: they request eight GPUs, have a positive
  `minRuntime`, and use `autoResume: true`.
- As of 16:40 UTC, all four new full-run jobs are submitted. The two half-K
  jobs have scheduled onto nodes; the two native-K jobs remain queued in
  Beaker's `created` state. They have not been relaunched or duplicated.

## Current DeepSeek and GLM full AIME 2026 runs

All four runs use `aime_2026:pass_at_32`: the full 30-problem split with 32
samples per problem (960 generations). They share vLLM 0.26.0, PyTorch
2.11.0+cu130, the CUDA 13.0.2 compiler toolkit, seed 0, one TP=8 provider,
`max_num_seqs=8`, and one eight-B300 Holmes node. vLLM profiles the available
KV cache at startup; the explicit concurrency cap prevents an unexpectedly
large initial batch.

Scheduling is identical across the four experiments: urgent priority,
`minRuntime=8h` (the maximum protected allocation interval), `autoResume=true`,
and a 72-hour task timeout. The source snapshot is
`/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/8e1485b8587f4c469748d0f9d0b3ac51f877a6c0477245acfe905da57cfdec7a`.

DeepSeek settings follow the model's non-agentic maximum-reasoning recipe:
temperature 1.0, top-p 1.0, reasoning effort `max`, `deepseek_v4` tokenizer and
reasoning parser, 393,216 output tokens (384 Ki tokens), and a 401,408-token
server context. The server uses TP=8 plus expert parallelism, FP8 KV cache, and
the FP4 indexer cache.

- Native K=6: [01M0ZEW3ATAQMEEMRK7JR21BE4](https://beaker.org/ex/01M0ZEW3ATAQMEEMRK7JR21BE4),
  job `01M0ZEW3NPF8N7BKBFX01XJJ02`.
- Half K=3, unnormalized: [01M0ZEWC6RF25Y5KCYAJ4ZTY56](https://beaker.org/ex/01M0ZEWC6RF25Y5KCYAJ4ZTY56),
  job `01M0ZEWCGSDVAEHQSRYF2T1Z7V`.

GLM settings follow the model's reasoning-task recipe: temperature 1.0, top-p
0.95, thinking enabled, the `glm45` reasoning parser, 163,840 output tokens,
and a 172,032-token server context.

- Native K=8: [01M0ZEWMC5FC23912FP976PV49](https://beaker.org/ex/01M0ZEWMC5FC23912FP976PV49),
  job `01M0ZEWMRCTWZGMPV962RCAYN1`.
- Half K=4, unnormalized: [01M0ZEWWZGP66WTFWP38Y9V6E2](https://beaker.org/ex/01M0ZEWWZGP66WTFWP38Y9V6E2),
  job `01M0ZEWX3FEAGQDJ8Q3C85GTR0`.

For the native baselines, only `num_experts_per_tok` is overridden, so each
checkpoint retains its trained `norm_topk_prob=true` behavior. For the half-K
runs, `norm_topk_prob=false` is explicit while the checkpoint's native routed
scaling factor remains unchanged. Thus “unnormalized” means the selected
expert weights are used without renormalizing their reduced-K sum; it is not a
reference-K denominator rescaling.

## Allocation semantics

The current Beaker API does not express allocation by setting the legacy
`preemptible` field to false. Local Beaker documentation states:

- a job is allocated exactly when `minRuntime > 0`;
- after that protected interval it becomes interruptible;
- `autoResume: true` creates a new execution with a fresh protected interval
  if the job is interrupted; and
- a nonzero `minRuntime` cannot be combined with the deprecated
  `preemptible` field.

The relevant local sources are:

- `/weka/oe-adapt-default/jacobm/math-olmo-test/beaker-docs/content/files/concept/allocations.md`
- `/weka/oe-adapt-default/jacobm/math-olmo-test/beaker-docs/content/files/scheduling/interruption.md`
- `/weka/oe-adapt-default/jacobm/math-olmo-test/beaker-docs/content/files/compute/clusters.md`

The live Qwen full and GLM smoke specs both resolve to:

```yaml
resources:
  gpuCount: 8
context:
  priority: urgent
  minRuntime: 2h
  autoResume: true
```

There is no `preemptible` field in either submitted spec.

## Branch and minimal source changes

- Repository: `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/olmo-eval`
- Branch: `codex/holmes-cuda13-eval`
- Base commit used by the active jobs:
  `e8b88f196767630acaf8a383d12a33d314788d27`
- Qwen full-run source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/6b01b0b285edf6f803495fc424d25543c8fcd3ec7747d2fa7d1c9a05a7222504`
- Corrected GLM smoke source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/fef0465d9f7fda442dfba2e85fd29a7f3af00e067077bac9cf6bf07c33c2845d`
- Packaged-CUDA-path retry source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/2610dac3bd50e77ce7c636fe12dbddfb99e64c394ae4c87645c20a7ee697e8d4`
- CUDA-13.0 compiler-pin retry source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/b07dfa7cce481a31c26bb6b3b6dff856421b0d2762d6413afdee710bf2395da5`
- CUDA-13.0 compiler/linker retry source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/e2ee3b2444e5bf21a0888642a171c78469355d836fc3d4913804682f8cbe0a01`
- Status-safe retry source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/8e1485b8587f4c469748d0f9d0b3ac51f877a6c0477245acfe905da57cfdec7a`

The Holmes-specific source work is deliberately narrow:

1. Admit CUDA 13.0.2 in `scripts/build_config.sh`.
2. Mark `ai2/holmes` as Weka-backed infrastructure.
3. Plumb `--min-runtime` through the launch config and Beaker launcher.
4. When `min_runtime` is set, omit the mutually exclusive deprecated
   `preemptible` field and pass `min_runtime` to Gantry.
5. Add Qwen3.6 and GLM-5.2 launch phases to
   `scripts/adaptive_experts/launch.sh`.
6. Derive the parent inference-worker initialization deadline from the
   provider's server-startup deadline. This prevents a large model configured
   with a long startup allowance from being killed by the runner's shorter
   historical 900-second default.
7. Point CUDA JITs at the CUDA 13 toolkit installed by the provider wheels
   under `site-packages/nvidia/cu13` when `/usr/local/cuda/bin/nvcc` is absent.
   This keeps the base image and model-specific vLLM versions unchanged.
8. For vLLM 0.26.0, reinstall only the CUDA compiler-side toolkit packages
   after vLLM with the `cuda-toolkit==13.0.2` compatibility pins. This prevents
   vLLM's unconstrained dependency from mixing an nvcc 13.3 compiler with
   PyTorch's CUDA 13.0 runtime headers.
9. Add the conventional `CUDA_HOME/lib64 -> lib` and unversioned
   `libcudart.so` aliases expected by PyTorch/FlashInfer JIT link commands.
   NVIDIA's CUDA 13 Python wheels provide only `lib/` and the versioned
   `libcudart.so.13` runtime object.
10. Make Beaker workload-description updates strictly best-effort. A transient
    control-plane outage previously blocked a model-startup callback for about
    16 minutes and then killed both healthy inference workers when the status
    RPC raised `Stream removed (recvmsg: Network is unreachable)`.

The worktree also contains substantial pre-existing adaptive-compute changes.
Do not stage or discard unrelated files when committing this bring-up.

Verification after the current runner/launcher edits:

```text
148 launcher, config, runner, provider-startup, Beaker-status, and CUDA-hook tests passed in 2.84s
bash -n scripts/adaptive_experts/launch.sh: passed
ruff and git diff --check for the scoped source/test files: passed
local `c++ ... -L$CUDA_HOME/lib64 -lcudart` reproduction: passed
```

## Runtime image and packages

Shared image:

- Beaker image name: `jacobm/olmo-eval-cu1302-trc2100-amd64`
- Beaker image ID: `01M0XWPT962A82AJXY6HWRWAZ0`
- Digest:
  `sha256:f530ec6f14456e6dcc46a5f2287713392cfcc2a4aeda4246acb5c4719f9e33e2`
- Base stack: PyTorch 2.10.0 + CUDA 13.0 userspace

Qwen keeps the project's vLLM version and installs the official CUDA 13 wheel:

- vLLM `0.19.1+cu130`
- PyTorch `2.10.0+cu130`

GLM-5.2 is not supported by vLLM 0.19.1. The exact BF16 checkpoint requires
the newer native implementation, so the GLM job reuses the same image but
installs these packages at runtime:

- vLLM `0.26.0`
- PyTorch `2.11.0+cu130`

The first two GLM smokes used vLLM 0.23.0. That version recognizes GLM-5.2
but predates vLLM PR 45895, which explicitly fixes BF16 indexer initialization
on the model's shared/skip-Top-K layers. vLLM 0.26.0 retains the same Torch
2.11 dependency while including the upstream fix.

Qwen3.5-397B-A17B keeps the same vLLM 0.19.1+cu130 / PyTorch 2.10.0+cu130
path as Qwen3.6. The official vLLM recipe lists v0.17.0 as its minimum. The
DeepSeek-V4-Flash-0731 checkpoint requires vLLM 0.25.0 or newer, so its smoke
uses the same vLLM 0.26.0 / PyTorch 2.11.0+cu130 runtime as the corrected GLM
smoke.

The unmodified vLLM 0.26.0 dependency set installs
`nvidia-cuda-nvcc==13.3.73`, but the compiler is at
`site-packages/nvidia/cu13/bin/nvcc`, not `/usr/local/cuda/bin/nvcc`. Both
FlashInfer and DeepGEMM compile kernels during startup. The opt-in vLLM startup
hook now sets `CUDA_HOME`, `CUDACXX`, and `PATH` to this packaged CUDA root
whenever the system CUDA root has no `nvcc`. A second install step then pins
`cudart`, `nvcc`, `nvvm`, `cccl`, and `crt` through
`cuda-toolkit[...] == 13.0.2`, yielding nvcc 13.0.88, CCCL 13.0.85, and runtime
13.0.96. Local header-preprocessing and `-lcudart` linker smokes passed with
this exact combination.

## Qwen3.6 AIME 2026

Configuration:

- Model: `Qwen/Qwen3.6-35B-A3B`
- Routing: native nested `text_config.num_experts_per_tok=8`
- Topology: eight independent TP=1 vLLM servers, one per B300
- Context: 40,960 tokens
- Per-server concurrency cap: 16 sequences
- Task: `aime_2026:pass_at_32`, temperature 1.0, top-p 0.95, top-k 20,
  seed 0, thinking enabled

Successful smoke:

- Experiment: [01M0XZ8EK4717JJJANJ7A9XSTZ](https://beaker.org/ex/01M0XZ8EK4717JJJANJ7A9XSTZ)
- Job: `01M0XZ8EQ5JWDPK1WHFS7DQY8W`
- Result dataset: `01M0XZ8EKANJHT4AYWTCS9XC62`
- Exit: 0
- One problem with 32 samples completed without serving or scoring errors.
  Its score was 0; this is only a protocol smoke, not a model-quality estimate.

Successful full run:

- Experiment: [01M0Y116BRRY7AW2FNV524KZS7](https://beaker.org/ex/01M0Y116BRRY7AW2FNV524KZS7)
- Job: `01M0Y116FX0XHBYGGN4EYJ1XHB`
- Result dataset: `01M0Y116BW63X3SREVN3BKT0H6`
- Scheduling: allocated, 2h minimum runtime, auto-resume
- Node: `holmes-cs-aus-524.reviz.ai2.in`
- Exit: 0
- All eight servers became ready in about 252 seconds. Six workers received
  the 30 problem shards; two empty workers shut down cleanly.
- End-to-end duration: 8,089 seconds (2h 14m 49s); task duration: 7,819
  seconds.
- Metrics: sample accuracy 0.8333, estimated pass@1 0.5990, pass@4 0.7020,
  pass@8 0.7445, pass@16 0.7894, and pass@32 0.8333.
- Output tokens across the 960 samples: 26,990,127 total; mean 28,115;
  median 30,120; p90/p95 32,768. 386 samples (40.2%) hit the configured
  32,768-token generation cap. These counts include the model's reasoning
  tokens and explain the long wall time.

The eight replicas are intentional: the model fits on one B300, and data
parallel serving makes use of the whole requested node. The per-server
`max_num_seqs=16` is conservative; vLLM chooses the KV-cache capacity during
startup. The full run exposed an independent queue-granularity issue:
with the runner's default 64-item chunk, six workers claimed shards of
13/5/5/3/2/2 problems and two received none. The launcher now sets
`batching.chunk_size=4`, matching `ceil(30 / 8)`, so future AIME runs cannot let
one worker drain most of the shared queue. The completed run was left intact
rather than restarting after the imbalance was discovered, because most
generation work had already finished.

## GLM-5.2 AIME 2026

Configuration:

- Model: exact `zai-org/GLM-5.2` BF16 checkpoint
- Routing: native `num_experts_per_tok=8`
- Topology: one TP=8 vLLM server spanning the entire B300 node
- Context: 40,960 tokens
- Concurrency cap: 8 sequences
- Task: `aime_2026:pass_at_32`, temperature 1.0, top-p 0.95, seed 0,
  thinking enabled

First smoke (diagnostic failure):

- Experiment: [01M0Y1AENYTX44J2XGHQ9F8CBC](https://beaker.org/ex/01M0Y1AENYTX44J2XGHQ9F8CBC)
- Job: `01M0Y1AEVZSVJN2RTQN440QFYY`
- Result dataset: `01M0Y1AEP36S1CPTZTYDCJY6A4`
- Scheduling: allocated, 2h minimum runtime, auto-resume
- Node: `holmes-cs-aus-528.reviz.ai2.in`
- Exit: 1 after 900 seconds of provider initialization. The architecture was
  recognized, all eight TP workers remained alive, and approximately 184 GiB
  was resident on each GPU. The failure was not vLLM's 7,200-second startup
  timeout: olmo-eval's parent runner independently stopped waiting after its
  historical 900-second default.

Second smoke (timeout correction, diagnostic failure):

- Experiment: [01M0Y2PXK0T7W60691PD4PGFR3](https://beaker.org/ex/01M0Y2PXK0T7W60691PD4PGFR3)
- Job: `01M0Y2PXQKAVPB45W5RR5SNV72`
- Result dataset: `01M0Y2PXKB8S8VRR1Q0TFJ2KXZ`
- Scheduling: allocated, 2h minimum runtime, auto-resume
- Source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/fef0465d9f7fda442dfba2e85fd29a7f3af00e067077bac9cf6bf07c33c2845d`
- Parent worker deadline: 7,260 seconds, derived automatically from the
  provider's 7,200-second server deadline plus a 60-second buffer.
- Exit: 1 after approximately 40 minutes of successful checkpoint loading.
  vLLM 0.23.0 instantiated an indexer on every layer and then reported missing
  `self_attn.indexer.k_norm` weights. The missing set exactly matched the
  official checkpoint's shared-indexer layers: checkpoint-owned indexers are
  present on layers 0, 1, 2, and every fourth layer from 6 through 78. This is
  the failure addressed by upstream vLLM PR 45895, not a corrupt or incomplete
  checkpoint.

Third smoke (shared-indexer fix, diagnostic failure):

- Experiment: [01M0Y5H502TF6T6HWD8BWRREMY](https://beaker.org/ex/01M0Y5H502TF6T6HWD8BWRREMY)
- Job: `01M0Y5H53CKH53ZE2K3DDK67X0`
- Result dataset: `01M0Y5H507RES274CGMR3PV106`
- Scheduling: allocated, 2h minimum runtime, auto-resume
- Node: `holmes-cs-aus-532.reviz.ai2.in`
- Runtime: vLLM 0.26.0, PyTorch 2.11.0+cu130
- Source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/fef0465d9f7fda442dfba2e85fd29a7f3af00e067077bac9cf6bf07c33c2845d`
- Exit: 1 after approximately 39 minutes. vLLM 0.26.0 passed the prior
  shared-indexer failure and loaded the BF16 checkpoint, then FlashInfer's
  sampling-kernel JIT attempted `/usr/local/cuda/bin/nvcc`. The CUDA compiler
  was installed in the Python environment but was not exposed at that system
  path.

Fourth smoke (canceled after DeepSeek exposed a shared version mismatch):

- Experiment: [01M0Y8STR984NJ5KQN7R5HSTTN](https://beaker.org/ex/01M0Y8STR984NJ5KQN7R5HSTTN)
- Job: `01M0Y8STX5A0X6DX2G3028WM8H`
- Result dataset: `01M0Y8STRGXT0MTSRCZW17T2WJ`
- Scheduling: allocated, 2h minimum runtime, auto-resume
- Runtime: vLLM 0.26.0, PyTorch 2.11.0+cu130
- Source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/2610dac3bd50e77ce7c636fe12dbddfb99e64c394ae4c87645c20a7ee697e8d4`
- Correction: the isolated-vLLM startup hook exposes the packaged CUDA 13
  compiler root to FlashInfer and PyTorch extension JITs.
- State: canceled after 10m 34s, before checkpoint loading completed. The
  concurrently running DeepSeek smoke proved that vLLM 0.26.0 had selected an
  nvcc 13.3 compiler against CUDA 13.0 headers; GLM would have reached the same
  mismatch at its later FlashInfer JIT stage.

Fifth smoke (install-forwarding diagnostic, canceled):

- Experiment: [01M0Y9KCY90EW0WEZC7FE3Q53R](https://beaker.org/ex/01M0Y9KCY90EW0WEZC7FE3Q53R)
- Job: `01M0Y9KD1TPGWX657A5Y6PXR9E`
- Result dataset: `01M0Y9KCYFBX5DKCFZ00M96VRN`
- Scheduling: allocated, 2h minimum runtime, auto-resume
- Source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/837f38890790a8b829ef159fc081bbe6cce560384e37e0e51f17f25c428c45b4`
- State: canceled after 2m 29s. The compiler pin was present in the Beaker task
  environment but omitted from Gantry's install-time environment allowlist,
  so the install command did not execute the pin.

Sixth smoke (coherent CUDA 13.0 install, proactively canceled):

- Experiment: [01M0Y9Z25D1EYV0WRK3C19FMSN](https://beaker.org/ex/01M0Y9Z25D1EYV0WRK3C19FMSN)
- Job: `01M0Y9Z29BY1AJAAKG88EFAM1N`
- Result dataset: `01M0Y9Z25TKMRGGS94VFXVBGXP`
- Scheduling: allocated, 2h minimum runtime, auto-resume
- Source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/b07dfa7cce481a31c26bb6b3b6dff856421b0d2762d6413afdee710bf2395da5`
- Install-log verification: nvcc 13.3.73 -> 13.0.88, CCCL 13.3.3.4.1 ->
  13.0.85, CRT 13.3.73 -> 13.0.88, and NVVM 13.3.73 -> 13.0.88.
- State: canceled before the roughly 40-minute checkpoint load completed.
  The concurrent DeepSeek retry showed that the coherent compiler reached
  FlashInfer but its final link used `CUDA_HOME/lib64` and could not find
  `-lcudart`; this GLM run would have reached the same failure later.

Seventh smoke (compiler and linker corrected; external status-RPC failure):

- Experiment: [01M0YATWT1C1CH7S49Z9FNXKEY](https://beaker.org/ex/01M0YATWT1C1CH7S49Z9FNXKEY)
- Job: `01M0YATWXFQWC29337ERF6CY9K`
- Result dataset: `01M0YATWT7PDF859VM3FDZZCZX`
- Scheduling: allocated, 2h minimum runtime, auto-resume
- Source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/e2ee3b2444e5bf21a0888642a171c78469355d836fc3d4913804682f8cbe0a01`
- Exit: 1. The model subprocess continued its expected BF16 shard load, but
  the parent inference worker blocked in a nonessential Beaker description
  update and later crashed when that RPC raised `Stream removed (recvmsg:
  Network is unreachable)`. DeepSeek failed identically at the same time,
  confirming a Beaker control-plane incident rather than a model failure.

Eighth smoke (successful, status updates made nonfatal):

- Experiment: [01M0YCRY8EVFNTY6DY3BMNKT2K](https://beaker.org/ex/01M0YCRY8EVFNTY6DY3BMNKT2K)
- Job: `01M0YCRYJPW7AYCR0J4W1BS5QE`
- Result dataset: `01M0YCRY8MP3B14H0SGKT8JATE`
- Scheduling: allocated, 2h minimum runtime, auto-resume
- Source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/8e1485b8587f4c469748d0f9d0b3ac51f877a6c0477245acfe905da57cfdec7a`
- Node: `holmes-cs-aus-490.reviz.ai2.in`
- Exit: 0. The exact BF16 checkpoint initialized successfully in 3,954.96
  seconds, and the one-problem, 32-sample task completed in 1,902 seconds.
- Sample accuracy / estimated pass@1: 0.59375 (19/32 correct). This is only a
  serving smoke on one problem, not a model-quality estimate.
- Output tokens: 988,197 total. Thirteen of 32 generations reached the former
  32,768-token cap, motivating the recommended 163,840-token output limit used
  for the full runs above.
- Beaker status initialization and update failures now self-disable the
  reporter and cannot abort the eval.

The TP=8 layout follows the model's serving requirement and is the natural
whole-node layout for a roughly 1.5 TB BF16 checkpoint. `max_num_seqs=8` is a
conservative first value; vLLM still auto-profiles the available KV cache.

The native and unnormalized half-K full runs are listed in the current-run
section above. The launcher phase is `glm52-aime2026`; exact escaped commands
and source provenance are recorded in `notes/beaker_jobs.jsonl`.

## Additional large-model smokes

### Qwen3.5-397B-A17B

- Model: exact `Qwen/Qwen3.5-397B-A17B` BF16 checkpoint
- Scale: 397B total / 17B active, approximately 807 GB of safetensor shards
- Routing: native nested `text_config.num_experts_per_tok=10`
- Topology: one text-only TP=8 vLLM server spanning one B300 node
- Runtime: vLLM 0.19.1+cu130, PyTorch 2.10.0+cu130
- Context / concurrency: 40,960 tokens / 8 sequences
- Sampling: temperature 0.6, top-p 0.95, top-k 20; 32 samples
- Experiment: [01M0Y7N7T1F00P9Z78A8ZRSJ3K](https://beaker.org/ex/01M0Y7N7T1F00P9Z78A8ZRSJ3K)
- Job: `01M0Y7N7XKHKMNTESWE5GGYJTN`
- Result dataset: `01M0Y7N7T8RZGZREY00BS2S7QN`
- Node: `holmes-cs-aus-512.reviz.ai2.in`
- Scheduling: allocated, 2h minimum runtime, auto-resume
- Exit: 0. vLLM 0.19.1 loaded the exact 807 GB checkpoint, became ready in
  2,610 seconds (43m 30s), and completed the one-problem, 32-sample smoke.
- End-to-end duration: 4,242 seconds (70m 42s); task generation/scoring
  duration: 1,625 seconds (27m 5s).
- This problem had 24/32 correct samples: sample accuracy/pass@1 0.75,
  pass@4 0.9981, pass@8 approximately 1.0, and pass@16/pass@32 1.0. This is a
  serving smoke on one item, not a model-quality estimate.
- Output tokens across the 32 samples: 890,069 total; mean 27,815; median
  27,564; minimum 22,413; p90/p95/maximum 32,768. Six samples (18.75%) hit
  the 32,768-token cap.

### DeepSeek-V4-Flash-0731

- Model: exact `deepseek-ai/DeepSeek-V4-Flash-0731` mixed-FP8 checkpoint
- Scale: 304B total, approximately 167 GB of safetensor shards
- Routing: native `num_experts_per_tok=6`
- Topology: one TP=8 + expert-parallel vLLM server spanning one B300 node
- Runtime: vLLM 0.26.0, PyTorch 2.11.0+cu130
- Context / concurrency: 40,960 tokens / 8 sequences
- Serving compatibility: FP8 KV cache, block size 256, DeepSeek-V4 tokenizer
  and reasoning parser, FP4 indexer cache. DSpark is disabled for the first
  smoke so this run isolates base-model serving.
- Sampling: temperature 1.0, top-p 0.95; 32 samples, Think High
- Experiment: [01M0Y7NP5WYXVA9SF00C9C41XD](https://beaker.org/ex/01M0Y7NP5WYXVA9SF00C9C41XD)
- Job: `01M0Y7NPAND5G8J53FQ00FK8E5`
- Result dataset: `01M0Y7NP60YQSXRRYYWRERM297`
- Node: `holmes-cs-aus-551.reviz.ai2.in`
- Scheduling: allocated, 2h minimum runtime, auto-resume
- Exit: 1 during DeepGEMM FP8 weight post-processing. Like GLM's FlashInfer
  failure, DeepGEMM required an `nvcc` path and did not discover the compiler
  installed under the Python environment.
- Retry experiment: [01M0Y8SEHSX51MRZ667KTVSHDK](https://beaker.org/ex/01M0Y8SEHSX51MRZ667KTVSHDK)
- Retry job: `01M0Y8SEPJBQN1XWRJA1DBM1F7`
- Retry result dataset: `01M0Y8SEHY80FGWN1XCQWD90GA`
- Retry source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/2610dac3bd50e77ce7c636fe12dbddfb99e64c394ae4c87645c20a7ee697e8d4`
- Retry scheduling: allocated, 2h minimum runtime, auto-resume
- Retry exit: 1 after the model loaded. DeepGEMM found `nvcc`, confirming the
  path correction, but compilation then rejected the mixed nvcc 13.3 / CUDA
  13.0 header set.
- CUDA-pin retry experiment: [01M0Y9JYN8BBBPX4851WE3YBGQ](https://beaker.org/ex/01M0Y9JYN8BBBPX4851WE3YBGQ)
- CUDA-pin retry job: `01M0Y9JZ3KKX5J3N5SETQJJQ57`
- CUDA-pin retry result dataset: `01M0Y9JYNHETT6XTGGT7CDD2SG`
- CUDA-pin retry scheduling: allocated, 2h minimum runtime, auto-resume
- CUDA-pin retry source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/837f38890790a8b829ef159fc081bbe6cce560384e37e0e51f17f25c428c45b4`
- CUDA-pin retry state: canceled after 3m because the install-time allowlist
  omission above meant the pin had not actually run.
- Verified-pin experiment: [01M0Y9YM4NYH7C1004PXX0BD7N](https://beaker.org/ex/01M0Y9YM4NYH7C1004PXX0BD7N)
- Verified-pin job: `01M0Y9YMVJJZ69EDMT0MZ1KDT6`
- Verified-pin result dataset: `01M0Y9YM4TFYGN48EF5Q0J7T6C`
- Verified-pin scheduling: allocated, 2h minimum runtime, auto-resume
- Verified-pin source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/b07dfa7cce481a31c26bb6b3b6dff856421b0d2762d6413afdee710bf2395da5`
- Install-log verification matches GLM's exact CUDA 13.0 downgrades.
- Verified-pin exit: 1 after CUDA compilation succeeded. FlashInfer's final
  link assumed `CUDA_HOME/lib64` and failed to find `-lcudart`; the CUDA 13
  wheel actually installs `lib/libcudart.so.13` without an unversioned linker
  name.
- Compiler/linker retry experiment:
  [01M0YATFN9ESA6D79J825B0VYN](https://beaker.org/ex/01M0YATFN9ESA6D79J825B0VYN)
- Compiler/linker retry job: `01M0YATFRPP89QZTBJY0MYRXSZ`
- Compiler/linker retry result dataset: `01M0YATFNE002KQ764TJ86XGZ8`
- Compiler/linker retry source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/e2ee3b2444e5bf21a0888642a171c78469355d836fc3d4913804682f8cbe0a01`
- Compiler/linker retry scheduling: allocated, 2h minimum runtime,
  auto-resume.
- Compiler/linker retry exit: 1 after all CUDA compiler/linker stages passed.
  The model proceeded into TileLang compilation and Blackwell cubin downloads,
  but the parent inference worker was killed by the same nonessential Beaker
  status-RPC failure as GLM. There was no model-server traceback before the
  parent stopped it.
- Status-safe retry experiment:
  [01M0YCRFB1A8HY81MP6BXAH22V](https://beaker.org/ex/01M0YCRFB1A8HY81MP6BXAH22V)
- Status-safe retry job: `01M0YCRFGCE0J0TSARX6QYY83J`
- Status-safe retry result dataset: `01M0YCRFBAZT8NX26YNV1KZJ5W`
- Status-safe retry source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/8e1485b8587f4c469748d0f9d0b3ac51f877a6c0477245acfe905da57cfdec7a`
- Status-safe retry scheduling: allocated, 2h minimum runtime, auto-resume;
  ran on `holmes-cs-aus-519.reviz.ai2.in`.
- Exit: 0. Provider initialization completed in 3,207.66 seconds after all
  compiler/linker stages, TileLang kernels, the 1,254-case DeepGEMM warmup,
  and FlashInfer FP4 MoE autotuning succeeded. The one-problem, 32-sample task
  then completed in 675 seconds.
- All 32 samples were correct. They emitted 583,304 output tokens in total.
  This is a serving smoke on one problem, not a model-quality estimate.
- The native and unnormalized half-K full runs are listed in the current-run
  section above.

## GLM-5.2 half-K NVRTC fix smoke

- Experiment: [01M0ZRQ60TP5Z1502SET6E1TYM](https://beaker.org/ex/01M0ZRQ60TP5Z1502SET6E1TYM)
- Job: `01M0ZRQ64JWYK53KY919T0GPB3`
- Result dataset: `01M0ZRQ6189K9G43V9KFA9JQ0A`
- Source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/2002995bb1dd7bf2506c0b4ef08e8bb6ccdd57b06901b2e2f2cdab3353dc22df`
- Configuration: vLLM 0.26.0, PyTorch 2.11.0+cu130, TP=8, K=4,
  `norm_topk_prob=false`, one AIME 2026 problem with 32 samples and a
  deliberately short 2,048-token output cap.
- Environment fix: explicitly include the CUDA 13.0 NVRTC toolkit component
  and create `libnvrtc.so -> libnvrtc.so.13` alongside the existing
  `libcudart.so` linker alias. The install resolved nvcc and NVRTC to 13.0.88.
- Exit: 0. Provider initialization completed in 3,272.90 seconds, including
  the K=4 FlashInfer fused-MoE JIT that previously failed at `-lnvrtc`. The
  32-sample request completed in 126.6 seconds and the server shut down
  normally. This confirms the CUDA/NVRTC environment issue is fixed.
- Quality caveat: 31/32 samples reached the artificial 2,048-token cap, but
  all 32 outputs were already severely incoherent, unlike the coherent K=8
  smoke outputs. The native checkpoint uses `norm_topk_prob=true`, sigmoid
  routing, and `routed_scaling_factor=2.5`; forcing unnormalized routing is
  therefore a likely cause. Do not launch the full unnormalized K=4 eval
  without first testing K=4 with native normalization.

## GLM-5.2 native-normalized K=4 smoke

- Experiment: [01M101XB5JW153YEDYG897RXBH](https://beaker.org/ex/01M101XB5JW153YEDYG897RXBH)
- Job: `01M101XB9M34P5F5A6DKCJYQNF`
- Result dataset: `01M101XB5RVMMKYGZ804NVZQ3G`
- Configuration: exact match to the successful K=4 NVRTC-fix smoke except
  `norm_topk_prob=true`. It uses vLLM 0.26.0, PyTorch 2.11.0+cu130,
  TP=8 on one full Holmes node, K=4, one AIME 2026 problem with 32 samples,
  temperature 1.0, top-p 0.95, the `glm45` reasoning parser, thinking enabled,
  and a 2,048-token diagnostic output cap.
- Source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/2002995bb1dd7bf2506c0b4ef08e8bb6ccdd57b06901b2e2f2cdab3353dc22df`
- Exit: 0. The provider initialized in 4,775.76 seconds and the one-problem
  evaluation completed in 104.97 seconds. All 32 generations reached the
  deliberately short 2,048-token cap while still in hidden reasoning, so the
  reasoning parser saved empty visible answers and the resulting score of zero
  is not a capability measurement.
- Normalization signal: mean generated-token log probability was `-0.419`,
  much closer to the coherent native K=8 smoke (`-0.261`) than to the corrupted
  raw-sigmoid K=4 smoke (`-4.262`). The raw run also emitted malformed visible
  text, whereas all normalized generations remained inside the reasoning
  channel through the cap. This is strong evidence that native normalization
  removes the raw-sigmoid degeneration, but the saved artifacts do not retain
  hidden reasoning text for direct inspection.
- Gate conclusion: infrastructure and token-distribution behavior pass. A
  longer normalized K=4 run is still required to measure answer quality; do not
  interpret the zero score from this capped smoke as degradation.

## GLM-5.2 native-normalized K=4 full-generation smoke

- Experiment: [01M1082NMW7NGKK13HHB0KEC4S](https://beaker.org/ex/01M1082NMW7NGKK13HHB0KEC4S)
- Job: `01M1082NRCA6ZGYAZ1RBMC9TQ6`
- Result dataset: `01M1082NN2T94S5CZHVANSMTMW`
- Configuration: one AIME 2026 problem with 32 samples, K=4,
  `norm_topk_prob=true`, TP=8 on one full Holmes node, vLLM 0.26.0,
  PyTorch 2.11.0+cu130, temperature 1.0, top-p 0.95, thinking enabled,
  the `glm45` reasoning parser, 163,840 output tokens, and 172,032 model
  context. This matches the recommended full GLM generation recipe while
  retaining the smoke's one-problem limit.
- Scheduling: urgent, allocated with an 8-hour minimum runtime, auto-resume,
  and a 72-hour timeout.
- Source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/2002995bb1dd7bf2506c0b4ef08e8bb6ccdd57b06901b2e2f2cdab3353dc22df`
- Exit: 0. Provider initialization took 4,050.90 seconds and generation plus
  scoring took 1,543.36 seconds.
- All 32 generations were nonempty and coherent, all reached the correct
  answer 107, and none hit the output cap. Output lengths ranged from 21,485
  to 69,173 tokens, with a mean of 28,655 and 916,966 tokens total. Mean
  generated-token log probability was `-0.333`.
- The automated score was 23/32 (`pass@1=0.71875`) because exactly 23 outputs
  used `\\boxed{107}`. The other nine also state 107 but use answer formatting
  that the Minerva extractor misreads; one has a stray trailing token artifact
  after an otherwise coherent correct solution. This is an extraction issue,
  not evidence of K=4 reasoning failure. The smoke passed promotion.

## GLM-5.2 native-normalized K=4 full AIME 2026

- Experiment: [01M10F01D8GJJ2SG2KWCPH0YGX](https://beaker.org/ex/01M10F01D8GJJ2SG2KWCPH0YGX)
- Job: `01M10F01HY1T92X2VRD4FXTR71`
- Result dataset: `01M10F01DFVVCZXRWZ5ECY0TZW`
- Configuration: complete 30-problem AIME 2026 evaluation with 32 samples per
  problem. It is identical to the passing full-generation smoke except that
  the one-problem limit is removed: normalized K=4, TP=8, one full Holmes
  node, temperature 1.0, top-p 0.95, thinking enabled, `glm45` parser,
  163,840 output tokens, and 172,032 model context.
- Scheduling: urgent, allocated with an 8-hour minimum runtime, auto-resume,
  and a 72-hour timeout.
- Source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/2002995bb1dd7bf2506c0b4ef08e8bb6ccdd57b06901b2e2f2cdab3353dc22df`

## GLM-5.2 native-reference-truncated K=4 full AIME 2026

- Experiment: [01M10H7055DX973Z9JSMP9FJ34](https://beaker.org/ex/01M10H7055DX973Z9JSMP9FJ34)
- Job: `01M10H70BHEZCSDJ8NW9NG2ZYT`
- Result dataset: `01M10H705FC5KHQ9MS35WZZ50P`
- Configuration: complete 30-problem AIME 2026 evaluation with 32 samples per
  problem. It matches the native-normalized K=4 full run's one-node Holmes
  configuration, vLLM 0.26.0 environment, generation settings, and parser.
- Routing intervention: the fused MoE kernel remains K=4, but the compatibility
  hook computes native sigmoid/noaux top-8 routing weights for the reference
  denominator and native top-4 routing separately for the retained expert IDs.
  The retained weights are `sigmoid_score_top4 / sum(sigmoid_score_top8)`;
  they are not renormalized over the surviving four experts. GLM's existing
  output-side routed scaling factor of 2.5 is then applied exactly once. This
  is the native-reference truncation analogue of an unnormalized K reduction,
  while preserving the checkpoint's native sigmoid-routing scale.
- The separate top-4 selection is intentional: vLLM may produce unsorted top-k
  tensors, so slicing the first four entries from its top-8 output would not
  guarantee retaining the four highest native selections.
- Scheduling: urgent and preemptible on one full eight-GPU Holmes node, with an
  8-hour minimum runtime, auto-resume, and a 72-hour timeout. It was scheduled
  immediately after submission.
- Source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/27801b80dd73e3c2a24eaed29c3efde4cb773e31cd88dcb54ebe98b803f78281`
- Local validation: all 41 compatibility tests passed, including exact checks
  for the top-8 denominator, independently selected top-4 IDs, dropped routing
  mass, and one-time scaling. Runtime validation still needs the startup log's
  `GLM native-reference truncation matched a router` marker after model
  construction begins.

## GLM-5.2 seed-0 quality and length analysis

Analysis completed 2026-08-31 against the three completed 30-problem,
32-sample seed-0 runs: native K=8, native-normalized K=4, and
native-reference-truncated K=4. Each run contains 960 generations.

Full-set AIME sample accuracy is 807/960 (84.06%) for native K=8, 713/960
(74.27%) for normalized K=4, and 742/960 (77.29%) for reference-truncated
K=4. Corresponding pass@32 is 29/30 (96.67%), 28/30 (93.33%), and 28/30
(93.33%). Reference truncation is 3.02 percentage points better than native
normalization at K=4.

The accuracy loss is not an artifact of parser failures. Pairing records by
`(AIME problem, sample index)` and retaining only the 948 generations for
which every setting has a nonempty, normally parsed visible output gives
84.18% native-K=8 accuracy, 74.79% normalized-K=4 accuracy, and 77.85%
reference-K=4 accuracy. On the stricter 810-generation subset where all three
settings also have a nonempty extracted answer, accuracy is 90.86%, 77.41%,
and 80.00%, respectively.

The loss is broad but task-dependent. Normalized K=4 is worse than native K=8
on 21/30 problems, equal on two, and better on seven. Reference K=4 is worse
on 19/30, equal on three, and better on eight. The largest losses occur on
problems 29 (25/32 native versus 3/32 normalized and 6/32 reference), 10
(24/32 versus 9/32 and 9/32), and 17 (32/32 versus 21/32 and 22/32).
Problem-cluster bootstrap uncertainty across only 30 tasks is still material:
the normalized-K=4 minus native-K=8 full-set difference is -9.79 points with
a 95% bootstrap interval of [-16.67, -3.54], while the reference-K=4 minus
native difference is -6.77 points with interval [-14.06, +0.52]. Additional
seeds are required for the final inference.

The visible outputs do not show a broad language-coherence collapse. Most K=4
solutions remain fluent and well structured, and bulk four-gram/line
repetition distributions are similar to native K=8. Manual review of
discordant examples instead finds plausible-looking but incorrect derivations,
unsupported algebraic steps, and confident invented simplifications: this is
primarily mathematical-reasoning degradation with preserved surface fluency.
Rare tail failures do increase. Native K=8 has no empty or grossly leaked
outputs; each K=4 run has three empty visible outputs and three obvious
reasoning-parser leaks, with no exact problem/sample overlap between the two
K=4 failure sets. A small number of additional K=4 outputs contain severe
repetition or control-tag artifacts.

Exact GLM-tokenizer accounting on the common extracted-answer subset gives
mean total/reasoning/visible-token counts of 20,245/19,140/1,105 for native
K=8, 19,737/18,770/967 for normalized K=4, and 19,467/18,509/958 for reference
K=4. Thus K=4 visible solutions are about 12--13% shorter, while total and
estimated hidden-reasoning length differences are smaller and sensitive to
the long-tailed distribution. The saved artifacts do not retain the hidden
`reasoning_content`; estimated reasoning tokens are total generated tokens
minus exactly tokenized visible output, so hidden-chain coherence cannot be
audited directly.

## GLM-5.2 AIME replicate runs

Status checked 2026-08-31 21:54 UTC. Two additional native-K=8 seeds and two
additional reference-truncated-K=4 seeds are active in experiment group
`holmes-glm52-aime2026-replicates-20260831`. Every run has a distinct allocated
8xB300 Holmes node, an eight-hour protected minimum runtime, auto-resume, and a
72-hour task timeout. All four vLLM servers completed the long GLM startup and
each worker entered the single 30-problem AIME batch without an error; no
relaunch is needed.

| Policy | Seed | Experiment | Job | Result dataset | Live gate |
|---|---:|---|---|---|---|
| Native K=8 | 1 | [01M1CGXRK99NDGD59P6SNM2TF2](https://beaker.org/ex/01M1CGXRK99NDGD59P6SNM2TF2) | `01M1CGXRQ9ZHFZV5PV6YFND1YH` | `01M1CGXRKSN0AAD3MZAKSP258G` | server ready; 30-item batch started |
| Native K=8 | 2 | [01M1CGYAJ5CA2M32PB6JBRWWF9](https://beaker.org/ex/01M1CGYAJ5CA2M32PB6JBRWWF9) | `01M1CGYANTZXWDBMWJJ135KS11` | `01M1CGYAJEAR8NRFH7WT0WXFS1` | server ready; 30-item batch started |
| Reference K=4 | 1 | [01M1CGYNEGP61W1RQBCYCGSVKD](https://beaker.org/ex/01M1CGYNEGP61W1RQBCYCGSVKD) | `01M1CGYNK169YNSG6JX0XXP4NW` | `01M1CGYNETZ4CJ7ZEEF95BGWG9` | server ready; 30-item batch started |
| Reference K=4 | 2 | [01M1CGYXXYSGAVNBS0NMR4NT1B](https://beaker.org/ex/01M1CGYXXYSGAVNBS0NMR4NT1B) | `01M1CGYY1MJBYR5872MFWTW27X` | `01M1CGYXY7JZR7ZE9DAPJGY9FP` | server ready; 30-item batch started |

After completion, combine seeds 0/1/2 for native K=8 and reference K=4;
report per-seed and pooled pass@1/pass@32, between-seed variance, paired
problem-level differences, output-length distributions, extraction/parser
failure rates, repetition/coherence outliers, and problem-specific effects.
The normalized K=4 condition currently has only seed 0 and should remain a
secondary routing-semantics comparison unless additional replicates are
explicitly requested.

Completion update (2026-09-02): all four repeat jobs exited successfully with
complete 30-problem, 960-generation artifacts. The three-seed score, length,
failure-tail, and qualitative-output analysis is in
[`glm52_aime_three_seed_analysis_20260902.md`](glm52_aime_three_seed_analysis_20260902.md).
No replicate requires a relaunch.

## Qwen3.5-397B long-context startup diagnostic

Status checked 2026-09-03 04:45 UTC. The earlier vLLM 0.19.1 40,960-context
smoke remains the proven reference recipe and has not been modified. A single
opt-in diagnostic is running to isolate the 90,112-context startup failure on
the newer stack:

- Experiment: [01M1JRWNAYDP7TAXHVEBYTKG2A](https://beaker.org/ex/01M1JRWNAYDP7TAXHVEBYTKG2A)
- Job: `01M1JRWNFF09GWB288R0XXCNHT`
- Result dataset: `01M1JRWNBHK1G9BEQ4C6CZDHEE`
- Node: `holmes-cs-aus-485.reviz.ai2.in` (one allocated 8xB300 node)
- Model/routing: exact BF16 `Qwen/Qwen3.5-397B-A17B`, native K=10
- Runtime: vLLM 0.26.0, PyTorch 2.11.0+cu130, consistently pinned CUDA 13.0.2
- Conservative startup settings: 90,112 context, one sequence, 2,096 batched
  tokens, 0.8 GPU-memory utilization, prefix caching disabled, eager mode,
  Triton attention/GDN prefill, DeepGEMM disabled
- Probe: one AIME problem, one 16-token generation; this tests server startup,
  not quality
- Scheduling: urgent, eight-hour protected minimum runtime, auto-resume,
  24-hour task timeout, 7,200-second server startup timeout

The environment install completed cleanly and all eight B300s were visible.
At the status timestamp, the engine was alive in checkpoint initialization
with no package, CUDA, architecture, or flag error. The preserved server log
had reached `EngineCore` worker initialization. The successful 40,960-context
reference run was likewise silent for approximately 38 minutes after this
line while loading the 807 GB checkpoint, so the current early silence is
expected rather than evidence of a hang. The first decisive gate is progress
after that reference loading interval.

The launcher exposes this only through `QWEN35_397B_CONSERVATIVE_STARTUP=true`
and a Qwen-specific `QWEN35_397B_VLLM_PROVIDER_PACKAGE`; defaults for prior
Qwen, GLM, and DeepSeek recipes are unchanged. The exact escaped launch command
and immutable source snapshot are recorded in `notes/beaker_jobs.jsonl`.

## Useful monitoring commands

```bash
beaker experiment get 01M10H7055DX973Z9JSMP9FJ34 --format json
beaker job logs 01M10H70BHEZCSDJ8NW9NG2ZYT --tail 200
beaker dataset ls 01M10H705FC5KHQ9MS35WZZ50P --format json

beaker experiment get 01M10F01D8GJJ2SG2KWCPH0YGX --format json
beaker job logs 01M10F01HY1T92X2VRD4FXTR71 --tail 200
beaker dataset ls 01M10F01DFVVCZXRWZ5ECY0TZW --format json

beaker experiment get 01M1082NMW7NGKK13HHB0KEC4S --format json
beaker job logs 01M1082NRCA6ZGYAZ1RBMC9TQ6 --tail 200
beaker dataset ls 01M1082NN2T94S5CZHVANSMTMW --format json

beaker experiment get 01M101XB5JW153YEDYG897RXBH --format json
beaker job logs 01M101XB9M34P5F5A6DKCJYQNF --tail 200

beaker experiment get 01M0ZEW3ATAQMEEMRK7JR21BE4 --format json
beaker job logs 01M0ZEW3NPF8N7BKBFX01XJJ02 --tail 200
beaker experiment get 01M0ZEWC6RF25Y5KCYAJ4ZTY56 --format json
beaker job logs 01M0ZEWCGSDVAEHQSRYF2T1Z7V --tail 200

beaker experiment get 01M0ZEWMC5FC23912FP976PV49 --format json
beaker job logs 01M0ZEWMRCTWZGMPV962RCAYN1 --tail 200
beaker experiment get 01M0ZEWWZGP66WTFWP38Y9V6E2 --format json
beaker job logs 01M0ZEWX3FEAGQDJ8Q3C85GTR0 --tail 200

beaker experiment get 01M0Y116BRRY7AW2FNV524KZS7 --format json
beaker job logs 01M0Y116FX0XHBYGGN4EYJ1XHB --tail 200
beaker job results 01M0Y116FX0XHBYGGN4EYJ1XHB --format json

beaker experiment get 01M0YCRY8EVFNTY6DY3BMNKT2K --format json
beaker job logs 01M0YCRYJPW7AYCR0J4W1BS5QE --tail 200
beaker dataset ls 01M0YCRY8MP3B14H0SGKT8JATE --format json

beaker experiment get 01M0ZRQ60TP5Z1502SET6E1TYM --format json
beaker job logs 01M0ZRQ64JWYK53KY919T0GPB3 --tail 200
beaker dataset ls 01M0ZRQ6189K9G43V9KFA9JQ0A --format json

beaker job logs 01M0Y7N7XKHKMNTESWE5GGYJTN --tail 200
beaker job logs 01M0YCRFGCE0J0TSARX6QYY83J --tail 200
beaker job results 01M0YCRFGCE0J0TSARX6QYY83J --format json
beaker dataset ls 01M0YCRFBAZT8NX26YNV1KZJ5W
```

The launch ledger is the source of truth for the fully escaped reproduction
commands:

```bash
tail -n 10 notes/beaker_jobs.jsonl
```
