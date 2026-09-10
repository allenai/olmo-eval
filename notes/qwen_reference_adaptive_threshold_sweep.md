# Qwen reference-preserving adaptive-threshold sweep

Launched: 2026-07-22

## Question

The earlier adaptive-mass sweep selected the smallest prefix of Qwen's native top eight whose
cumulative router mass reached `tau`, then renormalized that prefix to total mass one. This sweep
keeps the same dynamic K decision but leaves the retained native top-eight weights unchanged. It
tests whether adaptive selection benefits from the same K=8-reference-preserving scale that helped
fixed low-K inference.

## Conditions

- Model: `Qwen/Qwen3-30B-A3B` hybrid-thinking checkpoint.
- Thresholds: `tau={0.50,0.60,0.70,0.80}`.
- Bounds: minimum K=1, maximum K=8.
- Retained weights: native top-eight-normalized weights, with dropped tail weights set to zero and
  no renormalization.
- Replicates: three complete evaluations per threshold (12 jobs total).
- Tasks: MATH-500, GPQA Diamond, IFBench at a 32,768-token cap, and standard HumanEval. The launch
  also retains standalone historical IFEval OOD for backward comparison; it is excluded from the
  adopted four-eval macro.
- Telemetry: realized-K histograms and means for every one of Qwen's 48 MoE layers.
- Runtime layout: four independent one-H100 vLLM engines per full evaluation.
- Execution mode: vLLM eager mode. This avoids CUDA graph capture around the telemetry hook, so
  realized-K counts cover actual inference calls rather than a captured warm-up graph.
- Placement: `ai2/holmes-testing`, `ai2/jupiter`, urgent priority, preemptible/unallocated.

## Smoke gates

[Initial graph-mode smoke](https://beaker.org/ex/01KY61NCVWWF47EKH75P5CZ3DM) used one H100 and one
example each from MATH-500, IFBench MT, and HumanEval at a 512-token cap. It passed its tiny
workload, but concurrent production launches exposed that the telemetry operation could execute
during vLLM CUDA graph capture and invalidate the graph. The 12 graph-mode production attempts
were stopped or failed and are superseded; they must not be collected as evaluation results.

[Corrected eager-mode smoke](https://beaker.org/ex/01KY62BT5MEDS1X0SN46FD5PPA) repeated the gate with
`provider.kwargs.enforce_eager=true`. It succeeded with:

- one request and one prediction for each task;
- no metric errors;
- the vLLM log explicitly reporting
  `adaptive_mass(threshold=0.5,min_k=1,max_k=8,renormalize=False)`; and
- one valid telemetry artifact with 48 layers, roughly 18,185 observed token-layer calls per layer,
  histogram sums equal to token counts, and nonzero bins at K={1,2,3,4}; and
- four independent eager-mode server commands in the production configuration, each TP=1.

Eager mode is slower than CUDA-graph execution but does not change the routing policy or output
definition. It is required here for trustworthy realized-K measurement.

## Production launch

[Corrected eager-mode Beaker group](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KY62MA9VXSX3MQTT04M6974V)

| tau | Replicate 1 | Replicate 2 | Replicate 3 |
|---:|:--|:--|:--|
| 0.50 | [job](https://beaker.org/ex/01KY62MEGMKD974RKXY0PE0EMQ) | [job](https://beaker.org/ex/01KY62MPCNQDF351V25KSV17CH) | [job](https://beaker.org/ex/01KY62MY3NA32FXHMZG1XMPMEJ) |
| 0.60 | [job](https://beaker.org/ex/01KY62N6FREXWKDMA2PWF101XY) | [job](https://beaker.org/ex/01KY62NEKCFY1DJ982D1QPC44Q) | [job](https://beaker.org/ex/01KY62NP4KR6K50R9GP0MFZ32D) |
| 0.70 | [job](https://beaker.org/ex/01KY62NXKRZVTES9KRTXDX8HEG) | [job](https://beaker.org/ex/01KY62P5PQ55GFQEBA599QEA5V) | [job](https://beaker.org/ex/01KY62PDECWQY4M7QT42E2Z8P1) |
| 0.80 | [job](https://beaker.org/ex/01KY62PN5B633SPEA68J6YXCS7) | [job](https://beaker.org/ex/01KY62PX74K76NQG1DZAXTHC9Y) | [job](https://beaker.org/ex/01KY62Q5031QYFSF0XX3FFBMQX) |

The corrected post-launch audit found exactly 12 unique experiment IDs and run tags, three jobs at
every threshold, and no placement or routing-setting mismatches. Every command has four TP=1
engines, `enforce_eager=true`, `renormalize=false`, K bounds 1--8, and realized-K recording.
At 2026-07-22 23:35 UTC, eight jobs were running and four were queued behind the 32-GPU workspace
cap. A sampled running job brought all four eager-mode servers up in 88 seconds and began its first
four 64-request batches without a telemetry, CUDA, or serving error.

The superseded graph-mode group is retained for provenance at
[this link](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KY621D9K5206CJW008BS9P56).
All 12 attempts are terminal failures or user stops caused by the telemetry/CUDA-graph interaction,
not model-quality results.

## Final collection: 2026-07-24 16:56 UTC

Eleven valid bundles are locally collected. Tau=0.50 replicate 3 was preempted with exit code 143
before completing and is excluded, leaving two runs at tau=0.50 and three at every other
threshold. Each successful bundle has all seven configured task records, the expected 4,623 task
instances including the standalone 300-example historical IFEval OOD task, no metric errors, and
four valid 48-layer realized-K files. The adopted four-eval suite contains 4,323 responses per run.

Final score summary (percent; uncertainty is sample SD across available runs):

| tau | Runs | Realized mean K | MATH-500 | GPQA | IFBench | HumanEval | Four-eval macro |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 2/3 | 3.498 ± 0.002 | 92.90 ± 0.42 | 54.55 ± 4.29 | 51.75 ± 0.30 | 75.30 ± 0.43 | 68.63 ± 0.99 |
| 0.60 | 3/3 | 4.271 ± 0.006 | 94.40 ± 0.20 | 59.43 ± 2.04 | 54.24 ± 0.08 | 86.59 ± 2.79 | 73.66 ± 1.16 |
| 0.70 | 3/3 | 5.144 ± 0.005 | 94.87 ± 0.42 | 59.43 ± 1.91 | 54.01 ± 0.79 | 88.01 ± 1.27 | 74.08 ± 0.67 |
| 0.80 | 3/3 | 6.107 ± 0.006 | 95.40 ± 0.35 | 63.97 ± 0.77 | 56.88 ± 0.10 | 89.84 ± 0.35 | 76.52 ± 0.31 |

The reference-preserving scale is the main positive result. Relative to the earlier renormalized
adaptive policy, the macro gain is +37.81 at tau=0.50, +10.49 at tau=0.60, -0.19 at tau=0.70, and
-0.36 at tau=0.80. Scaling matters enormously below the viable mass boundary and essentially not
at all once tau reaches roughly 0.70.

The threshold rule is highly concentrated:

- tau=0.50 uses K=3 for 44.23% and K=4 for 52.83% of recorded token-layer calls; mean K=3.498.
- tau=0.60 uses K=4 for 58.58% and K=5 for 34.44%; mean K=4.271.
- tau=0.70 uses K=5 for 66.22% and K=6 for 24.40%; mean K=5.144.
- tau=0.80 uses K=6 for 74.25% and K=7 for 18.48%; mean K=6.107.

No collected call uses K>4/5/6/7 at tau=0.50/0.60/0.70/0.80 respectively. Across layers, per-run
mean K spans 3.03--3.66, 3.80--4.45, and 4.69--5.31 for the first three thresholds. Each estimate contains
0.85--1.21 billion token-layer observations; warm-up is included but negligible at this scale.
The older graph-mode realized-K figures should not be used as an apples-to-apples comparison,
because that hook ran during CUDA-graph capture rather than every inference call.

### Response behavior

Reference-preserving routing also sharply reduces the catastrophic response behavior seen under
low-tau renormalization:

| tau | Empty returned answers | 32k cap hits | Responses/run |
|---:|---:|---:|---:|
| 0.50 reference-preserving | 288 average (6.7%) | 323 average (7.5%) | 4,323 |
| 0.50 renormalized | 2,328 (53.8%) | 1,035 (23.9%) | 4,323 |
| 0.60 reference-preserving | 156 average (3.6%) | 189 average (4.4%) | 4,323 |
| 0.60 renormalized | 719 average (16.6%) | 303 average (7.0%) | 4,323 |
| 0.70 reference-preserving | 135 average (3.1%) | 152 average (3.5%) | 4,323 |
| 0.70 renormalized | 178 average (4.1%) | 144 average (3.3%) | 4,323 |
| 0.80 reference-preserving | 118 average (2.7%) | 141 average (3.3%) | 4,323 |

The remaining low-tau degradation is mostly long hidden reasoning and failure to return a final
answer, concentrated in IFBench, rather than broad lexical corruption. In a matched replicate-2
cohort, 461 MATH prompts were correct at all three collected thresholds. Their returned solutions
were almost identical in mean length (625/626/629 tokens at tau=0.50/0.60/0.70); inferred hidden
reasoning was modestly longer at lower tau (3,709/3,597/3,466 tokens). None hit the cap or returned
an empty answer. Manual inspection of low-tau mistakes found generally fluent, structured, but
occasionally self-contradictory or overlong reasoning rather than gibberish.

At matched compute, adaptive mass does not beat fixed K. Tau=0.60 averages K=4.27 and a 73.66
macro, essentially the fixed reference-scaled K=4 macro of 74.03. Tau=0.70 averages K=5.14 and a
74.08 macro, below fixed K=5's 75.81. Tau=0.80 averages K=6.11 and a 76.52 macro, essentially
identical to fixed K=6's 76.44 and native K=8's 76.46 macro. Tau=0.80 therefore preserves the
aggregate with roughly 24% fewer active experts than native K=8, but HumanEval remains 3.46 points
below native and is offset by stronger GPQA/IFBench scores. Reference-preserving scale is valuable;
router mass alone is not yet a superior compute-allocation signal.

Machine-readable results are in `notes/qwen_reference_adaptive_threshold_results.json`. The
reproducible collector/analysis is
`scripts/adaptive_experts/analyze_qwen_reference_adaptive_thresholds.py`; the plot generator is
`scripts/adaptive_experts/plot_qwen_reference_adaptive_thresholds.py`. Final plots:

- `notes/plots/qwen3_reference_adaptive_tau_performance.png`
- `notes/plots/qwen3_reference_adaptive_tau_performance.svg`

### Comparison with strict reference-preserving cutoffs

Plotting strict K at its nominal expert count and adaptive tau at its realized mean K shows that
the adaptive points mostly lie on, rather than above, the fixed-K frontier:

| Adaptive policy | Mean K | Adaptive macro | Nearest fixed K | Fixed macro | Difference |
|:--|--:|--:|--:|--:|--:|
| tau=0.50 | 3.50 | 68.63 | K=3 | 67.97 | +0.65 |
| tau=0.60 | 4.27 | 73.66 | K=4 | 74.03 | -0.36 |
| tau=0.70 | 5.14 | 74.08 | K=5 | 75.81 | -1.73 |
| tau=0.80 | 6.11 | 76.52 | K=6 | 76.44 | +0.08 |

Tau=0.80 and fixed K=6 have essentially identical aggregate quality. Adaptive routing shifts the
task mix—GPQA is +2.52 points and MATH/IFBench are slightly higher, while HumanEval is -3.05—but
does not provide a clear matched-compute macro advantage. The comparison plot is:

- `notes/plots/qwen3_reference_adaptive_vs_fixed_k.png`
- `notes/plots/qwen3_reference_adaptive_vs_fixed_k.svg`

All submissions, including the smoke, are recorded in `notes/beaker_jobs.jsonl`. The resumable
launcher is `scripts/adaptive_experts/launch_qwen_reference_adaptive_thresholds.sh`.
