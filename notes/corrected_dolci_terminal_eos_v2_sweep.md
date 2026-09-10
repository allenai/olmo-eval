# Corrected Dolci/OLMo-3 reasoning SFT expert sweep

Date: 2026-07-19

## Checkpoint

`/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints/qwen3-30b-a3b-dolci-think-olmo-core-sft-full-terminal-eos-v2-lr4e-5-20260715-071349-hf`

This is the corrected Qwen3-30B-A3B + OLMo-3 reasoning SFT checkpoint. Results from the earlier,
incorrectly trained checkpoints are excluded from this sweep and will not be combined with it.

## Evaluation matrix

- Target K: `{3,4,5,6,7,8,9,10,11,12}`.
- Three complete-evaluation replicates per K and routing scale.
- Normalized routing: native top-K routing normalized at the requested K.
- Unnormalized routing: K=8-reference scaling.
  - For K<8, run the native top eight, retain the top K, and preserve their original shares from
    the normalized top-eight mixture without survivor renormalization.
  - K=8 is an identity/sanity condition.
  - For K>8, run native top K and divide by the top-eight denominator, allowing the additional
    experts to add routed mass without renormalizing the full selected set to one.
- Total production jobs: 60.

Every production job runs the current six-task suite together:

- MATH-500;
- GPQA Diamond;
- IFBench IFEval OOD;
- IFBench multi-turn WildChat rewrite;
- IFBench multi-turn OOD WildChat rewrite; and
- standard HumanEval pass@1.

All six tasks allow up to 32,768 generated tokens. Serving uses a 40,960-token model window, four
one-GPU vLLM engines, `max_num_seqs=16`, and `reasoning_parser=olmo3` to match the checkpoint's
OLMo thinking template.

## Placement and tracking

- Smoke workspace: `ai2/olmo-instruct` for the active retries. Two initial attempts in
  `ai2/OLMo-3-moe-experiments` were manually canceled during workspace-placement coordination and
  are invalid infrastructure attempts.
- Production workspace: `ai2/holmes-testing`.
- Cluster: `ai2/jupiter`.
- Priority: urgent for smoke and production.
- Production group: `adaptive-experts-qwen-dolci-terminal-eos-v2-fullsuite-20260719`.
- Production group link:
  <https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KXWFJF6QVPDKMP4BKXN6K5A6>.
- Active smoke group:
  `adaptive-experts-qwen-dolci-terminal-eos-v2-smokes-olmo-instruct-20260719`.
- Smoke group link:
  <https://beaker.org/orgs/ai2/workspaces/olmo-instruct/groups/01KXWF9MPCWKXHM7F078S8Z2DV>.
- Every launch is appended to `notes/beaker_jobs.jsonl` with a unique corrected-checkpoint run tag.

## Smoke gate

Two one-GPU smokes exercised model conversion, the OLMo-3 reasoning parser, MATH scoring, IFBench
scoring, HumanEval sandbox scoring, and both routing paths:

- Normalized K=8: <https://beaker.org/ex/01KXWFAK22A5F6WWQJT3B26BF9>.
- K=8-reference-scaled K=3: <https://beaker.org/ex/01KXWFAW527F9XZK1PDQT5XX7Q>.

Both completed with exit code zero. Each produced and scored one MATH-500, one IFBench, and one
HumanEval example, with one prediction per task and an empty error list. The reference-scaled smoke
also installed and exercised the Qwen routing hook successfully.

## Production launch audit

The full sweep was submitted on 2026-07-19 after the smoke gate passed:

- 60 unique experiments and 60 unique run tags;
- 30 normalized and 30 K=8-reference-scaled experiments;
- exactly three replicates for every K/mode pair;
- all experiments use the corrected checkpoint, `ai2/holmes-testing`, `ai2/jupiter`, urgent
  priority, four GPUs, the six-task suite, a 32,768-token generation cap, a 40,960-token model
  window, and `reasoning_parser=olmo3`;
- reference-scaled K=3--7 jobs route over eight experts and retain K without renormalization;
- reference-scaled K=8 is the identity condition; and
- reference-scaled K=9--12 jobs route over the requested K and scale against the top-eight
  denominator.

At the completion of submission, all 60 jobs were present in the production group and waiting for
workspace capacity; there were no immediate launch failures.

## Final status and results

Last checked: 2026-07-22 16:01 UTC.

The production group is complete: all 60 jobs succeeded, and all 60 result bundles are downloaded
and fully validated. Every bundle has six task outputs, an empty metric-error list, and request and
prediction counts matching the configured example counts.

The table reports the mean and sample standard deviation across the three complete-evaluation
replicates at every point:

| K | Routing | n | MATH-500 | GPQA Diamond | IFBench macro (32k) | HumanEval | Four-eval mean |
|---:|:--|---:|---:|---:|---:|---:|---:|
| 3 | normalized | 3 | 82.47 ± 1.51 | 43.27 ± 1.54 | 38.93 ± 0.54 | 2.44 ± 1.06 | 41.78 ± 0.84 |
| 4 | normalized | 3 | 91.27 ± 0.31 | 53.20 ± 1.05 | 44.67 ± 0.33 | 6.30 ± 2.46 | 48.86 ± 0.67 |
| 5 | normalized | 3 | 93.33 ± 0.50 | 57.07 ± 2.81 | 47.53 ± 0.44 | 13.21 ± 1.96 | 52.79 ± 1.10 |
| 6 | normalized | 3 | 93.33 ± 0.46 | 56.06 ± 1.01 | 49.26 ± 0.53 | 12.60 ± 2.31 | 52.81 ± 0.99 |
| 7 | normalized | 3 | 93.40 ± 0.20 | 58.25 ± 1.77 | 49.30 ± 0.21 | 18.29 ± 1.06 | 54.81 ± 0.78 |
| 8 | normalized | 3 | 93.13 ± 0.58 | 56.23 ± 2.04 | 50.82 ± 0.33 | 25.41 ± 3.13 | 56.40 ± 0.32 |
| 9 | normalized | 3 | 93.53 ± 0.95 | 55.89 ± 1.27 | 50.49 ± 0.30 | 21.54 ± 2.88 | 55.37 ± 0.66 |
| 10 | normalized | 3 | 92.73 ± 0.31 | 54.21 ± 1.62 | 50.48 ± 0.10 | 23.58 ± 1.27 | 55.25 ± 0.81 |
| 11 | normalized | 3 | 92.93 ± 1.03 | 52.36 ± 0.77 | 49.50 ± 0.15 | 25.00 ± 4.00 | 54.95 ± 1.03 |
| 12 | normalized | 3 | 92.07 ± 0.83 | 50.67 ± 2.10 | 50.08 ± 0.32 | 25.20 ± 4.15 | 54.50 ± 1.26 |
| 3 | K=8-reference scaled | 3 | 89.93 ± 1.63 | 45.12 ± 3.58 | 44.81 ± 0.21 | 2.64 ± 0.93 | 45.63 ± 0.49 |
| 4 | K=8-reference scaled | 3 | 91.53 ± 1.30 | 47.98 ± 1.34 | 47.14 ± 0.26 | 3.46 ± 0.35 | 47.53 ± 0.64 |
| 5 | K=8-reference scaled | 3 | 92.93 ± 0.46 | 55.22 ± 3.09 | 48.32 ± 0.72 | 4.67 ± 1.96 | 50.29 ± 0.36 |
| 6 | K=8-reference scaled | 3 | 93.40 ± 0.69 | 55.89 ± 1.54 | 49.21 ± 0.53 | 9.96 ± 2.31 | 52.12 ± 0.26 |
| 7 | K=8-reference scaled | 3 | 93.27 ± 0.50 | 55.72 ± 2.54 | 49.79 ± 0.15 | 16.06 ± 0.93 | 53.71 ± 0.29 |
| 8 | K=8-reference scaled | 3 | 93.20 ± 0.60 | 59.93 ± 2.04 | 50.55 ± 0.29 | 26.83 ± 3.23 | 57.63 ± 0.91 |
| 9 | K=8-reference scaled | 3 | 93.73 ± 0.50 | 58.25 ± 0.77 | 50.93 ± 0.39 | 29.07 ± 2.14 | 57.99 ± 0.14 |
| 10 | K=8-reference scaled | 3 | 93.93 ± 0.58 | 57.74 ± 2.39 | 50.55 ± 0.19 | 27.85 ± 2.31 | 57.52 ± 1.07 |
| 11 | K=8-reference scaled | 3 | 94.07 ± 0.23 | 57.07 ± 0.87 | 50.74 ± 0.25 | 25.20 ± 0.35 | 56.77 ± 0.33 |
| 12 | K=8-reference scaled | 3 | 93.87 ± 0.83 | 55.22 ± 2.54 | 50.12 ± 0.50 | 24.59 ± 3.13 | 55.95 ± 1.57 |

This corrected post-trained checkpoint has a materially different curve from off-the-shelf Qwen.
MATH is nearly saturated by K=4--5 and IFBench by roughly K=6--8, but GPQA and especially
HumanEval remain K-sensitive. HumanEval rises from 2.44 at normalized K=3 to 25.41 at K=8; this
coding sensitivity drives much of the four-eval slope.

Normalized routing peaks at K=8 with a four-eval mean of 56.40 and then drifts down to 54.50 at
K=12. Reference scaling peaks at K=9 with 57.99, remains strongest across K=8--10, and declines to
55.95 by K=12. It helps at K=3 and K=8--12, but is worse than renormalization at K=4--7, so scale
preservation is not universally beneficial on this checkpoint. The independent normalized and
reference-scaled K=8 replicates are equivalent policies in expectation; their observed difference
is stochastic rather than a routing-definition effect. More than eight active experts does not
produce a consistent gain, and the mild high-K decline is concentrated in GPQA and HumanEval.

Final plot:
[current-suite normalized/reference curves](plots/qwen3_dolci_terminal_eos_v2_current_suite_expert_sweep.png).
Machine-readable values are included in [the current-suite CSV](current_suite_expert_sweeps.csv).

## Reproduction

```bash
bash scripts/adaptive_experts/launch_corrected_dolci_full_sweep.sh --stage smoke
bash scripts/adaptive_experts/launch_corrected_dolci_full_sweep.sh --stage production
```

The wrapper deduplicates non-dry-run submissions by phase and run tag against the append-only
ledger.
