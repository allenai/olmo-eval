# Expanded routing sweeps: launch record

Date: 2026-07-17

## Placement

- Smoke tests: `ai2/OLMo-3-moe-experiments`, `ai2/jupiter`, preemptible. The initial submissions
  used normal priority as the Beaker job-spec representation of unallocated capacity; they and the
  corrective retries were subsequently raised or submitted at urgent priority.
- Production: `ai2/holmes-testing`, `ai2/jupiter`, urgent priority.
- Production concurrency is moderated by the workspace's 32-GPU cap.
- Harness: `codex_python`.
- Every launch is recorded in `notes/beaker_jobs.jsonl`.
- Source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/4ae4a63f2629f19cf8e9e13563ac43c158d393b49414a5d678f3139b744ec44a`.

## Smoke gate

The initial smoke gate comprised:

1. [Qwen adaptive mass at tau=0.50 with realized-K recording](https://beaker.org/ex/01KXS0WDET6XQWJKS59JG5ZQSP)
2. [GPT-OSS reference-scaled K=2, TP=1](https://beaker.org/ex/01KXS0WP760TV1E7KVZPFC40XG)
3. [GPT-OSS reference-scaled K=3, TP=2](https://beaker.org/ex/01KXS0WZBBCR8NSDSPP1F2JSNF)

Each smoke runs one MATH-500 example, one IFBench MT example, and one HumanEval example with a
512-token cap. The Qwen smoke also writes per-layer realized-K histograms and mean K to
`/results/realized_k/qwen_realized_k_<pid>.json`.

The Qwen smoke passed end to end. Its result bundle is under
`results/adaptive_experts/01KXS0WDET6XQWJKS59JG5ZQSP/`, with no metric errors and one scored
prediction for each task. Its realized-K artifact has schema version 1, all 48 MoE layers, balanced
token counts per layer, and nonzero realized-K bins at K={1,2,3,4}. Mean realized K ranges from
1.203 to 3.952 across layers.

The two initial GPT-OSS smokes reached the old 900-second provider-startup limit while their model
servers were still alive. Both routing hooks had loaded and neither showed an OOM or routing
exception. They are invalid infrastructure attempts, not model results.

Corrective smokes used a 1,800-second startup allowance:

- [K=3, TP=2](https://beaker.org/ex/01KXS46XJJZ1Q59H6539GYQE5S) passed with all three predictions
  generated and scored. It confirmed the non-power-of-two compatibility hook, K=4 reference-scale
  hook, TP=2 eager mode, and NCCL fallback.
- The first corrected [K=2, TP=1](https://beaker.org/ex/01KXS46K53F384E4BFB7811D9P) loaded the
  model but failed vLLM's 64-request sampler warmup: only 37 MiB was free for a 50 MiB allocation.
  The final [K=2, TP=1 smoke](https://beaker.org/ex/01KXS4Q9F7G9WD47BHFFFVC6PB) retained
  `max_num_seqs=64`, reduced `gpu_memory_utilization` from 0.95 to 0.94 to reserve roughly 0.8 GiB
  of headroom, and passed all three generation/scoring paths with no metric errors.

GPT-OSS production therefore uses `startup_timeout=1800` everywhere, `max_num_seqs=64`, and
`gpu_memory_utilization=0.94` at TP=1. TP=2 remains at 0.90 with eager mode and custom all-reduce
disabled. A duplicate Qwen smoke caused by a JSONL de-duplication bug was canceled immediately;
the wrapper now tests all ledger rows correctly and resumes without duplicating recorded tags.

[Smoke group](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/groups/01KXS0W8P49778V5N201FM7B7K)

## Production matrix

All conditions use three independent complete-evaluation replicates.

| Group | Conditions | Jobs | GPUs per job |
|---|---|---:|---:|
| Qwen adaptive mass | tau={0.5, 0.6, 0.7, 0.8, 0.9}; min K=1, max K=8; renormalized; realized K recorded | 15 | 4 |
| Qwen K=8-reference-scaled | K={3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16}; not renormalized | 36 | 4 |
| Qwen normalized capability backfill | K={3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16} | 36 | 4 |
| GPT-OSS K=4-reference-scaled | K={1, 2, 3, 5, 6, 7, 8}; not renormalized | 21 | 4 for K={1,2,8}; 8 otherwise |
| GPT-OSS normalized capability backfill | K={1,2,3,4,5,6,7,8} | 24 | 4 for K={1,2,4,8}; 8 otherwise |
| **Total** |  | **132** |  |

The deliberately excluded broken Qwen endpoints K={1,2,32} were not launched.

### Full expanded suite

The adaptive-mass and reference-scaled jobs run:

- `adaptive_experts:hybrid_pilot` (MATH-500, GPQA, and Qwen-thinking IFEval OOD)
- `ifeval_ood`, explicitly capped at 32,768 generated tokens
- `ifeval_mt_wildchat_unused_withRewrite`, explicitly capped at 32,768 generated tokens
- `ifeval_mt_ood_wildchat_unused_withRewrite`, explicitly capped at 32,768 generated tokens
- `humaneval:chat:pass_at_1:qwen3_thinking`

The normalized capability backfills run only the three explicit 32k IFBench tasks plus standard
HumanEval. Previously collected core-suite results supply the remaining normalized points.

### Routing definitions

For Qwen adaptive mass, the router starts from its native top eight. The smallest prefix whose
cumulative selected probability reaches tau is retained and renormalized. The telemetry artifact
records a K histogram and mean realized K for every MoE layer. It includes vLLM server warmup
tokens, which should be negligible relative to the full evaluation but must be noted in analysis.

For Qwen reference scaling:

- K<8: obtain the native top eight, retain the top K, and keep their original top-eight-normalized
  weights without renormalizing.
- K>8: obtain the native top K and divide their exponentiated router logits by the top-eight
  denominator. This preserves native top-eight scale while allowing additional experts to add mass.

For GPT-OSS reference scaling, the analogous reference is its native K=4. K<4 therefore has total
mass below one, K=4 is the identity, and K>4 preserves the native top-four scale while adding
lower-ranked expert mass.

GPT-OSS K={3,5,6,7} uses four TP=2 vLLM engines with eager mode and custom all-reduce disabled.
The other GPT-OSS K values use four TP=1 engines.

## Launch audit

The initial production submitter launched 132 jobs. Before any GPT-OSS job started, all 45 original
GPT-OSS attempts were canceled so the smoke gate could be corrected; the 87 Qwen jobs were
preserved. After the corrected smokes passed, the resumable submitter exited successfully with
`submitted=45`, `skipped=87`.

The final active-matrix ledger audit found:

- 87 original Qwen jobs plus 45 corrected GPT-OSS jobs
- 45 corrected GPT-OSS rows, 45 unique run tags, and 45 unique Beaker URLs
- exact corrected GPT-OSS counts of 21 reference-scaled and 24 normalized capability jobs
- three replicates at every requested K in both GPT-OSS groups
- all corrected jobs in `ai2/holmes-testing`, urgent priority, on `ai2/jupiter`
- every TP=1 command has memory utilization 0.94, 64 max sequences, and a 1,800-second startup
  allowance
- every TP=2 command has memory utilization 0.90, 64 max sequences, eager mode, custom all-reduce
  disabled, and a 1,800-second startup allowance
- all 45 superseded GPT-OSS attempts are terminal and canceled; all 45 corrected attempts were
  queued at the 2026-07-17 23:08 UTC audit

Production groups:

- [Qwen adaptive mass](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KXS0YEBW1F92EKWB6ZDT057H)
- [Qwen reference-scaled](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KXS12MCJWSMFGK9DB9DW2GHJ)
- [Qwen normalized capability backfill](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KXS1D0B89C89YZMNZ4B645DR)
- [GPT-OSS reference-scaled](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KXS1Q255KERY750MBTHYGX76)
- [GPT-OSS normalized capability backfill](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KXS1XRWNYMMXSTV154F611TX)

The launch can be reproduced or safely resumed with:

```bash
bash scripts/adaptive_experts/launch_approved_expanded_sweeps.sh --stage smoke
bash scripts/adaptive_experts/launch_approved_expanded_sweeps.sh --stage production
```

The wrapper deduplicates non-dry-run submissions by phase and run tag against the ledger.

## Collection status

Last checked: 2026-07-22 22:49 UTC.

The original 132-job matrix is complete: all 132 jobs succeeded, and every result is downloaded
under `results/adaptive_experts/<experiment-id>/`. This includes all 87 Qwen jobs, all 21 planned
GPT-OSS reference-scaled jobs, and all 24 normalized GPT-OSS capability backfills. There are no
unrecovered failures or active jobs in the matrix.

The complete adaptive-mass current-suite results are:

| tau | Runs collected | Realized mean K | MATH-500 | GPQA Diamond | IFBench macro (32k) | HumanEval | Four-eval macro |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 3/3 | 4.125 ± 0.009 | 35.33 ± 2.53 | 32.49 ± 4.21 | 20.48 ± 0.38 | 34.96 ± 3.13 | 30.82 ± 2.27 |
| 0.60 | 3/3 | 4.791 ± 0.015 | 83.20 ± 0.72 | 52.36 ± 3.86 | 41.94 ± 0.94 | 75.20 ± 3.13 | 63.17 ± 1.66 |
| 0.70 | 3/3 | 5.395 ± 0.015 | 94.80 ± 0.40 | 59.09 ± 1.34 | 53.15 ± 0.26 | 90.04 ± 1.86 | 74.27 ± 0.15 |
| 0.80 | 3/3 | 6.260 ± 0.008 | 95.07 ± 0.31 | 63.97 ± 1.46 | 55.62 ± 0.36 | 92.89 ± 0.35 | 76.89 ± 0.31 |
| 0.90 | 3/3 | 7.196 ± 0.002 | 95.00 ± 0.60 | 61.45 ± 0.29 | 54.96 ± 0.25 | 93.09 ± 1.53 | 76.12 ± 0.59 |

The complete Qwen K=8-reference-scaled current-suite results are:

| K | Runs collected | MATH-500 | GPQA Diamond | IFBench macro (32k) | HumanEval | Four-eval macro |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3/3 | 93.20 ± 1.00 | 53.03 ± 1.52 | 49.03 ± 0.21 | 76.63 ± 1.53 | 67.97 ± 0.21 |
| 5 | 3/3 | 95.33 ± 0.31 | 59.76 ± 1.62 | 55.46 ± 0.39 | 92.68 ± 2.20 | 75.81 ± 0.25 |
| 6 | 3/3 | 94.80 ± 0.20 | 61.45 ± 1.27 | 56.64 ± 0.51 | 92.89 ± 0.93 | 76.44 ± 0.52 |
| 7 | 3/3 | 94.67 ± 0.42 | 63.47 ± 2.39 | 56.23 ± 0.17 | 91.26 ± 1.27 | 76.41 ± 1.03 |
| 9 | 3/3 | 95.20 ± 0.35 | 63.80 ± 2.92 | 56.15 ± 0.54 | 92.48 ± 1.53 | 76.91 ± 0.87 |
| 10 | 3/3 | 95.00 ± 0.53 | 64.65 ± 0.87 | 56.80 ± 0.55 | 90.85 ± 2.11 | 76.83 ± 0.44 |
| 11 | 3/3 | 95.33 ± 0.42 | 62.96 ± 3.58 | 56.77 ± 0.57 | 91.87 ± 0.93 | 76.73 ± 1.00 |
| 12 | 3/3 | 95.27 ± 0.50 | 62.79 ± 2.10 | 56.80 ± 0.51 | 91.26 ± 2.46 | 76.53 ± 0.36 |
| 13 | 3/3 | 95.00 ± 0.53 | 66.33 ± 2.04 | 57.28 ± 0.73 | 91.26 ± 2.46 | 77.47 ± 1.07 |
| 14 | 3/3 | 94.73 ± 0.64 | 62.96 ± 1.62 | 57.64 ± 0.12 | 91.06 ± 1.86 | 76.60 ± 0.70 |
| 15 | 3/3 | 95.47 ± 0.12 | 61.95 ± 0.77 | 56.39 ± 0.82 | 90.85 ± 1.22 | 76.17 ± 0.22 |
| 16 | 3/3 | 94.33 ± 0.31 | 62.12 ± 0.51 | 56.49 ± 0.49 | 90.04 ± 1.53 | 75.75 ± 0.41 |

The normalized capability backfill currently provides:

| K | Runs collected | IFBench macro (32k) | HumanEval |
|---:|---:|---:|---:|
| 3 | 3/3 | 27.12 ± 0.96 | 22.15 ± 3.97 |
| 5 | 3/3 | 53.97 ± 0.29 | 88.62 ± 1.96 |
| 6 | 3/3 | 56.23 ± 0.27 | 93.09 ± 2.14 |
| 7 | 3/3 | 56.22 ± 0.56 | 94.11 ± 1.53 |
| 9 | 3/3 | 56.78 ± 0.50 | 91.06 ± 1.86 |
| 10 | 3/3 | 56.85 ± 0.40 | 91.87 ± 0.93 |
| 11 | 3/3 | 56.65 ± 0.67 | 91.87 ± 0.93 |
| 12 | 3/3 | 56.30 ± 0.52 | 89.84 ± 0.35 |
| 13 | 3/3 | 55.81 ± 0.22 | 88.21 ± 1.86 |
| 14 | 3/3 | 55.22 ± 0.44 | 86.59 ± 0.61 |
| 15 | 3/3 | 55.21 ± 0.38 | 87.80 ± 0.61 |
| 16 | 3/3 | 55.35 ± 0.20 | 86.99 ± 2.31 |

Together with the previously collected normalized K=4 and native K=8 capability runs, this
completes the normalized current-suite curve at every K from 3 through 16. The unweighted
four-eval mean is 70.20 at K=4, 74.48 at K=5, 76.71 at K=6, 76.98 at K=7, and 76.46 at native
K=8. It remains between 73.58 and 76.70 through K=9--16. The K=8-reference-scaled K=4 mean is
74.03, preserving most of the K=8 baseline and outperforming normalized K=4 by 3.83 points.

The currently collected GPT-OSS normalized results are:

| K | Runs collected | MATH-500 | GPQA Diamond | IFBench macro (32k) | HumanEval | Four-eval mean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3/3 | 42.80 ± 2.16 | 29.97 ± 3.72 | 26.58 ± 0.92 | 7.93 ± 2.20 | 26.82 |
| 2 | 3/3 | 92.00 ± 0.53 | 70.03 ± 0.29 | 72.19 ± 0.56 | 98.17 ± 1.22 | 83.10 |
| 3 | 3/3 | 91.87 ± 1.10 | 74.58 ± 1.27 | 71.40 ± 0.40 | 97.97 ± 0.70 | 83.95 |
| 4 | 3/3 | 92.40 ± 0.35 | 70.54 ± 1.62 | 71.60 ± 0.36 | 98.37 ± 0.35 | 83.23 |
| 5 | 3/3 | 91.87 ± 0.90 | 72.22 ± 0.87 | 71.99 ± 0.48 | 97.56 ± 0.61 | 83.41 |
| 6 | 3/3 | 92.47 ± 0.99 | 70.54 ± 1.54 | 71.27 ± 0.60 | 96.95 ± 1.22 | 82.81 |
| 7 | 3/3 | 92.00 ± 0.53 | 70.20 ± 0.51 | 71.90 ± 0.50 | 97.76 ± 0.93 | 82.97 |
| 8 | 3/3 | 91.93 ± 0.76 | 68.52 ± 4.11 | 70.41 ± 0.19 | 97.56 ± 0.61 | 82.11 |

The complete GPT-OSS K=4-reference-scaled results are:

| K | Runs collected | MATH-500 | GPQA Diamond | IFBench macro (32k) | HumanEval | Four-eval mean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3/3 | 49.93 ± 1.89 | 21.72 ± 1.75 | 32.53 ± 0.49 | 44.72 ± 3.87 | 37.22 |
| 2 | 3/3 | 91.27 ± 0.23 | 67.51 ± 1.05 | 69.70 ± 0.50 | 97.15 ± 1.27 | 81.41 |
| 3 | 3/3 | 92.07 ± 0.83 | 70.71 ± 1.82 | 70.59 ± 0.84 | 97.97 ± 0.35 | 82.83 |
| 4 | 3/3 | 92.40 ± 0.35 | 70.54 ± 1.62 | 71.60 ± 0.36 | 98.37 ± 0.35 | 83.23 |
| 5 | 3/3 | 92.53 ± 0.50 | 74.58 ± 1.77 | 72.60 ± 0.60 | 98.37 ± 0.35 | 84.52 |
| 6 | 3/3 | 92.20 ± 1.25 | 71.38 ± 1.54 | 73.48 ± 0.53 | 97.97 ± 0.93 | 83.76 |
| 7 | 3/3 | 91.87 ± 0.99 | 71.72 ± 4.37 | 71.97 ± 0.28 | 98.17 ± 0.61 | 83.43 |
| 8 | 3/3 | 91.60 ± 0.20 | 72.22 ± 0.51 | 72.47 ± 0.70 | 98.17 ± 0.61 | 83.61 |

Both complete GPT-OSS curves have a sharp K=1-to-K=2 transition and are otherwise essentially
flat. Normalized K=2--8 spans 82.11--83.95, while reference-scaled K=2--8 spans 81.41--84.52.
Adding experts above native K=4 offers no consistent gain. Reference scaling gives a modest
average advantage at K=5--8 but not at K=2--3, and most pointwise differences are comparable to
the evaluation variability. The robust conclusion is that two of four default experts preserve
the measured capabilities under either scaling rule.

The first job inside GPT-OSS reference-scaled K=8 replicate 3
[failed during CUDA-graph warmup](https://beaker.org/ex/01KXS55XA6Q9582QS8ESGQB0AT), but its Beaker
restart completed successfully on 2026-07-20. The separately submitted
[replacement experiment](https://beaker.org/ex/01KXZ1GH969HK261P949Q9HKZ1) also subsequently
succeeded; it is valid but redundant and is excluded from the three-replicate summary above.

Current-suite plots and exact machine-readable values:

- [Qwen normalized versus reference-scaled plot](plots/qwen3_hybrid_current_suite_expert_sweep.png)
- [GPT-OSS normalized versus reference-scaled plot](plots/gptoss_120b_current_suite_expert_sweep.png)
- [Current-suite score CSV](current_suite_expert_sweeps.csv)

Scores are percentages and variability is the sample standard deviation across complete
evaluation replicates. The four-eval macro is the unweighted mean of MATH-500, GPQA Diamond,
IFBench macro, and HumanEval. Realized mean K is token-and-layer weighted across the four vLLM
engines; as documented above, it includes server warmup tokens.

Exact experiment IDs, result datasets, conditions, and Beaker URLs remain in
`notes/beaker_jobs.jsonl`; the production group links above provide the corresponding web views.

Validation passed for all 132 matrix bundles: `metrics.json` contains no task errors and
prediction counts match every configured full task size. Each of the 15 adaptive runs contains
four realized-K artifacts covering all 48 MoE layers, and every K histogram sums to its recorded
token count. No production jobs remain, and there are no unrecovered failures in this matrix.
