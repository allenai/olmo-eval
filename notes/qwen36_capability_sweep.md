# Qwen3.6-35B-A3B capability sweep

Last updated: 2026-08-18 03:20 UTC

## Extreme reference-scaled sweep

Nine jobs extend the corrected capability curve to K=64, 128, and 256. Each
K has three replicates (seeds 42/43/44) and bundles MATH-500, GPQA Diamond,
AIME 2025 pass@32, and the three-task IFBench macro. The serving policy uses
the nested `text_config.num_experts_per_tok=K` override, retains all K routed
experts, scales their weights to the native K=8 reference, and does not
renormalize them. Startup remains fail-closed on the live router assertion.

The remaining protocol matches the corrected capability sweep: four TP=1
engines (four 80-GB GPUs), a 32,768-token generation cap for every task,
110,000-token server context, Qwen thinking enabled, and urgent unallocated
scheduling in `ai2/holmes-testing` through the `80g` selector.

Group: `qwen36-capability-extreme-reference-20260816`.

| K | Replicate | Seed | Beaker experiment | Result dataset | Initial status |
|---:|---:|---:|---|---|---|
| 64 | 1 | 42 | [01M066GNQ9G7V1Z2XPRWFSDEFW](https://beaker.org/ex/01M066GNQ9G7V1Z2XPRWFSDEFW) | `01M066GNQFMFJJ9SAJCMHQM7EE` | running |
| 64 | 2 | 43 | [01M066GYCBEWKYH2Y6DCDD7FKX](https://beaker.org/ex/01M066GYCBEWKYH2Y6DCDD7FKX) | `01M066GYCKQAFVH7A1KC6BYF1Q` | scheduled |
| 64 | 3 | 44 | [01M066H6X6GHFF16QEEDRRWSV8](https://beaker.org/ex/01M066H6X6GHFF16QEEDRRWSV8) | `01M066H6XNFRC0CJSQEF0N5ZT5` | scheduled |
| 128 | 1 | 42 | [01M066HEVRTP1KVDREHAJP0ZR4](https://beaker.org/ex/01M066HEVRTP1KVDREHAJP0ZR4) | `01M066HEW3ZWNN7ZDQXJNQA7GK` | scheduled |
| 128 | 2 | 43 | [01M066HQ28CYXD53Q5BT7C9CRE](https://beaker.org/ex/01M066HQ28CYXD53Q5BT7C9CRE) | `01M066HQ2M1XV51ZW9WFJGK301` | running |
| 128 | 3 | 44 | [01M066HZHXQPGFA2YZ6PDE52YZ](https://beaker.org/ex/01M066HZHXQPGFA2YZ6PDE52YZ) | `01M066HZJ2TAPEX8Q0FR4J5X36` | scheduled |
| 256 | 1 | 42 | [01M066J7JP38VQS0KTCYEZRQRX](https://beaker.org/ex/01M066J7JP38VQS0KTCYEZRQRX) | `01M066J7K3493D756CXX655WSW` | scheduled |
| 256 | 2 | 43 | [01M066JG72F9326K6DV5HKJ28J](https://beaker.org/ex/01M066JG72F9326K6DV5HKJ28J) | `01M066JG781KM79NNNC95DDSV1` | scheduled |
| 256 | 3 | 44 | [01M066JRHWXM9PFYDZRJ8D18XV](https://beaker.org/ex/01M066JRHWXM9PFYDZRJ8D18XV) | `01M066JRJ42MDHTPY8S0HQQ61C` | queued |

Status update, 2026-08-17 07:45 UTC: all three K=256 jobs were manually
canceled to conserve compute after K=128 TBLite showed severe provisional
degradation. Replicates 1/2 stopped at 364/4189 scored instances and
replicate 3 never scheduled. These partial runs are excluded from scientific
results and plots. The K=64 and K=128 capability jobs remain active.

Status update, 2026-08-17 16:54 UTC: the six retained extreme jobs remain
healthy and active. K=64 replicates have scored 1920, 1984, and 1976 of 4189
instances (46--47%); K=128 replicates have scored 1172, 1215, and 1169
(28--29%). No retained capability job has failed.

Status update, 2026-08-17 19:42 UTC: all three K=128 capability jobs were
manually canceled because their additional information no longer justified
their very long runtimes. Their final partial counts were 1529, 1441, and
1482 of 4189 instances; they are excluded from all scientific results. The
three K=64 jobs remain active. A replacement bridge at K={16,32}, three runs
per K, was launched and post-submit validated; see
`qwen36_capability_bridge_jobs.md`.

Status update, 2026-08-18 03:20 UTC: all three K=64 jobs reached the explicit
24-hour Beaker timeout and exited with code 143 after scoring 2494, 2494, and
2439 of 4189 instances. These partial runs are excluded from scientific
results and plots. All six K={16,32} bridge jobs remain active; none has
completed yet.

## Corrected K=10/K=12 replacements

Dedicated K=10 and K=12 smokes now exit zero and explicitly report matched
router layers at the requested K. Six corrected production jobs—three seeds
per K—are queued in
[Beaker group 01M03A08C7SAKMBZA0P5G08W4Y](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01M03A08C7SAKMBZA0P5G08W4Y).
The maintained CSV and plot currently end at K=8 and will admit K=10/K=12
again only from the corrected `nestedfix1-20260815` bundles.

The complete replacement matrix and smoke evidence are in
`notes/qwen36_kgt8_replacement_jobs.md`.

> **Routing correction:** the historical K=10 and K=12 rows and plot points
> are invalid as expert-count interventions. Those jobs passed
> `{"num_experts_per_tok": K}` to the outer Qwen3.6 composite config, while
> the actual MoE setting is `text_config.num_experts_per_tok`. Consequently,
> those runs remained at native K=8. K=4/6/8 are valid. The launcher now uses
> the nested override for future K>8 runs, but this capability suite has not
> yet been rerun at corrected K=10/12.

Artifacts:

- `notes/plots/qwen36_capability_expert_sweep.png` and `.svg`
- `notes/qwen36_capability_results.csv`

## Goal

Evaluate the same expert-count conditions used by the Qwen3.6 Terminal-Bench sweep on the
non-coding post-training suite. The conditions are native K=8 and K=8-reference-scaled
K in {4, 6, 10, 12}. The latter preserves the native top-eight scale: K<8 truncates the
native top eight without renormalizing, while K>8 routes K experts and rescales them so the
top-eight prefix has the native K=8 mass.

## Evaluation matrix

- Model: `Qwen/Qwen3.6-35B-A3B`.
- K: `{4, 6, 8, 10, 12}`.
- Three replicates per K, using task seeds 42, 43, and 44.
- Tasks bundled in every job:
  - `math500:chat`;
  - `gpqa_diamond:qwen3_thinking`;
  - `aime_2025:pass_at_32`;
  - `ifeval_ood`;
  - `ifeval_mt_wildchat_unused_withRewrite`; and
  - `ifeval_mt_ood_wildchat_unused_withRewrite`.
- Every task has `sampling_params.max_tokens=32768`.
- Four independent TP=1 H100 vLLM engines per job.
- Server context: 110,000 tokens, which leaves ample prompt room above the 32,768-token output
  cap for the long IFBench examples.
- Qwen thinking template enabled; `qwen3` reasoning parser; language-only model path.
- Hybrid-model backends: `TRITON_ATTN` attention and Triton GDN prefill.

## Beaker

- Workspace: `ai2/holmes-testing`.
- Cluster selector: `80g` (multi-cluster 80 GB GPU alias).
- Priority: urgent, unallocated.
- Group: [adaptive-compute-qwen36-capability-reference-20260812](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KZVSKMN41N1SQ6SQ4TGP87J5).
- Submitted experiments: 15/15; all are recorded in `notes/beaker_jobs.jsonl` under this group.

Launch helper: `scripts/adaptive_experts/launch_qwen36_capability_sweep.sh`.

## Historical results

All three replicates for every nominal K completed, were downloaded under
`results/adaptive_experts`, and passed bundle validation. This validates the
evaluation outputs but not the K>8 routing intervention: only K=4/6/8 may be
used scientifically until corrected K=10/12 replacements run. Values below
are retained for provenance and are mean ± sample SD over the three task
seeds.

| K | MATH-500 | GPQA Diamond | IFBench macro | AIME 2025 pass@1 |
|---:|---:|---:|---:|---:|
| 4 | 95.07 ± 1.10 | 78.28 ± 0.87 | 66.83 ± 0.47 | 66.11 ± 0.53 |
| 6 | 93.40 ± 0.87 | 79.46 ± 2.54 | 69.79 ± 1.00 | 62.19 ± 0.58 |
| 8 | 93.53 ± 0.50 | 80.47 ± 2.04 | 71.92 ± 1.10 | 60.59 ± 0.52 |
| 10 (invalid; actually native K=8) | 93.53 ± 0.31 | 81.48 ± 1.77 | 72.02 ± 0.62 | 60.63 ± 0.38 |
| 12 (invalid; actually native K=8) | 93.80 ± 0.20 | 80.98 ± 0.29 | 71.33 ± 0.60 | 60.87 ± 0.22 |

The valid comparison currently ends at native K=8. IFBench and GPQA improve
from lower K toward K=8; MATH-500 and AIME show the opposite nominal trend in
this sample. No conclusion about a K=8--12 plateau can be drawn from these
historical outputs because nominal K=10/12 were actually native-K=8 runs.

Every active/completed run emitted the same item-level scorer warning for
`ifeval_mt_wildchat_unused_withRewrite[439]` (`ParagraphFirstWordCheck` received an unsupported
`keyword` argument). The aggregate `metrics.json` files still contain zero run-level errors and
report all 4,189 instances. Keep this item-level evaluator defect documented and consistent across
K rather than interpreting it as a model failure.
