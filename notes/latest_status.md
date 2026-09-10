# Latest adaptive-expert experiment status

Checked: 2026-08-19 15:55 UTC

## 2026-08-13 live eval audit

- Qwen3.6 capability suite: K=4/K=6 are complete at three replicates each; two of three native-K=8
  replicates are also complete and collected. The plot and CSV now contain K={4,6,8}. Seven jobs
  remain running and none are queued: K=8 replicate 3 is ~92%; K=10 replicates are ~94/92/90%;
  K=12 replicates are ~92/87/45%. No job has crashed or OOMed.
  There are no OOMs, model-server deaths, or tracebacks. Every active job reports the same isolated
  scoring failure for `ifeval_mt_wildchat_unused_withRewrite[439]`; that malformed IFBench scorer
  instance must be treated as invalid when results are aggregated.
- Terminal-Bench 2.0 K=12: all 10 full-suite shards completed successfully. The separate K=8
  published-settings control is also complete: 28/89 correct (31.46%) with 23 task exceptions.
- Canonical Terminal-Bench 2.1 K=8: both corrected full-suite shards are allocated on Holmes. The
  first Holmes pair used unprefixed globs; shard A therefore matched zero tasks and shard B matched
  all 89. Shard B was stopped after two trials, the filters were corrected, and the replacement
  shards are now starting/running as `01KZWW95RXHDKQ39D527QWXN8V` and
  `01KZWW9BSV4H502N2247XV2RS8`. Both replacements completed successfully: 31/89 correct (34.83%)
  with 38 task exceptions. The failed original launch was a Harbor filtering issue, not a
  CUDA/B300 compatibility failure.
- RL checkpoint evaluation refresh: all expected late-checkpoint MATH-500/AIME artifacts are
  complete and collected. One original Beaker job is marked canceled, but its complete result
  bundle is present and validated, so there is no missing checkpoint evaluation.

## 2026-08-12 Qwen3.6 Terminal-Bench K=12 launch

The full five-run Terminal-Bench 2.0 evaluation was launched for Qwen3.6 at K=12, matching the
completed K=10 recipe. Each replicate consists of complementary 45-task and 44-task eight-GPU
shards, for ten jobs and 445 total task attempts. Replicate 1 is unseeded under the original
convention; replicates 2--5 use the same paired seeds 4202--4205 as K=4/6/8/10. K=12 is
reference-scaled to native K=8 without renormalizing all twelve weights. All ten jobs were queued
at urgent priority in `ai2/olmo-instruct` on `ai2/jupiter` and are collected in
[group 01KZVM534SVXEQ02S9Q799E8WW](https://beaker.org/orgs/ai2/workspaces/olmo-instruct/groups/01KZVM534SVXEQ02S9Q799E8WW).

## 2026-08-12 late-checkpoint result collection

All 39 jobs in the late-checkpoint evaluation refresh succeeded and were downloaded into
`results/rl_checkpoint_evals`. The collection contains 13 combined seed-42 MATH-500 + AIME
pass@32 evaluations and 26 seed-43/44 MATH-only replicates. Validation found all 500 MATH-500
instances and all 30 AIME problems in every expected task, with no metric errors.

The three source tables and their MATH/AIME plots were regenerated, followed by the matched
adaptive-strategy comparison plot. The neutral even-K={4,6,8} curves now extend through step 500,
the original fixed-K=8 curve through step 500, and the even-K={4,8,12} curve through step 400.
At the newly added endpoints, the even-K={4,6,8} step-500 MATH means are 81.87%/83.40%/83.33%
at reference-scaled K=4/K=6 and native K=8; the even-K={4,8,12} step-400 means are
81.80%/84.73%/83.40% at reference-scaled K=4, native K=8, and reference-scaled K=12. The fixed
K=8 step-500 checkpoint scores 81.80% MATH-500 (three-run mean), 24.06% AIME Pass@1, and 41.73%
AIME Pass@8. The latter AIME point is a single pass@32 evaluation, like the other AIME points.

## 2026-08-12 late-checkpoint export and evaluation refresh

The local/Beaker result audit found zero successful Qwen adaptive-RL evaluations missing from
`results/rl_checkpoint_evals`; there were therefore no newly completed artifacts to download at
the start of this refresh.

Previously unevaluated checkpoints were exported successfully:

- neutral even K={4,6,8} steps 400/450/500:
  [01KZTA18V55MCGRHSFCHK8AE7N](https://beaker.org/ex/01KZTA18V55MCGRHSFCHK8AE7N)
- original fixed-K=8 step 500:
  [01KZTA441Y059HC2D07DX9M571](https://beaker.org/ex/01KZTA441Y059HC2D07DX9M571)
- neutral even K={4,8,12} step 400:
  [01KZVAFQEBH1973N0H1XDA2X47](https://beaker.org/ex/01KZVAFQEBH1973N0H1XDA2X47)

For every newly available checkpoint, the matched evaluation recipe was submitted: one combined
seed-42 MATH-500 + AIME 2025 pass@32 job and seed-43/44 MATH-only jobs, giving three MATH-500
samples and one AIME pass@32 evaluation per inference condition. The even-K checkpoints use all
of their trained routing arms (K={4,6,8} or K={4,8,12}); fixed K=8 uses native K=8. This is 13
checkpoint/routing conditions and 39 eval jobs total. All use the validated 20,480-token cap,
32,768-token context, four TP=1 vLLM engines, temperature 0.6/top-p 0.95/top-k 20, urgent
unallocated scheduling in `ai2/holmes-testing`, and the multi-cluster `80g` alias.

- combined MATH+AIME group:
  [01KZVAG9NR3P5HPHNDAH2GXZNE](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KZVAG9NR3P5HPHNDAH2GXZNE)
- MATH replicate group:
  [01KZVAKANVNDGYJTVDN42EEVN4](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KZVAKANVNDGYJTVDN42EEVN4)

At the post-launch check, 8/13 combined jobs were running and 5/13 queued; all 26 MATH-only
replicate jobs were queued. No submitted job had failed. The active even K={4,8,12} trainer has
also written a complete step-400 checkpoint and reached optimizer step 420, although its rollout
length pathology continues to make recent updates take roughly 20--33 minutes.

## 2026-08-12 K={4,8,12} late-training rollout slowdown

The neutral K={4,8,12} continuation remains healthy at the learner level, but its K=4 rollout
arm developed a severe answer-repetition/termination failure beginning around steps 342--350.
Mean K=4 response length rose from 680 tokens at step 325 to 5,608 at step 350 and 12,065 at
step 370; the corresponding K=8/K=12 means at step 370 were only 1,203/1,213. At step 384,
14/48 K=4 responses exceeded 8,192 tokens and one reached the 20,480-token cap. Direct response
inspection shows repeated strings such as `Answer: 105` continuing for thousands of tokens; some
of these still receive a correct reward, so the current reward does not discourage the failure.

This explains the approximately 31-minute recent optimizer steps. Step 384 spent 1,822 seconds
waiting for rollout generation and 64 seconds in learner work (`wait_time_ratio=0.966`); its
144-response rollout took 29:27. Because the mixed batch synchronizes across all K arms, K=4's
long tail gates the entire step. This is not a K=12 compute bottleneck or a learner-side stall.

The adaptive-strategy comparison plot now also includes the original fixed-K=4 reference-scaled
and fixed-K=8 native RL baselines in their matched inference columns. The older even K={4,6,8}
line ends at step 350 because only checkpoints
through 350 were exported/evaluated, although training completed through step 500. A one-GPU
urgent export for steps 400/450/500 was submitted as
[01KZTA18V55MCGRHSFCHK8AE7N](https://beaker.org/ex/01KZTA18V55MCGRHSFCHK8AE7N); once exported,
these checkpoints can receive the same 3x MATH-500 plus 1x AIME evaluation matrix.
The original fixed-K=8 baseline likewise has an unevaluated step-500 checkpoint; its export was
submitted as [01KZTA441Y059HC2D07DX9M571](https://beaker.org/ex/01KZTA441Y059HC2D07DX9M571).

## 2026-08-10 preallocated-K DAPO pilot

The Qwen3-30B-A3B Base cheapest-any-success prompt-preallocation pilot is queued/running as
[Beaker 01KZN5GJF4YKWES52H9T6N98FK](https://beaker.org/ex/01KZN5GJF4YKWES52H9T6N98FK). It uses
one 8xB200 Titan node, neutral DAPO rewards, reference-scaled K={4,6} and native-equivalent K=8,
eight siblings at one fixed assigned K per prompt, 100 updates, and checkpoints every 20 steps.
The immutable assignment is 4,575 / 2,469 / 10,347 prompts at K=4 / K=6 / K=8 (mean K=6.664).

## 2026-07-28 K=6 reference-scaled completion and evaluation launch

The K=6 reference-scaled DAPO arm completed all 500 updates successfully at 03:46 UTC. Its
steps 350/400/450/500 were exported together in a successful one-GPU Titan job:
[Beaker 01KYKFWV03TPCCYWN12RMBPMVX](https://beaker.org/ex/01KYKFWV03TPCCYWN12RMBPMVX).
Four native-policy MATH-500 + AIME 2025 pass@32 evaluations using the validated native-context
recipe were launched at urgent
priority in `ai2/holmes-testing` on `ai2/jupiter`:

- [step 350](https://beaker.org/ex/01KYKGEJCPT88B1NCEB5JBPZD9)
- [step 400](https://beaker.org/ex/01KYKGHB885D2HD5DMSBHVN4P4)
- [step 450](https://beaker.org/ex/01KYKGJG1AZ3PBS40DDF4SCDXD)
- [step 500](https://beaker.org/ex/01KYKGR4JXJSC30W0SZ2X8EB8J)

At the final post-launch check, all four were running and none had failed. K=8 normalized remains
the only active training arm, at step
437 with step 400 as its latest finalized checkpoint.

## 2026-07-28 live-job audit

Two RL jobs remain active and healthy. Native normalized K=8 reached optimizer step 433 and still
has step 400 as its latest finalized checkpoint; K=6 reference-scaled reached step 489 and has
finalized checkpoints through step 450. K=6 normalized and K=4 reference-scaled both completed
500 steps successfully. No checkpoint evaluations are currently running or queued. The newly
available but unevaluated checkpoints are K=6 reference-scaled steps 350/400/450; its step-500
checkpoint should follow after roughly ten more updates.

The four-B200 Qwen3.5 hybrid-attention router/shared-expert profile also completed successfully.
All 60 planned trajectories are present (20 each from MATH-500, GPQA Diamond, and IFEval OOD), and
the prompt-balanced and token-weighted plots/tables have been generated under
`notes/qwen35_router_shared/`.

## 2026-07-27 K=8 step-400 collection and new checkpoints

Both K=8 step-400 evaluations succeeded and were collected. Native K=8 scores 83.0% on
MATH-500, 16.15% AIME Pass@1, and 27.21% AIME Pass@8. Counterfactual normalized-K=6 inference
scores 84.8%, 15.73%, and 28.84%, respectively. Both bundles are complete and have no metric
errors. The cumulative checkpoint CSV now contains 73 rows, and all three RL score plots plus the
observed-compute plot have been regenerated.

Two additional distributed checkpoints are now ready but not yet exported or evaluated: K=6
reference-scaled steps 350 and 400. The K=8 run is live at optimizer step 414 with step 400 as its
latest checkpoint; K=6 reference-scaled is live at step 404 with step 400 as its latest checkpoint.
K=6 normalized and K=4 reference-scaled completed all 500 steps successfully.

## 2026-07-27 observed RL compute-cost accounting

An all-observed compute-cost figure and source table were added for the four SGLang/Megatron RL
arms and their checkpoint evaluations. Training cost is measured from finalized checkpoint
timestamps as B200 wall time times eight allocated GPUs, normalized to 50 updates; the idle gap
between the original and resumed jobs is excluded. Eval cost uses recorded olmo-eval experiment
duration times four H100s, both as total GPU-hours and normalized by recorded completion tokens.
The figure contains 38 continuous training intervals and all 56 checkpoint evals that retained
vLLM inference telemetry; no theoretical FLOP estimates are mixed into it.

Early training intervals are broadly around 20--25 GPU-hours per 50 updates across arms. The K=8
run rises sharply after step 250 (46.6, 89.9, and 85.2 GPU-hours per 50 updates for the intervals
ending at steps 300/350/400), reflecting its observed rollout-length/behavioral slowdown as well
as engine cost. K=6 normalized and K=4 reference-scaled also become more expensive late, but less
dramatically. Token-normalized eval cost remains noisy and does not yet show a clean monotonic K
speedup. The reference-scaled eval curves are explicitly labeled as K=8 zero-mask compute, even
though their RL training and SGLang rollout kernels really dispatch K=6 or K=4.

## 2026-07-27 K=8 step-400 evaluation

The live K=8 normalized DAPO run wrote its step-400 distributed checkpoint. Its immutable HF
export succeeded on one Titan GPU:
[Beaker 01KYJED1HFJYT928JER7SM3MD3](https://beaker.org/ex/01KYJED1HFJYT928JER7SM3MD3).
Two corrected MATH-500 + AIME 2025 pass@32 evaluations succeeded in the existing cross-routing
group: [native K=8](https://beaker.org/ex/01KYJER2PV95MEETA2HWX1F3S3) and
[counterfactual normalized K=6](https://beaker.org/ex/01KYJER928YCQB3MBBYG1H0XZP). Both use four
TP=1 engines, the 32,768-token context cap, and 20,480-token generation caps. They were collected
and incorporated into the 73-row checkpoint table and refreshed plots.

## 2026-07-27 newly available RL checkpoints

Eight newly available distributed checkpoints were exported successfully in four urgent,
one-GPU Titan jobs: K=8 normalized step 350; K=6 normalized steps 450/500; K=4
reference-scaled step 500; and K=6 reference-scaled steps 150/200/250/300. The corresponding
11 single-replicate MATH-500 + AIME 2025 pass@32 evaluations are queued at urgent priority in
`ai2/holmes-testing` on `ai2/jupiter`:

- K=8 normalized step 350 at native K=8 and counterfactual K=6;
- K=6 normalized steps 450/500 at native K=6 and counterfactual K=8;
- K=4 reference-scaled step 500 at native K=4; and
- K=6 reference-scaled steps 150/200/250/300 at native K=6.

All 11 are in [the existing cross-routing Beaker group](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KYGB9X18QP6ZD4QHBRAVHHR6)
and individually recorded in `beaker_jobs.jsonl`. All 11 subsequently succeeded and were
downloaded. Every bundle has 500 MATH-500 instances, 30 AIME problems with 32 samples each, and
an empty metric-error list. The first K=6-reference step-150 allocation was automatically canceled
because its node was cordoned after an SXid hardware error; Beaker rescheduled it onto a healthy
node and that attempt succeeded. This was not an eval/configuration failure.

The cumulative checkpoint table now has 71 rows, and the MATH-500, AIME Pass@1, and AIME Pass@8
plots have all been regenerated. Among the new late checkpoints, K=6 normalized scores 84.4% and
83.0% MATH-500 at steps 450/500; evaluating those checkpoints at K=8 gives 85.4% and 84.0%.
K=4 reference-scaled reaches 78.2% at step 500, partially recovering from 73.2%/76.0% at steps
400/450. K=6 reference-scaled is noisier through steps 150--300, ranging from 76.0% to 80.8%.

## 2026-07-27 complete cross-routing collection

All 23 pending Qwen3-base RL evaluations succeeded: the full 19-job cross-routing matrix plus
native K=8 step 300, K=6 step 350, and K=4-reference-scaled steps 350/400. Twenty newly available
bundles were downloaded in this collection pass; every bundle has a complete 500-instance
MATH-500 task, a complete 30-problem AIME 2025 pass@32 task, and no metric errors. The cumulative
CSV reached 55 rows in that pass, and all three plots were refreshed.

On MATH-500, the K=8-trained model becomes increasingly robust to K=6 inference: the K=6 penalty
falls from 7.6 points at step 0 to 0.4 at step 100 and is within -2.4 to +1.4 points thereafter.
For the K=6-trained model, switching inference to K=8 is also small by steps 100--350, although it
helps by 2.6--5.0 points at several earlier checkpoints. AIME Pass@8 is substantially noisier and
does not show a consistent direction from either inference-K swap. The K=4-reference-scaled
step-400 MATH-500 score drops to 73.2% from 81.4% at step 350; this is a notable single-evaluation
point and was checked against step 450 before treating it as a real training collapse.

The step-450 K=4 evaluation subsequently succeeded and was collected, bringing the CSV to 56
rows. MATH-500 partially recovered to 76.0%; AIME Pass@1 recovered from 11.35% to 13.65%, and
Pass@8 from 24.47% to 28.42%. The step-400 result was not an evaluation failure: it used the same
configuration as step 350, completed every instance, and had no metric errors. Instead, the model
developed a degenerate long-output tail. On MATH-500, mean output length went 917 -> 2,223 ->
1,817 tokens at steps 350/400/450, max-length outputs went 6 -> 32 -> 19, and invalid/missing
answers went 2 -> 19 -> 15. The same pattern appears on AIME. Training logs independently show
rising response length, falling policy entropy, and increasing reference-policy divergence after
step 350, without an OOM, NaN, or checkpoint-export error. This is best described as real
late-training instability or partial mode/length collapse, not total model collapse.

Newly appeared checkpoints are being caught up: K=6 normalized step 400, K=4 reference-scaled
step 450, and K=6 reference-scaled steps 50/100. Their one-GPU Titan export jobs all succeeded:
[K=6 step 400](https://beaker.org/ex/01KYGWPTW0BG89TC9ZCF6W2F16),
[K=4 step 450](https://beaker.org/ex/01KYGWQ3W9RATKB7DE9R8Y0E4W), and
[K=6 reference-scaled steps 50/100](https://beaker.org/ex/01KYGWQC2FA8SPJQ72CFCPBV1K).
The same corrected MATH-500 + AIME recipe is now queued for all four native points, plus
K=6-trained step 400 evaluated at K=8:

- K=6 native step 400: [Beaker 01KYGX0QQQFZV9YBQCJ8MWG33B](https://beaker.org/ex/01KYGX0QQQFZV9YBQCJ8MWG33B)
- K=6-trained step 400, evaluated at K=8: [Beaker 01KYGX0XXSB8CK927H4DY2AXXN](https://beaker.org/ex/01KYGX0XXSB8CK927H4DY2AXXN)
- K=4 reference-scaled step 450: [Beaker 01KYGX1X1KKJ48TXNR5YEEX4W4](https://beaker.org/ex/01KYGX1X1KKJ48TXNR5YEEX4W4)
- K=6 reference-scaled step 50: [Beaker 01KYGX29390BMPHCPCJ98A4W83](https://beaker.org/ex/01KYGX29390BMPHCPCJ98A4W83)
- K=6 reference-scaled step 100: [Beaker 01KYGX7A8BZ1EVH6W82DD45DGF](https://beaker.org/ex/01KYGX7A8BZ1EVH6W82DD45DGF)

## 2026-07-26 K=6 reference-scaled RL and base baselines

A fresh 500-step DAPO arm was launched on one 8xB200 Titan node for K=6 with
K=8-reference-scaled weights: [Beaker 01KYGAW9SY51BFRACV3PP4053D](https://beaker.org/ex/01KYGAW9SY51BFRACV3PP4053D).
It uses the same settings as the other three arms and saves every 50 steps. Four step-0 MATH-500
+ AIME 2025 evaluations were launched in `ai2/holmes-testing` on `ai2/jupiter` for K=8
normalized, K=6 normalized, K=6 reference-scaled, and K=4 reference-scaled. All four succeeded
and were collected. All five experiments are recorded in `beaker_jobs.jsonl`; links and the
chat-template equivalence check are documented in `slime_qwen3_base_dapo_plan.md`.

Nineteen cross-routing checkpoint evaluations were queued in `ai2/holmes-testing`: K=8
trained/K=6 evaluated through step 300 and K=6 trained/K=8 evaluated through step 350. The K=8
step-20 and step-40 K=8-to-K=6 cross evaluations, plus the step-40 K=6-to-K=8 evaluation, had
already succeeded in the first collection pass. Four newly
completed native checkpoints (K=8 step 300, K=6 step 350, and K=4-reference-scaled steps 350/400)
were exported on three one-GPU Titan jobs. The
checkpoint plotter keeps color keyed to the training arm and uses dashed, hollow-marker lines for
the inference-K swap. All of these evaluations are now complete and collected.

## 2026-07-26 incremental collection

Twenty-five newly successful result bundles were downloaded and validated:

- the remaining 14 valid-v4 Qwen3.5 K=4--8 normalized/reference-scaled evaluations, completing
  that matrix at 30/30 jobs and three replicates per condition; and
- 11 continuation-checkpoint evaluations for the Qwen3 base DAPO runs, covering K=8 through step
  250 and K=6 plus K=4-reference-scaled through step 300.

All 30 Qwen3.5 bundles contain six task rows and 4,323 instances, with empty metric-error lists.
The final score CSV and plot are
[qwen35_35b_a3b_results.csv](qwen35_35b_a3b_results.csv) and
[qwen35_35b_a3b_expert_sweep.png](plots/qwen35_35b_a3b_expert_sweep.png).

All 11 new RL-evaluation bundles contain complete 500-example MATH-500 and 30-problem AIME 2025
pass@32 tasks, with empty metric-error lists. The cumulative table now has 26 checkpoint rows in
[slime_qwen3_base_dapo_checkpoint_evals.csv](slime_qwen3_base_dapo_checkpoint_evals.csv), and all
three RL plots extend through the latest available steps. There is no K=8 step-300 point because
that checkpoint was not available when the continuation evaluations were launched.

Four relevant jobs remain active: the three step-100-to-500 RL continuations and the Qwen3.5
router/shared-expert profile. The latter has completed 30 of 60 trajectories without an error;
several MATH prompts legitimately ran to the 32,768-token cap. No new production failure appeared
in this refresh.

## Newly launched work

- The one-H100 Nemotron router/contribution aggregate
  [succeeded](https://beaker.org/ex/01KY61MX7B4T9WCZ4W41MT2WRA), was downloaded, and produced all
  19 expected artifacts.
- The initial Qwen graph-mode smoke passed, but production concurrency exposed a CUDA-graph capture
  incompatibility in the realized-K telemetry hook. The 12 graph-mode production attempts were
  stopped or failed and are marked superseded rather than treated as results.
- The corrected one-H100 Qwen non-renormalized adaptive-mass
  [eager-mode smoke](https://beaker.org/ex/01KY62BT5MEDS1X0SN46FD5PPA) succeeded with three
  predictions, no metric errors, valid telemetry for all 48 layers, and no CUDA capture error.
- Twelve corrected full Qwen evaluations were launched at tau={0.5,0.6,0.7,0.8}, three replicates
  each, in [this Beaker group](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KY62MA9VXSX3MQTT04M6974V).
  Every job is in `ai2/holmes-testing`, urgent, unallocated on `ai2/jupiter`, with four independent
  one-H100 TP=1 engines, reference-preserving retained weights, eager mode, and realized-K
  recording. Eleven succeeded and are downloaded. Tau=0.50 replicate 3 was preempted with exit
  code 143 before completion, leaving n=2 at tau=0.50 and n=3 at every other threshold. Nothing is
  running or queued.

Final macros are 68.63 ± 0.99 at tau=0.50 (mean K=3.498), 73.66 ± 1.16 at tau=0.60
(K=4.271), 74.08 ± 0.67 at tau=0.70 (K=5.144), and 76.52 ± 0.31 at tau=0.80 (K=6.107).
Full score, telemetry, response diagnostics, and plots are in
[the threshold sweep note](qwen_reference_adaptive_threshold_sweep.md).

## Full ledger audit

The historical reconciliation below predates today's launches. The append-only ledger now contains
793 rows representing 790 unique Beaker experiments across
`ai2/olmo-instruct`, `ai2/holmes-testing`, and `ai2/OLMo-3-moe-experiments`. A fresh reconciliation
before today's work found:

| State | Experiments |
|:--|--:|
| Succeeded | 533 |
| Failed | 61 |
| Canceled | 63 |
| Running | 0 |
| Queued/created | 0 |
| Deleted historical IDs | 106 |
| **Total** | **763** |

An experiment is classified as succeeded if any execution inside it succeeded, which correctly
handles Beaker restarts after a failed first execution. The failed and canceled totals are
historical serving smokes, superseded matrices, conversion diagnostics, and replaced attempts. No
experiment's latest failed execution is newer than 2026-07-19 16:35 UTC, so there is no new
production failure in this refresh.

## Active production matrices

| Matrix | Succeeded | Running | Queued | Failed |
|:--|--:|--:|--:|--:|
| Qwen adaptive mass | 15 | 0 | 0 | 0 |
| Qwen K=8-reference scaled | 36 | 0 | 0 | 0 |
| Qwen normalized current-suite backfill | 36 | 0 | 0 | 0 |
| GPT-OSS K=4-reference-scaled group | 22 | 0 | 0 | 0 |
| GPT-OSS normalized current-suite backfill | 24 | 0 | 0 | 0 |
| Corrected Qwen/Dolci terminal-EOS-v2 sweep | 60 | 0 | 0 | 0 |

The GPT-OSS reference-scaled group has 22 successes because the separately submitted K=8
replacement also succeeded after the original experiment's in-place restart. The replacement is
valid but redundant; plots and tables retain the planned three K=8 replicates, not four.

All older production matrices remain terminal and collected. The only valid active production
matrix is the corrected Qwen reference-preserving adaptive-threshold sweep above; the 12
superseded graph-mode attempts are terminal and excluded from analysis.

## Collection and outputs

The earlier 2026-07-22 refresh downloaded and validated 30 newly successful bundles:

- 11 corrected Qwen/Dolci bundles, completing its 60/60 normalized/reference-scaled K=3--12 sweep;
- 13 GPT-OSS reference-scaled bundles, completing the planned K=1--8 curve and also collecting the
  redundant K=8 replacement; and
- six GPT-OSS normalized bundles, completing K=3 and K=5.

This final refresh downloaded and validated the remaining six normalized GPT-OSS K={6,7}
bundles. Each contains all four expected tasks, 3,625 matched request/prediction rows, and an empty
metric-error list. The approved 132-job expanded matrix is therefore fully collected.

Every new bundle has an empty metric-error list, identical request/prediction task-key sets, and
positive row counts matching each configured task exactly. The regenerated plots and CSV include
all valid planned replicates and exclude the redundant fourth GPT-OSS K=8 run.

Updated outputs:

- [expanded sweep results](expanded_routing_sweeps_launch.md)
- [corrected Qwen/Dolci sweep](corrected_dolci_terminal_eos_v2_sweep.md)
- [Qwen current-suite plot](plots/qwen3_hybrid_current_suite_expert_sweep.png)
- [GPT-OSS current-suite plot](plots/gptoss_120b_current_suite_expert_sweep.png)
- [corrected Qwen/Dolci current-suite plot](plots/qwen3_dolci_terminal_eos_v2_current_suite_expert_sweep.png)
- [current-suite score CSV](current_suite_expert_sweeps.csv)
- [project overview](project_overview.md)

## 2026-07-28 RL-checkpoint MATH-500 replication sweep

Two additional MATH-500-only evaluations were queued for every distinct RL checkpoint/routing
condition so that, together with the MATH task in the original checkpoint evaluation, each point
will have three stochastic runs. The matrix contains 75 distinct conditions and 150 new jobs:

| Condition family | Checkpoints | New jobs |
|:--|--:|--:|
| K=8 normalized | 12 | 24 |
| K=8-trained, evaluated at normalized K=6 | 11 | 22 |
| K=6 normalized | 14 | 28 |
| K=6-trained, evaluated at normalized K=8 | 13 | 26 |
| K=4, K=8-reference-scaled | 14 | 28 |
| K=6, K=8-reference-scaled | 11 | 22 |
| **Total** | **75** | **150** |

Step-zero cross-routing aliases are not duplicated: the K=8-trained/K=6-eval baseline reuses the
K=6-normalized base point, and the K=6-trained/K=8-eval baseline reuses the K=8-normalized base
point. All jobs use `math500:chat`, the existing temperature-0.6 sampling recipe, native 32,768
context, a 20,480-token response cap, four independent TP=1 H100 engines, urgent priority, and
`ai2/jupiter` in `ai2/holmes-testing`. They are collected in
[Beaker group 01KYKJ2J1AAE8V0MGHG7HGK9GN](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KYKJ2J1AAE8V0MGHG7HGK9GN).

The launch audit found exactly 150 unique tracked tags and 150 unique experiment IDs, split 75/75
between replicates 2 and 3, with no missing or unexpected conditions. Within the first few
minutes, seven jobs were running, 143 were queued, and none had failed. The first live job brought
up all four vLLM engines and began its two 64-item batches per engine successfully. Every job is
recorded under phase `slime-checkpoint-math500-replicate` in `notes/beaker_jobs.jsonl`.

All 150 replication jobs subsequently succeeded and were downloaded on 2026-07-28; none failed.
The four outstanding original K=6-reference-scaled evaluations at steps 350/400/450/500 also
succeeded and were collected. Validation found exactly 75 real checkpoint/routing conditions,
each with one original 500-example MATH-500 result plus both new 500-example replicates, empty
metric-error lists, and no missing or extra keys. The two cross-routing step-zero aliases reuse
their corresponding three-run base-policy aggregates, producing 77 plotted rows.

The MATH-500 checkpoint plot now uses the arithmetic mean of the three independent evaluations at
every point and shows ±1 sample-standard-deviation whiskers. Across the 75 real conditions, the
median within-point sample SD is 0.92 percentage points, the mean is 0.97 points, and the maximum
is 2.34 points. AIME remains the original one-evaluation pass@32 estimate because it was not part
of the replication sweep. Updated outputs are
[the MATH-500 plot](plots/slime_qwen3_base_dapo_rl_math500_by_step.png),
[the checkpoint CSV](slime_qwen3_base_dapo_checkpoint_evals.csv), and the unchanged-definition
[AIME Pass@1](plots/slime_qwen3_base_dapo_rl_aime_by_step.png) and
[AIME Pass@8](plots/slime_qwen3_base_dapo_rl_aime_pass_at_8_by_step.png) plots.

## 2026-07-28 K=8-trained step-450 evaluation launch

The K=8 training arm's distributed step-450 checkpoint was exported successfully to
`slime-runs/qwen3-30b-a3b-base-dapo/hf-checkpoints/slime-qwen3-base-dapo-k8-pilot-100step-8gpu-v5-20260725-step450-hf`
by [Beaker experiment 01KYMSVTYJ6NYJN7VWC53TNZBW](https://beaker.org/ex/01KYMSVTYJ6NYJN7VWC53TNZBW).
The export contains all 12 expected safetensor shards and a complete weight index.

Both routing conditions used for the existing cross-routing comparison were launched:

| Trained policy | Evaluation policy | MATH-500 runs | AIME runs | Combined experiment |
|:--|:--|--:|--:|:--|
| normalized K=8 | normalized K=8 | 3 | 1 | [01KYMT3JFT4BFRQ1AV1JNJSB7Y](https://beaker.org/ex/01KYMT3JFT4BFRQ1AV1JNJSB7Y) |
| normalized K=8 | normalized K=6 | 3 | 1 | [01KYMT3RN1ZA1G5123QWJ4122T](https://beaker.org/ex/01KYMT3RN1ZA1G5123QWJ4122T) |

Each combined experiment runs one MATH-500 evaluation and one AIME 2025 pass@32 evaluation. The
two additional MATH-only replicates per routing condition are
[K=8 replicate 2](https://beaker.org/ex/01KYMT45JBD7SXWM4BKK6SBV9X),
[K=8 replicate 3](https://beaker.org/ex/01KYMT47QRVY8CVQ7A3VG2JMFT),
[K=6-eval replicate 2](https://beaker.org/ex/01KYMT46RD62TARQ3V6BZRXWV0), and
[K=6-eval replicate 3](https://beaker.org/ex/01KYMT48Z44RJPGZTJMG4MW16B).
All six jobs use four independent TP=1 H100 engines, 32,768-token native context, a 20,480-token
response cap, urgent priority, and `ai2/jupiter` in `ai2/holmes-testing`. They were queued at the
post-launch audit with no failed execution. Exact commands and IDs are recorded in
`notes/beaker_jobs.jsonl`.

## 2026-07-28 RL step-350 fixed-K plateau sweep

A full inference-time fixed-K plateau sweep was launched for three matched step-350 DAPO
checkpoints:

- normalized-K=8 training;
- K=8-reference-scaled K=6 training; and
- K=8-reference-scaled K=4 training.

For every checkpoint, the sweep covers every integer K from 2 through 12 under both normalized
and K=8-reference-scaled inference routing, with three independent evaluations per point. The
reference-scaled K=8 point is exactly identical to normalized K=8 and is therefore evaluated once
and reused in both plotted curves. This yields 66 logical conditions per replicate set and 189
unique Beaker experiments: 63 per trained checkpoint.

Every experiment bundles the current four-metric suite: MATH-500, GPQA Diamond, IFBench (all three
subtasks), and HumanEval. The initial jobs used native 32,768-token context, a 20,480-token
response cap, four independent TP=1 H100 engines, urgent priority, `ai2/jupiter`, and
`ai2/holmes-testing`. They also explicitly enabled the Qwen3 reasoning parser; the 2026-07-29
collection audit below found that this was incompatible with the RL checkpoints. The invalid
experiments are in
[Beaker group 01KYN5JCZTQB4XZ1J1PYZ8X9GE](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KYN5JCZTQB4XZ1J1PYZ8X9GE).

The launch audit found exactly 189 expected unique tags and experiment IDs, with no missing or
unexpected conditions. At the first post-launch status check, one job was running and 188 were
queued; none had failed. The running normalized-K=2 job brought up all four vLLM engines in about
100 seconds and began its 4,323-request workload successfully. Exact IDs and commands are tracked
under phase `slime-rl-step350-plateau` in `notes/beaker_jobs.jsonl`.

## 2026-07-29 collection and corrected plateau relaunch

Six newly completed K=8-trained step-450 bundles were downloaded and validated. Both routing
conditions have three complete 500-example MATH-500 evaluations and one complete AIME 2025
pass@32 evaluation, with empty metric-error lists:

| Evaluation policy | MATH-500 mean ± SD | AIME Pass@1 | AIME Pass@8 | AIME Pass@32 |
|:--|--:|--:|--:|--:|
| normalized K=8 | 85.33 ± 1.10 | 20.21 | 34.56 | 46.67 |
| normalized K=6 | 84.93 ± 0.99 | 18.96 | 32.91 | 43.33 |

The checkpoint CSV and MATH-500, AIME Pass@1, and AIME Pass@8 plots were regenerated through
step 450.

The first plateau-group collection exposed a serving error. All 61 completed bundles were
downloaded and were structurally complete—4,323 instances, all six underlying tasks, and no
metric errors—but every metric was zero. Prediction inspection showed positive generated-token
counts with empty returned `text` fields. Comparing Beaker specs against the known-good
checkpoint evaluations isolated the difference to
`provider.kwargs.reasoning_parser=qwen3`: these RL checkpoints do not emit a response structure
that the server-side parser can split correctly. The 61 zero-score bundles are invalid and
excluded from all analysis. The remaining 128 jobs were canceled, including eight that had
already started.

A one-GPU corrected smoke removed the reasoning parser while preserving the model, chat template,
and all other serving settings. It produced non-empty outputs for every task and completed
successfully in
[Beaker 01KYQ8HEFV00G4D0RWGRB407A7](https://beaker.org/ex/01KYQ8HEFV00G4D0RWGRB407A7).
The complete 189-job matrix was then relaunched with unique `v2-no-parser` tags in
[corrected Beaker group 01KYQ8RJ0NPKZQN6X6W9KY126G](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KYQ8RJ0NPKZQN6X6W9KY126G).
The corrected launch audit found exactly 189 expected unique IDs and tags, 63 per checkpoint.
At the final status check, eight jobs were running, 181 were queued, and none had finished or
failed.

## 2026-07-29 MATH-500-only plateau correction

The user clarified that the step-350 inference-time plateau sweep should measure MATH-500 only,
because all three checkpoints were trained with math RL. All 189 jobs in the corrected
four-eval-suite group were therefore canceled before collecting any production results.

The replacement sweep preserves the exact settings used by the existing RL MATH plots:
`math500:chat`, temperature 0.6, top-p 0.95, top-k 20, one sample, a 20,480-token response cap,
32,768-token context, four TP=1 vLLM engines, the same immutable source snapshot, and no
server-side reasoning parser. A direct Beaker-spec comparison against the prior step-350 jobs
confirmed that the only task-level change is omitting AIME and the unrelated evaluation suite.

Four model/policy/K conditions already had three valid MATH-500 replicates and are reused:

- K=8-normalized checkpoint at normalized K=8;
- K=8-normalized checkpoint at normalized K=6;
- K=6-reference-trained checkpoint at reference-scaled K=6; and
- K=4-reference-trained checkpoint at reference-scaled K=4.

This avoids 12 redundant jobs. Three K=8-checkpoint/normalized-K=6 duplicates were submitted
before the reuse audit and were immediately canceled. The final replacement matrix therefore
contains 177 active unique MATH-only jobs plus those 12 reused prior results. It is tracked under
phase `slime-rl-step350-plateau-math500` in `notes/beaker_jobs.jsonl` and grouped in
[Beaker group 01KYQA30P3M92HFVVJ0QAHAGRY](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KYQA30P3M92HFVVJ0QAHAGRY).
At the post-launch audit, eight jobs were running, 169 were queued, and the only three canceled
jobs were the identified duplicates.

## 2026-07-29 partial MATH-500 plateau collection

The first 117 successful nonduplicate MATH-only jobs were downloaded and validated. Every bundle
contains exactly 500 predictions, an empty metric-error list, and non-empty generated text for
all 500 examples. Combined with the 12 prior evaluations being reused, the partial dataset has
45 complete three-replicate logical points and two partial points. Replicate-level and
aggregated data are saved in
`slime_qwen3_base_dapo_step350_plateau_replicates.csv` and
`slime_qwen3_base_dapo_step350_plateau_summary.csv`; the partial plot is
`plots/slime_qwen3_base_dapo_step350_math500_plateau.png`.

The partial curves already show a large distinction between the inference policies. Normalized
inference nearly collapses at K=2 (0.07--0.20% mean across the three trained checkpoints), rises
to 32.07--38.07% at K=3 and 66.47--71.53% at K=4, and reaches its plateau around K=5--7.
Reference-scaled inference retains 47.80--54.20% at K=2, 73.53--76.60% at K=3, and approximately
80--81% at K=4 where complete or partially available. The remaining reference-scaled points are
still pending, so this interpretation is provisional.

At the collection snapshot, 117 jobs had succeeded, eight were running, 52 were queued, and the
three deliberately canceled duplicate jobs were the only canceled experiments.

The normalized-routing arm was retired after the partial results made the low-K inferiority
clear. A live cancellation audit found no unfinished normalized jobs: 93 new normalized jobs had
already succeeded, three duplicate normalized jobs had been canceled earlier, and the remaining
six normalized replicate slots are reused prior evaluations. The eight running and ten queued
jobs at this later audit were all reference-scaled, so no additional cancellation was necessary.
Completed normalized results are retained as the comparison baseline, and normalized K=8 is also
retained because it is mathematically identical to the shared reference-scaled K=8 point.

## 2026-07-30 complete MATH-500 plateau collection

All 177 nonduplicate new jobs have succeeded and are downloaded. Together with the 12 reused
prior replicate slots, this completes all 66 logical model/policy/K points with three MATH-500
evaluations per point. All 177 new bundles have exactly 500 predictions, empty metric-error
lists, and zero empty generated outputs. The three canceled experiments are only the known
duplicate K=8-trained/normalized-K=6 submissions; there are no failed or unfinished jobs.

The final reference-scaled curves show:

- K=8-normalized training: 47.80% at K=2, 73.67% at K=3, 80.93% at K=4,
  83.20% at K=5, and 84.07% at K=8.
- K=6-reference-scaled training: 48.87% at K=2, 73.53% at K=3, 78.47% at
  K=4, 81.13% at K=5, and 79.40% at K=8.
- K=4-reference-scaled training: 54.20% at K=2, 76.60% at K=3, 81.33% at
  K=4, 82.60% at K=5, and 82.67% at K=8.

Reference scaling therefore changes the low-K failure mode substantially: normalized K=2 is
effectively collapsed (0.07--0.20%) and normalized K=3 scores 32.07--38.07%, whereas
reference-scaled K=2 retains 47.80--54.20% and K=3 retains 73.53--76.60%. The
reference-scaled plateau begins around K=4--5, remains flat through K=12, and shows no consistent
benefit from selecting more than eight experts. Training at reference-scaled K=4 produces the
strongest graceful degradation at K=2 and K=3 while retaining its K=8-level performance from
approximately K=4 onward.

Final outputs are `slime_qwen3_base_dapo_step350_plateau_replicates.csv`,
`slime_qwen3_base_dapo_step350_plateau_summary.csv`, and
`plots/slime_qwen3_base_dapo_step350_math500_plateau.{png,svg}`.

## 2026-08-06 K=10 DAPO pilots

Two matched 100-step single-K DAPO pilots were submitted to test directly whether training can
benefit from more than Qwen3's native eight active experts. Both reuse the established fixed-K
recipe: one 8xB200 node, 16 prompts per rollout, eight samples per prompt, global batch 128,
dynamic-sampling candidate pool 64, 20,480-token response cap, and learning rate `1e-6`.
Checkpoints are saved every 20 steps (20, 40, 60, 80, and 100).

- Native normalized K=10: [Beaker 01KZAPQJG5SPREY96RV02F5T54](https://beaker.org/ex/01KZAPQJG5SPREY96RV02F5T54)
- K=8-reference-scaled K=10: [Beaker 01KZAPQVD2SVVEZ273HRDERX89](https://beaker.org/ex/01KZAPQVD2SVVEZ273HRDERX89)

The reference-scaled arm selects ten experts but divides their weights by the top-eight router
mass, so its selected weights sum above one. An explicit K=10/reference-K=8 regression test was
added for the shared policy and the SGLang wrapper; the reference-scaling test suite passes 9/9.
Both jobs are urgent in `ai2/linear-rnns` on `ai2/titan` and are tracked under group
`adaptive-compute-slime-qwen3-base-k10-pilots-20260806` in `notes/beaker_jobs.jsonl`.

## 2026-08-10 K={4,8,12} mixed-K pilot and K=10 checkpoint evaluations

A neutral, evenly split 100-step mixed-K pilot was launched to test whether adaptive RL can
learn across a wider expert-count range that includes more experts than Qwen3's native K=8.
Every rollout prompt receives three samples at each of K=4, K=8, and K=12 (nine samples total).
All three policies use K=8-reference-scaled router weights and receive the same task reward;
there is no compute-cost bonus. The learner dispatches K=12 and routing replay pads K=4 and K=8
trajectories to the learner width. The topology is one two-GPU K=4 SGLang engine, one two-GPU
K=8 engine, and two two-GPU K=12 engines, with the same eight B200s reused for learner updates.
Other settings match the established DAPO pilots: 16 prompts per rollout, global batch 144,
dynamic-sampling pool 64, 20,480-token response cap, learning rate `1e-6`, and checkpoints every
20 steps. The urgent, unallocated Titan job is
[Beaker 01KZN7H1T8B5RQWKXASWT2EQ55](https://beaker.org/ex/01KZN7H1T8B5RQWKXASWT2EQ55), in
workspace `ai2/OLMo-3-moe-experiments`. Explicit mixed-routing replay and K=12/reference-K=8
tests were added; the focused suite passes 21 tests with one skipped optional distributed test.

The five checkpoints from each completed fixed-K=10 pilot (steps 20/40/60/80/100) are being
exported and evaluated once as an initial sanity curve. Each checkpoint uses the same routing
policy it trained with: normalized K=10 for the normalized run and K=8-reference-scaled K=10 for
the reference-scaled run. Each eval combines MATH-500 and AIME 2025 pass@32 and uses the corrected
RL-checkpoint recipe: no server-side reasoning parser, four TP=1 vLLM engines, 32,768-token
context, 20,480-token generation cap, MATH temperature 0.6/top-p 0.95/top-k 20 with one sample,
AIME with 32 samples, and seed 42. Evals are urgent in `ai2/holmes-testing` and use the `80g`
cluster alias so Beaker can schedule them across Jupiter and the other compatible 80-GB clusters.
They share [Beaker group 01KZN85HY01YCJ9MD0BTQTZQN5](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KZN85HY01YCJ9MD0BTQTZQN5).
All exporter and eval experiment IDs are recorded in `notes/beaker_jobs.jsonl`.

The launch audit found exactly ten initial eval IDs: five checkpoints per policy and steps
20/40/60/80/100 for each. Both one-GPU exporter jobs completed successfully. Four initial eval
allocations had infrastructure/startup failures: the reference-scaled step-20/40/60 jobs failed
Beaker container health checks before the eval process started, and normalized step 100 timed out
waiting for its vLLM workers to open their ports. All four conditions were relaunched unchanged
with `retry1-20260810` tags in the same group and on the same `80g` alias. At the post-retry audit,
normalized steps 40 and 60 had already succeeded, normalized steps 20 and 80 plus reference step
100 and the normalized-step-100 retry were running, and the remaining reference jobs were queued.
The mixed-K trainer had completed six optimizer updates (through logged step 5) without an error.

### 2026-08-10 completion update

Both new adaptive pilots reached step 100 and wrote complete distributed checkpoints at steps
20/40/60/80/100. The K={4,8,12} pilot exited cleanly. The preallocated pilot's Slime/Ray job also
reported success and saved step 100, but its outer shell subsequently returned 127 because the
submitted source snapshot contained a stray standalone `--use-wandb` token after the Ray command.
This was post-training cleanup rather than a damaged or partial checkpoint; the current launcher no
longer contains that token.

Both pilots were resumed from their step-100 optimizer and scheduler states to a 500-step target,
with new checkpoints at steps 150/200/250/300/350/400/450/500. The jobs are urgent and unallocated
on 8xB200 Titan nodes in `ai2/OLMo-3-moe-experiments`:

- preallocated cheapest-success: [Beaker 01KZP8MN9PVY2RNC84AKJ6R3Q2](https://beaker.org/ex/01KZP8MN9PVY2RNC84AKJ6R3Q2)
- neutral K={4,8,12}: [Beaker 01KZP8MRHR3WXK122RXXJJKT68](https://beaker.org/ex/01KZP8MRHR3WXK122RXXJJKT68)

The two five-checkpoint HF exporters completed successfully:
[01KZP8MVVVN1GX5HZRCK4VPHWA](https://beaker.org/ex/01KZP8MVVVN1GX5HZRCK4VPHWA)
and [01KZP8MZ9J6MPVNG6Y064PY4X3](https://beaker.org/ex/01KZP8MZ9J6MPVNG6Y064PY4X3).
All 30 initial evals were then submitted: each step-20/40/60/80/100 checkpoint at every policy used
during training—reference-scaled K=4/K=6 plus native K=8 for the preallocated model, and
reference-scaled K=4/K=12 plus native K=8 for the K={4,8,12} model. Each job combines one MATH-500
evaluation and AIME 2025 pass@32 using the established corrected RL-checkpoint settings. They are
urgent in `ai2/holmes-testing` on the `80g` cluster alias, under
[Beaker group 01KZP8WAWEPSE5MJK2K93VMCSB](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KZP8WAWEPSE5MJK2K93VMCSB).

The completed K=10 pilot evals are collected in
`notes/slime_qwen3_base_dapo_k10_checkpoint_evals.csv`. The MATH-500, AIME Pass@1, and AIME Pass@8
figures are in `notes/plots/slime_qwen3_base_dapo_k10_*_by_step.{png,svg}`. At step 100, normalized
K=10 scores 78.8% MATH-500 and 12.08% AIME Pass@1; K=8-reference-scaled K=10 scores 79.6% and
10.21%, respectively. These are single seeded evaluations per checkpoint, so the AIME differences
should be treated as noisy.

## 2026-08-13 evaluation refresh

All ten Qwen3.6 K=12 Terminal-Bench shards completed with exit code 0. Their five complete 89-task
runs are now incorporated into `notes/qwen36_terminalbench_full_suite_summary.csv` and
`notes/plots/qwen36_terminalbench_expert_sweep.{png,svg}`. K=12 scores 24.72%, 31.46%, 30.34%,
26.97%, and 30.34%, for a mean of 28.76% (128/445), exactly tied with native K=8 in aggregate.
The source jobs are [Beaker group 01KZVM534SVXEQ02S9Q799E8WW](https://beaker.org/orgs/ai2/workspaces/olmo-instruct/groups/01KZVM534SVXEQ02S9Q799E8WW).

The Qwen3.6 MATH-500/GPQA/AIME/IFBench capability sweep has no finished job yet: all three K=4,
all three K=6, and two K=8 replicates are running; the third K=8 replicate and all K=10/K=12
replicates remain queued. No job in that group is failed. The live matrix is
[Beaker group 01KZVSKMN41N1SQ6SQ4TGP87J5](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KZVSKMN41N1SQ6SQ4TGP87J5).

## 2026-08-13 17:31 UTC collection and terminal-agent status

The Qwen3.6 capability matrix now has 14/15 successful runs. Six new bundles were downloaded and
validated: native-K=8 seed 44, all three K=10 reference-scaled seeds, and K=12 reference-scaled
seeds 42/43. Each has all six tasks, 4,189 aggregate instances, and zero run-level metric errors.
The only remaining job is K=12 seed 44, which is running at approximately 65% of scored instances;
its recent sustained rate suggests roughly three to four additional hours. The CSV and
`notes/plots/qwen36_capability_expert_sweep.{png,svg}` were regenerated with three seeds at
K={4,6,8,10} and two seeds at K=12. Full values and experiment IDs are in
`notes/qwen36_capability_sweep.md` and `notes/qwen36_capability_results.csv`.

The timeout-relaxed native-K=8 terminal diagnostics are all healthy and running with zero task
exceptions at this snapshot. TB-Lite has completed 3/55 and 9/45 trials on its two shards; the
current aggregate is 12/100. Terminal-Bench 2.1 has completed 2/45 and 1/44, or 3/89. A naive
early-rate extrapolation puts TB-Lite near one hour and TB2.1 near two hours, but both are still in
their easiest early tasks and the overall agent timeout is deliberately disabled. Operationally,
allow one to three more hours for TB-Lite and three to six hours for TB2.1, with a possible longer
TB2.1 tail if one or more trajectories consume all 64 steps or a long shell command.

No new adaptive-RL evaluation bundle appeared after the complete 72-point collection from
2026-08-12, so those plots did not need regeneration. The preallocated training run is complete at
step 500. The even K={4,8,12} continuation has a valid step-450 checkpoint and is still training
(step 466 at the snapshot); its existing evaluation plot remains complete only through step 400.

## 2026-08-13 step-450 adaptive-RL evaluation launch

The even K={4,8,12} run's complete distributed step-450 checkpoint was converted atomically to
the standard HF layout by [Beaker 01KZY4GC1GZME8TZMNGZ1NWN6B](https://beaker.org/ex/01KZY4GC1GZME8TZMNGZ1NWN6B).
Nine urgent evaluations were then submitted in `ai2/holmes-testing` on the `80g` cluster alias:
reference-scaled K=4, native K=8, and reference-scaled K=12, each with MATH-500 seeds 42/43/44 and
one AIME 2025 pass@32 run in the seed-42 bundle. This exactly matches the step-400 recipe. The
combined MATH/AIME jobs are in
[group 01KZVAG9NR3P5HPHNDAH2GXZNE](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KZVAG9NR3P5HPHNDAH2GXZNE),
and the two additional MATH replicates per policy are in
[group 01KZVAKANVNDGYJTVDN42EEVN4](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KZVAKANVNDGYJTVDN42EEVN4).
All nine experiment IDs and commands are recorded in `notes/beaker_jobs.jsonl`. Step 500 was not
available at launch time.

## 2026-08-13 18:56 UTC evaluation status

All nine even-K={4,8,12} step-450 evaluation jobs succeeded without a retry. Their metric bundles
were downloaded and validated with empty run-level error lists, and the adaptive-K CSV and plots
were regenerated. The three-seed MATH-500 means are 82.07% at reference-scaled K=4, 85.67% at
native K=8, and 84.87% at reference-scaled K=12. The corresponding single-bundle AIME pass@1
estimates are 13.02%, 16.67%, and 17.08%. The mixed trainer remains healthy at step 469, so step
500 is not yet available.

The timeout-relaxed TB-Lite shards are at 49/55 and 37/45 completed trials (86/100 total), both
with zero errors; their combined running reward mean is approximately 73.5%. The timeout-relaxed
TB2.1 shards are at 10/45 and 8/44 (18/89 total). Shard A has one infrastructure error on
`write-compressor`: Podman's Docker-compatible CLI received an argument list that exceeded the OS
limit. It is isolated to that task rather than a model-generation timeout; all other active trials
continue. Shard B has zero errors.

The last Qwen3.6 capability replicate, K=12 seed 44, is at 3,396/4,189 scored instances (81%) and
continues normally. A subsequent audit corrected an earlier status-classification error: six older,
superseded terminal-evaluation submissions were already manually canceled before start, rather than
still queued. These comprise four published-settings TB2.0 shard copies whose later copies
succeeded and two native-timeout TB2.1 shard copies superseded by the completed prefix-fixed jobs;
none can consume capacity and none is part of the current timeout-relaxed measurements.

## 2026-08-13 20:34 UTC TB-Lite long tail

The timeout-relaxed TB-Lite run reached 95/100 completed trials: shard A is 52/55 with zero
errors and shard B is 43/45 with zero errors. The five remaining trials are no longer making
normal agent-step progress. Shard A's latest agent-step log was at 19:39 UTC and shard B's was at
19:11 UTC; the one-minute progress monitors continue but the task counts do not change. This is
consistent with agents blocked inside long shell calls. Since the diagnostic deliberately disables
the overall agent timeout and sets `command_timeout=21600` (six hours), Harbor does not yet record
these as failures. Depending on when each outstanding command began, the command limits may not
fire until approximately 01:00--01:40 UTC. The completed-task running means are 0.763 and 0.659,
or approximately 71.6% when weighted across the 95 currently completed tasks; this is not a final
100-task score.

## 2026-08-15 above-native routing audit

The Qwen3.6 K=10/12 startup failures exposed a historical HF-override bug. Qwen3.6 stores its
active-expert count at `text_config.num_experts_per_tok`, but the historical K=10/12 launchers set
a nonexistent outer field. Those nominal K=10/12 jobs therefore remained at native K=8. This
invalidates the expert-count interpretation of the old Qwen3.6 K=10/12 capability, Terminal-Bench
2.0, TBLite, and TB2.1 points; it does not invalidate their raw outputs as additional K=8 samples.
The historical K=4/6 reference-scaled jobs remain valid because they intentionally routed the
native top eight and truncated that result after routing. Existing Qwen3.6 plots and summary CSVs
now exclude the invalid K=10/12 points.

The cross-model audit found no corresponding invalidation. Standard Qwen3 and GPT-OSS store K in
the top-level config field used by their historical launchers, so their retained above-native
sweeps are valid. Off-the-shelf Qwen3.5 was only evaluated through native K=8 and already used
nested overrides. Its OpenThoughts K=12 SFT checkpoint itself contains nested K=12 and passed the
exact router-match assertion. GLM-4.5-Air and Nemotron-3-Super were not evaluated above their
native K. Full evidence is in `above_native_k_routing_audit.md`.

## 2026-08-15 Qwen3.6 K>8 replacement launch

Dedicated K=10 and K=12 smokes passed in both the TMax and olmo-eval serving
paths. Logs explicitly reported matched router layers at K=10 and K=12. The
corrected replacement matrix is now submitted: six capability-suite jobs,
twenty legacy TB2.0 shards, six TBLite jobs, and two TB2.1 jobs. At the launch
snapshot the capability jobs, all TBLite/TB2.1 jobs, and four legacy TB2.0
shards were running; the remaining sixteen legacy TB2.0 shards were queued.
There were no failed production replacements. Historical invalid K=10/K=12
points remain excluded from plots until complete corrected conditions exist.
All links and settings are in `qwen36_kgt8_replacement_jobs.md` and all jobs
are recorded in `beaker_jobs.jsonl`.

## 2026-08-16 04:45 UTC full evaluation status

The corrected Qwen3.6 TBLite sweep is complete at all five expert counts with
three full 100-task runs per K. Pass@1 means are 60.0% at K=4, 62.0% at K=6,
63.0% at K=8, 63.67% at K=10, and 63.0% at K=12. Some corrected K=10/K=12
Beaker wrappers were manually canceled after their complete metrics had
already been written; all six corrected runs have `n_tasks=100` and are
scientifically usable.

The corrected legacy TB2.0 K=10/K=12 replacement outputs are also complete:
five 89-task logical runs per K. One K=10 shard wrapper was canceled after its
complete 45-task metric file was written, so no logical replicate is missing.
The new aggregate pass@1 means are 31.46% at K=10 and 31.24% at K=12. These
outputs have not yet been merged into the maintained legacy plot.

The first corrected TB2.1 run is complete for every K. Pass@1 is 25.84%,
33.71%, 32.58%, 31.46%, and 35.96% at K=4/6/8/10/12. All ten added replicate
2/3 jobs are running. At the status snapshot, their completed-task counts for
replicates 2/3 were K4 55/30, K6 22/15, K8 18/15, K10 15/16, and K12 12/11,
out of 89. They continue to emit agent-step and task-completion logs; no
Beaker job has failed. The reported per-job `errors` are task-level agent or
command timeouts that score as failures under the benchmark protocol, not
job-level crashes.

The three OpenThoughts-Agent SFT native-K TBLite baselines are complete:
36% for train/eval K=4, 47% for K=8, and 45% for K=12. Of the 24 newly added
train-K x eval-K jobs, four are running and twenty are queued. The running
jobs are train-K=4 evaluated at K=4 (replicates 2/3, 21/100 and 19/100 tasks)
and K=8 (replicates 1/2, 17/100 and 2/100). All four passed exact routing
validation and are making progress; none has failed.

Finally, all six corrected Qwen3.6 capability jobs at K=10/K=12 are running.
The three K=10 jobs are approximately 97--99% complete (4,076--4,152 of
4,189 scored instances), and K=12 is approximately 90--91% complete
(3,763--3,827 of 4,189). There are no production failures in this group.

## 2026-08-16 07:29 UTC full evaluation status

All six corrected Qwen3.6 capability jobs have now succeeded. Their result
bundles were downloaded and validated: every bundle contains all six tasks,
4,189 scored instances, and zero run-level metric errors. The maintained CSV
and plot now have three runs at every K in {4, 6, 8, 10, 12}. At K=10 the
means are MATH-500 92.73%, GPQA Diamond 80.64%, IFBench macro 72.65%, and
AIME 2025 pass@1 59.24%; at K=12 they are 92.60%, 81.82%, 72.59%, and
57.15%, respectively.

The expanded TB2.1 matrix has five newly successful Beaker jobs and ten
complete 89-task metric bundles overall when the five replicate-1 baselines
are included. Current aggregate pass@1 over completed bundles is 28.46%
(n=3) at K=4, 31.46% (n=2) at K=6, 34.27% (n=2) at K=8, 31.46% (n=1) at
K=10, and 35.96% (n=1) at K=12. K10/r3 then completed at 30.34%, updating
the K=10 mean to 30.90% (n=2). Five jobs remain active: K6/r3 and K8/r3 are
88/89, K10/r2 is 81/89, K12/r2 is 87/89, and K12/r3 is 77/89. All are
still making progress and none has crashed. The TB2.1 plot
was regenerated from complete bundles only.

The OpenThoughts-Agent SFT TBLite cross-evaluation matrix has eleven jobs
running and thirteen queued. Running train-K=4 evaluations range from 5/100
to 78/100 completed tasks across eval K={4,8,12}; train-K=8/eval-K=4 has one
job at 14/100 and two more loading vLLM. All started jobs have passed or are
approaching the fail-closed routing gate, and there are no failed jobs.

The new off-the-shelf Qwen3.6 expanded TBLite sweep is fully submitted: two
one-task routing smokes plus twelve full runs (three each at K=2,14,16,32).
All fourteen are queued behind the older allocated jobs. The smokes are
ordered first and will require K=2 to report router-K=8/keep-K=2 and K=32 to
report router-K=keep-K=32 before any benchmark score is accepted.

## 2026-08-16 17:04 UTC full evaluation status

The Qwen3.6 TB2.1 expansion is complete with three full 89-task runs at every
K. Mean pass@1 is 28.46% at K=4, 31.84% at K=6, 32.21% at K=8, 32.21% at
K=10, and 34.83% at K=12. No run failed, and the maintained TB2.1 CSV and
plot were regenerated from all fifteen complete bundles.

The OpenThoughts-Agent SFT TBLite matrix has 22/24 newly launched jobs
complete, or 25/27 total runs after including the three prior diagonal
baselines. The two remaining jobs are train-K=12/eval-K=4 replicate 3 at
99/100 tasks and train-K=12/eval-K=12 replicate 3 at 94/100. Both are still
emitting progress. There are no job-level failures.

Both expanded Qwen3.6 routing smokes passed exactly: K=2 reported
router-K=8/keep-K=2/reference-K=8, and K=32 reported
router-K=keep-K=32/reference-K=8. Of the twelve full expanded TBLite runs,
six have succeeded and six are running. Complete-run mean reward is 27.23%
at K=2 (n=1), 64.05% at K=14 (n=3), and 65.54% at K=16 (n=2). The remaining
active jobs are K=2 replicates 2/3, K=16 replicate 3, and all three K=32
runs. Their current progress is 58/100, 81/100, 79/100, 59/100, 47/100, and
47/100, respectively. The TBLite plot now includes only the newly complete
expanded points alongside the original K=4--12 sweep.

## 2026-08-16 22:06 UTC full evaluation status

The expanded Qwen3.6 TBLite sweep is now complete: three 100-task runs at
every K in {2,4,6,8,10,12,14,16,32}. Mean reward is 25.71%, 58.67%, 60.92%,
61.56%, 62.55%, 61.94%, 64.05%, 66.45%, and 65.91%, respectively. The
maintained CSV and plot were regenerated from all 27 complete runs. K=2 is a
clear failure point; K=14--32 remains at least as strong as K=8 within the
observed variance.

The Qwen3.6 TB2.1 sweep remains fully complete with three 89-task runs per K.
Mean pass@1 is 28.46%, 31.84%, 32.21%, 32.21%, and 34.83% at
K={4,6,8,10,12}. Its maintained CSV and plot were also regenerated.

The OpenThoughts-Agent SFT TBLite cross-evaluation matrix is complete: all
24 added jobs succeeded, giving 27 total runs and three runs in every
train-K/eval-K cell. Mean reward at eval K=4 is 32.43%, 31.41%, and 31.85%
for train K=4/8/12. At eval K=8 it is 39.24%, 39.40%, and 40.90%; at eval
K=12 it is 42.26%, 37.03%, and 39.92%. Full means, standard deviations, and
pass@1 values are recorded in `openthoughts_sft_tblite_matrix_jobs.md`.

All corrected K=10/K=12 Qwen3.6 capability jobs and all corrected legacy
TB2.0 K=10/K=12 shard jobs remain complete. No new production job in any
finished group has failed; the failed/canceled records in older groups are
superseded smokes, pre-fix K>8 runs, or wrappers canceled after complete
metrics were already written.

Two new extreme sweeps are active. For the capability suite at K=64/128/256,
eight jobs are running and K=256 replicate 3 is queued. Runtime logs confirm
the requested router-K, keep-K, and model `num_experts_per_tok` values for all
started jobs. K=64 and K=128 have begun scoring; K=256 is still processing
its first 64-example batches and is much slower, but has no OOM or traceback.
For extreme TBLite at the same K values, one K=64 job is scheduled and the
other eight production jobs are queued in `ai2/OLMo-3-moe-experiments`.
Earlier K=64/128/256 routing smokes all
validated the exact expert counts; the K=128 smoke was manually stopped only
after validation and 57 agent steps. There are no actionable production
failures at this snapshot.

## 2026-08-17 03:13 UTC full evaluation status

Twelve new Qwen3.6 Terminal-Bench 2.1 jobs were submitted at K={14,16,32,64},
three runs per K. All use the matched TMax-default protocol from the completed
K=4--12 sweep: the pinned 89-task dataset, Qwen thinking-mode generation,
Vanillux2, TP=2/DP=4, urgent allocated scheduling, and Jupiter/Ceres. Every
submitted spec was validated with router K = keep K = model
`text_config.num_experts_per_tok` = requested K, reference K=8, and
renormalize=false. All twelve are queued.

The first extreme TBLite production result is complete. K=64 replicate 1
scored 48.70% mean reward and 50.0% pass@1 over all 100 tasks, with 16 tasks
recording exceptions. This is a substantial provisional drop from the K=32
three-run mean of 65.91%, but it is currently only one replicate. K=64
replicate 2 is 99/100 with a provisional 52.4% mean, and replicate 3 is
77/100 with a provisional 52.3% mean. The three K=128 jobs are at 34/100,
36/100, and 23/100; their incomplete means are 23.6%, 33.1%, and 28.9% and
must not be treated as final scores.

All three K=256 TBLite jobs failed during vLLM startup. The exact cause is
insufficient memory for any KV-cache blocks at the 262,144-token context
under the current TP=2/DP=4 topology; this is not a routing failure. No K=256
result enters the plots. A retry would need a memory-oriented topology such
as TP=4/DP=2 or a smaller context, with TP=4 preferred if the generation
protocol must remain unchanged.

The extreme capability suite remains active with no job-level failure. K=64
replicates have scored 632, 580, and 579 of 4,189 instances (14--15%); K=128
replicates are all at 377 (9%). K=256 replicates 1/2 are at 202 and 238
(5--6%), while replicate 3 remains queued. K=256 has not emitted a scoring
heartbeat for roughly two hours, so it is flagged as very slow, but the jobs
remain alive with no OOM or traceback.

The maintained TBLite and TB2.1 CSVs and plots were regenerated from complete
runs only. The TBLite plot now includes K=64 replicate 1; the TB2.1 plot is
unchanged until one of the new 89-task runs finishes. All previously reported
K<=32 TBLite, K<=12 TB2.1, corrected capability, corrected legacy TB2.0, and
OpenThoughts-Agent SFT-matrix results remain complete.

## 2026-08-17 04:28 UTC full evaluation status

The missing even TBLite points between K=16 and K=32 have been launched:
K={18,20,22,24,26,28,30}, with three replicates each (21 jobs). All are
queued. Post-submit inspection verified the requested expert count in all
three relevant controls (router K, keep K, and nested model
`num_experts_per_tok`); all use reference scaling to K=8 with no
renormalization. Full links are in
`qwen36_tblite_intermediate_reference_jobs.md`.

The complete active-job inventory for the Beaker account contains 45 jobs:
11 running and 34 queued. The running jobs are three K=128 extreme TBLite
runs and eight extreme capability runs (K=64 x3, K=128 x3, K=256 x2). The
queued jobs are the third K=256 capability replicate, twelve TB2.1 jobs at
K={14,16,32,64} x3, and the 21 new intermediate TBLite jobs. There are no
scheduled-but-not-started jobs and no other active job groups.

All three extreme K=64 TBLite replicates have now succeeded. Their mean
rewards are 48.70%, 52.89%, and 53.80% (three-run mean 51.80%); pass@1 is
50%, 54%, and 55% (mean 53.0%). The maintained TBLite CSV and plot were
regenerated from all 30 complete runs. The three K=128 TBLite jobs remain
running. The three K=256 TBLite attempts remain failed at vLLM startup due
to KV-cache memory at 262K context under TP=2/DP=4 and are not active or
plotted.

## 2026-08-17 06:40 UTC full evaluation status

The Beaker account has 45 active jobs eligible for Jupiter: 15 running and
30 queued, with none in the scheduled-but-not-started state. No job has newly
failed or finished since the preceding 04:28 UTC refresh.

The three extreme K=128 TBLite runs remain active at 77/100, 84/100, and
84/100 completed trials. Their provisional mean rewards are 22.3%, 24.9%,
and 27.9%, with 36, 34, and 37 recorded task errors. They are continuing to
emit agent steps and trial completions rather than hanging; these incomplete
scores are excluded from maintained results and plots.

Eight extreme capability jobs are running. K=64 is at 871/4189, 883/4189,
and 883/4189 scored instances (about 21%); K=128 is at 684/4189, 632/4189,
and 579/4189 (14--16%); K=256 replicates 1 and 2 are both at 364/4189 (9%).
The third K=256 replicate remains queued. The K=256 jobs have multi-hour
64-item batches but no traceback or OOM, so they are classified as very slow
rather than failed or hung.

Four of the twelve expanded TB2.1 jobs are now running: all three K=14
replicates and K=16 replicate 1. They are at 20/89, 17/89, 14/89, and 11/89
completed trials and are actively producing agent steps. The remaining eight
TB2.1 jobs (K=16 replicates 2/3 and all K=32/K=64) are queued.

All 21 intermediate TBLite jobs at K={18,20,22,24,26,28,30} remain queued.
All previously completed Qwen3.6 standard TBLite/TB2.1/capability sweeps and
the OpenThoughts-Agent SFT matrix remain complete. The known three K=256
TBLite startup failures are not active and have not been relaunched.

## 2026-08-17 07:45 UTC K=256 capability cancellation

All three K=256 extreme capability jobs were canceled at the user's request
after the severely degraded provisional K=128 TBLite results made the added
compute unjustified. Replicates 1 and 2 were running and had each scored
364/4189 instances; replicate 3 was still queued. All three are now finalized
as canceled, and their incomplete outputs will not enter results or plots.

The post-cancellation account inventory is 42 active jobs: 14 running and 28
queued. The running jobs are K=128 TBLite x3, capability K=64 x3 and K=128
x3, and expanded TB2.1 x5. The queued jobs are the remaining seven TB2.1
runs and all 21 intermediate TBLite runs. No K=256 job remains active.

## 2026-08-17 16:54 UTC full evaluation status

The active Beaker inventory has fallen from 42 jobs to six: all six are
running and none are queued or scheduled. These are the retained extreme
capability jobs at K=64 x3 and K=128 x3. K=64 is 46--47% complete
(1920/4189, 1984/4189, and 1976/4189); K=128 is 28--29% complete
(1172/4189, 1215/4189, and 1169/4189). All have recent scoring heartbeats and
no traceback or OOM.

All three K=128 extreme TBLite runs finished successfully and validated as
complete 100-task results. Mean reward is 27.33% ± 3.16 percentage points,
mean pass@1 is 29.0%, and the runs average 42.7 task errors. Together with
K=64 at 51.80% ± 2.72 and K=32 at 65.91% ± 2.62, this confirms a clear
extreme-K degradation even with reference scaling.

All 21 intermediate TBLite jobs at K={18,20,22,24,26,28,30} succeeded.
Three-run mean rewards are 64.28%, 59.27%, 65.22%, 65.50%, 62.85%, 60.96%,
and 65.47%, respectively. The individual values are noisy, but the dense
curve remains broadly flat from K=16 through K=32 before falling at K=64.

All twelve expanded TB2.1 jobs also succeeded. Mean pass@1 is 31.84%,
32.96%, 33.71%, and 26.22% at K={14,16,32,64}. This independently shows a
plateau through K=32 and a clear K=64 drop. The K=64 runs average 44.3 task
errors versus 25.0--33.3 across K=14--32.

The maintained finished-only datasets and plots were regenerated. They now
contain 54 TBLite runs and 27 TB2.1 runs. No partial, canceled, failed, or
invalid-routing run is included.

## 2026-08-17 18:42 UTC full evaluation status

The Beaker account has 21 active jobs across the clusters used by this project:
14 running and seven queued. There are no active training jobs and no newly
failed production jobs.

Six retained Qwen3.6 extreme capability jobs remain healthy on Saturn. The
three K=64 replicates have scored 2174, 2176, and 2176 of 4189 instances
(52% each). The K=128 replicates have scored 1456, 1379, and 1440 instances
(33--35%). Every job has a recent scoring heartbeat and none has emitted an
OOM or traceback. K=256 remains intentionally canceled and excluded.

The mixed-K={4,8,12} OpenThoughts-Agent SFT TBLite matrix has 15 valid jobs:
three replicates at evaluation K={4,6,8,10,12}. Eight are running on Jupiter
(all K=4 and K=6 runs plus K=8 replicates 1/2), and seven are queued (K=8
replicate 3 and all K=10/K=12 runs). The active K=4 runs are at 15--24/100
trials, K=6 at 10--15/100, and K=8 at 10--15/100. All eight started jobs
passed the fail-closed routing assertion and continue to produce trials; the
partial rewards are too early and noisy for interpretation.

All other current result sets remain complete and locally collected: the
54-run Qwen3.6 TBLite curve (three runs at K=2, every even K from 4 through
32, K=64, and K=128), the 27-run TB2.1 curve (three runs at
K={4,6,8,10,12,14,16,32,64}), the corrected K={4,6,8,10,12} capability
curve, the five-run-per-K legacy TB2.0 sweep, and the 27-run fixed-SFT
train-K x eval-K TBLite matrix. There is no completed result bundle waiting
for collection at this snapshot.

## 2026-08-17 19:42 UTC cancellation and launch update

The three K=128 Qwen3.6 capability jobs were canceled after roughly 22.4
hours each. Their final partial scored-instance counts are 1529, 1441, and
1482 of 4189; these incomplete outputs are excluded. K=64 x3 remains active.

Six capability bridge jobs were launched at K={16,32}, three seeds per K.
All three K=16 jobs are running and all three K=32 jobs are queued. Exact
post-submit inspection confirmed nested text-config router widths, live
router K, and keep K equal the requested K, with reference K=8 and no
renormalization.

Eighteen fixed-SFT TBLite jobs were launched to backfill eval K={6,10} for
each train K={4,8,12}, with three runs per cell. All are queued. K=6 uses
router K=8/keep K=6; K=10 uses validated router-width-10 config files and
keep K=10. Both policies preserve the K=8 reference scale. The account now
has 42 active jobs: 15 running and 27 queued, with no active training jobs.

## 2026-08-17 21:01 UTC progress update

All 42 jobs in the current scope remain healthy: 23 are running and 19 are
queued, with no new completions or failures. Five of the six Qwen3.6
capability bridge jobs are running. The K=16 replicates are in active
generation (their current generation batches report 6%, 34%, and 44%); two
K=32 replicates report 49% and 57%, while the third K=32 replicate remains
queued.

The retained K=64 capability replicates continue to advance and have scored
2493, 2494, and 2439 of 4189 instances (approximately 58--60%). The mixed-K
OpenThoughts SFT TBLite matrix is now fully scheduled: all 15 runs are active,
with per-run progress ranging from 15/100 to 71/100 trials. The new fixed-SFT
K={6,10} backfill remains entirely queued (18/18 jobs).

## 2026-08-17 22:33 UTC SFT evaluation update

The first mixed-K SFT TBLite run completed successfully: evaluation K=8
replicate 1 scored 35.19% mean reward and 37.00% pass@1 on 100 tasks. The
other 14 mixed-SFT runs remain active, with K=4 at 77--99/100 trials, K=6
at 89--98/100, the two remaining K=8 runs at 58 and 91/100, K=10 at
48--52/100, and K=12 at 46--60/100. No mixed-SFT run has failed.

All 18 fixed-SFT K={6,10} backfill jobs remain queued. The six Qwen3.6
K={16,32} capability bridge jobs are all running but none is complete, so
the latest finished-results capability plot remains the validated
three-replicate K={4,6,8,10,12} curve.

## 2026-08-17 23:22 UTC mixed-SFT update

Four more mixed-SFT TBLite runs completed, bringing the matrix to 5/15
complete, 10 running, and zero failed. The available results are K=4 at
35.86 ± 1.44% mean reward and 38.00 ± 1.41% pass@1 (two runs), K=6 at
38.97 ± 2.01% mean reward and 40.50 ± 2.12% pass@1 (two runs), and K=8 at
35.19% mean reward and 37.00% pass@1 (one run). Near-complete stragglers are
K=4 replicate 3 at 94/100 tasks, K=6 replicate 2 at 98/100, and K=8
replicate 2 at 99/100. K=10 is at 64--75/100 and K=12 at 66--92/100.

## 2026-08-18 03:20 UTC results refresh

All 15 mixed-K OpenThoughts SFT TBLite runs completed successfully. Across
three replicates, mean reward / pass@1 were: K=4 33.56±4.11 / 35.67±4.16%,
K=6 37.90±2.33 / 39.33±2.52%, K=8 37.58±2.15 / 39.67±2.31%, K=10
41.98±2.98 / 43.67±3.21%, and K=12 42.05±1.03 / 43.67±0.58%.

The fixed-SFT K={6,10} backfill has one running job (train K=4, eval K=6,
replicate 1) at 70/100 trials and 17 jobs queued. The Qwen3.6 K={16,32}
capability bridge has all six jobs running, with no completed result yet.
All three older K=64 capability jobs hit their explicit 24-hour timeout after
scoring 2494, 2494, and 2439 of 4189 instances; these partials are excluded
from all results and plots.

## 2026-08-18 03:40 UTC mixed-SFT extension launch

Fifteen additional mixed-SFT TBLite runs were launched for K={2,14,16,24,32},
three paired-seed replicates per K. They use the exact completed-matrix recipe:
reference-K=8 scaling without renormalization, one 8xH100 node per run, urgent
allocated scheduling in `ai2/OLMo-3-moe-experiments`, and Jupiter/Ceres
targeting. All 15 post-submit Beaker specs passed exact routing-policy checks;
the jobs were initially queued.

## 2026-08-18 03:52 UTC SFT comparison plot

A unified TBLite mean-reward plot was produced for all four completed SFT
checkpoints: fixed train K={4,8,12} and mixed train K={4,8,12}. It uses
three-run means with sample-SD shading and includes only completed cells. The
pending fixed K={6,10} backfill and mixed K={2,14,16,24,32} extension will be
added as their full three-run cells finish.

## 2026-08-18 04:20 UTC fixed-SFT extension launch

Twenty-seven fixed-SFT TBLite jobs were launched for every combination of
train K={4,8,12}, eval K={2,14,16}, and three paired seeds. They match the
existing fixed-SFT matrix protocol and use reference-K=8 scaling without
renormalization. The K=2 runs route eight and truncate to two; K={14,16}
sets both the nested Qwen3.5 router width and runtime router K to the requested
count. All 27 post-submit specs passed the exact routing, parsing, dataset,
resource, priority, and cluster checks. The jobs were initially queued in
`ai2/OLMo-3-moe-experiments` on Jupiter/Ceres.

## 2026-08-18 16:09 UTC full refresh and phased-SFT launch

Seventeen of the eighteen fixed-SFT K={6,10} bridge jobs have completed. Five
complete three-run cells were added to the SFT table and plot; only fixed
train-K=8/eval-K=10 remains incomplete because replicate 2 is still running.
All three Qwen3.6 capability K=16 runs completed, were downloaded without
metric errors, and were added to the capability CSV and plot. K=16 scores are
92.73±0.31 MATH-500, 81.82±1.82 GPQA Diamond, 72.72±0.56 IFBench macro, and
56.22±0.33 AIME 2025 pass@1. The three K=32 capability jobs remain running.

All fifteen mixed-SFT extension jobs are running, while all twenty-seven
fixed-SFT K={2,14,16} extension jobs remain queued.

At the final live check, the mixed-SFT extension had reached 48--53/100
trials at K=2, 76--96/100 at K=14, 57--73/100 at K=16, 45--59/100 at K=24,
and 13--36/100 at K=32. The three Qwen3.6 K=32 capability runs had scored
2959--3206 of 4189 instances (71--77%). The lone fixed-SFT bridge straggler
(train K=8/eval K=10/replicate 2) remained at 48/100 trials; it has not
failed, but its last recorded progress was 09:29 UTC and it should be watched
as a likely long-task straggler.

Twenty-four TBLite jobs were launched for the phased/cooldown SFT checkpoint
trained at K=12, then K=8, then K=4. The submission order was K={4,8,12},
then K={6,10}, then K={2,14,16}; every K has three paired-seed replicates.
All 24 post-submit specs passed exact routing and evaluation-policy checks and
were initially queued in `ai2/OLMo-3-moe-experiments` on Jupiter/Ceres.
Across every tracked group in this refresh there were zero failed jobs.

At 16:19 UTC, the fixed-SFT train-K=8/eval-K=10/replicate-2 straggler was
manually canceled after remaining at 48/100 trials since 09:29 UTC. An exact
seed-4203 replacement was submitted as
[01M0ATRTQD55HFXYGVPVF80Q08](https://beaker.org/ex/01M0ATRTQD55HFXYGVPVF80Q08).
Post-submit inspection confirmed router/keep K=10, reference K=8, no
renormalization, and the original evaluation and scheduling settings.


At 16:36 UTC, all 51 still-queued jobs from the current SFT waves in
`ai2/OLMo-3-moe-experiments` were canceled: 24 phased jobs, 26 fixed-SFT
K={2,14,16} jobs, and the K=10 bridge replacement. The one already-running
fixed-SFT K=2 job was left untouched. No K={2,14,16} job was requeued.

The phased checkpoint was resubmitted in `ai2/holmes-testing` at
K={4,6,8,10,12}, three runs each. The queue order is K={4,8,12}, then
K={6,10}; the K=10 bridge replicate follows those 15 jobs. All 16 new jobs
are queued, and exact post-submit validation passed for workspace, routing,
evaluation settings, resources, priority, and clusters. The required Beaker
secrets were copied into the destination workspace without changing their
values.

At 16:43 UTC, the six active mixed-SFT extension runs at K={2,24} were
manually canceled to conserve compute. K=2 stopped at 59/52/56 of 100 trials
and K=24 at 60/67/49. These incomplete results are excluded. The three K=32
runs were left untouched and remain active on Jupiter.

At 16:49 UTC, the remaining fixed-SFT train-K=4/eval-K=2 replicate was
manually canceled at 2/100 trials. Its partial output is excluded. All other
running evaluations were left untouched.

## 2026-08-18 19:08 UTC results refresh

The mixed-SFT TBLite extension completed all three K=14 and all three K=16
replicates. Each artifact contains all 100 tasks. K=14 has mean reward
0.3987±0.0675 and pass@1 0.4167±0.0651; K=16 has mean reward 0.3805±0.0159
and pass@1 0.4000±0.0100 (mean ± sample SD over three runs). Both complete
cells were added to `notes/openthoughts_all_sft_tblite_results.csv` and the
shared SFT comparison plot. The canceled partial K={2,24} runs remain
excluded.

At the live check, the three retained mixed-SFT K=32 runs were at 94/100,
72/100, and 67/100 tasks. The Qwen3.6 K=32 capability runs were at
3811/4189, 3823/4189, and 3407/4189 scored instances. The first phased-SFT
Holmes evaluation (K=4 replicate 1) was at 54/100 tasks; the other fourteen
phased jobs and the fixed train-K=8/eval-K=10 bridge replacement remained
queued. There are no failed jobs in these active groups.

## 2026-08-18 21:14 UTC results refresh

All three mixed-SFT TBLite K=32 runs completed successfully with 100/100
tasks. Their aggregate mean reward is 0.3493±0.0670 and pass@1 is
0.3733±0.0666 (mean ± sample SD over three runs). The complete K=32 cell was
added to the shared SFT CSV and plot.

The three Qwen3.6 K=32 capability jobs did not complete: each reached the
24-hour Beaker runtime limit and exited with code 143. Their final scored
counts were 3839/4189, 3890/4189, and 3790/4189. These partial results are
invalid for the aggregate suite and remain excluded from the capability CSV
and plot. The first phased-SFT K=4 evaluation remains active at 94/100 tasks;
the other fourteen phased jobs and the fixed train-K=8/eval-K=10 bridge
replacement remain queued.

## 2026-08-18 22:06 UTC results refresh

The first phased/cooldown-SFT TBLite run completed successfully at eval K=4.
It contains all 100 tasks and scored mean reward 0.394332717236 and pass@1
0.41. This individual result is recorded in
`notes/openthoughts_phased_sft_tblite_results.csv`; it is not yet added to the
aggregate comparison plot because K=4 has only one of three replicates. K=4
replicate 2 has started and is bringing up its vLLM server, K=4 replicate 3
and the remaining twelve phased evaluations are queued, and the fixed
train-K=8/eval-K=10 bridge replacement remains queued. No new failures were
found.

## 2026-08-18 22:40 UTC unallocated queue migration

All fourteen eval jobs that were still queued in `ai2/holmes-testing` were
replaced with unallocated submissions: thirteen phased-SFT jobs and the one
fixed train-K=8/eval-K=10 bridge replacement. The old allocated jobs were
canceled only after their replacements were submitted and validated. Every
new spec is urgent, requests eight GPUs, targets exactly `ai2/jupiter` and
`ai2/ceres`, has unallocated/preemptible scheduling (`minRuntime=0`,
`autoResume=true`), and preserves the original routing and evaluation
settings. Four phased replacements started immediately; the other nine
phased replacements and the bridge replacement were queued. The already
running phased K=4 replicate 2 was left untouched and had reached 15/100
tasks.

## 2026-08-19 03:48 UTC refresh and MoE-workspace migration

Four new phased/cooldown-SFT results completed: K=4 replicate 2 and all three
K=8 replicates. K=4 replicate 2 scored mean reward 0.366110633903 and pass@1
0.38. The complete K=8 cell has mean reward 0.372572935660±0.051470417615 and
pass@1 0.386666666667±0.051316014394. All five completed phased runs are in
the raw phased-results CSV, and the K=8 aggregate was added to the shared SFT
CSV and plot.

The seven evals still queued in `ai2/holmes-testing` were moved back to
`ai2/OLMo-3-moe-experiments`: phased K={6,10} with three replicates each and
the fixed train-K=8/eval-K=10 bridge replicate. All replacements remain
urgent and unallocated, request eight GPUs, target exactly Jupiter and Ceres,
and passed exact post-submit validation. The seven old Holmes copies were
canceled after validation, and no queued evals from these sweeps remain in
Holmes. The phased K=4 replicate 3 remains in Holmes at 96/100 tasks, while
the three K=12 runs remain there at 17/100, 10/100, and 1/100.

## 2026-08-19 05:48 UTC results refresh

Phased-SFT K=4 replicate 3 completed with all 100 tasks, mean reward
0.334871050570, and pass@1 0.35. This closes the three-run K=4 cell at mean
reward 0.365104800570±0.029743591347 and pass@1 0.38±0.03. The raw phased
table, aggregate SFT table, and shared SFT plot were updated.

All ten remaining jobs are running and none has failed. The phased K=12 runs
are at 60/100, 48/100, and 52/100; phased K=6 at 47/100, 42/100, and 45/100;
phased K=10 at 43/100, 48/100, and 36/100; and the fixed train-K=8/eval-K=10
bridge at 48/100.

## 2026-08-19 15:55 UTC final sweep refresh

All ten remaining evaluations completed successfully. The phased/cooldown
SFT checkpoint now has complete three-run cells at every requested
K={4,6,8,10,12}: K=6 mean reward 0.407261564577±0.043032960285, K=10
0.397825405508±0.010968786050, and K=12
0.392068476496±0.052930768811. Their corresponding pass@1 aggregates are
0.423333333333±0.045092497528, 0.42±0.01, and 0.41±0.05. The raw and
aggregate phased tables and shared SFT plot were updated.

The fixed train-K=8/eval-K=10 bridge replicate also completed with mean
reward 0.439684115385 and pass@1 0.47. Together with its two earlier
replicates, the complete cell is mean reward 0.416034510446±0.028156850697
and pass@1 0.436666666667±0.035118845843. It was added to the shared SFT
table and plot, and its three raw runs were recorded separately. No jobs from
these active sweeps remain running or queued, and no failures occurred.
