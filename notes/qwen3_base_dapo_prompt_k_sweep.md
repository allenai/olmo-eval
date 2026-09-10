# Qwen3 base DAPO prompt-by-K generation sweep

Last updated: 2026-08-10

## Objective

Measure prompt-level success probabilities for `Qwen/Qwen3-30B-A3B-Base` at
K={4,6,8} on the DAPO-Math-17k training prompts.  The resulting paired samples
will show whether lower-K failures are prompt-dependent and monotonic enough to
support a prompt-level expert-budget policy.

## Prompt population

The Slime training JSONL contains 17,917 rows.  Exact message-text
deduplication produces 17,398 unique texts.  Seven texts have conflicting
labels and are excluded, leaving 17,391 clean prompts.  The clean manifest is:

`/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/slime-runs/qwen3-30b-a3b-base-dapo/generations/qwen3-base-dapo-k4-k6-k8-reference-n8-t1-p1-v1-20260805/prompt_manifest.json`

## Matched generation settings

- model: the local prepared HF copy of `Qwen/Qwen3-30B-A3B-Base`
- prompt: the original DAPO conversation rendered with the Qwen tokenizer's
  chat template using `add_generation_prompt=True`, matching Slime
- samples: eight per clean prompt and K
- K=4 and K=6: top-K experts with weights scaled by the native top-eight mass
- K=8: the identity case of the same reference-scaled policy
- temperature: 1.0
- top-p: 1.0
- top-k vocabulary filter: disabled (`-1`)
- response cap: 20,480 tokens
- total context: 32,768 tokens
- seed: stable hash of global seed 42, exact prompt hash, and sample index;
  therefore the eight sampling seeds are paired across K
- answer grading: Slime's exact DAPO answer extractor and reward

Every saved generation includes its prompt index/hash, sample index, sampling
seed, K policy, full response, extracted answer, correctness, prompt/completion
token counts, finish reason, and request latency.  Prompt text and labels are
stored once in the shared `prompts.jsonl` rather than duplicated in every
generation row.

## Topology and storage

The production experiment has 32 independent one-H100 tasks on unallocated
`ai2/jupiter`, urgent priority, in `ai2/holmes-testing`:

| K | Shards / H100s | Policy |
|---:|---:|---|
| 4 | 8 | reference-scaled to K=8 |
| 6 | 8 | reference-scaled to K=8 |
| 8 | 16 | identity/native K=8 |

The extra K=8 capacity follows the earlier mixed-K rollout topology and
compensates for its longer responses and larger expert kernel.  Shard outputs
are append-only and resumable under:

`/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/slime-runs/qwen3-30b-a3b-base-dapo/generations/qwen3-base-dapo-k4-k6-k8-reference-n8-t1-p1-v1-20260805`

Each shard writes `config.json`, `server.log`, `model_info.json`,
`generations.jsonl`, `progress.json`, and finally `completed.json`.

## Beaker jobs

- one-H100 K=4 reference-scaled smoke:
  [01KZ9VD7E34VCNA84VMH9KDXZE](https://beaker.org/ex/01KZ9VD7E34VCNA84VMH9KDXZE)
- 32-H100 production matrix:
  [01KZ9VMJ8N7KCAY8QF09S3SNZC](https://beaker.org/ex/01KZ9VMJ8N7KCAY8QF09S3SNZC)

At submission time, both experiments were queued behind older urgent work on
Jupiter's full unallocated queue.  The smoke is older than every production
task and validates the one-H100 server layout, output schema, DAPO scoring, and
K=4 retained top-eight mass when it starts.

Both experiments are also recorded in `notes/beaker_jobs.jsonl` under group
`adaptive-compute-qwen3-base-dapo-prompt-k-sweep-20260805`.

## Completed results

The production sweep completed successfully. Twelve initial shard executions were
preempted/canceled, but Beaker retried every affected logical shard and all final shard outputs
are complete. Each K has exactly 139,128 scored generations: 17,391 prompts times eight paired
samples, with no missing prompt/sample slots.

| Routing | Correct generations | Per-generation accuracy | Prompts with at least 1/8 correct |
|---|---:|---:|---:|
| K=4, reference-scaled | 6,949 / 139,128 | 4.995% | 4,575 / 17,391 (26.31%) |
| K=6, reference-scaled | 7,926 / 139,128 | 5.697% | 5,104 / 17,391 (29.35%) |
| K=8, native/identity | 7,928 / 139,128 | 5.698% | 5,063 / 17,391 (29.11%) |

The exact number of correct generations per prompt is:

| K | 0/8 | 1/8 | 2/8 | 3/8 | 4/8 | 5/8 | 6/8 | 7/8 | 8/8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 12,816 | 2,985 | 1,034 | 388 | 122 | 35 | 8 | 3 | 0 |
| 6 | 12,287 | 3,213 | 1,233 | 455 | 144 | 49 | 9 | 1 | 0 |
| 8 | 12,328 | 3,177 | 1,211 | 450 | 164 | 47 | 10 | 4 | 0 |

K=6 and K=8 are effectively identical in marginal per-generation accuracy. Their empirical
any-of-eight rates differ by only 0.24 percentage points in K=6's favor, which is too small to
interpret without the paired prompt/sample transition analysis. K=4 is lower but retains most
of the K=8 success rate. The concentration at zero or one success and absence of any 8/8-correct
prompt also show that single sampled outcomes are very noisy labels for a future adaptive-K
policy. Follow-up adaptive-K work should use the paired generations to estimate prompt-level
success probabilities, transition matrices, and an oracle quality-versus-average-K frontier
rather than assigning each prompt a hard minimum K from one response.

## Implementation

- generation and resumability:
  `slime/scripts/ai2/generate_qwen3_base_dapo_prompt_k.py`
- 32-task Beaker spec builder:
  `slime/scripts/ai2/build_qwen3_base_dapo_prompt_k_beaker_spec.py`
- frozen submitted production spec:
  `slime-runs/qwen3-30b-a3b-base-dapo/generations/qwen3-base-dapo-k4-k6-k8-reference-n8-t1-p1-v1-20260805/beaker_spec.json`

## Proposed preallocated-K RL pilot

The eight generations at each K are sufficient to build a conservative prompt-to-K curriculum,
but not to claim that every prompt has a deterministic minimum K. Of the 17,391 clean prompts,
8,840 (50.83%) were wrong in all 24 generations. Among the remaining prompts, success was often
specific to only one sampled K: 1,180 had a success only at K=4, 1,438 only at K=6, and 1,507 only
at K=8. That non-monotonicity makes a simple label such as “the smallest K that ever succeeded”
too sensitive to sampling noise.

The recommended first assignment uses the counts `c4`, `c6`, and `c8` of correct generations out
of eight:

1. assign K=4 when `c4 >= 1`, `c4 >= c6`, and `c4 >= c8`;
2. otherwise assign K=6 when `c6 >= 1` and `c6 >= c8`;
3. assign K=8 otherwise, including every all-zero prompt.

This is the empirically best K with a cheaper-K tie break, plus a conservative K=8 fallback when
there is no positive evidence at any K. Applied to all eight observations, it assigns 3,334 prompts
(19.17%) to K=4, 2,912 (16.74%) to K=6, and 11,145 (64.09%) to K=8. Mean assigned K is 6.898,
13.8% below static K=8 in routed-expert slots.

Selection-biased accuracy on the same samples used to choose K is 10.31% and must not be treated as
a real estimate. Leave-one-generation-out validation gives a much more credible result: 5.610%
accuracy at mean K=6.955, versus 4.995% for static K=4, 5.697% for static K=6, and 5.698% for static
K=8. Thus the proposed mapping preserves nearly all observed K=8 success while reducing assigned
expert slots by about 13%. A simpler policy based only on K=8 difficulty (`c8>=2 -> K=4`,
`c8=1 -> K=6`, `c8=0 -> K=8`) is more expensive at mean K=7.281 and has lower held-out accuracy
of 5.377%, so it is not the preferred pilot.

### Training design

- Start from the same Qwen3-30B-A3B Base checkpoint and DAPO recipe as the existing fixed- and
  mixed-K runs.
- Build an immutable assignment manifest keyed by exact prompt hash. Exact duplicate prompt texts
  receive the same K; the seven conflicting-label texts remain excluded.
- Assign one K to the entire prompt rollout group and generate eight siblings at that K. Do not mix
  K values within a prompt group in this experiment.
- Use neutral DAPO correctness/format rewards with no K bonus. Because K is constant within a
  group, a compute bonus would cancel during within-group advantage centering anyway.
- Reuse the validated named SGLang routers and variable-width routing replay. The existing one-node
  layout of one K=4 engine, one K=6 engine, and two K=8 engines is close to the natural assignment
  demand and keeps the learner/generator topology at eight physical B200s.
- Keep the natural assignment distribution for the first pilot rather than balancing K strata.
  Before dynamic filtering, the expectation is roughly 3.1 K=4, 2.7 K=6, and 10.3 K=8 candidate
  groups per 16 draws. The retained distribution will be less K=8-heavy because most all-zero
  prompts are assigned K=8 and are preferentially rejected.
- Run a 100-step pilot at the existing 1e-6 learning rate and 20,480-token response cap, saving
  every 20 steps. Match the fixed-K jobs at 16 retained prompts x eight responses = 128 responses
  per update.
- Log candidate and retained prompt counts by assigned K, realized mean K, accuracy, reward
  variance, response length, truncation, router mass, dynamic-filter rejection rate, rollout time,
  and learner time.
- Evaluate steps 20/40/60/80/100 with the corrected existing recipe at reference-scaled K=4/6 and
  native K=8: three MATH-500 seeds and one AIME pass@32 job per checkpoint/K condition.

The first pilot should keep the all-zero prompts in the population at K=8 so that the only changed
variable is prompt-specific expert allocation. A later curriculum experiment can oversample the
8,551 prompts with at least one observed success, or mix them with a smaller exploration fraction
of all-zero prompts, but doing that immediately would confound expert allocation with dataset
filtering.

### Five-hundred-step exposure and cheapest-success alternative

The current one-node recipe retains 16 prompt groups per update, so 500 updates train on exactly
8,000 retained prompt groups and 64,000 response trajectories at eight siblings per group. The
completed neutral mixed-K run actually evaluated 14,497 candidate prompt groups before dynamic
filtering to obtain those 8,000 retained groups, a 55.2% aggregate retention rate as the model
improved. Its nine-sibling mixed-K layout generated 130,473 candidate trajectories; an eight-sibling
preallocated run would generate about 116,000 if it consumed the same number of candidates. The
actual candidate count can change with the assignment policy and evolving reward rate.

The conservative empirical-best mapping is K=8-heavy before filtering: 64.09% of candidates and
mean K=6.898. It therefore reduces routed-expert slots by 13.8%, not by the 25--50% suggested by
the K=4/K=6 labels in isolation. A two-way four-sample/four-sample validation split predicts that
the retained groups would be approximately 26.1% K=4, 24.2% K=6, and 49.7% K=8, because the
all-zero K=8 candidates are rejected more often. That makes the learner's retained exposure more
balanced, but it does not recover rollout compute already spent on rejected candidates. The
learner also still executes its fixed K=8 padded replay shape, so the slot reduction applies to
SGLang rollout MoE work rather than all end-to-end training FLOPs.

An alternative “cheapest K with any observed success” rule assigns K=4 whenever `c4>0`, otherwise
K=6 whenever `c6>0`, and K=8 otherwise. Its exact distribution is:

| Assigned K | Prompts | Share |
|---:|---:|---:|
| 4 | 4,575 | 26.31% |
| 6 | 2,469 | 14.20% |
| 8 | 10,347 | 59.49% |

Mean K falls to 6.664, a 16.7% routed-expert-slot reduction from static K=8. Leave-one-generation-
out validation estimates mean K=6.757 and 5.522% held-out success, compared with mean K=6.955 and
5.610% for the conservative mapping and 5.698% for static K=8. Thus the cheaper rule buys about
three additional percentage points of expert-slot savings for an estimated 0.088-point absolute
success loss relative to the conservative mapping. A four-sample/four-sample split predicts a
retained mix of roughly 33.5% K=4, 21.9% K=6, and 44.6% K=8.

For a compute-focused first pilot, the cheapest-any-success rule is a reasonable choice: it exposes
more prompts to K=4, remains close to static K=8 under held-out validation, and better matches the
project's goal. Its labels are noisier because one lucky success is sufficient, so assignment and
retained distributions must be logged explicitly. If materially more low-K exposure is desired,
a later experiment can impose a per-update K quota such as 4/4/8 retained groups at K=4/6/8, but
that should be treated as a separate prompt-resampling intervention.

## Cheapest-any-success 100-step RL pilot

The compute-focused prompt-preallocation pilot was submitted on 2026-08-10 as
[Beaker 01KZN5GJF4YKWES52H9T6N98FK](https://beaker.org/ex/01KZN5GJF4YKWES52H9T6N98FK).
It runs urgent and unallocated in `ai2/OLMo-3-moe-experiments` on one 8xB200 Titan node.

The immutable 17,391-row training dataset assigns all eight siblings of a prompt to the cheapest
K with at least one success in the completed base-model sweep: 4,575 prompts at K=4, 2,469 at
K=6, and 10,347 at K=8. Its SHA-256 is
`03a1b8f33d1f2acbf023ecdff95d72d4cee65dc496596d9dfc628a415e727bf1`. The run uses neutral
DAPO rewards, 16 retained prompt groups x eight samples, dynamic sampling from 64 candidate
groups, LR 1e-6, a 20,480-token response cap, and checkpoints every 20 steps through step 100.
Candidate and retained distributions, realized mean K, correctness, response length, truncation,
and routing mass are written under the run's `preallocated-k-pilots/` directory.
