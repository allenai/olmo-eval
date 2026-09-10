# Adaptive-K training and routing roadmap

Last updated: 2026-07-22

## Goal

Train one Qwen3-30B-A3B checkpoint that remains capable across multiple active-expert budgets,
then learn when the additional experts are worth their cost. The first training experiments should
separate **robustness to K** from **learning a K policy**; combining both at once would make a gain
or failure hard to diagnose.

No training jobs have been launched from this plan.

## Recommended first pilot

Run three training conditions, initially with one training seed each:

| Condition | Rollout/training routing | Purpose |
|:--|:--|:--|
| Native control | fixed K=8 | establishes the DAPO gain without an adaptive intervention |
| Low-K control | fixed K=4, K=8-reference-scaled | tests whether training can recover low-K quality |
| Elastic | K sampled uniformly from {4,6,8}, K=8-reference-scaled | tests whether one checkpoint can support several budgets |

Use the same initial checkpoint, prompt order, optimizer settings, rollout count, and seed. One
training seed is enough for the first go/no-go comparison because each checkpoint will be evaluated
three times at every inference K. If elastic K is promising, add a second training seed for the
native and elastic conditions before making a strong training claim.

The local Qwen3-4B DAPO recipe uses eight H100s split into four learners and four TP=1 rollout
engines. Qwen3-30B-A3B has only about 3B active parameters per token, but optimizer and weight-sync
memory still scale with all 30B parameters. The current OLMo-core GRPO actor uses HSDP and does not
yet expose the MoE expert-parallel configuration used by the existing 64-H100 SFT recipe. A
realistic initial envelope is therefore 8--16 learner H100s plus 4--8 rollout H100s (12--24 total)
per training job. A short learner/weight-sync smoke must determine the actual layout before the
three production runs are queued; they should run sequentially unless substantially more capacity
is available.

## Stage 1: make the model elastic across fixed K

Start from the Qwen3 base checkpoint and use the existing DAPO math recipe/data. Compare a small,
controlled set of training conditions:

1. native K=8 throughout training;
2. fixed low K=4 with K=8-reference-scaled weights; and
3. elastic K sampled from {4, 6, 8}, also reference-scaled.

The elastic run should sample one K for an entire rollout group (all candidate responses for the
same prompt), not independently per candidate. That prevents the relative-reward normalization in
DAPO from accidentally rewarding or penalizing a candidate merely because it received a different
compute budget. Initially use the normal math correctness and format rewards with no compute
penalty. This stage asks whether exposure to several K values creates an elastic model without
sacrificing native-K quality.

Evaluate every resulting checkpoint at K={3,4,5,6,7,8} under the same current four-eval suite,
plus AIME/pass@k where useful. Cross-evaluating each checkpoint at every K is essential: a K=4
training gain that disappears at K=8 would be specialization, not useful adaptivity.

An optional fourth training condition, fixed K=6, is lower priority. It helps trace the training
curve but is not needed to answer the first elasticity question.

## Stage 2: choose one K per prompt or sequence

Sequence-level routing is the simplest policy with a real serving interpretation: inspect the
prompt once, choose K, and use it for all generated tokens and layers. Candidate selectors, in
increasing order of complexity, are:

- a hand-set mapping from task/domain or requested difficulty to K;
- a small classifier on the final prompt hidden state, trained from labels derived by evaluating
  each prompt at several K values; or
- a learned policy trained with a quality-minus-compute objective.

For supervised labels, define the cheapest K whose answer passes the task's verifier or whose
quality matches the native K=8 answer. Prompts for which no lower K succeeds remain labeled K=8.
This yields a directly interpretable classification task and an oracle upper bound: the best
possible prompt-level policy on the measured candidate K values.

Because a single sampled answer is noisy, estimate success rates from several generations per
prompt and K. A useful oracle label is the smallest K whose lower confidence bound is within a
small tolerance of K=8, rather than merely the smallest K that happened to solve the prompt once.
Start with K={4,6,8}; only add intermediate values if the three-way oracle shows useful headroom.

Three practical prompt-level implementations are:

1. a cheap text/embedding classifier that chooses K before model execution;
2. K=8 prompt prefill followed by a classifier on the final prompt state, then the selected K for
   decoding (particularly reasonable for long reasoning responses, where decode dominates); or
3. low-K prefill plus a tiny controller on early hidden/router summaries.

The first two establish whether prompt difficulty is predictable before attempting end-to-end RL.

The most useful first analysis is therefore offline and cheap: reuse matched generations across K
to estimate oracle quality/cost curves and determine how predictable the cheapest-successful K is
from prompt features. Only then train a selector.

## Stage 3: choose K per token

Token-level adaptation can save more compute because difficult reasoning steps may need more
experts than routine continuation, but it is harder to optimize and serve. A practical policy
should choose among a small menu such as K={4,6,8}, not regress an arbitrary K.

Useful signals include router-distribution features that are already available at the token:

- cumulative top-K router mass and entropy/effective expert count;
- top-rank margins;
- hidden-state or residual norms; and
- whether the token is in prompt prefill, hidden reasoning, or returned answer text.

A first nonlearned policy can map entropy or cumulative mass to K. A learned policy can add a tiny
controller head that emits a categorical K choice. For efficiency and stable kernels, choices may
need to be made per token block rather than every token. Any token-level experiment must record
realized K by layer and phase, total generated tokens, wall-clock throughput, and answer quality;
lower expert work can otherwise be offset by longer generations.

The current cumulative-mass implementation is already a token-and-layer-local policy: each router
chooses the smallest prefix of its native top eight that reaches tau. It should remain the strongest
nonlearned baseline. A serving-friendlier learned policy could instead choose one K for the next
16--32 decode tokens from rolling router entropy/margins, then apply that K across the block. This
reduces policy and kernel churn while retaining more flexibility than one K per prompt.

## Stage 4: cost-aware RL

Once an elastic checkpoint exists, add an explicit compute term. For a response with task reward
`R` and mean realized expert count `K_bar`, a simple starting objective is:

`R_total = R - lambda * K_bar / 8`.

Sweep `lambda` to produce a Pareto frontier rather than selecting one arbitrary tradeoff. For DAPO,
keep K shared within each prompt's rollout group at first. Later, allow a learned prompt-level
policy to assign group K. Token-level policy-gradient training should wait until the sequence-level
result establishes that learned allocation beats static K at a matched average budget.

Important normalization caveat: centered DAPO advantages are computed within a prompt's rollout
group. If every response in that group has the same K, the additive compute penalty is constant and
cancels during centering. That is desirable for the Stage-1 elasticity experiment, which should
have no cost term, but it will not train a K selector. A later cost-aware controller needs either a
separate policy loss across prompts, multiple K subgroups with carefully defined baselines, or
supervised oracle labels. Do not assume that adding `-lambda*K/8` to the existing centered response
reward is sufficient.

Guard against reward hacking with hard quality gates and report both raw task reward and compute
separately. The policy should not receive credit for shorter or malformed outputs that simply use
less compute.

## Required decisions before launch

- Exact Qwen3 base checkpoint and DAPO recipe revision.
- Whether all model parameters or only router/controller parameters are trainable.
- Training K distribution for the elastic run; uniform over {4,6,8} is the simplest default.
- Number of training tokens/steps needed for a meaningful but affordable pilot.
- Rollout/training GPU layout and total budget.
- Whether reference-scaled routing remains the default training intervention. Current inference
  evidence favors it because it removes much of low-K's reasoning-length penalty.

## Near-term mechanistic follow-up held for capacity

Run the planned layer-aware causal intervention after the current evaluation queue clears. The
observational layer profiles can nominate layer bands, but the causal test should hold total expert
work approximately fixed while assigning lower K to different layer groups. This distinguishes
layers that merely have diffuse router weights from layers whose extra experts actually matter to
capability.

Measure the observational quantities separately for every MoE layer: realized mean K, selected
gate mass, expert-output contribution mass, and the cosine/relative-L2 change from native K. For the
causal test, partition Qwen's 48 MoE layers into four contiguous 12-layer bands. Run K=4 in one band
and K=8 elsewhere, then rotate the low-K band; a matched-work control can exchange K=4 and K=8
between two bands so total expert calls remain constant. Three eval replicates per condition are
enough for the first pass. This experiment is explicitly deferred until the current threshold
sweep has cleared capacity, but should be the next mechanistic job rather than being forgotten.
