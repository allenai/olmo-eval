# Adaptive compute: three-phase research plan

**Date:** 2026-08-25  
**Status:** Research framing; concrete model, evaluation, and launch matrices to be specified next.  
**Companion status document:** [`adaptive_compute_handoff_20260824.md`](adaptive_compute_handoff_20260824.md)

## Research thesis

This project asks whether the number of routed experts, `K`, can become a practical deployment axis for modern sparse Mixture-of-Experts models.

The work is organized as three increasingly ambitious phases:

1. **Inference elasticity:** establish which models tolerate static inference-time changes to `K`, why they do or do not, and whether lower `K` produces real savings.
2. **Training for cheap deployment:** test whether additional expert compute during training can improve performance at a fixed lower inference `K`.
3. **Controllable and adaptive `K`:** train one checkpoint to provide a reliable manual compute dial, then explore automatic allocation of that budget.

Each phase should stand on its own. Phase 1 is independently reportable, Phase 2 is the primary training contribution, Phase 3A is a high-value deployment goal, and fully automatic adaptation in Phase 3B is a stretch goal or potential follow-up.

## Shared principles

- Use modern models and methods available when each experiment is frozen.
- Cover both within-family scaling and across-family architectural diversity.
- Treat router scaling as part of the intervention: native routing, normalized top-`K`, and native-reference-scaled top-`K` must be distinguished.
- Use static `K` for Phases 1 and 2 unless a condition explicitly studies mixed or adaptive routing.
- Count shared experts separately from routed `K`.
- Measure quality, active FLOPs, generated tokens, and elapsed time separately. Do not collapse them into one opaque efficiency score.
- Verify actual expert dispatch with runtime telemetry; masked native-width dispatch is not a systems speedup.
- Use paired prompts and sampling where possible, and report variance, regressions, and failure rates rather than only means.
- Separate model-server time from total task or agent wall time.

## Phase 1: Inference elasticity, mechanisms, and realized efficiency

### Objective

Determine how broadly static inference-time expert reduction works across modern MoE models, characterize its failure modes, explain model-family differences, and measure the savings realized by production inference stacks.

### 1A. Cross-model benchmark

Select a model panel that jointly covers:

- multiple sizes within the same modern families;
- a wide range of total and active parameter counts, including a very large model if accessible;
- different native `K`, total expert counts, and active/total parameter ratios;
- different router score functions and normalization rules;
- models with and without shared experts;
- base, instruct, reasoning, and code-specialized post-training histories.

Evaluate static `K` curves using both absolute `K` and normalized compute quantities such as `K / K_native`, routed FLOPs, and active parameters. The core conditions are:

- untouched native routing;
- normalized fixed `K`;
- native-reference-scaled fixed `K`;
- selected above-native points where the model and implementation support them correctly.

Use a tiered evaluation suite:

1. **Screening suite:** relatively inexpensive math, science, code, instruction-following, multilingual, and long-context/retrieval tasks. Run dense `K` sweeps here.
2. **Hard suite:** frontier-sensitive reasoning, difficult code, and agentic evaluations such as TB2.1. Run native routing and selected Pareto points, with adequate replication.
3. **Behavioral stability:** response length, repetition, coherence, formatting, truncation, refusal, timeout, and tool-use failure rates.

For each result, report:

- aggregate quality and per-task quality;
- run-to-run uncertainty;
- paired prompt-level improvements and regressions as `K` changes;
- monotonicity violations;
- catastrophic-failure and degenerate-generation rates;
- output-length distributions.

### 1B. Real serving savings

Implement or validate true lower-width expert dispatch in vLLM and SGLang, initially with one fixed `K` per server or request. Benchmark under controlled prompts, output lengths, hardware, parallelism, batching, and concurrency.

Primary measurements are:

- estimated active FLOPs per input and output token;
- generated tokens per request;
- model-server elapsed time;
- decode throughput and request throughput;
- time to first token and inter-token latency;
- variance and tail latency.

Secondary bottleneck diagnostics are:

- expert-parallel communication and load balance;
- activation and workspace memory;
- achievable batch size and concurrency;
- GPU utilization and, where practical, energy;
- the point at which lower `K` shifts execution to a new bottleneck.

Resident parameter memory should be reported separately: reducing routed experts does not automatically remove expert weights from memory.

For agentic evaluations, measure both model-serving time and total agent wall time so environment and tool latency do not obscure inference effects.

### 1C. Composition with other efficiency methods

Compare `K` reduction against and in combination with a small number of strong efficiency methods. A complete Cartesian product is unnecessary; use pairwise or selected compositions that answer whether benefits are complementary, redundant, or antagonistic.

Priority interactions are:

- **Quantization:** native precision, a production FP8 setting, and one aggressive weight-only setting where supported. Record router/top-`K` identity changes and numerical ties.
- **Speculative decoding:** use the strongest reproducible method supported by the selected model and serving stack at experiment freeze. DFlash is a leading block-diffusion foundation, while newer DFlash-derived work such as [DFlare](https://arxiv.org/abs/2606.02091) and [DDTree](https://arxiv.org/abs/2604.12989) means the exact baseline should be re-verified before execution rather than hard-coded now. Measure draft/target costs, acceptance rate and length, verification cost, and end-to-end speedup in addition to quality.
- **Reasoning effort or output budget:** cross selected `K` values with effort levels or token budgets. Measure output length, success, truncation, and total request cost to determine whether expert compute and test-time tokens substitute for or complement one another.
- **Serving configuration:** selected concurrency, batch-size, tensor-parallel, and expert-parallel settings to reveal bottleneck changes.

Where useful, compare the resulting quality/cost points with serving a smaller model or a more aggressively quantized native-`K` model. A specialized low-resource serving system is outside the initial scope.

### 1D. Mechanistic expert analysis

Build a compact, reusable, tool-free prompt set sampled from several task families. Use the same prompts across models where compatible so routing and expert contributions can be compared directly without agent-environment confounds.

Collect token- and layer-level measurements of:

- selected router weights and cumulative mass by rank;
- thresholds required to retain fixed fractions of router mass;
- weighted expert-contribution norms by rank;
- cosine alignment among individual weighted expert outputs;
- alignment of the lower half of selected experts with the higher-ranked aggregate;
- alignment and norm of the dropped tail relative to the native MoE output;
- expert-identity and prefix stability as `K` changes;
- shared-expert contribution;
- layerwise and taskwise sensitivity.

Use counterfactuals to determine whether lower-ranked experts are primarily:

- small in magnitude;
- redundant and aligned with higher-ranked experts;
- corrective or canceling;
- orthogonal and functionally additive.

Gate-weight concentration alone is not an adequate explanation. The analysis should connect routing statistics to actual expert contributions and downstream `K` sensitivity.

### Phase 1 success criteria

- A common protocol across a deliberately diverse panel of modern model families and scales.
- Hard evaluations that avoid saturation for the largest models.
- At least one robust and one sensitive model characterized in depth.
- Correct, telemetry-verified static routing interventions.
- Measured end-to-end savings from true lower-width dispatch in at least one production serving stack, preferably both vLLM and SGLang.
- A defensible explanation of at least some cross-model differences rather than only empirical curves.

## Phase 2: Train hard, deploy light

### Objective

At a fixed deployment width `K_deploy`, test whether spending additional expert compute during training yields better downstream performance than training directly at `K_deploy`.

The intended claim is economic rather than literally free: extra training cost is paid once, while inference savings recur over the deployed model's lifetime.

### 2A. Core training matrix

For one or more fixed deployment widths, compare:

1. **Low throughout:** train and evaluate at `K_deploy`.
2. **High throughout:** train at native or higher `K`, then directly evaluate at `K_deploy`.
3. **High-to-low cooldown:** train at high `K`, then continue training at `K_deploy` before evaluation.
4. **High-K policy distillation:** generate strong responses or trajectories with the high-`K` policy, then train low-`K` execution on those outputs.
5. **Mixed-K reference:** train over a distribution of K values as a comparison and as preparation for Phase 3.

Evaluate the full `K_train x K_eval` behavior when affordable, not only the target low-`K` endpoint. This reveals whether a method creates a strong low-`K` subnet, broad elasticity, or dependence on its training width.

Run two complementary fairness views:

- **Matched training tokens/data:** intentionally allows high-K training to spend more compute and directly tests the deployment-amortization hypothesis.
- **Matched training FLOPs:** gives low-K training additional tokens or updates, testing whether high-K expert computation is a better use of the same training budget.

### 2B. Cooldown and consistency methods

Vary the transition into `K_deploy`:

- no cooldown;
- short and long low-K cooldowns;
- abrupt versus gradual K schedules.

If direct high-to-low transfer degrades, test online consistency during cooldown. This need not use a separate teacher architecture: the high-K execution of the same checkpoint can provide a stop-gradient reference for its low-K execution.

Candidate objectives are:

- standard next-token prediction at low K;
- output-distribution KL or cross-entropy from high-K to low-K execution;
- selected hidden-state or residual consistency if output-only alignment is insufficient;
- during later RL experiments, a policy-level KL or consistency term between paired high- and low-K rollouts.

Online dual-forward consistency requires new infrastructure and should follow simpler cooldown and policy-distillation baselines. Offline high-K policy distillation is the lower-complexity proxy.

### 2C. Training stages and controlled OLMo experiments

Study the hypothesis progressively through:

1. SFT or other inexpensive post-training gates.
2. RL/policy training where task rewards are reliable.
3. Mid-training or continued pretraining.
4. From-scratch OLMo MoE training for controlled causal comparisons.

Instrument new OLMo runs from the beginning. Evaluate `K` curves at checkpoints to measure when elasticity appears, how it scales, and whether later SFT or RL improves or damages it. Where cost permits, branch a shared early checkpoint into low-K, high-K, cooldown, and mixed-K continuations.

### Measurements

Report:

- downstream quality at the fixed deployment K and across the wider K curve;
- matched-token and matched-FLOP training cost;
- repetition, coherence, formatting, truncation, and output length;
- optimization stability and per-K training behavior;
- routing/contribution changes through training;
- real deployment FLOPs, tokens, and serving time.

For economic interpretation, estimate the deployment break-even point:

`additional training cost / per-request or per-token inference savings`.

### Phase 2 success criteria

- At fixed `K_deploy`, a high-K, cooldown, consistency, or policy-distillation method reliably improves over training low throughout under matched data/tokens.
- The result remains informative under matched training FLOPs, ideally outperforming simply training the low-K condition longer.
- Improvements hold across multiple downstream task types and do not come from pathological length or repetition changes.
- Training cost and deployment break-even are reported explicitly.
- Controlled OLMo experiments identify when and how the low-K capability is acquired.

## Phase 3: Elastic and adaptive K

### Objective

Train one checkpoint that supports multiple inference budgets. First establish a reliable manual K dial; then test whether compute can be allocated automatically according to the input or model state.

### 3A. Elastic manual K dial

Train across a supported distribution of static K values, primarily through next-token prediction and later post-training. Candidate choices include uniform, deployment-weighted, or curriculum sampling over K. Explicit budget conditioning through a token or embedding is an ablation, not a requirement.

A successful elastic model should:

- use one checkpoint across several manually selected K values;
- remain competitive with specialized fixed-K checkpoints;
- provide predictable and preferably monotonic expected quality as K increases;
- avoid K-specific repetition, coherence, and length failures;
- realize corresponding serving-cost changes.

This is already a meaningful practical result: a reasoning-effort-like dial implemented through routed expert compute.

### 3B. Automatic/adaptive allocation

Treat automatic allocation as a separate and higher-risk problem. Establish heuristic and oracle bounds before training a learned controller.

Heuristic baselines include:

- cumulative router mass;
- router entropy;
- rank margins;
- simple layerwise schedules.

Budget-matched baselines include:

- the best fixed K at the same mean cost;
- random assignment of K budgets to prompts, independent of prompt content;
- heuristic adaptive assignment;
- a repeated-sampling oracle;
- specialized fixed-K checkpoints.

Random allocation refers to randomly assigning compute budgets, not randomly choosing expert identities. For example, compare an adaptive 50/50 mixture of `K=4` and `K=8` with a random 50/50 assignment and fixed `K=6`.

Potential allocation granularities are:

1. per request;
2. per layer or block;
3. per token and layer.

A hierarchical design is attractive: the user selects a request-level budget envelope and the model distributes it across layers or tokens. This preserves predictable cost while allowing conditional allocation.

Do not lead with unconstrained RL. Begin with mixed-K next-token training, offline repeated-sample analysis, and stable heuristic controllers. If learned allocation is attempted, optimize measured expert and token cost and add explicit defenses against repetition, excessive length, and reward hacking.

### Phase 3 success criteria

**Phase 3A:** one checkpoint provides a stable, useful, and realized-cost manual K dial across a meaningful quality range.

**Phase 3B:** an adaptive policy beats the best fixed-K and random-allocation baselines at matched mean compute on held-out tasks, remains calibrated out of distribution, and respects a predictable request-level budget.

Failure to achieve Phase 3B does not invalidate Phases 1, 2, or the practical value of Phase 3A.

## Cross-phase outputs

The program should converge on three central result forms:

1. **Inference frontiers:** cross-family quality versus static K, active FLOPs, generated tokens, and measured serving time.
2. **Training transfer:** `K_train x K_eval` matrices, cooldown/consistency comparisons, and deployment break-even curves.
3. **Elasticity and adaptation:** specialized fixed-K, elastic manual-dial, heuristic, random-allocation, learned, and oracle quality/cost frontiers.

Every phase should preserve machine-readable run provenance, routing telemetry, paired sample identities, output-length statistics, and serving configuration.

## Decisions for the concrete experiment plan

The next planning step should fix:

1. The initial modern model-family and within-family scale matrix for Phase 1.
2. The screening and hard evaluation tiers, including which K points advance to TB2.1.
3. The first true-dispatch serving implementation and benchmark hardware/configuration.
4. The minimal quantization, speculative-decoding, and reasoning-effort comparisons.
5. The shared prompt subset and tensors required for mechanistic analysis.
6. The first Phase 2 model, training stage, `K_deploy`, high-K condition, and cooldown schedule.
7. The matched-token and matched-FLOP budgets for that initial training experiment.
8. Whether offline policy distillation belongs in the first Phase 2 wave or follows the basic cooldown result.
