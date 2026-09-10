# TMax MoE RL experiment plan

Status timestamp: 2026-09-01 UTC.

## Objective

Establish whether TMax/DPPO tool-use RL works on Qwen3.5 and Qwen3.6 MoEs,
then use expert count during training as a controlled experimental variable.
The immediate target is basic end-to-end RL correctness at native K=8. Reduced
and mixed-K training follow only after that control is healthy.

## Stage 0: untouched Qwen3.5 baseline

Run three independent TBLite evaluations of `Qwen/Qwen3.5-35B-A3B` at its
native K=8. These use the same TMax-compatible protocol and serving envelope:

- dataset `openthoughts-tblite@2.0`;
- Vanillux2 agent, `qwen3_xml` tool parser, and `qwen3` reasoning parser;
- temperature 1.0, top-p 0.95, top-k 20, and 81,920 output-token ceiling;
- vLLM 0.19.1, TP=2, DP=4, four concurrent tasks, and one eight-GPU node;
- seeds 4202, 4203, and 4204;
- allocation-backed Jupiter/Ceres scheduling with `minRuntime=8h` and
  `autoResume=true`.

| Replicate | Seed | Beaker experiment | Initial status |
|---|---:|---|---|
| 1 | 4202 | `01M1DFWRBZFPHXDCRHEA1MZT4Y` | submitted |
| 2 | 4203 | `01M1DFWY00V5R35PKDPDHS8J1X` | submitted |
| 3 | 4204 | `01M1DFX3FB107XJ5VRJGG9RBAA` | submitted |

Results root:
`/weka/oe-adapt-default/jacobm/tmax-eval/qwen35-default-k8-tblite-20260901`.

## Stage 1: native-K RL bring-up

Status update (2026-09-01): paused before image completion or cluster launch.
The TMax Open Instruct branch can load a Transformers MoE under ZeRO-3, but it
does not currently carry learner expert parallelism or either form of MoE
routing replay: forward/backward routing replay or rollout-to-learner routing
replay. A shared K setting alone is therefore not sufficient to treat the
proposed run as the controlled MoE RL baseline. The next implementation
decision is between adding the full routing-trace/replay path to Open Instruct
and porting the TMax Vanillux environment to the existing Slime MoE path. The
current recommendation is the Slime port because that branch already supports
Qwen3.5/Qwen3.6 MoEs, Megatron expert parallelism, SGLang routed-expert traces,
rollout routing replay, and fixed/mixed-K controls.

Add one `moe_num_experts_per_tok` control that is applied to both the
Transformers learner and every vLLM rollout engine. Start with K=8 so this is
an explicit native-routing control rather than a reduced-compute intervention.

1. Debug Qwen3.5-35B-A3B first.
2. Run up to 100 optimizer steps, corresponding to 25,600 sampled episodes at
   eight prompts by 32 rollouts per step.
3. Save model and resumable trainer-state checkpoints every 20 steps.
4. Require successful model loading, learner/rollout weight synchronization,
   real sandbox rollouts, finite log probabilities and losses, nondegenerate
   rewards, and at least one completed optimizer step before calling the setup
   healthy.
5. Once healthy, launch the matched Qwen3.6-35B-A3B K=8 run.

The first topology uses eight eight-GPU Jupiter nodes: two learner nodes with
16 ZeRO-3 learner ranks and six rollout nodes containing 24 TP=2 vLLM engines.
This matches the published TMax rollout shape and 65,536-token response budget
while accounting for the 35B total parameter footprint of each MoE.

### Live native-K runs (2026-09-02)

The production bring-up ultimately uses a smaller three-node topology per
model: one eight-H100 Megatron learner node and two eight-H100 SGLang rollout
nodes. Both jobs run at native `K=8`, request urgent allocation-backed Jupiter
capacity, have an eight-hour protected runtime, disable automatic task retries,
and use the paid internal registry mirror. The earlier registry-authentication
failure is resolved; the mirror and all six model jobs are running.

- Qwen3.5 experiment `01M1GAYGNKYC3S4DVKXP3QTEJX`, leader job
  `01M1GAYGT36TMHG3XM7550FBHC`.
- Qwen3.6 experiment `01M1GAYJ83ZFYA2SKFTZN7DD0K`, leader job
  `01M1GAYJBM6TJQMZB3571P1CRH`.

Qwen3.5 has passed the full end-to-end gate. Rollout 0 produced 256 accepted
episodes, a mean response length of 9,865 tokens, raw reward 0.6641, no
truncation, and no repetition flag. Generation/dynamic filtering took 2,709
seconds, actor training took 688 seconds, and the SGLang weight update took 14
seconds, for a 3,415-second first optimizer step. Rollout 1 then generated its
initial 256 candidates in 1,183 seconds and continued draining already-issued
dynamic-sampling requests before postprocessing and training.

Qwen3.6 formed its full three-node placement group, loaded both learner and
rollout models, and generated all 256 rollout-0 candidates in 1,338 seconds.
At the status checkpoint its servers were still draining outstanding
dynamic-sampling requests before the first learner update. No CUDA OOM,
distributed failure, authentication failure, or model-load error is present.

Both runs have seen a few recoverable Vanillux sandbox failures: malformed
Podman stream demultiplexing and occasional missing hidden-test uploads. Slime
assigns those individual trajectories zero reward and dynamic sampling
replaces them; both models still reached 256 accepted candidates. This is a
low-rate environment-tail issue, not a run-level failure, so the healthy jobs
were left intact. Keep tracking the per-step count and fix the backend before
the internal PR if the rate grows or begins to constrain sampling.

The observed steady-state generation times plus the 11.5-minute Qwen3.5
learner update imply roughly 30--60 minutes per step. A practical 100-step ETA
is therefore about 2.5--4 days, with the first step-20 checkpoint expected
roughly 11--20 hours after steady training begins. The range remains broad
until each model completes several independent optimizer cycles.

## Stage 2: controlled-K RL

After native-K training works, compare runs from the same starting checkpoint
and seeds:

1. fixed native K=8 control;
2. fixed reduced K, initially K=4 and optionally K=6;
3. native-K training evaluated across K values;
4. reduced-K training evaluated across the same K values;
5. matched-token and, where useful, matched-training-FLOP comparisons.

Keep normalization policy explicit. The initial config-based Qwen control uses
the architecture's normal top-K renormalization. Reference-scaled or truncated
weight policies are separate arms and must be implemented identically in the
learner and rollout engine before use.

## Stage 3: combine with SFT strategies

Promote only stable RL strategies into the existing SFT matrix. Compare native,
fixed-low-K, phased/cool-down, and mixed-K SFT initializations under matched RL
conditions. Evaluate each promoted checkpoint at multiple inference K values
to separate peak performance from elasticity.

### Concrete SFT-to-RL sequence

The existing OpenThoughts SFT artifacts and TMax RL data are not fundamentally
incompatible, but they are not directly composable. The current OpenThoughts
pipeline uses the custom `olmo_thinker` tool protocol and batched terminal
observations, whereas TMax uses Qwen's native tool template and one bash call
and response per turn. Converting field names is trivial; preserving tool-call
semantics, assistant-only loss masks, and the model's native chat template is
the substantive work.

Use the following staged comparison once the native-K RL control is healthy:

1. Start from the same Qwen3.5 Instruct checkpoint used by RL and retokenize
   `allenai/tmax-sft` with the model's native tools-aware Qwen template and
   offset-derived assistant-only loss masks. Port the relevant native-tool
   normalization and masking path from the TMax Open Instruct fork into the
   OLMo-core preprocessing pipeline.
2. Train matched-token TMax-small SFT controls at K=8 and K=12, then compare
   `Instruct -> RL K=8`, `Instruct -> TMax SFT K=8 -> RL K=8`, and
   `Instruct -> TMax SFT K=12 -> RL K=8`. This isolates ordinary task-format
   alignment from the train-hard/eval-light K intervention.
3. If TMax-small helps, repeat with TMax SFT Big. Treat conversions of newer
   OpenThoughts trajectories as a later data-expansion arm because batched
   keystroke/observation turns make that conversion lossy rather than purely
   syntactic.
4. Evaluate each SFT checkpoint before RL on held-out TMax tasks, TBLite,
   TB2.1, tool-call validity, trajectory length, and a small general-capability
   suite. Retokenize separately for Qwen3.5 and Qwen3.6 because their native
   templates may differ, and scan sequence lengths before choosing the final
   context ceiling.
5. Later, collect successful high-reward RL trajectories in the native TMax
   format, rejection-sample them into an SFT set, and continue RL. Treat this
   as on-policy policy distillation, separate from the first controlled SFT/RL
   matrix.

Keep mixed-K, phased-K, and adaptive-K SFT/RL combinations behind these fixed-K
controls so any gain can be attributed cleanly.

## Evaluation and promotion criteria

- Use TBLite for rapid checkpoint monitoring and replicated summaries.
- Promote useful checkpoints to TB2.1 because TBLite alone can miss gains on
  harder tasks.
- Track reward, pass rate, task/runtime errors, tool-call validity, trajectory
  length, generated tokens, repetition/coherence diagnostics, and realized
  wall time.
- Do not interpret a run as a K experiment unless learner and rollout K are
  identical and recorded in the saved model configuration and run metadata.
