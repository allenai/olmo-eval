# Qwen3.6 Terminal-Bench evaluation

Updated: 2026-08-13

## Corrected K=10/K=12 replacements (2026-08-15)

Historical nominal K=10/K=12 jobs used a shallow Qwen3.6 config override and
actually routed the native eight experts. They are excluded from maintained
Terminal-Bench plots. K=10 and K=12 routing smokes now exit zero and explicitly
report the requested router K. The full legacy Terminal-Bench 2.0 replacements
have been submitted: five logical runs per K, each split into complementary
A/B shards, for twenty eight-H100 jobs total. Corrected TBLite and TB2.1
replacements were already in flight and passed the same assertion.

See `notes/qwen36_kgt8_replacement_jobs.md` for every Beaker link and exact
settings.

Summary artifacts:

- `notes/plots/qwen36_terminalbench_expert_sweep.png` and `.svg`: five-run expert-count plot
- `notes/qwen36_terminalbench_full_suite_summary.csv`: per-replicate scores and aggregate statistics

## Terminal-Bench 2.1 canonical-harness smoke

A one-task native-K=8 integration smoke was submitted on 2026-08-13 using the official Harbor
`terminal-bench/terminal-bench-2-1` dataset, Terminus 2, and Daytona. It deliberately leaves task
resources and all task/agent/verifier timeout settings at the dataset defaults. Qwen is served
locally with vLLM using TP=2, its native 262,144-token context, and Terminus receives temperature
1.0, top-p 0.95, top-k 20, and an 81,920-token output limit.

- Daytona job: [01KZWP2AEFBQ3XT3BR0TENR86V](https://beaker.org/ex/01KZWP2AEFBQ3XT3BR0TENR86V)
- Local-Podman fallback: [01KZWQ8H78HEX4MB9QCGCP1R8R](https://beaker.org/ex/01KZWQ8H78HEX4MB9QCGCP1R8R)
- Workspace: `ai2/OLMo-3-moe-experiments`
- Scheduling: urgent, allocated, two GPUs
- Cluster constraints: `ai2/jupiter`, `ai2/saturn`, `ai2/ceres`, and `ai2/titan`
- Results: `/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-35b-a3b/qwen36-tb21-k8-native-terminus2-daytona-canonical-smoke1-tp2-262k-20260813`

The smoke is successful only if local vLLM starts, Podman provisions the task, Terminus 2 makes
valid model calls and terminal actions, the verifier runs, and Harbor persists an ATIF trajectory
and reward. The task need not receive reward 1 for the infrastructure smoke to pass. Podman is the
default backend for subsequent launches; Daytona is not required.

The Daytona attempt loaded Qwen successfully on two Titan B200s, but Daytona rejected sandbox
creation before the first model call with `Organization is suspended: Depleted credits`. Harbor
records this as a per-trial exception and therefore the Beaker wrapper exits zero; it is not a
successful task. The fallback keeps the canonical TB2.1 dataset and Terminus 2 but uses the local
Podman backend to validate model calls, terminal actions, and verification independently of the
external Daytona billing state.

The Podman fallback completed on 2026-08-13 and validated the full local infrastructure path. It
created the `write-compressor` sandbox, made a valid first Qwen call, executed three terminal
commands, collected an ATIF trajectory, ran the verifier, and cleaned up normally. The task itself
scored zero with `AgentTimeoutError`: after the useful first turn, the second model call reached the
81,920-token output limit, Terminus retried it, and the task exhausted its native 900-second agent
timeout. This is a successful Podman integration smoke but also evidence that the published 80K
per-call cap can permit pathological long generations; do not treat its zero as a model score.

### Terminal-Bench 2.1 full-suite launch

Following the successful Podman infrastructure smoke, one native-K=8 attempt over the complete
89-task Terminal-Bench 2.1 suite was submitted on 2026-08-13. The launch is pinned to dataset digest
`sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a`.
The two prefix-aware task-name filters were checked against that manifest: shard A contains 45
tasks, shard B contains 44, and they have no overlap or missing tasks.

- Shard A: [01KZWW95RXHDKQ39D527QWXN8V](https://beaker.org/ex/01KZWW95RXHDKQ39D527QWXN8V)
- Shard B: [01KZWW9BSV4H502N2247XV2RS8](https://beaker.org/ex/01KZWW9BSV4H502N2247XV2RS8)
- Workspace/queue: `ai2/OLMo-3-moe-experiments`; urgent, allocated; `ai2/holmes`
- Topology: two eight-GPU shards, each with four independent TP=2 vLLM engines and four concurrent
  Harbor trials; 16 GPUs total if both schedule simultaneously
- Results root: `/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-35b-a3b/`

The full launch deliberately retains the validated smoke recipe: Terminus 2 with JSON action
parsing, Qwen3 reasoning and Qwen3-coder tool parsers, temperature 1.0, top-p 0.95, top-k 20, an
81,920-token per-call output cap, and a 262,144-token context. It leaves each task's CPU, memory,
storage, agent timeout, verifier timeout, and other timeout settings at the pinned TB2.1 manifest
defaults. No expert-routing patch is enabled, so this is native K=8.

The corrected canonical TB2.1 run completed successfully on 2026-08-13. Across both shards it
solved 31/89 tasks, for **34.83% pass@1**. There were 38 task-level exceptions: 37 native agent
timeouts (using the task-specific limits) and one terminal-input runtime error. All remain in the
89-task denominator. This is a one-attempt canonical-harness measurement, not a five-run
leaderboard estimate.

The original Jupiter/Ceres/Titan submissions (`01KZWSNNFBWRYEZ8YCV32WMEZ4` and
`01KZWSP0AE9DV1HPZF5V4T1X94`) were canceled before starting and moved unchanged to Holmes. Those
first Holmes launches (`01KZWTZ5BSWSKS12T0WP64XACT` and `01KZWTZE73MDJZWRPPNG8KDNXP`) exposed a
filtering error: package-registry task names include the `terminal-bench/` prefix, so the shard-A
glob matched zero tasks while the shard-B glob matched all 89 through the leading `t`. Shard A
failed before evaluation and shard B was stopped after two trials; both were replaced by the
prefix-aware jobs linked above. The corrected filters are `terminal-bench/[acdegikmoswy0-9]*` and
`terminal-bench/[bfhjlnpqrtuvxz]*`.

The package stack remains vLLM 0.19.1 with PyTorch 2.10.0+cu128. The first Holmes attempts proved
that this CUDA-12.8 userspace is compatible with Holmes' CUDA-13 driver: Qwen loaded, vLLM became
ready, and shard B began real Harbor trials. The unchanged vLLM wheel also contains native `sm_100`
and `sm_100a` kernels for the B300 GPUs.

## K=12 full-suite launch

Five K=12 attempts were submitted on 2026-08-12, using the same two-shard topology, Qwen coding
sampling, full 262,144-token context, 32,768-token per-turn cap, task-native timeouts, and matched
seeds as K=10. K=12 is reference-scaled to native K=8: twelve experts are routed and their weights
are scaled so that ranks 1--8 retain the native top-eight mass; the four additional expert weights
remain above that anchor. The ten urgent eight-GPU jobs are grouped at
[Beaker group 01KZVM534SVXEQ02S9Q799E8WW](https://beaker.org/orgs/ai2/workspaces/olmo-instruct/groups/01KZVM534SVXEQ02S9Q799E8WW).

## Production settings and official-Qwen comparison

The full-suite K=4/6/8/10/12 sweep uses:

- all 89 `terminal-bench@2.0` tasks through Harbor, with five complete attempts per K;
- TMax `Vanillux2Agent`, 64 agent steps, 64 recoverable format errors, and local Podman sandboxes;
- vLLM 0.19.1, native 262,144-token total context, and a 32,768-token cap on each agent turn;
- Qwen precise-coding sampling: temperature 0.6, top-p 0.95, top-k 20, with neutral min-p,
  presence-penalty, and repetition-penalty defaults;
- each eight-H100 shard contains four independent TP=2 engines, with one live trial per engine;
- the task manifest's native resources and 15--60 minute agent timeouts, without overrides.

These are controlled and identical across K, but they are not an exact reproduction of Qwen's
published Terminal-Bench 2.0 score. The Qwen3.6 model card reports its 51.5 score using
Harbor/Terminus-2, a 3-hour timeout, 32 CPUs and 48 GB RAM, temperature 1.0, top-p 0.95, top-k 20,
an 80K output cap, a 256K context, and an average of five runs. Our context and repetition count
align closely, while the agent, timeout/resources, temperature, and output cap differ. The current
sweep is therefore valid for within-scaffold expert-count comparisons, but its absolute score should
not be compared directly with Qwen's published 51.5.

### Qwen-published-settings K=8 control

One full 89-task native-K=8 control was submitted on 2026-08-13 to test how much of the absolute
score gap closes when we retain `Vanillux2Agent` but match the rest of Qwen's published setup as
closely as our harness allows. The run uses a 262,144-token context, an 81,920-token output cap,
temperature 1.0, top-p 0.95, top-k 20, a three-hour agent timeout, 32 CPUs, and 48 GB RAM per
task. It is one attempt, intended as a go/no-go test before paying for four additional attempts.

The 89 tasks are split into two complementary eight-GPU Beaker jobs. Each job hosts four TP=2
vLLM engines and runs four trials concurrently, so the pair is one logical full-suite run rather
than two repetitions. No expert-routing patch is enabled; this is the model's native K=8 behavior.

- Beaker group: [01KZWKV5YTHAJDPXDDB61BF4WF](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/groups/01KZWKV5YTHAJDPXDDB61BF4WF)
- Allocated shard A: [01KZWQN2PA2AMYC550H6H4RCVT](https://beaker.org/ex/01KZWQN2PA2AMYC550H6H4RCVT)
- Allocated shard B: [01KZWQN9Y7DHNW7WXY4Z1VYAMM](https://beaker.org/ex/01KZWQN9Y7DHNW7WXY4Z1VYAMM)
- Workspace/queue: `ai2/OLMo-3-moe-experiments`; explicit `ai2/jupiter`, `ai2/ceres`, and
  `ai2/titan` constraints; urgent and allocated
- Results: `/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-35b-a3b/`

This is still not an exact reproduction of the model-card score because the agent remains
Vanillux2 rather than Terminus-2. If it materially improves the score, repeat it four more times;
if it does not, investigate the agent/scaffold difference before expanding the run.

The control completed successfully on 2026-08-13 and solved 28/89 tasks (**31.46% pass@1**) with
23 task-level exceptions: 17 command timeouts, five three-hour agent timeouts, and one overlong
Docker argument. This is 2.70 points above the earlier five-run native-K=8 mean of 28.76%, but it
equals the best single-run score in that sweep and therefore is not evidence of a reliable settings
gain by itself. It also remains below Qwen's published 51.5, consistent with the remaining
Vanillux-versus-Terminus scaffold difference.

The original unallocated submissions (`01KZWKSKN8NSV33RS7QY94X77S` and
`01KZWKSTRRW4GBY2ZD87MJ05M0`) and Jupiter-only allocated replacements
(`01KZWM7M44QXW7NNZCKCE52TZW` and `01KZWM81RMH4G0XJZZ2ESWXHVB`) were canceled before scheduling.
The current jobs request Beaker's non-preemptible/allocated queue while retaining urgent priority
and all evaluation settings, and can schedule on any of the three explicitly listed clusters.

## Goal

Measure whether Qwen3.6-35B-A3B retains terminal-agent capability when fewer than its native eight
routed experts are evaluated. Qwen3.6 is the first model because it reports materially stronger
Terminal-Bench 2.0 performance than Qwen3.5 while retaining the same 256-expert, top-8 plus shared
expert architecture used by our existing Qwen3.5 routing work.

## Staged experiment

1. Run a one-task K=8 infrastructure smoke test.
2. After the smoke verifies model startup, reasoning/tool parsing, bash execution, verification, and
   result persistence, run the same 20 Terminal-Bench 2.0 tasks at:
   - K=8 default routing;
   - K=6 reference-scaled routing;
   - K=4 reference-scaled routing.
3. Use one attempt per task for the pilot. Decide whether to expand the task set and/or add repeated
   attempts only after inspecting scores, infrastructure-error rates, trajectories, token usage, and
   wall time.

The reference-scaled conditions route the native top eight experts, retain the highest K weights,
zero the lower-ranked weights, and do not renormalize. At K <= 8 this is exactly the same intervention
used in our olmo-eval experiments.

## Fixed settings

- Model: `Qwen/Qwen3.6-35B-A3B`
- Dataset: `terminal-bench@2.0`
- Agent: upstream `Vanillux2Agent`
- Environment: local Docker-compatible containers through tmax's Beaker Podman setup
- Model server: vLLM 0.19.1, one H100, TP=1, DP=1, language-model-only mode,
  `gpu_memory_utilization=0.95`, `max_num_seqs=4`, and Triton GDN prefill
- Context/output limits: 110,000 model context and up to 32,768 generated tokens per agent step
- Sampling: temperature 0.7 and top-p 0.95
- Agent limits: 64 steps and 64 recoverable format errors
- Parsing: `qwen3` reasoning parser and `qwen3_coder` tool-call parser
- Beaker: `ai2/olmo-instruct`, `ai2/jupiter`, urgent priority, no explicit allocation
- Results: `/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-35b-a3b/`

The pilot conditions deliberately differ only in the routed-expert intervention. The same dataset,
task limit, agent, sampling settings, and evaluator are used for all three.

The first K=8 smoke (`01KZDN09ZM59PA8ZM6GF6ZZ9K9`) proved that the model weights fit on one H100
(64.69 GiB), but tmax's original 0.85 memory-utilization default left no KV-cache space after its
default 512-sequence CUDA-graph reservation. It exited before serving or evaluating a task. The
revised values above match the actual four-request concurrency and leave substantially more cache
headroom.

The second K=8 smoke (`01KZDNTWRW0X1WQ127ZTV63HPF`) validated the corrected vLLM settings: the
server became ready and Harbor started. The stored `hamishivi_DAYTONA_API_KEY` was rejected before an
environment was created, so no prompt reached the model. Subsequent runs use the upstream pipeline's
local Podman backend and authenticated Docker Hub pulls instead of Daytona.

The third K=8 smoke (`01KZDPC8BR382AVF4QTN2SW20T`) validated the entire local sandbox and verifier
path, including ten correctly parsed bash-tool turns with no infrastructure exception. The selected
`gpt2-codegolf` trajectory then exceeded the initial 40,960-token context at step 10 and scored zero.
That result is invalid as a capability measurement. vLLM reported 9.01 GiB of available KV cache and
an effective maximum concurrency of 9.93 at 40,960 tokens, so the revised single-H100 smoke uses a
110,000-token context while retaining the 32,768-token per-turn generation cap.

The fourth K=8 smoke (`01KZDQ24ZT0EZXCPR3JSTBP828`) completed the full 64-step trajectory and a real
verifier run with no infrastructure exception or context truncation. The deliberately difficult
single task (`gpt2-codegolf`) scored zero, which has no scientific meaning for a one-task smoke. It
used 2,008,592 cumulative prompt tokens and 74,744 completion tokens across the repeated agent turns;
the agent phase took about 10.25 minutes after model startup. This validates the 110,000-token pilot
configuration while also showing why the 20-task results should be treated as an initial directional
sample rather than a cheap high-precision benchmark.

## Implementation

The upstream tmax Beaker pipeline is used at commit
`7387d2f9142397a458dc39f0827a2ab0b4c03cda`. Local launcher changes add:

- Qwen3.6's recommended reasoning and tool parsers;
- a vLLM `--hf-overrides` passthrough for future normalized-K runs;
- the validated Qwen reference-scaled selected-weight intervention;
- explicit propagation and logging of routing parameters.

Launcher: `scripts/adaptive_experts/launch_qwen36_terminalbench.sh`.

## Initial 20-task pilot results

All three pilots completed with Beaker exit code 0. Scores use the intended denominator of all 20
tasks, including tasks that timed out or raised a sandbox-command exception.

| Routing condition | Correct / 20 | Raw pass@1 | Agent timeouts | Sandbox command timeouts | Wall time |
|---|---:|---:|---:|---:|---:|
| K=8 native | 6 | 30% | 5 | 2 | 1h 25m 44s |
| K=6 reference-scaled | 6 | 30% | 7 | 2 | 1h 30m 57s |
| K=4 reference-scaled | 5 | 25% | 5 | 3 | 1h 26m 32s |

Four tasks were solved by all three conditions: `log-summary-date-ranges`,
`modernize-scientific-stack`, `portfolio-optimization`, and `prove-plus-comm`. K=8 and K=4 also
solved `largest-eigenval`; K=8 and K=6 also solved `merge-diff-arc-agi-task`; and K=6 alone solved
`reshard-c4-data`. Thus K=8 and K=6 tie in aggregate but are not identical task by task, while the
entire five-point K=4 gap is one task on this small sample.

The vLLM logs explicitly confirm that the intervention matched Qwen3.6 router layers at native
top-K=8 with 256 routed experts, retaining K=6 or K=4 without renormalization. The result is
directionally encouraging, especially for K=6, but the single attempt and unequal 7--9 exception
counts make the pilot too noisy for a strong claim about a five-point difference. A follow-up should
either repeat these exact tasks with corrected/expanded execution timeouts or expand the task sample
while reporting exception rates separately.

## Corrected full-suite plan

The initial runs were a 20-of-89-task, one-attempt pilot rather than a full Terminal-Bench 2.0
evaluation. The next phase keeps the official task timeout multiplier at 1.0 and leaves all task
resource and agent-timeout overrides unset, but removes two avoidable pressure points from the local
inference setup:

1. Validate TP=2 serving at Qwen3.6's native 262,144-token context with one concurrent trial and
   `max_num_seqs=1`.
2. On the same 20 K=8 tasks, compare the existing TMax sampling profile
   (`temperature=0.7`, `top_p=0.95`) against Qwen's precise-coding recommendation
   (`temperature=0.6`, `top_p=0.95`, `top_k=20`). Both retain the 32,768-token per-turn output cap.
3. Select one sampling profile using score, timeout/context-stop rate, and trajectory quality rather
   than score alone.
4. Run all 89 tasks once at native K=8 and reference-scaled K=6/K=4, with no `--n-tasks` limit.
5. Expand promising conditions to the leaderboard-quality minimum of five attempts per task only
   after the one-attempt full-suite comparison.

One trial per engine is important here: the benchmark's task-specific 15--60 minute agent timeouts
are wall-clock limits, so four long-context trajectories sharing one H100 can create avoidable
latency-related failures even though the configured timeout values themselves are correct.

### Execution status (2026-08-07)

The TP=2/full-context smoke (`01KZEEBS6MG8Y2Y3HBTKF4N2JR`) successfully loaded Qwen3.6 at a
262,144-token context and executed 42 parsed agent/tool steps without an OOM, server error, parser
failure, or context-limit warning. The first benchmark task then reached its native 15-minute agent
timeout and Harbor correctly persisted an `AgentTimeoutError`, metrics, and the trajectory. This is a
valid infrastructure smoke outcome, not a model-score measurement.

The initial one-engine versions of the matched 20-task sampling pilots were stopped and replaced with
four-engine versions to reduce wall time. Each replacement uses eight H100s as four independent TP=2
replicas, with four concurrent Harbor trials and `max_num_seqs=1` per replica:

- TMax defaults (`temperature=0.7`, `top_p=0.95`): `01KZEK50R2FNZZCHJB91FT8RN4`.
- Qwen precise-coding settings (`temperature=0.6`, `top_p=0.95`, `top_k=20`):
  `01KZEK55BW97J88FT48DP4N93V`.

Both use native K=8, the 262,144-token context, identical first 20 tasks, one attempt, and task-native
resources/timeouts. Each trial still has a dedicated engine, so this does not reintroduce the
contention from the original single-GPU/four-concurrent pilot. The all-89-task K=8/K=6/K=4 launch is
intentionally gated on this A/B comparison.

Both four-engine pilots completed successfully in about 95 minutes of Beaker wall time apiece:

| Sampling profile | Correct / 20 | Raw pass@1 | Agent timeouts | Other task errors |
|---|---:|---:|---:|---:|
| TMax defaults | 6 | 30% | 7 | 0 |
| Qwen precise-coding | 7 | 35% | 6 | 3 |

The six tasks solved by the TMax profile were also solved by the Qwen profile; Qwen additionally
solved `reshard-c4-data`. Neither run showed an OOM or context-length failure. Qwen's three other
errors were two sandbox-command timeouts (`RuntimeError`) and one generated command whose argument
list exceeded the OS limit (`OSError`), rather than model-server failures. This is only a one-task
difference on one attempt, but the Qwen profile is the preferred full-suite setting because it is the
model-recommended coding configuration, solved a strict superset here, and had one fewer native agent
timeout. The error counts must still be reported alongside full-suite scores.

The initial 16-GPU-per-job full-suite submissions were rejected because Jupiter nodes expose eight
GPUs each. The replacement preserves eight engines per routing condition by using two complementary
eight-GPU shards, each with four independent TP=2 engines. The actual TB2 manifest was checked before
launch: shard A contains 45 tasks, shard B contains 44, all 89 tasks are covered exactly once, and
their summed native agent-timeout budgets are 19.60 and 21.33 hours respectively.

The one-attempt full-suite sweep uses the Qwen profile, full context, and task-native settings:

| Routing condition | Shard A | Shard B |
|---|---|---|
| K=8 native | `01KZEZVJYQKP7TS9RG8PWHNMDC` | `01KZEZVR92752GFM8H1CKJ1NKA` |
| K=6 reference-scaled | `01KZEZVX7S4DX8113HR86XG259` | `01KZEZW2F7HNYWFNYNQW0ZAD9J` |
| K=4 reference-scaled | `01KZEZW76WKGX3H34E785FE5PC` | `01KZEZWC58X4YKVWD6J45EJSP3` |

## Full-suite five-run results

> **2026-08-15 routing audit:** the historical K=10 and K=12 rows below are
> not valid expert-count interventions. Their commands set a nonexistent
> outer `num_experts_per_tok` field on Qwen3.6's composite config. The vLLM
> logs contain no router-layer match marker, so those jobs silently remained
> at native K=8. K=4, K=6, and K=8 remain valid. Do not use the historical
> K=10/12 values or plot points for scientific comparisons; corrected nested-
> config replacements are tracked in
> `qwen36_tblite_reference_sweep_jobs.md` and
> `qwen36_tb21_reference_sweep_jobs.md`.

Five full 89-task runs are complete at every routing condition from K=4 through K=12. Replicates
2--5 use matched seeds 4202--4205 across all five conditions. Raw pass@1 retains all 89 tasks in the
denominator, so native agent timeouts and sandbox-command errors count as failures rather than being
dropped.

| Routing condition | Rep. 1 | Rep. 2 | Rep. 3 | Rep. 4 | Rep. 5 | Five-run mean | Correct / 445 | Task errors / 445 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K=8 native | 28.09% | 28.09% | 28.09% | 28.09% | 31.46% | **28.76%** | 128 | 169 |
| K=6 reference-scaled | 26.97% | 28.09% | 25.84% | 28.09% | 28.09% | **27.42%** | 122 | 178 |
| K=4 reference-scaled | 30.34% | 25.84% | 25.84% | 22.47% | 28.09% | **26.52%** | 118 | 176 |
| K=10 reference-scaled | 30.34% | 30.34% | 33.71% | 31.46% | 26.97% | **30.56%** | 136 | 163 |
| K=12 reference-scaled | 24.72% | 31.46% | 30.34% | 26.97% | 30.34% | **28.76%** | 128 | 160 |

The five-run point estimates order K=10 > K=8 = K=12 > K=6 > K=4. Relative to K=8, K=10 is
higher by 1.80 percentage points (eight tasks across 445 attempts), K=12 is exactly tied in aggregate,
K=6 is lower by 1.35 points (six tasks), and K=4 is lower by 2.25 points (ten tasks).
K=4 also varies more from run to run, showing that its initially strong 30.34% result was optimistic.
K=12 also varies substantially (24.72--31.46%), and therefore does not reproduce K=10's mean gain.
The differences remain modest compared with the high per-run task-error rate, so paired task-level
analysis is needed before treating the ordering as statistically decisive. The completed K=10 retry
also confirms that the apparent failed shard was infrastructure-only. There were no model-server or
routing failures in the retained runs; task-level agent timeouts and sandbox/runtime failures remain
in the score denominator.

### Output-token usage

Every one of the 445 task attempts per routing condition reports `agent_result.n_output_tokens`.
This is cumulative model output across all assistant turns in the terminal-agent trajectory, not
just the final message. The first standard deviation below is the sample standard deviation across
all 445 task trajectories; it is large because task length and the number of agent turns vary widely.

| Routing condition | Mean output tokens / task | SD across tasks | Min | Max |
|---|---:|---:|---:|---:|
| K=4 reference-scaled | 59,910 | 92,054 | 243 | 483,795 |
| K=6 reference-scaled | 63,548 | 94,319 | 308 | 482,651 |
| K=8 native | 69,308 | 106,264 | 390 | 537,501 |
| K=10 reference-scaled | 70,020 | 106,254 | 389 | 523,772 |
| K=12 reference-scaled | 69,296 | 104,834 | 407 | 515,990 |

Aggregating each complete 89-task replicate first gives a clearer view of run-to-run variation:

| Routing condition | Mean of five run means | SD of run means |
|---|---:|---:|
| K=4 reference-scaled | 59,910 | 2,906 |
| K=6 reference-scaled | 63,548 | 1,645 |
| K=8 native | 69,308 | 4,175 |
| K=10 reference-scaled | 70,020 | 4,331 |
| K=12 reference-scaled | 69,296 | 5,211 |

Cumulative output length rises from K=4 through K=8, but K=8, K=10, and K=12 are nearly identical:
K=10 is only 711 tokens per task (1.0%) above K=8, while K=12 is effectively equal to K=8. Both
differences are much smaller than run-to-run variation.

That all-attempt comparison is partly confounded by different success and failure trajectories. The
primary successful-task analysis therefore keeps every task that earned reward 1 at least once at
each of K=4, K=6, K=8, and K=10 across the five replicates. For each task and K, output length is
first averaged over that K's successful replicas; the resulting 29 common tasks then receive equal
weight, so a condition with more successful replicas cannot dominate the result:

| Routing condition | Mean output tokens / common task | SD across task means |
|---|---:|---:|
| K=4 reference-scaled | 12,229 | 16,475 |
| K=6 reference-scaled | 15,359 | 26,706 |
| K=8 native | 15,807 | 22,476 |
| K=10 reference-scaled | 12,721 | 19,540 |

On this common-task success subset, K=4 is 22.6% shorter than K=8, K=6 is 2.8% shorter, and K=10
is 19.5% shorter. Successful-replica coverage is similar but not identical: 105 K=4, 109 K=6, 115
K=8, and 112 K=10 successful attempts feed the within-task averages. The standard deviations remain
large and the set is selected toward mutually solvable tasks, so the differences are directional.

As a stricter sensitivity check, requiring the exact same seeded task-replicate attempt to succeed
at all four K values leaves only 59 attempts across 21 tasks and changes the lower-K ordering. K=10
remains the shortest condition in both definitions, while K=4 versus K=8 is not robust to the
success-matching rule.

### Failure decomposition

The five runs give 445 task attempts per routing condition. A task-level exception is not always a
failed verifier: four K=4 agent-timeout trajectories and two K=8 agent-timeout trajectories left a
correct artifact behind and still received reward 1. The table therefore classifies failures by the
actual reward first.

| Outcome | K=4 reference | K=6 reference | K=8 native |
|---|---:|---:|---:|
| Correct | 118 (26.52%) | 122 (27.42%) | 128 (28.76%) |
| Submitted, but verifier failed | 77 (17.30%) | 75 (16.85%) | 82 (18.43%) |
| Exhausted all 64 agent steps, verifier failed | 78 (17.53%) | 70 (15.73%) | 68 (15.28%) |
| Overall agent timeout, verifier failed | 111 (24.94%) | 119 (26.74%) | 108 (24.27%) |
| One shell command exceeded 120 seconds | 61 (13.71%) | 58 (13.03%) | 58 (13.03%) |
| Oversized Docker argument list | 0 | 1 (0.22%) | 1 (0.22%) |

These are predominantly model/agent trajectory failures, not model-server or Beaker failures. The
only reasonably smooth change with K is a reduction in 64-step exhaustion: 78 at K=4, 70 at K=6,
and 68 at K=8. Agent timeouts and completed-but-wrong submissions are not monotonic, so the modest
score gain is not explained by one single failure mechanism.

Across the 356 exactly seed-matched attempts in replicates 2--5, K=4 and K=8 had the same binary
outcome 302 times (84.8%). K=8 converted 33 K=4 failures into successes, while K=4 converted 21
K=8 failures into successes, for a net K=8 gain of 12. At the task level, 63 of 89 tasks have the
same five-run success count at K=4 and K=8; 18 favor K=8 and eight favor K=4. Forty-four tasks were
never solved in any of the 15 K=4/6/8 attempts, while eight were solved in all 15. This says that
task difficulty and sampling variance dominate expert count for most tasks, although K=8 has a
small positive aggregate effect.

The clearest K=8-over-K=4 gains were `regex-log`, `extract-elf`, `sqlite-db-truncate`,
`sparql-university`, and `llm-inference-batching-scheduler` (each +2 successes out of five). The
largest reversals were `cobol-modernization` and `kv-store-grpc` (-3 each), followed by
`overfull-hbox` (-2). Those reversals are further evidence that a single task attempt is too noisy
to label a capability as requiring one particular K.

## K=10 full-suite launch

Five K=10 attempts completed on 2026-08-10 with the same Qwen sampling, task-native timeouts,
262,144-token context, and two complementary 8-GPU shards per attempt. K=10 routes ten experts and
scales them so that the top-eight weights retain the native K=8 mass; the two additional weights sit
above that anchor. Replicates 2--5 reuse seeds 4202--4205.

| Replicate | Shard A | Shard B |
|---:|---|---|
| 1 | [01KZN34G8C4Q9N3GP23KP2MNQM](https://beaker.org/ex/01KZN34G8C4Q9N3GP23KP2MNQM) | [01KZN34N98T4A1DN4SJ6252V2G](https://beaker.org/ex/01KZN34N98T4A1DN4SJ6252V2G) |
| 2 | [01KZN34T3PE2FTGDEPZ6CPX8MR](https://beaker.org/ex/01KZN34T3PE2FTGDEPZ6CPX8MR) | [01KZN34Z9SRRFEZSKSFMPXCPW4](https://beaker.org/ex/01KZN34Z9SRRFEZSKSFMPXCPW4) |
| 3 | [01KZN354G1NFHTD6XNGQFWPN1H](https://beaker.org/ex/01KZN354G1NFHTD6XNGQFWPN1H) | [01KZN359DSRSP85RKZAWZFKP29](https://beaker.org/ex/01KZN359DSRSP85RKZAWZFKP29) |
| 4 | [01KZN35EXFBRCVDDKTCSY6ANVR](https://beaker.org/ex/01KZN35EXFBRCVDDKTCSY6ANVR) | [01KZN35KVFM6QPT5APQKXRJC0S](https://beaker.org/ex/01KZN35KVFM6QPT5APQKXRJC0S) |
| 5 | [01KZN35RJPS8YTK5D5SPH41X47](https://beaker.org/ex/01KZN35RJPS8YTK5D5SPH41X47) | [01KZN35XXFSTJJ93TQRDVAGF2W](https://beaker.org/ex/01KZN35XXFSTJJ93TQRDVAGF2W) |

Replicate-1 shard A was automatically retried after its first execution landed on a Jupiter node
that was cordoned for an unrecoverable hardware error. The retry succeeded and its complete output
is included in the five-run table above; this was a pre-execution cluster failure, not a model failure.

## K=12 full-suite completion

All ten K=12 shards completed successfully on 2026-08-13. They used the same Qwen precise-coding
sampling, task-native timeouts, native 262,144-token context, 32,768-token per-turn cap, two-shard
topology, and paired seeds as the earlier conditions. K=12 routes twelve experts and uses K=8 as its
reference scale. The resulting five-run mean is 28.76% (128/445), tied with native K=8 in aggregate
and below K=10's 30.56%; its observed run range is 24.72--31.46%. The jobs are grouped at
[Beaker group 01KZVM534SVXEQ02S9Q799E8WW](https://beaker.org/orgs/ai2/workspaces/olmo-instruct/groups/01KZVM534SVXEQ02S9Q799E8WW).

## Interpretation safeguards

- Infrastructure failures count as failures in the raw pass rate but must also be reported separately.
- The one-task smoke score has no scientific meaning.
- A 20-task, one-attempt pilot is directional and should not be treated as a precise full-suite score.
- Trajectories should be inspected for parser/format failures and output truncation before attributing
  differences to expert count.

## TB-Lite timeout-relaxed capability run (2026-08-13)

To separate Qwen's underlying capability from the high task-native timeout rate in the 89-task
Terminal-Bench 2.1 runs, we launched a one-attempt K=8 evaluation on the 100-task
`openthoughts-tblite@2.0` Harbor dataset. This is a capability-oriented diagnostic, not a directly
comparable Terminal-Bench 2.1 leaderboard number.

The run deliberately preserves the existing Qwen generation profile: temperature 1.0, top-p 0.95,
top-k 20, an 81,920-token per-call output ceiling, and a 262,144-token context. It uses the TMax
`Vanillux2Agent` with 64 agent steps. The overall Harbor agent wall-clock timeout is disabled; shell
commands retain a six-hour timeout, and task/setup/build/verifier timeout multipliers are set to 10.
The sandbox remains Podman, so the experiment measures how much removing premature timeouts helps
without changing the agent or Qwen sampling policy.

The corrected two-task smoke is [01KZY1408S70QYEKKG8BNGH8B4](https://beaker.org/ex/01KZY1408S70QYEKKG8BNGH8B4).
It confirmed that Harbor selected both requested tasks, vLLM served native K=8 routing, and the agent
executed terminal steps with the timeout-disable patch active. Both tasks completed without an
exception and earned reward 1; this validates the setup but is not an accuracy estimate. The first attempted smoke
([01KZY0JF16216FAH73CHY22AK8](https://beaker.org/ex/01KZY0JF16216FAH73CHY22AK8))
was intentionally superseded after Harbor reported that TB-Lite task filters use bare names rather
than dataset-prefixed names; no evaluation ran in that attempt.

The full 100 tasks are partitioned exactly once across two complementary first-letter shards, each
using four independent TP=2 engines on eight GPUs:

| Shard | Tasks | Beaker experiment |
|---|---:|---|
| A (`[acdegikmoswy0-9]*`) | 55 | [01KZY1QZ1JNK2BB4EJKC6WZ9MA](https://beaker.org/ex/01KZY1QZ1JNK2BB4EJKC6WZ9MA) |
| B (`[bfhjlnpqrtuvxz]*`) | 45 | [01KZY1R2SN6YWRYC6NNQGMAATB](https://beaker.org/ex/01KZY1R2SN6YWRYC6NNQGMAATB) |

Both full jobs are urgent, allocated, and target Jupiter, Ceres, or Titan in the
`ai2/OLMo-3-moe-experiments` workspace. Launch provenance and result directories are also recorded
in `notes/beaker_jobs.jsonl`.

The uncapped K=8 diagnostic was stopped manually on 2026-08-13 after reaching 95/100 completed
tasks (52/55 on shard A and 43/45 on shard B) with zero recorded task errors. The five remaining
trials had stopped making useful progress and, because the diagnostic disabled its overall agent
deadline, would not have terminated on their own. The observed reward sum over the 95 completed
tasks was approximately 68; this is retained as a partial diagnostic rather than reported as a
complete 100-task score.

## TB2.1 timeout-relaxed TMax run (2026-08-13)

To pair the TB-Lite diagnostic with the actual final benchmark, one native-K=8 attempt over the
pinned 89-task Terminal-Bench 2.1 suite was launched through the same TMax `Vanillux2Agent` path.
This retains the Qwen profile used by the TB-Lite diagnostic: temperature 1.0, top-p 0.95, top-k 20,
an 81,920-token per-call output ceiling, a 262,144-token context, and 64 agent steps. The overall
Harbor agent wall-clock timeout is disabled; individual shell commands retain a six-hour bound, and
task/setup/build/verifier timeout multipliers are 10. Local Podman remains the sandbox backend.

This run is directly useful for measuring the effect of relaxed timeouts relative to the earlier
canonical-timeout TB2.1 run, but it is not a canonical leaderboard configuration because the agent
and timeout policy differ. The pinned 89 tasks are partitioned exactly once into the already
validated 45/44 shards:

| Shard | Tasks | Beaker experiment |
|---|---:|---|
| A (`terminal-bench/[acdegikmoswy0-9]*`) | 45 | [01KZY22WXHT9NDMC8MSK4R35Z8](https://beaker.org/ex/01KZY22WXHT9NDMC8MSK4R35Z8) |
| B (`terminal-bench/[bfhjlnpqrtuvxz]*`) | 44 | [01KZY230T4CXRG6TWQGACS20A6](https://beaker.org/ex/01KZY230T4CXRG6TWQGACS20A6) |

Both jobs are urgent, allocated, and target Jupiter, Ceres, or Titan in the
`ai2/OLMo-3-moe-experiments` workspace. They use four independent TP=2 engines per eight-GPU shard.

## Recommended-timeout expert sweeps (2026-08-13)

One full attempt at each of K=4, K=6, K=10, and K=12 was submitted for both TB-Lite 2.0 and the
pinned 89-task Terminal-Bench 2.1 suite. Each logical attempt is split into the same two validated
complementary shards used above. The TB-Lite submissions were created first, followed by TB2.1.

The first submissions used an 8,100-second (2h15m) agent limit, but were canceled on user request
shortly after launch and superseded before scientific collection. The active replacements use
10,800 seconds (3h) per agent and 2,700 seconds (45m) per shell command. Environment build, agent
setup, overall task, and verifier multipliers remain at 10 so
Podman overhead is not charged against the model's agent allowance. All other settings match the
timeout-relaxed K=8 diagnostic: Vanillux2Agent, temperature 1.0, top-p 0.95, top-k 20, 81,920 output
tokens per call, 262,144-token context, 64 steps, Podman, and four independent TP=2 engines per
eight-GPU shard.

K=4 and K=6 truncate the native top-eight router distribution without renormalizing. K=10 and K=12
route the requested number of experts and use K=8 as the reference scale, matching the routing
definitions in the earlier TB2.0 expert sweep.

| Benchmark | K | Shard A | Shard B |
|---|---:|---|---|
| TB-Lite 2.0 | 4 | [01KZYQB8ZWFBHV98KKY6C94WZ2](https://beaker.org/ex/01KZYQB8ZWFBHV98KKY6C94WZ2) | [01KZYQBD2WZQ2JVGHF7ZQFHQWX](https://beaker.org/ex/01KZYQBD2WZQ2JVGHF7ZQFHQWX) |
| TB-Lite 2.0 | 6 | [01KZYQBHFT49KE26Y1C9SBZHX6](https://beaker.org/ex/01KZYQBHFT49KE26Y1C9SBZHX6) | [01KZYQBPCYQQKHCH6RHD34T38J](https://beaker.org/ex/01KZYQBPCYQQKHCH6RHD34T38J) |
| TB-Lite 2.0 | 10 | [01KZYQBT439J6EWKJ0NB1JJQWK](https://beaker.org/ex/01KZYQBT439J6EWKJ0NB1JJQWK) | [01KZYQBY4X4T7450C4A23HW399](https://beaker.org/ex/01KZYQBY4X4T7450C4A23HW399) |
| TB-Lite 2.0 | 12 | [01KZYQC20ETC6Q6FJ68AXYV63E](https://beaker.org/ex/01KZYQC20ETC6Q6FJ68AXYV63E) | [01KZYQC5PYQ5FKSFJP0Z7SKS1A](https://beaker.org/ex/01KZYQC5PYQ5FKSFJP0Z7SKS1A) |
| TB2.1 | 4 | [01KZYQCFE24QQVPC8VBQEHJP3P](https://beaker.org/ex/01KZYQCFE24QQVPC8VBQEHJP3P) | [01KZYQCK9G08ZGJDT4HWF4N2RC](https://beaker.org/ex/01KZYQCK9G08ZGJDT4HWF4N2RC) |
| TB2.1 | 6 | [01KZYQCQ4T29QFPJT0TC1KKW87](https://beaker.org/ex/01KZYQCQ4T29QFPJT0TC1KKW87) | [01KZYQCVKCA51ETYR740TFNEHD](https://beaker.org/ex/01KZYQCVKCA51ETYR740TFNEHD) |
| TB2.1 | 10 | [01KZYQCZVTCJV9NB7V0C6Q98Q0](https://beaker.org/ex/01KZYQCZVTCJV9NB7V0C6Q98Q0) | [01KZYQD3F9H8FX88QPKDJSZ2QY](https://beaker.org/ex/01KZYQD3F9H8FX88QPKDJSZ2QY) |
| TB2.1 | 12 | [01KZYQD780YME3YNPGVVVNR0XN](https://beaker.org/ex/01KZYQD780YME3YNPGVVVNR0XN) | [01KZYQDB49NEA6WF4X9QK1DMMR](https://beaker.org/ex/01KZYQDB49NEA6WF4X9QK1DMMR) |

All sixteen shards are urgent, allocated, and may schedule on Jupiter, Ceres, or Titan in
`ai2/OLMo-3-moe-experiments`. The reusable launcher is
`scripts/adaptive_experts/launch_qwen36_terminal_recommended_timeouts.sh`; setting `REPLICATE` and
`EXPERT_COUNTS` permits later matched repetitions without changing the evaluation recipe. The
replacement jobs and cancellation lifecycle are also recorded in
`notes/qwen36_terminal_3h_jobs.jsonl`.

## Finished-run-only TB-Lite and TB2.1 plots (2026-08-14)

Two incremental figures now track only complete, comparable logical runs:

- `notes/plots/qwen36_tblite_finished_expert_sweep.{png,svg}`
- `notes/plots/qwen36_tb21_finished_expert_sweep.{png,svg}`
- `notes/plots/qwen36_tblite_finished_expert_sweep_through_k32.{png,svg}`
- `notes/plots/qwen36_tb21_finished_expert_sweep_through_k16.{png,svg}`

The latter two are presentation-focused range views; they use the same completed-run data and
filtering as the full-range figures.

Their per-run source tables are `notes/qwen36_tblite_finished_runs.csv` and
`notes/qwen36_tb21_finished_runs.csv`. The reusable collector/plotter is
`scripts/adaptive_experts/plot_qwen36_tblite_tb21_finished.py`.

A run is admitted only when all named components exist, component task sets do not overlap, and
their union contains exactly 100 TB-Lite tasks or 89 TB2.1 tasks. Partial, canceled, OOM-killed,
and corrupted attempts are excluded automatically. TB2.1 is additionally restricted to the
Vanillux2/Qwen-generation recipe with the 10,800-second agent cap; the separate Terminus2 canonical
run and the uncapped K=8 capability diagnostic are not mixed into the expert-count curve.

At the first refresh, TB-Lite contains one complete point: reference-scaled K=4 replicate 1 has
mean reward 59.9482%, pass@1 61%, and one task exception across 100 tasks. TB2.1 has no complete
valid run under the comparable capped recipe yet, so its figure is an explicit empty placeholder.
K=6 TB2.1 shard B (`01KZYQCVKCA51ETYR740TFNEHD`) was excluded after Beaker reported an OOM kill
at 23/44 tasks. It was replaced by capped shard B
[`01KZZCYBGKAH707VQZXVWGTK7C`](https://beaker.org/ex/01KZZCYBGKAH707VQZXVWGTK7C).
The uncapped native-K=8 TB2.1 attempt was stopped at 76/89 after more than 12 hours because it had
no terminal agent deadline and did not match the capped sweep. Its comparable capped replacements
are shard A [`01KZZCYFDHZZ7HBB8TGP3YXC31`](https://beaker.org/ex/01KZZCYFDHZZ7HBB8TGP3YXC31)
and shard B [`01KZZCYKAEKCM264D122B0KEKY`](https://beaker.org/ex/01KZZCYKAEKCM264D122B0KEKY).
An additional native-K=8 TB-Lite replicate was also submitted as
[`01KZZCYSJZZ0Y39E3GV844SPKD`](https://beaker.org/ex/01KZZCYSJZZ0Y39E3GV844SPKD).

At the 2026-08-14 14:37 UTC refresh, K=10 TB2.1 shard A completed successfully with 45/45 tasks,
35.56% mean reward, and five task exceptions. Shard B remained active at 41/44, so the logical
89-task K=10 run remains excluded until that shard finalizes. No other new complete logical run was
available: the finished-run tables still contain only TB-Lite K=4 replicate 1 and no capped TB2.1
point. The tables and both plots were nevertheless regenerated so they remain strict views of the
latest complete data.

At 15:24 UTC, 13 still-running jobs whose progress counts had been unchanged for at least three
hours were stopped. Their missing tasks are explicitly counted as failures in
`notes/qwen36_terminal_forced_timeout_components.csv` and
`notes/qwen36_terminal_forced_timeout_logical.csv`; methodology and the readable summary table are
in `notes/qwen36_terminal_forced_timeout_results.md`. Forced-final results remain separate from the
strict complete-run curves because canceled jobs did not preserve task-level artifacts and their
TB-Lite scores had to be reconstructed from three-decimal heartbeat means.

The pathological TB-Lite K=10 replicate 2 forced score (52 unfinished tasks) and the forced-final
TB2.1 K=12 score are retained in the provenance CSV but explicitly excluded from analysis. The
complete K=10 replicate 1 result remains included.

To isolate whether the timeout-relaxed recipe caused the stalls, one native-K=8 TB-Lite logical
replicate was submitted on 2026-08-14 using the original stable Terminal-Bench sweep recipe. It is
split across two complementary 8-GPU shards with TP=2, DP=4, and four concurrent trials per shard;
uses temperature 0.6, top-p 0.95, top-k 20, a 32,768-token turn cap, and seed 4202; and restores the
native task timeouts and Vanillux's default 120-second command timeout. No timeout or resource
multipliers are set. Shard A is [01M00EHBXY7EAYEM1B5E3DGDMW](https://beaker.org/ex/01M00EHBXY7EAYEM1B5E3DGDMW)
and shard B is [01M00EHG2VDA0XSBNSNYCTKNZ5](https://beaker.org/ex/01M00EHG2VDA0XSBNSNYCTKNZ5).

The OpenThoughts-Agent-100K SFT checkpoint received the same native-K=8 TB-Lite comparison on
2026-08-14. It uses the identical stable sampling, seed, task timeouts, command timeout, and shard
topology, while retaining that checkpoint's native 65,536-token model context. Shard A is
[01M00ETH27K23D64FJGMS0T4SN](https://beaker.org/ex/01M00ETH27K23D64FJGMS0T4SN) and shard B is
[01M00ETNKSY10WSWS50DQ1M034](https://beaker.org/ex/01M00ETNKSY10WSWS50DQ1M034).
