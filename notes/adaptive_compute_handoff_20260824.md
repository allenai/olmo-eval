# Adaptive compute project handoff

**Audit date:** 2026-08-24  
**Workspace:** `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute`  
**Scope of this document:** persisted local code, notes, result bundles, plots, and job ledgers. Live Beaker state was not re-queried for this audit. The newest persisted result artifacts found locally were updated on 2026-08-21.

## Executive summary

This project studies whether a sparse Mixture-of-Experts model's number of active routed experts, `K`, can be reduced or otherwise controlled after training. There are three increasingly ambitious questions:

1. How much capability can be retained when inference uses fewer active experts than the model's native routing configuration, and does this produce real serving savings?
2. If deployment has a fixed low-`K` budget, can spending more compute during pretraining or post-training produce a better low-`K` model?
3. Can a single model expose `K` as a reliable runtime quality/cost knob, potentially selecting compute adaptively by prompt, token, or layer?

The strongest result so far is that inference-time expert reduction often works, but its reliability depends substantially on model family, task, and routing semantics. Several models retain most aggregate quality at roughly half their native routed width. GPT-OSS-120B is especially robust from native `K=4` to `K=2`; Qwen3 chat becomes broadly usable by about `K=4--5` versus native `K=8`; and Nemotron-3-Super degrades gradually from native `K=22` to `K=11`, with most of the loss concentrated in GPQA. The clearest counterexample is Qwen3.5-35B-A3B, where HumanEval collapses at low `K` even when other tasks remain usable. Qwen3.6 also shows that the best `K` can depend on the benchmark: higher `K` helps GPQA, IFBench, and Terminal-Bench over part of the range while hurting MATH and AIME.

Routing scale is part of the intervention, not an implementation detail. Standard top-`K` renormalization amplifies the surviving expert branch when `K` is reduced. Preserving the native-`K` denominator frequently recovers low-`K` quality and avoids excessive hidden-reasoning length, especially for off-the-shelf Qwen models. It is not universally superior: on the corrected Dolci reasoning SFT, reference scaling is worse than normalization at several sub-native `K` values. This difference is scientifically interesting because it suggests post-training changes the model's dependence on router calibration.

The current project has not yet established end-to-end production savings. Some vLLM evaluation hooks retain a fixed-width dispatch and zero masked slots, so those runs test a routing counterfactual but not a cheaper expert kernel. Slime's SGLang rollout path can execute true lower `K`; however, mixed-`K` learner batches are padded to the largest `K` and masked, and response-length changes can dominate end-to-end cost. There is no controlled vLLM/SGLang throughput, latency, GPU-memory, or energy study yet. Lower `K` also does not reduce parameter memory unless experts are offloaded or sharded differently.

Training work is promising but not yet a clean demonstration of "train hard, evaluate light." Fixed-`K` DAPO, mixed-`K` DAPO, prompt-preallocated DAPO, and fixed/mixed/phased Qwen3.5 SFT runs exist. A single mixed-`K` RL model can match specialized checkpoints at multiple inference widths on MATH, but the comparisons are not compute-matched. The OpenThoughts SFT pilot does not show a simple rule that training at high `K` produces the best `K=4` deployment model. No controlled from-scratch pretraining or mid-training experiment has been run.

Adaptive routing has a useful baseline but no decisive learned advantage. A cumulative-router-mass threshold can match the fixed-`K` frontier at some budgets; `tau=0.8` approximately matches native aggregate Qwen quality while using about 24% fewer routed experts on average. Across the sweep, however, threshold routing generally lies on rather than above the best fixed-`K` frontier. Prompt-level `K` labels are noisy and often non-monotonic, and a mixed `K={4,8,12}` RL run developed a severe low-`K` repetition pathology. These are important design constraints for a learned controller.

No approximately one-trillion-parameter model, systematic quantization interaction, or production serving benchmark is present in the current evidence. The largest evaluated families are around 120B total parameters. Those are major gaps for the proposed report.

## What was recovered from the earlier planning work

I did not find a standalone document containing the exact unified three-section tech-report outline described in the current request. The earlier session may have held that synthesis without persisting it, or it may have been spread across several documents.

The closest references are:

- [`adaptive_experts.md`](adaptive_experts.md): the original cross-model inference-time expert-count roadmap.
- [`adaptive_training_and_rl_plan.md`](adaptive_training_and_rl_plan.md): the staged fixed-`K`, mixed-`K`, selector, controller, and cost-aware RL roadmap. Its statement that no training jobs had launched is now stale.
- [`qwen35_adaptive_k_sft_plan.md`](qwen35_adaptive_k_sft_plan.md): fixed-`K`, mixed-`K`, and curriculum SFT design.
- [`mixed_k_cost_aware_rl.md`](mixed_k_cost_aware_rl.md): the implemented mixed-`K` RL system and results.
- [`slime_qwen3_base_dapo_prompt_k_transitions.md`](slime_qwen3_base_dapo_prompt_k_transitions.md): evidence about whether prompts have stable minimum-`K` labels.
- [`project_overview.md`](project_overview.md): the broadest historical synthesis, but it predates several Qwen3.6, SFT, and mixed-`K` results and should not be treated as the final authority.

The three-part report framing in the current request should therefore be treated as authoritative. A candidate mapping from the recovered work to those sections appears near the end of this handoff.

## Repository and system map

The workspace contains several nested repositories rather than one clean monorepo:

| Path | Role in this project |
| --- | --- |
| `olmo-eval/` | Capability evaluations, routing interventions, analysis, plots, result collection, and most experiment notes. |
| `slime/` | DAPO/RL training, mixed-`K` rollout routing, Megatron routing replay, and SGLang rollout engines. |
| `tmax-eval/` | Terminal-Bench/TBLite/TB2.1 agent evaluation, routing patches, task filtering, timeout and parser support. |
| `tmax/` | Terminal-agent runtime dependency; no current local changes were found. |
| `terminal-bench/` | Terminal-Bench checkout/data; not a Git repository in this workspace. |
| `results/` | Downloaded evaluation artifacts, roughly 3.9 GiB in the audit. |
| `slime-runs/` | Training checkpoints and run artifacts, roughly 52 TiB in the audit. |

### Local Git state and preservation warning

The workspace is not in a handoff-safe Git state. Do not clean, reset, switch branches destructively, or assume untracked files are disposable.

- The parent repository at `/weka/oe-adapt-default` sees the adaptive-compute workspace as untracked.
- `olmo-eval` is on `main` at `origin/main`, with 12 tracked modifications and major untracked project content including `notes/`, `scripts/adaptive_experts/`, compatibility code, suites, `sitecustomize`, and tests.
- `slime` is on `main` at `origin/main`, with eight tracked modifications plus untracked mixed-`K`, routing, startup, test, and lockfile content.
- `tmax-eval` is on `master`, one commit ahead of its upstream, with two modified launch scripts and untracked configurations/routing patches. The local commit adds deterministic sample seeding to `Vanillux2Agent`.
- `tmax` was clean at audit time.

This means a large fraction of the project's reproducibility-critical code and documentation may exist only in this workspace. Before major new development, capture the current changes in intentional branches/commits or a durable patch/archive, while preserving experiment artifacts separately.

## Definitions and routing semantics

The term `K` is easy to misuse in this project. Every future result should state the routing semantics explicitly.

### Native `K`

The model's trained number of routed experts per token. Shared experts, where present, are not counted in `K`. Examples in the current work include Qwen native `K=8`, GPT-OSS native `K=4`, GLM-4.5-Air native `K=8`, and Nemotron-3-Super native `K=22`.

### Normalized fixed `K`

Select the top `K` routed experts and normalize their weights to sum to one. This is the standard top-`K` interpretation, but at sub-native `K` it increases each survivor's effective scale because removed router mass is redistributed.

### Reference-scaled fixed `K`

Select the target top `K`, but preserve the denominator implied by a reference width, normally the native `K`. For `K < K_ref`, the retained weights sum to less than one; for `K > K_ref`, they can sum to more than one. This preserves the scale of the trained top-ranked branches more closely than renormalization. It should not be confused with simply multiplying arbitrary raw router scores.

The name "non-renormalized" appears in older notes. "Reference-scaled" is clearer because the exact denominator matters.

### Adaptive cumulative-mass routing

For each token/layer, select the smallest expert prefix whose cumulative router mass reaches a threshold `tau`. The selected weights can then be either renormalized or reference-preserved. Current evidence strongly favors reference preservation at low budgets.

### Routing intervention versus real sparse execution

Two implementations can produce similar logits while having different systems costs:

- A fixed-width kernel can select the native number of slots, mask unwanted experts, and assign zero weight. This is useful for quality experiments but may perform the same expert dispatch work.
- A true variable-width kernel dispatches only the selected experts. This can reduce routed expert FLOPs and communication, subject to load balance and kernel efficiency.

Every performance claim must identify which behavior was used. "Fewer active experts" in a result table does not by itself prove lower wall-clock cost.

## Source-of-truth hierarchy

The notes contain historical entries, launch-time expectations, partial collections, and later corrections. Use the following precedence:

1. Final machine-readable CSVs and downloaded result bundles in `notes/` and `results/adaptive_experts/`.
2. Focused experiment notes that explicitly describe corrections and plot-admission rules.
3. The append-only [`beaker_jobs.jsonl`](beaker_jobs.jsonl) ledger for launch provenance.
4. Live Beaker state for jobs that may have changed since the last local collection.
5. Broad summaries such as [`project_overview.md`](project_overview.md) and chronological [`latest_status.md`](latest_status.md), which can contain stale intermediate states.

The job ledger contained 2,981 records during this audit. It records submission state, not final correctness. The local tree contained 104 plot files and many derived CSVs. The newest persisted result updates were from 2026-08-21; no live cluster refresh was performed on 2026-08-24.

## Current inference-time evidence by model family

### Cross-family overview

| Model family | Native `K` | Evaluated range | Main finding | Important limitation |
| --- | ---: | ---: | --- | --- |
| Qwen3-30B-A3B Base | 8 | 1--16 | Quality improves rapidly through roughly `K=4--5`; high-`K` behavior is non-monotonic. | Some historical high-`K` math runs have a random-stream confound. |
| Qwen3-30B-A3B chat/hybrid | 8 | 1--32 | Broad quality plateau begins around `K=5`; reference scaling greatly improves low `K`. | Very low normalized `K` changes hidden-reasoning length; `K=32` has long-tail/truncation failure. |
| GPT-OSS-120B | 4 | 1--8 | Extremely robust at `K=2`, approximately half native width. | Mechanistic follow-up is still missing; non-power-of-two values use a compatibility fallback. |
| GLM-4.5-Air | 8 | 1--8 | Most quality recovered by `K=4--5`. | Historical protocol only; no current-suite, quantization, or adaptive-routing study. |
| Nemotron-3-Super-120B-A12B-FP8 | 22 | 11--22 | Smooth degradation; `K=11` retains most aggregate quality, with GPQA carrying most loss. | Evaluation is FP8 but mechanistic analysis is BF16 because Transformers lacks the ModelOpt FP8 path. |
| Qwen3.5-35B-A3B | 8 | 4--8 | Strong counterexample: low `K` remains costly, especially on HumanEval. | Only a narrow range was evaluated. |
| Qwen + corrected Dolci reasoning SFT | 8 | 3--12 | Training changes the low-`K` curve and reverses some reference-scaling benefits. | Direct attribution to a particular post-training stage is unresolved. |
| Qwen3.6-35B-A3B | 8 | 2--128, task-dependent | Different tasks prefer different widths; terminal benchmarks show a broad mid-range plateau and extreme-`K` collapse. | Above-native reference scaling adds expert-branch magnitude; normalized above-native comparisons remain incomplete. |

There is no evaluated approximately 1T-parameter model. The current largest models are around 120B total parameters.

### Qwen3-30B-A3B Base

The OLMoBase sweep covers `K={1,...,12,16}`. Score-based suites improve quickly through approximately `K=4--5`, remain strongest over a middle range, and can collapse at `K=12` or `K=16` even while BPB improves. This is a warning against treating language-model likelihood and downstream generation quality as interchangeable, and against assuming that more active experts monotonically improve an off-distribution routing intervention.

Some historical high-`K` math comparisons were not generated from fully paired random streams. They should support qualitative shape claims, not fine-grained effect sizes.

### Qwen3-30B-A3B chat/hybrid

The historical full sweep shows a clear transition from unusable to broadly capable:

| `K` | Historical macro score |
| ---: | ---: |
| 1 | 0.00 |
| 2 | 0.25 |
| 3 | 36.88 |
| 4 | 59.74 |
| 5--16 | roughly 63--67 |
| 32 | 48.62 |

The current four-evaluation suite has paired normalized and native-reference-scaled `K=3--16` runs. For normalized routing, the four-evaluation means are:

| `K` | Mean score |
| ---: | ---: |
| 4 | 70.20 |
| 5 | 74.48 |
| 6 | 76.71 |
| 7 | 76.98 |
| 8 | 76.46 |

Reference scaling is particularly important at the lower edge: `K=4` improves from 70.20 to 74.03, and `K=3` improves from 37.87 to 67.97. This supports the interpretation that normalized low-`K` routing harms the model partly by amplifying the surviving MoE branch rather than merely by removing expert contributions.

Response-length analysis strengthens that mechanism. Relative to native `K=8`, normalized routing increases all-prompt median generation length to approximately `1.29x` at `K=4`, `1.14x` at `K=5`, `1.09x` at `K=6`, and `1.03x` at `K=7`. On common-correct prompts, reference-scaled `K=4` uses about 3,816 full generated tokens on average versus 5,131 for normalized `K=4` and 4,024 for native `K=8`. Returned answer text is much more similar; most of the difference is hidden reasoning.

The same pattern appears by task. On GPQA, normalized `K=4` has `1.586x` the native median length, while reference-scaled `K=4` is `1.007x` and scores 4.71 points higher than normalized. On IFBench, normalized `K=4` is `1.308x`; reference scaling reduces this to `1.119x` and improves score by 6.14 points.

This matters for systems claims: reducing expert work per token can fail to reduce request cost if it causes the model to generate many more tokens.

### GPT-OSS-120B

GPT-OSS is the clearest positive cross-family result. Native routing uses `K=4`, but the current suite remains essentially flat from `K=2` upward:

- Normalized: `K=1` 26.82, `K=2` 83.10, and `K=2--8` spans approximately 82.11--83.95.
- Reference-scaled: `K=1` 37.22, `K=2` 81.41, and `K=3--8` spans approximately 82.83--84.52.

Thus half-width inference is close to native aggregate quality. Above-native routing also remains valid. Non-power-of-two expert counts required a vLLM compatibility fallback, so any serving benchmark should verify that its kernel path is comparable across values.

GPT-OSS is an important candidate for mechanistic comparison with Nemotron and Qwen because its router is unusually robust. That analysis has not yet been completed.

### GLM-4.5-Air

The historical macro curve is:

| `K` | Macro score |
| ---: | ---: |
| 1 | 0.17 |
| 2 | 31.04 |
| 3 | 57.58 |
| 4 | 63.42 |
| 5, 7, 8 | roughly 64--65 |

This is another strong half-width result, but it lacks the newer paired protocol, reference-scaling study, hard terminal benchmark, quantization comparison, and mechanistic follow-up. It should be replicated before carrying precise numbers into a report.

### Nemotron-3-Super-120B-A12B-FP8

Nemotron uses 512 routed experts, a shared expert, sigmoid router scores with a factor of five, and native `K=22`. Its normalized current-suite curve degrades gradually:

| `K` | Macro score |
| ---: | ---: |
| 11 | 74.67 |
| 13 | 76.43 |
| 15 | 77.30 |
| 17 | 78.64 |
| 19 | 78.54 |
| 21 | 79.00 |
| 22 | 79.38 |

Most of the gap is GPQA: 59.43 at `K=11` versus 71.55 at `K=22`; math is comparatively flat. This is a useful example of aggregate robustness hiding a benchmark-specific dependency.

An earlier "raw x5" `K=11/22` intervention produced near-zero generation. It is not a valid reference-scaled comparison because it does not preserve the trained denominator. It is useful only as a negative control demonstrating how sensitive the model is to branch scale.

The BF16 mechanistic analysis found a relatively flat selected-expert distribution: entropy 3.041 and effective selected width 20.97. The top 11 experts account for 60.08% of router mass but 71.25% of expert contribution norm. A reference-preserving top-11 counterfactual has total-output cosine 0.9774 to native. Router concentration alone therefore does not explain the residual quality loss. Layer-to-layer variation is material, and an actual native-weight truncation evaluation is still needed.

### Qwen3.5-35B-A3B off-the-shelf

Qwen3.5 is the best current example of a model that is not very inference-adaptive. All three repetitions of normalized and reference-scaled `K=4--8` are complete. Representative endpoints are:

| Routing | `K` | MATH | GPQA | IFBench | HumanEval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normalized | 4 | 78.80 | 78.96 | 34.02 | 24.80 |
| Normalized | 8 | 90.60 | 83.16 | 49.14 | 85.77 |
| Reference-scaled | 4 | 80.47 | 80.13 | 45.76 | 62.40 |
| Reference-scaled | 8 | about 91.13 | 82.49 | 48.04 | 87.80 |

Reference scaling rescues a large portion of low-`K` HumanEval and IFBench quality but leaves a substantial native-width gap. This model should be central to the "when and why does adaptivity fail?" part of the proposed report.

### Corrected Dolci terminal-EOS reasoning SFT

The corrected Qwen + OLMo-3 reasoning checkpoint has complete three-repetition normalized and reference-scaled `K=3--12` results. Four-evaluation means are:

| `K` | Normalized | Reference-scaled |
| ---: | ---: | ---: |
| 3 | 41.78 | 45.63 |
| 4 | 48.86 | 47.53 |
| 5 | 52.79 | 50.29 |
| 6 | 52.81 | 52.12 |
| 7 | 54.81 | 53.71 |
| 8 | 56.40 | 57.63 |
| 9 | — | 57.99 |
| 12 | 54.50 | 55.95 |

Unlike the off-the-shelf Qwen case, reference scaling is worse at `K=4--7`. HumanEval remains especially sensitive. Post-training therefore appears to change both the shape of the `K` curve and the appropriate routing scale. A report should avoid presenting reference scaling as a universal fix.

### Qwen3.6-35B-A3B capability results

The corrected capability table uses reference scaling for non-native `K` and native routing at `K=8`:

| `K` | MATH | GPQA | IFBench | AIME pass@1 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 95.07 | 78.28 | 66.83 | 66.11 |
| 6 | 93.40 | 79.46 | 69.79 | 62.19 |
| 8 | 93.53 | 80.47 | 71.92 | 60.59 |
| 10 | 92.73 | 80.64 | 72.65 | 59.24 |
| 12 | 92.60 | 81.82 | 72.59 | 57.15 |
| 16 | 92.73 | 81.82 | 72.72 | 56.22 |

There is no single task-independent optimum: GPQA and IFBench generally rise with `K`, while MATH and AIME fall over much of the same range. This could reflect routing specialization, sampling/length effects, or reference-scaled above-native branch magnitude. It is strong motivation for per-workload or adaptive compute allocation, but it also demands normalized above-native controls.

## Hard terminal-agent benchmarks

Terminal benchmarks are valuable because they combine long-horizon reasoning, tool use, coding, and environment interaction. They are also operationally noisy: task exceptions, parser failures, timeouts, scaffold differences, and context limits can materially affect scores.

### Qwen3.6 Terminal-Bench 2.0 full suite

The corrected five-run summary is:

| `K` | Pass rate | Standard deviation | Passed / 445 |
| ---: | ---: | ---: | ---: |
| 4 | 26.52% | 2.93 | 118 |
| 6 | 27.42% | 1.00 | 122 |
| 8 | 28.76% | 1.51 | 128 |
| 10 | 31.46% | 2.38 | 140 |
| 12 | 31.24% | 3.50 | 139 |

Use [`qwen36_terminalbench_full_suite_summary.csv`](qwen36_terminalbench_full_suite_summary.csv) as the authority. Earlier snippets in `latest_status.md` were written before the above-native routing audit and can refer to invalid nominal `K=10/12` runs.

### Qwen3.6 TBLite

Three repetitions are complete:

| `K` | Pass rate |
| ---: | ---: |
| 2 | 0.2700 |
| 4 | 0.6000 |
| 6 | 0.6200 |
| 8 | 0.6300 |
| 10 | 0.6367 |
| 12 | 0.6300 |
| 14 | 0.6500 |
| 16 | 0.6733 |
| 18 | 0.6533 |
| 20 | 0.6033 |
| 22 | 0.6667 |
| 24 | 0.6667 |
| 26 | 0.6400 |
| 28 | 0.6200 |
| 30 | 0.6667 |
| 32 | 0.6700 |
| 64 | 0.5300 |
| 128 | 0.2900 |

The main shape is a sharp transition from `K=2` to `K=4`, a noisy broad plateau from roughly `K=4--32`, then collapse at extreme width. The high-`K` collapse is important evidence that expert count is a calibration knob rather than a monotonic compute-for-quality axis.

### Qwen3.6 Terminal-Bench 2.1

Three repetitions are complete:

| `K` | Pass rate |
| ---: | ---: |
| 4 | 0.2846 |
| 6 | 0.3184 |
| 8 | 0.3221 |
| 10 | 0.3221 |
| 12 | 0.3483 |
| 14 | 0.3184 |
| 16 | 0.3296 |
| 32 | 0.3371 |
| 64 | 0.2622 |

This again suggests a noisy mid-range plateau and degradation at extreme `K`. A separate canonical Terminus2 native-`K=8` attempt passed 31/89 tasks (34.83%) while recording 38 task exceptions. A published-settings Vanillux control passed 28/89 (31.46%). Those values are scaffold checks, not directly interchangeable with model-card claims.

Terminal errors and timeouts remain in the denominator. Reported scores must include the exact agent scaffold, task revision, timeout, parser policy, context window, and exception counts.

### Above-native routing correction

The original Qwen3.6 nominal `K=10/12` runs changed only a shallow outer configuration field. Because Qwen3.6 is a composite model, the true routing value lives under `text_config.num_experts_per_tok`; the model continued to execute native `K=8`. Those points were excluded and replaced. Current admitted CSVs use the corrected nested configuration and a startup assertion that verifies the runtime value.

This failure mode is general enough to become a project rule: every run should log and assert observed selected width from telemetry, not infer it from a launch argument.

## Mechanistic findings

### Router weight order and scale matter

On Qwen, shuffling the selected weights while retaining the same expert identities collapses generation, and assigning uniform weights is also severely harmful. Expert identity alone is insufficient; relative rank weights carry essential information.

For native Qwen routing, the top four experts contain about 63.31% of gate mass and 69.23% of expert-contribution norm. Rank eight activation norm is only about 58.1% of rank one, so low-ranked experts do not secretly compensate with abnormally large activations.

A normalized local `K=4` counterfactual has cosine 0.9188 to the native `K=8` MoE output but norm ratio 1.377 and relative L2 error 0.596. This is direct evidence that low-`K` renormalization amplifies the expert update. It explains why preserving the native denominator can improve both quality and generation behavior.

### Qwen3.6 expanded-width behavior

The native top eight experts contain only 19.38% of all-256 raw router mass; top 16 contain 27.55%, and top 32 contain 39.30%. Ranks 9--32 jointly contain 19.93%.

At reference-scaled `K=32`, total routed weight is 2.205. With normalized `K=32`, the original top eight account for 47.09% of selected contribution. The added tail has 45.65% of native routed-output norm and is nearly orthogonal to the native routed output, with cosine 0.073. Reference-scaled `K=32` raises routed-output norm to 1.121 times native and complete-layer norm to 1.073; normalized `K=32` lowers those to 0.528 and 0.781 respectively.

The interpretation is that reference scaling preserves the trained top-eight branch while adding a substantial, almost orthogonal tail. This can plausibly help some tasks and hurt others. A normalized `K=16/32` benchmark is needed to separate "more experts" from "larger MoE branch."

There is also a numerical caveat: under BF16, widening from 8 to 32 changes top-eight membership for about 19% of token/layer cases because of router-score ties. The fused vLLM path may resolve ties differently. Exact expert-identity reproduction across implementations should not be assumed without telemetry.

### Why model families differ remains open

The evidence rules out a simple explanation based only on concentration of router mass. Nemotron's retained prefix has high output cosine but still loses GPQA, while GPT-OSS remains nearly flat at half width. Candidate explanatory variables include:

- router normalization and score function;
- relative expert-branch scale and shared-expert contribution;
- redundancy versus specialization across selected experts;
- layer-local sensitivity;
- effects of post-training on routing calibration;
- task dependence and generation length;
- quantization and fused-kernel numerics.

The project has measured pieces of these but not a matched cross-family mechanistic panel.

## Adaptive inference results

### Cumulative-mass threshold routing

Reference-preserving adaptive routing on Qwen produced:

| `tau` | Mean selected `K` | Macro score |
| ---: | ---: | ---: |
| 0.5 | 3.498 | 68.63 |
| 0.6 | 4.271 | 73.66 |
| 0.7 | 5.144 | 74.08 |
| 0.8 | 6.107 | 76.52 |

Relative to the nearest fixed-`K` point, the score differences are approximately `+0.65`, `-0.36`, `-1.73`, and `+0.08`. The adaptive policy therefore sits mostly on the fixed-width Pareto frontier rather than clearly above it.

`tau=0.8` is still operationally interesting: it approximately matches native `K=8` aggregate quality while selecting about 24% fewer experts on average. The aggregate hides task movement, including HumanEval down 3.46 points and offsets from stronger GPQA/IFBench.

Renormalized threshold routing is catastrophic at low thresholds. Adaptive selection and scale preservation must be treated jointly.

These experiments use a fixed-width vLLM masking path and do not demonstrate wall-clock speedup.

## Training and post-training evidence

### Fixed-`K` Qwen3 Base DAPO

Three main fixed-routing runs reached step 500:

- normalized `K=8`;
- normalized `K=6`;
- reference-scaled `K=4`.

Full three-repetition checkpoint MATH evaluations exist. The curves show that useful policies can be trained at each width and transferred across inference widths, but low-width training is less stable. The reference-scaled `K=4` run fell from 81.33 at step 350 to 73.00 at step 400 before recovering to 77.80/78.47 at steps 450/500. Reference-scaled `K=6` moved from 80.20 at step 350 to 78.93 at step 500; `K=8` reaches roughly 83 or higher.

Wall-clock cost is not a clean proxy for expert FLOPs. Early runs had similar end-to-end behavior, while later `K=8` rollouts became more expensive partly because generation behavior differed. Token-normalized and kernel-level measurements are still required.

### Mixed `K={4,6,8}` DAPO

The implementation uses four SGLang rollout engines on one eight-B200 node: one `K=4` EP2 engine, one `K=6` EP2 engine, and two `K=8` EP2 engines, with the learner colocated. Both a cost-shaped reward (`+0.10/+0.05/+0` for correct lower-cost responses) and a neutral control were run, first to 100 steps and then continued. The cost-shaped run stopped at step 400; the neutral run reached step 500.

At step 350, neutral mixed training versus specialized fixed training on MATH is:

| Inference `K` | Mixed model | Specialized fixed-`K` model |
| ---: | ---: | ---: |
| 4 | 81.60 | 81.33 |
| 6 | 82.80 | 80.20 |
| 8 | 84.00 | 84.07 |

This is evidence that one checkpoint can be elastic across widths. It is not a compute-matched proof: mixed training generated 144 responses per step versus 128 in the fixed runs. AIME is also noisy enough that the MATH result should carry more weight.

The SGLang rollout engines execute their assigned true `K`. The learner does not receive equivalent savings: mixed routing traces are padded to the maximum width and masked, so the learner performs the maximum-width dispatch.

### Mixed `K={4,8,12}` DAPO

The neutral even-split run uses learner dispatch width `K=12`, with `K=4/8` routes padded and masked. The latest persisted evaluated checkpoint is step 450:

| Inference `K` | MATH | AIME pass@1 |
| ---: | ---: | ---: |
| 4 | 82.07 | 13.02 |
| 8 | 85.67 | 16.67 |
| 12 | 84.87 | 17.08 |

The run developed a severe `K=4` repetition pathology after approximately step 342. Mean response length rose from about 680 tokens at step 325 to 5,608 at step 350 and 12,065 at step 370; rollout time reached roughly 31 minutes. Repeating the correct answer could still earn reward, so reward remained misleadingly healthy.

This is a central adaptive-training failure mode. Future training needs repetition/length defenses, stricter answer extraction, explicit token cost, and monitoring broken out by routing bucket.

### Prompt-level `K` sweep and selector evidence

The DAPO prompt sweep contains 17,391 clean prompts, eight paired samples per prompt at each of `K=4,6,8`, and 139,128 generations per `K`. Mean sample accuracies are 4.995, 5.697, and 5.698 correct out of eight. Any-success rates are 26.31%, 29.35%, and 29.11%.

Several static assignment rules were tested:

- A conservative mapping has mean `K=6.898` and held-out score 5.610.
- A cheapest-any-success mapping has mean `K=6.757` and held-out score 5.522.
- Always using `K=8` scores 5.698.
- A cheapest assignment labels 4,575 prompts as `K=4`, 2,469 as `K=6`, and 10,347 as `K=8`, for mean `K=6.664`.

These results do not yet show a selector beating a fixed policy at matched quality. The label construction is vulnerable to sampling noise and non-monotonic outcomes.

A smaller exact transition analysis on 500 MATH prompts from step-350 checkpoints found 73.6--78.0% correct at all three widths, 10.6--12.6% wrong at all widths, only 4.2--6.8% with a clean monotonic need for higher `K`, and 5.4--7.0% non-monotonic. Pairwise correctness correlation between `K=4` and `K=8` is 0.735--0.797, while total disagreement is 27.8--38.6% depending on sample view.

The implication is that "minimum K for this prompt" is not a stable hard label under current sampling. A useful oracle should estimate conditional success probability or marginal value of compute, use repeated paired samples, and be evaluated on held-out prompts.

### Prompt-preallocated DAPO

A run using precomputed prompt assignments reached step 500. Its step-500 MATH results are:

| Inference routing | Score |
| --- | ---: |
| Native `K=8` | 83.40 |
| Reference-scaled `K=4` | 80.93 |
| Reference-scaled `K=6` | 83.47 |

This demonstrates that assignment-aware training can complete and preserve an elastic checkpoint. It does not yet learn the selector, and the assignment quality and compute matching remain limiting factors.

### OpenThoughts Qwen3.5 SFT

The SFT pilot includes fixed train `K=4`, `K=8`, and `K=12`; mixed `K={4,8,12}`; and a phased `K=12 -> 8 -> 4` cooldown. Checkpoints were evaluated on `openthoughts-tblite@2.0` with Terminus2, reference-scaled inference, and three repetitions.

| Training policy | Eval K=4 | K=6 | K=8 | K=10 | K=12 | Other points |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fixed K=4 | 0.3400 | 0.3467 | 0.4100 | 0.4133 | 0.4400 | — |
| Fixed K=8 | 0.3333 | 0.4267 | 0.4100 | 0.4367 | 0.3867 | — |
| Fixed K=12 | 0.3400 | 0.4233 | 0.4300 | 0.4333 | 0.4167 | — |
| Mixed K=4/8/12 | 0.3567 | 0.3933 | 0.3967 | 0.4367 | 0.4367 | K=14 0.4167; K=16 0.4000; K=32 0.3733 |
| Phased K=12->8->4 | 0.3800 | 0.4233 | 0.3867 | 0.4200 | 0.4100 | — |

The phased model has the best observed `K=4` result in this table, but the pilot does not establish a simple train-high/deploy-low law. Several models improve when evaluated at `K=6--12`, and differences are noisy. There is no matched general-capability evaluation for these SFT checkpoints, so the evidence is currently limited to TBLite.

## Serving and systems implementation status

### vLLM evaluation path

`olmo-eval` contains local compatibility policies for:

- Qwen weight modes including normal, shuffle, uniform, temperature, rank-profile, adaptive-mass, and truncation behavior;
- GPT-OSS non-power-of-two fallback and reference scaling;
- routing telemetry and runtime assertions;
- fixed and adaptive evaluation variants and launchers.

Some policies operate through native-width slot masking. They are appropriate for capability and mechanistic counterfactuals but do not establish expert-kernel speedup.

### SGLang and Slime

`slime` contains:

- mixed-`K` rollout scheduling;
- named SGLang routers;
- reference-scaled routing in SGLang and Megatron;
- route-replay width padding and masking;
- synchronization of every active rollout engine;
- environment-based `K` overrides;
- one-node, eight-B200 topologies used by current mixed-`K` runs.

The rollout path can perform true `K`-specific routing. Mixed-`K` learner batches currently dispatch at the largest width, so only rollout-side expert work can be reduced. This is sufficient for algorithm experiments but incomplete for full training-efficiency claims.

### Terminal-agent evaluation

`tmax-eval` contains the Qwen routing patch, Qwen3.5 text-architecture compatibility, routing launch options, result parsers, context controls, task filtering, cluster/allocation options, and timeout handling. Exact environment and agent configuration must accompany every hard-benchmark result.

### What has not been measured

There is no controlled matrix reporting:

- tokens/second and requests/second versus `K`;
- time to first token and inter-token latency;
- median and tail request latency;
- expert communication volume and load balance;
- GPU utilization and energy;
- memory footprint or feasible batch-size change;
- prompt and generation token counts jointly with kernel time;
- vLLM versus SGLang at matched model, quantization, batch, and workload;
- concurrency scaling;
- engine behavior for variable `K` within a batch.

These measurements are required before claiming deployable inference savings.

## Quantization status

Quantization has not been factorially crossed with `K`. Nemotron is evaluated from an FP8 checkpoint while its mechanistic analysis is BF16 because the Hugging Face/Transformers path does not support its ModelOpt FP8 representation. This is a confound, not an interaction study.

Slime has infrastructure relevant to FP8 online rollout, but no current result isolates `K x quantization` effects on capability, routing identity, speed, or stability. The proposed report should include at least native precision, weight-only quantization, and an FP8-style serving configuration where supported, with routing telemetry to detect top-`K` changes from numerical ties.

## Validation and reproducibility status

The previous local validation pass found:

- Ruff passing on the custom `olmo-eval` files checked;
- Python compilation passing for the modified Slime and tmax-eval code checked;
- `bash -n` passing on the relevant shell launchers;
- full `olmo-eval` pytest collection did not complete within 118 seconds before executing tests;
- the Slime virtual environment did not have pytest installed.

This is lightweight validation, not a clean test-suite pass. Because much of the code is untracked and the repositories are dirty, source snapshots or commits attached to historical jobs are important provenance.

### Experimental conventions that must remain explicit

- Most stochastic evaluations use three repetitions; terminal full-suite summaries can use five.
- AIME `pass@k` is derived from multiple samples within a run and is not the same as run-to-run replication.
- Invalid infrastructure/parser runs are excluded only when documented; terminal task exceptions and timeouts generally remain in the denominator.
- Shared experts are not included in reported `K`.
- Native and intervention routing must be distinguished.
- Response length must be reported alongside expert count for cost claims.
- Above-native Qwen3.6 values must come from corrected nested configuration plus observed-width telemetry.

## Main conclusions supported now

1. **Many MoE models have meaningful sub-native inference headroom.** GPT-OSS, Qwen3, GLM, and Nemotron all retain substantial quality with fewer active experts.
2. **The headroom is neither universal nor task-independent.** Qwen3.5 HumanEval is a strong low-`K` failure, Nemotron's loss is concentrated in GPQA, and Qwen3.6 tasks prefer different widths.
3. **Router scaling materially changes the result.** Native-denominator preservation often rescues low-`K` quality and response length, but post-trained models can prefer standard normalization.
4. **More experts are not monotonically better under inference-time intervention.** Qwen Base, Qwen3.6 math, TBLite at extreme `K`, and high-width response behavior all show reversals.
5. **A single mixed-`K` trained model can function at several widths.** The mixed DAPO checkpoint is competitive with specialized checkpoints on MATH, although training compute is not matched.
6. **Simple adaptive heuristics do not yet beat strong fixed-`K` baselines.** Cumulative-mass routing mostly traces the fixed frontier, and prompt minimum-`K` labels are noisy.
7. **Real systems savings remain unproven.** Correct quality interventions exist, and true low-`K` SGLang rollout exists, but there is no production serving benchmark and several paths retain maximum-width dispatch.

## Claims that are not yet supported

The current results do not support saying that:

- reducing `K` always preserves quality;
- a particular percentage reduction is universal across model families;
- lower `K` currently produces a measured production latency or throughput win;
- lower `K` reduces model memory footprint;
- reference scaling is always the right sub-native policy;
- adaptive threshold routing beats the fixed-`K` Pareto frontier;
- training at high `K` reliably improves deployment at low `K`;
- mixed-`K` training is compute-equivalent to specialized training;
- quantization and expert reduction compose without interaction;
- current findings extend to trillion-parameter models.

## High-priority unresolved questions

- Why is GPT-OSS robust at half width while Qwen3.5 code generation is not?
- Is family-level robustness predicted by router entropy, contribution concentration, shared-expert strength, layer sensitivity, or post-training history?
- Does reference scaling help because it preserves native branch norm, and when does post-training make renormalization preferable?
- Are Qwen3.6 task reversals caused by useful added experts or by the larger reference-scaled MoE branch?
- Can lower `K` reduce total request cost after accounting for longer generations?
- Do quantized routers select meaningfully different experts because of ties or reduced score precision?
- Can training at high `K` transfer knowledge into a better low-`K` subnet, or does it teach dependencies that low `K` cannot realize?
- Is a prompt-level selector sufficient, or is most useful adaptivity token/layer-local?
- What objective makes quality monotonic in a requested budget while avoiding repetition and length reward hacking?

## Continuation checklist

Before launching a new wave of work:

1. Preserve the current dirty repositories in explicit branches/commits or durable patches without deleting untracked artifacts.
2. Refresh live Beaker state and collect any results newer than the 2026-08-21 local artifacts.
3. Regenerate the authoritative aggregate CSVs and plots from the refreshed bundles.
4. Decide and document one routing-semantics vocabulary for all new runs.
5. Add runtime telemetry assertions for actual selected width, selected-weight sum, dispatch width, and generation length.
6. Establish a small correctness and throughput gate in both vLLM and SGLang before expanding the benchmark matrix.
7. Re-run focused tests in reproducible environments; the current audit did not obtain a complete pytest pass.
8. Use fixed prompt/sample identities where possible and report both per-token kernel cost and end-to-end request cost.

## Key files for the next session

### Broad summaries and provenance

- [`project_overview.md`](project_overview.md)
- [`latest_status.md`](latest_status.md)
- [`beaker_jobs.jsonl`](beaker_jobs.jsonl)
- [`current_suite_expert_sweeps.csv`](current_suite_expert_sweeps.csv)

### Routing and fixed-`K` evidence

- [`qwen_reference_adaptive_threshold_sweep.md`](qwen_reference_adaptive_threshold_sweep.md)
- [`qwen_math_response_coherence.md`](qwen_math_response_coherence.md)
- [`qwen_gpqa_ifbench_response_analysis.md`](qwen_gpqa_ifbench_response_analysis.md)
- [`qwen35_35b_a3b_hybrid_sweep.md`](qwen35_35b_a3b_hybrid_sweep.md)
- [`corrected_dolci_terminal_eos_v2_sweep.md`](corrected_dolci_terminal_eos_v2_sweep.md)
- [`nemotron3_super_routing_sweep.md`](nemotron3_super_routing_sweep.md)
- [`nemotron_router_contribution_analysis.md`](nemotron_router_contribution_analysis.md)
- [`qwen36_expanded_k_diagnostic.md`](qwen36_expanded_k_diagnostic.md)
- [`above_native_k_routing_audit.md`](above_native_k_routing_audit.md)

### Hard benchmark evidence

- [`qwen36_terminalbench_full_suite_summary.csv`](qwen36_terminalbench_full_suite_summary.csv)
- [`qwen36_kgt8_replacement_jobs.md`](qwen36_kgt8_replacement_jobs.md)
- [`qwen36_tblite_extreme_reference_sweep_jobs.md`](qwen36_tblite_extreme_reference_sweep_jobs.md)

### Training and adaptive-`K` evidence

- [`slime_qwen3_base_dapo_plan.md`](slime_qwen3_base_dapo_plan.md)
- [`mixed_k_cost_aware_rl.md`](mixed_k_cost_aware_rl.md)
- [`slime_qwen3_base_dapo_prompt_k_transitions.md`](slime_qwen3_base_dapo_prompt_k_transitions.md)
- [`qwen35_adaptive_k_sft_plan.md`](qwen35_adaptive_k_sft_plan.md)
- [`openthoughts_all_sft_tblite_results.csv`](openthoughts_all_sft_tblite_results.csv)

## Bottom line

The project already has the ingredients for a strong report: broad evidence that native MoE routing is often overprovisioned for inference, clear exceptions that keep the story scientifically honest, a mechanism showing why routing scale matters, hard agentic benchmarks, and early evidence that one trained policy can operate at several widths. The missing work is concentrated rather than vague: standardize the cross-family protocol, prove real serving savings, add scale and quantization, run a controlled train-hard/eval-light study, and convert heuristic or preallocated adaptivity into a learned budget-conditioned policy.
