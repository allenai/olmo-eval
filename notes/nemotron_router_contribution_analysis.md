# Nemotron-3-Super router and contribution analysis

## Objective

Measure why normalized Nemotron-3-Super loses more performance when reducing routed experts than
Qwen3 and GPT-OSS. The analysis uses the checkpoint-default K=22 routing and distinguishes the
routed LatentMoE mixture from Nemotron's always-on shared expert.

The direct comparison uses the same 60-prompt manifest as the Qwen analysis: 20 prompts each from
MATH-500, GPQA Diamond, and the historical IFEval OOD task. Those prompts were stratified using
Qwen's baseline correctness, so the stratification is not interpreted as Nemotron correctness;
reusing the exact prompts is for paired architectural comparison.

## Measurements

For each token and MoE layer:

- Compute all 512 sigmoid router scores.
- Sort the selected 22 experts by their actual mixture weight.
- Record cumulative score mass at K={1,2,4,8,11,13,15,17,19,21,22}:
  - relative to the selected 22;
  - relative to the sum of all 512 sigmoid scores.
- Record the overlap between the selected set (which includes the learned routing correction bias)
  and the raw top-22 sigmoid scores.
- Record each selected expert's unweighted output norm after the latent output projection, weighted
  contribution norm, contribution share, and projection onto the routed mixture.
- Record routed, shared, and combined MoE-update norms separately.
- Recombine the same contributions at each smaller K in two ways:
  - reference preserving: retain the original K=22 weights and drop the tail;
  - renormalized: rescale retained weights back to the K=22 routed weight sum of five.
- Compare each counterfactual both to the native routed mixture and to the complete routed+shared
  MoE update.

The script captures activations during the same native-K autoregressive generation pass and saves
the exact token IDs. Prompt, reasoning, and final-answer statistics are separated online, while
metric tensors are buffered for 64 tokens on GPU before transfer. This avoids a redundant replay
and works around NVIDIA's fallback Mamba cache accepting only one continuation token per forward.
Existing olmo-eval Nemotron predictions cannot substitute for this step because the reasoning
parser does not preserve the hidden reasoning-token prefix.

## Implementation and validation

Implementation:
`scripts/adaptive_experts/nemotron_router_contribution_profile.py`.

The Hugging Face reference model's training-oriented MoE loop evaluates unused experts on zero
inputs. The profiler replaces that loop during inference with a selected-expert-only loop. Since
the expert projections have no bias, the omitted dummy outputs are identically zero. The smoke
gate nevertheless requires:

- native selected-weight sums equal five;
- individual projected contributions reconstruct the routed output;
- routed plus shared outputs reconstruct the complete MoE output;
- native K=22 counterfactuals reconstruct both routed and complete outputs; and
- prompt prefill and autoregressive generation work when the checkpoint is split across four
  H100s.

The deployed FP8 checkpoint uses NVIDIA's unified ModelOpt quantization format, which Transformers
5.7 does not execute faithfully: it ignores the ModelOpt scales and attempts to load the FP8
weights as an unquantized model. The mechanistic run therefore uses the corresponding
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` checkpoint. It has the same K=22, 512-expert,
shared-expert LatentMoE architecture and routing configuration; the difference is deployment
quantization. This choice is recorded explicitly when comparing these profiles to FP8 eval curves.

The passing smoke directly captured the routed latent projection rather than inferring it as
`(routed + shared) - shared`, which is numerically unstable in BF16. It profiled 381 prompt tokens
and 16 generation tokens across all 40 MoE layers (680 calls), with:

- maximum selected-weight-sum error: `9.54e-7`;
- maximum routed reconstruction relative L2: `0.00373`; and
- maximum complete routed-plus-shared reconstruction relative L2: `0.00419`.

## Placement

- Workspace: `ai2/OLMo-3-moe-experiments`
- Priority: urgent
- Initial cluster: `ai2/jupiter`
- Smoke resources: four H100s, one prompt, 16 generated tokens
- Full resources: four four-H100 shards (16 H100s total), with 15 prompts per shard
- Every shard contains five GPQA Diamond, five historical IFEval OOD, and five MATH-500 prompts.
- Every launch is appended to `notes/beaker_jobs.jsonl`.

## Launch status

- Passing four-H100 BF16 smoke:
  [01KXSHJ5P3MDA66NH78ETX8QYP](https://beaker.org/ex/01KXSHJ5P3MDA66NH78ETX8QYP)
- Four-shard, 60-prompt production run:
  [01KXSHV17C9FDKBWTWHKG694RG](https://beaker.org/ex/01KXSHV17C9FDKBWTWHKG694RG)
- Production was submitted at urgent priority on `ai2/jupiter` at 2026-07-18 02:45 UTC.
- Original shard 2 completed its first prompt, then OOMed while prefilling its second prompt,
  an 830-token GPQA outlier (all prompts on the other shards are at most 381 tokens). The
  unchanged shard was relaunched on five H100s at urgent priority:
  [01KXSJTPF0PYN3B6MMNW56FTCF](https://beaker.org/ex/01KXSJTPF0PYN3B6MMNW56FTCF).
  The rerun reproduced prompt 1 exactly and successfully cleared the 830-token prompt-2 prefill
  that failed in the four-H100 layout.
- At the 2026-07-19 02:30 UTC check, original shards 0, 1, and 3 and the five-H100 shard-2 repair
  had all succeeded. The full four-shard, 60-prompt production dataset is downloaded and
  validated. Each bundle contains the expected metadata, 15 compressed per-example records,
  JSON/NPZ summaries, 40 MoE layers, native K=22, routed scale five, and the shared-expert path.
  Maximum routed and total reconstruction relative-L2 errors across the complete collection are
  0.00884 and 0.00750. The original failed shard-2 artifact remains excluded and is replaced by
  the successful repair.
- Collected production artifacts are stored under
  `results/adaptive_experts/01KXSHV17C9FDKBWTWHKG694RG/shard-{0,1,3}/` and
  `results/adaptive_experts/01KXSJTPF0PYN3B6MMNW56FTCF/shard-2/`.
- The smoke artifact bundle is stored locally under
  `results/adaptive_experts/01KXSHJ5P3MDA66NH78ETX8QYP/`.

## Aggregate findings

The one-H100 aggregate job
[01KY61MX7B4T9WCZ4W41MT2WRA](https://beaker.org/ex/01KY61MX7B4T9WCZ4W41MT2WRA)
succeeded on 2026-07-22. It merged all 60 prompts and 23,845,360 token-layer observations, emitted
19 tables/plots/data artifacts, and passed the shard, example, schema, and reconstruction checks.
The downloaded aggregate is stored under
`results/adaptive_experts/01KY61MX7B4T9WCZ4W41MT2WRA/`; the stable analysis copy is under
`notes/nemotron_router_contribution/`.

The selected K=22 distribution is substantially flatter than Qwen's selected K=8 distribution:
its mean selected-distribution entropy is 3.041 and its effective selected expert count is 20.97.
However, activation magnitude makes the head more important than router mass alone suggests:

| Retained K | Router mass within top 22 | Contribution-norm share | Reference total cosine | Renormalized total cosine |
|---:|---:|---:|---:|---:|
| 8 | 47.23% | 60.66% | 0.9638 | 0.9380 |
| 11 | 60.08% | 71.25% | 0.9774 | 0.9652 |
| 17 | 83.04% | 88.25% | 0.9925 | 0.9909 |
| 21 | 96.80% | 97.72% | 0.9986 | 0.9985 |

At half the native K, Nemotron's top 11 contain about the same router share as Qwen's top four
(60.08% versus 63.31%) and slightly more contribution-norm share (71.25% versus 69.23%). Yet the
normalized Nemotron evaluation curve is less robust at half K. Router concentration alone
therefore does not explain the cross-model robustness difference.

The local recombinations consistently favor preserving the native K=22 scale over renormalizing
the retained prefix. At K=11, reference-preserving routed+shared relative L2 is 0.182 versus 0.379
after renormalization; at K=17 it is 0.099 versus 0.149. The earlier raw-x5 evaluation is not this
condition: it multiplied an unnormalized sigmoid prefix by five and produced an excessive routed
scale. A future causal evaluation should instead truncate the *native normalized K=22 weights*
without renormalizing, exactly as this counterfactual does.

Layer variation is material. At K=11, router mass ranges from 54.36% to 72.72%, contribution share
from 58.83% to 81.29%, and reference-preserving total cosine from 0.8996 to 0.9954. At K=17, the
corresponding ranges are 79.66%--89.49%, 79.17%--93.22%, and 0.9516--0.9985. Early layer 1 is the
largest local outlier, while the final MoE layer is among the most concentrated. Prompt tokens are
also more concentrated than reasoning tokens: K=11 contains 64.28% versus 60.01% router mass.

The learned routing correction changes about 14.6% of the raw sigmoid top-22 selections on
average (`raw_top22_selection_overlap=0.854`). The saved trajectories did not expose a separately
parsed final-answer phase, so the phase table contains prompt and reasoning observations only; the
overall and layer results remain complete.

Plots:
[overall counterfactuals](nemotron_router_contribution/nemotron_router_contribution_overall.png),
[router rank](nemotron_router_contribution/nemotron_router_contribution_by_rank.png), and
[layer profiles](nemotron_router_contribution/nemotron_router_contribution_by_layer.png).

## GPT-OSS matched follow-up

Run the same prompt set and output schema for GPT-OSS-120B after the Nemotron adapter is validated.
GPT-OSS has native K=4 and fits on one H100, so its matched profile is materially cheaper. The three
models will then span Qwen softmax K=8, GPT-OSS K=4, and Nemotron sigmoid LatentMoE K=22 with an
always-on shared expert. GPT-OSS is planned but is not part of the present Nemotron launch.
