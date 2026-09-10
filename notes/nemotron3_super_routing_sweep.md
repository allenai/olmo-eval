# Nemotron 3 Super routing sweep

## Objective

Compare native normalized routing with raw, unnormalized routing for
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` at:

- K=22 (the checkpoint default)
- K=11 (half the checkpoint default)

After the smoke gate, run three complete post-training evaluations per condition using
`adaptive_experts:hybrid_pilot` (MATH-500, GPQA Diamond, and IFEval OOD).

## Routing semantics

Nemotron applies sigmoid router scores, selects the top K routed experts, optionally divides the
selected weights by their sum, and finally multiplies the routed-expert output by the checkpoint's
fixed `routed_scaling_factor=5`. The shared expert remains active in every condition.

| Condition | `num_experts_per_tok` | `norm_topk_prob` | Effective routed weights |
|:--|--:|:--:|:--|
| K=22 normalized | 22 | `true` | selected sigmoid scores / selected sum, then ×5 |
| K=11 normalized | 11 | `true` | selected sigmoid scores / selected sum, then ×5 |
| K=22 unnormalized | 22 | `false` | raw selected sigmoid scores, then ×5 |
| K=11 unnormalized | 11 | `false` | raw selected sigmoid scores, then ×5 |

The unnormalized conditions intentionally do **not** preserve the normalized K=22 reference
scale. Their routed mass is `5 * sum(top-K sigmoid scores)`, which can exceed 5.

## Placement

- Workspace: `ai2/OLMo-3-moe-experiments`
- Cluster: `ai2/jupiter`
- Priority: `urgent`
- Beaker group:
  [adaptive-experts-nemotron3-super-routing-moe-20260717](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/groups/01KXRPYVJKDJPBKPK45V21RJD7)
- Source commit: `e8b88f196767630acaf8a383d12a33d314788d27`
- Source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/ca1cd217b47ac53ab70d6fa2bb176522b1051f7a95de73b02e392050d749867e`
- Append-only job ledger: `notes/beaker_jobs.jsonl`

The initial serving target is one TP=2, expert-parallel vLLM engine for smoke tests and four TP=2
engines per full evaluation.

## Smoke gate

| Attempt | Beaker experiment | Outcome |
|:--|:--|:--|
| K=22 normalized, original | [01KXRPZENJT7VM3X3DSBJWZCC7](https://beaker.org/ex/01KXRPZENJT7VM3X3DSBJWZCC7) | Invalid: FlashInfer JIT could not find `ninja` |
| K=22 normalized, `ninja` with incorrect PATH precedence | [01KXRQCQ7W3PAYWPKN9SVWGYEA](https://beaker.org/ex/01KXRQCQ7W3PAYWPKN9SVWGYEA) | Invalid: evaluator launched from the isolated vLLM environment |
| K=22 normalized, `ninja` exposed correctly | [01KXRQF90FR9ZR7VR8EEC51EJQ](https://beaker.org/ex/01KXRQF90FR9ZR7VR8EEC51EJQ) | Invalid: image has CUDA runtime but no CUDA compiler needed by FlashInfer JIT |
| K=22 normalized, force non-FlashInfer FP8 MoE backend | [01KXRQPAHZDTF5W5XTA1MS2WZ5](https://beaker.org/ex/01KXRQPAHZDTF5W5XTA1MS2WZ5) | Passed; TP=2 server ready in 106s and generated 256 tokens |
| K=11 normalized | [01KXRQY6Z7QZ8BDGFJYM6Y21T9](https://beaker.org/ex/01KXRQY6Z7QZ8BDGFJYM6Y21T9) | Passed; TP=2 server ready in 106s and generated 256 tokens |
| K=22 raw ×5 | [01KXRQY7PYV5JXNSXXPXC3KB1B](https://beaker.org/ex/01KXRQY7PYV5JXNSXXPXC3KB1B) | Passed; TP=2 server ready in about 174s, generated 256 tokens, and scored cleanly |
| K=11 raw ×5 | [01KXRQY7A485AXG8WGXQVZS68F](https://beaker.org/ex/01KXRQY7A485AXG8WGXQVZS68F) | Passed; TP=2 server ready in about 145s, generated 256 tokens, and scored cleanly |

No olmo-eval core changes are used for these conditions. K and normalization are checkpoint config
overrides; the fallback away from FlashInfer is a supported vLLM runtime environment setting.
The reusable launcher is `scripts/adaptive_experts/launch_nemotron3_super_routing_sweep.sh`.

The 256-token smoke cap can end while the reasoning parser is still inside hidden reasoning, so the
smoke gate checks server startup, finite generation, and successful scoring rather than task
accuracy. Full evaluations use the normal 32,768-token generation cap.

## Full evaluations

Each run evaluates all three tasks together with four TP=2 expert-parallel engines (8 H100s total).
There are three independent evaluation runs per condition.

| Condition | Replicate 1 | Replicate 2 | Replicate 3 |
|:--|:--|:--|:--|
| K=22 normalized | [01KXRRBYEBQZRNH267A9F77RX1](https://beaker.org/ex/01KXRRBYEBQZRNH267A9F77RX1) | [01KXRRCXGTE0H8JMW001Q13KE0](https://beaker.org/ex/01KXRRCXGTE0H8JMW001Q13KE0) | [01KXRRDXWFARQX0QJ7EMBS3J2T](https://beaker.org/ex/01KXRRDXWFARQX0QJ7EMBS3J2T) |
| K=11 normalized | [01KXRRC639BXGACTNNSYV6PEFF](https://beaker.org/ex/01KXRRC639BXGACTNNSYV6PEFF) | [01KXRRD5DR2QXVG3YZJYMX04Q1](https://beaker.org/ex/01KXRRD5DR2QXVG3YZJYMX04Q1) | [01KXRRE5RQT3X8P26S7YXVXMD8](https://beaker.org/ex/01KXRRE5RQT3X8P26S7YXVXMD8) |
| K=22 raw ×5 | [01KXRRCE1VME12FGTVPFPJHK4K](https://beaker.org/ex/01KXRRCE1VME12FGTVPFPJHK4K) | [01KXRRDDXR8A178KCT4P4A7CVN](https://beaker.org/ex/01KXRRDDXR8A178KCT4P4A7CVN) | [01KXRREDVJTHM76P85P73QGSTA](https://beaker.org/ex/01KXRREDVJTHM76P85P73QGSTA) |
| K=11 raw ×5 | [01KXRRCNVGVFRFEKFXXEKHF2ZP](https://beaker.org/ex/01KXRRCNVGVFRFEKFXXEKHF2ZP) | [01KXRRDNTMMDZCBEST4NT550YH](https://beaker.org/ex/01KXRRDNTMMDZCBEST4NT550YH) | [01KXRRENYWH03A5X57ZVPHFB80](https://beaker.org/ex/01KXRRENYWH03A5X57ZVPHFB80) |

Initial status audit at 2026-07-17 19:23 UTC: 8 running, 4 scheduled, 0 failed.

## Raw ×5 finding

The six raw ×5 evaluations completed with near-zero scores. Inspection confirmed this is not an
answer-extraction error: most generations ended without visible final-answer content, and the
small fraction with visible content was heavily corrupted. In contrast, normalized K=11 and K=22
runs using the same `nemotron_v3` reasoning parser and task scorers produced normal outputs and
strong scores.

Raw ×5 is therefore retained as a valid negative result, but no additional raw ×5 expert counts
will be launched for now.

## Intermediate normalized sweep

To resolve the normalized curve between K=11 and the native K=22 endpoint, three complete
evaluations were launched at each of K={13, 15, 17, 19, 21}. All settings match the original
normalized runs: `norm_topk_prob=true`, four TP=2 expert-parallel engines, the complete
`adaptive_experts:hybrid_pilot` suite, Jupiter, and urgent priority.

| K | Replicate 1 | Replicate 2 | Replicate 3 |
|--:|:--|:--|:--|
| 13 | [01KXRWPY7MT098PJWFTXF3HXYD](https://beaker.org/ex/01KXRWPY7MT098PJWFTXF3HXYD) | [01KXRWRB7AVEPKXX39E6RZDRCH](https://beaker.org/ex/01KXRWRB7AVEPKXX39E6RZDRCH) | [01KXRWSJZJD77ZDGJ5EFF195YJ](https://beaker.org/ex/01KXRWSJZJD77ZDGJ5EFF195YJ) |
| 15 | [01KXRWQ6HQD5692HQB88DZ3KRV](https://beaker.org/ex/01KXRWQ6HQD5692HQB88DZ3KRV) | [01KXRWRK57EEWHMKCDPXMZZPH9](https://beaker.org/ex/01KXRWRK57EEWHMKCDPXMZZPH9) | [01KXRWSVA89KYBSQFHX6DX9CRD](https://beaker.org/ex/01KXRWSVA89KYBSQFHX6DX9CRD) |
| 17 | [01KXRWQEKJZ86AQAE0XF7E42PY](https://beaker.org/ex/01KXRWQEKJZ86AQAE0XF7E42PY) | [01KXRWRTTXQT4WS3B9F26RWJ3Z](https://beaker.org/ex/01KXRWRTTXQT4WS3B9F26RWJ3Z) | [01KXRWT4100K9N02C5KD0ZWH3X](https://beaker.org/ex/01KXRWT4100K9N02C5KD0ZWH3X) |
| 19 | [01KXRWQPHFX7SVFMJ2FFZSXJCB](https://beaker.org/ex/01KXRWQPHFX7SVFMJ2FFZSXJCB) | [01KXRWS2SZE075XVDP3QPT8H6N](https://beaker.org/ex/01KXRWS2SZE075XVDP3QPT8H6N) | [01KXRWTC48F70DQQR59AC6RS8H](https://beaker.org/ex/01KXRWTC48F70DQQR59AC6RS8H) |
| 21 | [01KXRWQYM3JP4MB54KAHS53GDM](https://beaker.org/ex/01KXRWQYM3JP4MB54KAHS53GDM) | [01KXRWSAY6QFDRCB35M78XQ1N9](https://beaker.org/ex/01KXRWSAY6QFDRCB35M78XQ1N9) | [01KXRWTMC4Q1GC72WV310D427Y](https://beaker.org/ex/01KXRWTMC4Q1GC72WV310D427Y) |

All 15 intermediate normalized evaluations completed successfully. Together with the three
completed K=11 and K=22 evaluations, the normalized sweep now has three complete replicates at
every K.

Values are mean ± sample standard deviation across three complete evaluations. IFEval OOD is
prompt-level loose accuracy, and the macro is the unweighted mean of the three displayed task
scores.

| K | MATH-500 | GPQA Diamond | IFEval OOD | Macro average |
|--:|--:|--:|--:|--:|
| 11 | 94.80 ± 0.35 | 59.43 ± 1.27 | 69.78 ± 0.38 | 74.67 ± 0.54 |
| 13 | 94.60 ± 0.92 | 63.13 ± 1.52 | 71.56 ± 1.07 | 76.43 ± 0.36 |
| 15 | 95.40 ± 0.92 | 65.82 ± 1.27 | 70.67 ± 3.53 | 77.30 ± 1.25 |
| 17 | 94.93 ± 0.23 | 69.87 ± 2.04 | 71.11 ± 0.84 | 78.64 ± 0.71 |
| 19 | 95.20 ± 0.69 | 69.19 ± 0.87 | 71.22 ± 1.35 | 78.54 ± 0.42 |
| 21 | 95.53 ± 0.42 | 69.70 ± 2.20 | 71.78 ± 1.07 | 79.00 ± 0.79 |
| 22 | 95.27 ± 0.50 | 71.55 ± 1.27 | 71.33 ± 0.67 | 79.38 ± 0.31 |

[Normalized-sweep plot](plots/nemotron3_super_posttraining_expert_sweep.png)

The curve is smooth at the aggregate level: macro performance rises from 74.67 at K=11 to 79.38
at K=22. Nearly all of that separation is GPQA (a 12.12-point gain). MATH-500 is essentially flat,
and IFEval OOD is already within 1.55 points of K=22 at K=11. This is another example of capability
sensitivity differing substantially by evaluation rather than a uniform degradation from reduced
expert count.
