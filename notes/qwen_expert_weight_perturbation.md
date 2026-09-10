# Qwen top-8 expert-weight perturbation experiment

## Question

How much does Qwen3-30B-A3B depend on the router assigning the *right relative
weights* to its selected top-eight experts, as opposed to merely selecting the
right set of experts?

## Fixed setup

- Model: Qwen3-30B-A3B hybrid-thinking checkpoint (`qwen3-30b-a3b`)
- Active experts: K=8
- Evaluation suite: `adaptive_experts:hybrid_pilot`
  - MATH-500
  - GPQA Diamond
  - IFEval OOD
- Maximum generation length: 32,768 tokens
- Serving: four independent one-GPU vLLM engines per full run (TP=1)
- Repetitions: three full runs per condition
- Beaker workspace / priority: `ai2/olmo-instruct`, urgent
- Beaker cluster: `ai2/jupiter`

## Conditions

1. **Normal**: untouched native top-eight expert IDs and normalized weights.
2. **Shuffled**: keep the native selected expert IDs fixed, but deterministically
   permute their eight normalized weights independently by token and layer. The
   multiset and sum of the weights are unchanged. Full-run seeds are 101, 202,
   and 303.
3. **Uniform**: keep the native selected expert IDs fixed and replace their
   normalized weights with 1/8 each.

The normal condition deliberately does not install the patch or alter the MoE
backend, so it remains an exact baseline. Modified conditions use vLLM's modular
Triton MoE backend, transform the outputs of `FusedTopKRouter`, and then pass the
unchanged IDs plus modified weights to the fused expert kernel.

## Validation and launch protocol

Before the full sweep, run one one-example MATH-500 smoke evaluation for each
condition. Promote to the nine full evaluations only after all three smoke jobs
complete and the modified-mode logs confirm that the Qwen routing patch was
enabled. Every launch is recorded in `notes/beaker_jobs.jsonl`.

## Interpretation

- Normal versus shuffled tests whether matching larger router probabilities to
  their chosen experts matters while preserving the exact weight distribution.
- Normal versus uniform tests whether the router's within-top-eight confidence
  profile matters at all once the selected expert set is fixed.
- Shuffled versus uniform distinguishes harmful expert/weight mismatches from
  simply discarding within-set confidence information.

## Beaker runs

All runs below use source snapshot
`33719fd5c566deec8246d3d1827cfab55d31450527dc8297571acd612e336edb`.
The complete launch commands and metadata are also in `notes/beaker_jobs.jsonl`.

### Validated smokes

| Condition | Experiment | Outcome |
|---|---|---|
| Normal | [01KXNTDKD4D74TZTEM0Y8PEG0D](https://beaker.org/ex/01KXNTDKD4D74TZTEM0Y8PEG0D) | Exit 0 |
| Shuffled, seed 101 | [01KXNTJPZVGVGV84JB6RBNEAXN](https://beaker.org/ex/01KXNTJPZVGVGV84JB6RBNEAXN) | Patch enabled; exit 0 |
| Uniform | [01KXNTK54SA5VAKAH2FMM21TVK](https://beaker.org/ex/01KXNTK54SA5VAKAH2FMM21TVK) | Patch enabled; exit 0 |

Two earlier modified smokes were stopped before inference after an audit found
that their isolated vLLM environment had not linked the startup hook:
[shuffle](https://beaker.org/ex/01KXNTE4FQJQ2KFS6TE03GY6AR) and
[uniform](https://beaker.org/ex/01KXNTEP66JR541J05TV0VW8PT). They are invalid
and must not be used as results.

### Full evaluations

| Condition | Replicate / seed | Experiment |
|---|---:|---|
| Normal | 1 | [01KXNTSP1A5H4WNASZPVZBVQPG](https://beaker.org/ex/01KXNTSP1A5H4WNASZPVZBVQPG) |
| Normal | 2 | [01KXNTSZ5GTFV9D9W8S1R4FG1N](https://beaker.org/ex/01KXNTSZ5GTFV9D9W8S1R4FG1N) |
| Normal | 3 | [01KXNTT83Z7WBG958K4NCN34ZT](https://beaker.org/ex/01KXNTT83Z7WBG958K4NCN34ZT) |
| Shuffled | 1 / 101 | [01KXNTTS01YYWF5DNWRRZHT7Z4](https://beaker.org/ex/01KXNTTS01YYWF5DNWRRZHT7Z4) |
| Shuffled | 2 / 202 | [01KXNTV1WFJSKPWXYYEBTBTERH](https://beaker.org/ex/01KXNTV1WFJSKPWXYYEBTBTERH) |
| Shuffled | 3 / 303 | [01KXNTVARNMB767N72G7H29SRF](https://beaker.org/ex/01KXNTVARNMB767N72G7H29SRF) |
| Uniform | 1 | [01KXNTVTVNS0G8H2CSSJYEJN7X](https://beaker.org/ex/01KXNTVTVNS0G8H2CSSJYEJN7X) |
| Uniform | 2 | [01KXNTW3PNRKX141AQPGHC54ZN](https://beaker.org/ex/01KXNTW3PNRKX141AQPGHC54ZN) |
| Uniform | 3 | [01KXNTWCTXW53RQ81N4C8SY5HN](https://beaker.org/ex/01KXNTWCTXW53RQ81N4C8SY5HN) |

## Final results (2026-07-17)

All nine completed result bundles have been downloaded under
`results/adaptive_experts/<experiment-id>/`. Each bundle has all 998 expected
prediction rows (500 MATH-500, 198 GPQA Diamond, and 300 IFEval OOD) and an
empty task-error list. Scores use the full task denominators. Values below are
mean ± sample standard deviation across three runs.

| Condition | n | MATH-500 | GPQA Diamond | IFEval OOD | Macro | Hit 32,768-token cap | Empty parsed output |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Normal | 3 | 95.40 ± 0.20 | 64.81 ± 1.62 | 39.78 ± 0.77 | 66.66 ± 0.81 | 1.67 ± 0.29 | 0.84 ± 0.12 |
| Shuffled | 3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 99.90 ± 0.10 | 99.70 ± 0.27 |
| Uniform | 3 | 56.27 ± 1.10 | 21.72 ± 2.31 | 10.11 ± 1.95 | 29.36 ± 0.94 | 55.04 ± 0.75 | 53.81 ± 1.05 |

The shuffled result is a genuine generation collapse rather than a harness
failure: 2,991/2,994 outputs reached the full token allowance and 2,985/2,994
did not close reasoning with exposed answer content. Uniform weighting is less
severe but still makes 1,648/2,994 outputs reach the cap. By comparison, only
50/2,994 normal outputs reached it.

Machine-readable per-run and aggregate values are in
`notes/expert_weight_perturbation/qwen3_k8_runs.csv` and
`notes/expert_weight_perturbation/qwen3_k8_summary.csv`.
