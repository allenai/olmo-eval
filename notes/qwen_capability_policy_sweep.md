# Qwen capability-policy follow-up

This follow-up tests whether the routing-policy conclusions from MATH-500, GPQA Diamond, and
IFEval OOD extend to stricter instruction following and code generation.

## Approved launch

- Model: Qwen3-30B-A3B hybrid-thinking (`qwen3-30b-a3b`)
- Evals in every job:
  - `ifbench` (all three constituent tasks)
  - `humaneval:chat:pass_at_1:qwen3_thinking`
- Replicates: three per condition
- Conditions:
  - native normalized K=8
  - native normalized K=4
  - K=4 with weights preserving their native K=8 mass (no renormalization)
  - adaptive prefix at `tau=0.80`, renormalized
  - adaptive prefix at `tau=0.90`, renormalized
- Serving: four independent TP=1 vLLM engines per full evaluation
- Placement: `ai2/holmes-testing`, `ai2/jupiter`, urgent priority
- Beaker group:
  [adaptive-experts-qwen-capability-policies-20260717](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KXRM1V7SJMSJNEHT5J87PWQS)
- Append-only job ledger: `notes/beaker_jobs.jsonl`

The initial HumanEval+ smoke
[01KXRM2APHZ4F9P2Y7Y6B1YE10](https://beaker.org/ex/01KXRM2APHZ4F9P2Y7Y6B1YE10)
generated successfully, but one of its two hidden-test payloads exceeded the existing sandbox
transport's per-argument size limit. That infrastructure error would have been counted as an
incorrect model answer, so HumanEval+ was rejected without changing olmo-eval. The replacement
smoke used standard HumanEval:
[01KXRMPJBRF1PZ3R5BYPTA067S](https://beaker.org/ex/01KXRMPJBRF1PZ3R5BYPTA067S).
All eight HumanEval programs and all six IFBench examples scored without infrastructure errors.

HumanEval here is the one-sample `pass_at_1` task. Three complete evaluations therefore provide
three independent estimates of the same metric, matching the replication strategy used for the
other stochastic post-training evals. This is not equivalent to a single `pass@k` evaluation with
three samples per problem. If a later coding task produces `pass@1`, `pass@4`, ..., `pass@32` from
many samples in one run, retain all of those values; use repeated complete runs only when we also
want a between-run variance estimate.

## Full evaluation matrix

All 15 jobs were submitted and recorded on 2026-07-17. Each job contains the complete IFBench
suite and standard HumanEval, with four independent TP=1 Qwen vLLM engines.

| Condition | Replicate 1 | Replicate 2 | Replicate 3 |
|:--|:--|:--|:--|
| Native normalized K=8 | [01KXRN2N4RWWENY90DMYMSAT3T](https://beaker.org/ex/01KXRN2N4RWWENY90DMYMSAT3T) | [01KXRN40R1113FXYGBCA9HWEQB](https://beaker.org/ex/01KXRN40R1113FXYGBCA9HWEQB) | [01KXRN5B97XK1QK9S92PFTYN5R](https://beaker.org/ex/01KXRN5B97XK1QK9S92PFTYN5R) |
| Native normalized K=4 | [01KXRN2XR6E2K6B834EDWEP4PY](https://beaker.org/ex/01KXRN2XR6E2K6B834EDWEP4PY) | [01KXRN48Z0JJXEFND91A4MEPK8](https://beaker.org/ex/01KXRN48Z0JJXEFND91A4MEPK8) | [01KXRN5RZPQSEER8HY09YWDMZH](https://beaker.org/ex/01KXRN5RZPQSEER8HY09YWDMZH) |
| K=4 preserving native K=8 mass | [01KXRN36GZGWAKFWAMVBFMRZ27](https://beaker.org/ex/01KXRN36GZGWAKFWAMVBFMRZ27) | [01KXRN4H85YGEJ4DVDYY3AREWF](https://beaker.org/ex/01KXRN4H85YGEJ4DVDYY3AREWF) | [01KXRN6112XA0X14QA7SPPW57M](https://beaker.org/ex/01KXRN6112XA0X14QA7SPPW57M) |
| Adaptive `tau=0.80` | [01KXRN3EYFHS75PCT6D2G2S2AK](https://beaker.org/ex/01KXRN3EYFHS75PCT6D2G2S2AK) | [01KXRN4SHQTDPE5YZTC84YXWPJ](https://beaker.org/ex/01KXRN4SHQTDPE5YZTC84YXWPJ) | [01KXRN68YHNWCSAHT8GYQYZFX4](https://beaker.org/ex/01KXRN68YHNWCSAHT8GYQYZFX4) |
| Adaptive `tau=0.90` | [01KXRN3QBFFZR79H29QQWDK96H](https://beaker.org/ex/01KXRN3QBFFZR79H29QQWDK96H) | [01KXRN530KG2SW26VKSSJ9RF3H](https://beaker.org/ex/01KXRN530KG2SW26VKSSJ9RF3H) | [01KXRN6GVZQKGVYVCVCCFBQD0A](https://beaker.org/ex/01KXRN6GVZQKGVYVCVCCFBQD0A) |

## Results

All 15 full evaluations completed successfully. Their result bundles were downloaded under
`results/adaptive_experts/<experiment-id>/` and validated for all four expected task outputs and
complete prediction counts.

Values below are mean ± sample standard deviation across three complete runs. IFBench macro is the
unweighted mean of the three constituent tasks' prompt-level loose accuracies; it is a convenience
summary rather than a separately emitted metric.

| Condition | IFEval OOD loose | IFBench MT loose | IFBench MT OOD loose | IFBench macro | HumanEval pass@1 |
|:--|--:|--:|--:|--:|--:|
| Native normalized K=8 | 34.11 ± 1.95 | 80.38 ± 0.20 | 45.69 ± 0.54 | 53.39 ± 0.59 | 93.29 ± 0.00 |
| Native normalized K=4 | 18.78 ± 0.84 | 73.51 ± 0.29 | 42.42 ± 0.81 | 44.90 ± 0.37 | 82.32 ± 1.22 |
| K=4 preserving native K=8 mass | 26.78 ± 1.26 | 77.66 ± 0.12 | 43.52 ± 0.17 | 49.32 ± 0.44 | 86.18 ± 2.46 |
| Adaptive `tau=0.80` | 30.78 ± 0.77 | 79.91 ± 0.28 | 46.29 ± 0.62 | 52.33 ± 0.35 | 92.68 ± 0.61 |
| Adaptive `tau=0.90` | 34.56 ± 0.51 | 79.73 ± 0.48 | 45.90 ± 0.65 | 53.39 ± 0.18 | 94.11 ± 0.70 |

The capability follow-up strengthens the earlier conclusions:

- Adaptive `tau=0.90` matches the native K=8 IFBench macro and HumanEval performance within
  between-run variation.
- Adaptive `tau=0.80` is also close: 1.06 points below native K=8 on IFBench macro and 0.61 points
  below on HumanEval.
- Preserving native K=8 mass materially improves K=4 over survivor renormalization, but does not
  close the complete gap to K=8 on these more sensitive capabilities.
- The largest fixed-K separation is on IFEval OOD. This remains a useful discriminator in a wider
  sweep even though it is also included in the original three-task post-training suite.

## IFBench 32k refresh

The original IFBench task definitions used a 2,048-token generation cap. This was not merely a
nominal limit: across the completed runs, 30--48% of IFEval OOD generations, 24--34% of MT OOD
generations, and 5--7% of the standard MT generations reached that cap. Those IFBench scores should
therefore remain labeled as the 2k protocol and should not be mixed with the refreshed results.

All 15 original capability-policy jobs were terminal and successful when audited on 2026-07-17, so
there were no active jobs to cancel. The same five conditions were resubmitted three times with an
explicit `sampling_params.max_tokens=32768` override on each of the three IFBench constituent
tasks. Temperature remains 0, so the output cap is the only task-level protocol change. Standard
HumanEval was not repeated because its original runs already used a 32,768-token cap.

Qwen's server context remains 40,960 tokens. vLLM bounds each request to the smaller of the
requested 32,768-token cap and the space remaining after its prompt; this preserves the full cap
for ordinary prompts while safely accommodating the small number of multi-turn prompts longer
than 8,192 tokens without enabling YaRN.

- Placement: `ai2/holmes-testing`, `ai2/jupiter`, urgent priority
- Serving: four independent TP=1 vLLM engines per evaluation
- Beaker group:
  [adaptive-experts-qwen-capability-ifbench32k-20260717](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KXRXXF712W3WDCYM7QY4MZS2)
- Job ledger: `notes/beaker_jobs.jsonl`

| Condition | Replicate 1 | Replicate 2 | Replicate 3 |
|:--|:--|:--|:--|
| Native normalized K=8 | [01KXRXXKVCTBP0V40F6T88Y9V4](https://beaker.org/ex/01KXRXXKVCTBP0V40F6T88Y9V4) | [01KXRXYX3J33KFXTET3802EC94](https://beaker.org/ex/01KXRXYX3J33KFXTET3802EC94) | [01KXRY063JZ0HHYA0REHFVRV6F](https://beaker.org/ex/01KXRY063JZ0HHYA0REHFVRV6F) |
| Native normalized K=4 | [01KXRXXVTZYAJ8X2N1F05YKTXF](https://beaker.org/ex/01KXRXXVTZYAJ8X2N1F05YKTXF) | [01KXRXZ4W8DRE2N2QQJ69SHER5](https://beaker.org/ex/01KXRXZ4W8DRE2N2QQJ69SHER5) | [01KXRY0EG6HYA5D10YMTK820MD](https://beaker.org/ex/01KXRY0EG6HYA5D10YMTK820MD) |
| K=4 preserving native K=8 mass | [01KXRXY3MZ3DY62EP38FJV7KF5](https://beaker.org/ex/01KXRXY3MZ3DY62EP38FJV7KF5) | [01KXRXZDTGDTJPR7AESPGQMAAX](https://beaker.org/ex/01KXRXZDTGDTJPR7AESPGQMAAX) | [01KXRY0Q5DKYK6WPBXZXV0TY7V](https://beaker.org/ex/01KXRY0Q5DKYK6WPBXZXV0TY7V) |
| Adaptive `tau=0.80` | [01KXRXYC59XS61PY09BZ9DR3JP](https://beaker.org/ex/01KXRXYC59XS61PY09BZ9DR3JP) | [01KXRXZNGV1EDRGR22JH8HTJEH](https://beaker.org/ex/01KXRXZNGV1EDRGR22JH8HTJEH) | [01KXRY0Z9PC0H9NBXZX2G2AHY0](https://beaker.org/ex/01KXRY0Z9PC0H9NBXZX2G2AHY0) |
| Adaptive `tau=0.90` | [01KXRXYMR1VTYYGCB8RWQ4HKMP](https://beaker.org/ex/01KXRXYMR1VTYYGCB8RWQ4HKMP) | [01KXRXZXNGRZHJQSYV0S51DD7H](https://beaker.org/ex/01KXRXZXNGRZHJQSYV0S51DD7H) | [01KXRY171N0328BRFMP874RQSX](https://beaker.org/ex/01KXRY171N0328BRFMP874RQSX) |

### Complete 32k results

Status audit and collection at 2026-07-18 02:05 UTC: all fifteen evaluations succeeded, were
downloaded, and passed prediction-count validation. `±` is the sample standard deviation across
the three complete replicates. IFBench macro is the unweighted mean of the three displayed
prompt-level loose accuracies.

| Condition | n | IFEval OOD loose | IFBench MT loose | IFBench MT OOD loose | IFBench macro |
|:--|--:|--:|--:|--:|--:|
| Native normalized K=8 | 3 | 37.78 ± 1.07 | 82.68 ± 0.27 | 47.42 ± 0.23 | 55.96 ± 0.49 |
| Native normalized K=4 | 3 | 25.78 ± 1.71 | 75.80 ± 0.48 | 42.87 ± 0.67 | 48.15 ± 0.35 |
| K=4 preserving native K=8 mass | 3 | 36.22 ± 0.96 | 80.25 ± 0.03 | 46.38 ± 0.34 | 54.29 ± 0.41 |
| Adaptive `tau=0.80` | 3 | 34.78 ± 0.84 | 81.75 ± 0.38 | 48.14 ± 0.25 | 54.89 ± 0.39 |
| Adaptive `tau=0.90` | 3 | 39.00 ± 1.53 | 82.36 ± 0.43 | 47.63 ± 0.42 | 56.33 ± 0.53 |

The complete 32k refresh preserves the substantive ordering while producing materially higher
IFEval OOD values than the earlier 2k-capped runs. Reference-scaled K=4 recovers 6.14 macro
points over normalized K=4 and finishes only 1.67 points below native K=8. Both adaptive
conditions remain close to the native baseline, with `tau=0.90` slightly above it within
between-run variation.

## Expanded wider sweeps

### Adaptive router-mass threshold

The approved production continuation uses the following threshold grid and records the realized
number of active experts per token and per layer:

```text
tau = 0.50, 0.60, 0.70, 0.80, 0.90
```

Three replicates per threshold were launched on 2026-07-17. See
`expanded_routing_sweeps_launch.md` for the exact suite, realized-K schema, Beaker groups, and
launch audit.

### Reference-scaled expert-count curves

“Reference-scaled” means that routing weights retain their scale relative to the model's default
expert count instead of being renormalized at the tested K.

For Qwen, three new evaluations were launched at each missing K in
K={3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16}. Native K=8 is the identity point and existing K=4
coverage is reused. K=1, K=2, and K=32 remain excluded because of their earlier serving failures.

For GPT-OSS-120B, whose native routing count is K=4, three new reference-scaled evaluations were
launched at K={1, 2, 3, 5, 6, 7, 8}; native K=4 is reused as the identity. For K<4, selected
experts preserve their shares from the native top-four mixture without renormalizing. For K>4,
the first four retain their native scale and additional experts add mass. A dedicated vLLM routing
hook and TP=1/TP=2 smoke gate cover these conditions.
