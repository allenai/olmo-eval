# Qwen3.5-35B-A3B hybrid-model expert sweep

Last updated: 2026-07-25

## Question

Test whether the normalized and K=8-reference-preserving expert-count interventions used for
Qwen3-30B-A3B also work with Qwen3.5's hybrid DeltaNet/attention architecture.

Checkpoint: `Qwen/Qwen3.5-35B-A3B`. The model has 256 routed experts and selects eight per token.
Its Hugging Face configuration nests the MoE settings under `text_config`, so native normalized
K is set with:

```json
{"text_config": {"num_experts_per_tok": K}}
```

The language-only vLLM path is used, thinking is enabled, and the `qwen3` reasoning parser is
selected.

## Conditions

For K in {4,5,6,7,8}, run three independent evaluation replicates under both policies:

- **normalized:** vLLM routes and renormalizes the native top K;
- **reference-preserving:** vLLM first computes the normalized native top eight, then ranks
  K+1 through 8 are set to zero without renormalizing the retained weights.

Every job evaluates the current four-capability suite with a 32,768-token generation cap:

- MATH-500;
- GPQA Diamond;
- IFBench (the three existing task components, aggregated later); and
- standard HumanEval chat pass@1.

This is 5 K values x 2 policies x 3 replicates = 30 Beaker experiments. Each experiment uses
four independent TP=1 vLLM engines (four H100s).

## Smoke gate and backend findings

The v5 smokes below exited zero but are **invalid**. Their six limit-one task rows were recorded as
scored, but every `model_output` was empty. Downloading the vLLM logs showed that the first DeltaNet
forward pass failed because FlashInfer's EngineCore process could not locate `ninja`; the harness
then converted the API failures into zero-scored predictions without populating `metrics.errors`.

| Check | Beaker |
|:--|:--|
| Native normalized K=4, invalid v5 | [01KYBG2A76S37Q57QG7R7W5MAP](https://beaker.org/ex/01KYBG2A76S37Q57QG7R7W5MAP) |
| K=4 reference-preserving, invalid v5 | [01KYBG2SQA9JN5SWXEJNZYVD11](https://beaker.org/ex/01KYBG2SQA9JN5SWXEJNZYVD11) |
| Native/default normalized K=8, invalid v5 | [01KYBG39VJJA76PN374EPD36FK](https://beaker.org/ex/01KYBG39VJJA76PN374EPD36FK) |

The root cause was resolving the venv Python symlink before deriving its sibling `bin` directory,
which changed `/opt/vllm-venv/bin` to `/usr/local/bin`. The fix now preserves the literal venv path
and installs the startup hook for native normalized Qwen3.5 runs as well as patched runs.

The olmo-eval image uses CUDA 12.8.1 and PyTorch 2.10, so model loading and ordinary Triton kernels
are Blackwell-capable. Qwen3.5's default FlashInfer DeltaNet path is not portable to this image:
on B200 its TRTLLM path attempts a source build, and on H100 the automatically selected FlashInfer
GDN prefill kernel also attempts a source build; neither image exposes `nvcc`.

Two independent vLLM controls are therefore required:

```text
--attention-backend TRITON_ATTN --gdn-prefill-backend triton
```

The Titan smokes below passed with non-empty generations and no inference HTTP failures:

| Check | Beaker |
|:--|:--|
| Native normalized K=4 | [01KYBMZQ46XE2BJ44KNBGJJAQB](https://beaker.org/ex/01KYBMZQ46XE2BJ44KNBGJJAQB) |
| K=4, K=8-reference-preserving | [01KYBNBXHE9BK430HV3S1QWRT8](https://beaker.org/ex/01KYBNBXHE9BK430HV3S1QWRT8) |
| Native/default normalized K=8 | [01KYBNC9KM3XW18X91JX8RAF1D](https://beaker.org/ex/01KYBNC9KM3XW18X91JX8RAF1D) |

A final six-task H100/Jupiter smoke
[01KYBPHH6JBEF3Q4AX3CX6QVQF](https://beaker.org/ex/01KYBPHH6JBEF3Q4AX3CX6QVQF) exercised both
Triton controls, generated and scored one example from every task (including HumanEval's local
sandbox), saved all six prediction files, and exited zero.

## Production launch

Several invalid generations preceded the final matrix. The first production attempt produced
empty outputs after the Ninja/path setup failure. A later v2 submission was stopped before use
after duplicate conditions were discovered in the local submission flow. The 30 v3 jobs included
`TRITON_ATTN` but not the separate GDN prefill control; the first H100 job failed while trying to
invoke missing `nvcc`, and all 30 were stopped promptly. None of v1-v3 may enter a score table or
plot. They remain in the append-only ledger for auditability and are grouped under
[adaptive-experts-qwen35-35b-a3b-20260725](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KYBEQ5CA3XYSF0FSTVNFNSSQ).

The validated v4 production matrix was submitted on 2026-07-25 in `ai2/holmes-testing`,
unallocated on `ai2/jupiter`, at urgent priority. It contains exactly 30 unique run tags and 30
unique Beaker experiment IDs. Every job has four TP=1 H100 engines, both Triton backend controls,
all six task components, and a 32,768-token generation cap. Query only this matrix with:

```bash
jq -r 'select((.run_tag // "") |
  test("^qwen35-k[4-8]-(normalized|reference)-r[1-3]-v4-20260725$")) |
  [.run_tag, .beaker_experiment_id, .url] | @tsv' notes/beaker_jobs.jsonl
```

Results and plots must use only those v4 tags. The matrix can remain queued on Jupiter while the
one-node RL baseline uses Titan capacity.

## Final results and context-limit failures

All 30 valid-v4 jobs completed successfully and were collected by 2026-07-26. Every condition has
three independent evaluation replicates. The final outputs are:

- `notes/plots/qwen35_35b_a3b_expert_sweep.png`;
- `notes/plots/qwen35_35b_a3b_expert_sweep.svg`; and
- `notes/qwen35_35b_a3b_results.csv`.

Scores below are mean ± one sample standard deviation in percentage points:

| Routing | K | MATH-500 | GPQA Diamond | IFBench macro | HumanEval pass@1 |
|:--|--:|--:|--:|--:|--:|
| Normalized | 4 | 78.80 ± 1.60 | 78.96 ± 0.77 | 34.02 ± 0.61 | 24.80 ± 1.76 |
| Normalized | 5 | 81.27 ± 2.08 | 82.49 ± 1.46 | 42.59 ± 0.91 | 24.80 ± 3.36 |
| Normalized | 6 | 86.13 ± 1.10 | 83.00 ± 2.10 | 45.95 ± 0.47 | 45.93 ± 4.49 |
| Normalized | 7 | 87.60 ± 1.22 | 84.34 ± 0.87 | 47.40 ± 0.69 | 80.08 ± 3.47 |
| Normalized | 8 | 90.60 ± 2.16 | 83.16 ± 1.27 | 49.14 ± 0.14 | 85.77 ± 2.31 |
| K=8-reference scaled | 4 | 80.47 ± 0.64 | 80.13 ± 1.17 | 45.76 ± 0.58 | 62.40 ± 1.76 |
| K=8-reference scaled | 5 | 85.67 ± 1.01 | 82.15 ± 1.27 | 49.14 ± 0.64 | 65.04 ± 3.36 |
| K=8-reference scaled | 6 | 87.33 ± 0.12 | 82.49 ± 0.58 | 48.24 ± 0.82 | 76.22 ± 1.61 |
| K=8-reference scaled | 7 | 89.67 ± 1.47 | 82.66 ± 1.27 | 48.81 ± 0.31 | 85.77 ± 2.14 |
| K=8-reference scaled | 8 | 91.13 ± 0.95 | 82.49 ± 0.77 | 48.04 ± 1.04 | 87.80 ± 0.00 |

The reference-scaled intervention is markedly more robust below K=8, especially for instruction
following and code. At K=4, it improves IFBench by 11.73 points and HumanEval by 37.60 points over
normalized K=4. It does not fully preserve HumanEval, which still loses about 25 points relative
to reference-scaled K=8. GPQA is comparatively flat and noisy across K.

Each completed run logs 68 HTTP 400 context-limit events, but these are retries rather than 68
distinct examples. They correspond to 17 unique IFBench multilingual prompts, each attempted four
times: nine examples from `ifeval_mt_ood_wildchat_unused_withRewrite` and eight from
`ifeval_mt_wildchat_unused_withRewrite`, both sourced from `VGraf/ifeval_mt`. With a 40,960-token
server context and a 32,768-token output allowance, their prompts exceed the remaining 8,192-token
input budget. olmo-eval retains each as an empty output and scores it as a failure. Thus 17 of the
4,323 task instances per run (0.39%) are affected, consistently across conditions and replicates.
