# Adaptive expert sweep results

Last updated: 2026-07-22

All initial sweep experiments completed successfully. Pretraining reports only OLMoBase suite
summaries; individual task metrics are intentionally omitted. Accuracy-like metrics are
higher-is-better (↑), while bits per byte is lower-is-better (↓).

Three-replicate summaries are reported as mean ± sample standard deviation; score standard
deviations are in percentage points. Non-math OLMoBase suites are deterministic under these
configs and were run once per K. Downloaded artifacts are stored under
`results/adaptive_experts/<beaker-experiment-id>/`.

## Pretraining: Qwen3-30B-A3B-Base

### Experiment status

| K | Consolidated Beaker experiment | Status |
|---:|:-------------------------------|:-------|
| 1 | [01KX597HRNRSJD6Q028SGR7X1J](https://beaker.org/ex/01KX597HRNRSJD6Q028SGR7X1J) | Succeeded |
| 2 | [01KX597TE7GSPMXKQAHDKJ0TZG](https://beaker.org/ex/01KX597TE7GSPMXKQAHDKJ0TZG) | Succeeded |
| 3 | [01KX7GKDS69Y3P61MPTJD9BPWD](https://beaker.org/ex/01KX7GKDS69Y3P61MPTJD9BPWD) | Succeeded |
| 4 | [01KX598334MGBFC3RN01PR27RH](https://beaker.org/ex/01KX598334MGBFC3RN01PR27RH) | Succeeded |
| 5 | [01KX6M1VV8H27K4J2CVDV64Y82](https://beaker.org/ex/01KX6M1VV8H27K4J2CVDV64Y82) | Succeeded |
| 6 | [01KX6M2634H40FR7WZP0WGCSZ9](https://beaker.org/ex/01KX6M2634H40FR7WZP0WGCSZ9) | Succeeded |
| 7 | [01KX6M2QW5FG1BNH5DHW7F1PM1](https://beaker.org/ex/01KX6M2QW5FG1BNH5DHW7F1PM1) | Succeeded |
| 8 | [01KX598BVQG62BXW751MJMEZTG](https://beaker.org/ex/01KX598BVQG62BXW751MJMEZTG) | Succeeded |
| 9 | [01KX7GKR8K19TYT3Z26EA5RVQC](https://beaker.org/ex/01KX7GKR8K19TYT3Z26EA5RVQC) | Succeeded |
| 10 | [01KX7GM1MGHZYWYG9BWE55JABZ](https://beaker.org/ex/01KX7GM1MGHZYWYG9BWE55JABZ) | Succeeded |
| 11 | [01KX7GMAB9E6JEZ669XR33CEZA](https://beaker.org/ex/01KX7GMAB9E6JEZ669XR33CEZA) | Succeeded |
| 12 | [01KX7GMM4TCFTVA32T7RPTNZHB](https://beaker.org/ex/01KX7GMM4TCFTVA32T7RPTNZHB) | Succeeded |
| 16 | [01KX598MHGPFNET048MXHQGX64](https://beaker.org/ex/01KX598MHGPFNET048MXHQGX64) | Succeeded |

### OLMoBase suite summaries

| OLMoBase suite | Metric | Dir. | K=1 | K=2 | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 | K=9 | K=10 | K=11 | K=12 | K=16 |
|:---------------|:-------|:----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|-----:|-----:|-----:|-----:|
| `olmobase:mcqa_stem` | Score | ↑ | 22.70% | 32.34% | 66.27% | 75.22% | 79.71% | 77.13% | 79.52% | 69.69% | 74.55% | 68.36% | 72.89% | 44.39% | 39.60% |
| `olmobase:mcqa_non_stem` | Score | ↑ | 25.37% | 31.63% | 64.73% | 76.96% | 82.26% | 79.95% | 81.91% | 72.30% | 77.32% | 71.26% | 75.45% | 46.57% | 41.72% |
| `olmobase:gen` | Score | ↑ | 10.31% | 20.45% | 48.66% | 62.17% | 64.63% | 62.33% | 65.28% | 59.78% | 64.70% | 60.99% | 64.72% | 39.93% | 36.38% |
| `olmobase:math` | Score | ↑ | 0.57±0.01% | 0.95±0.02% | 24.33±0.82% | 45.62±1.19% | 58.93±0.81% | 60.19±2.40% | 66.15±1.86% | 65.02±6.45% | 66.96±4.42% | 66.44±7.60% | 67.93±5.06% | 60.35±18.53% | 58.47±20.61% |
| `olmobase:easy:qa:rc` | Score | ↑ | 24.98% | 40.38% | 57.66% | 69.04% | 74.39% | 71.80% | 74.27% | 65.61% | 70.11% | 64.73% | 68.61% | 42.47% | 38.46% |
| `olmobase:easy:qa:bpb` | BPB | ↓ | 4.2936 | 1.6105 | 0.9457 | 0.7571 | 0.7164 | 0.6904 | 0.6771 | 0.6615 | 0.6605 | 0.6524 | 0.6485 | 0.6180 | 0.6071 |
| `olmobase:easy:math:bpb` | BPB | ↓ | 4.9600 | 0.8427 | 0.4249 | 0.3470 | 0.3214 | 0.3120 | 0.3073 | 0.3063 | 0.3081 | 0.3082 | 0.3088 | 0.3087 | 0.3139 |
| `olmobase:easy:code:bpb` | BPB | ↓ | 4.6291 | 0.8391 | 0.4239 | 0.3179 | 0.2885 | 0.2754 | 0.2624 | 0.2581 | 0.2551 | 0.2582 | 0.2599 | 0.2594 | 0.2552 |

### OLMoBase math replicates

Only the stochastic `olmobase:math` suite was repeated. Replicate 1 was run inside the full
OLMoBase evaluation, while replicates 2 and 3 were math-only jobs. The large K=8 and K=16 shifts
may therefore include a suite-scope, batching, or random-stream effect rather than measuring pure
run-to-run noise; the individual values are preserved here rather than hidden inside the aggregate.

| K | Replicate 1 | Replicate 2 | Replicate 3 | Mean ± sample SD |
|---:|------------:|------------:|------------:|-----------------:|
| 1 | 0.57% | 0.57% | 0.58% | 0.57% ± 0.01 pp |
| 2 | 0.96% | 0.95% | 0.93% | 0.95% ± 0.02 pp |
| 3 | 23.39% | 24.88% | 24.73% | 24.33% ± 0.82 pp |
| 4 | 44.25% | 46.26% | 46.36% | 45.62% ± 1.19 pp |
| 5 | 57.99% | 59.36% | 59.43% | 58.93% ± 0.81 pp |
| 6 | 57.41% | 61.69% | 61.46% | 60.19% ± 2.40 pp |
| 7 | 64.00% | 67.31% | 67.13% | 66.15% ± 1.86 pp |
| 8 | 57.57% | 68.78% | 68.71% | 65.02% ± 6.45 pp |
| 9 | 61.85% | 69.46% | 69.57%† | 66.96% ± 4.42 pp |
| 10 | 57.67% | 70.85% | 70.82% | 66.44% ± 7.60 pp |
| 11 | 62.09% | 70.76% | 70.94% | 67.93% ± 5.06 pp |
| 12 | 38.95% | 70.98% | 71.11% | 60.35% ± 18.53 pp |
| 16 | 34.67% | 70.26% | 70.48% | 58.47% ± 20.61 pp |

### Constituent OLMoBase evaluations

The [constituent-evaluation plot](plots/pretraining_individual_evals_expert_sweep.png) replaces each
top-level OLMoBase summary with its meaningful direct dataset-level components and retains the
aggregate only as a dashed reference. The Minerva Math BPB panel is expanded into its seven subject
areas. The accompanying [long-form CSV](pretraining_individual_evals_expert_sweep.csv) records the
exact suite, evaluation, direction, K, and score used for every line.

This view uses the one consolidated full-suite result at every K so all constituent scores come
from the same run. Consequently, its math panel shows replicate 1 rather than the three-run mean in
the compact summary table above. Across the full K range, individual curves generally reproduce
the aggregate behavior: median constituent-to-aggregate correlation is 0.99 for both MCQA groups,
0.99 for math, at least 0.97 for easy QA RC, Minerva BPB, and code BPB, and 0.94 for generation.
Individual generation tasks are visibly noisier on the K=4--11 plateau—especially SQuAD, DROP, and
NaturalQuestions—while MCQA, math, and the Minerva/code BPB constituents mostly move together.
The common score collapse at K=12/16 is present across constituent datasets and is therefore not an
artifact of suite averaging.

## Post-training: Qwen3-30B-A3B hybrid thinking

All three evaluations use a 32,768-token generation cap. HumanEval+ is excluded. Accuracy
percentages use the full dataset as the denominator, so unfinished responses count as incorrect.

The table below is the historical three-eval suite. The adopted current four-eval normalized and
K=8-reference-scaled curves are now complete for K=3--16 and are shown in the
[current-suite plot](plots/qwen3_hybrid_current_suite_expert_sweep.png). Exact MATH-500, GPQA,
IFBench-32k, and HumanEval means/SDs are in
[current_suite_expert_sweeps.csv](current_suite_expert_sweeps.csv); collection details are in the
[expanded sweep record](expanded_routing_sweeps_launch.md).

### Primary scores

The [replicate-range plot](plots/posttraining_expert_sweep_replicate_range.png) is a variance-focused
copy of the original Qwen3 curve. Its shaded bands show the observed minimum-to-maximum score at
each K, while dotted lines show the replicate means. The exact values are recorded in the
[range summary CSV](qwen3_posttraining_replicate_ranges.csv).

| K | Beaker experiment | Status | MATH-500 accuracy ↑ | GPQA Diamond accuracy ↑ | IFEval OOD loose prompt accuracy ↑ | Macro average ↑ |
|---:|:------------------|:-------|--------------------:|------------------------:|-----------------------------------:|----------------:|
| 1 | [01KX56NQMVH3TVZKGAM9DPGF05](https://beaker.org/ex/01KX56NQMVH3TVZKGAM9DPGF05) | 3/3 complete | 0.00% ± 0.00 pp | 0.00% ± 0.00 pp | 0.00% ± 0.00 pp | 0.00% ± 0.00 pp |
| 2 | [01KX56NZQVWQBKXH6VP6CXV345](https://beaker.org/ex/01KX56NZQVWQBKXH6VP6CXV345) | 3/3 complete | 0.07% ± 0.12 pp | 0.67% ± 0.29 pp | 0.00% ± 0.00 pp | 0.25% ± 0.08 pp |
| 3 | [01KX7GFW9RCAPT76WTTBRWT0SR](https://beaker.org/ex/01KX7GFW9RCAPT76WTTBRWT0SR) | 3/3 complete | 62.13% ± 0.76 pp | 40.07% ± 5.56 pp | 8.44% ± 2.14 pp | 36.88% ± 2.49 pp |
| 4 | [01KX56P8M9TD1MPMCT582AXZ87](https://beaker.org/ex/01KX56P8M9TD1MPMCT582AXZ87) | 3/3 complete | 94.27% ± 0.23 pp | 56.06% ± 0.51 pp | 28.89% ± 1.39 pp | 59.74% ± 0.43 pp |
| 5 | [01KX6KSNE491THB9DT4Y9PPJ7W](https://beaker.org/ex/01KX6KSNE491THB9DT4Y9PPJ7W) | 3/3 complete | 95.07% ± 0.70 pp | 60.27% ± 2.54 pp | 34.44% ± 0.69 pp | 63.26% ± 0.69 pp |
| 6 | [01KX6KSYGH7H8RQPEGJFYC3PEW](https://beaker.org/ex/01KX6KSYGH7H8RQPEGJFYC3PEW) | 3/3 complete | 95.07% ± 0.23 pp | 62.46% ± 1.77 pp | 38.11% ± 0.69 pp | 65.21% ± 0.49 pp |
| 7 | [01KX6KT77JRKJSW67J2NY5BFQQ](https://beaker.org/ex/01KX6KT77JRKJSW67J2NY5BFQQ) | 3/3 complete | 95.13% ± 0.31 pp | 62.46% ± 2.04 pp | 37.78% ± 1.90 pp | 65.12% ± 0.87 pp |
| 8 | [01KX56PH4KDPVQFSBB3ZHDEMB7](https://beaker.org/ex/01KX56PH4KDPVQFSBB3ZHDEMB7) | 3/3 complete | 95.33% ± 0.12 pp | 61.28% ± 2.04 pp | 37.78% ± 0.51 pp | 64.80% ± 0.79 pp |
| 9 | [01KX7GG47N7XP7K8NYW9NE7B73](https://beaker.org/ex/01KX7GG47N7XP7K8NYW9NE7B73) | 3/3 complete | 94.40% ± 0.53 pp | 62.46% ± 1.77 pp | 40.56% ± 1.02 pp | 65.80% ± 0.51 pp |
| 10 | [01KX7GGCK1WZX5XKEW3VHHDCB4](https://beaker.org/ex/01KX7GGCK1WZX5XKEW3VHHDCB4) | 3/3 complete | 94.87% ± 0.61 pp | 62.46% ± 0.29 pp | 42.44% ± 1.26 pp | 66.59% ± 0.50 pp |
| 11 | [01KX7GGMF6J8VV1YJ8F1REKFQ3](https://beaker.org/ex/01KX7GGMF6J8VV1YJ8F1REKFQ3) | 3/3 complete | 95.00% ± 0.35 pp | 63.30% ± 1.27 pp | 41.11% ± 0.69 pp | 66.47% ± 0.61 pp |
| 12 | [01KX7GGVVD8NQS8W2RFNAWNWXS](https://beaker.org/ex/01KX7GGVVD8NQS8W2RFNAWNWXS) | 3/3 complete | 94.47% ± 0.23 pp | 60.94% ± 0.29 pp | 40.67% ± 2.00 pp | 65.36% ± 0.69 pp |
| 13 | [01KX7VBBDJDTWW1YK3010QB7R0](https://beaker.org/ex/01KX7VBBDJDTWW1YK3010QB7R0) | 2/3 complete; r3 not launched | 93.80% ± 0.00 pp | 60.10% ± 5.00 pp | 40.67% ± 1.89 pp | 64.86% ± 2.30 pp |
| 14 | [01KX7VBM2Z141G67RM7GB8C0R9](https://beaker.org/ex/01KX7VBM2Z141G67RM7GB8C0R9) | 2/3 complete; r3 not launched | 94.00% ± 0.28 pp | 61.62% ± 1.43 pp | 41.00% ± 0.00 pp | 65.54% ± 0.38 pp |
| 15 | [01KX7VBW0H13C928JGVQ9D3GV1](https://beaker.org/ex/01KX7VBW0H13C928JGVQ9D3GV1) | 2/3 complete; r3 not launched | 94.40% ± 0.57 pp | 63.38% ± 2.50 pp | 40.83% ± 0.71 pp | 66.21% ± 0.88 pp |
| 16 | [01KX56PSK9WJMQ68BTQ3P54P5D](https://beaker.org/ex/01KX56PSK9WJMQ68BTQ3P54P5D) | 3/3 complete | 93.73% ± 0.12 pp | 58.25% ± 2.96 pp | 42.44% ± 1.71 pp | 64.81% ± 1.11 pp |
| 32 | [01KX5ACWAEQSRM0E54YFBB4Z8H](https://beaker.org/ex/01KX5ACWAEQSRM0E54YFBB4Z8H) | 3/3 complete | 76.00% ± 1.44 pp | 36.20% ± 1.05 pp | 33.67% ± 1.33 pp | 48.62% ± 0.75 pp |

### Generation diagnostics (replicate 1)

Final-answer and cap-hit counts are independent and are reported against each task's full sample
count.

| K | MATH finals | MATH cap hits | GPQA finals | GPQA cap hits | IFEval finals | IFEval cap hits |
|---:|------------:|--------------:|------------:|--------------:|--------------:|----------------:|
| 1 | 0/500 | 261/500 | 0/198 | 95/198 | 0/300 | 141/300 |
| 2 | 7/500 | 482/500 | 7/198 | 183/198 | 0/300 | 300/300 |
| 3 | 346/500 | 59/500 | 183/198 | 4/198 | 111/300 | 137/300 |
| 4 | 494/500 | 3/500 | 198/198 | 0/198 | 248/300 | 32/300 |
| 5 | 499/500 | 2/500 | 198/198 | 0/198 | 283/300 | 17/300 |
| 6 | 499/500 | 2/500 | 198/198 | 0/198 | 294/300 | 14/300 |
| 7 | 500/500 | 0/500 | 198/198 | 0/198 | 292/300 | 12/300 |
| 8 | 499/500 | 1/500 | 198/198 | 0/198 | 292/300 | 15/300 |
| 9 | 499/500 | 1/500 | 198/198 | 0/198 | 291/300 | 15/300 |
| 10 | 500/500 | 0/500 | 198/198 | 0/198 | 287/300 | 21/300 |
| 11 | 499/500 | 1/500 | 198/198 | 0/198 | 286/300 | 18/300 |
| 12 | 495/500 | 6/500 | 197/198 | 1/198 | 279/300 | 28/300 |
| 13 | 497/500 | 3/500 | 197/198 | 1/198 | 284/300 | 28/300 |
| 14 | 497/500 | 4/500 | 197/198 | 1/198 | 272/300 | 35/300 |
| 15 | 497/500 | 3/500 | 197/198 | 1/198 | 279/300 | 29/300 |
| 16 | 496/500 | 4/500 | 196/198 | 2/198 | 269/300 | 37/300 |
| 32 | 414/500 | 87/500 | 137/198 | 61/198 | 207/300 | 103/300 |

### Mean generated tokens (replicate 1)

These counts include reasoning tokens consumed by the Qwen reasoning parser.

| K | MATH-500 | GPQA Diamond | IFEval OOD |
|---:|---------:|-------------:|-----------:|
| 1 | 22,815 | 22,664 | 21,681 |
| 2 | 31,887 | 30,850 | 32,768 |
| 3 | 11,152 | 7,511 | 16,622 |
| 4 | 6,234 | 6,611 | 5,339 |
| 5 | 5,603 | 6,504 | 3,566 |
| 6 | 5,294 | 6,016 | 3,286 |
| 7 | 5,043 | 5,924 | 3,018 |
| 8 | 5,021 | 5,862 | 3,348 |
| 9 | 4,951 | 5,708 | 3,382 |
| 10 | 4,835 | 5,722 | 3,945 |
| 11 | 4,677 | 5,437 | 3,714 |
| 12 | 4,885 | 5,438 | 4,695 |
| 13 | 4,773 | 5,436 | 4,815 |
| 14 | 4,794 | 5,302 | 5,606 |
| 15 | 4,724 | 5,227 | 4,941 |
| 16 | 4,668 | 5,363 | 5,729 |
| 32 | 8,308 | 12,618 | 12,341 |

## Variance launch status

Forty new experiments were submitted from the same source snapshot. All 40 succeeded. Together
with the 11 original sweep jobs, all 51 final-sweep result bundles are archived locally. The
archive contains 5,635 files (about 21 GB), and every local file count matches its Beaker dataset
manifest as of the timestamp above.

| Scope | New experiments | Replicate tags |
|:------|----------------:|:---------------|
| Post-training repeats at K=1/2/4/8/16/32 | 12 | `variance-r2-20260710`, `variance-r3-20260710` |
| Post-training curve at K=5/6/7 | 9 | `curve-r1-20260710`, `curve-r2-20260710`, `curve-r3-20260710` |
| Base math repeats at K=1/2/4/8/16 | 10 | `variance-r2-20260710`, `variance-r3-20260710` |
| Full base curve at K=5/6/7 | 3 | `curve-r1-20260710` |
| Base math curve repeats at K=5/6/7 | 6 | `curve-r2-20260710`, `curve-r3-20260710` |

## Curve expansion collection status

The K=3/9/10/11/12 expansion has 30 usable result bundles collected. All 15 post-training runs
succeeded. All five full-base runs succeeded. Nine of ten original
math-only repeats succeeded, and the failed K=9 r3 port-collision run was replaced by successful
[retry 01KX7MXDHY7QC41TSHQ0CFZRJV](https://beaker.org/ex/01KX7MXDHY7QC41TSHQ0CFZRJV).

The six Qwen post-training K=13/14/15 experiments also succeeded and are collected. They provide
two replicates per K; the third replicates were never launched. `†` marks the K=9 replacement r3
score.

| Scope | Usable results | Current status |
|:------|---------------:|:---------------|
| Post-training K=3/9/10/11/12 | 15 | Complete and collected |
| Full base K=3/9/10/11/12 | 5/5 | Complete and collected |
| Base math repeats K=3/9/10/11/12 | 10/10 | Complete after one replacement |
| Post-training K=13/14/15 | 6 | Complete and collected; n=2 per K |

## Post-training: GPT-OSS-120B

All K=1--8 points now have three valid replicates and use the same MATH-500, GPQA Diamond, and
IFEval OOD suite as Qwen. Values are mean ± sample standard deviation in percentage points.

These are the historical-suite values. The [current-suite plot](plots/gptoss_120b_current_suite_expert_sweep.png)
now contains complete three-replicate normalized and K=4-reference-scaled curves at every K from
1 through 8. Exact current-suite values are in
[current_suite_expert_sweeps.csv](current_suite_expert_sweeps.csv).

| K | Runs | MATH-500 accuracy ↑ | GPQA Diamond accuracy ↑ | IFEval OOD loose prompt accuracy ↑ | Macro average ↑ |
|---:|:----:|--------------------:|------------------------:|-----------------------------------:|----------------:|
| 1 | 3/3 | 42.80% ± 2.16 pp | 29.97% ± 3.72 pp | 20.22% ± 4.22 pp | 31.00% ± 0.62 pp |
| 2 | 3/3 | 92.00% ± 0.53 pp | 70.03% ± 0.29 pp | 64.67% ± 2.60 pp | 75.57% ± 0.84 pp |
| 3 | 3/3 | 91.87% ± 1.10 pp | 74.58% ± 1.27 pp | 63.22% ± 1.95 pp | 76.56% ± 1.08 pp |
| 4 | 3/3 | 92.40% ± 0.35 pp | 70.54% ± 1.62 pp | 65.11% ± 1.17 pp | 76.02% ± 0.66 pp |
| 5 | 3/3 | 91.87% ± 0.90 pp | 72.22% ± 0.87 pp | 65.00% ± 0.88 pp | 76.36% ± 0.36 pp |
| 6 | 3/3 | 92.47% ± 0.99 pp | 70.54% ± 1.54 pp | 64.00% ± 1.86 pp | 75.67% ± 1.42 pp |
| 7 | 3/3 | 92.00% ± 0.53 pp | 70.20% ± 0.51 pp | 64.89% ± 1.90 pp | 75.70% ± 0.75 pp |
| 8 | 3/3 | 91.93% ± 0.76 pp | 68.52% ± 4.11 pp | 62.11% ± 2.67 pp | 74.19% ± 1.84 pp |

K=3/5/6/7 use four TP=2 engines with NCCL and eager execution; all 12 reruns completed with full
outputs. The earlier TP=1 empty-output runs and failed TP=2 startup diagnostics remain excluded.

## Post-training: GLM-4.5-Air

Each point has three runs using two TP=4 vLLM engines. Values are mean ± sample standard deviation.

| K | Runs | MATH-500 accuracy ↑ | GPQA Diamond accuracy ↑ | IFEval OOD loose prompt accuracy ↑ | Macro average ↑ |
|---:|:----:|--------------------:|------------------------:|-----------------------------------:|----------------:|
| 1 | 3/3 | 0.07% ± 0.12 pp | 0.34% ± 0.58 pp | 0.11% ± 0.19 pp | 0.17% ± 0.14 pp |
| 2 | 3/3 | 53.67% ± 0.95 pp | 22.22% ± 4.40 pp | 17.22% ± 1.84 pp | 31.04% ± 1.03 pp |
| 3 | 3/3 | 91.07% ± 0.76 pp | 55.22% ± 2.04 pp | 26.44% ± 2.27 pp | 57.58% ± 1.19 pp |
| 4 | 3/3 | 94.73% ± 0.31 pp | 64.98% ± 1.54 pp | 30.56% ± 1.35 pp | 63.42% ± 0.38 pp |
| 5 | 3/3 | 94.80% ± 0.40 pp | 65.49% ± 1.27 pp | 34.11% ± 2.34 pp | 64.80% ± 0.81 pp |
| 7 | 3/3 | 95.60% ± 0.35 pp | 59.60% ± 4.40 pp | 37.44% ± 1.54 pp | 64.21% ± 1.36 pp |
| 8 | 3/3 | 95.53% ± 0.42 pp | 60.61% ± 2.31 pp | 39.33% ± 0.58 pp | 65.16% ± 0.63 pp |

## AIME 2026 pass@32

Each retained experiment evaluates 30 problems with 32 stochastic samples per problem. The plots
show pass@1/4/8/16/32; these compact tables show the primary pass@1 score and pass@32.

### Qwen3-30B-A3B hybrid-thinking

| K | Status | pass@1 ↑ | pass@32 ↑ |
|---:|:-------|---------:|----------:|
| 1 | Complete | 0.00% | 0.00% |
| 2 | Complete | 0.00% | 0.00% |
| 3 | Complete | 27.60% | 53.33% |
| 4 | Complete | 62.71% | 80.00% |
| 5 | Complete | 71.67% | 86.67% |
| 6 | Complete | 71.46% | 86.67% |
| 7 | Complete | 73.13% | 90.00% |
| 8 | Complete | 72.50% | 90.00% |
| 9 | Complete | 70.42% | 86.67% |
| 10 | Complete | 69.27% | 90.00% |
| 11 | Complete | 68.54% | 86.67% |
| 12 | Complete | 67.92% | 86.67% |
| 13 | Complete | 62.29% | 86.67% |
| 14 | Complete | 62.81% | 86.67% |
| 15 | Complete | 60.73% | 80.00% |
| 16 | Complete | 56.15% | 83.33% |
| 32 | Complete | 18.54% | 53.33% |

Qwen K=1 and K=2 produced all expected generations but no correct scorable answers; their zeros
are retained as model behavior rather than infrastructure failures. K=32 also completed, with
18.54% pass@1 and 53.33% pass@32.

### GPT-OSS-120B

| K | Status | pass@1 ↑ | pass@32 ↑ |
|---:|:-------|---------:|----------:|
| 1 | Invalid | — | — |
| 2 | Complete | 80.52% | 96.67% |
| 3 | Complete | 81.98% | 96.67% |
| 4 | Complete | 83.02% | 96.67% |
| 5 | Complete | 80.52% | 96.67% |
| 6 | Complete | 79.79% | 96.67% |
| 7 | Complete | 79.17% | 96.67% |
| 8 | Complete | 79.69% | 96.67% |

GPT-OSS K=1 returned no model outputs for all 30 requests. Its vLLM logs show repeated Harmony
decoding failures and HTTP 500 responses, so the emitted zero metrics and K=1 point are excluded
from the plot.

### GLM-4.5-Air

| K | Status | pass@1 ↑ | pass@32 ↑ |
|---:|:-------|---------:|----------:|
| 2 | Complete | 3.44% | 16.67% |
| 4 | Complete | 68.02% | 86.67% |
| 8 | Complete | 75.42% | 93.33% |

## Status snapshot

Historical deleted experiments remain in the append-only ledger and are reported as missing by
Beaker; smoke tests and failed diagnostics are excluded from the score tables and plots.

| Model and scope | New runs | Configuration | Status at 2026-07-12T02:09Z |
|:----------------|---------:|:--------------|:--------------------------------|
| GPT-OSS K=3/5/6/7, three replicates | 12 | 4 engines, TP=2, eager/NCCL, 8 H100s/job | Complete, collected, and plotted |
| GLM K=4/8, two added replicates | 4 | 2 engines, TP=4, 8 H100s/job | Complete, collected, and plotted |
| GLM K=2, three replicates | 3 | 2 engines, TP=4, 8 H100s/job | Complete, collected, and plotted |
| GLM K=1/3/5/7, three replicates | 12 | 2 engines, TP=4, 8 H100s/job | Complete, collected, and plotted; one preemption retried successfully |
| Qwen AIME 2026 | 17 | 30 prompts × 32 samples in each job | Complete and collected for all K |
| GPT-OSS AIME 2026 | 8 | 30 prompts × 32 samples in each job | K=2--8 complete/collected; K=1 invalid |
| GLM AIME 2026 | 3 | 30 prompts × 32 samples in each job | Complete, collected, and plotted |
| Dolci-think SFT100k K=2/4/6/8 pilot | 4 original + 4 replacements | 4 one-GPU engines, all three tasks together | Original runs invalid after native-context crashes; corrected replacements running/scheduled |

The 13 accidentally launched Qwen3 Base AIME experiments were permanently deleted after the scope
was clarified to chat models only. Their launch records remain in the append-only ledger and are
excluded from result collection and plots.

### Dolci-think SFT100k validation

The four `initial-r1-20260712` experiments reached Beaker's `succeeded` state, but none is a valid
evaluation result. Every vLLM engine eventually failed at sequence position 32,768 with a CUDA
index assertion. The checkpoint has `max_position_embeddings=32768` and no RoPE scaling, while
the initial launch advertised a 40,960-token server context. olmo-eval then recorded empty or
missing generations as zero scores instead of failing the Beaker task.

| K | MATH-500 outputs | GPQA outputs | IFEval outputs | Validation |
|---:|-----------------:|-------------:|---------------:|:-----------|
| 2 | 4 / 500 | 2 / 198 | 3 / 300 | Invalid |
| 4 | 118 / 500 | 45 / 198 | 64 / 300 | Invalid |
| 6 | 120 / 500 | 43 / 198 | 64 / 300 | Invalid |
| 8 | 108 / 500 | 46 / 198 | 61 / 300 | Invalid |

These four points are excluded from tables and plots. The first native-context replacements
(`native32k-r1-20260712`) fixed the engine crash but mistakenly retained the Qwen3 reasoning parser.
This checkpoint uses the OLMo thinking chat template, which hardcodes `<think>` into the prompt.
When a response omitted `</think>`, the Qwen3 parser classified the entire response as reasoning and
returned empty content; the OLMo3 parser correctly falls back to treating the raw response as content.
All four Qwen3-parser replacements and their K=4/6 scores are therefore superseded and excluded.

A second replacement sweep uses the same chat tasks, 30,000-token generation cap, native 32,768-token
total context, four one-GPU engines, and `reasoning_parser=olmo3`:

| K | OLMo3-parser experiment | Status at 2026-07-12T16:25Z |
|---:|:------------------------|:-----------------------------|
| 2 | [01KXAA986DGEJB76QXZDH4N9E1](https://beaker.org/ex/01KXAA986DGEJB76QXZDH4N9E1) | Complete, collected, plotted |
| 4 | [01KXAA9GRJJXDV4CH0FT3MBH8D](https://beaker.org/ex/01KXAA9GRJJXDV4CH0FT3MBH8D) | Complete, collected, plotted |
| 6 | [01KXAA9S8J21JTCQCXCJG8W4PW](https://beaker.org/ex/01KXAA9S8J21JTCQCXCJG8W4PW) | Complete, collected, plotted |
| 8 | [01KXAAA1ME9WC283P25QBE0MMV](https://beaker.org/ex/01KXAAA1ME9WC283P25QBE0MMV) | Complete, collected, plotted |

| K | MATH-500 accuracy ↑ | GPQA Diamond accuracy ↑ | IFEval OOD loose prompt accuracy ↑ | Macro average ↑ |
|---:|--------------------:|------------------------:|-----------------------------------:|----------------:|
| 2 | 0.60% | 12.63% | 10.67% | 7.96% |
| 4 | 72.80% | 42.93% | 22.33% | 46.02% |
| 6 | 84.40% | 50.00% | 23.33% | 52.58% |
| 8 | 84.40% | 46.97% | 23.00% | 51.46% |

All four completed runs contain all 500 MATH, 198 GPQA, and 300 IFEval prediction rows and have no
engine failures. One identical 2,769-token GPQA prompt was rejected in every run because the
30,000-token output request exceeded the 32,768-token total context by one token. It remains in the
198-example denominator as incorrect, matching the project scoring convention. At K=2, 843 of 997
generated responses reached the 30,000-token cap; those truncated responses remain in the task
denominators and explain much of the low-budget collapse.

## Qwen3 K=8 expert activation and contribution analysis

The [full 60-response experiment](https://beaker.org/ex/01KXMSXEM2K66D7TM89PQ8XQD3)
replayed 20 saved K=8 trajectories each from GPQA Diamond, IFEval OOD, and MATH-500 through all
48 MoE layers. Rank-8 raw expert-output norm is 58.1% of rank 1, so activation magnitude reinforces
rather than compensates for router rank: the top four contain 63.3% of normalized gate mass but
69.2% of summed weighted-contribution norm. Local cross-boundary swaps are common, but only 9.8%
of all bottom-four/top-four pairs invert their contribution-norm ordering.

The local renormalized K=4 update has cosine 0.919 to the K=8 update and 1.377 times its norm. This
is a fixed-K=8-trajectory counterfactual, not a full K=4 generation result. The detailed methodology,
bootstrap intervals, and caveats are in the
[experiment note](qwen_expert_contribution_experiment.md). Saved artifacts:

- [Rank-wise activation/contribution plot](plots/qwen3_k8_expert_contribution_by_rank.png)
- [Layer-wise local-counterfactual plot](plots/qwen3_k8_expert_contribution_by_layer.png)
- [Machine-readable summary](expert_contribution/qwen3_k8_expert_contribution_summary.json)
- [Per-layer CSV](expert_contribution/qwen3_k8_expert_contribution_by_layer.csv)

## Qwen3 K=8 expert-weight perturbation

The native top-eight expert IDs were held fixed while their normalized weights were either left
unchanged, deterministically shuffled among those experts per token/layer, or replaced with uniform
1/8 weights. Each condition has three complete runs of MATH-500, GPQA Diamond, and IFEval OOD.
Every run contains all 998 expected prediction rows and no task errors. Values are mean ± sample
standard deviation in percentage points.

| Condition | MATH-500 ↑ | GPQA Diamond ↑ | IFEval OOD ↑ | Macro ↑ | Hit token cap |
|:--|--:|--:|--:|--:|--:|
| Native weights | 95.40 ± 0.20 | 64.81 ± 1.62 | 39.78 ± 0.77 | 66.66 ± 0.81 | 1.67 ± 0.29 |
| Shuffled weights | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 99.90 ± 0.10 |
| Uniform weights | 56.27 ± 1.10 | 21.72 ± 2.31 | 10.11 ± 1.95 | 29.36 ± 0.94 | 55.04 ± 0.75 |

Shuffling causes a reproducible generation collapse: 2,991/2,994 outputs reach the 32,768-token
cap. Uniform weighting preserves some capability but reduces the macro score by 37.30 points and
makes 1,648/2,994 outputs reach the cap. Full methodology, run links, and per-run values are in the
[expert-weight perturbation note](qwen_expert_weight_perturbation.md).

## Qwen3 routing-policy interventions

All 40 approved full evaluations completed successfully and are collected. Every run has all 998
predictions and no task errors. The third K=12 reference-scaled experiment was initially canceled
during startup after an unrecoverable GPU Xid 31, then automatically retried successfully.

| Policy family | Condition | Runs | Macro average ↑ |
|:--|:--|--:|--:|
| Weight temperature | `p=0.5` | 3 | 64.75 ± 0.28 |
| Weight temperature | `p=0.75` | 3 | 65.35 ± 0.72 |
| Weight temperature | `p=1.0` identity | 1 | 66.01 |
| Weight temperature | `p=1.5` | 3 | 65.51 ± 0.26 |
| Weight temperature | `p=2.0` | 3 | 50.16 ± 0.88 |
| Fixed global rank profile | — | 3 | 62.50 ± 0.30 |
| Cumulative mass | `tau=0.50` | 3 | 25.43 ± 1.12 |
| Cumulative mass | `tau=0.60` | 3 | 52.88 ± 1.55 |
| Cumulative mass | `tau=0.70` | 3 | 62.74 ± 1.54 |
| Cumulative mass | `tau=0.80` | 3 | 64.27 ± 0.66 |
| Cumulative mass | `tau=0.90` | 3 | 65.86 ± 1.27 |
| K=8-reference scaled | K=4 | 3 | 64.14 ± 0.64 |
| K=8-reference scaled | K=6 | 3 | 64.78 ± 0.25 |
| K=8-reference scaled | K=12 | 3 | 64.89 ± 1.16 |

The cumulative-mass curve is smooth, mild temperature changes are robust, and `p=2` over-sharpening
is consistently harmful. K=4 reference scaling substantially outperforms the earlier renormalized
K=4 result, suggesting that survivor amplification is part of the low-K degradation. Full
per-task metrics, diagnostics, run links, and interpretation are in the
[routing-policy results note](qwen_routing_policy_sweep_results.md).
