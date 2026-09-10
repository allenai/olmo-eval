# OpenThoughts-Agent SFT reference-scaled TBLite evaluations (2026-08-15)

This tracker covers the SFT checkpoints trained with K=4 and K=12 using K=8
as the router-weight reference scale. The earlier corrected K=8 run is tracked
in `tmax_clean_k8_tblite_jobs.md`.

## Shared settings

- Dataset: `openthoughts-tblite@2.0` (100 tasks in each full run)
- Agent: Terminus-2 with JSON action parser
- Serving: vLLM 0.19.1, `qwen3_xml` tool parser, `qwen3` reasoning parser,
  `--language-model-only`, 262,144-token context, 81,920-token output ceiling
- Sampling: temperature 1.0, top-p 0.95, top-k 20
- Parallelism: 8 GPUs, TP=2, DP=4, four concurrent tasks, one active sequence
  per engine
- Beaker: `ai2/OLMo-3-moe-experiments`, urgent, allocated; Jupiter/Ceres/Titan
- K=4 routing: route 8, retain 4, reference K=8, no renormalization
- K=12 routing: route and retain 12, reference K=8, no renormalization

## Jobs

| Train/eval K | Stage | Beaker experiment | Status | Validation |
|---:|---|---|---|---|
| 4 | 1-task smoke | [01M01QGVD92Q21HY738N03CTE5](https://beaker.org/ex/01M01QGVD92Q21HY738N03CTE5) | canceled after validation | Server ready; Harbor started; exact Terminus JSON probe passed |
| 12 | 1-task smoke | [01M01QGVQ6K6ZWWE6MH9G0H0TR](https://beaker.org/ex/01M01QGVQ6K6ZWWE6MH9G0H0TR) | canceled after validation | Server ready; Harbor started; exact Terminus JSON probe passed |
| 4 | Full attempt 1 | [01M01R7C9KXDGE8PC2MYSFBKDN](https://beaker.org/ex/01M01R7C9KXDGE8PC2MYSFBKDN) | failed before evaluation | Assertion exposed shallow HF override: patch installed but matched no top-8 layer |
| 12 | Full attempt 1 | [01M01R7C1XHB7QM6FNQSBPDVBZ](https://beaker.org/ex/01M01R7C1XHB7QM6FNQSBPDVBZ) | failed before evaluation | Assertion itself compared lowercase `false` with logged Python `False` |
| 4 | Routing smoke 2 | [01M01S1G9TAGA0TXSXM0DHEMN6](https://beaker.org/ex/01M01S1G9TAGA0TXSXM0DHEMN6) | canceled by node healthcheck before start | Infra-only failure; no evaluation ran |
| 4 | Routing smoke 3 | [01M01S7SY9NGX9DPC2KFX3DGV4](https://beaker.org/ex/01M01S7SY9NGX9DPC2KFX3DGV4) | validated, then intentionally canceled | Exact assertion passed: router K=8, keep K=4, reference K=8, renormalize=False; Harbor started |
| 4 | Full TBLite run 3 | [01M01SPVGH66S3GR7PAZZBBTHF](https://beaker.org/ex/01M01SPVGH66S3GR7PAZZBBTHF) | running (18/100 at 05:16 UTC) | Exact policy is startup-asserted; partial mean 0.413 with four errors |
| 12 | Full TBLite run 2 | [01M01S1G4W3ABPVYQHTJJ2E3EW](https://beaker.org/ex/01M01S1G4W3ABPVYQHTJJ2E3EW) | running (32/100 at 05:17 UTC) | Exact assertion passed; partial mean 0.371 with six errors |
