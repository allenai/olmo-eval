# Qwen3.6 expanded TBLite expert sweep (2026-08-16)

## Design

- Model: `Qwen/Qwen3.6-35B-A3B`
- Benchmark: all 100 tasks from `openthoughts-tblite@2.0`
- Expert counts: K=2, 14, 16, and 32
- Replicates: three per K, using paired seeds 4202, 4203, and 4204
- Routing: K=2 retains the top two of the native top eight; K>8 changes
  `text_config.num_experts_per_tok` to the requested K. Every condition is
  reference-scaled to the native K=8 mass and is not renormalized.
- Protocol: identical to the corrected 2026-08-15 TBLite sweep: Vanillux2,
  TMax defaults, Qwen thinking generation settings, TP=2, DP=4, and one
  allocated eight-H100 node per replicate.
- Scheduling: urgent in `ai2/OLMo-3-moe-experiments`, eligible on
  `ai2/jupiter` and `ai2/ceres`.

Every job has a fail-closed startup assertion. K=2 must report
`router_k=8, keep_k=2, reference_k=8`; K=14/16/32 must report matching
`router_k=keep_k=K` after applying the nested text-config override.

## Routing smokes

| K | Beaker experiment | Expected validation |
|---:|---|---|
| 2 | [01M04EZWBA0Z1DA7D2R77XAW4S](https://beaker.org/ex/01M04EZWBA0Z1DA7D2R77XAW4S) | `router_k=8, keep_k=2` |
| 32 | [01M04F04PCHWSGZ8KFQF0NVJMX](https://beaker.org/ex/01M04F04PCHWSGZ8KFQF0NVJMX) | `router_k=32, keep_k=32` |

## Full sweep

| K | Replicate | Seed | Beaker experiment |
|---:|---:|---:|---|
| 2 | 1 | 4202 | [01M04F0XDX8XNRYW27D103Q06D](https://beaker.org/ex/01M04F0XDX8XNRYW27D103Q06D) |
| 2 | 2 | 4203 | [01M04F143P888RK0FB02K14YBP](https://beaker.org/ex/01M04F143P888RK0FB02K14YBP) |
| 2 | 3 | 4204 | [01M04F17V1NJ8YVSSSQ0EPRX78](https://beaker.org/ex/01M04F17V1NJ8YVSSSQ0EPRX78) |
| 14 | 1 | 4202 | [01M04F1BJA5PY4Q5M8A03271P0](https://beaker.org/ex/01M04F1BJA5PY4Q5M8A03271P0) |
| 14 | 2 | 4203 | [01M04F1FFE56QGJJ2PDNY7DWB1](https://beaker.org/ex/01M04F1FFE56QGJJ2PDNY7DWB1) |
| 14 | 3 | 4204 | [01M04F1K2WFH0P9Q92TR94ZHJE](https://beaker.org/ex/01M04F1K2WFH0P9Q92TR94ZHJE) |
| 16 | 1 | 4202 | [01M04F1PPVX9KDKZNNFEVA1FYN](https://beaker.org/ex/01M04F1PPVX9KDKZNNFEVA1FYN) |
| 16 | 2 | 4203 | [01M04F1XXTSHHX199992AT6MS4](https://beaker.org/ex/01M04F1XXTSHHX199992AT6MS4) |
| 16 | 3 | 4204 | [01M04F21VDDAW1X5KJP13EXNTK](https://beaker.org/ex/01M04F21VDDAW1X5KJP13EXNTK) |
| 32 | 1 | 4202 | [01M04F25HCSTG2WPGYJ6A508XH](https://beaker.org/ex/01M04F25HCSTG2WPGYJ6A508XH) |
| 32 | 2 | 4203 | [01M04F290F2G10FTAFZS6YXGA2](https://beaker.org/ex/01M04F290F2G10FTAFZS6YXGA2) |
| 32 | 3 | 4204 | [01M04F2CMKTKQ7ZE5SG7WKK5GJ](https://beaker.org/ex/01M04F2CMKTKQ7ZE5SG7WKK5GJ) |

Results root:
`/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-tblite-expanded-reference-sweep-20260816`

## 2026-08-16 17:04 UTC status

Both routing smokes succeeded. The K=2 smoke reported
`router_k=8, keep_k=2, reference_k=8, renormalize=False`; the K=32 smoke
reported `router_k=32, keep_k=32, reference_k=8, renormalize=False`.

Six full runs have succeeded: K=2 replicate 1, all three K=14 runs, and K=16
replicates 1 and 2. Six remain healthy and running: K=2 replicates 2/3, K=16
replicate 3, and all three K=32 runs. There are no job-level failures.

Complete-run mean reward is currently 27.23% at K=2 (n=1), 64.05% at K=14
(n=3), and 65.54% at K=16 (n=2). Incomplete runs are excluded from these
numbers and from the maintained plot.
