# Qwen3.6 TB2.1 reference-scaled baseline sweep (2026-08-15)

## Design

- Model: `Qwen/Qwen3.6-35B-A3B`
- Dataset: pinned Terminal-Bench 2.1 revision, all 89 tasks
- Expert counts: K=4, 6, 8, 10, 12; three matched runs per K. Replicates
  1/2/3 use seeds 4202/4203/4204.
- Protocol: identical to the 2026-08-15 Qwen3.6 TBLite sweep except for the dataset
- Routing: reference-scaled to native K=8 for every non-native expert count
- Agent: Vanillux2 with `qwen3_xml` tool parsing and `qwen3` reasoning parsing
- Generation: temperature 1.0, top-p 0.95, top-k 20, 81,920-token output
  ceiling, 262,144-token model context
- Timeouts/resources: task-native TMax defaults, with no timeout override
- Topology: one 8-GPU H100 node per K, TP=2, DP=4, four concurrent tasks
- Beaker: `ai2/OLMo-3-moe-experiments`, urgent, allocated; only
  `ai2/jupiter` and `ai2/ceres` are eligible

## Jobs

| K | Seed | Beaker experiment | Status |
|---:|---:|---|---|
| 4 | 4202 | [01M01Z9N9PQD1YZNY87KXTVA9W](https://beaker.org/ex/01M01Z9N9PQD1YZNY87KXTVA9W) | Completed: 23/89, pass@1 0.258427 |
| 6 | 4202 | [01M01Z9RZZ6RNP67QE2Z6DNVP5](https://beaker.org/ex/01M01Z9RZZ6RNP67QE2Z6DNVP5) | Completed: 30/89, pass@1 0.337079 |
| 8 | 4202 | [01M01Z9WMZB5MN1CDW0RJTPGF7](https://beaker.org/ex/01M01Z9WMZB5MN1CDW0RJTPGF7) | Completed: 29/89, pass@1 0.325843 |
| 10 | 4202 | [01M01ZA0N8MKZFGQ6B3BN0XMEH](https://beaker.org/ex/01M01ZA0N8MKZFGQ6B3BN0XMEH) | Failed before eval: shallow HF override did not change the nested text router |
| 12 | 4202 | [01M01ZA4C1W75HCSE1SHEWNMWB](https://beaker.org/ex/01M01ZA4C1W75HCSE1SHEWNMWB) | Failed before eval: shallow HF override did not change the nested text router |
| 10 | 4202 | [01M037GE22J067CWE9C8XGM334](https://beaker.org/ex/01M037GE22J067CWE9C8XGM334) | Running; exact-router assertion enabled |
| 12 | 4202 | [01M037GHJNZATGZFQ6JMEFGG3T](https://beaker.org/ex/01M037GHJNZATGZFQ6JMEFGG3T) | Running; exact-router assertion enabled |

Initial audit at `2026-08-15T05:46:43Z`: all five jobs are queued, each
requests eight GPUs, every cluster constraint is exactly
`ai2/jupiter,ai2/ceres`, and every job points to the pinned TB2.1 revision.

The K=10/12 replacements correct Qwen3.6's HF override from the nonexistent
outer `num_experts_per_tok` field to
`text_config.num_experts_per_tok`. The startup assertion remains enabled and
will reject any run that does not match a real router layer at its requested K.

## Replicate expansion

Launched at `2026-08-15T23:32Z`. These ten jobs add replicates 2 and 3 to
every corrected K point. All were audited from their live Beaker specs: eight
GPUs, TP=2, DP=4, urgent allocated scheduling, and exactly Jupiter/Ceres.

| K | Replicate | Seed | Beaker experiment | Launch status |
|---:|---:|---:|---|---|
| 4 | 2 | 4203 | [01M03W96M34YG48BBMBAM9TXZR](https://beaker.org/ex/01M03W96M34YG48BBMBAM9TXZR) | Submitted |
| 4 | 3 | 4204 | [01M03W9AFM4MX7T8F2CSN5QX7Q](https://beaker.org/ex/01M03W9AFM4MX7T8F2CSN5QX7Q) | Submitted |
| 6 | 2 | 4203 | [01M03W9E4RHM5X0F0JH54R0QGP](https://beaker.org/ex/01M03W9E4RHM5X0F0JH54R0QGP) | Submitted |
| 6 | 3 | 4204 | [01M03W9JHPS1C7W6JF7YB5JAHF](https://beaker.org/ex/01M03W9JHPS1C7W6JF7YB5JAHF) | Submitted |
| 8 | 2 | 4203 | [01M03W9PA8EYH788PHC7061ZAW](https://beaker.org/ex/01M03W9PA8EYH788PHC7061ZAW) | Submitted |
| 8 | 3 | 4204 | [01M03W9T1YHEFKT682NF1VFKTB](https://beaker.org/ex/01M03W9T1YHEFKT682NF1VFKTB) | Submitted |
| 10 | 2 | 4203 | [01M03W9XFCGNT2MV1V1A6APGQE](https://beaker.org/ex/01M03W9XFCGNT2MV1V1A6APGQE) | Submitted |
| 10 | 3 | 4204 | [01M03WA130SBGXB8PXVFPJWQ3V](https://beaker.org/ex/01M03WA130SBGXB8PXVFPJWQ3V) | Submitted |
| 12 | 2 | 4203 | [01M03WA4HCJBFE3AZ61CHXMWGW](https://beaker.org/ex/01M03WA4HCJBFE3AZ61CHXMWGW) | Submitted |
| 12 | 3 | 4204 | [01M03WACY18JT87SVWEN7PBA9X](https://beaker.org/ex/01M03WACY18JT87SVWEN7PBA9X) | Submitted |
