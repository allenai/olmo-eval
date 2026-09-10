# Qwen3.6 K>8 routing correction and replacement jobs

Updated: 2026-08-15

## Why these runs were replaced

Qwen3.6 stores its routed-expert count at `text_config.num_experts_per_tok`.
Historical nominal K=10 and K=12 jobs set a shallow outer field, so vLLM kept
routing the native eight experts. Those historical points are invalid as K>8
measurements and are excluded from every maintained plot and CSV.

Every replacement uses the nested override, for example
`{"text_config":{"num_experts_per_tok":10}}`, plus a startup assertion that
fails unless a real Qwen router layer is patched at the requested K. Weights
are reference-scaled to K=8 and are not renormalized.

## Smoke gate

| Path | K | Beaker experiment | Result |
|---|---:|---|---|
| TMax | 10 | [01M037AYMR4KMNHZBEPW907TKP](https://beaker.org/ex/01M037AYMR4KMNHZBEPW907TKP) | Exit 0; `router_k=10` |
| TMax | 12 | [01M039BQ52BC75NH8W7SWWH183](https://beaker.org/ex/01M039BQ52BC75NH8W7SWWH183) | Exit 0; `router_k=12` |
| olmo-eval | 10 | [01M039MS9QAVZ6W8GRSC2X7954](https://beaker.org/ex/01M039MS9QAVZ6W8GRSC2X7954) | Exit 0; matched router layer at `top_k=10` |
| olmo-eval | 12 | [01M039NA416XFAYKE849XXZRFJ](https://beaker.org/ex/01M039NA416XFAYKE849XXZRFJ) | Exit 0; matched router layer at `top_k=12` |

## Corrected capability suite

Group: [qwen36-capability-kgt8-nestedfix1-20260815](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01M03A08C7SAKMBZA0P5G08W4Y)

Each job bundles MATH-500, GPQA Diamond, AIME 2025 pass@32, and the
three-task IFBench aggregate. It uses four TP=1 engines, a 32,768-token output
cap, seeds 42/43/44, the original Qwen thinking settings, `ai2/holmes-testing`,
the `80g` selector, and urgent priority.

| K | Replicate | Seed | Beaker experiment |
|---:|---:|---:|---|
| 10 | 1 | 42 | [01M03A0CF4K0K70QC1BV9RQBCX](https://beaker.org/ex/01M03A0CF4K0K70QC1BV9RQBCX) |
| 10 | 2 | 43 | [01M03A0MQQJ2EG3JMGHVB74HYQ](https://beaker.org/ex/01M03A0MQQJ2EG3JMGHVB74HYQ) |
| 10 | 3 | 44 | [01M03A0WXQHVXXNFH8EAJTGXS6](https://beaker.org/ex/01M03A0WXQHVXXNFH8EAJTGXS6) |
| 12 | 1 | 42 | [01M03A15GQ2FEDAVBX833VWFV4](https://beaker.org/ex/01M03A15GQ2FEDAVBX833VWFV4) |
| 12 | 2 | 43 | [01M03A1EN23HQ983N8T5B7VWXW](https://beaker.org/ex/01M03A1EN23HQ983N8T5B7VWXW) |
| 12 | 3 | 44 | [01M03A1RVQVFJE0WR3F7AQ3Z3F](https://beaker.org/ex/01M03A1RVQVFJE0WR3F7AQ3Z3F) |

## Corrected legacy Terminal-Bench 2.0 sweep

These jobs reproduce the old protocol except for the fixed nested override and
a new output namespace. Each logical replicate is split across complementary
A/B shards: five replicates per K, ten eight-H100 jobs per K. Settings remain
TP=2, DP=4, four concurrent agents, 32,768 output tokens, 262,144 context,
temperature 0.6, top-p 0.95, top-k 20, Qwen3-coder tool parsing, Qwen3
reasoning parsing, native task timeouts, urgent and unallocated on
`ai2/jupiter` in `ai2/olmo-instruct`.

Results root:
`/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-35b-a3b-kgt8-nestedfix1-20260815`

| K | Rep | Seed | Shard A | Shard B |
|---:|---:|---:|---|---|
| 10 | 1 | unset | [01M03A2XXVQMJNMYASEATFZ2N0](https://beaker.org/ex/01M03A2XXVQMJNMYASEATFZ2N0) | [01M03A322RPMN0FV61RBZWMT57](https://beaker.org/ex/01M03A322RPMN0FV61RBZWMT57) |
| 10 | 2 | 4202 | [01M03A36239Y5PHXPW1R0VEV3H](https://beaker.org/ex/01M03A36239Y5PHXPW1R0VEV3H) | [01M03A3A4DA94WC2G9CYAMS0E7](https://beaker.org/ex/01M03A3A4DA94WC2G9CYAMS0E7) |
| 10 | 3 | 4203 | [01M03A3EFZXG6X1K2MFJG6MD5X](https://beaker.org/ex/01M03A3EFZXG6X1K2MFJG6MD5X) | [01M03A3JRD9TEJA9MRR20VG3X9](https://beaker.org/ex/01M03A3JRD9TEJA9MRR20VG3X9) |
| 10 | 4 | 4204 | [01M03A3PQW31Q5GRZFC28F99NX](https://beaker.org/ex/01M03A3PQW31Q5GRZFC28F99NX) | [01M03A3TJ4V8ESCG28XCPAMJCW](https://beaker.org/ex/01M03A3TJ4V8ESCG28XCPAMJCW) |
| 10 | 5 | 4205 | [01M03A3YGCRKBPKHEEB0WRJE43](https://beaker.org/ex/01M03A3YGCRKBPKHEEB0WRJE43) | [01M03A42HA04Y1DHDV1V1WV4ZW](https://beaker.org/ex/01M03A42HA04Y1DHDV1V1WV4ZW) |
| 12 | 1 | unset | [01M03A8J2E28RRTNER96JF7ZDZ](https://beaker.org/ex/01M03A8J2E28RRTNER96JF7ZDZ) | [01M03A8TZQPJG9J3BKG1E3TJCV](https://beaker.org/ex/01M03A8TZQPJG9J3BKG1E3TJCV) |
| 12 | 2 | 4202 | [01M03A8ZSFC3JQ59QJ47R2R1GY](https://beaker.org/ex/01M03A8ZSFC3JQ59QJ47R2R1GY) | [01M03A9413R9MD0FACEQW4B50J](https://beaker.org/ex/01M03A9413R9MD0FACEQW4B50J) |
| 12 | 3 | 4203 | [01M03A97XE74VGZ5FQ280SBPP8](https://beaker.org/ex/01M03A97XE74VGZ5FQ280SBPP8) | [01M03A9BPW8ZTS62G65JQTNZQT](https://beaker.org/ex/01M03A9BPW8ZTS62G65JQTNZQT) |
| 12 | 4 | 4204 | [01M03A9FNBD8Y533NY0VYQE5C3](https://beaker.org/ex/01M03A9FNBD8Y533NY0VYQE5C3) | [01M03A9KFQANEM1TAAE8FTJY61](https://beaker.org/ex/01M03A9KFQANEM1TAAE8FTJY61) |
| 12 | 5 | 4205 | [01M03A9QG6CDSWRZHB0RZ3EXZB](https://beaker.org/ex/01M03A9QG6CDSWRZHB0RZ3EXZB) | [01M03A9VKAHH6MP547GXHJ8K4R](https://beaker.org/ex/01M03A9VKAHH6MP547GXHJ8K4R) |

## Corrected TBLite and Terminal-Bench 2.1 jobs

These were submitted immediately after the audit with the same fail-closed
startup assertion. Live K=10 and K=12 jobs have emitted the expected routing
validation marker, so this corrected matrix was retained rather than duplicated.

| Benchmark | K | Replicate/seed | Beaker experiment |
|---|---:|---|---|
| TBLite | 10 | 1 / 4202 | [01M037FM56KKC2CB992JH85XHK](https://beaker.org/ex/01M037FM56KKC2CB992JH85XHK) |
| TBLite | 10 | 2 / 4203 | [01M037FQMJBCMZD52EVFACMFDB](https://beaker.org/ex/01M037FQMJBCMZD52EVFACMFDB) |
| TBLite | 10 | 3 / 4204 | [01M037FV25J52NME05HCJYY75D](https://beaker.org/ex/01M037FV25J52NME05HCJYY75D) |
| TBLite | 12 | 1 / 4202 | [01M037FYJZCZR8TEW7CZHCFFGF](https://beaker.org/ex/01M037FYJZCZR8TEW7CZHCFFGF) |
| TBLite | 12 | 2 / 4203 | [01M037G21BAJA0G4PHVC0V89V9](https://beaker.org/ex/01M037G21BAJA0G4PHVC0V89V9) |
| TBLite | 12 | 3 / 4204 | [01M037G5H0V0KQ7VCTT4B7K7XC](https://beaker.org/ex/01M037G5H0V0KQ7VCTT4B7K7XC) |
| TB2.1 | 10 | 1 / 4202 | [01M037GE22J067CWE9C8XGM334](https://beaker.org/ex/01M037GE22J067CWE9C8XGM334) |
| TB2.1 | 12 | 1 / 4202 | [01M037GHJNZATGZFQ6JMEFGG3T](https://beaker.org/ex/01M037GHJNZATGZFQ6JMEFGG3T) |

## Plot inclusion rule

- The capability plot reads historical K=4/6/8 plus only corrected
  `nestedfix1-20260815` K=10/12 bundles.
- The old TB2.0 plot reads historical K=4/6/8 plus only the new corrected root
  for K=10/12.
- TBLite and TB2.1 summaries ignore historical invalid K=10/12 directories and
  admit only corrected experiment IDs.

The append-only machine-readable source of record is `notes/beaker_jobs.jsonl`.
