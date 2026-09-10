# Qwen3.6 TBLite reference-scaled H100 sweep (2026-08-15)

## Design

- Model: `Qwen/Qwen3.6-35B-A3B`
- Dataset: `openthoughts-tblite@2.0`, all 100 tasks, one attempt per task
- Expert counts: K=4, 6, 8, 10, 12; three matched replicates each
- Seeds: 4202, 4203, and 4204, paired across expert counts
- Routing: K=4/6 retain the corresponding prefix of native top eight using
  K=8 as reference; K=8 is native; K=10/12 route the requested K and use the
  top-eight prefix as reference. No retained-prefix renormalization.
- Agent: Vanillux2 with TMax-default timeout and step settings
- Generation: temperature 1.0, top-p 0.95, top-k 20, 81,920-token output
  ceiling, 262,144-token model context
- Serving: vLLM 0.19.1, `qwen3_xml` tool parser, `qwen3` reasoning parser,
  language-model-only, GDN Triton prefill
- Topology: one 8-GPU H100 node per evaluation, TP=2, DP=4, four concurrent
  tasks, one sequence per engine
- Beaker: `ai2/OLMo-3-moe-experiments`, urgent, allocated; only
  `ai2/jupiter` and `ai2/ceres` are eligible

The completed Titan K=8 diagnostic `01M01EPN6H5R9PK15ESEKC2G10` is retained
as separate provenance and is not counted as one of these matched H100 runs.

## Jobs

| K | Replicate | Seed | Beaker experiment | Status |
|---:|---:|---:|---|---|
| 4 | 1 | 4202 | [01M01Y584PDEY0EKXF7T0EA7J8](https://beaker.org/ex/01M01Y584PDEY0EKXF7T0EA7J8) | Completed: reward 0.581181, pass@1 0.60 |
| 4 | 2 | 4203 | [01M01Y5CK8HZ8XM4T5SRMX6FT3](https://beaker.org/ex/01M01Y5CK8HZ8XM4T5SRMX6FT3) | Completed: reward 0.579621, pass@1 0.59 |
| 4 | 3 | 4204 | [01M01Y5G570WN8C8379Z72S877](https://beaker.org/ex/01M01Y5G570WN8C8379Z72S877) | Completed: reward 0.599395, pass@1 0.61 |
| 6 | 1 | 4202 | [01M01Y5MCRNG5VA6GKR8ABYTJP](https://beaker.org/ex/01M01Y5MCRNG5VA6GKR8ABYTJP) | Completed: reward 0.619419, pass@1 0.63 |
| 6 | 2 | 4203 | [01M01Y5RJHTGVDBTKGF6Y0V9B0](https://beaker.org/ex/01M01Y5RJHTGVDBTKGF6Y0V9B0) | Completed: reward 0.626738, pass@1 0.64 |
| 6 | 3 | 4204 | [01M01Y5W9PC81EZB05JXY7BYKG](https://beaker.org/ex/01M01Y5W9PC81EZB05JXY7BYKG) | Canceled after a host-port collision prevented vLLM readiness; superseded |
| 6 | 3 rerun | 4204 | [01M022YFMWKZAHPGR3N5HNJQJ4](https://beaker.org/ex/01M022YFMWKZAHPGR3N5HNJQJ4) | Completed: reward 0.581459, pass@1 0.59 |
| 8 | 1 | 4202 | [01M01Y608CWSZM5D0ZMH2BS81D](https://beaker.org/ex/01M01Y608CWSZM5D0ZMH2BS81D) | Completed: reward 0.631828, pass@1 0.65 |
| 8 | 2 | 4203 | [01M01Y6486ZQ2NRBWDNS6K0E9V](https://beaker.org/ex/01M01Y6486ZQ2NRBWDNS6K0E9V) | Completed: reward 0.605709, pass@1 0.62 |
| 8 | 3 | 4204 | [01M01Y681KG75HZ671NPZ01C08](https://beaker.org/ex/01M01Y681KG75HZ671NPZ01C08) | Completed: reward 0.609353, pass@1 0.62 |
| 10 | 1 | 4202 | [01M01Y6BT8C9MMYBWR048AEPTK](https://beaker.org/ex/01M01Y6BT8C9MMYBWR048AEPTK) | Failed before eval: shallow HF override did not change the nested text router |
| 10 | 2 | 4203 | [01M01Y6FBCFM2MR4D7PKCWKQYY](https://beaker.org/ex/01M01Y6FBCFM2MR4D7PKCWKQYY) | Failed before eval: shallow HF override did not change the nested text router |
| 10 | 3 | 4204 | [01M01Y6JTNCTX7KQTK3APMX1VE](https://beaker.org/ex/01M01Y6JTNCTX7KQTK3APMX1VE) | Failed before eval: shallow HF override did not change the nested text router |
| 12 | 1 | 4202 | [01M01Y6PAKNSJ03F0CNS49NJ8T](https://beaker.org/ex/01M01Y6PAKNSJ03F0CNS49NJ8T) | Failed before eval: shallow HF override did not change the nested text router |
| 12 | 2 | 4203 | [01M01Y6SZKW8WW3NP32Q23JWJD](https://beaker.org/ex/01M01Y6SZKW8WW3NP32Q23JWJD) | Failed before eval: shallow HF override did not change the nested text router |
| 12 | 3 | 4204 | [01M01Y6XMWJPN6CQT3Z39SZJ48](https://beaker.org/ex/01M01Y6XMWJPN6CQT3Z39SZJ48) | Failed before eval: shallow HF override did not change the nested text router |
| 10 | 1 nested-fix rerun | 4202 | [01M037FM56KKC2CB992JH85XHK](https://beaker.org/ex/01M037FM56KKC2CB992JH85XHK) | Running; exact-router assertion enabled |
| 10 | 2 nested-fix rerun | 4203 | [01M037FQMJBCMZD52EVFACMFDB](https://beaker.org/ex/01M037FQMJBCMZD52EVFACMFDB) | Running; exact-router assertion enabled |
| 10 | 3 nested-fix rerun | 4204 | [01M037FV25J52NME05HCJYY75D](https://beaker.org/ex/01M037FV25J52NME05HCJYY75D) | Running; exact-router assertion enabled |
| 12 | 1 nested-fix rerun | 4202 | [01M037FYJZCZR8TEW7CZHCFFGF](https://beaker.org/ex/01M037FYJZCZR8TEW7CZHCFFGF) | Running; exact-router assertion enabled |
| 12 | 2 nested-fix rerun | 4203 | [01M037G21BAJA0G4PHVC0V89V9](https://beaker.org/ex/01M037G21BAJA0G4PHVC0V89V9) | Running; exact-router assertion enabled |
| 12 | 3 nested-fix rerun | 4204 | [01M037G5H0V0KQ7VCTT4B7K7XC](https://beaker.org/ex/01M037G5H0V0KQ7VCTT4B7K7XC) | Running; exact-router assertion enabled |

Initial audit at `2026-08-15T05:29:10Z`: all 15 jobs are queued, each requests
eight GPUs, and every cluster constraint is exactly `ai2/jupiter,ai2/ceres`.

## K>8 correction

Qwen3.6 is a composite `Qwen3_5MoeForConditionalGeneration` config. Its real
router width is `text_config.num_experts_per_tok`; there is no corresponding
outer field. The original K=10/12 commands used the shallow outer override
`{"num_experts_per_tok": K}`. vLLM accepted that option but left the text
router at native K=8. The current startup assertion caught the mismatch and
stopped these six runs before any benchmark task was evaluated.

The replacement commands use
`{"text_config":{"num_experts_per_tok": K}}` and retain the startup
assertion. A two-GPU K=10 routing smoke is tracked as
[01M037AYMR4KMNHZBEPW907TKP](https://beaker.org/ex/01M037AYMR4KMNHZBEPW907TKP).
