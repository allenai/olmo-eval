# Qwen3.6 TB2.1 expanded reference-scaled sweep (2026-08-17)

## Configuration

- Model: `Qwen/Qwen3.6-35B-A3B`
- Expert counts: K={14,16,32,64}, three runs per K
- Routing: router K = keep K = requested K; reference K=8; no unit-sum
  renormalization; nested `text_config.num_experts_per_tok` override applied
- Dataset: pinned 89-task Terminal-Bench 2.1 revision
  `sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a`
- Generation: temperature 1.0, top-p 0.95, top-k 20, 81,920 output-token
  ceiling, 262,144-token context
- Agent: `Vanillux2Agent:Vanillux2Agent`, `qwen3_xml` tool parser, `qwen3`
  reasoning parser
- Topology: one eight-H100 node per run, TP=2, DP=4, four concurrent tasks
- Beaker: `ai2/OLMo-3-moe-experiments`, urgent, allocated, targeting
  `ai2/jupiter` and `ai2/ceres`
- Seeds: replicate 1/2/3 use 4202/4203/4204

## Jobs

| K | Replicate | Beaker experiment | Job ID | Initial status |
|---:|---:|---|---|---|
| 14 | 1 | [01M06V00FPS76CN0FC46NBEPDP](https://beaker.org/ex/01M06V00FPS76CN0FC46NBEPDP) | `01M06V00K9EZB2X6BD00AVV9M9` | queued |
| 14 | 2 | [01M06V048GHPYCDHH1R0A7WTX4](https://beaker.org/ex/01M06V048GHPYCDHH1R0A7WTX4) | `01M06V04DN7WZPRBFC6YC8B8N4` | queued |
| 14 | 3 | [01M06V08MWVEYJF052J2SM2G9A](https://beaker.org/ex/01M06V08MWVEYJF052J2SM2G9A) | `01M06V08S04S2Y0X8WWGB2ZH61` | queued |
| 16 | 1 | [01M06V0F31MKN2BQBT00H9JP9Z](https://beaker.org/ex/01M06V0F31MKN2BQBT00H9JP9Z) | `01M06V0F6DZMV8N17HN4XZA0K3` | queued |
| 16 | 2 | [01M06V0K67R9RMAMNBNF927G5Y](https://beaker.org/ex/01M06V0K67R9RMAMNBNF927G5Y) | `01M06V0K9PN5CM00G1QA9RFNXM` | queued |
| 16 | 3 | [01M06V0PV88R5YVVQ607PHSP5D](https://beaker.org/ex/01M06V0PV88R5YVVQ607PHSP5D) | `01M06V0PYVYM5M8E475N7MF47B` | queued |
| 32 | 1 | [01M06V0TQAH2430T9FKWQSPVK9](https://beaker.org/ex/01M06V0TQAH2430T9FKWQSPVK9) | `01M06V0TV1YSFWWE8JXF479ZJJ` | queued |
| 32 | 2 | [01M06V0YENAFFYH8TBKMYW6PNJ](https://beaker.org/ex/01M06V0YENAFFYH8TBKMYW6PNJ) | `01M06V0YJ6K1M16T3T8B5B1HVC` | queued |
| 32 | 3 | [01M06V16M19314PM990ZABH5MF](https://beaker.org/ex/01M06V16M19314PM990ZABH5MF) | `01M06V16RK9XQJXVWXVJXP9FRP` | queued |
| 64 | 1 | [01M06V1C9WRKX76PGP0GRN9ZPE](https://beaker.org/ex/01M06V1C9WRKX76PGP0GRN9ZPE) | `01M06V1CDD8QVKX63QCHWXYJRX` | queued |
| 64 | 2 | [01M06V1G5DPAY71RPHJ5N8A76W](https://beaker.org/ex/01M06V1G5DPAY71RPHJ5N8A76W) | `01M06V1G8X241QZ0AYPBNMF84F` | queued |
| 64 | 3 | [01M06V1M0Y517X79BNMJBRMY3C](https://beaker.org/ex/01M06V1M0Y517X79BNMJBRMY3C) | `01M06V1M4BDENK0V3M84SX5HFE` | queued |

All twelve submitted specs were checked after launch. Their keep K, router K,
reference K, renormalization flag, and nested model override match the table
above.

## Status at 2026-08-17 06:40 UTC

- Running: K=14 replicates 1/2/3 and K=16 replicate 1.
- Completed trials: 20/89, 17/89, 14/89, and 11/89, respectively.
- Queued: K=16 replicates 2/3 and every K=32 and K=64 run (eight jobs).
- No failures; all four running jobs are actively producing agent steps.

## Final results collected 2026-08-17

All twelve jobs succeeded with complete 89-task metric files.

| K | TB2.1 pass@1 (mean ± sample SD) | Mean task errors/run |
|---:|---:|---:|
| 14 | 31.84% ± 5.31 | 30.7 |
| 16 | 32.96% ± 1.72 | 25.0 |
| 32 | 33.71% ± 2.97 | 33.3 |
| 64 | 26.22% ± 0.65 | 44.3 |

K=14--32 remains comparable to the completed native-K region, while K=64
shows a clear and consistent degradation with substantially more task errors.
