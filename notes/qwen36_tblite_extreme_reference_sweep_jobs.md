# Qwen3.6 extreme above-native-K TBLite sweep

Last updated: 2026-08-16

## Design

- Model: `Qwen/Qwen3.6-35B-A3B`
- Benchmark: `openthoughts-tblite@2.0`
- New expert counts: K=64, 128, and 256
- Routing: widen Qwen3.6's nested `text_config.num_experts_per_tok` to K,
  retain all K selected experts, and reference-scale their weights to the
  first-eight mass without renormalization.
- Serving/eval protocol: unchanged from the corrected TBLite sweep: one
  allocated eight-H100 node, TP=2/DP=4, four vLLM engines, one sequence per
  engine, 262,144 context, 81,920 output ceiling, temperature 1.0, top-p 0.95,
  top-k 20, `qwen3_xml`, `qwen3`, and Vanillux2.
- Scheduling: urgent in `ai2/OLMo-3-moe-experiments`, eligible on Jupiter and
  Ceres.

Every K is gated on a one-task smoke. Startup must fail closed unless both the
nested config and live patch report `router_k=keep_k=K`, `reference_k=8`,
`renormalize=False`, and 256 global routed experts. The scheduling smoke uses
one TP=2 engine and must execute real TBLite agent/model/tool turns without a
serving error. Passing values expand to three full 100-task replicates using
seeds 4202/4203/4204.

## Routing/evaluation smokes

| K | Beaker experiment | Topology | Status |
|---:|---|---|---|
| 64 | [01M061DN08BWTD82QW4RZS09QE](https://beaker.org/ex/01M061DN08BWTD82QW4RZS09QE) | TP=2/DP=1, 2 H100s | Passed; exact K=64 assertion, 39 agent steps, task reward 1, exit 0 |
| 128 | [01M061DS1BJ5BPMZWGRTF3HWQP](https://beaker.org/ex/01M061DS1BJ5BPMZWGRTF3HWQP) | TP=2/DP=1, 2 H100s | Serving passed; exact K=128 assertion and 57 successful agent steps; canceled after the next action ran >4 min to free the node |
| 256 | [01M061DWSCN3KQ1R2G29291GDC](https://beaker.org/ex/01M061DWSCN3KQ1R2G29291GDC) | TP=2/DP=1, 2 H100s | Passed; exact K=256 assertion, all 64 agent steps, task reward 0, exit 0 |

The initial eight-GPU smoke submissions were canceled before scheduling after
four minutes unassigned. The replacement gate uses one TP=2 engine; this is
the same model, fused-MoE kernel, router, and reference-scaling path as each
production engine, without three redundant DP replicas.

Smoke results root:
`/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-tblite-extreme-reference-smokes-20260816`

## Full TBLite runs

All runs use TP=2/DP=4 on one allocated eight-H100 node, four concurrent
engines, urgent priority, and seeds 4202/4203/4204.

| K | Replicate | Seed | Beaker experiment | Result dataset | Status |
|---:|---:|---:|---|---|---|
| 64 | 1 | 4202 | [01M062BS15SCB41GV3CYC98H32](https://beaker.org/ex/01M062BS15SCB41GV3CYC98H32) | `01M062BS1EPA0GD2EYHQX73WBR` | queued |
| 64 | 2 | 4203 | [01M062BWZCAFW95QDWQ3DVRWJ4](https://beaker.org/ex/01M062BWZCAFW95QDWQ3DVRWJ4) | `01M062BWZPN8AVPW9G0DM41KA7` | queued |
| 64 | 3 | 4204 | [01M062C17MXBV5HW6KGYPXEYF7](https://beaker.org/ex/01M062C17MXBV5HW6KGYPXEYF7) | `01M062C1834X9XK25Z9QP1QDEW` | queued |
| 128 | 1 | 4202 | [01M062K48HR28GZPTBSCDNY1K6](https://beaker.org/ex/01M062K48HR28GZPTBSCDNY1K6) | `01M062K48Y28EV9TF8ZAGP9ZFY` | queued |
| 128 | 2 | 4203 | [01M062K80RR6TXJHFG77W8A5ZF](https://beaker.org/ex/01M062K80RR6TXJHFG77W8A5ZF) | `01M062K80ZAMCVCVZANZ7QN0S1` | queued |
| 128 | 3 | 4204 | [01M062KBKJ9Q0ZYJET986QVK70](https://beaker.org/ex/01M062KBKJ9Q0ZYJET986QVK70) | `01M062KBNF36AARPCVYBC1JA9G` | queued |
| 256 | 1 | 4202 | [01M062GA6KD2Y32HVSXR7H21PS](https://beaker.org/ex/01M062GA6KD2Y32HVSXR7H21PS) | `01M062GA6YFM0AEGM34WGZS8FJ` | queued |
| 256 | 2 | 4203 | [01M062GE1JXHTF4NWSX028WA8D](https://beaker.org/ex/01M062GE1JXHTF4NWSX028WA8D) | `01M062GE1TDE8MCQ997A7S8CZF` | queued |
| 256 | 3 | 4204 | [01M062GJAAN5BC2Y1RC4NG4X61](https://beaker.org/ex/01M062GJAAN5BC2Y1RC4NG4X61) | `01M062GJAH293FT438BA1GDKW6` | queued |

Full-results root:
`/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-tblite-extreme-reference-sweep-20260816`

## Final outcome

- K=64: all three runs succeeded; mean reward 51.80% ± 2.72, mean pass@1
  53.0%, and 18.0 task errors/run.
- K=128: all three runs succeeded; mean reward 27.33% ± 3.16, mean pass@1
  29.0%, and 42.7 task errors/run.
- K=256: all three production jobs failed before evaluation because TP=2/DP=4
  could not allocate KV-cache blocks at the 262,144-token context. No K=256
  point is plotted.

The extreme curve establishes a large degradation at K=64 and a near-collapse
at K=128 under reference scaling, despite exact routing validation.
