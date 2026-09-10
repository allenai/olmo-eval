# Qwen3.6 capability K=16/K=32 bridge

Launched 2026-08-17 to fill the capability-curve gap between the completed
K={4,6,8,10,12} sweep and the retained K=64 endpoint. Each condition bundles
MATH-500, GPQA Diamond, AIME 2025 pass@32, and the three-task IFBench macro.

Common settings: K=8 reference scaling without unit-sum renormalization,
32,768-token task generation caps, 110,000-token server context, four TP=1
vLLM engines, Qwen thinking enabled, urgent unallocated scheduling in
`ai2/holmes-testing`, and the multi-cluster `80g` selector.

Group: [qwen36-capability-bridge-reference-20260817](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01M08KNNYYQWNAKRAQYYHHRK2J)

| K | Rep | Seed | Beaker experiment | Job ID | Initial state |
|---:|---:|---:|---|---|---|
| 16 | 1 | 42 | [01M08KNV49PJJK9026V1APSC1J](https://beaker.org/ex/01M08KNV49PJJK9026V1APSC1J) | `01M08KNV81SQ4D6XRNVH98FGGD` | running |
| 16 | 2 | 43 | [01M08KP8W703NNGDH9J4YT6W4W](https://beaker.org/ex/01M08KP8W703NNGDH9J4YT6W4W) | `01M08KP8ZQXRMJJAZKZB6CKQYS` | running |
| 16 | 3 | 44 | [01M08KPNJC1V914E7GN26MRA76](https://beaker.org/ex/01M08KPNJC1V914E7GN26MRA76) | `01M08KPNNW529N8MSSWM3KF21B` | running/scheduling |
| 32 | 1 | 42 | [01M08KQ3C6K55GQTK7QT9K50HX](https://beaker.org/ex/01M08KQ3C6K55GQTK7QT9K50HX) | `01M08KQ3HHYFSTTZN2FJ6P3B4X` | queued |
| 32 | 2 | 43 | [01M08KQJB72P26KH1NXYQVX737](https://beaker.org/ex/01M08KQJB72P26KH1NXYQVX737) | `01M08KQJF6ET2XM8ZXJ0KB1DZP` | queued |
| 32 | 3 | 44 | [01M08KR0Z9FHQMEASY7Q9GWDZJ](https://beaker.org/ex/01M08KR0Z9FHQMEASY7Q9GWDZJ) | `01M08KR13273TDZH073F5S8EQ6` | queued |

Post-submit inspection confirmed `text_config.num_experts_per_tok`, router K,
and keep K equal the requested K in all six jobs, with reference K=8 and
`renormalize=false`.

## K=32 runtime-limit failures (2026-08-18)

All three K=32 jobs reached Beaker's 24-hour runtime limit and exited with
code 143 before the bundled suite completed. Final scored counts were
3839/4189, 3890/4189, and 3790/4189. These partial outputs are excluded from
the capability table and plot; no K=32 aggregate is reported from them.
