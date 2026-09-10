# OpenThoughts mixed-K SFT TBLite extension

Launched 2026-08-18 for the mixed-K={4,8,12} OpenThoughts-Agent SFT checkpoint:

`/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints/qwen35-openthoughts-agent-100k-mixed-k4-k8-k12-refk8-lr2e-5-flce-b64-20260817-hf`

This extends the completed K={4,6,8,10,12} matrix to
K={2,14,16,24,32}, with three paired-seed TBLite replicates at every K.
All settings match the original matrix: reference-K=8 scaling with
`renormalize=false`, Terminus-2 JSON, Qwen3 XML tool parsing, Qwen3 reasoning
parsing, temperature 1.0, top-p 0.95, top-k 20, 262,144-token context, and an
81,920-token generation ceiling. Each run requests one 8xH100 node with TP=2
and DP=4 at urgent allocated priority in `ai2/OLMo-3-moe-experiments`,
targeting Jupiter and Ceres.

K=2 routes the native top eight and truncates to the first two. K>8 sets both
the Qwen3.5 configuration routing width and the runtime router width to the
requested K. Every submitted spec was checked after launch: keep-K, router-K,
reference-K, renormalization, and the nested Hugging Face override are exact
for all 15 jobs.

Results root:
`/weka/oe-adapt-default/jacobm/tmax-eval/openthoughts-sft-mixed-tblite-extension-20260818`

| Eval K | Rep | Seed | Beaker experiment | Beaker job | Initial state |
|---:|---:|---:|---|---|---|
| 2 | 1 | 4202 | [01M09F4AV8S1BQTKATS2K1Y62E](https://beaker.org/ex/01M09F4AV8S1BQTKATS2K1Y62E) | `01M09F4AZA9X0T2TQ10FRHDD95` | queued |
| 2 | 2 | 4203 | [01M09F4EHF4TJQ68EH17N6E4E0](https://beaker.org/ex/01M09F4EHF4TJQ68EH17N6E4E0) | `01M09F4EMYD5404RHGDECD7ZEM` | queued |
| 2 | 3 | 4204 | [01M09F4J77ZM05XX3NT3VB7KQ3](https://beaker.org/ex/01M09F4J77ZM05XX3NT3VB7KQ3) | `01M09F4JAYDVTM3VKZDBKGFE1C` | queued |
| 14 | 1 | 4202 | [01M09F4PDXRE13SHPZAZ1E20NC](https://beaker.org/ex/01M09F4PDXRE13SHPZAZ1E20NC) | `01M09F4PHHX28GEJ9HRDDYWZPV` | queued |
| 14 | 2 | 4203 | [01M09F4SZ7BETNF4CW047M407D](https://beaker.org/ex/01M09F4SZ7BETNF4CW047M407D) | `01M09F4T2WHKEDEF7V5DYK8EYK` | queued |
| 14 | 3 | 4204 | [01M09F4XC82CZ0X7B26T4Z0CFN](https://beaker.org/ex/01M09F4XC82CZ0X7B26T4Z0CFN) | `01M09F4XFZ1G3N0DD13NH7K2PR` | queued |
| 16 | 1 | 4202 | [01M09F5121BRX6WXM8FZQRAY1H](https://beaker.org/ex/01M09F5121BRX6WXM8FZQRAY1H) | `01M09F515WRPPRYG7RGY5KCRNX` | queued |
| 16 | 2 | 4203 | [01M09F54VS4QH3SDJR4QZN3BYG](https://beaker.org/ex/01M09F54VS4QH3SDJR4QZN3BYG) | `01M09F54ZFAXVGAX8VNW9030ZB` | queued |
| 16 | 3 | 4204 | [01M09F58DBR7ZNZFTX5MC91JCX](https://beaker.org/ex/01M09F58DBR7ZNZFTX5MC91JCX) | `01M09F58HF6AXKMR9C8NTWRJSD` | queued |
| 24 | 1 | 4202 | [01M09F5BZ78GEWEF9PMHYX36BN](https://beaker.org/ex/01M09F5BZ78GEWEF9PMHYX36BN) | `01M09F5C2T5DX8F0F3EY88CY4X` | queued |
| 24 | 2 | 4203 | [01M09F5FGRH84WM17ZKWYYM0WP](https://beaker.org/ex/01M09F5FGRH84WM17ZKWYYM0WP) | `01M09F5FMBTQTBG9S6TXSBAK8K` | queued |
| 24 | 3 | 4204 | [01M09F5KA8VGC62PX10J356WEG](https://beaker.org/ex/01M09F5KA8VGC62PX10J356WEG) | `01M09F5KE0N1617QMBE57FRSG7` | queued |
| 32 | 1 | 4202 | [01M09F5PVXN0YH0SHPTZ98VGYT](https://beaker.org/ex/01M09F5PVXN0YH0SHPTZ98VGYT) | `01M09F5Q03JHT5ACB902VKX6SY` | queued |
| 32 | 2 | 4203 | [01M09F5TAYYP0SEV9WZBN4JEKP](https://beaker.org/ex/01M09F5TAYYP0SEV9WZBN4JEKP) | `01M09F5TEBHY9DK2EXXE9W6JX8` | queued |
| 32 | 3 | 4204 | [01M09F5Y323S6MSRDYM34S0CBR](https://beaker.org/ex/01M09F5Y323S6MSRDYM34S0CBR) | `01M09F5Y6PGE6GEB2WH5MNH728` | queued |

Launcher: `scripts/adaptive_experts/launch_openthoughts_mixed_sft_tblite.sh`

## K=2 and K=24 cancellation

At 16:43 UTC on 2026-08-18, all three K=2 and all three K=24 runs were
manually canceled to conserve compute. Their final completed-trial counts
were K=2: 59/52/56 and K=24: 60/67/49. These partial results are excluded
from tables and plots. All three K=32 runs were explicitly retained and
remained running.

## Results refresh (2026-08-18 19:08 UTC)

All three K=14 and all three K=16 runs completed with 100/100 tasks and were
collected into the shared SFT table and plot. K=14 has mean reward
0.398684877730±0.067532019933 and pass@1 0.416666666667±0.065064070986;
K=16 has mean reward 0.380454766619±0.015883260279 and pass@1
0.400000000000±0.010000000000. Uncertainty is sample SD over three runs.
The K=32 runs remain in progress at 94/100, 72/100, and 67/100 tasks.

## K=32 completion (2026-08-18 21:14 UTC)

All three K=32 runs completed with 100/100 tasks. Per-run mean rewards were
0.300645383903, 0.425637717236, and 0.321512115385; pass@1 values were 0.33,
0.45, and 0.34. The aggregate is mean reward 0.349265072175±0.066958500625
and pass@1 0.373333333333±0.066583281185. The complete cell was added to the
shared SFT comparison table and plot.
