# Qwen3.6 TBLite intermediate reference-scaled sweep

Launched 2026-08-17 in `ai2/OLMo-3-moe-experiments` at urgent, allocated
priority, targeting `ai2/jupiter` and `ai2/ceres`. This fills the even expert
counts strictly between K=16 and K=32: K={18,20,22,24,26,28,30}, with three
replicates per K.

All jobs use the same protocol as the maintained Qwen3.6 TBLite sweep:
`openthoughts-tblite@2.0`, one 8-GPU node, TP=2/DP=4, four concurrent agents,
Vanillux2, `qwen3_xml`, 262,144-token context, 81,920-token output ceiling,
temperature=1.0, top-p=0.95, and top-k=20. Routing is reference-scaled to
native K=8 (`renormalize=false`). Post-submit inspection verified router K,
keep K, and nested `text_config.num_experts_per_tok` equal the requested K in
all 21 specs.

| K | Replicate 1 | Replicate 2 | Replicate 3 | Status at 2026-08-17 04:28 UTC |
|---:|---|---|---|---|
| 18 | [01M06Z2DD0WG7RBTXXTA8ADDSQ](https://beaker.org/ex/01M06Z2DD0WG7RBTXXTA8ADDSQ) | [01M06Z2GW1NSDH954S0X37ZDYA](https://beaker.org/ex/01M06Z2GW1NSDH954S0X37ZDYA) | [01M06Z2MKPSA53XT94RR1J1T59](https://beaker.org/ex/01M06Z2MKPSA53XT94RR1J1T59) | 3 queued |
| 20 | [01M06Z2R9YEYWK6YSMYHB3MZ4A](https://beaker.org/ex/01M06Z2R9YEYWK6YSMYHB3MZ4A) | [01M06Z2VYD6M90S65TG9MR7EES](https://beaker.org/ex/01M06Z2VYD6M90S65TG9MR7EES) | [01M06Z2ZJV1K3A8818XPQSCR3Q](https://beaker.org/ex/01M06Z2ZJV1K3A8818XPQSCR3Q) | 3 queued |
| 22 | [01M06Z334SZNSM0TFV41SH12JC](https://beaker.org/ex/01M06Z334SZNSM0TFV41SH12JC) | [01M06Z36SWCG59C9NGGY1Z3EQG](https://beaker.org/ex/01M06Z36SWCG59C9NGGY1Z3EQG) | [01M06Z3AGE5917E6ZDP0NAVEWV](https://beaker.org/ex/01M06Z3AGE5917E6ZDP0NAVEWV) | 3 queued |
| 24 | [01M06Z3E0Z06W97XH10NDSB0H3](https://beaker.org/ex/01M06Z3E0Z06W97XH10NDSB0H3) | [01M06Z3HNZQ2GEY3J1ZJNTS2G2](https://beaker.org/ex/01M06Z3HNZQ2GEY3J1ZJNTS2G2) | [01M06Z3NFR5RJ69VH375DG8BKA](https://beaker.org/ex/01M06Z3NFR5RJ69VH375DG8BKA) | 3 queued |
| 26 | [01M06Z3S0YE73AJ4TVA6A62CEX](https://beaker.org/ex/01M06Z3S0YE73AJ4TVA6A62CEX) | [01M06Z3WX0XD7RPMXJYBRGXRQB](https://beaker.org/ex/01M06Z3WX0XD7RPMXJYBRGXRQB) | [01M06Z44103Y49C6A4Z617606G](https://beaker.org/ex/01M06Z44103Y49C6A4Z617606G) | 3 queued |
| 28 | [01M06Z47SRVGDZ0XJBF5YRQEV9](https://beaker.org/ex/01M06Z47SRVGDZ0XJBF5YRQEV9) | [01M06Z4BG234CBHZHT0K1G5N9E](https://beaker.org/ex/01M06Z4BG234CBHZHT0K1G5N9E) | [01M06Z4F86S8T063JZTTP5KPEE](https://beaker.org/ex/01M06Z4F86S8T063JZTTP5KPEE) | 3 queued |
| 30 | [01M06Z4K3N1BK4YV0MG2G8HKBW](https://beaker.org/ex/01M06Z4K3N1BK4YV0MG2G8HKBW) | [01M06Z4PYVH3A5EP0QDPKC68S4](https://beaker.org/ex/01M06Z4PYVH3A5EP0QDPKC68S4) | [01M06Z4TPCMHKZ6V3ZPVX1HCQ7](https://beaker.org/ex/01M06Z4TPCMHKZ6V3ZPVX1HCQ7) | 3 queued |

## Final results collected 2026-08-17

All 21 jobs succeeded with complete 100-task metric files. Values are the
mean and sample standard deviation of mean reward across three replicates.

| K | Mean reward | Mean pass@1 |
|---:|---:|---:|
| 18 | 64.28% ± 4.46 | 65.33% |
| 20 | 59.27% ± 0.66 | 60.33% |
| 22 | 65.22% ± 2.25 | 66.67% |
| 24 | 65.50% ± 2.38 | 66.67% |
| 26 | 62.85% ± 2.44 | 64.00% |
| 28 | 60.96% ± 2.80 | 62.00% |
| 30 | 65.47% ± 2.27 | 66.67% |

The individual points are noisy, but there is no monotonic degradation from
K=16 through K=32. The maintained plot and CSV contain all replicates.
