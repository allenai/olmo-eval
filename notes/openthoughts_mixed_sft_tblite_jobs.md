# OpenThoughts mixed-K SFT TBLite evaluation

Launched 2026-08-17 for the mixed-K={4,8,12} OpenThoughts-Agent SFT checkpoint:

`/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints/qwen35-openthoughts-agent-100k-mixed-k4-k8-k12-refk8-lr2e-5-flce-b64-20260817-hf`

The matrix evaluates K={4,6,8,10,12} with three independent TBLite runs per K. All
conditions use reference-K=8 scaling (`renormalize=false`). K=4/6 route eight experts
and truncate to the requested K; K=8 uses the native routing width; K=10/12 explicitly
set both the Qwen3.5 text-config routing width and the runtime router width to the
requested K. Post-submit inspection confirmed these settings in every Beaker spec.

Common recipe: `openthoughts-tblite@2.0`, Terminus-2 JSON agent, Qwen3 XML tool parser,
Qwen3 reasoning parser, temperature 1.0, top-p 0.95, top-k 20, 262,144-token context,
81,920-token generation ceiling, one 8xH100 node (TP=2, DP=4), urgent allocated
scheduling in `ai2/OLMo-3-moe-experiments` targeting Jupiter and Ceres.

Results root:
`/weka/oe-adapt-default/jacobm/tmax-eval/openthoughts-sft-mixed-tblite-20260817`

| Eval K | Rep | Seed | Beaker experiment | Beaker job | Initial state |
|---:|---:|---:|---|---|---|
| 4 | 1 | 4202 | [01M08C2A4NRNQV4FZK9YWGTSNN](https://beaker.org/ex/01M08C2A4NRNQV4FZK9YWGTSNN) | `01M08C2A9147ARNWCYWF366NTQ` | queued |
| 4 | 2 | 4203 | [01M08C2G0N15QK9VCW2B2THQ49](https://beaker.org/ex/01M08C2G0N15QK9VCW2B2THQ49) | `01M08C2G48KS7EW4K1HA20KGWS` | queued |
| 4 | 3 | 4204 | [01M08C2M5WAYZR3B9EQTE1G35F](https://beaker.org/ex/01M08C2M5WAYZR3B9EQTE1G35F) | `01M08C2MMYAC8EX62SASMFMT39` | queued |
| 6 | 1 | 4202 | [01M08C2RZ4JX698VR5VC483CVV](https://beaker.org/ex/01M08C2RZ4JX698VR5VC483CVV) | `01M08C2S2THEX2ZGEX8QHK372X` | queued |
| 6 | 2 | 4203 | [01M08C2WWY62BEKYHY1JFF2H6V](https://beaker.org/ex/01M08C2WWY62BEKYHY1JFF2H6V) | `01M08C2XE9RYY4C42BX7149QM8` | queued |
| 6 | 3 | 4204 | [01M08C314G3NZV6G221RTQAA9W](https://beaker.org/ex/01M08C314G3NZV6G221RTQAA9W) | `01M08C31FYFQ1DWGNQ4E20BW8F` | queued |
| 8 | 1 | 4202 | [01M08C35XRGT8RXT2XGTRX6AMC](https://beaker.org/ex/01M08C35XRGT8RXT2XGTRX6AMC) | `01M08C361J9V5JZKE159B4T1NN` | queued |
| 8 | 2 | 4203 | [01M08C3A04Q19A5T1SH2J8STFV](https://beaker.org/ex/01M08C3A04Q19A5T1SH2J8STFV) | `01M08C3A525M9ZEYR3442FPSVJ` | queued |
| 8 | 3 | 4204 | [01M08C3E3KFZ8WZF3F1KHQVJBB](https://beaker.org/ex/01M08C3E3KFZ8WZF3F1KHQVJBB) | `01M08C3EDZ50DSSHMW16REPKFD` | queued |
| 10 | 1 | 4202 | [01M08C3J6CHWH5QJDR7KDPXPJV](https://beaker.org/ex/01M08C3J6CHWH5QJDR7KDPXPJV) | `01M08C3J9WBZKNJNPAVGSM1EAQ` | queued |
| 10 | 2 | 4203 | [01M08C3PYZP3BEEFVDMR0MP22X](https://beaker.org/ex/01M08C3PYZP3BEEFVDMR0MP22X) | `01M08C3Q2W01BJG0MNGZY2MSBQ` | queued |
| 10 | 3 | 4204 | [01M08C3TGB95H8ZN6AHP5TMVW0](https://beaker.org/ex/01M08C3TGB95H8ZN6AHP5TMVW0) | `01M08C3TY47TASZC78KADDJWG0` | queued |
| 12 | 1 | 4202 | [01M08C3YQQKJYYA3Y13ZMDNH36](https://beaker.org/ex/01M08C3YQQKJYYA3Y13ZMDNH36) | `01M08C3YVC355DCEJ8MYBJKAEM` | queued |
| 12 | 2 | 4203 | [01M08C42JQJ2TZ3S3E11NDW7AE](https://beaker.org/ex/01M08C42JQJ2TZ3S3E11NDW7AE) | `01M08C42PCDQD4X4TSR253S6T0` | queued |
| 12 | 3 | 4204 | [01M08C46NE3WDPT26ZHPY9QFJQ](https://beaker.org/ex/01M08C46NE3WDPT26ZHPY9QFJQ) | `01M08C46VCXKXPVZXA8F9WBJAN` | queued |

Launcher: `scripts/adaptive_experts/launch_openthoughts_mixed_sft_tblite.sh`

## Status at 2026-08-17 18:42 UTC

- Running: K=4 replicates 1/2/3, K=6 replicates 1/2/3, and K=8
  replicates 1/2.
- Queued: K=8 replicate 3 and all K=10/K=12 runs.
- Running-trial ranges: K=4 15--24/100, K=6 10--15/100, and K=8
  10--15/100.
- All eight started jobs passed the exact routing-policy assertion. No job
  has failed, OOMed, or stalled.

## Status at 2026-08-17 22:33 UTC

- K=8 replicate 1 completed successfully with 35.19% mean reward and 37.00%
  pass@1 across all 100 TBLite tasks.
- The remaining 14 runs are active. Current trial progress is K=4 at
  77--99/100, K=6 at 89--98/100, the two remaining K=8 runs at 58 and
  91/100, K=10 at 48--52/100, and K=12 at 46--60/100.
- No mixed-SFT job has failed. The first completed K=8 result is one
  replicate and is not interpreted as a final K-level estimate yet.

## Status at 2026-08-17 23:22 UTC

Five of 15 runs are complete: K=4 replicates 1/2, K=6 replicates 1/3, and
K=8 replicate 1. The currently available summaries are:

| Eval K | Runs | Mean reward | Pass@1 |
|---:|---:|---:|---:|
| 4 | 2/3 | 35.86 ± 1.44 | 38.00 ± 1.41 |
| 6 | 2/3 | 38.97 ± 2.01 | 40.50 ± 2.12 |
| 8 | 1/3 | 35.19 | 37.00 |

The other ten runs remain active. K=4 replicate 3 is at 94/100 tasks, K=6
replicate 2 at 98/100, K=8 replicate 2 at 99/100 and replicate 3 at 71/100,
K=10 at 64--75/100, and K=12 at 66--92/100. No job has failed. These partial
K summaries should not be treated as final until all three paired seeds are
available.

## Final results (2026-08-18)

All 15 runs completed successfully and all metrics are locally collected.
Values are means ± sample standard deviations across the three paired seeds.

| Eval K | Runs | Mean reward | Pass@1 |
|---:|---:|---:|---:|
| 4 | 3/3 | 33.56 ± 4.11 | 35.67 ± 4.16 |
| 6 | 3/3 | 37.90 ± 2.33 | 39.33 ± 2.52 |
| 8 | 3/3 | 37.58 ± 2.15 | 39.67 ± 2.31 |
| 10 | 3/3 | 41.98 ± 2.98 | 43.67 ± 3.21 |
| 12 | 3/3 | 42.05 ± 1.03 | 43.67 ± 0.58 |

The mixed-K checkpoint does not produce a flat inference-K curve. K=4 is
weakest, K=6 and K=8 are statistically similar, and K=10/K=12 form the
strongest plateau. Compared with the completed fixed-SFT matrix, the mixed
checkpoint is modestly higher at eval K=4, slightly lower at eval K=8, and
competitive with the strongest fixed-checkpoint results at eval K=12; these
differences generally overlap the observed run-to-run variation. The exact
machine-readable summary is in `openthoughts_mixed_sft_tblite_results.csv`.
