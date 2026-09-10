# Fixed OpenThoughts SFT TBLite K=6/K=10 backfill

Launched 2026-08-17 to extend the completed fixed-SFT train-K x eval-K matrix
from eval K={4,8,12} to the full K={4,6,8,10,12} grid. The new matrix contains
three TBLite replicates at eval K={6,10} for each checkpoint trained at
K={4,8,12}, for 18 jobs total.

All jobs use the established protocol: K=8 reference scaling without unit-sum
renormalization, `openthoughts-tblite@2.0`, Terminus-2 JSON, Qwen3 XML tool
parser, Qwen3 reasoning parser, temperature 1.0, top-p 0.95, top-k 20,
262,144-token context, 81,920-token output ceiling, and one allocated 8xH100
node with TP=2/DP=4. Scheduling is urgent in
`ai2/OLMo-3-moe-experiments`, targeting Jupiter and Ceres.

K=6 routes the K=8 reference set and retains six experts. K=10 explicitly
loads a config-only router-width-10 text config for each checkpoint, retains
all ten experts, and scales against the top-eight reference. All 18 submitted
specs passed post-submit validation.

Results root:
`/weka/oe-adapt-default/jacobm/tmax-eval/openthoughts-sft-tblite-bridge-k6-k10-20260817`

| Train K | Eval K | Rep | Seed | Beaker experiment | Job ID |
|---:|---:|---:|---:|---|---|
| 4 | 6 | 1 | 4202 | [01M08KRJKBY40KM6WJHH57Q6H6](https://beaker.org/ex/01M08KRJKBY40KM6WJHH57Q6H6) | `01M08KRJQ1372WK1K4KCYQ36B4` |
| 4 | 6 | 2 | 4203 | [01M08KRPSHTGPNPTXZJPDKVFZ2](https://beaker.org/ex/01M08KRPSHTGPNPTXZJPDKVFZ2) | `01M08KRPX06QD4451GFWART2YA` |
| 4 | 6 | 3 | 4204 | [01M08KRTFD3DKPRD2ZS68NRCX1](https://beaker.org/ex/01M08KRTFD3DKPRD2ZS68NRCX1) | `01M08KRTQBTYJKA2ZMJ4EVWQ4N` |
| 4 | 10 | 1 | 4202 | [01M08KRYDPFBQVWD172Q6S5VDC](https://beaker.org/ex/01M08KRYDPFBQVWD172Q6S5VDC) | `01M08KRYH693XBY5FXDEY0QQP6` |
| 4 | 10 | 2 | 4203 | [01M08KS2E0NJG5X8GZ7BRRRJYS](https://beaker.org/ex/01M08KS2E0NJG5X8GZ7BRRRJYS) | `01M08KS2PXKXQEH7YT1FR6TGC7` |
| 4 | 10 | 3 | 4204 | [01M08KS64MCFPE07G0E9CFB6HK](https://beaker.org/ex/01M08KS64MCFPE07G0E9CFB6HK) | `01M08KS6851EJEDVKBVWCCQZF0` |
| 8 | 6 | 1 | 4202 | [01M08KSAX3FBJT75QJYVFQ5TWM](https://beaker.org/ex/01M08KSAX3FBJT75QJYVFQ5TWM) | `01M08KSB285WNT07CBNGTMBF2T` |
| 8 | 6 | 2 | 4203 | [01M08KSEWWXGW307R6G3BGHFHN](https://beaker.org/ex/01M08KSEWWXGW307R6G3BGHFHN) | `01M08KSF153C63SMCEK99WEPY7` |
| 8 | 6 | 3 | 4204 | [01M08KSK6ACRDTA1VQWM89WTY5](https://beaker.org/ex/01M08KSK6ACRDTA1VQWM89WTY5) | `01M08KSKABCVZG174794RQBVV8` |
| 8 | 10 | 1 | 4202 | [01M08KSQ0W1AMHB1XXZQJTSRA4](https://beaker.org/ex/01M08KSQ0W1AMHB1XXZQJTSRA4) | `01M08KSQ49C5Q1VPED44ND045M` |
| 8 | 10 | 2 | 4203 | [01M08KSV5RZNS0MV77QQXKPW08](https://beaker.org/ex/01M08KSV5RZNS0MV77QQXKPW08) | `01M08KSV9N7A232CQ72ZYRB87V` |
| 8 | 10 | 3 | 4204 | [01M08KSZ0QGJCFSA5JYP1RE1W7](https://beaker.org/ex/01M08KSZ0QGJCFSA5JYP1RE1W7) | `01M08KSZ4BB25794MZFYT3HJFG` |
| 12 | 6 | 1 | 4202 | [01M08KT2GKTWHQ7X54D2Q07VH2](https://beaker.org/ex/01M08KT2GKTWHQ7X54D2Q07VH2) | `01M08KT2MHE6T163E8EMZEHSPB` |
| 12 | 6 | 2 | 4203 | [01M08KT7B9EW3CVRHSA3CTCXWG](https://beaker.org/ex/01M08KT7B9EW3CVRHSA3CTCXWG) | `01M08KT7EYG16H9S3HTR7W0VG5` |
| 12 | 6 | 3 | 4204 | [01M08KTB5J26XRE7B44H08GMCN](https://beaker.org/ex/01M08KTB5J26XRE7B44H08GMCN) | `01M08KTB93QB6SFMT1W2C0GRR0` |
| 12 | 10 | 1 | 4202 | [01M08KTEYEBY6ZCD7VDZT2JBAD](https://beaker.org/ex/01M08KTEYEBY6ZCD7VDZT2JBAD) | `01M08KTF1T6CGVC3NFQ2HQR8VN` |
| 12 | 10 | 2 | 4203 | [01M08KTJKF1KY16G9DGWTM6EPX](https://beaker.org/ex/01M08KTJKF1KY16G9DGWTM6EPX) | `01M08KTJQQ56FB569S1H4YXAYE` |
| 12 | 10 | 3 | 4204 | [01M08KTP9WZNCZXF6B1A7ZB9JW](https://beaker.org/ex/01M08KTP9WZNCZXF6B1A7ZB9JW) | `01M08KTPHZEX48XN27VQGGNAS3` |

Initial state: all 18 jobs are queued.

Status at 2026-08-18 03:20 UTC: train-K=4/eval-K=6 replicate 1 is running
and has completed 70/100 TBLite tasks. The other 17 jobs remain queued; none
has failed.

## Train-K=8 / eval-K=10 / replicate-2 rerun

The original replicate-2 experiment `01M08KSV5RZNS0MV77QQXKPW08` stopped
advancing at 48/100 trials at 09:29 UTC and was manually canceled at 16:19
UTC on 2026-08-18. Its partial result is excluded.

An exact replacement using seed 4203 was submitted as
[01M0ATRTQD55HFXYGVPVF80Q08](https://beaker.org/ex/01M0ATRTQD55HFXYGVPVF80Q08)
(job `01M0ATRTTYAGD26VQ2VED87XE5`). It retains router/keep K=10,
reference K=8, `renormalize=false`, the same config-only K=10 router width,
and every original TBLite generation, parser, resource, priority, and cluster
setting. The replacement was initially queued.


The MoE-workspace replacement was subsequently canceled while still queued.
The same seed-4203 evaluation was moved to `ai2/holmes-testing` as
[01M0AVPNCD551BAEM1F540NWN6](https://beaker.org/ex/01M0AVPNCD551BAEM1F540NWN6) (job `01M0AVPNJPT5V8GA5D5D2Z28DF`).
It was submitted after all 15 phased K={4,6,8,10,12} jobs and passed exact
post-submit validation.

At 22:40 UTC on 2026-08-18, that still-queued allocated submission was
canceled and replaced by unallocated experiment
[01M0BGCHEWWPRF24YRYFYEF0EP](https://beaker.org/ex/01M0BGCHEWWPRF24YRYFYEF0EP)
(job `01M0BGCHKA9H995CZ1HGQ1JG1Q`). It remains urgent in
`ai2/holmes-testing`, targets exactly Jupiter and Ceres, and preserves every
routing and evaluation setting. Post-submit validation confirmed K=10,
reference K=8, no renormalization, the config-only router-width override,
eight GPUs, and unallocated/preemptible scheduling.

At 03:48 UTC on 2026-08-19, the still-queued Holmes submission was canceled
and replaced in `ai2/OLMo-3-moe-experiments` by
[01M0C22Y01HGAM72RJ31MWFXTS](https://beaker.org/ex/01M0C22Y01HGAM72RJ31MWFXTS)
(job `01M0C22Y4SJMF9M5XEC2QVNN10`). The new submission remains urgent and
unallocated, targets exactly Jupiter and Ceres, and passed exact validation
of the K=10 routing, reference scaling, parser, generation, resource, and
scheduling settings.

The replacement completed successfully with all 100 tasks: mean reward
0.4396841153846154 and pass@1 0.47. Across the three train-K=8/eval-K=10
replicates, mean reward is 0.41603451044634376±0.028156850697033053 and
pass@1 is 0.43666666666666665±0.03511884584284244. The completed cell was
added to the shared SFT table and plot; all three raw runs are recorded in
`notes/openthoughts_sft_tblite_bridge_results.csv`.
