# OpenThoughts phased/cooldown SFT TBLite sweep

Launched 2026-08-18 for the OpenThoughts-Agent SFT checkpoint trained with a
three-stage expert-count cooldown: K=12 for the first third, K=8 for the
second third, and K=4 for the final third.

Checkpoint:
`/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints/qwen35-openthoughts-agent-100k-phased-k12-k8-k4-refk8-lr2e-5-flce-b64-20260818-hf`

The sweep evaluates K={2,4,6,8,10,12,14,16}, with three paired-seed TBLite
replicates per K. Submission order was deliberately K={4,8,12}, then
K={6,10}, then K={2,14,16}, so the directly comparable training-regime points
were placed at the front of the queue.

All settings match the other SFT comparisons: `openthoughts-tblite@2.0`,
reference-K=8 scaling with `renormalize=false`, Terminus-2 JSON, Qwen3 XML
tool parsing, Qwen3 reasoning parsing, temperature 1.0, top-p 0.95, top-k 20,
262,144-token context, and an 81,920-token generation ceiling. Each run
requests one 8xH100 node with TP=2 and DP=4 at urgent allocated priority in
`ai2/OLMo-3-moe-experiments`, targeting Jupiter and Ceres.

K<=8 routes the top eight and truncates to the requested K. K>8 sets both the
nested Qwen3.5 text configuration and the runtime router width to the
requested K. All 24 submitted specs passed exact checks for keep-K, router-K,
reference-K, renormalization, nested config override, dataset, parsers,
resources, priority, and cluster targeting. The jobs retain the fail-closed
live router assertion.

Results root:
`/weka/oe-adapt-default/jacobm/tmax-eval/openthoughts-sft-phased-k12-k8-k4-tblite-20260818`

| Queue order | Eval K | Rep | Seed | Beaker experiment | Beaker job | Initial state |
|---:|---:|---:|---:|---|---|---|
| 1 | 4 | 1 | 4202 | [01M0AT0BBA2VGDR94XQ07K9ZHG](https://beaker.org/ex/01M0AT0BBA2VGDR94XQ07K9ZHG) | `01M0AT0BEV5FP2T53MDVHVKV4P` | queued |
| 2 | 4 | 2 | 4203 | [01M0AT0EYTYENHJE1PZ5VTR7K8](https://beaker.org/ex/01M0AT0EYTYENHJE1PZ5VTR7K8) | `01M0AT0F4XCAS5SCS52PXAVF9C` | queued |
| 3 | 4 | 3 | 4204 | [01M0AT0JDXF4ENMNCD56H7A9M5](https://beaker.org/ex/01M0AT0JDXF4ENMNCD56H7A9M5) | `01M0AT0JN093FASAYEMZZQCZ9D` | queued |
| 4 | 8 | 1 | 4202 | [01M0AT0PDA0CQPVTNSRSPN4DTQ](https://beaker.org/ex/01M0AT0PDA0CQPVTNSRSPN4DTQ) | `01M0AT0PGVV7ERNBXR94YNBWHX` | queued |
| 5 | 8 | 2 | 4203 | [01M0AT0TATHAYER53G6G20NX4Y](https://beaker.org/ex/01M0AT0TATHAYER53G6G20NX4Y) | `01M0AT0TH83CMFW446MFPXZQKS` | queued |
| 6 | 8 | 3 | 4204 | [01M0AT0Y8CDGQFXK0S8717SRFH](https://beaker.org/ex/01M0AT0Y8CDGQFXK0S8717SRFH) | `01M0AT0YBXDJMAFJ223JRKGR4E` | queued |
| 7 | 12 | 1 | 4202 | [01M0AT11WK5X65KMY2B9TDZ3GZ](https://beaker.org/ex/01M0AT11WK5X65KMY2B9TDZ3GZ) | `01M0AT122WM2JH2H66DBN0WVXD` | queued |
| 8 | 12 | 2 | 4203 | [01M0AT162J68WSHWR0JANB5BNQ](https://beaker.org/ex/01M0AT162J68WSHWR0JANB5BNQ) | `01M0AT169NQAQ5QW7SYCF6S0Q0` | queued |
| 9 | 12 | 3 | 4204 | [01M0AT19RQ6STN92S1ZXXJC8BR](https://beaker.org/ex/01M0AT19RQ6STN92S1ZXXJC8BR) | `01M0AT19W52YBGC3YJH7E35RBX` | queued |
| 10 | 6 | 1 | 4202 | [01M0AT1D4X39ETKH5YV1VRYK1E](https://beaker.org/ex/01M0AT1D4X39ETKH5YV1VRYK1E) | `01M0AT1DA91P10F2JZNAZ7K7R3` | queued |
| 11 | 6 | 2 | 4203 | [01M0AT1GPQKHFAQMVE2ZYBX6QV](https://beaker.org/ex/01M0AT1GPQKHFAQMVE2ZYBX6QV) | `01M0AT1GT7AHWDG9KRXTKH9N06` | queued |
| 12 | 6 | 3 | 4204 | [01M0AT1MDEKPJXRSZ03HRN4JSB](https://beaker.org/ex/01M0AT1MDEKPJXRSZ03HRN4JSB) | `01M0AT1MKZ4XAQ6KR2RC04SZFX` | queued |
| 13 | 10 | 1 | 4202 | [01M0AT1RANVNJEM8JCY08P1HEW](https://beaker.org/ex/01M0AT1RANVNJEM8JCY08P1HEW) | `01M0AT1REJQ4J7WMGSB7C1QA7P` | queued |
| 14 | 10 | 2 | 4203 | [01M0AT1WADF9XJXKND557ECVXX](https://beaker.org/ex/01M0AT1WADF9XJXKND557ECVXX) | `01M0AT1WE6SBSHDGT37MMRY1BA` | queued |
| 15 | 10 | 3 | 4204 | [01M0AT1ZY8VKFHXJB48MM4D48B](https://beaker.org/ex/01M0AT1ZY8VKFHXJB48MM4D48B) | `01M0AT201SMVYTD1K46JKZNWNB` | queued |
| 16 | 2 | 1 | 4202 | [01M0AT23VXXXMCDG83BNFHZKRQ](https://beaker.org/ex/01M0AT23VXXXMCDG83BNFHZKRQ) | `01M0AT2402JK26JXMT74DSATC0` | queued |
| 17 | 2 | 2 | 4203 | [01M0AT27NZNHMZN17W20ZXQHJS](https://beaker.org/ex/01M0AT27NZNHMZN17W20ZXQHJS) | `01M0AT27SE56WT8TPTH1JSRZ26` | queued |
| 18 | 2 | 3 | 4204 | [01M0AT2BH6E4SPHA8W2GKXMG0F](https://beaker.org/ex/01M0AT2BH6E4SPHA8W2GKXMG0F) | `01M0AT2BS1G2Q2R7BREZ7F9YA3` | queued |
| 19 | 14 | 1 | 4202 | [01M0AT2FSEQEP1MNB5T2YHBC0P](https://beaker.org/ex/01M0AT2FSEQEP1MNB5T2YHBC0P) | `01M0AT2FX091ZYTFA2772S7ZVC` | queued |
| 20 | 14 | 2 | 4203 | [01M0AT2KHCWFJ5M62JM482BWFC](https://beaker.org/ex/01M0AT2KHCWFJ5M62JM482BWFC) | `01M0AT2KN6T7CF4XARVV5708TN` | queued |
| 21 | 14 | 3 | 4204 | [01M0AT2QC5D7P4KQZKNTYV5N81](https://beaker.org/ex/01M0AT2QC5D7P4KQZKNTYV5N81) | `01M0AT2QFK9XS5PSVNDE58NVEX` | queued |
| 22 | 16 | 1 | 4202 | [01M0AT2TZ8AFTYA8S0F964RMYS](https://beaker.org/ex/01M0AT2TZ8AFTYA8S0F964RMYS) | `01M0AT2V2Z464SNHXFYK9A5R7T` | queued |
| 23 | 16 | 2 | 4203 | [01M0AT2YRFXRRF8D7MJX2TF8W6](https://beaker.org/ex/01M0AT2YRFXRRF8D7MJX2TF8W6) | `01M0AT2Z374GB9DTXEVXEGETJS` | queued |
| 24 | 16 | 3 | 4204 | [01M0AT32WFH0A6B1KJAS9DW7E1](https://beaker.org/ex/01M0AT32WFH0A6B1KJAS9DW7E1) | `01M0AT32ZW5071374Y6XBCPN9E` | queued |

Launcher: `scripts/adaptive_experts/launch_openthoughts_mixed_sft_tblite.sh`

## Holmes queue migration on 2026-08-18

All 24 original MoE-workspace submissions were canceled before allocation.
The reduced sweep was resubmitted in `ai2/holmes-testing` with three runs at
K={4,6,8,10,12}; K={2,14,16} was intentionally dropped. Submission order was
K={4,8,12}, then K={6,10}. All settings otherwise match the original sweep,
and all 15 post-submit specifications passed exact validation.

New results root:
`/weka/oe-adapt-default/jacobm/tmax-eval/openthoughts-sft-phased-k12-k8-k4-tblite-20260818-holmes`

| Queue order | Eval K | Rep | Seed | Beaker experiment | Beaker job | Initial state |
|---:|---:|---:|---:|---|---|---|
| 1 | 4 | 1 | 4202 | [01M0AVM8T8Q48AWYC17T16F8F5](https://beaker.org/ex/01M0AVM8T8Q48AWYC17T16F8F5) | `01M0AVM8Z2J7VEJ867D7DCBT19` | queued |
| 2 | 4 | 2 | 4203 | [01M0AVMCQPHP1ZSDP467PK63R6](https://beaker.org/ex/01M0AVMCQPHP1ZSDP467PK63R6) | `01M0AVMCVAWF1TZ1GGHSD6BV70` | queued |
| 3 | 4 | 3 | 4204 | [01M0AVMGZNWXV5S34T2DW6TBMW](https://beaker.org/ex/01M0AVMGZNWXV5S34T2DW6TBMW) | `01M0AVMH3VRGTKY6TN9R36JNZ6` | queued |
| 4 | 8 | 1 | 4202 | [01M0AVMMHMD48W9WMN75S9RC7C](https://beaker.org/ex/01M0AVMMHMD48W9WMN75S9RC7C) | `01M0AVMMQZXGQR1QWT4PFEN75S` | queued |
| 5 | 8 | 2 | 4203 | [01M0AVMRDYF9QR6W0A41MK9N56](https://beaker.org/ex/01M0AVMRDYF9QR6W0A41MK9N56) | `01M0AVMRMHD83EAJR0RCF00MHP` | queued |
| 6 | 8 | 3 | 4204 | [01M0AVMW0WPYRWNSB3FP9XXS5X](https://beaker.org/ex/01M0AVMW0WPYRWNSB3FP9XXS5X) | `01M0AVMW4EZ7VT1ZXS6PP7QZNQ` | queued |
| 7 | 12 | 1 | 4202 | [01M0AVMZRRGC6NFAZZ5SC9XDG8](https://beaker.org/ex/01M0AVMZRRGC6NFAZZ5SC9XDG8) | `01M0AVMZW5K1CA1WT0AYEHFYMZ` | queued |
| 8 | 12 | 2 | 4203 | [01M0AVN3PZGFTSN50XQAWE755V](https://beaker.org/ex/01M0AVN3PZGFTSN50XQAWE755V) | `01M0AVN3W8N7AV3QSY9SJ8TZQN` | queued |
| 9 | 12 | 3 | 4204 | [01M0AVN800TP38QMEZE8SNA41B](https://beaker.org/ex/01M0AVN800TP38QMEZE8SNA41B) | `01M0AVN8783J29X4NY39475687` | queued |
| 10 | 6 | 1 | 4202 | [01M0AVNMRDHQTH08N2758XSJYQ](https://beaker.org/ex/01M0AVNMRDHQTH08N2758XSJYQ) | `01M0AVNMXAQ2ZM053SH43RND0R` | queued |
| 11 | 6 | 2 | 4203 | [01M0AVNWX18KPDBBY82VNPH1XR](https://beaker.org/ex/01M0AVNWX18KPDBBY82VNPH1XR) | `01M0AVNX0HE0Z2SCFF6SR4ZSPG` | queued |
| 12 | 6 | 3 | 4204 | [01M0AVP0HVTDP46KHXAGQR75KT](https://beaker.org/ex/01M0AVP0HVTDP46KHXAGQR75KT) | `01M0AVP0YFTHX9DEZ39TVY8MKM` | queued |
| 13 | 10 | 1 | 4202 | [01M0AVP4GEPQSAGZ38GEMKD7VT](https://beaker.org/ex/01M0AVP4GEPQSAGZ38GEMKD7VT) | `01M0AVP4PK96K5HZRX3C02YFFV` | queued |
| 14 | 10 | 2 | 4203 | [01M0AVP8C4VKTVCCM7SZW1TKB9](https://beaker.org/ex/01M0AVP8C4VKTVCCM7SZW1TKB9) | `01M0AVP8G2QPZSK65DAAXB9YX6` | queued |
| 15 | 10 | 3 | 4204 | [01M0AVPC2F771P2SZR7Y4XK4SA](https://beaker.org/ex/01M0AVPC2F771P2SZR7Y4XK4SA) | `01M0AVPC7SR7SS3FQHGCN4CHFX` | queued |

## Results refresh (2026-08-18 22:06 UTC)

K=4 replicate 1 completed with all 100 tasks: mean reward
0.39433271723646723 and pass@1 0.41. Replicate 2 has started and is bringing
up its vLLM server; replicate 3 and the remaining twelve phased jobs are
queued. The completed run is recorded in
`notes/openthoughts_phased_sft_tblite_results.csv` and will enter the shared
comparison plot once all three K=4 replicates are complete.

## Unallocated Holmes replacements (2026-08-18 22:40 UTC)

The thirteen phased jobs that were still queued were replaced with urgent,
unallocated submissions in `ai2/holmes-testing`, still targeting exactly
Jupiter and Ceres. The running K=4 replicate 2 was left untouched. All new
specs passed checks for preemptible scheduling, routing, benchmark settings,
resources, workspace, priority, and cluster constraints.

| Eval K | Rep | Old allocated experiment | New unallocated experiment | New job |
|---:|---:|---|---|---|
| 4 | 3 | `01M0AVMGZNWXV5S34T2DW6TBMW` | [01M0BG9TVAZJ2073A32M4684M8](https://beaker.org/ex/01M0BG9TVAZJ2073A32M4684M8) | `01M0BG9V0793D4QV7NH27WWKBB` |
| 8 | 1 | `01M0AVMMHMD48W9WMN75S9RC7C` | [01M0BGANY0PJFA83Z1FW19VBBN](https://beaker.org/ex/01M0BGANY0PJFA83Z1FW19VBBN) | `01M0BGAP2F68XQ1GRYSTHFFC1N` |
| 8 | 2 | `01M0AVMRDYF9QR6W0A41MK9N56` | [01M0BGATF7V3F4KZP4EHPGGKK3](https://beaker.org/ex/01M0BGATF7V3F4KZP4EHPGGKK3) | `01M0BGATJTG8AV1307A2JYQT7M` |
| 8 | 3 | `01M0AVMW0WPYRWNSB3FP9XXS5X` | [01M0BGAYBJSY9A4WWEP65PPPB2](https://beaker.org/ex/01M0BGAYBJSY9A4WWEP65PPPB2) | `01M0BGAYEY2H6VX4Q4XCVBR5X9` |
| 12 | 1 | `01M0AVMZRRGC6NFAZZ5SC9XDG8` | [01M0BGB1YA701E32Y92W2JGF2K](https://beaker.org/ex/01M0BGB1YA701E32Y92W2JGF2K) | `01M0BGB22MNNEQ651GACPFSWSH` |
| 12 | 2 | `01M0AVN3PZGFTSN50XQAWE755V` | [01M0BGB5VT90QB1HA325A397JY](https://beaker.org/ex/01M0BGB5VT90QB1HA325A397JY) | `01M0BGB5ZCX2PC4SH5TSAR7A69` |
| 12 | 3 | `01M0AVN800TP38QMEZE8SNA41B` | [01M0BGB9S4T8ADSJ6X8M4NA87B](https://beaker.org/ex/01M0BGB9S4T8ADSJ6X8M4NA87B) | `01M0BGB9YJD0AEWZ1DWFNYP296` |
| 6 | 1 | `01M0AVNMRDHQTH08N2758XSJYQ` | [01M0BGBDQ3X4MVVG3KG8EH4649](https://beaker.org/ex/01M0BGBDQ3X4MVVG3KG8EH4649) | `01M0BGBDTW92GW90YANRX82RGN` |
| 6 | 2 | `01M0AVNWX18KPDBBY82VNPH1XR` | [01M0BGBJ44W4J7XP4MQG561TFP](https://beaker.org/ex/01M0BGBJ44W4J7XP4MQG561TFP) | `01M0BGBJ7QPZG5THRRXVTM1KGH` |
| 6 | 3 | `01M0AVP0HVTDP46KHXAGQR75KT` | [01M0BGBNSWSDZDQ1V2REFPH492](https://beaker.org/ex/01M0BGBNSWSDZDQ1V2REFPH492) | `01M0BGBNXCRXDDMGTVSBMXZ3DT` |
| 10 | 1 | `01M0AVP4GEPQSAGZ38GEMKD7VT` | [01M0BGBSHQAJSZ3MDNJB5VT3Y7](https://beaker.org/ex/01M0BGBSHQAJSZ3MDNJB5VT3Y7) | `01M0BGBSN9Z4TM09B4NTZ0SH62` |
| 10 | 2 | `01M0AVP8C4VKTVCCM7SZW1TKB9` | [01M0BGBWYS5T2845VETF32X1XZ](https://beaker.org/ex/01M0BGBWYS5T2845VETF32X1XZ) | `01M0BGBX3QDG6GXYP9DDMCXSFP` |
| 10 | 3 | `01M0AVPC2F771P2SZR7Y4XK4SA` | [01M0BGC0B6FKCJ4QKM6K79X40J](https://beaker.org/ex/01M0BGC0B6FKCJ4QKM6K79X40J) | `01M0BGC0ERG6B15QG8N1R1B03A` |

## Results and MoE-workspace migration (2026-08-19 03:48 UTC)

K=4 replicate 2 completed with mean reward 0.3661106339031339 and pass@1
0.38. All three K=8 runs also completed; their per-run mean rewards are
0.415860224359, 0.315660865385, and 0.386197717236, with pass@1 values 0.43,
0.33, and 0.40. The K=8 aggregate is mean reward
0.372572935660±0.051470417615 and pass@1
0.386666666667±0.051316014394. The raw and aggregate result files and shared
plot were updated.

The six unallocated K={6,10} jobs that remained queued in Holmes were
replaced in `ai2/OLMo-3-moe-experiments`. Every replacement remains urgent
and unallocated and targets exactly Jupiter and Ceres. All specs passed exact
routing, benchmark, resource, scheduling, workspace, and cluster checks; the
old Holmes copies were then canceled.

| Eval K | Rep | Old Holmes experiment | New MoE-workspace experiment | New job |
|---:|---:|---|---|---|
| 6 | 1 | `01M0BGBDQ3X4MVVG3KG8EH4649` | [01M0C21R1SP0NE0038K0SVT68J](https://beaker.org/ex/01M0C21R1SP0NE0038K0SVT68J) | `01M0C21R5FDPGJYQ4TT035Z7XT` |
| 6 | 2 | `01M0BGBJ44W4J7XP4MQG561TFP` | [01M0C21WG5BKN7TQ7RN3JXDP9J](https://beaker.org/ex/01M0C21WG5BKN7TQ7RN3JXDP9J) | `01M0C21WSJ4W3E5037VND05C3J` |
| 6 | 3 | `01M0BGBNSWSDZDQ1V2REFPH492` | [01M0C220M4RQB4RX9JW34QDJXR](https://beaker.org/ex/01M0C220M4RQB4RX9JW34QDJXR) | `01M0C220QTFG5YWSKVEFRKTH2T` |
| 10 | 1 | `01M0BGBSHQAJSZ3MDNJB5VT3Y7` | [01M0C22480TD563R6GRX8XZQVD](https://beaker.org/ex/01M0C22480TD563R6GRX8XZQVD) | `01M0C224BKYVZTRG2SYAH74NC2` |
| 10 | 2 | `01M0BGBWYS5T2845VETF32X1XZ` | [01M0C227VCMHPTCXFXVWBQVNYM](https://beaker.org/ex/01M0C227VCMHPTCXFXVWBQVNYM) | `01M0C227YWV452YAK5QA9Z4P4G` |
| 10 | 3 | `01M0BGC0B6FKCJ4QKM6K79X40J` | [01M0C22BGMFNEJ3M4FS4BCEY72](https://beaker.org/ex/01M0C22BGMFNEJ3M4FS4BCEY72) | `01M0C22BM5BSR2J55RPXB5BCCW` |

## K=4 completion (2026-08-19 05:48 UTC)

K=4 replicate 3 completed with mean reward 0.33487105056980054 and pass@1
0.35. Across all three K=4 runs, mean reward is
0.36510480056980055±0.02974359134696546 and pass@1 is 0.38±0.03. This
complete cell was added to the aggregate SFT table and shared plot. The ten
remaining phased and bridge evaluations are all running; none has failed.

## Sweep completion (2026-08-19 15:55 UTC)

All remaining phased jobs completed successfully. The final three-run
aggregates are:

| Eval K | Mean reward ± sample SD | Pass@1 ± sample SD |
|---:|---:|---:|
| 4 | 0.365104800570 ± 0.029743591347 | 0.380000000000 ± 0.030000000000 |
| 6 | 0.407261564577 ± 0.043032960285 | 0.423333333333 ± 0.045092497528 |
| 8 | 0.372572935660 ± 0.051470417615 | 0.386666666667 ± 0.051316014394 |
| 10 | 0.397825405508 ± 0.010968786050 | 0.420000000000 ± 0.010000000000 |
| 12 | 0.392068476496 ± 0.052930768811 | 0.410000000000 ± 0.050000000000 |

Every run contains all 100 tasks. The raw results, aggregate SFT table, and
shared plot are current.
