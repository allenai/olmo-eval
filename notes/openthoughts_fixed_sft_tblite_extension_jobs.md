# OpenThoughts fixed-K SFT TBLite extension

Launched 2026-08-18 for the three fixed-K OpenThoughts-Agent SFT checkpoints:

- train K=4: `/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints/qwen35-openthoughts-agent-100k-k4-refk8-lr2e-5-flce-b64-20260813/step1523-hf`
- train K=8: `/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints/qwen35-openthoughts-agent-100k-k8-refk8-lr2e-5-flce-b64-20260813/step1523-hf`
- train K=12: `/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints/qwen35-openthoughts-agent-100k-k12-refk8-lr2e-5-flce-b64-20260813/step1523-hf`

This adds eval K={2,14,16}, with three paired-seed TBLite replicates for
every train-K/eval-K cell (27 jobs total). Together with the existing matrix,
this gives comparable in-distribution train-K points and lower/higher-K OOD
points for all three fixed-SFT models.

All settings match the earlier SFT matrix: `openthoughts-tblite@2.0`,
reference-K=8 scaling with `renormalize=false`, Terminus-2 JSON, Qwen3 XML
tool parsing, Qwen3 reasoning parsing, temperature 1.0, top-p 0.95, top-k 20,
262,144-token context, and an 81,920-token generation ceiling. Each run
requests one 8xH100 node with TP=2 and DP=4 at urgent allocated priority in
`ai2/OLMo-3-moe-experiments`, targeting Jupiter and Ceres.

K=2 routes the top eight and truncates to two. K={14,16} sets both the nested
Qwen3.5 text configuration and the runtime router width to the requested K.
All 27 submitted Beaker specs passed exact checks for keep-K, router-K,
reference-K, renormalization, config override, parsers, dataset, resources,
priority, and cluster targeting. The jobs also retain the fail-closed live
router assertion used by the validated sweep.

Results root:
`/weka/oe-adapt-default/jacobm/tmax-eval/openthoughts-sft-fixed-tblite-extension-20260818`

| Train K | Eval K | Rep | Seed | Beaker experiment | Beaker job | Initial state |
|---:|---:|---:|---:|---|---|---|
| 4 | 2 | 1 | 4202 | [01M09HATW596H177Q2J8X1P3W0](https://beaker.org/ex/01M09HATW596H177Q2J8X1P3W0) | `01M09HATZSF7GKPG0BKRPRXE3Q` | queued |
| 4 | 2 | 2 | 4203 | [01M09HB1WHW3VHXKD2586YVDFT](https://beaker.org/ex/01M09HB1WHW3VHXKD2586YVDFT) | `01M09HB200CMX68MQPARVND9KZ` | queued |
| 4 | 2 | 3 | 4204 | [01M09HB8PZXPKE5X1C2MK2M7JG](https://beaker.org/ex/01M09HB8PZXPKE5X1C2MK2M7JG) | `01M09HB8TE8ZCB4TJWSK6RCWV4` | queued |
| 4 | 14 | 1 | 4202 | [01M09HBCYNSX46RWA6VX6A0R2S](https://beaker.org/ex/01M09HBCYNSX46RWA6VX6A0R2S) | `01M09HBD2MPF6RGBEV77BDJ80G` | queued |
| 4 | 14 | 2 | 4203 | [01M09HBN9P7376ZF1407M8B1RF](https://beaker.org/ex/01M09HBN9P7376ZF1407M8B1RF) | `01M09HBNDBKT69GY8EPQ9K5WW1` | queued |
| 4 | 14 | 3 | 4204 | [01M09HBWEKFT062TYE1V0MJ6M8](https://beaker.org/ex/01M09HBWEKFT062TYE1V0MJ6M8) | `01M09HBWJ696PCR63MKPN308MC` | queued |
| 4 | 16 | 1 | 4202 | [01M09HC0KWBVZMWN7SFZKVE7CJ](https://beaker.org/ex/01M09HC0KWBVZMWN7SFZKVE7CJ) | `01M09HC0Q6STXVSC0SZN8Q0E73` | queued |
| 4 | 16 | 2 | 4203 | [01M09HC4FGX75W4PCVAVMRD0RP](https://beaker.org/ex/01M09HC4FGX75W4PCVAVMRD0RP) | `01M09HC4JY6FA22F6JDBYHW09X` | queued |
| 4 | 16 | 3 | 4204 | [01M09HC8884XJHZ3PR09BVVTEN](https://beaker.org/ex/01M09HC8884XJHZ3PR09BVVTEN) | `01M09HC8D79NH76QB85JBSCYA4` | queued |
| 8 | 2 | 1 | 4202 | [01M09HCDEXAG5ST6JZDQGAENYQ](https://beaker.org/ex/01M09HCDEXAG5ST6JZDQGAENYQ) | `01M09HCDJCA299GV1586FXXEC7` | queued |
| 8 | 2 | 2 | 4203 | [01M09HCPP035NX17EYSJM6YZMT](https://beaker.org/ex/01M09HCPP035NX17EYSJM6YZMT) | `01M09HCPSFFQXKAJ1VNFXG3AWJ` | queued |
| 8 | 2 | 3 | 4204 | [01M09HCTBT298DEJP49FJVG70N](https://beaker.org/ex/01M09HCTBT298DEJP49FJVG70N) | `01M09HCTGGMVZJZ647BSE4S0JD` | queued |
| 8 | 14 | 1 | 4202 | [01M09HCYCT5BK6P9A6TFK5XBQR](https://beaker.org/ex/01M09HCYCT5BK6P9A6TFK5XBQR) | `01M09HCYH2MFDCJ472C1HNW4TJ` | queued |
| 8 | 14 | 2 | 4203 | [01M09HD38GG091Y9TMT20TR5J9](https://beaker.org/ex/01M09HD38GG091Y9TMT20TR5J9) | `01M09HD3D2CC0Y39XJXSR7F3VJ` | queued |
| 8 | 14 | 3 | 4204 | [01M09HD7EGDFRFSASAV7VVX8Z6](https://beaker.org/ex/01M09HD7EGDFRFSASAV7VVX8Z6) | `01M09HD7HX85M1EMZ1BTVWA7T6` | queued |
| 8 | 16 | 1 | 4202 | [01M09HDDZZE8ZD8T2E29ZGPSWE](https://beaker.org/ex/01M09HDDZZE8ZD8T2E29ZGPSWE) | `01M09HDE3Y9QVMJ2N31YSE1J7B` | queued |
| 8 | 16 | 2 | 4203 | [01M09HDHWW9KDJEM3VW8R8566C](https://beaker.org/ex/01M09HDHWW9KDJEM3VW8R8566C) | `01M09HDJ0Z58QK2YX3PQ454A04` | queued |
| 8 | 16 | 3 | 4204 | [01M09HDQS645BFBB2QNPFJXPMJ](https://beaker.org/ex/01M09HDQS645BFBB2QNPFJXPMJ) | `01M09HDQWWSA2RV1Y6EV1AXR30` | queued |
| 12 | 2 | 1 | 4202 | [01M09HDZRCVTVDDE0MJ3T6J7JC](https://beaker.org/ex/01M09HDZRCVTVDDE0MJ3T6J7JC) | `01M09HDZVP5YD6X594HT1NR8HC` | queued |
| 12 | 2 | 2 | 4203 | [01M09HE4730SNHJ22EAHF4YG4M](https://beaker.org/ex/01M09HE4730SNHJ22EAHF4YG4M) | `01M09HE4B7EMVD4PPZMHSDYMNT` | queued |
| 12 | 2 | 3 | 4204 | [01M09HE8HH64QF7M09A76PB55X](https://beaker.org/ex/01M09HE8HH64QF7M09A76PB55X) | `01M09HE8P62DQE2Z3KSJ4AJWWA` | queued |
| 12 | 14 | 1 | 4202 | [01M09HED2WZ4QRMM3Q41SC4FG9](https://beaker.org/ex/01M09HED2WZ4QRMM3Q41SC4FG9) | `01M09HEDR26NWJKX012STMQJ8H` | queued |
| 12 | 14 | 2 | 4203 | [01M09HEHAABWZ9JNNWNHJBH39A](https://beaker.org/ex/01M09HEHAABWZ9JNNWNHJBH39A) | `01M09HEHDVN0MQWWB33AWA4E82` | queued |
| 12 | 14 | 3 | 4204 | [01M09HER12S359TWKG36V91CZX](https://beaker.org/ex/01M09HER12S359TWKG36V91CZX) | `01M09HER4KTY01FGF33FPY9P0H` | queued |
| 12 | 16 | 1 | 4202 | [01M09HEVJC01VFK5YHEFNV8BS0](https://beaker.org/ex/01M09HEVJC01VFK5YHEFNV8BS0) | `01M09HEVNR0M8D3DRBR3SPNV5K` | queued |
| 12 | 16 | 2 | 4203 | [01M09HEZXGQWBXMMPXV6RS8CNQ](https://beaker.org/ex/01M09HEZXGQWBXMMPXV6RS8CNQ) | `01M09HF013NNQR6N5X8XN5PB09` | queued |
| 12 | 16 | 3 | 4204 | [01M09HF857W8VDHFT5FEM02QJ9](https://beaker.org/ex/01M09HF857W8VDHFT5FEM02QJ9) | `01M09HF88KBK83D2RPKD01R7XY` | queued |

Launcher: `scripts/adaptive_experts/launch_openthoughts_sft_tblite_matrix.sh`

## Queue migration on 2026-08-18

The original train-K=4/eval-K=2/replicate-1 job had already started and was
left running. All 26 queued jobs, plus the 26 short-lived queued replacements
created during the first reorder, were canceled before allocation. Because
this matrix only contains eval K={2,14,16}, none was submitted in
`ai2/holmes-testing`; these OOD points are intentionally deferred.

The lone running train-K=4/eval-K=2/replicate-1 job was subsequently stopped
at 16:49 UTC after completing 2/100 trials. Its partial result is excluded.
There is therefore no remaining active fixed-SFT K=2 evaluation.
