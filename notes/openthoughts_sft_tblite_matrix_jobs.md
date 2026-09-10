# OpenThoughts-Agent SFT train-K x eval-K TBLite matrix (2026-08-15)

## Design

- Checkpoints: the step-1523 OpenThoughts-Agent 100K models trained at
  K=4, K=8, and K=12.
- Evaluated expert counts: K=4, K=8, and K=12 for every checkpoint.
- Replication: three runs per train-K/eval-K cell. The prior native-K run is
  replicate 1 on each diagonal; this launch adds diagonal replicates 2 and 3
  plus three runs for every off-diagonal cell.
- New-run seeds: replicate 1/2/3 use 4202/4203/4204. The three earlier
  diagonal runs predate this paired-seed convention.
- Scaling: all evaluated policies use K=8 as the reference router-weight
  scale, without unit-sum renormalization.
- Dataset and agent: `openthoughts-tblite@2.0`, Terminus-2 JSON agent,
  `qwen3_xml` tool parser, and `qwen3` reasoning parser.
- Generation: temperature 1.0, top-p 0.95, top-k 20, 81,920-token output
  ceiling, and 262,144-token context.
- Topology: one 8-GPU H100 node per run, TP=2, DP=4, four concurrent tasks.
- Beaker: `ai2/OLMo-3-moe-experiments`, urgent, allocated, with only
  `ai2/jupiter` and `ai2/ceres` eligible.

## Job matrix

Every cell below will have three total runs. “Prior” denotes the already
completed native-K result; all links marked rep1/2/3 in off-diagonal cells and
rep2/3 on diagonal cells were submitted in this expansion.

| Train K | Eval K | Replicate 1 | Replicate 2 | Replicate 3 |
|---:|---:|---|---|---|
| 4 | 4 | [prior: 01M01SPVGH66S3GR7PAZZBBTHF](https://beaker.org/ex/01M01SPVGH66S3GR7PAZZBBTHF) | [01M03WCDV47YYG4F620RZE098X](https://beaker.org/ex/01M03WCDV47YYG4F620RZE098X) | [01M03WCH7JKAFPMYDREV828673](https://beaker.org/ex/01M03WCH7JKAFPMYDREV828673) |
| 4 | 8 | [01M03WCN1C9XS23ZNHDP9HZQ5M](https://beaker.org/ex/01M03WCN1C9XS23ZNHDP9HZQ5M) | [01M03WCRMXNCJGZE60C3JJSMGQ](https://beaker.org/ex/01M03WCRMXNCJGZE60C3JJSMGQ) | [01M03WCW2D2TQDB516MZ1MXP7P](https://beaker.org/ex/01M03WCW2D2TQDB516MZ1MXP7P) |
| 4 | 12 | [01M03WCZGNKJB2R1K505DNZ1A3](https://beaker.org/ex/01M03WCZGNKJB2R1K505DNZ1A3) | [01M03WD32VJ4G8SHZ002RB5B4R](https://beaker.org/ex/01M03WD32VJ4G8SHZ002RB5B4R) | [01M03WD6WMJATYSQZAPN9P8S4E](https://beaker.org/ex/01M03WD6WMJATYSQZAPN9P8S4E) |
| 8 | 4 | [01M03WDACJA3ENGTJS64F9RKJ4](https://beaker.org/ex/01M03WDACJA3ENGTJS64F9RKJ4) | [01M03WDDWN383P09FEM3E9Q1XG](https://beaker.org/ex/01M03WDDWN383P09FEM3E9Q1XG) | [01M03WDHVPJD1CS6A8H5Y44XQ1](https://beaker.org/ex/01M03WDHVPJD1CS6A8H5Y44XQ1) |
| 8 | 8 | [prior: 01M01F5KYDYCF4FRRRDKA168ZH](https://beaker.org/ex/01M01F5KYDYCF4FRRRDKA168ZH) | [01M03WDS8YQR8BA5E4VMGSDN9W](https://beaker.org/ex/01M03WDS8YQR8BA5E4VMGSDN9W) | [01M03WDWVB92WMKBFRJCJP9QMH](https://beaker.org/ex/01M03WDWVB92WMKBFRJCJP9QMH) |
| 8 | 12 | [01M03WE4XCFMGG2P3ZGWH7W6SY](https://beaker.org/ex/01M03WE4XCFMGG2P3ZGWH7W6SY) | [01M03WE8G2BQ5E3K8ZACPREFCF](https://beaker.org/ex/01M03WE8G2BQ5E3K8ZACPREFCF) | [01M03WEC9ND9VV8T3YPSESYYP4](https://beaker.org/ex/01M03WEC9ND9VV8T3YPSESYYP4) |
| 12 | 4 | [01M03WEFQAME2QVTFT7VM63Q1M](https://beaker.org/ex/01M03WEFQAME2QVTFT7VM63Q1M) | [01M03WEK8ZY28TTTGGVJWYEMBF](https://beaker.org/ex/01M03WEK8ZY28TTTGGVJWYEMBF) | [01M03WEPX6DJ2KER9F4J85QDZ1](https://beaker.org/ex/01M03WEPX6DJ2KER9F4J85QDZ1) |
| 12 | 8 | [01M03WETH6B5KGMK3QV5AS048P](https://beaker.org/ex/01M03WETH6B5KGMK3QV5AS048P) | [01M03WEYQGQB2196HM9BJG8JX6](https://beaker.org/ex/01M03WEYQGQB2196HM9BJG8JX6) | [01M03WF2CH5DVFSP7AFHTKBCB9](https://beaker.org/ex/01M03WF2CH5DVFSP7AFHTKBCB9) |
| 12 | 12 | [prior: 01M01S1G4W3ABPVYQHTJJ2E3EW](https://beaker.org/ex/01M01S1G4W3ABPVYQHTJJ2E3EW) | [01M03WF5WHGKJP6YGWQBZRYFQG](https://beaker.org/ex/01M03WF5WHGKJP6YGWQBZRYFQG) | [01M03WF9FS41YB4VEX5X7W0D9Z](https://beaker.org/ex/01M03WF9FS41YB4VEX5X7W0D9Z) |

The machine-readable launch records, including Beaker job IDs and exact
routing/config paths, are in `beaker_jobs.jsonl` under groups
`openthoughts-sft-tblite-matrix-20260815` and
`qwen36-tb21-reference-replicates-20260815`.

## Final results (2026-08-16)

All 27 runs are complete: three runs in each train-K/eval-K cell. Values are
mean TBLite reward and pass@1, in percent, with sample standard deviations.

| Train K | Eval K | Mean reward | Pass@1 |
|---:|---:|---:|---:|
| 4 | 4 | 32.43 ± 1.86 | 34.00 ± 1.73 |
| 4 | 8 | 39.24 ± 4.25 | 41.00 ± 4.00 |
| 4 | 12 | 42.26 ± 5.51 | 44.00 ± 5.57 |
| 8 | 4 | 31.41 ± 4.05 | 33.33 ± 3.51 |
| 8 | 8 | 39.40 ± 5.29 | 41.00 ± 5.57 |
| 8 | 12 | 37.03 ± 2.00 | 38.67 ± 2.08 |
| 12 | 4 | 31.85 ± 4.65 | 34.00 ± 5.20 |
| 12 | 8 | 40.90 ± 7.31 | 43.00 ± 7.94 |
| 12 | 12 | 39.92 ± 2.99 | 41.67 ± 2.89 |

The dominant effect is evaluation-time K, not training-time K: every model is
near 31--32% mean reward at eval K=4, while eval K=8 or K=12 is generally
better. The train-K=4 model improves monotonically as eval K rises. The
train-K=8 and train-K=12 differences between eval K=8 and K=12 are smaller
than their run-to-run variation. These results do not show that low-K SFT by
itself makes the model robust at eval K=4.

## K=6/K=10 backfill

On 2026-08-17, 18 additional jobs were launched to evaluate every fixed-SFT
checkpoint at K=6 and K=10 with three replicates. This completes the intended
five-point inference grid K={4,6,8,10,12} and makes the fixed checkpoints
directly comparable with the mixed-K SFT checkpoint. All submitted configs
passed exact routing validation; links are in
`openthoughts_sft_tblite_bridge_jobs.md`.

## Cross-model plot

The completed fixed-K matrix and mixed-K checkpoint results are plotted
together in `plots/qwen35_openthoughts_sft_tblite_expert_sweep.{png,svg}`.
The figure uses TBLite mean reward with ±1 sample-SD shading and excludes all
partial K=6/K=10 backfill and mixed-checkpoint extension runs. Its consolidated
machine-readable input is `openthoughts_all_sft_tblite_results.csv`.
