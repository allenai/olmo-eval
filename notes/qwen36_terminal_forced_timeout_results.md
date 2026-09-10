# Forced-final Terminal-Bench scores after stalled-job cancellation

Last updated: 2026-08-14 23:31 UTC

Twenty-six Qwen3.6 Terminal-Bench job components have now been stopped and
recorded. Every task absent from the last durable progress record is scored as
zero. The final eight active components were stopped at the user's request at
23:29 UTC; one of those had already lost its Beaker node.

The canceled jobs did not execute their normal result-copy epilogue, so Beaker retained only the
Gantry metadata and the task-level Harbor result trees were not recoverable. Scores are therefore
reconstructed from the last Beaker log heartbeat. TB-Lite heartbeats print their mean to three
decimal places, making the forced score accurate to about 0.05 percentage points. TB2.1 rewards are
binary, so the number correct is recovered exactly by rounding `completed × logged mean` to the
nearest integer.

## Retained logical forced-final results

| Benchmark | K | Replicate | Completed / expected | Task errors | Missing → failure | Forced-final score |
|---|---:|---:|---:|---:|---:|---:|
| TB-Lite | 4 | 2 | 99 / 100 | 1 | 1 | ≈60.49% |
| TB-Lite | 6 | 1 | 95 / 100 | 0 | 5 | ≈62.81% |
| TB-Lite | 6 | 2 | 99 / 100 | 2 | 1 | ≈63.86% |
| TB-Lite | 8 | 2 | 97 / 100 | 1 | 3 | ≈66.35% |
| TB-Lite | 10 | 1 | 97 / 100 | 3 | 3 | ≈67.91% |

The complete K=10 replicate 1 result (67.91%) remains usable. The following forced-final rows are
preserved for provenance but excluded from analysis and plots:

| Benchmark | K | Replicate | Completed / expected | Forced-final score | Exclusion reason |
|---|---:|---:|---:|---:|---|
| TB-Lite | 10 | 2 | 48 / 100 | ≈37.20% | Pathological stall left 52 tasks unfinished |
| TB2.1 | 12 | 1 | 86 / 89 | 35.96% | Frozen-shard forced result considered unreliable |

The abnormally low K=10 TB-Lite replicate 2 score is driven by 52 unfinished tasks and is a
timeout-policy artifact, not an estimate of model capability. The K=12 TB2.1 forced score is also
excluded because its frozen shards did not finish through the normal result-writing path.

The final stopped components produce the following additional logical records.
All are excluded from finished-only plots, including the shard-B reruns the
user judged non-comparable.

| Benchmark | K | Replicate | Completed / expected | Missing → failure | Forced-final score |
|---|---:|---:|---:|---:|---:|
| TB-Lite | 6 | 3 | 76 / 100 | 24 | ≈52.90% |
| TB-Lite | 8 | 3 | 98 / 100 | 2 | ≈68.21% |
| TB-Lite | 12 | 1 | 96 / 100 | 4 | ≈66.65% |
| TB2.1 | 4 | 1 | 86 / 89 | 3 | 34.83% |
| TB2.1 | 6 | 1 | 68 / 89 | 21 | 34.83% |
| TB2.1 | 8 | 1 | 85 / 89 | 4 | 40.45% |
| TB2.1 | 10 | 1 | 86 / 89 | 3 | 40.45% |

Component-level forced scores and all Beaker IDs are preserved in
`qwen36_terminal_forced_timeout_components.csv`. Combined logical scores are
in `qwen36_terminal_forced_timeout_logical.csv`. These forced-final rows remain
separate from strict complete-run plots so their timeout treatment is explicit.
