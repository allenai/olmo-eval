# Current active Beaker runs

Snapshot: 2026-08-14 23:31 UTC  
Scope: every AI2 cluster, Beaker author `jacobm`

## Adaptive-compute GPU jobs

There are no queued or running adaptive-compute GPU jobs. All eight
Terminal-Bench jobs from the previous snapshot were stopped; seven were
manually canceled and the K=8 TB2.1 shard A job had already lost its node.

| Benchmark | K | Component | Experiment | Final captured progress | Forced score for component | Final status |
|---|---:|---|---|---:|---:|---|
| TB-Lite 2.0 | 12 | rep 1, shard B | [01KZYQC5PYQ5FKSFJP0Z7SKS1A](https://beaker.org/ex/01KZYQC5PYQ5FKSFJP0Z7SKS1A) | 42/45, 3 errors | ≈62.91% | Canceled manually |
| TB2.1 | 10 | rep 1, shard B | [01KZYQD3F9H8FX88QPKDJSZ2QY](https://beaker.org/ex/01KZYQD3F9H8FX88QPKDJSZ2QY) | 41/44, 2 errors | 45.45% | Canceled manually |
| TB-Lite 2.0 | 6 | rep 3, single | [01KZYZ7ZDBS772SQB8Z0E67C17](https://beaker.org/ex/01KZYZ7ZDBS772SQB8Z0E67C17) | 76/100, 2 errors | ≈52.90% | Canceled manually |
| TB-Lite 2.0 | 8 | rep 3, single | [01KZYZ8351AGVGCJRAMPHD464G](https://beaker.org/ex/01KZYZ8351AGVGCJRAMPHD464G) | 98/100, 1 error | ≈68.21% | Canceled manually |
| TB2.1 | 4 | rep 1, shard B rerun | [01KZZ63TZZFYG2TZ1GK1CP162E](https://beaker.org/ex/01KZZ63TZZFYG2TZ1GK1CP162E) | 42/44, 1 error | 36.36% | Canceled; rerun excluded |
| TB2.1 | 6 | rep 1, shard B rerun | [01KZZCYBGKAH707VQZXVWGTK7C](https://beaker.org/ex/01KZZCYBGKAH707VQZXVWGTK7C) | 24/44, 1 error | 25.00% | Canceled; rerun excluded |
| TB2.1 | 8 | rep 1, shard A | [01KZZCYFDHZZ7HBB8TGP3YXC31](https://beaker.org/ex/01KZZCYFDHZZ7HBB8TGP3YXC31) | 43/45, 0 errors | 44.44% | Node lost/unavailable |
| TB2.1 | 8 | rep 1, shard B | [01KZZCYKAEKCM264D122B0KEKY](https://beaker.org/ex/01KZZCYKAEKCM264D122B0KEKY) | 42/44, 1 error | 36.36% | Canceled manually |

Every unfinished task is counted as a failure in the forced score. All eight
rows remain excluded from finished-only plots. Full component and combined
logical records are in `qwen36_terminal_forced_timeout_components.csv` and
`qwen36_terminal_forced_timeout_logical.csv`.

Total active adaptive-compute allocation: **0 GPUs**.

## RL status

No Slime/DAPO training job is queued or running.

| Run | Experiment | Final status |
|---|---|---|
| Mixed K={4,8,12}, resume from step 100 to 500 | [01KZP8MRHR3WXK122RXXJJKT68](https://beaker.org/ex/01KZP8MRHR3WXK122RXXJJKT68) | Succeeded, exit 0 at 2026-08-14 10:50 UTC |
| Prompt-preallocated K, resume from step 100 to 500 | [01KZP8MN9PVY2RNC84AKJ6R3Q2](https://beaker.org/ex/01KZP8MN9PVY2RNC84AKJ6R3Q2) | Succeeded, exit 0 at 2026-08-11 23:10 UTC |
| Static K=10 pilot, 100 steps | [01KZAPQVD2SVVEZ273HRDERX89](https://beaker.org/ex/01KZAPQVD2SVVEZ273HRDERX89) | Succeeded, exit 0 at 2026-08-06 11:53 UTC |

The old mixed-K continuation therefore does not need to be killed.

## Other Beaker activity owned by `jacobm`

- Two long-lived, zero-GPU control jobs are running on Hammond. They are not
  adaptive-compute experiments and consume no GPUs.
- A four-GPU scaling-ladder smoke in `ai2/linear-rnns` had been queued since
  August 11, but was manually canceled at 23:23 UTC during this audit. No action
  was taken on it here.
