# Clean K=8 TB-Lite comparison (2026-08-15)

This tracker supersedes the earlier protocol-mismatched TB-Lite attempts for
the two models below. Only completed runs from this group should be compared.

## Shared serving and evaluation settings

- Dataset: `openthoughts-tblite@2.0` (100 tasks for full runs)
- Native K: 8 (no routing override)
- vLLM: 0.19.1, TP=2, DP=4, 8 GPUs (four independent engines), matching
  the original completed Qwen3.6 K=4/6/8/10/12 Terminal-Bench sweep
- Four concurrent tasks for full runs; one task at a time for the smoke
- GPU-memory utilization 0.95 and one sequence per engine
- Context: 262,144 tokens; output ceiling: 81,920 tokens
- Sampling: temperature=1.0, top-p=0.95, top-k=20
- Parsers: `qwen3_xml` tool parser and `qwen3` reasoning parser
- Serving: `--language-model-only`, GDN prefill backend `triton`
- TMax agent defaults retained for agent steps, command timeout, format-error
  limit, native task resources/timeouts, and one attempt per task
- Beaker: `ai2/OLMo-3-moe-experiments`, urgent, allocated; eligible clusters
  `ai2/jupiter`, `ai2/ceres`, and `ai2/titan`

## Model-specific protocol

| Model | Agent protocol | Why |
|---|---|---|
| Qwen3.6-35B-A3B | Vanillux2 | Off-the-shelf TMax/Qwen agent protocol |
| Qwen3.5 OpenThoughts-Agent SFT step 1523 | Terminus-2, JSON parser | Matches the raw JSON action format used by OpenThoughts-Agent-SFT-100K |

## Jobs

| Model | Stage | Beaker experiment | Status | Validation/result |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | 1-task smoke 1 | [01M01C73PXWV6VX8Q5X2PAFFDE](https://beaker.org/ex/01M01C73PXWV6VX8Q5X2PAFFDE) | canceled before evaluation | Superseded during a topology provenance audit; vLLM was still initializing |
| Qwen3.5 OpenThoughts-Agent SFT step 1523 | 1-task smoke 1 | [01M01C73BFV4CT2FDS5H4RJZ3B](https://beaker.org/ex/01M01C73BFV4CT2FDS5H4RJZ3B) | canceled before evaluation | Superseded during a topology provenance audit; vLLM was still initializing |
| Qwen3.6-35B-A3B | 1-task smoke 2 | [01M01DP6NW8M81WSA0MPWY1GD1](https://beaker.org/ex/01M01DP6NW8M81WSA0MPWY1GD1) | canceled before start | Replaced with the original sweep's TP2/DP4 topology |
| Qwen3.5 OpenThoughts-Agent SFT step 1523 | 1-task smoke 2 | [01M01DP6JJCXKX8EQ0HHD4464H](https://beaker.org/ex/01M01DP6JJCXKX8EQ0HHD4464H) | canceled before start | Replaced with the original sweep's TP2/DP4 topology |
| Qwen3.6-35B-A3B | 1-task smoke 3 | [01M01DT1Z5Q27WNGSHB18FCCG5](https://beaker.org/ex/01M01DT1Z5Q27WNGSHB18FCCG5) | succeeded | qwen3_xml loaded; 64 real tool steps and verifier completed without infrastructure/protocol errors (task score 0) |
| Qwen3.5 OpenThoughts-Agent SFT step 1523 | 1-task smoke 3 | [01M01DT21YC7SJR3PC2Y3SQDYF](https://beaker.org/ex/01M01DT21YC7SJR3PC2Y3SQDYF) | canceled after diagnostic | Checkpoint served; Harbor provider alias failed before model request |
| Qwen3.5 OpenThoughts-Agent SFT step 1523 | 1-task smoke 4 | [01M01EGSX89DGCKB69NZQG42MR](https://beaker.org/ex/01M01EGSX89DGCKB69NZQG42MR) | completed, exit 0 | Protocol validated; sampled task ended in `AgentTimeoutError` and scored 0 after 15m52s |
| Qwen3.6-35B-A3B | Full TBLite run 1 | [01M01EPN6H5R9PK15ESEKC2G10](https://beaker.org/ex/01M01EPN6H5R9PK15ESEKC2G10) | completed, exit 0 | Mean reward 60.94%; pass@1 63%; 12 agent timeouts and three command-timeout runtime errors |
| Qwen3.5 OpenThoughts-Agent SFT step 1523 | Full TBLite run 1 | [01M01F5KYDYCF4FRRRDKA168ZH](https://beaker.org/ex/01M01F5KYDYCF4FRRRDKA168ZH) | running: 15/100 | Partial mean 0.688; four agent timeouts/errors; live inference active as of 02:03 UTC |

Smoke 3 established that the corrected SFT checkpoint serves successfully,
but Terminus-2's LiteLLM backend does not recognize Harbor's synthetic
`hosted_vllm/` provider prefix. Its replacement uses the equivalent
`openai/<served-name>` alias against the same local OpenAI-compatible endpoint,
with the 262K/81,920 model limits passed explicitly to Terminus-2.

Smoke 4 additionally returned valid `analysis`, `plan`, `commands`, and
`task_complete` fields under the exact Terminus-2 JSON prompt. Its commands
included correctly escaped newlines and were accepted without a provider or
serving error, so the full evaluation was promoted while the one-task smoke
continued to completion.

The completed Qwen3.6 run's lower score relative to the earlier 3-hour-timeout
attempts is consistent with timeout policy, rather than an obvious capability
regression. All 15 exceptions were timeout-related: 12 `AgentTimeoutError`s and
three 120-second command timeouts. The 85 trials without exceptions averaged
70.52% reward, close to the earlier relaxed-timeout runs' 69.86% average over
their completed trials. Including timeout failures, the new run scored 60.94%
versus 67.52% on average for the three earlier relaxed attempts when their
unfinished tasks are also assigned zero. This clean-trial comparison is
diagnostic rather than a matched causal estimate because the difficult tasks
that time out are not a random subset and the canceled older runs did not
preserve complete task-level artifacts.
