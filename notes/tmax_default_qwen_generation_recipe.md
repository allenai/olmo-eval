# TMax-default Qwen3.6 terminal-evaluation recipe

Status: prepared and locally validated as a dry run on 2026-08-14. No Beaker
jobs have been submitted from this recipe.

Launcher:
`scripts/adaptive_experts/launch_qwen36_tmax_defaults_qwen_generation.sh`

## What is changed

Only the Qwen thinking-mode generation envelope is overridden:

- temperature: `1.0`
- top-p: `0.95`
- top-k: `20`
- context/model input ceiling: `262144` tokens
- per-call output ceiling: `81920` tokens

The expert-count option remains available for the intended adaptive-compute
comparison. K=8 is native. Other K values use the same reference-scaled expert
truncation policy as the previous Qwen3.6 sweep.

## What stays at the TMax Beaker-launcher defaults

- `Vanillux2Agent`, with its default 64 steps and 64 format-error allowance
- task-native Harbor timeouts and resource settings
- 120-second shell-command timeout
- one attempt per task
- eight concurrent tasks
- one eight-GPU vLLM engine (`TP=8`, `DP=1`)
- vLLM 0.19.1, 0.85 GPU-memory utilization, and prefix caching
- the launcher's default tool parser (`hermes`), with no reasoning parser
- no `language_model_only`, `max_num_seqs`, uniform timeout, timeout multiplier,
  custom command timeout, task-resource override, or manual task sharding

The Beaker workspace, priority, and cluster list are scheduling choices rather
than eval-protocol changes. The prepared defaults are urgent in
`ai2/OLMo-3-moe-experiments`, targeting Jupiter, Ceres, and Titan.

## Datasets

- TB-Lite: `openthoughts-tblite@2.0` (100 tasks)
- TB2.1: pinned Harbor dataset
  `terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a`
  (89 tasks)

The only benchmark-specific change is the dataset identifier.

## Safe usage

Both commands below print their launch commands but do not submit anything:

```bash
bash scripts/adaptive_experts/launch_qwen36_tmax_defaults_qwen_generation.sh \
  --benchmark tblite --k 8
bash scripts/adaptive_experts/launch_qwen36_tmax_defaults_qwen_generation.sh \
  --benchmark tb21 --k 8
```

An explicit `--launch` is required to submit. Before a full run, use a one-task
smoke to verify that the generic TMax `hermes` tool parser is compatible with
Qwen3.6. The TMax README's manual Qwen-family example instead uses
`qwen3_xml`, while our earlier Qwen3.6 runs used `qwen3_coder`; changing the
parser would be a model-serving compatibility correction, but would no longer
be a literal invocation of the generic Beaker-launcher defaults.

## SFT checkpoint compatibility

TMax/Harbor can be used as the evaluation infrastructure for the OpenThoughts
Agent SFT checkpoint, but the default Vanillux2 agent is not the correct agent
protocol for that model. The checkpoint emits Terminus-2-style JSON commands;
in the validated Vanillux smoke, it generated repeatedly but Vanillux rejected
all 64 commands as malformed tool calls.

The honest evaluation path is TMax infrastructure with Harbor's `terminus-2`
agent and JSON parser. The current converted checkpoint can be served through
our compatibility shim, but a clean reconversion should retain the official
outer `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe` layout and the
`model.language_model.*` tensor namespace. After reconversion, run a one-task
Terminus-2 smoke before the full TB-Lite suite.
