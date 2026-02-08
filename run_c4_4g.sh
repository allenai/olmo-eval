#!/usr/bin/env bash
set -euo pipefail

models=(
  deepseek-llm-67b
  llama-3.1-70b
  qwen-1.5-72b
  qwen-1.5-110b
  # qwen-1.5-moe-a2.7b
  qwen-2-72b
  qwen-2.5-32b
  qwen-2.5-72b
  huggyllama-30b
  huggyllama-65b
)

TASK_SET="c4:ppl"
CLUSTER="h100"
WORKSPACE="ai2/perplexity-evals"
BUDGET="ai2/oe-base"
PRIORITY="high"

mkdir -p logs

for m in "${models[@]}"; do
  (
    set -euo pipefail
    run_id="${m}"                 # "run id" = model name (your request)
    name="c4-bpb-${run_id}"           # experiment/run name shown in Beaker

    # Auto-accept "create missing groups?" prompts.
    yes | olmo-eval beaker launch \
      -n "${name}" \
      -m "${m}" \
      -o gpus=4 \
      -t "${TASK_SET}" \
      -c "${CLUSTER}" \
      -w "${WORKSPACE}" \
      -B "${BUDGET}" \
      -p "${PRIORITY}" \
      >"logs/${name}.log" 2>&1
  ) &
done

wait
echo "Submitted ${#models[@]} launches."
