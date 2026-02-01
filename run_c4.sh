#!/usr/bin/env bash
set -euo pipefail

models=(
  yi-1.5-6b
  yi-1.5-9b
#   yi-6b
#   yi-9b
#   deepseek-llm-7b-base
#   gemma2-9b
#   gemma-7b
#   marin-8b-base
#   llama3-8b
#   orca-2-7b
#   mathstral-7b
#   mistral-7b-v0.1
#   codeqwen1.5-7b
#   qwen1.5-7b
#   qwen2-7b
#   qwen3-8b
#   stablelm-base-alpha-7b
#   falcon3-7b
#   falcon3-10b
#   olmo-7b-0424
#   olmo-7b-0724
#   olmo-7b-hf
#   olmo-7b-twin-2t
#   olmo-2-7b-stage1-step928646
#   olmo-2-7b-stage2-ingredient1
#   olmo-2-7b-stage2-ingredient2
#   olmo-2-7b-stage2-ingredient3
#   pythia-6.9b
#   llama-7b
#   aquila-7b
)

TASK_SET="c4_100k:ppl"
CLUSTER="aus80g"
WORKSPACE="ai2/perplexity-evals"
BUDGET="ai2/oe-base"
PRIORITY="high"

mkdir -p logs

for m in "${models[@]}"; do
  (
    set -euo pipefail
    run_id="${m}"                 # "run id" = model name (your request)
    name="c4-${run_id}"           # experiment/run name shown in Beaker

    # Auto-accept "create missing groups?" prompts.
    yes | olmo-eval beaker launch \
      -n "${name}" \
      -m "${m}::provider=vllm" \
      -t "${TASK_SET}" \
      -c "${CLUSTER}" \
      -w "${WORKSPACE}" \
      --budget "${BUDGET}" \
      --priority "${PRIORITY}" \
      >"logs/${name}.log" 2>&1
  ) &
done

wait
echo "Submitted ${#models[@]} launches."
