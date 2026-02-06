#!/usr/bin/env bash
set -euo pipefail

models=(
  llama-3-8b
  llama-3.1-8b
  llama-3.2-1b
  llama-3.2-3b
  yi-1.5-34b
  yi-1.5-9b
  yi-1.5-6b
  yi-34b
  yi-9b
  yi-6b
  olmo-1b-0724
  olmo-2-32b
  olmo-2-1b
  olmo-7b-0424
  olmo-7b-0724
  olmo-1b
  olmo-7b
  olmo-7b-twin-2t
  olmo-2-7b
  olmo-2-13b
  deepseek-llm-7b
  deepseek-moe-16b
  deepseek-v2-lite
  gemma-2-27b
  gemma-2-9b
  gemma-2-2b
  gemma-2b
  gemma-7b
  gemma-3-12b-pt
  gemma-3-4b-pt
  smollm-2-1.7b
  smollm-1.7b
  marin-8b
  orca-2-13b
  orca-2-7b
  phi-4
  phi-1.5
  phi-1
  codestral-22b
  mathstral-7b
  mistral-7b
  mistral-7b-v0.1
  mixtral-8x7b
  mistral-nemo-base
  mistral-small-24b
  mistral-small-3.1-24b
  codeqwen-1.5-7b
  qwen-1.5-0.5b
  qwen-1.5-1.8b
  qwen-1.5-4b
  qwen-1.5-7b
  qwen-1.5-32b
  qwen-2-0.5b
  qwen-2-1.5b
  qwen-2-7b
  qwen-2.5-0.5b
  qwen-2.5-1.5b
  qwen-2.5-3b
  qwen-2.5-7b
  qwen-2.5-14b
  qwen-3-0.6b
  qwen-3-1.7b
  qwen-3-4b
  qwen-3-8b
  qwen-3-14b
  stablelm-2-1.6b
  stablelm-base-alpha-7b
  falcon-3-10b
  falcon-3-7b
  falcon-3-3b
  falcon-3-1b
  pythia-14m
  pythia-70m
  pythia-160m
  pythia-410m
  pythia-1b
  pythia-1.4b
  pythia-6.9b
  pythia-12b
  huggyllama-7b
  huggyllama-13b
  aquila-7b
)

TASK_SET="c4:ppl"
CLUSTER="aus80g"
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
