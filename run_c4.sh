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

for m in "${models[@]}"; do
  # Use the model name as the run id / unique suffix
  RUN_ID="${m}"
  job_name="c4-${RUN_ID}"

  olmo-eval beaker launch \
    -n "${job_name}" \
    -m "${m}::provider=vllm" \
    -t c4_100k:ppl \
    -c aus80g \
    -w ai2/perplexity-evals \
    --budget ai2/oe-base \
    --priority high \
    --quiet &
done
