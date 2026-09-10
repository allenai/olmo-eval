#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMAX_LAUNCHER="${TMAX_LAUNCHER:-${ROOT_DIR}/tmax-eval/beaker_configs/launch_eval.sh}"
TMAX_REPO_REF="${TMAX_REPO_REF:-7387d2f9142397a458dc39f0827a2ab0b4c03cda}"
BEAKER_SCRIPTS_DATASET="${BEAKER_SCRIPTS_DATASET:-01KZFFNSKNAF0ARQ76R3G37EHY}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
PRIORITY="${PRIORITY:-urgent}"
RESULTS_ROOT="${RESULTS_ROOT:-/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-35b-a3b}"
JOB_NAME="${JOB_NAME:-qwen36-tb21-k8-native-terminus2-podman-canonical-smoke1-tp2-262k-20260813}"
HARBOR_ENV="${HARBOR_ENV:-docker}"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

# This is the explicit expansion of olmo-eval's `80g` selector plus Titan.
# Gantry accepts repeated --cluster constraints; it does not know olmo-eval's
# alias table itself.
CLUSTERS=(
    ai2/jupiter
    ai2/saturn
    ai2/ceres
    ai2/titan
)

cmd=(
    "$TMAX_LAUNCHER" "$MODEL"
    --name qwen36-35b-a3b
    --job-name "$JOB_NAME"
    --results-dir "${RESULTS_ROOT}/${JOB_NAME}"
    --gpus 2
    --tp 2
    --dp 1
    --workspace "$WORKSPACE"
    --priority "$PRIORITY"
    --allocated
    --repo-ref "$TMAX_REPO_REF"
    --beaker-scripts-dataset "$BEAKER_SCRIPTS_DATASET"
    --dataset terminal-bench/terminal-bench-2-1
    --harbor-env "$HARBOR_ENV"
    --agent terminus-2
    --n-attempts 1
    --n-concurrent 1
    --n-tasks 1
    --max-model-len 262144
    --gpu-memory-utilization 0.95
    --max-num-seqs 1
    --gdn-prefill-backend triton
    --language-model-only
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --hosted-vllm-model-info '{"max_input_tokens":262144,"max_output_tokens":81920,"input_cost_per_token":0.0,"output_cost_per_token":0.0}'
    --agent-kwarg parser_name=json
    --agent-kwarg temperature=1.0
    --agent-kwarg 'llm_call_kwargs={"max_tokens":81920,"top_p":0.95,"extra_body":{"top_k":20}}'
)

for cluster in "${CLUSTERS[@]}"; do
    cmd+=(--cluster "$cluster")
done

if [[ "$DRY_RUN" == true ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
else
    "${cmd[@]}"
fi
