#!/usr/bin/env bash

# Launch one bounded, opt-in sentinel-eval canary across the analysis model pool.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEV_LAUNCHER="${SCRIPT_DIR}/launch_dev_evals.sh"

EVAL="gpqa"
LIMIT="20"
SAMPLE_SEED="42"
RUN_TAG=""
MODE="launch"
CLUSTER="ai2/ceres"
WORKSPACE="ai2/olmo-eval-debug"

usage() {
    cat <<'EOF'
Usage: scripts/beaker/launch_sentinel_wave.sh [options]

Launch a fixed-seed canary across the eight-model analysis pool.

Options:
  --eval NAME       Sentinel group to launch (currently: gpqa; default: gpqa)
  --limit N         Per-model canary size (default: 20)
  --sample-seed N   Sampling seed (default: 42)
  --run-tag TAG     Beaker name/group suffix
  --cluster NAME    Beaker cluster (default: ai2/ceres)
  --workspace NAME  Beaker workspace (default: ai2/olmo-eval-debug)
  --print-only      Print commands without submitting
  --dry-run         Render Beaker launch specs without submitting
  -h, --help        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval) EVAL="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --sample-seed) SAMPLE_SEED="$2"; shift 2 ;;
        --run-tag) RUN_TAG="$2"; shift 2 ;;
        --cluster) CLUSTER="$2"; shift 2 ;;
        --workspace) WORKSPACE="$2"; shift 2 ;;
        --print-only) MODE="print-only"; shift ;;
        --dry-run) MODE="dry-run"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$EVAL" != gpqa ]]; then
    echo "--eval must currently be: gpqa" >&2
    exit 2
fi
if [[ ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "--limit must be a positive integer" >&2
    exit 2
fi
if [[ ! "$SAMPLE_SEED" =~ ^[0-9]+$ ]]; then
    echo "--sample-seed must be a non-negative integer" >&2
    exit 2
fi

RUN_TAG="${RUN_TAG:-gpqa-canary${LIMIT}-$(date -u +%Y%m%d)}"

# model|slug|gpus. The larger MoE checkpoints retain their validated 2-GPU
# profiles; the smaller checkpoints need only one H100 for this short task.
MODELS=(
    "allenai/Olmo-3-7B-Instruct|olmo3-7b-instruct|1"
    "allenai/Olmo-3-7B-Think|olmo3-7b-think|1"
    "Qwen/Qwen3.5-9B|qwen35-9b|1"
    "zai-org/GLM-4.1V-9B-Thinking|glm41v-9b-thinking|1"
    "openai/gpt-oss-20b|gpt-oss-20b|1"
    "google/gemma-4-26B-A4B-it|gemma4-26b-a4b-it|2"
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16|nemotron3-nano-30b-a3b|2"
    "Qwen/Qwen3.5-35B-A3B|qwen35-35b-a3b|2"
)

for spec in "${MODELS[@]}"; do
    IFS='|' read -r model slug gpus <<<"$spec"
    command=(
        "$DEV_LAUNCHER"
        --model "$model"
        --slug "$slug"
        --only "$EVAL"
        --gpqa-limit "$LIMIT"
        --sample-seed "$SAMPLE_SEED"
        --gpus "$gpus"
        --cluster "$CLUSTER"
        --workspace "$WORKSPACE"
        --run-tag "$RUN_TAG"
    )
    case "$MODE" in
        print-only) command+=(--print-only) ;;
        dry-run) command+=(--dry-run) ;;
    esac
    "${command[@]}"
done

echo
echo "Sentinel wave: ${EVAL}; ${MODE}; experiments: ${#MODELS[@]}; limit: ${LIMIT}; cluster: ${CLUSTER}; workspace: ${WORKSPACE}; run tag: ${RUN_TAG}"
