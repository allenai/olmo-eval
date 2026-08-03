#!/usr/bin/env bash

# Launch one analysis replica across the supported evaluation matrix.
# FrontierScience and DeepScholar are intentionally excluded.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEV_LAUNCHER="${SCRIPT_DIR}/launch_dev_evals.sh"

REPLICA=""
RUN_TAG=""
MODE="launch"

usage() {
    cat <<'EOF'
Usage: scripts/beaker/launch_replica_wave.sh --replica N [options]

Launches 35 Beaker experiments:
  - 16 full jobs: Core and MMLU for all eight included models
  - 19 fixed-dev agentic jobs for supported model/harness combinations

FrontierScience, DeepScholar, Gemma4 12B, and unsupported tool-harness cells
are excluded.

Options:
  --replica N      Positive analysis replica number (required)
  --run-tag TAG    Beaker name/group suffix
  --print-only     Print all commands without submitting
  --dry-run        Render launch specs without submitting
  -h, --help       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --replica) REPLICA="$2"; shift 2 ;;
        --run-tag) RUN_TAG="$2"; shift 2 ;;
        --print-only) MODE="print-only"; shift ;;
        --dry-run) MODE="dry-run"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! "$REPLICA" =~ ^[1-9][0-9]*$ ]]; then
    echo "--replica must be a positive integer" >&2
    exit 2
fi

RUN_TAG="${RUN_TAG:-analysis-replica${REPLICA}-$(date -u +%Y%m%d)}"

MODELS=(
    "allenai/Olmo-3-7B-Instruct|olmo3-7b-instruct"
    "allenai/Olmo-3-7B-Think|olmo3-7b-think"
    "Qwen/Qwen3.5-9B|qwen35-9b"
    "zai-org/GLM-4.1V-9B-Thinking|glm41v-9b-thinking"
    "openai/gpt-oss-20b|gpt-oss-20b"
    "google/gemma-4-26B-A4B-it|gemma4-26b-a4b-it"
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16|nemotron3-nano-30b-a3b"
    "Qwen/Qwen3.5-35B-A3B|qwen35-35b-a3b"
)

# model|slug|supported agentic groups
AGENTIC=(
    "allenai/Olmo-3-7B-Instruct|olmo3-7b-instruct|paper,sage"
    "allenai/Olmo-3-7B-Think|olmo3-7b-think|paper,sage"
    "Qwen/Qwen3.5-9B|qwen35-9b|paper,sage,expertqa"
    "openai/gpt-oss-20b|gpt-oss-20b|paper,sage,expertqa"
    "google/gemma-4-26B-A4B-it|gemma4-26b-a4b-it|paper,sage,expertqa"
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16|nemotron3-nano-30b-a3b|paper,sage,expertqa"
    "Qwen/Qwen3.5-35B-A3B|qwen35-35b-a3b|paper,sage,expertqa"
)

launch_group() {
    local model="$1"
    local slug="$2"
    local group="$3"
    local command=(
        "$DEV_LAUNCHER"
        --model "$model"
        --slug "$slug"
        --only "$group"
        --sample-seed 42
        --run-tag "$RUN_TAG"
    )
    case "$MODE" in
        print-only) command+=(--print-only) ;;
        dry-run) command+=(--dry-run) ;;
    esac
    "${command[@]}"
}

for spec in "${MODELS[@]}"; do
    IFS='|' read -r model slug <<<"$spec"
    launch_group "$model" "$slug" small
done

for spec in "${AGENTIC[@]}"; do
    IFS='|' read -r model slug groups <<<"$spec"
    IFS=',' read -ra selected_groups <<<"$groups"
    for group in "${selected_groups[@]}"; do
        launch_group "$model" "$slug" "$group"
    done
done

echo
echo "Replica ${REPLICA}: ${MODE}; expected experiments: 35; run tag: ${RUN_TAG}"
