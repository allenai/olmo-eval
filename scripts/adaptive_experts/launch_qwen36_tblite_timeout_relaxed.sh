#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMAX_LAUNCHER="${TMAX_LAUNCHER:-${ROOT_DIR}/tmax-eval/beaker_configs/launch_eval.sh}"
TMAX_REPO_REF="${TMAX_REPO_REF:-7387d2f9142397a458dc39f0827a2ab0b4c03cda}"
BEAKER_SCRIPTS_DATASET="${BEAKER_SCRIPTS_DATASET:-}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
PRIORITY="${PRIORITY:-urgent}"
RESULTS_ROOT="${RESULTS_ROOT:-/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-35b-a3b}"
DATASET="${DATASET:-openthoughts-tblite@2.0}"
CLUSTERS_CSV="${CLUSTERS:-ai2/jupiter,ai2/ceres,ai2/titan}"
STAGE=""
DRY_RUN=false

usage() {
    echo "usage: $0 --stage smoke|full [--dry-run]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

if [[ "$STAGE" != "smoke" && "$STAGE" != "full" ]]; then
    usage
    exit 2
fi
if [[ -z "$BEAKER_SCRIPTS_DATASET" ]]; then
    echo "BEAKER_SCRIPTS_DATASET must point to the uploaded timeout-aware Beaker scripts" >&2
    exit 2
fi

IFS=',' read -r -a TARGET_CLUSTERS <<< "$CLUSTERS_CSV"

launch_job() {
    local job_name="$1"
    local gpus="$2"
    local dp="$3"
    local concurrency="$4"
    shift 4

    local -a cmd=(
        "$TMAX_LAUNCHER" "$MODEL"
        --name qwen36-35b-a3b
        --job-name "$job_name"
        --results-dir "${RESULTS_ROOT}/${job_name}"
        --gpus "$gpus"
        --tp 2
        --dp "$dp"
        --workspace "$WORKSPACE"
        --priority "$PRIORITY"
        --allocated
        --repo-ref "$TMAX_REPO_REF"
        --beaker-scripts-dataset "$BEAKER_SCRIPTS_DATASET"
        --dataset "$DATASET"
        --harbor-env docker
        --agent Vanillux2Agent:Vanillux2Agent
        --n-attempts 1
        --n-concurrent "$concurrency"
        --max-model-len 262144
        --gpu-memory-utilization 0.95
        --max-num-seqs 1
        --gdn-prefill-backend triton
        --language-model-only
        --tool-call-parser qwen3_coder
        --reasoning-parser qwen3
        --hosted-vllm-model-info '{"max_input_tokens":262144,"max_output_tokens":81920,"input_cost_per_token":0.0,"output_cost_per_token":0.0}'
        --agent-kwarg temperature=1.0
        --agent-kwarg top_p=0.95
        --agent-kwarg top_k=20
        --agent-kwarg max_tokens=81920
        --agent-kwarg max_steps=64
        --agent-kwarg max_format_errors=64
        --agent-kwarg command_timeout=21600
        --disable-agent-timeout
        --timeout-multiplier 10
        --verifier-timeout-multiplier 10
        --agent-setup-timeout-multiplier 10
        --environment-build-timeout-multiplier 10
        "$@"
    )
    for cluster in "${TARGET_CLUSTERS[@]}"; do
        cmd+=(--cluster "$cluster")
    done

    if [[ "$DRY_RUN" == true ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
}

if [[ "$STAGE" == "smoke" ]]; then
    launch_job \
        qwen36-tblite-k8-native-vanillux-qwen-settings-timeout-relaxed-smoke3-tp2-c1-20260813 \
        2 1 1 \
        --include-task-name 'protein-sequence' \
        --include-task-name 'task-xxe-exploit'
    exit 0
fi

# Complementary first-letter globs partition all 100 TB-Lite 2.0 tasks into
# 55-task and 45-task shards. Each shard gets four independent TP=2 engines.
launch_job \
    qwen36-tblite-k8-native-vanillux-qwen-settings-timeout-relaxed-full100-rep1-tp2-dp4-c4-shard-a-20260813 \
    8 4 4 \
    --include-task-name '[acdegikmoswy0-9]*'
launch_job \
    qwen36-tblite-k8-native-vanillux-qwen-settings-timeout-relaxed-full100-rep1-tp2-dp4-c4-shard-b-20260813 \
    8 4 4 \
    --include-task-name '[bfhjlnpqrtuvxz]*'
