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
CLUSTERS_CSV="${CLUSTERS:-ai2/jupiter,ai2/ceres,ai2/titan}"
EXPERT_COUNTS="${EXPERT_COUNTS:-4 6 10 12}"
REPLICATE="${REPLICATE:-1}"
AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-8100}"
COMMAND_TIMEOUT_SEC="${COMMAND_TIMEOUT_SEC:-2700}"
SHARD_MODE="${SHARD_MODE:-split}"
SHARDS="${SHARDS:-a b}"
RUN_DATE="${RUN_DATE:-$(date -u +%Y%m%d)}"
STAGE=""
DRY_RUN=false

usage() {
    echo "usage: $0 --stage tblite|tb21 [--dry-run]" >&2
    echo "  SHARD_MODE=split|single (default: split)" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

if [[ "$STAGE" != "tblite" && "$STAGE" != "tb21" ]]; then
    usage
    exit 2
fi
if [[ "$SHARD_MODE" != "split" && "$SHARD_MODE" != "single" ]]; then
    echo "unsupported shard mode: $SHARD_MODE" >&2
    exit 2
fi
if [[ -z "$BEAKER_SCRIPTS_DATASET" ]]; then
    echo "BEAKER_SCRIPTS_DATASET must point to the uploaded timeout-aware Beaker scripts" >&2
    exit 2
fi

case "$STAGE" in
    tblite)
        DATASET="openthoughts-tblite@2.0"
        TASK_COUNT=100
        SHARD_A='[acdegikmoswy0-9]*'
        SHARD_B='[bfhjlnpqrtuvxz]*'
        ;;
    tb21)
        DATASET="terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
        TASK_COUNT=89
        SHARD_A='terminal-bench/[acdegikmoswy0-9]*'
        SHARD_B='terminal-bench/[bfhjlnpqrtuvxz]*'
        ;;
esac

IFS=',' read -r -a TARGET_CLUSTERS <<< "$CLUSTERS_CSV"

routing_args() {
    local expert_k="$1"
    if (( expert_k < 8 )); then
        printf '%s\n' \
            --qwen-expert-weight-mode truncate \
            --qwen-expert-keep-k "$expert_k" \
            --qwen-expert-router-k 8 \
            --qwen-expert-reference-k 8 \
            --qwen-expert-renormalize false \
            --qwen-num-experts 256
    elif (( expert_k > 8 )); then
        printf '%s\n' \
            --hf-overrides "{\"text_config\":{\"num_experts_per_tok\":${expert_k}}}" \
            --qwen-expert-weight-mode truncate \
            --qwen-expert-keep-k "$expert_k" \
            --qwen-expert-router-k "$expert_k" \
            --qwen-expert-reference-k 8 \
            --qwen-expert-renormalize false \
            --qwen-num-experts 256
    fi
}

launch_shard() {
    local expert_k="$1"
    local shard="$2"
    local include="$3"
    local routing_label="reference"
    if (( expert_k == 8 )); then
        routing_label="native"
    fi
    local job_name="qwen36-${STAGE}-k${expert_k}-${routing_label}-full${TASK_COUNT}-rep${REPLICATE}-qwen-settings-agent${AGENT_TIMEOUT_SEC}-cmd${COMMAND_TIMEOUT_SEC}-tp2-dp4-c4-${shard}-${RUN_DATE}"
    local -a expert_args=()
    mapfile -t expert_args < <(routing_args "$expert_k")
    local -a cmd=(
        "$TMAX_LAUNCHER" "$MODEL"
        --name qwen36-35b-a3b
        --job-name "$job_name"
        --results-dir "${RESULTS_ROOT}/${job_name}"
        --gpus 8
        --tp 2
        --dp 4
        --workspace "$WORKSPACE"
        --priority "$PRIORITY"
        --allocated
        --repo-ref "$TMAX_REPO_REF"
        --beaker-scripts-dataset "$BEAKER_SCRIPTS_DATASET"
        --dataset "$DATASET"
        --harbor-env docker
        --agent Vanillux2Agent:Vanillux2Agent
        --n-attempts 1
        --n-concurrent 4
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
        --agent-kwarg "command_timeout=${COMMAND_TIMEOUT_SEC}"
        --agent-timeout-sec "$AGENT_TIMEOUT_SEC"
        --timeout-multiplier 10
        --verifier-timeout-multiplier 10
        --agent-setup-timeout-multiplier 10
        --environment-build-timeout-multiplier 10
        "${expert_args[@]}"
    )

    if [[ -n "$include" ]]; then
        cmd+=(--include-task-name "$include")
    fi

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

for expert_k in $EXPERT_COUNTS; do
    case "$expert_k" in
        4|6|8|10|12) ;;
        *) echo "unsupported expert count: $expert_k" >&2; exit 2 ;;
    esac
    if [[ "$SHARD_MODE" == "single" ]]; then
        launch_shard "$expert_k" single ""
    else
        for shard in $SHARDS; do
            case "$shard" in
                a) launch_shard "$expert_k" shard-a "$SHARD_A" ;;
                b) launch_shard "$expert_k" shard-b "$SHARD_B" ;;
                *) echo "unsupported shard: $shard" >&2; exit 2 ;;
            esac
        done
    fi
done
