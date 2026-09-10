#!/usr/bin/env bash
set -euo pipefail

# Single matched Qwen3.6 TB2.1 sweep using exactly the current TBLite
# evaluation protocol, except for the benchmark dataset. Each K is one
# eight-GPU H100 job hosting four independent TP=2 engines.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMAX_LAUNCHER="${TMAX_LAUNCHER:-${ROOT_DIR}/tmax-eval/beaker_configs/launch_eval.sh}"
TMAX_REPO_REF="${TMAX_REPO_REF:-7387d2f9142397a458dc39f0827a2ab0b4c03cda}"
BEAKER_SCRIPTS_DATASET="${BEAKER_SCRIPTS_DATASET:-01M01S1E06RZRV98A39Y6KGCVX}"
DATASET="${DATASET:-terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
PRIORITY="${PRIORITY:-urgent}"
CLUSTERS_CSV="${CLUSTERS:-ai2/jupiter,ai2/ceres}"
EXPERT_COUNTS="${EXPERT_COUNTS:-4 6 8 10 12}"
REPLICATES="${REPLICATES:-1}"
RESULTS_ROOT="${RESULTS_ROOT:-/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-tb21-reference-sweep-20260815}"
RUN_DATE="${RUN_DATE:-20260815}"
RUN_VERSION="${RUN_VERSION:-}"
LAUNCH=false

usage() {
    cat >&2 <<'EOF'
usage: launch_qwen36_tb21_reference_sweep.sh [--launch]

The REPLICATES environment variable selects one or more of 1, 2, and 3.
Without --launch, prints every exact command and submits nothing.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --launch) LAUNCH=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

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

seed_for_replicate() {
    case "$1" in
        1) echo 4202 ;;
        2) echo 4203 ;;
        3) echo 4204 ;;
        *) echo "unsupported replicate: $1" >&2; exit 2 ;;
    esac
}

IFS=',' read -r -a target_clusters <<< "$CLUSTERS_CSV"

for expert_k in $EXPERT_COUNTS; do
    case "$expert_k" in
        4|6|8|10|12|14|16|32|64) ;;
        *) echo "unsupported expert count: $expert_k" >&2; exit 2 ;;
    esac

    routing_label="reference"
    if (( expert_k == 8 )); then
        routing_label="native"
    fi
    mapfile -t expert_args < <(routing_args "$expert_k")

    for replicate in $REPLICATES; do
        seed="$(seed_for_replicate "$replicate")"
        suffix="${RUN_DATE}"
        if [[ -n "$RUN_VERSION" ]]; then
            suffix+="-${RUN_VERSION}"
        fi
        job_name="qwen36-tb21-k${expert_k}-${routing_label}-tblite-protocol-full89-rep${replicate}-seed${seed}-tp2-dp4-c4-h100-${suffix}"
        cmd=(
            "$TMAX_LAUNCHER" "$MODEL"
            --name qwen36-35b-a3b
            --job-name "$job_name"
            --results-dir "${RESULTS_ROOT}/${job_name}"
            --gpus 8
            --tp 2
            --dp 4
            --dataset "$DATASET"
            --workspace "$WORKSPACE"
            --priority "$PRIORITY"
            --allocated
            --repo-ref "$TMAX_REPO_REF"
            --beaker-scripts-dataset "$BEAKER_SCRIPTS_DATASET"
            --tool-call-parser qwen3_xml
            --reasoning-parser qwen3
            --language-model-only
            --max-model-len 262144
            --gpu-memory-utilization 0.95
            --max-num-seqs 1
            --gdn-prefill-backend triton
            --n-concurrent 4
            --hosted-vllm-model-info '{"max_input_tokens":262144,"max_output_tokens":81920,"input_cost_per_token":0.0,"output_cost_per_token":0.0}'
            --agent Vanillux2Agent:Vanillux2Agent
            --agent-kwarg temperature=1.0
            --agent-kwarg top_p=0.95
            --agent-kwarg top_k=20
            --agent-kwarg max_tokens=81920
            --agent-kwarg "seed=${seed}"
            "${expert_args[@]}"
        )
        for cluster in "${target_clusters[@]}"; do
            cmd+=(--cluster "$cluster")
        done

        printf 'Prepared K=%s replicate=%s seed=%s launch=%s:\n' \
            "$expert_k" "$replicate" "$seed" "$LAUNCH"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        if [[ "$LAUNCH" == true ]]; then
            "${cmd[@]}"
        fi
    done
done
