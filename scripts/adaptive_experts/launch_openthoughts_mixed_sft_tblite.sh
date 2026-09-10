#!/usr/bin/env bash
set -euo pipefail

# Evaluate the OpenThoughts-Agent mixed-K SFT checkpoint across the same
# reference-scaled K grid used for the fixed-K SFT checkpoints. Each run uses
# one eight-H100 node with four independent TP=2 serving engines.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMAX_LAUNCHER="${TMAX_LAUNCHER:-${ROOT_DIR}/tmax-eval/beaker_configs/launch_eval.sh}"
TMAX_REPO_REF="${TMAX_REPO_REF:-7387d2f9142397a458dc39f0827a2ab0b4c03cda}"
BEAKER_SCRIPTS_DATASET="${BEAKER_SCRIPTS_DATASET:-01M01S1E06RZRV98A39Y6KGCVX}"
MODEL_PATH="${MODEL_PATH:-/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints/qwen35-openthoughts-agent-100k-mixed-k4-k8-k12-refk8-lr2e-5-flce-b64-20260817-hf}"
SERVED_NAME="${SERVED_NAME:-qwen35-openthoughts-agent-100k-mixed-k4-k8-k12}"
CHECKPOINT_ROUTER_K="${CHECKPOINT_ROUTER_K:-8}"
RESULTS_ROOT="${RESULTS_ROOT:-/weka/oe-adapt-default/jacobm/tmax-eval/openthoughts-sft-mixed-tblite-20260817}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
PRIORITY="${PRIORITY:-urgent}"
ALLOCATED="${ALLOCATED:-true}"
CLUSTERS_CSV="${CLUSTERS:-ai2/jupiter,ai2/ceres}"
EVAL_KS="${EVAL_KS:-4 6 8 10 12}"
REPLICATES="${REPLICATES:-1 2 3}"
RUN_DATE="${RUN_DATE:-20260817}"
RUN_VERSION="${RUN_VERSION:-mixed1}"
LAUNCH=false

usage() {
    cat >&2 <<'EOF'
usage: launch_openthoughts_mixed_sft_tblite.sh [--launch]

Without --launch, prints all commands and submits nothing. EVAL_KS and
REPLICATES may be overridden through the environment. Set ALLOCATED=false
to submit preemptible/unallocated jobs; the default is ALLOCATED=true.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --launch) LAUNCH=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

seed_for_replicate() {
    case "$1" in
        1) echo 4202 ;;
        2) echo 4203 ;;
        3) echo 4204 ;;
        *) echo "unsupported replicate: $1" >&2; exit 2 ;;
    esac
}

test -f "${MODEL_PATH}/config.json"
test "$(jq -r '.model_type' "${MODEL_PATH}/config.json")" = "qwen3_5_moe"
test "$(jq -r '.text_config.num_experts' "${MODEL_PATH}/config.json")" = "256"
test "$(jq -r '.text_config.num_experts_per_tok' "${MODEL_PATH}/config.json")" = "$CHECKPOINT_ROUTER_K"

for eval_k in $EVAL_KS; do
    case "$eval_k" in
        2|4|6|8|10|12|14|16|24|32) ;;
        *) echo "unsupported eval K: $eval_k" >&2; exit 2 ;;
    esac
done

IFS=',' read -r -a target_clusters <<< "$CLUSTERS_CSV"

for eval_k in $EVAL_KS; do
    router_k=8
    if (( eval_k > 8 )); then
        router_k="$eval_k"
    fi

    for replicate in $REPLICATES; do
        seed="$(seed_for_replicate "$replicate")"
        job_name="${SERVED_NAME}-tblite-eval-k${eval_k}-refk8-terminus2-json-rep${replicate}-seed${seed}-${RUN_VERSION}-${RUN_DATE}"
        llm_call_kwargs="{\"max_tokens\":81920,\"top_p\":0.95,\"seed\":${seed},\"extra_body\":{\"top_k\":20}}"
        cmd=(
            "$TMAX_LAUNCHER" "$MODEL_PATH"
            --name "$SERVED_NAME"
            --job-name "$job_name"
            --results-dir "${RESULTS_ROOT}/${job_name}"
            --gpus 8
            --tp 2
            --dp 4
            --dataset openthoughts-tblite@2.0
            --workspace "$WORKSPACE"
            --priority "$PRIORITY"
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
            --hf-overrides "{\"text_config\":{\"num_experts_per_tok\":${router_k}}}"
            --qwen-expert-weight-mode truncate
            --qwen-expert-keep-k "$eval_k"
            --qwen-expert-router-k "$router_k"
            --qwen-expert-reference-k 8
            --qwen-expert-renormalize false
            --qwen-num-experts 256
            --harbor-model-name "openai/${SERVED_NAME}"
            --agent terminus-2
            --agent-kwarg parser_name=json
            --agent-kwarg temperature=1.0
            --agent-kwarg 'model_info={"max_input_tokens":262144,"max_output_tokens":81920,"input_cost_per_token":0.0,"output_cost_per_token":0.0}'
            --agent-kwarg "llm_call_kwargs=${llm_call_kwargs}"
        )
        if [[ "$ALLOCATED" == true ]]; then
            cmd+=(--allocated)
        elif [[ "$ALLOCATED" != false ]]; then
            echo "ALLOCATED must be true or false, got: $ALLOCATED" >&2
            exit 2
        fi
        for cluster in "${target_clusters[@]}"; do
            cmd+=(--cluster "$cluster")
        done

        printf 'Prepared eval-K=%s router-K=%s replicate=%s seed=%s launch=%s:\n' \
            "$eval_k" "$router_k" "$replicate" "$seed" "$LAUNCH"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        if [[ "$LAUNCH" == true ]]; then
            "${cmd[@]}"
        fi
    done
done
