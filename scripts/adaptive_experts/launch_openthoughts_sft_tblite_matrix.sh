#!/usr/bin/env bash
set -euo pipefail

# Complete the train-K x eval-K TBLite grid for the OpenThoughts-Agent SFT
# checkpoints. Each production evaluation is one eight-H100 node with four
# independent TP=2 engines. All evaluated policies retain K=8 as their router
# weight reference and use the same Terminus-2 JSON protocol as the initial
# diagonal evaluations.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMAX_LAUNCHER="${TMAX_LAUNCHER:-${ROOT_DIR}/tmax-eval/beaker_configs/launch_eval.sh}"
TMAX_REPO_REF="${TMAX_REPO_REF:-7387d2f9142397a458dc39f0827a2ab0b4c03cda}"
BEAKER_SCRIPTS_DATASET="${BEAKER_SCRIPTS_DATASET:-01M01S1E06RZRV98A39Y6KGCVX}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
PRIORITY="${PRIORITY:-urgent}"
CLUSTERS_CSV="${CLUSTERS:-ai2/jupiter,ai2/ceres}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints}"
RESULTS_ROOT="${RESULTS_ROOT:-/weka/oe-adapt-default/jacobm/tmax-eval/openthoughts-sft-tblite-matrix-20260815}"
RUN_DATE="${RUN_DATE:-20260815}"
RUN_VERSION="${RUN_VERSION:-cross1}"
TRAIN_KS="${TRAIN_KS:-4 8 12}"
EVAL_KS="${EVAL_KS:-4 8 12}"
REPLICATES="${REPLICATES:-1 2 3}"
SKIP_EXISTING_DIAGONAL_REP1="${SKIP_EXISTING_DIAGONAL_REP1:-true}"
LAUNCH=false

usage() {
    cat >&2 <<'EOF'
usage: launch_openthoughts_sft_tblite_matrix.sh [--launch]

By default this launches the 24 missing cells/runs: all off-diagonal cells at
three seeds plus diagonal replicates 2 and 3. Override TRAIN_KS, EVAL_KS,
REPLICATES, or SKIP_EXISTING_DIAGONAL_REP1 for a smaller dry run.
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

for k in $TRAIN_KS; do
    case "$k" in
        4|8|12) ;;
        *) echo "unsupported train K: $k" >&2; exit 2 ;;
    esac
done

for k in $EVAL_KS; do
    case "$k" in
        2|4|6|8|10|12|14|16) ;;
        *) echo "unsupported eval K: $k" >&2; exit 2 ;;
    esac
done

IFS=',' read -r -a target_clusters <<< "$CLUSTERS_CSV"

for train_k in $TRAIN_KS; do
    model_path="${CHECKPOINT_ROOT}/qwen35-openthoughts-agent-100k-k${train_k}-refk8-lr2e-5-flce-b64-20260813/step1523-hf"
    test -f "${model_path}/config.json"
    served_name="qwen35-openthoughts-agent-100k-train-k${train_k}-step1523"

    for eval_k in $EVAL_KS; do
        router_k=8
        if (( eval_k > 8 )); then
            router_k="$eval_k"
        fi

        hf_config_args=()
        if (( router_k != train_k )); then
            hf_config_path="${ROOT_DIR}/tmax-eval/hf_configs/qwen35_openthoughts_train_k${train_k}_router${router_k}"
            if [[ -f "${hf_config_path}/config.json" ]]; then
                test "$(jq -r '.text_config.num_experts_per_tok' "${hf_config_path}/config.json")" = "$router_k"
                hf_config_args=(--hf-config-path "$hf_config_path")
            else
                hf_config_args=(--hf-overrides "{\"text_config\":{\"num_experts_per_tok\":${router_k}}}")
            fi
        fi

        for replicate in $REPLICATES; do
            if [[ "$SKIP_EXISTING_DIAGONAL_REP1" == true ]] && \
               (( train_k == eval_k && replicate == 1 )); then
                printf 'Skipping existing diagonal train-K=%s eval-K=%s replicate=1\n' \
                    "$train_k" "$eval_k"
                continue
            fi
            seed="$(seed_for_replicate "$replicate")"
            job_name="${served_name}-tblite-eval-k${eval_k}-refk8-terminus2-json-rep${replicate}-seed${seed}-${RUN_VERSION}-${RUN_DATE}"
            llm_call_kwargs="{\"max_tokens\":81920,\"top_p\":0.95,\"seed\":${seed},\"extra_body\":{\"top_k\":20}}"
            cmd=(
                "$TMAX_LAUNCHER" "$model_path"
                --name "$served_name"
                --job-name "$job_name"
                --results-dir "${RESULTS_ROOT}/${job_name}"
                --gpus 8
                --tp 2
                --dp 4
                --dataset openthoughts-tblite@2.0
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
                --qwen-expert-weight-mode truncate
                --qwen-expert-keep-k "$eval_k"
                --qwen-expert-router-k "$router_k"
                --qwen-expert-reference-k 8
                --qwen-expert-renormalize false
                --qwen-num-experts 256
                "${hf_config_args[@]}"
                --harbor-model-name "openai/${served_name}"
                --agent terminus-2
                --agent-kwarg parser_name=json
                --agent-kwarg temperature=1.0
                --agent-kwarg 'model_info={"max_input_tokens":262144,"max_output_tokens":81920,"input_cost_per_token":0.0,"output_cost_per_token":0.0}'
                --agent-kwarg "llm_call_kwargs=${llm_call_kwargs}"
            )
            for cluster in "${target_clusters[@]}"; do
                cmd+=(--cluster "$cluster")
            done

            printf 'Prepared train-K=%s eval-K=%s router-K=%s replicate=%s seed=%s launch=%s:\n' \
                "$train_k" "$eval_k" "$router_k" "$replicate" "$seed" "$LAUNCH"
            printf '%q ' "${cmd[@]}"
            printf '\n'
            if [[ "$LAUNCH" == true ]]; then
                "${cmd[@]}"
            fi
        done
    done
done
