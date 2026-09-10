#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMAX_LAUNCHER="${TMAX_LAUNCHER:-${ROOT_DIR}/tmax-eval/beaker_configs/launch_eval.sh}"
TMAX_REPO_REF="${TMAX_REPO_REF:-7387d2f9142397a458dc39f0827a2ab0b4c03cda}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
CLUSTERS="${CLUSTERS:-$CLUSTER}"
PRIORITY="${PRIORITY:-urgent}"
ALLOCATED="${ALLOCATED:-false}"
RESULTS_ROOT="${RESULTS_ROOT:-/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-35b-a3b}"
RUN_DATE="${RUN_DATE:-}"
HARBOR_ENV="${HARBOR_ENV:-docker}"
BEAKER_SCRIPTS_DATASET="${BEAKER_SCRIPTS_DATASET:-}"
STAGE=""
SAMPLING_PROFILE=""
DRY_RUN=false

usage() {
    echo "Usage: $0 --stage smoke|pilot|context-smoke|sampling-pilot|full|full-repeats|full-k10|full-k12|qwen-published-k8 [--sampling-profile tmax|qwen] [--dry-run]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --sampling-profile) SAMPLING_PROFILE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$STAGE" != "smoke" && "$STAGE" != "pilot" && \
      "$STAGE" != "context-smoke" && "$STAGE" != "sampling-pilot" && \
      "$STAGE" != "full" && "$STAGE" != "full-repeats" && \
      "$STAGE" != "full-k10" && "$STAGE" != "full-k12" && \
      "$STAGE" != "qwen-published-k8" ]]; then
    usage >&2
    exit 2
fi

if [[ ( "$STAGE" == "full" || "$STAGE" == "full-repeats" || \
        "$STAGE" == "full-k10" || "$STAGE" == "full-k12" ) && \
      "$SAMPLING_PROFILE" != "tmax" && "$SAMPLING_PROFILE" != "qwen" ]]; then
    echo "--stage $STAGE requires --sampling-profile tmax|qwen" >&2
    exit 2
fi

GPU_COUNT=1
TP_SIZE=1
DP_SIZE=1
MAX_MODEL_LEN=110000
MAX_NUM_SEQS=4
N_CONCURRENT=4
if [[ "$STAGE" == "context-smoke" || "$STAGE" == "sampling-pilot" || \
      "$STAGE" == "full" || "$STAGE" == "full-repeats" || \
      "$STAGE" == "full-k10" || "$STAGE" == "full-k12" || \
      "$STAGE" == "qwen-published-k8" ]]; then
    GPU_COUNT=2
    TP_SIZE=2
    MAX_MODEL_LEN=262144
    MAX_NUM_SEQS=1
    N_CONCURRENT=1
fi
if [[ "$STAGE" == "sampling-pilot" ]]; then
    # Four independent TP=2 replicas: one long-context trial per engine.
    GPU_COUNT=8
    DP_SIZE=4
    N_CONCURRENT=4
fi
if [[ "$STAGE" == "full" || "$STAGE" == "full-repeats" || \
      "$STAGE" == "full-k10" || "$STAGE" == "full-k12" || \
      "$STAGE" == "qwen-published-k8" ]]; then
    # One four-engine shard per eight-GPU node. Two disjoint shards below give
    # each routing condition eight engines in total without multi-node vLLM.
    GPU_COUNT=8
    DP_SIZE=4
    N_CONCURRENT=4
fi

run_condition() {
    local name=$1
    local n_tasks=$2
    local sampling_profile=$3
    shift 3
    local max_tokens=32768
    local -a sampling_args=()
    case "$sampling_profile" in
        tmax)
            sampling_args+=(
                --agent-kwarg temperature=0.7
                --agent-kwarg top_p=0.95
            )
            ;;
        qwen)
            sampling_args+=(
                --agent-kwarg temperature=0.6
                --agent-kwarg top_p=0.95
                --agent-kwarg top_k=20
            )
            ;;
        qwen-tb2-published)
            max_tokens=81920
            sampling_args+=(
                --agent-kwarg temperature=1.0
                --agent-kwarg top_p=0.95
                --agent-kwarg top_k=20
            )
            ;;
        *)
            echo "unknown sampling profile: $sampling_profile" >&2
            exit 2
            ;;
    esac
    local -a cmd=(
        "$TMAX_LAUNCHER" "$MODEL"
        --name qwen36-35b-a3b
        --job-name "$name"
        --results-dir "${RESULTS_ROOT}/${name}"
        --gpus "$GPU_COUNT"
        --tp "$TP_SIZE"
        --dp "$DP_SIZE"
        --workspace "$WORKSPACE"
        --priority "$PRIORITY"
        --repo-ref "$TMAX_REPO_REF"
        --dataset terminal-bench@2.0
        --harbor-env "$HARBOR_ENV"
        --n-attempts 1
        --n-concurrent "$N_CONCURRENT"
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization 0.95
        --max-num-seqs "$MAX_NUM_SEQS"
        --gdn-prefill-backend triton
        --language-model-only
        --tool-call-parser qwen3_coder
        --reasoning-parser qwen3
        --hosted-vllm-model-info "{\"max_input_tokens\":${MAX_MODEL_LEN},\"max_output_tokens\":${max_tokens},\"input_cost_per_token\":0.0,\"output_cost_per_token\":0.0}"
        --agent-kwarg "max_tokens=${max_tokens}"
        --agent-kwarg max_steps=64
        --agent-kwarg max_format_errors=64
        "${sampling_args[@]}"
        "$@"
    )
    local -a target_clusters=()
    IFS=',' read -r -a target_clusters <<< "$CLUSTERS"
    for cluster in "${target_clusters[@]}"; do
        cmd+=( --cluster "$cluster" )
    done
    if [[ -n "$n_tasks" ]]; then
        cmd+=( --n-tasks "$n_tasks" )
    fi
    if [[ -n "$BEAKER_SCRIPTS_DATASET" ]]; then
        cmd+=( --beaker-scripts-dataset "$BEAKER_SCRIPTS_DATASET" )
    fi
    if [[ "$ALLOCATED" == true ]]; then
        cmd+=( --allocated )
    fi
    if [[ "$DRY_RUN" == true ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
}

if [[ "$STAGE" == "smoke" ]]; then
    run_condition qwen36-tb2-k8-default-smoke-v4-docker-110k-20260807 1 tmax
    exit 0
fi

if [[ "$STAGE" == "context-smoke" ]]; then
    run_condition qwen36-tb2-k8-native-qwen-sampling-context-smoke-tp2-262k-c1-20260807 1 qwen
    exit 0
fi

if [[ "$STAGE" == "sampling-pilot" ]]; then
    run_condition qwen36-tb2-k8-native-tmax-sampling-pilot20-tp2-dp4-262k-c4-20260807 20 tmax
    run_condition qwen36-tb2-k8-native-qwen-coding-sampling-pilot20-tp2-dp4-262k-c4-20260807 20 qwen
    exit 0
fi

# The pilot uses the same deterministic first 20 Terminal-Bench 2.0 tasks.
# K=4/6 preserve the native top-eight scale by routing eight experts and
# zeroing ranks below K without renormalization.
if [[ "$STAGE" == "pilot" ]]; then
run_condition qwen36-tb2-k8-default-pilot20-docker-110k-20260807 20 tmax
run_condition qwen36-tb2-k6-reference-pilot20-docker-110k-20260807 20 tmax \
    --qwen-expert-weight-mode truncate \
    --qwen-expert-keep-k 6 \
    --qwen-expert-router-k 8 \
    --qwen-expert-reference-k 8 \
    --qwen-expert-renormalize false \
    --qwen-num-experts 256
run_condition qwen36-tb2-k4-reference-pilot20-docker-110k-20260807 20 tmax \
    --qwen-expert-weight-mode truncate \
    --qwen-expert-keep-k 4 \
    --qwen-expert-router-k 8 \
    --qwen-expert-reference-k 8 \
    --qwen-expert-renormalize false \
    --qwen-num-experts 256
exit 0
fi

# Full-suite stage: the complementary task-name globs partition all lowercase
# Terminal-Bench task names. Together, shard A and B evaluate all 89 tasks once.
SHARD_A='[acdegikmoswy0-9]*'
SHARD_B='[bfhjlnpqrtuvxz]*'

if [[ "$STAGE" == "qwen-published-k8" ]]; then
    # One full 89-task attempt using Vanillux2Agent but otherwise matching
    # Qwen's published TB2 recipe where its settings are public. The two
    # complementary shards retain the validated four-engine TP=2 topology.
    for shard in a b; do
        if [[ "$shard" == "a" ]]; then
            include="$SHARD_A"
        else
            include="$SHARD_B"
        fi
        run_condition "qwen36-tb2-k8-native-vanillux-qwen-published-settings-full89-rep1-tp2-dp4-262k-out80k-timeout3h-c4-shard-${shard}-20260813" "" qwen-tb2-published \
            --override-cpus 32 \
            --override-memory-mb 49152 \
            --agent-timeout-sec 10800 \
            --include-task-name "$include"
    done
    exit 0
fi

if [[ "$STAGE" == "full-k10" || "$STAGE" == "full-k12" ]]; then
    if [[ "$STAGE" == "full-k10" ]]; then
        expert_k=10
        date_tag="${RUN_DATE:-20260810}"
    else
        expert_k=12
        date_tag="${RUN_DATE:-20260812}"
    fi
    # Five full-suite attempts matching the existing K=8/K=6/K=4 population:
    # replicate 1 uses the original unseeded convention; replicates 2--5 use
    # the same paired request seeds. K>8 is scaled so that its top-eight
    # weights sum to the native K=8 mass, leaving the extra weights above
    # that anchor rather than renormalizing all selected weights to one.
    for rep in 1 2 3 4 5; do
        seed_args=()
        seed_label=""
        if (( rep >= 2 )); then
            seed=$((4200 + rep))
            seed_args=(--agent-kwarg "seed=${seed}")
            seed_label="-seed${seed}"
        fi
        for shard in a b; do
            if [[ "$shard" == "a" ]]; then
                include="$SHARD_A"
            else
                include="$SHARD_B"
            fi
            run_condition "qwen36-tb2-k${expert_k}-reference-full89-rep${rep}-${SAMPLING_PROFILE}${seed_label}-tp2-dp4-262k-c4-shard-${shard}-${date_tag}" "" "$SAMPLING_PROFILE" \
                "${seed_args[@]}" \
                --hf-overrides "{\"text_config\":{\"num_experts_per_tok\":${expert_k}}}" \
                --qwen-expert-weight-mode truncate \
                --qwen-expert-keep-k "$expert_k" \
                --qwen-expert-router-k "$expert_k" \
                --qwen-expert-reference-k 8 \
                --qwen-expert-renormalize false \
                --qwen-num-experts 256 \
                --include-task-name "$include"
        done
    done
    exit 0
fi

if [[ "$STAGE" == "full-repeats" ]]; then
    # Each replicate is its own Harbor job so that a fixed generation seed can
    # be paired across all three routing conditions. The two task-name shards
    # retain the exact full-suite topology used for replicate 1.
    for rep in 2 3 4 5; do
        seed=$((4200 + rep))
        for shard in a b; do
            if [[ "$shard" == "a" ]]; then
                include="$SHARD_A"
            else
                include="$SHARD_B"
            fi

            run_condition "qwen36-tb2-k8-native-full89-rep${rep}-${SAMPLING_PROFILE}-seed${seed}-tp2-dp4-262k-c4-shard-${shard}-20260808" "" "$SAMPLING_PROFILE" \
                --agent-kwarg "seed=${seed}" \
                --include-task-name "$include"
            run_condition "qwen36-tb2-k6-reference-full89-rep${rep}-${SAMPLING_PROFILE}-seed${seed}-tp2-dp4-262k-c4-shard-${shard}-20260808" "" "$SAMPLING_PROFILE" \
                --agent-kwarg "seed=${seed}" \
                --qwen-expert-weight-mode truncate \
                --qwen-expert-keep-k 6 \
                --qwen-expert-router-k 8 \
                --qwen-expert-reference-k 8 \
                --qwen-expert-renormalize false \
                --qwen-num-experts 256 \
                --include-task-name "$include"
            run_condition "qwen36-tb2-k4-reference-full89-rep${rep}-${SAMPLING_PROFILE}-seed${seed}-tp2-dp4-262k-c4-shard-${shard}-20260808" "" "$SAMPLING_PROFILE" \
                --agent-kwarg "seed=${seed}" \
                --qwen-expert-weight-mode truncate \
                --qwen-expert-keep-k 4 \
                --qwen-expert-router-k 8 \
                --qwen-expert-reference-k 8 \
                --qwen-expert-renormalize false \
                --qwen-num-experts 256 \
                --include-task-name "$include"
        done
    done
    exit 0
fi

run_condition "qwen36-tb2-k8-native-full89-rep1-${SAMPLING_PROFILE}-tp2-dp4-262k-c4-shard-a-20260807" "" "$SAMPLING_PROFILE" \
    --include-task-name "$SHARD_A"
run_condition "qwen36-tb2-k8-native-full89-rep1-${SAMPLING_PROFILE}-tp2-dp4-262k-c4-shard-b-20260807" "" "$SAMPLING_PROFILE" \
    --include-task-name "$SHARD_B"
run_condition "qwen36-tb2-k6-reference-full89-rep1-${SAMPLING_PROFILE}-tp2-dp4-262k-c4-shard-a-20260807" "" "$SAMPLING_PROFILE" \
    --qwen-expert-weight-mode truncate \
    --qwen-expert-keep-k 6 \
    --qwen-expert-router-k 8 \
    --qwen-expert-reference-k 8 \
    --qwen-expert-renormalize false \
    --qwen-num-experts 256 \
    --include-task-name "$SHARD_A"
run_condition "qwen36-tb2-k6-reference-full89-rep1-${SAMPLING_PROFILE}-tp2-dp4-262k-c4-shard-b-20260807" "" "$SAMPLING_PROFILE" \
    --qwen-expert-weight-mode truncate \
    --qwen-expert-keep-k 6 \
    --qwen-expert-router-k 8 \
    --qwen-expert-reference-k 8 \
    --qwen-expert-renormalize false \
    --qwen-num-experts 256 \
    --include-task-name "$SHARD_B"
run_condition "qwen36-tb2-k4-reference-full89-rep1-${SAMPLING_PROFILE}-tp2-dp4-262k-c4-shard-a-20260807" "" "$SAMPLING_PROFILE" \
    --qwen-expert-weight-mode truncate \
    --qwen-expert-keep-k 4 \
    --qwen-expert-router-k 8 \
    --qwen-expert-reference-k 8 \
    --qwen-expert-renormalize false \
    --qwen-num-experts 256 \
    --include-task-name "$SHARD_A"
run_condition "qwen36-tb2-k4-reference-full89-rep1-${SAMPLING_PROFILE}-tp2-dp4-262k-c4-shard-b-20260807" "" "$SAMPLING_PROFILE" \
    --qwen-expert-weight-mode truncate \
    --qwen-expert-keep-k 4 \
    --qwen-expert-router-k 8 \
    --qwen-expert-reference-k 8 \
    --qwen-expert-renormalize false \
    --qwen-num-experts 256 \
    --include-task-name "$SHARD_B"
