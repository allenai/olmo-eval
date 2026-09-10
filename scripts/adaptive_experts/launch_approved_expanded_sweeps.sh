#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="${ROOT_DIR}/scripts/adaptive_experts/launch.sh"
LEDGER="${LEDGER:-${ROOT_DIR}/notes/beaker_jobs.jsonl}"
WORKSPACE="${WORKSPACE:-ai2/holmes-testing}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"
SMOKE_WORKSPACE="${SMOKE_WORKSPACE:-ai2/OLMo-3-moe-experiments}"
SMOKE_PRIORITY="${SMOKE_PRIORITY:-normal}"
STAGE=""
DRY_RUN=false
SUBMITTED=0
SKIPPED=0

usage() {
    echo "Usage: $0 --stage smoke|production|all [--dry-run]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${STAGE}" != "smoke" && "${STAGE}" != "production" && "${STAGE}" != "all" ]]; then
    usage >&2
    exit 2
fi

already_recorded() {
    local phase=$1
    local run_tag=$2
    [[ -f "${LEDGER}" ]] \
        && jq -s -e \
            --arg phase "${phase}" \
            --arg run_tag "${run_tag}" \
            'any(.[]; .phase == $phase and .run_tag == $run_tag)' \
            "${LEDGER}" >/dev/null
}

launch_condition() {
    local phase=$1
    local expert_count=$2
    local run_tag=$3
    local group=$4
    shift 4

    if [[ "${DRY_RUN}" == "false" ]] && already_recorded "${phase}" "${run_tag}"; then
        echo "Skipping recorded condition: ${phase} ${run_tag}"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    local -a cmd=(
        env
        "LEDGER=${LEDGER}"
        "WORKSPACE=${WORKSPACE}"
        "CLUSTER=${CLUSTER}"
        "PRIORITY=${PRIORITY}"
        "GROUP=${group}"
        "RUN_TAG=${run_tag}"
    )
    while [[ $# -gt 0 ]]; do
        cmd+=("$1")
        shift
    done
    cmd+=("${LAUNCHER}" --phase "${phase}" --experts "${expert_count}")
    if [[ "${DRY_RUN}" == "true" ]]; then
        cmd+=(--dry-run)
    fi
    "${cmd[@]}"
    SUBMITTED=$((SUBMITTED + 1))
}

gpt_tp() {
    case "$1" in
        3|5|6|7) echo 2 ;;
        *) echo 1 ;;
    esac
}

gpt_memory_utilization() {
    if [[ "$(gpt_tp "$1")" -eq 2 ]]; then
        echo 0.90
    else
        # At 0.95, GPT-OSS-120B TP=1 leaves too little headroom for vLLM's
        # 64-request sampler warmup (a 50 MiB allocation failed with 37 MiB
        # free). Preserve max_num_seqs=64 and reserve ~0.8 GiB instead.
        echo 0.94
    fi
}

launch_smokes() {
    local group=adaptive-experts-expanded-sweep-smokes-20260717
    launch_condition \
        hybrid-expanded-policy-smoke 8 \
        expanded-smoke-qwen-mass-t0p50-20260717 \
        "${group}" \
        QWEN_EXPERT_WEIGHT_MODE=adaptive_mass \
        QWEN_EXPERT_MASS_THRESHOLD=0.50 \
        QWEN_EXPERT_MIN_K=1 \
        QWEN_EXPERT_MAX_K=8 \
        QWEN_EXPERT_RENORMALIZE=true \
        QWEN_RECORD_REALIZED_K=true \
        QWEN_REALIZED_K_INTERVAL_SECONDS=5 \
        WORKSPACE="${SMOKE_WORKSPACE}" \
        PRIORITY="${SMOKE_PRIORITY}" \
        SMOKE_MAX_TOKENS=512

    local k
    for k in 2 3; do
        local resource_tag=timeout1800
        if [[ "$(gpt_tp "${k}")" -eq 1 ]]; then
            resource_tag=timeout1800-mem0p94
        fi
        launch_condition \
            gptoss-reference-smoke "${k}" \
            "expanded-smoke-gptoss-reference-k${k}-${resource_tag}-20260717" \
            "${group}" \
            GPTOSS_REFERENCE_SCALE=true \
            GPTOSS_REFERENCE_K=4 \
            GPTOSS_STARTUP_TIMEOUT=1800 \
            TENSOR_PARALLEL_SIZE="$(gpt_tp "${k}")" \
            GPTOSS_MAX_NUM_SEQS=64 \
            GPTOSS_GPU_MEMORY_UTILIZATION="$(gpt_memory_utilization "${k}")" \
            WORKSPACE="${SMOKE_WORKSPACE}" \
            PRIORITY="${SMOKE_PRIORITY}" \
            SMOKE_MAX_TOKENS=512
    done
}

launch_qwen_adaptive_mass() {
    local group=adaptive-experts-qwen-adaptive-mass-expanded-20260717
    local replicate threshold threshold_tag
    for threshold in 0.50 0.60 0.70 0.80 0.90; do
        threshold_tag="${threshold/./p}"
        for replicate in 1 2 3; do
            launch_condition \
                hybrid-expanded-policy 8 \
                "expanded-qwen-mass-t${threshold_tag}-r${replicate}-20260717" \
                "${group}" \
                QWEN_EXPERT_WEIGHT_MODE=adaptive_mass \
                QWEN_EXPERT_MASS_THRESHOLD="${threshold}" \
                QWEN_EXPERT_MIN_K=1 \
                QWEN_EXPERT_MAX_K=8 \
                QWEN_EXPERT_RENORMALIZE=true \
                QWEN_RECORD_REALIZED_K=true
        done
    done
}

launch_qwen_reference_scaled() {
    local group=adaptive-experts-qwen-reference-scaled-expanded-20260717
    local replicate router_k target_k
    for target_k in 3 5 6 7 9 10 11 12 13 14 15 16; do
        router_k=8
        if ((target_k > 8)); then
            router_k=${target_k}
        fi
        for replicate in 1 2 3; do
            launch_condition \
                hybrid-expanded-policy "${router_k}" \
                "expanded-qwen-reference-k${target_k}-r${replicate}-20260717" \
                "${group}" \
                QWEN_EXPERT_WEIGHT_MODE=truncate \
                QWEN_EXPERT_KEEP_K="${target_k}" \
                QWEN_EXPERT_REFERENCE_K=8 \
                QWEN_EXPERT_RENORMALIZE=false
        done
    done
}

launch_qwen_normalized_backfill() {
    local group=adaptive-experts-qwen-normalized-capability-backfill-20260717
    local k replicate
    for k in 3 5 6 7 9 10 11 12 13 14 15 16; do
        for replicate in 1 2 3; do
            launch_condition \
                hybrid-capability-backfill "${k}" \
                "capability32k-qwen-normalized-k${k}-r${replicate}-20260717" \
                "${group}" \
                QWEN_EXPERT_WEIGHT_MODE=normal
        done
    done
}

launch_gptoss_reference_scaled() {
    local group=adaptive-experts-gptoss-reference-scaled-expanded-20260717
    local k replicate resource_tag
    for k in 1 2 3 5 6 7 8; do
        resource_tag=timeout1800
        if [[ "$(gpt_tp "${k}")" -eq 1 ]]; then
            resource_tag=timeout1800-mem0p94
        fi
        for replicate in 1 2 3; do
            launch_condition \
                gptoss-reference-expanded "${k}" \
                "expanded-gptoss-reference-k${k}-r${replicate}-${resource_tag}-20260717" \
                "${group}" \
                GPTOSS_REFERENCE_SCALE=true \
                GPTOSS_REFERENCE_K=4 \
                GPTOSS_STARTUP_TIMEOUT=1800 \
                TENSOR_PARALLEL_SIZE="$(gpt_tp "${k}")" \
                GPTOSS_MAX_NUM_SEQS=64 \
                GPTOSS_GPU_MEMORY_UTILIZATION="$(gpt_memory_utilization "${k}")"
        done
    done
}

launch_gptoss_normalized_backfill() {
    local group=adaptive-experts-gptoss-normalized-capability-backfill-20260717
    local k replicate resource_tag
    for k in 1 2 3 4 5 6 7 8; do
        resource_tag=timeout1800
        if [[ "$(gpt_tp "${k}")" -eq 1 ]]; then
            resource_tag=timeout1800-mem0p94
        fi
        for replicate in 1 2 3; do
            launch_condition \
                gptoss-capability-backfill "${k}" \
                "capability32k-gptoss-normalized-k${k}-r${replicate}-${resource_tag}-20260717" \
                "${group}" \
                GPTOSS_REFERENCE_SCALE=false \
                GPTOSS_STARTUP_TIMEOUT=1800 \
                TENSOR_PARALLEL_SIZE="$(gpt_tp "${k}")" \
                GPTOSS_MAX_NUM_SEQS=64 \
                GPTOSS_GPU_MEMORY_UTILIZATION="$(gpt_memory_utilization "${k}")"
        done
    done
}

launch_production() {
    launch_qwen_adaptive_mass
    launch_qwen_reference_scaled
    launch_qwen_normalized_backfill
    launch_gptoss_reference_scaled
    launch_gptoss_normalized_backfill
}

if [[ "${STAGE}" == "smoke" || "${STAGE}" == "all" ]]; then
    launch_smokes
fi
if [[ "${STAGE}" == "production" || "${STAGE}" == "all" ]]; then
    launch_production
fi

echo "Submission summary: submitted=${SUBMITTED} skipped=${SKIPPED} dry_run=${DRY_RUN}"
