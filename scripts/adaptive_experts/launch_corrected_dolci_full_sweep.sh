#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="${ROOT_DIR}/scripts/adaptive_experts/launch.sh"
LEDGER="${LEDGER:-${ROOT_DIR}/notes/beaker_jobs.jsonl}"
CHECKPOINT="${CHECKPOINT:-/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints/qwen3-30b-a3b-dolci-think-olmo-core-sft-full-terminal-eos-v2-lr4e-5-20260715-071349-hf}"
WORKSPACE="${WORKSPACE:-ai2/holmes-testing}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"
SMOKE_WORKSPACE="${SMOKE_WORKSPACE:-ai2/olmo-instruct}"
SMOKE_PRIORITY="${SMOKE_PRIORITY:-urgent}"
GROUP="${GROUP:-adaptive-experts-qwen-dolci-terminal-eos-v2-fullsuite-20260719}"
SMOKE_GROUP="${SMOKE_GROUP:-adaptive-experts-qwen-dolci-terminal-eos-v2-smokes-olmo-instruct-20260719}"
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
            'any(.[]; type == "object" and .phase == $phase and .run_tag == $run_tag)' \
            "${LEDGER}" >/dev/null
}

launch_condition() {
    local phase=$1
    local router_k=$2
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
        "DOLCI_QWEN_MODEL=${CHECKPOINT}"
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
    cmd+=("${LAUNCHER}" --phase "${phase}" --experts "${router_k}")
    if [[ "${DRY_RUN}" == "true" ]]; then
        cmd+=(--dry-run)
    fi
    "${cmd[@]}"
    SUBMITTED=$((SUBMITTED + 1))
}

launch_smokes() {
    launch_condition \
        dolci-qwen-full-smoke 8 \
        dolci-v2-norm-k8-smoke-instruct-20260719 \
        "${SMOKE_GROUP}" \
        WORKSPACE="${SMOKE_WORKSPACE}" \
        PRIORITY="${SMOKE_PRIORITY}" \
        QWEN_EXPERT_WEIGHT_MODE=normal \
        SMOKE_MAX_TOKENS=512

    launch_condition \
        dolci-qwen-full-smoke 8 \
        dolci-v2-ref-k3-smoke-instruct-20260719 \
        "${SMOKE_GROUP}" \
        WORKSPACE="${SMOKE_WORKSPACE}" \
        PRIORITY="${SMOKE_PRIORITY}" \
        QWEN_EXPERT_WEIGHT_MODE=truncate \
        QWEN_EXPERT_KEEP_K=3 \
        QWEN_EXPERT_REFERENCE_K=8 \
        QWEN_EXPERT_RENORMALIZE=false \
        SMOKE_MAX_TOKENS=512
}

launch_production() {
    local k replicate router_k
    for k in 3 4 5 6 7 8 9 10 11 12; do
        for replicate in 1 2 3; do
            launch_condition \
                dolci-qwen-full "${k}" \
                "dolci-v2-norm-k${k}-r${replicate}-20260719" \
                "${GROUP}" \
                QWEN_EXPERT_WEIGHT_MODE=normal

            router_k=8
            if ((k > 8)); then
                router_k=${k}
            fi
            launch_condition \
                dolci-qwen-full "${router_k}" \
                "dolci-v2-ref-k${k}-r${replicate}-20260719" \
                "${GROUP}" \
                QWEN_EXPERT_WEIGHT_MODE=truncate \
                QWEN_EXPERT_KEEP_K="${k}" \
                QWEN_EXPERT_REFERENCE_K=8 \
                QWEN_EXPERT_RENORMALIZE=false
        done
    done
}

if [[ "${STAGE}" == "smoke" || "${STAGE}" == "all" ]]; then
    launch_smokes
fi
if [[ "${STAGE}" == "production" || "${STAGE}" == "all" ]]; then
    launch_production
fi

echo "Submission summary: submitted=${SUBMITTED} skipped=${SKIPPED} dry_run=${DRY_RUN}"
