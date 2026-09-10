#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="${ROOT_DIR}/scripts/adaptive_experts/launch.sh"
LEDGER="${LEDGER:-${ROOT_DIR}/notes/beaker_jobs.jsonl}"
WORKSPACE="${WORKSPACE:-ai2/holmes-testing}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"
GROUP="${GROUP:-adaptive-experts-qwen35-35b-a3b-20260725}"
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
    local router_k=$1
    local run_tag=$2
    shift 2

    local phase=qwen35-expanded-policy
    if [[ "${run_tag}" == *-smoke-* ]]; then
        phase=qwen35-expanded-policy-smoke
    fi
    if [[ "${DRY_RUN}" == "false" ]] && already_recorded "${phase}" "${run_tag}"; then
        echo "Skipping recorded condition: ${run_tag}"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    local -a cmd=(
        env
        "LEDGER=${LEDGER}"
        "WORKSPACE=${WORKSPACE}"
        "CLUSTER=${CLUSTER}"
        "PRIORITY=${PRIORITY}"
        "GROUP=${GROUP}"
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
    launch_condition 4 qwen35-k4-normalized-smoke-triton-attn-titan-linear-rnns-v9-20260725 \
        QWEN_EXPERT_WEIGHT_MODE=normal \
        QWEN35_ATTENTION_BACKEND=TRITON_ATTN \
        ARTIFACT_REGISTRY_SECRET=GOOGLE_APPLICATION_CREDENTIALS \
        SMOKE_MAX_TOKENS=2048
    launch_condition 8 qwen35-k4-reference-smoke-triton-attn-gcp-titan-linear-rnns-v10-20260725 \
        QWEN_EXPERT_WEIGHT_MODE=truncate \
        QWEN_EXPERT_KEEP_K=4 \
        QWEN_EXPERT_REFERENCE_K=8 \
        QWEN_EXPERT_RENORMALIZE=false \
        QWEN35_ATTENTION_BACKEND=TRITON_ATTN \
        ARTIFACT_REGISTRY_SECRET=GOOGLE_APPLICATION_CREDENTIALS \
        SMOKE_MAX_TOKENS=2048
    launch_condition 8 qwen35-k8-normalized-smoke-triton-attn-gcp-titan-linear-rnns-v10-20260725 \
        QWEN_EXPERT_WEIGHT_MODE=normal \
        QWEN35_ATTENTION_BACKEND=TRITON_ATTN \
        ARTIFACT_REGISTRY_SECRET=GOOGLE_APPLICATION_CREDENTIALS \
        SMOKE_MAX_TOKENS=2048
}

launch_production() {
    local k replicate
    for k in 4 5 6 7 8; do
        for replicate in 1 2 3; do
            launch_condition "${k}" "qwen35-k${k}-normalized-r${replicate}-v4-20260725" \
                QWEN_EXPERT_WEIGHT_MODE=normal \
                QWEN35_ATTENTION_BACKEND=TRITON_ATTN \
                QWEN35_GDN_PREFILL_BACKEND=triton
            launch_condition 8 "qwen35-k${k}-reference-r${replicate}-v4-20260725" \
                QWEN_EXPERT_WEIGHT_MODE=truncate \
                QWEN_EXPERT_KEEP_K="${k}" \
                QWEN_EXPERT_REFERENCE_K=8 \
                QWEN_EXPERT_RENORMALIZE=false \
                QWEN35_ATTENTION_BACKEND=TRITON_ATTN \
                QWEN35_GDN_PREFILL_BACKEND=triton
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
