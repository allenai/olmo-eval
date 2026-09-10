#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="${ROOT_DIR}/scripts/adaptive_experts/launch.sh"
LEDGER="${LEDGER:-${ROOT_DIR}/notes/beaker_jobs.jsonl}"
WORKSPACE="${WORKSPACE:-ai2/holmes-testing}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"
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
    local run_tag=$2
    local group=$3
    local threshold=$4

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
        QWEN_EXPERT_WEIGHT_MODE=adaptive_mass
        "QWEN_EXPERT_MASS_THRESHOLD=${threshold}"
        QWEN_EXPERT_MIN_K=1
        QWEN_EXPERT_MAX_K=8
        QWEN_EXPERT_RENORMALIZE=false
        QWEN_RECORD_REALIZED_K=true
        QWEN_REALIZED_K_INTERVAL_SECONDS=5
        QWEN_ENFORCE_EAGER=true
    )
    if [[ "${phase}" == "hybrid-expanded-policy-smoke" ]]; then
        cmd+=(SMOKE_MAX_TOKENS=512)
    fi
    cmd+=("${LAUNCHER}" --phase "${phase}" --experts 8)
    if [[ "${DRY_RUN}" == "true" ]]; then
        cmd+=(--dry-run)
    fi
    "${cmd[@]}"
    SUBMITTED=$((SUBMITTED + 1))
}

launch_smoke() {
    launch_condition \
        hybrid-expanded-policy-smoke \
        qwen-adaptive-reference-eager-t0p50-smoke-20260722 \
        adaptive-experts-qwen-adaptive-reference-eager-smoke-20260722 \
        0.50
}

launch_production() {
    local threshold threshold_tag replicate
    local group=adaptive-experts-qwen-adaptive-reference-eager-thresholds-20260722
    for threshold in 0.50 0.60 0.70 0.80; do
        threshold_tag="${threshold/./p}"
        for replicate in 1 2 3; do
            launch_condition \
                hybrid-expanded-policy \
                "qwen-adaptive-reference-eager-t${threshold_tag}-r${replicate}-20260722" \
                "${group}" \
                "${threshold}"
        done
    done
}

if [[ "${STAGE}" == "smoke" || "${STAGE}" == "all" ]]; then
    launch_smoke
fi
if [[ "${STAGE}" == "production" || "${STAGE}" == "all" ]]; then
    launch_production
fi

echo "Submission summary: submitted=${SUBMITTED} skipped=${SKIPPED} dry_run=${DRY_RUN}"
