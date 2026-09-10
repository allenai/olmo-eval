#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="${ROOT_DIR}/scripts/adaptive_experts/launch.sh"
STAGE=""
DRY_RUN=false

GROUP="${GROUP:-adaptive-experts-qwen-capability-policies-20260717}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
WORKSPACE="${WORKSPACE:-ai2/holmes-testing}"
PRIORITY="${PRIORITY:-urgent}"

usage() {
    echo "Usage: $0 --stage smoke|full|ifbench32k [--dry-run]"
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

if [[ "${STAGE}" != "smoke" \
    && "${STAGE}" != "full" \
    && "${STAGE}" != "ifbench32k" ]]; then
    usage >&2
    exit 2
fi

launch_condition() {
    local phase=$1
    local run_tag=$2
    local expert_count=$3
    local mode=$4
    shift 4

    local -a cmd=(
        env
        GROUP="${GROUP}"
        CLUSTER="${CLUSTER}"
        WORKSPACE="${WORKSPACE}"
        PRIORITY="${PRIORITY}"
        RUN_TAG="${run_tag}"
        QWEN_EXPERT_WEIGHT_MODE="${mode}"
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
}

if [[ "${STAGE}" == "smoke" ]]; then
    launch_condition \
        hybrid-capability-policy-smoke \
        capability-smoke-mass-t0p80-20260717 \
        8 \
        adaptive_mass \
        QWEN_EXPERT_MASS_THRESHOLD=0.80 \
        QWEN_EXPERT_MIN_K=1 \
        QWEN_EXPERT_MAX_K=8 \
        QWEN_EXPERT_RENORMALIZE=true
    exit 0
fi

if [[ "${STAGE}" == "ifbench32k" ]]; then
    phase=hybrid-capability-policy-ifbench32k
    for replicate in 1 2 3; do
        launch_condition \
            "${phase}" \
            "capability32k-native-k8-r${replicate}-20260717" \
            8 \
            normal

        launch_condition \
            "${phase}" \
            "capability32k-normalized-k4-r${replicate}-20260717" \
            4 \
            normal

        launch_condition \
            "${phase}" \
            "capability32k-reference-k4-r${replicate}-20260717" \
            8 \
            truncate \
            QWEN_EXPERT_KEEP_K=4 \
            QWEN_EXPERT_REFERENCE_K=8 \
            QWEN_EXPERT_RENORMALIZE=false

        for threshold in 0.80 0.90; do
            threshold_tag="${threshold/./p}"
            launch_condition \
                "${phase}" \
                "capability32k-mass-t${threshold_tag}-r${replicate}-20260717" \
                8 \
                adaptive_mass \
                QWEN_EXPERT_MASS_THRESHOLD="${threshold}" \
                QWEN_EXPERT_MIN_K=1 \
                QWEN_EXPERT_MAX_K=8 \
                QWEN_EXPERT_RENORMALIZE=true
        done
    done
    exit 0
fi

for replicate in 1 2 3; do
    launch_condition \
        hybrid-capability-policy \
        "capability-native-k8-r${replicate}-20260717" \
        8 \
        normal

    launch_condition \
        hybrid-capability-policy \
        "capability-normalized-k4-r${replicate}-20260717" \
        4 \
        normal

    launch_condition \
        hybrid-capability-policy \
        "capability-reference-k4-r${replicate}-20260717" \
        8 \
        truncate \
        QWEN_EXPERT_KEEP_K=4 \
        QWEN_EXPERT_REFERENCE_K=8 \
        QWEN_EXPERT_RENORMALIZE=false

    for threshold in 0.80 0.90; do
        threshold_tag="${threshold/./p}"
        launch_condition \
            hybrid-capability-policy \
            "capability-mass-t${threshold_tag}-r${replicate}-20260717" \
            8 \
            adaptive_mass \
            QWEN_EXPERT_MASS_THRESHOLD="${threshold}" \
            QWEN_EXPERT_MIN_K=1 \
            QWEN_EXPERT_MAX_K=8 \
            QWEN_EXPERT_RENORMALIZE=true
    done
done
