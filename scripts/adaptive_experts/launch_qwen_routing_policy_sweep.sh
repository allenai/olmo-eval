#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="${ROOT_DIR}/scripts/adaptive_experts/launch.sh"
STAGE=""
DRY_RUN=false

WORKSPACE="${WORKSPACE:-ai2/holmes-testing}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"
GROUP="${GROUP:-adaptive-experts-qwen-routing-policies-20260717}"
SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-128}"
RANK_PROFILE="0.21562765,0.16642320,0.13938729,0.12015191,0.10476987,0.09296770,0.08376831,0.07690407"

usage() {
    echo "Usage: $0 --stage smoke|full [--dry-run]"
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

if [[ "${STAGE}" != "smoke" && "${STAGE}" != "full" ]]; then
    usage >&2
    exit 2
fi

launch_policy() {
    local mode=$1
    local run_tag=$2
    local experts=$3
    shift 3
    local phase="hybrid-weight-pilot"
    if [[ "${STAGE}" == "smoke" ]]; then
        phase="hybrid-weight-smoke"
    fi

    local -a args=(--phase "${phase}" --experts "${experts}")
    if [[ "${DRY_RUN}" == "true" ]]; then
        args+=(--dry-run)
    fi

    env \
        WORKSPACE="${WORKSPACE}" \
        CLUSTER="${CLUSTER}" \
        PRIORITY="${PRIORITY}" \
        GROUP="${GROUP}" \
        SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS}" \
        RUN_TAG="${run_tag}" \
        QWEN_EXPERT_WEIGHT_MODE="${mode}" \
        "$@" \
        bash "${LAUNCHER}" "${args[@]}"
}

launch_temperature() {
    local exponent=$1
    local tag=$2
    launch_policy \
        temperature "${tag}" 8 \
        QWEN_EXPERT_WEIGHT_EXPONENT="${exponent}"
}

launch_rank_profile() {
    local tag=$1
    launch_policy \
        rank_profile "${tag}" 8 \
        QWEN_EXPERT_RANK_PROFILE="${RANK_PROFILE}"
}

launch_adaptive_mass() {
    local threshold=$1
    local tag=$2
    launch_policy \
        adaptive_mass "${tag}" 8 \
        QWEN_EXPERT_MASS_THRESHOLD="${threshold}" \
        QWEN_EXPERT_MIN_K=1 \
        QWEN_EXPERT_MAX_K=8 \
        QWEN_EXPERT_RENORMALIZE=true
}

launch_reference_scaled() {
    local router_k=$1
    local keep_k=$2
    local tag=$3
    launch_policy \
        truncate "${tag}" "${router_k}" \
        QWEN_EXPERT_KEEP_K="${keep_k}" \
        QWEN_EXPERT_REFERENCE_K=8 \
        QWEN_EXPERT_RENORMALIZE=false
}

if [[ "${STAGE}" == "smoke" ]]; then
    launch_temperature 0.5 routing-smoke-temp-p0p5-20260717
    launch_temperature 2.0 routing-smoke-temp-p2-20260717
    launch_rank_profile routing-smoke-rank-profile-20260717
    launch_adaptive_mass 0.50 routing-smoke-mass-t0p50-20260717
    launch_adaptive_mass 0.90 routing-smoke-mass-t0p90-20260717
    launch_reference_scaled 8 4 routing-smoke-reference-k4-20260717
    launch_reference_scaled 12 12 routing-smoke-reference-k12-20260717
    exit 0
fi

for replicate in 1 2 3; do
    launch_temperature 0.5 "routing-temp-p0p5-r${replicate}-20260717"
    launch_temperature 0.75 "routing-temp-p0p75-r${replicate}-20260717"
    launch_temperature 1.5 "routing-temp-p1p5-r${replicate}-20260717"
    launch_temperature 2.0 "routing-temp-p2-r${replicate}-20260717"
done
launch_temperature 1.0 routing-temp-p1-sanity-20260717

for replicate in 1 2 3; do
    launch_rank_profile "routing-rank-profile-r${replicate}-20260717"
done

for threshold_tag in 0.50:t0p50 0.60:t0p60 0.70:t0p70 0.80:t0p80 0.90:t0p90; do
    threshold="${threshold_tag%%:*}"
    tag="${threshold_tag##*:}"
    for replicate in 1 2 3; do
        launch_adaptive_mass \
            "${threshold}" "routing-mass-${tag}-r${replicate}-20260717"
    done
done

for replicate in 1 2 3; do
    launch_reference_scaled 8 4 "routing-reference-k4-r${replicate}-20260717"
    launch_reference_scaled 8 6 "routing-reference-k6-r${replicate}-20260717"
    launch_reference_scaled 12 12 "routing-reference-k12-r${replicate}-20260717"
done
