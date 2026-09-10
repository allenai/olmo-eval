#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="${ROOT_DIR}/scripts/adaptive_experts/launch.sh"

GROUP="${GROUP:-adaptive-compute-qwen36-capability-reference-20260812}"
CLUSTER="${CLUSTER:-80g}"
WORKSPACE="${WORKSPACE:-ai2/holmes-testing}"
PRIORITY="${PRIORITY:-urgent}"
QWEN36_KS="${QWEN36_KS:-4 6 8 10 12}"
RUN_VERSION="${RUN_VERSION:-v1}"
RUN_DATE="${RUN_DATE:-20260812}"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

launch_condition() {
    local requested_k=$1
    local replicate=$2
    local seed=$((41 + replicate))
    local router_k=$requested_k
    local mode=truncate
    local policy=reference
    local -a policy_env=()

    if (( requested_k < 8 )); then
        router_k=8
    fi
    if (( requested_k == 8 )); then
        mode=normal
        policy=native
    else
        policy_env=(
            QWEN_EXPERT_KEEP_K="${requested_k}"
            QWEN_EXPERT_REFERENCE_K=8
            QWEN_EXPERT_RENORMALIZE=false
        )
    fi

    local -a cmd=(
        env
        GROUP="${GROUP}"
        CLUSTER="${CLUSTER}"
        WORKSPACE="${WORKSPACE}"
        PRIORITY="${PRIORITY}"
        RUN_TAG="qwen36-k${requested_k}-${policy}-full-suite-r${replicate}-seed${seed}-${RUN_VERSION}-${RUN_DATE}"
        QWEN_EXPERT_WEIGHT_MODE="${mode}"
        QWEN_EXPERT_WEIGHT_SEED="${seed}"
        "${policy_env[@]}"
        "${LAUNCHER}"
        --phase qwen36-expanded-policy
        --experts "${router_k}"
    )
    if [[ "${DRY_RUN}" == true ]]; then
        cmd+=(--dry-run)
    fi
    "${cmd[@]}"
}

for requested_k in ${QWEN36_KS}; do
    for replicate in 1 2 3; do
        launch_condition "${requested_k}" "${replicate}"
    done
done
