#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="${ROOT_DIR}/scripts/adaptive_experts/launch.sh"
STAGE=""
DRY_RUN=false

WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"
GROUP="${GROUP:-adaptive-experts-nemotron3-super-routing-moe-20260717}"
SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-256}"

usage() {
    echo "Usage: $0 --stage smoke|full|normalized-intermediate [--dry-run]"
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
    && "${STAGE}" != "normalized-intermediate" ]]; then
    usage >&2
    exit 2
fi

launch_condition() {
    local expert_count=$1
    local normalize=$2
    local run_tag=$3
    local phase=nemotron-routing-pilot
    if [[ "${STAGE}" == "smoke" ]]; then
        phase=nemotron-routing-smoke
    fi

    local -a args=(--phase "${phase}" --experts "${expert_count}")
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
        TENSOR_PARALLEL_SIZE=2 \
        NEMOTRON_NORM_TOPK_PROB="${normalize}" \
        bash "${LAUNCHER}" "${args[@]}"
}

if [[ "${STAGE}" == "smoke" ]]; then
    launch_condition 22 true nemotron-routing-smoke-k22-normalized-20260717
    launch_condition 11 true nemotron-routing-smoke-k11-normalized-20260717
    launch_condition 22 false nemotron-routing-smoke-k22-rawx5-20260717
    launch_condition 11 false nemotron-routing-smoke-k11-rawx5-20260717
    exit 0
fi

if [[ "${STAGE}" == "normalized-intermediate" ]]; then
    for replicate in 1 2 3; do
        for expert_count in 13 15 17 19 21; do
            launch_condition \
                "${expert_count}" true \
                "nemotron-routing-k${expert_count}-normalized-r${replicate}-20260717"
        done
    done
    exit 0
fi

for replicate in 1 2 3; do
    launch_condition 22 true "nemotron-routing-k22-normalized-r${replicate}-20260717"
    launch_condition 11 true "nemotron-routing-k11-normalized-r${replicate}-20260717"
    launch_condition 22 false "nemotron-routing-k22-rawx5-r${replicate}-20260717"
    launch_condition 11 false "nemotron-routing-k11-rawx5-r${replicate}-20260717"
done
