#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMAX_LAUNCHER="${TMAX_LAUNCHER:-${ROOT_DIR}/tmax-eval/beaker_configs/launch_eval.sh}"
TMAX_REPO_REF="${TMAX_REPO_REF:-7387d2f9142397a458dc39f0827a2ab0b4c03cda}"
BEAKER_SCRIPTS_DATASET="${BEAKER_SCRIPTS_DATASET:-01KZFFNSKNAF0ARQ76R3G37EHY}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
PRIORITY="${PRIORITY:-urgent}"
RESULTS_ROOT="${RESULTS_ROOT:-/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-35b-a3b}"
HARBOR_ENV="${HARBOR_ENV:-docker}"
DATASET="${DATASET:-terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a}"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

# These complementary globs cover all 89 tasks in the pinned TB2.1 dataset:
# 45 on shard A, 44 on shard B, with no overlap.
# Harbor exposes package-registry task names as `terminal-bench/<task-name>`
# when applying --include-task-name. Include that organization prefix here;
# otherwise a class containing `t` accidentally selects the entire dataset.
SHARD_A='terminal-bench/[acdegikmoswy0-9]*'
SHARD_B='terminal-bench/[bfhjlnpqrtuvxz]*'

# Gantry treats repeated --cluster arguments as alternatives for scheduling.
# Override CLUSTERS with a comma-separated list when moving the unchanged job
# between compatible GPU pools (for example, CLUSTERS=ai2/holmes).
CLUSTERS_CSV="${CLUSTERS:-ai2/jupiter,ai2/ceres,ai2/titan}"
IFS=',' read -r -a TARGET_CLUSTERS <<< "$CLUSTERS_CSV"

launch_shard() {
    local shard="$1"
    local include="$2"
    local job_name="qwen36-tb21-k8-native-terminus2-podman-full89-rep1-tp2-dp4-262k-out80k-native-timeouts-c4-prefixfix-shard-${shard}-20260813"
    local -a cmd=(
        "$TMAX_LAUNCHER" "$MODEL"
        --name qwen36-35b-a3b
        --job-name "$job_name"
        --results-dir "${RESULTS_ROOT}/${job_name}"
        --gpus 8
        --tp 2
        --dp 4
        --workspace "$WORKSPACE"
        --priority "$PRIORITY"
        --allocated
        --repo-ref "$TMAX_REPO_REF"
        --beaker-scripts-dataset "$BEAKER_SCRIPTS_DATASET"
        --dataset "$DATASET"
        --harbor-env "$HARBOR_ENV"
        --agent terminus-2
        --n-attempts 1
        --n-concurrent 4
        --include-task-name "$include"
        --max-model-len 262144
        --gpu-memory-utilization 0.95
        --max-num-seqs 1
        --gdn-prefill-backend triton
        --language-model-only
        --tool-call-parser qwen3_coder
        --reasoning-parser qwen3
        --hosted-vllm-model-info '{"max_input_tokens":262144,"max_output_tokens":81920,"input_cost_per_token":0.0,"output_cost_per_token":0.0}'
        --agent-kwarg parser_name=json
        --agent-kwarg temperature=1.0
        --agent-kwarg 'llm_call_kwargs={"max_tokens":81920,"top_p":0.95,"extra_body":{"top_k":20}}'
    )

    for cluster in "${TARGET_CLUSTERS[@]}"; do
        cmd+=(--cluster "$cluster")
    done

    if [[ "$DRY_RUN" == true ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
}

launch_shard a "$SHARD_A"
launch_shard b "$SHARD_B"
