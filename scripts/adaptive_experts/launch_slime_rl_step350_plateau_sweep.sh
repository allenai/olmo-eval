#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="$(dirname "${ROOT_DIR}")"
HF_ROOT="${PROJECT_DIR}/slime-runs/qwen3-30b-a3b-base-dapo/hf-checkpoints"
LEDGER="${LEDGER:-${ROOT_DIR}/notes/beaker_jobs.jsonl}"
SNAPSHOT="${SNAPSHOT:-${PROJECT_DIR}/.run_snapshots/olmo-eval/52060bdc2f8988ab7b74635ddadc00a71ebab5f88fa227c37d4797f265975f35}"
SOURCE_COMMIT="e8b88f196767630acaf8a383d12a33d314788d27"
SOURCE_HASH="52060bdc2f8988ab7b74635ddadc00a71ebab5f88fa227c37d4797f265975f35"
WORKSPACE="${WORKSPACE:-ai2/holmes-testing}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"
GROUP="${GROUP:-adaptive-compute-slime-qwen3-base-step350-plateau-math500-20260729}"
PHASE="${PHASE:-slime-rl-step350-plateau-math500}"
EXPERT_COUNTS="${EXPERT_COUNTS:-2 3 4 5 6 7 8 9 10 11 12}"
REPLICATES="${REPLICATES:-1 2 3}"
RUN_KEYS="${RUN_KEYS:-k8-normalized k6-reference-k8 k4-reference-k8}"
MODES="${MODES:-normalized reference}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-20480}"
NUM_INSTANCES="${NUM_INSTANCES:-4}"
LIMIT="${LIMIT:-}"
TASKS="${TASKS:-math500:chat}"
TAG_SUFFIX="${TAG_SUFFIX:-v3-math500-only-20260729}"
DRY_RUN=false
CLEAN_WORKTREE=""
LAUNCH_CWD="${ROOT_DIR}"
SUBMITTED=0
SKIPPED=0
REUSED=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

cleanup() {
    if [[ -n "${CLEAN_WORKTREE}" ]]; then
        git -C "${ROOT_DIR}" worktree remove --force "${CLEAN_WORKTREE}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [[ ! -d "${SNAPSHOT}/src" ]]; then
    echo "Missing immutable olmo-eval snapshot: ${SNAPSHOT}" >&2
    exit 1
fi

if [[ "${DRY_RUN}" == "false" ]]; then
    CLEAN_WORKTREE="$(mktemp -d "${TMPDIR:-/tmp}/olmo-eval-rl-plateau.XXXXXX")"
    rmdir "${CLEAN_WORKTREE}"
    git -C "${ROOT_DIR}" worktree add --detach --quiet "${CLEAN_WORKTREE}" "${SOURCE_COMMIT}"
    LAUNCH_CWD="${CLEAN_WORKTREE}"
fi

model_for_run() {
    case "$1" in
        k8-normalized)
            echo "${HF_ROOT}/slime-qwen3-base-dapo-k8-pilot-100step-8gpu-v5-20260725-step350-hf"
            ;;
        k6-reference-k8)
            echo "${HF_ROOT}/slime-qwen3-base-dapo-k6-reference-k8-500step-8gpu-v1-20260726-step350-hf"
            ;;
        k4-reference-k8)
            echo "${HF_ROOT}/slime-qwen3-base-dapo-k4-reference-k8-pilot-100step-8gpu-v1-20260725-step350-hf"
            ;;
        *)
            echo "Unknown run key: $1" >&2
            return 2
            ;;
    esac
}

already_recorded() {
    local run_tag=$1
    [[ -f "${LEDGER}" ]] \
        && jq -s -e --arg phase "${PHASE}" --arg tag "${run_tag}" \
            'any(.[]; type == "object" and .phase == $phase and .run_tag == $tag)' \
            "${LEDGER}" >/dev/null
}

has_prior_math500_replicates() {
    local run_key=$1
    local mode=$2
    local target_k=$3
    case "${run_key}:${mode}:${target_k}" in
        # These exact step-350 MATH-500 conditions already have three valid
        # replicates in slime_qwen3_base_dapo_checkpoint_evals.csv.
        k8-normalized:normalized:6|\
        k8-normalized:normalized:8|\
        k6-reference-k8:reference:6|\
        k4-reference-k8:reference:4)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

launch_one() {
    local run_key=$1
    local mode=$2
    local target_k=$3
    local replicate=$4
    local model router_k run_tag name
    model="$(model_for_run "${run_key}")"
    router_k=${target_k}
    run_tag="qwen3-base-dapo-${run_key}-step350-plateau-${mode}-k${target_k}-r${replicate}-${TAG_SUFFIX}"
    name="adaptive-${run_tag}"

    if [[ ! "${target_k}" =~ ^([2-9]|1[0-2])$ ]]; then
        echo "Invalid target K: ${target_k}" >&2
        return 2
    fi
    if [[ ! "${replicate}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid replicate: ${replicate}" >&2
        return 2
    fi
    if [[ "${mode}" != "normalized" && "${mode}" != "reference" ]]; then
        echo "Invalid routing mode: ${mode}" >&2
        return 2
    fi
    if [[ ! "${NUM_INSTANCES}" =~ ^[1-9][0-9]*$ ]]; then
        echo "NUM_INSTANCES must be a positive integer" >&2
        return 2
    fi
    if [[ -n "${LIMIT}" && ! "${LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
        echo "LIMIT must be empty or a positive integer" >&2
        return 2
    fi
    if [[ ! -f "${model}/config.json" || ! -f "${model}/model.safetensors.index.json" ]]; then
        echo "Incomplete HF checkpoint: ${model}" >&2
        return 1
    fi
    if has_prior_math500_replicates "${run_key}" "${mode}" "${target_k}"; then
        echo "Reusing prior MATH-500 replicate ${replicate}: ${run_key} ${mode} K=${target_k}"
        REUSED=$((REUSED + 1))
        return
    fi
    if already_recorded "${run_tag}"; then
        echo "Skipping recorded condition: ${run_tag}"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    local -a policy_args=()
    if [[ "${mode}" == "reference" ]]; then
        # At and below K=8, route the native top eight and truncate afterward.
        # Above K=8, route K experts but scale against the top-eight mass.
        if ((target_k <= 8)); then
            router_k=8
        fi
        policy_args+=(
            -e OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_MODE=truncate
            -e OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_SEED=0
            -e "OLMO_EVAL_VLLM_QWEN_EXPERT_ROUTER_K=${router_k}"
            -e "OLMO_EVAL_VLLM_QWEN_EXPERT_KEEP_K=${target_k}"
            -e OLMO_EVAL_VLLM_QWEN_EXPERT_REFERENCE_K=8
            -e OLMO_EVAL_VLLM_QWEN_EXPERT_RENORMALIZE=false
            -e VLLM_USE_FLASHINFER_MOE_FP16=0
        )
    fi

    local -a task_args=()
    local task
    for task in ${TASKS}; do
        task_args+=(-t "${task}")
        if [[ -n "${LIMIT}" ]]; then
            task_args+=(-o "limit=${LIMIT}")
        fi
        task_args+=(-o "sampling_params.max_tokens=${MAX_TOKENS}")
    done
    local -a cmd=(
        env "PYTHONPATH=${SNAPSHOT}/src"
        uv run --project "${LAUNCH_CWD}" --frozen --no-group vllm olmo-eval beaker launch
        -H codex_python
        -o "provider.num_instances=${NUM_INSTANCES}"
        -o "provider.kwargs.hf_overrides={\"num_experts_per_tok\":${router_k}}"
        -o metrics.collect_gpu=true
        -o "provider.max_model_len=${MAX_MODEL_LEN}"
        -o provider.kwargs.gpu_memory_utilization=0.9
        -o provider.kwargs.max_num_seqs=16
        -o provider.kwargs.startup_timeout=900
        "${policy_args[@]}"
        -m "${model}"
        "${task_args[@]}"
        --name "${name}"
        --cluster "${CLUSTER}"
        --workspace "${WORKSPACE}"
        --priority "${PRIORITY}"
        --group "${GROUP}"
        -e "PYTHONPATH=${SNAPSHOT}/src"
        --secret-env jacobm_HF_TOKEN:HF_TOKEN
        --no-store
        --no-follow
        -y
    )
    if [[ "${DRY_RUN}" == "true" ]]; then
        cmd+=(--dry-run)
    fi

    printf 'Launching %s\n' "${run_tag}"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" == "true" ]]; then
        (cd "${LAUNCH_CWD}" && "${cmd[@]}")
        SUBMITTED=$((SUBMITTED + 1))
        return
    fi

    local output_file experiment_id
    output_file="$(mktemp)"
    trap 'rm -f "${output_file}"' RETURN
    (cd "${LAUNCH_CWD}" && "${cmd[@]}") 2>&1 | tee "${output_file}"
    experiment_id="$(grep -Eo 'https://beaker.org/ex/[A-Za-z0-9]+' "${output_file}" | tail -1 | sed 's#https://beaker.org/ex/##')"
    if [[ -z "${experiment_id}" ]]; then
        echo "No Beaker experiment ID captured for ${run_tag}" >&2
        return 1
    fi

    local -a record_task_args=()
    for task in ${TASKS}; do
        record_task_args+=(--task "${task}")
    done
    python3 "${ROOT_DIR}/scripts/adaptive_experts/record_beaker_job.py" \
        --ledger "${LEDGER}" \
        --phase "${PHASE}" \
        --run-tag "${run_tag}" \
        --expert-count "${target_k}" \
        --model "${model}" \
        "${record_task_args[@]}" \
        --cluster "${CLUSTER}" \
        --workspace "${WORKSPACE}" \
        --priority "${PRIORITY}" \
        --group "${GROUP}" \
        --source-commit "${SOURCE_COMMIT}" \
        --source-hash "${SOURCE_HASH}" \
        --source-snapshot "${SNAPSHOT}" \
        --result-storage beaker_dataset \
        --experiment-id "${experiment_id}" \
        --command "${cmd[*]}"
    echo "Recorded https://beaker.org/ex/${experiment_id}"
    SUBMITTED=$((SUBMITTED + 1))
}

for run_key in ${RUN_KEYS}; do
    for mode in ${MODES}; do
        for target_k in ${EXPERT_COUNTS}; do
            # Reference-scaled K=8 is identical to normalized K=8. Reuse the
            # normalized result in both plotted curves instead of rerunning it.
            if [[ "${mode}" == "reference" && "${target_k}" == "8" ]]; then
                continue
            fi
            for replicate in ${REPLICATES}; do
                launch_one "${run_key}" "${mode}" "${target_k}" "${replicate}"
            done
        done
    done
done

echo "Submission summary: submitted=${SUBMITTED} skipped=${SKIPPED} reused=${REUSED} dry_run=${DRY_RUN}"
