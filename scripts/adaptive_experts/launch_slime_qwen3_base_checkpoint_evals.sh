#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="$(dirname "${ROOT_DIR}")"
RUN_ROOT="${PROJECT_DIR}/slime-runs/qwen3-30b-a3b-base-dapo"
HF_ROOT="${RUN_ROOT}/hf-checkpoints"
ORIGIN_HF_DIR="${RUN_ROOT}/hf/Qwen3-30B-A3B-Base"
LEDGER="${LEDGER:-${ROOT_DIR}/notes/beaker_jobs.jsonl}"
SNAPSHOT="${SNAPSHOT:-${PROJECT_DIR}/.run_snapshots/olmo-eval/52060bdc2f8988ab7b74635ddadc00a71ebab5f88fa227c37d4797f265975f35}"
WORKSPACE="${WORKSPACE:-ai2/holmes-testing}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"
GROUP="${GROUP:-adaptive-compute-slime-qwen3-base-eval-sweep-20260725}"
PHASE="${PHASE:-slime-checkpoint-eval-sweep}"
DRY_RUN=false
STEPS="${STEPS:-20 40 60 80 100}"
EVAL_TAG="${EVAL_TAG:-v1-20260725}"
EVAL_KIND="${EVAL_KIND:-math-eval}"
TASK_SET="${TASK_SET:-math-aime}"
TASK_SEED="${TASK_SEED:-42}"
RUN_KEYS="${RUN_KEYS:-k8-normalized k6-normalized k4-reference-k8}"
CLEAN_WORKTREE=""
LAUNCH_CWD="${ROOT_DIR}"

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

if [[ ! "${TASK_SEED}" =~ ^[0-9]+$ ]]; then
    echo "TASK_SEED must be a non-negative integer" >&2
    exit 2
fi

cleanup() {
    if [[ -n "${CLEAN_WORKTREE}" ]]; then
        git -C "${ROOT_DIR}" worktree remove --force "${CLEAN_WORKTREE}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [[ "${DRY_RUN}" == "false" ]]; then
    CLEAN_WORKTREE="$(mktemp -d "${TMPDIR:-/tmp}/olmo-eval-slime-evals.XXXXXX")"
    rmdir "${CLEAN_WORKTREE}"
    git -C "${ROOT_DIR}" worktree add --detach --quiet \
        "${CLEAN_WORKTREE}" e8b88f196767630acaf8a383d12a33d314788d27
    LAUNCH_CWD="${CLEAN_WORKTREE}"
fi

already_recorded() {
    local run_tag=$1
    jq -s -e --arg phase "${PHASE}" --arg tag "${run_tag}" \
        'any(.[]; .phase == $phase and .run_tag == $tag)' "${LEDGER}" >/dev/null
}

launch_one() {
    local run_key=$1
    local run_name=$2
    local step=$3
    local active_k=$4
    local mode=$5
    local model
    if [[ "${step}" == "0" ]]; then
        model="${ORIGIN_HF_DIR}"
    else
        model="${HF_ROOT}/${run_name}-step${step}-hf"
    fi
    local run_tag="qwen3-base-dapo-${run_key}-step${step}-${EVAL_KIND}-${EVAL_TAG}"
    local name="adaptive-${run_tag}"

    if already_recorded "${run_tag}"; then
        echo "Skipping recorded condition: ${run_tag}"
        return
    fi
    if [[ ! -f "${model}/config.json" || ! -f "${model}/model.safetensors.index.json" ]]; then
        echo "HF export is incomplete: ${model}" >&2
        return 1
    fi

    local router_k=${active_k}
    local -a policy_args=()
    if [[ "${mode}" == "reference" ]]; then
        router_k=8
        policy_args+=(
            -e OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_MODE=truncate
            -e OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_SEED=0
            -e OLMO_EVAL_VLLM_QWEN_EXPERT_ROUTER_K=8
            -e VLLM_USE_FLASHINFER_MOE_FP16=0
            -e "OLMO_EVAL_VLLM_QWEN_EXPERT_KEEP_K=${active_k}"
            -e OLMO_EVAL_VLLM_QWEN_EXPERT_REFERENCE_K=8
            -e OLMO_EVAL_VLLM_QWEN_EXPERT_RENORMALIZE=false
        )
    fi

    local -a task_args record_task_args
    case "${TASK_SET}" in
        math-aime)
            task_args=(
                -t math500:chat
                -o sampling_params.max_tokens=20480
                -o "seed=${TASK_SEED}"
                -t aime_2025:pass_at_32
                -o sampling_params.max_tokens=20480
                -o "seed=${TASK_SEED}"
            )
            record_task_args=(--task math500:chat --task aime_2025:pass_at_32)
            ;;
        math500)
            task_args=(
                -t math500:chat
                -o sampling_params.max_tokens=20480
                -o "seed=${TASK_SEED}"
            )
            record_task_args=(--task math500:chat)
            ;;
        *)
            echo "Unknown task set: ${TASK_SET}" >&2
            return 2
            ;;
    esac

    local -a cmd=(
        env "PYTHONPATH=${SNAPSHOT}/src"
        uv run --project "${LAUNCH_CWD}" --frozen --no-group vllm olmo-eval beaker launch
        -H codex_python
        -o provider.num_instances=4
        -o "provider.kwargs.hf_overrides={\"num_experts_per_tok\":${router_k}}"
        -o metrics.collect_gpu=true
        -o provider.max_model_len=32768
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

    python3 "${ROOT_DIR}/scripts/adaptive_experts/record_beaker_job.py" \
        --ledger "${LEDGER}" \
        --phase "${PHASE}" \
        --run-tag "${run_tag}" \
        --expert-count "${active_k}" \
        --model "${model}" \
        "${record_task_args[@]}" \
        --cluster "${CLUSTER}" \
        --workspace "${WORKSPACE}" \
        --priority "${PRIORITY}" \
        --group "${GROUP}" \
        --source-commit e8b88f196767630acaf8a383d12a33d314788d27 \
        --source-hash 52060bdc2f8988ab7b74635ddadc00a71ebab5f88fa227c37d4797f265975f35 \
        --source-snapshot "${SNAPSHOT}" \
        --result-storage beaker_dataset \
        --experiment-id "${experiment_id}" \
        --command "${cmd[*]}"
    echo "Recorded https://beaker.org/ex/${experiment_id}"
}

for step in ${STEPS}; do
    if [[ ! "${step}" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "Invalid checkpoint step: ${step}" >&2
        exit 2
    fi
    for run_key in ${RUN_KEYS}; do
        case "${run_key}" in
            k8-normalized)
                launch_one \
                    k8-normalized \
                    slime-qwen3-base-dapo-k8-pilot-100step-8gpu-v5-20260725 \
                    "${step}" 8 normalized
                ;;
            k6-normalized)
                launch_one \
                    k6-normalized \
                    slime-qwen3-base-dapo-k6-pilot-100step-8gpu-v1-20260725 \
                    "${step}" 6 normalized
                ;;
            k6-reference-k8)
                launch_one \
                    k6-reference-k8 \
                    slime-qwen3-base-dapo-k6-reference-k8-500step-8gpu-v1-20260726 \
                    "${step}" 6 reference
                ;;
            k8-trained-k6-eval)
                launch_one \
                    k8-trained-k6-eval \
                    slime-qwen3-base-dapo-k8-pilot-100step-8gpu-v5-20260725 \
                    "${step}" 6 normalized
                ;;
            k6-trained-k8-eval)
                launch_one \
                    k6-trained-k8-eval \
                    slime-qwen3-base-dapo-k6-pilot-100step-8gpu-v1-20260725 \
                    "${step}" 8 normalized
                ;;
            k4-reference-k8)
                launch_one \
                    k4-reference-k8 \
                    slime-qwen3-base-dapo-k4-reference-k8-pilot-100step-8gpu-v1-20260725 \
                    "${step}" 4 reference
                ;;
            k10-normalized)
                launch_one \
                    k10-normalized \
                    slime-qwen3-base-dapo-k10-normalized-pilot-100step-8gpu-v1-20260806 \
                    "${step}" 10 normalized
                ;;
            k10-reference-k8)
                launch_one \
                    k10-reference-k8 \
                    slime-qwen3-base-dapo-k10-reference-k8-pilot-100step-8gpu-v1-20260806 \
                    "${step}" 10 reference
                ;;
            mixed-cost-reference-k4)
                launch_one \
                    mixed-cost-reference-k4 \
                    slime-qwen3-base-dapo-mixed-k4-k6-k8-reference-pilot-100step-8gpu-v1-20260803 \
                    "${step}" 4 reference
                ;;
            mixed-cost-reference-k6)
                launch_one \
                    mixed-cost-reference-k6 \
                    slime-qwen3-base-dapo-mixed-k4-k6-k8-reference-pilot-100step-8gpu-v1-20260803 \
                    "${step}" 6 reference
                ;;
            mixed-cost-native-k8)
                launch_one \
                    mixed-cost-native-k8 \
                    slime-qwen3-base-dapo-mixed-k4-k6-k8-reference-pilot-100step-8gpu-v1-20260803 \
                    "${step}" 8 normalized
                ;;
            mixed-neutral-reference-k4)
                launch_one \
                    mixed-neutral-reference-k4 \
                    slime-qwen3-base-dapo-mixed-k4-k6-k8-reference-neutral-pilot-100step-8gpu-v1-20260803 \
                    "${step}" 4 reference
                ;;
            mixed-neutral-reference-k6)
                launch_one \
                    mixed-neutral-reference-k6 \
                    slime-qwen3-base-dapo-mixed-k4-k6-k8-reference-neutral-pilot-100step-8gpu-v1-20260803 \
                    "${step}" 6 reference
                ;;
            mixed-neutral-native-k8)
                launch_one \
                    mixed-neutral-native-k8 \
                    slime-qwen3-base-dapo-mixed-k4-k6-k8-reference-neutral-pilot-100step-8gpu-v1-20260803 \
                    "${step}" 8 normalized
                ;;
            preallocated-reference-k4)
                launch_one \
                    preallocated-reference-k4 \
                    slime-qwen3-base-dapo-preallocated-k-cheapest-any-success-reference-neutral-pilot-100step-8gpu-v1-20260810 \
                    "${step}" 4 reference
                ;;
            preallocated-reference-k6)
                launch_one \
                    preallocated-reference-k6 \
                    slime-qwen3-base-dapo-preallocated-k-cheapest-any-success-reference-neutral-pilot-100step-8gpu-v1-20260810 \
                    "${step}" 6 reference
                ;;
            preallocated-native-k8)
                launch_one \
                    preallocated-native-k8 \
                    slime-qwen3-base-dapo-preallocated-k-cheapest-any-success-reference-neutral-pilot-100step-8gpu-v1-20260810 \
                    "${step}" 8 normalized
                ;;
            mixed-k4-k8-k12-reference-k4)
                launch_one \
                    mixed-k4-k8-k12-reference-k4 \
                    slime-qwen3-base-dapo-mixed-k4-k8-k12-reference-neutral-pilot-100step-8gpu-v1-20260810 \
                    "${step}" 4 reference
                ;;
            mixed-k4-k8-k12-native-k8)
                launch_one \
                    mixed-k4-k8-k12-native-k8 \
                    slime-qwen3-base-dapo-mixed-k4-k8-k12-reference-neutral-pilot-100step-8gpu-v1-20260810 \
                    "${step}" 8 normalized
                ;;
            mixed-k4-k8-k12-reference-k12)
                launch_one \
                    mixed-k4-k8-k12-reference-k12 \
                    slime-qwen3-base-dapo-mixed-k4-k8-k12-reference-neutral-pilot-100step-8gpu-v1-20260810 \
                    "${step}" 12 reference
                ;;
            *)
                echo "Unknown run key: ${run_key}" >&2
                exit 2
                ;;
        esac
    done
done
