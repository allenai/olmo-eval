#!/usr/bin/env bash
set -euo pipefail

# Launch full or fixed-slice DeepScholar-Bench evaluations.

SCRIPT_NAME="$(basename "$0")"
DRY_RUN=false
QUERY_LIMIT="${QUERY_LIMIT:-}"
START_IDX="${START_IDX:-0}"
STAGE_MAX_TOKENS="${STAGE_MAX_TOKENS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
declare -a SELECTED_MODELS=()

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
GROUP="${GROUP:-}"
CLUSTER="${CLUSTER:-ai2/ceres}"
WORKSPACE="${WORKSPACE:-ai2/olmo-eval-debug}"
PRIORITY="${PRIORITY:-urgent}"
TIMEOUT="${TIMEOUT:-24h}"
GPUS="${GPUS:-2}"
OPENALEX_EMAIL="${OPENALEX_EMAIL:-roryd@allenai.org}"
S2_SECRET="${S2_SECRET:-roryd_S2_API_KEY}"

# slug|Hugging Face ref|provider profile
MODEL_SPECS=(
    "olmo3-7b-instruct|allenai/Olmo-3-7B-Instruct|text"
    "olmo3-7b-think|allenai/Olmo-3-7B-Think|text"
    "qwen35-9b|Qwen/Qwen3.5-9B|qwen35"
    "glm41v-9b-thinking|zai-org/GLM-4.1V-9B-Thinking|unified"
    "gemma4-12b-it|google/gemma-4-12B-it|unified"
    "gemma4-26b-a4b-it|google/gemma-4-26B-A4B-it|unified"
    "gpt-oss-20b|openai/gpt-oss-20b|gpt_oss"
    "qwen35-35b-a3b|Qwen/Qwen3.5-35B-A3B|qwen35"
    "nemotron3-nano-30b-a3b|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16|nemotron"
)

usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} [--dry-run|--print-only] [--start-idx N] [--limit N] [--stage-max-tokens N] [--max-model-len N] [--only-model MODEL ...]

Launch selected full or fixed-slice DeepScholar-Bench evaluations.

Options:
  --dry-run          Print commands without submitting jobs
  --print-only       Alias for --dry-run
  --start-idx N      Start at zero-based query index N (default: 0)
  --limit N          Run N benchmark queries beginning at --start-idx
  --stage-max-tokens N
                     Maximum generation tokens per DeepScholar stage
  --max-model-len N  Provider context length (default: 32768)
  --only-model MODEL Launch only a model slug, full HF ref, or ref basename;
                     repeat to select multiple models
  --help             Show this help

Environment overrides:
  RUN_TAG=${RUN_TAG}
  GROUP=${GROUP}
  CLUSTER=${CLUSTER}
  WORKSPACE=${WORKSPACE}
  PRIORITY=${PRIORITY}
  TIMEOUT=${TIMEOUT}
  GPUS=${GPUS}
  OPENALEX_EMAIL=${OPENALEX_EMAIL}
  S2_SECRET=${S2_SECRET}
  QUERY_LIMIT=${QUERY_LIMIT}
  START_IDX=${START_IDX}
  STAGE_MAX_TOKENS=${STAGE_MAX_TOKENS:-16384}
  MAX_MODEL_LEN=${MAX_MODEL_LEN}
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|--print-only)
            DRY_RUN=true
            shift
            ;;
        --limit)
            if [[ $# -lt 2 ]]; then
                echo "Error: --limit requires a positive integer." >&2
                exit 2
            fi
            QUERY_LIMIT="$2"
            shift 2
            ;;
        --start-idx)
            if [[ $# -lt 2 ]]; then
                echo "Error: --start-idx requires a non-negative integer." >&2
                exit 2
            fi
            START_IDX="$2"
            shift 2
            ;;
        --stage-max-tokens)
            if [[ $# -lt 2 ]]; then
                echo "Error: --stage-max-tokens requires a positive integer." >&2
                exit 2
            fi
            STAGE_MAX_TOKENS="$2"
            shift 2
            ;;
        --max-model-len)
            if [[ $# -lt 2 ]]; then
                echo "Error: --max-model-len requires a positive integer." >&2
                exit 2
            fi
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --only-model)
            if [[ $# -lt 2 ]]; then
                echo "Error: --only-model requires a value." >&2
                exit 2
            fi
            SELECTED_MODELS+=("$2")
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option '$1'." >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "${QUERY_LIMIT}" && ! "${QUERY_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: query limit must be a positive integer, got '${QUERY_LIMIT}'." >&2
    exit 2
fi

if [[ ! "${START_IDX}" =~ ^[0-9]+$ ]]; then
    echo "Error: start index must be a non-negative integer, got '${START_IDX}'." >&2
    exit 2
fi

if [[ -n "${STAGE_MAX_TOKENS}" && ! "${STAGE_MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: stage max tokens must be a positive integer, got '${STAGE_MAX_TOKENS}'." >&2
    exit 2
fi

if [[ ! "${MAX_MODEL_LEN}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: max model length must be a positive integer, got '${MAX_MODEL_LEN}'." >&2
    exit 2
fi

RUN_SCOPE="full"
if [[ -n "${QUERY_LIMIT}" ]]; then
    RUN_SCOPE="dev${QUERY_LIMIT}"
    if [[ "${START_IDX}" != "0" ]]; then
        RUN_SCOPE="${RUN_SCOPE}-from${START_IDX}"
    fi
fi
GROUP="${GROUP:-deepscholar-${RUN_SCOPE}-${RUN_TAG}}"

model_is_selected() {
    local slug=$1
    local model=$2
    local selected

    if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
        return 0
    fi

    for selected in "${SELECTED_MODELS[@]}"; do
        if [[ "${selected}" == "${slug}" || "${selected}" == "${model}" || \
              "${selected}" == "${model##*/}" ]]; then
            return 0
        fi
    done

    return 1
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

launched=0
declare -a failures=()

for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r slug model profile <<<"${spec}"
    if ! model_is_selected "${slug}" "${model}"; then
        continue
    fi

    # Keep the generation ceiling uniform across models. Context can be raised
    # explicitly for a checkpoint whose validated native window is larger.
    stage_max_tokens="${STAGE_MAX_TOKENS:-16384}"

    command=(
        uv run olmo-eval beaker launch
        --yes
        --no-follow
        --name "${slug}-deepscholar-${RUN_SCOPE}-${RUN_TAG}"
        --model "${model}"
        --external-eval deepscholar_bench
        --cluster "${CLUSTER}"
        --workspace "${WORKSPACE}"
        --group "${GROUP}"
        --priority "${PRIORITY}"
        --timeout "${TIMEOUT}"
        --gpus "${GPUS}"
        --eval-arg search_backend=s2
        --eval-arg "start_idx=${START_IDX}"
        --eval-arg allow_partial_generation=true
        --eval-arg "stage_max_tokens=${stage_max_tokens}"
        --provider-kwarg "max_model_len=${MAX_MODEL_LEN}"
        --provider-kwarg startup_timeout=1800
        --secret-env "${S2_SECRET}:S2_API_KEY"
        --env "OPENALEX_EMAIL=${OPENALEX_EMAIL}"
    )

    if [[ -n "${QUERY_LIMIT}" ]]; then
        command+=(--eval-arg "limit=${QUERY_LIMIT}")
    fi

    case "${profile}" in
        text)
            ;;
        unified)
            command+=(--provider-kwarg language_model_only=true)
            ;;
        qwen35)
            command+=(
                --provider-kwarg language_model_only=true
                --provider-kwarg "tensor_parallel_size=${GPUS}"
                --provider-kwarg gdn_prefill_backend=triton
                --provider-kwarg reasoning_parser=qwen3
                --provider-kwarg tool_call_parser=qwen3_coder
            )
            ;;
        gpt_oss)
            command+=(--provider-kwarg tool_call_parser=openai)
            ;;
        nemotron)
            command+=(
                --provider-kwarg trust_remote_code=true
                --provider-kwarg enable_prefix_caching=false
            )
            ;;
        *)
            echo "Error: unknown provider profile '${profile}' for ${model}." >&2
            exit 2
            ;;
    esac

    echo "Preparing ${model} as ${slug}-deepscholar-${RUN_SCOPE}-${RUN_TAG}"
    if [[ "${DRY_RUN}" == "true" ]]; then
        print_command "${command[@]}"
    elif ! "${command[@]}"; then
        failures+=("${model}")
    fi
    launched=$((launched + 1))
done

if [[ ${launched} -eq 0 ]]; then
    echo "Error: no models matched the selection." >&2
    exit 2
fi

if [[ ${#failures[@]} -gt 0 ]]; then
    echo "Failed to submit: ${failures[*]}" >&2
    exit 1
fi

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run complete: ${launched} command(s)."
else
    echo "Submitted ${launched} job(s) to group ${GROUP}."
fi
