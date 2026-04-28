#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
DRY_RUN=false
GEMMA_ONLY=false
declare -a SELECTED_GROUPS=()
declare -a RAW_SELECTED_MODELS=()
declare -a RAW_SELECTED_SUITES=()
declare -a SELECTED_MODELS=()
declare -a SELECTED_SUITES=()

GROUP="${GROUP:-olmo-eval-olmo3-baselines-04272026}"
WORKSPACE="${WORKSPACE:-ai2/olmo-eval-debug}"
BUDGET="${BUDGET:-ai2/oe-base}"
CLUSTER="${CLUSTER:-h100}"
EXEC_HARNESS="${EXEC_HARNESS:-${HARNESS:-codex_universal}}"
NON_EXEC_HARNESS="${NON_EXEC_HARNESS:-default}"
TASK_PRIORITY="${TASK_PRIORITY:-urgent}"
MODAL_ENVIRONMENT="${MODAL_ENVIRONMENT:-oe-eval}"
PROVIDER_NUM_INSTANCES="${PROVIDER_NUM_INSTANCES:-8}"

usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} [--dry-run] [--gemma-only] [--only-group GROUP ...] [--only-suite SUITE ...] [--only-model MODEL ...]

Launches Beaker variants for the OLMo 3 baseline sweep across:
  1. Code execution tasks
  2. MCQA stem tasks
  3. MCQA non-stem tasks
  4. Generation tasks
  5. Math tasks
  6. Easy QA tasks
  7. Easy math + easy code tasks

Each task group is launched for the standard baseline model bundle, for Qwen3 with
thinking disabled, and for Gemma. Nemotron Nano is temporarily disabled in this
launcher.

Options:
  --dry-run     Print the commands without launching them
  --gemma-only  Launch only the Gemma variants
  --only-group  Restrict launches to specific task groups (repeatable)
  --only-suite  Restrict launches to specific suites (repeatable)
  --only-model  Restrict launches to specific models (repeatable)
  --help        Show this help

Valid group names:
  code_exec, mcqa_stem, mcqa_non_stem, gen, math, easy_qa, easy_math_code

Legacy aliases:
  mcqa      -> mcqa_stem + mcqa_non_stem
  gen_math  -> gen + math

Valid suite names:
  olmobase:code
  olmobase:easy:code:bpb
  olmobase:easy:math:bpb
  olmobase:easy:qa:bpb
  olmobase:easy:qa:rc
  olmobase:gen
  olmobase:math
  olmobase:mcqa_non_stem
  olmobase:mcqa_stem

Models may be specified as full Hugging Face ids or trailing names such as
gemma-2-9b, qwen3-8b, or mimo-7b-base.

Examples:
  ${SCRIPT_NAME} --gemma-only --only-group mcqa_stem --only-group math
  ${SCRIPT_NAME} --only-group code_exec
  ${SCRIPT_NAME} --only-suite olmobase:math --only-suite olmobase:code
  ${SCRIPT_NAME} --only-model gemma-2-9b --only-model qwen3-8b --only-group gen

Environment overrides:
  GROUP=${GROUP}
  WORKSPACE=${WORKSPACE}
  BUDGET=${BUDGET}
  CLUSTER=${CLUSTER}
  EXEC_HARNESS=${EXEC_HARNESS}
  NON_EXEC_HARNESS=${NON_EXEC_HARNESS}
  TASK_PRIORITY=${TASK_PRIORITY}
  MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT}
  PROVIDER_NUM_INSTANCES=${PROVIDER_NUM_INSTANCES}
EOF
}

add_selected_groups_for_input() {
    case "$1" in
        code_exec|code-exec|code|exec)
            add_selected_group "code_exec"
            ;;
        mcqa)
            add_selected_group "mcqa_stem"
            add_selected_group "mcqa_non_stem"
            ;;
        mcqa_stem|mcqa-stem)
            add_selected_group "mcqa_stem"
            ;;
        mcqa_non_stem|mcqa-non-stem|mcqa_nonstem|mcqa-nonstem)
            add_selected_group "mcqa_non_stem"
            ;;
        gen_math|gen-math|gen+math)
            add_selected_group "gen"
            add_selected_group "math"
            ;;
        gen)
            add_selected_group "gen"
            ;;
        math)
            add_selected_group "math"
            ;;
        easy_qa|easy-qa)
            add_selected_group "easy_qa"
            ;;
        easy_math_code|easy-math-code|easy_math|easy-math|easy_code|easy-code)
            add_selected_group "easy_math_code"
            ;;
        *)
            return 1
            ;;
    esac
}

add_selected_group() {
    local group=$1
    local existing

    for existing in "${SELECTED_GROUPS[@]-}"; do
        if [[ "${existing}" == "${group}" ]]; then
            return 0
        fi
    done

    SELECTED_GROUPS+=("${group}")
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --gemma-only)
            GEMMA_ONLY=true
            shift
            ;;
        --only-group)
            if [[ $# -lt 2 ]]; then
                echo "Error: --only-group requires a value." >&2
                exit 1
            fi
            if ! add_selected_groups_for_input "$2"; then
                echo "Error: Unknown group '$2'." >&2
                echo "Valid groups: code_exec, mcqa_stem, mcqa_non_stem, gen, math, easy_qa, easy_math_code" >&2
                exit 1
            fi
            shift 2
            ;;
        --only-suite)
            if [[ $# -lt 2 ]]; then
                echo "Error: --only-suite requires a value." >&2
                exit 1
            fi
            RAW_SELECTED_SUITES+=("$2")
            shift 2
            ;;
        --only-model)
            if [[ $# -lt 2 ]]; then
                echo "Error: --only-model requires a value." >&2
                exit 1
            fi
            RAW_SELECTED_MODELS+=("$2")
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage." >&2
            exit 1
            ;;
    esac
done

group_is_selected() {
    local wanted=$1
    local selected

    if [[ "${#SELECTED_GROUPS[@]}" -eq 0 ]]; then
        return 0
    fi

    for selected in "${SELECTED_GROUPS[@]-}"; do
        if [[ "${selected}" == "${wanted}" ]]; then
            return 0
        fi
    done

    return 1
}

code_exec_tasks=(
    "olmobase:code"
)

mcqa_stem_tasks=(
    "olmobase:mcqa_stem"
)

mcqa_non_stem_tasks=(
    "olmobase:mcqa_non_stem"
)

gen_tasks=(
    "olmobase:gen"
)

math_tasks=(
    "olmobase:math"
)

non_exec_easy_qa_tasks=(
    "olmobase:easy:qa:rc"
    "olmobase:easy:qa:bpb"
)

non_exec_easy_math_code_tasks=(
    "olmobase:easy:math:bpb"
    "olmobase:easy:code:bpb"
)

baseline_models=(
    "allenai/OLMo-2-1124-7B"
    "allenai/Olmo-3-1025-7B"
    "marin-community/marin-8b-base"
    "swiss-ai/Apertus-8B-2509"
    "almanach/Gaperon-1125-8B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen2.5-7B"
    # Temporarily disabled while we sort out the current Nemotron launch issue.
    # "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    "ibm-granite/granite-3.3-8b-base"
    "XiaomiMiMo/MiMo-7B-Base"
)

gemma_models=(
    "google/gemma-2-9b"
)

NEMOTRON_NANO_MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2"
QWEN3_MODEL="Qwen/Qwen3-8B"

baseline_launch_args=(
    "-o" "provider.num_instances=${PROVIDER_NUM_INSTANCES}"
    "-o" "provider.trust_remote_code=true"
)

qwen3_launch_args=(
    "-o" "provider.num_instances=${PROVIDER_NUM_INSTANCES}"
    "-o" "provider.trust_remote_code=true"
    # Match completion-style baselines by disabling Qwen3 thinking mode at the vLLM server.
    "-o" "provider.kwargs.default_chat_template_kwargs.enable_thinking=false"
)

nemotron_launch_args=(
    "-o" "provider.num_instances=${PROVIDER_NUM_INSTANCES}"
    "-o" "provider.trust_remote_code=true"
    "-o" "provider.kwargs.enable_prefix_caching=false"
)

gemma_launch_args=(
    "-o" "provider.num_instances=${PROVIDER_NUM_INSTANCES}"
    "-o" "provider.kwargs.attention_backend=TRITON_ATTN"
)

exec_only_args=(
    "-o" 'sandboxes={"mode":"modal","instances":64, "min_instances": 56, "registry_auth":{"provider":"gcp"}}'
    "-e" "MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT}"
    "--secret-env" "ai2-tylerm_MODAL_TOKEN_ID:MODAL_TOKEN_ID"
    "--secret-env" "ai2-tylerm_MODAL_TOKEN_SECRET:MODAL_TOKEN_SECRET"
)

common_tail_args=(
    "-w" "${WORKSPACE}"
    "-B" "${BUDGET}"
    "--cluster" "${CLUSTER}"
    "--group" "${GROUP}"
    "--store"
    "--inspect"
    "--gcp-credentials"
    "--no-follow"
    "-y"
)

to_lower() {
    printf "%s" "$1" | tr '[:upper:]' '[:lower:]'
}

normalize_suite_name() {
    case "$1" in
        olmobase:code|code)
            printf "olmobase:code"
            ;;
        olmobase:mcqa_stem|mcqa_stem|mcqa-stem)
            printf "olmobase:mcqa_stem"
            ;;
        olmobase:mcqa_non_stem|mcqa_non_stem|mcqa-non-stem|mcqa_nonstem|mcqa-nonstem)
            printf "olmobase:mcqa_non_stem"
            ;;
        olmobase:gen|gen)
            printf "olmobase:gen"
            ;;
        olmobase:math|math)
            printf "olmobase:math"
            ;;
        olmobase:easy:qa:rc|easy:qa:rc|easy_qa_rc|easy-qa-rc)
            printf "olmobase:easy:qa:rc"
            ;;
        olmobase:easy:qa:bpb|easy:qa:bpb|easy_qa_bpb|easy-qa-bpb)
            printf "olmobase:easy:qa:bpb"
            ;;
        olmobase:easy:math:bpb|easy:math:bpb|easy_math_bpb|easy-math-bpb)
            printf "olmobase:easy:math:bpb"
            ;;
        olmobase:easy:code:bpb|easy:code:bpb|easy_code_bpb|easy-code-bpb)
            printf "olmobase:easy:code:bpb"
            ;;
        *)
            return 1
            ;;
    esac
}

normalize_model_name() {
    local candidate=$1
    local candidate_lower
    local model
    local model_lower
    local short_lower

    candidate_lower="$(to_lower "${candidate}")"

    case "${candidate_lower}" in
        gemma|gemma-2|gemma2|gemma-2-9b)
            printf "google/gemma-2-9b"
            return 0
            ;;
    esac

    for model in "${baseline_models[@]}" "${gemma_models[@]}"; do
        model_lower="$(to_lower "${model}")"
        short_lower="$(to_lower "${model##*/}")"
        if [[ "${candidate_lower}" == "${model_lower}" || "${candidate_lower}" == "${short_lower}" ]]; then
            printf "%s" "${model}"
            return 0
        fi
    done

    return 1
}

print_valid_suites() {
    local suite

    for suite in \
        "${code_exec_tasks[@]}" \
        "${mcqa_stem_tasks[@]}" \
        "${mcqa_non_stem_tasks[@]}" \
        "${gen_tasks[@]}" \
        "${math_tasks[@]}" \
        "${non_exec_easy_qa_tasks[@]}" \
        "${non_exec_easy_math_code_tasks[@]}"; do
        echo "  ${suite}" >&2
    done
}

print_valid_models() {
    local model

    for model in "${baseline_models[@]}" "${gemma_models[@]}"; do
        echo "  ${model}" >&2
    done
}

add_selected_suite() {
    local suite=$1
    local existing

    for existing in "${SELECTED_SUITES[@]-}"; do
        if [[ "${existing}" == "${suite}" ]]; then
            return 0
        fi
    done

    SELECTED_SUITES+=("${suite}")
}

add_selected_model() {
    local model=$1
    local existing

    for existing in "${SELECTED_MODELS[@]-}"; do
        if [[ "${existing}" == "${model}" ]]; then
            return 0
        fi
    done

    SELECTED_MODELS+=("${model}")
}

suite_is_selected() {
    local wanted=$1
    local selected

    if [[ "${#SELECTED_SUITES[@]}" -eq 0 ]]; then
        return 0
    fi

    for selected in "${SELECTED_SUITES[@]-}"; do
        if [[ "${selected}" == "${wanted}" ]]; then
            return 0
        fi
    done

    return 1
}

model_is_selected() {
    local wanted=$1
    local selected

    if [[ "${#SELECTED_MODELS[@]}" -eq 0 ]]; then
        return 0
    fi

    for selected in "${SELECTED_MODELS[@]-}"; do
        if [[ "${selected}" == "${wanted}" ]]; then
            return 0
        fi
    done

    return 1
}

filter_tasks() {
    local target=$1
    shift
    local filtered=()
    local task

    for task in "$@"; do
        if suite_is_selected "${task}"; then
            filtered+=("${task}")
        fi
    done

    if [[ "${#filtered[@]}" -gt 0 ]]; then
        eval "${target}=(\"\${filtered[@]}\")"
    else
        eval "${target}=()"
    fi
}

filter_models() {
    local target=$1
    shift
    local filtered=()
    local model

    for model in "$@"; do
        if model_is_selected "${model}"; then
            filtered+=("${model}")
        fi
    done

    if [[ "${#filtered[@]}" -gt 0 ]]; then
        eval "${target}=(\"\${filtered[@]}\")"
    else
        eval "${target}=()"
    fi
}

if [[ "${#RAW_SELECTED_SUITES[@]}" -gt 0 ]]; then
    for raw_selected_suite in "${RAW_SELECTED_SUITES[@]}"; do
        if ! normalized_suite="$(normalize_suite_name "${raw_selected_suite}")"; then
            echo "Error: Unknown suite '${raw_selected_suite}'." >&2
            echo "Valid suites:" >&2
            print_valid_suites
            exit 1
        fi
        add_selected_suite "${normalized_suite}"
    done
fi

if [[ "${#RAW_SELECTED_MODELS[@]}" -gt 0 ]]; then
    for raw_selected_model in "${RAW_SELECTED_MODELS[@]}"; do
        if ! normalized_model="$(normalize_model_name "${raw_selected_model}")"; then
            echo "Error: Unknown model '${raw_selected_model}'." >&2
            echo "Valid models:" >&2
            print_valid_models
            exit 1
        fi
        add_selected_model "${normalized_model}"
    done
fi

declare -a selected_code_exec_tasks
declare -a selected_mcqa_stem_tasks
declare -a selected_mcqa_non_stem_tasks
declare -a selected_gen_tasks
declare -a selected_math_tasks
declare -a selected_non_exec_easy_qa_tasks
declare -a selected_non_exec_easy_math_code_tasks
declare -a selected_baseline_models
declare -a selected_standard_baseline_models
declare -a selected_qwen3_models
declare -a selected_nemotron_nano_models
declare -a selected_gemma_models

filter_tasks selected_code_exec_tasks "${code_exec_tasks[@]}"
filter_tasks selected_mcqa_stem_tasks "${mcqa_stem_tasks[@]}"
filter_tasks selected_mcqa_non_stem_tasks "${mcqa_non_stem_tasks[@]}"
filter_tasks selected_gen_tasks "${gen_tasks[@]}"
filter_tasks selected_math_tasks "${math_tasks[@]}"
filter_tasks selected_non_exec_easy_qa_tasks "${non_exec_easy_qa_tasks[@]}"
filter_tasks selected_non_exec_easy_math_code_tasks "${non_exec_easy_math_code_tasks[@]}"
filter_models selected_baseline_models "${baseline_models[@]}"
filter_models selected_gemma_models "${gemma_models[@]}"

for model in "${selected_baseline_models[@]-}"; do
    if [[ "${model}" == "${NEMOTRON_NANO_MODEL}" ]]; then
        selected_nemotron_nano_models+=("${model}")
    elif [[ "${model}" == "${QWEN3_MODEL}" ]]; then
        selected_qwen3_models+=("${model}")
    else
        selected_standard_baseline_models+=("${model}")
    fi
done

declare -a current_cmd

start_command() {
    local harness=$1

    current_cmd=(
        "olmo-eval" "beaker" "launch"
        "-H" "${harness}"
    )
}

append_args() {
    current_cmd+=("$@")
}

append_models() {
    local model

    for model in "$@"; do
        current_cmd+=("-m" "${model}")
    done
}

append_tasks() {
    local task

    for task in "$@"; do
        current_cmd+=("-t" "${task}@${TASK_PRIORITY}")
    done
}

save_current_cmd() {
    local target=$1
    eval "${target}=(\"\${current_cmd[@]}\")"
}

build_variant_cmd() {
    local target=$1
    local harness=$2
    local launch_args_name=$3
    local models_name=$4
    local tasks_name=$5
    local include_exec_only=${6:-false}

    start_command "${harness}"
    eval "append_args \"\${${launch_args_name}[@]}\""
    if [[ "${include_exec_only}" == "true" ]]; then
        append_args "${exec_only_args[@]}"
    fi
    eval "append_models \"\${${models_name}[@]-}\""
    eval "append_tasks \"\${${tasks_name}[@]-}\""
    append_args "${common_tail_args[@]}"
    save_current_cmd "${target}"
}

print_command() {
    local cmd=("$@")
    local lines=()
    local index=3
    local last_line_index

    lines+=("$(format_tokens "${cmd[@]:0:3}")")

    while [[ "${index}" -lt "${#cmd[@]}" ]]; do
        if arg_takes_value "${cmd[${index}]}"; then
            lines+=("$(format_tokens "${cmd[${index}]}" "${cmd[$((index + 1))]}")")
            index=$((index + 2))
        else
            lines+=("$(format_tokens "${cmd[${index}]}")")
            index=$((index + 1))
        fi
    done

    last_line_index=$((${#lines[@]} - 1))

    for index in "${!lines[@]}"; do
        if [[ "${index}" -eq 0 ]]; then
            if [[ "${index}" -lt "${last_line_index}" ]]; then
                printf '%s \
' "${lines[${index}]}"
            else
                printf "%s\n" "${lines[${index}]}"
            fi
        else
            if [[ "${index}" -lt "${last_line_index}" ]]; then
                printf '  %s \
' "${lines[${index}]}"
            else
                printf "  %s\n" "${lines[${index}]}"
            fi
        fi
    done
}

arg_takes_value() {
    case "$1" in
        -H|-o|-e|--secret-env|-m|-t|-w|-B|--cluster|--group)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

quote_token() {
    local token=$1
    local escaped

    if [[ "${token}" =~ ^[A-Za-z0-9_./:@=%,+-]+$ ]]; then
        printf "%s" "${token}"
    else
        escaped=${token//\'/\'\\\'\'}
        printf "'%s'" "${escaped}"
    fi
}

format_tokens() {
    local token
    local formatted=""
    local quoted

    for token in "$@"; do
        quoted="$(quote_token "${token}")"
        if [[ -n "${formatted}" ]]; then
            formatted="${formatted} ${quoted}"
        else
            formatted="${quoted}"
        fi
    done

    printf "%s" "${formatted}"
}

run_variant() {
    local label=$1
    shift
    local cmd=("$@")

    echo "========================================="
    echo "${label}"
    echo "========================================="
    print_command "${cmd[@]}"
    echo ""

    if [[ "${DRY_RUN}" == "false" ]]; then
        "${cmd[@]}"
        echo ""
    fi
}

declare -a baseline_exec_cmd
declare -a baseline_mcqa_stem_cmd
declare -a baseline_mcqa_non_stem_cmd
declare -a baseline_gen_cmd
declare -a baseline_math_cmd
declare -a baseline_non_exec_easy_qa_cmd
declare -a baseline_non_exec_easy_math_code_cmd
declare -a qwen3_exec_cmd
declare -a qwen3_mcqa_stem_cmd
declare -a qwen3_mcqa_non_stem_cmd
declare -a qwen3_gen_cmd
declare -a qwen3_math_cmd
declare -a qwen3_non_exec_easy_qa_cmd
declare -a qwen3_non_exec_easy_math_code_cmd
declare -a nemotron_exec_cmd
declare -a nemotron_mcqa_stem_cmd
declare -a nemotron_mcqa_non_stem_cmd
declare -a nemotron_gen_cmd
declare -a nemotron_math_cmd
declare -a nemotron_non_exec_easy_qa_cmd
declare -a nemotron_non_exec_easy_math_code_cmd
declare -a gemma_exec_cmd
declare -a gemma_mcqa_stem_cmd
declare -a gemma_mcqa_non_stem_cmd
declare -a gemma_gen_cmd
declare -a gemma_math_cmd
declare -a gemma_non_exec_easy_qa_cmd
declare -a gemma_non_exec_easy_math_code_cmd

build_variant_cmd baseline_exec_cmd "${EXEC_HARNESS}" baseline_launch_args selected_standard_baseline_models selected_code_exec_tasks true
build_variant_cmd baseline_mcqa_stem_cmd "${NON_EXEC_HARNESS}" baseline_launch_args selected_standard_baseline_models selected_mcqa_stem_tasks
build_variant_cmd baseline_mcqa_non_stem_cmd "${NON_EXEC_HARNESS}" baseline_launch_args selected_standard_baseline_models selected_mcqa_non_stem_tasks
build_variant_cmd baseline_gen_cmd "${NON_EXEC_HARNESS}" baseline_launch_args selected_standard_baseline_models selected_gen_tasks
build_variant_cmd baseline_math_cmd "${NON_EXEC_HARNESS}" baseline_launch_args selected_standard_baseline_models selected_math_tasks
build_variant_cmd baseline_non_exec_easy_qa_cmd "${NON_EXEC_HARNESS}" baseline_launch_args selected_standard_baseline_models selected_non_exec_easy_qa_tasks
build_variant_cmd baseline_non_exec_easy_math_code_cmd "${NON_EXEC_HARNESS}" baseline_launch_args selected_standard_baseline_models selected_non_exec_easy_math_code_tasks
build_variant_cmd qwen3_exec_cmd "${EXEC_HARNESS}" qwen3_launch_args selected_qwen3_models selected_code_exec_tasks true
build_variant_cmd qwen3_mcqa_stem_cmd "${NON_EXEC_HARNESS}" qwen3_launch_args selected_qwen3_models selected_mcqa_stem_tasks
build_variant_cmd qwen3_mcqa_non_stem_cmd "${NON_EXEC_HARNESS}" qwen3_launch_args selected_qwen3_models selected_mcqa_non_stem_tasks
build_variant_cmd qwen3_gen_cmd "${NON_EXEC_HARNESS}" qwen3_launch_args selected_qwen3_models selected_gen_tasks
build_variant_cmd qwen3_math_cmd "${NON_EXEC_HARNESS}" qwen3_launch_args selected_qwen3_models selected_math_tasks
build_variant_cmd qwen3_non_exec_easy_qa_cmd "${NON_EXEC_HARNESS}" qwen3_launch_args selected_qwen3_models selected_non_exec_easy_qa_tasks
build_variant_cmd qwen3_non_exec_easy_math_code_cmd "${NON_EXEC_HARNESS}" qwen3_launch_args selected_qwen3_models selected_non_exec_easy_math_code_tasks
build_variant_cmd nemotron_exec_cmd "${EXEC_HARNESS}" nemotron_launch_args selected_nemotron_nano_models selected_code_exec_tasks true
build_variant_cmd nemotron_mcqa_stem_cmd "${NON_EXEC_HARNESS}" nemotron_launch_args selected_nemotron_nano_models selected_mcqa_stem_tasks
build_variant_cmd nemotron_mcqa_non_stem_cmd "${NON_EXEC_HARNESS}" nemotron_launch_args selected_nemotron_nano_models selected_mcqa_non_stem_tasks
build_variant_cmd nemotron_gen_cmd "${NON_EXEC_HARNESS}" nemotron_launch_args selected_nemotron_nano_models selected_gen_tasks
build_variant_cmd nemotron_math_cmd "${NON_EXEC_HARNESS}" nemotron_launch_args selected_nemotron_nano_models selected_math_tasks
build_variant_cmd nemotron_non_exec_easy_qa_cmd "${NON_EXEC_HARNESS}" nemotron_launch_args selected_nemotron_nano_models selected_non_exec_easy_qa_tasks
build_variant_cmd nemotron_non_exec_easy_math_code_cmd "${NON_EXEC_HARNESS}" nemotron_launch_args selected_nemotron_nano_models selected_non_exec_easy_math_code_tasks
build_variant_cmd gemma_exec_cmd "${EXEC_HARNESS}" gemma_launch_args selected_gemma_models selected_code_exec_tasks true
build_variant_cmd gemma_mcqa_stem_cmd "${NON_EXEC_HARNESS}" gemma_launch_args selected_gemma_models selected_mcqa_stem_tasks
build_variant_cmd gemma_mcqa_non_stem_cmd "${NON_EXEC_HARNESS}" gemma_launch_args selected_gemma_models selected_mcqa_non_stem_tasks
build_variant_cmd gemma_gen_cmd "${NON_EXEC_HARNESS}" gemma_launch_args selected_gemma_models selected_gen_tasks
build_variant_cmd gemma_math_cmd "${NON_EXEC_HARNESS}" gemma_launch_args selected_gemma_models selected_math_tasks
build_variant_cmd gemma_non_exec_easy_qa_cmd "${NON_EXEC_HARNESS}" gemma_launch_args selected_gemma_models selected_non_exec_easy_qa_tasks
build_variant_cmd gemma_non_exec_easy_math_code_cmd "${NON_EXEC_HARNESS}" gemma_launch_args selected_gemma_models selected_non_exec_easy_math_code_tasks

planned_variants=0

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "code_exec" \
    && [[ "${#selected_standard_baseline_models[@]}" -gt 0 ]] && [[ "${#selected_code_exec_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "code_exec" \
    && [[ "${#selected_qwen3_models[@]}" -gt 0 ]] && [[ "${#selected_code_exec_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "mcqa_stem" \
    && [[ "${#selected_standard_baseline_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_stem_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "mcqa_stem" \
    && [[ "${#selected_qwen3_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_stem_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "mcqa_non_stem" \
    && [[ "${#selected_standard_baseline_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_non_stem_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "mcqa_non_stem" \
    && [[ "${#selected_qwen3_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_non_stem_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "gen" \
    && [[ "${#selected_standard_baseline_models[@]}" -gt 0 ]] && [[ "${#selected_gen_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "gen" \
    && [[ "${#selected_qwen3_models[@]}" -gt 0 ]] && [[ "${#selected_gen_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "math" \
    && [[ "${#selected_standard_baseline_models[@]}" -gt 0 ]] && [[ "${#selected_math_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "math" \
    && [[ "${#selected_qwen3_models[@]}" -gt 0 ]] && [[ "${#selected_math_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "easy_qa" \
    && [[ "${#selected_standard_baseline_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_qa_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "easy_qa" \
    && [[ "${#selected_qwen3_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_qa_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "easy_math_code" \
    && [[ "${#selected_standard_baseline_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_math_code_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "easy_math_code" \
    && [[ "${#selected_qwen3_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_math_code_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "code_exec" \
    && [[ "${#selected_nemotron_nano_models[@]}" -gt 0 ]] && [[ "${#selected_code_exec_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "mcqa_stem" \
    && [[ "${#selected_nemotron_nano_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_stem_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "mcqa_non_stem" \
    && [[ "${#selected_nemotron_nano_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_non_stem_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "gen" \
    && [[ "${#selected_nemotron_nano_models[@]}" -gt 0 ]] && [[ "${#selected_gen_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "math" \
    && [[ "${#selected_nemotron_nano_models[@]}" -gt 0 ]] && [[ "${#selected_math_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "easy_qa" \
    && [[ "${#selected_nemotron_nano_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_qa_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "easy_math_code" \
    && [[ "${#selected_nemotron_nano_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_math_code_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if group_is_selected "code_exec" \
    && [[ "${#selected_gemma_models[@]}" -gt 0 ]] && [[ "${#selected_code_exec_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if group_is_selected "mcqa_stem" \
    && [[ "${#selected_gemma_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_stem_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if group_is_selected "mcqa_non_stem" \
    && [[ "${#selected_gemma_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_non_stem_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if group_is_selected "gen" \
    && [[ "${#selected_gemma_models[@]}" -gt 0 ]] && [[ "${#selected_gen_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if group_is_selected "math" \
    && [[ "${#selected_gemma_models[@]}" -gt 0 ]] && [[ "${#selected_math_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if group_is_selected "easy_qa" \
    && [[ "${#selected_gemma_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_qa_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if group_is_selected "easy_math_code" \
    && [[ "${#selected_gemma_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_math_code_tasks[@]}" -gt 0 ]]; then
    planned_variants=$((planned_variants + 1))
fi

if [[ "${planned_variants}" -eq 0 ]]; then
    echo "Error: No launch variants matched the selected groups, suites, and models." >&2
    exit 1
fi

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run enabled. Commands will be printed but not launched."
    echo ""
else
    if [[ "${GEMMA_ONLY}" == "true" ]]; then
        echo "Launching Gemma-only Beaker baseline variants..."
    else
        echo "Launching Beaker baseline variants..."
    fi
    echo ""
fi

if [[ "${#SELECTED_GROUPS[@]}" -gt 0 ]]; then
    echo "Selected groups: ${SELECTED_GROUPS[*]}"
    echo ""
fi

if [[ "${#SELECTED_SUITES[@]}" -gt 0 ]]; then
    echo "Selected suites: ${SELECTED_SUITES[*]}"
    echo ""
fi

if [[ "${#SELECTED_MODELS[@]}" -gt 0 ]]; then
    echo "Selected models: ${SELECTED_MODELS[*]}"
    echo ""
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "code_exec" \
    && [[ "${#selected_standard_baseline_models[@]}" -gt 0 ]] && [[ "${#selected_code_exec_tasks[@]}" -gt 0 ]]; then
    run_variant "Baseline models: code execution suites" "${baseline_exec_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "code_exec" \
    && [[ "${#selected_qwen3_models[@]}" -gt 0 ]] && [[ "${#selected_code_exec_tasks[@]}" -gt 0 ]]; then
    run_variant "Qwen3: code execution suites (thinking disabled)" "${qwen3_exec_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "mcqa_stem" \
    && [[ "${#selected_standard_baseline_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_stem_tasks[@]}" -gt 0 ]]; then
    run_variant "Baseline models: MCQA stem suites" "${baseline_mcqa_stem_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "mcqa_stem" \
    && [[ "${#selected_qwen3_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_stem_tasks[@]}" -gt 0 ]]; then
    run_variant "Qwen3: MCQA stem suites (thinking disabled)" "${qwen3_mcqa_stem_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "mcqa_non_stem" \
    && [[ "${#selected_standard_baseline_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_non_stem_tasks[@]}" -gt 0 ]]; then
    run_variant "Baseline models: MCQA non-stem suites" "${baseline_mcqa_non_stem_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "mcqa_non_stem" \
    && [[ "${#selected_qwen3_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_non_stem_tasks[@]}" -gt 0 ]]; then
    run_variant "Qwen3: MCQA non-stem suites (thinking disabled)" "${qwen3_mcqa_non_stem_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "gen" \
    && [[ "${#selected_standard_baseline_models[@]}" -gt 0 ]] && [[ "${#selected_gen_tasks[@]}" -gt 0 ]]; then
    run_variant "Baseline models: generation suites" "${baseline_gen_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "gen" \
    && [[ "${#selected_qwen3_models[@]}" -gt 0 ]] && [[ "${#selected_gen_tasks[@]}" -gt 0 ]]; then
    run_variant "Qwen3: generation suites (thinking disabled)" "${qwen3_gen_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "math" \
    && [[ "${#selected_standard_baseline_models[@]}" -gt 0 ]] && [[ "${#selected_math_tasks[@]}" -gt 0 ]]; then
    run_variant "Baseline models: math suites" "${baseline_math_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "math" \
    && [[ "${#selected_qwen3_models[@]}" -gt 0 ]] && [[ "${#selected_math_tasks[@]}" -gt 0 ]]; then
    run_variant "Qwen3: math suites (thinking disabled)" "${qwen3_math_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "easy_qa" \
    && [[ "${#selected_standard_baseline_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_qa_tasks[@]}" -gt 0 ]]; then
    run_variant "Baseline models: easy QA suites" "${baseline_non_exec_easy_qa_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "easy_qa" \
    && [[ "${#selected_qwen3_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_qa_tasks[@]}" -gt 0 ]]; then
    run_variant "Qwen3: easy QA suites (thinking disabled)" "${qwen3_non_exec_easy_qa_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "easy_math_code" \
    && [[ "${#selected_standard_baseline_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_math_code_tasks[@]}" -gt 0 ]]; then
    run_variant "Baseline models: easy math + easy code suites" "${baseline_non_exec_easy_math_code_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "easy_math_code" \
    && [[ "${#selected_qwen3_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_math_code_tasks[@]}" -gt 0 ]]; then
    run_variant "Qwen3: easy math + easy code suites (thinking disabled)" "${qwen3_non_exec_easy_math_code_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "code_exec" \
    && [[ "${#selected_nemotron_nano_models[@]}" -gt 0 ]] && [[ "${#selected_code_exec_tasks[@]}" -gt 0 ]]; then
    run_variant "Nemotron Nano: code execution suites (prefix caching disabled)" "${nemotron_exec_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "mcqa_stem" \
    && [[ "${#selected_nemotron_nano_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_stem_tasks[@]}" -gt 0 ]]; then
    run_variant "Nemotron Nano: MCQA stem suites (prefix caching disabled)" "${nemotron_mcqa_stem_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "mcqa_non_stem" \
    && [[ "${#selected_nemotron_nano_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_non_stem_tasks[@]}" -gt 0 ]]; then
    run_variant "Nemotron Nano: MCQA non-stem suites (prefix caching disabled)" "${nemotron_mcqa_non_stem_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "gen" \
    && [[ "${#selected_nemotron_nano_models[@]}" -gt 0 ]] && [[ "${#selected_gen_tasks[@]}" -gt 0 ]]; then
    run_variant "Nemotron Nano: generation suites (prefix caching disabled)" "${nemotron_gen_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "math" \
    && [[ "${#selected_nemotron_nano_models[@]}" -gt 0 ]] && [[ "${#selected_math_tasks[@]}" -gt 0 ]]; then
    run_variant "Nemotron Nano: math suites (prefix caching disabled)" "${nemotron_math_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "easy_qa" \
    && [[ "${#selected_nemotron_nano_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_qa_tasks[@]}" -gt 0 ]]; then
    run_variant "Nemotron Nano: easy QA suites (prefix caching disabled)" "${nemotron_non_exec_easy_qa_cmd[@]}"
fi

if [[ "${GEMMA_ONLY}" == "false" ]] && group_is_selected "easy_math_code" \
    && [[ "${#selected_nemotron_nano_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_math_code_tasks[@]}" -gt 0 ]]; then
    run_variant "Nemotron Nano: easy math + easy code suites (prefix caching disabled)" "${nemotron_non_exec_easy_math_code_cmd[@]}"
fi

if group_is_selected "code_exec" \
    && [[ "${#selected_gemma_models[@]}" -gt 0 ]] && [[ "${#selected_code_exec_tasks[@]}" -gt 0 ]]; then
    run_variant "Gemma: code execution suites" "${gemma_exec_cmd[@]}"
fi

if group_is_selected "mcqa_stem" \
    && [[ "${#selected_gemma_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_stem_tasks[@]}" -gt 0 ]]; then
    run_variant "Gemma: MCQA stem suites" "${gemma_mcqa_stem_cmd[@]}"
fi

if group_is_selected "mcqa_non_stem" \
    && [[ "${#selected_gemma_models[@]}" -gt 0 ]] && [[ "${#selected_mcqa_non_stem_tasks[@]}" -gt 0 ]]; then
    run_variant "Gemma: MCQA non-stem suites" "${gemma_mcqa_non_stem_cmd[@]}"
fi

if group_is_selected "gen" \
    && [[ "${#selected_gemma_models[@]}" -gt 0 ]] && [[ "${#selected_gen_tasks[@]}" -gt 0 ]]; then
    run_variant "Gemma: generation suites" "${gemma_gen_cmd[@]}"
fi

if group_is_selected "math" \
    && [[ "${#selected_gemma_models[@]}" -gt 0 ]] && [[ "${#selected_math_tasks[@]}" -gt 0 ]]; then
    run_variant "Gemma: math suites" "${gemma_math_cmd[@]}"
fi

if group_is_selected "easy_qa" \
    && [[ "${#selected_gemma_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_qa_tasks[@]}" -gt 0 ]]; then
    run_variant "Gemma: easy QA suites" "${gemma_non_exec_easy_qa_cmd[@]}"
fi

if group_is_selected "easy_math_code" \
    && [[ "${#selected_gemma_models[@]}" -gt 0 ]] && [[ "${#selected_non_exec_easy_math_code_tasks[@]}" -gt 0 ]]; then
    run_variant "Gemma: easy math + easy code suites" "${gemma_non_exec_easy_math_code_cmd[@]}"
fi
