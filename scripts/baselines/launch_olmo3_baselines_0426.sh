#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
DRY_RUN=false

GROUP="${GROUP:-olmo-eval-olmo3-baselines-0426}"
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
Usage: ${SCRIPT_NAME} [--dry-run]

Launches Beaker variants for the OLMo 3 baseline sweep across:
  1. Code execution tasks
  2. MCQA tasks
  3. Gen + math tasks
  4. Easy QA tasks
  5. Easy math + easy code tasks

Each task group is launched once for the baseline model bundle and once for Gemma.

Options:
  --dry-run   Print the commands without launching them
  --help      Show this help

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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
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

code_exec_tasks=(
    "olmobase:code"
    "olmobase:code_fim"
)

non_exec_mcqa_tasks=(
    "olmobase:mcqa_stem"
    "olmobase:mcqa_non_stem"
)

non_exec_gen_math_tasks=(
    "olmobase:gen"
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
    "marin-community/marin-8b-base"
    "swiss-ai/Apertus-8B-2509"
    "almanach/Gaperon-1125-8B"
    "allenai/OLMo-2-1124-7B"
    "Qwen/Qwen3-8B"
    "mistralai/Mistral-Nemo-Base-2407"
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    "Qwen/Qwen2.5-7B"
    "ibm-granite/granite-3.3-8b-base"
    "XiaomiMiMo/MiMo-7B-Base"
)

gemma_models=(
    "google/gemma-2-9b"
)

baseline_launch_args=(
    "-o" "provider.num_instances=${PROVIDER_NUM_INSTANCES}"
    "-o" "provider.trust_remote_code=true"
)

gemma_launch_args=(
    "-o" "provider.num_instances=${PROVIDER_NUM_INSTANCES}"
    "-o" "provider.kwargs.attention_backend=TRITON_ATTN"
)

exec_only_args=(
    "-o" 'sandboxes.0={"mode":"modal","instances":64, "min_instances": 56, "registry_auth":{"provider":"gcp"}}'
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
    "-y"
)

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

build_baseline_exec_cmd() {
    start_command "${EXEC_HARNESS}"
    append_args "${baseline_launch_args[@]}"
    append_args "${exec_only_args[@]}"
    append_models "${baseline_models[@]}"
    append_tasks "${code_exec_tasks[@]}"
    append_args "${common_tail_args[@]}"
    save_current_cmd baseline_exec_cmd
}

build_baseline_non_exec_mcqa_cmd() {
    start_command "${NON_EXEC_HARNESS}"
    append_args "${baseline_launch_args[@]}"
    append_models "${baseline_models[@]}"
    append_tasks "${non_exec_mcqa_tasks[@]}"
    append_args "${common_tail_args[@]}"
    save_current_cmd baseline_non_exec_mcqa_cmd
}

build_baseline_non_exec_gen_math_cmd() {
    start_command "${NON_EXEC_HARNESS}"
    append_args "${baseline_launch_args[@]}"
    append_models "${baseline_models[@]}"
    append_tasks "${non_exec_gen_math_tasks[@]}"
    append_args "${common_tail_args[@]}"
    save_current_cmd baseline_non_exec_gen_math_cmd
}

build_baseline_non_exec_easy_qa_cmd() {
    start_command "${NON_EXEC_HARNESS}"
    append_args "${baseline_launch_args[@]}"
    append_models "${baseline_models[@]}"
    append_tasks "${non_exec_easy_qa_tasks[@]}"
    append_args "${common_tail_args[@]}"
    save_current_cmd baseline_non_exec_easy_qa_cmd
}

build_baseline_non_exec_easy_math_code_cmd() {
    start_command "${NON_EXEC_HARNESS}"
    append_args "${baseline_launch_args[@]}"
    append_models "${baseline_models[@]}"
    append_tasks "${non_exec_easy_math_code_tasks[@]}"
    append_args "${common_tail_args[@]}"
    save_current_cmd baseline_non_exec_easy_math_code_cmd
}

build_gemma_exec_cmd() {
    start_command "${EXEC_HARNESS}"
    append_args "${gemma_launch_args[@]}"
    append_args "${exec_only_args[@]}"
    append_models "${gemma_models[@]}"
    append_tasks "${code_exec_tasks[@]}"
    append_args "${common_tail_args[@]}"
    save_current_cmd gemma_exec_cmd
}

build_gemma_non_exec_mcqa_cmd() {
    start_command "${NON_EXEC_HARNESS}"
    append_args "${gemma_launch_args[@]}"
    append_models "${gemma_models[@]}"
    append_tasks "${non_exec_mcqa_tasks[@]}"
    append_args "${common_tail_args[@]}"
    save_current_cmd gemma_non_exec_mcqa_cmd
}

build_gemma_non_exec_gen_math_cmd() {
    start_command "${NON_EXEC_HARNESS}"
    append_args "${gemma_launch_args[@]}"
    append_models "${gemma_models[@]}"
    append_tasks "${non_exec_gen_math_tasks[@]}"
    append_args "${common_tail_args[@]}"
    save_current_cmd gemma_non_exec_gen_math_cmd
}

build_gemma_non_exec_easy_qa_cmd() {
    start_command "${NON_EXEC_HARNESS}"
    append_args "${gemma_launch_args[@]}"
    append_models "${gemma_models[@]}"
    append_tasks "${non_exec_easy_qa_tasks[@]}"
    append_args "${common_tail_args[@]}"
    save_current_cmd gemma_non_exec_easy_qa_cmd
}

build_gemma_non_exec_easy_math_code_cmd() {
    start_command "${NON_EXEC_HARNESS}"
    append_args "${gemma_launch_args[@]}"
    append_models "${gemma_models[@]}"
    append_tasks "${non_exec_easy_math_code_tasks[@]}"
    append_args "${common_tail_args[@]}"
    save_current_cmd gemma_non_exec_easy_math_code_cmd
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
declare -a baseline_non_exec_mcqa_cmd
declare -a baseline_non_exec_gen_math_cmd
declare -a baseline_non_exec_easy_qa_cmd
declare -a baseline_non_exec_easy_math_code_cmd
declare -a gemma_exec_cmd
declare -a gemma_non_exec_mcqa_cmd
declare -a gemma_non_exec_gen_math_cmd
declare -a gemma_non_exec_easy_qa_cmd
declare -a gemma_non_exec_easy_math_code_cmd

build_baseline_exec_cmd
build_baseline_non_exec_mcqa_cmd
build_baseline_non_exec_gen_math_cmd
build_baseline_non_exec_easy_qa_cmd
build_baseline_non_exec_easy_math_code_cmd
build_gemma_exec_cmd
build_gemma_non_exec_mcqa_cmd
build_gemma_non_exec_gen_math_cmd
build_gemma_non_exec_easy_qa_cmd
build_gemma_non_exec_easy_math_code_cmd

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run enabled. Commands will be printed but not launched."
    echo ""
else
    echo "Launching Beaker baseline variants..."
    echo ""
fi

run_variant "Baseline models: code execution suites" "${baseline_exec_cmd[@]}"
run_variant "Baseline models: MCQA suites" "${baseline_non_exec_mcqa_cmd[@]}"
run_variant "Baseline models: gen + math suites" "${baseline_non_exec_gen_math_cmd[@]}"
run_variant "Baseline models: easy QA suites" "${baseline_non_exec_easy_qa_cmd[@]}"
run_variant "Baseline models: easy math + easy code suites" "${baseline_non_exec_easy_math_code_cmd[@]}"
run_variant "Gemma: code execution suites" "${gemma_exec_cmd[@]}"
run_variant "Gemma: MCQA suites" "${gemma_non_exec_mcqa_cmd[@]}"
run_variant "Gemma: gen + math suites" "${gemma_non_exec_gen_math_cmd[@]}"
run_variant "Gemma: easy QA suites" "${gemma_non_exec_easy_qa_cmd[@]}"
run_variant "Gemma: easy math + easy code suites" "${gemma_non_exec_easy_math_code_cmd[@]}"
