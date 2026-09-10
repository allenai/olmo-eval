#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEDGER="${LEDGER:-${ROOT_DIR}/notes/beaker_jobs.jsonl}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"
PRIORITY="${PRIORITY:-urgent}"
GROUP="${GROUP:-adaptive-experts-qwen3-30b-20260710}"
RUN_TAG="${RUN_TAG:-}"
SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-8192}"
GPTOSS_MAX_NUM_SEQS="${GPTOSS_MAX_NUM_SEQS:-1}"
GPTOSS_GPU_MEMORY_UTILIZATION="${GPTOSS_GPU_MEMORY_UTILIZATION:-0.95}"
GPTOSS_SMOKE_LIMIT="${GPTOSS_SMOKE_LIMIT:-1}"
GPTOSS_STARTUP_TIMEOUT="${GPTOSS_STARTUP_TIMEOUT:-1800}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
QWEN_EXPERT_WEIGHT_MODE="${QWEN_EXPERT_WEIGHT_MODE:-normal}"
QWEN_EXPERT_WEIGHT_SEED="${QWEN_EXPERT_WEIGHT_SEED:-0}"
QWEN_EXPERT_WEIGHT_EXPONENT="${QWEN_EXPERT_WEIGHT_EXPONENT:-}"
QWEN_EXPERT_RANK_PROFILE="${QWEN_EXPERT_RANK_PROFILE:-}"
QWEN_EXPERT_MASS_THRESHOLD="${QWEN_EXPERT_MASS_THRESHOLD:-}"
QWEN_EXPERT_MIN_K="${QWEN_EXPERT_MIN_K:-1}"
QWEN_EXPERT_MAX_K="${QWEN_EXPERT_MAX_K:-8}"
QWEN_EXPERT_KEEP_K="${QWEN_EXPERT_KEEP_K:-}"
QWEN_EXPERT_REFERENCE_K="${QWEN_EXPERT_REFERENCE_K:-8}"
QWEN_EXPERT_RENORMALIZE="${QWEN_EXPERT_RENORMALIZE:-true}"
QWEN_RECORD_REALIZED_K="${QWEN_RECORD_REALIZED_K:-false}"
QWEN_REALIZED_K_INTERVAL_SECONDS="${QWEN_REALIZED_K_INTERVAL_SECONDS:-30}"
QWEN_ENFORCE_EAGER="${QWEN_ENFORCE_EAGER:-false}"
QWEN35_ATTENTION_BACKEND="${QWEN35_ATTENTION_BACKEND:-}"
QWEN35_GDN_PREFILL_BACKEND="${QWEN35_GDN_PREFILL_BACKEND:-}"
ARTIFACT_REGISTRY_SECRET="${ARTIFACT_REGISTRY_SECRET:-}"
BEAKER_IMAGE="${BEAKER_IMAGE:-}"
BEAKER_BUDGET="${BEAKER_BUDGET:-}"
BEAKER_TASK_TIMEOUT="${BEAKER_TASK_TIMEOUT:-24h}"
ALLOCATED="${ALLOCATED:-false}"
ALLOCATED_MIN_RUNTIME="${ALLOCATED_MIN_RUNTIME:-1h}"
VLLM_PROVIDER_PACKAGE="${VLLM_PROVIDER_PACKAGE:-}"
QWEN36_AIME_MAX_MODEL_LEN="${QWEN36_AIME_MAX_MODEL_LEN:-40960}"
QWEN36_AIME_MAX_NUM_SEQS="${QWEN36_AIME_MAX_NUM_SEQS:-16}"
QWEN36_AIME_REPLICAS="${QWEN36_AIME_REPLICAS:-8}"
QWEN36_AIME_WORKER_CHUNK_SIZE="${QWEN36_AIME_WORKER_CHUNK_SIZE:-4}"
# Qwen recommends an 81,920-token thinking budget. Leave additional room for
# the AIME prompt and chat-template framing inside the model context.
QWEN35_397B_AIME_MAX_MODEL_LEN="${QWEN35_397B_AIME_MAX_MODEL_LEN:-90112}"
QWEN35_397B_AIME_MAX_TOKENS="${QWEN35_397B_AIME_MAX_TOKENS:-81920}"
QWEN35_397B_AIME_MAX_NUM_SEQS="${QWEN35_397B_AIME_MAX_NUM_SEQS:-8}"
QWEN35_397B_AIME_STARTUP_TIMEOUT="${QWEN35_397B_AIME_STARTUP_TIMEOUT:-3600}"
QWEN35_397B_AIME_GPU_MEMORY_UTILIZATION="${QWEN35_397B_AIME_GPU_MEMORY_UTILIZATION:-0.9}"
# Opt-in only: preserve the proven vLLM 0.19.1 recipe above while allowing a
# conservative long-context startup diagnostic in an isolated vLLM 0.26 venv.
QWEN35_397B_CONSERVATIVE_STARTUP="${QWEN35_397B_CONSERVATIVE_STARTUP:-false}"
QWEN35_397B_MAX_NUM_BATCHED_TOKENS="${QWEN35_397B_MAX_NUM_BATCHED_TOKENS:-2096}"
QWEN35_397B_VLLM_PROVIDER_PACKAGE="${QWEN35_397B_VLLM_PROVIDER_PACKAGE:-}"
GLM52_AIME_MAX_TOKENS="${GLM52_AIME_MAX_TOKENS:-163840}"
# Leave 8K tokens for the AIME prompt/chat framing above the recommended
# 163,840-token reasoning-task output budget.
GLM52_AIME_MAX_MODEL_LEN="${GLM52_AIME_MAX_MODEL_LEN:-172032}"
GLM52_AIME_MAX_NUM_SEQS="${GLM52_AIME_MAX_NUM_SEQS:-8}"
GLM52_AIME_TENSOR_PARALLEL_SIZE="${GLM52_AIME_TENSOR_PARALLEL_SIZE:-8}"
GLM52_AIME_STARTUP_TIMEOUT="${GLM52_AIME_STARTUP_TIMEOUT:-7200}"
GLM52_REFERENCE_K="${GLM52_REFERENCE_K:-8}"
GLM52_NUM_EXPERTS="${GLM52_NUM_EXPERTS:-256}"
# v0.23.0 can parse GLM-5.2 but predates upstream's shared-indexer fix
# (vllm-project/vllm#45895), so the BF16 checkpoint fails after loading.
GLM52_VLLM_PROVIDER_PACKAGE="${GLM52_VLLM_PROVIDER_PACKAGE:-vllm==0.26.0}"
DEEPSEEK_V4_AIME_MAX_TOKENS="${DEEPSEEK_V4_AIME_MAX_TOKENS:-393216}"
# DeepSeek recommends a 384K output budget for high/max effort. Interpret K as
# 1024 tokens and leave another 8K for the prompt/chat framing.
DEEPSEEK_V4_AIME_MAX_MODEL_LEN="${DEEPSEEK_V4_AIME_MAX_MODEL_LEN:-401408}"
DEEPSEEK_V4_AIME_MAX_NUM_SEQS="${DEEPSEEK_V4_AIME_MAX_NUM_SEQS:-8}"
DEEPSEEK_V4_AIME_STARTUP_TIMEOUT="${DEEPSEEK_V4_AIME_STARTUP_TIMEOUT:-7200}"
DEEPSEEK_V4_VLLM_PROVIDER_PACKAGE="${DEEPSEEK_V4_VLLM_PROVIDER_PACKAGE:-vllm==0.26.0}"
CUDA13_COMPILER_TOOLKIT_PACKAGE="${CUDA13_COMPILER_TOOLKIT_PACKAGE:-cuda-toolkit[cudart,nvcc,nvrtc,nvvm,cccl,crt]==13.0.2}"
MOE_ROUTING_NORMALIZATION="${MOE_ROUTING_NORMALIZATION:-native}"
GPTOSS_REFERENCE_SCALE="${GPTOSS_REFERENCE_SCALE:-false}"
GPTOSS_REFERENCE_K="${GPTOSS_REFERENCE_K:-4}"
NEMOTRON_NORM_TOPK_PROB="${NEMOTRON_NORM_TOPK_PROB:-true}"
DOLCI_QWEN_MODEL="${DOLCI_QWEN_MODEL:-/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints/qwen3-30b-a3b-dolci-think-olmo-core-sft-100k-20260710-174355-hf}"
DOLCI_QWEN_MAX_TOKENS="${DOLCI_QWEN_MAX_TOKENS:-30000}"
DOLCI_QWEN_FULL_MAX_TOKENS="${DOLCI_QWEN_FULL_MAX_TOKENS:-32768}"
DOLCI_QWEN_FULL_MAX_MODEL_LEN="${DOLCI_QWEN_FULL_MAX_MODEL_LEN:-40960}"
GPTOSS_TIKTOKEN_ENCODINGS_BASE="${GPTOSS_TIKTOKEN_ENCODINGS_BASE:-$(dirname "${ROOT_DIR}")/.runtime_assets/tiktoken_encodings}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-$(dirname "${ROOT_DIR}")/.run_snapshots/olmo-eval}"
LAUNCH_PROJECT="${LAUNCH_PROJECT:-${ROOT_DIR}}"
SOURCE_SNAPSHOT_OVERRIDE="${SOURCE_SNAPSHOT_OVERRIDE:-}"

DRY_RUN=false
PHASE=""
SOURCE_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
SOURCE_HASH="working-tree"
SOURCE_SNAPSHOT="${ROOT_DIR}"
LAUNCH_CWD="${ROOT_DIR}"
CLEAN_WORKTREE=""
declare -a EXPERT_COUNTS=()

usage() {
    echo "Usage: $0 --phase PHASE [--experts K ...] [--dry-run]"
    echo "Phases: base-smoke, base-math, hybrid-smoke, hybrid-capability-smoke, hybrid-capability-policy-smoke, hybrid-capability-policy, hybrid-capability-policy-ifbench32k, hybrid-expanded-policy-smoke, hybrid-expanded-policy, hybrid-capability-backfill, hybrid-weight-smoke, hybrid-weight-pilot, qwen35-expanded-policy-smoke, qwen35-expanded-policy, qwen36-expanded-policy-smoke, qwen36-expanded-policy, qwen36-aime2026-smoke, qwen36-aime2026, qwen35-397b-aime2026-smoke, qwen35-397b-aime2026, deepseek-v4-flash-aime2026-smoke, deepseek-v4-flash-aime2026, glm52-aime2026-smoke, glm52-aime2026, nemotron-routing-smoke, nemotron-routing-pilot, glm-air-gpu-smoke, gptoss-smoke, gptoss-reference-smoke, gptoss-reference-expanded, gptoss-capability-backfill, base, hybrid-pilot, dolci-qwen-pilot, dolci-qwen-full-smoke, dolci-qwen-full, gptoss-pilot, glm-pilot, hybrid-aime, gptoss-aime, glm-aime"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --experts)
            EXPERT_COUNTS+=("$2")
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

if [[ -z "${PHASE}" ]]; then
    usage >&2
    exit 2
fi

if [[ -n "${RUN_TAG}" && ! "${RUN_TAG}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "RUN_TAG may only contain letters, digits, dots, underscores, and dashes" >&2
    exit 2
fi
if [[ ! "${SMOKE_MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SMOKE_MAX_TOKENS must be a positive integer" >&2
    exit 2
fi
if [[ ! "${ALLOCATED}" =~ ^(true|false)$ ]]; then
    echo "ALLOCATED must be true or false" >&2
    exit 2
fi
if [[ ! "${BEAKER_TASK_TIMEOUT}" =~ ^[1-9][0-9]*(s|m|h|d)$ ]]; then
    echo "BEAKER_TASK_TIMEOUT must be a positive duration such as 24h or 3d" >&2
    exit 2
fi
if [[ ! "${QWEN36_AIME_WORKER_CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "QWEN36_AIME_WORKER_CHUNK_SIZE must be a positive integer" >&2
    exit 2
fi
if [[ "${ALLOCATED}" == "true" && ! "${ALLOCATED_MIN_RUNTIME}" =~ ^[1-9][0-9]*(s|m|h)$ ]]; then
    echo "ALLOCATED_MIN_RUNTIME must be a positive duration such as 30m or 1h" >&2
    exit 2
fi
if [[ -n "${TENSOR_PARALLEL_SIZE}" && ! "${TENSOR_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TENSOR_PARALLEL_SIZE must be a positive integer" >&2
    exit 2
fi
if [[ ! "${QWEN_EXPERT_WEIGHT_MODE}" =~ ^(normal|shuffle|uniform|temperature|rank_profile|adaptive_mass|truncate)$ ]]; then
    echo "QWEN_EXPERT_WEIGHT_MODE is not supported: ${QWEN_EXPERT_WEIGHT_MODE}" >&2
    exit 2
fi
if [[ ! "${QWEN_EXPERT_WEIGHT_SEED}" =~ ^[0-9]+$ ]]; then
    echo "QWEN_EXPERT_WEIGHT_SEED must be a non-negative integer" >&2
    exit 2
fi
case "${QWEN_EXPERT_WEIGHT_MODE}" in
    temperature)
        if [[ ! "${QWEN_EXPERT_WEIGHT_EXPONENT}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
            echo "temperature mode requires a non-negative QWEN_EXPERT_WEIGHT_EXPONENT" >&2
            exit 2
        fi
        ;;
    rank_profile)
        IFS=',' read -r -a rank_profile_values <<< "${QWEN_EXPERT_RANK_PROFILE}"
        if [[ "${#rank_profile_values[@]}" -ne 8 ]]; then
            echo "rank_profile mode requires eight comma-separated QWEN_EXPERT_RANK_PROFILE values" >&2
            exit 2
        fi
        rank_profile_sum=0
        for value in "${rank_profile_values[@]}"; do
            if [[ ! "${value}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
                echo "QWEN_EXPERT_RANK_PROFILE values must be non-negative numbers" >&2
                exit 2
            fi
            rank_profile_sum="$(awk -v total="${rank_profile_sum}" -v item="${value}" 'BEGIN { print total + item }')"
        done
        if ! awk -v total="${rank_profile_sum}" 'BEGIN { exit !(total > 0) }'; then
            echo "QWEN_EXPERT_RANK_PROFILE must have a positive sum" >&2
            exit 2
        fi
        ;;
    adaptive_mass)
        if [[ ! "${QWEN_EXPERT_MASS_THRESHOLD}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] \
            || ! awk -v threshold="${QWEN_EXPERT_MASS_THRESHOLD}" 'BEGIN { exit !(threshold > 0 && threshold <= 1) }'; then
            echo "adaptive_mass mode requires QWEN_EXPERT_MASS_THRESHOLD in (0, 1]" >&2
            exit 2
        fi
        if [[ ! "${QWEN_EXPERT_MIN_K}" =~ ^[1-8]$ || ! "${QWEN_EXPERT_MAX_K}" =~ ^[1-8]$ ]] \
            || ((QWEN_EXPERT_MIN_K > QWEN_EXPERT_MAX_K)); then
            echo "adaptive_mass bounds must satisfy 1 <= QWEN_EXPERT_MIN_K <= QWEN_EXPERT_MAX_K <= 8" >&2
            exit 2
        fi
        ;;
    truncate)
        if [[ ! "${QWEN_EXPERT_KEEP_K}" =~ ^[1-9][0-9]*$ ]]; then
            echo "truncate mode requires a positive QWEN_EXPERT_KEEP_K" >&2
            exit 2
        fi
        if [[ ! "${QWEN_EXPERT_REFERENCE_K}" =~ ^[1-9][0-9]*$ ]]; then
            echo "truncate mode requires a positive QWEN_EXPERT_REFERENCE_K" >&2
            exit 2
        fi
        ;;
esac
if [[ ! "${QWEN_EXPERT_RENORMALIZE}" =~ ^(true|false)$ ]]; then
    echo "QWEN_EXPERT_RENORMALIZE must be true or false" >&2
    exit 2
fi
if [[ ! "${QWEN_RECORD_REALIZED_K}" =~ ^(true|false)$ ]]; then
    echo "QWEN_RECORD_REALIZED_K must be true or false" >&2
    exit 2
fi
if [[ ! "${QWEN_ENFORCE_EAGER}" =~ ^(true|false)$ ]]; then
    echo "QWEN_ENFORCE_EAGER must be true or false" >&2
    exit 2
fi
if [[ ! "${QWEN_REALIZED_K_INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "QWEN_REALIZED_K_INTERVAL_SECONDS must be a positive integer" >&2
    exit 2
fi
if [[ ! "${GPTOSS_REFERENCE_SCALE}" =~ ^(true|false)$ ]]; then
    echo "GPTOSS_REFERENCE_SCALE must be true or false" >&2
    exit 2
fi
if [[ ! "${GPTOSS_REFERENCE_K}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPTOSS_REFERENCE_K must be a positive integer" >&2
    exit 2
fi
if [[ ! "${NEMOTRON_NORM_TOPK_PROB}" =~ ^(true|false)$ ]]; then
    echo "NEMOTRON_NORM_TOPK_PROB must be true or false" >&2
    exit 2
fi
if [[ ! "${MOE_ROUTING_NORMALIZATION}" =~ ^(native|normalized|unnormalized|native-reference-truncated)$ ]]; then
    echo "MOE_ROUTING_NORMALIZATION must be native, normalized, unnormalized, or native-reference-truncated" >&2
    exit 2
fi
for setting in \
    GLM52_AIME_MAX_TOKENS \
    GLM52_AIME_MAX_MODEL_LEN \
    GLM52_AIME_STARTUP_TIMEOUT \
    GLM52_REFERENCE_K \
    GLM52_NUM_EXPERTS \
    DEEPSEEK_V4_AIME_MAX_TOKENS \
    DEEPSEEK_V4_AIME_MAX_MODEL_LEN \
    DEEPSEEK_V4_AIME_STARTUP_TIMEOUT; do
    if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${setting} must be a positive integer" >&2
        exit 2
    fi
done
if [[ "${MOE_ROUTING_NORMALIZATION}" == "native-reference-truncated" \
    && "${PHASE}" != "glm52-aime2026-smoke" \
    && "${PHASE}" != "glm52-aime2026" ]]; then
    echo "native-reference-truncated routing is currently supported only for GLM-5.2 AIME phases" >&2
    exit 2
fi
if ((GLM52_AIME_MAX_MODEL_LEN <= GLM52_AIME_MAX_TOKENS)); then
    echo "GLM52_AIME_MAX_MODEL_LEN must exceed GLM52_AIME_MAX_TOKENS" >&2
    exit 2
fi
if ((DEEPSEEK_V4_AIME_MAX_MODEL_LEN <= DEEPSEEK_V4_AIME_MAX_TOKENS)); then
    echo "DEEPSEEK_V4_AIME_MAX_MODEL_LEN must exceed DEEPSEEK_V4_AIME_MAX_TOKENS" >&2
    exit 2
fi
if [[ ! "${GPTOSS_MAX_NUM_SEQS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPTOSS_MAX_NUM_SEQS must be a positive integer" >&2
    exit 2
fi
if [[ ! "${GPTOSS_GPU_MEMORY_UTILIZATION}" =~ ^0\.[0-9]+$ ]]; then
    echo "GPTOSS_GPU_MEMORY_UTILIZATION must be between 0 and 1" >&2
    exit 2
fi
if [[ ! "${GPTOSS_SMOKE_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPTOSS_SMOKE_LIMIT must be a positive integer" >&2
    exit 2
fi
if [[ ! "${GPTOSS_STARTUP_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPTOSS_STARTUP_TIMEOUT must be a positive integer" >&2
    exit 2
fi
if [[ ! "${DOLCI_QWEN_MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]] || ((DOLCI_QWEN_MAX_TOKENS > 30000)); then
    echo "DOLCI_QWEN_MAX_TOKENS must be between 1 and 30000" >&2
    exit 2
fi
if [[ ! "${DOLCI_QWEN_FULL_MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DOLCI_QWEN_FULL_MAX_TOKENS must be a positive integer" >&2
    exit 2
fi
if [[ ! "${DOLCI_QWEN_FULL_MAX_MODEL_LEN}" =~ ^[1-9][0-9]*$ ]] \
    || ((DOLCI_QWEN_FULL_MAX_MODEL_LEN <= DOLCI_QWEN_FULL_MAX_TOKENS)); then
    echo "DOLCI_QWEN_FULL_MAX_MODEL_LEN must exceed DOLCI_QWEN_FULL_MAX_TOKENS" >&2
    exit 2
fi
NAME_SUFFIX="${RUN_TAG:+-${RUN_TAG}}"

case "${PHASE}" in
    base-smoke|hybrid-smoke)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(1 8 16)
        fi
        ;;
    hybrid-capability-smoke)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(8)
        fi
        ;;
    hybrid-capability-policy-smoke|hybrid-capability-policy|hybrid-capability-policy-ifbench32k|hybrid-expanded-policy-smoke|hybrid-expanded-policy)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(8)
        fi
        ;;
    hybrid-capability-backfill)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(3 5 6 7 9 10 11 12 13 14 15 16)
        fi
        ;;
    hybrid-weight-smoke|hybrid-weight-pilot)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(8)
        fi
        ;;
    nemotron-routing-smoke|nemotron-routing-pilot)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(11 22)
        fi
        if [[ -z "${TENSOR_PARALLEL_SIZE}" ]]; then
            TENSOR_PARALLEL_SIZE=2
        elif [[ "${TENSOR_PARALLEL_SIZE}" -ne 2 ]]; then
            echo "${PHASE} currently uses the smoke-tested TP=2 layout" >&2
            exit 2
        fi
        ;;
    glm-air-gpu-smoke)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(8)
        fi
        if [[ -z "${TENSOR_PARALLEL_SIZE}" ]]; then
            echo "glm-air-gpu-smoke requires TENSOR_PARALLEL_SIZE" >&2
            exit 2
        fi
        ;;
    gptoss-pilot|gptoss-aime|gptoss-reference-expanded|gptoss-capability-backfill)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(1 2 3 4 5 6 7 8)
        fi
        if [[ -z "${TENSOR_PARALLEL_SIZE}" ]]; then
            echo "${PHASE} requires TENSOR_PARALLEL_SIZE" >&2
            exit 2
        fi
        ;;
    gptoss-smoke|gptoss-reference-smoke)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(1 16)
        fi
        if [[ -z "${TENSOR_PARALLEL_SIZE}" ]]; then
            echo "gptoss-smoke requires TENSOR_PARALLEL_SIZE" >&2
            exit 2
        fi
        ;;
    glm-pilot)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(4 8)
        fi
        if [[ -z "${TENSOR_PARALLEL_SIZE}" ]]; then
            echo "glm-pilot requires TENSOR_PARALLEL_SIZE" >&2
            exit 2
        fi
        ;;
    glm-aime)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(2 4 8)
        fi
        if [[ -z "${TENSOR_PARALLEL_SIZE}" ]]; then
            echo "glm-aime requires TENSOR_PARALLEL_SIZE" >&2
            exit 2
        fi
        ;;
    qwen35-expanded-policy-smoke|qwen35-expanded-policy|qwen36-expanded-policy-smoke|qwen36-expanded-policy|qwen36-aime2026-smoke|qwen36-aime2026|glm52-aime2026-smoke|glm52-aime2026)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(8)
        fi
        ;;
    qwen35-397b-aime2026-smoke|qwen35-397b-aime2026)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(10)
        fi
        ;;
    deepseek-v4-flash-aime2026-smoke|deepseek-v4-flash-aime2026)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(6)
        fi
        ;;
    hybrid-aime)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 32)
        fi
        ;;
    dolci-qwen-pilot|dolci-qwen-full-smoke|dolci-qwen-full)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            if [[ "${PHASE}" == "dolci-qwen-pilot" ]]; then
                EXPERT_COUNTS=(2 4 6 8)
            else
                EXPERT_COUNTS=(8)
            fi
        fi
        if [[ ! -f "${DOLCI_QWEN_MODEL}/config.json" || ! -f "${DOLCI_QWEN_MODEL}/chat_template.jinja" ]]; then
            echo "DOLCI_QWEN_MODEL is not a complete local HF checkpoint: ${DOLCI_QWEN_MODEL}" >&2
            exit 2
        fi
        ;;
    base|base-math|hybrid-pilot)
        if [[ "${#EXPERT_COUNTS[@]}" -eq 0 ]]; then
            EXPERT_COUNTS=(1 2 4 8 16)
        fi
        ;;
    *)
        echo "Unknown phase: ${PHASE}" >&2
        usage >&2
        exit 2
        ;;
esac
if [[ "${PHASE}" == "gptoss-pilot" \
    || "${PHASE}" == "gptoss-smoke" \
    || "${PHASE}" == "gptoss-aime" \
    || "${PHASE}" == "gptoss-reference-smoke" \
    || "${PHASE}" == "gptoss-reference-expanded" \
    || "${PHASE}" == "gptoss-capability-backfill" ]]; then
    for encoding in o200k_base.tiktoken cl100k_base.tiktoken; do
        if [[ ! -s "${GPTOSS_TIKTOKEN_ENCODINGS_BASE}/${encoding}" ]]; then
            echo "Missing GPT-OSS tokenizer asset: ${GPTOSS_TIKTOKEN_ENCODINGS_BASE}/${encoding}" >&2
            exit 2
        fi
    done
fi


for k in "${EXPERT_COUNTS[@]}"; do
    case "${k}" in
        1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|32|64|128|256) ;;
        *)
            echo "Unsupported expert count: ${k}" >&2
            exit 2
            ;;
    esac
done
if [[ "${MOE_ROUTING_NORMALIZATION}" == "native-reference-truncated" ]]; then
    if [[ "${#EXPERT_COUNTS[@]}" -ne 1 ]]; then
        echo "native-reference-truncated routing requires exactly one kernel expert count" >&2
        exit 2
    fi
    if ((EXPERT_COUNTS[0] >= GLM52_REFERENCE_K)); then
        echo "native-reference-truncated routing requires kernel K < GLM52_REFERENCE_K" >&2
        exit 2
    fi
fi
if [[ "${PHASE}" == "hybrid-weight-smoke" \
    || "${PHASE}" == "hybrid-weight-pilot" \
    || "${PHASE}" == "hybrid-capability-policy-smoke" \
    || "${PHASE}" == "hybrid-capability-policy" \
    || "${PHASE}" == "hybrid-capability-policy-ifbench32k" \
    || "${PHASE}" == "hybrid-expanded-policy-smoke" \
    || "${PHASE}" == "hybrid-expanded-policy" \
    || "${PHASE}" == "qwen35-expanded-policy-smoke" \
    || "${PHASE}" == "qwen35-expanded-policy" \
    || "${PHASE}" == "qwen36-expanded-policy-smoke" \
    || "${PHASE}" == "qwen36-expanded-policy" \
    || "${PHASE}" == "dolci-qwen-full-smoke" \
    || "${PHASE}" == "dolci-qwen-full" ]]; then
    if [[ "${#EXPERT_COUNTS[@]}" -ne 1 ]]; then
        echo "${PHASE} requires exactly one router expert count" >&2
        exit 2
    fi
    if [[ "${QWEN_EXPERT_WEIGHT_MODE}" == "truncate" ]]; then
        max_router_k=16
        if [[ "${PHASE}" == "qwen35-expanded-policy-smoke" \
            || "${PHASE}" == "qwen35-expanded-policy" \
            || "${PHASE}" == "qwen36-expanded-policy-smoke" \
            || "${PHASE}" == "qwen36-expanded-policy" ]]; then
            max_router_k=256
        fi
        if ((EXPERT_COUNTS[0] < 8 || EXPERT_COUNTS[0] > max_router_k)); then
            echo "truncate mode supports router K from 8 through ${max_router_k} for ${PHASE}" >&2
            exit 2
        fi
        if ((QWEN_EXPERT_KEEP_K > EXPERT_COUNTS[0])); then
            echo "QWEN_EXPERT_KEEP_K cannot exceed router K=${EXPERT_COUNTS[0]}" >&2
            exit 2
        fi
        if ((QWEN_EXPERT_REFERENCE_K > EXPERT_COUNTS[0])); then
            echo "QWEN_EXPERT_REFERENCE_K cannot exceed router K=${EXPERT_COUNTS[0]}" >&2
            exit 2
        fi
    elif [[ "${QWEN_EXPERT_WEIGHT_MODE}" != "normal" && "${EXPERT_COUNTS[0]}" -ne 8 ]]; then
        echo "${QWEN_EXPERT_WEIGHT_MODE} mode is defined only for checkpoint-default K=8" >&2
        exit 2
    fi
fi

cleanup() {
    if [[ -n "${CLEAN_WORKTREE}" ]]; then
        git -C "${ROOT_DIR}" worktree remove --force "${CLEAN_WORKTREE}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

prepare_launch_context() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        return
    fi

    # Gantry requires a clean Git checkout, but the job itself should run the exact
    # local evaluation code. Copy that code to a content-addressed Weka snapshot and
    # submit from a disposable clean worktree at the same base commit.
    if [[ -n "${SOURCE_SNAPSHOT_OVERRIDE}" ]]; then
        SOURCE_SNAPSHOT="${SOURCE_SNAPSHOT_OVERRIDE}"
        SOURCE_HASH="$(basename "${SOURCE_SNAPSHOT}")"
        if [[ ! -d "${SOURCE_SNAPSHOT}/src" ]]; then
            echo "SOURCE_SNAPSHOT_OVERRIDE lacks src/: ${SOURCE_SNAPSHOT}" >&2
            exit 2
        fi
    else
        SOURCE_HASH="$({
            printf '%s\n' "${SOURCE_COMMIT}"
            find "${ROOT_DIR}/src" -type f \
                ! -path '*/__pycache__/*' \
                ! -name '*.pyc' \
                -print0 \
                | sort -z \
                | xargs -0 sha256sum
        } | sha256sum | cut -d' ' -f1)"
        SOURCE_SNAPSHOT="${SNAPSHOT_ROOT}/${SOURCE_HASH}"

        if [[ ! -d "${SOURCE_SNAPSHOT}/src" ]]; then
            mkdir -p "${SNAPSHOT_ROOT}"
            local snapshot_tmp
            snapshot_tmp="$(mktemp -d "${SNAPSHOT_ROOT}/.${SOURCE_HASH}.XXXXXX")"
            mkdir -p "${snapshot_tmp}/src"
            cp -a "${ROOT_DIR}/src/." "${snapshot_tmp}/src/"
            mv "${snapshot_tmp}" "${SOURCE_SNAPSHOT}"
        fi
    fi

    CLEAN_WORKTREE="$(mktemp -d "${TMPDIR:-/tmp}/olmo-eval-gantry.XXXXXX")"
    rmdir "${CLEAN_WORKTREE}"
    git -C "${ROOT_DIR}" worktree add --detach --quiet "${CLEAN_WORKTREE}" "${SOURCE_COMMIT}"
    LAUNCH_CWD="${CLEAN_WORKTREE}"

    echo "Source snapshot: ${SOURCE_SNAPSHOT}"
    echo "Submission checkout: ${LAUNCH_CWD} (${SOURCE_COMMIT})"
}

prepare_launch_context

command_string() {
    local rendered=""
    local token
    for token in "$@"; do
        printf -v token '%q' "${token}"
        rendered+="${rendered:+ }${token}"
    done
    printf '%s' "${rendered}"
}

launch_one() {
    local phase=$1
    local expert_count=$2
    local model=$3
    local harness=$4
    local replicas=$5
    local name=$6
    shift 6
    local -a task_args=("$@")
    local -a task_specs=()
    local hf_overrides="{\"num_experts_per_tok\":${expert_count}}"
    if [[ "${phase}" == "qwen35-expanded-policy-smoke" \
        || "${phase}" == "qwen35-expanded-policy" \
        || "${phase}" == "qwen36-expanded-policy-smoke" \
        || "${phase}" == "qwen36-expanded-policy" \
        || "${phase}" == "qwen36-aime2026-smoke" \
        || "${phase}" == "qwen36-aime2026" \
        || "${phase}" == "qwen35-397b-aime2026-smoke" \
        || "${phase}" == "qwen35-397b-aime2026" ]]; then
        hf_overrides="{\"text_config\":{\"num_experts_per_tok\":${expert_count}}}"
    fi
    if [[ "${phase}" == "nemotron-routing-smoke" || "${phase}" == "nemotron-routing-pilot" ]]; then
        hf_overrides="{\"num_experts_per_tok\":${expert_count},\"norm_topk_prob\":${NEMOTRON_NORM_TOPK_PROB}}"
    fi
    if [[ ("${phase}" == "deepseek-v4-flash-aime2026-smoke" \
            || "${phase}" == "deepseek-v4-flash-aime2026" \
            || "${phase}" == "glm52-aime2026-smoke" \
            || "${phase}" == "glm52-aime2026") \
        && "${MOE_ROUTING_NORMALIZATION}" != "native" ]]; then
        local norm_topk_prob=true
        if [[ "${MOE_ROUTING_NORMALIZATION}" == "unnormalized" ]]; then
            norm_topk_prob=false
        fi
        hf_overrides="{\"num_experts_per_tok\":${expert_count},\"norm_topk_prob\":${norm_topk_prob}}"
    fi
    local i
    for ((i = 0; i < ${#task_args[@]}; i++)); do
        if [[ "${task_args[$i]}" == "-t" ]]; then
            task_specs+=("${task_args[$((i + 1))]}")
        fi
    done

    local -a cmd=(
        uv run --project "${LAUNCH_PROJECT}" --frozen --no-group vllm olmo-eval beaker launch
        -H "${harness}"
        -o "provider.num_instances=${replicas}"
        -o "provider.kwargs.hf_overrides=${hf_overrides}"
        -o "metrics.collect_gpu=true"
    )

    if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
        cmd+=(-o "provider.kwargs.tensor_parallel_size=${TENSOR_PARALLEL_SIZE}")
    fi
    if [[ "${QWEN_ENFORCE_EAGER}" == "true" ]]; then
        cmd+=(-o "provider.kwargs.enforce_eager=true")
    fi
    if [[ "${phase}" == "dolci-qwen-pilot" ]]; then
        cmd+=(
            # This checkpoint has a native 32,768-token context and no RoPE
            # scaling. Keep the server at the native limit; the task cap below
            # leaves room for the prompt inside that total context window.
            -o "provider.max_model_len=32768"
            -o "provider.kwargs.gpu_memory_utilization=0.9"
            -o "provider.kwargs.max_num_seqs=16"
            -o "provider.kwargs.reasoning_parser=olmo3"
            -o "provider.kwargs.startup_timeout=900"
        )
    fi
    if [[ "${phase}" == "dolci-qwen-full-smoke" || "${phase}" == "dolci-qwen-full" ]]; then
        cmd+=(
            -o "provider.max_model_len=${DOLCI_QWEN_FULL_MAX_MODEL_LEN}"
            -o "provider.kwargs.gpu_memory_utilization=0.9"
            -o "provider.kwargs.max_num_seqs=16"
            -o "provider.kwargs.reasoning_parser=olmo3"
            -o "provider.kwargs.startup_timeout=900"
        )
    fi
    if [[ "${phase}" == "qwen35-expanded-policy-smoke" || "${phase}" == "qwen35-expanded-policy" ]]; then
        cmd+=(
            -o "provider.max_model_len=40960"
            -o "provider.kwargs.gpu_memory_utilization=0.9"
            -o "provider.kwargs.language_model_only=true"
            -o "provider.kwargs.max_num_seqs=16"
            -o "provider.kwargs.reasoning_parser=qwen3"
            -o "provider.kwargs.startup_timeout=1800"
            -o 'provider.kwargs.chat_template_kwargs={"enable_thinking":true}'
        )
        if [[ -n "${QWEN35_ATTENTION_BACKEND}" ]]; then
            cmd+=(-o "provider.kwargs.attention_backend=${QWEN35_ATTENTION_BACKEND}")
        fi
        if [[ -n "${QWEN35_GDN_PREFILL_BACKEND}" ]]; then
            cmd+=(-o "provider.kwargs.gdn_prefill_backend=${QWEN35_GDN_PREFILL_BACKEND}")
        fi
    fi
    if [[ "${phase}" == "qwen36-expanded-policy-smoke" || "${phase}" == "qwen36-expanded-policy" ]]; then
        cmd+=(
            -o "provider.max_model_len=110000"
            -o "provider.kwargs.gpu_memory_utilization=0.95"
            -o "provider.kwargs.language_model_only=true"
            -o "provider.kwargs.max_num_seqs=4"
            -o "provider.kwargs.reasoning_parser=qwen3"
            -o "provider.kwargs.startup_timeout=1800"
            -o 'provider.kwargs.chat_template_kwargs={"enable_thinking":true}'
            -o "provider.kwargs.attention_backend=TRITON_ATTN"
            -o "provider.kwargs.gdn_prefill_backend=triton"
        )
    fi
    if [[ "${phase}" == "qwen36-aime2026-smoke" || "${phase}" == "qwen36-aime2026" ]]; then
        cmd+=(
            -e "OLMO_EVAL_RUNTIME_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130"
            # AIME 2026 has 30 problems. Capping queue claims at four prevents
            # one of eight data-parallel workers from draining most of the
            # shared queue before its peers can claim work.
            -o "batching.chunk_size=${QWEN36_AIME_WORKER_CHUNK_SIZE}"
            -o "provider.max_model_len=${QWEN36_AIME_MAX_MODEL_LEN}"
            -o "provider.kwargs.gpu_memory_utilization=0.95"
            -o "provider.kwargs.language_model_only=true"
            -o "provider.kwargs.max_num_seqs=${QWEN36_AIME_MAX_NUM_SEQS}"
            -o "provider.kwargs.reasoning_parser=qwen3"
            -o "provider.kwargs.startup_timeout=1800"
            -o 'provider.kwargs.chat_template_kwargs={"enable_thinking":true}'
            -o "provider.kwargs.attention_backend=TRITON_ATTN"
            -o "provider.kwargs.gdn_prefill_backend=triton"
        )
    fi
    if [[ "${phase}" == "qwen35-397b-aime2026-smoke" || "${phase}" == "qwen35-397b-aime2026" ]]; then
        cmd+=(
            -e "OLMO_EVAL_RUNTIME_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130"
            -o "provider.trust_remote_code=true"
            -o "provider.max_model_len=${QWEN35_397B_AIME_MAX_MODEL_LEN}"
            -o "provider.kwargs.tensor_parallel_size=8"
            -o "provider.kwargs.gpu_memory_utilization=${QWEN35_397B_AIME_GPU_MEMORY_UTILIZATION}"
            -o "provider.kwargs.language_model_only=true"
            -o "provider.kwargs.max_num_seqs=${QWEN35_397B_AIME_MAX_NUM_SEQS}"
            -o "provider.kwargs.reasoning_parser=qwen3"
            -o "provider.kwargs.startup_timeout=${QWEN35_397B_AIME_STARTUP_TIMEOUT}"
            -o 'provider.kwargs.chat_template_kwargs={"enable_thinking":true}'
            -o "provider.kwargs.attention_backend=TRITON_ATTN"
            -o "provider.kwargs.gdn_prefill_backend=triton"
        )
        if [[ "${QWEN35_397B_CONSERVATIVE_STARTUP}" == "true" ]]; then
            cmd+=(
                -e "OLMO_EVAL_RUNTIME_TORCH_VERSION=2.11.0"
                -e "OLMO_EVAL_VLLM_CUDA_TOOLKIT_PACKAGE=${CUDA13_COMPILER_TOOLKIT_PACKAGE}"
                -e "VLLM_DEEP_GEMM_WARMUP=skip"
                -e "VLLM_USE_DEEP_GEMM=0"
                -o "provider.kwargs.enable_prefix_caching=false"
                -o "provider.kwargs.max_num_batched_tokens=${QWEN35_397B_MAX_NUM_BATCHED_TOKENS}"
                -o "provider.kwargs.enforce_eager=true"
                -o "provider.kwargs.log_dir=/results/vllm-server"
            )
        fi
    fi
    if [[ "${phase}" == "deepseek-v4-flash-aime2026-smoke" || "${phase}" == "deepseek-v4-flash-aime2026" ]]; then
        cmd+=(
            -e "OLMO_EVAL_RUNTIME_TORCH_VERSION=2.11.0"
            -e "OLMO_EVAL_RUNTIME_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130"
            -e "OLMO_EVAL_VLLM_CUDA_TOOLKIT_PACKAGE=${CUDA13_COMPILER_TOOLKIT_PACKAGE}"
            -o "provider.trust_remote_code=true"
            -o "provider.max_model_len=${DEEPSEEK_V4_AIME_MAX_MODEL_LEN}"
            -o "provider.kwargs.tensor_parallel_size=8"
            -o "provider.kwargs.gpu_memory_utilization=0.9"
            -o "provider.kwargs.max_num_seqs=${DEEPSEEK_V4_AIME_MAX_NUM_SEQS}"
            -o "provider.kwargs.enable_expert_parallel=true"
            -o "provider.kwargs.kv_cache_dtype=fp8"
            -o "provider.kwargs.block_size=256"
            -o "provider.kwargs.tokenizer_mode=deepseek_v4"
            -o "provider.kwargs.reasoning_parser=deepseek_v4"
            -o 'provider.kwargs.attention_config={"use_fp4_indexer_cache":true}'
            -o "provider.kwargs.startup_timeout=${DEEPSEEK_V4_AIME_STARTUP_TIMEOUT}"
            -o 'provider.kwargs.chat_template_kwargs={"thinking":true,"reasoning_effort":"max"}'
        )
    fi
    if [[ "${phase}" == "glm52-aime2026-smoke" || "${phase}" == "glm52-aime2026" ]]; then
        cmd+=(
            -e "OLMO_EVAL_RUNTIME_TORCH_VERSION=2.11.0"
            -e "OLMO_EVAL_RUNTIME_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130"
            -e "OLMO_EVAL_VLLM_CUDA_TOOLKIT_PACKAGE=${CUDA13_COMPILER_TOOLKIT_PACKAGE}"
            -o "provider.max_model_len=${GLM52_AIME_MAX_MODEL_LEN}"
            -o "provider.kwargs.tensor_parallel_size=${GLM52_AIME_TENSOR_PARALLEL_SIZE}"
            -o "provider.kwargs.gpu_memory_utilization=0.9"
            -o "provider.kwargs.max_num_seqs=${GLM52_AIME_MAX_NUM_SEQS}"
            -o "provider.kwargs.reasoning_parser=glm45"
            -o "provider.kwargs.startup_timeout=${GLM52_AIME_STARTUP_TIMEOUT}"
            -o "provider.kwargs.log_dir=/results/vllm-server"
            -o 'provider.kwargs.chat_template_kwargs={"enable_thinking":true}'
        )
        if [[ "${MOE_ROUTING_NORMALIZATION}" == "native-reference-truncated" ]]; then
            cmd+=(
                -e "OLMO_EVAL_VLLM_GLM_REFERENCE_TRUNCATION=1"
                -e "OLMO_EVAL_VLLM_GLM_KEEP_K=${expert_count}"
                -e "OLMO_EVAL_VLLM_GLM_REFERENCE_K=${GLM52_REFERENCE_K}"
                -e "OLMO_EVAL_VLLM_GLM_NUM_EXPERTS=${GLM52_NUM_EXPERTS}"
            )
        fi
    fi
    if [[ "${phase}" == "glm-air-gpu-smoke" || "${phase}" == "glm-pilot" || "${phase}" == "glm-aime" ]]; then
        cmd+=(
            -o "provider.max_model_len=40960"
            -o "provider.kwargs.gpu_memory_utilization=0.95"
            -o "provider.kwargs.max_num_seqs=16"
            -o "provider.kwargs.reasoning_parser=glm45"
            -o "provider.kwargs.startup_timeout=900"
        )
    fi
    if [[ "${phase}" == "nemotron-routing-smoke" || "${phase}" == "nemotron-routing-pilot" ]]; then
        cmd+=(
            -o "provider.kind=vllm_server"
            -o "provider.trust_remote_code=true"
            -o "provider.max_model_len=40960"
            -o "provider.kwargs.enable_expert_parallel=true"
            -o "provider.kwargs.gpu_memory_utilization=0.9"
            -o "provider.kwargs.max_num_seqs=16"
            -o "provider.kwargs.enable_prefix_caching=false"
            -o "provider.kwargs.kv_cache_dtype=fp8"
            -o "provider.kwargs.mamba_ssm_cache_dtype=float32"
            -o "provider.kwargs.max_cudagraph_capture_size=128"
            -o "provider.kwargs.reasoning_parser=nemotron_v3"
            -o "provider.kwargs.startup_timeout=900"
            -o 'provider.kwargs.chat_template_kwargs={"enable_thinking":true}'
            -e "VLLM_USE_FLASHINFER_MOE_FP8=0"
        )
    fi
    if [[ "${phase}" == "gptoss-pilot" \
        || "${phase}" == "gptoss-aime" \
        || "${phase}" == "gptoss-reference-expanded" \
        || "${phase}" == "gptoss-capability-backfill" ]]; then
        cmd+=(
            -o "provider.max_model_len=40960"
            -o "provider.kwargs.gpu_memory_utilization=${GPTOSS_GPU_MEMORY_UTILIZATION}"
            -o "provider.kwargs.max_num_seqs=${GPTOSS_MAX_NUM_SEQS}"
            -o "provider.kwargs.max_num_batched_tokens=1024"
            -o "provider.kwargs.enable_prefix_caching=false"
            -o "provider.kwargs.max_cudagraph_capture_size=2048"
            -o "provider.kwargs.startup_timeout=${GPTOSS_STARTUP_TIMEOUT}"
        )
    fi
    if [[ "${phase}" == "gptoss-smoke" || "${phase}" == "gptoss-reference-smoke" ]]; then
        cmd+=(
            -o "provider.max_model_len=40960"
            -o "provider.kwargs.gpu_memory_utilization=${GPTOSS_GPU_MEMORY_UTILIZATION}"
            -o "provider.kwargs.max_num_seqs=${GPTOSS_MAX_NUM_SEQS}"
            -o "provider.kwargs.max_num_batched_tokens=1024"
            -o "provider.kwargs.enable_prefix_caching=false"
            -o "provider.kwargs.max_cudagraph_capture_size=2048"
            -o "provider.kwargs.startup_timeout=${GPTOSS_STARTUP_TIMEOUT}"
        )
    fi
    if [[ "${phase}" == "gptoss-pilot" \
        || "${phase}" == "gptoss-smoke" \
        || "${phase}" == "gptoss-aime" \
        || "${phase}" == "gptoss-reference-smoke" \
        || "${phase}" == "gptoss-reference-expanded" \
        || "${phase}" == "gptoss-capability-backfill" ]]; then
        cmd+=(
            -e "PYTORCH_ALLOC_CONF=expandable_segments:True"
            -e "TIKTOKEN_ENCODINGS_BASE=${GPTOSS_TIKTOKEN_ENCODINGS_BASE}"
            -e "OLMO_EVAL_VLLM_GPTOSS_NONPOW2_TOPK_FALLBACK=1"
        )
        if ((TENSOR_PARALLEL_SIZE > 1)); then
            # vLLM's custom all-reduce kernel can fail with CUDA invalid-argument
            # errors on the Jupiter H100 topology. CUDA-graph capture also exhausts
            # memory with the compatibility routing path. NCCL plus eager mode is
            # slower but avoids both startup failures.
            cmd+=(
                -o "provider.kwargs.disable_custom_all_reduce=true"
                -o "provider.kwargs.enforce_eager=true"
            )
        fi
        if [[ "${GPTOSS_REFERENCE_SCALE}" == "true" ]]; then
            cmd+=(
                -e "OLMO_EVAL_VLLM_GPTOSS_REFERENCE_SCALE=1"
                -e "OLMO_EVAL_VLLM_GPTOSS_REFERENCE_K=${GPTOSS_REFERENCE_K}"
            )
        fi
    fi
    if [[ "${phase}" == "hybrid-weight-smoke" \
        || "${phase}" == "hybrid-weight-pilot" \
        || "${phase}" == "hybrid-capability-policy-smoke" \
        || "${phase}" == "hybrid-capability-policy" \
        || "${phase}" == "hybrid-capability-policy-ifbench32k" \
        || "${phase}" == "hybrid-expanded-policy-smoke" \
        || "${phase}" == "hybrid-expanded-policy" \
        || "${phase}" == "qwen35-expanded-policy-smoke" \
        || "${phase}" == "qwen35-expanded-policy" \
        || "${phase}" == "qwen36-expanded-policy-smoke" \
        || "${phase}" == "qwen36-expanded-policy" \
        || "${phase}" == "qwen36-aime2026-smoke" \
        || "${phase}" == "qwen36-aime2026" \
        || "${phase}" == "dolci-qwen-full-smoke" \
        || "${phase}" == "dolci-qwen-full" ]]; then
        if [[ "${phase}" == "qwen35-expanded-policy-smoke" \
            || "${phase}" == "qwen35-expanded-policy" \
            || "${phase}" == "qwen36-expanded-policy-smoke" \
            || "${phase}" == "qwen36-expanded-policy" \
            || "${phase}" == "qwen36-aime2026-smoke" \
            || "${phase}" == "qwen36-aime2026" ]]; then
            # Qwen3.5's DeltaNet kernels JIT through the isolated vLLM venv's
            # ninja executable even for native normalized routing.
            cmd+=(-e "OLMO_EVAL_VLLM_INSTALL_SITECUSTOMIZE=1")
        fi
        if [[ "${QWEN_EXPERT_WEIGHT_MODE}" != "normal" ]]; then
            cmd+=(
                -e "OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_MODE=${QWEN_EXPERT_WEIGHT_MODE}"
                -e "OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_SEED=${QWEN_EXPERT_WEIGHT_SEED}"
                -e "OLMO_EVAL_VLLM_QWEN_EXPERT_ROUTER_K=${expert_count}"
                -e "VLLM_USE_FLASHINFER_MOE_FP16=0"
            )
            if [[ "${phase}" == "qwen35-expanded-policy-smoke" \
                || "${phase}" == "qwen35-expanded-policy" \
                || "${phase}" == "qwen36-expanded-policy-smoke" \
                || "${phase}" == "qwen36-expanded-policy" ]]; then
                cmd+=(-e "OLMO_EVAL_VLLM_QWEN_NUM_EXPERTS=256")
            fi
            case "${QWEN_EXPERT_WEIGHT_MODE}" in
                temperature)
                    cmd+=(-e "OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_EXPONENT=${QWEN_EXPERT_WEIGHT_EXPONENT}")
                    ;;
                rank_profile)
                    cmd+=(-e "OLMO_EVAL_VLLM_QWEN_EXPERT_RANK_PROFILE=${QWEN_EXPERT_RANK_PROFILE}")
                    ;;
                adaptive_mass)
                    cmd+=(
                        -e "OLMO_EVAL_VLLM_QWEN_EXPERT_MASS_THRESHOLD=${QWEN_EXPERT_MASS_THRESHOLD}"
                        -e "OLMO_EVAL_VLLM_QWEN_EXPERT_MIN_K=${QWEN_EXPERT_MIN_K}"
                        -e "OLMO_EVAL_VLLM_QWEN_EXPERT_MAX_K=${QWEN_EXPERT_MAX_K}"
                        -e "OLMO_EVAL_VLLM_QWEN_EXPERT_RENORMALIZE=${QWEN_EXPERT_RENORMALIZE}"
                    )
                    if [[ "${QWEN_RECORD_REALIZED_K}" == "true" ]]; then
                        cmd+=(
                            -e "OLMO_EVAL_VLLM_QWEN_RECORD_REALIZED_K=1"
                            -e "OLMO_EVAL_VLLM_QWEN_REALIZED_K_INTERVAL_SECONDS=${QWEN_REALIZED_K_INTERVAL_SECONDS}"
                        )
                    fi
                    ;;
                truncate)
                    cmd+=(
                        -e "OLMO_EVAL_VLLM_QWEN_EXPERT_KEEP_K=${QWEN_EXPERT_KEEP_K}"
                        -e "OLMO_EVAL_VLLM_QWEN_EXPERT_REFERENCE_K=${QWEN_EXPERT_REFERENCE_K}"
                        -e "OLMO_EVAL_VLLM_QWEN_EXPERT_RENORMALIZE=${QWEN_EXPERT_RENORMALIZE}"
                    )
                    ;;
            esac
        fi
    fi
    if [[ "${phase}" == "qwen35-397b-aime2026-smoke" \
        || "${phase}" == "qwen35-397b-aime2026" \
        || "${phase}" == "deepseek-v4-flash-aime2026-smoke" \
        || "${phase}" == "deepseek-v4-flash-aime2026" \
        || "${phase}" == "glm52-aime2026-smoke" \
        || "${phase}" == "glm52-aime2026" ]]; then
        # CUDA 13 provider wheels place nvcc under site-packages/nvidia/cu13.
        # The startup hook exposes that toolkit to FlashInfer/PyTorch JITs.
        cmd+=(-e "OLMO_EVAL_VLLM_INSTALL_SITECUSTOMIZE=1")
    fi
    cmd+=(-m "${model}")

    if [[ -n "${BEAKER_IMAGE}" ]]; then
        cmd+=(-I "${BEAKER_IMAGE}")
    fi
    if [[ -n "${BEAKER_BUDGET}" ]]; then
        cmd+=(-B "${BEAKER_BUDGET}")
    fi
    if [[ "${ALLOCATED}" == "true" ]]; then
        cmd+=(--min-runtime "${ALLOCATED_MIN_RUNTIME}")
    fi
    cmd+=(--timeout "${BEAKER_TASK_TIMEOUT}")
    local provider_package="${VLLM_PROVIDER_PACKAGE}"
    if [[ -z "${provider_package}" \
        && -n "${QWEN35_397B_VLLM_PROVIDER_PACKAGE}" \
        && ("${phase}" == "qwen35-397b-aime2026-smoke" \
            || "${phase}" == "qwen35-397b-aime2026") ]]; then
        provider_package="${QWEN35_397B_VLLM_PROVIDER_PACKAGE}"
    fi
    if [[ -z "${provider_package}" \
        && ("${phase}" == "glm52-aime2026-smoke" || "${phase}" == "glm52-aime2026") ]]; then
        provider_package="${GLM52_VLLM_PROVIDER_PACKAGE}"
    fi
    if [[ -z "${provider_package}" \
        && ("${phase}" == "deepseek-v4-flash-aime2026-smoke" \
            || "${phase}" == "deepseek-v4-flash-aime2026") ]]; then
        provider_package="${DEEPSEEK_V4_VLLM_PROVIDER_PACKAGE}"
    fi
    if [[ -n "${provider_package}" ]]; then
        cmd+=(-o "provider.package=${provider_package}")
    fi

    cmd+=("${task_args[@]}")
    cmd+=(
        --name "${name}"
        --cluster "${CLUSTER}"
        --workspace "${WORKSPACE}"
        --priority "${PRIORITY}"
        --group "${GROUP}"
        -e "PYTHONPATH=${SOURCE_SNAPSHOT}/src"
        --secret-env jacobm_HF_TOKEN:HF_TOKEN
    )
    if [[ -n "${ARTIFACT_REGISTRY_SECRET}" ]]; then
        cmd+=(--secret-env "${ARTIFACT_REGISTRY_SECRET}:GOOGLE_APPLICATION_CREDENTIALS")
    fi
    cmd+=(--no-store --no-follow -y)
    if [[ "${DRY_RUN}" == "true" ]]; then
        cmd+=(--dry-run)
    fi

    echo
    echo "Launching ${name}"
    echo "$(command_string "${cmd[@]}")"

    local output_file
    output_file="$(mktemp)"
    trap 'rm -f "${output_file}"' RETURN

    set +e
    (cd "${LAUNCH_CWD}" && "${cmd[@]}") 2>&1 | tee "${output_file}"
    local launch_status=${PIPESTATUS[0]}
    set -e

    if [[ "${DRY_RUN}" == "false" ]]; then
        local -a experiment_ids=()
        mapfile -t experiment_ids < <(
            grep -Eo 'https://beaker.org/ex/[A-Za-z0-9]+' "${output_file}" \
                | sed 's#https://beaker.org/ex/##' \
                | sort -u
        )

        if [[ "${#experiment_ids[@]}" -eq 0 && "${launch_status}" -eq 0 ]]; then
            echo "Launch succeeded but no Beaker experiment ID was captured; refusing to continue." >&2
            return 1
        fi

        local experiment_id
        local -a record_tasks=()
        for task in "${task_specs[@]}"; do
            record_tasks+=(--task "${task}")
        done
        for experiment_id in "${experiment_ids[@]}"; do
            python "${ROOT_DIR}/scripts/adaptive_experts/record_beaker_job.py" \
                --ledger "${LEDGER}" \
                --phase "${phase}" \
                --run-tag "${RUN_TAG}" \
                --expert-count "${expert_count}" \
                --model "${model}" \
                "${record_tasks[@]}" \
                --cluster "${CLUSTER}" \
                --workspace "${WORKSPACE}" \
                --priority "${PRIORITY}" \
                --group "${GROUP}" \
                --source-commit "${SOURCE_COMMIT}" \
                --source-hash "${SOURCE_HASH}" \
                --source-snapshot "${SOURCE_SNAPSHOT}" \
                --result-storage beaker_dataset \
                --experiment-id "${experiment_id}" \
                --command "$(command_string "${cmd[@]}")"
            echo "Recorded https://beaker.org/ex/${experiment_id} in ${LEDGER}"
        done
    fi

    rm -f "${output_file}"
    trap - RETURN
    return "${launch_status}"
}

for k in "${EXPERT_COUNTS[@]}"; do
    case "${PHASE}" in
        base-smoke)
            launch_one \
                base-smoke "${k}" qwen3-30b-a3b-base default 1 \
                "adaptive-qwen3-base-k${k}-smoke${NAME_SUFFIX}" \
                -t arc_challenge:mc:olmo3base -o limit=16 \
                -t minerva_math_algebra:olmo3base -o limit=16
            ;;
        hybrid-smoke)
            launch_one \
                hybrid-smoke "${k}" qwen3-30b-a3b default 1 \
                "adaptive-qwen3-hybrid-k${k}-smoke${NAME_SUFFIX}" \
                -t math500:chat -o limit=16 -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}" \
                -t gpqa_diamond:qwen3_thinking -o limit=16 \
                -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}"
            ;;
        hybrid-capability-smoke)
            launch_one \
                hybrid-capability-smoke "${k}" qwen3-30b-a3b codex_python 1 \
                "adaptive-qwen3-hybrid-k${k}-capability-smoke${NAME_SUFFIX}" \
                -t humaneval_plus:chat:pass_at_1:qwen3_thinking -o limit=8 \
                -t ifeval_ood -o limit=8
            ;;
        hybrid-capability-policy-smoke)
            launch_one \
                hybrid-capability-policy-smoke "${k}" qwen3-30b-a3b codex_python 1 \
                "adaptive-qwen3-hybrid-k${k}-capability-policy-${QWEN_EXPERT_WEIGHT_MODE}-smoke${NAME_SUFFIX}" \
                -t humaneval_plus:chat:pass_at_1:qwen3_thinking -o limit=2 \
                -o "sampling_params.max_tokens=2048" \
                -t ifeval_mt_wildchat_unused_withRewrite -o limit=2 \
                -t ifeval_mt_ood_wildchat_unused_withRewrite -o limit=2 \
                -t ifeval_ood -o limit=2
            ;;
        hybrid-capability-policy)
            launch_one \
                hybrid-capability-policy "${k}" qwen3-30b-a3b codex_python 4 \
                "adaptive-qwen3-hybrid-k${k}-capability-policy-${QWEN_EXPERT_WEIGHT_MODE}${NAME_SUFFIX}" \
                -t ifbench \
                -t humaneval_plus:chat:pass_at_1:qwen3_thinking
            ;;
        hybrid-capability-policy-ifbench32k)
            launch_one \
                hybrid-capability-policy-ifbench32k "${k}" qwen3-30b-a3b default 4 \
                "adaptive-qwen3-hybrid-k${k}-ifbench32k-${QWEN_EXPERT_WEIGHT_MODE}${NAME_SUFFIX}" \
                -t ifeval_ood -o "sampling_params.max_tokens=32768" \
                -t ifeval_mt_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=32768" \
                -t ifeval_mt_ood_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=32768"
            ;;
        hybrid-expanded-policy-smoke)
            launch_one \
                hybrid-expanded-policy-smoke "${k}" qwen3-30b-a3b codex_python 1 \
                "adaptive-qwen3-hybrid-k${k}-expanded-policy-${QWEN_EXPERT_WEIGHT_MODE}-smoke${NAME_SUFFIX}" \
                -t math500:chat -o limit=1 -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}" \
                -t ifeval_mt_wildchat_unused_withRewrite -o limit=1 \
                -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}" \
                -t humaneval:chat:pass_at_1:qwen3_thinking -o limit=1 \
                -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}"
            ;;
        hybrid-expanded-policy)
            launch_one \
                hybrid-expanded-policy "${k}" qwen3-30b-a3b codex_python 4 \
                "adaptive-qwen3-hybrid-k${k}-expanded-policy-${QWEN_EXPERT_WEIGHT_MODE}${NAME_SUFFIX}" \
                -t adaptive_experts:hybrid_pilot \
                -t ifeval_ood -o "sampling_params.max_tokens=32768" \
                -t ifeval_mt_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=32768" \
                -t ifeval_mt_ood_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=32768" \
                -t humaneval:chat:pass_at_1:qwen3_thinking
            ;;
        qwen35-expanded-policy-smoke)
            launch_one \
                qwen35-expanded-policy-smoke "${k}" qwen3.5-35b-a3b codex_python 1 \
                "adaptive-qwen35-35b-a3b-k${k}-${QWEN_EXPERT_WEIGHT_MODE}-smoke${NAME_SUFFIX}" \
                -t math500:chat -o limit=1 -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}" \
                -t gpqa_diamond:qwen3_thinking -o limit=1 \
                -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}" \
                -t ifeval_ood -o limit=1 -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}" \
                -t ifeval_mt_wildchat_unused_withRewrite -o limit=1 \
                -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}" \
                -t ifeval_mt_ood_wildchat_unused_withRewrite -o limit=1 \
                -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}" \
                -t humaneval:chat:pass_at_1:qwen3_thinking -o limit=1 \
                -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}"
            ;;
        qwen35-expanded-policy)
            launch_one \
                qwen35-expanded-policy "${k}" qwen3.5-35b-a3b codex_python 4 \
                "adaptive-qwen35-35b-a3b-k${k}-${QWEN_EXPERT_WEIGHT_MODE}${NAME_SUFFIX}" \
                -t math500:chat -o "sampling_params.max_tokens=32768" \
                -t gpqa_diamond:qwen3_thinking -o "sampling_params.max_tokens=32768" \
                -t ifeval_ood -o "sampling_params.max_tokens=32768" \
                -t ifeval_mt_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=32768" \
                -t ifeval_mt_ood_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=32768" \
                -t humaneval:chat:pass_at_1:qwen3_thinking \
                -o "sampling_params.max_tokens=32768"
            ;;
        qwen36-expanded-policy-smoke)
            launch_one \
                qwen36-expanded-policy-smoke "${k}" Qwen/Qwen3.6-35B-A3B codex_python 1 \
                "adaptive-qwen36-35b-a3b-k${k}-${QWEN_EXPERT_WEIGHT_MODE}-smoke${NAME_SUFFIX}" \
                -t math500:chat -o limit=1 -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}" \
                -t gpqa_diamond:qwen3_thinking -o limit=1 \
                -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}" \
                -t ifeval_ood -o limit=1 -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}"
            ;;
        qwen36-expanded-policy)
            launch_one \
                qwen36-expanded-policy "${k}" Qwen/Qwen3.6-35B-A3B codex_python 4 \
                "adaptive-qwen36-35b-a3b-k${k}-${QWEN_EXPERT_WEIGHT_MODE}${NAME_SUFFIX}" \
                -t math500:chat -o "sampling_params.max_tokens=32768" \
                -o "seed=${QWEN_EXPERT_WEIGHT_SEED}" \
                -t gpqa_diamond:qwen3_thinking -o "sampling_params.max_tokens=32768" \
                -o "seed=${QWEN_EXPERT_WEIGHT_SEED}" \
                -t aime_2025:pass_at_32 -o "sampling_params.max_tokens=32768" \
                -o "seed=${QWEN_EXPERT_WEIGHT_SEED}" \
                -t ifeval_ood -o "sampling_params.max_tokens=32768" \
                -o "seed=${QWEN_EXPERT_WEIGHT_SEED}" \
                -t ifeval_mt_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=32768" \
                -o "seed=${QWEN_EXPERT_WEIGHT_SEED}" \
                -t ifeval_mt_ood_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=32768" \
                -o "seed=${QWEN_EXPERT_WEIGHT_SEED}"
            ;;
        qwen36-aime2026-smoke)
            launch_one \
                qwen36-aime2026-smoke "${k}" Qwen/Qwen3.6-35B-A3B default "${QWEN36_AIME_REPLICAS}" \
                "qwen36-35b-a3b-aime2026-holmes-smoke${NAME_SUFFIX}" \
                -t aime_2026:pass_at_32 -o limit=1 \
                -o "sampling_params.temperature=1.0" \
                -o "sampling_params.top_p=0.95" \
                -o "sampling_params.top_k=20" \
                -o "seed=${QWEN_EXPERT_WEIGHT_SEED}"
            ;;
        qwen36-aime2026)
            launch_one \
                qwen36-aime2026 "${k}" Qwen/Qwen3.6-35B-A3B default "${QWEN36_AIME_REPLICAS}" \
                "qwen36-35b-a3b-aime2026-holmes${NAME_SUFFIX}" \
                -t aime_2026:pass_at_32 \
                -o "sampling_params.temperature=1.0" \
                -o "sampling_params.top_p=0.95" \
                -o "sampling_params.top_k=20" \
                -o "seed=${QWEN_EXPERT_WEIGHT_SEED}"
            ;;
        qwen35-397b-aime2026-smoke)
            launch_one \
                qwen35-397b-aime2026-smoke "${k}" Qwen/Qwen3.5-397B-A17B default 1 \
                "qwen35-397b-a17b-k${k}-aime2026-holmes-smoke${NAME_SUFFIX}" \
                -t aime_2026:pass_at_32 -o limit=1 \
                -o "sampling_params.max_tokens=${QWEN35_397B_AIME_MAX_TOKENS}" \
                -o "sampling_params.temperature=0.6" \
                -o "sampling_params.top_p=0.95" \
                -o "sampling_params.top_k=20" \
                -o "seed=${QWEN_EXPERT_WEIGHT_SEED}"
            ;;
        qwen35-397b-aime2026)
            launch_one \
                qwen35-397b-aime2026 "${k}" Qwen/Qwen3.5-397B-A17B default 1 \
                "qwen35-397b-a17b-k${k}-aime2026-holmes${NAME_SUFFIX}" \
                -t aime_2026:pass_at_32 \
                -o "sampling_params.max_tokens=${QWEN35_397B_AIME_MAX_TOKENS}" \
                -o "sampling_params.temperature=0.6" \
                -o "sampling_params.top_p=0.95" \
                -o "sampling_params.top_k=20" \
                -o "seed=${QWEN_EXPERT_WEIGHT_SEED}"
            ;;
        deepseek-v4-flash-aime2026-smoke)
            launch_one \
                deepseek-v4-flash-aime2026-smoke "${k}" deepseek-ai/DeepSeek-V4-Flash-0731 default 1 \
                "deepseek-v4-flash-0731-k${k}-${MOE_ROUTING_NORMALIZATION}-aime2026-holmes-smoke${NAME_SUFFIX}" \
                -t aime_2026:pass_at_32 -o limit=1 \
                -o "sampling_params.temperature=1.0" \
                -o "sampling_params.top_p=1.0" \
                -o "sampling_params.max_tokens=${DEEPSEEK_V4_AIME_MAX_TOKENS}" \
                -o "seed=${QWEN_EXPERT_WEIGHT_SEED}"
            ;;
        deepseek-v4-flash-aime2026)
            launch_one \
                deepseek-v4-flash-aime2026 "${k}" deepseek-ai/DeepSeek-V4-Flash-0731 default 1 \
                "deepseek-v4-flash-0731-k${k}-${MOE_ROUTING_NORMALIZATION}-aime2026-holmes${NAME_SUFFIX}" \
                -t aime_2026:pass_at_32 \
                -o "sampling_params.temperature=1.0" \
                -o "sampling_params.top_p=1.0" \
                -o "sampling_params.max_tokens=${DEEPSEEK_V4_AIME_MAX_TOKENS}" \
                -o "seed=${QWEN_EXPERT_WEIGHT_SEED}"
            ;;
        glm52-aime2026-smoke)
            launch_one \
                glm52-aime2026-smoke "${k}" zai-org/GLM-5.2 default 1 \
                "glm52-k${k}-${MOE_ROUTING_NORMALIZATION}-aime2026-holmes-smoke${NAME_SUFFIX}" \
                -t aime_2026:pass_at_32 -o limit=1 \
                -o "sampling_params.temperature=1.0" \
                -o "sampling_params.top_p=0.95" \
                -o "sampling_params.max_tokens=${GLM52_AIME_MAX_TOKENS}" \
                -o "seed=${QWEN_EXPERT_WEIGHT_SEED}"
            ;;
        glm52-aime2026)
            launch_one \
                glm52-aime2026 "${k}" zai-org/GLM-5.2 default 1 \
                "glm52-k${k}-${MOE_ROUTING_NORMALIZATION}-aime2026-holmes${NAME_SUFFIX}" \
                -t aime_2026:pass_at_32 \
                -o "sampling_params.temperature=1.0" \
                -o "sampling_params.top_p=0.95" \
                -o "sampling_params.max_tokens=${GLM52_AIME_MAX_TOKENS}" \
                -o "seed=${QWEN_EXPERT_WEIGHT_SEED}"
            ;;
        hybrid-capability-backfill)
            launch_one \
                hybrid-capability-backfill "${k}" qwen3-30b-a3b codex_python 4 \
                "adaptive-qwen3-hybrid-k${k}-capability-backfill${NAME_SUFFIX}" \
                -t ifeval_ood -o "sampling_params.max_tokens=32768" \
                -t ifeval_mt_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=32768" \
                -t ifeval_mt_ood_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=32768" \
                -t humaneval:chat:pass_at_1:qwen3_thinking
            ;;
        hybrid-weight-smoke)
            launch_one \
                hybrid-weight-smoke "${k}" qwen3-30b-a3b default 1 \
                "adaptive-qwen3-hybrid-k${k}-weight-${QWEN_EXPERT_WEIGHT_MODE}-smoke${NAME_SUFFIX}" \
                -t math500:chat -o limit=1 \
                -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}"
            ;;
        hybrid-weight-pilot)
            launch_one \
                hybrid-weight-pilot "${k}" qwen3-30b-a3b default 4 \
                "adaptive-qwen3-hybrid-k${k}-weight-${QWEN_EXPERT_WEIGHT_MODE}-pilot${NAME_SUFFIX}" \
                -t adaptive_experts:hybrid_pilot
            ;;
        nemotron-routing-smoke)
            nemotron_routing_label=normalized
            if [[ "${NEMOTRON_NORM_TOPK_PROB}" == "false" ]]; then
                nemotron_routing_label=rawx5
            fi
            launch_one \
                nemotron-routing-smoke "${k}" nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 default 1 \
                "adaptive-nemotron3-super-k${k}-${nemotron_routing_label}-smoke${NAME_SUFFIX}" \
                -t math500:chat -o limit=1 \
                -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}"
            ;;
        nemotron-routing-pilot)
            nemotron_routing_label=normalized
            if [[ "${NEMOTRON_NORM_TOPK_PROB}" == "false" ]]; then
                nemotron_routing_label=rawx5
            fi
            launch_one \
                nemotron-routing-pilot "${k}" nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 default 4 \
                "adaptive-nemotron3-super-k${k}-${nemotron_routing_label}-pilot${NAME_SUFFIX}" \
                -t adaptive_experts:hybrid_pilot
            ;;
        glm-air-gpu-smoke)
            launch_one \
                glm-air-gpu-smoke "${k}" zai-org/GLM-4.5-Air default 1 \
                "adaptive-glm45-air-k${k}-tp${TENSOR_PARALLEL_SIZE}-gpu-smoke${NAME_SUFFIX}" \
                -t math500:chat -o limit=1 \
                -o "sampling_params.max_tokens=64"
            ;;
        hybrid-pilot)
            launch_one \
                hybrid-pilot "${k}" qwen3-30b-a3b default 4 \
                "adaptive-qwen3-hybrid-k${k}-pilot${NAME_SUFFIX}" \
                -t adaptive_experts:hybrid_pilot
            ;;
        dolci-qwen-pilot)
            launch_one \
                dolci-qwen-pilot "${k}" "${DOLCI_QWEN_MODEL}" default 4 \
                "adaptive-qwen3-dolci-think-sft100k-k${k}-pilot${NAME_SUFFIX}" \
                -t math500:chat -o "sampling_params.max_tokens=${DOLCI_QWEN_MAX_TOKENS}" \
                -t gpqa_diamond:qwen3_thinking -o "sampling_params.max_tokens=${DOLCI_QWEN_MAX_TOKENS}" \
                -t ifeval_ood:qwen3_thinking -o "sampling_params.max_tokens=${DOLCI_QWEN_MAX_TOKENS}"
            ;;
        dolci-qwen-full-smoke)
            launch_one \
                dolci-qwen-full-smoke "${k}" "${DOLCI_QWEN_MODEL}" codex_python 1 \
                "qwen-dolci-v2-k${k}-${QWEN_EXPERT_WEIGHT_MODE}-smoke${NAME_SUFFIX}" \
                -t math500:chat -o limit=1 -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}" \
                -t ifeval_mt_wildchat_unused_withRewrite -o limit=1 \
                -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}" \
                -t humaneval:chat:pass_at_1:qwen3_thinking -o limit=1 \
                -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}"
            ;;
        dolci-qwen-full)
            launch_one \
                dolci-qwen-full "${k}" "${DOLCI_QWEN_MODEL}" codex_python 4 \
                "qwen-dolci-v2-k${k}-${QWEN_EXPERT_WEIGHT_MODE}${NAME_SUFFIX}" \
                -t math500:chat -o "sampling_params.max_tokens=${DOLCI_QWEN_FULL_MAX_TOKENS}" \
                -t gpqa_diamond:qwen3_thinking \
                -o "sampling_params.max_tokens=${DOLCI_QWEN_FULL_MAX_TOKENS}" \
                -t ifeval_ood -o "sampling_params.max_tokens=${DOLCI_QWEN_FULL_MAX_TOKENS}" \
                -t ifeval_mt_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=${DOLCI_QWEN_FULL_MAX_TOKENS}" \
                -t ifeval_mt_ood_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=${DOLCI_QWEN_FULL_MAX_TOKENS}" \
                -t humaneval:chat:pass_at_1:qwen3_thinking \
                -o "sampling_params.max_tokens=${DOLCI_QWEN_FULL_MAX_TOKENS}"
            ;;
        hybrid-aime)
            launch_one \
                hybrid-aime "${k}" qwen3-30b-a3b default 4 \
                "adaptive-qwen3-hybrid-k${k}-aime2026${NAME_SUFFIX}" \
                -t aime_2026:pass_at_32
            ;;
        gptoss-pilot)
            launch_one \
                gptoss-pilot "${k}" openai/gpt-oss-120b default 4 \
                "adaptive-gptoss-120b-k${k}-pilot${NAME_SUFFIX}" \
                -t adaptive_experts:hybrid_pilot
            ;;
        gptoss-aime)
            launch_one \
                gptoss-aime "${k}" openai/gpt-oss-120b default 4 \
                "adaptive-gptoss-120b-k${k}-aime2026${NAME_SUFFIX}" \
                -t aime_2026:pass_at_32
            ;;
        gptoss-smoke)
            launch_one \
                gptoss-smoke "${k}" openai/gpt-oss-120b default 1 \
                "adaptive-gptoss-120b-k${k}-tp${TENSOR_PARALLEL_SIZE}-smoke${NAME_SUFFIX}" \
                -t math500:chat -o "limit=${GPTOSS_SMOKE_LIMIT}" \
                -o "sampling_params.max_tokens=64"
            ;;
        gptoss-reference-smoke)
            launch_one \
                gptoss-reference-smoke "${k}" openai/gpt-oss-120b codex_python 1 \
                "adaptive-gptoss-120b-k${k}-reference-k${GPTOSS_REFERENCE_K}-smoke${NAME_SUFFIX}" \
                -t math500:chat -o limit=1 \
                -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}" \
                -t ifeval_mt_wildchat_unused_withRewrite -o limit=1 \
                -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}" \
                -t humaneval:chat:pass_at_1:qwen3_thinking -o limit=1 \
                -o "sampling_params.max_tokens=${SMOKE_MAX_TOKENS}"
            ;;
        gptoss-reference-expanded)
            launch_one \
                gptoss-reference-expanded "${k}" openai/gpt-oss-120b codex_python 4 \
                "adaptive-gptoss-120b-k${k}-reference-k${GPTOSS_REFERENCE_K}-expanded${NAME_SUFFIX}" \
                -t adaptive_experts:hybrid_pilot \
                -t ifeval_ood -o "sampling_params.max_tokens=32768" \
                -t ifeval_mt_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=32768" \
                -t ifeval_mt_ood_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=32768" \
                -t humaneval:chat:pass_at_1:qwen3_thinking
            ;;
        gptoss-capability-backfill)
            launch_one \
                gptoss-capability-backfill "${k}" openai/gpt-oss-120b codex_python 4 \
                "adaptive-gptoss-120b-k${k}-capability-backfill${NAME_SUFFIX}" \
                -t ifeval_ood -o "sampling_params.max_tokens=32768" \
                -t ifeval_mt_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=32768" \
                -t ifeval_mt_ood_wildchat_unused_withRewrite \
                -o "sampling_params.max_tokens=32768" \
                -t humaneval:chat:pass_at_1:qwen3_thinking
            ;;
        glm-pilot)
            launch_one \
                glm-pilot "${k}" zai-org/GLM-4.5-Air default 2 \
                "adaptive-glm45-air-k${k}-pilot${NAME_SUFFIX}" \
                -t adaptive_experts:hybrid_pilot
            ;;
        glm-aime)
            launch_one \
                glm-aime "${k}" zai-org/GLM-4.5-Air default 2 \
                "adaptive-glm45-air-k${k}-aime2026${NAME_SUFFIX}" \
                -t aime_2026:pass_at_32
            ;;
        base)
            launch_one \
                base "${k}" qwen3-30b-a3b-base default 8 \
                "adaptive-qwen3-base-k${k}-full${NAME_SUFFIX}" \
                -t olmobase:mcqa_stem \
                -t olmobase:mcqa_non_stem \
                -t olmobase:gen \
                -t olmobase:math \
                -t olmobase:easy:qa:rc \
                -t olmobase:easy:qa:bpb \
                -t olmobase:easy:math:bpb \
                -t olmobase:easy:code:bpb
            ;;
        base-math)
            launch_one \
                base-math "${k}" qwen3-30b-a3b-base default 8 \
                "adaptive-qwen3-base-k${k}-math${NAME_SUFFIX}" \
                -t olmobase:math
            ;;
    esac
done
