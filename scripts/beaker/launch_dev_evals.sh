#!/usr/bin/env bash

# Launch the standard comparison suite for one model:
#   - full datasets for the relatively quick non-agentic evaluations
#   - fixed development subsets for the expensive agentic evaluations

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SAFE_LAUNCHER="${SCRIPT_DIR}/launch_safe_evals.sh"
DEEPSCHOLAR_LAUNCHER="${SCRIPT_DIR}/launch_deepscholar_full.sh"

MODEL=""
MODEL_SLUG=""
ONLY="all"
SAMPLE_SEED="42"
EXPERTQA_LIMIT="100"
LITSEARCH_LIMIT="50"
SAGE_LIMIT="50"
DEEPSCHOLAR_LIMIT="10"

CLUSTER="ai2/ceres"
WORKSPACE="ai2/olmo-eval-debug"
PRIORITY="urgent"
TIMEOUT="24h"
GPUS="2"
S2_SECRET="roryd_S2_API_KEY"
SERPER_SECRET="roryd_SERPER_API_KEY"
OPENAI_SECRET="roryd_OPENAI_API_KEY"
OPENALEX_EMAIL="roryd@allenai.org"
RUN_TAG="$(date -u +%Y%m%d-%H%M%S)"
DRY_RUN=false
PRINT_ONLY=false

usage() {
    cat <<'EOF'
Usage: scripts/beaker/launch_dev_evals.sh --model REF --slug NAME [options]

Default suite:
  Full:     LitSearch-rerank, IFEval, MATH-500, and MMLU
  Dev 100:  ExpertQA
  Dev 50:   LitSearch-open, SAGE-open, and SAGE-short
  Dev 10:   DeepScholar-Bench (fixed indices 0-9)

Options:
  --model REF              Hugging Face model ref (required)
  --slug NAME              Name-safe model label (required)
  --only GROUP             all, small, large, core, mmlu, paper, sage,
                           expertqa, or deepscholar (default: all)
  --sample-seed N          Fixed random-sample seed for standard tasks (default: 42)
  --expertqa-limit N       ExpertQA dev size (default: 100)
  --litsearch-limit N      LitSearch-open dev size (default: 50)
  --sage-limit N           Per-task SAGE dev size (default: 50)
  --deepscholar-limit N    DeepScholar fixed-prefix size (default: 10)
  --gpus N                 GPUs per job (default: 2)
  --cluster NAME           Beaker cluster (default: ai2/ceres)
  --workspace NAME         Beaker workspace (default: ai2/olmo-eval-debug)
  --priority LEVEL         Beaker priority (default: urgent)
  --timeout DURATION       Beaker job timeout (default: 24h)
  --run-tag TAG            Shared job/group suffix
  --s2-secret NAME         Beaker secret mapped to S2_API_KEY
  --serper-secret NAME     Beaker secret mapped to SERPER_API_KEY
  --openai-secret NAME     Beaker secret mapped to OPENAI_API_KEY
  --openalex-email EMAIL   Contact email used by DeepScholar OpenAlex calls
  --dry-run                Render safe-eval specs and print DeepScholar commands
  --print-only             Print every shell-escaped command without running it
  -h, --help               Show this help

Use --print-only before launching a new model profile. The default `all` scope
submits six jobs; it does not follow them.
EOF
}

require_value() {
    local option="$1"
    local count="$2"
    if [[ "$count" -lt 2 ]]; then
        echo "${option} requires a value" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) require_value "$1" "$#"; MODEL="$2"; shift 2 ;;
        --slug) require_value "$1" "$#"; MODEL_SLUG="$2"; shift 2 ;;
        --only) require_value "$1" "$#"; ONLY="$2"; shift 2 ;;
        --sample-seed) require_value "$1" "$#"; SAMPLE_SEED="$2"; shift 2 ;;
        --expertqa-limit) require_value "$1" "$#"; EXPERTQA_LIMIT="$2"; shift 2 ;;
        --litsearch-limit) require_value "$1" "$#"; LITSEARCH_LIMIT="$2"; shift 2 ;;
        --sage-limit) require_value "$1" "$#"; SAGE_LIMIT="$2"; shift 2 ;;
        --deepscholar-limit) require_value "$1" "$#"; DEEPSCHOLAR_LIMIT="$2"; shift 2 ;;
        --gpus) require_value "$1" "$#"; GPUS="$2"; shift 2 ;;
        --cluster) require_value "$1" "$#"; CLUSTER="$2"; shift 2 ;;
        --workspace) require_value "$1" "$#"; WORKSPACE="$2"; shift 2 ;;
        --priority) require_value "$1" "$#"; PRIORITY="$2"; shift 2 ;;
        --timeout) require_value "$1" "$#"; TIMEOUT="$2"; shift 2 ;;
        --run-tag) require_value "$1" "$#"; RUN_TAG="$2"; shift 2 ;;
        --s2-secret) require_value "$1" "$#"; S2_SECRET="$2"; shift 2 ;;
        --serper-secret) require_value "$1" "$#"; SERPER_SECRET="$2"; shift 2 ;;
        --openai-secret) require_value "$1" "$#"; OPENAI_SECRET="$2"; shift 2 ;;
        --openalex-email) require_value "$1" "$#"; OPENALEX_EMAIL="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --print-only) PRINT_ONLY=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$MODEL" || -z "$MODEL_SLUG" ]]; then
    echo "--model and --slug are required" >&2
    usage >&2
    exit 2
fi

case "$ONLY" in
    all|small|large|core|mmlu|paper|sage|expertqa|deepscholar) ;;
    *)
        echo "--only must be one of: all, small, large, core, mmlu, paper, sage, expertqa, deepscholar" >&2
        exit 2
        ;;
esac

for value_name in EXPERTQA_LIMIT LITSEARCH_LIMIT SAGE_LIMIT DEEPSCHOLAR_LIMIT; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer" >&2
        exit 2
    fi
done

if [[ ! "$SAMPLE_SEED" =~ ^[0-9]+$ ]]; then
    echo "--sample-seed must be a non-negative integer" >&2
    exit 2
fi

BEAKER_GROUP="safe-evals-${MODEL_SLUG}-${RUN_TAG}"
failures=0

is_selected() {
    local group="$1"
    case "${ONLY}:${group}" in
        all:*|small:core|small:mmlu|large:paper|large:sage|large:expertqa|large:deepscholar)
            return 0
            ;;
        "${group}:${group}")
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

run_safe() {
    local group="$1"
    local limit="${2:-}"
    local command=(
        "$SAFE_LAUNCHER"
        --model "$MODEL"
        --slug "$MODEL_SLUG"
        --only "$group"
        --sample-seed "$SAMPLE_SEED"
        --gpus "$GPUS"
        --cluster "$CLUSTER"
        --workspace "$WORKSPACE"
        --priority "$PRIORITY"
        --timeout "$TIMEOUT"
        --run-tag "$RUN_TAG"
        --s2-secret "$S2_SECRET"
        --serper-secret "$SERPER_SECRET"
        --openai-secret "$OPENAI_SECRET"
    )

    if [[ -n "$limit" ]]; then
        command+=(--limit "$limit")
    fi
    if [[ "$PRINT_ONLY" == true ]]; then
        command+=(--print-only)
    elif [[ "$DRY_RUN" == true ]]; then
        command+=(--dry-run)
    fi

    if ! "${command[@]}"; then
        failures=$((failures + 1))
    fi
}

run_deepscholar() {
    local command=(
        "$DEEPSCHOLAR_LAUNCHER"
        --only-model "$MODEL"
        --start-idx 0
        --limit "$DEEPSCHOLAR_LIMIT"
    )
    if [[ "$PRINT_ONLY" == true || "$DRY_RUN" == true ]]; then
        command+=(--print-only)
    fi

    if ! RUN_TAG="$RUN_TAG" \
        GROUP="$BEAKER_GROUP" \
        CLUSTER="$CLUSTER" \
        WORKSPACE="$WORKSPACE" \
        PRIORITY="$PRIORITY" \
        TIMEOUT="$TIMEOUT" \
        GPUS="$GPUS" \
        OPENALEX_EMAIL="$OPENALEX_EMAIL" \
        S2_SECRET="$S2_SECRET" \
        "${command[@]}"; then
        failures=$((failures + 1))
    fi
}

if is_selected core; then
    run_safe core
fi
if is_selected mmlu; then
    run_safe mmlu
fi
if is_selected paper; then
    run_safe paper "$LITSEARCH_LIMIT"
fi
if is_selected sage; then
    run_safe sage "$SAGE_LIMIT"
fi
if is_selected expertqa; then
    run_safe expertqa "$EXPERTQA_LIMIT"
fi
if is_selected deepscholar; then
    run_deepscholar
fi

if [[ "$failures" -ne 0 ]]; then
    echo "${failures} launch group(s) failed" >&2
    exit 1
fi

echo
if [[ "$PRINT_ONLY" == true ]]; then
    echo "Printed the dev-suite commands for Beaker group: ${BEAKER_GROUP}"
elif [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete for Beaker group: ${BEAKER_GROUP}"
else
    echo "Submitted the dev suite to Beaker group: ${BEAKER_GROUP}"
fi
