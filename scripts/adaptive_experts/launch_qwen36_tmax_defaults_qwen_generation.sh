#!/usr/bin/env bash
set -euo pipefail

# Prepare a Qwen3.6 Terminal-Bench run that changes only the model's generation
# envelope from the TMax Beaker launcher's evaluation defaults. This script is
# deliberately dry-run-only unless --launch is supplied.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMAX_LAUNCHER="${TMAX_LAUNCHER:-${ROOT_DIR}/tmax-eval/beaker_configs/launch_eval.sh}"
TMAX_REPO_REF="${TMAX_REPO_REF:-7387d2f9142397a458dc39f0827a2ab0b4c03cda}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
EXPERT_K="${EXPERT_K:-8}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
PRIORITY="${PRIORITY:-urgent}"
CLUSTERS_CSV="${CLUSTERS:-ai2/jupiter,ai2/ceres,ai2/titan}"
RESULTS_ROOT="${RESULTS_ROOT:-/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-35b-a3b}"
RUN_DATE="${RUN_DATE:-$(date -u +%Y%m%d)}"
BENCHMARK=""
LAUNCH=false

usage() {
    cat >&2 <<'EOF'
usage: launch_qwen36_tmax_defaults_qwen_generation.sh \
  --benchmark tblite|tb21 [--k N] [--launch]

Without --launch, prints the exact command and submits nothing.

Environment overrides:
  MODEL, WORKSPACE, PRIORITY, CLUSTERS, RESULTS_ROOT, RUN_DATE,
  TMAX_LAUNCHER, TMAX_REPO_REF
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --k) EXPERT_K="$2"; shift 2 ;;
        --launch) LAUNCH=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

if [[ "$BENCHMARK" != "tblite" && "$BENCHMARK" != "tb21" ]]; then
    usage
    exit 2
fi
if ! [[ "$EXPERT_K" =~ ^[1-9][0-9]*$ ]] || (( EXPERT_K > 256 )); then
    echo "invalid expert count: $EXPERT_K" >&2
    exit 2
fi
if [[ ! -x "$TMAX_LAUNCHER" ]]; then
    echo "TMax launcher is missing or not executable: $TMAX_LAUNCHER" >&2
    exit 2
fi

case "$BENCHMARK" in
    tblite)
        DATASET="openthoughts-tblite@2.0"
        DATASET_LABEL="tblite2"
        ;;
    tb21)
        # Pin the exact TB2.1 revision already validated by our Harbor smoke.
        DATASET="terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
        DATASET_LABEL="tb21"
        ;;
esac

routing_args=()
routing_label="native"
if (( EXPERT_K < 8 )); then
    routing_label="reference"
    routing_args=(
        --qwen-expert-weight-mode truncate
        --qwen-expert-keep-k "$EXPERT_K"
        --qwen-expert-router-k 8
        --qwen-expert-reference-k 8
        --qwen-expert-renormalize false
        --qwen-num-experts 256
    )
elif (( EXPERT_K > 8 )); then
    routing_label="reference"
    routing_args=(
        --hf-overrides "{\"text_config\":{\"num_experts_per_tok\":${EXPERT_K}}}"
        --qwen-expert-weight-mode truncate
        --qwen-expert-keep-k "$EXPERT_K"
        --qwen-expert-router-k "$EXPERT_K"
        --qwen-expert-reference-k 8
        --qwen-expert-renormalize false
        --qwen-num-experts 256
    )
fi

job_name="qwen36-${DATASET_LABEL}-k${EXPERT_K}-${routing_label}-tmax-defaults-qwen-generation-${RUN_DATE}"
cmd=(
    "$TMAX_LAUNCHER" "$MODEL"
    --name qwen36-35b-a3b
    --job-name "$job_name"
    --results-dir "${RESULTS_ROOT}/${job_name}"
    --workspace "$WORKSPACE"
    --priority "$PRIORITY"
    --repo-ref "$TMAX_REPO_REF"
    --dataset "$DATASET"
    --max-model-len 262144
    --hosted-vllm-model-info '{"max_input_tokens":262144,"max_output_tokens":81920,"input_cost_per_token":0.0,"output_cost_per_token":0.0}'
    --agent-kwarg temperature=1.0
    --agent-kwarg top_p=0.95
    --agent-kwarg top_k=20
    --agent-kwarg max_tokens=81920
    "${routing_args[@]}"
)

# Cluster/workspace choices affect only Beaker placement, not the evaluation
# protocol. Everything else intentionally inherits TMax's launcher defaults.
IFS=',' read -r -a target_clusters <<< "$CLUSTERS_CSV"
for cluster in "${target_clusters[@]}"; do
    cmd+=(--cluster "$cluster")
done

printf 'Prepared command (%s, K=%s, launch=%s):\n' "$BENCHMARK" "$EXPERT_K" "$LAUNCH"
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "$LAUNCH" == true ]]; then
    "${cmd[@]}"
else
    echo "Dry run only. Re-run with --launch to submit."
fi
