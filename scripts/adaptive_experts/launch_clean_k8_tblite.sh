#!/usr/bin/env bash
set -euo pipefail

# Clean K=8 TB-Lite comparison. Both models use the same vLLM serving envelope
# and Qwen thinking-mode sampling recipe, while retaining the agent protocol
# appropriate to each model's training data.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMAX_LAUNCHER="${TMAX_LAUNCHER:-${ROOT_DIR}/tmax-eval/beaker_configs/launch_eval.sh}"
TMAX_REPO_REF="${TMAX_REPO_REF:-7387d2f9142397a458dc39f0827a2ab0b4c03cda}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
PRIORITY="${PRIORITY:-urgent}"
CLUSTERS_CSV="${CLUSTERS:-ai2/jupiter,ai2/ceres,ai2/titan}"
RESULTS_ROOT="${RESULTS_ROOT:-/weka/oe-adapt-default/jacobm/tmax-eval/clean-k8-tblite-20260815}"
RUN_DATE="${RUN_DATE:-20260815}"
RUN_ATTEMPT="${RUN_ATTEMPT:-1}"

MODEL_KIND=""
STAGE=""
LAUNCH=false

usage() {
    cat >&2 <<'EOF'
usage: launch_clean_k8_tblite.sh --model qwen36|qwen35-sft \
  --stage smoke|full [--launch]

Without --launch, prints the exact command and submits nothing.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_KIND="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --launch) LAUNCH=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

if [[ "$MODEL_KIND" != "qwen36" && "$MODEL_KIND" != "qwen35-sft" ]]; then
    usage
    exit 2
fi
if [[ "$STAGE" != "smoke" && "$STAGE" != "full" ]]; then
    usage
    exit 2
fi

common_args=(
    --gpus 8
    --tp 2
    --dp 4
    --dataset openthoughts-tblite@2.0
    --workspace "$WORKSPACE"
    --priority "$PRIORITY"
    --allocated
    --repo-ref "$TMAX_REPO_REF"
    --tool-call-parser qwen3_xml
    --reasoning-parser qwen3
    --language-model-only
    --max-model-len 262144
    --gpu-memory-utilization 0.95
    --max-num-seqs 1
    --gdn-prefill-backend triton
    --hosted-vllm-model-info '{"max_input_tokens":262144,"max_output_tokens":81920,"input_cost_per_token":0.0,"output_cost_per_token":0.0}'
)

if [[ "$STAGE" == "smoke" ]]; then
    common_args+=(--n-tasks 1 --n-concurrent 1)
else
    common_args+=(--n-concurrent 4)
fi

case "$MODEL_KIND" in
    qwen36)
        model_path="Qwen/Qwen3.6-35B-A3B"
        served_name="qwen36-35b-a3b"
        protocol_label="vanillux2-qwen3xml"
        agent_args=(
            --agent Vanillux2Agent:Vanillux2Agent
            --agent-kwarg temperature=1.0
            --agent-kwarg top_p=0.95
            --agent-kwarg top_k=20
            --agent-kwarg max_tokens=81920
        )
        ;;
    qwen35-sft)
        model_path="/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints/qwen35-openthoughts-agent-100k-k8-refk8-lr2e-5-flce-b64-20260813/step1523-hf"
        served_name="qwen35-openthoughts-agent-100k-step1523"
        protocol_label="terminus2-json-qwen3xml"
        agent_args=(
            --harbor-model-name "openai/${served_name}"
            --agent terminus-2
            --agent-kwarg parser_name=json
            --agent-kwarg temperature=1.0
            --agent-kwarg 'model_info={"max_input_tokens":262144,"max_output_tokens":81920,"input_cost_per_token":0.0,"output_cost_per_token":0.0}'
            --agent-kwarg 'llm_call_kwargs={"max_tokens":81920,"top_p":0.95,"extra_body":{"top_k":20}}'
        )
        ;;
esac

job_name="${served_name}-tblite-k8-${protocol_label}-${STAGE}${RUN_ATTEMPT}-${RUN_DATE}"
cmd=(
    "$TMAX_LAUNCHER" "$model_path"
    --name "$served_name"
    --job-name "$job_name"
    --results-dir "${RESULTS_ROOT}/${job_name}"
    "${common_args[@]}"
    "${agent_args[@]}"
)

IFS=',' read -r -a target_clusters <<< "$CLUSTERS_CSV"
for cluster in "${target_clusters[@]}"; do
    cmd+=(--cluster "$cluster")
done

printf 'Prepared command (model=%s, stage=%s, launch=%s):\n' "$MODEL_KIND" "$STAGE" "$LAUNCH"
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "$LAUNCH" == true ]]; then
    "${cmd[@]}"
else
    echo "Dry run only. Re-run with --launch to submit."
fi
