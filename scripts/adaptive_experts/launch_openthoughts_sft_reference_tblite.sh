#!/usr/bin/env bash
set -euo pipefail

# Evaluate the OpenThoughts-Agent SFT checkpoints at their training-time
# K=8-reference-scaled routing policy. K<8 keeps a native top-8 prefix without
# renormalizing it; K>8 normalizes the routed row against its top-8 prefix.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMAX_LAUNCHER="${TMAX_LAUNCHER:-${ROOT_DIR}/tmax-eval/beaker_configs/launch_eval.sh}"
TMAX_REPO_REF="${TMAX_REPO_REF:-7387d2f9142397a458dc39f0827a2ab0b4c03cda}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
PRIORITY="${PRIORITY:-urgent}"
CLUSTERS_CSV="${CLUSTERS:-ai2/jupiter,ai2/ceres,ai2/titan}"
RESULTS_ROOT="${RESULTS_ROOT:-/weka/oe-adapt-default/jacobm/tmax-eval/openthoughts-sft-reference-tblite-20260815}"
RUN_DATE="${RUN_DATE:-20260815}"
RUN_ATTEMPT="${RUN_ATTEMPT:-1}"

EXPERT_K=""
STAGE=""
LAUNCH=false

usage() {
    cat >&2 <<'EOF'
usage: launch_openthoughts_sft_reference_tblite.sh --expert-k 4|12 \
  --stage smoke|full [--launch]

Without --launch, prints the exact command and submits nothing.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --expert-k) EXPERT_K="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --launch) LAUNCH=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

if [[ "$EXPERT_K" != "4" && "$EXPERT_K" != "12" ]]; then
    usage
    exit 2
fi
if [[ "$STAGE" != "smoke" && "$STAGE" != "full" ]]; then
    usage
    exit 2
fi

if (( EXPERT_K < 8 )); then
    router_k=8
else
    router_k="$EXPERT_K"
fi

model_path="/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints/qwen35-openthoughts-agent-100k-k${EXPERT_K}-refk8-lr2e-5-flce-b64-20260813/step1523-hf"
served_name="qwen35-openthoughts-agent-100k-train-k${EXPERT_K}-step1523"
job_name="${served_name}-tblite-eval-k${EXPERT_K}-refk8-terminus2-json-${STAGE}${RUN_ATTEMPT}-${RUN_DATE}"

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
    --qwen-expert-weight-mode truncate
    --qwen-expert-keep-k "$EXPERT_K"
    --qwen-expert-router-k "$router_k"
    --qwen-expert-reference-k 8
    --qwen-expert-renormalize false
    --qwen-num-experts 256
)

# The multimodal outer config contains the MoE width under text_config. vLLM's
# --hf-overrides is shallow, so the K=4 checkpoint needs a config-only path to
# expose a top-8 router to the reference-scaling patch. K=12 is already native.
if (( EXPERT_K == 4 )); then
    common_args+=(
        --hf-config-path "${ROOT_DIR}/tmax-eval/hf_configs/qwen35_openthoughts_k4_router8"
    )
fi

if [[ "$STAGE" == "smoke" ]]; then
    common_args+=(--n-tasks 1 --n-concurrent 1)
else
    common_args+=(--n-concurrent 4)
fi

cmd=(
    "$TMAX_LAUNCHER" "$model_path"
    --name "$served_name"
    --job-name "$job_name"
    --results-dir "${RESULTS_ROOT}/${job_name}"
    "${common_args[@]}"
    --harbor-model-name "openai/${served_name}"
    --agent terminus-2
    --agent-kwarg parser_name=json
    --agent-kwarg temperature=1.0
    --agent-kwarg 'model_info={"max_input_tokens":262144,"max_output_tokens":81920,"input_cost_per_token":0.0,"output_cost_per_token":0.0}'
    --agent-kwarg 'llm_call_kwargs={"max_tokens":81920,"top_p":0.95,"extra_body":{"top_k":20}}'
)

IFS=',' read -r -a target_clusters <<< "$CLUSTERS_CSV"
for cluster in "${target_clusters[@]}"; do
    cmd+=(--cluster "$cluster")
done

printf 'Prepared command (expert_k=%s, router_k=%s, stage=%s, launch=%s):\n' \
    "$EXPERT_K" "$router_k" "$STAGE" "$LAUNCH"
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "$LAUNCH" == true ]]; then
    "${cmd[@]}"
else
    echo "Dry run only. Re-run with --launch to submit."
fi
