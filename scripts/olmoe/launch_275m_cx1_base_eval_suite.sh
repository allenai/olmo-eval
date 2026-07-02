#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
DO_LAUNCH=false
FOLLOW=false
STORE="${STORE:-0}"
INCLUDE_CODE_EXEC="${INCLUDE_CODE_EXEC:-0}"

usage() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [--launch] [--follow] [--store] [--include-code-exec]

Launches OLMoBase eval suites for the converted 275m Cx1 OLMoE checkpoints.
By default this prints Beaker specs with --dry-run. Pass --launch to submit.

Environment overrides:
  MODEL_SET=${MODEL_SET:-both}  # both, baseline, q3td
  GROUP=${GROUP:-olmoe3-275m-cx1-base-eval-suite-smoke}
  CLUSTER=${CLUSTER:-ai2/titan}
  WORKSPACE=${WORKSPACE:-ai2/OLMo-3-moe-experiments}
  BUDGET=${BUDGET:-ai2/oe-other}
  PRIORITY=${PRIORITY:-urgent}
  GPUS=${GPUS:-1}
  TIMEOUT=${TIMEOUT:-24h}
  NON_EXEC_HARNESS=${NON_EXEC_HARNESS:-default}
  EXEC_HARNESS=${EXEC_HARNESS:-codex_universal}
  PROVIDER_NUM_INSTANCES=${PROVIDER_NUM_INSTANCES:-1}
  HF_TOKEN_SECRET=${HF_TOKEN_SECRET:-jacobm_HF_TOKEN}
  NAME_SUFFIX=${NAME_SUFFIX:-}
  STORE=${STORE:-0}
  INCLUDE_CODE_EXEC=${INCLUDE_CODE_EXEC:-0}

By default this launches the non-code-exec OLMoBase suites:
  olmobase:mcqa_stem, olmobase:mcqa_non_stem, olmobase:gen, olmobase:math,
  olmobase:easy:qa:rc, olmobase:easy:qa:bpb, olmobase:easy:math:bpb,
  olmobase:easy:code:bpb

Pass --include-code-exec to also launch olmobase:code via codex_universal.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --launch)
      DO_LAUNCH=true
      ;;
    --follow)
      FOLLOW=true
      ;;
    --store)
      STORE=1
      ;;
    --include-code-exec)
      INCLUDE_CODE_EXEC=1
      ;;
    --dry-run)
      DO_LAUNCH=false
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
  shift
done

MODEL_SET="${MODEL_SET:-both}"
GROUP="${GROUP:-olmoe3-275m-cx1-base-eval-suite-smoke}"
CLUSTER="${CLUSTER:-ai2/titan}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
BUDGET="${BUDGET:-ai2/oe-other}"
PRIORITY="${PRIORITY:-urgent}"
GPUS="${GPUS:-1}"
TIMEOUT="${TIMEOUT:-24h}"
NON_EXEC_HARNESS="${NON_EXEC_HARNESS:-default}"
EXEC_HARNESS="${EXEC_HARNESS:-codex_universal}"
PROVIDER_NUM_INSTANCES="${PROVIDER_NUM_INSTANCES:-1}"
HF_TOKEN_SECRET="${HF_TOKEN_SECRET:-jacobm_HF_TOKEN}"
NAME_SUFFIX="${NAME_SUFFIX:-}"
MODAL_ENVIRONMENT="${MODAL_ENVIRONMENT:-oe-eval}"
S3_BUCKET="${S3_BUCKET:-ai2-llm}"
S3_PREFIX="${S3_PREFIX:-olmo-eval/olmoe3}"

BASELINE_CKPT="${BASELINE_CKPT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/hf-checkpoints/olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr2e-3-r2/step15365}"
Q3TD_CKPT="${Q3TD_CKPT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/hf-checkpoints/q3-275m-cx1-q3td128e8k-lr2e-3-r1/step15365}"

declare -a model_labels=()
declare -a model_paths=()
case "${MODEL_SET}" in
  both)
    model_labels+=("baseline" "q3td")
    model_paths+=("${BASELINE_CKPT}" "${Q3TD_CKPT}")
    ;;
  baseline)
    model_labels+=("baseline")
    model_paths+=("${BASELINE_CKPT}")
    ;;
  q3td|qwen|qwen3)
    model_labels+=("q3td")
    model_paths+=("${Q3TD_CKPT}")
    ;;
  *)
    echo "Unknown MODEL_SET='${MODEL_SET}'. Use both, baseline, or q3td." >&2
    exit 2
    ;;
esac

for model in "${model_paths[@]}"; do
  if [[ ! -f "${model}/config.json" ]]; then
    echo "Missing HF checkpoint config: ${model}/config.json" >&2
    exit 1
  fi
done

non_exec_suites=(
  "olmobase:mcqa_stem"
  "olmobase:mcqa_non_stem"
  "olmobase:gen"
  "olmobase:math"
  "olmobase:easy:qa:rc"
  "olmobase:easy:qa:bpb"
  "olmobase:easy:math:bpb"
  "olmobase:easy:code:bpb"
)

code_exec_suites=(
  "olmobase:code"
)

run_cmd() {
  local label=$1
  shift
  local cmd=("$@")

  echo "========================================="
  echo "${label}"
  echo "========================================="
  printf '+ %q ' "${cmd[@]}"
  printf '\n\n'
  "${cmd[@]}"
  echo ""
}

build_common_tail() {
  common_tail=(
    --cluster "${CLUSTER}"
    --workspace "${WORKSPACE}"
    --budget "${BUDGET}"
    --group "${GROUP}"
    --priority "${PRIORITY}"
    --preemptible
    --gpus "${GPUS}"
    --timeout "${TIMEOUT}"
    --yes
  )

  if [[ "${STORE}" == "1" ]]; then
    common_tail+=(
      --store
      --s3-bucket "${S3_BUCKET}"
      --s3-prefix "${S3_PREFIX}"
    )
  else
    common_tail+=(--no-store)
  fi

  if [[ "${FOLLOW}" == "false" ]]; then
    common_tail+=(--no-follow)
  fi

  if [[ "${DO_LAUNCH}" == "false" ]]; then
    common_tail+=(--dry-run)
  fi
}

launch_suite_group() {
  local model_label=$1
  local model_path=$2
  local suite_label=$3
  local harness=$4
  local include_exec_args=$5
  shift 5
  local suites=("$@")
  local cmd=(
    uv run --frozen --extra beaker olmo-eval beaker launch
    --name "olmoe3-275m-cx1-${model_label}-${suite_label}${NAME_SUFFIX}"
    --harness "${harness}"
    --override "provider.num_instances=${PROVIDER_NUM_INSTANCES}"
    --override provider.kind=hf
    --override provider.trust_remote_code=true
    --override provider.dtype=bfloat16
    --secret-env "${HF_TOKEN_SECRET}:HF_TOKEN"
    --model "${model_path}"
  )

  if [[ "${include_exec_args}" == "true" ]]; then
    cmd+=(
      --override 'sandboxes={"mode":"modal","instances":64, "min_instances": 56, "registry_auth":{"provider":"gcp"}}'
      --env "MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT}"
      --secret-env "ai2-tylerm_MODAL_TOKEN_ID:MODAL_TOKEN_ID"
      --secret-env "ai2-tylerm_MODAL_TOKEN_SECRET:MODAL_TOKEN_SECRET"
    )
  fi

  for suite in "${suites[@]}"; do
    cmd+=(--task "${suite}@${PRIORITY}")
  done

  build_common_tail
  cmd+=("${common_tail[@]}")
  run_cmd "${model_label}: ${suite_label}" "${cmd[@]}"
}

if [[ "${DO_LAUNCH}" == "false" ]]; then
  echo "Dry-run only. Pass --launch to submit." >&2
fi

for i in "${!model_paths[@]}"; do
  launch_suite_group "${model_labels[$i]}" "${model_paths[$i]}" "nonexec" "${NON_EXEC_HARNESS}" false "${non_exec_suites[@]}"
done

if [[ "${INCLUDE_CODE_EXEC}" == "1" ]]; then
  for i in "${!model_paths[@]}"; do
    launch_suite_group "${model_labels[$i]}" "${model_paths[$i]}" "codeexec" "${EXEC_HARNESS}" true "${code_exec_suites[@]}"
  done
fi
