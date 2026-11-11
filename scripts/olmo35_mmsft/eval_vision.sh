#!/bin/bash
# Image-QA evals of one Olmo 3.5 multimodal checkpoint through the olmo_core_vlm provider.
#
# usage: eval_vision.sh <short-name> <OLMo-core multimodal export dir> [olmo35|document]
#   olmo35    (default) the SFT chat layout: Olmo 3.5 chat template, think tags promoted,
#             thinking off (see src/olmo_eval/inference/providers/olmo_core_vlm/olmo35_chat.py)
#   document  the mid-training plain-document layout (base checkpoints)
# The export dir holds model.safetensors + olmo_core_config.json (OLMo-core's
# export_multimodal_olmoddp.py). Env: TASKS (default: the first suite below), LIMIT (examples per
# task; unset = all), GPUS (default 8, one model replica per GPU), OLMO_CORE_SHA (the OLMo-core the
# job installs; default: allenai/OLMo-core#942's head, the multimodal SFT code), WORKSPACE,
# CLUSTER, PRIORITY, BUDGET, TIMEOUT, GROUP, DRY_RUN=1. This checkout must be pushed and clean:
# the job runs its HEAD.
set -euo pipefail
[[ $# -ge 2 ]] || { sed -n 2,14p "$0"; exit 2; }
NAME=$1; MODEL=${2%/}; MODE=${3:-olmo35}
case $MODE in olmo35 | document) ;; *) echo "mode must be olmo35 or document" >&2; exit 2 ;; esac
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
TASKS=${TASKS:-"chart_qa vqa2 doc_qa text_vqa ai2d countbench_qa real_world_qa"}
GPUS=${GPUS:-8}; WORKSPACE=${WORKSPACE:-ai2/olmo-instruct}; CLUSTER=${CLUSTER:-ai2/holmes}
PRIORITY=${PRIORITY:-urgent}; TIMEOUT=${TIMEOUT:-24h}; GROUP=${GROUP:-olmo35-mmsft-eval}
OLMO_CORE_SHA=${OLMO_CORE_SHA:-accab373bae0f316f3e6092a0228ee298cefb5a7}
OLMO_EVAL_PY=${OLMO_EVAL_PY:-python}
[[ -f $MODEL/model.safetensors && -f $MODEL/olmo_core_config.json ]] \
  || { echo "not an OLMo-core multimodal export: $MODEL" >&2; exit 2; }
if [[ -n $(git -C "$ROOT" status --porcelain --untracked-files=no) ]]; then
  echo "commit and push $ROOT first (the job runs its HEAD)" >&2; exit 2
fi
TASK_ARGS=()
for t in $TASKS; do
  TASK_ARGS+=(-t "$t")
  [[ -n ${LIMIT:-} ]] && TASK_ARGS+=(-o "limit=$LIMIT")
done
DRY=(--yes); [[ -n ${DRY_RUN:-} ]] && DRY=(--dry-run)
BUD=(); [[ -n ${BUDGET:-} ]] && BUD=(-B "$BUDGET")
(cd "$ROOT" && PYTHONPATH=$ROOT/src "$OLMO_EVAL_PY" -c "from olmo_eval.cli import main; main()" \
  beaker launch -n "mmsft-vis-$NAME-$MODE" -m "$MODEL" "${TASK_ARGS[@]}" \
  --harness default -o provider.kind=olmo_core_vlm -o provider.num_instances="$GPUS" \
  -o "provider.kwargs.prompt_format=$MODE" \
  -o "provider.package=https://github.com/allenai/OLMo-core.git@$OLMO_CORE_SHA" \
  -o 'provider.dependencies=["flash-linear-attention==0.5.2"]' \
  -c "$CLUSTER" --gpus "$GPUS" -p "$PRIORITY" --no-preemptible -T "$TIMEOUT" \
  -w "$WORKSPACE" "${BUD[@]}" -g "$GROUP" \
  --env OLMO_EVAL_PROVIDER_INIT_TIMEOUT=3600 --env OMP_NUM_THREADS=8 \
  --no-save-requests --no-follow "${DRY[@]}")
