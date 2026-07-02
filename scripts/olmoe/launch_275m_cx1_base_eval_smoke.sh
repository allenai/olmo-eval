#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
DO_LAUNCH=false
FOLLOW=false
STORE="${STORE:-0}"

usage() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [--launch] [--follow] [--store]

Smoke-tests olmo-eval against the optimal 275m Cx1 baseline OLMo-core checkpoint.
By default this prints the Beaker spec with --dry-run. Pass --launch to submit.

Environment overrides:
  NAME=${NAME:-olmoe3-275m-cx1-baseline-arc-easy-smoke}
  GROUP=${GROUP:-olmoe3-base-eval-smoke}
  CHECKPOINT=${CHECKPOINT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr2e-3-r2/step15365}
  TASK=${TASK:-arc_easy:mc:olmo3base}
  LIMIT=${LIMIT:-10}
  CLUSTER=${CLUSTER:-ai2/titan}
  WORKSPACE=${WORKSPACE:-ai2/OLMo-3-moe-experiments}
  BUDGET=${BUDGET:-ai2/oe-other}
  PRIORITY=${PRIORITY:-urgent}
  GPUS=${GPUS:-1}
  TIMEOUT=${TIMEOUT:-2h}
  OLMO_CORE_PACKAGE=${OLMO_CORE_PACKAGE:-https://github.com/allenai/OLMo-core.git@jacobm/olmoe-dev-v2}
  EXTRA_PROVIDER_DEPS=${EXTRA_PROVIDER_DEPS:-matplotlib}
  STORE=${STORE:-0}  # Set to 1, or pass --store, once DB secrets exist in the workspace.
  S3_BUCKET=${S3_BUCKET:-ai2-llm}
  S3_PREFIX=${S3_PREFIX:-olmo-eval/olmoe3}

Results:
  Local Beaker artifacts are written under /results in the job.
  With STORE=1, metrics/predictions/requests are also uploaded to:
    s3://$S3_BUCKET/$S3_PREFIX/$GROUP/<model>_<hash>/<experiment_id>/
  And summary rows are saved to the olmo-eval Postgres DB.
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

NAME="${NAME:-olmoe3-275m-cx1-baseline-arc-easy-smoke}"
GROUP="${GROUP:-olmoe3-base-eval-smoke}"
CHECKPOINT="${CHECKPOINT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr2e-3-r2/step15365}"
TASK="${TASK:-arc_easy:mc:olmo3base}"
LIMIT="${LIMIT:-10}"
CLUSTER="${CLUSTER:-ai2/titan}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
BUDGET="${BUDGET:-ai2/oe-other}"
PRIORITY="${PRIORITY:-urgent}"
GPUS="${GPUS:-1}"
TIMEOUT="${TIMEOUT:-2h}"
OLMO_CORE_PACKAGE="${OLMO_CORE_PACKAGE:-https://github.com/allenai/OLMo-core.git@jacobm/olmoe-dev-v2}"
EXTRA_PROVIDER_DEPS="${EXTRA_PROVIDER_DEPS:-matplotlib}"
S3_BUCKET="${S3_BUCKET:-ai2-llm}"
S3_PREFIX="${S3_PREFIX:-olmo-eval/olmoe3}"

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "Missing reconstructed OLMo-core config: ${CHECKPOINT}/config.json" >&2
  exit 1
fi

cmd=(
  uv run --frozen --extra beaker olmo-eval beaker launch
  --name "${NAME}"
  --group "${GROUP}"
  --model "${CHECKPOINT}"
  --task "${TASK}"
  --override "limit=${LIMIT}"
  --harness default
  --override provider.kind=olmo_core
  --override provider.package="${OLMO_CORE_PACKAGE}"
  --override provider.dependencies="[${EXTRA_PROVIDER_DEPS}]"
  --override provider.max_model_len=8192
  --override provider.dtype=bfloat16
  --cluster "${CLUSTER}"
  --workspace "${WORKSPACE}"
  --budget "${BUDGET}"
  --priority "${PRIORITY}"
  --preemptible
  --gpus "${GPUS}"
  --timeout "${TIMEOUT}"
  --yes
)

if [[ "${STORE}" == "1" ]]; then
  cmd+=(
    --store
    --s3-bucket "${S3_BUCKET}"
    --s3-prefix "${S3_PREFIX}"
  )
else
  cmd+=(--no-store)
fi

if [[ "${FOLLOW}" == "false" ]]; then
  cmd+=(--no-follow)
fi

if [[ "${DO_LAUNCH}" == "false" ]]; then
  cmd+=(--dry-run)
  echo "Dry-run only. Pass --launch to submit." >&2
fi

printf '+ %q ' "${cmd[@]}"
printf '
'
exec "${cmd[@]}"
