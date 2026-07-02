#!/usr/bin/env bash
set -euo pipefail

GROUP="${GROUP:-olmoe3-base-eval-smoke}"
FORMAT="${FORMAT:-csv}"
OUT_DIR="${OUT_DIR:-/tmp/olmoe3-base-eval-results}"
mkdir -p "${OUT_DIR}"

status_out="${OUT_DIR}/${GROUP}.beaker.${FORMAT}"
db_out="${OUT_DIR}/${GROUP}.results.json"

echo "Writing Beaker group status to ${status_out}" >&2
uv run --frozen --extra beaker olmo-eval beaker group info "${GROUP}"   --format "${FORMAT}" > "${status_out}"

echo "Writing stored result rows to ${db_out}" >&2
uv run --frozen --extra postgres olmo-eval results query   --experiment-group "${GROUP}"   --format json > "${db_out}"

echo "${status_out}"
echo "${db_out}"
