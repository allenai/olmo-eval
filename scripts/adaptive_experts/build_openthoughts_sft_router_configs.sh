#!/usr/bin/env bash
set -euo pipefail

# Produce config-only directories for cross-evaluating the OpenThoughts SFT
# checkpoints at a router width different from the width stored by training.
# The model weights and tokenizer still come from MODEL_PATH; vLLM reads only
# config.json from these paths through --hf-config-path.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-adapt-default/jacobm/olmoe3/post-training/checkpoints}"
CONFIG_ROOT="${CONFIG_ROOT:-${ROOT_DIR}/tmax-eval/hf_configs}"

build_config() {
    local train_k="$1"
    local router_k="$2"
    local source_dir="${CHECKPOINT_ROOT}/qwen35-openthoughts-agent-100k-k${train_k}-refk8-lr2e-5-flce-b64-20260813/step1523-hf"
    local target_dir="${CONFIG_ROOT}/qwen35_openthoughts_train_k${train_k}_router${router_k}"
    local temporary

    test -f "${source_dir}/config.json"
    mkdir -p "$target_dir"
    temporary="$(mktemp "${target_dir}/config.json.XXXXXX")"
    jq --argjson router_k "$router_k" \
        '.text_config.num_experts_per_tok = $router_k' \
        "${source_dir}/config.json" > "$temporary"
    mv "$temporary" "${target_dir}/config.json"
    test "$(jq -r '.text_config.num_experts_per_tok' "${target_dir}/config.json")" = "$router_k"
    printf 'Built train-K=%s router-K=%s config: %s\n' "$train_k" "$router_k" "$target_dir"
}

build_config 4 8
build_config 4 10
build_config 4 12
build_config 8 10
build_config 8 12
build_config 12 8
build_config 12 10
