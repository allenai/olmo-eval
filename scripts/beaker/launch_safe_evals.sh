#!/usr/bin/env bash

# Launch the benchmark tasks that do not depend on OpenAlex.
#
# The tasks are split by harness so each benchmark receives the tools it expects:
#   core:      no tools
#   base:      base-model STEM multiple-choice suite
#   gpqa:      GPQA Diamond with base-style multiple-choice log-likelihood scoring
#   paper:     Semantic Scholar paper search (LitSearch)
#   sage:      Semantic Scholar + web search + webpage browsing
#   expertqa:  Serper web search/fetch plus the OpenAI scoring key

set -uo pipefail

MODEL="google/gemma-4-26B-A4B-it"
MODEL_SLUG="gemma4-26b-a4b-it"
CLUSTER="ai2/ceres"
WORKSPACE="ai2/olmo-eval-debug"
PRIORITY="urgent"
TIMEOUT="24h"
GPUS="2"
MAX_MODEL_LEN="32768"
SAGE_MAX_MODEL_LEN="65536"
STARTUP_TIMEOUT="1800"
BEAKER_IMAGE=""
MMLU_MAX_MODEL_LEN="4096"
GPQA_MAX_MODEL_LEN="8192"
MMLU_MAX_NUM_BATCHED_TOKENS="2048"
MMLU_GPU_MEMORY_UTILIZATION="0.85"
GEMMA4_MATH_MAX_TOKENS="4096"
S2_SECRET="roryd_S2_API_KEY"
SERPER_SECRET="roryd_SERPER_API_KEY"
OPENAI_SECRET="roryd_OPENAI_API_KEY"
ONLY="all"
LIMIT=""
SAMPLE_SEED="42"
BASE_SEED=""
DRY_RUN=false
PRINT_ONLY=false
RUN_TAG="$(date -u +%Y%m%d-%H%M%S)"

usage() {
    cat <<'EOF'
Usage: scripts/beaker/launch_safe_evals.sh [options]

Launches non-following Beaker jobs for one model:
  core      litsearch_rerank, ifeval_ood, math500; Gemma MATH runs separately
  math      math500 only (uses model-specific generation defaults)
  mmlu      mmlu (kept separate; prone to OOM, needs its own job)
  base      ARC, MMLU-STEM, MedMCQA, MedQA, and SciQ base-model suite
  gpqa      gpqa_diamond:mc (opt-in while the base-eval profile is validated)
  paper     litsearch
  sage      sage_open_ended, sage_short_form (agentic search + browsing)
  expertqa  expertqa

Options:
  --model REF              Hugging Face model ref
  --slug NAME              Name-safe model label used in Beaker job names
  --only GROUP             all, core, math, mmlu, base, gpqa, paper, sage, or expertqa (default: all)
  --limit N                Run a reproducible random sample of N instances
  --sample-seed N          Seed used with --limit (default: 42)
  --base-seed N            Seed and few-shot seed for a full base-suite run
  --gpus N                 GPUs per job (default: 2)
  --cluster NAME           Beaker cluster (default: ai2/ceres)
  --workspace NAME         Beaker workspace (default: ai2/olmo-eval-debug)
  --priority LEVEL         Beaker priority (default: urgent)
  --timeout DURATION       Beaker job timeout (default: 24h)
  --run-tag TAG            Shared suffix/group tag
  --max-model-len N        vLLM context length for non-MMLU jobs (default: 32768)
  --sage-max-model-len N   vLLM context length for SAGE jobs (default: 65536)
  --mmlu-max-model-len N   MMLU context length (default: 4096)
  --gpqa-max-model-len N   GPQA context length (default: 8192)
  --mmlu-max-batch-tokens N
                           MMLU vLLM prefill-token cap (default: 2048)
  --mmlu-gpu-memory-utilization F
                           MMLU vLLM memory fraction (default: 0.85)
  --startup-timeout N      vLLM startup timeout in seconds (default: 1800)
  --image NAME             Beaker image override
  --s2-secret NAME         Beaker secret mapped to S2_API_KEY
  --serper-secret NAME     Beaker secret mapped to SERPER_API_KEY
  --openai-secret NAME     Beaker secret mapped to OPENAI_API_KEY
  --dry-run                Ask olmo-eval to render specs without launching
  --print-only             Print shell-escaped commands without running them
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --slug) MODEL_SLUG="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --sample-seed) SAMPLE_SEED="$2"; shift 2 ;;
        --base-seed) BASE_SEED="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --cluster) CLUSTER="$2"; shift 2 ;;
        --workspace) WORKSPACE="$2"; shift 2 ;;
        --priority) PRIORITY="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --run-tag) RUN_TAG="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --sage-max-model-len) SAGE_MAX_MODEL_LEN="$2"; shift 2 ;;
        --mmlu-max-model-len) MMLU_MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpqa-max-model-len) GPQA_MAX_MODEL_LEN="$2"; shift 2 ;;
        --mmlu-max-batch-tokens) MMLU_MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
        --mmlu-gpu-memory-utilization) MMLU_GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --startup-timeout) STARTUP_TIMEOUT="$2"; shift 2 ;;
        --image) BEAKER_IMAGE="$2"; shift 2 ;;
        --s2-secret) S2_SECRET="$2"; shift 2 ;;
        --serper-secret) SERPER_SECRET="$2"; shift 2 ;;
        --openai-secret) OPENAI_SECRET="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --print-only) PRINT_ONLY=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$ONLY" in
    all|core|math|mmlu|base|gpqa|paper|sage|expertqa) ;;
    *) echo "--only must be one of: all, core, math, mmlu, base, gpqa, paper, sage, expertqa" >&2; exit 2 ;;
esac

if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "--limit must be a positive integer" >&2
    exit 2
fi

if [[ ! "$SAMPLE_SEED" =~ ^[0-9]+$ ]]; then
    echo "--sample-seed must be a non-negative integer" >&2
    exit 2
fi

if [[ -n "$BASE_SEED" && ! "$BASE_SEED" =~ ^[0-9]+$ ]]; then
    echo "--base-seed must be a non-negative integer" >&2
    exit 2
fi

BEAKER_GROUP="safe-evals-${MODEL_SLUG}-${RUN_TAG}"
SCOPE_SUFFIX=""
if [[ -n "$LIMIT" ]]; then
    SCOPE_SUFFIX="-dev${LIMIT}"
fi

common_args=(
    --yes
    --no-follow
    --model "$MODEL"
    --cluster "$CLUSTER"
    --workspace "$WORKSPACE"
    --priority "$PRIORITY"
    --timeout "$TIMEOUT"
    --gpus "$GPUS"
    --group "$BEAKER_GROUP"
)
if [[ -n "$BEAKER_IMAGE" ]]; then
    common_args+=(--image "$BEAKER_IMAGE")
fi

if [[ "$DRY_RUN" == true ]]; then
    common_args+=(--dry-run)
fi

provider_overrides=(
    --override "provider.max_model_len=${MAX_MODEL_LEN}"
    --override "provider.kwargs.startup_timeout=${STARTUP_TIMEOUT}"
    --override "provider.kwargs.language_model_only=true"
)
base_provider_overrides=("${provider_overrides[@]}")
paper_provider_overrides=("${provider_overrides[@]}")
qwen35_paper_overrides=()
qwen35_expertqa_overrides=()
qwen35_expertqa_task_overrides=()
qwen35_sage_short_overrides=()
nemotron_expertqa_overrides=()
olmo3_expertqa_harness_overrides=()
gpt_oss_harness_overrides=()
gpt_oss_sage_harness_overrides=()
math500_task_overrides=()
mmlu_task_overrides=()
base_eval_task_overrides=()
olmo3_expertqa_system_prompt="You are a web-search assistant for attributed question answering. Your first response to every question MUST be a call to serper_google_webpage_search. Do not answer from memory and do not emit the final JSON before at least one search result has been returned. After searching, fetch promising pages with serper_fetch_webpage_content, then return only the JSON object required by the user prompt. Use only those two tool names and quote only text copied from fetched page content."

# Prompt logprobs materialize a float32 token-by-vocabulary matrix. Keep the
# MMLU prefill batch small enough to bound that transient allocation, and use
# every GPU requested for the job. The longest current 5-shot MMLU request is
# 3,088 Gemma 4 tokens, so 4,096 preserves every prompt without truncation.
mmlu_provider_overrides=(
    --override "provider.max_model_len=${MMLU_MAX_MODEL_LEN}"
    --override "provider.kwargs.startup_timeout=${STARTUP_TIMEOUT}"
    --override "provider.kwargs.language_model_only=true"
    --override "provider.kwargs.tensor_parallel_size=${GPUS}"
    --override "provider.kwargs.max_num_batched_tokens=${MMLU_MAX_NUM_BATCHED_TOKENS}"
    --override "provider.kwargs.gpu_memory_utilization=${MMLU_GPU_MEMORY_UTILIZATION}"
)
mmlu_suite="mmlu"
base_eval_suite="olmobase:mcqa_stem"

# Qwen3.5 is a unified vision-language model with a hybrid GDN/MoE language
# stack. The 35B checkpoint does not leave enough useful headroom on one H100,
# so shard it across every requested GPU. These parser names match Qwen's
# official vLLM serving recipe and the model's XML tool-call template.
case "$MODEL" in
    google/gemma-4-26B-A4B-it)
        # Gemma emits its native structured reasoning and tool-call markup.
        # Without both parsers the agent harness receives literal
        # <|tool_call> text and silently scores every LitSearch query zero.
        provider_overrides+=(
            --override "provider.kwargs.reasoning_parser=gemma4"
            --override "provider.kwargs.tool_call_parser=gemma4"
        )
        paper_provider_overrides+=(
            --override "provider.kwargs.reasoning_parser=gemma4"
            --override "provider.kwargs.tool_call_parser=gemma4"
        )
        # Gemma 4 frequently uses more than the task's legacy 1,024-token
        # budget to finish a derivation. The previous run hit that ceiling on
        # 235/500 examples, so retain deterministic decoding but allow the
        # answer to finish before MinervaMathScorer extracts its boxed result.
        math500_task_overrides=(
            --override "max_tokens=${GEMMA4_MATH_MAX_TOKENS}"
            --override "stop_sequences="
            --override "temperature=0"
        )
        mmlu_suite="mmlu:chat"
        base_eval_suite="olmobase:mcqa_stem:chat"
        mmlu_provider_overrides[1]="provider.max_model_len=8192"
        mmlu_provider_overrides+=(--override "provider.kwargs.reasoning_parser=gemma4")
        ;;
    Qwen/Qwen3.5-*)
        qwen35_provider_overrides=(
            --override "provider.kwargs.tensor_parallel_size=${GPUS}"
            --override "provider.kwargs.gdn_prefill_backend=triton"
            --override "provider.kwargs.reasoning_parser=qwen3"
            --override "provider.kwargs.tool_call_parser=qwen3_coder"
        )
        provider_overrides+=("${qwen35_provider_overrides[@]}")
        # Preserve raw <think>...</think> content for the paper-search agent.
        # The agent otherwise treats a reasoning-only final response as empty;
        # SAGE's scorer already strips thinking regions before title matching.
        paper_provider_overrides+=(
            --override "provider.kwargs.tensor_parallel_size=${GPUS}"
            --override "provider.kwargs.gdn_prefill_backend=triton"
            --override "provider.kwargs.tool_call_parser=qwen3_coder"
        )
        mmlu_provider_overrides+=(
            --override "provider.kwargs.gdn_prefill_backend=triton"
            --override "provider.kwargs.reasoning_parser=qwen3"
            --override "provider.kwargs.tool_call_parser=qwen3_coder"
        )
        # Qwen's SAGE validation outputs use thinking mode. Give short-form
        # identification enough room to finish that reasoning, and extend the
        # per-request timeout accordingly. ExpertQA uses direct-response mode
        # because its forced-final path has a smaller fixed response budget.
        qwen35_paper_overrides=(
            --override "provider.kwargs.timeout=600"
        )
        qwen35_sage_short_overrides=(
            --override "max_tokens=8192"
        )
        qwen35_expertqa_overrides=(
            --override "provider.kwargs.chat_template_kwargs.enable_thinking=false"
        )
        ;;
    allenai/Olmo-3-7B-Think)
        # The checkpoint's stock template consumes legacy `functions` fields,
        # while the OpenAI-compatible agent harness sends `tools`. Enable the
        # repository's bundled OpenAI-tool-aware template and the matching
        # vLLM OLMo3 parser patch. The OLMo3 reasoning parser must first remove
        # the Think checkpoint's <think> block so the tool parser receives the
        # following <function_calls> block at the start of content. Without
        # this profile every agentic canary has zero parsed tool turns.
        olmo3_tool_overrides=(
            --override "provider.kwargs.tool_call_parser=olmo3"
            --override "provider.kwargs.reasoning_parser=olmo3"
            --override "provider.kwargs.patch_olmo3_tool_parser=true"
        )
        provider_overrides+=("${olmo3_tool_overrides[@]}")
        paper_provider_overrides+=(
            "${olmo3_tool_overrides[@]}"
            --override "provider.max_model_len=65536"
        )
        olmo3_expertqa_harness_overrides=(
            --override "tool_choice=required"
            --override "system_prompt=${olmo3_expertqa_system_prompt}"
        )
        ;;
    allenai/Olmo-3-7B-Instruct)
        # Use the same bundled OpenAI-tool-aware template as OLMo Think. A
        # 50-item SAGE run with the stock legacy template eventually leaked an
        # unparsed <function_calls> block, and ExpertQA made no tool calls.
        olmo3_tool_overrides=(
            --override "provider.kwargs.tool_call_parser=olmo3"
            --override "provider.kwargs.patch_olmo3_tool_parser=true"
        )
        provider_overrides+=("${olmo3_tool_overrides[@]}")
        paper_provider_overrides+=("${olmo3_tool_overrides[@]}")
        olmo3_expertqa_harness_overrides=(
            --override "tool_choice=required"
            --override "system_prompt=${olmo3_expertqa_system_prompt}"
        )
        ;;
    nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
        # NVIDIA's stock template enables reasoning by default, but the
        # checkpoint's nano_v3 reasoning parser is a separate plugin that is
        # not present in the pinned serving environment. ExpertQA uses the
        # model-supported direct mode and its native Qwen3-Coder tool format;
        # the compatible string-space OLMo3 parser strips any residual
        # <think> block before the strict structured final answer is scored.
        nemotron_expertqa_overrides=(
            --override "provider.kwargs.tool_call_parser=qwen3_coder"
            --override "provider.kwargs.reasoning_parser=olmo3"
            --override "provider.kwargs.chat_template_kwargs.enable_thinking=false"
        )
        ;;
esac

# GPT-OSS uses Harmony channels and can issue tool calls from its reasoning.
# Route the agent loop through vLLM's Harmony-native Responses implementation,
# which preserves reasoning output across tool turns. The Chat Completions
# parsers remain configured for direct provider calls outside the agent loop.
case "$MODEL" in
    openai/gpt-oss-20b)
        # GPT-OSS requires its Harmony chat template even for completion-style
        # tasks. Preserve MATH-500's rendered four-shot prompt by wrapping that
        # prompt in a user message, and remove the legacy double-newline stop
        # that can terminate Harmony reasoning before the final channel.
        math500_task_overrides=(
            --override "max_tokens=8192"
            --override "stop_sequences="
            --override "temperature=0"
        )
        mmlu_suite="mmlu:chat"
        base_eval_suite="olmobase:mcqa_stem:chat"
        # The medium-effort canaries exhausted both 4K and 8K completion
        # budgets on simple MMLU items. The paired low-effort seed-42 canary
        # completed 285/285 with zero caps or empty outputs and improved from
        # 0.8628 to 0.8805. Keep that validated GPT-OSS default while retaining
        # enough context for the unchanged 5-shot prompt plus an 8K completion.
        # Attach max_tokens to every expanded subject task below.
        mmlu_provider_overrides[1]="provider.max_model_len=16384"
        mmlu_task_overrides=(--override "max_tokens=8192")
        mmlu_provider_overrides+=(
            --override "provider.kwargs.reasoning_parser=openai_gptoss"
            --override "provider.kwargs.chat_template_kwargs.reasoning_effort=low"
        )
        # OpenAIResponsesCompactionSession defaults to the hosted gpt-4.1
        # responses.compact endpoint after ten response items. These local
        # vLLM evals should retain their native full history instead of making
        # an external OpenAI request (or failing when no OpenAI key is mounted).
        gpt_oss_harness_overrides=(
            --override "scaffold_kwargs.enable_compaction=false"
        )
        gpt_oss_sage_system_prompt="You are a research assistant that searches for evidence before answering. Only these tool names exist: semantic_scholar_snippet_search, serper_google_webpage_search, and browse_webpage. Use those names verbatim and never invent, abbreviate, or substitute a tool name. Put search queries and URLs in the tool arguments, never in the tool name. Use semantic_scholar_snippet_search for academic papers, serper_google_webpage_search for general web results, and browse_webpage to read a result URL. After gathering enough evidence, answer the user directly and stop calling tools."
        gpt_oss_sage_harness_overrides=(
            --override "system_prompt=${gpt_oss_sage_system_prompt}"
        )
        # vLLM 0.19.1 classifies completed functions.* calls from the analysis
        # channel as hosted MCP calls, so the Agents SDK does not execute them.
        # Backport the upstream channel-independent condition while retaining
        # this image's Torch-2.10-compatible vLLM build.
        provider_overrides+=(
            --override "provider.kwargs.completion_use_chat=true"
            --override "provider.kwargs.agent_api=responses"
            --override "provider.kwargs.patch_gpt_oss_responses_parser=true"
            --override "provider.kwargs.tool_call_parser=openai"
            --override "provider.kwargs.reasoning_parser=openai_gptoss"
        )
        paper_provider_overrides+=(
            --override "provider.kwargs.completion_use_chat=true"
            --override "provider.kwargs.agent_api=responses"
            --override "provider.kwargs.patch_gpt_oss_responses_parser=true"
            --override "provider.kwargs.tool_call_parser=openai"
            --override "provider.kwargs.reasoning_parser=openai_gptoss"
        )
        ;;
esac

paper_provider_overrides+=("${qwen35_paper_overrides[@]}")
sage_provider_overrides=("${paper_provider_overrides[@]}")
# The first paper-provider override is always max_model_len. Replace its value
# rather than appending a conflicting second override to the generated command.
sage_provider_overrides[1]="provider.max_model_len=${SAGE_MAX_MODEL_LEN}"
core_provider_overrides=("${provider_overrides[@]}")
expertqa_provider_overrides=(
    "${provider_overrides[@]}"
    "${qwen35_expertqa_overrides[@]}"
    "${nemotron_expertqa_overrides[@]}"
)

# Both Qwen checkpoints can exhaust a completion budget inside reasoning and
# then expose an empty answer after reasoning-parser extraction. Keep direct-
# response mode for Core and preserve raw assistant content by omitting the
# reasoning parser. ExpertQA uses the same raw/direct provider profile plus a
# strict JSON system prompt and an 8K completion budget. The agentic paper and
# SAGE harnesses retain thinking/raw output. MMLU uses a separate provider
# profile and, for chat-native models, an opt-in generative chat suite.
if [[ "$MODEL" == "Qwen/Qwen3.5-9B" || "$MODEL" == "Qwen/Qwen3.5-35B-A3B" ]]; then
    qwen35_direct_provider_overrides=(
        "${base_provider_overrides[@]}"
        --override "provider.kwargs.tensor_parallel_size=${GPUS}"
        --override "provider.kwargs.gdn_prefill_backend=triton"
        --override "provider.kwargs.tool_call_parser=qwen3_coder"
        --override "provider.kwargs.chat_template_kwargs.enable_thinking=false"
    )
    core_provider_overrides=("${qwen35_direct_provider_overrides[@]}")
    qwen35_expertqa_system_prompt="You are a web-search assistant for attributed question answering. Only these tool names exist: serper_google_webpage_search and serper_fetch_webpage_content. Use those names verbatim; never call serper_search, serper_search_webpage, or any other alias. Search first, then fetch promising pages. After gathering enough evidence, return only the JSON object required by the user prompt. Do not output analysis, planning text, Markdown fences, <think>, or </think>. Quote only text copied from fetched page content and do not invent citations."
    expertqa_provider_overrides=(
        "${qwen35_direct_provider_overrides[@]}"
        --override "system_prompt=${qwen35_expertqa_system_prompt}"
    )
    qwen35_expertqa_task_overrides=(
        --override "max_tokens=8192"
    )
fi

math_provider_overrides=("${core_provider_overrides[@]}")
if [[ "$MODEL" == "google/gemma-4-26B-A4B-it" ]]; then
    math_provider_overrides+=(--override "provider.kwargs.completion_use_chat=true")
fi

# GPQA's :mc variant issues four raw log-likelihood continuations per question.
# Keep this profile free of chat templates, reasoning parsers, and tool parsers:
# the canary is intended to test the same base-style protocol on every model.
gpqa_provider_overrides=(
    --override "provider.max_model_len=${GPQA_MAX_MODEL_LEN}"
    --override "provider.kwargs.startup_timeout=${STARTUP_TIMEOUT}"
    --override "provider.kwargs.language_model_only=true"
    --override "provider.kwargs.tensor_parallel_size=${GPUS}"
)
base_eval_provider_overrides=(
    --override "provider.max_model_len=${GPQA_MAX_MODEL_LEN}"
    --override "provider.kwargs.startup_timeout=${STARTUP_TIMEOUT}"
    --override "provider.kwargs.language_model_only=true"
    --override "provider.kwargs.tensor_parallel_size=${GPUS}"
)
if [[ "$MODEL" == "Qwen/Qwen3.5-9B" || "$MODEL" == "Qwen/Qwen3.5-35B-A3B" ]]; then
    gpqa_provider_overrides+=(--override "provider.kwargs.gdn_prefill_backend=triton")
    base_eval_provider_overrides+=(--override "provider.kwargs.gdn_prefill_backend=triton")
fi
case "$MODEL" in
    google/gemma-4-26B-A4B-it)
        base_eval_provider_overrides+=(
            --override "provider.kwargs.reasoning_parser=gemma4"
        )
        ;;
    openai/gpt-oss-20b)
        base_eval_provider_overrides[1]="provider.max_model_len=16384"
        base_eval_provider_overrides+=(
            --override "provider.kwargs.reasoning_parser=openai_gptoss"
            --override "provider.kwargs.chat_template_kwargs.reasoning_effort=low"
        )
        base_eval_task_overrides=(--override "max_tokens=8192")
        ;;
esac

core_tasks=(--task litsearch_rerank)
base_eval_tasks=(--task "$base_eval_suite")
gpqa_tasks=(--task gpqa_diamond:mc)
paper_tasks=(--task litsearch)
sage_tasks=(--task sage_open_ended)
expertqa_tasks=(--task expertqa)
mmlu_tasks=(--task "$mmlu_suite")
if [[ -n "$LIMIT" ]]; then
    core_tasks+=(--override "limit=${LIMIT}" --override "seed=${SAMPLE_SEED}")
    gpqa_tasks+=(--override "limit=${LIMIT}" --override "seed=${SAMPLE_SEED}")
    paper_tasks+=(--override "limit=${LIMIT}" --override "seed=${SAMPLE_SEED}")
    sage_tasks+=(--override "limit=${LIMIT}" --override "seed=${SAMPLE_SEED}")
    expertqa_tasks+=(--override "limit=${LIMIT}" --override "seed=${SAMPLE_SEED}")
fi
if [[ ("$ONLY" == all || "$ONLY" == mmlu) && (-n "$LIMIT" || ${#mmlu_task_overrides[@]} -gt 0) ]]; then
    # Overrides attached to a suite name are not inherited by its expanded
    # leaf tasks. Attach model-specific and sampling overrides to every MMLU
    # subject while retaining the suite spec so aggregation metadata survives.
    if ! mmlu_expanded_tasks="$(
        uv run python -c 'import sys; import olmo_eval.evals; from olmo_eval.evals.suites import get_suite; print("\n".join(get_suite(sys.argv[1]).expand()))' "$mmlu_suite"
    )"; then
        echo "Failed to expand the MMLU suite for task overrides." >&2
        exit 1
    fi
    while IFS= read -r mmlu_task; do
        [[ -n "$mmlu_task" ]] || continue
        mmlu_tasks+=(--task "$mmlu_task")
        if [[ -n "$LIMIT" ]]; then
            mmlu_tasks+=(
                --override "limit=${LIMIT}"
                --override "seed=${SAMPLE_SEED}"
            )
        fi
        mmlu_tasks+=("${mmlu_task_overrides[@]}")
    done <<<"$mmlu_expanded_tasks"
fi
if [[ "$ONLY" == base && (-n "$LIMIT" || -n "$BASE_SEED" || ${#base_eval_task_overrides[@]} -gt 0) ]]; then
    # As with MMLU above, suite-level overrides do not reach expanded tasks.
    # Retain the suite for aggregation metadata and attach overrides to every
    # leaf task explicitly.
    if ! base_eval_expanded_tasks="$(
        uv run python -c 'import sys; import olmo_eval.evals; from olmo_eval.evals.suites import get_suite; print("\n".join(get_suite(sys.argv[1]).expand()))' "$base_eval_suite"
    )"; then
        echo "Failed to expand the base-eval suite for task overrides." >&2
        exit 1
    fi
    while IFS= read -r base_eval_task; do
        [[ -n "$base_eval_task" ]] || continue
        base_eval_tasks+=(--task "$base_eval_task")
        if [[ -n "$LIMIT" ]]; then
            base_eval_tasks+=(
                --override "limit=${LIMIT}"
                --override "seed=${SAMPLE_SEED}"
            )
        elif [[ -n "$BASE_SEED" ]]; then
            base_eval_tasks+=(
                --override "seed=${BASE_SEED}"
                --override "fewshot_seed=${BASE_SEED}"
            )
        fi
        base_eval_tasks+=("${base_eval_task_overrides[@]}")
    done <<<"$base_eval_expanded_tasks"
fi
expertqa_tasks+=(
    "${qwen35_expertqa_task_overrides[@]}"
)
core_tasks+=(--task ifeval_ood)
if [[ -n "$LIMIT" ]]; then
    core_tasks+=(--override "limit=${LIMIT}" --override "seed=${SAMPLE_SEED}")
fi
# Gemma MATH-500 must use the chat-completion compatibility path, while the
# other Core tasks retain their validated provider protocol. Preserve a copy
# of the Core task list before adding MATH so Gemma can submit two experiments.
core_no_math_tasks=("${core_tasks[@]}")
core_tasks+=(--task math500)
core_tasks+=("${math500_task_overrides[@]}")
math_tasks=(--task math500 "${math500_task_overrides[@]}")
sage_tasks+=(--task sage_short_form)
if [[ -n "$LIMIT" ]]; then
    core_tasks+=(--override "limit=${LIMIT}" --override "seed=${SAMPLE_SEED}")
    math_tasks+=(--override "limit=${LIMIT}" --override "seed=${SAMPLE_SEED}")
    sage_tasks+=(--override "limit=${LIMIT}" --override "seed=${SAMPLE_SEED}")
fi
sage_tasks+=("${qwen35_sage_short_overrides[@]}")

failures=0

run_launch() {
    local label="$1"
    shift
    local command=(
        uv run python scripts/beaker/launch_local.py beaker launch "$@"
    )

    echo
    echo "[$label]"
    printf ' %q' "${command[@]}"
    echo

    if [[ "$PRINT_ONLY" == true ]]; then
        return 0
    fi

    if ! "${command[@]}"; then
        echo "[$label] submission failed" >&2
        failures=$((failures + 1))
    fi
}

if [[ "$ONLY" == all || "$ONLY" == core ]]; then
    if [[ "$MODEL" == "google/gemma-4-26B-A4B-it" ]]; then
        run_launch core \
            "${common_args[@]}" \
            --name "${MODEL_SLUG}-safe-core${SCOPE_SUFFIX}-${RUN_TAG}" \
            "${core_no_math_tasks[@]}" \
            --harness default \
            "${core_provider_overrides[@]}"
        run_launch math \
            "${common_args[@]}" \
            --name "${MODEL_SLUG}-math500${SCOPE_SUFFIX}-${RUN_TAG}" \
            "${math_tasks[@]}" \
            --harness default \
            "${math_provider_overrides[@]}"
    else
        run_launch core \
            "${common_args[@]}" \
            --name "${MODEL_SLUG}-safe-core${SCOPE_SUFFIX}-${RUN_TAG}" \
            "${core_tasks[@]}" \
            --harness default \
            "${core_provider_overrides[@]}"
    fi
fi

if [[ "$ONLY" == math ]]; then
    run_launch math \
        "${common_args[@]}" \
        --name "${MODEL_SLUG}-math500${SCOPE_SUFFIX}-${RUN_TAG}" \
        "${math_tasks[@]}" \
        --harness default \
        "${math_provider_overrides[@]}"
fi

if [[ "$ONLY" == all || "$ONLY" == mmlu ]]; then
    run_launch mmlu \
        "${common_args[@]}" \
        --name "${MODEL_SLUG}-mmlu${SCOPE_SUFFIX}-${RUN_TAG}" \
        "${mmlu_tasks[@]}" \
        --harness default \
        "${mmlu_provider_overrides[@]}"
fi

if [[ "$ONLY" == base ]]; then
    run_launch base \
        "${common_args[@]}" \
        --name "${MODEL_SLUG}-base-mcqa-stem-${RUN_TAG}" \
        "${base_eval_tasks[@]}" \
        --harness default \
        "${base_eval_provider_overrides[@]}"
fi

if [[ "$ONLY" == gpqa ]]; then
    run_launch gpqa \
        "${common_args[@]}" \
        --name "${MODEL_SLUG}-gpqa-diamond${SCOPE_SUFFIX}-${RUN_TAG}" \
        "${gpqa_tasks[@]}" \
        --harness default \
        "${gpqa_provider_overrides[@]}"
fi

if [[ "$ONLY" == all || "$ONLY" == paper ]]; then
    run_launch paper \
        "${common_args[@]}" \
        --name "${MODEL_SLUG}-paper-agent${SCOPE_SUFFIX}-${RUN_TAG}" \
        "${paper_tasks[@]}" \
        --harness paper_search_agent \
        "${gpt_oss_harness_overrides[@]}" \
        "${paper_provider_overrides[@]}" \
        --secret-env "${S2_SECRET}:S2_API_KEY"
fi

if [[ "$ONLY" == all || "$ONLY" == sage ]]; then
    run_launch sage \
        "${common_args[@]}" \
        --name "${MODEL_SLUG}-sage-agent${SCOPE_SUFFIX}-${RUN_TAG}" \
        "${sage_tasks[@]}" \
        --harness dr_tulu_crawl4ai \
        "${gpt_oss_harness_overrides[@]}" \
        "${gpt_oss_sage_harness_overrides[@]}" \
        "${sage_provider_overrides[@]}" \
        --secret-env "${S2_SECRET}:S2_API_KEY" \
        --secret-env "${SERPER_SECRET}:SERPER_API_KEY"
fi

if [[ "$ONLY" == all || "$ONLY" == expertqa ]]; then
    run_launch expertqa \
        "${common_args[@]}" \
        --name "${MODEL_SLUG}-expertqa-agent${SCOPE_SUFFIX}-${RUN_TAG}" \
        "${expertqa_tasks[@]}" \
        --harness web_search_agent \
        "${olmo3_expertqa_harness_overrides[@]}" \
        "${gpt_oss_harness_overrides[@]}" \
        "${expertqa_provider_overrides[@]}" \
        --secret-env "${SERPER_SECRET}:SERPER_API_KEY" \
        --secret-env "${OPENAI_SECRET}:OPENAI_API_KEY"
fi

if [[ "$failures" -ne 0 ]]; then
    echo "${failures} submission(s) failed" >&2
    exit 1
fi

echo
if [[ "$PRINT_ONLY" == true ]]; then
    echo "Printed commands for Beaker group: ${BEAKER_GROUP}"
elif [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete for Beaker group: ${BEAKER_GROUP}"
else
    echo "Submitted Beaker group: ${BEAKER_GROUP}"
fi
