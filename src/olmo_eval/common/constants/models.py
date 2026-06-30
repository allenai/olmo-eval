from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.harness.config import ProviderConfig


# =============================================================================
# Model Presets for Evaluation
# =============================================================================


def get_model_presets() -> dict[str, ProviderConfig]:
    """Get model presets dictionary.

    Returns a dictionary mapping preset names to ProviderConfig instances.
    """
    from olmo_eval.common.types import ProviderKind
    from olmo_eval.harness.config import ProviderConfig

    return {
        "llama3.1-8b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="meta-llama/Meta-Llama-3.1-8B",
        ),
        "llama3.1-8b-instruct": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="meta-llama/Llama-3.1-8B-Instruct",
        ),
        "llama3.1-70b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="meta-llama/Meta-Llama-3.1-70B",
        ),
        "llama3.1-70b-instruct": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="meta-llama/Llama-3.1-70B-Instruct",
        ),
        "olmo-3-1025-7b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="allenai/Olmo-3-1025-7B",
            trust_remote_code=True,
            max_model_len=4096,
            revision="stage2-step47684",
            kwargs={"gpu_memory_utilization": 0.7, "add_bos_token": False},
        ),
        # sftlab data-ablation checkpoints (OLMo-3-7B SFT-from-base, converted to HF).
        # Same ProviderConfig shape as the bk/olmo7b-sft-* presets — crucially
        # max_model_len=4096, which neutralizes the checkpoint's YaRN rope_scaling
        # (rope_type=yarn, factor 8, attention_factor ~1.21) on the short few-shot
        # science:nojudge prompts; without the cap vLLM applies YaRN/mscale to all
        # lengths and the model scores near-random. The model path is per-run, so it
        # comes from $SFTLAB_EVAL_MODEL (sftlab sets it on the eval job) rather than a
        # hardcoded path — one preset serves every sftlab arm.
        "sftlab-olmo3-7b-sft": ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model=os.environ.get("SFTLAB_EVAL_MODEL", ""),
            # trust_remote_code is the load-bearing fix: without it the bare-path
            # default eval scored MMLU ~2% (near-random); with it, ~57%. max_model_len
            # is env-tunable but defaults to the model's full 32768 — a diagnostic
            # (mmlu/gsm8k at 4096 vs 32768) gave identical scores, so the cap is NOT
            # needed for correctness, and 32768 is required for the long-CoT science
            # tasks (aime/math500) that request up to 32768 output tokens.
            trust_remote_code=True,
            max_model_len=int(os.environ.get("SFTLAB_EVAL_MAX_LEN", "32768")),
            kwargs={"gpu_memory_utilization": 0.7},
        ),
        # The released, post-trained OLMo-3-7B SFT checkpoint (general instruction tuning,
        # NOT our deep-research data). Native olmo3 (Olmo3ForCausalLM, YaRN rope). Used as an
        # ALTERNATE baseline anchor, evaluated on the STOCK pythonic harness (-H dr_tulu) — its
        # native tool dialect — NOT oi_contract. Pairs with the official olmo3 vLLM tool parser
        # (auto-inferred from the model name) for the openai_agents scaffold.
        "olmo-3-7b-instruct-sft": ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="allenai/Olmo-3-7B-Instruct-SFT",
            trust_remote_code=True,
            max_model_len=32768,
            kwargs={"gpu_memory_utilization": 0.7},
        ),
        "bk/olmo7b-sft-general-within-step17307": ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="/weka/oe-training-default/ai2-llm/checkpoints/baileyk/olmo-sft/olmo-7b-base-general-within-mix/step17307-hf",
            trust_remote_code=True,
            max_model_len=4096,
            kwargs={"gpu_memory_utilization": 0.7},
        ),
        "bk/olmo7b-sft-coding-heavy-step17075": ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="/weka/oe-training-default/ai2-llm/checkpoints/baileyk/olmo-sft/olmo-7b-base-coding-heavy-mix/step17075-hf",
            trust_remote_code=True,
            max_model_len=4096,
            kwargs={"gpu_memory_utilization": 0.7},
        ),
        "bk/olmo7b-sft-75pct-step17312": ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="/weka/oe-training-default/ai2-llm/checkpoints/baileyk/olmo-sft/olmo-7b-base-75pct-mix/step17312-hf",
            trust_remote_code=True,
            max_model_len=4096,
            kwargs={"gpu_memory_utilization": 0.7},
        ),
        "olmo-2-7b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="allenai/OLMo-2-1124-7B",
            trust_remote_code=True,
        ),
        "olmo-2-13b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="allenai/OLMo-2-1124-13B",
            trust_remote_code=True,
        ),
        "qwen2.5-7b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="Qwen/Qwen2.5-7B",
        ),
        "qwen3-coder-30b": ProviderConfig(
            kind=ProviderKind.VLLM_SERVER,
            model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            kwargs={"enable_expert_parallel": True, "tool_call_parser": "qwen3_coder"},
        ),
        # TODO(undfined): Leaving this here as reference. DeepGEMM is more involved
        # and we can add it to base image and toggle it on when needed.
        # "qwen3-coder-30b-fp8": ProviderConfig(
        #     kind=ProviderKind.VLLM_SERVER,
        #     model="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        #     kwargs={"enable_expert_parallel": True, "tool_call_parser": "qwen3_coder"},
        #     dependencies=(
        #         "git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation",
        #     ),
        # ),
        "deepseek-r1-distill-8b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            max_model_len=32768,
        ),
        "mistral-7b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="mistralai/Mistral-7B-v0.3",
        ),
        "o3-mini-2025-01-31-medium": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="openai/o3-mini-2025-01-31",
            api_base="https://api.openai.com/v1",
            required_secrets=("OPENAI_API_KEY",),
            kwargs={"reasoning_effort": "medium", "drop_params": True},
        ),
        "gpt-4o": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="gpt-4o",
            api_base="https://api.openai.com/v1",
            required_secrets=("OPENAI_API_KEY",),
        ),
        "gpt-4o-mini": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="openai/gpt-4o-mini",
            api_base="https://api.openai.com/v1",
            required_secrets=("OPENAI_API_KEY",),
        ),
        "gpt-4-turbo": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="openai/gpt-4-turbo",
            api_base="https://api.openai.com/v1",
            required_secrets=("OPENAI_API_KEY",),
        ),
        "claude-3-opus": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="anthropic/claude-3-opus-20240229",
            api_base="https://api.anthropic.com",
            required_secrets=("ANTHROPIC_API_KEY",),
        ),
        "claude-3-sonnet": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="anthropic/claude-3-sonnet-20240229",
            api_base="https://api.anthropic.com",
            required_secrets=("ANTHROPIC_API_KEY",),
        ),
        "mock": ProviderConfig(
            kind=ProviderKind.MOCK,
            model="mock",
        ),
        # Ai2 deployed models on Litellm Proxy
        "cirrascale-olmo-3-7b-instruct": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="litellm_proxy/openai/Olmo-3-7B-Instruct",
            api_base="https://ai2-model-hub.allen.ai",
            required_secrets=("LITELLM_PROXY_API_KEY",),
        ),
        "modal-olmo-3-7b-instruct": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="litellm_proxy/openai/ai2-release-partners/Olmo-3-7B-Instruct",
            api_base="https://ai2-model-hub.allen.ai",
            required_secrets=("LITELLM_PROXY_API_KEY",),
        ),
    }
