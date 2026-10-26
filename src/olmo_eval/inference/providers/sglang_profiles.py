"""Qualified SGLang serving defaults; resolved on the assigned GPU, once at startup."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

# Geometry from the measured BF16 Olmo 3.5 TP1 profile (open-instruct PR #2076).
MODEL_SHAPE = {
    "model_type": "olmo3moe",
    "hidden_size": 1024,
    "num_hidden_layers": 16,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "attention_hidden_size": 1024,
    "vocab_size": 100278,
    "tie_word_embeddings": False,
    "n_routed_experts": 512,
    "num_experts_per_tok": 16,
    "moe_intermediate_size": 1024,
    "latent_moe_dim": 512,
    "dense_layers_indices": [0],
    "dense_mlp_intermediate_size": 8192,
    "dense_layers_use_shared_expert": True,
    "shared_expert_intermediate_size": 1024,
    "linear_num_key_heads": 8,
    "linear_num_value_heads": 8,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 256,
    "linear_conv_kernel_dim": 4,
    "number_of_conv_states": 3,
    "layer_types": ["full_attention" if i in (7, 15) else "linear_attention" for i in range(16)],
}


def resolve_profile(
    selector: str,
    model: dict[str, Any],
    options: dict[str, Any],
    devices: list[tuple[str, float, float]],
) -> dict[str, Any]:
    """Pure resolver. Explicit engine options win; unknown configurations keep defaults."""
    if selector not in ("auto", "generic"):
        raise ValueError("SGLang profile must be 'auto' or 'generic'")
    report: dict[str, Any] = {
        "requested": selector,
        "selected": "generic",
        "reason": "Explicit generic profile",
        "defaults": {},
        "devices": devices,
    }

    def fallback(reason: str) -> dict[str, Any]:
        report["reason"] = reason
        return report

    if selector == "generic":
        return report
    shape = {k: model.get(k) for k in MODEL_SHAPE}
    report["model_shape_sha256"] = hashlib.sha256(
        json.dumps(shape, sort_keys=True).encode()
    ).hexdigest()
    if shape != MODEL_SHAPE or model.get("architectures") != ["Olmo3MoeForCausalLM"]:
        return fallback("No qualified profile for this model geometry/architecture")
    if model.get("dtype", model.get("torch_dtype")) != "bfloat16" or options.get(
        "dtype", "auto"
    ) not in ("auto", "bfloat16"):
        return fallback("Profiles require BF16 weights and inference")
    if model.get("quantization_config") or options.get("quantization"):
        return fallback("Quantized inference is not qualified")
    if any(options.get(k, 1) != 1 for k in ("tp_size", "dp_size", "ep_size", "pp_size", "nnodes")):
        return fallback("Profiles require single-GPU TP1 inference")
    if options.get("enable_multimodal") or model.get("vision_config"):
        return fallback("Vision profiles are not qualified")
    context = options.get("context_length", model.get("max_position_embeddings", 0))
    if not isinstance(context, int) or not 0 < context <= 34816:
        return fallback(
            "Profiles cover context limits up to 34,816 tokens; set max_model_len explicitly"
        )
    if len(devices) != 1:
        return fallback("Could not identify the engine GPU")
    name, total, free = devices[0]
    gpu = (
        "b300"
        if "b300" in name.lower() and total >= 270000
        else ("h100" if "h100" in name.lower() and total >= 80000 else None)
    )
    if gpu is None or "mig" in name.lower():
        return fallback("No qualified profile for this GPU/memory capacity")
    if free < total * 0.95:
        return fallback("Profiles require a dedicated GPU with at least 95% memory free at startup")
    cache_off = options.get("disable_radix_cache", gpu != "b300")
    running = options.get("max_running_requests", 512 if gpu == "b300" or cache_off else 320)
    if not isinstance(running, int) or isinstance(running, bool) or running <= 0:
        raise ValueError("max_running_requests must be a positive integer")
    if running > 512:
        return fallback("Requested admission exceeds the qualified profile range")
    kv_contexts = running if gpu == "b300" else min(running, 192 if cache_off else 128)
    defaults = {
        "disable_radix_cache": cache_off,
        "max_running_requests": running,
        "cuda_graph_backend_decode": "full",
        "cuda_graph_max_bs_decode": running,
        "cuda_graph_backend_prefill": "disabled",
        "max_total_tokens": kv_contexts * 34816,
        "max_mamba_cache_size": running * ((2 if gpu == "b300" else 1) if cache_off else 6),
        "mem_fraction_static": 0.7 if gpu == "b300" else 0.9,
        "chunked_prefill_size": 8192,
        "attention_backend": "triton",
        "sampling_backend": "flashinfer",
    }
    if not cache_off:
        defaults["mamba_radix_cache_strategy"] = "extra_buffer"
    report.update(
        selected=f"olmo35-{gpu}-v1",
        reason="Matched BF16 geometry, dedicated GPU and context range",
        defaults={k: v for k, v in defaults.items() if k not in options},
    )
    return report


def prepare_profile(
    selector: str, options: dict[str, Any], registration: str | None
) -> dict[str, Any]:
    """Read metadata without executing checkpoint code or initializing CUDA."""
    if selector not in ("auto", "generic"):
        raise ValueError("SGLang profile must be 'auto' or 'generic'")
    if selector == "generic" or registration != "olmo_sglang:register":
        return {
            "requested": selector,
            "selected": "generic",
            "defaults": {},
            "reason": "Generic profile or no Olmo model registration",
        }
    model_path = Path(options["model_path"]) / "config.json"
    try:
        if not model_path.is_file():
            from huggingface_hub import hf_hub_download

            model_path = Path(
                hf_hub_download(
                    options["model_path"], "config.json", revision=options.get("revision")
                )
            )
        model = json.loads(model_path.read_text())
        if not isinstance(model, dict):
            raise ValueError("config.json must be an object")
        mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        index = options.get("base_gpu_id", 0)
        device = mask.split(",")[index] if mask else str(index)
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
                f"--id={device}",
            ],
            text=True,
            timeout=15,
        )
        devices = [
            (name.strip(), float(total), float(free))
            for name, total, free in (line.rsplit(",", 2) for line in output.splitlines())
        ]
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        return {
            "requested": selector,
            "selected": "generic",
            "defaults": {},
            "reason": f"Profile metadata/hardware unavailable: {error}",
        }
    return resolve_profile(selector, model, options, devices)
