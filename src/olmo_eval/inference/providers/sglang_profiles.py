"""Qualified SGLang serving defaults; resolved on the assigned GPU, once at startup."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olmo_eval.inference.model_profiles import MODEL_SHAPES


def _check_profile_conflicts(
    profile: str, defaults: dict[str, Any], options: dict[str, Any]
) -> None:
    """Profiles own their settings, including equivalent SGLang graph controls."""
    explicit = {key: value for key, value in options.items() if key in defaults}
    conflicts = []

    def check(key: str, value: Any, source: str) -> None:
        if key in defaults and value != defaults[key]:
            conflicts.append(f"{source}={value!r} (profile requires {defaults[key]!r})")

    for key, value in explicit.items():
        check(key, value, key)
    for switch, phases in (
        ("disable_cuda_graph", ("decode", "prefill")),
        ("disable_decode_cuda_graph", ("decode",)),
        ("disable_prefill_cuda_graph", ("prefill",)),
    ):
        if options.get(switch):
            for phase in phases:
                check(f"cuda_graph_backend_{phase}", "disabled", switch)
    graph = options.get("cuda_graph_config") or {}
    if not isinstance(graph, dict):
        raise ValueError(
            f"Profile {profile!r} requires a dictionary cuda_graph_config to check conflicts; "
            "use profile=generic for manual configuration"
        )
    for phase in ("decode", "prefill"):
        fields = graph.get(phase, {})
        if not isinstance(fields, dict):
            raise ValueError(f"cuda_graph_config.{phase} must be a dictionary")
        for key in ("backend", "max_bs"):
            if key in fields:
                flat = f"cuda_graph_{'backend' if key == 'backend' else 'max_bs'}_{phase}"
                check(flat, fields[key], f"cuda_graph_config.{phase}.{key}")
        # SGLang derives max_bs from an explicit bucket list, overriding max_bs.
        for source, buckets in (
            (f"cuda_graph_bs_{phase}", options.get(f"cuda_graph_bs_{phase}")),
            (f"cuda_graph_config.{phase}.bs", fields.get("bs")),
        ):
            if buckets is not None and f"cuda_graph_max_bs_{phase}" in defaults:
                if (
                    not isinstance(buckets, (list, tuple))
                    or not buckets
                    or any(type(size) is not int or size <= 0 for size in buckets)
                ):
                    raise ValueError(f"{source} must contain positive integer batch sizes")
                check(f"cuda_graph_max_bs_{phase}", max(buckets), source)
    if conflicts:
        raise ValueError(
            f"Settings conflict with SGLang profile {profile!r}: {'; '.join(conflicts)}. "
            "Remove the conflicting settings or set provider.kwargs.profile=generic "
            "for manual configuration."
        )


def _olmo35_defaults(
    model: dict[str, Any],
    options: dict[str, Any],
    devices: list[tuple[str, float, float]],
) -> dict[str, Any]:
    """Measured SGLang settings for the 12.5B shape; geometry is checked by the resolver."""
    report: dict[str, Any] = {"selected": "generic", "defaults": {}}

    def fallback(reason: str) -> dict[str, Any]:
        report["reason"] = reason
        return report

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
    gpu = next(
        (
            family
            for family, minimum in (("b300", 270000), ("h100", 80000), ("a100", 80000))
            if family in name.lower() and total >= minimum
        ),
        None,
    )
    if gpu is None or "mig" in name.lower():
        return fallback("No qualified profile for this GPU/memory capacity")
    if free < total * 0.95:
        return fallback("Profiles require a dedicated GPU with at least 95% memory free at startup")
    cache_off = gpu != "b300"
    running = 512
    kv_contexts = running if gpu == "b300" else 192
    defaults = {
        "disable_radix_cache": cache_off,
        "max_running_requests": running,
        "cuda_graph_backend_decode": "full",
        "cuda_graph_max_bs_decode": running,
        "cuda_graph_backend_prefill": "disabled",
        "max_total_tokens": kv_contexts * 34816,
        "max_mamba_cache_size": running if cache_off else running * 6,
        "mem_fraction_static": 0.7 if gpu == "b300" else 0.9,
        "chunked_prefill_size": 8192,
        "attention_backend": "triton",
        "sampling_backend": "flashinfer",
    }
    if not cache_off:
        defaults["mamba_radix_cache_strategy"] = "extra_buffer"
    selected = f"olmo3.5-12.5b-sglang-{gpu}-v1"
    _check_profile_conflicts(selected, defaults, options)
    report.update(
        selected=selected,
        reason="Matched BF16 geometry, dedicated GPU and context range",
        defaults={k: v for k, v in defaults.items() if k not in options},
    )
    return report


@dataclass(frozen=True)
class EngineProfile:
    registration: str
    resolve: Callable[
        [dict[str, Any], dict[str, Any], list[tuple[str, float, float]]], dict[str, Any]
    ]


# Engine-specific qualification: adding a shared geometry does not automatically
# opt it into SGLang defaults (or qualify these settings for another engine).
ENGINE_PROFILES = {
    "olmo3.5-12.5b": EngineProfile("olmo_sglang:register", _olmo35_defaults),
}


def _validate_selector(selector: str) -> None:
    if selector not in ("auto", "generic") and selector not in ENGINE_PROFILES:
        choices = ", ".join(("auto", "generic", *sorted(ENGINE_PROFILES)))
        raise ValueError(f"Unknown SGLang profile {selector!r}; choose one of: {choices}")


def _fallback(selector: str, reason: str) -> dict[str, Any]:
    if selector not in ("auto", "generic"):
        raise ValueError(f"SGLang profile {selector!r} is not qualified: {reason}")
    return {"requested": selector, "selected": "generic", "defaults": {}, "reason": reason}


def resolve_profile(
    selector: str,
    model: dict[str, Any],
    options: dict[str, Any],
    devices: list[tuple[str, float, float]],
) -> dict[str, Any]:
    """Discover a registered shape or require an explicitly named, qualified profile."""
    _validate_selector(selector)
    if selector == "generic":
        return _fallback(selector, "Explicit generic profile")
    candidates = ENGINE_PROFILES if selector == "auto" else {selector: ENGINE_PROFILES[selector]}
    matches = [
        name
        for name in candidates
        if all(model.get(key) == value for key, value in MODEL_SHAPES[name].items())
    ]
    if not matches:
        return _fallback(selector, "No qualified profile for this model geometry/architecture")
    if len(matches) != 1:
        raise ValueError(f"Multiple SGLang profiles match {matches}; select one by name")
    name = matches[0]
    report = ENGINE_PROFILES[name].resolve(model, options, devices)
    if report["selected"] == "generic":
        report = _fallback(selector, report["reason"])
    shape = {key: model.get(key) for key in MODEL_SHAPES[name]}
    return {
        **report,
        "requested": selector,
        "model_profile": name,
        "devices": devices,
        "model_shape_sha256": hashlib.sha256(
            json.dumps(shape, sort_keys=True).encode()
        ).hexdigest(),
    }


def prepare_profile(
    selector: str, options: dict[str, Any], registration: str | None
) -> dict[str, Any]:
    """Read metadata without executing checkpoint code or initializing CUDA."""
    _validate_selector(selector)
    if selector == "generic":
        return _fallback(selector, "Explicit generic profile")
    candidates = ENGINE_PROFILES if selector == "auto" else {selector: ENGINE_PROFILES[selector]}
    qualified = {
        name for name, profile in candidates.items() if profile.registration == registration
    }
    if not qualified:
        return _fallback(selector, "No qualified profile for this model registration")
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
        return _fallback(selector, f"Profile metadata/hardware unavailable: {error}")
    report = resolve_profile(selector, model, options, devices)
    if report.get("model_profile") not in qualified and report["selected"] != "generic":
        return _fallback(selector, "No qualified profile for this model registration")
    return report
