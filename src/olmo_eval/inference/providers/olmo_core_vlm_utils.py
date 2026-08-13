"""Utilities for the multimodal OLMo-core inference provider.

Handles the three on-disk flavors of OLMo-core ``MultimodalLM`` weights:

1. ``olmo_core_dcp`` — a raw OLMo-core trainer checkpoint directory
   (``config.json`` with a serialized ``MultimodalLMConfig`` under ``model``
   plus a torch distributed checkpoint at ``model_and_optim/``).
2. ``olmo_core_unsharded`` — a consolidated export: ``olmo_core_config.json``
   (the full OLMo-core ``ExperimentConfig``) next to a single
   ``model.safetensors`` whose tensor keys use OLMo-core naming
   (``lm.*`` / ``vision.*`` / ``connector.*``).
3. ``mm_olmo_dcp`` — a checkpoint saved by the mm_olmo (Molmo2 ``video_olmo``)
   trainer: ``config.yaml`` plus a torch distributed checkpoint at
   ``model_and_optim/`` with legacy key naming (``model.transformer.*`` with
   fused ``att_proj`` / ``ff_proj``, ``model.vision_backbone.*``). These are
   translated into the OLMo-core ``MultimodalLM`` layout at load time by
   reusing OLMo-core's HF-Molmo2 converter.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

import olmo_eval.inference.providers.olmo_core_utils as core_utils

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

CheckpointFormat = Literal["olmo_core_dcp", "olmo_core_unsharded", "mm_olmo_dcp"]

_EXPECTED_FORMATS = (
    "expected one of: a raw OLMo-core multimodal checkpoint (config.json with a "
    "MultimodalLMConfig under 'model' + model_and_optim/.metadata), a consolidated "
    "OLMo-core export (olmo_core_config.json + model.safetensors with lm.*/vision.*/"
    "connector.* keys), or an mm_olmo/Molmo2-trainer checkpoint (config.yaml + "
    "model_and_optim/.metadata with model.transformer.*/model.vision_backbone.* keys)"
)

# Old name of MultimodalLMConfig kept by earlier vision-branch checkpoints.
_LEGACY_MULTIMODAL_CLASS_NAMES = {
    "olmo_core.nn.vision.multimodal.MultimodalTransformerConfig": (
        "olmo_core.nn.vision.multimodal.MultimodalLMConfig"
    ),
}

_MULTIMODAL_CLASS_SUFFIXES = ("MultimodalLMConfig", "MultimodalTransformerConfig")

# Default tokenizer for Molmo2-family models: Qwen tokenizer + Molmo2 image
# special tokens + the Molmo2 chat template.
DEFAULT_TOKENIZER = "allenai/Molmo2-4B"

# Molmo2 image special-token *names* — resolved to IDs through the tokenizer so
# the provider works for any vocab layout that defines them.
IMAGE_SPECIAL_TOKENS = (
    "<im_start>",
    "<im_end>",
    "<im_patch>",
    "<im_col>",
    "<low_res_im_start>",
)
IMAGE_PLACEHOLDER_TOKEN = "<|image|>"
END_OF_TURN_TOKEN = "<|im_end|>"


class VLMTokenizerProtocol(core_utils.TokenizerProtocol, Protocol):
    """Tokenizer surface the multimodal provider needs on top of the text one.

    ``convert_tokens_to_ids`` maps the image special-token names above to the
    vocab IDs the provider splices into prompts.
    """

    bos_token_id: int | None

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int | None: ...

    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...


@dataclass(frozen=True)
class MultimodalCheckpointInfo:
    """Resolved facts about a multimodal checkpoint directory."""

    format: CheckpointFormat
    config: dict[str, Any]
    """The full config dict (ExperimentConfig for OLMo-core, the yaml for mm_olmo)."""
    model_config: dict[str, Any]
    """The model section (MultimodalLMConfig dict, or the mm_olmo model yaml)."""


def _config_error(checkpoint_dir: str, reason: str) -> ValueError:
    return ValueError(
        f"Invalid multimodal OLMo-core checkpoint {checkpoint_dir!r}: {reason}. "
        f"{_EXPECTED_FORMATS}."
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open() as f:
        return yaml.safe_load(f)


def _is_multimodal_model_config(model_config: Any) -> bool:
    if not isinstance(model_config, dict):
        return False
    class_name = model_config.get("_CLASS_", "")
    return isinstance(class_name, str) and class_name.endswith(_MULTIMODAL_CLASS_SUFFIXES)


def _is_mm_olmo_model_config(model_config: Any) -> bool:
    return (
        isinstance(model_config, dict)
        and "llm" in model_config
        and "vision_backbone" in model_config
    )


def is_multimodal_checkpoint(checkpoint_dir: str) -> bool:
    """Cheaply check whether a local directory holds multimodal OLMo-core weights.

    Used to route ``provider.kind=olmo_core`` requests to the multimodal
    provider. Returns ``False`` on any read error so callers can fall back to
    the text provider's own validation.
    """
    try:
        detect_checkpoint_format(checkpoint_dir)
        return True
    except Exception:
        return False


def detect_checkpoint_format(checkpoint_dir: str) -> MultimodalCheckpointInfo:
    """Identify which multimodal checkpoint flavor ``checkpoint_dir`` holds.

    :raises ValueError: if the directory matches none of the known formats.
    """
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise _config_error(checkpoint_dir, "not a local directory")

    unsharded_config = root / "olmo_core_config.json"
    if unsharded_config.is_file() and (root / "model.safetensors").is_file():
        config = _load_json(unsharded_config)
        model_config = config.get("model")
        if not (isinstance(model_config, dict) and _is_multimodal_model_config(model_config)):
            raise _config_error(
                checkpoint_dir,
                "olmo_core_config.json 'model' is not a MultimodalLMConfig",
            )
        return MultimodalCheckpointInfo(
            format="olmo_core_unsharded", config=config, model_config=model_config
        )

    has_dcp = (root / "model_and_optim" / ".metadata").is_file()
    json_config = root / "config.json"
    yaml_config = root / "config.yaml"

    if json_config.is_file() and has_dcp:
        config = _load_json(json_config)
        model_config = config.get("model")
        if isinstance(model_config, dict) and _is_multimodal_model_config(model_config):
            return MultimodalCheckpointInfo(
                format="olmo_core_dcp", config=config, model_config=model_config
            )
        raise _config_error(
            checkpoint_dir,
            "config.json 'model' is not a MultimodalLMConfig (text-only OLMo-core "
            "checkpoints should use the 'olmo_core' provider)",
        )

    if yaml_config.is_file() and has_dcp:
        config = _load_yaml(yaml_config)
        model_config = config.get("model")
        if isinstance(model_config, dict) and _is_mm_olmo_model_config(model_config):
            return MultimodalCheckpointInfo(
                format="mm_olmo_dcp", config=config, model_config=model_config
            )
        raise _config_error(
            checkpoint_dir,
            "config.yaml 'model' does not look like an mm_olmo multimodal model "
            "(missing 'llm'/'vision_backbone')",
        )

    raise _config_error(checkpoint_dir, "no recognizable config/weights layout found")


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------


def _rewrite_legacy_class_names(config: Any) -> Any:
    """Recursively map renamed ``_CLASS_`` names to their current spelling."""
    if isinstance(config, dict):
        out = {key: _rewrite_legacy_class_names(value) for key, value in config.items()}
        class_name = out.get("_CLASS_")
        if isinstance(class_name, str) and class_name in _LEGACY_MULTIMODAL_CLASS_NAMES:
            out["_CLASS_"] = _LEGACY_MULTIMODAL_CLASS_NAMES[class_name]
        return out
    if isinstance(config, list):
        return [_rewrite_legacy_class_names(value) for value in config]
    return config


def build_model_config(
    info: MultimodalCheckpointInfo,
    *,
    image_patch_token_id: int,
    attention_backend: str | None = "torch",
) -> Any:
    """Build the OLMo-core ``MultimodalLMConfig`` for a checkpoint.

    :param image_patch_token_id: The ``<im_patch>`` vocab ID resolved from the
        tokenizer; only used for the mm_olmo format (OLMo-core configs carry
        their own).
    :param attention_backend: Normalize every attention config to this backend
        (``None`` keeps the checkpoint's). Training configs often pin ``flex``
        or ``flash_2``, but eval needs the dense ``torch`` backend: it is the
        one that supports the bidirectional image-token ``or_mask`` AND the
        KV-cached decoding installed by :func:`enable_kv_cache`.
    """
    from olmo_core.nn.vision.multimodal import MultimodalLMConfig

    if info.format in ("olmo_core_dcp", "olmo_core_unsharded"):
        model_config = _rewrite_legacy_class_names(info.model_config)
        cfg = MultimodalLMConfig.from_dict(model_config)
    else:
        cfg = _mm_olmo_model_config_to_multimodal_lm_config(
            info.model_config, image_patch_token_id=image_patch_token_id
        )
    if attention_backend is not None:
        _normalize_attention_backend(cfg, attention_backend)
    return cfg


def _normalize_attention_backend(cfg: Any, backend: str) -> None:
    """Set ``backend`` on every nested attention/sequence-mixer config."""
    from olmo_core.nn.attention import AttentionBackendName

    backend_value = AttentionBackendName(backend)

    def set_backend(config: Any) -> None:
        mixer = getattr(config, "sequence_mixer", None) or getattr(config, "attention", None)
        if mixer is None and hasattr(config, "backend"):
            mixer = config
        if mixer is not None and hasattr(mixer, "backend"):
            setattr(mixer, "backend", backend_value)  # noqa: B010

    for section in (cfg.lm, cfg.vision, cfg.connector):
        apply = getattr(section, "apply", None)
        if callable(apply):
            apply(set_backend)
        else:
            set_backend(section)


def _mm_olmo_model_config_to_multimodal_lm_config(
    model_config: dict[str, Any], *, image_patch_token_id: int
) -> Any:
    """Translate an mm_olmo (``video_olmo``) model config into a ``MultimodalLMConfig``.

    Builds lightweight HF-Molmo2-config lookalikes from the mm_olmo yaml fields
    and reuses OLMo-core's ``molmo2_config_from_hf_config`` so the architecture
    dispatch (qwen3_4B / qwen3_8B / olmo3_7B) and ViT-truncation logic stay in
    one place.
    """
    from olmo_core.nn.vision.molmo2_loader import molmo2_config_from_hf_config

    llm = model_config["llm"]
    backbone = model_config["vision_backbone"]
    vit = backbone["vit"]

    text_config = types.SimpleNamespace(
        hidden_size=llm["d_model"],
        num_hidden_layers=llm["n_layers"],
        qk_norm_type=(
            llm.get("attention_layer_norm_type") if llm.get("attention_layer_norm") else None
        ),
        rope_theta=llm["rope_theta"],
        vocab_size=llm["embedding_size"],
        additional_vocab_size=llm["additional_vocab_size"],
    )
    vit_config = types.SimpleNamespace(
        image_default_input_size=tuple(vit["image_default_input_size"]),
        image_patch_size=vit["image_patch_size"],
        hidden_size=vit["image_emb_dim"],
        num_attention_heads=vit["image_num_heads"],
        num_key_value_heads=vit["image_num_key_value_heads"],
        num_hidden_layers=vit["image_num_layers"],
        head_dim=vit["image_head_dim"],
        intermediate_size=vit["image_mlp_dim"],
        hidden_act=vit["image_mlp_activations"],
        image_num_pos=vit["image_num_pos"],
        layer_norm_eps=vit["image_norm_eps"],
    )
    # mm_olmo's image_pooling_2d reuses the ViT attention dims and half the LM's
    # fused MLP width, matching the released Molmo2 adapter_config.
    adapter_config = types.SimpleNamespace(
        vit_layers=list(backbone["vit_layers"]),
        num_attention_heads=vit["image_num_heads"],
        num_key_value_heads=vit["image_num_key_value_heads"],
        head_dim=vit["image_head_dim"],
        text_hidden_size=llm["d_model"],
        pooling_attention_mask=backbone["pooling_attention_mask"],
        intermediate_size=llm["mlp_hidden_size"] // 2,
    )
    hf_like_config = types.SimpleNamespace(
        text_config=text_config,
        vit_config=vit_config,
        adapter_config=adapter_config,
        image_patch_id=image_patch_token_id,
    )
    if backbone.get("image_pooling_2d") != "attention_meanq":
        raise ValueError(
            "mm_olmo checkpoint uses unsupported image pooling "
            f"{backbone.get('image_pooling_2d')!r}; only 'attention_meanq' is supported"
        )
    if backbone.get("image_projector") != "mlp":
        raise ValueError(
            "mm_olmo checkpoint uses unsupported image projector "
            f"{backbone.get('image_projector')!r}; only 'mlp' is supported"
        )
    return molmo2_config_from_hf_config(hf_like_config)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _untie_lm_head(model: Any) -> None:
    """Give ``lm_head.w_out`` its own storage when word embeddings are tied.

    Checkpoints in the wild store *different* tensors for
    ``lm.embeddings.weight`` and ``lm.lm_head.w_out.weight`` even when the
    config says ``tie_word_embeddings`` (the extra image-token rows legitimately
    differ). Loading both into a shared parameter would let one silently
    clobber the other, so break the tie before loading.
    """
    import torch.nn as nn

    lm = model.lm
    if getattr(lm, "tie_word_embeddings", False):
        lm.lm_head.w_out.weight = nn.Parameter(lm.embeddings.weight.detach().clone())


def _ensure_mm_olmo_unpickle_shim() -> None:
    """Register a stub ``olmo.train.remote_filesystem`` module for metadata unpickling.

    mm_olmo distributed checkpoints pickle a custom ``_StorageInfo`` dataclass
    (fields ``relative_path`` / ``offset`` / ``length``, mirroring torch's) into
    ``model_and_optim/.metadata``. The mm_olmo package isn't installed in the
    eval environment, so provide a minimal class with the same module path for
    pickle to resolve. No-op when the real package (or the shim) is importable.
    """
    import importlib.util

    try:
        if importlib.util.find_spec("olmo.train.remote_filesystem") is not None:
            return
    except (ImportError, ModuleNotFoundError):
        pass
    if "olmo.train.remote_filesystem" in sys.modules:
        return

    from dataclasses import dataclass as _dataclass

    @_dataclass
    class _StorageInfo:
        relative_path: str
        offset: int
        length: int

    olmo_mod = sys.modules.setdefault("olmo", types.ModuleType("olmo"))
    train_mod = sys.modules.setdefault("olmo.train", types.ModuleType("olmo.train"))
    fs_mod = sys.modules.setdefault(
        "olmo.train.remote_filesystem", types.ModuleType("olmo.train.remote_filesystem")
    )
    setattr(olmo_mod, "train", train_mod)  # noqa: B010
    setattr(train_mod, "remote_filesystem", fs_mod)  # noqa: B010
    _StorageInfo.__module__ = "olmo.train.remote_filesystem"
    setattr(fs_mod, "_StorageInfo", _StorageInfo)  # noqa: B010


def _mm_olmo_key_to_hf_key(key: str) -> str:
    """Rename an mm_olmo trainer tensor key to released-Molmo2 HF naming.

    The mm_olmo trainer keeps attention/MLP tensors directly on the block
    (``blocks.N.att_proj``); the HF release nests them (``blocks.N.self_attn.
    att_proj`` / ``blocks.N.mlp.ff_proj``). Everything else (wte, ln_f, the
    whole vision backbone) matches HF naming already.
    """
    parts = key.split(".")
    if len(parts) >= 4 and parts[:2] == ["model", "transformer"] and parts[2] == "blocks":
        leaf = parts[4]
        if leaf in ("att_proj", "attn_out", "q_norm", "k_norm"):
            parts.insert(4, "self_attn")
        elif leaf in ("ff_proj", "ff_out"):
            parts.insert(4, "mlp")
        return ".".join(parts)
    return key


def _load_mm_olmo_state_dict(checkpoint_dir: str, model_cfg: Any) -> dict[str, torch.Tensor]:
    """Consolidate an mm_olmo distributed checkpoint into MultimodalLM keys.

    Reads the (unsharded) ``model.*`` tensors, renames them to released-Molmo2
    HF naming, synthesizes the tied ``lm_head`` when absent, and runs OLMo-core's
    ``molmo2_hf_state_dict_to_multimodal_lm`` for the fused-weight splits and
    patch-embedding permutation.
    """
    from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys
    from olmo_core.nn.vision.molmo2_loader import molmo2_hf_state_dict_to_multimodal_lm

    _ensure_mm_olmo_unpickle_shim()

    dcp_dir = str(Path(checkpoint_dir) / "model_and_optim")
    metadata = get_checkpoint_metadata(dcp_dir)
    model_keys = sorted(key for key in metadata.state_dict_metadata if key.startswith("model."))
    if not model_keys:
        raise _config_error(checkpoint_dir, "distributed checkpoint has no 'model.*' keys")

    logger.info("Loading %d unsharded tensors from %s", len(model_keys), dcp_dir)
    hf_state_dict: dict[str, Any] = {}
    for key, tensor in zip(model_keys, load_keys(dcp_dir, model_keys), strict=True):
        hf_state_dict[_mm_olmo_key_to_hf_key(key)] = tensor

    if "lm_head.weight" not in hf_state_dict:
        # Weight-tied mm_olmo model (`can_predict_extra_tokens=False`): logits use
        # the base-vocab embedding table; the converter zero-pads the extra rows.
        base_embedding = hf_state_dict.get("model.transformer.wte.embedding")
        if base_embedding is None:
            raise _config_error(
                checkpoint_dir, "missing both 'lm_head' and 'wte.embedding' tensors"
            )
        hf_state_dict["lm_head.weight"] = base_embedding

    return molmo2_hf_state_dict_to_multimodal_lm(hf_state_dict, model_cfg)


def load_checkpoint_weights(
    info: MultimodalCheckpointInfo, checkpoint_dir: str, model: Any
) -> None:
    """Load checkpoint weights into a freshly built ``MultimodalLM`` (on CPU)."""
    _untie_lm_head(model)

    if info.format == "olmo_core_dcp":
        from olmo_core.distributed.checkpoint import load_model_and_optim_state

        load_model_and_optim_state(str(Path(checkpoint_dir) / "model_and_optim"), model)
        return

    if info.format == "olmo_core_unsharded":
        from safetensors.torch import load_file

        state_dict = load_file(str(Path(checkpoint_dir) / "model.safetensors"))
        model.load_state_dict(state_dict)
        return

    state_dict = _load_mm_olmo_state_dict(checkpoint_dir, model.cfg)
    model.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Tokenizer / preprocessing hints
# ---------------------------------------------------------------------------


def resolve_tokenizer_path(info: MultimodalCheckpointInfo, explicit_tokenizer: str | None) -> str:
    """Pick the tokenizer to load: explicit > checkpoint hint > Molmo2 default.

    OLMo-core multimodal checkpoints record the HF model they were bootstrapped
    from as ``model_id``; mm_olmo checkpoints record only the *base* tokenizer
    (e.g. ``Qwen/Qwen3-4B``) which lacks the Molmo2 image special tokens and
    chat template, so those fall through to the Molmo2 default.
    """
    if explicit_tokenizer is not None:
        return explicit_tokenizer
    if info.format in ("olmo_core_dcp", "olmo_core_unsharded"):
        model_id = info.config.get("model_id")
        if isinstance(model_id, str) and model_id:
            return model_id
    return DEFAULT_TOKENIZER


def resolve_max_crops(info: MultimodalCheckpointInfo, explicit_max_crops: int | None) -> int:
    """Pick the multi-crop budget: explicit > checkpoint hint > OLMo-core default."""
    from olmo_core.nn.vision.molmo2_tokens import DEFAULT_MAX_CROPS

    if explicit_max_crops is not None:
        return explicit_max_crops
    if info.format == "mm_olmo_dcp":
        image_cfg = (info.model_config.get("mm_preprocessor") or {}).get("image") or {}
        max_crops = image_cfg.get("max_crops")
        if isinstance(max_crops, int) and max_crops > 0:
            return max_crops
    else:
        dataset = info.config.get("dataset")
        if isinstance(dataset, dict):
            max_crops = dataset.get("max_crops")
            if isinstance(max_crops, int) and max_crops > 0:
                return max_crops
    return DEFAULT_MAX_CROPS


# ---------------------------------------------------------------------------
# OLMo-core export → released-Molmo2 HF state dict (for HuggingFaceProvider)
# ---------------------------------------------------------------------------


def is_olmo_core_hf_export(checkpoint_dir: str) -> bool:
    """Whether ``checkpoint_dir`` is a consolidated OLMo-core multimodal export
    (``olmo_core_config.json`` + ``model.safetensors`` with OLMo-core key names)."""
    try:
        return detect_checkpoint_format(checkpoint_dir).format == "olmo_core_unsharded"
    except Exception:
        return False


def olmo_core_export_reference_model(checkpoint_dir: str) -> str:
    """The HF repo whose config/processor/modeling files match an export's weights."""
    info = detect_checkpoint_format(checkpoint_dir)
    model_id = info.config.get("model_id")
    if isinstance(model_id, str) and model_id:
        return model_id
    return DEFAULT_TOKENIZER


def convert_olmo_core_to_molmo2_hf_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    base_vocab_size: int,
    patch_size: int,
) -> dict[str, torch.Tensor]:
    """Convert an OLMo-core ``MultimodalLM`` state dict to released-Molmo2 HF naming.

    The exact inverse of OLMo-core's
    ``molmo2_loader.molmo2_hf_state_dict_to_multimodal_lm`` (no olmo-core
    dependency): re-fuse the split QKV / SwiGLU projections, split the token
    embedding back into base + image-special tables, drop the extra-token
    ``lm_head`` output rows (input-only tokens; the released architecture has no
    columns for them), and permute the ViT patch embedding back to Molmo2's
    spatial-first flatten order.

    :param base_vocab_size: The released model's base vocab (``text_config.
        vocab_size``, e.g. 151936) — the split point of the embedding table.
    :param patch_size: ViT patch size (``vit_config.image_patch_size``) for the
        patch-embedding permutation.
    """
    import re

    import torch

    sd = dict(state_dict)
    out: dict[str, torch.Tensor] = {}

    def take(key: str) -> torch.Tensor:
        if key not in sd:
            raise ValueError(f"OLMo-core export is missing required tensor {key!r}")
        return sd.pop(key)

    def maybe(key: str) -> torch.Tensor | None:
        return sd.pop(key, None)

    # --- LM: embeddings / head / final norm ---------------------------------
    embeddings = take("lm.embeddings.weight")
    if embeddings.shape[0] <= base_vocab_size:
        raise ValueError(
            f"lm.embeddings.weight has {embeddings.shape[0]} rows; expected more than the "
            f"base vocab ({base_vocab_size}) to split off the image-special-token table"
        )
    out["model.transformer.wte.embedding"] = embeddings[:base_vocab_size].contiguous()
    out["model.transformer.wte.new_embedding"] = embeddings[base_vocab_size:].contiguous()
    out["model.transformer.ln_f.weight"] = take("lm.lm_head.norm.weight")
    out["lm_head.weight"] = take("lm.lm_head.w_out.weight")[:base_vocab_size].contiguous()

    def layer_indices(pattern: str) -> list[int]:
        found = {int(m.group(1)) for key in sd if (m := re.match(pattern, key))}
        return sorted(found)

    # --- LM blocks: re-fuse QKV and SwiGLU ----------------------------------
    for i in layer_indices(r"lm\.blocks\.(\d+)\."):
        src, dst = f"lm.blocks.{i}", f"model.transformer.blocks.{i}"
        out[f"{dst}.attn_norm.weight"] = take(f"{src}.attention_norm.weight")
        out[f"{dst}.ff_norm.weight"] = take(f"{src}.feed_forward_norm.weight")

        out[f"{dst}.self_attn.att_proj.weight"] = torch.cat(
            [take(f"{src}.attention.w_{p}.weight") for p in ("q", "k", "v")], dim=0
        )
        # `maybe` pops, so resolve all three before filtering: attention biases are
        # fused only when q, k and v all carry one.
        qkv_biases = [maybe(f"{src}.attention.w_{p}.bias") for p in ("q", "k", "v")]
        biases = [bias for bias in qkv_biases if bias is not None]
        if len(biases) == len(qkv_biases):
            out[f"{dst}.self_attn.att_proj.bias"] = torch.cat(biases, dim=0)
        out[f"{dst}.self_attn.attn_out.weight"] = take(f"{src}.attention.w_out.weight")
        for norm in ("q_norm", "k_norm"):
            tensor = maybe(f"{src}.attention.{norm}.weight")
            if tensor is not None:
                out[f"{dst}.self_attn.{norm}.weight"] = tensor

        # HF forward: x, gate = ff_proj.chunk(2) → rows [:H] are the multiplier
        # branch (our w3), rows [H:] the gate branch (our w1).
        out[f"{dst}.mlp.ff_proj.weight"] = torch.cat(
            [take(f"{src}.feed_forward.w3.weight"), take(f"{src}.feed_forward.w1.weight")], dim=0
        )
        out[f"{dst}.mlp.ff_out.weight"] = take(f"{src}.feed_forward.w2.weight")

    # --- Vision: patch embedding permute + blocks ---------------------------
    patch_weight = take("vision.patch_embedding.weight")
    d_model, total = patch_weight.shape
    if total != 3 * patch_size * patch_size:
        raise ValueError(
            f"vision.patch_embedding.weight has {total} columns; expected "
            f"{3 * patch_size * patch_size} for patch_size={patch_size}"
        )
    out["model.vision_backbone.image_vit.patch_embedding.weight"] = (
        patch_weight.reshape(d_model, 3, patch_size, patch_size)
        .permute(0, 2, 3, 1)
        .reshape(d_model, total)
        .contiguous()
    )
    out["model.vision_backbone.image_vit.patch_embedding.bias"] = take(
        "vision.patch_embedding.bias"
    )
    out["model.vision_backbone.image_vit.positional_embedding"] = take(
        "vision.positional_embedding"
    )

    for i in layer_indices(r"vision\.blocks\.(\d+)\."):
        src = f"vision.blocks.{i}"
        dst = f"model.vision_backbone.image_vit.transformer.resblocks.{i}"
        for ours, hf in (("attn_norm", "attention_norm"), ("ffn_norm", "ffn_norm")):
            for suffix in ("weight", "bias"):
                out[f"{dst}.{hf}.{suffix}"] = take(f"{src}.{ours}.{suffix}")
        for proj in ("wq", "wk", "wv", "wo"):
            for suffix in ("weight", "bias"):
                out[f"{dst}.attention.{proj}.{suffix}"] = take(f"{src}.attn.{proj}.{suffix}")
        for proj in ("w1", "w2"):
            for suffix in ("weight", "bias"):
                out[f"{dst}.feed_forward.{proj}.{suffix}"] = take(f"{src}.ffn.{proj}.{suffix}")

    # --- Connector -----------------------------------------------------------
    for proj in ("wq", "wk", "wv", "wo"):
        for suffix in ("weight", "bias"):
            out[f"model.vision_backbone.image_pooling_2d.{proj}.{suffix}"] = take(
                f"connector.pooling.{proj}.{suffix}"
            )
    for proj in ("w1", "w2", "w3"):
        out[f"model.vision_backbone.image_projector.{proj}.weight"] = take(
            f"connector.projector.{proj}.weight"
        )

    if sd:
        leftover = ", ".join(sorted(sd)[:8])
        raise ValueError(f"OLMo-core export has unconverted tensors: {leftover}")
    return out


# ---------------------------------------------------------------------------
# KV-cached decoding
# ---------------------------------------------------------------------------

_CACHED_BACKEND_CLS: type | None = None


def _cached_torch_backend_class() -> type:
    """Build (once) a ``TorchAttentionBackend`` subclass that supports KV caching.

    OLMo-core's dense ``torch`` backend is the only one that supports the
    multimodal ``or_mask`` / ``and_mask``, but it rejects KV caching — so
    multimodal decoding would otherwise re-run the LM over the whole sequence
    for every generated token. The ``Attention`` module already carries all the
    cache plumbing (``KVCacheManager``, RoPE ``start_pos`` from
    ``current_position()``); this subclass adds the missing piece: write the
    new K/V into the cache and run SDPA over the cached prefix.

    Scope (asserted): no sliding window, no context parallelism, no
    intra-document masking. Queries at absolute positions ``pos..pos+T-1``
    attend keys ``0..pos+T-1`` causally; ``or_mask`` / ``and_mask`` (sized
    ``(B, 1, T, T)`` over the current forward's tokens) are aligned to the key
    axis by left-padding, which is exact for the prefill call (``pos == 0``,
    the only call that passes them).

    Variable-length batches are supported through cache left-padding: rows are
    right-aligned in the cache (``kv_cache_manager.cache_leftpad`` holds each
    row's pad length, set by :func:`prepare_kv_caches`) so every row's last
    prompt token lands in the final prefill slot and all rows share one write
    position per decode step. Real queries then must not attend the pad slots
    (their K/V is garbage from pad tokens), while pad-slot queries keep their
    causal rows so no softmax row is fully masked (a fully-masked row turns
    into NaN attention that would poison later layers through the cached K/V).
    """
    global _CACHED_BACKEND_CLS
    if _CACHED_BACKEND_CLS is not None:
        return _CACHED_BACKEND_CLS

    import torch
    import torch.nn.functional as F
    from olmo_core.nn.attention.backend import TorchAttentionBackend, _repeat_kv

    class CachedTorchAttentionBackend(TorchAttentionBackend):
        @classmethod
        def assert_supports_kv_cache(cls) -> None:
            pass

        def forward(  # noqa: C901
            self,
            qkv,
            cu_doc_lens=None,
            cu_doc_lens_q=None,
            cu_doc_lens_k=None,
            max_doc_len=None,
            max_doc_len_q=None,
            max_doc_len_k=None,
            local_k_slice=None,
            kv_cache_manager=None,
            or_mask=None,
            and_mask=None,
        ):
            if kv_cache_manager is None:
                return super().forward(
                    qkv,
                    cu_doc_lens=cu_doc_lens,
                    cu_doc_lens_q=cu_doc_lens_q,
                    cu_doc_lens_k=cu_doc_lens_k,
                    max_doc_len=max_doc_len,
                    max_doc_len_q=max_doc_len_q,
                    max_doc_len_k=max_doc_len_k,
                    local_k_slice=local_k_slice,
                    or_mask=or_mask,
                    and_mask=and_mask,
                )

            if isinstance(qkv, torch.Tensor):
                raise RuntimeError(f"'{type(self).__name__}' doesn't support packed QKV")
            if self.window_size != (-1, -1):
                raise RuntimeError(
                    f"'{type(self).__name__}' doesn't support KV caching with sliding windows"
                )
            if any(
                opt is not None
                for opt in (
                    cu_doc_lens,
                    cu_doc_lens_q,
                    cu_doc_lens_k,
                    max_doc_len,
                    max_doc_len_q,
                    max_doc_len_k,
                )
            ):
                raise RuntimeError(
                    f"'{type(self).__name__}' doesn't support intra-document masking"
                )
            if self.cp_enabled:
                raise RuntimeError(
                    f"'{type(self).__name__}' doesn't support KV caching with context parallelism"
                )

            q, k, v = qkv
            seq_len = q.shape[1]
            # CPU-side mirror of ``cache_seqlens`` to avoid a GPU sync per layer
            # per step. ``Attention.sdpa`` calls ``update_seqlen(seq_len)`` right
            # after this forward, so advance the mirror by the same amount.
            pos = getattr(kv_cache_manager, "_position_mirror", None)
            if pos is None:
                pos = int(kv_cache_manager.cache_seqlens.item())
            kv_cache_manager._position_mirror = pos + seq_len
            total = pos + seq_len

            k_cache, v_cache = kv_cache_manager.k_cache, kv_cache_manager.v_cache
            if total > k_cache.shape[1]:
                raise RuntimeError(f"KV cache overflow: {total} > allocated {k_cache.shape[1]}")
            k_cache[:, pos:total] = k.to(k_cache.dtype)
            v_cache[:, pos:total] = v.to(v_cache.dtype)
            k_full = k_cache[:, :total].to(q.dtype)
            v_full = v_cache[:, :total].to(q.dtype)

            has_leftpad = getattr(kv_cache_manager, "_has_leftpad", False)
            attn_mask: torch.Tensor | None = None
            if seq_len > 1 or or_mask is not None or and_mask is not None or has_leftpad:
                base = torch.ones(seq_len, total, device=q.device, dtype=torch.bool).tril(
                    diagonal=pos
                )

                def _align_keys(mask: torch.Tensor, pad_value: bool) -> torch.Tensor:
                    mask = mask.to(device=q.device, dtype=torch.bool)
                    missing = total - mask.shape[-1]
                    if missing > 0:
                        pad = torch.full((*mask.shape[:-1], missing), pad_value, device=q.device)
                        mask = torch.cat([pad, mask], dim=-1)
                    return mask

                if or_mask is not None:
                    base = base | _align_keys(or_mask, False)
                if and_mask is not None:
                    base = base & _align_keys(and_mask, True)
                if has_leftpad:
                    leftpad = kv_cache_manager.cache_leftpad.to(device=q.device, dtype=torch.long)
                    # Real queries never see pad-slot keys; pad-slot queries keep
                    # their causal rows so no softmax row is fully masked.
                    key_ok = torch.arange(total, device=q.device) >= leftpad[:, None]
                    q_abs = pos + torch.arange(seq_len, device=q.device)
                    pad_query = q_abs[None, :] < leftpad[:, None]
                    base = base & (key_ok[:, None, None, :] | pad_query[:, None, :, None])
                attn_mask = base

            n_rep = self.n_heads // self.n_kv_heads
            k_full = _repeat_kv(k_full, n_rep)
            v_full = _repeat_kv(v_full, n_rep)
            q, k_full, v_full = q.transpose(1, 2), k_full.transpose(1, 2), v_full.transpose(1, 2)
            att = F.scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p,
                is_causal=False,
                scale=self.scale,
            )
            return att.transpose(1, 2).contiguous()

    _CACHED_BACKEND_CLS = CachedTorchAttentionBackend
    return CachedTorchAttentionBackend


def _lm_attention_modules(model: Any) -> list[Any]:
    return [
        block.attention
        for block in model.lm.blocks.values()
        if getattr(block, "attention", None) is not None
    ]


def enable_kv_cache(model: Any) -> bool:
    """Swap each LM attention's dense ``torch`` backend for the cached subclass.

    Returns ``False`` (leaving the model untouched) when any block uses a
    different backend, a sliding window, or has no attention module — callers
    should then fall back to no-cache decoding.
    """
    from olmo_core.nn.attention.backend import TorchAttentionBackend

    cached_cls = _cached_torch_backend_class()
    attentions = _lm_attention_modules(model)
    if len(attentions) != len(model.lm.blocks):
        return False
    for attention in attentions:
        backend = getattr(attention, "backend", None)
        if backend is None or type(backend) not in (TorchAttentionBackend, cached_cls):
            return False
        if backend.window_size != (-1, -1):
            return False
    for attention in attentions:
        attention.backend.__class__ = cached_cls
    return True


_EXPLICIT_POSITION_KV_MANAGER_CLS: type | None = None


def _explicit_position_kv_manager_class() -> type:
    """Build (once) a ``KVCacheManager`` subclass for explicit-position decoding.

    ``Attention.forward`` derives the RoPE ``start_pos`` from
    ``kv_cache_manager.current_position()``, but RoPE forbids combining
    ``start_pos`` with explicit ``position_ids``. The decode loop drives RoPE
    entirely through per-row ``position_ids`` (required for variable-length
    batches, where each row sits at a different absolute position), so this
    manager reports no position of its own.
    """
    global _EXPLICIT_POSITION_KV_MANAGER_CLS
    if _EXPLICIT_POSITION_KV_MANAGER_CLS is not None:
        return _EXPLICIT_POSITION_KV_MANAGER_CLS

    from olmo_core.nn.attention.kv_cache import KVCacheManager

    class ExplicitPositionKVCacheManager(KVCacheManager):
        def current_position(self) -> None:  # type: ignore[override]
            return None

    _EXPLICIT_POSITION_KV_MANAGER_CLS = ExplicitPositionKVCacheManager
    return ExplicitPositionKVCacheManager


def prepare_kv_caches(
    model: Any,
    batch_size: int,
    max_seq_len: int,
    leftpad: torch.Tensor | None = None,
) -> None:
    """Initialize (or reset) a KV cache on every LM attention block.

    :param leftpad: Per-row left-pad lengths ``(batch_size,)`` for
        variable-length batches; rows are right-aligned in the cache and the
        cached backend masks the pad slots out of every real query's row.
    """
    manager_cls = _explicit_position_kv_manager_class()
    has_leftpad = leftpad is not None and bool((leftpad > 0).any())
    for attention in _lm_attention_modules(model):
        manager = attention.kv_cache_manager
        if isinstance(manager, manager_cls):
            manager.reset(batch_size, max_seq_len)
        else:
            manager = manager_cls(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=attention.n_kv_heads,
                head_dim=attention.head_dim,
                device=attention.w_k.weight.device,
            )
            attention.kv_cache_manager = manager
        if leftpad is not None:
            manager.cache_leftpad.copy_(leftpad)
        manager._has_leftpad = has_leftpad
        manager._position_mirror = 0


def free_kv_caches(model: Any) -> None:
    """Drop all LM KV caches so subsequent forwards run uncached."""
    for attention in _lm_attention_modules(model):
        attention.kv_cache_manager = None


def rope_buffers(model: Any, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-length RoPE sin/cos tables covering positions ``0..max_seq_len - 1``.

    Explicit-``position_ids`` RoPE gathers rows from the sin/cos tables, but
    ``RotaryEmbedding.forward`` sizes its internal tables from the current
    forward's ``seq_len`` (1 during decode) when ``start_pos`` is absent —
    far short of the decode positions. The tables must therefore be passed in
    explicitly; this builds them once per batch. The Transformer broadcasts one
    buffer pair to every block, so all blocks must share one RoPE config
    (asserted here; per-position table rows do not depend on table length, so
    these buffers match what shorter internal tables would hold).
    """
    ropes = [
        rope
        for attention in _lm_attention_modules(model)
        if (rope := getattr(attention, "rope", None)) is not None
    ]
    if not ropes:
        raise ValueError("Model has no RoPE modules; cannot run explicit-position decoding")
    configs = {
        (
            type(rope).__name__,
            getattr(rope, "theta", None),
            getattr(rope, "rotary_dim", None),
            repr(getattr(rope, "scaling", None)),
        )
        for rope in ropes
    }
    if len(configs) != 1:
        raise ValueError(
            f"LM blocks use differing RoPE configs ({sorted(configs)}); batched "
            "explicit-position decoding requires one shared config"
        )
    buffers = ropes[0].get_buffers(max_seq_len, next(iter(model.lm.parameters())).device)
    if buffers.pos_sin is None or buffers.pos_cos is None:
        raise ValueError(
            f"{type(ropes[0]).__name__} does not expose sin/cos RoPE buffers; batched "
            "explicit-position decoding requires RotaryEmbedding"
        )
    return buffers.pos_sin, buffers.pos_cos


def resolve_max_length(info: MultimodalCheckpointInfo, explicit_max_length: int | None) -> int:
    """Pick the max sequence length: explicit > checkpoint hint > 4096."""
    if explicit_max_length is not None:
        return explicit_max_length
    if info.format == "mm_olmo_dcp":
        max_length = (info.model_config.get("llm") or {}).get("max_sequence_length")
        if isinstance(max_length, int) and max_length > 0:
            return max_length
    else:
        for section in ("dataset", "collator", "train_module"):
            value = info.config.get(section)
            if isinstance(value, dict):
                max_length = value.get("sequence_length") or value.get("max_sequence_length")
                if isinstance(max_length, int) and max_length > 0:
                    return max_length
    return 4096
