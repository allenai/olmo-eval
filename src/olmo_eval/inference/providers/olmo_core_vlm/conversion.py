"""Conversion of consolidated OLMo-core multimodal exports to the released Molmo2 layout.

Used by the HuggingFace provider to run OLMo-core safetensors exports through the
released remote-code model without an intermediate on-disk conversion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from olmo_eval.inference.providers.olmo_core_vlm.checkpoint import (
    detect_checkpoint_format,
)
from olmo_eval.inference.providers.olmo_core_vlm.preprocessing import DEFAULT_TOKENIZER

logger = logging.getLogger(__name__)


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
