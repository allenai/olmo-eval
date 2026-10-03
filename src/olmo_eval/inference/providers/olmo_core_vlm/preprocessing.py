"""Tokenizer and preprocessing hints resolved from a multimodal checkpoint.

The Molmo2 image special tokens are resolved to vocab IDs through the tokenizer so
the provider works for any vocab layout that defines them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, overload

import olmo_eval.inference.providers.olmo_core_utils as core_utils
from olmo_eval.inference.providers.olmo_core_vlm.checkpoint import (
    MultimodalCheckpointInfo,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

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


def resolve_tokenizer_path(info: MultimodalCheckpointInfo, explicit_tokenizer: str | None) -> str:
    """Pick the tokenizer to load: explicit > checkpoint hint > Molmo2 default.

    OLMo-core multimodal checkpoints record the HF model they were bootstrapped
    from as ``model_id``; a checkpoint trained on a text model's own tokenizer
    (e.g. OLMo 3.5 on Dolma2) records it as its dataset tokenizer instead, see
    :func:`dataset_tokenizer`. mm_olmo checkpoints record only the *base* tokenizer
    (e.g. ``Qwen/Qwen3-4B``) which lacks the Molmo2 image special tokens and
    chat template, so those fall through to the Molmo2 default.
    """
    if explicit_tokenizer is not None:
        return explicit_tokenizer
    if info.format in ("olmo_core_dcp", "olmo_core_unsharded"):
        model_id = info.config.get("model_id")
        if isinstance(model_id, str) and model_id:
            return model_id
        from_dataset = dataset_tokenizer(info)
        if from_dataset is not None:
            return from_dataset[0]
    return DEFAULT_TOKENIZER


def dataset_tokenizer(info: MultimodalCheckpointInfo) -> tuple[str, str | None] | None:
    """The ``(identifier, revision)`` of the text tokenizer an OLMo-core run adapted, if any.

    Runs that train on a text model's tokenizer (the multimodal alignment mixture)
    record it under ``dataset.tokenizer`` together with ``dataset.model_vocab_size``,
    the embedding rows the Molmo2 image tokens were appended into (OLMo-core's
    ``prepare_molmo2_tokenizer``). The provider appends the same tokens at load.
    """
    dataset = info.config.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("model_vocab_size"), int):
        return None
    tokenizer = dataset.get("tokenizer")
    identifier = tokenizer.get("identifier") if isinstance(tokenizer, dict) else None
    if not isinstance(identifier, str) or not identifier:
        return None
    revision = dataset.get("tokenizer_revision")
    return identifier, revision if isinstance(revision, str) and revision else None


def _contains_item(config: Any, key: str, value: object) -> bool:
    """Whether ``key: value`` appears anywhere in a nested config tree."""
    if isinstance(config, dict):
        return config.get(key) == value or any(
            _contains_item(child, key, value) for child in config.values()
        )
    if isinstance(config, list):
        return any(_contains_item(child, key, value) for child in config)
    return False


def _contains_class(config: Any, suffix: str) -> bool:
    """Whether a ``_CLASS_`` ending in ``suffix`` appears anywhere in a nested config tree."""
    if isinstance(config, dict):
        class_name = config.get("_CLASS_")
        if isinstance(class_name, str) and class_name.endswith(suffix):
            return True
        return any(_contains_class(child, suffix) for child in config.values())
    if isinstance(config, list):
        return any(_contains_class(child, suffix) for child in config)
    return False


def uses_document_layout(info: MultimodalCheckpointInfo) -> bool:
    """Whether the checkpoint was trained on OLMo-core's plain-document layout.

    Document-layout examples (OLMo-core ``data/multimodal/document_layout.py``) carry
    no chat template: ``[EOS] + image tokens + prompt``, with the response
    continuing after a space, and every token, image tokens included, attends
    causally. Runs whose dataset sources say ``message_format: document`` use it,
    as does any language model with Kimi Delta Attention blocks, whose recurrent
    mixers cannot apply the bidirectional image mask.
    """
    if info.format == "mm_olmo_dcp":
        return False
    if _contains_item(info.config.get("dataset"), "message_format", "document"):
        return True
    return _contains_class(info.model_config.get("lm"), "KimiDeltaAttentionConfig")


def trains_in_bfloat16(info: MultimodalCheckpointInfo) -> bool:
    """Whether the checkpoint's language model is an OLMoDDP model.

    The OLMoDDP train module runs the whole multimodal model in bfloat16 (FP32
    copies live only in its optimizer), so evaluating with bfloat16 weights and no
    autocast reproduces the training forward; autocast would also downcast the
    projections the model keeps in full precision on purpose.
    """
    if info.format == "mm_olmo_dcp":
        return False
    lm = info.model_config.get("lm")
    class_name = lm.get("_CLASS_") if isinstance(lm, dict) else None
    return isinstance(class_name, str) and class_name.endswith("OLMoDDPModelConfig")


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


#: Per-image crop budget for multi-image prompts when the checkpoint names none: the
#: released Molmo2 ``max_multi_image_crops``, which mm_olmo's ``MultiImagePreprocessor``
#: uses whenever an example has more than one image.
DEFAULT_MAX_MULTI_IMAGE_CROPS = 8

#: The sequence length mm_olmo's ``eval_molmo2.py`` evaluates Molmo2 models at. A
#: checkpoint's training window is often shorter (16,384 for Molmo2-4B), and many-image
#: prompts such as MMIU's exceed it.
EVAL_MAX_LENGTH = 64_000


def resolve_max_multi_image_crops(
    info: MultimodalCheckpointInfo, explicit_max_multi_image_crops: int | None
) -> int:
    """Pick the per-image crop budget for multi-image prompts: explicit > checkpoint hint > 8."""
    if explicit_max_multi_image_crops is not None:
        return explicit_max_multi_image_crops
    if info.format == "mm_olmo_dcp":
        image_cfg = (info.model_config.get("mm_preprocessor") or {}).get("image") or {}
        max_crops = image_cfg.get("max_multi_image_crops")
    else:
        dataset = info.config.get("dataset")
        max_crops = dataset.get("max_multi_image_crops") if isinstance(dataset, dict) else None
    if isinstance(max_crops, int) and max_crops > 0:
        return max_crops
    return DEFAULT_MAX_MULTI_IMAGE_CROPS


def resolve_max_length(info: MultimodalCheckpointInfo, explicit_max_length: int | None) -> int:
    """Pick the max sequence length: explicit, else the longer of the checkpoint's training
    window and :data:`EVAL_MAX_LENGTH`.

    Only prompts longer than the training window see the difference: they are answered,
    as mm_olmo answers them, instead of failing their instance.
    """
    if explicit_max_length is not None:
        return explicit_max_length
    return max(_checkpoint_max_length(info), EVAL_MAX_LENGTH)


def _checkpoint_max_length(info: MultimodalCheckpointInfo) -> int:
    """The checkpoint's training sequence length, or 4096 when it records none."""
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
