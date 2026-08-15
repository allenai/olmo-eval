"""Multimodal (vision-language) OLMo-core inference provider.

Runs OLMo-core ``MultimodalLM`` checkpoints (vision encoder + connector + LM)
for image+text generation. Unlike the text-only :class:`OlmoCoreProvider`,
which wraps ``TransformerGenerationModule``, this provider builds the
``MultimodalLM`` directly and drives its own decode loop: the vision branch of
OLMo-core has no multimodal generation module, and the attention backends that
support the bidirectional image-token mask (``torch`` / ``flex``) do not
support KV caching. The vision tower runs once per request (image features are
spliced into the prompt embeddings at prefill); each decode step re-runs the LM
over the growing sequence with ``logits_to_keep=1``.

Supported checkpoint formats (see ``olmo_core_vlm_utils``): raw OLMo-core
multimodal trainer checkpoints, consolidated OLMo-core safetensors exports, and
mm_olmo (Molmo2 ``video_olmo``) trainer checkpoints, which are key-remapped into
the OLMo-core layout at load time.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import gc
import logging
import threading
from typing import TYPE_CHECKING, Any, cast

import olmo_eval.inference.providers.olmo_core_vlm_utils as vlm_utils
from olmo_eval.common.debug import is_debug_requests
from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.inference.base import InferenceProvider

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Provider-foreign kwargs silently accepted for config compatibility.
_IGNORED_KWARGS = frozenset(
    {
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "use_tqdm_on_load",
        "add_bos_token",
        "load_format",
        "model_loader_extra_config",
        "enable_auto_tool_choice",
        "token",
        "cache_dir",
        "local_files_only",
    }
)


def _to_torch_dtype(name: str) -> Any:
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


class OlmoCoreVLMProvider(InferenceProvider):
    """Provider for multimodal OLMo-core (``MultimodalLM``) checkpoints."""

    def __init__(
        self,
        model_name: str,
        tokenizer: str | None = None,
        *,
        dtype: str = "float32",
        autocast_dtype: str | None = "bfloat16",
        max_crops: int | None = None,
        max_model_len: int | None = None,
        device: str | None = None,
        bidirectional_image_attention: bool = True,
        use_cache: bool = True,
        trust_remote_code: bool = False,
        revision: str | None = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoTokenizer

        max_model_len = self._resolve_max_model_len_alias(max_model_len, kwargs)
        for key in list(kwargs):
            if key in _IGNORED_KWARGS:
                kwargs.pop(key)
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported OlmoCoreVLMProvider kwargs: {unsupported}")
        if dtype == "auto":
            dtype = "float32"

        super().__init__(model_name)

        self.checkpoint_info = vlm_utils.detect_checkpoint_format(model_name)
        logger.info(
            "Detected multimodal checkpoint format %r for %s",
            self.checkpoint_info.format,
            model_name,
        )

        tokenizer_path = vlm_utils.resolve_tokenizer_path(self.checkpoint_info, tokenizer)
        tokenizer_kwargs = {
            key: value
            for key, value in {
                "revision": revision,
                "force_download": force_download,
                "trust_remote_code": trust_remote_code,
            }.items()
            if value
        }
        self.tokenizer: vlm_utils.VLMTokenizerProtocol = cast(
            vlm_utils.VLMTokenizerProtocol,
            AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs),
        )
        self._resolve_special_token_ids(tokenizer_path)

        self.model_config = vlm_utils.build_model_config(
            self.checkpoint_info, image_patch_token_id=self.image_patch_token_id
        )
        if use_cache and vlm_utils.use_dense_attention_backend(self.model_config):
            logger.info(
                "Checkpoint records the fused 'flex' attention backend, which cannot carry a "
                "KV cache; using the dense 'torch' backend, which has the same masking "
                "semantics, so decoding stays cached. Pass use_cache=False to keep 'flex'."
            )
        if self.model_config.image_patch_token_id != self.image_patch_token_id:
            raise ValueError(
                f"Checkpoint image_patch_token_id ({self.model_config.image_patch_token_id}) "
                f"does not match the tokenizer's <im_patch> id ({self.image_patch_token_id}); "
                f"pass a matching tokenizer (e.g. tokenizer={vlm_utils.DEFAULT_TOKENIZER!r})."
            )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.param_dtype = _to_torch_dtype(dtype)
        self.autocast_dtype = autocast_dtype
        self.bidirectional_image_attention = bidirectional_image_attention
        self.max_crops = vlm_utils.resolve_max_crops(self.checkpoint_info, max_crops)
        self.max_length = vlm_utils.resolve_max_length(self.checkpoint_info, max_model_len)

        logger.info("Building MultimodalLM and loading weights from %s", model_name)
        model = self.model_config.build(init_device="cpu")
        vlm_utils.load_checkpoint_weights(self.checkpoint_info, model_name, model)
        model = model.to(dtype=self.param_dtype).to(self.device)
        model.eval()
        model.requires_grad_(False)
        self.model = model

        self.use_cache = use_cache and vlm_utils.enable_kv_cache(model)
        if use_cache and not self.use_cache:
            logger.warning(
                "KV-cached decoding unavailable for this model's attention backend; "
                "falling back to full re-forward decoding."
            )
        # KV-cached decoding mutates per-block cache state on the shared model,
        # so concurrent agenerate/alogprobs threads must take turns.
        self._model_lock = threading.Lock()
        self._embedding_weight_cache: torch.Tensor | None = None

    @staticmethod
    def _resolve_max_model_len_alias(
        max_model_len: int | None, kwargs: dict[str, Any]
    ) -> int | None:
        max_length = kwargs.pop("max_length", None)
        if max_length is None:
            return max_model_len
        if not isinstance(max_length, int):
            raise ValueError("OlmoCoreVLMProvider max_length must be an integer when set")
        if max_model_len is not None and max_model_len != max_length:
            raise ValueError(
                "OlmoCoreVLMProvider received both max_model_len and max_length "
                "with different values"
            )
        return max_length

    def _resolve_special_token_ids(self, tokenizer_path: str) -> None:
        """Resolve Molmo2 image special-token IDs through the tokenizer."""
        ids = self.tokenizer.convert_tokens_to_ids(list(vlm_utils.IMAGE_SPECIAL_TOKENS))
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if any(token_id is None or token_id == unk_id for token_id in ids):
            raise ValueError(
                f"Tokenizer {tokenizer_path!r} does not define the Molmo2 image special tokens "
                f"{vlm_utils.IMAGE_SPECIAL_TOKENS}; pass tokenizer={vlm_utils.DEFAULT_TOKENIZER!r} "
                "or another Molmo2-family tokenizer."
            )
        by_name = dict(zip(vlm_utils.IMAGE_SPECIAL_TOKENS, ids, strict=True))
        self.image_patch_token_id = by_name["<im_patch>"]
        self.image_structural_token_ids = tuple(ids)

        placeholder = self.tokenizer.convert_tokens_to_ids(vlm_utils.IMAGE_PLACEHOLDER_TOKEN)
        if placeholder is None or placeholder == unk_id:
            raise ValueError(
                f"Tokenizer {tokenizer_path!r} does not define "
                f"{vlm_utils.IMAGE_PLACEHOLDER_TOKEN!r}"
            )
        self.image_placeholder_id = placeholder

        stop_ids = {self.tokenizer.eos_token_id}
        end_of_turn = self.tokenizer.convert_tokens_to_ids(vlm_utils.END_OF_TURN_TOKEN)
        if end_of_turn is not None and end_of_turn != unk_id:
            stop_ids.add(end_of_turn)
        self.stop_token_ids = frozenset(token_id for token_id in stop_ids if token_id is not None)

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    # ------------------------------------------------------------------
    # Request preparation
    # ------------------------------------------------------------------

    def _chat_text(self, request: LMRequest, num_images: int) -> str:
        """Render the request as chat-templated text with ``<|image|>`` markers.

        Passes structured content parts so the Molmo2 chat template itself
        hoists the image markers in front of the conversation (with ``Image N``
        prefixes for multi-image requests), exactly like the released
        processor's ``apply_chat_template``.
        """
        messages = list(request.messages or ({"role": "user", "content": request.prompt},))
        chat: list[dict[str, Any]] = []
        attached = False
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "") or ""
            content: Any = [{"type": "text", "text": text}]
            if role == "user" and not attached and num_images:
                content = [{"type": "image"} for _ in range(num_images)] + content
                attached = True
            chat.append({"role": role, "content": content})
        if not attached and num_images:
            chat.insert(
                0, {"role": "user", "content": [{"type": "image"} for _ in range(num_images)]}
            )
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def _preprocess_images(
        self, images: tuple[Any, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
        """Patchify images and build their expanded token sequences.

        Multiple images are concatenated along the crop axis; each image's
        pooled-patch indices are offset into the combined flat patch axis so a
        single ``pooled_patches_idx`` covers the whole request.

        :returns: ``(images, pooled_patches_idx, per_image_token_ids)``.
        """
        import torch
        from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2
        from olmo_core.nn.vision.molmo2_tokens import build_image_token_ids

        image_size = int(self.model_config.vision.image_default_input_size[0])
        patch_size = int(self.model_config.vision.image_patch_size)

        crop_tensors: list[torch.Tensor] = []
        pooling_tensors: list[torch.Tensor] = []
        token_sequences: list[list[int]] = []
        crop_offset = 0
        for image in images:
            if hasattr(image, "mode") and image.mode != "RGB":
                image = image.convert("RGB")
            crops, pooling, grid = preprocess_image_molmo2(
                image,
                self.param_dtype,
                self.device,
                image_size=image_size,
                patch_size=patch_size,
                max_crops=self.max_crops,
            )
            n_patches_per_crop = crops.shape[2]
            offset_pooling = pooling.clone()
            offset_pooling[offset_pooling >= 0] += crop_offset * n_patches_per_crop
            crop_offset += crops.shape[1]

            crop_tensors.append(crops)
            pooling_tensors.append(offset_pooling)
            token_sequences.append(
                build_image_token_ids(int(grid[0]), int(grid[1]), int(grid[2]), int(grid[3]))
            )

        return (
            torch.cat(crop_tensors, dim=1),
            torch.cat(pooling_tensors, dim=1),
            token_sequences,
        )

    def _encode_request(self, request: LMRequest) -> tuple[list[int], Any, Any]:
        """Tokenize a request, splicing expanded image tokens at each placeholder.

        :returns: ``(token_ids, images, pooled_patches_idx)`` where the tensors
            are ``None`` for text-only requests.
        """
        images = tuple(request.images or ())
        image_tensor = pooling_tensor = None
        token_sequences: list[list[int]] = []
        if images:
            image_tensor, pooling_tensor, token_sequences = self._preprocess_images(images)

        text = self._chat_text(request, len(images))
        token_ids: list[int] = self.tokenizer.encode(text, add_special_tokens=False)

        for image_tokens in token_sequences:
            position = token_ids.index(self.image_placeholder_id)
            token_ids = token_ids[:position] + image_tokens + token_ids[position + 1 :]
        if self.image_placeholder_id in token_ids:
            raise ValueError("Prompt contains more <|image|> placeholders than provided images")

        # The released Molmo2 processor prepends a BOS token (falling back to
        # EOS, matching `Molmo2Processor.insert_bos`).
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        if bos is not None and (not token_ids or token_ids[0] != bos):
            token_ids = [bos, *token_ids]
        return token_ids, image_tensor, pooling_tensor

    # ------------------------------------------------------------------
    # Model forward helpers
    # ------------------------------------------------------------------

    def _autocast(self) -> Any:
        import torch

        if self.autocast_dtype and self.device.type in ("cuda", "cpu"):
            return torch.autocast(
                device_type=self.device.type, dtype=_to_torch_dtype(self.autocast_dtype)
            )
        return contextlib.nullcontext()

    def _embedding_weight(self) -> torch.Tensor:
        """Token-embedding matrix spanning the extra vocab block when there is one.

        Checkpoints trained with extra vocab keep the image special-token rows in a
        second ``extra_weight`` parameter, so a lookup against ``weight`` alone would
        index out of bounds for those IDs. Cached because the weights are frozen for
        the provider's lifetime and the embedding runs once per decode step.
        """
        import torch

        if self._embedding_weight_cache is None:
            embeddings = self.model.lm.embeddings
            extra = getattr(embeddings, "extra_weight", None)
            self._embedding_weight_cache = (
                embeddings.weight if extra is None else torch.cat([embeddings.weight, extra], dim=0)
            )
        return self._embedding_weight_cache

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """LM token embeddings with the model's configured scale/norm applied.

        Mirrors ``MultimodalLM.forward``'s embedding stage so image features
        can be spliced in once at prefill and cheap per-token embeddings can be
        appended during decode.
        """
        import torch.nn.functional as F

        lm = self.model.lm
        h = F.embedding(token_ids, self._embedding_weight(), padding_idx=lm.embeddings.padding_idx)
        if lm.embed_scale is not None:
            h = h * lm.embed_scale
        if lm.embedding_norm is not None:
            h = lm.embedding_norm(h)
        return h

    def _prefill(
        self,
        token_ids: list[int],
        image_tensor: torch.Tensor | None,
        pooling_tensor: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Build prompt embeddings with image features spliced in.

        :returns: ``(input_ids (1, S), embeddings (1, S, d), is_image (S,) or None)``.
        """
        import torch

        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        h = self._embed(ids)

        if image_tensor is not None:
            assert pooling_tensor is not None
            features = self.model._encode_images(image_tensor, pooling_tensor)  # (1, P, d)
            valid_rows = (pooling_tensor >= 0).any(dim=-1)
            features = features[valid_rows]  # (P_valid, d)
            is_patch = ids[0] == self.image_patch_token_id
            n_patches = int(is_patch.sum())
            if n_patches != features.shape[0]:
                raise ValueError(
                    f"Number of <im_patch> tokens ({n_patches}) does not match the number of "
                    f"pooled image features ({features.shape[0]})"
                )
            h = h.clone()
            h[0, is_patch] = h[0, is_patch] + features.to(h.dtype)

        is_image = None
        if self.bidirectional_image_attention and image_tensor is not None:
            import torch

            structural = torch.tensor(
                self.image_structural_token_ids, dtype=torch.long, device=self.device
            )
            is_image = torch.isin(ids[0], structural)
        return ids, h, is_image

    def _step_logits(
        self,
        ids: torch.Tensor,
        h: torch.Tensor,
        is_image: torch.Tensor | None,
    ) -> torch.Tensor:
        """One full LM forward over the current sequence; returns last-position logits."""
        or_mask = None
        if is_image is not None:
            or_mask = (is_image[None, :, None] & is_image[None, None, :]).unsqueeze(1)
        logits = self.model.lm(ids, input_embeddings=h, or_mask=or_mask, logits_to_keep=1)
        return logits[:, -1, :].float()

    def _select_token(self, logits: torch.Tensor, params: SamplingParams) -> int:
        from olmo_core.generate.sampling import select_next_token

        do_sample = params.do_sample and params.temperature > 0
        return int(
            select_next_token(
                logits,
                do_sample=do_sample,
                temperature=params.temperature if do_sample else 0.0,
                top_k=params.top_k if do_sample and params.top_k is not None else -1,
                top_p=params.top_p if do_sample and params.top_p is not None else 1.0,
            )[0].item()
        )

    def _decode_loop(
        self,
        ids: torch.Tensor,
        h: torch.Tensor,
        is_image: torch.Tensor | None,
        params: SamplingParams,
    ) -> list[int]:
        if self.use_cache:
            return self._decode_loop_cached(ids, h, is_image, params)
        return self._decode_loop_uncached(ids, h, is_image, params)

    def _decode_loop_cached(
        self,
        ids: torch.Tensor,
        h: torch.Tensor,
        is_image: torch.Tensor | None,
        params: SamplingParams,
    ) -> list[int]:
        """KV-cached decode: one prefill over the prompt, then one token per step.

        Uses the cache-capable dense-SDPA backend installed by
        ``vlm_utils.enable_kv_cache``. Caches are freed afterwards so any other
        forward (e.g. ``logprobs``) runs uncached.
        """
        import torch

        or_mask = None
        if is_image is not None:
            or_mask = (is_image[None, :, None] & is_image[None, None, :]).unsqueeze(1)

        generated: list[int] = []
        vlm_utils.prepare_kv_caches(self.model, ids.shape[0], ids.shape[1] + params.max_tokens)
        try:
            logits = self.model.lm(ids, input_embeddings=h, or_mask=or_mask, logits_to_keep=1)
            logits = logits[:, -1, :].float()
            for step in range(params.max_tokens):
                next_id = self._select_token(logits, params)
                generated.append(next_id)
                if next_id in self.stop_token_ids or step == params.max_tokens - 1:
                    break
                next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                logits = self.model.lm(
                    next_tensor, input_embeddings=self._embed(next_tensor), logits_to_keep=1
                )
                logits = logits[:, -1, :].float()
        finally:
            vlm_utils.free_kv_caches(self.model)
        return generated

    def _decode_loop_uncached(
        self,
        ids: torch.Tensor,
        h: torch.Tensor,
        is_image: torch.Tensor | None,
        params: SamplingParams,
    ) -> list[int]:
        """Autoregressive decode re-running the LM over the full sequence per step."""
        import torch

        generated: list[int] = []
        for _ in range(params.max_tokens):
            logits = self._step_logits(ids, h, is_image)
            next_id = self._select_token(logits, params)
            generated.append(next_id)
            if next_id in self.stop_token_ids:
                break
            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            ids = torch.cat([ids, next_tensor], dim=1)
            h = torch.cat([h, self._embed(next_tensor)], dim=1)
            if is_image is not None:
                is_image = torch.cat(
                    [is_image, torch.zeros(1, dtype=torch.bool, device=self.device)]
                )
        return generated

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    def _text_before_stop(self, text: str, stop_sequences: tuple[str, ...] | None) -> str:
        for stop in stop_sequences or ():
            if stop and stop in text:
                text = text.split(stop)[0]
        return text

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        import torch

        params = self._default_sampling_params(sampling_params)
        if params.max_tokens <= 0:
            raise ValueError("OlmoCoreVLMProvider requires sampling max_tokens > 0")
        if params.num_samples <= 0:
            raise ValueError("OlmoCoreVLMProvider requires sampling num_samples > 0")

        results: list[list[LMOutput]] = []
        for request in requests:
            token_ids, image_tensor, pooling_tensor = self._encode_request(request)
            if is_debug_requests():
                logger.info("Prompt:\n%s", self.tokenizer.decode(token_ids))

            # Multimodal prompts cannot be truncated (that would corrupt the
            # image-token layout), so clamp the generation budget instead and
            # skip requests whose prompt alone exceeds the window.
            request_params = params
            if len(token_ids) >= self.max_length:
                logger.warning(
                    "Skipping request: prompt length (%d) >= max_length (%d)",
                    len(token_ids),
                    self.max_length,
                )
                results.append(
                    [
                        LMOutput(
                            text="",
                            logprobs=None,
                            metadata={
                                "num_tokens": 0,
                                "num_tokens_all": 0,
                                "skipped": "prompt_too_long",
                            },
                        )
                        for _ in range(params.num_samples)
                    ]
                )
                continue
            if len(token_ids) + params.max_tokens > self.max_length:
                clamped = self.max_length - len(token_ids)
                logger.warning(
                    "Clamping max_tokens from %d to %d for a %d-token prompt (max_length %d)",
                    params.max_tokens,
                    clamped,
                    len(token_ids),
                    self.max_length,
                )
                request_params = dataclasses.replace(params, max_tokens=clamped)

            request_outputs: list[LMOutput] = []
            with self._model_lock, torch.inference_mode(), self._autocast():
                ids, h, is_image = self._prefill(token_ids, image_tensor, pooling_tensor)
                for _ in range(request_params.num_samples):
                    generated = self._decode_loop(ids, h, is_image, request_params)
                    if generated and generated[-1] in self.stop_token_ids:
                        generated = generated[:-1]
                    text = self.tokenizer.decode(generated, skip_special_tokens=True)
                    text = self._text_before_stop(text, params.stop_sequences)
                    request_outputs.append(
                        LMOutput(
                            text=text.strip(),
                            logprobs=None,
                            metadata={
                                "num_tokens": len(generated),
                                "num_tokens_all": len(generated),
                            },
                        )
                    )
            results.append(request_outputs)
        return results

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        del sampling_params
        if any(request.images for request in requests):
            raise NotImplementedError(
                "OlmoCoreVLMProvider does not support loglikelihood scoring of image requests"
            )

        import torch
        import torch.nn.functional as F

        from olmo_eval.inference.tokenizer_utils import encode_context_and_continuation

        results: list[list[LMOutput]] = []
        for request in requests:
            request_outputs: list[LMOutput] = []
            cont_prompts = request.continuation_prompts
            for i, continuation in enumerate(request.continuations or ()):
                prompt = cont_prompts[i] if cont_prompts else request.prompt
                context_ids, continuation_ids = encode_context_and_continuation(
                    self.tokenizer, prompt, continuation
                )
                if not context_ids:
                    context_ids = [self.tokenizer.eos_token_id]
                full_ids = (context_ids + continuation_ids)[-(self.max_length + 1) :]
                model_input = full_ids[:-1]
                with self._model_lock, torch.inference_mode(), self._autocast():
                    ids = torch.tensor([model_input], dtype=torch.long, device=self.device)
                    logits = self.model.lm(ids, input_embeddings=self._embed(ids)).float()
                n_cont = len(continuation_ids)
                cont_logits = logits[0, -n_cont:]
                log_probs = F.log_softmax(cont_logits, dim=-1)
                cont_tensor = torch.tensor(
                    continuation_ids, dtype=torch.long, device=log_probs.device
                )
                token_log_probs = torch.gather(log_probs, 1, cont_tensor.unsqueeze(-1)).squeeze(-1)
                is_greedy = bool((log_probs.argmax(dim=-1) == cont_tensor).all().item())
                total = float(token_log_probs.sum().item())
                request_outputs.append(
                    LMOutput(
                        text=continuation,
                        logprobs=[
                            {
                                "token": self.tokenizer.decode([token_id]),
                                "logprob": float(logprob),
                            }
                            for token_id, logprob in zip(
                                continuation_ids, token_log_probs.tolist(), strict=True
                            )
                        ],
                        metadata={
                            "total_logprob": total,
                            "sum_logits": total,
                            "num_tokens": n_cont,
                            "num_tokens_all": len(full_ids),
                            "is_greedy": is_greedy,
                        },
                    )
                )
            results.append(request_outputs)
        return results

    def describe_request(
        self,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> dict[str, Any] | None:
        trace = super().describe_request(request, sampling_params)
        if trace is None:
            return None
        trace["provider"] = "OlmoCoreVLMProvider"
        if request.request_type != RequestType.LOGLIKELIHOOD:
            trace["endpoint"] = "multimodal_lm.decode_loop"
            trace["input_mode"] = "input_ids+images"
        return trace

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        return await asyncio.to_thread(self.generate, requests, sampling_params)

    async def alogprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        return await asyncio.to_thread(self.logprobs, requests, sampling_params)

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Failed to clear CUDA cache", exc_info=True)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
