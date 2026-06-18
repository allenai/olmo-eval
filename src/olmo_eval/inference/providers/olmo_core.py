"""OLMo-core inference provider."""

from __future__ import annotations

import asyncio
import gc
import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Literal, cast

import olmo_eval.inference.providers.olmo_core_utils as core_utils
from olmo_eval.common.debug import is_debug_requests
from olmo_eval.common.types import LMOutput, LMRequest, LogProbEntry, RequestType, SamplingParams
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.tokenizer_utils import (
    encode_context_and_continuation,
    get_bos_token_ids,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class OlmoCoreProvider(InferenceProvider):
    """Provider using OLMo-core's in-process generation module."""

    def __init__(
        self,
        model_name: str,
        tokenizer: str | None = None,
        *,
        dtype: str = "auto",
        max_model_len: int | None = None,
        attention_backend: str | None = None,
        use_cache: bool = True,
        retain_inference_cache: bool = False,
        compile_model: bool = False,
        batch_size: int | None = None,
        validate_checkpoint: bool = True,
        allow_tokenizer_fallback: bool = False,
        chat_template: str | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        device: str | None = None,
        load_thread_count: int | None = None,
        pre_download: bool = True,
        work_dir: str | None = None,
        revision: str | None = None,
        force_download: bool = False,
        trust_remote_code: bool = False,
        token: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        add_bos_token: bool | None = None,
        **kwargs: object,
    ) -> None:
        max_model_len = core_utils._resolve_max_model_len_alias(max_model_len, kwargs)
        core_utils._validate_max_model_len(max_model_len)
        core_utils._validate_batch_size(batch_size)
        core_utils._validate_tensor_parallel_size(kwargs.pop("tensor_parallel_size", 1))

        module_kwargs = core_utils._pop_module_kwargs(kwargs)
        core_utils._raise_for_unsupported_kwargs(kwargs)

        imports = core_utils._import_olmo_core()
        super().__init__(model_name)

        checkpoint_config, tokenizer_config = core_utils._resolve_checkpoint(
            model_name,
            imports=imports,
            validate_checkpoint=validate_checkpoint,
            allow_tokenizer_fallback=allow_tokenizer_fallback,
        )
        tokenizer_path, tokenizer_config = core_utils._resolve_tokenizer_path(
            model_name,
            explicit_tokenizer=tokenizer,
            tokenizer_config=tokenizer_config,
            TokenizerConfig=imports.TokenizerConfig,
            allow_tokenizer_fallback=allow_tokenizer_fallback,
        )

        tokenizer_kwargs = {
            key: value
            for key, value in {
                "revision": revision,
                "force_download": force_download,
                "trust_remote_code": trust_remote_code,
                "token": token,
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
            }.items()
            if value
        }
        self.tokenizer: core_utils.TokenizerProtocol = imports.AutoTokenizer.from_pretrained(
            tokenizer_path,
            **tokenizer_kwargs,
        )
        if add_bos_token is not None:
            self.tokenizer.add_bos_token = add_bos_token

        resolved_pad_token_id, resolved_eos_token_id = core_utils._validate_token_ids(
            checkpoint_dir=model_name,
            pad_token_id=core_utils._preferred_token_id(
                pad_token_id,
                tokenizer=self.tokenizer,
                tokenizer_config=tokenizer_config,
                attr="pad_token_id",
            ),
            eos_token_id=core_utils._preferred_token_id(
                eos_token_id,
                tokenizer=self.tokenizer,
                tokenizer_config=tokenizer_config,
                attr="eos_token_id",
            ),
        )
        self.pad_token_id = resolved_pad_token_id
        self.eos_token_id = resolved_eos_token_id
        self.use_cache = use_cache
        self.retain_inference_cache = retain_inference_cache
        self._inference_cache_retained = False
        self.batch_size = batch_size
        self.chat_template = chat_template
        self.max_length = core_utils._resolve_max_length(
            explicit_max_length=max_model_len,
            tokenizer=self.tokenizer,
            checkpoint_config=checkpoint_config,
            checkpoint_dir=model_name,
        )

        self.device = imports.torch.device(
            device or ("cuda" if imports.torch.cuda.is_available() else "cpu")
        )
        attention_backend_value = core_utils._resolve_attention_backend(
            attention_backend,
            AttentionBackendName=imports.AttentionBackendName,
            torch=imports.torch,
        )

        self.generation_config = imports.GenerationConfig(
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            use_cache=use_cache,
        )

        load_kwargs = {
            "checkpoint_dir": model_name,
            "generation_config": self.generation_config,
            "device": self.device,
            "pre_download": pre_download,
            "load_thread_count": load_thread_count,
            "work_dir": work_dir,
            "attention_backend": attention_backend_value,
            "compile_model": compile_model,
            **module_kwargs,
        }
        if dtype != "auto":
            load_kwargs["dtype"] = dtype
        self.generation_module: core_utils.GenerationModuleProtocol = (
            imports.TransformerGenerationModule.from_checkpoint(
                **{key: value for key, value in load_kwargs.items() if value is not None}
            )
        )

    def get_tokenizer(self) -> core_utils.TokenizerProtocol:
        return self.tokenizer

    def _free_inference_cache(self) -> None:
        self.generation_module.free_inference_cache()
        self._inference_cache_retained = False

    def _iter_chunks(self, requests: list[LMRequest]) -> list[list[LMRequest]]:
        if self.batch_size is None:
            return [requests]
        return [
            requests[start : start + self.batch_size]
            for start in range(0, len(requests), self.batch_size)
        ]

    def _encode_prompt(self, prompt: str) -> list[int]:
        if prompt == "":
            return get_bos_token_ids(self.tokenizer, fallback_to_eos=True)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if getattr(self.tokenizer, "add_bos_token", False):
            return get_bos_token_ids(self.tokenizer, fallback_to_eos=True) + token_ids
        return token_ids

    def _format_prompt(self, request: LMRequest) -> str:
        if request.request_type == RequestType.CHAT and request.messages:
            if not hasattr(self.tokenizer, "apply_chat_template"):
                raise ValueError("CHAT requests require a tokenizer with apply_chat_template")
            kwargs: dict[str, object] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.chat_template is not None:
                kwargs["chat_template"] = self.chat_template
            return self.tokenizer.apply_chat_template(list(request.messages), **kwargs)
        return request.prompt

    def _pad_sequences(
        self,
        sequences: list[list[int]],
        *,
        side: Literal["left", "right"],
        return_attention_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        import torch

        max_len = max(max((len(seq) for seq in sequences), default=0), 1)
        input_ids = torch.full(
            (len(sequences), max_len),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = (
            torch.zeros(
                (len(sequences), max_len),
                dtype=torch.long,
                device=self.device,
            )
            if return_attention_mask
            else None
        )
        for idx, seq in enumerate(sequences):
            if not seq:
                continue
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device)
            target = slice(-len(seq), None) if side == "left" else slice(0, len(seq))
            input_ids[idx, target] = seq_tensor
            if attention_mask is not None:
                attention_mask[idx, target] = 1
        return input_ids, attention_mask

    def _left_pad(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask = self._pad_sequences(
            sequences,
            side="left",
            return_attention_mask=True,
        )
        assert attention_mask is not None
        return input_ids, attention_mask

    def _right_pad(self, sequences: list[list[int]]) -> torch.Tensor:
        input_ids, _ = self._pad_sequences(sequences, side="right")
        return input_ids

    def _stop_token_ids(self, stop_sequences: tuple[str, ...] | None) -> list[int]:
        if not stop_sequences:
            return []

        stop_token_ids: list[int] = []
        for stop in stop_sequences:
            token_ids = self.tokenizer.encode(stop, add_special_tokens=False)
            if len(token_ids) == 1:
                stop_token_ids.append(token_ids[0])
        return sorted(set(stop_token_ids))

    def _build_generation_kwargs(self, params: SamplingParams) -> dict[str, object]:
        if params.max_tokens <= 0:
            raise ValueError("OlmoCoreProvider requires sampling max_tokens > 0")
        if params.num_samples <= 0:
            raise ValueError("OlmoCoreProvider requires sampling num_samples > 0")

        do_sample = params.do_sample and params.temperature > 0
        kwargs: dict[str, object] = {
            "max_new_tokens": params.max_tokens,
            "do_sample": do_sample,
            "temperature": params.temperature if do_sample else 0.0,
            "top_k": params.top_k if do_sample and params.top_k is not None else -1,
            "top_p": params.top_p if do_sample and params.top_p is not None else 1.0,
            "use_cache": self.use_cache,
        }
        stop_token_ids = self._stop_token_ids(params.stop_sequences)
        if stop_token_ids:
            kwargs["stop_token_ids"] = stop_token_ids
        return kwargs

    def _decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def _validate_generation_lengths(
        self,
        prompt_token_ids: list[list[int]],
        params: SamplingParams,
    ) -> None:
        for request_idx, token_ids in enumerate(prompt_token_ids):
            requested_length = len(token_ids) + params.max_tokens
            if requested_length > self.max_length:
                raise ValueError(
                    "OLMo-core generation request exceeds max_length for request "
                    f"{request_idx}: prompt length ({len(token_ids)}) + max_tokens "
                    f"({params.max_tokens}) = {requested_length}, but max_length is "
                    f"{self.max_length}."
                )

    def _normalize_generation_output(
        self,
        token_ids: list[int],
        token_logprobs: list[float] | None,
        stop_sequences: tuple[str, ...] | None,
    ) -> tuple[list[int], list[float] | None, str]:
        stop_token_ids = set(self._stop_token_ids(stop_sequences))
        hard_stop_ids = {self.eos_token_id, self.pad_token_id, *stop_token_ids}

        end = len(token_ids)
        for idx, token_id in enumerate(token_ids):
            if token_id in hard_stop_ids:
                end = idx if token_id == self.pad_token_id else idx + 1
                break

        token_ids = token_ids[:end]
        if token_logprobs is not None:
            token_logprobs = token_logprobs[:end]

        if stop_sequences:
            for idx in range(len(token_ids)):
                decoded = self._decode(token_ids[: idx + 1], skip_special_tokens=True)
                for stop in stop_sequences:
                    if stop and stop in decoded:
                        token_ids = token_ids[: idx + 1]
                        if token_logprobs is not None:
                            token_logprobs = token_logprobs[: idx + 1]
                        return token_ids, token_logprobs, decoded.split(stop, 1)[0]

        return token_ids, token_logprobs, self._decode(token_ids, skip_special_tokens=True)

    def _logprob_entries(
        self,
        token_ids: list[int],
        token_logprobs: list[float],
    ) -> list[LogProbEntry]:
        entries: list[LogProbEntry] = []
        for token_id, logprob in zip(token_ids, token_logprobs, strict=True):
            token_str = self._decode([token_id], skip_special_tokens=False)
            entries.append(
                {
                    "token": token_str,
                    "logprob": float(logprob),
                    "bytes": list(token_str.encode("utf-8")),
                }
            )
        return entries

    def _generation_output(
        self,
        token_ids: list[int],
        token_logprobs: list[float] | None,
        text: str,
    ) -> LMOutput:
        entries = (
            self._logprob_entries(token_ids, token_logprobs) if token_logprobs is not None else None
        )
        metadata: dict[str, object] = {}
        if entries:
            total = sum(entry["logprob"] for entry in entries)
            metadata = {
                "sum_logits": total,
                "num_tokens": len(entries),
                "num_tokens_all": len(entries),
            }
        return LMOutput(
            text=text,
            logprobs=entries,
            metadata=metadata,
        )

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        if not requests:
            return []

        params = self._default_sampling_params(sampling_params)
        results: list[list[LMOutput]] = []
        try:
            for chunk in self._iter_chunks(requests):
                results.extend(self._generate_chunk(chunk, params))
        except Exception:
            self._free_inference_cache()
            raise

        if self.retain_inference_cache and self.use_cache:
            self._inference_cache_retained = True
        else:
            self._free_inference_cache()
        return results

    def _generate_chunk(
        self,
        requests: list[LMRequest],
        params: SamplingParams,
    ) -> list[list[LMOutput]]:
        prompt_strs = [self._format_prompt(request) for request in requests]
        if is_debug_requests():
            for i, prompt in enumerate(prompt_strs):
                logger.info(f"Prompt {i}:\n{prompt}")

        generation_kwargs = self._build_generation_kwargs(params)
        prompt_token_ids = [self._encode_prompt(prompt) for prompt in prompt_strs]
        self._validate_generation_lengths(prompt_token_ids, params)
        expanded_token_ids = [
            token_ids for token_ids in prompt_token_ids for _ in range(params.num_samples)
        ]
        input_ids, attention_mask = self._left_pad(expanded_token_ids)

        generated_ids, _, logprobs = self.generation_module.generate_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_logprobs=True,
            completions_only=True,
            log_timing=False,
            **generation_kwargs,
        )

        generated_rows = cast(list[list[int]], generated_ids.tolist())
        logprob_rows = (
            cast(list[list[float]], logprobs.tolist())
            if logprobs is not None
            else [None] * len(generated_rows)
        )

        results = [[] for _ in requests]
        for row_idx, (row_ids, row_logprobs) in enumerate(
            zip(generated_rows, logprob_rows, strict=True)
        ):
            request_idx = row_idx // params.num_samples
            token_ids, token_logprobs, text = self._normalize_generation_output(
                row_ids,
                row_logprobs,
                params.stop_sequences,
            )
            results[request_idx].append(self._generation_output(token_ids, token_logprobs, text))
        return results

    def describe_request(
        self,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> dict[str, object] | None:
        params = self._default_sampling_params(sampling_params)
        trace = super().describe_request(request, sampling_params)
        if trace is None:
            return None

        trace["provider"] = "OlmoCoreProvider"
        if request.request_type == RequestType.LOGLIKELIHOOD:
            trace["endpoint"] = "generation_module.model_forward"
            trace["input_mode"] = "input_ids"
            trace["generation_kwargs"] = {
                "temperature": params.temperature,
            }
            trace["stop_sequences"] = []
            return trace

        trace["endpoint"] = "generation_module.generate_batch"
        trace["input_mode"] = "input_ids"
        trace["generation_kwargs"] = {
            **self._build_generation_kwargs(params),
            "num_samples": params.num_samples,
            "return_logprobs": True,
            "completions_only": True,
        }
        trace["stop_sequences"] = list(params.stop_sequences or ())
        return trace

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        del sampling_params
        if not requests:
            return []

        self._free_inference_cache()
        results: list[list[LMOutput]] = []
        for chunk in self._iter_chunks(requests):
            results.extend(self._logprobs_chunk(chunk))
        return results

    def _logprob_inputs_for_request(self, request: LMRequest) -> list[core_utils.LogprobInput]:
        max_len = request.max_length or self.max_length
        if max_len <= 0:
            raise ValueError("OlmoCoreProvider requires max_length > 0 for logprobs")

        rows: list[core_utils.LogprobInput] = []
        cont_prompts = request.continuation_prompts
        for i, continuation in enumerate(request.continuations or ()):
            prompt = cont_prompts[i] if cont_prompts else request.prompt
            context_ids, continuation_ids = encode_context_and_continuation(
                self.tokenizer,
                prompt,
                continuation,
            )
            if not continuation_ids:
                rows.append(
                    core_utils.LogprobInput(
                        input_ids=context_ids or [self.eos_token_id],
                        input_length=0,
                        num_tokens_all=len(context_ids),
                        continuation_token_ids=[],
                        continuation=continuation,
                    )
                )
                continue
            if len(continuation_ids) > max_len:
                raise ValueError(
                    "Continuation is longer than the OLMo-core provider max_length "
                    f"({len(continuation_ids)} > {max_len})"
                )

            truncated = (context_ids + continuation_ids)[-(max_len + 1) :]
            model_input = truncated[:-1]
            rows.append(
                core_utils.LogprobInput(
                    input_ids=model_input,
                    input_length=len(model_input),
                    num_tokens_all=len(truncated),
                    continuation_token_ids=continuation_ids,
                    continuation=continuation,
                )
            )
        return rows

    def _logprob_output_from_logits(
        self,
        row: core_utils.LogprobInput,
        logits: torch.Tensor,
    ) -> LMOutput:
        if not row.continuation_token_ids:
            return LMOutput(
                text=row.continuation,
                logprobs=[],
                metadata={
                    "total_logprob": 0.0,
                    "sum_logits": 0.0,
                    "num_tokens": 0,
                    "num_tokens_all": row.num_tokens_all,
                    "is_greedy": True,
                },
            )

        import torch
        import torch.nn.functional as F

        continuation_length = len(row.continuation_token_ids)
        continuation_logits = logits[
            row.input_length - continuation_length : row.input_length
        ].float()
        log_probs = F.log_softmax(continuation_logits, dim=-1)
        continuation_tensor = torch.tensor(
            row.continuation_token_ids,
            dtype=torch.long,
            device=log_probs.device,
        )
        token_log_probs = torch.gather(
            log_probs,
            1,
            continuation_tensor.unsqueeze(-1),
        ).squeeze(-1)
        greedy_tokens = log_probs.argmax(dim=-1)
        is_greedy = bool((greedy_tokens == continuation_tensor).all().item())

        token_logprob_values = cast(list[float], token_log_probs.tolist())
        entries = self._logprob_entries(row.continuation_token_ids, token_logprob_values)
        total = float(sum(token_logprob_values))
        return LMOutput(
            text=row.continuation,
            logprobs=entries,
            metadata={
                "total_logprob": total,
                "sum_logits": total,
                "num_tokens": len(entries),
                "num_tokens_all": row.num_tokens_all,
                "is_greedy": is_greedy,
            },
        )

    def _logprobs_chunk(self, requests: list[LMRequest]) -> list[list[LMOutput]]:
        import torch

        rows_by_request = [self._logprob_inputs_for_request(request) for request in requests]
        token_inputs = [row.input_ids for rows in rows_by_request for row in rows]

        if token_inputs:
            batched_inputs = self._right_pad(token_inputs)
            with torch.no_grad():
                batch_logits = self.generation_module.model_forward(input_ids=batched_inputs)
        else:
            batch_logits = []

        output_iter = iter(batch_logits)
        results: list[list[LMOutput]] = []

        for rows in rows_by_request:
            results.append(
                [self._logprob_output_from_logits(row, next(output_iter)) for row in rows]
            )

        return results

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
        if hasattr(self, "generation_module"):
            try:
                self._free_inference_cache()
            except Exception:
                logger.debug("Failed to free OLMo-core inference cache", exc_info=True)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Failed to clear CUDA cache", exc_info=True)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()
