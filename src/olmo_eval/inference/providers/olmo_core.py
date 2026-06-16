"""OLMo-core inference provider."""

from __future__ import annotations

import asyncio
import gc
import importlib
import json
import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

# Transformers uses int(1e30) as a sentinel when tokenizer.model_max_length is unset.
_TRANSFORMERS_UNSET_MODEL_MAX_LENGTH = int(1e30)
_EXPECTED_CHECKPOINT_FORMAT = (
    "expected a raw OLMo-core checkpoint with config.json containing 'model' and "
    "'dataset.tokenizer', plus distributed checkpoint metadata at either "
    "'model_and_optim/.metadata' or '.metadata'. HF-format checkpoints should use "
    "the 'hf', 'vllm', or 'vllm_server' provider instead"
)


@dataclass(frozen=True)
class OlmoCoreImports:
    AutoTokenizer: Any
    AttentionBackendName: Any
    GenerationConfig: Any
    TokenizerConfig: Any
    TransformerConfig: Any
    TransformerGenerationModule: Any
    cached_path: Callable[[str], Any]
    get_checkpoint_metadata: Callable[[str], Any]
    torch: Any


@dataclass(frozen=True)
class CheckpointInfo:
    config: dict[str, Any]
    tokenizer_config: Any
    metadata_dir: str | None = None


def _import_olmo_core() -> OlmoCoreImports:
    try:
        import torch
        from cached_path import cached_path
        from olmo_core.data import TokenizerConfig
        from olmo_core.distributed.checkpoint import get_checkpoint_metadata
        from olmo_core.generate.generation_module import (
            GenerationConfig,
            TransformerGenerationModule,
        )
        from olmo_core.nn.attention import AttentionBackendName
        from olmo_core.nn.transformer import TransformerConfig
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "ai2-olmo-core and transformers are required for OlmoCoreProvider. "
            "Install with: pip install 'olmo-eval[olmo_core]'"
        ) from e

    return OlmoCoreImports(
        AutoTokenizer=AutoTokenizer,
        AttentionBackendName=AttentionBackendName,
        GenerationConfig=GenerationConfig,
        TokenizerConfig=TokenizerConfig,
        TransformerConfig=TransformerConfig,
        TransformerGenerationModule=TransformerGenerationModule,
        cached_path=cached_path,
        get_checkpoint_metadata=get_checkpoint_metadata,
        torch=torch,
    )


def _is_remote_path(path: str) -> bool:
    return "://" in path


def _join_checkpoint_path(checkpoint_dir: str, *parts: str) -> str:
    if _is_remote_path(checkpoint_dir):
        return "/".join([checkpoint_dir.rstrip("/"), *parts])
    return str(Path(checkpoint_dir, *parts))


def _checkpoint_value_error(checkpoint_dir: str, reason: str) -> ValueError:
    return ValueError(
        f"Invalid OLMo-core checkpoint {checkpoint_dir!r}: {reason}. {_EXPECTED_CHECKPOINT_FORMAT}."
    )


def _read_checkpoint_config(
    checkpoint_dir: str,
    *,
    cached_path: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    config_path = _join_checkpoint_path(checkpoint_dir, "config.json")
    try:
        if _is_remote_path(checkpoint_dir):
            if cached_path is None:
                raise FileNotFoundError(config_path)
            with cached_path(config_path).open() as f:
                return json.load(f)
        with Path(config_path).open() as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise _checkpoint_value_error(checkpoint_dir, "missing config.json") from e
    except json.JSONDecodeError as e:
        raise _checkpoint_value_error(checkpoint_dir, f"config.json is not valid JSON: {e}") from e


def _validate_token_ids(
    *,
    checkpoint_dir: str,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> tuple[int, int]:
    if pad_token_id is None:
        raise _checkpoint_value_error(checkpoint_dir, "missing pad_token_id")
    if eos_token_id is None:
        raise _checkpoint_value_error(checkpoint_dir, "missing eos_token_id")
    if pad_token_id < 0:
        raise _checkpoint_value_error(
            checkpoint_dir,
            f"pad_token_id must be >= 0, got {pad_token_id}",
        )
    if eos_token_id < 0:
        raise _checkpoint_value_error(
            checkpoint_dir,
            f"eos_token_id must be >= 0, got {eos_token_id}",
        )
    if pad_token_id == eos_token_id:
        raise _checkpoint_value_error(
            checkpoint_dir,
            f"pad_token_id and eos_token_id must be different, got {pad_token_id}",
        )
    return pad_token_id, eos_token_id


def _tokenizer_config_from_checkpoint_config(
    checkpoint_dir: str,
    config: dict[str, Any],
    *,
    TokenizerConfig: Any,
) -> Any:
    try:
        return TokenizerConfig.from_dict(config["dataset"]["tokenizer"])
    except KeyError as e:
        raise _checkpoint_value_error(
            checkpoint_dir,
            f"config.json missing required field {e}",
        ) from e
    except Exception as e:
        raise _checkpoint_value_error(
            checkpoint_dir,
            f"config.json field 'dataset.tokenizer' is not a valid OLMo-core TokenizerConfig: {e}",
        ) from e


def _validate_olmo_core_checkpoint(
    checkpoint_dir: str,
    *,
    TransformerConfig: Any,
    TokenizerConfig: Any,
    get_checkpoint_metadata: Callable[[str], Any],
    cached_path: Callable[[str], Any] | None = None,
) -> CheckpointInfo:
    config = _read_checkpoint_config(checkpoint_dir, cached_path=cached_path)
    try:
        model_config = config["model"]
    except KeyError as e:
        raise _checkpoint_value_error(
            checkpoint_dir,
            f"config.json missing required field {e}",
        ) from e

    try:
        TransformerConfig.from_dict(model_config)
    except Exception as e:
        raise _checkpoint_value_error(
            checkpoint_dir,
            f"config.json field 'model' is not a valid OLMo-core TransformerConfig: {e}",
        ) from e

    tokenizer_config = _tokenizer_config_from_checkpoint_config(
        checkpoint_dir,
        config,
        TokenizerConfig=TokenizerConfig,
    )

    _validate_token_ids(
        checkpoint_dir=checkpoint_dir,
        pad_token_id=getattr(tokenizer_config, "pad_token_id", None),
        eos_token_id=getattr(tokenizer_config, "eos_token_id", None),
    )

    metadata = None
    metadata_dir = None
    metadata_errors: list[str] = []
    for candidate in (
        _join_checkpoint_path(checkpoint_dir, "model_and_optim"),
        checkpoint_dir,
    ):
        try:
            metadata = get_checkpoint_metadata(candidate)
            metadata_dir = candidate
            break
        except FileNotFoundError as e:
            metadata_errors.append(str(e))
        except Exception as e:
            raise _checkpoint_value_error(
                checkpoint_dir,
                f"could not read distributed checkpoint metadata at {candidate!r}: {e}",
            ) from e

    if metadata is None:
        detail = "; ".join(error for error in metadata_errors if error)
        reason = "missing distributed checkpoint metadata"
        if detail:
            reason = f"{reason}: {detail}"
        raise _checkpoint_value_error(checkpoint_dir, reason)

    state_metadata = getattr(metadata, "state_dict_metadata", {}) or {}
    if not any(str(key).startswith("model") for key in state_metadata):
        raise _checkpoint_value_error(
            checkpoint_dir,
            "distributed checkpoint metadata does not contain model state keys",
        )

    return CheckpointInfo(
        config=config,
        tokenizer_config=tokenizer_config,
        metadata_dir=metadata_dir,
    )


def _load_checkpoint_config_and_tokenizer_config(
    checkpoint_dir: str,
    *,
    TokenizerConfig: Any,
    cached_path: Callable[[str], Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    config = _read_checkpoint_config(checkpoint_dir, cached_path=cached_path)
    return config, _tokenizer_config_from_checkpoint_config(
        checkpoint_dir,
        config,
        TokenizerConfig=TokenizerConfig,
    )


def _valid_tokenizer_model_max_length(tokenizer: Any) -> int | None:
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if not isinstance(model_max_length, int) or model_max_length <= 0:
        return None
    if model_max_length >= _TRANSFORMERS_UNSET_MODEL_MAX_LENGTH:
        return None
    return model_max_length


def _flash_attention_3_available(torch: Any) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        importlib.import_module("flash_attn_interface")
    except Exception:
        return False
    with suppress(Exception):
        major, minor = torch.cuda.get_device_capability()
        return (9, 0) <= (major, minor) < (10, 0)
    return False


def _resolve_attention_backend(
    attention_backend: str | None,
    *,
    AttentionBackendName: Any,
    torch: Any,
) -> Any:
    if attention_backend is not None:
        return AttentionBackendName(attention_backend)
    if _flash_attention_3_available(torch):
        return AttentionBackendName("flash_3")
    return AttentionBackendName("torch")


def _resolve_max_length(
    *,
    explicit_max_length: int | None,
    tokenizer: Any,
    checkpoint_config: dict[str, Any] | None,
    checkpoint_dir: str,
) -> int:
    if explicit_max_length is not None:
        return explicit_max_length

    model_config = checkpoint_config.get("model", {}) if checkpoint_config else {}
    for key in ("max_sequence_length", "max_seq_len", "max_position_embeddings"):
        value = model_config.get(key)
        if isinstance(value, int) and value > 0:
            return value

    tokenizer_model_max_length = _valid_tokenizer_model_max_length(tokenizer)
    if tokenizer_model_max_length is not None:
        return tokenizer_model_max_length

    raise ValueError(
        "Could not determine OLMo-core max_model_len for checkpoint "
        f"{checkpoint_dir!r}: pass max_model_len explicitly, set one of "
        "model.max_sequence_length, model.max_seq_len, or model.max_position_embeddings "
        "in config.json, or use a tokenizer with a real model_max_length."
    )


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
        **kwargs: Any,
    ) -> None:
        max_length = kwargs.pop("max_length", None)
        if max_length is not None:
            if max_model_len is not None and max_model_len != max_length:
                raise ValueError(
                    "OlmoCoreProvider received both max_model_len and max_length with "
                    "different values"
                )
            max_model_len = max_length
        if max_model_len is not None and max_model_len <= 0:
            raise ValueError("OlmoCoreProvider max_model_len must be positive when set")

        if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
            raise ValueError("OlmoCoreProvider batch_size must be a positive integer or None")

        tensor_parallel_size = kwargs.pop("tensor_parallel_size", 1)
        if tensor_parallel_size not in (None, 1):
            raise ValueError(
                "OlmoCoreProvider does not support tensor_parallel_size > 1 in v1; "
                "run multiple provider processes instead."
            )

        module_kwargs: dict[str, Any] = {}
        for key in ("float8_config", "state_dict_load_opts", "load_key_mapping"):
            if key in kwargs:
                module_kwargs[key] = kwargs.pop(key)
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported OlmoCoreProvider kwargs: {unsupported}")

        imports = _import_olmo_core()
        super().__init__(model_name)

        checkpoint_info: CheckpointInfo | None = None
        checkpoint_config: dict[str, Any] | None = None
        tokenizer_config: Any | None = None
        if validate_checkpoint:
            checkpoint_info = _validate_olmo_core_checkpoint(
                model_name,
                TransformerConfig=imports.TransformerConfig,
                TokenizerConfig=imports.TokenizerConfig,
                get_checkpoint_metadata=imports.get_checkpoint_metadata,
                cached_path=imports.cached_path,
            )
            checkpoint_config = checkpoint_info.config
            tokenizer_config = checkpoint_info.tokenizer_config

        if tokenizer_config is None:
            try:
                checkpoint_config, tokenizer_config = _load_checkpoint_config_and_tokenizer_config(
                    model_name,
                    TokenizerConfig=imports.TokenizerConfig,
                    cached_path=imports.cached_path,
                )
            except ValueError:
                if not allow_tokenizer_fallback:
                    raise
                tokenizer_config = imports.TokenizerConfig.dolma2()

        tokenizer_path = tokenizer or getattr(tokenizer_config, "identifier", None)
        if tokenizer_path is None:
            if not allow_tokenizer_fallback:
                raise _checkpoint_value_error(
                    model_name,
                    "checkpoint tokenizer config does not include an identifier",
                )
            tokenizer_config = imports.TokenizerConfig.dolma2()
            tokenizer_path = tokenizer_config.identifier

        tokenizer_kwargs = {
            "revision": revision,
            "force_download": force_download,
            "trust_remote_code": trust_remote_code,
            "token": token,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        tokenizer_kwargs = {key: value for key, value in tokenizer_kwargs.items() if value}
        self.tokenizer: Any = imports.AutoTokenizer.from_pretrained(
            tokenizer_path,
            **tokenizer_kwargs,
        )
        if add_bos_token is not None:
            self.tokenizer.add_bos_token = add_bos_token

        resolved_pad_token_id, resolved_eos_token_id = _validate_token_ids(
            checkpoint_dir=model_name,
            pad_token_id=(
                pad_token_id
                if pad_token_id is not None
                else getattr(self.tokenizer, "pad_token_id", None)
                if getattr(self.tokenizer, "pad_token_id", None) is not None
                else getattr(tokenizer_config, "pad_token_id", None)
            ),
            eos_token_id=(
                eos_token_id
                if eos_token_id is not None
                else getattr(self.tokenizer, "eos_token_id", None)
                if getattr(self.tokenizer, "eos_token_id", None) is not None
                else getattr(tokenizer_config, "eos_token_id", None)
            ),
        )
        self.pad_token_id = resolved_pad_token_id
        self.eos_token_id = resolved_eos_token_id
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.chat_template = chat_template
        self.max_length = _resolve_max_length(
            explicit_max_length=max_model_len,
            tokenizer=self.tokenizer,
            checkpoint_config=checkpoint_config,
            checkpoint_dir=model_name,
        )

        self.device = imports.torch.device(
            device or ("cuda" if imports.torch.cuda.is_available() else "cpu")
        )
        attention_backend_value = _resolve_attention_backend(
            attention_backend,
            AttentionBackendName=imports.AttentionBackendName,
            torch=imports.torch,
        )

        self.generation_config = imports.GenerationConfig(
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            use_cache=use_cache,
        )

        load_kwargs: dict[str, Any] = {
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
        load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
        self.generation_module = imports.TransformerGenerationModule.from_checkpoint(**load_kwargs)

    def get_tokenizer(self) -> Any:
        return self.tokenizer

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
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.chat_template is not None:
                kwargs["chat_template"] = self.chat_template
            return self.tokenizer.apply_chat_template(list(request.messages), **kwargs)
        return request.prompt

    def _left_pad(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        max_len = max(max((len(seq) for seq in sequences), default=0), 1)
        input_ids = torch.full(
            (len(sequences), max_len),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (len(sequences), max_len),
            dtype=torch.long,
            device=self.device,
        )
        for idx, seq in enumerate(sequences):
            if not seq:
                continue
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device)
            input_ids[idx, -len(seq) :] = seq_tensor
            attention_mask[idx, -len(seq) :] = 1
        return input_ids, attention_mask

    def _right_pad(self, sequences: list[list[int]]) -> torch.Tensor:
        import torch

        max_len = max(max((len(seq) for seq in sequences), default=0), 1)
        input_ids = torch.full(
            (len(sequences), max_len),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        for idx, seq in enumerate(sequences):
            if not seq:
                continue
            input_ids[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
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

    def _build_generation_kwargs(self, params: SamplingParams) -> dict[str, Any]:
        if params.max_tokens <= 0:
            raise ValueError("OlmoCoreProvider requires sampling max_tokens > 0")
        if params.num_samples <= 0:
            raise ValueError("OlmoCoreProvider requires sampling num_samples > 0")

        do_sample = params.do_sample and params.temperature > 0
        kwargs: dict[str, Any] = {
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

    def _truncate_generation(
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

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        if not requests:
            return []

        params = self._default_sampling_params(sampling_params)
        results: list[list[LMOutput]] = []
        for chunk in self._iter_chunks(requests):
            results.extend(self._generate_chunk(chunk, params))
        self.generation_module.free_inference_cache()
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

        generated_rows = generated_ids.tolist()
        logprob_rows = logprobs.tolist() if logprobs is not None else [None] * len(generated_rows)

        results = [[] for _ in requests]
        for row_idx, (row_ids, row_logprobs) in enumerate(
            zip(generated_rows, logprob_rows, strict=True)
        ):
            request_idx = row_idx // params.num_samples
            token_ids, token_logprobs, text = self._truncate_generation(
                row_ids,
                row_logprobs,
                params.stop_sequences,
            )

            entries = (
                self._logprob_entries(token_ids, token_logprobs)
                if token_logprobs is not None
                else None
            )
            metadata: dict[str, Any] = {}
            if entries:
                total = sum(entry["logprob"] for entry in entries)
                metadata = {
                    "sum_logits": total,
                    "num_tokens": len(entries),
                    "num_tokens_all": len(entries),
                }

            results[request_idx].append(
                LMOutput(
                    text=text,
                    logprobs=entries,
                    metadata=metadata,
                )
            )
        return results

    def describe_request(
        self,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> dict[str, Any] | None:
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

        results: list[list[LMOutput]] = []
        for chunk in self._iter_chunks(requests):
            results.extend(self._logprobs_chunk(chunk))
        self.generation_module.free_inference_cache()
        return results

    def _logprobs_chunk(self, requests: list[LMRequest]) -> list[list[LMOutput]]:
        import torch
        import torch.nn.functional as F

        token_inputs: list[list[int]] = []
        request_meta: list[tuple[int, int, list[int], str]] = []

        for request in requests:
            max_len = request.max_length or self.max_length
            if max_len <= 0:
                raise ValueError("OlmoCoreProvider requires max_length > 0 for logprobs")

            cont_prompts = request.continuation_prompts
            for i, continuation in enumerate(request.continuations or ()):
                prompt = cont_prompts[i] if cont_prompts else request.prompt
                context_enc, continuation_enc = encode_context_and_continuation(
                    self.tokenizer,
                    prompt,
                    continuation,
                )
                if not continuation_enc:
                    request_meta.append((0, len(context_enc), [], continuation))
                    token_inputs.append(context_enc or [self.eos_token_id])
                    continue
                if len(continuation_enc) > max_len:
                    raise ValueError(
                        "Continuation is longer than the OLMo-core provider max_length "
                        f"({len(continuation_enc)} > {max_len})"
                    )

                full_ids = context_enc + continuation_enc
                truncated = full_ids[-(max_len + 1) :]
                model_input = truncated[:-1]
                input_length = len(model_input)
                token_inputs.append(model_input)
                request_meta.append(
                    (
                        input_length,
                        len(truncated),
                        continuation_enc,
                        continuation,
                    )
                )

        if token_inputs:
            batched_inputs = self._right_pad(token_inputs)
            with torch.no_grad():
                batch_logits = self.generation_module.model_forward(input_ids=batched_inputs)
        else:
            batch_logits = []

        output_iter = iter(batch_logits)
        meta_iter = iter(request_meta)
        results: list[list[LMOutput]] = []

        for request in requests:
            request_outputs: list[LMOutput] = []
            for _ in request.continuations or ():
                input_length, num_tokens_all, continuation_enc, continuation = next(meta_iter)
                logits = next(output_iter)

                if not continuation_enc:
                    request_outputs.append(
                        LMOutput(
                            text=continuation,
                            logprobs=[],
                            metadata={
                                "total_logprob": 0.0,
                                "sum_logits": 0.0,
                                "num_tokens": 0,
                                "num_tokens_all": num_tokens_all,
                                "is_greedy": True,
                            },
                        )
                    )
                    continue

                continuation_length = len(continuation_enc)
                continuation_logits = logits[
                    input_length - continuation_length : input_length
                ].float()
                log_probs = F.log_softmax(continuation_logits, dim=-1)
                continuation_tensor = torch.tensor(
                    continuation_enc,
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

                token_logprob_values = token_log_probs.tolist()
                entries = self._logprob_entries(continuation_enc, token_logprob_values)
                total = float(sum(token_logprob_values))
                request_outputs.append(
                    LMOutput(
                        text=continuation,
                        logprobs=entries,
                        metadata={
                            "total_logprob": total,
                            "sum_logits": total,
                            "num_tokens": len(entries),
                            "num_tokens_all": num_tokens_all,
                            "is_greedy": is_greedy,
                        },
                    )
                )

            results.append(request_outputs)

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
                self.generation_module.free_inference_cache()
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
