"""Hugging Face Transformers provider."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.tokenizer_utils import encode_context_and_continuation

if TYPE_CHECKING:
    import torch  # type: ignore[ty:unresolved-import]

T = TypeVar("T")
R = TypeVar("R")


def _get_device() -> torch.device:
    """Detect the best available device."""
    import torch  # type: ignore[ty:unresolved-import]

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HuggingFaceProvider(InferenceProvider):
    """Provider using Hugging Face Transformers for local inference."""

    def __init__(self, model_name: str, tokenizer: str | None = None, **model_kwargs) -> None:
        """Initialize the provider.

        Args:
            model_name: HuggingFace model identifier or local path.
            tokenizer: Tokenizer path/identifier. If not specified, uses the model path.
            **model_kwargs: Additional arguments passed to from_pretrained.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for HuggingFaceProvider. "
                "Install with: pip install transformers"
            ) from e

        super().__init__(model_name)
        self.max_batch_size = _coerce_optional_int(model_kwargs.pop("max_batch_size", None))
        self.max_generate_batch_size = _coerce_optional_int(
            model_kwargs.pop("max_generate_batch_size", None)
        )
        self.max_score_batch_size = _coerce_optional_int(
            model_kwargs.pop("max_score_batch_size", None)
        )
        tokenizer_path = tokenizer or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.device = _get_device()
        self.model.to(self.device)
        self.model.eval()

        # Ensure pad token is set for batched inference
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def get_tokenizer(self) -> Any:
        """Get the tokenizer for this provider."""
        return self.tokenizer

    def _clear_device_cache(self) -> None:
        """Release cached CUDA memory after a failed oversized batch."""
        import torch  # type: ignore[ty:unresolved-import]

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _should_retry_with_smaller_batch(self, exc: Exception) -> bool:
        """Whether a batch failure is worth retrying with smaller microbatches."""
        import torch  # type: ignore[ty:unresolved-import]

        if hasattr(torch, "OutOfMemoryError") and isinstance(exc, torch.OutOfMemoryError):
            return True

        message = str(exc).lower()
        if "out of memory" in message:
            return True

        if not isinstance(exc, RuntimeError):
            return False

        retry_markers = (
            "size of tensor a",
            "must match the size of tensor",
            "must be equal to the size of tensor",
            "non-singleton dimension",
        )
        return any(marker in message for marker in retry_markers)

    def _call_with_adaptive_splitting(
        self,
        items: list[T],
        fn: Callable[[list[T]], list[R]],
        *,
        max_batch_size: int | None = None,
    ) -> list[R]:
        """Execute `fn` and recursively split retryable oversized batches on failure."""
        if not items:
            return []

        if max_batch_size is not None and max_batch_size > 0 and len(items) > max_batch_size:
            results: list[R] = []
            for start in range(0, len(items), max_batch_size):
                results.extend(
                    self._call_with_adaptive_splitting(
                        items[start : start + max_batch_size],
                        fn,
                        max_batch_size=max_batch_size,
                    )
                )
            return results

        try:
            return fn(items)
        except Exception as exc:
            if len(items) == 1 or not self._should_retry_with_smaller_batch(exc):
                raise

            self._clear_device_cache()
            midpoint = len(items) // 2
            return self._call_with_adaptive_splitting(
                items[:midpoint],
                fn,
                max_batch_size=max_batch_size,
            ) + self._call_with_adaptive_splitting(
                items[midpoint:],
                fn,
                max_batch_size=max_batch_size,
            )

    def _get_generate_batch_limit(self, params: SamplingParams) -> int | None:
        """Choose a safer initial microbatch size for long generations."""
        if self.max_generate_batch_size is not None:
            return self.max_generate_batch_size
        if self.max_batch_size is not None:
            return self.max_batch_size

        if params.max_tokens >= 1024:
            limit = 4
        elif params.max_tokens >= 512:
            limit = 8
        elif params.max_tokens >= 256:
            limit = 16
        else:
            return None

        if params.num_samples > 1:
            limit = max(1, limit // min(params.num_samples, 4))

        return limit

    def _get_score_batch_limit(self, sequences: list[list[int]]) -> int | None:
        """Choose a safer initial microbatch size for full-sequence scoring."""
        if not sequences:
            return None
        if self.max_score_batch_size is not None:
            return self.max_score_batch_size
        if self.max_batch_size is not None:
            return self.max_batch_size

        longest = max(len(sequence) for sequence in sequences)
        if longest >= 4096:
            return 1
        if longest >= 2048:
            return 2
        if longest >= 1024:
            return 4
        if longest >= 512:
            return 8
        return None

    def _build_generate_kwargs(self, params: SamplingParams) -> dict[str, Any]:
        """Convert SamplingParams to HuggingFace generate kwargs."""
        # Use explicit do_sample flag, overriding temperature-based inference
        do_sample = params.do_sample and params.temperature > 0

        kwargs: dict[str, Any] = {
            "max_new_tokens": params.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if do_sample:
            if params.temperature > 0:
                kwargs["temperature"] = params.temperature
            if params.top_p is not None:
                kwargs["top_p"] = params.top_p
            if params.top_k is not None:
                kwargs["top_k"] = params.top_k

        return kwargs

    def _tokenize_prompts_for_generation(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize prompts with left padding for batched decoder-only generation."""
        padding_side = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = "left"
            encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        finally:
            self.tokenizer.padding_side = padding_side

        return encoded.to(self.device)

    def _pad_token_sequences(
        self,
        sequences: list[list[int]],
        *,
        padding_side: str = "right",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad token ID sequences and build an attention mask."""
        import torch  # type: ignore[ty:unresolved-import]

        max_len = max(len(sequence) for sequence in sequences)
        pad_token_id = self.tokenizer.pad_token_id

        input_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        for sequence in sequences:
            pad_len = max_len - len(sequence)
            if padding_side == "left":
                input_rows.append(([pad_token_id] * pad_len) + sequence)
                mask_rows.append(([0] * pad_len) + ([1] * len(sequence)))
            else:
                input_rows.append(sequence + ([pad_token_id] * pad_len))
                mask_rows.append(([1] * len(sequence)) + ([0] * pad_len))

        return (
            torch.tensor(input_rows, device=self.device),
            torch.tensor(mask_rows, device=self.device),
        )

    def _get_generated_sequence_length(self, generated_token_ids: torch.Tensor) -> int:
        """Infer the number of generated tokens before EOS/padding."""
        tokens = [int(token_id) for token_id in generated_token_ids.tolist()]
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        if eos_token_id is not None and eos_token_id in tokens:
            return tokens.index(eos_token_id) + 1

        length = len(tokens)
        if pad_token_id is not None:
            while length > 0 and tokens[length - 1] == pad_token_id:
                length -= 1

        return length

    def _score_token_sequences(
        self,
        jobs: list[tuple[int, list[int], int, list[int]]],
    ) -> list[tuple[int, list[dict[str, Any]], float]]:
        """Score target tokens inside full token sequences."""
        import torch  # type: ignore[ty:unresolved-import]

        sequences = [sequence for _, sequence, _, _ in jobs]
        input_ids, attention_mask = self._pad_token_sequences(sequences, padding_side="right")

        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits

        log_probs = torch.log_softmax(logits, dim=-1)
        scored: list[tuple[int, list[dict[str, Any]], float]] = []

        for row_idx, (item_idx, _, target_start, target_tokens) in enumerate(jobs):
            logprob_entries = []
            total = 0.0
            for token_offset, tok in enumerate(target_tokens):
                lp = log_probs[row_idx, target_start + token_offset - 1, tok].item()
                token_str = self.tokenizer.decode(tok, skip_special_tokens=False)
                logprob_entries.append(
                    {
                        "token": token_str,
                        "logprob": lp,
                        "bytes": list(token_str.encode("utf-8")),
                    }
                )
                total += lp

            scored.append((item_idx, logprob_entries, total))

        return scored

    def _generate_token_batch(
        self,
        requests: list[tuple[int, LMRequest]],
        *,
        gen_kwargs: dict[str, Any],
        stop_sequences: tuple[str, ...] | None,
    ) -> list[tuple[int, str, list[int], list[int]]]:
        """Generate one sample for a request microbatch."""
        import torch  # type: ignore[ty:unresolved-import]

        encoded = self._tokenize_prompts_for_generation([request.prompt for _, request in requests])
        prompt_width = encoded["input_ids"].shape[1]
        prompt_token_ids = [
            encoded["input_ids"][request_idx][encoded["attention_mask"][request_idx].bool()].tolist()
            for request_idx in range(len(requests))
        ]

        with torch.no_grad():
            generation = self.model.generate(**encoded, **gen_kwargs, return_dict_in_generate=True)

        generated_token_ids = generation.sequences[:, prompt_width:]
        outputs: list[tuple[int, str, list[int], list[int]]] = []

        for batch_idx, ((request_idx, _), prompt_ids) in enumerate(
            zip(requests, prompt_token_ids, strict=True)
        ):
            gen_len = self._get_generated_sequence_length(generated_token_ids[batch_idx])
            gen_ids = generated_token_ids[batch_idx, :gen_len]
            gen_ids, text = self._truncate_at_stop(gen_ids, stop_sequences)
            outputs.append((request_idx, text, prompt_ids, [int(tok) for tok in gen_ids.tolist()]))

        return outputs

    def _truncate_at_stop(
        self, tokens: torch.Tensor, stop_sequences: tuple[str, ...] | None
    ) -> tuple[torch.Tensor, str]:
        """Truncate generated tokens at first stop sequence."""
        if not stop_sequences:
            return tokens, self.tokenizer.decode(tokens, skip_special_tokens=True)

        decoded_parts: list[str] = []
        for idx, token in enumerate(tokens):
            decoded_parts.append(self.tokenizer.decode(token, skip_special_tokens=True))
            decoded = "".join(decoded_parts)
            for stop in stop_sequences:
                if stop in decoded:
                    return tokens[: idx + 1], decoded.split(stop)[0]

        return tokens, "".join(decoded_parts)

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        if not requests:
            return []

        params = self._default_sampling_params(sampling_params)
        gen_kwargs = self._build_generate_kwargs(params)
        results: list[list[LMOutput]] = [[] for _ in requests]
        indexed_requests = list(enumerate(requests))
        generate_batch_limit = self._get_generate_batch_limit(params)

        for _ in range(params.num_samples):
            generated_outputs = self._call_with_adaptive_splitting(
                indexed_requests,
                lambda batch: self._generate_token_batch(
                    batch,
                    gen_kwargs=gen_kwargs,
                    stop_sequences=params.stop_sequences,
                ),
                max_batch_size=generate_batch_limit,
            )

            score_sequences: list[list[int]] = []
            score_jobs: list[tuple[int, list[int], int, list[int]]] = []

            for request_idx, _, prompt_ids, generated_ids in generated_outputs:
                if generated_ids:
                    full_sequence = prompt_ids + generated_ids
                    score_sequences.append(full_sequence)
                    score_jobs.append((request_idx, full_sequence, len(prompt_ids), generated_ids))

            scored_outputs: dict[int, tuple[list[dict[str, Any]], float]] = {}
            if score_jobs:
                scored_batch_limit = self._get_score_batch_limit(score_sequences)
                scored_outputs = {
                    item_idx: (entries, total)
                    for item_idx, entries, total in self._call_with_adaptive_splitting(
                        score_jobs,
                        self._score_token_sequences,
                        max_batch_size=scored_batch_limit,
                    )
                }

            for request_idx, text, _, generated_ids in generated_outputs:
                logprob_entries = None
                metadata: dict[str, Any] = {}
                if generated_ids:
                    logprob_entries, total = scored_outputs[request_idx]
                    num_tokens = len(logprob_entries)
                    metadata = {
                        "sum_logits": total,
                        "num_tokens": num_tokens,
                        "num_tokens_all": num_tokens,
                    }

                results[request_idx].append(
                    LMOutput(text=text, logprobs=logprob_entries, metadata=metadata)
                )

        return results

    def logprobs(
        self,
        requests: list[LMRequest],
    ) -> list[list[LMOutput]]:
        results: list[list[LMOutput]] = [[] for _ in requests]
        score_jobs: list[tuple[int, list[int], int, list[int]]] = []
        flat_meta: list[tuple[int, str]] = []

        for request_idx, request in enumerate(requests):
            for continuation in request.continuations or ():
                context_enc, continuation_enc = encode_context_and_continuation(
                    self.tokenizer, request.prompt, continuation
                )
                score_jobs.append(
                    (len(flat_meta), context_enc + continuation_enc, len(context_enc), continuation_enc)
                )
                flat_meta.append((request_idx, continuation))

        if not score_jobs:
            return results

        scored_batch_limit = self._get_score_batch_limit([sequence for _, sequence, _, _ in score_jobs])
        scored_outputs = self._call_with_adaptive_splitting(
            score_jobs,
            self._score_token_sequences,
            max_batch_size=scored_batch_limit,
        )

        for flat_idx, logprob_entries, total in scored_outputs:
            request_idx, continuation = flat_meta[flat_idx]
            results[request_idx].append(
                LMOutput(
                    text=continuation,
                    logprobs=logprob_entries,
                    metadata={"total_logprob": total},
                )
            )

        return results

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async generate completions.

        Runs the synchronous HuggingFace generate in a thread pool to avoid blocking.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request.
        """
        return await asyncio.to_thread(self.generate, requests, sampling_params)

    async def alogprobs(
        self,
        requests: list[LMRequest],
    ) -> list[list[LMOutput]]:
        """Async compute logprobs for continuations.

        Runs the synchronous HuggingFace logprobs in a thread pool to avoid blocking.

        Args:
            requests: Batch of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        return await asyncio.to_thread(self.logprobs, requests)


def _coerce_optional_int(value: Any) -> int | None:
    """Normalize optional integer provider kwargs."""
    if value is None:
        return None
    return int(value)
