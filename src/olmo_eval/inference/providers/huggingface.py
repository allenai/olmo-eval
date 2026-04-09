"""Hugging Face Transformers provider."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.tokenizer_utils import encode_context_and_continuation

if TYPE_CHECKING:
    import torch  # type: ignore[ty:unresolved-import]


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
        import torch  # type: ignore[ty:unresolved-import]

        if not requests:
            return []

        params = self._default_sampling_params(sampling_params)
        gen_kwargs = self._build_generate_kwargs(params)
        encoded = self._tokenize_prompts_for_generation([request.prompt for request in requests])
        prompt_width = encoded["input_ids"].shape[1]
        prompt_token_ids = [
            encoded["input_ids"][request_idx][encoded["attention_mask"][request_idx].bool()].tolist()
            for request_idx in range(len(requests))
        ]
        results: list[list[LMOutput]] = [[] for _ in requests]

        for _ in range(params.num_samples):
            with torch.no_grad():
                generation = self.model.generate(**encoded, **gen_kwargs, return_dict_in_generate=True)

            generated_token_ids = generation.sequences[:, prompt_width:]
            generated_outputs: list[tuple[torch.Tensor, str]] = []
            score_sequences: list[list[int]] = []
            score_meta: list[tuple[int, int]] = []

            for request_idx, prompt_ids in enumerate(prompt_token_ids):
                gen_len = self._get_generated_sequence_length(generated_token_ids[request_idx])
                gen_ids = generated_token_ids[request_idx, :gen_len]
                gen_ids, text = self._truncate_at_stop(gen_ids, params.stop_sequences)
                generated_outputs.append((gen_ids, text))

                if len(gen_ids) > 0:
                    score_sequences.append(prompt_ids + gen_ids.tolist())
                    score_meta.append((request_idx, len(prompt_ids)))

            scored_log_probs = None
            if score_sequences:
                score_input_ids, score_attention_mask = self._pad_token_sequences(
                    score_sequences, padding_side="right"
                )
                with torch.no_grad():
                    score_logits = self.model(
                        input_ids=score_input_ids,
                        attention_mask=score_attention_mask,
                    ).logits
                scored_log_probs = torch.log_softmax(score_logits, dim=-1)

            score_row = 0
            for request_idx, (gen_ids, text) in enumerate(generated_outputs):

                logprob_entries = None
                metadata: dict[str, Any] = {}
                if len(gen_ids) > 0:
                    assert scored_log_probs is not None
                    _, prompt_len = score_meta[score_row]
                    logprob_entries = []
                    for token_offset, tok in enumerate(gen_ids):
                        score = scored_log_probs[score_row, prompt_len + token_offset - 1, tok]
                        token_str = self.tokenizer.decode(tok, skip_special_tokens=False)
                        logprob_entries.append(
                            {
                                "token": token_str,
                                "logprob": score.item(),
                                "bytes": list(token_str.encode("utf-8")),
                            }
                        )

                    sum_logits = sum(entry["logprob"] for entry in logprob_entries)
                    num_tokens = len(logprob_entries)
                    metadata = {
                        "sum_logits": sum_logits,
                        "num_tokens": num_tokens,
                        "num_tokens_all": num_tokens,
                    }
                    score_row += 1

                results[request_idx].append(
                    LMOutput(text=text, logprobs=logprob_entries, metadata=metadata)
                )

        return results

    def logprobs(
        self,
        requests: list[LMRequest],
    ) -> list[list[LMOutput]]:
        import torch  # type: ignore[ty:unresolved-import]

        results: list[list[LMOutput]] = [[] for _ in requests]
        flat_sequences: list[list[int]] = []
        flat_meta: list[tuple[int, str, int, list[int]]] = []

        for request_idx, request in enumerate(requests):
            for continuation in request.continuations or ():
                context_enc, continuation_enc = encode_context_and_continuation(
                    self.tokenizer, request.prompt, continuation
                )
                flat_sequences.append(context_enc + continuation_enc)
                flat_meta.append((request_idx, continuation, len(context_enc), continuation_enc))

        if not flat_sequences:
            return results

        input_ids, attention_mask = self._pad_token_sequences(flat_sequences, padding_side="right")
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        log_probs = torch.log_softmax(logits, dim=-1)

        for row_idx, (request_idx, continuation, ctx_len, continuation_enc) in enumerate(flat_meta):
            logprob_entries = []
            total = 0.0
            for token_offset, tok in enumerate(continuation_enc):
                lp = log_probs[row_idx, ctx_len + token_offset - 1, tok].item()
                token_str = self.tokenizer.decode(tok, skip_special_tokens=False)
                logprob_entries.append(
                    {
                        "token": token_str,
                        "logprob": lp,
                        "bytes": list(token_str.encode("utf-8")),
                    }
                )
                total += lp

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
