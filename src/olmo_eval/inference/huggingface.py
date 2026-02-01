"""Hugging Face Transformers provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from olmo_eval.core.types import LMOutput, LMRequest, SamplingParams

from .base import InferenceProvider
from .tokenizer_utils import encode_context_and_continuation

if TYPE_CHECKING:
    import torch


def _get_device() -> torch.device:
    """Detect the best available device."""
    import torch

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

    def get_tokenizer(self) -> Any:
        """Get the tokenizer for this provider."""
        return self.tokenizer

    def _build_generate_kwargs(self, params: SamplingParams) -> dict:
        """Convert SamplingParams to HuggingFace generate kwargs."""
        kwargs = {
            "max_new_tokens": params.max_tokens,
            "do_sample": params.temperature > 0,
        }

        if params.temperature > 0:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.top_k is not None:
            kwargs["top_k"] = params.top_k

        return kwargs

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
        import torch

        params = self._default_sampling_params(sampling_params)
        gen_kwargs = self._build_generate_kwargs(params)

        results = []
        for request in requests:
            prompt = request.prompt
            encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_len = encoded["input_ids"].shape[1]

            request_outputs = []
            for _ in range(params.num_samples):
                with torch.no_grad():
                    output_ids = self.model.generate(**encoded, **gen_kwargs)[0]

                gen_ids = output_ids[prompt_len:]
                gen_ids, text = self._truncate_at_stop(gen_ids, params.stop_sequences)

                # Compute logprobs if requested
                logprob_entries = None
                if params.logprobs is not None and len(gen_ids) > 0:
                    seq = torch.cat([encoded["input_ids"][0], gen_ids]).unsqueeze(0)
                    with torch.no_grad():
                        logits = self.model(seq).logits
                    log_probs = torch.log_softmax(logits, dim=-1)[0]

                    logprob_entries = []
                    for i, tok in enumerate(gen_ids):
                        lp = log_probs[prompt_len + i - 1, tok].item()
                        token_str = self.tokenizer.decode(tok, skip_special_tokens=False)
                        logprob_entries.append(
                            {
                                "token": token_str,
                                "logprob": lp,
                                "bytes": list(token_str.encode("utf-8")),
                            }
                        )

                request_outputs.append(LMOutput(text=text, logprobs=logprob_entries))

            results.append(request_outputs)

        return results

    def logprobs(
        self,
        requests: list[LMRequest],
    ) -> list[list[LMOutput]]:
        import torch

        results = []
        for request in requests:
            request_outputs = []
            for continuation in request.continuations or ():
                # Use shared utility for BOS handling and trailing space logic
                context_enc, continuation_enc = encode_context_and_continuation(
                    self.tokenizer, request.prompt, continuation
                )

                # Build full sequence as tensor
                full_ids = context_enc + continuation_enc
                full_enc = torch.tensor([full_ids], device=self.device)
                ctx_len = len(context_enc)

                with torch.no_grad():
                    logits = self.model(full_enc).logits

                log_probs = torch.log_softmax(logits, dim=-1)[0]

                logprob_entries = []
                total = 0.0
                for i, tok in enumerate(continuation_enc):
                    lp = log_probs[ctx_len + i - 1, tok].item()
                    token_str = self.tokenizer.decode(tok, skip_special_tokens=False)
                    logprob_entries.append(
                        {
                            "token": token_str,
                            "logprob": lp,
                            "bytes": list(token_str.encode("utf-8")),
                        }
                    )
                    total += lp

                request_outputs.append(
                    LMOutput(
                        text=continuation,
                        logprobs=logprob_entries,
                        metadata={"total_logprob": total},
                    )
                )

            results.append(request_outputs)

        return results
