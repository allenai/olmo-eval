"""Hugging Face Transformers provider."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from olmo_eval.common.logging import get_logger
from olmo_eval.common.types import (
    LMOutput,
    LMRequest,
    LogProbEntry,
    RequestType,
    SamplingParams,
)
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.tokenizer_utils import encode_context_and_continuation

logger = get_logger(__name__)

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

    # kwargs that may be passed by the runner but are not valid for HF from_pretrained
    _IGNORED_KWARGS = frozenset(
        {
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "attention_backend",
            "use_tqdm_on_load",
            "add_bos_token",
            "max_model_len",
            "load_format",
            "model_loader_extra_config",
            "enable_auto_tool_choice",
        }
    )

    _TOKENIZER_KWARGS = frozenset(
        {
            "cache_dir",
            "force_download",
            "local_files_only",
            "revision",
            "token",
            "trust_remote_code",
        }
    )

    def __init__(self, model_name: str, tokenizer: str | None = None, **model_kwargs) -> None:
        """Initialize the provider.

        Args:
            model_name: HuggingFace model identifier or local path.
            tokenizer: Tokenizer path/identifier. If not specified, uses the model path.
            **model_kwargs: Additional arguments passed to from_pretrained.
        """
        try:
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except ImportError as e:
            raise ImportError(
                "transformers is required for HuggingFaceProvider. "
                "Install with: pip install transformers"
            ) from e

        # Strip kwargs meant for other providers (e.g., vLLM)
        for key in self._IGNORED_KWARGS:
            model_kwargs.pop(key, None)

        super().__init__(model_name)
        tokenizer_path = tokenizer or model_name
        tokenizer_kwargs = {
            key: value for key, value in model_kwargs.items() if key in self._TOKENIZER_KWARGS
        }
        self.tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)

        # Encoder-decoder models (T5, BART, ...) need a different auto-class and
        # have different generate() semantics, so detect them from the config
        # rather than requiring callers to declare it.
        config = AutoConfig.from_pretrained(model_name, **tokenizer_kwargs)
        self.is_encoder_decoder: bool = bool(getattr(config, "is_encoder_decoder", False))
        auto_class = AutoModelForSeq2SeqLM if self.is_encoder_decoder else AutoModelForCausalLM
        logger.info(
            "Loading %s with %s (is_encoder_decoder=%s)",
            model_name,
            getattr(auto_class, "__name__", auto_class),
            self.is_encoder_decoder,
        )
        self.model: Any = auto_class.from_pretrained(model_name, **model_kwargs)
        self.device = _get_device()
        self.model.to(self.device)
        self.model.eval()

    def get_tokenizer(self) -> Any:
        """Get the tokenizer for this provider."""
        return self.tokenizer

    def _context_length(self) -> int:
        """Model context length, used to cap uncapped (max_tokens=None) generation."""
        for attr in ("max_position_embeddings", "n_positions", "max_sequence_length"):
            value = getattr(self.model.config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return 2048

    def _build_generate_kwargs(self, params: SamplingParams, prompt_len: int = 0) -> dict[str, Any]:
        """Convert SamplingParams to HuggingFace generate kwargs."""
        # Use explicit do_sample flag, overriding temperature-based inference
        do_sample = params.do_sample and params.temperature > 0

        # max_tokens=None means "uncapped"; transformers would otherwise fall back
        # to its tiny default max_length. max_new_tokens counts generated tokens on
        # top of the prompt, so reserve the room left in the context after it.
        if params.max_tokens is not None:
            max_new_tokens = params.max_tokens
        else:
            max_new_tokens = max(self._context_length() - prompt_len, 1)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }

        if do_sample:
            if params.temperature > 0:
                kwargs["temperature"] = params.temperature
            if params.top_p is not None:
                kwargs["top_p"] = params.top_p
            if params.top_k is not None:
                kwargs["top_k"] = params.top_k

        return kwargs

    def _scored_log_probs(self, sequence: torch.Tensor, num_scored: int) -> torch.Tensor:
        """Log-probs from the positions that predict the trailing scored tokens.

        Only the logits for those positions are materialized when the model supports
        limiting head computation, so scoring a long prompt does not allocate a
        vocabulary-sized distribution for every token in the sequence.

        Args:
            sequence: Single-row batch of token ids ending with the tokens to score.
            num_scored: Number of trailing tokens in `sequence` to score.

        Returns:
            Tensor of shape (num_scored, vocab_size); row j predicts trailing token j.
        """
        import torch

        # One extra position: token j is predicted by the position before it.
        keep = num_scored + 1
        if sequence.shape[1] < keep:
            raise ValueError(
                f"Cannot score {num_scored} trailing token(s) in a sequence of length "
                f"{sequence.shape[1]}: the first scored token has no preceding position."
            )

        with torch.no_grad():
            try:
                logits = self.model(sequence, logits_to_keep=keep).logits
            except TypeError:
                logits = self.model(sequence).logits

        # Slice from the end so this is correct both when the model honored the
        # request and when it returned logits for the whole sequence.
        return torch.log_softmax(logits[0, -keep:-1, :], dim=-1)

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
        if params.truncate_prompt_tokens is not None or params.truncation_side is not None:
            logger.warning(
                "truncate_prompt_tokens or truncation_side has been set in the params, "
                "but is not supported for the HuggingFaceProvider and will not be used."
            )
        gen_kwargs = self._build_generate_kwargs(params)

        results = []
        for request in requests:
            prompt = request.prompt
            encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_len = encoded["input_ids"].shape[1]
            gen_kwargs = self._build_generate_kwargs(params, prompt_len)

            request_outputs = []
            for _ in range(params.num_samples):
                with torch.no_grad():
                    output_ids = self.model.generate(**encoded, **gen_kwargs)[0]

                # A causal LM returns the prompt followed by the continuation, so
                # the prompt has to be sliced off. An encoder-decoder returns only
                # decoder tokens, and slicing would eat the answer.
                gen_ids = output_ids if self.is_encoder_decoder else output_ids[prompt_len:]
                gen_ids, text = self._truncate_at_stop(gen_ids, params.stop_sequences)

                # Always compute logprobs for metrics
                logprob_entries = None
                metadata: dict[str, Any] = {}
                # the scoring below reads logits at prompt-relative positions of a
                # single concatenated sequence, which only holds for a causal LM;
                # encoder-decoder logprobs would need a decoder-side forward pass
                if len(gen_ids) > 0 and not self.is_encoder_decoder:
                    seq = torch.cat([encoded["input_ids"][0], gen_ids]).unsqueeze(0)
                    log_probs = self._scored_log_probs(seq, len(gen_ids))

                    logprob_entries: list[LogProbEntry] = []
                    for i, tok in enumerate(gen_ids):
                        lp = log_probs[i, tok].item()
                        token_str = self.tokenizer.decode(tok, skip_special_tokens=False)
                        logprob_entries.append(
                            {
                                "token": token_str,
                                "logprob": lp,
                                "bytes": list(token_str.encode("utf-8")),
                            }
                        )

                    # Compute metadata from logprobs
                    sum_logits = sum(entry["logprob"] for entry in logprob_entries)
                    num_tokens = len(logprob_entries)
                    metadata = {
                        "sum_logits": sum_logits,
                        "num_tokens": num_tokens,
                        "num_tokens_all": num_tokens,
                    }

                request_outputs.append(
                    LMOutput(text=text, logprobs=logprob_entries, metadata=metadata)
                )

            results.append(request_outputs)

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

        if request.request_type != RequestType.LOGLIKELIHOOD:
            # Pass the prompt length so an uncapped (max_tokens=None) trace shows the
            # same per-prompt max_new_tokens budget that generate() will use.
            prompt_len = len(self.tokenizer.encode(request.prompt)) if request.prompt else 0
            trace["provider"] = "HuggingFaceProvider"
            trace["endpoint"] = "transformers.generate"
            trace["generation_kwargs"] = {
                "max_gen_toks": params.max_tokens,
                **self._build_generate_kwargs(params, prompt_len),
            }
            trace["stop_sequences"] = list(params.stop_sequences or ())
        return trace

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        import torch

        params = self._default_sampling_params(sampling_params)
        if params.truncate_prompt_tokens is not None or params.truncation_side is not None:
            logger.warning(
                "truncate_prompt_tokens or truncation_side has been set in the params, "
                "but is not supported for the HuggingFaceProvider and will not be used."
            )
        results = []
        for request in requests:
            request_outputs = []
            cont_prompts = request.continuation_prompts
            for i, continuation in enumerate(request.continuations or ()):
                prompt = cont_prompts[i] if cont_prompts else request.prompt
                # Use shared utility for BOS handling and trailing space logic
                context_enc, continuation_enc = encode_context_and_continuation(
                    self.tokenizer, prompt, continuation
                )

                # Build full sequence as tensor
                full_ids = context_enc + continuation_enc
                full_enc = torch.tensor([full_ids], device=self.device)

                log_probs = self._scored_log_probs(full_enc, len(continuation_enc))

                logprob_entries: list[LogProbEntry] = []
                total = 0.0
                for j, tok in enumerate(continuation_enc):
                    lp = log_probs[j, tok].item()
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
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async compute logprobs for continuations.

        Runs the synchronous HuggingFace logprobs in a thread pool to avoid blocking.

        Args:
            requests: Batch of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        return await asyncio.to_thread(self.logprobs, requests, sampling_params)
