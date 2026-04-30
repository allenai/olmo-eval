"""vLLM Server provider for agent tasks."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import httpx

from olmo_eval.common.beaker_status import BeakerStatusReporter
from olmo_eval.common.logging import get_logger
from olmo_eval.common.types import LMOutput, LMRequest, LogProbEntry, RequestType, SamplingParams
from olmo_eval.common.types.tools import ToolCall
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.tokenizer_utils import encode_context_and_continuation
from olmo_eval.inference.utils import run_async

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from .vllm_server_utils import VLLMServerProcess

logger = get_logger(__name__)


class RemoteTokenizer:
    """Tokenizer that uses vLLM server's /tokenize and /detokenize endpoints.

    Provides a tokenizer-like interface without requiring transformers locally.
    This is useful when the vLLM server runs in an isolated environment.
    """

    def __init__(self, base_url: str, model_name: str) -> None:
        """Initialize the remote tokenizer.

        Args:
            base_url: Base URL of the vLLM server (e.g., "http://localhost:8000/v1").
            model_name: Model name for tokenization requests.
        """
        # Strip /v1 suffix if present for tokenize endpoints
        self._base_url = base_url.rstrip("/").removesuffix("/v1")
        self._model_name = model_name
        self._client: httpx.Client | None = None
        self._all_special_ids: set[int] | None = None
        # BOS/EOS token IDs - not available via API, set to None
        # For logprobs computation, empty context handling uses fallbacks
        self.bos_token_id: int | None = None
        self.eos_token_id: int | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=30.0)
        return self._client

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs using the remote server."""
        client = self._get_client()
        response = client.post(
            f"{self._base_url}/tokenize",
            json={
                "model": self._model_name,
                "prompt": text,
                "add_special_tokens": add_special_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("tokens", [])

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text using the remote server."""
        client = self._get_client()
        response = client.post(
            f"{self._base_url}/detokenize",
            json={
                "model": self._model_name,
                "tokens": token_ids,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("prompt", "")

    @property
    def all_special_ids(self) -> set[int]:
        """Get special token IDs (cached after first call)."""
        if self._all_special_ids is None:
            # vLLM doesn't expose special tokens via API, return empty set
            # Token inspection will still work, just without special token highlighting
            self._all_special_ids = set()
        return self._all_special_ids

    def __call__(
        self, text: str, return_tensors: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Tokenize text (for compatibility with HuggingFace tokenizer interface)."""
        tokens = self.encode(text)
        result: dict[str, Any] = {"input_ids": tokens}
        if return_tensors == "pt":
            import torch

            result["input_ids"] = torch.tensor([tokens])
        return result

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None


# Enable with VLLM_DEBUG_REQUESTS=1
_DEBUG_REQUESTS = os.environ.get("VLLM_DEBUG_REQUESTS", "").lower() in ("1", "true", "yes")

# Disable retries for debugging with VLLM_DEBUG_NO_RETRY=1
_DEBUG_NO_RETRY = os.environ.get("VLLM_DEBUG_NO_RETRY", "").lower() in ("1", "true", "yes")


async def _log_request(request: httpx.Request) -> None:
    """Log outgoing HTTP request."""
    body = request.content.decode("utf-8", errors="replace") if request.content else ""
    # Truncate very long bodies
    if len(body) > 2000:
        body = body[:2000] + "... [truncated]"
    logger.info(f"vLLM request: {request.method} {request.url}\n  body: {body}")


async def _log_response(response: httpx.Response) -> None:
    """Log HTTP response, especially errors."""
    if response.status_code >= 400:
        # Read body for error details
        await response.aread()
        body = response.text[:1000] if response.text else "(empty)"
        logger.error(f"vLLM response error: {response.status_code} {response.url}\n  body: {body}")


class _DebugTransport(httpx.AsyncHTTPTransport):
    """Transport wrapper that logs all HTTP errors with full tracebacks."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            return await super().handle_async_request(request)
        except Exception as e:
            import traceback

            logger.error(
                f"vLLM transport error: {type(e).__name__}: {e}\n"
                f"  URL: {request.url}\n"
                f"  Traceback:\n{traceback.format_exc()}"
            )
            raise


class VLLMServerProvider(InferenceProvider):
    """Provider that uses a vLLM server's OpenAI-compatible API.

    This provider can either connect to an existing vLLM server (if base_url
    is provided) or start and manage its own server subprocess.

    Example:
        # Auto-start server (managed lifecycle)
        provider = VLLMServerProvider("meta-llama/Llama-3.1-8B-Instruct")

        # Or connect to existing server
        provider = VLLMServerProvider("model", base_url="http://localhost:8000/v1")
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        timeout: float = 86400.0,
        max_concurrency: int = 32,
        max_retries: int = 3,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        tokenizer: str | None = None,
        enable_auto_tool_choice: bool = False,
        tool_call_parser: str | None = None,
        trust_remote_code: bool = False,
        log_dir: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        **server_kwargs: Any,
    ) -> None:
        """Initialize the provider.

        Args:
            model_name: Model identifier for requests.
            base_url: Base URL of existing vLLM server. If None, starts own server.
            timeout: Request timeout in seconds.
            max_concurrency: Maximum number of concurrent requests.
            max_retries: Maximum number of retries for transient errors.
            tensor_parallel_size: Number of GPUs for tensor parallelism (server mode).
            max_model_len: Maximum model context length (server mode).
            tokenizer: Tokenizer path override (server mode).
            enable_auto_tool_choice: Enable automatic tool choice (server mode).
            tool_call_parser: Tool call parser name (server mode).
            trust_remote_code: Trust remote code for model loading (server mode).
            log_dir: Directory to write server logs to (server mode).
            chat_template_kwargs: Extra kwargs for chat template (e.g., {"enable_thinking": false}).
            **server_kwargs: Additional vLLM server arguments.
        """
        super().__init__(model_name)
        self._beaker_reporter = BeakerStatusReporter()
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.chat_template_kwargs = chat_template_kwargs
        self._tokenizer_path = tokenizer or model_name
        self._client: AsyncOpenAI | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._raw_http_client: httpx.AsyncClient | None = None
        self._openai_module: Any = None
        self._tokenizer: Any = None
        self._server: VLLMServerProcess | None = None  # type: ignore[possibly-unresolved-reference]
        self._max_length: int | None = None

        if base_url:
            # Connect to existing server
            self.base_url = base_url
        else:
            # Start our own server
            from .vllm_server_utils import VLLMServerProcess

            # Build server kwargs
            srv_kwargs: dict[str, Any] = dict(server_kwargs)
            if max_model_len:
                srv_kwargs["max_model_len"] = max_model_len
            if tokenizer:
                srv_kwargs["tokenizer"] = tokenizer
            if enable_auto_tool_choice:
                srv_kwargs["enable_auto_tool_choice"] = True
                if tool_call_parser:
                    srv_kwargs["tool_call_parser"] = tool_call_parser
            if trust_remote_code:
                srv_kwargs["trust_remote_code"] = True
            if chat_template_kwargs:
                srv_kwargs["chat_template_kwargs"] = chat_template_kwargs

            self._server = VLLMServerProcess(
                model_name=model_name,
                tensor_parallel_size=tensor_parallel_size,
                log_dir=log_dir,
                **srv_kwargs,
            )

            def _on_startup(msg: str) -> None:
                logger.info(msg)
                self._beaker_reporter.update(msg)

            self._server.start(progress_callback=_on_startup)
            self.base_url = self._server.base_url

    def close(self) -> None:
        """Close the provider and stop managed server if any."""
        if self._server is not None:
            self._server.stop()
            self._server = None

    def __del__(self) -> None:
        """Ensure server is stopped on garbage collection."""
        self.close()

    @property
    def max_length(self) -> int:
        """Get the maximum model context length from the server."""
        if self._max_length is None:
            # Query /v1/models endpoint for max_model_len
            client = httpx.Client(timeout=30.0)
            try:
                resp = client.get(f"{self.base_url}/models")
                resp.raise_for_status()
                data = resp.json()
                for model_info in data.get("data", []):
                    max_len = model_info.get("max_model_len")
                    if max_len is not None:
                        self._max_length = int(max_len)
                        break
            finally:
                client.close()
            if self._max_length is None:
                # Fallback to a large default if server doesn't report it
                self._max_length = 4096
                logger.warning(
                    "Could not determine max_model_len from server, defaulting to %d",
                    self._max_length,
                )
        assert self._max_length is not None
        return self._max_length

    def _get_or_create_client(self) -> AsyncOpenAI:
        """Get or create the AsyncOpenAI client."""
        if self._client is None:
            import openai
            from openai import AsyncOpenAI

            self._openai_module = openai

            # Configure connection pool limits to prevent exhaustion with large batches.
            limits = httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=60.0,  # Close idle connections after 120s
            )

            # Build event hooks for debug logging if enabled
            event_hooks: dict[str, list[Any]] | None = None
            if _DEBUG_REQUESTS:
                logger.info("vLLM debug request logging enabled (VLLM_DEBUG_REQUESTS=1)")
                event_hooks = {
                    "request": [_log_request],
                    "response": [_log_response],
                }

            # Use debug transport when enabled to catch connection errors
            transport = _DebugTransport() if _DEBUG_REQUESTS else None

            self._http_client = httpx.AsyncClient(
                transport=transport,
                limits=limits,
                timeout=self.timeout,
                event_hooks=event_hooks or {},
            )

            # Use 0 retries when debugging to see errors immediately
            effective_retries = 0 if _DEBUG_NO_RETRY else self.max_retries
            if _DEBUG_NO_RETRY:
                logger.info("vLLM debug: retries disabled (VLLM_DEBUG_NO_RETRY=1)")

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key="EMPTY",
                timeout=self.timeout,
                max_retries=effective_retries,
                http_client=self._http_client,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the provider and release resources."""
        if self._raw_http_client is not None:
            await self._raw_http_client.aclose()
            self._raw_http_client = None
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _get_raw_http_client(self) -> httpx.AsyncClient:
        """Get or create raw HTTP client for direct vLLM API calls.

        Used for prompt_logprobs requests where we need integer-keyed
        token ID data that the OpenAI SDK converts to string keys.
        """
        if self._raw_http_client is None:
            self._raw_http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._raw_http_client

    def get_openai_client(self) -> AsyncOpenAI:
        """Get the AsyncOpenAI client for this provider."""
        client = self._get_or_create_client()
        assert client is not None, "AsyncOpenAI client creation failed"
        return client

    def _get_tokenizer(self, *, require_local: bool = False) -> Any:
        """Get or create the tokenizer.

        Args:
            require_local: If True, loads HuggingFace tokenizer locally (slower but
                has BOS/EOS token IDs for logprobs computation). If False, uses
                the remote vLLM tokenization API (faster, no transformers needed).

        Returns:
            Tokenizer instance (RemoteTokenizer or HuggingFace AutoTokenizer).
        """
        if require_local:
            # Load HuggingFace tokenizer for full functionality (BOS/EOS handling)
            if self._tokenizer is None or isinstance(self._tokenizer, RemoteTokenizer):
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
            return self._tokenizer

        # Use remote tokenizer by default (no transformers dependency)
        if self._tokenizer is None:
            self._tokenizer = RemoteTokenizer(self.base_url, self.model_name)
        return self._tokenizer

    def get_tokenizer(self) -> Any:
        """Get the tokenizer for this provider (for external use like inspection)."""
        return self._get_tokenizer(require_local=False)

    async def _generate_single_impl(
        self, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate completions for a single request."""
        client = self._get_or_create_client()

        # Route to completions endpoint for COMPLETION requests without messages
        use_completions = (
            request.request_type == RequestType.COMPLETION
            and not request.messages
            and request.prompt
        )

        if use_completions:
            return await self._generate_completion(client, request, params)
        else:
            return await self._generate_chat(client, request, params)

    async def _generate_completion(
        self, client: AsyncOpenAI, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate using the /v1/completions endpoint."""
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "prompt": request.prompt,
            "n": params.num_samples,
            "max_tokens": params.max_tokens,
            "logprobs": 1,  # Request logprobs for metrics
        }

        # Always send temperature explicitly to avoid server defaults (OpenAI API defaults to 1.0)
        kwargs["temperature"] = params.temperature
        if params.do_sample and params.temperature > 0:
            if params.top_p is not None:
                kwargs["top_p"] = params.top_p
            if params.top_k is not None:
                kwargs["extra_body"] = {"top_k": params.top_k}
        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)

        response = await client.completions.create(**kwargs)

        # Capture usage for accurate metrics
        usage = getattr(response, "usage", None)

        outputs = []
        for choice in response.choices:
            text = choice.text or ""

            # Convert logprobs to standard format
            logprob_entries: list[LogProbEntry] | None = None
            metadata: dict[str, Any] = {}

            # Store token counts from server for accurate metrics
            if usage:
                metadata["prompt_tokens"] = usage.prompt_tokens
                metadata["completion_tokens"] = usage.completion_tokens

            logprobs_data = getattr(choice, "logprobs", None)
            if logprobs_data and hasattr(logprobs_data, "token_logprobs"):
                tokens = logprobs_data.tokens or []
                token_logprobs = logprobs_data.token_logprobs or []
                logprob_entries = []
                for token, logprob in zip(tokens, token_logprobs, strict=False):
                    if logprob is not None:
                        logprob_entries.append({"token": token, "logprob": logprob})

                if logprob_entries:
                    sum_logits = sum(entry["logprob"] for entry in logprob_entries)
                    num_tokens = len(logprob_entries)
                    metadata = {
                        "sum_logits": sum_logits,
                        "num_tokens": num_tokens,
                        "num_tokens_all": num_tokens,
                    }

            outputs.append(LMOutput(text=text, logprobs=logprob_entries, metadata=metadata))

        return outputs

    async def _generate_chat(
        self, client: AsyncOpenAI, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate using the /v1/chat/completions endpoint."""
        # Build messages
        if request.messages:
            messages: list[dict[str, Any]] = [dict(m) for m in request.messages]
        else:
            messages = [{"role": "user", "content": request.prompt}]

        # Build tools if present
        tools = None
        if request.tools:
            tools = [t.to_openai() for t in request.tools]

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "n": params.num_samples,
            "max_tokens": params.max_tokens,
        }

        # Always send temperature explicitly to avoid server defaults (OpenAI API defaults to 1.0)
        kwargs["temperature"] = params.temperature
        extra_body: dict[str, Any] = {}
        if params.do_sample and params.temperature > 0:
            if params.top_p is not None:
                kwargs["top_p"] = params.top_p
            if params.top_k is not None:
                extra_body["top_k"] = params.top_k
        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)
        if tools:
            kwargs["tools"] = tools
        # Always request logprobs for metrics computation
        # Both logprobs=True and top_logprobs are required for chat completions API
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 1

        # Pass chat_template_kwargs via extra_body for vLLM
        if self.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = self.chat_template_kwargs
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = await client.chat.completions.create(**kwargs)

        # Capture usage for accurate metrics
        usage = getattr(response, "usage", None)

        outputs = []
        for choice in response.choices:
            text = choice.message.content or ""
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    ToolCall.create(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                    for tc in choice.message.tool_calls
                ]

            # Convert logprobs to standard format
            logprob_entries: list[LogProbEntry] | None = None
            metadata: dict[str, Any] = {}

            # Store token counts from server for accurate metrics
            if usage:
                metadata["prompt_tokens"] = usage.prompt_tokens
                metadata["completion_tokens"] = usage.completion_tokens

            logprobs_data = getattr(choice, "logprobs", None)
            if logprobs_data and hasattr(logprobs_data, "content") and logprobs_data.content:
                logprob_entries = []
                for lp in logprobs_data.content:
                    entry: LogProbEntry = {"token": lp.token, "logprob": lp.logprob}
                    lp_bytes = getattr(lp, "bytes", None)
                    if lp_bytes is not None:
                        entry["bytes"] = lp_bytes
                    logprob_entries.append(entry)

                # Compute metadata from logprobs
                sum_logits = sum(entry["logprob"] for entry in logprob_entries)
                num_tokens = len(logprob_entries)
                metadata = {
                    "sum_logits": sum_logits,
                    "num_tokens": num_tokens,
                    "num_tokens_all": num_tokens,
                }

            outputs.append(
                LMOutput(
                    text=text, logprobs=logprob_entries, metadata=metadata, tool_calls=tool_calls
                )
            )

        return outputs

    async def _generate_single_async(
        self, request: LMRequest, params: SamplingParams
    ) -> list[LMOutput]:
        """Generate completions for a single request."""
        return await self._generate_single_impl(request, params)

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async generate completions via the vLLM server.

        This is the native async implementation that should be used
        in async contexts to avoid nested event loops.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request.
        """
        from olmo_eval.inference.dispatch import dispatch_concurrent

        params = self._default_sampling_params(sampling_params)

        async def process(req: LMRequest) -> list[LMOutput]:
            return await self._generate_single_async(req, params)

        results = await dispatch_concurrent(
            requests,
            process,
            max_in_flight=self.max_concurrency,
            max_retries=self.max_retries,
            on_progress=self._beaker_reporter.progress_callback("vLLM gen", units="req/sec"),
        )
        return [r if r is not None else [] for r in results]

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Generate completions via the vLLM server (sync wrapper).

        For async contexts, prefer using agenerate() directly.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request.
        """
        return run_async(self.agenerate(requests, sampling_params))

    async def _logprobs_single_impl(self, request: LMRequest) -> list[LMOutput]:
        """Compute logprobs for continuations using raw prompt_logprobs.

        Uses vLLM's prompt_logprobs feature with integer token ID keys
        (via raw HTTP) to avoid string-keyed dict collisions in is_greedy
        computation. This matches the inline vLLM provider's behavior exactly.

        Prefers local HuggingFace tokenizer for accurate context/continuation
        boundary computation (matches oe-eval-internal behavior). Falls back to
        RemoteTokenizer if transformers is not available.
        """
        # Prefer local tokenizer for exact boundary matching with inline vLLM
        try:
            tokenizer = self._get_tokenizer(require_local=True)
        except (ImportError, Exception):
            tokenizer = self._get_tokenizer(require_local=False)

        # Get the context/prompt text
        context = request.prompt
        if request.messages:
            context = request.messages[0].get("content", "") if request.messages else ""

        http_client = self._get_raw_http_client()

        outputs = []
        cont_prompts = request.continuation_prompts
        for i, continuation in enumerate(request.continuations or ()):
            # Use per-continuation prompt when available (e.g. Trinh & Le partial eval)
            ctx = cont_prompts[i] if cont_prompts else context
            # Use shared utility for proper tokenization (handles BOS, trailing spaces)
            context_enc, continuation_enc = encode_context_and_continuation(
                tokenizer, ctx, continuation
            )

            # RemoteTokenizer doesn't have BOS/EOS token IDs, so for empty contexts
            # encode_context_and_continuation returns empty context_enc.
            # In this case, encode with add_special_tokens=True to get BOS from server.
            if not context_enc and context == "":
                context_enc = tokenizer.encode("", add_special_tokens=True)

            # Left-truncate to max_length - 1 to match inline vLLM provider behavior.
            # This ensures long prompts are handled the same way as oe-eval-internal:
            # the context is left-truncated while preserving the continuation tokens.
            # Use per-request max_length if set (e.g., from task config), else provider default.
            max_len = request.max_length or self.max_length
            full_tokens = context_enc + continuation_enc
            if len(full_tokens) > max_len - 1:
                full_tokens = full_tokens[-(max_len - 1) :]
                overflow = len(context_enc) + len(continuation_enc) - (max_len - 1)
                context_len = max(0, len(context_enc) - overflow)
            else:
                context_len = len(context_enc)

            # Use raw HTTP to get integer-keyed prompt_logprobs from vLLM.
            # The OpenAI SDK's top_logprobs uses string keys (decoded tokens),
            # which can collide when different token IDs decode to the same string.
            # prompt_logprobs preserves integer token IDs through JSON serialization
            # (as string representations of ints), avoiding this collision.
            resp = await http_client.post(
                f"{self.base_url}/completions",
                json={
                    "model": self.model_name,
                    "prompt": full_tokens,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "prompt_logprobs": 5,
                    "add_special_tokens": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Extract prompt_logprobs: list of dict[str(token_id), {logprob, ...}]
            choice = data["choices"][0]
            prompt_logprobs_raw = choice.get("prompt_logprobs") or []

            logprob_entries: list[LogProbEntry] = []
            total = 0.0
            is_greedy = True

            # Process continuation tokens only (skip context positions)
            cont_logprobs = prompt_logprobs_raw[context_len : context_len + len(continuation_enc)]
            cont_token_ids = full_tokens[context_len:]

            for token_id, token_probs in zip(cont_token_ids, cont_logprobs, strict=True):
                if not token_probs:
                    continue

                # JSON serializes integer dict keys as strings
                token_id_str = str(token_id)

                # Check is_greedy BEFORE the lp_obj gate so we catch non-greedy tokens
                # even when they aren't in the top-k returned by prompt_logprobs.
                # Keys are string-serialized integers (e.g., "128000"), not decoded tokens,
                # so no collision between different token IDs.
                if is_greedy:
                    max_tid = max(
                        token_probs.keys(),
                        key=lambda tid: (
                            token_probs[tid]["logprob"]
                            if isinstance(token_probs[tid], dict)
                            else token_probs[tid]
                        ),
                    )
                    if max_tid != token_id_str:
                        is_greedy = False

                lp_obj = token_probs.get(token_id_str)
                if lp_obj is None:
                    continue

                logprob_val = lp_obj["logprob"] if isinstance(lp_obj, dict) else lp_obj

                # Get decoded token string
                if isinstance(lp_obj, dict) and lp_obj.get("decoded_token"):
                    token_str = lp_obj["decoded_token"]
                else:
                    token_str = tokenizer.decode([token_id])

                logprob_entries.append(
                    {
                        "token": token_str,
                        "logprob": logprob_val,
                        "bytes": list(token_str.encode("utf-8")),
                    }
                )
                total += logprob_val

            num_tokens = len(logprob_entries)
            outputs.append(
                LMOutput(
                    text=continuation,
                    logprobs=logprob_entries if logprob_entries else None,
                    metadata={
                        "total_logprob": total,
                        "sum_logits": total,
                        "num_tokens": num_tokens,
                        "num_tokens_all": num_tokens,
                        "is_greedy": is_greedy,
                    },
                )
            )

        return outputs

    async def _logprobs_single_async(self, request: LMRequest) -> list[LMOutput]:
        """Compute logprobs for a single request."""
        return await self._logprobs_single_impl(request)

    async def alogprobs(self, requests: list[LMRequest]) -> list[list[LMOutput]]:
        """Async compute logprobs for continuations.

        Args:
            requests: Batch of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        from olmo_eval.inference.dispatch import dispatch_concurrent

        results = await dispatch_concurrent(
            requests,
            self._logprobs_single_async,
            max_in_flight=self.max_concurrency,
            max_retries=self.max_retries,
        )

        # Log if any requests failed (result is None)
        failed_count = sum(1 for r in results if r is None)
        if failed_count > 0:
            # Try to get the actual error by running one request directly
            try:
                await self._logprobs_single_async(requests[0])
            except Exception as e:
                logger.error(
                    f"alogprobs: {failed_count}/{len(requests)} requests failed. First error: {e!r}"
                )

        # Replace None with empty list for failed requests
        return [r if r is not None else [] for r in results]

    def logprobs(self, requests: list[LMRequest]) -> list[list[LMOutput]]:
        """Compute logprobs for continuations.

        Args:
            requests: Batch of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        return run_async(self.alogprobs(requests))
