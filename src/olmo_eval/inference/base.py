"""Inference provider base class and protocol definition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class InferenceProvider(ABC):
    """Abstract base class for language model inference providers.

    All providers must implement `generate` and `logprobs` methods.
    Providers that support async operations should override `agenerate`
    and `alogprobs` for better performance in async contexts.
    """

    model_name: str

    def __init__(self, model_name: str) -> None:
        """Initialize the provider.

        Args:
            model_name: Model identifier or path.
        """
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Generate completions for a batch of requests.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request. Each inner list contains
            `sampling_params.num_samples` outputs.
        """
        ...

    @abstractmethod
    def logprobs(
        self,
        requests: list[LMRequest],
    ) -> list[list[LMOutput]]:
        """Compute log probabilities for continuations.

        Args:
            requests: Batch of requests with continuations to score.

        Returns:
            List of output lists. Each inner list has one LMOutput per
            continuation in the request, with logprobs populated.
        """
        ...

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async version of generate.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request.

        Raises:
            NotImplementedError: If provider doesn't support async generation.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support async generation")

    async def alogprobs(
        self,
        requests: list[LMRequest],
    ) -> list[list[LMOutput]]:
        """Async version of logprobs.

        Args:
            requests: Batch of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.

        Raises:
            NotImplementedError: If provider doesn't support async logprobs.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support async logprobs")

    def _default_sampling_params(self, sampling_params: SamplingParams | None) -> SamplingParams:
        """Return sampling params with defaults applied."""
        return sampling_params or SamplingParams()

    def get_tokenizer(self) -> Any:
        """Get the tokenizer for this provider.

        Returns:
            The tokenizer instance.

        Raises:
            NotImplementedError: If provider doesn't support tokenizer access.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide tokenizer access")

    def get_openai_client(self) -> AsyncOpenAI:
        """Get an AsyncOpenAI client for this provider.

        Used by backends that need an OpenAI-compatible client
        (e.g., OpenAI Agents SDK).

        Returns:
            AsyncOpenAI client.

        Raises:
            NotImplementedError: If provider doesn't have an OpenAI-compatible interface.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide an OpenAI client. "
            f"Use a provider with OpenAI API support (e.g., VLLMServerProvider, LiteLLMProvider)."
        )
