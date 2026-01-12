"""Backend base class and protocol definition."""

from abc import ABC, abstractmethod

from olmo_eval.core import LMOutput, LMRequest, SamplingParams


class Backend(ABC):
    """Abstract base class for language model backends.

    All backends must implement `generate` and `logprobs` methods.
    Common functionality like model name storage is handled here.
    """

    model_name: str

    def __init__(self, model_name: str) -> None:
        """Initialize the backend.

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

    def _default_sampling_params(self, sampling_params: SamplingParams | None) -> SamplingParams:
        """Return sampling params with defaults applied."""
        return sampling_params or SamplingParams()
