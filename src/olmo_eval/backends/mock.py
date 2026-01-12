"""Mock backend for testing."""

from olmo_eval.core import LMOutput, LMRequest, SamplingParams

from .base import Backend


class MockBackend(Backend):
    """Mock backend that returns fixed responses for testing."""

    def __init__(self, model_name: str = "mock-model") -> None:
        """Initialize the mock backend.

        Args:
            model_name: Model name (defaults to "mock-model").
        """
        super().__init__(model_name)

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        params = self._default_sampling_params(sampling_params)
        num_samples = params.num_samples

        mock_output = LMOutput(
            text=" mock response. The answer is (A), or \\boxed{42}",
            logprobs=[
                {"token": " mock", "logprob": -0.034},
                {"token": " response", "logprob": -0.123},
                {"token": ".", "logprob": -0.057},
                {"token": " The", "logprob": -0.089},
            ],
        )

        return [[mock_output for _ in range(num_samples)] for _ in requests]

    def logprobs(
        self,
        requests: list[LMRequest],
    ) -> list[list[LMOutput]]:
        results = []
        for request in requests:
            continuations = request.continuations or ()
            request_outputs = []
            for continuation in continuations:
                request_outputs.append(
                    LMOutput(
                        text=continuation,
                        logprobs=[
                            {"token": " mock", "logprob": -0.034},
                            {"token": " token", "logprob": -0.123},
                        ],
                        metadata={"total_logprob": -0.157},
                    )
                )
            results.append(request_outputs)
        return results
