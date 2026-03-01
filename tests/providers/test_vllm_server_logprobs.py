"""Unit tests for VLLMServerProvider logprobs implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from olmo_eval.common.types import LMRequest, RequestType


class TestVLLMServerProviderLogprobs:
    """Tests for VLLMServerProvider._logprobs_single_impl."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        # Simulate encoding: each word becomes one token
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: list(
            range(len(text.split()))
        )
        tokenizer.decode.side_effect = lambda ids: " ".join(f"tok{i}" for i in ids)
        tokenizer.bos_token_id = 0
        tokenizer.eos_token_id = 1
        return tokenizer

    @pytest.fixture
    def mock_completion_response(self):
        """Create a mock completion response with logprobs on choice."""

        def make_response(tokens, token_logprobs):
            logprobs_data = MagicMock()
            logprobs_data.tokens = tokens
            logprobs_data.token_logprobs = token_logprobs

            choice = MagicMock()
            choice.logprobs = logprobs_data

            response = MagicMock()
            response.choices = [choice]
            return response

        return make_response

    @pytest.mark.anyio
    async def test_logprobs_extracts_continuation_tokens(
        self, mock_tokenizer, mock_completion_response
    ):
        """Test that logprobs are correctly extracted for continuation tokens only."""
        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        # Mock the provider
        with patch.object(VLLMServerProvider, "__init__", lambda self, *a, **kw: None):
            provider = VLLMServerProvider.__new__(VLLMServerProvider)
            provider.model_name = "test-model"
            provider._tokenizer = mock_tokenizer
            provider._client = None
            provider._http_client = None
            provider._server = None

            # Mock encode_context_and_continuation to return known token sequences
            # Context: 3 tokens [0, 1, 2], Continuation: 1 token [3]
            with patch(
                "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
            ) as mock_encode:
                mock_encode.return_value = ([0, 1, 2], [3])
                mock_tokenizer.decode.return_value = "The answer is Paris"

                # Mock the completion response with logprobs
                # First 3 tokens are context (should be skipped)
                # 4th token is continuation token
                tokens = ["The", "answer", "is", "Paris"]
                token_logprobs = [None, -0.5, -0.3, -0.1]  # Paris has -0.1

                mock_client = AsyncMock()
                mock_client.completions.create.return_value = mock_completion_response(
                    tokens, token_logprobs
                )
                provider._client = mock_client
                provider._get_or_create_client = MagicMock(return_value=mock_client)
                provider._get_tokenizer = MagicMock(return_value=mock_tokenizer)

                request = LMRequest(
                    request_type=RequestType.COMPLETION,
                    prompt="The answer is",
                    continuations=[" Paris"],
                )

                outputs = await provider._logprobs_single_impl(request)

                assert len(outputs) == 1
                output = outputs[0]
                assert output.text == " Paris"
                # Should only have logprobs for continuation tokens (1 token)
                assert output.logprobs is not None
                assert len(output.logprobs) == 1
                assert output.logprobs[0]["token"] == "Paris"
                assert output.logprobs[0]["logprob"] == -0.1
                assert output.metadata["sum_logits"] == -0.1
                assert output.metadata["num_tokens"] == 1

    @pytest.mark.anyio
    async def test_logprobs_multiple_continuations(self, mock_tokenizer, mock_completion_response):
        """Test logprobs computation for multiple continuations."""
        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        with patch.object(VLLMServerProvider, "__init__", lambda self, *a, **kw: None):
            provider = VLLMServerProvider.__new__(VLLMServerProvider)
            provider.model_name = "test-model"
            provider._tokenizer = mock_tokenizer
            provider._server = None

            # Each continuation gets separate API call
            call_count = [0]

            def mock_encode(text, add_special_tokens=False):
                # Context is always 2 tokens, continuation is 1 token
                if "Paris" in text or "London" in text or "Berlin" in text:
                    return [0, 1, 2]  # context + continuation
                return [0, 1]  # context only

            mock_tokenizer.encode.side_effect = mock_encode
            mock_tokenizer.decode.return_value = "decoded text"

            async def mock_create(**kwargs):
                call_count[0] += 1
                # Different logprobs for different continuations
                if call_count[0] == 1:  # Paris
                    tokens = ["Capital", "is", "Paris"]
                    token_logprobs = [None, -0.2, -0.1]
                elif call_count[0] == 2:  # London
                    tokens = ["Capital", "is", "London"]
                    token_logprobs = [None, -0.2, -0.5]
                else:  # Berlin
                    tokens = ["Capital", "is", "Berlin"]
                    token_logprobs = [None, -0.2, -0.8]
                return mock_completion_response(tokens, token_logprobs)

            mock_client = AsyncMock()
            mock_client.completions.create.side_effect = mock_create
            provider._client = mock_client
            provider._get_or_create_client = MagicMock(return_value=mock_client)
            provider._get_tokenizer = MagicMock(return_value=mock_tokenizer)

            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Capital is",
                continuations=[" Paris", " London", " Berlin"],
            )

            outputs = await provider._logprobs_single_impl(request)

            assert len(outputs) == 3
            # Paris should have highest logprob (least negative)
            logprobs = [o.metadata["sum_logits"] for o in outputs]
            assert logprobs[0] > logprobs[1] > logprobs[2]

    @pytest.mark.anyio
    async def test_logprobs_empty_continuations(self, mock_tokenizer):
        """Test handling of empty continuations list."""
        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        with patch.object(VLLMServerProvider, "__init__", lambda self, *a, **kw: None):
            provider = VLLMServerProvider.__new__(VLLMServerProvider)
            provider.model_name = "test-model"
            provider._tokenizer = mock_tokenizer
            provider._server = None

            mock_client = AsyncMock()
            provider._client = mock_client
            provider._get_or_create_client = MagicMock(return_value=mock_client)
            provider._get_tokenizer = MagicMock(return_value=mock_tokenizer)

            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Test prompt",
                continuations=[],
            )

            outputs = await provider._logprobs_single_impl(request)

            assert len(outputs) == 0
            # API should not be called for empty continuations
            mock_client.completions.create.assert_not_called()

    @pytest.mark.anyio
    async def test_logprobs_uses_completions_endpoint(
        self, mock_tokenizer, mock_completion_response
    ):
        """Test that logprobs uses completions endpoint, not chat."""
        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        with patch.object(VLLMServerProvider, "__init__", lambda self, *a, **kw: None):
            provider = VLLMServerProvider.__new__(VLLMServerProvider)
            provider.model_name = "test-model"
            provider._tokenizer = mock_tokenizer
            provider._server = None

            mock_tokenizer.encode.return_value = [0, 1]
            mock_tokenizer.decode.return_value = "text"

            mock_client = AsyncMock()
            mock_response = mock_completion_response(["Test", "yes"], [None, -0.1])
            mock_client.completions.create.return_value = mock_response
            provider._client = mock_client
            provider._get_or_create_client = MagicMock(return_value=mock_client)
            provider._get_tokenizer = MagicMock(return_value=mock_tokenizer)

            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Test",
                continuations=[" yes"],
            )

            await provider._logprobs_single_impl(request)

            # Should use completions endpoint
            mock_client.completions.create.assert_called_once()
            # Should NOT use chat endpoint
            assert (
                not hasattr(mock_client.chat.completions, "create")
                or not mock_client.chat.completions.create.called
            )

    @pytest.mark.anyio
    async def test_logprobs_passes_logprobs_and_echo_params(
        self, mock_tokenizer, mock_completion_response
    ):
        """Test that logprobs and echo parameters are passed correctly."""
        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        with patch.object(VLLMServerProvider, "__init__", lambda self, *a, **kw: None):
            provider = VLLMServerProvider.__new__(VLLMServerProvider)
            provider.model_name = "test-model"
            provider._tokenizer = mock_tokenizer
            provider._server = None

            mock_tokenizer.encode.return_value = [0, 1]
            mock_tokenizer.decode.return_value = "text"

            mock_client = AsyncMock()
            mock_response = mock_completion_response(["Test", "yes"], [None, -0.1])
            mock_client.completions.create.return_value = mock_response
            provider._client = mock_client
            provider._get_or_create_client = MagicMock(return_value=mock_client)
            provider._get_tokenizer = MagicMock(return_value=mock_tokenizer)

            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Test",
                continuations=[" yes"],
            )

            await provider._logprobs_single_impl(request)

            # Check that logprobs and echo parameters are passed
            call_kwargs = mock_client.completions.create.call_args[1]
            assert call_kwargs["logprobs"] == 5
            assert call_kwargs["echo"] is True
            assert call_kwargs["max_tokens"] == 1
