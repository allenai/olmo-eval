"""Unit tests for VLLMServerProvider logprobs implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from olmo_eval.common.types import LMRequest, RequestType


def _make_prompt_logprobs_response(
    prompt_logprobs: list[dict | None],
) -> dict:
    """Build a raw vLLM JSON response with prompt_logprobs."""
    return {
        "choices": [
            {
                "prompt_logprobs": prompt_logprobs,
            }
        ],
    }


class TestVLLMServerProviderLogprobs:
    """Tests for VLLMServerProvider._logprobs_single_impl."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: list(
            range(len(text.split()))
        )
        tokenizer.decode.side_effect = lambda ids: " ".join(f"tok{i}" for i in ids)
        tokenizer.bos_token_id = 0
        tokenizer.eos_token_id = 1
        return tokenizer

    def _make_provider(self, mock_tokenizer, mock_http_client):
        """Create a VLLMServerProvider with mocked internals."""
        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        with patch.object(VLLMServerProvider, "__init__", lambda self, *a, **kw: None):
            provider = VLLMServerProvider.__new__(VLLMServerProvider)
            provider.model_name = "test-model"
            provider.base_url = "http://localhost:8000/v1"
            provider._tokenizer = mock_tokenizer
            provider._server = None
            provider._raw_http_client = mock_http_client
            provider._get_tokenizer = MagicMock(return_value=mock_tokenizer)
            return provider

    @pytest.mark.anyio
    async def test_logprobs_extracts_continuation_tokens(self, mock_tokenizer):
        """Test that logprobs are correctly extracted for continuation tokens only."""
        # Context: 3 tokens [0, 1, 2], Continuation: 1 token [3]
        # prompt_logprobs has one entry per token position; first position is None
        prompt_logprobs = [
            None,  # position 0 (context)
            {"0": {"logprob": -0.5, "decoded_token": "answer"}},  # position 1 (context)
            {"1": {"logprob": -0.3, "decoded_token": "is"}},  # position 2 (context)
            {"3": {"logprob": -0.1, "decoded_token": "Paris"}},  # position 3 (continuation)
        ]

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _make_prompt_logprobs_response(prompt_logprobs)

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_resp

        provider = self._make_provider(mock_tokenizer, mock_http)

        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.return_value = ([0, 1, 2], [3])

            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="The answer is",
                continuations=[" Paris"],
            )

            outputs = await provider._logprobs_single_impl(request)

        assert len(outputs) == 1
        output = outputs[0]
        assert output.text == " Paris"
        assert output.logprobs is not None
        assert len(output.logprobs) == 1
        assert output.logprobs[0]["token"] == "Paris"
        assert output.logprobs[0]["logprob"] == -0.1
        assert output.metadata["sum_logits"] == -0.1
        assert output.metadata["num_tokens"] == 1

    @pytest.mark.anyio
    async def test_logprobs_multiple_continuations(self, mock_tokenizer):
        """Test logprobs computation for multiple continuations."""
        call_count = [0]

        async def mock_post(url, json=None):
            call_count[0] += 1
            # Context: [0], Continuation: [2], so full_tokens = [0, 2]
            # prompt_logprobs has one entry per token; position 1 is for token ID 2
            if call_count[0] == 1:  # Paris
                pl = [None, {"2": {"logprob": -0.1, "decoded_token": "Paris"}}]
            elif call_count[0] == 2:  # London
                pl = [None, {"2": {"logprob": -0.5, "decoded_token": "London"}}]
            else:  # Berlin
                pl = [None, {"2": {"logprob": -0.8, "decoded_token": "Berlin"}}]
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = _make_prompt_logprobs_response(pl)
            return resp

        mock_http = AsyncMock()
        mock_http.post.side_effect = mock_post

        provider = self._make_provider(mock_tokenizer, mock_http)

        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.return_value = ([0], [2])

            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Capital is",
                continuations=[" Paris", " London", " Berlin"],
            )

            outputs = await provider._logprobs_single_impl(request)

        assert len(outputs) == 3
        logprobs = [o.metadata["sum_logits"] for o in outputs]
        assert logprobs[0] > logprobs[1] > logprobs[2]

    @pytest.mark.anyio
    async def test_logprobs_empty_continuations(self, mock_tokenizer):
        """Test handling of empty continuations list."""
        mock_http = AsyncMock()
        provider = self._make_provider(mock_tokenizer, mock_http)

        request = LMRequest(
            request_type=RequestType.COMPLETION,
            prompt="Test prompt",
            continuations=[],
        )

        outputs = await provider._logprobs_single_impl(request)

        assert len(outputs) == 0
        mock_http.post.assert_not_called()

    @pytest.mark.anyio
    async def test_logprobs_uses_completions_endpoint(self, mock_tokenizer):
        """Test that logprobs uses the completions endpoint via raw HTTP."""
        prompt_logprobs = [
            None,
            {"1": {"logprob": -0.1, "decoded_token": "yes"}},
        ]

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _make_prompt_logprobs_response(prompt_logprobs)

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_resp

        provider = self._make_provider(mock_tokenizer, mock_http)

        mock_tokenizer.encode.return_value = [0, 1]

        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.return_value = ([0], [1])

            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Test",
                continuations=[" yes"],
            )

            await provider._logprobs_single_impl(request)

        mock_http.post.assert_called_once()
        call_url = mock_http.post.call_args[0][0]
        assert "/completions" in call_url

    @pytest.mark.anyio
    async def test_logprobs_passes_prompt_logprobs_param(self, mock_tokenizer):
        """Test that prompt_logprobs parameter is passed in the raw HTTP request."""
        prompt_logprobs = [
            None,
            {"1": {"logprob": -0.1, "decoded_token": "yes"}},
        ]

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _make_prompt_logprobs_response(prompt_logprobs)

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_resp

        provider = self._make_provider(mock_tokenizer, mock_http)

        mock_tokenizer.encode.return_value = [0, 1]

        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.return_value = ([0], [1])

            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Test",
                continuations=[" yes"],
            )

            await provider._logprobs_single_impl(request)

        call_json = mock_http.post.call_args[1]["json"]
        assert call_json["prompt_logprobs"] == 5
        assert call_json["max_tokens"] == 1
