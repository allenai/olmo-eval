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
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: list(
            range(len(text.split()))
        )
        tokenizer.decode.side_effect = lambda ids: " ".join(f"tok{i}" for i in ids)
        tokenizer.bos_token_id = 0
        tokenizer.eos_token_id = 1
        return tokenizer

    @pytest.fixture
    def provider(self, mock_tokenizer):
        """Create a VLLMServerProvider with __init__ bypassed."""
        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        with patch.object(VLLMServerProvider, "__init__", lambda self, *a, **kw: None):
            p = VLLMServerProvider.__new__(VLLMServerProvider)
            p.model_name = "test-model"
            p.base_url = "http://localhost:8000/v1"
            p._tokenizer = mock_tokenizer
            p._client = None
            p._http_client = None
            p._raw_http_client = None
            p._server = None
            p._max_length = 4096
            p._get_tokenizer = MagicMock(return_value=mock_tokenizer)
            return p

    def _make_vllm_response(self, prompt_logprobs):
        """Build a JSON response matching vLLM's prompt_logprobs format."""
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"prompt_logprobs": prompt_logprobs}],
        }
        return resp

    @pytest.mark.anyio
    async def test_logprobs_extracts_continuation_tokens(self, provider, mock_tokenizer):
        """Test that logprobs are correctly extracted for continuation tokens only."""
        # Context: 3 tokens [0, 1, 2], Continuation: 1 token [3]
        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.return_value = ([0, 1, 2], [3])

            # prompt_logprobs has one entry per token position.
            # First 3 are context (skipped), 4th is the continuation token.
            prompt_logprobs = [
                None,
                {"0": {"logprob": -0.5, "decoded_token": "answer"}},
                {"1": {"logprob": -0.3, "decoded_token": "is"}},
                {"3": {"logprob": -0.1, "decoded_token": "Paris"}},
            ]

            mock_http = AsyncMock()
            mock_http.post.return_value = self._make_vllm_response(prompt_logprobs)
            provider._get_raw_http_client = MagicMock(return_value=mock_http)

            request = LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
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
            assert output.metadata["sum_logits"] == pytest.approx(-0.1)
            assert output.metadata["num_tokens"] == 1

    @pytest.mark.anyio
    async def test_logprobs_multiple_continuations(self, provider, mock_tokenizer):
        """Test logprobs computation for multiple continuations."""
        call_count = [0]

        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            # Context is always 2 tokens, continuation is 1 token
            mock_encode.side_effect = lambda tok, ctx, cont: ([0, 1], [2])

            async def mock_post(url, json=None):
                call_count[0] += 1
                # Different logprobs per continuation
                lp_val = {1: -0.1, 2: -0.5, 3: -0.8}[call_count[0]]
                prompt_logprobs = [
                    None,
                    {"1": {"logprob": -0.2, "decoded_token": "is"}},
                    {"2": {"logprob": lp_val, "decoded_token": "cont"}},
                ]
                return self._make_vllm_response(prompt_logprobs)

            mock_http = AsyncMock()
            mock_http.post.side_effect = mock_post
            provider._get_raw_http_client = MagicMock(return_value=mock_http)

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
    async def test_logprobs_empty_continuations(self, provider):
        """Test handling of empty continuations list."""
        mock_http = AsyncMock()
        provider._get_raw_http_client = MagicMock(return_value=mock_http)

        request = LMRequest(
            request_type=RequestType.COMPLETION,
            prompt="Test prompt",
            continuations=[],
        )

        outputs = await provider._logprobs_single_impl(request)

        assert len(outputs) == 0
        mock_http.post.assert_not_called()

    @pytest.mark.anyio
    async def test_logprobs_uses_completions_endpoint(self, provider, mock_tokenizer):
        """Test that logprobs uses the raw completions endpoint."""
        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.return_value = ([0], [1])

            prompt_logprobs = [
                None,
                {"1": {"logprob": -0.1, "decoded_token": "yes"}},
            ]
            mock_http = AsyncMock()
            mock_http.post.return_value = self._make_vllm_response(prompt_logprobs)
            provider._get_raw_http_client = MagicMock(return_value=mock_http)

            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Test",
                continuations=[" yes"],
            )

            await provider._logprobs_single_impl(request)

            mock_http.post.assert_called_once()
            call_args = mock_http.post.call_args
            assert "/completions" in call_args[0][0]

    @pytest.mark.anyio
    async def test_logprobs_passes_prompt_logprobs_param(self, provider, mock_tokenizer):
        """Test that prompt_logprobs parameter is passed correctly."""
        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.return_value = ([0], [1])

            prompt_logprobs = [
                None,
                {"1": {"logprob": -0.1, "decoded_token": "yes"}},
            ]
            mock_http = AsyncMock()
            mock_http.post.return_value = self._make_vllm_response(prompt_logprobs)
            provider._get_raw_http_client = MagicMock(return_value=mock_http)

            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Test",
                continuations=[" yes"],
            )

            await provider._logprobs_single_impl(request)

            call_kwargs = mock_http.post.call_args[1]
            json_body = call_kwargs["json"]
            assert json_body["prompt_logprobs"] == 5
            assert json_body["max_tokens"] == 1
            assert json_body["add_special_tokens"] is False
