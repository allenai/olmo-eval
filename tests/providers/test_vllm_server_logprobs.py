"""Unit tests for VLLMServerProvider completion and logprobs behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams


class TestVLLMServerProviderLogprobs:
    """Tests for VLLMServerProvider._logprobs_single_impl."""

    def _make_provider(self, **kwargs):
        """Create a real provider instance configured for an existing server."""
        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        with patch("olmo_eval.inference.providers.vllm_server.BeakerStatusReporter"):
            provider = VLLMServerProvider(
                "test-model", base_url="http://localhost:8000/v1", **kwargs
            )
        provider._max_length = 4096
        return provider

    def _make_fake_transformers(self, tokenizer):
        """Build a stub transformers module for local tokenizer loads."""
        from_pretrained = MagicMock(return_value=tokenizer)
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)
        )
        return from_pretrained, fake_transformers

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
            p._add_bos_token = None
            p._client = None
            p._http_client = None
            p._raw_http_client = None
            p._server = None
            p._max_length = 4096
            p._prompt_logprobs = 5
            p._completion_use_prompt_token_ids = False
            p._completion_client_side_stop_trim = False
            p._completion_sentencepiece_cleanup = False
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

    def _make_completion_response(self, text="test completion"):
        """Build a completion response matching the OpenAI client shape."""
        choice = MagicMock()
        choice.text = text
        choice.logprobs = None

        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = None
        return resp

    @pytest.mark.anyio
    async def test_generate_completion_sets_add_special_tokens_false(self, provider):
        """Completion payloads should disable server-side special token insertion."""
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=self._make_completion_response())

        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        params = SamplingParams(max_tokens=32, temperature=0.6, top_p=0.6)

        await provider._generate_completion(client, request, params)

        call_kwargs = client.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"]["add_special_tokens"] is False

    def test_describe_request_includes_chat_template_kwargs(self):
        """Chat traces should preserve template kwargs in generation metadata."""
        provider = self._make_provider(chat_template_kwargs={"enable_thinking": False})
        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=[{"role": "user", "content": "Hello"}],
        )
        params = SamplingParams(max_tokens=32, temperature=0.6, top_p=0.6)

        trace = provider.describe_request(request, params)

        assert trace is not None
        assert trace["generation_kwargs"]["chat_template_kwargs"] == {"enable_thinking": False}

    @pytest.mark.anyio
    async def test_generate_completion_appends_eos_to_stop_sequences(self, provider):
        """Completion payloads should include EOS in the stop list when available."""
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=self._make_completion_response())

        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        params = SamplingParams(max_tokens=32, stop_sequences=("Question:", "</s>"))

        await provider._generate_completion(client, request, params)

        call_kwargs = client.completions.create.call_args.kwargs
        assert call_kwargs["stop"] == ["Question:", "</s>", "tok1"]

    @pytest.mark.anyio
    async def test_generate_completion_with_explicit_runtime_flags_uses_token_ids(self):
        """Explicit completion runtime flags should tokenize locally and post-process output."""
        provider = self._make_provider(
            add_bos_token=False,
            completion_use_prompt_token_ids=True,
            completion_client_side_stop_trim=True,
            completion_sentencepiece_cleanup=True,
        )
        local_tokenizer = MagicMock()
        local_tokenizer.encode.return_value = [11, 12, 13]
        local_tokenizer.eos_token_id = None
        local_tokenizer.eos_token = None
        provider._get_tokenizer = MagicMock(return_value=local_tokenizer)

        raw_response = MagicMock()
        raw_response.raise_for_status = MagicMock()
        raw_response.json.return_value = {
            "choices": [{"text": "\u2581helloSTOP trailing", "logprobs": None}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        mock_http = AsyncMock()
        mock_http.post.return_value = raw_response
        provider._get_raw_http_client = MagicMock(return_value=mock_http)

        client = MagicMock()
        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        params = SamplingParams(max_tokens=8, stop_sequences=("STOP",))

        outputs = await provider._generate_completion(client, request, params)

        assert len(outputs) == 1
        assert outputs[0].text == " hello"
        payload = mock_http.post.call_args.kwargs["json"]
        assert payload["prompt"] == [11, 12, 13]
        assert payload["add_special_tokens"] is False
        client.completions.create.assert_not_called()

    def test_completion_eos_uses_revision_for_local_tokenizer(self):
        """EOS stop detection should respect the configured tokenizer revision."""
        provider = self._make_provider(
            tokenizer="custom-tokenizer",
            revision="stage2-step47684",
            trust_remote_code=True,
        )
        local_tokenizer = MagicMock()
        local_tokenizer.eos_token_id = 1
        local_tokenizer.decode.return_value = "</s>"
        from_pretrained, fake_transformers = self._make_fake_transformers(local_tokenizer)

        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            assert provider._get_completion_eos_stop() == "</s>"

        from_pretrained.assert_called_once_with(
            "custom-tokenizer",
            revision="stage2-step47684",
            trust_remote_code=True,
        )

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
    async def test_logprobs_threads_sampling_temperature(self, provider):
        """Logprob requests should honor the threaded sampling temperature."""
        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.return_value = ([0, 1], [2])

            prompt_logprobs = [
                None,
                {"1": {"logprob": -0.2, "decoded_token": "ctx"}},
                {"2": {"logprob": -0.1, "decoded_token": "choice"}},
            ]

            mock_http = AsyncMock()
            mock_http.post.return_value = self._make_vllm_response(prompt_logprobs)
            provider._get_raw_http_client = MagicMock(return_value=mock_http)

            request = LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt="Prompt",
                continuations=(" choice",),
            )
            params = SamplingParams(temperature=1.0)

            await provider._logprobs_single_impl(request, params)

            payload = mock_http.post.call_args.kwargs["json"]
            assert payload["temperature"] == 1.0

            trace = provider.describe_request(request, params)
            assert trace is not None
            assert trace["generation_kwargs"]["temperature"] == 1.0
            assert trace["generation_kwargs"]["max_gen_toks"] == 1

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

    @pytest.mark.anyio
    async def test_logprobs_respects_explicit_prompt_logprobs_override(self, mock_tokenizer):
        """Explicit prompt_logprobs override should be passed through to vLLM."""
        provider = self._make_provider(prompt_logprobs=1)
        provider._tokenizer = mock_tokenizer
        provider._get_tokenizer = MagicMock(return_value=mock_tokenizer)

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

            json_body = mock_http.post.call_args.kwargs["json"]
            assert json_body["prompt_logprobs"] == 1

    @pytest.mark.anyio
    async def test_logprobs_loads_local_tokenizer_with_revision(self):
        """Prompt logprob tokenization should use the configured tokenizer revision."""
        provider = self._make_provider(
            tokenizer="custom-tokenizer",
            revision="stage2-step47684",
            trust_remote_code=True,
        )
        local_tokenizer = MagicMock()
        from_pretrained, fake_transformers = self._make_fake_transformers(local_tokenizer)

        prompt_logprobs = [
            None,
            {"1": {"logprob": -0.1, "decoded_token": "yes"}},
        ]
        mock_http = AsyncMock()
        mock_http.post.return_value = self._make_vllm_response(prompt_logprobs)
        provider._get_raw_http_client = MagicMock(return_value=mock_http)

        with (
            patch.dict("sys.modules", {"transformers": fake_transformers}),
            patch(
                "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
            ) as mock_encode,
        ):
            mock_encode.return_value = ([0], [1])
            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Test",
                continuations=[" yes"],
            )

            outputs = await provider._logprobs_single_impl(request)

        assert len(outputs) == 1
        from_pretrained.assert_called_once_with(
            "custom-tokenizer",
            revision="stage2-step47684",
            trust_remote_code=True,
        )
