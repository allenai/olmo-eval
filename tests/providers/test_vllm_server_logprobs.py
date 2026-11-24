"""Unit tests for VLLMServerProvider completion and logprobs behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from olmo_eval.common.types import Instance, LMRequest, RequestType, Response, SamplingParams
from olmo_eval.runners.io.builders import build_predictions


class TestVLLMServerProviderLogprobs:
    """Tests for VLLMServerProvider._logprobs_single_impl."""

    def _make_provider(self, **kwargs):
        """Create a real provider instance configured for an existing server."""
        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        kwargs.setdefault("max_model_len", 4096)
        with patch("olmo_eval.inference.providers.vllm_server.BeakerStatusReporter"):
            provider = VLLMServerProvider(
                "test-model", base_url="http://localhost:8000/v1", **kwargs
            )
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
            p._request_model_name = "test-model"
            p.base_url = "http://localhost:8000/v1"
            p._tokenizer = mock_tokenizer
            p._add_bos_token = None
            p._client = None
            p._http_client = None
            p._raw_http_client = None
            p._server = None
            p._max_length = 4096
            p._max_length_source = "configured"
            p._max_length_discovery_attempted = True
            p._prompt_logprobs = 5
            # The server is not asked about the prompt unless a test says what it answers.
            p._request_opens_think = AsyncMock(return_value=False)
            p._get_think_close_ids = AsyncMock(return_value=None)
            p._completion_use_prompt_token_ids = False
            p._completion_client_side_stop_trim = False
            p._completion_sentencepiece_cleanup = False
            p._get_tokenizer = MagicMock(return_value=mock_tokenizer)
            return p

    def test_external_provider_uses_explicit_max_model_len_without_discovery(self):
        """An explicit external-server limit is authoritative and avoids /models."""
        with patch("olmo_eval.inference.providers.vllm_server.httpx.Client") as client_cls:
            provider = self._make_provider(max_model_len=131072)

            assert provider.max_length == 131072
            assert provider._max_length_source == "configured"

        client_cls.assert_not_called()

    def test_missing_models_endpoint_leaves_max_length_unknown(self):
        """A server without /models must not acquire a guessed context length."""
        provider = self._make_provider(max_model_len=None)
        request = httpx.Request("GET", f"{provider.base_url}/models")
        response = httpx.Response(404, request=request)
        client = MagicMock()
        client.get.return_value = response

        with patch(
            "olmo_eval.inference.providers.vllm_server.httpx.Client", return_value=client
        ) as client_cls:
            assert provider.max_length is None
            assert provider.max_length is None

        client_cls.assert_called_once_with(timeout=30.0)
        client.get.assert_called_once_with(f"{provider.base_url}/models")
        client.close.assert_called_once()
        assert provider._max_length is None
        assert provider._max_length_source is None

    def test_model_record_without_max_model_len_leaves_length_unknown(self):
        """Incomplete model metadata must not fall back to 4096."""
        provider = self._make_provider(max_model_len=None)
        response = MagicMock()
        response.json.return_value = {"data": [{"id": "test-model"}]}
        client = MagicMock()
        client.get.return_value = response

        with patch("olmo_eval.inference.providers.vllm_server.httpx.Client", return_value=client):
            assert provider.max_length is None

        assert provider._max_length is None
        assert provider._max_length_source is None

    def test_valid_model_record_provides_authoritative_length(self):
        """A valid matching model record is cached as a server-sourced limit."""
        provider = self._make_provider(max_model_len=None)
        response = MagicMock()
        response.json.return_value = {"data": [{"id": "test-model", "max_model_len": 65536}]}
        client = MagicMock()
        client.get.return_value = response

        with patch(
            "olmo_eval.inference.providers.vllm_server.httpx.Client", return_value=client
        ) as client_cls:
            assert provider.max_length == 65536
            assert provider.max_length == 65536

        client_cls.assert_called_once_with(timeout=30.0)
        assert provider._max_length_source == "server"

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

    def _make_chat_response(self, text="test reply"):
        """Build a chat completion response matching the OpenAI client shape."""
        message = MagicMock()
        message.content = text
        message.tool_calls = None

        choice = MagicMock()
        choice.message = message
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

    @pytest.mark.anyio
    async def test_generate_completion_omits_max_tokens_when_none(self, provider):
        """max_tokens=None means uncapped: omit the field rather than sending null."""
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=self._make_completion_response())

        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        await provider._generate_completion(client, request, SamplingParams(max_tokens=None))

        assert "max_tokens" not in client.completions.create.call_args.kwargs

    @pytest.mark.anyio
    async def test_generate_completion_includes_max_tokens_when_set(self, provider):
        """An explicit max_tokens is still forwarded to the completion payload."""
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=self._make_completion_response())

        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        await provider._generate_completion(client, request, SamplingParams(max_tokens=64))

        assert client.completions.create.call_args.kwargs["max_tokens"] == 64

    @pytest.mark.anyio
    async def test_generate_completion_does_not_synthesize_prompt_truncation(self, provider):
        """Long generation prompts remain intact unless truncation is explicitly requested."""
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=self._make_completion_response())

        prompt = "token " * 5000
        request = LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)
        await provider._generate_completion(client, request, SamplingParams(max_tokens=1024))

        call_kwargs = client.completions.create.call_args.kwargs
        assert call_kwargs["prompt"] == prompt
        assert "truncate_prompt_tokens" not in call_kwargs["extra_body"]
        assert "truncation_side" not in call_kwargs["extra_body"]

        trace = provider.describe_request(request, SamplingParams(max_tokens=1024))
        assert trace is not None
        assert "truncate_prompt_tokens" not in trace["generation_kwargs"]
        assert "truncation_side" not in trace["generation_kwargs"]

    @pytest.mark.anyio
    async def test_generate_completion_preserves_stricter_explicit_truncation(self, provider):
        """Explicit truncation settings remain effective when they fit the context budget."""
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=self._make_completion_response())

        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        params = SamplingParams(
            max_tokens=1024,
            truncate_prompt_tokens=512,
            truncation_side="right",
        )
        await provider._generate_completion(client, request, params)

        extra_body = client.completions.create.call_args.kwargs["extra_body"]
        assert extra_body["truncate_prompt_tokens"] == 512
        assert extra_body["truncation_side"] == "right"

    @pytest.mark.anyio
    async def test_generate_completion_preserves_exact_explicit_truncation(self, provider):
        """The provider must not silently lower an explicit prompt truncation limit."""
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=self._make_completion_response())

        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        params = SamplingParams(max_tokens=1024, truncate_prompt_tokens=3500)
        await provider._generate_completion(client, request, params)

        extra_body = client.completions.create.call_args.kwargs["extra_body"]
        assert extra_body["truncate_prompt_tokens"] == 3500
        assert "truncation_side" not in extra_body

    @pytest.mark.anyio
    async def test_generate_completion_leaves_context_validation_to_server(self, provider):
        """Finite generation must not depend on local context discovery or validation."""
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=self._make_completion_response())

        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        await provider._generate_completion(client, request, SamplingParams(max_tokens=4096))

        call_kwargs = client.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 4096
        assert "truncate_prompt_tokens" not in call_kwargs["extra_body"]

    @pytest.mark.anyio
    async def test_generate_chat_omits_max_tokens_when_none(self, provider):
        """max_tokens=None means uncapped: omit the field rather than sending null."""
        provider.chat_template_kwargs = None
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._make_chat_response())

        request = LMRequest(
            request_type=RequestType.CHAT, messages=[{"role": "user", "content": "Hi"}]
        )
        await provider._generate_chat(client, request, SamplingParams(max_tokens=None))

        assert "max_tokens" not in client.chat.completions.create.call_args.kwargs

    @pytest.mark.anyio
    async def test_generate_chat_includes_max_tokens_when_set(self, provider):
        """An explicit max_tokens is still forwarded to the chat payload."""
        provider.chat_template_kwargs = None
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._make_chat_response())

        request = LMRequest(
            request_type=RequestType.CHAT, messages=[{"role": "user", "content": "Hi"}]
        )
        await provider._generate_chat(client, request, SamplingParams(max_tokens=128))

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 128
        assert "extra_body" not in call_kwargs

    @pytest.mark.anyio
    async def test_chat_logprobs_preserve_termination_and_usage(self, provider):
        provider.chat_template_kwargs = None
        response = self._make_chat_response()
        response.usage = SimpleNamespace(prompt_tokens=12, completion_tokens=1)
        choice = response.choices[0]
        choice.finish_reason = "length"
        choice.logprobs = SimpleNamespace(
            content=[SimpleNamespace(token="reply", logprob=-0.5, bytes=None)]
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)
        request = LMRequest(
            request_type=RequestType.CHAT, messages=[{"role": "user", "content": "Hi"}]
        )
        outputs = await provider._generate_chat(client, request, SamplingParams(max_tokens=1))
        assert outputs[0].metadata["finish_reason"] == "length"
        assert outputs[0].metadata["completion_tokens"] == 1
        assert outputs[0].metadata["prompt_tokens"] == 12
        assert outputs[0].metadata["sum_logits"] == -0.5

    def _prompt_server(self, replies, posted):
        """A server whose /tokenize and /detokenize answer from ``replies`` by endpoint."""

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, json):
                posted.append((url, json))
                body = replies[url.rsplit("/", 1)[1]]
                return MagicMock(
                    json=MagicMock(return_value=body(json) if callable(body) else body)
                )

        return patch(
            "olmo_eval.inference.providers.vllm_server.httpx.AsyncClient",
            return_value=FakeClient(),
        )

    def _unstubbed(self, provider):
        """Give the provider back its real prompt and tag lookups."""
        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        del provider._request_opens_think
        del provider._get_think_close_ids
        assert isinstance(provider, VLLMServerProvider)
        provider.chat_template_kwargs = None
        return provider

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("detokenized", "opens"),
        [
            # A byte-level tokenizer reports the newline as "Ċ"; the detokenized text decides.
            ({"prompt": "<|im_start|>assistant\n<think>\n"}, True),
            ({"prompt": "<|im_start|>assistant\n<think>"}, True),
            ({"prompt": "<|im_start|>assistant\n"}, False),
        ],
    )
    async def test_the_prompt_end_is_read_from_the_request_as_sent(
        self, provider, detokenized, opens
    ):
        provider = self._unstubbed(provider)
        provider.chat_template_kwargs = {"enable_thinking": True}
        tokenized = {"tokens": list(range(20)), "token_strs": ["<", "think", ">", "Ċ"]}
        posted = []
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ]
        tools = [{"type": "function", "function": {"name": "f"}}]

        with self._prompt_server({"tokenize": tokenized, "detokenize": detokenized}, posted):
            assert await provider._request_opens_think(messages, tools) is opens
            assert await provider._request_opens_think(messages, tools) is opens

        # Each request is tokenized as it will be sent; one detokenize serves a shared tail.
        urls = [url.rsplit("/", 1)[1] for url, _ in posted]
        assert urls == ["tokenize", "detokenize", "tokenize"]
        assert posted[0][1]["messages"] == messages
        assert posted[0][1]["tools"] == tools
        assert posted[0][1]["chat_template_kwargs"] == {"enable_thinking": True}
        assert posted[0][1]["add_generation_prompt"] is True
        assert posted[1][1]["tokens"] == list(range(4, 20))

    @pytest.mark.anyio
    async def test_requests_that_render_differently_are_answered_separately(self, provider):
        """A system turn that turns thinking off must not inherit another request's answer."""
        provider = self._unstubbed(provider)

        def tokenize(payload):
            thinking = payload["messages"][0]["role"] != "system"
            return {"tokens": [1, 2, 3] if thinking else [1, 2, 4]}

        def detokenize(payload):
            return {"prompt": "<think>\n" if payload["tokens"][-1] == 3 else "assistant\n"}

        user = {"role": "user", "content": "Hi"}
        with self._prompt_server({"tokenize": tokenize, "detokenize": detokenize}, []):
            assert await provider._request_opens_think([user], None) is True
            off = [{"role": "system", "content": "/no_think"}, user]
            assert await provider._request_opens_think(off, None) is False
            assert await provider._request_opens_think([user], None) is True

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("tokenized", "detokenized"),
        [
            ({}, {"prompt": "<think>"}),
            ({"tokens": []}, {"prompt": "<think>"}),
            (["not", "a", "dict"], {"prompt": "<think>"}),
            ({"tokens": [1, 2]}, {}),
            ({"tokens": [1, 2]}, ["not", "a", "dict"]),
        ],
    )
    async def test_an_unusable_reply_leaves_the_prompt_unknown(
        self, provider, tokenized, detokenized, caplog
    ):
        provider = self._unstubbed(provider)
        user = [{"role": "user", "content": "Hi"}]
        replies = {"tokenize": tokenized, "detokenize": detokenized}

        with self._prompt_server(replies, []), caplog.at_level("ERROR"):
            assert await provider._request_opens_think(user, None) is None
            assert await provider._request_opens_think(user, None) is None

        # Reported once, not per request.
        assert caplog.text.count("Could not read the end of the rendered chat prompt") == 1

    @pytest.mark.anyio
    async def test_an_unreachable_server_leaves_the_prompt_unknown(self, provider, caplog):
        import httpx

        provider = self._unstubbed(provider)
        with (
            patch(
                "olmo_eval.inference.providers.vllm_server.httpx.AsyncClient",
                side_effect=httpx.ConnectError("no server"),
            ),
            caplog.at_level("ERROR"),
        ):
            assert (
                await provider._request_opens_think([{"role": "user", "content": "Hi"}], None)
                is None
            )
            assert await provider._get_think_close_ids() is None

        assert "thinking_prefilled=auto" in caplog.text

    @pytest.mark.anyio
    async def test_the_closing_tag_ids_are_asked_once(self, provider):
        provider = self._unstubbed(provider)
        posted = []

        with self._prompt_server({"tokenize": {"tokens": [27, 1, 28]}}, posted):
            assert await provider._get_think_close_ids() == [27, 1, 28]
            assert await provider._get_think_close_ids() == [27, 1, 28]

        assert len(posted) == 1
        assert posted[0][1]["prompt"] == "</think>"
        assert posted[0][1]["add_special_tokens"] is False

    @pytest.mark.parametrize(
        ("text", "token_ids", "token_strings", "close_ids", "closed"),
        [
            # The tag in the content is enough.
            ("plan</think>4", None, None, None, True),
            # Generated ids decide when the server sent them.
            ("4", [5, 9, 6], ["a", "", "b"], [9], True),
            ("trace", [5, 6], ["a", "b"], [9], False),
            ("4", [5, 27, 1, 28, 6], None, [27, 1, 28], True),
            ("trace", [5, 27, 28, 1], None, [27, 1, 28], False),
            # Token strings: the tag's pieces always show when it spans several tokens.
            ("4", None, ["plan", "</", "think", ">", "4"], [27, 1, 28], True),
            ("trace", None, ["tr", "ace"], [27, 1, 28], False),
            # A single-token tag can decode to nothing, so its absence shows nothing.
            ("4", None, ["plan", "", "4"], [9], None),
            ("4", None, ["plan", "</think>", "4"], [9], True),
            # No tag ids, no tokens, or tokens given as ids: nothing to read.
            ("4", None, ["plan", "4"], None, None),
            ("4", None, None, [27, 1, 28], None),
            ("4", None, [], [27, 1, 28], None),
            ("4", None, ["token_id:5", "token_id:6"], [27, 1, 28], None),
        ],
    )
    def test_closure_is_read_from_what_the_server_returned(
        self, text, token_ids, token_strings, close_ids, closed
    ):
        from olmo_eval.inference.providers.vllm_server import server_think_closed

        assert (
            server_think_closed(
                text, token_ids=token_ids, token_strings=token_strings, close_ids=close_ids
            )
            is closed
        )

    def _chat_choice(self, content, finish_reason, tokens, reasoning=None, token_ids=None):
        choice = MagicMock()
        choice.message = SimpleNamespace(
            content=content, tool_calls=None, reasoning_content=reasoning
        )
        choice.finish_reason = finish_reason
        choice.logprobs = SimpleNamespace(
            content=[SimpleNamespace(token=token, logprob=-0.1, bytes=None) for token in tokens]
        )
        choice.token_ids = token_ids
        choice.model_extra = {}
        return choice

    @pytest.mark.anyio
    async def test_chat_closure_does_not_follow_the_finish_reason_or_reasoning_field(
        self, provider
    ):
        """The cases a finish-reason rule gets wrong, with a reasoning parser on the server."""
        provider.chat_template_kwargs = None
        provider._request_opens_think = AsyncMock(return_value=True)
        provider._get_think_close_ids = AsyncMock(return_value=[27, 1, 28])
        response = self._make_chat_response()
        response.choices = [
            # The trace ended at end of turn without closing.
            self._chat_choice("so it might be", "stop", ["so", " it", " might", " be"]),
            # The trace closed, then the answer ran to the token limit.
            self._chat_choice(
                "The answer is 4 beca", "length", ["plan", "</", "think", ">", "The"]
            ),
            # The parser returned an unfinished trace as reasoning, with no content.
            self._chat_choice(None, "length", ["step", " 1"], reasoning="step 1"),
        ]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)
        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=[{"role": "user", "content": "Hi"}],
            record_thinking=True,
        )

        outputs = await provider._generate_chat(client, request, SamplingParams(num_samples=3))

        from olmo_eval.evals.extract import trace_state

        assert all(out.metadata["prompt_opens_think"] is True for out in outputs)
        assert [out.metadata["think_closed"] for out in outputs] == [False, True, False]
        assert [trace_state(out.text, out.metadata) for out in outputs] == [
            "unclosed",
            "closed",
            "unclosed",
        ]
        # Several-token tag: the token strings suffice, so no token ids are requested.
        assert "extra_body" not in client.chat.completions.create.call_args.kwargs
        saved = build_predictions(
            [Response(instance=Instance(question="Hi"), request=request, outputs=outputs)]
        )[0]["model_output"]
        assert [out["think_closed"] for out in saved] == [False, True, False]

    @pytest.mark.anyio
    async def test_chat_asks_for_token_ids_when_the_tag_is_one_token(self, provider):
        provider.chat_template_kwargs = None
        provider._request_opens_think = AsyncMock(return_value=True)
        provider._get_think_close_ids = AsyncMock(return_value=[9])
        response = self._make_chat_response()
        response.choices = [
            self._chat_choice("The answer is 4", "stop", ["plan", "", "4"], token_ids=[5, 9, 6]),
            self._chat_choice("so it might be", "stop", ["so", "it"], token_ids=[5, 6]),
            # A server that returned no ids leaves a single-token tag undecided.
            self._chat_choice("so it might be", "stop", ["so", "it"]),
        ]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)
        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=[{"role": "user", "content": "Hi"}],
            record_thinking=True,
        )

        outputs = await provider._generate_chat(client, request, SamplingParams(num_samples=3))

        sent = client.chat.completions.create.call_args.kwargs
        assert sent["extra_body"]["return_token_ids"] is True
        assert [out.metadata.get("think_closed") for out in outputs] == [True, False, None]

        from olmo_eval.evals.extract import trace_state

        assert trace_state(outputs[2].text, outputs[2].metadata) == "unknown"

    @pytest.mark.anyio
    async def test_chat_outputs_record_an_unknown_prompt(self, provider):
        provider.chat_template_kwargs = None
        provider._request_opens_think = AsyncMock(return_value=None)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._make_chat_response())
        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=[{"role": "user", "content": "Hi"}],
            record_thinking=True,
        )

        outputs = await provider._generate_chat(client, request, SamplingParams(max_tokens=4))

        assert outputs[0].metadata["prompt_opens_think"] is None
        assert "think_closed" not in outputs[0].metadata

    @pytest.mark.anyio
    async def test_chat_outputs_are_unmarked_when_the_prompt_opens_no_trace(self, provider):
        provider.chat_template_kwargs = None
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._make_chat_response())
        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=[{"role": "user", "content": "Hi"}],
            record_thinking=True,
        )

        outputs = await provider._generate_chat(client, request, SamplingParams(max_tokens=4))

        assert "prompt_opens_think" not in outputs[0].metadata
        assert "think_closed" not in outputs[0].metadata
        provider._get_think_close_ids.assert_not_awaited()

    @pytest.mark.anyio
    @pytest.mark.parametrize("request_type", [RequestType.CHAT, RequestType.COMPLETION])
    async def test_a_request_that_does_not_ask_is_sent_exactly_as_before(
        self, provider, request_type
    ):
        """No server lookup, no added request field and no record, whatever the template."""
        provider = self._unstubbed(provider)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._make_chat_response())
        client.completions.create = AsyncMock(return_value=self._make_completion_response())
        params = SamplingParams(max_tokens=4, temperature=0.6, top_p=0.95)
        posted = []
        # A template that opens a trace, with a single-token closing tag: the case
        # that would add ``return_token_ids`` if the request had asked.
        server = {"tokenize": {"tokens": [9]}, "detokenize": {"prompt": "<think>\n"}}

        with self._prompt_server(server, posted):
            if request_type == RequestType.CHAT:
                plain = LMRequest(request_type, messages=[{"role": "user", "content": "Hi"}])
                outputs = await provider._generate_chat(client, plain, params)
                sent = dict(client.chat.completions.create.call_args.kwargs)
                asking = LMRequest(
                    request_type, messages=[{"role": "user", "content": "Hi"}], record_thinking=True
                )
                await provider._generate_chat(client, asking, params)
                sent_when_asking = client.chat.completions.create.call_args.kwargs
            else:
                plain = LMRequest(request_type, prompt="Q<think>\n")
                outputs = await provider._generate_completion(client, plain, params)
                sent = dict(client.completions.create.call_args.kwargs)
                posted_before = list(posted)
                asking = LMRequest(request_type, prompt="Q<think>\n", record_thinking=True)
                await provider._generate_completion(client, asking, params)
                sent_when_asking = client.completions.create.call_args.kwargs
                assert posted_before == []

        assert "prompt_opens_think" not in outputs[0].metadata
        assert "think_closed" not in outputs[0].metadata
        if request_type == RequestType.CHAT:
            # The request as it was before this record existed.
            assert sent == {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "n": 1,
                "max_tokens": 4,
                "temperature": 0.6,
                "top_p": 0.95,
                "logprobs": True,
                "top_logprobs": 1,
            }
            assert sent_when_asking["extra_body"] == {"return_token_ids": True}
            # Only the asking request reached /tokenize and /detokenize.
            assert [json["messages"] for url, json in posted if "messages" in json] == [
                asking.messages
            ]
        else:
            assert sent == sent_when_asking
            assert len(posted) == 1  # the closing tag's ids, for the asking request

    @pytest.mark.anyio
    async def test_completion_outputs_record_a_prompt_that_opens_a_trace(self, provider):
        provider._get_think_close_ids = AsyncMock(return_value=[27, 1, 28])
        response = self._make_completion_response()
        response.choices[0].logprobs = SimpleNamespace(
            tokens=["test", " completion"], token_logprobs=[-0.1, -0.2], top_logprobs=[]
        )
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=response)
        params = SamplingParams(max_tokens=8)

        opened = await provider._generate_completion(
            client,
            LMRequest(
                request_type=RequestType.COMPLETION, prompt="Q<think>\n", record_thinking=True
            ),
            params,
        )
        plain = await provider._generate_completion(
            client, LMRequest(request_type=RequestType.COMPLETION, prompt="Q"), params
        )

        assert opened[0].metadata["prompt_opens_think"] is True
        # Read from the completion's own token strings.
        assert opened[0].metadata["think_closed"] is False
        assert "prompt_opens_think" not in plain[0].metadata

    @pytest.mark.anyio
    async def test_chat_usage_is_not_attributed_to_each_of_several_choices(self, provider):
        """Response usage sums all choices, so no single output may claim it."""
        provider.chat_template_kwargs = None
        response = self._make_chat_response()
        second = MagicMock()
        second.message = MagicMock(content="second reply", tool_calls=None)
        second.logprobs = None
        response.choices.append(second)
        for choice, reason in zip(response.choices, ("stop", "length"), strict=True):
            choice.finish_reason = reason
        response.usage = SimpleNamespace(prompt_tokens=12, completion_tokens=100)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)
        request = LMRequest(
            request_type=RequestType.CHAT, messages=[{"role": "user", "content": "Hi"}]
        )
        outputs = await provider._generate_chat(
            client, request, SamplingParams(max_tokens=100, num_samples=2)
        )
        assert [out.metadata["prompt_tokens"] for out in outputs] == [12, 12]
        assert all("completion_tokens" not in out.metadata for out in outputs)

        scored = Response(instance=Instance(question="Hi"), request=request, outputs=outputs)
        persisted = build_predictions([scored])[0]["model_output"]
        assert [out["finish_reason"] for out in persisted] == ["stop", "length"]
        assert all("completion_tokens" not in out for out in persisted)

    @pytest.mark.anyio
    async def test_completion_usage_is_not_attributed_to_each_of_several_choices(self, provider):
        """The SDK completions path keeps only per-output counts."""
        response = self._make_completion_response()
        response.choices.append(MagicMock(text="second", logprobs=None, finish_reason="length"))
        response.usage = SimpleNamespace(prompt_tokens=3, completion_tokens=100)
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=response)
        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        outputs = await provider._generate_completion(
            client, request, SamplingParams(max_tokens=100, num_samples=2)
        )
        assert [out.metadata["prompt_tokens"] for out in outputs] == [3, 3]
        assert all("completion_tokens" not in out.metadata for out in outputs)

    @pytest.mark.anyio
    async def test_completion_usage_is_kept_for_a_single_choice(self, provider):
        response = self._make_completion_response()
        response.usage = SimpleNamespace(prompt_tokens=3, completion_tokens=7)
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=response)
        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        outputs = await provider._generate_completion(
            client, request, SamplingParams(max_tokens=100)
        )
        assert outputs[0].metadata["completion_tokens"] == 7

    @pytest.mark.anyio
    async def test_generate_chat_preserves_exact_explicit_truncation(self, provider):
        """Chat generation forwards caller-requested truncation without tightening it."""
        provider.chat_template_kwargs = None
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._make_chat_response())
        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=[{"role": "user", "content": "Hi"}],
        )
        params = SamplingParams(
            max_tokens=1024,
            truncate_prompt_tokens=3500,
            truncation_side="right",
        )

        await provider._generate_chat(client, request, params)

        extra_body = client.chat.completions.create.call_args.kwargs["extra_body"]
        assert extra_body["truncate_prompt_tokens"] == 3500
        assert extra_body["truncation_side"] == "right"

        trace = provider.describe_request(request, params)
        assert trace is not None
        assert trace["generation_kwargs"]["truncate_prompt_tokens"] == 3500
        assert trace["generation_kwargs"]["truncation_side"] == "right"

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
        assert "truncate_prompt_tokens" not in payload
        assert "truncation_side" not in payload
        client.completions.create.assert_not_called()

    @pytest.mark.anyio
    async def test_raw_completion_usage_is_not_attributed_to_each_of_several_choices(self):
        """The prompt-token-ids completions path keeps only per-output counts."""
        provider = self._make_provider(completion_use_prompt_token_ids=True)
        local_tokenizer = MagicMock()
        local_tokenizer.encode.return_value = [11, 12, 13]
        local_tokenizer.eos_token_id = None
        local_tokenizer.eos_token = None
        provider._get_tokenizer = MagicMock(return_value=local_tokenizer)
        raw_response = MagicMock()
        raw_response.raise_for_status = MagicMock()
        raw_response.json.return_value = {
            "choices": [{"text": "a", "logprobs": None}, {"text": "b", "logprobs": None}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 100},
        }
        mock_http = AsyncMock()
        mock_http.post.return_value = raw_response
        provider._get_raw_http_client = MagicMock(return_value=mock_http)
        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        outputs = await provider._generate_completion(
            MagicMock(), request, SamplingParams(max_tokens=100, num_samples=2)
        )
        assert [out.metadata["prompt_tokens"] for out in outputs] == [3, 3]
        assert all("completion_tokens" not in out.metadata for out in outputs)

    def test_build_completion_output_sets_is_greedy_from_top_logprobs(self, provider):
        """Completion metadata should expose greedy status when top logprobs are available."""
        output = provider._build_completion_output(
            text=" yes",
            logprobs_payload={
                "tokens": [" yes"],
                "token_logprobs": [-0.1],
                "top_logprobs": [{" yes": -0.1}],
            },
            usage=None,
            stop_sequences=None,
        )

        assert output.metadata["is_greedy"] is True

    def test_build_completion_output_detects_non_greedy_from_top_logprobs(self, provider):
        """Completion metadata should mark sampled non-argmax tokens as non-greedy."""
        output = provider._build_completion_output(
            text=" no",
            logprobs_payload={
                "tokens": [" no"],
                "token_logprobs": [-0.7],
                "top_logprobs": [{" yes": -0.1}],
            },
            usage=None,
            stop_sequences=None,
        )

        assert output.metadata["is_greedy"] is False

    def test_build_completion_output_omits_unknown_is_greedy(self, provider):
        """Completion metadata should not invent greedy status without top logprobs."""
        output = provider._build_completion_output(
            text=" yes",
            logprobs_payload={
                "tokens": [" yes"],
                "token_logprobs": [-0.1],
            },
            usage=None,
            stop_sequences=None,
        )

        assert "is_greedy" not in output.metadata

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

    def test_local_tokenizer_load_uses_force_download(self):
        """Local tokenizer loads should refresh cache when force_download is enabled."""
        provider = self._make_provider(
            tokenizer="custom-tokenizer",
            revision="stage2-step47684",
            trust_remote_code=True,
            force_download=True,
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
            force_download=True,
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
    async def test_logprobs_does_not_truncate_when_context_length_is_unknown(self, provider):
        """Unknown server metadata must not impose a guessed loglikelihood window."""
        provider._max_length = None
        provider._max_length_source = None
        provider._max_length_discovery_attempted = True
        context_tokens = list(range(4097))
        continuation_tokens = [5000]

        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.return_value = (context_tokens, continuation_tokens)
            prompt_logprobs = [None] * len(context_tokens) + [
                {"5000": {"logprob": -0.1, "decoded_token": "answer"}}
            ]
            mock_http = AsyncMock()
            mock_http.post.return_value = self._make_vllm_response(prompt_logprobs)
            provider._get_raw_http_client = MagicMock(return_value=mock_http)

            request = LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt="long prompt",
                continuations=(" answer",),
            )
            outputs = await provider._logprobs_single_impl(request)

        payload = mock_http.post.call_args.kwargs["json"]
        assert payload["prompt"] == context_tokens + continuation_tokens
        assert len(outputs) == 1
        assert outputs[0].metadata["num_tokens"] == 1
        # num_tokens_all counts context plus continuation, as sent.
        assert outputs[0].metadata["num_tokens_all"] == len(context_tokens) + 1

    @pytest.mark.anyio
    async def test_logprobs_respects_authoritative_request_max_length(self, provider):
        """An explicit task limit still enables intentional loglikelihood truncation."""
        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.return_value = ([0, 1, 2, 3], [4])
            prompt_logprobs = [
                None,
                {"3": {"logprob": -0.2, "decoded_token": "ctx"}},
                {"4": {"logprob": -0.1, "decoded_token": "answer"}},
            ]
            mock_http = AsyncMock()
            mock_http.post.return_value = self._make_vllm_response(prompt_logprobs)
            provider._get_raw_http_client = MagicMock(return_value=mock_http)

            request = LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt="prompt",
                continuations=(" answer",),
                max_length=4,
            )
            outputs = await provider._logprobs_single_impl(request)

        payload = mock_http.post.call_args.kwargs["json"]
        assert payload["prompt"] == [2, 3, 4]
        assert len(outputs) == 1
        assert outputs[0].metadata["num_tokens"] == 1
        # After left-truncation, num_tokens_all is what was actually sent.
        assert outputs[0].metadata["num_tokens_all"] == 3

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
