"""Tests for vLLM server tool-call parser inference."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from olmo_eval.inference.providers.vllm_server_utils import (
    _chat_template_has_function_tag,
    _infer_tool_call_parser,
)


@pytest.fixture(autouse=True)
def clear_chat_template_cache():
    """Keep cached tokenizer probes from leaking across tests."""
    _chat_template_has_function_tag.cache_clear()
    yield
    _chat_template_has_function_tag.cache_clear()


class TestChatTemplateHasFunctionTag:
    """Tests for chat template inspection."""

    def test_get_chat_template_with_function_tag_returns_true(self):
        tokenizer = MagicMock()
        tokenizer.get_chat_template.return_value = "tools: <function=name>"

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            assert _chat_template_has_function_tag("test-parser-template-true") is True

    def test_get_chat_template_without_function_tag_returns_false(self):
        tokenizer = MagicMock()
        tokenizer.get_chat_template.return_value = "tools: {{ json_tools }}"

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            assert _chat_template_has_function_tag("test-parser-template-false") is False

    def test_chat_template_attribute_with_function_tag_returns_true(self):
        tokenizer = SimpleNamespace(chat_template="tools: <function=name>")

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            assert _chat_template_has_function_tag("test-parser-attr-template-true") is True

    def test_chat_template_attribute_none_returns_none(self):
        tokenizer = SimpleNamespace(chat_template=None)

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            assert _chat_template_has_function_tag("test-parser-attr-template-none") is None

    def test_from_pretrained_raises_returns_none(self):
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=RuntimeError("failed"),
        ):
            assert _chat_template_has_function_tag("test-parser-tokenizer-raises") is None

    def test_forwards_trust_remote_code_and_revision(self):
        tokenizer = MagicMock()
        tokenizer.get_chat_template.return_value = "tools: <function=name>"

        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
        ) as from_pretrained:
            assert (
                _chat_template_has_function_tag(
                    "test-parser-forward-kwargs",
                    trust_remote_code=True,
                    revision="main",
                )
                is True
            )

        from_pretrained.assert_called_once_with(
            "test-parser-forward-kwargs",
            trust_remote_code=True,
            revision="main",
        )


class TestInferToolCallParser:
    """Tests for parser inference branches."""

    @pytest.mark.parametrize(
        ("model_name", "parser"),
        [
            ("org/test-llama-parser", "llama3_json"),
            ("org/test-mistral-parser", "mistral"),
            ("org/test-olmo-parser", "olmo3"),
        ],
    )
    def test_name_shortcuts_skip_template_probe(self, model_name, parser):
        with patch(
            "olmo_eval.inference.providers.vllm_server_utils._chat_template_has_function_tag"
        ) as probe:
            assert _infer_tool_call_parser(model_name) == parser

        probe.assert_not_called()

    def test_template_tag_true_returns_qwen3_coder(self):
        with patch(
            "olmo_eval.inference.providers.vllm_server_utils._chat_template_has_function_tag",
            return_value=True,
        ) as probe:
            assert _infer_tool_call_parser("org/test-template-only-parser") == "qwen3_coder"

        probe.assert_called_once_with("org/test-template-only-parser", False, None)

    @pytest.mark.parametrize(
        "model_name",
        [
            "org/test-qwen3-coder-parser",
            "org/test-qwen3.5-parser",
            "org/test-qwen3.6-parser",
        ],
    )
    def test_probe_none_with_qwen3_name_returns_qwen3_coder(self, model_name, caplog):
        with (
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._chat_template_has_function_tag",
                return_value=None,
            ),
            caplog.at_level(
                logging.WARNING, logger="olmo_eval.inference.providers.vllm_server_utils"
            ),
        ):
            assert _infer_tool_call_parser(model_name) == "qwen3_coder"

        assert caplog.records == []

    def test_probe_none_with_unrelated_name_returns_hermes(self):
        with patch(
            "olmo_eval.inference.providers.vllm_server_utils._chat_template_has_function_tag",
            return_value=None,
        ):
            assert _infer_tool_call_parser("org/test-unrelated-parser-none") == "hermes"

    def test_probe_false_with_qwen35_name_returns_hermes_and_warns(self, caplog):
        with (
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._chat_template_has_function_tag",
                return_value=False,
            ),
            caplog.at_level(
                logging.WARNING, logger="olmo_eval.inference.providers.vllm_server_utils"
            ),
        ):
            assert _infer_tool_call_parser("org/test-qwen3.5-parser-false") == "hermes"

        assert len(caplog.records) == 1
        assert "org/test-qwen3.5-parser-false" in caplog.text
        assert "does not contain '<function='" in caplog.text
        assert "using 'hermes'" in caplog.text
        assert "pass tool_call_parser explicitly to override" in caplog.text

    def test_probe_false_with_unrelated_name_returns_hermes_without_warning(self, caplog):
        with (
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._chat_template_has_function_tag",
                return_value=False,
            ),
            caplog.at_level(
                logging.WARNING, logger="olmo_eval.inference.providers.vllm_server_utils"
            ),
        ):
            assert _infer_tool_call_parser("org/test-unrelated-parser-false") == "hermes"

        assert caplog.records == []
