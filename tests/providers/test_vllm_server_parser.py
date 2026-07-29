"""Tests for vLLM server tool-call parser inference."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from olmo_eval.inference.providers.vllm_server_utils import (
    _build_server_command,
    _chat_template_has_function_tag,
    _chat_template_has_think_tag,
    _infer_reasoning_parser,
    _infer_tool_call_parser,
    _load_chat_template,
)


@pytest.fixture(autouse=True)
def clear_chat_template_cache():
    """Keep cached tokenizer probes from leaking across tests."""
    _load_chat_template.cache_clear()
    _chat_template_has_function_tag.cache_clear()
    _chat_template_has_think_tag.cache_clear()
    yield
    _load_chat_template.cache_clear()
    _chat_template_has_function_tag.cache_clear()
    _chat_template_has_think_tag.cache_clear()


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


class TestChatTemplateHasThinkTag:
    """Tests for thinking-template detection."""

    def test_template_opening_think_returns_true(self):
        tokenizer = MagicMock()
        tokenizer.get_chat_template.return_value = "{{ '<|im_start|>assistant\\n<think>\\n' }}"

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            assert _chat_template_has_think_tag("test-think-template-true") is True

    def test_template_without_think_returns_false(self):
        tokenizer = MagicMock()
        tokenizer.get_chat_template.return_value = "{{ '<|im_start|>assistant\\n' }}"

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            assert _chat_template_has_think_tag("test-think-template-false") is False

    def test_unavailable_template_returns_none(self):
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=RuntimeError("failed"),
        ):
            assert _chat_template_has_think_tag("test-think-template-raises") is None


class TestInferReasoningParser:
    """Tests for reasoning parser inference.

    A thinking model served without a reasoning parser returns its monologue and
    its answer concatenated in one content field, so the flag has to be inferred;
    naming a parser for a model that never emits its delimiters would empty every
    answer instead, so absent evidence the flag has to be omitted.
    """

    @pytest.mark.parametrize(
        ("model_name", "parser"),
        [
            ("Qwen/Qwen3.5-35B-A3B", "qwen3"),
            ("allenai/OLMo-3-32B-Think", "olmo3"),
            ("deepseek-ai/DeepSeek-R1", "deepseek_r1"),
        ],
    )
    def test_thinking_template_selects_family_parser(self, model_name, parser):
        with patch(
            "olmo_eval.inference.providers.vllm_server_utils._chat_template_has_think_tag",
            return_value=True,
        ):
            assert _infer_reasoning_parser(model_name) == parser

    def test_non_thinking_template_returns_none(self):
        with patch(
            "olmo_eval.inference.providers.vllm_server_utils._chat_template_has_think_tag",
            return_value=False,
        ):
            assert _infer_reasoning_parser("Qwen/Qwen2.5-7B-Instruct") is None

    def test_unreadable_template_returns_none(self):
        with patch(
            "olmo_eval.inference.providers.vllm_server_utils._chat_template_has_think_tag",
            return_value=None,
        ):
            assert _infer_reasoning_parser("org/unreachable-model") is None

    def test_forwards_trust_remote_code_and_revision(self):
        with patch(
            "olmo_eval.inference.providers.vllm_server_utils._chat_template_has_think_tag",
            return_value=True,
        ) as probe:
            _infer_reasoning_parser("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True, revision="main")

        probe.assert_called_once_with("Qwen/Qwen3.5-35B-A3B", True, "main")


class TestServerCommandReasoningParser:
    """Tests that the inferred reasoning parser reaches the vLLM command line."""

    def test_thinking_model_gets_reasoning_parser_flag(self):
        with patch(
            "olmo_eval.inference.providers.vllm_server_utils._infer_reasoning_parser",
            return_value="qwen3",
        ):
            cmd = _build_server_command("Qwen/Qwen3.5-35B-A3B", port=8000)

        assert "--reasoning-parser" in cmd
        assert cmd[cmd.index("--reasoning-parser") + 1] == "qwen3"

    def test_non_thinking_model_omits_the_flag(self):
        with patch(
            "olmo_eval.inference.providers.vllm_server_utils._infer_reasoning_parser",
            return_value=None,
        ):
            cmd = _build_server_command("org/plain-instruct", port=8000)

        assert "--reasoning-parser" not in cmd

    def test_explicit_parser_overrides_inference(self):
        with patch(
            "olmo_eval.inference.providers.vllm_server_utils._infer_reasoning_parser"
        ) as probe:
            cmd = _build_server_command(
                "Qwen/Qwen3.5-35B-A3B", port=8000, reasoning_parser="deepseek_r1"
            )

        probe.assert_not_called()
        assert cmd[cmd.index("--reasoning-parser") + 1] == "deepseek_r1"
        assert cmd.count("--reasoning-parser") == 1

    def test_empty_string_suppresses_the_flag(self):
        with patch(
            "olmo_eval.inference.providers.vllm_server_utils._infer_reasoning_parser"
        ) as probe:
            cmd = _build_server_command("Qwen/Qwen3.5-35B-A3B", port=8000, reasoning_parser="")

        probe.assert_not_called()
        assert "--reasoning-parser" not in cmd

    def test_inference_probes_the_tokenizer_when_one_is_given(self):
        with patch(
            "olmo_eval.inference.providers.vllm_server_utils._infer_reasoning_parser",
            return_value="qwen3",
        ) as probe:
            _build_server_command(
                "/weka/checkpoints/step-1000",
                port=8000,
                tokenizer="Qwen/Qwen3.5-35B-A3B",
                revision="main",
            )

        probe.assert_called_once_with(
            "Qwen/Qwen3.5-35B-A3B", trust_remote_code=False, revision="main"
        )
