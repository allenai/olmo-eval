"""Tests for chat templates."""

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader


@pytest.fixture
def template_env():
    """Create Jinja2 environment with the templates directory."""
    templates_dir = (
        Path(__file__).parent.parent.parent / "src" / "olmo_eval" / "inference" / "templates"
    )
    return Environment(loader=FileSystemLoader(templates_dir))


@pytest.fixture
def olmo3_template(template_env):
    """Load the OLMo3 tool chat template."""
    return template_env.get_template("tool_chat_template_olmo3.jinja")


class TestOlmo3ToolChatTemplate:
    """Tests for the OLMo3 tool chat template."""

    def test_empty_messages_uses_default_system(self, olmo3_template):
        """Empty message list uses default system message without error."""
        result = olmo3_template.render(
            messages=[],
            bos_token="<s>",
            add_generation_prompt=True,
        )
        assert "You are a helpful function-calling AI assistant" in result
        assert "<|im_start|>system" in result

    def test_system_message_extracted(self, olmo3_template):
        """System message is properly extracted from first message."""
        result = olmo3_template.render(
            messages=[{"role": "system", "content": "You are a custom assistant."}],
            bos_token="<s>",
            add_generation_prompt=True,
        )
        assert "You are a custom assistant." in result
        assert "You are a helpful function-calling AI assistant" not in result

    def test_user_message_rendered(self, olmo3_template):
        """User messages are rendered correctly."""
        result = olmo3_template.render(
            messages=[{"role": "user", "content": "Hello!"}],
            bos_token="<s>",
            add_generation_prompt=True,
        )
        assert "<|im_start|>user\nHello!<|im_end|>" in result

    def test_tool_call_with_apostrophe(self, olmo3_template):
        """Tool calls with apostrophes in arguments are properly escaped."""
        result = olmo3_template.render(
            messages=[
                {"role": "user", "content": "Help me"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit",
                                "arguments": {"answer": "Model's response"},
                            }
                        }
                    ],
                },
            ],
            bos_token="<s>",
            add_generation_prompt=False,
        )
        # The template escapes single quotes in string values
        assert "answer='Model\\'s response'" in result

    def test_tool_call_with_backslash(self, olmo3_template):
        """Tool calls with backslashes in arguments are properly escaped."""
        result = olmo3_template.render(
            messages=[
                {"role": "user", "content": "Help me"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit",
                                "arguments": {"path": "C:\\Users\\test"},
                            }
                        }
                    ],
                },
            ],
            bos_token="<s>",
            add_generation_prompt=False,
        )
        # The template escapes backslashes
        assert "path='C:\\\\Users\\\\test'" in result

    def test_tool_call_with_newline(self, olmo3_template):
        """Tool calls with newlines in arguments are properly escaped."""
        result = olmo3_template.render(
            messages=[
                {"role": "user", "content": "Help me"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit",
                                "arguments": {"text": "line1\nline2"},
                            }
                        }
                    ],
                },
            ],
            bos_token="<s>",
            add_generation_prompt=False,
        )
        # The template escapes newlines
        assert "text='line1\\nline2'" in result

    def test_tools_rendered_as_json(self, olmo3_template):
        """Tools are rendered as JSON in system message."""
        result = olmo3_template.render(
            messages=[{"role": "user", "content": "Help me"}],
            bos_token="<s>",
            tools=[{"function": {"name": "get_weather", "parameters": {}}}],
            add_generation_prompt=True,
        )
        assert "get_weather" in result
        assert "<functions>" in result

    def test_generation_prompt_added(self, olmo3_template):
        """Generation prompt is added when requested."""
        result = olmo3_template.render(
            messages=[{"role": "user", "content": "Hello"}],
            bos_token="<s>",
            add_generation_prompt=True,
        )
        assert result.endswith("<|im_start|>assistant\n")

    def test_generation_prompt_not_added(self, olmo3_template):
        """Generation prompt is not added when not requested."""
        result = olmo3_template.render(
            messages=[{"role": "user", "content": "Hello"}],
            bos_token="<s>",
            add_generation_prompt=False,
        )
        assert not result.endswith("<|im_start|>assistant\n")
