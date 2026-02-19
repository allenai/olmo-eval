"""Tests for code extraction utilities."""

from olmo_eval.evals.extract.code import (
    _strip_thinking_tags,
    extract_code,
    extract_function_body,
    indent_code,
)

from .fixtures import REASONING_MODEL_EXPECTED_CODE, REASONING_MODEL_OUTPUT


class TestStripThinkingTags:
    """Tests for _strip_thinking_tags helper."""

    def test_strips_simple_think_block(self):
        text = "<think>Some reasoning</think>\nActual content"
        result = _strip_thinking_tags(text)
        assert result == "Actual content"

    def test_strips_multiline_think_block(self):
        text = "<think>\nLine 1\nLine 2\n</think>\nCode here"
        result = _strip_thinking_tags(text)
        assert result == "Code here"

    def test_case_insensitive(self):
        text = "<THINK>reasoning</THINK>\ncode"
        result = _strip_thinking_tags(text)
        assert result == "code"

    def test_no_think_tags(self):
        text = "Just regular code"
        result = _strip_thinking_tags(text)
        assert result == "Just regular code"


class TestExtractCode:
    """Tests for extract_code function."""

    def test_extracts_python_code_block(self):
        text = "Here is the code:\n```python\ndef foo():\n    pass\n```"
        result = extract_code(text)
        assert result == "def foo():\n    pass\n"

    def test_extracts_generic_code_block(self):
        text = "```\nsome code\n```"
        result = extract_code(text)
        assert result == "some code\n"

    def test_falls_back_to_full_text(self):
        text = "x = 1\ny = 2"
        result = extract_code(text)
        assert result == "x = 1\ny = 2"

    def test_strips_think_tags_before_extraction(self):
        """Test extraction from reasoning model output with <think> tags."""
        text = """<think>
Let me think about this problem...
The answer should be 42.
</think>

x = 42
print(x)"""
        result = extract_code(text)
        assert result == "x = 42\nprint(x)"

    def test_real_world_reasoning_model_output(self):
        """Test with realistic reasoning model output from HumanEval."""
        result = extract_code(REASONING_MODEL_OUTPUT)
        assert result == REASONING_MODEL_EXPECTED_CODE

    def test_think_tags_with_code_block_inside(self):
        """Test that code blocks inside think tags are not extracted."""
        text = """<think>
Here's some thinking with code:
```python
wrong_code = True
```
</think>

```python
correct_code = True
```"""
        result = extract_code(text)
        assert result == "correct_code = True\n"


class TestExtractFunctionBody:
    """Tests for extract_function_body function."""

    def test_extracts_body_after_signature(self):
        text = "def foo(x):\n    return x + 1"
        result = extract_function_body(text, signature="def foo(x)")
        assert result == "return x + 1"


class TestIndentCode:
    """Tests for indent_code function."""

    def test_adds_indentation_to_unindented_code(self):
        code = "x = 1\ny = 2"
        result = indent_code(code)
        assert result == "    x = 1\n    y = 2"

    def test_preserves_already_indented_code(self):
        code = "    x = 1\n    y = 2"
        result = indent_code(code)
        assert result == "    x = 1\n    y = 2"

    def test_handles_empty_string(self):
        assert indent_code("") == ""
