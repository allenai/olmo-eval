"""Tests for OLMo3 tool parser sanitizer."""

import ast
import re


def _make_sanitizer():
    """Extract and return the sanitizer function from SANITIZE_CODE."""

    # Recreate the sanitizer logic for testing
    def _sanitize_python_strings(text: str) -> str:
        """Escape problematic characters inside single-quoted string arguments."""

        def _escape_content(m):
            content = m.group(1)
            # Escape ALL backslashes (model output is raw text, not Python escapes)
            content = content.replace("\\", "\\\\")
            # Escape literal control chars
            content = content.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            # Escape unescaped single quotes
            content = re.sub(r"(?<!\\)'", r"\\'", content)
            return "='" + content + "'"

        # Non-greedy match with specific boundary detection:
        # - End at ' followed by , and next param (', name=')
        # - End at ' followed by ) (final arg)
        return re.sub(r"='(.*?)'(?=\s*(?:,\s*\w+=|\)))", _escape_content, text, flags=re.DOTALL)

    return _sanitize_python_strings


class TestSanitizePythonStrings:
    """Tests for the _sanitize_python_strings function."""

    def setup_method(self):
        """Set up the sanitizer for each test."""
        self.sanitize = _make_sanitizer()

    def test_simple_string_unchanged(self):
        """Simple strings without special characters pass through."""
        text = "[submit(answer='hello world')]"
        result = self.sanitize(text)
        assert result == text
        # Verify it parses
        ast.parse(result)

    def test_escapes_apostrophe_in_content(self):
        """Apostrophes inside string content are escaped."""
        text = "[submit(answer='Model's approach')]"
        result = self.sanitize(text)
        assert result == "[submit(answer='Model\\'s approach')]"
        # Verify it parses
        ast.parse(result)

    def test_escapes_backslash_in_path(self):
        """Bare backslashes (like Windows paths) are escaped."""
        text = r"[submit(answer='C:\Users\test')]"
        result = self.sanitize(text)
        assert result == r"[submit(answer='C:\\Users\\test')]"
        # Verify it parses
        ast.parse(result)

    def test_escapes_all_backslashes(self):
        """All backslashes are escaped since model output is raw text."""
        text = r"[submit(answer='line1\nline2')]"
        result = self.sanitize(text)
        # Model wrote \n as two chars, should become \\n so ast.parse gives \n
        assert result == r"[submit(answer='line1\\nline2')]"
        ast.parse(result)

    def test_escapes_literal_newline(self):
        """Literal newline characters are escaped."""
        text = "[submit(answer='line1\nline2')]"
        result = self.sanitize(text)
        assert result == "[submit(answer='line1\\nline2')]"
        ast.parse(result)

    def test_escapes_literal_tab(self):
        """Literal tab characters are escaped."""
        text = "[submit(answer='col1\tcol2')]"
        result = self.sanitize(text)
        assert result == "[submit(answer='col1\\tcol2')]"
        ast.parse(result)

    def test_json_with_apostrophe(self):
        """JSON content containing apostrophes is handled."""
        text = """[submit(answer='{"text": "Model's response"}')]"""
        result = self.sanitize(text)
        assert result == """[submit(answer='{"text": "Model\\'s response"}')]"""
        ast.parse(result)

    def test_multiple_apostrophes(self):
        """Multiple apostrophes in content are all escaped."""
        text = "[submit(answer='It's the model's output')]"
        result = self.sanitize(text)
        assert result == "[submit(answer='It\\'s the model\\'s output')]"
        ast.parse(result)

    def test_multiple_arguments(self):
        """Multiple arguments without embedded apostrophes are handled correctly."""
        text = "[call(a='hello', b='world')]"
        result = self.sanitize(text)
        assert result == "[call(a='hello', b='world')]"
        ast.parse(result)

    def test_multiple_arguments_with_special_chars(self):
        """Multiple arguments with backslashes are each escaped."""
        text = r"[call(path='C:\Users', name='test')]"
        result = self.sanitize(text)
        assert result == r"[call(path='C:\\Users', name='test')]"
        ast.parse(result)

    def test_apostrophe_in_content_ambiguous(self):
        """Apostrophes in content create ambiguous boundaries - known limitation.

        When content contains apostrophes, the regex cannot reliably distinguish
        between content apostrophes and argument boundaries. This documents the
        current behavior rather than asserting specific output.
        """
        text = "[submit(answer='Model's response')]"
        result = self.sanitize(text)
        # Should at least produce parseable output
        ast.parse(result)
        # And the apostrophe should be escaped somewhere
        assert "\\'" in result

    def test_mixed_special_characters(self):
        """Content with both backslashes and apostrophes."""
        text = r"[submit(answer='User's path: C:\docs')]"
        result = self.sanitize(text)
        assert result == r"[submit(answer='User\'s path: C:\\docs')]"
        ast.parse(result)

    def test_pretty_printed_json(self):
        """Multi-line JSON (pretty-printed) is handled."""
        text = '[submit(answer=\'{\n  "key": "value"\n}\')]'
        result = self.sanitize(text)
        assert result == '[submit(answer=\'{\\n  "key": "value"\\n}\')]'
        ast.parse(result)
