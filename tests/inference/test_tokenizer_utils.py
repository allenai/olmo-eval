"""Unit tests for tokenizer utilities.

These tests use mock tokenizers and don't require GPU.
"""

from __future__ import annotations

from olmo_eval.inference.tokenizer_utils import (
    encode_context_and_continuation,
    get_bos_token_ids,
    get_context_token_ids,
    has_bos_token,
)


class MockTokenizer:
    """Mock tokenizer for testing BOS handling behavior."""

    def __init__(
        self,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
    ):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        # Simple character-level tokenization for predictable testing
        self._vocab = {chr(i): i for i in range(256)}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Simple character-level encoding."""
        return [ord(c) for c in text]


class TestGetBosTokenIds:
    """Tests for get_bos_token_ids()."""

    def test_returns_bos_when_available(self):
        tokenizer = MockTokenizer(bos_token_id=1)
        result = get_bos_token_ids(tokenizer)
        assert result == [1]

    def test_falls_back_to_eos_when_bos_is_none(self):
        tokenizer = MockTokenizer(bos_token_id=None, eos_token_id=2)
        result = get_bos_token_ids(tokenizer, fallback_to_eos=True)
        assert result == [2]

    def test_no_fallback_when_disabled(self):
        tokenizer = MockTokenizer(bos_token_id=None, eos_token_id=2)
        result = get_bos_token_ids(tokenizer, fallback_to_eos=False)
        assert result == []

    def test_returns_empty_when_both_none(self):
        tokenizer = MockTokenizer(bos_token_id=None, eos_token_id=None)
        result = get_bos_token_ids(tokenizer)
        assert result == []


class TestHasBosToken:
    """Tests for has_bos_token()."""

    def test_returns_true_when_bos_defined(self):
        tokenizer = MockTokenizer(bos_token_id=1)
        assert has_bos_token(tokenizer) is True

    def test_returns_false_when_bos_none(self):
        tokenizer = MockTokenizer(bos_token_id=None)
        assert has_bos_token(tokenizer) is False


class TestGetContextTokenIds:
    """Tests for get_context_token_ids()."""

    def test_tokenizes_non_empty_context(self):
        tokenizer = MockTokenizer()
        result = get_context_token_ids(tokenizer, "abc")
        assert result == [ord("a"), ord("b"), ord("c")]

    def test_returns_bos_for_empty_context(self):
        tokenizer = MockTokenizer(bos_token_id=1)
        result = get_context_token_ids(tokenizer, "")
        assert result == [1]

    def test_returns_eos_fallback_for_empty_context(self):
        tokenizer = MockTokenizer(bos_token_id=None, eos_token_id=2)
        result = get_context_token_ids(tokenizer, "")
        assert result == [2]

    def test_no_bos_for_empty_when_disabled(self):
        tokenizer = MockTokenizer(bos_token_id=1)
        result = get_context_token_ids(tokenizer, "", use_bos_for_empty=False)
        assert result == []


class TestEncodeContextAndContinuation:
    """Tests for encode_context_and_continuation()."""

    def test_empty_context_uses_bos(self):
        tokenizer = MockTokenizer(bos_token_id=1)
        ctx, cont = encode_context_and_continuation(tokenizer, "", "hello")
        assert ctx == [1]
        assert cont == [ord("h"), ord("e"), ord("l"), ord("l"), ord("o")]

    def test_empty_context_falls_back_to_eos(self):
        tokenizer = MockTokenizer(bos_token_id=None, eos_token_id=2)
        ctx, cont = encode_context_and_continuation(tokenizer, "", "hi")
        assert ctx == [2]
        assert cont == [ord("h"), ord("i")]

    def test_empty_context_no_bos_when_disabled(self):
        tokenizer = MockTokenizer(bos_token_id=1)
        ctx, cont = encode_context_and_continuation(tokenizer, "", "hi", use_bos_for_empty=False)
        assert ctx == []
        assert cont == [ord("h"), ord("i")]

    def test_non_empty_context_basic(self):
        tokenizer = MockTokenizer()
        ctx, cont = encode_context_and_continuation(tokenizer, "ab", "cd")
        assert ctx == [ord("a"), ord("b")]
        assert cont == [ord("c"), ord("d")]

    def test_trailing_space_moved_to_continuation(self):
        """Trailing spaces from context should be moved to continuation."""
        tokenizer = MockTokenizer()
        ctx, cont = encode_context_and_continuation(tokenizer, "ab ", "cd")
        # Context should be "ab" (no trailing space)
        assert ctx == [ord("a"), ord("b")]
        # Continuation should be " cd" (space prepended)
        assert cont == [ord(" "), ord("c"), ord("d")]

    def test_multiple_trailing_spaces(self):
        tokenizer = MockTokenizer()
        ctx, cont = encode_context_and_continuation(tokenizer, "ab  ", "cd")
        assert ctx == [ord("a"), ord("b")]
        assert cont == [ord(" "), ord(" "), ord("c"), ord("d")]

    def test_context_only_spaces_becomes_empty(self):
        """If context is only spaces, they all move to continuation.

        Note: This matches lm_eval behavior - trailing space handling runs first
        (because context == "" is false), then the resulting empty context is
        tokenized normally without BOS injection. This is intentional: a context
        of "  " is semantically different from an empty context "".
        """
        tokenizer = MockTokenizer(bos_token_id=1)
        ctx, cont = encode_context_and_continuation(tokenizer, "  ", "cd")
        # Context becomes empty [] after spaces are stripped
        # (no BOS because the original context was not empty)
        assert ctx == []
        # Spaces get prepended to continuation
        assert cont == [ord(" "), ord(" "), ord("c"), ord("d")]


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_both_empty(self):
        """Empty context and continuation."""
        tokenizer = MockTokenizer(bos_token_id=1)
        ctx, cont = encode_context_and_continuation(tokenizer, "", "")
        assert ctx == [1]
        assert cont == []

    def test_no_special_tokens_available(self):
        """Tokenizer has no BOS or EOS."""
        tokenizer = MockTokenizer(bos_token_id=None, eos_token_id=None)
        ctx, cont = encode_context_and_continuation(tokenizer, "", "hi")
        assert ctx == []
        assert cont == [ord("h"), ord("i")]

    def test_unicode_handling(self):
        """Unicode characters in context and continuation."""
        tokenizer = MockTokenizer()
        ctx, cont = encode_context_and_continuation(tokenizer, "café", "résumé")
        # Our mock uses ord() which handles unicode
        assert ctx == [ord("c"), ord("a"), ord("f"), ord("é")]
        assert cont == [ord("r"), ord("é"), ord("s"), ord("u"), ord("m"), ord("é")]
