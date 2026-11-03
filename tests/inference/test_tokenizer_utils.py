"""Unit tests for tokenizer utilities.

These tests use mock tokenizers and don't require GPU.
"""

from __future__ import annotations

from olmo_eval.inference.tokenizer_utils import (
    default_prefix_ids,
    encode_context_and_continuation,
    encode_prompt,
    get_bos_token_ids,
    get_context_token_ids,
    has_bos_token,
    resolve_prefix_ids,
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


class TemplateBosTokenizer(MockTokenizer):
    """Adds BOS through its post-processor, as Llama 3, Gemma 3/4 and LFM2.5 do: encoding
    with special tokens puts BOS first, yet ``add_bos_token`` is False."""

    add_bos_token = False

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [ord(c) for c in text]
        return [self.bos_token_id, *ids] if add_special_tokens else ids


class FlagBosTokenizer(MockTokenizer):
    """Declares ``add_bos_token = True`` but puts nothing in front when encoding."""

    add_bos_token = True


class BosEosTokenizer(MockTokenizer):
    """Wraps text in BOS ... EOS when encoding with special tokens."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [ord(c) for c in text]
        return [self.bos_token_id, *ids, self.eos_token_id] if add_special_tokens else ids


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


class TestDefaultPrefixIds:
    """Tests for default_prefix_ids()."""

    def test_template_bos_is_measured(self):
        """The flag reads False, but encoding puts BOS first."""
        assert default_prefix_ids(TemplateBosTokenizer(bos_token_id=1)) == [1]

    def test_flag_alone_adds_nothing(self):
        """What the tokenizer does decides, not what its flag says."""
        assert default_prefix_ids(FlagBosTokenizer(bos_token_id=1)) == []

    def test_tokenizer_that_adds_nothing(self):
        assert default_prefix_ids(MockTokenizer(bos_token_id=1)) == []

    def test_trailing_eos_is_not_prefix(self):
        assert default_prefix_ids(BosEosTokenizer(bos_token_id=1, eos_token_id=2)) == [1]


class TestResolvePrefixIds:
    """Tests for resolve_prefix_ids()."""

    def test_none_follows_the_tokenizer(self):
        assert resolve_prefix_ids(TemplateBosTokenizer(bos_token_id=1), None) == [1]
        assert resolve_prefix_ids(MockTokenizer(bos_token_id=1), None) == []

    def test_false_adds_nothing(self):
        assert resolve_prefix_ids(TemplateBosTokenizer(bos_token_id=1), False) == []

    def test_true_keeps_the_tokenizer_prefix(self):
        assert resolve_prefix_ids(TemplateBosTokenizer(bos_token_id=1), True) == [1]

    def test_true_adds_bos_when_the_tokenizer_adds_nothing(self):
        assert resolve_prefix_ids(MockTokenizer(bos_token_id=1), True) == [1]
        assert resolve_prefix_ids(FlagBosTokenizer(bos_token_id=1), True) == [1]

    def test_true_falls_back_to_eos(self):
        assert resolve_prefix_ids(MockTokenizer(bos_token_id=None, eos_token_id=2), True) == [2]


class TestEncodePrompt:
    """Tests for encode_prompt()."""

    def test_prepends_prefix(self):
        tokenizer = MockTokenizer(bos_token_id=1)
        assert encode_prompt(tokenizer, "ab", [1]) == [1, ord("a"), ord("b")]

    def test_empty_prefix(self):
        tokenizer = TemplateBosTokenizer(bos_token_id=1)
        assert encode_prompt(tokenizer, "ab", []) == [ord("a"), ord("b")]

    def test_prefix_already_in_text_is_not_doubled(self):
        """A chat template that writes BOS itself keeps a single BOS."""
        tokenizer = MockTokenizer(bos_token_id=ord("<"))
        assert encode_prompt(tokenizer, "<ab", [ord("<")]) == [ord("<"), ord("a"), ord("b")]


class TestBosForTemplateTokenizers:
    """A tokenizer that adds BOS when encoding text gets BOS before every context."""

    def test_non_empty_context_starts_with_bos(self):
        tokenizer = TemplateBosTokenizer(bos_token_id=1)
        ctx, cont = encode_context_and_continuation(tokenizer, "ab", "cd")
        assert ctx == [1, ord("a"), ord("b")]
        assert cont == [ord("c"), ord("d")]

    def test_continuation_never_gets_bos(self):
        tokenizer = TemplateBosTokenizer(bos_token_id=1)
        _, cont = encode_context_and_continuation(tokenizer, "ab ", "cd")
        assert cont == [ord(" "), ord("c"), ord("d")]

    def test_empty_context_gets_one_bos(self):
        tokenizer = TemplateBosTokenizer(bos_token_id=1)
        ctx, _ = encode_context_and_continuation(tokenizer, "", "cd")
        assert ctx == [1]

    def test_provider_prefix_overrides_the_tokenizer(self):
        """A provider passes its resolved prefix, which wins over the tokenizer default."""
        ctx, _ = encode_context_and_continuation(
            TemplateBosTokenizer(bos_token_id=1), "ab", "cd", prefix_ids=[]
        )
        assert ctx == [ord("a"), ord("b")]
        ctx, _ = encode_context_and_continuation(
            MockTokenizer(bos_token_id=1), "ab", "cd", prefix_ids=[1]
        )
        assert ctx == [1, ord("a"), ord("b")]

    def test_tokenizer_without_bos_is_unchanged(self):
        tokenizer = MockTokenizer(bos_token_id=1)
        ctx, _ = encode_context_and_continuation(tokenizer, "ab", "cd")
        assert ctx == [ord("a"), ord("b")]

    def test_flag_only_tokenizer_gets_no_bos(self):
        tokenizer = FlagBosTokenizer(bos_token_id=1)
        ctx, _ = encode_context_and_continuation(tokenizer, "ab", "cd")
        assert ctx == [ord("a"), ord("b")]

    def test_generation_and_likelihood_start_the_same(self):
        """The context a provider scores starts like the prompt it generates from."""
        for tokenizer in (
            TemplateBosTokenizer(bos_token_id=1),
            MockTokenizer(bos_token_id=1),
            FlagBosTokenizer(bos_token_id=1),
        ):
            for setting in (None, True, False):
                prefix = resolve_prefix_ids(tokenizer, setting)
                ctx, _ = encode_context_and_continuation(tokenizer, "ab", "cd", prefix_ids=prefix)
                assert ctx == encode_prompt(tokenizer, "ab", prefix)


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
