"""Tests for olmo_eval.core.scorers module."""

import pytest

from olmo_eval.common.scorers import (
    ExactMatchScorer,
    MultipleChoiceScorer,
    NGramCopyingBPBScorer,
    compute_repeated_ngram_mask,
)
from olmo_eval.common.types import Instance, LMOutput


class TestExactMatchScorer:
    """Tests for ExactMatchScorer."""

    def test_exact_match_correct(self):
        """Test exact match with correct answer."""
        scorer = ExactMatchScorer()
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="Paris")
        output.extracted_answer = "Paris"

        score = scorer.score(instance, output)

        assert score == 1.0

    def test_exact_match_incorrect(self):
        """Test exact match with incorrect answer."""
        scorer = ExactMatchScorer()
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="London")
        output.extracted_answer = "London"

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_exact_match_case_insensitive(self):
        """Test case insensitive matching (default)."""
        scorer = ExactMatchScorer(case_sensitive=False)
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="paris")
        output.extracted_answer = "paris"

        score = scorer.score(instance, output)

        assert score == 1.0

    def test_exact_match_case_sensitive(self):
        """Test case sensitive matching."""
        scorer = ExactMatchScorer(case_sensitive=True)
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="paris")
        output.extracted_answer = "paris"

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_exact_match_strips_whitespace(self):
        """Test whitespace stripping (default)."""
        scorer = ExactMatchScorer(strip_whitespace=True)
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="  Paris  ")
        output.extracted_answer = "  Paris  "

        score = scorer.score(instance, output)

        assert score == 1.0

    def test_exact_match_no_strip_whitespace(self):
        """Test without whitespace stripping."""
        scorer = ExactMatchScorer(strip_whitespace=False)
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="  Paris  ")
        output.extracted_answer = "  Paris  "

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_exact_match_none_gold_answer(self):
        """Test with None gold answer."""
        scorer = ExactMatchScorer()
        instance = Instance(question="Q", gold_answer=None)
        output = LMOutput(text="answer")
        output.extracted_answer = "answer"

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_exact_match_none_extracted_answer(self):
        """Test with None extracted answer."""
        scorer = ExactMatchScorer()
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="text")
        output.extracted_answer = None

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_exact_match_both_none(self):
        """Test with both answers None."""
        scorer = ExactMatchScorer()
        instance = Instance(question="Q", gold_answer=None)
        output = LMOutput(text="text")
        output.extracted_answer = None

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_exact_match_name(self):
        """Test scorer name."""
        scorer = ExactMatchScorer()
        assert scorer.name == "exact_match"

        custom = ExactMatchScorer(name="custom_exact")
        assert custom.name == "custom_exact"

    def test_exact_match_converts_to_string(self):
        """Test that extracted answer is converted to string."""
        scorer = ExactMatchScorer()
        instance = Instance(question="Q", gold_answer="42")
        output = LMOutput(text="42")
        output.extracted_answer = 42  # Integer

        score = scorer.score(instance, output)

        assert score == 1.0


class TestMultipleChoiceScorer:
    """Tests for MultipleChoiceScorer."""

    def test_mc_correct(self):
        """Test multiple choice with correct answer."""
        scorer = MultipleChoiceScorer()
        instance = Instance(question="Q", gold_answer="B")
        output = LMOutput(text="B")
        output.extracted_answer = "B"

        score = scorer.score(instance, output)

        assert score == 1.0

    def test_mc_incorrect(self):
        """Test multiple choice with incorrect answer."""
        scorer = MultipleChoiceScorer()
        instance = Instance(question="Q", gold_answer="B")
        output = LMOutput(text="A")
        output.extracted_answer = "A"

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_mc_case_insensitive(self):
        """Test multiple choice is case insensitive."""
        scorer = MultipleChoiceScorer()
        instance = Instance(question="Q", gold_answer="B")
        output = LMOutput(text="b")
        output.extracted_answer = "b"

        score = scorer.score(instance, output)

        assert score == 1.0

    def test_mc_strips_whitespace(self):
        """Test multiple choice strips whitespace."""
        scorer = MultipleChoiceScorer()
        instance = Instance(question="Q", gold_answer="B")
        output = LMOutput(text=" B ")
        output.extracted_answer = " B "

        score = scorer.score(instance, output)

        assert score == 1.0

    def test_mc_none_gold_answer(self):
        """Test with None gold answer."""
        scorer = MultipleChoiceScorer()
        instance = Instance(question="Q", gold_answer=None)
        output = LMOutput(text="A")
        output.extracted_answer = "A"

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_mc_none_extracted_answer(self):
        """Test with None extracted answer."""
        scorer = MultipleChoiceScorer()
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="text")
        output.extracted_answer = None

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_mc_name(self):
        """Test scorer name."""
        scorer = MultipleChoiceScorer()
        assert scorer.name == "multiple_choice"

        custom = MultipleChoiceScorer(name="custom_mc")
        assert custom.name == "custom_mc"


def _brute_force_ngram_scores(tokens: list[str]) -> list[int]:
    """Reference (independent of compute_repeated_ngram_mask): for each position,
    the length of the longest suffix ending there that also occurs earlier in the
    sequence. Tries the longest candidate length first so it doesn't rely on the
    monotonicity assumption that the optimized implementation is built on.
    """
    n = len(tokens)
    scores = [0] * n
    for i in range(n):
        for length in range(i, 0, -1):
            suffix = tuple(tokens[i - length + 1 : i + 1])
            earlier_windows = (tuple(tokens[j : j + length]) for j in range(i - length + 1))
            if suffix in earlier_windows:
                scores[i] = length
                break
    return scores


class TestComputeRepeatedNgramMask:
    """Tests for compute_repeated_ngram_mask."""

    def test_worked_example(self):
        """`a b c a b` has per-position longest-repeat length 0 0 0 1 2."""
        tokens = ["a", "b", "c", "a", "b"]

        assert _brute_force_ngram_scores(tokens) == [0, 0, 0, 1, 2]

        assert compute_repeated_ngram_mask(tokens, k=1) == [
            False,
            False,
            False,
            True,
            True,
        ]
        assert compute_repeated_ngram_mask(tokens, k=2) == [
            False,
            False,
            False,
            False,
            True,
        ]
        assert compute_repeated_ngram_mask(tokens, k=3) == [
            False,
            False,
            False,
            False,
            False,
        ]

    def test_periodic_pattern_thresholds(self):
        """`x y x y x y` repeats at every period-2 offset, for increasing k."""
        tokens = ["x", "y", "x", "y", "x", "y"]

        assert _brute_force_ngram_scores(tokens) == [0, 0, 1, 2, 3, 4]

        assert compute_repeated_ngram_mask(tokens, k=1) == [
            False,
            False,
            True,
            True,
            True,
            True,
        ]
        assert compute_repeated_ngram_mask(tokens, k=2) == [
            False,
            False,
            False,
            True,
            True,
            True,
        ]
        assert compute_repeated_ngram_mask(tokens, k=4) == [
            False,
            False,
            False,
            False,
            False,
            True,
        ]

    def test_no_repeats(self):
        """A document with no repeated tokens at all is all-False for any k."""
        tokens = ["p", "q", "r", "s"]

        assert _brute_force_ngram_scores(tokens) == [0, 0, 0, 0]
        assert compute_repeated_ngram_mask(tokens, k=1) == [False] * 4
        assert compute_repeated_ngram_mask(tokens, k=2) == [False] * 4

    def test_invalid_k_raises(self):
        """k must be >= 1."""
        with pytest.raises(ValueError):
            compute_repeated_ngram_mask(["a", "b"], k=0)

    @pytest.mark.parametrize(
        "tokens",
        [
            ["a", "b", "c", "a", "b"],
            ["x", "y", "x", "y", "x", "y"],
            ["p", "q", "r", "s"],
            ["m", "m", "m", "m", "m"],
        ],
    )
    def test_mask_matches_thresholded_scores(self, tokens):
        """compute_repeated_ngram_mask(tokens, k) must equal (score >= k) for every k.

        This is the property the implementation relies on: checking a length-k
        match directly is equivalent to thresholding the full longest-repeat score,
        because any earlier match of length k implies a match of every shorter
        length against that same earlier occurrence.
        """
        scores = _brute_force_ngram_scores(tokens)
        for k in range(1, len(tokens) + 1):
            expected = [score >= k for score in scores]
            assert compute_repeated_ngram_mask(tokens, k) == expected


class TestNGramCopyingBPBScorer:
    """Tests for NGramCopyingBPBScorer repeat identity."""

    def test_colliding_token_strings_with_distinct_ids_are_not_repeats(self):
        """Distinct tokens whose decoded strings collide (e.g. multi-byte UTF-8
        fragments that all render as the replacement character) must not count
        as repeats of each other when token IDs are available."""
        scorer = NGramCopyingBPBScorer(k=1)
        output = LMOutput(
            text="",
            logprobs=[
                {"token": "�", "logprob": -1.0, "token_id": 10},
                {"token": "�", "logprob": -1.0, "token_id": 11},
            ],
        )

        assert scorer.masked_totals(output) is None

    def test_repeated_token_ids_are_repeats(self):
        """The same token ID appearing again counts as a repeat, regardless of
        what its decoded string looks like."""
        scorer = NGramCopyingBPBScorer(k=1)
        output = LMOutput(
            text="",
            logprobs=[
                {"token": "�", "logprob": -1.0, "token_id": 10},
                {"token": "�", "logprob": -2.0, "token_id": 10},
            ],
        )

        assert scorer.masked_totals(output) == (-2.0, len("�".encode()))

    def test_falls_back_to_token_strings_without_ids(self):
        """Entries lacking token_id fall back to string identity."""
        scorer = NGramCopyingBPBScorer(k=1)
        output = LMOutput(
            text="",
            logprobs=[
                {"token": "a", "logprob": -1.0},
                {"token": "a", "logprob": -2.0},
            ],
        )

        assert scorer.masked_totals(output) == (-2.0, 1)
