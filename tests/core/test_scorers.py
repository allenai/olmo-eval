"""Tests for olmo_eval.core.scorers module."""

from olmo_eval.core.scorers import ExactMatchScorer, MultipleChoiceScorer
from olmo_eval.core.types import Instance, LMOutput


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
