"""Tests for olmo_eval.core.llm_judge_scorer module."""

from olmo_eval.common.scorers import (
    RubricJudgeScorer,
    SimpleQAJudgeScorer,
)
from olmo_eval.common.types import Instance, LMOutput


class TestSimpleQAJudgeScorer:
    """Tests for SimpleQAJudgeScorer."""

    def test_correct_grade_a(self):
        """Test parsing grade A (CORRECT)."""

        def mock_judge(prompt: str) -> str:
            return "A"

        scorer = SimpleQAJudgeScorer(judge_fn=mock_judge)
        instance = Instance(question="What is 2+2?", gold_answer="4")
        output = LMOutput(text="4")
        output.extracted_answer = "4"

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_incorrect_grade_b(self):
        """Test parsing grade B (INCORRECT)."""

        def mock_judge(prompt: str) -> str:
            return "B"

        scorer = SimpleQAJudgeScorer(judge_fn=mock_judge)
        instance = Instance(question="What is 2+2?", gold_answer="4")
        output = LMOutput(text="5")
        output.extracted_answer = "5"

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_not_attempted_grade_c(self):
        """Test parsing grade C (NOT_ATTEMPTED)."""

        def mock_judge(prompt: str) -> str:
            return "C"

        scorer = SimpleQAJudgeScorer(judge_fn=mock_judge)
        instance = Instance(question="Complex question", gold_answer="Answer")
        output = LMOutput(text="I don't know")

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_parse_correct_word(self):
        """Test parsing 'CORRECT' text."""

        def mock_judge(prompt: str) -> str:
            return "The answer is CORRECT."

        scorer = SimpleQAJudgeScorer(judge_fn=mock_judge)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="A")

        score = scorer.score(instance, output)
        assert score == 1.0

    def test_parse_incorrect_word(self):
        """Test parsing 'INCORRECT' text."""

        def mock_judge(prompt: str) -> str:
            return "The answer is INCORRECT."

        scorer = SimpleQAJudgeScorer(judge_fn=mock_judge)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="B")

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_parse_not_attempted_text(self):
        """Test parsing 'NOT_ATTEMPTED' text."""

        def mock_judge(prompt: str) -> str:
            return "NOT_ATTEMPTED"

        scorer = SimpleQAJudgeScorer(judge_fn=mock_judge)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="?")

        score = scorer.score(instance, output)
        assert score == 0.0

    def test_get_grade(self):
        """Test get_grade method."""
        scorer = SimpleQAJudgeScorer()
        assert scorer.get_grade("A") == "CORRECT"
        assert scorer.get_grade("B") == "INCORRECT"
        assert scorer.get_grade("C") == "NOT_ATTEMPTED"
        assert scorer.get_grade("CORRECT") == "CORRECT"
        assert scorer.get_grade("garbage") == "INCORRECT"

    def test_format_prompt(self):
        """Test prompt formatting."""
        scorer = SimpleQAJudgeScorer()
        instance = Instance(question="What is 2+2?", gold_answer="4")
        output = LMOutput(text="The answer is 4")
        output.extracted_answer = "4"

        prompt = scorer.format_judge_prompt(instance, output)
        assert "What is 2+2?" in prompt
        assert "4" in prompt
        assert "CORRECT" in prompt
        assert "INCORRECT" in prompt
        assert "NOT_ATTEMPTED" in prompt

    def test_scorer_name(self):
        """Test scorer name."""
        scorer = SimpleQAJudgeScorer()
        assert scorer.name == "simpleqa_judge"


class TestRubricJudgeScorer:
    """Tests for RubricJudgeScorer."""

    def test_extract_score(self):
        """Test extracting score from response."""

        def mock_judge(prompt: str) -> str:
            return "The response is good. Score: 8"

        scorer = RubricJudgeScorer(judge_fn=mock_judge, max_score=10.0)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="Response")

        score = scorer.score(instance, output)
        assert score == 0.8

    def test_extract_decimal_score(self):
        """Test extracting decimal score."""

        def mock_judge(prompt: str) -> str:
            return "Score: 7.5"

        scorer = RubricJudgeScorer(judge_fn=mock_judge, max_score=10.0)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="Response")

        score = scorer.score(instance, output)
        assert score == 0.75

    def test_custom_score_pattern(self):
        """Test custom score pattern."""

        def mock_judge(prompt: str) -> str:
            return "Rating: 4/5"

        scorer = RubricJudgeScorer(
            judge_fn=mock_judge,
            score_pattern=r"Rating:\s*(\d+)/5",
            max_score=5.0,
        )
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="Response")

        score = scorer.score(instance, output)
        assert score == 0.8

    def test_no_score_found(self):
        """Test when no score is found."""

        def mock_judge(prompt: str) -> str:
            return "Good response with no score"

        scorer = RubricJudgeScorer(judge_fn=mock_judge, default_score=5.0, max_score=10.0)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="Response")

        score = scorer.score(instance, output)
        assert score == 0.5

    def test_get_raw_score(self):
        """Test getting raw score."""
        scorer = RubricJudgeScorer(max_score=10.0)
        raw = scorer.get_raw_score("The Score: 7 is given")
        assert raw == 7.0

    def test_custom_rubric(self):
        """Test with custom rubric."""
        custom_rubric = "Rate 0-5 based on accuracy"

        def mock_judge(prompt: str) -> str:
            assert custom_rubric in prompt
            return "Score: 4"

        scorer = RubricJudgeScorer(judge_fn=mock_judge, rubric=custom_rubric, max_score=5.0)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="Response")

        score = scorer.score(instance, output)
        assert score == 0.8

    def test_format_prompt(self):
        """Test prompt formatting."""
        scorer = RubricJudgeScorer(rubric="Custom rubric here")
        instance = Instance(question="Test question", gold_answer="Expected answer")
        output = LMOutput(text="Model response")

        prompt = scorer.format_judge_prompt(instance, output)
        assert "Test question" in prompt
        assert "Expected answer" in prompt
        assert "Model response" in prompt
        assert "Custom rubric here" in prompt

    def test_default_rubric_used(self):
        """Test default rubric when none provided."""
        scorer = RubricJudgeScorer(max_score=10.0)
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="Response")

        prompt = scorer.format_judge_prompt(instance, output)
        assert "10" in prompt
        assert "0" in prompt

    def test_scorer_name(self):
        """Test scorer name."""
        scorer = RubricJudgeScorer()
        assert scorer.name == "rubric_judge"


class TestLLMJudgeScorerIntegration:
    """Integration tests for LLM judge scorers."""

    def test_simpleqa_full_flow(self):
        """Test complete SimpleQA evaluation flow."""
        responses = []

        def tracking_judge(prompt: str) -> str:
            responses.append(prompt)
            # Check if "AI Assistant's Answer: 4" appears (correct answer)
            if "AI Assistant's Answer: 4" in prompt:
                return "A"
            return "B"

        scorer = SimpleQAJudgeScorer(judge_fn=tracking_judge)

        # Correct answer
        instance1 = Instance(question="2+2=?", gold_answer="4")
        output1 = LMOutput(text="4")
        output1.extracted_answer = "4"
        score1 = scorer.score(instance1, output1)

        # Incorrect answer
        instance2 = Instance(question="2+2=?", gold_answer="4")
        output2 = LMOutput(text="5")
        output2.extracted_answer = "5"
        score2 = scorer.score(instance2, output2)

        assert score1 == 1.0
        assert score2 == 0.0
        assert len(responses) == 2

    def test_rubric_different_scales(self):
        """Test rubric scorer with different score scales."""
        scales = [(5.0, 4), (10.0, 8), (100.0, 80)]

        for max_score, raw_score in scales:

            def make_judge(rs):
                return lambda p: f"Score: {rs}"

            scorer = RubricJudgeScorer(judge_fn=make_judge(raw_score), max_score=max_score)
            instance = Instance(question="Q", gold_answer="A")
            output = LMOutput(text="R")

            score = scorer.score(instance, output)
            assert score == 0.8
