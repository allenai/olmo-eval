"""Tests for the math equivalence scorers."""

from __future__ import annotations

import logging
import sys
import types

import pytest

from olmo_eval.common.scorers import MathVerifyScorer, MinervaMathScorer
from olmo_eval.common.scorers import base as scorers_base
from olmo_eval.common.types import Instance, LMOutput
from olmo_eval.evals.tasks.common import get_task


def _instance(gold: str | None, all_gold: list[str] | None = None) -> Instance:
    metadata = {"all_gold_answers": all_gold} if all_gold is not None else {}
    return Instance(question="Q", gold_answer=gold, metadata=metadata)


def _output(pred: str | None, all_extracted: list[str] | None = None) -> LMOutput:
    metadata = {"all_extracted_answers": all_extracted} if all_extracted is not None else {}
    return LMOutput(text="", extracted_answer=pred, metadata=metadata)


class TestMinervaMathScorer:
    def test_any_extracted_matches_any_gold(self):
        score = MinervaMathScorer().process_score(
            _instance("2", all_gold=["2", r"\frac{4}{2}"]),
            _output("7", all_extracted=["7", r"\frac{1}{2}", "2"]),
        )
        assert score == 1.0

    def test_symbolic_equivalence(self):
        score = MinervaMathScorer().process_score(_instance("2(x+1)"), _output("2x+2"))
        assert score == 1.0

    def test_mismatch(self):
        assert MinervaMathScorer().process_score(_instance("3"), _output("4")) == 0.0

    def test_missing_answers_score_zero(self):
        assert MinervaMathScorer().process_score(_instance(None), _output("4")) == 0.0
        assert MinervaMathScorer().process_score(_instance("4"), _output(None)) == 0.0


class TestMinervaMathTask:
    def test_process_doc_extracts_gold_from_solution(self):
        task = get_task("minerva_math_algebra")
        doc = {
            "problem": "What is 1+1?",
            "solution": r"Adding gives $\boxed{2}$.",
            "level": "Level 1",
            "type": "Algebra",
        }
        instance = task.process_doc(doc, index=3)
        assert instance is not None
        assert instance.gold_answer == "2"
        assert instance.metadata["all_gold_answers"] == ["2"]
        assert instance.metadata["id"] == 3

    def test_extract_answer_records_all_candidates(self):
        task = get_task("minerva_math_algebra")
        output = LMOutput(
            text=(
                r"Final Answer: The final answer is $\frac{3}{4}$. I hope it is correct."
                r" \boxed{0.75}"
            )
        )
        assert task.extract_answer(output) == r"\frac{3}{4}"
        assert output.metadata["all_extracted_answers"] == [r"\frac{3}{4}", "0.75"]

    def test_fixed_fewshot_has_gold_answers(self):
        task = get_task("minerva_math_algebra:olmes")
        fewshot = task.get_fewshot()
        assert len(fewshot) == 4
        assert all(ex.gold_answer for ex in fewshot)


class TestMathVerifyScorer:
    @pytest.fixture(autouse=True)
    def _reset_warning(self):
        scorers_base._warn_math_verify_unavailable.cache_clear()
        yield
        scorers_base._warn_math_verify_unavailable.cache_clear()

    def test_missing_answers_score_zero(self):
        assert MathVerifyScorer().score(_instance(None), _output("1")) == 0.0
        assert MathVerifyScorer().score(_instance("1"), _output(None)) == 0.0

    def test_falls_back_to_is_equiv_and_warns_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        monkeypatch.setitem(sys.modules, "math_verify", None)
        scorer = MathVerifyScorer()
        with caplog.at_level(logging.WARNING, logger=scorers_base.__name__):
            assert scorer.score(_instance("2(x+1)"), _output("2x+2")) == 1.0
            assert scorer.score(_instance("3"), _output("4")) == 0.0
        warnings = [r for r in caplog.records if "math_verify is not installed" in r.message]
        assert len(warnings) == 1

    def test_falls_back_to_is_equiv_when_verify_raises(self, monkeypatch: pytest.MonkeyPatch):
        fake = types.ModuleType("math_verify")

        def _parse(text: str) -> str:
            return text

        def _verify(gold: str, pred: str) -> bool:
            raise RuntimeError("boom")

        fake.parse = _parse  # type: ignore[attr-defined]
        fake.verify = _verify  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "math_verify", fake)
        assert MathVerifyScorer().score(_instance("2(x+1)"), _output("2x+2")) == 1.0

    def test_parses_answers_as_latex_before_verifying(self, monkeypatch: pytest.MonkeyPatch):
        fake = types.ModuleType("math_verify")
        seen: list[str] = []

        def _parse(text: str) -> str:
            seen.append(text)
            return text

        def _verify(gold: str, pred: str) -> bool:
            return False

        fake.parse = _parse  # type: ignore[attr-defined]
        fake.verify = _verify  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "math_verify", fake)
        assert MathVerifyScorer().score(_instance(r"\frac{1}{2}"), _output("$0.5$")) == 0.0
        assert seen == [r"$\frac{1}{2}$", "$0.5$"]

    def test_real_math_verify_equivalence(self):
        pytest.importorskip("math_verify")
        scorer = MathVerifyScorer()
        assert scorer.score(_instance(r"\frac{1}{2}"), _output("0.5")) == 1.0
        assert scorer.score(_instance("x+1"), _output("1+x")) == 1.0
        assert scorer.score(_instance("3"), _output("4")) == 0.0
