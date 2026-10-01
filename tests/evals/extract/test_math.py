"""Tests for Minerva/Hendrycks MATH answer extraction and equivalence."""

from __future__ import annotations

import pytest

from olmo_eval.evals.extract.math import (
    MathExtractor,
    extract_math_answer,
    get_unnormalized_answer,
    hendrycks_is_equiv,
    is_equiv,
    last_boxed_only_string,
    minerva_is_equiv,
    normalize_final_answer,
    remove_boxed,
    strip_string,
)


class TestExtractMathAnswer:
    """Tests for the combined Minerva / boxed / dollar-sign extractor."""

    def test_minerva_final_answer_phrase(self):
        text = "We get $x=5$.\nFinal Answer: The final answer is $5$. I hope it is correct."
        assert extract_math_answer(text) == ["5"]

    def test_minerva_phrase_without_trailing_sentence(self):
        assert extract_math_answer("Final Answer: The final answer is $7$.") == ["7"]

    def test_boxed_answer(self):
        assert extract_math_answer(r"So the area is $\boxed{\frac{1}{2}}$.") == [r"\frac{1}{2}"]

    def test_boxed_with_space(self):
        assert extract_math_answer(r"\boxed 7$ done") == ["7"]

    def test_minerva_and_boxed_both_returned_in_order(self):
        text = (
            r"Final Answer: The final answer is $\frac{3}{4}$. I hope it is correct."
            r" Also \boxed{0.75}"
        )
        assert extract_math_answer(text) == [r"\frac{3}{4}", "0.75"]

    def test_dollar_fallback_uses_last_math_span(self):
        assert extract_math_answer("The sum is $1,000$ and then $12$ remains.") == ["12"]

    def test_full_text_fallback(self):
        assert extract_math_answer("no math here 42") == ["nomathhere42"]

    def test_unclosed_boxed_falls_back_to_text(self):
        assert extract_math_answer(r"$\boxed{3") == [r"\boxed{3"]

    def test_units_removed_from_final_answer(self):
        text = "Final Answer: The final answer is 10 dollars. I hope it is correct."
        assert extract_math_answer(text) == ["10"]

    def test_text_wrapper_removed(self):
        text = r"Final Answer: The final answer is $\text{(B)}$. I hope it is correct."
        assert extract_math_answer(text) == ["(B)"]

    def test_math_extractor_delegates(self):
        assert MathExtractor.extract_answer(r"\boxed{4}") == ["4"]


class TestBoxedHelpers:
    def test_last_boxed_picks_last(self):
        assert last_boxed_only_string(r"a \boxed{x^{2}} b \boxed{y}") == r"\boxed{y}"

    def test_last_boxed_handles_nested_braces(self):
        assert last_boxed_only_string(r"\boxed{\frac{1}{2}}") == r"\boxed{\frac{1}{2}}"

    def test_last_boxed_fbox(self):
        assert last_boxed_only_string(r"\fbox{9}") == r"\fbox{9}"

    def test_last_boxed_none(self):
        assert last_boxed_only_string("no box") is None

    def test_last_boxed_unclosed(self):
        assert last_boxed_only_string(r"\boxed{3") is None

    def test_remove_boxed(self):
        assert remove_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
        assert remove_boxed(r"\boxed 7") == "7"

    def test_remove_boxed_rejects_other_wrappers(self):
        with pytest.raises(AssertionError):
            remove_boxed(r"\fbox{9}")


class TestNormalization:
    def test_get_unnormalized_answer_missing_phrase(self):
        assert get_unnormalized_answer("nothing") == "[invalidanswer]"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (r"x = \textbf{12}", "12"),
            ("100,000", "100000"),
            (r"\fracab", r"\frac{a}{b}"),
            (r"\sqrt3", r"\sqrt{3}"),
            ("5 inches", "5"),
            (r"$\boxed{6}$", "6"),
        ],
    )
    def test_normalize_final_answer(self, raw: str, expected: str):
        assert normalize_final_answer(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("0.5", r"\frac{1}{2}"),
            (r"k = \dfrac{1}{3}", r"\frac{1}{3}"),
            (r"\sqrt2", r"\sqrt{2}"),
            (".5", r"\frac{1}{2}"),
            ("3/4", r"\frac{3}{4}"),
            (r"\left(1,2\right)", "(1,2)"),
            (r"10\%", "10"),
        ],
    )
    def test_strip_string(self, raw: str, expected: str):
        assert strip_string(raw) == expected


class TestEquivalence:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (r"\frac{1}{2}", "0.5"),
            ("2x+2", "2(x+1)"),
            (r"\dfrac{1}{2}", r"\frac12"),
        ],
    )
    def test_minerva_symbolic_match(self, a: str, b: str):
        assert minerva_is_equiv(a, b)

    def test_minerva_mismatch(self):
        assert not minerva_is_equiv("3", "4")

    def test_minerva_unparseable_returns_false(self):
        assert not minerva_is_equiv(r"\left(1,2\right)", "(1,2)")

    def test_hendrycks_string_match(self):
        assert hendrycks_is_equiv(r"\left(1,2\right)", "(1,2)")
        assert not hendrycks_is_equiv("2x+2", "2(x+1)")

    def test_hendrycks_none_handling(self):
        assert hendrycks_is_equiv(None, None)
        assert not hendrycks_is_equiv("1", None)

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ("2x+2", "2(x+1)", True),
            (r"\left(1,2\right)", "(1,2)", True),
            (r"10\%", "10", True),
            ("3", "4", False),
        ],
    )
    def test_is_equiv_combines_both(self, a: str, b: str, expected: bool):
        assert is_equiv(a, b) is expected

    def test_is_equiv_falls_back_when_sympy_raises(self, monkeypatch: pytest.MonkeyPatch):
        import olmo_eval.evals.extract.math as math_mod

        def _boom(x1: str, x2: str) -> bool:
            raise RuntimeError("boom")

        monkeypatch.setattr(math_mod, "minerva_is_equiv", _boom)
        assert math_mod.is_equiv("(1,2)", r"\left(1,2\right)")
