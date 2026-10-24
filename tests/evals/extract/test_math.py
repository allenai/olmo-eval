"""Tests for the Minerva/Hendrycks math answer normalizer and equivalence."""

import pytest

from olmo_eval.common.scorers import MinervaMathScorer
from olmo_eval.common.types import Instance, LMOutput
from olmo_eval.evals.extract.math import (
    MathExtractor,
    is_equiv,
    normalize_final_answer,
    strip_size_delimiters,
)


def _score(solution: str, output: str) -> float:
    gold = MathExtractor.extract_answer(solution)
    instance = Instance(question="q", gold_answer=gold[0], metadata={"all_gold_answers": gold})
    lm_output = LMOutput(
        text=output, metadata={"all_extracted_answers": MathExtractor.extract_answer(output)}
    )
    return MinervaMathScorer().process_score(instance, lm_output)


class TestSizeDelimiters:
    @pytest.mark.parametrize(
        ("sized", "plain"),
        [
            (r"\left(3,\frac{\pi}{2}\right)", r"(3,\frac{\pi}{2})"),
            (
                r"\left[\frac{\pi^2}{8},\frac{5\pi^2}{4}\right]",
                r"[\frac{\pi^2}{8},\frac{5\pi^2}{4}]",
            ),
            (r"\left(\frac{3}{5},\frac{8}{3}\right]", r"(\frac{3}{5},\frac{8}{3}]"),
            (r"\left\{1,2\right\}", r"\{1,2\}"),
            (r"\bigl(1,2\bigr)", r"(1,2)"),
            (r"\Bigl[0,1\Bigr)", r"[0,1)"),
            (r"\left|x\right|", r"|x|"),
        ],
    )
    def test_sized_and_plain_normalize_the_same(self, sized: str, plain: str) -> None:
        assert normalize_final_answer(sized) == normalize_final_answer(plain)
        assert is_equiv(normalize_final_answer(sized), normalize_final_answer(plain))

    def test_left_is_not_corrupted_into_le(self) -> None:
        # Before the fix the unit "ft" was removed from inside \left, leaving \le.
        assert r"\le" not in normalize_final_answer(r"\left(3,\frac{\pi}{2}\right)")

    def test_null_delimiter_is_dropped(self) -> None:
        assert strip_size_delimiters(r"\left.\frac{x}{2}\right|") == r"\frac{x}{2}|"

    @pytest.mark.parametrize(
        "command", [r"\leftarrow", r"\Leftrightarrow", r"\rightarrow", r"\bigcup"]
    )
    def test_prefix_sharing_commands_are_kept(self, command: str) -> None:
        assert strip_size_delimiters(f"x{command}y") == f"x{command}y"

    @pytest.mark.parametrize("command", [r"\big", r"\Bigg", r"\biggr", r"\Bigl", r"\middle"])
    def test_sizing_commands_are_dropped(self, command: str) -> None:
        assert strip_size_delimiters(f"x{command}|y") == "x|y"

    def test_scorer_matches_left_paren_gold_with_plain_prediction(self) -> None:
        solution = r"So the answer is $\boxed{\left( 3, \frac{\pi}{2} \right)}.$"
        assert _score(solution, r"Thus \boxed{(3, \dfrac{\pi}{2})}") == 1.0
        assert _score(solution, r"Thus \boxed{\left(3, \dfrac{\pi}{2}\right)}") == 1.0
        assert _score(solution, r"Thus \boxed{(3, \pi)}") == 0.0

    def test_scorer_matches_left_bracket_gold_with_plain_prediction(self) -> None:
        solution = r"The range is $\boxed{\left[ \frac{\pi^2}{8}, \frac{5 \pi^2}{4} \right]}.$"
        assert _score(solution, r"\boxed{[\dfrac{\pi^2}{8}, \dfrac{5\pi^2}{4}]}") == 1.0
        assert _score(solution, r"\boxed{(\dfrac{\pi^2}{8}, \dfrac{5\pi^2}{4})}") == 0.0


class TestUnits:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("5 ft", "5"),
            (r"5\text{ ft}", "5"),
            (r"5\text{ft}", "5"),
            ("12 feet", "12"),
            ("100 square feet", "100"),
            ("7 inches", "7"),
            ("3 km", "3"),
            ("10 dollars", "10"),
            ("30 minutes", "30"),
            (r"45^\circ", "45"),
            ("6 units", "6"),
        ],
    )
    def test_units_are_removed(self, answer: str, expected: str) -> None:
        assert normalize_final_answer(answer) == expected

    @pytest.mark.parametrize("command", [r"\infty", r"\leftarrow", r"\Leftrightarrow"])
    def test_units_do_not_match_inside_commands(self, command: str) -> None:
        # "ft" used to be removed from inside these command names.
        assert normalize_final_answer(f"x{command}y") == f"x{command}y"

    def test_infinity_interval_keeps_infty(self) -> None:
        assert normalize_final_answer(r"(-\infty, 3]") == r"(-\infty,3]"


class TestUnchangedBehavior:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("x = 5", "5"),
            ("100,000", "100000"),
            (r"\frac12", r"\frac{1}{2}"),
            (r"\sqrt2", r"\sqrt{2}"),
            (r"\text{(A)}", "(A)"),
            (r"\textbf{7}", "7"),
            (r"$\frac{3}{4}$", r"\frac{3}{4}"),
            (r"1,\!000", "1000"),
            (r"3\pi", r"3\pi"),
            (r"\dfrac{1}{2}", r"\dfrac{1}{2}"),
        ],
    )
    def test_normalize(self, answer: str, expected: str) -> None:
        assert normalize_final_answer(answer) == expected

    @pytest.mark.parametrize(
        ("pred", "gold", "equal"),
        [
            (r"\frac{1}{2}", "0.5", True),
            (r"\dfrac{1}{2}", r"\frac{1}{2}", True),
            ("2", "3", False),
            (r"2\sqrt{2}", r"\sqrt{8}", True),
        ],
    )
    def test_is_equiv(self, pred: str, gold: str, equal: bool) -> None:
        assert is_equiv(normalize_final_answer(pred), normalize_final_answer(gold)) is equal

    def test_extract_prefers_minerva_then_boxed(self) -> None:
        text = r"Work. Final Answer: The final answer is $7$. I hope it is correct. \boxed{8}"
        assert MathExtractor.extract_answer(text) == ["7", "8"]

    def test_extract_falls_back_to_dollars(self) -> None:
        assert MathExtractor.extract_answer(r"so we get $x=12$ and $14$ done") == ["14"]
