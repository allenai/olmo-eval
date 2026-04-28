"""Tests for science suite composition."""

from collections import Counter

from olmo_eval.evals.suites import get_suite, suite_exists


def test_science_suites_are_registered():
    expected = (
        "science:core",
        "science:biology",
        "science:medicine",
        "science:physical",
        "science:research",
        "science:math",
        "science:nojudge",
        "science:judge",
        "science:all",
    )
    for name in expected:
        assert suite_exists(name), f"Expected suite {name!r} to be registered"


def test_science_all_has_no_duplicate_task_specs():
    expanded = get_suite("science:all").expand()
    counts = Counter(expanded)
    duplicates = {task: count for task, count in counts.items() if count > 1}
    assert duplicates == {}


def test_science_biology_owns_biology_gpqa_slice_only():
    expanded = get_suite("science:biology").expand()
    assert "gpqa_diamond_biology" in expanded
    assert "gpqa_main_biology" in expanded
    assert "gpqa_extended_biology" in expanded
    assert "gpqa_diamond" not in expanded
    assert "gpqa_main" not in expanded
    assert "gpqa_extended" not in expanded


def test_science_medicine_uses_single_medqa_family_entry():
    expanded = get_suite("science:medicine").expand()
    assert "medqa_en" in expanded
    assert "medqa" not in expanded


def test_science_all_keeps_physical_science_subject_specific():
    expanded = get_suite("science:all").expand()
    assert "gpqa_diamond_chemistry" in expanded
    assert "gpqa_main_physics" in expanded
    assert "gpqa_diamond" not in expanded
    assert "gpqa_main" not in expanded
    assert "gpqa_extended" not in expanded


def test_science_research_contains_literature_tasks():
    expanded = get_suite("science:research").expand()
    assert "qasper_yesno" in expanded
    assert "sciriff_yesno" in expanded
    assert "astabench_scholarqa" in expanded


def test_science_nojudge_excludes_judge_task():
    expanded = get_suite("science:nojudge").expand()
    assert "qasper_yesno" in expanded
    assert "sciriff_yesno" in expanded
    assert "astabench_scholarqa" not in expanded


def test_science_judge_contains_only_judge_task():
    expanded = get_suite("science:judge").expand()
    assert expanded == ("astabench_scholarqa",)
