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


def test_science_nojudge_base_swaps_only_generative_mc_tasks():
    """science:nojudge:base differs from science:nojudge only by scoring GPQA and LAB-Bench
    with their `:mc` variants; every other leaf is untouched."""
    from olmo_eval.evals.suites.registry import get_suite

    def leaves(name):
        out = []

        def rec(suite):
            for task in suite.tasks:
                rec(task) if hasattr(task, "tasks") else out.append(task)

        rec(get_suite(name))
        return set(out)

    gen, base = leaves("science:nojudge"), leaves("science:nojudge:base")
    assert len(gen) == len(base)
    only_gen, only_base = gen - base, base - gen
    assert only_base == {f"{t}:mc" for t in only_gen}
    assert all(t.startswith(("gpqa_", "lab_bench_")) for t in only_gen)
