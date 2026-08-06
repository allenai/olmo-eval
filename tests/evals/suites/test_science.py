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


def _leaves(name):
    from olmo_eval.evals.suites.registry import get_suite

    out = []

    def rec(suite):
        for task in suite.tasks:
            rec(task) if hasattr(task, "tasks") else out.append(task)

    rec(get_suite(name))
    return set(out)


def test_science_nojudge_base_norm_differs_only_in_mc_normalization():
    base, norm = _leaves("science:nojudge:base"), _leaves("science:nojudge:base_norm")
    assert len(base) == len(norm)
    assert {t.replace(":mc", ":mc_per_char") for t in base - norm} == norm - base


def test_science_expert_bpb_covers_the_gpqa_and_lab_bench_leaves():
    """Every leaf is a `:bpb` variant, and they are exactly the leaves that base/base_norm swap."""
    bpb = _leaves("science:expert:bpb")
    swapped = _leaves("science:nojudge") - _leaves("science:nojudge:base")
    assert all(t.endswith(":bpb") for t in bpb)
    assert bpb == {f"{t}:bpb" for t in swapped}


def test_science_expert_suites_are_the_swapped_leaves_under_three_scorings():
    """The narrow suites hold exactly the 15 tasks base/base_norm swap, one scoring each."""
    swapped = _leaves("science:nojudge") - _leaves("science:nojudge:base")
    for suite, suffix in (("science:expert:base", "mc"),
                          ("science:expert:base_norm", "mc_per_char"),
                          ("science:expert:bpb", "bpb")):
        assert _leaves(suite) == {f"{t}:{suffix}" for t in swapped}, suite


def test_science_grounding_is_likelihood_only_and_base_safe():
    """Both leaves score by likelihood, so an annealed base checkpoint can be measured."""
    from olmo_eval.common.types import RequestType
    from olmo_eval.evals.tasks.common.registry import get_task

    leaves = _leaves("science:grounding")
    assert leaves == {"scifact_claim_evidence_within", "scifact_claim_evidence_cross"}
    for name in leaves:
        assert get_task(name).request_type == RequestType.LOGLIKELIHOOD


def test_contrastive_metrics_are_at_chance_when_contexts_are_uninformative():
    from olmo_eval.common.types import Instance
    from olmo_eval.evals.tasks.scifact_claim_evidence import (
        ContrastiveEffectSize,
        ContrastiveWinRate,
    )

    class _Out:
        def __init__(self, lp):
            self.logprobs = [{"logprob": lp}]

    class _Resp:
        def __init__(self, inst, lp):
            self.instance, self.outputs = inst, [_Out(lp)]

    # identical logprobs on both sides of every pair -> no separation
    resps = []
    for i in range(50):
        for kind in ("gold", "neg"):
            inst = Instance(question="q", gold_answer="a",
                            metadata={"pair_id": str(i), "kind": kind, "cond": "within"})
            resps.append(_Resp(inst, -10.0))
    assert ContrastiveEffectSize().compute(resps) == 0.0
    assert ContrastiveWinRate().compute(resps) == 0.0  # no pair has gold > neg

    # gold strictly better on every pair -> win rate 1.0
    resps = []
    for i in range(50):
        for kind, lp in (("gold", -9.0), ("neg", -10.0)):
            inst = Instance(question="q", gold_answer="a",
                            metadata={"pair_id": str(i), "kind": kind, "cond": "within"})
            resps.append(_Resp(inst, lp))
    assert ContrastiveWinRate().compute(resps) == 1.0


def test_scifact_pair_ids_are_unique_per_side():
    """A claim can carry several SUPPORT rationales in one abstract; if they share a pair id the
    metric's gold/neg dict overwrites and silently discards pairs."""
    import collections

    from olmo_eval.evals.tasks.scifact_claim_evidence import _build_pairs

    recs = _build_pairs()
    for cond in ("within", "cross"):
        counts = collections.Counter(r["pair_id"] for r in recs if r["cond"] == cond)
        assert counts and all(v == 2 for v in counts.values()), cond
