from olmo_eval.common.metrics import BPBMetricInstanceAvg
from olmo_eval.evals.suites.registry import get_suite
from olmo_eval.evals.tasks.common import get_task


def test_olmobase_gen_includes_naturalqs() -> None:
    expanded = get_suite("olmobase:gen").expand()

    assert "naturalqs:gen:olmo3base" in expanded


def test_code_v2_uses_versioned_fewshot_tasks() -> None:
    legacy = get_suite("olmobase:code").expand()
    v2 = get_suite("olmobase:code:v2").expand()

    assert "mbpp:olmo3base" in legacy
    assert "deepseek_leetcode:olmo3base" in legacy
    assert "mbpp:olmo3base:v2" in v2
    assert "deepseek_leetcode:olmo3base:v2" in v2
    assert "ds1000:olmo3base:v2" in v2
    assert "bigcodebench:olmo3base:v2" in v2
    assert "mbpp:olmo3base:v2" not in legacy


def test_code_small_v2_uses_versioned_mbpp() -> None:
    legacy = get_suite("olmobase:code_small").expand()
    v2 = get_suite("olmobase:code_small:v2").expand()

    assert "mbpp:olmo3base" in legacy
    assert "mbpp:olmo3base:v2" in v2
    assert "ds1000:olmo3base:v2" in v2


def test_base_easy_code_v2_uses_versioned_fewshot_tasks() -> None:
    legacy = get_suite("olmo3:base_easy:code:bpb").expand()
    v2 = get_suite("olmo3:base_easy:code:bpb:v2").expand()

    assert "mbpp:3shot:bpb" in legacy
    assert "mbpp:3shot:v2:bpb" in v2
    assert any(task.endswith(":3shot:v2:bpb") for task in v2 if task.startswith("mt_mbpp"))


def test_easy_code_bpb_v2_uses_versioned_fewshot_tasks() -> None:
    legacy = get_suite("olmobase:easy:code:bpb").expand()
    v2 = get_suite("olmobase:easy:code:bpb:v2").expand()

    assert "codex_humaneval:bpb:olmo3base" in legacy
    assert "codex_humaneval:olmo3base:v2:bpb" in v2
    assert any(task.endswith(":olmo3base:v2:bpb") for task in v2 if task.startswith("mt_mbpp"))


def test_generation_code_v2_suites_declare_prompt_limits() -> None:
    expected_limits = {
        "bigcodebench": 4096,
        "humaneval": 1536,
        "deepseek_leetcode": 2048,
        "ds1000": 4096,
        "mbpp": 2048,
        "multipl_e": 1024,
    }
    for suite_name in ("olmobase:code_small:v2", "olmobase:code:v2"):
        for task_spec in get_suite(suite_name).expand():
            params = get_task(task_spec).config.sampling_params
            assert params is not None, task_spec
            family = next(name for name in expected_limits if task_spec.startswith(name))
            assert params.truncate_prompt_tokens == expected_limits[family], task_spec


def test_math_v2_uses_multiparagraph_variants() -> None:
    legacy = get_suite("olmobase:math").expand()
    v2 = get_suite("olmobase:math:v2").expand()

    assert len(v2) == len(legacy) == 11
    assert all(task.endswith(":olmo3base") for task in legacy)
    assert list(v2) == [task.replace(":olmo3base", ":olmo3base_multiparagraph") for task in legacy]
    for spec in v2:
        get_task(spec)


def test_v3_clusters_carry_the_fixes() -> None:
    swaps = {
        "olmobase:mcqa_stem": {},
        "olmobase:mcqa_non_stem": {"drop:mc:olmo3base": "drop:mc:olmo3base:v3"},
        "olmobase:gen": {},
        "olmobase:math:v2": {},
        "olmobase:code:v2": {"deepseek_leetcode:olmo3base:v2": "deepseek_leetcode:olmo3base:v3"},
    }
    clusters = []
    for base, swap in swaps.items():
        cluster = get_suite(f"olmobase:{base.split(':')[1]}:v3")
        assert list(cluster.expand()) == [swap.get(t, t) for t in get_suite(base).expand()]
        clusters.append(cluster)

    assert get_suite("olmobase:v3").tasks == tuple(clusters)


def test_easy_v3_bpb_clusters_score_only_bpb() -> None:
    for suite_name in (
        "olmobase:easy:qa:bpb:v3",
        "olmobase:easy:math:bpb:v3",
        "olmobase:easy:code:bpb:v3",
    ):
        for spec in get_suite(suite_name).expand():
            config = get_task(spec).config
            assert isinstance(config.get_primary_metric(), BPBMetricInstanceAvg), spec
            assert all(isinstance(m, BPBMetricInstanceAvg) for m in config.metrics), spec


def test_easy_qa_bpb_v3_swaps_only_the_accuracy_tasks() -> None:
    swaps = {
        "arc_challenge:bpb:olmo3base": "arc_challenge:olmo3base:bpb",
        "arc_easy:bpb:olmo3base": "arc_easy:olmo3base:bpb",
        "csqa:bpb:olmo3base": "csqa:olmo3base:bpb",
        "piqa:bpb:olmo3base": "piqa:olmo3base:bpb",
    }
    v1 = get_suite("olmobase:easy:qa:bpb").expand()

    assert list(get_suite("olmobase:easy:qa:bpb:v3").expand()) == [swaps.get(t, t) for t in v1]
    # The v1 tasks this replaces score accuracy, not BPB.
    for spec in swaps:
        assert not isinstance(get_task(spec).config.metrics[0], BPBMetricInstanceAvg), spec
