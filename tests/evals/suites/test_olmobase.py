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
