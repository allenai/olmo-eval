from olmo_eval.evals.suites.registry import get_suite
from olmo_eval.evals.tasks.common import get_task


def test_safety_base_v2_swaps_judged_tasks_to_multiparagraph() -> None:
    legacy = get_suite("safety_base").expand()
    v2 = get_suite("safety_base:v2").expand()

    mcq = {"bbq:base", "wmdp:base"}
    assert mcq <= set(legacy)
    assert all(task.endswith(":base") for task in legacy)
    assert list(v2) == [
        task if task in mcq else task.replace(":base", ":base_multiparagraph") for task in legacy
    ]
    for spec in v2:
        get_task(spec)
