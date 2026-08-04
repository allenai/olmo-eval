from olmo_eval.common.types import RequestType
from olmo_eval.evals.suites.registry import get_suite
from olmo_eval.evals.tasks.common import get_task


def test_olmobase_gen_includes_naturalqs() -> None:
    expanded = get_suite("olmobase:gen").expand()

    assert "naturalqs:gen:olmo3base" in expanded


def test_olmobase_chat_stem_suite_uses_chat_requests() -> None:
    expanded = get_suite("olmobase:mcqa_stem:chat").expand()

    assert len(expanded) == 23
    assert "arc_challenge:chat_olmo3base" in expanded
    assert "mmlu_high_school_physics:chat" in expanded
    assert "medqa_en:chat_olmo3base" in expanded
    assert "sciq:chat_olmo3base" in expanded
    assert all(get_task(task).request_type == RequestType.CHAT for task in expanded)
