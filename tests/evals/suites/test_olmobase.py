from olmo_eval.evals.suites.registry import get_suite


def test_olmobase_gen_includes_naturalqs() -> None:
    expanded = get_suite("olmobase:gen").expand()

    assert "naturalqs:gen:olmo3base" in expanded
