from olmo_eval.runners.asynq.monitoring import resolve_provider_init_timeout


def test_provider_init_timeout_defaults_to_fifteen_minutes():
    assert resolve_provider_init_timeout(None) == 900.0


def test_provider_init_timeout_exceeds_server_startup_timeout():
    assert resolve_provider_init_timeout(7200) == 7260.0


def test_provider_init_timeout_keeps_larger_default_for_short_startups():
    assert resolve_provider_init_timeout("120") == 900.0


def test_provider_init_timeout_ignores_invalid_values():
    assert resolve_provider_init_timeout("not-a-duration") == 900.0
