"""Tests for multi-key Semantic Scholar authentication.

Covers key discovery (zero / one / several / duplicates), that selection
actually rotates, and that the module-level helpers are plain functions rather
than tools accidentally captured by a nearby @registered_tool decorator.
"""

import inspect
from typing import Any

import pytest

from olmo_eval.harness.tools import search
from olmo_eval.harness.tools.tool import Tool

S2_BASE_VAR = "S2_API_KEY"


class _ResponseStub:
    status_code = 200
    text = ""
    headers: dict[str, str] = {}

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._data


@pytest.fixture
def clean_s2_env(monkeypatch):
    """Remove every S2 key variable so each test starts from a known state."""
    import os

    for name in list(os.environ):
        if name == S2_BASE_VAR or (
            name.startswith(f"{S2_BASE_VAR}_") and name[len(S2_BASE_VAR) + 1 :].isdigit()
        ):
            monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _capture_request_headers(monkeypatch) -> list[dict[str, str]]:
    """Stub the HTTP client and rate gate, returning the list of sent headers."""
    sent_headers: list[dict[str, str]] = []

    class ClientStub:
        async def get(self, url: str, *, params: dict, headers: dict) -> _ResponseStub:
            sent_headers.append(dict(headers))
            return _ResponseStub({"data": [{"title": "Paper", "year": 2020}]})

    async def no_rate_gate() -> None:
        pass

    monkeypatch.setattr(search, "_get_http_client", lambda: ClientStub())
    monkeypatch.setattr(search, "_s2_rate_gate", no_rate_gate)
    return sent_headers


def test_no_keys_configured(clean_s2_env) -> None:
    assert search._api_keys_from_env(S2_BASE_VAR) == []
    assert search._s2_rate_interval() == search._S2_MIN_INTERVAL_S


def test_single_key_is_used_unchanged(clean_s2_env) -> None:
    clean_s2_env.setenv(S2_BASE_VAR, "solo-key")

    assert search._api_keys_from_env(S2_BASE_VAR) == ["solo-key"]
    # Every request keeps using the only key, and the pacing is untouched.
    keys = search._api_keys_from_env(S2_BASE_VAR)
    assert {search._select_api_key(keys) for _ in range(10)} == {"solo-key"}
    assert search._s2_rate_interval() == search._S2_MIN_INTERVAL_S


def test_numbered_variables_are_discovered_in_numeric_order(clean_s2_env) -> None:
    clean_s2_env.setenv(S2_BASE_VAR, "first")
    clean_s2_env.setenv(f"{S2_BASE_VAR}_2", "second")
    clean_s2_env.setenv(f"{S2_BASE_VAR}_10", "tenth")
    clean_s2_env.setenv(f"{S2_BASE_VAR}_3", "third")

    # _10 sorts after _3 numerically, not lexicographically.
    assert search._api_keys_from_env(S2_BASE_VAR) == ["first", "second", "third", "tenth"]


def test_comma_separated_variable_is_split(clean_s2_env) -> None:
    clean_s2_env.setenv(S2_BASE_VAR, "one, two ,three")

    assert search._api_keys_from_env(S2_BASE_VAR) == ["one", "two", "three"]


def test_numbered_and_comma_forms_combine(clean_s2_env) -> None:
    clean_s2_env.setenv(S2_BASE_VAR, "a")
    clean_s2_env.setenv(f"{S2_BASE_VAR}_2", "b,c")

    assert search._api_keys_from_env(S2_BASE_VAR) == ["a", "b", "c"]


def test_blank_and_whitespace_values_are_ignored(clean_s2_env) -> None:
    clean_s2_env.setenv(S2_BASE_VAR, "")
    clean_s2_env.setenv(f"{S2_BASE_VAR}_2", "   ")
    clean_s2_env.setenv(f"{S2_BASE_VAR}_3", "real,,")

    assert search._api_keys_from_env(S2_BASE_VAR) == ["real"]


def test_unrelated_suffixes_are_not_treated_as_keys(clean_s2_env) -> None:
    clean_s2_env.setenv(S2_BASE_VAR, "real")
    clean_s2_env.setenv(f"{S2_BASE_VAR}_BACKUP", "not-a-key")
    clean_s2_env.setenv("S2_SEARCH_LIMIT", "20")

    assert search._api_keys_from_env(S2_BASE_VAR) == ["real"]


def test_duplicate_keys_collapse(clean_s2_env) -> None:
    clean_s2_env.setenv(S2_BASE_VAR, "dup")
    clean_s2_env.setenv(f"{S2_BASE_VAR}_2", " dup ")
    clean_s2_env.setenv(f"{S2_BASE_VAR}_3", "dup,other")

    assert search._api_keys_from_env(S2_BASE_VAR) == ["dup", "other"]


def test_duplicate_only_config_keeps_single_key_pacing(clean_s2_env) -> None:
    clean_s2_env.setenv(S2_BASE_VAR, "dup")
    clean_s2_env.setenv(f"{S2_BASE_VAR}_2", "dup")

    # The same key twice is still one key: it must not double the request rate.
    assert search._api_keys_from_env(S2_BASE_VAR) == ["dup"]
    assert search._s2_rate_interval() == search._S2_MIN_INTERVAL_S


def test_rate_interval_scales_with_distinct_key_count(clean_s2_env) -> None:
    clean_s2_env.setenv(S2_BASE_VAR, "a")
    clean_s2_env.setenv(f"{S2_BASE_VAR}_2", "b")

    assert search._s2_rate_interval() == pytest.approx(search._S2_MIN_INTERVAL_S / 2)

    clean_s2_env.setenv(f"{S2_BASE_VAR}_3", "c,d")
    assert search._s2_rate_interval() == pytest.approx(search._S2_MIN_INTERVAL_S / 4)


def test_selection_rotates_evenly_over_all_keys() -> None:
    keys = ["k1", "k2", "k3"]
    picks = [search._select_api_key(keys) for _ in range(30)]

    assert set(picks) == set(keys)
    # Round-robin: perfectly even shares and never the same key twice running.
    assert {key: picks.count(key) for key in keys} == {"k1": 10, "k2": 10, "k3": 10}
    assert all(earlier != later for earlier, later in zip(picks, picks[1:], strict=False))


def test_selection_cursor_is_shared_across_call_sites() -> None:
    # Two callers rotating over the same key set must not both get the same key.
    keys = ["k1", "k2"]
    assert search._select_api_key(keys) != search._select_api_key(keys)


@pytest.mark.anyio
async def test_semantic_scholar_sends_no_auth_header_without_keys(clean_s2_env) -> None:
    sent_headers = _capture_request_headers(clean_s2_env)

    await search.semantic_scholar_search(query="test query")

    assert sent_headers == [{}]


@pytest.mark.anyio
async def test_semantic_scholar_reuses_the_only_key(clean_s2_env) -> None:
    clean_s2_env.setenv(S2_BASE_VAR, "solo-key")
    sent_headers = _capture_request_headers(clean_s2_env)

    for _ in range(3):
        await search.semantic_scholar_search(query="test query")

    assert [headers["x-api-key"] for headers in sent_headers] == ["solo-key"] * 3


@pytest.mark.anyio
async def test_semantic_scholar_spreads_requests_over_keys(clean_s2_env) -> None:
    clean_s2_env.setenv(S2_BASE_VAR, "key-a")
    clean_s2_env.setenv(f"{S2_BASE_VAR}_2", "key-b")
    sent_headers = _capture_request_headers(clean_s2_env)

    for _ in range(4):
        await search.semantic_scholar_search(query="test query")

    used = [headers["x-api-key"] for headers in sent_headers]
    assert set(used) == {"key-a", "key-b"}
    assert used[0] != used[1]
    assert used[0] == used[2] and used[1] == used[3]


def test_key_helpers_are_plain_functions_not_registered_tools() -> None:
    """Guard against the helpers landing inside a @registered_tool sandwich.

    A helper defined between a decorator and its function would come back as a
    Tool (and calling it would yield a coroutine), so assert the module
    attributes are ordinary functions and that only real tools are registered.
    """
    helpers = (
        search._api_keys_from_env,
        search._select_api_key,
        search._s2_rate_interval,
    )
    for helper in helpers:
        assert inspect.isfunction(helper), helper
        assert not inspect.iscoroutinefunction(helper), helper
        assert not isinstance(helper, Tool), helper

    assert isinstance(search._api_keys_from_env("OLMO_EVAL_NO_SUCH_VAR"), list)

    # The decorator still binds to the real tool function, not to a helper: the
    # only Tool objects in the module are the four intended tools.
    assert {name for name, value in vars(search).items() if isinstance(value, Tool)} == {
        "semantic_scholar_search",
        "serper_web_search",
        "serper_fetch_page",
        "crawl4ai_browse",
    }
    assert search.semantic_scholar_search.name == "semantic_scholar_snippet_search"
