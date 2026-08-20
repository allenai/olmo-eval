import asyncio
from typing import Any

import pytest

from olmo_eval.harness.tools import search


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


def test_parse_cutoff() -> None:
    assert search._parse_cutoff("2010-01-13 00:00:00") == (
        2010,
        "01/13/2010",
        "2010-01-13",
    )
    assert search._parse_cutoff("2024-09-25") == (2024, "09/25/2024", "2024-09-25")
    assert search._parse_cutoff(None) is None
    non_string: Any = {"year": 2024}
    assert search._parse_cutoff(non_string) is None
    assert search._parse_cutoff("garbage") is None
    assert search._parse_cutoff("2024-02-30") is None
    assert search._parse_cutoff("24-09-25") is None


def test_search_date_cutoff_is_nested_and_exception_safe() -> None:
    assert search._search_date_cutoff.get() is None

    with search.search_date_cutoff("2010-01-13"):
        assert search._search_date_cutoff.get() == "2010-01-13"

        with (
            pytest.raises(RuntimeError, match="sentinel"),
            search.search_date_cutoff("2024-09-25"),
        ):
            assert search._search_date_cutoff.get() == "2024-09-25"
            raise RuntimeError("sentinel")

        assert search._search_date_cutoff.get() == "2010-01-13"

    assert search._search_date_cutoff.get() is None


@pytest.mark.anyio
async def test_search_date_cutoff_is_isolated_between_concurrent_tasks() -> None:
    first_ready = asyncio.Event()
    second_ready = asyncio.Event()

    async def observe_cutoff(
        cutoff: str, ready: asyncio.Event, peer_ready: asyncio.Event
    ) -> str | None:
        with search.search_date_cutoff(cutoff):
            ready.set()
            await peer_ready.wait()
            await asyncio.sleep(0)
            return search._search_date_cutoff.get()

    observed = await asyncio.gather(
        observe_cutoff("2010-01-13", first_ready, second_ready),
        observe_cutoff("2024-09-25", second_ready, first_ready),
    )

    assert observed == ["2010-01-13", "2024-09-25"]
    assert search._search_date_cutoff.get() is None


@pytest.mark.anyio
async def test_semantic_scholar_applies_server_and_client_cutoffs(monkeypatch) -> None:
    captured_params: dict[str, Any] = {}

    class ClientStub:
        async def get(self, url: str, *, params: dict, headers: dict) -> _ResponseStub:
            captured_params.update(params)
            return _ResponseStub(
                {
                    "data": [
                        {"title": "LaterSameYear", "publicationDate": "2010-12-01", "year": 2010},
                        {"title": "OnDay", "publicationDate": "2010-01-13", "year": 2010},
                        {"title": "YearOnly", "year": 2009},
                        {"title": "Later", "publicationDate": "2011-01-01", "year": 2011},
                        {"title": "Undated", "year": None},
                        {"title": "Missing year"},
                    ]
                }
            )

    async def no_rate_gate() -> None:
        pass

    monkeypatch.setattr(search, "_get_http_client", lambda: ClientStub())
    monkeypatch.setattr(search, "_s2_rate_gate", no_rate_gate)

    with search.search_date_cutoff("2010-01-13 00:00:00"):
        result = await search.semantic_scholar_search(query="test query")

    assert captured_params["publicationDateOrYear"] == ":2010-01-13"
    assert "year" not in captured_params
    assert "publicationDate" in captured_params["fields"]
    assert "LaterSameYear" not in result
    assert "OnDay" in result
    assert "YearOnly" in result
    assert "Later" not in result
    assert "Undated" not in result
    assert "Missing year" not in result


@pytest.mark.anyio
async def test_semantic_scholar_without_cutoff_preserves_undated_papers(monkeypatch) -> None:
    captured_params: dict[str, Any] = {}

    class ClientStub:
        async def get(self, url: str, *, params: dict, headers: dict) -> _ResponseStub:
            captured_params.update(params)
            return _ResponseStub({"data": [{"title": "Undated", "year": None}]})

    async def no_rate_gate() -> None:
        pass

    monkeypatch.setattr(search, "_get_http_client", lambda: ClientStub())
    monkeypatch.setattr(search, "_s2_rate_gate", no_rate_gate)

    result = await search.semantic_scholar_search(query="test query")

    assert "publicationDateOrYear" not in captured_params
    assert "year" not in captured_params
    assert "Undated" in result


@pytest.mark.anyio
async def test_serper_adds_tbs_only_with_cutoff(monkeypatch) -> None:
    request_bodies: list[dict[str, Any]] = []

    class ClientStub:
        async def post(self, url: str, *, json: dict, headers: dict) -> _ResponseStub:
            request_bodies.append(json)
            return _ResponseStub({"organic": []})

    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.setattr(search, "_get_http_client", lambda: ClientStub())

    with search.search_date_cutoff("2010-01-13 00:00:00"):
        await search.serper_web_search(query="dated query")
    await search.serper_web_search(query="undated query")

    assert request_bodies[0]["tbs"] == "cdr:1,cd_min:01/01/1000,cd_max:01/13/2010"
    assert "tbs" not in request_bodies[1]
