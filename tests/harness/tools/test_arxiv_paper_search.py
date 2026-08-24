"""Tests for the arXiv-filtered paper search tool.

The tool is held to DeepScholar-Bench's admission test -- arXiv provenance,
published strictly before the query's cutoff -- so these cover the client-side
filter, the widened page that filter needs, the cutoff on both sides of the
request, and the rendering the export pass later parses back. Nothing here
touches the live API.
"""

import types
from typing import Any

import pytest

from olmo_eval.harness.tools import search


class _ResponseStub:
    text = ""

    def __init__(
        self,
        data: dict[str, Any],
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._data = data
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._data


def _paper(
    title: str,
    *,
    arxiv: str | None = None,
    published: str | None = None,
    year: int | None = 2020,
) -> dict[str, Any]:
    paper: dict[str, Any] = {
        "title": title,
        "abstract": f"Abstract of {title}.",
        "year": year,
        "url": "",
        "authors": [{"name": "A. Author"}],
        "externalIds": {"DOI": "10.1/x"},
    }
    if arxiv is not None:
        paper["externalIds"]["ArXiv"] = arxiv
    if published is not None:
        paper["publicationDate"] = published
    return paper


def _stub_search(monkeypatch, papers: list[dict[str, Any]]) -> dict[str, Any]:
    """Answer any S2 search with ``papers``; returns the captured query params."""
    captured: dict[str, Any] = {}

    class ClientStub:
        async def get(self, url: str, *, params: dict, headers: dict) -> _ResponseStub:
            captured.update(params)
            return _ResponseStub({"data": papers})

    async def no_rate_gate() -> None:
        pass

    monkeypatch.setattr(search, "_get_http_client", lambda: ClientStub())
    monkeypatch.setattr(search, "_s2_rate_gate", no_rate_gate)
    return captured


@pytest.mark.anyio
async def test_requests_external_ids_and_widens_the_page(monkeypatch) -> None:
    monkeypatch.setenv("ARXIV_SEARCH_LIMIT", "10")
    captured = _stub_search(monkeypatch, [])

    await search.arxiv_paper_search(query="retrieval augmented generation")

    assert "externalIds" in captured["fields"]
    assert captured["limit"] == 10 * search._ARXIV_OVERFETCH_MULTIPLIER
    # arXiv provenance is decided on external IDs; the venue is not stable.
    assert "venue" not in captured


@pytest.mark.anyio
async def test_page_widening_is_capped_at_the_s2_maximum(monkeypatch) -> None:
    monkeypatch.setenv("ARXIV_SEARCH_LIMIT", "50")
    captured = _stub_search(monkeypatch, [])

    await search.arxiv_paper_search(query="q")

    assert captured["limit"] == search._S2_MAX_SEARCH_LIMIT


@pytest.mark.anyio
async def test_arxiv_hits_are_rendered_with_id_and_abs_url(monkeypatch) -> None:
    _stub_search(monkeypatch, [_paper("Preprint", arxiv="2401.01234v2")])

    result = await search.arxiv_paper_search(query="q")

    assert "**Preprint**" in result
    assert "Authors: A. Author" in result
    assert "Year: 2020" in result
    assert "Abstract: Abstract of Preprint." in result
    # The version suffix is stripped, and the abs URL is what the benchmark's
    # citation parser credits.
    assert "arXiv: 2401.01234" in result
    assert "URL: https://arxiv.org/abs/2401.01234" in result


@pytest.mark.anyio
async def test_hits_without_an_arxiv_id_are_marked_context_only_and_capped(monkeypatch) -> None:
    monkeypatch.setenv("ARXIV_SEARCH_LIMIT", "5")
    _stub_search(
        monkeypatch,
        [_paper(f"Journal {index}") for index in range(5)]
        + [_paper("Preprint", arxiv="2401.01234")],
    )

    result = await search.arxiv_paper_search(query="q")

    assert result.count("[context only: no arXiv ID, not citable]") == (
        search._ARXIV_CONTEXT_ONLY_LIMIT
    )
    # Nothing without an arXiv ID is offered an arxiv.org URL to cite.
    assert result.count("URL: https://arxiv.org/abs/") == 1


@pytest.mark.anyio
async def test_relevance_order_is_preserved_and_truncated_after_filtering(monkeypatch) -> None:
    monkeypatch.setenv("ARXIV_SEARCH_LIMIT", "2")
    _stub_search(
        monkeypatch,
        [
            _paper("Journal", published="2020-01-01"),
            _paper("First", arxiv="2401.00001"),
            _paper("Second", arxiv="2401.00002"),
            _paper("Third", arxiv="2401.00003"),
        ],
    )

    result = await search.arxiv_paper_search(query="q")

    assert result.index("**First**") < result.index("**Second**")
    assert "**Third**" not in result


@pytest.mark.anyio
async def test_cutoff_is_pushed_one_day_short_server_side(monkeypatch) -> None:
    captured = _stub_search(monkeypatch, [])

    with search.search_date_cutoff("2024-06-01 13:06:19+00:00"):
        await search.arxiv_paper_search(query="q")

    # The exporter's test is strictly-before and S2's range is inclusive.
    assert captured["publicationDateOrYear"] == ":2024-05-31"


@pytest.mark.anyio
async def test_cutoff_is_enforced_client_side(monkeypatch) -> None:
    _stub_search(
        monkeypatch,
        [
            _paper("Before", arxiv="2405.00001", published="2024-05-31"),
            _paper("OnCutoff", arxiv="2406.00001", published="2024-06-01"),
            _paper("After", arxiv="2407.00001", published="2024-07-01"),
        ],
    )

    with search.search_date_cutoff("2024-06-01"):
        result = await search.arxiv_paper_search(query="q")

    assert "**Before**" in result
    assert "**OnCutoff**" not in result
    assert "**After**" not in result


@pytest.mark.anyio
async def test_undated_hits_fall_back_to_the_arxiv_id_month(monkeypatch) -> None:
    _stub_search(
        monkeypatch,
        [
            _paper("EarlierMonth", arxiv="2405.00001"),
            _paper("CutoffMonth", arxiv="2406.00001"),
            _paper("Legacy", arxiv="cs/0501001"),
            _paper("Undatable", arxiv="not-an-id"),
        ],
    )

    with search.search_date_cutoff("2024-06-01"):
        result = await search.arxiv_paper_search(query="q")

    assert "**EarlierMonth**" in result
    assert "**Legacy**" in result
    # Same month as the cutoff cannot be shown to precede it, and an ID that
    # encodes no date is evidence the exporter would not have either.
    assert "**CutoffMonth**" not in result
    assert "**Undatable**" not in result


@pytest.mark.anyio
async def test_empty_result_set_reports_no_papers(monkeypatch) -> None:
    _stub_search(monkeypatch, [_paper("Journal")])

    with search.search_date_cutoff("2024-06-01"):
        result = await search.arxiv_paper_search(query="q")

    assert result == "No arXiv papers found for query."


@pytest.mark.anyio
async def test_empty_query_is_rejected_without_a_request(monkeypatch) -> None:
    def fail() -> None:
        raise AssertionError("no request should be made for an empty query")

    monkeypatch.setattr(search, "_get_http_client", lambda: fail())

    assert await search.arxiv_paper_search(query="   ") == "Error: Empty search query."


@pytest.mark.anyio
async def test_rate_limited_requests_are_retried_after_the_servers_delay(monkeypatch) -> None:
    slept: list[float] = []
    attempts: list[dict[str, Any]] = []
    responses = [
        _ResponseStub({}, status_code=429, headers={"Retry-After": "30"}),
        _ResponseStub({"data": [_paper("Preprint", arxiv="2401.01234")]}),
    ]

    async def record_sleep(seconds: float) -> None:
        slept.append(seconds)

    class ClientStub:
        async def get(self, url: str, *, params: dict, headers: dict) -> _ResponseStub:
            attempts.append(params)
            return responses.pop(0)

    async def no_rate_gate() -> None:
        pass

    monkeypatch.setattr(search, "_get_http_client", lambda: ClientStub())
    monkeypatch.setattr(search, "_s2_rate_gate", no_rate_gate)
    monkeypatch.setattr(search, "asyncio", types.SimpleNamespace(sleep=record_sleep))

    result = await search.arxiv_paper_search(query="q")

    assert len(attempts) == 2
    assert "arXiv: 2401.01234" in result
    # Retry-After wins over the exponential schedule, clamped to the module's
    # ceiling: a server asking for longer than _MAX_BACKOFF_S is under-waited.
    assert slept == [min(30.0, search._MAX_BACKOFF_S)]
