"""Tests for the arXiv-filtered paper search tool.

The tool is held to DeepScholar-Bench's admission test -- arXiv provenance,
published strictly before the query's cutoff -- so these cover the client-side
filter, the widened page that filter needs, the cutoff on both sides of the
request, and the rendering the export pass later parses back. Nothing here
touches the live API.
"""

import types
from typing import Any

import httpx
import pytest

from olmo_eval.harness.tools import search


class _ResponseStub:
    text = ""

    def __init__(
        self,
        data: Any,
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
    title: str | None,
    *,
    arxiv: str | None = None,
    published: str | None = None,
    year: int | None = 2020,
    abstract: str | None = "",
) -> dict[str, Any]:
    paper: dict[str, Any] = {
        "title": title,
        "abstract": f"Abstract of {title}." if abstract == "" else abstract,
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


BAD_JSON = "<body that is not json>"
BAD_SHAPE = "<a list where an object belongs>"
BAD_DATA = "<a data field that is not a list>"
NO_RESULTS = "<S2's real no-results body: total and offset, no data>"


def _stub_pages(
    monkeypatch,
    *pages,
    error_after: int | None = None,
    headers_out: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Answer successive S2 searches with ``pages``; returns the params of each.

    A page may be a row list, or one of the BAD_* sentinels standing for a
    response body S2 should never send but sometimes does. ``error_after`` makes
    every request past that many fail at the transport, which is how a page
    failing mid-pagination is exercised without a live API.
    """
    calls: list[dict[str, Any]] = []
    remaining = list(pages)

    class _BadJson:
        status_code = 200
        text = ""
        headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            pass

        def json(self) -> Any:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    class ClientStub:
        async def get(self, url: str, *, params: dict, headers: dict) -> Any:
            calls.append(dict(params))
            if headers_out is not None:
                headers_out.append(dict(headers))
            if error_after is not None and len(calls) > error_after:
                raise httpx.ConnectError("stubbed transport failure")
            page = remaining.pop(0) if remaining else []
            if page is BAD_JSON:
                return _BadJson()
            if page is BAD_SHAPE:
                return _ResponseStub(["not", "an", "object"])
            if page is BAD_DATA:
                return _ResponseStub({"total": 1, "data": "not a list"})
            if page is NO_RESULTS:
                return _ResponseStub({"total": 0, "offset": 0})
            return _ResponseStub({"data": page})

    async def no_rate_gate() -> None:
        pass

    monkeypatch.setattr(search, "_get_http_client", lambda: ClientStub())
    monkeypatch.setattr(search, "_s2_rate_gate", no_rate_gate)
    return calls


def _stub_search(monkeypatch, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Answer one S2 search with ``papers``; returns the params of each request."""
    return _stub_pages(monkeypatch, papers)


def _filler(count: int) -> list[dict[str, Any]]:
    """Rows with no arXiv ID -- about four fifths of what S2 really returns."""
    return [_paper(f"Journal {index}") for index in range(count)]


def _no_backoff_sleep(monkeypatch) -> None:
    """Skip the retry waits so a failure test does not sleep through backoff."""

    async def record_sleep(seconds: float) -> None:
        pass

    monkeypatch.setattr(search, "asyncio", types.SimpleNamespace(sleep=record_sleep))


@pytest.mark.anyio
async def test_requests_external_ids_and_a_full_first_page(monkeypatch) -> None:
    monkeypatch.setenv("ARXIV_SEARCH_LIMIT", "10")
    calls = _stub_search(monkeypatch, [])

    await search.arxiv_paper_search(query="retrieval augmented generation")

    assert "externalIds" in calls[0]["fields"]
    assert calls[0]["limit"] == search._ARXIV_PAGE_SIZE
    assert calls[0]["offset"] == 0
    # arXiv provenance is decided on external IDs; the venue is not stable.
    assert "venue" not in calls[0]


@pytest.mark.anyio
async def test_pagination_continues_until_the_wanted_count_is_reached(monkeypatch) -> None:
    # Only about a fifth of S2 rows carry an arXiv ID, so a sparse first page is
    # the normal case rather than the exception.
    monkeypatch.setenv("ARXIV_SEARCH_LIMIT", "3")
    calls = _stub_pages(
        monkeypatch,
        _filler(99) + [_paper("First", arxiv="2401.00001")],
        _filler(98) + [_paper("Second", arxiv="2401.00002"), _paper("Third", arxiv="2401.00003")],
    )

    result = await search.arxiv_paper_search(query="q")

    assert [call["offset"] for call in calls] == [0, search._ARXIV_PAGE_SIZE]
    assert "**First**" in result
    assert "**Third**" in result
    # Relevance order survives paging: page one's hit precedes page two's.
    assert result.index("**First**") < result.index("**Second**") < result.index("**Third**")


@pytest.mark.anyio
async def test_pagination_stops_at_the_page_budget_without_erroring(monkeypatch) -> None:
    monkeypatch.setenv("ARXIV_SEARCH_LIMIT", "10")
    calls = _stub_pages(
        monkeypatch,
        *[_filler(99) + [_paper(f"Sparse {n}", arxiv=f"2401.0000{n}")] for n in range(5)],
    )

    result = await search.arxiv_paper_search(query="q")

    # Three pages, then stop: returning fewer than wanted is a result, not an error.
    assert [call["offset"] for call in calls] == [0, 100, 200]
    assert result.count("URL: https://arxiv.org/abs/") == 3
    assert not result.startswith("Error")


@pytest.mark.anyio
async def test_pagination_stops_when_s2_runs_out_of_hits(monkeypatch) -> None:
    monkeypatch.setenv("ARXIV_SEARCH_LIMIT", "10")
    calls = _stub_pages(monkeypatch, _filler(4) + [_paper("Only", arxiv="2401.00001")])

    result = await search.arxiv_paper_search(query="q")

    # A short page means there is no next one; asking for it would waste the gate.
    assert len(calls) == 1
    assert "**Only**" in result


@pytest.mark.anyio
async def test_a_page_failing_late_keeps_the_earlier_pages_results(monkeypatch) -> None:
    monkeypatch.setenv("ARXIV_SEARCH_LIMIT", "10")
    _no_backoff_sleep(monkeypatch)
    _stub_pages(
        monkeypatch,
        _filler(99) + [_paper("Survivor", arxiv="2401.00001")],
        error_after=1,
    )

    result = await search.arxiv_paper_search(query="q")

    assert "**Survivor**" in result
    assert not result.startswith("Error")


@pytest.mark.anyio
async def test_a_first_page_failure_is_reported(monkeypatch) -> None:
    _no_backoff_sleep(monkeypatch)
    _stub_pages(monkeypatch, error_after=0)

    result = await search.arxiv_paper_search(query="q")

    assert result.startswith("Error searching arXiv papers")


def test_the_description_tells_the_model_what_the_backend_is() -> None:
    # 8 of the 12 operator-bearing queries in a 200-query sample returned zero
    # rows: the model was treating this as a web search engine.
    description = search.arxiv_paper_search.description

    assert "not a web search engine" in description
    assert "site:" in description
    assert "boolean" in description.casefold()


@pytest.mark.anyio
async def test_web_search_operators_are_passed_through_unmodified(monkeypatch) -> None:
    # The description is the fix. Rewriting the query in code would hide what
    # the model actually asked and make the empty result unattributable.
    calls = _stub_pages(monkeypatch, [])
    query = 'site:arxiv.org "retrieval augmented generation" -survey'

    await search.arxiv_paper_search(query=query)

    assert calls[0]["query"] == query


@pytest.mark.anyio
async def test_arxiv_hits_are_rendered_with_id_and_abs_url(monkeypatch) -> None:
    _stub_search(monkeypatch, [_paper("Preprint", arxiv="2401.01234v2")])

    result = await search.arxiv_paper_search(query="q")

    assert "**Preprint**" in result
    assert "Authors: A. Author" in result
    assert "Year: 2020" in result
    # S2 reported no date, so only the month the ID encodes can be claimed.
    assert "Published: 2024-01 (month precision)" in result
    assert "Abstract: Abstract of Preprint." in result
    # The version suffix is stripped, and the abs URL is what the benchmark's
    # citation parser credits.
    assert "arXiv: 2401.01234" in result
    assert "URL: https://arxiv.org/abs/2401.01234" in result


@pytest.mark.anyio
async def test_a_dated_hit_is_published_to_the_day(monkeypatch) -> None:
    _stub_search(monkeypatch, [_paper("Dated", arxiv="2401.01234", published="2024-01-15")])

    result = await search.arxiv_paper_search(query="q")

    # A day S2 knows must not be downgraded to a month: the benchmark contract
    # rejects a month-precise source dated inside the cutoff's own month.
    assert "Published: 2024-01-15" in result
    assert "month precision" not in result


@pytest.mark.anyio
async def test_a_hit_nothing_can_date_is_rejected(monkeypatch) -> None:
    # lit-agents' _classify_source calls this "undated" and drops it; the
    # exporter has no other evidence either.
    _stub_search(monkeypatch, [_paper("Undatable", arxiv="not-an-id")])

    result = await search.arxiv_paper_search(query="q")

    assert result == "No arXiv papers found for query."


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("field", "value"),
    [("title", None), ("title", "   "), ("abstract", None), ("abstract", "  ")],
)
async def test_hits_missing_a_title_or_abstract_are_rejected(monkeypatch, field, value) -> None:
    # paper.csv rows with an empty title or snippet fail the contract, so a hit
    # that would produce one is never offered as citable.
    kwargs = {"arxiv": "2401.01234", "published": "2024-01-15"}
    if field == "abstract":
        kwargs["abstract"] = value
        paper = _paper("Titled", **kwargs)
    else:
        paper = _paper(value, **kwargs)
    _stub_search(monkeypatch, [paper])

    result = await search.arxiv_paper_search(query="q")

    assert result == "No arXiv papers found for query."


@pytest.mark.anyio
async def test_a_malformed_publication_date_falls_back_to_the_id_month(monkeypatch) -> None:
    # Mirrors RetrievalPolicy.admits, whose date parser returns None for a date
    # it cannot read just as it does for one that is absent, and falls back to
    # the arXiv ID's month in both cases. Retrieval is the lenient layer; the
    # strict cutoff test that decides what is scored runs at export.
    _stub_search(monkeypatch, [_paper("Garbled", arxiv="2405.00001", published="not a date")])

    with search.search_date_cutoff("2024-06-01"):
        result = await search.arxiv_paper_search(query="q")

    assert "**Garbled**" in result
    assert "Published: 2024-05 (month precision)" in result


@pytest.mark.anyio
async def test_a_malformed_date_whose_id_month_is_too_late_is_still_rejected(monkeypatch) -> None:
    _stub_search(monkeypatch, [_paper("Garbled", arxiv="2406.00001", published="not a date")])

    with search.search_date_cutoff("2024-06-01"):
        result = await search.arxiv_paper_search(query="q")

    assert result == "No arXiv papers found for query."


@pytest.mark.anyio
async def test_context_only_hits_need_a_title_and_an_abstract(monkeypatch) -> None:
    _stub_search(monkeypatch, [_paper("Untitled", abstract=None), _paper(None)])

    result = await search.arxiv_paper_search(query="q")

    assert result == "No arXiv papers found for query."


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
    calls = _stub_search(monkeypatch, [])

    with search.search_date_cutoff("2024-06-01 13:06:19+00:00"):
        await search.arxiv_paper_search(query="q")

    # The exporter's test is strictly-before and S2's range is inclusive.
    assert calls[0]["publicationDateOrYear"] == ":2024-05-31"


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


@pytest.mark.anyio
async def test_identity_fields_precede_the_abstract(monkeypatch) -> None:
    # The abstract is the one field a paper controls the text of, so every
    # identifying field is rendered before it can start.
    _stub_search(monkeypatch, [_paper("Ordered", arxiv="2401.00001", published="2024-01-05")])

    result = await search.arxiv_paper_search(query="q")

    assert result.index("arXiv: 2401.00001") < result.index("Abstract:")
    assert result.index("URL: ") < result.index("Abstract:")
    assert result.index("Published: ") < result.index("Abstract:")


@pytest.mark.anyio
async def test_a_paper_repeated_across_pages_counts_once(monkeypatch) -> None:
    # S2 can return the same paper on more than one page; counting it twice
    # would fill the caller's budget with one paper.
    monkeypatch.setenv("ARXIV_SEARCH_LIMIT", "3")
    duplicate = _paper("Repeated", arxiv="2401.00001")
    calls = _stub_pages(
        monkeypatch,
        _filler(99) + [duplicate],
        _filler(98) + [_paper("Repeated again", arxiv="2401.00001v2"), duplicate],
        _filler(99) + [_paper("Distinct", arxiv="2401.00002")],
    )

    result = await search.arxiv_paper_search(query="q")

    assert len(calls) == 3
    assert result.count("arXiv: 2401.00001") == 1
    assert "**Repeated again**" not in result
    assert "**Distinct**" in result


@pytest.mark.anyio
async def test_each_page_rotates_through_the_configured_keys(monkeypatch) -> None:
    # The rate gate spaces requests for the key set as a whole, so a multi-page
    # search holding one key would hammer it at the interval meant for three.
    monkeypatch.setenv("S2_API_KEY", "key-a,key-b,key-c")
    monkeypatch.setenv("ARXIV_SEARCH_LIMIT", "99")
    headers: list[dict[str, str]] = []
    _stub_pages(
        monkeypatch,
        *[_filler(99) + [_paper(f"One {n}", arxiv=f"2401.0000{n}")] for n in range(3)],
        headers_out=headers,
    )

    await search.arxiv_paper_search(query="q")

    assert len(headers) == 3
    assert len({header["x-api-key"] for header in headers}) == 3


@pytest.mark.anyio
async def test_a_late_page_of_unreadable_json_keeps_earlier_results(monkeypatch) -> None:
    monkeypatch.setenv("ARXIV_SEARCH_LIMIT", "10")
    _stub_pages(monkeypatch, _filler(99) + [_paper("Survivor", arxiv="2401.00001")], BAD_JSON)

    result = await search.arxiv_paper_search(query="q")

    assert "**Survivor**" in result
    assert not result.startswith("Error")


@pytest.mark.anyio
async def test_a_late_page_of_unexpected_shape_keeps_earlier_results(monkeypatch) -> None:
    monkeypatch.setenv("ARXIV_SEARCH_LIMIT", "10")
    _stub_pages(monkeypatch, _filler(99) + [_paper("Survivor", arxiv="2401.00001")], BAD_SHAPE)

    result = await search.arxiv_paper_search(query="q")

    assert "**Survivor**" in result
    assert not result.startswith("Error")


@pytest.mark.anyio
@pytest.mark.parametrize("payload", [BAD_JSON, BAD_SHAPE])
async def test_a_first_page_of_garbage_is_reported(monkeypatch, payload) -> None:
    _stub_pages(monkeypatch, payload)

    result = await search.arxiv_paper_search(query="q")

    assert result.startswith("Error searching arXiv papers")


@pytest.mark.anyio
async def test_an_empty_data_array_is_not_an_error(monkeypatch) -> None:
    # An empty page is a legitimate answer and must stay distinguishable from a
    # response this code could not read.
    _stub_pages(monkeypatch, [])

    result = await search.arxiv_paper_search(query="q")

    assert result == "No arXiv papers found for query."


@pytest.mark.anyio
async def test_s2s_no_results_body_is_not_an_error(monkeypatch) -> None:
    # Verified against the live endpoint: a query matching nothing comes back as
    # {"total": 0, "offset": 0} with no data array. Calling that malformed would
    # tell the model its search broke when it merely found nothing.
    calls = _stub_pages(monkeypatch, NO_RESULTS)

    result = await search.arxiv_paper_search(query="zzzqqq nonexistent xyzzy")

    assert result == "No arXiv papers found for query."
    # And it is a complete page, so there is no next one to ask for.
    assert len(calls) == 1


@pytest.mark.anyio
async def test_a_data_field_that_is_not_a_list_is_still_an_error(monkeypatch) -> None:
    _stub_pages(monkeypatch, BAD_DATA)

    result = await search.arxiv_paper_search(query="q")

    assert result.startswith("Error searching arXiv papers")
