"""BrowseComp-Plus tool routing and HTTP failure handling."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from olmo_eval.harness.tools.browsecomp_plus import call_retriever
from olmo_eval.harness.tools.registry import get_tool


@pytest.mark.anyio
async def test_tools_preserve_original_names(monkeypatch):
    call = AsyncMock(return_value="[]")
    monkeypatch.setattr("olmo_eval.harness.tools.browsecomp_plus.call_retriever", call)
    tool = get_tool("search")
    assert tool.name == "search"
    assert await tool(query="cities") == "[]"
    call.assert_awaited_once_with("search", {"query": "cities"})
    document = get_tool("get_document")
    assert document.name == "get_document"
    await document(docid="001")
    call.assert_awaited_with("get_document", {"docid": "001"})


@pytest.fixture
def http_response(monkeypatch):
    original_client = httpx.AsyncClient

    def install(response):
        def client(**kwargs):
            return original_client(
                transport=httpx.MockTransport(lambda request: response), **kwargs
            )

        monkeypatch.setattr(httpx, "AsyncClient", client)

    return install


@pytest.mark.anyio
@pytest.mark.parametrize("status", [404, 500, 503])
async def test_http_errors_are_raised(http_response, status):
    http_response(httpx.Response(status, json={"detail": "unavailable"}))
    with pytest.raises(httpx.HTTPStatusError):
        await call_retriever("search", {"query": "Paris"})


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool_name", "value"),
    [
        ("search", {"error": "unavailable"}),
        ("search", None),
        ("search", [{"docid": "1"}]),
        ("get_document", {"docid": "1"}),
    ],
)
async def test_malformed_results_are_raised(http_response, tool_name, value):
    http_response(httpx.Response(200, content=json.dumps(value)))
    with pytest.raises(ValueError):
        await call_retriever(tool_name, {})


@pytest.mark.anyio
async def test_invalid_json_is_raised(http_response):
    http_response(httpx.Response(200, text="not JSON"))
    with pytest.raises(json.JSONDecodeError):
        await call_retriever("search", {"query": "Paris"})


@pytest.mark.anyio
async def test_connection_failures_are_raised(monkeypatch):
    original_client = httpx.AsyncClient

    def fail(request):
        raise httpx.ConnectError("unavailable", request=request)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(fail), **kwargs),
    )
    with pytest.raises(httpx.ConnectError):
        await call_retriever("search", {"query": "Paris"})
