"""The BrowseComp tool pair: ``search(query)`` and ``visit(url, goal)``."""

from typing import Any

import pytest

from olmo_eval.harness.tools import get_tool, search


class _ResponseStub:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, data: dict[str, Any] | None = None, text: str = "") -> None:
        self._data = data or {}
        self.text = text

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._data


def test_registry_keys_resolve_to_model_facing_names() -> None:
    web_search = get_tool("browsecomp_search")
    assert web_search.name == "search"
    assert web_search.schema.parameters["required"] == ["query"]

    visit = get_tool("browsecomp_visit")
    assert visit.name == "visit"
    assert visit.schema.parameters["required"] == ["url", "goal"]
    assert set(visit.schema.parameters["properties"]) == {"url", "goal"}

    # BrowseComp-Plus keeps the bare ``search`` registry name for its corpus tool.
    assert get_tool("search") is not web_search
    assert get_tool("search").name == "search"


@pytest.mark.anyio
async def test_browsecomp_search_returns_ten_dated_results(monkeypatch) -> None:
    request_bodies: list[dict[str, Any]] = []
    organic = [
        {
            "title": f"Result {i}",
            "snippet": f"Snippet {i}",
            "link": f"https://example.org/{i}",
            **({"date": f"Jan {i}, 2024"} if i % 2 else {}),
        }
        for i in range(12)
    ]

    class ClientStub:
        async def post(self, url: str, *, json: dict, headers: dict) -> _ResponseStub:
            request_bodies.append(json)
            return _ResponseStub({"organic": organic})

    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.setattr(search, "_get_http_client", lambda: ClientStub())

    text = await get_tool("browsecomp_search")(query="ten results")

    assert request_bodies[0]["num"] == 10
    blocks = text.split("\n\n---\n\n")
    assert len(blocks) == 10
    assert blocks[0] == "**Result 0**\nSnippet 0\nURL: https://example.org/0"
    assert (
        blocks[1]
        == "**Result 1**\nDate published: Jan 1, 2024\nSnippet 1\nURL: https://example.org/1"
    )
    assert "Result 10" not in text

    # The older five-result tool is unchanged: five results, no date line.
    legacy = await search.serper_web_search(query="five results")
    assert request_bodies[1]["num"] == 5
    assert len(legacy.split("\n\n---\n\n")) == 5
    assert "Date published" not in legacy


@pytest.mark.anyio
async def test_browsecomp_search_filters_blocked_domains_and_empty_results(monkeypatch) -> None:
    class ClientStub:
        async def post(self, url: str, *, json: dict, headers: dict) -> _ResponseStub:
            return _ResponseStub(
                {
                    "organic": [
                        {"title": "Leak", "snippet": "x", "link": "https://huggingface.co/x"},
                    ]
                }
            )

    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.setattr(search, "_get_http_client", lambda: ClientStub())

    assert await search.browsecomp_search("q") == "No search results found."


@pytest.mark.anyio
async def test_browsecomp_visit_ignores_goal_and_reads_jina(monkeypatch) -> None:
    requested: list[str] = []

    class ClientStub:
        async def get(self, url: str, *, headers: dict, timeout: float) -> _ResponseStub:
            requested.append(url)
            body = (
                "Title: A page\n\nURL Source: https://example.org/p\n\nMarkdown Content:\nBody text"
            )
            return _ResponseStub(text=body)

    monkeypatch.setenv("JINA_API_KEY", "test-key")
    monkeypatch.setattr(search, "_get_http_client", lambda: ClientStub())

    visit = get_tool("browsecomp_visit")
    text = await visit(url="https://example.org/p", goal="the body")
    assert requested == [f"{search._JINA_READER_URL}/https://example.org/p"]
    assert text == "Title: A page\n\nBody text"

    # The goal is optional at execution even though the schema requires it.
    assert await visit(url="https://example.org/p") == text
    blocked = await visit(url="https://hf.co/x", goal="g")
    assert blocked == "Error: Pages on hf.co are not available."
