"""Tests for the crawl4ai-backed browse_webpage tool."""

from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _install_fake_crawl4ai(monkeypatch: pytest.MonkeyPatch, result: object):
    class FakeCrawler:
        instances = []

        def __init__(self):
            self.arun_urls = []
            FakeCrawler.instances.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def arun(self, url: str):
            self.arun_urls.append(url)
            if isinstance(result, Exception):
                raise result
            return result

    module = ModuleType("crawl4ai")
    module.AsyncWebCrawler = FakeCrawler
    monkeypatch.setitem(sys.modules, "crawl4ai", module)
    return FakeCrawler


@pytest.mark.anyio
async def test_crawl4ai_browse_returns_markdown(monkeypatch):
    from olmo_eval.harness.tools import search

    markdown = SimpleNamespace(raw_markdown="# Page\ntext...")
    crawler_cls = _install_fake_crawl4ai(
        monkeypatch,
        SimpleNamespace(success=True, status_code=200, markdown=markdown, cleaned_html=""),
    )

    result = await search.crawl4ai_browse(url=" https://example.com/page ")

    assert result == "# Page\ntext..."
    assert crawler_cls.instances[0].arun_urls == ["https://example.com/page"]


@pytest.mark.anyio
async def test_crawl4ai_browse_empty_url():
    from olmo_eval.harness.tools import search

    result = await search.crawl4ai_browse(url="  ")

    assert result == "Error: Empty URL."


@pytest.mark.anyio
async def test_crawl4ai_browse_missing_package(monkeypatch):
    from olmo_eval.harness.tools import search

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "crawl4ai":
            raise ImportError("missing crawl4ai")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = await search.crawl4ai_browse(url="https://example.com")

    assert result == "Error: crawl4ai is not installed."


@pytest.mark.anyio
async def test_crawl4ai_browse_unsuccessful_fetch(monkeypatch):
    from olmo_eval.harness.tools import search

    _install_fake_crawl4ai(
        monkeypatch,
        SimpleNamespace(success=False, status_code=503, markdown="", cleaned_html=""),
    )

    result = await search.crawl4ai_browse(url="https://example.com")

    assert result == "Error fetching webpage: HTTP 503"


@pytest.mark.anyio
async def test_crawl4ai_browse_arun_exception(monkeypatch):
    from olmo_eval.harness.tools import search

    _install_fake_crawl4ai(monkeypatch, RuntimeError("crawl failed"))

    result = await search.crawl4ai_browse(url="https://example.com")

    assert result.startswith("Error fetching webpage:")


@pytest.mark.anyio
async def test_crawl4ai_browse_falls_back_to_string_markdown(monkeypatch):
    from olmo_eval.harness.tools import search

    class StringCompatibleMarkdown:
        raw_markdown = ""

        def __str__(self):
            return "# Page\nfrom string"

    _install_fake_crawl4ai(
        monkeypatch,
        SimpleNamespace(
            success=True,
            status_code=200,
            markdown=StringCompatibleMarkdown(),
            cleaned_html="",
        ),
    )

    result = await search.crawl4ai_browse(url="https://example.com")

    assert result == "# Page\nfrom string"


@pytest.mark.anyio
async def test_crawl4ai_browse_empty_markdown(monkeypatch):
    from olmo_eval.harness.tools import search

    _install_fake_crawl4ai(
        monkeypatch,
        SimpleNamespace(success=True, status_code=200, markdown="", cleaned_html=""),
    )

    result = await search.crawl4ai_browse(url="https://example.com")

    assert result == "No content extracted from webpage."


@pytest.mark.anyio
async def test_crawl4ai_browse_truncates_markdown(monkeypatch):
    from olmo_eval.harness.tools import search

    _install_fake_crawl4ai(
        monkeypatch,
        SimpleNamespace(success=True, status_code=200, markdown="x" * 4001, cleaned_html=""),
    )

    result = await search.crawl4ai_browse(url="https://example.com")

    assert result == ("x" * 4000) + "\n\n[Content truncated...]"
