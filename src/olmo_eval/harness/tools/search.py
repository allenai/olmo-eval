"""Search tools for web and academic search.

This module provides search tools that can be used with the Harness:
- semantic_scholar_search: Search academic papers via Semantic Scholar API
- serper_web_search: Search the web via Serper/Google API
- serper_fetch_page: Fetch and extract webpage content

These tools are pre-registered in the global registry.
Import the tool objects and use .name for HarnessConfig.tool_names.
"""

from __future__ import annotations

import logging
import os

import httpx

from .registry import registered_tool

logger = logging.getLogger(__name__)

# Module-level shared HTTP client for connection reuse
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Get or create a shared async HTTP client.

    Returns a module-level client that reuses connections across tool calls.
    The client is automatically closed on module/process exit.
    """
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


@registered_tool(
    name="semantic_scholar_snippet_search",
    description="Search Semantic Scholar for academic papers and snippets matching a query.",
)
async def semantic_scholar_search(query: str) -> str:
    """Search Semantic Scholar for academic papers and snippets matching a query.

    Args:
        query: Search query for academic papers and snippets.

    Returns:
        Formatted search results with paper titles, abstracts, and URLs.
    """
    api_key = os.getenv("S2_API_KEY")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    sanitized_query = query.strip()
    if not sanitized_query:
        return "Error: Empty search query."

    client = _get_http_client()
    try:
        resp = await client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": sanitized_query,
                "limit": 5,
                "fields": "title,abstract,url,year,authors",
            },
            headers=headers,
        )
        if resp.status_code != 200:
            logger.error(
                f"Semantic Scholar API error: status={resp.status_code}, "
                f"query={sanitized_query!r}, response={resp.text[:500]}"
            )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Semantic Scholar HTTP error: status={e.response.status_code}, "
            f"query={sanitized_query!r}, response={e.response.text[:500]}"
        )
        return f"Error searching Semantic Scholar: HTTP {e.response.status_code}"
    except httpx.RequestError as e:
        logger.error(f"Semantic Scholar request error: {e}, query={sanitized_query!r}")
        return f"Error searching Semantic Scholar: {e}"

    papers = data.get("data", [])
    if not papers:
        return "No papers found for query."

    results = []
    for paper in papers:
        title = paper.get("title", "Unknown")
        abstract = paper.get("abstract", "No abstract available")
        url = paper.get("url", "")
        year = paper.get("year", "")
        authors = paper.get("authors", [])
        author_names = ", ".join(a.get("name", "") for a in authors[:3])
        if len(authors) > 3:
            author_names += " et al."

        result = f"**{title}**"
        if year:
            result += f" ({year})"
        if author_names:
            result += f"\nAuthors: {author_names}"
        if abstract:
            # Truncate long abstracts
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            result += f"\nAbstract: {abstract}"
        if url:
            result += f"\nURL: {url}"
        results.append(result)

    return "\n\n---\n\n".join(results)


@registered_tool(
    name="serper_google_webpage_search",
    description="Search the web for information using Google via Serper.",
)
async def serper_web_search(query: str) -> str:
    """Search the web for information using Google via Serper.

    Args:
        query: The search query to find relevant web pages.

    Returns:
        Formatted search results with titles, snippets, and URLs.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY not configured."

    # Sanitize query - remove problematic characters
    sanitized_query = query.strip()
    if not sanitized_query:
        return "Error: Empty search query."

    client = _get_http_client()
    try:
        resp = await client.post(
            "https://google.serper.dev/search",
            json={"q": sanitized_query, "num": 5},
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
        )
        if resp.status_code != 200:
            logger.error(
                f"Serper API error: status={resp.status_code}, "
                f"query={sanitized_query!r}, response={resp.text[:500]}"
            )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Serper HTTP error: status={e.response.status_code}, "
            f"query={sanitized_query!r}, response={e.response.text[:500]}"
        )
        return f"Error searching web: HTTP {e.response.status_code}"
    except httpx.RequestError as e:
        logger.error(f"Serper request error: {e}, query={sanitized_query!r}")
        return f"Error searching web: {e}"

    results = []

    # Process organic results
    organic = data.get("organic", [])
    for item in organic[:5]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        result = f"**{title}**\n{snippet}\nURL: {link}"
        results.append(result)

    # Include knowledge graph if available
    kg = data.get("knowledgeGraph")
    if kg:
        kg_title = kg.get("title", "")
        kg_desc = kg.get("description", "")
        if kg_title and kg_desc:
            results.insert(0, f"**Knowledge Graph: {kg_title}**\n{kg_desc}")

    # Include answer box if available
    answer_box = data.get("answerBox")
    if answer_box:
        answer = answer_box.get("answer") or answer_box.get("snippet", "")
        if answer:
            results.insert(0, f"**Direct Answer:**\n{answer}")

    if not results:
        return "No search results found."

    return "\n\n---\n\n".join(results)


@registered_tool(
    name="serper_fetch_webpage_content",
    description="Fetch and extract content from a webpage URL.",
)
async def serper_fetch_page(url: str) -> str:
    """Fetch and extract content from a webpage URL.

    Args:
        url: The URL of the webpage to fetch.

    Returns:
        Extracted text content from the webpage.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY not configured."

    if not url or not url.strip():
        return "Error: Empty URL."

    client = _get_http_client()
    try:
        resp = await client.post(
            "https://scrape.serper.dev",
            json={"url": url.strip()},
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
        )
        if resp.status_code != 200:
            logger.error(
                f"Serper scrape API error: status={resp.status_code}, "
                f"url={url!r}, response={resp.text[:500]}"
            )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Serper scrape HTTP error: status={e.response.status_code}, "
            f"url={url!r}, response={e.response.text[:500]}"
        )
        return f"Error fetching webpage: HTTP {e.response.status_code}"
    except httpx.RequestError as e:
        logger.error(f"Serper scrape request error: {e}, url={url!r}")
        return f"Error fetching webpage: {e}"

    # Extract text content
    text = data.get("text", "")
    if not text:
        return "No content extracted from webpage."

    # Truncate if too long
    if len(text) > 4000:
        text = text[:4000] + "\n\n[Content truncated...]"

    return text
