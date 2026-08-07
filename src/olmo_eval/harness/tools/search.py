"""Search tools for web and academic search.

This module provides search tools that can be used with the Harness:
- semantic_scholar_search: Search academic papers via Semantic Scholar API
- serper_web_search: Search the web via Serper/Google API
- serper_fetch_page: Fetch and extract webpage content
- crawl4ai_browse: Fetch and extract webpage content via crawl4ai

These tools are pre-registered in the global registry.
Import the tool objects and use .name for HarnessConfig.tool_names.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import random
import re
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
from urllib.parse import urlsplit

import httpx

from .registry import registered_tool

logger = logging.getLogger(__name__)

_search_date_cutoff: ContextVar[str | None] = ContextVar("_search_date_cutoff", default=None)

# Module-level shared HTTP client for connection reuse
_http_client: httpx.AsyncClient | None = None

# Statuses worth retrying with backoff: rate limiting + transient server errors.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 5
_BASE_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 16.0
_WEBPAGE_CONTENT_LIMIT = 4000
_WEBPAGE_TRUNCATION_NOTICE = "\n\n[Content truncated...]"

# Semantic Scholar's introductory plan allows ~1 request/second cumulative across
# all endpoints, which several concurrent agents exhaust immediately. All S2
# requests pass through a process-global minimum interval (proactive throttle);
# backoff handles the occasional 429 that still gets through. The limit is
# enforced per API key, so the interval below is the budget for a single key and
# _s2_rate_interval() divides it by the number of configured keys.
# Default sized for Semantic Scholar's introductory plan, which allows about 1 request/second per
# key. A provisioned key allows far more, and this being a hard-coded constant meant every run
# throttled itself to the free-tier rate whatever key it was given: a 1,846-instance run took 6.5
# hours against keys measured to sustain 13 requests/second each, with 429s first appearing at 26.
#
# Set OLMO_EVAL_S2_MIN_INTERVAL to the per-key spacing your keys actually permit. It is divided by
# the number of configured keys, as before, so 0.2 with two keys gives 10 requests/second in
# aggregate. Measure before raising it, and do not read the error to decide: the 429 body S2
# returns to an authenticated caller over its limit is byte-identical to the one it returns to an
# unauthenticated caller, so the message cannot tell you which of the two you hit.
_S2_MIN_INTERVAL_S = float(os.environ.get("OLMO_EVAL_S2_MIN_INTERVAL", "1.1"))
_s2_rate_lock = asyncio.Lock()
_s2_last_request_ts = 0.0  # time.monotonic() of the last dispatched S2 request


@contextmanager
def search_date_cutoff(cutoff_raw: str | None) -> Iterator[None]:
    """Set the date cutoff used by search tools within this context."""
    token = _search_date_cutoff.set(cutoff_raw)
    try:
        yield
    finally:
        _search_date_cutoff.reset(token)


def _parse_cutoff(raw: str | None) -> tuple[int, str, str] | None:
    """Parse a leading ISO date into year, Google, and ISO date forms."""
    if not isinstance(raw, str):
        return None
    if not raw:
        return None

    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", raw)
    if match is None:
        logger.warning("Ignoring unparseable search date cutoff: %r", raw)
        return None

    year_text, month_text, day_text = match.groups()
    try:
        date(int(year_text), int(month_text), int(day_text))
    except ValueError:
        logger.warning("Ignoring unparseable search date cutoff: %r", raw)
        return None

    return (
        int(year_text),
        f"{month_text}/{day_text}/{year_text}",
        f"{year_text}-{month_text}-{day_text}",
    )


def _truncate_webpage_content(text: str) -> str:
    limit_value = os.environ.get("OLMO_WEBPAGE_CONTENT_LIMIT")
    if limit_value is None:
        limit = _WEBPAGE_CONTENT_LIMIT
    else:
        try:
            limit = int(limit_value)
        except ValueError:
            limit = _WEBPAGE_CONTENT_LIMIT

    if limit <= 0:
        return text
    if len(text) > limit:
        return text[:limit] + _WEBPAGE_TRUNCATION_NOTICE
    return text


def _api_keys_from_env(base_var: str) -> list[str]:
    """Collect every API key configured for ``base_var``, in a stable order.

    Reads ``<base_var>`` first, then every ``<base_var>_<n>`` in ascending
    numeric order (so ``_2`` precedes ``_10``). Each variable may hold one key
    or a comma-separated list, which keeps both deployment shapes usable: one
    Beaker secret per key, or a single secret holding all of them.

    Blank entries are dropped and duplicates collapse to their first
    occurrence, so one configured key always yields a one-element list and no
    key gets a larger share of the request stream than the others.
    """
    prefix = f"{base_var}_"
    numbered: list[tuple[int, str]] = []
    for name, value in os.environ.items():
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            numbered.append((int(suffix), value))

    raw = [os.getenv(base_var, "")]
    raw.extend(value for _, value in sorted(numbered, key=lambda item: item[0]))

    keys: list[str] = []
    for value in raw:
        for candidate in value.split(","):
            key = candidate.strip()
            if key and key not in keys:
                keys.append(key)
    return keys


# Round-robin cursor over the configured keys. The starting offset is random so
# that independent worker processes sharing a key set do not all open on the
# same key; within a process the cursor then advances deterministically.
_api_key_cursor = itertools.count(random.randrange(1 << 16))


def _select_api_key(keys: Sequence[str]) -> str:
    """Pick the key to use for one request, cycling through ``keys``.

    Round-robin rather than random choice: the rate limit is per key, so what
    matters is that no key receives two requests inside its own interval.
    Cycling guarantees an exact 1/N share, whereas independent random draws
    would sometimes land on the same key twice in a row and re-trigger the 429
    the extra key was added to avoid.
    """
    return keys[next(_api_key_cursor) % len(keys)]


def _s2_rate_interval() -> float:
    """Minimum spacing between S2 requests given the configured keys.

    S2 meters per key, so N distinct keys permit N times the aggregate rate
    while each individual key still sees at most one request per
    _S2_MIN_INTERVAL_S. Zero or one key keeps the original spacing exactly.
    """
    return _S2_MIN_INTERVAL_S / max(1, len(_api_keys_from_env("S2_API_KEY")))


async def _s2_rate_gate() -> None:
    """Block until the per-request S2 interval has elapsed since the last call.

    Serializes S2 requests across all concurrent agents in this process so the
    cumulative rate stays under the limit the configured keys allow.
    """
    global _s2_last_request_ts
    async with _s2_rate_lock:
        wait = _s2_last_request_ts + _s2_rate_interval() - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _s2_last_request_ts = time.monotonic()


def _get_http_client() -> httpx.AsyncClient:
    """Get or create a shared async HTTP client.

    Returns a module-level client that reuses connections across tool calls.
    The client is automatically closed on module/process exit.
    """
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


def _s2_search_limit() -> int:
    """Results per S2 search, overridable via S2_SEARCH_LIMIT (default 5).

    Raising this widens coverage per query (closer to Recall@20-style behavior)
    at the cost of longer tool output; set e.g. S2_SEARCH_LIMIT=20 in the run env.
    """
    try:
        return max(1, int(os.getenv("S2_SEARCH_LIMIT", "5")))
    except ValueError:
        return 5


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header value in seconds, ignoring HTTP-date form."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None  # HTTP-date form is rare here; fall back to exponential backoff


async def _get_with_backoff(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict,
    headers: dict,
    max_retries: int = _MAX_RETRIES,
    rate_gate: Callable[[], Awaitable[None]] | None = None,
) -> httpx.Response:
    """GET with exponential backoff + jitter on 429/5xx, honoring Retry-After.

    Retries transient failures (rate limits, 5xx, network errors). The Retry-After
    header takes precedence over the computed backoff. Returns the final response
    even if still failing after the last attempt; the caller handles its status.
    If ``rate_gate`` is given it is awaited before every attempt (including
    retries) to enforce a proactive request rate.
    """
    delay = _BASE_BACKOFF_S
    for attempt in range(max_retries + 1):
        retry_after: float | None = None
        try:
            if rate_gate is not None:
                await rate_gate()
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code not in _RETRYABLE_STATUS or attempt == max_retries:
                return resp
            retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
        except httpx.RequestError:
            if attempt == max_retries:
                raise

        wait = retry_after if retry_after is not None else delay + random.uniform(0, delay)
        await asyncio.sleep(min(wait, _MAX_BACKOFF_S))
        delay = min(delay * 2, _MAX_BACKOFF_S)

    raise RuntimeError("unreachable: backoff loop exited without returning")


@registered_tool(
    name="semantic_scholar_snippet_search",
    description="Search Semantic Scholar for academic papers and snippets matching a query.",
)
async def semantic_scholar_search(query: str) -> str:
    """Search Semantic Scholar for academic papers and snippets matching a query.

    Authenticates with one of the configured keys (S2_API_KEY and any
    S2_API_KEY_<n>, each optionally comma-separated), chosen per request so the
    per-key rate limit is shared across all of them. With no key configured the
    request is sent unauthenticated, as before.

    Args:
        query: Search query for academic papers and snippets.

    Returns:
        Formatted search results with paper titles, abstracts, and URLs.
    """
    api_keys = _api_keys_from_env("S2_API_KEY")
    headers = {}
    if api_keys:
        headers["x-api-key"] = _select_api_key(api_keys)

    sanitized_query = query.strip()
    if not sanitized_query:
        return "Error: Empty search query."

    cutoff = _parse_cutoff(_search_date_cutoff.get())
    params = {
        "query": sanitized_query,
        "limit": _s2_search_limit(),
        "fields": "title,abstract,url,year,publicationDate,authors,corpusId",
    }
    if cutoff is not None:
        params["publicationDateOrYear"] = f":{cutoff[2]}"

    client = _get_http_client()
    try:
        resp = await _get_with_backoff(
            client,
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            headers=headers,
            rate_gate=_s2_rate_gate,
        )
        if resp.status_code != 200:
            logger.error(
                f"Semantic Scholar API error (after retries): status={resp.status_code}, "
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
    if cutoff is not None:
        cutoff_year, _, cutoff_iso = cutoff
        papers = [
            paper
            for paper in papers
            if (
                isinstance(paper.get("publicationDate"), str)
                and paper["publicationDate"] <= cutoff_iso
            )
            or (
                not isinstance(paper.get("publicationDate"), str)
                and isinstance(paper.get("year"), int)
                and paper["year"] <= cutoff_year
            )
        ]
    if not papers:
        return "No papers found for query."

    results = []
    for paper in papers:
        title = paper.get("title", "Unknown")
        abstract = paper.get("abstract", "No abstract available")
        url = paper.get("url", "")
        year = paper.get("year", "")
        corpus_id = paper.get("corpusId")
        authors = paper.get("authors", [])
        author_names = ", ".join(a.get("name", "") for a in authors[:3])
        if len(authors) > 3:
            author_names += " et al."

        result = f"**{title}**"
        if year:
            result += f" ({year})"
        if corpus_id is not None:
            result += f"\nCorpus ID: {corpus_id}"
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

    cutoff = _parse_cutoff(_search_date_cutoff.get())
    request_body = {"q": sanitized_query, "num": 5}
    if cutoff is not None:
        _, cutoff_google, _ = cutoff
        request_body["tbs"] = f"cdr:1,cd_min:01/01/1000,cd_max:{cutoff_google}"

    client = _get_http_client()
    try:
        resp = await client.post(
            "https://google.serper.dev/search",
            json=request_body,
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

    text = _truncate_webpage_content(text)

    return text


@registered_tool(
    name="browse_webpage",
    description="Fetch and extract a webpage's content as clean markdown (via crawl4ai).",
)
async def crawl4ai_browse(url: str) -> str:
    """Fetch and extract a webpage's content as clean markdown via crawl4ai.

    Args:
        url: The URL of the webpage to fetch.

    Returns:
        Extracted markdown content from the webpage.
    """
    if not url or not url.strip():
        return "Error: Empty URL."

    sanitized_url = url.strip()
    if urlsplit(sanitized_url).scheme.lower() not in {"http", "https"}:
        return "Error: Only http(s) URLs are supported."

    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError:
        return "Error: crawl4ai is not installed."

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(sanitized_url)
    except Exception as e:
        logger.exception(f"crawl4ai browse error: url={sanitized_url!r}")
        return f"Error fetching webpage: {e}"

    if not getattr(result, "success", False):
        error_message = getattr(result, "error_message", None)
        if error_message:
            return f"Error fetching webpage: {error_message}"
        status_code = getattr(result, "status_code", None)
        if status_code is not None:
            return f"Error fetching webpage: HTTP {status_code}"
        return "Error fetching webpage: unknown error"

    markdown = getattr(result, "markdown", None)
    text = getattr(markdown, "raw_markdown", None) or str(markdown or "")
    if not text.strip():
        return "No content extracted from webpage."

    text = _truncate_webpage_content(text)

    return text
