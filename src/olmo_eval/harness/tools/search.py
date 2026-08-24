"""Search tools for web and academic search.

This module provides search tools that can be used with the Harness:
- semantic_scholar_search: Search academic papers via Semantic Scholar API
- arxiv_paper_search: Search arXiv papers via Semantic Scholar, filtered to arXiv
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
from datetime import date, datetime, timedelta
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
_S2_MIN_INTERVAL_S = 1.1  # slightly over 1s for margin
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


# DeepScholar-Bench only credits sources that carry arXiv provenance and were
# published strictly before a query's cutoff. The constants below mirror
# lit-agents' shared/retrieval_policy.py so the tool's admission test is the
# benchmark exporter's admission test.
_ARXIV_OVERFETCH_MULTIPLIER = 5
_S2_MAX_SEARCH_LIMIT = 100  # S2 caps `limit` on /paper/search at 100
# Hits with no arXiv ID cannot be cited; a couple are worth showing as
# background, a page of them would crowd out the citable results.
_ARXIV_CONTEXT_ONLY_LIMIT = 2
_ARXIV_ABSTRACT_LIMIT = 500

_ARXIV_URL_RE = re.compile(
    r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/([^)\s?#]+)",
    re.IGNORECASE,
)
_ARXIV_MODERN_ID_RE = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}$")
_ARXIV_LEGACY_ID_RE = re.compile(r"^[a-z-]+/(\d{2})(\d{2})\d{3}$", re.IGNORECASE)


def normalize_arxiv_id(value: str) -> str:
    """Strip the prefix, extension, and version suffix from an arXiv ID."""
    normalized = value.strip()
    normalized = re.sub(r"(?i)^arxiv:", "", normalized)
    normalized = re.sub(r"(?i)\.pdf$", "", normalized)
    return re.sub(r"v[1-9][0-9]*$", "", normalized)


def _arxiv_id_from_url(url: str) -> str:
    """Return the arXiv ID embedded in an arXiv abs/pdf URL, or ""."""
    match = _ARXIV_URL_RE.search(url)
    return normalize_arxiv_id(match.group(1)) if match else ""


def _arxiv_id_from_paper(paper: dict) -> str:
    """Return the arXiv ID of an S2 search hit, or "" when it has none.

    External IDs rather than the `venue` field: an arXiv preprint loses the
    "arXiv.org" venue once it appears at a conference, while its arXiv external
    ID stays. The key comparison is case-insensitive because S2 spells the key
    "ArXiv".
    """
    external_ids = paper.get("externalIds") or {}
    if isinstance(external_ids, dict):
        for key, value in external_ids.items():
            if str(key).casefold() == "arxiv" and value is not None:
                arxiv_id = normalize_arxiv_id(str(value))
                if arxiv_id:
                    return arxiv_id
    return _arxiv_id_from_url(str(paper.get("url") or ""))


def date_from_arxiv_id(arxiv_id: str) -> date | None:
    """Return the month an arXiv ID encodes, or None when it encodes none."""
    match = _ARXIV_MODERN_ID_RE.match(arxiv_id) or _ARXIV_LEGACY_ID_RE.match(arxiv_id)
    if match is None:
        return None
    year_value, month_value = (int(value) for value in match.groups())
    year = 1900 + year_value if year_value >= 91 else 2000 + year_value
    try:
        return date(year, month_value, 1)
    except ValueError:
        return None


def _publication_date(paper: dict) -> date | None:
    """Parse an S2 hit's publicationDate, or None when it reports none."""
    raw = paper.get("publicationDate")
    if not raw:
        return None
    text = str(raw).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def _parse_cutoff_date(raw: str | None) -> date | None:
    """Parse the active search cutoff into a date, or None when unset."""
    cutoff = _parse_cutoff(raw)
    return None if cutoff is None else date.fromisoformat(cutoff[2])


def _arxiv_admits(paper: dict, cutoff: date | None, *, require_arxiv: bool) -> bool:
    """Return whether a hit is one DeepScholar-Bench's exporter would also keep.

    arXiv provenance, then a publication date strictly before the cutoff,
    falling back to the month encoded in the arXiv ID when S2 reports no date.
    Holding the tool to the exporter's test is what keeps it from inviting the
    model to cite a source the scorer will later discard. A hit S2 cannot date
    and whose ID encodes no date is dropped, because the exporter has no other
    evidence either.
    """
    arxiv_id = _arxiv_id_from_paper(paper)
    if require_arxiv and not arxiv_id:
        return False
    if cutoff is None:
        return True
    published = _publication_date(paper)
    if published is not None:
        return published < cutoff
    fallback = date_from_arxiv_id(arxiv_id) if arxiv_id else None
    if fallback is None:
        return False
    return (fallback.year, fallback.month) < (cutoff.year, cutoff.month)


def _arxiv_search_limit() -> int:
    """Results per arXiv search, overridable via ARXIV_SEARCH_LIMIT (default 10).

    Ten is the page lit-agents' DeepScholar retriever keeps per query.
    """
    try:
        return max(1, int(os.getenv("ARXIV_SEARCH_LIMIT", "10")))
    except ValueError:
        return 10


def _arxiv_request_limit(wanted: int) -> int:
    """Page size to request so client-side filtering can still fill ``wanted``.

    arXiv provenance has no server-side form on S2's search endpoint, so the
    page is widened by the overfetch multiplier and trimmed after filtering.
    """
    return max(1, min(_S2_MAX_SEARCH_LIMIT, wanted * _ARXIV_OVERFETCH_MULTIPLIER))


def _arxiv_published(paper: dict, arxiv_id: str) -> tuple[date, str] | None:
    """The date to show for a hit and whether it is day- or month-precise.

    Semantic Scholar reports a day; an arXiv ID encodes only a month. The
    export path writes ``date_precision`` straight from this distinction, and
    the benchmark contract rejects a month-precise source dated inside the
    cutoff's own month -- so a day that is known must never be reported as a
    month, or the source is discarded for want of information the search
    already had.
    """
    published = _publication_date(paper)
    if published is not None:
        return published, "day"
    fallback = date_from_arxiv_id(arxiv_id) if arxiv_id else None
    if fallback is not None:
        return fallback, "month"
    return None


def _format_arxiv_result(paper: dict, arxiv_id: str) -> str:
    """Render one hit; an empty ``arxiv_id`` marks it as uncitable context."""
    title = paper.get("title") or "Unknown"
    authors = paper.get("authors") or []
    author_names = ", ".join(a.get("name", "") for a in authors[:3])
    if len(authors) > 3:
        author_names += " et al."
    abstract = paper.get("abstract") or ""
    if len(abstract) > _ARXIV_ABSTRACT_LIMIT:
        abstract = abstract[:_ARXIV_ABSTRACT_LIMIT] + "..."

    if arxiv_id:
        lines = [f"**{title}**"]
    else:
        lines = [f"**{title}** [context only: no arXiv ID, not citable]"]
    if author_names:
        lines.append(f"Authors: {author_names}")
    year = paper.get("year")
    if year:
        lines.append(f"Year: {year}")
    published = _arxiv_published(paper, arxiv_id)
    if published is not None:
        value, precision = published
        if precision == "day":
            lines.append(f"Published: {value.isoformat()}")
        else:
            lines.append(f"Published: {value.year:04d}-{value.month:02d} (month precision)")
    if abstract:
        lines.append(f"Abstract: {abstract}")
    if arxiv_id:
        lines.append(f"arXiv: {arxiv_id}")
        lines.append(f"URL: https://arxiv.org/abs/{arxiv_id}")
    return "\n".join(lines)


@registered_tool(
    name="arxiv_paper_search",
    description=(
        "Search arXiv papers by keyword. Returns each paper's title, authors, year, "
        "abstract, arXiv ID and arxiv.org URL."
    ),
)
async def arxiv_paper_search(query: str) -> str:
    """Search arXiv papers, backed by Semantic Scholar and filtered to arXiv.

    Shares the Semantic Scholar endpoint, API keys, rate gate and backoff with
    :func:`semantic_scholar_search`; only the admission test differs. S2's
    search endpoint cannot select on external IDs, so provenance is decided
    client-side on ``externalIds.ArXiv`` and the requested page is widened to
    compensate. Every result rendered with an arxiv.org URL is one
    DeepScholar-Bench's exporter would also keep -- arXiv provenance, published
    strictly before the active :func:`search_date_cutoff` -- so the model is
    never invited to cite a source the scorer would discard. S2's relevance
    order is preserved: filtering skips hits, it never reorders them.

    Args:
        query: Search query for arXiv papers.

    Returns:
        Formatted results carrying title, authors, year, publication date,
        abstract snippet, arXiv ID and arxiv.org URL. A few hits with no arXiv
        ID follow, marked as context the answer cannot cite.
    """
    api_keys = _api_keys_from_env("S2_API_KEY")
    headers = {}
    if api_keys:
        headers["x-api-key"] = _select_api_key(api_keys)

    sanitized_query = query.strip()
    if not sanitized_query:
        return "Error: Empty search query."

    wanted = _arxiv_search_limit()
    cutoff = _parse_cutoff_date(_search_date_cutoff.get())
    params = {
        "query": sanitized_query,
        "limit": _arxiv_request_limit(wanted),
        "fields": "title,abstract,url,year,publicationDate,authors,corpusId,externalIds",
    }
    if cutoff is not None:
        # publicationDateOrYear ranges are inclusive, so stop one day short of
        # the cutoff to match the exporter's strict "before the cutoff" test.
        params["publicationDateOrYear"] = f":{(cutoff - timedelta(days=1)).isoformat()}"

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
                f"arXiv paper search API error (after retries): status={resp.status_code}, "
                f"query={sanitized_query!r}, response={resp.text[:500]}"
            )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            f"arXiv paper search HTTP error: status={e.response.status_code}, "
            f"query={sanitized_query!r}, response={e.response.text[:500]}"
        )
        return f"Error searching arXiv papers: HTTP {e.response.status_code}"
    except httpx.RequestError as e:
        logger.error(f"arXiv paper search request error: {e}, query={sanitized_query!r}")
        return f"Error searching arXiv papers: {e}"

    citable: list[str] = []
    context_only: list[str] = []
    for paper in data.get("data", []):
        if len(citable) >= wanted and len(context_only) >= _ARXIV_CONTEXT_ONLY_LIMIT:
            break
        arxiv_id = _arxiv_id_from_paper(paper)
        if arxiv_id:
            if len(citable) < wanted and _arxiv_admits(paper, cutoff, require_arxiv=True):
                citable.append(_format_arxiv_result(paper, arxiv_id))
        elif len(context_only) < _ARXIV_CONTEXT_ONLY_LIMIT and _arxiv_admits(
            paper, cutoff, require_arxiv=False
        ):
            context_only.append(_format_arxiv_result(paper, ""))

    if not citable and not context_only:
        return "No arXiv papers found for query."

    return "\n\n---\n\n".join(citable + context_only)


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
