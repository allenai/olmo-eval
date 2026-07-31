# PR 250 review — In-process crawl4ai `browse_webpage` tool

**Branch:** `yilun/crawl4ai-browse-tool` → `main`
**Files:** `pyproject.toml`, `src/olmo_eval/harness/presets.py`, `src/olmo_eval/harness/tools/search.py`, `tests/core/harness/test_crawl4ai_browse_tool.py` (new), `tests/core/harness/test_presets.py`
**Verdict:** Clean, well-tested, low risk. The default (no crawl4ai installed, no env var) leaves every existing path byte-identical. One performance/maintainability item worth considering before heavy use.

Lazy import, http(s)-only scheme guard before the browser starts, error sentinels that mirror `serper_fetch_webpage_content`, and a configurable truncation limit shared by both fetch tools. `os` is already imported in `search.py:16`, so `_truncate_webpage_content` is fine. 31 tests, crawl4ai fully mocked.

## Must fix before merging

- None.

## Could fix if time

- **`search.py` `crawl4ai_browse` lines ~196–198.** A fresh `AsyncWebCrawler()` is created and entered/exited on *every* call (`async with AsyncWebCrawler() as crawler`). Under an agentic preset with `max_turns=20–30`, a single instance may fetch many pages, so this spins up and tears down a headless browser per fetch — the dominant cost of the tool. crawl4ai is designed to reuse one crawler across many `arun()` calls. Consider a module-level lazily-created crawler (like the shared `_http_client` already in this file) or pooling, to avoid per-fetch browser startup. Correctness is fine; this is throughput.

- **`search.py` `crawl4ai_browse` scheme guard (line ~188).** As the PR notes, this is scheme-only — no DNS/redirect validation, so a model-chosen `http://` URL can still resolve to a private/link-local address (SSRF). Documented as out of scope; acceptable for a research eval harness, but call it out to whoever runs this against untrusted model output on a networked host.

## Small nit but probably fine

- **`search.py` `_truncate_webpage_content` lines ~139–153.** Reads and parses `OLMO_WEBPAGE_CONTENT_LIMIT` on every call. Negligible, and it keeps the env var live-editable; fine as is.

- **`pyproject.toml` line ~10 (`crawl4ai>=0.8`).** Loose lower bound with no upper cap. crawl4ai's `result.markdown` / `raw_markdown` surface has moved between releases; the code already defends with `getattr` + `str()` fallback, so this is fine, just noting the optional dep is fast-moving.

- **`presets.py` `web_search_agent_crawl4ai` (system prompt, lines ~82–95).** Prompt is duplicated inline rather than sharing the `WEB_SEARCH_SYSTEM_PROMPT` constant that PR #251 introduces. Minor divergence to reconcile if both land (they will conflict on `presets.py` regardless — see below).

## Merge-order note

Both this PR and #253 edit `search.py` (imports block + `serper_web_search`), and both this PR and #251 add presets to `presets.py`. Expect manual conflict resolution on both files; the changes are additive and compatible in intent.
