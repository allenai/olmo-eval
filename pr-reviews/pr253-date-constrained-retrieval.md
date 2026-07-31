# PR 253 review — Date-constrain agentic retrieval to a per-instance cutoff

**Branch:** `yilun/date-constrained-retrieval` → `main`
**Files:** `src/olmo_eval/harness/scaffolds/openai_agents.py`, `src/olmo_eval/harness/tools/search.py` (+), `src/olmo_eval/runners/asynq/processing.py`, tests (new)
**Verdict:** Clean and well-scoped. The `ContextVar` threading matches the existing `_current_binding` pattern, the default (no cutoff) path is byte-identical, and the concurrency-isolation test is the right thing to have. No blocking issues.

I checked the metadata plumbing: `processing.py:98` reads `item.instance.metadata["retrieval_date_cutoff"]` into `trace_metadata["date_cutoff"]`, and the scaffold reads `trace_metadata["date_cutoff"]` — consistent. ResearchQA (#254) is the producer of `retrieval_date_cutoff`, so the two compose.

## Must fix before merging

- None.

## Could fix if time

- **`search.py` `semantic_scholar_search` client-side filter (lines ~228–241 of the new code).** The filter runs *after* `limit=_s2_search_limit()` is applied server-side. S2's `publicationDateOrYear=:<iso>` should already constrain the result set, so the client filter is belt-and-suspenders — but if S2 ever returns post-cutoff papers within the limited page, the client filter can shrink results below the limit and occasionally return "No papers found" even when older relevant papers exist beyond the page. In practice the server param handles it; just noting the interaction if coverage ever looks thin under a tight cutoff.

- **`search.py` string date comparison (lines ~130–137 of the diff).** `publicationDate <= cutoff_iso` is a lexicographic compare on ISO strings, which is correct for full `YYYY-MM-DD` values. If S2 ever returns a partial `publicationDate` like `"2010"` or `"2010-01"`, the prefix compare still behaves sanely (keeps it), so this is fine — worth a one-line comment that it relies on ISO ordering.

## Small nit but probably fine

- **`search.py` `serper_web_search` tbs value (line ~150 of the diff): `cd_min:01/01/1000`.** Magic sentinel for "no lower bound." Works; a named constant or comment would read better.

- **Naming: `retrieval_date_cutoff` (metadata) vs `date_cutoff` (trace_metadata) vs `_search_date_cutoff` (ContextVar).** Three names for one concept across the layers. All internally consistent, just a small readability tax when tracing the value end to end.

## Merge-order note

Overlaps `openai_agents.py` with #249 and #252 (the `with trace(...)` block that #253 wraps in `search_date_cutoff(...)` is a few lines above the `except` handler #249 rewrites) — likely a small manual merge. Also overlaps `search.py` with #250 (imports block + `serper_web_search`/`semantic_scholar_search` bodies). Both are additive; reconcile imports and the search-function bodies by hand.
