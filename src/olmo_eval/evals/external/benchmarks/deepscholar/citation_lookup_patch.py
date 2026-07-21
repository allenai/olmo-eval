"""In-place patch for the pinned deepscholar-bench citation lookup.

Upstream ``eval/utils.py::get_citation_count_from_title`` hardcodes
``mailto="your@email.com"`` and sends no API key, so at benchmark volume every
request lands in OpenAlex's anonymous tier and 429s. Since it returns ``None`` on
any failure, ``document_importance`` (median system-citation / median exemplar-
citation) collapses toward 0 on rate-limited runs.

This script rewrites two lines so the lookup reads ``OPENALEX_EMAIL`` (polite pool)
and ``OPENALEX_API_KEY`` (premium tier) from the environment at request time, with
no credentials embedded here. It is applied to the cloned repo at setup and fails
loudly if the upstream lines have changed (i.e. the pinned ref moved), which is the
signal to re-verify the substitution.

Usage: ``python citation_lookup_patch.py <path-to-eval/utils.py>``
"""

from __future__ import annotations

import pathlib
import sys

_MAILTO_OLD = 'title, mailto="your@email.com", similarity_threshold=0.8'
_MAILTO_NEW = (
    'title, mailto=os.environ.get("OPENALEX_EMAIL") or "your@email.com", similarity_threshold=0.8'
)

_URL_OLD = 'search_url = f"https://api.openalex.org/works?search={title}&mailto={mailto}"'
_URL_NEW = (
    'search_url = f"https://api.openalex.org/works?search={title}&mailto={mailto}"'
    " + (f\"&api_key={os.environ['OPENALEX_API_KEY']}\""
    ' if os.environ.get("OPENALEX_API_KEY") else "")'
)


def main() -> None:
    path = pathlib.Path(sys.argv[1])
    source = path.read_text()
    patched = source.replace(_MAILTO_OLD, _MAILTO_NEW).replace(_URL_OLD, _URL_NEW)
    if _MAILTO_NEW not in patched or _URL_NEW not in patched:
        raise SystemExit(
            f"citation patch: expected lines not found in {path}; "
            "the pinned deepscholar-bench ref may have changed."
        )
    path.write_text(patched)


if __name__ == "__main__":
    main()
