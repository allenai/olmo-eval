"""The DeepScholar-Bench export contract, as lit-agents implements it.

Verbatim excerpt of shared/retrieval_policy.py from ai2-multi-agent/integrated, which targets
github.com/guestrin-lab/deepscholar at commit
c95413b3b2f3255b461b90d0ce650f685ae2d1ff. Only the definitions the export tests
need are copied; nothing has been reworded.

It is here because the export is written to satisfy these functions, and a test
suite that only exercises our own reimplementation of them proves nothing about
whether the real ones would accept the output.

Do not edit. Re-extract from the reference if the benchmark is re-pinned.
"""

from __future__ import annotations

import re


_ARXIV_URL_RE = re.compile(
    r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/([^)\s?#]+)",
    re.IGNORECASE,
)


def normalize_arxiv_id(value: str) -> str:
    """Strip the prefix, extension, and version suffix from an arXiv ID."""

    normalized = value.strip()
    normalized = re.sub(r"(?i)^arxiv:", "", normalized)
    normalized = re.sub(r"(?i)\.pdf$", "", normalized)
    normalized = re.sub(r"v[1-9][0-9]*$", "", normalized)
    return normalized


def arxiv_id_from_url(url: str) -> str | None:
    """Return the arXiv ID embedded in an arXiv abs/pdf URL, if any."""

    match = _ARXIV_URL_RE.search(url)
    return normalize_arxiv_id(match.group(1)) if match else None

