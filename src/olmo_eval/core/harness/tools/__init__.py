"""Pre-built tools for common use cases.

This package provides ready-to-use tools for agent tasks:
- search: Web and academic search tools
"""

from .search import (
    SEARCH_TOOLS,
    semantic_scholar_search,
    serper_fetch_page,
    serper_web_search,
)

__all__ = [
    "SEARCH_TOOLS",
    "semantic_scholar_search",
    "serper_web_search",
    "serper_fetch_page",
]
