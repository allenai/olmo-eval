"""Helpers for loading pairwise viewer templates and static assets."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from string import Template

_ASSET_ROOT = Path(__file__).resolve().parent


@cache
def _read_text(*parts: str) -> str:
    return (_ASSET_ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def shared_css_text() -> str:
    return _read_text("static", "shared.css")


def standalone_js_text() -> str:
    return _read_text("static", "standalone.js")


def browser_css_text() -> str:
    return _read_text("static", "browser.css")


def browser_js_text() -> str:
    return _read_text("static", "browser.js")


def render_template(template_name: str, **context: str) -> str:
    return Template(_read_text("templates", template_name)).substitute(**context)
