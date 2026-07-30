"""Backport vLLM's GPT-OSS Responses function-recipient fix.

vLLM 0.19.1 only classifies ``functions.*`` recipients as function calls when
the completed Harmony message uses the commentary channel. Its streaming path
already accepts the same recipients from analysis, and GPT-OSS can emit them
there after an earlier tool result. The completed response is consequently
misclassified as an already-executed MCP call and local agent SDKs skip it.

vLLM 0.22 removes the channel restriction. This module applies the minimal
equivalent change to the Torch-2.10-compatible vLLM 0.19.1 serving environment.
"""

from __future__ import annotations

import site
from pathlib import Path

_OLD_CONDITION = 'elif message.channel == "commentary" and recipient.startswith("functions."):'
_NEW_CONDITION = 'elif recipient.startswith("functions."):'


def find_harmony_parser(venv_path: str | None = None) -> Path | None:
    """Find vLLM's Responses Harmony parser in an optional virtualenv."""
    if venv_path:
        search_dirs = Path(venv_path).glob("lib/python*/site-packages")
    else:
        search_dirs = (Path(path) for path in site.getsitepackages() + [site.getusersitepackages()])

    for site_dir in search_dirs:
        parser_path = site_dir / "vllm" / "entrypoints" / "openai" / "responses" / "harmony.py"
        if parser_path.exists():
            return parser_path
    return None


def patch_parser(parser_path: Path) -> bool:
    """Apply the channel-independent function-recipient condition.

    Returns ``True`` when the file changed and ``False`` when the upstream fix
    is already present. Raises if the installed parser has an unknown shape so
    an evaluation cannot silently proceed with an ineffective patch.
    """
    content = parser_path.read_text()
    if _OLD_CONDITION in content:
        parser_path.write_text(content.replace(_OLD_CONDITION, _NEW_CONDITION, 1))
        return True
    if _NEW_CONDITION in content:
        return False
    raise RuntimeError(
        f"Unsupported vLLM Responses Harmony parser layout: {parser_path}. "
        "Expected the vLLM 0.19.1 condition or its upstream-fixed form."
    )
