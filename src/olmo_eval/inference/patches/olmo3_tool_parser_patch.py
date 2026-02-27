"""Patch for vLLM's olmo3 tool parser to handle JSON content in string arguments.

The OLMo3 model outputs tool calls in Python-like syntax:
    [submit(answer='{"key": "value"}')]

The vLLM parser uses ast.parse() which fails when the JSON string contains:
- Newlines (pretty-printed JSON)
- Single quotes in text (e.g., "Model's approach")
- Unescaped backslashes (e.g., "C:\\path")

This patch adds preprocessing to escape these characters before ast.parse().

Related issues:
- https://github.com/vllm-project/vllm/issues/32534 (OLMo-3 tool calling issue)
- https://github.com/vllm-project/vllm/pull/32539 (Chat template fix)

Usage:
    python -m olmo_eval.inference.patches.olmo3_tool_parser_patch [venv_path]

    venv_path: Optional path to the venv containing vLLM (e.g., /opt/vllm-venv)
               If not provided, searches current Python's site-packages.
"""

from __future__ import annotations

import argparse
import re
import site
import sys
from pathlib import Path

# The code to insert before ast.parse() to sanitize string content
# This replaces problematic characters that break Python string parsing
SANITIZE_CODE = r'''
            # PATCHED: Sanitize string arguments before ast.parse
            # See: https://github.com/vllm-project/vllm/issues/32534
            # JSON content in single-quoted strings can break Python parsing due to:
            # - Newlines (pretty-printed JSON)
            # - Single quotes in text (e.g., "Model's approach")
            # - Unescaped backslashes (e.g., "C:\path")
            import re as _re
            def _sanitize_python_strings(text: str) -> str:
                """Escape problematic characters inside single-quoted string arguments."""
                def _escape_content(m):
                    content = m.group(1)
                    # Escape ALL backslashes (model output is raw text, not Python escapes)
                    content = content.replace('\\', '\\\\')
                    # Escape literal control chars
                    content = content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    # Escape unescaped single quotes
                    content = _re.sub(r"(?<!\\)'", r"\\'", content)
                    return "='" + content + "'"
                # Non-greedy match with specific boundary detection:
                # - End at ' followed by , and next param (', name=')
                # - End at ' followed by ) (final arg)
                pattern = r"='(.*?)'(?=\s*(?:,\s*\w+=|\)))"
                return _re.sub(pattern, _escape_content, text, flags=_re.DOTALL)

            model_output = _sanitize_python_strings(model_output)
'''


def find_olmo3_parser(venv_path: str | None = None) -> Path | None:
    """Find the olmo3_tool_parser.py file in site-packages."""
    if venv_path:
        venv = Path(venv_path)
        for lib_dir in venv.glob("lib/python*/site-packages"):
            parser_path = lib_dir / "vllm" / "tool_parsers" / "olmo3_tool_parser.py"
            if parser_path.exists():
                return parser_path
        return None

    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        parser_path = Path(site_dir) / "vllm" / "tool_parsers" / "olmo3_tool_parser.py"
        if parser_path.exists():
            return parser_path
    return None


def patch_parser(parser_path: Path) -> bool:
    """Patch the olmo3 parser to sanitize strings before ast.parse().

    Returns True if patch was applied, False if already patched or not needed.
    """
    content = parser_path.read_text()

    # Check if already patched
    if "# PATCHED: Sanitize string arguments" in content:
        print(f"Parser already patched: {parser_path}")
        return False

    # Find the ast.parse call in extract_tool_calls and add sanitization before it
    # Pattern matches the line with ast.parse(model_output) capturing the indentation
    pattern = r"(\n)(\s+)(module = ast\.parse\(model_output\))"

    def replacement(match: re.Match) -> str:
        newline = match.group(1)
        indent = match.group(2)
        ast_parse_line = match.group(3)
        # Insert sanitization code before ast.parse, adjusting indentation
        sanitize = SANITIZE_CODE.replace("\n            ", f"\n{indent}")
        return f"{newline}{sanitize}\n{indent}{ast_parse_line}"

    new_content, count = re.subn(pattern, replacement, content, count=1)

    if count == 0:
        print(f"Could not find ast.parse pattern in {parser_path}")
        print("The parser may have a different structure than expected.")
        return False

    # Write the patched content
    parser_path.write_text(new_content)
    return True


def main() -> int:
    """Main entry point for the patch script."""
    arg_parser = argparse.ArgumentParser(
        description="Patch vLLM's olmo3 tool parser to handle JSON in string arguments"
    )
    arg_parser.add_argument(
        "venv_path",
        nargs="?",
        default=None,
        help="Path to venv containing vLLM (e.g., /opt/vllm-venv)",
    )
    args = arg_parser.parse_args()

    print("[OLMo3 Tool Parser Patch] Applying patch for JSON content handling...")
    print("  See: https://github.com/vllm-project/vllm/issues/32534")

    parser_path = find_olmo3_parser(args.venv_path)

    if parser_path is None:
        print("Could not find olmo3_tool_parser.py in site-packages")
        if args.venv_path:
            print(f"Searched in: {args.venv_path}")
        print("Make sure vLLM is installed before running this patch.")
        return 1

    print(f"Found parser at: {parser_path}")

    try:
        if patch_parser(parser_path):
            print(f"Patch applied successfully: {parser_path}")
        return 0
    except Exception as e:
        print(f"Error applying patch: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
