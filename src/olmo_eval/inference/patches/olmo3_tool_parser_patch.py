"""Patch for vLLM's olmo3 tool parser to handle bracket-wrapped function calls.

The OLMo3 model sometimes outputs tool calls wrapped in square brackets like:
    [submit(answer='...')]
instead of:
    submit(answer='...')

This patch modifies the parser to strip brackets before parsing with ast.parse().

Related issues:
- https://github.com/vllm-project/vllm/issues/32534 (OLMo-3 tool calling issue)
- https://github.com/vllm-project/vllm/pull/32539 (Chat template fix)

Usage:
    python -m olmo_eval.inference.patches.olmo3_tool_parser_patch [venv_path]

    venv_path: Optional path to the venv containing vLLM (e.g., /opt/vllm-venv)
               If not provided, searches current Python's site-packages.

This script should be run after vLLM is installed.
"""

from __future__ import annotations

import argparse
import re
import site
import sys
from pathlib import Path


def find_olmo3_parser(venv_path: str | None = None) -> Path | None:
    """Find the olmo3_tool_parser.py file in site-packages.

    Args:
        venv_path: Optional path to a venv to search. If None, searches
                   the current Python's site-packages.
    """
    if venv_path:
        # Search in the specified venv
        venv = Path(venv_path)
        for lib_dir in venv.glob("lib/python*/site-packages"):
            parser_path = lib_dir / "vllm" / "tool_parsers" / "olmo3_tool_parser.py"
            if parser_path.exists():
                return parser_path
        return None

    # Search in current Python's site-packages
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        parser_path = Path(site_dir) / "vllm" / "tool_parsers" / "olmo3_tool_parser.py"
        if parser_path.exists():
            return parser_path
    return None


def patch_parser(parser_path: Path) -> bool:
    """Patch the olmo3 parser to strip brackets before ast.parse().

    Returns True if patch was applied, False if already patched or not needed.
    """
    content = parser_path.read_text()

    # Check if already patched
    if "# PATCHED: Strip brackets" in content:
        print(f"Parser already patched: {parser_path}")
        return False

    # Find the ast.parse call and add bracket stripping before it
    # The original code is typically:
    #     module = ast.parse(model_output)
    # We want to add bracket stripping before it

    # Pattern to find the ast.parse line in extract_tool_calls
    pattern = r"(\s+)module = ast\.parse\(model_output\)"

    def replacement(match: re.Match) -> str:
        indent = match.group(1)
        return f"""{indent}# PATCHED: Strip brackets from model output if present
{indent}# See: https://github.com/vllm-project/vllm/issues/32534
{indent}# OLMo3 sometimes outputs tool calls wrapped in brackets like [func(...)]
{indent}_model_output = model_output.strip()
{indent}if _model_output.startswith('[') and _model_output.endswith(']'):
{indent}    _model_output = _model_output[1:-1].strip()
{indent}module = ast.parse(_model_output)"""

    new_content, count = re.subn(pattern, replacement, content)

    if count == 0:
        print(f"Could not find ast.parse pattern in {parser_path}")
        print("The parser may have a different structure than expected.")
        return False

    # Write the patched content
    parser_path.write_text(new_content)
    print(f"Successfully patched: {parser_path}")
    return True


def main() -> int:
    """Main entry point for the patch script."""
    arg_parser = argparse.ArgumentParser(
        description="Patch vLLM's olmo3 tool parser to handle bracket-wrapped calls"
    )
    arg_parser.add_argument(
        "venv_path",
        nargs="?",
        default=None,
        help="Path to venv containing vLLM (e.g., /opt/vllm-venv)",
    )
    args = arg_parser.parse_args()

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
            print("Patch applied successfully!")
        return 0
    except Exception as e:
        print(f"Error applying patch: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
