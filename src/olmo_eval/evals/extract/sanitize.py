"""Post-processing LLM-generated Python code using tree-sitter.

Parses generated code into an AST, builds a dependency graph of top-level
definitions, and reconstructs only the imports and definitions reachable
from a given entrypoint.  This removes dead helper code, markdown artifacts,
and other noise that would cause test failures.

Ported from the BigCodeBench project via oe-eval:
https://github.com/bigcode-project/bigcodebench/blob/main/bigcodebench/sanitize.py
"""

from __future__ import annotations

import ast
import traceback
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tree_sitter import Node

CLASS_TYPE = "class_definition"
FUNCTION_TYPE = "function_definition"
IMPORT_TYPE = ["import_statement", "import_from_statement"]
IDENTIFIER_TYPE = "identifier"
ATTRIBUTE_TYPE = "attribute"
RETURN_TYPE = "return_statement"
EXPRESSION_TYPE = "expression_statement"
ASSIGNMENT_TYPE = "assignment"


def _syntax_check(code: str, verbose: bool = False) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def _code_extract(text: str) -> str:
    """Find the longest contiguous range of lines that is valid Python."""
    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i : j + 1])
            if _syntax_check(current_lines):
                current_length = sum(1 for line in lines[i : j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "\n".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])


def _get_deps(nodes: list[tuple[str, Node]]) -> dict[str, set[str]]:
    name2deps: dict[str, set[str]] = {}
    for name, node in nodes:
        deps: set[str] = set()
        visited: set[int] = set()
        stack: list[Any] = [node]
        while stack:
            current = stack.pop()
            node_id = id(current)
            if node_id in visited:
                continue
            visited.add(node_id)
            for child in current.children:
                if child.type == IDENTIFIER_TYPE:
                    deps.add(child.text.decode("utf8"))
                else:
                    stack.append(child)
        name2deps[name] = deps
    return name2deps


def _get_function_dependency(entrypoint: str, call_graph: dict[str, set[str]]) -> set[str]:
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for neighbour in call_graph[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def _get_definition_name(node: Node) -> str | None:
    for child in node.children:
        if child.type == IDENTIFIER_TYPE and child.text is not None:
            return child.text.decode("utf8")
    return None


def _extract_target_code_or_empty(code: str, entrypoint: str | None = None) -> str:
    import tree_sitter_python
    from tree_sitter import Language, Parser

    code = code.strip()
    if not _syntax_check(code):
        code = _code_extract(code)
    code_bytes = bytes(code, "utf8")
    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(code_bytes)
    class_names: set[str] = set()
    function_names: set[str] = set()
    variable_names: set[str] = set()

    root_node = tree.root_node
    import_nodes: list[Node] = []
    definition_nodes: list[tuple[str, Node]] = []

    for child in root_node.children:
        if child.type in IMPORT_TYPE:
            import_nodes.append(child)
        elif child.type == CLASS_TYPE:
            name = _get_definition_name(child)
            if name and name not in class_names | variable_names | function_names:
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == FUNCTION_TYPE:
            name = _get_definition_name(child)
            if name and name not in function_names | variable_names | class_names:
                definition_nodes.append((name, child))
                function_names.add(name)
        elif child.type == EXPRESSION_TYPE and child.children[0].type == ASSIGNMENT_TYPE:
            subchild = child.children[0]
            name = _get_definition_name(subchild)
            if name and name not in variable_names | function_names | class_names:
                definition_nodes.append((name, subchild))
                variable_names.add(name)

    reachable: set[str] | None = None
    if entrypoint:
        name2deps = _get_deps(definition_nodes)
        reachable = _get_function_dependency(entrypoint, name2deps)

    sanitized_output = b""

    for node in import_nodes:
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"

    for name, node in definition_nodes:
        if reachable is not None and name not in reachable:
            continue
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"

    return sanitized_output[:-1].decode("utf8") if sanitized_output else ""


def sanitize_code(code: str, entrypoint: str | None = None) -> str:
    """Sanitize generated Python code, keeping only the entrypoint and its dependencies.

    Uses tree-sitter to parse the code, build a dependency graph, and
    reconstruct only the imports and definitions reachable from *entrypoint*.

    Falls back to extracting the longest syntactically valid Python substring
    if tree-sitter parsing produces no output.

    Args:
        code: The generated Python code (may include markdown fences or other noise).
        entrypoint: The target function/class name.  If provided, only code
            reachable from this name is kept.

    Returns:
        Cleaned Python code.
    """
    sanitized = _extract_target_code_or_empty(code, entrypoint).strip()
    if not sanitized:
        return _code_extract(code)
    return sanitized
