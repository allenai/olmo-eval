"""Code extraction utilities for code generation tasks."""

import re


def extract_code(text: str, language: str = "python") -> str:
    """Extract code from model output.

    Looks for code blocks in the format:
    ```python
    code here
    ```

    Falls back to the full text if no code block is found.

    Args:
        text: The model output text.
        language: The programming language to look for (default: "python").

    Returns:
        The extracted code string.
    """
    # Try to extract from markdown code block
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[0]

    # Try generic code block
    pattern = re.compile(r"```\n?(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[0]

    # Fall back to full text
    return text


def extract_function_body(text: str, signature: str | None = None) -> str:
    """Extract a function body from code.

    Args:
        text: The code text.
        signature: Optional function signature to find.

    Returns:
        The function body.
    """
    code = extract_code(text)

    if signature:
        # Find where the signature ends and body begins
        idx = code.find(signature)
        if idx >= 0:
            code = code[idx + len(signature) :]
            # Find the colon and start of body
            colon_idx = code.find(":")
            if colon_idx >= 0:
                code = code[colon_idx + 1 :]

    return code.strip()


def indent_code(code: str, indent: str = "    ") -> str:
    """Ensure code has consistent indentation for use as a function body.

    If the code appears to be unindented (first non-empty line has no leading
    whitespace), adds the specified indentation to all lines. If the code
    already has indentation, returns it unchanged.

    This handles the common case where chat models output function body code
    without the leading indentation expected by HumanEval-style tasks.

    Args:
        code: The code to potentially indent.
        indent: The indentation to add (default: 4 spaces).

    Returns:
        The code with consistent indentation.
    """
    if not code:
        return code

    lines = code.split("\n")

    # Find first non-empty line to check current indentation
    first_content_line = None
    for line in lines:
        if line.strip():
            first_content_line = line
            break

    if first_content_line is None:
        return code

    # If first content line already has indentation, assume code is properly indented
    if first_content_line.startswith((" ", "\t")):
        return code

    # Add indentation to all non-empty lines
    indented_lines = []
    for line in lines:
        if line.strip():
            indented_lines.append(indent + line)
        else:
            indented_lines.append(line)

    return "\n".join(indented_lines)
