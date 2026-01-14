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
