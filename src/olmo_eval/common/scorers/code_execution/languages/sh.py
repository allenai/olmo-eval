"""Bash/Shell language evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import BaseLanguageEvaluator


@dataclass(frozen=True, slots=True)
class ShEvaluator(BaseLanguageEvaluator):
    """Evaluator for Bash/Shell scripts."""

    LANG_NAME: ClassVar[str] = "Bash"
    LANG_EXT: ClassVar[str] = "sh"
    LANG_ID: ClassVar[str] = "sh"
    DEFAULT_TIMEOUT: ClassVar[float] = 5.0

    filename: str = "code.sh"
    compile_cmd: str | None = None
    run_cmd: str = "/bin/bash {f}"

    def _is_syntax_error(self, exit_code: int, stdout: str, stderr: str) -> bool:
        # Bash syntax errors
        return exit_code != 0 and (
            "syntax error" in stderr.lower()
            or "unexpected token" in stderr.lower()
            or "parse error" in stderr.lower()
        )

    def _is_exception(self, exit_code: int, stdout: str, stderr: str) -> bool:
        # Command not found, permission denied, etc.
        return exit_code != 0 and (
            "command not found" in stderr.lower()
            or "permission denied" in stderr.lower()
            or "no such file" in stderr.lower()
        )


evaluator = ShEvaluator()
