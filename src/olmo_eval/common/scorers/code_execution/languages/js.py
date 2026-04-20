"""JavaScript language evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import BaseLanguageEvaluator


@dataclass(frozen=True, slots=True)
class JsEvaluator(BaseLanguageEvaluator):
    """Evaluator for JavaScript code using Node.js."""

    LANG_NAME: ClassVar[str] = "JavaScript"
    LANG_EXT: ClassVar[str] = "js"
    LANG_ID: ClassVar[str] = "js"
    DEFAULT_TIMEOUT: ClassVar[float] = 5.0
    RUN_TIMEOUT: ClassVar[float] = 5.0

    filename: str = "code.js"
    compile_cmd: str | None = None
    run_cmd: str = "node {f}"

    def _is_syntax_error(self, exit_code: int, stdout: str, stderr: str) -> bool:
        # Node.js syntax errors
        return exit_code != 0 and ("SyntaxError" in stderr or "SyntaxError" in stdout)

    def _is_exception(self, exit_code: int, stdout: str, stderr: str) -> bool:
        combined = stdout + stderr
        # JavaScript runtime errors
        return exit_code != 0 and (
            "TypeError" in combined
            or "ReferenceError" in combined
            or "RangeError" in combined
            or "Error:" in combined
        )


evaluator = JsEvaluator()
