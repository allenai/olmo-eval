"""PHP language evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import BaseLanguageEvaluator, EvalResult, ExecutionStatus


@dataclass(frozen=True, slots=True)
class PhpEvaluator(BaseLanguageEvaluator):
    """Evaluator for PHP code."""

    LANG_NAME: ClassVar[str] = "PHP"
    LANG_EXT: ClassVar[str] = "php"
    LANG_ID: ClassVar[str] = "php"
    DEFAULT_TIMEOUT: ClassVar[float] = 15.0

    filename: str = "code.php"
    compile_cmd: str | None = None
    run_cmd: str = "php {f}"

    def categorize_result(
        self,
        exit_code: int,
        stdout: str,
        stderr: str,
        timed_out: bool,
    ) -> EvalResult:
        """Categorize PHP execution result with specific error detection."""
        combined = stdout + stderr

        if timed_out:
            return EvalResult(ExecutionStatus.TIMEOUT, -1, stdout, stderr)

        # PHP-specific parse errors
        if "PHP Parse error" in combined or "Parse error" in combined:
            return EvalResult(ExecutionStatus.SYNTAX_ERROR, exit_code, stdout, stderr)

        # PHP fatal errors (runtime)
        if "PHP Fatal error" in combined or "Fatal error" in combined:
            return EvalResult(ExecutionStatus.EXCEPTION, exit_code, stdout, stderr)

        # PHP warnings and notices that cause failure
        if exit_code != 0 and (
            "PHP Warning" in combined or "PHP Notice" in combined or "PHP Error" in combined
        ):
            return EvalResult(ExecutionStatus.EXCEPTION, exit_code, stdout, stderr)

        if exit_code == 0:
            return EvalResult(ExecutionStatus.OK, 0, stdout, stderr)

        return EvalResult(ExecutionStatus.EXCEPTION, exit_code, stdout, stderr)


evaluator = PhpEvaluator()
