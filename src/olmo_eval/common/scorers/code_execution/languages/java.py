"""Java language evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import BaseLanguageEvaluator


@dataclass(frozen=True, slots=True)
class JavaEvaluator(BaseLanguageEvaluator):
    """Evaluator for Java code using javac and java."""

    LANG_NAME: ClassVar[str] = "Java"
    LANG_EXT: ClassVar[str] = "java"
    LANG_ID: ClassVar[str] = "java"
    DEFAULT_TIMEOUT: ClassVar[float] = 15.0

    filename: str = "Problem.java"
    compile_cmd: str | None = "cd {d} && javac -encoding UTF8 -cp '/runtime/java/*' Problem.java"
    run_cmd: str = "cd {d} && JAVA_TOOL_OPTIONS='-ea' java -ea -cp '/runtime/java/*:.' Problem 2>&1"

    def _is_syntax_error(self, exit_code: int, stdout: str, stderr: str) -> bool:
        # javac outputs "error:" for compilation errors
        return exit_code != 0 and ("error:" in stderr or "error:" in stdout)

    def _is_exception(self, exit_code: int, stdout: str, stderr: str) -> bool:
        combined = stdout + stderr
        # Java runtime exceptions
        return exit_code != 0 and (
            "Exception in thread" in combined
            or "java.lang." in combined
            or "at " in combined  # Stack trace indicator
        )


evaluator = JavaEvaluator()
