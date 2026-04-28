"""Rust language evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import BaseLanguageEvaluator


@dataclass(frozen=True, slots=True)
class RsEvaluator(BaseLanguageEvaluator):
    """Evaluator for Rust code using rustc."""

    LANG_NAME: ClassVar[str] = "Rust"
    LANG_EXT: ClassVar[str] = "rs"
    LANG_ID: ClassVar[str] = "rs"
    DEFAULT_TIMEOUT: ClassVar[float] = 15.0
    COMPILE_TIMEOUT: ClassVar[float] = 15.0
    RUN_TIMEOUT: ClassVar[float] = 5.0

    filename: str = "code.rs"
    compile_cmd: str | None = "rustc -o {d}/a.out {f}"
    run_cmd: str = "{d}/a.out"

    def _is_syntax_error(self, exit_code: int, stdout: str, stderr: str) -> bool:
        # rustc outputs "error[E" for compilation errors
        return exit_code != 0 and ("error[E" in stderr or "error:" in stderr)

    def _is_exception(self, exit_code: int, stdout: str, stderr: str) -> bool:
        combined = stdout + stderr
        # Rust runtime panics
        return exit_code != 0 and (
            "panicked at" in combined
            or "thread 'main' panicked" in combined
            or "SIGABRT" in combined
            or "SIGSEGV" in combined
        )


evaluator = RsEvaluator()
