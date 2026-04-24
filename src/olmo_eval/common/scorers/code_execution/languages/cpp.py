"""C++ language evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import BaseLanguageEvaluator


@dataclass(frozen=True, slots=True)
class CppEvaluator(BaseLanguageEvaluator):
    """Evaluator for C++ code using g++."""

    LANG_NAME: ClassVar[str] = "C++"
    LANG_EXT: ClassVar[str] = "cpp"
    LANG_ID: ClassVar[str] = "cpp"
    DEFAULT_TIMEOUT: ClassVar[float] = 15.0

    filename: str = "code.cpp"
    compile_cmd: str | None = "g++ -std=c++17 -o {d}/a.out {f}"
    run_cmd: str = "{d}/a.out"

    def _is_syntax_error(self, exit_code: int, stdout: str, stderr: str) -> bool:
        # g++ outputs "error:" for both syntax and semantic errors during compilation
        return exit_code != 0 and "error:" in stderr

    def _is_exception(self, exit_code: int, stdout: str, stderr: str) -> bool:
        # Runtime errors like segfaults, abort, etc.
        return exit_code != 0 and "error:" not in stderr


evaluator = CppEvaluator()
