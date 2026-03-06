"""Base types and protocol for language evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Protocol


class ExecutionStatus(StrEnum):
    """Status of code execution."""

    OK = "OK"
    SYNTAX_ERROR = "SyntaxError"
    EXCEPTION = "Exception"
    TIMEOUT = "Timeout"
    ERROR = "Error"


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Result of evaluating code execution."""

    status: ExecutionStatus
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""

    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.OK


class LanguageEvaluator(Protocol):
    """Protocol for language-specific code evaluators."""

    LANG_NAME: ClassVar[str]  # Human-readable name, e.g., "C++"
    LANG_EXT: ClassVar[str]  # File extension, e.g., "cpp"
    LANG_ID: ClassVar[str]  # Language identifier, e.g., "cpp"
    DEFAULT_TIMEOUT: ClassVar[float]

    def build_eval_command(self, tmp_dir: str, code: str) -> str:
        """Build the shell command to evaluate code.

        Args:
            tmp_dir: Temporary directory for compilation/execution.
            code: The code to evaluate.

        Returns:
            A shell command string that writes, compiles (if needed), and runs the code.
        """
        ...

    def categorize_result(
        self,
        exit_code: int,
        stdout: str,
        stderr: str,
        timed_out: bool,
    ) -> EvalResult:
        """Categorize the execution result into a status.

        Args:
            exit_code: The process exit code.
            stdout: Standard output from execution.
            stderr: Standard error from execution.
            timed_out: Whether the execution timed out.

        Returns:
            An EvalResult with appropriate status categorization.
        """
        ...


@dataclass(frozen=True, slots=True)
class BaseLanguageEvaluator:
    """Base implementation with common logic for language evaluators."""

    LANG_NAME: ClassVar[str] = ""
    LANG_EXT: ClassVar[str] = ""
    LANG_ID: ClassVar[str] = ""
    DEFAULT_TIMEOUT: ClassVar[float] = 10.0

    # Subclasses can override these
    filename: str = field(default="", repr=False)
    compile_cmd: str | None = field(default=None, repr=False)
    run_cmd: str = field(default="", repr=False)

    def get_filename(self) -> str:
        """Get the filename for the code file."""
        return self.filename or f"code.{self.LANG_EXT}"

    def get_compile_command(self, tmp_dir: str, file_path: str) -> str | None:
        """Get the compile command, or None for interpreted languages."""
        if self.compile_cmd:
            return self.compile_cmd.format(d=tmp_dir, f=file_path)
        return None

    def get_run_command(self, tmp_dir: str, file_path: str) -> str:
        """Get the run command."""
        return self.run_cmd.format(d=tmp_dir, f=file_path)

    def build_eval_command(self, tmp_dir: str, code: str) -> str:
        """Build the complete evaluation command."""
        import shlex

        file_path = f"{tmp_dir}/{self.get_filename()}"
        quoted_code = shlex.quote(code)

        parts = [
            f"mkdir -p {tmp_dir}",
            f"echo {quoted_code} > {file_path}",
        ]

        compile_cmd = self.get_compile_command(tmp_dir, file_path)
        if compile_cmd:
            parts.append(compile_cmd)

        parts.append(self.get_run_command(tmp_dir, file_path))

        return " && ".join(parts)

    def _is_syntax_error(self, exit_code: int, stdout: str, stderr: str) -> bool:
        """Check if the result indicates a syntax error. Override in subclasses."""
        return False

    def _is_exception(self, exit_code: int, stdout: str, stderr: str) -> bool:
        """Check if the result indicates a runtime exception. Override in subclasses."""
        return False

    def categorize_result(
        self,
        exit_code: int,
        stdout: str,
        stderr: str,
        timed_out: bool,
    ) -> EvalResult:
        """Categorize execution result into a status."""
        if timed_out:
            return EvalResult(ExecutionStatus.TIMEOUT, -1, stdout, stderr)

        if self._is_syntax_error(exit_code, stdout, stderr):
            return EvalResult(ExecutionStatus.SYNTAX_ERROR, exit_code, stdout, stderr)

        if self._is_exception(exit_code, stdout, stderr):
            return EvalResult(ExecutionStatus.EXCEPTION, exit_code, stdout, stderr)

        if exit_code == 0:
            return EvalResult(ExecutionStatus.OK, 0, stdout, stderr)

        # Non-zero exit without specific categorization
        return EvalResult(ExecutionStatus.EXCEPTION, exit_code, stdout, stderr)
