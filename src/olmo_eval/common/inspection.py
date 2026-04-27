"""Utilities for inspecting and pretty-printing evaluation objects.

This module provides reusable functions for displaying Instance objects
and other dataclasses using Rich panels and tables.
"""

from __future__ import annotations

import json
import re
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from olmo_eval.common.console import console as shared_console

if TYPE_CHECKING:
    from olmo_eval.common.types import Instance, LMRequest, Response

# Default truncation limits
DEFAULT_STRING_HALF = 100  # Show first 100 and last 100 chars for strings
DEFAULT_TOKEN_HALF = 50  # Show first 50 and last 50 tokens


def truncate_string(value: str, half: int = DEFAULT_STRING_HALF) -> tuple[str, bool]:
    """Truncate a long string showing first N and last N characters.

    Args:
        value: The string to potentially truncate.
        half: Number of characters to show from each end.

    Returns:
        Tuple of (truncated_string, was_truncated).
    """
    max_len = half * 2
    if len(value) <= max_len:
        return value, False

    first = value[:half]
    last = value[-half:]
    return f"{first}\n\n... ({len(value) - max_len} chars omitted) ...\n\n{last}", True


def format_value(
    value: Any,
    max_string_length: int = 0,
    max_list_items: int = 5,
) -> str:
    """Format any value for display with appropriate truncation.

    Args:
        value: The value to format.
        max_string_length: Maximum length for string values before truncation.
            When truncating, shows first half and last half of the limit.
        max_list_items: Maximum number of items to show for lists/tuples.

    Returns:
        A formatted string representation of the value.
    """
    if value is None:
        return "[dim]None[/dim]"

    if isinstance(value, str):
        if max_string_length > 0 and len(value) > max_string_length:
            half = max_string_length // 2
            truncated, _ = truncate_string(value, half=half)
            return f'"{truncated}" [dim]({len(value)} chars)[/dim]'
        return f'"{value}"'

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "[]" if isinstance(value, list) else "()"

        # For short lists of simple items, show inline
        if len(value) <= max_list_items:
            items = [_format_simple_value(v, max_string_length // 2) for v in value]
            sep = ", "
            brackets = ("(", ")") if isinstance(value, tuple) else ("[", "]")
            return f"{brackets[0]}{sep.join(items)}{brackets[1]}"

        # For longer lists, truncate
        items = [_format_simple_value(v, max_string_length // 2) for v in value[:max_list_items]]
        brackets = ("(", ")") if isinstance(value, tuple) else ("[", "]")
        return f"{brackets[0]}{', '.join(items)}, ...{brackets[1]} [dim]({len(value)} items)[/dim]"

    if isinstance(value, dict):
        if len(value) == 0:
            return "{}"

        # Pretty print as JSON
        import json

        return json.dumps(value, indent=2, ensure_ascii=False)

    if is_dataclass(value) and not isinstance(value, type):
        return f"<{type(value).__name__}>"

    # Fallback: use repr with truncation
    repr_str = repr(value)
    if max_string_length > 0 and len(repr_str) > max_string_length:
        return repr_str[:max_string_length] + "..."
    return repr_str


def _format_simple_value(value: Any, max_length: int = 0) -> str:
    """Format a value for inline display (no rich markup).

    Args:
        value: The value to format.
        max_length: Maximum length for string values (0 for no limit).
            When truncating, shows first half and last half.
    """
    if value is None:
        return "None"
    if isinstance(value, str):
        if max_length > 0 and len(value) > max_length:
            half = max_length // 2
            truncated, _ = truncate_string(value, half=half)
            return f'"{truncated}"'
        return f'"{value}"'
    if isinstance(value, (bool, int, float)):
        return str(value)
    if isinstance(value, dict):
        return "{...}" if value else "{}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]" if isinstance(value, list) else "()"
        return "[...]" if isinstance(value, list) else "(...)"
    repr_str = repr(value)
    if max_length > 0 and len(repr_str) > max_length:
        half = max_length // 2
        truncated, _ = truncate_string(repr_str, half=half)
        return truncated
    return repr_str


def inspect_object(
    obj: Any,
    *,
    console: Console | None = None,
    title: str | None = None,
    max_string_length: int = 0,
    show_none: bool = True,
) -> None:
    """Pretty-print any dataclass or object using Rich panels/tables.

    Args:
        obj: The object to inspect (typically a dataclass).
        console: Rich Console to print to. Uses shared console if not provided.
        title: Optional title for the panel.
        max_string_length: Maximum length for string values.
        show_none: Whether to show fields with None values.
    """
    if console is None:
        console = shared_console

    if not is_dataclass(obj) or isinstance(obj, type):
        console.print(f"[dim]Not a dataclass instance: {type(obj).__name__}[/dim]")
        return

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    for field in fields(obj):
        value = getattr(obj, field.name)
        if value is None and not show_none:
            continue
        formatted = format_value(value, max_string_length=max_string_length)
        table.add_row(field.name, formatted)

    panel_title = title or f"[bold]{type(obj).__name__}[/bold]"
    console.print(Panel(table, title=panel_title, border_style="cyan"))


def inspect_instance(
    instance: Instance,
    *,
    console: Console | None = None,
    task_name: str | None = None,
    native_id: str | None = None,
    max_string_length: int = 0,
) -> None:
    """Pretty-print an Instance object with task-aware formatting.

    Args:
        instance: The Instance to inspect.
        console: Rich Console to print to. Uses shared console if not provided.
        task_name: Optional task name for the panel title.
        native_id: Optional native_id for the panel title. If not provided,
            attempts to extract from instance.metadata["id"].
        max_string_length: Maximum length for string values.
    """
    # Try to get native_id from metadata if not provided
    if native_id is None:
        native_id = instance.metadata.get("id")
    import json

    if console is None:
        console = shared_console

    renderables: list[Any] = []

    # Use provided limit or default (first 100 + last 100 chars)
    half = max_string_length // 2 if max_string_length > 0 else DEFAULT_STRING_HALF
    limit = max_string_length if max_string_length > 0 else half * 2

    def add_field(name: str, value: Any) -> None:
        """Add a field with its label and value."""
        if renderables:
            renderables.append(Text(""))  # Blank line between fields

        renderables.append(Text(f"{name}:", style="bold cyan"))

        if isinstance(value, dict):
            # Pretty print dicts as JSON with word wrapping
            json_str = json.dumps(value, indent=2, ensure_ascii=False)
            renderables.append(Syntax(json_str, "json", theme="ansi_dark", word_wrap=True))
        elif isinstance(value, str):
            if len(value) > limit:
                truncated, was_truncated = truncate_string(value, half=half)
                if was_truncated:
                    renderables.append(Text(f"{truncated}\n({len(value)} chars total)"))
                else:
                    renderables.append(Text(value))
            else:
                renderables.append(Text(value))
        elif isinstance(value, (list, tuple)):
            # Format lists/tuples as JSON
            json_str = json.dumps(list(value), indent=2, ensure_ascii=False)
            renderables.append(Syntax(json_str, "json", theme="ansi_dark", word_wrap=True))
        else:
            renderables.append(Text(str(value)))

    # Core fields first
    add_field("question", instance.question)

    if instance.gold_answer is not None:
        add_field("gold_answer", instance.gold_answer)

    if instance.choices is not None:
        add_field("choices", instance.choices)

    if instance.metadata:
        add_field("metadata", instance.metadata)

    # Tool-related fields (only show if present)
    if instance.tools is not None:
        tool_names = [t.name for t in instance.tools] if instance.tools else []
        add_field("tools", tool_names)

    if instance.expected_tool_calls is not None:
        add_field("expected_tool_calls", list(instance.expected_tool_calls))

    if instance.should_abstain is not None:
        add_field("should_abstain", instance.should_abstain)

    if instance.required_trajectory is not None:
        add_field("required_trajectory", list(instance.required_trajectory))

    if instance.initial_state is not None:
        add_field("initial_state", instance.initial_state)

    if instance.expected_final_state is not None:
        add_field("expected_final_state", instance.expected_final_state)

    # Build panel title
    if task_name and native_id is not None:
        title = f"[bold]Instance #{native_id}[/bold] ({task_name})"
    elif task_name:
        title = f"[bold]Instance[/bold] ({task_name})"
    elif native_id is not None:
        title = f"[bold]Instance #{native_id}[/bold]"
    else:
        title = "[bold]Instance[/bold]"

    console.print(Panel(Group(*renderables), title=title, border_style="cyan"))


def inspect_request(
    request: LMRequest,
    *,
    console: Console | None = None,
    task_name: str | None = None,
    native_id: str | None = None,
    max_string_length: int = 0,
) -> None:
    """Pretty-print an LMRequest object.

    Args:
        request: The LMRequest to inspect.
        console: Rich Console to print to. Uses shared console if not provided.
        task_name: Optional task name for the panel title.
        native_id: Optional native_id for the panel title.
        max_string_length: Maximum length for string values.
    """
    if console is None:
        console = shared_console

    renderables: list[Any] = []

    def add_field(name: str, value: str) -> None:
        """Add a field with its label and value."""
        if renderables:
            renderables.append(Text(""))  # Blank line between fields
        renderables.append(Text(f"{name}:", style="bold cyan"))
        renderables.append(Text(value))

    add_field("request_type", request.request_type.name)

    # Use provided limit or default (first 100 + last 100 chars)
    half = max_string_length // 2 if max_string_length > 0 else DEFAULT_STRING_HALF
    limit = max_string_length if max_string_length > 0 else half * 2

    if request.messages:
        # Format messages nicely
        msg_strs = []
        for msg in request.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > limit:
                content, _ = truncate_string(content, half=half)
            msg_strs.append(f"[{role}]: {content}")
        add_field("messages", "\n".join(msg_strs))

    if request.prompt:
        prompt_val = request.prompt
        if len(prompt_val) > limit:
            original_len = len(prompt_val)
            prompt_val, _ = truncate_string(prompt_val, half=half)
            prompt_val = f"{prompt_val}\n({original_len} chars total)"
        add_field("prompt", prompt_val)

    if request.continuations:
        # Show each continuation on its own line
        cont_strs = []
        for c in request.continuations:
            if len(c) > limit:
                truncated, _ = truncate_string(c, half=half)
                cont_strs.append(truncated)
            else:
                cont_strs.append(c)
        add_field("continuations", "\n".join(cont_strs))

    # Agent-specific fields
    if request.system_prompt:
        prompt_val = request.system_prompt
        if len(prompt_val) > limit:
            original_len = len(prompt_val)
            prompt_val, _ = truncate_string(prompt_val, half=half)
            prompt_val = f"{prompt_val}\n({original_len} chars total)"
        add_field("system_prompt", prompt_val)

    if request.tools:
        tool_strs = []
        for tool in request.tools:
            tool_strs.append(f"• {tool.name}: {tool.description}")
        add_field("tools", "\n".join(tool_strs))

    # Build panel title
    if task_name and native_id is not None:
        panel_title = f"[bold]Request #{native_id}[/bold] ({task_name})"
    elif task_name:
        panel_title = f"[bold]Request[/bold] ({task_name})"
    elif native_id is not None:
        panel_title = f"[bold]Request #{native_id}[/bold]"
    else:
        panel_title = "[bold]Request[/bold]"

    console.print(Panel(Group(*renderables), title=panel_title, border_style="magenta"))


def instance_to_dict(instance: Instance) -> dict[str, Any]:
    """Convert an Instance to a JSON-serializable dictionary.

    Args:
        instance: The Instance to convert.

    Returns:
        A dictionary representation of the instance.
    """
    result: dict[str, Any] = {
        "question": instance.question,
    }

    if instance.gold_answer is not None:
        result["gold_answer"] = instance.gold_answer

    if instance.choices is not None:
        result["choices"] = list(instance.choices)

    if instance.metadata:
        result["metadata"] = instance.metadata

    if instance.tools is not None:
        result["tools"] = [{"name": t.name, "description": t.description} for t in instance.tools]

    if instance.expected_tool_calls is not None:
        result["expected_tool_calls"] = list(instance.expected_tool_calls)

    if instance.should_abstain is not None:
        result["should_abstain"] = instance.should_abstain

    if instance.required_trajectory is not None:
        result["required_trajectory"] = list(instance.required_trajectory)

    if instance.initial_state is not None:
        result["initial_state"] = instance.initial_state

    if instance.expected_final_state is not None:
        result["expected_final_state"] = instance.expected_final_state

    return result


def _get_isolated_venv_python() -> str | None:
    """Get the Python executable for the isolated vLLM venv if available."""
    import os

    # Check VLLM_PYTHON env var first
    vllm_python = os.environ.get("VLLM_PYTHON")
    if vllm_python and os.path.isfile(vllm_python):
        return vllm_python

    # Check standard location
    standard_path = "/opt/vllm-venv/bin/python"
    if os.path.isfile(standard_path):
        return standard_path

    return None


_TOKENIZER_SERVER_SCRIPT = """
import json
import sys
import os

# Silence HuggingFace downloads and warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import AutoTokenizer

tokenizer_name = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

# Signal ready
print(json.dumps({"status": "ready", "special_ids": list(tokenizer.all_special_ids)}), flush=True)

# Process requests
for line in sys.stdin:
    try:
        req = json.loads(line.strip())
        cmd = req.get("cmd")

        if cmd == "encode":
            add_special = req.get("add_special_tokens", True)
            tokens = tokenizer.encode(req["text"], add_special_tokens=add_special)
            print(json.dumps({"tokens": tokens}), flush=True)

        elif cmd == "decode":
            skip_special = req.get("skip_special_tokens", False)
            text = tokenizer.decode(req["token_ids"], skip_special_tokens=skip_special)
            print(json.dumps({"text": text}), flush=True)

        elif cmd == "apply_chat_template":
            result = tokenizer.apply_chat_template(
                req["messages"],
                tokenize=req.get("tokenize", False),
                add_generation_prompt=req.get("add_generation_prompt", True),
            )
            if isinstance(result, list):
                print(json.dumps({"tokens": result}), flush=True)
            else:
                print(json.dumps({"text": result}), flush=True)

        elif cmd == "quit":
            break

        else:
            print(json.dumps({"error": f"Unknown command: {cmd}"}), flush=True)

    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)
"""


class SubprocessTokenizer:
    """Tokenizer that runs in an isolated venv via a persistent subprocess.

    Uses a long-running subprocess that loads the tokenizer once and handles
    requests via JSON-line protocol over stdin/stdout.

    This class supports context manager protocol for proper resource cleanup:

        with SubprocessTokenizer(name, path) as tokenizer:
            tokens = tokenizer.encode("hello")

    Or call close() explicitly when done.
    """

    def __init__(self, tokenizer_name: str, python_path: str) -> None:
        import subprocess

        self._tokenizer_name = tokenizer_name
        self._python_path = python_path
        self._special_ids: set[int] = set()
        self._closed = False

        # Start persistent subprocess.
        # Redirect stderr to DEVNULL to prevent pipe buffer filling up and blocking.
        # Errors are communicated via JSON over stdout.
        self._proc = subprocess.Popen(
            [python_path, "-c", _TOKENIZER_SERVER_SCRIPT, tokenizer_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Wait for ready signal with timeout to prevent hanging on startup failures
        import select
        import sys

        timeout_seconds = 60
        if sys.platform != "win32" and hasattr(select, "select"):
            # Use select for timeout on Unix
            ready, _, _ = select.select([self._proc.stdout], [], [], timeout_seconds)
            if not ready:
                self._proc.kill()
                self._proc.wait()
                raise RuntimeError(
                    f"Tokenizer subprocess timed out after {timeout_seconds}s "
                    "waiting for ready signal"
                )

        ready_line = self._proc.stdout.readline()  # type: ignore[ty:unresolved-attribute]
        if not ready_line:
            self._proc.kill()
            self._proc.wait()
            raise RuntimeError("Tokenizer subprocess failed to start (no output)")

        ready = json.loads(ready_line)
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError(f"Tokenizer subprocess error: {ready.get('error', 'unknown')}")

        self._special_ids = set(ready.get("special_ids", []))

    def _request(self, req: dict[str, Any]) -> dict[str, Any]:
        """Send a request to the subprocess and get response."""
        if self._closed:
            raise RuntimeError("Tokenizer has been closed")
        if self._proc.poll() is not None:
            raise RuntimeError("Tokenizer subprocess has terminated")

        self._proc.stdin.write(json.dumps(req) + "\n")  # type: ignore[ty:unresolved-attribute]
        self._proc.stdin.flush()  # type: ignore[ty:unresolved-attribute]

        response_line = self._proc.stdout.readline()  # type: ignore[ty:unresolved-attribute]
        if not response_line:
            raise RuntimeError("Tokenizer subprocess returned empty response")

        response = json.loads(response_line)
        if "error" in response:
            raise RuntimeError(f"Tokenizer error: {response['error']}")

        return response

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        req = {"cmd": "encode", "text": text, "add_special_tokens": add_special_tokens}
        return self._request(req)["tokens"]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text."""
        req = {"cmd": "decode", "token_ids": token_ids, "skip_special_tokens": skip_special_tokens}
        return self._request(req)["text"]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str | list[int]:
        """Apply chat template."""
        resp = self._request(
            {
                "cmd": "apply_chat_template",
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return resp.get("tokens") or resp.get("text", "")

    @property
    def all_special_ids(self) -> set[int]:
        """Get all special token IDs."""
        return self._special_ids

    def close(self) -> None:
        """Close the subprocess and release resources."""
        if self._closed:
            return
        self._closed = True

        if not hasattr(self, "_proc"):
            return

        if self._proc.poll() is None:
            try:
                # Send quit command for graceful shutdown
                self._proc.stdin.write('{"cmd": "quit"}\n')  # type: ignore[ty:unresolved-attribute]
                self._proc.stdin.flush()  # type: ignore[ty:unresolved-attribute]
                self._proc.wait(timeout=2)
            except Exception:
                # Force kill if graceful shutdown fails
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=1)
                except Exception:
                    pass

    def __enter__(self) -> SubprocessTokenizer:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close subprocess."""
        self.close()

    def __del__(self) -> None:
        """Clean up subprocess (fallback if close() not called)."""
        import contextlib

        with contextlib.suppress(Exception):
            self.close()


def _load_tokenizer_from_isolated_venv(tokenizer_name: str, logger: Any) -> Any | None:
    """Try to load a tokenizer using the isolated vLLM venv.

    Args:
        tokenizer_name: HuggingFace tokenizer name or path.
        logger: Logger for warnings/errors.

    Returns:
        A SubprocessTokenizer if isolated venv is available, None otherwise.
    """
    python_path = _get_isolated_venv_python()
    if python_path is None:
        logger.warning(
            "transformers not available and no isolated venv found. "
            "Token inspection will be skipped."
        )
        return None

    try:
        # Verify the isolated venv has transformers
        import subprocess

        result = subprocess.run(
            [python_path, "-c", "import transformers"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning(
                f"Isolated venv at {python_path} does not have transformers installed. "
                "Token inspection will be skipped."
            )
            return None

        logger.info(f"Using isolated venv tokenizer via {python_path}")
        return SubprocessTokenizer(tokenizer_name, python_path)
    except Exception as e:
        logger.warning(f"Failed to initialize isolated venv tokenizer: {e}")
        return None


def load_tokenizer(tokenizer_name: str, trust_remote_code: bool = True) -> Any:
    """Load a tokenizer by name without loading the model.

    Args:
        tokenizer_name: HuggingFace tokenizer name or path.
        trust_remote_code: Whether to trust remote code for custom tokenizers.

    Returns:
        A HuggingFace tokenizer instance.

    Raises:
        ImportError: If transformers is not installed.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers is required for tokenizer loading. Install with: pip install transformers"
        ) from e

    return AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
    )


def format_with_chat_template(
    request: LMRequest,
    tokenizer: Any,
    add_generation_prompt: bool = True,
) -> str:
    """Apply chat template to request messages, return formatted string.

    Args:
        request: The LMRequest to format.
        tokenizer: A HuggingFace tokenizer with a chat template.
        add_generation_prompt: Whether to add generation prompt at the end.

    Returns:
        The formatted prompt string after applying the chat template.

    Raises:
        ValueError: If the request has no messages or the tokenizer has no chat template.
    """
    from olmo_eval.common.types import RequestType

    # For COMPLETION requests, just return the prompt directly
    if request.request_type == RequestType.COMPLETION:
        return request.prompt or ""

    # For LOGLIKELIHOOD requests, return prompt + first continuation
    if request.request_type == RequestType.LOGLIKELIHOOD:
        prompt = request.prompt or ""
        if request.continuations:
            # Show the prompt followed by the first continuation
            continuation_text = request.continuations[0] if request.continuations else ""
            return prompt + continuation_text
        return prompt

    # CHAT requests - apply chat template
    if not request.messages:
        raise ValueError("Request has no messages to format")

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support chat templates")

    return tokenizer.apply_chat_template(
        request.messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def tokenize_request(
    request: LMRequest,
    tokenizer: Any,
    apply_chat_template: bool = True,
) -> list[int]:
    """Tokenize a request and return token IDs.

    Args:
        request: The LMRequest to tokenize.
        tokenizer: A HuggingFace tokenizer.
        apply_chat_template: Whether to apply chat template before tokenizing.

    Returns:
        List of token IDs.
    """
    from olmo_eval.common.types import RequestType

    if request.request_type == RequestType.COMPLETION or not apply_chat_template:
        # For COMPLETION or if explicitly not applying template, tokenize prompt directly
        text = request.prompt or ""
        return tokenizer.encode(text, add_special_tokens=True)

    if request.request_type == RequestType.LOGLIKELIHOOD:
        # For LOGLIKELIHOOD, tokenize prompt + first continuation
        prompt = request.prompt or ""
        if request.continuations:
            continuation_text = request.continuations[0] if request.continuations else ""
            text = prompt + continuation_text
        else:
            text = prompt
        return tokenizer.encode(text, add_special_tokens=True)

    if request.messages and hasattr(tokenizer, "apply_chat_template"):
        # Use apply_chat_template with tokenize=True to get token IDs
        return tokenizer.apply_chat_template(
            request.messages,
            tokenize=True,
            add_generation_prompt=True,
        )

    # Fallback: tokenize the prompt
    text = request.prompt or ""
    return tokenizer.encode(text, add_special_tokens=True)


def inspect_formatted_request(
    formatted_prompt: str,
    *,
    console: Console | None = None,
    task_name: str | None = None,
    native_id: str | None = None,
    max_chars: int = 2000,
) -> None:
    """Pretty-print formatted prompt with special token highlighting.

    Args:
        formatted_prompt: The formatted prompt string to display.
        console: Rich Console to print to. Uses shared console if not provided.
        task_name: Optional task name for the panel title.
        native_id: Optional native_id for the panel title.
        max_chars: Max characters to display (0 for no limit, default 2000).
            When truncating, shows first half and last half.
    """
    if console is None:
        console = shared_console

    # Pattern to match common special tokens
    special_token_pattern = re.compile(
        r"(<\|[^|>]+\|>|<s>|</s>|<unk>|<pad>|<mask>|\[CLS\]|\[SEP\]|\[PAD\]|\[MASK\])"
    )

    def style_special_tokens(s: str) -> Text:
        """Apply styling to special tokens in a string."""
        result = Text()
        last_end = 0
        for match in special_token_pattern.finditer(s):
            if match.start() > last_end:
                result.append(s[last_end : match.start()])
            result.append(match.group(), style="bold cyan")
            last_end = match.end()
        if last_end < len(s):
            result.append(s[last_end:])
        return result

    # Build styled text with special token highlighting
    text = Text()

    if max_chars > 0 and len(formatted_prompt) > max_chars:
        # Show first half and last half
        half = max_chars // 2
        first_part = formatted_prompt[:half]
        last_part = formatted_prompt[-half:]
        omitted = len(formatted_prompt) - max_chars

        text.append_text(style_special_tokens(first_part))
        text.append(f"\n\n... ({omitted} chars omitted) ...\n\n", style="dim")
        text.append_text(style_special_tokens(last_part))
    else:
        text.append_text(style_special_tokens(formatted_prompt))

    # Add character count
    text.append(f"\n\n({len(formatted_prompt)} characters total)", style="dim")

    # Build panel title
    if task_name and native_id is not None:
        panel_title = f"[bold]Formatted Prompt #{native_id}[/bold] ({task_name})"
    elif task_name:
        panel_title = f"[bold]Formatted Prompt[/bold] ({task_name})"
    elif native_id is not None:
        panel_title = f"[bold]Formatted Prompt #{native_id}[/bold]"
    else:
        panel_title = "[bold]Formatted Prompt[/bold]"

    console.print(Panel(text, title=panel_title, border_style="green"))


def inspect_tokens(
    tokens: list[int],
    tokenizer: Any,
    *,
    console: Console | None = None,
    task_name: str | None = None,
    native_id: str | None = None,
    max_tokens: int = DEFAULT_TOKEN_HALF * 2,
    show_decoded: bool = True,
) -> None:
    """Pretty-print token IDs with decoded values, highlighting special tokens.

    Args:
        tokens: List of token IDs to display.
        tokenizer: A HuggingFace tokenizer for decoding.
        console: Rich Console to print to. Uses shared console if not provided.
        task_name: Optional task name for the panel title.
        native_id: Optional native_id for the panel title.
        max_tokens: Max tokens to display (0 for no limit, default 100).
            When truncating, shows first half and last half.
        show_decoded: Whether to show decoded token values.
    """
    if console is None:
        console = shared_console

    # Get special token IDs
    special_ids: set[int] = set()
    if hasattr(tokenizer, "all_special_ids"):
        special_ids = set(tokenizer.all_special_ids)

    def format_token(token_id: int) -> str:
        """Format a single token for display."""
        is_special = token_id in special_ids

        if show_decoded:
            try:
                decoded = tokenizer.decode([token_id])
                # Escape and format the decoded value
                decoded_display = (
                    repr(decoded) if decoded.strip() != decoded or not decoded else decoded
                )
            except Exception:
                decoded_display = "<decode error>"

            if is_special:
                return f"  [bold cyan][{token_id}][/bold cyan] {decoded_display}  [cyan]◆[/cyan]"
            else:
                return f"  [{token_id}] {decoded_display}"
        else:
            if is_special:
                return f"  [bold cyan][{token_id}][/bold cyan]  [dim cyan]◆[/dim cyan]"
            else:
                return f"  [{token_id}]"

    # Build display
    lines: list[str] = []
    lines.append(f"[bold]{len(tokens)} tokens[/bold]")
    lines.append("─" * 60)

    if max_tokens > 0 and len(tokens) > max_tokens:
        # Show first half and last half of max_tokens
        half = max_tokens // 2
        first_tokens = tokens[:half]
        last_tokens = tokens[-half:]

        # Add first tokens
        for token_id in first_tokens:
            lines.append(format_token(token_id))

        # Add ellipsis indicator
        omitted = len(tokens) - max_tokens
        lines.append(f"\n  [dim]... ({omitted} tokens omitted) ...[/dim]\n")

        # Add last tokens
        for token_id in last_tokens:
            lines.append(format_token(token_id))
    else:
        # Show all tokens
        for token_id in tokens:
            lines.append(format_token(token_id))

    # Build panel title
    if task_name and native_id is not None:
        panel_title = f"[bold]Tokens #{native_id}[/bold] ({task_name})"
    elif task_name:
        panel_title = f"[bold]Tokens[/bold] ({task_name})"
    elif native_id is not None:
        panel_title = f"[bold]Tokens #{native_id}[/bold]"
    else:
        panel_title = "[bold]Tokens[/bold]"

    console.print(Panel("\n".join(lines), title=panel_title, border_style="yellow"))


def formatted_request_to_dict(
    formatted_prompt: str,
    tokens: list[int],
    tokenizer: Any,
) -> dict[str, Any]:
    """Convert formatted request and tokens to JSON-serializable dict.

    Args:
        formatted_prompt: The formatted prompt string.
        tokens: List of token IDs.
        tokenizer: A HuggingFace tokenizer for getting special token info.

    Returns:
        Dictionary with formatted prompt, tokens, and metadata.
    """
    # Get special token IDs
    special_ids: set[int] = set()
    if hasattr(tokenizer, "all_special_ids"):
        special_ids = set(tokenizer.all_special_ids)

    # Build token details
    token_details = []
    for token_id in tokens:
        is_special = token_id in special_ids
        try:
            decoded = tokenizer.decode([token_id])
        except Exception:
            decoded = None

        token_details.append(
            {
                "id": token_id,
                "decoded": decoded,
                "is_special": is_special,
            }
        )

    return {
        "_formatted_prompt": formatted_prompt,
        "_token_ids": tokens,
        "_token_count": len(tokens),
        "_token_details": token_details,
    }


def inspect_response(
    response: Response,
    *,
    console: Console | None = None,
    task_name: str | None = None,
    native_id: str | None = None,
    max_string_length: int = 0,
) -> None:
    """Pretty-print a Response object showing instance, outputs, and scores.

    Args:
        response: The Response to inspect.
        console: Rich Console to print to. Uses shared console if not provided.
        task_name: Optional task name for the panel title.
        native_id: Optional native_id for the panel title. If not provided,
            attempts to extract from response.instance.metadata["id"].
        max_string_length: Maximum length for string values.
    """
    # Try to get native_id from instance metadata if not provided
    if native_id is None:
        native_id = response.instance.metadata.get("id")
    import json

    if console is None:
        console = shared_console

    renderables: list[Any] = []

    # Use provided limit or default (first 100 + last 100 chars)
    half = max_string_length // 2 if max_string_length > 0 else DEFAULT_STRING_HALF
    limit = max_string_length if max_string_length > 0 else half * 2

    def add_field(name: str, value: Any) -> None:
        """Add a field with its label and value."""
        if renderables:
            renderables.append(Text(""))  # Blank line between fields

        renderables.append(Text(f"{name}:", style="bold cyan"))

        if isinstance(value, dict):
            # Pretty print dicts as JSON with word wrapping
            json_str = json.dumps(value, indent=2, ensure_ascii=False)
            renderables.append(Syntax(json_str, "json", theme="ansi_dark", word_wrap=True))
        elif isinstance(value, str):
            if len(value) > limit:
                truncated, was_truncated = truncate_string(value, half=half)
                if was_truncated:
                    renderables.append(Text(f"{truncated}\n({len(value)} chars total)"))
                else:
                    renderables.append(Text(value))
            else:
                renderables.append(Text(value))
        elif isinstance(value, (list, tuple)):
            # Format lists/tuples as JSON
            json_str = json.dumps(list(value), indent=2, ensure_ascii=False, default=str)
            renderables.append(Syntax(json_str, "json", theme="ansi_dark", word_wrap=True))
        else:
            renderables.append(Text(str(value)))

    # Instance summary (question and gold_answer)
    add_field("question", response.instance.question)
    if response.instance.gold_answer is not None:
        add_field("gold_answer", response.instance.gold_answer)

    # Model outputs
    for i, output in enumerate(response.outputs):
        output_label = f"output[{i}]" if len(response.outputs) > 1 else "output"
        add_field(output_label, output.text)

        # Show extracted answer if different from text
        if output.extracted_answer is not None and output.extracted_answer != output.text:
            add_field("extracted_answer", str(output.extracted_answer))

        # Show tool calls if present
        if output.tool_calls:
            tool_calls_data = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in output.tool_calls
            ]
            add_field("tool_calls", tool_calls_data)

    # Scores
    if response.scores:
        add_field("scores", response.scores)

    # Build panel title
    if task_name and native_id is not None:
        title = f"[bold]Response #{native_id}[/bold] ({task_name})"
    elif task_name:
        title = f"[bold]Response[/bold] ({task_name})"
    elif native_id is not None:
        title = f"[bold]Response #{native_id}[/bold]"
    else:
        title = "[bold]Response[/bold]"

    console.print(Panel(Group(*renderables), title=title, border_style="green"))


def inspect_task_instances(
    trackers: dict[str, Any],
    provider_config: Any,
    *,
    inspect_instance_flag: bool = False,
    inspect_formatted: bool = False,
    inspect_tokens_flag: bool = False,
    inspect_request_flag: bool = False,
    console: Console | None = None,
) -> None:
    """Inspect first instance of each task in the trackers.

    Args:
        trackers: Dict mapping task spec to TaskTracker objects.
        provider_config: Provider config with tokenizer/model info.
        inspect_instance_flag: Show raw instance data.
        inspect_formatted: Show formatted prompt with chat template.
        inspect_tokens_flag: Show tokenized representation.
        inspect_request_flag: Show the LMRequest object.
        console: Rich Console to print to.
    """
    from olmo_eval.common.logging import get_logger

    logger = get_logger(__name__)

    if console is None:
        console = shared_console

    tokenizer = None
    if inspect_formatted or inspect_tokens_flag:
        tokenizer_name = provider_config.tokenizer or provider_config.model
        try:
            tokenizer = load_tokenizer(tokenizer_name)
        except ImportError:
            # transformers not available - try isolated venv
            tokenizer = _load_tokenizer_from_isolated_venv(tokenizer_name, logger)
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")

    for spec, tracker in trackers.items():
        if tracker.task and not tracker.error:
            first_instance = next(iter(tracker.task.instances), None)
            if first_instance:
                native_id = first_instance.metadata.get("id", "0")

                if inspect_instance_flag:
                    console.print()
                    inspect_instance(
                        first_instance, console=console, task_name=spec, native_id=native_id
                    )

                if inspect_request_flag or (
                    tokenizer and (inspect_formatted or inspect_tokens_flag)
                ):
                    request = tracker.task.format_request(first_instance)

                    if inspect_request_flag:
                        inspect_request(
                            request, console=console, task_name=spec, native_id=native_id
                        )

                    if tokenizer and inspect_formatted:
                        try:
                            formatted_prompt = format_with_chat_template(request, tokenizer)
                            inspect_formatted_request(
                                formatted_prompt,
                                console=console,
                                task_name=spec,
                                native_id=native_id,
                            )
                        except Exception as e:
                            logger.error(f"Error formatting request: {e}")

                    if tokenizer and inspect_tokens_flag:
                        try:
                            tokens = tokenize_request(request, tokenizer)
                            inspect_tokens(
                                tokens,
                                tokenizer,
                                console=console,
                                task_name=spec,
                                native_id=native_id,
                            )
                        except Exception as e:
                            logger.error(f"Error tokenizing request: {e}")


__all__ = [
    "truncate_string",
    "format_value",
    "inspect_object",
    "inspect_instance",
    "inspect_request",
    "inspect_response",
    "instance_to_dict",
    "load_tokenizer",
    "format_with_chat_template",
    "tokenize_request",
    "inspect_formatted_request",
    "inspect_tokens",
    "formatted_request_to_dict",
    "inspect_task_instances",
]
