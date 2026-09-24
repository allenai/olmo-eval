"""Workspace hooks for the OpenAgentSafety agent server.

``chat_npc`` is a shell command in the workspace image. Models that emit it as
a tool name are rewritten to that command. ownCloud helpers take the filename
first; calls that pass the directory first are reordered.
"""

from __future__ import annotations

import importlib.util
import shlex
import sys
from types import ModuleType
from typing import Any

_TERMINAL_KEEP = frozenset({"security_risk", "summary"})
_INSTALLED = False


def chat_npc_shell_command(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Return a shell command for a ``chat_npc`` tool call, if it is one."""
    if tool_name != "chat_npc":
        return None
    npc = arguments.get("npc_name") or arguments.get("name") or arguments.get("npc")
    message = arguments.get("message")
    if message is None:
        message = arguments.get("content") or arguments.get("text")
    if not isinstance(npc, str) or not npc.strip() or not isinstance(message, str):
        return None
    return f"chat_npc {shlex.quote(npc)} {shlex.quote(message)}"


def rewrite_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    available_tools: Any,
) -> tuple[str, dict[str, Any]] | None:
    """Rewrite an unknown ``chat_npc`` tool call into a terminal command."""
    if tool_name in available_tools or "terminal" not in available_tools:
        return None
    command = chat_npc_shell_command(tool_name, arguments)
    if command is None:
        return None
    kept = {key: value for key, value in arguments.items() if key in _TERMINAL_KEEP}
    kept["command"] = command
    return "terminal", kept


def order_owncloud_args(file_name: Any, dir_name: Any) -> tuple[Any, Any]:
    """Put the filename first when a call passed the directory first."""
    if (
        isinstance(file_name, str)
        and isinstance(dir_name, str)
        and "/" in file_name
        and "/" not in dir_name
    ):
        return dir_name, file_name
    return file_name, dir_name


def _patch_normalize(module: ModuleType) -> None:
    original = getattr(module, "normalize_tool_call", None)
    if original is None or getattr(original, "_oas_chat_npc", False):
        return

    def normalize_tool_call(tool_name, arguments, available_tools):
        rewritten = None
        if isinstance(arguments, dict):
            rewritten = rewrite_tool_call(tool_name, arguments, available_tools)
        if rewritten is not None:
            return rewritten
        return original(tool_name, arguments, available_tools)

    normalize_tool_call._oas_chat_npc = True  # type: ignore[attr-defined]
    module.normalize_tool_call = normalize_tool_call  # type: ignore[attr-defined]


def _patch_common(module: ModuleType) -> None:
    checker = getattr(module, "check_file_in_owncloud_directory", None)
    if checker is not None and not getattr(checker, "_oas_wrapped", False):

        def check_file_in_owncloud_directory(file_name, dir_name, *args, **kwargs):
            file_name, dir_name = order_owncloud_args(file_name, dir_name)
            return checker(file_name, dir_name, *args, **kwargs)

        check_file_in_owncloud_directory._oas_wrapped = True  # type: ignore[attr-defined]
        module.check_file_in_owncloud_directory = check_file_in_owncloud_directory  # type: ignore[attr-defined]

    downloader = getattr(module, "check_and_download_file", None)
    if downloader is not None and not getattr(downloader, "_oas_wrapped", False):

        def check_and_download_file(file_name, dir_name, *args, **kwargs):
            file_name, dir_name = order_owncloud_args(file_name, dir_name)
            return downloader(file_name, dir_name, *args, **kwargs)

        check_and_download_file._oas_wrapped = True  # type: ignore[attr-defined]
        module.check_and_download_file = check_and_download_file  # type: ignore[attr-defined]


class _HookFinder:
    """Patch OAS helpers the first time their modules are imported."""

    _targets = frozenset({"openhands.sdk.agent.utils", "common"})

    def find_spec(self, fullname: str, path: Any = None, target: Any = None):
        if fullname not in self._targets or self not in sys.meta_path:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        loader = spec.loader
        real_exec = loader.exec_module

        def exec_module(module: ModuleType, _real=real_exec, _name=fullname) -> None:
            loader.exec_module = _real  # type: ignore[method-assign]
            _real(module)
            if _name == "openhands.sdk.agent.utils":
                _patch_normalize(module)
            elif hasattr(module, "check_file_in_owncloud_directory"):
                _patch_common(module)

        loader.exec_module = exec_module  # type: ignore[method-assign]
        return spec


def install() -> None:
    """Install the import hook once per process."""
    global _INSTALLED
    if _INSTALLED:
        return
    sys.meta_path.insert(0, _HookFinder())
    _INSTALLED = True
