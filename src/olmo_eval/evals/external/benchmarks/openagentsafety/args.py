"""Arguments for the OpenAgentSafety external evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "mgulavani/openagentsafety_full_updated_v3"
DEFAULT_SPLIT = "train"
DEFAULT_CRITIC = "pass"
KNOWN_CRITICS = ("pass", "finish_with_patch", "empty_patch_critic")


def _parse_optional(data: dict[str, Any], key: str, type_fn: type) -> Any:
    value = data.get(key)
    return type_fn(value) if value is not None else None


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _parse_select(value: Any) -> str | list[str] | None:
    """Parse ``select`` as a file path or a list of instance IDs."""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        return str(value)
    text = value.strip()
    if not text:
        return None
    path = Path(text)
    if path.exists():
        return text
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return text


@dataclass
class OpenAgentSafetyArgs:
    """Arguments for openagentsafety evaluation."""

    dataset: str = DEFAULT_DATASET
    split: str = DEFAULT_SPLIT
    n_limit: int | None = None
    num_workers: int = 1
    critic: str = DEFAULT_CRITIC
    select: str | list[str] | None = None
    prompt_path: str | None = None
    repo_path: str | None = None
    repo_ref: str | None = None
    max_iterations: int = 500
    n_critic_runs: int = 1
    max_retries: int = 3
    tool_preset: str = "default"
    enable_delegation: bool = False
    note: str = "olmo-eval"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenAgentSafetyArgs:
        critic = data.get("critic", DEFAULT_CRITIC)
        if critic not in KNOWN_CRITICS:
            known = ", ".join(KNOWN_CRITICS)
            raise ValueError(f"Unknown critic {critic!r}. Available: {known}")

        return cls(
            dataset=data.get("dataset", DEFAULT_DATASET),
            split=data.get("split", DEFAULT_SPLIT),
            n_limit=_parse_optional(data, "n_limit", int),
            num_workers=int(data.get("num_workers", 1)),
            critic=critic,
            select=_parse_select(data.get("select")),
            prompt_path=data.get("prompt_path"),
            repo_path=data.get("repo_path"),
            repo_ref=data.get("repo_ref"),
            max_iterations=int(data.get("max_iterations", 500)),
            n_critic_runs=int(data.get("n_critic_runs", 1)),
            max_retries=int(data.get("max_retries", 3)),
            tool_preset=data.get("tool_preset", "default"),
            enable_delegation=_parse_bool(data.get("enable_delegation")),
            note=data.get("note", "olmo-eval"),
        )
