"""Upload settings resolved from CLI flags and environment variables."""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_API_URL = "https://prod-ingest.olmo-eval.apps.allenai.org"
DEFAULT_TIMEOUT_S = 1800.0

UPLOAD_ENV = "OLMO_EVAL_UPLOAD"
API_URL_ENV = "OLMO_EVAL_API_URL"
TIMEOUT_ENV = "OLMO_EVAL_UPLOAD_TIMEOUT"
LAUNCH_ID_ENV = "OLMO_EVAL_LAUNCH_ID"
# Uploader service account key (JSON content or a file path). Beaker jobs get it from the
# shared olmo_eval_uploader_key secret; when unset, uploads use local credentials.
UPLOAD_CREDENTIALS_ENV = "OLMO_EVAL_UPLOAD_CREDENTIALS"

MAX_TAGS = 50
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,63}$")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class UploadConfig:
    """Where and whether to upload results.

    Attributes:
        enabled: Upload results to the ingest service.
        api_url: Base URL of the ingest service.
        tags: Free-form labels attached to the run.
        timeout_s: Overall deadline for one upload, in seconds.
    """

    enabled: bool = True
    api_url: str = DEFAULT_API_URL
    tags: tuple[str, ...] = ()
    timeout_s: float = DEFAULT_TIMEOUT_S

    @property
    def is_default_api_url(self) -> bool:
        return self.api_url.rstrip("/") == DEFAULT_API_URL


def parse_bool(value: str | None) -> bool | None:
    """Parse a boolean environment value. Returns None when unset or unrecognized."""
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    return None


def env_upload_enabled() -> bool | None:
    """Return the OLMO_EVAL_UPLOAD setting, or None when it is unset."""
    return parse_bool(os.environ.get(UPLOAD_ENV))


def validate_tags(tags: Sequence[str]) -> tuple[str, ...]:
    """Check tags against the ingest contract and drop duplicates.

    Raises:
        ValueError: If a tag is malformed or there are too many.
    """
    unique = tuple(dict.fromkeys(tags))
    bad = [tag for tag in unique if not _TAG_RE.match(tag)]
    if bad:
        raise ValueError(
            f"Invalid tag(s): {', '.join(bad)}. Tags start with a letter or digit and use "
            "only letters, digits and . _ : / + - (at most 64 characters)."
        )
    if len(unique) > MAX_TAGS:
        raise ValueError(f"At most {MAX_TAGS} tags are allowed, got {len(unique)}.")
    return unique


def resolve_upload_config(
    cli_upload: bool | None = None,
    cli_api_url: str | None = None,
    cli_tags: Sequence[str] = (),
) -> UploadConfig:
    """Combine CLI flags and environment variables. CLI flags win.

    Args:
        cli_upload: Value of --upload/--no-upload, or None when not passed.
        cli_api_url: Value of --api-url, or None when not passed.
        cli_tags: Values of --tag.
    """
    enabled = cli_upload if cli_upload is not None else env_upload_enabled()
    api_url = cli_api_url or os.environ.get(API_URL_ENV) or DEFAULT_API_URL
    timeout_s = DEFAULT_TIMEOUT_S
    raw_timeout = os.environ.get(TIMEOUT_ENV)
    if raw_timeout:
        try:
            timeout_s = max(1.0, float(raw_timeout))
        except ValueError:
            timeout_s = DEFAULT_TIMEOUT_S
    return UploadConfig(
        enabled=True if enabled is None else enabled,
        api_url=api_url.rstrip("/"),
        tags=validate_tags(cli_tags),
        timeout_s=timeout_s,
    )


def retry_command(output_dir: str, config: UploadConfig) -> str:
    """The command that re-uploads a results directory."""
    parts = ["olmo-eval", "results", "upload", output_dir]
    if not config.is_default_api_url:
        parts += ["--api-url", config.api_url]
    for tag in config.tags:
        parts += ["--tag", tag]
    return shlex.join(parts)
