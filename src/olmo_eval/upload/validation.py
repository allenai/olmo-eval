"""Payload checks against the ingest contract.

The full JSON Schema lives in the repository at
``dashboard/contract/ingest-v1.schema.json``. When it and ``jsonschema`` are
available, payloads are validated against it; otherwise only required keys are
checked.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA_RELATIVE_PATH = Path("dashboard") / "contract" / "ingest-v1.schema.json"

_REQUIRED: dict[str, tuple[str, ...]] = {
    "RunUpsertRequest": ("protocol_version", "client", "run", "model"),
    "SignArtifactsRequest": ("artifacts",),
    "TaskResultIn": (
        "task_name",
        "task_hash",
        "primary_metric",
        "metrics",
        "metric_meta",
        "config",
        "num_instances",
        "suites",
        "instance_count",
    ),
    "InstanceBatchRequest": ("batch_index", "instances"),
    "InferenceUploadRequest": ("source_paths", "batches", "series", "gpu_devices"),
    "CompleteRequest": ("status", "suites", "expected_task_results", "expected_artifacts"),
}


def find_contract_schema(start: Path | None = None) -> Path | None:
    """Look for the contract schema above the working directory and the package."""
    import olmo_eval

    candidates = [Path.cwd() if start is None else start, Path(olmo_eval.__file__).resolve()]
    for base in candidates:
        for directory in [base, *base.parents]:
            path = directory / SCHEMA_RELATIVE_PATH
            if path.is_file():
                return path
    return None


@lru_cache(maxsize=4)
def _validator_for(schema_path: str | None, definition: str) -> Any:
    if schema_path is None:
        return None
    try:
        import jsonschema
    except ImportError:
        return None
    schema = json.loads(Path(schema_path).read_text())
    wrapper = {
        "$schema": schema.get("$schema", "http://json-schema.org/draft-07/schema#"),
        "$ref": f"#/definitions/{definition}",
        "definitions": schema["definitions"],
    }
    return jsonschema.Draft7Validator(wrapper, format_checker=jsonschema.FormatChecker())


class PayloadValidator:
    """Validates payloads by contract definition name."""

    def __init__(self, schema_path: Path | None = None) -> None:
        self.schema_path = schema_path if schema_path is not None else find_contract_schema()

    @property
    def uses_schema(self) -> bool:
        return (
            self.schema_path is not None
            and _validator_for(str(self.schema_path), "RunUpsertRequest") is not None
        )

    def errors(self, definition: str, payload: Mapping[str, Any]) -> list[str]:
        """Return validation messages (empty when the payload is valid)."""
        validator = _validator_for(str(self.schema_path) if self.schema_path else None, definition)
        if validator is not None:
            messages = []
            for error in validator.iter_errors(payload):
                location = "/".join(str(part) for part in error.absolute_path)
                prefix = f"{definition}/{location}" if location else definition
                messages.append(f"{prefix}: {error.message}")
            return messages
        missing = [key for key in _REQUIRED.get(definition, ()) if key not in payload]
        return [f"{definition}: missing required key {key!r}" for key in missing]
