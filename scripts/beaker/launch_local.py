#!/usr/bin/env python3
"""Run the olmo-eval CLI while honoring explicit Beaker secret mappings.

This is a local submission shim. It removes environment variables supplied by
``--secret-env BEAKER_SECRET:ENV_VAR`` from the launcher's user-prefixed secret
preflight; the normal job assembler still mounts the explicitly named secret.
"""

from __future__ import annotations

import sys


def _explicit_secret_env_vars(argv: list[str]) -> set[str]:
    env_vars: set[str] = set()
    index = 0
    while index < len(argv):
        argument = argv[index]
        mapping: str | None = None
        if argument == "--secret-env" and index + 1 < len(argv):
            index += 1
            mapping = argv[index]
        elif argument.startswith("--secret-env="):
            mapping = argument.split("=", 1)[1]

        if mapping and ":" in mapping:
            _, env_var = mapping.split(":", 1)
            env_vars.add(env_var)
        index += 1
    return env_vars


def main() -> None:
    from olmo_eval.launch.beaker import secrets

    explicit_env_vars = _explicit_secret_env_vars(sys.argv[1:])
    original_ensure_task_secrets = secrets.ensure_task_secrets

    def ensure_task_secrets(workspace: str, required_secrets: set[str]) -> list[tuple[str, str]]:
        return original_ensure_task_secrets(workspace, required_secrets - explicit_env_vars)

    secrets.ensure_task_secrets = ensure_task_secrets

    from olmo_eval.cli import main as olmo_eval_main

    olmo_eval_main()


if __name__ == "__main__":
    main()
