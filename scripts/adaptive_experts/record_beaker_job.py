#!/usr/bin/env python3
"""Append one submitted experiment to the adaptive-experts Beaker ledger."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--expert-count", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--source-snapshot", required=True)
    parser.add_argument("--result-storage", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--command", required=True)
    args = parser.parse_args()

    record = {
        "launched_at_utc": datetime.now(UTC).isoformat(),
        "phase": args.phase,
        "run_tag": args.run_tag,
        "expert_count": args.expert_count,
        "model": args.model,
        "tasks": args.task,
        "cluster": args.cluster,
        "workspace": args.workspace,
        "priority": args.priority,
        "group": args.group,
        "source_commit": args.source_commit,
        "source_hash": args.source_hash,
        "source_snapshot": args.source_snapshot,
        "result_storage": args.result_storage,
        "beaker_experiment_id": args.experiment_id,
        "url": f"https://beaker.org/ex/{args.experiment_id}",
        "status_at_recording": "submitted",
        "command": args.command,
    }

    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.open("a+", encoding="utf-8") as ledger:
        fcntl.flock(ledger.fileno(), fcntl.LOCK_EX)
        ledger.seek(0)
        for line in ledger:
            if not line.strip():
                continue
            existing = json.loads(line)
            if existing.get("beaker_experiment_id") == args.experiment_id:
                return
        ledger.write(json.dumps(record, sort_keys=True) + "\n")
        ledger.flush()
        os.fsync(ledger.fileno())


if __name__ == "__main__":
    main()
