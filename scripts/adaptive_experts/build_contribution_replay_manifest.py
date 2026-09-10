#!/usr/bin/env python3
"""Join the router-profile prompt manifest with its saved generated token trajectories."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt") as handle:
        while True:
            try:
                line = handle.readline()
            except (EOFError, gzip.BadGzipFile):
                break
            if not line:
                break
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    results = repo / "results" / "router_profile"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=repo / "notes" / "router_profile" / "qwen3_k8_manifest.jsonl",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        nargs="+",
        default=[
            *sorted(
                (
                    results
                    / "01KXBSST8ZP8VBZ469GWB1CWCR.partial"
                    / "workers"
                ).glob("rank*/examples.jsonl.gz")
            ),
            *sorted(
                (
                    results
                    / "01KXD4JDGY5ZC96XHB5E3VKKDC"
                    / "workers"
                ).glob("rank*/examples.jsonl.gz")
            ),
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repo
            / "notes"
            / "router_profile"
            / "qwen3_k8_contribution_replay_manifest.jsonl.gz"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = [json.loads(line) for line in args.prompt_manifest.read_text().splitlines() if line]
    trajectories: dict[tuple[str, str], dict[str, Any]] = {}
    for path in args.examples:
        for row in read_gzip_jsonl(path):
            key = (row["task"], str(row["doc_id"]))
            if key in trajectories:
                raise ValueError(f"duplicate trajectory: {key}")
            trajectories[key] = row

    output_rows = []
    for prompt in prompts:
        key = (prompt["task"], str(prompt["doc_id"]))
        trajectory = trajectories.get(key)
        if trajectory is None:
            raise ValueError(f"missing trajectory: {key}")
        output_rows.append(
            {
                **prompt,
                "prompt_tokens_recorded": trajectory["prompt_tokens"],
                "generated_tokens_recorded": trajectory["generated_tokens"],
                "finish_reason": trajectory["finish_reason"],
                "generated_token_ids": trajectory["generated_token_ids"],
            }
        )

    extra = set(trajectories) - {
        (prompt["task"], str(prompt["doc_id"])) for prompt in prompts
    }
    if extra:
        raise ValueError(f"unexpected trajectories: {sorted(extra)}")
    if len(output_rows) != 60:
        raise ValueError(f"expected 60 trajectories, got {len(output_rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(output_rows)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
