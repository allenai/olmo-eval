#!/usr/bin/env python3
"""Launch the ledger's text eval cells on a text-only HF export of a multimodal SFT checkpoint.

The cells are copied from the GPU text SFT control's recorded Beaker specs (olmo-post-training-ledger
``experiments/h182/eval/specs`` and its holmes copies ``.../holmes-cu130/specs``), so task, sampling,
olmo-eval commit, vLLM and plugins stay exactly as recorded. Only these change: the model path, the
experiment name and group, the per-user secrets (HF and W&B to ``<user>_*``; the Beaker token is
dropped, it only posts cosmetic status), and placement (``--cluster``, priority urgent, not
preemptible).

    python text_cells.py <text hf dir> <short-name> --ledger <ledger checkout> [--cells ifeval gpqa ...]
                         [--user jasonr] [--workspace ai2/olmo-instruct] [--cluster ai2/holmes] [--dry-run]

``<text hf dir>`` comes from ``mm_hf_to_text_hf.py`` (Olmo 3.5 chat tokenizer, think tags promoted).
A holmes copy of a cell is preferred when it exists (the B300 cu130 runtime the control pair scored on).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys

SOURCE_MODEL = "/weka/olmo-3p5-post-checkpoints/abhishekr/h182-gpu/h182gpu-s34521/hf_step18410"
SOURCE_PREFIX = "h182-b1-c12r-gpu-s34521-"
SELECTION_CELLS = ("gpqa", "ifeval", "ifbench", "ruler64k", "rp-niah-smk", "rp-mvmqvt", "rp-aggqa")


def build(spec_path: str, model: str, name: str, user: str, cluster: str, priority: str = "urgent") -> dict:
    spec = copy.deepcopy(json.load(open(spec_path)))
    task = spec["tasks"][0]
    args = task["arguments"]
    if args.count(SOURCE_MODEL) != 1:
        raise SystemExit(f"{spec_path}: expected the source model path once")
    args[args.index(SOURCE_MODEL)] = model
    for flag, value in (("--experiment-group", f"{name}-group"), ("--experiment-name", name)):
        if flag in args:
            args[args.index(flag) + 1] = value
    env = []
    for entry in task["envVars"]:
        secret = entry.get("secret")
        if secret and secret.endswith("_BEAKER_TOKEN"):
            continue
        if secret:
            entry = {**entry, "secret": re.sub(r"^[a-z0-9]+_", f"{user}_", secret)}
        env.append(entry)
    task["envVars"] = env
    mounts = {d.get("mountPath") for d in task.get("datasets", [])}
    for bucket in ("olmo-3p5-post-checkpoints", "oe-training-default"):
        if f"/weka/{bucket}" not in mounts:
            task.setdefault("datasets", []).append({"mountPath": f"/weka/{bucket}", "source": {"weka": bucket}})
    task["constraints"] = {"cluster": cluster.split(",")}
    task["context"] = {"priority": priority, "preemptible": False}
    spec["description"] = f"multimodal SFT text cell {name} (copied from {os.path.basename(spec_path)})"
    return spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("short_name")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--cells", nargs="*", default=list(SELECTION_CELLS))
    ap.add_argument("--user", default="jasonr")
    ap.add_argument("--workspace", default="ai2/olmo-instruct")
    ap.add_argument("--cluster", default="ai2/holmes", help="comma-separated clusters")
    ap.add_argument("--priority", default="urgent")
    ap.add_argument("--suffix", default="", help="appended to each job name (e.g. -alt)")
    ap.add_argument("--out", default=None, help="where to write the specs (default: <model>/eval-specs)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-local-check", action="store_true", help="model on a bucket this machine does not mount")
    args = ap.parse_args()
    if not args.skip_local_check and not os.path.exists(os.path.join(args.model, "config.json")):
        sys.exit(f"not an HF model dir: {args.model}")
    base = os.path.join(args.ledger, "experiments", "h182", "eval")
    out = args.out or ("/root/mmsft-eval/text-specs/" + args.short_name if args.skip_local_check else os.path.join(args.model, "eval-specs"))
    os.makedirs(out, exist_ok=True)
    for cell in args.cells:
        holmes = os.path.join(base, "holmes-cu130", "specs", f"{SOURCE_PREFIX}{cell}.json")
        plain = os.path.join(base, "specs", f"{SOURCE_PREFIX}{cell}.json")
        # The holmes copies install the B300 cu130 runtime; other clusters take the recorded
        # (jupiter, cu128) spec.
        on_holmes = "ai2/holmes" in args.cluster.split(",")
        src = holmes if on_holmes and os.path.exists(holmes) else plain
        if not os.path.exists(src):
            sys.exit(f"no recorded spec for cell {cell!r}")
        name = f"mmsft-{args.short_name}-{cell}{args.suffix}"
        spec = build(src, os.path.abspath(args.model), name, args.user, args.cluster, args.priority)
        path = os.path.join(out, f"{name}.json")
        json.dump(spec, open(path, "w"), indent=1)
        if args.dry_run:
            print(f"{name}: {path}")
            continue
        res = subprocess.run(
            ["beaker", "experiment", "create", path, "--workspace", args.workspace, "--name", name],
            capture_output=True, text=True,
        )
        ids = re.findall(r"01[0-9A-Z]{24}", res.stdout + res.stderr)
        print(f"{name}: {ids[0] if ids else 'FAILED ' + (res.stderr or res.stdout)[-300:]}")


if __name__ == "__main__":
    main()
