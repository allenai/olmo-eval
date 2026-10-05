"""Run size peers with a shared, explicit evaluation contract."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Peer:
    model: str
    revision: str
    external_server: bool = False


PEERS = {
    "ling-tiny": Peer(
        "inclusionAI/Ling-3.0-tiny", "9a98e35fe1c9ee255f78dd64771c7ae15a799481", True
    ),
    "lfm": Peer("LiquidAI/LFM2.5-8B-A1B", "5dd22602c2e9f6a097b1de4c4efe0658b605015c"),
    "minicpm": Peer("openbmb/MiniCPM5-1B", "87179e5c1f455ef22e6223592d2d61351b525bfc"),
}

CORE_TASKS = (
    "aime_2025",
    "aime_2026",
    "hmmt_feb_2026",
    "gpqa_diamond:cot",
    "ifeval",
    "ifeval_ood",
)


def build_command(
    peer_name: str, phase: str, output_dir: Path, base_url: str | None = None
) -> list[str]:
    """Build a command for a smoke check or a complete benchmark pass."""
    peer = PEERS[peer_name]
    if peer.external_server and not base_url:
        raise ValueError("Ling needs an external server with BailingMoeV3 support")
    if phase not in {"smoke", "core", "knowledge"}:
        raise ValueError(f"Unknown phase: {phase}")
    command = [
        "olmo-eval",
        "run",
        "--model",
        peer.model,
        "--harness",
        "default",
        "-o",
        "provider.kind=vllm_server",
        "-o",
        f"provider.revision={peer.revision}",
        "-o",
        "provider.trust_remote_code=true",
        "-o",
        "provider.max_model_len=65536",
        "-o",
        "provider.max_concurrency=16",
        "-o",
        "max_hard_failure_rate=0.0",
        "-o",
        'provider.kwargs.chat_template_kwargs={"enable_thinking":true}',
    ]
    if base_url:
        command += ["-o", f"provider.base_url={base_url}"]
    else:
        command += ["-o", "provider.kwargs.gpu_memory_utilization=0.85"]
    tasks = ("mmlu_pro:cot",) if phase == "knowledge" else CORE_TASKS
    for task in tasks:
        command += ["-t", task]
        for override in (
            "temperature=1.0",
            "do_sample=true",
            "top_p=0.95",
            "top_k=20",
            "num_samples=1",
            "strip_thinking=true",
            f"max_tokens={128 if phase == 'smoke' else 32768}",
        ):
            command += ["-o", override]
        if phase == "smoke":
            command += ["-o", "limit=2"]
    command += [
        "--output-dir",
        str(output_dir),
        "--experiment-name",
        f"peer-{peer_name}-{phase}-t1",
        "--experiment-group",
        "size-peers-2026-10-05",
        "--save-predictions",
        "--save-requests",
    ]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peer", choices=PEERS, required=True)
    parser.add_argument("--phase", choices=("smoke", "core", "knowledge"), required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        command = build_command(args.peer, args.phase, args.output_dir, args.base_url)
    except ValueError as error:
        parser.error(str(error))
    print(shlex.join(command), flush=True)
    if args.dry_run:
        subprocess.run([*command, "--dry-run"], check=True)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "commit": commit,
        "peer": args.peer,
        "model_revision": PEERS[args.peer].revision,
        "phase": args.phase,
        "command": command,
        "smoke_only": args.phase == "smoke",
        "sampling": "T=1, top_p=0.95, top_k=20, one draw per item",
    }
    (args.output_dir / "peer-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
