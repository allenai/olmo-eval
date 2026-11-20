"""Offline deterministic workload for real Beaker preemption/recovery qualification.

Launch through the shared-run supervisor. This script registers a local fixture task
only in its process; it never changes established task specs or contacts a model API.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType
from olmo_eval.evals.tasks.common import Task, register


@dataclass(frozen=True)
class ProbeScorer(Scorer):
    name: str = "probe"

    def score(self, instance: Instance, output: LMOutput) -> float:
        time.sleep(float(os.environ.get("WEKA_PROBE_DELAY", "0.5")))
        return float("42" in (output.text or ""))


@register("weka_resume_probe")
class ProbeTask(Task):
    data_source = "local/fixture"
    metrics = (AccuracyMetric(scorer=ProbeScorer),)

    @property
    def instances(self) -> Iterator[Instance]:
        for index in range(80):
            yield Instance(
                question=f"Fixture question {index}", gold_answer="42", metadata={"id": index}
            )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question)

    def extract_answer(self, output: LMOutput) -> str:
        return "42" if "42" in (output.text or "") else ""


def main() -> None:
    from olmo_eval.cli import main as cli

    gib = int(os.environ.get("WEKA_PROBE_PAYLOAD_GIB", "0"))
    if not 0 <= gib <= 10:
        raise ValueError("Probe payload must be between 0 and 10 GiB")
    if gib:
        started = time.monotonic()
        path = Path(sys.argv[-1]) / "scale.bin"
        expected = hashlib.sha256()
        chunk = bytes(range(256)) * 4096
        for _ in range(gib * 1024):
            expected.update(chunk)
        if not path.exists():
            with path.open("wb") as stream:
                for _ in range(gib * 1024):
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == expected.hexdigest()
        print(
            json.dumps(
                {"payload_gib": gib, "sha256": actual, "verify_seconds": time.monotonic() - started}
            ),
            flush=True,
        )
    args = [
        "run",
        "-m",
        "mock",
        "-t",
        "weka_resume_probe",
        "--resume",
        "--no-upload",
        "--harness",
        "default",
        "-o",
        "scoring_concurrency=1",
        "-o",
        "max_concurrency=1",
    ]
    sys.argv = ["olmo-eval", *args, *sys.argv[1:]]
    cli()


if __name__ == "__main__":
    main()
