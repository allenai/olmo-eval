"""Run both GLM scoring interpreters with independent caches and outer process deadlines."""

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

root = Path(__file__).resolve().parent
output = Path("/results/w0184")
output.mkdir(parents=True, exist_ok=True)
results = []
for label, interpreter in [
    ("planner", "/opt/venv/bin/python"),
    ("single", "/opt/oe-venv/bin/python"),
]:
    env = os.environ.copy()
    for key in list(env):
        if any(part in key for part in ("TOKEN", "API_KEY", "SECRET", "PASSWORD")):
            env.pop(key)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HF_HUB_OFFLINE"] = "1"
    env["DEEPRESEARCH_SKIP_FACT"] = "0"
    env["DEEPRESEARCH_FACT_PAGE_TIMEOUT_MS"] = "1000"
    env["DEEPRESEARCH_FACT_JUDGE_TIMEOUT_S"] = "10"
    env["DEEPRESEARCH_FACT_CRAWLER_RECYCLE_AFTER"] = "200"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="w0184-") as scratch:
        env["W0184_CACHE_DIR"] = scratch
        env["CRAWL4_AI_BASE_DIRECTORY"] = scratch
        with (output / f"{label}.log").open("w") as log:
            started = time.monotonic()
            proc = subprocess.Popen(
                [interpreter, str(root / "browser_probe.py"), str(output / f"{label}.json")],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                code = proc.wait(timeout=240)
            except subprocess.TimeoutExpired:
                timed_out = True
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
                code = 124
            finally:
                # Dedicated process group only, including children left after parent failure.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        result_path = output / f"{label}.json"
        payload = json.loads(result_path.read_text()) if result_path.exists() else {}
        passed = (
            code == 0
            and payload.get("status") == "passed"
            and any(e["event"] == "PASS" for e in payload.get("events", []))
        )
        results.append(
            {
                "interpreter": interpreter,
                "exit_code": code,
                "timeout": timed_out,
                "elapsed_seconds": time.monotonic() - started,
                "passed": passed,
                "log_sha256": hashlib.sha256((output / f"{label}.log").read_bytes()).hexdigest(),
            }
        )
(output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
sys.exit(0 if len(results) == 2 and all(r["passed"] for r in results) else 1)
