"""CPU-only live FACT acceptance probe. Judges are local; crawl4ai/Chromium stay real."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import psutil
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

from olmo_eval.common.types import LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks import deepresearch_bench as dr
from olmo_eval.evals.tasks.common import get_task

SENTINEL = "W0184_HEALTHY_EVIDENCE_SENTINEL"
EVENTS = []
CRAWLERS = []
OWNED = {}
HANG_SEEN = threading.Event()
STOP_SERVER = threading.Event()


def emit(kind, **values):
    row = {"event": kind, "time_monotonic": time.monotonic(), **values}
    EVENTS.append(row)
    print(json.dumps(row, sort_keys=True), flush=True)


def descendants():
    return {p.pid: p for p in psutil.Process().children(recursive=True)}


def track_children():
    for pid, process in descendants().items():
        with contextlib.suppress(psutil.NoSuchProcess):
            OWNED.setdefault(pid, process.create_time())


def living_owned():
    live = []
    for pid, created in OWNED.items():
        try:
            p = psutil.Process(pid)
            if p.create_time() == created and p.status() != psutil.STATUS_ZOMBIE:
                live.append(p)
        except psutil.NoSuchProcess:
            pass
    return live


class Fixture(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/hang":
            HANG_SEEN.set()
            STOP_SERVER.wait(180)
            return
        status = 403 if route == "/blocked" else 200
        self.send_response(302 if route.startswith("/redirect") else status)
        if route.startswith("/redirect"):
            target = "/redirect-loop" if route == "/redirect-loop" else "/healthy"
            self.send_header("Location", target + "?" + uuid4().hex)
        if route == "/cookie-set":
            self.send_header("Set-Cookie", "w0184_cookie=present; Path=/")
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        cookie = (
            "COOKIE_PRESENT"
            if "w0184_cookie=present" in self.headers.get("Cookie", "")
            else "COOKIE_ABSENT"
        )
        content = "BLOCKED_FIXTURE" if status == 403 else SENTINEL
        body = (
            f"<html><body><h1>{content}</h1><p>{cookie}</p><p>"
            + ("Substantive local evidence about the synthetic claim. " * 35)
            + "</p></body></html>"
        )
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body.encode())


class InstrumentedCrawler(AsyncWebCrawler):
    def __init__(self):
        super().__init__(
            base_directory=tempfile.mkdtemp(prefix="crawler-", dir=os.environ["W0184_CACHE_DIR"])
        )
        self.probe_id = len(CRAWLERS)
        self.probe_children = set()
        CRAWLERS.append(self)

    async def start(self):
        before = set(descendants())
        value = await super().start()
        track_children()
        self.probe_children = set(descendants()) - before
        emit(
            "start",
            crawler=self.probe_id,
            owned_children=len(self.probe_children),
            chromium_version=self.crawler_strategy.browser_manager.browser.version,
        )
        return value

    async def arun(self, url, *args, **kwargs):
        emit("fetch_begin", crawler=self.probe_id, url_id=urlsplit(url).path)
        try:
            result = await super().arun(url, *args, **kwargs)
        except BaseException as exc:
            emit("fetch_raise", crawler=self.probe_id, error_type=type(exc).__name__)
            raise
        track_children()
        emit(
            "fetch_end",
            crawler=self.probe_id,
            success=bool(result.success),
            status=getattr(result, "status_code", None),
            error_sha256=hashlib.sha256(str(result.error_message).encode()).hexdigest(),
            child_count=len(living_owned()),
        )
        return result

    async def close(self):
        try:
            return await super().close()
        finally:
            emit("close", crawler=self.probe_id)


def count(kind, since=0):
    return sum(e["event"] == kind for e in EVENTS[since:])


@contextlib.asynccontextmanager
async def session(recycle=200):
    os.environ["DEEPRESEARCH_FACT_CRAWLER_RECYCLE_AFTER"] = str(recycle)
    async with dr._FactCrawlerSession() as value:
        token = dr._FACT_CRAWLER_SESSION.set(value)
        try:
            yield value
        finally:
            dr._FACT_CRAWLER_SESSION.reset(token)
    assert value._crawler is None
    assert dr._FACT_CRAWLER_SESSION.get() is None


def url(base, route):
    return base + route + "?" + uuid4().hex


async def fetch(base, route, healthy=True):
    text = await dr.fetch_crawl4ai_page(url(base, route))
    emit(
        "text",
        route=route,
        failed=dr.is_obvious_scrape_failure(text),
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        length=len(text),
        outer_deadline="page fetch timed out after" in text,
        navigation_timeout="timeout" in text.lower() or "timed out" in text.lower(),
    )
    if healthy:
        assert not dr.is_obvious_scrape_failure(text), text[:300]
        assert SENTINEL in text
    return text


async def healthy_sessions(base):
    for recycle, expected in ((200, 1), (2, 3)):
        begin = len(EVENTS)
        async with session(recycle):
            for _ in range(5):
                await fetch(base, "/healthy")
        assert (count("start", begin), count("close", begin)) == (expected, expected)
    for cap in (2, 4):
        begin = len(EVENTS)

        async def one():
            async with session():
                await fetch(base, "/healthy")

        await asyncio.gather(*(one() for _ in range(cap)))
        assert (count("start", begin), count("close", begin)) == (cap, cap)
        emit("concurrency_pass", cap=cap)


async def navigation_failures(base):
    async with session():
        await fetch(base, "/redirect-finite")
        for route in ("/redirect-loop", "/hang"):
            text = await fetch(base, route, healthy=False)
            assert dr.is_obvious_scrape_failure(text)
            assert "page fetch timed out after" not in text, (
                "Expected crawl4ai's navigation failure before outer timer"
            )
            await fetch(base, "/healthy")
        begin = len(EVENTS)
        text = await fetch(base, "/blocked", healthy=False)
        # Real crawl4ai can return a successful scrape of a 403 response body. Preserve it.
        assert "BLOCKED_FIXTURE" in text or dr.is_obvious_scrape_failure(text)
        await fetch(base, "/healthy")
        assert count("start", begin) == 0, "Ordinary 403 must keep the current browser"


async def crash_recovery(base):
    begin = len(EVENTS)
    async with session() as owner:
        await fetch(base, "/healthy")
        crawler = owner._crawler
        HANG_SEEN.clear()
        pending = asyncio.create_task(fetch(base, "/hang", healthy=False))
        while not HANG_SEEN.is_set():
            assert not pending.done(), "Crawl ended before crash injection"
            await asyncio.sleep(0.01)
        track_children()
        candidates = []
        for pid in crawler.probe_children:
            try:
                process = psutil.Process(pid)
                args = process.cmdline()
                if (
                    process.create_time() == OWNED[pid]
                    and any("chrom" in arg for arg in args[:1])
                    and not any(arg.startswith("--type=") for arg in args)
                ):
                    candidates.append(process)
            except psutil.NoSuchProcess:
                pass
        assert len(candidates) == 1, "Cannot uniquely identify this crawler's owned Chromium root"
        emit("owned_crash", crawler=crawler.probe_id, pid=candidates[0].pid)
        candidates[0].kill()
        text = await pending
        assert dr.is_obvious_scrape_failure(text)
        assert any(
            e["event"] == "fetch_end" and e["crawler"] == crawler.probe_id and not e["success"]
            for e in EVENTS[begin:]
        ), "Need a real unsuccessful result, not only an escaped exception"
        assert owner._crawler is None
        await fetch(base, "/healthy")
    assert (count("start", begin), count("close", begin)) == (2, 2)


async def cancellation(base):
    begin = len(EVENTS)

    async def operation():
        async with session():
            await fetch(base, "/healthy")
            HANG_SEEN.clear()
            await fetch(base, "/hang", healthy=False)

    wrapper = asyncio.create_task(operation())
    while not HANG_SEEN.is_set():
        assert not wrapper.done()
        await asyncio.sleep(0.01)
    wrapper.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(wrapper, 10)
    assert wrapper.cancelled()
    assert count("start", begin) == count("close", begin) == 1


async def cookies(base):
    observations = []
    for recycle in (200, 1):
        async with session(recycle):
            await fetch(base, "/cookie-set")
            text = await fetch(base, "/cookie-read")
            observations.append(
                {
                    "recycle": recycle,
                    "cookie_present": "COOKIE_PRESENT" in text,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
    assert not observations[1]["cookie_present"], "Per-fetch browser isolation leaked a cookie"
    emit("cookie_comparison", observations=observations)


async def scoring(base):
    task = get_task("deepresearch_bench")
    criteria = {
        "dimension_weight": {d: 0.25 for d in dr.DEEPRESEARCH_DIMENSIONS},
        "criterions": {
            d: [{"criterion": d, "explanation": "Synthetic fixture", "weight": 1.0}]
            for d in dr.DEEPRESEARCH_DIMENSIONS
        },
    }
    race_payload = {
        d: [{"criterion": d, "article_1_score": 8, "article_2_score": 2}]
        for d in dr.DEEPRESEARCH_DIMENSIONS
    }
    stub_errors = []
    validation_calls = []
    sources = []

    async def race_judge(_prompt):
        return json.dumps(race_payload)

    async def fact_judge(prompt):
        if "Here is the main text" in prompt:
            return json.dumps(
                [
                    {"fact": "Synthetic supported claim.", "ref_idx": i + 1, "url": source}
                    for i, source in enumerate(sources)
                ]
            )
        if SENTINEL not in prompt:
            stub_errors.append("Validation called without healthy evidence")
            raise AssertionError(stub_errors[-1])
        validation_calls.append(1)
        return json.dumps([{"idx": 1, "result": "supported"}])

    dr.build_deepresearch_race_judge_fn = lambda: race_judge
    dr.build_deepresearch_fact_judge_fn = lambda: fact_judge
    os.environ["DEEPRESEARCH_FACT_CRAWLER_RECYCLE_AFTER"] = "200"
    for case in range(2):
        sources[:] = ([url(base, "/hang")] if case == 0 else []) + [url(base, "/healthy")]
        instance = task.process_doc(
            {
                "id": case,
                "language": "en",
                "topic": "Fixture",
                "prompt": "Describe fixture evidence.",
                "criteria": criteria,
                "reference_article": "Synthetic reference.",
            },
            case,
        )
        response = Response(
            instance=instance,
            request=LMRequest(request_type=RequestType.CHAT, messages=()),
            outputs=[
                LMOutput(text="A synthetic report " + " ".join(f"[source]({s})" for s in sources))
            ],
            scores={},
        )
        await task.score_responses([response])
        verdicts = [r["result"] for r in response.outputs[0].metadata["deepresearch_fact"]]
        assert verdicts == (["unknown"] if case == 0 else []) + ["supported"]
        assert response.scores["race_overall"] == 0.8
        for key in (
            "fact_citation_accuracy",
            "fact_avg_effective_citations",
            "fact_avg_citations",
            "fact_has_citations",
        ):
            assert response.scores[key] == 1.0
        assert all(
            math.isfinite(v) for v in response.scores.values() if isinstance(v, (int, float))
        )
        assert dr._FACT_CRAWLER_SESSION.get() is None
        emit(
            "scoring_pass",
            case=case,
            verdicts=verdicts,
            scores=response.scores,
            extracted_citations=len(verdicts),
            evaluated_citations=1,
            supported_citations=1,
        )
    assert not stub_errors and len(validation_calls) == 2


async def main(base):
    baseline = asyncio.all_tasks()
    dr._import_crawl4ai = lambda: (InstrumentedCrawler, CrawlerRunConfig)
    emit(
        "versions",
        python=platform.python_version(),
        executable=sys.executable,
        packages={
            p: importlib.metadata.version(p)
            for p in ("crawl4ai", "playwright", "olmo-eval", "psutil")
        },
        default_page_timeout=CrawlerRunConfig().page_timeout,
        page_timeout=dr.fact_page_timeout_ms(),
        fetch_margin=dr.DEEPRESEARCH_FACT_FETCH_TIMEOUT_MARGIN_SECONDS,
        judge_timeout=dr.fact_judge_timeout_seconds(),
        recycle=dr.fact_crawler_recycle_after(),
    )
    assert CrawlerRunConfig().page_timeout == 60000
    assert dr.fact_page_timeout_ms() == 1000
    assert dr.DEEPRESEARCH_FACT_FETCH_TIMEOUT_MARGIN_SECONDS == 30
    assert dr.fact_judge_timeout_seconds() == 10
    await fetch(base, "/healthy")
    await healthy_sessions(base)
    await navigation_failures(base)
    await crash_recovery(base)
    HANG_SEEN.clear()
    await cancellation(base)
    await cookies(base)
    await scoring(base)
    await asyncio.sleep(3)
    pending = [t for t in asyncio.all_tasks() - baseline if not t.done()]
    assert not pending, f"Pending preflight tasks: {pending}"
    assert not living_owned(), "Owned child processes survived session cleanup"
    assert count("start") == count("close")
    emit("PASS", starts=count("start"), completed_fetches=count("fetch_end"), closes=count("close"))


if __name__ == "__main__":
    # The supervisor gives each interpreter a new cache directory and a process group.
    assert not any(
        os.getenv(k) for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "S2_API_KEY")
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    status = "failed"
    try:
        asyncio.run(main(f"http://127.0.0.1:{server.server_port}"))
        status = "passed"
    finally:
        STOP_SERVER.set()
        server.shutdown()
        server.server_close()
        survivors = living_owned()
        emit("cleanup", status=status, survivors_before_forced_cleanup=len(survivors))
        for process in reversed(survivors):
            with contextlib.suppress(psutil.NoSuchProcess):
                process.kill()
        psutil.wait_procs(survivors, timeout=3)
        Path(sys.argv[1]).write_text(
            json.dumps({"status": status, "events": EVENTS}, indent=2) + "\n"
        )
