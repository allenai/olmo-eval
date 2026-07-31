"""Probe the effective Semantic Scholar Graph API request rate.

This sends requests at fixed rates without retrying 429 responses. The API key
is read from ``S2_API_KEY`` and is never printed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


@dataclass(frozen=True)
class Result:
    status: int | None
    latency: float
    retry_after: str | None = None
    rate_headers: tuple[tuple[str, str], ...] = ()
    error: str | None = None


def parse_rates(value: str) -> list[float]:
    try:
        rates = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rates must be comma-separated numbers") from exc
    if not rates or any(rate <= 0 for rate in rates):
        raise argparse.ArgumentTypeError("rates must all be greater than zero")
    return rates


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def response_rate_headers(headers: Any) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (name, value)
            for name, value in headers.items()
            if name.lower().startswith("x-ratelimit")
        )
    )


def request_once(url: str, api_key: str, timeout: float) -> Result:
    request = Request(
        url,
        headers={
            "x-api-key": api_key,
            "User-Agent": "olmo-eval-s2-rate-limit-check/1.0",
        },
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read()
            return Result(
                status=response.status,
                latency=time.monotonic() - started,
                retry_after=response.headers.get("Retry-After"),
                rate_headers=response_rate_headers(response.headers),
            )
    except HTTPError as exc:
        body = exc.read(500).decode("utf-8", errors="replace")
        return Result(
            status=exc.code,
            latency=time.monotonic() - started,
            retry_after=exc.headers.get("Retry-After"),
            rate_headers=response_rate_headers(exc.headers),
            error=body,
        )
    except (URLError, TimeoutError, OSError) as exc:
        return Result(
            status=None,
            latency=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_phase(
    *,
    rate: float,
    duration: float,
    url: str,
    api_key: str,
    timeout: float,
    workers: int,
) -> list[Result]:
    request_count = max(1, round(rate * duration))
    interval = 1.0 / rate
    started = time.monotonic()
    futures: list[concurrent.futures.Future[Result]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for index in range(request_count):
            scheduled = started + index * interval
            delay = scheduled - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            futures.append(executor.submit(request_once, url, api_key, timeout))
        return [future.result() for future in futures]


def summarize(rate: float, duration: float, results: list[Result]) -> bool:
    counts: dict[str, int] = {}
    for result in results:
        label = "network-error" if result.status is None else str(result.status)
        counts[label] = counts.get(label, 0) + 1

    latencies_ms = [result.latency * 1_000 for result in results]
    ok = sum(1 for result in results if result.status is not None and 200 <= result.status < 300)
    rate_limited = sum(result.status == 429 for result in results)
    other_failures = len(results) - ok - rate_limited

    print(f"\n{rate:g} req/s for {duration:g}s ({len(results)} requests)")
    print(f"  statuses: {json.dumps(counts, sort_keys=True)}")
    print(f"  success: {ok}/{len(results)}; 429: {rate_limited}; other: {other_failures}")
    print(
        "  latency: "
        f"p50={statistics.median(latencies_ms):.0f}ms "
        f"p95={percentile(latencies_ms, 0.95):.0f}ms "
        f"max={max(latencies_ms):.0f}ms"
    )

    retry_after = sorted(
        {result.retry_after for result in results if result.retry_after is not None}
    )
    if retry_after:
        print(f"  Retry-After values: {', '.join(retry_after)}")

    rate_headers = next((result.rate_headers for result in results if result.rate_headers), ())
    if rate_headers:
        print("  rate-limit headers: " + ", ".join(f"{k}={v}" for k, v in rate_headers))

    errors = [result.error for result in results if result.error]
    if errors and other_failures:
        print(f"  first non-429 error: {errors[0][:300]}")

    return rate_limited == 0 and other_failures == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rates",
        type=parse_rates,
        default=parse_rates("1,5,10"),
        help="comma-separated request rates to test (default: 1,5,10)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="seconds per rate phase (default: 3)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="seconds between rate phases (default: 2)",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--query",
        default="large language model evaluation",
        help="search query used for every request",
    )
    args = parser.parse_args()

    if args.duration <= 0 or args.cooldown < 0 or args.timeout <= 0 or args.workers <= 0:
        parser.error("duration, timeout, and workers must be positive; cooldown cannot be negative")

    api_key = os.getenv("S2_API_KEY")
    if not api_key:
        print("error: S2_API_KEY is not set", file=sys.stderr)
        return 2

    url = f"{S2_SEARCH_URL}?{urlencode({'query': args.query, 'limit': 1, 'fields': 'paperId'})}"
    total = sum(max(1, round(rate * args.duration)) for rate in args.rates)
    print(f"Semantic Scholar rate-limit probe: {args.rates} req/s, {total} total requests")
    print("API key found in S2_API_KEY (value intentionally hidden)")

    all_clean = True
    for index, rate in enumerate(args.rates):
        if index:
            time.sleep(args.cooldown)
        results = run_phase(
            rate=rate,
            duration=args.duration,
            url=url,
            api_key=api_key,
            timeout=args.timeout,
            workers=args.workers,
        )
        all_clean = summarize(rate, args.duration, results) and all_clean

    return 0 if all_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
