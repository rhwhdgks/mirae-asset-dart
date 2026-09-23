"""실행 중인 평가 API의 end-to-end P50/P95를 본문 비노출로 측정한다."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time

import httpx
from pydantic import ValidationError

from agent.evaluation_api import EvaluationResponse


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[index], 3)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--question", default="삼성전자의 2025년 연결 매출액은 얼마야?")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=65.0)
    args = parser.parse_args()
    if not 1 <= args.requests <= 10_000:
        parser.error("--requests는 1..10000이어야 합니다")
    if not 1 <= args.concurrency <= 64:
        parser.error("--concurrency는 1..64여야 합니다")
    if not 0 <= args.warmup <= 100:
        parser.error("--warmup은 0..100이어야 합니다")
    if (not math.isfinite(args.timeout_seconds)
            or not 0 < args.timeout_seconds <= 300):
        parser.error("--timeout-seconds는 0 초과, 300 이하여야 합니다")
    if not args.question.strip() or len(args.question) > 10_000:
        parser.error("--question 길이가 잘못되었습니다")

    base_url = httpx.URL(args.url)
    if base_url.scheme not in {"http", "https"} or not base_url.host:
        parser.error("--url은 http(s) URL이어야 합니다")

    latencies: list[float] = []
    errors: Counter[str] = Counter()
    timeout = httpx.Timeout(args.timeout_seconds)
    limits = httpx.Limits(
        max_connections=max(2, args.concurrency),
        max_keepalive_connections=max(2, args.concurrency))
    with httpx.Client(
            base_url=str(base_url), timeout=timeout, limits=limits,
            follow_redirects=False) as client:
        try:
            readiness = client.get("/readyz")
        except httpx.HTTPError as exc:
            print(json.dumps({
                "status": "unreachable", "error_type": type(exc).__name__,
            }, sort_keys=True))
            return 2
        if readiness.status_code != 200:
            print(json.dumps({
                "status": "not_ready", "http_status": readiness.status_code,
            }, sort_keys=True))
            return 2

        def invoke(ordinal: int, *, measured: bool) -> None:
            started = time.perf_counter()
            try:
                response = client.get("/answer", params={
                    "question_id": f"BENCH-{ordinal:06d}",
                    "question": args.question,
                })
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if response.status_code != 200:
                    errors[f"http_{response.status_code}"] += 1
                    return
                parsed = EvaluationResponse.model_validate(
                    response.json(), strict=True)
                if parsed.question != args.question:
                    errors["question_echo_mismatch"] += 1
                    return
                if measured:
                    latencies.append(elapsed_ms)
            except (httpx.HTTPError, ValidationError, ValueError) as exc:
                errors[type(exc).__name__] += 1

        for ordinal in range(args.warmup):
            invoke(ordinal, measured=False)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            list(executor.map(
                lambda ordinal: invoke(ordinal + args.warmup, measured=True),
                range(args.requests)))
        wall_ms = (time.perf_counter() - started) * 1000.0

    report = {
        "status": "pass" if len(latencies) == args.requests and not errors else "fail",
        "requests": args.requests,
        "successes": len(latencies),
        "errors": dict(sorted(errors.items())),
        "concurrency": args.concurrency,
        "wall_ms": round(wall_ms, 3),
        "latency_ms": {
            "min": round(min(latencies), 3) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 3) if latencies else None,
        },
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
