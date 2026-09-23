#!/usr/bin/env python3
"""Run EDGE-002..044 sequentially and preserve every HTTP response body.

The previous ad-hoc runner lost the JSON body for HTTP 503 and allowed one
timed-out request to obscure the next two cases.  This runner keeps status,
body, transport error, elapsed time and attempt count.  A 503 is retried only
to isolate queue-cascade effects; it is never rewritten as a successful first
attempt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_edge43_semantics import EXPECTED_IDS, read_jsonl


def questions(path: Path) -> list[tuple[str, str]]:
    rows = read_jsonl(path)
    by_id = {str(row.get("question_id")): str(row.get("question") or "") for row in rows}
    if tuple(sorted(by_id)) != EXPECTED_IDS or any(not by_id[qid] for qid in EXPECTED_IDS):
        raise ValueError("질문 원천은 EDGE-002..044를 정확히 포함해야 합니다")
    return [(qid, by_id[qid]) for qid in EXPECTED_IDS]


def invoke(
        client: httpx.Client, base_url: str, question_id: str, question: str,
        *, retry_503: int, retry_wait_s: float,
        ) -> dict:
    attempts: list[dict] = []
    for attempt in range(1, retry_503 + 2):
        started = time.monotonic()
        try:
            response = client.get(
                f"{base_url.rstrip('/')}/answer",
                params={"question_id": question_id, "question": question})
            elapsed = round(time.monotonic() - started, 3)
            try:
                body = response.json()
            except Exception:  # noqa: BLE001 - preserve non-JSON server body
                body = {"raw_body": response.text}
            attempts.append({"attempt": attempt, "http": response.status_code,
                             "elapsed": elapsed, "response": body})
            if response.status_code != 503 or attempt > retry_503:
                return {
                    "question_id": question_id, "question": question,
                    "http": response.status_code, "elapsed": elapsed,
                    "attempt_count": attempt, "attempts": attempts,
                    "response": body, "error": None,
                }
        except Exception as exc:  # noqa: BLE001 - transport evidence is output
            elapsed = round(time.monotonic() - started, 3)
            attempts.append({"attempt": attempt, "http": None, "elapsed": elapsed,
                             "error": f"{type(exc).__name__}: {exc}"})
            if attempt > retry_503:
                return {
                    "question_id": question_id, "question": question,
                    "http": None, "elapsed": elapsed,
                    "attempt_count": attempt, "attempts": attempts,
                    "response": None, "error": attempts[-1]["error"],
                }
        time.sleep(retry_wait_s)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--retry-503", type=int, default=3)
    parser.add_argument("--retry-wait-s", type=float, default=3.0)
    parser.add_argument("--timeout-s", type=float, default=70.0)
    args = parser.parse_args()
    if args.retry_503 < 0 or args.retry_wait_s < 0 or args.timeout_s <= 0:
        parser.error("retry/timeout 값이 잘못되었습니다")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=args.timeout_s) as client, args.output.open(
            "w", encoding="utf-8") as stream:
        for qid, question in questions(args.questions):
            row = invoke(client, args.base_url, qid, question,
                         retry_503=args.retry_503, retry_wait_s=args.retry_wait_s)
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            print(f"{qid}: HTTP={row['http']} attempts={row['attempt_count']} "
                  f"elapsed={row['elapsed']:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
