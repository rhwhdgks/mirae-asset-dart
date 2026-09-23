#!/usr/bin/env python3
"""Run one explicit JSONL question set through the full `/answer` API.

This runner is for curated sets that do not have a release-specific runner.
It preserves every response body and refuses duplicate IDs, blank questions,
an unexpected row count, or an existing output path.  Gold70 must continue to
use ``run_final70_live_full.py`` because that runner also verifies its release
manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_edge43_live import invoke  # noqa: E402


def _questions(path: Path, *, expected_count: int) -> list[tuple[str, str]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs = [
        (str(row.get("question_id") or row.get("id") or ""),
         str(row.get("question") or ""))
        for row in rows
    ]
    if (len(pairs) != expected_count
            or len({question_id for question_id, _ in pairs}) != expected_count
            or any(not question_id or not question for question_id, question in pairs)):
        raise ValueError(
            f"question inventory must contain {expected_count} unique nonblank rows")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--retry-503", type=int, default=0)
    parser.add_argument("--retry-wait-s", type=float, default=3.0)
    args = parser.parse_args()
    if (args.expected_count <= 0 or args.timeout_s <= 0
            or args.retry_503 < 0 or args.retry_wait_s < 0):
        parser.error("count/timeout/retry values are invalid")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    questions = _questions(args.questions, expected_count=args.expected_count)
    ready = httpx.get(f"{args.base_url.rstrip('/')}/readyz", timeout=10.0)
    if ready.status_code != 200:
        raise RuntimeError(
            f"server is not ready: HTTP {ready.status_code} {ready.text[:300]}")
    ready_stage1 = ready.json().get("stage1", {})
    if ready_stage1.get("native") is not True or ready_stage1.get("fixture_fallback") is not False:
        raise RuntimeError(
            "live evaluation requires native Stage1 with fixture fallback disabled; "
            f"readyz.stage1={ready_stage1}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=args.timeout_s) as client, args.output.open(
            "x", encoding="utf-8") as stream:
        for index, (question_id, question) in enumerate(questions, start=1):
            result = invoke(
                client, args.base_url, question_id, question,
                retry_503=args.retry_503,
                retry_wait_s=args.retry_wait_s,
            )
            stream.write(json.dumps(
                result, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            print(
                f"[{index:03d}/{len(questions)}] {question_id}: "
                f"HTTP={result['http']} attempts={result['attempt_count']} "
                f"elapsed={result['elapsed']:.3f}s",
                flush=True,
            )
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
