#!/usr/bin/env python3
"""Run the pinned final70 through the full API and stream every response.

The runner verifies the release-manifest binding before the first request and
writes one JSONL row immediately after each response.  Each row preserves the
complete five-field API body, HTTP status, elapsed time, transport error and
all attempts.  The default performs exactly one HTTP request per question;
provider-internal retries remain the server's responsibility.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_stage1_final70 import (  # noqa: E402
    QUESTIONS_PATH,
    verify_live_question_input,
)
from scripts.run_edge43_live import invoke  # noqa: E402


def _questions() -> list[tuple[str, str]]:
    pinned_digest = verify_live_question_input()
    payload = QUESTIONS_PATH.read_bytes()
    if sha256(payload).hexdigest() != pinned_digest:
        raise ValueError("final70 question bytes changed after verification")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    pairs = [(str(row.get("question_id") or ""),
              str(row.get("question") or "")) for row in rows]
    if (len(pairs) != 70 or len({qid for qid, _question in pairs}) != 70
            or any(not qid or not question for qid, question in pairs)):
        raise ValueError("final70 question inventory must be 70 unique rows")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--retry-503", type=int, default=0)
    parser.add_argument("--retry-wait-s", type=float, default=3.0)
    args = parser.parse_args()
    if (args.timeout_s <= 0 or args.retry_503 < 0
            or args.retry_wait_s < 0):
        parser.error("timeout/retry values are invalid")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    questions = _questions()
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
                f"[{index:02d}/70] {question_id}: HTTP={result['http']} "
                f"attempts={result['attempt_count']} "
                f"elapsed={result['elapsed']:.3f}s",
                flush=True,
            )
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
