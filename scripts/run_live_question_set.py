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
import math
from pathlib import Path
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_edge43_live import invoke  # noqa: E402
from scripts.live_run_evidence import (  # noqa: E402
    LiveRunEvidence, collect_diagnostics, runner_identity, safe_base_url,
)


def _questions(path: Path, *, expected_count: int) -> list[tuple[str, str]]:
    return _question_pairs(path.read_bytes(), expected_count=expected_count)


def _question_pairs(raw: bytes, *, expected_count: int) -> list[tuple[str, str]]:
    rows = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    pairs = [
        (str(row.get("question_id") or row.get("id") or ""),
         str(row.get("question") or ""))
        for row in rows
    ]
    if (len(pairs) != expected_count
            or len({question_id for question_id, _ in pairs}) != expected_count
            or any(not question_id.strip() or not question.strip() for question_id, question in pairs)):
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
    parser.add_argument("--run-label", default="curated-live")
    args = parser.parse_args()
    if (args.expected_count <= 0 or not math.isfinite(args.timeout_s) or args.timeout_s <= 0
            or args.retry_503 < 0 or not math.isfinite(args.retry_wait_s) or args.retry_wait_s < 0):
        parser.error("count/timeout/retry values are invalid")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    args.base_url = safe_base_url(args.base_url)
    question_bytes = args.questions.read_bytes()
    questions = _question_pairs(question_bytes, expected_count=args.expected_count)
    ready = httpx.get(f"{args.base_url.rstrip('/')}/readyz", timeout=10.0)
    if ready.status_code != 200:
        raise RuntimeError(
            f"server is not ready: HTTP {ready.status_code} {ready.text[:300]}")
    ready_stage1 = ready.json().get("stage1", {})
    if ready_stage1.get("native") is not True or ready_stage1.get("fixture_fallback") is not False:
        raise RuntimeError(
            "live evaluation requires native Stage1 with fixture fallback disabled; "
            f"readyz.stage1={ready_stage1}")
    if ready.json().get("latency_capture") is not True:
        raise RuntimeError("restart the server with ANSWER_CAPTURE_LATENCY=1 before live recording")

    manifest = {"run_label": args.run_label, "expected_count": args.expected_count,
                "base_url": args.base_url, "timeout_s": args.timeout_s,
                "retry_503": args.retry_503, "retry_wait_s": args.retry_wait_s,
                "runner_code": runner_identity(ROOT), "server_ready": ready.json(),
                "server_code_revision": (ready.json().get("code_identity") or {}).get("git_head"),
                "notes": ["runner git identity is not remote server revision",
                          "missing timing/cache fields are unknown, not zero or a cache miss",
                          "provider_http events currently describe Stage1, not composer HTTP calls"]}
    with httpx.Client(timeout=args.timeout_s) as client, LiveRunEvidence(
            args.output, question_bytes=question_bytes, manifest=manifest) as evidence:
        for index, (question_id, question) in enumerate(questions, start=1):
            result = invoke(
                client, args.base_url, question_id, question,
                retry_503=args.retry_503,
                retry_wait_s=args.retry_wait_s,
                capture_diagnostics=True,
            )
            # Flush raw outcomes before a diagnostic lookup can fail.
            evidence.save_response(result)
            diagnostics = collect_diagnostics(client, args.base_url, result)
            evidence.save_diagnostics(diagnostics)
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
