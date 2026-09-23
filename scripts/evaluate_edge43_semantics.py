#!/usr/bin/env python3
"""Digest-bound human semantic evaluation for EDGE-002..EDGE-044.

HTTP success and ``final_status`` are mechanical facts; neither proves that an
answer fulfilled the question.  This tool binds a human verdict to the exact
stored response so an older 18/10/15 judgment cannot be silently reused after
a new live run.

With ``--init`` it prints a review JSONL template to stdout.  With
``--reviews`` it validates every response digest and prints the verdict counts.
``--require-all-pass`` is the release gate for a claimed semantic 43/43.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_IDS = tuple(f"EDGE-{number:03d}" for number in range(2, 45))
VERDICTS = {"pass", "partial", "fail"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def response_digest(row: dict[str, Any]) -> str:
    bound = {
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "http": row.get("http"),
        "response": row.get("response"),
        "error": row.get("error"),
    }
    payload = json.dumps(
        bound, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def indexed_responses(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_id = {str(row.get("question_id")): row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError("EDGE43 response question_id가 중복되었습니다")
    actual = tuple(sorted(by_id))
    if actual != EXPECTED_IDS:
        missing = sorted(set(EXPECTED_IDS) - set(actual))
        extra = sorted(set(actual) - set(EXPECTED_IDS))
        raise ValueError(f"EDGE43 response inventory 불일치 missing={missing} extra={extra}")
    return by_id


def review_template(responses: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "question_id": qid,
        "response_sha256": response_digest(responses[qid]),
        "verdict": "unreviewed",
        "reason": "",
    } for qid in EXPECTED_IDS]


def validate_reviews(
        responses: dict[str, dict[str, Any]], reviews_path: Path,
        ) -> dict[str, int]:
    rows = read_jsonl(reviews_path)
    by_id = {str(row.get("question_id")): row for row in rows}
    if len(rows) != len(by_id) or tuple(sorted(by_id)) != EXPECTED_IDS:
        raise ValueError("의미 평가가 EDGE-002..044를 정확히 한 번씩 포함해야 합니다")
    counts = {verdict: 0 for verdict in sorted(VERDICTS)}
    for qid in EXPECTED_IDS:
        review = by_id[qid]
        expected_digest = response_digest(responses[qid])
        if review.get("response_sha256") != expected_digest:
            raise ValueError(f"{qid} 의미 판정이 현재 응답과 결속되지 않았습니다")
        verdict = review.get("verdict")
        if verdict not in VERDICTS:
            raise ValueError(f"{qid} verdict가 잘못되었습니다: {verdict!r}")
        if not isinstance(review.get("reason"), str) or not review["reason"].strip():
            raise ValueError(f"{qid} 의미 판정 사유가 비었습니다")
        counts[verdict] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--require-all-pass", action="store_true")
    args = parser.parse_args()
    responses = indexed_responses(args.responses)
    if args.init:
        for row in review_template(responses):
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
        return 0
    if args.reviews is None:
        parser.error("--reviews 또는 --init 중 하나가 필요합니다")
    counts = validate_reviews(responses, args.reviews)
    print(json.dumps({"total": sum(counts.values()), **counts},
                     ensure_ascii=False, sort_keys=True))
    return 0 if not args.require_all_pass or counts["pass"] == 43 else 1


if __name__ == "__main__":
    raise SystemExit(main())
