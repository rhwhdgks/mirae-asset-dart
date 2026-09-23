#!/usr/bin/env python3
"""Run an edge_gap batch live, repeating each question to expose variance.

`HCX-007` intent wobbles between calls (issue #19) and both full 293-case runs
so far failed **exactly one** case to that wobble.  A single green call is
therefore not evidence.  The set's README requires three identical results
before a case counts as passing, so this runner repeats every question and
keeps each attempt's whole body.

Question bytes are checked against `SHA256SUMS` before the first call: a set
edited after its answer key was bound is no longer the set that was
preregistered.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_edge43_live import invoke


def verify_checksums(questions: Path) -> None:
    sums = questions.parent / "SHA256SUMS"
    if not sums.exists():
        raise SystemExit(f"{sums} 가 없어 질문 결속을 확인할 수 없습니다")
    expected = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        expected[name.strip().lstrip("*")] = digest
    want = expected.get(questions.name)
    if want is None:
        raise SystemExit(f"SHA256SUMS 에 {questions.name} 항목이 없습니다")
    got = hashlib.sha256(questions.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(
            f"{questions.name} 이 결속된 바이트와 다릅니다\n"
            f"  기대 {want}\n  실제 {got}")


def read_questions(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid, text = str(row["question_id"]), str(row["question"])
        if not qid or not text:
            raise SystemExit(f"질문 원천이 비어 있습니다: {row!r}")
        rows.append((qid, text))
    if len({qid for qid, _ in rows}) != len(rows):
        raise SystemExit("question_id 가 중복됩니다")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8031")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--retry-503", type=int, default=3)
    parser.add_argument("--retry-wait-s", type=float, default=3.0)
    parser.add_argument("--timeout-s", type=float, default=70.0)
    parser.add_argument("--skip-checksum", action="store_true")
    # 서버는 `MIRAE_API_MAX_CONCURRENCY`(기본 4) 를 넘는 in-flight 요청을 503
    # 으로 막는다. 한도 아래로 두면 답변 내용은 그대로면서 벽시계만 줄어든다.
    # **지연 수치는 동시 실행에서 의미를 잃는다.** 공식 기준선을 남길 때는
    # --concurrency 1 로 다시 돌린다.
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1 or args.retry_503 < 0 or args.timeout_s <= 0:
        parser.error("repeat/retry/timeout 값이 잘못되었습니다")
    if not 1 <= args.concurrency <= 4:
        parser.error("concurrency 는 1..4 여야 합니다 (서버 admission 한도)")

    if not args.skip_checksum:
        verify_checksums(args.questions)
    rows = read_questions(args.questions)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(qid, question, call)
            for qid, question in rows
            for call in range(1, args.repeat + 1)]
    write_lock = threading.Lock()
    local = threading.local()
    started = time.monotonic()
    done = 0

    with args.output.open("w", encoding="utf-8") as stream:
        def run_one(job):
            nonlocal done
            qid, question, call = job
            client = getattr(local, "client", None)
            if client is None:
                client = httpx.Client(timeout=args.timeout_s)
                local.client = client
            row = invoke(client, args.base_url, qid, question,
                         retry_503=args.retry_503,
                         retry_wait_s=args.retry_wait_s)
            row["call"] = call
            with write_lock:
                stream.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                done += 1
                elapsed = row["elapsed"] or 0.0
                print(f"[{done}/{len(jobs)}] {qid} call{call}: "
                      f"HTTP={row['http']} {elapsed:.1f}s", flush=True)

        if args.concurrency == 1:
            for job in jobs:
                run_one(job)
        else:
            with ThreadPoolExecutor(args.concurrency) as pool:
                list(pool.map(run_one, jobs))
    print(f"총 {len(rows)}문항 × {args.repeat}회, "
          f"{time.monotonic() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
