#!/usr/bin/env python3
"""실행 중인 평가 서버에 최종 release 70문항(+옵션 추가 질문)을 순차 전송하고 결과를 기록한다.

서버 경로 전체(Stage1 HCX-007 → Stage2~4 → 5필드)를 실제로 태우므로 HCX 비용이 발생한다
(문항당 HCX-007 1회 + answer 문항 HCX-005 1회, 실측 ≈ 7원/호출).

    PYTHONPATH=. .venv/bin/python scripts/run_server_eval.py --url http://127.0.0.1:8000 --out out/server_eval.jsonl
    PYTHONPATH=. .venv/bin/python scripts/run_server_eval.py --only G-A-001 R-A-005 --extra "현대차 작년 영업이익 얼마야?"

응답 5필드·지연을 JSONL로 남기고, 서버 요청 로그(out/requests/answers_*.jsonl)의 Stage1 meta와
question_id로 join해 경로별(native/fixture) 분포와 handoff/final status를 요약한다.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIELDS = {"question_id", "question", "retrieved_context", "think_trace", "answer"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--extra", action="append", default=[], help="release 밖 추가 질문 (반복 가능)")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    from app.orchestrator import load_handoffs
    items = [(r.question_id, r.question, r.handoff.status) for r in load_handoffs() if r.question]
    if args.only:
        items = [it for it in items if it[0] in set(args.only)]
    for i, q in enumerate(args.extra, 1):
        items.append((f"X-{i:03d}", q, None))

    ready = httpx.get(f"{args.url}/readyz", timeout=10)
    print("readyz:", ready.status_code, ready.text[:300])
    if ready.status_code != 200:
        return 1
    ready_stage1 = ready.json().get("stage1", {})
    if ready_stage1.get("native") is not True or ready_stage1.get("fixture_fallback") is not False:
        print("refusing non-native or fixture-enabled server:", ready_stage1)
        return 2

    rows, times = [], []
    started = datetime.now(timezone.utc)
    for qid, q, expected in items:
        t0 = time.time()
        try:
            r = httpx.get(f"{args.url}/answer", params={"question_id": qid, "question": q}, timeout=args.timeout)
            body = r.json(); code = r.status_code
        except Exception as e:  # noqa
            body = {"error": f"{type(e).__name__}: {e}"}; code = -1
        dt = round(time.time() - t0, 3); times.append(dt)
        trace_head = (body.get("think_trace") or "").split("\n", 1)[0]
        ok = code == 200 and set(body) == FIELDS and bool((body.get("answer") or "").strip())
        rows.append({"question_id": qid, "question": q, "release_status": expected, "http": code, "ok": ok,
                     "elapsed": dt, "stage1_line": trace_head, "answer": body.get("answer"),
                     "think_trace": body.get("think_trace"), "retrieved_context": body.get("retrieved_context")})
        print(f"{'ok  ' if ok else 'FAIL'} {qid:<8} {dt:6.2f}s  {trace_head[:110]}")
        print(f"     {str(body.get('answer') or body)[:150].replace(chr(10), ' ')}")

    # 서버 요청 로그와 join (같은 호스트에서 돌릴 때)
    meta_by_qid: dict[str, dict] = {}
    for p in sorted((ROOT / "out" / "requests").glob("answers_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("ts", "") >= started.isoformat()[:19]:
                meta_by_qid[rec.get("question_id")] = rec
    src = Counter(); hs = Counter(); fs = Counter(); hcx = 0
    for row in rows:
        m = meta_by_qid.get(row["question_id"], {})
        row["server_meta"] = {k: m.get(k) for k in ("stage1", "matched", "handoff_status", "final_status", "t_stage1", "t_pipeline", "limitations")}
        s1 = m.get("stage1_meta") or {}
        row["server_meta"]["hcx_request_id"] = s1.get("request_id"); row["server_meta"]["hcx_tokens"] = s1.get("total_tokens")
        src[m.get("stage1", "?")] += 1; hs[m.get("handoff_status", "?")] += 1; fs[m.get("final_status", "?")] += 1
        hcx += 1 if s1.get("request_id") else 0

    ts = sorted(times)
    print("\n══ 요약")
    print(f"  문항 {len(rows)} · 5필드 OK {sum(r['ok'] for r in rows)} · HCX-007 호출 {hcx}")
    print(f"  지연 P50 {statistics.median(ts):.2f}s · P95 {ts[int(len(ts)*0.95)] if len(ts) > 1 else ts[0]:.2f}s · max {ts[-1]:.2f}s")
    print(f"  stage1 경로 {dict(src)}\n  handoff status {dict(hs)}\n  final status {dict(fs)}")
    # release status와 native handoff status 비교
    diff = [(r["question_id"], r["release_status"], r["server_meta"]["handoff_status"]) for r in rows
            if r["release_status"] and r["server_meta"]["handoff_status"] and r["release_status"] != r["server_meta"]["handoff_status"]]
    print(f"  release status와 다른 handoff {len(diff)}건: {diff}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("→", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
