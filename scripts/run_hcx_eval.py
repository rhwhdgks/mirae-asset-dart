"""HCX composer 평가 러너 — 70문항 handoff → AnswerPipeline(use_hcx) → 채택률·문장 채점 기록.

out/hcx_compose_run1~3.json과 같은 스키마(qid/status/hcx/path/t/text_ok/answer)로 저장해 런 간 비교가 가능하다.
text_ok = app.evals.scorer.score_answer_text(필수 값·근거 접수번호 포함, 금지 값 미포함) 전항 통과.

사용:
  ./.venv/bin/python scripts/run_hcx_eval.py --out out/hcx_compose_run4.json      # 전체 70문항 실호출
  ./.venv/bin/python scripts/run_hcx_eval.py --only G-A-001 G-A-005               # 일부만 (스모크, --out 생략 시 저장 안 함)
  ./.venv/bin/python scripts/run_hcx_eval.py --rescore out/hcx_compose_run*.json  # 저장된 run의 text_ok 재계산만 (HCX 호출 없음)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app.env  # noqa: F401  (.env 자동 로드)
from app.orchestrator import load_answer_requirements, load_handoffs  # noqa: E402
from app.evals.scorer import score_answer_text  # noqa: E402


def text_checks(answer: str, req) -> tuple[bool, list[str]]:
    checks = score_answer_text(answer or "", req, None)
    return all(c.passed for c in checks), [f"{c.name}:{'ok' if c.passed else c.detail}" for c in checks]


def run(only: set[str] | None) -> list[dict]:
    from app.pipeline import AnswerPipeline
    from app.tools import CanonicalToolBackend
    reqs = load_answer_requirements()
    pl = AnswerPipeline(CanonicalToolBackend(), use_hcx=True)
    assert pl.composer is not None and pl.composer.enabled, "HCX_API_KEY 미설정 — 실호출 러너는 키가 필요하다"
    rows: list[dict] = []
    for rec in load_handoffs():
        if only and rec.question_id not in only:
            continue
        t0 = time.monotonic()
        resp, payload = pl.run(rec.handoff, question_id=rec.question_id, question=rec.question or "")
        t = round(time.monotonic() - t0, 2)
        compose = [e.summary for e in payload.trace if e.stage == "compose"]
        path = compose[-1] if compose else "?"
        ok, checks = text_checks(resp["answer"], reqs[rec.question_id])
        rows.append({"qid": rec.question_id, "status": payload.final_status,
                     "hcx": path.startswith("hcx-005 composer"), "path": path, "t": t,
                     "retried": any("교정 재시도" in e for e in compose),
                     "text_ok": ok, "checks": checks, "answer": resp["answer"]})
        print(f"{rec.question_id:<9} {payload.final_status:<15} hcx={str(rows[-1]['hcx']):<5} "
              f"text_ok={str(ok):<5} t={t:>5}s  {path[:70]}", flush=True)
        time.sleep(0.3)   # QPM 여유 (테스트 앱 60/분)
    u = pl.composer.client.usage_total
    print(f"\n토큰 사용: {u['calls']}콜, prompt {u['prompt']:,} / completion {u['completion']:,}")
    return rows


def summarize(rows: list[dict]) -> None:
    n = len(rows)
    att = [r for r in rows if r["status"] in ("answer", "partial_answer")]
    hcx = [r for r in att if r["hcx"]]
    print(f"\n요약: {n}문항 | answer/partial {len(att)} | hcx 채택 {len(hcx)}/{len(att)}"
          f" | 교정 재시도 회수 {sum(1 for r in rows if r['retried'] and r['hcx'])}"
          f" | text_ok {sum(1 for r in rows if r['text_ok'])}/{n}"
          f" | 평균 t {round(sum(r['t'] for r in rows)/n, 2)}s")
    for r in rows:
        if r["path"].startswith("hcx 출력 검증 실패"):
            print(f"  폴백: {r['qid']} — {r['path']}")


def rescore(paths: list[str]) -> None:
    """저장된 run 파일의 answer를 현재 채점 기준으로 재채점 — 파일은 수정하지 않는다."""
    reqs = load_answer_requirements()
    for path in paths:
        rows = json.loads(Path(path).read_text())
        ok = sum(text_checks(r.get("answer") or "", reqs[r["qid"]])[0] for r in rows)
        print(f"{path}: text_ok(재채점) {ok}/{len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="결과 JSON 저장 경로 (예: out/hcx_compose_run4.json)")
    ap.add_argument("--only", nargs="*", help="특정 question_id만 실행 (스모크)")
    ap.add_argument("--rescore", nargs="*", help="저장된 run 파일 재채점만 수행 (HCX 호출 없음)")
    a = ap.parse_args()
    if a.rescore:
        rescore(a.rescore)
        return
    rows = run(set(a.only) if a.only else None)
    summarize(rows)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {a.out}")


if __name__ == "__main__":
    main()
