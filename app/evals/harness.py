"""채점 harness — 70문항 handoff → Orchestrator → AnswerPayload → 채점 → 성적표.

사용:
  ./.venv/bin/python -m app.evals.harness            # 성적표 출력
  ./.venv/bin/python -m app.evals.harness --json out.json
"""
from __future__ import annotations

import app.env  # noqa: F401  (.env 자동 로드)
import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.orchestrator import Orchestrator, load_answer_requirements, load_handoffs  # noqa: E402
from app.evals.scorer import score_payload, score_answer_text, ScoreCard  # noqa: E402


@dataclass
class HarnessReport:
    total: int
    passed: int
    by_group: dict[str, tuple[int, int]] = field(default_factory=dict)
    by_check: dict[str, tuple[int, int]] = field(default_factory=dict)   # name → (pass, total)
    failures: list[dict] = field(default_factory=list)
    cards: list[ScoreCard] = field(default_factory=list)

    @property
    def score_pct(self) -> float:
        return round(100.0 * self.passed / self.total, 1) if self.total else 0.0


def run_harness(backend=None, *, only: set[str] | None = None, compose: bool = False) -> HarnessReport:
    records = load_handoffs()
    reqs = load_answer_requirements()
    orch = Orchestrator(backend)
    composer = None
    if compose:
        from app.composer import HcxComposer
        composer = HcxComposer()

    cards: list[ScoreCard] = []
    failures: list[dict] = []
    grp_tot: Counter = Counter()
    grp_pass: Counter = Counter()
    chk_tot: Counter = Counter()
    chk_pass: Counter = Counter()

    for rec in records:
        if only and rec.question_id not in only:
            continue
        req = reqs[rec.question_id]
        try:
            payload = orch.run(rec.handoff, question_id=rec.question_id)
        except Exception as ex:  # 파이프라인 예외도 채점 대상 (failure)
            from app.orchestrator.payload import AnswerPayload, Limitation
            payload = AnswerPayload(question_id=rec.question_id, handoff_id=str(rec.handoff.handoff_id),
                                    final_status="failure",
                                    limitations=[Limitation(code="pipeline_exception",
                                                            detail=f"{type(ex).__name__}: {ex}"[:200])])
        card = score_payload(payload, req)
        if composer is not None:
            text = composer.compose(payload, question=rec.question, trace=payload.trace)
            card.checks.extend(score_answer_text(text, req, payload))
        cards.append(card)
        g = (rec.group or "?").split("_")[0]
        grp_tot[g] += 1
        if card.passed:
            grp_pass[g] += 1
        else:
            failures.append({"question_id": rec.question_id, "group": rec.group,
                             "expected": req.expected_action, "got": payload.final_status,
                             "checks": card.summary})
        for c in card.checks:
            chk_tot[c.name] += 1
            if c.passed:
                chk_pass[c.name] += 1

    return HarnessReport(
        total=len(cards), passed=sum(c.passed for c in cards),
        by_group={g: (grp_pass[g], grp_tot[g]) for g in sorted(grp_tot)},
        by_check={n: (chk_pass[n], chk_tot[n]) for n in sorted(chk_tot)},
        failures=failures, cards=cards,
    )


def print_report(r: HarnessReport) -> None:
    print(f"══ 채점 결과: {r.passed}/{r.total} ({r.score_pct}%)")
    print("── 그룹별")
    for g, (p, t) in r.by_group.items():
        print(f"   {g:<4} {p:>3}/{t:<3}")
    print("── 항목별")
    for n, (p, t) in r.by_check.items():
        print(f"   {n:<15} {p:>3}/{t:<3}")
    if r.failures:
        print(f"── 오답 {len(r.failures)}건")
        for f in r.failures:
            print(f"   {f['question_id']:<8} want={f['expected']:<14} got={f['got']:<14} {f['checks']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="성적표 JSON 저장 경로")
    ap.add_argument("--only", nargs="*", help="특정 question_id만")
    ap.add_argument("--compose", action="store_true", help="composer로 답변 문장까지 만들어 문장 채점 항목 추가")
    ap.add_argument("--backend", choices=["null", "canonical"], default="canonical",
                    help="canonical=실데이터 ToolBackend(기본) / null=미구현 자리표시자")
    a = ap.parse_args()
    backend = None
    if a.backend == "canonical":
        from app.tools import CanonicalToolBackend
        backend = CanonicalToolBackend()
    rep = run_harness(backend, only=set(a.only) if a.only else None, compose=a.compose)
    print_report(rep)
    if a.json:
        Path(a.json).write_text(json.dumps({
            "backend": a.backend,
            "total": rep.total, "passed": rep.passed, "score_pct": rep.score_pct,
            "by_group": rep.by_group, "by_check": rep.by_check, "failures": rep.failures,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {a.json}")


if __name__ == "__main__":
    main()
