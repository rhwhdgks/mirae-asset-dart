#!/usr/bin/env python3
"""통합 스모크 (HCX 호출 없음).

정본(out/canonical) → Stage2 검색 인덱스 일치 → Stage1 v1(저장된 SemanticIntent 재생, HCX 미호출)
→ QueryPlanHandoff 0.4 → Stage2~4(조회·Evidence 검증·계산·전제검증) → 템플릿 문장화 → /answer 5필드.

    PYTHONPATH=. .venv/bin/python scripts/smoke_integration.py             # 대표 슬라이스 3건
    PYTHONPATH=. .venv/bin/python scripts/smoke_integration.py --all       # fixtures/stage1_v1_vertical_slices 전부
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SLICES = ROOT / "fixtures" / "stage1_v1_vertical_slices"
DEFAULT = ("g_a_001.json", "g_i_004.json", "g_a_010.json")


class _NoHCX:
    def invoke(self, question: str, **_):
        raise AssertionError(f"smoke는 HCX를 호출하지 않는다: {question}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--slice", action="append", dest="slices")
    args = ap.parse_args()

    from app.tools.canonical_env import CanonicalNotBuilt, canonical_status, read_model, search_index
    st = canonical_status()
    print(f"canonical  build={st['build_id']} schema={st['schema_version']} "
          f"facts={st['counts'].get('facts')} evidence={st['counts'].get('evidence')}")
    rm = read_model()
    try:
        idx = search_index()
        print(f"search idx build={idx.index_build_id} rows={idx.manifest.get('meta_rows')}")
    except CanonicalNotBuilt as e:
        idx = None
        print(f"search idx 없음 — narrative task는 tool_not_implemented로 닫힘 ({e})")

    from agent.semantic_intent_v1 import SemanticIntent
    from agent.stage1_assembly import CORPUS_CUTOFF, REFERENCE_DATE
    from agent.stage1_v1_backend_composition import build_stage1_v1_runtime
    from agent.stage1_v1_service import Stage1V1NativeService
    from app.pipeline import AnswerPipeline
    from app.tools import CanonicalToolBackend
    from server.stage1 import handoff_from_clarification_view

    t0 = time.time()
    db = Path(tempfile.mkdtemp(prefix="smoke-clar-")) / "clarifications.sqlite3"
    service = Stage1V1NativeService(_NoHCX(), build_stage1_v1_runtime(
        rm, reference_date=REFERENCE_DATE, corpus_cutoff=CORPUS_CUTOFF, clarification_db_path=db))
    pipeline = AnswerPipeline(CanonicalToolBackend(rm, idx), use_hcx=False)
    print(f"assembled  stage1 runtime + stage2 backend ({time.time() - t0:.1f}s)")

    names = sorted(p.name for p in SLICES.glob("*.json")) if args.all else (args.slices or list(DEFAULT))
    failures = 0
    for name in names:
        row = json.loads((SLICES / name).read_text(encoding="utf-8"))
        intent = SemanticIntent.model_validate(row["intent"], strict=True)
        t1 = time.time()
        result = service.start_from_intent(row["question_id"], row["question"], intent)
        handoff = result.handoff
        if handoff is None and result.view is not None and result.view.clarification is not None:
            handoff = handoff_from_clarification_view(result.view)
        if handoff is None:
            failures += 1
            print(f"FAIL {row['question_id']}: stage1={result.status} handoff 없음")
            continue
        resp, payload = pipeline.run(handoff, question_id=row["question_id"], question=row["question"])
        ok = set(resp) == {"question_id", "question", "retrieved_context", "think_trace", "answer"} \
            and resp["answer"].strip()
        failures += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {row['question_id']}: stage1={result.status} "
              f"handoff={handoff.status} stage2={payload.final_status} claims={len(payload.claims)} "
              f"limitations={[l.code for l in payload.limitations]} ({time.time() - t1:.2f}s)")
        print("     answer: " + resp["answer"].replace("\n", " ")[:160])
    print("SMOKE_PASS" if not failures else f"SMOKE_FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
