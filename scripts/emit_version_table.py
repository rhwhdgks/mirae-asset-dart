#!/usr/bin/env python3
"""계약 version 표를 **소스에서 읽어** `docs/IMPLEMENTATION.md` 에 써 넣는다.

6차 검토 §7.4 는 문서마다 회차·wire version·output 모델 표기가 달라서 외부
검토자가 어떤 계약이 authoritative 인지 잘못 판단한다고 지적했다. 손으로 적은
표는 다음 회차에 또 어긋난다. 그래서 표를 **생성물**로 만든다.

읽는 곳은 상수 하나뿐이다 — 값을 여기에 다시 적지 않는다. 상수가 사라지거나
이름이 바뀌면 import 가 실패하고, 그때 표가 조용히 낡는 대신 빌드가 깨진다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python scripts/emit_version_table.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python scripts/emit_version_table.py --check
```

``--check`` 는 쓰지 않고 최신 여부만 판정한다 (drift 면 exit 1). release 전
게이트로 쓴다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TARGET = ROOT / "docs/IMPLEMENTATION.md"
BEGIN = "<!-- BEGIN generated: scripts/emit_version_table.py -->"
END = "<!-- END generated -->"


def _rows() -> list[tuple[str, str, str]]:
    """(계약, 현재 값, 정의 위치). 값은 전부 import 로만 얻는다."""
    from agent.compiled_answer_contract_v1 import (
        COMPILED_ANSWER_CONTRACT_V1, LIMITATION_REGISTRY_V1)
    from agent.deterministic_plan_compiler_v1 import (
        COMPILED_SLICE_VERSION, EXECUTION_PLAN_VERSION, RESOLUTION_VERSION)
    from agent.hcx_semantic_intent_v1 import HCX_SEMANTIC_INTENT_PROMPT_VERSION
    from agent.hcx_semantic_intent_v1_eval import (
        HCX_SEMANTIC_INTENT_EVAL_GOLD10_ROW_VERSION,
        HCX_SEMANTIC_INTENT_EVAL_SUMMARY_VERSION,
        HCX_SEMANTIC_INTENT_EVAL_TECHNICAL_FAILURE_ROW_VERSION,
    )
    from agent.query_plan import PLAN_PROPOSAL_VERSION, QUERY_PLAN_HANDOFF_VERSION
    from agent.query_plan_v04_final_release import RELEASE_ID
    from agent.semantic_intent_v1 import (
        HCX_SEMANTIC_INTENT_WIRE_V1, STAGE1_SEMANTIC_INTENT_V1)
    from agent.semantic_intent_v1_boundary import SEMANTIC_INTENT_BOUNDARY_V1
    from app.serving.index import FTS_SCHEMA_VERSION, INDEX_BUILDER_VERSION
    from agent.stage1_ready_envelope_v02 import ENVELOPE_VERSION
    from agent.stage1_v1_clarification_session import (
        CLARIFICATION_CONTEXT_VERSION,
        CLARIFICATION_READY_BINDING_VERSION,
        CLARIFICATION_SESSION_VERSION,
    )
    from agent.stage1_v1_outcome import STAGE1_OUTCOME_VERSION
    from agent.stage1_v1_query_plan_v04_emitter import EMITTER_VERSION
    from agent.stage1_v1_resolver import RESOLUTION_DECISION_VERSION
    from agent.stage1_v1_overlay import (
        MANIFEST_SCHEMA_VERSION, ROW_SCHEMA_VERSION)
    from agent.stage1_v1_gold_expectations import (
        MANIFEST_SCHEMA_VERSION as GOLD_EXPECTATION_MANIFEST_VERSION,
        ROW_SCHEMA_VERSION as GOLD_EXPECTATION_ROW_VERSION,
    )
    from src.canonical.schema import SCHEMA_VERSION

    return [
        ("HCX semantic intent wire (7차 v1 shadow)",
         HCX_SEMANTIC_INTENT_WIRE_V1, "`agent/semantic_intent_v1.py`"),
        ("normalized semantic intent (7차 v1 shadow)",
         STAGE1_SEMANTIC_INTENT_V1, "`agent/semantic_intent_v1.py`"),
        ("deterministic semantic boundary (7차 v1 shadow, offline)",
         SEMANTIC_INTENT_BOUNDARY_V1,
         "`agent/semantic_intent_v1_boundary.py`"),
        ("compiled answer contract (7차 v1 shadow)",
         COMPILED_ANSWER_CONTRACT_V1,
         "`agent/compiled_answer_contract_v1.py`"),
        ("limitation registry (7차 v1 shadow)", LIMITATION_REGISTRY_V1,
         "`agent/compiled_answer_contract_v1.py`"),
        ("authoritative resolution (7차 v1 shadow)", RESOLUTION_VERSION,
         "`agent/deterministic_plan_compiler_v1.py`"),
        ("execution plan (7차 v1 shadow)", EXECUTION_PLAN_VERSION,
         "`agent/deterministic_plan_compiler_v1.py`"),
        ("deterministic compiled slice (7차 v1 shadow, 8 slices)",
         COMPILED_SLICE_VERSION,
         "`agent/deterministic_plan_compiler_v1.py`"),
        ("resolution decision (7차 v1 shadow)",
         RESOLUTION_DECISION_VERSION, "`agent/stage1_v1_resolver.py`"),
        ("four-state Stage1 outcome (7차 v1 shadow)",
         STAGE1_OUTCOME_VERSION, "`agent/stage1_v1_outcome.py`"),
        ("clarification session (7차 v1 shadow)",
         CLARIFICATION_SESSION_VERSION,
         "`agent/stage1_v1_clarification_session.py`"),
        ("clarification resolution context (7차 v1 shadow)",
         CLARIFICATION_CONTEXT_VERSION,
         "`agent/stage1_v1_clarification_session.py`"),
        ("clarification ready binding (7차 v1 shadow)",
         CLARIFICATION_READY_BINDING_VERSION,
         "`agent/stage1_v1_clarification_session.py`"),
        ("v1 -> QueryPlanHandoff 0.4 final query-plan emitter",
         EMITTER_VERSION,
         "`agent/stage1_v1_query_plan_v04_emitter.py`"),
        ("QueryPlanHandoff 0.4 final fixture release",
         RELEASE_ID, "`agent/query_plan_v04_final_release.py`"),
        ("Gold correction overlay manifest (7차 v1 shadow)",
         MANIFEST_SCHEMA_VERSION, "`agent/stage1_v1_overlay.py`"),
        ("Gold correction overlay row (7차 v1 shadow)",
         ROW_SCHEMA_VERSION, "`agent/stage1_v1_overlay.py`"),
        ("Gold expectation manifest (7차 v1 shadow, 10/10)",
         GOLD_EXPECTATION_MANIFEST_VERSION,
         "`agent/stage1_v1_gold_expectations.py`"),
        ("Gold expectation row (7차 v1 shadow, 10/10)",
         GOLD_EXPECTATION_ROW_VERSION,
         "`agent/stage1_v1_gold_expectations.py`"),
        ("HCX semantic eval summary (7차 v1 shadow)",
         HCX_SEMANTIC_INTENT_EVAL_SUMMARY_VERSION,
         "`agent/hcx_semantic_intent_v1_eval.py`"),
        ("HCX Gold10 result row evidence (7차 v1 shadow)",
         HCX_SEMANTIC_INTENT_EVAL_GOLD10_ROW_VERSION,
         "`agent/hcx_semantic_intent_v1_eval.py`"),
        ("HCX technical failure row evidence (7차 v1 shadow)",
         HCX_SEMANTIC_INTENT_EVAL_TECHNICAL_FAILURE_ROW_VERSION,
         "`agent/hcx_semantic_intent_v1_eval.py`"),
        ("Stage1 ready envelope (7차 v1 shadow, 8-slice path)", ENVELOPE_VERSION,
         "`agent/stage1_ready_envelope_v02.py`"),
        ("HCX-007 system prompt (현행 — SemanticIntent v1)",
         HCX_SEMANTIC_INTENT_PROMPT_VERSION, "`agent/hcx_semantic_intent_v1.py`"),
        ("PlanProposal", PLAN_PROPOSAL_VERSION, "`agent/query_plan.py`"),
        ("QueryPlan handoff (Stage2 인계)", QUERY_PLAN_HANDOFF_VERSION,
         "`agent/query_plan.py`"),
        ("canonical schema", SCHEMA_VERSION, "`src/canonical/schema.py`"),
        ("Stage2 chunk FTS schema", FTS_SCHEMA_VERSION, "`app/serving/index.py`"),
        ("Stage2 search index builder", INDEX_BUILDER_VERSION, "`app/serving/index.py`"),
    ]


def render() -> str:
    lines = ["| 계약 | 현재 값 | 정의 위치 |", "|---|---|---|"]
    for name, value, where in _rows():
        lines.append(f"| {name} | `{value}` | {where} |")
    return "\n".join(lines)


def splice(text: str, block: str) -> str:
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{TARGET} 에 생성 블록 표시가 없습니다: {BEGIN}")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{BEGIN}\n{block}\n{END}{tail}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="쓰지 않고 최신 여부만 판정한다 (drift 면 exit 1)")
    args = ap.parse_args()

    current = TARGET.read_text(encoding="utf-8")
    updated = splice(current, render())

    if args.check:
        if current != updated:
            print(f"drift: {TARGET.relative_to(ROOT)} 가 소스 상수와 다릅니다")
            return 1
        print(f"ok: {TARGET.relative_to(ROOT)}")
        return 0

    if current == updated:
        print(f"unchanged: {TARGET.relative_to(ROOT)}")
        return 0
    TARGET.write_text(updated, encoding="utf-8")
    print(f"wrote: {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
