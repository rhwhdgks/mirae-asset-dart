"""입력 어댑터 — 팀원 v0.4 fixture/handoff를 읽어 typed 객체로 로드한다.

2~4단계는 질문 원문을 입력으로 받지 않는다. HandoffRecord.question은 리포트 표시용일 뿐
Orchestrator에 전달되지 않는다 (adapter가 구조적으로 분리).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:          # `agent.*`(Stage1 계약)·`src.canonical.*`(정본)은 저장소 루트 패키지
    sys.path.insert(0, str(ROOT))

from agent.query_plan import (  # noqa: E402
    AnswerRequirement, QueryPlanHandoff, QueryPlanHandoffFixtureV04, QuestionFixtureV04,
)

# 최종 release(`query-plan-v04-final/2026.08.30.1`, fixtures/query_plan_v04_final/release_manifest.json)를 소비한다.
# fixtures/query_plan_v04/ 는 frozen archive — 직접 읽지 않는다.
FIXTURE_DIR = ROOT / "fixtures" / "query_plan_v04_final"


@dataclass
class HandoffRecord:
    question_id: str
    handoff: QueryPlanHandoff
    migration_notes: tuple[str, ...] = ()
    question: str | None = None            # 표시용 — Orchestrator에 넘기지 않음
    semantic_id: str | None = None
    group: str | None = None
    fixture_reference_date: str | None = None

    @property
    def status(self) -> str:
        return self.handoff.status


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_handoffs(fixture_dir: Path = FIXTURE_DIR, *, with_questions: bool = True) -> list[HandoffRecord]:
    """query_plan_handoffs_v0.4.jsonl → HandoffRecord 목록 (v0.4 계약으로 strict 검증)."""
    questions: dict[str, QuestionFixtureV04] = {}
    if with_questions:
        for row in _read_jsonl(fixture_dir / "questions_v0.4.jsonl"):
            q = QuestionFixtureV04.model_validate(row)
            questions[q.question_id] = q

    records: list[HandoffRecord] = []
    for row in _read_jsonl(fixture_dir / "query_plan_handoffs_v0.4.jsonl"):
        fx = QueryPlanHandoffFixtureV04.model_validate(row)
        q = questions.get(fx.question_id)
        records.append(HandoffRecord(
            question_id=fx.question_id,
            handoff=fx.handoff,
            migration_notes=fx.migration_notes,
            question=q.question if q else None,
            semantic_id=q.semantic_id if q else None,
            group=q.group if q else None,
            fixture_reference_date=q.reference_date.isoformat() if q else None,
        ))
    return records


def load_answer_requirements(fixture_dir: Path = FIXTURE_DIR) -> dict[str, AnswerRequirement]:
    out: dict[str, AnswerRequirement] = {}
    for row in _read_jsonl(fixture_dir / "answer_requirements_v0.4.jsonl"):
        ar = AnswerRequirement.model_validate(row)
        out[ar.question_id] = ar
    return out


def load_handoff_json(payload: dict) -> QueryPlanHandoff:
    """서버 경로: 1단계가 보낸 JSON 한 건을 계약으로 검증해 반환."""
    return QueryPlanHandoff.model_validate(payload)
