"""Canonical schema 1.9 위에 놓이는 공시 Agent 애플리케이션 계층.

이 패키지는 의도적으로 ``src/`` 밖에 있다. ``src/``는 canonical 전처리
``code_hash``의 입력이므로 Agent·검색 구현 변경이 정본 재빌드를 강제하면 안 된다.

런타임 경로는 SemanticIntent v1 하나다: ``agent.stage1_v1_service.Stage1V1NativeService``
(HCX-007 → ``semantic_intent_v1`` → ``stage1_v1_resolver`` → ``deterministic_plan_compiler_v1``
→ ``stage1_v1_query_plan_v04_emitter`` → ``query_plan.QueryPlanHandoff``).
"""

from .contracts import (
    EvidenceCitation,
    FinancialToolResult,
    PlanValidation,
    ResolvedQueryPlan,
    ToolResult,
)
from .query_plan import AnswerRequirement, QueryPlanHandoff
from .stage1_v1_outcome import Stage1ReadyEnvelope
from .stage1_v1_service import Stage1V1Service

__all__ = [
    "AnswerRequirement",
    "EvidenceCitation",
    "FinancialToolResult",
    "PlanValidation",
    "QueryPlanHandoff",
    "ResolvedQueryPlan",
    "Stage1ReadyEnvelope",
    "Stage1V1Service",
    "ToolResult",
]
