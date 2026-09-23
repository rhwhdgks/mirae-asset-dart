"""Stage 2~4 애플리케이션 계층 — 조회(typed tool) · 검증(Evidence) · 계산 · 문장화.

입력: Stage1 의 QueryPlanHandoff v0.4 (`agent.query_plan`, `app/orchestrator/adapter.py` 가 로드)
출력: AnswerPayload (`app/orchestrator/payload.py`) → composer → `GET /answer` 5필드
"""
