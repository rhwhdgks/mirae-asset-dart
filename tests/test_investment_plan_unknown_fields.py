"""Keep incomplete investment-plan rows visible without interpreting dashes."""

from types import SimpleNamespace

from app.composer.narrative_matrix import NarrativeDigest, _investment_rows
from app.orchestrator.payload import AnswerClaim, AnswerPayload, ClaimCitation

_TABLE = """마. 설비 투자 현황 및 계획
(단위: 백만원)
| 해당회사 | 투자명 | 투자목적 | 내용 | 기간 | 총 소요자금 | 기 지출금액 | 향후 기대효과 |
|---|---|---|---|---|---:|---:|---|
| 한화에어로스페이스 | KF-21 엔진설비 | 생산설비 증설 | 엔진설비 확대 | 2026년 | 100 | 50 | 생산능력 향상 |
| 한화에어로스페이스 | 창원2 설비 | 생산능력 향상 | 설비 도입 |  | 200 | - | 생산능력 향상 |
"""


def _payload(text: str) -> AnswerPayload:
    claim = AnswerClaim(
        output_id="investment-plan",
        label="설비 투자 현황 및 계획 [확인 항목: 투자계획]",
        text=text,
        citations=[ClaimCitation(
            doc_id="exchange_20260513000860",
            rcept_no="20260513000860",
            section_id="section-investment",
            locator="table[0]",
            excerpt_prompt_safe=text,
            verification="source_roundtrip",
        )],
    )
    payload = AnswerPayload(
        question_id="G-A-010", handoff_id="offline-test",
        final_status="answer", claims=[claim])
    payload.set_narrative_sidecars([
        SimpleNamespace(plan=SimpleNamespace(cells=()))])
    return payload


def test_incomplete_project_row_disables_partial_compact_projection():
    assert _investment_rows(_TABLE) == ()


def test_plan_digest_shows_missing_cell_and_preserves_dash_meaning():
    digest = NarrativeDigest.from_payload(
        _payload(_TABLE),
        question="한화에어로스페이스 투자계획과 집행 현황을 정리해줘")

    assert digest is not None
    answer = digest.deterministic()

    assert "KF-21 엔진설비" in answer
    assert "창원2 설비" in answer
    assert "원문에 기재되지 않음" in answer
    assert "0원이나 미집행으로 해석하지 않았습니다" in answer
    assert "| 해당회사 | 투자명 | 투자목적 | 내용 | 기간 | 총 소요자금 |" in answer
