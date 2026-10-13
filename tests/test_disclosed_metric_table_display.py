"""An exact disclosure heading can prove a table without repeated topic words."""

from app.composer.narrative_matrix import (
    NarrativeDigest, _authority_bound_metric_table_digest, _public_narrative_surface,
)
from app.orchestrator.payload import AnswerClaim, AnswerPayload, ClaimCitation
from app.tools.narrative import NarrativeCell, NarrativeFanoutPlan, NarrativeFanoutResult


_TOPIC = "증권의 발행을 통한 자금조달 실적"
_PATH = "III. 재무에 관한 사항 > 7. 증권의 발행을 통한 자금조달에 관한 사항 > 7-1. 증권의 발행 실적"
_TABLE = """| (기준일 : | 2025년 12월 31일 | ) | (단위 : 백만원, %) |
|---|---|---|---|

| 발행회사 | 증권종류 | 발행일자 | 권면총액 | 이자율 | 상환여부 |
|---|---|---|---|---|---|
| 가상제조 | 회사채 | 2020.01.02 | 123,456 | 3.25% | 일부상환 |

나. 기업어음 미상환 잔액 해당사항 없습니다.
"""


def _digest(*texts):
    claims = [AnswerClaim(
        output_id=f"task.block{index}", label="가상제조 " + _PATH, text=text,
        citations=[ClaimCitation(doc_id="periodic_synthetic", rcept_no="20260310000001",
                                 section_id="funding", verification="source_roundtrip")])
        for index, text in enumerate(texts, start=1)]
    cell = NarrativeCell("cell-1", "00000001", "가상제조", None, _TOPIC)
    sidecar = NarrativeFanoutResult(
        NarrativeFanoutPlan("task", (cell,)), "ready", (cell.cell_id,), (),
        ((cell.cell_id, tuple(claim.output_id for claim in claims)),))
    payload = AnswerPayload(
        question_id="synthetic", handoff_id="synthetic", final_status="answer", claims=claims)
    payload.set_narrative_sidecars([sidecar])
    return NarrativeDigest.from_payload(payload)


def test_source_bound_table_keeps_amount_unit_and_actual_issuance_date():
    digest = _authority_bound_metric_table_digest(
        _TABLE, _TOPIC, _PATH, limit=1100)

    assert "123,456" in digest
    assert "백만원, %" in digest
    assert "2020.01.02" in digest
    assert "3.25%" in digest
    assert "일부상환" in digest


def test_unrelated_liquidity_path_does_not_inherit_securities_authority():
    assert _authority_bound_metric_table_digest(
        _TABLE, _TOPIC, "IV. 경영진단 > 유동성 및 자금조달", limit=1100) == ""


def test_chapter_wide_topic_does_not_authorize_an_incidental_money_table():
    assert _authority_bound_metric_table_digest(
        _TABLE, "임상 3상", "II. 사업의 내용 > 연구개발 조직", limit=1100) == ""


def test_missing_declared_unit_never_borrows_a_monetary_scale():
    source = _TABLE[_TABLE.index("| 발행회사"):]
    assert _authority_bound_metric_table_digest(
        source, _TOPIC, _PATH, limit=1100) == ""


def test_placeholder_balance_table_does_not_become_a_zero_amount():
    source = """(단위: 백만원)
| 구분 | 1년 이하 | 합계 |
|---|---|---|
| 공모 | - | - |
| 사모 | - | - |
"""
    assert _authority_bound_metric_table_digest(
        source, _TOPIC, _PATH, limit=1100) == ""


def test_mixed_share_and_money_units_are_preserved_without_conversion():
    source = """(단위: 원, 주)
| 발행일자 | 발행주식수 | 주당발행가액 |
|---|---|---|
| 2025.04.01 | 1,234 | 56,789 |
"""
    digest = _authority_bound_metric_table_digest(source, _TOPIC, _PATH, limit=1100)

    assert "원, 주" in digest
    assert "1,234" in digest
    assert "56,789" in digest


def test_unit_is_not_borrowed_across_an_intervening_source_paragraph():
    source = "(단위: 백만원)\n이 단위는 앞의 별도 표에만 해당합니다.\n\n" + _TABLE[_TABLE.index("| 발행회사"):]

    assert _authority_bound_metric_table_digest(source, _TOPIC, _PATH, limit=1100) == ""


def test_adjacent_markdown_tables_keep_distinct_header_boundaries():
    source = """| 구분 | 잔액 |
|---|---|
| 공모 | 10 |
| 채권명 | 발행일 | 발행액 |
|---|---|---|
| 가상채권 | 2020.01.02 | 100 |
"""
    result = _public_narrative_surface(source)

    assert "| 공모 | 10 |\n\n| 채권명 | 발행일 | 발행액 |" in result


def test_final_narrative_digest_displays_table_instead_of_trailing_heading():
    digest = _digest(_TABLE)

    assert digest is not None
    answer = digest.deterministic()
    assert "123,456" in answer
    assert "2020.01.02" in answer
    assert "백만원, %" in answer


def test_heading_only_block_is_omitted_when_a_substantive_table_answers_topic():
    digest = _digest("라. 회사채 미상환 잔액", _TABLE)

    assert digest is not None
    assert "라. 회사채 미상환 잔액" not in digest.deterministic()
    assert "123,456" in digest.deterministic()


def test_heading_with_an_explicit_value_is_not_discarded():
    digest = _digest("라. 회사채 미상환 잔액 12,345백만원")

    assert digest is not None
    assert "12,345" in digest.deterministic()
    assert "백만원" in digest.deterministic()


def test_explicit_no_issuance_statement_is_retained():
    digest = _digest("나. 기업어음증권 미상환 잔액 해당사항 없습니다.")

    assert digest is not None
    assert "해당사항 없습니다" in digest.deterministic()
