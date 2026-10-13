"""Return every independently identified input to an investment-plan sum."""

from decimal import Decimal

from app.composer.template import TemplateComposer
from app.orchestrator.payload import AnswerClaim, AnswerPayload, ClaimCitation
from app.tools.narrative import (
    InvestmentAggregationRequest,
    _aggregate_investment_table,
    _investment_sum_component_claims,
)


_TABLE = """(단위: 백만원)
| 투자명 | 투자목적 | 투자금액 | 투자기간 |
|---|---|---:|---|
| KF-21 엔진설비 | 생산설비 증설 | 20,465 | 2026년 |
| PW 엔진설비 | 생산설비 증설 | 3,320 | 2027년 |
"""


def _aggregate():
    request = InvestmentAggregationRequest(
        task_id="investment", issuer_name="한화에어로스페이스",
        aggregate_surface="투자계획 합계", expected_row_count=2)
    result = _aggregate_investment_table(_TABLE, request)
    assert result is not None
    return result


def test_aggregate_keeps_source_row_for_each_component():
    aggregate = _aggregate()

    assert aggregate.row_count == 2
    assert aggregate.value == Decimal("23785")
    assert [component.label for component in aggregate.components] == [
        "KF-21 엔진설비", "PW 엔진설비"]
    assert all("투자금액" in component.source_excerpt
               for component in aggregate.components)
    assert "20,465" in aggregate.components[0].source_excerpt
    assert "3,320" in aggregate.components[1].source_excerpt
    assert all("단위: 백만원" in component.source_excerpt
               for component in aggregate.components)


def test_composed_sum_shows_every_cited_component():
    aggregate = _aggregate()
    components = _investment_sum_component_claims(
        task_id="investment", corp_name="한화에어로스페이스",
        aggregate=aggregate, doc_id="periodic_20260513000860",
        receipt="20260513000860", section_id="section-investment",
        locator="table[0]")
    citation = ClaimCitation(
        doc_id="periodic_20260513000860", rcept_no="20260513000860",
        section_id="section-investment", locator="table[0]",
        excerpt_prompt_safe=_TABLE, verification="source_roundtrip")
    total = AnswerClaim(
        output_id="investment.investment-sum",
        label="한화에어로스페이스 투자계획 2건 계획금액 합계",
        value_text="23,785", raw_unit="백만원",
        canonical_value=format(aggregate.value_won, "f"),
        canonical_unit="원", operator="sum",
        derived_from=[claim.output_id for claim in components],
        citations=[citation])
    payload = AnswerPayload(
        question_id="DEV-INV-005", handoff_id="offline-test",
        final_status="answer", claims=[*components, total])

    answer = TemplateComposer().compose(
        payload, question="한화에어로스페이스 투자계획 두 건의 합계를 알려줘")

    assert "237억 8,500만원" in answer
    assert "KF-21" in answer
    assert "PW" in answer
    assert "계산 근거:" in answer
    assert "204억 6,500만원" in answer
    assert "33억 2,000만원" in answer
    assert answer.count("접수번호 20260513000860") == 1
    assert all(claim.citations[0].verification == "source_roundtrip"
               for claim in components)
