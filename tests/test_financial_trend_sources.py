"""Keep annual values, conclusions, and report citations linked in trends."""

from app.composer.template import TemplateComposer
from app.orchestrator.payload import AnswerClaim, AnswerPayload, ClaimCitation

_YEARS = (
    (2023, "7408312265643", "20240328000773"),
    (2024, "10776004933627", "20250314001630"),
    (2025, "12783512278323", "20260317000644"),
)
_QUESTION = "한화오션의 최근 3년간(2023~2025년) 매출액 추이를 알려줘."


def _payload(*, shared_source: bool) -> AnswerPayload:
    claims = []
    for year, value, receipt in _YEARS:
        if shared_source:
            receipt = "20260317000644"
            report_name = "사업보고서 (2025.12)"
        else:
            report_name = f"사업보고서 ({year}.12)"
        citation = ClaimCitation(
            doc_id=f"periodic_{receipt}",
            rcept_no=receipt,
            evidence_id=f"offline-revenue-{year}",
            report_name=report_name,
            source_path="2-2. 연결 손익계산서 > 매출액",
        )
        claims.append(AnswerClaim(
            output_id=f"revenue-{year}",
            label=f"한화오션 {year}년 연결 매출액",
            value_text=value,
            raw_unit="원",
            canonical_value=value,
            canonical_unit="원",
            citations=[citation],
        ))
    return AnswerPayload(
        question_id="SG-012",
        handoff_id="offline-test",
        final_status="answer",
        claims=claims,
    )


def test_trend_rows_keep_year_specific_values_and_sources_without_duplicate_footer():
    answer = TemplateComposer().compose(
        _payload(shared_source=False), question=_QUESTION)

    assert "한화오션 연결 매출액: 계속 증가했습니다." in answer
    for year, _value, receipt in _YEARS:
        assert f"{year}년 |" in answer
        assert f"사업보고서 ({year}.12) · 접수번호 {receipt}" in answer
        assert answer.count(receipt) == 1
    assert "출처\n" not in answer


def test_shared_annual_report_citation_is_grouped_once_for_the_whole_trend_table():
    answer = TemplateComposer().compose(
        _payload(shared_source=True), question=_QUESTION)

    assert "| 기간 | 금액 |" in answer
    assert (
        "근거: 사업보고서 (2025.12) · 접수번호 20260317000644 · "
        "2-2. 연결 손익계산서 > 매출액 (아래 표 전체)"
    ) in answer
    assert answer.count("20260317000644") == 1
    assert all(f"{year}년 |" in answer for year, _value, _receipt in _YEARS)
    assert "출처\n" not in answer
