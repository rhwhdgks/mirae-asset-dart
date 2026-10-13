"""Keep the LGES/Ford disclosed estimate distinct from an amount increase."""

from app.composer.template import (
    TemplateComposer,
    _withheld_contract_estimate_explanation,
)
from app.orchestrator.payload import AnswerClaim, AnswerPayload, ClaimCitation

_RECEIPT = "20251217800853"
_BEFORE = (
    "가. 상기 계약은 Ford Motor Company에 대하여 총 75GWh 규모의 전기차 "
    "배터리를 공급하는 계약을 체결하는 건으로 예상 계약금액은 확정되지 "
    "않았으나, 수시공시 기준 금액 이상으로 추정됨에 따라 공시하는 건임 "
    "나. 상기 공급계약 관련하여 공급물량 및 계약기간 등의 계약조건은 "
    "추후 고객과의 협의에 따라 변동될 수 있음 라. 상기 계약금액은 "
    "유보기한 종료 후 공시 예정임"
)
_AFTER = (
    "가. 상기 계약은 Ford Motor Company에 대하여 총 75GWh 규모의 전기차 "
    "배터리를 공급하는 계약을 체결하는 건으로 예상 계약금액은 확정되지 "
    "않아, 수시공시 기준 금액 이상으로 추정됨에 따라 공시한 건임 "
    "나. (삭제) 라. (삭제) 마. 계약금액은 계약 공급물량에 계약 체결일"
    "(2024년 10월 14일) 기준 배터리 판가 가격을 적용하여 산정한 추정금액임 "
    "바. 본 계약은 2025년 12월 17일자로 해지되었음."
)


def _payload(after: str = _AFTER) -> AnswerPayload:
    citation = ClaimCitation(
        doc_id=f"exchange_{_RECEIPT}",
        rcept_no=_RECEIPT,
        locator="TABLE[1]/TBODY[0]/TR[9]/TD[1]",
        excerpt_prompt_safe=f"{_BEFORE}\n→\n{after}",
        verification="source_roundtrip",
    )
    claims = [
        AnswerClaim(
            output_id="correction.contract_amount",
            label="2025-12-17 정정 계약금액",
            value_text="9,603,075,000,000",
            raw_unit="원",
            canonical_value="9603075000000",
            canonical_unit="원",
            state="disclosed_from_withheld",
            text=("- → 9,603,075,000,000 "
                  "[사유: 공시유보 해제 및 계약해지 사실 안내]"),
            operator="correction_diff",
            citations=[citation],
        ),
        AnswerClaim(
            output_id="correction.narrative",
            label="2025-12-17 정정 기타 투자판단과 관련한 중요사항",
            text=(f"{_BEFORE} → {after} "
                  "[사유: 공시유보 해제 및 계약해지 사실 안내]"),
            operator="correction_diff",
            citations=[citation],
        ),
    ]
    return AnswerPayload(
        question_id="R-F-002",
        handoff_id="offline-test",
        final_status="answer",
        claims=claims,
    )


def test_answer_explains_disclosure_estimate_and_later_termination_compactly():
    payload = _payload()
    answer = TemplateComposer().compose(
        payload,
        question="LG엔솔 Ford 계약금액이 10조원으로 늘어난 이유가 뭐야?",
    )

    assert "10조 원으로 늘어난 것이 아니라" in answer
    assert "9조 6,030억 7,500만원" in answer
    assert "총 75GWh 공급물량" in answer
    assert "계약 체결일(2024년 10월 14일) 기준" in answer
    assert "2025년 12월 17일 해지됐습니다" in answer
    assert answer.count("접수번호 20251217800853") == 1
    assert "정정별 변경 내용" not in answer


def test_estimate_explanation_fails_closed_when_method_or_termination_is_missing():
    no_method = _AFTER.replace(
        "계약 공급물량에 계약 체결일(2024년 10월 14일) 기준 배터리 판가 "
        "가격을 적용하여 산정한 추정금액임",
        "계약금액은 추정금액임",
    )
    no_termination = _AFTER.replace(
        "바. 본 계약은 2025년 12월 17일자로 해지되었음.",
        "바. 기타 안내 사항임.",
    )
    question = "LG엔솔 Ford 계약금액이 10조원으로 늘어난 이유가 뭐야?"

    assert _withheld_contract_estimate_explanation(
        _payload(no_method).claims, question) is None
    assert _withheld_contract_estimate_explanation(
        _payload(no_termination).claims, question) is None
    assert _withheld_contract_estimate_explanation(
        _payload().claims, "LG엔솔 Ford 계약금액이 늘어난 이유가 뭐야?") is None
