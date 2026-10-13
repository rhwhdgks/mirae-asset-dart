"""Keep a planned-to-completed contract change legible in the lead answer."""

from app.composer.template import TemplateComposer
from app.orchestrator.payload import AnswerClaim, AnswerPayload, ClaimCitation


_BEFORE = (
    "다. 기존 Stellantis N.V.가 NextStar Energy Inc.에 보증하던 $659M은 "
    "당사로 이관될 예정이며, 구체적인 차입계약 및 보증계약은 2026년 "
    "상반기 중 기존 채권자와 세부 내용을 논의하여 최종 체결할 예정임. "
    "라. 채권자 협의 완료를 전제로 하며 보증 총액의 일부 변동 등 구체적인 "
    "사항은 추후 협의에 따라 변경될 수 있음"
)

_AFTER = (
    "다. 기존 Stellantis N.V.가 NextStar Energy Inc.에 보증하던 $659M은 "
    "당사로 이관되었으며, 이에 대한 차입계약 및 보증계약은 2026년 "
    "3월 30일 체결됨. 라. (삭제)"
)


def _payload():
    claim = AnswerClaim(
        output_id="correction.transition",
        label="정정(20260331801967) 4. 기타 투자판단과 관련한 중요사항",
        text=f"{_BEFORE} → {_AFTER} [사유: 합작법인 지분구조 변경에 따른 재계약 완료]",
        operator="correction_diff",
        citations=[ClaimCitation(
            doc_id="exchange_20260331801967",
            rcept_no="20260331801967",
            locator="TABLE[1]/TBODY[0]/TR[5]/TD[1:2]",
            excerpt_prompt_safe=f"{_BEFORE}\n→\n{_AFTER}",
            verification="source_roundtrip",
        )],
    )
    return AnswerPayload(
        question_id="CG-052", handoff_id="offline-test",
        final_status="answer", claims=[claim])


def test_correction_answer_leads_with_verified_status_and_removed_condition():
    answer = TemplateComposer().compose(
        _payload(),
        question="LG에너지솔루션이 2026년 3월 31일 정정한 투자판단 관련 주요경영사항은 무엇이 바뀌었어?",
    )

    assert answer.startswith("2026-03-31 정정에서")
    assert "기존 보증액 $659M의 당사 이관은 예정에서 완료로 바뀌었습니다." in answer
    assert "2026-03-30 체결됐다고 정정됐습니다." in answer
    assert "보증 총액 등 일부 변동 가능성을 적은 종전 문구도 삭제됐습니다." in answer
    assert answer.count("기존 보증액 $659M") == 1
