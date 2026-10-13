"""Source-bound company/product identity through retrieval-cell composition."""

from types import SimpleNamespace

import pytest

from agent.contracts import DocumentSelector
from app.composer.narrative_matrix import (
    NarrativeMatrix,
    _business_table_axis_digest,
    _compact_cell_text,
    _named_company_product_digest,
)
from app.orchestrator.payload import AnswerClaim, AnswerPayload, ClaimCitation
from app.tools.narrative import (
    NarrativeCell, NarrativeFanoutPlan, NarrativeFanoutResult, NarrativeTool,
)


_RECEIPT = "20260310000001"
_PARENT = """가. 당사 주요 제품 및 서비스
당사는 자회사에 지배와 투자, 경영자문 서비스를 제공하고 있습니다.

가. 당사 매출 실적
| 사업 부문 | 품목 | 금액 |
|---|---|---|
| 임대 | 임대수익 | 100 |
| 용역 | 용역수익 | 200 |
"""
_PROFILE = """(주)가상금속

1. 사업의 개요
당사는 합금 소재를 생산하고 판매하는 회사입니다.

2. 주요 제품 및 서비스
| 제품 | 설명 |
|---|---|
| 항공·방산용 합금 | 항공기와 방위산업에 쓰이는 합금 |
| 정밀봉강 | 정밀 부품에 쓰이는 봉 형태의 제품 |

3. 원재료 및 생산설비
제품 답변에 섞이면 안 되는 원재료 설명입니다.
"""
_ENGLISH_PROFILE = """Example Motors Inc.

1. 사업의 개요
당사는 모터를 유통하고 판매하는 회사입니다.

2. 주요 제품 및 서비스
당사는 산업용 모터 제품을 해외 시장에 유통 및 판매하고 있습니다.

가. 재무현황
재무현황은 아래 표와 같습니다.

3. 원재료 및 생산설비
원재료 설명입니다.
"""
_SEGMENTS = """가. 사업부문별 재무현황
| (단위: 천원) > 계정과목 | (단위: 천원) > 사업부문 | 2025년 |
|---|---|---|
| 매출액 | 금속소재부문 | 2000 |
| 매출액 | 모터부문 | 1000 |
| 영업이익 | 금속소재부문 | 200 |
| 영업이익 | 모터부문 | 100 |
"""


def _claim(output_id, text, section):
    return AnswerClaim(
        output_id=output_id, label="가상지주 공시 근거", text=text,
        citations=[ClaimCitation(
            doc_id=f"periodic_{_RECEIPT}", rcept_no=_RECEIPT,
            section_id=section, verification="source_roundtrip")])


def _payload(*, product_output_ids=None):
    selector = DocumentSelector(rcept_no=_RECEIPT)
    plan = NarrativeFanoutPlan("task-1", (
        NarrativeCell("cell-1", "00000001", "가상지주", None, "사업부문", selector),
        NarrativeCell("cell-2", "00000001", "가상지주", None,
                      "주요 제품 및 서비스", selector),
    ))
    claims = [
        _claim("task.block1", _SEGMENTS, "overview"),
        _claim("task.block2", _PARENT, "parent"),
        _claim("task.block10", _PROFILE, "subsidiary"),
        _claim("task.block11", _ENGLISH_PROFILE, "subsidiary"),
    ]
    payload = AnswerPayload(
        question_id="synthetic", handoff_id="synthetic",
        final_status="answer", claims=claims)
    payload.set_narrative_sidecars([NarrativeFanoutResult(
        plan, "ready", ("cell-1", "cell-2"), (), (
            ("cell-1", ("task.block1",)),
            ("cell-2", product_output_ids or (
                "task.block2", "task.block10", "task.block11")),
        ))])
    return payload


def test_matrix_preserves_source_block_order_and_paragraph_boundaries():
    matrix = NarrativeMatrix.from_payload(_payload())

    assert matrix is not None
    excerpt = matrix.cells[1].excerpt
    assert excerpt.index("임대수익") < excerpt.index("(주)가상금속")
    assert "| 용역 | 용역수익 | 200 |\n\n(주)가상금속" in excerpt


def test_product_cell_keeps_source_company_names_and_table_or_prose_products():
    matrix = NarrativeMatrix.from_payload(_payload())
    assert matrix is not None

    digest = _compact_cell_text(
        matrix.cells[1].excerpt, topic=matrix.cells[1].topic, limit=700)

    assert "| (주)가상금속 | 항공·방산용 합금·정밀봉강 |" in digest
    assert "| Example Motors Inc. |" in digest
    assert "산업용 모터 제품" in digest
    assert "임대수익" not in digest
    assert "원재료 설명" not in digest
    assert "재무현황" not in digest
    assert "당사는" not in digest


def test_final_answer_keeps_every_named_product_row_and_uses_visible_evidence():
    matrix = NarrativeMatrix.from_payload(_payload())
    assert matrix is not None

    answer = matrix.deterministic()

    assert "(주)가상금속" in answer
    assert "Example Motors Inc." in answer
    assert "항공·방산용 합금" in answer
    assert "산업용 모터 제품" in answer
    assert "금속소재부문·모터부문" in answer
    assert "임대수익" not in answer
    assert "원재료 설명" not in answer
    assert "재무현황" not in answer
    assert "당사는" not in answer
    assert "Example Motors Inc." not in answer.split("\n", 1)[0]


def test_product_digest_never_cuts_a_source_product_at_its_middle_dot():
    name = "고강도항공용·방위산업용알루미늄합금"
    source = _PROFILE.replace("항공·방산용 합금", name)

    digest = _named_company_product_digest(source, limit=40)

    assert name in digest


def test_generic_heading_does_not_become_a_source_company_name():
    source = _PROFILE.replace("(주)가상금속", "그룹 주요 사업")

    assert _named_company_product_digest(source, limit=700) == ""


def test_duplicate_claim_binding_fails_closed():
    payload = _payload(product_output_ids=("task.block2", "task.block2"))

    assert NarrativeMatrix.from_payload(payload) is None


def test_segment_account_table_precedes_parent_revenue_type_for_business_axis():
    digest = _business_table_axis_digest(
        _PARENT + "\n\n" + _SEGMENTS, limit=700)

    assert "금속소재부문·모터부문" in digest
    assert "임대수익" not in digest
    assert "용역수익" not in digest


@pytest.mark.parametrize(("slot", "expected_missing"), [
    ("주요 사업 내용", False),
    ("주요사업내용", False),
    ("핵심사업의내용", False),
    ("계약 상대방", True),
    ("태양광사업내용", True),
])
def test_completed_business_axis_confirms_only_its_own_requested_topic(
        monkeypatch, slot, expected_missing):
    plan = _payload().narrative_sidecars[0].plan
    tool = NarrativeTool(SimpleNamespace(), SimpleNamespace())
    task = SimpleNamespace(
        task_id=plan.task_id, operation="search", corp_codes=["00000001"],
        periods=[], document_selector=DocumentSelector(rcept_no=_RECEIPT),
        as_of="20260619", retrieval_query="사업내용", requested_slots=[slot])
    monkeypatch.setattr(tool, "_run_one_task", lambda cell_task, **kwargs: (
        [_claim(cell_task.task_id + ".block1", _SEGMENTS, "overview")],
        [], [_RECEIPT], None))

    _, limitations, _, _ = tool.run_task(
        task, trace=[], corp_name="가상지주", sidecar=plan)

    assert any(limitation.code == "slot_not_confirmed"
               for limitation in limitations) is expected_missing


def test_missing_business_cell_is_not_confirmed_by_a_sibling_product_cell(monkeypatch):
    plan = _payload().narrative_sidecars[0].plan
    tool = NarrativeTool(SimpleNamespace(), SimpleNamespace())
    task = SimpleNamespace(
        task_id=plan.task_id, operation="search", corp_codes=["00000001"],
        periods=[], document_selector=DocumentSelector(rcept_no=_RECEIPT),
        as_of="20260619", retrieval_query="사업내용",
        requested_slots=["주요 사업 내용"])

    def execute_cell(cell_task, **kwargs):
        if cell_task.retrieval_query == "사업부문":
            return [], [], [], None
        return [_claim(cell_task.task_id + ".block1", _PROFILE, "products")], [], [_RECEIPT], None

    monkeypatch.setattr(tool, "_run_one_task", execute_cell)
    _, limitations, _, _ = tool.run_task(
        task, trace=[], corp_name="가상지주", sidecar=plan)

    assert any(limitation.code == "narrative_cell_not_found" for limitation in limitations)
    assert any(limitation.code == "slot_not_confirmed" for limitation in limitations)
