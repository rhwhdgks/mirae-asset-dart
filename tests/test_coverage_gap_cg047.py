"""Offline regression for CG-047's exact securities-funding section."""

from types import SimpleNamespace
from dataclasses import replace

import pytest

from agent.disclosed_metric_topics import (
    DisclosedMetricTopicRegrounder,
    disclosed_metric_for_query,
    search_terms_for,
)
from agent.semantic_intent_v1 import SemanticIntent
from app.tools.narrative import (
    NarrativeTool,
    _authority_bound_requested_slots,
    _relevance_filtered_disclosed_metric_hits,
)


_QUESTION = "삼성전자의 2025년 사업보고서에서 증권의 발행을 통한 자금조달 실적을 알려줘."
_TOPIC = "증권의 발행을 통한 자금조달 실적"
_RECEIPT = "20260310002820"


class _CompanyPreflight:
    @staticmethod
    def unique_question_company_surface(question):
        return "삼성전자" if "삼성전자" in question else None


class _ExactSectionIndex:
    def __init__(self, hits):
        self.hits = hits

    def search(self, _query, *, path_prefix=None, top_k=10, **_kwargs):
        rows = [hit for hit in self.hits
                if path_prefix is None or hit.path.startswith(path_prefix)]
        return tuple(rows[:top_k])


def test_cg047_resolves_report_entity_to_the_securities_funding_section():
    intent = SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [
            {"entity_id": "entity-1", "kind_hint": "company",
             "surface": "삼성전자"},
            {"entity_id": "entity-2", "kind_hint": "document",
             "surface": "2025년 사업보고서"},
        ],
        "answer_items": [{
            "item_id": "item-1", "operation": "retrieve",
            "target": {
                "kind": "document", "surface": _TOPIC,
                "entity_refs": ["entity-1", "entity-2"],
            },
            "scope": {
                "target_period_expressions": ["2025년"],
                "as_of_expression": None,
                "document_group_expression": "사업보고서",
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": "narrative", "projection_mode": "whole_target",
                "field_surfaces": [], "presentation": "auto",
            },
        }],
        "answer_groups": [], "premises": [], "unresolved_mentions": [],
        "presentation": "auto",
    }, strict=True)

    result = DisclosedMetricTopicRegrounder(_CompanyPreflight())(
        _QUESTION, intent)
    entry = disclosed_metric_for_query(_TOPIC)

    assert result is not intent
    assert result.answer_items[0].target.kind == "topic"
    assert result.answer_items[0].target.surface == _TOPIC
    assert result.answer_items[0].scope.document_group_expression == "2025년 사업보고서"
    assert entry is not None
    assert entry.section_hint == (
        "III. 재무에 관한 사항 > 7. 증권의 발행을 통한 자금조달에 관한 사항")
    assert {"자금조달", "증권발행"} <= set(search_terms_for(entry, _TOPIC))


def test_cg047_retrieval_stays_out_of_similarly_named_liquidity_section():
    entry = disclosed_metric_for_query(_TOPIC)
    assert entry is not None
    expected = SimpleNamespace(
        rank=1, chunk_id="securities-funding", section_id="section-funding",
        doc_id=f"periodic_{_RECEIPT}", source_file_id="source-file",
        rcept_dt="20260310", path=(entry.section_hint
                                    + " > 7-1. 증권의 발행 실적"),
        locator="part[0]",
        text_prompt_safe=("회사채 공모 발행일자 1997.10.02 권면총액 143,490 "
                          "백만원, 상환여부 일부상환"),
    )
    distractor = SimpleNamespace(
        rank=2, chunk_id="liquidity", section_id="section-liquidity",
        doc_id=f"periodic_{_RECEIPT}", source_file_id="source-file",
        rcept_dt="20260310",
        path="IV. 이사의 경영진단 및 분석의견 > 4. 유동성 및 자금조달과 지출",
        locator="part[1]",
        text_prompt_safe="유동자금 125,847,114백만원; 현금과 금융상품 합계",
    )
    tool = NarrativeTool(SimpleNamespace(), _ExactSectionIndex((expected, distractor)))

    selected = tool._disclosed_metric_document_hits(
        (entry,), search_terms_for(entry, _TOPIC), topic=_TOPIC,
        rcept_no=_RECEIPT, as_of="20260619", corp_codes=("00126380",),
        doc_groups=("periodic",), date_from=None)
    relevance_checked = _relevance_filtered_disclosed_metric_hits(
        selected, _TOPIC)

    assert relevance_checked
    assert all(hit.path.startswith(entry.section_hint) for hit in relevance_checked)
    assert any("143,490" in hit.text_prompt_safe for hit in relevance_checked)
    assert all("125,847,114" not in hit.text_prompt_safe for hit in relevance_checked)


def test_broad_chapter_hint_can_still_fallback_to_exact_topic_hit():
    entry = replace(
        disclosed_metric_for_query(_TOPIC),
        section_hint="II. 사업의 내용",
    )
    alias_hit = SimpleNamespace(
        rank=1, chunk_id="broad-alias", section_id="section-broad",
        doc_id=f"periodic_{_RECEIPT}", source_file_id="source-file",
        rcept_dt="20260310", path="II. 사업의 내용 > 자금 관련 설명",
        locator="part[0]", text_prompt_safe="증권의 발행을 통한 자금조달",
    )
    exact_hit = SimpleNamespace(
        rank=2, chunk_id="exact-topic", section_id="section-exact",
        doc_id=f"periodic_{_RECEIPT}", source_file_id="source-file",
        rcept_dt="20260310", path="IV. 경영진단 > 자금조달 실적",
        locator="part[1]", text_prompt_safe="자금조달 실적에 관한 구체적 내용",
    )
    tool = NarrativeTool(SimpleNamespace(), _ExactSectionIndex(
        (alias_hit, exact_hit)))

    selected = tool._disclosed_metric_document_hits(
        (entry,), search_terms_for(entry, _TOPIC), topic=_TOPIC,
        rcept_no=_RECEIPT, as_of="20260619", corp_codes=("00126380",),
        doc_groups=("periodic",), date_from=None)

    assert selected[0].chunk_id == "exact-topic"


def test_exact_disclosed_metric_topic_does_not_report_its_own_slot_missing():
    task = SimpleNamespace(
        retrieval_query=_TOPIC,
        requested_slots=(_TOPIC,),
    )
    assert _authority_bound_requested_slots(task, None) == (_TOPIC,)


def test_disclosed_metric_topic_does_not_confirm_a_narrower_unrequested_value():
    task = SimpleNamespace(
        retrieval_query=_TOPIC,
        requested_slots=("회사채 미상환 잔액",),
    )
    assert _authority_bound_requested_slots(task, None) == ()


@pytest.mark.parametrize(("query", "slot"), [
    ("가동률", "평균 가동률"),
    ("자금조달 실적", _TOPIC),
])
def test_authority_bound_metric_slot_accepts_only_declared_topic_aliases(query, slot):
    task = SimpleNamespace(retrieval_query=query, requested_slots=(slot,))

    assert _authority_bound_requested_slots(task, None) == (slot,)


@pytest.mark.parametrize("query", [
    "주요 계약 조건",
    "가동률의 하락 원인",
    "가동률과 생산능력",
])
def test_self_named_slot_does_not_prove_unknown_or_qualified_topic(query):
    task = SimpleNamespace(retrieval_query=query, requested_slots=(query,))

    assert _authority_bound_requested_slots(task, None) == ()
