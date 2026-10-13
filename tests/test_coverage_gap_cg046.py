"""Offline regressions for CG-046's two-section annual-report question."""

from types import SimpleNamespace

import pytest

from agent.disclosed_metric_topics import (
    DisclosedMetricTopicRegrounder,
    disclosed_metrics_for_query,
    search_terms_for,
)
from agent.semantic_intent_v1 import SemanticIntent
from app.tools.narrative import NarrativeTool, _slot_present


_QUESTION = "삼성전자의 2025년 사업보고서에서 이사회 운영과 주주총회 투표제도를 설명해줘."
_TOPIC = "이사회 운영과 주주총회 투표제도"
_RECEIPT = "20260310002820"


class _CompanyPreflight:
    @staticmethod
    def unique_question_company_surface(question):
        return "삼성전자" if "삼성전자" in question else None


def _intent(target_surface, field_surfaces):
    projection = "named_fields" if field_surfaces else "whole_target"
    return SemanticIntent.model_validate({
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
                "kind": "document", "surface": target_surface,
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
                "shape": "narrative", "projection_mode": projection,
                "field_surfaces": field_surfaces, "presentation": "auto",
            },
        }],
        "answer_groups": [], "premises": [], "unresolved_mentions": [],
        "presentation": "auto",
    }, strict=True)


@pytest.mark.parametrize(("target", "fields"), [
    (_TOPIC, ["이사회 운영", "주주총회 투표제도"]),
    ("이사회 운영", ["주주총회 투표제도"]),
])
def test_cg046_keeps_both_topics_when_document_is_an_entity_reference(target, fields):
    intent = _intent(target, fields)
    regrounded = DisclosedMetricTopicRegrounder(_CompanyPreflight())(
        _QUESTION, intent)

    assert regrounded is not intent
    assert regrounded.answer_items[0].target.kind == "topic"
    assert regrounded.answer_items[0].target.surface == _TOPIC
    assert regrounded.answer_items[0].output.field_surfaces == [
        "이사회 운영", "주주총회 투표제도",
    ]
    assert regrounded.answer_items[0].scope.document_group_expression == "2025년 사업보고서"


class _SectionIndex:
    def __init__(self, hits):
        self.hits = hits

    def search(self, _query, *, path_prefix=None, top_k=10, **_kwargs):
        rows = [hit for hit in self.hits
                if path_prefix is None or hit.path.startswith(path_prefix)]
        return tuple(rows[:top_k])


def test_cg046_search_budget_preserves_both_annual_report_sections():
    entries = disclosed_metrics_for_query(_TOPIC)
    assert len(entries) == 2
    board, voting = entries
    hits = tuple(
        SimpleNamespace(
            rank=index,
            chunk_id=f"board-{index}",
            section_id=f"section-board-{index}",
            doc_id=f"periodic_{_RECEIPT}",
            source_file_id="source-file",
            rcept_dt="20260310",
            path=f"{board.section_hint} > item {index}",
            locator=f"part[{index}]",
            text_prompt_safe="이사회 운영 절차",
        )
        for index in range(8)
    ) + tuple(
        SimpleNamespace(
            rank=index + 20,
            chunk_id=f"voting-{index}",
            section_id=f"section-voting-{index}",
            doc_id=f"periodic_{_RECEIPT}",
            source_file_id="source-file",
            rcept_dt="20260310",
            path=f"{voting.section_hint} > item {index}",
            locator=f"part[{index + 20}]",
            text_prompt_safe="2025년 정기주주총회에서 전자투표제를 실시",
        )
        for index in range(2)
    )
    tool = NarrativeTool(SimpleNamespace(), _SectionIndex(hits))
    sub_queries = tuple(
        query for entry in entries
        for query in search_terms_for(entry, _TOPIC))

    selected = tool._disclosed_metric_document_hits(
        entries, sub_queries, topic=_TOPIC, rcept_no=_RECEIPT,
        as_of="20260619", corp_codes=("00126380",),
        doc_groups=("periodic",), date_from=None)

    paths = {hit.path for hit in selected}
    assert any("1. 이사회에 관한 사항" in path for path in paths)
    assert any("3. 주주총회 등에 관한 사항" in path for path in paths)
    assert sum("1. 이사회에 관한 사항" in path for path in paths) == 5
    assert sum("3. 주주총회 등에 관한 사항" in path for path in paths) == 2
    assert _slot_present("주주총회 투표제도", "2025년 정기주주총회에서 전자투표제를 실시")
