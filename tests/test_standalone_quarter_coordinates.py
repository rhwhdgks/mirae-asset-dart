"""A stated standalone quarter needs two same-scope cumulative endpoints."""

from datetime import date
from types import SimpleNamespace

import agent.stage1_v1_financial_backend as financial
from agent.planning import FinancialConcept
from agent.semantic_intent_v1 import SemanticIntent


QUESTION = "삼성전자의 2025년 3분기 단독 연결 매출액은 얼마인가?"


def _intent(target: str) -> SemanticIntent:
    return SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [{
            "entity_id": "entity-1", "kind_hint": "company",
            "surface": "삼성전자",
        }],
        "answer_items": [{
            "item_id": "item-1", "operation": "retrieve",
            "target": {
                "kind": "metric", "surface": target,
                "entity_refs": ["entity-1"], "qualifier_surfaces": [],
            },
            "scope": {
                "as_of_expression": None,
                "document_group_expression": None,
                "target_period_expressions": ["2025년 3분기"],
                "scope_qualifier_expressions": ["연결"],
            },
            "selection": None,
            "output": {
                "shape": "scalar", "projection_mode": "named_fields",
                "field_surfaces": ["매출액"], "presentation": "auto",
            },
        }],
        "answer_groups": [], "premises": [], "unresolved_mentions": [],
        "presentation": "auto",
    }, strict=True)


def test_standalone_q3_keeps_revenue_and_cfs_for_both_operands(monkeypatch):
    calls = []

    def coordinate(_item, **kwargs):
        calls.append(kwargs)
        return financial.FinancialCoordinate(
            corp_code="00126380", corp_name="삼성전자",
            concept=kwargs["concept_override"],
            period_start=date(2025, 7, 1), period_end=date(2025, 9, 30),
            period_type="quarter", cumulative=False,
            scope="CFS", statement="IS")

    monkeypatch.setattr(financial, "_one_coordinate", coordinate)
    intent = _intent("단독 연결 매출액")
    rows = financial._single_quarter_cumulative_coordinates(
        intent.answer_items[0], intent=intent,
        companies=SimpleNamespace(), reference_date=date(2026, 6, 19),
        corpus_cutoff="20260619", question=QUESTION)

    assert len(calls) == 1
    assert calls[0]["concept_override"] == FinancialConcept.REVENUE
    assert [(row.period_start, row.period_end) for row in rows] == [
        (date(2025, 1, 1), date(2025, 9, 30)),
        (date(2025, 1, 1), date(2025, 6, 30)),
    ]
    assert {(row.concept, row.scope, row.statement, row.cumulative)
            for row in rows} == {(FinancialConcept.REVENUE, "CFS", "IS", True)}


def test_bare_q3_is_not_silently_treated_as_standalone(monkeypatch):
    def coordinate(*_args, **_kwargs):
        raise AssertionError("A bare Q3 question must not enter this derivation")

    monkeypatch.setattr(financial, "_one_coordinate", coordinate)
    intent = _intent("연결 매출액")
    rows = financial._single_quarter_cumulative_coordinates(
        intent.answer_items[0], intent=intent,
        companies=SimpleNamespace(), reference_date=date(2026, 6, 19),
        corpus_cutoff="20260619",
        question="삼성전자의 2025년 3분기 연결 매출액은 얼마인가?")
    assert rows == ()
