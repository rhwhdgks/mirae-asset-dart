"""Offline regressions for CG-051's dated investment-judgment filing."""

from agent.disclosed_metric_topics import (
    DisclosedMetricTopic,
    DisclosedMetricTopicRegrounder,
)
from agent.semantic_intent_v1 import SemanticIntent
from agent.stage1_v1_exact_event_regrounder import (
    ExactDatedEventFieldsRegrounder,
    exact_dated_event_locator,
)


_QUESTION = (
    "셀트리온이 2026년 3월 25일 공시한 투자판단 관련 주요경영사항의 "
    "내용을 설명해줘."
)
_FORM = "투자판단 관련 주요경영사항"
_DATE = "2026년 3월 25일"


class _CompanyPreflight:
    @staticmethod
    def unique_question_company_surface(question):
        return "셀트리온" if "셀트리온" in question else None


def _intent(*, date_axis="qualifier"):
    qualifier_surfaces = [_DATE] if date_axis == "qualifier" else []
    target_period_expressions = [_DATE] if date_axis == "period" else []
    return SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [{
            "entity_id": "entity-1", "kind_hint": "company",
            "surface": "셀트리온",
        }],
        "answer_items": [{
            "item_id": "item-1", "operation": "retrieve",
            "target": {
                "kind": "metric", "surface": _FORM,
                "entity_refs": ["entity-1"],
                "qualifier_surfaces": qualifier_surfaces,
            },
            "scope": {
                "target_period_expressions": target_period_expressions,
                "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": "narrative", "projection_mode": "named_fields",
                "field_surfaces": ["내용"], "presentation": "auto",
            },
        }],
        "answer_groups": [], "premises": [], "unresolved_mentions": [],
        "presentation": "auto",
    }, strict=True)


def test_cg051_locator_extracts_the_literal_date_and_filing_form():
    assert exact_dated_event_locator(_QUESTION) == (_DATE, _FORM)
    assert exact_dated_event_locator(
        "셀트리온의 2026년 3월 25일 사업보고서에서 내용을 설명해줘."
    ) is None


def test_cg051_exact_filing_is_not_rewritten_as_a_periodic_topic(monkeypatch):
    # Inject an overlapping topic to prove the literal-form guard itself holds;
    # the production topic dictionary need not contain this exchange form.
    topic = DisclosedMetricTopic(
        term=_FORM, aliases=(), topic_query=_FORM,
        section_hint="II. 사업의 내용", sector_hint="공통",
    )
    monkeypatch.setattr(
        DisclosedMetricTopicRegrounder,
        "_financial_dictionary_knows",
        staticmethod(lambda *_args: False),
    )
    regrounder = DisclosedMetricTopicRegrounder(
        _CompanyPreflight(), topics=(topic,))

    intent = _intent()
    result = regrounder(_QUESTION, intent)

    assert result is intent
    assert result.answer_items[0].target.kind == "metric"


def test_cg051_exact_event_regrounder_restores_single_content_field():
    intent = _intent()

    result = ExactDatedEventFieldsRegrounder()(_QUESTION, intent)

    item = result.answer_items[0]
    assert item.target.kind == "event"
    assert item.target.surface == _FORM
    assert item.target.entity_refs == ["entity-1"]
    assert item.target.qualifier_surfaces == [_DATE]
    assert item.output.shape == "record"
    assert item.output.field_surfaces == ["내용"]


def test_cg051_dated_form_in_period_axis_is_not_a_narrative_topic(monkeypatch):
    topic = DisclosedMetricTopic(
        term=_FORM, aliases=(), topic_query=_FORM,
        section_hint="II. 사업의 내용", sector_hint="공통",
    )
    monkeypatch.setattr(
        DisclosedMetricTopicRegrounder,
        "_financial_dictionary_knows",
        staticmethod(lambda *_args: False),
    )

    result = DisclosedMetricTopicRegrounder(
        _CompanyPreflight(), topics=(topic,))(_QUESTION, _intent(date_axis="period"))

    assert result.answer_items[0].target.kind == "metric"
    restored = ExactDatedEventFieldsRegrounder()(_QUESTION, result)
    assert restored.answer_items[0].target.kind == "event"
    assert restored.answer_items[0].target.qualifier_surfaces == [_DATE]
