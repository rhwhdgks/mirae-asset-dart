"""Colloquial filing-existence requests do not require a termination to exist."""

from types import SimpleNamespace as NS

import pytest

from agent.deterministic_plan_compiler_v1 import (
    _lifecycle_item_mode, _lifecycle_status_requested_slots,
)
from agent.stage1_v1_lifecycle_composite import (
    _TERMINATION_OBSERVATION_QUESTION, _counterparty_terms, _status_days, _target_terms,
)


@pytest.mark.parametrize("surface,field", [
    ("해지 공시", "확인된 것"), ("종료공시", "여부"),
    ("해지·종료 공시", "유무"),
])
def test_literal_termination_filing_target_preserves_generic_existence_field(surface, field):
    source = NS(target=NS(surface=surface), output=NS(field_surfaces=[field]))
    typed = NS(attributes=[], status_timepoints=["20260619"], correction_receipts=[])
    assert _lifecycle_status_requested_slots(source) == ["해지종료공시관측여부"]
    assert _lifecycle_item_mode(source, typed) == "status"


@pytest.mark.parametrize("surface,field", [
    ("공급계약", "확인된 것"), ("해지 공시", "계약금액"),
    ("해지 공시", "해지사유"),
])
def test_other_target_or_attribute_is_not_rewritten_as_an_existence_query(surface, field):
    source = NS(target=NS(surface=surface), output=NS(field_surfaces=[field]))
    assert _lifecycle_status_requested_slots(source) == []


@pytest.mark.parametrize("question", [
    "가상 공급계약은 기준일까지 해지 공시가 나왔어?",
    "가상 설비 계약에 종료 공시가 나온 게 있어?",
    "가상 계약의 해지·종료 공시가 확인되는지 알려줘.",
])
def test_colloquial_observation_question_is_not_a_terminated_event_identity(question):
    assert _TERMINATION_OBSERVATION_QUESTION.search(question)


def test_terminated_event_description_is_still_an_identity_cue():
    assert not _TERMINATION_OBSERVATION_QUESTION.search("가상회사의 해지된 공급계약")


def test_named_event_is_a_contract_filter_not_a_counterparty_on_observation_query():
    company = NS(corp_code="00000001", corp_name="가상회사")
    intent = NS(entities=[
        NS(entity_id="entity-1", kind_hint="company", surface="가상회사"),
        NS(entity_id="entity-2", kind_hint="event", surface="Delta Hill BESS 공급계약"),
    ], answer_items=[NS(
        target=NS(surface="해지 공시", entity_refs=["entity-2"], qualifier_surfaces=[]),
        output=NS(field_surfaces=["확인된 것"]),
        scope=NS(target_period_expressions=["2026년 6월 19일까지"], as_of_expression=None),
    )])
    reader = NS(resolve_company=lambda surface: [company] if surface == "가상회사" else [])
    assert _counterparty_terms(reader, intent, company) == ()
    assert _target_terms(intent) == ("Delta", "Hill", "BESS")
    assert _status_days(intent, question="기준일까지 해지 공시가 나왔어?",
                        corpus_cutoff="20260619") == ("20260619",)
