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
    ("해지공시", "해지 공시"), ("해지·종료 공시", "해지·종료 공시"),
    ("해지공시", "해지 공시 확인 여부"), ("종료공시", "종료공시 유무"),
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


@pytest.mark.parametrize("field", ["확인된 것", "해지 공시"])
def test_named_event_is_a_contract_filter_not_a_counterparty_on_observation_query(field):
    company = NS(corp_code="00000001", corp_name="가상회사")
    intent = NS(entities=[
        NS(entity_id="entity-1", kind_hint="company", surface="가상회사"),
        NS(entity_id="entity-2", kind_hint="event", surface="Delta Hill BESS 공급계약"),
    ], answer_items=[NS(
        target=NS(surface="해지 공시", entity_refs=["entity-2"], qualifier_surfaces=[]),
        output=NS(field_surfaces=[field]),
        scope=NS(target_period_expressions=["2026년 6월 19일까지"], as_of_expression=None),
    )])
    reader = NS(resolve_company=lambda surface: [company] if surface == "가상회사" else [])
    assert _counterparty_terms(reader, intent, company) == ()
    assert _target_terms(intent) == ("Delta", "Hill", "BESS")
    assert _status_days(intent, question="기준일까지 해지 공시가 나왔어?",
                        corpus_cutoff="20260619") == ("20260619",)


@pytest.mark.canonical
@pytest.mark.parametrize("field", ["확인된 것", "해지 공시"])
def test_native_named_contract_observation_accepts_both_provider_field_shapes(field, tmp_path):
    from pathlib import Path
    from agent.semantic_intent_v1 import SemanticIntent
    from agent.stage1_assembly import CORPUS_CUTOFF, REFERENCE_DATE
    from agent.stage1_v1_backend_composition import build_stage1_v1_runtime
    from agent.stage1_v1_service import Stage1V1NativeService
    from app.composer.template import TemplateComposer
    from app.orchestrator import Orchestrator
    from app.tools import CanonicalToolBackend
    from src.canonical.read import CanonicalReadModel

    root = Path(__file__).resolve().parents[1]
    if not (root / "out/canonical/run.json").is_file():
        pytest.skip("local canonical corpus is not built")
    canonical = CanonicalReadModel(root / "out/canonical")
    class _NoProvider:
        def invoke(self, *_args, **_kwargs):
            raise AssertionError("saved-intent regression must not invoke the provider")
    question = (
        "LS ELECTRIC의 Widow Hill BESS PJT 공급계약은 2026년 6월 19일까지 "
        "해지 공시가 나왔어? 공시에서 확인된 것만 알려줘.")
    intent = SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [
            {"entity_id": "entity-1", "kind_hint": "company", "surface": "LS ELECTRIC"},
            {"entity_id": "entity-2", "kind_hint": "event", "surface": "Widow Hill BESS PJT 공급계약"},
        ], "answer_items": [{
            "item_id": "item-1", "operation": "retrieve",
            "target": {"kind": "event", "surface": "해지 공시",
                       "entity_refs": ["entity-2"], "qualifier_surfaces": []},
            "scope": {"target_period_expressions": ["2026년 6월 19일까지"],
                      "as_of_expression": None, "document_group_expression": None,
                      "scope_qualifier_expressions": []},
            "selection": None,
            "output": {"shape": "scalar", "projection_mode": "named_fields",
                       "field_surfaces": [field], "presentation": "auto"},
        }], "answer_groups": [], "premises": [], "unresolved_mentions": [],
        "presentation": "auto",
    }, strict=True)
    service = Stage1V1NativeService(_NoProvider(), build_stage1_v1_runtime(
        canonical, reference_date=REFERENCE_DATE, corpus_cutoff=CORPUS_CUTOFF,
        clarification_db_path=tmp_path / "observation.sqlite3"))
    result = service.start_from_intent("named-observation-offline", question, intent)
    assert result.handoff is not None and result.handoff.status == "ready"
    payload = Orchestrator(CanonicalToolBackend(rm=canonical)).run(result.handoff)
    assert payload.final_status == "answer"
    assert all(claim.state == "no_termination_observed" for claim in payload.claims)
    answer = TemplateComposer().compose(payload, question=question)
    assert "2026-06-19" in answer and "법적 유효성" in answer
