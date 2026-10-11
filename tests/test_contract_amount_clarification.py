"""A stated contract-amount axis must survive clarification unchanged."""

from datetime import date
from types import SimpleNamespace

from agent.deterministic_plan_compiler_v1 import compile_stage1_v1_generic
from agent.semantic_intent_v1 import SemanticIntent
from agent.stage1_v1_context_clarification_backend import (
    MissingContextClarificationBackend,
)
from agent.stage1_v1_context_resume_backend import (
    CanonicalContractAmountChangeResumeBackend,
    CanonicalGenericEventContextResumeBackend,
    CompositeClarificationResumeBackend,
    _explicit_contract_amount_change,
)
from agent.stage1_v1_resolver import ClarificationAuthority, ResolvedAuthority


def _intent(surface: str) -> SemanticIntent:
    return SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [], "answer_groups": [], "premises": [],
        "unresolved_mentions": [], "presentation": "auto",
        "answer_items": [{
            "item_id": "item-1", "operation": "retrieve",
            "target": {"kind": "metric", "surface": surface,
                       "entity_refs": [], "qualifier_surfaces": []},
            "scope": {"as_of_expression": None,
                      "document_group_expression": None,
                      "target_period_expressions": [],
                      "scope_qualifier_expressions": []},
            "selection": None,
            "output": {"shape": "comparison",
                       "projection_mode": "named_fields",
                       "field_surfaces": ["변동"], "presentation": "auto"},
        }],
    }, strict=True)


def _current_intent(field: str = "얼마") -> SemanticIntent:
    payload = _intent("계약 금액").model_dump(mode="python")
    payload["answer_items"][0]["output"].update({
        "shape": "scalar", "field_surfaces": [field],
    })
    return SemanticIntent.model_validate(payload, strict=True)


def test_current_contract_amount_confirms_corpus_date_without_value_kind():
    authority = MissingContextClarificationBackend(
        corpus_cutoff="20260619").resolve(
            question_id="synthetic", question="그 계약 금액이 지금 얼마야?",
            source_intent=_current_intent())
    assert authority is not None
    assert [slot.role_hint for slot in authority.slots] == [
        "entity", "event", "timepoint"]
    assert authority.slots[-1].response_kind == "provide_value"
    assert "2026-06-19" in authority.slots[-1].prompt


def test_contract_amount_without_current_time_keeps_timepoint_question():
    authority = MissingContextClarificationBackend(
        corpus_cutoff="20260619").resolve(
            question_id="synthetic", question="그 계약 금액이 얼마야?",
            source_intent=_current_intent())
    assert authority is not None
    assert [slot.role_hint for slot in authority.slots] == [
        "entity", "event", "value_kind", "timepoint"]


def test_current_contract_amount_resume_compiles_single_amount_field():
    question = "그 계약 금액이 지금 얼마야?"
    intent = _current_intent("금액")
    company = SimpleNamespace(corp_code="00000001", corp_name="가상회사")
    receipt = "20250619000001"
    timeline = SimpleNamespace(
        event_key="a" * 32, corp_code=company.corp_code,
        corp_name=company.corp_name, root_rcept_no=receipt)
    canonical = SimpleNamespace(
        resolve_company=lambda value: [company] if value == "가상회사" else [],
        event_timeline=lambda **_: timeline)
    backend = CanonicalGenericEventContextResumeBackend(
        canonical, canonical_build_id="b" * 32,
        resolver_version="offline/1.0", reference_date=date(2026, 6, 19),
        corpus_cutoff="20260619")
    answers = [
        SimpleNamespace(role_hint=role, value=value, mention_ids=(),
                        applies_to_item_ids=("item-1",))
        for role, value in (
            ("entity", "가상회사"), ("event", receipt),
            ("timepoint", "2026-06-19"))
    ]
    context = SimpleNamespace(current=SimpleNamespace(answers=answers), history=[])
    result = backend.resume(
        question_id="synthetic", question=question,
        original_source_intent=intent, current_source_intent=intent,
        context=context)
    assert isinstance(result.authority, ResolvedAuthority)
    compiled = compile_stage1_v1_generic(
        question, result.derived_source_intent,
        result.authority.resolution)
    assert compiled.execution_plan.resolved_plan.tasks[0].requested_slots == [
        "계약금액"]
    assert compiled.execution_plan.resolved_plan.tasks[0].timepoints == [
        "20260619"]


def test_current_contract_amount_rejects_another_date():
    question = "그 계약 금액이 지금 얼마야?"
    intent = _current_intent()
    canonical = SimpleNamespace(resolve_company=lambda _: [])
    backend = CanonicalGenericEventContextResumeBackend(
        canonical, canonical_build_id="b" * 32,
        resolver_version="offline/1.0", reference_date=date(2026, 6, 19),
        corpus_cutoff="20260619")
    answers = [
        SimpleNamespace(role_hint=role, value=value, mention_ids=(),
                        applies_to_item_ids=("item-1",))
        for role, value in (
            ("entity", "가상회사"), ("event", "20250619000001"),
            ("timepoint", "2025-12-31"))
    ]
    context = SimpleNamespace(current=SimpleNamespace(answers=answers), history=[])
    result = backend.resume(
        question_id="synthetic", question=question,
        original_source_intent=intent, current_source_intent=intent,
        context=context)
    assert isinstance(result.authority, ClarificationAuthority)
    assert result.authority.slots[0].role_hint == "timepoint"


def test_current_contract_amount_routes_to_event_context():
    intent = _current_intent()

    class Backend:
        def __init__(self, name):
            self.name = name

        def resume(self, **_kwargs):
            return self.name

    router = CompositeClarificationResumeBackend(
        event_backend=Backend("event"),
        financial_backend=Backend("financial"),
        contract_change_backend=Backend("contract"),
        generic_event_backend=Backend("generic"),
        narrative_matrix_backend=Backend("narrative"),
    )
    answers = [SimpleNamespace(role_hint=role, reason_code="missing_context")
               for role in ("entity", "event", "timepoint")]
    context = SimpleNamespace(
        current=SimpleNamespace(answers=answers), history=[])
    assert router.resume(
        question="그 계약 금액이 지금 얼마야?", context=context,
        current_source_intent=intent) == "generic"


def test_explicit_contract_amount_change_has_no_duplicate_value_kind_slot():
    intent = _intent("계약금액 변동")
    authority = MissingContextClarificationBackend().resolve(
        question_id="synthetic", question="계약금액 변동 얼마나 됐어?",
        source_intent=intent)
    assert authority is not None
    assert [slot.role_hint for slot in authority.slots] == [
        "entity", "event", "timepoint"]
    assert _explicit_contract_amount_change(intent.answer_items[0])


def test_ambiguous_amount_axis_still_asks_value_kind():
    intent = _intent("계약 변동 금액")
    authority = MissingContextClarificationBackend().resolve(
        question_id="synthetic", question="계약 변동 금액은?",
        source_intent=intent)
    assert authority is not None
    assert [slot.role_hint for slot in authority.slots] == [
        "entity", "event", "timepoint", "value_kind"]
    assert not _explicit_contract_amount_change(intent.answer_items[0])


def test_three_answer_resume_routes_to_contract_change_not_financial():
    intent = _intent("계약금액 변동")

    class Backend:
        def __init__(self, name):
            self.name = name

        def resume(self, **_kwargs):
            return self.name

    router = CompositeClarificationResumeBackend(
        event_backend=Backend("event"),
        financial_backend=Backend("financial"),
        contract_change_backend=Backend("contract"),
        generic_event_backend=Backend("generic"),
        narrative_matrix_backend=Backend("narrative"),
    )
    answers = [SimpleNamespace(role_hint=role, reason_code="missing_context")
               for role in ("entity", "event", "timepoint")]
    context = SimpleNamespace(
        current=SimpleNamespace(answers=answers), history=[])
    assert router.resume(
        context=context, current_source_intent=intent) == "contract"


def test_resume_infers_only_explicit_amount_axis_without_extra_answer(monkeypatch):
    intent = _intent("계약금액 변동")
    company = SimpleNamespace(corp_code="00000001", corp_name="가상회사")
    canonical = SimpleNamespace(resolve_company=lambda _value: [company])
    backend = CanonicalContractAmountChangeResumeBackend(
        canonical, canonical_build_id="a" * 32,
        resolver_version="offline/1.0", reference_date=date(2026, 6, 19),
        corpus_cutoff="20260619",
        amount_observation_preflight=SimpleNamespace(resolve_pair=lambda **_: None),
    )
    captured = {}

    def resolved(**kwargs):
        captured.update(kwargs)
        return backend._terminal(item_id="item-1")

    monkeypatch.setattr(backend, "_resolved", resolved)
    answers = [
        SimpleNamespace(role_hint=role, value=value,
                        applies_to_item_ids=("item-1",), mention_ids=())
        for role, value in (
            ("entity", "가상회사"),
            ("event", "20251208000001"),
            ("timepoint", "2025-12-08, 2025-12-31"),
        )
    ]
    context = SimpleNamespace(current=SimpleNamespace(answers=answers), history=[])
    backend.resume(
        question_id="offline", question="계약금액 변동 얼마나 됐어?",
        original_source_intent=intent, current_source_intent=intent,
        context=context)
    assert captured["amount_slot"] == "계약금액"
    assert captured["timepoints"] == ["20251208", "20251231"]
