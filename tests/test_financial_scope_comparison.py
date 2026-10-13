"""Successful explicit scope differences must not become partial answers."""

from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest

from agent.contracts import Derivation, FactSpec, OutputRef, ResolvedFinancialTask, ResolvedQueryPlan
from agent.deterministic_plan_compiler_v1 import compile_stage1_v1_generic
from agent.semantic_intent_v1 import SemanticIntent
from agent.stage1_v1_financial_backend import FinancialResolutionBackend
from app.orchestrator.payload import AnswerClaim, ClaimCitation
from app.tools.backend import CanonicalToolBackend
from app.tools.derivation import DerivationExecutor, VerifiedScalar


def _case(operator="absolute_difference", *, values=(60, 100), as_of=None,
          views=None, statements=None, units=None):
    dates = as_of or ("20260619", "20260619")
    views = views or ("restated", "restated")
    statements = statements or ("IS", "IS")
    units = units or ("원", "원")
    tasks = []
    claims = []
    verified = {}
    for index, scope in enumerate(("CFS", "SFS")):
        output = f"value-{index}"
        tasks.append(ResolvedFinancialTask(
            task_id=f"task-{index}", as_of=dates[index], view=views[index],
            facts=[FactSpec(
                output_id=output, corp_code="00000001", corp_name="가상제조",
                concept="revenue", period_start=date(2025, 1, 1),
                period_end=date(2025, 12, 31), period_type="annual", cumulative=True,
                scope=scope, statement=statements[index],
            )],
        ))
        claim = AnswerClaim(
            output_id=output, label=f"가상제조 2025년 {scope} 매출액",
            value_text=str(values[index]), raw_unit=units[index],
            canonical_value=str(values[index]) if units[index] == "원" else None,
            canonical_unit="원" if units[index] == "원" else None,
            citations=[ClaimCitation(doc_id="periodic_20260310000001",
                                     rcept_no="20260310000001", evidence_id=f"proof-{index}")],
        )
        claims.append(claim)
        verified[output] = VerifiedScalar(
            output, Decimal(values[index]), units[index],
            Decimal(values[index]) if units[index] == "원" else None, claim)
    derivations = [Derivation(
        output_id="gap", operator=operator,
        operands=[OutputRef(output_id=claim.output_id) for claim in claims],
    )] if operator else []
    plan = ResolvedQueryPlan(
        revision=0, reference_date=date(2026, 6, 19), corpus_cutoff="20260619",
        tasks=tasks, derivations=derivations,
    )
    derived, limits = DerivationExecutor().execute(derivations, verified, trace=[])
    if units[0] == units[1]:
        assert not limits
    return plan, claims + derived


def _notice(plan, claims):
    tool = CanonicalToolBackend.__new__(CanonicalToolBackend)
    return tool._supplement_scope_mismatch_notice(plan, claims)


@pytest.mark.parametrize(("operator", "values", "expected"), [
    ("absolute_difference", (60, 100), 40),
    ("absolute_difference", (100, 60), 40),
    ("difference", (60, 100), -40),
    ("difference", (100, 60), 40),
])
def test_verified_explicit_scope_difference_is_not_incomplete(operator, values, expected):
    plan, claims = _case(operator, values=values)

    assert Decimal(claims[-1].value_text) == Decimal(expected)
    assert _notice(plan, claims) == []


def test_parallel_scope_lookup_still_warns_against_implicit_time_growth():
    plan, claims = _case(None)

    assert [row.code for row in _notice(plan, claims)] == ["scope_mismatch"]


def test_missing_difference_result_never_suppresses_scope_warning():
    plan, claims = _case()

    assert _notice(plan, claims[:2])


@pytest.mark.parametrize("update", [
    {"derived_from": ["value-0", "another-value"]},
    {"operator": "percent_change"},
    {"citations": []},
    {"value_text": None, "canonical_value": None},
    {"value_text": "NaN", "canonical_value": "NaN"},
])
def test_unproved_or_misbound_result_never_suppresses_scope_warning(update):
    plan, claims = _case()
    claims[-1] = claims[-1].model_copy(update=update)

    assert _notice(plan, claims)


@pytest.mark.parametrize("kwargs", [
    {"as_of": ("20260619", "20260310")},
    {"views": ("restated", "as_filed")},
    {"statements": ("IS", "CI")},
])
def test_other_accounting_axis_mismatch_is_not_excused_by_a_difference(kwargs):
    plan, claims = _case(**kwargs)

    assert _notice(plan, claims)


def test_percent_change_across_scopes_is_not_reclassified_as_complete():
    plan, claims = _case("percent_change")

    assert _notice(plan, claims)


def test_scope_difference_does_not_hide_an_additional_cross_scope_growth_request():
    plan, claims = _case()
    growth = Derivation(
        output_id="growth", operator="percent_change",
        operands=[OutputRef(output_id="value-0"), OutputRef(output_id="value-1")],
    )
    plan = plan.model_copy(update={"derivations": [*plan.derivations, growth]})

    assert _notice(plan, claims)


def test_unknown_mixed_units_keep_the_warning_when_no_difference_can_be_proved():
    plan, claims = _case(units=("주", "원"))

    assert len(claims) == 2
    assert _notice(plan, claims)


def test_nonmonetary_but_same_unit_scope_difference_is_also_supported():
    plan, claims = _case(units=("주", "주"))

    assert claims[-1].raw_unit == "주"
    assert _notice(plan, claims) == []


def _native_plan(question):
    intent = SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [{"entity_id": "entity-1", "kind_hint": "company", "surface": "가상제조"}],
        "answer_items": [{
            "item_id": "item-1", "operation": "compare",
            "target": {"kind": "metric", "surface": "매출액", "entity_refs": ["entity-1"],
                       "qualifier_surfaces": []},
            "scope": {"as_of_expression": None, "document_group_expression": None,
                      "target_period_expressions": ["2025년"],
                      "scope_qualifier_expressions": ["연결", "별도"]},
            "selection": None,
            "output": {"shape": "comparison", "projection_mode": "named_fields",
                       "field_surfaces": ["차이"], "presentation": "auto"},
        }],
        "answer_groups": [], "premises": [], "unresolved_mentions": [], "presentation": "auto",
    }, strict=True)
    canonical = SimpleNamespace(resolve_company=lambda surface: [
        SimpleNamespace(corp_code="00000001", corp_name="가상제조")])
    result = FinancialResolutionBackend(
        canonical, canonical_build_id="a" * 32, resolver_version="synthetic/1.0",
        reference_date=date(2026, 6, 19), corpus_cutoff="20260619",
    ).resolve(question_id="synthetic", question=question, source_intent=intent)
    if result is None:
        return None
    return compile_stage1_v1_generic(question, intent, result["resolution"]).execution_plan.resolved_plan


@pytest.mark.parametrize(("expression", "operator", "scopes"), [
    ("연결 매출액과 별도 매출액의 차이", "absolute_difference", ["CFS", "SFS"]),
    ("연결 매출액 − 별도 매출액의 차이", "difference", ["CFS", "SFS"]),
    ("별도 매출액 - 연결 매출액의 차이", "difference", ["SFS", "CFS"]),
    ("별도 매출액에서 연결 매출액을 뺀 차이", "difference", ["SFS", "CFS"]),
])
def test_native_scope_relation_preserves_explicit_subtraction_order(expression, operator, scopes):
    plan = _native_plan(f"가상제조의 2025년 {expression}를 알려줘.")

    assert plan is not None
    assert [derivation.operator for derivation in plan.derivations] == [operator]
    assert [fact.scope for task in plan.tasks for fact in task.facts] == scopes


def test_two_conflicting_subtraction_directions_are_not_replaced_with_one_absolute_gap():
    assert _native_plan(
        "가상제조의 2025년 연결 매출액 - 별도 매출액 차이와 "
        "별도 매출액 - 연결 매출액 차이를 알려줘.") is None
