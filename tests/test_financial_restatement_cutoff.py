"""Period comparisons use a shared cutoff, not each year's filing date."""

from datetime import date
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from agent.deterministic_plan_compiler_v1 import compile_stage1_v1_generic
from agent.semantic_intent_v1 import SemanticIntent
from agent.stage1_v1_financial_backend import FinancialResolutionBackend
from agent.stage1_v1_backend_composition import (
    ClosedFinancialComparisonRegrounder, QuestionGroundedFinancialRegrounder,
)
from app.tools.derivation import DerivationExecutor, VerifiedScalar
from app.tools.financial import FinancialTool
from src.canonical.read import CanonicalReadModel, FactRow, Money


_CUTOFF = "20260619"
_CORP = "00000001"
_NAME = "가상테크"


def _fact(receipt, period_end, value, *, concept="revenue", scope="CFS",
          cumulative=True, period_start=None, period_type="annual"):
    instant = concept == "total_assets"
    return FactRow(
        doc_id=f"periodic_{receipt}", rcept_dt=receipt[:8], scope=scope,
        statement="BS" if instant else "IS", account_raw=concept,
        account_path=concept, account_norm=concept,
        period_start=(None if instant else period_start or period_end[:4] + "-01-01"),
        period_end=period_end, period_type="instant" if instant else period_type,
        cumulative=None if instant else cumulative, money=Money(value, "원", str(value)),
        value_status="literal", locator=f"TABLE[1]/{scope}/{concept}/{period_end}",
        source_file_id=f"source-{receipt}", table_locator="TABLE[1]",
        evidence_id=f"evidence-{receipt}-{scope}-{concept}-{period_end}",
        evidence_status="verified", statement_title="가상 재무제표",
    )


class _Snapshot(CanonicalReadModel):
    """Use production fact selection with synthetic verified source rows."""

    def __init__(self):
        self.rows = (
            _fact("20250310000001", "2024-12-31", 80),
            _fact("20260310000001", "2025-12-31", 150),
            _fact("20260310000001", "2024-12-31", 100),
            _fact("20260505000001", "2025-12-31", 155),
            _fact("20260505000001", "2024-12-31", 110),
            _fact("20260701000001", "2024-12-31", 999),
            _fact("20260310000001", "2025-12-31", 500, concept="total_assets"),
            _fact("20260505000001", "2025-12-31", 550, concept="total_assets"),
            _fact("20260520000001", "2025-12-31", 900, concept="total_assets"),
            _fact("20260520000001", "2026-03-31", 9_000, concept="total_assets"),
            _fact("20260520000001", "2026-03-31", 9_999, period_type="quarter"),
        )
        self.docs = (
            SimpleNamespace(doc_id="periodic_20250310000001", rcept_dt="20250310",
                            base_year=2024, base_month=12),
            SimpleNamespace(doc_id="periodic_20260310000001", rcept_dt="20260310",
                            base_year=2025, base_month=12),
            SimpleNamespace(doc_id="periodic_20260505000001", rcept_dt="20260505",
                            base_year=2025, base_month=12),
            SimpleNamespace(doc_id="periodic_20260520000001", rcept_dt="20260520",
                            base_year=2026, base_month=3),
        )

    @staticmethod
    def resolve_company(surface):
        return [SimpleNamespace(corp_code=_CORP, corp_name=_NAME)] if surface == _NAME else []

    @staticmethod
    def held_company_candidates(surface):
        return []

    def _load_facts(self, corp_code):
        return self.rows if corp_code == _CORP else ()

    def documents(self, *, as_of, corp_code, doc_group):
        assert corp_code == _CORP and doc_group == "periodic"
        return tuple(row for row in self.docs if row.rcept_dt <= as_of)

    @staticmethod
    def _unread_docs(*args, **kwargs):
        return ()

    @staticmethod
    def _verified_fact(row):
        return row

    @staticmethod
    def verify_fact_evidence(row):
        return row

    def get_evidence(self, evidence_id):
        row = next(row for row in self.rows if row.evidence_id == evidence_id)
        return SimpleNamespace(excerpt_safe=row.money.text)

    @staticmethod
    def document_label(doc_id):
        return "가상테크 정기보고서 " + doc_id


def _intent(*, concept="매출액", periods=("2025년", "2024년"),
            as_of=None, operation="compare", qualifiers=(), fields=("차이",)):
    return SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [{"entity_id": "entity-1", "kind_hint": "company", "surface": _NAME}],
        "answer_items": [{
            "item_id": "item-1", "operation": operation,
            "target": {"kind": "metric", "surface": concept,
                       "entity_refs": ["entity-1"], "qualifier_surfaces": list(qualifiers)},
            "scope": {"as_of_expression": as_of, "document_group_expression": None,
                      "target_period_expressions": list(periods),
                      "scope_qualifier_expressions": ["연결"]},
            "selection": None,
            "output": {"shape": "comparison" if operation == "compare" else "scalar",
                       "projection_mode": "named_fields", "field_surfaces": list(fields),
                       "presentation": "auto"},
        }],
        "answer_groups": [], "premises": [], "unresolved_mentions": [], "presentation": "auto",
    }, strict=True)


def _plan(snapshot, intent, question, *, view="restated"):
    authority = FinancialResolutionBackend(
        snapshot, canonical_build_id="a" * 32, resolver_version="synthetic/1.0",
        reference_date=date(2026, 6, 19), corpus_cutoff=_CUTOFF, view=view,
    ).resolve(question_id="synthetic", question=question, source_intent=intent)
    assert authority is not None and authority["kind"] == "resolved"
    compiled = compile_stage1_v1_generic(question, intent, authority["resolution"])
    return compiled.execution_plan.resolved_plan


def _execute(snapshot, plan):
    verified = {}
    originals = []
    trace = []
    for task in plan.tasks:
        for outcome in FinancialTool(snapshot).run_task(task, trace=trace):
            assert outcome.limitation is None and outcome.clarification is None
            assert outcome.claim is not None
            originals.append(outcome.claim)
            verified[outcome.output_id] = VerifiedScalar(
                outcome.output_id, outcome.decimal_value, outcome.unit, outcome.won, outcome.claim)
    derived, limitations = DerivationExecutor().execute(plan.derivations, verified, trace=trace)
    assert not limitations
    return originals, derived


def test_comparison_uses_latest_restatement_instead_of_older_original_value():
    snapshot = _Snapshot()
    question = "가상테크의 2025년과 2024년 연결 매출액 차이를 알려줘."
    plan = _plan(snapshot, _intent(), question)

    assert {task.as_of for task in plan.tasks} == {_CUTOFF}
    originals, derived = _execute(snapshot, plan)
    assert [Decimal(row.canonical_value) for row in originals] == [Decimal(155), Decimal(110)]
    assert Decimal(derived[0].canonical_value) == Decimal(45)
    assert all(row.citations[0].rcept_no == "20260505000001" for row in originals)


@pytest.mark.parametrize(("as_of", "expected_cutoff", "expected_values"), [
    ("2026년 3월 15일", "20260315", (150, 100)),
    ("2026년 3월", "20260331", (150, 100)),
    ("2027년 12월 31일", _CUTOFF, (155, 110)),
])
def test_explicit_as_of_is_preserved_and_never_exposes_future_facts(
        as_of, expected_cutoff, expected_values):
    snapshot = _Snapshot()
    question = f"{as_of} 기준 가상테크의 2025년과 2024년 연결 매출액 차이를 알려줘."
    plan = _plan(snapshot, _intent(as_of=as_of), question)

    assert {task.as_of for task in plan.tasks} == {expected_cutoff}
    originals, _ = _execute(snapshot, plan)
    assert tuple(Decimal(row.canonical_value) for row in originals) == expected_values
    assert all("20260701000001" not in {cite.rcept_no for cite in row.citations} for row in originals)


def test_as_filed_keeps_original_values_at_the_shared_cutoff():
    snapshot = _Snapshot()
    question = "가상테크의 2025년과 2024년 연결 매출액 차이를 알려줘."
    plan = _plan(snapshot, _intent(), question, view="as_filed")

    originals, derived = _execute(snapshot, plan)
    assert [Decimal(row.canonical_value) for row in originals] == [Decimal(150), Decimal(80)]
    assert Decimal(derived[0].canonical_value) == Decimal(70)
    assert {task.view for task in plan.tasks} == {"as_filed"}


@pytest.mark.parametrize(("quarter", "current_end", "prior_end", "expected"), [
    (2, "2025-06-30", "2025-03-31", 70),
    (3, "2025-09-30", "2025-06-30", 55),
])
def test_standalone_quarter_uses_restatement_for_both_cumulative_operands(
        quarter, current_end, prior_end, expected):
    snapshot = _Snapshot()
    snapshot.rows += (
        _fact("20250515000001", "2025-03-31", 30, period_type="quarter"),
        _fact("20250815000001", "2025-06-30", 90, period_type="half"),
        _fact("20251115000001", "2025-09-30", 140, period_type="quarter"),
        _fact("20260310000001", "2025-03-31", 35, period_type="quarter"),
        _fact("20260310000001", "2025-06-30", 105, period_type="half"),
        _fact("20260310000001", "2025-09-30", 160, period_type="quarter"),
    )
    snapshot.docs += tuple(
        SimpleNamespace(doc_id=f"periodic_{receipt}", rcept_dt=receipt[:8],
                        base_year=2025, base_month=month)
        for receipt, month in (("20250515000001", 3), ("20250815000001", 6),
                               ("20251115000001", 9)))
    question = f"가상테크의 2025년 {quarter}분기 단일 분기 연결 매출액은 얼마야?"
    intent = _intent(concept="단일 분기 연결 매출액", periods=(f"2025년 {quarter}분기",),
                     operation="retrieve", fields=("매출액",))
    plan = _plan(snapshot, intent, question)

    assert {task.as_of for task in plan.tasks} == {_CUTOFF}
    assert [fact.period_end.isoformat() for task in plan.tasks for fact in task.facts] == [
        current_end, prior_end]
    originals, derived = _execute(snapshot, plan)
    assert Decimal(derived[0].canonical_value) == Decimal(expected)
    assert derived[0].operator == "discrete_from_cumulative"
    assert all(row.citations[0].rcept_no == "20260310000001" for row in originals)


def test_instant_year_end_uses_latest_restatement_of_that_same_balance_date():
    snapshot = _Snapshot()
    question = "가상테크의 2025년 연결 자산총계는 얼마야?"
    plan = _plan(snapshot, _intent(concept="자산총계", periods=("2025년",),
                                   operation="retrieve", fields=("얼마",)), question)

    assert plan.tasks[0].as_of == _CUTOFF
    assert plan.tasks[0].facts[0].period_end == date(2025, 12, 31)
    originals, _ = _execute(snapshot, plan)
    assert Decimal(originals[0].canonical_value) == Decimal(900)
    assert originals[0].citations[0].rcept_no == "20260520000001"


def test_instant_year_end_never_overrides_explicit_earlier_as_of():
    snapshot = _Snapshot()
    question = "2026년 3월 15일 기준 가상테크의 2025년 연결 자산총계는 얼마야?"
    plan = _plan(snapshot, _intent(concept="자산총계", periods=("2025년",),
                                   as_of="2026년 3월 15일", operation="retrieve", fields=("얼마",)), question)

    assert plan.tasks[0].as_of <= "20260315"
    originals, _ = _execute(snapshot, plan)
    assert Decimal(originals[0].canonical_value) == Decimal(500)


def test_latest_disclosure_does_not_replace_requested_annual_period_with_current_quarter():
    snapshot = _Snapshot()
    question = "가상테크의 2025년 연결 매출액은 얼마야?"
    plan = _plan(snapshot, _intent(periods=("2025년",), operation="retrieve",
                                   fields=("매출액",)), question)

    fact = plan.tasks[0].facts[0]
    assert (fact.period_start, fact.period_end, fact.period_type) == (
        date(2025, 1, 1), date(2025, 12, 31), "annual")
    originals, _ = _execute(snapshot, plan)
    assert Decimal(originals[0].canonical_value) == Decimal(155)


def test_year_end_balance_comparison_keeps_each_date_at_the_shared_cutoff():
    snapshot = _Snapshot()
    snapshot.rows += (
        _fact("20250310000001", "2024-12-31", 200, concept="total_assets"),
        _fact("20260520000001", "2024-12-31", 250, concept="total_assets"),
    )
    question = "가상테크의 2025년 말과 2024년 말 연결 자산총계 차이를 알려줘."
    plan = _plan(snapshot, _intent(concept="자산총계"), question)

    assert {task.as_of for task in plan.tasks} == {_CUTOFF}
    assert [fact.period_end for task in plan.tasks for fact in task.facts] == [
        date(2025, 12, 31), date(2024, 12, 31)]
    originals, derived = _execute(snapshot, plan)
    assert [Decimal(row.canonical_value) for row in originals] == [Decimal(900), Decimal(250)]
    assert Decimal(derived[0].canonical_value) == Decimal(650)


def test_unread_later_restatement_still_reports_incomplete_coverage(monkeypatch):
    snapshot = _Snapshot()
    monkeypatch.setattr(snapshot, "_unread_docs", lambda *args, **kwargs: (
        "periodic_20260610000001",))
    question = "가상테크의 2025년 연결 자산총계는 얼마야?"
    plan = _plan(snapshot, _intent(concept="자산총계", periods=("2025년",),
                                   operation="retrieve", fields=("얼마",)), question)

    outcome, = FinancialTool(snapshot).run_task(plan.tasks[0], trace=[])
    assert outcome.claim is not None
    assert Decimal(outcome.claim.canonical_value) == Decimal(900)
    assert outcome.limitation.code == "partial_unread"
    assert outcome.limitation.affected_doc_ids == ["periodic_20260610000001"]


def test_unparseable_explicit_as_of_is_not_replaced_with_corpus_cutoff():
    intent = _intent(as_of="정해지지 않은 검토 시점")
    result = FinancialResolutionBackend(
        _Snapshot(), canonical_build_id="a" * 32, resolver_version="synthetic/1.0",
        reference_date=date(2026, 6, 19), corpus_cutoff=_CUTOFF,
    ).resolve(question_id="synthetic", question="가상테크의 연결 매출액 차이", source_intent=intent)

    assert result is None


@pytest.mark.parametrize(("as_of", "cutoff"), [("2026년 3월 15일", "20260315"),
                                                ("2026년 6월 19일", "20260619")])
def test_comparison_shape_repair_never_adds_observation_year_to_target_periods(as_of, cutoff):
    question = f"{as_of} 기준 가상테크의 2025년과 2024년 연결 매출액 차이를 알려줘."
    received = _intent(concept="매출액 차이", as_of=as_of, operation="retrieve",
                       fields=("과 2024년 연결 매출액 차이",))
    intent = ClosedFinancialComparisonRegrounder()(question, received)

    assert list(intent.answer_items[0].scope.target_period_expressions) == ["2025년", "2024년"]
    assert intent.answer_items[0].operation == "compare"
    plan = _plan(_Snapshot(), intent, question)
    assert {task.as_of for task in plan.tasks} == {cutoff}


def test_view_comparison_without_an_ancestor_uses_each_unique_source_account_path():
    snapshot = _Snapshot()
    question = "가상테크의 2024년 연결 매출액을 최초 제출값과 최신 재작성값으로 나눠서 비교해줘."
    intent = _intent(periods=("2024년",), qualifiers=("최초 제출값", "최신 재작성값"),
                     fields=("최초 제출값", "최신 재작성값"))
    intent = QuestionGroundedFinancialRegrounder(snapshot)(question, intent)
    assert list(intent.answer_items[0].output.field_surfaces) == ["최초 제출값", "최신 재작성값", "비교"]
    plan = _plan(snapshot, intent, question)

    originals, derived = _execute(snapshot, plan)
    assert [Decimal(row.canonical_value) for row in originals] == [Decimal(80), Decimal(110)]
    assert Decimal(derived[0].canonical_value) == Decimal(30)


def _ppe_snapshot():
    snapshot = _Snapshot()
    snapshot.rows = tuple(
        replace(_fact(receipt, "2023-12-31", value, concept="total_assets"),
                account_norm="ppe", account_raw="유형자산", account_path=path)
        for receipt, value, path in (
            ("20240310000001", 100, "Ⅱ. 비유동자산 > 유형자산"),
            ("20260310000001", 120, "자산 > 비유동자산 > 유형자산"),
            ("20260310000001", 900, "자산 > 금융업자산 > 유형자산"),
        )
    )
    return snapshot


@pytest.mark.parametrize("target", ["유형자산", "비유동자산 중 유형자산"])
def test_explicit_account_ancestor_is_kept_for_a_scalar_restatement(target):
    snapshot = _ppe_snapshot()
    question = "가상테크의 2023년 말 연결 비유동자산 중 유형자산은 얼마야?"
    intent = _intent(concept=target, periods=("2023년 말",), operation="retrieve", fields=("얼마",))
    plan = _plan(snapshot, intent, question)

    assert plan.tasks[0].facts[0].account_path == "자산 > 비유동자산 > 유형자산"
    originals, _ = _execute(snapshot, plan)
    assert Decimal(originals[0].canonical_value) == Decimal(120)


def test_missing_account_ancestor_does_not_silently_pick_one_of_two_ppe_accounts():
    snapshot = _ppe_snapshot()
    question = "가상테크의 2023년 말 연결 유형자산은 얼마야?"
    intent = _intent(concept="유형자산", periods=("2023년 말",), operation="retrieve", fields=("얼마",))
    plan = _plan(snapshot, intent, question)

    assert plan.tasks[0].facts[0].account_path is None
    outcome, = FinancialTool(snapshot).run_task(plan.tasks[0], trace=[])
    assert outcome.claim is None and outcome.clarification is not None


def test_account_ancestor_matching_does_not_confuse_current_and_noncurrent_assets():
    snapshot = _ppe_snapshot()
    question = "가상테크의 2023년 말 연결 유동자산 중 유형자산은 얼마야?"
    intent = _intent(concept="유형자산", periods=("2023년 말",), operation="retrieve", fields=("얼마",))
    plan = _plan(snapshot, intent, question)

    assert plan.tasks[0].facts[0].account_path is None
    outcome, = FinancialTool(snapshot).run_task(plan.tasks[0], trace=[])
    assert outcome.claim is None and outcome.clarification is not None


def test_view_comparison_does_not_choose_between_competing_latest_account_paths():
    snapshot = _Snapshot()
    snapshot.rows += (replace(_fact("20260505000001", "2024-12-31", 500),
                              account_path="another revenue account"),)
    question = "가상테크의 2024년 연결 매출액을 최초 제출값과 최신 재작성값으로 나눠서 비교해줘."
    intent = _intent(periods=("2024년",), qualifiers=("최초 제출값", "최신 재작성값"),
                     fields=("최초 제출값", "최신 재작성값", "비교"))
    result = FinancialResolutionBackend(
        snapshot, canonical_build_id="a" * 32, resolver_version="synthetic/1.0",
        reference_date=date(2026, 6, 19), corpus_cutoff=_CUTOFF,
    ).resolve(question_id="synthetic", question=question, source_intent=intent)

    assert result is None
