"""이슈 #124(합계) — `sum` 실행기 계약.

`agent/contracts.py` 가 파생 연산자 `sum`(operands 2개 이상)을 승인한 뒤
`app/tools/derivation.py` 가 Decimal 정확 합을 계산한다. 단위 통일은 기존
규칙(같은 unit이면 Decimal 그대로, 다르면 원 환산, 둘 다 안 되면
`unit_mismatch`)을 그대로 따른다. 라벨은 PR #135의 `fold_shared_label_prefix`
로 공유 접두를 접는다 — 같은 회사의 다른 기간 합계는 회사명이 한 번만 나오고,
회사가 다른 합계는 두 라벨을 그대로 잇는다. 정본 없이 돈다.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from app.orchestrator.payload import AnswerClaim, ClaimCitation
from app.tools.derivation import DerivationExecutor, VerifiedScalar


def _scalar(output_id: str, value: str, *, label: str, unit: str = "백만원",
            won: "Decimal | None" = None,
            rcept_no: str = "20260310002820") -> VerifiedScalar:
    claim = AnswerClaim(
        output_id=output_id, label=label, value_text=value, raw_unit=unit,
        canonical_value=str(won) if won is not None else None,
        canonical_unit="원" if won is not None else None,
        citations=[ClaimCitation(doc_id=f"periodic_{output_id}",
                                 rcept_no=rcept_no)])
    return VerifiedScalar(output_id, Decimal(value), unit, won, claim)


def _sum_deriv(output_id: str, *operand_ids: str):
    return SimpleNamespace(
        operator="sum", output_id=output_id,
        operands=[SimpleNamespace(output_id=oid) for oid in operand_ids])


def _run(derivations, verified, meta=None):
    trace: list = []
    claims, lims = DerivationExecutor().execute(
        derivations, verified, trace=trace, fact_meta=meta)
    return {c.output_id: c for c in claims}, lims, trace


def _num(value_text: str) -> Decimal:
    """``_fmt`` 천단위 콤마 표기를 되돌려 Decimal로 비교한다."""

    return Decimal(value_text.replace(",", ""))


def test_sum_two_companies_same_unit() -> None:
    verified = {
        "ss": _scalar("ss", "333605938", label="삼성전자 2025년 연결 매출액",
                      rcept_no="20260310002820"),
        "sk": _scalar("sk", "97146675", label="SK하이닉스 2025년 연결 매출액",
                      rcept_no="20260317000635"),
    }
    by_id, lims, _ = _run([_sum_deriv("sum1", "ss", "sk")], verified)
    assert not lims
    claim = by_id["sum1"]
    assert _num(claim.value_text) == Decimal("430752613")
    assert claim.raw_unit == "백만원"
    assert claim.operator == "sum"
    assert claim.derived_from == ["ss", "sk"]
    # 회사가 다르면 공유 접두가 없으므로 두 라벨을 그대로 잇는다.
    assert claim.label == (
        "삼성전자 2025년 연결 매출액과 SK하이닉스 2025년 연결 매출액의 합계")
    assert len(claim.citations) == 2


def test_sum_same_company_different_periods_folds_shared_prefix() -> None:
    verified = {
        "y2024": _scalar("y2024", "300870903", label="삼성전자 2024년 연결 매출액"),
        "y2025": _scalar("y2025", "333605938", label="삼성전자 2025년 연결 매출액"),
    }
    by_id, lims, _ = _run([_sum_deriv("sum1", "y2024", "y2025")], verified)
    assert not lims
    claim = by_id["sum1"]
    assert _num(claim.value_text) == Decimal("634476841")
    # 같은 회사면 공유 접두("삼성전자")를 한 번만 쓴다.
    assert claim.label == "삼성전자 2024년 연결 매출액과 2025년 연결 매출액의 합계"
    assert claim.label.count("삼성전자") == 1


def test_sum_converts_mismatched_units_to_won() -> None:
    verified = {
        "a": _scalar("a", "12", label="A", unit="억원", won=Decimal("1200000000")),
        "b": _scalar("b", "500", label="B", unit="백만원", won=Decimal("500000000")),
    }
    by_id, lims, _ = _run([_sum_deriv("sum1", "a", "b")], verified)
    assert not lims
    claim = by_id["sum1"]
    assert _num(claim.value_text) == Decimal("1700000000")
    assert claim.raw_unit == "원"
    assert Decimal(claim.canonical_value) == Decimal("1700000000")
    assert claim.canonical_unit == "원"


def test_sum_unit_mismatch_without_won_conversion_is_a_limitation() -> None:
    verified = {
        "a": _scalar("a", "10", label="A", unit="주"),   # 주식 수 등 원 환산 불가
        "b": _scalar("b", "5", label="B", unit="백만원"),
    }
    by_id, lims, _ = _run([_sum_deriv("sum1", "a", "b")], verified)
    assert "sum1" not in by_id
    assert [row.code for row in lims] == ["unit_mismatch"]


def test_sum_exact_decimal_no_rounding_error() -> None:
    """Decimal 정확 합 — 부동소수점 오차가 없어야 한다."""

    verified = {
        "a": _scalar("a", "333605938", label="A"),
        "b": _scalar("b", "97146675", label="B"),
    }
    by_id, lims, _ = _run([_sum_deriv("sum1", "a", "b")], verified)
    assert not lims
    assert by_id["sum1"].value_text == "430,752,613"
