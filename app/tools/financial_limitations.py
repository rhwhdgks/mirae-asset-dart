"""Typed financial-source limitations carried from the approved support matrix.

The canonical fact store deliberately contains only answerable values.  A
missing fact can nevertheless have a reviewed source-boundary reason (rather
than merely being absent).  This module reads that approved, value-free
boundary metadata by financial coordinate so the execution and composition
layers can explain the limitation without a question-id branch or a synthetic
financial claim.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


_SUPPORT_MATRIX = (
    Path(__file__).resolve().parents[2]
    / "tests" / "gold" / "account_support_2025_v1.csv"
)


@dataclass(frozen=True, slots=True)
class FinancialSourceLimitation:
    """A reviewed non-answer boundary for one closed financial coordinate."""

    code: str
    detail: str


_PUBLIC_DETAIL = {
    "latest_annual_extract_unsupported": (
        "최신 연차 공시를 구조적으로 읽지 못해, 이전 공시나 유사 계정 값으로 "
        "대체하지 않았습니다."),
    "financial_sector_statement_semantics": (
        "요청하신 금융·보험사 공시에서는 일반 매출액으로 확인할 단일 항목을 찾지 못해, "
        "보험서비스결과 등 다른 성격의 계정을 매출액으로 대신하지 않았습니다."),
    "raw_total_cell_empty": (
        "연결 당기순이익 합계 행의 값 셀이 비어 있어, 하위 귀속 항목을 합산해 "
        "새 당기순이익을 만들지 않았습니다."),
    "combined_concept_line": (
        "유형자산과 투자부동산 취득이 한 결합 계정으로만 공시되어, 순수 유형자산 "
        "취득액을 분리하거나 배분하지 않았습니다."),
}


def _key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["corp_code"], row["canonical_concept"],
        row["period_end"], row["scope"],
    )


@lru_cache(maxsize=1)
def _matrix() -> dict[tuple[str, str, str, str], FinancialSourceLimitation]:
    """Load only typed non-value boundaries, with latest-extract precedence."""

    if not _SUPPORT_MATRIX.is_file():
        return {}
    rows: dict[tuple[str, str, str, str], FinancialSourceLimitation] = {}
    with _SUPPORT_MATRIX.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            key = _key(row)
            coverage = row.get("coverage_note", "")
            if "newer_annual_fact_extract_unsupported:" in coverage:
                rows[key] = FinancialSourceLimitation(
                    "latest_annual_extract_unsupported",
                    _PUBLIC_DETAIL["latest_annual_extract_unsupported"],
                )
                continue
            limitation_type = row.get("limitation_type", "")
            if (limitation_type in _PUBLIC_DETAIL
                    and key not in rows):
                rows[key] = FinancialSourceLimitation(
                    limitation_type, _PUBLIC_DETAIL[limitation_type])
    return rows


def financial_source_limitation(
        *, corp_code: str, concept: str, period_end: str, scope: str,
        ) -> FinancialSourceLimitation | None:
    """Return the reviewed boundary for exactly one financial lookup."""

    return _matrix().get((corp_code, concept, period_end, scope))


__all__ = ["FinancialSourceLimitation", "financial_source_limitation"]
