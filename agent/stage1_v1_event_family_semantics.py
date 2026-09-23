"""Shared question-grounded semantics for form-backed event families."""

from __future__ import annotations

import re
from typing import Iterable


_FACILITIES_INVESTMENT = re.compile(r"(?:신규\s*)?시설\s*투자")


def is_facilities_investment_request(*surfaces: str) -> bool:
    return any(_FACILITIES_INVESTMENT.search(surface or "") for surface in surfaces)


def facilities_investment_requested_slots(
        field_surfaces: Iterable[str],
        ) -> list[str]:
    slots: list[str] = []
    for surface in field_surfaces:
        compact = re.sub(r"\s+", "", surface)
        if "투자대상" in compact or compact == "대상":
            slot = "투자대상"
        elif "목적" in compact:
            slot = "목적"
        elif "금액" in compact or "규모" in compact:
            slot = "금액"
        elif "기간" in compact:
            slot = "기간"
        elif ("정정" in compact or "후속" in compact) and "공시" in compact:
            slot = "정정후속공시상태"
        else:
            slot = compact
        if slot and slot not in slots:
            slots.append(slot)
    return slots


_BOND_FACE_VALUE = re.compile(r"사채.{0,4}권면.{0,10}총액")


def is_bond_face_value_request(*surfaces: str) -> bool:
    """Whether a literal-form field surface names 사채의 권면(전자등록)총액.

    The exchange form (예: 「주요사항보고서(전환사채권발행결정)」) repeats a
    near-identical label for each currency/issue-market breakdown next to the
    one grand-total cell a plain "사채 권면(전자등록)총액" question means. A
    free-text label match cannot tell those apart; only the exact registered
    canonical slot (``app/tools/events.py`` ``SLOT_CANONICAL_PATHS``) binds to
    the full form path instead of the leaf label. Recognizing this exact axis
    both when a backend requests the slot and when the compiler re-derives it
    from the intent keeps the two in agreement without duplicating the regex.
    """

    return any(_BOND_FACE_VALUE.search(surface or "") for surface in surfaces)


def bond_face_value_slot() -> str:
    return "사채의권면(전자등록)총액"


__all__ = [
    "bond_face_value_slot",
    "facilities_investment_requested_slots",
    "is_bond_face_value_request",
    "is_facilities_investment_request",
]
