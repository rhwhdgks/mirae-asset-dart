"""Deterministic Korean won display formatting for public answers.

Typed claims keep the source spelling (for example ``1,234백만원``) and the
canonical won value.  Scaled source units use the exact mixed-unit public
form (``12억 3,400만원``); already-won source surfaces remain as stated.  No
rounding or LLM-side arithmetic is involved.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation

_SOURCE_UNIT_WON = {
    "원": Decimal(1),
    "천원": Decimal(10**3),
    "백만원": Decimal(10**6),
    "억원": Decimal(10**8),
    "조원": Decimal(10**12),
}


def normalize_source_money_unit(unit: str | None) -> str:
    """Normalize harmless source spacing such as ``백만 원``.

    Korean disclosures and HCX output use both attached and spaced forms.  The
    spelling changes no amount or scale, so normalize only whitespace inside a
    closed money-unit vocabulary before lookup.
    """

    return "".join((unit or "").split())


def format_won_exact(value: str | None, *, absolute: bool = False) -> str | None:
    """Render an integral won value as exact 조·억·만원·원 groups."""

    if value is None:
        return None
    try:
        decimal = Decimal(value)
    except (InvalidOperation, ValueError):
        return None
    if not decimal.is_finite() or decimal != decimal.to_integral_value():
        return None

    number = int(decimal)
    negative = number < 0 and not absolute
    remainder = abs(number)
    if remainder == 0:
        return "0원"

    groups: list[tuple[int, str]] = []
    for divisor, unit in ((10**12, "조"), (10**8, "억"), (10**4, "만")):
        quotient, remainder = divmod(remainder, divisor)
        if quotient:
            groups.append((quotient, unit))
    if remainder:
        groups.append((remainder, "원"))

    parts: list[str] = []
    for index, (amount, unit) in enumerate(groups):
        last = index == len(groups) - 1
        if unit == "만":
            suffix = "만원" if last else "만"
        elif unit in {"조", "억"} and last:
            suffix = f"{unit}원"
        else:
            suffix = unit
        parts.append(f"{amount:,}{suffix}")
    return ("-" if negative else "") + " ".join(parts)


def claim_won_display(claim) -> str | None:
    """Return an exact public amount for a typed won claim, if available."""

    if getattr(claim, "canonical_unit", None) != "원":
        return None
    label = str(getattr(claim, "label", "") or "")
    # A cash-outflow or loss label already carries the negative meaning.  Show
    # its magnitude as users ordinarily read it; other negative claims keep a
    # leading minus sign so a decrease cannot silently become an increase.
    # A base loss/outflow claim already names the negative meaning, so its
    # public amount is normally read as a magnitude. A derived signed delta
    # can inherit the same words through its operand label; hiding the minus
    # sign there erases the requested direction of change.
    absolute = (
        ("현금유출" in label or "손실" in label)
        and not bool(getattr(claim, "derived_from", ()) or ())
    )

    # ``억원`` and ``조원`` are already public-friendly units.  Keep an exact
    # source spelling such as ``28.5억원`` instead of mechanically expanding
    # it to the longer ``28억 5,000만원``.  Accounting-scale ``백만원`` and
    # ``천원`` amounts still go through the mixed-unit renderer below.
    raw_unit = str(getattr(claim, "raw_unit", "") or "")
    raw_value = str(getattr(claim, "value_text", "") or "").strip()
    if raw_unit in {"억원", "조원"} and raw_value:
        negative = raw_value.startswith("-") or (
            raw_value.startswith("(") and raw_value.endswith(")"))
        magnitude = raw_value.strip("+ -() ")
        if magnitude:
            sign = "" if absolute or not negative else "-"
            return f"{sign}{magnitude}{raw_unit}"
    return format_won_exact(
        getattr(claim, "canonical_value", None), absolute=absolute)


def format_source_money_exact(value: str, unit: str) -> str | None:
    """Render one unambiguous source amount/unit pair without scale loss."""

    normalized_unit = normalize_source_money_unit(unit)
    multiplier = _SOURCE_UNIT_WON.get(normalized_unit)
    if multiplier is None:
        return None
    raw = (value or "").strip()
    negative = raw.startswith("-") or (
        raw.startswith("(") and raw.endswith(")"))
    raw = raw.strip("+ -() ").replace(",", "")
    try:
        numeric = Decimal(raw)
    except (InvalidOperation, ValueError):
        return None
    if not numeric.is_finite():
        return None
    # ``원`` is already the source's exact scale.  Reformatting it as a
    # 조·억·만 group changes a table's declared unit, which is especially
    # misleading when a narrative section also contains 백만원 financial
    # tables.  Keep that source surface verbatim while still converting
    # accounting-scale units below.
    if normalized_unit == "원":
        source = (value or "").strip().strip("+ -() ")
        sign = "-" if negative else ""
        return f"{sign}{source}{normalized_unit}"
    # A decimal ``억원``/``조원`` surface is both exact and more concise than
    # its expanded equivalent.  Preserve it; this function is about readable
    # exact display, not compulsory expansion.
    if normalized_unit in {"억원", "조원"}:
        source = (value or "").strip().strip("+ -() ")
        sign = "-" if negative else ""
        return f"{sign}{source}{normalized_unit}"
    won = numeric * multiplier
    if negative:
        won = -abs(won)
    return format_won_exact(str(won))


__all__ = [
    "format_won_exact", "claim_won_display", "format_source_money_exact",
    "normalize_source_money_unit",
]
