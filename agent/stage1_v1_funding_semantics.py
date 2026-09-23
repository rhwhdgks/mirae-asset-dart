"""Question-grounded semantics shared by funding collection resolution.

Funding categories are selector facets, while phrases such as ``주요 조건``
and ``정정 이력`` are answer demands.  HCX can flatten both into the same
``field_surfaces`` array.  This module keeps the distinction deterministic and
shared between regrounding, resolution, and compilation.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


FUNDING_CATEGORY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("유상증자", ("유상증자",)),
    ("전환사채", ("전환사채", "CB")),
    ("신주인수권부사채", ("신주인수권부사채", "BW")),
    ("교환사채", ("교환사채", "EB")),
)
FUNDING_CATEGORY_TYPES = frozenset(
    canonical for canonical, _aliases in FUNDING_CATEGORY_ALIASES)


def _contains_alias(text: str, alias: str) -> bool:
    if alias.isascii():
        return re.search(
            rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])",
            text, flags=re.IGNORECASE) is not None
    return alias in text


def funding_categories(*surfaces: str) -> tuple[str, ...]:
    """Return only canonical funding types explicitly named by the user."""

    text = " ".join(surface for surface in surfaces if surface)
    return tuple(
        canonical
        for canonical, aliases in FUNDING_CATEGORY_ALIASES
        if any(_contains_alias(text, alias) for alias in aliases)
    )


def funding_requested_slots(field_surfaces: Iterable[str]) -> list[str]:
    """Map grounded answer-demand surfaces to stable Stage2 slot names."""

    fields = list(field_surfaces)
    categories = funding_categories(*fields)
    if fields and len(categories) == len(fields):
        # A pure type enumeration means "return the standard funding record
        # grouped by type".  The categories themselves remain selector facets.
        return ["조달유형", "결정금액", "결정일", "최신유효본여부"]

    slots: list[str] = []
    for surface in fields:
        compact = re.sub(r"\s+", "", surface)
        if "유형" in compact:
            slot = "조달유형"
        elif "금액" in compact or "규모" in compact:
            slot = "결정금액"
        elif "조건" in compact:
            slot = "주요조건"
        elif "결정일" in compact or "날짜" in compact or "일자" in compact:
            slot = "결정일"
        elif "정정" in compact and "이력" in compact:
            slot = "정정이력"
        elif "최신유효본" in compact:
            slot = "최신유효본여부"
        else:
            slot = compact
        if slot and slot not in slots:
            slots.append(slot)
    return slots


_COMPARISON_DEMANDS = (
    re.compile(r"유형\s*별"),
    re.compile(r"(?:주요\s*)?조건"),
    re.compile(r"정정\s*이력"),
    re.compile(r"최신\s*유효본(?:\s*여부)?"),
)


def funding_comparison_demand_surfaces(question: str) -> tuple[str, ...]:
    """Return question-literal funding comparison demands in question order.

    Funding category names select the event set; they do not by themselves
    describe what to return for a comparison.  Keeping this extraction in one
    place lets a one-item provider output and a tightly split two-item output
    recover the same literal demand surfaces without inventing canonical
    slots.
    """

    matches = [
        match
        for pattern in _COMPARISON_DEMANDS
        if (match := pattern.search(question)) is not None
    ]
    matches.sort(key=lambda match: match.start())
    return tuple(dict.fromkeys(match.group(0) for match in matches))


def recover_funding_comparison_fields(
        question: str, item: Any,
        ) -> tuple[str, ...] | None:
    """Recover explicit comparison demands lost behind category labels.

    No canonical slot is invented here.  Every returned surface is an exact
    span of the original question.  Single-category and non-comparison requests
    are deliberately left untouched.
    """

    if item.operation != "compare":
        return None
    categories = funding_categories(question, item.target.surface)
    if len(categories) < 2:
        return None
    current = list(item.output.field_surfaces)
    if not current or len(funding_categories(*current)) != len(current):
        return None
    surfaces = funding_comparison_demand_surfaces(question)
    return surfaces or None


__all__ = [
    "FUNDING_CATEGORY_ALIASES", "FUNDING_CATEGORY_TYPES",
    "funding_categories", "funding_requested_slots",
    "funding_comparison_demand_surfaces",
    "recover_funding_comparison_fields",
]
