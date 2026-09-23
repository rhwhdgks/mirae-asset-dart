"""Section-title-backed proof coordinates for structured narrative routes.

The canonical ``sections.parquet`` already contains the source title and table
count.  Stage1 therefore does not require the optional cell extraction merely
to route Stage2 to a uniquely identified investment-plan section.  This module
never reads answer values: it proves only the section coordinate and the
stable output-slot contract that Stage2 must extract from that section.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


INVESTMENT_PLAN_SLOTS = ("투자대상", "목적", "금액", "기간")
INVESTMENT_SPENT_SLOT = "기지출금액"
_INVESTMENT_SLOT_ALIASES = {
    "투자대상": ("투자대상", "대상자산", "투자명"),
    "목적": ("투자목적", "목적"),
    "금액": ("투자액", "투자금액", "총소요자금", "총투자액", "금액"),
    "기간": ("투자기간", "기간"),
    INVESTMENT_SPENT_SLOT: (
        "기지출금액", "기지출액", "누적지출금액", "실제지출금액",
        "집행금액", "기투자금액", "투자실적",
    ),
}


def _key(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", value or "")


def canonical_investment_slot(surface: str) -> str | None:
    """Map plan and execution amount labels to distinct semantic roles."""
    key = _key(surface)
    matches = [role for role, aliases in _INVESTMENT_SLOT_ALIASES.items()
               if key in {_key(alias) for alias in aliases}]
    return matches[0] if len(matches) == 1 else None


def investment_plan_section_slot_proofs(
        canonical: Any, *, document_id: str,
        requested_slots: tuple[str, ...] = INVESTMENT_PLAN_SLOTS,
        ) -> tuple[str, list[str], list[str]] | None:
    """Return unique section heading, four slot proofs, and canonical slots.

    A candidate must be a section in the exact selected document, explicitly
    titled as an equipment investment status/plan section, and contain at
    least one table.  Multiple candidates fail closed.  The per-slot suffixes
    describe the extraction contract; they are not claims that values were
    already extracted during Stage1.
    """

    import pyarrow.parquet as pq

    root = Path(getattr(canonical, "root", "out/canonical"))
    path = root / "sections.parquet"
    if not path.is_file():
        return None
    table = pq.read_table(
        path,
        columns=[
            "doc_id", "block_id", "source_file_id", "locator", "title",
            "n_tables",
        ],
        filters=[("doc_id", "=", document_id)],
    )
    rows = table.to_pylist()
    candidates = [
        row for row in rows
        if int(row.get("n_tables") or 0) > 0
        and "설비투자현황및계획" in _key(str(row.get("title") or ""))
    ]
    if len(candidates) != 1:
        return None
    row = candidates[0]
    raw_title = str(row["title"]).strip()
    heading = re.search(r"설비\s*투자\s*현황\s*및\s*계획", raw_title)
    if heading is None:
        return None
    # Some viewer titles append the first body sentence to the section name.
    # The fixed heading grammar is part of this section contract; values and
    # body prose remain outside Stage1.
    title = heading.group(0)
    coordinate = (
        f"source-section:{document_id}:{row['source_file_id']}:"
        f"{row['block_id']}:{row['locator']}"
    )
    canonical_slots = tuple(canonical_investment_slot(slot)
                            for slot in requested_slots)
    if (not canonical_slots or any(slot is None for slot in canonical_slots)
            or len(set(canonical_slots)) != len(canonical_slots)):
        return None
    slots = tuple(slot for slot in canonical_slots if slot is not None)
    proofs = [f"{coordinate}:slot:{slot}" for slot in slots]
    return title, proofs, list(slots)


__all__ = [
    "INVESTMENT_PLAN_SLOTS", "INVESTMENT_SPENT_SLOT",
    "canonical_investment_slot", "investment_plan_section_slot_proofs",
]
