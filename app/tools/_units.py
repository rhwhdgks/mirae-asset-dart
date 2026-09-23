from decimal import Decimal

_SCALE = {"원": 1, "천원": 10**3, "백만원": 10**6, "십억원": 10**9, "억원": 10**8, "조원": 10**12}


def to_won(v: Decimal, unit: str | None) -> Decimal | None:
    s = _SCALE.get((unit or "").strip())
    return None if s is None else v * Decimal(s)


import re as _re


def parse_money(text: str):
    """'9,603,075,000,000원' / '(10,833,917) 백만원' / '28.5%' → (Decimal, unit, won)."""
    if not text:
        return None, None, None
    m = _re.search(r"([\(\-]?)(\d[\d,]*(?:\.\d+)?)\)?\s*(원|백만원|천원|십억원|억원|조원|%)?", text.replace(" ", ""))
    if not m:
        return None, None, None
    neg = m.group(1) in ("(", "-")
    d = Decimal(m.group(2).replace(",", ""))
    if neg:
        d = -d
    unit = m.group(3)
    return d, unit, (to_won(d, unit) if unit and unit != "%" else None)
