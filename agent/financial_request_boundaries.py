"""Literal financial safety boundaries; no issuer or answer-value fixtures."""

from __future__ import annotations

import re


def eps_cumulative_subtraction(question: str) -> bool:
    """EPS is a ratio with a period-specific weighted share denominator."""
    return bool(
        re.search(r"주당\s*(?:순)?이익|\beps\b", question, re.I)
        and re.search(r"누적|반기|연간", question)
        and re.search(r"빼서|빼면|차감|뺀\s*값", question)
        and re.search(r"분기", question)
    )


def explicit_krw_fx_fallback(question: str) -> bool:
    """Only an explicit no-invented-rate request offering KRW as fallback."""
    return bool(
        re.search(r"달러|USD|미국\s*달러", question, re.I)
        and re.search(r"환산", question)
        and re.search(r"환율", question)
        and re.search(r"기준일", question)
        and re.search(r"확인할\s*수\s*없|못\s*찾|없다면", question)
        and re.search(r"원화", question)
        and re.search(r"임의.*(?:쓰지|사용하지)|추정하지", question)
    )
