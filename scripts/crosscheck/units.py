"""한국어 숫자 표기 파서 — 접수번호/금액/비율/주식수/날짜 추출.

app/tools/_units.py 의 to_won 규칙(원/천원/백만원/십억원/억원/조원)을 그대로 쓰되,
답변 텍스트에 흔한 「333조 6,059억 3,800만원」류 복합 표기를 조/억/만/원 토큰을
인접순으로 병합해 원 단위 Decimal 로 환산한다(app/tools/_units.py to_won 의 역방향).
"""
from __future__ import annotations

import re
from decimal import Decimal
from dataclasses import dataclass

UNIT_SCALE = {
    "원": Decimal(1),
    "천원": Decimal(10) ** 3,
    "백만원": Decimal(10) ** 6,
    "십억원": Decimal(10) ** 9,
    "억원": Decimal(10) ** 8,
    "조원": Decimal(10) ** 12,
}
CHAIN_SCALE = {"조": Decimal(10) ** 12, "억": Decimal(10) ** 8, "만": Decimal(10) ** 4}
CHAIN_ORDER = {"조": 0, "억": 1, "만": 2}

RCEPT_RE = re.compile(r"(?<!\d)\d{14}(?!\d)")

TOKEN_RE = re.compile(
    r"(?P<num>-?\d[\d,]*(?:\.\d+)?)"
    r"(?P<unit>조원|억원|천원|백만원|십억원|조|억|만|원|%p|%|배|주(?!당))"
)


@dataclass
class MoneySpan:
    start: int
    end: int
    value_won: Decimal
    text: str
    approx: bool
    granularity: Decimal  # 표시 최소 자릿수(반올림 허용폭 판단용)


@dataclass
class NumSpan:
    start: int
    end: int
    value: Decimal
    text: str
    kind: str  # 'percent' | 'percentpoint' | 'multiple' | 'share'


def _adjacent(text: str, prev_end: int, start: int) -> bool:
    return text[prev_end:start].strip() == ""


def _to_decimal(num_str: str) -> Decimal:
    return Decimal(num_str.replace(",", ""))


def _decimals(num_str: str) -> int:
    return len(num_str.split(".", 1)[1]) if "." in num_str else 0


def _gran(scale: Decimal, num_str: str) -> Decimal:
    """소수점 표기(예: '3.33조원', '80,991.5억원')만큼 반올림 허용폭을 좁힌다."""
    d = _decimals(num_str)
    return scale / (Decimal(10) ** d) if d else scale


def extract_money(text: str) -> list[MoneySpan]:
    """조/억/만/원 인접 체인과 단독 백만원류 표기를 원 단위 Decimal 로 병합."""
    out: list[MoneySpan] = []
    pending = None  # dict(start,end,parts:list[Decimal],last_unit)

    def flush():
        nonlocal pending
        if pending and pending["parts"]:
            val = pending["sign"] * sum(pending["parts"], Decimal(0))
            span_text = text[pending["start"]:pending["end"]]
            approx = _is_approx(text, pending["start"])
            gran = min(pending["grans"])
            out.append(MoneySpan(pending["start"], pending["end"], val, span_text, approx, gran))
        pending = None

    for m in TOKEN_RE.finditer(text):
        unit = m.group("unit")
        num_str = m.group("num")
        start, end = m.span()
        if unit in CHAIN_SCALE:
            num = _to_decimal(num_str)
            val = abs(num) * CHAIN_SCALE[unit]
            gran = _gran(CHAIN_SCALE[unit], num_str)
            if (pending and _adjacent(text, pending["end"], start)
                    and CHAIN_ORDER[unit] > CHAIN_ORDER[pending["last_unit"]]):
                # 체인 전체 부호는 첫 항에서 정해진다 — 「-30조 1,146억원」은
                # -(30조+1,146억)이지 -30조+1,146억 이 아니다.
                pending["parts"].append(val)
                pending["end"] = end
                pending["last_unit"] = unit
                pending["grans"].append(gran)
            else:
                flush()
                pending = {"start": start, "end": end, "parts": [val],
                           "last_unit": unit, "grans": [gran],
                           "sign": Decimal(-1) if num < 0 else Decimal(1)}
        elif unit == "원":
            num = _to_decimal(num_str)
            if pending and _adjacent(text, pending["end"], start):
                pending["parts"].append(abs(num))
                pending["end"] = end
                pending["grans"].append(_gran(Decimal(1), num_str))
                flush()
            else:
                flush()
                approx = _is_approx(text, start)
                out.append(MoneySpan(start, end, num, text[start:end], approx,
                                      _gran(Decimal(1), num_str)))
        elif unit in ("억원", "조원"):
            # 「6조 5,765억원」처럼 조/억 체인 뒤에 공백 없이 붙는 복합표기 —
            # '억'/'조' 체인 항으로 이어붙이고 '원'으로 종결한다.
            base = unit[0]  # '억' or '조'
            num = _to_decimal(num_str)
            val = abs(num) * CHAIN_SCALE[base]
            gran = _gran(CHAIN_SCALE[base], num_str)
            if (pending and _adjacent(text, pending["end"], start)
                    and CHAIN_ORDER[base] > CHAIN_ORDER[pending["last_unit"]]):
                pending["parts"].append(val)
                pending["end"] = end
                pending["grans"].append(gran)
                flush()
            else:
                flush()
                approx = _is_approx(text, start)
                out.append(MoneySpan(start, end, num * CHAIN_SCALE[base], text[start:end],
                                      approx, gran))
        elif unit in UNIT_SCALE:  # 천원/백만원/십억원 (단독 복합표기, 체인 없음)
            flush()
            num = _to_decimal(num_str)
            approx = _is_approx(text, start)
            out.append(MoneySpan(start, end, num * UNIT_SCALE[unit], text[start:end],
                                  approx, _gran(UNIT_SCALE[unit], num_str)))
        else:
            flush()
    flush()
    return out


def _is_approx(text: str, start: int) -> bool:
    window = text[max(0, start - 6):start]
    return "약" in window


def extract_other(text: str) -> list[NumSpan]:
    """%·%p·배·주 표기를 추출한다(금액 체인과는 병합하지 않음)."""
    out: list[NumSpan] = []
    for m in TOKEN_RE.finditer(text):
        unit = m.group("unit")
        if unit not in ("%p", "%", "배", "주"):
            continue
        num = _to_decimal(m.group("num"))
        kind = {"%p": "percentpoint", "%": "percent", "배": "multiple", "주": "share"}[unit]
        out.append(NumSpan(*m.span(), value=num, text=m.group(0), kind=kind))
    return out


def extract_rcept_nos(text: str) -> list[str]:
    return list(dict.fromkeys(RCEPT_RE.findall(text)))


DATE_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}"),
    re.compile(r"\d{4}\.\d{2}\.\d{2}"),
    re.compile(r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일"),
]


def extract_dates(text: str) -> list[str]:
    out = []
    for pat in DATE_PATTERNS:
        out.extend(pat.findall(text))
    return out
