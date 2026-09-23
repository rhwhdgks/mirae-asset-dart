#!/usr/bin/env python3
"""한국어 금액 표면을 원 단위 정수로 환산한다.

읽기 쉬운 답변으로 바꾼 뒤 채점기가 자릿수 대조를 하면 옳은 답이 전부 실패로
찍힌다. ``4,049,402백만원`` 과 ``4조 494억 200만원`` 은 같은 값인데 숫자 문자열이
겹치지 않기 때문이다. 값 비교는 표면이 아니라 환산한 정수로 해야 한다.
"""
from __future__ import annotations

import re

#: 만 단위 계층. 큰 것부터 읽어야 `조`·`억`·`만` 순서를 지킨다.
_TIERS = (("조", 10 ** 12), ("억", 10 ** 8), ("만", 10 ** 4))
#: 원문 표기 단위. `백만원`처럼 단위 자체가 배수인 경우.
_SCALES = (("백만원", 10 ** 6), ("십억원", 10 ** 9), ("억원", 10 ** 8),
           ("조원", 10 ** 12), ("천원", 10 ** 3), ("원", 1))

_TOKEN = re.compile(
    r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<tier>조|억|만)?")


def parse_amounts(text: str) -> set[int]:
    """``text`` 안에서 읽어낼 수 있는 원 단위 금액을 모두 돌려준다.

    ``4조 494억 200만원`` 처럼 여러 계층이 이어지면 하나로 합산하고, 동시에
    각 조각도 따로 담는다. 채점은 기대값이 **어느 하나와 같으면** 통과로 본다.
    """
    values: set[int] = set()
    for match in re.finditer(
            r"(?:\d[\d,]*(?:\.\d+)?\s*(?:조|억|만)\s*)+\d*[\d,]*\s*원?"
            r"|\d[\d,]*(?:\.\d+)?\s*(?:백만원|십억원|억원|조원|천원|원)",
            text):
        chunk = match.group(0)
        total = 0
        used = False
        for token in _TOKEN.finditer(chunk):
            raw = token.group("num").replace(",", "")
            if not raw:
                continue
            number = float(raw)
            tier = token.group("tier")
            if tier:
                mult = dict(_TIERS)[tier]
                total += int(number * mult)
                values.add(int(number * mult))
                used = True
            else:
                scale = 1
                tail = chunk[token.end():]
                for name, factor in _SCALES:
                    if tail.lstrip().startswith(name):
                        scale = factor
                        break
                if used:
                    total += int(number * scale)
                else:
                    values.add(int(number * scale))
        if used and total:
            values.add(total)
    return values


def expected_value(text: str) -> int | None:
    """정답지 표기(``재고자산 1,788,772백만원``)를 원 단위로 환산한다."""
    match = re.search(
        r"\(?(\d[\d,]*(?:\.\d+)?)\)?\s*(백만원|십억원|억원|조원|천원|원|%)", text)
    if match is None:
        return None
    if match.group(2) == "%":
        return None
    factor = dict(_SCALES).get(match.group(2), 1)
    return int(float(match.group(1).replace(",", "")) * factor)
