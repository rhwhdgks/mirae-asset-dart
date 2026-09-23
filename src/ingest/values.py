"""값 상태 판정 — 「0」과 「없음」을 구분한다 (3차 검수 `value_status`).

## 왜 필요한가

원문은 값이 없을 때 `-` 를 쓴다. 지금은 그것이 문자열로만 남아 있어
**「0원」과 「미기재」가 구분되지 않는다.**

```text
계약금액(원)  =  "-"     →  0 원인가?  공시유보인가?  해당사항 없음인가?
```

실측: `Field` 1,640,264건 중 `-` 가 **572,001건(34.9%)** 이다.
공급계약 67건은 공시유보로 `-` 였고(§6.13), 이를 0 으로 읽으면 계약금액이 0 원이 된다.

`Fact` 는 사정이 다르다 — 재무제표 셀은 전부 숫자라 `not_reported` 가 0건이고
갈리는 것은 `literal`(95.4%) 과 `explicit_zero`(4.6%) 뿐이다.
**`0` 을 「값 없음」으로 접지 않는 것**이 여기서의 요점이다.
"""

from __future__ import annotations

import re
from typing import Literal

__all__ = ["ValueStatus", "classify", "STATUSES"]

ValueStatus = Literal["literal", "explicit_zero", "not_reported", "empty",
                      "percentage", "non_numeric", "parse_error"]

STATUSES: tuple[str, ...] = ("literal", "explicit_zero", "not_reported", "empty",
                             "percentage", "non_numeric", "parse_error")

#: 원문이 「값 없음」을 나타내는 표기. 전각 대시와 빈 괄호도 포함한다.
_NOT_REPORTED = frozenset({"-", "–", "—", "ㅡ", "()", "( )", "N/A", "n/a"})
_NUMERIC = re.compile(r"^\(?\s*-?[\d,]+(?:\.\d+)?\s*\)?$")
#: `14.10%` — 숫자지만 **단위가 아니라 비율**이다. 금액과 섞으면 안 되므로 따로 표시한다.
_PERCENT = re.compile(r"^-?[\d,]+(?:\.\d+)?\s*%$")


def classify(text: str | None, value: float | None) -> ValueStatus:
    """`(원문 텍스트, 파싱된 값)` → 상태.

    파싱 결과만으로는 판정할 수 없다 — `None` 이 `-`·공란·문자열을 전부 뭉치기 때문이다.
    그래서 **원문 텍스트를 함께 본다.**
    """
    raw = (text or "").strip()
    if value is not None:
        return "explicit_zero" if value == 0 else "literal"
    if not raw:
        return "empty"
    if raw in _NOT_REPORTED:
        return "not_reported"
    if _PERCENT.match(raw):
        # 실측 32건이 전부 `14.10%` 형태였다. 금액 파서가 못 읽는 것이 정상이며
        # **결함이 아니다** — `parse_error` 로 두면 진짜 결함 신호가 묻힌다.
        return "percentage"
    if _NUMERIC.match(raw):
        return "parse_error"        # 숫자처럼 보이는데 못 읽었다 — 조용히 넘기지 않는다
    return "non_numeric"
