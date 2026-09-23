"""질문 표면의 날짜 표기를 **한 곳에서** 읽는다.

같은 날을 가리키는 표기가 여러 개다 — ``2025년 12월 17일``·``2025-12-17``·
``25.12.17``. 이 해석이 모듈마다 다르면 한쪽은 통과시키고 다른 쪽은 거부하는
상태가 조용히 생긴다. 그래서 표기 → (연,월,일) 변환을 여기 하나만 둔다.

여기서 하는 일은 **표기 정규화뿐**이다. 그 날짜가 어느 회계기간인지, 조회 가능
범위인지는 각 호출자가 정한다.

생략형 복원의 범위
------------------
``2025년 12월 25일과 26일`` 의 ``26일`` 처럼 **병렬 구문에서 공통 접두부가
생략된** 경우만 복원한다. 「앞에 나온 날짜에서 물려받는다」로 넓히지 않는다.
거리만 보면 ``25일엔 살아있고 26일엔 끝난`` 같은 절(節) 경계도 넘어가고,
질문에 날짜가 여럿일 때 엉뚱한 쪽에서 상속받아 **틀린 날짜를 통과시킨다.**

복원은 **빠진 상위 슬롯만** 채우고 명시된 슬롯은 건드리지 않는다. 맨숫자는
원칙적으로 날짜가 아니다. 다만 완전한 일자 바로 뒤의 ``vs 26``은 날짜 비교임이
문법적으로 닫혀 있으므로 같은 연·월의 일자로만 복원한다.
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
import re
import unicodedata


#: 2000년대만 다룬다. 코퍼스가 그 범위이고, 1900년대로 해석할 여지를 두면
#: ``99년`` 같은 입력이 조용히 엉뚱한 기간이 된다.
_TWO_DIGIT_YEAR = re.compile(r"^([0-9]{2})(?=년|[-./][01]?[0-9][-./])")

#: 긴 것부터 시도한다. ``2025-12-17`` 에서 ``2025-12`` 를 먼저 잡으면 일자를 잃는다.
_DATE_FORMS = (
    re.compile(r"^(?P<y>20[0-9]{2})[-./](?P<m>[01]?[0-9])[-./](?P<d>[0-3]?[0-9])$"),
    re.compile(r"^(?P<y>20[0-9]{2})년(?P<m>[01]?[0-9])월(?P<d>[0-3]?[0-9])일?$"),
    re.compile(r"^(?P<y>20[0-9]{2})[-./](?P<m>[01]?[0-9])$"),
    re.compile(r"^(?P<y>20[0-9]{2})년(?P<m>[01]?[0-9])월$"),
    re.compile(r"^(?P<y>20[0-9]{2})년(?:말|연말|기말)$"),
    re.compile(r"^(?P<y>20[0-9]{2})년?$"),
)

#: 연도가 빠진 형태. 병렬 복원의 **대상**이지 단독으로는 날짜가 아니다.
_PARTIAL_FORMS = (
    re.compile(r"^(?P<m>[01]?[0-9])월(?P<d>[0-3]?[0-9])일$"),
    re.compile(r"^(?P<d>[0-3]?[0-9])일$"),
    re.compile(r"^(?P<m>[01]?[0-9])월$"),
)

#: 질문 본문의 날짜꼴 토막. 앞에 숫자가 붙은 경우를 제외해 ``2025년`` 의 뒤
#: 두 자리가 두 자리 연도로 잡히지 않게 한다. 긴 형태를 먼저 둔다.
_QUESTION_TOKEN = re.compile(
    r"(?<![0-9])(?:"
    r"(?:20[0-9]{2}|[0-9]{2})[-./][01]?[0-9][-./][0-3]?[0-9]"
    r"|(?:20[0-9]{2}|[0-9]{2})년\s*[01]?[0-9]월\s*[0-3]?[0-9]일"
    r"|(?:20[0-9]{2}|[0-9]{2})[-./][01]?[0-9](?![-./0-9])"
    r"|(?:20[0-9]{2}|[0-9]{2})년\s*[01]?[0-9]월"
    r"|20[0-9]{2}년"
    r"|[01]?[0-9]월\s*[0-3]?[0-9]일"
    r"|[0-3]?[0-9]일"
    r"|[01]?[0-9]월"
    r")"
)

#: 두 날짜 사이가 **이것뿐**일 때만 병렬로 본다. 사이에 다른 말이 끼면
#: 절이 갈린 것으로 보고 복원하지 않는다.
_COORDINATOR = re.compile(
    r"^\s*(?:과|와|이랑|랑|및|또는|vs\.?|versus|,|;|·|~|-)\s*$",
    flags=re.IGNORECASE)

#: 병렬을 **연결어미**로 잇는 경우. 「25일엔 살아있**고** 26일엔 끝난」처럼 한국어는
#: 접속사 없이 어미로 두 절을 잇는다. 그때도 뒤 날짜의 연·월은 앞에서 이어받는다.
#:
#: 짧게 제한한다 — 사이에 긴 말이 끼면 절이 갈린 것이고, 그때 이어받으면 엉뚱한
#: 달의 날짜를 만든다. 숫자가 끼어도 이어받지 않는다(다른 날짜가 사이에 있다).
_CLAUSE_LINK = re.compile(r"^[^0-9]{0,12}?(?:고|며|지만|인데|이고)\s*$")

_VS_BARE_DAY = re.compile(
    r"(?P<full>(?:20[0-9]{2}|[0-9]{2})[-./][01]?[0-9][-./][0-3]?[0-9]"
    r"|(?:20[0-9]{2}|[0-9]{2})년\s*[01]?[0-9]월\s*[0-3]?[0-9]일?)"
    r"\s*vs\.?\s*(?P<day>[0-3]?[0-9])(?![0-9])",
    flags=re.IGNORECASE,
)

DateParts = tuple[int, int | None, int | None]


@dataclass(frozen=True, slots=True)
class DateGrounding:
    """근거 판정과 **왜 그렇게 판정했는지**.

    bool 하나만 돌려주면 「질문에 아예 없는 날짜(cutoff 누출)」와 「병렬 복원이
    모호해서 막은 것」이 같은 실패로 보인다. 둘은 고칠 곳이 다르다.
    """

    grounded: bool
    reason: str | None = None


def compact_surface(value: str) -> str:
    """공백·전각·대소문자 차이를 지운다. 날짜 해석 전 공통 전처리다."""

    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value).casefold())


def expand_two_digit_year(compact: str) -> str:
    """``25년`` 처럼 두 자리로 쓴 연도를 네 자리로 편다.

    구어체 질문에서 흔한 표기이고 **특정 연도·회사와 무관한 표기 규칙**이다.
    이미 네 자리인 것은 건드리지 않는다.

    **문자열 맨 앞의 연도 한 번만** 편다. 뒤따르는 월·일까지 잡으면
    ``25.12.17`` 이 ``2025.2012.17`` 로 깨진다.
    """

    return _TWO_DIGIT_YEAR.sub(lambda m: f"20{m.group(1)}", compact, count=1)


def _valid(year: int, month: int | None, day: int | None) -> bool:
    if month is not None and not 1 <= month <= 12:
        return False
    if day is not None:
        assert month is not None
        return 1 <= day <= monthrange(year, month)[1]
    return True


def parse_date_surface(value: str) -> DateParts | None:
    """날짜 표기를 ``(연, 월, 일)`` 로 읽는다. 날짜가 아니면 ``None``.

    월·일이 없는 표기는 해당 자리가 ``None`` 이다 — ``2025년`` 은
    ``(2025, None, None)``, ``2025년 12월`` 은 ``(2025, 12, None)``.
    **연도가 없는 표기는 날짜가 아니다** — 병렬 복원을 거쳐야 날짜가 된다.
    존재하지 않는 날짜(``2025-02-30``)도 날짜로 보지 않는다.
    """

    compact = expand_two_digit_year(compact_surface(value))
    for pattern in _DATE_FORMS:
        match = pattern.match(compact)
        if match is None:
            continue
        groups = match.groupdict()
        year = int(groups["y"])
        month = int(groups["m"]) if groups.get("m") else None
        day = int(groups["d"]) if groups.get("d") else None
        return (year, month, day) if _valid(year, month, day) else None
    return None


def _parse_partial(value: str) -> tuple[int | None, int | None] | None:
    """연도가 빠진 ``(월, 일)``. 복원 대상이 아니면 ``None``."""

    compact = compact_surface(value)
    for pattern in _PARTIAL_FORMS:
        match = pattern.match(compact)
        if match is None:
            continue
        groups = match.groupdict()
        month = int(groups["m"]) if groups.get("m") else None
        day = int(groups["d"]) if groups.get("d") else None
        if month is not None and not 1 <= month <= 12:
            return None
        if day is not None and not 1 <= day <= 31:
            return None
        return (month, day)
    return None


def _restore(anchor: DateParts, month: int | None, day: int | None,
             ) -> DateParts | None:
    """빠진 **상위** 슬롯만 anchor에서 채운다. 명시 슬롯은 건드리지 않는다."""

    year = anchor[0]
    filled_month = month if month is not None else anchor[1]
    if filled_month is None or day is None and month is None:
        return None
    return (year, filled_month, day) if _valid(year, filled_month, day) else None


def question_date_surfaces(question: str) -> frozenset[DateParts]:
    """질문이 실제로 가리키는 날짜들. 병렬 생략형은 복원해서 포함한다."""

    return _scan(question)[0]


def _scan(question: str) -> tuple[frozenset[DateParts], frozenset[DateParts]]:
    """``(확정 날짜, 복원이 모호해 버린 날짜)``.

    복원은 **직전 토막과의 사이가 접속사뿐일 때만** 한다. 그렇게 해도 후보가
    여럿 나오면(해석이 유일하지 않으면) 통째로 버린다 — fail-closed.
    """

    tokens: list[tuple[int, int, str]] = [
        (m.start(), m.end(), m.group(0))
        for m in _QUESTION_TOKEN.finditer(question)
    ]
    dates: set[DateParts] = set()
    ambiguous: set[DateParts] = set()
    resolved: list[tuple[int, DateParts]] = []   # (토큰 끝 위치, 확정 날짜)
    for start, end, text in tokens:
        parts = parse_date_surface(text)
        if parts is not None:
            dates.add(parts)
            resolved.append((end, parts))
            continue
        partial = _parse_partial(text)
        if partial is None:
            continue
        month, day = partial
        candidates = {
            restored
            for anchor_end, anchor in resolved
            if (_COORDINATOR.match(gap := question[anchor_end:start])
                or _CLAUSE_LINK.match(gap))
            and (restored := _restore(anchor, month, day)) is not None
        }
        if len(candidates) == 1:
            restored_date = next(iter(candidates))
            dates.add(restored_date)
            resolved.append((end, restored_date))
        elif candidates:
            ambiguous |= candidates

    # A bare number is accepted only in the closed comparison grammar
    # ``<full day> vs <day>``.  The adjacent complete date provides both
    # missing upper components; invalid calendar days are rejected.
    for match in _VS_BARE_DAY.finditer(question):
        anchor = parse_date_surface(match.group("full"))
        if anchor is None or anchor[1] is None or anchor[2] is None:
            continue
        candidate = _restore(anchor, None, int(match.group("day")))
        if candidate is not None:
            dates.add(candidate)
    return frozenset(dates), frozenset(ambiguous)


def check_question_date(expression: str, question: str) -> DateGrounding:
    """``expression`` 이 질문에 적힌 날짜와 **같은 날**을 가리키는가.

    글자가 아니라 날짜로 비교하되 **단위까지 같아야** 한다 — 질문이 일 단위인데
    연 단위로 답하면 조회 범위가 달라지므로 근거로 보지 않는다.
    """

    parts = parse_date_surface(expression)
    if parts is None:
        return DateGrounding(False, "not_a_date")
    dates, ambiguous = _scan(question)
    if parts in dates:
        return DateGrounding(True)
    if parts in ambiguous:
        return DateGrounding(False, "date_restoration_ambiguous")
    if any(existing[0] == parts[0] for existing in dates):
        return DateGrounding(False, "date_granularity_mismatch")
    return DateGrounding(False, "date_absent_from_question")


def denotes_question_date(expression: str, question: str) -> bool:
    return check_question_date(expression, question).grounded
