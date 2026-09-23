"""답변 문장 조립에 쓰는 한국어 보조 — 조사·상태·공개 발췌 경계."""
from __future__ import annotations

import re

#: 지분공시 보고 주체 성명 부분 가림은 agent/(Stage1)·app/(Stage2) 공용이라
#: agent 쪽에 단일 출처를 두고 여기서는 재노출만 한다 — app은 agent에
#: 의존해도 되지만 반대는 안 되기 때문이다. 자세한 설명은 그 모듈 docstring.
from agent.holding_subject_mask import (  # noqa: F401
    mask_confirmed_person_name,
    mask_holding_subject_name,
    party_is_private_person,
)

#: typed 상태값 → 사용자 문구. 사건 상태(active…)·문서 정정 여부·비교 판정을 한 곳에서 관리한다.
STATE_KO = {
    "active": "유효(진행 중)", "terminated": "해지됨", "not_disclosed": "미공시",
    "no_termination_observed": "해지·종료 공시 미확인",
    "termination_observed": "해지·종료 공시 확인",
    "facility_before_start": "공시상 투자 시작일 이전",
    "facility_usage_approved": "건축물 사용승인 확인",
    "withheld": "공시유보(확인 불가)", "corrected": "정정 있음", "no_correction": "정정 없음",
    "same": "동일", "different": "상이",
}


#: 공시 표가 두 글자 머리말을 시각적으로 맞추려고 넣는 정렬용 공백.
#: ``년 말``·``진행 중``·``가 더`` 같은 정상 띄어쓰기와 구분해야 하므로
#: 일괄 제거하지 않고 실제로 관측된 머리말만 목록으로 둔다.
_PADDED_HEADINGS = (
    "품 목", "구 분", "합 계", "금 액", "부 문", "소 계", "총 계", "비 고",
    "당 기", "전 기", "대 상", "순 위", "회 사", "기 타", "내 용", "지 역",
)
_PADDED_HEADING = re.compile(
    "|".join(re.escape(value) for value in _PADDED_HEADINGS))


def collapse_padded_headings(text: str) -> str:
    """공시 표의 정렬용 공백을 사용자 표면에서 지운다.

    DART 표는 ``구 분``·``합 계`` 처럼 두 글자 머리말 사이에 공백을 넣어 폭을
    맞춘다. 원문에서는 정렬이지만 답변 문장 안에서는 오타로 읽힌다.

    **정상 띄어쓰기와 섞이므로 일괄 제거하지 않는다.** ``2025년 말``·``진행
    중``·``삼성전자가 더 큽니다`` 도 같은 모양이라, 실제로 관측된 머리말만
    지운다.
    """

    return _PADDED_HEADING.sub(lambda m: m.group(0).replace(" ", ""), text or "")


#: 숫자를 우리말로 읽었을 때 끝소리에 받침이 있는가. 0 영·1 일·3 삼·6 육·
#: 7 칠·8 팔은 받침으로 끝나고 2 이·4 사·5 오·9 구는 그렇지 않다.
_DIGIT_BATCHIM = {"0": True, "1": True, "2": False, "3": True, "4": False,
                  "5": False, "6": True, "7": True, "8": True, "9": False}

#: 알파벳을 우리말로 읽었을 때의 받침. L 엘·M 엠·N 엔·R 알만 받침으로 끝난다.
_LATIN_BATCHIM = frozenset("LMNR")

#: 글자로 읽는 약어의 최대 길이. 코퍼스의 HMM·KT·LG·SK·NC·SDI 가 여기 든다.
_ACRONYM_MAX = 3

#: 낱말 끝에 붙은 알파벳 토막.
_TRAILING_LATIN = re.compile(r"[A-Za-z]+$")

#: 받침이 ㄹ 이면 「으로」가 아니라 「로」를 쓴다. 조사 짝이 이것일 때만 다르다.
_RIEUL_DIGITS = frozenset("178")
_RIEUL_LATIN = frozenset("LR")


def _ends_with_batchim(word: str) -> tuple[bool, bool]:
    """(받침 있음, 그 받침이 ㄹ) — 읽는 소리 기준.

    「HMM가 더 큽니다」·「2025-05-30로」가 나온 것은 영문·숫자로 끝나면
    무조건 받침이 없다고 봤기 때문이다. 조사는 글자가 아니라 읽는 소리가
    정한다. HMM 은 「에이치엠엠」이라 ㅁ 으로 끝나고, 30 은 「삼십」이라
    ㅂ 으로 끝난다.
    """

    ch = word[-1]
    if "가" <= ch <= "힣":
        code = (ord(ch) - 0xAC00) % 28
        return bool(code), code == 8          # 8 = ㄹ
    if ch.isdigit():
        return _DIGIT_BATCHIM[ch], ch in _RIEUL_DIGITS
    if ch.isascii() and ch.isalpha():
        # 약어는 글자로 읽고(HMM = 에이치엠엠) 낱말은 낱말로 읽는다
        # (NAVER = 네이버). 글자로 읽는 규칙을 낱말에 대면 끝 글자 R 을
        # 「알」로 보아 「NAVER이」가 된다.
        #
        # 어느 쪽인지는 표기가 말해 준다. 짧은 전대문자는 글자로 읽고, 그보다
        # 길거나 소문자가 섞이면 낱말로 읽는다. 낱말의 우리말 소리는 알 수
        # 없으므로 받침 없음으로 둔다 — 예전 동작이고, 틀려도 「가」 쪽이다.
        run = _TRAILING_LATIN.search(word).group(0)
        if not (run.isupper() and len(run) <= _ACRONYM_MAX):
            return False, False
        upper = ch.upper()
        return upper in _LATIN_BATCHIM, upper in _RIEUL_LATIN
    return False, False


def josa(word: str, with_batchim: str, without: str) -> str:
    """받침 유무로 조사 선택 (예: ``josa(name, "이", "가")``).

    영문·숫자로 끝나면 그 글자를 우리말로 읽은 소리를 본다 — ``HMM`` 은
    ``HMM이``, ``2025-05-30`` 은 ``2025-05-30으로`` 가 된다. 받침이 ㄹ 이면
    ``으로/로`` 짝에서만 받침 없는 쪽을 고른다.
    """

    if not word:
        return without
    batchim, rieul = _ends_with_batchim(word)
    if rieul and {with_batchim, without} == {"으로", "로"}:
        return "로"
    return with_batchim if batchim else without


_COMPLETE_SENTENCE_END = re.compile(r"(?<![A-Za-z0-9])[.!?](?=\s|$)")
_NEXT_KOREAN_CLAUSE = re.compile(r"(?<!\S)[가-힣]\.\s+")


def complete_source_excerpt(text: str | None, *, max_chars: int) -> str:
    """Return a bounded excerpt ending at a complete source unit.

    DART form prose often consists of labelled clauses (``가.``, ``나.``)
    without terminal punctuation.  A raw character slice can therefore turn
    ``2023년말`` into ``2023`` and expose a different-looking fact.  This
    helper keeps only a complete sentence or a complete labelled clause.  If
    no such boundary fits inside the budget it returns an empty string rather
    than publishing an unmarked fragment.  The ellipsis is display metadata;
    the underlying claim citation still points to the full verified cell.
    """

    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    value = " ".join((text or "").split())
    if len(value) <= max_chars:
        return value

    window = value[:max_chars]
    clause_markers = list(_NEXT_KOREAN_CLAUSE.finditer(window))
    clause_periods = {match.start() + 1 for match in clause_markers}
    boundaries = [
        match.end() for match in _COMPLETE_SENTENCE_END.finditer(window)
        if match.start() not in clause_periods
    ]
    boundaries.extend(
        match.start() for match in clause_markers if match.start() > 0)
    # Tiny headings or abbreviations near the start are not useful excerpts.
    minimum = max(20, min(80, max_chars // 4))
    usable = [position for position in boundaries if position >= minimum]
    if not usable:
        return ""
    excerpt = value[:max(usable)].rstrip(" ,;:-")
    return excerpt + " …"


def fold_shared_label_prefix(
        label_a: str, label_b: str) -> tuple[str, str, str] | None:
    """Split two operand labels at their shared leading words.

    이슈 #118 — 두 피연산자의 **완성 라벨을 통째로** 이으면 회사명·기간·기준이
    두 벌 들어간다(``아모레퍼시픽 2025-12-31 기준 연결 자산총계와 아모레퍼시픽
    2025-12-31 기준 별도 자산총계의 차이``). 낱말 단위로 앞에서부터 일치하는
    부분만 한 번 떼어 두고, 서로 다른 나머지만 나란히 둔다.

        전: 아모레퍼시픽 2025-12-31 기준 연결 자산총계와
            아모레퍼시픽 2025-12-31 기준 별도 자산총계의 차이는 …
        후: 아모레퍼시픽 2025-12-31 기준 연결 자산총계와 별도 자산총계의
            차이는 …

    회사가 다르면(따라서 맨 앞 낱말부터 갈리면) 공유 접두가 없어 ``None`` 을
    돌려주므로 종전처럼 두 라벨을 그대로 쓴다. 접두가 라벨 전체를 삼켜 한쪽
    나머지가 비어도(동일 라벨) 마찬가지로 ``None`` — 뜻이 서지 않는 접기를
    피한다.

    Returns ``(shared_prefix, remainder_a, remainder_b)`` or ``None``.
    """

    tokens_a = label_a.split()
    tokens_b = label_b.split()
    if not tokens_a or not tokens_b:
        return None
    shared = 0
    limit = min(len(tokens_a), len(tokens_b))
    while shared < limit and tokens_a[shared] == tokens_b[shared]:
        shared += 1
    if shared == 0 or shared == len(tokens_a) or shared == len(tokens_b):
        return None
    prefix = " ".join(tokens_a[:shared])
    remainder_a = " ".join(tokens_a[shared:])
    remainder_b = " ".join(tokens_b[shared:])
    return prefix, remainder_a, remainder_b


#: 라벨 첫 낱말이 회사명 자리다(``아모레퍼시픽 2025-12-31 기준 …``). 순수
#: 숫자·기호뿐인 토큰은 회사명일 수 없으므로 우연히 같은 연도로 시작하는
#: 두 문장을 같은 회사로 착각해 접지 않는다.
_NON_SUBJECT_TOKEN = re.compile(r"^[0-9.,%\-~()]+$")


def fold_repeated_claim_subject(
        previous_label: str | None, label: str) -> str:
    """Drop a claim label's leading word when the prior sentence had it.

    이슈 #118 ② — 한 답변의 연속한 claim 문장이 모두 같은 회사로 시작하면
    (라벨이 항상 ``<회사> <날짜 등> <지표>`` 꼴이라 첫 낱말이 회사다) 두
    번째 문장부터는 그 회사명을 생략한다.

        전: NAVER 2025년 연결 매출액은 …
            NAVER 2024년 연결 매출액은 …
        후: NAVER 2025년 연결 매출액은 …
            2024년 연결 매출액은 …

    첫 낱말이 다르면(다른 회사가 섞이면) 아무것도 하지 않는다 — 회사를
    구별할 유일한 단서를 잃지 않기 위해서다. 첫 낱말이 숫자·기호뿐이면
    (연도 등 우연의 일치) 마찬가지로 접지 않는다. 라벨이 한 낱말뿐이면
    (접으면 빈 문장이 될 라벨) 그대로 둔다.
    """

    if not previous_label or not label:
        return label
    tokens = label.split()
    if len(tokens) < 2:
        return label
    previous_tokens = previous_label.split()
    if not previous_tokens or tokens[0] != previous_tokens[0]:
        return label
    if _NON_SUBJECT_TOKEN.match(tokens[0]):
        return label
    return " ".join(tokens[1:])


#: 공시 원문이 항목을 나열할 때 쓰는 앞머리 표시 — ``가.``·``나)``·``1)``·
#: ``(1)``. 형제 항목이 함께 오지 않으면(인용문 하나만 옮겨 온 문맥에서는
#: 그런 경우가 대부분이다) 답에서는 무엇을 세는 번호인지 알 수 없는 낱글자만
#: 남는다(이슈 #119, 실례 G-I-006 ``나. '2. 해지내역'의 …``). 숫자 뒤에
#: 곧장 숫자가 이어지면(``12. 31일`` · ``2. 5억원`` 처럼 날짜·금액이 공백
#: 하나로 갈라진 것일 수 있다) 항목 번호로 보지 않는다.
_LEADING_ITEM_NUMBER = re.compile(
    r"^(?:[가-힣]\.\s+|[가-힣]\)\s+|\(?\d{1,2}\)\s+|\d{1,2}\.\s+)(?!\d)")


def strip_leading_item_number(text: str) -> str:
    """Drop a disclosure item marker at the very front of a quoted excerpt.

    ``인용문 맨 앞``만 본다 — 라벨이 이미 그 절을 가리키므로 앞머리 번호는
    중복이지만, 문장 **중간**의 같은 모양(``'2. 해지내역'`` 같은 인용 안
    참조)은 원문 그대로 남긴다. 이 함수는 문자열 맨 앞 한 번만 본다.
    """

    if not text:
        return text
    return _LEADING_ITEM_NUMBER.sub("", text, count=1)


__all__ = [
    "STATE_KO", "complete_source_excerpt", "fold_repeated_claim_subject",
    "fold_shared_label_prefix", "josa", "mask_confirmed_person_name",
    "mask_holding_subject_name", "party_is_private_person",
    "strip_leading_item_number",
]
