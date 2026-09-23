"""think_trace / retrieved_context 렌더러 + /answer 5필드 조립.

think_trace는 사후 작문이 아니라 orchestrator·tool·composer가 실행 중 남긴 TraceEvent를 그대로 텍스트로
직렬화한 것이다. safe 경계: 모든 문자열은 payload/trace에 이미 prompt-safe로 들어온 값이다.
"""
from __future__ import annotations

import re

from agent.display_units_v1 import parse_display_units_directive
from .money import claim_won_display, format_source_money_exact, format_won_exact

_STAGE_KO = {"handoff": "1 해석", "route": "2 라우팅", "tool": "3 조회", "evidence": "4 근거검증",
             "derivation": "5 계산", "premise": "6 전제검증", "verify": "7 확정판정", "compose": "8 답변생성"}

_INTERNAL_TABLE_LOCATOR = re.compile(
    r"(?:,\s*)?TABLE(?:-GROUP)?\[\d+\]"
    r"(?:/[A-Z][A-Z0-9_-]*\[\d+\])*"
)
_INTERNAL_MATRIX_LOCATOR = re.compile(
    r"(?:"
    r"[\[(]\s*(?:matrix/)?(?:task-\d+/)?cell-\d+\s*[\])]"
    r"|(?:matrix/)?task-\d+/cell-\d+"
    r"|(?<![A-Za-z0-9_])cell-\d+(?![A-Za-z0-9_])"
    r")"
)
_PUBLIC_MONEY_SURFACE = re.compile(
    # Parentheses are an accounting sign only as a complete pair.  Matching
    # the opening parenthesis of ``(15백만원 증가)`` as part of the number
    # would erase it during unit conversion and leave the public sentence as
    # ``1,500만원 증가)``.
    r"(?P<value>(?:[+-]?\d[\d,]*(?:\.\d+)?|\(\d[\d,]*(?:\.\d+)?\)))"
    # Disclosure tables and HCX sometimes insert one space before 원
    # (``13,429,929백만 원``).  It is the same closed unit vocabulary,
    # not a prose inference, so accept both forms at this final display layer.
    r"(?P<unit>조\s*원|억\s*원|백만\s*원|천\s*원|원)"
)

# Lifecycle observation states remain typed English tokens in the payload for
# audit and orchestration.  They are not user-facing vocabulary, however, and
# an HCX-written explanation can echo them outside the deterministic claim
# sentence.  Translate only these closed event-state tokens at the final
# display boundary, after all typed verification has completed.
_PUBLIC_EVENT_STATE = {
    "no_termination_observed": "해지·종료 공시 미확인",
    "termination_observed": "해지·종료 공시 확인",
    "facility_before_start": "공시상 투자 시작일 이전",
    "facility_usage_approved": "건축물 사용승인 확인",
}
_INTERNAL_EVENT_STATE = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(
        re.escape(state)
        for state in sorted(_PUBLIC_EVENT_STATE, key=len, reverse=True)
    ) + r")(?![A-Za-z0-9_])"
)

# HCX may alternate between ``근거 2026...``, ``출처: 2026...`` and the
# canonical ``접수번호 2026...`` even though all three refer to the same typed
# DART receipt.  Normalize only a citation marker immediately before a
# 14-digit receipt number; ordinary prose such as ``계산 근거:`` is untouched.
_PUBLIC_RECEIPT_MARKER = re.compile(
    r"(?<![A-Za-z0-9가-힣])(?:근거|출처|공시)\s*[:：]?\s*"
    r"(?:(?:접수번호)\s*)?(?=\d{14}\b)"
)
_ADJACENT_DUPLICATE_CITATION = re.compile(
    r"(?P<citation>\(접수번호\s+[^)]+\))(?:\s+(?P=citation))+"
)
_INLINE_DUPLICATE_CITATION = re.compile(
    r"(?P<prefix>(?:상태\s*근거\s*)?접수번호\s+"
    r"(?P<receipt>\d{14}))\s+\(접수번호\s+(?P=receipt)\)"
)

#: 문장 끝이나 중간에 붙는 괄호 인용 한 덩어리.
_CITATION_GROUP = re.compile(r"\s*\((?:근거[:：]\s*)?접수번호[^)]*\)")

#: 접수번호 14자리.
_RECEIPT_NO = re.compile(r"\d{14}")


def _demote_citations_to_basis(text: str) -> str:
    """계산 결과 문장의 접수번호를 지우고 "계산 근거:" 줄에만 남긴다.

    파생 계산 답변은 결론 문장과 그 아래 계산 근거 줄이 같은 접수번호를
    싣는다. 읽는 쪽에서는 같은 번호를 두 번 지나치게 되고, 그 번호가 어느
    값에서 나왔는지는 피연산자가 나열된 계산 근거 줄에서만 알 수 있다.
    근거 줄이 그 접수번호를 모두 담고 있을 때에만 결론 쪽을 지운다 —
    담지 못한 번호가 하나라도 있으면 근거가 사라지므로 줄을 그대로 둔다.
    """

    lines = text.splitlines()
    basis = [i for i, line in enumerate(lines)
             if line.lstrip().startswith("계산 근거:")]
    if not basis:
        return text
    covered = {no for i in basis for no in _RECEIPT_NO.findall(lines[i])}
    if not covered:
        return text

    def strip(match: re.Match) -> str:
        group = match.group(0)
        found = set(_RECEIPT_NO.findall(group))
        return "" if found and found <= covered else group

    for i, line in enumerate(lines):
        if i in basis:
            continue
        stripped = _CITATION_GROUP.sub(strip, line)
        # 인용만으로 이루어진 줄이 통째로 비면 근거 표시가 아니라 빈 줄이
        # 남는다. 그런 줄은 원래대로 둔다.
        if stripped.strip():
            lines[i] = stripped
    return "\n".join(lines)


#: 정본이 개인정보를 가린 자리표시. 저장·감사 좌표이지 읽는 말이 아니다.
_REDACTION = re.compile(r"['\u2018\u2019\"]?\[REDACTED:([A-Z_]+)\]['\u2018\u2019\"]?")

#: 가린 종류별로 사용자가 읽을 말.
_REDACTION_KO = {
    "PHONE": "비공개",
    "EMAIL": "비공개",
    "PERSON_NAME": "비공개",
    "RESIDENT_REGISTRATION_NO": "비공개",
    "REGISTRATION_NO": "비공개",
    "BANK_ACCOUNT": "비공개",
    "EMPLOYEE_ID": "비공개",
    "BIRTH_DATE": "비공개",
    "ADDRESS": "비공개",
    "OCCUPATION": "비공개",
}


#: 해지 공시를 찾지 못했다는 상태 표시.
_NO_TERMINATION = "해지·종료 공시 미확인"

#: 그 상태를 어떻게 읽어야 하는지 이미 밝히고 있는 표현.
_ALREADY_QUALIFIED = ("단정할 수 없", "확정할 수 없", "미확정", "볼 수는 없")

_NO_TERMINATION_NOTE = (
    "다만, 해지 공시를 찾지 못했다는 것이지 계약이 현재 정상 진행 중이라는 "
    "뜻은 아닙니다.")

#: 사건 상태를 typed 시점(``@YYYYMMDD``)별로 나란히 비교하는 답인지 가리는
#: 상태 값. output_id 는 ``{task_id}.{kind}@{tp}`` 꼴이라 ``@`` 뒤 시점 표시가
#: 서로 다른 claim이 둘 이상이면 같은 사건을 여러 기준일로 비교한 것이다.
_LIFECYCLE_STATES = frozenset({
    "no_termination_observed", "termination_observed", "terminated", "active",
    "facility_before_start", "facility_usage_approved",
})


#: 조·억·만·원이 모두 붙은 금액. 앞 두 자리만으로도 규모가 잡힌다.
_LONG_AMOUNT = re.compile(
    r"(?<![\d,])((?:[\d,]+조\s)?[\d,]+억)\s[\d,]+만\s[\d,]+원(?![\d,])")


def _lead_with_a_graspable_amount(text: str) -> str:
    """첫 문장의 긴 금액 앞에 어림한 규모를 먼저 놓는다.

    「105조 1,924억 3,796만 3,565원」을 첫 문장부터 들이대면 규모가 한눈에
    들어오지 않는다. 읽는 사람이 먼저 알고 싶은 것은 105조인지 10조인지다.

    정확한 값은 지우지 않고 바로 아래 줄로 내린다 — 단일 조회 답변에서는
    그 자리가 정확값의 유일한 자리이고, 계산 답변에서도 어림값만 남기면
    답 자체가 어림이 된다. 괄호로 같은 줄에 남기면 줄이 오히려 길어져,
    첫 줄을 짧게 만들려던 목적과 어긋난다.

    첫 문장의 첫 금액 하나만 바꾼다. 아래 표와 계산 근거 줄은 원문 그대로
    두어야 대조가 된다.
    """

    lines = (text or "").splitlines()
    if not lines:
        return text
    head = lines[0]
    match = _LONG_AMOUNT.search(head)
    if not match or "약 " in head:
        return text
    major = match.group(1)
    groups = re.fullmatch(r"(?:(?P<jo>[\d,]+)조\s)?(?P<eok>[\d,]+)억", major)
    minor = re.search(r"억\s([\d,]+)만", match.group(0))
    if groups and minor:
        eok = (int((groups.group('jo') or '0').replace(',', '')) * 10000
               + int(groups.group('eok').replace(',', ''))
               + (int(minor.group(1).replace(',', '')) >= 5000))
        rounded = format_won_exact(str(eok * 100000000))
    else:
        rounded = f"{major}원"
    lines[0] = (head[:match.start()] + f"약 {rounded}"
                + head[match.end():])
    lines.insert(1, f"정확한 금액: {match.group(0)}")
    return "\n".join(lines)


#: 왜 계산하지 않았는지를 밝히는 문장의 끝. 이 말이 있으면 대상을 확인하지
#: 못한 것이 아니라 계산을 하지 않기로 한 것이다.
_SAYS_WHY_NOT_COMPUTED = "계산하지 않았습니다"

#: 대상 자체를 확인하지 못했다는 총칭 거절. 위 문장과 함께 있으면 어긋난다.
_GENERIC_REFUSAL_SENTENCE = (
    "요청하신 대상이나 항목을 이번 공시자료에서 근거와 함께 확인할 수 없어, "
    "비슷하지만 다른 대상이나 항목의 내용을 대신 제시하지 않았습니다.")


def _drop_generic_refusal_after_a_reason(text: str) -> str:
    """왜 계산하지 않았는지 밝힌 답변에서 총칭 거절을 지운다.

    DEV-SAFE-012 는 부채총계와 자본총계를 제시하고, 부채비율은 공시에서 직접
    확인되는 값이 아니라 계산하지 않았다고 밝힌 **직후에** 그 대상을 확인할
    수 없다고 말한다. 바로 위에 값을 내놓고 확인하지 못했다고 하는 셈이다.

    44d5069 가 한계 문구를 붙이는 층에서 이미 닫았지만, 작문 모델이 두 문구를
    본문에 함께 옮겨 적으면 그 층을 지나쳐 나온다. 여기는 최종 표시 경계라
    본문에 옮겨 적힌 것도 함께 본다.

    지우는 것은 총칭 문장 하나뿐이고, 왜 계산하지 않았는지는 그대로 남는다.
    구체적인 사유가 없는 답변에서는 총칭 거절이 유일한 설명이므로 건드리지
    않는다.
    """

    if (_SAYS_WHY_NOT_COMPUTED not in (text or "")
            or _GENERIC_REFUSAL_SENTENCE not in text):
        return text
    out = []
    for line in text.splitlines():
        stripped = line.replace(_GENERIC_REFUSAL_SENTENCE, "").strip()
        # 총칭 문장만 있던 줄은 통째로 지우고, 다른 말과 섞여 있으면 그 말만 남긴다.
        if _GENERIC_REFUSAL_SENTENCE in line and not stripped.strip("다만, "):
            continue
        out.append(stripped if _GENERIC_REFUSAL_SENTENCE in line else line)
    return _merge_restated_qualifications("\n".join(out))


#: 한계 문구가 같은 말인지 가리는 꼬리 길이. 이보다 짧게 겹치는 것은 우연이다.
_QUALIFICATION_TAIL = 20


def _merge_restated_qualifications(text: str) -> str:
    """같은 한계를 말만 바꿔 두 번 적지 않는다.

    작문 모델이 한계 문구를 제 말로 옮겨 적고, 그 뒤에 typed 한계가 원문
    그대로 또 붙으면 같은 말이 두 줄이 된다.

        다만, 요청하신 부채비율은 … 계산하지 않았습니다.
        다만, 요청한 합계·비율은 … 계산하지 않았습니다.

    글자가 달라 기존의 같은-문장 제거로는 걸리지 않는다. 꼬리가 길게 겹치면
    같은 말로 보고 먼저 나온 쪽을 남긴다 — 앞의 것이 대개 무엇을 계산하지
    않았는지 이름을 대고 있다.

    한계 문구(「다만,」로 시작하는 줄)끼리만 본다. 정정 사유 목록처럼 꼬리가
    같은 것이 정상인 줄들은 건드리지 않는다 — 「2023-05-31 정정 사유: 계약금액
    변경.」과 「2023-11-23 …」은 서로 다른 공시다.
    """

    lines = text.splitlines()
    kept: list[str] = []
    tails: list[str] = []
    for line in lines:
        if not line.startswith("다만,"):
            kept.append(line)
            continue
        tail = line.rstrip()[-_QUALIFICATION_TAIL:]
        if any(tail == seen for seen in tails):
            continue
        tails.append(tail)
        kept.append(line)
    return "\n".join(kept)


def _is_multi_timepoint_state_comparison(payload) -> bool:
    """두 시점 이상의 사건 상태를 나란히 비교하는 답인지 판정한다.

    ``R-P-016`` 같은 「2025-12-25 vs 26 상태 비교」질문은 각 기준일의 상태를
    표로 나란히 보여준다. 그 표 자체가 「미확인」과 「해지됨」의 대비를
    드러내므로, 미확인 쪽에만 붙는 한계 문구는 이미 말한 것을 반복한다.
    """

    timepoints = {
        claim.output_id.rsplit("@", 1)[1]
        for claim in getattr(payload, "claims", ()) or ()
        if claim.state in _LIFECYCLE_STATES and "@" in (claim.output_id or "")
    }
    return len(timepoints) >= 2


def _qualify_absent_termination(text: str, payload=None) -> str:
    """해지 공시를 못 찾았다는 말에 그 뜻의 한계를 한 번 붙인다.

    「해지·종료 공시 미확인」은 정확한 말이지만 읽는 사람은 「계약이 살아
    있다」로 받아들이기 쉽다. 공시를 찾지 못한 것과 계약이 유효한 것은
    다르다.

    이 표시는 상태 라벨이라 결론 문장과 표 칸에 여러 번 나온다. 줄마다
    붙이면 표가 같은 경고로 뒤덮이므로 답변 끝에 한 번만 놓는다. 이미
    한계를 밝히고 있는 답변에는 덧붙이지 않는다. 여러 시점의 상태를 나란히
    비교하는 답에서는, 다른 시점에 이미 나온 상태(해지됨 등)가 같은 뜻의
    대비를 보여주므로 붙이지 않는다(#86-13).
    """

    if _NO_TERMINATION not in (text or ""):
        return text
    if any(mark in text for mark in _ALREADY_QUALIFIED):
        return text
    if payload is not None and _is_multi_timepoint_state_comparison(payload):
        return text
    return f"{text.rstrip()}\n{_NO_TERMINATION_NOTE}"


def _speak_redactions(text: str) -> str:
    """가린 자리에 내부 토큰 대신 읽을 수 있는 말을 놓는다.

    정본은 전화번호·이름 같은 개인정보를 `[REDACTED:PHONE]` 으로 바꿔
    저장한다. 가리는 것은 맞지만 그 토큰이 답변 문장에 그대로 실려
    「대표전화 '[REDACTED:PHONE]'를 통해」처럼 나갔다. 감싼 따옴표까지
    함께 걷어내고 「비공개」로 바꾼다. 가린 사실은 그대로 남는다.
    """

    return _REDACTION.sub(
        lambda m: _REDACTION_KO.get(m.group(1), "비공개"), text)


def _drop_self_named_citations(text: str) -> str:
    """문장이 이미 말한 접수번호를 괄호로 한 번 더 붙이지 않는다.

    정정 계보 답변에서는 접수번호가 곧 문장의 주어다.

        20230515001843 버전은 20230515부터 … 유효 (접수번호 20230515001843)

    괄호 쪽은 결속이 아니라 같은 숫자의 반복이다. 줄 본문이 그 접수번호를
    이미 담고 있을 때만 지우므로 근거는 줄에 그대로 남는다.
    """

    lines = text.splitlines()
    for i, line in enumerate(lines):
        groups = _CITATION_GROUP.findall(line)
        if not groups:
            continue
        body = _CITATION_GROUP.sub("", line)
        cited = set(_RECEIPT_NO.findall(" ".join(groups)))
        if not cited or not cited <= set(_RECEIPT_NO.findall(body)):
            continue
        stripped = body.rstrip()
        # 괄호를 떼고 나면 아무것도 안 남는 줄은 그대로 둔다.
        if stripped:
            lines[i] = stripped
    return "\n".join(lines)


def _hoist_run_citations(text: str) -> str:
    """연속한 줄이 같은 근거를 쓰면 그 구간의 머리에만 한 번 적는다.

    한 공시에서 읽어낸 값들이 줄줄이 이어질 때 줄마다 같은 14자리가 붙는다.

        제공 코퍼스의 최종 상태:
        … 상태: 해지됨 (접수번호 20251217800800)
        해지금액은 9조 6,030억 7,500만원입니다. (접수번호 20251217800800)
        해지사유: 거래 상대방의 계약 해지 통보 (접수번호 20251217800800)

    그 접수번호는 줄이 아니라 구간의 근거다. 구간 바로 위에 인용이 없는
    머리글이 있으면 거기로 올리고, 없으면 구간 첫 줄에만 남긴다. 어느 쪽이든
    그 근거가 어디까지 걸리는지는 구간의 경계가 그대로 말해 준다.
    """

    lines = text.splitlines()
    cited = [(i, tuple(sorted(set(_RECEIPT_NO.findall(
        " ".join(_CITATION_GROUP.findall(line)))))))
        for i, line in enumerate(lines) if _CITATION_GROUP.findall(line)]

    drop: list[int] = []
    heads: dict[int, str] = {}
    start = 0
    while start < len(cited):
        end = start
        while (end + 1 < len(cited)
               and cited[end + 1][1] == cited[start][1]
               and cited[end + 1][0] == cited[end][0] + 1
               # 이슈 #112 — 표 행(``|`` 로 시작)은 줄마다 독립된 셀이다.
               # 표 행 하나가 마침 이웃 불릿과 같은 접수번호를 인용한다고
               # 그 표 행과 옆 불릿을 한 구간으로 묶으면, 표 안쪽 줄이
               # "머리글"로 뽑혀 표가 아닌 문장처럼 다시 쓰이거나(콜론이
               # 붙는 등) 옆 불릿의 근거가 지워진다. 표 행이 낀 이웃은
               # 절대 한 구간으로 묶지 않는다.
               and not lines[cited[end][0]].lstrip().startswith("|")
               and not lines[cited[end + 1][0]].lstrip().startswith("|")):
            end += 1
        receipts = cited[start][1]
        if end > start and receipts:
            first = cited[start][0]
            head = first - 1
            # 머리글은 인용이 없고 내용이 있는 바로 윗줄이어야 한다. 표
            # 행은 머리글이 될 수 없다 — 거기에 이어붙이면 표가 깨진다.
            if (head >= 0 and lines[head].strip()
                    and not lines[head].rstrip().endswith((".", "!", "?", "。"))
                    and not lines[head].lstrip().startswith("|")
                    and not _CITATION_GROUP.findall(lines[head])
                    and not _RECEIPT_NO.findall(lines[head])):
                heads[head] = (lines[head].rstrip().rstrip(":")
                               + " (근거: "
                               + ", ".join("접수번호 " + r for r in receipts)
                               + "):")
                drop.extend(cited[k][0] for k in range(start, end + 1))
            else:
                drop.extend(cited[k][0] for k in range(start + 1, end + 1))
        start = end + 1

    for i, head in heads.items():
        lines[i] = head
    for i in drop:
        stripped = _CITATION_GROUP.sub("", lines[i]).rstrip()
        if stripped:
            lines[i] = stripped
    return "\n".join(lines)


def _collect_uniform_citations(text: str) -> str:
    """답변 전체가 같은 근거를 쓸 때 접수번호를 맨 끝 한 줄로 모은다.

    14자리 숫자는 읽는 사람에게 아무 뜻이 없는데 문장마다 끼어 있어 시선이
    반드시 걸린다. 전수 395건에서 인용 표기가 전체 글자의 12%를 차지했고,
    978자 답변의 42%가 접수번호인 경우도 있었다.

    옮겨도 잃는 것이 없다고 증명되는 경우에만 옮긴다 — 인용이 달린 줄들이
    전부 똑같은 접수번호 묶음일 때다. 줄마다 근거가 바뀌는 답변(정정 이력
    처럼 한 줄이 곧 한 공시인 경우)은 어느 사실이 어느 문서에서 왔는지가
    줄 위치로만 표현되므로 그대로 둔다.

    "계산 근거:" 줄을 가진 답변도 손대지 않는다. 그 줄이 이미 지정된 근거
    자리이고, _demote_citations_to_basis 가 그리로 모으고 있다.
    """

    lines = text.splitlines()
    if any(line.lstrip().startswith("계산 근거:") for line in lines):
        return text

    cited = {}
    for i, line in enumerate(lines):
        groups = _CITATION_GROUP.findall(line)
        if groups:
            cited[i] = tuple(sorted(set(
                _RECEIPT_NO.findall(" ".join(groups)))))
    if not cited or len(set(cited.values())) != 1:
        return text
    receipts = next(iter(cited.values()))
    if not receipts:
        return text

    for i in cited:
        stripped = _CITATION_GROUP.sub("", lines[i]).rstrip()
        # 인용만으로 이루어진 줄을 비우면 근거가 아니라 빈 줄이 남는다.
        if not stripped:
            return text
        lines[i] = stripped

    # 표의 근거 열처럼 답변 안에 이미 같은 접수번호가 남아 있으면 끝줄은
    # 같은 말을 세 번째로 적는 셈이다.
    remaining = set(_RECEIPT_NO.findall("\n".join(lines)))
    if not set(receipts) <= remaining:
        lines.append("")
        lines.append("근거: " + ", ".join("접수번호 " + r for r in receipts))
    return "\n".join(lines)


_CALCULATION_OPERATORS = {
    "difference", "absolute_difference", "percent_change",
    "discrete_from_cumulative", "concept_ratio", "sum",
}


def _has_arithmetic_claim(payload) -> bool:
    return any(
        claim.operator in _CALCULATION_OPERATORS
        and len(claim.derived_from or ()) >= 2
        for claim in getattr(payload, "claims", ())
    )


def _replace_public_money(text: str, payload, *, question: str | None = None) -> str:
    """Replace source-unit money spellings with exact Korean won groups.

    이슈 #170(M35-c) — 질문이 명시적으로 그 단위를 요청했을 때(``agent.
    display_units_v1``)는 예외다.  이 함수는 보통 HCX가 원문 그대로의
    출처 단위("14,289,390백만원")를 그대로 써 버렸을 때 읽기 좋은 조·억·
    만원 표기로 되돌리는 미화 단계다.  하지만 요청 단위가 마침 그 fact의
    ``raw_unit``과 같으면("백만원 단위로 얼마인가?"), 요청을 정확히 지킨
    표기를 이 단계가 되돌려 버려 요청 단위 표시가 조용히 사라졌다.  그
    raw_unit이 이번 질문이 요청한 표시 단위 중 하나면 치환 대상에서 뺀다.
    """

    directive = parse_display_units_directive(question)
    requested_units = set(directive.units) if directive is not None else set()
    targets: dict[str, set[str]] = {}
    rows = []
    for claim in getattr(payload, "claims", ()):
        rows.append(claim)
        rows.extend(getattr(claim, "ranking", ()) or ())
    for claim in rows:
        source_value = getattr(claim, "value_text", None)
        source_unit = getattr(claim, "raw_unit", None)
        display = claim_won_display(claim)
        if not source_value or not source_unit or not display:
            continue
        if source_unit in requested_units:
            continue
        source = f"{source_value}{source_unit}"
        if source != display:
            targets.setdefault(source, set()).add(display)
            stripped_value = source_value.strip()
            if stripped_value.startswith("(") and stripped_value.endswith(")"):
                targets.setdefault(
                    f"{stripped_value[1:-1]}{source_unit}", set()).add(display)

    public = text
    replaced = False
    # A source spelling with conflicting typed meanings is left untouched;
    # guessing by surrounding prose would weaken the deterministic boundary.
    for source, displays in sorted(
            targets.items(), key=lambda item: len(item[0]), reverse=True):
        if len(displays) != 1:
            continue
        display = next(iter(displays))
        variants = {source, source.replace(",", "")}
        for variant in sorted(variants, key=len, reverse=True):
            pattern = re.compile(
                r"(?<![0-9,])"
                + re.escape(variant)
                + r"(?:\s*\(약\s*[0-9][0-9,.]*\s*(?:조|억|만)?원\))?"
                + r"(?![0-9,])"
            )
            public, count = pattern.subn(display, public)
            replaced = replaced or bool(count)
    if replaced:
        # Older HCX wording occasionally put the rounded helper in a separate
        # sentence.  The exact mixed-unit amount now makes that sentence both
        # redundant and less precise.
        public = re.sub(
            r"\s*쉬운 표기로는\s*약\s*[0-9][0-9,.]*\s*"
            r"(?:조|억|만)?원입니다\.?",
            "",
            public,
        )
    # Narrative/correction text can contain verified monetary surfaces inside
    # a typed text claim rather than a scalar claim.  Convert only an explicit
    # number immediately followed by an unambiguous won unit.
    def replace_surface(match: re.Match[str]) -> str:
        # ``약`` marks source-level approximation.  Unit expansion must not
        # make that rounded number look newly precise.
        if re.search(r"약\s*$", match.string[:match.start()]):
            return match.group(0)
        # 이슈 #170(M35-c) — 위 targets 예외와 같은 이유: 이 자리의 단위가
        # 마침 질문이 명시적으로 요청한 표시 단위 중 하나면, 그 요청을 정확히
        # 지킨 표기를 다시 조·억·만원 장문형으로 되돌리지 않는다.
        if "".join(match.group("unit").split()) in requested_units:
            return match.group(0)
        return (
            format_source_money_exact(
                match.group("value"), match.group("unit"))
            or match.group(0)
        )

    return _PUBLIC_MONEY_SURFACE.sub(replace_surface, public)


def _ensure_citation_survives(text: str, payload) -> str:
    """확정 답변 본문에 접수번호가 하나도 남지 않았으면 끝에 붙인다.

    이슈 #84 SG-009 — HCX는 이 값의 유일한 인용을 실제 계산이 아닌
    "계산 근거:" 줄에 적었다(사채 권면총액을 확인하는 접수번호 세 개를
    나열했을 뿐 덧셈·뺄셈은 없었다). 바로 위 계산-근거 환각 방지 가드는
    실제 산술 claim이 없는 payload에서 그 줄을 통째로 지우므로, 본문에는
    접수번호가 하나도 남지 않게 됐다 — claim 자체에는 검증된 인용이 있는데도.

    어느 경로로 인용이 사라지든 이 함수가 마지막 안전망이다. 본문에 이미
    접수번호가 하나라도 있으면 손대지 않는다 — "본문이 이미 말하면
    되풀이하지 않는다"는 이슈 #93 원칙은 그대로 존중한다. 자리(좌표)를 아는
    인용이 있어도 없어도 상관없이, 남은 것이 하나도 없을 때만 접수번호만으로
    한 줄을 보탠다.
    """

    if payload is None or _RECEIPT_NO.search(text or ""):
        return text
    receipts = sorted({
        ct.rcept_no
        for c in getattr(payload, "claims", ()) or ()
        for ct in getattr(c, "citations", ()) or ()
        if getattr(ct, "rcept_no", None)
    })
    if not receipts:
        return text
    body = (text or "").rstrip()
    footer = "근거: " + ", ".join("접수번호 " + r for r in receipts)
    return f"{body}\n\n{footer}" if body else footer


def _explain_first_financial_basis(text: str, question: str | None, payload=None) -> str:
    """Explain a basis once, without changing account names or source rows."""
    if re.search(r"숫자만|값만|설명\s*없이|설명은\s*빼", question or ""):
        return text
    if not any(
            getattr(claim, "canonical_unit", None) == "원"
            and re.search(r"매출|영업수익|이익|자산|부채|현금흐름", getattr(claim, "label", ""))
            for claim in getattr(payload, "claims", ())):
        return text
    lines = text.splitlines()
    explanations = {"연결": "자회사 등을 포함해 계산", "별도": "회사 자체 기준"}
    seen: set[str] = set()
    for index, line in enumerate(lines):
        if line.startswith(("근거", "출처", "계산 근거", "|")):
            continue
        if len(seen) == len(explanations):
            break
        for basis, explanation in explanations.items():
            if basis in seen:
                continue
            pattern = (rf"{basis}\s*기준(?![가-힣])"
                       r"(?=\s*(?:매출|영업수익|영업이익|당기순이익|순이익|수익|자산|부채|현금흐름))")
            match = re.search(pattern, line)
            if match is None:
                continue
            seen.add(basis)
            # Existing explanatory parentheses are already sufficient.
            if re.match(r"\s*[(（]", line[match.end():]):
                continue
            line = line[:match.end()] + f"({explanation})" + line[match.end():]
        lines[index] = line
    return "\n".join(lines)


def _readable_record_prose(text: str, question: str | None) -> str:
    """Clarify exact record values without rewriting source tables or units."""
    query = question or ""
    directive = parse_display_units_directive(question)
    readable_money = (directive is None
                      and not re.search(r"원\s*단위|숫자만|값만", query)
                      and (bool(re.search(r"쉽게|얼마나\s*벌", query))
                           or ("해지" in query and "계약금액" in query)))
    lines = text.splitlines()
    in_code = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
        if in_code or line.lstrip().startswith(("|", "정확한 금액:", "근거:", "출처:")):
            continue
        if readable_money:
            line = re.sub(
                r"(?<![\d,.+\-])(?P<value>\d{1,3}(?:,\d{3}){2,})원",
                lambda m: format_won_exact(m.group('value').replace(',', '')) or m.group(0),
                line)
        if "주" in query and re.search(r"변동|증감", query):
            line = re.sub(r"증감은 -([\d,]+)주입니다\.",
                          r"증감은 \1주 감소입니다.", line)
        if "사채" in query and "취득" in query and re.search(r"사채\s*취득", line):
            line = re.sub(
                r"(사유로\s*[\d,조억만.\s]+원)\s*을\s*취득(?:했|하였)습니다"
                r"(?=\s*(?:\((?:접수번호\s+|근거\s*:?\s*)\d{14}\))?\s*\.)",
                r"\1을 들여 사채를 취득했습니다", line)
        lines[index] = line
    return "\n".join(lines)


def _amount_only_facility_prose(text: str, payload, question: str | None) -> str:
    """Remove only generic nontermination prose from a typed facility scalar.

    This runs after model wording: filtering TemplateComposer alone does not
    constrain an HCX answer assembled directly from the original payload.
    Actual ending observations, extra claims and limitations remain untouched.
    """
    query = question or ""
    if (payload is None or not re.search(r"시설\s*투자", query)
            or not re.search(r"투자\s*금액", query)
            or re.search(r"상태|종료|해지|완료|진행|유효|기간|변경|정정|이력", query)
            or getattr(payload, "limitations", ())
            or getattr(payload, "premise_verdicts", ())):
        return text
    claims = list(getattr(payload, "claims", ()))
    amounts = [c for c in claims if re.fullmatch(r"투자\s*금액|\d{14}\s+금액", c.label or "")
               and re.fullmatch(r"\d[\d,]*(?:\.\d+)?", c.value_text or "")]
    states = [c for c in claims if c.state == "no_termination_observed"]
    # The event-list route carries collection metadata even for one requested
    # record.  Admit only an explicitly single event with zero terminations.
    metadata = [c for c in claims if
                (c.label == "사건 건수" and c.value_text == "1")
                or (c.label == "해지 공시 관측 건수" and c.value_text == "0")
                or c.label == "공시 주체"]
    if (len(amounts) != 1 or len(states) != 1
            or len(claims) != 2 + len(metadata)
            or any(c.operator or c.derived_from for c in claims)):
        return text
    if re.match(r"\d{14}", amounts[0].label):
        receipt = amounts[0].label[:14]
        if (states[0].label != f"사건 {receipt}"
                or not any(c.rcept_no == receipt for c in amounts[0].citations)):
            return text
    # Match complete, closed boilerplate clauses, never arbitrary sentences
    # containing 종료.  In particular citations and source-specific caveats
    # cannot be consumed by these patterns.
    receipts = ({c.rcept_no for c in amounts[0].citations if c.rcept_no}
                & {c.rcept_no for c in states[0].citations if c.rcept_no})
    citation = (rf"(?:\s*\(접수번호\s+{re.escape(next(iter(receipts)))}\))?"
                if len(receipts) == 1 else "")
    status = (r"(?:이\s*공시는\s*)?(?:현재\s*)?해지[·ㆍ/]종료\s*공시가\s*"
              r"(?:미확인된|확인되지\s*않은)\s*상태이며" + citation + r",\s*")
    qualified = (status + r"현재\s*진행\s*여부와\s*법적\s*유효성이\s*"
                 r"확정되지\s*않았습니다" + citation + r"\.")

    def retain_citations(match: re.Match) -> str:
        # HCX attaches citations inside a clause, before the comma/period.
        # Preserve only the exact same-event groups that the regex admitted;
        # the normal later citation collector can move/deduplicate them.
        groups = dict.fromkeys(re.findall(r"\(접수번호\s+\d{14}\)", match.group(0)))
        return " ".join(groups) + (" " if groups else "")

    text = re.sub(qualified, retain_citations, text)
    # Keep the following scalar/citation instead of deleting its sentence.
    text = re.sub(status + r"(?=투자\s*금액은)", retain_citations, text)
    return text


def render_public_answer(
        text: str, payload=None, *, question: str | None = None) -> str:
    """최종 사용자 답에서만 내부 실행 좌표를 제거한다.

    ``retrieved_context``와 typed citation은 감사·디버깅을 위해 원좌표를 계속
    보존한다. 이 함수는 모든 모델·템플릿 검증이 끝난 뒤 호출되므로 matrix 셀
    결속이나 숫자·근거 검사를 약화하지 않는 마지막 표시 방어선이다.
    """

    public = (_replace_public_money(text or "", payload, question=question)
              if payload is not None else (text or ""))
    public = _INTERNAL_TABLE_LOCATOR.sub("", public)
    public = _INTERNAL_MATRIX_LOCATOR.sub("", public)
    public = _INTERNAL_EVENT_STATE.sub(
        lambda match: _PUBLIC_EVENT_STATE[match.group(0)], public)
    public = _PUBLIC_RECEIPT_MARKER.sub("접수번호 ", public)
    # Verified event prose can already carry the same receipt that the typed
    # claim citation appends.  Remove adjacent identical copies only; distinct
    # receipts and citations attached to separate statements stay intact.
    public = _ADJACENT_DUPLICATE_CITATION.sub(
        lambda match: match.group("citation"), public)
    public = _INLINE_DUPLICATE_CITATION.sub(
        lambda match: match.group("prefix"), public)
    # A wording model can hallucinate a calculation-basis line for a scalar
    # selection even when the prompt contains no operand marker.  At the final
    # typed boundary, remove that entire line unless the payload contains an
    # actual arithmetic result.  This affects presentation only; the verified
    # scalar and citation remain untouched.
    if payload is not None and not _has_arithmetic_claim(payload):
        public = "\n".join(
            line for line in public.splitlines()
            if (not line.startswith(("계산 근거:", "계산 접수번호 "))
                or any(state in line for state in _PUBLIC_EVENT_STATE.values()))
        )
        # Inline citations are not computation claims: keep their evidence,
        # but do not label a direct record lookup as a calculation.
        public = re.sub(r"계산\s+(?=근거\s*[:：]|접수번호\s+\d{14})", "", public)
    # Only after the line above has settled whether a calculation-basis line
    # survives: receipts move onto it, so demoting before that could strip a
    # conclusion's citation and then delete the line that was holding it.
    public = _demote_citations_to_basis(public)
    public = _drop_generic_refusal_after_a_reason(public)
    public = _readable_record_prose(public, question)
    public = _lead_with_a_graspable_amount(public)
    public = _amount_only_facility_prose(public, payload, question)
    public = _qualify_absent_termination(public, payload)
    public = _speak_redactions(public)
    public = _drop_self_named_citations(public)
    public = _hoist_run_citations(public)
    public = _collect_uniform_citations(public)
    public = _ensure_citation_survives(public, payload)
    # These are canonical execution terms, not language a general reader
    # should have to decode.  The underlying receipt sequence and verified
    # state are unchanged; only the final public labels are simplified.
    public = public.replace("논리적 최신본", "마지막으로 확인된 정정본")
    # Collapse only this repeated display label; keep the receipt and all
    # chronology/limitation text exactly as produced by the verified payload.
    public = re.sub(
        r"(?m)^([ \t]*)마지막으로 확인된 정정본:[ \t]*마지막으로 확인된 정정본은[ \t]+",
        r"\1마지막으로 확인된 정정본: ", public)
    public = public.replace("논리적 정정 순서", "공시 정정 순서")
    public = public.replace("문서 정정 계보", "공시 정정 이력")
    public = public.replace("접근번호", "접수번호")
    public = _explain_first_financial_basis(public, question, payload)
    public = re.sub(r"[ \t]+([,;)])", r"\1", public)
    public = re.sub(r";\s*;", "; ", public)
    public = re.sub(r"(?<=[가-힣])\.\.(?!\.)", ".", public)
    public = re.sub(r"\[\s*\]", "", public)
    public = re.sub(r"[ \t]{2,}", " ", public)
    return "\n".join(line.rstrip() for line in public.splitlines()).strip()


def render_think_trace(payload) -> str:
    lines = []
    for e in payload.trace:
        tag = _STAGE_KO.get(e.stage, e.stage)
        lines.append(f"[{tag}] {e.summary}")
    # 마지막에 확정 요약
    lines.append(f"[결과] final_status={payload.final_status} claims={len(payload.claims)} "
                 f"limitations={[l.code for l in payload.limitations]} premise={[v.verdict for v in payload.premise_verdicts]}")
    return "\n".join(lines)


def _retrieved_claim_excerpt(claim, citation) -> str:
    """Reuse an already verified narrative block, never unsourced claim prose.

    Narrative tools bind one prompt-safe source block to one section citation.
    Their short citation preview is not the whole evidence used by the answer.
    Recover that existing block at the API boundary only, without enlarging the
    composer prompt or fetching any additional (potentially unsafe) source.
    """
    text = getattr(claim, "text", None) or ""
    if (len(claim.citations) == 1
            and citation.section_id
            and citation.verification == "source_roundtrip"
            and text.startswith("<<<UNTRUSTED_DART_DISCLOSURE_DATA_BEGIN>>>")
            and text.rstrip().endswith("<<<UNTRUSTED_DART_DISCLOSURE_DATA_END>>>")):
        return text
    return citation.excerpt_prompt_safe or ""


def render_retrieved_context(payload) -> str:
    """Return all distinct evidence already attached to the selected claims.

    A display preview limit must not drop a later company/date or slice off a
    table row and its safe-data boundary. Selection and block-size budgets are
    enforced upstream; this renderer reads neither full documents nor raw data.
    """
    seen = set()
    items = []
    for c in payload.claims:
        for ct in c.citations:
            body = _retrieved_claim_excerpt(c, ct)
            # One section may supply several independently selected blocks.
            # Deduplicate repeated evidence, not the whole section identity.
            key = (ct.doc_id, ct.evidence_id or ct.section_id or ct.locator, body)
            if key in seen:
                continue
            seen.add(key)
            head = f"[{len(items)+1}] 접수번호 {ct.rcept_no or '-'} · {ct.doc_id}"
            if ct.locator and ct.locator != "document":
                head += f" · {ct.locator}"
            head += f" · {ct.verification}"
            items.append(head + ("\n" + body if body else ""))
    return "\n\n".join(items)


def build_answer_response(question_id: str, question: str, payload, answer_text: str) -> dict:
    return {
        "question_id": question_id,
        "question": question,
        "retrieved_context": render_retrieved_context(payload),
        "think_trace": render_think_trace(payload),
        "answer": render_public_answer(answer_text, payload, question=question),
    }
