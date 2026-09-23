"""Public-safe rendering for limitations that must reach the final answer.

Tool limitation ``detail`` values are diagnostic data.  In particular,
correction lineage details can contain internal event keys and provisional
identity markers, so composers must render the stable public message below
instead of copying those details to users.
"""
from __future__ import annotations

import re

from collections.abc import Iterable
from typing import Protocol


class _LimitationLike(Protocol):
    code: str


_SAFE_MESSAGES = {
    "explicit_disclosure_cutoff": (
        "질문에 지정한 날짜까지 공개된 공시만 조회했습니다. 그 이후에 제출된 "
        "보고서의 숫자로 대신하지 않았습니다."
    ),
    "not_found_interest_expense": (
        "요청한 회사·기간·재무제표 기준에서 손익계산서의 이자비용은 확인하지 "
        "못했습니다. 현금흐름표의 이자 지급액이나 0원으로 대신하지 않았습니다."
    ),
    "not_found_interest_paid": (
        "요청한 회사·기간·재무제표 기준에서 현금흐름표의 이자 지급액은 확인하지 "
        "못했습니다. 손익계산서의 이자비용이나 0원으로 대신하지 않았습니다."
    ),
    "non_additive_eps": (
        "반기나 연간 누적 기본주당순이익에서 앞선 분기의 값을 빼서 "
        "한 분기의 기본주당순이익을 구할 수는 없습니다. 주당순이익은 "
        "이익을 그 기간의 평균 주식 수로 나눈 값인데, 기간마다 이 평균이 "
        "달라질 수 있기 때문입니다. 따라서 요청한 뺄셈은 하지 않았습니다."
    ),
    "fx_conversion_not_supported": (
        "이 시스템은 공시의 환율과 기준일을 환산 근거로 확정하는 기능을 "
        "지원하지 않아 달러 금액은 계산하지 않았습니다. 임의 환율은 쓰지 "
        "않았으며, 공시에 환율이 없다고 단정한 것은 아닙니다."
    ),
    "ambiguous_event_identity": (
        "후속 공시는 확인되지만 동일 상대방의 원공시가 복수여서 "
        "어느 원계약에 해당하는지는 공시만으로 특정할 수 없습니다."
    ),
    "ambiguous_event_origin": (
        "후속 공시는 확인되지만 동일 상대방의 원공시가 복수여서 "
        "어느 원계약에 해당하는지는 공시만으로 특정할 수 없습니다."
    ),
    "counterparty_not_reported_may_hide_match": (
        "계약상대가 공개되지 않은 관련 후보 공시가 있어, 이 후보가 요청한 "
        "상대방의 계약인지 아닌지는 확인할 수 없습니다."
    ),
    "intraday_order_unavailable": (
        "공시의 정정 순서는 확인되지만 장중 접수시각이 없어 실제 제출 시각과 "
        "분·초 단위 선후는 확인할 수 없습니다."
    ),
    "correction_change_details_unavailable": (
        "정정 전후의 구체적인 변경 항목 본문은 원문 근거로 확인할 수 없어, "
        "확인된 정정 대상·사유와 문서 계보·유효기간만 제시합니다."
    ),
    "ambiguous_correction_sequence": (
        "정정공시는 확인했지만 공시 간 법적 정정 순서를 모두 확정할 수 없어, "
        "아래 내용은 확인된 공시 기준이며 완전한 공시 계보를 뜻하지 않습니다."
    ),
    "correction_identity_provisional": (
        "일부 정정공시는 어느 원공시에 이어지는지 확정할 수 없습니다. 공시일만으로 "
        "같은 날의 여러 후보 중 하나를 임의로 고르지 않았으므로 공시별 이력이 "
        "일부 빠질 수 있습니다."
    ),
    "source_scope_prevents_complete_lineage": (
        "제공된 자료보다 앞선 원공시가 없어 해당 공시의 "
        "최초 기재 내용과 그 이전 변경 이력은 확인할 수 없습니다."
    ),
    "source_excerpt_boundary_unavailable": (
        "검증된 원문 값은 있으나 공개 길이 안에 완결된 문장이나 항목 경계가 "
        "없어, 내용을 중간에서 잘라 제시하지 않았습니다."
    ),
    "comparison_explanation_excerpt_unavailable": (
        "두 금액의 차이는 계산했지만 산정 근거 원문을 공개 길이 안에서 "
        "완결된 문장으로 제시할 수 없어, 원인을 임의로 요약하지 않았습니다."
    ),
    "future_forecast": (
        "미래 실적 예측은 제공 공시로 답할 수 없어 생성하지 않았습니다."
    ),
    "causal_inference_beyond_scope": (
        "제공된 자료에는 주가 시계열이 없고 회사 정보의 시가총액도 한 시점의 "
        "스냅샷이므로, 공시 실적과 주가 사이의 인과관계는 확인할 수 없습니다."
    ),
    "corp_not_in_universe": (
        "제공된 70개 기업의 DART 공시자료(회계기간 2023년 1월~2026년 "
        "1분기, 접수일은 2026년 6월 19일까지)에 일부 기업이 포함돼 있지 "
        "않아, 해당 기업의 값과 회사 간 순위·차이는 계산하지 않았습니다."
    ),
    "unsupported_operator": (
        # States only what was not done.  The previous wording also promised
        # that «확인된 개별 값은 제공하지만» — true when a policy fallback
        # supplies the operands, false when none is eligible, and the notice
        # is rendered from the code alone with no view of the claims.  Where
        # values are supplied they are already in the answer above.
        # 이슈 #124 — 「합계」는 이제 지원 연산이라(2사·2기간 sum) 미지원의
        # 대표 예시로 두면 답과 모순된 안내가 된다. 「평균」은 여전히 v0.4
        # 밖이므로 그 자리를 대신한다.
        "요청한 평균·비율은 공시에서 직접 확인되는 값이 아니므로 이번 답변에서는 "
        "계산하지 않았습니다."
    ),
    "unsupported_semantic_target": (
        "요청하신 대상이나 항목을 이번 공시자료에서 근거와 함께 확인할 수 없어, "
        "비슷하지만 다른 대상이나 항목의 내용을 대신 제시하지 않았습니다."
    ),
    # A sealed fallback already produced the cited claim below, so — unlike
    # the code above — this one must not say the alternative was withheld.
    "unsupported_semantic_target_substituted": (
        "요청하신 항목은 공시에서 확인할 수 없어, "
        "질문에서 명시한 대체 항목만 제시했습니다."
    ),
    "source_scope_raw_absent": (
        "원공시가 제공된 코퍼스보다 앞서 제출되어 있어 최초 기재 원문은 "
        "확인할 수 없고, 코퍼스에 포함된 이후 정정 이력만 제시합니다."
    ),
    "incomparable_aggregation_scope": (
        "투자계획의 프로젝트별 계획금액과 현금흐름표의 기간 실제 집계액은 "
        "의미와 집계 범위가 달라 서로 차감하지 않았습니다."
    ),
    # 이슈 #64 — 연간(12개월)처럼 긴 기간과 분기 단독(3개월)처럼 짧은 기간은
    # 같은 계정이라도 길이가 달라, 그 차이를 시간 증감률(분기 성장률 등)로
    # 표현하면 실제로는 없는 성장을 만든다. 두 값은 각각 그대로 답한다.
    "period_length_mismatch": (
        "연간 값과 분기 단독 값은 기간 길이가 달라, "
        "두 값의 차이를 분기 증감률로 해석할 수 없습니다."
    ),
    # 이슈 RPC-002 — 연결과 별도는 같은 회사·계정·기간이라도 종속회사 포함
    # 여부가 다른 서로 다른 집계 범위다. 두 값의 차이를 「전년 대비 증감률」
    # 처럼 시간이 만든 변화로 부르면, 실제로는 집계 범위가 다른 것이 만든
    # 차이를 시간이 만든 것처럼 보이게 한다.
    "scope_mismatch": (
        "연결과 별도는 종속회사 포함 범위가 달라, "
        "두 값의 차이를 시간에 따른 증감으로 해석할 수 없습니다."
    ),
    "partial_unread": (
        "같은 기간의 일부 공시 내용을 확인할 수 없어 최신 정정값까지 반영됐다고 "
        "단정할 수 없습니다."
    ),
    "latest_annual_extract_unsupported": (
        "최신 연차 공시의 관련 내용을 읽지 못해 요청한 값을 확인할 수 없습니다. "
        "이전 공시나 비슷한 항목의 값으로 대신하지 않았습니다."
    ),
    "not_found_financial_account": (
        "요청하신 회사·기간·재무제표 기준에서 해당 계정의 값을 찾지 못했습니다. "
        "다른 계정이나 0원으로 대신하지 않았으며, 자료 밖에서도 이 계정이 없다는 뜻은 아닙니다."
    ),
    "not_found_financial_sector_statement_semantics": (
        "요청하신 금융·보험사 공시에서는 일반 매출액으로 확인할 단일 항목을 찾지 못해, "
        "다른 성격의 계정을 매출액으로 대신하지 않았습니다."
    ),
    "not_found_raw_total_cell_empty": (
        "공시에 연결 당기순이익 합계가 기재돼 있지 않아, 하위 항목을 임의로 더해 "
        "당기순이익을 만들지 않았습니다."
    ),
    "not_found_combined_concept_line": (
        "유형자산과 투자부동산 취득이 한 결합 계정으로만 공시되어, 순수 유형자산 "
        "취득액을 분리하거나 배분하지 않았습니다."
    ),
    "personal_data_omitted": (
        "개인 식별정보와 개인 연락처는 제외하고 공개된 공시 정보만 제시했습니다."
    ),
    "requested_field_redacted": (
        "요청한 항목 중 일부는 제공 자료에서 가려졌으며, 확인 가능한 "
        "공개 항목만 제시했습니다."
    ),
    "ambiguous_field": (
        "요청한 항목 중 일부는 공시 안의 후보가 복수여서 하나의 값으로 특정하지 "
        "않았으며, 확인 가능한 항목만 제시했습니다."
    ),
    "holding_slot_unavailable": (
        "요청한 지분공시 항목 일부는 해당 보고서에서 근거와 함께 확인할 수 없었습니다."
    ),
    "holding_filer_occupation_redacted": (
        "보고자의 직업(사업내용)은 제공 자료에서 가려져 있어 확인할 수 없습니다."
    ),
    "holding_filer_nationality_redacted": (
        "보고자의 국적은 제공 자료에서 가려져 있어 확인할 수 없습니다."
    ),
    "holding_party_rows_unavailable": (
        "특별관계자의 이름·관계·보유량을 같은 원문 행으로 안전하게 결합할 수 없어 목록은 제시하지 않았습니다."
    ),
    "holding_lineage_ambiguous": (
        "지분공시의 정정 대상이 하나로 확정되지 않아 임의의 원보고서나 최신본을 선택하지 않았습니다."
    ),
    "holding_lineage_root_missing": (
        "원보고서가 제공된 자료보다 앞서 제출되어 확인되는 정정 문서 내용만 제시합니다."
    ),
    "holding_subject_name_partially_masked": (
        "보고자 성명은 개인정보 보호를 위해 일부만 표시했습니다."
    ),
    "abusive_input": (
        "부적절한 표현이 포함된 요청에는 답변하지 않습니다. 기업 공시에 관한 "
        "질문(예: 회사명·연도·항목)을 보내 주세요."
    ),
    "off_topic_request": (
        "이 서비스는 제공된 DART 공시 자료에 관한 질문에만 답합니다. "
        "회사명·기간·항목을 포함해 질문해 주세요."
    ),
    # 서버 경계 정규화(server/stage1.py safe_question_id)를 뚫고 들어온
    # question_id 형식 오류에 대한 안전망 (#110). server/runtime.py의
    # MSG_INVALID_QUESTION_ID 가 이 문장을 그대로 쓴다 — 원문은 여기 하나뿐이다.
    "invalid_question_id": (
        "question_id 형식이 올바르지 않습니다. 글자로 시작하고 영숫자와 "
        "`_`·`.`·`:`·`-`만 쓸 수 있습니다. question_id 를 고쳐 다시 요청해 주세요."
    ),
    # #165 — HCX-007 provider 전송 실패(429/5xx/timeout)를 "질문을 해석하지
    # 못했다"는 질문 탓 문구(MSG_UNRESOLVED)와 구분한다. 두 코드
    # (server/stage1.py safe_exception_diagnostics: upstream_rate_limited=429,
    # upstream_unavailable=5xx/timeout)는 원인이 다르지만 사용자에게 보일
    # 말은 같다 — 둘 다 "지금 다시 해도 될 수 있다"는 재시도 안내일 뿐,
    # 질문을 고치라는 안내가 아니다. server/runtime.py의
    # MSG_UPSTREAM_RATE_LIMITED/MSG_UPSTREAM_UNAVAILABLE 가 이 문장을 그대로
    # 쓴다 — 원문은 여기 하나뿐이다(ambiguous_event_identity/
    # ambiguous_event_origin과 같은 "코드는 다르지만 문구는 같다" 선례).
    "upstream_rate_limited": (
        "일시적인 처리 지연으로 이번 요청에 답변을 만들지 못했습니다. 잠시 후 같은 "
        "질문을 다시 요청해 주세요. 제공된 DART 공시 자료 범위 안에서 확인 가능한 "
        "내용만 답변드립니다."
    ),
    "upstream_unavailable": (
        "일시적인 처리 지연으로 이번 요청에 답변을 만들지 못했습니다. 잠시 후 같은 "
        "질문을 다시 요청해 주세요. 제공된 DART 공시 자료 범위 안에서 확인 가능한 "
        "내용만 답변드립니다."
    ),
}


def safe_limitation_message(code: str) -> str | None:
    """Return a stable public message for a protected limitation code."""
    return _SAFE_MESSAGES.get((code or "").split(":", 1)[0])


_AMBIGUOUS_FIELD_DETAIL = re.compile(
    r"^(?:[0-9]{14}\s+)?(?P<field>[0-9A-Za-z가-힣·ㆍ_()/%\s-]{1,80})"
    r":\s*(?:ambiguous|후보\s*복수)\s*$",
    re.IGNORECASE,
)


def ambiguous_field_names(limitations: Iterable[_LimitationLike]) -> list[str]:
    """Return only public-safe field names from typed ambiguity details.

    Event lookup details use the closed form ``receipt field: ambiguous``.
    The receipt and diagnostic token stay private; naming the requested field
    is nevertheless important because "not disclosed" and "several values"
    are materially different states (issue #196).
    """

    fields: list[str] = []
    for limitation in limitations:
        if (getattr(limitation, "code", "") or "").split(":", 1)[0] != "ambiguous_field":
            continue
        match = _AMBIGUOUS_FIELD_DETAIL.fullmatch(
            str(getattr(limitation, "detail", "") or "").strip())
        if match is None:
            continue
        field = " ".join(match.group("field").split())
        if field and field not in fields:
            fields.append(field)
    return fields


# 이 두 limitation은 payload 안전 계약으로는 항상 보존한다. 다만 단순 값·차이
# 조회의 공개 답변에서까지 「하지 않은 시간 계산」을 설명할 필요는 없다(#192).
# 질문이 실제로 증감률 또는 그 해석 가능성을 물을 때만 사용자 문장으로 만든다.
_TIME_CHANGE_INTERPRETATION_QUESTION = re.compile(
    r"(?:증감률|성장률|변화율|변동률|%p?|퍼센트|"
    r"시간(?:에\s*따른)?\s*(?:증감|변화)|"
    r"(?:증감|성장|변화|변동)(?:으로|이라고)?\s*(?:해석|표현)|"
    r"(?:해석|표현)해도|불러도|봐도\s*되|볼\s*수\s*있)",
    re.IGNORECASE,
)


def _show_time_change_limitation(code: str, question: str | None) -> bool:
    base = (code or "").split(":", 1)[0]
    if base not in {"scope_mismatch", "period_length_mismatch"}:
        return True
    # 호출자가 질문을 전달하지 않는 기존 검증·안전 API는 보수적으로 종전 계약을
    # 유지한다. 실제 /answer 경로는 질문을 전달해 관련성까지 판정한다.
    return question is None or bool(
        _TIME_CHANGE_INTERPRETATION_QUESTION.search(question))


#: 코퍼스보다 앞선 원공시가 없다는 같은 사실을 각자 다른 문장으로 말하는
#: 두 코드. 정정 이력 답에서는 둘 다 typed 로 함께 붙을 수 있는데, 그러면
#: 같은 경계를 두 번 말하게 된다(EDGE-028, #86-15). 더 구체적인 뒤 문장
#: (``source_scope_raw_absent``)만 남긴다.
_LINEAGE_BOUNDARY_DUPLICATE = "source_scope_prevents_complete_lineage"
_LINEAGE_BOUNDARY_KEPT = "source_scope_raw_absent"

#: 집계 범위가 달라 **차감만** 하지 않은 답에는 두 값이 다 실려 있다.  거기에
#: 「요청하신 항목은 공시에서 확인할 수 없어」를 덧붙이면 같은 답이 찾았다고도
#: 못 찾았다고도 말한다 — 둘 중 하나는 거짓이다.  더 구체적인 쪽만 남긴다.
_SUBSTITUTION_CONTRADICTED_BY = {
    "fx_conversion_not_supported",
    "incomparable_aggregation_scope",
    "period_length_mismatch",
}
_SUBSTITUTION_NOTE = "unsupported_semantic_target_substituted"


def render_safe_limitations(
        limitations: Iterable[_LimitationLike], *, question: str | None = None,
        ) -> list[str]:
    """Render protected limitations once each, preserving their first order."""
    rows = list(limitations)
    codes = {(l.code or "").split(":", 1)[0] for l in rows}
    drop_duplicate_boundary = (
        _LINEAGE_BOUNDARY_DUPLICATE in codes and _LINEAGE_BOUNDARY_KEPT in codes)
    drop_substitution_note = bool(codes & _SUBSTITUTION_CONTRADICTED_BY)
    messages: list[str] = []
    for limitation in rows:
        code = (limitation.code or "").split(":", 1)[0]
        if not _show_time_change_limitation(code, question):
            continue
        if drop_duplicate_boundary and code == _LINEAGE_BOUNDARY_DUPLICATE:
            continue
        if drop_substitution_note and code == _SUBSTITUTION_NOTE:
            continue
        if code == "ambiguous_field":
            fields = ambiguous_field_names([limitation])
            message = (
                f"공시에는 ‘{fields[0]}’ 후보가 여러 개 있어 하나로 특정하지 "
                "않았으며, 확인 가능한 항목만 제시했습니다."
                if fields else safe_limitation_message(limitation.code)
            )
        else:
            message = safe_limitation_message(limitation.code)
        if message and message not in messages:
            messages.append(message)
    return messages


#: 절이 아니라 이름표로 시작하는 한계 — 「원공시 후보 접수번호: …」처럼
#: 앞에 접속부사를 붙일 자리가 없다.
_LABELLED_NOTE = re.compile(r"^[^.:：\n]{2,20}[:：]\s")

#: 「주요사업내용」 slot 부재 한계는 사업부문·주요 제품 내용을 답이 이미
#: 보여줬으면 같은 사업 내용을 다시 「본문에 나오지 않습니다」로 부정하는
#: 셈이다(K-001·K-025 등, #86-19). ``slot_not_confirmed`` detail 은
#: 「확인한 공시 본문에는 다음 항목이 나오지 않습니다: A, B. 추측하지
#: 않고…」 꼴이라, 그 목록이 정확히 이 slot 하나뿐일 때만 판단한다 — 다른
#: slot 이 함께 빠졌으면 그 목록은 그대로 필요한 정보다. 답이 claim
#: label(``_public_notes``)이 아니라 완성된 텍스트로만 오는 narrative
#: matrix 경로(``append_public_qualifications`` 직접 호출)에서도 같은
#: 판단을 하려면 텍스트 자체에서 커버리지를 본다.
_BUSINESS_CONTENT_SLOT = "주요사업내용"
_BUSINESS_CONTENT_COVERAGE_MARKERS = ("사업부문", "주요 제품")
_SLOT_NOT_CONFIRMED_ITEMS = re.compile(
    r"다음 항목이 나오지 않습니다: (?P<items>.+?)\.\s*추측하지 않고")


def is_business_content_only_gap(message: str) -> bool:
    """Is a ``slot_not_confirmed`` message's sole missing item the

    redundant 「주요사업내용」 slot (already covered by a 사업부문/주요
    제품 breakdown elsewhere in the answer)?
    """

    match = _SLOT_NOT_CONFIRMED_ITEMS.search(message or "")
    if match is None:
        return False
    items = [item.strip() for item in match.group("items").split(",")]
    return items == [_BUSINESS_CONTENT_SLOT]

#: 요청을 왜 그대로 계산하지 않았는지 구체적으로 밝히는 사유들.
_SPECIFIC_NON_COMPUTATION = (
    _SAFE_MESSAGES["non_additive_eps"],
    _SAFE_MESSAGES["fx_conversion_not_supported"],
    _SAFE_MESSAGES["incomparable_aggregation_scope"],
    # 「비율은 공시에서 직접 확인되는 값이 아니라 계산하지 않았다」가 이미
    # 왜인지 말한다. 그 뒤에 「확인할 수 없다」를 덧붙이면 부채총계·자본총계를
    # 바로 위에 제시해 놓고 확인하지 못했다고 말하는 셈이다(DEV-SAFE-012).
    _SAFE_MESSAGES["unsupported_operator"],
)

#: 대상 자체를 확인하지 못했다는 총칭 거절.
_GENERIC_UNSUPPORTED = _SAFE_MESSAGES["unsupported_semantic_target"]


def append_public_qualifications(text: str, messages: Iterable[str]) -> str:
    """Attach only material qualifications without a generic ``참고:`` tail.

    A qualification should explain how the conclusion must be read.  Rendering
    every execution default under a trailing ``참고:`` made ordinary answers
    look uncertain and trained readers to ignore the genuinely important
    boundaries.  Keep each material sentence, but place it in normal prose.
    Definitions that already begin with ``여기서`` remain direct; other
    limitations use ``다만`` so their relationship to the conclusion is clear.
    """

    answer = (text or "").rstrip()
    additions: list[str] = []
    pending = [(raw or "").strip() for raw in messages]
    # 왜 계산하지 않았는지를 이미 밝힌 답변에 「확인할 수 없다」를 덧붙이면
    # 앞 문장과 정면으로 어긋난다. DEV-INV-009 는 요청한 두 값을 제시하고
    # 집계 범위가 달라 차감하지 않았다고 설명한 직후에 그 값을 확인할 수
    # 없다고 말했다. 구체적인 사유가 있으면 총칭 거절은 싣지 않는다.
    if any(note in pending for note in _SPECIFIC_NON_COMPUTATION):
        pending = [note for note in pending if note != _GENERIC_UNSUPPORTED]
    # 사업부문·주요 제품 내용을 답이 이미 담고 있으면(K-001 등의 narrative
    # matrix 답처럼 claim label이 아니라 완성된 텍스트로만 판단해야 하는
    # 경로 포함) 「주요사업내용을 못 찾았다」는 같은 내용의 재부정을
    # 붙이지 않는다(#86-19 후속).
    if any(marker in answer for marker in _BUSINESS_CONTENT_COVERAGE_MARKERS):
        pending = [note for note in pending
                   if not is_business_content_only_gap(note)]
    qualified = False
    for message in pending:
        if not message or message in answer or message in additions:
            continue
        # 「다만」은 앞 결론과의 관계를 밝히는 말이라 한 번이면 족하다. 한계가
        # 여럿일 때 줄마다 붙이면 세 줄이 「다만, … 다만, … 다만, …」이 되어
        # 각 문장이 앞 문장을 뒤집는 것처럼 읽힌다. 실제로는 나란한 단서들이다.
        #
        # 「원공시 후보 접수번호: …」처럼 절이 아니라 이름표로 시작하는 줄에는
        # 애초에 붙일 자리가 없다.
        if message.startswith("여기서 ") or _LABELLED_NOTE.match(message):
            additions.append(message)
            continue
        additions.append(f"다만, {message}" if not qualified else message)
        qualified = True
    if not additions:
        return answer
    suffix = "\n".join(additions)
    return f"{answer}\n{suffix}" if answer else suffix


_PARTICLE = (
    "의", "는", "은", "이", "가", "를", "을", "에", "와", "과", "도", "만",
    "으로", "로", "에서", "부터", "까지", "보다", "처럼", "라", "이라",
)


def _states_subject(text: str, subject: str) -> bool:
    """Does the answer name this issuer, rather than merely contain the string?

    ``기아`` occurs inside ``기아자동차`` — the former name a user may have
    typed — so plain containment would read that as the canonical name being
    present.  An occurrence counts only when what follows it is a particle or
    a non-Hangul boundary, which is how a Korean sentence ends a proper noun.
    """

    start = text.find(subject)
    while start != -1:
        tail = text[start + len(subject):]
        head = tail[:1]
        if not head or not ("가" <= head <= "힣"):
            return True
        if any(tail.startswith(particle) for particle in _PARTICLE):
            return True
        start = text.find(subject, start + 1)
    return False


def subject_disclosure(text: str, claims: Iterable[object]) -> str | None:
    """Name the issuer whose filing the figures came from, when it is absent.

    A question may use a former or shortened name (``대우조선해양``,
    ``삼성엔지니어링``).  Retrieval resolves it correctly and the claim label
    carries the canonical ``corp_name``, but the composer sometimes echoes the
    surface from the question instead, leaving an answer that attributes a
    filing to a company that did not file it.  The template composer renders
    the label verbatim and never has this problem; this restores the same
    guarantee for a model-worded answer without discarding its wording.
    """

    subjects: list[str] = []
    for claim in claims:
        subject = (getattr(claim, "state", None) or "").strip()
        label = (getattr(claim, "label", None) or "")
        # ``state`` also carries event states (``terminated``).  Only a value
        # the label itself opens with is this claim's issuer.
        if not subject or not label.startswith(subject):
            continue
        if subject not in subjects and not _states_subject(text, subject):
            subjects.append(subject)
    if not subjects:
        return None
    named = "·".join(subjects)
    return (f"위 수치는 공시 제출 법인명 기준 «{named}» 의 공시에서 "
            "확인한 값입니다.")


def ensure_safe_limitations(
        text: str, limitations: Iterable[_LimitationLike], *,
        question: str | None = None,
        ) -> str:
    """Deterministically append protected notices omitted by a composer."""
    answer = (text or "").rstrip()
    missing = [message for message in render_safe_limitations(
        limitations, question=question)
               if message not in answer]
    return append_public_qualifications(answer, missing)
