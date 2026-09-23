"""TemplateComposer — payload → 결정적 자연어 답변 (HCX 없이 항상 동작하는 기준·폴백).

원칙: 확정 3조건을 통과한 claim만 값으로 문장화한다. 상태별 템플릿. 근거(접수번호·섹션/셀)를
문장에 붙인다. 숫자는 payload 값을 그대로 쓰고 새로 계산하지 않는다.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
import re

from agent.display_units_v1 import (
    parse_display_units_directive, render_display_units)
from agent.stage1_assembly import CORPUS_CUTOFF
from src.canonical.security import PROMPT_DATA_BEGIN, PROMPT_DATA_END
from .limitations import (_BUSINESS_CONTENT_COVERAGE_MARKERS,
                          append_public_qualifications,
                          is_business_content_only_gap,
                          render_safe_limitations, safe_limitation_message)
from .definitions import required_definition_messages
from .money import (
    claim_won_display, format_won_exact, normalize_source_money_unit)
from .narrative_matrix import NarrativeDigest, NarrativeMatrix

from app.textkit import (
    STATE_KO as _STATE_KO, collapse_padded_headings,
    fold_repeated_claim_subject, josa as _josa,
    strip_leading_item_number)

#: ``CORPUS_CUTOFF`` (YYYYMMDD) rendered as a public Korean date. Used only by
#: the more specific `_FUTURE_SUBMISSION_REQUEST` quality branch in
#: `_outside_corpus_reason`; the generic `unsupported_temporal_scope` wording
#: below stays a literal string so it matches the registered public text.
_CORPUS_CUTOFF_KO = (
    f"{int(CORPUS_CUTOFF[:4])}년 {int(CORPUS_CUTOFF[4:6])}월 "
    f"{int(CORPUS_CUTOFF[6:8])}일")

_REASON_KO = {
    "corp_not_in_universe": "",
    "future_forecast": "미래 실적 예측은 제공 공시로 답할 수 없으며 생성하지 않습니다.",
    "investment_advice": "투자 의견(매수/매도 등)은 제공하지 않습니다.",
    # The typed safe limitation below already explains why causality cannot be
    # established.  A second generic refusal sentence only repeats it.
    "causal_inference_beyond_scope": "",
    "prompt_injection_direct": "시스템 정보나 비밀 정보는 출력하지 않습니다.",
    "prompt_injection_role_impersonation": "시스템 지시를 사칭한 요청은 실행하지 않습니다.",
    "external_tool_request": "외부 시스템·API 접속은 수행하지 않으며 제공된 공시 데이터만 사용합니다.",
    "external_url_request": "외부 URL 접속은 수행하지 않으며 제공된 공시 데이터만 사용합니다.",
    "personal_data_request": "개인정보나 서비스에서 보호하는 식별정보는 제공하지 않습니다. 주민등록번호·집 주소뿐 아니라 법인·기관 등록번호도 보호 대상입니다.",
    # 이 둘은 종결 사유로 쓰이는데 문구가 없어 「대상이나 항목을 확인할 수
    # 없어…」라는 총칭으로 떨어졌다. DEV-EVT-028 「2026년 7월 1일까지 제출한
    # 최신 정기보고서」는 접수일 상한을 넘은 것이지 대상이 없는 것이 아니다 —
    # 자료에 없는 것인지 시스템이 못 하는 것인지 구별되지 않았다.
    "unsupported_temporal_scope": (
        "요청하신 시점이 제공 자료의 범위를 벗어납니다. 이 자료의 접수일은 "
        "2026년 6월 19일까지이며, 그 이후에 제출된 공시는 들어 있지 않습니다."),
    "corpus_coverage_unavailable": (
        "요청하신 범위를 제공 자료가 담고 있지 않아 확인해 드릴 수 없습니다."),
    "pressure_resisted": "",
    "ambiguous_event_identity": "후속 공시는 확인되지만 동일 상대방의 원공시가 복수여서 어느 원계약에 해당하는지는 공시만으로 특정할 수 없습니다.",
    "ambiguous_event_origin": "후속 공시는 확인되지만 동일 상대방의 원공시가 복수여서 어느 원계약에 해당하는지는 공시만으로 특정할 수 없습니다.",
    "counterparty_not_reported_may_hide_match": "계약상대가 공개되지 않은 관련 후보 공시가 있어, 이 후보가 요청한 상대방의 계약인지 아닌지는 확인할 수 없습니다.",
    "intraday_order_unavailable": "공시의 정정 순서는 확인되지만 장중 접수시각이 없어 실제 제출 시각과 분·초 단위 선후는 확인할 수 없습니다.",
    "source_scope_prevents_complete_lineage": "제공된 자료보다 앞선 원공시가 없어 해당 공시의 최초 기재 내용과 그 이전 변경 이력은 확인할 수 없습니다.",
    "source_excerpt_boundary_unavailable": "검증된 원문 값은 있으나 공개 길이 안에 완결된 문장이나 항목 경계가 없어 내용을 중간에서 잘라 제시하지 않았습니다.",
    "comparison_explanation_excerpt_unavailable": "두 금액의 차이는 계산했지만 산정 근거 원문을 공개 길이 안에서 완결된 문장으로 제시할 수 없어 원인을 임의로 요약하지 않았습니다.",
    # Source cross-check coverage is retained in the typed payload and trace.
    # It does not change a verified public claim and therefore is not a reader
    # qualification.
    "source_cross_check_partial": "",
    "slot_not_confirmed": "",
    "narrative_record_budget_exhausted": "",
    "narrative_result_budget_exhausted": "",
    "not_found": "이번에 확인한 공시자료에서는 요청하신 정보를 찾지 못했습니다.",
    "partial_unread": "같은 기간의 일부 공시 내용을 확인할 수 없어 최신 정정값까지 반영됐다고 단정할 수 없습니다.",
    "evidence_unavailable": "값 후보는 있으나 원문 근거 검증에 실패해 확정하지 않았습니다.",
    "extract_unsupported": "이번에 확인한 공시자료에서는 해당 항목을 확인하지 못했습니다.",
    "latest_annual_extract_unsupported": (
        "이번에 확인한 최신 연차 공시에서는 요청한 값을 찾지 못했으며, 이전 공시나 "
        "비슷한 항목의 값으로 대신하지 않았습니다."),
    "not_found_financial_sector_statement_semantics": (
        "금융·보험업 공시에는 일반 매출액으로 확정할 단일 공시 라인이 없어, "
        "다른 성격의 계정을 매출액으로 승격하지 않았습니다."),
    "not_found_raw_total_cell_empty": (
        "공시에 연결 당기순이익 합계가 기재돼 있지 않아, 하위 항목을 임의로 더해 "
        "당기순이익을 만들지 않았습니다."),
    "not_found_combined_concept_line": (
        "유형자산과 투자부동산 취득이 한 결합 계정으로만 공시되어, 순수 유형자산 "
        "취득액을 분리하거나 배분하지 않았습니다."),
    "unsupported_semantic_target": (
        "요청하신 대상이나 항목을 이번 공시자료에서 근거와 함께 확인할 수 없어, "
        "비슷하지만 다른 대상이나 항목의 내용을 대신 제시하지 않았습니다."),
}


#: 사용자에게 노출하면 안 되는 내부 필드명 → 사용자 어휘 (issue #172 M33-b,
#: 「내부 필드명 노출」#94 19 계열). `tests/test_public_wording_p0.py`가 이
#: 목록에 없는 새 내부 식별자가 그대로 새는지 계속 감시한다.
_INTERNAL_CLARIFY_FIELD_LABELS = {
    "account_path": "계정 항목",
    "item_id": "항목",
    "concept_override": "지표",
}

#: `app/tools/financial.py`가 계정 모호성(``ambiguous_account_path`` 등)일 때
#: 만드는 raw 역질문 문형 — ``"{corp_name} {concept} 조회에 (discriminator,)
#: 지정이 필요합니다."``. concept 는 항상 소문자 snake_case 내부 식별자라서
#: 회사명(한글·영문 혼용, 공백 가능)과 안전하게 구분된다.
_INTERNAL_CONCEPT_CLARIFY_QUESTION = re.compile(
    r"^(?P<corp>.+?)\s+(?P<concept>[a-z][a-z0-9_]*)\s+조회에\s+"
    r"\(.+?\)\s+지정이\s+필요합니다\.$")


def _extend_public_qualifications(lines: list[str], notes) -> None:
    """Append material boundaries as prose, never as a generic note block."""

    rendered = append_public_qualifications("\n".join(lines), notes)
    lines[:] = rendered.splitlines() if rendered else []

# A refusal should still tell the user which adjacent, evidence-backed query
# the system can answer.  These are capability-level alternatives keyed only
# to typed reason codes; they contain no company, date, value or fixture fact.
_ALTERNATIVE_KO = {
    "future_forecast": (
        "대신 제공 코퍼스에서 확인되는 과거 실적은 회사명·연도·지표를 "
        "지정하면 공시 근거와 함께 조회할 수 있습니다."
    ),
    "incomparable_aggregation_scope": (
        "의미와 집계 범위가 다른 투자계획 금액과 현금흐름표 실제액은 "
        "임의로 합산하거나 차감하지 않습니다."
    ),
    "investment_advice": (
        "투자 판단 대신 공시에서 확인되는 과거 실적과 계약·투자 현황은 "
        "근거와 함께 조회할 수 있습니다."
    ),
    "causal_inference_beyond_scope": (
        "대신 회사명과 기간을 알려주시면 공시 매출액과 증감률은 근거와 "
        "함께 조회할 수 있습니다."
    ),
}

_COMBINED_POLICY_ALTERNATIVE_KO = (
    "대신 회사명·기간·지표를 지정하면 과거 실적과 계약·투자 현황을 "
    "공시 근거와 함께 조회할 수 있습니다."
)


def _display_amount(claim, *, with_scale: bool = True) -> str:
    """사용자용 정확한 원화 표기. 원자료와 canonical 값은 변경하지 않는다.

    canonical 원 값을 반올림하지 않고 ``조·억·만원·원``으로 쪼갠다. 현금유출과
    손실은 label이 음수 의미를 이미 담으므로 절댓값을 표시하고, 다른 음수는
    선행 ``-``를 유지한다. ``with_scale``은 기존 호출 계약과의 호환을 위해
    남겨 두었으며 exact 표기에서는 별도 축약값을 추가하지 않는다.
    """

    exact_won = claim_won_display(claim)
    if exact_won:
        return exact_won
    return f"{claim.value_text or ''}{claim.raw_unit or ''}"



def _cite(c) -> str:
    parts = []
    seen = set()
    for ct in c.citations:
        # ``doc_id`` is an internal canonical key.  A public citation is
        # emitted only when the DART receipt number is available.
        loc = ct.rcept_no
        if not loc or loc in seen:
            continue
        seen.add(loc)
        # Locator strings such as ``TABLE[0]`` and ``TABLE-GROUP[18]`` are
        # internal preprocessing coordinates.  Keep them in retrieved_context
        # and typed citations for auditability, but expose only the public DART
        # receipt number in the natural-language answer.
        parts.append(f"접수번호 {loc}")
        if len(parts) >= 4:
            break
    return f" (근거: {'; '.join(parts)})" if parts else ""


def _is_event_summary(claim) -> bool:
    """Whether a claim is the requested lifecycle conclusion for a cutoff."""

    output_id = claim.output_id or ""
    return (
        ".state@" in output_id
        or ".termination_observed@" in output_id
        or ".semantic_status@" in output_id
    )


#: 이슈 #64 — 파생값(예: 4분기 단독)의 재료로만 쓰인 누적 operand를 "계산
#: 근거:" 줄에만 남기면, 그 값을 따로 물은 질문(「연간 매출액과 4분기 단독
#: 매출액을 각각 알려주고」)에서는 계산 부속물로만 읽힌다.  다만 이슈 #115 —
#: 질문이 그 재료 값을 따로 요구하지 않았을 때도 항상 독립 문장을 냈더니,
#: 재료 값이 독립 문장 한 번·계산 근거 한 번, 두 번 나왔다.  아래 표면
#: 신호(또는 「와 …도」처럼 추가 항목을 부르는 결합)가 있을 때만 독립
#: 문장을 남기고, 없으면 계산 근거에만 둔다.
_SEPARATE_OPERAND_REQUEST_SURFACE = re.compile(r"각각|모두|함께")
_ALSO_REQUESTED_OPERAND_SURFACE = re.compile(r"[와과][^,.!?와과]{0,12}도(?:\s|[,.!?]|$)")


def _question_requests_material_value_separately(question: str | None) -> bool:
    """Whether the question asked for a derivation's cumulative operand as
    its own answer item, not merely as calculation material (#64, #115)."""

    if not question:
        return False
    return bool(_SEPARATE_OPERAND_REQUEST_SURFACE.search(question)
                or _ALSO_REQUESTED_OPERAND_SURFACE.search(question))


def _is_state_only_event_question(question: str | None) -> bool:
    """Whether the public request asks only for lifecycle state timepoints."""

    if not question or re.search(r"상태|유효|살아\s*있|해지됐|종료됐", question) is None:
        return False
    # These nouns request additional facts or history and therefore keep the
    # detailed claims.  Dates and ``그날까지 접수된 공시`` merely qualify the
    # state axis and intentionally do not disqualify the concise path.
    return re.search(
        r"금액|대금|규모|사유|이유|원인|계약\s*상대|상대방|계약\s*명|"
        r"명칭|최초\s*(?:체결|공시).*내용|(?:변경|정정)\s*(?:이력|내역|흐름)|"
        r"전체\s*이력|효력\s*발생\s*조건|공시\s*(?:내용|내역|상세)",
        question,
    ) is None


def _event_state_time_key(claim) -> str:
    """Return a stable public-time key for an event state claim.

    The cutoff encoded after ``@`` is the semantic time axis.  Receipt numbers
    deliberately do not participate because they are not an authoritative
    intraday filing order.
    """

    output_match = re.search(r"@(\d{8})(?:\D|$)", claim.output_id or "")
    label_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", claim.label or "")
    return (
        output_match.group(1)
        if output_match
        else "".join(label_match.groups()) if label_match else ""
    )


def _final_termination_state(states):
    """Select the final explicitly terminated state on the semantic axis."""

    terminated = [
        claim for claim in states
        if claim.state == "terminated"
        or ".termination_observed@" in (claim.output_id or "")
    ]
    return max(terminated, key=_event_state_time_key) if terminated else None


def _final_termination_receipts(states) -> set[str]:
    """Receipts that prove the final selected termination state only."""

    final_state = _final_termination_state(states)
    if final_state is None:
        return set()
    return {
        citation.rcept_no
        for citation in (final_state.citations or ())
        if citation.rcept_no
    }


_TERMINATION_DETAIL_LABELS = {
    "계약금액", "최근매출액", "매출액대비", "계약상대", "해지금액",
    "해지사유", "해지일자", "해지계약명", "효력발생조건",
}


def _same_termination_receipt_details(claims, states) -> list:
    """Keep material details proven by the selected termination receipt.

    Event backends may return earlier contract snapshots beside the final
    termination.  Public yes/no validity answers need the latter's amount,
    date, counterparty, reason and explanatory paragraph, not an unrelated
    historical advisory that merely appeared first in claim order.
    """

    receipts = _final_termination_receipts(states)
    if not receipts:
        return []
    return [
        claim for claim in claims
        if claim.label in _TERMINATION_DETAIL_LABELS
        and any(citation.rcept_no in receipts
                for citation in (claim.citations or ()))
    ]


def _needs_termination_detail_answer(payload, question: str | None,
                                     states, claims) -> bool:
    """Whether a compact state answer would omit a material explanation."""

    if not _same_termination_receipt_details(claims, states):
        return False
    if any(verdict.verdict == "false" for verdict in payload.premise_verdicts):
        return True
    # Binary validity/termination questions are answered more safely with the
    # date/reason that explains a negative conclusion.  Plain requests such as
    # ``계약 상태 알려줘`` remain on the concise path.
    return bool(question and re.search(
        r"유효(?:한가|한지|하지|해(?:\?|$))|"
        r"해지(?:됐|되었|인가)|종료(?:됐|되었|인가)",
        question,
    ))


_TYPED_STATE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(
        re.escape(state) for state in sorted(_STATE_KO, key=len, reverse=True)
    ) + r")(?![A-Za-z0-9_])"
)


def _public_premise_detail(detail: str | None) -> str:
    """Translate only the backend-generated actual-state part of a premise.

    The text before ``실제 상태`` can contain the user's literal wording.  It
    must not be globally rewritten merely because the user happened to type an
    English word such as ``active``.  Backend typed state values appear only in
    the suffix, so translating that closed suffix keeps the stored
    ``PremiseVerdict`` and its audit value unchanged.
    """

    value = detail or ""
    amount_match = re.fullmatch(
        r"주장\s+(?P<claimed>[\d,]+)원\s+vs\s+실제\s+(?P<actual>[\d,]+)원",
        value,
    )
    if amount_match:
        claimed = format_won_exact(amount_match.group("claimed").replace(",", ""))
        actual = format_won_exact(amount_match.group("actual").replace(",", ""))
        if claimed and actual:
            return (
                f"질문에 제시된 금액은 {claimed}이지만 공시에서 확인된 "
                f"금액은 {actual}입니다"
            )
    subject_match = re.fullmatch(
        r"주장 주체\s+'(?P<claimed>[^']+)'\s+vs\s+실제 최대\s+'(?P<actual>[^']+)'",
        value,
    )
    if subject_match:
        return (
            f"질문에서는 {subject_match.group('claimed')}를 가장 큰 값의 "
            f"주체로 전제했지만, 공시 기준으로는 "
            f"{subject_match.group('actual')}입니다"
        )
    state_match = re.fullmatch(
        r"주장\s+'(?P<claimed>[^']+)'\s+vs\s+실제 상태\s+(?P<actual>[A-Za-z0-9_]+)",
        value,
    )
    if state_match:
        actual = _STATE_KO.get(state_match.group("actual"), state_match.group("actual"))
        return (
            f"질문에 제시된 상태는 ‘{state_match.group('claimed')}’이고, "
            f"공시에서는 ‘{actual}’으로 확인됩니다"
        )
    comparison_match = re.fullmatch(
        r"주장\s+'(?P<claimed>[^']+)'\s+vs\s+검증 결과\s+(?P<actual>.+)",
        value,
    )
    if comparison_match:
        actual = comparison_match.group("actual").strip()
        # The verifier stores a compact audit token such as ``다르다`` or
        # ``같다``.  Keep arbitrary longer verifier details verbatim, but turn
        # only these closed tokens into a complete public sentence.
        actual = {
            "다르다": "두 값이 다릅니다",
            "다름": "두 값이 다릅니다",
            "같다": "두 값이 같습니다",
            "같음": "두 값이 같습니다",
        }.get(actual, actual)
        return (
            f"질문의 전제는 ‘{comparison_match.group('claimed')}’이지만, "
            f"공시를 대조한 결과 {actual}"
        )
    correction_match = re.fullmatch(
        r"주장\s+'(?P<claimed>[^']+)'\s+vs\s+정정\s+(?P<actual>있음|없음)",
        value,
    )
    if correction_match:
        claimed = correction_match.group("claimed")
        actual_exists = correction_match.group("actual") == "있음"
        claimed_absent = "없" in claimed or "무정정" in claimed
        if actual_exists and claimed_absent:
            return (
                "질문에서는 정정 이력이 없다고 전제했지만, 공시에는 "
                "정정 이력이 확인됩니다"
            )
        if not actual_exists and not claimed_absent:
            return (
                "질문에서는 정정 이력이 있다고 전제했지만, 공시에서는 "
                "정정 이력이 확인되지 않습니다"
            )
        return (
            "요청한 공시에는 정정 이력이 확인됩니다"
            if actual_exists else
            "요청한 공시에서는 정정 이력이 확인되지 않습니다"
        )
    if value == "정정 전 값이 공시유보·미기재 상태여서 정정 후 공개를 증감으로 판정할 수 없음":
        return (
            "정정 전에는 계약금액이 공개되지 않아 이번 공개를 "
            "금액의 증가나 감소로 볼 수 없습니다"
        )
    head, marker, actual = value.partition("실제 상태")
    if not marker:
        return value
    actual = _TYPED_STATE_TOKEN.sub(
        lambda match: _STATE_KO.get(match.group(0), match.group(0)), actual)
    return head + marker + actual


#: 「질문에서는 A를 가장 큰 값의 주체로 전제했지만, 공시 기준으로는 B입니다」.
#: 같은 답변의 「비교 결과 B가 더 큽니다」와 같은 말이다.
_ARGMAX_PREMISE_DETAIL = re.compile(r"가장 큰 값의 주체로 전제했지만")


def _premise_lines(verdicts, *, include_true: bool = False,
                   claims=()) -> list[str]:
    """Render premise verdicts once, with reader-facing details below it.

    전제가 틀렸다는 사실과 무엇이 맞는지는 다른 말이다. 그런데 최대값 주체를
    잘못 짚은 질문에서는 뒤엣것을 답변이 이미 말한다 — 「비교 결과 삼성전자가
    더 큽니다」. 그때는 전제 쪽 부연을 싣지 않는다. 전제가 틀렸다는 첫 문장은
    그대로 남으므로 사용자는 자기 질문이 어긋났다는 것을 안다.
    """

    has_argmax = any(getattr(c, "operator", None) == "argmax" for c in claims)
    false_details = [
        _public_premise_detail(verdict.detail).strip()
        for verdict in verdicts
        if verdict.verdict == "false" and (verdict.detail or "").strip()
    ]
    lines: list[str] = []
    if has_argmax:
        false_details = [d for d in false_details
                         if not _ARGMAX_PREMISE_DETAIL.search(d)]
    if false_details or (has_argmax and any(
            v.verdict == "false" for v in verdicts)):
        lines.append("질문의 전제는 공시 내용과 다릅니다.")
        lines.extend(
            detail if detail.endswith((".", "?", "!")) else detail + "."
            for detail in dict.fromkeys(false_details)
        )
    if include_true:
        true_details = [
            _public_premise_detail(verdict.detail).strip()
            for verdict in verdicts
            if verdict.verdict == "true" and (verdict.detail or "").strip()
        ]
        lines.extend(
            "질문의 전제는 공시 내용과 일치합니다. "
            + (detail if detail.endswith((".", "?", "!")) else detail + ".")
            for detail in dict.fromkeys(true_details)
        )
    return lines


#: 현금흐름표의 지급·상환성 항목. 라벨 자체가 방향을 말하므로 음수 부호를
#: 붙이면 뜻이 겹치면서 오히려 읽히지 않는다.
_CASH_PAYMENT_ITEM = re.compile(r"지급|상환|납부|취득|매입")


#: 라벨의 주어가 「차이」인 표면. 차이는 증가·감소하는 주체가 아니다.
_DIFFERENCE_SUBJECT = re.compile(r"차이$")

#: ``… 전기 대비 증감률`` 라벨의 꼬리. 앞부분이 문장의 주어다. 비율 피연산자의
#: 상대 증감률(이슈 #62, ``… 전기 대비 상대 증감률``)도 같은 꼬리 모양이다.
_PERIOD_CHANGE_SUBJECT = re.compile(r"\s*전기\s*대비\s*(?:상대\s*)?증감률$")

#: 주어가 한쪽 극성만 가리키는 자리. 부호가 바뀐 변화는 양쪽을 함께
#: 가리켜야 하므로 ``손익`` 으로 바꾼다.
_POLARITY_IN_SUBJECT = re.compile(r"(?:이익|손실)$")

#: 한 문장으로 합칠 claim 수의 위쪽 한계. 축 하나가 후보를 아무리 늘려도
#: (「빚」 셋, 「벌었어」 셋) 이 세션에서 실제로 나오는 값은 최대 3이다.
#: 그 이상은 한 문장이 오히려 읽기 어려워지므로 합치지 않고 줄마다 낸다.
_PARALLEL_AXIS_LIMIT = 4

#: 시점을 가리키는 토큰 — 연도·분기·반기·「말」. `_parallel_axis_sentence`
#: 가 유일하게 다른 자리로 이것을 보면 합치지 않는다. 시점만 다른 두 값을
#: 파생 없이 나란히 두면 암묵적인 증감 비교로 읽힌다.
_PERIOD_LIKE_TOKEN = re.compile(
    r"(?:19|20)[0-9]{2}년|분기|반기|말$")

#: 값이 있을 때 `_claim_line` 이 그 값만의 특별한 문장을 쓰는 state.
#: `_parallel_axis_sentence` 가 이 state 를 가진 claim 을 합치면 그 특별한
#: 뜻(예: `explicit_zero` 의 "명시된 0원", 이슈 #63/RPC-006)을 조용히
#: 지우게 된다.
_STATE_MARKED_AMOUNT = frozenset({"explicit_zero", "decreased", "increased"})


#: 라벨 안의 기간 표면. 「전기 대비」를 실제 연도로 바꿀 때 쓴다(#94 26).
#: 연도만 떼면 「2025년 말」이 「2024년 대비」가 되어 시점이 흐려지므로
#: 말·분기·반기 꼬리까지 함께 잡는다.
_PERIOD_SURFACE = re.compile(
    r"(?<![0-9])(?:19|20)[0-9]{2}\s*년"
    r"(?:\s*(?:말|상반기|반기|[1-4]\s*분기(?:\s*누적)?))?")


def _explicit_base_period(subject: str, pair_label: str) -> str | None:
    """비교 **대상** 기간의 표면. 하나로 못 정하면 ``None``.

    이슈 #94 26 — 「전기 대비」는 공시 원문의 말이라 읽는 사람이 어느 해인지
    바로 알 수 없다. 그런데 그 해는 이미 손에 있다: difference 의 라벨이
    두 피연산자를 모두 적는다.

    ```
    주어   기아 2025년 연결 영업이익
    라벨   기아 2025년 연결 영업이익과 기아 2024년 연결 영업이익의 차이
                                        ~~~~~~ 이것이 비교 대상이다
    ```

    주어에 있는 기간을 빼고 **정확히 하나**가 남을 때만 답한다. 둘 이상이
    남거나(피연산자가 기간 말고도 갈린다) 하나도 없으면(같은 기간을 다른
    축으로 비교한다) 지어내지 않고 「전기 대비」를 그대로 둔다.
    """

    def _key(value: str) -> str:
        return re.sub(r"\s+", "", value)

    in_subject = {_key(row) for row in _PERIOD_SURFACE.findall(subject)}
    others: list[str] = []
    for row in _PERIOD_SURFACE.findall(pair_label):
        if _key(row) in in_subject or _key(row) in {_key(v) for v in others}:
            continue
        others.append(row)
    if len(others) != 1 or not in_subject:
        return None
    # 비교에는 공백을 지운 형태를 썼지만 내보내는 것은 읽는 형태다
    # (「2024년말」이 아니라 「2024년 말」).
    return re.sub(r"\s+", " ", others[0]).strip()


def _claim_numeric_sign(claim) -> int | None:
    """Return the exact typed sign without changing the stored claim value."""

    for raw in (claim.canonical_value, claim.value_text):
        if raw is None:
            continue
        value = str(raw).strip().replace(",", "")
        if value.startswith("(") and value.endswith(")"):
            value = "-" + value[1:-1]
        try:
            number = Decimal(value)
        except InvalidOperation:
            continue
        return 1 if number > 0 else (-1 if number < 0 else 0)
    return None


def _directional_difference_ids(claims) -> set[str]:
    """Identify ordered period changes, never cross-sectional distances.

    A ``difference`` paired with ``percent_change`` over the same ordered
    operands is necessarily a change over an ordered baseline in the current
    derivation contract.  The explicit ``증감/전기 대비/전년 대비`` labels
    cover the deterministic supplement path even when the percentage claim is
    absent.  ``absolute_difference`` is intentionally excluded because it is
    an unsigned distance.
    """

    percent_pairs = {
        tuple(claim.derived_from)
        for claim in claims
        if claim.operator == "percent_change" and len(claim.derived_from) == 2
    }
    cues = ("증감", "전기 대비", "전년 대비", "직전 대비")
    return {
        claim.output_id
        for claim in claims
        if claim.operator == "difference"
        and (
            tuple(claim.derived_from) in percent_pairs
            or any(cue in (claim.label or "") for cue in cues)
        )
    }


_EXPLICIT_TABLE_REQUEST = re.compile(
    r"(?:(?<![가-힣])표로|(?:비교|요약|정리)표로|(?:표|도표)\s*형식|테이블)"
)
_EVENT_TABLE_QUESTION_SLOTS = {
    "상대방": re.compile(r"상대(?:방)?|계약\s*상대"),
    "금액": re.compile(r"금액"),
    "사유": re.compile(r"사유"),
}
_EVENT_NONDEFAULT_TABLE_SLOT = re.compile(
    r"효력\s*발생|상태|계약\s*기간|시작일|종료일"
)
_EVENT_TABLE_SLOTS = {
    "상대방": "상대방",
    "계약상대": "상대방",
    "해지금액": "금액",
    "계약금액": "금액",
    "해지사유": "사유",
    "해지일자": "해지일",
    "매출액대비": "최근매출액 대비",
}

# ``derived_from`` is also used by a few selector/projection claims to retain
# provenance.  It does not, by itself, mean that the answer performed
# arithmetic.  Only these operators have operands that a reader needs as a
# separate "계산 근거" line.
_CALCULATION_OPERATORS = {
    "difference", "absolute_difference", "percent_change",
    "discrete_from_cumulative", "sum",
}


def _is_calculation_claim(claim) -> bool:
    """Whether ``claim`` is a real arithmetic result, not a provenance link."""

    return bool(
        claim.operator in _CALCULATION_OPERATORS
        and len(claim.derived_from or ()) >= 2
    )


#: 이슈 #38 — concept_ratio 결과 문장의 지표명으로 쓸 이름 있는 비율 표면.
#: `agent.stage1_v1_financial_backend.NAMED_RATIO_CONCEPTS` 의 키(사전
#: 이름)와 같은 집합이다 — composer 는 agent 계층을 import 하지 않으므로
#: 여기서는 질문 문장에 그 이름이 그대로 나타나는지만 본다(개념 해석은
#: 이미 backend 가 끝냈고, composer 는 표시용 이름만 고른다).
_CONCEPT_RATIO_NAMED_SURFACES = (
    "영업이익률", "순이익률", "매출원가율", "부채비율",
    "유동비율", "자기자본비율", "ROE", "ROA",
)


def _concept_ratio_named_surface(question: str | None) -> str | None:
    """질문 표면에 이름 있는 비율(영업이익률 등)이 있으면 그 이름을 돌려준다."""

    if not question:
        return None
    for name in _CONCEPT_RATIO_NAMED_SURFACES:
        if name in question:
            return name
    return None


def _concept_ratio_common_prefix(a_label: str, b_label: str) -> str:
    """분자·분모 라벨이 공유하는 회사·기간·scope 접두(어절 단위 공통 선행부)."""

    a_tokens = (a_label or "").split()
    b_tokens = (b_label or "").split()
    common: list[str] = []
    for ta, tb in zip(a_tokens, b_tokens):
        if ta != tb:
            break
        common.append(ta)
    return " ".join(common)


def _concept_ratio_metric_suffix(label: str, prefix: str) -> str:
    """공통 접두를 뗀 나머지 — 개념 이름만 남긴다(못 떼면 라벨 그대로)."""

    label = label or ""
    if prefix and label.startswith(prefix):
        rest = label[len(prefix):].lstrip()
        return rest or label
    return label


def _concept_ratio_subject(numerator, denominator, *, is_percent: bool,
                           question: str | None) -> tuple[str, str | None]:
    """concept_ratio 결과 문장의 주어(지표명 포함)와, 있다면 계산식 문구.

    지표명은 질문 표면에 이름 있는 비율(영업이익률 등)이 있으면 그것을 쓰고,
    없으면 ``<분자> ÷ <분모>`` 로 대신한다(사전에 이름 있는 비율은 전부
    percent 이므로 multiple 은 아예 찾지 않는다). 이름 있는 비율을 썼을
    때만 ``(계산: 분자 ÷ 분모)`` 문구를 함께 낸다 — 이름이 없는 경우에는
    주어 자체가 이미 그 계산식이라 되풀이하지 않는다.
    """

    prefix = _concept_ratio_common_prefix(numerator.label, denominator.label)
    numerator_metric = _concept_ratio_metric_suffix(numerator.label, prefix)
    denominator_metric = _concept_ratio_metric_suffix(denominator.label, prefix)
    formula = f"{numerator_metric} ÷ {denominator_metric}"
    named = _concept_ratio_named_surface(question) if is_percent else None
    metric = named or formula
    subject = f"{prefix} {metric}".strip() if prefix else metric
    return subject, (formula if named else None)


def _normal_public_label(label: str) -> str:
    """Normalize mechanical period labels without changing their meaning.

    Stage 2 may join independently typed period/scope surfaces.  At the public
    boundary, ``3분기 단독 연결 3분기`` is needlessly repetitive and difficult
    to read even though each token is individually correct.  This recognizes
    only that exact duplicated shape; it never infers a period or a scope.
    """

    value = " ".join((label or "").split())
    # 접수번호는 근거 열이 이미 싣는다. 라벨 앞에 또 붙으면 같은 값을 두 번
    # 읽게 되고, 뒤따르는 표 경로와 붙어 기계 좌표처럼 보인다.
    value = re.sub(r"^\d{14}\s+", "", value)
    # 표 경로의 ``>`` 는 기계 문법이다. 같은 표의 형제 행이 쓰는
    # ``'2. 계약내역'의 '계약금액'`` 표기로 맞춘다.
    if " > " in value:
        parts = [part.strip() for part in value.split(" > ") if part.strip()]
        if len(parts) >= 2:
            value = "의 ".join(f"'{part}'" for part in parts)
    # Numeric values already carry their public unit.  Leaving the source
    # header suffix in a sentence produces mechanical Korean such as
    # ``시설자금 (원)는 3,541억원``.
    value = re.sub(
        r"\s*\((?:원|천원|백만원|억원|조원|%)\)\s*$", "", value)
    # A standalone quarter is derived from a cumulative filing period.  The
    # typed label can consequently contain both the requested quarter and the
    # source period (``2분기 단독 연결 상반기``).  The calculation is correct,
    # but showing both periods makes a lay reader wonder whether the answer is
    # quarterly or half-yearly.  Remove only the known cumulative source role
    # after ``단독``; the requested quarter and statement scope stay intact.
    value = re.sub(
        r"(?P<head>.*?)(?P<quarter>[1-4]분기) 단독 "
        r"(?P<scope>연결|별도)(?: 기준)? "
        r"(?:상반기|[1-4]분기\s*누적|누적)(?P<tail>.*)",
        lambda match: (
            f"{match.group('head')}{match.group('scope')} 기준 "
            f"{match.group('quarter')} 단일 분기{match.group('tail')}"),
        value,
    )
    value = re.sub(
        r"(?P<head>.*?)(?P<quarter>[1-4]분기) 단독 "
        r"(?P<scope>연결|별도)(?: 기준)? (?P=quarter)(?P<tail>.*)",
        lambda match: (
            f"{match.group('head')}{match.group('scope')} 기준 "
            f"{match.group('quarter')} 단일 분기{match.group('tail')}"),
        value,
    )
    return value


def _is_free_share_allocation_ratio_claim(claim) -> bool:
    """Whether this scalar needs an explicit existing-share denominator."""

    label = _normal_public_label(str(getattr(claim, "label", "") or ""))
    return bool(
        getattr(claim, "value_text", None)
        and getattr(claim, "raw_unit", None) == "주"
        and "신주" in label and "배정" in label and "비율" in label
    )


def _view_label_overrides(claims) -> dict[str, str]:
    """Keep only view labels already bound by the financial backend.

    ``derived_from`` order is an arithmetic operand order, not a public view
    contract; subtraction can intentionally be latest-minus-original.  Never
    infer as-filed/restated from that order.  The backend prefixes the typed
    claim label, and this display helper merely normalizes whitespace while
    preventing a second prefix from being added.
    """

    prefixes = ("최초 제출값", "최신 재작성값")
    return {
        claim.output_id: _normal_public_label(claim.label)
        for claim in claims
        if any(_normal_public_label(claim.label).startswith(prefix)
               for prefix in prefixes)
    }


def _markdown_cell(value: str) -> str:
    """Keep verified text in one Markdown cell without changing its meaning."""

    return " ".join((value or "").split()).replace("|", r"\|")


#: 「주요사업내용」은 사업부문·주요 제품 표가 이미 전달하는 것과 같은 사업
#: 내용을 묻는 별도 slot 이름일 뿐이다. 그 표를 이미 제시한 답에서 이
#: slot 만 못 찾았다고 따로 밝히면, 방금 준 내용을 못 찾았다고 말하는
#: 것처럼 읽힌다(K-001·K-025 등, #86-19). 판정 자체(``_SLOT_NOT_CONFIRMED_
#: ITEMS`` 정규식)는 narrative matrix 경로도 함께 쓰도록 limitations.py
#: 에 있다 — 여기서는 claim label 이 있는 이 경로에만 필요한 커버리지
#: 판정만 더한다.
_BUSINESS_CONTENT_COVERAGE_LABELS = _BUSINESS_CONTENT_COVERAGE_MARKERS
_is_business_content_only_gap = is_business_content_only_gap


def _business_segment_table_already_shown(claims) -> bool:
    return any(
        any(marker in (claim.label or "")
            for marker in _BUSINESS_CONTENT_COVERAGE_LABELS)
        for claim in claims
    )


def _event_qualification_notes(limitations) -> list[str]:
    """Render event-identity limits together with their candidate receipts.

    The affected documents are the evidence for the ambiguity itself.  They
    must not be promoted to an event edge, but omitting them makes a public
    ``cannot identify the original`` statement impossible to audit.
    """

    notes = list(render_safe_limitations(limitations))
    superseded_by_candidates: set[str] = set()
    for limitation in limitations:
        code = limitation.code.split(":", 1)[0]
        if code not in {"ambiguous_event_identity", "ambiguous_event_origin"}:
            continue
        origins = list(dict.fromkeys(
            receipt for receipt in limitation.affected_doc_ids
            if re.fullmatch(r"\d{14}", receipt or "")
        ))
        if len(origins) >= 2:
            # 「후보」라는 말이 이미 확정이 아님을 뜻한다. 안전 문구가 바로
            # 위에서 같은 「어느 원계약에 해당하는지는 공시만으로 특정할 수
            # 없습니다」를 이미 말했으므로, 후보 목록을 붙이는 자리에서는
            # 그 안전 문구를 지운다 — 둘 다 남기면 같은 말을 두 번 한다
            # (#86-14).
            notes.append("원공시 후보 접수번호: " + ", ".join(origins) + ".")
            superseded_by_candidates.add(code)
    if superseded_by_candidates:
        hidden = {
            safe_limitation_message(code) for code in superseded_by_candidates
        }
        notes = [note for note in notes if note not in hidden]
    return list(dict.fromkeys(notes))


def _public_date(value: str) -> str:
    """Display a typed YYYYMMDD time key without changing its ordering."""

    return (f"{value[:4]}-{value[4:6]}-{value[6:]}"
            if re.fullmatch(r"\d{8}", value or "") else value or "-")


def _event_state_table(states) -> str:
    """Render distinct typed lifecycle observations on their semantic axis."""

    ordered = sorted(states, key=_event_state_time_key)
    rows = []
    seen = set()
    for claim in ordered:
        date = _event_state_time_key(claim)
        key = (date, claim.state, tuple(
            citation.rcept_no for citation in (claim.citations or ())))
        if not date or key in seen:
            continue
        seen.add(key)
        rows.append((date, claim))
    if len(rows) < 2:
        return ""

    first_date, first = rows[0]
    last_date, last = rows[-1]
    first_state = _STATE_KO.get(first.state, first.text or first.state or "-")
    last_state = _STATE_KO.get(last.state, last.text or last.state or "-")
    lines = [
        (f"{_public_date(first_date)}에는 {first_state}, "
         f"{_public_date(last_date)}에는 {last_state} 상태로 확인됩니다."),
        "",
        "| 기준일 | 공시상 상태 | 근거 |",
        "|---|---|---|",
    ]
    for date, claim in rows:
        evidence = _cite(claim).strip().removeprefix("(근거: ").removesuffix(")")
        state = _STATE_KO.get(claim.state, claim.text or claim.state or "-")
        if claim.state in {"facility_before_start", "facility_usage_approved"} and claim.text:
            state = claim.text
        lines.append(
            f"| {_markdown_cell(_public_date(date))} | "
            f"{_markdown_cell(state)} | {_markdown_cell(evidence or '-')} |"
        )
    return "\n".join(lines)


def _withheld_contract_amount_answer(payload, question: str | None) -> str | None:
    """Answer an availability question from a complete set of withheld amounts."""

    if not question or re.search(r"확인할 수|알 수|공개|금액", question) is None:
        return None
    amount_claims = [
        claim for claim in payload.claims
        if claim.output_id.endswith(".계약금액") and claim.state == "withheld"
    ]
    if not amount_claims:
        return None

    # Some withheld contract filings still disclose a physical supply scale
    # (for example GWh).  It is often the only useful quantitative answer left
    # once money is unavailable, so preserve that verified surface without
    # replaying the full advisory paragraph.
    scale_by_root: dict[str, tuple[str, object]] = {}
    for claim in payload.claims:
        if not claim.output_id.endswith(".기타사항") or not claim.text:
            continue
        match = re.search(
            r"(?<![\d,.])(?P<value>\d[\d,.]*\s*"
            r"(?:TWh|GWh|MWh|kWh|Wh))(?![A-Za-z])",
            claim.text,
            re.IGNORECASE,
        )
        if match:
            scale_by_root[claim.output_id.removesuffix(".기타사항")] = (
                match.group("value").replace(" ", ""), claim)
    show_scale = bool(scale_by_root)

    lines = [
        ("해당 기준일까지 확인된 공시에서는 계약금액을 "
         f"확인할 수 없습니다. 확인된 {len(amount_claims)}건 모두 금액이 "
         "공시유보 상태입니다."),
        "",
        ("| 원공시 접수번호 | 계약금액 공개 여부 | 공개된 공급 규모 | 근거 |"
         if show_scale else
         "| 원공시 접수번호 | 계약금액 공개 여부 | 근거 |"),
        ("|---|---|---:|---|" if show_scale else "|---|---|---|"),
    ]
    for claim in amount_claims:
        receipt = next((citation.rcept_no for citation in claim.citations
                        if citation.rcept_no), "-")
        root = claim.output_id.removesuffix(".계약금액")
        scale, scale_claim = scale_by_root.get(root, ("-", None))
        cited_claims = (claim, scale_claim) if scale_claim is not None else (claim,)
        evidence = ", ".join(dict.fromkeys(
            citation.rcept_no
            for cited in cited_claims
            for citation in (cited.citations or ())
            if citation.rcept_no
        ))
        cells = [
            _markdown_cell(receipt),
            _markdown_cell(_display_text(claim.text or _STATE_KO["withheld"])),
        ]
        if show_scale:
            cells.append(_markdown_cell(scale))
        cells.append(_markdown_cell(
            ", ".join(f"접수번호 {value}" for value in evidence.split(", "))
            if evidence else "-"))
        lines.append("| " + " | ".join(cells) + " |")

    unknowns = [
        claim for claim in payload.claims
        if claim.output_id.endswith(".counterparty_unknown") and claim.text
    ]
    if unknowns:
        receipts = list(dict.fromkeys(
            citation.rcept_no
            for claim in unknowns
            for citation in (claim.citations or ())
            if citation.rcept_no
        ))
        evidence = (f"(접수번호 {', '.join(receipts)})" if receipts else "")
        lines.extend((
            "",
            f"계약상대가 비공개인 관련 후보 공시도 {len(unknowns)}건 "
            f"확인됩니다{evidence}.",
        ))
    notes = render_safe_limitations(payload.limitations)
    if notes:
        _extend_public_qualifications(lines, notes)
    return "\n".join(lines)


def _termination_membership_answer(payload, question: str | None) -> str | None:
    """Show only positive members for a verified termination-set query.

    The event tool keeps every scanned contract claim so the closed-world
    membership proof remains auditable.  A reader asking *which* contracts
    have a termination disclosure, or simply *whether any* do (이슈 #74 후속,
    SG-004~006: "...체결한 계약 중 이후 해지된 계약이 존재하는가?"), does not
    need that negative remainder in the public answer.  This projection
    therefore uses only typed terminated roots and typed ambiguous
    termination observations; it never infers a termination from a
    candidate original.
    """

    if not question:
        return None
    compact = re.sub(r"\s+", "", question)
    which_query = (
        "해지공시" in compact and "확인" in compact and "계약" in compact
        and re.search(r"무엇|어느|어떤|뭐|알려|찾", compact) is not None)
    existence_query = (
        "계약" in compact and "해지" in compact
        and re.search(r"존재하(?:는가|나|해)|있(?:는가|나요?)", compact) is not None)
    if not (which_query or existence_query):
        return None

    members = [
        claim for claim in payload.claims
        if (claim.state in {"terminated", "termination_observed"}
            and claim.operator is None and not claim.derived_from)
    ]
    confirmed = [claim for claim in members if claim.state == "terminated"]
    ambiguous = [
        claim for claim in members if claim.state == "termination_observed"]
    if not members:
        count = next((
            claim for claim in payload.claims
            if claim.output_id.endswith(".terminated_count")
        ), None)
        if count is None or count.value_text != "0":
            return None
        return f"제공된 자료에서는 해지 공시가 확인되는 계약을 찾지 못했습니다.{_cite(count)}"

    lines: list[str] = []
    if confirmed:
        lines.append(
            "제공 코퍼스에서 원계약까지 해지 연결이 확인되는 계약은 "
            f"{len(confirmed)}건입니다."
        )
        lines.append("원계약까지 확인된 계약:")
        lines.extend(
            f"- {_display_text(claim.text or claim.label)}{_cite(claim)}"
            for claim in confirmed
        )
    if ambiguous:
        lines.append(
            ("별도로 해지 공시는 확인되지만 원계약을 특정할 수 없는 건은 "
             f"{len(ambiguous)}건입니다.")
        )
        lines.append("원계약이 확정되지 않은 해지 공시:")
        lines.extend(
            f"- {_display_text(claim.text or claim.label)}{_cite(claim)}"
            for claim in ambiguous
        )

    # The answer above already explains the ambiguous state.  Keep the
    # candidate receipts, which are the audit evidence for that limitation,
    # and avoid repeating the same generic caveat twice.
    notes = _event_qualification_notes(payload.limitations)
    if ambiguous:
        notes = [note for note in notes
                 if not note.startswith("후속 공시는 확인되지만")]
    if notes:
        _extend_public_qualifications(lines, notes)
    return "\n".join(lines)


def _ambiguous_termination_observation_answer(
        payload, question: str | None) -> str | None:
    """Lead with the observed filing when its original event is ambiguous.

    A termination filing can state its counterparty, amount and date exactly
    while two earlier originals remain indistinguishable. Rendering the
    generic ``... contract is terminated`` claim first makes the ambiguity
    look like an afterthought. This projection instead answers only from the
    termination receipt, then states the unresolved original-event boundary.
    It is keyed to the typed limitation and requested fields, not an issuer or
    fixture.
    """

    if not question:
        return None
    compact = re.sub(r"\s+", "", question)
    # Full history/latest-document requests need the dedicated chronology
    # renderer; this projection is only for a direct state/amount question.
    if re.search(r"정정|이력|흐름|최신공시|원공시|전체", compact):
        return None
    if re.search(r"상태|유효|살아있|해지|종료|끝난", compact) is None:
        return None

    ambiguous = [
        limitation for limitation in payload.limitations
        if limitation.code.split(":", 1)[0] in {
            "ambiguous_event_identity", "ambiguous_event_origin"}
    ]
    origins = list(dict.fromkeys(
        receipt
        for limitation in ambiguous
        for receipt in limitation.affected_doc_ids
        if re.fullmatch(r"\d{14}", receipt or "")
    ))
    if len(origins) < 2:
        return None

    states = [claim for claim in payload.claims if _is_event_summary(claim)]
    final_state = _final_termination_state(states)
    if final_state is None:
        return None
    details = _same_termination_receipt_details(payload.claims, states)
    by_label = {claim.label: claim for claim in details}
    counterparty = by_label.get("계약상대")
    contract_name = by_label.get("해지계약명")
    termination_date = by_label.get("해지일자")
    subject_parts = []
    if counterparty is not None and counterparty.text:
        subject_parts.append(f"{_display_text(counterparty.text)}를 상대방으로 한")
    if contract_name is not None and contract_name.text:
        subject_parts.append(_display_text(contract_name.text))
    subject = " ".join(subject_parts) or "해당 계약"
    date = (
        _display_text(termination_date.text)
        if termination_date is not None and termination_date.text
        else _public_date(_event_state_time_key(final_state))
    )
    prefix = (
        "아니요. "
        if any(verdict.verdict == "false"
               for verdict in payload.premise_verdicts)
        else ""
    )
    lines = [
        (f"{prefix}{date}에 {subject} 해지 공시는 확인됩니다."
         f"{_cite(final_state)}"),
        ("다만 같은 상대방의 원공시가 복수라, 이 해지 공시가 어느 "
         "원계약에 해당하는지는 공시만으로 특정할 수 없습니다."),
    ]

    requested_patterns = {
        "해지금액": r"금액|대금|규모|얼마",
        "매출액대비": r"매출액대비|매출대비|비율|퍼센트|%",
        "해지사유": r"사유|이유|원인|왜",
    }
    for label, pattern in requested_patterns.items():
        claim = by_label.get(label)
        if claim is not None and re.search(pattern, question):
            lines.append(TemplateComposer()._claim_line(claim))
    lines.append("원공시 후보 접수번호: " + ", ".join(origins) + ".")
    # Other limitations remain visible; the two ambiguity sentences above
    # already carry both the boundary and its candidate evidence.
    ambiguity_messages = set(render_safe_limitations(ambiguous))
    notes = [note for note in render_safe_limitations(payload.limitations)
             if note not in ambiguity_messages]
    if notes:
        _extend_public_qualifications(lines, notes)
    return "\n".join(lines)


def _event_condition_and_termination_answer(
        payload, question: str | None) -> str | None:
    """Project only the requested condition and termination clauses.

    Disclosure rows often store a requested condition inside a long advisory
    paragraph.  Replaying the whole paragraph hides the two facts a reader
    asked for.  This projection is enabled only when the question requests
    both axes and both typed claims exist; it copies complete source bullets
    and never paraphrases a missing field.
    """

    if not question:
        return None
    compact = re.sub(r"\s+", "", question)
    asks_condition = "효력발생" in compact
    asks_reason = bool(re.search(
        r"해지(?:된)?(?:이유|사유)|(?:이유|사유).*해지", compact))
    if not (asks_condition and asks_reason):
        return None

    condition_claims = [
        claim for claim in payload.claims
        if "효력발생조건" in (claim.label or "") and claim.text
    ]
    reason_claims = [
        claim for claim in payload.claims
        if "해지사유" in (claim.label or "")
        and (claim.text or claim.value_text)
    ]
    if not condition_claims or not reason_claims:
        return None

    condition = None
    condition_owner = None
    for claim in condition_claims:
        condition = next((
            bullet for bullet in _advisory_bullets(claim.text or "")
            if "효력" in bullet and "발생" in bullet
        ), None)
        if condition:
            condition_owner = claim
            break
    if condition is None or condition_owner is None:
        return None

    reason_owner = reason_claims[-1]
    reason = _display_text(
        reason_owner.text or reason_owner.value_text or "").rstrip(". ")
    lines = [
        ("계약 효력발생 조건: "
         f"{_display_text(condition).rstrip('. ')}.{_cite(condition_owner)}"),
        f"해지 이유: {reason}.{_cite(reason_owner)}",
    ]

    # A later filing may add the verified notice that made the conditional
    # contract ineffective.  It is useful context, but only when a complete
    # source bullet is present; the scalar reason above remains the answer.
    context = None
    context_owner = None
    for claim in reversed(condition_claims):
        context = next((
            bullet for bullet in _advisory_bullets(claim.text or "")
            if "계약무효" in bullet or (
                "무산" in bullet and "통보" in bullet)
        ), None)
        if context:
            context_owner = claim
            break
    if context and context_owner:
        lines.append(
            "해지 경과: "
            f"{_display_text(context).rstrip('. ')}.{_cite(context_owner)}")

    notes = render_safe_limitations(payload.limitations)
    if notes:
        _extend_public_qualifications(lines, notes)
    return "\n".join(lines)


def _same_day_documents_and_final_state(payload) -> str | None:
    """Separate a dated document snapshot from a later corpus-final state."""

    documents = [
        claim for claim in payload.claims
        if claim.label.startswith("문서 ") and claim.state
    ]
    states = [claim for claim in payload.claims if _is_event_summary(claim)]
    document_days = {
        citation.rcept_no[:8]
        for claim in documents
        for citation in (claim.citations or ())
        if citation.rcept_no and re.fullmatch(r"\d{14}", citation.rcept_no)
    }
    if len(documents) < 2 or len(document_days) != 1 or not states:
        return None

    day = next(iter(document_days))
    final_state = max(states, key=_event_state_time_key)
    state_text = _STATE_KO.get(
        final_state.state, final_state.text or final_state.state or "-")
    doc_types = "·".join(dict.fromkeys(str(claim.state) for claim in documents))
    lines = [
        (f"{_public_date(day)}에는 {doc_types}가 모두 "
         f"접수됐고, 제공 코퍼스의 최종 공시상 상태는 {state_text}입니다."),
        "",
        f"{_public_date(day)} 당시 확인된 공시:",
        "",
        "| 공시 유형 | 접수번호 | 근거 |",
        "|---|---|---|",
    ]
    for claim in documents:
        receipt = next((citation.rcept_no for citation in claim.citations
                        if citation.rcept_no), "-")
        evidence = _cite(claim).strip().removeprefix("(근거: ").removesuffix(")")
        lines.append(
            f"| {_markdown_cell(str(claim.state))} | {_markdown_cell(receipt)} | "
            f"{_markdown_cell(evidence or '-')} |"
        )

    lineage = next((
        claim for claim in payload.claims
        if claim.label.endswith("정정 계보")
    ), None)
    if lineage is not None:
        lines.extend(("", TemplateComposer()._claim_line(lineage)))
    dated_prefixes = tuple(claim.output_id for claim in documents)
    material = [
        claim for claim in payload.claims
        if claim.output_id.startswith(dated_prefixes)
        and (claim.label.endswith("정정사유") or claim.label.endswith("계약금액"))
    ]
    if material:
        lines.append("")
        lines.extend(TemplateComposer()._claim_line(claim) for claim in material)

    lines.extend(("", "제공 코퍼스의 최종 상태:",
                  TemplateComposer()._claim_line(final_state)))
    details = [
        claim for claim in _same_termination_receipt_details(
            payload.claims, states)
        if claim.label in {"해지금액", "매출액대비", "해지사유", "해지일자"}
    ]
    lines.extend(TemplateComposer()._claim_line(claim) for claim in details)
    notes = _event_qualification_notes(payload.limitations)
    if notes:
        _extend_public_qualifications(lines, notes)
    return "\n".join(line for line in lines if line or line == "")


def _event_amount_difference_answer(payload, question: str | None) -> str | None:
    """Explain a typed contract-versus-termination amount calculation briefly."""

    if not question or re.search(r"차이|다르|왜|이유", question) is None:
        return None
    differences = [
        claim for claim in payload.claims
        if claim.operator in {"difference", "absolute_difference"}
        and len(claim.derived_from or ()) == 2
    ]
    by_id = {claim.output_id: claim for claim in payload.claims}
    result = next((
        claim for claim in differences
        if all(output_id in by_id for output_id in claim.derived_from)
        and any(by_id[output_id].operator == "correction_diff"
                for output_id in claim.derived_from)
        and any("해지금액" in (by_id[output_id].label or "")
                for output_id in claim.derived_from)
    ), None)
    if result is None:
        return None
    operands = [by_id[output_id] for output_id in result.derived_from]
    verdicts = payload.premise_verdicts
    if re.search(r"다르(?:다면|면)|같은지|같은가|같나요|같습니까", question):
        # A conditional comparison asks whether the amounts agree; it does
        # not assert their equality. Keep the typed audit verdict unchanged.
        verdicts = [verdict for verdict in verdicts if not (
            verdict.verdict == "false"
            and re.fullmatch(r"주장\s+'[^']+'\s+vs\s+검증 결과\s+(?:다르다|다름)",
                             verdict.detail or ""))]
    lines = _premise_lines(verdicts, claims=payload.claims)
    lines.extend((
        f"두 금액은 같지 않고 차이는 {_display_amount(result)}입니다.{_cite(result)}",
        "",
        "| 구분 | 금액 | 근거 |",
        "|---|---:|---|",
    ))
    for claim in operands:
        evidence = _cite(claim).strip().removeprefix("(근거: ").removesuffix(")")
        lines.append(
            f"| {_markdown_cell(_normal_public_label(claim.label))} | "
            f"{_markdown_cell(_display_amount(claim))} | "
            f"{_markdown_cell(evidence or '-')} |"
        )

    context = next((
        claim for claim in payload.claims
        if (claim.operator == "source_explanation"
            or "기타 투자판단" in (claim.label or ""))
        and "해지금액" in (claim.text or "")
    ), None)
    if context is not None:
        # Split only disclosure list markers (가./나./다. ...), not ordinary
        # Korean sentence endings such as ``금액임. 이행금액``.
        clauses = re.split(
            r"(?=\s*[가나다라마바사아자차카타파하]\.\s+)",
            context.text or "",
        )
        reason = next((
            " ".join(clause.split()) for clause in clauses
            if "해지금액" in clause
        ), "")
        # 절 경계로 나눈 clause 는 그 절의 원문 항목 번호(가./나) …)를 그대로
        # 달고 나온다. 라벨이 이미 "차이가 나는 이유"라고 말했으니 그 번호는
        # 형제 항목 없이 홀로 남는 여분이다(이슈 #119, 실례 G-I-006).
        reason = strip_leading_item_number(reason).strip()
        if reason:
            lines.extend(("", f"차이가 나는 이유: {reason}{_cite(context)}"))
    notes = render_safe_limitations(payload.limitations)
    if notes:
        _extend_public_qualifications(lines, notes)
    return "\n".join(lines)


def _event_flow_table(lineage, states, claims=()) -> str:
    """Join an authoritative correction lineage with a final termination."""

    if lineage is None:
        return ""
    receipts = re.findall(r"\b\d{14}\b", lineage.text or "")
    final_state = _final_termination_state(states)
    if len(receipts) < 2 or final_state is None:
        return ""
    termination_receipt = next((
        citation.rcept_no for citation in final_state.citations
        if citation.rcept_no
    ), None)
    post_termination_corrections = set()
    for claim in claims:
        if claim.operator != "correction_diff":
            continue
        text = " ".join((claim.text or "").split())
        if not re.search(r"해지(?:되|됐|사실|공시|되었습니다)", text):
            continue
        post_termination_corrections.update(
            citation.rcept_no for citation in (claim.citations or ())
            if citation.rcept_no in receipts
        )

    rows = []
    for index, receipt in enumerate(receipts):
        rows.append((receipt[:8], "최초 공시" if index == 0 else "정정 공시",
                     receipt))
    if termination_receipt and termination_receipt not in receipts:
        date = _event_state_time_key(final_state) or termination_receipt[:8]
        rows.append((date, "해지 공시", termination_receipt))
    def stage_order(row) -> int:
        _date, stage, receipt = row
        if stage == "최초 공시":
            return 0
        # A correction that explicitly says the contract was terminated and
        # points readers to a termination filing is downstream of that filing,
        # even when both have the same calendar date. Other same-day
        # corrections retain the conservative correction-before-termination
        # order; receipt-number magnitude is never used as time authority.
        if receipt in post_termination_corrections:
            return 2
        if stage == "해지 공시" and any(
                correction[:8] == _date
                for correction in post_termination_corrections):
            return 1
        return 1 if stage == "정정 공시" else 2

    rows.sort(key=lambda row: (row[0], stage_order(row)))
    lines = [
        "공시 흐름:", "",
        "| 날짜 | 단계 | 접수번호 |",
        "|---|---|---|",
    ]
    for date, stage, receipt in rows:
        lines.append(
            f"| {_public_date(date)} | {stage} | 접수번호 {receipt} |"
        )
    return "\n".join(lines)


#: 이슈 #44 — 「투자판단관련주요경영사항」제목·공시일 나열.  `_event_collection_table`
#: 과 별개의 단순 목록이다: 이쪽은 슬롯(상대·금액·사유) 없이 문서 하나당
#: 제목+공시일 두 값만 있는 「find」 task claim을 그대로 다시 묶는다.
_INVESTMENT_JUDGMENT_LIST_REQUEST = re.compile(
    r"투자\s*판단\s*(?:관련)?\s*주요\s*경영\s*사항")
_TITLE_AND_DATE_LIST_REQUEST = re.compile(
    r"제목.{0,10}공시\s*일|공시\s*일.{0,10}제목")
_DOCUMENT_FIND_CLAIM_TEXT = re.compile(
    r"^(?P<label>.+?)\s*\(접수번호\s*(?P<rcept>\d{14}),\s*접수일\s*(?P<date>\d{8})\)$")


def _strip_wrapping_parens(text: str) -> str:
    """Drop one *fully enclosing* parenthesis pair, keeping inner ones intact.

    DART filing titles are commonly recorded as ``폼이름 (본문 제목)``; once
    the form-name prefix is stripped, only the outer wrapper is left, e.g.
    ``(인간 히알루로니다제 원천 기술(ALT-B4) 독점적 라이선스 계약 체결)``.
    Printing that literal outer parenthesis pair around a numbered list item
    reads as an aside, not a title.  Only the pair whose open/close spans the
    entire string is removed — an inner one such as ``(ALT-B4)`` is never
    touched because the running depth never returns to zero before the end.
    """

    if not (text.startswith("(") and text.endswith(")")):
        return text
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text if index != len(text) - 1 else text[1:-1].strip()
    return text


def _investment_judgment_title_date_list(
        payload, question: str | None) -> str | None:
    """Render a 「서식명 + 연도」whole document collection as a title/date list.

    Scoped to ``투자판단관련주요경영사항`` (issue #44).  The collection's
    single ``find`` task already emits one claim per matching document whose
    ``text`` carries the filing title, receipt number and disclosure date —
    this only regroups those claims; it invents no new fact.

    The question already asked for exactly ``제목``·``공시일`` — the numbered
    item itself *is* the title, so a leading ``제목: `` label only repeats
    what the list shape already says (#115).  Two distinct filings can share
    one title and disclosure date (the 2025-03-17 알테오젠 pair, corp
    universe fixture); merge those into a single row naming both receipt
    numbers instead of printing the same title twice.
    """

    if not question:
        return None
    if (_INVESTMENT_JUDGMENT_LIST_REQUEST.search(question) is None
            or _TITLE_AND_DATE_LIST_REQUEST.search(question) is None):
        return None
    grouped: dict[tuple[str, str], list[str]] = {}
    correction_by_key: dict[tuple[str, str], bool] = {}
    order: list[tuple[str, str]] = []
    for claim in payload.claims:
        if claim.derived_from or claim.operator is not None:
            continue
        match = _DOCUMENT_FIND_CLAIM_TEXT.match(claim.text or "")
        if match is None:
            continue
        label = match.group("label").strip()
        is_correction = re.search(r"정정", label) is not None
        title = label
        for prefix in ("정정공시", "해지공시", "투자판단관련주요경영사항"):
            if title.startswith(prefix):
                remainder = title[len(prefix):].strip()
                if remainder:
                    title = remainder
                break
        title = _strip_wrapping_parens(title)
        key = (match.group("date"), title)
        if key not in grouped:
            grouped[key] = []
            correction_by_key[key] = False
            order.append(key)
        grouped[key].append(match.group("rcept"))
        correction_by_key[key] = correction_by_key[key] or is_correction
    if not grouped:
        return None
    order.sort(key=lambda key: key)
    total = sum(len(set(receipts)) for receipts in grouped.values())
    lines = [f"투자판단관련주요경영사항 공시는 {total}건입니다.", ""]
    if total != len(order):
        lines.insert(1, f"제목과 공시일이 같은 공시를 묶어 아래 {len(order)}개 항목으로 정리했습니다.")
    for index, key in enumerate(order, start=1):
        date, title = key
        marker = "[정정] " if correction_by_key[key] else ""
        receipts = ", ".join(dict.fromkeys(grouped[key]))
        lines.append(f"{index}. {marker}{title}")
        lines.append(f" - 공시일: {_public_date(date)} · 접수번호 {receipts}")
    notes = render_safe_limitations(payload.limitations)
    if notes:
        _extend_public_qualifications(lines, notes)
    return "\n".join(lines)


def _event_collection_table(payload, question: str | None) -> str | None:
    """Render a closed event collection as a deterministic public table.

    This route is intentionally narrow: the user must explicitly request a
    table, at least two event roots must each have a typed state summary, and
    every row must contain the same verified counterparty/amount/reason slots.
    Counts, advisory paragraphs and other automatically retrieved context are
    therefore not mistaken for requested table columns.
    """

    if not question:
        return None
    explicit_table = _EXPLICIT_TABLE_REQUEST.search(question) is not None
    explicit_comparison = re.search(r"비교|대조", question) is not None
    requested = {
        slot for slot, pattern in _EVENT_TABLE_QUESTION_SLOTS.items()
        if pattern.search(question) is not None
    }
    required = set(_EVENT_TABLE_QUESTION_SLOTS)
    default_complete_shape = (
        not requested and _EVENT_NONDEFAULT_TABLE_SLOT.search(question) is None
    )
    if not (
        ((explicit_table or explicit_comparison) and required <= requested)
        or ((explicit_table or explicit_comparison) and default_complete_shape)
    ):
        return None

    summaries = {
        claim.output_id: claim
        for claim in payload.claims
        if (claim.state in _STATE_KO
            and claim.operator is None
            and not claim.derived_from)
    }
    grouped: dict[str, dict[str, object]] = {}
    for claim in payload.claims:
        prefix, dot, suffix = claim.output_id.rpartition(".")
        public_slot = _EVENT_TABLE_SLOTS.get(suffix)
        if not dot or public_slot is None or prefix not in summaries:
            continue
        grouped.setdefault(prefix, {})[public_slot] = claim

    event_roots = set(summaries)
    rows = [
        (root, summaries[root], slots)
        for root, slots in grouped.items()
        if required <= slots.keys()
    ]
    # A table is all-or-nothing.  Omitting one partially retrieved event would
    # make a complete-looking table silently narrower than the typed payload.
    if len(event_roots) < 2 or len(rows) != len(event_roots):
        return None

    issuer = next((
        claim for claim in payload.claims
        if claim.label == "공시 주체" and claim.text), None)
    optional_columns = [
        column for column in ("해지일", "최근매출액 대비")
        if all(column in slots for _, _, slots in rows)
    ]
    lead = f"공시상 확인된 해지 사건은 {len(rows)}건입니다."
    if issuer is not None:
        lead += f" 공시 주체는 {issuer.text}입니다.{_cite(issuer)}"
    headers = ["공시 접수번호", "상대방", "금액", *optional_columns,
               "사유", "근거"]
    lines = [
        lead,
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---:" if header in {"금액", "최근매출액 대비"}
                         else "---" for header in headers) + "|",
    ]
    for _, summary, slots in rows:
        counterparty = slots["상대방"]
        amount = slots["금액"]
        reason = slots["사유"]
        event_receipt = next((
            citation.rcept_no for citation in summary.citations
            if citation.rcept_no
        ), "-")
        displayed_claims = [counterparty, amount, reason] + [
            slots[column] for column in optional_columns
        ]
        receipts = list(dict.fromkeys(
            citation.rcept_no
            for claim in displayed_claims
            for citation in claim.citations
            if citation.rcept_no
        ))
        evidence = ", ".join(f"접수번호 {receipt}" for receipt in receipts)
        optional_cells = []
        for column in optional_columns:
            claim = slots[column]
            value = (_display_amount(claim) if claim.value_text
                     else claim.text or claim.state or "-")
            optional_cells.append(_markdown_cell(value))
        lines.append(
            "| " + " | ".join((
                _markdown_cell(event_receipt),
                _markdown_cell(counterparty.text or counterparty.value_text or "-"),
                _markdown_cell(_display_amount(amount)),
                *optional_cells,
                _markdown_cell(reason.text or reason.value_text or "-"),
                _markdown_cell(evidence or "-"),
            )) + " |"
        )

    notes = _event_qualification_notes(payload.limitations)
    if notes:
        lines.append("")
        _extend_public_qualifications(lines, notes)
    return "\n".join(lines)


#: K-008 shape: 「계약상대·계약명·체결일별로 표로」 lists every matching
#: contract event, not only ones proven terminated — ``_event_collection_table``
#: requires 금액/사유 slots and stays scoped to termination collections.
_CONTRACT_LISTING_TABLE_REQUEST = re.compile(r"계약\s*상대")
_CONTRACT_LISTING_TABLE_NAME_COLUMN = re.compile(r"계약\s*명")
_EVENT_ROOT_RECEIPT_LABEL = re.compile(r"^사건\s+(?P<rcept>\d{14})$")


def _contract_event_listing_table(payload, question: str | None) -> str | None:
    """Render a plain contract-event collection as a deterministic table.

    Unlike ``_event_collection_table`` this never requires a termination
    amount/reason slot: every event with 계약상대·계약명 counts, whether or
    not a later disclosure terminated it.  체결일 is not itself a disclosed
    form field — the receipt number's own leading 8 digits are the DART
    접수번호 date, so the column is labelled 체결일(접수일) to keep that
    substitution explicit rather than inventing a field that was never
    verified.
    """

    if (not question
            or _CONTRACT_LISTING_TABLE_REQUEST.search(question) is None
            or _CONTRACT_LISTING_TABLE_NAME_COLUMN.search(question) is None):
        return None
    explicit_table = _EXPLICIT_TABLE_REQUEST.search(question) is not None

    roots: dict[str, str] = {}
    for claim in payload.claims:
        if claim.derived_from or claim.operator is not None:
            continue
        match = _EVENT_ROOT_RECEIPT_LABEL.match(claim.label or "")
        if match is not None:
            roots[claim.output_id] = match.group("rcept")
    # Few records may read more naturally as short prose unless the user
    # explicitly requested a table.  A long event collection is different:
    # repeating ``계약상대`` and ``계약명`` on every line obscures the rows, so
    # use the table's stable header even when the user only said "정리".
    if len(roots) < 2 or (not explicit_table and len(roots) < 8):
        return None

    slots: dict[str, dict[str, object]] = {}
    for claim in payload.claims:
        prefix, dot, suffix = claim.output_id.rpartition(".")
        if dot and prefix in roots and suffix in ("계약상대", "계약명"):
            slots.setdefault(prefix, {})[suffix] = claim

    # A large plain listing (79 real-world contract events) is unevenly
    # disclosed: some events state neither counterparty nor contract name in
    # the corpus at all.  Requiring every one of them, all-or-nothing, would
    # make this route never fire on real data.  A row with both verified
    # slots is still shown for every other event; incomplete ones are named
    # in a count, not dropped without saying so.
    rows: list[tuple[str, str, object, object]] = []
    incomplete = 0
    for output_id, rcept_no in roots.items():
        row_slots = slots.get(output_id, {})
        counterparty = row_slots.get("계약상대")
        contract_name = row_slots.get("계약명")
        if counterparty is None or contract_name is None:
            incomplete += 1
            continue
        rows.append((rcept_no, rcept_no[:8], counterparty, contract_name))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda row: (row[1], row[0]))

    issuer = next((
        claim for claim in payload.claims
        if claim.label == "공시 주체" and claim.text), None)
    subject = f"{issuer.text}의 " if issuer is not None else ""
    lead = (
        f"{subject}계약 공시는 {len(roots)}건이며, 이 중 계약상대·계약명이 "
        f"모두 확인된 {len(rows)}건을 표로 정리합니다."
        if incomplete else
        f"{subject}계약 공시는 {len(rows)}건입니다.")
    lines = [
        lead, "",
        "| 계약상대 | 계약명 | 체결일(접수일) | 접수번호 |",
        "|---|---|---|---|",
    ]
    for rcept_no, date, counterparty, contract_name in rows:
        lines.append("| " + " | ".join((
            _markdown_cell(counterparty.text or counterparty.value_text or "-"),
            _markdown_cell(contract_name.text or contract_name.value_text or "-"),
            _markdown_cell(_public_date(date)),
            _markdown_cell(rcept_no),
        )) + " |")
    notes = render_safe_limitations(payload.limitations)
    if notes:
        lines.append("")
        _extend_public_qualifications(lines, notes)
    return "\n".join(lines)


_CORRECTION_DATE = re.compile(
    r"(?:계약 식별 접수번호 (?P<contract>\d{14}) · )?"
    r"(?P<date>\d{4}-\d{2}-\d{2}) 정정\s*(?P<field>.+)$"
)


#: 정정 diff 본문의 기계 숫자 — ``6,554,463,847,260 → 3,278,064,777,929 (차이 …)``.
#: 표시 문자열이 아니라 이것을 읽어야 총 변화를 셈할 수 있다.
_CORRECTION_NUMERIC_CHANGE = re.compile(
    r"(?<![\d,])(?P<before>\d[\d,]*)\s*→\s*(?P<after>\d[\d,]*)(?![\d,])")

#: 결론으로 앞세울 금액 항목. 정정이 실제로 되풀이되는 자리다.
_CORRECTION_AMOUNT_FIELD = re.compile(r"^계약금액$|^해지금액$")


def _correction_amount_summary(claims) -> "str | None":
    """정정 이력의 **결론** — 최초값 → 최종값 → 총 변화. 없으면 ``None``.

    이슈 #94 24. 정정이 열다섯 번이면 아래 표에 열다섯 줄이 쌓이는데, 정작
    사용자가 알고 싶은 것은 「그래서 얼마에서 얼마가 됐나」다. 그것이 표
    첫 줄과 마지막 줄에 흩어져 있으면 핵심 변화가 묻힌다.

    표에 이미 있는 값을 다시 셈하는 것이라 **새로 조회하지 않는다.** 기계
    숫자(`claim.text`)를 읽으므로 표시 문자열을 되파싱하지도 않는다.

    **한 계약일 때만 셈한다.** 원공시가 여럿이면 「최초」와 「최종」이 어느
    계약의 것인지 알 수 없어, 합치면 없는 변화를 만들어 낸다.

    정정이 한 번뿐이면 붙이지 않는다 — 표의 그 한 줄이 이미 같은 말이다.
    """

    rows: list[tuple[str, str, int, int]] = []
    contracts: set[str] = set()
    seen_changes: set[tuple] = set()
    for claim in claims:
        if claim.operator != "correction_diff":
            continue
        matched = _CORRECTION_DATE.search(claim.label or "")
        if matched is None:
            continue
        field = _public_correction_field(matched.group("field").strip())
        if _CORRECTION_AMOUNT_FIELD.match(field) is None:
            continue
        change = _CORRECTION_NUMERIC_CHANGE.search(str(claim.text or ""))
        if change is None:
            continue
        try:
            before = int(change.group("before").replace(",", ""))
            after = int(change.group("after").replace(",", ""))
        except ValueError:
            continue
        if before == after:
            continue
        contracts.add(matched.group("contract") or "")
        # Snapshot and lineage tasks may emit the same correction twice.
        # Receipt identity keeps distinct same-day corrections distinct.
        receipts = tuple(sorted({c.rcept_no for c in claim.citations}))
        key = (matched.group("contract") or "", matched.group("date"),
               receipts, field, before, after)
        if key in seen_changes:
            continue
        seen_changes.add(key)
        rows.append((matched.group("date"), field, before, after))
    if len(rows) < 2 or len(contracts) != 1:
        return None
    if len({field for _date, field, _b, _a in rows}) != 1:
        return None

    rows.sort(key=lambda row: row[0])
    field = rows[0][1]
    first, last = rows[0][2], rows[-1][3]
    total = last - first
    opening = format_won_exact(str(first))
    closing = format_won_exact(str(last))
    moved = format_won_exact(str(total), absolute=True)
    if not opening or not closing or not moved:
        return None
    # 「정정 N회」라고만 쓰면 전체 정정 횟수로 읽힌다. 실제로는 **이 항목이**
    # 바뀐 횟수다 — 같은 정정에서 다른 항목만 바뀐 회차도 있다.
    if total == 0:
        # 여러 번 고쳐 제자리로 돌아온 것도 결론이다.
        return (f"{field}: 최초 {opening} → 최종 {closing} "
                f"({len(rows)}회 변경, 최초값과 같음)")
    direction = "증가" if total > 0 else "감소"
    return (f"{field}: 최초 {opening} → 최종 {closing} "
            f"({len(rows)}회 변경, 총 {moved} {direction})")


def _requested_correction_fields(question: str | None) -> set[str]:
    """Return only correction axes the user explicitly named."""

    if not question or re.search(r"전체|모든", question):
        return set()
    fields = set()
    if re.search(r"계약\s*금액", question):
        fields.add("contract_amount")
    if re.search(
            r"계약\s*기간|시작일|종료일|"
            r"계약\s*금액\s*(?:[·ㆍ/,]|및|과|와)\s*기간", question):
        fields.add("contract_period")
    # A named merger ratio is a different axis from the contract/sales ratio.
    # Remove that phrase before resolving an otherwise generic ratio request;
    # a question may still explicitly request both kinds of ratios.
    ratio_question = re.sub(r"(?:분할)?합병\s*비율", "", question)
    if ratio_question != question:
        fields.add("merger_ratio")
    if re.search(r"매출액\s*대비|매출\s*대비|비율|퍼센트|%", ratio_question):
        fields.add("sales_ratio")
    if re.search(r"공시유보|유보사유|유보기한", question):
        fields.add("withholding")
    return fields


def _correction_field_category(field: str) -> str | None:
    compact = re.sub(r"\s+", "", field)
    if "계약금액" in compact:
        return "contract_amount"
    if any(marker in compact for marker in ("계약기간", "시작일", "종료일")):
        return "contract_period"
    if "매출액대비" in compact:
        return "sales_ratio"
    if "합병비율" in compact:
        return "merger_ratio"
    if "공시유보" in compact or "유보사유" in compact or "유보기한" in compact:
        return "withholding"
    return None


def _correction_history_table(
        claims, *, requested_fields: set[str] | None = None,
        ) -> str:
    """Render one cited correction row per typed transition.

    Event tools emit both a document snapshot and a ``correction_diff`` sidecar
    for the same fact.  Printing both was accurate but made a short status
    question turn into thousands of characters of duplicated disclosure text.
    The sidecar is the authoritative before/after transition, so this compact
    table keeps every such transition once and deliberately leaves snapshot
    fields and advisory paragraphs out of the history section.
    """

    correction_claims = [
        claim for claim in claims if claim.operator == "correction_diff"
    ]
    # A multi-task plan can contain the same transition once under the event
    # snapshot and again under the authoritative lineage sidecar.  When the
    # latter carries an explicit root receipt, prefer it wholesale instead of
    # showing two almost identical blocks with one root marked unknown.
    rooted = [claim for claim in correction_claims
              if re.search(r"계약 식별 접수번호 \d{14}", claim.label or "")]
    if rooted:
        correction_claims = rooted
    scalar_exists = any(not _is_narrative_correction(claim)
                        for claim in correction_claims)
    rows = []
    seen = set()
    for claim in correction_claims:
        if scalar_exists and _is_narrative_correction(claim):
            continue
        if _unchanged_numeric_correction(claim):
            continue
        match = _CORRECTION_DATE.search(claim.label or "")
        receipt_day = next((
            citation.rcept_no[:8]
            for citation in (claim.citations or ())
            if citation.rcept_no and re.fullmatch(r"\d{14}", citation.rcept_no)
        ), None)
        contract = match.group("contract") if match else None
        date = (match.group("date") if match else
                f"{receipt_day[:4]}-{receipt_day[4:6]}-{receipt_day[6:]}"
                if receipt_day else "-")
        field = ("정정 " + match.group("field")
                 if match else _normal_public_label(claim.label))
        field = _public_correction_field(field)
        if requested_fields and _correction_field_category(field) not in requested_fields:
            continue
        change = _public_correction_change(
            claim,
            _compact_correction_change(
                _display_text(claim.text or claim.value_text or "-")),
        )
        key = (contract, date, field, change)
        if key in seen:
            continue
        seen.add(key)
        rows.append((contract, date, field, change, _cite(claim)))

    if not rows:
        return ""
    contract_receipts = list(dict.fromkeys(
        match.group(1)
        for claim in claims
        if claim.operator == "correction_diff"
        for match in [re.search(r"계약 식별 접수번호 (\d{14})", claim.label or "")]
        if match
    ))
    # 바로 위에 「공시 정정 이력: A → B → …」 줄이 있다. 같은 말을 두 번
    # 머리글로 쓰면 무엇이 다른 목록인지 알 수 없다.
    lines = ["정정별 변경 내용:", ""]
    if contract_receipts:
        subject = _correction_subject(correction_claims)
        identifier = ("공시 식별 접수번호" if subject == "공시"
                      else "계약 식별 공시")
        lines.extend((
            f"| {identifier} | 정정일 | 변경 항목 | 변경 내용 | 근거 |",
            "|---|---|---|---|---|",
        ))
    else:
        lines.extend((
            "| 정정일 | 변경 항목 | 변경 내용 | 근거 |",
            "|---|---|---|---|",
        ))
    for contract, date, field, change, citation in rows:
        # ``_cite`` already contains a user-safe receipt number.  Trim only
        # its surrounding prose for a compact Markdown cell.
        evidence = citation.strip().removeprefix("(근거: ").removesuffix(")") or "-"
        cells = [date, field, change, evidence]
        if contract_receipts:
            cells.insert(0, contract or "확인되지 않음")
        lines.append("| " + " | ".join(
            _markdown_cell(cell) for cell in cells) + " |")
    return "\n".join(lines)


def _correction_subject(claims) -> str:
    """Use document provenance, not the inherited event label, for headings."""

    kinds = {citation.doc_id.split("_", 1)[0]
             for claim in claims for citation in claim.citations}
    return "공시" if kinds == {"major"} else "계약"


def _unchanged_numeric_correction(claim) -> bool:
    """Ignore unchanged numeric cells replayed inside a correction table."""

    value = re.sub(r"^\[[^\]]+\]\s*", "", claim.text or "").strip()
    number = r"[+-]?\d[\d,]*(?:\.\d+)?"
    match = re.fullmatch(
        rf"(?P<before>{number})\s*(?P<unit>%|원)?\s*→\s*"
        rf"(?P<after>{number})\s*(?P=unit)?"
        rf"(?:\s*\(차이\s+{number}(?:%p|%)?\))?"
        r"(?:\s*\[사유:[^\]]*\])?", value)
    if match is None:
        return False
    return (Decimal(match.group("before").replace(",", ""))
            == Decimal(match.group("after").replace(",", "")))


def _public_correction_field(field: str) -> str:
    """Shorten disclosure-path labels to the field a reader recognizes."""

    mappings = (
        ("계약금액", "계약금액"),
        ("매출액대비", "최근매출액 대비"),
        ("유보사유", "공시유보 사유"),
        ("유보기한", "공시유보 기한"),
    )
    for marker, display in mappings:
        if marker in field:
            return display
    return _normal_public_label(field)


#: 정정 전후 문장 안의 원 단위 금액(억 이상). 비율·연도·접수번호는 자릿수가
#: 모자라 걸리지 않는다.
_CORRECTION_WON = re.compile(r"(?<![\d,.])\d{1,3}(?:,\d{3}){2,}(?![\d,.])")
#: 단위가 선언되지 않아도 원 금액임이 이름으로 확정되는 공시 항목.
_CORRECTION_MONEY_LABEL = re.compile(
    r"계약금액|해지금액|공급금액|수주금액|발행총액|취득금액|처분금액|"
    r"자금조달|투자금액|매출액(?!\s*대비)")


def _public_correction_amounts(claim, value: str) -> str:
    """정정 전후 금액을 답변의 다른 문장과 같은 한국식 단위로 적는다.

    같은 답변이 본문에서는 ``계약금액은 2조 453억원입니다`` 라고 쓰면서 정정
    이력 표에서는 ``793,841,180,000 → 1,100,000,000,000`` 을 그대로 실었다.
    한 답변 안에서 같은 종류의 금액이 두 표기로 나오면 독자는 둘을 견줄 수 없다.

    ``_correction_scalar_projection`` 이 이미 이 목적으로 typed 단위를 붙여
    두었다(그 docstring: 「``72,200,000,000원`` 을 ``722억원`` 으로 보여주기
    위해」). 정정 표만 그 필드를 쓰지 않고 원문 문자열을 그대로 냈다.

    단위가 원으로 확정된 claim 에서만 바꾼다. 단위를 모르면 그 숫자가 금액인지
    수량인지 알 수 없으므로 손대지 않는다.
    """

    unit = normalize_source_money_unit(getattr(claim, "raw_unit", None) or "")
    if unit not in {"", "원"}:
        # 백만원·억원 표는 선언 단위가 따로 있다. 그 표면을 원으로 펴면 표가
        # 밝힌 단위와 어긋난다.
        return value
    if not unit and _CORRECTION_MONEY_LABEL.search(claim.label or "") is None:
        # 단위가 선언되지 않았고 라벨도 금액임을 증명하지 못하면, 그 숫자가
        # 금액인지 수량인지 알 수 없으므로 손대지 않는다.
        return value

    def swap(match: re.Match[str]) -> str:
        won = format_won_exact(match.group(0).replace(",", ""))
        return won or match.group(0)

    value = _CORRECTION_WON.sub(swap, value)
    # Small differences do not match the large-amount formatter above, but
    # their unit is equally known from the same typed monetary claim.
    return re.sub(
        r"(차이\s+)([+-]?\d[\d,]*)(?=\s*\))",
        lambda match: match.group(1) + (
            format_won_exact(match.group(2).replace(",", ""))
            or match.group(2)), value)


def _public_correction_change(claim, change: str) -> str:
    """Remove repeated correction boilerplate and explain withheld values."""

    value = re.sub(r"^\[[^\]]+\]\s*", "", change).strip()
    label = claim.label or ""
    if claim.state == "disclosed_from_withheld":
        after = _display_amount(claim)
        # A malformed/legacy claim must never make a verified after-value
        # disappear from the public answer.  Typed amount metadata is required
        # for the friendly wording; otherwise preserve the raw correction.
        if not after:
            return value
        if claim.raw_unit == "%" and not after.endswith("%"):
            after += "%"
        return f"비공개 → {after}"
    if "계약금액" in label:
        # 일부 거래소 정정표는 계약금액과 매출액 대비 비율을 한 셀 문자열로
        # 직렬화한다: ``금액 비율 → 금액 비율``. 계약금액 행에서는 금액 쌍만
        # 보여 주고, 비율은 별도 typed 행이 요청됐을 때 표시한다.
        mixed = re.search(
            r"(?P<before>\d{1,3}(?:,\d{3}){2,})\s+"
            r"\d+(?:\.\d+)?\s*→\s*"
            r"(?P<after>\d{1,3}(?:,\d{3}){2,})\s+"
            r"\d+(?:\.\d+)?",
            value,
        )
        if mixed is not None:
            before = (format_won_exact(
                mixed.group("before").replace(",", ""))
                or mixed.group("before"))
            after = (format_won_exact(
                mixed.group("after").replace(",", ""))
                or mixed.group("after"))
            value = (
                value[:mixed.start()]
                + f"{before} → {after}"
                + value[mixed.end():]
            )
    value = _public_correction_amounts(claim, value)
    if (claim.raw_unit in {None, "", "%"}
            and (claim.raw_unit == "%"
                 or _correction_field_category(label) == "sales_ratio")):
        ratio = re.match(
            r"(?P<before>[+-]?\d+(?:\.\d+)?)%?\s*→\s*"
            r"(?P<after>[+-]?\d+(?:\.\d+)?)%?(?=\s|$)", value)
        if ratio:
            tail = re.sub(
                r"(차이\s+)([+-]?\d+(?:\.\d+)?)(?:%p|%)?(?=\s*\))",
                r"\1\2%p", value[ratio.end():])
            value = f"{ratio.group('before')}% → {ratio.group('after')}%{tail}"
    if _correction_field_category(label) == "contract_period":
        value = re.sub(r"^-\s*→", "- (원문에 날짜 미기재) →", value)
        value = re.sub(r"(→\s*)-(?=\s*(?:$|\[))",
                       r"\1- (원문에 날짜 미기재)", value)
    if "유보사유" in label or "유보기한" in label:
        released = re.sub(
            r"\s*→\s*-\s*(?:\[사유:[^\]]+\])?\s*$",
            " → 유보 해제",
            value,
        )
        if released != value:
            return released
    return value


def _is_narrative_correction(claim) -> bool:
    """Whether a correction row is a long prose field rather than a scalar."""

    label = claim.label or ""
    return any(token in label for token in (
        "기타 투자판단", "기타사항", "관련 중요사항", "참고사항",
    ))


#: 「이 공시가 어느 원계약에 속하는지 확정할 수 있는가」를 묻는 표면.
_ASKS_ATTRIBUTION = re.compile(
    r"(?:어느|어떤)[^?\n]{0,40}(?:계보|원계약|원공시)[^?\n]{0,40}"
    r"(?:속하|해당|확정|특정)|"
    r"(?:하나로|단일로)\s*(?:확정|특정)")


def _attribution_conclusion(payload, question: str | None) -> str | None:
    """귀속을 확정할 수 있는지 물으면 그 답을 결론으로 세운다.

    이런 질문의 답은 「확정할 수 있다/없다」이지 「무엇이 바뀌었다」가 아니다.
    정정 이력은 근거일 뿐이다. 답이 정정 표 뒤에 한계 문구로만 붙으면 독자는
    1,800자를 읽고 나서야 자기 질문의 답을 만난다 — 응답품질감사의 「결론이
    먼저 나오고 질문에 직접 답하는가」 항목이다.
    """

    if not question or _ASKS_ATTRIBUTION.search(question) is None:
        return None
    codes = {getattr(limitation, "code", "")
             for limitation in getattr(payload, "limitations", ())}
    if "ambiguous_event_origin" not in codes:
        return None
    return ("코퍼스만으로는 하나로 확정할 수 없습니다. "
            "같은 상대방의 원공시가 복수여서 이 정정이 어느 원계약에 "
            "속하는지는 공시만으로 특정되지 않습니다.")


def _correction_conclusion(claims, *, reject_increase: bool = False) -> str | None:
    """Lead with the material typed correction instead of its row count."""

    changes = [claim for claim in claims
               if claim.operator == "correction_diff"]
    contract_ids = {
        match.group("contract") for claim in changes
        for match in [_CORRECTION_DATE.search(claim.label or "")]
        if match and match.group("contract")
    }
    if len(contract_ids) > 1:
        subject = _correction_subject(changes)
        return f"여러 {subject}의 정정 내용입니다. {subject}별 변경 사항은 아래 표에 정리했습니다."
    disclosed_amount = next((
        claim for claim in changes
        if claim.state == "disclosed_from_withheld"
        and "계약금액" in (claim.label or "")
        and claim.value_text
    ), None)
    if disclosed_amount is not None:
        correction = (
            "계약금액이 늘어난 것이 아니라 " if reject_increase else ""
        )
        amount = _display_amount(disclosed_amount)
        return (
            f"{correction}공시유보로 "
            f"비공개였던 금액이 {amount}{_josa(amount, '으로', '로')} "
            f"공개된 정정입니다.{_cite(disclosed_amount)}"
        )
    if changes:
        dates = list(dict.fromkeys(
            match.group("date")
            for claim in changes
            for match in [_CORRECTION_DATE.search(claim.label or "")]
            if match
        ))
        when = f"{dates[-1]} 정정에서 " if dates else "정정에서 "
        return f"{when}공시 내용이 변경됐습니다."
    return None


def _narrative_correction_summary(claims) -> str | None:
    """Describe a verified prose-row change without replaying both documents."""

    narratives = [claim for claim in claims
                  if claim.operator == "correction_diff"
                  and _is_narrative_correction(claim)]
    if not narratives:
        return None
    rooted = [
        claim for claim in narratives
        if re.search(r"계약 식별 접수번호 \d{14}", claim.label or "")
    ]
    if rooted:
        narratives = rooted

    rendered = []
    seen = set()
    for claim in narratives:
        text = _display_text(claim.text or claim.value_text or "")
        receipt = next((
            citation.rcept_no for citation in (claim.citations or ())
            if citation.rcept_no and re.fullmatch(r"\d{14}", citation.rcept_no)
        ), None)
        date = (
            f"{receipt[:4]}-{receipt[4:6]}-{receipt[6:8]} "
            if receipt else ""
        )
        # A prose row with no verified before→after arrow carries no span to
        # compact: replaying it puts the whole source paragraph in the answer
        # and the reader still has to find what changed.  Name the substance
        # instead. Termination is the one substance a reader must not miss.
        if " → " not in text and len(text) > 320:
            body = ("기타 투자판단 사항에도 계약 해지 관련 내용이 반영됐습니다."
                    if "해지" in text
                    else "기타 투자판단 사항도 함께 정정됐습니다.")
            key = (receipt, body)
            if key not in seen:
                seen.add(key)
                rendered.append(f"{date}{body}{_cite(claim)}")
            continue
        structural = re.search(
            r"\[구조 변경:\s*(?P<summary>.+?)\]\s*$", text)
        if structural is not None:
            summary = structural.group("summary").strip().rstrip(". ")
            body = f"기타 투자판단 사항 구조 변경: {summary}."
        else:
            # The reason has its own receipt-bound line above.  Removing only
            # that typed wrapper keeps the before/after source text readable.
            change_source = re.sub(r"^\[[^\]]+\]\s*", "", text)
            change_source = re.sub(
                r"\s*\[사유:[^\]]+\]\s*$", "", change_source)
            after_bullets = _correction_after_bullets(change_source)
            compacted = _compact_correction_change(change_source)
            # The after side often repeats the whole advisory paragraph that
            # the before side already carried, so a single "정정 후 확인되는
            # 내용" bullet can replay prose the correction table above has
            # already compacted to its changed span.  Keep the bullet only
            # when it is the shorter reading; otherwise show the changed span.
            if after_bullets and len(after_bullets) > len(compacted):
                after_bullets = ""
            change = (after_bullets or compacted).strip().rstrip(". ")
            body = (
                f"기타 투자판단 사항 변경: {change}." if change
                else "기타 투자판단 사항도 함께 정정됐습니다."
            )
        key = (receipt, body)
        if key in seen:
            continue
        seen.add(key)
        rendered.append(f"{date}{body}{_cite(claim)}")
    return "\n".join(rendered) if rendered else None


#: 불릿 안의 수량·금액. 자릿점은 포함하고 소수는 끝까지 하나로 읽는다.
_ADVISORY_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numeric_change_marked(before: str, after: str) -> "str | None":
    """숫자만 달라진 두 불릿을 바뀐 자리마다 ``정정 전 → 정정 후``로 겹쳐 적는다.

    산문 유의사항의 정정은 문장 구조를 그대로 두고 수량 몇 개만 고치는 일이
    많다(`EG-010`: 삼성전자 자기주식취득 정정 — 「① 취득 신고 주식수의 10%」의
    보통주가 5,014,463주에서 5,014,462주로, 「③ 발행주식총수의 1%」가
    59,697,826주에서 59,697,825주로 바뀌었다).  정정 후 불릿만 보여주면 독자는
    5,014,462 라는 **결과**만 보고 무엇이 정정됐는지 알 수 없다 — 정정공시
    질문이 정확히 그것을 묻는데도 그렇다.

    두 문단을 다 싣지 않는다(`_correction_after_bullets` 가 그것을 피하려고
    만들어졌다).  숫자를 지운 골격이 완전히 같을 때만, 달라진 숫자 자리에만
    화살표를 끼워 넣는다.  라벨은 이미 그 숫자 바로 앞에 있으므로 어느 값이
    무엇에서 무엇으로 바뀌었는지가 모호해지지 않는다.  골격이 다르면 문장이
    새로 쓰인 것이므로 겹쳐 적지 않고 ``None`` 을 돌려준다.
    """

    if _ADVISORY_NUMBER.sub("#", before) != _ADVISORY_NUMBER.sub("#", after):
        return None
    olds = _ADVISORY_NUMBER.findall(before)
    if olds == _ADVISORY_NUMBER.findall(after):
        return None
    index = 0

    def _mark(match: "re.Match[str]") -> str:
        nonlocal index
        old = olds[index]
        index += 1
        return match.group(0) if old == match.group(0) else f"{old} → {match.group(0)}"

    return _ADVISORY_NUMBER.sub(_mark, after)


def _correction_after_bullets(change: str) -> str:
    """Summarize a long advisory correction with newly stated bullets.

    Scalar before/after values already live in the correction table. For a
    prose advisory, repeating both full paragraphs hides the actual update.
    When the verified row has one arrow and explicit list bullets on both
    sides, show only complete after-side bullets that were not present before.

    같은 불릿이 숫자만 바뀐 것이라면 그 자리에 ``정정 전 → 정정 후`` 를 끼워
    넣는다(`_numeric_change_marked`) — 정정 후 값만으로는 무엇이 바뀌었는지
    읽을 수 없기 때문이다.
    """

    if change.count(" → ") != 1:
        return ""
    before, after = change.split(" → ", 1)
    before_bullets = _advisory_bullets(before)
    after_bullets = _advisory_bullets(after)
    if not before_bullets or not after_bullets:
        return ""
    before_keys = {re.sub(r"\s+", " ", value).strip()
                   for value in before_bullets}
    prior_by_skeleton: dict[str, str] = {}
    for value in before_bullets:
        collapsed = " ".join(value.split())
        prior_by_skeleton.setdefault(
            _ADVISORY_NUMBER.sub("#", collapsed), collapsed)
    added: list[str] = []
    for value in after_bullets:
        collapsed = " ".join(value.split())
        if collapsed in before_keys:
            continue
        prior = prior_by_skeleton.get(_ADVISORY_NUMBER.sub("#", collapsed))
        marked = _numeric_change_marked(prior, collapsed) if prior else None
        added.append(marked or collapsed)
    if not added or len(added) > 4:
        return ""
    # 원문이 내용 없이 ``-`` 만 적은 자리를 값처럼 옮기면 「정정 후 확인되는
    # 내용: -.」가 되어 무엇이 바뀌었는지 읽을 수 없다. 이 서비스는 0·대시·
    # 공란·미공시·공시유보를 서로 다른 상태로 다루므로, 대시는 값이 아니라
    # 「원문에 내용이 기재되지 않음」으로 밝힌다 (이슈 #249 DEV-EVT-013).
    stated = [value for value in added if not _IS_BLANK_ADVISORY.fullmatch(value)]
    if not stated:
        return "정정 후 기타 투자판단 사항은 원문에 내용 없이 「-」로 기재됐습니다"
    if len(stated) != len(added):
        return ("정정 후 확인되는 내용: " + " / ".join(stated)
                + " (나머지 항목은 원문에 내용 없이 「-」로 기재됐습니다)")
    return "정정 후 확인되는 내용: " + " / ".join(added)


#: 원문이 값 자리에 내용 없이 적어 두는 표시. 값이 아니라 상태다.
_IS_BLANK_ADVISORY = re.compile(r"[-–—ー・.\s]*")


def _compact_correction_change(change: str, *, limit: int = 320) -> str:
    """Collapse duplicated long before/after prose to its changed spans.

    Correction forms often repeat a complete advisory paragraph on both sides
    of ``before → after`` even when only three reported quantities changed.
    Replaying both paragraphs is evidence-preserving but makes the actual
    correction nearly impossible to find.  This display-only projection uses
    the already verified before/after text, keeps every non-equal span with
    local context, and never computes or invents a value.

    Short transitions remain byte-for-byte unchanged.  A typed ``본문 변경
    N곳`` value may contain one verified arrow per changed span; compact those
    spans independently instead of treating the second arrow as ambiguity.
    """

    if len(change) <= limit:
        return change

    def compact_one(value: str, *, context: int = 30) -> list[str] | None:
        if value.count(" → ") != 1:
            return None
        before, after = value.split(" → ", 1)
        opcodes = [
            opcode for opcode in SequenceMatcher(
                None, before, after, autojunk=False).get_opcodes()
            if opcode[0] != "equal"
        ]
        if not opcodes:
            return None

        # Nearby character edits belong to one semantic value.  Joining them
        # avoids output such as separate changes for the comma and final digit
        # of one formatted quantity.
        groups: list[list[int]] = []
        for _tag, i1, i2, j1, j2 in opcodes:
            if (groups and i1 - groups[-1][1] <= 24
                    and j1 - groups[-1][3] <= 24):
                groups[-1][1] = i2
                groups[-1][3] = j2
            else:
                groups.append([i1, i2, j1, j2])

        def excerpt(text: str, start: int, end: int) -> str:
            left = max(0, start - context)
            right = min(len(text), end + context)
            surface = " ".join(text[left:right].split())
            return (("…" if left else "") + surface
                    + ("…" if right < len(text) else ""))

        return [
            f"{excerpt(before, i1, i2)} → {excerpt(after, j1, j2)}"
            for i1, i2, j1, j2 in groups
        ]

    marker = re.match(r"^(\[[^\]]{1,40}\]\s*)", change)
    prefix = marker.group(1) if marker is not None else ""
    body = change[marker.end():] if marker is not None else change

    structured = re.match(r"^본문 변경 (?P<count>\d+)곳:\s*(?P<body>.*)$", body)
    if structured is not None:
        count = int(structured.group("count"))
        parts = structured.group("body").split("; ", max(0, count - 1))
        if len(parts) == count:
            compacted: list[str] = []
            for part in parts:
                spans = compact_one(part, context=20)
                if spans is None:
                    return change
                compacted.append(" / ".join(spans))
            return f"{prefix}본문 변경 {count}곳: " + "; ".join(compacted)

    spans = compact_one(body)
    if spans is None:
        return change
    return f"{prefix}본문 변경 {len(spans)}곳: " + "; ".join(spans)


#: 줄 끝에 붙는 인용 표기.
_CITATION_TAIL = re.compile(r"\s*\((?:근거[:：]\s*)?접수번호[^)]*\)\s*$")

#: 접수번호 14자리.
_RECEIPT_NO = re.compile(r"\d{14}")


def _demote_citations_to_basis(lines: list[str], start: int, end: int) -> None:
    """계산 결과 문장의 접수번호를 지우고 "계산 근거:" 줄 한 곳에만 남긴다.

    파생 계산에서는 결론 문장과 그 아래 계산 근거 줄이 같은 접수번호를 싣는다.
    같은 답변 안에서 같은 근거를 두 번 읽게 되고, 정작 그 번호가 어느 값에서
    나왔는지는 피연산자가 나열된 계산 근거 줄에서만 알 수 있다. 근거 줄이
    결론의 접수번호를 모두 담고 있을 때에 한해 결론 쪽을 지운다 — 담지 못하면
    지울 경우 근거가 사라지므로 그대로 둔다.
    """

    basis = lines[-1] if lines else ""
    if not basis.startswith("계산 근거:"):
        return
    covered = set(_RECEIPT_NO.findall(basis))
    if not covered:
        return
    for i in range(start, min(end, len(lines))):
        tail = _CITATION_TAIL.search(lines[i])
        if tail and set(_RECEIPT_NO.findall(tail.group(0))) <= covered:
            lines[i] = lines[i][:tail.start()].rstrip()


#: 표의 마지막 열이 근거일 때 그 값을 알아보는 표기.
_TABLE_RECEIPT_CELL = re.compile(r"^접수번호\s*\d{14}$")


#: 문장 끝에 붙는 인라인 근거. 출처를 하단에 모을 때 걷어낸다.
_INLINE_CITATION = re.compile(r"\s*\(근거: [^)]*\)")

#: 라벨 안의 연도 하나. 전년 대비 맥락에서 **같은 기준**인지 가릴 때 쓴다.
_LABEL_YEAR = re.compile(r"(?<![0-9])((?:19|20)[0-9]{2})년")


def _year_over_year_context(base, derived) -> "str | None":
    """이미 확인된 전년 값이 있으면 변화를 한 줄 덧붙인다. 없으면 ``None``.

    이슈 #94 27 — 「매출액은 ○○원입니다」만 있으면 그 값이 좋아진 것인지
    나빠진 것인지 알 수 없다.

    **새로 조회하지 않는다.** 이 답변이 이미 근거와 함께 확정한 값들만 쓴다.
    전년 값을 여기서 끌어오면 검증되지 않은 숫자가 답에 들어간다.

    **같은 기준일 때만 견준다.** 라벨에서 연도만 빼고 나머지가 글자까지 같아야
    한다 — 「연결 매출액」과 「별도 매출액」, 「매출액」과 「영업이익」은 나란히
    놓을 수 없다. 단위도 같아야 한다.

    **부호가 바뀌면 말하지 않는다.** 적자에서 흑자로(또는 그 반대로) 넘어간
    변화를 「증가」·「감소」로 부르면 정반대로 읽힌다. 그 말은 적자·흑자
    전환인데, 그것은 이 줄이 아니라 파생 계산이 할 말이다.

    이미 difference·percent_change·absolute_difference 파생이 있으면
    손대지 않는다. difference·percent_change 는 `_period_change_sentence`
    가 그 말을 더 정확히 한다. `absolute_difference`(부호 없는 절대
    차이)는 애초에 방향을 말하지 않으려고 고른 연산인데, 이 함수가 같은
    원자료에서 독자적으로 증감을 계산해 붙이면 그 뜻을 무너뜨린다 —
    실측: `test_signed_time_series_..._absolute_gap_does_not` 가 이 자리
    에서 걸렸다.
    """

    if any(getattr(row, "operator", None)
           in ("difference", "percent_change", "absolute_difference", "sum")
           for row in derived or ()):
        return None

    groups: dict[str, list[tuple[int, object]]] = {}
    for claim in base or ():
        label = str(getattr(claim, "label", "") or "")
        years = _LABEL_YEAR.findall(label)
        if len(years) != 1:
            continue
        if not claim.canonical_value or not claim.canonical_unit:
            continue
        key = f"{_LABEL_YEAR.sub('〈년〉', label)}|{claim.canonical_unit}"
        groups.setdefault(key, []).append((int(years[0]), claim))

    for rows in groups.values():
        if len(rows) != 2:
            continue
        rows.sort(key=lambda row: row[0])
        (earlier_year, earlier), (later_year, later) = rows
        if later_year - earlier_year != 1:
            continue
        try:
            before = Decimal(str(earlier.canonical_value))
            after = Decimal(str(later.canonical_value))
        except (InvalidOperation, TypeError, ValueError):
            continue
        if not before or before < 0 or after < 0:
            # 0 으로 나눌 수 없고, 부호가 걸린 값은 증감으로 부를 수 없다.
            continue
        if after == before:
            continue
        rate = (after - before) / before * 100
        direction = "증가" if after > before else "감소"
        percent = abs(rate).quantize(Decimal("0.1"))
        subject = _normal_public_label(str(later.label or ""))
        return (f"{subject}{_josa(subject, '은', '는')} 전년 대비 "
                f"{percent}% {direction}했습니다.")
    return None


def _source_footer(payload) -> "str | None":
    """출처를 답변 하단에 모은다. 실을 것이 없으면 ``None``.

    이슈 #94 30 — 접수번호는 어느 **문서**인지만 말한다. 그 값이 보고서
    어디에 있는지는 정본이 이미 알고 있고(`ClaimCitation.report_name` ·
    `source_path`), 그것을 알려 줘야 사용자가 원문에서 직접 확인할 수 있다.

    ```
    출처
    - 삼성전자 사업보고서 (2025.12) · 접수번호 20260310002820 ·
      2-1. 연결 재무상태표 > 부채 > Ⅰ.유동부채
    ```

    **한 출처는 한 줄이다.** 보고서 줄과 자리 줄을 나눠 적으면 출처가 둘
    이상일 때 어느 자리가 어느 접수번호 것인지 줄만 봐서는 알 수 없다.
    들여쓰기로 묶을 수도 없다 — `render_public_answer` 가 두 칸 이상의 공백을
    한 칸으로 줄인다.

    **자리를 아는 인용이 하나라도 있을 때만 붙인다.** 사건·서술 인용은
    재무제표 행이 아니라 그 좌표가 없고, 접수번호만 다시 늘어놓는 것은
    본문이 이미 하는 말을 되풀이하는 것이다.

    **한 문서의 여러 자리는 한 줄에 모은다(이슈 #118 ③).** 같은
    접수번호·보고서명을 여러 행에서 인용했으면(예: 연결·별도 손익계산서의
    매출액을 각각 인용) 문서 머리글(보고서명·접수번호)을 두 번 쓰지 않고
    자리만 ``/`` 로 이어 붙인다.

    ```
    출처
    - 삼성전자 사업보고서 (2025.12) · 접수번호 20260310002820 ·
      2-2. 연결 손익계산서 > 매출액 (주30) / 4-2. 손익계산서 > 매출액 (주29)
    ```

    「한 출처는 한 줄이다」(#93 `4b7fd2e`) 원칙은 그대로다 — 이 병합은 보고서
    줄과 자리 줄을 나누지 않는다. 서로 다른 접수번호(다른 문서·다른 회사)는
    여전히 줄을 나눈다.
    """

    by_receipt: dict[str, dict[str, object]] = {}
    for claim in getattr(payload, "claims", ()) or ():
        for citation in getattr(claim, "citations", ()) or ():
            receipt = getattr(citation, "rcept_no", None)
            if not receipt:
                continue
            row = by_receipt.setdefault(
                receipt, {"report": None, "paths": []})
            report = getattr(citation, "report_name", None)
            if report and not row["report"]:
                row["report"] = report
            path = getattr(citation, "source_path", None)
            if path and path not in row["paths"]:
                row["paths"].append(path)
    if not any(row["paths"] for row in by_receipt.values()):
        return None

    lines = ["출처"]
    for receipt, row in by_receipt.items():
        head = (f"{row['report']} · 접수번호 {receipt}"
                if row["report"] else f"접수번호 {receipt}")
        paths = row["paths"]
        if not paths:
            lines.append(f"- {head}")
            continue
        # 같은 문서에서 여러 행을 인용했어도 문서 자체는 하나다 — 자리만
        # ``/`` 로 이어 한 줄에 담는다(이슈 #118 ③).
        lines.append(f"- {head} · " + " / ".join(paths))
    return "\n".join(lines)


def _with_source_footer(answer: str, payload) -> str:
    """본문에 출처 블록을 붙인다. 근거가 한 문서뿐이면 인라인을 걷어낸다.

    문서가 둘 이상이면 어느 문장이 어느 공시에서 왔는지를 인라인이 말하고
    있으므로 그대로 둔다 — 걷어내면 하단 목록만 남아 대응을 잃는다.
    """

    footer = _source_footer(payload)
    if not footer:
        return answer
    receipts = {
        citation.rcept_no
        for claim in getattr(payload, "claims", ()) or ()
        for citation in getattr(claim, "citations", ()) or ()
        if getattr(citation, "rcept_no", None)
    }
    body = answer
    if len(receipts) == 1:
        body = _INLINE_CITATION.sub("", body)
    return f"{body}\n\n{footer}" if body.strip() else answer


def _hoist_uniform_table_source(answer: str) -> str:
    """표의 한 열이 모든 행에서 같은 값이면 머리에서 한 번만 밝힌다.

    25행짜리 투자계획 표가 행마다 같은 접수번호를 다시 쓰면 그 값이 25번
    반복되고, 한 번만 밝히면 되는 정보가 답변의 17%를 차지한다(`EDGE-031`
    실측 475자). 정정 이력 표의 ``계약(원공시 접수번호)`` 열도 16행이 모두
    같은 원공시를 가리킨다. 어느 쪽이든 그 열은 행을 구별하지 않으므로 표
    안에 있을 이유가 없고, 빼면 「표 전체가 같은 값」이라는 사실이 오히려
    분명해진다.

    **행마다 값이 다르면 손대지 않는다.** 정정 이력 표의 ``근거`` 열은 정정
    건별로 접수번호가 달라서, 지우면 어느 정정이 어느 문서에서 왔는지 알 수
    없게 된다.
    """

    lines = answer.split("\n")
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip().startswith("|") or index + 1 >= len(lines):
            out.append(line)
            index += 1
            continue
        block: list[str] = []
        cursor = index
        while cursor < len(lines) and lines[cursor].strip().startswith("|"):
            block.append(lines[cursor])
            cursor += 1
        rows = [[cell.strip() for cell in row.strip("| ").split("|")]
                for row in block]
        header = rows[0] if rows else []
        data = rows[2:]
        width = len(header)
        # 열을 빼고도 표가 남아야 한다. 두 열짜리 표에서 한 열을 빼면 목록이다.
        if len(block) < 3 or width < 3 or len(data) < 2 or any(
                len(row) != width for row in data):
            out.extend(block)
            index = cursor
            continue

        drop: list[int] = []
        hoists: list[str] = []
        for position in (0, width - 1):
            if position in drop:
                continue
            values = {row[position] for row in data}
            if len(values) != 1:
                continue
            value = values.pop()
            if not value or value == "-":
                continue
            # 근거 열은 접수번호일 때만 뺀다. 마지막 열이 값이면 그것은 데이터다.
            if position == width - 1:
                if header[-1] != "근거" or not _TABLE_RECEIPT_CELL.match(value):
                    continue
                hoists.append(f"근거: {value} (아래 표 전체)")
            else:
                hoists.append(f"{header[position]}: {value}")
            drop.append(position)
        if not drop:
            out.extend(block)
            index = cursor
            continue

        # 표 위 문장이 이미 그 값을 밝혔으면 아무것도 덧붙이지 않는다. 캡션 끝에
        # 괄호를 하나 더 다는 것은 줄만 길어지고 같은 정보가 두 줄 연속으로 나온다.
        above = "\n".join(out)
        for hoisted in hoists:
            if hoisted.split(": ", 1)[-1].split(" (")[0] not in above:
                out.append(hoisted)
        for row in rows:
            out.append("| " + " | ".join(
                cell for position, cell in enumerate(row)
                if position not in drop) + " |")
        index = cursor
    return "\n".join(out)


def _drop_repeated_lines(answer: str) -> str:
    """같은 내용을 두 번 말하는 줄을 한 번만 남긴다.

    조립 경로가 여럿이라 한 사실이 서로 다른 라벨로 두 번 실릴 수 있다.
    ``기준시점 공시 상태`` 와 ``… 기준 공시상 상태`` 가 글자까지 같은 문장을
    각각 싣거나, 같은 원문 문장이 두 축의 근거로 두 번 인용되는 식이다. 값이
    맞아도 독자에게는 같은 말을 두 번 읽는 것으로 나타난다.

    라벨을 뗀 **본문이 완전히 같을 때만** 뒤의 줄을 버린다. 표 행(``|`` 로
    시작)은 원래 반복 구조라 건드리지 않는다.
    """

    seen: set[str] = set()
    kept: list[str] = []
    for line in answer.split("\n"):
        # Narrative section titles sometimes arrive as
        # ``주요 제품 및 서비스 — 가. 주요 제품 ...``.  ``가.`` is a
        # source-document outline marker, not answer content; leaving it after
        # an em dash reads like a broken sentence fragment(#215).  Remove only
        # this tightly bounded heading position and retain the actual title.
        line = re.sub(r"( — )[가-하]\.\s+(?=[가-힣])", r"\1", line)
        stripped = line.strip()
        if not stripped or stripped.startswith("|"):
            kept.append(line)
            continue
        # 넓은 narrative 비교의 근거 색인은 짧은 본문이 같아도 좌표와
        # 접수번호가 서로 다르다. 일반 문장 중복 규칙으로 지우면 검증된
        # 회사·기간 좌표가 사라지므로 이 전용 표면은 모두 보존한다.
        if (stripped.startswith("- ")
                and "(근거: 접수번호 " in stripped
                and re.search(r": [^:()\n]+ 확인 \(근거:", stripped)):
            kept.append(line)
            continue
        # 인용을 먼저 뗀다. 떼지 않으면 ``(근거: 접수번호 …)`` 안의 첫
        # ``": "`` 에서 잘려 인용문이 본문으로 잡히고, 같은 문서를 인용한
        # 서로 다른 줄이 중복으로 판정된다.
        body = _CITATION_TAIL.sub("", stripped).strip()
        # 라벨은 길 수 있다(``대우건설–리비아 … 계약의 2026-06-19 기준 공시상
        # 상태``). 본문만 남으면 되므로 길이로 제한하지 않는다. 이 값은 중복
        # 판정에만 쓰고 출력에는 쓰지 않아, 잘못 갈라도 문장이 손상되지 않는다.
        head, sep, tail = body.partition(": ")
        if sep and len(tail.strip()) >= 20:
            body = tail
        body = body.lstrip("- ").strip()
        if len(body) >= 20 and body in seen:
            continue
        seen.add(body)
        kept.append(_drop_repeated_segments(line))
    return "\n".join(kept)


#: 한 줄 안에서 조각을 가르는 구분자. 결론 줄은 주제별 소결론을 이렇게 잇는다.
_SEGMENT_SPLIT = re.compile(r"(?<=[.;])\s+")


def _drop_repeated_segments(line: str) -> str:
    """한 줄 안에서 같은 말을 두 번 하지 않게 한다.

    비교 결론은 주제별 소결론을 한 줄에 이어 붙인다. 두 주제가 같은 근거를
    쓰면 ``알테오젠: 기술용역수익·상품 매출·ALT-B4`` 같은 나열이 글자까지
    똑같이 두 번 나오고, 한 원문 문장이 다른 조각에 통째로 들어 있기도 하다
    (`EDGE-025`·`DEV-NAR-014` 실측).

    줄 단위 중복 제거는 이것을 보지 못한다 — 반복이 한 줄 안에 있기 때문이다.
    표 행은 원래 반복 구조라 건드리지 않는다.
    """

    if line.strip().startswith("|") or len(line) < 120:
        return line
    parts = _SEGMENT_SPLIT.split(line)
    if len(parts) < 2:
        return line
    kept: list[str] = []
    for part in parts:
        body = part.strip()
        if len(body) >= 30 and any(
                body in other or other in body for other in kept):
            continue
        kept.append(body)
    return " ".join(kept) if len(kept) != len(parts) else line


def _group_amendment_detail_lines(rendered: list[str]) -> list[str]:
    """Say ``최신 유효본``/``원본`` once per run instead of once per line.

    G-O-002 listed six amounts in a row, every one of them opening with
    ``최신 유효본``, then three more opening with ``원본``.  Which version a
    figure belongs to is the same fact for the whole run, so it belongs at the
    head of the run.  ``원본 대비`` states a change across both versions and is
    not part of either run, so it keeps its own wording.
    """

    lines: list[str] = []
    current = ""
    for line in rendered:
        prefix = next(
            (candidate for candidate in ("최신 유효본 ", "원본 ")
             if line.startswith(candidate)
             and not line.startswith("원본 대비")), "")
        if not prefix:
            current = ""
            lines.append(line)
            continue
        if prefix != current:
            lines.append(f"{prefix.strip()}:")
            current = prefix
        lines.append(line[len(prefix):])
    return lines


def _display_units_replacement(payload, question: str | None) -> tuple[str, str] | None:
    """Return ``(old, new)`` to splice a requested unit display into the text.

    Issue #61: a question can ask for a scalar money value in specific units
    ("정확한 원 단위와 조원 단위로 함께 보여줘") instead of the ordinary
    Korean 조·억·만원 grouping.  The instruction never reaches the typed
    payload (``agent.semantic_intent_v1_boundary`` strips it during
    grounding — see ``agent.display_units_v1``), so it is recovered here,
    independently, straight from the original question text.

    Only a single, non-derived scalar money claim is eligible: the question
    names one value to convert, and choosing among several would be a
    semantic decision this deterministic splice must not make.
    """

    directive = parse_display_units_directive(question)
    if directive is None:
        return None
    candidates = [
        c for c in payload.claims
        if not c.derived_from and c.canonical_unit == "원" and c.canonical_value
    ]
    if len(candidates) != 1:
        return None
    claim = candidates[0]
    rendered = render_display_units(claim.canonical_value, directive)
    if not rendered:
        return None
    exact = claim_won_display(claim)
    if not exact or exact == rendered:
        return None
    return exact, rendered


def _comprehensive_matrix_reduction_disclosure(
        question: str | None, matrix: NarrativeMatrix) -> str | None:
    """Name what an over-limit multi-company/period matrix answer left out.

    ``agent.stage1_v1_narrative_matrix`` answers any explicit multi-company
    or multi-period comparison over the 16-unit execution cap from the
    widest schema-legal subset instead of clarifying (#86 17) — every
    company and reducible period there is already a literal span of the
    question, so the question itself names what a narrower subset looks
    like.  That narrowing must still be said, not left for the reader to
    notice by counting rows.  This compares the literal company/year spans
    named in the question against what the rendered matrix actually covers;
    it never touches a fact in the answer itself, so a question this cannot
    parse simply gets no disclosure, not a wrong one.
    """

    from agent.stage1_v1_narrative_matrix import literal_matrix_grammar

    if not question:
        return None
    grammar = literal_matrix_grammar(question)
    if grammar is None:
        return None
    companies, years, _topics = grammar
    shown_companies = {cell.label.partition(" · ")[0] for cell in matrix.cells}
    shown_periods = {cell.label.partition(" · ")[2] for cell in matrix.cells}
    if len(companies) > len(shown_companies):
        return (
            f"요청하신 {len(companies)}개사 중 {len(shown_companies)}개사만 "
            "한 번에 보여드립니다. 나머지는 회사를 지정해 다시 질문해 주세요.")
    if len(years) > len(shown_periods):
        return (
            f"요청하신 {len(years)}개 연도 중 최신 {len(shown_periods)}개 "
            "연도만 한 번에 보여드립니다. 나머지는 연도를 지정해 다시 "
            "질문해 주세요.")
    return None


def _apply_display_units_directive(text: str, payload, question: str | None) -> str:
    from .guard import display_units_target_claims
    claims = display_units_target_claims(payload)
    directive = parse_display_units_directive(question)
    if directive is not None and len(claims) == 2:
        replacements = {}
        for claim in claims:
            old = claim_won_display(claim)
            new = render_display_units(claim.canonical_value, directive)
            if old and new:
                replacements[old] = new
        if replacements:
            text = re.sub("|".join(re.escape(s) for s in sorted(
                replacements, key=len, reverse=True)),
                lambda match: replacements[match.group(0)], text)
        return text
    replacement = _display_units_replacement(payload, question)
    if replacement is None:
        return text
    old, new = replacement
    if old not in text:
        return text
    return text.replace(old, new, 1)


def _chained_quarter_comparison_answer(payload):
    """Show both actual quarter operands, not only internal FY/9M facts."""
    claims = {claim.output_id: claim for claim in payload.claims}
    rates = [claim for claim in payload.claims
             if claim.operator == "percent_change" and len(claim.derived_from) == 2]
    if len(rates) != 1:
        return None
    rate = rates[0]
    operands = [claims.get(key) for key in rate.derived_from]
    if any(claim is None for claim in operands):
        return None
    discrete = [claim for claim in operands if claim.operator == "discrete_from_cumulative"]
    direct = [claim for claim in operands if not claim.derived_from]
    if len(discrete) != 1 or len(direct) != 1 or len(discrete[0].derived_from) != 2:
        return None
    quarter, prior = discrete[0], direct[0]
    base = [claims.get(key) for key in quarter.derived_from]
    if any(claim is None or claim.canonical_unit != "원" for claim in base + operands):
        return None
    lines = [
        f"{prior.label}: {_display_amount(prior)}{_cite(prior)}",
        f"{quarter.label}: {_display_amount(quarter)}{_cite(quarter)}",
        "두 금액은 연초부터의 누적액이 아니라 각각 해당 분기 3개월의 값입니다.",
        f"전기 대비 증감률: {rate.value_text}%{_cite(rate)}",
        "",
        f"단독 분기 계산: {_display_amount(base[0])} − {_display_amount(base[1])}"
        f" = {_display_amount(quarter)}.",
        "증감률은 반올림 전 두 분기 금액으로 (이번 분기 − 앞선 분기) ÷ 앞선 분기 × 100을 계산했습니다.",
    ]
    return append_public_qualifications("\n".join(lines), render_safe_limitations(payload.limitations))


def _financial_trend_rows(payload, question: str | None):
    """Sort only a closed, single-metric 3+ period series; never infer scope."""
    if not question or "추이" not in question:
        return None
    operand_ids = {oid for c in payload.claims for oid in c.derived_from}
    rows = []
    for claim in payload.claims:
        if claim.output_id in operand_ids:
            # A standalone quarter can be both requested and an operand of
            # the next quarter. Without output-selection metadata, fall back
            # instead of silently dropping that requested quarter.
            if re.search(r"\d{4}년 [1-4]분기(?: 단독)? (?:연결|별도) ", claim.label):
                return None
            continue
        if claim.operator not in {None, "discrete_from_cumulative"} or not claim.citations:
            return None
        match = re.fullmatch(
            r"(?P<company>.+?) (?P<year>\d{4})년(?: (?P<quarter>[1-4])분기(?: 단독)?)? "
            r"(?P<scope>연결|별도) (?P<metric>.+)", claim.label)
        if not match or claim.canonical_unit != "원":
            return None
        try:
            value = Decimal(claim.canonical_value)
            if not value.is_finite():
                return None
        except (InvalidOperation, TypeError, ValueError):
            return None
        row = match.groupdict()
        rows.append((int(row['year']), int(row['quarter'] or 0), value, claim,
                     (row['company'], row['scope'], row['metric'])))
    if (len(rows) < 3 or len({r[4] for r in rows}) != 1
            or len({(r[0], r[1]) for r in rows}) != len(rows)
            or len({bool(r[1]) for r in rows}) != 1):
        return None
    return sorted(rows, key=lambda row: (row[0], row[1]))


class TemplateComposer:
    def compose(self, payload, *, question: str | None = None) -> str:
        text = _hoist_uniform_table_source(
            _drop_repeated_lines(self._compose(payload, question=question)))
        text = _apply_display_units_directive(text, payload, question)
        # 출처는 마지막이다 — 위의 정리들이 본문만 보게 하고, 하단 블록이
        # 「같은 줄 반복」이나 「표의 균일 열」로 잘못 걸리지 않게 한다.
        return _with_source_footer(text, payload)

    def _compose(self, payload, *, question: str | None = None) -> str:
        fs = payload.final_status
        if fs == "clarify":
            return self._clarify(payload)
        if fs in ("refuse", "not_found", "failure"):
            return self._refuse(payload, question=question)
        policy_partial = self._policy_partial_answer(payload, question=question)
        if policy_partial is not None:
            return policy_partial
        trend = _financial_trend_rows(payload, question)
        if trend is not None:
            directions = [(right[2] > left[2]) - (right[2] < left[2])
                          for left, right in zip(trend, trend[1:])]
            words = {1: "증가", -1: "감소", 0: "변동 없음"}
            if len(set(directions)) == 1:
                summary = {1: "계속 증가했습니다.", -1: "계속 감소했습니다.",
                           0: "같은 수준을 유지했습니다."}[directions[0]]
            else:
                summary = "시간순으로 " + " → ".join(words[d] for d in directions) + " 흐름입니다."
            company, scope, metric = trend[0][4]
            lines = [f"{company} {scope} {metric}: {summary}", "",
                     "| 기간 | 금액 | 근거 |", "|---|---|---|"]
            for year, quarter, _, claim, _ in trend:
                period = f"{year}년" + (f" {quarter}분기" if quarter else "")
                lines.append(f"| {period} | {_display_amount(claim)} | {_cite(claim)} |")
            lines.extend(self._public_notes(payload, question=question))
            return "\n".join(lines)
        amount_availability = _withheld_contract_amount_answer(payload, question)
        if amount_availability is not None:
            return amount_availability
        termination_membership = _termination_membership_answer(payload, question)
        if termination_membership is not None:
            return termination_membership
        extrema = [c for c in payload.claims if c.state == "argmax_summary"]
        if len(extrema) == 1 and extrema[0].value_text:
            summary = extrema[0]
            lines = [
                f"제공된 자료에서 값이 확인된 사건 중 {summary.label}은 "
                f"{_display_amount(summary)}입니다.{_cite(summary)}"]
            # The backend already determined the winner and whether values
            # were withheld. Do not recalculate or widen that finite scope.
            if summary.text and "전체" in summary.text and "확정할 수 없음" in summary.text:
                lines.append("비공개된 값이 있어 전체 사건의 최댓값·최솟값은 확정할 수 없습니다.")
            for claim in payload.claims:
                if claim is summary:
                    continue
                if (claim.value_text == summary.value_text
                        and claim.citations == summary.citations):
                    continue
                lines.append(self._claim_line(claim))
            lines.extend(self._public_notes(payload, question=question))
            return "\n".join(lines)
        ambiguous_termination = _ambiguous_termination_observation_answer(
            payload, question)
        if ambiguous_termination is not None:
            return ambiguous_termination
        condition_and_termination = _event_condition_and_termination_answer(
            payload, question)
        if condition_and_termination is not None:
            return condition_and_termination
        investment_judgment_list = _investment_judgment_title_date_list(
            payload, question)
        if investment_judgment_list is not None:
            return investment_judgment_list
        event_table = _event_collection_table(payload, question)
        if event_table is not None:
            return event_table
        contract_listing_table = _contract_event_listing_table(payload, question)
        if contract_listing_table is not None:
            return contract_listing_table
        dated_snapshot = _same_day_documents_and_final_state(payload)
        if dated_snapshot is not None:
            return dated_snapshot
        amount_difference = _event_amount_difference_answer(payload, question)
        if amount_difference is not None:
            return amount_difference
        quarter_comparison = _chained_quarter_comparison_answer(payload)
        if quarter_comparison is not None:
            return quarter_comparison
        financing = self._financing_answer(payload, question=question)
        if financing is not None:
            return financing
        document_history = self._concise_document_history_answer(
            payload, question=question)
        if document_history is not None:
            return document_history
        # A state-only question remains state-only even when the backend
        # attaches correction evidence used to prove that state.  Let the
        # ordinary event renderer select the typed cutoff states before the
        # generic correction-history route can expand every changed field.
        if (_is_state_only_event_question(question)
                and any(_is_event_summary(claim)
                        for claim in payload.claims)):
            return self._answer(payload, question=question)
        # A correction sidecar is a compact, typed representation of the
        # otherwise duplicated document snapshots.  Prefer its table for every
        # correction history, not just a fixture-specific long answer.
        if (any(claim.operator == "correction_diff" for claim in payload.claims)
                and not any(
                    claim.derived_from
                    and claim.operator != "correction_diff"
                    for claim in payload.claims)):
            return self._correction_answer(payload, question=question)
        # Only a typed, multi-cell execution sidecar takes this concise route.
        # Legacy one-cell long-form narratives still preserve their complete
        # structure below through _claim_line.
        matrix = NarrativeMatrix.from_payload(payload)
        if matrix is not None:
            answer = matrix.deterministic(question=question)
            disclosure = _comprehensive_matrix_reduction_disclosure(
                question, matrix)
            return f"{answer}\n{disclosure}" if disclosure else answer
        # One-cell narrative retrievals used to fall through to `_claim_line`,
        # which faithfully printed a whole canonical block.  Keep the claim
        # itself untouched for provenance, but render a bounded sentence or a
        # complete typed investment-table row here.  This is deterministic, so
        # HCX can neither drop a row/citation nor add an unsupported fact.
        digest = NarrativeDigest.from_payload(payload, question=question)
        if digest is not None:
            return digest.deterministic()
        return self._answer(payload, question=question)

    @staticmethod
    def _reason_messages(p, *, include_alternatives: bool = True) -> list[str]:
        """Render each typed terminal reason once at its public precedence.

        The limitation registry owns a reason's canonical wording when it has
        one.  ``_REASON_KO`` is only a fallback for codes without a registry
        message; otherwise two differently worded sentences can repeat the
        same boundary in a refusal.
        """

        messages = render_safe_limitations(p.limitations)
        codes = list(p.reasons) + [limitation.code for limitation in p.limitations]
        # 어떤 사유 코드는 더 정확한 코드의 운반체로 쓰인다. 접수시각을 분·초
        # 단위로 묻는 요청은 unsupported_temporal_scope 를 달고 오지만 실제
        # 뜻은 intraday_order_unavailable 이고, 그 문구가 이미 나온다. 둘 다
        # 내면 「범위를 벗어납니다」가 범위 안의 날짜에 붙어 틀린 말이 된다.
        carried = {code.split(":", 1)[0] for code in codes}
        superseded = ({"unsupported_temporal_scope"}
                      if "intraday_order_unavailable" in carried else set())
        for code in codes:
            base = code.split(":", 1)[0]
            if base in superseded:
                continue
            if safe_limitation_message(base) is not None:
                continue
            message = _REASON_KO.get(base)
            if message and message not in messages:
                messages.append(message)
        if include_alternatives:
            bases = {code.split(":", 1)[0] for code in codes}
            combined_policy = {"future_forecast", "investment_advice"} <= bases
            if combined_policy and _COMBINED_POLICY_ALTERNATIVE_KO not in messages:
                messages.append(_COMBINED_POLICY_ALTERNATIVE_KO)
            for code in codes:
                base = code.split(":", 1)[0]
                if combined_policy and base in {"future_forecast", "investment_advice"}:
                    continue
                alternative = _ALTERNATIVE_KO.get(base)
                if alternative and alternative not in messages:
                    messages.append(alternative)
        return messages

    def _policy_partial_answer(self, p, *, question: str | None = None) -> str | None:
        """Put a policy boundary before sealed, cited fallback facts.

        A policy terminal can carry a separately executed factual fallback.
        It remains a partial answer: the refused request must be the first
        thing a reader sees, while the independently verified fallback facts
        remain available as an alternative.  This is keyed to terminal policy
        codes, not an issuer, value, or fixture identity.
        """

        if p.final_status != "partial_answer" or not p.reasons or not p.claims:
            return None
        policy_codes = {
            "future_forecast", "investment_advice",
            "causal_inference_beyond_scope",
            "unsupported_semantic_target_substituted",
        }
        reasons = {code.split(":", 1)[0] for code in p.reasons}
        if not reasons or not reasons.issubset(policy_codes):
            return None
        messages = self._reason_messages(p, include_alternatives=False)
        if not messages:
            return None
        # 「대신 확인 가능한 …」 머리글이 바로 아래 대체 사실을 잇는다는 것을
        # 이미 말한다. ``unsupported_semantic_target_substituted`` 문장은
        # 같은 뜻이므로 머리글 앞에 두지 않고, 답 끝의 「다만,」 절로 한 번만
        # 남긴다(DEV-INV-009, #86-16). 다른 사유(예: 집계 범위 불일치)는
        # 대체 사실과 무관한 별개 사실이라 그대로 첫머리에 남는다.
        substituted_message = safe_limitation_message(
            "unsupported_semantic_target_substituted")
        top_messages = [m for m in messages if m != substituted_message]
        lines = list(top_messages)
        # 「대신 … 과거 공시 사실」은 요청한 값을 못 찾았을 때의 말이다.  집계
        # 범위나 기간 길이가 달라 **계산만** 하지 않은 답에는 두 값이 다 실려
        # 있으므로, 그것을 대체물이라 부르면 찾은 값을 못 찾았다고 말하는 셈이
        # 된다(DEV-INV-009).
        limitation_codes = {
            (getattr(row, "code", "") or "").split(":", 1)[0]
            for row in (p.limitations or ())}
        found_but_not_combined = bool(
            (reasons | limitation_codes)
            & {"incomparable_aggregation_scope", "period_length_mismatch"})
        lines.append("확인된 값:" if found_but_not_combined
                     else "대신 확인 가능한 과거 공시 사실:")
        lines.extend(self._claim_lines(p.claims))
        notes = self._public_notes(p, question=question)
        # The policy boundary above already renders these same messages.
        notes = [note for note in notes if note not in top_messages]
        if notes:
            _extend_public_qualifications(lines, notes)
        return "\n".join(line for line in lines if line)

    def _financing_answer(self, p, *, question: str | None = None) -> str | None:
        """Render a logical financing decision once, not once per correction.

        Disclosure list tasks retain every filing as a claim for auditability,
        while ``_financing_summary`` emits a typed logical-decision projection.
        When that projection is present, the repeated source-document and
        same-decision-date lines add no public fact.  Keep the complete
        lineage, latest effective terms, original comparison and scoped zero
        results instead.
        """

        claims = list(p.claims)
        logical = [claim for claim in claims
                   if (claim.output_id or "").endswith(".financing.logical_count")]
        lineages = [claim for claim in claims
                    if (claim.output_id or "").endswith(".lineage")
                    and ".financing." in (claim.output_id or "")]
        type_claims = [claim for claim in claims
                       if ".financing.type." in (claim.output_id or "")]
        if not logical or not lineages or not type_claims:
            return None

        def members(claim):
            chain = (claim.text or "").partition(";")[0]
            return [value.strip() for value in chain.split("→") if value.strip()]

        confirmed = [claim for claim in type_claims if claim.state == "confirmed"]
        missing = [claim for claim in type_claims
                   if claim.state == "not_found_in_scope"]
        first = logical[0]
        count = first.value_text or ""
        if len(lineages) == 1:
            lineage = lineages[0]
            chain = members(lineage)
            form = (lineage.label or "자금조달").removesuffix(" 정정 계보")
            correction_count = max(0, len(chain) - 1)
            decision = (
                f"{form} 1건"
                + (f"(원본 1건·정정 {correction_count}건)" if correction_count else "")
            )
            lines = [f"확인된 자금조달은 {decision}입니다.{_cite(lineage)}"]
        else:
            lines = [
                f"확인된 자금조달 의사결정은 {count}건입니다.{_cite(first)}"
            ]
        if confirmed and (len(lineages) != 1 or len(confirmed) > 1):
            labels = [claim.label.removesuffix(" 확인 여부") for claim in confirmed]
            lines.append(f"확인된 유형: {'·'.join(labels)}.")
        if missing:
            labels = [claim.label.removesuffix(" 확인 여부") for claim in missing]
            scope = (missing[0].text or "").removesuffix(" 범위에서 확인되지 않음")
            match = re.fullmatch(
                r"(?P<company>.+?)·(?P<start>\d{8})~(?P<end>\d{8})·"
                r"(?P<family>.+)", scope)
            if match:
                start, end = match.group("start"), match.group("end")
                if (start[4:] == "0101" and end[4:] == "1231"
                        and start[:4] == end[:4]):
                    period = f"{start[:4]}년"
                else:
                    period = f"{_public_date(start)}~{_public_date(end)}"
                family = match.group("family")
                if family == "주요사항보고":
                    family = "주요사항보고서"
                suffix = (
                    f" {match.group('company')}의 {period} {family}에서")
            else:
                suffix = f" {scope} 범위에서" if scope else ""
            label_text = "·".join(labels)
            lines.append(
                f"{label_text}{_josa(label_text, '은', '는')}"
                f"{suffix} 확인되지 않았습니다.")

        lines.append("정정 계보와 최신 유효본:")
        lines.extend(self._claim_line(claim) for claim in lineages)
        detail_claims = [
            claim for claim in claims
            if ".financing." in (claim.output_id or "")
            and claim not in logical + lineages + type_claims
        ]
        if detail_claims:
            lines.append("정정 전후 내용:" if any(
                claim.operator == "correction_diff" for claim in detail_claims)
                else "추가 확인 사항:")
            lines.extend(_group_amendment_detail_lines(
                [self._claim_line(claim) for claim in detail_claims]))
        notes = self._public_notes(p, question=question)
        if notes:
            _extend_public_qualifications(lines, notes)
        return "\n".join(line for line in lines if line)

    def _concise_document_history_answer(
            self, p, *, question: str | None = None) -> str | None:
        """Answer a false no-correction premise with chain and latest version.

        Effective intervals and individual SUPERSEDES edges are valuable for a
        chronology request, but obscure a simple ``was it corrected?`` answer.
        Restrict this projection to a false premise plus one typed document
        lineage and no correction-field diff, so detailed history paths retain
        their existing renderer.
        """

        if (not any(verdict.verdict == "false" for verdict in p.premise_verdicts)
                or any(claim.operator == "correction_diff" for claim in p.claims)):
            return None
        lineages = [claim for claim in p.claims
                    if claim.label == "문서 정정 계보" and claim.state == "corrected"]
        if not lineages:
            return None
        parsed = []
        for candidate in lineages:
            chain_text = (candidate.text or "").removeprefix("정정 있음: ")
            members = [value.strip() for value in chain_text.split("→") if value.strip()]
            # A DART receipt begins with the public filing date.  We only
            # select a latest chain when that date is unique; same-day chains
            # retain the detailed path because receipt number order is not a
            # valid intraday ordering authority.
            day = members[-1][:8] if members and re.fullmatch(r"\d{14}", members[-1]) else ""
            if len(members) >= 2 and day:
                parsed.append((day, candidate, members))
        if not parsed:
            return None
        latest_day = max(day for day, _, _ in parsed)
        latest = [(candidate, members) for day, candidate, members in parsed
                  if day == latest_day]
        if len(latest) != 1:
            return None
        lineage, members = latest[0]
        if len(members) < 2:
            return None
        latest_receipt = members[-1]
        lines = [
            f"아니요. 정정된 적 있습니다. 원본 뒤 정정 {len(members) - 1}회가 확인됐고, "
            f"최신본은 {latest_receipt}입니다.{_cite(lineage)}"]
        lines.append(f"정정 계보: {' → '.join(members)}{_cite(lineage)}")
        notes = self._public_notes(p, question=question)
        if notes:
            _extend_public_qualifications(lines, notes)
        return "\n".join(line for line in lines if line)

    def _correction_answer(self, p, *, question: str | None = None) -> str:
        """Current event status followed by one non-duplicated change table."""

        compact_question = re.sub(r"\s+", "", question or "")
        complete_lineage_request = (
            any(token in compact_question for token in ("최초", "원공시"))
            and "이력" in compact_question
            and any(token in compact_question for token in ("전체", "모두"))
        )
        source_scope_limited = complete_lineage_request and any(
            limitation.code.split(":", 1)[0]
            == "source_scope_prevents_complete_lineage"
            for limitation in p.limitations
        )

        lines = _premise_lines(p.premise_verdicts, claims=p.claims)
        if source_scope_limited:
            scope_message = safe_limitation_message(
                "source_scope_prevents_complete_lineage")
            if scope_message:
                lines.append(scope_message)
        attribution = _attribution_conclusion(p, question)
        conclusion = attribution or _correction_conclusion(
            p.claims,
            reject_increase=(
                any(verdict.verdict == "false"
                    for verdict in p.premise_verdicts)
                or bool(question and re.search(r"늘|증가|커진|10\s*조", question))
            ),
        )
        asks_correction_inventory = bool(question and re.search(
            r"정정\s*(?:이력|내역|내용)|변경\s*(?:이력|내역|내용)|"
            r"(?:전체|모든)[^?\n]{0,20}(?:정정|변경)|정정\s*전후\s*흐름",
            question,
        ))
        if attribution is not None and not asks_correction_inventory:
            # 「어느 원계약인가」의 답은 귀속 가능 여부와 후보 접수번호다.
            # 정정 사실 전체는 이 결론의 내부 근거이지 요청 결과가 아니므로,
            # 사용자가 이력까지 명시한 경우에만 아래 장문 표로 확장한다.
            lines.append(attribution)
            notes = self._public_notes(p, question=question)
            if notes:
                _extend_public_qualifications(lines, notes)
            return "\n".join(line for line in lines if line)

        requested_fields = _requested_correction_fields(question)
        asks_correction_reason = bool(question and re.search(
            r"왜|이유|사유|원인", question))
        if (requested_fields and not asks_correction_inventory
                and not asks_correction_reason):
            # 명시된 필드가 있으면 상태·정정사유·기타 투자판단 문안까지 모두
            # 펼치지 않는다. typed correction_diff 가운데 요청 축만 투영한다.
            amount_summary = (
                _correction_amount_summary(p.claims)
                if "contract_amount" in requested_fields else None
            )
            if amount_summary:
                if any(limitation.code.split(":", 1)[0] in {
                        "source_scope_prevents_complete_lineage", "source_scope_raw_absent"}
                       for limitation in p.limitations):
                    amount_summary = amount_summary.replace(
                        "최초값", "확인된 첫값").replace("최초 ", "확인된 첫값 ")
                lines.append(amount_summary)
            history = _correction_history_table(
                p.claims, requested_fields=requested_fields)
            if history:
                if lines:
                    lines.append("")
                lines.append(history)
            elif conclusion:
                lines.append(conclusion)
            notes = self._public_notes(p, question=question)
            if notes:
                _extend_public_qualifications(lines, notes)
            return "\n".join(line for line in lines if line)

        explicit_states = [claim for claim in p.claims if _is_event_summary(claim)]
        # Some event backends additionally emit a prose ``기준시점 공시 상태``
        # claim.  The date-bearing state is more useful to a reader; retain
        # the prose variant only when there is no typed cutoff state.
        states = explicit_states or [
            claim for claim in p.claims
            if claim.label == "기준시점 공시 상태"
        ]
        state_table = _event_state_table(states)
        if conclusion and not state_table and not source_scope_limited:
            lines.append(conclusion)
        correction_reasons = [
            claim for claim in p.claims
            if (claim.label or "").startswith("정정사유")
            and (claim.text or claim.value_text)
        ]
        seen_reasons = set()
        correction_reason_lines = []
        for correction_reason in correction_reasons:
            reason = correction_reason.text or correction_reason.value_text
            receipt = next((
                citation.rcept_no for citation in correction_reason.citations
                if citation.rcept_no and re.fullmatch(
                    r"\d{14}", citation.rcept_no)
            ), None)
            key = (reason, receipt)
            if key in seen_reasons:
                continue
            seen_reasons.add(key)
            when = (
                f"{receipt[:4]}-{receipt[4:6]}-{receipt[6:8]} "
                if len(correction_reasons) > 1 and receipt else ""
            )
            correction_reason_lines.append(
                f"{when}정정 사유: {str(reason).rstrip('. ')}."
                f"{_cite(correction_reason)}")
        if len(correction_reason_lines) > 1:
            # DEV-EVT-001 은 여덟 줄이 저마다 「정정 사유:」로 시작했다. 무엇을
            # 늘어놓는 줄인지는 묶음마다 한 번이면 되고, 줄마다 달라지는 것은
            # 날짜와 사유다.
            correction_reason_lines = ["정정 사유:"] + [
                line.replace(" 정정 사유: ", " ", 1)
                for line in correction_reason_lines]

        if source_scope_limited:
            observed_names = {
                "계약금액", "계약상대", "계약(수주)일자", "시작일",
            }
            observed_fields = []
            seen_observed = set()
            for claim in p.claims:
                if (claim.label not in observed_names
                        or (claim.value_text is None and claim.text is None)):
                    continue
                key = (
                    claim.label, claim.value_text, claim.text,
                    tuple(citation.rcept_no
                          for citation in (claim.citations or ())),
                )
                if key in seen_observed:
                    continue
                seen_observed.add(key)
                observed_fields.append(claim)
            if observed_fields:
                lines.append("제공 자료에서 확인되는 계약 주요 내용:")
                lines.extend(self._claim_line(claim)
                             for claim in observed_fields)

            dated_context = []
            seen_context = set()
            for claim in p.claims:
                if claim.label != "효력발생조건" or not claim.text:
                    continue
                for bullet in _advisory_bullets(claim.text):
                    match = re.match(r"(?P<date>\d{4}-\d{2}-\d{2})", bullet)
                    if match is None:
                        continue
                    key = (match.group("date"), bullet)
                    if key in seen_context:
                        continue
                    seen_context.add(key)
                    dated_context.append((match.group("date"), bullet, claim))
            if dated_context:
                lines.append("제공 자료에서 확인되는 계약 경과:")
                for _date, bullet, claim in sorted(
                        dated_context, key=lambda row: (row[0], row[1])):
                    public_bullet = re.sub(
                        r"\s+\(\d+\)\s+(?:관련 내용|진행 경과|기타)\s*$",
                        "", _display_text(bullet))
                    lines.append(
                        f"- {public_bullet.rstrip('. ')}."
                        f"{_cite(claim)}")
        if state_table:
            if lines:
                lines.append("")
            lines.append(state_table)
        else:
            lines.extend(self._claim_line(claim) for claim in states)
        # 이슈 #94 24 — 정정이 열다섯 번이면 아래에 열다섯 줄이 쌓이는데
        # 사용자가 알고 싶은 것은 「얼마에서 얼마가 됐나」다. 그 결론을
        # **맨 앞에** 둔다. 아래 이력·표는 그대로 남는다.
        history = _correction_history_table(p.claims)
        amount_summary = _correction_amount_summary(p.claims)
        if amount_summary:
            if any(limitation.code.split(":", 1)[0] in {
                    "source_scope_prevents_complete_lineage", "source_scope_raw_absent"}
                   for limitation in p.limitations):
                amount_summary = amount_summary.replace(
                    "최초값", "확인된 첫값").replace("최초 ", "확인된 첫값 ")
            lines.append(amount_summary)
        lines.extend(correction_reason_lines)

        # A lineage receipt sequence and the current public contract fields
        # provide context without repeating every historical snapshot.  The
        # before/after table below remains the complete change inventory.
        lineage = next((claim for claim in p.claims
                        if claim.label == "문서 정정 계보"), None)
        observed_history = [
            claim for claim in p.claims
            if claim.label.startswith("확인된 공시 이력")]
        latest_observed = next((
            claim for claim in p.claims
            if claim.label == "기준시점까지 확인된 마지막 공시"), None)
        flow = _event_flow_table(lineage, states, p.claims)
        if flow:
            lines.append(flow)
        elif not observed_history and lineage is not None and not history:
            # 이슈 #94 24 — 「공시 정정 이력: A → B → …」 계보 줄은 접수번호를
            # 늘어놓기만 한다. 아래 정정 표가 그 접수번호를 근거 열에 전부
            # 싣고 날짜·항목·변경 내용까지 함께 보여 주므로, 표가 나가는
            # 답변에서 이 줄은 같은 말을 세 번째로 하는 것이다.
            #
            # 표가 없으면 그대로 둔다 — 그때는 이 줄이 유일한 계보다.
            lines.append(self._claim_line(lineage))
        if latest_observed is not None:
            lines.append(self._claim_line(latest_observed))
        current_field_names = (
            {"해지금액", "매출액대비", "해지사유", "해지일자"}
            if states else {
                "계약금액", "최근매출액", "매출액대비", "계약상대",
                "계약(수주)일자", "시작일", "종료일",
                "해지금액", "해지사유", "해지일자", "해지계약명",
            }
        )
        final_details = _same_termination_receipt_details(p.claims, states)
        final_detail_ids = {claim.output_id for claim in final_details}
        current_fields = []
        seen_current_fields = set()
        for claim in p.claims:
            if (claim.label not in current_field_names
                    or (states and claim.output_id not in final_detail_ids)
                    or (claim.value_text is None
                        and claim.text is None
                        and claim.state is None)):
                continue
            key = (
                claim.label, claim.value_text, claim.text, claim.state,
                tuple(citation.rcept_no for citation in (claim.citations or ())),
            )
            if key in seen_current_fields:
                continue
            seen_current_fields.add(key)
            current_fields.append(claim)
        if current_fields:
            lines.append("최종 해지 공시 기준 주요 내용:" if states
                         else "현재 공시 기준 주요 내용:")
            lines.extend(self._claim_line(claim) for claim in current_fields)

        # If the final termination receipt has no structured amount/date/reason
        # at all, retain its verified explanatory field rather than reducing
        # the answer to an unexplained state.  Structured cases stay compact.
        if states and not current_fields:
            state_receipts = _final_termination_receipts(states)
            termination_context = next((
                claim for claim in p.claims
                if claim.label == "효력발생조건"
                and any(citation.rcept_no in state_receipts
                        for citation in (claim.citations or ()))
            ), None)
            if termination_context is not None:
                lines.append("해지 공시 주요 내용:")
                lines.append(self._claim_line(termination_context))

        if history:
            if lines:
                lines.append("")
            lines.append(history)

        narrative_summary = _narrative_correction_summary(p.claims)
        if narrative_summary:
            lines.append(narrative_summary)

        notes = self._public_notes(p, question=question)
        # A document correction query can carry an upstream event-identity
        # diagnostic even though its exact original-to-correction document
        # chain is resolved. Without at least two public original candidates
        # that diagnostic is not an auditable answer boundary and contradicts
        # the lineage shown above, so keep it in trace only.
        if lineage is not None:
            singleton_ambiguities = [
                limitation for limitation in p.limitations
                if (limitation.code.split(":", 1)[0] in {
                    "ambiguous_event_identity", "ambiguous_event_origin"}
                    and len({receipt for receipt in limitation.affected_doc_ids
                             if re.fullmatch(r"\d{14}", receipt or "")}) < 2)
            ]
            hidden_messages = set(render_safe_limitations(
                singleton_ambiguities))
            notes = [note for note in notes if note not in hidden_messages]
        if notes:
            _extend_public_qualifications(lines, notes)
        return "\n".join(line for line in lines if line)

    # ── 역질문 ──────────────────────────────────────────────────────────
    def _clarify(self, p) -> str:
        c = p.clarification
        lines = [self._clarify_question(c)]
        for tgt, opts in (c.options or {}).items():
            if opts:
                label = _INTERNAL_CLARIFY_FIELD_LABELS.get(tgt, tgt)
                lines.append(f"- {label}: " + " / ".join(opts[:6]))
        return "\n".join(lines)

    @staticmethod
    def _clarify_question(c) -> str:
        """Rewrite a raw internal-parameter-name clarify question, if any.

        `app/tools/financial.py`'s ambiguous-account-path path builds
        `Clarification.question` as a raw f-string that bakes in the
        internal concept id and a Python tuple repr of the discriminator
        field name (``"삼성SDI net_income 조회에 ('account_path',) 지정이
        필요합니다."``) — issue #172 M33-b. That backend is out of scope for
        this fix, so the leak is repaired here at render time: detect the
        fixed internal-id shape and rebuild the sentence in plain wording,
        using the already-public concept display-name table. Any other
        clarify question (the typed `ClarificationAuthority` path most
        questions use) never matches and passes through unchanged.
        """

        match = _INTERNAL_CONCEPT_CLARIFY_QUESTION.match(c.question)
        if match is None:
            return c.question
        from agent.planning import concept_display_name

        corp = match.group("corp").strip()
        concept_label = concept_display_name(match.group("concept"))
        targets = list(getattr(c, "targets", None) or [])
        field_labels = " / ".join(
            _INTERNAL_CLARIFY_FIELD_LABELS.get(target, target)
            for target in targets) or "세부 항목"
        return f"{corp} {concept_label} 조회에 {field_labels} 확인이 필요합니다."

    # ── 거부·한계 ────────────────────────────────────────────────────────
    @staticmethod
    def _question_company_surface(question: str | None) -> str | None:
        """Return a short company surface for an out-of-universe explanation.

        Terminal authority intentionally carries only the typed reason code, so
        the original question is the sole public source for the missing name.
        These patterns recover only a leading possessive or ``회사 + 자료``
        surface; an uncertain parse stays generic instead of guessing.
        """

        if not question:
            return None
        text = re.sub(r"\s+", " ", question).strip()
        patterns = (
            r"^(?P<name>[^?.,]{1,40}?)의\s+",
            r"^(?P<name>[^?.,]{1,40}?)\s+(?:자료|공시|데이터)(?:가|는|도|를|만|\s)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match is None:
                continue
            name = match.group("name").strip(" \t'\"‘’“”")
            if name and not re.search(r"\d{4}년|매출|실적|계약|투자", name):
                return name
        return None

    def _refuse(self, p, *, question: str | None = None) -> str:
        msgs = self._reason_messages(p)
        codes = list(p.reasons) + [l.code for l in p.limitations]
        outside_universe = any(
            code.split(":", 1)[0] == "corp_not_in_universe"
            for code in codes
        )
        company = self._question_company_surface(question) if outside_universe else None
        if company:
            generic = safe_limitation_message("corp_not_in_universe")
            msgs = [msg for msg in msgs if msg != generic]
            msgs.insert(
                0,
                f"「{company}」{_josa(company, '은', '는')} 제공된 70개 기업의 DART 공시자료 범위에 "
                "포함되지 않습니다. 이 자료는 회계기간 2023년 1월부터 2026년 "
                "1분기까지이며, 접수일은 2026년 6월 19일까지입니다. 따라서 "
                "해당 기업의 값은 외부 지식으로 추정하지 않았습니다.",
            )
        injected = self._injection_prefix(question) if any(
            code.split(":", 1)[0].startswith("prompt_injection")
            for code in codes) else None
        if injected:
            # 주입이 섞였다고 앞의 정상 요청까지 버리지는 않는다 — 다만 **실행도
            # 하지 않는다.** 주입 절을 떼어낸 나머지가 안전하다는 보장이 없어
            # fail-closed 를 유지한다. 대신 무엇을 왜 처리하지 않았는지 밝혀,
            # 사용자가 자기 질문이 버려진 이유를 알고 바로 고칠 수 있게 한다.
            # 인용한 요청이 물음표로 끝나면 뒤에 조사를 붙일 수 없다
            # (``「매출액은?」는``). 조사가 필요 없는 형태로 잇는다.
            msgs.append(
                f"함께 요청하신 「{injected}」 부분은 "
                "답변할 수 있는 질문입니다. "
                "다만 지시를 바꾸려는 문구가 같이 들어 있어 이번 요청은 "
                "처리하지 않았습니다. 그 문구를 빼고 다시 질문해 주세요.")
        outside_corpus = self._outside_corpus_reason(question, codes=codes)
        if outside_corpus and not outside_universe and not injected:
            # 거절은 사실을 만들어 내는 자리가 아니라 왜 못 하는지 알리는
            # 자리다. 무엇을 물었는지 되짚어 말하는 것은 회사명을 되짚는
            # 위 분기와 같은 성격이고, 여기서 새로 주장하는 사실은 없다.
            # 구체적인 사유를 말했으면 총칭 거절은 같은 말을 흐리기만 한다.
            generics = {
                safe_limitation_message("unsupported_semantic_target"),
                _REASON_KO.get("unsupported_temporal_scope"),
                safe_limitation_message("corpus_coverage_unavailable"),
                _REASON_KO.get("corpus_coverage_unavailable"),
            }
            msgs = [msg for msg in msgs if msg not in generics]
            msgs.insert(0, outside_corpus)
        if not msgs:
            msgs.append("제공된 공시 데이터로는 요청하신 내용을 확정할 수 없습니다.")
        return " ".join(msgs)

    #: 자료 범위 밖이라 답할 수 없는 요청의 종류. 총칭 거절만 내보내면
    #: 「자료에 없다」와 「그런 일이 없었다」가 구별되지 않는다.
    # 「메리츠금융지주가」 안에 「주가」가 들어 있다. 한국어에는 낱말 경계가
    # 없어서 회사명 + 주격 조사가 그대로 걸린다 — 코퍼스에 「지주」로 끝나는
    # 회사가 다섯 곳이라 그 회사의 모든 거절에 엉뚱한 주가 문구가 붙었다.
    # 앞뒤로 낱말이 이어지지 않을 때만 본다.
    # 뒤에는 조사가 붙으므로(「주가는」) 앞만 본다.
    _PRICE_REQUEST = re.compile(
        r"(?<![가-힣])(?:주가|시세|종가)|주식\s*가격")
    _DELISTING_REQUEST = re.compile(r"상장\s*폐지|상폐")
    #: 코퍼스 cutoff 이후 날짜를 "까지/기준"으로 제출·공시·보고서를 묻는
    #: 요청. `agent/stage1_v1_policy_backend.py`의 future_cutoff 판정과 같은
    #: 모양을 찾아, 총칭 문구 대신 회사명·요청일을 넣은 문장을 앞세운다.
    _FUTURE_SUBMISSION_REQUEST = re.compile(
        r"(20[0-9]{2})\s*년\s*(1[0-2]|0?[1-9])\s*월"
        r"(?:\s*(3[01]|[12][0-9]|0?[1-9])\s*일)?"
        r".{0,12}(?:까지|기준).{0,12}(?:제출|공시|보고서)")
    #: `_question_company_surface` only recovers a leading possessive
    #: (``~의``) or ``~ 자료/공시/데이터`` surface.  A future-submission
    #: request instead opens with the issuer as the grammatical subject
    #: (``삼성전자가 ... 제출한``), so this narrow, locally-scoped pattern
    #: covers that one shape without widening the shared helper's reach into
    #: every other refusal branch that calls it.
    _LEADING_SUBJECT_SURFACE = re.compile(
        r"^(?P<name>[^\s?.,]{1,40}?)(?:가|는|은|이)\s")
    #: 「N년 M월 대량보유보고서」처럼 지정 기간에 문서가 없는 지분공시 요청.
    #: `HoldingDisclosureResolutionBackend`(agent/stage1_v1_holding_backend.py)가
    #: 이 모양에서 발행사·서식은 이미 증명하고 지정 기간에 문서 0건일 때만
    #: `corpus_coverage_unavailable`을 낸다 — 「질문을 못 알아들음」이 아니라
    #: 「그 기간에는 이 서식의 문서가 없음」이므로, 0·공란과 혼동하지 않는
    #: 문장을 앞세운다(issue #63 — RPC-008).
    _HOLDING_PERIOD_ABSENCE = re.compile(
        r"(?P<year>\d{4})\s*년\s*(?P<month>1[0-2]|0?[1-9])\s*월"
        r"[^,.?？。\n]{0,10}?대량\s*보유(?:상황)?\s*(?:보고서|공시)")

    def _outside_corpus_reason(
            self, question: str | None, *,
            codes: list[str] | None = None) -> str | None:
        """왜 답할 수 없는지를 요청 종류에 맞게 말한다.

        주가 4문항과 상장폐지 3문항이 같은 총칭 거절을 받고 있었다. 앞은
        자료에 그런 종류의 값이 없는 것이고, 뒤는 그런 공시가 없는 것이다.
        특히 상장폐지 쪽은 총칭 거절이 「있었을 수도 있는데 못 찾았다」로
        읽혀, 일어나지 않은 일을 열어 둔 채로 남긴다.

        없다고 말할 수 있는 것은 제공 자료 안에서의 부재뿐이다. 회사가
        상장폐지되지 않았다고까지 말하지 않는다.

        이 판단은 질문 표면의 정규식만으로 내린다 — 정본을 다시 조회하지
        않는다.  그래서 실제로는 정본에 해당 종류의 공시가 있는데 값 근거
        검증만 실패한 경우(``evidence_unavailable``)까지 "확인되지
        않습니다"로 단정하면 거짓이 된다(issue #121: 상장폐지 8건이 있는
        삼성전자에 이 총칭 거절이 나갔다).  ``evidence_unavailable``은
        Stage2가 receipt를 이미 하나로 확정했을 때만 나오는 코드라, 그
        확정 자체가 "공시가 있다"는 양성 증거다.  그런 코드가 같이 있으면
        이 자리는 물러나고, ``evidence_unavailable``의 고유 문구("값 후보는
        있으나 원문 근거 검증에 실패해 확정하지 않았습니다")만 남긴다 —
        "없다"와 "있는데 확정 못 했다"가 한 문단에서 서로를 뒤집지 않도록.
        """

        if codes is not None and any(
                code.split(":", 1)[0] == "evidence_unavailable"
                for code in codes):
            return None
        text = question or ""
        if self._PRICE_REQUEST.search(text):
            return ("제공된 자료는 DART 공시 문서라 주가·시세가 들어 있지 "
                    "않습니다. 공시에 적힌 값만 답할 수 있어 주가는 "
                    "확인해 드릴 수 없습니다.")
        if self._DELISTING_REQUEST.search(text):
            company = self._question_company_surface(text)
            subject = f"「{company}」의 " if company else ""
            return (f"제공된 공시자료에서 {subject}상장폐지 공시는 확인되지 "
                    "않습니다. 상장폐지가 있었다는 전제는 이 자료로 "
                    "뒷받침되지 않으며, 자료 밖 사실로 추정하지 않았습니다.")
        future_match = self._FUTURE_SUBMISSION_REQUEST.search(text)
        if future_match:
            year, month, day = future_match.groups()
            requested = (f"{int(year):04d}{int(month):02d}{int(day):02d}"
                         if day else f"{int(year):04d}{int(month):02d}01")
            if requested > CORPUS_CUTOFF:
                company = self._question_company_surface(text)
                if company is None:
                    leading = self._LEADING_SUBJECT_SURFACE.match(text)
                    if leading and not re.search(
                            r"\d{4}년", leading.group("name")):
                        company = leading.group("name")
                subject = (f"「{company}」{_josa(company, '이', '가')} "
                           if company else "")
                date_text = (f"{int(year)}년 {int(month)}월"
                             + (f" {int(day)}일" if day else ""))
                return (
                    f"{subject}{date_text}까지 제출한 공시를 기준으로 한 "
                    "요청이지만, 이 자료의 접수일은 "
                    f"{_CORPUS_CUTOFF_KO}까지이며 그 이후에 제출된 공시는 "
                    "들어 있지 않습니다.")
        # RPC-017 회귀: 이 모양 매치는 텍스트 표면만 본다 — 정본이 그 기간에
        # 실제로 문서를 가지고 있어도(예: 특별관계자 성명이 레거시 보안
        # 정책에서 인명으로 오분류돼 party 매칭이 실패한 경우) 걸려서
        # 「문서 자체가 코퍼스에 없습니다」라는 확정 사실을 지어내게 된다.
        # `HoldingDisclosureResolutionBackend`가 실제로 낸 사유가
        # `corpus_coverage_unavailable`일 때만 이 구체적인 부재 문장을
        # 쓴다 — 다른 사유(``unsupported_semantic_target`` 등)는 총칭
        # 거절 문구로 남겨 둔다.
        holding_match = (
            self._HOLDING_PERIOD_ABSENCE.search(text)
            if codes is not None and "corpus_coverage_unavailable" in codes
            else None)
        if holding_match:
            company = self._question_company_surface(text)
            subject = f"{company}의 " if company else ""
            date_text = f"{int(holding_match.group('year'))}년 {int(holding_match.group('month'))}월"
            return (
                f"제공된 자료에서 {subject}{date_text} 대량보유보고서를 찾지 못해 "
                "이 기간의 보고자·보유비율은 확인할 수 없습니다. 값이 0이거나 "
                "공란이어서가 아니라, 그 기간에 해당하는 문서 자체가 제공 자료에 "
                "없습니다.")
        return None

    #: 지시를 바꾸려는 문구의 시작. 이 앞이 사용자의 실제 요청이다.
    _INJECTION_MARKER = re.compile(
        r"(?:그리고\s*)?(?:이전|앞선|위의?)\s*(?:지시|규칙)")

    def _injection_prefix(self, question: str | None) -> str | None:
        """주입 문구 앞에 남는 실제 요청을 돌려준다.

        주입만 있는 질문(`G-U-004`)은 앞이 비어 있어 ``None`` 이 된다. 정상
        질문에 주입이 덧붙은 경우(`K-018`·`K-036`)만 사용자에게 되돌려 준다.
        """

        if not question:
            return None
        match = self._INJECTION_MARKER.search(question)
        if match is None:
            return None
        prefix = question[:match.start()].strip().rstrip(". ,·")
        # 회사명과 요청이 함께 있어야 「답변할 수 있는 질문」이라 말할 수 있다.
        if (len(prefix) < 10 or not re.search(r"[가-힣A-Za-z]{2,}", prefix)
                or not re.search(r"알려|보여|설명|정리|비교|얼마|무엇|어떻|인가|\?", prefix)):
            return None
        return prefix

    # ── 답변 ────────────────────────────────────────────────────────────
    def _answer(self, p, *, question: str | None = None) -> str:
        # 전제 정정 먼저
        lines = _premise_lines(p.premise_verdicts, include_true=True,
                               claims=p.claims)
        # 계산 결과가 있으면 그것이 답이다. operand는 "계산 근거"로 구분해 표기해 답과 혼동되지 않게 한다.
        claims = list(p.claims)
        if any(l.code.split(":", 1)[0] == "corp_not_in_universe"
               for l in p.limitations):
            # An incomplete entity set cannot support a cross-company result.
            # Preserve independently verified base values but suppress every
            # comparison/derivation conclusion.
            claims = [c for c in claims if c.operator not in {
                "argmax", "compare", "equal", "difference",
                "absolute_difference", "percent_change", "sum",
            }]
        derived = [c for c in claims if c.derived_from]
        context_derived = tuple(derived)
        calculation_derived = [c for c in derived if _is_calculation_claim(c)]
        label_overrides = _view_label_overrides(claims)
        directional_difference_ids = _directional_difference_ids(claims)
        quantitative_pairs = {
            frozenset(c.derived_from) for c in derived
            if c.operator in {
                "difference", "absolute_difference", "argmax",
                "discrete_from_cumulative",
            } and len(c.derived_from) >= 2
        }
        # A numeric delta already proves whether the two values differ.  Do
        # not follow it with the low-information sentence "두 값은 다릅니다".
        # Equality-only plans have no quantitative result and retain their
        # explicit same/different answer.
        derived = [
            c for c in derived
            if not (
                c.operator == "compare"
                and frozenset(c.derived_from) in quantitative_pairs
            )
        ]
        base = [c for c in claims if not c.derived_from]
        # 사건 질의는 기준일 상태가 사용자가 묻는 결론이고, 정정 행·계약 필드는
        # 그 결론의 상세 근거다.  도구가 증거 순서대로 claim을 내더라도 표시할
        # 때는 모든 기준일 상태를 먼저 모아 보여 준다.  claim 자체를 버리거나
        # 시점별 상태를 합치지 않으므로 긴 이력 질문의 완전성은 그대로다.
        event_summary = [c for c in base if _is_event_summary(c)]
        # An amount-only facilities question does not request the generic
        # absence-of-termination disclaimer.  Keep actual ending observations,
        # additional requested axes, ambiguous identities and all limitations.
        investment_amounts = [c for c in base
                              if re.fullmatch(r"투자\s*금액", c.label or "")]
        if (question and re.search(r"시설\s*투자", question)
                and re.search(r"투자\s*금액", question)
                and not re.search(r"상태|종료|해지|완료|진행|유효|기간|변경|정정|이력", question)
                and len(investment_amounts) == 1
                and event_summary
                and all(c.state == "no_termination_observed" for c in event_summary)
                and len(base) == len(event_summary) + 1):
            base = investment_amounts
            event_summary = []
        if event_summary:
            summary_ids = {c.output_id for c in event_summary}
            base = event_summary + [c for c in base
                                    if c.output_id not in summary_ids]
            if _is_state_only_event_question(question):
                state_table = _event_state_table(event_summary)
                if state_table:
                    lines.append(state_table)
                else:
                    lines.extend(self._claim_line(c) for c in event_summary)
                if _needs_termination_detail_answer(
                        p, question, event_summary, base):
                    details = _same_termination_receipt_details(
                        base, event_summary)
                    if details:
                        scalar_details = [
                            claim for claim in details
                            if claim.label != "효력발생조건"
                        ]
                        short_context = [
                            claim for claim in details
                            if claim.label == "효력발생조건"
                            and len(" ".join((claim.text or "").split())) <= 240
                        ]
                        details = scalar_details + short_context or details
                        lines.append("현재 해지 공시 기준 주요 내용:")
                        lines.extend(self._claim_line(c) for c in details)
                notes = self._public_notes(p, question=question)
                if notes:
                    _extend_public_qualifications(lines, notes)
                return "\n".join(x for x in lines if x)
        operand_ids = {o for c in calculation_derived for o in c.derived_from}
        claim_by_id = {claim.output_id: claim for claim in claims}
        ranked_operand_ids = {
            entry.output_id
            for claim in derived
            if claim.operator == "argmax" and claim.ranking
            for entry in claim.ranking
        }
        # 이슈 #38 concept_ratio — 지표명(주어)·계산식 문구·"계산 근거:" 줄을
        # 여기서 미리 만든다. 분자·분모의 원값은 .numerator/.denominator
        # 보조 claim(과, 있다면 같은 output_id 의 원 fact claim)으로 대신
        # 나오므로 아래에서 각각 한 번씩만 보이도록 숨긴다.
        concept_ratio_formulas: dict[str, str] = {}
        concept_ratio_basis_lines: dict[str, str] = {}
        concept_ratio_hidden_ids: set[str] = set()
        for rc in derived:
            if rc.operator != "concept_ratio":
                continue
            numerator = claim_by_id.get(f"{rc.output_id}.numerator")
            denominator = claim_by_id.get(f"{rc.output_id}.denominator")
            if numerator is None or denominator is None:
                continue
            subject, formula = _concept_ratio_subject(
                numerator, denominator, is_percent=(rc.raw_unit == "%"),
                question=question)
            label_overrides[rc.output_id] = subject
            if formula:
                concept_ratio_formulas[rc.output_id] = formula
            scale = self._mixed_operand_units((numerator, denominator))
            concept_ratio_basis_lines[rc.output_id] = (
                "계산 근거: "
                + self._operand_basis((numerator, denominator), scale=scale,
                                      label_overrides=label_overrides)
                + self._brief_citations((numerator, denominator)))
            concept_ratio_hidden_ids.update(
                (numerator.output_id, denominator.output_id))
            concept_ratio_hidden_ids.update(rc.derived_from)
        if concept_ratio_hidden_ids:
            derived = [c for c in derived
                       if c.output_id not in concept_ratio_hidden_ids]
        if derived:
            derived_start = len(lines)
            merged = self._period_change_sentence(
                derived, directional_difference_ids, label_overrides)
            if merged is not None:
                lines.append(merged)
                derived = [c for c in derived
                           if c.operator not in ("difference", "percent_change")]
            # 이슈 #118 ② — 연속한 파생 문장이 같은 회사로 시작하면 두 번째
            # 부터 회사명을 접는다.
            derived_previous_label: str | None = None
            for c in derived:
                line_index = len(lines)
                claim_label = label_overrides.get(c.output_id)
                lines.append(self._claim_line(
                    c,
                    directional_difference=(
                        c.output_id in directional_difference_ids),
                    label=claim_label,
                    ranking_sources=claim_by_id,
                    ratio_formula=concept_ratio_formulas.get(c.output_id),
                    previous_label=derived_previous_label,
                ))
                derived_previous_label = _normal_public_label(
                    claim_label or c.label)
                if c.operator == "concept_ratio":
                    basis = concept_ratio_basis_lines.get(c.output_id)
                    if basis:
                        lines.append(basis)
                        _demote_citations_to_basis(
                            lines, line_index, line_index + 1)
            operands = [c for c in base if c.output_id in operand_ids]
            # A ranked sidecar already prints every operand and its own
            # citation.  Repeating the source claims below makes the answer
            # longer and duplicates receipts without adding evidence.  A
            # concept_ratio operand is likewise already shown in its own
            # "계산 근거:" line above.
            others = [c for c in base if c.output_id not in operand_ids
                      and c.output_id not in ranked_operand_ids
                      and c.output_id not in concept_ratio_hidden_ids]
            if operands:
                discrete = next(
                    (c for c in calculation_derived
                     if c.operator == "discrete_from_cumulative"
                     and len(c.derived_from) == 2),
                    None,
                )
                by_id = {c.output_id: c for c in operands}
                if (discrete is not None
                        and all(output_id in by_id
                                for output_id in discrete.derived_from)):
                    left, right = (by_id[output_id]
                                   for output_id in discrete.derived_from)
                    scale = self._mixed_operand_units((left, right))
                    citations = self._brief_citations((left, right))
                    # 이슈 #64 — 4분기 단독 값의 재료(누적 구간, 보통 연간
                    # 값)로만 쓰인 원값을 "계산 근거:" 줄에만 남기면, 그
                    # 값을 따로 물은 질문(「연간 매출액과 4분기 단독 매출액을
                    # 각각 알려주고」)에서는 독립 문장이 아니라 계산 부속물로만
                    # 읽힌다. 그 값도 자기 문장으로 낸다 — 새 claim을 만들지
                    # 않고 이미 검증된 operand를 한 번 더 보여줄 뿐이다.
                    #
                    # 이슈 #115 — 질문이 그 값을 따로 요구하지 않았을 때도
                    # 항상 이 문장을 냈더니 같은 재료 값이 독립 문장과 계산
                    # 근거에 두 번 나왔다(DEV-FDR-027·028). 질문이 실제로
                    # 그 값도 따로 요구했을 때만(위 표면 신호) 남긴다.
                    if _question_requests_material_value_separately(question):
                        lines.append(self._claim_line(
                            left, label=label_overrides.get(left.output_id)))
                    lines.append(
                        "계산 근거: "
                        f"{self._brief(left, with_scale=scale, cite=False, label=label_overrides.get(left.output_id))} - "
                        f"{self._brief(right, with_scale=scale, cite=False, label=label_overrides.get(right.output_id))} = "
                        f"{_display_amount(discrete)}{citations}")
                else:
                    scale = self._mixed_operand_units(operands)
                    lines.append(
                        "계산 근거: "
                        + self._operand_basis(operands, scale=scale,
                                              label_overrides=label_overrides)
                        + self._brief_citations(operands))
                _demote_citations_to_basis(
                    lines, derived_start, derived_start + len(derived))
            # 이슈 #118 ② — 연속한 문장이 같은 회사로 시작하면 두 번째부터
            # 회사명을 접는다.
            others_previous_label: str | None = None
            for c in others:
                claim_label = label_overrides.get(c.output_id)
                lines.append(self._claim_line(
                    c, label=claim_label, previous_label=others_previous_label))
                others_previous_label = _normal_public_label(claim_label or c.label)
        else:
            parallel_axis = self._parallel_axis_sentence(base)
            if parallel_axis is not None:
                lines.append(parallel_axis)
                base = []
            shared, remaining = _shared_advisory_lines(base)
            emitted_shared = False
            emitted_event_detail_header = False
            # 이슈 #118 ② — 사이에 유의사항·이력 머리글이 끼면 그 지점에서
            # 접기를 다시 시작한다(끊긴 뒤 첫 문장은 항상 전체 라벨).
            base_previous_label: str | None = None
            for c in base:
                if (event_summary and c.output_id not in summary_ids
                        and not emitted_event_detail_header):
                    lines.append("세부 근거와 변경 이력:")
                    emitted_event_detail_header = True
                    base_previous_label = None
                if c.output_id in remaining:
                    if not emitted_shared:
                        receipts = sorted({
                            ct.rcept_no for other in base
                            if other.output_id in remaining
                            for ct in (other.citations or ())
                            if ct.rcept_no})
                        cite = (f"(근거: {', '.join('접수번호 ' + r for r in receipts)})"
                                if receipts else "")
                        lines.append(
                            "공시 유의사항(아래 공시들에 공통): "
                            + " ".join(f"- {b}" for b in shared) + f" {cite}".rstrip())
                        emitted_shared = True
                        base_previous_label = None
                    rest = remaining[c.output_id]
                    if not rest:
                        # 이 공시의 유의사항이 전부 공통 문구였다. 공통 문단의
                        # 접수번호 목록이 어느 공시들에 해당하는지도 이미 밝히므로
                        # 사용자를 위한 답에는 지시대상뿐인 반복 문장을 더하지 않는다.
                        base_previous_label = None
                        continue
                    lines.append(
                        f"{c.label}(이 공시에만): "
                        + " ".join(f"- {b}" for b in rest) + _cite(c))
                    base_previous_label = None
                    continue
                claim_label = label_overrides.get(c.output_id)
                lines.append(self._claim_line(
                    c, label=claim_label, previous_label=base_previous_label))
                base_previous_label = _normal_public_label(claim_label or c.label)
        # 이슈 #94 27 — 값만 있으면 좋아진 것인지 나빠진 것인지 알 수 없다.
        # 이 답변이 **이미 확정한** 전년 값이 있을 때만 변화를 한 줄 덧붙인다.
        year_over_year = _year_over_year_context(base, context_derived)
        if year_over_year:
            lines.append(year_over_year)
        # 한계·기본값 고지
        notes = self._public_notes(p, question=question)
        if notes:
            _extend_public_qualifications(lines, notes)
        return "\n".join(x for x in lines if x)

    @staticmethod
    def _public_notes(p, *, question: str | None) -> list[str]:
        """Return de-duplicated, user-safe qualification text once each."""

        public_limitations = [
            limitation for limitation in p.limitations
            if limitation.code.split(":", 1)[0] != "source_cross_check_partial"
        ]
        notes = render_safe_limitations(public_limitations, question=question)
        if (question and re.search(r"사업자\s*등록번호|법인\s*등록번호", question)
                and any(limitation.code == "personal_data_omitted"
                        for limitation in public_limitations)):
            generic = safe_limitation_message("personal_data_omitted")
            notes = [note for note in notes if note != generic]
            notes.append("사업자등록번호 등 식별번호는 이 서비스의 제공 제한 항목이므로 공개하지 않습니다. 공개된 보유비율과는 별도로 적용되는 제한입니다.")
        # 후보 접수번호를 나열하면 그 자체가 「특정할 수 없다」는 뜻을 이미
        # 담는다. 안전 문구까지 함께 남으면 같은 말을 두 번 하게 되므로,
        # 후보 목록을 붙이는 자리에서는 그 한계의 안전 문구를 지운다(#86-14).
        superseded_by_candidates: set[str] = set()
        for limitation in p.limitations:
            if limitation.code.split(":", 1)[0] == "source_cross_check_partial":
                continue
            if safe_limitation_message(limitation.code) is None:
                message = _REASON_KO.get(limitation.code.split(":")[0])
                if message:
                    notes.append(message)
                elif limitation.code in (
                        "slot_not_confirmed", "narrative_record_budget_exhausted",
                        "narrative_result_budget_exhausted",
                        "narrative_source_roundtrip_mismatch",
                        "narrative_cell_not_found", "narrative_fanout_limit"):
                    if (limitation.code == "slot_not_confirmed"
                            and _is_business_content_only_gap(limitation.detail)
                            and _business_segment_table_already_shown(p.claims)):
                        continue
                    notes.append(limitation.detail)
            # The affected receipts are evidence for the *ambiguity itself*,
            # not an inferred edge from either candidate to the later event.
            # Exposing both lets a reader audit why the answer deliberately
            # refuses to select one origin.  Do this only for the two typed
            # event-identity limitations, never for arbitrary affected docs.
            code = limitation.code.split(":", 1)[0]
            if code in {"ambiguous_event_identity", "ambiguous_event_origin"}:
                origins = list(dict.fromkeys(
                    receipt for receipt in limitation.affected_doc_ids
                    if re.fullmatch(r"\d{14}", receipt or "")
                ))
                if len(origins) >= 2:
                    notes.append(
                        "원공시 후보 접수번호: " + ", ".join(origins) + ".")
                    superseded_by_candidates.add(code)
        if superseded_by_candidates:
            hidden = {
                safe_limitation_message(code)
                for code in superseded_by_candidates
            }
            notes = [note for note in notes if note not in hidden]
        # Execution defaults such as ``view=restated`` belong in the trace,
        # not at the end of every ordinary answer.  Answer-critical public
        # definitions are derived below from the typed claim scope.
        notes.extend(required_definition_messages(p, question))
        return list(dict.fromkeys(note for note in notes if note))

    @staticmethod
    def _mixed_operand_units(operands) -> bool:
        """피연산자들이 서로 다른 원문 단위로 공시됐는가.

        원문 단위는 근거이므로 바꾸지 않는다.  다만 한 줄에 ``1,878,732백만원``
        과 ``1,734,697,253,045원`` 을 나란히 두면 어느 쪽이 큰지 읽는 사람이
        암산으로 환산해야 한다.  그럴 때만 공통 배수 표기를 함께 적는다.
        """

        units = {(c.raw_unit or "").strip()
                 for c in operands if c.value_text and (c.raw_unit or "").strip()}
        return len(units) > 1

    @staticmethod
    def _combined_cite(claims) -> str:
        """여러 claim 의 근거를 하나로. 같은 문서면 접수번호가 겹쳐도 한 번만 싣는다."""

        parts: list[str] = []
        seen: set[str] = set()
        for claim in claims:
            for citation in (claim.citations or ()):
                if not citation.rcept_no or citation.rcept_no in seen:
                    continue
                seen.add(citation.rcept_no)
                parts.append(f"접수번호 {citation.rcept_no}")
                if len(parts) >= 4:
                    break
        return f" (근거: {'; '.join(parts)})" if parts else ""

    def _parallel_axis_sentence(self, base) -> "str | None":
        """N개 값이 라벨 한 토큰만 다르면 한 문장으로 합친다. 아니면 ``None``.

        「기아 2025년 연결 매출원가는 …입니다.」「기아 2025년 별도 매출원가는
        …입니다.」두 줄, 「삼성전자 2025년 연결 매출액은 …입니다.」「…영업이익은
        …입니다.」「…당기순이익은 …입니다.」세 줄 — 둘 다 같은 주어를 축 하나
        (범위·개념)만 바꿔 되풀이한다. `_period_change_sentence`(전기 대비 두
        줄을 한 문장으로, #94 26)와 같은 이유이고, 축이 시간이 아닐 뿐이다.

            기아 2025년 매출원가는 연결 152조 376억 3,600만원,
            별도 63조 4,485억 7,600만원입니다.

            삼성전자 2025년 연결 매출액은 333조 6,059억 3,800만원,
            영업이익은 43조 6,010억 5,100만원,
            당기순이익은 45조 2,068억 500만원입니다.

        **라벨을 공백으로 자른 토큰이 개수가 같고 정확히 한 자리만 다를 때만
        합친다.** 그 자리가 어디든(중간의 「연결/별도」든 끝의 개념명이든)
        상관없다 — 나머지 모든 자리가 글자까지 같다는 것 자체가 같은 사실을
        축 하나로만 나눈 것이라는 증거다. 다른 자리가 하나라도 다르면(회사·
        기간 등) 진짜 두 개의 다른 사실이라 합치면 안 된다.

        **다른 자리가 무엇이냐에 따라 문장 모양이 갈린다.** 「연결」「별도」
        「개별」은 그 자체로 문장 주어가 될 수 없는 수식어다 — 주어에서 빼고
        각 값 앞에 붙인다(위 첫 예). 그 밖(개념명 등)은 그 자체로 완결된
        명사라 자기 조사를 받을 수 있다 — 첫 값만 주어에 포함하고 나머지는
        각자 「는」을 받는다(위 둘째 예). 하나로 합치면(「연결은 매출액…」)
        「연결」이 문장 주어처럼 읽혀 어색해진다.

        **다른 자리가 시점(연도·분기·「말」)이면 아예 합치지 않는다.** 시점만
        다른 두 값을 파생 없이 나란히 두면 암묵적인 증감 비교로 읽힌다 —
        `scope_mismatch`·`period_length_mismatch` 가 막는 것과 같은 위험이다.
        실제 시간 비교는 `_period_change_sentence` 가 파생과 함께 답한다.

        파생값(비율·증감 등)에는 적용하지 않는다 — 이미 다른 문장 규칙이
        그 값을 맡는다. `_PARALLEL_AXIS_LIMIT` 를 넘으면 한 문장이 오히려
        읽기 어려워지므로 합치지 않는다.
        """

        if not 2 <= len(base) <= _PARALLEL_AXIS_LIMIT or any(
                claim.operator is not None or claim.value_text is None
                or claim.state in _STATE_MARKED_AMOUNT
                or _is_free_share_allocation_ratio_claim(claim)
                for claim in base):
            # 값이 있으면서 `state` 가 `explicit_zero`/`decreased`/`increased`
            # 인 claim 은 `_claim_line` 이 그 값만의 특별한 문장을 쓴다 —
            # 예를 들어 `explicit_zero` 는 "명시된 0원"이라고 못박아 「확인
            # 불가」로 잘못 읽히지 않게 한다(이슈 #63, RPC-006). 이 함수는
            # 그 문장 규칙을 모르고 `_display_amount` 로만 값을 적으므로,
            # 그런 claim 이 섞이면 특별한 뜻을 조용히 지우게 된다. 함께 있는
            # 다른 값까지 전부 손대지 않는다. (다른 state — 예: 회사명을
            # 담는 용도 — 는 `value_text` 가 있으면 `_claim_line` 의 렌더를
            # 바꾸지 않으므로 여기서 막을 이유가 없다.)
            return None
        token_lists = [claim.label.split() for claim in base]
        lengths = {len(tokens) for tokens in token_lists}
        if len(lengths) != 1 or not lengths.pop():
            return None
        diff_positions = [
            index for index in range(len(token_lists[0]))
            if len({tokens[index] for tokens in token_lists}) > 1
        ]
        if len(diff_positions) != 1 or diff_positions[0] == 0:
            # 첫 토큰은 이 시스템의 라벨 관례에서 항상 회사(주체)다. 회사가
            # 다르면 「같은 사실을 축 하나로 나눈 것」이 아니라 서로 다른
            # 주체의 값이다 — 병렬조회 답 형식이 따로 있다.
            return None
        pos = diff_positions[0]
        distinguishers = [tokens[pos] for tokens in token_lists]
        if len(set(distinguishers)) != len(distinguishers):
            return None
        if any(_PERIOD_LIKE_TOKEN.search(word) for word in distinguishers):
            # 시점(연도·분기 등)이 유일한 차이면 합치지 않는다. 파생 없이
            # 나란히 두면 「같은 것을 시간만 다르게 본 값」으로 읽혀
            # 암묵적인 증감 비교를 만든다 — `scope_mismatch`·
            # `period_length_mismatch` 가 막는 것과 같은 위험이다. 실제
            # 시간 비교는 `_period_change_sentence` 가 파생과 함께 답한다.
            return None
        citation = self._combined_cite(base)
        if all(word in ("연결", "별도", "개별") for word in distinguishers):
            subject_tokens = list(token_lists[0])
            del subject_tokens[pos]
            subject = _normal_public_label(" ".join(subject_tokens))
            if not subject:
                return None
            values = ", ".join(
                f"{distinguishers[index]} {_display_amount(claim)}"
                for index, claim in enumerate(base))
            return (f"{subject}{_josa(subject, '은', '는')} {values}"
                    f"입니다.{citation}")
        # 개념명처럼 그 자체로 완결된 명사는 주어에 포함해 자기 조사를 받는다.
        first_subject = _normal_public_label(base[0].label)
        if not first_subject:
            return None
        parts = [f"{first_subject}{_josa(first_subject, '은', '는')} "
                 f"{_display_amount(base[0])}"]
        for distinguisher, claim in zip(distinguishers[1:], base[1:]):
            parts.append(
                f"{distinguisher}{_josa(distinguisher, '은', '는')} "
                f"{_display_amount(claim)}")
        return f"{', '.join(parts)}입니다.{citation}"

    @staticmethod
    def _brief_citations(claims) -> str:
        """Render one ordered receipt union for a calculation-basis line."""

        receipts = list(dict.fromkeys(
            citation.rcept_no
            for claim in claims
            for citation in (claim.citations or ())
            if citation.rcept_no
        ))
        return (f"(접수번호 {', '.join(receipts)})" if receipts else "")

    @staticmethod
    def _fold_shared_operand_labels(
            labels: list[str], values: list[str]) -> str | None:
        """피연산자들이 공유하는 말을 앞에 한 번만 두고 다른 것만 나열한다.

        계산 근거는 같은 라벨을 피연산자 수만큼 되풀이했다.  216자짜리 답에
        「SK하이닉스 … 연결 이익잉여금」이 세 번 나오고 실제로 다른 것은
        날짜뿐이었다.

            전: SK하이닉스 2025-12-31 기준 연결 이익잉여금 106조 … /
                SK하이닉스 2024-12-31 기준 연결 이익잉여금 65조 …
            후: SK하이닉스 연결 이익잉여금 — 2025-12-31 기준 106조 … /
                2024-12-31 기준 65조 …

        낱말 단위로 공통 앞·뒤를 뽑고 가운데 다른 부분만 값 앞에 남긴다.
        「기준」은 날짜에 붙는 말이라 공통 꼬리의 머리에 남으면 문장이
        깨지므로(「SK하이닉스 기준 연결 …」) 각 날짜 뒤로 돌려보낸다.

        접는 이득이 없거나(공통이 한 낱말 이하) 어느 한쪽이 통째로 공통이면
        건드리지 않고 ``None`` 을 돌려 종전 표기를 쓴다.
        """

        if len(labels) < 2 or len(labels) != len(values):
            return None
        token_rows = [label.split() for label in labels]
        if any(not row for row in token_rows):
            return None
        head = 0
        while (head < min(len(row) for row in token_rows)
               and len({row[head] for row in token_rows}) == 1):
            head += 1
        tail = 0
        while (tail < min(len(row) - head for row in token_rows)
               and len({row[len(row) - 1 - tail] for row in token_rows}) == 1):
            tail += 1
        prefix = token_rows[0][:head]
        suffix = token_rows[0][len(token_rows[0]) - tail:] if tail else []
        middles = [row[head:len(row) - tail] for row in token_rows]
        if any(not middle for middle in middles):
            return None
        if suffix and suffix[0] == "기준":
            suffix = suffix[1:]
            middles = [middle + ["기준"] for middle in middles]
        # 공통 부분이 지표 이름으로 끝나야 그것만 떼어 주어로 읽힌다.  다른
        # 것이 지표 자체일 때는(영업이익 ÷ 매출액) 공통이 「가온 2031년 연결」
        # 처럼 수식어에서 끊겨, 접는 것이 오히려 문장을 깬다.
        if not suffix:
            return None
        shared = prefix + suffix
        if len(shared) < 2:
            return None
        return (" ".join(shared) + " — " + " / ".join(
            " ".join(middle) + " " + value
            for middle, value in zip(middles, values)))

    @staticmethod
    def _brief_parts(c, *, with_scale: bool = False,
                     label: str | None = None) -> tuple[str, str]:
        """계산 근거 축약 표기의 라벨과 값을 따로 돌려준다."""
        value = (_display_amount(c, with_scale=with_scale)
                 if c.value_text else (c.text or c.state or ""))
        return _normal_public_label(label or c.label), value

    def _brief(self, c, *, with_scale: bool = False, cite: bool = True,
               label: str | None = None) -> str:
        """계산 근거용 축약 표기 — 값과 근거만."""
        public_label, v = self._brief_parts(c, with_scale=with_scale, label=label)
        rc = c.citations[0].rcept_no if c.citations and c.citations[0].rcept_no else ""
        return f"{public_label} {v}" + (f"(접수번호 {rc})" if cite and rc else "")

    def _operand_basis(self, operands, *, scale: bool,
                       label_overrides: dict) -> str:
        """계산 근거 본문 — 공통 라벨이 있으면 한 번만 쓴다."""
        parts = [self._brief_parts(
            c, with_scale=scale, label=label_overrides.get(c.output_id))
            for c in operands]
        folded = self._fold_shared_operand_labels(
            [label for label, _value in parts],
            [value for _label, value in parts])
        if folded is not None:
            return folded
        return " / ".join(f"{label} {value}" for label, value in parts)

    def _period_change_sentence(self, derived, directional_ids,
                                label_overrides) -> str | None:
        """증감액과 증감률을 한 문장으로 합친다.

        전기 대비 변화는 두 줄로 나갔다.

            기아 2025년 연결 영업이익과 기아 2024년 연결 영업이익의 차이는
            3조 5,889억 9,100만원(감소)입니다.
            기아 2025년 연결 영업이익 전기 대비 증감률은 -28.33%(감소)입니다.

        같은 긴 라벨을 두 번 읽어야 하고, 방향은 세 번(감소·-·감소) 나온다.
        한 문장이면 주어가 한 번이고 방향도 서술어가 한 번 말한다.

            기아 2025년 연결 영업이익은 2024년 대비 3조 5,889억 9,100만원
            (28.33%) 감소했습니다.

        비교 대상 기간은 difference 라벨이 이미 적고 있다(「…2025년…과
        …2024년…의 차이」). 「전기」는 공시 원문의 말이라 읽는 사람이 어느
        해인지 바로 알 수 없으므로 실제 연도로 바꾼다(이슈 #94 26). 대상을
        하나로 못 정하면 「전기 대비」를 그대로 둔다.

        같은 피연산자 쌍에 대한 방향성 difference 와 percent_change 가 나란히
        있을 때만 합친다. 절대차이·순위·분기 단독 계산은 방향이 없거나 뜻이
        달라 대상이 아니다.
        """

        if len(derived) != 2:
            return None
        by_op = {c.operator: c for c in derived}
        if set(by_op) != {"difference", "percent_change"}:
            return None
        diff, rate = by_op["difference"], by_op["percent_change"]
        if frozenset(diff.derived_from) != frozenset(rate.derived_from):
            return None
        if diff.output_id not in directional_ids:
            return None
        # 이슈 #62 — %p(절대 변화)와 상대 증감률(%)을 한 괄호 문장으로
        # 합치면 「0.01%p(0.05%)」처럼 두 단위가 나란히 붙어 오히려 %와 %p를
        # 헷갈리기 쉽다. 이 조합은 대신 별도 두 문장으로 두어(아래 개별
        # `_claim_line`) 「…의 차이는 0.01%p(감소)」·「…상대 증감률은
        # -0.05%(감소)」처럼 단위마다 이름을 밝힌다.
        if diff.raw_unit == "%p":
            return None
        direction = rate.text
        if direction not in ("증가", "감소"):
            return None

        subject = _normal_public_label(
            label_overrides.get(rate.output_id) or rate.label)
        subject = _PERIOD_CHANGE_SUBJECT.sub("", subject).strip()
        if not subject:
            return None
        # 이슈 #94 26 — 「전기 대비」는 공시 원문의 말이라 읽는 사람이 어느
        # 해와 견준 것인지 바로 알 수 없다. difference 의 라벨이 두 피연산자를
        # 모두 적으므로 거기서 비교 대상을 꺼낸다. 하나로 못 정하면 지어내지
        # 않고 원래 말을 그대로 둔다.
        against = _explicit_base_period(subject, str(diff.label or ""))
        basis = f"{against} 대비" if against else "전기 대비"
        amount = _display_amount(diff)
        if amount.startswith("-"):
            amount = amount[1:]
        # 서술어가 방향을 말하므로 비율의 부호는 같은 말을 한 번 더 하는 것이다.
        percent = (rate.value_text or "").lstrip("-+")
        citations = _cite(rate) or _cite(diff)

        # 이익에서 손실로(또는 그 반대로) 넘어갔으면 「감소」로 쓸 수 없다.
        # 「당기순손실은 … 감소했습니다」는 손실이 줄었다는 뜻으로 읽히는데,
        # 실제로는 이익에서 손실로 돌아선 것이라 정반대다. 부호가 바뀐 변화를
        # 가리키는 말은 적자·흑자 전환이다. 주어에서도 한쪽 극성을 지운다 —
        # 「당기순손익」이라야 두 기간을 함께 가리킬 수 있다.
        pair = str(diff.label or "")
        if "손실" in pair and "이익" in pair:
            turned = "적자" if "손실" in subject else "흑자"
            moved = "줄어" if direction == "감소" else "늘어"
            subject = _POLARITY_IN_SUBJECT.sub("손익", subject)
            return (f"{subject}{_josa(subject, '은', '는')} {basis} "
                    f"{amount}({percent}%) {moved} {turned} 전환했습니다."
                    f"{citations}")
        return (f"{subject}{_josa(subject, '은', '는')} {basis} "
                f"{amount}({percent}%) {direction}했습니다.{citations}")

    def _claim_line(self, c, *, directional_difference: bool = False,
                    label: str | None = None,
                    ranking_sources: dict[str, object] | None = None,
                    ratio_formula: str | None = None,
                    previous_label: str | None = None) -> str:
        public_label = _normal_public_label(label or c.label)
        # 이슈 #118 ② — 같은 답의 바로 앞 문장이 같은 회사(라벨 첫 낱말)로
        # 시작했으면 이 문장에서는 그 회사명을 생략한다. `subject` 만
        # 화면에 쓰고, `public_label` 은 아래 정규식 판정(차이·현금흐름
        # 항목 판별 등)에 그대로 쓴다 — 접힌 낱말이 뒤쪽 판정을 바꾸지
        # 않아야 하기 때문이다.
        subject = fold_repeated_claim_subject(previous_label, public_label)
        if c.operator == "concept_ratio":
            # 이슈 #38 — 「<지표명>은 12.34%입니다(계산: 분자 ÷ 분모).」 형태.
            # 분자·분모 원값은 별도 "계산 근거:" 줄로만 나오므로 여기서는
            # 비율값과, 이름 있는 비율일 때만 계산식 문구를 덧붙인다.
            amount = f"{c.value_text or ''}{c.raw_unit or ''}"
            formula = f"(계산: {ratio_formula})" if ratio_formula else ""
            return (f"{subject}{_josa(subject, '은', '는')} "
                    f"{amount}입니다{formula}.{_cite(c)}")
        # CorrectionTool binds this text to the two independently verified
        # before/after cells.  Prefer the complete typed transition over the
        # numeric `value_text` shortcut, which would otherwise hide the before
        # value and signed difference in deterministic fallback answers.
        if c.operator == "correction_diff":
            # 정정 전후가 통째로 표일 때 양쪽을 다 실으면 같은 행이 두 번 나오고
            # 바뀐 한 줄이 그 안에 묻힌다. 다른 정정 경로는 이미 변경 구간으로
            # 줄이고 있었는데 이 렌더만 원문을 그대로 냈다.
            return (f"{subject}: "
                    f"{_compact_correction_change(c.text or '')}{_cite(c)}")
        if c.operator == "argmax":
            if c.ranking:
                rows = []
                for entry in c.ranking:
                    value = (
                        f" {_display_amount(entry, with_scale=False)}"
                        if entry.value_text else "")
                    # RankingEntry intentionally stays a compact derivation
                    # sidecar and therefore has no independent citations.
                    # Its source operand is the authoritative receipt owner.
                    # Cite the displayed company/value at that operand, not
                    # only the aggregate argmax claim, so a public ranking
                    # remains auditable even when every company filed a
                    # different report.
                    source = ((ranking_sources or {}).get(entry.output_id)
                              if ranking_sources else None)
                    rows.append(
                        f"{entry.rank}위 {entry.label}{value}"
                        + (_cite(source) if source is not None else ""))
                # A malformed legacy payload can have a ranking sidecar with
                # no recoverable operand receipts.  Retain the argmax claim's
                # own citation as a conservative fallback, but avoid adding
                # it when every row already has its source receipt.
                suffix = ("" if all(
                    (ranking_sources or {}).get(entry.output_id) is not None
                    and _cite((ranking_sources or {})[entry.output_id])
                    for entry in c.ranking)
                    else _cite(c))
                ranking_text = f"비교 순위: {'; '.join(rows)}.{suffix}"
                # ``direction=minimum``에서 1위는 "가장 작은 값"의
                # 주체다. 순위표만 내면 일반적인 1위(최댓값)로 읽힐 수
                # 있으므로 결론의 방향을 먼저 밝힌다(CG-053, #233).
                if c.direction == "minimum":
                    winner = c.ranking[0]
                    amount = (
                        f"이며, 값은 {_display_amount(winner, with_scale=False)}"
                        if winner.value_text else "")
                    return (
                        f"비교한 값이 가장 작은 기업은 {winner.label}{amount}입니다. "
                        f"{ranking_text}")
                return ranking_text
            # 이슈 #124 — direction="minimum"이면 승자는 최솟값 주체다.
            comparative = "더 작습니다" if c.direction == "minimum" else "더 큽니다"
            return f"비교 결과 {c.state}{_josa(c.state, '이', '가')} {comparative}.{_cite(c)}"
        if c.operator == "compare":
            return f"두 값은 {'같습니다' if c.state == 'same' else '다릅니다'}.{_cite(c)}"
        if c.operator == "percent_change":
            return f"{subject}{_josa(subject, '은', '는')} {c.value_text}%({c.text})입니다.{_cite(c)}"
        if c.operator == "difference" and directional_difference:
            amount = _display_amount(c)
            sign = _claim_numeric_sign(c)
            magnitude = amount[1:] if sign == -1 and amount.startswith("-") else amount
            # 라벨이 ``…의 차이`` 로 끝나면 주어는 「차이」다. 차이는 증가하지
            # 않으므로 ``차이는 32조원 증가했습니다`` 는 주술이 어긋난 비문이다.
            # 같은 답변의 형제 문장(``증감률은 10.88%(증가)입니다``)과 같은
            # 명사형으로 맞춘다. 판정은 접기 전 원 라벨(`public_label`)로
            # 한다 — 접힌 낱말이 이 판정을 바꾸면 안 된다.
            if _DIFFERENCE_SUBJECT.search(public_label):
                if sign == 0:
                    return (f"{subject}{_josa(subject, '은', '는')} "
                            f"없습니다.{_cite(c)}")
                if sign in (1, -1):
                    way = "증가" if sign == 1 else "감소"
                    return (f"{subject}{_josa(subject, '은', '는')} "
                            f"{magnitude}({way})입니다.{_cite(c)}")
            elif sign == -1:
                return f"{subject}{_josa(subject, '은', '는')} {magnitude} 감소했습니다.{_cite(c)}"
            elif sign == 1:
                return f"{subject}{_josa(subject, '은', '는')} {magnitude} 증가했습니다.{_cite(c)}"
            elif sign == 0:
                return f"{subject}{_josa(subject, '은', '는')} 변동 없습니다.{_cite(c)}"
            # Malformed numeric text is not reinterpreted.  Preserve the
            # existing amount surface and let upstream typed validation decide.
            return f"{subject}{_josa(subject, '은', '는')} {amount}입니다.{_cite(c)}"
        if c.operator in (
                "difference", "absolute_difference", "discrete_from_cumulative",
                "sum"):
            return f"{subject}{_josa(subject, '은', '는')} {_display_amount(c)}입니다.{_cite(c)}"
        if c.state in _STATE_KO and not c.value_text:
            body = c.text or _STATE_KO[c.state]
            return f"{subject}: {body}{_cite(c)}"
        if c.value_text:
            amount = _display_amount(c)
            # 무상증자의 ``NEW_ASN_CST``는 기존 주식 1주를 분모로 하는
            # 배정 비율이다. 원문 단위 "주"만 출력하면 "3주"가 총수량처럼
            # 보이므로, 분모와 새 주식 수를 함께 적는다(CG-006, #233).
            if _is_free_share_allocation_ratio_claim(c):
                return (
                    f"{subject}{_josa(subject, '은', '는')} 기존 주식 1주당 "
                    f"새 주식 {amount}입니다.{_cite(c)}")
            if c.state == "explicit_zero":
                # schema 1.8 `value_status=explicit_zero` proves the source
                # itself wrote this value as ``0`` — a disclosed fact, not a
                # blank/withheld/absent field.  Say so explicitly so a
                # wording pass never quietly drops it as "확인 불가" (issue
                # #63 — RPC-006).
                return (f"{subject}{_josa(subject, '은', '는')} "
                        f"명시된 {amount}입니다.{_cite(c)}")
            if c.state in {"decreased", "increased"}:
                direction = "감소" if c.state == "decreased" else "증가"
                return (
                    f"{subject}{_josa(subject, '은', '는')} "
                    f"{amount} {direction}했습니다.{_cite(c)}"
                )
            if (_claim_numeric_sign(c) == -1
                    and "현금유출" not in public_label
                    and not c.derived_from
                    and ("현금흐름" in public_label
                         or _CASH_PAYMENT_ITEM.search(public_label))):
                # 현금흐름표는 유출을 괄호로 적는다. 그것을 그대로 음수로
                # 옮기면 ``이자의지급은 -893억원입니다`` 처럼 일반 독자가
                # 읽을 수 없는 문장이 된다 — 지급이 음수라는 말은 뜻이 서지
                # 않는다. 부호 대신 방향을 말로 밝힌다. 판정은 원 라벨로,
                # 표시는 접힌 주어(`subject`)로 한다.
                magnitude = amount[1:] if amount.startswith("-") else amount
                flow = "순유출" if "현금흐름" in public_label else "유출"
                return (f"{subject}{_josa(subject, '은', '는')} "
                        f"{magnitude} {flow}입니다.{_cite(c)}")
            return f"{subject}{_josa(subject, '은', '는')} {amount}입니다.{_cite(c)}"
        if c.text:
            # NarrativeTool이 구조 블록 단위와 최대 크기를 이미 보장한다. 여기서 문자 수로
            # 다시 자르면 표 행·목록 항목이 깨지므로 prompt 전송 경계만 제거하고 온전히 표시한다.
            t = _display_text(c.text)
            if "작성기준일" in re.sub(r"\s+", "", public_label) and re.fullmatch(r"\d{8}", t):
                t = _public_date(t)
            return f"{subject}: {t}{_cite(c)}"
        return f"{subject}{_cite(c)}"

    def _claim_lines(self, claims, *,
                     label_overrides: dict[str, str] | None = None) -> list[str]:
        """Render consecutive claims, folding a repeated leading subject.

        이슈 #118 ② — 호출자가 한 답변 안에서 연이어 보여 줄 claim 목록을
        그대로 넘기면, 바로 앞 문장과 라벨 첫 낱말(회사)이 같은 문장부터는
        그 낱말을 생략한다. `_claim_line` 자체는 상태가 없으므로(다른
        자리에서도 그대로 쓰이므로) 그 접기는 여기서 문장 순서를 따라가며
        `previous_label` 로 넘긴다.
        """

        lines: list[str] = []
        previous_label: str | None = None
        for claim in claims:
            label = (label_overrides or {}).get(claim.output_id)
            lines.append(self._claim_line(
                claim, label=label, previous_label=previous_label))
            previous_label = _normal_public_label(label or claim.label)
        return lines


_ADVISORY_LABEL = re.compile(r"^(?:효력발생조건|관련\s*중요사항\(.*\))$")
_ADVISORY_BULLET = re.compile(r"(?:^|\s)-\s+")


def _advisory_bullets(text: str) -> list[str]:
    """공시 유의사항 문단을 원문 불릿 그대로 쪼갠다."""

    return [b.strip() for b in _ADVISORY_BULLET.split(text or "") if b.strip()]


def _shared_advisory_lines(claims: list) -> tuple[list[str], dict[str, list[str]]]:
    """여러 공시가 되풀이하는 유의사항 불릿과, 공시별로 다른 불릿을 갈라준다.

    원공시와 그 정정공시들은 같은 유의사항 문단을 거의 그대로 다시 싣는다.  그
    문단을 공시마다 통째로 보여주면 한 답변에 같은 문장이 네다섯 번 나온다.
    반대로 겹치는 줄을 그냥 지우면 어느 공시가 무엇을 담았는지가 사라진다 —
    이 계약들은 「1월에는 계약금액이 유보, 4월에는 공개」처럼 **같은 항목의
    문구가 시점마다 달라지는 것**이 답인 경우가 많다.

    그래서 문장은 한 글자도 바꾸지 않고 묶는 방식만 바꾼다.  둘 이상의 공시가
    공유하는 불릿은 앞에 한 번 모으고, 각 공시에는 그 공시에만 있는 불릿을
    남긴다.  무엇이 같고 무엇이 달라졌는지가 오히려 드러난다.
    """

    advisory = [c for c in claims if _ADVISORY_LABEL.match(c.label or "")]
    if len(advisory) < 2:
        return [], {}
    per_claim = {c.output_id: _advisory_bullets(c.text or "") for c in advisory}
    counts: dict[str, int] = {}
    for bullets in per_claim.values():
        for bullet in dict.fromkeys(bullets):
            counts[bullet] = counts.get(bullet, 0) + 1
    shared = [b for b in dict.fromkeys(
        bullet for bullets in per_claim.values() for bullet in bullets)
        if counts[b] >= 2]
    if not shared:
        return [], {}
    shared_set = set(shared)
    remaining = {
        output_id: [b for b in bullets if b not in shared_set]
        for output_id, bullets in per_claim.items()
    }
    return shared, remaining


def _display_text(text: str) -> str:
    boundary = {PROMPT_DATA_BEGIN, PROMPT_DATA_END}
    # Transport fences must never reach the public answer, even if an upstream
    # producer placed a marker beside other text instead of on its own line.
    # This removes protocol tokens only; the already prompt-safe body remains
    # intact and is never length-truncated here.
    value = (text or "").replace(PROMPT_DATA_BEGIN, "").replace(PROMPT_DATA_END, "")
    lines = [line for line in value.splitlines() if line.strip() not in boundary]
    return collapse_padded_headings("\n".join(lines).strip())
