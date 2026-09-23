"""Deterministic public definitions for answer-critical financial concepts."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


#: 질문이 실제로 이 약어를 썼는가. 쓰지 않았다면 답변도 꺼내지 않는다.
_ASKED_IN_CAPEX = re.compile(r"capex", re.IGNORECASE)

#: 질문이 이 개념을 전문용어로 지목했는가. 지목했다면 풀이가 질문을 되읊는
#: 것이 된다 — 「연결현금흐름표상 유형자산 취득 현금유출액은?」에 「여기서
#: 말한 금액은 연결현금흐름표에 적힌 유형자산 취득 현금유출액…」을 붙이는 꼴.
_ASKED_IN_CONCEPT_TERMS = re.compile(r"유형자산|현금흐름표|capex", re.IGNORECASE)


def capex_ppe_definition(scope: str | None,
                         question: str | None = None) -> str:
    """Return the CAPEX notice for the resolved statement scope.

    CAPEX is a typed financial fact, so its public definition must retain the
    same consolidated/separate axis as the selected fact.  In particular, a
    separate-financial-statement answer must not acquire a consolidated-note
    qualifier merely because the concept is usually queried on a CFS basis.

    질문이 CAPEX 라는 말을 쓰지 않았으면 답변도 쓰지 않는다. 「설비를
    사들이는 데 쓴 돈을 알려줘」라고 물은 사람에게 나오지도 않은 영어 약어를
    새로 던지고 그것을 정의하는 꼴이었다.
    """

    statement = {"CFS": "연결현금흐름표", "SFS": "별도현금흐름표"}.get(
        scope, "해당 현금흐름표")
    subject = ("여기서 CAPEX는" if _ASKED_IN_CAPEX.search(question or "")
               else "여기서 말한 금액은")
    return (
        f"{subject} {statement}에 적힌 유형자산 취득 현금유출액, 즉 "
        "설비·장비 등을 사는 데 실제로 나간 돈을 뜻합니다. 회사가 발표한 모든 "
        "설비투자 계획과 같은 금액은 아닙니다.")


CAPEX_PPE_DEFINITION = capex_ppe_definition("CFS")


def _requested_net_loss_definition(payload, question: str | None) -> str | None:
    """Explain an explicitly requested loss term only for a direct negative fact."""
    query = question or ""
    if ("순손실" not in query or not re.search(r"뜻|의미|설명", query)
            or re.search(r"숫자만|값만|설명\s*없이|설명은\s*빼", query)):
        return None
    claims = tuple(getattr(payload, "claims", ()))
    if len(claims) != 1:
        return None
    claim = claims[0]
    if (getattr(claim, "operator", None) or getattr(claim, "derived_from", ())
            or getattr(claim, "canonical_unit", None) != "원"
            or not re.search(r"당기순(?:이익|손실|손익)$", getattr(claim, "label", ""))):
        return None
    try:
        value = Decimal(str(getattr(claim, "canonical_value", None)))
    except (InvalidOperation, ValueError):
        return None
    if not value.is_finite() or value >= 0:
        return None
    return "순손실은 해당 기간의 최종 손익이 적자라는 뜻입니다."


def required_definition_messages(payload,
                                 question: str | None = None) -> list[str]:
    """Return typed scope notices that must survive wording composition.

    ``applied_defaults`` is the primary authority.  A claim-label fallback
    closes the execution-only gap where a derived comparison consumes
    ``capex_ppe`` facts but the tool note was not carried into the payload.
    It deliberately requires the resolved *type-PPE acquisition cash outflow*
    surface, so generic investment or intangible-asset questions do not gain
    an unrelated CAPEX definition.
    """
    values = [
        value.strip()
        for value in getattr(payload, "applied_defaults", ())
        if isinstance(value, str) and value.strip()
        and ("capex" in value.casefold() or "정의" in value)
    ]
    loss_definition = _requested_net_loss_definition(payload, question)
    if loss_definition:
        values.append(loss_definition)
    # 질문이 개념을 전문용어로 지목했으면 풀이가 질문을 되읊는 것이 된다.
    # 17건 중 16건이 「연결현금흐름표상 유형자산 취득 현금유출액」처럼 이미
    # 정확히 물은 질문이었다. 계획액과 집행액을 가르는 경계는 그 둘을 실제로
    # 맞대려 할 때 incomparable_aggregation_scope 가 더 정확하게 말한다.
    if _ASKED_IN_CONCEPT_TERMS.search(question or ""):
        return _merge_capex_scopes(list(dict.fromkeys(values)))
    labels = " ".join(
        str(getattr(claim, "label", ""))
        for claim in getattr(payload, "claims", ())
    )
    compact = re.sub(r"[^0-9A-Za-z가-힣]", "", labels).casefold()
    if "유형자산취득현금유출" in compact:
        # Older derived-only payloads have no scope-bearing operand label;
        # retain their established consolidated default.  A resolved separate
        # fact always carries ``별도`` in its typed label and overrides it.
        scope = "SFS" if "별도" in labels else "CFS"
        values.append(capex_ppe_definition(scope, question))
    return _merge_capex_scopes(list(dict.fromkeys(values)))


_CAPEX_STATEMENT = re.compile(
    r"^(?P<subject>여기서 CAPEX는|여기서 말한 금액은) (?P<statement>\S+현금흐름표)에 적힌 (?P<tail>.+)$",
    flags=re.DOTALL)


def _merge_capex_scopes(values: list[str]) -> list[str]:
    """연결·별도 CAPEX 정의가 함께 나오면 한 문장으로 합친다.

    연결과 별도를 함께 묻는 비교에서는 두 정의가 모두 실린다. 둘은 재무제표
    이름만 다르고 나머지 문장이 글자까지 같아서, 독자에게는 같은 설명을 두 번
    읽는 것으로 나타난다. 축은 지우지 않고 이름만 나란히 적는다.
    """

    statements: list[str] = []
    tail = None
    subject = "여기서 말한 금액은"
    rest: list[str] = []
    for value in values:
        match = _CAPEX_STATEMENT.match(value)
        if match is None:
            rest.append(value)
            continue
        statements.append(match.group("statement"))
        tail = match.group("tail")
        subject = match.group("subject")
    if len(statements) <= 1 or tail is None:
        return values
    names = "·".join(dict.fromkeys(statements))
    merged = f"{subject} {names}에 적힌 {tail}"
    return [*rest, merged]


__all__ = ["CAPEX_PPE_DEFINITION", "capex_ppe_definition", "required_definition_messages"]
