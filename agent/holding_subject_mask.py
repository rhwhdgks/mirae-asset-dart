"""지분공시 보고 주체(보고자·특별관계자) 성명 부분 가림 — Stage1·Stage2 공용.

``app/``(Stage2 tool·composer)는 ``agent/``(Stage1)에 의존하지만 그 반대는
아니다. 이 값-전용 순수 함수는 Stage1의 역질문(clarification) 선택지
라벨(``agent/stage1_v1_holding_backend.py``)과 Stage2의 답변 claim/guard
(``app/textkit.py`` 재노출 경유)에서 동시에 필요하므로, 단일 출처를 이
계층(agent)에 둔다.

이슈 #46·#51 — 지분공시 보고자 성명 노출 방어. **규칙의 출처는
``src/canonical/security.py`` 하나다** — 정본 마스킹(빌드 시점)과 답변 계층이
같은 함수를 써야 저장된 값과 표시된 값이 어긋나지 않는다. 여기서는 재노출만
한다 (이슈 #140, 정책 판 1.4 에서 정본 마스킹으로 승격). ``[REDACTED:PERSON_NAME]`` 처럼 통째로 가리면 「보고자가 누구인가」에
답할 수 없고, 그 자리에는 국민연금공단·삼성물산 같은 공개 법인·기관명도
온다. 코퍼스 전수 조사로 순수 한글 2~3글자 값은
전부 자연인 이름이고 기관 꼬리(공단·재단·은행…)로 끝나는 3글자 값은 0개임을
확인했다 — 그래서 값의 글자 수·구성만으로 사람 이름을 가른다. 성과 끝
글자를 남기면 같은 사람인지 대조는 가능하면서 특정은 어려워진다.

**한계**: 이 함수는 값 문자열만 보는 순수 함수라 접미사가 없는 맨 그룹명
(예: ``영풍``·``두산``·``한화``·``효성`` — 실제로는 전부 법인)을 사람
이름과 구별하지 못하고, 반대로 ``Scott Samuel Braun`` 같은 라틴 표기
자연인은 법인처럼 읽는다. 정본에 문서별 ``보고자 구분``(``aunit=CRP_TP``)과
행별 ``구분``(``aunit=SPC_TP``)이 있어 「개인(국내)/개인(외국)」인지
「국내법인/외국법인/연기금등 전문투자자/금융기관/기타단체/법령상 조합」인지
정확히 구분된다 — 판 1.5 부터 정본이 **적재 시점에** 그 구분으로 갈라
둔다(이슈 #199). 그래서 문서 컨텍스트가 있는 호출자는
:func:`party_is_private_person` 으로 먼저 판정하고, 이 함수는 구분을 찾지
못한 행에서만 쓰는 폴백이다.
"""
from __future__ import annotations

from typing import Iterable

from src.canonical.security import (
    MASK_CHAR,
    PARTY_TYPE_POLICY_VERSION,
    is_organization_name,  # noqa: F401  (재노출 — 기존 import 경로 보존)
    mask_confirmed_person_name,
    mask_holding_subject_name,
    party_is_person,
    security_policy_at_least,
)


def party_is_private_person(
        name_row: object, party_type: str | None = None,
        corporate_names: Iterable[str] = (), *,
        name: str | None = None) -> bool:
    """지분공시 성명 행의 주체가 자연인인가. 판정 순서는 하나뿐이다 (이슈 #199).

    1. 공시가 스스로 적어 둔 ``구분``(``CRP_TP``/``SPC_TP``)이 있으면 그것이
       정한다. ``에코프로``·``영풍`` 같은 접미사 없는 법인명도, ``Scott
       Samuel Braun`` 같은 라틴 표기 자연인도 값 모양으로는 갈리지 않는다.
    2. 구분이 없더라도 정본이 판 1.5 이후면 정본의 판정(``pii_type``)을
       믿는다 — 적재가 이미 같은 규칙으로 갈라 두었으므로 여기서 다시
       판정하면 두 층이 어긋난다.
    3. 그보다 오래된 정본에서만 값 휴리스틱(``is_organization_name``)으로
       내려간다.

    ``name`` 을 주면 판정 대상 문자열을 그 값으로 바꾼다 — 옛 판이 기관명을
    사람으로 잘못 가려 둔 행에서 원문으로 되짚을 때 쓴다(RPC-017).
    """

    decided = party_is_person(party_type)
    if decided is not None:
        return decided
    if security_policy_at_least(
            getattr(name_row, "security_policy_version", None),
            PARTY_TYPE_POLICY_VERSION):
        return "person_name" in str(getattr(name_row, "pii_type", "") or "")
    text = (name if name is not None
            else str(getattr(name_row, "value", None) or "")).strip()
    return not is_organization_name(text, corporate_names)


def unmasked_personal_name_candidates(
        labels: Iterable[str], corporate_names: Iterable[str] = (),
        ) -> list[str]:
    """역질문 선택지 라벨 목록에서 아직 가려지지 않은 자연인 성명을 찾는다.

    이미 올바르게 가려진 값(``최○범``)이거나, 판정 불가라 원문을 보수적으로
    유지한 값(4글자 이상 등)은 ``mask_holding_subject_name``을 다시 적용해도
    같은 문자열이 나온다(멱등). 다시 적용했을 때 값이 달라진다면 원문 성명이
    그대로 남아 있었다는 뜻이다.

    이 함수는 값 문자열만 보므로, 접미사 없는 법인 그룹명(``영풍`` 등)은
    스스로 판별하지 못한다. 호출자가 문서 컨텍스트(DART ``보고자 구분``
    같은)로 이미 기관임을 확인했다면 ``corporate_names``로 넘겨 오탐을
    피해야 한다 — 그러지 않으면 이 함수가 올바르게 보존된 기관명까지
    거짓 양성으로 잡는다.
    """

    hits = []
    for label in labels:
        if label and mask_holding_subject_name(label, corporate_names) != label:
            hits.append(label)
    return hits


__all__ = ["MASK_CHAR", "PARTY_TYPE_POLICY_VERSION",
           "mask_confirmed_person_name", "mask_holding_subject_name",
           "party_is_person", "party_is_private_person",
           "security_policy_at_least",
           "unmasked_personal_name_candidates"]
