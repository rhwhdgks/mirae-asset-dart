"""Question-grounded repair for a literal disclosure-form-name mention.

SG-007·009 실호출(이슈 #75, 2026-09-03): 질문이 공시 서식명을 낫표(「」)로
직접 지정하면(예 "「신규시설투자」 공시의 투자금액은 얼마인가?", "「주요사항
보고서(전환사채권발행결정)」의 사채 권면(전자등록)총액은 얼마인가?") HCX는
이를 재무 지표 질문으로 오인해 ``target.kind=metric`` 으로 낸다.  그 결과
financial 백엔드가 먼저 항목을 붙잡아, 사전에 없는 계정명은 되묻거나
(SG-007: 유형자산의취득/무형자산의취득/투자활동현금흐름 clarify) 아예
unsupported_request 로 닫는다(SG-009).

서식명이 낫표로 리터럴 지정된 질문은 재무 개념 해석의 대상이 아니라 사건
슬롯(그 서식 하나의 필드 값) 조회다 — `EventCollectionResolutionBackend`의
`_facilities_investment_collection`(신규시설투자등)과
`DocumentAttributeEvidenceBackend`(그 밖의 단일 서식 필드)가 이미 이 모양을
처리할 수 있지만, 둘 다 ``target.kind in {"document","event","topic"}`` 을
요구하고 financial 백엔드는 이들보다 먼저 시도된다.  이 regrounder는 회사·
필드 값 등 어떤 것도 새로 고르지 않는다 — 낫표 서식명과 그 바로 뒤에 이어지는
"~은/는 얼마" 절 앞의 계정 표면을 질문에서 그대로 잘라내 ``target.kind``와
``output.shape``만 사건 슬롯 경로로 돌린다.
"""

from __future__ import annotations

import re

from .semantic_intent_v1 import SemanticIntent


_LITERAL_FORM_FIELD = re.compile(
    r"[「『](?P<form>[^」』]{2,80})[」』]\s*(?:공시)?\s*(?:의|에서)\s*"
    r"(?P<field>[^?？。\n]{2,60}?)\s*(?:은|는|이|가)\s*(?:얼마|몇)")


class LiteralDisclosureFormFieldRegrounder:
    """Restore the event-slot shape from a quoted disclosure-form question.

    Only the quoted form name and the literal account phrase immediately
    asked for are recovered; company, period, and every other wire surface
    stay exactly as HCX produced them.
    """

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str) or not question.strip()
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        if (item.target.kind not in {"metric", "document", "event", "topic"}
                or item.operation not in {"retrieve", "compare"}
                or item.output.projection_mode != "named_fields"
                or item.selection is not None):
            return intent
        match = _LITERAL_FORM_FIELD.search(question)
        if match is None:
            return intent
        form = match.group("form").strip()
        field = match.group("field").strip()
        if not form or not field or form not in question or field not in question:
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        answer_item = payload["answer_items"][0]
        answer_item["target"]["kind"] = "document"
        answer_item["target"]["surface"] = form
        answer_item["output"].update({
            "shape": "record",
            "field_surfaces": [field],
        })
        return SemanticIntent.model_validate(payload, strict=True)


__all__ = ["LiteralDisclosureFormFieldRegrounder"]
