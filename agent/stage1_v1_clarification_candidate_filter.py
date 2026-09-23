"""Drop concept-clarification candidates the canonical build has no facts for.

issue #172 M31 — 「현대모비스의 2025년 이자는 얼마야?」는 이자수익/이자비용/
이자의수취/이자의지급 4택 역질문을 던지지만, 그 4개 중 절반은 실제로
현대모비스 정본에 값이 아예 없는 죽은 선택지였다(이자수익·이자비용을 직접
지정해 물으면 둘 다 ``not_found``). 4택 CLARIFY는
``agent.stage1_v1_financial_backend.FinancialResolutionBackend``가
``agent.concept_alias.resolve_from_question``으로 얻은 후보를 그대로
``ClarificationAuthority`` 로 감싸서 낸다 — 그 후보 목록이 실제 정본
존재와 무관하게 TSV 패턴(``agent/concept_question_patterns.tsv``)의
``candidates`` 열에서 그대로 나온다.

`stage1_v1_financial_backend.py`(기간·개념 표기 담당 팀 소유, issue #172
작업 범위 밖)를 직접 고치지 않고, 그 결과를 감싸는 이 백엔드를
`agent.stage1_v1_backend_composition.build_stage1_v1_backend`의 체인
등록 한 줄에서 원래 자리에 끼워 넣는다. 이 파일 하나가 필터링 책임
전체를 진다 — 개념·기간 정규화 로직은 그대로 두고 결과만 거른다.
"""

from __future__ import annotations

from typing import Any

from .semantic_intent_v1 import SemanticIntent
from .stage1_v1_resolver import (
    BackendResolutionResult,
    ClarificationAuthority,
    ClarificationSlot,
    TerminalAuthority,
    TerminalReasonBinding,
    normalize_backend_resolution_result,
)

#: `FinancialResolutionBackend.resolve()`가 「이자」 같은 구어 다의어를 여러
#: 재무 개념 후보로 되물을 때 붙이는 고정 reason_code. 이 값이 아닌
#: clarification(예: 회사·기간 역질문)은 손대지 않는다.
_CONCEPT_CLARIFY_REASON = "metric_surface_multiple_candidates"


class ConceptClarificationExistenceFilterBackend:
    """Wrap a backend and prune concept-CLARIFY options without canonical facts.

    Every other decision (resolved / other clarifications / terminal) passes
    through untouched. Only a single-slot ``metric_surface_multiple_candidates``
    clarification is inspected: options whose concept has **no** fact row at
    all for the question's company (any period — a cheap existence check, not
    a period-exact lookup) are dropped.

    - 2개 이상 남으면: 좁힌 옵션 목록으로 다시 물어본다.
    - 0~1개만 남으면: 회사가 그 개념을 아예 보고하지 않는 쪽에 가까우므로
      깨진 단일 선택지 역질문 대신 정본 부재 고지(terminal
      ``corpus_coverage_unavailable``)로 답한다. 정확히 1개가 남는 경우를
      해당 값으로 자동 확정하려면 그 값을 실제로 조회·인용하는 로직이
      필요한데, 그건 `financial_backend.py` 안쪽 책임이라 이 래퍼의 범위를
      벗어난다 — 알려진 한계로 남긴다(issue #172 보고 참고).
    """

    def __init__(self, inner: Any, canonical: Any, *, corpus_cutoff: str) -> None:
        if not callable(getattr(inner, "resolve", None)):
            raise TypeError("inner backend에는 resolve()가 필요합니다")
        # `canonical`의 `facts`/`resolve_company`는 요구하지 않는다 — 다른
        # 백엔드 테스트가 이 자리를 좁은 목적의 fake canonical로 채우는
        # 경우가 많고(예: event preflight 전용 스텁), 이 필터는 개념
        # CLARIFY가 실제로 나올 때만 그 메서드를 쓴다. 없거나 실패하면
        # `_company_corp_code`/`_has_any_fact`가 존재를 "확인 불가"로 보고
        # 원래 후보를 그대로 둔다(fail-open).
        self.inner = inner
        self.canonical = canonical
        self.corpus_cutoff = corpus_cutoff

    def _company_corp_code(self, source_intent: SemanticIntent) -> str | None:
        """Single explicit company entity bound to a metric target, if any."""

        referenced = {
            ref
            for item in source_intent.answer_items
            if item.target.kind == "metric"
            for ref in item.target.entity_refs
        }
        surfaces = {
            entity.surface.strip()
            for entity in source_intent.entities
            if entity.entity_id in referenced and entity.kind_hint == "company"
        }
        if len(surfaces) != 1:
            return None
        try:
            rows = self.canonical.resolve_company(next(iter(surfaces)))
        except Exception:
            return None
        if not rows or len(rows) != 1:
            return None
        return rows[0].corp_code

    def _has_any_fact(self, corp_code: str, concept: str) -> bool:
        try:
            rows = self.canonical.facts(
                corp_code, as_of=self.corpus_cutoff, concept=concept)
        except Exception:
            # 조회 실패는 "없음"의 근거가 아니다 — 걸러내지 않고 그대로 둔다.
            return True
        return bool(rows)

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> BackendResolutionResult:
        result = normalize_backend_resolution_result(
            self.inner.resolve(
                question_id=question_id, question=question,
                source_intent=source_intent))
        if result.status != "resolved":
            return result
        authority = result.authority
        if not isinstance(authority, ClarificationAuthority):
            return result
        if (len(authority.slots) != 1
                or authority.slots[0].reason_code != _CONCEPT_CLARIFY_REASON):
            return result

        slot = authority.slots[0]
        corp_code = self._company_corp_code(source_intent)
        if corp_code is None:
            # 회사를 하나로 못 좁히면 존재 여부를 확인할 근거가 없다 —
            # 원래 후보 그대로 되묻는다(fail-open).
            return result

        kept = [
            option for option in slot.options
            if self._has_any_fact(corp_code, option.value)
        ]
        if len(kept) == len(slot.options):
            return result
        if len(kept) >= 2:
            narrowed_slot = ClarificationSlot.model_validate(
                {**slot.model_dump(mode="json"),
                 "options": [option.model_dump(mode="json") for option in kept]})
            narrowed_authority = ClarificationAuthority.model_validate(
                {**authority.model_dump(mode="json"),
                 "slots": [narrowed_slot.model_dump(mode="json")]})
            return BackendResolutionResult.resolved(narrowed_authority)

        item_ids = [item.item_id for item in source_intent.answer_items]
        return BackendResolutionResult.resolved(TerminalAuthority(reasons=[
            TerminalReasonBinding(
                code="corpus_coverage_unavailable", scope="question",
                item_ids=item_ids,
                diagnostic_code="concept_clarify_candidates_absent"),
        ]))


__all__ = ["ConceptClarificationExistenceFilterBackend"]
