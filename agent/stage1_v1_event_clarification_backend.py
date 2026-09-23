"""Canonical receipt-selected event resume backend for the v1 shadow path.

The backend has no question-ID, Gold fixture, company, or receipt inventory.
It accepts only a root receipt selected through a typed event-target slot,
proves that receipt against the canonical event timeline, and returns one
``SelectedEventResolution`` for the ordinary v1 resolver/compiler boundary.
"""

from __future__ import annotations

from datetime import date
import re
from typing import Any, Protocol

from .event_clarification_options_v1 import build_event_clarification_options
from .deterministic_plan_compiler_v1 import (
    AuthoritativeResolution,
    ResolutionFieldProof,
    ResolutionSourceProof,
    ResolvedItem,
    SelectedEventResolution,
)
from .semantic_intent_v1 import SemanticIntent, semantic_intent_digest
from .stage1_v1_clarification_session import (
    ClarificationResolutionContext,
    ClarificationResumeResult,
)
from .stage1_v1_resolver import (
    ClarificationAuthority,
    ResolvedAuthority,
    TerminalAuthority,
)


class CanonicalCompanyLike(Protocol):
    corp_code: str
    corp_name: str


class CanonicalEventTimelineLike(Protocol):
    event_key: str
    corp_code: str
    corp_name: str
    root_rcept_no: str


class CanonicalEventReadLike(Protocol):
    def resolve_company(self, name: str) -> list[CanonicalCompanyLike]: ...

    def event_timeline(
            self, *, as_of: str, rcept_no: str,
            verify_evidence: bool = False,
            ) -> CanonicalEventTimelineLike | None: ...


class EventClarificationResolutionError(RuntimeError):
    """A selected receipt could not become a canonical single-event authority."""


def _selected_event_answer(context: ClarificationResolutionContext) -> str:
    matches = [
        row for row in context.current.answers
        if (
            row.role_hint in {"target", "event"}
            and row.response_kind in {"select_one", "provide_value"}
        )
    ]
    if len(matches) != 1:
        raise EventClarificationResolutionError(
            "event resume에는 event/target 답변 하나가 필요합니다")
    return matches[0].value


def _event_item(intent: SemanticIntent):
    if len(intent.answer_items) != 1:
        raise EventClarificationResolutionError(
            "selected event resume은 answer item 하나만 지원합니다")
    item = intent.answer_items[0]
    # **표현이 아니라 계획에 쓰이는 축만 본다.**
    #
    # `target.kind` 가 event 냐 document 냐, 답 모양이 scalar 냐 record 냐는 모델이
    # 판마다 흔드는 축이다. 그것으로 잠그면 **역질문에 답을 받고도 재개가 거부된다**
    # — 사용자가 고른 접수번호가 있는데 문을 못 여는 것이 가장 나쁜 실패다.
    #
    # 조회 동작·선택 없음·이름 붙은 필드는 남긴다. 어긋나면 내려갈 계획이 달라진다.
    # 접수번호를 이미 들고 들어온 자리다.  그러므로 「가장 최근」 선택은
    # **해소된 것**이지 남은 요구가 아니다 — 고른 접수번호가 그 답이다.
    # 최대·최소처럼 무엇의 극값인지 따로 정해야 하는 선택은 그대로 거절한다.
    selection_mode = getattr(item.selection, "mode", None)
    if (item.operation != "retrieve"
            or not item.target.entity_refs
            or selection_mode not in {None, "latest"}
            or item.scope.as_of_expression is not None
            or item.scope.scope_qualifier_expressions
            or item.output.projection_mode != "named_fields"
            or not item.output.field_surfaces):
        raise EventClarificationResolutionError(
            "selected event semantic topology가 지원 범위를 벗어납니다")
    entity_by_id = {row.entity_id: row for row in intent.entities}
    # 계약 상대도 `company` 로 표시돼 함께 딸려 올 수 있다. 후보를 보존한 뒤
    # canonical 회사 registry와 선택 receipt 소유권으로 issuer를 고른다.
    company_entities = [
        entity_by_id[ref] for ref in item.target.entity_refs
        if ref in entity_by_id and entity_by_id[ref].kind_hint == "company"
    ]
    if not company_entities:
        raise EventClarificationResolutionError(
            "selected event에는 확정 가능한 company entity가 필요합니다")
    return item, company_entities


def _unique_canonical_issuer(
        canonical: CanonicalEventReadLike,
        entities: list[Any],
        ) -> tuple[Any, CanonicalCompanyLike]:
    matches: list[tuple[Any, CanonicalCompanyLike]] = []
    for entity in entities:
        candidates = canonical.resolve_company(entity.surface)
        if len(candidates) == 1:
            matches.append((entity, candidates[0]))
    if len(matches) != 1:
        raise EventClarificationResolutionError(
            "selected event issuer가 canonical에서 유일하게 확정되지 않습니다")
    return matches[0]


def _selected_event_authority(
        canonical: CanonicalEventReadLike,
        *,
        question_id: str,
        source_intent: SemanticIntent,
        receipt: str,
        canonical_build_id: str,
        resolver_version: str,
        reference_date: date,
        corpus_cutoff: str,
        ) -> ResolvedAuthority:
    if re.fullmatch(r"[0-9]{14}", receipt) is None:
        raise EventClarificationResolutionError(
            "selected event receipt 형식이 잘못되었습니다")
    item, entities = _event_item(source_intent)
    timeline = canonical.event_timeline(
        as_of=corpus_cutoff, rcept_no=receipt, verify_evidence=True)
    if timeline is None or timeline.root_rcept_no != receipt:
        raise EventClarificationResolutionError(
            "선택 receipt의 canonical event를 찾을 수 없습니다")

    issuer_matches: list[tuple[Any, CanonicalCompanyLike]] = []
    for entity in entities:
        candidates = canonical.resolve_company(entity.surface)
        if len(candidates) != 1:
            continue
        company = candidates[0]
        if (
                timeline.corp_code == company.corp_code
                and timeline.corp_name == company.corp_name
        ):
            issuer_matches.append((entity, company))
    if len(issuer_matches) != 1:
        raise EventClarificationResolutionError(
            "선택 receipt의 issuer canonical ownership이 유일하지 않습니다")
    entity, company = issuer_matches[0]

    proof = f"canonical:event-root:{timeline.event_key}:{receipt}"
    resolution = AuthoritativeResolution.create(
        question_id=question_id,
        source_intent_digest=semantic_intent_digest(source_intent),
        canonical_build_id=canonical_build_id,
        resolver_version=resolver_version,
        reference_date=reference_date,
        corpus_cutoff=corpus_cutoff,
        items=[ResolvedItem(
            item_id=item.item_id,
            target_surface=item.target.surface,
            resolution=SelectedEventResolution(
                corp_code=company.corp_code,
                corp_name=company.corp_name,
                entity_surface=entity.surface,
                event_key=timeline.event_key,
                root_receipt=receipt,
                selector_proof=ResolutionSourceProof(
                    source_receipt=receipt, proof_ref=proof),
            ),
            field_proofs=[ResolutionFieldProof(
                source_field_index=index,
                surface=surface,
                proof_ref=f"source-field:{item.item_id}:{index}",
            ) for index, surface in enumerate(item.output.field_surfaces)],
        )],
    )
    return ResolvedAuthority(resolution=resolution)


def _remove_answered_mentions(
        intent: SemanticIntent,
        context: ClarificationResolutionContext,
        ) -> SemanticIntent:
    answered_ids = {
        mention_id for answer in context.current.answers
        for mention_id in answer.mention_ids
    }
    payload = intent.model_dump(mode="python", warnings=False)
    payload["unresolved_mentions"] = [
        row for row in payload["unresolved_mentions"]
        if row["mention_id"] not in answered_ids
    ]
    return SemanticIntent.model_validate(payload, strict=True)


class CanonicalSelectedEventResumeBackend:
    """Resolve a typed event-target answer through canonical event ownership."""

    def __init__(
            self, canonical: CanonicalEventReadLike, *,
            canonical_build_id: str, resolver_version: str,
            reference_date: date, corpus_cutoff: str,
            event_preflight: Any | None = None,
            ) -> None:
        if re.fullmatch(r"[0-9a-f]{32}", canonical_build_id) is None:
            raise ValueError("canonical_build_id 형식이 잘못되었습니다")
        if re.fullmatch(r"[A-Za-z0-9_.-]+/[0-9]+(?:\.[0-9]+)*$", resolver_version) is None:
            raise ValueError("resolver_version 형식이 잘못되었습니다")
        if re.fullmatch(r"[0-9]{8}", corpus_cutoff) is None:
            raise ValueError("corpus_cutoff 형식이 잘못되었습니다")
        self.canonical = canonical
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff
        self.event_preflight = event_preflight
        if (event_preflight is not None and not callable(
                getattr(event_preflight, "resolve_event_key", None))):
            raise TypeError("event clarification preflight 계약이 잘못되었습니다")

    def resume(
            self, *, question_id: str, question: str,
            original_source_intent: SemanticIntent,
            current_source_intent: SemanticIntent,
            context: ClarificationResolutionContext,
            ) -> ClarificationResumeResult:
        del question, original_source_intent
        derived = _remove_answered_mentions(current_source_intent, context)
        answer = _selected_event_answer(context)
        receipt = answer
        if re.fullmatch(r"[0-9]{14}", receipt) is None:
            item, entities = _event_item(current_source_intent)
            entity, _company = _unique_canonical_issuer(self.canonical, entities)
            lookup = build_event_clarification_options(
                self.canonical,
                company_surface=entity.surface,
                as_of=self.corpus_cutoff,
                contract_surface=answer,
                preflight=self.event_preflight,
                corpus_cutoff=self.corpus_cutoff,
            )
            if lookup.status == "ambiguous" and len(lookup.options) > 1:
                return ClarificationResumeResult(
                    derived_source_intent=derived,
                    authority=ClarificationAuthority.model_validate({
                        "kind": "clarification", "slots": [{
                            "slot_id": "slot-1", "role_hint": "event",
                            "reason_code": "event_target_multiple_candidates",
                            "response_kind": "select_one",
                            "prompt": "어떤 계약을 확인할까요?",
                            "applies_to_item_ids": [item.item_id],
                            "mention_ids": [],
                            "options": [row.model_dump(mode="json")
                                        for row in lookup.options],
                        }],
                    }, strict=True),
                )
            if lookup.status != "resolved" or len(lookup.options) != 1:
                return ClarificationResumeResult(
                    derived_source_intent=derived,
                    authority=TerminalAuthority(reasons=[{
                        "code": "corpus_coverage_unavailable",
                        "scope": "question",
                        "item_ids": [item.item_id],
                    }]),
                )
            receipt = lookup.options[0].value
        return ClarificationResumeResult(
            derived_source_intent=derived,
            authority=_selected_event_authority(
                self.canonical,
                question_id=question_id,
                source_intent=derived,
                receipt=receipt,
                canonical_build_id=self.canonical_build_id,
                resolver_version=self.resolver_version,
                reference_date=self.reference_date,
                corpus_cutoff=self.corpus_cutoff,
            ),
        )


__all__ = [
    "CanonicalEventReadLike", "CanonicalSelectedEventResumeBackend",
    "EventClarificationResolutionError", "_selected_event_authority",
]
