"""Typed clarification resume backends for non-event semantic axes.

User answers are used only to derive canonical authority.  The original
question-grounded ``SemanticIntent`` remains immutable except that answered
unresolved mentions are removed, as required by the session contract.
"""

from __future__ import annotations

from datetime import date
import json
from typing import Any

from .date_surface import question_date_surfaces
from .deterministic_plan_compiler_v1 import (
    AuthoritativeResolution,
    EventAmountChangeResolution,
    EventAmountObservationProof,
    ResolutionFieldProof,
    ResolutionSourceProof,
    ResolvedItem,
    SelectedEventResolution,
)
from .event_amount_observation_preflight_v1 import (
    EventAmountObservationPreflight,
    EventAmountObservationStatus,
)
from .event_clarification_options_v1 import build_event_clarification_options
from .semantic_intent_v1 import SemanticIntent, semantic_intent_digest
from .stage1_v1_clarification_session import (
    ClarificationResolutionContext,
    ClarificationResumeResult,
)
from .stage1_v1_event_clarification_backend import (
    CanonicalSelectedEventResumeBackend,
)
from .stage1_v1_financial_backend import FinancialResolutionBackend
from .stage1_v1_holding_backend import HoldingDisclosureResolutionBackend
from .stage1_v1_narrative_matrix import (
    NarrativeMatrixResolutionBackend,
    apply_narrative_matrix_resume,
    narrative_matrix_reduction_options,
)
from .stage1_v1_resolver import (
    ClarificationAuthority,
    ClarificationOption,
    ResolvedAuthority,
    TerminalAuthority,
)


class ContextClarificationResolutionError(RuntimeError):
    """The supplied slot values cannot be closed to canonical authority."""


_EVENT_AMOUNT_DIAGNOSTIC_CODE_BY_STATUS: dict[
        EventAmountObservationStatus, str] = {
    "event_not_disclosed": "event_amount_observation_event_not_disclosed",
    "ownership_mismatch": "event_amount_observation_ownership_mismatch",
    "amount_missing": "event_amount_observation_amount_missing",
    "amount_unavailable": "event_amount_observation_amount_unavailable",
    "amount_ambiguous": "event_amount_observation_amount_ambiguous",
    "evidence_unavailable": "event_amount_observation_evidence_unavailable",
    "numeric_invalid": "event_amount_observation_numeric_invalid",
    "unit_unsupported": "event_amount_observation_unit_unsupported",
    "unit_mismatch": "event_amount_observation_unit_mismatch",
}


def _remove_answered_mentions(
        intent: SemanticIntent,
        context: ClarificationResolutionContext,
        ) -> SemanticIntent:
    answered_ids = {
        mention_id
        for answer in context.current.answers
        for mention_id in answer.mention_ids
    }
    payload = intent.model_dump(mode="python", warnings=False)
    payload["unresolved_mentions"] = [
        row for row in payload["unresolved_mentions"]
        if row["mention_id"] not in answered_ids
    ]
    return SemanticIntent.model_validate(payload, strict=True)


def _scope_surface(value: str) -> str:
    normalized = "".join(value.split()).casefold()
    if normalized in {"cfs", "연결", "연결재무제표"}:
        return "CFS"
    if normalized in {"sfs", "별도", "개별", "별도재무제표"}:
        return "SFS"
    raise ContextClarificationResolutionError(
        "scope 답변은 CFS(연결) 또는 SFS(별도)여야 합니다")


class CanonicalFinancialContextResumeBackend:
    """Resolve company, metric, period and scope answers as one atomic turn."""

    _SUPPORTED_ROLES = frozenset({
        "entity", "target", "time", "timepoint", "qualifier", "value_kind",
    })

    def __init__(
            self, canonical: Any, *, canonical_build_id: str,
            resolver_version: str, reference_date: date, corpus_cutoff: str,
            ) -> None:
        self.financial = FinancialResolutionBackend(
            canonical,
            scope_authority=canonical,
            canonical_build_id=canonical_build_id,
            resolver_version=resolver_version,
            reference_date=reference_date,
            corpus_cutoff=corpus_cutoff,
        )

    @staticmethod
    def _retry_authority(
            *, item_ids: list[str], role: str,
            ) -> ClarificationAuthority:
        if role == "qualifier":
            prompt = "연결과 별도 중 어느 기준인지 다시 선택해주세요."
            response_kind = "select_one"
            options = [
                ClarificationOption(
                    value="CFS", label="연결",
                    proof_refs=["policy:financial-scope:CFS"]),
                ClarificationOption(
                    value="SFS", label="별도",
                    proof_refs=["policy:financial-scope:SFS"]),
            ]
        elif role in {"time", "timepoint"}:
            prompt = "확인할 기간을 다시 알려주세요."
            response_kind = "provide_value"
            options = []
        else:
            prompt = "회사를 정본에서 찾지 못했습니다. 회사명을 다시 알려주세요."
            response_kind = "provide_value"
            options = []
            role = "entity"
        return ClarificationAuthority.model_validate({
            "kind": "clarification", "slots": [{
                "slot_id": "slot-1", "role_hint": role,
                "reason_code": "missing_context",
                "response_kind": response_kind, "prompt": prompt,
                "applies_to_item_ids": item_ids, "mention_ids": [],
                "options": [row.model_dump(mode="json") for row in options],
            }],
        }, strict=True)

    @staticmethod
    def _terminal(item_ids: list[str]) -> TerminalAuthority:
        return TerminalAuthority(reasons=[{
            "code": "unsupported_semantic_target", "scope": "items",
            "item_ids": item_ids,
        }])

    def resume(
            self, *, question_id: str, question: str,
            original_source_intent: SemanticIntent,
            current_source_intent: SemanticIntent,
            context: ClarificationResolutionContext,
            ) -> ClarificationResumeResult:
        del original_source_intent
        answers = context.current.answers
        if not answers or any(
                row.role_hint not in self._SUPPORTED_ROLES for row in answers):
            raise ContextClarificationResolutionError(
                "financial resume이 지원하지 않는 clarification role입니다")
        if any(
                item.target.kind != "metric"
                for item in current_source_intent.answer_items):
            raise ContextClarificationResolutionError(
                "financial resume에는 metric item만 허용됩니다")

        derived = _remove_answered_mentions(current_source_intent, context)
        enriched = current_source_intent.model_dump(
            mode="python", warnings=False)
        item_by_id = {row["item_id"]: row for row in enriched["answer_items"]}
        next_entity = len(enriched["entities"]) + 1

        for answer in answers:
            items = [item_by_id[value] for value in answer.applies_to_item_ids]
            if answer.role_hint == "entity":
                entity_id = f"entity-{next_entity}"
                next_entity += 1
                enriched["entities"].append({
                    "entity_id": entity_id,
                    "kind_hint": "company",
                    "surface": answer.value,
                })
                for item in items:
                    if entity_id not in item["target"]["entity_refs"]:
                        item["target"]["entity_refs"].append(entity_id)
            elif answer.role_hint in {"target", "value_kind"}:
                # The financial resolver names an ambiguous colloquial metric
                # slot ``value_kind`` because the options are canonical
                # concepts.  Semantically it fills the same target coordinate
                # as an ordinary ``target`` clarification.  Contract amount
                # changes are dispatched before reaching this backend.
                for item in items:
                    item["target"]["kind"] = "metric"
                    item["target"]["surface"] = answer.value
            elif answer.role_hint in {"time", "timepoint"}:
                for item in items:
                    item["scope"]["target_period_expressions"] = [answer.value]
            elif answer.role_hint == "qualifier":
                try:
                    scope = _scope_surface(answer.value)
                except ContextClarificationResolutionError:
                    return ClarificationResumeResult(
                        derived_source_intent=derived,
                        authority=self._retry_authority(
                            item_ids=list(answer.applies_to_item_ids),
                            role="qualifier"))
                for item in items:
                    item["scope"]["scope_qualifier_expressions"] = [scope]

        # This enriched copy is an internal lookup request, not a replacement
        # for the immutable question-grounded intent.
        lookup_intent = SemanticIntent.model_validate(enriched, strict=True)
        authority = self.financial.resolve(
            question_id=question_id,
            question=question,
            source_intent=lookup_intent,
        )
        if not authority:
            return ClarificationResumeResult(
                derived_source_intent=derived,
                authority=self._terminal([
                    row.item_id for row in current_source_intent.answer_items]))
        if authority.get("kind") == "clarification":
            return ClarificationResumeResult(
                derived_source_intent=derived,
                authority=ClarificationAuthority.model_validate(
                    authority, strict=True))
        if authority.get("kind") == "terminal":
            return ClarificationResumeResult(
                derived_source_intent=derived,
                authority=TerminalAuthority.model_validate(
                    authority, strict=True))
        if authority.get("kind") != "resolved":
            return ClarificationResumeResult(
                derived_source_intent=derived,
                authority=self._terminal([
                    row.item_id for row in current_source_intent.answer_items]))
        resolution_payload = dict(authority["resolution"])
        original_by_id = {
            item.item_id: item for item in derived.answer_items
        }
        items = []
        for row in resolution_payload["items"]:
            item = dict(row)
            item["target_surface"] = original_by_id[item["item_id"]].target.surface
            items.append(item)
        rebound = AuthoritativeResolution.create(
            question_id=question_id,
            source_intent_digest=semantic_intent_digest(derived),
            canonical_build_id=resolution_payload["canonical_build_id"],
            resolver_version=resolution_payload["resolver_version"],
            reference_date=resolution_payload["reference_date"],
            corpus_cutoff=resolution_payload["corpus_cutoff"],
            items=items,
            premise_proofs=resolution_payload.get("premise_proofs", []),
        )
        return ClarificationResumeResult(
            derived_source_intent=derived,
            authority=ResolvedAuthority(resolution=rebound),
        )


class CanonicalHoldingContextResumeBackend:
    """Resume filer/receipt/party selection for an exact holding disclosure.

    The public semantic intent stays immutable.  The selected option is a
    digest-bound execution context value and is revalidated against the same
    canonical build before an exact receipt can be emitted.
    """

    _REASON_TO_AXIS = {
        "holding_filer_multiple_candidates": "filer",
        "holding_receipt_multiple_candidates": "receipt",
        "holding_party_multiple_or_unknown": "party",
    }

    def __init__(
            self, canonical: Any, *, canonical_build_id: str,
            resolver_version: str, reference_date: date, corpus_cutoff: str,
            ) -> None:
        self.backend = HoldingDisclosureResolutionBackend(
            canonical,
            canonical_build_id=canonical_build_id,
            resolver_version=resolver_version,
            reference_date=reference_date,
            corpus_cutoff=corpus_cutoff,
        )

    @staticmethod
    def _terminal(item_ids: list[str]) -> TerminalAuthority:
        return TerminalAuthority(reasons=[{
            "code": "corpus_coverage_unavailable",
            "scope": "items",
            "item_ids": item_ids,
        }])

    def resume(
            self, *, question_id: str, question: str,
            original_source_intent: SemanticIntent,
            current_source_intent: SemanticIntent,
            context: ClarificationResolutionContext,
            ) -> ClarificationResumeResult:
        del original_source_intent
        answers = [
            answer
            for turn in context.history
            for answer in turn.request.answers
            if answer.reason_code in self._REASON_TO_AXIS
        ] + [
            answer for answer in context.current.answers
            if answer.reason_code in self._REASON_TO_AXIS
        ]
        if not answers or len(answers) != len({
                answer.reason_code for answer in answers}):
            raise ContextClarificationResolutionError(
                "holding resume 선택 축이 없거나 중복되었습니다")
        selected: dict[str, str] = {}
        for answer in answers:
            if answer.response_kind != "select_one":
                raise ContextClarificationResolutionError(
                    "holding resume은 select_one 답변만 허용합니다")
            selected[self._REASON_TO_AXIS[answer.reason_code]] = answer.value

        derived = _remove_answered_mentions(current_source_intent, context)
        authority = self.backend.resolve_with_selection(
            question_id=question_id,
            question=question,
            source_intent=derived,
            selected_filer=selected.get("filer"),
            selected_receipt=selected.get("receipt"),
            selected_party=selected.get("party"),
        )
        if authority is None:
            return ClarificationResumeResult(
                derived_source_intent=derived,
                authority=self._terminal([
                    item.item_id for item in derived.answer_items]),
            )
        if isinstance(authority, ClarificationAuthority):
            return ClarificationResumeResult(
                derived_source_intent=derived, authority=authority)
        if isinstance(authority, TerminalAuthority):
            return ClarificationResumeResult(
                derived_source_intent=derived, authority=authority)
        if authority.get("kind") != "resolved":
            return ClarificationResumeResult(
                derived_source_intent=derived,
                authority=self._terminal([
                    item.item_id for item in derived.answer_items]),
            )
        resolution = AuthoritativeResolution.model_validate_json(
            json.dumps(
                authority["resolution"], ensure_ascii=False,
                sort_keys=True, separators=(",", ":")),
            strict=True,
        )
        return ClarificationResumeResult(
            derived_source_intent=derived,
            authority=ResolvedAuthority(resolution=resolution),
        )


def _all_context_answers(
        context: ClarificationResolutionContext,
        ) -> dict[str, Any]:
    """Latest answer per semantic role across a chained clarification."""
    result: dict[str, Any] = {}
    for turn in context.history:
        for answer in turn.request.answers:
            result[answer.role_hint] = answer
    for answer in context.current.answers:
        result[answer.role_hint] = answer
    return result


def _amount_slot(value: str) -> str:
    normalized = "".join(value.split()).casefold()
    if normalized in {"contract_amount", "계약금액"}:
        return "계약금액"
    if normalized in {"termination_amount", "해지금액"}:
        return "해지금액"
    raise ContextClarificationResolutionError(
        "value_kind 답변은 계약금액 또는 해지금액이어야 합니다")


def _two_timepoints(value: str, *, corpus_cutoff: str) -> list[str]:
    parsed = question_date_surfaces(value)
    if len(parsed) != 2 or any(
            month is None or day is None for _, month, day in parsed):
        raise ContextClarificationResolutionError(
            "timepoint 답변에는 서로 다른 완전한 날짜 두 개가 필요합니다")
    points = sorted(
        f"{year:04d}{month:02d}{day:02d}"
        for year, month, day in parsed
        if month is not None and day is not None
    )
    if len(points) != 2 or points[0] >= points[1]:
        raise ContextClarificationResolutionError(
            "timepoint는 이전 시점과 이후 시점 순서로 달라야 합니다")
    if points[1] > corpus_cutoff:
        raise ContextClarificationResolutionError(
            "timepoint는 corpus cutoff 이후일 수 없습니다")
    return points


class CanonicalContractAmountChangeResumeBackend:
    """Close company -> event -> two observations as a two-turn flow."""

    def __init__(
            self, canonical: Any, *, canonical_build_id: str,
            resolver_version: str, reference_date: date, corpus_cutoff: str,
            event_preflight: Any | None = None,
            amount_observation_preflight: Any | None = None,
            ) -> None:
        self.canonical = canonical
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff
        self.event_preflight = event_preflight
        if (event_preflight is not None and not callable(
                getattr(event_preflight, "resolve_event_key", None))):
            raise TypeError("contract amount change preflight 계약이 잘못되었습니다")
        self.amount_observation_preflight = (
            amount_observation_preflight
            if amount_observation_preflight is not None
            else EventAmountObservationPreflight(canonical)
        )
        if not callable(getattr(
                self.amount_observation_preflight, "resolve_pair", None)):
            raise TypeError("contract amount observation preflight 계약이 잘못되었습니다")

    @staticmethod
    def _retry_authority(*, item_id: str, role: str) -> ClarificationAuthority:
        if role == "entity":
            prompt = "회사를 정본에서 찾지 못했습니다. 회사명을 다시 알려주세요."
            response_kind = "provide_value"
            options: list[ClarificationOption] = []
        else:
            prompt = (
                "서로 다른 완전한 날짜 두 개를 다시 알려주세요. "
                "(예: 2024-12-31, 2025-12-31)")
            response_kind = "provide_value"
            options = []
        return ClarificationAuthority.model_validate({
            "kind": "clarification",
            "slots": [{
                "slot_id": "slot-1", "role_hint": role,
                "reason_code": "missing_context",
                "response_kind": response_kind, "prompt": prompt,
                "applies_to_item_ids": [item_id], "mention_ids": [],
                "options": [row.model_dump(mode="json") for row in options],
            }],
        }, strict=True)

    @staticmethod
    def _terminal(*, item_id: str, diagnostic_code: str | None = None
                  ) -> TerminalAuthority:
        return TerminalAuthority(reasons=[{
            "code": "corpus_coverage_unavailable", "scope": "items",
            "item_ids": [item_id],
            "diagnostic_code": diagnostic_code,
        }])

    @staticmethod
    def _amount_observation_terminal(
            *, item_id: str, status: EventAmountObservationStatus,
            ) -> TerminalAuthority:
        diagnostic_code = _EVENT_AMOUNT_DIAGNOSTIC_CODE_BY_STATUS.get(status)
        if diagnostic_code is None:
            raise ContextClarificationResolutionError(
                f"알 수 없는 amount observation 상태입니다: {status}")
        return CanonicalContractAmountChangeResumeBackend._terminal(
            item_id=item_id, diagnostic_code=diagnostic_code)

    @staticmethod
    def _event_choice_authority(
            *, item_id: str, amount_slot: str, options: list[Any],
            ) -> ClarificationAuthority:
        return ClarificationAuthority.model_validate({
            "kind": "clarification",
            "slots": [{
                "slot_id": "slot-1", "role_hint": "event",
                "reason_code": "missing_context",
                "response_kind": "select_one",
                "prompt": f"어느 계약의 {amount_slot} 변동을 확인할까요?",
                "applies_to_item_ids": [item_id], "mention_ids": [],
                "options": [row.model_dump(mode="json") for row in options],
            }],
        }, strict=True)

    def _resolved(
            self, *, question_id: str, derived: SemanticIntent,
            company: Any, entity_surface: str, receipt: str,
            timepoints: list[str], amount_slot: str,
            ) -> ResolvedAuthority | TerminalAuthority:
        timeline = self.canonical.event_timeline(
            as_of=timepoints[1], rcept_no=receipt, verify_evidence=True)
        if timeline is None or (
                timeline.root_rcept_no != receipt
                or timeline.corp_code != company.corp_code
                or timeline.corp_name != company.corp_name):
            raise ContextClarificationResolutionError(
                "선택 계약의 canonical root/issuer ownership이 다릅니다")
        item = derived.answer_items[0]
        observed = self.amount_observation_preflight.resolve_pair(
            event_key=timeline.event_key, root_receipt=receipt,
            corp_code=company.corp_code, corp_name=company.corp_name,
            timepoints=timepoints, requested_slot=amount_slot)
        if observed.status != "resolved":
            return self._amount_observation_terminal(
                item_id=item.item_id, status=observed.status)
        proof = f"canonical:event-root:{timeline.event_key}:{receipt}"
        authority = AuthoritativeResolution.create(
            question_id=question_id,
            source_intent_digest=semantic_intent_digest(derived),
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            items=[ResolvedItem(
                item_id=item.item_id, target_surface=item.target.surface,
                resolution=EventAmountChangeResolution(
                    corp_code=company.corp_code, corp_name=company.corp_name,
                    entity_surface=entity_surface,
                    event_key=timeline.event_key, root_receipt=receipt,
                    selector_proof=ResolutionSourceProof(
                        source_receipt=receipt, proof_ref=proof),
                    timepoints=timepoints, requested_slot=amount_slot,
                    observations=[EventAmountObservationProof(
                        timepoint=row.timepoint,
                        source_receipt=row.source_receipt,
                        evidence_id=row.evidence_id,
                        normalized_value=row.normalized_value,
                        currency=row.currency, unit=row.unit, scale=row.scale,
                        proof=ResolutionSourceProof(
                            source_receipt=row.source_receipt,
                            proof_ref=row.proof_ref),
                    ) for row in observed.observations],
                ),
                field_proofs=[ResolutionFieldProof(
                    source_field_index=0,
                    surface=item.output.field_surfaces[0],
                    proof_ref=f"source-field:{item.item_id}:0",
                )],
            )],
        )
        return ResolvedAuthority(resolution=authority)

    def resume(
            self, *, question_id: str, question: str,
            original_source_intent: SemanticIntent,
            current_source_intent: SemanticIntent,
            context: ClarificationResolutionContext,
            ) -> ClarificationResumeResult:
        del question, original_source_intent
        if len(current_source_intent.answer_items) != 1:
            raise ContextClarificationResolutionError(
                "contract amount change는 answer item 하나만 지원합니다")
        item = current_source_intent.answer_items[0]
        answers = _all_context_answers(context)
        derived = _remove_answered_mentions(current_source_intent, context)
        entity_answer = answers.get("entity")
        time_answer = answers.get("timepoint")
        value_answer = answers.get("value_kind")
        if entity_answer is None or time_answer is None or value_answer is None:
            raise ContextClarificationResolutionError(
                "contract amount change 필수 clarification 답변이 없습니다")
        companies = self.canonical.resolve_company(entity_answer.value)
        if len(companies) != 1:
            return ClarificationResumeResult(
                derived_source_intent=derived,
                authority=self._retry_authority(
                    item_id=item.item_id, role="entity"),
            )
        company = companies[0]
        try:
            timepoints = _two_timepoints(
                time_answer.value, corpus_cutoff=self.corpus_cutoff)
        except ContextClarificationResolutionError:
            return ClarificationResumeResult(
                derived_source_intent=derived,
                authority=self._retry_authority(
                    item_id=item.item_id, role="timepoint"),
            )
        try:
            amount_slot = _amount_slot(value_answer.value)
        except ContextClarificationResolutionError:
            return ClarificationResumeResult(
                derived_source_intent=derived,
                authority=ClarificationAuthority.model_validate({
                    "kind": "clarification", "slots": [{
                        "slot_id": "slot-1", "role_hint": "value_kind",
                        "reason_code": "missing_context",
                        "response_kind": "select_one",
                        "prompt": "계약금액과 해지금액 중 다시 선택해주세요.",
                        "applies_to_item_ids": [item.item_id],
                        "mention_ids": [],
                        "options": [
                            ClarificationOption(
                                value="contract_amount", label="계약금액",
                                proof_refs=[
                                    "policy:event-amount-slot:contract_amount"]
                            ).model_dump(mode="json"),
                            ClarificationOption(
                                value="termination_amount", label="해지금액",
                                proof_refs=[
                                    "policy:event-amount-slot:termination_amount"]
                            ).model_dump(mode="json"),
                        ],
                    }],
                }, strict=True))

        event_answer = answers.get("event")
        if event_answer is not None:
            receipt = event_answer.value
            if not (receipt.isdigit() and len(receipt) == 14):
                lookup = build_event_clarification_options(
                    self.canonical, company_surface=entity_answer.value,
                    as_of=timepoints[1], target_surface=receipt,
                    preflight=self.event_preflight,
                    corpus_cutoff=self.corpus_cutoff,
                )
                if lookup.status == "resolved" and len(lookup.options) == 1:
                    receipt = lookup.options[0].value
                elif lookup.status == "ambiguous" and len(lookup.options) > 1:
                    return ClarificationResumeResult(
                        derived_source_intent=derived,
                        authority=self._event_choice_authority(
                            item_id=item.item_id,
                            amount_slot=amount_slot,
                            options=list(lookup.options),
                        ),
                    )
                else:
                    return ClarificationResumeResult(
                        derived_source_intent=derived,
                        authority=self._terminal(item_id=item.item_id),
                    )
            return ClarificationResumeResult(
                derived_source_intent=derived,
                authority=self._resolved(
                    question_id=question_id, derived=derived,
                    company=company, entity_surface=entity_answer.value,
                    receipt=receipt, timepoints=timepoints,
                    amount_slot=amount_slot,
                ),
            )

        lookup = build_event_clarification_options(
            self.canonical, company_surface=entity_answer.value,
            as_of=timepoints[1], target_surface="계약",
            preflight=self.event_preflight,
            corpus_cutoff=self.corpus_cutoff,
        )
        if lookup.status == "resolved" and lookup.resolution is not None:
            candidate = lookup.resolution.candidates[0]
            return ClarificationResumeResult(
                derived_source_intent=derived,
                authority=self._resolved(
                    question_id=question_id, derived=derived,
                    company=company, entity_surface=entity_answer.value,
                    receipt=candidate.seed_rcept_no, timepoints=timepoints,
                    amount_slot=amount_slot,
                ),
            )
        if lookup.status != "ambiguous" or not lookup.options:
            return ClarificationResumeResult(
                derived_source_intent=derived,
                authority=self._terminal(item_id=item.item_id),
            )
        return ClarificationResumeResult(
            derived_source_intent=derived,
            authority=self._event_choice_authority(
                item_id=item.item_id,
                amount_slot=amount_slot,
                options=list(lookup.options),
            ),
        )


class CanonicalGenericEventContextResumeBackend:
    """Close missing issuer context, then require a canonical event choice."""

    def __init__(
            self, canonical: Any, *, canonical_build_id: str,
            resolver_version: str, reference_date: date, corpus_cutoff: str,
            event_preflight: Any | None = None,
            ) -> None:
        self.canonical = canonical
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff
        self.event_preflight = event_preflight

    @staticmethod
    def _retry_entity(item_id: str) -> ClarificationAuthority:
        return ClarificationAuthority.model_validate({
            "kind": "clarification", "slots": [{
                "slot_id": "slot-1", "role_hint": "entity",
                "reason_code": "missing_context",
                "response_kind": "provide_value",
                "prompt": "회사를 정본에서 찾지 못했습니다. 회사명을 다시 알려주세요.",
                "applies_to_item_ids": [item_id], "mention_ids": [],
                "options": [],
            }],
        }, strict=True)

    @staticmethod
    def _terminal(item_id: str) -> TerminalAuthority:
        return TerminalAuthority(reasons=[{
            "code": "corpus_coverage_unavailable", "scope": "items",
            "item_ids": [item_id],
        }])

    def resume(
            self, *, question_id: str, question: str,
            original_source_intent: SemanticIntent,
            current_source_intent: SemanticIntent,
            context: ClarificationResolutionContext,
            ) -> ClarificationResumeResult:
        del question, original_source_intent
        if (len(current_source_intent.answer_items) != 1
                or current_source_intent.entities):
            raise ContextClarificationResolutionError(
                "generic event context는 issuer 없는 event item 하나만 지원합니다")
        item = current_source_intent.answer_items[0]
        if (item.target.kind != "event" or item.target.entity_refs
                or item.operation != "retrieve"):
            raise ContextClarificationResolutionError(
                "generic event context topology가 지원 범위를 벗어납니다")
        derived = _remove_answered_mentions(current_source_intent, context)
        answers = _all_context_answers(context)
        entity_answer = answers.get("entity")
        if entity_answer is None:
            raise ContextClarificationResolutionError(
                "generic event issuer 답변이 없습니다")
        companies = self.canonical.resolve_company(entity_answer.value)
        if len(companies) != 1:
            return ClarificationResumeResult(
                derived_source_intent=derived,
                authority=self._retry_entity(item.item_id))
        company = companies[0]

        target_answer = answers.get("target") or answers.get("event")
        receipt: str | None = None
        if target_answer is None:
            lookup = build_event_clarification_options(
                self.canonical, company_surface=entity_answer.value,
                as_of=self.corpus_cutoff, target_surface=item.target.surface,
                preflight=self.event_preflight,
                corpus_cutoff=self.corpus_cutoff)
            if lookup.status not in {"resolved", "ambiguous"} or not lookup.options:
                return ClarificationResumeResult(
                    derived_source_intent=derived,
                    authority=self._terminal(item.item_id))
            if lookup.status == "resolved" and len(lookup.options) == 1:
                receipt = lookup.options[0].value
            elif lookup.status == "ambiguous" and len(lookup.options) > 1:
                return ClarificationResumeResult(
                    derived_source_intent=derived,
                    authority=ClarificationAuthority.model_validate({
                        "kind": "clarification", "slots": [{
                            "slot_id": "slot-1", "role_hint": "target",
                            "reason_code": "missing_context",
                            "response_kind": "select_one",
                            "prompt": "어떤 계약을 확인할까요?",
                            "applies_to_item_ids": [item.item_id],
                            "mention_ids": [],
                            "options": [row.model_dump(mode="json")
                                        for row in lookup.options],
                        }],
                    }, strict=True))
            else:
                return ClarificationResumeResult(
                    derived_source_intent=derived,
                    authority=self._terminal(item.item_id))

        if receipt is None:
            assert target_answer is not None
            receipt = target_answer.value
        if not (receipt.isdigit() and len(receipt) == 14):
            lookup = build_event_clarification_options(
                self.canonical, company_surface=entity_answer.value,
                as_of=self.corpus_cutoff, target_surface=receipt,
                preflight=self.event_preflight,
                corpus_cutoff=self.corpus_cutoff)
            if lookup.status not in {"resolved", "ambiguous"} or not lookup.options:
                return ClarificationResumeResult(
                    derived_source_intent=derived,
                    authority=self._terminal(item.item_id))
            if lookup.status == "ambiguous" and len(lookup.options) > 1:
                return ClarificationResumeResult(
                    derived_source_intent=derived,
                    authority=ClarificationAuthority.model_validate({
                        "kind": "clarification", "slots": [{
                            "slot_id": "slot-1", "role_hint": "event",
                            "reason_code": "missing_context",
                            "response_kind": "select_one",
                            "prompt": "어떤 계약을 확인할까요?",
                            "applies_to_item_ids": [item.item_id],
                            "mention_ids": [],
                            "options": [row.model_dump(mode="json")
                                        for row in lookup.options],
                        }],
                    }, strict=True))
            receipt = lookup.options[0].value
        timeline = self.canonical.event_timeline(
            as_of=self.corpus_cutoff, rcept_no=receipt,
            verify_evidence=True)
        if timeline is None or (
                timeline.root_rcept_no != receipt
                or timeline.corp_code != company.corp_code
                or timeline.corp_name != company.corp_name):
            return ClarificationResumeResult(
                derived_source_intent=derived,
                authority=self._terminal(item.item_id))
        proof = f"canonical:event-root:{timeline.event_key}:{receipt}"
        resolution = AuthoritativeResolution.create(
            question_id=question_id,
            source_intent_digest=semantic_intent_digest(derived),
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            items=[ResolvedItem(
                item_id=item.item_id, target_surface=item.target.surface,
                resolution=SelectedEventResolution(
                    corp_code=company.corp_code, corp_name=company.corp_name,
                    entity_surface=entity_answer.value,
                    event_key=timeline.event_key, root_receipt=receipt,
                    selector_proof=ResolutionSourceProof(
                        source_receipt=receipt, proof_ref=proof)),
                field_proofs=[ResolutionFieldProof(
                    source_field_index=index, surface=surface,
                    proof_ref=f"source-field:{item.item_id}:{index}")
                    for index, surface in enumerate(
                        item.output.field_surfaces)],
            )],
        )
        return ClarificationResumeResult(
            derived_source_intent=derived,
            authority=ResolvedAuthority(resolution=resolution))


class CanonicalNarrativeMatrixResumeBackend:
    """Resume an over-limit narrative matrix from one regenerated option."""

    def __init__(
            self, canonical: Any, *, canonical_build_id: str,
            resolver_version: str, reference_date: date, corpus_cutoff: str,
            ) -> None:
        self.backend = NarrativeMatrixResolutionBackend(
            canonical, canonical_build_id=canonical_build_id,
            resolver_version=resolver_version, reference_date=reference_date,
            corpus_cutoff=corpus_cutoff)

    def resume(
            self, *, question_id: str, question: str,
            original_source_intent: SemanticIntent,
            current_source_intent: SemanticIntent,
            context: ClarificationResolutionContext,
            ) -> ClarificationResumeResult:
        del original_source_intent
        answers = context.current.answers
        if (len(answers) != 1
                or answers[0].role_hint != "selection"
                or answers[0].reason_code != "narrative_matrix_limit_exceeded"):
            raise ContextClarificationResolutionError(
                "narrative matrix resume 답변 계약이 잘못되었습니다")
        request = self.backend.regrounder(question, current_source_intent)
        if request is None:
            raise ContextClarificationResolutionError(
                "원 질문에서 narrative matrix 범위를 재구성할 수 없습니다")
        choices = {
            choice.option.value: choice
            for choice in narrative_matrix_reduction_options(request)
        }
        selected = choices.get(answers[0].value)
        if selected is None:
            raise ContextClarificationResolutionError(
                "현재 matrix 범위에 없는 선택지입니다")
        resumed_request = apply_narrative_matrix_resume(
            request, selected.selection)
        authority = self.backend.resolve_request(
            question_id=question_id, request=resumed_request,
            source_intent=current_source_intent)
        if authority is None:
            raise ContextClarificationResolutionError(
                "선택한 narrative matrix 범위를 정본에 결속할 수 없습니다")
        resolution = self.backend.authoritative_resolution(
            source_intent=current_source_intent, authority=authority)
        return ClarificationResumeResult(
            derived_source_intent=_remove_answered_mentions(
                current_source_intent, context),
            authority=ResolvedAuthority(resolution=resolution),
        )


class CompositeClarificationResumeBackend:
    """Dispatch event receipt selection or semantic context atomically."""

    def __init__(
            self,
            event_backend: CanonicalSelectedEventResumeBackend,
            financial_backend: CanonicalFinancialContextResumeBackend,
            contract_change_backend: CanonicalContractAmountChangeResumeBackend,
            generic_event_backend: CanonicalGenericEventContextResumeBackend,
            narrative_matrix_backend: CanonicalNarrativeMatrixResumeBackend,
            holding_backend: CanonicalHoldingContextResumeBackend | None = None,
            ) -> None:
        self.event_backend = event_backend
        self.financial_backend = financial_backend
        self.holding_backend = holding_backend
        self.contract_change_backend = contract_change_backend
        self.generic_event_backend = generic_event_backend
        self.narrative_matrix_backend = narrative_matrix_backend

    def resume(self, **kwargs: Any) -> ClarificationResumeResult:
        context: ClarificationResolutionContext = kwargs["context"]
        answers = context.current.answers
        if any(answer.reason_code == "narrative_matrix_limit_exceeded"
               for answer in answers):
            return self.narrative_matrix_backend.resume(**kwargs)
        if any(answer.reason_code in {
                "holding_filer_multiple_candidates",
                "holding_receipt_multiple_candidates",
                "holding_party_multiple_or_unknown",
                } for answer in answers):
            if self.holding_backend is None:
                raise ContextClarificationResolutionError(
                    "holding clarification resume backend가 없습니다")
            return self.holding_backend.resume(**kwargs)
        all_roles = set(_all_context_answers(context))
        # ``value_kind`` is shared by ordinary financial metric
        # clarifications (revenue / operating income / net income) and the
        # four-slot contract amount-change flow.  Route only when the latter's
        # full semantic context is present; otherwise a financial answer such
        # as ``revenue`` is incorrectly parsed as a contract amount kind.
        if {"entity", "timepoint", "value_kind"} <= all_roles:
            return self.contract_change_backend.resume(**kwargs)
        intent: SemanticIntent = kwargs["current_source_intent"]
        if (len(intent.answer_items) == 1
                and intent.answer_items[0].target.kind == "event"
                and not intent.entities
                and not intent.answer_items[0].target.entity_refs):
            return self.generic_event_backend.resume(**kwargs)
        is_event_receipt = (
            len(answers) == 1
            and answers[0].role_hint in {"target", "event"}
            and answers[0].value.isdigit()
            and len(answers[0].value) == 14
        )
        is_named_event_answer = (
            len(answers) == 1
            and answers[0].role_hint in {"target", "event"}
            and len(intent.answer_items) == 1
            and intent.answer_items[0].target.kind in {"event", "document"}
        )
        # A bounded "too_many" contract lookup cannot offer a select-one
        # list.  Its single event value is nevertheless resolved by the same
        # canonical receipt/contract-name backend as ordinary event answers;
        # do not send it to the financial resume path merely because the
        # question asks for the contract amount as a metric.
        item = intent.answer_items[0] if len(intent.answer_items) == 1 else None
        normalized_target = (
            "".join(item.target.surface.split()).casefold()
            if item is not None else ""
        )
        is_contract_amount_event_answer = (
            len(answers) == 1
            and answers[0].role_hint == "event"
            and item is not None
            and item.operation == "retrieve"
            and item.target.kind == "metric"
            and bool(item.target.entity_refs)
            and item.selection is None
            and item.scope.as_of_expression is None
            and not item.scope.scope_qualifier_expressions
            and item.output.projection_mode == "named_fields"
            and bool(item.output.field_surfaces)
            and "계약" in normalized_target
            and "금액" in normalized_target
        )
        backend = (
            self.event_backend
            if (is_event_receipt or is_named_event_answer
                or is_contract_amount_event_answer)
            else self.financial_backend
        )
        return backend.resume(**kwargs)


__all__ = [
    "CanonicalFinancialContextResumeBackend",
    "CanonicalHoldingContextResumeBackend",
    "CanonicalNarrativeMatrixResumeBackend",
    "CanonicalContractAmountChangeResumeBackend",
    "CanonicalGenericEventContextResumeBackend",
    "CompositeClarificationResumeBackend",
    "ContextClarificationResolutionError",
]
