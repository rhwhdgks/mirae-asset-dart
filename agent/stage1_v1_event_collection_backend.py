"""Native resolution backend for canonical event collections.

The selector owns set membership and canonical identity.  This adapter only
binds that typed result to the v1 authoritative-resolution boundary; it does
not inspect Gold fixtures or extract answer values.
"""

from __future__ import annotations

from datetime import date
import re
from typing import Any

from agent.deterministic_plan_compiler_v1 import AuthoritativeResolution
from agent.semantic_intent_v1 import SemanticIntent, semantic_intent_digest
from agent.stage1_v1_event_collection import select_event_collection
from agent.stage1_v1_event_collection import _date_range
from agent.stage1_v1_funding_semantics import (
    funding_categories,
    funding_requested_slots,
)
from agent.stage1_v1_event_family_semantics import (
    bond_face_value_slot,
    facilities_investment_requested_slots,
    is_bond_face_value_request,
    is_facilities_investment_request,
)
from agent.stage1_v1_document_backends import normalize_form_label
from agent.stage1_v1_resolver import build_resolution_premise_proofs


_FULL_YEAR = re.compile(r"(?<![0-9])(?P<year>20[0-9]{2})\s*년(?!\s*[0-9]{1,2}\s*월)")


def _literal_form_field_slot(field: str) -> str:
    # Stage2 (``app/tools/events.py``) registers a full canonical form path
    # for a few slots whose leaf label repeats across unrelated sections of
    # the same exchange filing (예: 국내/해외 발행 각각의 권면총액 옆에 있는
    # 전체 사채 권면총액).  A free-text label match cannot tell those apart;
    # only the exact registered slot name routes to Stage2's disambiguated
    # path lookup.
    if is_bond_face_value_request(field):
        return bond_face_value_slot()
    return re.sub(r"\s+", "", field)


def _single_company_for_collection(
        intent: SemanticIntent, item: Any, canonical: Any,
        ) -> Any | None:
    by_id = {entity.entity_id: entity for entity in intent.entities}
    surfaces = [
        by_id[ref].surface for ref in item.target.entity_refs
        if ref in by_id and by_id[ref].kind_hint == "company"
    ]
    rows = [
        candidates[0] for surface in surfaces
        if len(candidates := canonical.resolve_company(surface)) == 1
    ]
    unique = {row.corp_code: row for row in rows}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _funding_date_range(
        question: str, item: Any, reference_date: date,
        ) -> tuple[str | None, str | None]:
    """Preserve an explicit consecutive multi-year funding scope.

    The shared event parser intentionally returns one ordinary date window.
    Funding comparisons can explicitly name several whole years, so collapsing
    them to the first year would silently omit requested events.  A single
    contiguous range represents consecutive full years exactly.  Disjoint
    years fail closed because ``EventSelector`` has no union-of-ranges field.
    """

    surfaces = [question, *item.scope.target_period_expressions]
    years = sorted({
        int(match.group("year"))
        for surface in surfaces
        for match in _FULL_YEAR.finditer(surface)
    })
    if len(years) < 2:
        return _date_range(question, reference_date)
    if years != list(range(years[0], years[-1] + 1)):
        return None, None
    return f"{years[0]}0101", f"{years[-1]}1231"


class EventCollectionResolutionBackend:
    """Resolve a multi-event request only when the canonical set is unique."""

    def __init__(
            self, canonical: Any, *, canonical_build_id: str,
            resolver_version: str, reference_date: date, corpus_cutoff: str,
            ) -> None:
        self.canonical = canonical
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> "dict[str, Any] | None":
        funding = self._funding_collection(question_id, question, source_intent)
        if funding is not None:
            return funding
        facilities = self._facilities_investment_collection(
            question_id, question, source_intent)
        if facilities is not None:
            return facilities
        literal_form = self._literal_form_document_collection(
            question_id, question, source_intent)
        if literal_form is not None:
            return literal_form
        if (len(source_intent.answer_items) == 1
                and source_intent.answer_items[0].target.qualifier_surfaces
                and any(re.search(r"정정|변경\s*이력|전후|흐름", surface)
                        for surface in source_intent.answer_items[0].output.field_surfaces)):
            return None
        from agent.quantity_contract_collection import select_quantity_contract_collection
        selected = select_quantity_contract_collection(
            self.canonical, source_intent, question, self.corpus_cutoff)
        selected = selected or select_event_collection(
            canonical=self.canonical, intent=source_intent, question=question,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
        )
        if selected is None or len(source_intent.answer_items) != 1:
            return None
        item = source_intent.answer_items[0]
        surfaces = list(item.output.field_surfaces)
        requested_slots = list(selected.requested_slots)
        if not requested_slots:
            return None
        item_payload = {
            "item_id": item.item_id,
            "target_surface": item.target.surface,
            "projection_mode": item.output.projection_mode,
            "resolution": {
                "kind": "event_collection",
                "corp_code": selected.issuer_corp_code,
                "corp_name": selected.issuer_corp_name,
                "as_of": selected.as_of,
                "event_from": selected.event_from,
                "event_to": selected.event_to,
                "requires_termination": selected.requires_termination,
                "root_contracts_with_confirmed_termination": (
                    selected.root_contracts_with_confirmed_termination
                ),
                "event_type": selected.selector_event_type,
                "counterparty": selected.selector_counterparty,
                "keywords": list(selected.selector_keywords),
                "public_task_kind": selected.public_task_kind,
                "availability_query": selected.availability_query,
                "requested_slots": requested_slots,
                "collection_proof_ref": selected.collection_proof_ref,
                "argmax_slot": selected.argmax_slot,
                "argmax_direction": selected.argmax_direction,
                "events": [{
                    "event_key": event.event_key,
                    "root_receipt": event.root_receipt,
                    "matched_receipts": list(event.matched_receipts),
                    "proof_refs": list(event.proof_refs),
                } for event in selected.events],
            },
            "applied_defaults": [],
            "field_proofs": [{
                "proof_ref": f"source-field:{item.item_id}:{index}",
                "source_field_index": index,
                "surface": surface,
            } for index, surface in enumerate(surfaces)],
        }
        resolution = AuthoritativeResolution.create(
            question_id=question_id,
            source_intent_digest=semantic_intent_digest(source_intent),
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            items=[item_payload],
            premise_proofs=build_resolution_premise_proofs(
                source_intent, [item_payload]),
        )
        return {
            "kind": "resolved",
            "resolution": resolution.model_dump(mode="json"),
        }

    def _facilities_investment_collection(
            self, question_id: str, question: str, intent: SemanticIntent,
            ) -> "dict[str, Any] | None":
        if (len(intent.answer_items) != 1 or intent.premises
                or intent.unresolved_mentions):
            return None
        item = intent.answer_items[0]
        if (item.target.qualifier_surfaces
                and any(re.search(r"상태|유효|해지|종료", surface)
                        for surface in item.output.field_surfaces)):
            # One exact filing-date seed plus observation points selects a
            # lifecycle.  It is not an issuer-wide facilities collection.
            return None
        if (item.target.kind not in {"document", "event", "topic"}
                or item.operation not in {"retrieve", "compare"}
                or item.output.projection_mode != "named_fields"
                or item.output.shape not in {
                    "record", "record_list", "comparison"}
                or not is_facilities_investment_request(
                    question, item.target.surface,
                    *item.target.qualifier_surfaces)):
            return None
        company = _single_company_for_collection(intent, item, self.canonical)
        if company is None:
            return None
        start, end = _date_range(question, self.reference_date)
        if start is None or end is None:
            return None
        roots = [
            row for row in self.canonical.documents(
                as_of=self.corpus_cutoff, corp_code=company.corp_code,
                form="신규시설투자등", is_correction=False)
            if start <= str(row.rcept_no)[:8] <= end
        ]
        events = []
        for row in roots:
            timeline = self.canonical.event_timeline(
                as_of=self.corpus_cutoff, rcept_no=row.rcept_no,
                verify_evidence=True)
            if (timeline is None or timeline.root_rcept_no != row.rcept_no
                    or not re.fullmatch(r"[0-9a-f]{32}", timeline.event_key)):
                return None
            receipts = sorted({
                str(observation.rcept_no)
                for observation in timeline.observations
                if str(observation.rcept_no)[:8] <= self.corpus_cutoff
            })
            if not receipts:
                return None
            events.append({
                "event_key": timeline.event_key,
                "root_receipt": row.rcept_no,
                "matched_receipts": receipts,
                "proof_refs": [
                    f"canonical:event:{timeline.event_key}:{receipt}"
                    for receipt in receipts
                ],
            })
        if not events:
            return None
        events.sort(key=lambda event: event["event_key"])
        surfaces = list(item.output.field_surfaces)
        payload = {
            "item_id": item.item_id,
            "target_surface": item.target.surface,
            "projection_mode": item.output.projection_mode,
            "applied_defaults": [],
            "field_proofs": [{
                "proof_ref": f"source-field:{item.item_id}:{index}",
                "source_field_index": index,
                "surface": surface,
            } for index, surface in enumerate(surfaces)],
            "resolution": {
                "kind": "event_collection",
                "corp_code": company.corp_code,
                "corp_name": company.corp_name,
                "as_of": self.corpus_cutoff,
                "event_from": start,
                "event_to": end,
                "requires_termination": False,
                "root_contracts_with_confirmed_termination": False,
                "event_type": "신규시설투자",
                "counterparty": None,
                "keywords": [],
                "public_task_kind": "event",
                "availability_query": False,
                "requested_slots": facilities_investment_requested_slots(
                    surfaces),
                "events": events,
            },
        }
        resolution = AuthoritativeResolution.create(
            question_id=question_id,
            source_intent_digest=semantic_intent_digest(intent),
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            items=[payload],
            premise_proofs=[],
        )
        return {
            "kind": "resolved",
            "resolution": resolution.model_dump(mode="json"),
        }

    def _literal_form_document_collection(
            self, question_id: str, question: str, intent: SemanticIntent,
            ) -> "dict[str, Any] | None":
        """One issuer's single exchange filing addressed by its literal title.

        SG-009 실호출(이슈 #75, 2026-09-03): "카카오가 2025년에 공시한 「주요
        사항보고서(전환사채권발행결정)」의 사채 권면(전자등록)총액은 얼마인가?"
        처럼 낫표로 서식명을 직접 지정한 질문을
        `LiteralDisclosureFormFieldRegrounder`가 ``target.kind=document``,
        ``output.shape=record``, 단일 field surface 모양으로 되돌린다.  이
        메서드는 그 서식명을 ``report_name_contains``로 정확히 하나의 문서로
        좁힌 뒤, 자유 표면 매칭이 아니라 Stage2 슬롯 레지스트리
        (``app/tools/events.py`` ``SLOT_CANONICAL_PATHS``)가 아는 canonical
        slot 이름으로 값을 요청한다.  같은 서식 안에 라벨이 비슷한 필드가
        여럿이면(예 국내/해외 발행 각각의 권면총액과 전체 권면총액) 자유
        표면 매칭은 엉뚱한 셀을 고를 수 있다 — Stage1에서 문서 하나를
        확정하고 정확한 canonical slot 이름을 넘기면 Stage2가 등록된 전체
        경로로 명확히 구분한다.
        """
        if (len(intent.answer_items) != 1 or intent.premises
                or intent.unresolved_mentions):
            return None
        item = intent.answer_items[0]
        if (item.target.kind not in {"document", "event", "topic"}
                or item.operation not in {"retrieve", "compare"}
                or item.output.projection_mode != "named_fields"
                or item.output.shape != "record"
                or len(item.output.field_surfaces) != 1
                or not normalize_form_label(item.target.surface)
                # Only a genuine 낫표 quoted form-name question may reach this
                # single-document path.  Without this literal quoting check,
                # any other "document, one field, resolves to one candidate
                # document by coincidence" intent (예 이슈 #74 의 termination
                # existence 질문) could be silently rerouted here instead of
                # its intended handler.
                or not (f"「{item.target.surface}」" in question
                        or f"『{item.target.surface}』" in question)):
            return None
        company = _single_company_for_collection(intent, item, self.canonical)
        if company is None:
            return None
        start, end = _date_range(question, self.reference_date)
        try:
            candidates = list(self.canonical.documents(
                as_of=self.corpus_cutoff, corp_code=company.corp_code,
                report_name_contains=item.target.surface, is_correction=False))
        except (AttributeError, TypeError, ValueError):
            return None
        if start is not None:
            candidates = [
                row for row in candidates
                if start <= str(row.rcept_no)[:8] <= end]
        if len(candidates) != 1:
            return None
        root = candidates[0]
        if not str(getattr(root, "form", "") or "").strip():
            return None
        timeline = self.canonical.event_timeline(
            as_of=self.corpus_cutoff, rcept_no=root.rcept_no,
            verify_evidence=True)
        if (timeline is None or timeline.root_rcept_no != root.rcept_no
                or not re.fullmatch(r"[0-9a-f]{32}", timeline.event_key)):
            return None
        receipts = sorted({
            str(observation.rcept_no)
            for observation in timeline.observations
            if str(observation.rcept_no)[:8] <= self.corpus_cutoff
        })
        if not receipts:
            return None
        slot = _literal_form_field_slot(item.output.field_surfaces[0])
        surfaces = list(item.output.field_surfaces)
        payload = {
            "item_id": item.item_id,
            "target_surface": item.target.surface,
            "projection_mode": item.output.projection_mode,
            "applied_defaults": [],
            "field_proofs": [{
                "proof_ref": f"source-field:{item.item_id}:{index}",
                "source_field_index": index,
                "surface": surface,
            } for index, surface in enumerate(surfaces)],
            "resolution": {
                "kind": "event_collection",
                "corp_code": company.corp_code,
                "corp_name": company.corp_name,
                "as_of": self.corpus_cutoff,
                "event_from": start,
                "event_to": end,
                "requires_termination": False,
                "root_contracts_with_confirmed_termination": False,
                # Stage2 candidate discovery (``app/tools/events.py``
                # ``find_candidates``) matches ``event_type`` against the
                # canonical DART form code (``Document.form``), not the
                # human-facing report title — a literal report title still
                # narrowed Stage1 to one proven document above, but it would
                # find zero candidates here (e.g. "전환사채권발행결정" is the
                # form; "주요사항보고서(전환사채권발행결정)" is only the
                # report_nm wrapper around it).
                "event_type": root.form,
                "counterparty": None,
                "keywords": [],
                "public_task_kind": "disclosure",
                "availability_query": False,
                "requested_slots": [slot],
                "events": [{
                    "event_key": timeline.event_key,
                    "root_receipt": root.rcept_no,
                    "matched_receipts": receipts,
                    "proof_refs": [
                        f"canonical:event:{timeline.event_key}:{receipt}"
                        for receipt in receipts],
                }],
            },
        }
        resolution = AuthoritativeResolution.create(
            question_id=question_id,
            source_intent_digest=semantic_intent_digest(intent),
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            items=[payload],
            premise_proofs=[],
        )
        return {
            "kind": "resolved",
            "resolution": resolution.model_dump(mode="json"),
        }

    def _funding_collection(self, question_id: str, question: str, intent: SemanticIntent) -> "dict[str, Any] | None":
        """One issuer's public financing-decision family in an explicit year.

        Categories are DART disclosure vocabulary; the canonical document
        scan proves the in-range membership and preserves correction lineage.
        """
        if len(intent.answer_items) != 1 or intent.premises or intent.unresolved_mentions:
            return None
        item = intent.answer_items[0]
        categories = funding_categories(
            question, item.target.surface, *item.output.field_surfaces)
        if (item.target.kind not in {"document", "event", "topic"}
                # A type enumeration is a list request even when the model
                # flattens it to a record.  The four literal categories still
                # make membership and the requested grouping unambiguous.
                or item.output.shape not in {
                    "record_list", "record", "comparison", "narrative"}
                # Correction-lineage regrounding deliberately turns an
                # open-ended "what changed" request into a whole-target
                # narrative.  Funding categories can still occur in that
                # question (for example a rights offering), but they are a
                # selected event identity rather than a collection request.
                # Admitting an empty whole-target projection here creates an
                # invalid EventCollectionResolution with no requested slots
                # and prevents the correction backend from seeing it.
                or item.output.projection_mode != "named_fields"
                or not item.output.field_surfaces
                or not categories
                or (len(categories) < 2 and "자금조달" not in question)):
            return None
        companies = [self.canonical.resolve_company(entity.surface) for entity in intent.entities
                     if entity.kind_hint == "company"]
        rows = [row[0] for row in companies if len(row) == 1]
        if len({row.corp_code for row in rows}) != 1:
            return None
        company = rows[0]
        start, end = _funding_date_range(question, item, self.reference_date)
        if start is None or end is None:
            return None
        docs = [doc for doc in self.canonical.documents(corp_code=company.corp_code,
                  as_of=self.corpus_cutoff, doc_group="major")
                if start <= doc.rcept_no[:8] <= end
                and any(keyword in str(getattr(doc, "report_nm", ""))
                        for keyword in categories)]
        by_event: dict[str, list[Any]] = {}
        for doc in docs:
            key = self.canonical.event_of(doc.rcept_no)
            if isinstance(key, str) and key:
                by_event.setdefault(key, []).append(doc)
        if not by_event:  # category absence is answerable only after a scoped scan;
            return None    # no positive canonical identity means no safe support root.
        events = []
        for key, members in sorted(by_event.items()):
            members.sort(key=lambda row: row.rcept_no)
            events.append({"event_key": key, "root_receipt": members[0].rcept_no,
                           "matched_receipts": [row.rcept_no for row in members],
                           "proof_refs": [f"canonical:event:{key}:{row.rcept_no}" for row in members]})
        payload = {"item_id": item.item_id, "target_surface": item.target.surface,
            "projection_mode": item.output.projection_mode, "applied_defaults": [],
            "field_proofs": [{"proof_ref": f"source-field:{item.item_id}:{i}",
                              "source_field_index": i, "surface": surface}
                             for i, surface in enumerate(item.output.field_surfaces)],
            "resolution": {"kind": "event_collection", "corp_code": company.corp_code,
              "corp_name": company.corp_name, "as_of": self.corpus_cutoff,
              "event_from": start, "event_to": end, "requires_termination": False,
              "root_contracts_with_confirmed_termination": False,
              "event_type": "주요사항보고", "counterparty": None,
              "keywords": list(categories), "public_task_kind": "disclosure",
              "availability_query": False,
              "requested_slots": funding_requested_slots(
                  item.output.field_surfaces), "events": events}}
        resolution = AuthoritativeResolution.create(question_id=question_id,
            source_intent_digest=semantic_intent_digest(intent), canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version, reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff, items=[payload], premise_proofs=[])
        # Question trace is repaired by outer resolver only if it accepts this authority.
        return {"kind": "resolved", "resolution": resolution.model_dump(mode="json")}


__all__ = ["EventCollectionResolutionBackend"]
