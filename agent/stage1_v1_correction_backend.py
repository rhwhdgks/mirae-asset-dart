"""Adapter from correction-lineage selection to compiler-neutral authority.

The deterministic compiler does not yet expose a ``correction_lineage``
resolution kind.  Keep this adapter typed and isolated so composition can add
it only with that compiler union/handler in the same change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Any, Literal

from .deterministic_plan_compiler_v1 import AuthoritativeResolution
from .semantic_intent_v1 import SemanticIntent
from .semantic_intent_v1 import semantic_intent_digest
from .stage1_v1_correction_lineage import (
    CanonicalCorrectionLineageSelector,
    CorrectionDiffResolution,
    ContractAmountHistoryResolution,
    RootMissingCorrectionResolution,
)


@dataclass(frozen=True, slots=True)
class CorrectionLineageIntermediate:
    """Compiler-ready coordinate inventory without answer values."""

    kind: Literal["correction_lineage"]
    operation: Literal["diff", "history"]
    issuer_corp_code: str
    issuer_corp_name: str
    event_key: str
    root_receipt: str
    correction_receipt: str
    correction_date: str
    event_proof_refs: tuple[str, ...]
    changed_paths: tuple[str, ...]
    change_proof_refs: tuple[str, ...]
    question_premises: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RootMissingCorrectionIntermediate:
    """Compiler-neutral partial lineage, with no invented root coordinate."""

    kind: Literal["root_missing_correction"]
    issuer_corp_code: str
    issuer_corp_name: str
    correction_receipt: str
    correction_date: str
    relation_proof_ref: str
    target_hint: str
    limitation_code: Literal["source_scope_prevents_complete_lineage"]
    question_premises: tuple[str, ...]


class CorrectionLineageResolutionBackend:
    """Lower canonical lineage coordinates into compiler authority.

    The selector remains the only code that chooses issuer/event/receipts.
    This adapter copies those proof-bearing coordinates into a closed typed
    resolution; it never reads a Gold record or a fixture answer.
    """

    def __init__(self, selector: CanonicalCorrectionLineageSelector, *,
                 canonical_build_id: str = "a" * 32,
                 resolver_version: str = "stage1-resolver/1.0",
                 reference_date: date = date(2026, 6, 19),
                 corpus_cutoff: str | None = None) -> None:
        self.selector = selector
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff or getattr(selector, "_cutoff", None)
        if not isinstance(self.corpus_cutoff, str):
            raise TypeError("correction lineage corpus_cutoff 계약이 잘못되었습니다")

    @staticmethod
    def _same_day_chain_is_proven(observations: list[Any]) -> bool:
        grouped: dict[str, list[Any]] = {}
        for observation in observations:
            grouped.setdefault(str(getattr(observation, "observed_at", "")), []).append(
                observation)
        for same_day in grouped.values():
            if len(same_day) < 2:
                continue
            by_receipt = {
                getattr(row, "rcept_no", None): row for row in same_day}
            if len(by_receipt) != len(same_day) or None in by_receipt:
                return False
            internal_links = [
                row for row in same_day
                if getattr(row, "previous_observation_rcept_no", None) in by_receipt]
            if len(internal_links) != len(same_day) - 1:
                return False
        return True

    def _correction_collection(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> dict[str, object] | None:
        """Build one bounded issuer-wide correction collection authority."""

        if (len(source_intent.answer_items) != 1 or source_intent.premises
                or source_intent.unresolved_mentions):
            return None
        item = source_intent.answer_items[0]
        fields = list(item.output.field_surfaces)
        if (item.output.projection_mode != "named_fields"
                or len(fields) != 3
                or not any("계약별" in value.replace(" ", "") for value in fields)
                or not any("전후" in value.replace(" ", "")
                           and "금액" in value.replace(" ", "") for value in fields)
                or not any(re.search(r"차이|증감", value) for value in fields)
                or re.search(r"정정\s*공시", question) is None):
            return None
        entities = {entity.entity_id: entity for entity in source_intent.entities}
        companies = [
            entities[ref] for ref in item.target.entity_refs
            if ref in entities and entities[ref].kind_hint == "company"]
        if len(companies) != 1:
            return None
        canonical = getattr(self.selector, "_canonical", None)
        if canonical is None or not callable(getattr(canonical, "documents", None)):
            return None
        resolved_companies = canonical.resolve_company(companies[0].surface)
        if len(resolved_companies) != 1:
            return None
        company = resolved_companies[0]
        corp_code = getattr(company, "corp_code", None)
        corp_name = getattr(company, "corp_name", None)
        if not isinstance(corp_code, str) or not isinstance(corp_name, str):
            return None

        family_surface = re.sub(
            r"\s*(?:기재\s*)?정정\s*공시.*$", "",
            item.target.surface).strip()
        family_key = re.sub(
            r"[^0-9a-z가-힣]+", "", family_surface.casefold())
        if len(family_key) < 3:
            return None
        matching_receipts: list[str] = []
        try:
            for document in canonical.documents(
                    as_of=self.corpus_cutoff, corp_code=corp_code,
                    doc_group="exchange", is_correction=True):
                searchable = re.sub(
                    r"[^0-9a-z가-힣]+", "",
                    (f"{getattr(document, 'form', '')} "
                     f"{getattr(document, 'report_nm', '')}").casefold())
                receipt = str(getattr(document, "rcept_no", ""))
                if (family_key not in searchable
                        or re.fullmatch(r"[0-9]{14}", receipt) is None
                        or receipt[:8] > self.corpus_cutoff):
                    continue
                matching_receipts.append(receipt)
                if len(matching_receipts) > 128:
                    return None
        except Exception:
            return None
        if not matching_receipts:
            return None

        timelines: dict[str, Any] = {}
        correction_receipts: set[str] = set()
        for receipt in sorted(set(matching_receipts)):
            try:
                timeline = canonical.event_timeline(
                    as_of=self.corpus_cutoff, rcept_no=receipt,
                    verify_evidence=True)
            except Exception:
                continue
            event_key = getattr(timeline, "event_key", None)
            if (timeline is None or getattr(timeline, "corp_code", None) != corp_code
                    or not isinstance(event_key, str)
                    or re.fullmatch(r"[0-9a-f]{32}", event_key) is None
                    or getattr(timeline, "identity_status", None)
                    in {None, "ambiguous", "unavailable"}):
                continue
            corrections = [
                row for row in tuple(getattr(timeline, "observations", ()) or ())
                if bool(getattr(row, "is_correction", False))
                and str(getattr(row, "observed_at", "")) <= self.corpus_cutoff]
            if not corrections or not self._same_day_chain_is_proven(corrections):
                continue
            timelines[event_key] = timeline
            correction_receipts.update(
                str(getattr(row, "rcept_no", "")) for row in corrections)
            if len(timelines) > 64 or len(correction_receipts) > 128:
                return None
        if not timelines:
            return None

        try:
            correction_rows = tuple(canonical.correction_items(
                as_of=self.corpus_cutoff, corp_code=corp_code,
                rcept_nos=tuple(sorted(correction_receipts)),
                include_restricted_raw=False, verify_evidence=True))
        except (TypeError, ValueError):
            return None
        if len(correction_rows) > 4096:
            return None
        amount_receipts = {
            str(getattr(row, "rcept_no", "")) for row in correction_rows
            if "계약금액" in re.sub(r"\s+", "", str(getattr(row, "path", "")))
            and (
                (getattr(row, "before_evidence_status", None) == "verified"
                 and getattr(row, "before_evidence_id", None))
                or (getattr(row, "after_evidence_status", None) == "verified"
                    and getattr(row, "after_evidence_id", None)))
        }
        events: list[dict[str, object]] = []
        for event_key, timeline in sorted(timelines.items()):
            receipts = sorted({
                str(getattr(row, "rcept_no", ""))
                for row in tuple(getattr(timeline, "observations", ()) or ())
                if bool(getattr(row, "is_correction", False))
                and str(getattr(row, "rcept_no", "")) in amount_receipts
            })
            root = str(getattr(timeline, "root_rcept_no", ""))
            if (not receipts or re.fullmatch(r"[0-9]{14}", root) is None):
                continue
            events.append({
                "event_key": event_key, "root_receipt": root,
                "matched_receipts": receipts,
                "proof_refs": [
                    f"canonical:event:{event_key}:{receipt}"
                    for receipt in receipts],
            })
        if not events:
            return None
        resolution_item = {
            "item_id": item.item_id,
            "target_surface": item.target.surface,
            "projection_mode": item.output.projection_mode,
            "resolution": {
                "kind": "event_collection", "corp_code": corp_code,
                "corp_name": corp_name, "as_of": self.corpus_cutoff,
                "event_from": None, "event_to": None,
                "requires_termination": False,
                "root_contracts_with_confirmed_termination": False,
                "event_type": family_surface, "counterparty": None,
                "keywords": [], "public_task_kind": "correction",
                "availability_query": False,
                "requested_slots": [
                    "계약별구분", "계약금액", "difference"],
                "events": events,
            },
            "field_proofs": [{
                "source_field_index": index, "surface": surface,
                "proof_ref": f"source-field:{item.item_id}:{index}",
            } for index, surface in enumerate(fields)],
            "applied_defaults": [],
        }
        authority = AuthoritativeResolution.create(
            question_id=question_id,
            source_intent_digest=semantic_intent_digest(source_intent),
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            items=[resolution_item], premise_proofs=[])
        return {"kind": "resolved", "resolution": authority.model_dump(mode="json")}

    def build_payload(
            self, *, question: str, source_intent: SemanticIntent,
            cutoff: str,
            ) -> CorrectionLineageIntermediate | RootMissingCorrectionIntermediate | None:
        result = self.selector.resolve(
            question=question, semantic_intent=source_intent, cutoff=cutoff)
        typed = result.resolution
        if typed is None:
            return None
        if isinstance(typed, CorrectionDiffResolution):
            operation: Literal["diff", "history"] = "diff"
            changes = typed.changes
        elif isinstance(typed, ContractAmountHistoryResolution):
            operation = "history"
            changes = (typed.amount_change,)
        elif isinstance(typed, RootMissingCorrectionResolution):
            return RootMissingCorrectionIntermediate(
                kind="root_missing_correction",
                issuer_corp_code=typed.issuer_corp_code,
                issuer_corp_name=typed.issuer_corp_name,
                correction_receipt=typed.correction_receipt,
                correction_date=typed.correction_date,
                relation_proof_ref=typed.relation_proof_ref,
                target_hint=typed.target_hint,
                limitation_code="source_scope_prevents_complete_lineage",
                question_premises=typed.question_premises,
            )
        else:  # closed union defensive guard
            return None
        refs = tuple(
            proof.proof_ref for change in changes
            for proof in (change.before_proof, change.after_proof)
            if proof is not None)
        if not refs:
            return None
        return CorrectionLineageIntermediate(
            kind="correction_lineage", operation=operation,
            issuer_corp_code=typed.issuer_corp_code,
            issuer_corp_name=typed.issuer_corp_name,
            event_key=typed.event_key, root_receipt=typed.root_receipt,
            correction_receipt=typed.correction_receipt,
            correction_date=typed.correction_date,
            event_proof_refs=tuple(proof.proof_ref for proof in typed.event_proofs),
            changed_paths=tuple(change.path for change in changes),
            change_proof_refs=refs, question_premises=typed.question_premises,
        )

    @staticmethod
    def _typed_payload(typed: CorrectionDiffResolution | ContractAmountHistoryResolution,
                       *, answer_role: Literal["diff", "amount", "reason"],
                       operation: Literal["diff", "history"],) -> dict[str, object]:
        changes = typed.changes if isinstance(typed, CorrectionDiffResolution) else (typed.amount_change,)
        payload: dict[str, object] = {
            "kind": "correction_lineage", "operation": operation,
            "answer_role": answer_role, "corp_code": typed.issuer_corp_code,
            "corp_name": typed.issuer_corp_name, "event_key": typed.event_key,
            "root_receipt": typed.root_receipt,
            "correction_receipt": typed.correction_receipt,
            "correction_date": typed.correction_date,
            "root_selector_proof": {
                "source_receipt": typed.root_receipt,
                "proof_ref": f"canonical:event:{typed.event_key}:{typed.root_receipt}",
            },
            "correction_selector_proof": {
                "source_receipt": typed.correction_receipt,
                "proof_ref": f"canonical:event:{typed.event_key}:{typed.correction_receipt}",
            },
            "event_proofs": [
                {"source_receipt": proof.source_receipt, "proof_ref": proof.proof_ref}
                for proof in typed.event_proofs
            ],
            "changes": [
                {"path": change.path,
                 "before_proof": (None if change.before_proof is None else {
                     "source_receipt": change.before_proof.source_receipt,
                     "proof_ref": change.before_proof.proof_ref}),
                 "after_proof": (None if change.after_proof is None else {
                     "source_receipt": change.after_proof.source_receipt,
                     "proof_ref": change.after_proof.proof_ref})}
                for change in changes
            ],
            "counterparty": typed.counterparty,
            "product_keywords": list(typed.product_keywords),
        }
        if (isinstance(typed, CorrectionDiffResolution)
                and typed.source_root_missing):
            payload["source_root_missing"] = True
        # Preserve the legacy one-receipt payload byte-for-byte.  The internal
        # sidecar is only needed when the question assigns an explicit root
        # observation/range or the canonical lineage actually has >1
        # correction step.
        emit_sequence = (
            isinstance(typed, CorrectionDiffResolution)
            and bool(typed.sequence)
            and typed.date_roles is not None
            and (
                typed.date_roles.range_requested
                or typed.date_roles.root_observed_at is not None
                or len(typed.sequence) > 1
            )
        )
        if emit_sequence:
            payload["date_roles"] = (
                None if typed.date_roles is None else {
                    "root_observed_at": typed.date_roles.root_observed_at,
                    "correction_from": typed.date_roles.correction_from,
                    "correction_to": typed.date_roles.correction_to,
                    "as_of": typed.date_roles.as_of,
                    "range_requested": typed.date_roles.range_requested,
                })
            payload["sequence"] = [{
                "correction_receipt": step.correction_receipt,
                "correction_date": step.correction_date,
                "previous_observation_receipt": step.previous_observation_receipt,
                "event_proofs": [
                    {"source_receipt": proof.source_receipt,
                     "proof_ref": proof.proof_ref}
                    for proof in step.event_proofs],
                "changes": [
                    {"path": change.path,
                     "before_proof": (None if change.before_proof is None else {
                         "source_receipt": change.before_proof.source_receipt,
                         "proof_ref": change.before_proof.proof_ref}),
                     "after_proof": (None if change.after_proof is None else {
                         "source_receipt": change.after_proof.source_receipt,
                         "proof_ref": change.after_proof.proof_ref})}
                    for change in step.changes],
            } for step in typed.sequence]
        return payload

    def resolve(self, *, question_id: str, question: str,
                source_intent: SemanticIntent) -> dict[str, object] | None:
        collection = self._correction_collection(
            question_id=question_id, question=question,
            source_intent=source_intent)
        if collection is not None:
            return collection
        result = self.selector.resolve(question=question, semantic_intent=source_intent,
                                       cutoff=self.corpus_cutoff)
        typed = result.resolution
        if typed is None:
            return None
        if isinstance(typed, CorrectionDiffResolution):
            if len(source_intent.answer_items) != 1:
                return None
            item = source_intent.answer_items[0]
            # A bare mention of ``정정`` is not enough to seize a request that
            # still has named comparison fields.  The correction-diff compiler
            # owns a whole-target narrative diff; a fielded amount/reason
            # request must first be question-regrounded or be left for the
            # document comparison authority.  Declining here is essential:
            # otherwise an invalid partial payload blocks the later backend.
            if (item.output.projection_mode != "whole_target"
                    or item.output.field_surfaces):
                return None
            operation: Literal["diff", "history"] = (
                "history"
                if (typed.date_roles is not None
                    and typed.date_roles.range_requested)
                else "diff"
            )
            items = [{
                "item_id": item.item_id, "target_surface": item.target.surface,
                "projection_mode": item.output.projection_mode,
                "resolution": self._typed_payload(
                    typed, answer_role="diff", operation=operation),
                "field_proofs": [], "applied_defaults": [],
            }]
            premise_proofs: list[dict[str, object]] = []
        elif isinstance(typed, ContractAmountHistoryResolution):
            if len(source_intent.answer_items) != 2 or len(source_intent.premises) != 2:
                return None
            roles: tuple[Literal["amount", "reason"], Literal["amount", "reason"]] = ("amount", "reason")
            items = [{
                "item_id": item.item_id, "target_surface": item.target.surface,
                "projection_mode": item.output.projection_mode,
                "resolution": self._typed_payload(typed, answer_role=role, operation="history"),
                "field_proofs": [{"source_field_index": 0,
                                   "surface": item.output.field_surfaces[0],
                                   "proof_ref": f"source-field:{item.item_id}:0"}],
                "applied_defaults": [],
            } for item, role in zip(source_intent.answer_items, roles)]
            refs = [proof.proof_ref for proof in (typed.amount_change.before_proof,
                                                  typed.amount_change.after_proof)
                    if proof is not None]
            if not refs:
                return None
            premise_proofs = [{"premise_id": premise.premise_id, "proof_refs": refs}
                              for premise in source_intent.premises]
        elif isinstance(typed, RootMissingCorrectionResolution):
            # The frozen compiler union has no root-missing correction kind.
            # Keep the proven partial available through ``build_payload`` and
            # decline here until the compiler/handoff owner wires that typed
            # intermediate.  Emitting a fake root receipt would be unsafe.
            return None
        else:
            return None
        authority = AuthoritativeResolution.create(
            question_id=question_id, source_intent_digest=semantic_intent_digest(source_intent),
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version, reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff, items=items, premise_proofs=premise_proofs)
        return {"kind": "resolved", "resolution": authority.model_dump(mode="json")}


__all__ = [
    "CorrectionLineageIntermediate",
    "CorrectionLineageResolutionBackend",
    "RootMissingCorrectionIntermediate",
]
