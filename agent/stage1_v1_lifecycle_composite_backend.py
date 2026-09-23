"""Native authority adapter for a proven canonical event lifecycle."""

from __future__ import annotations

from datetime import date
import re
from typing import Any

from agent.deterministic_plan_compiler_v1 import AuthoritativeResolution
from agent.semantic_intent_v1 import SemanticIntent, semantic_intent_digest
from agent.stage1_v1_lifecycle_composite import select_lifecycle_composite
from agent.stage1_v1_resolver import build_resolution_premise_proofs


class LifecycleCompositeResolutionBackend:
    """Bind one unambiguous lifecycle to every semantic item that asks for it.

    Same-day and event-identity ambiguity intentionally return ``None``: the
    existing partial/clarification backends own those limitations.
    """

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
            ) -> dict[str, Any] | None:
        # The lifecycle compiler handler binds only a self-contained item set:
        # it rejects an intent carrying a premise or an unresolved mention.
        # Claiming one here would resolve successfully and then fail closed as
        # a *technical* compiler error, losing the answer entirely, so the
        # acceptance test matches the compiler contract.
        if source_intent.premises or source_intent.unresolved_mentions:
            return None
        # A single, exact-date status assertion plus a termination amount is
        # owned by the existing reported-termination authority.  That path
        # preserves ambiguous-original provenance and adds the stated reason
        # as a verification slot; a lifecycle coordinate must not erase it.
        if self._is_reported_termination_status_request(source_intent):
            return None
        # lifecycle 은 「이 사건에 무슨 일이 있었나」를 답한다 — 질문이 이미
        # 사건을 못박았을 때의 권위다.  아직 「가장 최근 건」처럼 **고르는 중**
        # 이면 그건 「어느 사건인가」를 묻는 것이고, 선택된-사건 경로의 일이다.
        # 여기서 낚아채면 정체(계약상대·계약명)를 물었는데 생애 계획을 세운다.
        if (any(
                getattr(item.selection, "mode", None) == "latest"
                for item in source_intent.answer_items)
                and not self._is_latest_document_and_status_request(
                    source_intent)):
            return None
        selected = select_lifecycle_composite(
            canonical=self.canonical, intent=source_intent, question=question,
            reference_date=self.reference_date, corpus_cutoff=self.corpus_cutoff)
        if selected is None:
            return None
        if selected.status == "ambiguous_same_day":
            return self._resolve_same_day_latest_and_status(
                question_id=question_id, question=question, intent=source_intent)
        if (selected.status != "resolved" or selected.event_key is None
                or selected.root_receipt is None):
            return None
        # A mere contract match is not lifecycle authority.  It must carry an
        # observed status, a correction flow, or a source-proven attribute.
        if not (selected.status_timepoints or selected.correction_receipts
                or selected.attributes):
            return None
        receipts = sorted({
            selected.root_receipt, *selected.termination_receipts,
            *selected.correction_receipts,
            *(attribute.source_receipt for attribute in selected.attributes),
        })
        event_proofs = [{
            "source_receipt": receipt,
            "proof_ref": f"canonical:event:{selected.event_key}:{receipt}",
        } for receipt in receipts]
        resolution_payload = {
            "kind": "event_lifecycle_composite",
            "corp_code": selected.issuer_corp_code,
            "corp_name": selected.issuer_corp_name,
            "as_of": selected.as_of,
            "event_key": selected.event_key,
            "root_receipt": selected.root_receipt,
            "selector_proof": event_proofs[receipts.index(selected.root_receipt)],
            "event_proofs": event_proofs,
            "status_timepoints": list(selected.status_timepoints),
            "termination_receipts": list(selected.termination_receipts),
            "correction_receipts": list(selected.correction_receipts),
            "attributes": [{
                "kind": attribute.kind,
                "source_receipt": attribute.source_receipt,
                "evidence_id": attribute.evidence_id,
                "path": attribute.path,
                "locator": attribute.locator,
            } for attribute in selected.attributes],
        }
        from agent.stage1_v1_public_event_selector import (
            select_public_event_selector,
        )
        selector_payloads: list[dict[str, Any]] = []
        counterparty_surface = (
            selected.counterparty_terms[0]
            if len(selected.counterparty_terms) == 1 else None
        )
        for source_item in source_intent.answer_items:
            public_selector = select_public_event_selector(
                self.canonical, as_of=self.corpus_cutoff,
                receipt=selected.root_receipt, intent=source_intent,
                item=source_item, counterparty_surface=counterparty_surface,
                # A lifecycle/correction flow may identify the unique event by
                # a canonical attribute rather than repeating its product.
                # The already selected root contract may expose that product,
                # but only through an exact field proof.
                allow_canonical_product=True,
            )
            if not public_selector.empty:
                selector_payloads.append(public_selector.as_payload())
        unique_selectors = {
            repr(sorted(payload.items())): payload for payload in selector_payloads
        }
        if len(unique_selectors) == 1:
            resolution_payload["public_selector"] = next(
                iter(unique_selectors.values()))
        items: list[dict[str, Any]] = []
        for item in source_intent.answer_items:
            fields = list(item.output.field_surfaces)
            if not fields:
                return None
            items.append({
                "item_id": item.item_id,
                "target_surface": item.target.surface,
                "projection_mode": item.output.projection_mode,
                "resolution": resolution_payload,
                "applied_defaults": [],
                "field_proofs": [{
                    "proof_ref": f"source-field:{item.item_id}:{index}",
                    "source_field_index": index, "surface": surface,
                } for index, surface in enumerate(fields)],
            })
        resolution = AuthoritativeResolution.create(
            question_id=question_id,
            source_intent_digest=semantic_intent_digest(source_intent),
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date, corpus_cutoff=self.corpus_cutoff,
            items=items,
            premise_proofs=build_resolution_premise_proofs(source_intent, items),
        )
        return {"kind": "resolved", "resolution": resolution.model_dump(mode="json")}

    @staticmethod
    def _is_latest_document_and_status_request(intent: SemanticIntent) -> bool:
        """Admit only the closed latest-document plus event-status topology.

        Ordinary ``latest`` requests remain outside lifecycle authority.  The
        exception is the two-item composite whose document half needs
        same-day ordering provenance while its event half needs an independent
        status authority.
        """

        if len(intent.answer_items) != 2:
            return False
        latest = [
            item for item in intent.answer_items
            if item.selection is not None and item.selection.mode == "latest"
            and item.target.kind == "document"
        ]
        status = [
            item for item in intent.answer_items
            if item.selection is None and item.target.kind == "event"
            and any(re.search(r"상태|유효|최종", surface)
                    for surface in item.output.field_surfaces)
        ]
        return len(latest) == 1 and len(status) == 1

    @staticmethod
    def _is_reported_termination_status_request(intent: SemanticIntent) -> bool:
        if len(intent.answer_items) != 1:
            return False
        item = intent.answer_items[0]
        fields = " ".join(item.output.field_surfaces)
        return bool(
            item.target.kind == "event"
            and item.scope.as_of_expression is not None
            and item.target.qualifier_surfaces == [item.scope.as_of_expression]
            and re.search(r"상태|유효|살아\s*있|끝난|해지", fields)
            and re.search(r"해지\s*금액", fields)
        )

    def _resolve_same_day_latest_and_status(
            self, *, question_id: str, question: str, intent: SemanticIntent,
            ) -> dict[str, Any] | None:
        """Combine only an actual latest-document item with a status item.

        A same-day pair cannot decide intraday recency, while the companion
        state request can still be answered through the reported-termination
        authority.  Narrowing each backend to its own semantic item prevents
        the document ambiguity from replacing the lifecycle request.
        """
        latest = [item for item in intent.answer_items
                  if item.selection is not None and item.selection.mode == "latest"]
        status = [item for item in intent.answer_items
                  if item.selection is None and any(
                      marker in " ".join(item.output.field_surfaces)
                      for marker in ("상태", "유효", "최종"))]
        if len(latest) != 1 or len(status) != 1 or len(intent.answer_items) != 2:
            return None
        from agent.stage1_v1_document_backends import (
            SameDayDocumentCandidatesBackend, TerminationReportedStatusBackend,
        )
        common = {
            "canonical_build_id": self.canonical_build_id,
            "resolver_version": self.resolver_version,
            "reference_date": self.reference_date,
            "corpus_cutoff": self.corpus_cutoff,
        }
        # A narrowed resolver input cannot retain groups that reference the
        # omitted item; groups are source-presentation topology, not evidence.
        latest_original, status_original = latest[0], status[0]
        # Existing single-item authorities deliberately require their input to
        # have the local deterministic ``item-1`` identity.  The combined
        # authority restores the original source IDs below.
        latest_narrow = latest_original.model_copy(update={"item_id": "item-1"})
        status_narrow = status_original.model_copy(update={"item_id": "item-1"})
        latest_intent = intent.model_copy(update={
            "answer_items": [latest_narrow], "answer_groups": [], "premises": [],
            "unresolved_mentions": [],
        })
        status_intent = intent.model_copy(update={
            "answer_items": [status_narrow], "answer_groups": [], "premises": [],
            "unresolved_mentions": [],
        })
        latest_answer = SameDayDocumentCandidatesBackend(self.canonical, **common).resolve(
            question_id=question_id, question=question, source_intent=latest_intent)
        status_answer = TerminationReportedStatusBackend(self.canonical, **common).resolve(
            question_id=question_id, question=question, source_intent=status_intent)
        if not isinstance(latest_answer, dict) or not isinstance(status_answer, dict):
            return None
        latest_resolution = latest_answer.get("resolution")
        status_resolution = status_answer.get("resolution")
        if not isinstance(latest_resolution, dict) or not isinstance(status_resolution, dict):
            return None
        latest_rows = latest_resolution.get("items")
        status_rows = status_resolution.get("items")
        if not (isinstance(latest_rows, list) and len(latest_rows) == 1
                and isinstance(status_rows, list) and len(status_rows) == 1):
            return None
        def bind(row: dict[str, Any], item: Any) -> dict[str, Any]:
            value = dict(row)
            value["item_id"] = item.item_id
            value["target_surface"] = item.target.surface
            value["projection_mode"] = item.output.projection_mode
            value["field_proofs"] = [{
                "proof_ref": f"source-field:{item.item_id}:{index}",
                "source_field_index": index, "surface": surface,
            } for index, surface in enumerate(item.output.field_surfaces)]
            return value
        items = [
            bind(latest_rows[0], latest_original),
            bind(status_rows[0], status_original),
        ]
        items.sort(key=lambda row: int(str(row["item_id"]).rsplit("-", 1)[-1]))
        resolution = AuthoritativeResolution.create(
            question_id=question_id, source_intent_digest=semantic_intent_digest(intent),
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date, corpus_cutoff=self.corpus_cutoff,
            items=items, premise_proofs=build_resolution_premise_proofs(intent, items))
        return {"kind": "resolved", "resolution": resolution.model_dump(mode="json")}


__all__ = ["LifecycleCompositeResolutionBackend"]
