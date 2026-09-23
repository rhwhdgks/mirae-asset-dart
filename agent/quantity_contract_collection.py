"""A closed quantity-labelled contract pair, with signing and observation dates."""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import re

from agent.semantic_intent_v1 import SemanticIntent
from agent.stage1_v1_event_collection import EventCollectionIdentity, EventCollectionSelection

_DATE = re.compile(r"20[0-9]{2}년\s*[0-9]{1,2}월\s*[0-9]{1,2}일")
_QUANTITY = re.compile(r"[0-9]+(?:\.[0-9]+)?\s*GWh", re.I)


def quantity_contract_axes(question):
    dates = list(_DATE.finditer(question))
    quantities = list(dict.fromkeys(m.group(0) for m in _QUANTITY.finditer(question)))
    if (len(dates) != 2 or len(quantities) != 2
            or not all(s in question for s in ("각각", "접수번호", "금액 공개 여부", "최종 상태"))
            or not question[dates[1].end():].startswith("까지")):
        return None
    counterparty = re.match(r"\s*(.+?)와\s*체결", question[dates[0].end():])
    if counterparty is None:
        return None
    stamps = []
    for match in dates:
        y, m, d = map(int, re.findall(r"[0-9]+", match.group(0)))
        try:
            stamps.append(date(y, m, d).strftime("%Y%m%d"))
        except ValueError:
            return None
    return dates, quantities, counterparty.group(1).strip(), stamps


class QuantityContractCollectionRegrounder:
    def __init__(self, company_surface_regrounder):
        self.company_surface_regrounder = company_surface_regrounder

    def __call__(self, question, intent):
        axes = quantity_contract_axes(question)
        if axes is None or intent.premises or intent.answer_groups or intent.unresolved_mentions:
            return intent
        from agent.stage1_v1_closed_grounding_fallback import _company_surfaces
        companies = _company_surfaces(question, self.company_surface_regrounder)
        if len(companies) != 1:
            return intent
        dates, _quantities, counterparty, _stamps = axes
        return SemanticIntent.model_validate({
            "schema_version": "stage1-semantic-intent/1.1",
            "entities": [
                {"entity_id": "entity-1", "kind_hint": "company", "surface": companies[0]},
                {"entity_id": "entity-2", "kind_hint": "counterparty", "surface": counterparty}],
            "answer_items": [{
                "item_id": "item-1", "operation": "retrieve",
                "target": {"kind": "event", "surface": "계약", "entity_refs": ["entity-1", "entity-2"],
                           "qualifier_surfaces": [dates[0].group(0), *_quantities]},
                "scope": {"target_period_expressions": [], "as_of_expression": dates[1].group(0),
                          "document_group_expression": None, "scope_qualifier_expressions": []},
                "selection": None,
                "output": {"shape": "record_list", "projection_mode": "named_fields",
                           "field_surfaces": ["접수번호", "금액 공개 여부", "최종 상태"], "presentation": "auto"}}],
            "answer_groups": [], "premises": [], "unresolved_mentions": [], "presentation": "auto",
        }, strict=True)


def select_quantity_contract_collection(canonical, intent, question, corpus_cutoff):
    axes = quantity_contract_axes(question)
    if axes is None or len(intent.answer_items) != 1:
        return None
    dates, quantities, counterparty, stamps = axes
    company_entities = [e for e in intent.entities if e.kind_hint == "company"]
    if len(company_entities) != 1:
        return None
    companies = canonical.resolve_company(company_entities[0].surface)
    if len(companies) != 1:
        return None
    company = companies[0]
    as_of = min(stamps[1], corpus_cutoff)
    groups = defaultdict(list)
    for row in canonical.fields(as_of=as_of, corp_code=company.corp_code, form="단일판매공급계약체결"):
        groups[row.rcept_no].append(row)
    selected = []
    seen_quantities = []
    norm = lambda value: re.sub(r"\s+", "", value).casefold()
    for receipt, rows in groups.items():
        party_rows = [r for r in rows if "계약상대" in r.path and norm(counterparty) == norm(str(r.value))]
        signing_rows = [r for r in rows if "계약" in r.path and "일자" in r.path
                        and re.sub(r"[^0-9]", "", str(r.value)) == stamps[0]]
        quantity_rows = [(r, norm(m.group(0))) for r in rows for m in _QUANTITY.finditer(str(r.value))
                         if norm(m.group(0)) in {norm(q) for q in quantities}]
        if not party_rows or not signing_rows or len({q for _r, q in quantity_rows}) != 1:
            continue
        timeline = canonical.event_timeline(as_of=as_of, rcept_no=receipt, verify_evidence=True)
        if timeline is None or timeline.root_rcept_no != receipt:
            continue
        proofs = []
        for row in [party_rows[0], signing_rows[0], quantity_rows[0][0]]:
            looked = canonical.lookup_field(as_of=as_of, rcept_no=receipt, path=row.path, locator=row.locator)
            if looked.status != "ok" or looked.selected.evidence_status != "verified":
                return None
            proofs.append(f"canonical:field:{row.evidence_id}")
        proofs.append(f"canonical:event:{timeline.event_key}:{receipt}")
        selected.append(EventCollectionIdentity(
            timeline.event_key, receipt, (receipt,), (receipt[:8],),
            ("단일판매공급계약체결",), (counterparty,), ("계약",), tuple(proofs)))
        seen_quantities.append(quantity_rows[0][1])
    if sorted(seen_quantities) != sorted(norm(q) for q in quantities):
        return None
    receipt_dates = {e.root_receipt[:8] for e in selected}
    if len(receipt_dates) != 1:
        return None
    # Public list selector must reproduce the complete proved member set.
    filing_date = next(iter(receipt_dates))
    from agent.event_preflight import _contains_any
    public_keys = set()
    for receipt, rows in groups.items():
        if receipt[:8] != filing_date or not any(
                "계약상대" in r.path and _contains_any(counterparty, [str(r.value)]) for r in rows):
            continue
        timeline = canonical.event_timeline(as_of=as_of, rcept_no=receipt, verify_evidence=True)
        if timeline is None:
            return None
        public_keys.add(timeline.event_key)
    if public_keys != {e.event_key for e in selected}:
        return None
    return EventCollectionSelection(
        company.corp_code, company.corp_name, as_of, filing_date, filing_date,
        False, False, "단일판매공급계약체결", counterparty, ("계약",), "event", False,
        ("접수번호", "금액공개여부", "최종상태"), tuple(sorted(selected, key=lambda e: e.event_key)),
        tuple(p for e in selected for p in e.proof_refs))
