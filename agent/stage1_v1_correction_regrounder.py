"""Question-grounded repair for correction lineage intent topology.

The regrounder never consults a fixture, answer, receipt, company list, or
question id.  It merely restores literal roles HCX sometimes flattens before a
canonical correction-lineage selector proves issuer and event identity.
"""

from __future__ import annotations

import re
from typing import Any

from .semantic_intent_v1 import SemanticIntent


_CORRECTION_DIFF = re.compile(
    r"(?P<target>.+?)의\s*최초\s*공시와\s*"
    r"(?P<correction>[0-9]{4}\s*년\s*[0-9]{1,2}\s*월\s*[0-9]{1,2}\s*일\s*정정공시)"
    r"(?:에서|의)?\s*.*?(?:무엇이?\s*달라|무엇이?\s*바뀌|어떻게\s*달라)",
)
_RANGED_CORRECTION_DIFF = re.compile(
    r"(?P<root>[0-9]{4}\s*년\s*[0-9]{1,2}\s*월\s*[0-9]{1,2}\s*일)\s*"
    r"최초\s*공시(?:한|된)?\s*(?P<target>.+?)(?:은|는)\s*"
    r"(?P<correction>[0-9]{4}\s*년\s*[0-9]{1,2}\s*월\s*[0-9]{1,2}\s*일)\s*"
    r"(?:이후|부터)\s*정정공시.*?(?:어떤|무엇).*?(?:변경|바뀌|달라)",
)
_ROOTED_CORRECTION_HISTORY_DIFF = re.compile(
    r"(?P<root>[0-9]{4}\s*년\s*[0-9]{1,2}\s*월\s*[0-9]{1,2}\s*일)\s*"
    r"공시(?:한|된)?\s*(?P<target>.+?)(?:은|는)\s*(?:그\s*)?이후\s*"
    r"정정공시.*?(?:어떤|무엇|어떻게).*?(?:변경|바뀌|달라)",
)
_AMOUNT_CHANGE = re.compile(
    r"(?P<amount>계약\s*금액).*?"
    r"(?P<numeric>(?<![0-9])[0-9]+(?:\.[0-9]+)?\s*(?:조|억|만)?\s*원)"
    r"(?:으로|로)?\s*(?P<state>늘어(?:났|난|나)|증가(?:했|한|하)|감소(?:했|한|하)|줄어(?:들었|든|든다))",
)
_COLLECTION_GROUP = re.compile(r"계약\s*별(?:로)?\s*구분")
_COLLECTION_AMOUNT = re.compile(r"정정\s*전후\s*계약\s*금액")
_COLLECTION_DIFFERENCE = re.compile(r"차이|증감")
_QUOTED_LINEAGE_HISTORY = re.compile(
    r"['\"](?P<target>[^'\"]{2,})['\"]\s*계약\s*관련\s*"
    r"(?P<date>[0-9]{4}\s*년\s*[0-9]{1,2}\s*월\s*[0-9]{1,2}\s*일)\s*"
    r"(?:기재\s*)?정정\s*공시",
)
_EXACT_CORRECTION = re.compile(
    r"(?P<coordinate>(?:접수\s*번호\s*)?20[0-9]{12}|"
    r"[0-9]{4}\s*년\s*[0-9]{1,2}\s*월\s*[0-9]{1,2}\s*일)\s*"
    r"(?P<family>[^?。\n]{2,60}?)\s*(?:기재\s*)?정정\s*공시")
_GENERIC_EXACT_CORRECTION = re.compile(
    r"(?P<coordinate>(?:접수\s*번호\s*)?20[0-9]{12}|"
    r"[0-9]{4}\s*년\s*[0-9]{1,2}\s*월\s*[0-9]{1,2}\s*일)\s*"
    r"(?P<surface>(?:기재\s*)?정정\s*공시)",
)


class CorrectionLineageIntentRegrounder:
    """Restore two closed correction grammars from exact question spans."""

    @staticmethod
    def _exact_document(payload: dict[str, Any], question: str) -> bool:
        """Restore one exact correction filing selected by date or receipt."""

        match = _EXACT_CORRECTION.search(question)
        generic_match = (
            _GENERIC_EXACT_CORRECTION.search(question)
            if match is None else None
        )
        if match is None and generic_match is None:
            return False
        family = (
            match.group("family").strip()
            if match is not None
            else generic_match.group("surface").strip()
        )
        # ``<original date> 공시한 X는 이후 정정공시`` is a lineage-range
        # request, not an exact correction filing coordinate.
        if (not family or re.search(r"공시\s*한|최초|이후|부터", family)
                or re.search(r"보고서", family)):
            return False
        items = payload.get("answer_items", [])
        if (len(items) != 1 or payload.get("answer_groups")
                or payload.get("premises") or payload.get("unresolved_mentions")):
            return False
        companies = [
            entity for entity in payload.get("entities", [])
            if entity.get("kind_hint") == "company"
            and str(entity.get("surface", "")) in question
        ]
        if len(companies) != 1:
            return False
        company = dict(companies[0])
        company_id = str(company.get("entity_id", ""))
        if not company_id:
            return False
        coordinate = (
            match.group("coordinate")
            if match is not None
            else generic_match.group("coordinate")
        )
        receipt = re.search(r"20[0-9]{12}", coordinate)
        coordinate_surface = receipt.group(0) if receipt is not None else coordinate
        presentation = str(payload.get("presentation", "auto"))
        payload["entities"] = [company]
        payload["answer_items"] = [{
            "item_id": "item-1",
            "target": {
                "kind": "document", "surface": family,
                "entity_refs": [company_id], "qualifier_surfaces": [],
            },
            "operation": "retrieve",
            "scope": {
                "target_period_expressions": [coordinate_surface],
                "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": "narrative", "projection_mode": "whole_target",
                "field_surfaces": [], "presentation": presentation,
            },
        }]
        payload["answer_groups"] = []
        payload["premises"] = []
        payload["unresolved_mentions"] = []
        return True

    @staticmethod
    def _quoted_lineage_history(payload: dict[str, Any], question: str) -> bool:
        """Restore one quoted-contract correction-history request.

        A quoted product acronym can be a contract name rather than a
        counterparty (for example ``ABC(한글 설명) 1기``).  The issuer is the
        unique entity in the grammatical prefix; canonical company and event
        resolution still prove both coordinates later.
        """

        match = _QUOTED_LINEAGE_HISTORY.search(question)
        if (match is None
                or re.search(r"원\s*계약", question) is None
                or re.search(r"코퍼스", question) is None
                or re.search(r"특정", question) is None
                or re.search(r"정정\s*(?:공시\s*)?(?:변경\s*)?이력", question) is None):
            return False
        prefix = question[:match.start()]
        issuer_entities = [
            entity for entity in payload.get("entities", [])
            if str(entity.get("surface", "")).strip()
            and str(entity.get("surface", "")) in prefix
        ]
        if len(issuer_entities) != 1:
            return False
        issuer = dict(issuer_entities[0])
        issuer["kind_hint"] = "company"
        company_id = str(issuer.get("entity_id", ""))
        if not company_id:
            return False
        presentation = str(payload.get("presentation", "auto"))
        payload["entities"] = [issuer]
        payload["answer_items"] = [{
            "item_id": "item-1",
            "target": {
                "kind": "document", "surface": match.group("target").strip(),
                "entity_refs": [company_id], "qualifier_surfaces": [],
            },
            "operation": "retrieve",
            "scope": {
                "target_period_expressions": [match.group("date")],
                "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": "timeline", "projection_mode": "named_fields",
                "field_surfaces": ["원계약", "확인되는 정정 이력"],
                "presentation": presentation,
            },
        }]
        payload["answer_groups"] = []
        payload["premises"] = []
        payload["unresolved_mentions"] = []
        return True

    @staticmethod
    def _collection(payload: dict[str, Any], question: str) -> bool:
        """Restore one issuer-wide correction collection from literal cues."""

        items = payload.get("answer_items", [])
        entities = payload.get("entities", [])
        if (len(items) != 1 or payload.get("premises")
                or payload.get("unresolved_mentions")):
            return False
        item = items[0]
        target = str(item.get("target", {}).get("surface", ""))
        group = _COLLECTION_GROUP.search(question)
        amount = _COLLECTION_AMOUNT.search(question)
        difference = _COLLECTION_DIFFERENCE.search(question)
        company_ids = [
            str(entity.get("entity_id")) for entity in entities
            if entity.get("kind_hint") == "company"
            and str(entity.get("surface", "")) in question
        ]
        if (group is None or amount is None or difference is None
                or len(company_ids) != 1
                or re.search(r"정정\s*공시", question) is None
                or re.search(r"계약|공급|판매", target) is None
                or company_ids[0] not in item.get("target", {}).get(
                    "entity_refs", [])):
            return False
        presentation = str(payload.get("presentation", "auto"))
        item["item_id"] = "item-1"
        item["operation"] = "retrieve"
        item["selection"] = None
        item["output"] = {
            "shape": "record", "projection_mode": "named_fields",
            # Every surface remains a literal question span.  The compiler
            # maps these three roles to amount/before/difference slots.
            "field_surfaces": [
                group.group(0), amount.group(0), difference.group(0)],
            "presentation": presentation,
        }
        payload["answer_items"] = [item]
        payload["answer_groups"] = []
        payload["premises"] = []
        payload["unresolved_mentions"] = []
        return True

    @staticmethod
    def _diff(payload: dict[str, Any], question: str) -> bool:
        match = _CORRECTION_DIFF.search(question)
        if match is None:
            return False
        target, correction = match.group("target").strip(), match.group("correction")
        # Reject a broad sentence subject.  The contract/event phrase must be
        # the same literal surface an upstream item already retained.
        source_surfaces = [
            str(item.get("target", {}).get("surface", ""))
            for item in payload.get("answer_items", [])
        ]
        if (not target or not correction or not any(
                target == surface or target in surface or surface in target
                for surface in source_surfaces if surface)):
            return False
        presentation = str(payload.get("presentation", "auto"))
        payload["entities"] = []
        payload["answer_items"] = [{
            "item_id": "item-1",
            "target": {
                "kind": "document", "surface": target,
                "entity_refs": [], "qualifier_surfaces": [],
            },
            "operation": "retrieve",
            "scope": {
                "target_period_expressions": [correction],
                "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": "narrative", "projection_mode": "whole_target",
                "field_surfaces": [], "presentation": presentation,
            },
        }]
        payload["answer_groups"] = []
        payload["premises"] = []
        payload["unresolved_mentions"] = []
        return True

    @staticmethod
    def _ranged_diff(payload: dict[str, Any], question: str) -> bool:
        match = (_RANGED_CORRECTION_DIFF.search(question)
                 or _ROOTED_CORRECTION_HISTORY_DIFF.search(question))
        if match is None:
            return False
        target = match.group("target").strip()
        correction_date = match.groupdict().get("correction")
        # The open-ended form spells out ``이후`` only after the event name,
        # not next to its root date.  Keep the literal root span here; the
        # lineage date-role parser supplies the open correction range.
        correction = (correction_date + " 이후"
                      if correction_date is not None else match.group("root"))
        source_items = payload.get("answer_items", [])
        source_surfaces = [
            str(item.get("target", {}).get("surface", ""))
            for item in source_items
        ]
        source_surfaces.extend(
            str(entity.get("surface", ""))
            for entity in payload.get("entities", []))
        if not target or not any(
                target == surface or target in surface or surface in target
                for surface in source_surfaces if surface):
            return False
        entities = payload.get("entities", [])
        # HCX can tag an issuer abbreviation as ``event`` when the following
        # dated contract phrase is another event entity.  Syntax only proposes
        # the unique subject before the first date; the canonical company
        # resolver still has to prove it in the lineage selector.
        prefix = question[:match.start()]
        issuer_entities = [
            entity for entity in entities
            if str(entity.get("surface", "")).strip()
            and str(entity.get("surface", "")) in prefix
        ]
        if len(issuer_entities) != 1:
            return False
        issuer = dict(issuer_entities[0])
        issuer["kind_hint"] = "company"
        company_id = str(issuer.get("entity_id"))
        payload["entities"] = [issuer]
        presentation = str(payload.get("presentation", "auto"))
        payload["answer_items"] = [{
            "item_id": "item-1",
            "target": {
                "kind": "document", "surface": target,
                "entity_refs": [company_id], "qualifier_surfaces": [],
            },
            "operation": "retrieve",
            "scope": {
                "target_period_expressions": [correction],
                "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": "narrative", "projection_mode": "whole_target",
                "field_surfaces": [], "presentation": presentation,
            },
        }]
        payload["answer_groups"] = []
        payload["premises"] = []
        payload["unresolved_mentions"] = []
        return True

    @staticmethod
    def _amount_change(payload: dict[str, Any], question: str) -> bool:
        match = _AMOUNT_CHANGE.search(question)
        items = payload.get("answer_items")
        if match is None or not isinstance(items, list) or len(items) not in {1, 2}:
            return False
        if payload.get("premises") or payload.get("unresolved_mentions"):
            return False
        amount, numeric, state = (match.group(name) for name in ("amount", "numeric", "state"))
        presentation = str(payload.get("presentation", "auto"))
        first = items[0]
        if len(items) == 1:
            # A scalar model response can flatten the amount and its causal
            # follow-up into one item.  The question itself supplies both
            # roles, so restore the same two-item topology as the split form
            # without adding a corpus-specific coordinate.
            surfaces = first.get("output", {}).get("field_surfaces", [])
            if not (
                    first.get("operation") == "retrieve"
                    and (amount.replace(" ", "") in str(
                        first.get("target", {}).get("surface", "")).replace(" ", "")
                         or any(amount.replace(" ", "") in str(value).replace(" ", "")
                                or "금액" in str(value) for value in surfaces))
                    and any("이유" in str(value) or state in str(value)
                            for value in surfaces)):
                return False
            second = {
                "item_id": "item-2", "operation": "retrieve",
                "target": {}, "scope": {}, "selection": None,
                "output": {},
            }
        else:
            second = items[1]
        refs = list(first.get("target", {}).get("entity_refs", []))
        if not refs:
            return False
        first.update({
            "item_id": "item-1", "operation": "retrieve",
            "target": {"kind": "metric", "surface": amount,
                       "entity_refs": refs, "qualifier_surfaces": []},
            "scope": {"target_period_expressions": [], "as_of_expression": None,
                      "document_group_expression": None,
                      "scope_qualifier_expressions": []},
            "selection": None,
            "output": {"shape": "scalar", "projection_mode": "named_fields",
                       "field_surfaces": [amount], "presentation": presentation},
        })
        second.update({
            "item_id": "item-2", "operation": "retrieve",
            "target": {"kind": "event", "surface": "계약",
                       "entity_refs": refs, "qualifier_surfaces": []},
            "scope": {"target_period_expressions": [], "as_of_expression": None,
                      "document_group_expression": None,
                      "scope_qualifier_expressions": []},
            "selection": None,
            "output": {"shape": "narrative", "projection_mode": "named_fields",
                       "field_surfaces": [f"{state} 이유"], "presentation": presentation},
        })
        payload["answer_items"] = [first, second]
        payload["answer_groups"] = [{
            "group_id": "group-1", "item_ids": ["item-1", "item-2"],
        }]
        payload["premises"] = [
            {"premise_id": "premise-1", "kind": "numeric", "raw_text": numeric,
             "applies_to_item_ids": ["item-1"]},
            {"premise_id": "premise-2", "kind": "state", "raw_text": state,
             "applies_to_item_ids": ["item-1", "item-2"]},
        ]
        return True

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if not isinstance(question, str) or not question.strip():
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        changed = (self._exact_document(payload, question)
                   or self._quoted_lineage_history(payload, question)
                   or self._collection(payload, question)
                   or self._ranged_diff(payload, question)
                   or self._diff(payload, question)
                   or self._amount_change(payload, question))
        return SemanticIntent.model_validate(payload, strict=True) if changed else intent


__all__ = ["CorrectionLineageIntentRegrounder"]
