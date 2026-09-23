"""Question-grounded topology repair for one exact-dated event filing.

The adapter does not choose a company, receipt, or value.  It only moves the
literal filing date onto the event selector axis and expands the closed Korean
``전후`` compound into its two requested roles.  The selected-event backend
must still prove one canonical event before a plan can be emitted.
"""

from __future__ import annotations

import re

from .semantic_intent_v1 import SemanticIntent


_FULL_DATE = re.compile(
    r"(?<![0-9])(?P<date>[0-9]{4}\s*년\s*[0-9]{1,2}\s*월\s*"
    r"[0-9]{1,2}\s*일)(?![0-9])")
_EVENT_FILING = re.compile(
    r"[0-9]{4}\s*년\s*[0-9]{1,2}\s*월\s*[0-9]{1,2}\s*일\s*"
    r"(?:"
    # ``의`` is a delimiter only when it is followed by whitespace.  Without
    # that boundary, a form such as ``소송 등의 제기에서`` is truncated to
    # ``소송 등`` at the first syllable ``의`` inside the form name.
    r"공시한\s*(?P<family_after>[^?。\n]{2,80}?)\s*"
    r"(?:에서|(?<!등)의(?=\s))|"
    r"(?P<family_before>[^?。\n]{2,80}?)\s*공시(?:에서|의)"
    r")")
_NAMED_FIELD_PAIR = re.compile(
    r"(?:에서|의)\s*"
    r"(?P<first>[^,?？。\n]{1,80}?)\s*(?:와|과|및|·)\s*"
    r"(?P<second>[^,?？。\n]{1,80}?)(?:을|를)\s*"
    r"(?:알려|말해|정리)")
_BEFORE_AFTER_AMOUNT = re.compile(r"해지\s*전후\s*계약\s*금액")
_INVESTMENT_JUDGMENT_FORM = re.compile(
    r"투자\s*판단\s*(?:관련)?\s*주요\s*경영\s*사항")
_BUSINESS_EVENT_FAMILY = re.compile(
    r"계약|취득|양수|양도|처분|합병|분할|증자|감자|해지|발행|사채|증권|"
    r"교환|이전|영업\s*정지|소송|상장")
_GENERIC_SINGLE_FIELD = re.compile(r"^(?:내용|정보|값|얼마)$")


def exact_dated_event_locator(question: str) -> tuple[str, str] | None:
    """Return one literal filing day and event family from either word order.

    Organizer-style questions commonly say both ``날짜 X 공시에서`` and
    ``날짜 공시한 X에서``.  These are the same public document coordinate.
    This helper extracts only those literal axes; it never chooses a receipt,
    company, or answer value.
    """

    if not isinstance(question, str):
        return None
    dates = list(_FULL_DATE.finditer(question))
    filing = _EVENT_FILING.search(question)
    if len(dates) != 1 or filing is None:
        return None
    family = (filing.group("family_after")
              or filing.group("family_before") or "").strip()
    # Keep this recovery to business-event filings.  The exact-date canonical
    # lookup below still has to prove a single matching disclosure.
    if (not family or re.search(r"보고서", family)
            or not (_BUSINESS_EVENT_FAMILY.search(family)
                    or _INVESTMENT_JUDGMENT_FORM.search(family))):
        return None
    return dates[0].group("date"), family


def exact_dated_event_named_fields(
        question: str,
        ) -> tuple[str, str, tuple[str, str]] | None:
    """Return one exact event coordinate with two explicit answer fields.

    This deliberately accepts only the closed ``A와 B를 알려줘`` family.
    Wider prose, lists with three or more fields, or implicit projections stay
    with the provider rather than being guessed by a fallback parser.
    """

    locator = exact_dated_event_locator(question)
    fields = _NAMED_FIELD_PAIR.search(question) if locator is not None else None
    if locator is None or fields is None:
        return None
    first = fields.group("first").strip()
    second = fields.group("second").strip()
    if (not first or not second or first == second
            or re.search(r"\s(?:와|과|및)\s|[·,]", first)
            or re.search(r"\s(?:와|과|및)\s|[·,]", second)
            or re.search(r"[가-힣](?:와|과)\s", second)):
        return None
    return locator[0], locator[1], (first, second)


class ExactDatedEventFieldsRegrounder:
    """Restore one exact event-record request without selecting its receipt."""

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str) or not question.strip()
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        locator = exact_dated_event_locator(question)
        if locator is None:
            return intent
        filing_date, family = locator
        item = intent.answer_items[0]
        # A lone ``해지 전후 계약금액`` field is already a closed compound of
        # two typed roles (해지 전/해지 후), so it is exempt from the >=2
        # literal-span count below — the expansion below turns it into two
        # fields on its own without guessing anything from prose.
        surfaces = item.output.field_surfaces
        solo_before_after = (
            len(surfaces) == 1 and bool(_BEFORE_AFTER_AMOUNT.fullmatch(surfaces[0])))
        # ``내용`` alone is ordinarily too vague to turn into a field read.
        # The literal 투자판단 관련 주요경영사항 form is the closed exception:
        # its ``2. 주요내용`` cell is the filing's public body (including its
        # disclosed table/list), so this asks for that selected document body,
        # not a periodic-report narrative.
        solo_investment_judgment_content = (
            len(surfaces) == 1
            and re.sub(r"\s+", "", surfaces[0]) == "내용"
            and bool(_INVESTMENT_JUDGMENT_FORM.search(family)))
        # An exact issuer + filing day + form plus one literal, specifically
        # named field is as closed as the existing two-field record request.
        # Keep generic requests (``내용``/``정보``/``값``/``얼마``) out: they do not
        # identify one public form cell.  The canonical selected-document
        # backend still has to prove one receipt before this can execute.
        solo_specific_field = (
            len(surfaces) == 1
            and surfaces[0] in question
            and _GENERIC_SINGLE_FIELD.fullmatch(surfaces[0].strip()) is None)
        if (item.operation != "retrieve"
                or item.output.projection_mode != "named_fields"
                or item.output.shape not in {"scalar", "record", "narrative"}
                or (len(surfaces) < 2 and not solo_before_after
                    and not solo_investment_judgment_content
                    and not solo_specific_field)
                or not all(surface in question for surface in surfaces)):
            return intent

        entity_by_id = {entity.entity_id: entity for entity in intent.entities}
        # Prefer the issuer that the provider already attached to the target.
        # This distinguishes an issuer from a second company-like legal person
        # named as a row/counterparty (for example an investor in an issuance
        # table).  Fall back to the legacy single-company case only when the
        # target did not carry that role explicitly.
        companies = [
            entity_by_id[ref] for ref in item.target.entity_refs
            if ref in entity_by_id
            and entity_by_id[ref].kind_hint == "company"
            and entity_by_id[ref].surface in question
        ]
        if not companies:
            companies = [
                entity for entity in intent.entities
                if entity.kind_hint == "company" and entity.surface in question
            ]
        if len(companies) != 1:
            return intent
        company = companies[0]
        fields: list[str] = []
        for surface in item.output.field_surfaces:
            if _BEFORE_AFTER_AMOUNT.fullmatch(surface):
                fields.extend(["해지 전 계약금액", "해지 후 계약금액"])
            else:
                fields.append(surface)
        if fields == list(item.output.field_surfaces):
            # The date-axis repair is useful for any exact event record, but
            # only when every output demand is already a literal question span.
            fields = list(item.output.field_surfaces)

        payload = intent.model_dump(mode="python", warnings=False)
        # Preserve one explicitly named non-issuer legal person as a
        # counterparty/row scope.  Dropping it would make a multi-row filing
        # ambiguous; leaving it as a second issuer would invert the issuer
        # heuristic in the selected-event compiler.  This role change is made
        # only for a literal, target-unreferenced entity in the closed exact
        # filing request above.
        retained_entities = []
        for entity in payload["entities"]:
            if entity.get("entity_id") == company.entity_id:
                retained_entities.append(entity)
                continue
            if (entity.get("kind_hint") in {"company", "counterparty"}
                    and str(entity.get("surface", "")) in question):
                entity["kind_hint"] = "counterparty"
                retained_entities.append(entity)
        payload["entities"] = retained_entities
        target = payload["answer_items"][0]["target"]
        target.update({
            "kind": "event",
            "surface": family,
            "entity_refs": [company.entity_id],
            "qualifier_surfaces": [filing_date],
        })
        row = payload["answer_items"][0]
        row["item_id"] = "item-1"
        row["selection"] = None
        row["scope"] = {
            "target_period_expressions": [],
            "as_of_expression": None,
            "document_group_expression": None,
            "scope_qualifier_expressions": [],
        }
        row["output"] = {
            "shape": "record",
            "projection_mode": "named_fields",
            "field_surfaces": fields,
            "presentation": str(payload.get("presentation", "auto")),
        }
        payload["answer_groups"] = []
        payload["premises"] = []
        payload["unresolved_mentions"] = []
        return SemanticIntent.model_validate(payload, strict=True)


__all__ = [
    "ExactDatedEventFieldsRegrounder",
    "exact_dated_event_locator",
    "exact_dated_event_named_fields",
]
