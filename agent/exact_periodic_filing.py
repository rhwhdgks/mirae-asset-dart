"""Fail-closed resolution of an explicitly dated periodic filing.

This helper reads metadata only and intentionally does not follow a correction
lineage.  A user who names the filing day and form is asking for that as-filed
document; latest-version semantics remain the periodic preflight's job.
"""

from __future__ import annotations

import re
from typing import Any

from agent.date_surface import question_date_surfaces
from agent.planning import resolve_metric_concept
from agent.semantic_intent_v1 import SemanticIntent


_FORM_TOKENS = {
    "annual": re.compile(r"(?:사업|연간)\s*보고서"),
    "half": re.compile(r"반기\s*보고서"),
    "quarter": re.compile(r"분기\s*보고서"),
}
_FULL_DATE = re.compile(
    r"(?:19|20)[0-9]{2}(?:\s*년\s*[0-9]{1,2}\s*월\s*[0-9]{1,2}\s*일|"
    r"[-./][0-9]{1,2}[-./][0-9]{1,2})")
_AS_OF_ONLY = re.compile(r"(?:기준|현재|시점|말|이전|까지|최신)")


def resolve_exact_periodic_filing(
        reader: Any, *, corp_code: str, question: str, as_of: str,
        ) -> Any | None:
    """Return one metadata row matching an explicit filing coordinate.

    The coordinate requires one full calendar day and one report form.  An
    explicit correction/original polarity is also enforced.  Zero or multiple
    matches return ``None`` so callers can fall back or ask for clarification.
    """

    if not isinstance(question, str) or not question.strip():
        return None
    parsed_days = {
        f"{year:04d}{month:02d}{day:02d}"
        for year, month, day in question_date_surfaces(question)
        if month is not None and day is not None
    }
    # A full date elsewhere in the question can be an ordinary as-of boundary.
    # Require the date to be locally attached to the named filing form and
    # reject typical latest/as-of linkers between them.
    coordinates: list[tuple[str, str]] = []
    for date_match in _FULL_DATE.finditer(question):
        tail = question[date_match.end():date_match.end() + 32]
        for form, pattern in _FORM_TOKENS.items():
            form_match = pattern.search(tail)
            if form_match is None:
                continue
            linker = tail[:form_match.start()]
            if _AS_OF_ONLY.search(linker):
                continue
            local_days = {
                f"{year:04d}{month:02d}{day:02d}"
                for year, month, day in question_date_surfaces(date_match.group())
                if month is not None and day is not None
            }
            coordinates.extend((day, form) for day in local_days)
    coordinates = list(dict.fromkeys(coordinates))
    if len(coordinates) != 1 or len(parsed_days) != 1:
        return None
    filing_day, form = coordinates[0]
    correction: bool | None = None
    if re.search(r"(?:기재\s*)?정정\s*(?:된\s*)?(?:사업|연간|반기|분기)\s*보고서", question):
        correction = True
    elif re.search(r"(?:최초\s*제출|원본)\s*(?:사업|연간|반기|분기)\s*보고서", question):
        correction = False

    rows = []
    for row in reader.documents(
            corp_code=corp_code, as_of=as_of, doc_group="periodic"):
        row_form = getattr(row, "form", None) or getattr(row, "doc_subtype", None)
        if (str(getattr(row, "corp_code", "")) != corp_code
                or str(getattr(row, "rcept_dt", "")) != filing_day
                or row_form != form):
            continue
        if correction is not None and bool(getattr(row, "is_correction", False)) != correction:
            continue
        rows.append(row)
    unique = {str(getattr(row, "rcept_no", "")): row for row in rows}
    return next(iter(unique.values())) if len(unique) == 1 else None


class ExactPeriodicFinancialRegrounder:
    """Bind a scalar metric to one explicitly named periodic filing day.

    Only the intent topology is repaired here.  The helper above first proves
    the issuer/date/form receipt from metadata, while the financial backend
    still resolves the account, period, scope, and fact evidence.
    """

    _YEAR = re.compile(r"(?<![0-9])20[0-9]{2}\s*년(?:도)?")

    def __init__(self, reader: Any, *, as_of: str) -> None:
        self.reader = reader
        self.as_of = as_of

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str) or not question.strip()
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        if (item.target.kind != "metric" or item.operation != "retrieve"
                or item.selection is not None
                or item.output.shape != "scalar"
                or item.output.projection_mode != "named_fields"
                or len(item.output.field_surfaces) != 1):
            return intent
        by_id = {entity.entity_id: entity for entity in intent.entities}
        companies = []
        for ref in item.target.entity_refs:
            entity = by_id.get(ref)
            if entity is None or entity.kind_hint != "company":
                continue
            rows = list(self.reader.resolve_company(entity.surface) or ())
            if len(rows) == 1:
                companies.append((entity, rows[0]))
        corp_codes = {getattr(row, "corp_code", None) for _entity, row in companies}
        if len(companies) != 1 or len(corp_codes) != 1:
            return intent
        entity, company = companies[0]
        document = resolve_exact_periodic_filing(
            self.reader, corp_code=company.corp_code,
            question=question, as_of=self.as_of)
        if document is None:
            return intent

        date_matches = list(_FULL_DATE.finditer(question))
        if len(date_matches) != 1:
            return intent
        date_span = date_matches[0].span()
        fiscal_years = [
            match.group(0).strip() for match in self._YEAR.finditer(question)
            if not (date_span[0] <= match.start() < date_span[1])
        ]
        fiscal_years = list(dict.fromkeys(fiscal_years))
        if len(fiscal_years) != 1:
            return intent
        concept_surface = item.target.surface
        if resolve_metric_concept(concept_surface) is None:
            # A provider can leave the metric inside its sole field surface.
            field = item.output.field_surfaces[0]
            candidates = [
                token for token in re.findall(r"[A-Za-z0-9가-힣&+]+", field)
                if resolve_metric_concept(token) is not None
            ]
            if len(candidates) != 1:
                return intent
            concept_surface = candidates[0]

        scopes = []
        if "연결" in question:
            scopes.append("연결")
        if "별도" in question or "개별" in question:
            scopes.append("별도")
        if len(scopes) > 1:
            return intent

        payload = intent.model_dump(mode="python", warnings=False)
        # SemanticIntent local IDs are positional. Removing an unrelated
        # document entity can leave the issuer as ``entity-2``; remap the one
        # retained issuer and all refs instead of emitting a sparse ID list.
        retained = next(
            row for row in payload["entities"]
            if row.get("entity_id") == entity.entity_id)
        payload["entities"] = [{**retained, "entity_id": "entity-1"}]
        target = payload["answer_items"][0]["target"]
        target.update({
            "kind": "metric", "surface": concept_surface,
            "entity_refs": ["entity-1"], "qualifier_surfaces": [],
        })
        row = payload["answer_items"][0]
        row["item_id"] = "item-1"
        row["selection"] = None
        row["scope"] = {
            "target_period_expressions": fiscal_years,
            "as_of_expression": date_matches[0].group(0),
            "document_group_expression": None,
            "scope_qualifier_expressions": scopes,
        }
        row["output"] = {
            "shape": "scalar", "projection_mode": "named_fields",
            "field_surfaces": [concept_surface],
            "presentation": str(payload.get("presentation", "auto")),
        }
        payload["answer_groups"] = []
        payload["premises"] = []
        payload["unresolved_mentions"] = []
        return SemanticIntent.model_validate(payload, strict=True)


#: One literal ``N년 <form>보고서( 기준)?`` mention.  ``[1-4]분기`` is matched
#: so an unsupported quarter (2·4 — no such periodic filing form exists) is
#: still counted for the "exactly one mention" fail-closed gate below; only
#: the four forms in ``_REPORT_FORM_PERIOD_MAP`` ever produce a rewrite.
_REPORT_FORM_MENTION = re.compile(
    r"(?<![0-9])(20[0-9]{2})\s*년\s*(사업|반기|상반기|[1-4]분기)\s*보고서"
    r"(?:\s*기준)?")

#: Replacement suffix for ``_financial_period`` (``agent/planning.py``), which
#: already knows 「분기」「상반기」「반기」 but not the *document* form word
#: 「보고서」.  사업보고서 stays the bare year — annual is already the
#: resolver's default so nothing needs to change there.
_REPORT_FORM_PERIOD_MAP = {
    "사업": "", "반기": "반기", "상반기": "상반기",
    "1분기": "1분기", "3분기": "3분기",
}


class ReportFormFinancialPeriodRegrounder:
    """Rewrite a bare ``N년 X보고서`` mention into a token the period
    resolver (``agent.planning._financial_period``) already accepts.

    HCX's periodic *document* selector vocabulary (사업보고서/반기보고서/…,
    see ``agent.planning._document_report_fragment``) is unrelated to the
    financial *period* grammar, which only knows the 「분기」「상반기」「반기」
    suffixes.  When a question names only a report form and a year (「2025년
    반기보고서 기준」) with no calendar day, ``ExactPeriodicFinancialRegrounder``
    cannot bind a filing either (it requires one full day), so the period
    resolver never sees the half-year boundary and the item's
    ``target_period_expressions`` collapses to the bare annual year —
    silently reading a full-year fact instead of the half the user asked for.

    This regrounder only rewrites the literal report-form wording into the
    equivalent period wording; it proves nothing about the filing itself.  It
    is fail-closed: more than one report-form mention in the question, more
    than one company, an unsupported quarter (2·4), or a period expression
    that is not already a bare year for that same year leaves the intent
    untouched.
    """

    _YEAR = re.compile(r"(?<![0-9])(20[0-9]{2})\s*년(?:도)?")

    def __init__(self, reader: Any) -> None:
        if not callable(getattr(reader, "resolve_company", None)):
            raise TypeError("financial period regrounder에는 company resolver가 필요합니다")
        self.reader = reader

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if not isinstance(question, str) or not question.strip():
            return intent
        mentions = list(_REPORT_FORM_MENTION.finditer(question))
        if len(mentions) != 1:
            return intent
        year, form = mentions[0].group(1), mentions[0].group(2)
        suffix = _REPORT_FORM_PERIOD_MAP.get(form)
        if suffix is None:
            return intent
        if (len(intent.answer_items) != 1 or intent.answer_groups
                or intent.premises or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        if (item.target.kind != "metric" or item.operation != "retrieve"
                or item.selection is not None):
            return intent
        by_id = {entity.entity_id: entity for entity in intent.entities}
        companies = []
        for ref in item.target.entity_refs:
            entity = by_id.get(ref)
            if entity is None or entity.kind_hint != "company":
                continue
            rows = list(self.reader.resolve_company(entity.surface) or ())
            if len(rows) == 1:
                companies.append(rows[0])
        corp_codes = {getattr(row, "corp_code", None) for row in companies}
        if len(companies) != 1 or len(corp_codes) != 1:
            return intent
        expressions = item.scope.target_period_expressions
        if len(expressions) != 1:
            return intent
        normalized = expressions[0].strip()
        year_match = self._YEAR.fullmatch(normalized)
        if year_match is None or year_match.group(1) != year:
            return intent

        new_expression = f"{year}년" + (f" {suffix}" if suffix else "")
        if new_expression == normalized:
            return intent
        # ``target_period_expressions`` must be a literal question substring
        # (``agent.stage1_v1_resolver.validate_semantic_intent_grounding``).
        # A ``누적`` written elsewhere in the question — commonly right next
        # to the metric name, as in「연결 영업이익(누적)」— is not adjacent to
        # the report-form mention and cannot be glued on without fabricating
        # text the question never actually contains contiguously (P9-017
        # regression: this previously raised ``resolver_authority_failed``).
        # 반기/상반기/1분기 do not need it anyway — ``_financial_period``
        # already treats them as cumulative unconditionally; only an
        # explicit-cumulative 3분기/9개월 reading is left unrecovered here.
        if new_expression not in question:
            return intent

        payload = intent.model_dump(mode="python", warnings=False)
        payload["answer_items"][0]["scope"]["target_period_expressions"] = [
            new_expression]
        return SemanticIntent.model_validate(payload, strict=True)
