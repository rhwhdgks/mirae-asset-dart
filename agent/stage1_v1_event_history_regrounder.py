"""Question-grounded event history and correction intent recovery.

The semantic model occasionally flattens a fully specified lifecycle request
into an unsupported document/topic shape.  This adapter restores only roles
that are literal in the question: one canonically unique issuer, an optional
quoted contract name, an exact filing date, correction/history demand, and
status observation points.  Canonical backends still own event identity,
lineage, values, and ambiguity.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from agent.semantic_intent_v1 import SemanticIntent
from agent.stage1_v1_lifecycle_composite import (
    is_complete_named_document_timeline_request,
)

_FULL_DATE = re.compile(
    r"(?P<value>(?P<year>20[0-9]{2})\s*년\s*"
    r"(?P<month>1[0-2]|0?[1-9])\s*월\s*"
    r"(?P<day>3[01]|[12][0-9]|0?[1-9])\s*일)"
)
_PARTIAL_DATE = re.compile(
    r"(?P<month>1[0-2]|0?[1-9])\s*월\s*"
    r"(?P<day>3[01]|[12][0-9]|0?[1-9])\s*일"
)
_QUOTED_CONTRACT = re.compile(r"['\"](?P<value>[^'\"]{2,})['\"]")
_CORRECTION = re.compile(r"정정")
_HISTORY = re.compile(
    r"전체\s*정정\s*내역|정정\s*(?:변경\s*)?(?:이력|흐름)|"
    r"변경\s*이력|정정\s*전후|최초\s*내용과\s*최신\s*정정|"
    r"정정공시마다|처음.*마지막\s*정정|최초\s*공시부터\s*최신\s*정정|"
    r"전체\s*이력|원\s*계약\s*계보|원계약\s*계보"
)
_STATUS = re.compile(
    r"최신\s*(?:유효본|상태)|현재\s*(?:유효|상태)|각각\s*어떤\s*상태|"
    r"어떤\s*상태|(?:기준으로|기준)\s*(?:각각\s*)?(?:어떤\s*)?상태|"
    r"해지[·ㆍ/]?종료\s*공시가\s*확인|지금\s*해지|"
    r"해지됐|하나로\s*확정"
)
_ROOT_AFTER_DATE = re.compile(
    r"\s*(?:에\s*)?(?:공시|제출)(?:한|된)?\s*"
    r"(?:그\s*)?(?:계약|신규시설투자)"
)
_CORRECTION_AFTER_DATE = re.compile(
    r"\s*(?:에\s*)?(?:제출|공시)?(?:한|된)?\s*"
    r"(?:계약\s*)?(?:기재\s*)?정정\s*공시"
)
_CORPUS_POINT = re.compile(r"코퍼스\s*기준일|2026\s*년\s*6\s*월\s*19\s*일")
_CURRENT_POINT = re.compile(r"최신|현재|지금")
_YEAR_END = re.compile(r"(?<![0-9])20[0-9]{2}\s*년\s*말")
_ROOT_MISSING = re.compile(
    r"원\s*공시|원공시|원\s*계약|원계약|최초\s*원본|"
    r"2020\s*년\s*원공시")


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


class EventHistoryIntentRegrounder:
    """Recover closed event/correction lifecycle topologies from literals."""

    def __init__(self, company_preflight: Any) -> None:
        finder = getattr(company_preflight, "unique_question_company_surface", None)
        if not callable(finder):
            raise TypeError("event history regrounder에는 company preflight가 필요합니다")
        self.company_preflight = company_preflight

    @staticmethod
    def _dates(question: str) -> list[re.Match[str]]:
        return list(_FULL_DATE.finditer(question))

    @classmethod
    def _root_date_surface(cls, question: str) -> str | None:
        values = [
            match.group("value")
            for match in cls._dates(question)
            if _ROOT_AFTER_DATE.match(question[match.end():match.end() + 40])
            and not _CORRECTION_AFTER_DATE.match(
                question[match.end():match.end() + 40])
        ]
        return values[0] if len(_ordered_unique(values)) == 1 else None

    @classmethod
    def _correction_date_surface(cls, question: str) -> str | None:
        values = [
            match.group("value")
            for match in cls._dates(question)
            if _CORRECTION_AFTER_DATE.match(
                question[match.end():match.end() + 40])
        ]
        unique = _ordered_unique(values)
        return unique[0] if len(unique) == 1 else None

    @staticmethod
    def _target_surface(question: str) -> str | None:
        quoted = _QUOTED_CONTRACT.search(question)
        if (quoted is not None
                and re.search(r"계약|공급|EPC", question[quoted.end():quoted.end() + 40],
                              flags=re.IGNORECASE)):
            return quoted.group("value").strip()
        investment = re.search(r"신규시설투자", question)
        if investment is not None:
            return investment.group(0)
        contract = re.search(r"계약", question)
        return None if contract is None else contract.group(0)

    @staticmethod
    def _field(pattern: re.Pattern[str], question: str) -> str | None:
        match = pattern.search(question)
        return None if match is None else match.group(0)

    @classmethod
    def _already_specific(
            cls, question: str, intent: SemanticIntent, target: str,
            ) -> bool:
        """Return True when the intent already names the event more exactly.

        This adapter exists to repair a *flattened* lifecycle request.  When
        the semantic model already returned an event target whose surface is a
        literal question span strictly more specific than the generic ``계약``
        fallback, replacing it throws away the only identity the canonical
        event backends can select on.
        """

        items = intent.answer_items
        if not items:
            return False
        # A quoted, singular contract whose public output already separates
        # the root content from the complete change history is a selected
        # lifecycle.  The closed-literal regrounder deliberately emits this
        # document/timeline shape.  Rebuilding it here as a generic event
        # drops the root-content role and lets the downstream collection path
        # widen one named contract into every issuer contract merely because
        # the Korean request contains ``모두``.  Preserve only the two literal
        # roles; plural requests (``계약들을 각각``) never satisfy this gate.
        if len(items) == 1:
            item = items[0]
            fields = tuple(item.output.field_surfaces)
            if (
                    item.target.kind == "document"
                    and item.output.shape == "timeline"
                    and item.target.surface.strip() == target
                    and item.target.surface in question
                    and any(re.search(r"최초\s*(?:체결|공시)?.*내용", field)
                            for field in fields)
                    and any(re.search(r"(?:전체\s*)?(?:변경|정정).*이력", field)
                            for field in fields)
                    and not re.search(r"계약들|계약들을|각각", question)):
                return True
        if is_complete_named_document_timeline_request(intent):
            item = items[0]
            return (
                item.target.surface.strip() == target
                and item.target.surface in question
            )
        # A ``document`` target naming one contract is as specific as an
        # ``event`` one, and the selected-event route already resolves it to a
        # single lifecycle.  Accept it only when the question carries no filing
        # date at all: a stated date is the identity this adapter exists to
        # rebuild, so those requests must still be repaired here.
        document_ok = (
            not _FULL_DATE.search(question)
            and _ROOT_MISSING.search(question) is None)
        root_surface = cls._root_date_surface(question)
        status_field = cls._field(_STATUS, question)
        status_periods = cls._status_periods(
            question, root_surface=root_surface)
        for item in items:
            if item.target.kind == "document":
                if not document_ok:
                    return False
            elif item.target.kind != "event":
                return False
            surface = item.target.surface.strip()
            if (len(surface) <= len(target) or surface not in question):
                return False
            # A literal event phrase is not enough to preserve an output
            # container that cannot represent every requested status point.
            # ``최초 공시일과 코퍼스 기준일에 각각`` is a two-observation
            # record even when HCX labels the answer as free-form narrative.
            # Rebuild only this question-proved mismatch; an already-record
            # event keeps its more specific target and follows the established
            # selected-event path unchanged.
            if status_field is not None and len(status_periods) > 1:
                root_aligned = (
                    root_surface is None
                    or root_surface in item.target.qualifier_surfaces
                )
                periods_aligned = (
                    list(item.scope.target_period_expressions)
                    == status_periods
                )
                field_aligned = status_field in item.output.field_surfaces
                if (item.output.shape != "record" or not root_aligned
                        or not periods_aligned or not field_aligned):
                    return False
        return True

    @classmethod
    def _preserve_literal_status_axes(
            cls, question: str, intent: SemanticIntent, target: str,
            ) -> SemanticIntent | None:
        """Keep a specific event and complete literal observation dates.

        A coordinated date can state its year once (``2025년 1월 23일과
        4월 2일``).  When HCX already keeps those two literal observation
        surfaces on one specific event, rebuilding the whole item loses both
        the event name and the inherited-year date.  Only repair the answer
        container in this closed shape; incomplete year/month axes still use
        the ordinary history rebuild below.
        """

        if (len(intent.answer_items) != 1 or intent.answer_groups
                or intent.unresolved_mentions):
            return None
        item = intent.answer_items[0]
        surface = item.target.surface.strip()
        periods = list(item.scope.target_period_expressions)
        fields = list(item.output.field_surfaces)
        if (item.target.kind != "event" or item.selection is not None
                or len(surface) <= len(target) or surface not in question
                or len(periods) < 2
                or any(period not in question for period in periods)
                or not fields or any(field not in question for field in fields)
                or not all(re.search(r"상태|유효|해지|종료", field)
                           for field in fields)):
            return None

        inherited_year: int | None = None
        for period in periods:
            value = period.strip()
            full = _FULL_DATE.fullmatch(value)
            partial = _PARTIAL_DATE.fullmatch(value)
            if full is not None:
                inherited_year = int(full.group("year"))
                month, day = int(full.group("month")), int(full.group("day"))
            elif partial is not None and inherited_year is not None:
                month = int(partial.group("month"))
                day = int(partial.group("day"))
            else:
                return None
            try:
                date(inherited_year, month, day)
            except ValueError:
                return None

        if item.output.shape == "record":
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        payload["answer_items"][0]["output"]["shape"] = "record"
        return SemanticIntent.model_validate(payload, strict=True)

    def _company_surface(
            self, question: str, intent: SemanticIntent,
            ) -> str | None:
        """Return one literal issuer surface, preferring grounded intent.

        A full-text company scan can also see a short group alias inside a
        longer legal name (for example the suffix ``건설``).  The provider's
        company mention, when uniquely canonical, is therefore checked first;
        otherwise the longest uniquely resolvable token prefix is used.
        """

        canonical = getattr(self.company_preflight, "canonical", None)
        resolver = getattr(canonical, "resolve_company", None)
        candidates: list[tuple[str, str]] = []
        if callable(resolver):
            for entity in intent.entities:
                if entity.kind_hint != "company":
                    continue
                surface = self.company_preflight.question_company_surface(
                    entity.surface, question)
                rows = resolver(surface) if surface else []
                if len(rows) == 1:
                    candidates.append((rows[0].corp_code, surface))
            if not candidates:
                for token in re.split(r"[\s,.?!·/()\[\]'\"]+", question):
                    for size in range(len(token), 1, -1):
                        surface = token[:size]
                        rows = resolver(surface)
                        if len(rows) == 1:
                            candidates.append((rows[0].corp_code, surface))
                            break
        by_code: dict[str, str] = {}
        for code, surface in candidates:
            current = by_code.get(code)
            if current is None or len(surface) > len(current):
                by_code[code] = surface
        if len(by_code) == 1:
            return next(iter(by_code.values()))
        return self.company_preflight.unique_question_company_surface(question)

    @classmethod
    def _status_periods(
            cls, question: str, *, root_surface: str | None,
            ) -> list[str]:
        dates = [match.group("value") for match in cls._dates(question)]
        # The root filing date is an event selector unless the question asks
        # for the state at the first disclosure itself.
        asks_first_state = bool(re.search(r"최초\s*공시일.*각각\s*어떤\s*상태", question))
        values = [
            value for value in dates
            if value != root_surface or asks_first_state
        ]
        values.extend(match.group(0) for match in _YEAR_END.finditer(question))
        corpus = _CORPUS_POINT.search(question)
        if corpus is not None:
            values.append(corpus.group(0))
        elif _CURRENT_POINT.search(question) is not None:
            values.append(_CURRENT_POINT.search(question).group(0))
        return _ordered_unique(values)

    @staticmethod
    def _item(
            *, item_id: str, target: str, root_surface: str | None,
            field: str, shape: str, periods: list[str],
            ) -> dict[str, object]:
        return {
            "item_id": item_id,
            "operation": "retrieve",
            "target": {
                "kind": "event", "surface": target,
                "entity_refs": ["entity-1"],
                "qualifier_surfaces": (
                    [] if root_surface is None else [root_surface]),
            },
            "scope": {
                "target_period_expressions": periods,
                "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": shape, "projection_mode": "named_fields",
                "field_surfaces": [field], "presentation": "auto",
            },
        }

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if not isinstance(question, str) or not question.strip():
            return intent
        company = self._company_surface(question, intent)
        # One explicit public receipt fixes the event identity.  A count of
        # actual changes or a first/latest observed-value comparison is a
        # bounded history, even when the provider splits its roles into
        # several metric/event items.  Keep all requested axes in the question;
        # the correction backend proves every selected step independently.
        receipts = list(dict.fromkeys(re.findall(r"(?<!\d)20\d{12}(?!\d)", question)))
        observed_history = (
            "정정" in question and "계약" in question
            and (bool(re.search(r"처음.*최신", question))
                 or bool(re.search(r"실제로.*(?:접수번호별|한\s*번씩|세어)", question))))
        if company and len(receipts) == 1 and observed_history:
            return SemanticIntent.model_validate({
                "schema_version": intent.schema_version,
                "entities": [{"entity_id": "entity-1", "kind_hint": "company", "surface": company}],
                "answer_items": [{
                    "item_id": "item-1", "operation": "retrieve",
                    "target": {"kind": "document", "surface": "계약",
                               "entity_refs": ["entity-1"], "qualifier_surfaces": []},
                    "scope": {"target_period_expressions": [receipts[0]],
                              # The history backend binds the literal until
                              # date to CorrectionDateRoles.correction_to.
                              "as_of_expression": None,
                              "document_group_expression": None, "scope_qualifier_expressions": []},
                    "selection": None,
                    "output": {"shape": "narrative", "projection_mode": "whole_target",
                               "field_surfaces": [], "presentation": "auto"}}],
                "answer_groups": [], "premises": [], "unresolved_mentions": [], "presentation": "auto",
            }, strict=True)
        target = self._target_surface(question)
        history_field = self._field(_HISTORY, question)
        status_field = self._field(_STATUS, question)
        correction_date = self._correction_date_surface(question)
        root_date = self._root_date_surface(question)
        asks_origin = bool(
            correction_date and _ROOT_MISSING.search(question)
            and _CORRECTION.search(question))
        if company is None or target is None:
            return intent
        preserved = self._preserve_literal_status_axes(question, intent, target)
        if preserved is not None:
            return preserved
        if self._already_specific(question, intent, target):
            return intent

        # A dated correction whose root is requested is a *bounded observable
        # timeline*, not an issuer-wide document collection.  Keep two literal
        # field roles so the selected-event authority can return the visible
        # correction chain and attach a typed missing-root limitation.  It is
        # important that this is not whole-target: a generic document fallback
        # would otherwise widen one dated contract to every filing of issuer.
        if asks_origin:
            root_role = _ROOT_MISSING.search(question)
            lineage_role = history_field or status_field
            if root_role is None or lineage_role is None:
                return intent
            return SemanticIntent.model_validate({
                "schema_version": intent.schema_version,
                "entities": [{
                    "entity_id": "entity-1", "kind_hint": "company",
                    "surface": company,
                }],
                "answer_items": [{
                    "item_id": "item-1", "operation": "retrieve",
                    "target": {
                        "kind": "document", "surface": target,
                        "entity_refs": ["entity-1"],
                        "qualifier_surfaces": [],
                    },
                    "scope": {
                        "target_period_expressions": [correction_date],
                        "as_of_expression": None,
                        "document_group_expression": None,
                        "scope_qualifier_expressions": [],
                    },
                    "selection": None,
                    "output": {
                        "shape": "timeline",
                        "projection_mode": "named_fields",
                        "field_surfaces": [
                            root_role.group(0), lineage_role],
                        "presentation": "auto",
                    },
                }],
                "answer_groups": [], "premises": [],
                "unresolved_mentions": [], "presentation": "auto",
            }, strict=True)

        # A dated correction history without a missing-root demand belongs to
        # the dedicated correction relation backend.  Whole-target output lets
        # that backend enumerate only the selected canonical lineage.
        if correction_date is not None and history_field is not None:
            surface_match = re.search(r"(?:계약\s*)?(?:기재\s*)?정정\s*공시", question)
            surface = target if target != "계약" else (
                surface_match.group(0).strip() if surface_match is not None else target)
            return SemanticIntent.model_validate({
                "schema_version": intent.schema_version,
                "entities": [{
                    "entity_id": "entity-1", "kind_hint": "company",
                    "surface": company,
                }],
                "answer_items": [{
                    "item_id": "item-1", "operation": "retrieve",
                    "target": {
                        "kind": "document", "surface": surface,
                        "entity_refs": ["entity-1"],
                        "qualifier_surfaces": [],
                    },
                    "scope": {
                        "target_period_expressions": [correction_date],
                        "as_of_expression": None,
                        "document_group_expression": None,
                        "scope_qualifier_expressions": [],
                    },
                    "selection": None,
                    "output": {
                        "shape": "narrative", "projection_mode": "whole_target",
                        "field_surfaces": [], "presentation": "auto",
                    },
                }],
                "answer_groups": [], "premises": [],
                "unresolved_mentions": [], "presentation": "auto",
            }, strict=True)

        if history_field is None and status_field is None:
            return intent
        items: list[dict[str, object]] = []
        if history_field is not None:
            items.append(self._item(
                item_id=f"item-{len(items) + 1}", target=target,
                root_surface=root_date, field=history_field,
                shape="timeline", periods=[]))
        if status_field is not None:
            periods = self._status_periods(
                question, root_surface=root_date)
            items.append(self._item(
                item_id=f"item-{len(items) + 1}", target=target,
                root_surface=root_date, field=status_field,
                shape=("record" if len(periods) > 1 else "scalar"),
                periods=periods))
        return SemanticIntent.model_validate({
            "schema_version": intent.schema_version,
            "entities": [{
                "entity_id": "entity-1", "kind_hint": "company",
                "surface": company,
            }],
            "answer_items": items,
            "answer_groups": [], "premises": [],
            "unresolved_mentions": [], "presentation": "auto",
        }, strict=True)


__all__ = ["EventHistoryIntentRegrounder"]
