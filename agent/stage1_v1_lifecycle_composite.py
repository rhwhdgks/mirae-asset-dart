"""Canonical lifecycle intermediate for event status and source-attribute pairs.

The result is deliberately below the compiler boundary: it contains canonical
coordinates and proofs, not answer values or fixture-shaped plans.  A future
backend can turn it into one of several typed resolutions without rediscovering
issuers, event identity, correction chronology, or same-day ambiguity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Literal, Protocol

from .event_preflight import _contains_any
from .semantic_intent_v1 import SemanticIntent

_COUNTERPARTY_PATH = re.compile(r"계약\s*상대|상대방|거래\s*상대")
_CONTRACT_NAME_PATH = re.compile(r"계약\s*명|계약명칭")
_CONTRACT = re.compile(r"계약|공급|수주|판매")
_TERMINATION = re.compile(r"해지|종료|취소|철회|깨진|파기")
_CORRECTION = re.compile(r"정정|변경\s*이력|정정\s*전후|변경\s*흐름")
# ``해지된 계약`` names the event; ``해지 공시가 확인되는지`` asks about it.
# Only the first is an identity cue.  Treating the second as one is circular:
# the event can be found only if a termination exists, so the request can
# report "terminated" and never the equally factual "no termination observed".
# Mirrors the compiler's ``해지종료공시관측여부`` slot rule.
_TERMINATION_OBSERVATION_QUESTION = re.compile(
    r"(?=[^.?!]*(?:해지|종료))(?=[^.?!]*공시)"
    r"[^.?!]*(?:확인|존재|관측|여부)")
_FULL_DATE = re.compile(
    r"(?:(?P<year>(?:20)?[0-9]{2})\s*년\s*)?(?P<month>1[0-2]|0?[1-9])\s*월\s*(?P<day>3[01]|[12][0-9]|0?[1-9])\s*일?")
_CUTOFF = re.compile(r"20[0-9]{2}\s*년\s*(?:1[0-2]|0?[1-9])\s*월\s*(?:3[01]|[12][0-9]|0?[1-9])\s*일?\s*까지")
_YEAR_END = re.compile(r"(?<![0-9])(?P<year>20[0-9]{2})\s*년\s*말")
_TARGET_STOP = frozenset({
    "계약", "공급", "배터리", "해지", "정정", "전후", "흐름", "이유", "조건",
    "금액", "상태", "공시", "계약이", "깨진", "알려줘",
})


class CanonicalCompanyLike(Protocol):
    corp_code: str
    corp_name: str


class CanonicalTimelineLike(Protocol):
    event_key: str
    root_rcept_no: str
    corp_code: str
    corp_name: str
    observations: Iterable[Any]


class CanonicalLifecycleLike(Protocol):
    def resolve_company(self, name: str) -> list[CanonicalCompanyLike]: ...
    def fields(self, **kwargs: object) -> Iterable[Any]: ...
    def event_timeline(self, **kwargs: object) -> CanonicalTimelineLike | None: ...


LifecycleStatus = Literal["resolved", "ambiguous_same_day", "ambiguous_event"]
LifecycleAttributeKind = Literal[
    "contract_amount", "termination_amount", "termination_reason",
    "effectiveness_condition",
]


def is_complete_named_document_timeline_request(intent: SemanticIntent) -> bool:
    """Whether one named document already carries a complete timeline contract.

    The selected-event authority owns this shape: its document selector is the
    strictest contract-name binding and it emits one timeline task for the
    history and status coordinates together.  Lifecycle composite remains for
    flattened or multi-item lifecycle intents.
    """

    if len(intent.answer_items) != 1:
        return False
    item = intent.answer_items[0]
    fields = item.output.field_surfaces
    return bool(
        item.target.kind == "document"
        and item.output.shape == "timeline"
        and len(item.scope.target_period_expressions) >= 2
        and item.scope.as_of_expression is not None
        and any(re.search(r"변경|정정|이력", field) for field in fields)
        and any(re.search(r"상태|유효", field) for field in fields)
    )


@dataclass(frozen=True, slots=True)
class LifecycleAttributeProof:
    kind: LifecycleAttributeKind
    source_receipt: str
    evidence_id: str
    path: str
    locator: str


@dataclass(frozen=True, slots=True)
class LifecycleCompositeSelection:
    status: LifecycleStatus
    issuer_corp_code: str
    issuer_corp_name: str
    as_of: str
    counterparty_terms: tuple[str, ...]
    event_key: str | None
    root_receipt: str | None
    same_day_receipts: tuple[str, ...]
    status_timepoints: tuple[str, ...]
    termination_receipts: tuple[str, ...]
    correction_receipts: tuple[str, ...]
    attributes: tuple[LifecycleAttributeProof, ...]
    proof_refs: tuple[str, ...]


def _key(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", value).casefold()


def _dates(surfaces: Iterable[str]) -> tuple[str, ...]:
    values: list[str] = []
    # A coordinated question states the year once: ``2025년 1월 23일과 4월
    # 2일에 각각``.  The provider may keep those two points as one surface or
    # as two, and only the first then carries the year.  Carry it across the
    # ordered surfaces exactly as it is already carried within one surface,
    # so the second observation point is not silently dropped.
    year: int | None = None
    for surface in surfaces:
        for match in _FULL_DATE.finditer(surface):
            if match.group("year"):
                raw_year = int(match.group("year"))
                year = raw_year + 2000 if raw_year < 100 else raw_year
            if year is None:
                continue
            try:
                stamp = f"{year:04d}{int(match.group('month')):02d}{int(match.group('day')):02d}"
                date(year, int(match.group("month")), int(match.group("day")))
            except ValueError:
                continue
            if stamp not in values:
                values.append(stamp)
        for match in _YEAR_END.finditer(surface):
            stamp = f"{int(match.group('year')):04d}1231"
            if stamp not in values:
                values.append(stamp)
    return tuple(values)


def _observation_day(observation: Any) -> str | None:
    """Return a canonical observation day without trusting display wording."""
    value = str(getattr(observation, "observed_at", ""))
    match = re.search(r"(?<![0-9])(20[0-9]{2})([01][0-9])([0-3][0-9])", value)
    return match.group(0) if match else None


def _item_surfaces(intent: SemanticIntent) -> list[str]:
    values: list[str] = []
    for item in intent.answer_items:
        values.append(item.target.surface)
        values.extend(item.target.qualifier_surfaces)
        values.extend(item.scope.target_period_expressions)
        values.extend(item.output.field_surfaces)
        if item.scope.as_of_expression:
            values.append(item.scope.as_of_expression)
    return values


def _seed_days(intent: SemanticIntent) -> tuple[str, ...]:
    """Return exact event-origin dates carried on the target axis."""

    values = _dates(
        surface
        for item in intent.answer_items
        for surface in item.target.qualifier_surfaces
    )
    # Two or more qualifier dates are the established external-event status
    # topology.  A seed coordinate is deliberately singular.
    return values if len(values) == 1 else ()


def _status_days(
        intent: SemanticIntent, *, question: str, corpus_cutoff: str,
        ) -> tuple[str, ...]:
    """Return observation dates without reusing an event seed as status.

    A filing date on ``target.qualifier_surfaces`` identifies the lifecycle.
    Observation points live in scope/output.  Current/latest wording is a
    deliberate corpus-cutoff default and does not invent a filing time.
    """

    surfaces: list[str] = []
    asks_status = False
    for item in intent.answer_items:
        fields = list(item.output.field_surfaces)
        if not any(re.search(r"상태|유효|살아|해지|종료", value)
                   for value in fields):
            continue
        asks_status = True
        surfaces.extend(item.scope.target_period_expressions)
        surfaces.extend(fields)
        if item.scope.as_of_expression:
            surfaces.append(item.scope.as_of_expression)
        qualifier_days = _dates(item.target.qualifier_surfaces)
        if not item.scope.target_period_expressions and len(qualifier_days) > 1:
            surfaces.extend(item.target.qualifier_surfaces)
    values = list(_dates(surfaces))
    if asks_status and (
            re.search(r"최신\s*(?:유효본|상태)|현재\s*(?:유효|상태)|지금\s*해지", question)
            or any(re.sub(r"\s+", "", value) in {
                "최신", "현재", "지금", "코퍼스기준일"}
                for value in surfaces)):
        values.append(corpus_cutoff)
    return tuple(dict.fromkeys(values))


def _issuer(canonical: CanonicalLifecycleLike, intent: SemanticIntent) -> CanonicalCompanyLike | None:
    rows: dict[str, CanonicalCompanyLike] = {}
    for entity in intent.entities:
        if entity.kind_hint != "company":
            continue
        found = canonical.resolve_company(entity.surface)
        if len(found) == 1:
            rows[found[0].corp_code] = found[0]
    return next(iter(rows.values())) if len(rows) == 1 else None


def _counterparty_terms(
        canonical: CanonicalLifecycleLike, intent: SemanticIntent,
        issuer: CanonicalCompanyLike | None,
        ) -> tuple[str, ...]:
    values: list[str] = []
    for entity in intent.entities:
        if entity.kind_hint == "counterparty":
            values.append(entity.surface)
        elif entity.kind_hint == "company":
            found = canonical.resolve_company(entity.surface)
            if not (issuer is not None and len(found) == 1
                    and found[0].corp_code == issuer.corp_code):
                values.append(entity.surface)
        elif entity.kind_hint == "event":
            values.extend(re.findall(r"[A-Za-z가-힣]{2,}", entity.surface))
    return tuple(dict.fromkeys(value for value in values if value.strip()))


def _infer_issuer(
        canonical: CanonicalLifecycleLike, *, as_of: str,
        counterparties: tuple[str, ...],
        ) -> CanonicalCompanyLike | None:
    if not counterparties:
        return None
    matches: dict[tuple[str, str], set[str]] = {}
    for row in canonical.fields(as_of=as_of, label="계약상대"):
        value = getattr(row, "value", None)
        if not isinstance(value, str) or not _COUNTERPARTY_PATH.search(str(getattr(row, "path", ""))):
            continue
        if not any(_contains_any(term, [value]) for term in counterparties):
            continue
        code, name = str(getattr(row, "corp_code", "")), str(getattr(row, "corp_name", ""))
        if re.fullmatch(r"[0-9]{8}", code) and name:
            matches.setdefault((code, name), set()).add(_key(value))
    candidates = [
        (code, name) for (code, name), found in matches.items()
        if all(any(_key(term) in value or value in _key(term) for value in found)
               for term in counterparties)
    ]
    if len(candidates) != 1:
        return None
    return _Company(*candidates[0])


@dataclass(frozen=True, slots=True)
class _Company:
    corp_code: str
    corp_name: str


def _receipt_rows(canonical: CanonicalLifecycleLike, *, as_of: str, corp_code: str) -> dict[str, list[Any]]:
    rows: dict[str, list[Any]] = {}
    for row in canonical.fields(as_of=as_of, corp_code=corp_code):
        receipt = str(getattr(row, "rcept_no", ""))
        if re.fullmatch(r"[0-9]{14}", receipt):
            rows.setdefault(receipt, []).append(row)
    return rows


def _receipt_matches_counterparty(rows: list[Any], terms: tuple[str, ...]) -> bool:
    if not terms:
        return True
    values = [str(getattr(row, "value", "")) for row in rows
              if _COUNTERPARTY_PATH.search(str(getattr(row, "path", "")))]
    return bool(values) and any(_contains_any(term, values) for term in terms)


def _event_matches_contract(rows: list[Any], needs_contract: bool) -> bool:
    if not needs_contract:
        return True
    # ``event_type`` is intentionally absent on some schema-1.9 field rows.
    # Contract identity is still explicit in canonical form paths such as
    # ``체결계약명`` and ``판매·공급계약 구분``.  This is only a domain gate;
    # issuer/date/target facets and the event timeline perform selection.
    return any(
        _CONTRACT.search(" ".join((
            str(getattr(row, "event_type", "") or ""),
            str(getattr(row, "path", "") or ""),
        ))) is not None
        for row in rows
    )


def _target_terms(intent: SemanticIntent) -> tuple[str, ...]:
    """Return identity-bearing target words only.

    An acronym such as ``PF`` is normally the only event discriminator in a
    conversational target.  Korean connective words around it ("안", "돼서",
    "건") describe the request, not a canonical event identity, and requiring
    each of them would make a proven PF timeline look unmatched.
    """

    raw_tokens: list[str] = []
    values: list[str] = []
    for item in intent.answer_items:
        for token in re.findall(r"[A-Za-z가-힣]{2,}", item.target.surface):
            raw_tokens.append(token)
            compact = _key(token)
            if compact and compact not in {_key(value) for value in _TARGET_STOP}:
                values.append(token)
    acronyms = [token for token in raw_tokens if re.fullmatch(r"[A-Za-z]{2,}", token)]
    if acronyms:
        return tuple(dict.fromkeys(acronyms))
    return tuple(dict.fromkeys(values))


def _timeline_matches_target_terms(
        timeline: CanonicalTimelineLike, rows_by_receipt: dict[str, list[Any]],
        terms: tuple[str, ...]) -> bool:
    if not terms:
        return False
    receipts = {timeline.root_rcept_no, *(
        str(getattr(row, "rcept_no", "")) for row in timeline.observations)}
    values = [str(getattr(row, "value", ""))
              for receipt in receipts for row in rows_by_receipt.get(receipt, ())]
    return bool(values) and all(_contains_any(term, values) for term in terms)


# Fields whose value *names* the contract, as opposed to describing it.
_CONTRACT_NAME_LABEL = re.compile(r"계약\s*명|계약\s*건\s*명|공사\s*명|사업\s*명")


def _named_target_score(
        timeline: CanonicalTimelineLike, rows_by_receipt: dict[str, list[Any]],
        terms: tuple[str, ...]) -> int:
    """How many of the named terms appear in a field that *names* the contract?

    Two contracts can share a project: one filing carries it as its
    ``체결계약명`` while another mentions the same project only inside
    ``세부내용``.  A question that names a contract points at the first.

    A count rather than an all-or-nothing test, because a question names a
    contract with more than its registered name — «'Widow Hill BESS PJT' EPC
    공급계약» adds the contract type, and ``EPC`` is nowhere in the filing's
    name field.  Requiring every term there would score both candidates zero
    and settle nothing.
    """

    if not terms:
        return 0
    receipts = {timeline.root_rcept_no, *(
        str(getattr(row, "rcept_no", "")) for row in timeline.observations)}
    values = [
        str(getattr(row, "value", ""))
        for receipt in receipts
        for row in rows_by_receipt.get(receipt, ())
        if _CONTRACT_NAME_LABEL.search(str(getattr(row, "label", "") or ""))
    ]
    if not values:
        return 0
    return sum(1 for term in terms if _contains_any(term, values))


def _receipt_matches_target_terms(rows: list[Any], terms: tuple[str, ...]) -> bool:
    """Cheap prefilter for a literal named target before timeline reads."""

    values = [str(getattr(row, "value", "")) for row in rows]
    return bool(values) and all(_contains_any(term, values) for term in terms)


def _requested_attribute_kinds(intent: SemanticIntent) -> tuple[LifecycleAttributeKind, ...]:
    text = " ".join(_item_surfaces(intent))
    if re.search(r"정정|변경\s*이력|전후|흐름", text):
        # In a correction history, amount/period are changed-field axes.  They
        # are not one scalar lifecycle attribute selected from the latest
        # correction receipt.
        return ()
    values: list[LifecycleAttributeKind] = []
    def add(kind: LifecycleAttributeKind) -> None:
        if kind not in values:
            values.append(kind)
    if re.search(r"계약\s*금액", text):
        add("contract_amount")
    if re.search(r"해지\s*금액", text):
        add("termination_amount")
    if re.search(r"해지.*(?:사유|이유)|(?:사유|이유).*해지", text):
        add("termination_reason")
    if re.search(r"효력\s*발생\s*조건|효력.*조건", text):
        add("effectiveness_condition")
    return tuple(values)


def _has_seeded_single_status_axis(intent: SemanticIntent) -> bool:
    """Whether one status request already separates event seed from as-of.

    An exact contract date in ``target_period_expressions`` identifies the
    event; an explicit ``as_of_expression`` identifies when its status is
    asked.  Those are distinct axes, not two lifecycle observation points.
    The selected-event authority preserves that distinction (including the
    corpus-cutoff default), whereas lifecycle lowering would reinterpret the
    seed date as the answer timepoint.
    """

    if len(intent.answer_items) != 1:
        return False
    item = intent.answer_items[0]
    fields = tuple(item.output.field_surfaces)
    return bool(
        item.output.shape == "scalar"
        and len(fields) == 1
        and re.search(r"상태|유효|살아\s*있|끝난|해지", fields[0])
        and item.scope.as_of_expression is not None
        and len(item.scope.target_period_expressions) == 1
    )


def _attribute_pattern(kind: LifecycleAttributeKind) -> re.Pattern[str]:
    return {
        "contract_amount": re.compile(r"계약\s*금액"),
        "termination_amount": re.compile(r"해지\s*금액"),
        "termination_reason": re.compile(
            r"해지.*(?:사유|이유|원인)|(?:사유|이유|원인).*해지"),
        "effectiveness_condition": re.compile(r"효력\s*발생\s*조건|효력.*조건"),
    }[kind]


def _attribute_source_receipts(
        kind: LifecycleAttributeKind, *, root: str,
        termination: tuple[str, ...], corrections: tuple[str, ...],
        ) -> tuple[str, ...]:
    if kind in {"termination_amount", "termination_reason"}:
        return termination
    if kind == "contract_amount" and corrections:
        return corrections
    return (root,)


def _select_attribute(
        kind: LifecycleAttributeKind, rows_by_receipt: dict[str, list[Any]],
        *, root: str, termination: tuple[str, ...], corrections: tuple[str, ...],
        ) -> LifecycleAttributeProof | None:
    receipts = _attribute_source_receipts(
        kind, root=root, termination=termination, corrections=corrections)
    def matches(row: Any) -> bool:
        path = str(getattr(row, "path", ""))
        if _attribute_pattern(kind).search(path):
            return True
        # Some filings put a negotiated effectiveness condition in their
        # proof-bearing explanatory narrative rather than a dedicated table
        # row.  The lifecycle/root coordinate is already uniquely verified;
        # admit only a concrete effectiveness statement from that coordinate.
        if kind == "effectiveness_condition":
            value = str(getattr(row, "value", ""))
            return bool(re.search(r"효력(?:이)?\s*(?:발생|생기)", value))
        return False

    candidates = [
        row for receipt in receipts for row in rows_by_receipt.get(receipt, ())
        if matches(row)
        and isinstance(getattr(row, "evidence_id", None), str)
        and str(getattr(row, "evidence_id"))
    ]
    if len(candidates) != 1:
        return None
    row = candidates[0]
    return LifecycleAttributeProof(
        kind=kind, source_receipt=str(getattr(row, "rcept_no")),
        evidence_id=str(getattr(row, "evidence_id")), path=str(getattr(row, "path")),
        locator=str(getattr(row, "locator", "")),
    )


def select_lifecycle_composite(
        *, canonical: CanonicalLifecycleLike, intent: SemanticIntent,
        question: str, reference_date: date, corpus_cutoff: str,
        ) -> LifecycleCompositeSelection | None:
    """Return one lifecycle coordinate set, or ``None`` when out of scope.

    Ambiguity is represented explicitly only after the issuer was proven.  No
    receipt ordering or value comparison is used to pick one event.
    """
    del reference_date
    if not re.fullmatch(r"[0-9]{8}", corpus_cutoff):
        raise ValueError("lifecycle corpus cutoff 형식이 잘못되었습니다")
    if is_complete_named_document_timeline_request(intent):
        return None
    surfaces = _item_surfaces(intent)
    all_text = " ".join((question, *surfaces))
    needs_contract = bool(_CONTRACT.search(all_text))
    needs_termination = bool(_TERMINATION.search(all_text))
    asks_termination_observation = bool(
        _TERMINATION_OBSERVATION_QUESTION.search(all_text))
    needs_correction = bool(_CORRECTION.search(all_text))
    seed_days = _seed_days(intent)
    status_points = _status_days(
        intent, question=question, corpus_cutoff=corpus_cutoff)
    attributes = _requested_attribute_kinds(intent)
    # A seed-date plus an explicit as-of status request is a normal selected
    # event query, not a lifecycle comparison.  Do not let the existence of a
    # historical correction/termination in the corpus rewrite its requested
    # answer timepoint.
    if _has_seeded_single_status_axis(intent):
        status_points = ()
    asks_root_content_and_history = bool(
        needs_correction and not seed_days and not status_points
        and re.search(r"최초\s*(?:체결|공시)?\s*내용", all_text)
    )
    # A named contract's root-content + whole-history request belongs to the
    # selected-event/correction authority.  Lifecycle lowering owns status
    # replay and mixed lifecycle attributes; it must not seize this pure
    # document-history shape merely because the timeline is unique.
    if asks_root_content_and_history:
        return None
    # A contract noun only establishes an event-search domain.  Lifecycle
    # authority needs an actual lifecycle demand: status observations,
    # corrections/termination, or source attributes.  Otherwise a normal
    # document timeline should remain with the selected-event backend, which
    # owns the stricter contract-name selector.
    if not (needs_termination or needs_correction or status_points or attributes
            or any(item.selection is not None and item.selection.mode == "latest"
                   for item in intent.answer_items)):
        return None
    issuer = _issuer(canonical, intent)
    counterparties = _counterparty_terms(canonical, intent, issuer)
    if issuer is None:
        issuer = _infer_issuer(canonical, as_of=corpus_cutoff, counterparties=counterparties)
    if issuer is None:
        return None
    try:
        rows_by_receipt = _receipt_rows(canonical, as_of=corpus_cutoff, corp_code=issuer.corp_code)
    except (AttributeError, TypeError, ValueError):
        return None
    matching_receipts = [
        receipt for receipt, rows in rows_by_receipt.items()
        if _receipt_matches_counterparty(rows, counterparties)
        and _event_matches_contract(rows, needs_contract)
    ]
    if seed_days:
        # The exact public filing day is the cheapest and strongest event
        # coordinate.  Apply it before timeline reads so an issuer with years
        # of unrelated contracts cannot turn one bounded query into a full
        # corpus scan.
        matching_receipts = [
            receipt for receipt in matching_receipts
            if receipt[:8] == seed_days[0]
        ]
    else:
        target_terms = _target_terms(intent)
        if target_terms:
            matching_receipts = [
                receipt for receipt in matching_receipts
                if _receipt_matches_target_terms(
                    rows_by_receipt[receipt], target_terms)
            ]
    if not matching_receipts:
        return None
    cutoff_days = _dates([match.group(0) for match in _CUTOFF.finditer(question)])
    if cutoff_days and any(item.selection is not None and item.selection.mode == "latest"
                           for item in intent.answer_items):
        same_day = sorted(receipt for receipt in matching_receipts
                          if receipt[:8] == cutoff_days[0])
        if len(same_day) > 1:
            return LifecycleCompositeSelection(
                status="ambiguous_same_day", issuer_corp_code=issuer.corp_code,
                issuer_corp_name=issuer.corp_name, as_of=cutoff_days[0],
                counterparty_terms=counterparties, event_key=None, root_receipt=None,
                same_day_receipts=tuple(same_day), status_timepoints=(),
                termination_receipts=(), correction_receipts=(), attributes=(),
                proof_refs=tuple(f"canonical:same-day:{receipt}" for receipt in same_day),
            )
    timelines: dict[str, CanonicalTimelineLike] = {}
    for receipt in matching_receipts:
        try:
            timeline = canonical.event_timeline(as_of=corpus_cutoff, rcept_no=receipt,
                                                verify_evidence=True)
        except (AttributeError, TypeError, ValueError):
            return None
        if timeline is None or timeline.corp_code != issuer.corp_code:
            continue
        timelines[timeline.event_key] = timeline
    if seed_days:
        if len(seed_days) != 1:
            return None
        timelines = {
            key: candidate for key, candidate in timelines.items()
            if str(candidate.root_rcept_no)[:8] == seed_days[0]
        }
    # Lifecycle cues constrain event identity through canonical observation
    # roles.  A target such as ``PF ... 깨진`` must not retain a different
    # active PF contract merely because both timelines contain the acronym.
    if needs_termination and not asks_termination_observation:
        timelines = {
            key: candidate for key, candidate in timelines.items()
            if any(bool(getattr(row, "is_termination", False))
                   for row in candidate.observations)
        }
    if needs_correction:
        timelines = {
            key: candidate for key, candidate in timelines.items()
            if any(bool(getattr(row, "is_correction", False))
                   for row in candidate.observations)
        }
    # A broad issuer-level contract surface is not itself an identity.  When
    # the question requests a source attribute pair, however, the pair may
    # prove exactly one lifecycle without relying on wording or receipt IDs.
    if len(timelines) > 1 and attributes:
        eligible: dict[str, CanonicalTimelineLike] = {}
        for key, candidate in timelines.items():
            observations = tuple(candidate.observations)
            termination = tuple(sorted(
                row.rcept_no for row in observations
                if bool(getattr(row, "is_termination", False))))
            corrections = tuple(sorted(
                row.rcept_no for row in observations
                if bool(getattr(row, "is_correction", False))))
            if all(_select_attribute(
                    kind, rows_by_receipt, root=candidate.root_rcept_no,
                    termination=termination, corrections=corrections) is not None
                   for kind in attributes):
                eligible[key] = candidate
        if len(eligible) == 1:
            timelines = eligible
    if len(timelines) > 1:
        terms = _target_terms(intent)
        eligible = {
            key: candidate for key, candidate in timelines.items()
            if _timeline_matches_target_terms(candidate, rows_by_receipt, terms)
        }
        if len(eligible) == 1:
            timelines = eligible
    if len(timelines) > 1:
        scores = {
            key: _named_target_score(candidate, rows_by_receipt, terms)
            for key, candidate in timelines.items()
        }
        best = max(scores.values())
        # Narrow only on a strict, non-zero win.  A tie means the source does
        # not distinguish the candidates and the request stays ambiguous.
        if best and sum(1 for value in scores.values() if value == best) == 1:
            timelines = {key: candidate for key, candidate in timelines.items()
                         if scores[key] == best}
    if len(timelines) != 1:
        return LifecycleCompositeSelection(
            status="ambiguous_event", issuer_corp_code=issuer.corp_code,
            issuer_corp_name=issuer.corp_name, as_of=corpus_cutoff,
            counterparty_terms=counterparties, event_key=None, root_receipt=None,
            same_day_receipts=(), status_timepoints=(), termination_receipts=(),
            correction_receipts=(), attributes=(),
            proof_refs=tuple(sorted(f"canonical:event-candidate:{key}" for key in timelines)),
        )
    timeline = next(iter(timelines.values()))
    observations = tuple(timeline.observations)
    # Status timepoints are as-of coordinates, not necessarily filing days.
    # Stage2 replays the last canonical observation available at each point;
    # requiring an observation on the exact calendar day would reject normal
    # year-end and corpus-cutoff questions.
    termination = tuple(sorted(
        observation.rcept_no for observation in observations
        if bool(getattr(observation, "is_termination", False))))
    corrections = tuple(sorted(
        observation.rcept_no for observation in observations
        if bool(getattr(observation, "is_correction", False))))
    if needs_correction and needs_termination and not status_points:
        # When the same lifecycle has corrections both before and on the
        # termination day, the meaningful state transition is the last
        # observation before termination -> the termination observation.  A
        # same-day correction is provenance, not a third semantic timepoint.
        termination_rows = sorted(
            (row for row in observations
             if bool(getattr(row, "is_termination", False))),
            key=lambda row: (_observation_day(row) or "", row.rcept_no))
        if len(termination_rows) != 1:
            return None
        termination_day = _observation_day(termination_rows[0])
        prior_rows = [
            row for row in observations
            if (_observation_day(row) is not None and termination_day is not None
                and _observation_day(row) < termination_day)
        ]
        if termination_day is None or not prior_rows:
            return None
        prior_day = max(_observation_day(row) for row in prior_rows)
        if prior_day is None:
            return None
        status_points = (prior_day, termination_day)
    elif needs_correction and not status_points:
        correction_rows = [row for row in observations if bool(getattr(row, "is_correction", False))]
        # A single correction can still expose its before/after status pair.
        # A multi-correction history needs no synthetic status axis: the
        # correction task below already enumerates every proven observation.
        if len(correction_rows) == 1:
            previous = getattr(
                correction_rows[0], "previous_observation_rcept_no", None)
            if previous is None:
                return None
            previous_rows = [row for row in observations if row.rcept_no == previous]
            if len(previous_rows) != 1:
                return None
            previous_day = _observation_day(previous_rows[0])
            correction_day = _observation_day(correction_rows[0])
            if previous_day is None or correction_day is None:
                return None
            status_points = (previous_day, correction_day)
    proofs: set[str] = {f"canonical:event:{timeline.event_key}:{timeline.root_rcept_no}"}
    for observation in observations:
        proofs.add(f"canonical:event:{timeline.event_key}:{observation.rcept_no}")
        proofs.update(f"canonical:event-support:{value}"
                     for value in getattr(observation, "supporting_evidence_ids", ()))
    chosen_attributes: list[LifecycleAttributeProof] = []
    for kind in attributes:
        selected = _select_attribute(
            kind, rows_by_receipt, root=timeline.root_rcept_no,
            termination=termination, corrections=corrections)
        if selected is None:
            return None
        chosen_attributes.append(selected)
        proofs.add(f"canonical:field:{selected.evidence_id}")
    return LifecycleCompositeSelection(
        status="resolved", issuer_corp_code=issuer.corp_code,
        issuer_corp_name=issuer.corp_name, as_of=corpus_cutoff,
        counterparty_terms=counterparties, event_key=timeline.event_key,
        root_receipt=timeline.root_rcept_no, same_day_receipts=(),
        status_timepoints=status_points, termination_receipts=termination,
        correction_receipts=corrections, attributes=tuple(chosen_attributes),
        proof_refs=tuple(sorted(proofs)),
    )


__all__ = [
    "CanonicalLifecycleLike", "LifecycleAttributeProof",
    "LifecycleCompositeSelection", "is_complete_named_document_timeline_request",
    "select_lifecycle_composite",
]
