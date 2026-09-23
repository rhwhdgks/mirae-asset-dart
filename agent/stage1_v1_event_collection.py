"""Canonical-first selector for *collections* of disclosure events.

This module deliberately stops before compiler lowering.  A single event can
be represented by ``SelectedEventResolution`` already; a collection needs to
retain every canonical identity and the receipt(s) which caused it to enter
the set.  Keeping that intermediate form typed prevents list questions from
being accidentally reduced to one ``latest`` receipt.

No fixture/Ground-truth identifiers are read here.  Selection is driven by
the normalized semantic intent, the question's grounded surfaces and the
canonical read model only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import calendar
import re
from typing import Any, Iterable, Literal, Protocol

from .date_surface import parse_date_surface
from .event_preflight import _contains_any
from .semantic_intent_v1 import SemanticIntent


_DATE = re.compile(r"(?<![0-9])(?P<year>20[0-9]{2})\s*년(?:\s*(?P<month>1[0-2]|0?[1-9])\s*월(?:\s*(?P<day>3[01]|[12][0-9]|0?[1-9])\s*일?)?)?")
_QUARTER = re.compile(
    r"(?<![0-9])(?P<year>20[0-9]{2})\s*년?\s*"
    r"(?P<quarter>[1-4])\s*(?:분기|Q)", re.IGNORECASE)
_MONTH = re.compile(r"(?<![0-9년])(?P<month>1[0-2]|0?[1-9])\s*월")
_OBSERVATION_DATE = re.compile(
    r"(?:(?P<year>(?:20)?[0-9]{2})\s*년\s*)?(?P<month>1[0-2]|0?[1-9])\s*월\s*(?P<day>3[01]|[12][0-9]|0?[1-9])\s*일?")
_AS_OF = re.compile(r"(?:까지\s*(?:공개|공시|발표)|기준(?:으로)?|as\s*of)", re.I)
_TERMINATION = re.compile(r"해지|종료|취소|철회|깨진|파기")
_CONTRACT = re.compile(r"계약|공급|수주|판매")
_COUNTERPARTY_PATH = re.compile(r"계약\s*상대|상대방|거래\s*상대")
_CONTRACT_NAME_PATH = re.compile(r"계약\s*명|계약명칭")
_CONTRACT_FORM_PATH = re.compile(
    r"판매[ㆍ·]?\s*공급계약\s*구분|계약\s*\(\s*수주\s*\)\s*일|계약\s*내역")
_CORRECTION_FLOW = re.compile(r"정정\s*전후|전후\s*(?:정정|흐름)|정정.*흐름")
_INITIAL_SIGNING = re.compile(
    r"(?:최초|처음).{0,12}(?:체결|계약|공시)|"
    r"(?:체결|계약|공시).{0,12}(?:최초|처음)")
_FULL_CHANGE_HISTORY = re.compile(
    r"(?:전체|모든|전부).{0,12}(?:변경|정정).{0,12}(?:이력|내역|흐름)|"
    r"(?:변경|정정).{0,12}(?:이력|내역|흐름).{0,12}(?:전체|모든|전부)|"
    r"(?:후|이후|부터).{0,24}(?:(?:어떤|무슨)\s*내용이\s*)?"
    r"(?:변경|바뀌|달라)")
_MULTI_EVENT_IDENTITY = re.compile(
    r"(?:계약|사건|공시)(?:들|마다|별)|"
    r"(?:각|여러)\s*(?:계약|사건|공시)|"
    r"두\s*(?:건|계약)|둘\s*(?:의|을|를|은|는)?")
_PUBLIC_AVAILABILITY = re.compile(
    r"(?:공개된|공시된)\s*공시.*(?:확인(?:할)?|알)\s*수\s*있(?:는가|나)|"
    r"(?:확인(?:할)?|알)\s*수\s*있(?:는가|나).*?(?:공개된|공시된)\s*공시|"
    # A value-state question ("0원인지 비공개인지 구분해줘") asks the same
    # disclosed-vs-withheld question as the "확인할 수 있는가" phrasing above;
    # it just names the two candidate states instead of asking a yes/no.
    r"(?:0원|공란|미공시|비공개|유보).{0,6}(?:인지|아닌지).{0,20}구분"
)
_MANY = re.compile(
    r"표(?:로)?|목록|리스트|모두|전부|각각|별로|건들|둘|두\s*건|여럿|여러\s*건|"
    r"(?:중|가운데).*?(?:무엇|어떤|어느|확인되는|해당하는)"
)
_SET_EXISTENCE = re.compile(
    r"(?:계약|공시|건).{0,30}(?:존재(?:하는가|하나|해)|있(?:는가|나|나요))|"
    r"(?:존재(?:하는가|하나|해)|있(?:는가|나|나요)).{0,30}(?:계약|공시|건)"
)
# These are canonical DART event families, not question-specific labels.  The
# timeline proves membership; the type merely makes the resulting Stage2
# selector explicit enough to reproduce that membership from the corpus.
_CONTRACT_EVENT_TYPE = "단일판매공급계약체결"
_TERMINATION_EVENT_TYPE = "단일판매공급계약해지"
_SLOT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"상대|거래처|계약처"), "counterparty"),
    (re.compile(r"금액|대금|규모"), "amount"),
    (re.compile(r"사유|이유|원인"), "reason"),
    (re.compile(r"접수|공시번호|receipt", re.I), "rcept_no"),
    (re.compile(r"상태|유효|해지연결"), "status"),
    (re.compile(r"날짜|일자|언제"), "date"),
)
_STOP_TOKENS = frozenset({
    "공시", "공개", "계약", "공급", "해지", "확인", "비교", "정리", "알려줘",
    "무엇", "어떤", "있는", "중", "제공", "코퍼스", "각각", "표로", "계약을",
    "공급계약", "판매계약", "계약해지", "계약체결",
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


class CanonicalEventCollectionLike(Protocol):
    def resolve_company(self, name: str) -> list[CanonicalCompanyLike]: ...

    def fields(self, **kwargs: object) -> Iterable[Any]: ...

    def event_timeline(self, **kwargs: object) -> CanonicalTimelineLike | None: ...


@dataclass(frozen=True, slots=True)
class EventCollectionIdentity:
    """One canonical event identity retained by a collection selector."""

    event_key: str
    root_receipt: str
    matched_receipts: tuple[str, ...]
    matched_dates: tuple[str, ...]
    event_types: tuple[str, ...]
    counterparties: tuple[str, ...]
    contract_names: tuple[str, ...]
    proof_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EventCollectionSelection:
    """Typed, compiler-independent result for a canonical event list."""

    issuer_corp_code: str
    issuer_corp_name: str
    as_of: str
    event_from: str | None
    event_to: str | None
    requires_termination: bool
    root_contracts_with_confirmed_termination: bool
    selector_event_type: str | None
    selector_counterparty: str | None
    selector_keywords: tuple[str, ...]
    public_task_kind: Literal["event", "disclosure"]
    availability_query: bool
    requested_slots: tuple[str, ...]
    events: tuple[EventCollectionIdentity, ...]
    proof_refs: tuple[str, ...]
    collection_proof_ref: str | None = None
    argmax_slot: str | None = None
    # 이슈 #124 — argmax_slot 이 줄이는 극값의 방향. optional — 기존 list
    # 요청(argmax_slot=None)은 의미가 없으므로 "maximum" 그대로.
    argmax_direction: Literal["maximum", "minimum"] = "maximum"


@dataclass(frozen=True, slots=True)
class _ReceiptFacts:
    receipt: str
    receipt_date: str
    event_type: str | None
    is_contract_form: bool
    counterparties: tuple[str, ...]
    contract_names: tuple[str, ...]


def _key(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", value).casefold()


def _date_range(question: str, reference_date: date) -> tuple[str | None, str | None]:
    """Read one grounded year/month/day scope without assuming a fixture year."""

    quarter = _QUARTER.search(question)
    if quarter is not None:
        year = int(quarter.group("year"))
        number = int(quarter.group("quarter"))
        first_month, last_month = (number - 1) * 3 + 1, number * 3
        return (
            f"{year}{first_month:02d}01",
            f"{year}{last_month:02d}"
            f"{calendar.monthrange(year, last_month)[1]:02d}",
        )
    full = list(_DATE.finditer(question))
    # An explicit year dominates a bare month.  If the surface is an as-of
    # clause it constrains visibility, not the event set.
    if full:
        match = full[0]
        year = int(match.group("year"))
        month = match.group("month")
        day = match.group("day")
        if month is None:
            return f"{year}0101", f"{year}1231"
        month_i = int(month)
        if day is not None:
            stamp = f"{year}{month_i:02d}{int(day):02d}"
            return stamp, stamp
        return (f"{year}{month_i:02d}01",
                f"{year}{month_i:02d}{calendar.monthrange(year, month_i)[1]:02d}")
    bare = _MONTH.search(question)
    if bare is None:
        return None, None
    month_i = int(bare.group("month"))
    # A bare month denotes the most recent completed occurrence at the
    # supplied reference date.  This is date arithmetic, not corpus fitting.
    year = reference_date.year if month_i <= reference_date.month else reference_date.year - 1
    return (f"{year}{month_i:02d}01",
            f"{year}{month_i:02d}{calendar.monthrange(year, month_i)[1]:02d}")


def _as_of(question: str, corpus_cutoff: str, reference_date: date) -> str | None:
    if not _AS_OF.search(question):
        return corpus_cutoff
    matches = list(_DATE.finditer(question))
    if not matches:
        return None
    match = matches[0]
    if match.group("month") is None or match.group("day") is None:
        return None
    stamp = f"{int(match.group('year')):04d}{int(match.group('month')):02d}{int(match.group('day')):02d}"
    return stamp if stamp <= corpus_cutoff else None


def _explicit_observation_dates(question: str, item: Any) -> tuple[str, ...]:
    """Read two observation points (including Korean shortened second date).

    They describe status timepoints, not a collection membership window; the
    selected-event status path owns those questions.
    """
    surfaces = [question, *item.target.qualifier_surfaces,
                *item.scope.target_period_expressions]
    if item.scope.as_of_expression:
        surfaces.append(item.scope.as_of_expression)
    values: list[str] = []
    for surface in surfaces:
        carried_year: int | None = None
        for match in _OBSERVATION_DATE.finditer(surface):
            if match.group("year") is not None:
                raw_year = int(match.group("year"))
                carried_year = raw_year + 2000 if raw_year < 100 else raw_year
            if carried_year is None:
                continue
            try:
                stamp = f"{carried_year:04d}{int(match.group('month')):02d}{int(match.group('day')):02d}"
                date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))
            except ValueError:
                continue
            if stamp not in values:
                values.append(stamp)
    return tuple(values)


def _availability_contract_query(question: str, item: Any) -> bool:
    """Whether a disclosed-contract availability question needs a list source.

    The answer is a verdict, but it can only be produced after Stage2 reads
    the contract amount and whether that field was withheld.  This keeps that
    evidence collection in the native event path instead of letting a generic
    same-day document candidate choose a receipt.
    """
    return bool(_PUBLIC_AVAILABILITY.search(question) and _CONTRACT.search(question)
                and re.search(r"금액|대금|규모", item.target.surface))


_AMOUNT_CRITERION = re.compile(r"계약\s*금액|금액|대금|규모")


def _contract_amount_extremum_selection(item: Any) -> str | None:
    """Direction of a closed 「계약금액이 가장 크다/작다」 argmax/argmin request.

    이슈 #59 2단계 — HCX 실호출(SG-010/011)은 이 모양을 ``target.kind=event``,
    ``operation=retrieve``, ``selection={mode: maximum, criterion_surface:
    "계약금액", k: null}``, ``output.shape=scalar``,
    ``output.field_surfaces=["계약금액"]`` 로 낸다. 값은 보지 않는다 — 모양만
    검사한다. 이슈 #124 — 같은 모양의 ``selection.mode=minimum``(「가장 작은」)
    도 같은 자리에서 방향만 다르게 받는다. ``top_k``/``bottom_k`` 등 다른
    selection 모드는 여전히 지원 범위 밖이라 ``None`` 을 돌려주고 아래
    ``_list_item`` 이 그대로 닫는다.
    """
    selection = item.selection
    if not (
            selection is not None
            and selection.mode in ("maximum", "minimum")
            and selection.k is None
            and bool(_AMOUNT_CRITERION.search(selection.criterion_surface))
            and item.output.shape == "scalar"
            and item.output.projection_mode == "named_fields"
            and len(item.output.field_surfaces) == 1
            and bool(_AMOUNT_CRITERION.search(item.output.field_surfaces[0]))):
        return None
    return selection.mode


def _single_named_contract_timeline_request(
        intent: SemanticIntent, item: Any, question: str,
        ) -> bool:
    """Keep one named contract's full chronology out of collection routing.

    Korean ``모두`` may quantify the requested history rather than event
    identities.  A retrieve request for the initial signing and the complete
    change history is therefore a single-event timeline when its target carries
    a contract-specific term.  Explicit per-contract/multiple-contract wording
    still owns the collection interpretation, including provider outputs that
    flatten the answer shape to ``narrative``.
    """
    if (item.operation != "retrieve"
            or item.target.kind not in {"event", "document"}
            or item.output.shape in {"record_list", "comparison"}
            # The closed-literal pipeline intentionally keeps only the quoted
            # contract name in ``target.surface`` (for example ``블록, 기자재
            # 및 설계``).  The contract role is still a literal in the public
            # question, so do not require the normalized target to repeat the
            # generic noun ``계약``.
            or _CONTRACT.search(" ".join((question, item.target.surface))) is None
            or _MULTI_EVENT_IDENTITY.search(question) is not None):
        return False
    demands = " ".join((question, *item.output.field_surfaces))
    if (_INITIAL_SIGNING.search(demands) is None
            or _FULL_CHANGE_HISTORY.search(demands) is None):
        return False

    # Remove referenced entities before testing target specificity.  Otherwise
    # an issuer plus the generic noun ``계약`` would look like a named event.
    target = item.target.surface
    entities = {entity.entity_id: entity for entity in intent.entities}
    for ref in item.target.entity_refs:
        entity = entities.get(ref)
        if entity is not None:
            target = target.replace(entity.surface, " ")
    generic = {_key(value) for value in (
        *_STOP_TOKENS, "체결", "내용", "최초", "처음", "전체", "모든", "전부",
        "변경", "정정", "이력", "내역", "흐름", "설명",
    )}
    for token in re.findall(r"[A-Za-z가-힣]{2,}", target):
        stem = re.sub(r"(?:으로|에서|에게|의|을|를|은|는|이|가)$", "", token)
        if _key(stem) and _key(stem) not in generic:
            return True
    return False


def _list_item(intent: SemanticIntent, question: str) -> Any | None:
    if len(intent.answer_items) != 1:
        return None
    item = intent.answer_items[0]
    if item.operation not in {"retrieve", "compare"}:
        return None
    availability = _availability_contract_query(question, item)
    argmax = _contract_amount_extremum_selection(item) is not None
    if item.target.kind not in {"event", "document", "topic", "entity", "metric", "attribute"}:
        return None
    # 이슈 #59 2단계/#124 — 「계약금액이 가장 큰/작은 건」 argmax/argmin
    # selection 만 예외로 통과시킨다. 다른 selection(예: k>0)은 여전히
    # fail-closed.
    if item.selection is not None and not argmax:
        return None
    if _single_named_contract_timeline_request(intent, item, question):
        return None
    # HCX often uses ``narrative`` for both one-event questions and set
    # questions.  Only the typed list shape or an explicit plural surface may
    # create a collection; otherwise "최근 누구", "해지 이유" would silently
    # expand to every matching contract.
    if (item.output.shape != "record_list" and not availability and not argmax
            and _MANY.search(question) is None
            and _SET_EXISTENCE.search(question) is None):
        return None
    # An event list must be able to return more than one identity.  A scalar
    # request is intentionally left to the single-event resolver — except the
    # argmax selection above, which is deliberately a scalar reduction over a
    # canonically-proven multi-member collection.
    if (item.output.shape not in {"record_list", "comparison", "narrative", "timeline", "record"}
            and not availability and not argmax):
        return None
    return item


def _single_event_status_request(
        item: Any, counterparties: tuple[str, ...], question: str,
        ) -> bool:
    """Keep scalar/record state reads with the selected-event status resolver.

    A collection needs an explicit many-member signal.  A state field on a
    record is instead a request about one lifecycle identity, even when it
    also asks for a termination amount.  Explicit counterparties are a union
    only when two or more were actually named.
    """
    # HCX can flatten a set-membership answer to ``record`` even though the
    # Korean surface still explicitly asks which members of a set satisfy a
    # condition ("... 중 ... 확인되는 계약은 무엇인가").  Preserve that
    # collection signal; otherwise a requested ``해지 공시`` field is mistaken
    # for the state of one already-selected event.
    membership_question = bool(re.search(
        r"(?:중|가운데).*?(?:무엇|어떤|어느|확인되는|해당하는)", question))
    state_requested = any(re.search(r"상태|유효|살아|종료|끝난|해지", surface)
                          for surface in item.output.field_surfaces)
    return (state_requested and item.output.shape in {"record", "scalar"}
            and len(counterparties) < 2 and not membership_question
            and _SET_EXISTENCE.search(question) is None)


def _root_contract_termination_membership(question: str) -> bool:
    """Whether the date scopes original contracts later checked for termination."""

    return re.search(
        r"(?:공시한|체결(?:한|된)?)\s*.{0,50}?계약\s*.{0,30}?"
        r"(?:중|가운데)\s*.{0,40}?해지",
        question,
    ) is not None


def _requested_slots(
        item: Any, *, requires_termination: bool,
        root_contracts_with_confirmed_termination: bool,
        availability_query: bool,
        argmax: bool = False,
        ) -> tuple[str, ...]:
    if argmax:
        # 이슈 #59 2단계 — 「계약금액이 가장 큰 건」은 답을 계약명·계약상대·
        # 계약금액 세 slot 으로 낸다(접수번호는 field 조회가 아니라 선택된
        # 후보의 canonical receipt 에서 직접 나온다). deterministic_plan_
        # compiler_v1._event_collection_slots_from_intent 와 짝을 이룬다.
        return ("계약명", "계약상대", "계약금액")
    if availability_query:
        return ("계약금액", "공시유보여부")
    if root_contracts_with_confirmed_termination:
        return ("해지연결상태",)
    if not requires_termination:
        from .planning import normalize_slot_names
        return tuple(normalize_slot_names(item.output.field_surfaces))
    # A comparison of terminated contracts asks for the comparable termination
    # record even when the model collapsed its output field to the generic noun
    # ``계약``.  These are the canonical columns of that record, not a
    # question-specific fixture expansion.
    if (requires_termination and item.operation == "compare"
            and not any(pattern.search(surface)
                        for surface in item.output.field_surfaces
                        for pattern, _name in _SLOT_PATTERNS)):
        return ("상대방", "해지금액", "해지사유")
    slots: list[str] = []
    for surface in item.output.field_surfaces:
        matched = next((name for pattern, name in _SLOT_PATTERNS if pattern.search(surface)), None)
        if requires_termination and matched == "counterparty":
            matched = "상대방"
        elif requires_termination and matched == "amount":
            matched = "해지금액"
        elif requires_termination and matched == "reason":
            matched = "해지사유"
        elif matched is None:
            # Preserve a user-named field rather than invent a canonical name.
            matched = re.sub(r"\s+", "", surface)
        if matched not in slots:
            slots.append(matched)
    return tuple(slots)


def _issuer_from_intent(canonical: CanonicalEventCollectionLike, intent: SemanticIntent, item: Any) -> CanonicalCompanyLike | None:
    by_id = {entity.entity_id: entity for entity in intent.entities}
    surfaces = [
        by_id[ref].surface for ref in item.target.entity_refs
        if ref in by_id and by_id[ref].kind_hint == "company"
    ]
    if not surfaces:
        surfaces = [entity.surface for entity in intent.entities if entity.kind_hint == "company"]
    candidates: dict[str, CanonicalCompanyLike] = {}
    for surface in surfaces:
        rows = canonical.resolve_company(surface)
        if len(rows) == 1:
            candidates[rows[0].corp_code] = rows[0]
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _counterparty_surfaces(
        canonical: CanonicalEventCollectionLike, intent: SemanticIntent,
        item: Any, issuer: CanonicalCompanyLike | None,
        ) -> tuple[str, ...]:
    by_id = {entity.entity_id: entity for entity in intent.entities}
    values = [
        entity.surface for entity in intent.entities
        if entity.kind_hint == "counterparty"
    ]
    for ref in item.target.entity_refs:
        entity = by_id.get(ref)
        if entity is not None and entity.kind_hint == "company":
            resolved = canonical.resolve_company(entity.surface)
            is_issuer_alias = (
                issuer is not None and len(resolved) == 1
                and resolved[0].corp_code == issuer.corp_code)
            if not is_issuer_alias:
                values.append(entity.surface)
    return tuple(dict.fromkeys(value for value in values if value.strip()))


def _receipt_facts(rows: Iterable[Any]) -> dict[str, _ReceiptFacts]:
    raw: dict[str, dict[str, Any]] = {}
    for row in rows:
        receipt = str(getattr(row, "rcept_no", ""))
        stamp = str(getattr(row, "rcept_dt", ""))
        if not re.fullmatch(r"[0-9]{14}", receipt) or not re.fullmatch(r"[0-9]{8}", stamp):
            continue
        entry = raw.setdefault(receipt, {
            "date": stamp, "event_type": getattr(row, "event_type", None),
            "is_contract_form": False,
            "counterparties": set(), "contract_names": set(),
        })
        if entry["date"] != stamp:
            continue
        path = str(getattr(row, "path", ""))
        if _CONTRACT_FORM_PATH.search(path):
            entry["is_contract_form"] = True
        value = getattr(row, "value", None)
        if not isinstance(value, str) or not value.strip():
            value = getattr(row, "value_prompt_safe", None)
        if not isinstance(value, str) or not value.strip():
            continue
        if _COUNTERPARTY_PATH.search(path):
            entry["counterparties"].add(value.strip())
        if _CONTRACT_NAME_PATH.search(path):
            entry["contract_names"].add(value.strip())
    return {
        receipt: _ReceiptFacts(
            receipt=receipt, receipt_date=entry["date"],
            event_type=(str(entry["event_type"]) if entry["event_type"] else None),
            is_contract_form=bool(entry["is_contract_form"]),
            counterparties=tuple(sorted(entry["counterparties"])),
            contract_names=tuple(sorted(entry["contract_names"])),
        )
        for receipt, entry in raw.items()
    }


def _matches_terms(facts: _ReceiptFacts, *, requires_termination: bool,
                   requires_contract: bool, counterparties: tuple[str, ...]) -> bool:
    searchable = " ".join((facts.event_type or "", *facts.contract_names))
    # A value stored under the canonical contract-name role already proves
    # the document family even when the value itself is just a project title
    # (for example "공동주택 신축공사") and contains no literal 계약 noun.
    if (requires_contract and not facts.is_contract_form and not facts.contract_names
            and not _CONTRACT.search(searchable)):
        return False
    # A collection can explicitly name several counterparties.  Those names
    # select the union of their separate event identities; requiring all on a
    # single receipt would incorrectly reject the very list being requested.
    if counterparties and not any(_contains_any(value, list(facts.counterparties))
                                  for value in counterparties):
        return False
    # Termination is a verified timeline observation, not a report-name
    # heuristic.  A changed/rectified disclosure title must not make the
    # issuer discovery or its later evidence check silently lose the event.
    return True


@dataclass(frozen=True, slots=True)
class _ResolvedCompany:
    corp_code: str
    corp_name: str


def _infer_issuer_from_counterparties(
        canonical: CanonicalEventCollectionLike, *, as_of: str,
        counterparties: tuple[str, ...],
        ) -> CanonicalCompanyLike | None:
    """Infer only when every named counterparty proves the same one issuer."""

    if not counterparties:
        return None
    by_corp: dict[tuple[str, str], set[str]] = {}
    for row in canonical.fields(as_of=as_of, label="계약상대"):
        path = str(getattr(row, "path", ""))
        value = getattr(row, "value", None)
        if not _COUNTERPARTY_PATH.search(path) or not isinstance(value, str):
            continue
        if not any(_contains_any(surface, [value]) for surface in counterparties):
            continue
        code = str(getattr(row, "corp_code", ""))
        name = str(getattr(row, "corp_name", ""))
        if re.fullmatch(r"[0-9]{8}", code) and name.strip():
            by_corp.setdefault((code, name), set()).add(_key(value))
    candidates = [
        (code, name) for (code, name), values in by_corp.items()
        if all(any(_key(surface) in value or value in _key(surface)
                   for value in values)
               for surface in counterparties)
    ]
    if len(candidates) != 1:
        return None
    code, name = candidates[0]
    return _ResolvedCompany(code, name)


def _selector_keywords(item: Any, events: tuple[EventCollectionIdentity, ...]) -> tuple[str, ...]:
    """Preserve only target terms actually evidenced by selected contracts."""
    haystack = " ".join(name for event in events for name in event.contract_names)
    values: list[str] = []
    for token in re.findall(r"[A-Za-z가-힣]{2,}", item.target.surface):
        compact = re.sub(r"\s+", "", token)
        if compact in _STOP_TOKENS or not compact:
            continue
        if _key(compact) in _key(haystack) and compact not in values:
            values.append(compact)
    if values or len(events) < 2:
        return tuple(values)

    # When a question names multiple counterparties but omits the shared
    # product noun, retain a keyword only if every selected canonical contract
    # independently contains the same single domain token.  This narrows the
    # Stage2 list without inventing a user-specific alias or consulting Gold.
    common: set[str] | None = None
    for event in events:
        tokens: set[str] = set()
        for name in event.contract_names:
            content = [
                token for token in re.findall(r"[A-Za-z가-힣]{2,}", name)
                if token not in _STOP_TOKENS
            ]
            # Korean contract titles are normally modifier -> product head ->
            # ``공급계약``.  The rightmost surviving content noun is therefore
            # the stable product facet; earlier nouns are application modifiers.
            if content:
                tokens.add(content[-1])
        common = tokens if common is None else common & tokens
    if common is not None and len(common) == 1:
        return (next(iter(common)),)
    return ()


def select_event_collection(
        *, canonical: CanonicalEventCollectionLike, intent: SemanticIntent,
        question: str, reference_date: date, corpus_cutoff: str,
        ) -> EventCollectionSelection | None:
    """Select a canonical event set or return ``None`` on any ambiguity.

    ``None`` means the caller should use its normal clarification/fallback
    policy.  A scoped empty set is returned only for a closed membership query
    whose root-candidate scan is complete and carries a collection proof.
    """

    if not re.fullmatch(r"[0-9]{8}", corpus_cutoff) or not question.strip():
        raise ValueError("event collection selector 입력 형식이 잘못되었습니다")
    item = _list_item(intent, question)
    if item is None:
        return None
    extremum_direction = _contract_amount_extremum_selection(item)
    argmax = extremum_direction is not None
    # A correction lineage is one event's chronology, not a membership list.
    # Let the lifecycle resolver preserve before/after proof coordinates.
    if _CORRECTION_FLOW.search(question):
        return None
    if len(_explicit_observation_dates(question, item)) >= 2:
        return None
    as_of = _as_of(question, corpus_cutoff, reference_date)
    if as_of is None:
        return None
    event_from, event_to = _date_range(question, reference_date)
    # A date in an explicit "until disclosed" clause has already been
    # consumed as visibility, not event occurrence scope.
    if _AS_OF.search(question):
        event_from = event_to = None
    requires_termination = bool(_TERMINATION.search(question))
    root_contracts_with_confirmed_termination = bool(
        requires_termination and _root_contract_termination_membership(question))
    requires_contract = bool(_CONTRACT.search(question))
    availability_query = _availability_contract_query(question, item)
    issuer = _issuer_from_intent(canonical, intent, item)
    counterparties = _counterparty_surfaces(canonical, intent, item, issuer)
    if _single_event_status_request(item, counterparties, question):
        return None
    # A generic topic/narrative (notably investment questions) may have the
    # same broad output shape.  It is not an event collection until the
    # question supplies an event-family cue or a concrete counterparty anchor.
    if not (requires_contract or requires_termination or counterparties):
        return None
    if issuer is None:
        issuer = _infer_issuer_from_counterparties(
            canonical, as_of=as_of, counterparties=counterparties)
    if issuer is None:
        return None

    try:
        family_form = (
            _CONTRACT_EVENT_TYPE
            if root_contracts_with_confirmed_termination or not requires_termination
            else _TERMINATION_EVENT_TYPE
        ) if requires_contract or requires_termination else None
        facts_by_receipt = _receipt_facts(canonical.fields(
            as_of=as_of, corp_code=issuer.corp_code, form=family_form))
    except (AttributeError, TypeError, ValueError):
        return None
    selected: dict[str, dict[str, Any]] = {}
    scanned_root_candidates = 0
    for facts in facts_by_receipt.values():
        if not _matches_terms(facts, requires_termination=requires_termination,
                              requires_contract=requires_contract,
                              counterparties=counterparties):
            continue
        try:
            timeline = canonical.event_timeline(
                as_of=as_of, rcept_no=facts.receipt,
                verify_evidence=not root_contracts_with_confirmed_termination)
        except (AttributeError, TypeError, ValueError):
            return None
        if (timeline is None or timeline.corp_code != issuer.corp_code
                or not re.fullmatch(r"[0-9a-f]{32}", timeline.event_key)
                or not re.fullmatch(r"[0-9]{14}", timeline.root_rcept_no)):
            return None
        observations = tuple(timeline.observations)
        matching_observations = [
            observation for observation in observations
            if observation.rcept_no == facts.receipt
        ]
        if len(matching_observations) != 1:
            return None
        observation = matching_observations[0]
        if root_contracts_with_confirmed_termination:
            # The year/quarter scopes original contracts.  Evaluate the
            # termination predicate over their complete verified timelines;
            # do not first select termination receipts and then swap dates.
            if facts.receipt != timeline.root_rcept_no:
                continue
            if (event_from is not None
                    and (facts.receipt_date < event_from
                         or facts.receipt_date > event_to)):
                continue
            scanned_root_candidates += 1
            termination_observations = [
                row for row in observations
                if bool(getattr(row, "is_termination", False))
                and str(getattr(row, "rcept_no", ""))[:8] <= as_of
            ]
            if not termination_observations:
                continue
            # Membership can be rejected from the lightweight verified
            # identity timeline.  Only positive members pay Field/Evidence
            # ownership verification before their proofs are exposed.
            try:
                verified_timeline = canonical.event_timeline(
                    as_of=as_of, rcept_no=facts.receipt,
                    verify_evidence=True)
            except (AttributeError, TypeError, ValueError):
                return None
            if verified_timeline is None:
                return None
            timeline = verified_timeline
            observations = tuple(timeline.observations)
            termination_observations = [
                row for row in observations
                if bool(getattr(row, "is_termination", False))
                and str(getattr(row, "rcept_no", ""))[:8] <= as_of
            ]
            bucket = selected.setdefault(timeline.event_key, {
                "root": timeline.root_rcept_no,
                "receipts": [], "dates": [], "types": set(),
                "selector_types": {_CONTRACT_EVENT_TYPE},
                "counterparties": set(facts.counterparties),
                "contracts": set(facts.contract_names), "proofs": set(),
            })
            if bucket["root"] != timeline.root_rcept_no:
                return None
            for termination in termination_observations:
                receipt = str(termination.rcept_no)
                bucket["receipts"].append(receipt)
                bucket["dates"].append(receipt[:8])
                bucket["proofs"].add(
                    f"canonical:event:{timeline.event_key}:{receipt}")
                bucket["proofs"].update(
                    f"canonical:event-support:{proof}"
                    for proof in getattr(
                        termination, "supporting_evidence_ids", ()))
            if facts.event_type:
                bucket["types"].add(facts.event_type)
            continue
        if (requires_termination
                and not bool(getattr(observation, "is_termination", False))):
            continue
        # For lifecycle questions such as "2024년 contracts with a confirmed
        # termination", date scopes the root contract; plain "December
        # terminated" scopes the termination observation.
        date_to_check = (
            timeline.root_rcept_no if root_contracts_with_confirmed_termination
            else facts.receipt)
        date_value = (facts_by_receipt.get(date_to_check).receipt_date
                      if date_to_check in facts_by_receipt else facts.receipt_date)
        if event_from is not None and (date_value < event_from or date_value > event_to):
            continue
        bucket = selected.setdefault(timeline.event_key, {
            "root": timeline.root_rcept_no, "receipts": [], "dates": [],
            "types": set(), "selector_types": set(), "counterparties": set(), "contracts": set(), "proofs": set(),
        })
        if bucket["root"] != timeline.root_rcept_no:
            return None
        bucket["receipts"].append(facts.receipt)
        bucket["dates"].append(facts.receipt_date)
        if facts.event_type:
            bucket["types"].add(facts.event_type)
        root_facts = facts_by_receipt.get(timeline.root_rcept_no)
        selector_type = (
            _CONTRACT_EVENT_TYPE if root_contracts_with_confirmed_termination
            else _TERMINATION_EVENT_TYPE if requires_termination
            else (root_facts.event_type if root_facts is not None else facts.event_type))
        if selector_type:
            bucket["selector_types"].add(selector_type)
        bucket["counterparties"].update(facts.counterparties)
        bucket["contracts"].update(facts.contract_names)
        bucket["proofs"].add(f"canonical:event:{timeline.event_key}:{facts.receipt}")
        bucket["proofs"].update(
            f"canonical:event-support:{proof}"
            for proof in getattr(observation, "supporting_evidence_ids", ())
        )
    collection_proof_ref = None
    if not selected:
        closed_empty_allowed = bool(
            root_contracts_with_confirmed_termination
            and scanned_root_candidates
            and ("코퍼스" in question or _SET_EXISTENCE.search(question)))
        if not closed_empty_allowed:
            return None
        collection_proof_ref = (
            "canonical:event-scan:"
            f"{issuer.corp_code}:{_CONTRACT_EVENT_TYPE}:"
            f"{event_from or '-'}:{event_to or '-'}:{as_of}:"
            "termination-confirmed:none"
        )
    events = tuple(
        EventCollectionIdentity(
            event_key=key, root_receipt=bucket["root"],
            matched_receipts=tuple(sorted(set(bucket["receipts"]))),
            matched_dates=tuple(sorted(set(bucket["dates"]))),
            event_types=tuple(sorted(bucket["types"])),
            counterparties=tuple(sorted(bucket["counterparties"])),
            contract_names=tuple(sorted(bucket["contracts"])),
            proof_refs=tuple(sorted(bucket["proofs"])),
        )
        for key, bucket in sorted(selected.items())
    )
    proofs = (
        (collection_proof_ref,) if collection_proof_ref is not None
        else tuple(sorted({
            proof for event in events for proof in event.proof_refs}))
    )
    selector_types = {
        selector_type for bucket in selected.values()
        for selector_type in bucket["selector_types"]
    }
    if collection_proof_ref is not None:
        selector_types = {_CONTRACT_EVENT_TYPE}
    if len(selector_types) > 1:
        return None
    # A counterparty-union list with no textual month still has a deterministic
    # month selector when the selected *canonical observations* agree on it.
    if event_from is None and len(counterparties) >= 2:
        months = {stamp[:6] for event in events for stamp in event.matched_dates}
        if len(months) == 1:
            year, month = int(next(iter(months))[:4]), int(next(iter(months))[4:])
            event_from = f"{year}{month:02d}01"
            event_to = f"{year}{month:02d}{calendar.monthrange(year, month)[1]:02d}"
    broad_contract_table = bool(
        item.output.shape == "record_list"
        and _MANY.search(question) is not None
        and not any(
            token not in _STOP_TOKENS
            for token in re.findall(r"[A-Za-z가-힣]{2,}", item.target.surface)
        )
    )
    selector_keywords = (
        () if broad_contract_table else _selector_keywords(item, events))
    if not selector_keywords and requires_contract:
        # A broad collection still needs a public Stage2 selector.  Retain the
        # exact category noun from the question/target; member identities have
        # already been proven by canonical timelines above.
        category = next((value for value in ("공급계약", "판매계약", "계약")
                         if value in question or value in item.target.surface), None)
        if category is not None:
            selector_keywords = (category,)
    if availability_query and "공급계약" not in selector_keywords:
        selector_keywords = (*selector_keywords, "공급계약")
    if not any((selector_types, selector_keywords, event_from, event_to)):
        return None
    return EventCollectionSelection(
        issuer_corp_code=issuer.corp_code, issuer_corp_name=issuer.corp_name,
        as_of=as_of, event_from=event_from, event_to=event_to,
        requires_termination=requires_termination,
        root_contracts_with_confirmed_termination=(
            root_contracts_with_confirmed_termination),
        selector_event_type=(next(iter(selector_types)) if selector_types else None),
        selector_counterparty=(
            counterparties[0] if availability_query and len(counterparties) == 1
            else None),
        selector_keywords=selector_keywords,
        public_task_kind=("disclosure" if availability_query else "event"),
        availability_query=availability_query,
        requested_slots=_requested_slots(
            item, requires_termination=requires_termination,
            root_contracts_with_confirmed_termination=(
                root_contracts_with_confirmed_termination),
            availability_query=availability_query, argmax=argmax),
        events=events, proof_refs=proofs,
        collection_proof_ref=collection_proof_ref,
        argmax_slot=("계약금액" if argmax else None),
        argmax_direction=(extremum_direction or "maximum"),
    )


def _qualifier_surface_span(question: str, surfaces: list[str]) -> str | None:
    """``surfaces`` 가 순서대로, 사이에 공백만 두고 이루는 질문 안 연속 구간.

    `agent.disclosed_metric_topics.coordinated_surface_span` 과 같은 모양이지만
    그쪽은 이음말(「과」·「와」·「및」…)로 이어진 두 표기만 받는다. 여기서는
    질문이 그냥 띄어 쓴 여러 표기(「공시한」+「단일판매·공급계약」)를 다뤄야
    하므로 사이 공백만 허용한다. 표기가 하나뿐이면 질문에 그대로 있는지만
    본다. 순서가 어긋나거나 사이에 다른 낱말이 있으면 뜻을 지어내는 것이므로
    ``None`` — 안전하게 손을 뗀다.
    """

    if not surfaces:
        return None
    cursor = 0
    spans: list[tuple[int, int]] = []
    for surface in surfaces:
        start = question.find(surface, cursor)
        if start < 0:
            return None
        spans.append((start, start + len(surface)))
        cursor = start + len(surface)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        if question[end:start].strip(" ") != "":
            return None
    return question[spans[0][0]:spans[-1][1]]


class EventArgmaxAmountFieldTargetRegrounder:
    """SG-011 — 사건의 금액 필드를 지표로 읽어 흩어진 argmax wire 를 되돌린다.

    2026-09-04 실호출 감사(``out/logs/stage1_unsupported.jsonl`` 02:49·06:13,
    오늘만 세 번)에서 확인한 모양 — 「LG에너지솔루션이 2025년에 공시한
    단일판매·공급계약 중 계약금액이 가장 큰 건은 얼마인가?」에서 HCX 가 사건의
    금액 필드(「계약금액」) 자체를 ``target.kind="metric"`` 으로 읽고,
    최상급 표현은 ``selection.criterion_surface``(「가장 큰」)에, 무엇을
    답할지는 ``output.field_surfaces``(「얼마」)에 담아 표면을 셋으로
    흩어 버린다::

        target={kind: "metric", surface: "계약금액",
                qualifier_surfaces: ["공시한", "단일판매·공급계약"]}
        scope={target_period_expressions: ["2025년"],
               document_group_expression: "단일판매·공급계약"}
        selection={mode: "maximum", criterion_surface: "가장 큰", k: None}
        output={shape: "scalar", field_surfaces: ["얼마"]}

    ``_contract_amount_extremum_selection``(이슈 #59 2단계/#124)이 이미
    통과시키는 event argmax 모양(SG-010, ``test_stage1_v1_event_collection_
    argmax.py``)은 값은 같고 표면이 있는 자리만 다르다 —
    ``target.kind="event"``, ``target.surface`` 는 기간을 뺀 나머지
    qualifier(「공시한」+「단일판매·공급계약」)가 질문에서 이루는 구간,
    ``selection.criterion_surface`` 와 ``output.field_surfaces[0]`` 는 원래
    metric 표면(「계약금액」)이다.

    이 regrounder는 그 세 자리로 표면을 옮기기만 한다 — 값·회사·기간을 새로
    고르지 않는다. 옮겨 붙일 구간이 질문에 그대로 없으면 손을 뗀다.
    """

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str) or not question.strip()
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        target = item.target
        selection = item.selection
        document_group = item.scope.document_group_expression
        if (
                item.operation != "retrieve"
                or target.kind != "metric"
                or not _AMOUNT_CRITERION.search(target.surface)
                or not document_group
                or not _CONTRACT.search(document_group)
                or selection is None
                or selection.mode not in ("maximum", "minimum")
                or selection.k is not None
                # 이미 금액 표면을 담고 있으면(예: 이미 옮겨진 shape) 손을
                # 뗀다 — 다른 regrounder/원래 모양과 겹치지 않는다.
                or _AMOUNT_CRITERION.search(selection.criterion_surface)
                or item.output.shape != "scalar"
                or item.output.projection_mode != "named_fields"
                or len(item.output.field_surfaces) != 1
        ):
            return intent
        period_keys = {
            re.sub(r"\s+", "", value)
            for value in item.scope.target_period_expressions}
        remaining = [
            surface for surface in target.qualifier_surfaces
            if re.sub(r"\s+", "", surface) not in period_keys]
        new_surface = (
            _qualifier_surface_span(question, remaining) if remaining else None)
        if new_surface is None and document_group in question:
            new_surface = document_group
        if new_surface is None or not _CONTRACT.search(new_surface):
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        item_payload = payload["answer_items"][0]
        item_payload["target"]["kind"] = "event"
        item_payload["target"]["surface"] = new_surface
        item_payload["target"]["qualifier_surfaces"] = []
        item_payload["selection"]["criterion_surface"] = target.surface
        item_payload["scope"]["document_group_expression"] = None
        item_payload["output"]["field_surfaces"] = [target.surface]
        return SemanticIntent.model_validate(payload, strict=True)


#: A field clause that asks which disclosed value-state a `-` cell is in
#: ("0원인지 비공개인지", "0인지 공란인지" …).  HCX can flatten the closed
#: question's trailing "구분해줘"/"판단해줘" verb out of the field surface
#: itself (RPC-007 실호출: field_surfaces=["계약금액이 `-`로 표시된 것은
#: 0원인지 비공개인지"], the verb stays only in the raw question), so this
#: only requires the value-state enumeration itself, not the verb.
_VALUE_STATE_FIELD_CLAUSE = re.compile(
    r"(?:0원|공란|미공시|비공개|유보).{0,6}(?:인지|아닌지)"
)


class ContractAmountAvailabilityFieldRegrounder:
    """Restore the availability shape from a flattened narrative clause.

    RPC-007 실호출(2026-09-02): HCX는 「N년 M월 D일까지 공개된 A 의 B
    배터리 공급계약 공시에서 계약금액이 `-`로 표시된 것은 0원인지
    비공개인지 구분해줘」를 ``target.kind=document``,
    ``output.shape=narrative``, ``projection_mode=named_fields``,
    ``field_surfaces=["계약금액이 `-`로 표시된 것은 0원인지 비공개인지"]``로
    내고, 기간은 ``target.surface`` 안에만 남기며 ``scope.as_of_expression``
    은 비운다. `_availability_contract_query`(→ EventCollectionResolutionBackend)
    는 ``item.target.surface``에 "금액"이 있고 shape가 이 narrative 모양이
    아니어야 걸리므로, 이 모양은 그 게이트에 닿지 못하고 event candidate
    scan(상대방·as-of 미적용)으로 떨어져 13건을 나열했다.

    이 regrounder는 값·회사·상대방을 하나도 새로 고르지 않는다. 이미
    검증된 두 항목(발행사로 확정되는 회사 entity, 그 회사가 아닌 나머지
    entity의 문면)만 이어붙여 ``target.surface``를 "상대방…계약금액"의
    literal span으로 좁히고, 기간 표현을 as-of 축으로 옮기며,
    field_surfaces를 ``["계약금액"]`` 하나로 좁힌다 — 전부 질문에 이미 있는
    글자만 재배치한 것이라 `validate_semantic_intent_grounding`을 그대로
    통과한다.
    """

    def __init__(self, canonical: CanonicalEventCollectionLike) -> None:
        if not callable(getattr(canonical, "resolve_company", None)):
            raise TypeError("regrounder에는 company resolver가 필요합니다")
        self.canonical = canonical

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str) or not question.strip()
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        if (item.target.kind != "document"
                or item.operation != "retrieve"
                or item.output.projection_mode != "named_fields"
                or item.output.shape not in {"narrative", "record", "verdict"}
                or len(item.output.field_surfaces) != 1
                or _VALUE_STATE_FIELD_CLAUSE.search(
                    item.output.field_surfaces[0]) is None
                or "계약금액" not in item.output.field_surfaces[0]
                or item.scope.as_of_expression is not None
                or len(item.scope.target_period_expressions) != 1
                or not item.scope.target_period_expressions[0].rstrip().endswith("까지")):
            return intent
        as_of_surface = item.scope.target_period_expressions[0]
        by_id = {entity.entity_id: entity for entity in intent.entities}
        referenced = [by_id[ref] for ref in item.target.entity_refs if ref in by_id]
        if len(referenced) < 2:
            return intent
        issuers = [
            entity for entity in referenced
            if len(self.canonical.resolve_company(entity.surface) or ()) == 1
        ]
        if len(issuers) != 1:
            return intent
        issuer = issuers[0]
        counterparties = [
            entity for entity in referenced
            if entity.entity_id != issuer.entity_id
            and not self.canonical.resolve_company(entity.surface)
        ]
        if len(counterparties) != 1:
            return intent
        counterparty = counterparties[0]
        span_match = re.search(
            re.escape(counterparty.surface) + r"[^?？。\n]{0,60}?계약금액",
            question)
        if span_match is None:
            return intent
        new_target_surface = span_match.group(0)

        payload = intent.model_dump(mode="python", warnings=False)
        target = payload["answer_items"][0]["target"]
        target.update({
            "surface": new_target_surface,
            "entity_refs": [issuer.entity_id, counterparty.entity_id],
        })
        payload["answer_items"][0]["scope"].update({
            "target_period_expressions": [],
            "as_of_expression": as_of_surface,
        })
        payload["answer_items"][0]["output"].update({
            "shape": "verdict",
            "projection_mode": "named_fields",
            "field_surfaces": ["계약금액"],
        })
        payload["answer_groups"] = []
        payload["premises"] = []
        payload["unresolved_mentions"] = []
        return SemanticIntent.model_validate(payload, strict=True)


class EventSingleFieldNarrativeShapeRegrounder:
    """CG-011 자매 케이스(#132) — 사건 단일 필드 질문의 narrative 모양을 되돌린다.

    실호출(2026-09-04, ``out/logs/stage1_unsupported.jsonl``, Q132-1)
    「현대자동차가 2024년 10월 24일 공시한 해외증권시장 주권 상장폐지 결정의
    사유를 알려줘」는 이미 ``target.kind="event"``·
    ``target.surface="해외증권시장 주권 상장폐지 결정"`` 으로 사건까지는
    올바르게 읽으면서도, 필드가 **하나**뿐이면 ``output.shape="narrative"``
    로 내고 그 날짜를 ``target.qualifier_surfaces`` 대신
    ``scope.target_period_expressions`` 에 둔다::

        target={kind: "event", surface: "해외증권시장 주권 상장폐지 결정",
                qualifier_surfaces: []}
        scope={target_period_expressions: ["2024년 10월 24일"], ...}
        output={shape: "narrative", field_surfaces: ["사유"]}

    같은 서식에 두 필드(「상장폐지 사유」·「해당 시장」)를 물으면(#121 CG-011,
    ``test_delisting_evidence_grounding_canonical.py``) HCX 는 날짜를
    ``target.qualifier_surfaces`` 에 두고 ``output.shape="record"`` 로
    낸다 — ``SelectedEventResolutionBackend.resolve`` 가 실제로 받는 모양은
    이쪽이다. 이 regrounder는 필드가 하나뿐인 wire 를 그 자리(날짜 축·shape)
    로만 옮긴다 — 회사·필드 표면·값은 어떤 것도 새로 고르지 않는다. 날짜가
    이미 ``target.qualifier_surfaces`` 에 있거나 아예 없는 wire는 옮길 것이
    없으므로 손대지 않는다.

    (기존 「사업 내용」류 narrative 질문은 ``target.kind`` 가 "document"/
    "topic" 이라 이 규칙에 걸리지 않는다.)
    """

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str) or not question.strip()
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        target = item.target
        scope = item.scope
        output = item.output
        if (
                target.kind != "event"
                or item.operation != "retrieve"
                or output.shape != "narrative"
                or output.projection_mode != "named_fields"
                or len(output.field_surfaces) != 1
                or output.field_surfaces[0] not in question
                or item.selection is not None
                or scope.as_of_expression is not None
                or scope.document_group_expression is not None
                or scope.scope_qualifier_expressions
        ):
            return intent
        date_axes = [*target.qualifier_surfaces, *scope.target_period_expressions]
        # This adapter exists only to move one literal event date from the
        # generic period axis back to the event qualifier axis.  With no date
        # there is nothing to move: changing only narrative -> record would
        # bypass the missing-event-context clarification and turn a useful
        # question such as "계약 해지된 거 왜 그런거야?" into an unsupported
        # field lookup (issue #196).
        if len(date_axes) != 1 or date_axes[0] not in question:
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        item_payload = payload["answer_items"][0]
        item_payload["target"]["qualifier_surfaces"] = date_axes
        item_payload["scope"]["target_period_expressions"] = []
        item_payload["output"]["shape"] = "record"
        return SemanticIntent.model_validate(payload, strict=True)


class TerminationExistenceIntentRegrounder:
    """Restore an open-ended contract-termination existence verdict.

    SG-004·005 실호출(이슈 #74, 2026-09-03): "LG에너지솔루션이 체결한
    단일판매·공급계약 중 이후 해지된 계약이 존재하는가?"·"삼성전자가 2025년에
    체결한 단일판매·공급계약 중 이후 해지된 계약이 존재하는가?" 처럼, 계약명을
    지정하지 않는 개방형 존재 질의를 HCX는 ``target.kind=event``,
    ``output.shape=verdict``, ``projection_mode=whole_target``,
    ``field_surfaces=[]`` 로 낸다.  같은 질문(SG-006, 2026-09-03 후속
    실호출)을 다른 호출에서는 ``output.shape=scalar``,
    ``projection_mode=named_fields``, ``field_surfaces=["존재하는가"]`` 처럼
    field가 하나 실린 모양으로도 낸다 — 낱말은 남지만 존재 판정에 쓰이지
    않는 표면이다.  ``select_event_collection``의 root-contract-termination
    경로(``_root_contract_termination_membership``·``_SET_EXISTENCE``)는
    이미 이 질문을 한 field 슬롯(``해지연결상태``, 존재하면 접속된 해지
    접수번호를, 없으면 코퍼스 스캔 완료 증거를 낸다)으로 답할 수 있다.
    그러나 ``whole_target`` projection에는 field가 없어
    ``_compile_event_collection``이 field/slot 카디널리티(0 대 1)를 맞추지
    못하고, named_fields 모양이어도 field 표면이 "해지연결상태"가 아니면
    같은 카디널리티가 어긋나 둘 다 컴파일이 실패해 unsupported_request 로
    닫혔다.

    이 regrounder는 회사·기간·계약 종류 등 어떤 값도 새로 고르지 않는다 —
    field 표면이 있었든 없었든 질문에 이미 등장하는 해지 어휘 한 글자(예
    "해지")로 덮어써 ``named_fields`` projection 하나로 통일할 뿐이다.
    나머지(발행사 결속·코퍼스 스캔·존재/부재 판정)는 전부
    ``select_event_collection``의 root-contract-termination 경로가 진행한다.
    """

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str) or not question.strip()
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        whole_target_shape = (
            item.output.shape == "verdict"
            and item.output.projection_mode == "whole_target"
            and not item.output.field_surfaces)
        named_field_shape = (
            item.output.shape in {"verdict", "scalar"}
            and item.output.projection_mode == "named_fields"
            and len(item.output.field_surfaces) == 1)
        if (item.operation not in {"retrieve", "compare"}
                or not (whole_target_shape or named_field_shape)
                or item.selection is not None
                or not _root_contract_termination_membership(question)
                or _SET_EXISTENCE.search(question) is None):
            return intent
        termination_match = _TERMINATION.search(question)
        if termination_match is None:
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        payload["answer_items"][0]["target"]["kind"] = "event"
        payload["answer_items"][0]["output"].update({
            "shape": "record",
            "projection_mode": "named_fields",
            "field_surfaces": [termination_match.group(0)],
        })
        return SemanticIntent.model_validate(payload, strict=True)


class EventRecordDateQualifierRegrounder:
    """Restore a single exact event date onto its ``target`` axis.

    실호출(2026-09-04, ``out/logs/stage1_unsupported.jsonl``, 이슈 #151
    CG-005·CG-025) — 소송등의제기·제3자의전환사채매수선택권행사처럼 사건
    하나·필드 여럿(record shape)을 record로 읽는 닫힌 단일-item 질문에서
    HCX는 사건 발생일을 ``target.qualifier_surfaces``에 올바르게 낸다
    (예: CG-005 ``target={kind:event, surface:"소송 등의 제기",
    qualifier_surfaces:["2026년 1월 8일"]}``). 그런데 이 규정집 앞쪽의
    ``MisplacedPeriodQualifierRegrounder``는(이 두 질문과 무관한 다른
    wire 모양을 고치려고 있는 일반 규칙) "질문에 등장하는 날짜 표면 하나가
    ``target.qualifier_surfaces``에 있으면 ``scope.target_period_expressions``
    로 옮긴다"는 조건만 보므로 이 record 사건도 함께 옮겨 버린다.

    ``SelectedEventResolutionBackend.resolve``의 단일사건 record 읽기
    (``event_field_read``)는 ``scope.target_period_expressions``가 비어
    있어야 확정하므로, 옮겨진 날짜가 그 조건을 깨뜨려 사건이 정본에
    있는데도 ``unsupported_semantic_target``으로 닫힌다 — #132가 고친
    narrative 단일 필드 자매 버그와 같은 축 문제를, 이번엔 record/scalar
    모양에서 다시 만난다.

    이 regrounder는 그 축만 되돌린다 — 회사·필드 표면·값은 어떤 것도 새로
    고르지 않는다. ``MisplacedPeriodQualifierRegrounder`` 바로 다음 자리에
    두어 그 규칙이 옮긴 날짜만 되돌리고, narrative 모양(#132)은 이미
    ``EventSingleFieldNarrativeShapeRegrounder``가 맡으므로 손대지 않는다.
    단일 answer_item·비교/선택 없는 닫힌 질문에만 적용해, 사건 컬렉션의
    범위 필터로 쓰이는 다른 ``scope.target_period_expressions``(연도만·
    범위 등)는 건드리지 않는다.
    """

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str) or not question.strip()
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        target = item.target
        scope = item.scope
        output = item.output
        if (
                target.kind != "event"
                or item.operation != "retrieve"
                or output.shape not in {"record", "scalar"}
                or output.projection_mode != "named_fields"
                or not output.field_surfaces
                or any(surface not in question
                       for surface in output.field_surfaces)
                or item.selection is not None
                or target.qualifier_surfaces
                or scope.as_of_expression is not None
                or scope.document_group_expression is not None
                or scope.scope_qualifier_expressions
                or len(scope.target_period_expressions) != 1
        ):
            return intent
        value = scope.target_period_expressions[0]
        if value not in question:
            return intent
        year, month, day = parse_date_surface(value) or (None, None, None)
        if year is None or month is None or day is None:
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        item_payload = payload["answer_items"][0]
        item_payload["target"]["qualifier_surfaces"] = [value]
        item_payload["scope"]["target_period_expressions"] = []
        return SemanticIntent.model_validate(payload, strict=True)


#: A qualifier/entity surface that spells out both the disclosure day and
#: the form name in one literal span (``"2025년 12월 19일 공시한 유형자산
#: 양수결정"``). Splitting on the connective recovers each half without
#: inventing any character not already in the question.
_DATE_FORM_SURFACE = re.compile(r"^(?P<date>.+?)\s*공시한\s*(?P<form>.+)$")


class EventTwoFieldItemMergeRegrounder:
    """CG-008·009·010·019·022(#156) — 사건 두 필드 질문의 쪼개진 answer_items를 되돌린다.

    실호출(2026-09-04, ``out/logs/stage1_unsupported.jsonl``) — 한 공시의
    필드 두 개를 묻는 닫힌 질문(예 CG-008 "양수 자산의 소재지와 자산총액
    대비 비율을 알려줘", CG-009 "양수 금액과 양수 상대방을 알려줘")을 HCX가
    ``answer_items`` 두 개(kind=attribute/metric/entity 제각각)로 쪼갠다.
    사건 백엔드(``SelectedEventResolutionBackend.resolve``의
    ``event_field_read`` 경로)는 **단일** event 항목 +
    ``output.shape=record``(#151 CG-011이 이미 통과시키는 모양)만 받으므로,
    쪼개진 두 항목은 그 이유만으로 ``unsupported_semantic_target``으로
    닫힌다.

    관측된 다섯 모양 전부에서, 두 항목이 가리키는 공시 자체는
    ``intent.entities`` 안의 한 ``kind_hint=event`` entity 표면
    (``"YYYY년 M월 D일 공시한 서식명"``)으로 공통 식별된다 — 두 항목은 이
    entity를 ``target.entity_refs``로 직접 참조하거나(CG-009·CG-010의
    사건쪽 항목), ``target.qualifier_surfaces``에 그 표면을 글자 그대로
    되풀이한다(CG-008·019·022). 이 regrounder는 그 표면을 "공시한" 앞뒤로
    나눠 — 자르기만 한다, 새 낱말은 없다 — ``target.kind=event``·
    ``surface=서식``·``scope.target_period_expressions=[날짜]``인 사건
    항목 하나로 두 항목을 합친다. ``field_surfaces``는 두 항목의 field
    표면을 질문에 등장한 순서 그대로 합집합한다.

    날짜는 일부러 ``target.qualifier_surfaces``가 아니라
    ``scope.target_period_expressions``에 둔다 — 바로 뒤에 오는
    ``EventRecordDateQualifierRegrounder``(#151)가 그 축을
    ``event_field_read``가 요구하는 자리로 그대로 옮겨 주므로, 이
    regrounder는 축 이동 규칙을 중복하지 않는다. 체인에서는 반드시 그
    규칙 앞에 둔다.

    회사는 두 entity 중 사건 표면이 아닌 나머지 하나를 그대로 쓴다.
    CG-009처럼 회사 entity 자체가 통째로 빠진 wire는 ``company_preflight``
    (다른 regrounder들과 같은, 질문 전체를 canonical 회사 레지스트리로
    스캔하는 헬퍼)로 유일하게 확정될 때만 복구한다 — 새 표면을 짓지 않고
    질문에 이미 있는 회사명 글자만 그대로 쓴다.

    조건이 하나라도 안 맞으면 항목 두 개를 그대로 둔다.
    """

    def __init__(self, *, company_preflight: Any) -> None:
        self.company_preflight = company_preflight
        if not callable(getattr(
                company_preflight, "unique_question_company_surface", None)):
            raise TypeError(
                "event two-field merge company preflight 계약이 잘못되었습니다")

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str) or not question.strip()
                or len(intent.answer_items) != 2
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        first, second = intent.answer_items
        for item in (first, second):
            if (
                    item.operation != "retrieve"
                    or item.selection is not None
                    or item.output.projection_mode != "named_fields"
                    or item.output.shape != "scalar"
                    or len(item.output.field_surfaces) != 1
                    or item.output.field_surfaces[0] not in question
                    or item.target.kind not in {
                        "metric", "attribute", "entity", "event"}
                    or len(item.target.entity_refs) != 1
                    or item.scope.target_period_expressions
                    or item.scope.as_of_expression is not None
                    or item.scope.document_group_expression is not None
                    or item.scope.scope_qualifier_expressions
            ):
                return intent

        form_matches = [
            (entity, match)
            for entity in intent.entities
            if entity.kind_hint == "event"
            and (match := _DATE_FORM_SURFACE.match(entity.surface)) is not None
        ]
        if len(form_matches) != 1:
            return intent
        form_entity, match = form_matches[0]
        date_surface = match.group("date").strip()
        form_surface = match.group("form").strip()
        if (not date_surface or not form_surface
                or date_surface not in question
                or form_surface not in question):
            return intent
        parsed = parse_date_surface(date_surface)
        if parsed is None or parsed[1] is None or parsed[2] is None:
            return intent

        # Each item must resolve the shared disclosure only through the
        # composite surface itself — a qualifier naming something else is a
        # different axis this repair does not know how to merge. HCX does
        # not emit one stable qualifier shape across calls (observed live:
        # the full composite, a date-stripped fragment such as "공시한
        # 유형자산 양도결정", or a bare form fragment such as "영업양수") —
        # accept any literal fragment of the shared entity surface, never a
        # qualifier naming something that surface does not contain.
        for item in (first, second):
            references_form = form_entity.entity_id in item.target.entity_refs
            if not references_form and not all(
                    qualifier in form_entity.surface
                    for qualifier in item.target.qualifier_surfaces):
                return intent

        other_entities = [
            entity for entity in intent.entities
            if entity.entity_id != form_entity.entity_id]
        new_entity_payload: dict[str, str] | None = None
        existing_company_kind: dict[str, str] | None = None
        if len(other_entities) == 1:
            company_entity = other_entities[0]
            company_entity_id = company_entity.entity_id
            if company_entity.kind_hint != "company":
                # HCX can label the issuer itself as ``event`` (CG-022's
                # literal ``NC``).  The merged event target must reference a
                # company entity for the selected-event backend, but only
                # correct that label when the same literal question span is
                # uniquely recognized as one canonical company.
                surface = self.company_preflight.unique_question_company_surface(
                    question)
                if surface != company_entity.surface:
                    return intent
                existing_company_kind = {
                    "entity_id": company_entity.entity_id,
                    "kind_hint": "company",
                }
        elif not other_entities:
            surface = self.company_preflight.unique_question_company_surface(
                question)
            if surface is None:
                return intent
            company_entity_id = f"entity-{len(intent.entities) + 1}"
            new_entity_payload = {
                "entity_id": company_entity_id, "kind_hint": "company",
                "surface": surface,
            }
        else:
            return intent

        fields: list[str] = []
        for item in (first, second):
            surface = item.output.field_surfaces[0]
            if surface not in fields:
                fields.append(surface)
        if len(fields) != 2:
            return intent
        fields.sort(key=question.index)

        payload = intent.model_dump(mode="python", warnings=False)
        if new_entity_payload is not None:
            payload["entities"].append(new_entity_payload)
        if existing_company_kind is not None:
            for entity in payload["entities"]:
                if entity["entity_id"] == existing_company_kind["entity_id"]:
                    entity["kind_hint"] = existing_company_kind["kind_hint"]
                    break
        merged_item = payload["answer_items"][0]
        merged_item.update({
            "item_id": "item-1",
            "target": {
                "kind": "event", "surface": form_surface,
                "entity_refs": [company_entity_id],
                "qualifier_surfaces": [],
            },
            "operation": "retrieve",
            "scope": {
                "target_period_expressions": [date_surface],
                "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": "record", "projection_mode": "named_fields",
                "field_surfaces": fields, "presentation": "auto",
            },
        })
        payload["answer_items"] = [merged_item]
        return SemanticIntent.model_validate(payload, strict=True)


class ContractAmountPairClarificationRegrounder:
    """Turn a split contract amount pair into one ambiguity-aware event read.

    HCX may emit ``최종 계약금액`` and ``해지금액`` as two independent
    metric items even though both belong to one named supply contract.  A
    generic contract phrase can map to several canonical roots, so selecting
    either root would be unsafe.  This adapter only restores the shared event
    shape and marks that literal contract target unresolved; the canonical
    clarification backend must still prove multiple candidates and build the
    choices.
    """

    _FINAL_AMOUNT = re.compile(r"최종\s*계약\s*금액")
    _TERMINATION_AMOUNT = re.compile(r"해지\s*금액")

    def __init__(self, *, company_preflight: Any) -> None:
        self.company_preflight = company_preflight
        if not callable(getattr(
                company_preflight, "unique_question_company_surface", None)):
            raise TypeError(
                "contract amount pair company preflight 계약이 잘못되었습니다")

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str) or not question.strip()
                or len(intent.answer_items) != 2
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        first, second = intent.answer_items
        items = (first, second)
        if any(
                item.operation != "retrieve"
                or item.target.kind != "metric"
                or item.selection is not None
                or item.output.shape != "scalar"
                or item.output.projection_mode != "named_fields"
                or len(item.output.field_surfaces) != 1
                or item.scope.target_period_expressions
                or item.scope.as_of_expression is not None
                or item.scope.document_group_expression is not None
                or item.scope.scope_qualifier_expressions
                or item.target.qualifier_surfaces
                for item in items):
            return intent

        targets = [item.target.surface for item in items]
        final_indexes = [
            index for index, surface in enumerate(targets)
            if self._FINAL_AMOUNT.search(surface)]
        termination_indexes = [
            index for index, surface in enumerate(targets)
            if self._TERMINATION_AMOUNT.fullmatch(surface.strip())]
        final_demand = self._FINAL_AMOUNT.search(question)
        termination_demand = self._TERMINATION_AMOUNT.search(question)
        if (len(final_indexes) != 1 or len(termination_indexes) != 1
                or final_indexes == termination_indexes
                or final_demand is None or termination_demand is None):
            return intent

        final_target = targets[final_indexes[0]]
        contract_surface = self._FINAL_AMOUNT.sub("", final_target).strip()
        if (len(contract_surface) < 2 or "계약" not in contract_surface
                or contract_surface not in question):
            return intent

        issuer_surface = self.company_preflight.unique_question_company_surface(
            question)
        if issuer_surface is None:
            return intent
        issuer_entities = [
            entity for entity in intent.entities
            if entity.kind_hint == "company"
            and entity.surface == issuer_surface]
        counterparties = [
            entity for entity in intent.entities
            if entity.entity_id not in {
                row.entity_id for row in issuer_entities}
            and entity.surface in question]
        if len(issuer_entities) != 1 or len(counterparties) != 1:
            return intent
        issuer = issuer_entities[0]
        counterparty = counterparties[0]
        required_refs = {issuer.entity_id, counterparty.entity_id}
        if any(set(item.target.entity_refs) != required_refs for item in items):
            return intent

        payload = intent.model_dump(mode="python", warnings=False)
        for entity in payload["entities"]:
            if entity["entity_id"] == issuer.entity_id:
                entity["kind_hint"] = "company"
            elif entity["entity_id"] == counterparty.entity_id:
                entity["kind_hint"] = "counterparty"
        row = payload["answer_items"][0]
        row.update({
            "item_id": "item-1",
            "operation": "retrieve",
            "target": {
                "kind": "event", "surface": contract_surface,
                "entity_refs": [issuer.entity_id],
                "qualifier_surfaces": [],
            },
            "scope": {
                "target_period_expressions": [],
                "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": "record", "projection_mode": "named_fields",
                "field_surfaces": [
                    final_demand.group(0), termination_demand.group(0)],
                "presentation": "auto",
            },
        })
        payload["answer_items"] = [row]
        payload["unresolved_mentions"] = [{
            "mention_id": "unresolved-1", "raw_text": contract_surface,
            "role_hint": "target", "applies_to_item_ids": ["item-1"],
        }]
        return SemanticIntent.model_validate(payload, strict=True)


__all__ = [
    "CanonicalEventCollectionLike", "EventCollectionIdentity",
    "EventCollectionSelection", "select_event_collection",
    "ContractAmountAvailabilityFieldRegrounder",
    "EventArgmaxAmountFieldTargetRegrounder",
    "EventSingleFieldNarrativeShapeRegrounder",
    "EventTwoFieldItemMergeRegrounder",
    "ContractAmountPairClarificationRegrounder",
    "TerminationExistenceIntentRegrounder",
    "EventRecordDateQualifierRegrounder",
]
