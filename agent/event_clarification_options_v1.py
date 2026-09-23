"""Canonical-backed event candidate options for the Stage1 v1 boundary.

This adapter supplies the user-facing part of an event clarification.  It does
not choose an event, patch a plan, or create a Stage2-ready result.  Candidate
identity comes from :class:`CanonicalEventKeyPreflight`; display metadata comes
from the canonical ``fields`` API.  The only value exposed to a caller is the
canonical root receipt number, so a later resume can ask the resolver again
with an opaque selection.

The adapter deliberately has no question IDs, Gold rows, issuer names, or
receipt allowlists.  A caller supplies a company surface and a contract
category surface (for example, ``target_surface="공급계약"``); the canonical
company resolver must resolve that surface to exactly one company before event
preflight is attempted.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Literal, Mapping, Protocol

from src.canonical.events import event_support_role

from .event_preflight import (
    CanonicalEventKeyPreflight,
    EventKeyCandidate,
    EventKeyPreflight,
    EventKeyResolution,
)
from .semantic_intent_v1 import SemanticIntent
from .stage1_v1_resolver import (
    ClarificationAuthority,
    ClarificationOption,
    ClarificationSlot,
)


EventCandidateOptionsStatus = Literal[
    "ambiguous",
    "resolved",
    "not_found",
    "too_many",
    "company_not_found",
    "company_ambiguous",
    "metadata_incomplete",
    "intent_unsupported",
]


_ENERGY_QUANTITY_RE = re.compile(
    r"(?<![0-9.])([0-9]+(?:\.[0-9]+)?)\s*(TWh|GWh|MWh|kWh)(?![A-Za-z])",
    re.IGNORECASE,
)


class EventCompanyResolver(Protocol):
    def resolve_company(self, name: str) -> list[object]: ...


class EventMetadataCanonical(Protocol):
    def fields(self, **kwargs: object): ...

    def resolve_company(self, name: str) -> list[object]: ...

    def event_timeline(self, **kwargs: object): ...

    def documents(self, **kwargs: object): ...


@dataclass(frozen=True, slots=True)
class EventClarificationOptions:
    """The result of candidate lookup, before a v1 slot is assembled."""

    status: EventCandidateOptionsStatus
    options: tuple[ClarificationOption, ...] = ()
    corp_code: str | None = None
    resolution: EventKeyResolution | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class EventClarificationDecision:
    """Candidate lookup plus an intent-bound v1 authority when ambiguous."""

    lookup: EventClarificationOptions
    authority: ClarificationAuthority | None = None


def _object_josa(word: str) -> str:
    """받침 유무로 목적격 조사를 고른다 (``정정공시를``·``계약을``).

    ``app.textkit.josa`` 와 같은 규칙이지만 ``agent`` 는 ``app`` 을 임포트하지
    않는다(계층 규칙). 조사를 ``을`` 로 고정하면 받침 없는 대상에서
    ``정정공시을`` 같은 비문이 사용자 답변에 그대로 나간다.
    """

    if not word:
        return "를"
    last = word[-1]
    if "가" <= last <= "힣":
        return "을" if (ord(last) - 0xAC00) % 28 else "를"
    return "를"


def _surface(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _metadata_by_receipt(
        canonical: EventMetadataCanonical, *, corp_code: str,
        receipts: Iterable[str], as_of: str,
        ) -> dict[str, dict[str, tuple[str, ...]]]:
    """Read display metadata in one bounded canonical scan.

    Calling ``fields()`` once per option makes a 12-candidate clarification
    rescan the same Parquet artifact 12 times.  Candidate identity is already
    closed by preflight, so one company-scoped scan is enough.
    """

    wanted = set(receipts)
    values: dict[str, dict[str, set[str]]] = {
        receipt: {
            "counterparty": set(),
            "effective_date": set(),
            "event_name": set(),
            "differentiator": set(),
            "proof_refs": set(),
        }
        for receipt in wanted
    }
    for field in canonical.fields(
            as_of=as_of, corp_code=corp_code,
            is_pii=False, include_restricted_raw=False):
        receipt = getattr(field, "rcept_no", None)
        if receipt not in wanted:
            continue
        value = getattr(field, "value", None)
        # 계약상대·계약일·계약명이 같은 공시도 있다. 그런 후보를 접수번호만
        # 보고 고르게 하지 않도록, 공개 정본 본문에 명시된 공급물량을 안전한
        # 보조 식별자로 수집한다. 단위를 동반한 값만 허용하며 계산하거나
        # 다른 접수의 값을 끌어오지 않는다(R-A-003, #233).
        if isinstance(value, str):
            quantities = _ENERGY_QUANTITY_RE.findall(value)
            for number, unit in quantities:
                values[receipt]["differentiator"].add(
                    f"공급물량 {number}{unit}")
            if quantities:
                coordinate = getattr(field, "source_coordinate", None)
                if isinstance(coordinate, str) and coordinate:
                    values[receipt]["proof_refs"].add(
                        f"canonical:field:{coordinate}")
                evidence_id = getattr(field, "citation", None)
                if isinstance(evidence_id, str) and evidence_id:
                    values[receipt]["proof_refs"].add(
                        f"canonical:evidence:{evidence_id}")
        role = event_support_role(
            field.path, value_status=field.value_status, is_pii=field.is_pii)
        if role not in values[receipt]:
            continue
        if isinstance(value, str) and value.strip():
            values[receipt][role].add(value.strip())
            coordinate = getattr(field, "source_coordinate", None)
            if isinstance(coordinate, str) and coordinate:
                values[receipt]["proof_refs"].add(
                    f"canonical:field:{coordinate}")
            evidence_id = getattr(field, "citation", None)
            if isinstance(evidence_id, str) and evidence_id:
                values[receipt]["proof_refs"].add(
                    f"canonical:evidence:{evidence_id}")
    return {
        receipt: {
            role: tuple(sorted(items))
            for role, items in receipt_values.items()
        }
        for receipt, receipt_values in values.items()
    }


def _display_option(
        metadata: Mapping[str, Mapping[str, tuple[str, ...]]], *,
        candidate: EventKeyCandidate,
        include_differentiator: bool = False,
        ) -> ClarificationOption | None:
    values = metadata[candidate.seed_rcept_no]
    counterparty = " / ".join(values["counterparty"])
    contract_date = " / ".join(values["effective_date"])
    contract_name = " / ".join(values["event_name"])
    # Date and name identify what the receipt represents and remain required.
    # A filing may legitimately omit its counterparty.  Hiding that root would
    # make the candidate universe incomplete, so expose the absence explicitly
    # and keep the canonical receipt as the actual selected value.
    if not contract_date or not contract_name:
        return None
    reason = None
    if not counterparty:
        counterparty = "계약상대 미보고"
        reason = "정본 공시에 계약상대가 기재되지 않아 접수번호로 구분합니다."
    parts = [counterparty, contract_date, contract_name]
    if include_differentiator and values["differentiator"]:
        parts.append(" / ".join(values["differentiator"]))
    parts.append(candidate.seed_rcept_no)
    label = " · ".join(parts)
    proof_refs = {
        f"canonical:event-root:{candidate.event_key}:"
        f"{candidate.seed_rcept_no}",
        *values["proof_refs"],
    }
    return ClarificationOption(
        value=candidate.seed_rcept_no, label=label, reason=reason,
        proof_refs=sorted(proof_refs))


def build_event_clarification_options(
        canonical: EventMetadataCanonical,
        *,
        company_surface: str,
        as_of: str,
        target_surface: str | None = None,
        contract_surface: str | None = None,
        event_type_surface: str | None = None,
        counterparty_surface: str | None = None,
        event_from: str | None = None,
        event_to: str | None = None,
        correction_observed_on: str | None = None,
        preflight: EventKeyPreflight | None = None,
        corpus_cutoff: str | None = None,
        ) -> EventClarificationOptions:
    """Build typed v1 options for an ambiguous canonical event lookup.

    ``contract_surface`` is preferred when present and ``target_surface`` is
    its semantic-target fallback.  The surfaces are lookup hints only; no
    value is copied into a plan.  Company resolution is intentionally strict:
    zero or multiple canonical companies returns without scanning event rows.
    """

    surface = _surface(company_surface)
    if surface is None:
        return EventClarificationOptions(
            status="company_not_found", reason="company_surface_empty")
    companies = canonical.resolve_company(surface)
    if not companies:
        return EventClarificationOptions(
            status="company_not_found", reason="company_surface_not_found")
    if len(companies) != 1:
        return EventClarificationOptions(
            status="company_ambiguous", reason="company_surface_multiple")
    company = companies[0]
    corp_code = getattr(company, "corp_code", None)
    if not isinstance(corp_code, str) or not corp_code:
        return EventClarificationOptions(
            status="company_not_found", reason="company_corp_code_missing")

    if preflight is None:
        if corpus_cutoff is None:
            raise ValueError("preflight 또는 corpus_cutoff 중 하나가 필요합니다")
        preflight = CanonicalEventKeyPreflight(
            canonical, corpus_cutoff=corpus_cutoff)
    category = _surface(contract_surface) or _surface(target_surface)
    # ``contract_name`` is a hard identity filter.  A generic category such
    # as "계약" must not select whichever one canonical title happens to
    # contain that word.  Use the preflight classifier when available and
    # fail open only into the *unfiltered candidate set*, never into a hard
    # filter.  Candidate ambiguity is handled below as a clarification.
    contract_name = category
    classify = getattr(preflight, "names_a_single_contract", None)
    if category is not None and callable(classify):
        try:
            if not classify(corp_code=corp_code, surface=category):
                contract_name = None
        except Exception:                                  # noqa: BLE001
            contract_name = None
    correction_date = _surface(correction_observed_on)
    if correction_date is not None and re.fullmatch(r"[0-9]{8}", correction_date) is None:
        raise ValueError("correction_observed_on 형식이 잘못되었습니다")
    resolution = preflight.resolve_event_key(
        corp_code=corp_code,
        as_of=as_of,
        event_type=_surface(event_type_surface),
        counterparty=_surface(counterparty_surface),
        contract_name=contract_name,
        event_from=_surface(event_from),
        event_to=_surface(event_to),
    )

    # The ordinary role index is deliberately capped at 20 candidates.  An
    # exact correction date is a stronger public selector, but a capped
    # ``too_many`` result contains no candidates and therefore cannot be
    # filtered by the loop below.  Reconstruct only the bounded same-day set
    # from canonical document metadata, then round-trip each receipt through
    # the verified event timeline.  This is generic across issuers and dates;
    # it never guesses from receipt ordering or a fixture answer.
    if resolution.status == "too_many" and correction_date is not None:
        documents = getattr(canonical, "documents", None)
        if callable(documents):
            discovered: dict[str, EventKeyCandidate] = {}
            try:
                rows = documents(
                    as_of=as_of, corp_code=corp_code, is_correction=True)
                for document in rows:
                    observed = str(getattr(document, "rcept_dt", ""))
                    if observed > correction_date:
                        break
                    if observed != correction_date:
                        continue
                    if (category is not None and "계약" in category
                            and getattr(document, "doc_group", None) != "exchange"):
                        continue
                    receipt = str(getattr(document, "rcept_no", ""))
                    timeline = canonical.event_timeline(
                        as_of=as_of, rcept_no=receipt, verify_evidence=True)
                    if (timeline is None
                            or getattr(timeline, "corp_code", None) != corp_code):
                        continue
                    observations = tuple(
                        getattr(timeline, "observations", ()) or ())
                    if not any(
                            getattr(row, "rcept_no", None) == receipt
                            and bool(getattr(row, "is_correction", False))
                            and getattr(row, "observed_at", None) == correction_date
                            and getattr(row, "evidence_verification_status", None)
                            == "verified"
                            for row in observations):
                        continue
                    event_key = getattr(timeline, "event_key", None)
                    root = getattr(timeline, "root_rcept_no", None)
                    if (not isinstance(event_key, str)
                            or re.fullmatch(r"[0-9a-f]{32}", event_key) is None
                            or not isinstance(root, str)
                            or re.fullmatch(r"[0-9]{14}", root) is None):
                        continue
                    discovered[event_key] = EventKeyCandidate(
                        event_key=event_key,
                        seed_rcept_no=root,
                        label=(str(getattr(document, "report_nm", ""))
                               or f"{root} 기준 사건"),
                        identity_fingerprint=getattr(
                            timeline, "identity_fingerprint", None),
                        identity_status=getattr(
                            timeline, "identity_status", None),
                        observation_count=len(observations),
                    )
                    if len(discovered) > 20:
                        discovered.clear()
                        break
            except Exception:                              # noqa: BLE001
                discovered.clear()
            candidates = tuple(sorted(
                discovered.values(),
                key=lambda row: (row.event_key, row.seed_rcept_no)))
            if candidates:
                resolution = EventKeyResolution(
                    status=("resolved" if len(candidates) == 1
                            else "ambiguous"),
                    candidates=candidates,
                )
    if resolution.status not in {"ambiguous", "resolved"}:
        return EventClarificationOptions(
            status=resolution.status, corp_code=corp_code,
            resolution=resolution)

    # A correction filing date is an observation facet, not an event/root
    # date.  Apply it only after ordinary identity preflight and retain every
    # candidate whose verified canonical timeline contains such a correction.
    # This never uses receipt-number ordering and never chooses among multiple
    # same-day matches.
    if correction_date is not None:
        matching: list[EventKeyCandidate] = []
        try:
            for candidate in resolution.candidates:
                timeline = canonical.event_timeline(
                    as_of=as_of, event_key=candidate.event_key,
                    verify_evidence=True)
                if timeline is None:
                    continue
                if any(
                        bool(getattr(row, "is_correction", False))
                        and getattr(row, "observed_at", None) == correction_date
                        for row in tuple(getattr(timeline, "observations", ()) or ())):
                    matching.append(candidate)
        except Exception:
            return EventClarificationOptions(
                status="not_found", corp_code=corp_code,
                reason="correction_observation_lookup_failed")
        if not matching:
            return EventClarificationOptions(
                status="not_found", corp_code=corp_code,
                reason="correction_observation_not_found")
        resolution = EventKeyResolution(
            status="resolved" if len(matching) == 1 else "ambiguous",
            candidates=tuple(matching),
        )

    metadata = _metadata_by_receipt(
        canonical,
        corp_code=corp_code,
        receipts=(row.seed_rcept_no for row in resolution.candidates),
        as_of=as_of,
    )
    options: list[ClarificationOption] = []
    # 같은 상대·계약일·계약명 조합끼리만 보조 식별자를 붙인다. 서로 다른
    # 계약까지 장황하게 만들지 않으면서, Ford의 75GWh/34GWh처럼 실제로
    # 겹치는 후보는 공개 정본 값으로 구분한다.
    identity_counts: dict[tuple[tuple[str, ...], ...], int] = {}
    for candidate in resolution.candidates:
        values = metadata[candidate.seed_rcept_no]
        identity = (
            values["counterparty"], values["effective_date"],
            values["event_name"],
        )
        identity_counts[identity] = identity_counts.get(identity, 0) + 1
    # ``EventKeyResolution`` is ordered by the opaque event key because that is
    # its canonical identity contract.  A user-facing list has a different
    # contract: keep it stable and chronological so option positions do not
    # move when event-key generation or canonical scan order changes.  The
    # 14-digit root receipt gives a total filing-date/receipt order without a
    # question ID, issuer allowlist, or Gold fixture.
    for candidate in sorted(
            resolution.candidates, key=lambda row: row.seed_rcept_no):
        values = metadata[candidate.seed_rcept_no]
        identity = (
            values["counterparty"], values["effective_date"],
            values["event_name"],
        )
        option = _display_option(
            metadata, candidate=candidate,
            include_differentiator=identity_counts[identity] > 1)
        if option is None:
            return EventClarificationOptions(
                status="metadata_incomplete", corp_code=corp_code,
                resolution=resolution,
                reason="event_candidate_display_metadata_incomplete")
        options.append(option)
    return EventClarificationOptions(
        status=resolution.status, options=tuple(options), corp_code=corp_code,
        resolution=resolution)


def build_event_clarification_decision(
        canonical: EventMetadataCanonical,
        *,
        source_intent: SemanticIntent,
        as_of: str,
        preflight: EventKeyPreflight | None = None,
        corpus_cutoff: str | None = None,
        contract_surface: str | None = None,
        event_type_surface: str | None = None,
        counterparty_surface: str | None = None,
        event_from: str | None = None,
        event_to: str | None = None,
        correction_observed_on: str | None = None,
        ) -> EventClarificationDecision:
    """Bind a single event/document item to a canonical candidate slot.

    The binding is structural and question-ID free.  If the intent contains
    unresolved mentions, all of them must be target mentions owned by the one
    event item; otherwise this helper declines instead of emitting a partial
    clarification authority that the v1 resolver would reject.
    """

    items = [
        item for item in source_intent.answer_items
        if item.target.kind in {"event", "document"}
    ]
    if len(items) != 1:
        lookup = EventClarificationOptions(
            status="intent_unsupported", reason="single_event_item_required")
        return EventClarificationDecision(lookup=lookup)
    item = items[0]
    entity_by_id = {row.entity_id: row for row in source_intent.entities}
    company_entities = [
        entity_by_id[ref]
        for ref in item.target.entity_refs
        if ref in entity_by_id and entity_by_id[ref].kind_hint == "company"
    ]
    if len(company_entities) != 1:
        lookup = EventClarificationOptions(
            status="intent_unsupported", reason="single_company_entity_required")
        return EventClarificationDecision(lookup=lookup)

    mentions = list(source_intent.unresolved_mentions)
    if any(
            mention.role_hint != "target"
            or mention.applies_to_item_ids != [item.item_id]
            for mention in mentions
    ):
        lookup = EventClarificationOptions(
            status="intent_unsupported",
            reason="event_target_clarification_cannot_cover_all_mentions",
        )
        return EventClarificationDecision(lookup=lookup)

    lookup = build_event_clarification_options(
        canonical,
        company_surface=company_entities[0].surface,
        as_of=as_of,
        target_surface=item.target.surface,
        contract_surface=contract_surface,
        event_type_surface=event_type_surface,
        counterparty_surface=counterparty_surface,
        event_from=event_from,
        event_to=event_to,
        correction_observed_on=correction_observed_on,
        preflight=preflight,
        corpus_cutoff=corpus_cutoff,
    )
    if lookup.status != "ambiguous":
        return EventClarificationDecision(lookup=lookup)
    # 후보 목록은 이미 각각 어떤 계약인지 보여준다. 출력 필드를 조사와
    # 기계 결합하면 ``바뀌었어을`` 같은 비문이 생기므로, 이 slot의 실제
    # 결정 대상인 사건/계약만 자연스럽게 묻는다.
    prompt = (
        f"어느 {item.target.surface}"
        f"{_object_josa(item.target.surface)} 확인할까요?")
    authority = ClarificationAuthority(slots=[ClarificationSlot(
        slot_id="slot-1",
        role_hint="target",
        reason_code="event_target_multiple_candidates",
        response_kind="select_one",
        prompt=prompt,
        applies_to_item_ids=[item.item_id],
        mention_ids=[row.mention_id for row in mentions],
        options=list(lookup.options),
    )])
    return EventClarificationDecision(lookup=lookup, authority=authority)


__all__ = [
    "EventCandidateOptionsStatus",
    "EventClarificationDecision",
    "EventClarificationOptions",
    "EventCompanyResolver",
    "EventMetadataCanonical",
    "build_event_clarification_decision",
    "build_event_clarification_options",
]
