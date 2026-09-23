"""Fail-closed native selector for correction lineage questions.

This module deliberately stops before the v0.4 emitter/compiler boundary.  It
turns a question-grounded :class:`SemanticIntent` and public canonical reads
into a small typed lineage payload.  A caller may later map that payload to a
plan, but cannot recover a receipt, issuer, or value from Gold/fixture data.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal, Protocol

from .event_preflight import (
    CanonicalEventKeyPreflight, EventKeyCandidate, EventKeyResolution,
)
from .semantic_intent_v1 import SemanticIntent


_DATE = re.compile(r"^[0-9]{8}$")
_RECEIPT = re.compile(r"^[0-9]{14}$")
_EVIDENCE = re.compile(r"^[0-9a-f]{32}$")
_KOREAN_DATE = re.compile(
    r"(?<![0-9])(?P<year>[0-9]{4})\s*년\s*(?P<month>[0-9]{1,2})\s*월\s*(?P<day>[0-9]{1,2})\s*일")
_COMPACT_DATE = re.compile(r"(?<![0-9])(?P<date>[0-9]{8})(?![0-9])")
_CORRECTION_WORD = re.compile(r"정정|기재정정|수정", re.IGNORECASE)
_HISTORY_WORD = re.compile(r"계약금액|계약\s*금액", re.IGNORECASE)
_AMOUNT_PATH = re.compile(r"계약\s*금액\s*(?:\(\s*원\s*\))?", re.IGNORECASE)
_EXPLICIT_RECEIPT = re.compile(r"(?<![0-9])20[0-9]{12}(?![0-9])")
_BROAD_CHANGE_REQUEST = re.compile(
    r"변경\s*(?:된\s*)?(?:내용|사항)|무엇(?:이|을)?.*변경|"
    r"어떤.*변경|전체.*변경")
#: 한 건이 아니라 **두 정정공시 사이의 구간**을 가리키는 표면.
#: 날짜를 대더라도 끝점을 둘 다 지목하면 문서 하나를 고른 것이 아니다.
_CORRECTION_SPAN = re.compile(
    r"마지막\s*(?:기재\s*)?정정\s*공시|"
    r"(?:최초|처음|첫)[^?\n]{0,40}(?:과|와)[^?\n]{0,40}사이|"
    r"정정\s*공시[^?\n]{0,40}사이에")

_EXPLICIT_ALL_CHANGES = re.compile(
    r"(?:전체|모든)\s*(?:변경\s*(?:된\s*)?)?(?:내용|사항|항목)|"
    r"변경\s*(?:된\s*)?(?:내용|사항|항목)\s*(?:전체|전부|모두)")


CorrectionLineageStatus = Literal[
    "resolved", "not_applicable", "not_found", "ambiguous", "unavailable",
]
CorrectionLineageKind = Literal["correction_diff", "contract_amount_history"]


class _CanonicalLike(Protocol):
    def resolve_company(self, name: str) -> list[Any]: ...

    def event_timeline(self, **kwargs: object) -> Any | None: ...

    def correction_items(self, **kwargs: object) -> Any: ...

    def documents(self, **kwargs: object) -> Any: ...


@dataclass(frozen=True, slots=True)
class CanonicalProof:
    """One verified canonical source coordinate, never an inferred citation."""

    proof_ref: str
    source_receipt: str
    evidence_id: str

    def __post_init__(self) -> None:
        if (not self.proof_ref.startswith("canonical:")
                or not _RECEIPT.fullmatch(self.source_receipt)
                or not _EVIDENCE.fullmatch(self.evidence_id)):
            raise ValueError("canonical correction proof 계약이 잘못되었습니다")


@dataclass(frozen=True, slots=True)
class CorrectionFieldChange:
    path: str
    before: str | None
    after: str | None
    reason: str | None
    before_proof: CanonicalProof | None
    after_proof: CanonicalProof | None

    def __post_init__(self) -> None:
        if not self.path.strip() or (self.before_proof is None
                                     and self.after_proof is None):
            raise ValueError("correction field change는 path와 proof가 필요합니다")


@dataclass(frozen=True, slots=True)
class CorrectionDateRoles:
    """Question dates after assigning their non-interchangeable roles."""

    root_observed_at: str | None
    correction_from: str | None
    correction_to: str | None
    as_of: str
    range_requested: bool = False

    def __post_init__(self) -> None:
        values = tuple(value for value in (
            self.root_observed_at, self.correction_from,
            self.correction_to, self.as_of) if value is not None)
        if any(_DATE.fullmatch(value) is None for value in values):
            raise ValueError("correction date role 형식이 잘못되었습니다")
        if (self.correction_from and self.correction_to
                and self.correction_from > self.correction_to):
            raise ValueError("correction date range 순서가 잘못되었습니다")
        if self.correction_to and self.correction_to > self.as_of:
            raise ValueError("correction_to는 as_of 이후일 수 없습니다")


@dataclass(frozen=True, slots=True)
class CorrectionSequenceStep:
    correction_receipt: str
    correction_date: str
    previous_observation_receipt: str | None
    event_proofs: tuple[CanonicalProof, ...]
    changes: tuple[CorrectionFieldChange, ...]

    def __post_init__(self) -> None:
        if (not _RECEIPT.fullmatch(self.correction_receipt)
                or not _DATE.fullmatch(self.correction_date)
                or self.correction_receipt[:8] != self.correction_date
                or not self.event_proofs or not self.changes):
            raise ValueError("correction sequence step 계약이 잘못되었습니다")


@dataclass(frozen=True, slots=True)
class CorrectionDiffResolution:
    kind: Literal["correction_diff"]
    issuer_corp_code: str
    issuer_corp_name: str
    event_key: str
    root_receipt: str
    correction_receipt: str
    correction_date: str
    event_proofs: tuple[CanonicalProof, ...]
    changes: tuple[CorrectionFieldChange, ...]
    question_premises: tuple[str, ...]
    counterparty: str | None = None
    product_keywords: tuple[str, ...] = ()
    date_roles: CorrectionDateRoles | None = None
    sequence: tuple[CorrectionSequenceStep, ...] = ()
    # Some corpora begin at the correction filing.  Verified before/after
    # cells can still answer the bounded diff even though no separate original
    # receipt is present.  Equality is explicit; a fabricated earlier receipt
    # remains forbidden.
    source_root_missing: bool = False

    def __post_init__(self) -> None:
        _validate_lineage_identity(
            self.issuer_corp_code, self.issuer_corp_name, self.event_key,
            self.root_receipt, self.correction_receipt, self.correction_date,
            allow_same_receipt=self.source_root_missing,
        )
        if not self.event_proofs or not self.changes:
            raise ValueError("correction diff는 proof와 change가 필요합니다")
        if (not self.source_root_missing
                and self.root_receipt == self.correction_receipt):
            raise ValueError("correction diff root-missing 표시가 receipt와 다릅니다")
        if self.sequence:
            if (self.sequence[-1].correction_receipt != self.correction_receipt
                    or self.sequence[-1].correction_date != self.correction_date
                    or tuple(proof for step in self.sequence
                             for proof in step.event_proofs) != self.event_proofs
                    or tuple(change for step in self.sequence
                             for change in step.changes) != self.changes):
                raise ValueError("correction diff sequence aggregate가 다릅니다")


@dataclass(frozen=True, slots=True)
class ContractAmountHistoryResolution:
    kind: Literal["contract_amount_history"]
    issuer_corp_code: str
    issuer_corp_name: str
    event_key: str
    root_receipt: str
    correction_receipt: str
    correction_date: str
    event_proofs: tuple[CanonicalProof, ...]
    amount_change: CorrectionFieldChange
    question_premises: tuple[str, ...]
    counterparty: str | None = None
    product_keywords: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_lineage_identity(
            self.issuer_corp_code, self.issuer_corp_name, self.event_key,
            self.root_receipt, self.correction_receipt, self.correction_date,
        )
        if not self.event_proofs or not _AMOUNT_PATH.search(self.amount_change.path):
            raise ValueError("contract amount history 계약이 잘못되었습니다")


@dataclass(frozen=True, slots=True)
class RootMissingCorrectionResolution:
    """One observed correction whose original filing predates the corpus.

    This authority intentionally contains no original value and no fabricated
    root receipt.  It only preserves the canonical ``CORRECTS`` edge and its
    typed absence reason so a later compiler can produce a partial answer.
    """

    kind: Literal["root_missing_correction"]
    issuer_corp_code: str
    issuer_corp_name: str
    correction_receipt: str
    correction_date: str
    relation_id: str
    relation_proof_ref: str
    target_hint: str
    root_missing_reason: Literal["submitted_before_corpus"]
    question_premises: tuple[str, ...]

    def __post_init__(self) -> None:
        if (not re.fullmatch(r"[0-9]{8}", self.issuer_corp_code)
                or not self.issuer_corp_name.strip()
                or not _RECEIPT.fullmatch(self.correction_receipt)
                or not _DATE.fullmatch(self.correction_date)
                or self.correction_receipt[:8] != self.correction_date
                or not self.relation_id.strip()
                or self.relation_proof_ref != f"canonical:relation:{self.relation_id}"
                or not self.target_hint.strip()):
            raise ValueError("root-missing correction 계약이 잘못되었습니다")


@dataclass(frozen=True, slots=True)
class CorrectionLineageResult:
    status: CorrectionLineageStatus
    resolution: (
        CorrectionDiffResolution
        | ContractAmountHistoryResolution
        | RootMissingCorrectionResolution
        | None
    ) = None

    def __post_init__(self) -> None:
        if (self.status == "resolved") != (self.resolution is not None):
            raise ValueError("correction lineage result 상태 계약이 잘못되었습니다")


def _validate_lineage_identity(
        corp_code: str, corp_name: str, event_key: str, root: str,
        correction: str, correction_date: str, *,
        allow_same_receipt: bool = False) -> None:
    if (not _DATE.fullmatch(correction_date)
            or not _RECEIPT.fullmatch(root)
            or not _RECEIPT.fullmatch(correction)
            or not re.fullmatch(r"[0-9a-f]{32}", event_key)
            or not re.fullmatch(r"[0-9]{8}", corp_code)
            or not corp_name.strip()
            or root > correction
            or (root == correction and not allow_same_receipt)):
        raise ValueError("correction lineage identity 계약이 잘못되었습니다")


def _date_surfaces(question: str) -> tuple[str, ...]:
    dates = []
    for match in _KOREAN_DATE.finditer(question):
        try:
            value = f"{int(match.group('year')):04d}{int(match.group('month')):02d}{int(match.group('day')):02d}"
        except ValueError:
            continue
        dates.append(value)
    dates.extend(match.group("date") for match in _COMPACT_DATE.finditer(question))
    return tuple(sorted(set(dates)))


def _date_roles(question: str, *, as_of: str) -> CorrectionDateRoles:
    """Assign literal dates by nearby role words, never by positional order."""

    root: str | None = None
    correction_from: str | None = None
    correction_to: str | None = None
    range_requested = False
    matches = list(_KOREAN_DATE.finditer(question))
    for match in matches:
        value = (f"{int(match.group('year')):04d}"
                 f"{int(match.group('month')):02d}"
                 f"{int(match.group('day')):02d}")
        left = question[max(0, match.start() - 20):match.start()]
        right = question[match.end():min(len(question), match.end() + 28)]
        context = left + " " + right
        # Bind the role to the date immediately next to the cue.  A broad
        # left+right search mislabels the later correction date when a short
        # contract name sits between the two dates ("최초 공시한 X는 <date>").
        root_after_date = (
            re.match(r"\s*(?:에\s*)?(?:최초\s*(?:제출|공시)|원\s*공시)", right)
            or re.match(
                r"\s*공시(?:한|된)?(?=[^?。\n]{0,80}(?:은|는)\s*"
                r"(?:그\s*)?이후\s*정정\s*공시)", right))
        root_before_date = re.search(
            r"(?:최초\s*(?:제출|공시)|원\s*공시)\s*(?:일자|일|날짜)?\s*$",
            left)
        if root_after_date or root_before_date:
            if root is not None and root != value:
                raise ValueError("root_observed_at 후보가 복수입니다")
            root = value
            if re.search(r"(?:이후|부터).*?정정|정정.*?(?:이후|부터)", right):
                correction_from = value
                range_requested = True
            continue
        if re.search(r"(?:이후|부터).*?정정|정정.*?(?:이후|부터)", right):
            if correction_from is not None and correction_from != value:
                raise ValueError("correction_from 후보가 복수입니다")
            correction_from = value
            range_requested = True
            continue
        if re.search(r"(?:까지).*?정정|정정.*?(?:까지)", context):
            if correction_to is not None and correction_to != value:
                raise ValueError("correction_to 후보가 복수입니다")
            correction_to = value
            range_requested = True
            continue
        if re.search(r"정정\s*공시", right):
            if correction_from is not None and correction_from != value:
                raise ValueError("정정공시 날짜 후보가 복수입니다")
            correction_from = correction_to = value
    # "<date> correction filing ... correction history" asks for the
    # observable history beginning at that filing, not only that one row.
    # The phrase is issuer/date independent and cannot fabricate history: the
    # canonical timeline and verified correction items remain authoritative.
    if (correction_from is not None and correction_from == correction_to
            and re.search(
                r"정정\s*(?:공시\s*)?(?:변경\s*)?이력|"
                r"마지막\s*정정|접수번호\s*별|"
                r"전체\s*(?:정정\s*)?이력",
                question)):
        correction_to = as_of
        range_requested = True
    if correction_from and correction_to is None:
        correction_to = as_of
    return CorrectionDateRoles(
        root_observed_at=root,
        correction_from=correction_from,
        correction_to=correction_to,
        as_of=as_of,
        range_requested=range_requested,
    )


def _question_premises(question: str, intent: SemanticIntent) -> tuple[str, ...]:
    """Keep only literal premise spans the normalizer already grounded."""

    return tuple(p.raw_text for p in intent.premises if p.raw_text in question)


def _contract_product_keywords(label: str) -> tuple[str, ...]:
    """Recover a product facet immediately modifying a contract noun.

    The label is canonical preflight output.  Restricting this to the token
    directly before a public sale/supply contract head avoids treating issuer
    or counterparty text as a retrieval keyword.
    """
    match = re.search(
        r"(?P<product>[가-힣A-Za-z0-9-]{2,})\s+(?:공급|판매|구매|제조)\s*계약",
        label)
    return (match.group("product"),) if match is not None else ()


def _kind(question: str, intent: SemanticIntent) -> CorrectionLineageKind | None:
    fields = tuple(
        surface for item in intent.answer_items for surface in item.output.field_surfaces)
    asks_amount = bool(_HISTORY_WORD.search(question) or any(
        _AMOUNT_PATH.search(surface) for surface in fields))
    # A literal correction surface wins over a broad amount noun.  The latter
    # is history only when a grounded premise makes it a verification question.
    if _CORRECTION_WORD.search(question):
        return "correction_diff"
    if asks_amount and _question_premises(question, intent):
        return "contract_amount_history"
    return None


def _one_company(canonical: _CanonicalLike, intent: SemanticIntent) -> Any | None:
    candidates: dict[str, Any] = {}
    for entity in intent.entities:
        if entity.kind_hint != "company":
            continue
        rows = canonical.resolve_company(entity.surface)
        if len(rows) == 1:
            row = rows[0]
            code = getattr(row, "corp_code", None)
            if isinstance(code, str):
                candidates[code] = row
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _counterparty(canonical: _CanonicalLike, intent: SemanticIntent) -> str | None:
    explicit = [entity.surface for entity in intent.entities
                if entity.kind_hint == "counterparty"]
    if len(explicit) == 1:
        return explicit[0]
    unresolved = [entity.surface for entity in intent.entities
                  if entity.kind_hint in {"company", "event", "document"}
                  and not canonical.resolve_company(entity.surface)]
    if len(unresolved) == 1:
        return unresolved[0]
    return _counterparty_from_target_surfaces(intent)


def _counterparty_from_target_surfaces(intent: SemanticIntent) -> str | None:
    """Extract one literal external-party candidate from a target surface.

    This is intentionally only a surface split.  ``event_preflight`` still
    proves the role against canonical contract fields.  A unique Latin token
    (or one alias key) is useful when HCX has collapsed ``<party> <contract>``
    into a document/event target and emitted no counterparty entity.
    """

    from .event_preflight import _counterparty_aliases, _key

    surfaces = tuple(item.target.surface for item in intent.answer_items)
    candidates: list[tuple[str, tuple[str, ...]]] = []
    aliases = _counterparty_aliases()
    for surface in surfaces:
        normalized = _key(surface)
        alias_hits = [
            (alias, values) for alias, values in aliases.items()
            if len(alias) >= 2 and alias in normalized
        ]
        if alias_hits:
            identities = {values for _, values in alias_hits}
            if len(identities) == 1:
                alias, values = max(alias_hits, key=lambda row: len(row[0]))
                candidates.append((alias, values))
        latin = re.findall(r"[A-Za-z][A-Za-z0-9.&'-]*", surface)
        expanded_acronym = bool(latin and re.search(
            rf"{re.escape(latin[0])}\s*\([^)]*[가-힣][^)]*\)", surface))
        if len(latin) == 1 and not expanded_acronym:
            candidates.append((latin[0], (latin[0].casefold(),)))
    by_identity: dict[tuple[str, ...], str] = {}
    for surface, identity in candidates:
        current = by_identity.get(identity)
        if current is None or len(surface) > len(current):
            by_identity[identity] = surface
    return next(iter(by_identity.values())) if len(by_identity) == 1 else None


def _contract_surface_candidates(surface: str) -> tuple[str, ...]:
    """Return literal contract-name candidates without filing/date wrappers.

    HCX may keep ``<contract> 관련 <date> 정정공시`` as one document
    target.  The suffix is a filing selector, not part of the canonical event
    name.  Only a trailing, literal correction-filing wrapper is removed;
    arbitrary words in the middle of a contract name are never rewritten.
    """

    values = [surface.strip()]
    stripped = re.sub(
        r"\s*(?:관련\s*)?(?:(?:[0-9]{4}\s*년\s*[0-9]{1,2}\s*월\s*"
        r"[0-9]{1,2}\s*일)|(?:[0-9]{8}))\s*(?:기재\s*)?정정\s*공시\s*$",
        "", surface, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"\s*(?:관련|에\s*대한)\s*$", "", stripped).strip()
    if stripped and stripped not in values:
        values.append(stripped)
    output: list[str] = []
    for value in values:
        without_head = re.sub(r"\s*계약\s*$", "", value).strip()
        for candidate in (value, without_head):
            if candidate and candidate not in output:
                output.append(candidate)
    return tuple(output)


def _event_attempts(intent: SemanticIntent, counterparty: str | None) -> tuple[dict[str, str], ...]:
    surfaces = tuple(sorted({item.target.surface.strip() for item in intent.answer_items
                             if item.target.surface.strip()}))
    attempts: list[dict[str, str]] = []
    for surface in surfaces:
        for contract_name in _contract_surface_candidates(surface):
            if contract_name:
                value = {"contract_name": contract_name}
                if counterparty is not None:
                    value["counterparty"] = counterparty
                attempts.append(value)
    if counterparty is not None:
        attempts.append({"counterparty": counterparty})
    seen: set[tuple[tuple[str, str], ...]] = set()
    output: list[dict[str, str]] = []
    for attempt in attempts:
        key = tuple(sorted(attempt.items()))
        if key not in seen:
            seen.add(key)
            output.append(attempt)
    return tuple(output)


def _verified_event_proofs(timeline: Any, receipt: str) -> tuple[CanonicalProof, ...]:
    for observation in tuple(getattr(timeline, "observations", ()) or ()):
        if getattr(observation, "rcept_no", None) != receipt:
            continue
        if getattr(observation, "evidence_verification_status", None) != "verified":
            return ()
        values = tuple(getattr(observation, "supporting_evidence_ids", ()) or ())
        if not values or any(not _EVIDENCE.fullmatch(value) for value in values):
            return ()
        return tuple(CanonicalProof(
            proof_ref=f"canonical:event:{value}", source_receipt=receipt,
            evidence_id=value) for value in values)
    return ()


def _field_change(row: Any) -> CorrectionFieldChange | None:
    if getattr(row, "evidence_status", None) != "verified":
        return None
    receipt = getattr(row, "rcept_no", None)
    if not isinstance(receipt, str) or not _RECEIPT.fullmatch(receipt):
        return None
    def proof(value: Any, side: str) -> CanonicalProof | None:
        if value is None or not _EVIDENCE.fullmatch(value):
            return None
        if getattr(row, f"{side}_evidence_status", None) != "verified":
            return None
        return CanonicalProof(
            proof_ref=f"canonical:correction_{side}:{value}",
            source_receipt=receipt, evidence_id=value)
    before, after = proof(getattr(row, "before_evidence_id", None), "before"), proof(
        getattr(row, "after_evidence_id", None), "after")
    if before is None and after is None:
        return None
    path = getattr(row, "path", None)
    if not isinstance(path, str) or not path.strip():
        return None
    return CorrectionFieldChange(
        path=path,
        before=(getattr(row, "value_before", None) if before is not None else None),
        after=(getattr(row, "value_after", None) if after is not None else None),
        reason=getattr(row, "reason", None),
        before_proof=before, after_proof=after)


class CanonicalCorrectionLineageSelector:
    """Resolve one unique correction event entirely from canonical authority."""

    def __init__(self, canonical: _CanonicalLike, *, corpus_cutoff: str,
                 event_preflight: Any | None = None) -> None:
        if (not _DATE.fullmatch(corpus_cutoff)
                or any(not callable(getattr(canonical, name, None)) for name in (
                    "resolve_company", "event_timeline", "correction_items"))):
            raise TypeError("correction lineage canonical reader 계약이 잘못되었습니다")
        self._canonical = canonical
        self._cutoff = corpus_cutoff
        self._preflight = event_preflight or CanonicalEventKeyPreflight(
            canonical, corpus_cutoff=corpus_cutoff)
        if not callable(getattr(self._preflight, "resolve_event_key", None)):
            raise TypeError("correction lineage event preflight 계약이 잘못되었습니다")

    def resolve(self, *, question: str, semantic_intent: SemanticIntent,
                cutoff: str) -> CorrectionLineageResult:
        if not isinstance(question, str) or not question.strip() or not _DATE.fullmatch(cutoff):
            raise ValueError("correction lineage 질문/cutoff 계약이 잘못되었습니다")
        if cutoff > self._cutoff:
            return CorrectionLineageResult("unavailable")
        kind = _kind(question, semantic_intent)
        if kind is None:
            return CorrectionLineageResult("not_applicable")
        try:
            date_roles = _date_roles(question, as_of=cutoff)
        except ValueError:
            return CorrectionLineageResult("ambiguous")
        counterparty = _counterparty(self._canonical, semantic_intent)
        company = _one_company(self._canonical, semantic_intent)
        if company is not None:
            observed = self._explicit_receipt_history(
                company, question=question, intent=semantic_intent, cutoff=cutoff)
            if observed is not None:
                return observed
            exact = self._select_exact_correction_document(
                company, question=question, intent=semantic_intent,
                cutoff=cutoff)
            if exact is not None:
                return exact
        selected_pair: tuple[Any, EventKeyResolution] | None = None
        if company is not None:
            selected_pair = self._select_for_company(
                company, semantic_intent, counterparty, cutoff,
                root_observed_at=date_roles.root_observed_at,
                correction_observed_on=(
                    date_roles.correction_from
                    if date_roles.correction_from == date_roles.correction_to
                    else None
                ))
        elif counterparty is not None:
            selected_pair = self._select_external_issuer(
                semantic_intent, counterparty, cutoff)
        if selected_pair is None and company is not None:
            literal_dates = _date_surfaces(question)
            # ``<exact day> 정정공시의 ... 전체 이력`` expands the requested
            # range from that filing through the corpus cutoff.  The range
            # endpoint must not erase the exact filing day that identifies a
            # root-missing correction seed.  Reuse it only when the question
            # contains exactly one literal day; wider or multi-date ranges
            # remain unresolved.
            correction_seed_on = (
                date_roles.correction_from
                if (date_roles.correction_from is not None
                    and len(literal_dates) == 1
                    and literal_dates[0] == date_roles.correction_from)
                else None
            )
            missing = self._select_root_missing_correction(
                company, semantic_intent, question=question, cutoff=cutoff,
                correction_observed_on=correction_seed_on,
            )
            if missing is not None:
                return CorrectionLineageResult("resolved", missing)
        if selected_pair is None:
            return CorrectionLineageResult("not_found")
        company, selected = selected_pair
        corp_code, corp_name = getattr(company, "corp_code", None), getattr(company, "corp_name", None)
        if not isinstance(corp_code, str) or not isinstance(corp_name, str):
            return CorrectionLineageResult("unavailable")
        candidate = selected.candidates[0]
        timeline = self._canonical.event_timeline(
            as_of=cutoff, event_key=candidate.event_key, verify_evidence=True)
        if (timeline is None or getattr(timeline, "event_key", None) != candidate.event_key
                or getattr(timeline, "corp_code", None) != corp_code
                or getattr(timeline, "corp_name", None) != corp_name
                or getattr(timeline, "root_rcept_no", None) != candidate.seed_rcept_no):
            return CorrectionLineageResult("unavailable")
        root_observations = [
            row for row in tuple(getattr(timeline, "observations", ()) or ())
            if getattr(row, "rcept_no", None) == candidate.seed_rcept_no
        ]
        if len(root_observations) != 1:
            return CorrectionLineageResult("ambiguous")
        if (date_roles.root_observed_at is not None
                and getattr(root_observations[0], "observed_at", None)
                != date_roles.root_observed_at):
            return CorrectionLineageResult("not_found")
        corrections = [row for row in tuple(getattr(timeline, "observations", ()) or ())
                       if bool(getattr(row, "is_correction", False))]
        if date_roles.correction_from is not None:
            corrections = [row for row in corrections
                           if str(getattr(row, "observed_at", ""))
                           >= date_roles.correction_from]
        if date_roles.correction_to is not None:
            corrections = [row for row in corrections
                           if str(getattr(row, "observed_at", ""))
                           <= date_roles.correction_to]
        dates = _date_surfaces(question)
        if (dates and date_roles.correction_from is None
                and date_roles.root_observed_at is None):
            corrections = [row for row in corrections
                           if str(getattr(row, "observed_at", "")) in dates]
        if not corrections:
            return CorrectionLineageResult("not_found")
        if len(corrections) != 1 and not date_roles.range_requested:
            return CorrectionLineageResult("ambiguous" if corrections else "not_found")
        corrections = sorted(corrections, key=lambda row: (
            str(getattr(row, "observed_at", "")),
            int(getattr(row, "seq", 0)), str(getattr(row, "rcept_no", ""))))
        if len({getattr(row, "rcept_no", None) for row in corrections}) != len(corrections):
            return CorrectionLineageResult("ambiguous")
        # Multiple same-day corrections require explicit previous links; a
        # receipt number's order is not source evidence for temporal order.
        by_date: dict[str, list[Any]] = {}
        for row in corrections:
            by_date.setdefault(str(getattr(row, "observed_at", "")), []).append(row)
        for same_day in by_date.values():
            if len(same_day) > 1:
                receipts = {getattr(row, "rcept_no", None) for row in same_day}
                linked = sum(
                    1 for row in same_day
                    if getattr(row, "previous_observation_rcept_no", None) in receipts)
                if linked != len(same_day) - 1:
                    return CorrectionLineageResult("ambiguous")

        receipts = tuple(str(getattr(row, "rcept_no", "")) for row in corrections)
        try:
            rows = tuple(self._canonical.correction_items(
                as_of=cutoff,
                **({"rcept_no": receipts[0]} if len(receipts) == 1
                   else {"rcept_nos": receipts}),
                include_restricted_raw=False, verify_evidence=True))
        except TypeError:
            rows = tuple(item for receipt in receipts
                         for item in self._canonical.correction_items(
                             as_of=cutoff, rcept_no=receipt,
                             include_restricted_raw=False, verify_evidence=True))
        rows_by_receipt: dict[str, list[Any]] = {receipt: [] for receipt in receipts}
        for row in rows:
            receipt = getattr(row, "rcept_no", None)
            if receipt not in rows_by_receipt:
                return CorrectionLineageResult("unavailable")
            rows_by_receipt[receipt].append(row)
        steps: list[CorrectionSequenceStep] = []
        for correction in corrections:
            correction_receipt = getattr(correction, "rcept_no", None)
            correction_date = getattr(correction, "observed_at", None)
            if (not isinstance(correction_receipt, str)
                    or not _RECEIPT.fullmatch(correction_receipt)
                    or not isinstance(correction_date, str)
                    or not _DATE.fullmatch(correction_date)):
                return CorrectionLineageResult("unavailable")
            proofs = _verified_event_proofs(timeline, correction_receipt)
            source_rows = rows_by_receipt[correction_receipt]
            changes = tuple(
                change for row in source_rows
                if (change := _field_change(row)) is not None)
            if not proofs or (source_rows and not changes):
                return CorrectionLineageResult("unavailable")
            changes = self._question_selected_changes(question, changes)
            if not changes:
                continue
            steps.append(CorrectionSequenceStep(
                correction_receipt=correction_receipt,
                correction_date=correction_date,
                previous_observation_receipt=getattr(
                    correction, "previous_observation_rcept_no", None),
                event_proofs=proofs,
                changes=changes,
            ))
        if not steps:
            return CorrectionLineageResult("not_found")
        event_proofs = tuple(proof for step in steps for proof in step.event_proofs)
        changes = tuple(change for step in steps for change in step.changes)
        correction_receipt = steps[-1].correction_receipt
        correction_date = steps[-1].correction_date
        if candidate.seed_rcept_no >= correction_receipt:
            # A correction cannot prove itself to be an earlier original.
            # Preserve the ambiguous-origin path for document/event backends
            # instead of constructing an invalid resolved lineage.
            return CorrectionLineageResult("ambiguous")
        premises = _question_premises(question, semantic_intent)
        if kind == "correction_diff":
            return CorrectionLineageResult("resolved", CorrectionDiffResolution(
                kind="correction_diff", issuer_corp_code=corp_code, issuer_corp_name=corp_name,
                event_key=candidate.event_key, root_receipt=candidate.seed_rcept_no,
                correction_receipt=correction_receipt, correction_date=correction_date,
                event_proofs=event_proofs, changes=changes, question_premises=premises,
                counterparty=counterparty,
                product_keywords=_contract_product_keywords(candidate.label),
                date_roles=date_roles,
                sequence=tuple(steps)))
        amount_rows = tuple(change for change in changes if _AMOUNT_PATH.search(change.path))
        if len(amount_rows) != 1:
            return CorrectionLineageResult("ambiguous" if amount_rows else "not_found")
        return CorrectionLineageResult("resolved", ContractAmountHistoryResolution(
            kind="contract_amount_history", issuer_corp_code=corp_code,
            issuer_corp_name=corp_name, event_key=candidate.event_key,
            root_receipt=candidate.seed_rcept_no, correction_receipt=correction_receipt,
            correction_date=correction_date, event_proofs=event_proofs,
            amount_change=amount_rows[0], question_premises=premises,
            counterparty=counterparty,
            product_keywords=_contract_product_keywords(candidate.label)))

    def _explicit_receipt_history(self, company: Any, *, question: str,
                                  intent: SemanticIntent, cutoff: str):
        receipts = tuple(dict.fromkeys(_EXPLICIT_RECEIPT.findall(question)))
        if (len(receipts) != 1 or len(intent.answer_items) != 1
                or intent.answer_items[0].output.projection_mode != "whole_target"
                or "계약" not in question or "정정" not in question
                or not (re.search(r"처음.*최신", question)
                        or re.search(r"실제로.*(?:접수번호별|한\s*번씩|세어)", question))):
            return None
        until = re.search(r"(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일까지", question)
        if until:
            cutoff = min(cutoff, f"{int(until[1]):04}{int(until[2]):02}{int(until[3]):02}")
        timeline = self._canonical.event_timeline(
            as_of=cutoff, rcept_no=receipts[0], verify_evidence=True)
        if (timeline is None or timeline.corp_code != company.corp_code
                or not any(o.rcept_no == receipts[0] for o in timeline.observations)):
            return CorrectionLineageResult("not_found")
        observations = sorted(timeline.observations, key=lambda o: (o.observed_at, o.seq, o.rcept_no))
        roots = [o for o in observations if o.rcept_no == timeline.root_rcept_no]
        if len(roots) != 1 or not roots[0].is_correction:
            return CorrectionLineageResult("ambiguous")
        # Same-day ordering is not established by receipt-number sorting.
        # Require the public canonical predecessor links before aggregating.
        days: dict[str, list[Any]] = {}
        for observation in observations:
            days.setdefault(observation.observed_at, []).append(observation)
        for same_day in days.values():
            day_receipts = {o.rcept_no for o in same_day}
            if len(day_receipts) != len(same_day):
                return CorrectionLineageResult("ambiguous")
            if len(same_day) > 1 and sum(
                    o.previous_observation_rcept_no in day_receipts
                    for o in same_day) != len(same_day) - 1:
                return CorrectionLineageResult("ambiguous")
        steps = []
        for observation in observations:
            if not observation.is_correction or observation.observed_at > cutoff:
                continue
            source_rows = tuple(self._canonical.correction_items(
                as_of=cutoff, rcept_no=observation.rcept_no,
                include_restricted_raw=False, verify_evidence=True))
            changes = tuple(change for row in source_rows if (change := _field_change(row)) is not None)
            changes = self._question_selected_changes(question, changes)
            if not changes:
                continue
            proofs = _verified_event_proofs(timeline, observation.rcept_no)
            if not proofs:
                side = next((p for c in changes for p in (c.before_proof, c.after_proof) if p), None)
                if side is None:
                    return CorrectionLineageResult("unavailable")
                proofs = (CanonicalProof(
                    proof_ref=f"canonical:event:{timeline.event_key}:{observation.rcept_no}",
                    source_receipt=observation.rcept_no, evidence_id=side.evidence_id),)
            steps.append(CorrectionSequenceStep(
                correction_receipt=observation.rcept_no, correction_date=observation.observed_at,
                previous_observation_receipt=observation.previous_observation_rcept_no,
                event_proofs=proofs, changes=changes))
        if not steps:
            return CorrectionLineageResult("not_found")
        return CorrectionLineageResult("resolved", CorrectionDiffResolution(
            kind="correction_diff", issuer_corp_code=company.corp_code, issuer_corp_name=company.corp_name,
            event_key=timeline.event_key, root_receipt=timeline.root_rcept_no,
            correction_receipt=steps[-1].correction_receipt, correction_date=steps[-1].correction_date,
            event_proofs=tuple(p for step in steps for p in step.event_proofs),
            changes=tuple(c for step in steps for c in step.changes), question_premises=(),
            source_root_missing=roots[0].is_correction,
            date_roles=CorrectionDateRoles(root_observed_at=None, correction_from=None,
                                          correction_to=cutoff, as_of=cutoff, range_requested=True),
            sequence=tuple(steps)))

    @staticmethod
    def _document_family_matches_question(document: Any, question: str) -> bool:
        """Require the selected correction form to be named by the user."""

        question_key = re.sub(r"[^0-9a-z가-힣]+", "", question.casefold())
        form_key = re.sub(
            r"[^0-9a-z가-힣]+", "",
            str(getattr(document, "form", "") or "").casefold())
        if len(form_key) >= 3 and form_key in question_key:
            return True
        report = re.sub(
            r"\[(?:기재)?정정\]|주요사항보고서|정정공시|공시", "",
            str(getattr(document, "report_nm", "") or ""),
            flags=re.IGNORECASE)
        report_key = re.sub(r"[^0-9a-z가-힣]+", "", report.casefold())
        if len(report_key) >= 3 and report_key in question_key:
            return True
        # A user may identify a correction only by issuer and an exact public
        # date/receipt ("2025년 4월 3일 정정공시").  Admit that generic surface
        # here; the caller still requires exactly one correction document on
        # the date, so same-day siblings remain ambiguous and fail closed.
        # Keeping the coordinate adjacent to ``정정공시`` also prevents this
        # fallback from overriding an explicitly named but mismatching form.
        #
        # **A span question names the same coordinate but does not select one
        # document.**  "2023년 5월 31일 정정공시와 마지막 정정공시 사이에 …"
        # uses the date as one endpoint of a lineage.  Admitting the generic
        # surface there matches every correction in the chain, the caller sees
        # more than one and fails closed, and a question that used to answer
        # with the whole lineage becomes a refusal.  Stand down for spans and
        # let the lineage route keep it.
        if _CORRECTION_SPAN.search(question):
            return False
        return re.search(
            r"(?:(?:접수\s*번호\s*)?20[0-9]{12}|"
            r"[0-9]{4}\s*년\s*[0-9]{1,2}\s*월\s*[0-9]{1,2}\s*일)\s*"
            r"(?:기재\s*)?정정\s*공시",
            question,
        ) is not None

    @staticmethod
    def _question_selected_changes(
            question: str, changes: tuple[CorrectionFieldChange, ...],
            ) -> tuple[CorrectionFieldChange, ...]:
        """Narrow an exact correction to a literally requested field path.

        A broad ``변경된 내용`` request keeps every verified row.  Otherwise
        an exact canonical leaf already present in the question is stronger
        than a longer neighbouring path such as ``합병비율 산출근거``.
        """

        if _EXPLICIT_ALL_CHANGES.search(question):
            return changes
        question_key = re.sub(r"[^0-9a-z가-힣%]+", "", question.casefold())
        path_keys: list[tuple[CorrectionFieldChange, tuple[str, ...]]] = []
        matched: list[CorrectionFieldChange] = []
        for change in changes:
            raw_segments = tuple(
                re.sub(r"^\s*[0-9]+\.\s*", "", segment).strip()
                for segment in change.path.split(">")
            )
            segments = tuple(
                re.sub(
                    r"[^0-9a-z가-힣%]+", "", segment.casefold(),
                )
                for segment in raw_segments
            )
            path_keys.append((change, segments))
            leaf = raw_segments[-1] if raw_segments else ""
            # Unit suffixes are display coordinates, not sibling field names.
            # Let ``계약금액`` literally select ``계약금액(원)`` before the
            # parent-prefix route could also include ``매출액대비(%)``.
            leaf_without_unit = re.sub(
                r"\(\s*(?:%|원|천원|백만원|십억원|억원|조원|주|건)\s*\)\s*$",
                "", leaf, flags=re.IGNORECASE,
            )
            leaf_keys = {
                re.sub(r"[^0-9a-z가-힣%]+", "", value.casefold())
                for value in (leaf, leaf_without_unit)
            }
            if any(len(key) >= 2 and key in question_key for key in leaf_keys):
                matched.append(change)
        # A literal field path is stronger than a generic question word.  In
        # particular ``산출근거가 무엇으로 변경됐나`` contains ``무엇`` but is
        # still a request for one named field, not every row in the filing.
        if matched:
            return tuple(matched)
        # A public section/theme is sometimes requested instead of a literal
        # correction-table leaf.  Keep this mapping form-semantic and bounded:
        # it applies to every rights-offering filing and contains no issuer,
        # receipt, date, or expected value.  Otherwise an issuance/funding
        # question expands to unrelated schedule and subscription changes.
        topic_path_patterns: list[re.Pattern[str]] = []
        if re.search(r"자금\s*조달(?:의)?\s*목적", question):
            topic_path_patterns.append(
                re.compile(r"자금\s*조달(?:의)?\s*목적"))
        if (re.search(r"발행\s*내용", question)
                and re.search(r"유상\s*증자|신주", question)):
            topic_path_patterns.extend((
                re.compile(r"^(?:[0-9]+\.\s*)?신주의\s*종류와\s*수(?:\s*>|$)"),
                re.compile(r"^(?:[0-9]+\.\s*)?신주\s+발행가액(?:\s*>|$)"),
                re.compile(r"^(?:[0-9]+\.\s*)?증자\s*방식(?:\s*>|$)"),
            ))
        if topic_path_patterns:
            return tuple(
                change for change in changes
                if any(pattern.search(change.path)
                       for pattern in topic_path_patterns))
        # A BR/nested-table split turns one public parent field into several
        # scalar children (for example ``계약기간 > 시작일/종료일``).
        # When the question names that parent rather than one child, retain all
        # and only its descendants.  Prefer the longest literal parent segment:
        # a shared generic ancestor such as ``계약`` must not pull sibling
        # ``계약금액`` rows into a narrower ``계약기간`` request.
        parent_scores: list[tuple[CorrectionFieldChange, int]] = []
        for change, segments in path_keys:
            score = max(
                (len(key) for key in segments[:-1]
                 if len(key) >= 2 and key in question_key),
                default=0,
            )
            if score:
                parent_scores.append((change, score))
        if parent_scores:
            best = max(score for _change, score in parent_scores)
            return tuple(
                change for change, score in parent_scores if score == best)
        if _BROAD_CHANGE_REQUEST.search(question):
            return changes
        return changes

    def _select_exact_correction_document(
            self, company: Any, *, question: str, intent: SemanticIntent,
            cutoff: str) -> CorrectionLineageResult | None:
        """Bind one explicitly dated/numbered correction, fail-closed.

        This path is independent of event-name inference.  Issuer, correction
        polarity, document family, exact date/receipt, timeline observation,
        and before/after evidence all have to agree in canonical metadata.
        """

        documents = getattr(self._canonical, "documents", None)
        if not callable(documents):
            return None
        corp_code = getattr(company, "corp_code", None)
        corp_name = getattr(company, "corp_name", None)
        if not isinstance(corp_code, str) or not isinstance(corp_name, str):
            return CorrectionLineageResult("unavailable")
        receipts = tuple(dict.fromkeys(_EXPLICIT_RECEIPT.findall(question)))
        dates = _date_surfaces(question)
        # Receipt digits must not be reinterpreted as a second date axis.
        exact_day = receipts[0][:8] if len(receipts) == 1 else (
            dates[0] if len(dates) == 1 else None)
        if (len(receipts) > 1 or (not receipts and len(dates) != 1)
                or exact_day is None or exact_day > cutoff):
            return None
        try:
            candidates = [
                row for row in documents(
                    as_of=cutoff, corp_code=corp_code, is_correction=True)
                if (not receipts or getattr(row, "rcept_no", None) == receipts[0])
                and getattr(row, "rcept_dt", None) == exact_day
                and self._document_family_matches_question(row, question)
            ]
        except Exception:
            return CorrectionLineageResult("unavailable")
        if not candidates:
            return None
        if len(candidates) != 1:
            return CorrectionLineageResult("ambiguous")
        receipt = str(getattr(candidates[0], "rcept_no", ""))
        if not _RECEIPT.fullmatch(receipt):
            return CorrectionLineageResult("unavailable")
        try:
            timeline = self._canonical.event_timeline(
                as_of=cutoff, rcept_no=receipt, verify_evidence=True)
        except Exception:
            return CorrectionLineageResult("unavailable")
        observations = tuple(getattr(timeline, "observations", ()) or ())
        selected = [
            row for row in observations
            if getattr(row, "rcept_no", None) == receipt
            and bool(getattr(row, "is_correction", False))
        ]
        event_key = getattr(timeline, "event_key", None)
        root_receipt = getattr(timeline, "root_rcept_no", None)
        if (timeline is None or getattr(timeline, "corp_code", None) != corp_code
                or len(selected) != 1
                or not isinstance(event_key, str)
                or re.fullmatch(r"[0-9a-f]{32}", event_key) is None
                or not isinstance(root_receipt, str)
                or _RECEIPT.fullmatch(root_receipt) is None
                or root_receipt > receipt):
            return CorrectionLineageResult("unavailable")
        try:
            rows = tuple(self._canonical.correction_items(
                as_of=cutoff, rcept_no=receipt,
                include_restricted_raw=False, verify_evidence=True))
        except Exception:
            return CorrectionLineageResult("unavailable")
        changes = tuple(
            change for row in rows
            if (change := _field_change(row)) is not None)
        changes = self._question_selected_changes(question, changes)
        if not changes:
            return CorrectionLineageResult("unavailable")
        event_proofs = _verified_event_proofs(timeline, receipt)
        if not event_proofs:
            # Some exchange/major correction observations have no standalone
            # event-support cell.  A verified correction side still proves the
            # selected receipt; reuse its evidence id without claiming a
            # separate event fact.
            side_proof = next((
                proof for change in changes
                for proof in (change.before_proof, change.after_proof)
                if proof is not None), None)
            if side_proof is None:
                return CorrectionLineageResult("unavailable")
            event_proofs = (CanonicalProof(
                proof_ref=f"canonical:event:{event_key}:{receipt}",
                source_receipt=receipt,
                evidence_id=side_proof.evidence_id),)
        source_root_missing = root_receipt == receipt
        roles = CorrectionDateRoles(
            root_observed_at=None, correction_from=exact_day,
            correction_to=exact_day, as_of=cutoff, range_requested=False)
        return CorrectionLineageResult("resolved", CorrectionDiffResolution(
            kind="correction_diff", issuer_corp_code=corp_code,
            issuer_corp_name=corp_name, event_key=event_key,
            root_receipt=root_receipt, correction_receipt=receipt,
            correction_date=exact_day, event_proofs=event_proofs,
            changes=changes,
            question_premises=_question_premises(question, intent),
            date_roles=roles, source_root_missing=source_root_missing))

    def _select_for_company(
            self, company: Any, intent: SemanticIntent, counterparty: str | None,
            cutoff: str, *, root_observed_at: str | None = None,
            correction_observed_on: str | None = None,
            ) -> tuple[Any, EventKeyResolution] | None:
        code = getattr(company, "corp_code", None)
        if not isinstance(code, str):
            return None
        for attempt in _event_attempts(intent, counterparty):
            result = self._preflight.resolve_event_key(
                corp_code=code, as_of=cutoff,
                event_from=root_observed_at, event_to=root_observed_at,
                **attempt)
            if result.status == "resolved" and len(result.candidates) == 1:
                return company, result
            # A correction-lineage question supplies a stronger public
            # predicate than a bare counterparty: retain an otherwise
            # ambiguous candidate only when exactly one candidate actually
            # has a correction observation in the admitted corpus.  This is
            # canonical lineage evidence, not a question-id or fixture rule.
            if result.status == "ambiguous" and result.candidates:
                corrected = []
                for candidate in result.candidates:
                    timeline = self._canonical.event_timeline(
                        as_of=cutoff, event_key=candidate.event_key,
                        verify_evidence=True)
                    observations = tuple(getattr(timeline, "observations", ()) or ())
                    if any(
                            bool(getattr(row, "is_correction", False))
                            and (
                                correction_observed_on is None
                                or getattr(row, "observed_at", None)
                                == correction_observed_on
                            )
                            for row in observations):
                        corrected.append(candidate)
                if len(corrected) == 1:
                    return company, EventKeyResolution(
                        status="resolved", candidates=tuple(corrected))
        # A filing-family surface (for example "단일판매·공급계약" or
        # "유상증자 결정") is not an event attribute value, so the ordinary
        # role preflight may correctly return no candidate.  An explicit root
        # observation date can still select one business event through bounded
        # public metadata: issuer, exact date and the literal document family
        # must all agree, and the canonical timeline must round-trip to that
        # root receipt.  This path intentionally covers both exchange and
        # major-report events; periodic document lineages have a separate
        # resolver and are excluded below.
        documents = getattr(self._canonical, "documents", None)
        if root_observed_at is None or not callable(documents):
            return None
        family_surfaces = [
            item.target.surface for item in intent.answer_items
            if item.target.kind in {"event", "document"}
            and not re.fullmatch(
                r"(?:(?:기재)?정정)?\s*공시|정정",
                item.target.surface.strip(), flags=re.IGNORECASE)]
        family_keys = {
            re.sub(r"[^0-9a-z가-힣]+", "", re.sub(
                r"(?:기재)?정정공시|정정|공시", "", surface).casefold())
            for surface in family_surfaces}
        family_keys.discard("")
        if len(family_keys) != 1:
            return None
        family_key = next(iter(family_keys))
        candidates: dict[str, EventKeyCandidate] = {}
        try:
            for document in documents(
                    as_of=cutoff, corp_code=code, is_correction=False):
                observed_at = str(getattr(document, "rcept_dt", ""))
                if observed_at > root_observed_at:
                    break
                if observed_at != root_observed_at:
                    continue
                if getattr(document, "doc_group", None) not in {
                        "exchange", "major"}:
                    continue
                searchable = re.sub(
                    r"[^0-9a-z가-힣]+", "",
                    (f"{getattr(document, 'form', '')} "
                     f"{getattr(document, 'report_nm', '')}").casefold())
                if family_key not in searchable:
                    continue
                receipt = str(getattr(document, "rcept_no", ""))
                timeline = self._canonical.event_timeline(
                    as_of=cutoff, rcept_no=receipt, verify_evidence=False)
                event_key = getattr(timeline, "event_key", None)
                if (timeline is None or getattr(timeline, "corp_code", None) != code
                        or getattr(timeline, "root_rcept_no", None) != receipt
                        or not isinstance(event_key, str)
                        or re.fullmatch(r"[0-9a-f]{32}", event_key) is None):
                    continue
                candidates[event_key] = EventKeyCandidate(
                    event_key=event_key, seed_rcept_no=receipt,
                    label=str(getattr(document, "report_nm", "")) or family_surfaces[0],
                    identity_status=getattr(timeline, "identity_status", None),
                )
                if len(candidates) > 8:
                    return None
        except Exception:
            return None
        if len(candidates) == 1:
            return company, EventKeyResolution(
                status="resolved", candidates=tuple(candidates.values()))
        return None

    def _select_root_missing_correction(
            self, company: Any, intent: SemanticIntent, *, question: str,
            cutoff: str,
            correction_observed_on: str | None,
            ) -> RootMissingCorrectionResolution | None:
        """Select one correction with a proven pre-corpus missing root.

        The lookup is deliberately narrower than text retrieval: exact issuer,
        exact correction date, one literal document family, and one canonical
        ``CORRECTS`` relation must all agree.  No earliest-observation fallback
        is allowed because that would silently promote a correction to a root.
        """

        documents = getattr(self._canonical, "documents", None)
        relations = getattr(self._canonical, "relation_summaries", None)
        code = getattr(company, "corp_code", None)
        name = getattr(company, "corp_name", None)
        if (correction_observed_on is None or not callable(documents)
                or not callable(relations) or not isinstance(code, str)
                or not isinstance(name, str)):
            return None
        family_keys = {
            re.sub(r"[^0-9a-z가-힣]+", "", re.sub(
                r"(?:기재\s*)?정정\s*공시|정정|공시", "",
                item.target.surface, flags=re.IGNORECASE).casefold())
            for item in intent.answer_items
            if item.target.kind in {"event", "document"}
        }
        family_keys.discard("")
        if len(family_keys) != 1:
            return None
        family_key = next(iter(family_keys))
        # ``계약 정정공시`` is a literal DART family even though its normalized
        # Korean key has only two syllables.  It is safe here because issuer,
        # exact filing day, correction polarity, and one root-missing relation
        # must still close to exactly one document.  Other short/generic keys
        # stay rejected.
        if len(family_key) < 3 and family_key != "계약":
            return None
        matches: list[tuple[Any, Any]] = []
        try:
            rows = documents(
                as_of=cutoff, corp_code=code, is_correction=True)
            for document in rows:
                receipt = str(getattr(document, "rcept_no", ""))
                observed = str(getattr(document, "rcept_dt", ""))
                if (observed != correction_observed_on
                        or not _RECEIPT.fullmatch(receipt)):
                    continue
                searchable = re.sub(
                    r"[^0-9a-z가-힣]+", "",
                    (f"{getattr(document, 'form', '')} "
                     f"{getattr(document, 'report_nm', '')}").casefold())
                if family_key not in searchable:
                    continue
                relation_rows = [
                    relation for relation in relations(
                        source_rcept_no=receipt, as_of=cutoff)
                    if (getattr(relation, "relation_type", None) == "CORRECTS"
                        and getattr(relation, "resolution_status", None)
                        == "root_missing"
                        and getattr(relation, "dst_rcept_no", None) is None
                        and getattr(relation, "root_missing_reason", None)
                        == "submitted_before_corpus"
                        and isinstance(getattr(relation, "target_hint", None), str)
                        and getattr(relation, "target_hint").strip())
                ]
                if len(relation_rows) == 1:
                    matches.append((document, relation_rows[0]))
                elif len(relation_rows) > 1:
                    return None
        except Exception:
            return None
        if len(matches) != 1:
            return None
        document, relation = matches[0]
        relation_id = str(getattr(relation, "relation_id", ""))
        if not relation_id:
            return None
        return RootMissingCorrectionResolution(
            kind="root_missing_correction",
            issuer_corp_code=code,
            issuer_corp_name=name,
            correction_receipt=str(getattr(document, "rcept_no")),
            correction_date=correction_observed_on,
            relation_id=relation_id,
            relation_proof_ref=f"canonical:relation:{relation_id}",
            target_hint=str(getattr(relation, "target_hint")),
            root_missing_reason="submitted_before_corpus",
            question_premises=_question_premises(question, intent),
        )

    def _select_external_issuer(
            self, intent: SemanticIntent, counterparty: str, cutoff: str,
            ) -> tuple[Any, EventKeyResolution] | None:
        """Find one issuer/event pair only when canonical documents prove it.

        This is the same safe direction as the native selected-event backend:
        an external counterparty is not upgraded to an issuer.  We enumerate
        canonical issuer coordinates and retain a pair only when the standard
        event preflight resolves exactly one event for exactly one issuer.
        """

        documents = getattr(self._canonical, "documents", None)
        if not callable(documents):
            return None
        companies: dict[str, Any] = {}
        try:
            rows = documents(as_of=cutoff)
            for row in rows:
                code = getattr(row, "corp_code", None)
                name = getattr(row, "corp_name", None)
                if isinstance(code, str) and isinstance(name, str) and code and name:
                    companies.setdefault(code, type("_Company", (), {
                        "corp_code": code, "corp_name": name})())
        except Exception:  # canonical read failures do not create a candidate
            return None
        matches = [
            selected for company in companies.values()
            if (selected := self._select_for_company(
                company, intent, counterparty, cutoff)) is not None
        ]
        return matches[0] if len(matches) == 1 else None


__all__ = [
    "CanonicalCorrectionLineageSelector", "CanonicalProof", "CorrectionDiffResolution",
    "CorrectionFieldChange", "CorrectionLineageResult", "ContractAmountHistoryResolution",
    "RootMissingCorrectionResolution",
]
