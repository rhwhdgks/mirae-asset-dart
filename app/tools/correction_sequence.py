"""Typed, bounded correction history assembly for one or more events.

The public QueryPlan v0.4 fields are interpreted without changing the wire
schema: ``EventSelector.event_from/event_to`` select the original observation,
``DocumentSelector.rcept_from/rcept_to`` bound correction observations, and
``ResolvedCorrectionTask.as_of`` is the immutable corpus cutoff.  These roles
must never be substituted for one another.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


_DATE = re.compile(r"^[0-9]{8}$")
_RECEIPT = re.compile(r"^[0-9]{14}$")
_MAX_CORRECTION_DOCS = 128
_MAX_EVENTS = 64
_MAX_CHANGES = 4096


@dataclass(frozen=True, slots=True)
class CorrectionDateRoles:
    root_observed_at: str | None
    correction_from: str | None
    correction_to: str
    as_of: str

    def __post_init__(self) -> None:
        values = tuple(value for value in (
            self.root_observed_at, self.correction_from,
            self.correction_to, self.as_of) if value is not None)
        if any(_DATE.fullmatch(value) is None for value in values):
            raise ValueError("correction date role은 YYYYMMDD여야 합니다")
        if self.correction_from and self.correction_from > self.correction_to:
            raise ValueError("correction_from은 correction_to 이후일 수 없습니다")
        if self.correction_to > self.as_of:
            raise ValueError("correction_to는 as_of 이후일 수 없습니다")
        if self.root_observed_at and self.root_observed_at > self.as_of:
            raise ValueError("root_observed_at은 as_of 이후일 수 없습니다")

    @classmethod
    def from_task(cls, task) -> "CorrectionDateRoles":
        event_selector = getattr(task, "event_selector", None)
        document_selector = getattr(task, "document_selector", None)
        event_from = getattr(event_selector, "event_from", None)
        event_to = getattr(event_selector, "event_to", None)
        if (event_from is None) != (event_to is None):
            raise ValueError("root observation은 exact date 또는 미지정이어야 합니다")
        if event_from is not None and event_from != event_to:
            raise ValueError("root_observed_at을 날짜 범위로 대체할 수 없습니다")
        correction_from = getattr(document_selector, "rcept_from", None)
        correction_to = getattr(document_selector, "rcept_to", None)
        as_of = getattr(task, "as_of", None)
        if not isinstance(as_of, str):
            raise ValueError("correction task as_of가 없습니다")
        return cls(
            root_observed_at=event_from,
            correction_from=correction_from,
            correction_to=correction_to or as_of,
            as_of=as_of,
        )


@dataclass(frozen=True, slots=True)
class CorrectionChange:
    path: str
    order: int
    reason: str | None
    before: str | None
    after: str | None
    before_evidence_id: str | None
    after_evidence_id: str | None
    row: object

    def __post_init__(self) -> None:
        if not self.path.strip() or type(self.order) is not int or self.order < 0:
            raise ValueError("correction change coordinate가 잘못되었습니다")
        if self.before is not None and self.before_evidence_id is None:
            raise ValueError("before 값에는 검증 Evidence가 필요합니다")
        if self.after is not None and self.after_evidence_id is None:
            raise ValueError("after 값에는 검증 Evidence가 필요합니다")
        if self.before_evidence_id is None and self.after_evidence_id is None:
            raise ValueError("correction change에는 한쪽 이상의 Evidence가 필요합니다")


@dataclass(frozen=True, slots=True)
class CorrectionStep:
    rcept_no: str
    observed_at: str
    previous_observation_rcept_no: str | None
    event_evidence_ids: tuple[str, ...]
    changes: tuple[CorrectionChange, ...]


@dataclass(frozen=True, slots=True)
class CorrectionSequence:
    event_key: str
    corp_code: str
    root_receipt: str
    root_observed_at: str
    identity_status: str
    roles: CorrectionDateRoles
    steps: tuple[CorrectionStep, ...]


SequenceStatus = Literal[
    "resolved", "partial", "not_found", "ambiguous", "unavailable",
]


@dataclass(frozen=True, slots=True)
class CorrectionSequenceResult:
    status: SequenceStatus
    sequences: tuple[CorrectionSequence, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if (self.status in {"resolved", "partial"}) != bool(self.sequences):
            raise ValueError("correction sequence result 상태가 잘못되었습니다")
        if self.status == "partial" and not self.reason:
            raise ValueError("partial correction sequence에는 limitation이 필요합니다")


def _ordered_same_day(rows: list[object]) -> list[object] | None:
    """Use explicit previous links; receipt order is not temporal evidence."""

    if len(rows) < 2:
        return rows
    by_receipt = {getattr(row, "rcept_no", None): row for row in rows}
    if len(by_receipt) != len(rows) or None in by_receipt:
        return None
    children: dict[str, list[object]] = {}
    starts: list[object] = []
    for row in rows:
        previous = getattr(row, "previous_observation_rcept_no", None)
        if previous in by_receipt:
            children.setdefault(previous, []).append(row)
        else:
            starts.append(row)
    if len(starts) != 1 or any(len(value) != 1 for value in children.values()):
        return None
    ordered, current = [], starts[0]
    while current is not None:
        ordered.append(current)
        next_rows = children.get(getattr(current, "rcept_no"), [])
        current = next_rows[0] if next_rows else None
        if len(ordered) > len(rows):
            return None
    return ordered if len(ordered) == len(rows) else None


def _ordered_corrections(observations: tuple[object, ...]) -> list[object] | None:
    grouped: dict[str, list[object]] = {}
    for row in observations:
        if bool(getattr(row, "is_correction", False)):
            grouped.setdefault(str(getattr(row, "observed_at", "")), []).append(row)
    ordered: list[object] = []
    for observed_at in sorted(grouped):
        if _DATE.fullmatch(observed_at) is None:
            return None
        same_day = _ordered_same_day(grouped[observed_at])
        if same_day is None:
            return None
        ordered.extend(same_day)
    return ordered


def _verified_change(row: object) -> CorrectionChange | None:
    before_verified = (
        getattr(row, "before_evidence_status", None) == "verified"
        and isinstance(getattr(row, "before_evidence_id", None), str))
    after_verified = (
        getattr(row, "after_evidence_status", None) == "verified"
        and isinstance(getattr(row, "after_evidence_id", None), str))
    if not before_verified and not after_verified:
        return None
    path = getattr(row, "path", None)
    order = getattr(row, "order", None)
    if not isinstance(path, str) or type(order) is not int:
        return None
    return CorrectionChange(
        path=path,
        order=order,
        reason=getattr(row, "reason", None),
        before=(getattr(row, "value_before", None) if before_verified else None),
        after=(getattr(row, "value_after", None) if after_verified else None),
        before_evidence_id=(
            getattr(row, "before_evidence_id", None) if before_verified else None),
        after_evidence_id=(
            getattr(row, "after_evidence_id", None) if after_verified else None),
        row=row,
    )


def build_event_correction_sequence(
        canonical, *, corp_code: str, event_key: str,
        roles: CorrectionDateRoles,
        verified_rows_by_receipt: dict[str, tuple[object, ...]] | None = None,
        ) -> CorrectionSequenceResult:
    """Build one exact event sequence using one grouped correction scan."""

    try:
        timeline = canonical.event_timeline(
            as_of=roles.as_of, event_key=event_key, verify_evidence=True)
    except Exception:
        return CorrectionSequenceResult("unavailable", reason="event_timeline_error")
    if (timeline is None or getattr(timeline, "event_key", None) != event_key
            or getattr(timeline, "corp_code", None) != corp_code):
        return CorrectionSequenceResult("not_found", reason="event_owner_mismatch")
    identity_status = getattr(timeline, "identity_status", None)
    if identity_status in {None, "ambiguous", "unavailable"}:
        return CorrectionSequenceResult("ambiguous", reason="event_identity_ambiguous")
    observations = tuple(getattr(timeline, "observations", ()) or ())
    roots = [row for row in observations
             if getattr(row, "rcept_no", None) == getattr(timeline, "root_rcept_no", None)]
    if len(roots) != 1:
        return CorrectionSequenceResult("ambiguous", reason="root_observation_ambiguous")
    root = roots[0]
    root_date = getattr(root, "observed_at", None)
    if not isinstance(root_date, str) or _DATE.fullmatch(root_date) is None:
        return CorrectionSequenceResult("unavailable", reason="root_date_invalid")
    if roles.root_observed_at is not None and root_date != roles.root_observed_at:
        return CorrectionSequenceResult("not_found", reason="root_date_mismatch")

    corrections = _ordered_corrections(observations)
    if corrections is None:
        return CorrectionSequenceResult("ambiguous", reason="same_day_order_unproven")
    corrections = [row for row in corrections
                   if (roles.correction_from is None
                       or getattr(row, "observed_at", "") >= roles.correction_from)
                   and getattr(row, "observed_at", "") <= roles.correction_to]
    if not corrections:
        return CorrectionSequenceResult("not_found", reason="correction_window_empty")
    receipts = tuple(getattr(row, "rcept_no", "") for row in corrections)
    if (len(receipts) > _MAX_CORRECTION_DOCS or len(receipts) != len(set(receipts))
            or any(_RECEIPT.fullmatch(value) is None for value in receipts)):
        return CorrectionSequenceResult("ambiguous", reason="correction_receipt_bound")
    if verified_rows_by_receipt is not None:
        rows = tuple(
            item for receipt in receipts
            for item in verified_rows_by_receipt.get(receipt, ()))
    else:
        try:
            rows = tuple(canonical.correction_items(
                as_of=roles.as_of, corp_code=corp_code, rcept_nos=receipts,
                include_restricted_raw=False, verify_evidence=True))
        except TypeError:
            # Compatibility for a test/dummy reader without grouped lookup.
            rows = tuple(item for receipt in receipts for item in canonical.correction_items(
                as_of=roles.as_of, corp_code=corp_code, rcept_no=receipt,
                include_restricted_raw=False, verify_evidence=True))
        except Exception:
            return CorrectionSequenceResult("unavailable", reason="correction_items_error")
    if len(rows) > _MAX_CHANGES:
        return CorrectionSequenceResult("ambiguous", reason="correction_change_bound")
    grouped: dict[str, list[object]] = {receipt: [] for receipt in receipts}
    coordinates: set[tuple[str, str, int]] = set()
    for row in rows:
        receipt = getattr(row, "rcept_no", None)
        if receipt not in grouped or getattr(row, "corp_code", corp_code) != corp_code:
            return CorrectionSequenceResult("unavailable", reason="correction_item_owner_mismatch")
        coordinate = (receipt, str(getattr(row, "path", "")),
                      int(getattr(row, "order", -1)))
        if coordinate in coordinates:
            return CorrectionSequenceResult("ambiguous", reason="correction_coordinate_duplicate")
        coordinates.add(coordinate)
        grouped[receipt].append(row)

    steps: list[CorrectionStep] = []
    omitted_steps: list[str] = []
    for observation in corrections:
        receipt = getattr(observation, "rcept_no")
        event_ids = tuple(getattr(observation, "supporting_evidence_ids", ()) or ())
        if (getattr(observation, "evidence_verification_status", None) != "verified"
                or not event_ids):
            # One unsupported correction must not erase later independently
            # verified changes in the same typed event lineage.
            omitted_steps.append(f"{receipt}:event_evidence_unverified")
            continue
        changes = tuple(filter(None, (
            _verified_change(row) for row in sorted(
                grouped[receipt], key=lambda item: (
                    getattr(item, "order", -1), getattr(item, "path", ""))))))
        if not changes:
            omitted_steps.append(f"{receipt}:change_evidence_unverified")
            continue
        steps.append(CorrectionStep(
            rcept_no=receipt,
            observed_at=getattr(observation, "observed_at"),
            previous_observation_rcept_no=getattr(
                observation, "previous_observation_rcept_no", None),
            event_evidence_ids=event_ids,
            changes=changes,
        ))
    if not steps:
        return CorrectionSequenceResult(
            "unavailable", reason="|".join(omitted_steps) or "change_evidence_unverified")
    sequence = CorrectionSequence(
        event_key=event_key,
        corp_code=corp_code,
        root_receipt=getattr(timeline, "root_rcept_no"),
        root_observed_at=root_date,
        identity_status=str(identity_status),
        roles=roles,
        steps=tuple(steps),
    )
    if omitted_steps:
        return CorrectionSequenceResult(
            "partial", (sequence,), "omitted_unverified_steps=" + "|".join(omitted_steps))
    return CorrectionSequenceResult("resolved", (sequence,))


def collect_correction_sequences(
        canonical, *, corp_code: str, roles: CorrectionDateRoles,
        doc_group: str | None = None, event_type: str | None = None,
        ) -> CorrectionSequenceResult:
    """Collect issuer correction events via bounded metadata then exact keys."""

    def key(value: str | None) -> str:
        return re.sub(r"[^0-9a-z가-힣]+", "", (value or "").casefold())

    try:
        documents = canonical.documents(
            as_of=roles.as_of, corp_code=corp_code,
            doc_group=doc_group, is_correction=True)
        receipts: list[str] = []
        for document in documents:
            date_value = getattr(document, "rcept_dt", "")
            if ((roles.correction_from and date_value < roles.correction_from)
                    or date_value > roles.correction_to):
                continue
            if event_type and key(event_type) not in key(
                    f"{getattr(document, 'form', '')} {getattr(document, 'report_nm', '')}"):
                continue
            receipts.append(getattr(document, "rcept_no", ""))
            if len(receipts) > _MAX_CORRECTION_DOCS:
                return CorrectionSequenceResult("ambiguous", reason="correction_document_bound")
    except Exception:
        return CorrectionSequenceResult("unavailable", reason="correction_document_lookup_error")
    if not receipts:
        return CorrectionSequenceResult("not_found", reason="correction_document_not_found")

    event_keys: set[str] = set()
    for receipt in receipts:
        try:
            timeline = canonical.event_timeline(
                as_of=roles.as_of, rcept_no=receipt, verify_evidence=False)
        except Exception:
            return CorrectionSequenceResult("ambiguous", reason="correction_event_mapping_ambiguous")
        if timeline is None or getattr(timeline, "corp_code", None) != corp_code:
            return CorrectionSequenceResult("unavailable", reason="correction_event_owner_mismatch")
        event_keys.add(getattr(timeline, "event_key", ""))
        if len(event_keys) > _MAX_EVENTS:
            return CorrectionSequenceResult("ambiguous", reason="correction_event_bound")

    # Verify every selected correction FK once before splitting them into event
    # lineages. This uses the reader's bounded rcept_nos batch path and avoids
    # one evidence/semantic scan per event on a cold collection request.
    verified_rows_by_receipt: dict[str, tuple[object, ...]] | None = None
    try:
        all_rows = tuple(canonical.correction_items(
            as_of=roles.as_of, corp_code=corp_code, rcept_nos=tuple(receipts),
            include_restricted_raw=False, verify_evidence=True))
        grouped_rows: dict[str, list[object]] = {receipt: [] for receipt in receipts}
        for row in all_rows:
            receipt = getattr(row, "rcept_no", None)
            if receipt not in grouped_rows:
                raise ValueError("correction batch returned an unselected receipt")
            grouped_rows[receipt].append(row)
        verified_rows_by_receipt = {
            receipt: tuple(rows) for receipt, rows in grouped_rows.items()}
    except Exception:
        # Keep the existing per-event path for older readers or a failed batch;
        # correctness remains fail-closed inside build_event_correction_sequence.
        verified_rows_by_receipt = None

    sequences: list[CorrectionSequence] = []
    skipped: list[str] = []
    for event_key in sorted(event_keys):
        result = build_event_correction_sequence(
            canonical, corp_code=corp_code, event_key=event_key, roles=roles,
            verified_rows_by_receipt=verified_rows_by_receipt)
        if result.status in {"resolved", "partial"}:
            sequences.extend(result.sequences)
            if result.status == "partial":
                skipped.append(f"{event_key}:{result.reason or result.status}")
        else:
            skipped.append(f"{event_key}:{result.reason or result.status}")
    if not sequences:
        return CorrectionSequenceResult(
            "ambiguous" if skipped else "not_found",
            reason="|".join(skipped) or "correction_sequence_not_found")
    if skipped:
        return CorrectionSequenceResult(
            "partial", tuple(sequences), "skipped_unresolved_events=" + "|".join(skipped))
    return CorrectionSequenceResult("resolved", tuple(sequences))


__all__ = [
    "CorrectionChange", "CorrectionDateRoles", "CorrectionSequence",
    "CorrectionSequenceResult", "CorrectionStep",
    "build_event_correction_sequence", "collect_correction_sequences",
]
