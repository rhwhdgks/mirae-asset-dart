"""Fail-closed canonical money observations for event amount changes.

Selecting an event proves identity only.  It does not prove that a requested
amount existed, was public and numeric, or used a compatible money unit at two
observation points.  This preflight closes those independent conditions before
Stage1 may emit an executable difference plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
import unicodedata
from typing import Any, Literal

from src.canonical.events import event_support_role


EventAmountObservationStatus = Literal[
    "resolved",
    "event_not_disclosed",
    "ownership_mismatch",
    "amount_missing",
    "amount_unavailable",
    "amount_ambiguous",
    "evidence_unavailable",
    "numeric_invalid",
    "unit_unsupported",
    "unit_mismatch",
]

_RECEIPT = re.compile(r"^[0-9]{14}$")
_DATE = re.compile(r"^[0-9]{8}$")
_EVIDENCE = re.compile(r"^[0-9a-f]{32}$")
_NUMBER = re.compile(r"^[+-]?(?:0|[1-9][0-9]{0,39})(?:\.[0-9]+)?$")


@dataclass(frozen=True, slots=True)
class EventAmountObservation:
    timepoint: str
    source_receipt: str
    evidence_id: str
    normalized_value: str
    currency: Literal["KRW"] = "KRW"
    unit: Literal["원"] = "원"
    scale: Literal[1] = 1

    def __post_init__(self) -> None:
        if not _DATE.fullmatch(self.timepoint):
            raise ValueError("event amount observation timepoint 형식이 잘못되었습니다")
        if (not _RECEIPT.fullmatch(self.source_receipt)
                or self.source_receipt[:8] > self.timepoint):
            raise ValueError("event amount observation source receipt가 시점보다 늦습니다")
        if not _EVIDENCE.fullmatch(self.evidence_id):
            raise ValueError("event amount observation evidence ID가 잘못되었습니다")
        if not _NUMBER.fullmatch(self.normalized_value):
            raise ValueError("event amount observation numeric value가 잘못되었습니다")

    @property
    def proof_ref(self) -> str:
        return f"canonical:field:{self.evidence_id}"


@dataclass(frozen=True, slots=True)
class EventAmountObservationResult:
    status: EventAmountObservationStatus
    observations: tuple[EventAmountObservation, ...] = ()

    def __post_init__(self) -> None:
        if self.status == "resolved":
            if len(self.observations) != 2:
                raise ValueError("resolved amount observation에는 두 시점이 필요합니다")
            first, second = self.observations
            if (first.timepoint >= second.timepoint
                    or (first.currency, first.unit, first.scale)
                    != (second.currency, second.unit, second.scale)):
                raise ValueError("resolved amount observation 시점/단위가 다릅니다")
        elif self.observations:
            raise ValueError("non-resolved amount observation은 값을 노출하지 않습니다")


def _leaf_key(path: str) -> str:
    leaf = re.split(r"\s*>\s*", unicodedata.normalize("NFKC", path or ""))[-1]
    leaf = re.sub(r"^\s*[-–—]?\s*\d+(?:[-.)])?\s*", "", leaf)
    return re.sub(r"[\s\[\](){}<>._,:;·ㆍᆞ/\\-]+", "", leaf).casefold()


def _expected_leaf(requested_slot: str) -> str:
    if requested_slot == "계약금액":
        return "계약금액원"
    if requested_slot == "해지금액":
        return "해지금액원"
    raise ValueError("event amount requested slot이 닫힌 vocabulary 밖입니다")


def _numeric(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    compact = value.strip().replace(",", "")
    if not _NUMBER.fullmatch(compact):
        return None
    try:
        parsed = Decimal(compact)
    except InvalidOperation:
        return None
    if not parsed.is_finite():
        return None
    canonical = format(parsed, "f")
    if "." in canonical:
        canonical = canonical.rstrip("0").rstrip(".")
    return "0" if canonical in {"-0", "+0", ""} else canonical


class EventAmountObservationPreflight:
    """Resolve two proven KRW observations for one canonical event."""

    def __init__(self, canonical: Any) -> None:
        if not callable(getattr(canonical, "event_timeline", None)) \
                or not callable(getattr(canonical, "fields", None)):
            raise TypeError("event amount canonical read 계약이 잘못되었습니다")
        self.canonical = canonical

    def _one(
            self, *, event_key: str, root_receipt: str, corp_code: str,
            corp_name: str, timepoint: str, requested_slot: str,
            ) -> EventAmountObservation | EventAmountObservationStatus:
        timeline = self.canonical.event_timeline(
            as_of=timepoint, event_key=event_key, verify_evidence=True)
        if timeline is None:
            return "event_not_disclosed"
        if (getattr(timeline, "event_key", None) != event_key
                or getattr(timeline, "root_rcept_no", None) != root_receipt
                or getattr(timeline, "corp_code", None) != corp_code
                or getattr(timeline, "corp_name", None) != corp_name):
            return "ownership_mismatch"

        expected_leaf = _expected_leaf(requested_slot)
        observations = tuple(getattr(timeline, "observations", ()) or ())
        for observation in reversed(observations):
            receipt = getattr(observation, "rcept_no", None)
            if not isinstance(receipt, str) or not _RECEIPT.fullmatch(receipt):
                return "evidence_unavailable"
            rows = [
                field for field in self.canonical.fields(
                    as_of=timepoint, rcept_no=receipt,
                    doc_group="exchange", is_pii=False,
                    include_restricted_raw=False)
                if _leaf_key(getattr(field, "path", "")) == expected_leaf
            ]
            if not rows:
                continue
            if len(rows) != 1:
                return "amount_ambiguous"
            field = rows[0]
            if getattr(field, "value_status", None) not in {
                    "literal", "explicit_zero"}:
                return "amount_unavailable"
            if event_support_role(
                    getattr(field, "path", ""),
                    value_status=getattr(field, "value_status", None),
                    is_pii=getattr(field, "is_pii", None)) != "amount":
                return "evidence_unavailable"
            # Only exact won fields are accepted.  Converting a table/header
            # unit or a foreign currency belongs to a separately versioned
            # normalization contract, not to this resolver.
            if _leaf_key(getattr(field, "path", "")) != expected_leaf:
                return "unit_unsupported"
            value = _numeric(getattr(field, "value", None))
            if value is None:
                return "numeric_invalid"
            evidence_id = getattr(field, "evidence_id", None)
            support_pairs = set(zip(
                tuple(getattr(observation, "supporting_evidence_ids", ()) or ()),
                tuple(getattr(observation, "support_roles", ()) or ()),
            ))
            if (not isinstance(evidence_id, str)
                    or not _EVIDENCE.fullmatch(evidence_id)
                    or (evidence_id, "amount") not in support_pairs
                    or getattr(observation, "evidence_verification_status", None)
                    != "verified"):
                return "evidence_unavailable"
            return EventAmountObservation(
                timepoint=timepoint, source_receipt=receipt,
                evidence_id=evidence_id, normalized_value=value)
        return "amount_missing"

    def resolve_pair(
            self, *, event_key: str, root_receipt: str, corp_code: str,
            corp_name: str, timepoints: list[str] | tuple[str, str],
            requested_slot: str,
            ) -> EventAmountObservationResult:
        if (len(timepoints) != 2 or timepoints[0] >= timepoints[1]
                or any(not isinstance(value, str)
                       or not _DATE.fullmatch(value) for value in timepoints)):
            raise ValueError("event amount observation timepoint 계약이 잘못되었습니다")
        resolved: list[EventAmountObservation] = []
        for timepoint in timepoints:
            observation = self._one(
                event_key=event_key, root_receipt=root_receipt,
                corp_code=corp_code, corp_name=corp_name,
                timepoint=timepoint, requested_slot=requested_slot)
            if isinstance(observation, str):
                return EventAmountObservationResult(observation)
            resolved.append(observation)
        first, second = resolved
        if (first.currency, first.unit, first.scale) != (
                second.currency, second.unit, second.scale):
            return EventAmountObservationResult("unit_mismatch")
        return EventAmountObservationResult("resolved", tuple(resolved))


__all__ = [
    "EventAmountObservation",
    "EventAmountObservationPreflight",
    "EventAmountObservationResult",
    "EventAmountObservationStatus",
]
