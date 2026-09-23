"""Correction diff용 exact receipt와 document-lineage preflight."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Protocol


CorrectionSeedStatus = Literal["resolved", "ambiguous", "not_found", "invalid"]


@dataclass(frozen=True, slots=True)
class CorrectionSeedCandidate:
    rcept_no: str
    label: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9]{14}", self.rcept_no) is None:
            raise ValueError("correction candidate 접수번호 형식이 잘못되었습니다")
        if not self.label.strip():
            raise ValueError("correction candidate label은 비어 있을 수 없습니다")


@dataclass(frozen=True, slots=True)
class CorrectionSeedResolution:
    status: CorrectionSeedStatus
    selected_receipt: str | None = None
    candidates: tuple[CorrectionSeedCandidate, ...] = ()

    def __post_init__(self) -> None:
        receipts = [row.rcept_no for row in self.candidates]
        if receipts != sorted(set(receipts)):
            raise ValueError("correction 후보는 접수번호 기준 정렬·고유해야 합니다")
        if self.status == "resolved":
            if (self.selected_receipt is None
                    or re.fullmatch(r"[0-9]{14}", self.selected_receipt) is None
                    or self.candidates):
                raise ValueError("resolved correction preflight 계약 오류")
        elif self.status == "ambiguous":
            if self.selected_receipt is not None or len(self.candidates) < 2:
                raise ValueError("ambiguous correction preflight 계약 오류")
        elif self.selected_receipt is not None or self.candidates:
            raise ValueError("후보 없는 correction 상태 계약 오류")


class CorrectionSeedPreflight(Protocol):
    def resolve_seed(
            self, *, corp_code: str, as_of: str, seed_rcept_no: str,
            operation: Literal["diff", "history"],
            ) -> CorrectionSeedResolution: ...


class ObservationLike(Protocol):
    rcept_no: str
    observed_at: str
    is_correction: bool


class TimelineLike(Protocol):
    corp_code: str
    observations: tuple[ObservationLike, ...]


class LineageLike(Protocol):
    status: str
    selected: str | None
    members: tuple[str, ...]


class CorrectionCanonicalLike(Protocol):
    def event_timeline(self, **kwargs: object) -> TimelineLike | None: ...

    def resolve_document_version(
            self, rcept_no: str, *, as_of: str) -> LineageLike: ...


class CanonicalCorrectionSeedPreflight:
    """공개 read API로 receipt 소유권과 diff 대상 정정본만 확인한다."""

    def __init__(self, canonical: CorrectionCanonicalLike) -> None:
        if any(not callable(getattr(canonical, name, None)) for name in (
                "event_timeline", "resolve_document_version")):
            raise TypeError(
                "correction preflight에는 event_timeline/resolve_document_version이 필요합니다")
        self._canonical = canonical

    def resolve_seed(
            self, *, corp_code: str, as_of: str, seed_rcept_no: str,
            operation: Literal["diff", "history"],
            ) -> CorrectionSeedResolution:
        timeline = self._canonical.event_timeline(
            as_of=as_of, rcept_no=seed_rcept_no, verify_evidence=False)
        if timeline is None or timeline.corp_code != corp_code:
            return CorrectionSeedResolution("not_found")
        by_receipt = {
            row.rcept_no: row for row in timeline.observations
        }
        if len(by_receipt) != len(timeline.observations):
            return CorrectionSeedResolution("invalid")
        seed = by_receipt.get(seed_rcept_no)
        if seed is None:
            return CorrectionSeedResolution("not_found")
        if operation == "history" or seed.is_correction:
            return CorrectionSeedResolution(
                "resolved", selected_receipt=seed_rcept_no)

        lineage = self._canonical.resolve_document_version(
            seed_rcept_no, as_of=as_of)
        if lineage.status in {"invalid", "not_found"}:
            return CorrectionSeedResolution(
                "invalid" if lineage.status == "invalid" else "not_found")
        member_set = set(lineage.members)
        corrections = tuple(sorted(
            (row for receipt, row in by_receipt.items()
             if receipt in member_set and row.is_correction),
            key=lambda row: row.rcept_no,
        ))
        if not corrections:
            return CorrectionSeedResolution("not_found")
        if lineage.status == "ok" and len(corrections) == 1:
            return CorrectionSeedResolution(
                "resolved", selected_receipt=corrections[0].rcept_no)
        if len(corrections) < 2:
            return CorrectionSeedResolution("invalid")
        return CorrectionSeedResolution("ambiguous", candidates=tuple(
            CorrectionSeedCandidate(
                rcept_no=row.rcept_no,
                label=f"{row.observed_at} 정정공시 · {row.rcept_no}",
            )
            for row in corrections
        ))


__all__ = [
    "CanonicalCorrectionSeedPreflight", "CorrectionSeedCandidate",
    "CorrectionSeedPreflight", "CorrectionSeedResolution",
    "CorrectionSeedStatus",
]
