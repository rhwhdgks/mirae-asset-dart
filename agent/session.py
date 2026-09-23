"""원 질문 보존과 typed 역질문 순환을 관리하는 in-memory 세션.

사용자는 JSON path를 제출하지 않는다. resolver가 만든 path는 서버 내부 allowlist로
semantic target에 매핑되고, 사용자는 clarification ID·revision·값만 돌려준다.
최종 팀 QueryPlan이 오면 proposal adapter만 교체하고 상태·동시성 계약은 유지한다.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime, timezone
from enum import StrEnum
from hashlib import sha256
import hmac
from threading import RLock
import time
from typing import Any, Callable, Generic, Literal, Mapping, Protocol, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import ContractModel, PlanValidation, ResolvedQueryPlan
from .drafts import (
    DraftCorrectionTask,
    DraftDisclosureTask,
    DraftDocumentTask,
    DraftEventTask,
    DraftFinancialTask,
    DraftNarrativeTask,
    DraftQueryPlan,
)


MAX_CLARIFICATION_TURNS = 3
# revision은 받아들인 clarification answer에서만 정확히 1 증가한다.
MAX_PLAN_REVISIONS = MAX_CLARIFICATION_TURNS
ProposalT = TypeVar("ProposalT", bound=BaseModel)


class ClarificationTarget(StrEnum):
    COMPANY = "company"
    METRIC = "metric"
    YEAR = "year"
    SCOPE = "scope"
    AS_OF = "as_of"
    VIEW = "view"
    RETRIEVAL_QUERY = "retrieval_query"
    DOCUMENT_GROUP = "document_group"
    DOCUMENT_TARGET_PERIOD = "document_target_period"
    DOCUMENT_RECEIPT = "document_receipt"
    EVENT_TYPE = "event_type"
    EVENT_TIMEPOINT = "event_timepoint"
    COUNTERPARTY = "counterparty"
    CONTRACT_NAME = "contract_name"
    SEED_RECEIPT = "seed_receipt"
    REQUESTED_SLOTS = "requested_slots"


_PATH_TARGETS = {
    "draft.tasks[0].company_text": ClarificationTarget.COMPANY,
    "draft.tasks[0].metric_text": ClarificationTarget.METRIC,
    "draft.tasks[0].year": ClarificationTarget.YEAR,
    "draft.tasks[0].scope": ClarificationTarget.SCOPE,
    "draft.tasks[0].as_of": ClarificationTarget.AS_OF,
    "draft.tasks[0].view": ClarificationTarget.VIEW,
    "draft.tasks[0].retrieval_query": ClarificationTarget.RETRIEVAL_QUERY,
    "draft.tasks[0].doc_group": ClarificationTarget.DOCUMENT_GROUP,
    "draft.tasks[0].target_period_expressions": (
        ClarificationTarget.DOCUMENT_TARGET_PERIOD),
    "draft.tasks[0].selected_document_receipt": (
        ClarificationTarget.DOCUMENT_RECEIPT),
    "draft.tasks[0].event_type_text": ClarificationTarget.EVENT_TYPE,
    "draft.tasks[0].as_of_expression": ClarificationTarget.EVENT_TIMEPOINT,
    "draft.tasks[0].counterparty_text": ClarificationTarget.COUNTERPARTY,
    "draft.tasks[0].contract_name_text": ClarificationTarget.CONTRACT_NAME,
    "draft.tasks[0].seed_receipt_text": ClarificationTarget.SEED_RECEIPT,
    "draft.tasks[0].requested_slots": ClarificationTarget.REQUESTED_SLOTS,
}
_TARGET_FIELDS = {
    ClarificationTarget.COMPANY: "company_text",
    ClarificationTarget.METRIC: "metric_text",
    ClarificationTarget.YEAR: "year",
    ClarificationTarget.SCOPE: "scope",
    ClarificationTarget.AS_OF: "as_of",
    ClarificationTarget.VIEW: "view",
    ClarificationTarget.RETRIEVAL_QUERY: "retrieval_query",
    ClarificationTarget.DOCUMENT_GROUP: "doc_group",
    ClarificationTarget.DOCUMENT_TARGET_PERIOD: "target_period_expressions",
    ClarificationTarget.DOCUMENT_RECEIPT: "selected_document_receipt",
    ClarificationTarget.EVENT_TYPE: "event_type_text",
    ClarificationTarget.EVENT_TIMEPOINT: "as_of_expression",
    ClarificationTarget.COUNTERPARTY: "counterparty_text",
    ClarificationTarget.CONTRACT_NAME: "contract_name_text",
    ClarificationTarget.SEED_RECEIPT: "seed_receipt_text",
    ClarificationTarget.REQUESTED_SLOTS: "requested_slots",
}


class SessionError(ValueError):
    """질문 세션 또는 clarification 상태 계약 오류."""


class SessionNotFoundError(SessionError):
    pass


class SessionExpiredError(SessionError):
    pass


class StaleClarificationError(SessionError):
    pass


class UnsafeClarificationError(SessionError):
    pass


class ClarificationLimitError(SessionError):
    pass


class QuestionOrigin(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1)
    original_question: str = Field(min_length=1, repr=False)
    reference_date: date
    created_at: datetime
    question_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("original_question")
    @classmethod
    def nonblank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("original_question은 비어 있을 수 없습니다")
        return value

    @field_validator("created_at")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at은 timezone-aware여야 합니다")
        return value

    @classmethod
    def create(
            cls, question: str, *, reference_date: date, hmac_key: bytes,
            now: datetime | None = None,
            ) -> "QuestionOrigin":
        if not isinstance(hmac_key, bytes) or len(hmac_key) < 32:
            raise ValueError("question HMAC key는 32 bytes 이상이어야 합니다")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question은 비어 있지 않은 문자열이어야 합니다")
        created = now or datetime.now(timezone.utc)
        fingerprint = hmac.new(
            hmac_key, question.encode("utf-8"), sha256).hexdigest()
        return cls(
            session_id=str(uuid4()), original_question=question,
            reference_date=reference_date, created_at=created,
            question_fingerprint=fingerprint,
        )


class ClarificationSlot(ContractModel):
    """사용자에게는 slot_id만 노출되는 서버 소유 patch 단위."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot_id: str = Field(pattern=r"^slot-[1-9][0-9]*$")
    target: str = Field(min_length=1)
    patch_path: str = Field(min_length=1, repr=False, exclude=True)
    allowed_values: tuple[Any, ...] = Field(default_factory=tuple)


class PendingClarification(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    clarification_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    slots: tuple[ClarificationSlot, ...] = Field(min_length=1)
    created_revision: int = Field(ge=0)

    @model_validator(mode="after")
    def unique_slots(self) -> "PendingClarification":
        slot_ids = [slot.slot_id for slot in self.slots]
        patch_paths = [slot.patch_path for slot in self.slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("clarification slot_id는 중복될 수 없습니다")
        if len(patch_paths) != len(set(patch_paths)):
            raise ValueError("clarification patch path는 중복될 수 없습니다")
        return self

    @property
    def target(self) -> str | None:
        """단일-slot 기존 호출자를 위한 읽기 전용 호환 projection."""

        return self.slots[0].target if len(self.slots) == 1 else None

    @property
    def allowed_values(self) -> tuple[Any, ...]:
        """단일-slot 기존 호출자를 위한 읽기 전용 호환 projection."""

        return self.slots[0].allowed_values if len(self.slots) == 1 else ()


class ClarificationAnswer(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    clarification_id: str = Field(min_length=1)
    expected_revision: int = Field(ge=0)
    values: dict[str, Any] = Field(min_length=1)


class ClarificationRecord(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    clarification_id: str = Field(min_length=1)
    slot_ids: tuple[str, ...] = Field(min_length=1)
    targets: tuple[str, ...] = Field(min_length=1)
    from_revision: int = Field(ge=0)
    to_revision: int = Field(ge=1)
    answered_at: datetime

    @model_validator(mode="after")
    def revision_advanced_once(self) -> "ClarificationRecord":
        if len(self.slot_ids) != len(self.targets):
            raise ValueError("clarification record slot/target 수가 다릅니다")
        if len(self.slot_ids) != len(set(self.slot_ids)):
            raise ValueError("clarification record slot_id는 중복될 수 없습니다")
        if self.to_revision != self.from_revision + 1:
            raise ValueError("clarification revision은 정확히 1 증가해야 합니다")
        if self.answered_at.tzinfo is None or self.answered_at.utcoffset() is None:
            raise ValueError("answered_at은 timezone-aware여야 합니다")
        return self


SessionStatus = Literal["waiting_for_clarification", "ready", "terminal"]


class QuestionSessionState(ContractModel, Generic[ProposalT]):
    origin: QuestionOrigin
    current_revision: int = Field(ge=0, le=MAX_PLAN_REVISIONS)
    proposal: ProposalT = Field(repr=False)
    resolved_plan: ResolvedQueryPlan | None = None
    pending_clarification: PendingClarification | None = None
    clarification_history: tuple[ClarificationRecord, ...] = Field(default_factory=tuple)
    status: SessionStatus
    terminal_status: str | None = None
    terminal_reasons: tuple[str, ...] = Field(default_factory=tuple)
    last_accessed_at: datetime

    @model_validator(mode="after")
    def enforce_state(self) -> "QuestionSessionState":
        if self.last_accessed_at.tzinfo is None or self.last_accessed_at.utcoffset() is None:
            raise ValueError("last_accessed_at은 timezone-aware여야 합니다")
        if len(self.clarification_history) > MAX_CLARIFICATION_TURNS:
            raise ValueError("clarification turn 상한을 넘었습니다")
        if self.status == "waiting_for_clarification":
            if (self.pending_clarification is None or self.resolved_plan is not None
                    or self.terminal_status is not None or self.terminal_reasons):
                raise ValueError("waiting 상태 계약 오류")
        elif self.status == "ready":
            if (self.resolved_plan is None or self.pending_clarification is not None
                    or self.terminal_status is not None or self.terminal_reasons):
                raise ValueError("ready 상태 계약 오류")
        elif (self.resolved_plan is not None or self.pending_clarification is not None
              or not self.terminal_status or not self.terminal_reasons):
            raise ValueError("terminal 상태 계약 오류")
        return self


class DraftResolver(Protocol):
    def resolve(self, draft: DraftQueryPlan, *, revision: int = 0) -> PlanValidation: ...


class ProposalAdapter(Protocol[ProposalT]):
    """세션 코어가 proposal 필드를 몰라도 되는 최소 신뢰경계."""

    def resolve(
            self, proposal: ProposalT, *, revision: int = 0,
            origin: QuestionOrigin | None = None,
            ) -> PlanValidation: ...

    def target_for_path(self, proposal: ProposalT, path: str) -> str | None: ...

    def apply(
            self, proposal: ProposalT, values_by_path: Mapping[str, Any],
            ) -> ProposalT: ...


class InMemoryQuestionSessionStore(Generic[ProposalT]):
    """TTL과 LRU 상한이 있는 단일-process 개발용 store."""

    def __init__(
            self, *, ttl_seconds: float = 1800.0, max_sessions: int = 256,
            clock: Callable[[], float] = time.monotonic,
            wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
            ) -> None:
        if ttl_seconds <= 0 or type(max_sessions) is not int or max_sessions <= 0:
            raise ValueError("session TTL과 max_sessions는 양수여야 합니다")
        self.ttl_seconds = float(ttl_seconds)
        self.max_sessions = max_sessions
        self._clock = clock
        self._wall_clock = wall_clock
        self._rows: OrderedDict[
            str, tuple[float, QuestionSessionState[ProposalT]]
        ] = OrderedDict()
        self._expired: set[str] = set()
        self._lock = RLock()

    def _purge(self, now: float) -> None:
        expired = [
            session_id for session_id, (last_seen, _) in self._rows.items()
            if now - last_seen >= self.ttl_seconds
        ]
        for session_id in expired:
            self._rows.pop(session_id, None)
            self._expired.add(session_id)
        while len(self._expired) > self.max_sessions * 4:
            self._expired.pop()

    def put(self, state: QuestionSessionState[ProposalT]) -> None:
        with self._lock:
            now = self._clock()
            self._purge(now)
            session_id = state.origin.session_id
            existing = self._rows.get(session_id)
            if existing is not None:
                previous = existing[1]
                if previous.origin != state.origin:
                    raise SessionError("동일 session_id의 QuestionOrigin을 바꿀 수 없습니다")
                if state.current_revision < previous.current_revision:
                    raise SessionError("session revision을 되돌릴 수 없습니다")
            self._rows.pop(session_id, None)
            while len(self._rows) >= self.max_sessions:
                evicted, _ = self._rows.popitem(last=False)
                self._expired.add(evicted)
            stored = state.model_copy(
                update={"last_accessed_at": self._wall_clock()}, deep=True)
            self._rows[session_id] = (now, stored)
            self._expired.discard(session_id)

    def get(self, session_id: str) -> QuestionSessionState[ProposalT]:
        with self._lock:
            now = self._clock()
            self._purge(now)
            if session_id not in self._rows:
                if session_id in self._expired:
                    raise SessionExpiredError("질문 세션이 만료되었거나 LRU로 제거되었습니다")
                raise SessionNotFoundError("질문 세션을 찾을 수 없습니다")
            _, state = self._rows.pop(session_id)
            touched = state.model_copy(
                update={"last_accessed_at": self._wall_clock()}, deep=True)
            self._rows[session_id] = (now, touched)
            return touched.model_copy(deep=True)

    def compare_and_put(
            self, state: QuestionSessionState[ProposalT], *, expected_revision: int,
            expected_clarification_id: str,
            ) -> None:
        """현재 clarification과 revision이 그대로일 때만 다음 상태를 저장한다."""

        with self._lock:
            now = self._clock()
            self._purge(now)
            session_id = state.origin.session_id
            existing = self._rows.get(session_id)
            if existing is None:
                if session_id in self._expired:
                    raise SessionExpiredError("질문 세션이 만료되었거나 LRU로 제거되었습니다")
                raise SessionNotFoundError("질문 세션을 찾을 수 없습니다")
            previous = existing[1]
            pending = previous.pending_clarification
            if (previous.current_revision != expected_revision
                    or previous.status != "waiting_for_clarification"
                    or pending is None
                    or pending.clarification_id != expected_clarification_id):
                raise StaleClarificationError(
                    "clarification 처리 중 session revision이 변경되었습니다")
            if previous.origin != state.origin:
                raise SessionError("동일 session_id의 QuestionOrigin을 바꿀 수 없습니다")
            if state.current_revision != expected_revision + 1:
                raise SessionError("compare-and-put은 revision을 정확히 1 증가시켜야 합니다")
            stored = state.model_copy(
                update={"last_accessed_at": self._wall_clock()}, deep=True)
            self._rows.pop(session_id, None)
            self._rows[session_id] = (now, stored)


def _pending_from_validation(
        validation: PlanValidation, *, proposal: ProposalT,
        adapter: ProposalAdapter[ProposalT],
        ) -> PendingClarification:
    request = validation.clarification
    if validation.status != "needs_clarification" or request is None:
        raise UnsafeClarificationError("typed clarification 결과가 아닙니다")
    paths = tuple(request.patch_paths)
    if (not paths or tuple(request.field_paths) != paths
            or len(paths) != len(set(paths))):
        raise UnsafeClarificationError(
            "clarification field/patch path가 비었거나 서로 다릅니다")
    if set(request.options).difference(paths):
        raise UnsafeClarificationError("clarification option에 다른 path가 섞였습니다")
    slots: list[ClarificationSlot] = []
    for index, path in enumerate(paths, 1):
        target = adapter.target_for_path(proposal, path)
        if target is None:
            raise UnsafeClarificationError(
                "resolver가 proposal adapter allowlist 밖 patch path를 반환했습니다")
        slots.append(ClarificationSlot(
            slot_id=f"slot-{index}", target=target, patch_path=path,
            allowed_values=tuple(
                option.value for option in request.options.get(path, [])),
        ))
    return PendingClarification(
        clarification_id=request.request_id,
        question=request.question,
        slots=tuple(slots),
        created_revision=request.plan_revision,
    )


class DraftProposalAdapter:
    """기존 단일-task Draft를 generic session seam에 연결하는 호환 adapter."""

    def __init__(self, resolver: DraftResolver) -> None:
        self.resolver = resolver

    def resolve(
            self, proposal: DraftQueryPlan, *, revision: int = 0,
            origin: QuestionOrigin | None = None,
            ) -> PlanValidation:
        del origin
        return self.resolver.resolve(proposal, revision=revision)

    @staticmethod
    def _path_target(
            proposal: DraftQueryPlan, path: str,
            ) -> tuple[int, str, ClarificationTarget] | None:
        prefix = "draft.tasks["
        if not path.startswith(prefix) or "]." not in path:
            return None
        index_text, field_name = path[len(prefix):].split("].", 1)
        if not index_text.isascii() or not index_text.isdigit() or "." in field_name:
            return None
        task_index = int(index_text)
        if not 0 <= task_index < len(proposal.tasks):
            return None
        task = proposal.tasks[task_index]
        allowed_fields: set[str]
        if isinstance(task, DraftFinancialTask):
            allowed_fields = {
                "company_text", "metric_text", "year", "scope", "as_of", "view",
            }
        elif isinstance(task, DraftNarrativeTask):
            allowed_fields = {
                "company_text", "retrieval_query", "as_of", "doc_group",
            }
        elif isinstance(task, DraftDocumentTask):
            allowed_fields = {
                "company_text", "as_of", "doc_group",
                "target_period_expressions", "selected_document_receipt",
            }
        elif isinstance(task, DraftDisclosureTask):
            allowed_fields = {
                "company_text", "as_of", "doc_group", "event_type_text",
                "requested_slots",
            }
        elif isinstance(task, DraftEventTask):
            allowed_fields = {
                "company_text", "as_of_expression", "event_type_text",
                "counterparty_text", "contract_name_text",
                "seed_receipt_text", "requested_slots",
            }
        elif isinstance(task, DraftCorrectionTask):
            allowed_fields = {
                "company_text", "as_of", "doc_group", "event_type_text",
                "counterparty_text", "contract_name_text",
                "seed_receipt_text", "requested_slots",
            }
        else:
            return None
        if field_name not in allowed_fields:
            return None
        target = _PATH_TARGETS.get(f"draft.tasks[0].{field_name}")
        if target is None or _TARGET_FIELDS[target] != field_name:
            return None
        return task_index, field_name, target

    def target_for_path(self, proposal: DraftQueryPlan, path: str) -> str | None:
        parsed = self._path_target(proposal, path)
        return parsed[2].value if parsed is not None else None

    def apply(
            self, proposal: DraftQueryPlan,
            values_by_path: Mapping[str, Any],
            ) -> DraftQueryPlan:
        if not values_by_path:
            raise UnsafeClarificationError("적용할 clarification 값이 없습니다")
        task_payloads = [task.model_dump(mode="python") for task in proposal.tasks]
        for path, value in values_by_path.items():
            parsed = self._path_target(proposal, path)
            if parsed is None:
                raise UnsafeClarificationError(
                    "proposal adapter allowlist 밖 path를 적용할 수 없습니다")
            task_index, field_name, _ = parsed
            if field_name == "target_period_expressions" and isinstance(value, str):
                value = [value]
            if field_name == "requested_slots" and isinstance(value, str):
                value = [value]
            task_payloads[task_index][field_name] = value
        try:
            return DraftQueryPlan.model_validate({
                "schema_version": proposal.schema_version,
                "tasks": task_payloads,
            })
        except ValueError as exc:
            raise UnsafeClarificationError(
                "clarification batch가 proposal schema를 통과하지 못했습니다") from exc


class ClarificationCoordinator(Generic[ProposalT]):
    """proposal schema와 독립적인 atomic multi-slot clarification coordinator."""

    def __init__(
            self, adapter: ProposalAdapter[ProposalT],
            store: InMemoryQuestionSessionStore[ProposalT],
            *, hmac_key: bytes,
            wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
            ) -> None:
        if not isinstance(hmac_key, bytes) or len(hmac_key) < 32:
            raise ValueError("session question HMAC key는 32 bytes 이상이어야 합니다")
        self.adapter = adapter
        self.store = store
        self._hmac_key = hmac_key
        self._wall_clock = wall_clock

    def _state_from_validation(
            self, *, origin: QuestionOrigin, proposal: ProposalT,
            revision: int, history: tuple[ClarificationRecord, ...],
            validation: PlanValidation,
            ) -> QuestionSessionState[ProposalT]:
        now = self._wall_clock()
        state_type = QuestionSessionState[type(proposal)]
        if validation.status == "ready":
            assert validation.plan is not None
            if validation.plan.revision != revision:
                raise SessionError("resolved plan revision이 session revision과 다릅니다")
            return state_type(
                origin=origin, current_revision=revision, proposal=proposal,
                resolved_plan=validation.plan, status="ready",
                clarification_history=history, last_accessed_at=now,
            )
        if validation.status == "needs_clarification":
            pending = _pending_from_validation(
                validation, proposal=proposal, adapter=self.adapter)
            if pending.created_revision != revision:
                raise SessionError("clarification revision이 session revision과 다릅니다")
            return state_type(
                origin=origin, current_revision=revision, proposal=proposal,
                pending_clarification=pending, status="waiting_for_clarification",
                clarification_history=history, last_accessed_at=now,
            )
        return state_type(
            origin=origin, current_revision=revision, proposal=proposal,
            status="terminal", terminal_status=validation.status,
            terminal_reasons=tuple(validation.reasons),
            clarification_history=history, last_accessed_at=now,
        )

    def open(
            self, *, question: str, reference_date: date,
            proposal: ProposalT,
            ) -> QuestionSessionState[ProposalT]:
        origin = QuestionOrigin.create(
            question, reference_date=reference_date, hmac_key=self._hmac_key,
            now=self._wall_clock())
        validation = self.adapter.resolve(
            proposal, revision=0, origin=origin)
        state = self._state_from_validation(
            origin=origin, proposal=proposal, revision=0, history=(),
            validation=validation,
        )
        self.store.put(state)
        return self.store.get(origin.session_id)

    def answer(
            self, session_id: str, answer: ClarificationAnswer,
            ) -> QuestionSessionState[ProposalT]:
        state = self.store.get(session_id)
        if state.status != "waiting_for_clarification" or state.pending_clarification is None:
            raise StaleClarificationError("현재 답변할 clarification이 없습니다")
        pending = state.pending_clarification
        if (answer.clarification_id != pending.clarification_id
                or answer.expected_revision != state.current_revision
                or pending.created_revision != state.current_revision):
            raise StaleClarificationError("clarification ID 또는 revision이 오래되었습니다")
        if len(state.clarification_history) >= MAX_CLARIFICATION_TURNS:
            raise ClarificationLimitError("역질문 최대 횟수에 도달했습니다")
        if state.current_revision >= MAX_PLAN_REVISIONS:
            raise ClarificationLimitError("plan revision 최대 횟수에 도달했습니다")
        expected_slots = {slot.slot_id for slot in pending.slots}
        if set(answer.values) != expected_slots:
            raise SessionError(
                "clarification 답변은 서버가 요청한 slot을 정확히 모두 포함해야 합니다")
        values_by_path: dict[str, Any] = {}
        for slot in pending.slots:
            value = answer.values[slot.slot_id]
            if (isinstance(value, str) and not value.strip()) or value is None:
                raise SessionError("clarification 답변은 비어 있을 수 없습니다")
            if slot.allowed_values and value not in slot.allowed_values:
                raise SessionError("clarification 답변이 서버 허용값에 없습니다")
            values_by_path[slot.patch_path] = value

        # batch apply·resolver 중 하나라도 실패하면 store/revision을 갱신하지 않는다.
        patched = self.adapter.apply(state.proposal, values_by_path)
        next_revision = state.current_revision + 1
        validation = self.adapter.resolve(
            patched, revision=next_revision, origin=state.origin)
        record = ClarificationRecord(
            clarification_id=pending.clarification_id,
            slot_ids=tuple(slot.slot_id for slot in pending.slots),
            targets=tuple(slot.target for slot in pending.slots),
            from_revision=state.current_revision, to_revision=next_revision,
            answered_at=self._wall_clock(),
        )
        next_state = self._state_from_validation(
            origin=state.origin, proposal=patched, revision=next_revision,
            history=state.clarification_history + (record,),
            validation=validation,
        )
        self.store.compare_and_put(
            next_state, expected_revision=state.current_revision,
            expected_clarification_id=pending.clarification_id)
        return self.store.get(session_id)


class DraftClarificationCoordinator(ClarificationCoordinator[DraftQueryPlan]):
    """기존 Draft resolver 호출부를 보존하는 얇은 compatibility wrapper."""

    def __init__(
            self, resolver: DraftResolver,
            store: InMemoryQuestionSessionStore[DraftQueryPlan], *,
            hmac_key: bytes,
            wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
            ) -> None:
        super().__init__(
            DraftProposalAdapter(resolver), store,
            hmac_key=hmac_key, wall_clock=wall_clock)


__all__ = [
    "ClarificationAnswer", "ClarificationCoordinator", "ClarificationLimitError",
    "ClarificationRecord", "ClarificationSlot", "ClarificationTarget",
    "DraftClarificationCoordinator", "DraftProposalAdapter",
    "InMemoryQuestionSessionStore", "MAX_CLARIFICATION_TURNS",
    "MAX_PLAN_REVISIONS", "PendingClarification", "ProposalAdapter", "QuestionOrigin",
    "QuestionSessionState", "SessionError", "SessionExpiredError",
    "SessionNotFoundError", "StaleClarificationError",
    "UnsafeClarificationError",
]
