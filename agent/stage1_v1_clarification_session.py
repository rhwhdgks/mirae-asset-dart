"""Digest-bound multi-turn clarification sidecar for Stage1 v1.

The original question and ``SemanticIntent`` are immutable session inputs.
User answers are recorded in a separate, typed resolution context; neither a
JSON patch path nor a resolved-plan mutation is part of this boundary.

A resume backend returns a newly derived immutable intent together with a v1
resolution authority.  The derived intent may only remove unresolved mentions
owned by the answered slots.  All question-grounded semantic surfaces remain
byte-semantic identical.  The pair is then passed through a fresh
``Stage1V1Resolver`` and ``Stage1V1Orchestrator`` run.
"""

from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Annotated, Any, Literal, Mapping, Protocol
from uuid import UUID, uuid4, uuid5

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)

from .semantic_intent_v1 import SemanticIntent, semantic_intent_digest
from .stage1_ready_envelope_v02 import Stage1ReadyEnvelope
from .stage1_v1_outcome import (
    Stage1NormalStatus,
    Stage1Outcome,
    Stage1V1Orchestrator,
    canonical_json,
)
from .stage1_v1_resolver import (
    ClarificationAuthority,
    ClarificationResponseKind,
    ClarificationRole,
    ClarificationSlot,
    ReasonCode,
    ResolutionAuthority,
    Stage1ResolutionBackend,
    Stage1V1Resolver,
    TerminalAuthority,
    validate_semantic_intent_grounding,
)


CLARIFICATION_SESSION_VERSION = "stage1-v1-clarification-session/1.0"
CLARIFICATION_CONTEXT_VERSION = "stage1-v1-clarification-context/1.0"
CLARIFICATION_READY_BINDING_VERSION = (
    "stage1-v1-clarification-ready-binding/1.0")
MAX_CLARIFICATION_TURNS = 3

_CLARIFICATION_NAMESPACE = UUID("59cc83ab-d04f-55b0-b7e7-eaac5594e453")

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
BuildId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
NonEmpty = Annotated[str, StringConstraints(min_length=1)]
Identifier = Annotated[str, StringConstraints(
    min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")]
Versioned = Annotated[str, StringConstraints(
    min_length=3,
    pattern=r"^[A-Za-z0-9_.-]+/[0-9]+(?:\.[0-9]+)*$",
)]
SlotId = Annotated[str, StringConstraints(pattern=r"^slot-[1-9][0-9]*$")]


class ClarificationSessionError(RuntimeError):
    """Base error for the v1 clarification sidecar."""


class ClarificationSessionNotFoundError(ClarificationSessionError):
    pass


class StaleClarificationAnswerError(ClarificationSessionError):
    pass


class InvalidClarificationAnswerError(ClarificationSessionError):
    pass


class ClarificationTurnLimitError(ClarificationSessionError):
    pass


class ClarificationResumeTechnicalError(ClarificationSessionError):
    """Resume backend/schema/compiler failure; never a normal outcome."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )


def _strict_model(model: type[BaseModel], value: Any) -> BaseModel:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", warnings=False)
    if isinstance(value, Mapping):
        return model.model_validate_json(canonical_json(value), strict=True)
    raise TypeError(f"{model.__name__} instance 또는 mapping이 필요합니다")


def _valid_uuid(value: str, *, label: str) -> None:
    try:
        UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"{label}는 UUID여야 합니다") from exc


def _digest_payload(value: BaseModel | Mapping[str, Any], *, omit: str) -> str:
    body = (
        value.model_dump(mode="json", warnings=False)
        if isinstance(value, BaseModel) else dict(value)
    )
    body.pop(omit, None)
    return sha256(canonical_json(body).encode("utf-8")).hexdigest()


class Stage1V1ClarificationAnswer(_StrictFrozenModel):
    """Public answer shape with explicit user escape actions."""

    session_id: NonEmpty
    clarification_id: NonEmpty
    expected_revision: int = Field(ge=0)
    action: Literal["submit", "none_of_above", "unknown"] = "submit"
    values: dict[SlotId, NonEmpty] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_answer(self) -> "Stage1V1ClarificationAnswer":
        _valid_uuid(self.session_id, label="session_id")
        _valid_uuid(self.clarification_id, label="clarification_id")
        if self.action == "submit" and not self.values:
            raise ValueError("submit clarification 답변에는 slot 값이 필요합니다")
        if self.action != "submit" and self.values:
            raise ValueError("clarification 탈출 응답에는 slot 값을 함께 보낼 수 없습니다")
        if any(not value.strip() for value in self.values.values()):
            raise ValueError("clarification 답변은 비공백 문자열이어야 합니다")
        return self


class ClarificationAnswerBinding(_StrictFrozenModel):
    """One accepted slot answer and its semantic ownership metadata."""

    slot_id: SlotId
    role_hint: ClarificationRole
    reason_code: ReasonCode
    response_kind: ClarificationResponseKind
    applies_to_item_ids: tuple[Identifier, ...] = Field(min_length=1)
    mention_ids: tuple[Identifier, ...] = Field(default_factory=tuple)
    value: NonEmpty
    selected_option_label: NonEmpty | None = None

    @model_validator(mode="after")
    def validate_value(self) -> "ClarificationAnswerBinding":
        if not self.value.strip():
            raise ValueError("accepted clarification value는 공백일 수 없습니다")
        if (self.selected_option_label is not None
                and not self.selected_option_label.strip()):
            raise ValueError("selected option label은 공백일 수 없습니다")
        if len(self.applies_to_item_ids) != len(set(self.applies_to_item_ids)):
            raise ValueError("clarification item binding이 중복되었습니다")
        if len(self.mention_ids) != len(set(self.mention_ids)):
            raise ValueError("clarification mention binding이 중복되었습니다")
        return self


class ClarificationTurnRequest(_StrictFrozenModel):
    """Answer plus the exact prior v1 authority state it is answering."""

    clarification_id: NonEmpty
    from_revision: int = Field(ge=0)
    to_revision: int = Field(ge=1)
    previous_outcome_digest: Digest
    previous_decision_digest: Digest
    previous_source_intent_digest: Digest
    canonical_build_id: BuildId
    resolver_version: Versioned
    answers: tuple[ClarificationAnswerBinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_turn(self) -> "ClarificationTurnRequest":
        _valid_uuid(self.clarification_id, label="clarification_id")
        if self.to_revision != self.from_revision + 1:
            raise ValueError("clarification revision은 정확히 1 증가해야 합니다")
        slot_ids = [row.slot_id for row in self.answers]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("clarification answer slot이 중복되었습니다")
        return self


class ClarificationHistoryTurn(_StrictFrozenModel):
    """Completed resume turn retained for chained clarification provenance."""

    request: ClarificationTurnRequest
    request_context_digest: Digest
    derived_source_intent_digest: Digest
    resumed_decision_digest: Digest
    resumed_outcome_digest: Digest
    resumed_status: Stage1NormalStatus


class ClarificationResolutionContext(_StrictFrozenModel):
    """Cumulative, digest-bound sidecar supplied to the resume backend."""

    schema_version: Literal[CLARIFICATION_CONTEXT_VERSION] = (
        CLARIFICATION_CONTEXT_VERSION)
    session_id: NonEmpty
    question_id: Identifier
    original_question_sha256: Digest
    original_source_intent_digest: Digest
    history: tuple[ClarificationHistoryTurn, ...] = Field(default_factory=tuple)
    current: ClarificationTurnRequest
    context_digest: Digest

    @classmethod
    def create(cls, **payload: Any) -> "ClarificationResolutionContext":
        body = dict(payload)
        body.setdefault("schema_version", CLARIFICATION_CONTEXT_VERSION)
        body["context_digest"] = "0" * 64
        digest_body = cls.model_construct(**body).model_dump(
            mode="json", warnings=False)
        body["context_digest"] = _digest_payload(
            digest_body, omit="context_digest")
        return cls.model_validate(body, strict=True)

    @model_validator(mode="after")
    def validate_context(self) -> "ClarificationResolutionContext":
        _valid_uuid(self.session_id, label="session_id")
        if self.current.from_revision != len(self.history):
            raise ValueError("context history와 current revision이 다릅니다")
        for index, turn in enumerate(self.history):
            if turn.request.from_revision != index:
                raise ValueError("clarification history revision이 연속적이지 않습니다")
            if index == 0:
                if (turn.request.previous_source_intent_digest
                        != self.original_source_intent_digest):
                    raise ValueError("첫 clarification이 original intent와 다릅니다")
            else:
                previous = self.history[index - 1]
                if (
                    turn.request.previous_outcome_digest
                    != previous.resumed_outcome_digest
                    or turn.request.previous_decision_digest
                    != previous.resumed_decision_digest
                    or turn.request.previous_source_intent_digest
                    != previous.derived_source_intent_digest
                ):
                    raise ValueError("clarification history digest chain이 끊겼습니다")
        if self.history:
            previous = self.history[-1]
            if (
                self.current.previous_outcome_digest
                != previous.resumed_outcome_digest
                or self.current.previous_decision_digest
                != previous.resumed_decision_digest
                or self.current.previous_source_intent_digest
                != previous.derived_source_intent_digest
            ):
                raise ValueError("current clarification이 직전 turn과 결속되지 않았습니다")
        elif (self.current.previous_source_intent_digest
              != self.original_source_intent_digest):
            raise ValueError("첫 current clarification이 original intent와 다릅니다")
        identities = {
            (row.request.canonical_build_id, row.request.resolver_version)
            for row in self.history
        }
        identities.add((
            self.current.canonical_build_id, self.current.resolver_version))
        if len(identities) != 1:
            raise ValueError("clarification 중 canonical/resolver identity가 바뀌었습니다")
        if self.context_digest != _digest_payload(self, omit="context_digest"):
            raise ValueError("clarification context digest가 일치하지 않습니다")
        return self


_AUTHORITY_ADAPTER = TypeAdapter(ResolutionAuthority)


def _strict_authority(value: ResolutionAuthority | Mapping[str, Any]) -> ResolutionAuthority:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", warnings=False)
    if not isinstance(value, Mapping):
        raise TypeError("resume backend authority는 typed authority여야 합니다")
    return _AUTHORITY_ADAPTER.validate_json(canonical_json(value), strict=True)


class ClarificationResumeResult(_StrictFrozenModel):
    """Resume backend output before a fresh resolver/orchestrator run."""

    derived_source_intent: SemanticIntent
    authority: ResolutionAuthority

    @field_validator("derived_source_intent", mode="before")
    @classmethod
    def strict_intent(cls, value: Any) -> SemanticIntent:
        return _strict_model(SemanticIntent, value)  # type: ignore[return-value]

    @field_validator("authority", mode="before")
    @classmethod
    def strict_authority(cls, value: Any) -> ResolutionAuthority:
        return _strict_authority(value)


class Stage1V1ClarificationResumeBackend(Protocol):
    def resume(
            self,
            *,
            question_id: str,
            question: str,
            original_source_intent: SemanticIntent,
            current_source_intent: SemanticIntent,
            context: ClarificationResolutionContext,
            ) -> ClarificationResumeResult | Mapping[str, Any]: ...


class PublicClarification(_StrictFrozenModel):
    clarification_id: NonEmpty
    plan_revision: int = Field(ge=0)
    slots: tuple[ClarificationSlot, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_public(self) -> "PublicClarification":
        _valid_uuid(self.clarification_id, label="clarification_id")
        slot_ids = [row.slot_id for row in self.slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("public clarification slot이 중복되었습니다")
        return self


class ClarificationReadyBinding(_StrictFrozenModel):
    """The only Stage2-facing resumed-ready artifact."""

    schema_version: Literal[CLARIFICATION_READY_BINDING_VERSION] = (
        CLARIFICATION_READY_BINDING_VERSION)
    session_id: NonEmpty
    initial_outcome_digest: Digest
    original_source_intent_digest: Digest
    context_digests: tuple[Digest, ...] = Field(min_length=1)
    final_outcome: Stage1Outcome
    binding_digest: Digest

    @classmethod
    def create(cls, **payload: Any) -> "ClarificationReadyBinding":
        body = dict(payload)
        body.setdefault("schema_version", CLARIFICATION_READY_BINDING_VERSION)
        body["binding_digest"] = "0" * 64
        digest_body = cls.model_construct(**body).model_dump(
            mode="json", warnings=False)
        body["binding_digest"] = _digest_payload(
            digest_body, omit="binding_digest")
        return cls.model_validate(body, strict=True)

    @field_validator("final_outcome", mode="before")
    @classmethod
    def strict_outcome(cls, value: Any) -> Stage1Outcome:
        return _strict_model(Stage1Outcome, value)  # type: ignore[return-value]

    @model_validator(mode="after")
    def validate_binding(self) -> "ClarificationReadyBinding":
        _valid_uuid(self.session_id, label="session_id")
        if self.final_outcome.status not in {"ready", "partial_ready"}:
            raise ValueError("ready binding에는 ready outcome만 허용됩니다")
        if self.final_outcome.ready_envelope is None:
            raise ValueError("ready binding final outcome에 envelope가 없습니다")
        if len(self.context_digests) != len(set(self.context_digests)):
            raise ValueError("ready binding context digest가 중복되었습니다")
        if self.binding_digest != _digest_payload(self, omit="binding_digest"):
            raise ValueError("clarification ready binding digest가 일치하지 않습니다")
        return self

    @property
    def ready_envelope(self) -> Stage1ReadyEnvelope:
        assert self.final_outcome.ready_envelope is not None
        return self.final_outcome.ready_envelope


class ClarificationSessionView(_StrictFrozenModel):
    session_id: NonEmpty
    revision: int = Field(ge=0)
    status: Stage1NormalStatus
    clarification: PublicClarification | None = None
    ready: ClarificationReadyBinding | None = None
    terminal_outcome: Stage1Outcome | None = None

    @model_validator(mode="after")
    def validate_view(self) -> "ClarificationSessionView":
        _valid_uuid(self.session_id, label="session_id")
        if self.status == "needs_clarification":
            if (self.clarification is None or self.ready is not None
                    or self.terminal_outcome is not None):
                raise ValueError("waiting view는 clarification만 노출해야 합니다")
        elif self.status in {"ready", "partial_ready"}:
            if (self.clarification is not None or self.ready is None
                    or self.terminal_outcome is not None):
                raise ValueError("ready view는 bound ready artifact만 노출해야 합니다")
        elif (self.clarification is not None or self.ready is not None
              or self.terminal_outcome is None):
            raise ValueError("terminal view는 terminal outcome만 노출해야 합니다")
        return self


class _ClarificationSessionState(_StrictFrozenModel):
    schema_version: Literal[CLARIFICATION_SESSION_VERSION] = (
        CLARIFICATION_SESSION_VERSION)
    session_id: NonEmpty
    question_id: Identifier
    original_question: NonEmpty = Field(repr=False)
    original_source_intent: SemanticIntent = Field(repr=False)
    original_source_intent_digest: Digest
    current_source_intent: SemanticIntent = Field(repr=False)
    initial_outcome: Stage1Outcome
    current_outcome: Stage1Outcome
    contexts: tuple[ClarificationResolutionContext, ...] = Field(
        default_factory=tuple)
    history: tuple[ClarificationHistoryTurn, ...] = Field(default_factory=tuple)
    current_revision: int = Field(ge=0)

    @field_validator("original_source_intent", "current_source_intent", mode="before")
    @classmethod
    def strict_intent(cls, value: Any) -> SemanticIntent:
        return _strict_model(SemanticIntent, value)  # type: ignore[return-value]

    @field_validator("initial_outcome", "current_outcome", mode="before")
    @classmethod
    def strict_outcome(cls, value: Any) -> Stage1Outcome:
        return _strict_model(Stage1Outcome, value)  # type: ignore[return-value]

    @model_validator(mode="after")
    def validate_state(self) -> "_ClarificationSessionState":
        _valid_uuid(self.session_id, label="session_id")
        original = validate_semantic_intent_grounding(
            self.original_question, self.original_source_intent)
        current = validate_semantic_intent_grounding(
            self.original_question, self.current_source_intent)
        if self.original_source_intent_digest != semantic_intent_digest(original):
            raise ValueError("session original source intent digest가 다릅니다")
        initial_decision = self.initial_outcome.resolution_decision
        current_decision = self.current_outcome.resolution_decision
        if (
            self.initial_outcome.status != "needs_clarification"
            or initial_decision.question_id != self.question_id
            or initial_decision.question != self.original_question
            or initial_decision.source_intent_digest
            != self.original_source_intent_digest
        ):
            raise ValueError("session initial outcome binding이 잘못되었습니다")
        if (
            current_decision.question_id != self.question_id
            or current_decision.question != self.original_question
            or current_decision.source_intent_digest != semantic_intent_digest(current)
        ):
            raise ValueError("session current outcome binding이 잘못되었습니다")
        if self.current_revision != len(self.history) or len(self.contexts) != len(
                self.history):
            raise ValueError("session revision/context/history 수가 다릅니다")
        for index, (context, turn) in enumerate(zip(
                self.contexts, self.history, strict=True)):
            if (
                context.history != self.history[:index]
                or context.current != turn.request
                or context.context_digest != turn.request_context_digest
            ):
                raise ValueError("session context/history binding이 다릅니다")
        if self.history:
            final = self.history[-1]
            if (
                final.resumed_outcome_digest != self.current_outcome.outcome_digest
                or final.resumed_decision_digest != current_decision.decision_digest
                or final.derived_source_intent_digest
                != semantic_intent_digest(current)
                or final.resumed_status != self.current_outcome.status
            ):
                raise ValueError("session final outcome/history binding이 다릅니다")
        elif self.current_outcome.outcome_digest != self.initial_outcome.outcome_digest:
            raise ValueError("빈 session history의 current outcome이 바뀌었습니다")
        return self


def _copy_state(value: _ClarificationSessionState) -> _ClarificationSessionState:
    return _ClarificationSessionState.model_validate_json(
        canonical_json(value), strict=True)


class Stage1V1ClarificationStore(Protocol):
    """Durable-store contract required by the clarification coordinator."""

    def put_new(self, state: _ClarificationSessionState) -> None: ...

    def get(self, session_id: str) -> _ClarificationSessionState: ...

    def compare_and_put(
            self, state: _ClarificationSessionState, *,
            expected_revision: int,
            expected_outcome_digest: str,
            ) -> None: ...


class InMemoryStage1V1ClarificationStore:
    """Thread-safe development store; process restart intentionally loses rows."""

    def __init__(self, *, max_sessions: int = 256) -> None:
        if type(max_sessions) is not int or max_sessions <= 0:
            raise ValueError("max_sessions는 양수여야 합니다")
        self.max_sessions = max_sessions
        self._rows: OrderedDict[str, _ClarificationSessionState] = OrderedDict()
        self._lock = RLock()

    def put_new(self, state: _ClarificationSessionState) -> None:
        with self._lock:
            if state.session_id in self._rows:
                raise ClarificationSessionError("동일 session_id가 이미 존재합니다")
            while len(self._rows) >= self.max_sessions:
                self._rows.popitem(last=False)
            self._rows[state.session_id] = _copy_state(state)

    def get(self, session_id: str) -> _ClarificationSessionState:
        with self._lock:
            state = self._rows.get(session_id)
            if state is None:
                raise ClarificationSessionNotFoundError(
                    "v1 clarification session을 찾을 수 없습니다")
            self._rows.move_to_end(session_id)
            return _copy_state(state)

    def compare_and_put(
            self,
            state: _ClarificationSessionState,
            *,
            expected_revision: int,
            expected_outcome_digest: str,
            ) -> None:
        with self._lock:
            previous = self._rows.get(state.session_id)
            if previous is None:
                raise ClarificationSessionNotFoundError(
                    "v1 clarification session을 찾을 수 없습니다")
            if (
                previous.current_revision != expected_revision
                or previous.current_outcome.outcome_digest
                != expected_outcome_digest
            ):
                raise StaleClarificationAnswerError(
                    "clarification 처리 중 session authority가 변경되었습니다")
            if state.current_revision != expected_revision + 1:
                raise ClarificationSessionError(
                    "compare-and-put은 revision을 정확히 1 증가시켜야 합니다")
            self._rows[state.session_id] = _copy_state(state)
            self._rows.move_to_end(state.session_id)


class SQLiteStage1V1ClarificationStore:
    """Process-restart-safe SQLite store with atomic revision/digest CAS.

    The serialized state remains validated by ``_ClarificationSessionState`` on
    every read. SQLite owns durability and concurrency; the existing digest
    chain remains the semantic integrity boundary.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path, timeout=10.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS clarification_sessions ("
                "session_id TEXT PRIMARY KEY, "
                "revision INTEGER NOT NULL, "
                "outcome_digest TEXT NOT NULL, "
                "state_json TEXT NOT NULL"
                ")"
            )

    @staticmethod
    def _payload(state: _ClarificationSessionState) -> str:
        return canonical_json(_copy_state(state))

    def put_new(self, state: _ClarificationSessionState) -> None:
        payload = self._payload(state)
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO clarification_sessions "
                    "(session_id, revision, outcome_digest, state_json) "
                    "VALUES (?, ?, ?, ?)",
                    (state.session_id, state.current_revision,
                     state.current_outcome.outcome_digest, payload),
                )
        except sqlite3.IntegrityError as exc:
            raise ClarificationSessionError(
                "동일 session_id가 이미 존재합니다") from exc

    def get(self, session_id: str) -> _ClarificationSessionState:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM clarification_sessions "
                "WHERE session_id = ?", (session_id,),
            ).fetchone()
        if row is None:
            raise ClarificationSessionNotFoundError(
                "v1 clarification session을 찾을 수 없습니다")
        try:
            return _ClarificationSessionState.model_validate_json(
                row[0], strict=True)
        except Exception as exc:
            raise ClarificationSessionError(
                "저장된 v1 clarification session 계약이 손상되었습니다") from exc

    def compare_and_put(
            self, state: _ClarificationSessionState, *,
            expected_revision: int,
            expected_outcome_digest: str,
            ) -> None:
        if state.current_revision != expected_revision + 1:
            raise ClarificationSessionError(
                "compare-and-put은 revision을 정확히 1 증가시켜야 합니다")
        payload = self._payload(state)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE clarification_sessions "
                "SET revision = ?, outcome_digest = ?, state_json = ? "
                "WHERE session_id = ? AND revision = ? AND outcome_digest = ?",
                (state.current_revision, state.current_outcome.outcome_digest,
                 payload, state.session_id, expected_revision,
                 expected_outcome_digest),
            )
            if cursor.rowcount != 1:
                exists = connection.execute(
                    "SELECT 1 FROM clarification_sessions WHERE session_id = ?",
                    (state.session_id,),
                ).fetchone()
                connection.execute("ROLLBACK")
                if exists is None:
                    raise ClarificationSessionNotFoundError(
                        "v1 clarification session을 찾을 수 없습니다")
                raise StaleClarificationAnswerError(
                    "clarification 처리 중 session authority가 변경되었습니다")
            connection.execute("COMMIT")


class _StaticAuthorityBackend(Stage1ResolutionBackend):
    def __init__(self, authority: ResolutionAuthority) -> None:
        self.authority = authority

    def resolve(self, **_: Any) -> ResolutionAuthority:
        return self.authority


def _clarification_id(state: _ClarificationSessionState) -> str:
    return str(uuid5(
        _CLARIFICATION_NAMESPACE,
        f"{state.session_id}:{state.current_revision}:"
        f"{state.current_outcome.outcome_digest}",
    ))


def _public_clarification(
        state: _ClarificationSessionState,
        ) -> PublicClarification:
    authority = state.current_outcome.resolution_decision.authority
    if not isinstance(authority, ClarificationAuthority):
        raise ClarificationSessionError("현재 outcome은 clarification이 아닙니다")
    return PublicClarification(
        clarification_id=_clarification_id(state),
        plan_revision=state.current_revision,
        slots=tuple(authority.slots),
    )


def _accepted_answers(
        authority: ClarificationAuthority,
        values: Mapping[str, str],
        ) -> tuple[ClarificationAnswerBinding, ...]:
    expected = [slot.slot_id for slot in authority.slots]
    if set(values) != set(expected) or len(values) != len(expected):
        raise InvalidClarificationAnswerError(
            "clarification 답변은 요청한 slot을 정확히 모두 포함해야 합니다")
    accepted: list[ClarificationAnswerBinding] = []
    for slot in authority.slots:
        value = values[slot.slot_id]
        if not isinstance(value, str) or not value.strip():
            raise InvalidClarificationAnswerError(
                "clarification 답변은 비공백 문자열이어야 합니다")
        option_by_value = {option.value: option for option in slot.options}
        selected = option_by_value.get(value)
        if slot.response_kind != "provide_value" and selected is None:
            raise InvalidClarificationAnswerError(
                "clarification 답변이 허용된 선택지에 없습니다")
        accepted.append(ClarificationAnswerBinding(
            slot_id=slot.slot_id,
            role_hint=slot.role_hint,
            reason_code=slot.reason_code,
            response_kind=slot.response_kind,
            applies_to_item_ids=tuple(slot.applies_to_item_ids),
            mention_ids=tuple(slot.mention_ids),
            value=value,
            selected_option_label=(None if selected is None else selected.label),
        ))
    return tuple(accepted)


def _accepted_escape_answers(
        authority: ClarificationAuthority,
        action: Literal["none_of_above", "unknown"],
        ) -> tuple[ClarificationAnswerBinding, ...]:
    """Record an explicit escape against every unanswered slot."""

    label = "어느 것도 아님" if action == "none_of_above" else "모르겠음"
    return tuple(ClarificationAnswerBinding(
        slot_id=slot.slot_id,
        role_hint=slot.role_hint,
        reason_code=slot.reason_code,
        response_kind=slot.response_kind,
        applies_to_item_ids=tuple(slot.applies_to_item_ids),
        mention_ids=tuple(slot.mention_ids),
        value=f"__{action}__",
        selected_option_label=label,
    ) for slot in authority.slots)


def _validate_derived_intent(
        *,
        question: str,
        current: SemanticIntent,
        derived: SemanticIntent,
        answers: tuple[ClarificationAnswerBinding, ...],
        ) -> SemanticIntent:
    normalized = validate_semantic_intent_grounding(question, derived)
    current_payload = current.model_dump(mode="json", warnings=False)
    derived_payload = normalized.model_dump(mode="json", warnings=False)
    current_mentions = current_payload.pop("unresolved_mentions")
    derived_mentions = derived_payload.pop("unresolved_mentions")
    if derived_payload != current_payload:
        raise InvalidClarificationAnswerError(
            "resume backend은 원 질문의 semantic surface를 바꿀 수 없습니다")
    answered_ids = {
        mention_id for answer in answers for mention_id in answer.mention_ids}
    live_ids = {row["mention_id"] for row in current_mentions}
    if not answered_ids <= live_ids:
        raise InvalidClarificationAnswerError(
            "answered mention이 current source intent에 없습니다")
    expected_mentions = [
        row for row in current_mentions if row["mention_id"] not in answered_ids]
    if derived_mentions != expected_mentions:
        raise InvalidClarificationAnswerError(
            "derived intent는 답한 unresolved mention만 제거해야 합니다")
    return normalized


def _resume_result(value: Any) -> ClarificationResumeResult:
    try:
        return _strict_model(
            ClarificationResumeResult, value)  # type: ignore[return-value]
    except Exception as exc:
        raise ClarificationResumeTechnicalError(
            "resume backend 결과가 strict v1 계약을 통과하지 못했습니다") from exc


class Stage1V1ClarificationCoordinator:
    """Open, answer, re-resolve and atomically persist v1 clarification turns."""

    def __init__(
            self,
            backend: Stage1V1ClarificationResumeBackend,
            store: Stage1V1ClarificationStore,
            *,
            max_turns: int = MAX_CLARIFICATION_TURNS,
            current_canonical_build_id: str | None = None,
            ) -> None:
        if type(max_turns) is not int or max_turns <= 0:
            raise ValueError("max_turns는 양수여야 합니다")
        self.backend = backend
        self.store = store
        self.max_turns = max_turns
        if (current_canonical_build_id is not None
                and (len(current_canonical_build_id) != 32
                     or any(character not in "0123456789abcdef"
                            for character in current_canonical_build_id))):
            raise ValueError("current canonical build ID 형식이 잘못되었습니다")
        self.current_canonical_build_id = current_canonical_build_id

    def open(self, initial_outcome: Stage1Outcome | Mapping[str, Any]) -> ClarificationSessionView:
        outcome = _strict_model(
            Stage1Outcome, initial_outcome)  # type: ignore[assignment]
        assert isinstance(outcome, Stage1Outcome)
        if (outcome.status != "needs_clarification"
                or not isinstance(
                    outcome.resolution_decision.authority,
                    ClarificationAuthority)):
            raise ClarificationSessionError(
                "clarification session은 needs_clarification outcome으로 열어야 합니다")
        decision = outcome.resolution_decision
        state = _ClarificationSessionState(
            session_id=str(uuid4()),
            question_id=decision.question_id,
            original_question=decision.question,
            original_source_intent=decision.source_intent,
            original_source_intent_digest=decision.source_intent_digest,
            current_source_intent=decision.source_intent,
            initial_outcome=outcome,
            current_outcome=outcome,
            current_revision=0,
        )
        self.store.put_new(state)
        return self._view(self.store.get(state.session_id))

    def answer(
            self,
            answer: Stage1V1ClarificationAnswer | Mapping[str, Any],
            ) -> ClarificationSessionView:
        public = _strict_model(
            Stage1V1ClarificationAnswer, answer)  # type: ignore[assignment]
        assert isinstance(public, Stage1V1ClarificationAnswer)
        state = self.store.get(public.session_id)
        stored_build_id = (
            state.current_outcome.resolution_decision.canonical_build_id)
        if (self.current_canonical_build_id is not None
                and stored_build_id != self.current_canonical_build_id):
            raise ClarificationResumeTechnicalError(
                "clarification session의 canonical snapshot이 바뀌었습니다")
        authority = state.current_outcome.resolution_decision.authority
        if (state.current_outcome.status != "needs_clarification"
                or not isinstance(authority, ClarificationAuthority)):
            raise StaleClarificationAnswerError(
                "현재 답변할 clarification이 없습니다")
        if (
            public.clarification_id != _clarification_id(state)
            or public.expected_revision != state.current_revision
        ):
            raise StaleClarificationAnswerError(
                "clarification ID 또는 revision이 오래되었습니다")
        if len(state.history) >= self.max_turns:
            raise ClarificationTurnLimitError(
                "clarification 최대 turn에 도달했습니다")
        accepted = (
            _accepted_answers(authority, public.values)
            if public.action == "submit"
            else _accepted_escape_answers(authority, public.action)
        )
        decision = state.current_outcome.resolution_decision
        request = ClarificationTurnRequest(
            clarification_id=public.clarification_id,
            from_revision=state.current_revision,
            to_revision=state.current_revision + 1,
            previous_outcome_digest=state.current_outcome.outcome_digest,
            previous_decision_digest=decision.decision_digest,
            previous_source_intent_digest=decision.source_intent_digest,
            canonical_build_id=decision.canonical_build_id,
            resolver_version=decision.resolver_version,
            answers=accepted,
        )
        context = ClarificationResolutionContext.create(
            session_id=state.session_id,
            question_id=state.question_id,
            original_question_sha256=sha256(
                state.original_question.encode("utf-8")).hexdigest(),
            original_source_intent_digest=state.original_source_intent_digest,
            history=state.history,
            current=request,
        )

        try:
            if public.action != "submit":
                # An escape is a normal, auditable terminal outcome. It does
                # not guess a slot value and does not call a resume backend.
                derived = state.current_source_intent
                code = (
                    "corpus_coverage_unavailable"
                    if public.action == "none_of_above"
                    else "unsupported_semantic_target"
                )
                resumed_authority: ResolutionAuthority = TerminalAuthority(
                    reasons=[{
                        "code": code,
                        "scope": "question",
                        "item_ids": [
                            row.item_id
                            for row in state.current_source_intent.answer_items
                        ],
                    }]
                )
            else:
                # Pydantic's frozen models prevent attribute reassignment, but
                # list members remain mutable Python containers. Never lend a
                # store-owned copy to an untrusted resume backend.
                original_for_backend = _strict_model(
                    SemanticIntent, state.original_source_intent)
                current_for_backend = _strict_model(
                    SemanticIntent, state.current_source_intent)
                assert isinstance(original_for_backend, SemanticIntent)
                assert isinstance(current_for_backend, SemanticIntent)
                raw = self.backend.resume(
                    question_id=state.question_id,
                    question=state.original_question,
                    original_source_intent=original_for_backend,
                    current_source_intent=current_for_backend,
                    context=context,
                )
                resumed = _resume_result(raw)
                derived = _validate_derived_intent(
                    question=state.original_question,
                    current=state.current_source_intent,
                    derived=resumed.derived_source_intent,
                    answers=accepted,
                )
                resumed_authority = resumed.authority
            fresh_resolver = Stage1V1Resolver(
                _StaticAuthorityBackend(resumed_authority),
                canonical_build_id=decision.canonical_build_id,
                resolver_version=decision.resolver_version,
            )
            next_outcome = Stage1V1Orchestrator(fresh_resolver).run(
                question_id=state.question_id,
                question=state.original_question,
                source_intent=derived,
            )
        except (ClarificationSessionError, InvalidClarificationAnswerError):
            raise
        except Exception as exc:
            raise ClarificationResumeTechnicalError(
                "clarification resume resolver/orchestrator가 실패했습니다") from exc

        turn = ClarificationHistoryTurn(
            request=request,
            request_context_digest=context.context_digest,
            derived_source_intent_digest=semantic_intent_digest(derived),
            resumed_decision_digest=(
                next_outcome.resolution_decision.decision_digest),
            resumed_outcome_digest=next_outcome.outcome_digest,
            resumed_status=next_outcome.status,
        )
        next_state = _ClarificationSessionState(
            session_id=state.session_id,
            question_id=state.question_id,
            original_question=state.original_question,
            original_source_intent=state.original_source_intent,
            original_source_intent_digest=state.original_source_intent_digest,
            current_source_intent=derived,
            initial_outcome=state.initial_outcome,
            current_outcome=next_outcome,
            contexts=state.contexts + (context,),
            history=state.history + (turn,),
            current_revision=state.current_revision + 1,
        )
        self.store.compare_and_put(
            next_state,
            expected_revision=state.current_revision,
            expected_outcome_digest=state.current_outcome.outcome_digest,
        )
        return self._view(self.store.get(state.session_id))

    def inspect(self, session_id: str) -> ClarificationSessionView:
        return self._view(self.store.get(session_id))

    @staticmethod
    def _view(state: _ClarificationSessionState) -> ClarificationSessionView:
        outcome = state.current_outcome
        if outcome.status == "needs_clarification":
            return ClarificationSessionView(
                session_id=state.session_id,
                revision=state.current_revision,
                status=outcome.status,
                clarification=_public_clarification(state),
            )
        if outcome.status in {"ready", "partial_ready"}:
            ready = ClarificationReadyBinding.create(
                session_id=state.session_id,
                initial_outcome_digest=state.initial_outcome.outcome_digest,
                original_source_intent_digest=(
                    state.original_source_intent_digest),
                context_digests=tuple(
                    row.context_digest for row in state.contexts),
                final_outcome=outcome,
            )
            return ClarificationSessionView(
                session_id=state.session_id,
                revision=state.current_revision,
                status=outcome.status,
                ready=ready,
            )
        return ClarificationSessionView(
            session_id=state.session_id,
            revision=state.current_revision,
            status="terminal",
            terminal_outcome=outcome,
        )


class Stage1V1ClarificationRuntime:
    """The v1 shadow execution entry point with a typed clarification loop.

    This deliberately does not extend the legacy ``Stage1Service``.  Callers
    provide a question-grounded ``SemanticIntent`` and receive either the
    ordinary v1 outcome or a public clarification session view.  Only a
    ``ClarificationReadyBinding`` is returned after a clarification turn, so
    Stage2 cannot consume an unresolved or unbound intermediate outcome.

    The runtime has no question-ID policy.  Event-target selection, financial
    scope selection, and any later clarification all use the same authority
    slot and resume-backend contracts.
    """

    def __init__(
            self,
            orchestrator: Stage1V1Orchestrator,
            clarification_coordinator: Stage1V1ClarificationCoordinator,
            ) -> None:
        self.orchestrator = orchestrator
        self.clarification_coordinator = clarification_coordinator

    def start(
            self,
            *,
            question_id: str,
            question: str,
            source_intent: SemanticIntent | Mapping[str, Any],
            ) -> Stage1Outcome | ClarificationSessionView:
        """Run one v1 turn; open a sidecar only for typed clarification."""

        outcome = self.orchestrator.run(
            question_id=question_id,
            question=question,
            source_intent=source_intent,
        )
        if outcome.status == "needs_clarification":
            return self.clarification_coordinator.open(outcome)
        return outcome

    def answer_clarification(
            self,
            answer: Stage1V1ClarificationAnswer | Mapping[str, Any],
            ) -> ClarificationSessionView:
        """Resume an open v1 clarification without accepting plan patches."""

        return self.clarification_coordinator.answer(answer)

    def inspect_clarification(self, session_id: str) -> ClarificationSessionView:
        """Return the public state of one clarification session."""

        return self.clarification_coordinator.inspect(session_id)


__all__ = [
    "CLARIFICATION_CONTEXT_VERSION",
    "CLARIFICATION_READY_BINDING_VERSION",
    "CLARIFICATION_SESSION_VERSION",
    "MAX_CLARIFICATION_TURNS",
    "ClarificationAnswerBinding",
    "ClarificationHistoryTurn",
    "ClarificationReadyBinding",
    "ClarificationResolutionContext",
    "ClarificationResumeResult",
    "ClarificationResumeTechnicalError",
    "ClarificationSessionError",
    "ClarificationSessionNotFoundError",
    "ClarificationSessionView",
    "ClarificationTurnLimitError",
    "ClarificationTurnRequest",
    "InMemoryStage1V1ClarificationStore",
    "InvalidClarificationAnswerError",
    "PublicClarification",
    "SQLiteStage1V1ClarificationStore",
    "Stage1V1ClarificationStore",
    "Stage1V1ClarificationAnswer",
    "Stage1V1ClarificationCoordinator",
    "Stage1V1ClarificationRuntime",
    "Stage1V1ClarificationResumeBackend",
    "StaleClarificationAnswerError",
]
