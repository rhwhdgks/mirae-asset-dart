"""미래에셋 예선 평가 API의 QueryPlan 독립 실행 계약.

공식 외부 응답은 다섯 문자열 필드만 반환한다. 내부 backend는 아직 팀 QueryPlan을
모르며, 구조화된 AnswerMaterial projection과 고정 audit step만 gateway에 넘긴다.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from enum import StrEnum
import math
import os
from threading import BoundedSemaphore
import time
from typing import Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EVALUATION_API_VERSION = "evaluation-api/0.1"
MAX_QUESTION_ID_LENGTH = 128
MAX_QUESTION_LENGTH = 10_000
MAX_CONTEXT_LENGTH = 200_000
MAX_ANSWER_LENGTH = 50_000
QUESTION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _has_unsafe_control(value: str) -> bool:
    return any(ord(char) < 32 and char not in "\n\t" for char in value)


class EvaluationRequest(_StrictFrozenModel):
    question_id: str = Field(
        min_length=1, max_length=MAX_QUESTION_ID_LENGTH,
        pattern=QUESTION_ID_PATTERN)
    question: str = Field(
        min_length=1, max_length=MAX_QUESTION_LENGTH, repr=False)

    @field_validator("question")
    @classmethod
    def valid_question(cls, value: str) -> str:
        if not value.strip() or _has_unsafe_control(value):
            raise ValueError("question이 비어 있거나 제어문자를 포함합니다")
        return value


class AnswerDisposition(StrEnum):
    ANSWER = "answer"
    PARTIAL_ANSWER = "partial_answer"
    CLARIFICATION = "clarification"
    REFUSE = "refuse"
    UNSUPPORTED = "unsupported"


class AuditStep(StrEnum):
    QUESTION_ACCEPTED = "question_accepted"
    QUERY_PLAN_VALIDATED = "query_plan_validated"
    DATA_RETRIEVED = "data_retrieved"
    EVIDENCE_VERIFIED = "evidence_verified"
    ANSWER_COMPOSED = "answer_composed"


_AUDIT_ORDER = {step: index for index, step in enumerate(AuditStep)}
_AUDIT_LABELS = {
    AuditStep.QUESTION_ACCEPTED: "질문 접수",
    AuditStep.QUERY_PLAN_VALIDATED: "QueryPlan 검증",
    AuditStep.DATA_RETRIEVED: "정본 조회",
    AuditStep.EVIDENCE_VERIFIED: "근거 검증",
    AuditStep.ANSWER_COMPOSED: "답변 조립",
}


class EvaluationMaterial(_StrictFrozenModel):
    """B의 조립 계층이 반환할 provider-independent projection."""

    disposition: AnswerDisposition
    retrieved_context: str = Field(default="", max_length=MAX_CONTEXT_LENGTH, repr=False)
    audit_steps: tuple[AuditStep, ...] = Field(min_length=2, max_length=5)
    answer: str = Field(min_length=1, max_length=MAX_ANSWER_LENGTH, repr=False)

    @model_validator(mode="after")
    def validate_material(self) -> "EvaluationMaterial":
        if (not self.answer.strip() or _has_unsafe_control(self.answer)
                or _has_unsafe_control(self.retrieved_context)):
            raise ValueError("answer/context 문자열 계약 오류")
        if len(self.audit_steps) != len(set(self.audit_steps)):
            raise ValueError("audit step은 중복될 수 없습니다")
        if tuple(sorted(self.audit_steps, key=_AUDIT_ORDER.__getitem__)) != self.audit_steps:
            raise ValueError("audit step 순서가 잘못되었습니다")
        if (self.audit_steps[0] != AuditStep.QUESTION_ACCEPTED
                or self.audit_steps[-1] != AuditStep.ANSWER_COMPOSED):
            raise ValueError("audit trace 시작·종료 단계가 잘못되었습니다")
        evidence_steps = {
            AuditStep.DATA_RETRIEVED, AuditStep.EVIDENCE_VERIFIED,
        }
        has_verified_evidence = evidence_steps.issubset(self.audit_steps)
        factual = self.disposition in {
            AnswerDisposition.ANSWER, AnswerDisposition.PARTIAL_ANSWER,
        }
        if factual and (not self.retrieved_context.strip() or not has_verified_evidence):
            raise ValueError("factual answer에는 검색 근거와 Evidence 검증이 필요합니다")
        if not factual and (self.retrieved_context or evidence_steps.intersection(
                self.audit_steps)):
            raise ValueError("non-answer는 검색 근거를 확정 사실처럼 노출할 수 없습니다")
        return self


class EvaluationResponse(_StrictFrozenModel):
    """과제자료 8페이지의 성공 응답 다섯 필드와 byte-level로 맞춘다."""

    question_id: str
    question: str = Field(repr=False)
    retrieved_context: str = Field(repr=False)
    think_trace: str = Field(repr=False)
    answer: str = Field(repr=False)


class ApiErrorDetail(_StrictFrozenModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    message: str = Field(min_length=1, max_length=200)


class ApiErrorResponse(_StrictFrozenModel):
    error: ApiErrorDetail


class BackendReadiness(_StrictFrozenModel):
    ready: bool
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")


class EvaluationBackend(Protocol):
    def readiness(self) -> BackendReadiness: ...

    def answer(
            self, request: EvaluationRequest, *, deadline_monotonic: float,
            ) -> EvaluationMaterial: ...


class AnswerPostValidator(Protocol):
    def validate(
            self, request: EvaluationRequest,
            material: EvaluationMaterial,
            ) -> EvaluationMaterial: ...


class StructuralAnswerValidator:
    """QueryPlan/AnswerMaterial 확정 전 적용 가능한 최소 후검증기."""

    def validate(
            self, request: EvaluationRequest,
            material: EvaluationMaterial,
            ) -> EvaluationMaterial:
        del request
        if not isinstance(material, EvaluationMaterial):
            raise AnswerValidationError("backend output type이 잘못되었습니다")
        # frozen Pydantic 객체라도 호출 지점에서 전체 모델 validator를 다시 실행한다.
        try:
            return EvaluationMaterial.model_validate(
                material.model_dump(mode="python"), strict=True)
        except ValueError as exc:
            raise AnswerValidationError("answer material 구조 검증에 실패했습니다") from exc


class UnavailableEvaluationBackend:
    """팀 QueryPlan adapter가 주입되기 전 기본 backend."""

    def readiness(self) -> BackendReadiness:
        return BackendReadiness(ready=False, code="query_plan_not_configured")

    def answer(
            self, request: EvaluationRequest, *, deadline_monotonic: float,
            ) -> EvaluationMaterial:
        del request, deadline_monotonic
        raise BackendNotReadyError("QueryPlan backend가 구성되지 않았습니다")


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    max_concurrency: int = 4
    max_queue: int = 8
    queue_timeout_seconds: float = 1.0
    request_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if type(self.max_concurrency) is not int or not 1 <= self.max_concurrency <= 64:
            raise ValueError("max_concurrency는 1..64여야 합니다")
        if type(self.max_queue) is not int or not 0 <= self.max_queue <= 1024:
            raise ValueError("max_queue는 0..1024여야 합니다")
        for name, value, maximum in (
                ("queue_timeout_seconds", self.queue_timeout_seconds, 30.0),
                ("request_timeout_seconds", self.request_timeout_seconds, 300.0)):
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or not 0 < value <= maximum):
                raise ValueError(f"{name} 범위가 잘못되었습니다")
        if self.queue_timeout_seconds > self.request_timeout_seconds:
            raise ValueError("queue timeout은 전체 request timeout보다 길 수 없습니다")

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        try:
            return cls(
                max_concurrency=int(os.environ.get(
                    "MIRAE_API_MAX_CONCURRENCY", "4")),
                max_queue=int(os.environ.get("MIRAE_API_MAX_QUEUE", "8")),
                queue_timeout_seconds=float(os.environ.get(
                    "MIRAE_API_QUEUE_TIMEOUT_SECONDS", "1.0")),
                request_timeout_seconds=float(os.environ.get(
                    "MIRAE_API_REQUEST_TIMEOUT_SECONDS", "60.0")),
            )
        except ValueError as exc:
            raise ValueError("평가 API 환경변수 형식이 잘못되었습니다") from exc


class EvaluationGatewayError(RuntimeError):
    code = "evaluation_gateway_error"
    http_status = 500
    public_message = "평가 요청 처리에 실패했습니다"


class InvalidEvaluationRequestError(EvaluationGatewayError):
    code = "invalid_request"
    http_status = 422
    public_message = "question_id 또는 question 형식이 잘못되었습니다"


class BackendNotReadyError(EvaluationGatewayError):
    code = "backend_not_ready"
    http_status = 503
    public_message = "평가 backend가 준비되지 않았습니다"


class GatewayBusyError(EvaluationGatewayError):
    code = "server_busy"
    http_status = 429
    public_message = "처리 중인 요청이 많습니다"


class GatewayTimeoutError(EvaluationGatewayError):
    code = "request_timeout"
    http_status = 504
    public_message = "평가 요청 처리 시간이 초과되었습니다"


class BackendExecutionError(EvaluationGatewayError):
    code = "backend_execution_failed"
    http_status = 500
    public_message = "평가 backend 실행에 실패했습니다"


class AnswerValidationError(EvaluationGatewayError):
    code = "answer_validation_failed"
    http_status = 502
    public_message = "최종 답변 검증에 실패했습니다"


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    response: EvaluationResponse = field(repr=False)
    queue_wait_ms: float
    backend_latency_ms: float
    validation_latency_ms: float
    total_latency_ms: float


@dataclass(frozen=True, slots=True)
class _BackendExecution:
    material: EvaluationMaterial = field(repr=False)
    started_at: float
    backend_finished_at: float
    validation_finished_at: float


class EvaluationGateway:
    """동기 backend를 bounded worker pool에서 실행하는 fail-closed gateway."""

    def __init__(
            self, backend: EvaluationBackend, *,
            validator: AnswerPostValidator | None = None,
            settings: GatewaySettings | None = None,
            clock: Callable[[], float] = time.monotonic,
            ) -> None:
        self.backend = backend
        self.validator = validator or StructuralAnswerValidator()
        self.settings = settings or GatewaySettings.from_env()
        self._clock = clock
        self._executor = ThreadPoolExecutor(
            max_workers=self.settings.max_concurrency,
            thread_name_prefix="mirae-eval")
        self._slots = BoundedSemaphore(
            self.settings.max_concurrency + self.settings.max_queue)
        self._closed = False

    def _run_backend(
            self, request: EvaluationRequest, deadline: float,
            ) -> _BackendExecution:
        started = self._clock()
        if started >= deadline:
            raise GatewayTimeoutError("backend 실행 전에 deadline을 초과했습니다")
        try:
            material = self.backend.answer(
                request, deadline_monotonic=deadline)
        except EvaluationGatewayError:
            raise
        except Exception as exc:
            raise BackendExecutionError("backend가 예외를 반환했습니다") from exc
        backend_finished = self._clock()
        try:
            validated = self.validator.validate(request, material)
        except EvaluationGatewayError:
            raise
        except Exception as exc:
            raise AnswerValidationError("후검증기가 예외를 반환했습니다") from exc
        return _BackendExecution(
            material=validated,
            started_at=started,
            backend_finished_at=backend_finished,
            validation_finished_at=self._clock(),
        )

    def readiness(self) -> BackendReadiness:
        if self._closed:
            return BackendReadiness(ready=False, code="gateway_closed")
        try:
            result = self.backend.readiness()
        except Exception:
            return BackendReadiness(ready=False, code="backend_readiness_failed")
        if not isinstance(result, BackendReadiness):
            return BackendReadiness(ready=False, code="backend_readiness_invalid")
        return result

    def evaluate(self, request: EvaluationRequest) -> EvaluationOutcome:
        if self._closed:
            raise BackendNotReadyError("gateway가 종료됐습니다")
        readiness = self.readiness()
        if not readiness.ready:
            raise BackendNotReadyError("backend readiness가 false입니다")
        started = self._clock()
        acquired = self._slots.acquire(timeout=min(
            self.settings.queue_timeout_seconds,
            self.settings.request_timeout_seconds))
        if not acquired:
            raise GatewayBusyError("queue capacity를 초과했습니다")
        deadline = started + self.settings.request_timeout_seconds
        future: Future[_BackendExecution]
        try:
            future = self._executor.submit(
                self._run_backend, request, deadline)
        except Exception as exc:
            self._slots.release()
            raise BackendExecutionError("backend submit에 실패했습니다") from exc
        future.add_done_callback(lambda _: self._slots.release())
        try:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise FutureTimeout
            execution = future.result(timeout=remaining)
        except FutureTimeout as exc:
            future.cancel()
            raise GatewayTimeoutError("request deadline을 초과했습니다") from exc
        except EvaluationGatewayError:
            raise
        except Exception as exc:
            raise BackendExecutionError("backend가 예외를 반환했습니다") from exc
        if execution.validation_finished_at > deadline:
            raise GatewayTimeoutError("완료 결과가 request deadline 이후 도착했습니다")
        think_trace = " → ".join(
            _AUDIT_LABELS[step] for step in execution.material.audit_steps)
        think_trace += " → 출력 후검증"
        finished = self._clock()
        return EvaluationOutcome(
            response=EvaluationResponse(
                question_id=request.question_id,
                question=request.question,
                retrieved_context=execution.material.retrieved_context,
                think_trace=think_trace,
                answer=execution.material.answer,
            ),
            queue_wait_ms=max(0.0, (execution.started_at - started) * 1000.0),
            backend_latency_ms=max(
                0.0, (execution.backend_finished_at - execution.started_at) * 1000.0),
            validation_latency_ms=max(
                0.0, (execution.validation_finished_at
                      - execution.backend_finished_at) * 1000.0),
            total_latency_ms=max(0.0, (finished - started) * 1000.0),
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = [
    "AnswerDisposition", "AnswerPostValidator", "AnswerValidationError",
    "ApiErrorDetail", "ApiErrorResponse", "AuditStep", "BackendExecutionError",
    "BackendNotReadyError", "BackendReadiness", "EVALUATION_API_VERSION",
    "EvaluationBackend", "EvaluationGateway", "EvaluationGatewayError",
    "EvaluationMaterial", "EvaluationOutcome", "EvaluationRequest",
    "EvaluationResponse", "GatewayBusyError", "GatewaySettings",
    "GatewayTimeoutError", "InvalidEvaluationRequestError",
    "MAX_ANSWER_LENGTH", "MAX_CONTEXT_LENGTH", "MAX_QUESTION_ID_LENGTH",
    "MAX_QUESTION_LENGTH", "QUESTION_ID_PATTERN", "StructuralAnswerValidator",
    "UnavailableEvaluationBackend",
]
