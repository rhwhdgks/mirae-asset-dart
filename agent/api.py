"""미래에셋 예선 평가용 FastAPI shell.

실행 예: ``uvicorn agent.api:app --host 0.0.0.0 --port 8000 --no-access-log``.
기본 app은 QueryPlan backend가 없어 readiness 503으로 fail-closed한다.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .evaluation_api import (
    AnswerPostValidator,
    ApiErrorDetail,
    ApiErrorResponse,
    BackendReadiness,
    EVALUATION_API_VERSION,
    EvaluationBackend,
    EvaluationGateway,
    EvaluationGatewayError,
    EvaluationRequest,
    EvaluationResponse,
    GatewaySettings,
    InvalidEvaluationRequestError,
    MAX_QUESTION_ID_LENGTH,
    MAX_QUESTION_LENGTH,
    QUESTION_ID_PATTERN,
    UnavailableEvaluationBackend,
)


def _error_response(code: str, message: str, status_code: int) -> JSONResponse:
    body = ApiErrorResponse(error=ApiErrorDetail(code=code, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def create_app(
        backend: EvaluationBackend | None = None, *,
        validator: AnswerPostValidator | None = None,
        settings: GatewaySettings | None = None,
        ) -> FastAPI:
    gateway = EvaluationGateway(
        backend or UnavailableEvaluationBackend(),
        validator=validator,
        settings=settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        gateway.close()

    application = FastAPI(
        title="Mirae DART Evaluation API",
        version=EVALUATION_API_VERSION,
        lifespan=lifespan,
    )
    application.state.evaluation_gateway = gateway

    @application.middleware("http")
    async def secure_response_headers(request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
            _: Request, __: RequestValidationError) -> JSONResponse:
        return _error_response(
            "invalid_request", "question_id 또는 question 형식이 잘못되었습니다", 422)

    @application.exception_handler(EvaluationGatewayError)
    async def gateway_error_handler(
            _: Request, error: EvaluationGatewayError) -> JSONResponse:
        return _error_response(error.code, error.public_message, error.http_status)

    @application.exception_handler(Exception)
    async def internal_error_handler(_: Request, __: Exception) -> JSONResponse:
        return _error_response(
            "internal_error", "평가 요청 처리 중 내부 오류가 발생했습니다", 500)

    @application.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "service_version": EVALUATION_API_VERSION}

    @application.get("/readyz")
    def readyz(response: Response) -> dict[str, str]:
        readiness: BackendReadiness = gateway.readiness()
        if not readiness.ready:
            response.status_code = 503
        return {
            "status": "ready" if readiness.ready else "not_ready",
            "code": readiness.code,
        }

    @application.get(
        "/answer",
        response_model=EvaluationResponse,
        responses={
            422: {"model": ApiErrorResponse},
            429: {"model": ApiErrorResponse},
            500: {"model": ApiErrorResponse},
            502: {"model": ApiErrorResponse},
            503: {"model": ApiErrorResponse},
            504: {"model": ApiErrorResponse},
        },
    )
    def answer(
            response: Response,
            question_id: Annotated[str, Query(
                min_length=1, max_length=MAX_QUESTION_ID_LENGTH,
                pattern=QUESTION_ID_PATTERN)],
            question: Annotated[str, Query(
                min_length=1, max_length=MAX_QUESTION_LENGTH)],
            ) -> EvaluationResponse:
        try:
            evaluation_request = EvaluationRequest(
                question_id=question_id, question=question)
        except ValidationError as exc:
            raise InvalidEvaluationRequestError(
                "request model validation 실패") from exc
        outcome = gateway.evaluate(evaluation_request)
        response.headers["Server-Timing"] = (
            f"queue;dur={outcome.queue_wait_ms:.3f}, "
            f"backend;dur={outcome.backend_latency_ms:.3f}, "
            f"validate;dur={outcome.validation_latency_ms:.3f}, "
            f"total;dur={outcome.total_latency_ms:.3f}")
        return outcome.response

    return application


app = create_app()


__all__ = ["app", "create_app"]
