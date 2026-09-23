"""도메인 예외 → HTTP 오류 매핑. 라우터는 `to_api_exception(exc)` 로 옮겨 raise 한다 (`api_v1._call`).

오류 본문은 항상 `{"error": {"code", "message", "detail"?}}` (`server/schemas.py:ApiError`).
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from server.runtime import ExecTimeout, NotReady, QueueTimeout, SearchIndexUnavailable, Stage1Unavailable


class ApiException(Exception):
    def __init__(self, status_code: int, code: str, message: str, detail: Any | None = None):
        super().__init__(message)
        self.status_code, self.code, self.message, self.detail = status_code, code, message, detail

    def response(self) -> JSONResponse:
        body: dict = {"error": {"code": self.code, "message": self.message}}
        if self.detail is not None:
            body["error"]["detail"] = jsonable_encoder(self.detail)
        return JSONResponse(body, status_code=self.status_code)


def to_api_exception(exc: BaseException) -> ApiException:
    """typed 도메인 예외를 HTTP 코드로. 알 수 없는 예외는 500 으로 닫되 유형만 노출한다."""
    from agent.stage1_v1_clarification_session import (
        ClarificationResumeTechnicalError, ClarificationSessionNotFoundError, ClarificationTurnLimitError,
        InvalidClarificationAnswerError, StaleClarificationAnswerError,
    )
    msg = str(exc)[:300]
    if isinstance(exc, ApiException):
        return exc
    if isinstance(exc, NotReady):
        # Do not expose filesystem layout (or deployment paths) in a public error.
        return ApiException(503, "not_ready", "서비스 준비 중입니다(정본·인덱스 로딩). 잠시 후 다시 요청해 주세요.")
    if isinstance(exc, QueueTimeout):
        return ApiException(503, "busy", "동시 처리 한도로 지연되었습니다. 잠시 후 다시 요청해 주세요.")
    if isinstance(exc, ExecTimeout):
        return ApiException(504, "timeout", f"제한 시간 안에 처리를 완료하지 못했습니다: {msg}")
    if isinstance(exc, Stage1Unavailable):
        return ApiException(503, "stage1_unavailable", msg)
    if isinstance(exc, SearchIndexUnavailable):
        return ApiException(503, "search_index_unavailable", msg)
    if isinstance(exc, ClarificationSessionNotFoundError):
        return ApiException(404, "session_not_found", msg)
    if isinstance(exc, StaleClarificationAnswerError):
        return ApiException(409, "stale_revision", msg)
    if isinstance(exc, ClarificationTurnLimitError):
        return ApiException(409, "turn_limit", msg)
    if isinstance(exc, InvalidClarificationAnswerError):
        return ApiException(422, "invalid_answer", msg)
    if isinstance(exc, ClarificationResumeTechnicalError):
        return ApiException(500, "resume_failed", msg)
    if isinstance(exc, ValidationError):
        return ApiException(422, "invalid_request", "요청 본문이 계약과 맞지 않습니다", detail=exc.errors(include_url=False))
    if isinstance(exc, KeyError):
        return ApiException(404, "not_found", f"찾을 수 없습니다: {msg}")
    if isinstance(exc, (ValueError, TypeError)):
        return ApiException(422, "invalid_argument", msg)
    if isinstance(exc, NotImplementedError):      # read.py NotReadyError (산출물 없음)
        return ApiException(503, "artifact_unavailable", msg)
    if isinstance(exc, LookupError):              # read.py Ambiguous*Error
        return ApiException(409, "ambiguous", msg)
    return ApiException(500, "internal_error", f"내부 오류: {type(exc).__name__}")


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiException)
    async def _api_exc(_: Request, exc: ApiException) -> JSONResponse:
        return exc.response()

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return ApiException(422, "invalid_request", "요청 형식이 잘못되었습니다", detail=exc.errors()).response()


__all__ = ["ApiException", "to_api_exception", "register_error_handlers"]
