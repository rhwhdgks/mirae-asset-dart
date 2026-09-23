"""공식 평가 API + `/v1` 개발·검수 API.

공식 계약 (주최 측 클라이언트: `requests.get(URL, params={question_id, question})`)
    GET /answer?question_id=&question=  → question_id · question · retrieved_context · think_trace · answer
    설계서 v3 §7: 45s 실행 컷 → 보유 근거 폴백, 55s 한계 고지(무응답 금지). 요청별 JSONL 로그.
    /healthz(프로세스) · /readyz(정본 build·인덱스·Stage1 상태, 준비 전 503).

경로: question → Stage1(HCX-007 SemanticIntent → resolver → QueryPlanHandoff 0.4)
             → Stage2~4(typed tool 조회 → Evidence 검증 → 계산·전제검증 → HCX-005/템플릿 문장화)

역질문(native)일 때도 GET body 는 정확히 5필드다. 재개에 필요한 최소 공개 authority(opaque session/
clarification id·revision·base64url slot 묶음)만 ``X-Clarification-*`` 헤더에 싣고,
`POST /answer/resume` 은 그 세션에 typed 답을 적용해 다음 turn(또는 확정 답변)을 같은 계약으로 돌려준다.
resume 은 release fixture 로 폴백하지 않는다 — 내구 native 세션만 authority 다.

`/v1/*` (server/api_v1.py) 는 같은 런타임 위에서 단계별 진입점을 연다. 컴포넌트 수명·단일 실행 스레드·예산·
슬롯 수명 가드·메모리 반환은 `server/runtime.py:ServerRuntime` 이 맡는다. 서버는 read.py safe 경로만 사용한다.

실행: ./scripts/serve.sh (uvicorn server.app:app) · 문서: /docs
"""
from __future__ import annotations

import base64
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from server.api_v1 import router as v1_router
from server.errors import ApiException, register_error_handlers
from server.runtime import (
    MSG_MISSING_PARAMS, SERVICE_NAME, SERVICE_VERSION, AnswerResult, ExecTimeout, NotReady, QueueTimeout,
    ServerRuntime, limit_response,
)

_TAGS = [
    {"name": "official", "description": "주최 측 계약 — 5필드 응답, 무응답 금지, 역질문 재개"},
    {"name": "status", "description": "진행 현황"},
    {"name": "answer", "description": "question → 답변 (JSON body, meta·payload 포함)"},
    {"name": "stage1", "description": "질문 해석 · 역질문 세션 (HCX-007)"},
    {"name": "stage2", "description": "QueryPlanHandoff → 조회·검증·계산·문장화"},
    {"name": "fixtures", "description": "release 70문항 조회·실행·채점"},
    {"name": "tools", "description": "정본 read model 직접 조회 (safe 경로)"},
    {"name": "eval", "description": "채점표"},
    {"name": "requests", "description": "요청 로그"},
]


class ClarificationResumeRequest(BaseModel):
    """공식 `POST /answer/resume` 본문 — typed Stage1 역질문 한 turn."""

    model_config = ConfigDict(extra="forbid", strict=True)

    session_id: str = Field(min_length=1, max_length=128)
    clarification_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=0)
    answers: dict[str, str] = Field(default_factory=dict)
    action: Literal["submit", "none_of_above", "unknown"] = "submit"


def _clarification_headers(meta: dict) -> dict[str, str]:
    """공개 resume authority 만 ASCII-safe 헤더로 싣는다.

    GET /answer 의 JSON body 는 정확히 5필드인 평가 계약이다. URL-safe base64 JSON slot 묶음으로
    typed 대화 authority 를 전달하되 세션 상태·digest·추가 JSON 필드는 body 에 넣지 않는다.
    """
    clarification = (meta.get("stage1_meta") or {}).get("clarification")
    if not isinstance(clarification, dict):
        return {}
    slots = json.dumps(clarification["slots"], ensure_ascii=False, separators=(",", ":"),
                       allow_nan=False).encode("utf-8")
    return {
        "X-Clarification-Session-Id": clarification["session_id"],
        "X-Clarification-Id": clarification["clarification_id"],
        "X-Clarification-Revision": str(clarification["revision"]),
        "X-Clarification-Slots": base64.urlsafe_b64encode(slots).decode("ascii"),
    }


def _official_response(result: AnswerResult) -> JSONResponse:
    return JSONResponse(result.response, status_code=result.http_status,
                        headers=_clarification_headers(result.meta))


_RESUME_MESSAGES = {
    "clarification_session_not_found": "역질문 세션을 찾을 수 없습니다.",
    "clarification_revision_conflict": "세션 revision 이 현재 상태와 맞지 않거나 turn 한도를 넘었습니다.",
    "invalid_clarification_answer": "역질문 답이 허용 후보와 맞지 않습니다.",
    "clarification_resume_unavailable": "역질문 재개를 처리할 수 없습니다.",
    "service_not_ready": "서비스 준비 중입니다(정본·인덱스 로딩). 잠시 후 다시 요청해 주세요.",
    "queue_timeout": "동시 처리 한도로 지연되었습니다. 잠시 후 다시 요청해 주세요.",
    "clarification_resume_timeout": "제한 시간 안에 역질문 재개를 완료하지 못했습니다.",
}


def _resume_error(status: int, code: str) -> ApiException:
    """`POST /answer/resume` 오류 — `/v1` 과 같은 `{"error": {"code", "message"}}` 봉투. 메시지는 고정 문구만
    쓴다(세션 저장소·digest 등 내부 상태를 예외 메시지로 노출하지 않는다)."""
    return ApiException(status, code, _RESUME_MESSAGES[code])


def _typed_resume_error(exc: Exception) -> ApiException:
    """Stage1 의 typed 세션 실패를 HTTP 코드로 옮긴다 (fail-closed). 모르는 예외는 503."""
    from agent.stage1_v1_clarification_session import (
        ClarificationSessionNotFoundError, ClarificationTurnLimitError,
        InvalidClarificationAnswerError, StaleClarificationAnswerError,
    )
    if isinstance(exc, ClarificationSessionNotFoundError):
        return _resume_error(404, "clarification_session_not_found")
    if isinstance(exc, (StaleClarificationAnswerError, ClarificationTurnLimitError)):
        return _resume_error(409, "clarification_revision_conflict")
    if isinstance(exc, (InvalidClarificationAnswerError, ValidationError)):
        return _resume_error(422, "invalid_clarification_answer")
    return _resume_error(503, "clarification_resume_unavailable")


def create_app(runtime: ServerRuntime | None = None, *, autostart: bool = True) -> FastAPI:
    """앱 팩토리. 테스트는 `runtime` 에 가짜 컴포넌트를 `install()` 한 런타임을 넣고 `autostart=False` 로 만든다."""
    rt = runtime or ServerRuntime.from_env()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if autostart:
            rt.start()          # 정본·arrow 초기화를 실행 스레드에서 — 이후 모든 조회가 같은 스레드
        yield
        rt.close()

    app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION, lifespan=lifespan, openapi_tags=_TAGS,
                  description="DART 공시 질의응답 Agent — 공식 `GET /answer` 5필드(+역질문 재개)와 단계별 `/v1` API. "
                              "확정 답변은 status ok + coverage complete + 검증된 Evidence citation 일 때만 낸다.")
    app.state.runtime = rt
    register_error_handlers(app)

    @app.get("/healthz", tags=["official"], summary="프로세스 생존")
    def healthz():
        return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

    @app.get("/readyz", tags=["official"], summary="정본·인덱스·Stage1 준비 상태 (준비 전 503)")
    def readyz():
        if not rt.ready:
            return JSONResponse({"ready": False, "error": rt.error}, status_code=503)
        return {"ready": True, **rt.build}

    @app.get("/answer", tags=["official"],
             summary="공식 평가 엔드포인트 — 5필드 (question_id · question · retrieved_context · think_trace · answer)")
    def answer(question_id: str = Query(default=""), question: str = Query(default="")):
        req_id = str(uuid4()); started = time.time()
        if not question_id.strip() or not question.strip():
            # 주최 계약은 5필드 고정이다. 파라미터가 비어도 오류 봉투 대신 한계 고지로
            # 5필드를 돌려준다(무응답 금지와 같은 원칙).
            result = AnswerResult(
                limit_response(question_id, question, MSG_MISSING_PARAMS, ""),
                "unresolved", 200, meta={"final_status": "unresolved", "missing_params": True})
            rt.log({"req_id": req_id, "endpoint": "/answer", "question_id": question_id,
                    "question": question, "status": "invalid_params", "elapsed": 0.0})
            return _official_response(result)
        result = rt.answer(question_id, question)
        elapsed = round(time.time() - started, 3)
        rt.log({"req_id": req_id, "endpoint": "/answer", "question_id": question_id, "question": question,
                "status": result.status, "elapsed": elapsed, **result.meta,
                "answer_len": len(result.response.get("answer", "")),
                "trace_lines": result.response.get("think_trace", "").count("\n") + 1})
        return _official_response(result)

    @app.post("/answer/resume", tags=["official"],
              summary="역질문 재개 — 열린 native 세션에 typed 답을 적용하고 다음 turn/확정 답변을 5필드로")
    def resume_clarification(body: ClarificationResumeRequest):
        """release fixture 로 폴백하지 않는다 — 내구 native Stage1 세션만 authority 다.
        세션 없음 404 · revision 불일치/turn 한도 409 · 후보 밖 값 422 · 준비 전/큐 초과/시간 초과 503."""
        req_id = str(uuid4()); started = time.time()
        log = {"req_id": req_id, "endpoint": "/answer/resume", "session_id": body.session_id}
        try:
            result = rt.resume(session_id=body.session_id, clarification_id=body.clarification_id,
                               revision=body.revision, answers=body.answers, action=body.action)
        except NotReady:
            rt.log({**log, "status": "resume_not_ready", "elapsed": round(time.time() - started, 3)})
            return _resume_error(503, "service_not_ready").response()
        except QueueTimeout:
            rt.log({**log, "status": "resume_queue_timeout"})
            return _resume_error(503, "queue_timeout").response()
        except ExecTimeout:
            rt.log({**log, "status": "resume_timeout", "final_status": "timeout"})
            return _resume_error(503, "clarification_resume_timeout").response()
        except Exception as exc:  # typed 세션 예외는 fail-closed 로 매핑
            error = _typed_resume_error(exc)
            rt.log({**log, "status": "resume_error", "final_status": "resume_error",
                    "error_code": error.code, "elapsed": round(time.time() - started, 3)})
            return error.response()
        rt.log({**log, "status": "resume_ok", "elapsed": round(time.time() - started, 3), **result.meta,
                "answer_len": len(result.response.get("answer", ""))})
        return _official_response(result)

    app.include_router(v1_router)
    return app


app = create_app()

__all__ = ["app", "create_app", "ClarificationResumeRequest"]
