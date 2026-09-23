"""`/v1` — 개발·검수용 API. 공식 `GET /answer` 5필드 계약은 `server/app.py` 에 그대로 있고, 여기는 그 위에
파이프라인의 각 단계를 **따로** 열어 둔다 (정확도 검수·역질문 turn·fixture 재생·정본 조회·채점표).

경로 (자세한 설명은 docs/api.md)
  status      GET  /v1/status                      진행 현황 — 정본·인덱스·Stage1·composer·채점표·요청 통계·세션
  answer      POST /v1/answer                      question → 5필드 + meta(+payload). stage1/compose 모드 선택
  stage1      POST /v1/stage1/interpret            question → QueryPlanHandoff v0.4 (Stage2 실행 없음)
              POST /v1/stage1/clarify              역질문 세션 재개 (HCX 미호출) → 다음 역질문 또는 답변
              GET  /v1/stage1/sessions/{id}        세션의 현재 공개 상태
  stage2      POST /v1/stage2/execute              handoff JSON → Stage2~4 (조회·검증·계산·문장화)
  fixtures    GET  /v1/fixtures[/{id}][/run]       release 70문항 조회·실행·AnswerRequirement 채점
  tools       GET  /v1/tools/{companies,lookup,corrections,events,document-version,search}
  eval        GET  /v1/eval/score                  최근 harness 성적표 파일
  requests    GET  /v1/requests                    요청별 JSONL 로그 tail

모든 정본 접근은 `ServerRuntime.submit` 을 통해 단일 실행 스레드에서 일어난다.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Path as PathParam, Query, Request, Response

from server import schemas as S
from server.errors import ApiException, to_api_exception
from server.runtime import ROOT, AnswerResult, ServerRuntime
from server.stage1 import Stage1Resolution
from agent.stage1_assembly import CORPUS_CUTOFF as DEFAULT_AS_OF   # 코퍼스 끝 = 기본 기준시점


router = APIRouter(prefix="/v1")
_ERR = {422: {"model": S.ApiError}, 503: {"model": S.ApiError}, 504: {"model": S.ApiError}}


def _rt(request: Request) -> ServerRuntime:
    return request.app.state.runtime


def _call(fn, *args, **kwargs):
    """런타임 호출 하나를 감싸 도메인 예외를 HTTP 오류로 옮긴다."""
    try:
        return fn(*args, **kwargs)
    except ApiException:
        raise
    except Exception as e:  # noqa
        raise to_api_exception(e) from e


def _jsonable(v: Any) -> Any:
    return json.loads(json.dumps(v, ensure_ascii=False, default=str))


# ── 응답 조립 ─────────────────────────────────────────────────────────────────

def _session(res: Stage1Resolution | None) -> S.SessionRef | None:
    return S.SessionRef(**res.session) if res is not None and res.session else None


def _clarification(res: Stage1Resolution | None) -> S.ClarificationOut | None:
    if res is None or not res.clarification:
        return None
    c = res.clarification
    return S.ClarificationOut(
        clarification_id=c["clarification_id"], plan_revision=c["plan_revision"],
        message=res.meta.get("clarification_message"),
        slots=[S.ClarificationSlotOut(
            slot_id=s["slot_id"], role_hint=s["role_hint"], reason_code=s["reason_code"],
            response_kind=s["response_kind"], prompt=s["prompt"],
            options=[S.ClarificationOptionOut(value=str(o["value"]), label=o["label"], reason=o.get("reason"))
                     for o in s.get("options", [])])
            for s in c["slots"]])


def _stage1_summary(res: Stage1Resolution | None) -> S.Stage1Summary | None:
    if res is None:
        return None
    return S.Stage1Summary(source=res.source,
                           handoff_status=res.handoff.status if res.handoff is not None else None,
                           matched_fixture=res.question_id_hint, session=_session(res), meta=_jsonable(res.meta))


def _answer_response(result: AnswerResult, *, request_id: str, total_s: float, include_payload: bool) -> S.AnswerResponse:
    m = result.meta
    meta = S.AnswerMeta(
        request_id=request_id, status=result.status, final_status=m.get("final_status"),
        stage1=_stage1_summary(result.stage1), claims=m.get("claims"),
        limitations=list(m.get("limitations", [])), used_documents=list(m.get("used_documents", [])),
        compose=m.get("compose"),
        timing={"stage1_s": m.get("t_stage1"), "pipeline_s": m.get("t_pipeline"), "total_s": round(total_s, 3)})
    payload = result.payload.model_dump(mode="json") if include_payload and result.payload is not None else None
    return S.AnswerResponse(**result.response, meta=meta, payload=payload)


def _handoff_dump(res: Stage1Resolution | None) -> dict | None:
    return res.handoff.model_dump(mode="json") if res is not None and res.handoff is not None else None


def _log_answer(rt: ServerRuntime, *, endpoint: str, request_id: str, question_id: str, question: str,
                result: AnswerResult, elapsed: float) -> None:
    rt.log({"req_id": request_id, "endpoint": endpoint, "question_id": question_id, "question": question,
            "status": result.status, "elapsed": round(elapsed, 3), **_jsonable(result.meta),
            "answer_len": len(result.response.get("answer", "")),
            "trace_lines": result.response.get("think_trace", "").count("\n") + 1})


# ── status ───────────────────────────────────────────────────────────────────

@router.get("/status", response_model=S.StatusResponse, tags=["status"],
            summary="진행 현황 — 정본·인덱스·Stage1·composer·채점표·요청 통계·역질문 세션")
def status(request: Request) -> dict:
    return _rt(request).status()


# ── answer ───────────────────────────────────────────────────────────────────

@router.post("/answer", response_model=S.AnswerResponse, tags=["answer"], responses=_ERR,
             summary="question → Stage1 → Stage2~4 → 5필드 + meta (JSON body 판)")
def answer(body: S.AnswerRequest, request: Request, response: Response) -> S.AnswerResponse:
    rt = _rt(request); req_id = str(uuid4()); t0 = time.time()
    if body.stage1 == "fixture":
        _call(rt.require_fixture_access)
    result = rt.answer(body.question_id, body.question, stage1_mode=body.stage1, compose=body.compose)
    elapsed = time.time() - t0
    _log_answer(rt, endpoint="/v1/answer", request_id=req_id, question_id=body.question_id,
                question=body.question, result=result, elapsed=elapsed)
    response.status_code = result.http_status
    return _answer_response(result, request_id=req_id, total_s=elapsed, include_payload=body.include_payload)


# ── stage1 ───────────────────────────────────────────────────────────────────

@router.post("/stage1/interpret", response_model=S.InterpretResponse, tags=["stage1"], responses=_ERR,
             summary="question → QueryPlanHandoff v0.4 (HCX-007 1회, Stage2 실행 없음)")
def interpret(body: S.InterpretRequest, request: Request) -> S.InterpretResponse:
    rt = _rt(request); req_id = str(uuid4()); t0 = time.time()
    res: Stage1Resolution = _call(rt.interpret, body.question_id, body.question, stage1_mode=body.stage1)
    rt.log({"req_id": req_id, "endpoint": "/v1/stage1/interpret", "question_id": body.question_id,
            "question": body.question, "status": "ok" if res.handoff is not None else "unresolved",
            "elapsed": round(time.time() - t0, 3), "stage1": res.source, "stage1_meta": _jsonable(res.meta)})
    return S.InterpretResponse(request_id=req_id, source=res.source,
                               handoff_status=res.handoff.status if res.handoff is not None else None,
                               handoff=_handoff_dump(res), session=_session(res), clarification=_clarification(res),
                               meta=_jsonable(res.meta))


@router.post("/stage1/clarify", response_model=S.ClarifyResponse, tags=["stage1"],
             responses={**_ERR, 404: {"model": S.ApiError}, 409: {"model": S.ApiError}},
             summary="역질문 세션 재개 — HCX 미호출, revision·후보 밖 값은 fail-closed")
def clarify(body: S.ClarifyRequest, request: Request) -> S.ClarifyResponse:
    rt = _rt(request); req_id = str(uuid4()); t0 = time.time()
    answer_wire = {"session_id": body.session_id, "clarification_id": body.clarification_id,
                   "expected_revision": body.expected_revision, "action": body.action, "values": body.values}
    res, result = _call(rt.clarify, answer_wire, execute=body.run_stage2, question_id=body.question_id,
                        question=body.question, compose=body.compose)
    elapsed = time.time() - t0
    answer_out = None
    if result is not None:
        qid = body.question_id or result.response["question_id"]
        _log_answer(rt, endpoint="/v1/stage1/clarify", request_id=req_id, question_id=qid,
                    question=body.question, result=result, elapsed=elapsed)
        answer_out = _answer_response(result, request_id=req_id, total_s=elapsed, include_payload=body.include_payload)
    else:
        rt.log({"req_id": req_id, "endpoint": "/v1/stage1/clarify", "session_id": body.session_id,
                "status": "ok", "elapsed": round(elapsed, 3), "stage1_meta": _jsonable(res.meta)})
    return S.ClarifyResponse(request_id=req_id, session=_session(res),
                             handoff_status=res.handoff.status if res.handoff is not None else None,
                             handoff=_handoff_dump(res), clarification=_clarification(res), answer=answer_out,
                             meta=_jsonable(res.meta))


@router.get("/stage1/sessions/{session_id}", response_model=S.InterpretResponse, tags=["stage1"],
            responses={**_ERR, 404: {"model": S.ApiError}}, summary="역질문 세션의 현재 공개 상태")
def session(session_id: Annotated[str, PathParam(min_length=1)], request: Request) -> S.InterpretResponse:
    rt = _rt(request)
    res: Stage1Resolution = _call(rt.inspect_session, session_id)
    return S.InterpretResponse(request_id=str(uuid4()), source=res.source,
                               handoff_status=res.handoff.status if res.handoff is not None else None,
                               handoff=_handoff_dump(res), session=_session(res), clarification=_clarification(res),
                               meta=_jsonable(res.meta))


# ── stage2 ───────────────────────────────────────────────────────────────────

@router.post("/stage2/execute", response_model=S.AnswerResponse, tags=["stage2"], responses=_ERR,
             summary="QueryPlanHandoff v0.4 JSON → typed tool 조회·Evidence 검증·계산·문장화")
def execute(body: S.ExecuteRequest, request: Request) -> S.AnswerResponse:
    from app.orchestrator.adapter import load_handoff_json
    rt = _rt(request); req_id = str(uuid4()); t0 = time.time()
    handoff = _call(load_handoff_json, body.handoff)
    resp, payload, meta = _call(rt.execute, handoff, question_id=body.question_id, question=body.question,
                                compose=body.compose)
    elapsed = time.time() - t0
    result = AnswerResult(resp, "ok", 200, meta={"handoff_status": handoff.status, **meta}, payload=payload)
    _log_answer(rt, endpoint="/v1/stage2/execute", request_id=req_id, question_id=body.question_id,
                question=body.question, result=result, elapsed=elapsed)
    return _answer_response(result, request_id=req_id, total_s=elapsed, include_payload=body.include_payload)


# ── fixtures ─────────────────────────────────────────────────────────────────

def _fixture_summary(rt: ServerRuntime, rec) -> S.FixtureSummary:
    req = rt.requirements.get(rec.question_id)
    return S.FixtureSummary(question_id=rec.question_id, group=rec.group, semantic_id=rec.semantic_id,
                            question=rec.question, handoff_status=rec.handoff.status,
                            expected_action=req.expected_action if req else None,
                            expected_handoff_status=req.expected_handoff_status if req else None)


@router.get("/fixtures", response_model=S.FixtureList, tags=["fixtures"],
            summary="release 70문항 목록 (query plan v0.4 최종 fixture)")
def fixtures(request: Request,
             group: Annotated[str | None, Query(description="그룹 접두(A·I·O·R·S·U) 또는 전체 이름")] = None,
             status: Annotated[str | None, Query(description="handoff status 필터")] = None) -> S.FixtureList:
    rt = _rt(request)
    _call(rt.require_fixture_access)
    items = []
    for rec in rt.fixtures:
        if group and not ((rec.group or "").startswith(group)):
            continue
        if status and rec.handoff.status != status:
            continue
        items.append(_fixture_summary(rt, rec))
    return S.FixtureList(total=len(items), items=items)


@router.get("/fixtures/{question_id}", response_model=S.FixtureDetail, tags=["fixtures"],
            responses={404: {"model": S.ApiError}}, summary="문항의 handoff v0.4 + AnswerRequirement")
def fixture(question_id: str, request: Request) -> S.FixtureDetail:
    rt = _rt(request)
    rec = _call(rt.fixture, question_id)
    req = rt.requirements.get(question_id)
    return S.FixtureDetail(**_fixture_summary(rt, rec).model_dump(), handoff=rec.handoff.model_dump(mode="json"),
                           requirement=req.model_dump(mode="json") if req else None,
                           migration_notes=list(rec.migration_notes))


@router.post("/fixtures/{question_id}/run", response_model=S.FixtureRunResponse, tags=["fixtures"],
             responses={**_ERR, 404: {"model": S.ApiError}},
             summary="문항 handoff 를 Stage2~4 로 실행하고 AnswerRequirement 로 채점 (HCX-007 미호출)")
def run_fixture(question_id: str, request: Request,
                compose: S.ComposeMode = Query("template", description="template=결정적(기본) / auto=HCX-005"),
                include_payload: bool = Query(True)) -> S.FixtureRunResponse:
    rt = _rt(request); req_id = str(uuid4()); t0 = time.time()
    resp, payload, meta, card = _call(rt.run_fixture, question_id, compose=compose)
    elapsed = time.time() - t0
    result = AnswerResult(resp, "ok", 200, meta=meta, payload=payload)
    _log_answer(rt, endpoint="/v1/fixtures/run", request_id=req_id, question_id=question_id,
                question=resp.get("question", ""), result=result, elapsed=elapsed)
    score = None
    if card is not None:
        score = S.ScoreOut(question_id=card.question_id, passed=card.passed,
                           checks=[S.CheckOut(name=c.name, passed=c.passed, detail=c.detail, required=c.required)
                                   for c in card.checks])
    return S.FixtureRunResponse(answer=_answer_response(result, request_id=req_id, total_s=elapsed,
                                                        include_payload=include_payload), score=score)


# ── tools (read.py safe 경로) ─────────────────────────────────────────────────

def _fact_row(r) -> S.FactRowOut:
    return S.FactRowOut(doc_id=r.doc_id, rcept_dt=r.rcept_dt, scope=r.scope, statement=r.statement,
                        account_raw=r.account_raw, account_path=r.account_path, account_norm=r.account_norm,
                        acode=getattr(r, "acode", None), account_norm_source=getattr(r, "account_norm_source", None),
                        period_start=r.period_start, period_end=r.period_end, period_type=r.period_type,
                        cumulative=r.cumulative, value_text=r.money.text, value=r.money.value, unit=r.money.unit,
                        value_status=r.value_status, locator=r.locator, evidence_id=r.evidence_id,
                        evidence_status=r.evidence_status, citation=r.citation)


@router.get("/tools/companies", response_model=S.CompaniesResponse, tags=["tools"], responses=_ERR,
            summary="회사 표기 → corp_code 후보 (퍼지 없음, 여럿이면 역질문 대상)")
def companies(request: Request, name: Annotated[str, Query(min_length=1, examples=["현대차"])]) -> S.CompaniesResponse:
    rt = _rt(request)
    rows = _call(rt.tool, lambda: rt.rm.resolve_company(name))
    cands = [S.CompanyOut(corp_code=c.corp_code, corp_name=c.corp_name) for c in rows]
    return S.CompaniesResponse(name=name, resolved=len(cands) == 1, candidates=cands)


@router.get("/tools/lookup", response_model=S.LookupResponse, tags=["tools"], responses=_ERR,
            summary="재무 개념 단일 조회 — 못 고른 이유·다음 판별 축까지 (read.py lookup)")
def lookup(request: Request,
           corp_code: Annotated[str, Query(pattern=r"^\d{8}$", examples=["00126380"])],
           concept: Annotated[str, Query(min_length=1, examples=["revenue"])],
           period_end: Annotated[str, Query(pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2025-12-31"])],
           as_of: S.AsOf = DEFAULT_AS_OF,
           scope: Annotated[str, Query(pattern=r"^(CFS|SFS)$")] = "CFS",
           view: Annotated[str, Query(pattern=r"^(restated|as_filed)$")] = "restated",
           max_candidates: Annotated[int, Query(ge=0, le=50)] = 10) -> S.LookupResponse:
    rt = _rt(request)
    q = {"corp_code": corp_code, "concept": concept, "period_end": period_end, "as_of": as_of, "scope": scope, "view": view}
    lk = _call(rt.tool, lambda: rt.rm.lookup(corp_code, concept, period_end, as_of=as_of, scope=scope, view=view))
    return S.LookupResponse(query=q, status=lk.status, coverage_status=lk.coverage_status, confirmed=bool(lk),
                            next_discriminator=list(lk.next_discriminator), unread_documents=list(lk.unread_documents),
                            view=lk.view, selected=_fact_row(lk.selected) if lk.selected is not None else None,
                            candidates=[_fact_row(c) for c in lk.candidates[:max_candidates]])


@router.get("/tools/corrections", response_model=S.CorrectionsResponse, tags=["tools"], responses=_ERR,
            summary="정정 항목(선언된 전·후 값) — 기준시점까지 공개된 것만, raw 열 없음")
def corrections(request: Request, as_of: S.AsOf = DEFAULT_AS_OF,
                corp_code: str | None = Query(None, pattern=r"^\d{8}$"), corp_name: str | None = None,
                rcept_no: str | None = Query(None, pattern=r"^\d{14}$"), doc_group: str | None = None,
                event_type: str | None = None, form: str | None = None, reason_contains: str | None = None,
                limit: Annotated[int, Query(ge=1, le=500)] = 100) -> S.CorrectionsResponse:
    rt = _rt(request)
    if not any([corp_code, corp_name, rcept_no]):
        raise ApiException(422, "invalid_argument", "corp_code·corp_name·rcept_no 중 하나는 지정해야 합니다")
    q = {k: v for k, v in dict(as_of=as_of, corp_code=corp_code, corp_name=corp_name, rcept_no=rcept_no,
                                doc_group=doc_group, event_type=event_type, form=form,
                                reason_contains=reason_contains).items() if v is not None}

    def _run():
        out, truncated = [], False
        for i, r in enumerate(rt.rm.correction_items(**q)):
            if i >= limit:
                truncated = True
                break
            out.append(r)
        return out, truncated

    rows, truncated = _call(rt.tool, _run)
    items = [S.CorrectionItemOut(
        doc_id=r.doc_id, rcept_no=r.rcept_no, rcept_dt=r.rcept_dt, corp_code=r.corp_code, corp_name=r.corp_name,
        doc_group=r.doc_group, event_type=r.event_type, form=r.form, report_nm=r.report_nm, path=r.path,
        order=r.order, reason=r.reason, value_before=r.value_before, value_after=r.value_after,
        diff_kind=r.diff_kind, before_kind=r.before_kind, after_kind=r.after_kind,
        required_by_authority=r.required_by_authority, evidence_status=r.evidence_status,
        citations=list(r.citations)) for r in rows]
    return S.CorrectionsResponse(query=q, total=len(items), truncated=truncated, items=items)


@router.get("/tools/events", response_model=S.EventTimelineOut | None, tags=["tools"], responses=_ERR,
            summary="사건 timeline — 기준시점까지의 관측과 그 시점의 상태 (미래 해지 누설 없음)")
def events(request: Request, as_of: S.AsOf = DEFAULT_AS_OF,
           rcept_no: str | None = Query(None, pattern=r"^\d{14}$"), event_key: str | None = None):
    rt = _rt(request)
    if (rcept_no is None) == (event_key is None):
        raise ApiException(422, "invalid_argument", "rcept_no 와 event_key 중 정확히 하나를 지정해야 합니다")
    t = _call(rt.tool, lambda: rt.rm.event_timeline(as_of=as_of, rcept_no=rcept_no, event_key=event_key))
    if t is None:
        raise ApiException(404, "not_found", "기준시점까지 관측된 사건이 없습니다")
    return S.EventTimelineOut(
        event_key=t.event_key, kind=t.kind, corp_code=t.corp_code, corp_name=t.corp_name, doc_group=t.doc_group,
        form=t.form, root_rcept_no=t.root_rcept_no, as_of=t.as_of,
        state=S.EventStateOut(status=t.state.status, n_observations=t.state.n_observations,
                              last_rcept_no=t.state.last_rcept_no, last_observed_at=t.state.last_observed_at),
        observations=[S.EventObservationOut(
            seq=o.seq, rcept_no=o.rcept_no, observed_at=o.observed_at, is_correction=o.is_correction,
            is_termination=o.is_termination, previous_observation_rcept_no=o.previous_observation_rcept_no,
            support_status=o.support_status, citations=list(o.citations)) for o in t.observations],
        support_status=t.support_status, support_limitation=t.support_limitation,
        observation_support_status=t.observation_support_status,
        observation_support_limitation=t.observation_support_limitation,
        identity_verification_status=t.identity_verification_status, identity_limitation=t.identity_limitation,
        query_coverage_status=t.query_coverage_status, query_coverage_limitation=t.query_coverage_limitation,
        citations=list(t.citations))


@router.get("/tools/document-version", response_model=S.DocumentVersionOut, tags=["tools"], responses=_ERR,
            summary="기준시점의 최신 정정본 — 갈라짐(ambiguous)·cycle(invalid)은 선택값 없이 드러낸다")
def document_version(request: Request, rcept_no: Annotated[str, Query(pattern=r"^\d{14}$")],
                     as_of: S.AsOf = DEFAULT_AS_OF) -> S.DocumentVersionOut:
    rt = _rt(request)
    r = _call(rt.tool, lambda: rt.rm.resolve_document_version(rcept_no, as_of=as_of))
    return S.DocumentVersionOut(rcept_no=rcept_no, as_of=as_of, status=r.status, selected=r.selected,
                                leaves=list(r.leaves), members=list(r.members), reason=r.reason)


@router.get("/tools/search", response_model=S.SearchResponse, tags=["tools"], responses=_ERR,
            summary="서술형 chunk 검색 (FTS5·kiwi) — prompt-safe 본문만")
def search(request: Request, q: Annotated[str, Query(min_length=1, examples=["투자 계획"])],
           as_of: S.AsOf = DEFAULT_AS_OF,
           corp_code: Annotated[list[str] | None, Query()] = None,
           doc_group: Annotated[list[str] | None, Query()] = None,
           date_from: str | None = Query(None, pattern=r"^\d{8}$"),
           top_k: Annotated[int, Query(ge=1, le=50)] = 10) -> S.SearchResponse:
    rt = _rt(request)
    hits = _call(rt.tool, lambda: rt.search_index.search(
        q, as_of=as_of, corp_codes=tuple(corp_code or ()), doc_groups=tuple(doc_group or ()),
        date_from=date_from, top_k=top_k))
    out = [S.SearchHitOut(rank=h.rank, score=h.score, chunk_id=h.chunk_id, section_id=h.section_id, doc_id=h.doc_id,
                          corp_code=h.corp_code, corp_name=h.corp_name, doc_group=h.doc_group, rcept_dt=h.rcept_dt,
                          path=h.path, locator=h.locator, text_prompt_safe=h.text_prompt_safe,
                          evidence_id=h.evidence_id) for h in hits]
    return S.SearchResponse(query={"q": q, "as_of": as_of, "corp_code": corp_code or [], "doc_group": doc_group or [],
                                   "date_from": date_from, "top_k": top_k}, total=len(out), hits=out)


# ── eval · requests ──────────────────────────────────────────────────────────

def _score_file(path: Path) -> S.ScoreFileOut | None:
    if not path.exists():
        return None
    return S.ScoreFileOut(path=str(path.relative_to(ROOT)),
                          updated_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                          report=json.loads(path.read_text(encoding="utf-8")))


@router.get("/eval/score", response_model=S.EvalScoreResponse, tags=["eval"], responses={404: {"model": S.ApiError}},
            summary="최근 채점표 — make harness-json(out/score.json) · harness-compose(out/score_compose.json)")
def eval_score() -> S.EvalScoreResponse:
    out = S.EvalScoreResponse(harness=_score_file(ROOT / "out" / "score.json"),
                              harness_compose=_score_file(ROOT / "out" / "score_compose.json"))
    if out.harness is None and out.harness_compose is None:
        raise ApiException(404, "not_found", "채점표가 없습니다 — make harness-json 을 먼저 실행하세요")
    return out


@router.get("/requests", response_model=S.RequestLogResponse, tags=["requests"],
            summary="요청별 JSONL 로그 tail (out/requests/answers_YYYYMMDD.jsonl)")
def requests_log(request: Request, limit: Annotated[int, Query(ge=1, le=500)] = 50,
                 date: str | None = Query(None, pattern=r"^\d{8}$")) -> S.RequestLogResponse:
    rt = _rt(request)
    day = date or datetime.now(timezone.utc).strftime("%Y%m%d")
    items = rt.recent_requests(limit=limit, date=day)
    return S.RequestLogResponse(date=day, total=len(items), items=items)
