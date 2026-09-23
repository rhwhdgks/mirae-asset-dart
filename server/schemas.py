"""`/v1` 요청·응답 스키마 (pydantic v2). OpenAPI 문서(`/docs`)가 이 모델에서 생성된다.

원칙
- 공식 5필드(`OfficialAnswer`)는 그대로 두고, 진단 정보는 `meta`·`payload` 로 **덧붙인다**.
- 정본 row 는 read.py safe 필드만 옮긴다 — `*_raw`·`restricted_raw_included` 는 스키마에 없다.
- 역질문 세션은 공개 식별자(session_id·revision·clarification_id)와 slot prompt·선택지 label 만 노출한다.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Stage1Mode = Literal["auto", "native", "fixture"]
ComposeMode = Literal["auto", "template"]
QuestionId = Annotated[str, Field(min_length=1, max_length=128, examples=["Q-001"])]
QuestionText = Annotated[str, Field(min_length=1, max_length=2000,
                                    examples=["삼성전자의 2025년 연결기준 매출액은 얼마인가?"])]
AsOf = Annotated[str, Field(pattern=r"^\d{8}$", description="기준시점 YYYYMMDD (이후 접수 문서는 존재하지 않는 것으로 본다)",
                            examples=["20260619"])]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── 오류 ─────────────────────────────────────────────────────────────────────

class ApiErrorDetail(_Model):
    code: str = Field(examples=["session_not_found"])
    message: str
    detail: Any | None = None


class ApiError(_Model):
    error: ApiErrorDetail


# ── 답변 ─────────────────────────────────────────────────────────────────────

class OfficialAnswer(_Model):
    """주최 측 계약 5필드 — `GET /answer` 와 동일."""
    question_id: str
    question: str
    retrieved_context: str
    think_trace: str
    answer: str


class SessionRef(_Model):
    """열린 역질문 세션의 공개 식별자. `POST /v1/stage1/clarify` 에 그대로 되돌려 준다."""
    session_id: str
    revision: int = Field(ge=0)
    status: str = Field(description="needs_clarification | ready | partial_ready | terminal")
    clarification_id: str | None = None


class Stage1Summary(_Model):
    source: str = Field(description="stage1_v1_native | fixture | none")
    handoff_status: str | None = Field(default=None, description="ready | needs_clarification | out_of_scope | unsupported_request | policy_refusal")
    matched_fixture: str | None = Field(default=None, description="fixture 경로일 때 매칭된 release 문항 ID")
    session: SessionRef | None = None
    meta: dict[str, Any] = Field(default_factory=dict, description="HCX-007 request_id·토큰·지연·폴백 사유")


class AnswerMeta(_Model):
    request_id: str
    status: str = Field(description="ok | unresolved | timeout | error | not_ready | queue_timeout")
    final_status: str | None = Field(default=None, description="answer | partial_answer | clarify | refuse | not_found | failure | unresolved | timeout | error")
    stage1: Stage1Summary | None = None
    claims: int | None = None
    limitations: list[str] = Field(default_factory=list)
    used_documents: list[str] = Field(default_factory=list, description="근거로 실제 사용한 접수번호")
    compose: str | None = None
    timing: dict[str, float | None] = Field(default_factory=dict, description="stage1_s · pipeline_s · total_s")


class AnswerRequest(_Model):
    question_id: QuestionId
    question: QuestionText
    stage1: Stage1Mode = Field(
        default="auto",
        description=("auto=native 경로(오프라인 opt-in 서버에서만 fixture 폴백) / "
                     "native=HCX-007만 / fixture=오프라인 opt-in release 문항 매칭"),
    )
    compose: ComposeMode = Field(default="auto", description="auto=HCX-005(사후 검증 실패 시 템플릿) / template=결정적 템플릿만")
    include_payload: bool = Field(default=False, description="AnswerPayload(claims·citations·limitations·trace) 전체를 함께 반환")


class AnswerResponse(OfficialAnswer):
    meta: AnswerMeta
    payload: dict[str, Any] | None = None


# ── Stage1 ───────────────────────────────────────────────────────────────────

class ClarificationOptionOut(_Model):
    value: str
    label: str
    reason: str | None = None


class ClarificationSlotOut(_Model):
    slot_id: str
    role_hint: str = Field(description="entity | event | time | timepoint | qualifier | selection | value_kind")
    reason_code: str
    response_kind: str = Field(description="select_one=허용 후보 밖 값은 거부 / provide_value=자유 입력")
    prompt: str
    options: list[ClarificationOptionOut] = Field(default_factory=list)


class ClarificationOut(_Model):
    clarification_id: str
    plan_revision: int = Field(ge=0)
    message: str | None = Field(default=None, description="사용자에게 보여줄 역질문 문장(번호 목록)")
    slots: list[ClarificationSlotOut]


class InterpretRequest(_Model):
    question_id: QuestionId
    question: QuestionText
    stage1: Stage1Mode = "auto"


class InterpretResponse(_Model):
    request_id: str
    source: str
    handoff_status: str | None = None
    handoff: dict[str, Any] | None = Field(default=None, description="QueryPlanHandoff v0.4 (Stage2 입력 계약)")
    session: SessionRef | None = None
    clarification: ClarificationOut | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class ClarifyRequest(_Model):
    """`Stage1V1ClarificationAnswer` 와 같은 모양 + Stage2 실행 옵션."""
    session_id: str = Field(min_length=1)
    clarification_id: str = Field(min_length=1)
    expected_revision: int = Field(ge=0, description="응답받은 session.revision — 불일치는 409")
    action: Literal["submit", "none_of_above", "unknown"] = "submit"
    values: dict[str, str] = Field(default_factory=dict, description="slot_id → 값 (select_one 은 options.value 중 하나)")
    question_id: QuestionId | None = Field(default=None, description="Stage2 payload 의 question_id (없으면 session 기반)")
    question: str = Field(default="", max_length=2000, description="표시용 원문 질문")
    run_stage2: bool = Field(default=True, description="재개 결과가 ready 면 곧바로 Stage2~4 를 돌려 답변까지 반환")
    compose: ComposeMode = "auto"
    include_payload: bool = False


class ClarifyResponse(_Model):
    request_id: str
    session: SessionRef | None = None
    handoff_status: str | None = None
    handoff: dict[str, Any] | None = None
    clarification: ClarificationOut | None = Field(default=None, description="아직 남은 역질문 (다음 turn)")
    answer: AnswerResponse | None = Field(default=None, description="run_stage2=true 이고 handoff 가 있을 때")
    meta: dict[str, Any] = Field(default_factory=dict)


# ── Stage2 ───────────────────────────────────────────────────────────────────

class ExecuteRequest(_Model):
    question_id: QuestionId
    question: str = Field(default="", max_length=2000)
    handoff: dict[str, Any] = Field(description="QueryPlanHandoff v0.4 JSON (strict 검증)")
    compose: ComposeMode = "auto"
    include_payload: bool = True


# ── fixtures ─────────────────────────────────────────────────────────────────

class FixtureSummary(_Model):
    question_id: str
    group: str | None = None
    semantic_id: str | None = None
    question: str | None = None
    handoff_status: str
    expected_action: str | None = None
    expected_handoff_status: str | None = None


class FixtureList(_Model):
    total: int
    items: list[FixtureSummary]


class FixtureDetail(FixtureSummary):
    handoff: dict[str, Any]
    requirement: dict[str, Any] | None = None
    migration_notes: list[str] = Field(default_factory=list)


class CheckOut(_Model):
    name: str
    passed: bool
    detail: str = ""
    required: bool = True


class ScoreOut(_Model):
    question_id: str
    passed: bool
    checks: list[CheckOut]


class FixtureRunResponse(_Model):
    answer: AnswerResponse
    score: ScoreOut | None = Field(default=None, description="AnswerRequirement 채점 (payload + 문장)")


# ── tools ────────────────────────────────────────────────────────────────────

class CompanyOut(_Model):
    corp_code: str
    corp_name: str


class CompaniesResponse(_Model):
    name: str
    resolved: bool = Field(description="후보가 정확히 하나일 때만 true — 여럿이면 역질문 대상")
    candidates: list[CompanyOut]


class FactRowOut(_Model):
    doc_id: str
    rcept_dt: str
    scope: str
    statement: str
    account_raw: str
    account_path: str
    account_norm: str | None = None
    acode: str | None = Field(default=None, description="값 셀의 XBRL 택소노미 요소 ID (있을 때만)")
    account_norm_source: str | None = Field(default=None, description="account_norm 출처 — acode | label")
    period_start: str | None = None
    period_end: str | None = None
    period_type: str
    cumulative: bool | None = None
    value_text: str | None = Field(default=None, description="원문 문자열 — 권위 있는 값")
    value: float | None = None
    unit: str | None = None
    value_status: str
    locator: str
    evidence_id: str | None = None
    evidence_status: str
    citation: str | None = Field(default=None, description="검증된 Evidence 일 때만 evidence_id")


class LookupResponse(_Model):
    query: dict[str, Any]
    status: str = Field(description="ok | not_found | ambiguous_*")
    coverage_status: str = Field(description="complete | partial_unread | evidence_unavailable")
    confirmed: bool = Field(description="status==ok 이고 coverage_status==complete — 이때만 확정 답변")
    next_discriminator: list[str] = Field(default_factory=list, description="모호할 때 다음에 지정해야 할 축 (역질문 근거)")
    unread_documents: list[str] = Field(default_factory=list)
    view: str
    selected: FactRowOut | None = None
    candidates: list[FactRowOut] = Field(default_factory=list)


class CorrectionItemOut(_Model):
    doc_id: str
    rcept_no: str
    rcept_dt: str
    corp_code: str
    corp_name: str
    doc_group: str
    event_type: str | None = None
    form: str
    report_nm: str
    path: str
    order: int
    reason: str | None = None
    value_before: str | None = None
    value_after: str | None = None
    diff_kind: str
    before_kind: str
    after_kind: str
    required_by_authority: bool
    evidence_status: str
    citations: list[str] = Field(default_factory=list)


class CorrectionsResponse(_Model):
    query: dict[str, Any]
    total: int
    truncated: bool
    items: list[CorrectionItemOut]


class EventStateOut(_Model):
    status: str = Field(description="active | terminated | not_disclosed")
    n_observations: int
    last_rcept_no: str | None = None
    last_observed_at: str | None = None


class EventObservationOut(_Model):
    seq: int
    rcept_no: str
    observed_at: str
    is_correction: bool
    is_termination: bool
    previous_observation_rcept_no: str | None = None
    support_status: str
    citations: list[str] = Field(default_factory=list)


class EventTimelineOut(_Model):
    event_key: str
    kind: str
    corp_code: str
    corp_name: str
    doc_group: str
    form: str
    root_rcept_no: str
    as_of: str
    state: EventStateOut
    observations: list[EventObservationOut]
    support_status: str
    support_limitation: str | None = None
    observation_support_status: str
    observation_support_limitation: str | None = None
    identity_verification_status: str
    identity_limitation: str | None = None
    query_coverage_status: str
    query_coverage_limitation: str | None = None
    citations: list[str] = Field(default_factory=list)


class DocumentVersionOut(_Model):
    rcept_no: str
    as_of: str
    status: str = Field(description="ok | ambiguous | invalid | not_found")
    selected: str | None = Field(default=None, description="기준시점의 최신 정정본 접수번호 (ambiguous/invalid 면 None)")
    leaves: list[str] = Field(default_factory=list)
    members: list[str] = Field(default_factory=list)
    reason: str | None = None


class SearchHitOut(_Model):
    rank: int
    score: float
    chunk_id: str
    section_id: str
    doc_id: str
    corp_code: str
    corp_name: str
    doc_group: str
    rcept_dt: str
    path: str
    locator: str
    text_prompt_safe: str
    evidence_id: str | None = None


class SearchResponse(_Model):
    query: dict[str, Any]
    total: int
    hits: list[SearchHitOut]


# ── 진행 현황·평가·로그 ───────────────────────────────────────────────────────

class StatusResponse(_Model):
    service: dict[str, Any]
    canonical: dict[str, Any]
    search_index: dict[str, Any]
    stage1: dict[str, Any]
    composer: dict[str, Any]
    warmup: dict[str, Any]
    evaluation: dict[str, Any]
    requests: dict[str, Any]
    clarification_sessions: dict[str, Any]


class ScoreFileOut(_Model):
    path: str
    updated_at: str
    report: dict[str, Any]


class EvalScoreResponse(_Model):
    harness: ScoreFileOut | None = None
    harness_compose: ScoreFileOut | None = None


class RequestLogResponse(_Model):
    date: str
    total: int
    items: list[dict[str, Any]]
