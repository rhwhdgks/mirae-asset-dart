"""자연어 Planner의 비공개 Draft 경계.

LLM이나 로컬 parser는 이 파일의 Draft만 만들 수 있다. ``corp_code``, ``evidence_id``,
기준일과 corpus cutoff 같은 권위 값은 Draft에 존재하지 않으며, 실행 가능한
``ResolvedQueryPlan``은 :mod:`agent.planning`의 결정론적 resolver만 생성한다.
"""

from __future__ import annotations

from datetime import date
import json
import re
from typing import Any, Literal, Protocol

from pydantic import Field, ValidationError, field_validator

from .contracts import ContractModel


class DraftFinancialTask(ContractModel):
    kind: Literal["financial"] = "financial"
    company_text: str | None = None
    metric_text: str | None = None
    period_expression: str | None = Field(
        default=None, min_length=1, max_length=100)
    year: int | None = Field(default=None, ge=1900, le=9999)
    scope: str | None = None
    as_of: str | None = None
    view: Literal["as_filed", "restated"] = "restated"


class DraftNarrativeTask(ContractModel):
    kind: Literal["narrative"] = "narrative"
    operation: Literal["search", "summarize", "compare"] = "search"
    company_text: str | None = None
    retrieval_query: str | None = None
    as_of: str | None = None
    doc_group: str | None = None
    target_period_expressions: list[str] = Field(
        default_factory=list, max_length=8)
    selected_document_receipt: str | None = Field(
        default=None, pattern=r"^[0-9]{14}$")
    requested_slots: list[str] | None = Field(default=None, max_length=32)

    @field_validator("target_period_expressions")
    @classmethod
    def unique_target_periods(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not row.strip() for row in value):
            raise ValueError("narrative target period는 비공백·고유해야 합니다")
        return value

    @field_validator("requested_slots")
    @classmethod
    def unique_requested_slots(
            cls, value: list[str] | None,
            ) -> list[str] | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("narrative requested slot은 중복될 수 없습니다")
        return value


class DraftDocumentTask(ContractModel):
    """PlanProposal 밖 sidecar까지 결합한 내부 document resolver 입력."""

    kind: Literal["document"] = "document"
    operation: Literal["find", "latest", "version_history"]
    company_text: str | None = None
    as_of: str | None = None
    doc_group: str | None = None
    event_type_text: str | None = None
    counterparty_text: str | None = None
    contract_name_text: str | None = None
    seed_receipt_text: str | None = None
    target_period_expressions: list[str] = Field(
        default_factory=list, max_length=8)
    selected_document_receipt: str | None = Field(
        default=None, pattern=r"^[0-9]{14}$")

    @field_validator("target_period_expressions")
    @classmethod
    def unique_target_periods(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not row.strip() for row in value):
            raise ValueError("document target period는 비공백·고유해야 합니다")
        return value


class DraftDisclosureTask(ContractModel):
    """Disclosure proposal을 권위 selector로 내리기 위한 내부 입력."""

    kind: Literal["disclosure"] = "disclosure"
    operation: Literal["lookup", "list"]
    company_text: str | None = None
    as_of: str | None = None
    doc_group: str | None = None
    event_type_text: str | None = None
    counterparty_text: str | None = None
    contract_name_text: str | None = None
    seed_receipt_text: str | None = None
    target_period_expressions: list[str] = Field(
        default_factory=list, max_length=8)
    requested_slots: list[str] = Field(default_factory=list, max_length=32)
    result_labels: list[str] = Field(default_factory=list, max_length=32)

    @field_validator(
        "target_period_expressions", "requested_slots", "result_labels")
    @classmethod
    def unique_nonblank_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not row.strip() for row in value):
            raise ValueError("disclosure 배열 값은 비공백·고유해야 합니다")
        return value


class DraftEventTask(ContractModel):
    """Event proposal과 timepoint/selector를 결합하는 내부 입력."""

    kind: Literal["event"] = "event"
    operation: Literal["status", "timeline", "list"]
    company_text: str | None = None
    as_of_expression: str | None = None
    event_type_text: str | None = None
    counterparty_text: str | None = None
    contract_name_text: str | None = None
    seed_receipt_text: str | None = None
    target_period_expressions: list[str] = Field(
        default_factory=list, max_length=8)
    requested_slots: list[str] = Field(default_factory=list, max_length=32)
    result_labels: list[str] = Field(default_factory=list, max_length=32)

    @field_validator(
        "target_period_expressions", "requested_slots", "result_labels")
    @classmethod
    def unique_event_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not row.strip() for row in value):
            raise ValueError("event 배열 값은 비공백·고유해야 합니다")
        return value


class DraftCorrectionTask(ContractModel):
    """Correction proposal과 문서/사건 selector를 결합하는 내부 입력."""

    kind: Literal["correction"] = "correction"
    operation: Literal["diff", "history"]
    company_text: str | None = None
    as_of: str | None = None
    doc_group: str | None = None
    event_type_text: str | None = None
    counterparty_text: str | None = None
    contract_name_text: str | None = None
    seed_receipt_text: str | None = None
    target_period_expressions: list[str] = Field(
        default_factory=list, max_length=8)
    requested_slots: list[str] = Field(default_factory=list, max_length=32)
    result_labels: list[str] = Field(default_factory=list, max_length=32)

    @field_validator(
        "target_period_expressions", "requested_slots", "result_labels")
    @classmethod
    def unique_correction_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not row.strip() for row in value):
            raise ValueError("correction 배열 값은 비공백·고유해야 합니다")
        return value


DraftTask = (
    DraftFinancialTask | DraftNarrativeTask | DraftDocumentTask
    | DraftDisclosureTask | DraftEventTask | DraftCorrectionTask
)


class DraftQueryPlan(ContractModel):
    """현재 수직 슬라이스의 내부 Draft.

    첫 단계는 한 사용자 발화당 task 하나만 받는다. 복합 질문을 조용히 분할하지 않고,
    다중 task 지원은 별도 평가셋과 함께 계약 버전을 올려 추가한다.
    """

    schema_version: Literal["draft/0.1"] = "draft/0.1"
    tasks: list[DraftTask] = Field(min_length=1, max_length=1)


class DraftBackend(Protocol):
    """LLM provider 또는 로컬 parser가 구현하는 최소 경계."""

    def create_draft(
            self, question: str, *, reference_date: date,
            corpus_cutoff: str) -> DraftQueryPlan | dict[str, Any] | str: ...


class PlannerOutputError(ValueError):
    """Provider 출력이 비공개 Draft schema를 통과하지 못했을 때 발생한다."""


class PlannerAdapter:
    """Provider 출력을 strict Draft로 바꾸는 유일한 입구."""

    def __init__(self, backend: DraftBackend) -> None:
        self.backend = backend

    def create_draft(
            self, question: str, *, reference_date: date,
            corpus_cutoff: str) -> DraftQueryPlan:
        if not isinstance(question, str) or not question.strip():
            raise PlannerOutputError("질문은 비어 있지 않은 문자열이어야 합니다")
        try:
            payload = self.backend.create_draft(
                question.strip(), reference_date=reference_date,
                corpus_cutoff=corpus_cutoff)
            if isinstance(payload, DraftQueryPlan):
                return payload
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                raise TypeError("Draft provider는 object 또는 JSON object를 반환해야 합니다")
            return DraftQueryPlan.model_validate(payload)
        except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as exc:
            if isinstance(exc, PlannerOutputError):
                raise
            raise PlannerOutputError(f"Planner Draft schema 오류: {exc}") from exc


_KNOWN_COMPANIES = (
    "LG에너지솔루션", "SK하이닉스", "삼성전자", "LG엔솔", "LGES", "삼전", "하닉",
)
_FINANCIAL_METRICS = (
    "operating income", "operating profit", "total assets", "net income",
    "operating_income", "total_assets", "net_income", "capex_ppe",
    "유형자산 취득", "유형자산취득", "영업이익", "당기순이익", "자산총계",
    "매출액", "총자산", "순이익", "revenue", "sales", "capex", "매출",
)
_REQUEST_WORDS = re.compile(
    r"(?:관련|공시|내용|자료|정보|좀|을|를|은|는|이|가)?\s*"
    r"(?:찾아\s*줘|검색해\s*줘|알려\s*줘|보여\s*줘|찾아|검색해|알려줘|보여줘)[?.!\s]*$"
)


def _first_in_text(question: str, values: tuple[str, ...]) -> str | None:
    folded = question.casefold()
    for value in values:
        if value.casefold() in folded:
            return value
    return None


def _company_text(question: str) -> str | None:
    known = _first_in_text(question, _KNOWN_COMPANIES)
    if known is not None:
        return known
    # 로컬 fallback은 회사처럼 보이는 첫 토큰만 Draft에 옮긴다. 이것을 회사로
    # 확정하는 권한은 없으며, resolver가 canonical company catalog로 재검증한다.
    for token in re.findall(r"[A-Za-z가-힣][A-Za-z가-힣0-9&.()㈜·ㆍ_-]*", question):
        stripped = re.sub(r"(?:에서|에게|의|은|는|이|가|을|를)$", "", token)
        if (stripped and stripped not in {"연결", "별도", "최근", "올해", "작년", "재작년"}
                and not any(metric.casefold() in stripped.casefold()
                            for metric in _FINANCIAL_METRICS)):
            return stripped
    return None


def _full_date(question: str) -> str | None:
    match = re.search(
        r"(?<!\d)(20\d{2})\s*(?:년|[./-])\s*(\d{1,2})\s*"
        r"(?:월|[./-])\s*(\d{1,2})\s*일?\s*(?:기준|까지)", question)
    if match is None:
        return None
    year, month, day = (int(value) for value in match.groups())
    try:
        parsed = date(year, month, day)
    except ValueError:
        # 잘못된 날짜도 provider가 숨기지 않고 resolver 검증으로 전달한다.
        return f"{year:04d}{month:02d}{day:02d}"
    return parsed.strftime("%Y%m%d")


def _financial_year(question: str, reference_date: date) -> int | None:
    if "재작년" in question:
        return reference_date.year - 2
    if "작년" in question:
        return reference_date.year - 1
    if "올해" in question:
        return reference_date.year
    match = re.search(r"(?<!\d)(20\d{2})\s*(?:년|회계연도)?", question)
    return int(match.group(1)) if match else None


def _retrieval_query(question: str, company_text: str | None) -> str:
    value = question
    if company_text:
        value = re.sub(re.escape(company_text), " ", value, flags=re.IGNORECASE)
    value = re.sub(
        r"(?<!\d)20\d{2}\s*(?:년)?\s*(?:\d{1,2}\s*월\s*\d{1,2}\s*일)?\s*"
        r"(?:기준|까지)?", " ", value)
    value = re.sub(r"\b(?:최근|현재|기준)\b", " ", value)
    value = _REQUEST_WORDS.sub("", value)
    value = re.sub(r"\s+", " ", value).strip(" ?.!을를은는이가")
    return value


class KoreanRuleDraftBackend:
    """네트워크 없이 Planner 계약을 시연하는 좁은 한국어 fallback.

    이 parser는 LLM 대체품이 아니다. provider를 붙이기 전에도 동일한 validation,
    clarification, Tool 경로를 실행하기 위한 결정론적 개발 backend다.
    """

    def create_draft(
            self, question: str, *, reference_date: date,
            corpus_cutoff: str) -> DraftQueryPlan:
        del corpus_cutoff  # 권위 cutoff는 resolver가 주입한다.
        company = _company_text(question)
        metric = _first_in_text(question, _FINANCIAL_METRICS)
        as_of = _full_date(question)
        if metric is not None:
            scope = "CFS" if "연결" in question else (
                "SFS" if "별도" in question else None)
            view: Literal["as_filed", "restated"] = (
                "as_filed" if any(word in question for word in ("최초 제출", "당시 제출", "최초값"))
                else "restated")
            return DraftQueryPlan(tasks=[DraftFinancialTask(
                company_text=company, metric_text=metric,
                year=_financial_year(question, reference_date), scope=scope,
                as_of=as_of, view=view,
            )])
        return DraftQueryPlan(tasks=[DraftNarrativeTask(
            company_text=company,
            retrieval_query=_retrieval_query(question, company),
            as_of=as_of,
        )])


__all__ = [
    "DraftBackend", "DraftCorrectionTask", "DraftDisclosureTask",
    "DraftDocumentTask", "DraftEventTask",
    "DraftFinancialTask", "DraftNarrativeTask",
    "DraftQueryPlan", "KoreanRuleDraftBackend", "PlannerAdapter",
    "PlannerOutputError",
]
