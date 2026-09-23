"""QueryPlan과 ToolResult의 실행 계약.

LLM이 만든 자유 형식 초안은 이 모듈의 ``ResolvedQueryPlan``이 아니다.
Python validator를 통과한 plan만 Tool에 전달하고, 검증된 Evidence가 없는
사실 결과는 success가 될 수 없다.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
import re
from typing import (
    Annotated, Any, Generic, Iterable, Literal, Mapping, TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.canonical.security import PROMPT_DATA_BEGIN, PROMPT_DATA_END


AsOf = Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
HexId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
CorpCode = Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
NonEmptyString = Annotated[str, StringConstraints(min_length=1)]

# Public plan note for a recent-quarter series that crosses Q4.  It is both a
# transparent calculation policy for downstream consumers and the marker that
# distinguishes an internal FY operand from a separately requested annual
# answer when deciding whether period-length mismatch applies.
RECENT_Q4_FANOUT_APPLIED_DEFAULT = (
    "최근 분기 추이의 4분기 단독값은 연간 누계에서 9개월 누계를 차감합니다.")


def validate_as_of(value: str, *, field_name: str = "as_of") -> str:
    """ASCII YYYYMMDD와 실제 달력 날짜를 함께 검증한다."""
    if not isinstance(value, str) or len(value) != 8 or not value.isascii() or not value.isdigit():
        raise ValueError(f"{field_name}는 ASCII YYYYMMDD 8자리여야 합니다")
    try:
        date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    except ValueError as exc:
        raise ValueError(f"{field_name}는 실제로 존재하는 날짜여야 합니다") from exc
    return value


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FinancialConcept(StrEnum):
    """agent 가 조회할 수 있는 개념. **정본 사전이 아는 것의 부분집합이다.**

    여기에 이름을 올리면 ``agent/planning.py`` 의 ``_canonical_account_aliases``
    가 ``src/ingest/account_map.tsv`` 에서 그 개념의 표기를 전부 끌어온다 —
    표면형을 손으로 적지 않는다. 개념마다 회계 축을 ``_CONCEPT_AXES`` 에
    **함께** 올려야 한다. 축이 없으면 기본값(duration·PL)이 붙어 조용히 틀린다.

    ``non_controlling_interests``·``owners_of_parent`` 은 **자본 잔액만**
    가리킨다 (BS·instant). 예전에는 한 account_norm 에 세 값이 섞여 있어
    등록하지 못했는데, XBRL 요소 ID 가 이미 셋을 가르고 있었다 — 적재의
    ``acode_map.tsv`` 에 손익 쪽 코드가 없어 표기 사전으로 흘러 합쳐진 것이다
    (이슈 #127). 이제 넷이 따로 있다.

    ====================================  ==========  ==================
    개념                                  표          뜻
    ====================================  ==========  ==================
    non_controlling_interests             BS instant  자본 잔액
    net_income_non_controlling            IS/CI       귀속 당기순이익
    comprehensive_income_non_controlling  CI          귀속 총포괄손익
    owners_of_parent                      BS instant  자본 잔액
    net_income_owners_of_parent           IS/CI       귀속 당기순이익
    comprehensive_income_owners_of_parent CI          귀속 총포괄손익
    ====================================  ==========  ==================
    """

    # 이미 쓰던 5종
    REVENUE = "revenue"
    OPERATING_INCOME = "operating_income"
    NET_INCOME = "net_income"
    TOTAL_ASSETS = "total_assets"
    CAPEX_PPE = "capex_ppe"

    # 재무상태표 잔액 17종
    AOCI = "aoci"
    CASH_AND_EQUIVALENTS = "cash_and_equivalents"
    CURRENT_ASSETS = "current_assets"
    CURRENT_LIABILITIES = "current_liabilities"
    INTANGIBLE_ASSETS = "intangible_assets"
    INVESTMENT_PROPERTY = "investment_property"
    NON_CURRENT_ASSETS = "non_current_assets"
    NON_CURRENT_LIABILITIES = "non_current_liabilities"
    PPE = "ppe"
    RETAINED_EARNINGS = "retained_earnings"
    RIGHT_OF_USE_ASSETS = "right_of_use_assets"
    SHARE_CAPITAL = "share_capital"
    SHARE_PREMIUM = "share_premium"
    TOTAL_EQUITY = "total_equity"
    TOTAL_LIABILITIES = "total_liabilities"
    TOTAL_LIABILITIES_AND_EQUITY = "total_liabilities_and_equity"
    TRADE_AND_OTHER_RECEIVABLES = "trade_and_other_receivables"

    # 현금흐름 10종
    CAPEX_INTANGIBLE = "capex_intangible"
    CF_FINANCING = "cf_financing"
    CF_INVESTING = "cf_investing"
    CF_OPERATING = "cf_operating"
    DISPOSAL_INTANGIBLE = "disposal_intangible"
    DISPOSAL_PPE = "disposal_ppe"
    DIVIDENDS_PAID = "dividends_paid"
    INTEREST_PAID = "interest_paid"
    INTEREST_RECEIVED = "interest_received"
    NET_CHANGE_IN_CASH = "net_change_in_cash"
    # 기초·기말 잔액. CF 표에 duration 좌표로 실리지만 시점은 개념 ID 가 갖는다.
    CASH_BEGINNING = "cash_beginning"
    CASH_ENDING = "cash_ending"

    # 연결 자본·손익의 귀속 구획 6종. 잔액(BS)과 귀속액(IS/CI)은 자릿수부터
    # 다르므로 개념을 갈라 둔다 — 하나로 두면 셋 중 둘을 물은 사람이 조용히
    # 틀린 값을 받는다 (이슈 #127).
    NON_CONTROLLING_INTERESTS = "non_controlling_interests"
    OWNERS_OF_PARENT = "owners_of_parent"
    NET_INCOME_NON_CONTROLLING = "net_income_non_controlling"
    NET_INCOME_OWNERS_OF_PARENT = "net_income_owners_of_parent"
    COMPREHENSIVE_INCOME_NON_CONTROLLING = "comprehensive_income_non_controlling"
    COMPREHENSIVE_INCOME_OWNERS_OF_PARENT = "comprehensive_income_owners_of_parent"

    # 재무상태표 잔액 · 원천이 CF 에도 나타나는 7종. CF 쪽은 재고·매출채권·
    # 충당부채의 **증감**(간접법 조정)이고 잔액과 다른 값이므로 statement=BS 로
    # 걸러낸다. 유동/비유동 충당부채는 정본 사전에 아예 없어 섞이지 않는다.
    CURRENT_TAX_ASSETS = "current_tax_assets"
    CURRENT_TAX_LIABILITIES = "current_tax_liabilities"
    DEFERRED_TAX_ASSETS = "deferred_tax_assets"
    DEFERRED_TAX_LIABILITIES = "deferred_tax_liabilities"
    INVENTORIES = "inventories"
    PROVISIONS = "provisions"
    TRADE_RECEIVABLES = "trade_receivables"

    # 손익 · 원천이 CF 에도 나타나는 8종. 마찬가지로 간접법 조정항목이다.
    # ``income_tax`` 는 법인세**비용**만 받는다 — 법인세 납부액은 정본 사전에 없다.
    # ``interest_income``/``interest_expense``(이자수익·이자비용, PL)와
    # ``interest_received``/``interest_paid``(이자의수취·이자의지급, CF)는 원천
    # 계정이 서로 다른 별개 개념이다.
    EQUITY_METHOD_INCOME = "equity_method_income"
    FINANCE_COSTS = "finance_costs"
    FINANCE_INCOME = "finance_income"
    INCOME_TAX = "income_tax"
    INTEREST_EXPENSE = "interest_expense"
    INTEREST_INCOME = "interest_income"
    OTHER_EXPENSES = "other_expenses"
    OTHER_INCOME = "other_income"

    # 손익 9종
    COST_OF_SALES = "cost_of_sales"
    DILUTED_EPS = "diluted_eps"
    EPS = "eps"
    GROSS_PROFIT = "gross_profit"
    OPERATING_EXPENSES = "operating_expenses"
    OTHER_COMPREHENSIVE_INCOME = "other_comprehensive_income"
    PRETAX_INCOME = "pretax_income"
    SGANDA = "sganda"
    TOTAL_COMPREHENSIVE_INCOME = "total_comprehensive_income"


class OutputRef(ContractModel):
    output_id: str = Field(min_length=1)
    field: Literal["value"] = "value"


class TaskVerificationRef(ContractModel):
    """값 하나가 아니라 task 실행 결과의 존재·상태를 검증하는 참조."""

    task_id: str = Field(min_length=1)


class Derivation(ContractModel):
    output_id: str = Field(min_length=1)
    operator: Literal[
        "difference", "absolute_difference", "percent_change",
        "discrete_from_cumulative", "argmax", "equal", "concept_ratio",
        "sum",
    ]
    operands: list[OutputRef] = Field(min_length=1)
    rounding_rule: Literal[
        "round_half_up_0", "round_half_up_1", "round_half_up_2", "truncate_0",
    ] | None = None
    #: ``concept_ratio`` 전용 — 「률/비율」은 ``percent``(×100, `%`), 「몇 배」는
    #: ``multiple``(그대로, `배`)로 읽는다. 다른 연산자는 이 축이 없다 — 차이·
    #: 증감률은 이미 스스로 단위(원·%)를 정한다.
    presentation: Literal["percent", "multiple"] | None = None
    # 이슈 #124 — argmax 연산자의 극값 방향. optional — 기존 plan은 그대로
    # "maximum"(최댓값)이고, "argmax"가 아닌 연산자는 이 축이 없다.
    direction: Literal["maximum", "minimum"] = "maximum"

    @model_validator(mode="after")
    def validate_arity(self) -> "Derivation":
        expected = {
            "difference": 2,
            "absolute_difference": 2,
            "percent_change": 2,
            "discrete_from_cumulative": 2,
            "equal": 2,
            "concept_ratio": 2,
        }
        if self.operator in expected and len(self.operands) != expected[self.operator]:
            raise ValueError(
                f"{self.operator} 연산자는 operand {expected[self.operator]}개가 필요합니다")
        if self.operator == "argmax" and len(self.operands) < 2:
            raise ValueError("argmax 연산자는 operand 2개 이상이 필요합니다")
        if self.operator == "sum" and len(self.operands) < 2:
            raise ValueError("sum 연산자는 operand 2개 이상이 필요합니다")
        if self.operator != "argmax" and self.direction != "maximum":
            raise ValueError("direction=minimum은 argmax 연산자에만 허용됩니다")
        if self.operator == "equal" and self.rounding_rule is not None:
            raise ValueError("equal 연산자는 rounding_rule을 가질 수 없습니다")
        if self.operator == "concept_ratio" and self.presentation is None:
            raise ValueError("concept_ratio 연산자는 presentation이 필요합니다")
        if self.operator != "concept_ratio" and self.presentation is not None:
            raise ValueError("concept_ratio가 아닌 연산자는 presentation을 가질 수 없습니다")
        return self


class PremiseClaim(ContractModel):
    claim_id: str = Field(min_length=1)
    kind: Literal["numeric", "state", "comparison", "existence", "causal"]
    raw_text: str = Field(min_length=1)
    value: str | None = None
    unit: str | None = None
    verify_with: list[OutputRef] = Field(default_factory=list)
    verify_tasks: list[TaskVerificationRef] = Field(
        default_factory=list, exclude_if=lambda refs: not refs)

    @model_validator(mode="after")
    def unique_verification_refs(self) -> "PremiseClaim":
        scalar_ids = [ref.output_id for ref in self.verify_with]
        task_ids = [ref.task_id for ref in self.verify_tasks]
        if len(scalar_ids) != len(set(scalar_ids)):
            raise ValueError("PremiseClaim scalar 검증 참조가 중복되었습니다")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("PremiseClaim task 검증 참조가 중복되었습니다")
        return self


class DateRange(ContractModel):
    start: date | None = None
    end: date

    @model_validator(mode="after")
    def valid_order(self) -> "DateRange":
        if self.start is not None and self.start > self.end:
            raise ValueError("DateRange.start는 end 이후일 수 없습니다")
        return self


class DocumentSelector(ContractModel):
    doc_id: str | None = None
    rcept_no: str | None = None
    doc_group: str | None = None
    event_type: str | None = None
    form: str | None = None
    report_name_contains: str | None = None
    rcept_from: AsOf | None = None
    rcept_to: AsOf | None = None
    is_correction: bool | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> "DocumentSelector":
        if self.rcept_from is not None:
            validate_as_of(self.rcept_from, field_name="rcept_from")
        if self.rcept_to is not None:
            validate_as_of(self.rcept_to, field_name="rcept_to")
        if self.rcept_from and self.rcept_to and self.rcept_from > self.rcept_to:
            raise ValueError("rcept_from은 rcept_to 이후일 수 없습니다")
        return self

    def has_identity_condition(self) -> bool:
        return any((
            self.doc_id, self.rcept_no, self.doc_group, self.event_type, self.form,
            self.report_name_contains, self.rcept_from, self.rcept_to,
            self.is_correction is not None,
        ))


class EventSelector(ContractModel):
    event_key: str | None = None
    seed_rcept_no: str | None = None
    event_type: str | None = None
    counterparty: str | None = None
    contract_name: str | None = None
    keywords: list[str] = Field(default_factory=list)
    event_from: AsOf | None = None
    event_to: AsOf | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> "EventSelector":
        if self.event_from is not None:
            validate_as_of(self.event_from, field_name="event_from")
        if self.event_to is not None:
            validate_as_of(self.event_to, field_name="event_to")
        if self.event_from and self.event_to and self.event_from > self.event_to:
            raise ValueError("event_from은 event_to 이후일 수 없습니다")
        return self

    def has_identity_condition(self) -> bool:
        return any((
            self.event_key, self.seed_rcept_no, self.event_type, self.counterparty,
            self.contract_name, self.keywords, self.event_from, self.event_to,
        ))


class FieldOutputSpec(ContractModel):
    """서식형 Tool 결과의 한 slot을 계산 가능한 output으로 연결한다."""

    output_id: str = Field(min_length=1)
    slot: str = Field(min_length=1)
    value_kind: Literal["money", "percent", "date", "count", "text"] = "money"


class FactSpec(ContractModel):
    output_id: str = Field(min_length=1)
    corp_code: CorpCode
    corp_name: str = Field(min_length=1)
    concept: FinancialConcept
    period_start: date | None
    period_end: date
    period_type: Literal["annual", "half", "quarter", "instant"]
    cumulative: bool | None
    scope: Literal["CFS", "SFS"]
    statement: Literal["BS", "IS", "CI", "CF"] | None = None
    account_path: str | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def validate_period(self) -> "FactSpec":
        if self.period_type == "instant":
            if self.period_start is not None:
                raise ValueError("instant Fact는 period_start가 없어야 합니다")
        elif self.period_start is None:
            raise ValueError("duration Fact는 period_start가 필요합니다")
        if self.period_start is not None and self.period_start > self.period_end:
            raise ValueError("period_start는 period_end 이후일 수 없습니다")
        return self


class ResolvedFinancialTask(ContractModel):
    task_id: str = Field(min_length=1)
    kind: Literal["financial"] = "financial"
    as_of: AsOf
    view: Literal["as_filed", "restated"]
    facts: list[FactSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_as_of(self) -> "ResolvedFinancialTask":
        validate_as_of(self.as_of)
        return self


class ResolvedDisclosureTask(ContractModel):
    task_id: str = Field(min_length=1)
    kind: Literal["disclosure"] = "disclosure"
    operation: Literal["lookup", "list"]
    corp_code: CorpCode
    corp_name: str = Field(min_length=1)
    as_of: AsOf
    document_selector: DocumentSelector | None = None
    event_selector: EventSelector | None = None
    requested_slots: list[str] = Field(min_length=1)
    output_id: NonEmptyString | None = None
    field_outputs: list[FieldOutputSpec] = Field(default_factory=list)
    # 이슈 #59 2단계 — list 로 모인 후보 중 이 slot 의 검증된 최댓값 하나로
    # 줄인다(예: "계약금액"). optional — 기존 list task 는 그대로 None.
    argmax_slot: NonEmptyString | None = None
    # 이슈 #124 — argmax_slot 이 줄이는 극값의 방향. optional — argmax_slot이
    # None이면 의미가 없으므로 "maximum" 그대로.
    argmax_direction: Literal["maximum", "minimum"] = "maximum"

    @model_validator(mode="after")
    def valid_selector(self) -> "ResolvedDisclosureTask":
        validate_as_of(self.as_of)
        if not (
                (self.document_selector is not None
                 and self.document_selector.has_identity_condition())
                or (self.event_selector is not None
                    and self.event_selector.has_identity_condition())):
            raise ValueError("DisclosureTask에는 selector가 필요합니다")
        if len(self.requested_slots) != len(set(self.requested_slots)):
            raise ValueError("Disclosure requested_slots가 중복되었습니다")
        if any(row.slot not in self.requested_slots for row in self.field_outputs):
            raise ValueError("Disclosure field_output slot은 requested_slots에 있어야 합니다")
        if len({row.slot for row in self.field_outputs}) != len(self.field_outputs):
            raise ValueError("Disclosure field_output slot이 중복되었습니다")
        if self.argmax_slot is not None and (
                self.operation != "list"
                or self.argmax_slot not in self.requested_slots):
            raise ValueError(
                "Disclosure argmax_slot은 list operation의 requested_slots에 있어야 합니다")
        if self.argmax_slot is None and self.argmax_direction != "maximum":
            raise ValueError(
                "Disclosure argmax_direction은 argmax_slot에만 허용됩니다")
        return self


class ResolvedEventTask(ContractModel):
    task_id: str = Field(min_length=1)
    kind: Literal["event"] = "event"
    operation: Literal["status", "timeline", "list"]
    corp_code: CorpCode
    corp_name: str = Field(min_length=1)
    selector: EventSelector
    timepoints: list[AsOf] = Field(min_length=1)
    requested_slots: list[str] = Field(default_factory=list)
    output_id: NonEmptyString | None = None
    field_outputs: list[FieldOutputSpec] = Field(default_factory=list)
    # 이슈 #59 2단계 — list 로 모인 후보 중 이 slot 의 검증된 최댓값 하나로
    # 줄인다(예: "계약금액"). optional — 기존 list task 는 그대로 None.
    argmax_slot: NonEmptyString | None = None
    # 이슈 #124 — argmax_slot 이 줄이는 극값의 방향. optional — argmax_slot이
    # None이면 의미가 없으므로 "maximum" 그대로.
    argmax_direction: Literal["maximum", "minimum"] = "maximum"

    @model_validator(mode="after")
    def valid_selector(self) -> "ResolvedEventTask":
        for value in self.timepoints:
            validate_as_of(value, field_name="timepoint")
        if not self.selector.has_identity_condition():
            raise ValueError("EventTask의 selector가 비어 있습니다")
        if self.operation in {"status", "timeline"} and not self.selector.event_key:
            raise ValueError("Event status/timeline에는 확정 event_key가 필요합니다")
        if len(self.timepoints) != len(set(self.timepoints)):
            raise ValueError("Event timepoint가 중복되었습니다")
        if len(self.requested_slots) != len(set(self.requested_slots)):
            raise ValueError("Event requested_slots가 중복되었습니다")
        if any(row.slot not in self.requested_slots for row in self.field_outputs):
            raise ValueError("Event field_output slot은 requested_slots에 있어야 합니다")
        if len({row.slot for row in self.field_outputs}) != len(self.field_outputs):
            raise ValueError("Event field_output slot이 중복되었습니다")
        if self.argmax_slot is not None and (
                self.operation != "list"
                or self.argmax_slot not in self.requested_slots):
            raise ValueError(
                "Event argmax_slot은 list operation의 requested_slots에 있어야 합니다")
        if self.argmax_slot is None and self.argmax_direction != "maximum":
            raise ValueError(
                "Event argmax_direction은 argmax_slot에만 허용됩니다")
        return self


class ResolvedCorrectionTask(ContractModel):
    task_id: str = Field(min_length=1)
    kind: Literal["correction"] = "correction"
    operation: Literal["diff", "history"]
    corp_code: CorpCode
    corp_name: str = Field(min_length=1)
    as_of: AsOf
    document_selector: DocumentSelector | None = None
    event_selector: EventSelector | None = None
    requested_slots: list[str] = Field(default_factory=list)
    output_id: NonEmptyString | None = None
    field_outputs: list[FieldOutputSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_selector(self) -> "ResolvedCorrectionTask":
        validate_as_of(self.as_of)
        if not (
                (self.document_selector is not None
                 and self.document_selector.has_identity_condition())
                or (self.event_selector is not None
                    and self.event_selector.has_identity_condition())):
            raise ValueError("CorrectionTask에는 selector가 필요합니다")
        if len(self.requested_slots) != len(set(self.requested_slots)):
            raise ValueError("Correction requested_slots가 중복되었습니다")
        if any(row.slot not in self.requested_slots for row in self.field_outputs):
            raise ValueError("Correction field_output slot은 requested_slots에 있어야 합니다")
        if len({row.slot for row in self.field_outputs}) != len(self.field_outputs):
            raise ValueError("Correction field_output slot이 중복되었습니다")
        return self


class ResolvedDocumentTask(ContractModel):
    task_id: str = Field(min_length=1)
    kind: Literal["document"] = "document"
    operation: Literal["find", "latest", "version_history"]
    corp_code: CorpCode
    corp_name: str = Field(min_length=1)
    as_of: AsOf
    selector: DocumentSelector
    event_selector: EventSelector | None = None

    @model_validator(mode="after")
    def valid_selector(self) -> "ResolvedDocumentTask":
        validate_as_of(self.as_of)
        if (not self.selector.has_identity_condition()
                and not (self.event_selector is not None
                         and self.event_selector.has_identity_condition())):
            raise ValueError("DocumentTask의 selector가 비어 있습니다")
        return self


class ResolvedNarrativeTask(ContractModel):
    task_id: str = Field(min_length=1)
    kind: Literal["narrative"] = "narrative"
    operation: Literal["search", "summarize", "compare"]
    corp_codes: list[CorpCode] = Field(min_length=1)
    corp_names: list[str] = Field(default_factory=list)
    as_of: AsOf
    retrieval_query: str = Field(min_length=1)
    document_selector: DocumentSelector | None = None
    periods: list[DateRange] = Field(default_factory=list)
    requested_slots: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_as_of(self) -> "ResolvedNarrativeTask":
        validate_as_of(self.as_of)
        if not self.retrieval_query.strip():
            raise ValueError("retrieval_query에는 내용어가 필요합니다")
        if self.corp_names and len(self.corp_names) != len(self.corp_codes):
            raise ValueError("Narrative corp_codes와 corp_names 수가 다릅니다")
        if len(self.corp_codes) != len(set(self.corp_codes)):
            raise ValueError("Narrative corp_codes가 중복되었습니다")
        return self


class PresentationSpec(ContractModel):
    """정답 의미를 바꾸지 않는 composer 출력 형식 힌트."""

    format: Literal["prose", "table", "list"]


ResolvedTask = Annotated[
    ResolvedFinancialTask
    | ResolvedDisclosureTask
    | ResolvedEventTask
    | ResolvedCorrectionTask
    | ResolvedDocumentTask
    | ResolvedNarrativeTask,
    Field(discriminator="kind"),
]


class ResolvedQueryPlan(ContractModel):
    schema_version: Literal["0.1"] = "0.1"
    revision: int = Field(ge=0)
    reference_date: date
    corpus_cutoff: AsOf
    tasks: list[ResolvedTask] = Field(min_length=1)
    derivations: list[Derivation] = Field(default_factory=list)
    premise_claims: list[PremiseClaim] = Field(default_factory=list)
    applied_defaults: list[str] = Field(default_factory=list)
    presentation: PresentationSpec | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "ResolvedQueryPlan":
        cutoff = validate_as_of(self.corpus_cutoff, field_name="corpus_cutoff")
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task_id는 plan 전체에서 유일해야 합니다")
        claim_ids = [claim.claim_id for claim in self.premise_claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("PremiseClaim claim_id는 plan 전체에서 유일해야 합니다")
        if len(self.applied_defaults) != len(set(self.applied_defaults)):
            raise ValueError("applied_defaults가 중복되었습니다")

        output_ids: list[str] = []
        for task in self.tasks:
            values: list[str]
            if isinstance(task, ResolvedFinancialTask):
                values = [fact.output_id for fact in task.facts]
                task_dates = [task.as_of]
                if any(
                        fact.period_end.strftime("%Y%m%d") > task.as_of
                        for fact in task.facts):
                    raise ValueError("Fact 기간은 Task as_of 이후일 수 없습니다")
            elif isinstance(task, ResolvedEventTask):
                values = ([task.output_id] if task.output_id else []) + [
                    row.output_id for row in task.field_outputs]
                task_dates = list(task.timepoints)
                for value in (task.selector.event_from, task.selector.event_to):
                    if value is not None:
                        task_dates.append(value)
            elif isinstance(task, (ResolvedDisclosureTask, ResolvedCorrectionTask)):
                values = ([task.output_id] if task.output_id else []) + [
                    row.output_id for row in task.field_outputs]
                task_dates = [task.as_of]
                for selector in (task.document_selector,):
                    if selector is not None:
                        for value in (selector.rcept_from, selector.rcept_to):
                            if value is not None:
                                task_dates.append(value)
                if task.event_selector is not None:
                    for value in (
                            task.event_selector.event_from,
                            task.event_selector.event_to):
                        if value is not None:
                            task_dates.append(value)
            else:
                values = []
                task_dates = [task.as_of]
                if isinstance(task, ResolvedDocumentTask):
                    for value in (task.selector.rcept_from, task.selector.rcept_to):
                        if value is not None:
                            task_dates.append(value)
                    if task.event_selector is not None:
                        for value in (
                                task.event_selector.event_from,
                                task.event_selector.event_to):
                            if value is not None:
                                task_dates.append(value)
                elif isinstance(task, ResolvedNarrativeTask):
                    if task.document_selector is not None:
                        for value in (
                                task.document_selector.rcept_from,
                                task.document_selector.rcept_to):
                            if value is not None:
                                task_dates.append(value)
                    if any(
                            period.end.strftime("%Y%m%d") > cutoff
                            for period in task.periods):
                        raise ValueError("Narrative 기간은 corpus_cutoff를 넘을 수 없습니다")
            if any(value > cutoff for value in task_dates):
                raise ValueError("Task 조회시점은 corpus_cutoff를 넘을 수 없습니다")
            output_ids.extend(values)

        known = set(output_ids)
        if len(output_ids) != len(known):
            raise ValueError("Fact output_id는 plan 전체에서 유일해야 합니다")
        for derivation in self.derivations:
            if derivation.output_id in known:
                raise ValueError("Fact/Derivation output_id는 plan 전체에서 유일해야 합니다")
            missing = [ref.output_id for ref in derivation.operands if ref.output_id not in known]
            if missing:
                raise ValueError(f"Derivation operand가 선행 output을 참조하지 않습니다: {missing}")
            known.add(derivation.output_id)
        for claim in self.premise_claims:
            missing = [ref.output_id for ref in claim.verify_with if ref.output_id not in known]
            if missing:
                raise ValueError(f"PremiseClaim 참조 output이 없습니다: {missing}")
            missing_tasks = [
                ref.task_id for ref in claim.verify_tasks
                if ref.task_id not in set(task_ids)
            ]
            if missing_tasks:
                raise ValueError(
                    f"PremiseClaim 참조 task가 없습니다: {missing_tasks}")

        namespaces = {
            "task_id": set(task_ids),
            "output_id": known,
            "claim_id": set(claim_ids),
        }
        for left_name, left in namespaces.items():
            for right_name, right in namespaces.items():
                if left_name >= right_name:
                    continue
                overlap = left & right
                if overlap:
                    raise ValueError(
                        f"{left_name}/{right_name} namespace가 겹칩니다: "
                        f"{sorted(overlap)}")
        return self


class ClarificationOption(ContractModel):
    value: Any
    label: str = Field(min_length=1)
    reason: str | None = None


def reason_token(name: str) -> str:
    """이름을 reason code 문자셋(``[a-z][a-z0-9_]{0,63}``)으로 좁힌다.

    역질문 사유는 field 이름·discriminator 처럼 **동적인 값**에서 만들어진다
    (`company_mention`·`account_path`·`facts[0]`). 형식이 어긋나면 계약
    validator 가 거절해 **역질문 대신 예외**가 나가는데, 그건 물어야 할 때
    터지는 것이라 최악이다. 그래서 만드는 쪽에서 맞춰 둔다.
    """

    token = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    if not token or not token[0].isalpha():
        token = f"field_{token}" if token else "field"
    return token[:56]


#: 역질문의 **상호작용 모양**. renderer 가 위젯을 고르고 평가기가 묶는 축이다.
ClarificationType = Literal[
    "SELECT_ONE", "PROVIDE_VALUE", "CONFIRM_INTERPRETATION",
]


class ClarificationRequest(ContractModel):
    """typed 역질문. **`question` 문자열을 파싱해 종류를 추측하지 않는다.**

    예전에는 `type`·`reason_code` 가 없어서 renderer 와 평가기가 둘 다 자연어
    질문을 보고 종류를 짐작해야 했다. 13건의 역질문을 「무엇을 왜 물었나」로
    묶을 방법이 없었다.

    `type` 은 요청 **전체**의 모양이다. 결정서 §3.4 는 `target_field_path` 단수를
    가정했지만 구현은 원자적 batch 역질문을 지원한다 — 회사와 scope 를 한 번에
    묻는 경우 회사는 후보가 없고(값을 받아야 한다) scope 는 후보가 있다. 그때
    묶는 제약은 「값을 받아야 한다」쪽이므로 `PROVIDE_VALUE` 다.

    아래 validator 가 `type` 과 `options` 의 정합성을 강제한다 — 틀린 `type` 은
    거절된다. 그래서 이 필드는 장식이 아니다.
    """

    request_id: str = Field(min_length=1)
    plan_revision: int = Field(ge=0)
    type: ClarificationType
    #: 왜 물었나. reason code 형식은 terminal reason 과 같다.
    reason_code: str = Field(min_length=1, max_length=64)
    field_paths: list[str] = Field(min_length=1)
    question: str = Field(min_length=1)
    options: dict[str, list[ClarificationOption]]
    patch_paths: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_typed_shape(self) -> "ClarificationRequest":
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.reason_code):
            raise ValueError(
                f"reason_code 형식이 잘못되었습니다: {self.reason_code!r}")
        empty = [path for path in self.field_paths if not self.options.get(path)]
        if self.type == "PROVIDE_VALUE":
            if not empty:
                raise ValueError(
                    "PROVIDE_VALUE 역질문은 후보가 없는 field가 있어야 합니다")
        elif empty:
            # SELECT_ONE·CONFIRM_INTERPRETATION 은 resolver 가 후보를 준 경우다.
            # 후보 없이 「고르세요」라고 물으면 사용자가 답할 수 없다.
            raise ValueError(
                f"{self.type} 역질문에 후보가 없는 field가 있습니다: {sorted(empty)}")
        return self

    def with_reason(self, reason_code: str) -> "ClarificationRequest":
        """사유만 바꾼 사본. 후보·경로·질문은 그대로다."""

        return self if reason_code == self.reason_code else self.model_copy(
            update={"reason_code": reason_code})

    @staticmethod
    def shape_for(options: Mapping[str, list[ClarificationOption]],
                  field_paths: Iterable[str]) -> ClarificationType:
        """후보 유무로 `type` 을 정한다 — 두 값은 파생 가능하다.

        `CONFIRM_INTERPRETATION` 은 파생할 수 없다(후보가 있는 SELECT_ONE 과
        구별되지 않는다). 그건 호출부가 명시한다.
        """

        return ("PROVIDE_VALUE"
                if any(not options.get(path) for path in field_paths)
                else "SELECT_ONE")


class PlanValidation(ContractModel):
    status: Literal[
        "ready", "needs_clarification", "out_of_scope",
        "unsupported_request", "policy_refusal",
    ]
    plan: ResolvedQueryPlan | None = None
    clarification: ClarificationRequest | None = None
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_status_contract(self) -> "PlanValidation":
        if self.status == "ready":
            if self.plan is None or self.clarification is not None or self.reasons:
                raise ValueError("ready에는 resolved plan만 있어야 합니다")
        elif self.status == "needs_clarification":
            if self.plan is not None or self.clarification is None or self.reasons:
                raise ValueError("needs_clarification에는 clarification만 있어야 합니다")
        elif self.plan is not None or self.clarification is not None or not self.reasons:
            raise ValueError("non-ready validation에는 typed reason만 있어야 합니다")
        return self


class ToolStatus(StrEnum):
    SUCCESS = "success"
    NEEDS_CLARIFICATION = "needs_clarification"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    OUT_OF_SCOPE = "out_of_scope"
    UNSUPPORTED = "unsupported"
    CONFLICT = "conflict"
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"
    FAILURE = "failure"


CoverageStatus = Literal["complete", "partial", "not_applicable"]
EvidenceKind = Literal["fact_value", "field_value", "correction_value", "chunk_text"]


class EvidenceCitation(ContractModel):
    kind: Literal["evidence"] = "evidence"
    evidence_kind: EvidenceKind
    evidence_id: HexId
    doc_id: str = Field(min_length=1)
    source_file_id: HexId
    rcept_dt: AsOf
    locator: str = Field(min_length=1)
    excerpt_prompt_safe: str = Field(min_length=1)
    verification_status: Literal["verified"] = "verified"

    @model_validator(mode="after")
    def validate_safe_excerpt(self) -> "EvidenceCitation":
        validate_as_of(self.rcept_dt, field_name="citation.rcept_dt")
        prefix = PROMPT_DATA_BEGIN + "\n"
        suffix = "\n" + PROMPT_DATA_END
        if not (self.excerpt_prompt_safe.startswith(prefix)
                and self.excerpt_prompt_safe.endswith(suffix)):
            raise ValueError("Evidence citation은 prompt-safe 경계를 보존해야 합니다")
        return self


class SourceReference(ContractModel):
    kind: Literal["source_reference"] = "source_reference"
    section_id: HexId
    doc_id: str = Field(min_length=1)
    source_file_id: HexId
    rcept_dt: AsOf
    locator: str = Field(min_length=1)
    excerpt_prompt_safe: str = Field(min_length=1)
    verification_status: Literal["source_roundtrip"] = "source_roundtrip"


class ToolLimitation(ContractModel):
    code: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    affected_doc_ids: list[str] = Field(default_factory=list)


DataT = TypeVar("DataT")


class ToolResult(ContractModel, Generic[DataT]):
    status: ToolStatus
    data: DataT | None = None
    domain_status: str | None = None
    coverage_status: CoverageStatus = "not_applicable"
    clarification: ClarificationRequest | None = None
    citations: list[EvidenceCitation] = Field(default_factory=list)
    limitations: list[ToolLimitation] = Field(default_factory=list)
    trace_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def enforce_status_contract(self) -> "ToolResult[DataT]":
        if self.status == ToolStatus.SUCCESS:
            if self.data is None or not self.citations:
                raise ValueError("success에는 data와 verified citation이 필요합니다")
            if (self.coverage_status != "complete" or self.clarification is not None
                    or self.limitations):
                raise ValueError("success에는 complete coverage와 무제한 결과만 허용됩니다")
        elif self.status == ToolStatus.PARTIAL:
            if self.data is None or not self.citations or not self.limitations:
                raise ValueError("partial에는 data, citation, limitation이 필요합니다")
            if self.coverage_status != "partial" or self.clarification is not None:
                raise ValueError("partial coverage 계약 오류")
        elif self.status == ToolStatus.NEEDS_CLARIFICATION:
            if (self.clarification is None or self.data is not None or self.citations
                    or self.limitations or self.coverage_status != "not_applicable"):
                raise ValueError("clarification은 확정 data/citation/limitation을 노출할 수 없습니다")
        else:
            if self.data is not None or self.citations or self.clarification is not None:
                raise ValueError("non-answer status는 data나 citation을 노출할 수 없습니다")
            if not self.limitations or self.coverage_status != "not_applicable":
                raise ValueError("non-answer status에는 typed limitation이 필요합니다")
        return self


class FinancialDomainStatus(StrEnum):
    OK = "ok"
    NOT_FOUND = "not_found"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS_ACCOUNT_PATH = "ambiguous_account_path"
    AMBIGUOUS_STATEMENT = "ambiguous_statement"
    AMBIGUOUS_PERIOD_ROLE = "ambiguous_period_role"
    AMBIGUOUS_UNIT = "ambiguous_unit"
    SAME_DOCUMENT_CONFLICT = "same_document_conflict"
    EXTRACT_UNSUPPORTED = "extract_unsupported"
    EVIDENCE_INVALID = "evidence_invalid"
    FAILURE = "failure"


class FinancialValue(ContractModel):
    output_id: str = Field(min_length=1)
    corp_code: str = Field(min_length=1)
    corp_name: str = Field(min_length=1)
    concept: FinancialConcept
    period_start: date | None
    period_end: date
    period_type: Literal["annual", "half", "quarter", "instant"]
    cumulative: bool | None
    scope: Literal["CFS", "SFS"]
    statement: Literal["BS", "IS", "CI", "CF"]
    account_path: str = Field(min_length=1)
    value_text: str = Field(min_length=1)
    raw_unit: str = Field(min_length=1)
    evidence_id: HexId


class FinancialResultData(ContractModel):
    values: list[FinancialValue] = Field(min_length=1)


class FinancialToolResult(ToolResult[FinancialResultData]):
    domain_status: FinancialDomainStatus | None = None

    @model_validator(mode="after")
    def enforce_financial_contract(self) -> "FinancialToolResult":
        allowed: dict[ToolStatus, set[FinancialDomainStatus]] = {
            ToolStatus.SUCCESS: {FinancialDomainStatus.OK},
            ToolStatus.PARTIAL: {FinancialDomainStatus.OK},
            ToolStatus.NEEDS_CLARIFICATION: {
                FinancialDomainStatus.AMBIGUOUS_ACCOUNT_PATH,
                FinancialDomainStatus.AMBIGUOUS_STATEMENT,
                FinancialDomainStatus.AMBIGUOUS_PERIOD_ROLE,
                FinancialDomainStatus.AMBIGUOUS_UNIT,
            },
            ToolStatus.NOT_FOUND: {FinancialDomainStatus.NOT_FOUND},
            ToolStatus.OUT_OF_SCOPE: {FinancialDomainStatus.OUT_OF_SCOPE},
            ToolStatus.UNSUPPORTED: {FinancialDomainStatus.EXTRACT_UNSUPPORTED},
            ToolStatus.CONFLICT: {FinancialDomainStatus.SAME_DOCUMENT_CONFLICT},
            ToolStatus.EVIDENCE_UNAVAILABLE: {FinancialDomainStatus.EVIDENCE_INVALID},
            ToolStatus.FAILURE: {FinancialDomainStatus.FAILURE},
        }
        if self.domain_status not in allowed[self.status]:
            raise ValueError("ToolStatus와 FinancialDomainStatus가 일치하지 않습니다")
        if self.status in {ToolStatus.SUCCESS, ToolStatus.PARTIAL}:
            assert self.data is not None
            output_ids = [row.output_id for row in self.data.values]
            if len(output_ids) != len(set(output_ids)):
                raise ValueError("Financial output_id가 중복되었습니다")
            expected = {row.evidence_id for row in self.data.values}
            actual = {citation.evidence_id for citation in self.citations}
            if expected != actual or any(
                    citation.evidence_kind != "fact_value" for citation in self.citations):
                raise ValueError("모든 FinancialValue에는 fact_value citation이 필요합니다")
        return self


class NarrativeHit(ContractModel):
    rank: int = Field(ge=1)
    score: float
    chunk_id: HexId
    evidence_id: HexId
    section_id: HexId
    doc_id: str = Field(min_length=1)
    source_file_id: HexId
    corp_code: str = Field(min_length=1)
    corp_name: str = Field(min_length=1)
    doc_group: str = Field(min_length=1)
    rcept_dt: AsOf
    path: str
    locator: str = Field(min_length=1)
    text_prompt_safe: str = Field(min_length=1)


class NarrativeResultData(ContractModel):
    hits: list[NarrativeHit] = Field(min_length=1)


class ExecutionResult(ContractModel):
    plan_schema_version: Literal["0.1"] = "0.1"
    plan_revision: int
    results: list[ToolResult[Any]] = Field(min_length=1)
