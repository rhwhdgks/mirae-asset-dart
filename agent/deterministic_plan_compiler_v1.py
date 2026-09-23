"""Typed Stage1 v1 resolution and execution-plan boundary.

This first boundary owns only source binding, compiler-generated roots, and
hashes.  Semantic interpretation and resolver policy remain separate inputs.
The generic lowering core dispatches only on a qid-free semantic structural
signature plus ordered typed resolution kinds.  ``compile_stage1_v1`` is an
approved fixture/demo adapter for exact question-ID/text checks; its ID is not
a production lowering key.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Literal, Mapping
import unicodedata

from pydantic import (
    BaseModel, ConfigDict, Field, StringConstraints, field_validator,
    model_validator,
)
from typing_extensions import Annotated

from agent.contracts import (
    DateRange, Derivation, DocumentSelector, EventSelector, FactSpec,
    FieldOutputSpec,
    FinancialConcept, OutputRef, PremiseClaim, ResolvedDisclosureTask,
    ResolvedCorrectionTask, ResolvedDocumentTask,
    ResolvedEventTask, ResolvedFinancialTask, ResolvedNarrativeTask,
    ResolvedQueryPlan, RECENT_Q4_FANOUT_APPLIED_DEFAULT,
    TaskVerificationRef,
)
from agent.compiled_answer_contract_v1 import (
    AnswerGroup as CompiledAnswerGroup,
    AnswerProjection,
    ActivationPredicate,
    CompiledAnswerContract,
    CompiledAnswerItem,
    CoveragePartition,
    CompiledPremiseContract,
    LimitationBinding,
    ProjectionField,
    WholeTargetBinding,
    SourceFieldBinding,
    SourcePremiseAuthority,
    SupportRequirement,
)
from agent.planning import _slot_value_kind, _target_date_range, normalize_slot_names
from agent.semantic_intent_v1 import (
    HCX_SEMANTIC_INTENT_WIRE_V1,
    HcxSemanticIntentWire,
    SemanticIntent,
    normalize_semantic_intent,
    semantic_intent_digest,
)
from agent.stage1_v1_funding_semantics import (
    FUNDING_CATEGORY_TYPES,
    funding_categories,
    funding_requested_slots,
)
from agent.stage1_v1_event_family_semantics import (
    bond_face_value_slot,
    facilities_investment_requested_slots,
    is_bond_face_value_request,
    is_facilities_investment_request,
)
RESOLUTION_VERSION = "stage1-authoritative-resolution/1.2"
EXECUTION_PLAN_VERSION = "stage1-execution-plan/1.1"
COMPILED_SLICE_VERSION = "stage1-deterministic-compiled-slice/1.1"
# ═══════════════════════════════════════════════════════════════════════════════
# 동결 vertical slice 8문항 — **오프라인 fixture 생성·검증 전용** (question ID 결속 코드는 이 구획들뿐이다)
#   · 상수(이 블록) · `_validate_g_*_approved_resolution` · `compile_stage1_v1`/`compile_g_*` · `_demo_*`
#   · `write/verify_vertical_slice_artifacts` (`python -m agent.deterministic_plan_compiler_v1 --write/--check`)
# 런타임 dispatch(`compile_stage1_v1_generic` → `_selected_handler` → `STAGE1_V1_HANDLER_REGISTRY`)는
# 이 구획의 어떤 상수·함수도 참조하지 않는다. 슬라이스는 「같은 입력이면 같은 계획」의 회귀 고정점이다.
# ═══════════════════════════════════════════════════════════════════════════════
G_A_001 = "G-A-001"
G_A_004 = "G-A-004"
G_A_010 = "G-A-010"
G_I_004 = "G-I-004"
G_I_006 = "G-I-006"
G_I_009 = "G-I-009"
G_O_001 = "G-O-001"
R_A_002 = "R-A-002"
G_A_001_QUESTION = "삼성전자의 2025년 연결기준 매출액은 얼마인가?"
R_A_002_QUESTION = "삼성전자 2025년 매출 얼마야?"
G_A_004_QUESTION = (
    "삼성전자와 SK하이닉스 중 2025년 연결 매출액이 큰 기업은 어디이며 "
    "차이는 얼마인가?"
)
G_A_010_QUESTION = (
    "한화에어로스페이스의 2026년 1분기보고서에서 확인되는 주요 투자계획을 "
    "투자 대상·목적·금액·기간별로 정리하고, 확인되지 않는 항목은 구분해줘."
)
G_I_004_QUESTION = (
    "2025년 12월 17일까지 접수된 Ford 배터리 계약 관련 최신 공시 내용과, "
    "제공 코퍼스 기준 계약의 최종 상태를 구분해 설명해줘."
)
G_I_006_QUESTION = (
    "Freudenberg 계약의 정정 후 계약금액과 해지금액은 같으며, 다르다면 왜 다른가?"
)
G_I_009_QUESTION = (
    "두산퓨얼셀 계약이 해지된 이유와 계약 효력발생 조건은 무엇이었는가?"
)
G_O_001_QUESTION = (
    "삼성전자의 2023년과 2025년 사업보고서를 비교해 사업부문·주요 제품 및 "
    "서비스·매출구성에서 확인되는 핵심 변화를 근거와 함께 설명해줘."
)
G_I_004_ORDERING_DETAIL = (
    "same-day candidate order is fixture inventory order, not chronology"
)
G_I_004_ORDERING_EVIDENCE_REFS = ("source-ordering:G-I-004",)
G_I_004_IDENTITY_DETAIL = (
    "termination report cannot be attributed to either original contract"
)
G_I_004_IDENTITY_EVIDENCE_REFS = ("source-identity:G-I-004",)
SLICE_ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / (
    "fixtures/stage1_v1_vertical_slices")
G_A_001_ARTIFACT = SLICE_ARTIFACT_ROOT / "g_a_001.json"
G_A_004_ARTIFACT = SLICE_ARTIFACT_ROOT / "g_a_004.json"
G_A_010_ARTIFACT = SLICE_ARTIFACT_ROOT / "g_a_010.json"
G_I_004_ARTIFACT = SLICE_ARTIFACT_ROOT / "g_i_004.json"
G_I_006_ARTIFACT = SLICE_ARTIFACT_ROOT / "g_i_006.json"
G_I_009_ARTIFACT = SLICE_ARTIFACT_ROOT / "g_i_009.json"
G_O_001_ARTIFACT = SLICE_ARTIFACT_ROOT / "g_o_001.json"
R_A_002_ARTIFACT = SLICE_ARTIFACT_ROOT / "r_a_002.json"

G_A_010_CORP_CODE = "00126566"
G_A_010_CORP_NAME = "한화에어로스페이스"
G_A_010_DOCUMENT_ID = "periodic_20260513000860"
G_A_010_RECEIPT_NO = "20260513000860"
G_A_010_DOCUMENT_PROOF = (
    f"source-document:{G_A_010_DOCUMENT_ID}")
G_A_010_NARRATIVE_PROOF = (
    f"source-narrative:{G_A_010_DOCUMENT_ID}")
G_A_010_SOURCE_CROSS_CHECK_DETAIL = (
    "partial_consistent: 414 claim anchors match, 0 conflicts, "
    "unmatched source ranges remain unverified"
)
G_A_010_SOURCE_CROSS_CHECK_EVIDENCE_REFS = (
    f"source-cross-check:{G_A_010_DOCUMENT_ID}",
)
R_A_002_DEFAULT_BASIS = (
    "canonical primary statement for the company and period")
R_A_002_DEFAULT_EVIDENCE_REFS = (
    "canonical:primary-statement:00126380:2025-12-31",
)
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
BuildId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
NonEmpty = Annotated[str, StringConstraints(min_length=1)]
WholeTargetId = Annotated[str, StringConstraints(
    pattern=r"^whole-target-[1-9][0-9]*$")]
DateStamp = Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
ReceiptStamp = Annotated[str, StringConstraints(pattern=r"^[0-9]{14}$")]
CoordinateProofId = Annotated[
    str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
DocumentFactSourceClass = Literal["correction", "disclosure"]

RESOLUTION_SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_authoritative_resolution_v1.schema.json")
RESOLUTION_SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_authoritative_resolution_v1.schema.sha256")
PLAN_SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_execution_plan_v1.schema.json")
PLAN_SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_execution_plan_v1.schema.sha256")
COMPILED_SLICE_SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_deterministic_compiled_slice_v1.schema.json")
COMPILED_SLICE_SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_deterministic_compiled_slice_v1.schema.sha256")

class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True,
        revalidate_instances="always",
    )


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False,
                      default=_json_default)

def canonical_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", warnings=False)
    if isinstance(value, (date, Enum)):
        return value.value if isinstance(value, Enum) else value.isoformat()
    raise TypeError(f"canonical JSON으로 직렬화할 수 없는 값: {type(value).__name__}")


def _strict_json_model(model: type[BaseModel], value: Any) -> BaseModel:
    """Revalidate models strictly; mappings use canonical JSON scalar rules."""
    if isinstance(value, model):
        return model.model_validate(
            value.model_dump(mode="python", warnings=False), strict=True)
    if isinstance(value, Mapping):
        return model.model_validate_json(canonical_json(value), strict=True)
    if isinstance(value, (str, bytes, bytearray)):
        return model.model_validate_json(value, strict=True)
    return model.model_validate(value, strict=True)


def _unique(values: list[Any], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label}에는 중복 값이 있을 수 없습니다")


def _valid_date_stamp(value: str, label: str) -> str:
    try:
        date(int(value[:4]), int(value[4:6]), int(value[6:]))
    except ValueError as exc:
        raise ValueError(f"{label}는 실제 달력 날짜여야 합니다") from exc
    return value


class AppliedDefault(_StrictFrozenModel):
    policy: NonEmpty
    basis: NonEmpty
    value: NonEmpty
    evidence_refs: list[NonEmpty] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_evidence(self) -> "AppliedDefault":
        _unique(list(self.evidence_refs), "applied default evidence_refs")
        return self


class ResolutionFieldProof(_StrictFrozenModel):
    source_field_index: int = Field(ge=0)
    surface: NonEmpty
    proof_ref: NonEmpty


class FinancialResolution(_StrictFrozenModel):
    kind: Literal["financial"] = "financial"
    corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    corp_name: NonEmpty
    concept: FinancialConcept
    period_start: date | None
    period_end: date
    period_type: Literal["annual", "half", "quarter", "instant"]
    scope: Literal["CFS", "SFS"]
    statement: Literal["BS", "IS", "CI", "CF"]
    view: Literal["as_filed", "restated"]
    as_of: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    cumulative: bool | None

    @model_validator(mode="after")
    def valid_period(self) -> "FinancialResolution":
        _valid_date_stamp(self.as_of, "as_of")
        if self.period_type == "instant":
            if self.period_start is not None:
                raise ValueError("instant resolution에는 period_start가 없어야 합니다")
        elif self.period_start is None:
            raise ValueError("duration resolution에는 period_start가 필요합니다")
        if self.period_start is not None and self.period_start > self.period_end:
            raise ValueError("period_start는 period_end 이후일 수 없습니다")
        return self


class FinancialComparisonOperand(_StrictFrozenModel):
    """One typed financial fact used by a comparison lowering.

    Operands are deliberately not answer roots.  They are source-side facts
    consumed by compiler-owned ``argmax``/``absolute_difference`` derivations.  The
    explicit tag keeps this branch closed when it is embedded in a
    ``FinancialComparisonResolution``.
    """

    kind: Literal["financial_operand"] = "financial_operand"
    operand_id: NonEmpty
    corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    corp_name: NonEmpty
    concept: FinancialConcept
    period_start: date | None
    period_end: date
    period_type: Literal["annual", "half", "quarter", "instant"]
    scope: Literal["CFS", "SFS"]
    statement: Literal["BS", "IS", "CI", "CF"]
    view: Literal["as_filed", "restated"]
    as_of: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    cumulative: bool | None
    proof_ref: NonEmpty

    @model_validator(mode="after")
    def valid_period(self) -> "FinancialComparisonOperand":
        _valid_date_stamp(self.as_of, "comparison operand as_of")
        if self.period_type == "instant":
            if self.period_start is not None:
                raise ValueError(
                    "instant comparison operand에는 period_start가 없어야 합니다")
        elif self.period_start is None:
            raise ValueError(
                "duration comparison operand에는 period_start가 필요합니다")
        if self.period_start is not None and self.period_start > self.period_end:
            raise ValueError(
                "comparison operand period_start는 period_end 이후일 수 없습니다")
        return self


class FinancialComparisonResolution(_StrictFrozenModel):
    """Tagged multi-fact resolution owned by the comparison compiler path."""

    kind: Literal["financial_comparison"] = "financial_comparison"
    operands: list[FinancialComparisonOperand] = Field(min_length=2, max_length=8)
    requested_operators: list[Literal[
        "argmax", "difference", "absolute_difference", "percent_change",
        "discrete_from_cumulative", "concept_ratio", "sum",
    ]] = Field(default_factory=lambda: ["argmax", "absolute_difference"],
              min_length=1, max_length=2)
    verification_claim: NonEmpty | None = None
    verification_premise_id: NonEmpty | None = None
    #: ``requested_operators == ["concept_ratio"]`` 일 때만 있다 — 그 파생이
    #: 낳을 ``Derivation.presentation`` 을 여기 실어 compiler 로 옮긴다.
    presentation: Literal["percent", "multiple"] | None = None
    # 이슈 #124 — 「가장 작은/최솟값」은 「가장 큰/최댓값」과 같은 ``argmax``
    # 연산자를 그대로 쓰고 방향만 뒤집는다. optional — 기존 authority는 그대로
    # "maximum"(무방향과 동일). "argmax"가 requested_operators에 없으면 방향은
    # 의미가 없으므로 "maximum"으로 고정한다.
    direction: Literal["maximum", "minimum"] = "maximum"

    @model_validator(mode="after")
    def validate_operands(self) -> "FinancialComparisonResolution":
        _unique([row.operand_id for row in self.operands],
                "comparison operand_id")
        _unique(self.requested_operators, "comparison requested operators")
        if self.direction == "minimum" and "argmax" not in self.requested_operators:
            raise ValueError(
                "comparison direction=minimum은 argmax 연산자에만 허용됩니다")
        first = self.operands[0]
        second = self.operands[1]
        if (self.verification_claim is None) != (
                self.verification_premise_id is None):
            raise ValueError(
                "financial verification claim과 premise ID는 함께 있어야 합니다")
        if (self.requested_operators == ["concept_ratio"]) != (
                self.presentation is not None):
            raise ValueError(
                "concept_ratio presentation은 concept_ratio 요청에만 있어야 합니다")
        corp_codes = [operand.corp_code for operand in self.operands]
        company_comparison = len(set(corp_codes)) > 1
        if company_comparison:
            _unique(corp_codes, "company comparison corp_code")
            # 이슈 #59 1단계 — 회사가 다른 concept_ratio(「A사 매출은 B사의
            # 몇 배」)는 argmax/absolute_difference 와 같은 자리에 둔다.
            # 「같은 concept·같은 기간, 회사만 다름」은 아래 coordinate_fields
            # 동등성 검사가 그대로 강제한다(그 검사는 operator 를 가리지
            # 않는다) — concept_ratio 전용 추가 제약이 필요 없다.
            # 이슈 #124 (합계) — 회사가 다른 sum(「A사와 B사의 매출액 합계는
            # 얼마인가」)도 같은 자리에 둔다. 같은 이유로 coordinate_fields
            # 동등성 검사가 이미 「같은 concept·같은 기간·같은 scope, 회사만
            # 다름」을 강제하므로 sum 전용 추가 제약이 필요 없다.
            if any(operator not in {
                    "argmax", "absolute_difference", "concept_ratio", "sum"}
                   for operator in self.requested_operators):
                raise ValueError(
                    "company comparison operator는 "
                    "argmax/absolute_difference/concept_ratio/sum이어야 합니다")
            if (self.requested_operators == ["concept_ratio"]
                    and len(self.operands) != 2):
                raise ValueError(
                    "company comparison concept_ratio는 operand 두 개만 지원합니다")
            if (self.requested_operators == ["sum"]
                    and len(self.operands) != 2):
                raise ValueError(
                    "company comparison sum은 operand 두 개만 지원합니다")
            # 이슈 #182 — `as_of`는 여기서 일부러 뺀다. 시점(instant) 개념의
            # 회계연도말 좌표는 `_instant_fiscal_year_end_operand_as_of`(#149)
            # 가 **좌표별로 독립적으로** 자기 회사의 사업보고서(와 그 정정)로
            # 좁힌다 — 서로 다른 회사는 애초에 서로 다른 날짜에 각자의
            # 사업보고서를 낸다(실호출로 직접 확인함: KB금융
            # 20260324/신한지주 20260318처럼 같은 2025년 말 총자산이라도
            # `as_of`가 일치할 이유가 없다). concept·기간·scope·statement·
            # view·cumulative가 모두 같으면 같은 사실을 가리키는 것이 이미
            # 충분히 증명되므로, 회사마다 다른 제출일 자체를 좌표 불일치로
            # 거절하면 안 된다 — 「KB금융과 신한지주 중 2025년 연결
            # 총자산이 큰 곳과 차이를 알려줘」류(DEV-FDR-013/017,
            # EG2-023)가 그렇게 거절됐다.
            coordinate_fields = (
                "concept", "period_start", "period_end", "period_type", "scope",
                "statement", "view", "cumulative",
            )
            if any(
                    getattr(operand, field) != getattr(first, field)
                    for operand in self.operands[1:]
                    for field in coordinate_fields):
                raise ValueError(
                    "company comparison operands는 동일한 financial coordinate여야 합니다")
            if len(self.operands) > 2 and (
                    self.requested_operators != ["argmax"]
                    or self.verification_claim is not None):
                raise ValueError(
                    "N-ary company ranking은 argmax 단일 연산만 지원합니다")
            if (self.verification_claim is not None
                    and self.requested_operators != ["argmax"]):
                raise ValueError(
                    "company verification claim은 argmax 단독 비교에만 허용됩니다")
        else:
            if len(self.operands) != 2:
                raise ValueError(
                    "same-company financial relation은 operand 두 개가 필요합니다")
            if self.verification_claim is not None:
                raise ValueError(
                    "same-company financial relation에는 verification claim을 둘 수 없습니다")
            if first.corp_name != second.corp_name:
                raise ValueError("time derivation operands는 동일한 회사를 가리켜야 합니다")
            if self.requested_operators == ["concept_ratio"]:
                # 지표 간 나눗셈은 **다른 concept, 같은 시점**이다 — 시점 비교
                # (`same_period`/`same_scope`/`same_view` 중 한 축만 다름)와는
                # 반대 모양이라 별도 분기로 닫는다. statement 는 일부러 강제하지
                # 않는다 — ROE·ROA처럼 분자(IS)·분모(BS)가 다른 표라서다.
                if first.concept == second.concept:
                    raise ValueError(
                        "concept_ratio operands는 서로 다른 concept이어야 합니다")
                if first.scope != second.scope:
                    raise ValueError(
                        "concept_ratio operands는 동일한 scope여야 합니다")
                if first.view != second.view or first.as_of != second.as_of:
                    raise ValueError(
                        "concept_ratio operands는 동일한 view·as_of여야 합니다")
                if first.period_end != second.period_end:
                    raise ValueError(
                        "concept_ratio operands는 동일한 기간이어야 합니다")
            else:
                if any(operator not in {
                        "difference", "absolute_difference", "percent_change",
                        "discrete_from_cumulative", "sum",
                } for operator in self.requested_operators):
                    raise ValueError(
                        "time comparison operator가 financial axis와 맞지 않습니다")
                for field in ("concept", "statement"):
                    if getattr(second, field) != getattr(first, field):
                        raise ValueError(
                            "time derivation operands는 동일한 financial axis여야 합니다")
                same_period = all(
                        getattr(second, field) == getattr(first, field)
                        for field in ("period_start", "period_end", "period_type", "cumulative")
                )
                same_scope = first.scope == second.scope
                same_view = first.view == second.view
                if sum((not same_period, not same_scope, not same_view)) != 1:
                    raise ValueError(
                        "financial relation은 기간, statement scope, view 중 한 축만 달라야 합니다")
                if (not same_scope
                        and self.requested_operators != ["absolute_difference"]):
                    raise ValueError(
                        "scope comparison은 absolute_difference 하나만 요청할 수 있습니다")
                if not same_scope and first.as_of != second.as_of:
                    raise ValueError(
                        "scope comparison operands는 동일한 as_of여야 합니다")
                if not same_view and (
                        first.as_of != second.as_of
                        or [first.view, second.view] != ["as_filed", "restated"]
                        or self.requested_operators != ["difference"]):
                    raise ValueError(
                        "view comparison은 as_filed→restated 동일 as_of difference여야 합니다")
        return self


class DocumentFactOperand(_StrictFrozenModel):
    """One source coordinate for a generic document-fact comparison.

    This is a proof coordinate, not an extracted answer value.  The source
    class is intentionally open-ended so the same lowering can compare any
    two document-reported monetary facts.  The relation between the two
    operands (issuer, identity and uniqueness) is closed by the enclosing
    resolution and by the item-level validator.
    """

    operand_id: NonEmpty
    issuer_corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    issuer_corp_name: NonEmpty
    source_class: DocumentFactSourceClass
    doc_id: NonEmpty
    receipt_no: ReceiptStamp
    path: NonEmpty
    locator: NonEmpty
    source_file_id: CoordinateProofId
    evidence_id: CoordinateProofId
    value_kind: Literal["money"] = "money"

    @model_validator(mode="after")
    def validate_document_identity(self) -> "DocumentFactOperand":
        if not self.doc_id.endswith(self.receipt_no):
            raise ValueError("document fact doc_id는 receipt_no와 구조적으로 결속되어야 합니다")
        if not self.path.strip() or not self.locator.strip():
            raise ValueError("document fact path/locator는 비어 있을 수 없습니다")
        return self


class DocumentReasonEvidence(_StrictFrozenModel):
    """A text-evidence coordinate bound to one comparison operand."""

    operand_id: NonEmpty
    issuer_corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    issuer_corp_name: NonEmpty
    source_class: DocumentFactSourceClass
    doc_id: NonEmpty
    receipt_no: ReceiptStamp
    path: NonEmpty
    locator: NonEmpty
    source_file_id: CoordinateProofId
    evidence_id: CoordinateProofId
    value_kind: Literal["text"] = "text"

    @model_validator(mode="after")
    def validate_document_identity(self) -> "DocumentReasonEvidence":
        if not self.doc_id.endswith(self.receipt_no):
            raise ValueError("reason evidence doc_id는 receipt_no와 구조적으로 결속되어야 합니다")
        if not self.path.strip() or not self.locator.strip():
            raise ValueError("reason evidence path/locator는 비어 있을 수 없습니다")
        return self


class DocumentFactComparisonResolution(_StrictFrozenModel):
    """Two ordered, same-issuer monetary document facts.

    The model owns only generic source coordinates and proof identities.  It
    deliberately has no extracted amounts, answer-root IDs, or answer text.
    """

    kind: Literal["document_fact_comparison"] = "document_fact_comparison"
    operands: list[DocumentFactOperand] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_operands(self) -> "DocumentFactComparisonResolution":
        operands = self.operands
        _unique([row.operand_id for row in operands], "document comparison operand_id")
        _unique([row.doc_id for row in operands], "document comparison doc_id")
        _unique([row.receipt_no for row in operands], "document comparison receipt_no")
        _unique([row.evidence_id for row in operands], "document comparison evidence_id")
        _unique(
            [(row.source_file_id, row.path, row.locator) for row in operands],
            "document comparison coordinate",
        )
        issuer = (operands[0].issuer_corp_code, operands[0].issuer_corp_name)
        if any((row.issuer_corp_code, row.issuer_corp_name) != issuer
               for row in operands[1:]):
            raise ValueError("document comparison operands는 동일 issuer여야 합니다")
        if any(row.value_kind != "money" for row in operands):
            raise ValueError("document comparison operands는 money kind여야 합니다")
        return self


class DocumentReasonEvidenceResolution(_StrictFrozenModel):
    """One source-bound text proof for the comparison's conditional reason."""

    kind: Literal["document_reason_evidence"] = "document_reason_evidence"
    evidence: DocumentReasonEvidence


class DocumentAttributeEvidence(_StrictFrozenModel):
    """One text attribute proven by one exact disclosure coordinate.

    The payload deliberately excludes extracted text and answer-root IDs.  A
    relative document role closes cross-document chronology without making a
    question ID, field surface, or known receipt a generic routing key.
    """

    issuer_corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    issuer_corp_name: NonEmpty
    document_role: Literal["earlier", "later"]
    doc_id: NonEmpty
    receipt_no: ReceiptStamp
    path: NonEmpty
    locator: NonEmpty
    source_file_id: CoordinateProofId
    evidence_id: CoordinateProofId
    value_kind: Literal["text"] = "text"

    @model_validator(mode="after")
    def validate_document_identity(self) -> "DocumentAttributeEvidence":
        if not self.doc_id.endswith(self.receipt_no):
            raise ValueError(
                "document attribute doc_id는 receipt_no와 결속되어야 합니다")
        _valid_date_stamp(
            self.receipt_no[:8], "document attribute receipt date")
        if not self.path.strip() or not self.locator.strip():
            raise ValueError(
                "document attribute path/locator는 비어 있을 수 없습니다")
        return self


class DocumentAttributeEvidenceResolution(_StrictFrozenModel):
    """Item-scoped generic text-attribute coordinate resolution."""

    kind: Literal[
        "document_attribute_evidence"] = "document_attribute_evidence"
    evidence: DocumentAttributeEvidence


class PeriodicNarrativeEvidence(_StrictFrozenModel):
    """One report-section coordinate for a user comparison axis.

    The coordinate contains no extracted prose. ``source_field_index`` binds
    the section to the ordered semantic field; the one remaining final field
    is compiled as the cross-axis synthesis field.
    """

    source_field_index: int = Field(ge=0)
    axis_id: NonEmpty
    path: NonEmpty
    locator: NonEmpty
    source_file_id: CoordinateProofId
    evidence_id: CoordinateProofId
    value_kind: Literal["text"] = "text"

    @model_validator(mode="after")
    def validate_coordinate(self) -> "PeriodicNarrativeEvidence":
        if not self.axis_id.strip() or not self.path.strip() or not self.locator.strip():
            raise ValueError("periodic narrative axis/path/locator는 비어 있을 수 없습니다")
        return self


class PeriodicNarrativeDocument(_StrictFrozenModel):
    """One exact periodic document and its ordered narrative coordinates."""

    source_period_index: int = Field(ge=0)
    issuer_corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    issuer_corp_name: NonEmpty
    doc_id: NonEmpty
    receipt_no: ReceiptStamp
    period_start: date
    period_end: date
    source_file_id: CoordinateProofId
    evidence: list[PeriodicNarrativeEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_document(self) -> "PeriodicNarrativeDocument":
        if not self.doc_id.endswith(self.receipt_no):
            raise ValueError(
                "periodic narrative doc_id는 receipt_no와 결속되어야 합니다")
        _valid_date_stamp(self.receipt_no[:8], "periodic narrative receipt date")
        if self.period_start > self.period_end:
            raise ValueError("periodic narrative period_start는 period_end 이후일 수 없습니다")
        indexes = [row.source_field_index for row in self.evidence]
        if indexes != sorted(indexes):
            raise ValueError("periodic narrative evidence는 source field 순서여야 합니다")
        _unique(indexes, "periodic narrative source_field_index")
        _unique([row.axis_id for row in self.evidence], "periodic narrative axis_id")
        _unique([row.evidence_id for row in self.evidence],
                "periodic narrative evidence_id")
        _unique(
            [(row.source_file_id, row.path, row.locator) for row in self.evidence],
            "periodic narrative evidence coordinate",
        )
        if any(row.source_file_id != self.source_file_id for row in self.evidence):
            raise ValueError(
                "periodic narrative evidence는 document main source_file에 결속되어야 합니다")
        return self


class PeriodicNarrativeComparisonResolution(_StrictFrozenModel):
    """Two ordered report coordinates for a generic open narrative compare.

    This payload is deliberately coordinate-only: it carries neither extracted
    report text nor a proposed comparison answer.
    """

    kind: Literal[
        "periodic_narrative_comparison"] = "periodic_narrative_comparison"
    document_group: NonEmpty
    documents: list[PeriodicNarrativeDocument] = Field(
        min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_documents(self) -> "PeriodicNarrativeComparisonResolution":
        documents = self.documents
        if [row.source_period_index for row in documents] != [0, 1]:
            raise ValueError(
                "periodic narrative documents는 semantic period 0,1 순서여야 합니다")
        issuer = (
            documents[0].issuer_corp_code, documents[0].issuer_corp_name)
        if any((row.issuer_corp_code, row.issuer_corp_name) != issuer
               for row in documents[1:]):
            raise ValueError("periodic narrative documents는 동일 issuer여야 합니다")
        if documents[0].period_end >= documents[1].period_start:
            raise ValueError("periodic narrative documents의 기간 순서가 겹치거나 뒤집혔습니다")
        _unique([row.doc_id for row in documents], "periodic narrative doc_id")
        _unique([row.receipt_no for row in documents],
                "periodic narrative receipt_no")
        _unique([row.source_file_id for row in documents],
                "periodic narrative source_file_id")
        inventories = [[
            (evidence.source_field_index, evidence.axis_id)
            for evidence in row.evidence
        ] for row in documents]
        if inventories[0] != inventories[1]:
            raise ValueError(
                "periodic narrative documents의 ordered axis inventory가 다릅니다")
        all_evidence = [evidence for row in documents for evidence in row.evidence]
        _unique([row.evidence_id for row in all_evidence],
                "periodic narrative global evidence_id")
        _unique(
            [(row.source_file_id, row.path, row.locator) for row in all_evidence],
            "periodic narrative global coordinate",
        )
        return self


class NarrativeMatrixCell(_StrictFrozenModel):
    """One grounded company/document/topic coordinate for narrative fan-out."""

    cell_id: NonEmpty
    corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    corp_name: NonEmpty
    doc_id: NonEmpty
    receipt_no: ReceiptStamp
    period_start: date
    period_end: date
    topics: list[NonEmpty] = Field(min_length=1)
    document_proof: ResolutionSourceProof

    @model_validator(mode="after")
    def validate_cell(self) -> "NarrativeMatrixCell":
        if (self.doc_id != f"periodic_{self.receipt_no}"
                or self.period_start > self.period_end
                or self.document_proof.source_receipt != self.receipt_no):
            raise ValueError("narrative matrix cell document/proof 결속이 잘못되었습니다")
        _unique(self.topics, "narrative matrix topics")
        return self


class NarrativeMatrixResolution(_StrictFrozenModel):
    """Bounded internal matrix; public QueryPlan still carries normal tasks."""

    kind: Literal["narrative_matrix"] = "narrative_matrix"
    cells: list[NarrativeMatrixCell] = Field(min_length=2, max_length=16)

    @model_validator(mode="after")
    def validate_cells(self) -> "NarrativeMatrixResolution":
        work_units = sum(len(row.topics) for row in self.cells)
        if not 2 <= work_units <= 16:
            raise ValueError("narrative matrix work unit 수는 2..16이어야 합니다")
        _unique([(row.corp_code, row.doc_id) for row in self.cells],
                "narrative matrix corp/document cells")
        _unique([row.cell_id for row in self.cells], "narrative matrix cell_id")
        return self


class LimitationProvenance(_StrictFrozenModel):
    """Resolver-side provenance for a field-level typed limitation."""

    code: Literal[
        "intraday_order_unavailable", "ambiguous_event_origin",
        "source_cross_check_partial",
    ]
    family: Literal["ordering", "identity_lineage", "source_scope"]
    detail: NonEmpty
    evidence_refs: list[NonEmpty] = Field(min_length=1)
    original_receipts: list[ReceiptStamp] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_registry(self) -> "LimitationProvenance":
        expected = {
            "intraday_order_unavailable": "ordering",
            "ambiguous_event_origin": "identity_lineage",
            "source_cross_check_partial": "source_scope",
        }[self.code]
        if self.family != expected:
            raise ValueError("resolution limitation code/family 불일치")
        _unique(list(self.evidence_refs), "resolution limitation evidence_refs")
        _unique(list(self.original_receipts), "resolution limitation original_receipts")
        return self


class SameDayDocumentCandidate(_StrictFrozenModel):
    rcept_no: ReceiptStamp
    document_kind: Literal["termination", "correction"]
    proof_ref: NonEmpty


class ResolutionSourceProof(_StrictFrozenModel):
    """A non-user-facing proof tied to one source receipt."""

    source_receipt: ReceiptStamp
    proof_ref: NonEmpty


class PeriodicDocumentNarrativeResolution(_StrictFrozenModel):
    """Coordinates/provenance for one periodic-document narrative retrieval.

    This branch intentionally carries no extracted investment values.  The
    document and narrative proofs identify the exact source that the two
    execution tasks must use; field-level coverage is a typed resolver
    authority that the compiler must preserve.
    """

    kind: Literal["periodic_document_narrative"] = (
        "periodic_document_narrative")
    corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    corp_name: NonEmpty
    document_id: NonEmpty
    receipt_no: ReceiptStamp
    document_proof: ResolutionSourceProof
    narrative_proof: ResolutionSourceProof
    # A source-coordinate retrieval hint, distinct from the user-facing
    # semantic target.  It is optional for generic counterfactuals whose
    # source section is not yet known, but becomes authoritative when the
    # resolver has proved the exact filing heading.
    source_retrieval_query: NonEmpty | None = None
    # Structured table routes retain the exact source header coordinates in
    # source-field order.  Empty is reserved for ordinary narrative sections;
    # a non-empty inventory proves that every requested table slot was present
    # in one unambiguous source table.
    source_header_proof_refs: list[NonEmpty] = Field(default_factory=list)
    canonical_requested_slots: list[NonEmpty] = Field(default_factory=list)
    executable_field_indexes: list[int] = Field(default_factory=lambda: [0, 1, 2])
    limited_field_indexes: list[int] = Field(default_factory=lambda: [3])
    source_cross_check_provenance: LimitationProvenance | None = None

    @model_validator(mode="after")
    def validate_coordinates(self) -> "PeriodicDocumentNarrativeResolution":
        _valid_date_stamp(self.receipt_no[:8], "periodic document receipt date")
        if (
                self.document_proof.source_receipt != self.receipt_no
                or self.narrative_proof.source_receipt != self.receipt_no
        ):
            raise ValueError(
                "periodic document/narrative proof가 source receipt와 다릅니다")
        if self.document_proof.proof_ref == self.narrative_proof.proof_ref:
            raise ValueError(
                "periodic document proof와 narrative proof는 분리되어야 합니다")
        executable = list(self.executable_field_indexes)
        limited = list(self.limited_field_indexes)
        _unique(executable, "periodic executable field indexes")
        _unique(limited, "periodic limited field indexes")
        if executable != sorted(executable) or limited != sorted(limited):
            raise ValueError("periodic field coverage index는 오름차순이어야 합니다")
        if not executable or min(executable + limited) < 0:
            raise ValueError(
                "periodic resolution에는 executable field가 하나 이상 필요합니다")
        coverage = set(executable) | set(limited)
        if set(executable) & set(limited) or sorted(coverage) != list(
                range(max(coverage) + 1)):
            raise ValueError(
                "periodic field coverage authority가 겹치거나 누락되었습니다")
        provenance = self.source_cross_check_provenance
        if limited and provenance is None:
            raise ValueError(
                "limited periodic field에는 source cross-check provenance가 필요합니다")
        if not limited and provenance is not None:
            raise ValueError(
                "complete periodic resolution에는 source cross-check limitation이 없어야 합니다")
        if provenance is not None and (
                provenance.code != "source_cross_check_partial"
                or provenance.family != "source_scope"
                or provenance.original_receipts
        ):
            raise ValueError(
                "periodic source cross-check limitation provenance가 다릅니다")
        headers = list(self.source_header_proof_refs)
        _unique(headers, "periodic source header proof refs")
        if headers and not self.canonical_requested_slots and len(headers) != len(coverage):
            raise ValueError(
                "periodic source header proof는 complete field coverage와 같아야 합니다")
        slots = list(self.canonical_requested_slots)
        _unique(slots, "periodic canonical requested slots")
        if slots and len(headers) != len(slots):
            raise ValueError(
                "periodic canonical slots는 source header proof와 하나씩 대응해야 합니다")
        return self

class SameDayDocumentCandidatesResolution(_StrictFrozenModel):
    kind: Literal["same_day_document_candidates"] = "same_day_document_candidates"
    issuer_corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    issuer_corp_name: NonEmpty
    as_of: DateStamp
    candidates: list[SameDayDocumentCandidate] = Field(min_length=2, max_length=2)
    ordering_provenance: LimitationProvenance

    @model_validator(mode="after")
    def validate_inventory(self) -> "SameDayDocumentCandidatesResolution":
        _valid_date_stamp(self.as_of, "same-day candidate as_of")
        if any(row.rcept_no[:8] != self.as_of for row in self.candidates):
            raise ValueError("same-day document candidate가 as_of 날짜와 다릅니다")
        _unique(
            [row.rcept_no for row in self.candidates],
            "same-day candidate receipt",
        )
        _unique(
            [row.proof_ref for row in self.candidates],
            "same-day candidate proof",
        )
        if {row.document_kind for row in self.candidates} != {
                "termination", "correction"}:
            raise ValueError(
                "same-day candidate에는 termination/correction이 각각 하나씩 필요합니다")
        if self.ordering_provenance.code != "intraday_order_unavailable" \
                or self.ordering_provenance.family != "ordering":
            raise ValueError("same-day ordering limitation provenance가 다릅니다")
        if self.ordering_provenance.original_receipts:
            raise ValueError("ordering provenance에는 original receipt가 없어야 합니다")
        if set(self.ordering_provenance.evidence_refs) & {
                row.proof_ref for row in self.candidates}:
            raise ValueError("ordering provenance는 candidate proof를 재사용할 수 없습니다")
        return self


class PublicEventSelectorFacets(_StrictFrozenModel):
    """User-visible event selector facets with exact canonical field proofs."""

    counterparty: NonEmpty | None = None
    counterparty_proof: ResolutionSourceProof | None = None
    product_keywords: list[NonEmpty] = Field(default_factory=list)
    product_keyword_proofs: list[ResolutionSourceProof] = Field(default_factory=list)
    contract_name: NonEmpty | None = None
    contract_name_proof: ResolutionSourceProof | None = None

    @model_validator(mode="after")
    def validate_facets(self) -> "PublicEventSelectorFacets":
        if (self.counterparty is None) != (self.counterparty_proof is None):
            raise ValueError("public counterparty와 proof는 함께 있어야 합니다")
        if len(self.product_keywords) != len(self.product_keyword_proofs):
            raise ValueError("public product keyword와 proof 수가 다릅니다")
        if (self.contract_name is None) != (self.contract_name_proof is None):
            raise ValueError("public contract name과 proof는 함께 있어야 합니다")
        if not (self.counterparty or self.product_keywords or self.contract_name):
            raise ValueError("public event selector facet은 비어 있을 수 없습니다")
        _unique(list(self.product_keywords), "public product keyword")
        proofs = [
            proof for proof in (
                self.counterparty_proof, *self.product_keyword_proofs,
                self.contract_name_proof,
            ) if proof is not None
        ]
        _unique([proof.proof_ref for proof in proofs], "public selector proof")
        return self

    def source_receipts(self) -> set[str]:
        return {
            proof.source_receipt for proof in (
                self.counterparty_proof, *self.product_keyword_proofs,
                self.contract_name_proof,
            ) if proof is not None
        }


class TerminationReportedStatusResolution(_StrictFrozenModel):
    kind: Literal["termination_reported_status"] = "termination_reported_status"
    issuer_corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    issuer_corp_name: NonEmpty
    status_receipt: ReceiptStamp
    event_key: NonEmpty
    event_key_proof: ResolutionSourceProof
    status_proof: ResolutionSourceProof
    identity_provenance: LimitationProvenance
    public_selector: PublicEventSelectorFacets | None = None

    @model_validator(mode="after")
    def validate_status(self) -> "TerminationReportedStatusResolution":
        _valid_date_stamp(self.status_receipt[:8], "termination status receipt date")
        if (
                self.event_key_proof.source_receipt != self.status_receipt
                or self.status_proof.source_receipt != self.status_receipt
        ):
            raise ValueError(
                "event-key/status proof가 status receipt에서 파생되지 않았습니다")
        if self.event_key_proof.proof_ref == self.status_proof.proof_ref:
            raise ValueError("event-key proof와 status proof는 분리되어야 합니다")
        if self.identity_provenance.code != "ambiguous_event_origin" \
                or self.identity_provenance.family != "identity_lineage":
            raise ValueError("termination identity limitation provenance가 다릅니다")
        original_receipts = list(self.identity_provenance.original_receipts)
        _unique(original_receipts, "termination original receipt")
        if len(original_receipts) < 2 or self.status_receipt in original_receipts:
            raise ValueError(
                "ambiguous event origin은 서로 다른 original receipt 2개 이상이 필요합니다")
        if set(self.identity_provenance.evidence_refs) & {
                self.event_key_proof.proof_ref, self.status_proof.proof_ref}:
            raise ValueError(
                "identity provenance는 event-key/status proof를 재사용할 수 없습니다")
        if (self.public_selector is not None
                and self.public_selector.source_receipts() != {self.status_receipt}):
            raise ValueError("termination public selector는 status receipt 증명이어야 합니다")
        return self


class DocumentCollectionResolution(_StrictFrozenModel):
    """Generic source authority for returning a company's document set.

    The resolver supplies only the company coordinate, corpus cutoff, and a
    proof that the selector is source-backed.  It deliberately carries no
    invented field list and no document answer values; those belong to the
    task result addressed by the whole-target binding.
    """

    kind: Literal["document_collection"] = "document_collection"
    corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    corp_name: NonEmpty
    as_of: DateStamp
    selector_proof_ref: NonEmpty
    # A topic-level collection remains a broad narrative search.  The query is
    # copied from the grounded semantic target and does not select a filing.
    retrieval_query: NonEmpty | None = None
    recent_selection: bool = False
    # An explicit latest periodic-report expression is closed by the resolver,
    # rather than being reinterpreted by the compiler as a broad collection.
    # These three fields are all-or-none so a public receipt selector always
    # retains its canonical identity and source proof together.
    selected_document_id: NonEmpty | None = None
    selected_receipt_no: ReceiptStamp | None = None
    selected_document_proof: ResolutionSourceProof | None = None

    @model_validator(mode="after")
    def validate_coordinate(self) -> "DocumentCollectionResolution":
        _valid_date_stamp(self.as_of, "document collection as_of")
        selected = (
            self.selected_document_id,
            self.selected_receipt_no,
            self.selected_document_proof,
        )
        if any(value is None for value in selected):
            if any(value is not None for value in selected):
                raise ValueError(
                    "selected periodic document identity/proof는 함께 필요합니다")
            return self
        assert self.selected_document_id is not None
        assert self.selected_receipt_no is not None
        assert self.selected_document_proof is not None
        # Canonical document identities are family-prefixed receipts.  The
        # selected-document path is shared by periodic, major and exchange
        # filings; restricting it to ``periodic_`` made a proved visible
        # correction impossible to return when its missing root predates the
        # corpus.  Keep the identity check exact and closed to supported public
        # document families rather than accepting an arbitrary doc_id suffix.
        expected_ids = {
            f"{family}_{self.selected_receipt_no}"
            for family in ("periodic", "major", "exchange", "holding")
        }
        if self.selected_document_id not in expected_ids:
            raise ValueError("selected document_id/receipt가 다릅니다")
        if self.selected_document_proof.source_receipt != self.selected_receipt_no:
            raise ValueError("selected document proof receipt가 다릅니다")
        return self


class HoldingSlotBinding(_StrictFrozenModel):
    """One requested holding field bound to a canonical DART coordinate.

    The binding contains no answer value.  It only freezes the user-facing
    field order and the ACODE/slot that Stage2 must read from the exact filing.
    """

    source_field_index: int = Field(ge=0)
    surface: NonEmpty
    slot: Literal[
        "previous_count", "previous_ratio", "current_count", "current_ratio",
        "delta_count", "delta_ratio", "report_reason", "holding_purpose",
        "change_method", "change_reason", "issuer", "filer", "report_type",
        "receipt", "receipt_date", "base_date", "parties", "public_entities",
        "nationality", "occupation", "filer_nationality", "filer_occupation",
        "restricted_personal_data",
    ]
    acode: NonEmpty | None = None
    value_kind: Literal["money", "percent", "date", "count", "text"]
    binding_status: Literal["executable", "qualified", "limited"] = "executable"
    restricted_types: list[NonEmpty] = Field(default_factory=list)
    party_name: NonEmpty | None = None
    party_proof: ResolutionSourceProof | None = None
    proof_ref: NonEmpty

    @model_validator(mode="after")
    def validate_coordinate(self) -> "HoldingSlotBinding":
        acode_slots = {
            "previous_count": "SUM_BMT_CNT", "previous_ratio": "SUM_BMT_RT",
            "current_count": "SUM_TMT_CNT", "current_ratio": "SUM_TMT_RT",
            "delta_count": "MDF_STK_CNT", "delta_ratio": "MDF_STK_RT",
            "report_reason": "SUM_CHN_RWN", "holding_purpose": "HLD_OBJ_DTL",
            "change_method": "CHN_HOW", "change_reason": "CHN_RSN",
        }
        if self.binding_status == "limited":
            if (self.slot != "restricted_personal_data" or self.acode is not None
                    or self.value_kind != "text" or not self.restricted_types):
                raise ValueError("제한된 holding 개인정보 binding이 잘못되었습니다")
        elif self.slot == "restricted_personal_data":
            raise ValueError("실행 가능한 holding slot이 개인정보 제한 slot일 수 없습니다")
        elif self.binding_status == "qualified" and not self.restricted_types:
            raise ValueError("qualified holding slot에는 제한 개인정보 유형이 필요합니다")
        elif self.binding_status == "executable" and self.restricted_types:
            raise ValueError("executable holding slot에는 제한 개인정보가 없어야 합니다")
        party_slots = {
            "previous_count", "previous_ratio", "current_count",
            "current_ratio", "delta_count", "delta_ratio",
        }
        if (self.party_name is None) != (self.party_proof is None):
            raise ValueError("holding party 이름과 proof는 함께 필요합니다")
        if self.party_name is not None:
            if self.slot not in party_slots or self.binding_status == "limited":
                raise ValueError("holding party qualifier가 허용되지 않는 slot입니다")
        expected = acode_slots.get(self.slot)
        if expected is not None and self.acode != expected:
            raise ValueError("holding slot/ACODE binding이 다릅니다")
        if expected is None and self.acode is not None:
            raise ValueError("holding metadata/party slot에는 ACODE를 넣지 않습니다")
        if re.fullmatch(
                rf"source-field:item-[1-9][0-9]*:{self.source_field_index}",
                self.proof_ref) is None:
            raise ValueError("holding slot proof_ref가 source field와 다릅니다")
        return self


class HoldingDisclosureResolution(_StrictFrozenModel):
    """Exact issuer/reporter/effective-filing authority for one holding read."""

    kind: Literal["holding_disclosure"] = "holding_disclosure"
    issuer_corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    issuer_corp_name: NonEmpty
    filer_name: NonEmpty
    document_id: NonEmpty
    receipt_no: ReceiptStamp
    report_type: Literal["general", "short"]
    as_of: DateStamp
    document_proof: ResolutionSourceProof
    slot_bindings: list[HoldingSlotBinding] = Field(min_length=1)
    privacy_notice_required: bool = False
    lineage_status: Literal["complete", "root_missing"] = "complete"
    selection_basis: Literal[
        "date_scoped", "explicit_latest", "reporter_latest_default",
        "selected_receipt",
    ] = "reporter_latest_default"

    @model_validator(mode="after")
    def validate_holding(self) -> "HoldingDisclosureResolution":
        _valid_date_stamp(self.as_of, "holding as_of")
        _valid_date_stamp(self.receipt_no[:8], "holding receipt date")
        if self.document_id != f"holding_{self.receipt_no}":
            raise ValueError("holding document_id/receipt가 다릅니다")
        if self.document_proof.source_receipt != self.receipt_no:
            raise ValueError("holding document proof receipt가 다릅니다")
        indexes = [row.source_field_index for row in self.slot_bindings]
        if indexes != list(range(len(indexes))):
            raise ValueError("holding slot binding은 질문 field 순서여야 합니다")
        _unique([
            (row.slot, row.party_name) for row in self.slot_bindings
            if row.binding_status != "limited"
        ], "holding executable slots")
        if any(row.party_proof is not None
               and row.party_proof.source_receipt != self.receipt_no
               for row in self.slot_bindings):
            raise ValueError("holding party proof receipt가 선택 문서와 다릅니다")
        if (any(row.binding_status != "executable" for row in self.slot_bindings)
                and not self.privacy_notice_required):
            raise ValueError("제한된 holding field에는 개인정보 제외 고지가 필요합니다")
        return self


class DocumentVersionHistoryResolution(_StrictFrozenModel):
    """One annual filing's correction lineage, without a legacy projection."""

    kind: Literal["document_version_history"] = "document_version_history"
    corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    corp_name: NonEmpty
    base_year: int = Field(ge=2000, le=2100)
    root_receipt: ReceiptStamp
    latest_receipt: ReceiptStamp
    lineage_receipts: list[ReceiptStamp] = Field(min_length=1)
    selector_proof: ResolutionSourceProof
    lineage_proof_ref: NonEmpty

    @model_validator(mode="after")
    def validate_lineage(self) -> "DocumentVersionHistoryResolution":
        _unique(list(self.lineage_receipts), "document version lineage receipts")
        if self.root_receipt not in self.lineage_receipts or self.latest_receipt not in self.lineage_receipts:
            raise ValueError("document version root/latest는 lineage에 있어야 합니다")
        if self.selector_proof.source_receipt != self.root_receipt:
            raise ValueError("document version selector proof가 root receipt와 다릅니다")
        return self


class SelectedEventResolution(_StrictFrozenModel):
    """One canonical event selected by a user-visible event receipt.

    ``root_receipt`` keeps the canonical event identity.  ``selector_proof``
    may instead cite an exact correction/termination observation named by the
    user.  The compiler emits the canonical key plus that existing public
    seed coordinate, so Stage2 can read the requested observation without
    pretending it is the event root.
    """

    kind: Literal["selected_event"] = "selected_event"
    corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    corp_name: NonEmpty
    entity_surface: NonEmpty
    event_key: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    root_receipt: ReceiptStamp
    selector_proof: ResolutionSourceProof
    operation: Literal["status", "timeline"] = "status"
    lineage_missing_root_date: DateStamp | None = None
    # A status task has an intrinsic result for each observation point.  Keep
    # the points on the resolver authority rather than inventing a disclosure
    # field-output for ``상태``.  The empty default preserves the older
    # one-field selected-event path, whose observation is the corpus cutoff.
    timepoints: list[DateStamp] = Field(
        default_factory=list,
        exclude_if=lambda values: not values,
    )
    public_selector: PublicEventSelectorFacets | None = None

    @model_validator(mode="after")
    def validate_selected_event(self) -> "SelectedEventResolution":
        _valid_date_stamp(self.root_receipt[:8], "selected event root receipt date")
        _valid_date_stamp(
            self.selector_proof.source_receipt[:8],
            "selected event selector receipt date")
        _unique(list(self.timepoints), "selected event timepoints")
        for timepoint in self.timepoints:
            _valid_date_stamp(timepoint, "selected event timepoint")
        if self.lineage_missing_root_date is not None:
            _valid_date_stamp(self.lineage_missing_root_date,
                              "selected event missing root date")
        if self.operation == "timeline":
            missing_root_timeline = (
                len(self.timepoints) == 1
                and self.lineage_missing_root_date is not None
            )
            complete_timeline = (
                len(self.timepoints) in {2, 3}
                and self.lineage_missing_root_date is None
            )
            if not (missing_root_timeline or complete_timeline):
                raise ValueError(
                    "timeline selected event에는 missing-root 기준시점 하나 또는 "
                    "완전한 시작/종료/기준 시점 2~3개가 필요합니다")
        elif self.lineage_missing_root_date is not None:
            raise ValueError("status selected event에는 lineage missing root를 넣지 않습니다")
        if (self.public_selector is not None
                and self.public_selector.source_receipts() != {self.root_receipt}):
            raise ValueError("selected-event public selector는 root receipt 증명이어야 합니다")
        return self


class EventCollectionMember(_StrictFrozenModel):
    """One exact canonical identity admitted to a list result.

    ``matched_receipts`` are the observations that made this identity satisfy
    the collection predicate.  They are intentionally distinct from the
    root receipt: a lifecycle query can select a termination observation while
    preserving its original contract identity.
    """

    event_key: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    root_receipt: ReceiptStamp
    matched_receipts: list[ReceiptStamp] = Field(min_length=1)
    proof_refs: list[NonEmpty] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_member(self) -> "EventCollectionMember":
        _unique(list(self.matched_receipts), "event collection matched receipt")
        if self.matched_receipts != sorted(self.matched_receipts):
            raise ValueError("event collection matched receipts는 정렬되어야 합니다")
        _unique(list(self.proof_refs), "event collection member proof")
        required = {
            f"canonical:event:{self.event_key}:{receipt}"
            for receipt in self.matched_receipts
        }
        if not required.issubset(set(self.proof_refs)):
            raise ValueError(
                "event collection member에는 모든 matched receipt의 canonical event proof가 필요합니다")
        return self


class EventCollectionResolution(_StrictFrozenModel):
    """Canonical event identities for a multi-row event answer.

    This is a source coordinate only.  It carries neither event values nor a
    fixture-shaped result; each member is lowered into an exact event-key task
    and Stage2 performs the field reads.
    """

    kind: Literal["event_collection"] = "event_collection"
    corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    corp_name: NonEmpty
    as_of: DateStamp
    event_from: DateStamp | None = None
    event_to: DateStamp | None = None
    requires_termination: bool = False
    root_contracts_with_confirmed_termination: bool = False
    event_type: NonEmpty | None = None
    counterparty: NonEmpty | None = None
    keywords: list[NonEmpty] = Field(default_factory=list)
    public_task_kind: Literal["event", "disclosure", "correction"] = "event"
    availability_query: bool = False
    requested_slots: list[NonEmpty] = Field(min_length=1)
    events: list[EventCollectionMember] = Field(default_factory=list)
    # A closed canonical scan may prove an empty result.  This proof binds the
    # executable list task without fabricating an event identity.
    collection_proof_ref: NonEmpty | None = None
    # 이슈 #59 2단계 — 거래소 사건 collection 을 단일 최댓값으로 낮추는
    # 지시.  값 자체는 여전히 compiler 가 아니라 Stage2 가 검증된 Field
    # evidence 로 정한다; 이 필드는 "어느 slot 을 argmax 로 줄일지"만
    # 싣는다(예: "계약금액"). optional — 기존 list 요청은 그대로 None.
    argmax_slot: NonEmpty | None = None
    # 이슈 #124 — argmax_slot 이 줄이는 극값의 방향. optional — argmax_slot이
    # None이면 의미가 없으므로 "maximum" 그대로.
    argmax_direction: Literal["maximum", "minimum"] = "maximum"

    @model_validator(mode="after")
    def validate_collection(self) -> "EventCollectionResolution":
        _valid_date_stamp(self.as_of, "event collection as_of")
        if self.event_from is not None:
            _valid_date_stamp(self.event_from, "event collection event_from")
        if self.event_to is not None:
            _valid_date_stamp(self.event_to, "event collection event_to")
        if ((self.event_from is None) != (self.event_to is None)
                or (self.event_from is not None and self.event_from > self.event_to)
                or (self.event_to is not None and self.event_to > self.as_of)):
            raise ValueError("event collection event range가 as_of와 맞지 않습니다")
        _unique(list(self.requested_slots), "event collection requested slots")
        _unique(list(self.keywords), "event collection keywords")
        if not any((self.event_type, self.counterparty, self.keywords,
                    self.event_from, self.event_to)):
            raise ValueError("event collection에는 public selector 조건이 필요합니다")
        if self.root_contracts_with_confirmed_termination and not self.requires_termination:
            raise ValueError("root termination collection에는 termination predicate가 필요합니다")
        if self.availability_query and self.public_task_kind != "disclosure":
            raise ValueError(
                "availability event collection은 disclosure task로 공개되어야 합니다")
        if (self.public_task_kind == "correction"
                and (self.event_from is not None or self.event_to is not None)):
            raise ValueError(
                "correction collection의 event date는 root exact 역할과 혼용할 수 없습니다")
        keys = [member.event_key for member in self.events]
        _unique(keys, "event collection event keys")
        if keys != sorted(keys):
            raise ValueError("event collection members는 event_key 순서여야 합니다")
        if not self.events and self.collection_proof_ref is None:
            raise ValueError(
                "empty event collection에는 canonical scan proof가 필요합니다")
        if self.argmax_slot is not None and self.argmax_slot not in self.requested_slots:
            raise ValueError(
                "event collection argmax_slot은 requested_slots에 있어야 합니다")
        if self.argmax_slot is None and self.argmax_direction != "maximum":
            raise ValueError(
                "event collection argmax_direction은 argmax_slot에만 허용됩니다")
        return self


class EventAmountObservationProof(_StrictFrozenModel):
    """One canonical numeric KRW field that existed at an observation point."""

    timepoint: DateStamp
    source_receipt: ReceiptStamp
    evidence_id: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    normalized_value: Annotated[
        str, StringConstraints(pattern=r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")]
    currency: Literal["KRW"] = "KRW"
    unit: Literal["원"] = "원"
    scale: Literal[1] = 1
    proof: ResolutionSourceProof

    @model_validator(mode="after")
    def validate_observation(self) -> "EventAmountObservationProof":
        _valid_date_stamp(self.timepoint, "event amount observation timepoint")
        _valid_date_stamp(
            self.source_receipt[:8], "event amount observation receipt date")
        if self.source_receipt[:8] > self.timepoint:
            raise ValueError("event amount observation receipt가 기준시점보다 늦습니다")
        if (self.proof.source_receipt != self.source_receipt
                or self.proof.proof_ref
                != f"canonical:field:{self.evidence_id}"):
            raise ValueError("event amount observation proof가 field와 다릅니다")
        return self


class EventAmountChangeResolution(_StrictFrozenModel):
    """One selected canonical contract observed at two explicit timepoints.

    The user supplies the missing company, observation points, amount axis and
    canonical root receipt through the clarification protocol.  Those values
    remain resolver authority; the compiler owns the two reads and the
    ``difference`` derivation.
    """

    kind: Literal["event_amount_change"] = "event_amount_change"
    corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    corp_name: NonEmpty
    entity_surface: NonEmpty
    event_key: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    root_receipt: ReceiptStamp
    selector_proof: ResolutionSourceProof
    timepoints: list[DateStamp] = Field(min_length=2, max_length=2)
    requested_slot: Literal["계약금액", "해지금액"]
    observations: list[EventAmountObservationProof] = Field(
        min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_event_amount_change(self) -> "EventAmountChangeResolution":
        _valid_date_stamp(
            self.root_receipt[:8], "event amount change root receipt date")
        if self.selector_proof.source_receipt != self.root_receipt:
            raise ValueError(
                "event amount change selector proof가 root receipt와 다릅니다")
        _unique(list(self.timepoints), "event amount change timepoints")
        for timepoint in self.timepoints:
            _valid_date_stamp(timepoint, "event amount change timepoint")
        if self.timepoints != sorted(self.timepoints):
            raise ValueError("event amount change timepoints는 시간순이어야 합니다")
        if self.root_receipt[:8] > self.timepoints[0]:
            raise ValueError("event amount change event가 이전 시점에 존재하지 않습니다")
        if [row.timepoint for row in self.observations] != self.timepoints:
            raise ValueError("event amount observations가 요청 시점과 다릅니다")
        typings = {(row.currency, row.unit, row.scale)
                   for row in self.observations}
        if len(typings) != 1:
            raise ValueError("event amount observation money typing이 다릅니다")
        return self


class LifecycleCompositeAttribute(_StrictFrozenModel):
    """One exact canonical field participating in an event lifecycle answer."""

    kind: Literal[
        "contract_amount", "termination_amount", "termination_reason",
        "effectiveness_condition",
    ]
    source_receipt: ReceiptStamp
    evidence_id: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    path: NonEmpty
    locator: NonEmpty


class LifecycleCompositeResolution(_StrictFrozenModel):
    """One proven event lifecycle, without values or fixture coordinates."""

    kind: Literal["event_lifecycle_composite"] = "event_lifecycle_composite"
    corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    corp_name: NonEmpty
    as_of: DateStamp
    event_key: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    root_receipt: ReceiptStamp
    selector_proof: ResolutionSourceProof
    event_proofs: list[ResolutionSourceProof] = Field(min_length=1)
    status_timepoints: list[DateStamp] = Field(default_factory=list)
    termination_receipts: list[ReceiptStamp] = Field(default_factory=list)
    correction_receipts: list[ReceiptStamp] = Field(default_factory=list)
    attributes: list[LifecycleCompositeAttribute] = Field(default_factory=list)
    public_selector: PublicEventSelectorFacets | None = None

    @model_validator(mode="after")
    def validate_lifecycle_composite(self) -> "LifecycleCompositeResolution":
        _valid_date_stamp(self.as_of, "lifecycle as_of")
        _valid_date_stamp(self.root_receipt[:8], "lifecycle root receipt date")
        if self.selector_proof.source_receipt != self.root_receipt:
            raise ValueError("lifecycle selector proof가 root receipt와 다릅니다")
        proofs = {proof.source_receipt for proof in self.event_proofs}
        if self.root_receipt not in proofs:
            raise ValueError("lifecycle event proofs에는 root receipt가 필요합니다")
        _unique([proof.proof_ref for proof in self.event_proofs], "lifecycle event proof")
        for timepoint in self.status_timepoints:
            _valid_date_stamp(timepoint, "lifecycle status timepoint")
            if timepoint > self.as_of:
                raise ValueError("lifecycle status timepoint가 as_of 이후입니다")
        _unique(list(self.status_timepoints), "lifecycle status timepoint")
        if self.status_timepoints != sorted(self.status_timepoints):
            raise ValueError("lifecycle status timepoint는 시간순이어야 합니다")
        for receipt in [*self.termination_receipts, *self.correction_receipts]:
            if receipt not in proofs:
                raise ValueError("lifecycle observation receipt는 event proof가 필요합니다")
        _unique(list(self.termination_receipts), "lifecycle termination receipt")
        _unique(list(self.correction_receipts), "lifecycle correction receipt")
        _unique([row.kind for row in self.attributes], "lifecycle attribute kind")
        for attribute in self.attributes:
            if attribute.source_receipt not in proofs:
                raise ValueError("lifecycle attribute는 event proof receipt에 있어야 합니다")
        if (self.public_selector is not None
                and not self.public_selector.source_receipts().issubset(proofs)):
            raise ValueError("lifecycle public selector에는 event proof receipt가 필요합니다")
        if not (self.status_timepoints or self.correction_receipts or self.attributes):
            raise ValueError("lifecycle resolution에는 상태 시점, 정정 또는 속성이 필요합니다")
        return self


class CorrectionLineageChange(_StrictFrozenModel):
    """One source-proven changed coordinate in a correction lineage."""

    path: NonEmpty
    before_proof: ResolutionSourceProof | None = None
    after_proof: ResolutionSourceProof | None = None

    @model_validator(mode="after")
    def validate_change(self) -> "CorrectionLineageChange":
        if self.before_proof is None and self.after_proof is None:
            raise ValueError("correction lineage change에는 source proof가 필요합니다")
        return self


class CorrectionLineageDateRoles(_StrictFrozenModel):
    """Internal date sidecar; public v0.4 fields keep their original schema."""

    root_observed_at: DateStamp | None = None
    correction_from: DateStamp | None = None
    correction_to: DateStamp | None = None
    as_of: DateStamp
    range_requested: bool = False

    @model_validator(mode="after")
    def validate_roles(self) -> "CorrectionLineageDateRoles":
        for field_name in (
                "root_observed_at", "correction_from", "correction_to", "as_of"):
            value = getattr(self, field_name)
            if value is not None:
                _valid_date_stamp(value, f"correction lineage {field_name}")
        if (self.correction_from and self.correction_to
                and self.correction_from > self.correction_to):
            raise ValueError("correction lineage correction range 순서가 다릅니다")
        if self.correction_to and self.correction_to > self.as_of:
            raise ValueError("correction lineage correction_to가 as_of 이후입니다")
        return self


class CorrectionLineageStep(_StrictFrozenModel):
    correction_receipt: ReceiptStamp
    correction_date: DateStamp
    previous_observation_receipt: ReceiptStamp | None = None
    event_proofs: list[ResolutionSourceProof] = Field(min_length=1)
    changes: list[CorrectionLineageChange] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_step(self) -> "CorrectionLineageStep":
        if self.correction_receipt[:8] != self.correction_date:
            raise ValueError("correction sequence receipt/date가 다릅니다")
        if any(proof.source_receipt != self.correction_receipt
               for proof in self.event_proofs):
            raise ValueError("correction sequence event proof receipt가 다릅니다")
        return self


class CorrectionLineageResolution(_StrictFrozenModel):
    """Canonical original-to-correction lineage, with no answer values.

    ``answer_role`` lets one history lineage close the amount and its stated
    change reason without inventing a second event identity.  It is a role of
    the question topology, never a question-id dispatch key.
    """

    kind: Literal["correction_lineage"] = "correction_lineage"
    operation: Literal["diff", "history"]
    answer_role: Literal["diff", "amount", "reason"]
    corp_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{8}$")]
    corp_name: NonEmpty
    event_key: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    root_receipt: ReceiptStamp
    correction_receipt: ReceiptStamp
    correction_date: DateStamp
    # 일부 거래소 정정 문서는 제공 코퍼스에서 정정본 자체가 lineage의 첫
    # 관측이다. 이 경우에만 root==correction을 명시적으로 허용한다. 플래그가
    # 없으면 기존의 엄격한 선후 관계를 그대로 지킨다.
    source_root_missing: bool = Field(
        default=False, exclude_if=lambda value: not value)
    root_selector_proof: ResolutionSourceProof
    correction_selector_proof: ResolutionSourceProof
    event_proofs: list[ResolutionSourceProof] = Field(min_length=1)
    changes: list[CorrectionLineageChange] = Field(min_length=1)
    counterparty: NonEmpty | None = None
    product_keywords: list[NonEmpty] = Field(default_factory=list)
    date_roles: CorrectionLineageDateRoles | None = Field(
        default=None, exclude_if=lambda value: value is None)
    sequence: list[CorrectionLineageStep] = Field(
        default_factory=list, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def validate_lineage(self) -> "CorrectionLineageResolution":
        _valid_date_stamp(self.root_receipt[:8], "correction lineage root date")
        _valid_date_stamp(self.correction_receipt[:8], "correction lineage correction date")
        _valid_date_stamp(self.correction_date, "correction lineage correction date")
        if (self.root_receipt > self.correction_receipt
                or (self.root_receipt == self.correction_receipt
                    and not self.source_root_missing)):
            raise ValueError("correction lineage root는 correction보다 앞서야 합니다")
        if (not self.source_root_missing
                and self.root_receipt == self.correction_receipt):
            raise ValueError("correction lineage source-root-missing 표지가 좌표와 다릅니다")
        if self.correction_receipt[:8] != self.correction_date:
            raise ValueError("correction lineage correction date가 receipt와 다릅니다")
        if (self.root_selector_proof.source_receipt != self.root_receipt
                or self.correction_selector_proof.source_receipt
                != self.correction_receipt):
            raise ValueError("correction lineage selector proof가 receipt와 다릅니다")
        if (not self.sequence
                and any(proof.source_receipt != self.correction_receipt
                        for proof in self.event_proofs)):
            raise ValueError("correction lineage event proof는 correction receipt에 결속돼야 합니다")
        if self.operation == "diff" and self.answer_role != "diff":
            raise ValueError("correction diff는 diff answer role이 필요합니다")
        if self.operation == "history" and self.answer_role not in {
                "diff", "amount", "reason"}:
            raise ValueError("correction history answer role이 다릅니다")
        if self.sequence:
            if (self.sequence[-1].correction_receipt != self.correction_receipt
                    or self.sequence[-1].correction_date != self.correction_date):
                raise ValueError("correction sequence final coordinate가 다릅니다")
            sequence_event_proofs = [
                proof for step in self.sequence for proof in step.event_proofs]
            sequence_changes = [
                change for step in self.sequence for change in step.changes]
            if (sequence_event_proofs != self.event_proofs
                    or sequence_changes != self.changes):
                raise ValueError("correction sequence aggregate가 다릅니다")
            coordinates = [
                (step.correction_date, step.correction_receipt)
                for step in self.sequence]
            if coordinates != sorted(coordinates) or len(coordinates) != len(set(coordinates)):
                raise ValueError("correction sequence가 시간순·고유하지 않습니다")
            if self.date_roles is None:
                raise ValueError("correction sequence에는 date role이 필요합니다")
        _unique(list(self.product_keywords), "correction lineage product keywords")
        return self


# ``kind`` is the closed discriminator.  Keeping the existing single-fact
# branch first preserves the serialized shape and digests of G-A-001/R-A-002.
ResolutionPayload = Annotated[
    FinancialResolution | FinancialComparisonResolution
    | DocumentFactComparisonResolution | DocumentReasonEvidenceResolution
    | DocumentAttributeEvidenceResolution
    | PeriodicNarrativeComparisonResolution | NarrativeMatrixResolution
    | PeriodicDocumentNarrativeResolution
    | SameDayDocumentCandidatesResolution
    | TerminationReportedStatusResolution
    | DocumentCollectionResolution | HoldingDisclosureResolution
    | DocumentVersionHistoryResolution
    | SelectedEventResolution | EventCollectionResolution
    | EventAmountChangeResolution | LifecycleCompositeResolution
    | CorrectionLineageResolution,
    Field(discriminator="kind"),
]


class ResolvedItem(_StrictFrozenModel):
    item_id: NonEmpty
    target_surface: NonEmpty
    projection_mode: Literal["named_fields", "whole_target"] = "named_fields"
    resolution: ResolutionPayload
    field_proofs: list[ResolutionFieldProof] = Field(default_factory=list)
    applied_defaults: list[AppliedDefault] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_fields(self) -> "ResolvedItem":
        indexes = [row.source_field_index for row in self.field_proofs]
        _unique(indexes, "source field indexes")
        if indexes != sorted(indexes):
            raise ValueError("source field proofs는 질문 순서로 정렬되어야 합니다")
        _unique([row.surface for row in self.field_proofs], "field surfaces")
        if self.projection_mode == "named_fields" and not self.field_proofs:
            raise ValueError("named_fields resolution에는 field proof가 필요합니다")
        if self.projection_mode == "whole_target" and self.field_proofs:
            raise ValueError("whole_target resolution에는 field proof가 없어야 합니다")
        return self


class ResolutionPremiseProof(_StrictFrozenModel):
    premise_id: NonEmpty
    proof_refs: list[NonEmpty] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_refs(self) -> "ResolutionPremiseProof":
        _unique(list(self.proof_refs), "premise proof_refs")
        return self


class AuthoritativeResolution(_StrictFrozenModel):
    """Strict item-scoped resolver output; root identifiers are not accepted."""

    schema_version: Literal[RESOLUTION_VERSION] = RESOLUTION_VERSION
    question_id: NonEmpty
    source_intent_digest: Digest
    canonical_build_id: BuildId
    resolver_version: Annotated[str, StringConstraints(
        min_length=3, pattern=r"^[A-Za-z0-9_.-]+/[0-9]+(?:\.[0-9]+)*$")]
    reference_date: date
    corpus_cutoff: DateStamp
    items: list[ResolvedItem] = Field(min_length=1)
    premise_proofs: list[ResolutionPremiseProof] = Field(default_factory=list)
    resolution_digest: Digest

    @classmethod
    def compute_digest(cls, payload: Mapping[str, Any]) -> str:
        body = dict(payload)
        body.pop("resolution_digest", None)
        return canonical_sha256(body)

    @classmethod
    def create(cls, **payload: Any) -> "AuthoritativeResolution":
        body = dict(payload)
        body.setdefault("schema_version", RESOLUTION_VERSION)
        body.setdefault("premise_proofs", [])
        body["items"] = [
            _strict_json_model(ResolvedItem, row)
            for row in body.get("items", [])
        ]
        body["premise_proofs"] = [
            _strict_json_model(ResolutionPremiseProof, row)
            for row in body["premise_proofs"]
        ]
        digest_body = cls.model_construct(**body).model_dump(
            mode="json", warnings=False)
        body["resolution_digest"] = cls.compute_digest(digest_body)
        return _strict_json_model(cls, body)

    @model_validator(mode="after")
    def validate_resolution(self) -> "AuthoritativeResolution":
        _valid_date_stamp(self.corpus_cutoff, "corpus_cutoff")
        _unique([row.item_id for row in self.items], "resolved items")
        _unique([row.premise_id for row in self.premise_proofs], "premise_proofs")
        if self.resolution_digest != self.compute_digest(
                self.model_dump(mode="json")):
            raise ValueError("resolution_digest가 일치하지 않습니다")
        return self


class ExecutionAnswerRoot(_StrictFrozenModel):
    root_id: Annotated[str, StringConstraints(pattern=r"^output-[1-9][0-9]*$")]
    item_id: NonEmpty
    projection_mode: Literal["named_fields", "whole_target"] = "named_fields"
    field_id: NonEmpty | None = Field(
        default=None, exclude_if=lambda value: value is None)
    whole_target_id: WholeTargetId | None = Field(
        default=None, exclude_if=lambda value: value is None)
    plan_output_id: NonEmpty | None = None
    plan_task_id: NonEmpty | None = None
    proof_ref: NonEmpty

    @model_validator(mode="after")
    def exactly_one_plan_ref(self) -> "ExecutionAnswerRoot":
        if (self.plan_output_id is None) == (self.plan_task_id is None):
            raise ValueError("answer root는 plan output/task ref 중 정확히 하나여야 합니다")
        if self.projection_mode == "named_fields":
            if self.field_id is None or self.whole_target_id is not None:
                raise ValueError("named_fields answer root에는 field_id만 필요합니다")
            if self.field_id.startswith("whole-target-"):
                raise ValueError(
                    "named_fields answer root는 whole-target namespace를 사용할 수 없습니다")
        elif self.whole_target_id is None or self.field_id is not None:
            raise ValueError("whole_target answer root에는 whole_target_id만 필요합니다")
        return self

    @property
    def projection_binding_id(self) -> str:
        value = self.field_id if self.projection_mode == "named_fields" \
            else self.whole_target_id
        assert value is not None
        return value


AnswerRootId = Annotated[str, StringConstraints(
    pattern=r"^output-[1-9][0-9]*$")]


PlanRootId = Annotated[str, StringConstraints(
    pattern=r"^plan-root-[1-9][0-9]*$")]


class ExecutionPlanValueRoot(_StrictFrozenModel):
    """Typed registry entry for a plan-internal proof value.

    Plan-value roots are deliberately separate from ``ExecutionAnswerRoot``.
    They can prove an intermediate value (including a boolean predicate), and
    may mirror a user answer root only through an exact bidirectional link.
    """

    plan_root_id: PlanRootId
    plan_output_id: NonEmpty
    value_kind: Literal["money", "boolean", "text"]
    producer_task_id: NonEmpty | None = None
    producer_derivation_output_id: NonEmpty | None = None
    answer_mirror_root_id: AnswerRootId | None = Field(
        default=None, exclude_if=lambda root_id: root_id is None)

    @model_validator(mode="after")
    def exactly_one_producer(self) -> "ExecutionPlanValueRoot":
        if (self.producer_task_id is None) == (
                self.producer_derivation_output_id is None):
            raise ValueError(
                "plan value root는 task 또는 derivation producer 중 정확히 하나여야 합니다")
        return self


class ExecutionSupportRoot(_StrictFrozenModel):
    support_id: NonEmpty
    kind: NonEmpty
    item_id: NonEmpty
    projection_mode: Literal["named_fields", "whole_target"] = "named_fields"
    field_id: NonEmpty | None = Field(
        default=None, exclude_if=lambda value: value is None)
    whole_target_id: WholeTargetId | None = Field(
        default=None, exclude_if=lambda value: value is None)
    root_id: Annotated[str, StringConstraints(pattern=r"^output-[1-9][0-9]*$")]
    plan_root_ids: list[PlanRootId] = Field(
        default_factory=list, exclude_if=lambda refs: not refs)

    @model_validator(mode="after")
    def unique_plan_roots(self) -> "ExecutionSupportRoot":
        _unique(list(self.plan_root_ids), "support plan_root_ids")
        if self.projection_mode == "named_fields":
            if self.field_id is None or self.whole_target_id is not None:
                raise ValueError("named_fields support root에는 field_id만 필요합니다")
            if self.field_id.startswith("whole-target-"):
                raise ValueError(
                    "named_fields support root는 whole-target namespace를 사용할 수 없습니다")
        elif self.whole_target_id is None or self.field_id is not None:
            raise ValueError("whole_target support root에는 whole_target_id만 필요합니다")
        return self

    @property
    def projection_binding_id(self) -> str:
        value = self.field_id if self.projection_mode == "named_fields" \
            else self.whole_target_id
        assert value is not None
        return value


class ExecutionPremiseRoot(_StrictFrozenModel):
    premise_id: NonEmpty
    root_ids: list[NonEmpty] = Field(default_factory=list)
    task_refs: list[NonEmpty] = Field(default_factory=list)
    plan_root_ids: list[PlanRootId] = Field(
        default_factory=list, exclude_if=lambda refs: not refs)

    @model_validator(mode="after")
    def unique_roots(self) -> "ExecutionPremiseRoot":
        _unique(list(self.root_ids), "premise root_ids")
        _unique(list(self.task_refs), "premise task_refs")
        _unique(list(self.plan_root_ids), "premise plan_root_ids")
        namespaces = {
            "answer": set(self.root_ids),
            "plan": set(self.plan_root_ids),
            "task": set(self.task_refs),
        }
        for left, left_values in namespaces.items():
            for right, right_values in namespaces.items():
                if left >= right:
                    continue
                if left_values & right_values:
                    raise ValueError(
                        "premise answer/plan/task ref namespace가 겹칩니다")
        if not self.root_ids and not self.task_refs and not self.plan_root_ids:
            raise ValueError("premise binding은 answer/plan/task ref가 필요합니다")
        return self


def _plan_namespaces(
        plan: ResolvedQueryPlan,
        ) -> tuple[list[str], list[str], list[str]]:
    task_ids = [task.task_id for task in plan.tasks]
    outputs: list[str] = []
    for task in plan.tasks:
        if task.kind == "financial":
            outputs.extend(row.output_id for row in task.facts)
        else:
            output_id = getattr(task, "output_id", None)
            if output_id:
                outputs.append(output_id)
            outputs.extend(row.output_id for row in getattr(task, "field_outputs", []))
    outputs.extend(row.output_id for row in plan.derivations)
    if len(outputs) != len(set(outputs)):
        raise ValueError("resolved plan output namespace가 중복되었습니다")
    premise_ids = [claim.claim_id for claim in plan.premise_claims]
    return task_ids, outputs, premise_ids


class ExecutionPlan(_StrictFrozenModel):
    schema_version: Literal[EXECUTION_PLAN_VERSION] = EXECUTION_PLAN_VERSION
    source_intent_digest: Digest
    resolution_digest: Digest
    canonical_build_id: BuildId
    resolver_version: Annotated[str, StringConstraints(
        min_length=3, pattern=r"^[A-Za-z0-9_.-]+/[0-9]+(?:\.[0-9]+)*$")]
    resolved_plan: ResolvedQueryPlan
    applied_defaults: list[AppliedDefault] = Field(default_factory=list)
    answer_roots: list[ExecutionAnswerRoot] = Field(min_length=1)
    support_roots: list[ExecutionSupportRoot] = Field(default_factory=list)
    premise_roots: list[ExecutionPremiseRoot] = Field(default_factory=list)
    plan_value_roots: list[ExecutionPlanValueRoot] = Field(
        default_factory=list, exclude_if=lambda roots: not roots)
    execution_plan_digest: Digest

    @field_validator("resolved_plan", mode="before")
    @classmethod
    def strict_resolved_plan(cls, value: Any) -> ResolvedQueryPlan:
        return _strict_json_model(ResolvedQueryPlan, value)  # type: ignore[return-value]

    @classmethod
    def compute_digest(cls, payload: Mapping[str, Any]) -> str:
        body = dict(payload)
        body.pop("execution_plan_digest", None)
        return canonical_sha256(body)

    @classmethod
    def create(cls, **payload: Any) -> "ExecutionPlan":
        body = dict(payload)
        body.setdefault("schema_version", EXECUTION_PLAN_VERSION)
        plan_value = body.get("resolved_plan")
        body["resolved_plan"] = _strict_json_model(
            ResolvedQueryPlan, plan_value)
        body.setdefault("applied_defaults", [])
        body.setdefault("support_roots", [])
        body.setdefault("premise_roots", [])
        body.setdefault("plan_value_roots", [])
        body["applied_defaults"] = [
            _strict_json_model(AppliedDefault, row)
            for row in body["applied_defaults"]
        ]
        body["answer_roots"] = [
            _strict_json_model(ExecutionAnswerRoot, row)
            for row in body.get("answer_roots", [])
        ]
        body["support_roots"] = [
            _strict_json_model(ExecutionSupportRoot, row)
            for row in body["support_roots"]
        ]
        body["premise_roots"] = [
            _strict_json_model(ExecutionPremiseRoot, row)
            for row in body["premise_roots"]
        ]
        body["plan_value_roots"] = [
            _strict_json_model(ExecutionPlanValueRoot, row)
            for row in body["plan_value_roots"]
        ]
        digest_body = cls.model_construct(**body).model_dump(
            mode="json", warnings=False)
        body["execution_plan_digest"] = cls.compute_digest(digest_body)
        return _strict_json_model(cls, body)

    @model_validator(mode="after")
    def validate_plan(self) -> "ExecutionPlan":
        task_ids, outputs, premise_ids = _plan_namespaces(self.resolved_plan)
        _unique([row.root_id for row in self.answer_roots], "answer_roots")
        _unique([row.support_id for row in self.support_roots], "support_roots")
        _unique([row.premise_id for row in self.premise_roots], "premise_roots")
        _unique([row.plan_root_id for row in self.plan_value_roots],
                "plan_value_roots")
        _unique([row.plan_output_id for row in self.plan_value_roots],
                "plan value root outputs")
        expected_plan_root_ids = [
            f"plan-root-{index}"
            for index in range(1, len(self.plan_value_roots) + 1)
        ]
        actual_plan_root_ids = [
            row.plan_root_id for row in self.plan_value_roots]
        if actual_plan_root_ids != expected_plan_root_ids:
            raise ValueError(
                "plan value root ID는 serialized order의 plan-root-1..N이어야 합니다")
        root_ids = {row.root_id for row in self.answer_roots}
        plan_root_ids = {row.plan_root_id for row in self.plan_value_roots}
        answer_by_root = {row.root_id: row for row in self.answer_roots}
        value_root_by_output = {
            row.plan_output_id: row for row in self.plan_value_roots}
        mirror_root_ids = [
            row.answer_mirror_root_id for row in self.plan_value_roots
            if row.answer_mirror_root_id is not None
        ]
        _unique(mirror_root_ids, "plan value root answer mirrors")
        for value_root in self.plan_value_roots:
            mirror_root_id = value_root.answer_mirror_root_id
            if mirror_root_id is None:
                continue
            answer_root = answer_by_root.get(mirror_root_id)
            if answer_root is None:
                raise ValueError(
                    "plan value root answer mirror가 live answer root가 아닙니다")
            if answer_root.plan_output_id != value_root.plan_output_id:
                raise ValueError(
                    "plan value root answer mirror의 output binding이 다릅니다")
        for answer_root in self.answer_roots:
            if answer_root.plan_output_id is None:
                continue
            value_root = value_root_by_output.get(answer_root.plan_output_id)
            if value_root is None:
                continue
            if value_root.answer_mirror_root_id != answer_root.root_id:
                raise ValueError(
                    "registered plan output answer root는 exact mirror가 필요합니다")
        derivation_outputs = {
            derivation.output_id for derivation in self.resolved_plan.derivations}
        task_output_to_task: dict[str, str] = {}
        task_output_value_kinds: dict[str, str] = {}
        for task in self.resolved_plan.tasks:
            if task.kind == "financial":
                task_outputs = [row.output_id for row in task.facts]
                for output_id in task_outputs:
                    task_output_value_kinds[output_id] = "money"
            else:
                task_outputs = []
                output_id = getattr(task, "output_id", None)
                if output_id:
                    task_outputs.append(output_id)
                    task_output_value_kinds[output_id] = "text"
                task_outputs.extend(
                    row.output_id for row in getattr(task, "field_outputs", []))
                for row in getattr(task, "field_outputs", []):
                    if row.value_kind in {"money", "text"}:
                        task_output_value_kinds[row.output_id] = row.value_kind
            for output_id in task_outputs:
                task_output_to_task[output_id] = task.task_id

        # Derivation output kinds are closed by the operator and the known
        # producer kinds.  Unknown inputs remain unknown; a registered root is
        # still checked whenever the producer type is determinable.
        derived_output_value_kinds: dict[str, str] = {}
        known_output_value_kinds = dict(task_output_value_kinds)
        for derivation in self.resolved_plan.derivations:
            operand_kinds = [
                known_output_value_kinds.get(ref.output_id)
                for ref in derivation.operands
            ]
            derived_kind: str | None = None
            if derivation.operator == "equal":
                derived_kind = "boolean"
            elif derivation.operator == "argmax":
                derived_kind = "text"
            elif derivation.operator in {
                    "difference", "absolute_difference",
                    "discrete_from_cumulative",
            } and operand_kinds and all(
                    kind == "money" for kind in operand_kinds):
                derived_kind = "money"
            elif derivation.operator == "percent_change" and operand_kinds \
                    and all(kind == "money" for kind in operand_kinds):
                # **비율은 금액이 아니다.** 금액 둘의 증감률이므로 `percent` 다.
                # 이 분기가 없어 「몇 퍼센트 늘었나」류 파생은 값 종류를 못 정했다.
                derived_kind = "percent"
            elif derivation.operator == "concept_ratio" and operand_kinds \
                    and all(kind == "money" for kind in operand_kinds) \
                    and derivation.presentation == "percent":
                # 배수(``multiple``)는 등록된 value_kind 축에 없어 미정으로 둔다 —
                # percent_change 와 같은 이유로 concept_ratio 도 금액이 아니다.
                derived_kind = "percent"
            if derived_kind is not None:
                derived_output_value_kinds[derivation.output_id] = derived_kind
                known_output_value_kinds[derivation.output_id] = derived_kind

        registered_value_kinds = {
            row.plan_output_id: row.value_kind
            for row in self.plan_value_roots
        }
        for value_root in self.plan_value_roots:
            if value_root.plan_output_id not in outputs:
                raise ValueError("plan value root가 resolved plan output을 참조하지 않습니다")
            if value_root.producer_task_id is not None:
                if value_root.producer_task_id not in task_ids:
                    raise ValueError("plan value root producer task가 resolved plan에 없습니다")
                if task_output_to_task.get(value_root.plan_output_id) != (
                        value_root.producer_task_id):
                    raise ValueError(
                        "plan value root producer task와 output의 소유권이 다릅니다")
            else:
                if value_root.producer_derivation_output_id not in derivation_outputs:
                    raise ValueError(
                        "plan value root producer derivation이 resolved plan에 없습니다")
                if value_root.producer_derivation_output_id != value_root.plan_output_id:
                    raise ValueError(
                        "plan value root derivation producer는 plan_output_id와 같아야 합니다")
            expected_value_kind = (
                task_output_value_kinds.get(value_root.plan_output_id)
                or derived_output_value_kinds.get(value_root.plan_output_id))
            if expected_value_kind is not None and (
                    value_root.value_kind != expected_value_kind):
                raise ValueError(
                    "plan value root value_kind가 producer value_kind와 다릅니다")
            if value_root.producer_derivation_output_id is not None:
                derivation = next(
                    row for row in self.resolved_plan.derivations
                    if row.output_id == value_root.plan_output_id)
                if derivation.operator in {"difference", "absolute_difference"}:
                    operand_registered_kinds = [
                        registered_value_kinds.get(ref.output_id)
                        for ref in derivation.operands
                    ]
                    if operand_registered_kinds and all(
                            kind == "money" for kind in operand_registered_kinds):
                        if value_root.value_kind != "money":
                            raise ValueError(
                                "money difference derivation root는 money여야 합니다")
        expected_premise_ids = [claim.claim_id
                                for claim in self.resolved_plan.premise_claims]
        actual_premise_ids = [row.premise_id for row in self.premise_roots]
        if actual_premise_ids != expected_premise_ids:
            raise ValueError(
                "premise root inventory가 resolved plan claim 순서와 다릅니다")
        if any(
                (row.plan_output_id is not None
                 and row.plan_output_id not in outputs)
                or (row.plan_task_id is not None
                    and row.plan_task_id not in task_ids)
                for row in self.answer_roots):
            raise ValueError("answer root가 resolved plan namespace에 없습니다")
        if any(row.root_id not in root_ids for row in self.support_roots):
            raise ValueError("support root가 live answer root가 아닙니다")
        if any(plan_root not in plan_root_ids
               for row in self.support_roots
               for plan_root in row.plan_root_ids):
            raise ValueError("support root가 live plan value root가 아닙니다")
        if any(
                answer_by_root[row.root_id].item_id != row.item_id
                or answer_by_root[row.root_id].projection_mode
                != row.projection_mode
                or answer_by_root[row.root_id].projection_binding_id
                != row.projection_binding_id
                for row in self.support_roots):
            raise ValueError(
                "support root의 item/projection binding이 answer root와 다릅니다")
        if any(root not in root_ids for row in self.premise_roots
               for root in row.root_ids):
            raise ValueError("premise root가 live answer root가 아닙니다")
        if any(plan_root not in plan_root_ids for row in self.premise_roots
               for plan_root in row.plan_root_ids):
            raise ValueError("premise plan root가 live plan value root가 아닙니다")
        if any(task not in task_ids for row in self.premise_roots
               for task in row.task_refs):
            raise ValueError("premise task ref가 resolved plan에 없습니다")
        if any(row.premise_id not in premise_ids for row in self.premise_roots):
            raise ValueError("premise root가 resolved plan claim namespace에 없습니다")
        if self.execution_plan_digest != self.compute_digest(
                self.model_dump(mode="json")):
            raise ValueError("execution_plan_digest가 일치하지 않습니다")
        return self


class DeterministicCompiledSlice(_StrictFrozenModel):
    """Auditable question -> intent -> resolution -> plan -> contract bundle."""

    schema_version: Literal[COMPILED_SLICE_VERSION] = COMPILED_SLICE_VERSION
    # This is resolver trace/provenance.  Generic lowering copies the strict
    # resolution question_id verbatim; the exact approved ID check belongs to
    # compile_stage1_v1, not to core dispatch.
    question_id: NonEmpty
    question: NonEmpty
    source_intent_digest: Digest
    resolution_digest: Digest
    overlay_digest: Digest | None = None
    intent: SemanticIntent
    resolution: AuthoritativeResolution
    execution_plan: ExecutionPlan
    answer_contract: CompiledAnswerContract
    bundle_digest: Digest

    @classmethod
    def compute_digest(cls, payload: Mapping[str, Any]) -> str:
        body = dict(payload)
        body.pop("bundle_digest", None)
        return canonical_sha256(body)

    @classmethod
    def create(cls, **payload: Any) -> "DeterministicCompiledSlice":
        body = dict(payload)
        body.setdefault("schema_version", COMPILED_SLICE_VERSION)
        candidate = cls.model_construct(**body)
        digest_body = candidate.model_dump(mode="json", warnings=False)
        body["bundle_digest"] = cls.compute_digest(digest_body)
        return cls.model_validate(body, strict=True)

    @model_validator(mode="after")
    def validate_slice(self) -> "DeterministicCompiledSlice":
        intent = SemanticIntent.model_validate(
            self.intent.model_dump(mode="python", warnings=False), strict=True)
        resolution = _strict_resolution(self.resolution)
        plan = load_execution_plan_json(
            canonical_json(self.execution_plan.model_dump(
                mode="json", warnings=False)))
        contract = CompiledAnswerContract.model_validate(
            self.answer_contract.model_dump(mode="json", warnings=False),
            strict=True)
        contract.validate_source_intent(intent)
        _validate_intent_grounding(self.question, intent)
        if self.source_intent_digest != semantic_intent_digest(intent):
            raise ValueError("slice source_intent_digest가 intent와 다릅니다")
        if resolution.question_id != self.question_id:
            raise ValueError("resolution question_id가 slice trace와 다릅니다")
        if resolution.source_intent_digest != self.source_intent_digest:
            raise ValueError("resolution source_intent_digest가 slice와 다릅니다")
        if self.resolution_digest != resolution.resolution_digest:
            raise ValueError("slice resolution_digest가 resolution과 다릅니다")
        if plan.source_intent_digest != self.source_intent_digest:
            raise ValueError("execution plan source digest가 다릅니다")
        if plan.resolution_digest != self.resolution_digest:
            raise ValueError("execution plan resolution digest가 다릅니다")
        if contract.source_intent_digest != self.source_intent_digest:
            raise ValueError("answer contract source digest가 다릅니다")
        if contract.execution_plan_digest != plan.execution_plan_digest:
            raise ValueError("answer contract execution plan digest가 다릅니다")
        intent_item, _ = _validate_intent_slice(intent, resolution)
        resolution_item, expected_overlay_digest = _validate_resolution_slice(
            intent_item, resolution)
        _validate_slice_cross_authority(
            intent, intent_item, resolution, resolution_item, plan, contract)
        if self.overlay_digest != expected_overlay_digest:
            raise ValueError("slice overlay digest가 selected lowering과 다릅니다")
        if self.bundle_digest != self.compute_digest(
                self.model_dump(mode="json", warnings=False)):
            raise ValueError("compiled slice bundle_digest가 일치하지 않습니다")
        return self


def _strict_instance(value: Any, model: type[BaseModel]) -> BaseModel:
    return _strict_json_model(model, value)


def verify_authoritative_resolution_digest(
        value: AuthoritativeResolution | Mapping[str, Any],
        ) -> str:
    validated = _strict_instance(value, AuthoritativeResolution)
    return validated.resolution_digest


def verify_execution_plan_digest(
        value: ExecutionPlan | Mapping[str, Any],
        ) -> str:
    validated = _strict_instance(value, ExecutionPlan)
    return validated.execution_plan_digest


class DeterministicPlanCompilerError(ValueError):
    """Raised when a bounded Stage1 v1 slice cannot close safely."""


def _expected_question(question_id: str) -> str:
    if question_id == G_A_001:
        return G_A_001_QUESTION
    if question_id == G_A_004:
        return G_A_004_QUESTION
    if question_id == G_A_010:
        return G_A_010_QUESTION
    if question_id == G_I_004:
        return G_I_004_QUESTION
    if question_id == G_I_006:
        return G_I_006_QUESTION
    if question_id == G_I_009:
        return G_I_009_QUESTION
    if question_id == G_O_001:
        return G_O_001_QUESTION
    if question_id == R_A_002:
        return R_A_002_QUESTION
    raise DeterministicPlanCompilerError(
        f"지원하지 않는 vertical slice question_id입니다: {question_id}")


def _strict_intent(value: SemanticIntent | Mapping[str, Any]) -> SemanticIntent:
    if isinstance(value, SemanticIntent):
        return SemanticIntent.model_validate(
            value.model_dump(mode="python", warnings=False), strict=True)
    if isinstance(value, Mapping):
        return SemanticIntent.model_validate(value, strict=True)
    raise TypeError("intent는 SemanticIntent 또는 mapping이어야 합니다")


def _strict_resolution(
        value: AuthoritativeResolution | Mapping[str, Any],
        ) -> AuthoritativeResolution:
    if isinstance(value, AuthoritativeResolution):
        return AuthoritativeResolution.model_validate(
            value.model_dump(mode="python", warnings=False), strict=True)
    if isinstance(value, Mapping):
        return load_authoritative_resolution_json(canonical_json(value))
    raise TypeError("resolution은 AuthoritativeResolution 또는 mapping이어야 합니다")


def _validate_intent_grounding(question: str, intent: SemanticIntent) -> None:
    """Re-ground every normalized semantic surface against supplied text.

    Generic compilation deliberately does not consult an exact question-ID
    table, but it must preserve the normalizer's NFC/exact-substring boundary
    when callers provide an already-normalized model.
    """
    if not isinstance(question, str) or not question.strip():
        raise DeterministicPlanCompilerError("question은 비어 있을 수 없습니다")
    question_nfc = unicodedata.normalize("NFC", question)

    question_compact = re.sub(r"\s+", "", question_nfc)

    def require_grounded(
            value: str, path: str, *, allow_closed_pair: bool = False,
            ) -> None:
        normalized = unicodedata.normalize("NFC", value)
        if not normalized or not normalized.strip():
            raise DeterministicPlanCompilerError(
                f"{path} surface는 비어 있을 수 없습니다")
        if normalized in question_nfc:
            return
        # 「A 전후 B」는 두 값을 요구하는 닫힌 합성 표현이다. 출력 필드에
        # 한해서만 이를 「A 전 B」와 「A 후 B」로 분해한다. 임의 동의어·값을
        # 허용하는 예외가 아니라 질문의 동일 토큰을 가역적으로 재배치하는
        # 규칙이므로 회사·문항·공시서식과 무관하게 적용된다.
        pair = re.fullmatch(
            r"(?P<prefix>.+?)\s+(?P<axis>전|후)\s+(?P<suffix>.+)",
            normalized.strip())
        if (allow_closed_pair and pair is not None
                and re.sub(r"\s+", "", (
                    f"{pair.group('prefix')} 전후 {pair.group('suffix')}"
                )) in question_compact):
            return
        if normalized not in question_nfc:
            raise DeterministicPlanCompilerError(
                f"{path} surface가 supplied question에 결속되지 않습니다")

    for index, entity in enumerate(intent.entities):
        require_grounded(entity.surface, f"entities[{index}].surface")
    for index, item in enumerate(intent.answer_items):
        prefix = f"answer_items[{index}]"
        require_grounded(item.target.surface, f"{prefix}.target.surface")
        for qualifier_index, surface in enumerate(item.target.qualifier_surfaces):
            require_grounded(
                surface,
                f"{prefix}.target.qualifier_surfaces[{qualifier_index}]",
            )
        for period_index, surface in enumerate(item.scope.target_period_expressions):
            require_grounded(
                surface,
                f"{prefix}.scope.target_period_expressions[{period_index}]",
            )
        if item.scope.as_of_expression is not None:
            require_grounded(item.scope.as_of_expression,
                             f"{prefix}.scope.as_of_expression")
        if item.scope.document_group_expression is not None:
            require_grounded(item.scope.document_group_expression,
                             f"{prefix}.scope.document_group_expression")
        for qualifier_index, surface in enumerate(
                item.scope.scope_qualifier_expressions):
            require_grounded(
                surface,
                f"{prefix}.scope.scope_qualifier_expressions[{qualifier_index}]",
            )
        if item.selection is not None:
            require_grounded(item.selection.criterion_surface,
                             f"{prefix}.selection.criterion_surface")
        for field_index, surface in enumerate(item.output.field_surfaces):
            require_grounded(
                surface,
                f"{prefix}.output.field_surfaces[{field_index}]",
                allow_closed_pair=True,
            )
    for index, premise in enumerate(intent.premises):
        require_grounded(premise.raw_text, f"premises[{index}].raw_text")
    for index, mention in enumerate(intent.unresolved_mentions):
        require_grounded(
            mention.raw_text, f"unresolved_mentions[{index}].raw_text")


def _validate_financial_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, Any]:
    """단일 재무 intent 의 **구조**를 검증한다.

    예전에는 값을 대조했다.

    ```
    intent.entities[0].surface  != "삼성전자"        → 거절
    expected_qualifiers_by_target = {"매출액": ["연결기준"], "매출": []}
    item.output.field_surfaces  != ["얼마"]          → 거절
    item.scope.target_period_expressions != ["2025년"] → 거절
    ```

    그래서 승인된 두 슬라이스 말고는 **어떤 재무 질문도 통과하지 못했다.**
    이것은 일반 경로에 등록된 검증기이므로 여기 값이 박히면 경로 전체가
    그 두 문항 전용이 된다.

    회사·계정·기간·표현이 무엇이든 **단일 재무 조회의 모양이면** 통과시킨다.
    """

    companies = [
        entity for entity in intent.entities
        if entity.kind_hint == "company"
    ]
    if len(companies) != 1:
        raise DeterministicPlanCompilerError(
            f"단일 재무 조회에는 회사 entity 하나가 필요합니다: {len(companies)}")
    if len(intent.answer_items) != 1:
        raise DeterministicPlanCompilerError(
            "단일 재무 조회는 answer item 하나만 지원합니다")
    if intent.answer_groups or intent.premises or intent.unresolved_mentions:
        raise DeterministicPlanCompilerError(
            "bounded slice에는 group/premise/unresolved mention이 없어야 합니다")
    item = intent.answer_items[0]
    if item.target.entity_refs != [companies[0].entity_id]:
        raise DeterministicPlanCompilerError("source entity binding이 다릅니다")
    if item.operation != "retrieve":
        raise DeterministicPlanCompilerError(
            "bounded compiler는 retrieve operation만 지원합니다")
    if item.output.shape != "scalar":
        raise DeterministicPlanCompilerError(
            "bounded compiler는 scalar output만 지원합니다")
    if len(item.output.field_surfaces) != 1:
        raise DeterministicPlanCompilerError(
            "단일 재무 조회의 요청 필드는 하나여야 합니다: "
            f"{len(item.output.field_surfaces)}")
    if item.selection is not None:
        raise DeterministicPlanCompilerError(
            "bounded financial slice에는 selection이 없습니다")
    if len(item.scope.target_period_expressions) != 1:
        raise DeterministicPlanCompilerError(
            "단일 재무 조회에는 대상 기간 표현 하나가 필요합니다")
    return item, companies[0]


def _validate_clarification_financial_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, None]:
    """Validate a financial source shape whose missing axis came from a turn.

    The immutable source intent intentionally remains incomplete. Canonical
    company/period/scope/concept coordinates live in the typed resolution and
    the surrounding ``ClarificationReadyBinding`` proves the accepted answer
    context. This validator is therefore structural only and is registered
    solely for signatures with a missing company or period axis.
    """

    companies = [
        entity for entity in intent.entities if entity.kind_hint == "company"]
    if len(companies) > 1 or len(intent.answer_items) != 1:
        raise DeterministicPlanCompilerError(
            "clarification 재무 조회는 company 최대 하나와 item 하나만 지원합니다")
    if intent.answer_groups or intent.premises or intent.unresolved_mentions:
        raise DeterministicPlanCompilerError(
            "clarification 재무 조회에 미해결 의미가 남아 있습니다")
    item = intent.answer_items[0]
    expected_refs = [] if not companies else [companies[0].entity_id]
    if (
        item.target.kind != "metric"
        or item.target.entity_refs != expected_refs
        or item.operation != "retrieve"
        or item.output.shape != "scalar"
        or item.output.projection_mode != "named_fields"
        or len(item.output.field_surfaces) != 1
        or item.selection is not None
        or len(item.scope.target_period_expressions) > 1
    ):
        raise DeterministicPlanCompilerError(
            "clarification 재무 조회 topology가 지원 범위를 벗어납니다")
    if companies and item.scope.target_period_expressions:
        raise DeterministicPlanCompilerError(
            "완전한 재무 source shape는 일반 financial handler의 소유입니다")
    return item, None


def _validate_premise_financial_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, tuple[Any, ...]]:
    """Validate a single financial lookup whose answer also verifies premise(s).

    The premise is not a second user answer.  It is a semantic assertion whose
    verdict is produced from the same source-bound financial fact, so every
    premise must apply to the one retrieved item.  Literal values and premise
    kinds deliberately remain source data rather than routing keys.
    """
    companies = [entity for entity in intent.entities
                 if entity.kind_hint == "company"]
    if len(companies) != 1 or len(intent.answer_items) != 1:
        raise DeterministicPlanCompilerError(
            "premise financial lookup에는 company/entity answer item이 각각 하나 필요합니다")
    if intent.answer_groups or intent.unresolved_mentions or not intent.premises:
        raise DeterministicPlanCompilerError(
            "premise financial lookup에는 group/unresolved가 없고 premise가 필요합니다")
    item = intent.answer_items[0]
    if (
            item.target.kind != "metric"
            or item.target.entity_refs != [companies[0].entity_id]
            or item.operation != "retrieve"
            or item.output.shape != "scalar"
            or item.output.projection_mode != "named_fields"
            or len(item.output.field_surfaces) != 1
            or item.selection is not None
            or len(item.scope.target_period_expressions) != 1
    ):
        raise DeterministicPlanCompilerError(
            "premise financial lookup의 단일 scalar retrieve topology가 다릅니다")
    if any(premise.applies_to_item_ids != [item.item_id]
           for premise in intent.premises):
        raise DeterministicPlanCompilerError(
            "premise financial lookup premise는 단일 financial item에만 적용돼야 합니다")
    # The registry convention returns ``(authority, secondary)``; keep the
    # premise inventory inside the authority passed to resolution/lowering.
    return (item, tuple(intent.premises)), companies[0]


def _validate_intent_slice(
        intent: SemanticIntent,
        resolution: "AuthoritativeResolution | None" = None,
        ) -> tuple[Any, Any]:
    """Validate the bounded semantic shapes without consulting a trace id.

    **확정된 것이 있으면 그것으로 고른다.**  아래 모양 분기는 resolution 이 없을
    때만 쓰는 되돌림 경로다.  resolver 가 정본에서 kind 를 확정했다면 어떤
    validator 를 태울지는 이미 정해진 것이고, 그때 `target.kind` 나 output shape
    으로 다시 고르면 모델의 표현 흔들림이 경로를 바꾼다.  고른 validator 는 그대로
    돌므로 검사가 느슨해지지는 않는다.
    """

    if resolution is not None:
        # Reuse the same qid-free selection rule as generic compilation.
        # Looking only at kind used to send same-company two-period financial
        # relations through the single-fact validator once several
        # ``financial_comparison`` handlers existed.
        return _selected_handler(intent, resolution).validate_intent(intent)
    if (len(intent.answer_items) == 1 and (
            intent.answer_items[0].output.projection_mode == "whole_target"
            or (intent.answer_items[0].target.kind == "topic"
                and intent.answer_items[0].target.surface.replace(" ", "") == "최근투자계획"
                and intent.answer_items[0].output.projection_mode == "named_fields"))):
        return _validate_document_collection_intent(intent)
    if len(intent.answer_items) == 2 and {
            item.output.shape for item in intent.answer_items} == {
                "verdict", "narrative"}:
        return _validate_document_fact_comparison_intent(intent)
    if len(intent.answer_items) == 2 and all(
            item.target.kind == "event" and item.output.shape == "scalar"
            for item in intent.answer_items):
        return _validate_document_attribute_intent(intent)
    if len(intent.answer_items) == 2:
        return _validate_same_day_status_intent(intent)
    if len(intent.answer_items) == 1:
        item = intent.answer_items[0]
        if item.target.kind == "event":
            return _validate_selected_event_intent(intent)
        if item.target.kind == "topic" and item.operation == "compare":
            return _validate_periodic_narrative_comparison_intent(intent)
        if item.target.kind == "topic":
            return _validate_periodic_document_narrative_intent(intent)
        if item.operation == "compare":
            return _validate_financial_comparison_intent(intent)
        if item.target.kind == "metric":
            return _validate_financial_intent(intent)
    raise DeterministicPlanCompilerError(
        "지원하지 않는 bounded SemanticIntent structural shape입니다")


def _validate_document_collection_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, Any]:
    """Validate a qid- and surface-free whole-document-set request."""
    if len(intent.entities) != 1 or intent.entities[0].kind_hint != "company":
        raise DeterministicPlanCompilerError(
            "document collection에는 company entity 하나가 필요합니다")
    if len(intent.answer_items) != 1:
        raise DeterministicPlanCompilerError(
            "document collection에는 answer item 하나가 필요합니다")
    if intent.answer_groups or intent.premises or intent.unresolved_mentions:
        raise DeterministicPlanCompilerError(
            "document collection에는 group/premise/unresolved가 없어야 합니다")
    item = intent.answer_items[0]
    if (
            item.item_id != "item-1"
            or item.target.entity_refs != ["entity-1"]
            or item.target.qualifier_surfaces
            or item.operation != "retrieve"
            or item.scope.as_of_expression is not None
            or item.scope.document_group_expression is not None
            or item.scope.scope_qualifier_expressions
            or item.selection is not None
            or item.target.kind not in {"document", "topic"}
            or item.output.presentation != "auto"
            or intent.presentation != "auto"
    ):
        raise DeterministicPlanCompilerError(
            "document collection whole-target intent topology가 다릅니다")
    is_whole_target = (item.output.projection_mode == "whole_target"
                       and item.output.shape in {
                           "record_list", "record", "narrative"}
                       and not item.output.field_surfaces)
    is_recent_investment_narrative = (
        item.target.kind == "topic"
        and item.target.surface.replace(" ", "") == "최근투자계획"
        and item.output.projection_mode == "named_fields"
        and item.output.shape == "narrative"
        and len(item.output.field_surfaces) == 1)
    if not (is_whole_target or is_recent_investment_narrative):
        raise DeterministicPlanCompilerError(
            "document collection output topology가 다릅니다")
    # Keep semantic entity authority with the item.  Neither literal surface
    # is part of handler selection; both are checked only after dispatch.
    return (item, intent.entities[0]), None


def _validate_holding_disclosure_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, Any]:
    """Accept one named-field holding request and no unresolved coordinate."""

    if (len(intent.answer_items) != 1 or intent.answer_groups
            or intent.premises or intent.unresolved_mentions):
        raise DeterministicPlanCompilerError(
            "holding disclosure에는 확정된 answer item 하나가 필요합니다")
    item = intent.answer_items[0]
    if (item.item_id != "item-1"
            # `특별관계자 직업` 처럼 문서의 한 칸을 묻는 질문을 HCX 가
            # `attribute` 로 내기도 한다 — 같은 질문이 실행마다 `document` 와
            # 갈린다. 아래 의미 표지 검사가 지분 질문임을 따로 확인하므로
            # 여기서 모양만으로 닫으면 답할 수 있는 것을 못 답한다 (#138).
            or item.target.kind not in {"document", "event", "topic", "attribute"}
            or item.operation != "retrieve"
            or item.output.projection_mode != "named_fields"
            or item.output.shape not in {"scalar", "record", "record_list", "narrative"}
            or not item.output.field_surfaces
            or item.output.presentation != "auto"
            or intent.presentation != "auto"
            or len(item.scope.target_period_expressions) > 1
            or item.scope.scope_qualifier_expressions
            or item.selection is not None):
        raise DeterministicPlanCompilerError(
            "holding disclosure semantic topology가 다릅니다")
    surfaces = " ".join([
        item.target.surface,
        item.scope.document_group_expression or "",
        *item.target.qualifier_surfaces,
        *item.output.field_surfaces,
    ])
    if re.search(
            r"대량\s*보유|보유\s*(?:주식|비율|목적)|지분\s*율|"
            r"특별\s*관계자|주식을\s*몇\s*주\s*보유|"
            # `국적`·`직업(사업내용)` 은 대량보유보고서가 공개하는 칸이고 슬롯도
            # 있다(#138). 표지 어휘에 없으면 규격이 지분 질문으로 알아보지 못해
            # 「의미 표지가 없습니다」로 닫힌다. `사업내용` 단독은 사업보고서
            # 질문과 겹치므로 넣지 않는다.
            r"국적|직업|"
            r"몇\s*주\s*변동|변동\s*(?:주식\s*수|방법|사유)|holding",
            surfaces, re.I) is None:
        raise DeterministicPlanCompilerError(
            "holding disclosure 의미 표지가 없습니다")
    company_entities = [row for row in intent.entities if row.kind_hint == "company"]
    if not company_entities:
        raise DeterministicPlanCompilerError(
            "holding disclosure에는 발행회사 company entity가 필요합니다")
    return (item, tuple(company_entities)), None


def _validate_document_version_history_intent(
        intent: SemanticIntent) -> tuple[Any, Any]:
    if (len(intent.entities) != 1 or intent.entities[0].kind_hint != "company"
            or len(intent.answer_items) != 1 or len(intent.premises) != 1
            or intent.answer_groups or intent.unresolved_mentions):
        raise DeterministicPlanCompilerError("document version intent inventory가 다릅니다")
    item, premise = intent.answer_items[0], intent.premises[0]
    if (
            item.item_id != "item-1" or item.target.entity_refs != ["entity-1"]
            or item.target.kind != "document" or item.operation != "retrieve"
            or item.scope.target_period_expressions or item.scope.as_of_expression is not None
            or item.scope.document_group_expression is not None
            or item.scope.scope_qualifier_expressions or item.selection is not None
            or item.output.shape != "scalar" or item.output.projection_mode != "named_fields"
            or len(item.output.field_surfaces) != 1
            or re.search(r"정정|버전|변경\s*이력|원본",
                         item.output.field_surfaces[0]) is None
            or premise.kind != "existence"
            or premise.applies_to_item_ids != [item.item_id]):
        raise DeterministicPlanCompilerError("document version premise topology가 다릅니다")
    return (item, intent.entities[0]), None


def _validate_selected_event_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, Any]:
    """One selected business event, with no residual ambiguity or plan patch.

    The ordinary one-field read and the two-observation status read both use
    the same selected-event authority.  The latter may keep the issuer and
    the named external event separate, or may keep only the issuer plus a
    seed-event date.  Neither topology carries a literal answer or a fixture
    identity.
    """
    if len(intent.answer_items) != 1:
        raise DeterministicPlanCompilerError(
            "selected event에는 answer item 하나가 필요합니다")
    item = intent.answer_items[0]
    referenced_ids = set(item.target.entity_refs)
    company_entities = [entity for entity in intent.entities
                        if entity.kind_hint == "company"]
    unreferenced_companies = [entity for entity in company_entities
                              if entity.entity_id not in referenced_ids]
    issuer = (
        unreferenced_companies[0]
        if len(unreferenced_companies) == 1 and len(intent.entities) == 2
        else (company_entities[0] if len(company_entities) == 1 else None)
    )
    missing_issuer_context = (
        not intent.entities and not item.target.entity_refs
        and item.target.kind == "event"
        and not item.target.qualifier_surfaces
        and not item.scope.target_period_expressions
        and item.scope.as_of_expression is None
        and item.output.shape == "scalar"
    )
    issuer_omitted_two_observation = (
        issuer is None
        and len(intent.entities) == 1
        and intent.entities[0].kind_hint in {"event", "document"}
        and item.target.entity_refs == [intent.entities[0].entity_id]
        and item.target.kind in {"event", "document"}
        and len(item.target.qualifier_surfaces) == 2
        and not item.scope.target_period_expressions
        and item.scope.as_of_expression is None
        and item.output.shape in {"record", "narrative", "comparison"}
    )
    if issuer is None and not (
            missing_issuer_context or issuer_omitted_two_observation):
        raise DeterministicPlanCompilerError(
            "selected event에는 issuer company entity 하나가 필요합니다")
    if intent.answer_groups or intent.unresolved_mentions:
        raise DeterministicPlanCompilerError(
            "selected event에는 group/unresolved가 없어야 합니다")
    if any(
            premise.kind != "state"
            or premise.applies_to_item_ids != [item.item_id]
            for premise in intent.premises):
        raise DeterministicPlanCompilerError(
            "selected event premise는 단일 item의 state 주장만 지원합니다")
    # 「가장 최근 건」은 **해소된 선택**이다.  resolver 가 이미 접수번호 하나를
    # 골랐고 그 접수번호가 이 계획의 유일한 사건이다.  남은 요구가 아니므로
    # 여기서 잠그면 사용자가 준 기준을 이유로 답을 거절하게 된다.
    # 최대·최소처럼 무엇의 극값인지 따로 정해야 하는 선택은 그대로 거절한다.
    selection_mode = getattr(item.selection, "mode", None)
    common = (
        item.item_id == "item-1"
        and item.operation == "retrieve"
        and item.scope.document_group_expression is None
        and not item.scope.scope_qualifier_expressions
        and selection_mode in {None, "latest"}
        and item.target.kind in {"event", "document", "metric", "attribute"}
        and item.output.projection_mode == "named_fields"
        and item.output.presentation == "auto"
        and intent.presentation == "auto"
    )
    legacy = (
        len(intent.entities) == 1
        and issuer is not None
        and item.target.entity_refs == [issuer.entity_id]
        and not item.target.qualifier_surfaces
        and not item.scope.target_period_expressions
        and item.output.shape in {"scalar", "narrative"}
        and len(item.output.field_surfaces) == 1
    )
    two_observation_status = (
        item.output.shape in {"record", "narrative", "comparison"}
        and (
            # External event/company plus two observation dates on target.
            (issuer is not None and len(intent.entities) == 2
             and item.target.entity_refs != [issuer.entity_id]
             and len(item.target.qualifier_surfaces) == 2
             and not item.scope.target_period_expressions)
            # One issuer, seed event date on target and observation dates in scope.
            or (issuer is not None and len(intent.entities) == 1
                and item.target.entity_refs == [issuer.entity_id]
                and len(item.target.qualifier_surfaces) in {0, 1}
                and len(item.scope.target_period_expressions) == 2)
            # The question names only an external event/counterparty.  The
            # canonical typed resolution is the issuer authority; source target
            # and observation coordinates remain independently bound below.
            or issuer_omitted_two_observation
        )
    )
    one_observation_status = (
        len(intent.entities) == 2 and item.target.kind == "document"
        and item.target.entity_refs != [issuer.entity_id]
        and len(item.target.qualifier_surfaces) == 1
        and len(item.scope.target_period_expressions) == 1
        and item.scope.as_of_expression is not None
        and item.output.shape == "scalar" and len(item.output.field_surfaces) == 1
    )
    partial_timeline = (
        len(intent.entities) == 1 and item.target.kind == "document"
        and item.target.entity_refs == [issuer.entity_id]
        and item.scope.as_of_expression is None
        and not item.target.qualifier_surfaces
        and len(item.scope.target_period_expressions) <= 1
        and item.output.shape in {"narrative", "timeline"}
        and len(item.output.field_surfaces) == 2
    )
    complete_timeline = (
        issuer is not None and len(intent.entities) == 2
        and item.target.kind in {"event", "document"}
        and item.target.entity_refs != [issuer.entity_id]
        and not item.target.qualifier_surfaces
        and len(item.scope.target_period_expressions) == 2
        and item.scope.as_of_expression is not None
        and item.output.shape == "timeline"
        and len(item.output.field_surfaces) == 2
    )
    record_qualifier_exact = not item.target.qualifier_surfaces
    if len(item.target.qualifier_surfaces) == 1:
        qualifier_start, qualifier_end, qualifier_error = _target_date_range(
            item.target.qualifier_surfaces[0], reference_date=date(2026, 1, 1))
        record_qualifier_exact = (
            qualifier_error is None and qualifier_start is not None
            and qualifier_start == qualifier_end)
    record_scope_bounded = False
    if (not item.target.qualifier_surfaces
            and len(item.scope.target_period_expressions) == 1):
        period_start, period_end, period_error = _target_date_range(
            item.scope.target_period_expressions[0],
            reference_date=date(2026, 1, 1))
        if period_error is None and period_start and period_end:
            start_day = date(
                int(period_start[:4]), int(period_start[4:6]),
                int(period_start[6:8]))
            end_day = date(
                int(period_end[:4]), int(period_end[4:6]),
                int(period_end[6:8]))
            record_scope_bounded = (
                0 <= (end_day - start_day).days <= 30)
    # One already-selected event/document can expose several named fields.
    # The resolver must still bind all fields to one canonical event key; this
    # branch merely admits that closed record topology.
    event_attribute_record = (
        selection_mode in {None, "latest"}
        and len(intent.entities) in {1, 2}
        and issuer is not None
        and len(item.target.entity_refs) == 1
        and item.target.entity_refs[0] in {
            entity.entity_id for entity in intent.entities}
        and item.target.kind in {"event", "document"}
        # 정확한 공시일 하나는 사건을 고르는 selector 좌표다. 여러 qualifier나
        # 비날짜 표현은 resolver가 유일 사건으로 증명하지 못하므로 받지 않는다.
        and (
            (len(item.target.qualifier_surfaces) <= 1
             and record_qualifier_exact
             and not item.scope.target_period_expressions)
            or record_scope_bounded
        )
        and item.scope.as_of_expression is None
        and item.output.shape == "record"
        # 필드 하나짜리 record 도 같은 닫힌 topology 다. ``누구랑 공급계약
        # 맺었어`` 는 한 사건에서 칸 하나(`계약상대`)를 뽑는 요청인데,
        # `legacy` 는 필드 1개를 받되 record 를 받지 않고 여기는 record 를 받되
        # 2개 이상을 요구해서 그 사이로 빠졌다 (`coverage_set_v01` K-029).
        #
        # **이 조건만 넓히면 안 된다.** 9개 selected_event handler 가 이
        # 함수를 공유하므로, signature 정확 매칭이 없으면 kind 후보 단계에서
        # 9개가 동시에 통과해 「모호」로 실패한다. 아래 registry 에 같은 모양의
        # signature 를 등록해 1단계에서 하나로 정해지게 해 두었다.
        and len(item.output.field_surfaces) >= 1
        and not intent.premises
    )
    # The exchange form ``투자판단관련주요경영사항`` exposes the disclosed
    # body as ``주요내용``.  The regrounder normalizes its exact filing date into
    # the one target qualifier and its one body read into a record.  This is
    # still one receipt-selected event, but it must remain narrower than the
    # ordinary selected-event path: one issuer, one exact date, the literal
    # form, and the one literal ``내용`` request only.
    investment_judgment_exact_content = (
        selection_mode in {None, "latest"}
        and len(intent.entities) == 1
        and issuer is not None
        and item.target.kind == "event"
        and item.target.entity_refs == [issuer.entity_id]
        and len(item.target.qualifier_surfaces) == 1
        and record_qualifier_exact
        and re.search(
            r"투자\s*판단\s*(?:관련)?\s*주요\s*경영\s*사항",
            item.target.surface) is not None
        and not item.scope.target_period_expressions
        and item.scope.as_of_expression is None
        and item.output.shape == "record"
        and item.output.projection_mode == "named_fields"
        and len(item.output.field_surfaces) == 1
        and re.sub(r"\s+", "", item.output.field_surfaces[0]) == "내용"
        and item.output.presentation == "prose"
        and intent.presentation == "prose"
        and not intent.premises
    )
    if not ((common and (
            legacy or missing_issuer_context or two_observation_status
            or one_observation_status or partial_timeline
            or complete_timeline or event_attribute_record))
            or investment_judgment_exact_content):
        raise DeterministicPlanCompilerError(
            "selected event intent topology가 다릅니다")
    return (item, issuer, tuple(intent.premises)), None


def _ordered_semantic_observation_dates(
        surfaces: list[str], *, reference_date: date,
        ) -> list[str] | None:
    """Recover an ordered exact-date pair from semantic surfaces only.

    A provider may keep the second coordinate as a question-grounded omission
    (``2025-12-25``, ``26`` or ``25년 1월 23일``, ``4월 2일``).  Recovery is
    allowed only from the immediately preceding exact date and never rolls a
    month/year forward implicitly.
    """

    if len(surfaces) != 2:
        return None
    result: list[str] = []
    previous: date | None = None
    for surface in surfaces:
        start, end, error = _target_date_range(
            surface, reference_date=reference_date)
        if error is None and start is not None and start == end:
            current = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
        elif previous is not None:
            compact = surface.strip()
            month_day = re.fullmatch(
                r"(?P<month>[0-9]{1,2})\s*월\s*(?P<day>[0-9]{1,2})\s*일?",
                compact)
            bare_day = re.fullmatch(
                r"(?P<day>[0-9]{1,2})\s*일?", compact)
            try:
                if month_day is not None:
                    current = date(
                        previous.year, int(month_day.group("month")),
                        int(month_day.group("day")))
                elif bare_day is not None:
                    current = date(
                        previous.year, previous.month,
                        int(bare_day.group("day")))
                else:
                    return None
            except ValueError:
                return None
        else:
            return None
        if previous is not None and current < previous:
            return None
        result.append(current.strftime("%Y%m%d"))
        previous = current
    return result


def _validate_event_amount_change_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, Any]:
    """Validate an underspecified contract-amount change request.

    Clarification answers are digest-bound sidecar authority and therefore do
    not get patched into the immutable source intent.  This validator keeps
    the accepted source shape deliberately narrow while leaving company,
    event, timepoints and amount axis to the typed resolution.
    """
    if (
            intent.entities
            or len(intent.answer_items) != 1
            or intent.answer_groups
            or intent.premises
            or intent.unresolved_mentions
            or intent.presentation != "auto"
    ):
        raise DeterministicPlanCompilerError(
            "event amount change intent inventory가 다릅니다")
    item = intent.answer_items[0]
    normalized = "".join(item.target.surface.split()).casefold()
    if (
            item.item_id != "item-1"
            or item.target.kind != "metric"
            or item.target.entity_refs
            or item.target.qualifier_surfaces
            or item.operation != "retrieve"
            or item.scope.target_period_expressions
            or item.scope.as_of_expression is not None
            or item.scope.document_group_expression is not None
            or item.scope.scope_qualifier_expressions
            or item.selection is not None
            or item.output.shape != "scalar"
            or item.output.projection_mode != "named_fields"
            or len(item.output.field_surfaces) != 1
            or item.output.presentation != "auto"
            or "계약" not in normalized
            or "금액" not in normalized
            or not any(token in normalized for token in ("변동", "차이", "증감"))
    ):
        raise DeterministicPlanCompilerError(
            "event amount change intent topology가 다릅니다")
    return item, None


def _validate_periodic_document_narrative_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, Any]:
    """Validate a source-grounded periodic record-list retrieval topology."""
    if len(intent.entities) != 1 \
            or intent.entities[0].kind_hint != "company":
        raise DeterministicPlanCompilerError(
            "periodic document narrative에는 company entity 하나가 필요합니다")
    if len(intent.answer_items) != 1:
        raise DeterministicPlanCompilerError(
            "periodic document narrative에는 answer item 하나가 필요합니다")
    if intent.answer_groups or intent.premises or intent.unresolved_mentions:
        raise DeterministicPlanCompilerError(
            "periodic document narrative에는 group/premise/unresolved가 없어야 합니다")
    item = intent.answer_items[0]
    # **어느 보고서인가는 resolver 가 이미 증명했다.**
    #
    # 그래서 여기서는 그 증명을 다시 요구하지 않는다. `target.kind` 가 topic
    # 이냐 document 냐, 보고서 표현이 scope 에 있느냐 target 에 있느냐는 모델이
    # 판마다 흔드는 축이고, 그것으로 문을 잠그면 같은 질문이 판마다 거절된다.
    #
    # 대신 **계획을 세우는 데 실제로 쓰이는 것**만 본다. 회사 하나, 항목 하나,
    # 조회 동작, 없음/최신 선택, 서로 다른 이름 필드 하나 이상.  최신 선택은
    # resolver가 exact filing을 증명한 뒤에만 이 validator에 도달한다. 이 중 하나라도 어긋나면
    # 내려갈 계획이 달라지므로 예전처럼 거절한다.
    if (
            item.item_id != "item-1"
            or item.target.entity_refs != ["entity-1"]
            or item.operation != "retrieve"
            or item.scope.as_of_expression is not None
            or item.scope.scope_qualifier_expressions
            or (item.selection is not None
                and item.selection.mode != "latest")
            or item.output.shape not in {"record_list", "record", "narrative"}
            or item.output.projection_mode != "named_fields"
            or len(item.output.field_surfaces) < 1
            or len(set(item.output.field_surfaces)) != len(
                item.output.field_surfaces)
            or item.output.presentation != "auto"
            or intent.presentation != "auto"
    ):
        raise DeterministicPlanCompilerError(
            "periodic document narrative intent topology가 다릅니다")
    return item, intent.entities[0]


def _validate_periodic_narrative_comparison_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, Any]:
    """Validate a surface-free two-period open narrative comparison."""
    if len(intent.entities) != 1 or intent.entities[0].kind_hint != "company":
        raise DeterministicPlanCompilerError(
            "periodic narrative comparison에는 company entity 하나가 필요합니다")
    if len(intent.answer_items) != 1:
        raise DeterministicPlanCompilerError(
            "periodic narrative comparison에는 answer item 하나가 필요합니다")
    if intent.answer_groups or intent.premises or intent.unresolved_mentions:
        raise DeterministicPlanCompilerError(
            "periodic narrative comparison에는 group/premise/unresolved가 없어야 합니다")
    item = intent.answer_items[0]
    if (
            item.item_id != "item-1"
            or item.target.entity_refs != ["entity-1"]
            or item.operation != "compare"
            or len(item.scope.target_period_expressions) != 2
            or len(set(item.scope.target_period_expressions)) != 2
            or item.scope.as_of_expression is not None
            or item.scope.scope_qualifier_expressions
            or item.selection is not None
            or item.output.shape not in {"narrative", "comparison"}
            or len(item.output.field_surfaces) < 2
            or len(set(item.output.field_surfaces)) != len(item.output.field_surfaces)
            or item.output.presentation != "auto"
            or intent.presentation != "auto"
    ):
        raise DeterministicPlanCompilerError(
            "periodic narrative comparison intent topology가 다릅니다")
    # Keep both ordered semantic and entity authorities together without
    # introducing question IDs or literal surfaces into handler selection.
    return (item, intent.entities[0]), None


def _validate_same_day_status_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, Any]:
    if len(intent.entities) != 1 \
            or intent.entities[0].kind_hint != "counterparty":
        raise DeterministicPlanCompilerError(
            "same-day status에는 counterparty entity 하나가 필요합니다")
    if (
            len(intent.answer_items) != 2
            or len(intent.answer_groups) != 1
            or intent.answer_groups[0].group_id != "group-1"
            or intent.answer_groups[0].item_ids != ["item-1", "item-2"]
    ):
        raise DeterministicPlanCompilerError(
            "same-day status item/group inventory가 다릅니다")
    if intent.premises or intent.unresolved_mentions:
        raise DeterministicPlanCompilerError(
            "same-day status에는 premise/unresolved mention이 없어야 합니다")
    document_item, event_item = intent.answer_items
    if (
            document_item.item_id != "item-1"
            or document_item.target.entity_refs != ["entity-1"]
            or document_item.target.qualifier_surfaces
            or document_item.operation != "retrieve"
            or document_item.scope.as_of_expression is None
            or document_item.scope.document_group_expression is not None
            or document_item.scope.target_period_expressions
            or document_item.scope.scope_qualifier_expressions
            or document_item.selection is None
            or document_item.selection.mode != "latest"
            or document_item.selection.k is not None
            or document_item.output.shape != "narrative"
            or len(document_item.output.field_surfaces) != 1
            or document_item.output.presentation != "auto"
    ):
        raise DeterministicPlanCompilerError(
            "same-day document intent topology가 다릅니다")
    if (
            event_item.item_id != "item-2"
            or event_item.target.entity_refs != ["entity-1"]
            or event_item.target.qualifier_surfaces
            or event_item.operation != "retrieve"
            or len(event_item.scope.scope_qualifier_expressions) != 1
            or event_item.scope.document_group_expression is not None
            or event_item.scope.target_period_expressions
            or event_item.scope.as_of_expression is not None
            or event_item.selection is not None
            or event_item.output.shape != "scalar"
            or len(event_item.output.field_surfaces) != 1
            or event_item.output.presentation != "auto"
            or intent.presentation != "auto"
    ):
        raise DeterministicPlanCompilerError(
            "same-day event intent topology가 다릅니다")
    return (document_item, event_item), intent.entities[0]


def _validate_termination_reported_status_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, Any]:
    """One reported termination with issuer/external roles and optional premise."""
    if len(intent.entities) != 2 or any(
            row.kind_hint != "company" for row in intent.entities):
        raise DeterministicPlanCompilerError(
            "reported termination에는 company entity 둘이 필요합니다")
    if (len(intent.answer_items) != 1 or intent.answer_groups
            or intent.unresolved_mentions or len(intent.premises) > 1):
        raise DeterministicPlanCompilerError(
            "reported termination item topology가 다릅니다")
    item = intent.answer_items[0]
    if any(
            premise.kind != "state"
            or premise.applies_to_item_ids != [item.item_id]
            for premise in intent.premises):
        raise DeterministicPlanCompilerError(
            "reported termination premise는 단일 item의 state 주장만 지원합니다")
    if (
            item.item_id != "item-1"
            or item.target.kind != "event"
            or item.target.entity_refs != ["entity-1", "entity-2"]
            or item.operation != "retrieve"
            or item.scope.target_period_expressions
            or item.scope.document_group_expression is not None
            or item.scope.scope_qualifier_expressions
            or item.selection is not None
            or item.output.projection_mode != "named_fields"
            or item.output.presentation != "auto"
            or intent.presentation != "auto"
    ):
        raise DeterministicPlanCompilerError(
            "reported termination semantic topology가 다릅니다")
    exact_observation = (
        len(item.target.qualifier_surfaces) == 1
        and item.scope.as_of_expression is not None
        and item.target.qualifier_surfaces[0] == item.scope.as_of_expression
    )
    cutoff_observation = (
        not item.target.qualifier_surfaces
        and item.scope.as_of_expression is None
        and item.output.shape == "scalar"
    )
    if not (exact_observation or cutoff_observation):
        raise DeterministicPlanCompilerError(
            "reported termination 관측 시점 topology가 다릅니다")
    _normalized, roles = _termination_reported_status_field_roles(item)
    if (cutoff_observation and roles != ["reported_status"]) or (
            exact_observation and tuple(sorted(roles)) not in {
                ("reported_status",),
                ("reported_status", "termination_amount"),
            }):
        raise DeterministicPlanCompilerError(
            "reported termination output 의미 topology가 다릅니다")
    return (item, intent.entities[0], tuple(intent.premises)), None


def _termination_reported_status_field_roles(
        intent_item: Any,
        ) -> tuple[list[str], list[str]]:
    """해지 상태 출력을 질문의 필드 순서가 아닌 의미로 분류한다."""
    normalized = [
        normalize_slot_names([surface])[0]
        for surface in intent_item.output.field_surfaces
    ]
    roles = [
        (
            "termination_amount" if (
                name.endswith("해지금액") or "얼마짜리" in name)
            else "reported_status" if re.search(
                r"상태|유효|살아\s*있|끝난|해지(?:됨|됐다|되었)", name)
            else ""
        )
        for name in normalized
    ]
    if (not roles or "" in roles or roles.count("reported_status") != 1
            or roles.count("termination_amount") > 1):
        raise DeterministicPlanCompilerError(
            "reported termination 필드는 상태 하나와 선택적 해지금액이어야 합니다")
    return normalized, roles


def _validate_document_fact_comparison_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, Any]:
    """Validate the qid/surface-free two-item comparison topology."""
    if len(intent.entities) != 1 or intent.entities[0].kind_hint != "event":
        raise DeterministicPlanCompilerError(
            "document fact comparison에는 event entity 하나가 필요합니다")
    if [row.item_id for row in intent.answer_items] != ["item-1", "item-2"]:
        raise DeterministicPlanCompilerError(
            "document fact comparison item 순서/count가 다릅니다")
    if len(intent.answer_groups) != 1 \
            or intent.answer_groups[0].group_id != "group-1" \
            or intent.answer_groups[0].item_ids != [
            "item-1", "item-2"]:
        raise DeterministicPlanCompilerError(
            "document fact comparison group이 두 item을 순서대로 묶어야 합니다")
    if len(intent.premises) != 1 or intent.premises[0].premise_id != "premise-1":
        raise DeterministicPlanCompilerError(
            "document fact comparison에는 정확히 하나의 premise가 필요합니다")
    premise = intent.premises[0]
    if premise.kind != "comparison" or premise.applies_to_item_ids != [
            "item-1", "item-2"]:
        raise DeterministicPlanCompilerError(
            "document fact comparison premise kind/applicability가 다릅니다")
    if premise.raw_text != intent.answer_items[0].output.field_surfaces[0]:
        raise DeterministicPlanCompilerError(
            "document fact comparison premise raw_text는 verdict source field여야 합니다")
    if intent.unresolved_mentions or intent.presentation != "auto":
        raise DeterministicPlanCompilerError(
            "document fact comparison에는 unresolved mention/비자동 presentation이 없어야 합니다")

    first, second = intent.answer_items
    if first.target.entity_refs != ["entity-1"] \
            or second.target.entity_refs != ["entity-1"] \
            or first.target.surface != second.target.surface \
            or first.target.surface != intent.entities[0].surface:
        raise DeterministicPlanCompilerError(
            "document fact comparison target entity/surface binding이 다릅니다")
    for item in (first, second):
        if (
                item.operation != "retrieve"
                or item.scope.as_of_expression is not None
                or item.scope.document_group_expression is not None
                or item.scope.scope_qualifier_expressions
                or item.selection is not None
                or len(item.output.field_surfaces) != 1
                or item.output.presentation != "auto"
        ):
            raise DeterministicPlanCompilerError(
                "document fact comparison item scope/selection/output shape가 다릅니다")
    if first.output.shape != "verdict" or second.output.shape != "narrative":
        raise DeterministicPlanCompilerError(
            "document fact comparison output은 verdict 후 narrative여야 합니다")
    # Handler plumbing reserves the second tuple slot for an entity or an
    # overlay; keep the ordered item pair nested so both items reach lowering.
    return (first, second), None


def _validate_document_attribute_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, Any]:
    """Validate a surface-free two-document attribute retrieval topology."""
    if len(intent.entities) != 1 or intent.entities[0].kind_hint != "company":
        raise DeterministicPlanCompilerError(
            "document attribute retrieval에는 company entity 하나가 필요합니다")
    if [row.item_id for row in intent.answer_items] != ["item-1", "item-2"]:
        raise DeterministicPlanCompilerError(
            "document attribute item 순서/count가 다릅니다")
    if (
            len(intent.answer_groups) != 1
            or intent.answer_groups[0].group_id != "group-1"
            or intent.answer_groups[0].item_ids != ["item-1", "item-2"]
    ):
        raise DeterministicPlanCompilerError(
            "document attribute group이 두 item을 순서대로 묶어야 합니다")
    if intent.premises or intent.unresolved_mentions or intent.presentation != "auto":
        raise DeterministicPlanCompilerError(
            "document attribute에는 premise/unresolved/비자동 presentation이 없어야 합니다")
    for item in intent.answer_items:
        if (
                item.target.entity_refs != ["entity-1"]
                or item.target.qualifier_surfaces
                or item.operation != "retrieve"
                or item.scope.as_of_expression is not None
                or item.scope.document_group_expression is not None
                or item.scope.scope_qualifier_expressions
                or item.selection is not None
                or item.output.shape != "scalar"
                or len(item.output.field_surfaces) != 1
                or item.output.presentation != "auto"
        ):
            raise DeterministicPlanCompilerError(
                "document attribute item의 target/scope/output binding이 다릅니다")
    if intent.answer_items[0].target.surface != intent.answer_items[1].target.surface:
        raise DeterministicPlanCompilerError(
            "document attribute items는 동일 event target을 공유해야 합니다")
    if intent.answer_items[0].output.field_surfaces[0] == \
            intent.answer_items[1].output.field_surfaces[0]:
        raise DeterministicPlanCompilerError(
            "document attribute source fields는 서로 달라야 합니다")
    # Preserve both ordered items and their issuer authority for resolution
    # validation without consulting a question ID or literal company name.
    return (tuple(intent.answer_items), intent.entities[0]), None


_RANKING_FIELD = re.compile(r"(?:규모\s*)?순위|\brank(?:ing)?\b", re.IGNORECASE)


def _nary_ranking_field_indexes(item: Any) -> tuple[int, int] | None:
    """Return (ranking field, raw-values field) without inventing a surface."""

    surfaces = list(item.output.field_surfaces)
    if len(surfaces) == 1:
        return 0, 0
    if len(surfaces) != 2:
        return None
    matches = [index for index, surface in enumerate(surfaces)
               if _RANKING_FIELD.search(surface)]
    if len(matches) != 1:
        return None
    rank_index = matches[0]
    return rank_index, 1 - rank_index


def _validate_financial_comparison_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, tuple[Any, ...]]:
    """두 financial facts의 관계 intent **구조**를 검증한다.

    예전 이름은 `_validate_financial_comparison_intent` 였고, 이름만 그랬던 것이 아니라
    **값까지 한 문항에 묶여 있었다.**

    ```
    entities  != ["삼성전자", "SK하이닉스"]      → 거절
    target.surface != "매출액"                 → 거절
    period != ["2025년"] · scope != ["연결"]    → 거절
    field_surfaces != ["큰 기업", "차이"]        → 거절
    selection.criterion != "큰"                → 거절
    ```

    그래서 이 경로는 그 한 질문에만 통했고, 처음 보는 비교 질문은 전부 막혔다.
    회사·계정·기간·표현이 무엇이든 **비교의 모양이 맞으면** 통과시키도록 바꾼다.

    지키는 것은 모양뿐이다.

    - 회사 entity 정확히 둘, 그 둘을 참조하는 answer item 하나
    - `operation=compare` · `output.shape=comparison`
    - 요청 필드 둘 (승자·차이 두 답 루트가 그것에 대응한다)
    - 최대 선택이 있고 그 기준 표면이 질문에서 왔다

    값을 보지 않으므로 blind 질문에도 같은 규칙이 적용된다.
    """

    companies = [entity for entity in intent.entities
                 if entity.kind_hint == "company"]
    if len(intent.answer_items) != 1:
        raise DeterministicPlanCompilerError(
            "재무 비교는 answer item 하나만 지원합니다")
    if intent.answer_groups or intent.unresolved_mentions:
        raise DeterministicPlanCompilerError(
            "재무 비교에는 group/unresolved mention이 없어야 합니다")
    item = intent.answer_items[0]
    refs = list(item.target.entity_refs)
    if len(companies) in {2, 3, 4, 5, 6, 7, 8}:
        if any(
                premise.kind != "comparison"
                or premise.applies_to_item_ids != [item.item_id]
                for premise in intent.premises
        ) or len(intent.premises) > 1:
            raise DeterministicPlanCompilerError(
                "company comparison premise는 단일 item 비교 주장 하나만 지원합니다")
        if len(companies) > 2 and intent.premises:
            raise DeterministicPlanCompilerError(
                "N-ary company ranking에는 premise가 없어야 합니다")
        if [entity.entity_id for entity in companies] != refs:
            raise DeterministicPlanCompilerError(
                "company comparison item은 회사 entity를 질문 순서로 참조해야 합니다")
        if item.operation != "compare" or item.output.shape != "comparison":
            raise DeterministicPlanCompilerError(
                "company comparison operation/output shape가 다릅니다")
        if len(item.output.field_surfaces) not in {1, 2}:
            raise DeterministicPlanCompilerError(
                "company comparison 요청 필드는 하나 또는 둘이어야 합니다")
        if len(companies) > 2 and _nary_ranking_field_indexes(item) is None:
            raise DeterministicPlanCompilerError(
                "N-ary company ranking의 순위/원값 필드 결속이 모호합니다")
        selection = item.selection
        # 이슈 #124 — argmax 연산자는 그대로 두고 극값 방향(direction)만
        # "minimum"으로 뒤집어 승자를 반대편으로 읽는다.  다른 selection
        # 모드(top_k 등)는 여전히 fail-closed.
        if selection is None or selection.mode not in ("maximum", "minimum") \
                or not selection.criterion_surface:
            raise DeterministicPlanCompilerError(
                "company comparison에는 executable maximum/minimum selection이 "
                "필요합니다")
        if len(item.scope.target_period_expressions) != 1:
            raise DeterministicPlanCompilerError(
                "company comparison에는 대상 기간 표현 하나가 필요합니다")
        return item, tuple(companies)

    # A same-company relation is distinct from a company winner comparison:
    # the semantic operation selects derivations while the two resolved
    # financial coordinates supply their ordered operands.  One broad demand
    # may bind both roots; two explicit demands bind amount and rate separately.
    if len(companies) != 1 or refs != [companies[0].entity_id]:
        raise DeterministicPlanCompilerError(
            "time derivation에는 company entity 하나와 matching item ref가 필요합니다")
    if intent.premises:
        raise DeterministicPlanCompilerError(
            "time derivation에는 premise가 없어야 합니다")
    normalized_qualifiers = [
        re.sub(r"\s+", "", value)
        for value in item.target.qualifier_surfaces
    ]
    view_pair = (
        len(normalized_qualifiers) == 2
        and set(normalized_qualifiers) == {"최초제출값", "최신재작성값"}
    )
    allowed_field_counts = {3} if view_pair else {1, 2}
    if (item.selection is not None
            or len(item.output.field_surfaces) not in allowed_field_counts):
        raise DeterministicPlanCompilerError(
            "time derivation 요청 필드 수가 relation role과 맞지 않습니다")
    if item.operation == "retrieve":
        if item.output.shape != "scalar" or len(item.scope.target_period_expressions) < 1:
            raise DeterministicPlanCompilerError(
                "single-quarter derivation retrieve topology가 다릅니다")
    elif item.operation == "compare":
        period_count = len(item.scope.target_period_expressions)
        scope_count = len(item.scope.scope_qualifier_expressions)
        # A source-grounded annual ``증가``/``감소`` qualifier has the same
        # closed YoY meaning as ``전년비``.  This is restricted to one-period
        # compare topology; ordinary retrieve paths do not enter here.
        yoy = any(
            value.strip() in {"전년비", "증가", "감소"}
            or re.fullmatch(
                r"전년(?:도)?\s*(?:대비|보다)", value.strip())
            for value in item.target.qualifier_surfaces)
        valid_axes = (
            (period_count == 2 and scope_count <= 1)
            or (period_count == 1 and scope_count == 2)
            or (period_count == 1 and scope_count <= 1 and yoy)
            or (period_count == 1 and scope_count <= 1 and view_pair)
        )
        if item.output.shape not in {"scalar", "comparison"} or not valid_axes:
            raise DeterministicPlanCompilerError(
                "financial relation derivation topology가 다릅니다")
    else:
        raise DeterministicPlanCompilerError(
            "time derivation operation은 retrieve 또는 compare여야 합니다")
    return item, (companies[0], companies[0])


def _validate_financial_cross_company_ratio_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, tuple[Any, ...]]:
    """두 회사·같은 concept·같은 기간의 배수(concept_ratio) intent **구조**.

    이슈 #59 1단계 — 「삼성전자의 2025년 연결 매출액은 SK하이닉스의 몇
    배인가?」. `_validate_financial_comparison_intent` 의 company-comparison
    분기는 승자·차이(`selection.mode == "maximum"`, `output.shape ==
    "comparison"`)만 받는다 — 이 요청은 순위가 아니라 나눗셈 값 하나를
    묻는 것이라 답의 모양 자체가 다르다(같은 회사 concept_ratio가 이미
    쓰는 `selection is None`/scalar 모양과 같다).

    값은 보지 않는다: 회사 entity 정확히 둘, 그 둘을 참조하는 answer item
    하나, selection 없음, 수식어 없는 metric target, output.shape ==
    scalar, 요청 필드 하나, 대상 기간 표현 하나.
    """

    companies = [entity for entity in intent.entities
                 if entity.kind_hint == "company"]
    if len(intent.answer_items) != 1:
        raise DeterministicPlanCompilerError(
            "cross-company ratio는 answer item 하나만 지원합니다")
    if intent.answer_groups or intent.unresolved_mentions or intent.premises:
        raise DeterministicPlanCompilerError(
            "cross-company ratio에는 group/premise/unresolved mention이 없어야 합니다")
    item = intent.answer_items[0]
    refs = list(item.target.entity_refs)
    if len(companies) != 2 or [entity.entity_id for entity in companies] != refs:
        raise DeterministicPlanCompilerError(
            "cross-company ratio item은 회사 entity 둘을 질문 순서로 참조해야 합니다")
    if item.target.kind != "metric" or item.target.qualifier_surfaces:
        raise DeterministicPlanCompilerError(
            "cross-company ratio target은 수식어 없는 metric이어야 합니다")
    if item.operation not in ("retrieve", "compare"):
        raise DeterministicPlanCompilerError(
            "cross-company ratio operation은 retrieve 또는 compare여야 합니다")
    if item.selection is not None:
        raise DeterministicPlanCompilerError(
            "cross-company ratio에는 selection이 없어야 합니다")
    if (item.output.shape != "scalar"
            or item.output.projection_mode != "named_fields"
            or len(item.output.field_surfaces) != 1):
        raise DeterministicPlanCompilerError(
            "cross-company ratio output은 named field 하나인 scalar여야 합니다")
    if len(item.scope.target_period_expressions) != 1:
        raise DeterministicPlanCompilerError(
            "cross-company ratio에는 대상 기간 표현 하나가 필요합니다")
    return item, tuple(companies)


def _validate_financial_retrieve_comparison_intent(
        intent: SemanticIntent,
        ) -> tuple[tuple[Any, Any], Any]:
    """Validate one scalar fact followed by a scalar relation over that fact.

    The two source items remain independent answer requests, while the second
    item's typed comparison authority may safely own the shared financial
    fetches.  Selection is structural: company, metric, period, and field
    surfaces are validated only as source bindings after dispatch.
    """

    if (
            len(intent.answer_items) != 2
            or intent.answer_groups
            or intent.premises
            or intent.unresolved_mentions
    ):
        raise DeterministicPlanCompilerError(
            "financial retrieve/comparison에는 ordered item 두 개만 필요합니다")
    retrieve_item, comparison_item = intent.answer_items
    retrieve_slice = intent.model_copy(update={
        "answer_items": [retrieve_item],
    })
    comparison_slice = intent.model_copy(update={
        "answer_items": [comparison_item],
    })
    _, company = _validate_financial_intent(retrieve_slice)
    _, comparison_companies = _validate_financial_comparison_intent(
        comparison_slice)
    if comparison_companies != (company, company):
        raise DeterministicPlanCompilerError(
            "financial retrieve/comparison은 동일 company 권위를 공유해야 합니다")
    if (
            retrieve_item.target.kind != "metric"
            or comparison_item.target.kind != "metric"
            or retrieve_item.target.surface != comparison_item.target.surface
            or retrieve_item.target.entity_refs
            != comparison_item.target.entity_refs
            or comparison_item.operation != "compare"
            or comparison_item.output.shape != "scalar"
            or comparison_item.output.projection_mode != "named_fields"
            or len(comparison_item.output.field_surfaces) != 1
            or comparison_item.selection is not None
            or not comparison_item.scope.target_period_expressions
            or retrieve_item.scope.target_period_expressions[0]
            != comparison_item.scope.target_period_expressions[0]
    ):
        raise DeterministicPlanCompilerError(
            "financial retrieve/comparison source item 결속이 다릅니다")
    return (retrieve_item, comparison_item), company


def _validate_parallel_financial_retrieval_intent(
        intent: SemanticIntent,
        ) -> tuple[tuple[Any, Any], Any]:
    """Validate two independent values on one financial axis.

    This is the non-arithmetic counterpart of a CFS/SFS gap: the user asks to
    list both values, so each source item remains a scalar retrieval and no
    derivation is licensed.
    """

    if (len(intent.answer_items) != 2 or intent.answer_groups
            or intent.premises or intent.unresolved_mentions):
        raise DeterministicPlanCompilerError(
            "parallel financial retrieval에는 item 두 개만 필요합니다")
    first, second = intent.answer_items
    interest_pair = {
        re.sub(r"\s+", "", item.target.surface) for item in (first, second)
    } == {"이자비용", "이자지급액"}
    companies = []
    for item in (first, second):
        item_slice = intent.model_copy(update={"answer_items": [item]})
        _validated, company = _validate_financial_intent(item_slice)
        companies.append(company)
    if companies[0].entity_id != companies[1].entity_id:
        raise DeterministicPlanCompilerError(
            "parallel financial retrieval은 동일 company여야 합니다")
    if ((first.target.surface != second.target.surface and not interest_pair)
            or first.target.entity_refs != second.target.entity_refs
            or first.scope.target_period_expressions
            != second.scope.target_period_expressions
            or first.scope.as_of_expression != second.scope.as_of_expression
            or first.scope.document_group_expression
            != second.scope.document_group_expression):
        raise DeterministicPlanCompilerError(
            "parallel financial retrieval의 공통 좌표가 다릅니다")
    normalized_scopes = [
        tuple(re.sub(r"\s+", "", value) for value in
              item.scope.scope_qualifier_expressions)
        for item in (first, second)
    ]
    if interest_pair:
        if normalized_scopes[0] != normalized_scopes[1] or normalized_scopes[0] not in (
                ("연결",), ("별도",), ("개별",)):
            raise DeterministicPlanCompilerError(
                "interest expense/payment pair에는 동일 명시 scope가 필요합니다")
    elif set(normalized_scopes) not in (
            {("연결",), ("별도",)}, {("연결",), ("개별",)}):
        raise DeterministicPlanCompilerError(
            "parallel financial retrieval에는 연결/별도 scope가 필요합니다")
    return (first, second), companies[0]


def _validate_summary_metric_fanout_intent(
        intent: SemanticIntent,
        ) -> tuple[tuple[Any, ...], Any]:
    """Validate N sibling scalar retrievals that share one colloquial surface.

    이슈 #94 25. 「얼마나 벌었어」처럼 표기 하나가 후보 개념 여럿을 가리킬 때,
    되묻는 대신 후보를 **전부** 답한다. 그 intent 는 `SummaryMetricFanoutRegrounder`
    가 item 하나를 후보 수만큼 복제해 만든다 — 표면을 개념 이름으로 바꾸면
    질문에 없는 낱말이라 결속 검사가 거절하므로 **표면은 전부 같다.**

    그래서 여기서 볼 것은 「형제가 정말 같은 모양인가」다. 어느 item 이 어느
    개념을 맡는지는 표면으로 구분할 수 없고 **형제 순서**가 정한다.

    2-item 규격(`_validate_parallel_financial_retrieval_intent`)과 겹치지 않는다.
    그쪽은 연결/별도라 scope 가 **달라야** 하고, 이쪽은 scope 까지 같다.

    ``intent.presentation`` 이 ``"auto"`` 여야 한다 — `SummaryMetricFanoutRegrounder`
    가 늘 그렇게 만든다. #58 2단계의 「최근 N년」기간 fanout(같은 회사·
    scalar·표면 반복이라 형제 모양 검사만으로는 이것과 항목 구조가 구분되지
    않는다)이 ``"table"`` 을 쓰는 것도 이 자리에서 갈라 둔다 — 아니면
    `_selected_handler` 의 fallback(정확한 signature 가 없을 때 kind
    inventory로 후보를 좁히고 의미 검증을 다시 묻는 경로)이 두 규격을 모두
    통과시켜 ambiguous 오류로 거절한다(직접 확인함).
    """

    items = tuple(intent.answer_items)
    if (len(items) < 3 or intent.answer_groups or intent.premises
            or intent.unresolved_mentions or intent.presentation != "auto"):
        raise DeterministicPlanCompilerError(
            "summary metric fanout에는 item 세 개 이상과 auto presentation이 "
            "필요합니다")
    companies = []
    for item in items:
        item_slice = intent.model_copy(update={"answer_items": [item]})
        _validated, company = _validate_financial_intent(item_slice)
        companies.append(company)
    if len({company.entity_id for company in companies}) != 1:
        raise DeterministicPlanCompilerError(
            "summary metric fanout은 동일 company여야 합니다")
    first = items[0]
    for item in items[1:]:
        if (
                item.target.surface != first.target.surface
                or item.target.kind != first.target.kind
                or item.target.entity_refs != first.target.entity_refs
                or item.target.qualifier_surfaces
                != first.target.qualifier_surfaces
                or item.operation != first.operation
                or item.selection != first.selection
                or item.scope.target_period_expressions
                != first.scope.target_period_expressions
                or item.scope.as_of_expression != first.scope.as_of_expression
                or item.scope.document_group_expression
                != first.scope.document_group_expression
                or item.scope.scope_qualifier_expressions
                != first.scope.scope_qualifier_expressions
                or item.output.shape != first.output.shape
                or item.output.projection_mode != first.output.projection_mode
                or item.output.field_surfaces != first.output.field_surfaces
                or item.output.presentation != first.output.presentation
        ):
            raise DeterministicPlanCompilerError(
                "summary metric fanout item이 형제 모양이 아닙니다")
    return items, companies[0]


#: 이 fanout이 받는 ``output.shape``. HCX-007 실호출로 직접 확인함 —
#: 「최근 N개년 …」류는 ``scalar``, 「… 추이」류는 ``timeline``으로 낸다.
#: registry 구조 서명도 이 둘을 각각 등록한다.
_RECENT_PERIODS_FANOUT_SHAPES = frozenset({"scalar", "timeline"})


def _recent_quarter_resolution_kind_patterns(
        item_count: int,
        ) -> "tuple[tuple[str, ...], ...]":
    """Ordered direct/Q4 kind inventories for sequential quarter windows."""

    patterns: list[tuple[str, ...]] = []
    for starting_quarter in range(1, 5):
        quarters = [
            ((starting_quarter - 1 + offset) % 4) + 1
            for offset in range(item_count)
        ]
        kinds = tuple(
            "financial_comparison" if quarter == 4 else "financial"
            for quarter in quarters)
        if "financial_comparison" in kinds and kinds not in patterns:
            patterns.append(kinds)
    return tuple(patterns)


def _validate_recent_periods_fanout_intent(
        intent: SemanticIntent,
        ) -> tuple[tuple[Any, ...], Any]:
    """Validate 2~5 sibling scalar retrievals that share one relative-period surface.

    이슈 #58 2단계. 「최근 N년」「최근 N개 분기」는 실제 연도·분기가 질문에
    리터럴로 없어 grounding이 목록으로 펼치지 못한다(`agent.relative_
    financial_period` 모듈 docstring). `RecentPeriodsFanoutRegrounder`가
    item 하나를 N개로 복제해 같은 리터럴을 반복하고, 형제 순서로 실제
    기간을 나눠 맡긴다 — `SummaryMetricFanoutRegrounder`가 개념을 나눠
    맡기는 것과 같은 구조다.

    개념 fanout과 항목 모양이 구조적으로 같으므로(회사 하나·재무 스칼라
    하나·같은 표면 반복) ``intent.presentation``(top-level)을 ``"table"``
    로 못박아 structural signature를 개념 fanout(``"auto"``)과 갈라 둔다 —
    그러지 않으면 두 규격이 같은 signature/kind inventory 쌍을 등록하게
    되어 `_selected_handler`가 컴파일 오류로 거절한다(직접 확인함).

    **회사 추출에 ``_validate_financial_intent`` 를 재사용하지 않는 이유.**
    HCX-007 실호출로 직접 확인함 — 「…추이」류 질문은 ``output.shape`` 를
    ``scalar`` 가 아니라 ``timeline`` 으로 낸다(위 규격 등록도 그 값을 함께
    받는다). ``_validate_financial_intent`` 는 일반 단일-재무 경로 전체가
    공유하는 검증기라 ``scalar`` 만 받도록 못박혀 있다 — 여기서 그대로
    쓰면 그 경로 전체를 헐겁게 하지 않고는 ``timeline`` 을 통과시킬 수
    없다. 대신 회사 결속만 직접 확인하고, 나머지(단일 필드·selection
    없음·기간 표현 하나)는 아래에서 ``first`` 에 대해 직접 확인한다.
    """

    items = tuple(intent.answer_items)
    if (
            len(items) < 2 or len(items) > 5
            or intent.answer_groups or intent.premises
            or intent.unresolved_mentions
            or intent.presentation != "table"
    ):
        raise DeterministicPlanCompilerError(
            "recent periods fanout에는 item 2~5개와 table presentation이 "
            "필요합니다")
    companies = [row for row in intent.entities if row.kind_hint == "company"]
    if len(companies) != 1:
        raise DeterministicPlanCompilerError(
            "recent periods fanout에는 회사 entity 하나가 필요합니다")
    company = companies[0]
    first = items[0]
    if (
            first.target.kind != "metric"
            or first.target.entity_refs != [company.entity_id]
            or first.operation != "retrieve"
            or first.selection is not None
            or first.output.shape not in _RECENT_PERIODS_FANOUT_SHAPES
            or first.output.projection_mode != "named_fields"
            or len(first.output.field_surfaces) != 1
            or len(first.scope.target_period_expressions) != 1
    ):
        raise DeterministicPlanCompilerError(
            "recent periods fanout item topology가 지원 범위를 벗어납니다")
    for item in items[1:]:
        if (
                item.target.surface != first.target.surface
                or item.target.kind != first.target.kind
                or item.target.entity_refs != first.target.entity_refs
                or item.target.qualifier_surfaces
                != first.target.qualifier_surfaces
                or item.operation != first.operation
                or item.selection != first.selection
                or item.scope.target_period_expressions
                != first.scope.target_period_expressions
                or item.scope.as_of_expression != first.scope.as_of_expression
                or item.scope.document_group_expression
                != first.scope.document_group_expression
                or item.scope.scope_qualifier_expressions
                != first.scope.scope_qualifier_expressions
                or item.output.shape != first.output.shape
                or item.output.projection_mode != first.output.projection_mode
                or item.output.field_surfaces != first.output.field_surfaces
                or item.output.presentation != first.output.presentation
        ):
            raise DeterministicPlanCompilerError(
                "recent periods fanout item이 형제 모양이 아닙니다")
    return items, company


@dataclass(frozen=True, slots=True)
class _RecentPeriodLogicalCoordinate:
    """The one user-visible period produced by a recent-period item."""

    corp_code: str
    corp_name: str
    concept: FinancialConcept
    period_start: date
    period_end: date
    period_type: str
    scope: str
    statement: str
    view: str
    as_of: str


def _recent_period_logical_coordinate(
        resolved: Any) -> "_RecentPeriodLogicalCoordinate | None":
    """Project a direct fact or cumulative subtraction to one period.

    Recent-quarter fanout may cross a calendar Q4.  DART does not report Q4
    as an independent income-statement row, so the resolver emits FY and 9M
    cumulative operands.  This projection lets the registry and authority
    validators compare the *resulting quarter* with adjacent direct quarters
    without weakening ordinary financial-comparison validation.
    """

    if isinstance(resolved, FinancialResolution):
        if resolved.period_start is None:
            return None
        return _RecentPeriodLogicalCoordinate(
            corp_code=resolved.corp_code, corp_name=resolved.corp_name,
            concept=resolved.concept, period_start=resolved.period_start,
            period_end=resolved.period_end, period_type=resolved.period_type,
            scope=resolved.scope, statement=resolved.statement,
            view=resolved.view, as_of=resolved.as_of)
    if not isinstance(resolved, FinancialComparisonResolution):
        return None
    if (
            resolved.requested_operators != ["discrete_from_cumulative"]
            or len(resolved.operands) != 2
            or resolved.verification_claim is not None
            or resolved.verification_premise_id is not None
            or resolved.presentation is not None
    ):
        return None
    current, prior = resolved.operands
    shared = (
        "corp_code", "corp_name", "concept", "scope", "statement", "view",
    )
    if (
            any(getattr(current, field) != getattr(prior, field)
                for field in shared)
            or current.cumulative is not True
            or prior.cumulative is not True
            or current.period_start is None
            or prior.period_start is None
            or current.period_start != prior.period_start
            or current.period_start != date(current.period_end.year, 1, 1)
            or current.period_end.year != prior.period_end.year
    ):
        return None
    allowed_endpoints = {
        (date(current.period_end.year, 6, 30),
         date(current.period_end.year, 3, 31), "half", "quarter"),
        (date(current.period_end.year, 9, 30),
         date(current.period_end.year, 6, 30), "quarter", "half"),
        (date(current.period_end.year, 12, 31),
         date(current.period_end.year, 9, 30), "annual", "quarter"),
    }
    if (
            current.period_end, prior.period_end,
            current.period_type, prior.period_type,
    ) not in allowed_endpoints:
        return None
    return _RecentPeriodLogicalCoordinate(
        corp_code=current.corp_code, corp_name=current.corp_name,
        concept=current.concept,
        period_start=prior.period_end + timedelta(days=1),
        period_end=current.period_end, period_type="quarter",
        scope=current.scope, statement=current.statement,
        view=current.view, as_of=current.as_of)


def _is_recent_periods_fanout_resolution(
        resolution: AuthoritativeResolution) -> bool:
    """이 resolution이 직접값/파생값 혼합 최근 기간 fanout 모양인가.

    자취(trace)가 없는 audit 재검증 지점(`_validate_resolution_slice`·
    `_validate_slice_cross_authority`)은 어느 handler가 이 resolution을
    만들었는지 모르고 오직 ``resolution.items`` 만 본다. 그래서 여기서도
    좌표 자체로 다시 판단한다 — **개념이 전부 같고 기간이 전부 다르면**
    (N==2 포함) 이 fanout이다. CFS/SFS 짝은 기간이 같고 scope 만 갈리므로
    여기 걸리지 않는다. 요약 구어 fanout(#94 25)은 반대로 기간이 같고
    개념이 갈리므로 역시 걸리지 않는다.
    """

    coordinates = [
        _recent_period_logical_coordinate(row.resolution)
        for row in resolution.items
    ]
    if len(coordinates) < 2 or any(row is None for row in coordinates):
        return False
    concrete = [row for row in coordinates if row is not None]
    if len({row.concept for row in concrete}) != 1:
        return False
    return len({(row.period_start, row.period_end)
                for row in concrete}) == len(concrete)


def _is_recent_periods_fanout_intent_items(intent_items: Any) -> bool:
    """Recognise the sibling authority without relying on handler provenance."""

    if not isinstance(intent_items, tuple) or not (2 <= len(intent_items) <= 5):
        return False
    first = intent_items[0]
    if (
            first.target.kind != "metric"
            or first.operation != "retrieve"
            or first.selection is not None
            or first.output.shape not in _RECENT_PERIODS_FANOUT_SHAPES
            or len(first.scope.target_period_expressions) != 1
            or len(first.output.field_surfaces) != 1
    ):
        return False
    return all(
        item.target == first.target
        and item.operation == first.operation
        and item.scope == first.scope
        and item.selection == first.selection
        and item.output == first.output
        for item in intent_items[1:])


def _is_calendar_recent_period(
        coordinate: _RecentPeriodLogicalCoordinate) -> bool:
    """Whether a logical coordinate is one complete calendar year/quarter."""

    year = coordinate.period_end.year
    if coordinate.period_type == "annual":
        return (
            coordinate.period_start == date(year, 1, 1)
            and coordinate.period_end == date(year, 12, 31)
        )
    if coordinate.period_type != "quarter":
        return False
    endpoints = {
        (date(year, 1, 1), date(year, 3, 31)),
        (date(year, 4, 1), date(year, 6, 30)),
        (date(year, 7, 1), date(year, 9, 30)),
        (date(year, 10, 1), date(year, 12, 31)),
    }
    return (coordinate.period_start, coordinate.period_end) in endpoints


def _is_q4_cumulative_subtraction(resolved: Any) -> bool:
    """Limit mixed recent-quarter fanout derivations to FY minus 9M."""

    if not isinstance(resolved, FinancialComparisonResolution):
        return False
    current, prior = resolved.operands
    year = current.period_end.year
    return (
        resolved.requested_operators == ["discrete_from_cumulative"]
        and current.period_start == date(year, 1, 1)
        and prior.period_start == date(year, 1, 1)
        and current.period_end == date(year, 12, 31)
        and prior.period_end == date(year, 9, 30)
        and current.period_type == "annual"
        and prior.period_type == "quarter"
        and current.cumulative is True
        and prior.cumulative is True
    )


def _validate_recent_periods_fanout_resolution(
        intent_items: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    """Close N direct/derived financial authorities differing only by period.

    개념 fanout(`_validate_summary_metric_fanout_resolution`)의 정반대
    축이다 — 거기서는 기간이 같고 개념이 갈려야 하고, 여기서는 개념(과
    scope·재무제표)이 같고 기간만 갈려야 한다. 기간까지 같으면 같은
    사실을 되풀이한 것이라 거절한다.
    """

    if not isinstance(intent_items, tuple) or not (2 <= len(intent_items) <= 5):
        raise DeterministicPlanCompilerError(
            "recent periods fanout intent item 2~5개가 필요합니다")
    if (len(resolution.items) != len(intent_items)
            or not set(row.resolution.kind for row in resolution.items)
            .issubset({"financial", "financial_comparison"})
            or [row.item_id for row in resolution.items]
            != [row.item_id for row in intent_items]
            or resolution.premise_proofs):
        raise DeterministicPlanCompilerError(
            "recent periods fanout resolution inventory가 다릅니다")
    for source_item, resolved_item in zip(
            intent_items, resolution.items, strict=True):
        one = resolution.model_copy(update={"items": [resolved_item]})
        if isinstance(resolved_item.resolution, FinancialResolution):
            _validate_financial_resolution(source_item, one)
        elif isinstance(
                resolved_item.resolution, FinancialComparisonResolution):
            _validate_financial_comparison_resolution(source_item, one)
        else:
            raise DeterministicPlanCompilerError(
                "recent periods fanout typed branch가 다릅니다")
    coordinates = [
        _recent_period_logical_coordinate(row.resolution)
        for row in resolution.items
    ]
    if any(row is None for row in coordinates):
        raise DeterministicPlanCompilerError(
            "recent periods fanout 파생 좌표가 닫히지 않았습니다")
    concrete = [row for row in coordinates if row is not None]
    first = concrete[0]
    # ``cumulative`` 는 여기서 뺀다 — 값이 아니라 그 분기 숫자가 정본에
    # **어떻게 태깅됐는지**(1분기는 누적==단독이라 종종 ``True``, 2~4분기
    # discrete는 ``False``)라 기간마다 자연히 갈린다(SK하이닉스 매출액
    # 「최근 4개 분기」에서 직접 확인함 — 2026 1Q만 cumulative=True, 나머지
    # 셋은 False). ``period_type`` 이 이미 "전부 같은 분기 단위"를 보장한다.
    shared = (
        "corp_code", "corp_name", "concept", "period_type", "view",
        "scope", "statement",
    )
    if any(getattr(row, field) != getattr(first, field)
           for row in concrete[1:] for field in shared):
        raise DeterministicPlanCompilerError(
            "recent periods fanout의 canonical 좌표가 기간 말고도 다릅니다")
    if not all(_is_calendar_recent_period(row) for row in concrete):
        raise DeterministicPlanCompilerError(
            "recent periods fanout은 완전한 달력 연도 또는 분기여야 합니다")
    comparisons = [
        row.resolution for row in resolution.items
        if isinstance(row.resolution, FinancialComparisonResolution)
    ]
    if comparisons and (
            first.period_type != "quarter"
            or not all(_is_q4_cumulative_subtraction(row)
                       for row in comparisons)):
        raise DeterministicPlanCompilerError(
            "recent periods 혼합 fanout은 FY-9M으로 만든 4분기만 허용합니다")
    period_keys = [(row.period_start, row.period_end) for row in concrete]
    if len(set(period_keys)) != len(period_keys):
        raise DeterministicPlanCompilerError(
            "recent periods fanout이 같은 기간을 되풀이합니다")
    if period_keys != sorted(period_keys):
        raise DeterministicPlanCompilerError(
            "recent periods fanout 기간이 과거에서 최근 순서가 아닙니다")
    if any(
            previous.period_end + timedelta(days=1) != following.period_start
            for previous, following in zip(concrete, concrete[1:])
    ):
        raise DeterministicPlanCompilerError(
            "recent periods fanout 기간이 연속된 연도 또는 분기가 아닙니다")
    return resolution.items[0], None


def _validate_explicit_periods_fanout_intent(
        intent: SemanticIntent,
        ) -> tuple[tuple[Any, ...], Any]:
    """Validate 2~5 sibling scalar retrievals, each already grounded to its
    own distinct literal period.

    이슈 #171 M16. 「삼성전자의 2023년, 2024년, 2025년 연결 매출액을 각각
    알려줘」— 같은 회사·같은 개념을 놓고 형제 item마다 **이미 확정된, 서로
    다른** 리터럴 기간(``target_period_expressions``)을 낼 때가 있다(연간·
    분기 혼합도 흔하다, 예: 2024년/2025년/2026년 1분기). 위
    `_validate_summary_metric_fanout_intent`(#94 25, 개념 fanout)의 유일한
    3-item 재무 handler는 형제 전원 **동일** 기간을 요구해 이 모양을
    거절하고, `_validate_recent_periods_fanout_intent`(#58 2단계)는
    정반대로 형제 전원이 같은 **상대** 기간 표현(「최근 N년」)을 반복해야
    편다 — 이 fanout은 리터럴이 형제마다 이미 다르므로 그 둘 다와 다르다.

    `ExplicitPeriodsFanoutRegrounder`(agent/stage1_v1_backend_composition.py)
    는 값을 하나도 계산하지 않는다 — 이미 닫힌 형제 N개를 그대로 두고
    ``intent.presentation`` 만 ``"list"`` 로 못박아 개념 fanout(``"auto"``)
    ·recent-periods-fanout(``"table"``)과 구조 서명을 가른다. 그러지
    않으면 세 규격이 같은 signature/kind inventory 쌍을 등록해
    `_selected_handler` 가 컴파일 오류로 거절한다(#58 2단계와 같은 이유,
    직접 확인함).
    """

    items = tuple(intent.answer_items)
    if (
            len(items) < 2 or len(items) > 5
            or intent.answer_groups or intent.premises
            or intent.unresolved_mentions
            or intent.presentation != "list"
    ):
        raise DeterministicPlanCompilerError(
            "explicit periods fanout에는 item 2~5개와 list presentation이 "
            "필요합니다")
    companies = [row for row in intent.entities if row.kind_hint == "company"]
    if len(companies) != 1:
        raise DeterministicPlanCompilerError(
            "explicit periods fanout에는 회사 entity 하나가 필요합니다")
    company = companies[0]
    first = items[0]
    if (
            first.target.kind != "metric"
            or first.target.entity_refs != [company.entity_id]
            or first.operation != "retrieve"
            or first.selection is not None
            or first.output.shape not in _RECENT_PERIODS_FANOUT_SHAPES
            or first.output.projection_mode != "named_fields"
            or len(first.output.field_surfaces) != 1
            or len(first.scope.target_period_expressions) != 1
    ):
        raise DeterministicPlanCompilerError(
            "explicit periods fanout item topology가 지원 범위를 벗어납니다")
    period_keys = {tuple(first.scope.target_period_expressions)}
    for item in items[1:]:
        if (
                item.target.surface != first.target.surface
                or item.target.kind != first.target.kind
                or item.target.entity_refs != first.target.entity_refs
                or item.target.qualifier_surfaces
                != first.target.qualifier_surfaces
                or item.operation != first.operation
                or item.selection != first.selection
                or len(item.scope.target_period_expressions) != 1
                or item.scope.as_of_expression != first.scope.as_of_expression
                or item.scope.document_group_expression
                != first.scope.document_group_expression
                or item.scope.scope_qualifier_expressions
                != first.scope.scope_qualifier_expressions
                or item.output.shape != first.output.shape
                or item.output.projection_mode != first.output.projection_mode
                or item.output.field_surfaces != first.output.field_surfaces
                or item.output.presentation != first.output.presentation
        ):
            raise DeterministicPlanCompilerError(
                "explicit periods fanout item이 형제 모양이 아닙니다")
        period_key = tuple(item.scope.target_period_expressions)
        if period_key in period_keys:
            raise DeterministicPlanCompilerError(
                "explicit periods fanout이 같은 기간을 되풀이합니다")
        period_keys.add(period_key)
    return items, company


def _validate_explicit_periods_fanout_resolution(
        intent_items: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    """Close N scalar financial authorities that differ **only** by period —
    including a mixed annual/quarter period axis.

    `_validate_recent_periods_fanout_resolution`(#58 2단계)의 **거의** 같은
    복제다 — 유일한 차이는 좌표 동일성 검사(``shared``)에서
    ``period_type`` 을 뺀 것이다. 그 fanout은 「최근 N년」이든 「최근 N개
    분기」든 형제가 **한 축**만(전부 연간이거나 전부 분기이거나) 펴므로
    ``period_type`` 이 전부 같은 것도 안전하게 요구할 수 있었다. 이슈
    #171 M16(「2024년/2025년/2026년 1분기」처럼 연간·분기가 섞인 리터럴
    나열)은 형제마다 이미 다른, 서로 다른 축의 기간을 낼 수 있으므로 그
    요구를 걸면 정상적인 혼합 질문이 거절된다.

    `agent.deterministic_plan_compiler_v1._validate_resolution_slice`(qid
    없는 audit 재검증)의 kind-only 재확인도 이 함수를 쓴다 —
    `_validate_recent_periods_fanout_resolution`의 승인 기준을 포함하는
    상위집합이므로(둘 다 concept·scope·statement 등은 요구하고, 이 함수만
    period_type을 안 본다), recent-periods-fanout(#58 2단계)이 실제로
    만드는(늘 단일 축) resolution도 그대로 통과한다 — 그 기존 시험은
    변하지 않는다.
    """

    if not isinstance(intent_items, tuple) or not (2 <= len(intent_items) <= 5):
        raise DeterministicPlanCompilerError(
            "explicit periods fanout intent item 2~5개가 필요합니다")
    if (len(resolution.items) != len(intent_items)
            or set(row.resolution.kind for row in resolution.items)
            != {"financial"}
            or [row.item_id for row in resolution.items]
            != [row.item_id for row in intent_items]
            or resolution.premise_proofs):
        raise DeterministicPlanCompilerError(
            "explicit periods fanout resolution inventory가 다릅니다")
    for source_item, resolved_item in zip(
            intent_items, resolution.items, strict=True):
        _validate_financial_resolution(
            source_item,
            resolution.model_copy(update={"items": [resolved_item]}),
        )
    coordinates = [row.resolution for row in resolution.items]
    if not all(isinstance(row, FinancialResolution) for row in coordinates):
        raise DeterministicPlanCompilerError(
            "explicit periods fanout typed branch가 다릅니다")
    first = coordinates[0]
    # ``cumulative``·``period_type`` 둘 다 여기서 뺀다. ``cumulative``는
    # 위 recent-periods-fanout과 같은 이유(그 분기 숫자가 정본에 **어떻게
    # 태깅됐는지**라 기간마다 자연히 갈린다). ``period_type``은 이 fanout만
    # 다른 이유(위 docstring) — 연간·분기가 형제마다 자연히 갈릴 수 있다.
    shared = (
        "corp_code", "corp_name", "concept", "view", "as_of",
        "scope", "statement",
    )
    if any(getattr(row, field) != getattr(first, field)
           for row in coordinates[1:] for field in shared):
        raise DeterministicPlanCompilerError(
            "explicit periods fanout의 canonical 좌표가 기간 말고도 다릅니다")
    period_keys = [(row.period_start, row.period_end) for row in coordinates]
    if len(set(period_keys)) != len(period_keys):
        raise DeterministicPlanCompilerError(
            "explicit periods fanout이 같은 기간을 되풀이합니다")
    return resolution.items[0], None


def _validate_parallel_annual_change_intent(
        intent: SemanticIntent,
        ) -> tuple[tuple[Any, Any], Any]:
    """Validate split amount/rate items over one identical annual axis."""

    if (
            len(intent.entities) != 1
            or intent.entities[0].kind_hint != "company"
            or len(intent.answer_items) != 2
            or intent.answer_groups
            or intent.premises
            or intent.unresolved_mentions
    ):
        raise DeterministicPlanCompilerError(
            "parallel annual change intent inventory가 다릅니다")
    first, second = intent.answer_items
    if any(
            item.target.kind != "metric"
            or item.target.entity_refs != [intent.entities[0].entity_id]
            or item.output.shape != "scalar"
            or item.output.projection_mode != "named_fields"
            or len(item.output.field_surfaces) != 1
            or len(item.scope.target_period_expressions) != 2
            or item.selection is not None
            for item in (first, second)
    ):
        raise DeterministicPlanCompilerError(
            "parallel annual change item topology가 다릅니다")
    if (
            [first.operation, second.operation] != ["retrieve", "compare"]
            or first.target.surface != second.target.surface
            or first.target.qualifier_surfaces != second.target.qualifier_surfaces
            or first.scope != second.scope
    ):
        raise DeterministicPlanCompilerError(
            "parallel annual change source axes가 다릅니다")
    return (first, second), intent.entities[0]


def _validate_recovered_single_quarter_derivation_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, tuple[Any, Any]]:
    """Validate the qualifier-carried single-quarter derivation topology.

    Some semantic intents retain the only period expression on the metric
    target rather than the generic scope period list.  The financial backend
    may return a two-endpoint authority only after independently proving the
    user's explicit single-quarter or cumulative-subtraction wording.  The
    compiler therefore validates the *shape* here, without re-parsing Korean
    or depending on a particular company, year, account, or question ID.
    """

    companies = [entity for entity in intent.entities
                 if entity.kind_hint == "company"]
    if len(companies) != 1 or len(intent.answer_items) != 1:
        raise DeterministicPlanCompilerError(
            "qualifier single-quarter derivation에는 company와 item이 각각 하나 필요합니다")
    if intent.answer_groups or intent.premises or intent.unresolved_mentions:
        raise DeterministicPlanCompilerError(
            "qualifier single-quarter derivation에는 group/premise/unresolved가 없어야 합니다")
    item = intent.answer_items[0]
    if (
            item.target.kind != "metric"
            or item.target.entity_refs != [companies[0].entity_id]
            or len(item.target.qualifier_surfaces) != 1
            or item.operation != "retrieve"
            or item.scope.target_period_expressions
            or item.scope.as_of_expression is not None
            or item.scope.document_group_expression is not None
            or item.scope.scope_qualifier_expressions
            or item.selection is not None
            or item.output.shape != "scalar"
            or item.output.projection_mode != "named_fields"
            or len(item.output.field_surfaces) != 1
            or item.output.presentation != "auto"
    ):
        raise DeterministicPlanCompilerError(
            "qualifier single-quarter derivation topology가 다릅니다")
    return item, (companies[0], companies[0])


def _validate_financial_resolution(
        intent_item: Any,
        resolution: AuthoritativeResolution,
        *, expected_premise_ids: list[str] | None = None,
        ) -> tuple[Any, str | None]:
    """단일 재무 권위의 **구조**를 검증한다.

    예전에는 좌표를 통째로 승인값과 대조했다.

    ```
    coordinate.corp_code   != "00126380"
    coordinate.period_start != date(2025, 1, 1)
    coordinate.scope       != "CFS"
    coordinate.as_of       != "20260619"
    proof.surface          != "얼마"
    item.resolution.corp_name != "삼성전자"
    ```

    그래서 「삼성전자의 2025년 연결 매출」 딱 하나만 통과했고, **처음 보는 재무
    질문은 좌표가 옳아도 거절**됐다. 이것은 일반 경로에 등록된 검증기이므로
    거기에 한 문항의 정답이 박혀 있으면 경로 전체가 그 문항 전용이 된다.

    여기서 보는 것은 권위가 **source intent 와 결속돼 있는가**뿐이다. 좌표의 값이
    옳은지는 정본 조회가 답할 일이고, 컴파일러가 정답을 들고 있으면 그 순간
    fixture 적응이 된다.
    """

    if len(resolution.items) != 1:
        raise DeterministicPlanCompilerError("resolution item order/count가 다릅니다")
    item = resolution.items[0]
    if item.target_surface != intent_item.target.surface:
        raise DeterministicPlanCompilerError("resolution target surface가 source와 다릅니다")
    if len(item.field_proofs) != len(intent_item.output.field_surfaces):
        raise DeterministicPlanCompilerError("resolution field proof count가 다릅니다")
    for index, (proof, surface) in enumerate(zip(
            item.field_proofs, intent_item.output.field_surfaces, strict=True)):
        if proof.source_field_index != index or proof.surface != surface:
            raise DeterministicPlanCompilerError(
                "resolution source field index/surface가 source와 다릅니다")
    actual_premise_ids = [row.premise_id for row in resolution.premise_proofs]
    if expected_premise_ids is None:
        if actual_premise_ids:
            raise DeterministicPlanCompilerError(
                "bounded slice에는 resolution premise proof가 없습니다")
    elif actual_premise_ids != expected_premise_ids:
        raise DeterministicPlanCompilerError(
            "financial premise proof inventory가 source premise 순서와 다릅니다")
    coordinate = item.resolution
    if not isinstance(coordinate, FinancialResolution):
        raise DeterministicPlanCompilerError(
            "financial resolution kind가 다릅니다")
    if not coordinate.corp_name.strip():
        raise DeterministicPlanCompilerError("resolution corp_name이 비었습니다")

    defaults = item.applied_defaults
    if not defaults:
        return item, None
    # **기본값을 썼으면 무엇을 왜 썼는지가 남아야 한다.**
    #
    # 예전에는 Gold overlay 를 열어 그 근거 문자열과 같은지 봤다. 그러면 정본
    # 근거로 채운 권위는 거절되고, resolver 는 정답지를 읽어야만 통과한다.
    # 근거의 **출처**는 검증하되 그 값을 정답지에서 가져오지 않는다.
    if len(defaults) != len({default.policy for default in defaults}):
        raise DeterministicPlanCompilerError(
            "재무 기본값 정책은 중복될 수 없습니다")
    unsupported = {
        default.policy for default in defaults
        if default.policy not in {
            "primary_statement_scope", "canonical_company_alias_merge"}
    }
    if unsupported:
        raise DeterministicPlanCompilerError(
            f"지원하지 않는 재무 기본값 정책입니다: {sorted(unsupported)}")
    primary = next((default for default in defaults
                    if default.policy == "primary_statement_scope"), None)
    alias = next((default for default in defaults
                  if default.policy == "canonical_company_alias_merge"), None)
    # 기본값의 값은 정본의 해당 회사·기간 primary-statement 조회로만 뒷받침한다.
    # 정답지·fixture·overlay 문자열은 evidence provenance가 될 수 없다.
    expected_evidence = {
        "canonical:primary-statement:"
        f"{coordinate.corp_code}:{coordinate.period_end.isoformat()}"
    }
    if primary is not None:
        if primary.value != coordinate.scope:
            raise DeterministicPlanCompilerError(
                "기본값이 실제 좌표의 scope 와 다릅니다")
        if primary.basis != (
                "canonical primary statement for the company and period"):
            raise DeterministicPlanCompilerError(
                "재무 기본값 basis가 canonical primary-statement 정책과 다릅니다")
        if set(primary.evidence_refs) != expected_evidence:
            raise DeterministicPlanCompilerError(
                "재무 기본값 evidence ref가 canonical primary-statement 좌표와 다릅니다")
    if alias is not None:
        try:
            alias_value = json.loads(alias.value)
        except (TypeError, ValueError) as exc:
            raise DeterministicPlanCompilerError(
                "회사 alias provenance 값이 JSON object가 아닙니다") from exc
        if not isinstance(alias_value, dict):
            raise DeterministicPlanCompilerError(
                "회사 alias provenance 값이 JSON object가 아닙니다")
        surfaces = alias_value.get("surfaces")
        if (
                alias.basis != (
                    "question surfaces resolve uniquely to one canonical corp_code")
                or alias_value.get("corp_code") != coordinate.corp_code
                or alias_value.get("corp_name") != coordinate.corp_name
                or not isinstance(surfaces, list) or len(surfaces) < 2
                or len(surfaces) != len(set(surfaces))
                or any(not isinstance(surface, str) or not surface
                       for surface in surfaces)
                or alias.evidence_refs != [
                    f"canonical:company:{coordinate.corp_code}",
                    *[f"question:company-alias:{index}"
                      for index in range(1, len(surfaces) + 1)],
                ]
        ):
            raise DeterministicPlanCompilerError(
                "회사 alias provenance가 canonical coordinate와 다릅니다")
    # 기본값 provenance는 AppliedDefault 자체와 canonical evidence 좌표에 모두
    # 남는다. 평가용 overlay snapshot은 production lowering의 입력 권위가
    # 아니므로 compiled slice에 binding하지 않는다.
    return item, None


def _validate_document_fact_resolution(
        intent_items: tuple[Any, Any],
        resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    """Validate generic source/proof relations for the two-item slice."""
    if len(resolution.items) != 2 \
            or [row.item_id for row in resolution.items] != ["item-1", "item-2"]:
        raise DeterministicPlanCompilerError(
            "document fact comparison resolution item inventory가 다릅니다")
    if not isinstance(intent_items, tuple) or len(intent_items) != 2:
        raise DeterministicPlanCompilerError(
            "document fact comparison intent item pair가 필요합니다")
    first_intent, second_intent = intent_items
    for row, source in zip(
            resolution.items, (first_intent, second_intent), strict=True):
        if row.target_surface != source.target.surface \
                or len(row.field_proofs) != 1 \
                or row.applied_defaults:
            raise DeterministicPlanCompilerError(
                "document fact comparison resolution target/field/default가 source와 다릅니다")
        proof = row.field_proofs[0]
        if proof.source_field_index != 0 \
                or proof.surface != source.output.field_surfaces[0]:
            raise DeterministicPlanCompilerError(
                "document fact comparison source field proof가 다릅니다")
    proof_refs = [row.field_proofs[0].proof_ref for row in resolution.items]
    _unique(proof_refs, "document fact resolution field proof_ref")

    comparison = resolution.items[0].resolution
    reason = resolution.items[1].resolution
    if not isinstance(comparison, DocumentFactComparisonResolution) \
            or not isinstance(reason, DocumentReasonEvidenceResolution):
        raise DeterministicPlanCompilerError(
            "document fact comparison resolution kind inventory가 다릅니다")
    operands = comparison.operands
    evidence = reason.evidence
    if len(operands) != 2:
        raise DeterministicPlanCompilerError(
            "document fact comparison에는 정확히 두 operand가 필요합니다")
    if operands[0].source_class != "correction" \
            or operands[1].source_class != "disclosure":
        raise DeterministicPlanCompilerError(
            "document fact comparison operand source_class 순서가 다릅니다")
    bound = [row for row in operands if row.operand_id == evidence.operand_id]
    if len(bound) != 1 or evidence.operand_id != operands[1].operand_id:
        raise DeterministicPlanCompilerError(
            "reason evidence는 ordered second comparison operand에 결속되어야 합니다")
    operand = bound[0]
    if (
            (evidence.issuer_corp_code, evidence.issuer_corp_name,
             evidence.source_class, evidence.doc_id, evidence.receipt_no,
             evidence.source_file_id)
            != (operand.issuer_corp_code, operand.issuer_corp_name,
                operand.source_class, operand.doc_id, operand.receipt_no,
                operand.source_file_id)
    ):
        raise DeterministicPlanCompilerError(
            "reason evidence source가 bound comparison operand와 다릅니다")
    proof_ids = [row.evidence_id for row in operands] + [evidence.evidence_id]
    _unique(proof_ids, "document fact comparison evidence_id")
    coordinates = [
        (row.source_file_id, row.path, row.locator) for row in operands
    ] + [(evidence.source_file_id, evidence.path, evidence.locator)]
    _unique(coordinates, "document fact comparison proof coordinate")
    issuer = (operands[0].issuer_corp_code, operands[0].issuer_corp_name)
    if (evidence.issuer_corp_code, evidence.issuer_corp_name) != issuer:
        raise DeterministicPlanCompilerError(
            "reason evidence issuer가 comparison issuer와 다릅니다")
    if len(resolution.premise_proofs) != 1 \
            or resolution.premise_proofs[0].proof_refs != [
                operands[0].evidence_id, operands[1].evidence_id,
            ]:
        raise DeterministicPlanCompilerError(
            "document fact resolution premise proof는 ordered operand evidence만 가져야 합니다")
    return resolution.items[0], None


def _validate_document_attribute_resolution(
        intent_authority: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    """Close two item-scoped attribute proofs without literal fixture data."""
    if (
            not isinstance(intent_authority, tuple)
            or len(intent_authority) != 2
            or not isinstance(intent_authority[0], tuple)
            or len(intent_authority[0]) != 2
    ):
        raise DeterministicPlanCompilerError(
            "document attribute intent authority가 다릅니다")
    intent_items, entity = intent_authority
    if (
            len(resolution.items) != 2
            or [row.item_id for row in resolution.items] != ["item-1", "item-2"]
            or resolution.premise_proofs
    ):
        raise DeterministicPlanCompilerError(
            "document attribute resolution item/premise inventory가 다릅니다")

    evidence_rows: list[DocumentAttributeEvidence] = []
    for row, source in zip(
            resolution.items, intent_items, strict=True):
        if (
                row.target_surface != source.target.surface
                or row.applied_defaults
                or len(row.field_proofs) != 1
                or not isinstance(
                    row.resolution, DocumentAttributeEvidenceResolution)
        ):
            raise DeterministicPlanCompilerError(
                "document attribute resolution target/field/default가 source와 다릅니다")
        proof = row.field_proofs[0]
        if (
                proof.source_field_index != 0
                or proof.surface != source.output.field_surfaces[0]
                or proof.proof_ref != f"source-field:{row.item_id}:0"
        ):
            raise DeterministicPlanCompilerError(
                "document attribute field proof가 source authority와 다릅니다")
        evidence = row.resolution.evidence
        if evidence.issuer_corp_name != entity.surface:
            raise DeterministicPlanCompilerError(
                "document attribute issuer가 semantic company entity와 다릅니다")
        evidence_rows.append(evidence)

    first, second = evidence_rows
    if (
            (first.issuer_corp_code, first.issuer_corp_name)
            != (second.issuer_corp_code, second.issuer_corp_name)
    ):
        raise DeterministicPlanCompilerError(
            "document attribute proofs는 동일 issuer여야 합니다")
    if {first.document_role, second.document_role} != {"earlier", "later"}:
        raise DeterministicPlanCompilerError(
            "document attribute proofs에는 earlier/later가 각각 하나씩 필요합니다")
    by_role = {row.document_role: row for row in evidence_rows}
    if by_role["earlier"].receipt_no[:8] >= by_role["later"].receipt_no[:8]:
        raise DeterministicPlanCompilerError(
            "document attribute earlier/later receipt chronology가 다릅니다")
    _unique([row.doc_id for row in evidence_rows], "document attribute doc_id")
    _unique(
        [row.receipt_no for row in evidence_rows],
        "document attribute receipt_no",
    )
    _unique(
        [row.evidence_id for row in evidence_rows],
        "document attribute evidence_id",
    )
    _unique(
        [(row.source_file_id, row.path, row.locator) for row in evidence_rows],
        "document attribute proof coordinate",
    )
    return resolution.items[0], None


def _validate_resolution_slice(
        intent_item: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[Any, str | None]:
    """Validate a typed resolution by its ordered branch inventory."""
    kinds = tuple(row.resolution.kind for row in resolution.items)
    if kinds == ("document_collection",):
        return _validate_document_collection_resolution(
            intent_item, resolution)
    if kinds == ("holding_disclosure",):
        return _validate_holding_disclosure_resolution(
            intent_item, resolution)
    if kinds == ("document_version_history",):
        return _validate_document_version_history_resolution(
            intent_item, resolution)
    if kinds == ("selected_event",):
        return _validate_selected_event_resolution(intent_item, resolution)
    if kinds == ("event_collection",):
        return _validate_event_collection_resolution(intent_item, resolution)
    if kinds == ("event_amount_change",):
        return _validate_event_amount_change_resolution(
            intent_item, resolution)
    if kinds in {
            ("event_lifecycle_composite",),
            ("event_lifecycle_composite", "event_lifecycle_composite"),
    }:
        return _validate_lifecycle_composite_resolution(intent_item, resolution)
    if kinds == ("correction_lineage",):
        return _validate_correction_diff_resolution(intent_item, resolution)
    if kinds == ("correction_lineage", "correction_lineage"):
        return _validate_correction_history_resolution(intent_item, resolution)
    if kinds == ("termination_reported_status",):
        return _validate_termination_reported_status_resolution(
            intent_item, resolution)
    if kinds == ("same_day_document_candidates", "termination_reported_status"):
        return _validate_same_day_status_resolution(intent_item, resolution)
    if kinds == ("financial_comparison",):
        return _validate_financial_comparison_resolution(intent_item, resolution)
    if (len(kinds) >= 2
            and set(kinds).issubset({"financial", "financial_comparison"})
            and "financial_comparison" in kinds
            and _is_recent_periods_fanout_intent_items(intent_item)
            and _is_recent_periods_fanout_resolution(resolution)):
        return _validate_recent_periods_fanout_resolution(
            intent_item, resolution)
    if kinds == ("financial", "financial_comparison"):
        return _validate_financial_retrieve_comparison_resolution(
            intent_item, resolution)
    if len(kinds) >= 2 and set(kinds) == {"financial"} and (
            _is_recent_periods_fanout_resolution(resolution)):
        # 개념은 같고 기간이 전부(N==2 포함) 다르면 「최근 N년」「최근 N개
        # 분기」fanout(#58 2단계) 또는 이슈 #171 M16(형제마다 이미 다른
        # 리터럴 기간, 연간·분기 혼합 포함)이다. CFS/SFS 짝은 기간이 아니라
        # scope만 갈리므로(기간은 둘 다 같다) 여기 걸리지 않고 아래 옛
        # 분기로 그대로 떨어진다.
        #
        # 이 qid 없는 kind-only 재검증은 어느 intent-side handler가 이
        # resolution을 만들었는지 모른다 — period_type(연간/분기) 동일까지
        # 요구하는 `_validate_recent_periods_fanout_resolution`보다
        # `_validate_explicit_periods_fanout_resolution`이 승인 기준의
        # 상위집합이므로(그 필드 하나만 덜 본다) 그것을 쓴다. 위 두
        # fanout이 실제로 만드는 resolution은 이미 period_type이 전부
        # 같으므로(recent-periods-fanout은 한 축만 펴고, 순수 연간/분기
        # M16 항목도 마찬가지) 이 상위집합 검증이 기존 recent-periods-
        # fanout 시험의 판정을 조금도 바꾸지 않는다.
        return _validate_explicit_periods_fanout_resolution(
            intent_item, resolution)
    if kinds == ("financial", "financial"):
        return _validate_parallel_financial_retrieval_resolution(
            intent_item, resolution)
    if len(kinds) > 2 and set(kinds) == {"financial"}:
        # 「벌었어」·「빚」의 후보를 나란히 답하는 fanout (#94 25). 2-item 짝은
        # 위에서 이미 갈렸다 — 그쪽은 연결/별도라 scope 가 달라야 하고 이쪽은
        # scope 까지 같으므로 규격을 나눠야 한다.
        return _validate_summary_metric_fanout_resolution(
            intent_item, resolution)
    if kinds == ("financial_comparison", "financial_comparison"):
        return _validate_parallel_annual_change_resolution(
            intent_item, resolution)
    if kinds == ("document_fact_comparison", "document_reason_evidence"):
        return _validate_document_fact_resolution(intent_item, resolution)
    if kinds == (
            "document_attribute_evidence", "document_attribute_evidence"):
        return _validate_document_attribute_resolution(intent_item, resolution)
    if kinds == ("periodic_narrative_comparison",):
        return _validate_periodic_narrative_comparison_resolution(
            intent_item, resolution)
    if kinds == ("narrative_matrix",):
        return _validate_narrative_matrix_resolution(intent_item, resolution)
    if kinds == ("periodic_document_narrative",):
        return _validate_periodic_document_narrative_resolution(
            intent_item, resolution)
    if kinds == ("financial",):
        if (
                isinstance(intent_item, tuple)
                and len(intent_item) == 2
                and isinstance(intent_item[1], tuple)
        ):
            return _validate_premise_financial_resolution(intent_item, resolution)
        return _validate_financial_resolution(intent_item, resolution)
    raise DeterministicPlanCompilerError(
        "지원하지 않는 ordered resolution.kind inventory입니다")


def _validate_document_collection_resolution(
        intent_authority: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    """Bind the whole-target intent to a generic company-document selector."""
    if (
            not isinstance(intent_authority, tuple)
            or len(intent_authority) != 2
    ):
        raise DeterministicPlanCompilerError(
            "document collection intent authority가 다릅니다")
    intent_item, entity = intent_authority
    if (
            len(resolution.items) != 1
            or resolution.items[0].item_id != intent_item.item_id
            or resolution.premise_proofs
    ):
        raise DeterministicPlanCompilerError(
            "document collection resolution item/premise inventory가 다릅니다")
    item = resolution.items[0]
    typed = item.resolution
    if (
            not isinstance(typed, DocumentCollectionResolution)
            or item.target_surface != intent_item.target.surface
            or (item.projection_mode != intent_item.output.projection_mode)
            or ([(row.source_field_index, row.surface) for row in item.field_proofs]
                != ([] if intent_item.output.projection_mode == "whole_target" else
                    [(0, intent_item.output.field_surfaces[0])]))
            or item.applied_defaults
            or typed.as_of != resolution.corpus_cutoff
    ):
        raise DeterministicPlanCompilerError(
            "document collection resolution authority가 semantic/cutoff와 다릅니다")
    return item, None


def _validate_holding_disclosure_resolution(
        intent_authority: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    if (not isinstance(intent_authority, tuple)
            or len(intent_authority) != 2):
        raise DeterministicPlanCompilerError(
            "holding disclosure intent authority가 다릅니다")
    intent_item, _companies = intent_authority
    if (len(resolution.items) != 1 or resolution.premise_proofs
            or resolution.items[0].item_id != intent_item.item_id):
        raise DeterministicPlanCompilerError(
            "holding disclosure resolution inventory가 다릅니다")
    item = resolution.items[0]
    typed = item.resolution
    expected_proofs = [
        (index, surface, f"source-field:{item.item_id}:{index}")
        for index, surface in enumerate(intent_item.output.field_surfaces)
    ]
    actual_proofs = [
        (row.source_field_index, row.surface, row.proof_ref)
        for row in item.field_proofs
    ]
    if (not isinstance(typed, HoldingDisclosureResolution)
            or item.target_surface != intent_item.target.surface
            or item.projection_mode != "named_fields"
            or item.applied_defaults
            or actual_proofs != expected_proofs
            or typed.as_of > resolution.corpus_cutoff
            or [
                (row.source_field_index, row.surface, row.proof_ref)
                for row in typed.slot_bindings
            ] != expected_proofs):
        raise DeterministicPlanCompilerError(
            "holding disclosure authority가 semantic intent와 다릅니다")
    return item, None


def _validate_document_version_history_resolution(
        intent_authority: Any, resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    if not isinstance(intent_authority, tuple) or len(intent_authority) != 2:
        raise DeterministicPlanCompilerError("document version intent authority가 다릅니다")
    intent_item, entity = intent_authority
    if (len(resolution.items) != 1 or len(resolution.premise_proofs) != 1
            or resolution.items[0].item_id != intent_item.item_id):
        raise DeterministicPlanCompilerError("document version resolution inventory가 다릅니다")
    item = resolution.items[0]
    typed = item.resolution
    if (
            not isinstance(typed, DocumentVersionHistoryResolution)
            or item.target_surface != intent_item.target.surface
            or item.projection_mode != "named_fields"
            or typed.corp_name != entity.surface
            or [(row.source_field_index, row.surface) for row in item.field_proofs]
            != [(0, intent_item.output.field_surfaces[0])]
            or resolution.premise_proofs[0].premise_id != "premise-1"
    ):
        raise DeterministicPlanCompilerError("document version authority가 semantic intent와 다릅니다")
    return item, None


def _build_document_version_history_typed_plan(
        resolution: AuthoritativeResolution, resolution_item: ResolvedItem,
        intent_item: Any, premise_raw_text: str) -> ExecutionPlan:
    """전제 인용은 **질문 원문 그대로** 싣는다.

    예전에는 `"정정된 적이 없다"` 를 박아 넣었다. 옛 Gold 가 평서형이어서
    맞춘 것인데, 그러면 사용자가 무엇을 물었는지 잃는다. 원문 여부는
    `require_grounded` 가 이미 강제한다.
    """
    typed = resolution_item.resolution
    if not isinstance(typed, DocumentVersionHistoryResolution):
        raise DeterministicPlanCompilerError("document version typed resolution이 필요합니다")
    plan = ResolvedQueryPlan(
        revision=0, reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=[ResolvedDocumentTask(
            task_id="task-1", operation="version_history", corp_code=typed.corp_code,
            corp_name=typed.corp_name, as_of=resolution.corpus_cutoff,
            # The resolver already proved the lineage root.  Execute from that
            # coordinate instead of rescanning every annual filing in the
            # receipt window and treating each family member as a new seed.
            selector=DocumentSelector(
                doc_group="periodic", form="사업보고서",
                rcept_no=typed.root_receipt),
        )], derivations=[], premise_claims=[PremiseClaim(
            claim_id="premise-1", kind="existence",
            raw_text=premise_raw_text,
            verify_tasks=[TaskVerificationRef(task_id="task-1")],
        )], applied_defaults=[], presentation=None)
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version, resolved_plan=plan,
        applied_defaults=[], answer_roots=[ExecutionAnswerRoot(
            root_id="output-1", item_id=intent_item.item_id, field_id="field-1",
            plan_task_id="task-1", proof_ref=typed.lineage_proof_ref)],
        support_roots=[], premise_roots=[ExecutionPremiseRoot(
            premise_id="premise-1", root_ids=["output-1"],
            task_refs=["task-1"])], plan_value_roots=[])


def _validate_premise_financial_resolution(
        intent_authority: tuple[Any, tuple[Any, ...]],
        resolution: AuthoritativeResolution,
        ) -> tuple[Any, str | None]:
    item, premises = intent_authority
    return _validate_financial_resolution(
        item, resolution,
        expected_premise_ids=[premise.premise_id for premise in premises],
    )


def _validate_selected_event_resolution(
        intent_authority: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    """Bind one receipt-selected event without accepting legacy selectors."""
    if not isinstance(intent_authority, tuple) or len(intent_authority) != 3:
        raise DeterministicPlanCompilerError(
            "selected event intent authority가 다릅니다")
    intent_item, entity, premises = intent_authority
    if (len(resolution.items) != 1
            or resolution.items[0].item_id != intent_item.item_id):
        raise DeterministicPlanCompilerError(
            "selected event resolution item/premise inventory가 다릅니다")
    item = resolution.items[0]
    typed = item.resolution
    expected_proofs = [
        (index, surface, f"source-field:{item.item_id}:{index}")
        for index, surface in enumerate(intent_item.output.field_surfaces)
    ]
    actual_proofs = [
        (row.source_field_index, row.surface, row.proof_ref)
        for row in item.field_proofs
    ]
    if [proof.premise_id for proof in resolution.premise_proofs] != [
            premise.premise_id for premise in premises]:
        raise DeterministicPlanCompilerError(
            "selected event premise proof 순서가 source와 다릅니다")
    allowed_proof_refs = {proof.proof_ref for proof in item.field_proofs}
    if any(
            not proof.proof_refs
            or not set(proof.proof_refs).issubset(allowed_proof_refs)
            for proof in resolution.premise_proofs):
        raise DeterministicPlanCompilerError(
            "selected event premise proof가 answer source에 결속되지 않았습니다")
    expected_entity_surface = (
        intent_item.target.surface if typed.timepoints
        else (typed.entity_surface if entity is None else entity.surface))
    if (
            not isinstance(typed, SelectedEventResolution)
            or item.target_surface != intent_item.target.surface
            or item.projection_mode != "named_fields"
            or item.applied_defaults
            or actual_proofs != expected_proofs
            or typed.entity_surface != expected_entity_surface
    ):
        raise DeterministicPlanCompilerError(
            "selected event resolution authority가 semantic intent와 다릅니다")
    if typed.operation == "timeline":
        partial_timeline = (
            len(typed.timepoints) == 1
            and typed.lineage_missing_root_date is not None
            and intent_item.output.shape in {"narrative", "timeline"}
            and len(intent_item.output.field_surfaces) == 2
        )
        complete_timeline = (
            len(typed.timepoints) in {2, 3}
            and typed.lineage_missing_root_date is None
            and intent_item.output.shape == "timeline"
            and len(intent_item.output.field_surfaces) == 2
        )
        bounded_lineage = (
            len(typed.timepoints) == 1
            and typed.timepoints[0] == resolution.corpus_cutoff
            and intent_item.output.shape in {"narrative", "timeline"}
            and len(intent_item.output.field_surfaces) == 2
            and _selected_event_requested_slots(intent_item, typed)
            == ["원계약계보관측"]
        )
        if not (partial_timeline or complete_timeline or bounded_lineage):
            raise DeterministicPlanCompilerError(
                "event timeline authority가 semantic intent와 다릅니다")
        if complete_timeline:
            semantic_timepoints = _ordered_semantic_observation_dates(
                list(intent_item.scope.target_period_expressions),
                reference_date=resolution.reference_date)
            as_of_start, as_of_end, as_of_error = _target_date_range(
                intent_item.scope.as_of_expression or "",
                reference_date=resolution.reference_date)
            if (semantic_timepoints is None or as_of_error is not None
                    or as_of_start is None or as_of_start != as_of_end):
                raise DeterministicPlanCompilerError(
                    "complete event timeline 시점이 exact date가 아닙니다")
            expected_timepoints = list(semantic_timepoints)
            if as_of_start not in expected_timepoints:
                expected_timepoints.append(as_of_start)
            if expected_timepoints != list(typed.timepoints):
                raise DeterministicPlanCompilerError(
                    "complete event timeline 시점이 semantic 좌표와 다릅니다")
    elif typed.timepoints:
        if (len(typed.timepoints) not in {1, 2}
                or (len(typed.timepoints) == 1
                    and intent_item.output.shape != "scalar")
                or (len(typed.timepoints) == 2
                    and intent_item.output.shape not in {
                        "record", "narrative", "comparison"})):
            raise DeterministicPlanCompilerError(
                "multi-timepoint selected event authority가 semantic intent와 다릅니다")
        if entity is None and len(typed.timepoints) == 2:
            semantic_timepoints = _ordered_semantic_observation_dates(
                list(intent_item.target.qualifier_surfaces),
                reference_date=resolution.reference_date)
            if semantic_timepoints != list(typed.timepoints):
                raise DeterministicPlanCompilerError(
                    "issuer 생략 selected event timepoints가 semantic 좌표와 다릅니다")
    elif intent_item.output.shape not in {"scalar", "narrative"} and not (
            # A no-timepoint record reads one or more fields from the one
            # event already proved by the canonical selector. 칸이 하나여도
            # 같은 좌표다 — ``누구랑 공급계약 맺었어`` 는 `계약상대` 한 칸을
            # 뽑는 요청이고, 관측 시점이 필요한 상태 질문이 아니다.
            getattr(intent_item.selection, "mode", None) in {None, "latest"}
            and intent_item.output.shape == "record"
            and len(intent_item.output.field_surfaces) >= 1):
        raise DeterministicPlanCompilerError(
            "selected event status에는 observation timepoints가 필요합니다")
    return item, None


def _validate_event_collection_intent(
        intent: SemanticIntent,
        ) -> tuple[Any, None]:
    """Accept a native multi-event named-field list, independent of wording."""
    if (len(intent.answer_items) != 1 or intent.answer_groups or intent.premises
            or intent.unresolved_mentions):
        raise DeterministicPlanCompilerError(
            "event collection에는 item 하나와 빈 group/premise/unresolved가 필요합니다")
    item = intent.answer_items[0]
    # 이슈 #59 2단계/#124 — 「계약금액이 가장 큰/작은 건은 얼마인가」는
    # 명시적 argmax/argmin selection 을 지닌 event collection 이다. 나머지
    # selection 모양(예: top_k)은 여전히 지원 범위 밖이라 fail-closed 로
    # 남긴다.
    selection = item.selection
    argmax_selection_ok = (
        selection is None
        or (selection.mode in ("maximum", "minimum") and selection.k is None
            and item.output.shape == "scalar"
            and len(item.output.field_surfaces) == 1))
    if (
            item.item_id != "item-1"
            or item.operation not in {"retrieve", "compare"}
            or item.target.kind not in {
                "event", "document", "topic", "entity", "metric", "attribute"}
            or not argmax_selection_ok
            or item.output.projection_mode != "named_fields"
            or item.output.shape not in {
                "record_list", "comparison", "record", "narrative", "timeline",
                "scalar", "verdict"}
            or not item.output.field_surfaces
    ):
        raise DeterministicPlanCompilerError(
            "event collection semantic topology가 지원 범위를 벗어납니다")
    return item, None


def _validate_event_collection_resolution(
        intent_authority: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    if (not hasattr(intent_authority, "item_id")
            or len(resolution.items) != 1 or resolution.premise_proofs):
        raise DeterministicPlanCompilerError(
            "event collection resolution inventory가 다릅니다")
    source = intent_authority
    item = resolution.items[0]
    typed = item.resolution
    expected_proofs = [
        (index, surface, f"source-field:{source.item_id}:{index}")
        for index, surface in enumerate(source.output.field_surfaces)
    ]
    actual_proofs = [
        (proof.source_field_index, proof.surface, proof.proof_ref)
        for proof in item.field_proofs
    ]
    if (
            not isinstance(typed, EventCollectionResolution)
            or item.item_id != source.item_id
            or item.target_surface != source.target.surface
            or item.projection_mode != "named_fields"
            or item.applied_defaults
            or actual_proofs != expected_proofs
            or typed.as_of > resolution.corpus_cutoff
            or typed.requested_slots != _event_collection_slots_from_intent(
                source, typed)
            # 이슈 #59 2단계 — argmax selection 이 있으면 authority 도
            # argmax_slot 을 실어야 하고, 없으면 실으면 안 된다.
            or (source.selection is not None) != (typed.argmax_slot is not None)
            # 이슈 #124 — 있다면 방향(maximum/minimum)도 wire selection과
            # 일치해야 한다.
            or (source.selection is not None and typed.argmax_slot is not None
                and source.selection.mode != typed.argmax_direction)
    ):
        raise DeterministicPlanCompilerError(
            "event collection resolution authority가 semantic intent와 다릅니다")
    return item, None


def _event_collection_slots_from_intent(
        source: Any, typed: EventCollectionResolution,
        ) -> list[str]:
    """Canonical list slots derived from the request category, never a qid."""
    if typed.argmax_slot is not None:
        # 이슈 #59 2단계 — 「계약금액이 가장 큰 건」은 계약명·계약상대·
        # 계약금액 세 slot 을 모아 argmax 로 줄인다. field_surfaces 는
        # 여전히 하나("계약금액")뿐이라 collapsed_projection 이 그대로
        # 이 셋을 field-1 하나에 결속한다.
        return ["계약명", "계약상대", typed.argmax_slot]
    if typed.public_task_kind == "correction":
        compact = [re.sub(r"\s+", "", value)
                   for value in source.output.field_surfaces]
        if (len(compact) == 3
                and "계약별" in compact[0]
                and "전후" in compact[1] and "금액" in compact[1]
                and re.search(r"차이|증감", compact[2])):
            return ["계약별구분", "계약금액", "difference"]
        return []
    source_categories = funding_categories(
        source.target.surface, *source.output.field_surfaces)
    if (typed.event_type == "주요사항보고"
            and source_categories
            and set(typed.keywords) <= FUNDING_CATEGORY_TYPES
            and set(typed.keywords) == set(source_categories)):
        return funding_requested_slots(source.output.field_surfaces)
    if (typed.event_type == "신규시설투자"
            and is_facilities_investment_request(
                source.target.surface, *source.target.qualifier_surfaces)):
        return facilities_investment_requested_slots(
            source.output.field_surfaces)
    if typed.availability_query:
        return ["계약금액", "공시유보여부"]
    if typed.root_contracts_with_confirmed_termination:
        return ["해지연결상태"]
    if (len(source.output.field_surfaces) == 1
            and is_bond_face_value_request(*source.output.field_surfaces)):
        return [bond_face_value_slot()]
    if not typed.requires_termination:
        return normalize_slot_names(list(source.output.field_surfaces))
    if (source.operation == "compare"
            and not any(re.search(r"상대|거래처|계약처|금액|대금|규모|사유|이유|원인", surface)
                        for surface in source.output.field_surfaces)):
        return ["상대방", "해지금액", "해지사유"]
    slots: list[str] = []
    for surface in source.output.field_surfaces:
        compact = re.sub(r"\s+", "", surface)
        if any(cue in compact for cue in ("상대", "거래처", "계약처")):
            slot = "상대방"
        elif any(cue in compact for cue in ("금액", "대금", "규모")):
            slot = "해지금액"
        elif any(cue in compact for cue in ("사유", "이유", "원인")):
            slot = "해지사유"
        else:
            slot = compact
        if slot not in slots:
            slots.append(slot)
    return slots


_LIFECYCLE_ATTRIBUTE_SLOTS: dict[str, str] = {
    "contract_amount": "계약금액",
    "termination_amount": "해지금액",
    "termination_reason": "해지사유",
    "effectiveness_condition": "효력발생조건",
}


def _lifecycle_attribute_kind_for_surface(surface: str) -> str | None:
    compact = re.sub(r"\s+", "", surface)
    if "해지" in compact and any(cue in compact for cue in ("사유", "이유", "원인")):
        return "termination_reason"
    if "해지" in compact and any(cue in compact for cue in ("금액", "대금", "규모")):
        return "termination_amount"
    if "효력" in compact and "조건" in compact:
        return "effectiveness_condition"
    if "계약" in compact and any(cue in compact for cue in ("금액", "대금", "규모")):
        return "contract_amount"
    return None


def _lifecycle_item_mode(source: Any, typed: LifecycleCompositeResolution) -> str | None:
    kinds = [_lifecycle_attribute_kind_for_surface(surface)
             for surface in source.output.field_surfaces]
    if all(kind is not None for kind in kinds):
        available = {row.kind for row in typed.attributes}
        return "attributes" if set(kinds).issubset(available) else None
    state_requested = any(re.search(r"상태|유효|살아|최종|해지|종료", surface)
                          for surface in source.output.field_surfaces)
    if state_requested and typed.status_timepoints:
        return "status"
    correction_requested = any(re.search(r"정정|전후|흐름|변경", surface)
                               for surface in source.output.field_surfaces)
    if correction_requested and typed.correction_receipts:
        return "correction"
    return None


def _lifecycle_status_requested_slots(source: Any) -> list[str]:
    """Carry a disclosure-observation question without implying legal state.

    ``active`` is a canonical event-state label, but a user may ask the much
    narrower question whether a termination/closure filing is *observed* in
    the supplied corpus.  Preserve that literal role as a public task slot so
    Stage2 can answer the closed-world observation and avoid wording it as
    legal validity.  The rule is semantic-surface based and applies to any
    issuer/event/date combination.
    """

    compact = [re.sub(r"\s+", "", value)
               for value in source.output.field_surfaces]
    # Naming the *filing* is what makes this an observation role.  The
    # interrogative that carried the request ("확인되는지") often stays in the
    # question and never reaches the field surface, which arrives as the bare
    # ``해지·종료 공시``; requiring it here let such a request fall through to
    # ordinary status lowering and be answered as «유효(진행 중)» — a legal
    # validity claim the corpus cannot support.  An attribute request such as
    # ``해지 공시의 계약금액`` never arrives here: attribute mode is selected
    # first in ``_lifecycle_item_mode``.
    if any(
            "공시" in value
            and any(token in value for token in ("해지", "종료"))
            for value in compact):
        return ["해지종료공시관측여부"]
    return []


#: 서식 라벨을 가리키지 못하는 구어 의문사. 사용자가 말한 필드 이름이 아니라
#: 「그 칸이 무엇인지 모른다」는 표시다.
_COLLOQUIAL_ONLY_SLOT = re.compile(
    r"^(?:누구|누가|언제|얼마|어디)(?:랑|와|과|를|을|에|의|에게)?$")
#: 구어 의문사가 가리키는 공시 서식 라벨. 공시 서식의 라벨은 닫힌 어휘이고
#: 맞춤은 정규화 후 포함 관계로만 하므로, ``누구`` 는 `3. 계약상대` 를 품지
#: 못해 조회가 빈손이 된다 — `K-020` 은 코퍼스에 ``계약상대 = 유럽 소재 제약사``
#: 가 있는데도 「근거 검증에 실패」로 끝났다.
#:
#: **가리킬 칸이 하나로 정해지는 말만 넣는다.** ``무엇``·``어떻게`` 는 여러 칸을
#: 가리키므로 넣지 않는다 — 그건 되물어야 할 질문이다.
_COLLOQUIAL_SLOT_LABEL = {
    "누구": "계약상대", "누구랑": "계약상대", "누구와": "계약상대",
    "누구에게": "계약상대", "누가": "계약상대",
    "언제": "계약일자", "얼마": "계약금액",
}


def _resolve_colloquial_slots(surfaces: Any) -> list[str]:
    """구어 의문사로만 적힌 출력 필드를 공시 서식 라벨로 바꾼다.

    사용자가 말한 이름은 보존한다 — ``계약금액`` 처럼 이미 라벨을 가리키는
    표면은 그대로 두고, 가리킬 칸이 없는 의문사만 옮긴다.
    """

    out: list[str] = []
    for surface in surfaces or ():
        compact = re.sub(r"\s+", "", str(surface))
        label = (_COLLOQUIAL_SLOT_LABEL.get(compact)
                 if _COLLOQUIAL_ONLY_SLOT.match(compact) else None)
        out.append(label or surface)
    return out


def _selected_event_requested_slots(source: Any, typed: Any) -> list[str]:
    """Preserve the narrow observable role of selected-event timelines.

    Selected events normally use their timepoints as an intrinsic status
    request.  Two timeline questions are intentionally narrower than a legal
    lifecycle-state assertion: whether a termination disclosure was observed,
    and what correction lineage is observable when the canonical root is
    missing.  Carry those roles as internal task slots so Stage2 can render
    bounded facts without calling either result a legally valid contract.
    """

    observation_slots = _lifecycle_status_requested_slots(source)
    if observation_slots:
        return observation_slots
    surfaces = [re.sub(r"\s+", "", value)
                for value in source.output.field_surfaces]
    if (any(any(token in value for token in (
                "원계약", "원공시", "최초체결", "최초공시",
                "최신유효본", "유효본", "최신본", "마지막공시",
            )) for value in surfaces)
            and any(any(token in value for token in ("이력", "계보", "흐름"))
                    for value in surfaces)):
        return ["원계약계보관측"]
    return []


def _validate_lifecycle_composite_intent(
        intent: SemanticIntent,
        ) -> tuple[tuple[Any, ...], None]:
    if (not intent.answer_items or intent.premises or intent.unresolved_mentions):
        raise DeterministicPlanCompilerError(
            "lifecycle composite에는 완결된 answer item이 필요합니다")
    if any(item.operation not in {"retrieve", "compare"}
           or item.output.projection_mode != "named_fields"
           or not item.output.field_surfaces
           for item in intent.answer_items):
        raise DeterministicPlanCompilerError(
            "lifecycle composite semantic topology가 지원 범위를 벗어납니다")
    return tuple(intent.answer_items), None


def _validate_lifecycle_composite_resolution(
        intent_authority: Any, resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    if (not isinstance(intent_authority, tuple)
            or len(intent_authority) != len(resolution.items)
            or resolution.premise_proofs):
        raise DeterministicPlanCompilerError(
            "lifecycle composite resolution inventory가 다릅니다")
    typed_rows: list[LifecycleCompositeResolution] = []
    for source, row in zip(intent_authority, resolution.items, strict=True):
        expected_proofs = [
            (index, surface, f"source-field:{source.item_id}:{index}")
            for index, surface in enumerate(source.output.field_surfaces)]
        actual_proofs = [
            (proof.source_field_index, proof.surface, proof.proof_ref)
            for proof in row.field_proofs]
        if (not isinstance(row.resolution, LifecycleCompositeResolution)
                or row.item_id != source.item_id
                or row.target_surface != source.target.surface
                or row.projection_mode != source.output.projection_mode
                or row.applied_defaults or actual_proofs != expected_proofs
                or row.resolution.as_of > resolution.corpus_cutoff
                or _lifecycle_item_mode(source, row.resolution) is None):
            raise DeterministicPlanCompilerError(
                "lifecycle composite source/proof authority가 다릅니다")
        typed_rows.append(row.resolution)
    coordinates = {
        (row.corp_code, row.corp_name, row.event_key, row.root_receipt, row.as_of)
        for row in typed_rows
    }
    if len(coordinates) != 1:
        raise DeterministicPlanCompilerError(
            "lifecycle composite item은 하나의 canonical event여야 합니다")
    return resolution.items[0], None


def _validate_event_amount_change_resolution(
        intent_authority: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    if (
            len(resolution.items) != 1
            or resolution.premise_proofs
            or not hasattr(intent_authority, "item_id")
    ):
        raise DeterministicPlanCompilerError(
            "event amount change resolution inventory가 다릅니다")
    source = intent_authority
    item = resolution.items[0]
    typed = item.resolution
    expected_proofs = [(
        0, source.output.field_surfaces[0],
        f"source-field:{source.item_id}:0",
    )]
    actual_proofs = [
        (row.source_field_index, row.surface, row.proof_ref)
        for row in item.field_proofs
    ]
    if (
            item.item_id != source.item_id
            or item.target_surface != source.target.surface
            or item.projection_mode != "named_fields"
            or item.applied_defaults
            or actual_proofs != expected_proofs
            or not isinstance(typed, EventAmountChangeResolution)
    ):
        raise DeterministicPlanCompilerError(
            "event amount change resolution authority가 source intent와 다릅니다")
    if any(value > resolution.corpus_cutoff for value in typed.timepoints):
        raise DeterministicPlanCompilerError(
            "event amount change timepoint가 corpus cutoff 이후입니다")
    if (typed.root_receipt[:8] > typed.timepoints[0]
            or [row.timepoint for row in typed.observations] != typed.timepoints
            or len({(row.currency, row.unit, row.scale)
                    for row in typed.observations}) != 1
            or any(row.source_receipt[:8] > row.timepoint
                   for row in typed.observations)):
        raise DeterministicPlanCompilerError(
            "event amount change observation authority가 실행 불가능합니다")
    return item, None


def _validate_periodic_narrative_comparison_resolution(
        intent_authority: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    """Bind generic report coordinates to semantic periods and fields."""
    if (
            not isinstance(intent_authority, tuple)
            or len(intent_authority) != 2
    ):
        raise DeterministicPlanCompilerError(
            "periodic narrative intent authority가 다릅니다")
    intent_item, entity = intent_authority
    if (
            len(resolution.items) != 1
            or resolution.items[0].item_id != intent_item.item_id
            or resolution.premise_proofs
    ):
        raise DeterministicPlanCompilerError(
            "periodic narrative resolution item/premise inventory가 다릅니다")
    item = resolution.items[0]
    if (
            item.target_surface != intent_item.target.surface
            or item.applied_defaults
            or not isinstance(
                item.resolution, PeriodicNarrativeComparisonResolution)
    ):
        raise DeterministicPlanCompilerError(
            "periodic narrative resolution target/default/kind가 다릅니다")
    expected_proofs = [
        (index, surface, f"source-field:{item.item_id}:{index}")
        for index, surface in enumerate(intent_item.output.field_surfaces)
    ]
    actual_proofs = [
        (row.source_field_index, row.surface, row.proof_ref)
        for row in item.field_proofs
    ]
    if actual_proofs != expected_proofs:
        raise DeterministicPlanCompilerError(
            "periodic narrative field proof가 semantic field authority와 다릅니다")
    typed = item.resolution
    annual_surface = (intent_item.scope.document_group_expression
                      or intent_item.target.surface)
    if typed.document_group != annual_surface:
        raise DeterministicPlanCompilerError(
            "periodic narrative document group binding이 다릅니다")
    if len(typed.documents) != len(intent_item.scope.target_period_expressions):
        raise DeterministicPlanCompilerError(
            "periodic narrative document/semantic period count가 다릅니다")
    expected_axis_indexes = list(range(
        len(item.field_proofs)
        if intent_item.output.shape == "comparison"
        else len(item.field_proofs) - 1))
    for document in typed.documents:
        if document.issuer_corp_name != entity.surface:
            raise DeterministicPlanCompilerError(
                "periodic narrative issuer가 semantic company와 다릅니다")
        if [row.source_field_index for row in document.evidence] != (
                expected_axis_indexes):
            raise DeterministicPlanCompilerError(
                "periodic narrative evidence axis가 semantic fields와 닫히지 않습니다")
        if document.source_period_index >= len(
                intent_item.scope.target_period_expressions):
            raise DeterministicPlanCompilerError(
                "periodic narrative source period index가 범위를 벗어났습니다")
    return item, None


def _validate_periodic_document_narrative_resolution(
        intent_item: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[Any, None]:
    if len(resolution.items) != 1:
        raise DeterministicPlanCompilerError(
            "periodic document resolution item inventory가 다릅니다")
    item = resolution.items[0]
    if item.item_id != "item-1" or item.target_surface != intent_item.target.surface:
        raise DeterministicPlanCompilerError(
            "periodic document resolution item/target surface가 source와 다릅니다")
    expected_fields = list(intent_item.output.field_surfaces)
    if [
            (proof.source_field_index, proof.surface, proof.proof_ref)
            for proof in item.field_proofs
    ] != [
            (index, surface, f"source-field:item-1:{index}")
            for index, surface in enumerate(expected_fields)
    ]:
        raise DeterministicPlanCompilerError(
            "periodic document ordered ResolutionFieldProof가 source와 다릅니다")
    expected_defaults = [AppliedDefault(
        policy="as_of", basis="corpus_cutoff", value=resolution.corpus_cutoff,
        evidence_refs=[f"canonical:corpus-cutoff:{resolution.corpus_cutoff}"])]
    if item.applied_defaults not in ([], expected_defaults) or resolution.premise_proofs:
        raise DeterministicPlanCompilerError(
            "periodic document resolution default/premise proof가 다릅니다")
    if not isinstance(item.resolution, PeriodicDocumentNarrativeResolution):
        raise DeterministicPlanCompilerError(
            "periodic document resolution kind가 다릅니다")
    coordinate = item.resolution
    proof_indexes = [proof.source_field_index for proof in item.field_proofs]
    if (
            len(proof_indexes) != len(set(proof_indexes))
            or set(proof_indexes) != (
                set(coordinate.executable_field_indexes)
                | set(coordinate.limited_field_indexes))
            or set(coordinate.executable_field_indexes)
            & set(coordinate.limited_field_indexes)
    ):
        raise DeterministicPlanCompilerError(
            "periodic document field coverage가 source field proofs와 닫히지 않습니다")
    return item, None


def _validate_same_day_status_resolution(
        intent_items: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[Any, None]:
    if not isinstance(intent_items, tuple) or len(intent_items) != 2:
        raise DeterministicPlanCompilerError(
            "same-day status intent item pair가 필요합니다")
    document_item, event_item = intent_items
    if len(resolution.items) != 2:
        raise DeterministicPlanCompilerError(
            "same-day status resolution item inventory가 다릅니다")
    if [row.item_id for row in resolution.items] != ["item-1", "item-2"]:
        raise DeterministicPlanCompilerError(
            "same-day status resolution item 순서가 다릅니다")
    for row, source in zip(
            resolution.items, (document_item, event_item), strict=True):
        if row.target_surface != source.target.surface \
                or len(row.field_proofs) != 1:
            raise DeterministicPlanCompilerError(
                "same-day status resolution target/field 수가 다릅니다")
        proof = row.field_proofs[0]
        if (proof.source_field_index, proof.surface, proof.proof_ref) != (
                0, source.output.field_surfaces[0],
                f"source-field:{row.item_id}:0"):
            raise DeterministicPlanCompilerError(
                "same-day status field proof가 source surface와 다릅니다")
        if row.applied_defaults:
            raise DeterministicPlanCompilerError(
                "same-day status에는 applied default가 없어야 합니다")
    if resolution.premise_proofs:
        raise DeterministicPlanCompilerError(
            "same-day status에는 premise proof가 없어야 합니다")
    document_resolution = resolution.items[0].resolution
    status_resolution = resolution.items[1].resolution
    if not isinstance(document_resolution, SameDayDocumentCandidatesResolution):
        raise DeterministicPlanCompilerError(
            "same-day status item-1 resolution kind가 다릅니다")
    if not isinstance(status_resolution, TerminationReportedStatusResolution):
        raise DeterministicPlanCompilerError(
            "same-day status item-2 resolution kind가 다릅니다")
    termination_receipts = [
        row.rcept_no for row in document_resolution.candidates
        if row.document_kind == "termination"
    ]
    if termination_receipts != [status_resolution.status_receipt]:
        raise DeterministicPlanCompilerError(
            "status receipt가 termination candidate와 닫히지 않습니다")
    return resolution.items[0], None


def _validate_termination_reported_status_resolution(
        intent_authority: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    """Close one reported termination without pretending its origin is known."""
    if (not isinstance(intent_authority, tuple) or len(intent_authority) != 3
            or len(resolution.items) != 1):
        raise DeterministicPlanCompilerError(
            "reported termination resolution inventory가 다릅니다")
    intent_item, _issuer, premises = intent_authority
    item = resolution.items[0]
    if (item.item_id != intent_item.item_id
            or item.target_surface != intent_item.target.surface
            or item.applied_defaults
            or len(item.field_proofs) != len(intent_item.output.field_surfaces)
            or not isinstance(item.resolution, TerminationReportedStatusResolution)):
        raise DeterministicPlanCompilerError(
            "reported termination item authority가 다릅니다")
    if [
            (proof.source_field_index, proof.surface, proof.proof_ref)
            for proof in item.field_proofs
    ] != [
            (index, surface, f"source-field:item-1:{index}")
            for index, surface in enumerate(intent_item.output.field_surfaces)
    ]:
        raise DeterministicPlanCompilerError(
            "reported termination field proof가 source와 다릅니다")
    if [proof.premise_id for proof in resolution.premise_proofs] != [
            premise.premise_id for premise in premises]:
        raise DeterministicPlanCompilerError(
            "reported termination premise proof 순서가 source와 다릅니다")
    _normalized, roles = _termination_reported_status_field_roles(intent_item)
    status_index = roles.index("reported_status")
    allowed_refs = {proof.proof_ref for proof in item.field_proofs}
    status_ref = item.field_proofs[status_index].proof_ref
    if any(
            not proof.proof_refs
            or not set(proof.proof_refs).issubset(allowed_refs)
            or status_ref not in proof.proof_refs
            for proof in resolution.premise_proofs):
        raise DeterministicPlanCompilerError(
            "reported termination premise proof가 상태 answer source와 닫히지 않습니다")
    typed = item.resolution
    if intent_item.scope.as_of_expression is not None:
        as_of, as_of_end, error = _target_date_range(
            intent_item.scope.as_of_expression,
            reference_date=resolution.reference_date)
        if error is not None or as_of is None or as_of != as_of_end \
                or as_of != typed.status_receipt[:8]:
            raise DeterministicPlanCompilerError(
                "reported termination as_of가 status receipt와 닫히지 않습니다")
    elif typed.status_receipt[:8] > resolution.corpus_cutoff:
        raise DeterministicPlanCompilerError(
            "reported termination status receipt가 corpus cutoff 이후입니다")
    return item, None


def _validate_financial_comparison_resolution(
        intent_item: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[Any, None]:
    if len(resolution.items) != 1 or resolution.items[0].item_id != intent_item.item_id:
        raise DeterministicPlanCompilerError(
            "G-A-004 resolution item order/count가 다릅니다")
    item = resolution.items[0]
    if item.target_surface != intent_item.target.surface:
        raise DeterministicPlanCompilerError(
            "G-A-004 resolution target surface가 source와 다릅니다")
    if len(item.field_proofs) != len(intent_item.output.field_surfaces):
        raise DeterministicPlanCompilerError(
            "G-A-004 resolution field proof count가 다릅니다")
    # 필드 표면은 **질문에서 온 것**과 같아야 한다. 승인 슬라이스의 낱말을 박아
    # 두면 그 두 낱말을 쓴 질문만 통과한다.
    if [proof.source_field_index for proof in item.field_proofs] != list(
            range(len(intent_item.output.field_surfaces))) \
            or [proof.surface for proof in item.field_proofs] != list(
                intent_item.output.field_surfaces):
        raise DeterministicPlanCompilerError(
            "resolution source field proof가 source와 다릅니다")
    comparison = item.resolution
    if not isinstance(comparison, FinancialComparisonResolution):
        raise DeterministicPlanCompilerError(
            "G-A-004 resolution은 financial_comparison tagged branch여야 합니다")
    if comparison.verification_claim is None:
        if resolution.premise_proofs:
            raise DeterministicPlanCompilerError(
                "ordinary financial comparison에는 premise proof가 없어야 합니다")
    else:
        if (
                len(resolution.premise_proofs) != 1
                or resolution.premise_proofs[0].premise_id
                != comparison.verification_premise_id
                or not resolution.premise_proofs[0].proof_refs
                or not set(resolution.premise_proofs[0].proof_refs).issubset(
                    {proof.proof_ref for proof in item.field_proofs})
        ):
            raise DeterministicPlanCompilerError(
                "financial verification premise proof가 source fields와 닫히지 않습니다")
    if len(comparison.operands) not in {2, 3, 4, 5, 6, 7, 8}:
        raise DeterministicPlanCompilerError(
            "financial comparison은 2~8개 operand가 필요합니다")
    operands = comparison.operands
    if [operand.operand_id for operand in operands] != [
            f"operand-{index}" for index in range(1, len(operands) + 1)]:
        raise DeterministicPlanCompilerError(
            "financial comparison operand 순서가 entity 순서와 다릅니다")
    # Values are the canonical resolver's job.  The compiler closes only the
    # relation topology: different companies at one coordinate, or one
    # company at two distinct coordinates.
    first, second = operands[0], operands[1]
    company_comparison = len({operand.corp_code for operand in operands}) > 1
    if company_comparison:
        if len(operands) != len(intent_item.target.entity_refs):
            raise DeterministicPlanCompilerError(
                "company comparison operand 수가 source company ref 수와 다릅니다")
        if len({operand.corp_code for operand in operands}) != len(operands):
            raise DeterministicPlanCompilerError(
                "company comparison에 canonical corp_code가 중복됐습니다")
        for operand in operands[1:]:
            # 이슈 #182 — `as_of`는 여기서도 뺀다. `FinancialComparisonResolution.
            # validate_operands`(위 pydantic 모델 검증)와 같은 이유: 회사마다
            # 자기 회계연도말 사업보고서를 각자 다른 날짜에 낸다(#149,
            # `_instant_fiscal_year_end_operand_as_of`). 나머지 좌표축이 모두
            # 같으면 그것으로 이미 같은 사실을 가리킨다.
            for name in ("concept", "period_start", "period_end", "period_type",
                         "scope", "statement", "view", "cumulative"):
                if getattr(first, name) == getattr(operand, name):
                    continue
                raise DeterministicPlanCompilerError(
                    f"company comparison operand의 {name}가 서로 다릅니다")
    else:
        if len(operands) != 2:
            raise DeterministicPlanCompilerError(
                "same-company financial relation은 operand 두 개가 필요합니다")
        if first.corp_name != second.corp_name:
            raise DeterministicPlanCompilerError(
                "time derivation operands의 회사명이 다릅니다")
        if comparison.requested_operators == ["concept_ratio"]:
            if first.concept == second.concept:
                raise DeterministicPlanCompilerError(
                    "concept_ratio operand의 concept가 서로 같습니다")
            if first.scope != second.scope:
                raise DeterministicPlanCompilerError(
                    "concept_ratio operand의 scope가 서로 다릅니다")
            if first.view != second.view or first.as_of != second.as_of:
                raise DeterministicPlanCompilerError(
                    "concept_ratio operand의 view·as_of가 서로 다릅니다")
            if first.period_end != second.period_end:
                raise DeterministicPlanCompilerError(
                    "concept_ratio operand의 기간이 서로 다릅니다")
        else:
            for name in ("concept", "statement"):
                if getattr(first, name) != getattr(second, name):
                    raise DeterministicPlanCompilerError(
                        f"financial relation operand의 {name}가 서로 다릅니다")
            same_period = (first.period_start, first.period_end, first.period_type,
                    first.cumulative) == (
                        second.period_start, second.period_end, second.period_type,
                        second.cumulative)
            same_scope = first.scope == second.scope
            same_view = first.view == second.view
            if sum((not same_period, not same_scope, not same_view)) != 1:
                raise DeterministicPlanCompilerError(
                    "financial relation은 기간, statement scope, view 중 한 축만 달라야 합니다")
            if not same_scope and first.as_of != second.as_of:
                raise DeterministicPlanCompilerError(
                    "scope relation operand의 as_of가 서로 다릅니다")
            if not same_view and (
                    first.as_of != second.as_of
                    or [first.view, second.view] != ["as_filed", "restated"]
                    or comparison.requested_operators != ["difference"]):
                raise DeterministicPlanCompilerError(
                    "view relation은 as_filed→restated 동일 as_of difference여야 합니다")
    for operand in operands:
        if not operand.corp_name.strip():
            raise DeterministicPlanCompilerError("operand corp_name이 비었습니다")
        if operand.proof_ref != f"source-operand:{operand.operand_id}":
            raise DeterministicPlanCompilerError(
                "operand evidence proof 결속이 다릅니다")
    _financial_view_account_paths(item, comparison)
    return item, None


def _financial_view_account_paths(
        resolution_item: ResolvedItem,
        comparison: FinancialComparisonResolution,
        ) -> tuple[str | None, ...]:
    """Read the digest-bound exact paths for a filing-view comparison."""

    policies = [
        row for row in resolution_item.applied_defaults
        if row.policy == "question_grounded_view_account_paths"
    ]
    views = [operand.view for operand in comparison.operands]
    if len(set(views)) == 1:
        if policies:
            raise DeterministicPlanCompilerError(
                "non-view comparison에 view account path sidecar가 있습니다")
        return tuple(None for _operand in comparison.operands)
    if views != ["as_filed", "restated"] or len(policies) != 1:
        raise DeterministicPlanCompilerError(
            "view comparison account path sidecar 결속이 다릅니다")
    try:
        payload = json.loads(policies[0].value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DeterministicPlanCompilerError(
            "view comparison account path sidecar JSON이 잘못되었습니다") from exc
    if (not isinstance(payload, dict)
            or set(payload) != {"as_filed", "restated"}
            or any(not isinstance(payload[view], str) or not payload[view].strip()
                   for view in views)):
        raise DeterministicPlanCompilerError(
            "view comparison account path sidecar 값이 잘못되었습니다")
    return tuple(payload[view] for view in views)


def _validate_financial_retrieve_comparison_resolution(
        intent_items: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    """Close ordered financial + financial_comparison authorities.

    The first scalar lookup is executable from the comparison's first operand
    only when every typed financial coordinate agrees.  This permits one
    shared task without turning a merely similar fact into the user's first
    requested answer.
    """

    if not isinstance(intent_items, tuple) or len(intent_items) != 2:
        raise DeterministicPlanCompilerError(
            "financial retrieve/comparison intent item pair가 필요합니다")
    retrieve_intent, comparison_intent = intent_items
    if (
            len(resolution.items) != 2
            or [row.item_id for row in resolution.items]
            != [retrieve_intent.item_id, comparison_intent.item_id]
            or tuple(row.resolution.kind for row in resolution.items)
            != ("financial", "financial_comparison")
            or resolution.premise_proofs
    ):
        raise DeterministicPlanCompilerError(
            "financial retrieve/comparison resolution order/count가 다릅니다")
    retrieve_resolution, comparison_resolution = resolution.items
    _validate_financial_resolution(
        retrieve_intent,
        resolution.model_copy(update={"items": [retrieve_resolution]}),
    )
    _validate_financial_comparison_resolution(
        comparison_intent,
        resolution.model_copy(update={"items": [comparison_resolution]}),
    )
    coordinate = retrieve_resolution.resolution
    comparison = comparison_resolution.resolution
    if not isinstance(coordinate, FinancialResolution) or not isinstance(
            comparison, FinancialComparisonResolution):
        raise DeterministicPlanCompilerError(
            "financial retrieve/comparison typed branch가 다릅니다")
    if (
            comparison.verification_claim is not None
            or comparison.verification_premise_id is not None
            or len(comparison.requested_operators) != 1
            or comparison.requested_operators[0]
            not in {"difference", "percent_change", "concept_ratio"}
    ):
        raise DeterministicPlanCompilerError(
            "scalar financial comparison에는 단일 change operator가 필요합니다")
    first_operand = comparison.operands[0]
    shared_fields = (
        "corp_code", "corp_name", "concept", "period_start", "period_end",
        "period_type", "scope", "statement", "view", "as_of", "cumulative",
    )
    if any(
            getattr(coordinate, field) != getattr(first_operand, field)
            for field in shared_fields
    ):
        raise DeterministicPlanCompilerError(
            "financial retrieve fact가 comparison first operand와 다릅니다")
    return comparison_resolution, None


def _validate_parallel_financial_retrieval_resolution(
        intent_items: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    """Close two scalar financial authorities without adding an operator."""

    if not isinstance(intent_items, tuple) or len(intent_items) != 2:
        raise DeterministicPlanCompilerError(
            "parallel financial retrieval intent pair가 필요합니다")
    if (len(resolution.items) != 2
            or tuple(row.resolution.kind for row in resolution.items)
            != ("financial", "financial")
            or [row.item_id for row in resolution.items]
            != [row.item_id for row in intent_items]
            or resolution.premise_proofs):
        raise DeterministicPlanCompilerError(
            "parallel financial retrieval resolution inventory가 다릅니다")
    for source_item, resolved_item in zip(
            intent_items, resolution.items, strict=True):
        _validate_financial_resolution(
            source_item,
            resolution.model_copy(update={"items": [resolved_item]}),
        )
    coordinates = [row.resolution for row in resolution.items]
    if not all(isinstance(row, FinancialResolution) for row in coordinates):
        raise DeterministicPlanCompilerError(
            "parallel financial retrieval typed branch가 다릅니다")
    first, second = coordinates
    assert isinstance(first, FinancialResolution)
    assert isinstance(second, FinancialResolution)
    interest_pair = {first.concept, second.concept} == {"interest_expense", "interest_paid"}
    shared = (
        "corp_code", "corp_name", "concept", "period_start", "period_end",
        "period_type", "statement", "view", "as_of", "cumulative",
    )
    if interest_pair:
        shared = tuple(field for field in shared if field not in {"concept", "statement"})
    if any(getattr(first, field) != getattr(second, field) for field in shared):
        raise DeterministicPlanCompilerError(
            "parallel financial retrieval의 canonical 좌표가 다릅니다")
    if interest_pair:
        if first.scope != second.scope or {
                (first.concept, first.statement), (second.concept, second.statement)
        } != {("interest_expense", "IS"), ("interest_paid", "CF")}:
            raise DeterministicPlanCompilerError("interest expense/payment statement 결속이 다릅니다")
    elif {first.scope, second.scope} != {"CFS", "SFS"}:
        raise DeterministicPlanCompilerError(
            "parallel financial retrieval scope가 CFS/SFS가 아닙니다")
    return resolution.items[0], None


def _validate_summary_metric_fanout_resolution(
        intent_items: Any,
        resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    """Close N scalar financial authorities that differ **only** by concept.

    후보를 나란히 답하는 것이 뜻이 통하려면 좌표가 개념 말고는 전부 같아야
    한다. 기간이나 scope 까지 갈리면 그것은 편 결과가 아니라 서로 다른 질문
    여럿이고, 한 줄로 묶어 답하면 읽는 사람이 같은 조건의 비교로 읽는다.
    """

    if not isinstance(intent_items, tuple) or len(intent_items) < 3:
        raise DeterministicPlanCompilerError(
            "summary metric fanout intent item 세 개 이상이 필요합니다")
    if (len(resolution.items) != len(intent_items)
            or set(row.resolution.kind for row in resolution.items)
            != {"financial"}
            or [row.item_id for row in resolution.items]
            != [row.item_id for row in intent_items]
            or resolution.premise_proofs):
        raise DeterministicPlanCompilerError(
            "summary metric fanout resolution inventory가 다릅니다")
    for source_item, resolved_item in zip(
            intent_items, resolution.items, strict=True):
        _validate_financial_resolution(
            source_item,
            resolution.model_copy(update={"items": [resolved_item]}),
        )
    coordinates = [row.resolution for row in resolution.items]
    if not all(isinstance(row, FinancialResolution) for row in coordinates):
        raise DeterministicPlanCompilerError(
            "summary metric fanout typed branch가 다릅니다")
    first = coordinates[0]
    # `statement` 도 같아야 한다. 손익 세 단계(「벌었어」)나 부채 총계와 그
    # 분해(「빚」)는 한 재무제표 안에서 위아래로 읽는 값이라 나란히 놓는 것이
    # 뜻이 통하지만, 재무제표를 건너뛴 값을 한 줄로 묶으면 읽는 사람이 같은
    # 조건의 비교로 읽는다 — 「현금」을 승인에서 뺀 이유와 같다.
    # `SummaryMetricFanoutRegrounder` 가 펴기 전에 같은 것을 확인하므로
    # 여기까지 오면 이미 참이다. 이 층은 그것을 다시 증명한다.
    shared = (
        "corp_code", "corp_name", "period_start", "period_end",
        "period_type", "view", "as_of", "cumulative", "scope", "statement",
    )
    if any(getattr(row, field) != getattr(first, field)
           for row in coordinates[1:] for field in shared):
        raise DeterministicPlanCompilerError(
            "summary metric fanout의 canonical 좌표가 개념 말고도 다릅니다")
    concepts = [row.concept for row in coordinates]
    if len(set(concepts)) != len(concepts):
        # 같은 개념을 두 번 답하면 후보를 편 것이 아니라 한 값을 되풀이한
        # 것이다. 읽는 사람에게는 서로 다른 지표로 보인다.
        raise DeterministicPlanCompilerError(
            "summary metric fanout이 같은 concept을 되풀이합니다")
    return resolution.items[0], None


def _validate_parallel_annual_change_resolution(
        intent_items: Any, resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    """Close two split derivations to the exact same annual operands."""

    if not isinstance(intent_items, tuple) or len(intent_items) != 2:
        raise DeterministicPlanCompilerError(
            "parallel annual change intent pair가 필요합니다")
    if (
            len(resolution.items) != 2
            or [row.item_id for row in resolution.items]
            != [row.item_id for row in intent_items]
            or tuple(row.resolution.kind for row in resolution.items)
            != ("financial_comparison", "financial_comparison")
            or resolution.premise_proofs
    ):
        raise DeterministicPlanCompilerError(
            "parallel annual change resolution inventory가 다릅니다")
    for source_item, resolved_item in zip(
            intent_items, resolution.items, strict=True):
        _validate_financial_comparison_resolution(
            source_item,
            resolution.model_copy(update={"items": [resolved_item]}),
        )
    first = resolution.items[0].resolution
    second = resolution.items[1].resolution
    if not isinstance(first, FinancialComparisonResolution) or not isinstance(
            second, FinancialComparisonResolution):
        raise DeterministicPlanCompilerError(
            "parallel annual change typed authority가 다릅니다")
    operators = [*first.requested_operators, *second.requested_operators]
    if (
            len(first.requested_operators) != 1
            or len(second.requested_operators) != 1
            or set(operators) != {"difference", "percent_change"}
            or any(
                operand.period_type != "annual"
                and not (operand.period_type == "instant"
                         and operand.period_end.month == 12)
                for operand in first.operands)
            or [operand.model_dump(mode="json") for operand in first.operands]
            != [operand.model_dump(mode="json") for operand in second.operands]
    ):
        raise DeterministicPlanCompilerError(
            "parallel annual change operator/operand authority가 다릅니다")
    return resolution.items[0], None


# ── 동결 슬라이스 승인 좌표 검증 (오프라인 전용 — 런타임 dispatch 미참조) ──────────────────────

def _validate_g_a_010_approved_resolution(
        resolution: AuthoritativeResolution,
        ) -> None:
    """Check exact G-A-010 coordinates only at the approval boundary."""
    if (
            resolution.question_id != G_A_010
            or len(resolution.items) != 1
            or resolution.reference_date != date(2026, 6, 19)
            or resolution.corpus_cutoff != "20260619"
            or resolution.resolver_version != "stage1-resolver/1.0"
            or resolution.premise_proofs
    ):
        raise DeterministicPlanCompilerError(
            "G-A-010 approved resolution inventory가 다릅니다")
    item = resolution.items[0]
    typed = item.resolution
    if not isinstance(typed, PeriodicDocumentNarrativeResolution):
        raise DeterministicPlanCompilerError(
            "G-A-010 approved resolution kind가 다릅니다")
    if (
            item.item_id != "item-1"
            or item.target_surface != "주요 투자계획"
            or item.applied_defaults
            or [
                (proof.source_field_index, proof.surface, proof.proof_ref)
                for proof in item.field_proofs
            ] != [
                (0, "투자 대상", "source-field:item-1:0"),
                (1, "목적", "source-field:item-1:1"),
                (2, "금액", "source-field:item-1:2"),
                (3, "기간", "source-field:item-1:3"),
            ]
            or (
                typed.corp_code, typed.corp_name, typed.document_id,
                typed.receipt_no,
            ) != (
                G_A_010_CORP_CODE, G_A_010_CORP_NAME,
                G_A_010_DOCUMENT_ID, G_A_010_RECEIPT_NO,
            )
            or typed.document_proof != ResolutionSourceProof(
                source_receipt=G_A_010_RECEIPT_NO,
                proof_ref=G_A_010_DOCUMENT_PROOF)
            or typed.narrative_proof != ResolutionSourceProof(
                source_receipt=G_A_010_RECEIPT_NO,
                proof_ref=G_A_010_NARRATIVE_PROOF)
            or typed.executable_field_indexes != [0, 1, 2, 3]
            or typed.limited_field_indexes != []
            or typed.source_retrieval_query != "설비 투자 현황 및 계획"
    ):
        raise DeterministicPlanCompilerError(
            "G-A-010 approved item/coordinate authority가 다릅니다")
    if typed.source_cross_check_provenance is not None:
        raise DeterministicPlanCompilerError(
            "G-A-010 approved complete resolution에는 limitation이 없어야 합니다")


def _validate_g_i_004_approved_resolution(
        resolution: AuthoritativeResolution,
        ) -> None:
    """Check exact G-I-004 coordinates only at the approval boundary."""
    if (
            resolution.question_id != G_I_004
            or len(resolution.items) != 2
            or resolution.reference_date != date(2026, 6, 19)
            or resolution.corpus_cutoff != "20260619"
            or resolution.resolver_version != "stage1-resolver/1.0"
            or resolution.premise_proofs
    ):
        raise DeterministicPlanCompilerError(
            "G-I-004 approved resolution inventory가 다릅니다")
    first, second = resolution.items
    if (
            first.item_id, second.item_id,
            first.target_surface, second.target_surface,
            [(p.source_field_index, p.surface, p.proof_ref)
             for p in first.field_proofs],
            [(p.source_field_index, p.surface, p.proof_ref)
             for p in second.field_proofs],
    ) != (
            "item-1", "item-2", "공시", "계약",
            [(0, "내용", "source-field:item-1:0")],
            [(0, "최종 상태", "source-field:item-2:0")],
    ) or first.applied_defaults or second.applied_defaults:
        raise DeterministicPlanCompilerError(
            "G-I-004 approved item/field authority가 다릅니다")
    document = first.resolution
    status = second.resolution
    if not isinstance(document, SameDayDocumentCandidatesResolution) \
            or not isinstance(status, TerminationReportedStatusResolution):
        raise DeterministicPlanCompilerError(
            "G-I-004 approved resolution kind inventory가 다릅니다")
    if (
            document.issuer_corp_code,
            document.issuer_corp_name,
            document.as_of,
            [(row.rcept_no, row.document_kind, row.proof_ref)
             for row in document.candidates],
            document.ordering_provenance.detail,
            tuple(document.ordering_provenance.evidence_refs),
    ) != (
            "01515323", "LG에너지솔루션", "20251217",
            [
                ("20251217800800", "termination",
                 "source-document:20251217800800"),
                ("20251217800853", "correction",
                 "source-document:20251217800853"),
            ],
            G_I_004_ORDERING_DETAIL,
            G_I_004_ORDERING_EVIDENCE_REFS,
    ) or document.ordering_provenance.original_receipts:
        raise DeterministicPlanCompilerError(
            "G-I-004 approved document coordinates가 다릅니다")
    if (
            status.status_receipt,
            status.event_key,
            status.event_key_proof,
            status.status_proof,
            status.identity_provenance.detail,
            tuple(status.identity_provenance.evidence_refs),
            status.identity_provenance.original_receipts,
    ) != (
            "20251217800800",
            "45eaec69e2677af016187d5523c48439",
            ResolutionSourceProof(
                source_receipt="20251217800800",
                proof_ref="source-event-key:20251217800800"),
            ResolutionSourceProof(
                source_receipt="20251217800800",
                proof_ref="source-status:20251217800800"),
            G_I_004_IDENTITY_DETAIL,
            G_I_004_IDENTITY_EVIDENCE_REFS,
            ["20241015800258", "20241015800261"],
    ):
        raise DeterministicPlanCompilerError(
            "G-I-004 approved status coordinates가 다릅니다")


def _validate_g_i_006_approved_resolution(
        resolution: AuthoritativeResolution,
        ) -> None:
    """Check exact approved G-I-006 coordinates after generic lowering."""
    if (
            resolution.question_id != G_I_006
            or len(resolution.items) != 2
            or resolution.reference_date != date(2026, 6, 19)
            or resolution.corpus_cutoff != "20260619"
            or resolution.resolver_version != "stage1-resolver/1.0"
    ):
        raise DeterministicPlanCompilerError("G-I-006 approved resolution inventory가 다릅니다")
    first, second = resolution.items
    if (
            first.item_id, second.item_id,
            first.target_surface, second.target_surface,
            [row.source_field_index for row in first.field_proofs],
            [row.surface for row in first.field_proofs],
            [row.proof_ref for row in first.field_proofs],
            [row.source_field_index for row in second.field_proofs],
            [row.surface for row in second.field_proofs],
            [row.proof_ref for row in second.field_proofs],
    ) != (
            "item-1", "item-2", "Freudenberg 계약", "Freudenberg 계약",
            [0], ["정정 후 계약금액과 해지금액은 같으며"],
            ["source-field:item-1:0"],
            [0], ["왜 다른가"], ["source-field:item-2:0"],
    ):
        raise DeterministicPlanCompilerError(
            "G-I-006 approved item/field surfaces가 다릅니다")
    comparison = first.resolution
    reason = second.resolution
    if not isinstance(comparison, DocumentFactComparisonResolution) \
            or not isinstance(reason, DocumentReasonEvidenceResolution):
        raise DeterministicPlanCompilerError(
            "G-I-006 approved resolution kind inventory가 다릅니다")
    expected_operands = [
        (
            "operand-1", "01515323", "LG에너지솔루션", "correction",
            "exchange_20251226800767", "20251226800767",
            "2. 계약내역 > 계약금액(원)",
            "TABLE[3]/TBODY[0]/TR[2]/TD[2]",
            "2daec4ae341769ce1570af9b3b38ebcd",
            "c9858d6fe020992832d655dfcbdb7318",
        ),
        (
            "operand-2", "01515323", "LG에너지솔루션", "disclosure",
            "exchange_20251226800706", "20251226800706",
            "2. 해지내역 > 해지금액(원)",
            "TABLE[0]/TBODY[0]/TR[2]/TD[2]",
            "81da2131f5aa77e9fda1e9a2fa77eada",
            "ea48d491bf3d342228762e0ab6d3991f",
        ),
    ]
    actual_operands = [
        (row.operand_id, row.issuer_corp_code, row.issuer_corp_name,
         row.source_class, row.doc_id, row.receipt_no, row.path, row.locator,
         row.source_file_id, row.evidence_id)
        for row in comparison.operands
    ]
    if actual_operands != expected_operands:
        raise DeterministicPlanCompilerError(
            "G-I-006 approved operand coordinate/proof가 다릅니다")
    expected_reason = (
        "operand-2", "01515323", "LG에너지솔루션", "disclosure",
        "exchange_20251226800706", "20251226800706",
        "8. 기타 투자판단과 관련한 중요사항",
        "TABLE[0]/TBODY[0]/TR[14]/TD[1]",
        "81da2131f5aa77e9fda1e9a2fa77eada",
        "0a90701ead2898d351f9da450e317847",
    )
    actual_reason = (
        reason.evidence.operand_id, reason.evidence.issuer_corp_code,
        reason.evidence.issuer_corp_name, reason.evidence.source_class,
        reason.evidence.doc_id, reason.evidence.receipt_no,
        reason.evidence.path, reason.evidence.locator,
        reason.evidence.source_file_id, reason.evidence.evidence_id,
    )
    if actual_reason != expected_reason:
        raise DeterministicPlanCompilerError(
            "G-I-006 approved reason evidence coordinate/proof가 다릅니다")
    if (
            len(resolution.premise_proofs) != 1
            or resolution.premise_proofs[0].premise_id != "premise-1"
            or resolution.premise_proofs[0].proof_refs != [
                "c9858d6fe020992832d655dfcbdb7318",
                "ea48d491bf3d342228762e0ab6d3991f",
            ]
    ):
        raise DeterministicPlanCompilerError(
            "G-I-006 approved premise proof가 다릅니다")


def _validate_g_i_009_approved_resolution(
        resolution: AuthoritativeResolution,
        ) -> None:
    """Check the approved G-I-009 source tuple at the demo boundary only."""
    if (
            resolution.question_id != G_I_009
            or len(resolution.items) != 2
            or resolution.reference_date != date(2026, 6, 19)
            or resolution.corpus_cutoff != "20260619"
            or resolution.resolver_version != "stage1-resolver/1.0"
            or resolution.premise_proofs
    ):
        raise DeterministicPlanCompilerError(
            "G-I-009 approved resolution inventory가 다릅니다")
    expected_items = [
        (
            "item-1", "계약", "해지된 이유",
            "later", "exchange_20250402800768", "20250402800768",
            "5. 해지 주요사유", "TABLE[0]/TBODY[0]/TR[10]/TD[1]",
            "af55a3ac349da577311a38f4a00b1fe8",
            "f1994ac666ab08b264bbd47366002fda",
        ),
        (
            "item-2", "계약", "계약 효력발생 조건",
            "earlier", "exchange_20231006800130", "20231006800130",
            "9. 기타 투자판단과 관련한 중요사항",
            "TABLE[0]/TBODY[0]/TR[16]/TD[0]",
            "340ede957a6b679fea415cb2dbed82b0",
            "b4481081263b078817549657bcea5393",
        ),
    ]
    actual_items = []
    for row in resolution.items:
        if not isinstance(row.resolution, DocumentAttributeEvidenceResolution):
            raise DeterministicPlanCompilerError(
                "G-I-009 approved resolution kind inventory가 다릅니다")
        if (
                len(row.field_proofs) != 1
                or row.field_proofs[0].source_field_index != 0
                or row.field_proofs[0].proof_ref
                != f"source-field:{row.item_id}:0"
                or row.applied_defaults
        ):
            raise DeterministicPlanCompilerError(
                "G-I-009 approved field/default authority가 다릅니다")
        evidence = row.resolution.evidence
        actual_items.append((
            row.item_id, row.target_surface, row.field_proofs[0].surface,
            evidence.document_role, evidence.doc_id, evidence.receipt_no,
            evidence.path, evidence.locator, evidence.source_file_id,
            evidence.evidence_id,
        ))
        if (
                evidence.issuer_corp_code != "01412725"
                or evidence.issuer_corp_name != "두산퓨얼셀"
                or evidence.value_kind != "text"
        ):
            raise DeterministicPlanCompilerError(
                "G-I-009 approved issuer/value kind가 다릅니다")
    if actual_items != expected_items:
        raise DeterministicPlanCompilerError(
            "G-I-009 approved item/source coordinate가 다릅니다")


def _validate_g_o_001_approved_resolution(
        resolution: AuthoritativeResolution,
        ) -> None:
    """Check exact Gold report coordinates only at the approved adapter."""
    if (
            resolution.question_id != G_O_001
            or len(resolution.items) != 1
            or resolution.reference_date != date(2026, 6, 19)
            or resolution.corpus_cutoff != "20260619"
            or resolution.resolver_version != "stage1-resolver/1.0"
            or resolution.premise_proofs
    ):
        raise DeterministicPlanCompilerError(
            "G-O-001 approved resolution inventory가 다릅니다")
    item = resolution.items[0]
    if not isinstance(item.resolution, PeriodicNarrativeComparisonResolution):
        raise DeterministicPlanCompilerError(
            "G-O-001 approved resolution kind가 다릅니다")
    if (
            item.item_id != "item-1"
            or item.target_surface != "사업부문·주요 제품 및 서비스·매출구성"
            or item.applied_defaults
            or item.resolution.document_group != "사업보고서"
            or [
                (row.source_field_index, row.surface, row.proof_ref)
                for row in item.field_proofs
            ] != [
                (0, "사업부문", "source-field:item-1:0"),
                (1, "주요 제품 및 서비스", "source-field:item-1:1"),
                (2, "매출구성", "source-field:item-1:2"),
                (3, "핵심 변화", "source-field:item-1:3"),
            ]
    ):
        raise DeterministicPlanCompilerError(
            "G-O-001 approved item/field authority가 다릅니다")
    expected_documents = [
        (
            0, "00126380", "삼성전자", "periodic_20240312000736",
            "20240312000736", date(2023, 1, 1), date(2023, 12, 31),
            "d90027283461c8da56846716e009b074",
            [
                (0, "business_segments", "II. 사업의 내용 > 1. 사업의 개요",
                 "BODY[0]/SECTION-1[2]/LIBRARY[0]/SECTION-2[0]/TITLE[0]/PART[0]",
                 "58fa81bdaad0f10df735453f46bfe001"),
                (1, "products_services", "II. 사업의 내용 > 2. 주요 제품 및 서비스",
                 "BODY[0]/SECTION-1[2]/LIBRARY[0]/SECTION-2[1]/TITLE[0]/PART[0]",
                 "5bb6823bc687fc2ac9a368c259425fd6"),
                (2, "sales_mix", "II. 사업의 내용 > 4. 매출 및 수주상황",
                 "BODY[0]/SECTION-1[2]/LIBRARY[0]/SECTION-2[3]/TITLE[0]/PART[0]",
                 "742866eb17bcf1f316c1d29df7746c56"),
            ],
        ),
        (
            1, "00126380", "삼성전자", "periodic_20260310002820",
            "20260310002820", date(2025, 1, 1), date(2025, 12, 31),
            "7f5865931f7b2520c77b5d87f829a022",
            [
                (0, "business_segments", "II. 사업의 내용 > 1. 사업의 개요",
                 "BODY[0]/SECTION-1[2]/LIBRARY[0]/SECTION-2[0]/TITLE[0]/PART[0]",
                 "80703106bee3ac4841fd8a3530fbdc5b"),
                (1, "products_services", "II. 사업의 내용 > 2. 주요 제품 및 서비스",
                 "BODY[0]/SECTION-1[2]/LIBRARY[0]/SECTION-2[1]/TITLE[0]/PART[0]",
                 "264bcdbb4a54ddafda352623d1c6b6e0"),
                (2, "sales_mix", "II. 사업의 내용 > 4. 매출 및 수주상황",
                 "BODY[0]/SECTION-1[2]/LIBRARY[0]/SECTION-2[3]/TITLE[0]/PART[0]",
                 "acd6daf3b73c06c574866ba96fba10fb"),
            ],
        ),
    ]
    actual_documents = []
    for document in item.resolution.documents:
        evidence = [
            (row.source_field_index, row.axis_id, row.path, row.locator,
             row.evidence_id)
            for row in document.evidence
        ]
        actual_documents.append((
            document.source_period_index, document.issuer_corp_code,
            document.issuer_corp_name, document.doc_id, document.receipt_no,
            document.period_start, document.period_end,
            document.source_file_id, evidence,
        ))
    if actual_documents != expected_documents:
        raise DeterministicPlanCompilerError(
            "G-O-001 approved document/evidence coordinates가 다릅니다")


def _document_fact_support_specs(
        comparison: DocumentFactComparisonResolution,
        evidence: DocumentReasonEvidence,
        ) -> list[tuple[str, str, str, str, str, str, list[str]]]:
    first, second = comparison.operands
    return [
        ("support-1", "coordinate", _document_coordinate_detail(first),
         "item-1", "field-1", "output-1", ["plan-root-1"]),
        ("support-2", "coordinate", _document_coordinate_detail(second),
         "item-1", "field-1", "output-1", ["plan-root-2"]),
        ("support-3", "comparison",
         "operator=equal;operands=output-1,output-2",
         "item-1", "field-1", "output-1", ["plan-root-3"]),
        ("support-4", "difference",
         "operator=difference;operands=output-1,output-2",
         "item-1", "field-1", "output-1", ["plan-root-4"]),
        ("support-5", "reason_evidence", _document_coordinate_detail(evidence),
         "item-2", "field-2", "output-2", ["plan-root-5"]),
    ]


def _validate_document_fact_cross_authority(
        intent_items: tuple[Any, Any],
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    """Close generic document-fact lowering to every source/proof root."""
    if not isinstance(intent_items, tuple) or len(intent_items) != 2:
        raise ValueError("document fact intent item pair가 다릅니다")
    first_item, second_item = intent_items
    comparison = resolution.items[0].resolution
    reason = resolution.items[1].resolution
    if not isinstance(comparison, DocumentFactComparisonResolution) \
            or not isinstance(reason, DocumentReasonEvidenceResolution):
        raise ValueError("document fact resolution tagged branches가 다릅니다")
    first_operand, second_operand = comparison.operands
    evidence = reason.evidence
    if (
            plan.canonical_build_id != resolution.canonical_build_id
            or plan.resolver_version != resolution.resolver_version
            or plan.applied_defaults
            or len(plan.resolved_plan.tasks) != 2
            or len(plan.resolved_plan.derivations) != 2
            or len(plan.resolved_plan.premise_claims) != 1
            or plan.premise_roots != [ExecutionPremiseRoot(
                premise_id=resolution.premise_proofs[0].premise_id,
                root_ids=[],
                plan_root_ids=["plan-root-1", "plan-root-2"],
                task_refs=["task-1", "task-2"],
            )]
            or plan.resolved_plan.reference_date != resolution.reference_date
            or plan.resolved_plan.corpus_cutoff != resolution.corpus_cutoff
            or plan.resolved_plan.revision != 0
            or plan.resolved_plan.applied_defaults
            or plan.resolved_plan.presentation is not None
    ):
        raise ValueError("document fact execution plan metadata/shape가 다릅니다")
    task1, task2 = plan.resolved_plan.tasks
    if not isinstance(task1, ResolvedCorrectionTask) or (
            task1.task_id, task1.operation, task1.corp_code, task1.corp_name,
            task1.as_of, task1.event_selector,
    ) != (
            "task-1", "diff", first_operand.issuer_corp_code,
            first_operand.issuer_corp_name, resolution.corpus_cutoff, None,
    ):
        raise ValueError("document fact correction task identity가 다릅니다")
    if not isinstance(task2, ResolvedDisclosureTask) or (
            task2.task_id, task2.operation, task2.corp_code, task2.corp_name,
            task2.as_of, task2.event_selector,
    ) != (
            "task-2", "lookup", second_operand.issuer_corp_code,
            second_operand.issuer_corp_name, resolution.corpus_cutoff, None,
    ):
        raise ValueError("document fact disclosure task identity가 다릅니다")
    expected_selectors = [
        (task1.document_selector, first_operand),
        (task2.document_selector, second_operand),
    ]
    for selector, operand in expected_selectors:
        if selector != DocumentSelector(
                doc_id=operand.doc_id, rcept_no=operand.receipt_no):
            raise ValueError("document fact task selector는 exact doc_id/rcept_no여야 합니다")
    if (
            task1.requested_slots != [first_operand.path]
            or [(row.output_id, row.slot, row.value_kind)
                for row in task1.field_outputs]
            != [("output-1", first_operand.path, "money")]
            or task2.requested_slots != [second_operand.path, evidence.path]
            or [(row.output_id, row.slot, row.value_kind)
                for row in task2.field_outputs]
            != [("output-2", second_operand.path, "money"),
                ("output-5", evidence.path, "text")]
    ):
        raise ValueError("document fact task slot/output inventory가 다릅니다")
    expected_derivations = [
        ("output-3", "equal", ("output-1", "output-2"), None),
        ("output-4", "difference", ("output-1", "output-2"), None),
    ]
    actual_derivations = [
        (row.output_id, row.operator,
         tuple(ref.output_id for ref in row.operands), row.rounding_rule)
        for row in plan.resolved_plan.derivations
    ]
    if actual_derivations != expected_derivations:
        raise ValueError("document fact equality/difference derivation이 다릅니다")
    claim = plan.resolved_plan.premise_claims[0]
    if len(resolution.premise_proofs) != 1:
        raise ValueError("document fact resolution premise proof는 정확히 하나여야 합니다")
    # The authoritative source premise is recovered from the contract's source
    # authority; its raw text is checked again by contract.validate_source_intent.
    if (
            claim.claim_id != resolution.premise_proofs[0].premise_id
            or claim.kind != "comparison"
            or len(contract.premise_contracts) != 1
            or claim.raw_text != contract.premise_contracts[0].raw_text
            or claim.kind != contract.premise_contracts[0].kind
            or claim.verify_with != [
                OutputRef(output_id="output-1"),
                OutputRef(output_id="output-2"),
            ]
            or claim.verify_tasks != [
                TaskVerificationRef(task_id="task-1"),
                TaskVerificationRef(task_id="task-2"),
            ]
            or resolution.premise_proofs[0].premise_id != claim.claim_id
            or resolution.premise_proofs[0].proof_refs != [
                first_operand.evidence_id, second_operand.evidence_id,
            ]
    ):
        raise ValueError("document fact premise proof/verification binding이 다릅니다")
    expected_roots = [
        ("output-1", "item-1", "field-1", "output-3", None,
         resolution.items[0].field_proofs[0].proof_ref),
        ("output-2", "item-2", "field-2", "output-5", None,
         resolution.items[1].field_proofs[0].proof_ref),
    ]
    actual_roots = [
        (root.root_id, root.item_id, root.field_id, root.plan_output_id,
         root.plan_task_id, root.proof_ref)
        for root in plan.answer_roots
    ]
    if actual_roots != expected_roots:
        raise ValueError("document fact answer roots가 plan/proof authority와 다릅니다")
    expected_plan_values = [
        ("plan-root-1", "output-1", "money", "task-1", None, None),
        ("plan-root-2", "output-2", "money", "task-2", None, None),
        ("plan-root-3", "output-3", "boolean", None, "output-3", "output-1"),
        ("plan-root-4", "output-4", "money", None, "output-4", None),
        ("plan-root-5", "output-5", "text", "task-2", None, "output-2"),
    ]
    actual_plan_values = [
        (row.plan_root_id, row.plan_output_id, row.value_kind,
         row.producer_task_id, row.producer_derivation_output_id,
         row.answer_mirror_root_id)
        for row in plan.plan_value_roots
    ]
    if actual_plan_values != expected_plan_values:
        raise ValueError("document fact plan value root inventory가 다릅니다")

    expected_support_specs = _document_fact_support_specs(comparison, evidence)
    expected_supports = [
        (support_id, kind, True, detail, [field_id], [root_id], plan_root_ids)
        for support_id, kind, detail, _, field_id, root_id, plan_root_ids
        in expected_support_specs
    ]
    actual_supports = [
        (support.support_id, support.kind, support.required, support.detail,
         support.applies_to_field_ids, support.answer_root_refs,
         support.plan_root_refs)
        for item in contract.items for support in item.support_requirements
    ]
    if actual_supports != expected_supports:
        raise ValueError("document fact contract support inventory가 다릅니다")
    actual_support_roots = [
        (support.support_id, support.kind, support.item_id, support.field_id,
         support.root_id, support.plan_root_ids)
        for support in plan.support_roots
    ]
    expected_support_roots = [
        (support_id, kind, item_id, field_id, root_id, plan_root_ids)
        for support_id, kind, _, item_id, field_id, root_id, plan_root_ids
        in expected_support_specs
    ]
    if actual_support_roots != expected_support_roots:
        raise ValueError("document fact execution support root inventory가 다릅니다")

    if (
            len(contract.items) != 2
            or contract.groups != [CompiledAnswerGroup(
                group_id="group-1", item_ids=["item-1", "item-2"])]
            or contract.presentation != "auto"
            or contract.completion != "complete"
            or len(contract.premise_contracts) != 1
    ):
        raise ValueError("document fact answer contract item/group shape가 다릅니다")
    contract_item1, contract_item2 = contract.items
    if (
            contract_item1.item_id != first_item.item_id
            or contract_item2.item_id != second_item.item_id
            or contract_item1.status != "ready"
            or contract_item2.status != "ready"
            or contract_item1.projection.shape != first_item.output.shape
            or contract_item2.projection.shape != second_item.output.shape
            or contract_item1.projection.presentation != "auto"
            or contract_item2.projection.presentation != "auto"
            or contract_item1.projection.sort is not None
            or contract_item2.projection.sort is not None
    ):
        raise ValueError("document fact answer contract item/projection shape가 다릅니다")
    fields = [contract_item1.projection.fields[0], contract_item2.projection.fields[0]]
    if (
            [(field.field_id, field.field_key, field.binding_status,
              field.answer_root_refs, field.limitation_refs,
              field.activation_predicate)
             for field in fields]
            != [
                ("field-1", "verdict", "executable", ["output-1"], [], None),
                ("field-2", "reason", "executable", ["output-2"], [],
                 ActivationPredicate(
                     predicate_answer_root_ref="output-1",
                     expected_boolean=False)),
            ]
    ):
        raise ValueError("document fact projection field binding이 다릅니다")
    if contract_item1.coverage != CoveragePartition(
            required=["field-1"], executable=["field-1"], limited=[]):
        raise ValueError("document fact verdict coverage가 다릅니다")
    if contract_item2.coverage != CoveragePartition(
            required=["field-2"], executable=["field-2"], limited=[]):
        raise ValueError("document fact reason coverage가 다릅니다")
    premise_contract = contract.premise_contracts[0]
    if (
            premise_contract.premise_id != resolution.premise_proofs[0].premise_id
            or premise_contract.applies_to_item_ids != ["item-1", "item-2"]
            or premise_contract.verification_root_refs
            or premise_contract.verification_plan_root_refs != [
                "plan-root-1", "plan-root-2"]
            or premise_contract.verification_task_refs != ["task-1", "task-2"]
            or premise_contract.verdict_requirement != "required"
    ):
        raise ValueError("document fact compiled premise binding이 다릅니다")


def _document_attribute_coordinate_detail(
        evidence: DocumentAttributeEvidence,
        ) -> str:
    """Stable coordinate-only support detail with no extracted answer text."""
    return (
        f"issuer_corp_code={evidence.issuer_corp_code};"
        f"document_role={evidence.document_role};"
        f"doc_id={evidence.doc_id};receipt_no={evidence.receipt_no};"
        f"path={evidence.path};locator={evidence.locator};"
        f"source_file_id={evidence.source_file_id};"
        f"evidence_id={evidence.evidence_id}"
    )


def _validate_document_attribute_cross_authority(
        intent_authority: Any,
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    """Close two attribute coordinates through plan, roots, and contract."""
    if (
            not isinstance(intent_authority, tuple)
            or len(intent_authority) != 2
            or not isinstance(intent_authority[0], tuple)
    ):
        raise ValueError("document attribute intent authority가 다릅니다")
    intent_items, _ = intent_authority
    evidence_rows = []
    for row in resolution.items:
        if not isinstance(row.resolution, DocumentAttributeEvidenceResolution):
            raise ValueError("document attribute resolution kind가 다릅니다")
        evidence_rows.append(row.resolution.evidence)

    resolved_plan = plan.resolved_plan
    if (
            resolution_item != resolution.items[0]
            or plan.canonical_build_id != resolution.canonical_build_id
            or plan.resolver_version != resolution.resolver_version
            or plan.applied_defaults
            or resolved_plan.reference_date != resolution.reference_date
            or resolved_plan.corpus_cutoff != resolution.corpus_cutoff
            or resolved_plan.revision != 0
            or len(resolved_plan.tasks) != 2
            or resolved_plan.derivations
            or resolved_plan.premise_claims
            or resolved_plan.applied_defaults
            or resolved_plan.presentation is not None
            or plan.premise_roots
            or contract.premise_contracts
    ):
        raise ValueError("document attribute execution metadata/shape가 다릅니다")

    for index, (task, evidence) in enumerate(
            zip(resolved_plan.tasks, evidence_rows, strict=True), start=1):
        if not isinstance(task, ResolvedDisclosureTask):
            raise ValueError("document attribute task는 disclosure여야 합니다")
        if (
                task.task_id != f"task-{index}"
                or task.operation != "lookup"
                or task.corp_code != evidence.issuer_corp_code
                or task.corp_name != evidence.issuer_corp_name
                or task.as_of != resolution.corpus_cutoff
                or task.document_selector != DocumentSelector(
                    doc_id=evidence.doc_id, rcept_no=evidence.receipt_no)
                or task.event_selector is not None
                or task.requested_slots != [evidence.path]
                or task.output_id is not None
                or [
                    (row.output_id, row.slot, row.value_kind)
                    for row in task.field_outputs
                ] != [(f"output-{index}", evidence.path, "text")]
        ):
            raise ValueError(
                "document attribute task selector/slot/output binding이 다릅니다")

    expected_answer_roots = [
        (
            f"output-{index}", f"item-{index}", f"field-{index}",
            f"output-{index}", None,
            resolution.items[index - 1].field_proofs[0].proof_ref,
        )
        for index in (1, 2)
    ]
    actual_answer_roots = [
        (row.root_id, row.item_id, row.field_id, row.plan_output_id,
         row.plan_task_id, row.proof_ref)
        for row in plan.answer_roots
    ]
    if actual_answer_roots != expected_answer_roots:
        raise ValueError("document attribute answer root inventory가 다릅니다")

    expected_plan_roots = [
        (f"plan-root-{index}", f"output-{index}", "text",
         f"task-{index}", None, f"output-{index}")
        for index in (1, 2)
    ]
    actual_plan_roots = [
        (row.plan_root_id, row.plan_output_id, row.value_kind,
         row.producer_task_id, row.producer_derivation_output_id,
         row.answer_mirror_root_id)
        for row in plan.plan_value_roots
    ]
    if actual_plan_roots != expected_plan_roots:
        raise ValueError("document attribute plan value root inventory가 다릅니다")

    expected_supports = [
        (
            f"support-{index}", "coordinate", True,
            _document_attribute_coordinate_detail(evidence_rows[index - 1]),
            [f"field-{index}"], [f"output-{index}"],
            [f"plan-root-{index}"],
        )
        for index in (1, 2)
    ]
    actual_supports = [
        (row.support_id, row.kind, row.required, row.detail,
         row.applies_to_field_ids, row.answer_root_refs, row.plan_root_refs)
        for item in contract.items for row in item.support_requirements
    ]
    if actual_supports != expected_supports:
        raise ValueError("document attribute contract support inventory가 다릅니다")
    expected_support_roots = [
        (f"support-{index}", "coordinate", f"item-{index}",
         f"field-{index}", f"output-{index}", [f"plan-root-{index}"])
        for index in (1, 2)
    ]
    actual_support_roots = [
        (row.support_id, row.kind, row.item_id, row.field_id, row.root_id,
         row.plan_root_ids)
        for row in plan.support_roots
    ]
    if actual_support_roots != expected_support_roots:
        raise ValueError("document attribute execution support roots가 다릅니다")

    if (
            len(contract.items) != 2
            or contract.groups != [CompiledAnswerGroup(
                group_id="group-1", item_ids=["item-1", "item-2"])]
            or contract.presentation != "auto"
            or contract.completion != "complete"
    ):
        raise ValueError("document attribute contract item/group shape가 다릅니다")
    for index, (compiled_item, source_item) in enumerate(
            zip(contract.items, intent_items, strict=True), start=1):
        expected_field = ProjectionField(
            field_id=f"field-{index}", field_key=f"attribute_{index}",
            binding_status="executable",
            answer_root_refs=[f"output-{index}"],
        )
        if (
                compiled_item.item_id != source_item.item_id
                or compiled_item.status != "ready"
                or compiled_item.projection.shape != source_item.output.shape
                or compiled_item.projection.fields != [expected_field]
                or compiled_item.projection.presentation != "auto"
                or compiled_item.projection.sort is not None
                or compiled_item.coverage != CoveragePartition(
                    required=[f"field-{index}"],
                    executable=[f"field-{index}"], limited=[])
                or compiled_item.limitation_bindings
        ):
            raise ValueError(
                "document attribute contract field/coverage binding이 다릅니다")


def _periodic_narrative_coordinate_detail(
        document: PeriodicNarrativeDocument,
        evidence: PeriodicNarrativeEvidence,
        ) -> str:
    """Stable source-only citation detail; extracted report prose is excluded."""
    return (
        f"issuer_corp_code={document.issuer_corp_code};"
        f"source_period_index={document.source_period_index};"
        f"doc_id={document.doc_id};receipt_no={document.receipt_no};"
        f"axis_id={evidence.axis_id};path={evidence.path};"
        f"locator={evidence.locator};source_file_id={evidence.source_file_id};"
        f"evidence_id={evidence.evidence_id}"
    )


def _periodic_narrative_root_id(
        source_field_index: int, source_period_index: int,
        ) -> str:
    return f"output-{source_field_index * 2 + source_period_index + 1}"


def _periodic_narrative_support_specs(
        typed: PeriodicNarrativeComparisonResolution,
        field_count: int,
        ) -> list[tuple[str, str, str, str]]:
    """Bind source citations to each axis and to the derived summary field."""
    specs: list[tuple[str, str, str, str]] = []
    source_field_indexes = {
        evidence.source_field_index
        for document in typed.documents
        for evidence in document.evidence
    }
    summary_field_index = field_count - 1
    has_derived_summary = (
        field_count > 0
        and source_field_indexes
        and summary_field_index not in source_field_indexes
        and source_field_indexes == set(range(summary_field_index))
    )
    for document in typed.documents:
        for evidence in document.evidence:
            detail = _periodic_narrative_coordinate_detail(document, evidence)
            field_index = evidence.source_field_index
            specs.append((
                f"support-{len(specs) + 1}",
                f"field-{field_index + 1}",
                _periodic_narrative_root_id(
                    field_index, document.source_period_index),
                detail,
            ))
            if has_derived_summary:
                specs.append((
                    f"support-{len(specs) + 1}",
                    f"field-{summary_field_index + 1}",
                    _periodic_narrative_root_id(
                        summary_field_index, document.source_period_index),
                    detail,
                ))
    return specs


def _validate_periodic_narrative_comparison_cross_authority(
        intent_authority: Any,
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    """Close two exact reports through tasks, roots, citations, and fields."""
    if not isinstance(intent_authority, tuple) or len(intent_authority) != 2:
        raise ValueError("periodic narrative intent authority가 다릅니다")
    intent_item, _ = intent_authority
    typed = resolution_item.resolution
    if not isinstance(typed, PeriodicNarrativeComparisonResolution):
        raise ValueError("periodic narrative resolution kind가 다릅니다")
    resolved_plan = plan.resolved_plan
    if (
            resolution_item != resolution.items[0]
            or plan.canonical_build_id != resolution.canonical_build_id
            or plan.resolver_version != resolution.resolver_version
            or plan.applied_defaults
            or resolved_plan.reference_date != resolution.reference_date
            or resolved_plan.corpus_cutoff != resolution.corpus_cutoff
            or resolved_plan.revision != 0
            or len(resolved_plan.tasks) != 2
            or resolved_plan.derivations
            or resolved_plan.premise_claims
            or resolved_plan.applied_defaults
            or resolved_plan.presentation is not None
            or plan.plan_value_roots
            or plan.premise_roots
            or contract.premise_contracts
    ):
        raise ValueError("periodic narrative execution metadata/shape가 다릅니다")

    for index, (task, document) in enumerate(
            zip(resolved_plan.tasks, typed.documents, strict=True), start=1):
        if not isinstance(task, ResolvedNarrativeTask):
            raise ValueError("periodic narrative task는 narrative여야 합니다")
        if (
                task.task_id != f"task-{index}"
                or task.operation != "search"
                or task.corp_codes != [document.issuer_corp_code]
                or task.corp_names != [document.issuer_corp_name]
                or task.as_of != resolution.corpus_cutoff
            or task.retrieval_query != intent_item.target.surface
                or task.document_selector != DocumentSelector(
                    doc_id=document.doc_id, rcept_no=document.receipt_no)
                or task.periods != [DateRange(
                    start=document.period_start, end=document.period_end)]
                or task.requested_slots != [
                    row.path for row in document.evidence]
        ):
            raise ValueError(
                "periodic narrative task selector/period/slot binding이 다릅니다")

    field_count = len(intent_item.output.field_surfaces)
    expected_answer_roots = []
    for field_index, proof in enumerate(resolution_item.field_proofs):
        for document in typed.documents:
            root_id = _periodic_narrative_root_id(
                field_index, document.source_period_index)
            expected_answer_roots.append((
                root_id, intent_item.item_id, f"field-{field_index + 1}",
                None, f"task-{document.source_period_index + 1}",
                f"{proof.proof_ref}:period-{document.source_period_index}",
            ))
    actual_answer_roots = [
        (row.root_id, row.item_id, row.field_id, row.plan_output_id,
         row.plan_task_id, row.proof_ref)
        for row in plan.answer_roots
    ]
    if actual_answer_roots != expected_answer_roots:
        raise ValueError("periodic narrative answer root inventory가 다릅니다")

    support_specs = _periodic_narrative_support_specs(typed, field_count)
    expected_supports = [
        (support_id, "citation", True, detail, [field_id], [root_id], [])
        for support_id, field_id, root_id, detail in support_specs
    ]
    actual_supports = [
        (row.support_id, row.kind, row.required, row.detail,
         row.applies_to_field_ids, row.answer_root_refs, row.plan_root_refs)
        for item in contract.items for row in item.support_requirements
    ]
    if actual_supports != expected_supports:
        raise ValueError("periodic narrative contract citation inventory가 다릅니다")
    expected_support_roots = [
        (support_id, "citation", intent_item.item_id, field_id, root_id, [])
        for support_id, field_id, root_id, _ in support_specs
    ]
    actual_support_roots = [
        (row.support_id, row.kind, row.item_id, row.field_id, row.root_id,
         row.plan_root_ids)
        for row in plan.support_roots
    ]
    if actual_support_roots != expected_support_roots:
        raise ValueError("periodic narrative execution citation roots가 다릅니다")

    if (
            len(contract.items) != 1
            or contract.groups
            or contract.presentation != "auto"
            or contract.completion != "complete"
    ):
        raise ValueError("periodic narrative contract item/group shape가 다릅니다")
    compiled_item = contract.items[0]
    expected_fields = []
    for field_index in range(field_count):
        field_key = f"axis_{field_index + 1}"
        expected_fields.append(ProjectionField(
            field_id=f"field-{field_index + 1}", field_key=field_key,
            binding_status="executable",
            answer_root_refs=[
                _periodic_narrative_root_id(field_index, period_index)
                for period_index in range(2)
            ],
        ))
    field_ids = [f"field-{index + 1}" for index in range(field_count)]
    if (
            compiled_item.item_id != intent_item.item_id
            or compiled_item.status != "ready"
            or compiled_item.projection.shape != intent_item.output.shape
            or compiled_item.projection.fields != expected_fields
            or compiled_item.projection.presentation != "auto"
            or compiled_item.projection.sort is not None
            or compiled_item.coverage != CoveragePartition(
                required=field_ids, executable=field_ids, limited=[])
            or compiled_item.limitation_bindings
    ):
        raise ValueError(
            "periodic narrative contract field/coverage binding이 다릅니다")


def _validate_document_collection_cross_authority(
        intent_authority: Any,
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    """Close the task-bound whole target across resolution, plan, and contract."""
    if (
            not isinstance(intent_authority, tuple)
            or len(intent_authority) != 2
    ):
        raise ValueError("document collection intent authority가 다릅니다")
    intent_item, _entity = intent_authority
    expected_plan = _build_document_collection_typed_plan(
        resolution, resolution_item, intent_item)
    if plan.model_dump(mode="json") != expected_plan.model_dump(mode="json"):
        raise ValueError(
            "document collection execution plan이 typed resolution과 다릅니다")
    if (
            len(contract.items) != 1
            or contract.groups
            or contract.premise_contracts
            or contract.presentation != intent_item.output.presentation
            or contract.completion != "complete"
    ):
        raise ValueError(
            "document collection contract item/group/completion이 다릅니다")
    compiled_item = contract.items[0]
    projection = compiled_item.projection
    if intent_item.output.projection_mode == "named_fields":
        expected_field = ProjectionField(
            field_id="field-1", field_key="narrative",
            binding_status="executable", answer_root_refs=["output-1"])
        if (
                compiled_item.item_id != intent_item.item_id
                or compiled_item.status != "ready"
                or projection.projection_mode != "named_fields"
                or projection.shape != "narrative"
                or projection.fields != [expected_field]
                or projection.whole_target is not None
                or compiled_item.coverage != CoveragePartition(
                    required=["field-1"], executable=["field-1"], limited=[])
                or compiled_item.support_requirements
                or compiled_item.limitation_bindings):
            raise ValueError("document collection narrative contract binding이 다릅니다")
        return
    whole_target = projection.whole_target
    if (
            compiled_item.item_id != intent_item.item_id
            or compiled_item.status != "ready"
            or projection.projection_mode != "whole_target"
            or projection.shape != intent_item.output.shape
            or projection.fields
            or whole_target is None
            or whole_target.whole_target_id != "whole-target-1"
            or whole_target.required is not True
            or whole_target.binding_status != "executable"
            or whole_target.answer_root_refs != ["output-1"]
            or whole_target.limitation_refs
            or whole_target.activation_predicate is not None
            or projection.presentation != intent_item.output.presentation
            or projection.sort is not None
            or compiled_item.coverage != CoveragePartition(
                required=["whole-target-1"],
                executable=["whole-target-1"], limited=[])
            or compiled_item.support_requirements
            or compiled_item.limitation_bindings
    ):
        raise ValueError(
            "document collection whole-target contract binding이 다릅니다")


def _holding_contract_shape(
        typed: HoldingDisclosureResolution,
        ) -> tuple[list[ProjectionField], list[str], list[LimitationBinding]]:
    """Return field bindings and typed limitations for one holding answer.

    개인정보 제한과 코퍼스 밖 원공시는 서로 독립된 축이다. 한 공개 field에
    둘 다 적용되면 두 limitation ref를 유지하고, 값 자체는 citation-bound
    answer root로 계속 실행하는 qualified field가 된다.
    """

    privacy_indexes = [
        index for index, row in enumerate(typed.slot_bindings)
        if row.binding_status != "executable"]
    lineage_indexes = (
        [index for index, row in enumerate(typed.slot_bindings)
         if row.binding_status != "limited"]
        if typed.lineage_status == "root_missing" else [])
    limitation_ids: dict[str, str] = {}
    limitations: list[LimitationBinding] = []
    if privacy_indexes:
        limitation_ids["privacy"] = f"limitation-{len(limitations) + 1}"
        limitations.append(LimitationBinding(
            limitation_id=limitation_ids["privacy"],
            code="personal_data_omitted",
            family="capability",
            applies_to_field_ids=[
                f"field-{index + 1}" for index in privacy_indexes],
            detail="restricted personal data omitted from public disclosure answer",
        ))
    if lineage_indexes:
        limitation_ids["lineage"] = f"limitation-{len(limitations) + 1}"
        limitations.append(LimitationBinding(
            limitation_id=limitation_ids["lineage"],
            code="holding_lineage_root_missing",
            family="source_scope",
            applies_to_field_ids=[
                f"field-{index + 1}" for index in lineage_indexes],
            detail="original holding filing is outside the provided corpus",
        ))

    fields: list[ProjectionField] = []
    limited_ids: list[str] = []
    for index, row in enumerate(typed.slot_bindings):
        field_id = f"field-{index + 1}"
        refs: list[str] = []
        if index in privacy_indexes:
            refs.append(limitation_ids["privacy"])
        if index in lineage_indexes:
            refs.append(limitation_ids["lineage"])
        if row.binding_status == "limited":
            status = "limited"
        elif refs:
            status = "qualified"
        else:
            status = "executable"
        if status != "executable":
            limited_ids.append(field_id)
        fields.append(ProjectionField(
            field_id=field_id,
            field_key=row.slot,
            binding_status=status,
            answer_root_refs=([] if row.binding_status == "limited"
                              else [f"output-{index + 1}"]),
            limitation_refs=refs,
        ))
    return fields, limited_ids, limitations


def _validate_holding_disclosure_cross_authority(
        intent_authority: Any,
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    if not isinstance(intent_authority, tuple) or len(intent_authority) != 2:
        raise ValueError("holding disclosure intent authority가 다릅니다")
    intent_item, _companies = intent_authority
    expected_plan = _build_holding_disclosure_typed_plan(
        resolution, resolution_item, intent_item)
    if plan.model_dump(mode="json") != expected_plan.model_dump(mode="json"):
        raise ValueError(
            "holding disclosure execution plan이 typed resolution과 다릅니다")
    field_count = len(intent_item.output.field_surfaces)
    bindings = resolution_item.resolution.slot_bindings
    expected_fields, limited_ids, limitation = _holding_contract_shape(
        resolution_item.resolution)
    expected_ids = [f"field-{index + 1}" for index in range(field_count)]
    executable_ids = [
        f"field-{index + 1}" for index, row in enumerate(bindings)
        if row.binding_status != "limited"]
    expected_completion = "partial" if limited_ids else "complete"
    if (len(contract.items) != 1 or contract.groups
            or contract.premise_contracts
            or contract.completion != expected_completion):
        raise ValueError("holding disclosure contract inventory가 다릅니다")
    compiled = contract.items[0]
    expected_status = "partial" if limited_ids else "ready"
    if (compiled.item_id != intent_item.item_id or compiled.status != expected_status
            or compiled.projection.projection_mode != "named_fields"
            or compiled.projection.shape != intent_item.output.shape
            or compiled.projection.fields != expected_fields
            or compiled.coverage != CoveragePartition(
                required=expected_ids, executable=executable_ids,
                limited=limited_ids)
            or compiled.limitation_bindings != limitation):
        raise ValueError("holding disclosure answer field binding이 다릅니다")


def _validate_document_version_history_cross_authority(
        intent_authority: Any, resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem, plan: ExecutionPlan,
        contract: CompiledAnswerContract) -> None:
    intent_item, _entity = intent_authority
    # 인용 원문 자체는 접지 검증이 이미 강제한다. 여기서는 구조를 대조한다.
    claims = plan.resolved_plan.premise_claims
    expected = _build_document_version_history_typed_plan(
        resolution, resolution_item, intent_item,
        claims[0].raw_text if claims else "")
    if plan.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise ValueError("document version execution plan이 typed lineage와 다릅니다")
    if (len(contract.items) != 1 or len(contract.premise_contracts) != 1
            or contract.items[0].projection.fields[0].field_key != "correction_status"):
        raise ValueError("document version answer contract가 다릅니다")


def _validate_selected_event_cross_authority(
        intent_authority: Any,
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    if not isinstance(intent_authority, tuple) or len(intent_authority) != 3:
        raise ValueError("selected event intent authority가 다릅니다")
    intent_item, _entity, premises = intent_authority
    expected_plan, expected_supports = _build_selected_event_typed_plan(
        resolution, resolution_item, intent_item, premises)
    if plan.model_dump(mode="json") != expected_plan.model_dump(mode="json"):
        raise ValueError("selected event execution plan이 typed resolution과 다릅니다")
    field_ids = [f"field-{index + 1}"
                 for index in range(len(intent_item.output.field_surfaces))]
    typed = resolution_item.resolution
    assert isinstance(typed, SelectedEventResolution)
    partial = (typed.operation == "timeline"
               and typed.lineage_missing_root_date is not None)
    expected_premise_contracts = [CompiledPremiseContract(
        premise_id=premise.premise_id, kind=premise.kind,
        raw_text=premise.raw_text,
        applies_to_item_ids=list(premise.applies_to_item_ids),
        verification_root_refs=_premise_bound_answer_roots(
            resolution, resolution_item, plan.answer_roots,
            premise.premise_id),
        verification_task_refs=["task-1"],
        verification_plan_root_refs=[], verdict_requirement="required",
    ) for premise in premises]
    expected_fields = [ProjectionField(
        field_id=field_id, field_key=f"value_{index + 1}",
        binding_status=("qualified" if partial else "executable"),
        answer_root_refs=[f"output-{index + 1}"],
        limitation_refs=([f"limitation-{index + 1}"] if partial else []),
    ) for index, field_id in enumerate(field_ids)]
    if (
            len(contract.items) != 1 or contract.groups
            or contract.premise_contracts != expected_premise_contracts
            or contract.presentation != intent_item.output.presentation
            or contract.completion != ("partial" if partial else "complete")
    ):
        raise ValueError("selected event answer contract shape가 다릅니다")
    item = contract.items[0]
    if (
            item.item_id != intent_item.item_id or item.status != ("partial" if partial else "ready")
            or item.projection.shape != intent_item.output.shape
            or item.projection.fields != expected_fields
            or item.projection.presentation != intent_item.output.presentation
            or item.coverage != CoveragePartition(
                required=field_ids, executable=field_ids,
                limited=(field_ids if partial else []))
            or item.support_requirements != expected_supports
            or item.limitation_bindings != ([] if not partial else [
                LimitationBinding(limitation_id="limitation-1", code="source_scope_raw_absent", family="source_scope", applies_to_field_ids=["field-1"], detail=f"original disclosure before corpus: {typed.lineage_missing_root_date}"),
                LimitationBinding(limitation_id="limitation-2", code="source_scope_prevents_complete_lineage", family="source_scope", applies_to_field_ids=["field-2"], detail=f"complete lineage unavailable because root before corpus: {typed.lineage_missing_root_date}"),
            ])
    ):
        raise ValueError("selected event answer contract binding이 다릅니다")


def _validate_event_collection_cross_authority(
        intent_item: Any,
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    typed = resolution_item.resolution
    if not isinstance(typed, EventCollectionResolution):
        raise ValueError("event collection typed resolution이 필요합니다")
    if typed.public_task_kind == "correction":
        expected_tasks = [(
            "task-1", typed.corp_code, typed.corp_name,
            typed.event_type, typed.as_of,
            list(typed.requested_slots),
        )]
        actual_tasks = [
            (task.task_id, task.corp_code, task.corp_name,
             task.event_selector.event_type if task.event_selector else None,
             task.as_of, task.requested_slots)
            for task in plan.resolved_plan.tasks
            if (isinstance(task, ResolvedCorrectionTask)
                and task.operation == "history"
                and task.document_selector == DocumentSelector(
                    doc_group="exchange", rcept_to=typed.as_of,
                    is_correction=True)
                and task.event_selector is not None
                and task.event_selector.event_key is None
                and task.event_selector.seed_rcept_no is None
                and task.event_selector.counterparty is None
                and task.event_selector.contract_name is None
                and not task.event_selector.keywords
                and task.event_selector.event_from is None
                and task.event_selector.event_to is None)
        ]
    elif typed.public_task_kind == "disclosure":
        expected_tasks = [(
            "task-1", typed.event_type, typed.counterparty, typed.keywords,
            typed.event_from, typed.event_to, typed.as_of,
            list(typed.requested_slots), [], typed.argmax_slot,
            typed.argmax_direction,
        )]
        actual_tasks = [
            (task.task_id,
             task.event_selector.event_type if task.event_selector else None,
             task.event_selector.counterparty if task.event_selector else None,
             task.event_selector.keywords if task.event_selector else [],
             task.event_selector.event_from if task.event_selector else None,
             task.event_selector.event_to if task.event_selector else None,
             task.as_of, task.requested_slots,
             [row.output_id for row in task.field_outputs],
             getattr(task, "argmax_slot", None),
             getattr(task, "argmax_direction", "maximum"))
            for task in plan.resolved_plan.tasks
            if isinstance(task, ResolvedDisclosureTask)
        ]
    else:
        expected_tasks = [(
            "task-1", typed.event_type, typed.keywords, typed.event_from,
            typed.event_to, [typed.as_of], list(typed.requested_slots), [],
            typed.argmax_slot, typed.argmax_direction,
        )]
        actual_tasks = [
            (task.task_id, task.selector.event_type, task.selector.keywords,
             task.selector.event_from, task.selector.event_to, task.timepoints,
             task.requested_slots, [row.output_id for row in task.field_outputs],
             getattr(task, "argmax_slot", None),
             getattr(task, "argmax_direction", "maximum"))
            for task in plan.resolved_plan.tasks
            if isinstance(task, ResolvedEventTask)
        ]
    if (len(actual_tasks) != len(plan.resolved_plan.tasks)
            or actual_tasks != expected_tasks):
        raise ValueError("event collection task binding이 다릅니다")
    collapsed_projection = (
        len(intent_item.output.field_surfaces) == 1
        and len(typed.requested_slots) != 1)
    expected_roots = (
        [
            (f"output-{field_index}",
             "field-1" if collapsed_projection else f"field-{field_index}",
             "task-1")
            for field_index in range(1, len(typed.requested_slots) + 1)
        ] if not typed.events else [
            (f"output-{(event_index - 1) * len(typed.requested_slots) + field_index}",
             "field-1" if collapsed_projection else f"field-{field_index}",
             "task-1")
            for event_index in range(1, len(typed.events) + 1)
            for field_index in range(1, len(typed.requested_slots) + 1)
        ]
    )
    actual_roots = [
        (root.root_id, root.field_id, root.plan_task_id)
        for root in plan.answer_roots
    ]
    if actual_roots != expected_roots:
        raise ValueError("event collection answer root binding이 다릅니다")
    if (len(contract.items) != 1 or contract.groups or contract.premise_contracts
            or contract.presentation != intent_item.output.presentation
            or contract.completion != "complete"):
        raise ValueError("event collection answer contract shape가 다릅니다")
    item = contract.items[0]
    if len(intent_item.output.field_surfaces) == len(typed.requested_slots):
        expected_fields = [
            (f"field-{field_index}", slot,
             ([f"output-{field_index}"] if not typed.events else [
                 f"output-{(event_index - 1) * len(typed.requested_slots) + field_index}"
                 for event_index in range(1, len(typed.events) + 1)
             ]))
            for field_index, slot in enumerate(typed.requested_slots, start=1)
        ]
    elif len(intent_item.output.field_surfaces) == 1:
        expected_fields = [("field-1", intent_item.output.field_surfaces[0],
                            [root.root_id for root in plan.answer_roots])]
    else:
        raise ValueError("event collection source/slot projection cardinality가 다릅니다")
    actual_fields = [
        (field.field_id, field.field_key, field.answer_root_refs)
        for field in item.projection.fields
    ]
    if (
            item.item_id != intent_item.item_id or item.status != "ready"
            or item.projection.shape != intent_item.output.shape
            or actual_fields != expected_fields
            or item.coverage != CoveragePartition(
                required=[field[0] for field in expected_fields],
                executable=[field[0] for field in expected_fields], limited=[])
    ):
        raise ValueError("event collection answer projection binding이 다릅니다")


def _validate_event_amount_change_cross_authority(
        intent_authority: Any,
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    expected_plan, expected_supports = _build_event_amount_change_typed_plan(
        resolution, resolution_item)
    if plan.model_dump(mode="json") != expected_plan.model_dump(mode="json"):
        raise ValueError(
            "event amount change execution plan이 typed resolution과 다릅니다")
    item = contract.items[0] if len(contract.items) == 1 else None
    expected_field = ProjectionField(
        field_id="field-1", field_key="change",
        binding_status="executable", answer_root_refs=["output-3"])
    if (
            item is None
            or contract.groups
            or contract.premise_contracts
            or contract.completion != "complete"
            or item.item_id != intent_authority.item_id
            or item.status != "ready"
            or item.projection.shape != intent_authority.output.shape
            or item.projection.fields != [expected_field]
            or item.projection.presentation
            != intent_authority.output.presentation
            or item.coverage != CoveragePartition(
                required=["field-1"], executable=["field-1"], limited=[])
            or item.support_requirements != expected_supports
            or item.limitation_bindings
    ):
        raise ValueError("event amount change answer contract binding이 다릅니다")


def _validate_termination_reported_status_cross_authority(
        intent_authority: Any,
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    if not isinstance(intent_authority, tuple) or len(intent_authority) != 3:
        raise ValueError("reported termination intent authority가 다릅니다")
    intent_item, _issuer, premises = intent_authority
    typed = resolution_item.resolution
    if not isinstance(typed, TerminationReportedStatusResolution):
        raise ValueError("reported termination resolution kind가 다릅니다")
    expected_plan, _expected_supports = \
        _build_termination_reported_status_typed_plan(
            resolution, resolution_item, intent_item, premises)
    if plan.model_dump(mode="json") != expected_plan.model_dump(mode="json"):
        raise ValueError(
            "reported termination execution plan이 typed resolution과 다릅니다")
    if (
            plan.canonical_build_id != resolution.canonical_build_id
            or plan.resolver_version != resolution.resolver_version
            or plan.applied_defaults
            or len(plan.resolved_plan.tasks) != 1
            or plan.resolved_plan.reference_date != resolution.reference_date
            or plan.resolved_plan.corpus_cutoff != resolution.corpus_cutoff
            or plan.resolved_plan.revision != 0
            or plan.resolved_plan.derivations
            or plan.resolved_plan.applied_defaults
            or plan.resolved_plan.presentation is not None
    ):
        raise ValueError("reported termination execution plan metadata가 다릅니다")
    task = plan.resolved_plan.tasks[0]
    _normalized_fields, field_roles = _termination_reported_status_field_roles(
        intent_item)
    amount_indexes = [index for index, role in enumerate(field_roles)
                      if role == "termination_amount"]
    expected_slots = ["해지금액" for _index in amount_indexes]
    if amount_indexes:
        expected_slots.append("해지사유")
    expected_timepoint = resolution.corpus_cutoff
    if intent_item.scope.as_of_expression is not None:
        start, end, error = _target_date_range(
            intent_item.scope.as_of_expression,
            reference_date=resolution.reference_date)
        if error is not None or start is None or start != end:
            raise ValueError("reported termination observation date가 다릅니다")
        expected_timepoint = start
    if not isinstance(task, ResolvedEventTask) or (
            task.task_id, task.operation, task.corp_code, task.corp_name,
            task.timepoints, task.requested_slots, task.output_id,
            task.field_outputs,
    ) != (
            "task-1", "status", typed.issuer_corp_code,
            typed.issuer_corp_name, [expected_timepoint], expected_slots,
            None, [],
    ):
        raise ValueError("reported termination event task binding이 다릅니다")
    selector = task.selector
    if (
            selector.event_key != typed.event_key
            or selector.seed_rcept_no is not None
            or selector.event_type is not None
            or selector.counterparty is not None
            or selector.contract_name is not None
            or selector.keywords
            or selector.event_from is not None
            or selector.event_to is not None
    ):
        raise ValueError("reported termination selector는 exact event_key여야 합니다")
    expected_roots = [
        (f"output-{index + 1}", intent_item.item_id, f"field-{index + 1}",
         "task-1", proof.proof_ref)
        for index, proof in enumerate(resolution_item.field_proofs)
    ]
    actual_roots = [
        (root.root_id, root.item_id, root.field_id, root.plan_task_id,
         root.proof_ref)
        for root in plan.answer_roots
    ]
    if actual_roots != expected_roots:
        raise ValueError("reported termination answer root binding이 다릅니다")
    field_ids = [f"field-{index + 1}" for index in range(len(field_roles))]
    expected_fields = [ProjectionField(
        field_id=field_id,
        field_key=field_roles[index],
        binding_status="qualified", answer_root_refs=[f"output-{index + 1}"],
        limitation_refs=["limitation-1"],
    ) for index, field_id in enumerate(field_ids)]
    if (
            len(contract.items) != 1 or contract.groups
            or contract.presentation != intent_item.output.presentation
            or contract.completion != "partial"
    ):
        raise ValueError("reported termination answer contract shape가 다릅니다")
    item = contract.items[0]
    if (
            item.item_id != intent_item.item_id or item.status != "partial"
            or item.projection.shape != intent_item.output.shape
            or item.projection.fields != expected_fields
            or item.coverage != CoveragePartition(
                required=field_ids, executable=field_ids, limited=field_ids)
            or item.support_requirements != [SupportRequirement(
                support_id=f"support-{index + 1}", kind="citation",
                detail=(f"status_receipt={typed.status_receipt};"
                        f"event_key={typed.event_key};"
                        f"proof_ref={typed.status_proof.proof_ref}"),
                applies_to_field_ids=[field_id],
                answer_root_refs=[f"output-{index + 1}"],
            ) for index, field_id in enumerate(field_ids)]
            or item.limitation_bindings != [LimitationBinding(
                limitation_id="limitation-1", code="ambiguous_event_origin",
                family="identity_lineage", applies_to_field_ids=field_ids,
                detail=typed.identity_provenance.detail,
            )]
    ):
        raise ValueError("reported termination answer contract binding이 다릅니다")
    expected_premise_contracts = [CompiledPremiseContract(
        premise_id=premise.premise_id, kind=premise.kind,
        raw_text=premise.raw_text,
        applies_to_item_ids=list(premise.applies_to_item_ids),
        verification_root_refs=_premise_bound_answer_roots(
            resolution, resolution_item, plan.answer_roots,
            premise.premise_id),
        verification_task_refs=["task-1"],
        verification_plan_root_refs=[], verdict_requirement="required",
    ) for premise in premises]
    if contract.premise_contracts != expected_premise_contracts:
        raise ValueError("reported termination premise contract binding이 다릅니다")


def _validate_correction_lineage_cross_authority(
        intent_authority: Any, resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem, plan: ExecutionPlan,
        contract: CompiledAnswerContract) -> None:
    """Close correction task selectors and answer roots to typed lineage."""
    typed = resolution_item.resolution
    if not isinstance(typed, CorrectionLineageResolution):
        raise ValueError("correction lineage typed authority가 필요합니다")
    tasks = plan.resolved_plan.tasks
    ranged_whole_history = bool(
        typed.answer_role == "diff" and typed.sequence
        and typed.date_roles is not None and typed.date_roles.range_requested)
    expected_as_of = (
        resolution.corpus_cutoff if ranged_whole_history
        else typed.correction_date if typed.operation == "diff"
        else resolution.corpus_cutoff)
    if (len(tasks) != 1 or not isinstance(tasks[0], ResolvedCorrectionTask)
            or tasks[0].task_id != "task-1" or tasks[0].operation != typed.operation
            or tasks[0].corp_code != typed.corp_code or tasks[0].corp_name != typed.corp_name
            or tasks[0].as_of != expected_as_of):
        raise ValueError("correction lineage execution task가 typed authority와 다릅니다")
    task = tasks[0]
    if typed.answer_role == "diff":
        if ranged_whole_history:
            roles = typed.date_roles
            assert roles is not None
            valid_selector = (
                task.document_selector is not None
                and task.document_selector.rcept_no is None
                and task.document_selector.is_correction is True
                and task.document_selector.rcept_from == roles.correction_from
                and task.document_selector.rcept_to == roles.correction_to
                and task.event_selector is not None
                and task.event_selector.event_key == typed.event_key
                and task.event_selector.event_from == roles.root_observed_at
                and task.event_selector.event_to == roles.root_observed_at)
        else:
            valid_selector = (
                typed.operation == "diff"
                and
                task.document_selector is not None
                and task.document_selector.rcept_no == typed.correction_receipt
                and task.event_selector is None)
        if (not valid_selector or len(plan.answer_roots) != 1
                or contract.items[0].projection.projection_mode != "whole_target"):
            raise ValueError("correction diff plan/contract binding이 다릅니다")
    else:
        if (task.event_selector is None or task.event_selector.event_key is not None
                or task.event_selector.seed_rcept_no is not None
                or task.event_selector.counterparty != typed.counterparty
                or task.event_selector.keywords != typed.product_keywords
                or task.document_selector is not None or len(plan.answer_roots) != 2
                or len(contract.items) != 2):
            raise ValueError("correction history plan/contract binding이 다릅니다")
        if [row.item_id for row in plan.answer_roots] != ["item-1", "item-2"]:
            raise ValueError("correction history answer root order가 다릅니다")


def _financial_period_support_detail(coordinate: Any) -> str:
    """Serialize duration and instant financial coordinates without guessing."""

    end = coordinate.period_end.isoformat()
    if coordinate.period_start is None:
        return f"period={end} ({coordinate.period_type})"
    return (
        f"period={coordinate.period_start.isoformat()}..{end} "
        f"({coordinate.period_type})"
    )


def _validate_narrative_matrix_cross_authority(
        intent_item: Any, resolution_item: ResolvedItem, plan: ExecutionPlan,
        contract: CompiledAnswerContract) -> None:
    """Close every matrix unit to its exact task, root and citation support."""
    typed = resolution_item.resolution
    if not isinstance(typed, NarrativeMatrixResolution):
        raise ValueError("narrative matrix cross authority kind가 다릅니다")
    units = [(cell, topic) for cell in typed.cells for topic in cell.topics]
    tasks = plan.resolved_plan.tasks
    if len(tasks) != len(units) or len(plan.answer_roots) != len(units):
        raise ValueError("narrative matrix task/root 수가 다릅니다")
    fields = {surface: index for index, surface in enumerate(intent_item.output.field_surfaces)}
    for index, ((cell, topic), task, root) in enumerate(zip(
            units, tasks, plan.answer_roots, strict=True), start=1):
        field_id = f"field-{fields[topic] + 1}"
        if (not isinstance(task, ResolvedNarrativeTask)
                or task.task_id != f"task-{index}"
                or task.corp_codes != [cell.corp_code]
                or task.corp_names != [cell.corp_name]
                or task.retrieval_query != topic
                or task.requested_slots
                or task.document_selector != DocumentSelector(
                    doc_id=cell.doc_id, rcept_no=cell.receipt_no)
                or task.periods != [DateRange(start=cell.period_start, end=cell.period_end)]
                or root.root_id != f"output-{index}"
                or root.field_id != field_id
                or root.plan_task_id != task.task_id):
            raise ValueError("narrative matrix task/root binding이 다릅니다")
    supports = [support for item in contract.items for support in item.support_requirements]
    if len(supports) != len(units) or any(
            support.kind != "citation" or len(support.answer_root_refs) != 1
            for support in supports):
        raise ValueError("narrative matrix citation support binding이 다릅니다")


def _validate_slice_cross_authority(
        intent: SemanticIntent,
        intent_item: Any,
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    """Close plan/contract authorities to bounded intent and resolution."""
    if isinstance(resolution_item, tuple):
        raise ValueError("resolution item authority는 ResolvedItem이어야 합니다")
    slice_kinds = tuple(row.resolution.kind for row in resolution.items)
    if (len(slice_kinds) >= 2
            and set(slice_kinds).issubset(
                {"financial", "financial_comparison"})
            and "financial_comparison" in slice_kinds
            and intent.presentation == "table"
            and _is_recent_periods_fanout_intent_items(intent_item)
            and _is_recent_periods_fanout_resolution(resolution)):
        _validate_recent_periods_fanout_cross_authority(
            intent, intent_item, resolution, plan, contract)
        return
    if slice_kinds == (
            "financial", "financial_comparison"):
        _validate_financial_retrieve_comparison_cross_authority(
            intent, intent_item, resolution, plan, contract)
        return
    if tuple(row.resolution.kind for row in resolution.items) == (
            "financial", "financial"):
        if not isinstance(intent_item, tuple) or len(intent_item) != 2:
            raise ValueError(
                "parallel financial retrieval intent authority가 다릅니다")
        expected_plan, _supports = (
            _build_parallel_financial_retrieval_typed_plan(
                resolution, intent_item))
        if plan.model_dump(mode="json") != expected_plan.model_dump(mode="json"):
            raise ValueError(
                "parallel financial retrieval execution plan이 resolution과 다릅니다")
        if ([item.item_id for item in contract.items]
                != [item.item_id for item in intent_item]
                or contract.groups or contract.premise_contracts
                or [root.item_id for root in plan.answer_roots]
                != [item.item_id for item in intent_item]):
            raise ValueError(
                "parallel financial retrieval answer contract가 intent와 다릅니다")
        return
    if tuple(row.resolution.kind for row in resolution.items) == (
            "financial_comparison", "financial_comparison"):
        _validate_parallel_annual_change_cross_authority(
            intent, intent_item, resolution, plan, contract)
        return
    if len(slice_kinds) > 2 and set(slice_kinds) == {"financial"}:
        _validate_summary_metric_fanout_cross_authority(
            intent_item, resolution, plan, contract)
        return
    if isinstance(resolution_item.resolution, CorrectionLineageResolution):
        _validate_correction_lineage_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if isinstance(resolution_item.resolution, DocumentCollectionResolution):
        _validate_document_collection_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if isinstance(resolution_item.resolution, HoldingDisclosureResolution):
        _validate_holding_disclosure_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if isinstance(resolution_item.resolution, DocumentVersionHistoryResolution):
        _validate_document_version_history_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if isinstance(resolution_item.resolution, SelectedEventResolution):
        _validate_selected_event_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if isinstance(resolution_item.resolution, EventCollectionResolution):
        _validate_event_collection_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if isinstance(resolution_item.resolution, EventAmountChangeResolution):
        _validate_event_amount_change_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if isinstance(resolution_item.resolution, LifecycleCompositeResolution):
        if not isinstance(intent_item, tuple):
            raise ValueError("lifecycle composite intent authority가 다릅니다")
        expected_plan, _supports, _bindings = _build_lifecycle_composite_typed_plan(
            resolution, intent_item)
        if plan.model_dump(mode="json") != expected_plan.model_dump(mode="json"):
            raise ValueError("lifecycle composite execution plan이 typed authority와 다릅니다")
        return
    if isinstance(resolution_item.resolution, NarrativeMatrixResolution):
        _validate_narrative_matrix_cross_authority(
            intent_item, resolution_item, plan, contract)
        return
    if isinstance(resolution_item.resolution, TerminationReportedStatusResolution):
        _validate_termination_reported_status_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if isinstance(resolution_item.resolution, DocumentFactComparisonResolution):
        _validate_document_fact_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if isinstance(
            resolution_item.resolution, DocumentAttributeEvidenceResolution):
        _validate_document_attribute_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if isinstance(
            resolution_item.resolution,
            PeriodicNarrativeComparisonResolution):
        _validate_periodic_narrative_comparison_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if isinstance(resolution_item.resolution, SameDayDocumentCandidatesResolution):
        _validate_same_day_status_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if isinstance(resolution_item.resolution,
                  PeriodicDocumentNarrativeResolution):
        _validate_periodic_document_narrative_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if isinstance(resolution_item.resolution, FinancialComparisonResolution):
        _validate_financial_comparison_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if (
            isinstance(intent_item, tuple)
            and len(intent_item) == 2
            and isinstance(intent_item[1], tuple)
            and isinstance(resolution_item.resolution, FinancialResolution)
    ):
        _validate_premise_financial_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if (
            plan.canonical_build_id != resolution.canonical_build_id
            or plan.resolver_version != resolution.resolver_version
            or plan.applied_defaults != resolution_item.applied_defaults):
        raise ValueError(
            "execution plan build/resolver/default authority가 resolution과 다릅니다")

    coordinate = resolution_item.resolution
    resolved_plan = plan.resolved_plan
    if (
            resolved_plan.reference_date != resolution.reference_date
            or resolved_plan.corpus_cutoff != resolution.corpus_cutoff
            or resolved_plan.revision != 0
            or resolved_plan.derivations
            or resolved_plan.premise_claims
            or resolved_plan.applied_defaults
            or resolved_plan.presentation is not None
            or len(resolved_plan.tasks) != 1):
        raise ValueError("execution plan shape가 bounded resolution과 다릅니다")
    task = resolved_plan.tasks[0]
    if (
            not isinstance(task, ResolvedFinancialTask)
            or task.task_id != "task-1"
            or task.as_of != coordinate.as_of
            or task.view != coordinate.view
            or len(task.facts) != 1):
        raise ValueError("execution plan task binding이 resolution과 다릅니다")
    fact = task.facts[0]
    if (
            fact.output_id != "output-1"
            or fact.corp_code != coordinate.corp_code
            or fact.corp_name != coordinate.corp_name
            or fact.concept != coordinate.concept
            or fact.period_start != coordinate.period_start
            or fact.period_end != coordinate.period_end
            or fact.period_type != coordinate.period_type
            or fact.cumulative != coordinate.cumulative
            or fact.scope != coordinate.scope
            or fact.statement != coordinate.statement
            or fact.account_path is not None
            or fact.unit is not None):
        raise ValueError("execution plan fact coordinate가 resolution과 다릅니다")

    if (
            len(contract.items) != 1
            or contract.items[0].item_id != intent_item.item_id
            or contract.groups
            or contract.premise_contracts
            or contract.presentation != intent_item.output.presentation):
        raise ValueError("answer contract item/order authority가 intent와 다릅니다")
    contract_item = contract.items[0]
    expected_field_ids = [
        f"field-{proof.source_field_index + 1}"
        for proof in resolution_item.field_proofs
    ]
    fields = contract_item.projection.fields
    if (
            contract.completion != "complete"
            or contract_item.status != "ready"
            or contract_item.projection.shape != intent_item.output.shape
            or contract_item.projection.presentation != intent_item.output.presentation
            or contract_item.projection.sort is not None
            or [field.field_id for field in fields] != expected_field_ids
            or [field.field_key for field in fields] != ["value"]
            or contract_item.coverage.required != expected_field_ids
            or contract_item.coverage.executable != expected_field_ids
            or contract_item.coverage.limited
            or contract_item.limitation_bindings):
        raise ValueError("answer contract field shape가 intent/resolution과 다릅니다")
    if len(fields) != len(resolution_item.field_proofs):
        raise ValueError("answer contract field proof count가 resolution과 다릅니다")

    expected_answer_roots = []
    for field, proof in zip(fields, resolution_item.field_proofs, strict=True):
        if (
                field.required is not True
                or field.binding_status != "executable"
                or field.limitation_refs
                or field.answer_root_refs != ["output-1"]):
            raise ValueError("answer contract field root가 resolution proof와 다릅니다")
        expected_answer_roots.append((
            "output-1", resolution_item.item_id, field.field_id,
            fact.output_id, None, proof.proof_ref,
        ))
    actual_answer_roots = [
        (root.root_id, root.item_id, root.field_id, root.plan_output_id,
         root.plan_task_id, root.proof_ref)
        for root in plan.answer_roots
    ]
    if actual_answer_roots != expected_answer_roots:
        raise ValueError("execution plan answer roots가 contract/resolution과 다릅니다")

    expected_support_specs = [
        ("support-1", "unit", "source-reported unit"),
        ("support-2", "scope", f"scope={coordinate.scope}"),
        ("support-3", "period", _financial_period_support_detail(coordinate)),
        ("support-4", "evidence",
         f"proof_ref={resolution_item.field_proofs[0].proof_ref}"),
    ]
    actual_contract_supports = [
        (support.support_id, support.kind, support.required, support.detail,
         support.applies_to_field_ids, support.answer_root_refs)
        for support in contract_item.support_requirements
    ]
    expected_contract_supports = [
        (support_id, kind, True, detail, expected_field_ids, ["output-1"])
        for support_id, kind, detail in expected_support_specs
    ]
    if actual_contract_supports != expected_contract_supports:
        raise ValueError("answer contract support inventory가 resolution과 다릅니다")

    expected_support_roots = [
        (support_id, kind, resolution_item.item_id, expected_field_ids[0],
         "output-1")
        for support_id, kind, _ in expected_support_specs
    ]
    actual_support_roots = [
        (support.support_id, support.kind, support.item_id, support.field_id,
         support.root_id)
        for support in plan.support_roots
    ]
    if actual_support_roots != expected_support_roots:
        raise ValueError("execution plan support roots가 contract support와 다릅니다")


def _validate_premise_financial_cross_authority(
        intent_authority: tuple[Any, tuple[Any, ...]],
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    """Close premise verification to the one live financial source value."""
    intent_item, premises = intent_authority
    coordinate = resolution_item.resolution
    if not isinstance(coordinate, FinancialResolution):
        raise ValueError("premise financial resolution kind가 다릅니다")
    # Named-field financial intents prove the premise through their one field
    # proof; whole-target variants use a canonical coordinate instead.  This
    # handler accepts only the named-field shape, so the resolver proof must
    # be exactly the source field proof that also backs the answer root.
    expected_proof_ref = resolution_item.field_proofs[0].proof_ref
    if (
            [row.premise_id for row in resolution.premise_proofs]
            != [premise.premise_id for premise in premises]
            or any(row.proof_refs != [expected_proof_ref]
                   for row in resolution.premise_proofs)
    ):
        raise ValueError("premise financial resolution proof가 canonical fact와 다릅니다")
    if (
            plan.canonical_build_id != resolution.canonical_build_id
            or plan.resolver_version != resolution.resolver_version
            or plan.applied_defaults
            or plan.resolved_plan.reference_date != resolution.reference_date
            or plan.resolved_plan.corpus_cutoff != resolution.corpus_cutoff
            or plan.resolved_plan.revision != 0
            or plan.resolved_plan.derivations
            or plan.resolved_plan.applied_defaults
            or plan.resolved_plan.presentation is not None
            or len(plan.resolved_plan.tasks) != 1
            or len(plan.answer_roots) != 1
            or len(plan.plan_value_roots) != 1
    ):
        raise ValueError("premise financial execution plan metadata/shape가 다릅니다")
    task = plan.resolved_plan.tasks[0]
    if not isinstance(task, ResolvedFinancialTask) or (
            task.task_id != "task-1" or task.as_of != coordinate.as_of
            or task.view != coordinate.view or len(task.facts) != 1
    ):
        raise ValueError("premise financial task identity가 다릅니다")
    fact = task.facts[0]
    if (
            fact.output_id != "output-1" or fact.corp_code != coordinate.corp_code
            or fact.corp_name != coordinate.corp_name or fact.concept != coordinate.concept
            or fact.period_start != coordinate.period_start
            or fact.period_end != coordinate.period_end
            or fact.period_type != coordinate.period_type
            or fact.cumulative != coordinate.cumulative or fact.scope != coordinate.scope
            or fact.statement != coordinate.statement
    ):
        raise ValueError("premise financial fact coordinate가 resolution과 다릅니다")
    if [row.model_dump(mode="json") for row in plan.resolved_plan.premise_claims] != [
            PremiseClaim(
                claim_id=premise.premise_id, kind=premise.kind,
                raw_text=premise.raw_text,
                verify_with=[OutputRef(output_id="output-1")],
                verify_tasks=[TaskVerificationRef(task_id="task-1")],
            ).model_dump(mode="json")
            for premise in premises
    ]:
        raise ValueError("premise financial claim verification binding이 다릅니다")
    if plan.premise_roots != [
            ExecutionPremiseRoot(
                premise_id=premise.premise_id, root_ids=["output-1"],
                task_refs=["task-1"], plan_root_ids=["plan-root-1"],
            ) for premise in premises
    ]:
        raise ValueError("premise financial execution root binding이 다릅니다")
    if plan.plan_value_roots != [ExecutionPlanValueRoot(
            plan_root_id="plan-root-1", plan_output_id="output-1",
            value_kind="money", producer_task_id="task-1",
            answer_mirror_root_id="output-1",
    )]:
        raise ValueError("premise financial plan value root가 다릅니다")
    if (
            len(contract.items) != 1 or contract.groups
            or contract.presentation != intent_item.output.presentation
            or contract.completion != "complete"
            or contract.items[0].item_id != intent_item.item_id
            or contract.items[0].status != "ready"
            or contract.items[0].projection.shape != intent_item.output.shape
            or contract.items[0].projection.fields != [ProjectionField(
                field_id="field-1", field_key="value", binding_status="executable",
                answer_root_refs=["output-1"],
            )]
            or contract.items[0].coverage != CoveragePartition(
                required=["field-1"], executable=["field-1"], limited=[])
    ):
        raise ValueError("premise financial answer contract item shape가 다릅니다")
    expected_contracts = [CompiledPremiseContract(
        premise_id=premise.premise_id, kind=premise.kind,
        raw_text=premise.raw_text,
        applies_to_item_ids=list(premise.applies_to_item_ids),
        verification_root_refs=["output-1"], verification_task_refs=["task-1"],
        verification_plan_root_refs=["plan-root-1"], verdict_requirement="required",
    ) for premise in premises]
    if contract.premise_contracts != expected_contracts:
        raise ValueError("premise financial contract verification binding이 다릅니다")


def _financial_comparison_support_specs(
        comparison: FinancialComparisonResolution,
        ) -> list[tuple[str, str, str]]:
    """두 operand 비교의 근거 목록. 값은 전부 operand 에서 나온다."""
    operand_ids = ",".join(operand.operand_id for operand in comparison.operands)
    proof_refs = ",".join(operand.proof_ref for operand in comparison.operands)
    first = comparison.operands[0]
    common = f"operands={operand_ids}"
    return [
        ("unit", f"{common}; source-reported unit"),
        ("scope", f"{common}; scope={first.scope}"),
        ("period", f"{common}; {_financial_period_support_detail(first)}"),
        ("evidence", f"{common}; proof_refs={proof_refs}"),
    ]


def _periodic_document_support_specs(
        typed: PeriodicDocumentNarrativeResolution,
        resolution_item: ResolvedItem,
        ) -> list[tuple[str, str, str, str, str]]:
    """Return one exact evidence support for each live narrative root."""
    proof_by_index = {
        proof.source_field_index: proof for proof in resolution_item.field_proofs
    }
    return [
        (
            f"support-{index + 1}",
            f"field-{index + 1}",
            f"output-{index + 1}",
            "evidence",
            (
                f"document_proof={typed.document_proof.proof_ref}; "
                f"narrative_proof={typed.narrative_proof.proof_ref}; "
                f"field_proof={proof_by_index[index].proof_ref}"
            ),
        )
        for index in typed.executable_field_indexes
    ]


def _validate_same_day_status_cross_authority(
        intent_items: Any,
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    if not isinstance(intent_items, tuple) or len(intent_items) != 2:
        raise ValueError("same-day status intent item pair가 필요합니다")
    document_item, event_item = intent_items
    if len(resolution.items) != 2:
        raise ValueError("G-I-004 resolution item inventory가 다릅니다")
    status_item = resolution.items[1]
    document_resolution = resolution_item.resolution
    status_resolution = status_item.resolution
    if not isinstance(document_resolution, SameDayDocumentCandidatesResolution) \
            or not isinstance(status_resolution, TerminationReportedStatusResolution):
        raise ValueError("G-I-004 resolution tagged branches가 다릅니다")
    if (
            plan.canonical_build_id != resolution.canonical_build_id
            or plan.resolver_version != resolution.resolver_version
            or plan.applied_defaults
            or len(plan.resolved_plan.tasks) != 3
            or plan.resolved_plan.reference_date != resolution.reference_date
            or plan.resolved_plan.corpus_cutoff != resolution.corpus_cutoff
            or plan.resolved_plan.revision != 0
            or plan.resolved_plan.derivations
            or plan.resolved_plan.premise_claims
            or plan.resolved_plan.applied_defaults
            or plan.resolved_plan.presentation is not None
    ):
        raise ValueError("G-I-004 execution plan metadata/shape가 다릅니다")
    task1, task2, task3 = plan.resolved_plan.tasks
    expected_documents = [
        (task, f"task-{index + 1}", candidate.rcept_no)
        for index, (task, candidate) in enumerate(zip(
            (task1, task2), document_resolution.candidates, strict=True))
    ]
    for task, task_id, receipt in expected_documents:
        if not isinstance(task, ResolvedDocumentTask) or (
                task.task_id, task.operation, task.corp_code, task.corp_name,
                task.as_of, task.event_selector,
        ) != (
                task_id, "find", document_resolution.issuer_corp_code,
                document_resolution.issuer_corp_name,
                document_resolution.as_of, None,
        ):
            raise ValueError("G-I-004 document task identity가 다릅니다")
        selector = task.selector
        if (
                selector.rcept_no != receipt
                or selector.doc_id is not None
                or selector.doc_group is not None
                or selector.event_type is not None
                or selector.form is not None
                or selector.report_name_contains is not None
                or selector.rcept_from is not None
                or selector.rcept_to is not None
                or selector.is_correction is not None
        ):
            raise ValueError("G-I-004 document task selector는 exact receipt만 허용합니다")
    if not isinstance(task3, ResolvedEventTask) or (
            task3.task_id, task3.operation, task3.corp_code, task3.corp_name,
            task3.timepoints, task3.requested_slots, task3.output_id,
            task3.field_outputs,
    ) != (
            "task-3", "status", document_resolution.issuer_corp_code,
            document_resolution.issuer_corp_name, [resolution.corpus_cutoff],
            list(event_item.output.field_surfaces), None, [],
    ):
        raise ValueError("G-I-004 event task identity가 다릅니다")
    event_selector = task3.selector
    if (
            event_selector.event_key != status_resolution.event_key
            or event_selector.seed_rcept_no is not None
            or event_selector.event_type is not None
            or event_selector.counterparty is not None
            or event_selector.contract_name is not None
            or event_selector.keywords
            or event_selector.event_from is not None
            or event_selector.event_to is not None
    ):
        raise ValueError("G-I-004 event selector는 exact qualified event_key만 허용합니다")
    expected_roots = [
        (
            f"output-{index + 1}", document_item.item_id, "field-1", None,
            f"task-{index + 1}", resolution.items[0].field_proofs[0].proof_ref,
        )
        for index in range(len(document_resolution.candidates))
    ] + [(
        "output-3", event_item.item_id, "field-2", None, "task-3",
        status_item.field_proofs[0].proof_ref,
    )]
    actual_roots = [
        (root.root_id, root.item_id, root.field_id, root.plan_output_id,
         root.plan_task_id, root.proof_ref)
        for root in plan.answer_roots
    ]
    if actual_roots != expected_roots:
        raise ValueError("G-I-004 answer roots가 task/field/proof authority와 다릅니다")
    expected_supports = _same_day_status_support_specs(
        document_resolution, status_resolution)
    actual_supports = [
        (support.support_id, support.kind, support.required, support.detail,
         support.applies_to_field_ids, support.answer_root_refs)
        for item in contract.items for support in item.support_requirements
    ]
    expected_contract_supports = [
        (support_id, "evidence", True, detail, [field_id], [root_id])
        for support_id, _, field_id, root_id, detail in expected_supports
    ]
    if actual_supports != expected_contract_supports:
        raise ValueError("G-I-004 evidence support inventory가 다릅니다")
    actual_support_roots = [
        (support.support_id, support.kind, support.item_id, support.field_id,
         support.root_id)
        for support in plan.support_roots
    ]
    expected_support_roots = [
        (support_id, "evidence", item_id, field_id, root_id)
        for support_id, item_id, field_id, root_id, _ in expected_supports
    ]
    if actual_support_roots != expected_support_roots:
        raise ValueError("G-I-004 support roots가 evidence support와 다릅니다")
    if (
            len(contract.items) != 2
            or contract.groups != [CompiledAnswerGroup(
                group_id="group-1",
                item_ids=[document_item.item_id, event_item.item_id])]
            or contract.premise_contracts
            or contract.presentation != document_item.output.presentation
            or contract.completion != "partial"
    ):
        raise ValueError("G-I-004 answer contract item/group shape가 다릅니다")
    item1, item2 = contract.items
    if (
            item1.item_id != document_item.item_id or item1.status != "partial"
            or item1.projection.shape != document_item.output.shape
            or item1.projection.presentation != document_item.output.presentation
            or item1.projection.sort is not None
    ):
        raise ValueError("G-I-004 item-1 contract shape가 다릅니다")
    if (
            item2.item_id != event_item.item_id or item2.status != "partial"
            or item2.projection.shape != event_item.output.shape
            or item2.projection.presentation != event_item.output.presentation
            or item2.projection.sort is not None
    ):
        raise ValueError("G-I-004 item-2 contract shape가 다릅니다")
    fields = [item1.projection.fields, item2.projection.fields]
    expected_field_specs = [
        (fields[0][0], "field-1", "contents", ["output-1", "output-2"],
         ["limitation-1"]),
        (fields[1][0], "field-2", "final_status", ["output-3"],
         ["limitation-2"]),
    ]
    for field, field_id, key, roots, limitations in expected_field_specs:
        if (
                field.field_id != field_id
                or field.field_key != key
                or field.required is not True
                or field.binding_status != "qualified"
                or field.answer_root_refs != roots
                or field.limitation_refs != limitations
        ):
            raise ValueError("G-I-004 qualified projection field binding이 다릅니다")
    if item1.coverage != CoveragePartition(
            required=["field-1"], executable=["field-1"], limited=["field-1"]):
        raise ValueError("G-I-004 item-1 coverage가 다릅니다")
    if item2.coverage != CoveragePartition(
            required=["field-2"], executable=["field-2"], limited=["field-2"]):
        raise ValueError("G-I-004 item-2 coverage가 다릅니다")
    actual_limitations = [
        (item, binding.limitation_id, binding.code, binding.family,
         binding.applies_to_field_ids, binding.detail)
        for item in contract.items for binding in item.limitation_bindings
    ]
    expected_limitations = [
        (item1, "limitation-1", "intraday_order_unavailable", "ordering",
         ["field-1"], document_resolution.ordering_provenance.detail),
        (item2, "limitation-2", "ambiguous_event_origin", "identity_lineage",
         ["field-2"], status_resolution.identity_provenance.detail),
    ]
    if actual_limitations != expected_limitations:
        raise ValueError("G-I-004 limitation binding inventory가 다릅니다")


def _validate_periodic_document_narrative_cross_authority(
        intent_item: Any,
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    """Close periodic document/narrative tasks to typed source authority."""
    typed = resolution_item.resolution
    if not isinstance(typed, PeriodicDocumentNarrativeResolution):
        raise ValueError("periodic document resolution tagged branch가 다릅니다")
    if (
            plan.canonical_build_id != resolution.canonical_build_id
            or plan.resolver_version != resolution.resolver_version
            or plan.applied_defaults != resolution_item.applied_defaults
            or len(plan.resolved_plan.tasks) != 2
            or plan.resolved_plan.reference_date != resolution.reference_date
            or plan.resolved_plan.corpus_cutoff != resolution.corpus_cutoff
            or plan.resolved_plan.revision != 0
            or plan.resolved_plan.derivations
            or plan.resolved_plan.premise_claims
            or plan.resolved_plan.applied_defaults != [
                f"as_of=corpus_cutoff({row.value})"
                for row in resolution_item.applied_defaults
                if row.policy == "as_of" and row.basis == "corpus_cutoff"]
            or plan.resolved_plan.presentation is not None
    ):
        raise ValueError("periodic document execution plan metadata/shape가 다릅니다")

    task1, task2 = plan.resolved_plan.tasks
    expected_selector = DocumentSelector(
        doc_id=typed.document_id, rcept_no=typed.receipt_no)
    if not isinstance(task1, ResolvedDocumentTask) or (
            task1.task_id, task1.operation, task1.corp_code, task1.corp_name,
            task1.as_of, task1.event_selector, task1.selector,
    ) != (
            "task-1", "find", typed.corp_code, typed.corp_name,
            resolution.corpus_cutoff, None, expected_selector,
    ):
        raise ValueError(
            "G-A-010 document task는 exact document/receipt selector여야 합니다")
    if not isinstance(task2, ResolvedNarrativeTask) or (
            task2.task_id, task2.operation, task2.corp_codes, task2.corp_names,
            task2.as_of, task2.retrieval_query, task2.document_selector,
            task2.periods, task2.requested_slots,
    ) != (
            "task-2", "search", [typed.corp_code], [typed.corp_name],
            resolution.corpus_cutoff,
            typed.source_retrieval_query or intent_item.target.surface,
            expected_selector, [], (
                list(typed.canonical_requested_slots)
                if typed.canonical_requested_slots else [
                    "".join(surface.split())
                    for surface in intent_item.output.field_surfaces
                ]),
    ):
        raise ValueError(
            "G-A-010 narrative task는 exact document selector/field surfaces여야 합니다")

    expected_roots = [
        (
            f"output-{index + 1}", intent_item.item_id,
            f"field-{index + 1}", None, "task-2",
            resolution_item.field_proofs[index].proof_ref,
        )
        for index in typed.executable_field_indexes
    ]
    actual_roots = [
        (root.root_id, root.item_id, root.field_id, root.plan_output_id,
         root.plan_task_id, root.proof_ref)
        for root in plan.answer_roots
    ]
    if actual_roots != expected_roots:
        raise ValueError(
            "G-A-010 answer roots는 executable narrative fields에만 있어야 합니다")
    limited_field_ids = {
        f"field-{index + 1}" for index in typed.limited_field_indexes
    }
    if any(root.field_id in limited_field_ids for root in plan.answer_roots):
        raise ValueError("limited periodic field에는 answer root가 없어야 합니다")

    if (
            len(contract.items) != 1
            or contract.items[0].item_id != intent_item.item_id
            or contract.groups
            or contract.premise_contracts
            or contract.presentation != intent_item.output.presentation
            or contract.completion != (
                "partial" if typed.limited_field_indexes else "complete")
    ):
        raise ValueError("G-A-010 answer contract item/coverage shape가 다릅니다")
    contract_item = contract.items[0]
    fields = contract_item.projection.fields
    expected_fields = []
    for index in range(len(resolution_item.field_proofs)):
        executable = index in typed.executable_field_indexes
        expected_fields.append((
            f"field-{index + 1}", f"record_{index + 1}",
            "executable" if executable else "limited",
            [f"output-{index + 1}"] if executable else [],
            [] if executable or typed.source_cross_check_provenance is None
            else ["limitation-1"],
        ))
    actual_fields = [
        (field.field_id, field.field_key, field.binding_status,
         field.answer_root_refs, field.limitation_refs)
        for field in fields
    ]
    if actual_fields != expected_fields:
        raise ValueError("G-A-010 ordered projection field binding이 다릅니다")
    if (
            contract_item.status != (
                "partial" if typed.limited_field_indexes else "ready")
            or contract_item.projection.shape != intent_item.output.shape
            or contract_item.projection.presentation != intent_item.output.presentation
            or contract_item.projection.sort is not None
            or contract_item.coverage != CoveragePartition(
                required=[
                    f"field-{index + 1}"
                    for index in range(len(resolution_item.field_proofs))
                ],
                executable=[
                    f"field-{index + 1}"
                    for index in typed.executable_field_indexes
                ],
                limited=[
                    f"field-{index + 1}"
                    for index in typed.limited_field_indexes
                ],
            )
    ):
        raise ValueError("G-A-010 projection coverage/status가 다릅니다")

    expected_support_specs = _periodic_document_support_specs(
        typed, resolution_item)
    actual_supports = [
        (support.support_id, support.kind, support.required, support.detail,
         support.applies_to_field_ids, support.answer_root_refs)
        for support in contract_item.support_requirements
    ]
    expected_contract_supports = [
        (support_id, kind, True, detail, [field_id], [root_id])
        for support_id, field_id, root_id, kind, detail in expected_support_specs
    ]
    if actual_supports != expected_contract_supports:
        raise ValueError("G-A-010 evidence support inventory가 다릅니다")
    actual_support_roots = [
        (support.support_id, support.kind, support.item_id, support.field_id,
         support.root_id)
        for support in plan.support_roots
    ]
    expected_support_roots = [
        (support_id, kind, intent_item.item_id, field_id, root_id)
        for support_id, field_id, root_id, kind, _ in expected_support_specs
    ]
    if actual_support_roots != expected_support_roots:
        raise ValueError("G-A-010 support roots는 live answer roots와 1:1이어야 합니다")

    limitation = typed.source_cross_check_provenance
    expected_limitations = []
    if limitation is not None:
        expected_limitations = [(
            "limitation-1", "source_cross_check_partial", "source_scope",
            [f"field-{index + 1}" for index in typed.limited_field_indexes],
            limitation.detail, None,
        )]
    actual_limitations = [
        (binding.limitation_id, binding.code, binding.family,
         binding.applies_to_field_ids, binding.detail,
         binding.required_followup)
        for binding in contract_item.limitation_bindings
    ]
    if actual_limitations != expected_limitations:
        raise ValueError("G-A-010 limitation binding inventory가 다릅니다")


def _validate_financial_retrieve_comparison_cross_authority(
        intent: SemanticIntent,
        intent_items: Any,
        resolution: AuthoritativeResolution,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    if not isinstance(intent_items, tuple) or len(intent_items) != 2:
        raise ValueError("financial retrieve/comparison intent pair가 필요합니다")
    expected_plan, retrieve_supports, comparison_supports = (
        _build_financial_retrieve_comparison_typed_plan(
            resolution, intent_items))
    if plan.model_dump(mode="json") != expected_plan.model_dump(mode="json"):
        raise ValueError(
            "financial retrieve/comparison execution plan이 typed authority와 다릅니다")
    expected_contract = _build_financial_retrieve_comparison_contract(
        intent,
        intent_items,
        expected_plan,
        retrieve_supports,
        comparison_supports,
    )
    if contract.model_dump(mode="json") != expected_contract.model_dump(mode="json"):
        raise ValueError(
            "financial retrieve/comparison answer contract가 source authority와 다릅니다")


def _validate_summary_metric_fanout_cross_authority(
        intent_items: Any,
        resolution: AuthoritativeResolution,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    """Close an N-item financial fanout to its own plan/contract shape.

    **일반 꼬리를 쓰지 않는 이유.** `_validate_slice_cross_authority` 의 일반
    꼬리는 단일 항목 모양으로 못박혀 있다 — `len(task.facts) != 1`,
    `plan.applied_defaults != resolution_item.applied_defaults`, 답 뿌리
    하나, support 4개. fanout 을 끼우려고 그것을 넓히면 **단일 항목 경로가
    원래 지키던 것까지 헐거워진다.** 다른 타입들이 이미 그렇게 하듯 규격마다
    검증기를 따로 둔다.

    계획은 통째로 다시 세워 대조한다(`_build_parallel_financial_retrieval_typed_plan`).
    그 한 번의 비교가 task·fact 좌표·답 뿌리·support 뿌리를 전부 덮으므로,
    여기서는 계약 쪽 — 항목 순서와 필드 결속 — 만 따로 본다.
    """

    if not isinstance(intent_items, tuple) or len(intent_items) < 3:
        raise DeterministicPlanCompilerError(
            "summary metric fanout intent authority가 다릅니다")
    if len(resolution.items) != len(intent_items):
        raise DeterministicPlanCompilerError(
            "summary metric fanout resolution 항목 수가 intent와 다릅니다")
    expected_plan, expected_supports = (
        _build_parallel_financial_retrieval_typed_plan(
            resolution, intent_items))
    if plan.model_dump(mode="json") != expected_plan.model_dump(mode="json"):
        raise ValueError(
            "summary metric fanout execution plan이 resolution과 다릅니다")

    item_ids = [item.item_id for item in intent_items]
    if ([item.item_id for item in contract.items] != item_ids
            or contract.groups or contract.premise_contracts
            or contract.completion != "complete"
            or [root.item_id for root in plan.answer_roots] != item_ids):
        raise ValueError(
            "summary metric fanout answer contract가 intent와 다릅니다")

    for index, (intent_item, contract_item, supports) in enumerate(
            zip(intent_items, contract.items, expected_supports, strict=True),
            start=1):
        field_id = f"field-{index}"
        root_id = f"output-{index}"
        fields = contract_item.projection.fields
        if (
                contract_item.status != "ready"
                or contract_item.projection.shape != intent_item.output.shape
                or contract_item.projection.presentation
                != intent_item.output.presentation
                or contract_item.projection.sort is not None
                or [field.field_id for field in fields] != [field_id]
                or [field.field_key for field in fields] != ["value"]
                or contract_item.coverage.required != [field_id]
                or contract_item.coverage.executable != [field_id]
                or contract_item.coverage.limited
                or contract_item.limitation_bindings):
            raise ValueError(
                "summary metric fanout field shape가 intent와 다릅니다")
        field = fields[0]
        if (field.required is not True
                or field.binding_status != "executable"
                or field.limitation_refs
                or field.answer_root_refs != [root_id]):
            raise ValueError(
                "summary metric fanout field root가 resolution proof와 다릅니다")
        if ([(row.support_id, row.kind, row.detail,
              row.applies_to_field_ids, row.answer_root_refs)
             for row in contract_item.support_requirements]
                != [(row.support_id, row.kind, row.detail,
                     row.applies_to_field_ids, row.answer_root_refs)
                    for row in supports]):
            raise ValueError(
                "summary metric fanout support inventory가 resolution과 다릅니다")


def _validate_recent_periods_fanout_cross_authority(
        intent: SemanticIntent, intent_items: Any,
        resolution: AuthoritativeResolution, plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    """Rebuild and compare the complete mixed recent-period boundary."""

    if not isinstance(intent_items, tuple) or not (2 <= len(intent_items) <= 5):
        raise ValueError("recent periods fanout intent authority가 다릅니다")
    expected_plan, supports = _build_recent_periods_fanout_typed_plan(
        resolution, intent_items)
    if plan.model_dump(mode="json") != expected_plan.model_dump(mode="json"):
        raise ValueError(
            "recent periods fanout execution plan이 authority와 다릅니다")
    expected_contract = _build_recent_periods_fanout_contract(
        intent, intent_items, expected_plan, supports)
    if contract.model_dump(mode="json") != expected_contract.model_dump(
            mode="json"):
        raise ValueError(
            "recent periods fanout contract가 authority와 다릅니다")


def _validate_parallel_annual_change_cross_authority(
        intent: SemanticIntent, intent_items: Any,
        resolution: AuthoritativeResolution, plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    if not isinstance(intent_items, tuple) or len(intent_items) != 2:
        raise ValueError("parallel annual change intent pair가 필요합니다")
    expected_plan, supports = _build_parallel_annual_change_typed_plan(
        resolution, intent_items)
    if plan.model_dump(mode="json") != expected_plan.model_dump(mode="json"):
        raise ValueError("parallel annual change execution plan이 authority와 다릅니다")
    expected_contract = _build_parallel_annual_change_contract(
        intent, intent_items, expected_plan, supports)
    if contract.model_dump(mode="json") != expected_contract.model_dump(mode="json"):
        raise ValueError("parallel annual change contract가 authority와 다릅니다")


def _validate_financial_comparison_cross_authority(
        intent_item: Any,
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    """Close every G-A-004 lowering node to the two source operands."""
    comparison = resolution_item.resolution
    assert isinstance(comparison, FinancialComparisonResolution)
    operands = comparison.operands
    if len(operands) > 2:
        expected_plan, expected_supports = (
            _build_nary_financial_ranking_typed_plan(
                resolution, resolution_item, intent_item))
        if plan.model_dump(mode="json") != expected_plan.model_dump(mode="json"):
            raise ValueError(
                "N-ary financial ranking execution plan이 typed authority와 다릅니다")
        field_indexes = _nary_ranking_field_indexes(intent_item)
        if field_indexes is None or len(contract.items) != 1 \
                or contract.groups or contract.premise_contracts:
            raise ValueError(
                "N-ary financial ranking answer contract topology가 다릅니다")
        rank_index, values_index = field_indexes
        fact_output_ids = [f"output-{index}"
                           for index in range(1, len(operands) + 1)]
        rank_output_id = f"output-{len(operands) + 1}"
        expected_fields = []
        for index in range(len(intent_item.output.field_surfaces)):
            refs = []
            keys = []
            if index == rank_index:
                refs.append(rank_output_id)
                keys.append("ranking")
            if index == values_index:
                refs.extend(fact_output_ids)
                keys.append("values")
            expected_fields.append(ProjectionField(
                field_id=f"field-{index + 1}",
                field_key="_and_".join(keys),
                binding_status="executable", answer_root_refs=refs,
            ))
        contract_item = contract.items[0]
        field_ids = [field.field_id for field in expected_fields]
        if (contract_item.item_id != intent_item.item_id
                or contract_item.status != "ready"
                or contract_item.projection.shape != intent_item.output.shape
                or contract_item.projection.fields != expected_fields
                or contract_item.coverage != CoveragePartition(
                    required=field_ids, executable=field_ids, limited=[])
                or contract_item.support_requirements != expected_supports
                or contract_item.limitation_bindings):
            raise ValueError(
                "N-ary financial ranking answer contract가 source/plan과 다릅니다")
        return
    # 이슈 #59 1단계 — 회사가 다른 concept_ratio는 승자/차이(G-A-004) 모양이
    # 아니라 same-company concept_ratio 와 같은 "A÷B 몇 배" 스칼라 투영을
    # 쓴다(위 `_compile_financial_comparison`/`_build_financial_comparison_
    # typed_plan` 이 이미 그 분기로 낮췄다). 그 lowering 을 감사하는 것도
    # 아래 G-A-004 전용 검사가 아니라 같은 time-derivation cross-authority다.
    # 이슈 #124 (합계) — 회사가 다른 sum도 같은 이유로 같은 자리에 둔다.
    if (operands[0].corp_code == operands[1].corp_code
            or comparison.requested_operators == ["concept_ratio"]
            or comparison.requested_operators == ["sum"]):
        _validate_financial_time_derivation_cross_authority(
            intent_item, resolution, resolution_item, plan, contract)
        return
    if (
            plan.canonical_build_id != resolution.canonical_build_id
            or plan.resolver_version != resolution.resolver_version
            or plan.applied_defaults != resolution_item.applied_defaults):
        raise ValueError(
            "G-A-004 execution plan build/resolver/default authority가 resolution과 다릅니다")

    # 이슈 #182 — 회사가 다른 두 operand의 (as_of, view)가 다르면(#149
    # 회계연도말 pin이 회사마다 독립적으로 자기 사업보고서 날짜를 낸다 —
    # 두 회사가 같은 날 사업보고서를 낼 이유가 없다) task를 회사당 하나씩
    # 나눈다. 한 task에 몰아넣으면 그 task의 as_of가 먼저 신고한 회사
    # 날짜로 고정돼, 더 늦게 신고한 회사의 사실이 그 시점엔 아직 존재하지
    # 않아 조회가 거절된다(EG2-023: LG이노텍 20260312 · SK하이닉스
    # 20260317 실측으로 직접 확인함). same-company 두-시점 관계
    # (`_validate_financial_time_derivation_cross_authority`)가 이미 하는
    # split과 같은 모양 — 다만 이 G-A-004 계약의 "winner"/"difference"
    # field_key는 그대로 유지한다.
    split_tasks = (
        (operands[0].as_of, operands[0].view)
        != (operands[1].as_of, operands[1].view)
    )
    expected_task_count = 2 if split_tasks else 1
    resolved_plan = plan.resolved_plan
    if (
            resolved_plan.reference_date != resolution.reference_date
            or resolved_plan.corpus_cutoff != resolution.corpus_cutoff
            or resolved_plan.revision != 0
            or resolved_plan.applied_defaults != [
                f"as_of=corpus_cutoff({row.value})"
                for row in resolution_item.applied_defaults
                if row.policy == "as_of" and row.basis == "corpus_cutoff"]
            or resolved_plan.presentation is not None
            or len(resolved_plan.tasks) != expected_task_count
            or len(resolved_plan.derivations) != len(
                comparison.requested_operators)):
        raise ValueError("G-A-004 execution plan shape가 resolution과 다릅니다")
    tasks = resolved_plan.tasks
    if split_tasks:
        if any(
                not isinstance(task, ResolvedFinancialTask)
                or task.task_id != f"task-{index}"
                or task.as_of != operand.as_of
                or task.view != operand.view
                or len(task.facts) != 1
                for index, (task, operand) in enumerate(
                    zip(tasks, operands, strict=True), start=1)):
            raise ValueError("G-A-004 execution plan task binding이 resolution과 다릅니다")
    else:
        task = tasks[0]
        if (
                not isinstance(task, ResolvedFinancialTask)
                or task.task_id != "task-1"
                or task.as_of != operands[0].as_of
                or task.view != operands[0].view
                or len(task.facts) != 2):
            raise ValueError("G-A-004 execution plan task binding이 resolution과 다릅니다")
    all_facts = [fact for task in tasks for fact in task.facts]
    expected_facts = []
    for output_id, operand in zip(("output-1", "output-2"), operands, strict=True):
        expected_facts.append((
            output_id, operand.corp_code, operand.corp_name, operand.concept,
            operand.period_start, operand.period_end, operand.period_type,
            operand.cumulative, operand.scope, operand.statement,
        ))
    actual_facts = [
        (fact.output_id, fact.corp_code, fact.corp_name, fact.concept,
         fact.period_start, fact.period_end, fact.period_type, fact.cumulative,
         fact.scope, fact.statement)
        for fact in all_facts
    ]
    if actual_facts != expected_facts or any(
            fact.account_path is not None or fact.unit is not None
            for fact in all_facts):
        raise ValueError("G-A-004 execution plan fact coordinates가 resolution과 다릅니다")

    expected_derivations = [
        (f"output-{index + 3}", operator, ("output-1", "output-2"))
        for index, operator in enumerate(comparison.requested_operators)
    ]
    actual_derivations = [
        (derivation.output_id, derivation.operator,
         tuple(ref.output_id for ref in derivation.operands))
        for derivation in resolved_plan.derivations
    ]
    if actual_derivations != expected_derivations:
        raise ValueError("G-A-004 derivation lowering이 승인값과 다릅니다")
    expected_claims = ([] if comparison.verification_claim is None else [
        PremiseClaim(
            claim_id=comparison.verification_premise_id, kind="comparison",
            raw_text=comparison.verification_claim,
            verify_with=[OutputRef(output_id="output-3")],
        ).model_dump(mode="json")
    ])
    if [row.model_dump(mode="json")
            for row in resolved_plan.premise_claims] != expected_claims:
        raise ValueError("financial verification claim lowering이 다릅니다")

    if (
            len(contract.items) != 1
            or contract.items[0].item_id != intent_item.item_id
            or contract.groups
            or contract.presentation != intent_item.output.presentation
            or contract.completion != "complete"):
        raise ValueError("G-A-004 answer contract item/order authority가 intent와 다릅니다")
    contract_item = contract.items[0]
    fields = contract_item.projection.fields
    expected_field_ids = [
        f"field-{index}" for index in range(
            1, len(comparison.requested_operators) + 1)]
    expected_field_keys = [
        "winner" if operator == "argmax" else "difference"
        for operator in comparison.requested_operators]
    if (
            contract_item.status != "ready"
            or contract_item.projection.shape != "comparison"
            or contract_item.projection.presentation != intent_item.output.presentation
            or contract_item.projection.sort is not None
            or [field.field_id for field in fields] != expected_field_ids
            or [field.field_key for field in fields] != expected_field_keys
            or contract_item.coverage.required != expected_field_ids
            or contract_item.coverage.executable != expected_field_ids
            or contract_item.coverage.limited
            or contract_item.limitation_bindings):
        raise ValueError("G-A-004 answer contract field shape가 intent/resolution과 다릅니다")

    expected_answer_roots = [
        (f"output-{index + 3}", intent_item.item_id, f"field-{index + 1}",
         f"output-{index + 3}", None,
         resolution_item.field_proofs[index].proof_ref)
        for index in range(len(comparison.requested_operators))
    ]
    for field, root, expected_field_id in zip(
            fields, expected_answer_roots, expected_field_ids, strict=True):
        if (
                field.required is not True
                or field.binding_status != "executable"
                or field.limitation_refs
                or field.answer_root_refs != [root[0]]
                or field.field_id != expected_field_id):
            raise ValueError("G-A-004 answer contract field root가 다릅니다")
    actual_answer_roots = [
        (root.root_id, root.item_id, root.field_id, root.plan_output_id,
         root.plan_task_id, root.proof_ref)
        for root in plan.answer_roots
    ]
    if actual_answer_roots != expected_answer_roots:
        raise ValueError(
            "G-A-004 execution plan answer roots가 derived outputs와 다릅니다")
    if {root.root_id for root in plan.answer_roots} & {"output-1", "output-2"}:
        raise ValueError("G-A-004 raw operand는 answer root가 될 수 없습니다")

    support_specs = _financial_comparison_support_specs(comparison)
    comparison_bindings = [
        (f"field-{index + 1}", f"output-{index + 3}",
         index * len(support_specs))
        for index in range(len(comparison.requested_operators))
    ]
    expected_contract_supports = [
        (f"support-{index}", kind, True, detail,
         [field_id], [root_id])
        for field_id, root_id, offset in comparison_bindings
        for index, (kind, detail) in enumerate(support_specs, start=1 + offset)
    ]
    actual_contract_supports = [
        (support.support_id, support.kind, support.required, support.detail,
         support.applies_to_field_ids, support.answer_root_refs)
        for support in contract_item.support_requirements
    ]
    if actual_contract_supports != expected_contract_supports:
        raise ValueError("G-A-004 answer contract two-operand support inventory가 다릅니다")
    expected_support_roots = [
        (f"support-{index}", kind, intent_item.item_id, field_id, root_id)
        for field_id, root_id, offset in comparison_bindings
        for index, (kind, _) in enumerate(support_specs, start=1 + offset)
    ]
    actual_support_roots = [
        (support.support_id, support.kind, support.item_id, support.field_id,
         support.root_id)
        for support in plan.support_roots
    ]
    if actual_support_roots != expected_support_roots:
        raise ValueError("G-A-004 execution plan support roots가 contract와 다릅니다")
    expected_premise_roots = ([] if comparison.verification_claim is None else [
        ExecutionPremiseRoot(
            premise_id=comparison.verification_premise_id,
            root_ids=["output-3"],
            task_refs=["task-1"],
        )
    ])
    if plan.premise_roots != expected_premise_roots:
        raise ValueError("financial verification premise root가 다릅니다")
    expected_contract_premises = ([] if comparison.verification_claim is None else [
        CompiledPremiseContract(
            premise_id=comparison.verification_premise_id,
            kind="comparison", raw_text=comparison.verification_claim,
            applies_to_item_ids=[intent_item.item_id],
            verification_root_refs=["output-3"],
            verification_task_refs=["task-1"],
            verification_plan_root_refs=[],
            verdict_requirement="required",
        )
    ])
    if contract.premise_contracts != expected_contract_premises:
        raise ValueError("financial verification answer premise contract가 다릅니다")


def _validate_financial_time_derivation_cross_authority(
        intent_item: Any,
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        plan: ExecutionPlan,
        contract: CompiledAnswerContract,
        ) -> None:
    """Close same-company two-period lowering without a question-ID overlay."""
    typed = resolution_item.resolution
    assert isinstance(typed, FinancialComparisonResolution)
    if _is_discrete_q4_time_comparison(typed):
        expected, _ = _build_financial_comparison_typed_plan(
            resolution, resolution_item, intent_item)
        if plan != expected:
            raise ValueError("Q4 comparison cumulative expansion differs from source authority")
        # Validate the unchanged logical answer contract with the ordinary
        # two-quarter plan after proving the expanded execution plan exactly.
        plan, _ = _build_financial_comparison_typed_plan(
            resolution, resolution_item, intent_item, _expand_discrete_q4=False)
    operands = typed.operands
    if (
            plan.canonical_build_id != resolution.canonical_build_id
            or plan.resolver_version != resolution.resolver_version
            or plan.applied_defaults != resolution_item.applied_defaults):
        raise ValueError("financial time derivation build/default authority가 다릅니다")
    resolved_plan = plan.resolved_plan
    expected_operators = list(typed.requested_operators)
    split_tasks = (
        (operands[0].as_of, operands[0].view)
        != (operands[1].as_of, operands[1].view)
    )
    expected_task_count = 2 if split_tasks else 1
    account_paths = _financial_view_account_paths(resolution_item, typed)
    if (
            resolved_plan.reference_date != resolution.reference_date
            or resolved_plan.corpus_cutoff != resolution.corpus_cutoff
            or resolved_plan.revision != 0
            or len(resolved_plan.tasks) != expected_task_count
            or len(resolved_plan.derivations) != len(expected_operators)
            or resolved_plan.premise_claims or resolved_plan.applied_defaults
            or resolved_plan.presentation is not None):
        raise ValueError("financial time derivation execution plan shape가 다릅니다")
    tasks = resolved_plan.tasks
    if split_tasks:
        if any(
                not isinstance(task, ResolvedFinancialTask)
                or task.task_id != f"task-{index}"
                or task.as_of != operand.as_of
                or task.view != operand.view
                or len(task.facts) != 1
                for index, (task, operand) in enumerate(
                    zip(tasks, operands, strict=True), start=1)):
            raise ValueError("financial time derivation split task binding이 다릅니다")
    else:
        task = tasks[0]
        if (
                not isinstance(task, ResolvedFinancialTask)
                or task.task_id != "task-1"
                or task.as_of != operands[0].as_of
                or task.view != operands[0].view
                or len(task.facts) != 2):
            raise ValueError("financial time derivation task binding이 다릅니다")
    expected_facts = [
        (f"output-{index}", operand.corp_code, operand.corp_name,
         operand.concept, operand.period_start, operand.period_end,
         operand.period_type, operand.cumulative, operand.scope,
         operand.statement, account_paths[index - 1])
        for index, operand in enumerate(operands, start=1)
    ]
    all_facts = [fact for task in tasks for fact in task.facts]
    actual_facts = [
        (fact.output_id, fact.corp_code, fact.corp_name, fact.concept,
         fact.period_start, fact.period_end, fact.period_type, fact.cumulative,
         fact.scope, fact.statement, fact.account_path)
        for fact in all_facts
    ]
    if actual_facts != expected_facts or any(
            fact.unit is not None for fact in all_facts):
        raise ValueError("financial time derivation facts가 resolution과 다릅니다")
    operand_output_ids = (
        ("output-2", "output-1")
        if operands[0].view != operands[1].view
        else ("output-1", "output-2")
    )
    expected_derivations = [
        (f"output-{index + 3}", operator, operand_output_ids)
        for index, operator in enumerate(expected_operators)
    ]
    actual_derivations = [
        (row.output_id, row.operator,
         tuple(ref.output_id for ref in row.operands))
        for row in resolved_plan.derivations
    ]
    if actual_derivations != expected_derivations:
        raise ValueError("financial time derivation operator lowering이 다릅니다")

    view_relation = operands[0].view != operands[1].view
    root_ids = (
        ["output-1", "output-2", "output-3"] if view_relation else
        [row[0] for row in expected_derivations]
    )
    combined_field = not view_relation and len(
        intent_item.output.field_surfaces) == 1
    expected_roots = [
        (root_id, intent_item.item_id,
         "field-1" if combined_field else f"field-{index + 1}",
         root_id, None,
         resolution_item.field_proofs[
             0 if combined_field else index].proof_ref)
        for index, root_id in enumerate(root_ids)
    ]
    actual_roots = [
        (row.root_id, row.item_id, row.field_id, row.plan_output_id,
         row.plan_task_id, row.proof_ref)
        for row in plan.answer_roots
    ]
    if actual_roots != expected_roots:
        raise ValueError("financial time derivation answer roots가 다릅니다")
    if (not view_relation
            and {row.root_id for row in plan.answer_roots} & {"output-1", "output-2"}):
        raise ValueError("financial raw operands는 answer root가 될 수 없습니다")

    if (
            len(contract.items) != 1 or contract.groups
            or contract.premise_contracts
            or contract.presentation != intent_item.output.presentation
            or contract.completion != "complete"):
        raise ValueError("financial time derivation answer contract shape가 다릅니다")
    item = contract.items[0]
    expected_field_ids = (
        ["field-1", "field-2", "field-3"] if view_relation else
        ["field-1"] if combined_field else
        [f"field-{index + 1}" for index in range(len(expected_operators))]
    )
    if (
            item.item_id != intent_item.item_id or item.status != "ready"
            or item.projection.shape != intent_item.output.shape
            or item.projection.presentation != intent_item.output.presentation
            or item.projection.sort is not None
            or item.coverage != CoveragePartition(
                required=expected_field_ids, executable=expected_field_ids,
                limited=[])
            or item.limitation_bindings
            or len(item.projection.fields) != len(expected_field_ids)):
        raise ValueError("financial time derivation projection shape가 다릅니다")
    expected_fields = ([
        ("field-1", "as_filed", ["output-1"]),
        ("field-2", "restated", ["output-2"]),
        ("field-3", "difference", ["output-3"]),
    ] if view_relation else [
        ("field-1", "change", root_ids)
    ] if combined_field else [
        (f"field-{index + 1}", operator, [root_ids[index]])
        for index, operator in enumerate(expected_operators)
    ])
    actual_fields = [
        (field.field_id, field.field_key, field.answer_root_refs)
        for field in item.projection.fields
        if (field.binding_status == "executable"
            and not field.limitation_refs)
    ]
    if actual_fields != expected_fields:
        raise ValueError("financial time derivation projection root가 다릅니다")

    support_specs = _financial_comparison_support_specs(typed)
    expected_supports = [
        (f"support-{offset + index}", kind, True, detail,
         ["field-1" if combined_field else f"field-{root_index + 1}"],
         [root_id])
        for root_index, root_id in enumerate(root_ids)
        for offset in [root_index * len(support_specs)]
        for index, (kind, detail) in enumerate(support_specs, start=1)
    ]
    actual_supports = [
        (support.support_id, support.kind, support.required, support.detail,
         support.applies_to_field_ids, support.answer_root_refs)
        for support in item.support_requirements
    ]
    if actual_supports != expected_supports:
        raise ValueError("financial time derivation support inventory가 다릅니다")


def _source_authority(intent: SemanticIntent) -> dict[str, Any]:
    return {
        "source_item_order": [item.item_id for item in intent.answer_items],
        "source_groups": [
            CompiledAnswerGroup(group_id=group.group_id, item_ids=list(group.item_ids))
            for group in intent.answer_groups
        ],
        "source_field_surfaces": {
            item.item_id: list(item.output.field_surfaces)
            for item in intent.answer_items
        },
        "source_field_bindings": {},
        "source_premise_order": [premise.premise_id for premise in intent.premises],
        "source_premises": [
            SourcePremiseAuthority(
                premise_id=premise.premise_id,
                kind=premise.kind,
                raw_text=premise.raw_text,
                applies_to_item_ids=list(premise.applies_to_item_ids),
            )
            for premise in intent.premises
        ],
    }


#: 이슈 #44 — 「투자판단관련주요경영사항」서식명 + 연도 whole-target 문서 묶음.
#: `agent/stage1_v1_document_backends.py` 의 동명 함수와 의도적으로 중복이다
#: (resolver/compiler 계층 분리 관례 — `_validate_document_collection_intent`
#: 근처 「최근투자계획」 케이스도 같은 패턴).  두 계층이 서로 다른 것을
#: 골라내면 안 되므로 로직은 반드시 동일해야 한다.
_INVESTMENT_JUDGMENT_FORM = "투자판단관련주요경영사항"
_INVESTMENT_JUDGMENT_FORM_PATTERN = re.compile(
    r"투자\s*판단\s*(?:관련)?\s*주요\s*경영\s*사항")
_DOCUMENT_COLLECTION_YEAR_PATTERN = re.compile(
    r"(?<![0-9])(20[0-9]{2})\s*년(?!\s*[0-9]{1,2}\s*월)")


def _document_collection_form_year(item: Any) -> "tuple[str, str] | None":
    """Detect a closed 「서식명 + 연도」whole-target document collection."""

    if (item.output.projection_mode != "whole_target"
            or item.output.shape != "record_list"
            or item.output.field_surfaces):
        return None
    if _INVESTMENT_JUDGMENT_FORM_PATTERN.search(item.target.surface) is None:
        return None
    years = {
        match.group(1)
        for match in _DOCUMENT_COLLECTION_YEAR_PATTERN.finditer(item.target.surface)
    }
    for expression in item.scope.target_period_expressions:
        years |= {
            match.group(1)
            for match in _DOCUMENT_COLLECTION_YEAR_PATTERN.finditer(expression)
        }
    if len(years) != 1:
        return None
    return (_INVESTMENT_JUDGMENT_FORM, next(iter(years)))


def _build_document_collection_typed_plan(
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        intent_item: Any,
        ) -> ExecutionPlan:
    """Lower a whole-target document collection without inventing fields."""
    typed = resolution_item.resolution
    if not isinstance(typed, DocumentCollectionResolution):
        raise DeterministicPlanCompilerError(
            "document collection resolution이 필요합니다")
    selected_document = (
        typed.selected_document_id is not None
        and typed.selected_receipt_no is not None
    )
    # A named-form + year request narrows the otherwise-blanket selector to
    # exactly that form's receipts within the calendar year.  A year beyond
    # the corpus cutoff would otherwise build an inverted (empty) range —
    # fall back to the ordinary unscoped selector instead of raising.
    form_year = _document_collection_form_year(intent_item)
    if form_year is not None and f"{form_year[1]}0101" > typed.as_of:
        form_year = None
    # A resolver may bind a whole ``사업 내용`` overview to the latest annual
    # filing so its segment and product cells describe one reporting point.
    # Do not generalize that identity to a period/document-qualified request
    # or to a comparison: those coordinates already have their own selection
    # semantics.
    compact_target = re.sub(r"\s+", "", intent_item.target.surface)
    snapshot_selector = (
        selected_document
        and intent_item.target.kind == "topic"
        and compact_target in {"사업내용", "사업의내용"}
        and intent_item.operation == "retrieve"
        and intent_item.output.projection_mode == "whole_target"
        and not intent_item.output.field_surfaces
        and intent_item.selection is None
        and not intent_item.scope.target_period_expressions
        and intent_item.scope.as_of_expression is None
        and intent_item.scope.document_group_expression is None
        and not intent_item.scope.scope_qualifier_expressions
    )
    tasks = ([ResolvedNarrativeTask(
        task_id="task-1", operation="search", corp_codes=[typed.corp_code],
        corp_names=[typed.corp_name], as_of=typed.as_of,
        retrieval_query=typed.retrieval_query,
        document_selector=(
            DocumentSelector(
                doc_id=typed.selected_document_id,
                rcept_no=typed.selected_receipt_no,
            ) if snapshot_selector else None),
        periods=[], requested_slots=[],
    )] if typed.retrieval_query is not None else [ResolvedDocumentTask(
        task_id="task-1",
        operation="find",
        corp_code=typed.corp_code,
        corp_name=typed.corp_name,
        as_of=typed.as_of,
        selector=(
            DocumentSelector(
                doc_id=typed.selected_document_id,
                rcept_no=typed.selected_receipt_no,
            ) if selected_document else
            DocumentSelector(
                form=form_year[0], rcept_from=f"{form_year[1]}0101",
                rcept_to=min(f"{form_year[1]}1231", typed.as_of),
            ) if form_year is not None else
            DocumentSelector(rcept_to=typed.as_of)
        ),
    )])
    # A topic such as ``최근 투자계획`` deliberately remains a broad
    # narrative search: the newest periodic filing is not necessarily the
    # newest filing that contains the requested authoritative topic.  Preserve
    # the resolver-owned latest-selection decision as a public execution
    # default so Stage2 can rank only matching structural evidence by receipt
    # date.  Without this hand-off the query is silently weakened to the plain
    # surface ``투자 계획`` and FTS rank can select an older report.
    latest_default = (
        "최근→reference_date("
        f"{resolution.reference_date.isoformat()}) 이전 코퍼스 최신 관련 공시 "
        "선택 + 선택 기간·공시일을 답변에 명시"
        if typed.recent_selection else None
    )
    resolved_plan = ResolvedQueryPlan(
        revision=0,
        reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=tasks,
        derivations=[], premise_claims=[],
        applied_defaults=([] if latest_default is None else [latest_default]),
        presentation=None,
    )
    recent_narrative = intent_item.output.projection_mode == "named_fields"
    answer_roots = ([ExecutionAnswerRoot(
        root_id="output-1", item_id=intent_item.item_id,
        projection_mode="named_fields", field_id="field-1",
        plan_task_id="task-1", proof_ref=typed.selector_proof_ref,
    )] if recent_narrative else [ExecutionAnswerRoot(
        root_id="output-1", item_id=intent_item.item_id,
        projection_mode="whole_target", whole_target_id="whole-target-1",
        plan_task_id="task-1", proof_ref=typed.selector_proof_ref,
    )])
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan,
        applied_defaults=([] if latest_default is None else [AppliedDefault(
            policy="latest_relevant_disclosure_before_reference_date",
            basis="resolver-proved recent narrative selection",
            value=latest_default,
            evidence_refs=[typed.selector_proof_ref],
        )]),
        answer_roots=answer_roots,
        support_roots=[], premise_roots=[], plan_value_roots=[],
    )


def _build_holding_disclosure_typed_plan(
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        intent_item: Any,
        ) -> ExecutionPlan:
    """Lower an exact holding authority without extending public QueryPlan."""

    typed = resolution_item.resolution
    if not isinstance(typed, HoldingDisclosureResolution):
        raise DeterministicPlanCompilerError(
            "holding disclosure resolution이 필요합니다")
    executable = [
        row for row in typed.slot_bindings
        if row.binding_status != "limited"
    ]

    def public_slot(row: HoldingSlotBinding) -> str:
        if row.party_name is None:
            return row.slot
        return "holding-party:" + canonical_json({
            "slot": row.slot, "party_name": row.party_name,
        })

    field_outputs = [
        FieldOutputSpec(
            output_id=f"value-{row.source_field_index + 1}",
            slot=public_slot(row),
            value_kind=row.value_kind,
        )
        for row in executable
    ]
    requested_slots = [public_slot(row) for row in executable]
    if typed.privacy_notice_required:
        requested_slots.append("privacy_notice")
    plan = ResolvedQueryPlan(
        revision=0,
        reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=[ResolvedDisclosureTask(
            task_id="task-1",
            operation="lookup",
            corp_code=typed.issuer_corp_code,
            corp_name=typed.issuer_corp_name,
            as_of=typed.as_of,
            document_selector=DocumentSelector(
                doc_id=typed.document_id,
                rcept_no=typed.receipt_no,
                doc_group="holding",
            ),
            requested_slots=requested_slots,
            field_outputs=field_outputs,
        )],
        derivations=[], premise_claims=[], applied_defaults=[], presentation=None,
    )
    applied_defaults = ([] if typed.selection_basis != "reporter_latest_default"
                        else [AppliedDefault(
                            policy="latest_holding_filing_before_corpus_cutoff",
                            basis=("no filing date was supplied after the filer "
                                   "was uniquely resolved"),
                            value=typed.receipt_no,
                            evidence_refs=[typed.document_proof.proof_ref],
                        )])
    answer_roots = [ExecutionAnswerRoot(
        root_id=f"output-{row.source_field_index + 1}",
        item_id=intent_item.item_id,
        projection_mode="named_fields",
        field_id=f"field-{row.source_field_index + 1}",
        plan_output_id=f"value-{row.source_field_index + 1}",
        proof_ref=row.proof_ref,
    ) for row in executable]
    support_roots = [ExecutionSupportRoot(
        support_id=f"support-{row.source_field_index + 1}",
        kind="citation",
        item_id=intent_item.item_id,
        projection_mode="named_fields",
        field_id=f"field-{row.source_field_index + 1}",
        root_id=f"output-{row.source_field_index + 1}",
    ) for row in executable]
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=plan,
        applied_defaults=applied_defaults,
        answer_roots=answer_roots,
        support_roots=support_roots, premise_roots=[], plan_value_roots=[],
    )


def _premise_bound_answer_roots(
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        roots: list[ExecutionAnswerRoot],
        premise_id: str,
        ) -> list[str]:
    """Map one resolver premise proof to the exact source-bound answer roots."""

    proofs = [row for row in resolution.premise_proofs
              if row.premise_id == premise_id]
    if len(proofs) != 1:
        raise DeterministicPlanCompilerError(
            "premise proof inventory가 source premise와 다릅니다")
    root_by_proof = {
        proof.proof_ref: root.root_id
        for proof, root in zip(resolution_item.field_proofs, roots, strict=True)
    }
    root_ids = [root_by_proof[proof_ref]
                for proof_ref in proofs[0].proof_refs
                if proof_ref in root_by_proof]
    if not root_ids or len(root_ids) != len(proofs[0].proof_refs):
        raise DeterministicPlanCompilerError(
            "premise proof가 source-bound answer root와 닫히지 않습니다")
    return root_ids


def _build_selected_event_typed_plan(
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        intent_item: Any,
        premises: tuple[Any, ...] = (),
        ) -> tuple[ExecutionPlan, list[SupportRequirement]]:
    typed = resolution_item.resolution
    if not isinstance(typed, SelectedEventResolution):
        raise DeterministicPlanCompilerError("selected_event resolution이 필요합니다")
    intrinsic_status = bool(typed.timepoints)
    if intrinsic_status:
        requested_slots = _selected_event_requested_slots(intent_item, typed)
    else:
        requested_slots = normalize_slot_names(
            _resolve_colloquial_slots(intent_item.output.field_surfaces))
        # The literal 투자판단 관련 주요경영사항 form exposes its disclosed
        # body under ``2. 주요내용``.  Keep the semantic surface ``내용`` intact
        # (it is the user's literal request and grounding contract), but bind
        # its one closed implementation slot to that form label.  This is not
        # a generic synonym for all event documents.
        investment_judgment_content = (
            len(requested_slots) == 1
            and re.sub(r"\s+", "", requested_slots[0]) == "내용"
            and re.search(
                r"투자\s*판단\s*(?:관련)?\s*주요\s*경영\s*사항",
                intent_item.target.surface) is not None)
        if investment_judgment_content:
            requested_slots = ["주요내용"]
    # 「계약상대와 계약명」처럼 한 사건에서 칸 **여럿**을 뽑는 물음이 있다.
    # 근거·지원은 아래에서 이미 `field_proofs` 수만큼 만들므로, 출력 칸도 그
    # 수와 같으면 자리가 어긋나지 않는다.  수가 다르면 어느 근거가 어느 칸을
    # 받치는지 정해지지 않으므로 그대로 거절한다.
    if not intrinsic_status and (
            not requested_slots
            or len(requested_slots) != len(resolution_item.field_proofs)):
        raise DeterministicPlanCompilerError(
            "selected event 요청 slot 수가 field proof 수와 다릅니다")
    outputs = ([] if intrinsic_status else [FieldOutputSpec(
        output_id=f"output-{index + 1}", slot=slot,
        value_kind=_slot_value_kind(slot))
        for index, slot in enumerate(requested_slots)
    ])
    premise_claims = [PremiseClaim(
        claim_id=premise.premise_id, kind=premise.kind,
        raw_text=premise.raw_text,
        verify_tasks=[TaskVerificationRef(task_id="task-1")],
    ) for premise in premises]
    resolved_plan = ResolvedQueryPlan(
        revision=0, reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=[ResolvedEventTask(
            task_id="task-1", operation=typed.operation, corp_code=typed.corp_code,
            corp_name=typed.corp_name,
            selector=EventSelector(
                event_key=typed.event_key,
                seed_rcept_no=(
                    typed.selector_proof.source_receipt
                    if typed.selector_proof.proof_ref == (
                        f"source-event:{typed.event_key}:"
                        f"{typed.selector_proof.source_receipt}")
                    else None),
                # The event key remains the execution identity.  A
                # proof-bound public counterparty is additionally preserved
                # as a row selector for forms that repeat one requested field
                # for several recipients/investors.
                counterparty=(
                    typed.public_selector.counterparty
                    if typed.public_selector is not None else None)),
            timepoints=(list(typed.timepoints) if intrinsic_status
                        else [resolution.corpus_cutoff]),
            requested_slots=requested_slots,
            field_outputs=outputs,
        )],
        derivations=[], premise_claims=premise_claims,
        applied_defaults=[], presentation=None,
    )
    roots = [ExecutionAnswerRoot(
        root_id=f"output-{index + 1}", item_id=resolution_item.item_id,
        field_id=f"field-{index + 1}",
        **({"plan_task_id": "task-1"} if intrinsic_status else {
            "plan_output_id": f"output-{index + 1}"}),
        proof_ref=proof.proof_ref,
    ) for index, proof in enumerate(resolution_item.field_proofs)]
    supports = [SupportRequirement(
        support_id=f"support-{index + 1}", kind="citation",
        detail=(f"event_key={typed.event_key};root_receipt={typed.root_receipt};"
                f"proof_ref={typed.selector_proof.proof_ref}"),
        applies_to_field_ids=[f"field-{index + 1}"],
        answer_root_refs=[f"output-{index + 1}"],
    ) for index in range(len(roots))]
    support_roots = [ExecutionSupportRoot(
        support_id=support.support_id, kind=support.kind,
        item_id=resolution_item.item_id,
        field_id=support.applies_to_field_ids[0],
        root_id=support.answer_root_refs[0],
    ) for support in supports]
    premise_roots = [ExecutionPremiseRoot(
        premise_id=premise.premise_id,
        root_ids=_premise_bound_answer_roots(
            resolution, resolution_item, roots, premise.premise_id),
        task_refs=["task-1"],
    ) for premise in premises]
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan,
        applied_defaults=list(resolution_item.applied_defaults), answer_roots=roots,
        support_roots=support_roots, premise_roots=premise_roots,
        plan_value_roots=[],
    ), supports


def _build_event_collection_typed_plan(
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        intent_item: Any,
        ) -> tuple[ExecutionPlan, list[SupportRequirement]]:
    """Lower a proved event set to one general list task.

    Exact member identities remain answer/support roots.  They are not task
    selectors: Stage2 receives the public event-family selector and retrieves
    the complete set once, rather than executing one private lookup per event.
    """
    typed = resolution_item.resolution
    if not isinstance(typed, EventCollectionResolution):
        raise DeterministicPlanCompilerError("event_collection resolution이 필요합니다")
    roots: list[ExecutionAnswerRoot] = []
    supports: list[SupportRequirement] = []
    support_roots: list[ExecutionSupportRoot] = []
    collapsed_projection = (
        len(intent_item.output.field_surfaces) == 1
        and len(typed.requested_slots) != 1)
    for event_index, member in enumerate(typed.events, start=1):
        event_proof = next(
            proof for proof in member.proof_refs
            if proof.startswith(f"canonical:event:{member.event_key}:")
        )
        for field_index, _slot in enumerate(typed.requested_slots, start=1):
            root_id = f"output-{(event_index - 1) * len(typed.requested_slots) + field_index}"
            field_id = "field-1" if collapsed_projection else f"field-{field_index}"
            roots.append(ExecutionAnswerRoot(
                root_id=root_id, item_id=resolution_item.item_id, field_id=field_id,
                plan_task_id="task-1", proof_ref=event_proof,
            ))
            support_id = f"support-{event_index}-{field_index}"
            supports.append(SupportRequirement(
                support_id=support_id, kind="citation",
                detail=(f"event_key={member.event_key};root_receipt={member.root_receipt};"
                        f"proof_ref={event_proof}"),
                applies_to_field_ids=[field_id], answer_root_refs=[root_id],
            ))
            support_roots.append(ExecutionSupportRoot(
                support_id=support_id, kind="citation", item_id=resolution_item.item_id,
                field_id=field_id, root_id=root_id,
            ))
    if not typed.events:
        if typed.collection_proof_ref is None:
            raise DeterministicPlanCompilerError(
                "empty event collection scan proof가 필요합니다")
        for field_index, _slot in enumerate(
                typed.requested_slots, start=1):
            root_id = f"output-{field_index}"
            field_id = (
                "field-1" if collapsed_projection
                else f"field-{field_index}")
            roots.append(ExecutionAnswerRoot(
                root_id=root_id, item_id=resolution_item.item_id,
                field_id=field_id, plan_task_id="task-1",
                proof_ref=typed.collection_proof_ref,
            ))
            support_id = f"support-{field_index}"
            supports.append(SupportRequirement(
                support_id=support_id, kind="citation",
                detail=f"collection_scan={typed.collection_proof_ref}",
                applies_to_field_ids=[field_id], answer_root_refs=[root_id],
            ))
            support_roots.append(ExecutionSupportRoot(
                support_id=support_id, kind="citation",
                item_id=resolution_item.item_id, field_id=field_id,
                root_id=root_id,
            ))
    task = (ResolvedCorrectionTask(
        task_id="task-1", operation="history", corp_code=typed.corp_code,
        corp_name=typed.corp_name, as_of=typed.as_of,
        document_selector=DocumentSelector(
            doc_group="exchange", rcept_to=typed.as_of,
            is_correction=True),
        event_selector=EventSelector(event_type=typed.event_type),
        requested_slots=list(typed.requested_slots), field_outputs=[],
    ) if typed.public_task_kind == "correction" else ResolvedDisclosureTask(
        task_id="task-1", operation="list", corp_code=typed.corp_code,
        corp_name=typed.corp_name, as_of=typed.as_of,
        event_selector=EventSelector(
            event_type=typed.event_type, counterparty=typed.counterparty,
            keywords=list(typed.keywords), event_from=typed.event_from,
            event_to=typed.event_to),
        requested_slots=list(typed.requested_slots), field_outputs=[],
        argmax_slot=typed.argmax_slot, argmax_direction=typed.argmax_direction,
    ) if typed.public_task_kind == "disclosure" else ResolvedEventTask(
        task_id="task-1", operation="list", corp_code=typed.corp_code,
        corp_name=typed.corp_name,
        selector=EventSelector(event_type=typed.event_type, counterparty=typed.counterparty,
            keywords=list(typed.keywords), event_from=typed.event_from,
            event_to=typed.event_to),
        timepoints=[typed.as_of], requested_slots=list(typed.requested_slots),
        field_outputs=[], argmax_slot=typed.argmax_slot,
        argmax_direction=typed.argmax_direction))
    plan = ResolvedQueryPlan(
        revision=0, reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff, tasks=[task], derivations=[],
        premise_claims=[], applied_defaults=[], presentation=None,
    )
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version, resolved_plan=plan,
        applied_defaults=[], answer_roots=roots, support_roots=support_roots,
        premise_roots=[], plan_value_roots=[],
    ), supports


def _build_event_amount_change_typed_plan(
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        ) -> tuple[ExecutionPlan, list[SupportRequirement]]:
    typed = resolution_item.resolution
    if not isinstance(typed, EventAmountChangeResolution):
        raise DeterministicPlanCompilerError(
            "event_amount_change resolution이 필요합니다")
    tasks = [
        ResolvedEventTask(
            task_id=f"task-{index}", operation="status",
            corp_code=typed.corp_code, corp_name=typed.corp_name,
            selector=EventSelector(event_key=typed.event_key),
            timepoints=[timepoint], requested_slots=[typed.requested_slot],
            field_outputs=[FieldOutputSpec(
                output_id=f"output-{index}", slot=typed.requested_slot,
                value_kind="money")],
        )
        for index, timepoint in enumerate(typed.timepoints, start=1)
    ]
    resolved_plan = ResolvedQueryPlan(
        revision=0, reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=tasks,
        derivations=[Derivation(
            output_id="output-3", operator="difference",
            # Contract change is later minus earlier, never an unsigned gap.
            operands=[OutputRef(output_id="output-2"),
                      OutputRef(output_id="output-1")],
        )],
        premise_claims=[], applied_defaults=[], presentation=None,
    )
    detail = (
        f"event_key={typed.event_key};root_receipt={typed.root_receipt};"
        f"timepoints={','.join(typed.timepoints)};slot={typed.requested_slot};"
        f"proof_ref={typed.selector_proof.proof_ref};"
        f"observation_proofs={','.join(row.proof.proof_ref for row in typed.observations)};"
        f"money_type={typed.observations[0].currency}:"
        f"{typed.observations[0].unit}:scale-{typed.observations[0].scale}"
    )
    supports = [SupportRequirement(
        support_id="support-1", kind="coordinate", detail=detail,
        applies_to_field_ids=["field-1"],
        answer_root_refs=["output-3"],
        plan_root_refs=["plan-root-1", "plan-root-2", "plan-root-3"],
    )]
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan, applied_defaults=[],
        answer_roots=[ExecutionAnswerRoot(
            root_id="output-3", item_id=resolution_item.item_id,
            field_id="field-1", plan_output_id="output-3",
            proof_ref=resolution_item.field_proofs[0].proof_ref,
        )],
        support_roots=[ExecutionSupportRoot(
            support_id="support-1", kind="coordinate",
            item_id=resolution_item.item_id, field_id="field-1",
            root_id="output-3",
            plan_root_ids=["plan-root-1", "plan-root-2", "plan-root-3"],
        )],
        premise_roots=[],
        plan_value_roots=[
            ExecutionPlanValueRoot(
                plan_root_id="plan-root-1", plan_output_id="output-1",
                value_kind="money", producer_task_id="task-1",
            ),
            ExecutionPlanValueRoot(
                plan_root_id="plan-root-2", plan_output_id="output-2",
                value_kind="money", producer_task_id="task-2",
            ),
            ExecutionPlanValueRoot(
            plan_root_id="plan-root-3", plan_output_id="output-3",
            value_kind="money",
            producer_derivation_output_id="output-3",
            answer_mirror_root_id="output-3",
            ),
        ],
    ), supports


def _same_day_status_support_specs(
        document: SameDayDocumentCandidatesResolution,
        status: TerminationReportedStatusResolution,
        ) -> list[tuple[str, str, str, str, str]]:
    specs = [
        (
            f"support-{index + 1}", "item-1", "field-1",
            f"output-{index + 1}", f"proof_ref={candidate.proof_ref}",
        )
        for index, candidate in enumerate(document.candidates)
    ]
    specs.append((
        f"support-{len(specs) + 1}", "item-2", "field-2",
        f"output-{len(specs) + 1}", f"proof_ref={status.status_proof.proof_ref}",
    ))
    return specs


def _build_periodic_document_narrative_typed_plan(
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        intent_item: Any,
        ) -> tuple[ExecutionPlan, list[SupportRequirement]]:
    typed = resolution_item.resolution
    if not isinstance(typed, PeriodicDocumentNarrativeResolution):
        raise DeterministicPlanCompilerError(
            "periodic document narrative resolution이 필요합니다")
    selector = DocumentSelector(
        doc_id=typed.document_id, rcept_no=typed.receipt_no)
    resolved_plan = ResolvedQueryPlan(
        revision=0,
        reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=[
            ResolvedDocumentTask(
                task_id="task-1", operation="find",
                corp_code=typed.corp_code, corp_name=typed.corp_name,
                as_of=resolution.corpus_cutoff,
                selector=selector,
            ),
            ResolvedNarrativeTask(
                task_id="task-2", operation="search",
                corp_codes=[typed.corp_code], corp_names=[typed.corp_name],
                as_of=resolution.corpus_cutoff,
                retrieval_query=(
                    typed.source_retrieval_query or intent_item.target.surface),
                document_selector=selector,
                periods=[],
                requested_slots=(
                    list(typed.canonical_requested_slots)
                    if typed.canonical_requested_slots else [
                        "".join(surface.split())
                        for surface in intent_item.output.field_surfaces
                    ]),
            ),
        ],
        derivations=[], premise_claims=[],
        applied_defaults=[
            f"as_of=corpus_cutoff({row.value})"
            for row in resolution_item.applied_defaults
            if row.policy == "as_of" and row.basis == "corpus_cutoff"],
        presentation=None,
    )
    support_specs = _periodic_document_support_specs(typed, resolution_item)
    supports = [SupportRequirement(
        support_id=support_id, kind=kind, detail=detail,
        applies_to_field_ids=[field_id], answer_root_refs=[root_id],
    ) for support_id, field_id, root_id, kind, detail in support_specs]
    roots = [
        ExecutionAnswerRoot(
            root_id=f"output-{index + 1}", item_id=resolution_item.item_id,
            field_id=f"field-{index + 1}", plan_task_id="task-2",
            proof_ref=next(
                proof.proof_ref for proof in resolution_item.field_proofs
                if proof.source_field_index == index),
        )
        for index in typed.executable_field_indexes
    ]
    # Support roots are item-scoped; keep the item id explicit rather than
    # deriving it from field IDs (which would be a reverse inference).
    support_roots = [ExecutionSupportRoot(
        support_id=support.support_id, kind=support.kind,
        item_id=resolution_item.item_id,
        field_id=support.applies_to_field_ids[0],
        root_id=support.answer_root_refs[0],
    ) for support in supports]
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan,
        applied_defaults=list(resolution_item.applied_defaults), answer_roots=roots,
        support_roots=support_roots, premise_roots=[],
    ), supports


def _build_same_day_status_typed_plan(
        resolution: AuthoritativeResolution,
        intent_items: tuple[Any, Any],
        ) -> tuple[ExecutionPlan, list[SupportRequirement]]:
    document_resolution = resolution.items[0].resolution
    status_resolution = resolution.items[1].resolution
    assert isinstance(document_resolution, SameDayDocumentCandidatesResolution)
    assert isinstance(status_resolution, TerminationReportedStatusResolution)
    document_item, event_item = intent_items
    document_proof = resolution.items[0].field_proofs[0]
    event_proof = resolution.items[1].field_proofs[0]
    document_tasks = [
        ResolvedDocumentTask(
            task_id=f"task-{index + 1}", operation="find",
            corp_code=document_resolution.issuer_corp_code,
            corp_name=document_resolution.issuer_corp_name,
            as_of=document_resolution.as_of,
            selector=DocumentSelector(rcept_no=candidate.rcept_no),
        )
        for index, candidate in enumerate(document_resolution.candidates)
    ]
    resolved_plan = ResolvedQueryPlan(
        revision=0, reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=[*document_tasks,
            ResolvedEventTask(
                task_id="task-3", operation="status",
                corp_code=document_resolution.issuer_corp_code,
                corp_name=document_resolution.issuer_corp_name,
                selector=EventSelector(event_key=status_resolution.event_key),
                timepoints=[resolution.corpus_cutoff],
                requested_slots=list(event_item.output.field_surfaces),
            ),
        ], derivations=[], premise_claims=[], applied_defaults=[],
        presentation=None,
    )
    support_specs = _same_day_status_support_specs(
        document_resolution, status_resolution)
    supports = [SupportRequirement(
        support_id=support_id, kind="evidence", detail=detail,
        applies_to_field_ids=[field_id], answer_root_refs=[root_id],
    ) for support_id, _, field_id, root_id, detail in support_specs]
    roots = [
        ExecutionAnswerRoot(
            root_id=f"output-{index + 1}", item_id=document_item.item_id,
            field_id="field-1", plan_task_id=f"task-{index + 1}",
            proof_ref=document_proof.proof_ref,
        )
        for index in range(len(document_resolution.candidates))
    ] + [
        ExecutionAnswerRoot(
            root_id="output-3", item_id=event_item.item_id, field_id="field-2",
            plan_task_id="task-3", proof_ref=event_proof.proof_ref,
        ),
    ]
    support_roots = [ExecutionSupportRoot(
        support_id=support.support_id, kind=support.kind,
        item_id=item_id, field_id=field_id, root_id=root_id,
    ) for support, (_, item_id, field_id, root_id, _) in zip(
        supports, support_specs, strict=True)]
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan, applied_defaults=[], answer_roots=roots,
        support_roots=support_roots, premise_roots=[],
    ), supports


def _build_termination_reported_status_typed_plan(
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        intent_item: Any,
        premises: tuple[Any, ...] = (),
        ) -> tuple[ExecutionPlan, list[SupportRequirement]]:
    typed = resolution_item.resolution
    if not isinstance(typed, TerminationReportedStatusResolution):
        raise DeterministicPlanCompilerError(
            "termination reported status resolution이 필요합니다")
    # Status is an operation intrinsic; only source fields after it become
    # requested slots.  A termination report also carries its stated reason,
    # which remains an explicit event slot rather than a synthetic status.
    _normalized_fields, field_roles = _termination_reported_status_field_roles(
        intent_item)
    amount_indexes = [index for index, role in enumerate(field_roles)
                      if role == "termination_amount"]
    requested_slots = ["해지금액" for _index in amount_indexes]
    if amount_indexes:
        requested_slots.append("해지사유")
    if len(requested_slots) != len(set(requested_slots)):
        raise DeterministicPlanCompilerError(
            "termination reported status slot inventory가 중복됩니다")
    observation_timepoint = resolution.corpus_cutoff
    if intent_item.scope.as_of_expression is not None:
        as_of, as_of_end, error = _target_date_range(
            intent_item.scope.as_of_expression,
            reference_date=resolution.reference_date)
        if error is not None or as_of is None or as_of != as_of_end:
            raise DeterministicPlanCompilerError(
                "termination reported status observation date가 다릅니다")
        observation_timepoint = as_of
    premise_claims = [PremiseClaim(
        claim_id=premise.premise_id, kind=premise.kind,
        raw_text=premise.raw_text,
        verify_tasks=[TaskVerificationRef(task_id="task-1")],
    ) for premise in premises]
    resolved_plan = ResolvedQueryPlan(
        revision=0, reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=[ResolvedEventTask(
            task_id="task-1", operation="status",
            corp_code=typed.issuer_corp_code, corp_name=typed.issuer_corp_name,
            selector=EventSelector(event_key=typed.event_key),
            timepoints=[observation_timepoint],
            requested_slots=requested_slots, field_outputs=[],
        )],
        derivations=[], premise_claims=premise_claims,
        applied_defaults=[], presentation=None,
    )
    roots = [ExecutionAnswerRoot(
        root_id=f"output-{index + 1}", item_id=resolution_item.item_id,
        field_id=f"field-{index + 1}", plan_task_id="task-1",
        proof_ref=proof.proof_ref,
    ) for index, proof in enumerate(resolution_item.field_proofs)]
    supports = [SupportRequirement(
        support_id=f"support-{index + 1}", kind="citation",
        detail=(f"status_receipt={typed.status_receipt};event_key={typed.event_key};"
                f"proof_ref={typed.status_proof.proof_ref}"),
        applies_to_field_ids=[f"field-{index + 1}"],
        answer_root_refs=[f"output-{index + 1}"],
    ) for index in range(len(roots))]
    support_roots = [ExecutionSupportRoot(
        support_id=support.support_id, kind=support.kind,
        item_id=resolution_item.item_id,
        field_id=support.applies_to_field_ids[0],
        root_id=support.answer_root_refs[0],
    ) for support in supports]
    premise_roots = [ExecutionPremiseRoot(
        premise_id=premise.premise_id,
        root_ids=_premise_bound_answer_roots(
            resolution, resolution_item, roots, premise.premise_id),
        task_refs=["task-1"],
    ) for premise in premises]
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan, applied_defaults=[], answer_roots=roots,
        support_roots=support_roots, premise_roots=premise_roots,
        plan_value_roots=[],
    ), supports


def _build_typed_plan(
        resolution: AuthoritativeResolution,
        resolution_item: Any,
        defaults: list[AppliedDefault],
        ) -> tuple[ExecutionPlan, list[SupportRequirement]]:
    if isinstance(resolution_item.resolution, SameDayDocumentCandidatesResolution):
        raise DeterministicPlanCompilerError(
            "same-day status plan은 paired intent authority가 필요합니다")
    if isinstance(resolution_item.resolution,
                  PeriodicDocumentNarrativeResolution):
        raise DeterministicPlanCompilerError(
            "periodic document plan은 source intent authority가 필요합니다")
    if isinstance(resolution_item.resolution, FinancialComparisonResolution):
        raise DeterministicPlanCompilerError(
            "financial relation plan에는 source intent authority가 필요합니다")
    coordinate = resolution_item.resolution
    fact = FactSpec(
        output_id="output-1",
        corp_code=coordinate.corp_code,
        corp_name=coordinate.corp_name,
        concept=coordinate.concept,
        period_start=coordinate.period_start,
        period_end=coordinate.period_end,
        period_type=coordinate.period_type,
        cumulative=coordinate.cumulative,
        scope=coordinate.scope,
        statement=coordinate.statement,
    )
    resolved_plan = ResolvedQueryPlan(
        revision=0,
        reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=[ResolvedFinancialTask(
            task_id="task-1", as_of=coordinate.as_of,
            view=coordinate.view, facts=[fact],
        )],
        derivations=[], premise_claims=[], applied_defaults=[], presentation=None,
    )
    support_specs = (
        ("unit", "source-reported unit"),
        ("scope", f"scope={coordinate.scope}"),
        ("period", _financial_period_support_detail(coordinate)),
        ("evidence", f"proof_ref={resolution_item.field_proofs[0].proof_ref}"),
    )
    supports = [SupportRequirement(
        support_id=f"support-{index}", kind=kind, detail=detail,
        applies_to_field_ids=["field-1"], answer_root_refs=["output-1"],
    ) for index, (kind, detail) in enumerate(support_specs, start=1)]
    answer_root = ExecutionAnswerRoot(
        root_id="output-1", item_id=resolution_item.item_id,
        field_id="field-1", plan_output_id="output-1",
        proof_ref=resolution_item.field_proofs[0].proof_ref,
    )
    support_roots = [ExecutionSupportRoot(
        support_id=support.support_id, kind=support.kind,
        item_id=resolution_item.item_id, field_id="field-1", root_id="output-1",
    ) for support in supports]
    plan = ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan,
        applied_defaults=defaults,
        answer_roots=[answer_root], support_roots=support_roots,
        premise_roots=[],
    )
    return plan, supports


def _build_premise_financial_typed_plan(
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        premises: tuple[Any, ...],
        ) -> tuple[ExecutionPlan, list[SupportRequirement]]:
    """Lower one financial fact and its source-bound premise verification.

    A premise has no independent lookup: it is verified against the same fact
    that answers the user item.  The plan-value root makes that shared proof
    explicit and lets the contract bind each premise to answer, task, and
    canonical value namespaces without inventing a verdict literal.
    """
    coordinate = resolution_item.resolution
    if not isinstance(coordinate, FinancialResolution):
        raise DeterministicPlanCompilerError(
            "premise financial lowering에는 financial resolution이 필요합니다")
    if len(resolution.premise_proofs) != len(premises):
        raise DeterministicPlanCompilerError(
            "premise financial lowering proof/source inventory가 다릅니다")
    fact = FactSpec(
        output_id="output-1", corp_code=coordinate.corp_code,
        corp_name=coordinate.corp_name, concept=coordinate.concept,
        period_start=coordinate.period_start, period_end=coordinate.period_end,
        period_type=coordinate.period_type, cumulative=coordinate.cumulative,
        scope=coordinate.scope, statement=coordinate.statement,
    )
    claims = [PremiseClaim(
        claim_id=premise.premise_id, kind=premise.kind,
        raw_text=premise.raw_text,
        verify_with=[OutputRef(output_id="output-1")],
        verify_tasks=[TaskVerificationRef(task_id="task-1")],
    ) for premise in premises]
    resolved_plan = ResolvedQueryPlan(
        revision=0, reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=[ResolvedFinancialTask(
            task_id="task-1", as_of=coordinate.as_of,
            view=coordinate.view, facts=[fact],
        )],
        derivations=[], premise_claims=claims, applied_defaults=[],
        presentation=None,
    )
    support_specs = (
        ("unit", "source-reported unit"),
        ("scope", f"scope={coordinate.scope}"),
        ("period", _financial_period_support_detail(coordinate)),
        ("evidence", f"proof_ref={resolution_item.field_proofs[0].proof_ref}"),
    )
    supports = [SupportRequirement(
        support_id=f"support-{index}", kind=kind, detail=detail,
        applies_to_field_ids=["field-1"], answer_root_refs=["output-1"],
        plan_root_refs=["plan-root-1"],
    ) for index, (kind, detail) in enumerate(support_specs, start=1)]
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan, applied_defaults=[],
        answer_roots=[ExecutionAnswerRoot(
            root_id="output-1", item_id=resolution_item.item_id,
            field_id="field-1", plan_output_id="output-1",
            proof_ref=resolution_item.field_proofs[0].proof_ref,
        )],
        support_roots=[ExecutionSupportRoot(
            support_id=support.support_id, kind=support.kind,
            item_id=resolution_item.item_id, field_id="field-1",
            root_id="output-1", plan_root_ids=["plan-root-1"],
        ) for support in supports],
        premise_roots=[ExecutionPremiseRoot(
            premise_id=premise.premise_id, root_ids=["output-1"],
            task_refs=["task-1"], plan_root_ids=["plan-root-1"],
        ) for premise in premises],
        plan_value_roots=[ExecutionPlanValueRoot(
            plan_root_id="plan-root-1", plan_output_id="output-1",
            value_kind="money", producer_task_id="task-1",
            answer_mirror_root_id="output-1",
        )],
    ), supports


def _is_discrete_q4_time_comparison(comparison) -> bool:
    if not isinstance(comparison, FinancialComparisonResolution) or len(comparison.operands) != 2:
        return False
    first, second = comparison.operands
    from agent.planning import concept_axes
    return (comparison.requested_operators == ["percent_change"]
            and first.corp_code == second.corp_code
            and first.scope == second.scope and first.view == second.view
            and first.concept == second.concept
            and concept_axes(first.concept).aggregation == "additive_duration"
            and first.period_type == second.period_type == "quarter"
            and first.cumulative is False and second.cumulative is False
            and first.period_start == date(first.period_end.year, 10, 1)
            and first.period_end == date(first.period_end.year, 12, 31)
            and second.period_start == date(first.period_end.year, 7, 1)
            and second.period_end == date(first.period_end.year, 9, 30))


def _build_financial_comparison_typed_plan(
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        intent_item: Any,
        *, _expand_discrete_q4: bool = True,
        ) -> tuple[ExecutionPlan, list[SupportRequirement]]:
    """두 재무 operand 비교를 **문항과 무관하게** 낮춘다.

    예전 이름은 `_build_g_a_004_typed_plan` 이었다. 몸통은 처음부터 operand 를
    순회할 뿐 문항별 값이 없었는데 **이름과 오류 메시지가 한 문항을 가리켜**,
    처음 보는 질문에는 쓸 수 없는 코드처럼 보였다. 실제로 blind 문항이 오면
    분기 이름을 찾지 못해 막힌다.

    승자(`argmax`)와 비방향 차이(`absolute_difference`)는 **컴파일러가 소유한다** — operand 는
    답 루트가 아니라 그 두 파생이 소비하는 사실이다.
    """

    comparison = resolution_item.resolution
    if not isinstance(comparison, FinancialComparisonResolution):
        raise DeterministicPlanCompilerError(
            "financial_comparison resolution이 필요합니다")
    operands = comparison.operands
    if len(operands) > 2:
        return _build_nary_financial_ranking_typed_plan(
            resolution, resolution_item, intent_item)
    account_paths = _financial_view_account_paths(resolution_item, comparison)
    facts = [FactSpec(
        output_id=f"output-{index}",
        corp_code=operand.corp_code,
        corp_name=operand.corp_name,
        concept=operand.concept,
        period_start=operand.period_start,
        period_end=operand.period_end,
        period_type=operand.period_type,
        cumulative=operand.cumulative,
        scope=operand.scope,
        statement=operand.statement,
        account_path=account_paths[index - 1],
    ) for index, operand in enumerate(operands, start=1)]
    from agent.contracts import Derivation, OutputRef
    same_company = operands[0].corp_code == operands[1].corp_code
    # 이슈 #59 1단계 — 회사가 다른 concept_ratio(「A사 매출은 B사의 몇
    # 배」)는 same-company 분기의 concept_ratio 처리를 그대로 재사용한다
    # (operand 순서·task·field 결속 로직이 corp_code 가 아니라 아래
    # `same_company` 변수 값으로만 갈리므로, 조건만 넓히면 뒤따르는 모든
    # 참조가 자동으로 맞게 닫힌다). `same_company` 변수 자체는 바꾸지
    # 않는다 — 아래 task/operand 순서 결정이 실제 corp_code 동일 여부를
    # 그대로 봐야 한다.
    if same_company or comparison.requested_operators in (
            ["concept_ratio"], ["sum"]):
        operators = list(comparison.requested_operators)
        if operators == ["concept_ratio"]:
            # 지표 간 나눗셈은 시점·scope·view가 아니라 concept 만 다르다 —
            # `FinancialComparisonResolution.validate_operands` 가 이미 그
            # 좌표 관계를 닫았으므로 여기서는 operation 모양만 본다.
            if intent_item.operation not in ("retrieve", "compare"):
                raise DeterministicPlanCompilerError(
                    "concept_ratio operation은 retrieve 또는 compare여야 합니다")
        elif operators == ["sum"]:
            # 이슈 #124 (합계) — 같은 concept·같은 scope·같은 period(회사만
            # 다름) 또는 같은 회사·같은 concept의 서로 다른 period(회사는
            # 같음)는 `FinancialComparisonResolution.validate_operands`가
            # 이미 좌표 관계를 닫았다. concept_ratio와 같은 이유로 여기서는
            # operation 모양만 본다 — 「연간」 제약은 두지 않는다: 회사가
            # 다른 sum은 애초에 period가 같아 그 축이 아예 없고, 회사가 같은
            # sum(예: 2024·2025년 매출액 합계)은 연간이 아닌 분기 합계도
            # 막을 이유가 없다.
            if intent_item.operation not in ("retrieve", "compare"):
                raise DeterministicPlanCompilerError(
                    "sum operation은 retrieve 또는 compare여야 합니다")
        elif intent_item.operation == "retrieve":
            if operators == ["discrete_from_cumulative"]:
                if not all(operand.cumulative is True for operand in operands):
                    raise DeterministicPlanCompilerError(
                        "discrete_from_cumulative operands는 cumulative=true여야 합니다")
            elif (
                    not all(
                        operand.period_type == "annual"
                        or (operand.period_type == "instant"
                            and operand.period_end.month == 12)
                        for operand in operands)
                    or any(operator not in {"difference", "percent_change"}
                           for operator in operators)
            ):
                raise DeterministicPlanCompilerError(
                    "annual retrieve relation operator가 잘못되었습니다")
        elif intent_item.operation == "compare":
            allowed = (
                {"absolute_difference"}
                if operands[0].scope != operands[1].scope
                else {"difference"}
                if operands[0].view != operands[1].view
                else {"difference", "percent_change"})
            if any(operator not in allowed for operator in operators):
                raise DeterministicPlanCompilerError(
                    "compare time derivation operator가 잘못되었습니다")
        else:
            raise DeterministicPlanCompilerError(
                "time derivation operation은 retrieve 또는 compare여야 합니다")
        view_relation = operands[0].view != operands[1].view
        expected_field_count = (
            3 if view_relation else
            1 if len(intent_item.output.field_surfaces) == 1 else len(operators)
        )
        if (len(resolution_item.field_proofs) != expected_field_count
                or (len(intent_item.output.field_surfaces) == 2
                    and len(operators) != 2)):
            raise DeterministicPlanCompilerError(
                "time derivation 요청 필드와 operator 결속이 다릅니다")
    else:
        operators = list(comparison.requested_operators)
        if len(resolution_item.field_proofs) != len(operators):
            raise DeterministicPlanCompilerError(
                "company comparison 요청 필드와 operator 수가 다릅니다")
        if any(operator not in {"argmax", "absolute_difference"}
               for operator in operators):
            raise DeterministicPlanCompilerError(
                "company comparison operator가 잘못되었습니다")

    derivation_operands = (
        [OutputRef(output_id="output-2"), OutputRef(output_id="output-1")]
        if same_company and operands[0].view != operands[1].view
        else [OutputRef(output_id="output-1"), OutputRef(output_id="output-2")]
    )
    derivations = [
        Derivation(
            output_id=f"output-{index + 3}", operator=operator,
            operands=derivation_operands,
            presentation=(
                comparison.presentation if operator == "concept_ratio" else None),
            direction=(
                comparison.direction if operator == "argmax" else "maximum"),
        )
        for index, operator in enumerate(operators)
    ]
    premise_claims = ([] if comparison.verification_claim is None else [
        PremiseClaim(
            claim_id=comparison.verification_premise_id, kind="comparison",
            raw_text=comparison.verification_claim,
            verify_with=[OutputRef(output_id="output-3")],
        )
    ])
    # 이슈 #182 — 이 split은 원래 same-company 두-시점 관계 전용이었다. 회사가
    # 다른 argmax/absolute_difference도 (as_of, view)가 다르면 같은 이유로
    # 나눈다: #149 회계연도말 pin은 회사마다 독립적으로 자기 사업보고서
    # 날짜를 낸다 — 두 회사가 같은 날 사업보고서를 낼 이유가 없다. 한
    # task에 두 회사를 몰아넣으면 task 하나의 as_of가 먼저 신고한 회사
    # 날짜로 고정돼, 그보다 늦게 신고한 회사의 사실이 그 시점엔 아직
    # 존재하지 않아 조회가 "not_found"로 닫힌다(EG2-023: LG이노텍
    # 20260312 · SK하이닉스 20260317 실측으로 직접 확인함).
    tasks = (
        [ResolvedFinancialTask(
            task_id=f"task-{index}", as_of=operand.as_of,
            view=operand.view, facts=[fact],
        ) for index, (operand, fact) in enumerate(
            zip(operands, facts, strict=True), start=1)]
        if (
            (operands[0].as_of, operands[0].view)
            != (operands[1].as_of, operands[1].view)
        )
        else [ResolvedFinancialTask(
            task_id="task-1", as_of=operands[0].as_of,
            view=operands[0].view, facts=facts,
        )]
    )
    if _expand_discrete_q4 and _is_discrete_q4_time_comparison(comparison):
        q4 = facts[0]
        year = q4.period_end.year
        cumulative_facts = [
            q4.model_copy(update={"output_id": "output-4", "period_start": date(year, 1, 1),
                                  "period_type": "annual", "cumulative": True}),
            q4.model_copy(update={"output_id": "output-5", "period_start": date(year, 1, 1),
                                  "period_end": date(year, 9, 30), "period_type": "quarter", "cumulative": True}),
        ]
        tasks = [task.model_copy(update={"facts": [expanded
                    for fact in task.facts
                    for expanded in (cumulative_facts if fact.output_id == "output-1" else [fact])]})
                 for task in tasks]
        derivations.insert(0, Derivation(
            output_id="output-1", operator="discrete_from_cumulative",
            operands=[OutputRef(output_id="output-4"), OutputRef(output_id="output-5")]))
    resolved_plan = ResolvedQueryPlan(
        revision=0,
        reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=tasks,
        derivations=derivations,
        premise_claims=premise_claims, applied_defaults=[], presentation=None,
    )
    support_specs = _financial_comparison_support_specs(comparison)
    supports: list[SupportRequirement] = []
    view_relation = same_company and operands[0].view != operands[1].view
    combined_time_field = same_company and not view_relation and len(
        intent_item.output.field_surfaces) == 1
    bindings = (
        [("field-1", "output-1"), ("field-2", "output-2"),
         ("field-3", "output-3")]
        if view_relation else [
            ("field-1" if combined_time_field else f"field-{index + 1}",
             f"output-{index + 3}")
            for index in range(len(operators))
        ]
    )
    for binding_index, (field_id, root_id) in enumerate(bindings):
        offset = binding_index * len(support_specs)
        for index, (kind, detail) in enumerate(support_specs, start=1 + offset):
            supports.append(SupportRequirement(
                support_id=f"support-{index}", kind=kind, detail=detail,
                applies_to_field_ids=[field_id], answer_root_refs=[root_id],
            ))
    answer_roots = [
        ExecutionAnswerRoot(
            root_id=root_id, item_id=resolution_item.item_id,
            field_id=field_id, plan_output_id=root_id,
            proof_ref=resolution_item.field_proofs[int(field_id.removeprefix("field-")) - 1].proof_ref,
        )
        for field_id, root_id in bindings
    ]
    support_roots = [
        ExecutionSupportRoot(
            support_id=support.support_id, kind=support.kind,
            item_id=resolution_item.item_id,
            field_id=support.applies_to_field_ids[0],
            root_id=support.answer_root_refs[0],
        )
        for support in supports
    ]
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan,
        applied_defaults=list(resolution_item.applied_defaults),
        answer_roots=answer_roots,
        support_roots=support_roots,
        premise_roots=([] if comparison.verification_claim is None else [
            ExecutionPremiseRoot(
                premise_id=comparison.verification_premise_id,
                root_ids=["output-3"],
                task_refs=["task-1"],
            )
        ]),
    ), supports


def _build_nary_financial_ranking_typed_plan(
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        intent_item: Any,
        ) -> tuple[ExecutionPlan, list[SupportRequirement]]:
    """Lower 3~8 canonical companies with public N-ary argmax unchanged."""

    comparison = resolution_item.resolution
    if (not isinstance(comparison, FinancialComparisonResolution)
            or len(comparison.operands) not in {3, 4, 5, 6, 7, 8}
            or comparison.requested_operators != ["argmax"]
            or comparison.verification_claim is not None
            or comparison.verification_premise_id is not None):
        raise DeterministicPlanCompilerError(
            "N-ary financial ranking authority가 닫히지 않았습니다")
    field_indexes = _nary_ranking_field_indexes(intent_item)
    if field_indexes is None:
        raise DeterministicPlanCompilerError(
            "N-ary financial ranking 필드 결속이 모호합니다")
    rank_index, values_index = field_indexes
    operands = comparison.operands
    fact_output_ids = [f"output-{index}"
                       for index in range(1, len(operands) + 1)]
    rank_output_id = f"output-{len(operands) + 1}"
    facts = [FactSpec(
        output_id=output_id,
        corp_code=operand.corp_code,
        corp_name=operand.corp_name,
        concept=operand.concept,
        period_start=operand.period_start,
        period_end=operand.period_end,
        period_type=operand.period_type,
        cumulative=operand.cumulative,
        scope=operand.scope,
        statement=operand.statement,
    ) for output_id, operand in zip(fact_output_ids, operands, strict=True)]
    from agent.contracts import Derivation, OutputRef
    resolved_plan = ResolvedQueryPlan(
        revision=0,
        reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=[ResolvedFinancialTask(
            task_id="task-1", as_of=operands[0].as_of,
            view=operands[0].view, facts=facts,
        )],
        derivations=[Derivation(
            output_id=rank_output_id, operator="argmax",
            operands=[OutputRef(output_id=output_id)
                      for output_id in fact_output_ids],
            direction=comparison.direction,
        )],
        premise_claims=[], applied_defaults=[], presentation=None,
    )
    rank_field_id = f"field-{rank_index + 1}"
    values_field_id = f"field-{values_index + 1}"
    rank_proof = resolution_item.field_proofs[rank_index].proof_ref
    values_proof = resolution_item.field_proofs[values_index].proof_ref
    answer_roots = [ExecutionAnswerRoot(
        root_id=rank_output_id, item_id=resolution_item.item_id,
        field_id=rank_field_id, plan_output_id=rank_output_id,
        proof_ref=rank_proof,
    ), *[
        ExecutionAnswerRoot(
            root_id=output_id, item_id=resolution_item.item_id,
            field_id=values_field_id, plan_output_id=output_id,
            proof_ref=values_proof,
        )
        for output_id in fact_output_ids
    ]]
    supports: list[SupportRequirement] = []
    for index, (kind, detail) in enumerate(
            _financial_comparison_support_specs(comparison), start=1):
        supports.append(SupportRequirement(
            support_id=f"support-{index}", kind=kind, detail=detail,
            applies_to_field_ids=[rank_field_id],
            answer_root_refs=[rank_output_id],
        ))
    for index, (output_id, operand) in enumerate(
            zip(fact_output_ids, operands, strict=True),
            start=len(supports) + 1):
        supports.append(SupportRequirement(
            support_id=f"support-{index}", kind="evidence",
            detail=(f"operand={operand.operand_id};"
                    f"proof_ref={operand.proof_ref}"),
            applies_to_field_ids=[values_field_id],
            answer_root_refs=[output_id],
        ))
    support_roots = [ExecutionSupportRoot(
        support_id=support.support_id, kind=support.kind,
        item_id=resolution_item.item_id,
        field_id=support.applies_to_field_ids[0],
        root_id=support.answer_root_refs[0],
    ) for support in supports]
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan,
        applied_defaults=list(resolution_item.applied_defaults),
        answer_roots=answer_roots,
        support_roots=support_roots,
        premise_roots=[],
    ), supports


def _build_financial_retrieve_comparison_typed_plan(
        resolution: AuthoritativeResolution,
        intent_items: tuple[Any, Any],
        ) -> tuple[
            ExecutionPlan, list[SupportRequirement], list[SupportRequirement]]:
    """Reuse one comparison task for an exposed first operand and derivation."""

    retrieve_item, comparison_item = intent_items
    retrieve_resolution, comparison_resolution = resolution.items
    comparison_plan, _ = _build_financial_comparison_typed_plan(
        resolution, comparison_resolution, comparison_item)
    coordinate = retrieve_resolution.resolution
    comparison = comparison_resolution.resolution
    assert isinstance(coordinate, FinancialResolution)
    assert isinstance(comparison, FinancialComparisonResolution)
    derivation_root_id = "output-3"
    retrieve_specs = (
        ("unit", "source-reported unit"),
        ("scope", f"scope={coordinate.scope}"),
        ("period", _financial_period_support_detail(coordinate)),
        ("evidence", f"proof_ref={retrieve_resolution.field_proofs[0].proof_ref}"),
    )
    comparison_specs = _financial_comparison_support_specs(comparison)
    retrieve_supports = [SupportRequirement(
        support_id=f"support-{index}", kind=kind, detail=detail,
        applies_to_field_ids=["field-1"], answer_root_refs=["output-1"],
    ) for index, (kind, detail) in enumerate(retrieve_specs, start=1)]
    comparison_supports = [SupportRequirement(
        support_id=f"support-{index}", kind=kind, detail=detail,
        applies_to_field_ids=["field-2"],
        answer_root_refs=[derivation_root_id],
    ) for index, (kind, detail) in enumerate(
        comparison_specs, start=1 + len(retrieve_supports))]
    answer_roots = [
        ExecutionAnswerRoot(
            root_id="output-1", item_id=retrieve_item.item_id,
            field_id="field-1", plan_output_id="output-1",
            proof_ref=retrieve_resolution.field_proofs[0].proof_ref,
        ),
        ExecutionAnswerRoot(
            root_id=derivation_root_id, item_id=comparison_item.item_id,
            field_id="field-2", plan_output_id=derivation_root_id,
            proof_ref=comparison_resolution.field_proofs[0].proof_ref,
        ),
    ]
    supports = [*retrieve_supports, *comparison_supports]
    support_roots = [ExecutionSupportRoot(
        support_id=support.support_id, kind=support.kind,
        item_id=(retrieve_item.item_id if index < len(retrieve_supports)
                 else comparison_item.item_id),
        field_id=support.applies_to_field_ids[0],
        root_id=support.answer_root_refs[0],
    ) for index, support in enumerate(supports)]
    defaults = [
        default
        for item in resolution.items
        for default in item.applied_defaults
    ]
    plan = ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=comparison_plan.resolved_plan,
        applied_defaults=defaults,
        answer_roots=answer_roots,
        support_roots=support_roots,
        premise_roots=[],
    )
    return plan, retrieve_supports, comparison_supports


def _build_parallel_financial_retrieval_typed_plan(
        resolution: AuthoritativeResolution,
        intent_items: tuple[Any, ...],
        ) -> tuple[ExecutionPlan, list[list[SupportRequirement]]]:
    """Lower N proved scalar coordinates as independent answer roots.

    **항목 수에 무관하다.** CFS/SFS 짝(2개)과 요약 구어 fanout(3개 이상)이
    같은 모양을 쓴다 — 어느 쪽이든 파생 없이 사실 하나에 답 뿌리 하나다.
    """

    coordinates = [row.resolution for row in resolution.items]
    if not all(isinstance(row, FinancialResolution) for row in coordinates):
        raise DeterministicPlanCompilerError(
            "parallel financial retrieval resolution이 필요합니다")
    facts = [FactSpec(
        output_id=f"output-{index}", corp_code=coordinate.corp_code,
        corp_name=coordinate.corp_name, concept=coordinate.concept,
        period_start=coordinate.period_start, period_end=coordinate.period_end,
        period_type=coordinate.period_type, cumulative=coordinate.cumulative,
        scope=coordinate.scope, statement=coordinate.statement,
    ) for index, coordinate in enumerate(coordinates, start=1)]
    first = coordinates[0]
    resolved_plan = ResolvedQueryPlan(
        revision=0, reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=[ResolvedFinancialTask(
            task_id="task-1", as_of=first.as_of, view=first.view,
            facts=facts,
        )],
        derivations=[], premise_claims=[], applied_defaults=[],
        presentation=None,
    )
    supports_by_item: list[list[SupportRequirement]] = []
    support_index = 1
    for index, (coordinate, resolved_item) in enumerate(
            zip(coordinates, resolution.items, strict=True), start=1):
        specs = (
            ("unit", "source-reported unit"),
            ("scope", f"scope={coordinate.scope}"),
            ("period", _financial_period_support_detail(coordinate)),
            ("evidence", f"proof_ref={resolved_item.field_proofs[0].proof_ref}"),
        )
        supports = []
        for kind, detail in specs:
            supports.append(SupportRequirement(
                support_id=f"support-{support_index}", kind=kind,
                detail=detail, applies_to_field_ids=[f"field-{index}"],
                answer_root_refs=[f"output-{index}"],
            ))
            support_index += 1
        supports_by_item.append(supports)
    answer_roots = [ExecutionAnswerRoot(
        root_id=f"output-{index}", item_id=intent_item.item_id,
        field_id=f"field-{index}", plan_output_id=f"output-{index}",
        proof_ref=resolution.items[index - 1].field_proofs[0].proof_ref,
    ) for index, intent_item in enumerate(intent_items, start=1)]
    support_roots = [ExecutionSupportRoot(
        support_id=support.support_id, kind=support.kind,
        item_id=intent_items[index].item_id,
        field_id=support.applies_to_field_ids[0],
        root_id=support.answer_root_refs[0],
    ) for index, supports in enumerate(supports_by_item)
      for support in supports]
    defaults = [
        default for resolved_item in resolution.items
        for default in resolved_item.applied_defaults
    ]
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan, applied_defaults=defaults,
        answer_roots=answer_roots, support_roots=support_roots,
        premise_roots=[],
    ), supports_by_item


def _build_recent_periods_fanout_typed_plan(
        resolution: AuthoritativeResolution,
        intent_items: tuple[Any, ...],
        ) -> tuple[ExecutionPlan, list[list[SupportRequirement]]]:
    """Lower a recent-period series with direct quarters and derived Q4s.

    Each answer item still owns exactly one public root.  A direct item points
    to one fact output; a Q4 item points to a
    ``discrete_from_cumulative(FY, 9M)`` output.  Facts are grouped only when
    their ``(as_of, view)`` execution boundary is identical.
    """

    if len(resolution.items) != len(intent_items):
        raise DeterministicPlanCompilerError(
            "recent periods fanout resolution 항목 수가 다릅니다")

    task_keys: list[tuple[str, str]] = []
    facts_by_key: dict[tuple[str, str], list[FactSpec]] = {}
    derivations: list[Derivation] = []
    answer_roots: list[ExecutionAnswerRoot] = []
    supports_by_item: list[list[SupportRequirement]] = []
    output_index = 1
    support_index = 1

    for item_index, (intent_item, resolved_item) in enumerate(
            zip(intent_items, resolution.items, strict=True), start=1):
        typed = resolved_item.resolution
        logical = _recent_period_logical_coordinate(typed)
        if logical is None:
            raise DeterministicPlanCompilerError(
                "recent periods fanout logical period가 닫히지 않았습니다")
        field_id = f"field-{item_index}"

        if isinstance(typed, FinancialResolution):
            fact_output_ids = [f"output-{output_index}"]
            output_index += 1
            root_id = fact_output_ids[0]
            facts_with_keys = [((typed.as_of, typed.view), FactSpec(
                output_id=root_id, corp_code=typed.corp_code,
                corp_name=typed.corp_name, concept=typed.concept,
                period_start=typed.period_start, period_end=typed.period_end,
                period_type=typed.period_type, cumulative=typed.cumulative,
                scope=typed.scope, statement=typed.statement,
            ))]
            support_specs = [
                ("unit", "source-reported unit"),
                ("scope", f"scope={typed.scope}"),
                ("period", _financial_period_support_detail(typed)),
                ("evidence",
                 f"proof_ref={resolved_item.field_proofs[0].proof_ref}"),
            ]
        elif isinstance(typed, FinancialComparisonResolution):
            fact_output_ids = [
                f"output-{output_index}", f"output-{output_index + 1}"]
            output_index += 2
            facts_with_keys = [(
                (operand.as_of, operand.view),
                FactSpec(
                    output_id=output_id, corp_code=operand.corp_code,
                    corp_name=operand.corp_name, concept=operand.concept,
                    period_start=operand.period_start,
                    period_end=operand.period_end,
                    period_type=operand.period_type,
                    cumulative=operand.cumulative, scope=operand.scope,
                    statement=operand.statement,
                ),
            ) for output_id, operand in zip(
                fact_output_ids, typed.operands, strict=True)]
            root_id = f"output-{output_index}"
            output_index += 1
            derivations.append(Derivation(
                output_id=root_id, operator="discrete_from_cumulative",
                operands=[OutputRef(output_id=value)
                          for value in fact_output_ids],
            ))
            support_specs = _financial_comparison_support_specs(typed)
        else:  # guarded by the logical-coordinate projection above
            raise DeterministicPlanCompilerError(
                "recent periods fanout resolution kind가 다릅니다")

        for task_key, fact in facts_with_keys:
            if task_key not in facts_by_key:
                task_keys.append(task_key)
                facts_by_key[task_key] = []
            facts_by_key[task_key].append(fact)

        item_supports: list[SupportRequirement] = []
        for kind, detail in support_specs:
            item_supports.append(SupportRequirement(
                support_id=f"support-{support_index}", kind=kind,
                detail=detail, applies_to_field_ids=[field_id],
                answer_root_refs=[root_id],
            ))
            support_index += 1
        supports_by_item.append(item_supports)
        answer_roots.append(ExecutionAnswerRoot(
            root_id=root_id, item_id=intent_item.item_id,
            field_id=field_id, plan_output_id=root_id,
            proof_ref=resolved_item.field_proofs[0].proof_ref,
        ))

    tasks = [ResolvedFinancialTask(
        task_id=f"task-{index}", as_of=key[0], view=key[1],
        facts=facts_by_key[key],
    ) for index, key in enumerate(task_keys, start=1)]
    resolved_plan = ResolvedQueryPlan(
        revision=0, reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff, tasks=tasks,
        derivations=derivations, premise_claims=[],
        applied_defaults=(
            [RECENT_Q4_FANOUT_APPLIED_DEFAULT] if derivations else []),
        presentation=None,
    )
    support_roots = [ExecutionSupportRoot(
        support_id=support.support_id, kind=support.kind,
        item_id=intent_items[item_index].item_id,
        field_id=support.applies_to_field_ids[0],
        root_id=support.answer_root_refs[0],
    ) for item_index, supports in enumerate(supports_by_item)
      for support in supports]
    defaults = [
        default for resolved_item in resolution.items
        for default in resolved_item.applied_defaults
    ]
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan, applied_defaults=defaults,
        answer_roots=answer_roots, support_roots=support_roots,
        premise_roots=[],
    ), supports_by_item


def _build_parallel_annual_change_typed_plan(
        resolution: AuthoritativeResolution,
        intent_items: tuple[Any, Any],
        ) -> tuple[ExecutionPlan, list[list[SupportRequirement]]]:
    """Lower split amount/rate fields through one shared financial task."""

    first_typed = resolution.items[0].resolution
    assert isinstance(first_typed, FinancialComparisonResolution)
    operands = first_typed.operands
    facts = [FactSpec(
        output_id=f"output-{index}", corp_code=operand.corp_code,
        corp_name=operand.corp_name, concept=operand.concept,
        period_start=operand.period_start, period_end=operand.period_end,
        period_type=operand.period_type, cumulative=operand.cumulative,
        scope=operand.scope, statement=operand.statement,
    ) for index, operand in enumerate(operands, start=1)]
    operators = [
        row.resolution.requested_operators[0]
        for row in resolution.items
        if isinstance(row.resolution, FinancialComparisonResolution)
    ]
    resolved_plan = ResolvedQueryPlan(
        revision=0, reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=[ResolvedFinancialTask(
            task_id="task-1", as_of=operands[0].as_of,
            view=operands[0].view, facts=facts,
        )],
        derivations=[Derivation(
            output_id=f"output-{index + 3}", operator=operator,
            operands=[OutputRef(output_id="output-1"),
                      OutputRef(output_id="output-2")],
        ) for index, operator in enumerate(operators)],
        premise_claims=[], applied_defaults=[], presentation=None,
    )
    supports_by_item: list[list[SupportRequirement]] = []
    support_index = 1
    for index, resolved_item in enumerate(resolution.items):
        typed = resolved_item.resolution
        assert isinstance(typed, FinancialComparisonResolution)
        field_id = f"field-{index + 1}"
        root_id = f"output-{index + 3}"
        supports = []
        for kind, detail in _financial_comparison_support_specs(typed):
            supports.append(SupportRequirement(
                support_id=f"support-{support_index}", kind=kind,
                detail=detail, applies_to_field_ids=[field_id],
                answer_root_refs=[root_id],
            ))
            support_index += 1
        supports_by_item.append(supports)
    supports = [row for rows in supports_by_item for row in rows]
    answer_roots = [ExecutionAnswerRoot(
        root_id=f"output-{index + 3}", item_id=intent_item.item_id,
        field_id=f"field-{index + 1}",
        plan_output_id=f"output-{index + 3}",
        proof_ref=resolution.items[index].field_proofs[0].proof_ref,
    ) for index, intent_item in enumerate(intent_items)]
    support_roots = [ExecutionSupportRoot(
        support_id=support.support_id, kind=support.kind,
        item_id=intent_items[index].item_id,
        field_id=support.applies_to_field_ids[0],
        root_id=support.answer_root_refs[0],
    ) for index, rows in enumerate(supports_by_item) for support in rows]
    defaults = [default for item in resolution.items
                for default in item.applied_defaults]
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan, applied_defaults=defaults,
        answer_roots=answer_roots, support_roots=support_roots,
        premise_roots=[],
    ), supports_by_item


def _document_coordinate_detail(
        coordinate: DocumentFactOperand | DocumentReasonEvidence,
        ) -> str:
    """Stable coordinate-only support detail; never includes extracted values."""
    return (
        f"operand_id={coordinate.operand_id};"
        f"issuer_corp_code={coordinate.issuer_corp_code};"
        f"source_class={coordinate.source_class};"
        f"doc_id={coordinate.doc_id};receipt_no={coordinate.receipt_no};"
        f"path={coordinate.path};locator={coordinate.locator};"
        f"source_file_id={coordinate.source_file_id};"
        f"evidence_id={coordinate.evidence_id}"
    )


def _build_document_fact_typed_plan(
        resolution: AuthoritativeResolution,
        premise_raw_text: str,
        ) -> tuple[ExecutionPlan, list[SupportRequirement]]:
    """Lower generic correction/disclosure coordinates into typed tasks."""
    comparison = resolution.items[0].resolution
    reason = resolution.items[1].resolution
    if not isinstance(comparison, DocumentFactComparisonResolution) \
            or not isinstance(reason, DocumentReasonEvidenceResolution):
        raise DeterministicPlanCompilerError(
            "document fact comparison typed resolution이 필요합니다")
    operands = comparison.operands
    evidence = reason.evidence
    first, second = operands
    selector_1 = DocumentSelector(doc_id=first.doc_id, rcept_no=first.receipt_no)
    selector_2 = DocumentSelector(doc_id=second.doc_id, rcept_no=second.receipt_no)
    resolved_plan = ResolvedQueryPlan(
        revision=0,
        reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=[
            ResolvedCorrectionTask(
                task_id="task-1", operation="diff",
                corp_code=first.issuer_corp_code,
                corp_name=first.issuer_corp_name,
                as_of=resolution.corpus_cutoff,
                document_selector=selector_1, event_selector=None,
                requested_slots=[first.path],
                field_outputs=[FieldOutputSpec(
                    output_id="output-1", slot=first.path, value_kind="money")],
            ),
            ResolvedDisclosureTask(
                task_id="task-2", operation="lookup",
                corp_code=second.issuer_corp_code,
                corp_name=second.issuer_corp_name,
                as_of=resolution.corpus_cutoff,
                document_selector=selector_2,
                requested_slots=[second.path, evidence.path],
                field_outputs=[
                    FieldOutputSpec(
                        output_id="output-2", slot=second.path, value_kind="money"),
                    FieldOutputSpec(
                        output_id="output-5", slot=evidence.path, value_kind="text"),
                ],
            ),
        ],
        derivations=[
            Derivation(
                output_id="output-3", operator="equal",
                operands=[OutputRef(output_id="output-1"),
                          OutputRef(output_id="output-2")],
            ),
            Derivation(
                output_id="output-4", operator="difference",
                operands=[OutputRef(output_id="output-1"),
                          OutputRef(output_id="output-2")],
            ),
        ],
        premise_claims=[PremiseClaim(
            claim_id=resolution.premise_proofs[0].premise_id, kind="comparison",
            raw_text=premise_raw_text,
            verify_with=[
                OutputRef(output_id="output-1"),
                OutputRef(output_id="output-2"),
            ],
            verify_tasks=[
                TaskVerificationRef(task_id="task-1"),
                TaskVerificationRef(task_id="task-2"),
            ],
        )],
        applied_defaults=[], presentation=None,
    )
    support_specs = _document_fact_support_specs(comparison, evidence)
    supports = [SupportRequirement(
        support_id=support_id, kind=kind, detail=detail,
        applies_to_field_ids=[field_id], answer_root_refs=[root_id],
        plan_root_refs=plan_root_ids,
    ) for support_id, kind, detail, _, field_id, root_id, plan_root_ids
        in support_specs]
    answer_roots = [
        ExecutionAnswerRoot(
            root_id="output-1", item_id="item-1", field_id="field-1",
            plan_output_id="output-3",
            proof_ref=resolution.items[0].field_proofs[0].proof_ref,
        ),
        ExecutionAnswerRoot(
            root_id="output-2", item_id="item-2", field_id="field-2",
            plan_output_id="output-5",
            proof_ref=resolution.items[1].field_proofs[0].proof_ref,
        ),
    ]
    plan_value_roots = [
        ExecutionPlanValueRoot(
            plan_root_id="plan-root-1", plan_output_id="output-1",
            value_kind="money", producer_task_id="task-1"),
        ExecutionPlanValueRoot(
            plan_root_id="plan-root-2", plan_output_id="output-2",
            value_kind="money", producer_task_id="task-2"),
        ExecutionPlanValueRoot(
            plan_root_id="plan-root-3", plan_output_id="output-3",
            value_kind="boolean", producer_derivation_output_id="output-3",
            answer_mirror_root_id="output-1"),
        ExecutionPlanValueRoot(
            plan_root_id="plan-root-4", plan_output_id="output-4",
            value_kind="money", producer_derivation_output_id="output-4"),
        ExecutionPlanValueRoot(
            plan_root_id="plan-root-5", plan_output_id="output-5",
            value_kind="text", producer_task_id="task-2",
            answer_mirror_root_id="output-2"),
    ]
    support_roots = [ExecutionSupportRoot(
        support_id=support.support_id, kind=support.kind,
        item_id=item_id,
        field_id=support.applies_to_field_ids[0],
        root_id=support.answer_root_refs[0],
        plan_root_ids=list(support.plan_root_refs),
    ) for support, (_, _, _, item_id, _, _, _) in zip(
        supports, support_specs, strict=True)]
    premise_roots = [ExecutionPremiseRoot(
        premise_id=resolution.premise_proofs[0].premise_id, root_ids=[],
        plan_root_ids=["plan-root-1", "plan-root-2"],
        task_refs=["task-1", "task-2"],
    )]
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan, applied_defaults=[],
        answer_roots=answer_roots, support_roots=support_roots,
        premise_roots=premise_roots, plan_value_roots=plan_value_roots,
    ), supports


def _build_document_attribute_typed_plan(
        resolution: AuthoritativeResolution,
        ) -> tuple[ExecutionPlan, list[SupportRequirement]]:
    """Lower two generic attribute coordinates to independent lookups."""
    evidence_rows: list[DocumentAttributeEvidence] = []
    for row in resolution.items:
        if not isinstance(row.resolution, DocumentAttributeEvidenceResolution):
            raise DeterministicPlanCompilerError(
                "document attribute typed resolution이 필요합니다")
        evidence_rows.append(row.resolution.evidence)

    tasks = [
        ResolvedDisclosureTask(
            task_id=f"task-{index}", operation="lookup",
            corp_code=evidence.issuer_corp_code,
            corp_name=evidence.issuer_corp_name,
            as_of=resolution.corpus_cutoff,
            document_selector=DocumentSelector(
                doc_id=evidence.doc_id, rcept_no=evidence.receipt_no),
            event_selector=None,
            requested_slots=[evidence.path],
            field_outputs=[FieldOutputSpec(
                output_id=f"output-{index}", slot=evidence.path,
                value_kind="text")],
        )
        for index, evidence in enumerate(evidence_rows, start=1)
    ]
    resolved_plan = ResolvedQueryPlan(
        revision=0,
        reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=tasks, derivations=[], premise_claims=[], applied_defaults=[],
        presentation=None,
    )
    supports = [
        SupportRequirement(
            support_id=f"support-{index}", kind="coordinate",
            detail=_document_attribute_coordinate_detail(evidence),
            applies_to_field_ids=[f"field-{index}"],
            answer_root_refs=[f"output-{index}"],
            plan_root_refs=[f"plan-root-{index}"],
        )
        for index, evidence in enumerate(evidence_rows, start=1)
    ]
    answer_roots = [
        ExecutionAnswerRoot(
            root_id=f"output-{index}", item_id=f"item-{index}",
            field_id=f"field-{index}", plan_output_id=f"output-{index}",
            proof_ref=resolution.items[index - 1].field_proofs[0].proof_ref,
        )
        for index in (1, 2)
    ]
    plan_value_roots = [
        ExecutionPlanValueRoot(
            plan_root_id=f"plan-root-{index}",
            plan_output_id=f"output-{index}", value_kind="text",
            producer_task_id=f"task-{index}",
            answer_mirror_root_id=f"output-{index}",
        )
        for index in (1, 2)
    ]
    support_roots = [
        ExecutionSupportRoot(
            support_id=f"support-{index}", kind="coordinate",
            item_id=f"item-{index}", field_id=f"field-{index}",
            root_id=f"output-{index}",
            plan_root_ids=[f"plan-root-{index}"],
        )
        for index in (1, 2)
    ]
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan, applied_defaults=[],
        answer_roots=answer_roots, support_roots=support_roots,
        premise_roots=[], plan_value_roots=plan_value_roots,
    ), supports


def _build_periodic_narrative_comparison_typed_plan(
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        intent_item: Any,
        ) -> tuple[ExecutionPlan, list[SupportRequirement]]:
    """Lower two exact periodic reports without persisting extracted prose."""
    typed = resolution_item.resolution
    if not isinstance(typed, PeriodicNarrativeComparisonResolution):
        raise DeterministicPlanCompilerError(
            "periodic narrative comparison typed resolution이 필요합니다")
    tasks = [
        ResolvedNarrativeTask(
            task_id=f"task-{index}", operation="search",
            corp_codes=[document.issuer_corp_code],
            corp_names=[document.issuer_corp_name],
            as_of=resolution.corpus_cutoff,
            retrieval_query=intent_item.target.surface,
            document_selector=DocumentSelector(
                doc_id=document.doc_id, rcept_no=document.receipt_no),
            periods=[DateRange(
                start=document.period_start, end=document.period_end)],
            requested_slots=[row.path for row in document.evidence],
        )
        for index, document in enumerate(typed.documents, start=1)
    ]
    resolved_plan = ResolvedQueryPlan(
        revision=0, reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff,
        tasks=tasks, derivations=[], premise_claims=[], applied_defaults=[],
        presentation=None,
    )
    answer_roots = []
    for field_index, proof in enumerate(resolution_item.field_proofs):
        for document in typed.documents:
            answer_roots.append(ExecutionAnswerRoot(
                root_id=_periodic_narrative_root_id(
                    field_index, document.source_period_index),
                item_id=resolution_item.item_id,
                field_id=f"field-{field_index + 1}",
                plan_task_id=f"task-{document.source_period_index + 1}",
                proof_ref=(
                    f"{proof.proof_ref}:period-{document.source_period_index}"),
            ))
    support_specs = _periodic_narrative_support_specs(
        typed, len(resolution_item.field_proofs))
    supports = [
        SupportRequirement(
            support_id=support_id, kind="citation", detail=detail,
            applies_to_field_ids=[field_id], answer_root_refs=[root_id],
        )
        for support_id, field_id, root_id, detail in support_specs
    ]
    support_roots = [
        ExecutionSupportRoot(
            support_id=support_id, kind="citation",
            item_id=resolution_item.item_id, field_id=field_id,
            root_id=root_id,
        )
        for support_id, field_id, root_id, _ in support_specs
    ]
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=resolved_plan, applied_defaults=[],
        answer_roots=answer_roots, support_roots=support_roots,
        premise_roots=[], plan_value_roots=[],
    ), supports


def _build_narrative_matrix_typed_plan(
        resolution: AuthoritativeResolution,
        resolution_item: ResolvedItem,
        intent_item: Any,
        ) -> tuple[ExecutionPlan, list[SupportRequirement]]:
    """Lower each grounded company/document/topic coordinate to one task."""
    typed = resolution_item.resolution
    if not isinstance(typed, NarrativeMatrixResolution):
        raise DeterministicPlanCompilerError("narrative matrix typed resolution이 필요합니다")
    field_index = {surface: index for index, surface in enumerate(
        intent_item.output.field_surfaces)}
    units = [
        (cell, topic) for cell in typed.cells for topic in cell.topics
    ]
    if not 2 <= len(units) <= 16 or any(topic not in field_index for _, topic in units):
        raise DeterministicPlanCompilerError("narrative matrix fanout/topic binding이 잘못되었습니다")
    tasks = [ResolvedNarrativeTask(
        task_id=f"task-{index}", operation="search",
        corp_codes=[cell.corp_code], corp_names=[cell.corp_name],
        as_of=resolution.corpus_cutoff, retrieval_query=topic,
        document_selector=DocumentSelector(doc_id=cell.doc_id, rcept_no=cell.receipt_no),
        periods=[DateRange(start=cell.period_start, end=cell.period_end)],
        # Topic is a retrieval coordinate, not a literal source-table header.
        # Field/root coverage is closed by the compiler's per-task citation
        # binding below; asking Stage2 to find the question wording verbatim
        # would manufacture a false slot_not_confirmed limitation.
        requested_slots=[],
    ) for index, (cell, topic) in enumerate(units, start=1)]
    resolved_plan = ResolvedQueryPlan(
        revision=0, reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff, tasks=tasks,
        derivations=[], premise_claims=[], applied_defaults=[], presentation=None)
    answer_roots, supports, support_roots = [], [], []
    for task_index, (cell, topic) in enumerate(units, start=1):
        index = field_index[topic]
        root_id = f"output-{task_index}"
        proof = resolution_item.field_proofs[index].proof_ref
        answer_roots.append(ExecutionAnswerRoot(
            root_id=root_id, item_id=resolution_item.item_id,
            field_id=f"field-{index + 1}", plan_task_id=f"task-{task_index}",
            proof_ref=f"{proof}:{cell.cell_id}"))
        support_id = f"support-{task_index}"
        detail = (
            f"corp_code={cell.corp_code};doc_id={cell.doc_id};"
            f"receipt_no={cell.receipt_no};period={cell.period_start}:{cell.period_end};"
            f"topic={topic};proof={cell.document_proof.proof_ref}")
        supports.append(SupportRequirement(
            support_id=support_id, kind="citation", detail=detail,
            applies_to_field_ids=[f"field-{index + 1}"], answer_root_refs=[root_id]))
        support_roots.append(ExecutionSupportRoot(
            support_id=support_id, kind="citation", item_id=resolution_item.item_id,
            field_id=f"field-{index + 1}", root_id=root_id))
    return ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version, resolved_plan=resolved_plan,
        applied_defaults=[], answer_roots=answer_roots, support_roots=support_roots,
        premise_roots=[], plan_value_roots=[]), supports


def _compile_financial(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    """Lower the two bounded single-fact financial registrations."""
    defaults = list(resolution_item.applied_defaults)
    plan, supports = _build_typed_plan(
        resolution, resolution_item, defaults)
    source_digest = semantic_intent_digest(intent)
    authority = _source_authority(intent)
    authority["source_field_bindings"] = {
        intent_item.item_id: [SourceFieldBinding(
            surface=resolution_item.field_proofs[0].surface, field_id="field-1",
        )]
    }
    compiled_item = CompiledAnswerItem(
        item_id=intent_item.item_id,
        status="ready",
        projection=AnswerProjection(
            shape=intent_item.output.shape,
            fields=[ProjectionField(
                field_id="field-1", field_key="value",
                binding_status="executable", answer_root_refs=["output-1"],
            )],
            presentation=intent_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=["field-1"], executable=["field-1"], limited=[]),
        support_requirements=supports,
    )
    contract = CompiledAnswerContract.build(
        source_intent=intent,
        source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest,
        items=[compiled_item],
        groups=authority["source_groups"],
        source_field_bindings=authority["source_field_bindings"],
        live_answer_root_refs=["output-1"],
        live_plan_root_refs=[row.plan_root_id for row in plan.plan_value_roots],
        live_task_refs=["task-1"],
        premise_contracts=[], presentation=intent.presentation,
    )
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=source_digest, resolution_digest=resolution.resolution_digest,
        overlay_digest=overlay_digest, intent=intent, resolution=resolution,
        execution_plan=plan,
        answer_contract=contract,
    )


def _compile_premise_financial(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_authority: tuple[Any, tuple[Any, ...]],
        resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    intent_item, premises = intent_authority
    source_digest = semantic_intent_digest(intent)
    plan, supports = _build_premise_financial_typed_plan(
        resolution, resolution_item, premises)
    authority = _source_authority(intent)
    authority["source_field_bindings"] = {
        intent_item.item_id: [SourceFieldBinding(
            surface=resolution_item.field_proofs[0].surface, field_id="field-1",
        )]
    }
    compiled_item = CompiledAnswerItem(
        item_id=intent_item.item_id, status="ready",
        projection=AnswerProjection(
            shape=intent_item.output.shape,
            fields=[ProjectionField(
                field_id="field-1", field_key="value",
                binding_status="executable", answer_root_refs=["output-1"],
            )], presentation=intent_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=["field-1"], executable=["field-1"], limited=[]),
        support_requirements=supports,
    )
    premise_contracts = [CompiledPremiseContract(
        premise_id=premise.premise_id, kind=premise.kind,
        raw_text=premise.raw_text,
        applies_to_item_ids=list(premise.applies_to_item_ids),
        verification_root_refs=["output-1"],
        verification_task_refs=["task-1"],
        verification_plan_root_refs=["plan-root-1"],
        verdict_requirement="required",
    ) for premise in premises]
    contract = CompiledAnswerContract.build(
        source_intent=intent, source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest,
        items=[compiled_item], groups=authority["source_groups"],
        source_field_bindings=authority["source_field_bindings"],
        live_answer_root_refs=["output-1"],
        live_plan_root_refs=["plan-root-1"], live_task_refs=["task-1"],
        premise_contracts=premise_contracts, presentation=intent.presentation,
    )
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=source_digest,
        resolution_digest=resolution.resolution_digest, overlay_digest=None,
        intent=intent, resolution=resolution, execution_plan=plan,
        answer_contract=contract,
    )


def _compile_same_day_status(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    if not isinstance(intent_item, tuple) or len(intent_item) != 2:
        raise DeterministicPlanCompilerError(
            "same-day status compiler에는 intent item pair가 필요합니다")
    document_item, event_item = intent_item
    if not isinstance(resolution_item.resolution,
                      SameDayDocumentCandidatesResolution):
        raise DeterministicPlanCompilerError(
            "same-day document resolution branch가 다릅니다")
    status_item = resolution.items[1]
    if not isinstance(status_item.resolution, TerminationReportedStatusResolution):
        raise DeterministicPlanCompilerError(
            "same-day event resolution branch가 다릅니다")
    source_digest = semantic_intent_digest(intent)
    plan, supports = _build_same_day_status_typed_plan(
        resolution, intent_item)
    authority = _source_authority(intent)
    authority["source_field_bindings"] = {
        document_item.item_id: [SourceFieldBinding(
            surface=document_item.output.field_surfaces[0],
            field_id="field-1")],
        event_item.item_id: [SourceFieldBinding(
            surface=event_item.output.field_surfaces[0],
            field_id="field-2")],
    }
    item1 = CompiledAnswerItem(
        item_id=document_item.item_id, status="partial",
        projection=AnswerProjection(
            shape=document_item.output.shape,
            fields=[ProjectionField(
                field_id="field-1", field_key="contents",
                binding_status="qualified", answer_root_refs=["output-1", "output-2"],
                limitation_refs=["limitation-1"],
            )], presentation=document_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=["field-1"], executable=["field-1"], limited=["field-1"]),
        support_requirements=supports[:2],
        limitation_bindings=[LimitationBinding(
            limitation_id="limitation-1", code="intraday_order_unavailable",
            family="ordering", applies_to_field_ids=["field-1"],
            detail=resolution_item.resolution.ordering_provenance.detail,
        )],
    )
    item2 = CompiledAnswerItem(
        item_id=event_item.item_id, status="partial",
        projection=AnswerProjection(
            shape=event_item.output.shape,
            fields=[ProjectionField(
                field_id="field-2", field_key="final_status",
                binding_status="qualified", answer_root_refs=["output-3"],
                limitation_refs=["limitation-2"],
            )], presentation=event_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=["field-2"], executable=["field-2"], limited=["field-2"]),
        support_requirements=supports[2:],
        limitation_bindings=[LimitationBinding(
            limitation_id="limitation-2", code="ambiguous_event_origin",
            family="identity_lineage", applies_to_field_ids=["field-2"],
            detail=status_item.resolution.identity_provenance.detail,
        )],
    )
    contract = CompiledAnswerContract.build(
        source_intent=intent, source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest,
        items=[item1, item2], groups=authority["source_groups"],
        source_field_bindings=authority["source_field_bindings"],
        live_answer_root_refs=[row.root_id for row in plan.answer_roots],
        live_plan_root_refs=[row.plan_root_id for row in plan.plan_value_roots],
        live_task_refs=[row.task_id for row in plan.resolved_plan.tasks],
        premise_contracts=[], presentation=intent.presentation,
    )
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=source_digest,
        resolution_digest=resolution.resolution_digest,
        overlay_digest=None, intent=intent, resolution=resolution,
        execution_plan=plan, answer_contract=contract,
    )


def _build_nary_financial_ranking_contract(
        intent: SemanticIntent,
        intent_item: Any,
        plan: ExecutionPlan,
        supports: list[SupportRequirement],
        ) -> CompiledAnswerContract:
    """Bind ranking plus every raw operand to source fields in source order."""

    field_indexes = _nary_ranking_field_indexes(intent_item)
    if field_indexes is None:
        raise DeterministicPlanCompilerError(
            "N-ary financial ranking source field binding이 모호합니다")
    rank_index, values_index = field_indexes
    task = plan.resolved_plan.tasks[0]
    fact_output_ids = [fact.output_id for fact in task.facts]
    rank_output_id = plan.resolved_plan.derivations[0].output_id
    fields: list[ProjectionField] = []
    for index in range(len(intent_item.output.field_surfaces)):
        refs: list[str] = []
        keys: list[str] = []
        if index == rank_index:
            refs.append(rank_output_id)
            keys.append("ranking")
        if index == values_index:
            refs.extend(fact_output_ids)
            keys.append("values")
        fields.append(ProjectionField(
            field_id=f"field-{index + 1}",
            field_key="_and_".join(keys),
            binding_status="executable",
            answer_root_refs=refs,
        ))
    field_ids = [field.field_id for field in fields]
    compiled_item = CompiledAnswerItem(
        item_id=intent_item.item_id,
        status="ready",
        projection=AnswerProjection(
            shape=intent_item.output.shape,
            fields=fields,
            presentation=intent_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=field_ids, executable=field_ids, limited=[]),
        support_requirements=supports,
    )
    return CompiledAnswerContract.build(
        source_intent=intent,
        source_intent_digest=semantic_intent_digest(intent),
        execution_plan_digest=plan.execution_plan_digest,
        items=[compiled_item], groups=[],
        source_field_bindings={intent_item.item_id: [
            SourceFieldBinding(
                surface=surface, field_id=f"field-{index + 1}")
            for index, surface in enumerate(intent_item.output.field_surfaces)
        ]},
        live_answer_root_refs=[root.root_id for root in plan.answer_roots],
        live_plan_root_refs=[], live_task_refs=["task-1"],
        premise_contracts=[], presentation=intent.presentation,
    )


def _compile_financial_comparison(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    comparison = resolution_item.resolution
    if not isinstance(comparison, FinancialComparisonResolution):
        raise DeterministicPlanCompilerError(
            "financial_comparison resolution이 필요합니다")
    if len(comparison.operands) > 2:
        if intent.premises:
            raise DeterministicPlanCompilerError(
                "N-ary financial ranking source에 premise가 남았습니다")
        source_digest = semantic_intent_digest(intent)
        plan, supports = _build_nary_financial_ranking_typed_plan(
            resolution, resolution_item, intent_item)
        contract = _build_nary_financial_ranking_contract(
            intent, intent_item, plan, supports)
        return DeterministicCompiledSlice.create(
            question_id=resolution.question_id, question=question,
            source_intent_digest=source_digest,
            resolution_digest=resolution.resolution_digest,
            overlay_digest=None, intent=intent, resolution=resolution,
            execution_plan=plan, answer_contract=contract,
        )
    if comparison.verification_claim is None:
        if intent.premises:
            raise DeterministicPlanCompilerError(
                "ordinary financial comparison source에 premise가 남았습니다")
        premise_contracts: list[CompiledPremiseContract] = []
    else:
        if (
                len(intent.premises) != 1
                or intent.premises[0].premise_id
                != comparison.verification_premise_id
                or intent.premises[0].kind != "comparison"
                or intent.premises[0].raw_text != comparison.verification_claim
                or intent.premises[0].applies_to_item_ids != [intent_item.item_id]
        ):
            raise DeterministicPlanCompilerError(
                "financial verification resolution이 source premise와 다릅니다")
        premise = intent.premises[0]
        premise_contracts = [CompiledPremiseContract(
            premise_id=premise.premise_id, kind=premise.kind,
            raw_text=premise.raw_text,
            applies_to_item_ids=list(premise.applies_to_item_ids),
            verification_root_refs=["output-3"],
            verification_task_refs=["task-1"],
            verification_plan_root_refs=[],
            verdict_requirement="required",
        )]
    source_digest = semantic_intent_digest(intent)
    plan, supports = _build_financial_comparison_typed_plan(
        resolution, resolution_item, intent_item)
    authority = _source_authority(intent)
    # **필드 표면은 질문에서 온다.** 예전에는 `"큰 기업"`·`"차이"` 를 박아 두어
    # 그 두 낱말을 쓴 질문만 통과했다. intent 가 이미 순서대로 들고 있다.
    authority["source_field_bindings"] = {
        intent_item.item_id: [
            SourceFieldBinding(surface=surface, field_id=f"field-{position}")
            for position, surface in enumerate(
                intent_item.output.field_surfaces, start=1)
        ]
    }
    same_company = comparison.operands[0].corp_code == comparison.operands[1].corp_code
    # 이슈 #59 1단계 — 회사 간 concept_ratio 도 same-company 분기의 "A÷B
    # 몇 배" 문장 투영(scalar 한 필드, field_key=operator)을 그대로 쓴다.
    # 승자/차이(comparison shape) 분기는 argmax류 전용으로 남긴다.
    # 이슈 #124 (합계) — 회사 간 sum도 같은 이유로 같은 scalar 투영을 쓴다.
    if same_company or comparison.requested_operators in (
            ["concept_ratio"], ["sum"]):
        root_ids = [row.root_id for row in plan.answer_roots]
        view_relation = comparison.operands[0].view != comparison.operands[1].view
        if view_relation:
            fields = [
                ProjectionField(
                    field_id="field-1", field_key="as_filed",
                    binding_status="executable", answer_root_refs=["output-1"]),
                ProjectionField(
                    field_id="field-2", field_key="restated",
                    binding_status="executable", answer_root_refs=["output-2"]),
                ProjectionField(
                    field_id="field-3", field_key="difference",
                    binding_status="executable", answer_root_refs=["output-3"]),
            ]
        elif len(intent_item.output.field_surfaces) == 1:
            fields = [ProjectionField(
                field_id="field-1", field_key="change",
                binding_status="executable", answer_root_refs=root_ids,
            )]
        else:
            fields = [ProjectionField(
                field_id=f"field-{index + 1}", field_key=operator,
                binding_status="executable",
                answer_root_refs=[f"output-{index + 3}"],
            ) for index, operator in enumerate(comparison.requested_operators)]
        field_ids = [field.field_id for field in fields]
        compiled_item = CompiledAnswerItem(
            item_id=intent_item.item_id,
            status="ready",
            projection=AnswerProjection(
                shape=intent_item.output.shape, fields=fields,
                presentation=intent_item.output.presentation,
            ),
            coverage=CoveragePartition(
                required=field_ids, executable=field_ids, limited=[]),
            support_requirements=supports,
        )
    else:
        field_keys = [
            "winner" if operator == "argmax" else "difference"
            for operator in comparison.requested_operators]
        field_ids = [
            f"field-{index}" for index in range(1, len(field_keys) + 1)]
        compiled_item = CompiledAnswerItem(
        item_id=intent_item.item_id,
        status="ready",
        projection=AnswerProjection(
            shape="comparison",
            fields=[
                ProjectionField(
                    field_id=field_id, field_key=field_key,
                    binding_status="executable",
                    answer_root_refs=[f"output-{index + 3}"],
                )
                for index, (field_id, field_key) in enumerate(
                    zip(field_ids, field_keys, strict=True))
            ],
            presentation=intent_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=field_ids,
            executable=field_ids, limited=[],
        ),
        support_requirements=supports,
        )
    contract = CompiledAnswerContract.build(
        source_intent=intent,
        source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest,
        items=[compiled_item], groups=authority["source_groups"],
        source_field_bindings=authority["source_field_bindings"],
        live_answer_root_refs=[row.root_id for row in plan.answer_roots],
        live_plan_root_refs=[row.plan_root_id for row in plan.plan_value_roots],
        live_task_refs=["task-1"], premise_contracts=premise_contracts,
        presentation=intent.presentation,
    )
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=source_digest,
        resolution_digest=resolution.resolution_digest,
        overlay_digest=None, intent=intent, resolution=resolution,
        execution_plan=plan, answer_contract=contract,
    )


def _build_financial_retrieve_comparison_contract(
        intent: SemanticIntent,
        intent_items: tuple[Any, Any],
        plan: ExecutionPlan,
        retrieve_supports: list[SupportRequirement],
        comparison_supports: list[SupportRequirement],
        ) -> CompiledAnswerContract:
    retrieve_item, comparison_item = intent_items
    comparison = plan.resolved_plan.derivations[0]
    source_bindings = {
        retrieve_item.item_id: [SourceFieldBinding(
            surface=retrieve_item.output.field_surfaces[0], field_id="field-1")],
        comparison_item.item_id: [SourceFieldBinding(
            surface=comparison_item.output.field_surfaces[0], field_id="field-2")],
    }
    items = [
        CompiledAnswerItem(
            item_id=retrieve_item.item_id, status="ready",
            projection=AnswerProjection(
                shape=retrieve_item.output.shape,
                fields=[ProjectionField(
                    field_id="field-1", field_key="value",
                    binding_status="executable", answer_root_refs=["output-1"],
                )],
                presentation=retrieve_item.output.presentation,
            ),
            coverage=CoveragePartition(
                required=["field-1"], executable=["field-1"], limited=[]),
            support_requirements=retrieve_supports,
        ),
        CompiledAnswerItem(
            item_id=comparison_item.item_id, status="ready",
            projection=AnswerProjection(
                shape=comparison_item.output.shape,
                fields=[ProjectionField(
                    field_id="field-2", field_key=comparison.operator,
                    binding_status="executable", answer_root_refs=["output-3"],
                )],
                presentation=comparison_item.output.presentation,
            ),
            coverage=CoveragePartition(
                required=["field-2"], executable=["field-2"], limited=[]),
            support_requirements=comparison_supports,
        ),
    ]
    authority = _source_authority(intent)
    return CompiledAnswerContract.build(
        source_intent=intent,
        source_intent_digest=semantic_intent_digest(intent),
        execution_plan_digest=plan.execution_plan_digest,
        items=items,
        groups=authority["source_groups"],
        source_field_bindings=source_bindings,
        live_answer_root_refs=[row.root_id for row in plan.answer_roots],
        live_plan_root_refs=[
            row.plan_root_id for row in plan.plan_value_roots],
        live_task_refs=[row.task_id for row in plan.resolved_plan.tasks],
        premise_contracts=[],
        presentation=intent.presentation,
    )


def _compile_financial_retrieve_comparison(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_items: tuple[Any, Any],
        ) -> DeterministicCompiledSlice:
    plan, retrieve_supports, comparison_supports = (
        _build_financial_retrieve_comparison_typed_plan(
            resolution, intent_items))
    contract = _build_financial_retrieve_comparison_contract(
        intent, intent_items, plan, retrieve_supports, comparison_supports)
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id,
        question=question,
        source_intent_digest=semantic_intent_digest(intent),
        resolution_digest=resolution.resolution_digest,
        overlay_digest=None,
        intent=intent,
        resolution=resolution,
        execution_plan=plan,
        answer_contract=contract,
    )


def _compile_parallel_financial_retrieval(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_items: tuple[Any, ...],
        ) -> DeterministicCompiledSlice:
    plan, supports_by_item = _build_parallel_financial_retrieval_typed_plan(
        resolution, intent_items)
    compiled_items = []
    source_bindings: dict[str, list[SourceFieldBinding]] = {}
    for index, intent_item in enumerate(intent_items, start=1):
        field_id = f"field-{index}"
        root_id = f"output-{index}"
        compiled_items.append(CompiledAnswerItem(
            item_id=intent_item.item_id, status="ready",
            projection=AnswerProjection(
                shape=intent_item.output.shape,
                fields=[ProjectionField(
                    field_id=field_id, field_key="value",
                    binding_status="executable", answer_root_refs=[root_id],
                )],
                presentation=intent_item.output.presentation,
            ),
            coverage=CoveragePartition(
                required=[field_id], executable=[field_id], limited=[]),
            support_requirements=supports_by_item[index - 1],
        ))
        source_bindings[intent_item.item_id] = [SourceFieldBinding(
            surface=intent_item.output.field_surfaces[0], field_id=field_id)]
    contract = CompiledAnswerContract.build(
        source_intent=intent,
        source_intent_digest=semantic_intent_digest(intent),
        execution_plan_digest=plan.execution_plan_digest,
        items=compiled_items, groups=[], source_field_bindings=source_bindings,
        live_answer_root_refs=[row.root_id for row in plan.answer_roots],
        live_plan_root_refs=[], live_task_refs=["task-1"],
        premise_contracts=[], presentation=intent.presentation,
    )
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=semantic_intent_digest(intent),
        resolution_digest=resolution.resolution_digest,
        overlay_digest=None, intent=intent, resolution=resolution,
        execution_plan=plan, answer_contract=contract,
    )


def _build_recent_periods_fanout_contract(
        intent: SemanticIntent, intent_items: tuple[Any, ...],
        plan: ExecutionPlan,
        supports_by_item: list[list[SupportRequirement]],
        ) -> CompiledAnswerContract:
    """Bind each direct or derived period root to one visible table field."""

    if len(plan.answer_roots) != len(intent_items):
        raise DeterministicPlanCompilerError(
            "recent periods fanout answer root 수가 다릅니다")
    compiled_items: list[CompiledAnswerItem] = []
    source_bindings: dict[str, list[SourceFieldBinding]] = {}
    for index, (intent_item, root, supports) in enumerate(zip(
            intent_items, plan.answer_roots, supports_by_item, strict=True),
            start=1):
        field_id = f"field-{index}"
        compiled_items.append(CompiledAnswerItem(
            item_id=intent_item.item_id, status="ready",
            projection=AnswerProjection(
                shape=intent_item.output.shape,
                fields=[ProjectionField(
                    field_id=field_id, field_key="value",
                    binding_status="executable",
                    answer_root_refs=[root.root_id],
                )],
                presentation=intent_item.output.presentation,
            ),
            coverage=CoveragePartition(
                required=[field_id], executable=[field_id], limited=[]),
            support_requirements=supports,
        ))
        source_bindings[intent_item.item_id] = [SourceFieldBinding(
            surface=intent_item.output.field_surfaces[0], field_id=field_id)]
    return CompiledAnswerContract.build(
        source_intent=intent,
        source_intent_digest=semantic_intent_digest(intent),
        execution_plan_digest=plan.execution_plan_digest,
        items=compiled_items, groups=[], source_field_bindings=source_bindings,
        live_answer_root_refs=[row.root_id for row in plan.answer_roots],
        live_plan_root_refs=[],
        live_task_refs=[row.task_id for row in plan.resolved_plan.tasks],
        premise_contracts=[], presentation=intent.presentation,
    )


def _compile_recent_periods_fanout(
        question: str, intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_items: tuple[Any, ...],
        ) -> DeterministicCompiledSlice:
    plan, supports_by_item = _build_recent_periods_fanout_typed_plan(
        resolution, intent_items)
    contract = _build_recent_periods_fanout_contract(
        intent, intent_items, plan, supports_by_item)
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=semantic_intent_digest(intent),
        resolution_digest=resolution.resolution_digest,
        overlay_digest=None, intent=intent, resolution=resolution,
        execution_plan=plan, answer_contract=contract,
    )


def _build_parallel_annual_change_contract(
        intent: SemanticIntent, intent_items: tuple[Any, Any],
        plan: ExecutionPlan,
        supports_by_item: list[list[SupportRequirement]],
        ) -> CompiledAnswerContract:
    items: list[CompiledAnswerItem] = []
    source_bindings: dict[str, list[SourceFieldBinding]] = {}
    for index, intent_item in enumerate(intent_items):
        field_id = f"field-{index + 1}"
        root_id = f"output-{index + 3}"
        operator = plan.resolved_plan.derivations[index].operator
        items.append(CompiledAnswerItem(
            item_id=intent_item.item_id, status="ready",
            projection=AnswerProjection(
                shape=intent_item.output.shape,
                fields=[ProjectionField(
                    field_id=field_id, field_key=operator,
                    binding_status="executable", answer_root_refs=[root_id],
                )],
                presentation=intent_item.output.presentation,
            ),
            coverage=CoveragePartition(
                required=[field_id], executable=[field_id], limited=[]),
            support_requirements=supports_by_item[index],
        ))
        source_bindings[intent_item.item_id] = [SourceFieldBinding(
            surface=intent_item.output.field_surfaces[0], field_id=field_id)]
    return CompiledAnswerContract.build(
        source_intent=intent,
        source_intent_digest=semantic_intent_digest(intent),
        execution_plan_digest=plan.execution_plan_digest,
        items=items, groups=[], source_field_bindings=source_bindings,
        live_answer_root_refs=[row.root_id for row in plan.answer_roots],
        live_plan_root_refs=[], live_task_refs=["task-1"],
        premise_contracts=[], presentation=intent.presentation,
    )


def _compile_parallel_annual_change(
        question: str, intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_items: tuple[Any, Any],
        ) -> DeterministicCompiledSlice:
    plan, supports_by_item = _build_parallel_annual_change_typed_plan(
        resolution, intent_items)
    contract = _build_parallel_annual_change_contract(
        intent, intent_items, plan, supports_by_item)
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=semantic_intent_digest(intent),
        resolution_digest=resolution.resolution_digest,
        overlay_digest=None, intent=intent, resolution=resolution,
        execution_plan=plan, answer_contract=contract,
    )


def _compile_document_fact_comparison(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_items: tuple[Any, Any],
        resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    first_item, second_item = intent_items
    source_digest = semantic_intent_digest(intent)
    plan, supports = _build_document_fact_typed_plan(
        resolution, intent.premises[0].raw_text)
    authority = _source_authority(intent)
    authority["source_field_bindings"] = {
        first_item.item_id: [SourceFieldBinding(
            surface=first_item.output.field_surfaces[0], field_id="field-1")],
        second_item.item_id: [SourceFieldBinding(
            surface=second_item.output.field_surfaces[0], field_id="field-2")],
    }
    item1 = CompiledAnswerItem(
        item_id=first_item.item_id, status="ready",
        projection=AnswerProjection(
            shape="verdict",
            fields=[ProjectionField(
                field_id="field-1", field_key="verdict",
                binding_status="executable", answer_root_refs=["output-1"],
            )], presentation=first_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=["field-1"], executable=["field-1"], limited=[]),
        support_requirements=supports[:4],
    )
    item2 = CompiledAnswerItem(
        item_id=second_item.item_id, status="ready",
        projection=AnswerProjection(
            shape="narrative",
            fields=[ProjectionField(
                field_id="field-2", field_key="reason",
                binding_status="executable", answer_root_refs=["output-2"],
                activation_predicate=ActivationPredicate(
                    predicate_answer_root_ref="output-1",
                    expected_boolean=False,
                ),
            )], presentation=second_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=["field-2"], executable=["field-2"], limited=[]),
        support_requirements=supports[4:],
    )
    premise = intent.premises[0]
    premise_contract = CompiledPremiseContract(
        premise_id=premise.premise_id, kind=premise.kind,
        raw_text=premise.raw_text,
        applies_to_item_ids=list(premise.applies_to_item_ids),
        verification_root_refs=[],
        verification_task_refs=["task-1", "task-2"],
        verification_plan_root_refs=["plan-root-1", "plan-root-2"],
        verdict_requirement="required",
    )
    contract = CompiledAnswerContract.build(
        source_intent=intent, source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest,
        items=[item1, item2], groups=authority["source_groups"],
        source_field_bindings=authority["source_field_bindings"],
        live_answer_root_refs=["output-1", "output-2"],
        live_plan_root_refs=[row.plan_root_id for row in plan.plan_value_roots],
        live_task_refs=["task-1", "task-2"],
        premise_contracts=[premise_contract], presentation=intent.presentation,
    )
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=source_digest,
        resolution_digest=resolution.resolution_digest, overlay_digest=None,
        intent=intent, resolution=resolution,
        execution_plan=plan, answer_contract=contract,
    )


def _compile_document_attributes(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_authority: Any,
        resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    if (
            not isinstance(intent_authority, tuple)
            or len(intent_authority) != 2
            or not isinstance(intent_authority[0], tuple)
    ):
        raise DeterministicPlanCompilerError(
            "document attribute compiler intent authority가 다릅니다")
    intent_items, _ = intent_authority
    if resolution_item != resolution.items[0]:
        raise DeterministicPlanCompilerError(
            "document attribute compiler resolution authority가 다릅니다")
    source_digest = semantic_intent_digest(intent)
    plan, supports = _build_document_attribute_typed_plan(resolution)
    authority = _source_authority(intent)
    authority["source_field_bindings"] = {
        item.item_id: [SourceFieldBinding(
            surface=item.output.field_surfaces[0], field_id=f"field-{index}")]
        for index, item in enumerate(intent_items, start=1)
    }
    compiled_items = [
        CompiledAnswerItem(
            item_id=item.item_id, status="ready",
            projection=AnswerProjection(
                shape="scalar",
                fields=[ProjectionField(
                    field_id=f"field-{index}",
                    field_key=f"attribute_{index}",
                    binding_status="executable",
                    answer_root_refs=[f"output-{index}"],
                )],
                presentation=item.output.presentation,
            ),
            coverage=CoveragePartition(
                required=[f"field-{index}"],
                executable=[f"field-{index}"], limited=[]),
            support_requirements=[supports[index - 1]],
        )
        for index, item in enumerate(intent_items, start=1)
    ]
    contract = CompiledAnswerContract.build(
        source_intent=intent, source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest,
        items=compiled_items, groups=authority["source_groups"],
        source_field_bindings=authority["source_field_bindings"],
        live_answer_root_refs=["output-1", "output-2"],
        live_plan_root_refs=["plan-root-1", "plan-root-2"],
        live_task_refs=["task-1", "task-2"],
        premise_contracts=[], presentation=intent.presentation,
    )
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=source_digest,
        resolution_digest=resolution.resolution_digest,
        overlay_digest=None, intent=intent, resolution=resolution,
        execution_plan=plan, answer_contract=contract,
    )


def _compile_periodic_narrative_comparison(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_authority: Any,
        resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    if not isinstance(intent_authority, tuple) or len(intent_authority) != 2:
        raise DeterministicPlanCompilerError(
            "periodic narrative compiler intent authority가 다릅니다")
    intent_item, _ = intent_authority
    if resolution_item != resolution.items[0] or not isinstance(
            resolution_item.resolution,
            PeriodicNarrativeComparisonResolution):
        raise DeterministicPlanCompilerError(
            "periodic narrative compiler resolution authority가 다릅니다")
    source_digest = semantic_intent_digest(intent)
    plan, supports = _build_periodic_narrative_comparison_typed_plan(
        resolution, resolution_item, intent_item)
    authority = _source_authority(intent)
    authority["source_field_bindings"] = {
        intent_item.item_id: [
            SourceFieldBinding(
                surface=surface, field_id=f"field-{index + 1}")
            for index, surface in enumerate(intent_item.output.field_surfaces)
        ]
    }
    field_count = len(intent_item.output.field_surfaces)
    field_ids = [f"field-{index + 1}" for index in range(field_count)]
    fields = []
    for field_index in range(field_count):
        field_key = f"axis_{field_index + 1}"
        fields.append(ProjectionField(
            field_id=f"field-{field_index + 1}", field_key=field_key,
            binding_status="executable",
            answer_root_refs=[
                _periodic_narrative_root_id(field_index, period_index)
                for period_index in range(2)
            ],
        ))
    compiled_item = CompiledAnswerItem(
        item_id=intent_item.item_id, status="ready",
        projection=AnswerProjection(
            shape=intent_item.output.shape, fields=fields,
            presentation=intent_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=field_ids, executable=field_ids, limited=[]),
        support_requirements=supports, limitation_bindings=[],
    )
    contract = CompiledAnswerContract.build(
        source_intent=intent, source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest,
        items=[compiled_item], groups=authority["source_groups"],
        source_field_bindings=authority["source_field_bindings"],
        live_answer_root_refs=[row.root_id for row in plan.answer_roots],
        live_plan_root_refs=[],
        live_task_refs=[row.task_id for row in plan.resolved_plan.tasks],
        premise_contracts=[], presentation=intent.presentation,
    )
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=source_digest,
        resolution_digest=resolution.resolution_digest,
        overlay_digest=None, intent=intent, resolution=resolution,
        execution_plan=plan, answer_contract=contract,
    )


def _validate_narrative_matrix_intent(intent: SemanticIntent) -> tuple[Any, None]:
    if (len(intent.answer_items) != 1 or intent.answer_groups or intent.premises
            or intent.unresolved_mentions):
        raise DeterministicPlanCompilerError("narrative matrix intent inventory가 다릅니다")
    item = intent.answer_items[0]
    if (item.target.kind not in {"document", "topic"}
            or item.operation not in {"retrieve", "compare"}
            or item.output.projection_mode != "named_fields"
            or not item.output.field_surfaces):
        raise DeterministicPlanCompilerError("narrative matrix intent shape가 다릅니다")
    return item, None


def _validate_narrative_matrix_resolution(
        intent_item: Any, resolution: AuthoritativeResolution,
        ) -> tuple[ResolvedItem, None]:
    if len(resolution.items) != 1 or resolution.premise_proofs:
        raise DeterministicPlanCompilerError("narrative matrix resolution inventory가 다릅니다")
    item = resolution.items[0]
    if (item.item_id != intent_item.item_id
            or item.target_surface != intent_item.target.surface
            or item.projection_mode != intent_item.output.projection_mode
            or item.applied_defaults
            or not isinstance(item.resolution, NarrativeMatrixResolution)):
        raise DeterministicPlanCompilerError("narrative matrix resolution authority가 다릅니다")
    expected = [(index, surface, f"source-field:{item.item_id}:{index}")
                for index, surface in enumerate(intent_item.output.field_surfaces)]
    actual = [(row.source_field_index, row.surface, row.proof_ref)
              for row in item.field_proofs]
    if actual != expected:
        raise DeterministicPlanCompilerError("narrative matrix field proofs가 다릅니다")
    expected_topics = set(intent_item.output.field_surfaces)
    if any(set(cell.topics) != expected_topics for cell in item.resolution.cells):
        raise DeterministicPlanCompilerError("narrative matrix cell topics가 answer fields와 다릅니다")
    return item, None


def _compile_narrative_matrix(
        question: str, intent: SemanticIntent, resolution: AuthoritativeResolution,
        intent_item: Any, resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    plan, supports = _build_narrative_matrix_typed_plan(
        resolution, resolution_item, intent_item)
    fields = [ProjectionField(
        field_id=f"field-{index + 1}", field_key=f"topic_{index + 1}",
        binding_status="executable",
        answer_root_refs=[root.root_id for root in plan.answer_roots
                          if root.field_id == f"field-{index + 1}"],
    ) for index in range(len(intent_item.output.field_surfaces))]
    field_ids = [field.field_id for field in fields]
    contract = CompiledAnswerContract.build(
        source_intent=intent, source_intent_digest=semantic_intent_digest(intent),
        execution_plan_digest=plan.execution_plan_digest,
        items=[CompiledAnswerItem(
            item_id=intent_item.item_id, status="ready",
            projection=AnswerProjection(shape=intent_item.output.shape, fields=fields,
                                        presentation=intent_item.output.presentation),
            coverage=CoveragePartition(required=field_ids, executable=field_ids, limited=[]),
            support_requirements=supports)], groups=[],
        source_field_bindings={intent_item.item_id: [SourceFieldBinding(
            surface=surface, field_id=f"field-{index + 1}")
            for index, surface in enumerate(intent_item.output.field_surfaces)]},
        live_answer_root_refs=[row.root_id for row in plan.answer_roots],
        live_plan_root_refs=[], live_task_refs=[row.task_id for row in plan.resolved_plan.tasks],
        premise_contracts=[], presentation=intent.presentation)
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=semantic_intent_digest(intent),
        resolution_digest=resolution.resolution_digest, overlay_digest=None,
        intent=intent, resolution=resolution, execution_plan=plan,
        answer_contract=contract)


def _compile_periodic_document_narrative(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    typed = resolution_item.resolution
    if not isinstance(typed, PeriodicDocumentNarrativeResolution):
        raise DeterministicPlanCompilerError(
            "periodic document narrative resolution이 필요합니다")
    source_digest = semantic_intent_digest(intent)
    plan, supports = _build_periodic_document_narrative_typed_plan(
        resolution, resolution_item, intent_item)
    authority = _source_authority(intent)
    authority["source_field_bindings"] = {
        intent_item.item_id: [
            SourceFieldBinding(surface=surface, field_id=f"field-{index + 1}")
            for index, surface in enumerate(intent_item.output.field_surfaces)
        ]
    }
    fields = []
    for index in range(len(intent_item.output.field_surfaces)):
        executable = index in typed.executable_field_indexes
        fields.append(ProjectionField(
            field_id=f"field-{index + 1}",
            field_key=f"record_{index + 1}",
            binding_status="executable" if executable else "limited",
            answer_root_refs=[f"output-{index + 1}"] if executable else [],
            limitation_refs=[] if executable else ["limitation-1"],
        ))
    limitation = typed.source_cross_check_provenance
    compiled_item = CompiledAnswerItem(
        item_id=intent_item.item_id,
        status="partial" if typed.limited_field_indexes else "ready",
        projection=AnswerProjection(
            shape=intent_item.output.shape,
            fields=fields,
            presentation=intent_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=[
                f"field-{index + 1}"
                for index in range(len(intent_item.output.field_surfaces))
            ],
            executable=[
                f"field-{index + 1}"
                for index in typed.executable_field_indexes
            ],
            limited=[
                f"field-{index + 1}"
                for index in typed.limited_field_indexes
            ],
        ),
        support_requirements=supports,
        limitation_bindings=([] if limitation is None else [LimitationBinding(
            limitation_id="limitation-1",
            code="source_cross_check_partial",
            family="source_scope",
            applies_to_field_ids=[
                f"field-{index + 1}"
                for index in typed.limited_field_indexes
            ],
            detail=limitation.detail,
        )]),
    )
    contract = CompiledAnswerContract.build(
        source_intent=intent,
        source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest,
        items=[compiled_item], groups=authority["source_groups"],
        source_field_bindings=authority["source_field_bindings"],
        live_answer_root_refs=[row.root_id for row in plan.answer_roots],
        live_plan_root_refs=[row.plan_root_id for row in plan.plan_value_roots],
        live_task_refs=["task-1", "task-2"], premise_contracts=[],
        presentation=intent.presentation,
    )
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=source_digest,
        resolution_digest=resolution.resolution_digest,
        overlay_digest=None, intent=intent, resolution=resolution,
        execution_plan=plan, answer_contract=contract,
    )


def _compile_document_collection(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_authority: Any,
        resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    """Compile one whole-target document-set binding to one document task."""
    if (
            not isinstance(intent_authority, tuple)
            or len(intent_authority) != 2
    ):
        raise DeterministicPlanCompilerError(
            "document collection intent authority가 다릅니다")
    intent_item, _entity = intent_authority
    source_digest = semantic_intent_digest(intent)
    plan = _build_document_collection_typed_plan(
        resolution, resolution_item, intent_item)
    is_recent_narrative = intent_item.output.projection_mode == "named_fields"
    compiled_item = CompiledAnswerItem(
        item_id=intent_item.item_id,
        status="ready",
        projection=AnswerProjection(
            projection_mode=("named_fields" if is_recent_narrative else "whole_target"),
            shape=intent_item.output.shape,
            fields=([ProjectionField(
                field_id="field-1", field_key="narrative",
                binding_status="executable", answer_root_refs=["output-1"])]
                if is_recent_narrative else []),
            whole_target=(None if is_recent_narrative else WholeTargetBinding(
                whole_target_id="whole-target-1", binding_status="executable",
                answer_root_refs=["output-1"], limitation_refs=[])),
            presentation=intent_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=(["field-1"] if is_recent_narrative else ["whole-target-1"]),
            executable=(["field-1"] if is_recent_narrative else ["whole-target-1"]),
            limited=[],
        ),
        support_requirements=[], limitation_bindings=[],
    )
    contract = CompiledAnswerContract.build(
        source_intent=intent,
        source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest,
        items=[compiled_item], groups=[],
        source_field_bindings={intent_item.item_id: (
            [SourceFieldBinding(surface=intent_item.output.field_surfaces[0],
                                field_id="field-1")]
            if is_recent_narrative else [])},
        live_answer_root_refs=["output-1"],
        live_plan_root_refs=[], live_task_refs=["task-1"],
        premise_contracts=[], presentation=intent.presentation,
    )
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=source_digest,
        resolution_digest=resolution.resolution_digest,
        overlay_digest=None, intent=intent, resolution=resolution,
        execution_plan=plan, answer_contract=contract,
    )


def _compile_holding_disclosure(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_authority: Any,
        resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    if not isinstance(intent_authority, tuple) or len(intent_authority) != 2:
        raise DeterministicPlanCompilerError(
            "holding disclosure intent authority가 다릅니다")
    intent_item, _companies = intent_authority
    typed = resolution_item.resolution
    if not isinstance(typed, HoldingDisclosureResolution):
        raise DeterministicPlanCompilerError(
            "holding disclosure typed resolution이 필요합니다")
    source_digest = semantic_intent_digest(intent)
    plan = _build_holding_disclosure_typed_plan(
        resolution, resolution_item, intent_item)
    field_ids = [
        f"field-{index + 1}"
        for index in range(len(typed.slot_bindings))
    ]
    executable_indexes = [
        index for index, row in enumerate(typed.slot_bindings)
        if row.binding_status != "limited"]
    projection_fields, limited_field_ids, limitation_bindings = (
        _holding_contract_shape(typed))
    compiled_item = CompiledAnswerItem(
        item_id=intent_item.item_id,
        status="partial" if limited_field_ids else "ready",
        projection=AnswerProjection(
            projection_mode="named_fields",
            shape=intent_item.output.shape,
            fields=projection_fields,
            presentation=intent_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=field_ids,
            executable=[field_ids[index] for index in executable_indexes],
            limited=limited_field_ids),
        support_requirements=[SupportRequirement(
            support_id=f"support-{index + 1}",
            kind="citation", required=True,
            detail="exact holding filing field evidence",
            applies_to_field_ids=[field_ids[index]],
            answer_root_refs=[f"output-{index + 1}"],
        ) for index in executable_indexes],
        limitation_bindings=limitation_bindings,
    )
    contract = CompiledAnswerContract.build(
        source_intent=intent,
        source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest,
        items=[compiled_item], groups=[],
        source_field_bindings={intent_item.item_id: [SourceFieldBinding(
            surface=row.surface, field_id=field_ids[index],
        ) for index, row in enumerate(typed.slot_bindings)]},
        live_answer_root_refs=[
            f"output-{index + 1}" for index in executable_indexes],
        live_plan_root_refs=[], live_task_refs=["task-1"],
        premise_contracts=[], presentation=intent.presentation,
    )
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=source_digest,
        resolution_digest=resolution.resolution_digest,
        overlay_digest=None, intent=intent, resolution=resolution,
        execution_plan=plan, answer_contract=contract,
    )


def _compile_document_version_history(
        question: str, intent: SemanticIntent, resolution: AuthoritativeResolution,
        intent_authority: Any, resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    intent_item, _entity = intent_authority
    source_digest = semantic_intent_digest(intent)
    if not intent.premises:
        raise DeterministicPlanCompilerError(
            "document version history에는 전제가 하나 필요합니다")
    plan = _build_document_version_history_typed_plan(
        resolution, resolution_item, intent_item, intent.premises[0].raw_text)
    field = ProjectionField(field_id="field-1", field_key="correction_status",
        binding_status="executable", answer_root_refs=["output-1"])
    item = CompiledAnswerItem(item_id=intent_item.item_id, status="ready",
        projection=AnswerProjection(shape="scalar", fields=[field],
            presentation=intent_item.output.presentation),
        coverage=CoveragePartition(required=["field-1"], executable=["field-1"], limited=[]))
    premise = intent.premises[0]
    contract = CompiledAnswerContract.build(
        source_intent=intent, source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest, items=[item], groups=[],
        source_field_bindings={intent_item.item_id: [SourceFieldBinding(
            surface=intent_item.output.field_surfaces[0], field_id="field-1")]},
        live_answer_root_refs=["output-1"], live_plan_root_refs=[], live_task_refs=["task-1"],
        premise_contracts=[CompiledPremiseContract(
            premise_id=premise.premise_id, kind=premise.kind, raw_text=premise.raw_text,
            applies_to_item_ids=list(premise.applies_to_item_ids),
            verification_root_refs=["output-1"], verification_task_refs=["task-1"],
            verification_plan_root_refs=[], verdict_requirement="required")],
        presentation=intent.presentation)
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=source_digest, resolution_digest=resolution.resolution_digest,
        overlay_digest=None, intent=intent, resolution=resolution,
        execution_plan=plan, answer_contract=contract)


def _compile_selected_event(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_authority: Any,
        resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    if not isinstance(intent_authority, tuple) or len(intent_authority) != 3:
        raise DeterministicPlanCompilerError(
            "selected event intent authority가 다릅니다")
    intent_item, _entity, premises = intent_authority
    source_digest = semantic_intent_digest(intent)
    plan, supports = _build_selected_event_typed_plan(
        resolution, resolution_item, intent_item, premises)
    typed = resolution_item.resolution
    assert isinstance(typed, SelectedEventResolution)
    partial = (typed.operation == "timeline"
               and typed.lineage_missing_root_date is not None)
    fields = [ProjectionField(
        field_id=f"field-{index + 1}", field_key=f"value_{index + 1}",
        binding_status=("qualified" if partial else "executable"),
        answer_root_refs=[f"output-{index + 1}"],
        limitation_refs=([f"limitation-{index + 1}"] if partial else []),
    ) for index in range(len(intent_item.output.field_surfaces))]
    field_ids = [row.field_id for row in fields]
    compiled_item = CompiledAnswerItem(
        item_id=intent_item.item_id, status=("partial" if partial else "ready"),
        projection=AnswerProjection(
            shape=intent_item.output.shape, fields=fields,
            presentation=intent_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=field_ids, executable=field_ids,
            limited=(field_ids if partial else [])),
        support_requirements=supports,
        limitation_bindings=([] if not partial else [
            LimitationBinding(limitation_id="limitation-1", code="source_scope_raw_absent", family="source_scope", applies_to_field_ids=["field-1"], detail=f"original disclosure before corpus: {typed.lineage_missing_root_date}"),
            LimitationBinding(limitation_id="limitation-2", code="source_scope_prevents_complete_lineage", family="source_scope", applies_to_field_ids=["field-2"], detail=f"complete lineage unavailable because root before corpus: {typed.lineage_missing_root_date}"),
        ]),
    )
    source_bindings = {intent_item.item_id: [SourceFieldBinding(
        surface=surface, field_id=f"field-{index + 1}")
        for index, surface in enumerate(intent_item.output.field_surfaces)]}
    premise_contracts = [CompiledPremiseContract(
        premise_id=premise.premise_id, kind=premise.kind,
        raw_text=premise.raw_text,
        applies_to_item_ids=list(premise.applies_to_item_ids),
        verification_root_refs=_premise_bound_answer_roots(
            resolution, resolution_item, plan.answer_roots,
            premise.premise_id),
        verification_task_refs=["task-1"],
        verification_plan_root_refs=[], verdict_requirement="required",
    ) for premise in premises]
    contract = CompiledAnswerContract.build(
        source_intent=intent, source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest,
        items=[compiled_item], groups=[], source_field_bindings=source_bindings,
        live_answer_root_refs=[row.root_id for row in plan.answer_roots],
        live_plan_root_refs=[], live_task_refs=["task-1"],
        premise_contracts=premise_contracts, presentation=intent.presentation,
    )
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=source_digest,
        resolution_digest=resolution.resolution_digest, overlay_digest=None,
        intent=intent, resolution=resolution, execution_plan=plan,
        answer_contract=contract,
    )


def _compile_event_collection(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    source_digest = semantic_intent_digest(intent)
    plan, supports = _build_event_collection_typed_plan(
        resolution, resolution_item, intent_item)
    typed = resolution_item.resolution
    assert isinstance(typed, EventCollectionResolution)
    fields = []
    if len(intent_item.output.field_surfaces) == len(typed.requested_slots):
        for field_index, slot in enumerate(typed.requested_slots, start=1):
            roots = (
                [f"output-{field_index}"] if not typed.events
                else [
                    f"output-{(event_index - 1) * len(typed.requested_slots) + field_index}"
                    for event_index in range(1, len(typed.events) + 1)
                ]
            )
            fields.append(ProjectionField(
                field_id=f"field-{field_index}", field_key=slot,
                binding_status="executable", answer_root_refs=roots,
            ))
    elif len(intent_item.output.field_surfaces) == 1:
        supports = [support.model_copy(update={
            "applies_to_field_ids": ["field-1"],
        }) for support in supports]
        fields.append(ProjectionField(
            field_id="field-1", field_key=intent_item.output.field_surfaces[0],
            binding_status="executable",
            answer_root_refs=[root.root_id for root in plan.answer_roots],
        ))
    else:
        raise DeterministicPlanCompilerError(
            "event collection source/slot projection cardinality가 다릅니다")
    field_ids = [field.field_id for field in fields]
    compiled_item = CompiledAnswerItem(
        item_id=intent_item.item_id, status="ready",
        projection=AnswerProjection(
            shape=intent_item.output.shape, fields=fields,
            presentation=intent_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=field_ids, executable=field_ids, limited=[]),
        support_requirements=supports, limitation_bindings=[],
    )
    contract = CompiledAnswerContract.build(
        source_intent=intent, source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest,
        items=[compiled_item], groups=[],
        source_field_bindings={intent_item.item_id: [SourceFieldBinding(
            surface=surface, field_id=f"field-{index + 1}")
            for index, surface in enumerate(intent_item.output.field_surfaces)]},
        live_answer_root_refs=[row.root_id for row in plan.answer_roots],
        live_plan_root_refs=[],
        live_task_refs=[row.task_id for row in plan.resolved_plan.tasks],
        premise_contracts=[], presentation=intent.presentation,
    )
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=source_digest,
        resolution_digest=resolution.resolution_digest, overlay_digest=None,
        intent=intent, resolution=resolution, execution_plan=plan,
        answer_contract=contract,
    )


def _lifecycle_attribute_by_kind(
        typed: LifecycleCompositeResolution, kind: str,
        ) -> LifecycleCompositeAttribute:
    matches = [row for row in typed.attributes if row.kind == kind]
    if len(matches) != 1:
        raise DeterministicPlanCompilerError("lifecycle attribute proof가 유일하지 않습니다")
    return matches[0]


def _build_lifecycle_composite_typed_plan(
        resolution: AuthoritativeResolution, intent_items: tuple[Any, ...],
        ) -> tuple[ExecutionPlan, list[SupportRequirement], list[tuple[str, str, str]]]:
    """Lower status, source attributes, and correction history by role."""
    tasks: list[Any] = []
    roots: list[ExecutionAnswerRoot] = []
    supports: list[SupportRequirement] = []
    support_roots: list[ExecutionSupportRoot] = []
    root_index = 0
    bindings: list[tuple[str, str, str]] = []
    for item, resolved_item in zip(intent_items, resolution.items, strict=True):
        typed = resolved_item.resolution
        assert isinstance(typed, LifecycleCompositeResolution)
        mode = _lifecycle_item_mode(item, typed)
        if mode is None:
            raise DeterministicPlanCompilerError("lifecycle item role이 닫히지 않습니다")
        if mode == "status":
            task_id = f"task-{len(tasks) + 1}"
            tasks.append(ResolvedEventTask(
                task_id=task_id, operation="status", corp_code=typed.corp_code,
                corp_name=typed.corp_name,
                selector=EventSelector(event_key=typed.event_key),
                timepoints=list(typed.status_timepoints),
                requested_slots=_lifecycle_status_requested_slots(item),
                field_outputs=[]))
            for field_index, surface in enumerate(item.output.field_surfaces, start=1):
                field_id = f"{item.item_id}-field-{field_index}"
                root_index += 1
                root_id = f"output-{root_index}"
                roots.append(ExecutionAnswerRoot(
                    root_id=root_id, item_id=item.item_id,
                    field_id=field_id, plan_task_id=task_id,
                    proof_ref=typed.selector_proof.proof_ref))
                bindings.append((item.item_id, surface, root_id))
                support_id = f"support-{root_index}"
                supports.append(SupportRequirement(
                    support_id=support_id, kind="citation",
                    detail=(f"event_key={typed.event_key};root_receipt={typed.root_receipt};"
                            f"timepoints={','.join(typed.status_timepoints)}"),
                    applies_to_field_ids=[field_id], answer_root_refs=[root_id]))
                support_roots.append(ExecutionSupportRoot(
                    support_id=support_id, kind="citation", item_id=item.item_id,
                    field_id=field_id, root_id=root_id))
        elif mode == "correction":
            task_id = f"task-{len(tasks) + 1}"
            tasks.append(ResolvedCorrectionTask(
                task_id=task_id, operation="history", corp_code=typed.corp_code,
                corp_name=typed.corp_name, as_of=typed.as_of,
                event_selector=EventSelector(event_key=typed.event_key),
                requested_slots=[], field_outputs=[]))
            for field_index, surface in enumerate(item.output.field_surfaces, start=1):
                field_id = f"{item.item_id}-field-{field_index}"
                root_index += 1
                root_id = f"output-{root_index}"
                roots.append(ExecutionAnswerRoot(
                    root_id=root_id, item_id=item.item_id,
                    field_id=field_id, plan_task_id=task_id,
                    proof_ref=typed.selector_proof.proof_ref))
                bindings.append((item.item_id, surface, root_id))
                support_id = f"support-{root_index}"
                supports.append(SupportRequirement(
                    support_id=support_id, kind="citation",
                    detail=(f"event_key={typed.event_key};root_receipt={typed.root_receipt};"
                            f"corrections={','.join(typed.correction_receipts)}"),
                    applies_to_field_ids=[field_id], answer_root_refs=[root_id]))
                support_roots.append(ExecutionSupportRoot(
                    support_id=support_id, kind="citation", item_id=item.item_id,
                    field_id=field_id, root_id=root_id))
        else:
            for field_index, surface in enumerate(item.output.field_surfaces, start=1):
                field_id = f"{item.item_id}-field-{field_index}"
                kind = _lifecycle_attribute_kind_for_surface(surface)
                assert kind is not None
                attribute = _lifecycle_attribute_by_kind(typed, kind)
                slot = _LIFECYCLE_ATTRIBUTE_SLOTS[kind]
                task_id = f"task-{len(tasks) + 1}"
                root_index += 1
                root_id = f"output-{root_index}"
                value_kind = "money" if kind.endswith("amount") else "text"
                tasks.append(ResolvedDisclosureTask(
                    task_id=task_id, operation="lookup", corp_code=typed.corp_code,
                    corp_name=typed.corp_name, as_of=typed.as_of,
                    document_selector=DocumentSelector(rcept_no=attribute.source_receipt),
                    requested_slots=[slot], field_outputs=[FieldOutputSpec(
                        output_id=root_id, slot=slot, value_kind=value_kind)]))
                roots.append(ExecutionAnswerRoot(
                    root_id=root_id, item_id=item.item_id,
                    field_id=field_id, plan_output_id=root_id,
                    proof_ref=f"canonical:field:{attribute.evidence_id}"))
                bindings.append((item.item_id, surface, root_id))
                support_id = f"support-{root_index}"
                supports.append(SupportRequirement(
                    support_id=support_id, kind="citation",
                    detail=(f"event_key={typed.event_key};receipt={attribute.source_receipt};"
                            f"path={attribute.path};locator={attribute.locator};"
                            f"proof_ref=canonical:field:{attribute.evidence_id}"),
                    applies_to_field_ids=[field_id], answer_root_refs=[root_id]))
                support_roots.append(ExecutionSupportRoot(
                    support_id=support_id, kind="citation", item_id=item.item_id,
                    field_id=field_id, root_id=root_id))
    plan = ExecutionPlan.create(
        source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest,
        canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version,
        resolved_plan=ResolvedQueryPlan(
            revision=0, reference_date=resolution.reference_date,
            corpus_cutoff=resolution.corpus_cutoff, tasks=tasks, derivations=[],
            premise_claims=[], applied_defaults=[], presentation=None),
        applied_defaults=[], answer_roots=roots, support_roots=support_roots,
        premise_roots=[], plan_value_roots=[])
    return plan, supports, bindings


def _compile_lifecycle_composite(
        question: str, intent: SemanticIntent, resolution: AuthoritativeResolution,
        intent_items: Any, resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    del resolution_item
    if not isinstance(intent_items, tuple):
        raise DeterministicPlanCompilerError("lifecycle intent authority가 다릅니다")
    plan, supports, bindings = _build_lifecycle_composite_typed_plan(
        resolution, intent_items)
    roots_by_item_field = {(item_id, surface): root_id
                           for item_id, surface, root_id in bindings}
    compiled_items: list[CompiledAnswerItem] = []
    source_bindings: dict[str, list[SourceFieldBinding]] = {}
    for item in intent_items:
        fields = [ProjectionField(
            field_id=f"{item.item_id}-field-{index}", field_key=surface,
            binding_status="executable",
            answer_root_refs=[roots_by_item_field[(item.item_id, surface)]],
        ) for index, surface in enumerate(item.output.field_surfaces, start=1)]
        field_ids = [field.field_id for field in fields]
        item_supports = [support for support in supports
                         if any(root in {value for key, value in roots_by_item_field.items()
                                         if key[0] == item.item_id}
                                for root in support.answer_root_refs)]
        compiled_items.append(CompiledAnswerItem(
            item_id=item.item_id, status="ready",
            projection=AnswerProjection(shape=item.output.shape, fields=fields,
                presentation=item.output.presentation),
            coverage=CoveragePartition(required=field_ids, executable=field_ids, limited=[]),
            support_requirements=item_supports, limitation_bindings=[]))
        source_bindings[item.item_id] = [SourceFieldBinding(
            surface=surface, field_id=f"{item.item_id}-field-{index}")
            for index, surface in enumerate(item.output.field_surfaces, start=1)]
    contract = CompiledAnswerContract.build(
        source_intent=intent, source_intent_digest=semantic_intent_digest(intent),
        execution_plan_digest=plan.execution_plan_digest, items=compiled_items,
        groups=[
            CompiledAnswerGroup(
                group_id=group.group_id, item_ids=list(group.item_ids))
            for group in intent.answer_groups
        ], source_field_bindings=source_bindings,
        live_answer_root_refs=[row.root_id for row in plan.answer_roots],
        live_plan_root_refs=[], live_task_refs=[row.task_id for row in plan.resolved_plan.tasks],
        premise_contracts=[], presentation=intent.presentation)
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=semantic_intent_digest(intent),
        resolution_digest=resolution.resolution_digest, overlay_digest=None,
        intent=intent, resolution=resolution, execution_plan=plan,
        answer_contract=contract)


def _compile_event_amount_change(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    source_digest = semantic_intent_digest(intent)
    plan, supports = _build_event_amount_change_typed_plan(
        resolution, resolution_item)
    compiled_item = CompiledAnswerItem(
        item_id=intent_item.item_id, status="ready",
        projection=AnswerProjection(
            shape=intent_item.output.shape,
            fields=[ProjectionField(
                field_id="field-1", field_key="change",
                binding_status="executable",
                answer_root_refs=["output-3"],
            )],
            presentation=intent_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=["field-1"], executable=["field-1"], limited=[]),
        support_requirements=supports,
    )
    contract = CompiledAnswerContract.build(
        source_intent=intent, source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest,
        items=[compiled_item], groups=[],
        source_field_bindings={intent_item.item_id: [SourceFieldBinding(
            surface=intent_item.output.field_surfaces[0], field_id="field-1",
        )]},
        live_answer_root_refs=["output-3"],
        live_plan_root_refs=[
            row.plan_root_id for row in plan.plan_value_roots],
        live_task_refs=["task-1", "task-2"], premise_contracts=[],
        presentation=intent.presentation,
    )
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=source_digest,
        resolution_digest=resolution.resolution_digest,
        overlay_digest=None, intent=intent, resolution=resolution,
        execution_plan=plan, answer_contract=contract,
    )


def _compile_termination_reported_status(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_authority: Any,
        resolution_item: ResolvedItem,
        ) -> DeterministicCompiledSlice:
    if not isinstance(intent_authority, tuple) or len(intent_authority) != 3:
        raise DeterministicPlanCompilerError(
            "reported termination intent authority가 다릅니다")
    intent_item, _issuer, premises = intent_authority
    typed = resolution_item.resolution
    if not isinstance(typed, TerminationReportedStatusResolution):
        raise DeterministicPlanCompilerError(
            "reported termination resolution kind가 다릅니다")
    source_digest = semantic_intent_digest(intent)
    plan, supports = _build_termination_reported_status_typed_plan(
        resolution, resolution_item, intent_item, premises)
    _normalized_fields, field_roles = _termination_reported_status_field_roles(
        intent_item)
    fields = [ProjectionField(
        field_id=f"field-{index + 1}", field_key=field_roles[index],
        binding_status="qualified", answer_root_refs=[f"output-{index + 1}"],
        limitation_refs=["limitation-1"],
    ) for index in range(len(field_roles))]
    field_ids = [field.field_id for field in fields]
    compiled_item = CompiledAnswerItem(
        item_id=intent_item.item_id, status="partial",
        projection=AnswerProjection(
            shape=intent_item.output.shape, fields=fields,
            presentation=intent_item.output.presentation,
        ),
        coverage=CoveragePartition(
            required=field_ids, executable=field_ids, limited=field_ids),
        support_requirements=supports,
        limitation_bindings=[LimitationBinding(
            limitation_id="limitation-1", code=typed.identity_provenance.code,
            family=typed.identity_provenance.family,
            applies_to_field_ids=field_ids,
            detail=typed.identity_provenance.detail,
        )],
    )
    source_bindings = {intent_item.item_id: [SourceFieldBinding(
        surface=surface, field_id=f"field-{index + 1}")
        for index, surface in enumerate(intent_item.output.field_surfaces)]}
    premise_contracts = [CompiledPremiseContract(
        premise_id=premise.premise_id, kind=premise.kind,
        raw_text=premise.raw_text,
        applies_to_item_ids=list(premise.applies_to_item_ids),
        verification_root_refs=_premise_bound_answer_roots(
            resolution, resolution_item, plan.answer_roots,
            premise.premise_id),
        verification_task_refs=["task-1"],
        verification_plan_root_refs=[], verdict_requirement="required",
    ) for premise in premises]
    contract = CompiledAnswerContract.build(
        source_intent=intent, source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest,
        items=[compiled_item], groups=[], source_field_bindings=source_bindings,
        live_answer_root_refs=[row.root_id for row in plan.answer_roots],
        live_plan_root_refs=[], live_task_refs=["task-1"],
        premise_contracts=premise_contracts, presentation=intent.presentation,
    )
    return DeterministicCompiledSlice.create(
        question_id=resolution.question_id, question=question,
        source_intent_digest=source_digest,
        resolution_digest=resolution.resolution_digest, overlay_digest=None,
        intent=intent, resolution=resolution, execution_plan=plan,
        answer_contract=contract,
    )


@dataclass(frozen=True)
class Stage1V1Handler:
    """One closed lowering registration selected without a trace question id.

    Selection uses only the intent structural signature and the ordered typed
    resolution-kind inventory.  Question IDs are deliberately absent from a
    registration so they cannot become a dispatch key.
    """

    name: str
    intent_signature: str
    resolution_kind_inventory: tuple[str, ...]
    validate_intent: Callable[[SemanticIntent], tuple[Any, Any]]
    validate_resolution: Callable[
        [Any, AuthoritativeResolution], tuple[ResolvedItem, str | None]]
    lower: Callable[
        [str, SemanticIntent, AuthoritativeResolution, Any, ResolvedItem,
         str, str | None], DeterministicCompiledSlice]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("handler name은 비어 있을 수 없습니다")
        if not self.intent_signature:
            raise ValueError("handler intent_signature는 비어 있을 수 없습니다")
        if not self.resolution_kind_inventory:
            raise ValueError("handler resolution_kind_inventory는 비어 있을 수 없습니다")
        if any(not isinstance(kind, str) or not kind for kind in
               self.resolution_kind_inventory):
            raise ValueError("handler resolution.kind inventory가 유효하지 않습니다")


def _structural_signature_payload(
        intent: SemanticIntent,
        ) -> dict[str, Any]:
    """Project an intent to qid/surface-free enum/topology shape."""
    entity_positions = {
        row.entity_id: index for index, row in enumerate(intent.entities)
    }
    item_positions = {
        row.item_id: index for index, row in enumerate(intent.answer_items)
    }

    def entity_refs(values: list[str]) -> list[int]:
        try:
            return [entity_positions[value] for value in values]
        except KeyError as exc:
            raise DeterministicPlanCompilerError(
                "intent structural signature entity ref가 유효하지 않습니다") from exc

    def item_refs(values: list[str]) -> list[int]:
        try:
            return [item_positions[value] for value in values]
        except KeyError as exc:
            raise DeterministicPlanCompilerError(
                "intent structural signature item ref가 유효하지 않습니다") from exc

    payload = {
        "entities": [
            {"kind_hint": row.kind_hint}
            for row in intent.entities
        ],
        "answer_items": [
            {
                "target": {
                    "kind": row.target.kind,
                    "entity_positions": entity_refs(row.target.entity_refs),
                    "qualifier_count": len(row.target.qualifier_surfaces),
                },
                "operation": row.operation,
                "scope": {
                    "target_period_count": len(
                        row.scope.target_period_expressions),
                    "has_as_of": row.scope.as_of_expression is not None,
                    "has_document_group": (
                        row.scope.document_group_expression is not None),
                    "scope_qualifier_count": len(
                        row.scope.scope_qualifier_expressions),
                },
                "selection": (
                    None if row.selection is None else {
                        "mode": row.selection.mode,
                        "has_k": row.selection.k is not None,
                    }
                ),
                "output": {
                    "shape": row.output.shape,
                    "projection_mode": row.output.projection_mode,
                    "field_count": len(row.output.field_surfaces),
                    "presentation": row.output.presentation,
                },
            }
            for row in intent.answer_items
        ],
        "answer_groups": [item_refs(row.item_ids) for row in intent.answer_groups],
        "premises": [
            {
                "kind": row.kind,
                "item_positions": item_refs(row.applies_to_item_ids),
            }
            for row in intent.premises
        ],
        "unresolved_mentions": [
            {
                "role_hint": row.role_hint,
                "item_positions": item_refs(row.applies_to_item_ids),
            }
            for row in intent.unresolved_mentions
        ],
        "presentation": intent.presentation,
    }
    return payload


def _structural_signature_from_payload(
        payload: Mapping[str, Any],
        ) -> str:
    return canonical_json(payload)


def semantic_intent_structural_signature(
        intent: SemanticIntent | Mapping[str, Any],
        ) -> str:
    """Return a deterministic qid- and surface-free signature of intent shape.

    Generated local identifiers are projected to array positions. Literal
    question/entity/target/field/date/premise surfaces are intentionally
    excluded: registration is based on enum, shape, cardinality, and
    reference topology. Exact source authority remains in selected handler
    validators after dispatch.
    """
    normalized = _strict_intent(intent)
    return _structural_signature_from_payload(
        _structural_signature_payload(normalized))


def _resolution_kind_inventory(
        resolution: AuthoritativeResolution,
        ) -> tuple[str, ...]:
    return tuple(row.resolution.kind for row in resolution.items)


def _selected_handler(
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        ) -> Stage1V1Handler:
    signature = semantic_intent_structural_signature(intent)
    kinds = _resolution_kind_inventory(resolution)
    matches = [
        handler for handler in STAGE1_V1_HANDLER_REGISTRY
        if handler.intent_signature == signature
        and handler.resolution_kind_inventory == kinds
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise DeterministicPlanCompilerError(
            "semantic intent structural signature와 ordered resolution.kind "
            f"inventory가 여러 Stage1 handler와 일치합니다: {len(matches)}")

    # **증명된 것으로 route 한다.**
    #
    # structural signature 는 모델이 질문을 어떤 모양으로 읽었는지다.  같은
    # 질문도 판마다 `topic`/`document`, `record`/`record_list` 로 갈린다.  그
    # 표현 차이로 경로를 잠그면 처음 보는 질문이 값은 맞아도 거절된다.
    #
    # resolution.kind inventory 는 다르다.  resolver 가 정본에서 **확정한** 것이고
    # 지어낼 수 없다.  그래서 정확한 signature 가 없을 때는 kind inventory 로
    # 후보를 좁히고, 그 handler 자신의 `validate_intent` 가 의미를 다시 본다.
    # 통과하지 못하면 예전처럼 컴파일이 실패하므로 문이 열리는 것이 아니다.
    by_kind = [
        handler for handler in STAGE1_V1_HANDLER_REGISTRY
        if handler.resolution_kind_inventory == kinds
    ]
    if len(by_kind) == 1:
        return by_kind[0]
    if not by_kind:
        raise DeterministicPlanCompilerError(
            "semantic intent structural signature와 ordered resolution.kind "
            "inventory에 맞는 Stage1 handler가 없습니다")

    # kind 가 같은 handler 가 여럿이면(`financial` 은 연결기준 명시본과 기본본이
    # 있다) **각자의 의미 검증에게 물어본다.**  통과하는 것이 하나면 그것이
    # 답이다.  둘 이상 통과하면 고르지 않는다 — 임의로 집으면 다른 계획이
    # 나간다.  반대로 하나도 통과하지 못한 경우는 *중복*이 아니다.  이것은
    # resolver 가 확정한 kind와 semantic topology를 등록된 lowering이 아직
    # 지원하지 않는 no-match다.  두 상태를 같은 오류로 합치면, 예를 들어
    # 단일 financial authority를 받은 comparison intent가 duplicate handler로
    # 잘못 집계되어 원인 분류와 후속 fallback 선택이 모두 흐려진다.
    accepted = []
    for handler in by_kind:
        try:
            handler.validate_intent(intent)
        except DeterministicPlanCompilerError:
            continue
        accepted.append(handler)
    if len(accepted) == 1:
        return accepted[0]
    if not accepted:
        raise DeterministicPlanCompilerError(
            "semantic intent structural signature와 ordered resolution.kind "
            "inventory에 맞는 의미 handler가 없습니다 "
            f"(kind 후보 {len(by_kind)}, 의미 검증 통과 0)")
    # Several structural registrations may describe surface variants of the
    # same capability (explicit/default scope, qualifier placement, one/two
    # requested outputs).  If they converge on the exact same resolution
    # validator and lowering function, choosing the first registry entry is
    # deterministic and cannot change the compiled plan.  Only genuinely
    # different lowering authorities remain fail-closed.
    equivalent_lowerings = {
        (handler.validate_resolution, handler.lower) for handler in accepted}
    if len(equivalent_lowerings) == 1:
        return accepted[0]
    raise DeterministicPlanCompilerError(
        "ordered resolution.kind inventory가 여러 Stage1 handler와 "
        f"일치합니다: {len(by_kind)} (의미 검증 통과 {len(accepted)})")


def _lower_financial(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del trace_question_id
    return _compile_financial(
        question, intent, resolution, intent_item, resolution_item,
        overlay_digest)


def _lower_premise_financial(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    if not isinstance(intent_item, tuple) or len(intent_item) != 2:
        raise DeterministicPlanCompilerError(
            "premise financial lowerer에는 item/premise authority가 필요합니다")
    return _compile_premise_financial(
        question, intent, resolution, intent_item, resolution_item)


def _lower_same_day_status(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_same_day_status(
        question, intent, resolution, intent_item, resolution_item)


def _lower_termination_reported_status(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_termination_reported_status(
        question, intent, resolution, intent_item, resolution_item)


def _lower_financial_comparison(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_financial_comparison(
        question, intent, resolution, intent_item, resolution_item)


def _lower_financial_retrieve_comparison(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_items: Any,
        resolution_item: ResolvedItem,
        trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del resolution_item, trace_question_id, overlay_digest
    if not isinstance(intent_items, tuple) or len(intent_items) != 2:
        raise DeterministicPlanCompilerError(
            "financial retrieve/comparison lowerer에는 item pair가 필요합니다")
    return _compile_financial_retrieve_comparison(
        question, intent, resolution, intent_items)


def _lower_parallel_financial_retrieval(
        question: str, intent: SemanticIntent,
        resolution: AuthoritativeResolution, intent_items: Any,
        resolution_item: ResolvedItem, trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del resolution_item, trace_question_id, overlay_digest
    if not isinstance(intent_items, tuple) or len(intent_items) != 2:
        raise DeterministicPlanCompilerError(
            "parallel financial retrieval lowerer에는 item pair가 필요합니다")
    return _compile_parallel_financial_retrieval(
        question, intent, resolution, intent_items)


def _lower_summary_metric_fanout(
        question: str, intent: SemanticIntent,
        resolution: AuthoritativeResolution, intent_items: Any,
        resolution_item: ResolvedItem, trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del resolution_item, trace_question_id, overlay_digest
    if not isinstance(intent_items, tuple) or len(intent_items) < 3:
        raise DeterministicPlanCompilerError(
            "summary metric fanout lowerer에는 item 세 개 이상이 필요합니다")
    # 파생 없이 사실 하나에 답 뿌리 하나 — CFS/SFS 짝과 모양이 같고 항목 수만
    # 다르다. 그 lowering 은 이미 항목 수에 무관하므로 그대로 쓴다.
    return _compile_parallel_financial_retrieval(
        question, intent, resolution, intent_items)


def _lower_recent_periods_fanout(
        question: str, intent: SemanticIntent,
        resolution: AuthoritativeResolution, intent_items: Any,
        resolution_item: ResolvedItem, trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del resolution_item, trace_question_id, overlay_digest
    if not isinstance(intent_items, tuple) or not (2 <= len(intent_items) <= 5):
        raise DeterministicPlanCompilerError(
            "recent periods fanout lowerer에는 item 2~5개가 필요합니다")
    # 직접 공시된 분기는 사실 하나, Q4는 FY-9M 파생 하나를 공개 답 뿌리로
    # 갖는다. 두 형태를 한 series에서 순서대로 조립한다.
    return _compile_recent_periods_fanout(
        question, intent, resolution, intent_items)


def _lower_explicit_periods_fanout(
        question: str, intent: SemanticIntent,
        resolution: AuthoritativeResolution, intent_items: Any,
        resolution_item: ResolvedItem, trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del resolution_item, trace_question_id, overlay_digest
    if not isinstance(intent_items, tuple) or not (2 <= len(intent_items) <= 5):
        raise DeterministicPlanCompilerError(
            "explicit periods fanout lowerer에는 item 2~5개가 필요합니다")
    # 이슈 #171 M16 — 위 두 fanout과 같은 모양이고(파생 없이 사실 하나에
    # 답 뿌리 하나) 형제 사이에서 갈리는 축(개념 vs 상대 기간 vs 이미 확정된
    # 리터럴 기간)만 다르다. lowering은 그 축을 보지 않으므로 그대로 쓴다.
    return _compile_parallel_financial_retrieval(
        question, intent, resolution, intent_items)


def _lower_parallel_annual_change(
        question: str, intent: SemanticIntent,
        resolution: AuthoritativeResolution, intent_items: Any,
        resolution_item: ResolvedItem, trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del resolution_item, trace_question_id, overlay_digest
    if not isinstance(intent_items, tuple) or len(intent_items) != 2:
        raise DeterministicPlanCompilerError(
            "parallel annual change lowerer에는 item pair가 필요합니다")
    return _compile_parallel_annual_change(
        question, intent, resolution, intent_items)


def _lower_document_fact_comparison(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest, resolution_item
    if not isinstance(intent_item, tuple) or len(intent_item) != 2:
        raise DeterministicPlanCompilerError(
            "document fact lowerer에는 ordered intent item pair가 필요합니다")
    return _compile_document_fact_comparison(
        question, intent, resolution, intent_item, resolution.items[0])


def _lower_document_attributes(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_document_attributes(
        question, intent, resolution, intent_item, resolution_item)


def _lower_periodic_narrative_comparison(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_periodic_narrative_comparison(
        question, intent, resolution, intent_item, resolution_item)


def _lower_narrative_matrix(
        question: str, intent: SemanticIntent, resolution: AuthoritativeResolution,
        intent_item: Any, resolution_item: ResolvedItem, trace_question_id: str,
        overlay_digest: str | None) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_narrative_matrix(
        question, intent, resolution, intent_item, resolution_item)


def _lower_periodic_document_narrative(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_periodic_document_narrative(
        question, intent, resolution, intent_item, resolution_item)


def _lower_document_collection(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_document_collection(
        question, intent, resolution, intent_item, resolution_item)


def _lower_holding_disclosure(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_holding_disclosure(
        question, intent, resolution, intent_item, resolution_item)


def _lower_document_version_history(
        question: str, intent: SemanticIntent, resolution: AuthoritativeResolution,
        intent_item: Any, resolution_item: ResolvedItem, trace_question_id: str,
        overlay_digest: str | None) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_document_version_history(
        question, intent, resolution, intent_item, resolution_item)


def _lower_selected_event(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_selected_event(
        question, intent, resolution, intent_item, resolution_item)


def _lower_event_collection(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_event_collection(
        question, intent, resolution, intent_item, resolution_item)


def _lower_event_amount_change(
        question: str,
        intent: SemanticIntent,
        resolution: AuthoritativeResolution,
        intent_item: Any,
        resolution_item: ResolvedItem,
        trace_question_id: str,
        overlay_digest: str | None,
        ) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_event_amount_change(
        question, intent, resolution, intent_item, resolution_item)


def _validate_correction_diff_intent(intent: SemanticIntent) -> tuple[Any, None]:
    if (len(intent.answer_items) != 1 or intent.answer_groups
            or intent.premises or intent.unresolved_mentions):
        raise DeterministicPlanCompilerError("correction diff intent inventory가 다릅니다")
    item = intent.answer_items[0]
    if intent.entities:
        if (len(intent.entities) != 1
                or intent.entities[0].kind_hint != "company"
                or item.target.entity_refs != [intent.entities[0].entity_id]):
            raise DeterministicPlanCompilerError(
                "correction diff company authority가 다릅니다")
    elif item.target.entity_refs:
        raise DeterministicPlanCompilerError(
            "correction diff target entity ref가 닫히지 않았습니다")
    if (item.item_id != "item-1" or item.target.kind != "document"
            or item.target.qualifier_surfaces
            or item.operation != "retrieve" or len(item.scope.target_period_expressions) != 1
            or item.scope.as_of_expression is not None or item.scope.document_group_expression is not None
            or item.scope.scope_qualifier_expressions or item.selection is not None
            or item.output.shape != "narrative" or item.output.projection_mode != "whole_target"
            or item.output.field_surfaces):
        raise DeterministicPlanCompilerError("correction diff intent topology가 다릅니다")
    return item, None


def _validate_correction_history_intent(intent: SemanticIntent) -> tuple[tuple[Any, Any], None]:
    if (len(intent.answer_items) != 2 or len(intent.answer_groups) != 1
            or len(intent.premises) != 2 or intent.unresolved_mentions):
        raise DeterministicPlanCompilerError("correction history intent inventory가 다릅니다")
    amount, reason = intent.answer_items
    numeric, state = intent.premises
    if (amount.item_id != "item-1" or reason.item_id != "item-2"
            or amount.target.kind != "metric" or reason.target.kind != "event"
            or amount.operation != "retrieve" or reason.operation != "retrieve"
            or amount.output.shape != "scalar" or reason.output.shape != "narrative"
            or amount.output.projection_mode != "named_fields"
            or reason.output.projection_mode != "named_fields"
            or len(amount.output.field_surfaces) != 1 or len(reason.output.field_surfaces) != 1
            or numeric.kind != "numeric" or numeric.applies_to_item_ids != ["item-1"]
            or state.kind != "state" or state.applies_to_item_ids != ["item-1", "item-2"]
            or intent.answer_groups[0].item_ids != ["item-1", "item-2"]):
        raise DeterministicPlanCompilerError("correction history intent topology가 다릅니다")
    return (amount, reason), None


def _validate_correction_diff_resolution(intent_item: Any, resolution: AuthoritativeResolution) -> tuple[ResolvedItem, None]:
    if len(resolution.items) != 1 or resolution.premise_proofs:
        raise DeterministicPlanCompilerError("correction diff resolution inventory가 다릅니다")
    item = resolution.items[0]
    typed = item.resolution
    ranged_history = bool(
        isinstance(typed, CorrectionLineageResolution)
        and typed.operation == "history" and typed.sequence
        and typed.date_roles is not None
        and typed.date_roles.range_requested)
    if (not isinstance(typed, CorrectionLineageResolution)
            or typed.operation not in {"diff", "history"}
            or (typed.operation == "history" and not ranged_history)
            or typed.answer_role != "diff" or item.item_id != intent_item.item_id
            or item.target_surface != intent_item.target.surface or item.projection_mode != "whole_target"
            or item.field_proofs or item.applied_defaults):
        raise DeterministicPlanCompilerError("correction diff authority가 intent와 다릅니다")
    intent_coordinate = _correction_coordinate_from_intent(intent_item)
    expected_coordinate = (
        typed.correction_receipt
        if re.fullmatch(r"[0-9]{14}", intent_coordinate)
        else typed.date_roles.correction_from
        if typed.sequence and typed.date_roles is not None
        and typed.date_roles.range_requested
        else typed.correction_date)
    observed_receipt_bound = bool(
        ranged_history and typed.source_root_missing
        and intent_coordinate in {
            step.correction_receipt for step in typed.sequence})
    if intent_coordinate != expected_coordinate and not observed_receipt_bound:
        raise DeterministicPlanCompilerError(
            "correction diff intent/date role 결속이 다릅니다")
    return item, None


def _correction_coordinate_from_intent(item: Any) -> str:
    value = item.scope.target_period_expressions[0]
    if re.fullmatch(r"[0-9]{14}", value):
        return value
    start, end, error = _target_date_range(value, reference_date=date(2026, 1, 1))
    if error is not None or start is None or start != end:
        raise DeterministicPlanCompilerError("correction date surface가 exact day가 아닙니다")
    return start


def _validate_correction_history_resolution(intent_items: Any, resolution: AuthoritativeResolution) -> tuple[ResolvedItem, None]:
    if not isinstance(intent_items, tuple) or len(intent_items) != 2:
        raise DeterministicPlanCompilerError("correction history intent authority가 다릅니다")
    amount, reason = intent_items
    if len(resolution.items) != 2 or [p.premise_id for p in resolution.premise_proofs] != ["premise-1", "premise-2"]:
        raise DeterministicPlanCompilerError("correction history resolution inventory가 다릅니다")
    first, second = resolution.items
    typed = first.resolution
    expected = [(amount, first, "amount"), (reason, second, "reason")]
    if not isinstance(typed, CorrectionLineageResolution) or typed.operation != "history":
        raise DeterministicPlanCompilerError("correction history resolution kind가 다릅니다")
    for source, row, role in expected:
        candidate = row.resolution
        if (not isinstance(candidate, CorrectionLineageResolution)
                or candidate.model_dump(mode="json") != typed.model_dump(mode="json") | {"answer_role": role}
                or row.item_id != source.item_id or row.target_surface != source.target.surface
                or row.projection_mode != "named_fields" or row.applied_defaults
                or [(p.source_field_index, p.surface) for p in row.field_proofs]
                != [(0, source.output.field_surfaces[0])]):
            raise DeterministicPlanCompilerError("correction history answer binding이 다릅니다")
    proofs = {proof.proof_ref for change in typed.changes for proof in (change.before_proof, change.after_proof) if proof is not None}
    if not proofs or any(not set(row.proof_refs).issubset(proofs) for row in resolution.premise_proofs):
        raise DeterministicPlanCompilerError("correction history premise proof가 lineage에 결속되지 않았습니다")
    return first, None


def _build_correction_lineage_plan(resolution: AuthoritativeResolution, items: list[ResolvedItem], intent_items: list[Any], premises: tuple[Any, ...] = ()) -> tuple[ExecutionPlan, list[SupportRequirement]]:
    typed = items[0].resolution
    if not isinstance(typed, CorrectionLineageResolution):
        raise DeterministicPlanCompilerError("correction lineage typed resolution이 필요합니다")
    is_whole_target = typed.answer_role == "diff"
    is_diff = typed.operation == "diff"
    ranged_history = bool(
        is_whole_target and typed.sequence and typed.date_roles is not None
        and typed.date_roles.range_requested)
    task = ResolvedCorrectionTask(
        task_id="task-1", operation=typed.operation, corp_code=typed.corp_code,
        corp_name=typed.corp_name,
        # A correction diff is read at the selected correction observation;
        # history stays bounded by the corpus snapshot.
        as_of=(resolution.corpus_cutoff if ranged_history
               else typed.correction_date if is_diff
               else resolution.corpus_cutoff),
        document_selector=(
            DocumentSelector(
                doc_group="exchange", is_correction=True,
                rcept_from=typed.date_roles.correction_from,
                rcept_to=typed.date_roles.correction_to)
            if ranged_history and typed.date_roles is not None
            else DocumentSelector(rcept_no=typed.correction_receipt,
                                  doc_group="exchange") if is_diff else None),
        # Public Stage2 handoff keeps the verified observable facets.  The
        # exact canonical event key remains resolver-only proof authority,
        # rather than leaking an implementation-local identity into v0.4.
        event_selector=(
            EventSelector(
                event_key=typed.event_key,
                event_from=typed.date_roles.root_observed_at,
                event_to=typed.date_roles.root_observed_at)
            if ranged_history and typed.date_roles is not None
            else None if is_diff else EventSelector(
                counterparty=typed.counterparty,
                keywords=list(typed.product_keywords))),
        requested_slots=(
            list(dict.fromkeys([
                *(change.path for change in typed.changes),
                "before", "after", "정정사유",
            ])) if is_whole_target else ["계약금액"]),
        field_outputs=([FieldOutputSpec(output_id="output-1", slot="계약금액", value_kind="money")]
                       if not is_whole_target else []),
    )
    roots: list[ExecutionAnswerRoot] = []
    supports: list[SupportRequirement] = []
    for index, (item, source) in enumerate(zip(items, intent_items), start=1):
        root_id = f"output-{index}"
        proof_ref = (typed.event_proofs[0].proof_ref
                     if is_whole_target else item.field_proofs[0].proof_ref)
        if is_whole_target:
            roots.append(ExecutionAnswerRoot(root_id=root_id, item_id=item.item_id,
                projection_mode="whole_target", whole_target_id="whole-target-1",
                plan_task_id="task-1", proof_ref=proof_ref))
            field_ids = ["whole-target-1"]
        else:
            roots.append(ExecutionAnswerRoot(root_id=root_id, item_id=item.item_id,
                field_id=f"field-{index}", plan_output_id="output-1" if index == 1 else None,
                plan_task_id=None if index == 1 else "task-1", proof_ref=proof_ref))
            field_ids = [f"field-{index}"]
        supports.append(SupportRequirement(
            support_id=f"support-{index}", kind="coordinate",
            detail=(f"event_key={typed.event_key};root_receipt={typed.root_receipt};"
                    f"correction_receipt={typed.correction_receipt};"
                    f"proof_refs={','.join(p.proof_ref for p in typed.event_proofs)}"),
            applies_to_field_ids=([] if is_whole_target else field_ids),
            applies_to_whole_target_ids=(
                ["whole-target-1"] if is_whole_target else []),
            answer_root_refs=[root_id]))
    # 수치 전제는 그 수치를 실제로 담는 출력에 건다.  `output-1` 이 금액
    # 출력이고, 상태 전제("늘어난")는 견줄 수치가 없어 비운다.
    premise_claims = ([] if is_whole_target else [PremiseClaim(
        claim_id=premise.premise_id, kind=premise.kind, raw_text=premise.raw_text,
        verify_with=([OutputRef(output_id="output-1")]
                     if premise.kind == "numeric" else []),
        verify_tasks=[TaskVerificationRef(task_id="task-1")]) for premise in premises])
    # The task's correction-history result is the verifier; premises remain in
    # execution roots even though their claim text is source intent authority.
    resolved_plan = ResolvedQueryPlan(revision=0, reference_date=resolution.reference_date,
        corpus_cutoff=resolution.corpus_cutoff, tasks=[task], derivations=[],
        premise_claims=premise_claims, applied_defaults=[], presentation=None)
    support_roots = [ExecutionSupportRoot(
        support_id=f"support-{index}", kind="coordinate", item_id=item.item_id,
        projection_mode=("whole_target" if is_whole_target else "named_fields"),
        field_id=(None if is_whole_target else f"field-{index}"),
        whole_target_id=("whole-target-1" if is_whole_target else None),
        root_id=f"output-{index}",
    ) for index, item in enumerate(items, start=1)]
    return ExecutionPlan.create(source_intent_digest=resolution.source_intent_digest,
        resolution_digest=resolution.resolution_digest, canonical_build_id=resolution.canonical_build_id,
        resolver_version=resolution.resolver_version, resolved_plan=resolved_plan,
        applied_defaults=[], answer_roots=roots, support_roots=support_roots,
        premise_roots=[] if is_whole_target else [ExecutionPremiseRoot(
            premise_id=premise.premise_id, root_ids=["output-1", "output-2"],
            task_refs=["task-1"])
            for premise in premises], plan_value_roots=[]), supports


def _compile_correction_diff(question: str, intent: SemanticIntent, resolution: AuthoritativeResolution, intent_item: Any, resolution_item: ResolvedItem) -> DeterministicCompiledSlice:
    source_digest = semantic_intent_digest(intent)
    plan, supports = _build_correction_lineage_plan(resolution, [resolution_item], [intent_item])
    item = CompiledAnswerItem(item_id=intent_item.item_id, status="ready",
        projection=AnswerProjection(projection_mode="whole_target", shape="narrative", fields=[],
            whole_target=WholeTargetBinding(whole_target_id="whole-target-1", binding_status="executable", answer_root_refs=["output-1"]),
            presentation=intent_item.output.presentation),
        coverage=CoveragePartition(required=["whole-target-1"], executable=["whole-target-1"], limited=[]), support_requirements=supports)
    contract = CompiledAnswerContract.build(source_intent=intent, source_intent_digest=source_digest,
        execution_plan_digest=plan.execution_plan_digest, items=[item], groups=[], source_field_bindings={intent_item.item_id: []},
        live_answer_root_refs=["output-1"], live_plan_root_refs=[], live_task_refs=["task-1"], premise_contracts=[], presentation=intent.presentation)
    return DeterministicCompiledSlice.create(question_id=resolution.question_id, question=question, source_intent_digest=source_digest,
        resolution_digest=resolution.resolution_digest, overlay_digest=None, intent=intent, resolution=resolution, execution_plan=plan, answer_contract=contract)


def _compile_correction_history(question: str, intent: SemanticIntent, resolution: AuthoritativeResolution, intent_items: Any, resolution_item: ResolvedItem) -> DeterministicCompiledSlice:
    amount, reason = intent_items
    source_digest = semantic_intent_digest(intent)
    plan, supports = _build_correction_lineage_plan(
        resolution, list(resolution.items), [amount, reason], tuple(intent.premises))
    compiled = []
    for index, source in enumerate((amount, reason), start=1):
        compiled.append(CompiledAnswerItem(item_id=source.item_id, status="ready",
            projection=AnswerProjection(shape=source.output.shape, fields=[ProjectionField(field_id=f"field-{index}", field_key=("amount" if index == 1 else "reason"), binding_status="executable", answer_root_refs=[f"output-{index}"])], presentation=source.output.presentation),
            coverage=CoveragePartition(required=[f"field-{index}"], executable=[f"field-{index}"], limited=[]), support_requirements=[supports[index - 1]]))
    premise_contracts = [CompiledPremiseContract(premise_id=p.premise_id, kind=p.kind, raw_text=p.raw_text, applies_to_item_ids=list(p.applies_to_item_ids), verification_root_refs=["output-1", "output-2"], verification_task_refs=["task-1"], verification_plan_root_refs=[], verdict_requirement="required") for p in intent.premises]
    contract = CompiledAnswerContract.build(source_intent=intent, source_intent_digest=source_digest, execution_plan_digest=plan.execution_plan_digest,
        items=compiled, groups=_source_authority(intent)["source_groups"], source_field_bindings={source.item_id: [SourceFieldBinding(surface=source.output.field_surfaces[0], field_id=f"field-{index}")] for index, source in enumerate((amount, reason), start=1)},
        live_answer_root_refs=["output-1", "output-2"], live_plan_root_refs=[], live_task_refs=["task-1"], premise_contracts=premise_contracts, presentation=intent.presentation)
    return DeterministicCompiledSlice.create(question_id=resolution.question_id, question=question, source_intent_digest=source_digest,
        resolution_digest=resolution.resolution_digest, overlay_digest=None, intent=intent, resolution=resolution, execution_plan=plan, answer_contract=contract)


def _lower_correction_diff(question: str, intent: SemanticIntent, resolution: AuthoritativeResolution, intent_item: Any, resolution_item: ResolvedItem, trace_question_id: str, overlay_digest: str | None) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_correction_diff(question, intent, resolution, intent_item, resolution_item)


def _lower_correction_history(question: str, intent: SemanticIntent, resolution: AuthoritativeResolution, intent_items: Any, resolution_item: ResolvedItem, trace_question_id: str, overlay_digest: str | None) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_correction_history(question, intent, resolution, intent_items, resolution_item)


def _lower_lifecycle_composite(
        question: str, intent: SemanticIntent, resolution: AuthoritativeResolution,
        intent_items: Any, resolution_item: ResolvedItem, trace_question_id: str,
        overlay_digest: str | None) -> DeterministicCompiledSlice:
    del trace_question_id, overlay_digest
    return _compile_lifecycle_composite(
        question, intent, resolution, intent_items, resolution_item)


def compile_stage1_v1_generic(
        question: str,
        intent: SemanticIntent | Mapping[str, Any],
        resolution: AuthoritativeResolution | Mapping[str, Any],
        ) -> DeterministicCompiledSlice:
    """Compile a strict typed Stage1 slice through the qid-free core registry.

    ``question`` is carried into the auditable output.  It does not select a
    lowering.  ``resolution.question_id`` is copied into the compiled trace
    only; the approved exact-ID/text adapter is :func:`compile_stage1_v1`.
    """
    if not isinstance(question, str) or not question:
        raise TypeError("question은 비어 있지 않은 문자열이어야 합니다")
    normalized_intent = _strict_intent(intent)
    normalized_resolution = _strict_resolution(resolution)
    _validate_intent_grounding(question, normalized_intent)
    source_digest = semantic_intent_digest(normalized_intent)
    if normalized_resolution.source_intent_digest != source_digest:
        raise DeterministicPlanCompilerError(
            "resolution source_intent_digest가 semantic intent와 다릅니다")
    handler = _selected_handler(normalized_intent, normalized_resolution)
    intent_item, _ = handler.validate_intent(normalized_intent)
    resolution_item, overlay_digest = handler.validate_resolution(
        intent_item, normalized_resolution)
    return handler.lower(
        question, normalized_intent, normalized_resolution, intent_item,
        resolution_item, normalized_resolution.question_id, overlay_digest)


# The typed/core entry is intentionally aliased under a second descriptive
# name for callers migrating from the legacy adapter.  Both names resolve to
# the same implementation and neither accepts a question_id.
compile_stage1_v1_typed = compile_stage1_v1_generic


def _validate_approved_financial_resolution(
        question_id: str, resolution: AuthoritativeResolution,
        ) -> None:
    """승인된 재무 슬라이스가 **드리프트하지 않았는지** 확인한다.

    좌표 대조는 원래 `_validate_financial_resolution` 안에 있었다. 그런데 그것은
    **일반 경로에 등록된 검증기**여서, 한 문항의 정답이 거기 박혀 있으면 처음 보는
    재무 질문이 좌표가 옳아도 거절됐다.

    그래서 대조는 이 승인 어댑터로 옮긴다. 일반 경로는 구조만 보고, 승인된
    슬라이스의 정확한 좌표는 여기서 지킨다 — `_validate_g_i_009_approved_resolution`
    등과 같은 자리다.
    """

    expected = _demo_resolution(question_id, _demo_intent(question_id))
    if resolution.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise DeterministicPlanCompilerError(
            f"{question_id} approved resolution coordinate가 다릅니다")


# ── 동결 슬라이스 어댑터 (오프라인 전용): 정확한 question ID·원문에 결속된 컴파일 ──────────────────

def compile_stage1_v1(
        question_id: str,
        question: str,
        intent: SemanticIntent | Mapping[str, Any],
        resolution: AuthoritativeResolution | Mapping[str, Any],
        ) -> DeterministicCompiledSlice:
    """Approved fixture/demo adapter around :func:`compile_stage1_v1_generic`.

    Exact question-ID/text checks live here for the checked-in vertical
    slices.  The production lowering path below this adapter never dispatches
    on the supplied ID.
    """
    expected_question = _expected_question(question_id)
    if question != expected_question:
        raise DeterministicPlanCompilerError("question이 승인된 slice 원문과 다릅니다")
    normalized_intent = _strict_intent(intent)
    expected_intent = _demo_intent(question_id)
    if normalized_intent.model_dump(mode="json") != expected_intent.model_dump(
            mode="json"):
        raise DeterministicPlanCompilerError(
            "approved adapter semantic intent가 checked-in expectation과 다릅니다")
    result = compile_stage1_v1_generic(
        question, normalized_intent, resolution)
    if result.question_id != question_id:
        raise DeterministicPlanCompilerError(
            "approved adapter question_id가 selected structural handler와 다릅니다")
    if question_id in (G_A_001, R_A_002):
        _validate_approved_financial_resolution(question_id, result.resolution)
    if question_id == G_A_010:
        _validate_g_a_010_approved_resolution(result.resolution)
    if question_id == G_I_004:
        _validate_g_i_004_approved_resolution(result.resolution)
    if question_id == G_I_006:
        _validate_g_i_006_approved_resolution(result.resolution)
    if question_id == G_I_009:
        _validate_g_i_009_approved_resolution(result.resolution)
    if question_id == G_O_001:
        _validate_g_o_001_approved_resolution(result.resolution)
    return result


def compile_deterministic_plan(
        *, question_id: str, question: str,
        semantic_intent: SemanticIntent | Mapping[str, Any],
        authoritative_resolution: AuthoritativeResolution | Mapping[str, Any],
        ) -> DeterministicCompiledSlice:
    return compile_stage1_v1(
        question_id, question, semantic_intent, authoritative_resolution)


def compile_g_a_001(
        intent: SemanticIntent | Mapping[str, Any],
        resolution: AuthoritativeResolution | Mapping[str, Any],
        ) -> DeterministicCompiledSlice:
    return compile_stage1_v1(G_A_001, G_A_001_QUESTION, intent, resolution)


def compile_g_a_010(
        intent: SemanticIntent | Mapping[str, Any],
        resolution: AuthoritativeResolution | Mapping[str, Any],
        ) -> DeterministicCompiledSlice:
    return compile_stage1_v1(G_A_010, G_A_010_QUESTION, intent, resolution)


def compile_r_a_002(
        intent: SemanticIntent | Mapping[str, Any],
        resolution: AuthoritativeResolution | Mapping[str, Any],
        ) -> DeterministicCompiledSlice:
    return compile_stage1_v1(R_A_002, R_A_002_QUESTION, intent, resolution)


def verify_compiled_slice_digest(
        value: DeterministicCompiledSlice | Mapping[str, Any],
        ) -> str:
    if isinstance(value, DeterministicCompiledSlice):
        payload = value.model_dump(mode="json", warnings=False)
    elif isinstance(value, Mapping):
        payload = value
    else:
        raise TypeError("compiled slice는 model 또는 mapping이어야 합니다")
    validated = load_compiled_slice_json(canonical_json(payload))
    return validated.bundle_digest


def load_compiled_slice_json(
        payload: str | bytes | bytearray,
        ) -> DeterministicCompiledSlice:
    return DeterministicCompiledSlice.model_validate_json(payload, strict=True)


# ── 동결 슬라이스 기대 wire·resolution (오프라인 fixture 재료 — 런타임 미참조) ──────────────────────

def _demo_wire(question_id: str) -> HcxSemanticIntentWire:
    _expected_question(question_id)      # 지원 슬라이스인지 검증 (미지원이면 예외)
    if question_id == G_I_009:
        return HcxSemanticIntentWire.model_validate({
            "schema_version": HCX_SEMANTIC_INTENT_WIRE_V1,
            "entities": [
                {"kind_hint": "company", "surface": "두산퓨얼셀"},
            ],
            "answer_items": [
                {
                    "target": {
                        "kind": "event", "surface": "계약",
                        "entity_indexes": [0], "qualifier_surfaces": [],
                    },
                    "operation": "retrieve",
                    "scope": {
                        "target_period_expressions": [],
                        "as_of_expression": "",
                        "document_group_expression": "",
                        "scope_qualifier_expressions": [],
                    },
                    "selection": {
                        "mode": "none", "criterion_surface": "", "k": 0,
                    },
                    "output": {
                        "shape": "scalar", "projection_mode": "named_fields",
                        "field_surfaces": ["해지된 이유"],
                        "presentation": "auto",
                    },
                },
                {
                    "target": {
                        "kind": "event", "surface": "계약",
                        "entity_indexes": [0], "qualifier_surfaces": [],
                    },
                    "operation": "retrieve",
                    "scope": {
                        "target_period_expressions": [],
                        "as_of_expression": "",
                        "document_group_expression": "",
                        "scope_qualifier_expressions": [],
                    },
                    "selection": {
                        "mode": "none", "criterion_surface": "", "k": 0,
                    },
                    "output": {
                        "shape": "scalar",
                        "projection_mode": "named_fields",
                        "field_surfaces": ["계약 효력발생 조건"],
                        "presentation": "auto",
                    },
                },
            ],
            "answer_groups": [{"item_indexes": [0, 1]}],
            "premises": [], "unresolved_mentions": [], "presentation": "auto",
        })
    if question_id == G_I_006:
        return HcxSemanticIntentWire.model_validate({
            "schema_version": HCX_SEMANTIC_INTENT_WIRE_V1,
            "entities": [
                {"kind_hint": "event", "surface": "Freudenberg 계약"},
            ],
            "answer_items": [
                {
                    "target": {
                        "kind": "event", "surface": "Freudenberg 계약",
                        "entity_indexes": [0], "qualifier_surfaces": [],
                    },
                    "operation": "retrieve",
                    "scope": {
                        "target_period_expressions": [],
                        "as_of_expression": "",
                        "document_group_expression": "",
                        "scope_qualifier_expressions": [],
                    },
                    "selection": {
                        "mode": "none", "criterion_surface": "", "k": 0,
                    },
                    "output": {
                        "shape": "verdict",
                        "projection_mode": "named_fields",
                        "field_surfaces": [
                            "정정 후 계약금액과 해지금액은 같으며",
                        ],
                        "presentation": "auto",
                    },
                },
                {
                    "target": {
                        "kind": "event", "surface": "Freudenberg 계약",
                        "entity_indexes": [0], "qualifier_surfaces": [],
                    },
                    "operation": "retrieve",
                    "scope": {
                        "target_period_expressions": [],
                        "as_of_expression": "",
                        "document_group_expression": "",
                        "scope_qualifier_expressions": [],
                    },
                    "selection": {
                        "mode": "none", "criterion_surface": "", "k": 0,
                    },
                    "output": {
                        "shape": "narrative", "projection_mode": "named_fields",
                        "field_surfaces": ["왜 다른가"],
                        "presentation": "auto",
                    },
                },
            ],
            "answer_groups": [{"item_indexes": [0, 1]}],
            "premises": [{
                "kind": "comparison",
                "raw_text": "정정 후 계약금액과 해지금액은 같으며",
                "applies_to_item_indexes": [0, 1],
            }],
            "unresolved_mentions": [], "presentation": "auto",
        })
    if question_id == G_I_004:
        return HcxSemanticIntentWire.model_validate({
            "schema_version": HCX_SEMANTIC_INTENT_WIRE_V1,
            "entities": [
                {"kind_hint": "counterparty", "surface": "Ford"},
            ],
            "answer_items": [
                {
                    "target": {
                        "kind": "document", "surface": "공시",
                        "entity_indexes": [0], "qualifier_surfaces": [],
                    },
                    "operation": "retrieve",
                    "scope": {
                        "target_period_expressions": [],
                        "as_of_expression": "2025년 12월 17일까지",
                        "document_group_expression": "",
                        "scope_qualifier_expressions": [],
                    },
                    "selection": {
                        "mode": "latest", "criterion_surface": "최신", "k": 0,
                    },
                    "output": {
                        "shape": "narrative", "projection_mode": "named_fields",
                        "field_surfaces": ["내용"],
                        "presentation": "auto",
                    },
                },
                {
                    "target": {
                        "kind": "event", "surface": "계약",
                        "entity_indexes": [0], "qualifier_surfaces": [],
                    },
                    "operation": "retrieve",
                    "scope": {
                        "target_period_expressions": [],
                        "as_of_expression": "",
                        "document_group_expression": "",
                        "scope_qualifier_expressions": ["제공 코퍼스 기준"],
                    },
                    "selection": {
                        "mode": "none", "criterion_surface": "", "k": 0,
                    },
                    "output": {
                        "shape": "scalar", "projection_mode": "named_fields",
                        "field_surfaces": ["최종 상태"],
                        "presentation": "auto",
                    },
                },
            ],
            "answer_groups": [{"item_indexes": [0, 1]}],
            "premises": [], "unresolved_mentions": [], "presentation": "auto",
        })
    if question_id == G_A_004:
        return HcxSemanticIntentWire.model_validate({
            "schema_version": HCX_SEMANTIC_INTENT_WIRE_V1,
            "entities": [
                {"kind_hint": "company", "surface": "삼성전자"},
                {"kind_hint": "company", "surface": "SK하이닉스"},
            ],
            "answer_items": [{
                "target": {
                    "kind": "metric", "surface": "매출액",
                    "entity_indexes": [0, 1], "qualifier_surfaces": [],
                },
                "operation": "compare",
                "scope": {
                    "target_period_expressions": ["2025년"],
                    "as_of_expression": "", "document_group_expression": "",
                    "scope_qualifier_expressions": ["연결"],
                },
                "selection": {
                    "mode": "maximum", "criterion_surface": "큰", "k": 0,
                },
                "output": {
                    "shape": "comparison", "projection_mode": "named_fields",
                    "field_surfaces": ["큰 기업", "차이"],
                    "presentation": "auto",
                },
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        })
    if question_id == G_A_010:
        return HcxSemanticIntentWire.model_validate({
            "schema_version": HCX_SEMANTIC_INTENT_WIRE_V1,
            "entities": [
                {"kind_hint": "company", "surface": G_A_010_CORP_NAME},
            ],
            "answer_items": [{
                "target": {
                    "kind": "topic", "surface": "주요 투자계획",
                    "entity_indexes": [0], "qualifier_surfaces": [],
                },
                "operation": "retrieve",
                "scope": {
                    "target_period_expressions": [],
                    "as_of_expression": "",
                    "document_group_expression": "2026년 1분기보고서",
                    "scope_qualifier_expressions": [],
                },
                "selection": {
                    "mode": "none", "criterion_surface": "", "k": 0,
                },
                "output": {
                    "shape": "record_list",
                    "projection_mode": "named_fields",
                    "field_surfaces": ["투자 대상", "목적", "금액", "기간"],
                    "presentation": "auto",
                },
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        })
    if question_id == G_O_001:
        return HcxSemanticIntentWire.model_validate({
            "schema_version": HCX_SEMANTIC_INTENT_WIRE_V1,
            "entities": [
                {"kind_hint": "company", "surface": "삼성전자"},
            ],
            "answer_items": [{
                "target": {
                    "kind": "topic",
                    "surface": "사업부문·주요 제품 및 서비스·매출구성",
                    "entity_indexes": [0], "qualifier_surfaces": [],
                },
                "operation": "compare",
                "scope": {
                    "target_period_expressions": ["2023년", "2025년"],
                    "as_of_expression": "",
                    "document_group_expression": "사업보고서",
                    "scope_qualifier_expressions": [],
                },
                "selection": {
                    "mode": "none", "criterion_surface": "", "k": 0,
                },
                "output": {
                    "shape": "narrative",
                    "projection_mode": "named_fields",
                    "field_surfaces": [
                        "사업부문", "주요 제품 및 서비스", "매출구성", "핵심 변화",
                    ],
                    "presentation": "auto",
                },
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        })
    target_surface = "매출액" if question_id == G_A_001 else "매출"
    qualifiers = ["연결기준"] if question_id == G_A_001 else []
    return HcxSemanticIntentWire.model_validate({
        "schema_version": HCX_SEMANTIC_INTENT_WIRE_V1,
        "entities": [{"kind_hint": "company", "surface": "삼성전자"}],
        "answer_items": [{
            "target": {
                "kind": "metric", "surface": target_surface,
                "entity_indexes": [0], "qualifier_surfaces": [],
            },
            "operation": "retrieve",
            "scope": {
                "target_period_expressions": ["2025년"],
                "as_of_expression": "", "document_group_expression": "",
                "scope_qualifier_expressions": qualifiers,
            },
            "selection": {"mode": "none", "criterion_surface": "", "k": 0},
            "output": {
                "shape": "scalar", "projection_mode": "named_fields",
                "field_surfaces": ["얼마"],
                "presentation": "auto",
            },
        }],
        "answer_groups": [], "premises": [], "unresolved_mentions": [],
        "presentation": "auto",
    })


def _demo_intent(question_id: str) -> SemanticIntent:
    question = _expected_question(question_id)
    return normalize_semantic_intent(question, _demo_wire(question_id))


def _demo_resolution(
        question_id: str, intent: SemanticIntent,
        ) -> AuthoritativeResolution:
    if question_id == G_I_009:
        return _demo_g_i_009_resolution(intent)
    if question_id == G_I_006:
        return _demo_g_i_006_resolution(intent)
    if question_id == G_I_004:
        return _demo_g_i_004_resolution(intent)
    if question_id == G_A_004:
        return _demo_g_a_004_resolution(intent)
    if question_id == G_A_010:
        return _demo_g_a_010_resolution(intent)
    if question_id == G_O_001:
        return _demo_g_o_001_resolution(intent)
    defaults: list[AppliedDefault] = []
    if question_id == R_A_002:
        defaults = [AppliedDefault(
            policy="primary_statement_scope",
            value="CFS",
            basis=R_A_002_DEFAULT_BASIS,
            evidence_refs=list(R_A_002_DEFAULT_EVIDENCE_REFS),
        )]
    item = intent.answer_items[0]
    resolution = FinancialResolution(
        corp_code="00126380", corp_name="삼성전자",
        concept=FinancialConcept.REVENUE,
        period_start=date(2025, 1, 1), period_end=date(2025, 12, 31),
        period_type="annual", scope="CFS", statement="IS",
        view="restated", as_of="20260619", cumulative=True,
    )
    return AuthoritativeResolution.create(
        question_id=question_id,
        source_intent_digest=semantic_intent_digest(intent),
        canonical_build_id="a" * 32,
        resolver_version="stage1-resolver/1.0",
        reference_date=date(2026, 6, 19), corpus_cutoff="20260619",
        items=[ResolvedItem(
            item_id=item.item_id,
            target_surface=item.target.surface,
            resolution=resolution,
            field_proofs=[ResolutionFieldProof(
                source_field_index=0, surface=item.output.field_surfaces[0],
                proof_ref="source-field:item-1:0",
            )],
            applied_defaults=defaults,
        )],
        premise_proofs=[],
    )


def _demo_g_i_004_resolution(intent: SemanticIntent) -> AuthoritativeResolution:
    return AuthoritativeResolution.create(
        question_id=G_I_004,
        source_intent_digest=semantic_intent_digest(intent),
        canonical_build_id="a" * 32,
        resolver_version="stage1-resolver/1.0",
        reference_date=date(2026, 6, 19), corpus_cutoff="20260619",
        items=[
            ResolvedItem(
                item_id="item-1", target_surface="공시",
                resolution=SameDayDocumentCandidatesResolution(
                    issuer_corp_code="01515323", issuer_corp_name="LG에너지솔루션",
                    as_of="20251217",
                    candidates=[
                        SameDayDocumentCandidate(
                            rcept_no="20251217800800", document_kind="termination",
                            proof_ref="source-document:20251217800800"),
                        SameDayDocumentCandidate(
                            rcept_no="20251217800853", document_kind="correction",
                            proof_ref="source-document:20251217800853"),
                    ],
                    ordering_provenance=LimitationProvenance(
                        code="intraday_order_unavailable", family="ordering",
                        detail=G_I_004_ORDERING_DETAIL,
                        evidence_refs=list(G_I_004_ORDERING_EVIDENCE_REFS),
                    ),
                ),
                field_proofs=[ResolutionFieldProof(
                    source_field_index=0, surface="내용",
                    proof_ref="source-field:item-1:0",
                )],
            ),
            ResolvedItem(
                item_id="item-2", target_surface="계약",
                resolution=TerminationReportedStatusResolution(
                    issuer_corp_code="01515323",
                    issuer_corp_name="LG에너지솔루션",
                    status_receipt="20251217800800",
                    event_key="45eaec69e2677af016187d5523c48439",
                    event_key_proof=ResolutionSourceProof(
                        source_receipt="20251217800800",
                        proof_ref="source-event-key:20251217800800",
                    ),
                    status_proof=ResolutionSourceProof(
                        source_receipt="20251217800800",
                        proof_ref="source-status:20251217800800",
                    ),
                    identity_provenance=LimitationProvenance(
                        code="ambiguous_event_origin", family="identity_lineage",
                        detail=G_I_004_IDENTITY_DETAIL,
                        evidence_refs=list(G_I_004_IDENTITY_EVIDENCE_REFS),
                        original_receipts=["20241015800258", "20241015800261"],
                    ),
                ),
                field_proofs=[ResolutionFieldProof(
                    source_field_index=0, surface="최종 상태",
                    proof_ref="source-field:item-2:0",
                )],
            ),
        ], premise_proofs=[],
    )


def _demo_g_i_006_resolution(intent: SemanticIntent) -> AuthoritativeResolution:
    """Exact approved G-I-006 coordinates, isolated from generic lowering."""
    return AuthoritativeResolution.create(
        question_id=G_I_006,
        source_intent_digest=semantic_intent_digest(intent),
        canonical_build_id="a" * 32,
        resolver_version="stage1-resolver/1.0",
        reference_date=date(2026, 6, 19), corpus_cutoff="20260619",
        items=[
            ResolvedItem(
                item_id="item-1", target_surface="Freudenberg 계약",
                resolution=DocumentFactComparisonResolution(operands=[
                    DocumentFactOperand(
                        operand_id="operand-1",
                        issuer_corp_code="01515323",
                        issuer_corp_name="LG에너지솔루션",
                        source_class="correction",
                        doc_id="exchange_20251226800767",
                        receipt_no="20251226800767",
                        path="2. 계약내역 > 계약금액(원)",
                        locator="TABLE[3]/TBODY[0]/TR[2]/TD[2]",
                        source_file_id="2daec4ae341769ce1570af9b3b38ebcd",
                        evidence_id="c9858d6fe020992832d655dfcbdb7318",
                    ),
                    DocumentFactOperand(
                        operand_id="operand-2",
                        issuer_corp_code="01515323",
                        issuer_corp_name="LG에너지솔루션",
                        source_class="disclosure",
                        doc_id="exchange_20251226800706",
                        receipt_no="20251226800706",
                        path="2. 해지내역 > 해지금액(원)",
                        locator="TABLE[0]/TBODY[0]/TR[2]/TD[2]",
                        source_file_id="81da2131f5aa77e9fda1e9a2fa77eada",
                        evidence_id="ea48d491bf3d342228762e0ab6d3991f",
                    ),
                ]),
                field_proofs=[ResolutionFieldProof(
                    source_field_index=0,
                    surface="정정 후 계약금액과 해지금액은 같으며",
                    proof_ref="source-field:item-1:0",
                )],
            ),
            ResolvedItem(
                item_id="item-2", target_surface="Freudenberg 계약",
                resolution=DocumentReasonEvidenceResolution(
                    evidence=DocumentReasonEvidence(
                        operand_id="operand-2",
                        issuer_corp_code="01515323",
                        issuer_corp_name="LG에너지솔루션",
                        source_class="disclosure",
                        doc_id="exchange_20251226800706",
                        receipt_no="20251226800706",
                        path="8. 기타 투자판단과 관련한 중요사항",
                        locator="TABLE[0]/TBODY[0]/TR[14]/TD[1]",
                        source_file_id="81da2131f5aa77e9fda1e9a2fa77eada",
                        evidence_id="0a90701ead2898d351f9da450e317847",
                    ),
                ),
                field_proofs=[ResolutionFieldProof(
                    source_field_index=0, surface="왜 다른가",
                    proof_ref="source-field:item-2:0",
                )],
            ),
        ],
        premise_proofs=[ResolutionPremiseProof(
            premise_id="premise-1",
            proof_refs=[
                "c9858d6fe020992832d655dfcbdb7318",
                "ea48d491bf3d342228762e0ab6d3991f",
            ],
        )],
    )


def _demo_g_i_009_resolution(intent: SemanticIntent) -> AuthoritativeResolution:
    """Exact approved G-I-009 coordinates, isolated from generic lowering."""
    approved = [
        DocumentAttributeEvidence(
            issuer_corp_code="01412725", issuer_corp_name="두산퓨얼셀",
            document_role="later",
            doc_id="exchange_20250402800768",
            receipt_no="20250402800768",
            path="5. 해지 주요사유",
            locator="TABLE[0]/TBODY[0]/TR[10]/TD[1]",
            source_file_id="af55a3ac349da577311a38f4a00b1fe8",
            evidence_id="f1994ac666ab08b264bbd47366002fda",
        ),
        DocumentAttributeEvidence(
            issuer_corp_code="01412725", issuer_corp_name="두산퓨얼셀",
            document_role="earlier",
            doc_id="exchange_20231006800130",
            receipt_no="20231006800130",
            path="9. 기타 투자판단과 관련한 중요사항",
            locator="TABLE[0]/TBODY[0]/TR[16]/TD[0]",
            source_file_id="340ede957a6b679fea415cb2dbed82b0",
            evidence_id="b4481081263b078817549657bcea5393",
        ),
    ]
    return AuthoritativeResolution.create(
        question_id=G_I_009,
        source_intent_digest=semantic_intent_digest(intent),
        canonical_build_id="a" * 32,
        resolver_version="stage1-resolver/1.0",
        reference_date=date(2026, 6, 19), corpus_cutoff="20260619",
        items=[
            ResolvedItem(
                item_id=f"item-{index}",
                target_surface=intent.answer_items[index - 1].target.surface,
                resolution=DocumentAttributeEvidenceResolution(
                    evidence=evidence),
                field_proofs=[ResolutionFieldProof(
                    source_field_index=0,
                    surface=intent.answer_items[index - 1].output.field_surfaces[0],
                    proof_ref=f"source-field:item-{index}:0",
                )],
            )
            for index, evidence in enumerate(approved, start=1)
        ],
        premise_proofs=[],
    )


def _demo_g_o_001_resolution(intent: SemanticIntent) -> AuthoritativeResolution:
    """Exact six report-section coordinates for the approved G-O-001 slice."""
    source_rows = [
        (
            0, "periodic_20240312000736", "20240312000736",
            date(2023, 1, 1), date(2023, 12, 31),
            "d90027283461c8da56846716e009b074",
            [
                (0, "business_segments", "II. 사업의 내용 > 1. 사업의 개요",
                 "BODY[0]/SECTION-1[2]/LIBRARY[0]/SECTION-2[0]/TITLE[0]/PART[0]",
                 "58fa81bdaad0f10df735453f46bfe001"),
                (1, "products_services", "II. 사업의 내용 > 2. 주요 제품 및 서비스",
                 "BODY[0]/SECTION-1[2]/LIBRARY[0]/SECTION-2[1]/TITLE[0]/PART[0]",
                 "5bb6823bc687fc2ac9a368c259425fd6"),
                (2, "sales_mix", "II. 사업의 내용 > 4. 매출 및 수주상황",
                 "BODY[0]/SECTION-1[2]/LIBRARY[0]/SECTION-2[3]/TITLE[0]/PART[0]",
                 "742866eb17bcf1f316c1d29df7746c56"),
            ],
        ),
        (
            1, "periodic_20260310002820", "20260310002820",
            date(2025, 1, 1), date(2025, 12, 31),
            "7f5865931f7b2520c77b5d87f829a022",
            [
                (0, "business_segments", "II. 사업의 내용 > 1. 사업의 개요",
                 "BODY[0]/SECTION-1[2]/LIBRARY[0]/SECTION-2[0]/TITLE[0]/PART[0]",
                 "80703106bee3ac4841fd8a3530fbdc5b"),
                (1, "products_services", "II. 사업의 내용 > 2. 주요 제품 및 서비스",
                 "BODY[0]/SECTION-1[2]/LIBRARY[0]/SECTION-2[1]/TITLE[0]/PART[0]",
                 "264bcdbb4a54ddafda352623d1c6b6e0"),
                (2, "sales_mix", "II. 사업의 내용 > 4. 매출 및 수주상황",
                 "BODY[0]/SECTION-1[2]/LIBRARY[0]/SECTION-2[3]/TITLE[0]/PART[0]",
                 "acd6daf3b73c06c574866ba96fba10fb"),
            ],
        ),
    ]
    documents = []
    for (period_index, doc_id, receipt_no, period_start, period_end,
         source_file_id, evidence_rows) in source_rows:
        documents.append(PeriodicNarrativeDocument(
            source_period_index=period_index,
            issuer_corp_code="00126380", issuer_corp_name="삼성전자",
            doc_id=doc_id, receipt_no=receipt_no,
            period_start=period_start, period_end=period_end,
            source_file_id=source_file_id,
            evidence=[PeriodicNarrativeEvidence(
                source_field_index=field_index, axis_id=axis_id,
                path=path, locator=locator,
                source_file_id=source_file_id, evidence_id=evidence_id,
            ) for field_index, axis_id, path, locator, evidence_id
                in evidence_rows],
        ))
    item = intent.answer_items[0]
    return AuthoritativeResolution.create(
        question_id=G_O_001,
        source_intent_digest=semantic_intent_digest(intent),
        canonical_build_id="a" * 32,
        resolver_version="stage1-resolver/1.0",
        reference_date=date(2026, 6, 19), corpus_cutoff="20260619",
        items=[ResolvedItem(
            item_id=item.item_id, target_surface=item.target.surface,
            resolution=PeriodicNarrativeComparisonResolution(
                document_group="사업보고서", documents=documents),
            field_proofs=[ResolutionFieldProof(
                source_field_index=index, surface=surface,
                proof_ref=f"source-field:item-1:{index}",
            ) for index, surface in enumerate(item.output.field_surfaces)],
        )],
        premise_proofs=[],
    )


def _demo_g_a_004_resolution(intent: SemanticIntent) -> AuthoritativeResolution:
    item = intent.answer_items[0]
    operands = [
        FinancialComparisonOperand(
            operand_id="operand-1", corp_code="00126380", corp_name="삼성전자",
            concept=FinancialConcept.REVENUE,
            period_start=date(2025, 1, 1), period_end=date(2025, 12, 31),
            period_type="annual", scope="CFS", statement="IS",
            view="restated", as_of="20260619", cumulative=True,
            proof_ref="source-operand:operand-1",
        ),
        FinancialComparisonOperand(
            operand_id="operand-2", corp_code="00164779", corp_name="SK하이닉스",
            concept=FinancialConcept.REVENUE,
            period_start=date(2025, 1, 1), period_end=date(2025, 12, 31),
            period_type="annual", scope="CFS", statement="IS",
            view="restated", as_of="20260619", cumulative=True,
            proof_ref="source-operand:operand-2",
        ),
    ]
    return AuthoritativeResolution.create(
        question_id=G_A_004,
        source_intent_digest=semantic_intent_digest(intent),
        canonical_build_id="a" * 32,
        resolver_version="stage1-resolver/1.0",
        reference_date=date(2026, 6, 19), corpus_cutoff="20260619",
        items=[ResolvedItem(
            item_id=item.item_id, target_surface=item.target.surface,
            resolution=FinancialComparisonResolution(
                operands=operands,
                requested_operators=["argmax", "absolute_difference"],
            ),
            field_proofs=[
                ResolutionFieldProof(
                    source_field_index=0, surface="큰 기업",
                    proof_ref="source-field:item-1:0",
                ),
                ResolutionFieldProof(
                    source_field_index=1, surface="차이",
                    proof_ref="source-field:item-1:1",
                ),
            ],
        )], premise_proofs=[],
    )


def _demo_g_a_010_resolution(intent: SemanticIntent) -> AuthoritativeResolution:
    item = intent.answer_items[0]
    typed = PeriodicDocumentNarrativeResolution(
        corp_code=G_A_010_CORP_CODE,
        corp_name=G_A_010_CORP_NAME,
        document_id=G_A_010_DOCUMENT_ID,
        receipt_no=G_A_010_RECEIPT_NO,
        document_proof=ResolutionSourceProof(
            source_receipt=G_A_010_RECEIPT_NO,
            proof_ref=G_A_010_DOCUMENT_PROOF,
        ),
        narrative_proof=ResolutionSourceProof(
            source_receipt=G_A_010_RECEIPT_NO,
            proof_ref=G_A_010_NARRATIVE_PROOF,
        ),
        source_retrieval_query="설비 투자 현황 및 계획",
        executable_field_indexes=[0, 1, 2, 3],
        limited_field_indexes=[],
        source_cross_check_provenance=None,
    )
    return AuthoritativeResolution.create(
        question_id=G_A_010,
        source_intent_digest=semantic_intent_digest(intent),
        canonical_build_id="a" * 32,
        resolver_version="stage1-resolver/1.0",
        reference_date=date(2026, 6, 19), corpus_cutoff="20260619",
        items=[ResolvedItem(
            item_id=item.item_id, target_surface=item.target.surface,
            resolution=typed,
            field_proofs=[ResolutionFieldProof(
                source_field_index=index, surface=surface,
                proof_ref=f"source-field:item-1:{index}",
            ) for index, surface in enumerate(
                ["투자 대상", "목적", "금액", "기간"])
            ],
    )], premise_proofs=[],
    )


_STAGE1_V1_STRUCTURAL_SPECS = (
    (
        "correction/whole-target-diff",
        _structural_signature_from_payload({
            "entities": [], "answer_items": [{
                "target": {"kind": "document", "entity_positions": [], "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 1, "has_as_of": False,
                          "has_document_group": False, "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "narrative", "projection_mode": "whole_target",
                           "field_count": 0, "presentation": "auto"},
            }], "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("correction_lineage",), _validate_correction_diff_intent,
        _validate_correction_diff_resolution, _lower_correction_diff,
    ),
    (
        "correction/whole-target-diff-company",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}], "answer_items": [{
                "target": {"kind": "document", "entity_positions": [0],
                           "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 1, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "narrative",
                           "projection_mode": "whole_target",
                           "field_count": 0, "presentation": "auto"},
            }], "answer_groups": [], "premises": [],
            "unresolved_mentions": [], "presentation": "auto",
        }),
        ("correction_lineage",), _validate_correction_diff_intent,
        _validate_correction_diff_resolution, _lower_correction_diff,
    ),
    (
        "correction/amount-history-with-reason",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}, {"kind_hint": "company"}],
            "answer_items": [
                {"target": {"kind": "metric", "entity_positions": [0, 1], "qualifier_count": 0},
                 "operation": "retrieve", "scope": {"target_period_count": 0, "has_as_of": False, "has_document_group": False, "scope_qualifier_count": 0},
                 "selection": None, "output": {"shape": "scalar", "projection_mode": "named_fields", "field_count": 1, "presentation": "auto"}},
                {"target": {"kind": "event", "entity_positions": [0, 1], "qualifier_count": 0},
                 "operation": "retrieve", "scope": {"target_period_count": 0, "has_as_of": False, "has_document_group": False, "scope_qualifier_count": 0},
                 "selection": None, "output": {"shape": "narrative", "projection_mode": "named_fields", "field_count": 1, "presentation": "auto"}},
            ],
            "answer_groups": [[0, 1]],
            "premises": [{"kind": "numeric", "item_positions": [0]}, {"kind": "state", "item_positions": [0, 1]}],
            "unresolved_mentions": [], "presentation": "auto",
        }),
        ("correction_lineage", "correction_lineage"), _validate_correction_history_intent,
        _validate_correction_history_resolution, _lower_correction_history,
    ),
    (
        "event/canonical-collection",
        _structural_signature_from_payload({
            "entities": [],
            "answer_items": [{
                "target": {"kind": "event", "entity_positions": [],
                           "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "record_list", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("event_collection",), _validate_event_collection_intent,
        _validate_event_collection_resolution, _lower_event_collection,
    ),
    (
        "event/clarified-amount-change",
        _structural_signature_from_payload({
            "entities": [],
            "answer_items": [{
                "target": {"kind": "metric", "entity_positions": [],
                           "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "scalar", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("event_amount_change",), _validate_event_amount_change_intent,
        _validate_event_amount_change_resolution,
        _lower_event_amount_change,
    ),
    (
        "financial/clarified-company",
        _structural_signature_from_payload({
            "entities": [],
            "answer_items": [{
                "target": {"kind": "metric", "entity_positions": [],
                           "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 1, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "scalar", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("financial",), _validate_clarification_financial_intent,
        _validate_financial_resolution, _lower_financial,
    ),
    (
        "financial/clarified-period",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "metric", "entity_positions": [0],
                           "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "scalar", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("financial",), _validate_clarification_financial_intent,
        _validate_financial_resolution, _lower_financial,
    ),
    (
        "financial/clarified-company-period",
        _structural_signature_from_payload({
            "entities": [],
            "answer_items": [{
                "target": {"kind": "metric", "entity_positions": [],
                           "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "scalar", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("financial",), _validate_clarification_financial_intent,
        _validate_financial_resolution, _lower_financial,
    ),
    (
        "financial/annual-consolidated",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "metric", "entity_positions": [0],
                            "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 1, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 1},
                "selection": None,
                "output": {"shape": "scalar", "projection_mode": "named_fields",
                           "field_count": 1,
                           "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("financial",), _validate_financial_intent,
        _validate_financial_resolution, _lower_financial,
    ),
    (
        "financial/annual-scope-default",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "metric", "entity_positions": [0],
                            "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 1, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "scalar", "projection_mode": "named_fields",
                           "field_count": 1,
                           "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("financial",), _validate_financial_intent,
        _validate_financial_resolution, _lower_financial,
    ),
    (
        "financial/premise-aware-retrieve",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "metric", "entity_positions": [0],
                           "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 1, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 1},
                "selection": None,
                "output": {"shape": "scalar", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [],
            "premises": [{"kind": "existence", "item_positions": [0]}],
            "unresolved_mentions": [], "presentation": "auto",
        }),
        ("financial",), _validate_premise_financial_intent,
        _validate_premise_financial_resolution, _lower_premise_financial,
    ),
    (
        "financial/parallel-scope-retrieval",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [
                {
                    "target": {"kind": "metric", "entity_positions": [0],
                               "qualifier_count": 0},
                    "operation": "retrieve",
                    "scope": {"target_period_count": 1, "has_as_of": False,
                              "has_document_group": False,
                              "scope_qualifier_count": 1},
                    "selection": None,
                    "output": {"shape": "scalar",
                               "projection_mode": "named_fields",
                               "field_count": 1, "presentation": "auto"},
                },
                {
                    "target": {"kind": "metric", "entity_positions": [0],
                               "qualifier_count": 0},
                    "operation": "retrieve",
                    "scope": {"target_period_count": 1, "has_as_of": False,
                              "has_document_group": False,
                              "scope_qualifier_count": 1},
                    "selection": None,
                    "output": {"shape": "scalar",
                               "projection_mode": "named_fields",
                               "field_count": 1, "presentation": "auto"},
                },
            ],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("financial", "financial"),
        _validate_parallel_financial_retrieval_intent,
        _validate_parallel_financial_retrieval_resolution,
        _lower_parallel_financial_retrieval,
    ),
    # 이슈 #94 25 — 「얼마나 벌었어」·「빚이 얼마야」. 되묻던 요약 구어의 후보를
    # 나란히 답한다. `SummaryMetricFanoutRegrounder` 가 item 하나를 후보 수만큼
    # 복제하므로 형제 셋의 모양이 완전히 같다.
    #
    # scope 수식(「연결」)이 있는 판과 없는 판을 **둘 다** 등록한다. 표면 하나로
    # 물은 질문은 대개 scope 를 밝히지 않지만(0), 밝히면 세 item 이 그것을 함께
    # 물려받아(1) signature 가 갈린다. 나머지 축은 같다.
    *(
        (
            f"financial/summary-metric-fanout-{scope_qualifier_count}",
            _structural_signature_from_payload({
                "entities": [{"kind_hint": "company"}],
                "answer_items": [
                    {
                        "target": {"kind": "metric", "entity_positions": [0],
                                   "qualifier_count": 0},
                        "operation": "retrieve",
                        "scope": {"target_period_count": 1,
                                  "has_as_of": False,
                                  "has_document_group": False,
                                  "scope_qualifier_count":
                                      scope_qualifier_count},
                        "selection": None,
                        "output": {"shape": "scalar",
                                   "projection_mode": "named_fields",
                                   "field_count": 1, "presentation": "auto"},
                    },
                ] * 3,
                "answer_groups": [], "premises": [], "unresolved_mentions": [],
                "presentation": "auto",
            }),
            ("financial", "financial", "financial"),
            _validate_summary_metric_fanout_intent,
            _validate_summary_metric_fanout_resolution,
            _lower_summary_metric_fanout,
        )
        for scope_qualifier_count in (0, 1)
    ),
    # 이슈 #58 2단계 — 「최근 N년」「최근 N개 분기」. `RecentPeriodsFanoutRegrounder`
    # 가 item 하나를 N개(2~5)로 복제해 같은 리터럴 기간 표현을 반복하고,
    # 형제 순서로 실제 연도·분기를 나눠 맡긴다.
    #
    # 위 개념 fanout과 항목 모양이 구조적으로 같으므로(회사 하나·재무
    # 스칼라 하나·같은 표면 반복) top-level presentation을 ``"table"`` 로
    # 못박아 그 규격과 signature가 겹치지 않게 한다 — 스키마는 건드리지
    # 않는다(``presentation`` 은 이미 `agent/semantic_intent_v1.py` 의
    # ``Literal["auto", "prose", "table", "list"]`` 로 허용된 값이다).
    #
    # scope 수식(「연결」)이 있는 판과 없는 판을 개념 fanout과 같은 이유로
    # **둘 다** 등록한다(「최근 3개년 연결 영업이익」처럼 흔히 같이 온다).
    #
    # ``shape`` 도 둘 다 등록한다 — HCX-007 실호출로 직접 확인했다: 같은
    # 「최근 N…」류 질문도 판마다 ``scalar``(예: 「최근 3개년 …」)와
    # ``timeline``(예: 「… 추이」)을 오간다. `RecentPeriodsFanoutRegrounder`
    # 는 원래 값을 그대로 복제하므로(개념도 지어내지 않듯 shape도 지어내지
    # 않는다) 등록도 둘 다 받아야 한다.
    *(
        (
            f"financial/recent-periods-fanout-{item_count}-"
            f"{scope_qualifier_count}-{shape}",
            _structural_signature_from_payload({
                "entities": [{"kind_hint": "company"}],
                "answer_items": [
                    {
                        "target": {"kind": "metric", "entity_positions": [0],
                                   "qualifier_count": 0},
                        "operation": "retrieve",
                        "scope": {"target_period_count": 1,
                                  "has_as_of": False,
                                  "has_document_group": False,
                                  "scope_qualifier_count":
                                      scope_qualifier_count},
                        "selection": None,
                        "output": {"shape": shape,
                                   "projection_mode": "named_fields",
                                   "field_count": 1, "presentation": "auto"},
                    },
                ] * item_count,
                "answer_groups": [], "premises": [], "unresolved_mentions": [],
                "presentation": "table",
            }),
            ("financial",) * item_count,
            _validate_recent_periods_fanout_intent,
            _validate_recent_periods_fanout_resolution,
            _lower_recent_periods_fanout,
        )
        for item_count in (2, 3, 4, 5)
        for scope_qualifier_count in (0, 1)
        for shape in ("scalar", "timeline")
    ),
    # 같은 semantic topology가 연말 경계를 지나면 resolution inventory만
    # 달라진다. Q1~Q3은 직접 fact, Q4는 FY-9M 누계 파생이며 가능한 순차
    # 분기 배열만 등록한다. 임의의 financial/comparison 혼합은 열지 않는다.
    *(
        (
            f"financial/recent-periods-fanout-mixed-{item_count}-"
            f"{scope_qualifier_count}-{shape}-"
            f"{''.join('d' if kind == 'financial' else 'q4' for kind in kinds)}",
            _structural_signature_from_payload({
                "entities": [{"kind_hint": "company"}],
                "answer_items": [
                    {
                        "target": {"kind": "metric", "entity_positions": [0],
                                   "qualifier_count": 0},
                        "operation": "retrieve",
                        "scope": {"target_period_count": 1,
                                  "has_as_of": False,
                                  "has_document_group": False,
                                  "scope_qualifier_count":
                                      scope_qualifier_count},
                        "selection": None,
                        "output": {"shape": shape,
                                   "projection_mode": "named_fields",
                                   "field_count": 1,
                                   "presentation": "auto"},
                    },
                ] * item_count,
                "answer_groups": [], "premises": [],
                "unresolved_mentions": [], "presentation": "table",
            }),
            kinds,
            _validate_recent_periods_fanout_intent,
            _validate_recent_periods_fanout_resolution,
            _lower_recent_periods_fanout,
        )
        for item_count in (2, 3, 4, 5)
        for scope_qualifier_count in (0, 1)
        for shape in ("scalar", "timeline")
        for kinds in _recent_quarter_resolution_kind_patterns(item_count)
    ),
    # 이슈 #171 M16 — 「삼성전자의 2023년, 2024년, 2025년 연결 매출액을
    # 각각 알려줘」(연간·분기 혼합도 흔하다). 위 recent-periods-fanout과
    # 달리 이 형제들은 **이미** 서로 다른 리터럴 기간을 갖고 있다 —
    # `ExplicitPeriodsFanoutRegrounder`(agent/stage1_v1_backend_composition.py)
    # 는 아무 값도 계산하지 않고 top-level presentation만 ``"list"`` 로
    # 못박아 개념 fanout(``"auto"``)·recent-periods-fanout(``"table"``)과
    # signature를 가른다.
    #
    # resolution 검증은 recent-periods-fanout의 것과 **거의** 같지만
    # (`_validate_explicit_periods_fanout_resolution` 참고) period_type
    # (연간/분기)까지 같기를 요구하지 않는다 — 이 fanout은 형제마다 이미
    # 다른, 서로 다른 축의 기간(연간·분기 혼합)을 낼 수 있어 그 요구를
    # 걸면 정상적인 혼합 질문이 거절된다. lowering은 그대로 재사용한다 —
    # 파생 없이 사실 하나에 답 뿌리 하나라 축이 뭐든 보지 않는다.
    *(
        (
            f"financial/explicit-periods-fanout-{item_count}-"
            f"{scope_qualifier_count}-{shape}",
            _structural_signature_from_payload({
                "entities": [{"kind_hint": "company"}],
                "answer_items": [
                    {
                        "target": {"kind": "metric", "entity_positions": [0],
                                   "qualifier_count": 0},
                        "operation": "retrieve",
                        "scope": {"target_period_count": 1,
                                  "has_as_of": False,
                                  "has_document_group": False,
                                  "scope_qualifier_count":
                                      scope_qualifier_count},
                        "selection": None,
                        "output": {"shape": shape,
                                   "projection_mode": "named_fields",
                                   "field_count": 1, "presentation": "auto"},
                    },
                ] * item_count,
                "answer_groups": [], "premises": [], "unresolved_mentions": [],
                "presentation": "list",
            }),
            ("financial",) * item_count,
            _validate_explicit_periods_fanout_intent,
            _validate_explicit_periods_fanout_resolution,
            _lower_explicit_periods_fanout,
        )
        for item_count in (2, 3, 4, 5)
        for scope_qualifier_count in (0, 1)
        for shape in ("scalar", "timeline")
    ),
    (
        "financial/retrieve-then-scalar-comparison",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [
                {
                    "target": {"kind": "metric", "entity_positions": [0],
                               "qualifier_count": 0},
                    "operation": "retrieve",
                    "scope": {"target_period_count": 1, "has_as_of": False,
                              "has_document_group": False,
                              "scope_qualifier_count": 1},
                    "selection": None,
                    "output": {"shape": "scalar",
                               "projection_mode": "named_fields",
                               "field_count": 1, "presentation": "auto"},
                },
                {
                    "target": {"kind": "metric", "entity_positions": [0],
                               "qualifier_count": 0},
                    "operation": "compare",
                    "scope": {"target_period_count": 2, "has_as_of": False,
                              "has_document_group": False,
                              "scope_qualifier_count": 0},
                    "selection": None,
                    "output": {"shape": "scalar",
                               "projection_mode": "named_fields",
                               "field_count": 1, "presentation": "auto"},
                },
            ],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("financial", "financial_comparison"),
        _validate_financial_retrieve_comparison_intent,
        _validate_financial_retrieve_comparison_resolution,
        _lower_financial_retrieve_comparison,
    ),
    (
        "financial/parallel-annual-amount-rate-change",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [
                {
                    "target": {"kind": "metric", "entity_positions": [0],
                               "qualifier_count": 0},
                    "operation": "retrieve",
                    "scope": {"target_period_count": 2, "has_as_of": False,
                              "has_document_group": False,
                              "scope_qualifier_count": 1},
                    "selection": None,
                    "output": {"shape": "scalar",
                               "projection_mode": "named_fields",
                               "field_count": 1, "presentation": "auto"},
                },
                {
                    "target": {"kind": "metric", "entity_positions": [0],
                               "qualifier_count": 0},
                    "operation": "compare",
                    "scope": {"target_period_count": 2, "has_as_of": False,
                              "has_document_group": False,
                              "scope_qualifier_count": 1},
                    "selection": None,
                    "output": {"shape": "scalar",
                               "projection_mode": "named_fields",
                               "field_count": 1, "presentation": "auto"},
                },
            ],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("financial_comparison", "financial_comparison"),
        _validate_parallel_annual_change_intent,
        _validate_parallel_annual_change_resolution,
        _lower_parallel_annual_change,
    ),
    (
        "financial/comparison",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}, {"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "metric", "entity_positions": [0, 1],
                            "qualifier_count": 0},
                "operation": "compare",
                "scope": {"target_period_count": 1, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 1},
                "selection": {"mode": "maximum", "has_k": False},
                "output": {"shape": "comparison", "projection_mode": "named_fields",
                           "field_count": 2,
                           "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("financial_comparison",), _validate_financial_comparison_intent,
        _validate_financial_comparison_resolution, _lower_financial_comparison,
    ),
    *(
        (
            f"financial/cross-company-ratio-{operation}-{scope_qualifier_count}",
            _structural_signature_from_payload({
                "entities": [{"kind_hint": "company"}, {"kind_hint": "company"}],
                "answer_items": [{
                    "target": {"kind": "metric", "entity_positions": [0, 1],
                               "qualifier_count": 0},
                    "operation": operation,
                    "scope": {"target_period_count": 1, "has_as_of": False,
                              "has_document_group": False,
                              "scope_qualifier_count": scope_qualifier_count},
                    "selection": None,
                    "output": {"shape": "scalar",
                               "projection_mode": "named_fields",
                               "field_count": 1, "presentation": "auto"},
                }],
                "answer_groups": [], "premises": [], "unresolved_mentions": [],
                "presentation": "auto",
            }),
            ("financial_comparison",),
            _validate_financial_cross_company_ratio_intent,
            _validate_financial_comparison_resolution,
            _lower_financial_comparison,
        )
        # 이슈 #59 1단계 — 「삼성전자의 2025년 연결 매출액은 SK하이닉스의
        # 몇 배인가?」. HCX가 이 배수 요청을 retrieve/compare 중 어느
        # operation으로, 「연결」을 scope qualifier로 담을지 target 표면에
        # 녹일지 판마다 다르게 낼 수 있어 두 축을 모두 등록해 둔다.
        for operation in ("retrieve", "compare")
        for scope_qualifier_count in (0, 1)
    ),
    *(
        (
            f"financial/nary-ranking-{company_count}-{field_count}",
            _structural_signature_from_payload({
                "entities": [
                    {"kind_hint": "company"}
                    for _ in range(company_count)
                ],
                "answer_items": [{
                    "target": {
                        "kind": "metric",
                        "entity_positions": list(range(company_count)),
                        "qualifier_count": 0,
                    },
                    "operation": "compare",
                    "scope": {
                        "target_period_count": 1,
                        "has_as_of": False,
                        "has_document_group": False,
                        "scope_qualifier_count": 1,
                    },
                    "selection": {"mode": "maximum", "has_k": False},
                    "output": {
                        "shape": "comparison",
                        "projection_mode": "named_fields",
                        "field_count": field_count,
                        "presentation": "auto",
                    },
                }],
                "answer_groups": [], "premises": [],
                "unresolved_mentions": [], "presentation": "auto",
            }),
            ("financial_comparison",),
            _validate_financial_comparison_intent,
            _validate_financial_comparison_resolution,
            _lower_financial_comparison,
        )
        for company_count in (3, 4, 5, 6, 7, 8)
        for field_count in (1, 2)
    ),
    (
        "financial/comparison-verification",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}, {"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "metric", "entity_positions": [0, 1],
                           "qualifier_count": 0},
                "operation": "compare",
                "scope": {"target_period_count": 1, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 1},
                "selection": {"mode": "maximum", "has_k": False},
                "output": {"shape": "comparison", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("financial_comparison",), _validate_financial_comparison_intent,
        _validate_financial_comparison_resolution, _lower_financial_comparison,
    ),
    (
        "financial/single-quarter-derivation",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "metric", "entity_positions": [0],
                           "qualifier_count": 1},
                "operation": "retrieve",
                "scope": {"target_period_count": 2, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "scalar", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("financial_comparison",), _validate_financial_comparison_intent,
        _validate_financial_comparison_resolution, _lower_financial_comparison,
    ),
    (
        "financial/qualifier-single-quarter-derivation",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "metric", "entity_positions": [0],
                           "qualifier_count": 1},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "scalar", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("financial_comparison",),
        _validate_recovered_single_quarter_derivation_intent,
        _validate_financial_comparison_resolution, _lower_financial_comparison,
    ),
    (
        "financial/time-comparison-derivation",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "metric", "entity_positions": [0],
                           "qualifier_count": 0},
                "operation": "compare",
                "scope": {"target_period_count": 2, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 1},
                "selection": None,
                "output": {"shape": "comparison", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("financial_comparison",), _validate_financial_comparison_intent,
        _validate_financial_comparison_resolution, _lower_financial_comparison,
    ),
    (
        "financial/scope-comparison-derivation",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "metric", "entity_positions": [0],
                           "qualifier_count": 0},
                "operation": "compare",
                "scope": {"target_period_count": 1, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 2},
                "selection": None,
                "output": {"shape": "comparison", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("financial_comparison",), _validate_financial_comparison_intent,
        _validate_financial_comparison_resolution, _lower_financial_comparison,
    ),
    (
        "financial/year-over-year-derivation",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "metric", "entity_positions": [0],
                           "qualifier_count": 1},
                "operation": "compare",
                "scope": {"target_period_count": 1, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 1},
                "selection": None,
                "output": {"shape": "comparison", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("financial_comparison",), _validate_financial_comparison_intent,
        _validate_financial_comparison_resolution, _lower_financial_comparison,
    ),
    (
        "event/status-two-timepoints-external-event",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}, {"kind_hint": "event"}],
            "answer_items": [{
                "target": {"kind": "event", "entity_positions": [1],
                           "qualifier_count": 2},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "record", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("selected_event",), _validate_selected_event_intent,
        _validate_selected_event_resolution, _lower_selected_event,
    ),
    (
        "event/status-two-timepoints-external-company",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}, {"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "event", "entity_positions": [1],
                           "qualifier_count": 2},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "record", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("selected_event",), _validate_selected_event_intent,
        _validate_selected_event_resolution, _lower_selected_event,
    ),
    (
        "event/status-two-timepoints-canonical-issuer-authority",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "event"}],
            "answer_items": [{
                "target": {"kind": "event", "entity_positions": [0],
                           "qualifier_count": 2},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "record", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("selected_event",), _validate_selected_event_intent,
        _validate_selected_event_resolution, _lower_selected_event,
    ),
    (
        "event/status-two-timepoints-seed-date",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "event", "entity_positions": [0],
                           "qualifier_count": 1},
                "operation": "retrieve",
                "scope": {"target_period_count": 2, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "narrative", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("selected_event",), _validate_selected_event_intent,
        _validate_selected_event_resolution, _lower_selected_event,
    ),
    (
        "event/status-two-timepoints-seed-date-comparison",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "event", "entity_positions": [0],
                           "qualifier_count": 1},
                "operation": "retrieve",
                "scope": {"target_period_count": 2, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "comparison", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("selected_event",), _validate_selected_event_intent,
        _validate_selected_event_resolution, _lower_selected_event,
    ),
    (
        "event/selected-single-field",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "event", "entity_positions": [0],
                            "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "scalar", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("selected_event",), _validate_selected_event_intent,
        _validate_selected_event_resolution, _lower_selected_event,
    ),
    (
        "event/selected-single-field-record",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "event", "entity_positions": [0],
                            "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": {"mode": "latest", "has_k": False},
                "output": {"shape": "record", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("selected_event",), _validate_selected_event_intent,
        _validate_selected_event_resolution, _lower_selected_event,
    ),
    (
        "event/selected-bounded-period-two-field-record",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "event", "entity_positions": [0],
                           "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 1, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "record",
                           "projection_mode": "named_fields",
                           "field_count": 2, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("selected_event",), _validate_selected_event_intent,
        _validate_selected_event_resolution, _lower_selected_event,
    ),
    (
        "event/selected-single-field-missing-issuer-context",
        _structural_signature_from_payload({
            "entities": [],
            "answer_items": [{
                "target": {"kind": "event", "entity_positions": [],
                           "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "scalar",
                           "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("selected_event",), _validate_selected_event_intent,
        _validate_selected_event_resolution, _lower_selected_event,
    ),
    (
        "event/investment-judgment-exact-content",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "event", "entity_positions": [0],
                           "qualifier_count": 1},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "record",
                           "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "prose"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "prose",
        }),
        ("selected_event",), _validate_selected_event_intent,
        _validate_selected_event_resolution, _lower_selected_event,
    ),
    (
        "event/status-one-timepoint-seeded-document",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}, {"kind_hint": "event"}],
            "answer_items": [{
                "target": {"kind": "document", "entity_positions": [1], "qualifier_count": 1},
                "operation": "retrieve",
                "scope": {"target_period_count": 1, "has_as_of": True,
                          "has_document_group": False, "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "scalar", "projection_mode": "named_fields", "field_count": 1, "presentation": "auto"},
            }], "answer_groups": [], "premises": [], "unresolved_mentions": [], "presentation": "auto",
        }),
        ("selected_event",), _validate_selected_event_intent,
        _validate_selected_event_resolution, _lower_selected_event,
    ),
    (
        "event/timeline-partial-missing-root",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "document", "entity_positions": [0], "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False, "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "narrative", "projection_mode": "named_fields", "field_count": 2, "presentation": "auto"},
            }], "answer_groups": [], "premises": [], "unresolved_mentions": [], "presentation": "auto",
        }),
        ("selected_event",), _validate_selected_event_intent,
        _validate_selected_event_resolution, _lower_selected_event,
    ),
    (
        "document/version-history-premise",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}], "answer_items": [{
                "target": {"kind": "document", "entity_positions": [0], "qualifier_count": 0},
                "operation": "retrieve", "scope": {"target_period_count": 0, "has_as_of": False, "has_document_group": False, "scope_qualifier_count": 0},
                "selection": None, "output": {"shape": "scalar", "projection_mode": "named_fields", "field_count": 1, "presentation": "auto"},
            }], "answer_groups": [], "premises": [{"kind": "existence", "item_positions": [0]}], "unresolved_mentions": [], "presentation": "auto",
        }),
        ("document_version_history",), _validate_document_version_history_intent,
        _validate_document_version_history_resolution, _lower_document_version_history,
    ),
    (
        "holding/exact-reporter-filing",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "document", "entity_positions": [0],
                           "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": True,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "scalar",
                           "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("holding_disclosure",),
        _validate_holding_disclosure_intent,
        _validate_holding_disclosure_resolution,
        _lower_holding_disclosure,
    ),
    (
        "document/whole-target-collection",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "document", "entity_positions": [0],
                            "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "record_list",
                           "projection_mode": "whole_target",
                           "field_count": 0, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("document_collection",),
        _validate_document_collection_intent,
        _validate_document_collection_resolution,
        _lower_document_collection,
    ),
    (
        "topic/whole-target-collection",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "topic", "entity_positions": [0],
                           "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "record", "projection_mode": "whole_target",
                           "field_count": 0, "presentation": "auto"},
            }], "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("document_collection",),
        _validate_document_collection_intent,
        _validate_document_collection_resolution,
        _lower_document_collection,
    ),
    (
        "topic/whole-target-narrative",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "topic", "entity_positions": [0],
                           "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "narrative",
                           "projection_mode": "whole_target",
                           "field_count": 0, "presentation": "auto"},
            }], "answer_groups": [], "premises": [],
            "unresolved_mentions": [], "presentation": "auto",
        }),
        ("document_collection",),
        _validate_document_collection_intent,
        _validate_document_collection_resolution,
        _lower_document_collection,
    ),
    (
        "topic/recent-investment-narrative",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "topic", "entity_positions": [0],
                           "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "narrative", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }], "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("document_collection",),
        _validate_document_collection_intent,
        _validate_document_collection_resolution,
        _lower_document_collection,
    ),
    (
        "narrative/periodic-document",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "topic", "entity_positions": [0],
                            "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": True,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "record_list", "projection_mode": "named_fields",
                           "field_count": 4,
                           "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("periodic_document_narrative",),
        _validate_periodic_document_narrative_intent,
        _validate_periodic_document_narrative_resolution,
        _lower_periodic_document_narrative,
    ),
    (
        "narrative/matrix",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "document", "entity_positions": [0],
                           "qualifier_count": 0},
                "operation": "compare",
                "scope": {"target_period_count": 2, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "comparison", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("narrative_matrix",),
        _validate_narrative_matrix_intent,
        _validate_narrative_matrix_resolution,
        _lower_narrative_matrix,
    ),
    (
        "narrative/periodic-comparison",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "topic", "entity_positions": [0],
                            "qualifier_count": 0},
                "operation": "compare",
                "scope": {"target_period_count": 2, "has_as_of": False,
                          "has_document_group": True,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "narrative", "projection_mode": "named_fields",
                           "field_count": 4,
                           "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("periodic_narrative_comparison",),
        _validate_periodic_narrative_comparison_intent,
        _validate_periodic_narrative_comparison_resolution,
        _lower_periodic_narrative_comparison,
    ),
    (
        "document-fact/comparison-reason",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "event"}],
            "answer_items": [
                {
                    "target": {"kind": "event", "entity_positions": [0],
                                "qualifier_count": 0},
                    "operation": "retrieve",
                    "scope": {"target_period_count": 0, "has_as_of": False,
                              "has_document_group": False,
                              "scope_qualifier_count": 0},
                    "selection": None,
                    "output": {"shape": "verdict", "projection_mode": "named_fields",
                               "field_count": 1,
                               "presentation": "auto"},
                },
                {
                    "target": {"kind": "event", "entity_positions": [0],
                                "qualifier_count": 0},
                    "operation": "retrieve",
                    "scope": {"target_period_count": 0, "has_as_of": False,
                              "has_document_group": False,
                              "scope_qualifier_count": 0},
                    "selection": None,
                    "output": {"shape": "narrative", "projection_mode": "named_fields",
                               "field_count": 1,
                               "presentation": "auto"},
                },
            ],
            "answer_groups": [[0, 1]],
            "premises": [{"kind": "comparison", "item_positions": [0, 1]}],
            "unresolved_mentions": [], "presentation": "auto",
        }),
        ("document_fact_comparison", "document_reason_evidence"),
        _validate_document_fact_comparison_intent,
        _validate_document_fact_resolution, _lower_document_fact_comparison,
    ),
    (
        "disclosure/cross-document-attributes",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}],
            "answer_items": [
                {
                    "target": {"kind": "event", "entity_positions": [0],
                               "qualifier_count": 0},
                    "operation": "retrieve",
                    "scope": {"target_period_count": 0, "has_as_of": False,
                              "has_document_group": False,
                              "scope_qualifier_count": 0},
                    "selection": None,
                    "output": {"shape": "scalar", "projection_mode": "named_fields",
                               "field_count": 1,
                               "presentation": "auto"},
                },
                {
                    "target": {"kind": "event", "entity_positions": [0],
                               "qualifier_count": 0},
                    "operation": "retrieve",
                    "scope": {"target_period_count": 0, "has_as_of": False,
                              "has_document_group": False,
                              "scope_qualifier_count": 0},
                    "selection": None,
                    "output": {"shape": "scalar", "projection_mode": "named_fields",
                               "field_count": 1,
                               "presentation": "auto"},
                },
            ],
            "answer_groups": [[0, 1]], "premises": [],
            "unresolved_mentions": [], "presentation": "auto",
        }),
        ("document_attribute_evidence", "document_attribute_evidence"),
        _validate_document_attribute_intent,
        _validate_document_attribute_resolution, _lower_document_attributes,
    ),
    (
        "event/reported-termination-status",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}, {"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "event", "entity_positions": [0, 1],
                           "qualifier_count": 1},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": True,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "record", "projection_mode": "named_fields",
                           "field_count": 2, "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }),
        ("termination_reported_status",),
        _validate_termination_reported_status_intent,
        _validate_termination_reported_status_resolution,
        _lower_termination_reported_status,
    ),
    (
        "event/reported-termination-status-premise",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}, {"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "event", "entity_positions": [0, 1],
                           "qualifier_count": 1},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": True,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "record", "projection_mode": "named_fields",
                           "field_count": 2, "presentation": "auto"},
            }],
            "answer_groups": [],
            "premises": [{"kind": "state", "item_positions": [0]}],
            "unresolved_mentions": [], "presentation": "auto",
        }),
        ("termination_reported_status",),
        _validate_termination_reported_status_intent,
        _validate_termination_reported_status_resolution,
        _lower_termination_reported_status,
    ),
    (
        "event/reported-termination-cutoff-status-premise",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "company"}, {"kind_hint": "company"}],
            "answer_items": [{
                "target": {"kind": "event", "entity_positions": [0, 1],
                           "qualifier_count": 0},
                "operation": "retrieve",
                "scope": {"target_period_count": 0, "has_as_of": False,
                          "has_document_group": False,
                          "scope_qualifier_count": 0},
                "selection": None,
                "output": {"shape": "scalar", "projection_mode": "named_fields",
                           "field_count": 1, "presentation": "auto"},
            }],
            "answer_groups": [],
            "premises": [{"kind": "state", "item_positions": [0]}],
            "unresolved_mentions": [], "presentation": "auto",
        }),
        ("termination_reported_status",),
        _validate_termination_reported_status_intent,
        _validate_termination_reported_status_resolution,
        _lower_termination_reported_status,
    ),
    (
        "disclosure/same-day-status",
        _structural_signature_from_payload({
            "entities": [{"kind_hint": "counterparty"}],
            "answer_items": [
                {
                    "target": {"kind": "document", "entity_positions": [0],
                                "qualifier_count": 0},
                    "operation": "retrieve",
                    "scope": {"target_period_count": 0, "has_as_of": True,
                              "has_document_group": False,
                              "scope_qualifier_count": 0},
                    "selection": {"mode": "latest", "has_k": False},
                    "output": {"shape": "narrative", "projection_mode": "named_fields",
                               "field_count": 1,
                               "presentation": "auto"},
                },
                {
                    "target": {"kind": "event", "entity_positions": [0],
                                "qualifier_count": 0},
                    "operation": "retrieve",
                    "scope": {"target_period_count": 0, "has_as_of": False,
                              "has_document_group": False,
                              "scope_qualifier_count": 1},
                    "selection": None,
                    "output": {"shape": "scalar", "projection_mode": "named_fields",
                               "field_count": 1,
                               "presentation": "auto"},
                },
            ],
            "answer_groups": [[0, 1]], "premises": [],
            "unresolved_mentions": [], "presentation": "auto",
        }),
        ("same_day_document_candidates", "termination_reported_status"),
        _validate_same_day_status_intent,
        _validate_same_day_status_resolution, _lower_same_day_status,
    ),
    (
        "event/lifecycle-composite-single",
        canonical_json({"lifecycle": "single"}),
        ("event_lifecycle_composite",),
        _validate_lifecycle_composite_intent,
        _validate_lifecycle_composite_resolution,
        _lower_lifecycle_composite,
    ),
    (
        "event/lifecycle-composite-pair",
        canonical_json({"lifecycle": "pair"}),
        ("event_lifecycle_composite", "event_lifecycle_composite"),
        _validate_lifecycle_composite_intent,
        _validate_lifecycle_composite_resolution,
        _lower_lifecycle_composite,
    ),
)


def _build_stage1_v1_handler_registry() -> tuple[Stage1V1Handler, ...]:
    """Build the closed registry from static qid-free structural specs."""
    return tuple(
        Stage1V1Handler(
            name=name,
            intent_signature=signature,
            resolution_kind_inventory=kind_inventory,
            validate_intent=intent_validator,
            validate_resolution=resolution_validator,
            lower=lower,
        )
        for (name, signature, kind_inventory, intent_validator,
             resolution_validator, lower) in _STAGE1_V1_STRUCTURAL_SPECS
    )


# Public for deterministic audit/tests. Replacing this tuple in a controlled
# integration is the supported way to exercise no-match/multiple-match
# fail-closed behavior.
STAGE1_V1_HANDLER_REGISTRY = _build_stage1_v1_handler_registry()


def _demo_slice(question_id: str) -> DeterministicCompiledSlice:
    intent = _demo_intent(question_id)
    resolution = _demo_resolution(question_id, intent)
    return compile_stage1_v1(
        question_id, _expected_question(question_id), intent, resolution)


def vertical_slice_inputs(
        question_id: str,
        ) -> tuple[str, SemanticIntent, AuthoritativeResolution]:
    """Return deterministic typed input records for checked-in slices."""
    question = _expected_question(question_id)
    intent = _demo_intent(question_id)
    return question, intent, _demo_resolution(question_id, intent)


def write_vertical_slice_artifacts() -> tuple[str, ...]:
    """Write canonical Stage1 v1 end-to-end proof artifacts."""
    outputs: list[str] = []
    for question_id, path in (
            (G_A_001, G_A_001_ARTIFACT),
            (G_A_004, G_A_004_ARTIFACT),
            (G_A_010, G_A_010_ARTIFACT),
            (G_I_004, G_I_004_ARTIFACT),
            (G_I_006, G_I_006_ARTIFACT),
            (G_I_009, G_I_009_ARTIFACT),
            (G_O_001, G_O_001_ARTIFACT),
            (R_A_002, R_A_002_ARTIFACT)):
        result = _demo_slice(question_id)
        payload = canonical_json(result.model_dump(
            mode="json", warnings=False)).encode("utf-8") + b"\n"
        _atomic_write(path, payload)
        outputs.append(result.bundle_digest)
    return tuple(outputs)  # type: ignore[return-value]


def verify_vertical_slice_artifacts() -> tuple[str, ...]:
    expected: list[str] = []
    for question_id, path in (
            (G_A_001, G_A_001_ARTIFACT),
            (G_A_004, G_A_004_ARTIFACT),
            (G_A_010, G_A_010_ARTIFACT),
            (G_I_004, G_I_004_ARTIFACT),
            (G_I_006, G_I_006_ARTIFACT),
            (G_I_009, G_I_009_ARTIFACT),
            (G_O_001, G_O_001_ARTIFACT),
            (R_A_002, R_A_002_ARTIFACT)):
        result = _demo_slice(question_id)
        expected_bytes = canonical_json(result.model_dump(
            mode="json", warnings=False)).encode("utf-8") + b"\n"
        try:
            actual = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"vertical slice artifact가 없습니다: {path}") from exc
        if actual != expected_bytes:
            raise RuntimeError(f"vertical slice artifact drift: {path}")
        loaded = load_compiled_slice_json(actual)
        if loaded.bundle_digest != result.bundle_digest:
            raise RuntimeError(f"vertical slice digest drift: {path}")
        expected.append(result.bundle_digest)
    return tuple(expected)  # type: ignore[return-value]


def load_authoritative_resolution_json(
        payload: str | bytes | bytearray,
        ) -> AuthoritativeResolution:
    return AuthoritativeResolution.model_validate_json(payload, strict=True)


def load_execution_plan_json(
        payload: str | bytes | bytearray,
        ) -> ExecutionPlan:
    return ExecutionPlan.model_validate_json(payload, strict=True)


def _artifact_bytes(model: type[BaseModel], path: Path) -> tuple[bytes, bytes]:
    text = canonical_json(model.model_json_schema(mode="validation"))
    digest = sha256(text.encode("utf-8")).hexdigest()
    return text.encode("utf-8"), f"{digest}  {path.name}\n".encode("ascii")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_schema_artifacts() -> tuple[str, str, str]:
    resolution, resolution_digest = _artifact_bytes(
        AuthoritativeResolution, RESOLUTION_SCHEMA_ARTIFACT)
    plan, plan_digest = _artifact_bytes(ExecutionPlan, PLAN_SCHEMA_ARTIFACT)
    compiled_slice, compiled_slice_digest = _artifact_bytes(
        DeterministicCompiledSlice, COMPILED_SLICE_SCHEMA_ARTIFACT)
    _atomic_write(RESOLUTION_SCHEMA_ARTIFACT, resolution)
    _atomic_write(RESOLUTION_SCHEMA_DIGEST_ARTIFACT, resolution_digest)
    _atomic_write(PLAN_SCHEMA_ARTIFACT, plan)
    _atomic_write(PLAN_SCHEMA_DIGEST_ARTIFACT, plan_digest)
    _atomic_write(COMPILED_SLICE_SCHEMA_ARTIFACT, compiled_slice)
    _atomic_write(COMPILED_SLICE_SCHEMA_DIGEST_ARTIFACT, compiled_slice_digest)
    return (resolution_digest.decode().split()[0], plan_digest.decode().split()[0],
            compiled_slice_digest.decode().split()[0])


def verify_schema_artifacts() -> tuple[str, str, str]:
    expected = (
        (RESOLUTION_SCHEMA_ARTIFACT, *_artifact_bytes(
            AuthoritativeResolution, RESOLUTION_SCHEMA_ARTIFACT)),
        (RESOLUTION_SCHEMA_DIGEST_ARTIFACT, None,
         _artifact_bytes(AuthoritativeResolution, RESOLUTION_SCHEMA_ARTIFACT)[1]),
        (PLAN_SCHEMA_ARTIFACT, *_artifact_bytes(ExecutionPlan, PLAN_SCHEMA_ARTIFACT)),
        (PLAN_SCHEMA_DIGEST_ARTIFACT, None,
         _artifact_bytes(ExecutionPlan, PLAN_SCHEMA_ARTIFACT)[1]),
        (COMPILED_SLICE_SCHEMA_ARTIFACT, *_artifact_bytes(
            DeterministicCompiledSlice, COMPILED_SLICE_SCHEMA_ARTIFACT)),
        (COMPILED_SLICE_SCHEMA_DIGEST_ARTIFACT, None,
         _artifact_bytes(DeterministicCompiledSlice,
                         COMPILED_SLICE_SCHEMA_ARTIFACT)[1]),
    )
    for path, schema, digest in expected:
        actual = path.read_bytes()
        expected_bytes = schema if schema is not None else digest
        # A text editor/apply-patch round-trip may retain one terminal newline
        # on the JSON schema.  It is not part of the canonical schema text;
        # digest sidecars continue to hash the exact canonical bytes.
        if actual != expected_bytes and not (
                schema is not None
                and actual.rstrip(b"\r\n") == expected_bytes):
            raise RuntimeError(f"schema artifact drift: {path}")
    return (
        _artifact_bytes(AuthoritativeResolution, RESOLUTION_SCHEMA_ARTIFACT)[1]
        .decode().split()[0],
        _artifact_bytes(ExecutionPlan, PLAN_SCHEMA_ARTIFACT)[1]
        .decode().split()[0],
        _artifact_bytes(DeterministicCompiledSlice,
                        COMPILED_SLICE_SCHEMA_ARTIFACT)[1]
        .decode().split()[0],
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    schema_digests = write_schema_artifacts() if args.write else verify_schema_artifacts()
    slice_digests = (
        write_vertical_slice_artifacts()
        if args.write else verify_vertical_slice_artifacts())
    print(
        f"PASS: {RESOLUTION_VERSION}={schema_digests[0]} "
        f"{EXECUTION_PLAN_VERSION}={schema_digests[1]} "
        f"{COMPILED_SLICE_VERSION}={schema_digests[2]} "
        f"g_a_001={slice_digests[0]} "
        f"g_a_004={slice_digests[1]} g_a_010={slice_digests[2]} "
        f"g_i_004={slice_digests[3]} g_i_006={slice_digests[4]} "
        f"g_i_009={slice_digests[5]} g_o_001={slice_digests[6]} "
        f"r_a_002={slice_digests[7]}"
    )
    return 0


__all__ = [
    "AppliedDefault", "AuthoritativeResolution", "DeterministicCompiledSlice",
    "DeterministicPlanCompilerError", "ExecutionAnswerRoot",
    "ExecutionPlan", "ExecutionPremiseRoot", "ExecutionSupportRoot",
    "ResolvedItem", "FinancialResolution", "FinancialComparisonOperand",
    "FinancialComparisonResolution", "DocumentFactOperand",
    "DocumentReasonEvidence", "DocumentFactComparisonResolution",
    "DocumentReasonEvidenceResolution", "DocumentAttributeEvidence",
    "DocumentAttributeEvidenceResolution", "PeriodicNarrativeEvidence",
    "PeriodicNarrativeDocument", "PeriodicNarrativeComparisonResolution",
    "ResolutionPayload",
    "LimitationProvenance", "SameDayDocumentCandidate",
    "SameDayDocumentCandidatesResolution", "ResolutionSourceProof",
    "PeriodicDocumentNarrativeResolution",
    "PublicEventSelectorFacets", "TerminationReportedStatusResolution",
    "DocumentCollectionResolution",
    "HoldingSlotBinding", "HoldingDisclosureResolution",
    "DocumentVersionHistoryResolution",
    "SelectedEventResolution", "EventCollectionMember",
    "EventCollectionResolution", "EventAmountChangeResolution",
    "LifecycleCompositeAttribute", "LifecycleCompositeResolution",
    "CorrectionLineageChange", "CorrectionLineageResolution",
    "ResolutionFieldProof",
    "ResolutionPremiseProof",
    "RESOLUTION_VERSION",
    "EXECUTION_PLAN_VERSION", "RESOLUTION_SCHEMA_ARTIFACT",
    "RESOLUTION_SCHEMA_DIGEST_ARTIFACT", "PLAN_SCHEMA_ARTIFACT",
    "PLAN_SCHEMA_DIGEST_ARTIFACT", "COMPILED_SLICE_SCHEMA_ARTIFACT",
    "COMPILED_SLICE_SCHEMA_DIGEST_ARTIFACT", "canonical_json", "canonical_sha256",
    "write_schema_artifacts", "verify_schema_artifacts",
    "verify_authoritative_resolution_digest", "verify_execution_plan_digest",
    "load_authoritative_resolution_json", "load_execution_plan_json",
    "load_compiled_slice_json", "verify_compiled_slice_digest",
    "Stage1V1Handler", "STAGE1_V1_HANDLER_REGISTRY",
    "semantic_intent_structural_signature",
    "compile_stage1_v1_generic", "compile_stage1_v1_typed",
    "compile_stage1_v1", "compile_deterministic_plan",
    "compile_g_a_001", "compile_g_a_010", "compile_r_a_002",
    "vertical_slice_inputs",
    "write_vertical_slice_artifacts", "verify_vertical_slice_artifacts",
    "G_A_001", "G_A_004", "G_A_010", "G_I_004", "G_I_006", "G_I_009",
    "G_O_001",
    "R_A_002",
    "G_A_001_QUESTION", "G_A_004_QUESTION", "G_A_010_QUESTION",
    "G_I_004_QUESTION", "G_I_006_QUESTION", "G_I_009_QUESTION",
    "G_O_001_QUESTION",
    "R_A_002_QUESTION",
    "G_A_001_ARTIFACT",
    "G_A_004_ARTIFACT", "G_A_010_ARTIFACT", "G_I_004_ARTIFACT",
    "G_I_006_ARTIFACT", "G_I_009_ARTIFACT", "G_O_001_ARTIFACT",
    "R_A_002_ARTIFACT", "COMPILED_SLICE_VERSION",
    "G_A_010_CORP_CODE", "G_A_010_CORP_NAME", "G_A_010_DOCUMENT_ID",
    "G_A_010_RECEIPT_NO", "G_A_010_DOCUMENT_PROOF", "G_A_010_NARRATIVE_PROOF",
    "G_A_010_SOURCE_CROSS_CHECK_DETAIL",
    "G_A_010_SOURCE_CROSS_CHECK_EVIDENCE_REFS",
]


if __name__ == "__main__":
    raise SystemExit(_main())
