"""SemanticIntent persistence support for final70 and historical diagnostics.

The official path is ``all70``: it reads the pinned final release questions,
persists every candidate row, and opens no expected label.  The older
``gold10``/``remaining60`` phases remain only to read historical diagnostic
artifacts; they are not acceptance prerequisites for final70.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
import re
from pathlib import Path
import tempfile
from typing import Annotated, Any, Literal, Mapping, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from agent.hcx_semantic_intent_v1 import (
    HCX_SEMANTIC_INTENT_PROMPT_VERSION,
    HcxSemanticIntentInvocationError,
    HcxSemanticIntentNormalizationInvocationError,
    HcxSemanticIntentRunner,
    SemanticIntentInvocation,
)
from agent.hcx_schema import (
    HcxSafePayloadShape,
    safe_validation_issue_codes,
    safe_validation_issue_paths,
)
from agent.providers.hcx007 import HcxError
from agent.query_plan import QuestionFixtureV04
from agent.semantic_intent_v1 import (
    HcxSemanticIntentWire,
    SemanticIntent,
    SemanticIntentNormalizationError,
    canonical_sha256,
    semantic_intent_digest,
)
from agent.semantic_intent_v1_boundary import (
    SEMANTIC_INTENT_BOUNDARY_V1,
    SEMANTIC_INTENT_NORMALIZATION_CODES,
    SEMANTIC_INTENT_PRE_SCHEMA_REPAIR_CODES,
    SemanticIntentBoundaryEvidence,
    normalize_semantic_intent_bounded,
)
from agent.stage1_v1_gold_expectations import (
    EXPECTED_SOURCE_FIXTURE_DIGESTS,
    MANDATORY_GOLD_QUESTION_IDS,
    LoadedGoldExpectations,
    load_gold_expectations,
)


HCX_SEMANTIC_INTENT_EVAL_VERSION = "hcx-semantic-intent-eval/1.3"
HCX_SEMANTIC_INTENT_EVAL_SUMMARY_VERSION = (
    "hcx-semantic-intent-eval-summary/1.4"
)
HCX_SEMANTIC_INTENT_EVAL_GOLD10_ROW_VERSION = (
    "hcx-semantic-intent-eval-gold10-row/1.3"
)
HCX_SEMANTIC_INTENT_EVAL_TECHNICAL_FAILURE_ROW_VERSION = (
    "hcx-semantic-intent-eval-technical-failure-row/1.1"
)
EXPECTED_QUESTION_COUNT = 70
EvalPhase = Literal["gold10", "remaining60", "all70"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 정본은 final release 다 (Stage1_v0.4_최종인계_결정추가서_20260821).
# base archive 는 비교·복구용 snapshot 이므로 기본 평가 입력이 아니다.
DEFAULT_QUESTIONS_PATH = (
    PROJECT_ROOT / "fixtures/query_plan_v04_final/questions_v0.4.jsonl"
)
RELEASE_MANIFEST_FILENAME = "release_manifest.json"
MAX_QUESTIONS_BYTES = 4 * 1024 * 1024
MAX_SUMMARY_BYTES = 256 * 1024
MAX_RESULT_ROW_BYTES = 4 * 1024 * 1024

SUMMARY_SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "hcx_semantic_intent_eval_summary_v14.schema.json")
SUMMARY_SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "hcx_semantic_intent_eval_summary_v14.schema.sha256")
GOLD10_ROW_SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "hcx_semantic_intent_eval_gold10_row_v13.schema.json")
GOLD10_ROW_SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "hcx_semantic_intent_eval_gold10_row_v13.schema.sha256")
TECHNICAL_FAILURE_ROW_SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "hcx_semantic_intent_eval_technical_failure_row_v11.schema.json")
TECHNICAL_FAILURE_ROW_SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name(
    "schemas") / (
    "hcx_semantic_intent_eval_technical_failure_row_v11.schema.sha256")

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
QuestionId = Annotated[str, StringConstraints(
    min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")]
NonEmpty = Annotated[str, StringConstraints(min_length=1)]
DiagnosticCode = Annotated[str, StringConstraints(
    pattern=r"^[a-z][a-z0-9_]{0,63}$")]
DiagnosticPath = Annotated[str, StringConstraints(
    pattern=(
        r"^[a-z][a-z0-9_]*(?:\[[0-9]+\])?"
        r"(?:\.[a-z][a-z0-9_]*(?:\[[0-9]+\])?)*$"
    ),
)]
SafeRelativePath = Annotated[str, StringConstraints(
    min_length=1, max_length=256,
    pattern=r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$",
)]


class SemanticIntentEvalError(ValueError):
    """The frozen evaluation or its phased gate is invalid."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SemanticIntentEvalError(f"JSON key가 중복됩니다: {key}")
        result[key] = value
    return result


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )


def _strict_nested(model: type[BaseModel], value: Any) -> BaseModel:
    if isinstance(value, model):
        value = value.model_dump(mode="json", warnings=False)
    if isinstance(value, Mapping):
        return model.model_validate_json(_canonical_json(dict(value)), strict=True)
    raise TypeError(f"{model.__name__} instance 또는 mapping이 필요합니다")


class Gold10ScoreEvidence(_StrictFrozenModel):
    exact: StrictBool
    issue_codes: list[NonEmpty]
    expected_intent_digest: Digest
    actual_intent_digest: Digest

    @model_validator(mode="after")
    def validate_score(self) -> "Gold10ScoreEvidence":
        if len(self.issue_codes) != len(set(self.issue_codes)):
            raise ValueError("Gold score issue_codes가 중복되었습니다")
        return self


class EvalRequestEvidence(_StrictFrozenModel):
    request_id: NonEmpty
    attempts: Literal[1]
    provider_schema_sha256: Digest
    system_prompt_sha256: Digest
    request_prompt_sha256: Digest
    generation_config_sha256: Digest


class EvalUsageEvidence(_StrictFrozenModel):
    total_latency_ms: StrictFloat = Field(ge=0)
    prompt_tokens: StrictInt = Field(ge=0)
    completion_tokens: StrictInt = Field(ge=0)
    total_tokens: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_tokens(self) -> "EvalUsageEvidence":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens가 prompt+completion과 다릅니다")
        return self


class EvalSemanticBoundaryEvidence(_StrictFrozenModel):
    """Persisted proof of the exact deterministic semantic boundary."""

    boundary_version: Literal[SEMANTIC_INTENT_BOUNDARY_V1]
    question_sha256: Digest
    source_wire_digest: Digest
    repaired_wire_digest: Digest
    semantic_intent_digest: Digest
    schema_repair_codes: list[DiagnosticCode]
    normalization_codes: list[DiagnosticCode]
    evidence_digest: Digest

    @model_validator(mode="after")
    def validate_evidence(self) -> "EvalSemanticBoundaryEvidence":
        if self.schema_repair_codes != sorted(set(self.schema_repair_codes)) \
                or any(code not in SEMANTIC_INTENT_PRE_SCHEMA_REPAIR_CODES
                       for code in self.schema_repair_codes):
            raise ValueError("eval schema repair code가 유효하지 않습니다")
        if self.normalization_codes != sorted(set(self.normalization_codes)) \
                or any(code not in SEMANTIC_INTENT_NORMALIZATION_CODES
                       for code in self.normalization_codes):
            raise ValueError("eval normalization code가 유효하지 않습니다")
        body = self.model_dump(mode="json", warnings=False)
        body.pop("evidence_digest")
        if self.evidence_digest != _canonical_sha256(body):
            raise ValueError("eval semantic boundary evidence digest가 다릅니다")
        return self


class EvalSafePayloadShapeEvidence(_StrictFrozenModel):
    """Content-free provider structure; never stores user or model surfaces."""

    root_fields: list[DiagnosticCode]
    proposed_disposition: NonEmpty
    speech_act: NonEmpty
    task_count: StrictInt = Field(ge=-1)
    task_modes: list[NonEmpty]
    task_fields: list[list[DiagnosticCode]]
    task_fact_counts: list[StrictInt]
    task_source_slot_counts: list[StrictInt]
    task_target_period_counts: list[StrictInt]
    analysis_count: StrictInt = Field(ge=-1)
    analysis_operators: list[NonEmpty]
    claim_count: StrictInt = Field(ge=-1)
    claim_kinds: list[NonEmpty]
    requested_output_count: StrictInt = Field(ge=-1)
    requested_output_refs: list[NonEmpty]
    reason_count: StrictInt = Field(ge=-1)
    blank_field_paths: list[DiagnosticPath]
    unspecified_field_paths: list[DiagnosticPath]
    duplicate_array_paths: list[DiagnosticPath]
    semantic_entity_count: StrictInt = Field(ge=-1)
    semantic_answer_item_count: StrictInt = Field(ge=-1)
    semantic_target_kinds: list[NonEmpty]
    semantic_operations: list[NonEmpty]
    semantic_projection_modes: list[NonEmpty]
    semantic_output_shapes: list[NonEmpty]
    semantic_field_counts: list[StrictInt]
    semantic_answer_group_count: StrictInt = Field(ge=-1)
    semantic_premise_count: StrictInt = Field(ge=-1)
    semantic_unresolved_count: StrictInt = Field(ge=-1)
    semantic_presentation: NonEmpty

    @classmethod
    def from_shape(
            cls, value: HcxSafePayloadShape,
            ) -> "EvalSafePayloadShapeEvidence":
        if not isinstance(value, HcxSafePayloadShape):
            raise TypeError("safe payload shape authority가 잘못되었습니다")
        return cls.model_validate_json(
            _canonical_json(value.as_dict()), strict=True)

    @model_validator(mode="after")
    def validate_shape(self) -> "EvalSafePayloadShapeEvidence":
        ordered_unique = (
            self.root_fields,
            self.blank_field_paths,
            self.unspecified_field_paths,
            self.duplicate_array_paths,
        )
        if any(rows != sorted(set(rows)) for rows in ordered_unique):
            raise ValueError("safe diagnostic path/key는 정렬·중복제거되어야 합니다")
        legacy_task_rows = 0 if self.task_count < 0 else min(self.task_count, 12)
        if any(len(rows) != legacy_task_rows for rows in (
                self.task_modes,
                self.task_fields,
                self.task_fact_counts,
                self.task_source_slot_counts,
                self.task_target_period_counts,
                )):
            raise ValueError("safe legacy task shape 길이가 count와 다릅니다")
        semantic_rows = (
            0 if self.semantic_answer_item_count < 0
            else min(self.semantic_answer_item_count, 50)
        )
        if any(len(rows) != semantic_rows for rows in (
                self.semantic_target_kinds,
                self.semantic_operations,
                self.semantic_projection_modes,
                self.semantic_output_shapes,
                self.semantic_field_counts,
                )):
            raise ValueError("safe SemanticIntent item shape 길이가 count와 다릅니다")
        return self


class EvalTechnicalFailureEvidence(_StrictFrozenModel):
    layer: Literal[
        "transport", "schema", "normalization", "binding", "local_contract"
    ]
    code: DiagnosticCode
    issue_codes: list[DiagnosticCode]
    issue_paths: list[DiagnosticPath]
    boundary_version: Literal[SEMANTIC_INTENT_BOUNDARY_V1]
    schema_repair_codes: list[DiagnosticCode]
    normalization_codes: list[DiagnosticCode]
    payload_shape: EvalSafePayloadShapeEvidence | None = None

    @field_validator("payload_shape", mode="before")
    @classmethod
    def strict_shape(cls, value: Any) -> EvalSafePayloadShapeEvidence | None:
        if value is None:
            return None
        return _strict_nested(  # type: ignore[return-value]
            EvalSafePayloadShapeEvidence, value)

    @model_validator(mode="after")
    def validate_diagnostics(self) -> "EvalTechnicalFailureEvidence":
        if self.issue_codes != sorted(set(self.issue_codes)):
            raise ValueError("technical failure issue_codes가 정렬·중복제거되지 않았습니다")
        if self.issue_paths != sorted(set(self.issue_paths)):
            raise ValueError("technical failure issue_paths가 정렬·중복제거되지 않았습니다")
        if self.schema_repair_codes != sorted(set(self.schema_repair_codes)) \
                or any(code not in SEMANTIC_INTENT_PRE_SCHEMA_REPAIR_CODES
                       for code in self.schema_repair_codes):
            raise ValueError("technical failure schema repair code가 유효하지 않습니다")
        if self.normalization_codes != sorted(set(self.normalization_codes)) \
                or any(code not in SEMANTIC_INTENT_NORMALIZATION_CODES
                       for code in self.normalization_codes):
            raise ValueError("technical failure normalization code가 유효하지 않습니다")
        return self


class EvalFailureRequestEvidence(_StrictFrozenModel):
    request_id: NonEmpty | None
    attempts: StrictInt = Field(ge=0, le=1)
    provider_schema_sha256: Digest
    system_prompt_sha256: Digest
    request_prompt_sha256: Digest
    generation_config_sha256: Digest


class EvalFailureUsageEvidence(_StrictFrozenModel):
    known: StrictBool
    total_latency_ms: StrictFloat = Field(ge=0)
    prompt_tokens: StrictInt = Field(ge=0)
    completion_tokens: StrictInt = Field(ge=0)
    total_tokens: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_usage(self) -> "EvalFailureUsageEvidence":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("failure total_tokens가 prompt+completion과 다릅니다")
        if not self.known and any((
                self.prompt_tokens, self.completion_tokens, self.total_tokens)):
            raise ValueError("unknown failure usage는 token count를 가질 수 없습니다")
        return self


class TechnicalFailureEvalRow(_StrictFrozenModel):
    """Persisted technical failure with no provider-authored free text."""

    schema_version: Literal[HCX_SEMANTIC_INTENT_EVAL_VERSION]
    order: StrictInt = Field(ge=0, lt=EXPECTED_QUESTION_COUNT)
    question_id: QuestionId
    question_sha256: Digest
    status: Literal["technical_failure"]
    failure: EvalTechnicalFailureEvidence
    request: EvalFailureRequestEvidence
    usage: EvalFailureUsageEvidence

    @field_validator("failure", mode="before")
    @classmethod
    def strict_failure(cls, value: Any) -> EvalTechnicalFailureEvidence:
        return _strict_nested(  # type: ignore[return-value]
            EvalTechnicalFailureEvidence, value)

    @field_validator("request", mode="before")
    @classmethod
    def strict_request(cls, value: Any) -> EvalFailureRequestEvidence:
        return _strict_nested(  # type: ignore[return-value]
            EvalFailureRequestEvidence, value)

    @field_validator("usage", mode="before")
    @classmethod
    def strict_usage(cls, value: Any) -> EvalFailureUsageEvidence:
        return _strict_nested(  # type: ignore[return-value]
            EvalFailureUsageEvidence, value)


class Gold10EvalSuccessRow(_StrictFrozenModel):
    """One exact live Gold row revalidated by the remaining60 gate."""

    schema_version: Literal[HCX_SEMANTIC_INTENT_EVAL_VERSION]
    order: StrictInt = Field(ge=0, lt=EXPECTED_QUESTION_COUNT)
    question_id: QuestionId
    question_sha256: Digest
    status: Literal["success"]
    provider_wire: HcxSemanticIntentWire
    semantic_intent: SemanticIntent
    semantic_intent_digest: Digest
    boundary: EvalSemanticBoundaryEvidence
    gold_score: Gold10ScoreEvidence
    request: EvalRequestEvidence
    usage: EvalUsageEvidence

    @field_validator("provider_wire", mode="before")
    @classmethod
    def strict_wire(cls, value: Any) -> HcxSemanticIntentWire:
        return _strict_nested(HcxSemanticIntentWire, value)  # type: ignore[return-value]

    @field_validator("semantic_intent", mode="before")
    @classmethod
    def strict_intent(cls, value: Any) -> SemanticIntent:
        return _strict_nested(SemanticIntent, value)  # type: ignore[return-value]

    @field_validator("boundary", mode="before")
    @classmethod
    def strict_boundary(cls, value: Any) -> EvalSemanticBoundaryEvidence:
        return _strict_nested(  # type: ignore[return-value]
            EvalSemanticBoundaryEvidence, value)

    @field_validator("gold_score", mode="before")
    @classmethod
    def strict_score(cls, value: Any) -> Gold10ScoreEvidence:
        return _strict_nested(Gold10ScoreEvidence, value)  # type: ignore[return-value]

    @field_validator("request", mode="before")
    @classmethod
    def strict_request(cls, value: Any) -> EvalRequestEvidence:
        return _strict_nested(EvalRequestEvidence, value)  # type: ignore[return-value]

    @field_validator("usage", mode="before")
    @classmethod
    def strict_usage(cls, value: Any) -> EvalUsageEvidence:
        return _strict_nested(EvalUsageEvidence, value)  # type: ignore[return-value]

    @model_validator(mode="after")
    def validate_digest(self) -> "Gold10EvalSuccessRow":
        if self.semantic_intent_digest != semantic_intent_digest(
                self.semantic_intent):
            raise ValueError("Gold row semantic_intent_digest가 payload와 다릅니다")
        if self.gold_score.actual_intent_digest != self.semantic_intent_digest:
            raise ValueError("Gold row actual digest가 semantic intent와 다릅니다")
        if (
            self.boundary.question_sha256 != self.question_sha256
            or self.boundary.source_wire_digest
            != canonical_sha256(self.provider_wire)
            or self.boundary.semantic_intent_digest
            != self.semantic_intent_digest
        ):
            raise ValueError("Gold row boundary evidence binding이 다릅니다")
        return self


class EvalRowManifestEntry(_StrictFrozenModel):
    position: StrictInt = Field(ge=1)
    question_id: QuestionId
    relative_path: SafeRelativePath
    row_sha256: Digest
    semantic_intent_digest: Digest | None = None
    score_digest: Digest | None = None
    boundary_evidence_digest: Digest | None = None


class HcxSemanticIntentEvalSummary(_StrictFrozenModel):
    """Exact summary manifest; row bytes remain the evidence authority."""

    schema_version: Literal[HCX_SEMANTIC_INTENT_EVAL_SUMMARY_VERSION]
    eval_version: Literal[HCX_SEMANTIC_INTENT_EVAL_VERSION]
    semantic_boundary_version: Literal[SEMANTIC_INTENT_BOUNDARY_V1]
    phase: EvalPhase
    prompt_version: NonEmpty
    prompt_sha256: Digest
    provider_schema_sha256: Digest
    generation_config_sha256: Digest
    questions_sha256: Digest
    selected_count: StrictInt = Field(ge=1, le=EXPECTED_QUESTION_COUNT)
    completed_count: StrictInt = Field(ge=0, le=EXPECTED_QUESTION_COUNT)
    success_count: StrictInt = Field(ge=0, le=EXPECTED_QUESTION_COUNT)
    technical_failure_count: StrictInt = Field(ge=0, le=EXPECTED_QUESTION_COUNT)
    gold_scored_count: StrictInt = Field(ge=0, le=10)
    gold_exact_count: StrictInt = Field(ge=0, le=10)
    unscored_structural_count: StrictInt = Field(ge=0, le=EXPECTED_QUESTION_COUNT)
    selected_question_ids: list[QuestionId] = Field(min_length=1)
    rows: list[EvalRowManifestEntry] = Field(min_length=1)
    rows_digest: Digest
    gold10_gate_pass: StrictBool
    summary_digest: Digest

    @field_validator("rows", mode="before")
    @classmethod
    def strict_rows(cls, value: Any) -> list[EvalRowManifestEntry]:
        if not isinstance(value, list):
            raise TypeError("summary rows는 list여야 합니다")
        return [
            _strict_nested(EvalRowManifestEntry, row)  # type: ignore[list-item]
            for row in value
        ]

    @classmethod
    def compute_digest(cls, payload: Mapping[str, Any]) -> str:
        body = dict(payload)
        body.pop("summary_digest", None)
        return _canonical_sha256(body)

    @model_validator(mode="after")
    def validate_summary(self) -> "HcxSemanticIntentEvalSummary":
        if len(self.selected_question_ids) != len(set(self.selected_question_ids)):
            raise ValueError("selected_question_ids가 중복되었습니다")
        expected_positions = list(range(1, len(self.rows) + 1))
        if [row.position for row in self.rows] != expected_positions:
            raise ValueError("summary row position이 1부터 연속이어야 합니다")
        if [row.question_id for row in self.rows] != self.selected_question_ids:
            raise ValueError("summary row/question ID 순서가 다릅니다")
        if (
            self.selected_count != len(self.selected_question_ids)
            or self.completed_count != len(self.rows)
            or self.success_count + self.technical_failure_count
            != self.completed_count
            or self.gold_scored_count + self.unscored_structural_count
            != self.success_count
            or self.gold_exact_count > self.gold_scored_count
        ):
            raise ValueError("summary aggregate count가 row manifest와 다릅니다")
        row_payload = [row.model_dump(mode="json", warnings=False)
                       for row in self.rows]
        if self.rows_digest != _canonical_sha256(row_payload):
            raise ValueError("summary rows_digest가 row manifest와 다릅니다")
        expected_gate = (
            self.phase == "gold10"
            and self.selected_count == 10
            and self.completed_count == 10
            and self.success_count == 10
            and self.technical_failure_count == 0
            and self.gold_scored_count == 10
            and self.gold_exact_count == 10
            and self.unscored_structural_count == 0
            and all(row.semantic_intent_digest is not None
                    and row.score_digest is not None
                    and row.boundary_evidence_digest is not None
                    for row in self.rows)
        )
        if self.gold10_gate_pass is not expected_gate:
            raise ValueError("gold10_gate_pass가 aggregate에서 파생된 값과 다릅니다")
        if self.summary_digest != self.compute_digest(
                self.model_dump(mode="json", warnings=False)):
            raise ValueError("eval summary digest가 일치하지 않습니다")
        return self


@dataclass(frozen=True, slots=True)
class FrozenSemanticIntentCase:
    order: int
    question_id: str
    semantic_id: str
    group: str
    question: str = field(repr=False)
    question_sha256: str
    reference_date: str
    corpus_cutoff: str


@dataclass(frozen=True, slots=True)
class FrozenSemanticIntentQuestions:
    cases: tuple[FrozenSemanticIntentCase, ...]
    sha256: str
    path: Path = field(repr=False, compare=False)

    def select(self, phase: EvalPhase) -> tuple[FrozenSemanticIntentCase, ...]:
        gold = frozenset(MANDATORY_GOLD_QUESTION_IDS)
        if phase == "gold10":
            by_id = {row.question_id: row for row in self.cases}
            try:
                return tuple(by_id[question_id]
                             for question_id in MANDATORY_GOLD_QUESTION_IDS)
            except KeyError as exc:
                raise SemanticIntentEvalError(
                    "frozen 70에 mandatory Gold 문항이 없습니다") from exc
        if phase == "remaining60":
            rows = tuple(row for row in self.cases if row.question_id not in gold)
            if len(rows) != 60:
                raise SemanticIntentEvalError("remaining phase가 60문항이 아닙니다")
            return rows
        if phase == "all70":
            return self.cases
        raise SemanticIntentEvalError(f"지원하지 않는 eval phase입니다: {phase}")


def _select_eval_cases(
        questions: FrozenSemanticIntentQuestions, *,
        phase: EvalPhase,
        question_ids: Sequence[str] | None = None,
        ) -> tuple[FrozenSemanticIntentCase, ...]:
    selected = questions.select(phase)
    if question_ids is None:
        return selected
    requested = tuple(question_ids)
    if not requested:
        raise SemanticIntentEvalError("question_ids가 비어 있습니다")
    if len(requested) != len(set(requested)):
        raise SemanticIntentEvalError("question_ids가 중복되었습니다")
    by_id = {case.question_id: case for case in selected}
    unknown = [question_id for question_id in requested if question_id not in by_id]
    if unknown:
        raise SemanticIntentEvalError(
            f"선택한 phase에 없는 question_id입니다: {', '.join(unknown)}")
    return tuple(by_id[question_id] for question_id in requested)


def _expected_questions_digest(source: Path) -> str:
    """Return the digest that pins this questions file to its own release.

    A published release carries its own manifest, so the release is the
    authority for its bytes.  The frozen base archive has no manifest and keeps
    the module-level constant.  Neither path lets an unpinned file through.
    """

    manifest_path = source.parent / RELEASE_MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return EXPECTED_SOURCE_FIXTURE_DIGESTS["questions_v0.4.jsonl"]
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object)
        digest = manifest["release_fixture_digests"][source.name]
    except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise SemanticIntentEvalError(
            f"release manifest에서 questions digest를 읽지 못했습니다: {manifest_path}"
        ) from exc
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SemanticIntentEvalError("release manifest digest 형식이 잘못되었습니다")
    return digest


def load_frozen_semantic_intent_questions(
        path: str | Path = DEFAULT_QUESTIONS_PATH,
        ) -> FrozenSemanticIntentQuestions:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise SemanticIntentEvalError("frozen questions는 일반 파일이어야 합니다")
    payload = source.read_bytes()
    if (not payload or len(payload) > MAX_QUESTIONS_BYTES
            or payload.startswith(b"\xef\xbb\xbf")
            or b"\r" in payload or not payload.endswith(b"\n")):
        raise SemanticIntentEvalError("frozen questions byte 계약이 잘못되었습니다")
    digest = sha256(payload).hexdigest()
    if digest != _expected_questions_digest(source):
        raise SemanticIntentEvalError("frozen questions SHA-256이 동결값과 다릅니다")
    try:
        lines = payload.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise SemanticIntentEvalError("frozen questions가 UTF-8이 아닙니다") from exc
    if len(lines) != EXPECTED_QUESTION_COUNT:
        raise SemanticIntentEvalError("frozen questions가 정확히 70행이 아닙니다")

    cases: list[FrozenSemanticIntentCase] = []
    ids: set[str] = set()
    for index, line in enumerate(lines):
        try:
            raw = json.loads(line, object_pairs_hook=_strict_object)
            if not isinstance(raw, dict):
                raise SemanticIntentEvalError("frozen question 행은 object여야 합니다")
            row = QuestionFixtureV04.model_validate_json(line, strict=True)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise SemanticIntentEvalError(
                f"frozen question {index + 1}행이 strict하지 않습니다") from exc
        if line != _canonical_json(row.model_dump(mode="json", warnings=False)):
            raise SemanticIntentEvalError("frozen question 행이 canonical JSON이 아닙니다")
        if row.question_id in ids:
            raise SemanticIntentEvalError("frozen question ID가 중복됩니다")
        ids.add(row.question_id)
        cases.append(FrozenSemanticIntentCase(
            order=index,
            question_id=row.question_id,
            semantic_id=row.semantic_id,
            group=row.group,
            question=row.question,
            question_sha256=sha256(row.question.encode("utf-8")).hexdigest(),
            reference_date=row.reference_date.isoformat(),
            corpus_cutoff=str(row.corpus_cutoff),
        ))
    return FrozenSemanticIntentQuestions(
        cases=tuple(cases), sha256=digest, path=source.resolve())


@dataclass(frozen=True, slots=True)
class GoldSemanticIntentScore:
    exact: bool
    issue_codes: tuple[str, ...]
    expected_intent_digest: str
    actual_intent_digest: str


def score_gold_semantic_intent(
        actual: SemanticIntent,
        expected: SemanticIntent,
        *, expected_digest: str,
        actual_digest: str,
        ) -> GoldSemanticIntentScore:
    """Return component diagnostics while keeping exact digest authoritative."""

    actual_body = actual.model_dump(mode="json", warnings=False)
    expected_body = expected.model_dump(mode="json", warnings=False)
    components = (
        "entities",
        "answer_items",
        "answer_groups",
        "premises",
        "unresolved_mentions",
        "presentation",
    )
    issues = tuple(
        f"{name}_mismatch"
        for name in components
        if actual_body[name] != expected_body[name]
    )
    exact = actual_digest == expected_digest and not issues
    if exact != (actual_body == expected_body):
        raise SemanticIntentEvalError("Gold exact digest와 payload equality가 어긋났습니다")
    return GoldSemanticIntentScore(
        exact=exact,
        issue_codes=issues,
        expected_intent_digest=expected_digest,
        actual_intent_digest=actual_digest,
    )


def _success_record(
        case: FrozenSemanticIntentCase,
        invocation: SemanticIntentInvocation,
        gold: LoadedGoldExpectations | None,
        ) -> dict[str, object]:
    expected = None if gold is None else gold.by_question_id.get(case.question_id)
    score: GoldSemanticIntentScore | None = None
    if expected is not None:
        score = score_gold_semantic_intent(
            invocation.semantic_intent,
            expected.semantic_intent,
            expected_digest=expected.intent_digest,
            actual_digest=invocation.semantic_intent_digest,
        )
    return {
        "schema_version": HCX_SEMANTIC_INTENT_EVAL_VERSION,
        "order": case.order,
        "question_id": case.question_id,
        "question_sha256": case.question_sha256,
        "status": "success",
        "provider_wire": invocation.provider_wire.model_dump(
            mode="json", warnings=False),
        "semantic_intent": invocation.semantic_intent.model_dump(
            mode="json", warnings=False),
        "semantic_intent_digest": invocation.semantic_intent_digest,
        "boundary": invocation.boundary_evidence.as_dict(),
        "gold_score": None if score is None else {
            "exact": score.exact,
            "issue_codes": list(score.issue_codes),
            "expected_intent_digest": score.expected_intent_digest,
            "actual_intent_digest": score.actual_intent_digest,
        },
        "request": {
            "request_id": invocation.request_id,
            "attempts": invocation.attempts,
            "provider_schema_sha256": invocation.provider_schema_sha256,
            "system_prompt_sha256": invocation.system_prompt_sha256,
            "request_prompt_sha256": invocation.request_prompt_sha256,
            "generation_config_sha256": invocation.generation_config_sha256,
        },
        "usage": {
            "total_latency_ms": invocation.total_latency_ms,
            "prompt_tokens": invocation.prompt_tokens,
            "completion_tokens": invocation.completion_tokens,
            "total_tokens": invocation.total_tokens,
        },
    }


def _failure_record(
        case: FrozenSemanticIntentCase,
        error: Exception,
        *, prompt_sha256: str,
        schema_sha256: str,
        config_sha256: str,
        request_prompt_sha256: str,
        ) -> dict[str, object]:
    request_id: str | None = None
    attempts = 0
    usage_known = False
    latency_ms = 0.0
    prompt_tokens = completion_tokens = total_tokens = 0
    shape: HcxSafePayloadShape | None = None
    schema_repair_codes: tuple[str, ...] = ()
    normalization_codes: tuple[str, ...] = ()
    if isinstance(error, HcxError):
        layer = "schema" if error.code in {
            "response_validation_failed", "invalid_schema",
            "incomplete_generation",
        } else "transport"
        code = error.code
        issue_codes = error.diagnostic_codes
        issue_paths = error.diagnostic_paths
        shape = error.diagnostic_shape
        schema_repair_codes = error.repairs
        request_id = error.request_id
        attempts = error.attempts
        usage_known = error.usage_known
        latency_ms = error.latency_ms
        prompt_tokens = error.prompt_tokens
        completion_tokens = error.completion_tokens
        total_tokens = error.total_tokens
    elif isinstance(error, HcxSemanticIntentNormalizationInvocationError):
        layer, code = "normalization", "grounding_rejected"
        issue_codes = error.diagnostic_codes
        issue_paths = error.diagnostic_paths
        shape = error.diagnostic_shape
        schema_repair_codes = error.schema_repair_codes
        normalization_codes = error.normalization_codes
        request_id = error.request_id
        attempts = error.attempts
        usage_known = True
        latency_ms = error.total_latency_ms
        prompt_tokens = error.prompt_tokens
        completion_tokens = error.completion_tokens
        total_tokens = error.total_tokens
        if (
            error.provider_schema_sha256 != schema_sha256
            or error.system_prompt_sha256 != prompt_sha256
            or error.request_prompt_sha256 != request_prompt_sha256
            or error.generation_config_sha256 != config_sha256
        ):
            raise SemanticIntentEvalError(
                "normalization failure가 exact provider request와 다릅니다")
    elif isinstance(error, SemanticIntentNormalizationError):
        layer, code = "normalization", "grounding_rejected"
        issue_codes = error.diagnostic_codes
        issue_paths = error.diagnostic_paths
        normalization_codes = tuple(getattr(
            error, "normalization_codes", ()))
    elif isinstance(error, HcxSemanticIntentInvocationError):
        layer, code = "binding", "request_binding_mismatch"
        issue_codes = ()
        issue_paths = ()
    elif isinstance(error, ValidationError):
        layer, code = "local_contract", "strict_contract_rejected"
        issue_codes = safe_validation_issue_codes(error)
        issue_paths = safe_validation_issue_paths(error)
    else:
        layer, code = "local_contract", "strict_contract_rejected"
        issue_codes = ()
        issue_paths = ()
    payload = {
        "schema_version": HCX_SEMANTIC_INTENT_EVAL_VERSION,
        "order": case.order,
        "question_id": case.question_id,
        "question_sha256": case.question_sha256,
        "status": "technical_failure",
        "failure": {
            "layer": layer,
            "code": code,
            "issue_codes": list(issue_codes),
            "issue_paths": list(issue_paths),
            "boundary_version": SEMANTIC_INTENT_BOUNDARY_V1,
            "schema_repair_codes": list(schema_repair_codes),
            "normalization_codes": list(normalization_codes),
            "payload_shape": None if shape is None else shape.as_dict(),
        },
        "request": {
            "request_id": request_id,
            "attempts": attempts,
            "provider_schema_sha256": schema_sha256,
            "system_prompt_sha256": prompt_sha256,
            "request_prompt_sha256": request_prompt_sha256,
            "generation_config_sha256": config_sha256,
        },
        "usage": {
            "known": usage_known,
            "total_latency_ms": float(latency_ms),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
    try:
        validated = TechnicalFailureEvalRow.model_validate_json(
            _canonical_json(payload), strict=True)
    except ValidationError as exc:
        raise SemanticIntentEvalError(
            "technical failure row strict 생성에 실패했습니다") from exc
    return validated.model_dump(mode="json", warnings=False)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: object) -> None:
    _atomic_write(path, (_canonical_json(value) + "\n").encode("utf-8"))


def build_eval_summary(
        *, phase: EvalPhase,
        selected_cases: tuple[FrozenSemanticIntentCase, ...],
        records: list[Mapping[str, object]],
        questions_sha256: str,
        prompt_sha256: str,
        provider_schema_sha256: str,
        generation_config_sha256: str,
        ) -> dict[str, object]:
    if len(records) != len(selected_cases):
        raise SemanticIntentEvalError(
            "summary selected case와 result row 수가 다릅니다")
    successful = [row for row in records if row.get("status") == "success"]
    failed = [row for row in records if row.get("status") != "success"]
    gold_scores = [
        row.get("gold_score") for row in successful
        if isinstance(row.get("gold_score"), Mapping)
    ]
    exact_count = sum(
        1 for score in gold_scores
        if isinstance(score, Mapping) and score.get("exact") is True)
    row_entries: list[dict[str, object]] = []
    for position, (case, record) in enumerate(
            zip(selected_cases, records, strict=True), start=1):
        row_bytes = (_canonical_json(dict(record)) + "\n").encode("utf-8")
        score = record.get("gold_score")
        boundary = record.get("boundary")
        intent_digest = record.get("semantic_intent_digest")
        row_entries.append({
            "position": position,
            "question_id": case.question_id,
            "relative_path": f"rows/{position:02d}_{case.question_id}.json",
            "row_sha256": sha256(row_bytes).hexdigest(),
            "semantic_intent_digest": (
                intent_digest
                if isinstance(intent_digest, str) else None),
            "score_digest": (
                _canonical_sha256(dict(score))
                if isinstance(score, Mapping) else None),
            "boundary_evidence_digest": (
                boundary.get("evidence_digest")
                if isinstance(boundary, Mapping)
                and isinstance(boundary.get("evidence_digest"), str)
                else None),
        })
    summary: dict[str, object] = {
        "schema_version": HCX_SEMANTIC_INTENT_EVAL_SUMMARY_VERSION,
        "eval_version": HCX_SEMANTIC_INTENT_EVAL_VERSION,
        "semantic_boundary_version": SEMANTIC_INTENT_BOUNDARY_V1,
        "phase": phase,
        "prompt_version": HCX_SEMANTIC_INTENT_PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "provider_schema_sha256": provider_schema_sha256,
        "generation_config_sha256": generation_config_sha256,
        "questions_sha256": questions_sha256,
        "selected_count": len(selected_cases),
        "completed_count": len(records),
        "success_count": len(successful),
        "technical_failure_count": len(failed),
        "gold_scored_count": len(gold_scores),
        "gold_exact_count": exact_count,
        "unscored_structural_count": len(successful) - len(gold_scores),
        "selected_question_ids": [case.question_id for case in selected_cases],
        "rows": row_entries,
        "rows_digest": _canonical_sha256(row_entries),
    }
    summary["gold10_gate_pass"] = (
        phase == "gold10"
        and len(selected_cases) == 10
        and len(records) == 10
        and not failed
        and len(gold_scores) == 10
        and exact_count == 10
    )
    summary["summary_digest"] = HcxSemanticIntentEvalSummary.compute_digest(summary)
    try:
        validated = HcxSemanticIntentEvalSummary.model_validate_json(
            _canonical_json(summary), strict=True)
    except ValidationError as exc:
        raise SemanticIntentEvalError("eval summary strict 생성에 실패했습니다") from exc
    return validated.model_dump(mode="json", warnings=False)


def _secure_result_row(
        root: Path, relative_path: str, *, expected_path: str,
        ) -> tuple[Path, bytes]:
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != relative_path
        or relative_path != expected_path
        or len(relative.parts) != 2
        or relative.parts[0] != "rows"
    ):
        raise SemanticIntentEvalError("Gold10 row relative path가 안전하지 않습니다")
    root_abs = Path(os.path.abspath(os.fspath(root)))
    candidate = root_abs / relative
    current = root_abs
    try:
        for component in relative.parts:
            current /= component
            if current.is_symlink():
                raise SemanticIntentEvalError("Gold10 row path에 symlink가 있습니다")
        if not candidate.is_file():
            raise SemanticIntentEvalError("Gold10 row가 일반 파일이 아닙니다")
        candidate.resolve(strict=True).relative_to(root_abs.resolve(strict=True))
        payload = candidate.read_bytes()
    except SemanticIntentEvalError:
        raise
    except (OSError, ValueError) as exc:
        raise SemanticIntentEvalError("Gold10 row path를 안전하게 읽지 못했습니다") from exc
    if (
        not payload
        or len(payload) > MAX_RESULT_ROW_BYTES
        or payload.startswith(b"\xef\xbb\xbf")
        or b"\r" in payload
        or not payload.endswith(b"\n")
    ):
        raise SemanticIntentEvalError("Gold10 row byte 계약이 잘못되었습니다")
    return candidate, payload


def validate_gold10_gate_summary(
        path: str | Path, *,
        prompt_sha256: str,
        provider_schema_sha256: str,
        generation_config_sha256: str,
        questions: FrozenSemanticIntentQuestions,
        request_prompt_sha256_by_question: Mapping[str, str],
        ) -> Mapping[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise SemanticIntentEvalError("Gold10 gate summary는 일반 파일이어야 합니다")
    payload = source.read_bytes()
    if (not payload or len(payload) > MAX_SUMMARY_BYTES
            or b"\r" in payload or not payload.endswith(b"\n")):
        raise SemanticIntentEvalError("Gold10 gate summary byte 계약이 잘못되었습니다")
    try:
        text = payload.decode("utf-8", errors="strict")
        raw = json.loads(text, object_pairs_hook=_strict_object)
        summary = HcxSemanticIntentEvalSummary.model_validate_json(
            payload, strict=True)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise SemanticIntentEvalError(
            "Gold10 gate summary strict JSON이 유효하지 않습니다") from exc
    if (not isinstance(raw, dict)
            or text != _canonical_json(summary.model_dump(
                mode="json", warnings=False)) + "\n"):
        raise SemanticIntentEvalError("Gold10 gate summary가 canonical exact object가 아닙니다")
    expected = {
        "schema_version": HCX_SEMANTIC_INTENT_EVAL_SUMMARY_VERSION,
        "eval_version": HCX_SEMANTIC_INTENT_EVAL_VERSION,
        "semantic_boundary_version": SEMANTIC_INTENT_BOUNDARY_V1,
        "phase": "gold10",
        "prompt_version": HCX_SEMANTIC_INTENT_PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "provider_schema_sha256": provider_schema_sha256,
        "generation_config_sha256": generation_config_sha256,
        "questions_sha256": questions.sha256,
        "selected_count": 10,
        "completed_count": 10,
        "success_count": 10,
        "technical_failure_count": 0,
        "gold_scored_count": 10,
        "gold_exact_count": 10,
        "unscored_structural_count": 0,
        "gold10_gate_pass": True,
    }
    for key, value in expected.items():
        if getattr(summary, key) != value:
            raise SemanticIntentEvalError(f"Gold10 gate summary가 현재 run과 다릅니다: {key}")
    selected = questions.select("gold10")
    expected_ids = list(MANDATORY_GOLD_QUESTION_IDS)
    if (
        summary.selected_question_ids != expected_ids
        or [case.question_id for case in selected] != expected_ids
    ):
        raise SemanticIntentEvalError("Gold10 ordered question ID가 frozen authority와 다릅니다")
    if set(request_prompt_sha256_by_question) != set(expected_ids):
        raise SemanticIntentEvalError("Gold10 request prompt hash inventory가 다릅니다")
    if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in request_prompt_sha256_by_question.values()):
        raise SemanticIntentEvalError("Gold10 request prompt hash 형식이 잘못되었습니다")

    root = source.parent
    rows_root = root / "rows"
    if rows_root.is_symlink() or not rows_root.is_dir():
        raise SemanticIntentEvalError("Gold10 rows directory가 안전하지 않습니다")
    expected_filenames = {
        Path(entry.relative_path).name for entry in summary.rows}
    actual_entries = list(rows_root.iterdir())
    if (
        any(entry.is_symlink() or not entry.is_file()
            for entry in actual_entries)
        or {entry.name for entry in actual_entries} != expected_filenames
    ):
        raise SemanticIntentEvalError("Gold10 rows directory inventory가 정확하지 않습니다")

    gold = load_gold_expectations()
    validated_rows: list[Gold10EvalSuccessRow] = []
    for entry, case in zip(summary.rows, selected, strict=True):
        expected_relative = (
            f"rows/{entry.position:02d}_{case.question_id}.json")
        _, row_payload = _secure_result_row(
            root, entry.relative_path, expected_path=expected_relative)
        if sha256(row_payload).hexdigest() != entry.row_sha256:
            raise SemanticIntentEvalError("Gold10 row byte SHA-256이 manifest와 다릅니다")
        try:
            row_raw = json.loads(
                row_payload.decode("utf-8", errors="strict"),
                object_pairs_hook=_strict_object,
            )
            row = Gold10EvalSuccessRow.model_validate_json(
                row_payload, strict=True)
        except (
                UnicodeDecodeError, json.JSONDecodeError, ValidationError,
                ) as exc:
            raise SemanticIntentEvalError("Gold10 result row strict 검증 실패") from exc
        if (
            not isinstance(row_raw, dict)
            or row_payload.decode("utf-8")
            != _canonical_json(row.model_dump(mode="json", warnings=False)) + "\n"
        ):
            raise SemanticIntentEvalError("Gold10 result row가 canonical exact object가 아닙니다")
        if (
            row.question_id != case.question_id
            or row.order != case.order
            or row.question_sha256 != case.question_sha256
        ):
            raise SemanticIntentEvalError("Gold10 row question identity가 frozen row와 다릅니다")
        if (
            row.request.provider_schema_sha256 != provider_schema_sha256
            or row.request.system_prompt_sha256 != prompt_sha256
            or row.request.generation_config_sha256 != generation_config_sha256
            or row.request.request_prompt_sha256
            != request_prompt_sha256_by_question[case.question_id]
        ):
            raise SemanticIntentEvalError("Gold10 row가 exact provider request와 다릅니다")

        bounded = normalize_semantic_intent_bounded(
            case.question, row.provider_wire)
        computed_boundary = SemanticIntentBoundaryEvidence.create(
            bounded,
            schema_repair_codes=tuple(row.boundary.schema_repair_codes),
        )
        normalized = bounded.semantic_intent
        if (
            normalized.model_dump(mode="json", warnings=False)
            != row.semantic_intent.model_dump(mode="json", warnings=False)
            or bounded.semantic_intent_digest != row.semantic_intent_digest
            or computed_boundary.as_dict()
            != row.boundary.model_dump(mode="json", warnings=False)
            or entry.boundary_evidence_digest
            != row.boundary.evidence_digest
        ):
            raise SemanticIntentEvalError("Gold10 provider wire normalization proof가 다릅니다")
        expected_gold = gold.by_question_id.get(case.question_id)
        if expected_gold is None:
            raise SemanticIntentEvalError("Gold10 expectation row가 없습니다")
        computed = score_gold_semantic_intent(
            normalized,
            expected_gold.semantic_intent,
            expected_digest=expected_gold.intent_digest,
            actual_digest=row.semantic_intent_digest,
        )
        computed_score = {
            "exact": computed.exact,
            "issue_codes": list(computed.issue_codes),
            "expected_intent_digest": computed.expected_intent_digest,
            "actual_intent_digest": computed.actual_intent_digest,
        }
        if (
            not computed.exact
            or row.gold_score.model_dump(mode="json", warnings=False)
            != computed_score
            or entry.semantic_intent_digest != row.semantic_intent_digest
            or entry.score_digest != _canonical_sha256(computed_score)
        ):
            raise SemanticIntentEvalError("Gold10 exact semantic score proof가 다릅니다")
        validated_rows.append(row)

    if len(validated_rows) != 10:
        raise SemanticIntentEvalError("Gold10 result row가 정확히 10개가 아닙니다")
    return summary.model_dump(mode="json", warnings=False)


GOLD10_PREREQUISITE_VERSION = "hcx-semantic-intent-eval-prerequisite/1.0"


@dataclass(frozen=True, slots=True)
class Gold10Prerequisite:
    """검증을 **실제로 통과한** Gold10 증거. 10차 외부 검수 P1-EVAL-002.

    예전에는 `run_live_evaluation` 이 임의의 `Mapping` 을 받아 네 해시만 비교했다.
    그래서 CLI 를 우회한 내부 호출이 **가짜 증거로 `gated=true` 경로에 들어갈 수**
    있었다 — summary/row proof 는 재검증하지 않았다.

    이제 이 타입은 `require_gold10_gate` 만 만든다. dict 를 지어내도 타입이 다르므로
    받아들여지지 않고, 값 자체도 검증기가 돌려준 것만 담긴다.
    """

    summary_path: Path
    gold10_summary_sha256: str
    gold10_summary_digest: str
    gold10_rows_digest: str
    prompt_sha256: str
    provider_schema_sha256: str
    generation_config_sha256: str
    questions_sha256: str

    def as_record(self) -> "dict[str, object]":
        return {
            "schema_version": GOLD10_PREREQUISITE_VERSION,
            "gold10_summary_sha256": self.gold10_summary_sha256,
            "gold10_summary_digest": self.gold10_summary_digest,
            "gold10_rows_digest": self.gold10_rows_digest,
            "prompt_sha256": self.prompt_sha256,
            "provider_schema_sha256": self.provider_schema_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "questions_sha256": self.questions_sha256,
        }

    def reverify(self) -> None:
        """**쓰기 직전에 파일을 다시 읽어 대조한다.**

        검증 뒤 실행 도중에 summary 가 바뀌었다면 증거가 아니다.
        """

        source = Path(self.summary_path)
        if source.is_symlink() or not source.is_file():
            raise SemanticIntentEvalError("Gold10 gate summary 가 사라졌습니다")
        if sha256(source.read_bytes()).hexdigest() != self.gold10_summary_sha256:
            raise SemanticIntentEvalError(
                "Gold10 gate summary 가 검증 이후 바뀌었습니다")


def require_gold10_gate(
        summary_path: str | Path, *,
        runner: HcxSemanticIntentRunner,
        questions: FrozenSemanticIntentQuestions,
        ) -> "Gold10Prerequisite":
    """remaining60 를 돌리기 **전에** Gold10 10/10 을 검증한다. 9차 검수 P0-EVAL-001.

    `validate_gold10_gate_summary` 는 이미 촘촘했지만 **아무도 호출하지 않았다.**
    그래서 Gold10 이 exact 2/10 · `gold10_gate_pass=false` 인 프롬프트로도
    remaining60 live 를 돌려 산출물을 만들 수 있었다. 평가 순서와 비용 통제가
    코드가 아니라 사람의 약속에만 걸려 있었다.

    **provider client 를 만들기 전에** 부른다. 검증에 실패하면 호출은 0회다.

    질문별 request prompt 해시는 `runner.build_request(...)` 로 **호출 없이**
    계산한다 — 게이트 자체가 토큰을 쓰지 않는다.
    """

    gold10_cases = questions.select("gold10")
    request_hashes = {
        case.question_id: runner.build_request(case.question).prompt_hash
        for case in gold10_cases
    }
    summary = validate_gold10_gate_summary(
        summary_path,
        prompt_sha256=runner.prompt.sha256,
        provider_schema_sha256=runner.compiled_schema.sha256,
        generation_config_sha256=runner.generation_config.fingerprint,
        questions=questions,
        request_prompt_sha256_by_question=request_hashes,
    )
    source = Path(summary_path)
    return Gold10Prerequisite(
        summary_path=source.resolve(),
        gold10_summary_sha256=sha256(source.read_bytes()).hexdigest(),
        gold10_summary_digest=str(summary["summary_digest"]),
        gold10_rows_digest=str(summary["rows_digest"]),
        prompt_sha256=runner.prompt.sha256,
        provider_schema_sha256=runner.compiled_schema.sha256,
        generation_config_sha256=runner.generation_config.fingerprint,
        questions_sha256=questions.sha256,
    )


def run_live_evaluation(
        runner: HcxSemanticIntentRunner, *,
        phase: EvalPhase,
        questions: FrozenSemanticIntentQuestions,
        output_dir: str | Path,
        max_calls: int | None = None,
        question_ids: Sequence[str] | None = None,
        gold10_prerequisite: "Gold10Prerequisite | None" = None,
        rate_limit_retries: int = 0,
        ) -> Mapping[str, object]:
    """Persist a selected run without opening final expected handoffs.

    ``all70`` is the official production/evaluation path.  Its output contains
    no per-row score and the expected fixture is intentionally not loaded here.
    Historical split phases retain their former local scoring metadata only for
    compatibility with already issued diagnostic artifacts.
    """

    runner.prompt.require_live_approval()
    gold = load_gold_expectations() if phase != "all70" else None
    # **차단하지 않는다** (사용자 결정 2026-08-22).
    #
    # 9차 검수 P0-EVAL-001 로 remaining60 을 Gold10 10/10 뒤로 막아 두었는데, 그
    # 게이트가 **규칙이 먹히는지 확인하는 진단까지** 막았다. Gold10 이 2/9 인 동안은
    # remaining60 을 한 번도 못 태우고, 그래서 remaining60 에만 있는 실패를 고칠
    # 근거를 얻을 수 없다.
    #
    # 차단은 걷어내되 **무엇을 근거로 돌았는지는 남긴다.** 증거가 없는 실행은
    # `gold10_prerequisite.json` 에 `gated=false` 로 기록되어, 나중에 어떤 산출물이
    # 게이트를 통과한 것인지 구분할 수 있다.
    if gold10_prerequisite is not None:
        # **검증기가 만든 토큰만 받는다** (10차 검수 P1-EVAL-002). 예전에는 임의의
        # Mapping 을 받아 네 해시만 비교해서, 지어낸 dict 로 `gated=true` 경로에
        # 들어갈 수 있었다. 타입을 좁히면 그 길이 막힌다.
        if not isinstance(gold10_prerequisite, Gold10Prerequisite):
            raise SemanticIntentEvalError(
                "Gold10 prerequisite 는 require_gold10_gate 가 만든 것이어야 합니다")
        # gold10 이 자기 자신의 통과를 전제로 삼는 것은 순환이다.
        if phase != "remaining60":
            raise SemanticIntentEvalError(
                "gold10 phase 에는 Gold10 prerequisite 를 넣지 않습니다")
        if (gold10_prerequisite.prompt_sha256 != runner.prompt.sha256
                or gold10_prerequisite.provider_schema_sha256
                != runner.compiled_schema.sha256
                or gold10_prerequisite.generation_config_sha256
                != runner.generation_config.fingerprint
                or gold10_prerequisite.questions_sha256 != questions.sha256):
            raise SemanticIntentEvalError(
                "Gold10 prerequisite 가 이 run 의 prompt/schema/config/questions 와 다릅니다")
        # 검증 이후 summary 가 바뀌지 않았는지 **호출 전에** 다시 확인한다.
        gold10_prerequisite.reverify()
    selected = _select_eval_cases(
        questions, phase=phase, question_ids=question_ids)
    if max_calls is not None:
        if type(max_calls) is not int or not 1 <= max_calls <= len(selected):
            raise SemanticIntentEvalError("max_calls가 phase 범위 밖입니다")
        selected = selected[:max_calls]
    destination = Path(output_dir)
    if (destination.is_symlink()
            or (destination.exists() and (
                not destination.is_dir() or any(destination.iterdir())))):
        raise SemanticIntentEvalError("evaluation output directory는 새 빈 경로여야 합니다")
    rows_dir = destination / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    before_digest = sha256(questions.path.read_bytes()).hexdigest()
    records: list[Mapping[str, object]] = []
    #: 재전송한 문항과 횟수. 성적이 아니라 **측정 신뢰도**의 기록이다.
    retransmissions: dict[str, int] = {}

    def _call_once(case: object) -> "tuple[Mapping[str, object], str | None]":
        """한 번 호출하고 (record, 실패코드) 를 돌려준다."""

        request_prompt_sha256 = runner.build_request(case.question).prompt_hash
        try:
            return _success_record(case, runner.invoke(case.question), gold), None
        except (
                HcxError,
                SemanticIntentNormalizationError,
                HcxSemanticIntentInvocationError,
                ValidationError,
                ) as exc:
            record = _failure_record(
                case, exc,
                prompt_sha256=runner.prompt.sha256,
                schema_sha256=runner.compiled_schema.sha256,
                config_sha256=runner.generation_config.fingerprint,
                request_prompt_sha256=request_prompt_sha256,
            )
            return record, getattr(exc, "code", None)

    for position, case in enumerate(selected, start=1):
        request_prompt_sha256 = runner.build_request(case.question).prompt_hash
        try:
            invocation = runner.invoke(case.question)
            record = _success_record(case, invocation, gold)
        except (
                HcxError,
                SemanticIntentNormalizationError,
                HcxSemanticIntentInvocationError,
                ValidationError,
                ) as exc:
            record = _failure_record(
                case,
                exc,
                prompt_sha256=runner.prompt.sha256,
                schema_sha256=runner.compiled_schema.sha256,
                config_sha256=runner.generation_config.fingerprint,
                request_prompt_sha256=request_prompt_sha256,
            )
        # **속도 제한은 성적이 아니다. 같은 요청을 다시 보낸다.**
        #
        # 실행마다 4~7건이 `rate_limited` 로 떨어져, 한 번에 60건을 태우지 못했다.
        # 그래서 개선인지 속도 제한인지 헷갈리는 일이 반복됐다 — 실제로 「보정이
        # 3건을 깼다」고 잘못 읽은 적이 있다.
        #
        # 재시도는 **transport 재전송**이지 semantic 재시도가 아니다. 같은 질문·같은
        # prompt hash 로 새 호출을 낸다. 그래서 각 행은 `attempts=1` 을 유지하고
        # 증거 계약(`EvalRequestEvidence.attempts: Literal[1]`)을 건드리지 않는다.
        # 몇 번 재전송했는지는 sidecar 에 남긴다.
        attempt = 0
        while (attempt < rate_limit_retries
               and (record.get("failure") or {}).get("code") == "rate_limited"):
            attempt += 1
            retransmissions[case.question_id] = attempt
            record, _ = _call_once(case)

        records.append(record)
        _write_json(rows_dir / f"{position:02d}_{case.question_id}.json", record)

    if sha256(questions.path.read_bytes()).hexdigest() != before_digest:
        raise SemanticIntentEvalError("evaluation 중 frozen questions bytes가 바뀌었습니다")
    summary = build_eval_summary(
        phase=phase,
        selected_cases=selected,
        records=records,
        questions_sha256=questions.sha256,
        prompt_sha256=runner.prompt.sha256,
        provider_schema_sha256=runner.compiled_schema.sha256,
        generation_config_sha256=runner.generation_config.fingerprint,
    )
    _write_json(destination / "summary.json", summary)
    if retransmissions:
        # summary 모델은 동결이므로 sidecar 로 남긴다. 이 실행의 숫자를 읽을 때
        # **몇 건이 재전송을 거쳤는지** 함께 봐야 한다.
        _write_json(destination / "rate_limit_retransmissions.json", {
            "schema_version": "hcx-semantic-intent-eval-retransmission/1.0",
            "note": "속도 제한으로 같은 요청을 다시 보낸 문항. 각 행은 attempts=1 이다.",
            "max_retries": rate_limit_retries,
            "questions": dict(sorted(retransmissions.items())),
            "summary_digest": summary["summary_digest"],
        })
    if phase == "remaining60" and gold10_prerequisite is None:
        # 게이트 증거 없이 돈 실행이라는 사실을 산출물에 남긴다.
        _write_json(destination / "gold10_prerequisite.json", {
            "schema_version": GOLD10_PREREQUISITE_VERSION,
            "gated": False,
            "note": "Gold10 10/10 증거 없이 실행됨 (차단 해제, 2026-08-22)",
            "prompt_sha256": runner.prompt.sha256,
            "provider_schema_sha256": runner.compiled_schema.sha256,
            "generation_config_sha256": runner.generation_config.fingerprint,
            "questions_sha256": questions.sha256,
            "remaining60_summary_digest": summary["summary_digest"],
            "remaining60_rows_digest": summary["rows_digest"],
        })
    if gold10_prerequisite is not None:
        # **summary 모델에 필드를 더하지 않는다.** 더하면 canonical JSON 과
        # `summary_digest` 가 바뀌어 이미 발행된 Gold10 summary 가 전부 거부된다.
        # 대신 양방향으로 묶인 sidecar 를 남긴다 — 어느 Gold10 증거 위에서 돌았는지
        # 와 그 결과 summary 가 무엇인지가 함께 박힌다.
        gold10_prerequisite.reverify()
        _write_json(destination / "gold10_prerequisite.json", {
            **gold10_prerequisite.as_record(),
            "gated": True,
            "remaining60_summary_digest": summary["summary_digest"],
            "remaining60_rows_digest": summary["rows_digest"],
        })
    return summary


def build_dry_run_plan(
        runner: HcxSemanticIntentRunner, *,
        phase: EvalPhase,
        questions: FrozenSemanticIntentQuestions,
        question_ids: Sequence[str] | None = None,
        ) -> Mapping[str, object]:
    selected = _select_eval_cases(
        questions, phase=phase, question_ids=question_ids)
    sample_request = runner.build_request(selected[0].question)
    user_body = json.loads(sample_request.messages[1].content)
    if set(user_body) != {"question"}:
        raise SemanticIntentEvalError("provider user message에 question 외 field가 섞였습니다")
    return {
        "eval_version": HCX_SEMANTIC_INTENT_EVAL_VERSION,
        "semantic_boundary_version": SEMANTIC_INTENT_BOUNDARY_V1,
        "mode": "dry_run_no_provider_call",
        "phase": phase,
        "selected_count": len(selected),
        "selected_question_ids": [case.question_id for case in selected],
        "questions_sha256": questions.sha256,
        "prompt_version": runner.prompt.prompt_version,
        "prompt_approval_state": runner.prompt.approval_state,
        "prompt_sha256": runner.prompt.sha256,
        "provider_schema_sha256": runner.compiled_schema.sha256,
        "generation_config_sha256": runner.generation_config.fingerprint,
        "provider_user_fields": ["question"],
        "expectation_fields_sent_to_provider": [],
        "gold10_scoring": "exact_semantic_intent",
        "remaining60_scoring": "strict_schema_and_grounding_only",
        "live_all70_allowed": True,
    }


def _schema_artifact_bytes(
        model: type[BaseModel], artifact: Path,
        ) -> tuple[bytes, bytes, str]:
    schema = _canonical_json(
        model.model_json_schema(mode="validation")).encode("utf-8")
    digest = sha256(schema).hexdigest()
    sidecar = f"{digest}  {artifact.name}\n".encode("ascii")
    return schema, sidecar, digest


def _eval_schema_artifacts() -> tuple[tuple[Path, bytes], ...]:
    summary, summary_sidecar, _ = _schema_artifact_bytes(
        HcxSemanticIntentEvalSummary, SUMMARY_SCHEMA_ARTIFACT)
    row, row_sidecar, _ = _schema_artifact_bytes(
        Gold10EvalSuccessRow, GOLD10_ROW_SCHEMA_ARTIFACT)
    failure, failure_sidecar, _ = _schema_artifact_bytes(
        TechnicalFailureEvalRow, TECHNICAL_FAILURE_ROW_SCHEMA_ARTIFACT)
    return (
        (SUMMARY_SCHEMA_ARTIFACT, summary),
        (SUMMARY_SCHEMA_DIGEST_ARTIFACT, summary_sidecar),
        (GOLD10_ROW_SCHEMA_ARTIFACT, row),
        (GOLD10_ROW_SCHEMA_DIGEST_ARTIFACT, row_sidecar),
        (TECHNICAL_FAILURE_ROW_SCHEMA_ARTIFACT, failure),
        (TECHNICAL_FAILURE_ROW_SCHEMA_DIGEST_ARTIFACT, failure_sidecar),
    )


def write_hcx_semantic_intent_eval_schema_artifacts() -> tuple[str, str, str]:
    for path, payload in _eval_schema_artifacts():
        _atomic_write(path, payload)
    return (
        _schema_artifact_bytes(
            HcxSemanticIntentEvalSummary, SUMMARY_SCHEMA_ARTIFACT)[2],
        _schema_artifact_bytes(
            Gold10EvalSuccessRow, GOLD10_ROW_SCHEMA_ARTIFACT)[2],
        _schema_artifact_bytes(
            TechnicalFailureEvalRow,
            TECHNICAL_FAILURE_ROW_SCHEMA_ARTIFACT)[2],
    )


def verify_hcx_semantic_intent_eval_schema_artifacts() -> tuple[str, str, str]:
    for path, expected in _eval_schema_artifacts():
        try:
            actual = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"eval schema artifact가 없습니다: {path}") from exc
        if actual != expected:
            raise RuntimeError(f"eval schema artifact drift: {path}")
    return (
        _schema_artifact_bytes(
            HcxSemanticIntentEvalSummary, SUMMARY_SCHEMA_ARTIFACT)[2],
        _schema_artifact_bytes(
            Gold10EvalSuccessRow, GOLD10_ROW_SCHEMA_ARTIFACT)[2],
        _schema_artifact_bytes(
            TechnicalFailureEvalRow,
            TECHNICAL_FAILURE_ROW_SCHEMA_ARTIFACT)[2],
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    digests = (
        write_hcx_semantic_intent_eval_schema_artifacts()
        if args.write else verify_hcx_semantic_intent_eval_schema_artifacts())
    print(
        f"PASS: {HCX_SEMANTIC_INTENT_EVAL_SUMMARY_VERSION}={digests[0]} "
        f"{HCX_SEMANTIC_INTENT_EVAL_GOLD10_ROW_VERSION}={digests[1]} "
        f"{HCX_SEMANTIC_INTENT_EVAL_TECHNICAL_FAILURE_ROW_VERSION}={digests[2]}"
    )
    return 0


__all__ = [
    "DEFAULT_QUESTIONS_PATH",
    "EXPECTED_QUESTION_COUNT",
    "FrozenSemanticIntentCase",
    "FrozenSemanticIntentQuestions",
    "GOLD10_ROW_SCHEMA_ARTIFACT",
    "GOLD10_ROW_SCHEMA_DIGEST_ARTIFACT",
    "HCX_SEMANTIC_INTENT_EVAL_TECHNICAL_FAILURE_ROW_VERSION",
    "GoldSemanticIntentScore",
    "Gold10EvalSuccessRow",
    "HcxSemanticIntentEvalSummary",
    "HCX_SEMANTIC_INTENT_EVAL_GOLD10_ROW_VERSION",
    "HCX_SEMANTIC_INTENT_EVAL_SUMMARY_VERSION",
    "HCX_SEMANTIC_INTENT_EVAL_VERSION",
    "SemanticIntentEvalError",
    "SUMMARY_SCHEMA_ARTIFACT",
    "SUMMARY_SCHEMA_DIGEST_ARTIFACT",
    "TECHNICAL_FAILURE_ROW_SCHEMA_ARTIFACT",
    "TECHNICAL_FAILURE_ROW_SCHEMA_DIGEST_ARTIFACT",
    "TechnicalFailureEvalRow",
    "build_dry_run_plan",
    "build_eval_summary",
    "load_frozen_semantic_intent_questions",
    "run_live_evaluation",
    "score_gold_semantic_intent",
    "validate_gold10_gate_summary",
    "verify_hcx_semantic_intent_eval_schema_artifacts",
    "write_hcx_semantic_intent_eval_schema_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(_main())
