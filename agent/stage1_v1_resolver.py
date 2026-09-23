"""Generic Stage1 v1 resolver authority boundary.

The resolver is deliberately separated from both the HCX semantic interpreter
and the deterministic plan compiler.  A backend may reuse proven canonical
lookup code from the legacy resolver, but it must return one of the typed v1
authorities in this module.  ``PlanProposal``, route/task hints and
``ResolvedQueryPlan`` are not accepted at this boundary and therefore cannot be
reverse-read as v1 answer semantics or completeness.

Malformed backend output and binding failures raise
``Stage1V1ResolverTechnicalError``.  They are never converted into a normal
terminal/limitation outcome.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Annotated, Any, Literal, Mapping, Protocol, TypeAlias
import unicodedata

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)

from .deterministic_plan_compiler_v1 import AuthoritativeResolution
from .semantic_intent_v1 import SemanticIntent, semantic_intent_digest


RESOLUTION_DECISION_VERSION = "stage1-resolution-decision/1.1"
SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_resolution_decision_v1.schema.json")
SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_resolution_decision_v1.schema.sha256")

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
BuildId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
NonEmpty = Annotated[str, StringConstraints(min_length=1)]
Identifier = Annotated[str, StringConstraints(
    min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")]
ReasonCode = Annotated[str, StringConstraints(
    min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]{0,63}$")]
Versioned = Annotated[str, StringConstraints(
    min_length=3,
    pattern=r"^[A-Za-z0-9_.-]+/[0-9]+(?:\.[0-9]+)*$",
)]

ClarificationRole: TypeAlias = Literal[
    "entity", "target", "event", "time", "timepoint", "qualifier", "selection", "value_kind"
]
ClarificationResponseKind: TypeAlias = Literal[
    "select_one", "provide_value", "confirm_interpretation"
]
TerminalReasonCode: TypeAlias = Literal[
    "out_of_scope",
    "unsupported_request",
    "policy_refusal",
    "corp_not_in_universe",
    "future_forecast",
    "investment_advice",
    "prompt_injection_direct",
    "prompt_injection_role_impersonation",
    "external_tool_request",
    "external_url_request",
    "personal_data_request",
    "corpus_coverage_unavailable",
    "unsupported_temporal_scope",
    "unsupported_semantic_target",
    "causal_inference_beyond_scope",
    "pressure_resisted",
    "abusive_input",
    "off_topic_request",
]


class Stage1V1ResolverTechnicalError(RuntimeError):
    """A resolver transport/schema/binding failure outside normal outcomes."""



BackendResolutionStatus: TypeAlias = Literal[
    "resolved",
    "not_applicable",
    "coverage_unavailable",
    "capability_unmatched",
]


@dataclass(frozen=True, slots=True)
class BackendResolutionResult:
    """Internal backend disposition before public Stage1 authority binding.

    Legacy backends historically returned ``None`` for every kind of decline.
    The adapter keeps that local meaning as ``not_applicable`` while a composite
    can report the distinct aggregate ``capability_unmatched`` state.  A backend
    may claim ``coverage_unavailable`` only explicitly; the resolver never
    infers corpus absence from a bare ``None``.

    This type is internal to Stage1 v1.  Public QueryPlanHandoff v0.4 continues
    to receive only the existing resolved/clarification/terminal authorities.
    """

    status: BackendResolutionStatus
    authority: ResolutionAuthority | Mapping[str, Any] | None = None
    diagnostic_code: str | None = None

    def __post_init__(self) -> None:
        if (self.status == "resolved") != (self.authority is not None):
            raise ValueError(
                "resolved backend result만 authority를 가져야 합니다")
        if self.diagnostic_code is not None and re.fullmatch(
                r"[a-z][a-z0-9_]{0,63}", self.diagnostic_code) is None:
            raise ValueError("backend diagnostic_code 형식이 잘못되었습니다")

    @classmethod
    def resolved(
            cls, authority: ResolutionAuthority | Mapping[str, Any],
            ) -> "BackendResolutionResult":
        return cls(status="resolved", authority=authority)

    @classmethod
    def not_applicable(
            cls, *, diagnostic_code: str | None = None,
            ) -> "BackendResolutionResult":
        return cls(status="not_applicable", diagnostic_code=diagnostic_code)

    @classmethod
    def coverage_unavailable(
            cls, *, diagnostic_code: str | None = None,
            ) -> "BackendResolutionResult":
        return cls(status="coverage_unavailable", diagnostic_code=diagnostic_code)

    @classmethod
    def capability_unmatched(
            cls, *, diagnostic_code: str | None = None,
            ) -> "BackendResolutionResult":
        return cls(status="capability_unmatched", diagnostic_code=diagnostic_code)


def normalize_backend_resolution_result(
        value: ResolutionAuthority | Mapping[str, Any]
        | BackendResolutionResult | None,
        *, none_status: Literal["not_applicable", "capability_unmatched"] = (
            "capability_unmatched"),
        ) -> BackendResolutionResult:
    """Adapt legacy authority/``None`` returns into an explicit disposition."""

    if isinstance(value, BackendResolutionResult):
        return value
    if value is None:
        if none_status == "not_applicable":
            return BackendResolutionResult.not_applicable(
                diagnostic_code="backend_not_applicable")
        return BackendResolutionResult.capability_unmatched(
            diagnostic_code="resolver_capability_unmatched")
    return BackendResolutionResult.resolved(value)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", warnings=False)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_resolution_premise_proofs(
        source_intent: SemanticIntent,
        items: list[Mapping[str, Any]],
        *,
        overrides: Mapping[str, list[str]] | None = None,
        ) -> list[dict[str, Any]]:
    """Build source-bound proof inventory for every semantic premise.

    Backends used to omit ``premise_proofs`` entirely, even when the source
    intent carried a premise.  The decision boundary quite correctly rejected
    that authority, but the omission was in the adapter layer: premise IDs and
    source proof references must be emitted together with the resolved items.

    ``overrides`` is for typed multi-source relations whose premise is proved
    by a subset of nested evidence (for example the two monetary operands of a
    document comparison rather than its explanatory-text evidence).  It is
    keyed by the semantic premise ID, never by question ID or Gold literal.
    Other items use their field proofs; whole-target coordinates fall back to
    a stable canonical-coordinate proof.
    """
    if not source_intent.premises:
        return []
    by_item = {str(row["item_id"]): row for row in items}
    supplied = dict(overrides or {})
    result: list[dict[str, Any]] = []
    for index, premise in enumerate(source_intent.premises, start=1):
        premise_id = premise.premise_id
        refs = list(supplied.get(premise_id, ()))
        if not refs:
            for item_id in premise.applies_to_item_ids:
                item = by_item.get(item_id)
                if item is None:
                    raise ValueError(
                        f"premise {premise_id}가 resolved item을 참조하지 않습니다")
                field_refs = [
                    str(row["proof_ref"])
                    for row in item.get("field_proofs", ())
                    if row.get("proof_ref")
                ]
                refs.extend(field_refs or _coordinate_proof_refs(item))
        refs = list(dict.fromkeys(refs))
        if not refs:
            raise ValueError(f"premise {premise_id}에 source proof가 없습니다")
        result.append({"premise_id": premise_id, "proof_refs": refs})
    if [row["premise_id"] for row in result] != [
            premise.premise_id for premise in source_intent.premises]:
        raise ValueError("premise proof inventory가 source intent 순서와 다릅니다")
    return result


def _coordinate_proof_refs(item: Mapping[str, Any]) -> list[str]:
    """Return deterministic canonical proof refs for a whole-target item."""
    resolution = item.get("resolution")
    if not isinstance(resolution, Mapping):
        return []
    kind = resolution.get("kind")
    if kind == "financial":
        start = resolution.get("period_start") or "instant"
        return [
            "canonical:financial:"
            f"{resolution.get('corp_code')}:{start}:"
            f"{resolution.get('period_end')}:{resolution.get('scope')}:"
            f"{resolution.get('statement')}"
        ]
    if kind == "financial_comparison":
        refs = [
            str(row["proof_ref"])
            for row in resolution.get("operands", ())
            if row.get("proof_ref")
        ]
        if refs:
            return refs
    refs: list[str] = []
    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {"proof_ref", "selector_proof_ref"} \
                        and isinstance(child, str) and child:
                    refs.append(child)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(resolution)
    return list(dict.fromkeys(refs))


def _strict_model(model: type[BaseModel], value: Any) -> BaseModel:
    if isinstance(value, model):
        return model.model_validate_json(
            canonical_json(value.model_dump(mode="json", warnings=False)),
            strict=True,
        )
    if isinstance(value, Mapping):
        return model.model_validate_json(canonical_json(value), strict=True)
    raise TypeError(f"{model.__name__} instance 또는 mapping이 필요합니다")


def _ordered_local_ids(values: list[str], *, prefix: str, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label}에는 중복 ID가 있을 수 없습니다")
    positions: list[int] = []
    for value in values:
        match = re.fullmatch(rf"{re.escape(prefix)}-([1-9][0-9]*)", value)
        if match is None:
            raise ValueError(f"{label} ID 형식이 잘못되었습니다: {value}")
        positions.append(int(match.group(1)))
    if positions != sorted(positions):
        raise ValueError(f"{label}은 source 순서여야 합니다")


def _strict_intent(value: SemanticIntent | Mapping[str, Any]) -> SemanticIntent:
    return _strict_model(SemanticIntent, value)  # type: ignore[return-value]


def validate_semantic_intent_grounding(
        question: str,
        intent: SemanticIntent | Mapping[str, Any],
        ) -> SemanticIntent:
    """Strictly revalidate and re-ground every semantic surface.

    This closes callers that bypassed ``normalize_semantic_intent`` with
    ``model_copy(update=...)`` or hand-built normalized models.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question은 비어 있지 않은 문자열이어야 합니다")
    normalized_question = unicodedata.normalize("NFC", question)
    question_compact = re.sub(r"\s+", "", normalized_question)
    normalized_intent = _strict_intent(intent)

    def grounded(
            value: str, *, path: str, allow_closed_pair: bool = False,
            ) -> None:
        normalized = unicodedata.normalize("NFC", value)
        if not normalized or not normalized.strip() or normalized != value:
            raise ValueError(f"{path}는 NFC 비공백 표면이어야 합니다")
        if normalized in normalized_question:
            return
        # Output-only closed decomposition: 「A 전후 B」 requests both
        # observable values and can be represented as 「A 전 B」「A 후 B」.
        # The exact remaining tokens must still be present in the question.
        pair = re.fullmatch(
            r"(?P<prefix>.+?)\s+(?P<axis>전|후)\s+(?P<suffix>.+)",
            normalized.strip())
        if (allow_closed_pair and pair is not None
                and re.sub(r"\s+", "", (
                    f"{pair.group('prefix')} 전후 {pair.group('suffix')}"
                )) in question_compact):
            return
        if normalized not in normalized_question:
            raise ValueError(f"{path}가 원 질문에 결속되지 않았습니다")

    for index, entity in enumerate(normalized_intent.entities):
        grounded(entity.surface, path=f"entities[{index}].surface")
    for index, item in enumerate(normalized_intent.answer_items):
        prefix = f"answer_items[{index}]"
        grounded(item.target.surface, path=f"{prefix}.target.surface")
        for position, value in enumerate(item.target.qualifier_surfaces):
            grounded(value, path=f"{prefix}.target.qualifier_surfaces[{position}]")
        for position, value in enumerate(item.scope.target_period_expressions):
            grounded(value, path=f"{prefix}.scope.target_period_expressions[{position}]")
        if item.scope.as_of_expression is not None:
            grounded(item.scope.as_of_expression,
                     path=f"{prefix}.scope.as_of_expression")
        if item.scope.document_group_expression is not None:
            grounded(item.scope.document_group_expression,
                     path=f"{prefix}.scope.document_group_expression")
        for position, value in enumerate(item.scope.scope_qualifier_expressions):
            grounded(value, path=f"{prefix}.scope.scope_qualifier_expressions[{position}]")
        if item.selection is not None:
            grounded(item.selection.criterion_surface,
                     path=f"{prefix}.selection.criterion_surface")
        for position, value in enumerate(item.output.field_surfaces):
            grounded(
                value, path=f"{prefix}.output.field_surfaces[{position}]",
                allow_closed_pair=True)
    for index, premise in enumerate(normalized_intent.premises):
        grounded(premise.raw_text, path=f"premises[{index}].raw_text")
    for index, mention in enumerate(normalized_intent.unresolved_mentions):
        grounded(mention.raw_text,
                 path=f"unresolved_mentions[{index}].raw_text")
    return normalized_intent


class ClarificationOption(_StrictFrozenModel):
    value: NonEmpty
    label: NonEmpty
    reason: NonEmpty | None = None
    proof_refs: list[NonEmpty] = Field(
        default_factory=list, exclude_if=lambda refs: not refs)

    @model_validator(mode="after")
    def nonblank(self) -> "ClarificationOption":
        if (not self.value.strip() or not self.label.strip()
                or (self.reason is not None and not self.reason.strip())):
            raise ValueError("clarification option은 공백일 수 없습니다")
        if len(self.proof_refs) != len(set(self.proof_refs)):
            raise ValueError("clarification option proof_ref가 중복되었습니다")
        return self


class ClarificationSlot(_StrictFrozenModel):
    """One user-answerable semantic axis; no internal JSON patch path leaks."""

    slot_id: Annotated[str, StringConstraints(pattern=r"^slot-[1-9][0-9]*$")]
    role_hint: ClarificationRole
    reason_code: ReasonCode
    response_kind: ClarificationResponseKind
    prompt: NonEmpty
    applies_to_item_ids: list[Identifier] = Field(min_length=1)
    mention_ids: list[Identifier] = Field(default_factory=list)
    options: list[ClarificationOption] = Field(default_factory=list)

    @field_validator("options", mode="before")
    @classmethod
    def strict_options(cls, value: Any) -> list[ClarificationOption]:
        if not isinstance(value, list):
            raise TypeError("clarification options는 list여야 합니다")
        return [
            _strict_model(ClarificationOption, row)  # type: ignore[list-item]
            for row in value
        ]

    @model_validator(mode="after")
    def validate_slot(self) -> "ClarificationSlot":
        if not self.prompt.strip():
            raise ValueError("clarification prompt는 공백일 수 없습니다")
        _ordered_local_ids(
            list(self.applies_to_item_ids), prefix="item",
            label="clarification applies_to_item_ids")
        _ordered_local_ids(
            list(self.mention_ids), prefix="unresolved",
            label="clarification mention_ids")
        option_values = [row.value for row in self.options]
        option_labels = [row.label for row in self.options]
        if (len(option_values) != len(set(option_values))
                or len(option_labels) != len(set(option_labels))):
            raise ValueError("clarification option value/label은 고유해야 합니다")
        if self.response_kind == "provide_value":
            if self.options:
                raise ValueError("provide_value slot에는 제한 선택지를 넣지 않습니다")
        elif len(self.options) < 2:
            raise ValueError("select/confirm slot에는 선택지가 2개 이상 필요합니다")
        if (self.response_kind == "confirm_interpretation"
                and len(self.options) != 2):
            raise ValueError("confirm_interpretation에는 선택지 2개가 필요합니다")
        return self


class ClarificationAuthority(_StrictFrozenModel):
    kind: Literal["clarification"] = "clarification"
    slots: list[ClarificationSlot] = Field(min_length=1)

    @field_validator("slots", mode="before")
    @classmethod
    def strict_slots(cls, value: Any) -> list[ClarificationSlot]:
        if not isinstance(value, list):
            raise TypeError("clarification slots는 list여야 합니다")
        return [
            _strict_model(ClarificationSlot, row)  # type: ignore[list-item]
            for row in value
        ]

    @model_validator(mode="after")
    def deterministic_slots(self) -> "ClarificationAuthority":
        expected = [f"slot-{index}" for index in range(1, len(self.slots) + 1)]
        if [row.slot_id for row in self.slots] != expected:
            raise ValueError("clarification slot_id는 source 순서의 local ID여야 합니다")
        return self


class TerminalReasonBinding(_StrictFrozenModel):
    code: TerminalReasonCode
    scope: Literal["question", "items"]
    item_ids: list[Identifier] = Field(min_length=1)
    diagnostic_code: ReasonCode | None = Field(
        default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def ordered_items(self) -> "TerminalReasonBinding":
        _ordered_local_ids(
            list(self.item_ids), prefix="item", label="terminal item_ids")
        return self


class TerminalAuthority(_StrictFrozenModel):
    kind: Literal["terminal"] = "terminal"
    reasons: list[TerminalReasonBinding] = Field(min_length=1)

    @field_validator("reasons", mode="before")
    @classmethod
    def strict_reasons(cls, value: Any) -> list[TerminalReasonBinding]:
        if not isinstance(value, list):
            raise TypeError("terminal reasons는 list여야 합니다")
        return [
            _strict_model(TerminalReasonBinding, row)  # type: ignore[list-item]
            for row in value
        ]


class ResolvedAuthority(_StrictFrozenModel):
    kind: Literal["resolved"] = "resolved"
    resolution: AuthoritativeResolution

    @field_validator("resolution", mode="before")
    @classmethod
    def strict_resolution(cls, value: Any) -> AuthoritativeResolution:
        return _strict_model(AuthoritativeResolution, value)  # type: ignore[return-value]


ResolutionAuthority: TypeAlias = Annotated[
    ResolvedAuthority | ClarificationAuthority | TerminalAuthority,
    Field(discriminator="kind"),
]
_AUTHORITY_ADAPTER = TypeAdapter(ResolutionAuthority)


def _strict_authority(value: ResolutionAuthority | Mapping[str, Any]) -> ResolutionAuthority:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", warnings=False)
    elif isinstance(value, Mapping):
        value = dict(value)
    else:
        raise TypeError("resolver backend은 typed v1 authority를 반환해야 합니다")
    return _AUTHORITY_ADAPTER.validate_json(canonical_json(value), strict=True)


class Stage1ResolutionDecision(_StrictFrozenModel):
    """Digest-bound resolver result before deterministic plan compilation."""

    schema_version: Literal[RESOLUTION_DECISION_VERSION] = (
        RESOLUTION_DECISION_VERSION)
    question_id: Identifier
    question: NonEmpty
    question_sha256: Digest
    source_intent: SemanticIntent
    source_intent_digest: Digest
    canonical_build_id: BuildId
    resolver_version: Versioned
    authority: ResolutionAuthority
    decision_digest: Digest

    @field_validator("source_intent", mode="before")
    @classmethod
    def strict_source_intent(cls, value: Any) -> SemanticIntent:
        return _strict_model(SemanticIntent, value)  # type: ignore[return-value]

    @field_validator("authority", mode="before")
    @classmethod
    def strict_authority(cls, value: Any) -> ResolutionAuthority:
        return _strict_authority(value)

    @classmethod
    def compute_digest(cls, payload: Mapping[str, Any]) -> str:
        body = dict(payload)
        body.pop("decision_digest", None)
        return canonical_sha256(body)

    @classmethod
    def create(
            cls,
            *,
            question_id: str,
            question: str,
            source_intent: SemanticIntent | Mapping[str, Any],
            canonical_build_id: str,
            resolver_version: str,
            authority: ResolutionAuthority | Mapping[str, Any],
            ) -> "Stage1ResolutionDecision":
        intent = validate_semantic_intent_grounding(question, source_intent)
        normalized_authority = _strict_authority(authority)
        body: dict[str, Any] = {
            "schema_version": RESOLUTION_DECISION_VERSION,
            "question_id": question_id,
            "question": question,
            "question_sha256": sha256(question.encode("utf-8")).hexdigest(),
            "source_intent": intent,
            "source_intent_digest": semantic_intent_digest(intent),
            "canonical_build_id": canonical_build_id,
            "resolver_version": resolver_version,
            "authority": normalized_authority,
        }
        digest_body = cls.model_construct(
            **body, decision_digest="0" * 64).model_dump(
                mode="json", warnings=False)
        body["decision_digest"] = cls.compute_digest(digest_body)
        return cls.model_validate(body, strict=True)

    @model_validator(mode="after")
    def validate_decision(self) -> "Stage1ResolutionDecision":
        intent = validate_semantic_intent_grounding(self.question, self.source_intent)
        if self.question_sha256 != sha256(self.question.encode("utf-8")).hexdigest():
            raise ValueError("question_sha256가 embedded question과 다릅니다")
        if self.source_intent_digest != semantic_intent_digest(intent):
            raise ValueError("source_intent_digest가 embedded intent와 다릅니다")

        item_ids = [row.item_id for row in intent.answer_items]
        item_id_set = set(item_ids)
        mention_by_id = {
            row.mention_id: row for row in intent.unresolved_mentions}

        if isinstance(self.authority, ResolvedAuthority):
            resolution = self.authority.resolution
            if intent.unresolved_mentions:
                raise ValueError(
                    "unresolved mention이 남은 intent는 resolved authority가 될 수 없습니다")
            if (
                resolution.question_id != self.question_id
                or resolution.source_intent_digest != self.source_intent_digest
                or resolution.canonical_build_id != self.canonical_build_id
                or resolution.resolver_version != self.resolver_version
            ):
                raise ValueError("authoritative resolution identity/binding이 다릅니다")
            if [row.item_id for row in resolution.items] != item_ids:
                raise ValueError("resolution item inventory가 source intent와 다릅니다")
            for intent_item, resolved_item in zip(
                    intent.answer_items, resolution.items, strict=True):
                if resolved_item.target_surface != intent_item.target.surface:
                    raise ValueError("resolution target_surface이 intent와 다릅니다")
                if resolved_item.projection_mode != intent_item.output.projection_mode:
                    raise ValueError("resolution projection_mode가 intent와 다릅니다")
                expected_indexes = list(range(len(intent_item.output.field_surfaces)))
                if [row.source_field_index for row in resolved_item.field_proofs] != (
                        expected_indexes):
                    raise ValueError("resolution field proof coverage가 완전하지 않습니다")
                if [row.surface for row in resolved_item.field_proofs] != (
                        intent_item.output.field_surfaces):
                    raise ValueError("resolution field proof surface/order가 intent와 다릅니다")
            if [row.premise_id for row in resolution.premise_proofs] != [
                    row.premise_id for row in intent.premises]:
                raise ValueError("resolution premise proof inventory가 intent와 다릅니다")

        elif isinstance(self.authority, ClarificationAuthority):
            seen_mentions: list[str] = []
            for slot in self.authority.slots:
                if any(item_id not in item_id_set
                       for item_id in slot.applies_to_item_ids):
                    raise ValueError("clarification item ref가 intent에 없습니다")
                if any(mention_id not in mention_by_id
                       for mention_id in slot.mention_ids):
                    raise ValueError("clarification mention ref가 intent에 없습니다")
                if slot.mention_ids:
                    mentions = [mention_by_id[value] for value in slot.mention_ids]
                    if any(row.role_hint != slot.role_hint for row in mentions):
                        raise ValueError("clarification role이 unresolved mention과 다릅니다")
                    expected_items = [
                        item_id for item_id in item_ids
                        if any(item_id in row.applies_to_item_ids for row in mentions)
                    ]
                    if slot.applies_to_item_ids != expected_items:
                        raise ValueError(
                            "clarification item coverage가 mention coverage와 다릅니다")
                seen_mentions.extend(slot.mention_ids)
            if len(seen_mentions) != len(set(seen_mentions)):
                raise ValueError("unresolved mention은 한 slot에서만 소유해야 합니다")
            if seen_mentions != list(mention_by_id):
                raise ValueError(
                    "clarification은 unresolved mention 전부를 source 순서로 덮어야 합니다")

        else:
            seen_bindings: set[tuple[str, str, tuple[str, ...]]] = set()
            question_scoped = False
            covered: set[str] = set()
            for reason in self.authority.reasons:
                if any(item_id not in item_id_set for item_id in reason.item_ids):
                    raise ValueError("terminal item ref가 intent에 없습니다")
                token = (reason.code, reason.scope, tuple(reason.item_ids))
                if token in seen_bindings:
                    raise ValueError("terminal reason binding이 중복되었습니다")
                seen_bindings.add(token)
                covered.update(reason.item_ids)
                if reason.scope == "question":
                    question_scoped = True
                    if reason.item_ids != item_ids:
                        raise ValueError(
                            "question-scope terminal은 모든 item을 명시해야 합니다")
            if not question_scoped and covered != item_id_set:
                raise ValueError(
                    "item-scope terminal은 모든 answer item을 덮어야 합니다")

        if self.decision_digest != self.compute_digest(
                self.model_dump(mode="json", warnings=False)):
            raise ValueError("resolution decision digest가 일치하지 않습니다")
        return self

    @property
    def decision_kind(self) -> Literal[
            "resolved", "clarification", "terminal"]:
        return self.authority.kind


class Stage1ResolutionBackend(Protocol):
    """Pluggable canonical authority provider; never a legacy plan output."""

    def resolve(
            self,
            *,
            question_id: str,
            question: str,
            source_intent: SemanticIntent,
            ) -> ResolutionAuthority | Mapping[str, Any] \
                | BackendResolutionResult | None: ...


class Stage1V1Resolver:
    """Bind one backend authority to the exact v1 intent and build snapshot."""

    def __init__(
            self,
            backend: Stage1ResolutionBackend,
            *,
            canonical_build_id: str,
            resolver_version: str,
            ) -> None:
        if re.fullmatch(r"[0-9a-f]{32}", canonical_build_id) is None:
            raise ValueError("canonical_build_id 형식이 잘못되었습니다")
        if re.fullmatch(
                r"[A-Za-z0-9_.-]+/[0-9]+(?:\.[0-9]+)*$",
                resolver_version) is None:
            raise ValueError("resolver_version 형식이 잘못되었습니다")
        self.backend = backend
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version

    def resolve(
            self,
            *,
            question_id: str,
            question: str,
            source_intent: SemanticIntent | Mapping[str, Any],
            ) -> Stage1ResolutionDecision:
        try:
            intent = validate_semantic_intent_grounding(question, source_intent)
            # A canonical-backed backend may recover one omitted company from
            # the exact question before authority binding.  Re-ground the
            # derived intent again so the hook cannot introduce an ungrounded
            # surface or a malformed cross-reference.
            reground = getattr(self.backend, "reground_source_intent", None)
            if callable(reground):
                intent = validate_semantic_intent_grounding(
                    question, reground(question, intent))
            backend_result = normalize_backend_resolution_result(
                self.backend.resolve(
                question_id=question_id,
                question=question,
                source_intent=intent,
            ))
            if backend_result.status == "resolved":
                authority = backend_result.authority
            elif backend_result.status == "coverage_unavailable":
                authority = TerminalAuthority(reasons=[TerminalReasonBinding(
                    code="corpus_coverage_unavailable", scope="question",
                    item_ids=[row.item_id for row in intent.answer_items],
                    diagnostic_code=(backend_result.diagnostic_code
                                     or "resolver_coverage_unavailable"))])
            elif backend_result.status in {"not_applicable", "capability_unmatched"}:
                # A capability miss is not proof that the corpus lacks data.
                # Keep the public v0.4 status class (unsupported_request), but
                # expose the truthful existing terminal reason instead of the
                # former blanket corpus_coverage_unavailable assertion.
                authority = TerminalAuthority(reasons=[TerminalReasonBinding(
                    code="unsupported_semantic_target", scope="question",
                    item_ids=[row.item_id for row in intent.answer_items],
                    diagnostic_code=(backend_result.diagnostic_code
                                     or "resolver_capability_unmatched"))])
            else:  # pragma: no cover - closed internal status guard
                raise TypeError("unknown backend resolution status")
            return Stage1ResolutionDecision.create(
                question_id=question_id,
                question=question,
                source_intent=intent,
                canonical_build_id=self.canonical_build_id,
                resolver_version=self.resolver_version,
                authority=authority,
            )
        except Stage1V1ResolverTechnicalError:
            raise
        except Exception as exc:
            raise Stage1V1ResolverTechnicalError(
                "Stage1 v1 resolver authority 생성/결속에 실패했습니다") from exc


def verify_resolution_decision_digest(
        value: Stage1ResolutionDecision | Mapping[str, Any],
        ) -> str:
    validated = _strict_model(Stage1ResolutionDecision, value)
    return validated.decision_digest  # type: ignore[attr-defined]


def load_resolution_decision_json(
        payload: str | bytes | bytearray,
        ) -> Stage1ResolutionDecision:
    return Stage1ResolutionDecision.model_validate_json(payload, strict=True)


def _schema_artifact_bytes() -> tuple[bytes, bytes, str]:
    schema = canonical_json(
        Stage1ResolutionDecision.model_json_schema(mode="validation")).encode(
            "utf-8")
    digest = sha256(schema).hexdigest()
    sidecar = f"{digest}  {SCHEMA_ARTIFACT.name}\n".encode("ascii")
    return schema, sidecar, digest


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_resolution_decision_schema_artifacts() -> str:
    schema, sidecar, digest = _schema_artifact_bytes()
    _atomic_write(SCHEMA_ARTIFACT, schema)
    _atomic_write(SCHEMA_DIGEST_ARTIFACT, sidecar)
    return digest


def verify_resolution_decision_schema_artifacts() -> str:
    schema, sidecar, digest = _schema_artifact_bytes()
    try:
        if SCHEMA_ARTIFACT.read_bytes() != schema:
            raise RuntimeError("resolution decision schema artifact drift")
        if SCHEMA_DIGEST_ARTIFACT.read_bytes() != sidecar:
            raise RuntimeError("resolution decision schema digest drift")
    except OSError as exc:
        raise RuntimeError("resolution decision schema artifact missing") from exc
    return digest


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    digest = (
        write_resolution_decision_schema_artifacts()
        if args.write else verify_resolution_decision_schema_artifacts())
    print(f"PASS: {RESOLUTION_DECISION_VERSION} schema_sha256={digest}")
    return 0


__all__ = [
    "BackendResolutionResult",
    "BackendResolutionStatus",
    "ClarificationAuthority",
    "ClarificationOption",
    "ClarificationSlot",
    "RESOLUTION_DECISION_VERSION",
    "ResolutionAuthority",
    "ResolvedAuthority",
    "SCHEMA_ARTIFACT",
    "SCHEMA_DIGEST_ARTIFACT",
    "Stage1ResolutionBackend",
    "Stage1ResolutionDecision",
    "Stage1V1Resolver",
    "Stage1V1ResolverTechnicalError",
    "build_resolution_premise_proofs",
    "TerminalAuthority",
    "TerminalReasonBinding",
    "canonical_json",
    "canonical_sha256",
    "load_resolution_decision_json",
    "normalize_backend_resolution_result",
    "validate_semantic_intent_grounding",
    "verify_resolution_decision_digest",
    "verify_resolution_decision_schema_artifacts",
    "write_resolution_decision_schema_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(_main())
