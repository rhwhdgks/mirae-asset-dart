"""Stage 1 semantic intent v1 wire and normalization contracts.

The provider model in this module deliberately carries only question meaning:
array positions, closed semantic hints, and surfaces copied from the question.
The normalizer is the owner of local identifiers and of all cross-reference
validation.  It is intentionally independent from the legacy planner modules.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import unicodedata
from typing import Any, Literal, Mapping, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .hcx_schema import (
    compile_hcx_schema,
    safe_validation_issue_codes,
    safe_validation_issue_paths,
)


HCX_SEMANTIC_INTENT_WIRE_V1 = "hcx-semantic-intent-wire/1.1"
STAGE1_SEMANTIC_INTENT_V1 = "stage1-semantic-intent/1.1"

SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "hcx_semantic_intent_wire_v1.schema.json")
SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "hcx_semantic_intent_wire_v1.schema.sha256")
NORMALIZED_SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_semantic_intent_v1.schema.json")
NORMALIZED_SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_semantic_intent_v1.schema.sha256")


SemanticOperation: TypeAlias = Literal["retrieve", "compare"]
TargetKind: TypeAlias = Literal[
    "metric", "attribute", "event", "document", "topic", "entity"
]
Presentation: TypeAlias = Literal["auto", "prose", "table", "list"]
ProjectionMode: TypeAlias = Literal["named_fields", "whole_target"]
OutputShape: TypeAlias = Literal[
    "scalar", "record", "record_list", "comparison", "timeline",
    "narrative", "verdict",
]
SelectionMode: TypeAlias = Literal[
    "latest", "earliest", "maximum", "minimum", "top_k", "bottom_k"
]
WireSelectionMode: TypeAlias = Literal[
    "none", "latest", "earliest", "maximum", "minimum", "top_k", "bottom_k"
]
EntityKindHint: TypeAlias = Literal[
    "company", "counterparty", "document", "event", "other"
]
PremiseKind: TypeAlias = Literal[
    "numeric", "state", "comparison", "existence", "causal"
]
UnresolvedRole: TypeAlias = Literal[
    "entity", "target", "time", "qualifier", "selection"
]


class SemanticIntentError(ValueError):
    """Base error for malformed semantic intent input."""


class SemanticIntentNormalizationError(SemanticIntentError):
    """A provider wire cannot be safely normalized against its question."""

    def __init__(
            self, message: str, *,
            diagnostic_codes: tuple[str, ...] = ("normalization_rejected",),
            diagnostic_paths: tuple[str, ...] = (),
            ) -> None:
        codes = tuple(sorted(set(diagnostic_codes)))
        paths = tuple(sorted(set(diagnostic_paths)))
        if (not codes or any(not isinstance(code, str) or not re.fullmatch(
                r"[a-z][a-z0-9_]{0,63}", code) for code in codes)):
            raise ValueError("normalization diagnostic code 형식이 잘못되었습니다")
        if any(not isinstance(path, str) or not re.fullmatch(
                r"[a-z][a-z0-9_]*(?:\[[0-9]+\])?"
                r"(?:\.[a-z][a-z0-9_]*(?:\[[0-9]+\])?)*",
                path,
                ) for path in paths):
            raise ValueError("normalization diagnostic path 형식이 잘못되었습니다")
        self.diagnostic_codes = codes
        self.diagnostic_paths = paths
        super().__init__(message)


class _StrictFrozenModel(BaseModel):
    """Strict/frozen boundary inherited by every nested and top-level model."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _unique(values: list[Any], *, name: str) -> None:
    encoded = [json.dumps(
        item.model_dump(mode="json") if isinstance(item, BaseModel) else item,
        ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")) for item in values]
    if len(encoded) != len(set(encoded)):
        raise ValueError(f"{name}에는 중복 값이 있을 수 없습니다")


def _nonempty(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}은 비어 있을 수 없습니다")
    return value


def _nonempty_surfaces(values: list[str], *, name: str) -> list[str]:
    for value in values:
        _nonempty(value, name=name)
    _unique(values, name=name)
    return values


def _nonempty_strings(values: list[str], *, name: str) -> list[str]:
    for value in values:
        _nonempty(value, name=name)
    return values


def _valid_indices(values: list[int], *, name: str) -> list[int]:
    for value in values:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name}에는 0 이상 정수 인덱스만 허용됩니다")
    _unique(values, name=name)
    return values


class WireEntityMention(_StrictFrozenModel):
    kind_hint: EntityKindHint
    surface: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_surface(self) -> "WireEntityMention":
        _nonempty(self.surface, name="entity surface")
        return self


class WireTarget(_StrictFrozenModel):
    kind: TargetKind
    surface: str = Field(min_length=1)
    entity_indexes: list[int] = Field(default_factory=list)
    qualifier_surfaces: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_target(self) -> "WireTarget":
        _nonempty(self.surface, name="target surface")
        _valid_indices(self.entity_indexes, name="target entity_indexes")
        _nonempty_surfaces(self.qualifier_surfaces,
                           name="target qualifier_surfaces")
        return self


class WireScope(_StrictFrozenModel):
    target_period_expressions: list[str] = Field(default_factory=list)
    as_of_expression: str = ""
    document_group_expression: str = ""
    scope_qualifier_expressions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scope(self) -> "WireScope":
        _nonempty_surfaces(
            self.target_period_expressions,
            name="scope target_period_expressions",
        )
        _nonempty_surfaces(
            self.scope_qualifier_expressions,
            name="scope scope_qualifier_expressions",
        )
        for name, value in (
                ("as_of_expression", self.as_of_expression),
                ("document_group_expression", self.document_group_expression)):
            if not isinstance(value, str):
                raise ValueError(f"scope {name}은 문자열이어야 합니다")
            if value and not value.strip():
                raise ValueError(f"scope {name}은 공백만 될 수 없습니다")
        return self


class WireSelection(_StrictFrozenModel):
    """HCX-safe selection shape; ``none`` is the provider null sentinel."""

    # Pydantic's cross-field validator is authoritative locally, but those
    # relations are otherwise absent from the HCX JSON Schema.  Keep every
    # sentinel field required and project the closed mode/k relations into
    # provider-side ``anyOf`` branches so HCX cannot emit a locally impossible
    # combination and only discover that after the paid response returns.
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        json_schema_extra={
            "anyOf": [
                {
                    "properties": {
                        "mode": {"const": "none"},
                        "criterion_surface": {"const": ""},
                        "k": {"const": 0},
                    },
                    "required": ["mode", "criterion_surface", "k"],
                },
                {
                    "properties": {
                        "mode": {
                            "enum": [
                                "latest", "earliest", "maximum", "minimum",
                            ],
                        },
                        "criterion_surface": {"type": "string"},
                        "k": {"const": 0},
                    },
                    "required": ["mode", "criterion_surface", "k"],
                },
                {
                    "properties": {
                        "mode": {"enum": ["top_k", "bottom_k"]},
                        "criterion_surface": {"type": "string"},
                        "k": {"type": "integer", "minimum": 1},
                    },
                    "required": ["mode", "criterion_surface", "k"],
                },
            ],
        },
    )

    mode: WireSelectionMode
    criterion_surface: str
    k: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_selection(self) -> "WireSelection":
        if self.criterion_surface and not self.criterion_surface.strip():
            raise ValueError("selection criterion_surface은 공백만 될 수 없습니다")
        if self.mode == "none":
            if self.criterion_surface or self.k != 0:
                raise ValueError("none selection은 빈 표면과 k=0이어야 합니다")
        elif self.mode in {"latest", "earliest"}:
            if self.k != 0:
                raise ValueError("latest/earliest selection은 k=0이어야 합니다")
        elif self.mode in {"maximum", "minimum"}:
            if self.k != 0:
                raise ValueError("maximum/minimum selection은 k=0이어야 합니다")
        elif self.mode in {"top_k", "bottom_k"}:
            if self.k < 1:
                raise ValueError("top_k/bottom_k selection은 양의 k가 필요합니다")
        return self


class WireOutputRequest(_StrictFrozenModel):
    # ``projection_mode`` and ``field_surfaces`` form one closed relation.
    # The explicit branches mirror ``validate_output`` in the provider schema;
    # without them HCX sees an unconstrained optional array and can return
    # ``named_fields`` with no requested field.
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        json_schema_extra={
            "anyOf": [
                {
                    "properties": {
                        "shape": {
                            "enum": [
                                "scalar", "record", "record_list",
                                "comparison", "timeline", "narrative",
                                "verdict",
                            ],
                        },
                        "projection_mode": {"const": "named_fields"},
                        "field_surfaces": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                        "presentation": {
                            "enum": ["auto", "prose", "table", "list"],
                        },
                    },
                    "required": [
                        "shape", "projection_mode", "field_surfaces",
                        "presentation",
                    ],
                },
                {
                    "properties": {
                        "shape": {
                            "enum": [
                                "scalar", "record", "record_list",
                                "comparison", "timeline", "narrative",
                                "verdict",
                            ],
                        },
                        "projection_mode": {"const": "whole_target"},
                        "field_surfaces": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 0,
                        },
                        "presentation": {
                            "enum": ["auto", "prose", "table", "list"],
                        },
                    },
                    "required": [
                        "shape", "projection_mode", "field_surfaces",
                        "presentation",
                    ],
                },
            ],
        },
    )

    shape: OutputShape
    projection_mode: ProjectionMode
    field_surfaces: list[str]
    presentation: Presentation

    @model_validator(mode="after")
    def validate_output(self) -> "WireOutputRequest":
        _nonempty_surfaces(self.field_surfaces, name="output field_surfaces")
        if self.projection_mode == "named_fields" and not self.field_surfaces:
            raise ValueError("named_fields output에는 사용자 field surface가 필요합니다")
        if self.projection_mode == "whole_target" and self.field_surfaces:
            raise ValueError("whole_target output에는 field surface가 없어야 합니다")
        return self


class WireAnswerItem(_StrictFrozenModel):
    """One user promise; membership is declared by answer groups."""

    target: WireTarget
    operation: SemanticOperation
    scope: WireScope
    selection: WireSelection
    output: WireOutputRequest


class WireAnswerGroup(_StrictFrozenModel):
    item_indexes: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_group(self) -> "WireAnswerGroup":
        _valid_indices(self.item_indexes, name="group item_indexes")
        if tuple(self.item_indexes) != tuple(sorted(self.item_indexes)):
            raise ValueError("group item_indexes는 질문 순서여야 합니다")
        return self


class WirePremise(_StrictFrozenModel):
    kind: PremiseKind
    raw_text: str = Field(min_length=1)
    applies_to_item_indexes: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_premise(self) -> "WirePremise":
        _nonempty(self.raw_text, name="premise raw_text")
        _valid_indices(self.applies_to_item_indexes,
                       name="premise applies_to_item_indexes")
        return self


class WireUnresolvedMention(_StrictFrozenModel):
    raw_text: str = Field(min_length=1)
    role_hint: UnresolvedRole
    applies_to_item_indexes: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unresolved(self) -> "WireUnresolvedMention":
        _nonempty(self.raw_text, name="unresolved raw_text")
        _valid_indices(self.applies_to_item_indexes,
                       name="unresolved applies_to_item_indexes")
        return self


class HcxSemanticIntentWire(_StrictFrozenModel):
    """Question-semantic HCX input/output wire.

    Optional normalized values are represented here by explicit sentinels:
    empty strings and ``selection.mode=none``.
    """

    schema_version: Literal[HCX_SEMANTIC_INTENT_WIRE_V1]
    entities: list[WireEntityMention] = Field(default_factory=list)
    answer_items: list[WireAnswerItem] = Field(min_length=1)
    answer_groups: list[WireAnswerGroup] = Field(default_factory=list)
    premises: list[WirePremise] = Field(default_factory=list)
    unresolved_mentions: list[WireUnresolvedMention] = Field(default_factory=list)
    presentation: Presentation

    @model_validator(mode="after")
    def validate_wire(self) -> "HcxSemanticIntentWire":
        _unique(self.entities, name="entities")
        return self


class EntityMention(_StrictFrozenModel):
    entity_id: str = Field(min_length=1)
    kind_hint: EntityKindHint
    surface: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entity(self) -> "EntityMention":
        _nonempty(self.entity_id, name="entity_id")
        _nonempty(self.surface, name="entity surface")
        return self


class Target(_StrictFrozenModel):
    kind: TargetKind
    surface: str = Field(min_length=1)
    entity_refs: list[str] = Field(default_factory=list)
    qualifier_surfaces: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_target(self) -> "Target":
        _nonempty(self.surface, name="target surface")
        _nonempty_surfaces(self.qualifier_surfaces,
                           name="target qualifier_surfaces")
        _nonempty_strings(self.entity_refs, name="target entity_refs")
        _unique(self.entity_refs, name="target entity_refs")
        return self


class Scope(_StrictFrozenModel):
    target_period_expressions: list[str] = Field(default_factory=list)
    as_of_expression: str | None = None
    document_group_expression: str | None = None
    scope_qualifier_expressions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scope(self) -> "Scope":
        _nonempty_surfaces(
            self.target_period_expressions,
            name="scope target_period_expressions",
        )
        _nonempty_surfaces(
            self.scope_qualifier_expressions,
            name="scope scope_qualifier_expressions",
        )
        for name, value in (
                ("as_of_expression", self.as_of_expression),
                ("document_group_expression", self.document_group_expression)):
            if value is not None:
                _nonempty(value, name=f"scope {name}")
        return self


class Selection(_StrictFrozenModel):
    mode: SelectionMode
    criterion_surface: str
    k: int | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> "Selection":
        _nonempty(self.criterion_surface, name="selection criterion_surface")
        if self.mode in {"top_k", "bottom_k"}:
            if self.k is None or self.k < 1:
                raise ValueError("top_k/bottom_k selection은 양의 k가 필요합니다")
        elif self.k is not None:
            raise ValueError("latest/earliest/maximum/minimum selection은 k가 없습니다")
        return self


class OutputRequest(_StrictFrozenModel):
    shape: OutputShape
    projection_mode: ProjectionMode
    field_surfaces: list[str] = Field(default_factory=list)
    presentation: Presentation

    @model_validator(mode="after")
    def validate_output(self) -> "OutputRequest":
        _nonempty_surfaces(self.field_surfaces, name="output field_surfaces")
        if self.projection_mode == "named_fields" and not self.field_surfaces:
            raise ValueError("named_fields output에는 사용자 field surface가 필요합니다")
        if self.projection_mode == "whole_target" and self.field_surfaces:
            raise ValueError("whole_target output에는 field surface가 없어야 합니다")
        return self


class AnswerItem(_StrictFrozenModel):
    item_id: str = Field(min_length=1)
    target: Target
    operation: SemanticOperation
    scope: Scope
    selection: Selection | None = None
    output: OutputRequest

    @model_validator(mode="after")
    def validate_item(self) -> "AnswerItem":
        _nonempty(self.item_id, name="item_id")
        return self


class AnswerGroup(_StrictFrozenModel):
    group_id: str = Field(min_length=1)
    item_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_group(self) -> "AnswerGroup":
        _nonempty(self.group_id, name="group group_id")
        _nonempty_strings(self.item_ids, name="group item_ids")
        _unique(self.item_ids, name="group item_ids")
        positions: list[int] = []
        for item_id in self.item_ids:
            prefix, separator, suffix = item_id.rpartition("-")
            if separator and prefix == "item" and suffix.isdigit():
                positions.append(int(suffix))
        if len(positions) == len(self.item_ids) and positions != sorted(positions):
            raise ValueError("group item_ids는 질문 순서여야 합니다")
        return self


class Premise(_StrictFrozenModel):
    premise_id: str = Field(min_length=1)
    kind: PremiseKind
    raw_text: str = Field(min_length=1)
    applies_to_item_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_premise(self) -> "Premise":
        _nonempty(self.premise_id, name="premise_id")
        _nonempty(self.raw_text, name="premise raw_text")
        _nonempty_strings(self.applies_to_item_ids,
                          name="premise applies_to_item_ids")
        _unique(self.applies_to_item_ids,
                name="premise applies_to_item_ids")
        positions = []
        for item_id in self.applies_to_item_ids:
            prefix, separator, suffix = item_id.rpartition("-")
            if separator and prefix == "item" and suffix.isdigit():
                positions.append(int(suffix))
        if len(positions) == len(self.applies_to_item_ids) and positions != sorted(positions):
            raise ValueError("premise item 참조는 질문 순서여야 합니다")
        return self


class UnresolvedMention(_StrictFrozenModel):
    mention_id: str = Field(min_length=1)
    raw_text: str = Field(min_length=1)
    role_hint: UnresolvedRole
    applies_to_item_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unresolved(self) -> "UnresolvedMention":
        _nonempty(self.mention_id, name="mention_id")
        _nonempty(self.raw_text, name="unresolved raw_text")
        _nonempty_strings(self.applies_to_item_ids,
                          name="unresolved applies_to_item_ids")
        _unique(self.applies_to_item_ids,
                name="unresolved applies_to_item_ids")
        return self


class SemanticIntent(_StrictFrozenModel):
    """Normalized v1 semantic intent with deterministic local identifiers."""

    schema_version: Literal[STAGE1_SEMANTIC_INTENT_V1]
    entities: list[EntityMention] = Field(default_factory=list)
    answer_items: list[AnswerItem] = Field(min_length=1)
    answer_groups: list[AnswerGroup] = Field(default_factory=list)
    premises: list[Premise] = Field(default_factory=list)
    unresolved_mentions: list[UnresolvedMention] = Field(default_factory=list)
    presentation: Presentation

    @model_validator(mode="after")
    def validate_intent(self) -> "SemanticIntent":
        expected_entities = [f"entity-{i}" for i in range(1, len(self.entities) + 1)]
        expected_items = [f"item-{i}" for i in range(1, len(self.answer_items) + 1)]
        expected_groups = [f"group-{i}" for i in range(1, len(self.answer_groups) + 1)]
        expected_premises = [f"premise-{i}" for i in range(1, len(self.premises) + 1)]
        expected_mentions = [
            f"unresolved-{i}" for i in range(1, len(self.unresolved_mentions) + 1)
        ]
        if [row.entity_id for row in self.entities] != expected_entities:
            raise ValueError("entity_id는 입력 순서의 결정론적 local ID여야 합니다")
        if [row.item_id for row in self.answer_items] != expected_items:
            raise ValueError("item_id는 입력 순서의 결정론적 local ID여야 합니다")
        if [row.group_id for row in self.answer_groups] != expected_groups:
            raise ValueError("group_id는 입력 순서의 결정론적 local ID여야 합니다")
        if [row.premise_id for row in self.premises] != expected_premises:
            raise ValueError("premise_id는 입력 순서의 결정론적 local ID여야 합니다")
        if [row.mention_id for row in self.unresolved_mentions] != expected_mentions:
            raise ValueError("mention_id는 입력 순서의 결정론적 local ID여야 합니다")

        item_ids = set(expected_items)
        entity_ids = set(expected_entities)
        memberships: dict[str, str] = {}
        for group in self.answer_groups:
            if group.group_id in memberships.values():
                raise ValueError("group_id는 중복될 수 없습니다")
            for item_id in group.item_ids:
                if item_id not in item_ids:
                    raise ValueError("group item_id가 answer_items를 가리키지 않습니다")
                if item_id in memberships:
                    raise ValueError("item은 최대 한 group에만 속할 수 있습니다")
                memberships[item_id] = group.group_id

        for item in self.answer_items:
            for ref in item.target.entity_refs:
                if ref not in entity_ids:
                    raise ValueError("target entity_ref가 유효하지 않습니다")
        for premise in self.premises:
            if any(ref not in item_ids for ref in premise.applies_to_item_ids):
                raise ValueError("premise item 참조가 유효하지 않습니다")
        for mention in self.unresolved_mentions:
            if any(ref not in item_ids for ref in mention.applies_to_item_ids):
                raise ValueError("unresolved item 참조가 유효하지 않습니다")
        return self


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _ground(value: str, question: str, *, path: str,
            optional: bool = False) -> str | None:
    normalized = _nfc(value)
    if not normalized:
        if optional:
            return None
        raise SemanticIntentNormalizationError(
            f"{path}: 표면이 비어 있습니다",
            diagnostic_codes=("surface_empty",),
            diagnostic_paths=(path,),
        )
    if not normalized.strip():
        raise SemanticIntentNormalizationError(
            f"{path}: 공백 표면은 허용되지 않습니다",
            diagnostic_codes=("surface_blank",),
            diagnostic_paths=(path,),
        )
    if normalized not in question:
        raise SemanticIntentNormalizationError(
            f"{path}: 원 질문에 결속되지 않은 표면입니다",
            diagnostic_codes=("surface_not_grounded",),
            diagnostic_paths=(path,),
        )
    return normalized


def _ground_list(values: list[str], question: str, *, path: str) -> list[str]:
    grounded: list[str] = []
    for index, value in enumerate(values):
        item = _ground(value, question, path=f"{path}[{index}]")
        assert item is not None
        grounded.append(item)
    _unique(grounded, name=path)
    return grounded


def _wire(value: HcxSemanticIntentWire | Mapping[str, Any]) -> HcxSemanticIntentWire:
    if isinstance(value, HcxSemanticIntentWire):
        try:
            # model_copy(update=...) does not run validators.  Re-enter the
            # ordinary strict boundary even when the caller supplied a model.
            return HcxSemanticIntentWire.model_validate(
                value.model_dump(mode="python", warnings=False), strict=True)
        except ValidationError as exc:
            raise SemanticIntentNormalizationError(
                "provider wire strict validation에 실패했습니다",
                diagnostic_codes=safe_validation_issue_codes(exc),
                diagnostic_paths=safe_validation_issue_paths(exc),
            ) from exc
    if isinstance(value, Mapping):
        try:
            return HcxSemanticIntentWire.model_validate(value, strict=True)
        except ValidationError as exc:
            raise SemanticIntentNormalizationError(
                "provider wire strict validation에 실패했습니다",
                diagnostic_codes=safe_validation_issue_codes(exc),
                diagnostic_paths=safe_validation_issue_paths(exc),
            ) from exc
    raise TypeError("provider wire는 HcxSemanticIntentWire 또는 mapping이어야 합니다")


def normalize_semantic_intent(
        question: str,
        provider: HcxSemanticIntentWire | Mapping[str, Any],
        ) -> SemanticIntent:
    """Normalize a parsed provider wire against the exact original question.

    NFC is applied to both question and every non-empty semantic surface.  No
    aliasing, canonical expansion, trimming, or fuzzy matching is performed.
    """

    if not isinstance(question, str) or not question.strip():
        raise SemanticIntentNormalizationError(
            "원 질문은 비어 있을 수 없습니다",
            diagnostic_codes=("question_invalid",),
        )
    question_nfc = _nfc(question)
    wire = _wire(provider)
    if wire.schema_version != HCX_SEMANTIC_INTENT_WIRE_V1:
        raise SemanticIntentNormalizationError(
            "provider schema_version이 정확하지 않습니다",
            diagnostic_codes=("schema_version_mismatch",),
            diagnostic_paths=("schema_version",),
        )

    entities: list[EntityMention] = []
    for index, entity in enumerate(wire.entities, start=1):
        surface = _ground(entity.surface, question_nfc,
                          path=f"entities[{index - 1}].surface")
        assert surface is not None
        entities.append(EntityMention(
            entity_id=f"entity-{index}", kind_hint=entity.kind_hint,
            surface=surface))

    entity_count = len(entities)
    items: list[AnswerItem] = []
    memberships: set[int] = set()
    for group_index, group in enumerate(wire.answer_groups):
        for ref_index, item_index in enumerate(group.item_indexes):
            if item_index >= len(wire.answer_items):
                raise SemanticIntentNormalizationError(
                    "group item index가 answer_items 범위를 벗어났습니다",
                    diagnostic_codes=("item_reference_invalid",),
                    diagnostic_paths=(
                        f"answer_groups[{group_index}].item_indexes[{ref_index}]",),
                )
            if item_index in memberships:
                raise SemanticIntentNormalizationError(
                    "item은 최대 한 group에만 속할 수 있습니다",
                    diagnostic_codes=("group_membership_duplicate",),
                    diagnostic_paths=(
                        f"answer_groups[{group_index}].item_indexes[{ref_index}]",),
                )
            memberships.add(item_index)

    for index, item in enumerate(wire.answer_items):
        target_entities: list[str] = []
        for ref_index, entity_index in enumerate(item.target.entity_indexes):
            if entity_index >= entity_count:
                raise SemanticIntentNormalizationError(
                    "target entity index가 entities 범위를 벗어났습니다",
                    diagnostic_codes=("entity_reference_invalid",),
                    diagnostic_paths=(
                        f"answer_items[{index}].target.entity_indexes[{ref_index}]",),
                )
            target_entities.append(f"entity-{entity_index + 1}")
        target_surface = _ground(
            item.target.surface, question_nfc,
            path=f"answer_items[{index}].target.surface")
        assert target_surface is not None
        target_qualifiers = _ground_list(
            item.target.qualifier_surfaces, question_nfc,
            path=f"answer_items[{index}].target.qualifier_surfaces")

        scope = item.scope
        periods = _ground_list(
            scope.target_period_expressions, question_nfc,
            path=f"answer_items[{index}].scope.target_period_expressions")
        as_of = _ground(
            scope.as_of_expression, question_nfc,
            path=f"answer_items[{index}].scope.as_of_expression",
            optional=True)
        document_group = _ground(
            scope.document_group_expression, question_nfc,
            path=f"answer_items[{index}].scope.document_group_expression",
            optional=True)
        scope_qualifiers = _ground_list(
            scope.scope_qualifier_expressions, question_nfc,
            path=f"answer_items[{index}].scope.scope_qualifier_expressions")

        selection: Selection | None = None
        if item.selection.mode != "none":
            criterion = _ground(
                item.selection.criterion_surface, question_nfc,
                path=f"answer_items[{index}].selection.criterion_surface")
            assert criterion is not None
            selection = Selection(
                mode=item.selection.mode, criterion_surface=criterion,
                k=item.selection.k or None)

        field_surfaces = _ground_list(
            item.output.field_surfaces, question_nfc,
            path=f"answer_items[{index}].output.field_surfaces")
        items.append(AnswerItem(
            item_id=f"item-{index + 1}",
            target=Target(
                kind=item.target.kind, surface=target_surface,
                entity_refs=target_entities,
                qualifier_surfaces=target_qualifiers,
            ),
            operation=item.operation,
            scope=Scope(
                target_period_expressions=periods,
                as_of_expression=as_of,
                document_group_expression=document_group,
                scope_qualifier_expressions=scope_qualifiers,
            ),
            selection=selection,
            output=OutputRequest(
                shape=item.output.shape,
                projection_mode=item.output.projection_mode,
                field_surfaces=field_surfaces,
                presentation=item.output.presentation,
            ),
        ))

    groups: list[AnswerGroup] = []
    for index, group in enumerate(wire.answer_groups, start=1):
        groups.append(AnswerGroup(
            group_id=f"group-{index}",
            item_ids=[f"item-{item_index + 1}" for item_index in group.item_indexes],
        ))

    premises: list[Premise] = []
    for index, premise in enumerate(wire.premises, start=1):
        raw_text = _ground(premise.raw_text, question_nfc,
                           path=f"premises[{index - 1}].raw_text")
        assert raw_text is not None
        for ref_index, item_index in enumerate(premise.applies_to_item_indexes):
            if item_index >= len(items):
                raise SemanticIntentNormalizationError(
                    "premise item index가 answer_items 범위를 벗어났습니다",
                    diagnostic_codes=("item_reference_invalid",),
                    diagnostic_paths=(
                        f"premises[{index - 1}].applies_to_item_indexes[{ref_index}]",),
                )
        premises.append(Premise(
            premise_id=f"premise-{index}", kind=premise.kind,
            raw_text=raw_text,
            applies_to_item_ids=[
                f"item-{item_index + 1}"
                for item_index in premise.applies_to_item_indexes
            ],
        ))

    unresolved: list[UnresolvedMention] = []
    for index, mention in enumerate(wire.unresolved_mentions, start=1):
        raw_text = _ground(mention.raw_text, question_nfc,
                           path=f"unresolved_mentions[{index - 1}].raw_text")
        assert raw_text is not None
        for ref_index, item_index in enumerate(mention.applies_to_item_indexes):
            if item_index >= len(items):
                raise SemanticIntentNormalizationError(
                    "unresolved item index가 answer_items 범위를 벗어났습니다",
                    diagnostic_codes=("item_reference_invalid",),
                    diagnostic_paths=(
                        "unresolved_mentions"
                        f"[{index - 1}].applies_to_item_indexes[{ref_index}]",),
                )
        unresolved.append(UnresolvedMention(
            mention_id=f"unresolved-{index}", role_hint=mention.role_hint,
            raw_text=raw_text,
            applies_to_item_ids=[
                f"item-{item_index + 1}"
                for item_index in mention.applies_to_item_indexes
            ],
        ))

    return SemanticIntent(
        schema_version=STAGE1_SEMANTIC_INTENT_V1,
        entities=entities,
        answer_items=items,
        answer_groups=groups,
        premises=premises,
        unresolved_mentions=unresolved,
        presentation=wire.presentation,
    )


def canonical_json(value: Any) -> str:
    """Canonical sorted compact JSON for v1 artifact and intent digests."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def semantic_intent_digest(
        intent: SemanticIntent | Mapping[str, Any],
        ) -> str:
    if isinstance(intent, SemanticIntent):
        validated = SemanticIntent.model_validate(
            intent.model_dump(mode="python", warnings=False), strict=True)
    elif isinstance(intent, Mapping):
        validated = SemanticIntent.model_validate(intent, strict=True)
    else:
        raise TypeError("intent는 SemanticIntent 또는 mapping이어야 합니다")
    return canonical_sha256(validated)


def provider_schema():
    """Compile and return the HCX-safe provider schema contract."""

    # Local import avoids a module cycle: the post-schema boundary itself uses
    # these wire models.  Only the semantic wire receives this repairer; the
    # legacy planner/smoke schemas retain their existing repair policy.
    from .semantic_intent_v1_boundary import (
        repair_semantic_intent_wire_payload,
    )

    return compile_hcx_schema(
        HcxSemanticIntentWire,
        payload_repairer=repair_semantic_intent_wire_payload,
    )


def compile_hcx_semantic_intent_v1_schema():
    """Return the HCX-compiled provider wire schema."""

    return provider_schema()


def normalized_schema() -> dict[str, Any]:
    """Return the formal JSON Schema for the normalized intent."""

    return SemanticIntent.model_json_schema(mode="validation")


def normalized_schema_canonical_json() -> str:
    return canonical_json(normalized_schema())


def _schema_artifact_bytes() -> tuple[bytes, bytes, bytes, bytes]:
    provider = provider_schema()
    provider_schema_bytes = provider.canonical_json.encode("utf-8")
    provider_digest_bytes = (
        f"{provider.sha256}  {SCHEMA_ARTIFACT.name}\n").encode("ascii")
    normalized_text = normalized_schema_canonical_json()
    normalized_schema_bytes = normalized_text.encode("utf-8")
    normalized_digest = canonical_sha256(normalized_schema())
    normalized_digest_bytes = (
        f"{normalized_digest}  {NORMALIZED_SCHEMA_ARTIFACT.name}\n"
    ).encode("ascii")
    return (
        provider_schema_bytes, provider_digest_bytes,
        normalized_schema_bytes, normalized_digest_bytes,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_semantic_intent_v1_schema_artifacts() -> tuple[str, str]:
    """Regenerate provider and normalized schema artifacts from source models."""

    provider_bytes, provider_digest, normalized_bytes, normalized_digest = (
        _schema_artifact_bytes())
    _atomic_write(SCHEMA_ARTIFACT, provider_bytes)
    _atomic_write(SCHEMA_DIGEST_ARTIFACT, provider_digest)
    _atomic_write(NORMALIZED_SCHEMA_ARTIFACT, normalized_bytes)
    _atomic_write(NORMALIZED_SCHEMA_DIGEST_ARTIFACT, normalized_digest)
    return provider_schema().sha256, canonical_sha256(normalized_schema())


def verify_semantic_intent_v1_schema_artifacts() -> tuple[str, str]:
    """Require all four artifacts to be byte-exact with source/compiler output."""

    expected = _schema_artifact_bytes()
    paths = (
        SCHEMA_ARTIFACT, SCHEMA_DIGEST_ARTIFACT,
        NORMALIZED_SCHEMA_ARTIFACT, NORMALIZED_SCHEMA_DIGEST_ARTIFACT,
    )
    for path, payload in zip(paths, expected, strict=True):
        try:
            actual = path.read_bytes()
        except FileNotFoundError as exc:
            raise RuntimeError(f"semantic intent schema artifact가 없습니다: {path}") from exc
        if actual != payload:
            raise RuntimeError(
                f"semantic intent schema artifact가 source와 다릅니다: {path}")
    return provider_schema().sha256, canonical_sha256(normalized_schema())


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage1 semantic intent v1 schema artifact management")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    digests = (
        write_semantic_intent_v1_schema_artifacts()
        if args.write else verify_semantic_intent_v1_schema_artifacts())
    print(
        f"PASS: {STAGE1_SEMANTIC_INTENT_V1} "
        f"provider_schema_sha256={digests[0]} "
        f"normalized_schema_sha256={digests[1]}"
    )
    return 0


# Descriptive aliases make the boundary pleasant to import while retaining one
# Pydantic class per wire/normalized shape.
HcxSemanticIntentEntityMention = WireEntityMention
HcxSemanticIntentTarget = WireTarget
HcxSemanticIntentScope = WireScope
HcxSemanticIntentSelection = WireSelection
HcxSemanticIntentOutputRequest = WireOutputRequest
HcxSemanticIntentAnswerItem = WireAnswerItem
HcxSemanticIntentAnswerGroup = WireAnswerGroup
HcxSemanticIntentPremise = WirePremise
HcxSemanticIntentUnresolvedMention = WireUnresolvedMention
SemanticIntentEntityMention = EntityMention
SemanticIntentTarget = Target
SemanticIntentScope = Scope
SemanticIntentSelection = Selection
SemanticIntentOutputRequest = OutputRequest
SemanticIntentAnswerItem = AnswerItem
SemanticIntentAnswerGroup = AnswerGroup
SemanticIntentPremise = Premise
SemanticIntentUnresolvedMention = UnresolvedMention


__all__ = [
    "HCX_SEMANTIC_INTENT_WIRE_V1", "STAGE1_SEMANTIC_INTENT_V1",
    "SCHEMA_ARTIFACT", "SCHEMA_DIGEST_ARTIFACT",
    "NORMALIZED_SCHEMA_ARTIFACT", "NORMALIZED_SCHEMA_DIGEST_ARTIFACT",
    "SemanticIntentError", "SemanticIntentNormalizationError", "ProjectionMode",
    "HcxSemanticIntentWire", "WireEntityMention", "WireTarget", "WireScope",
    "WireSelection", "WireOutputRequest", "WireAnswerItem", "WireAnswerGroup",
    "WirePremise", "WireUnresolvedMention", "SemanticIntent", "EntityMention",
    "Target", "Scope", "Selection", "OutputRequest", "AnswerItem", "AnswerGroup",
    "Premise", "UnresolvedMention", "normalize_semantic_intent",
    "canonical_json", "canonical_sha256", "semantic_intent_digest", "provider_schema",
    "compile_hcx_semantic_intent_v1_schema", "normalized_schema",
    "normalized_schema_canonical_json", "write_semantic_intent_v1_schema_artifacts",
    "verify_semantic_intent_v1_schema_artifacts",
    "normalize_hcx_semantic_intent",
    "normalize_hcx_semantic_intent_v1", "normalize_semantic_intent_v1",
    "HcxSemanticIntentWireV1", "Stage1SemanticIntentV1", "SemanticIntentV1",
    "HcxSemanticIntentEntityMention", "HcxSemanticIntentTarget",
    "HcxSemanticIntentScope", "HcxSemanticIntentSelection",
    "HcxSemanticIntentOutputRequest", "HcxSemanticIntentAnswerItem",
    "HcxSemanticIntentAnswerGroup", "HcxSemanticIntentPremise",
    "HcxSemanticIntentUnresolvedMention", "SemanticIntentEntityMention",
    "SemanticIntentTarget", "SemanticIntentScope", "SemanticIntentSelection",
    "SemanticIntentOutputRequest", "SemanticIntentAnswerItem",
    "SemanticIntentAnswerGroup", "SemanticIntentPremise",
    "SemanticIntentUnresolvedMention",
]


normalize_hcx_semantic_intent = normalize_semantic_intent
normalize_hcx_semantic_intent_v1 = normalize_semantic_intent
normalize_semantic_intent_v1 = normalize_semantic_intent
HcxSemanticIntentWireV1 = HcxSemanticIntentWire
Stage1SemanticIntentV1 = SemanticIntent
SemanticIntentV1 = SemanticIntent


if __name__ == "__main__":
    raise SystemExit(_main())
