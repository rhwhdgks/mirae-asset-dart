"""Strict, hash-bound Stage1 compiled answer contract v1.

This module is deliberately independent of the legacy v0.4/v0.5 plan models.  It
contains the small contract that sits between the resolver/compiler and a v1
consumer.  The important boundary is field-level: every user requested field is
bound to a live answer root (``executable``/``qualified``), a registered typed
limitation (``qualified``/``limited``), or both, never silently dropped.

The models are strict and frozen.  A normal ``model_validate`` therefore does
not coerce values such as ``1`` to ``True`` or ``1`` to ``"1"``.  The top-level
contract owns a canonical SHA-256 digest over itself with ``contract_digest``
removed; this makes both creation and later tamper detection deterministic.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Iterable, Literal, Mapping

from pydantic import (
    BaseModel, ConfigDict, Field, StrictBool, StringConstraints, model_validator,
)
from typing_extensions import Annotated

from agent.semantic_intent_v1 import SemanticIntent, semantic_intent_digest


COMPILED_ANSWER_CONTRACT_V1 = "stage1-compiled-answer-contract/1.1"
LIMITATION_REGISTRY_V1 = "stage1-limitation-registry/1.0"

SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "compiled_answer_contract_v1.schema.json")
SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "compiled_answer_contract_v1.schema.sha256")


Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=256)]
PlanRootId = Annotated[str, StringConstraints(
    pattern=r"^plan-root-[1-9][0-9]*$")]
WholeTargetId = Annotated[str, StringConstraints(
    pattern=r"^whole-target-[1-9][0-9]*$")]


# These are field-level limitations observed in the frozen Gold/typed-limitation
# material.  The registry is intentionally closed: question-level ambiguity,
# terminal policy decisions, and resolver/transport/compiler failures belong to
# their own Stage1 disposition and must never be laundered as an answer field
# limitation.
_LIMITATION_CODE_FAMILY: dict[str, Literal[
    "source_scope", "identity_lineage", "ordering", "capability"
]] = {
    "source_cross_check_partial": "source_scope",
    "counterparty_not_reported_may_hide_match": "identity_lineage",
    "ambiguous_event_origin": "identity_lineage",
    "intraday_order_unavailable": "ordering",
    "source_scope_prevents_complete_lineage": "source_scope",
    "source_scope_raw_absent": "source_scope",
    "holding_lineage_root_missing": "source_scope",
    "personal_data_omitted": "capability",
}
LIMITATION_CODE_FAMILY: Mapping[str, str] = MappingProxyType(
    _LIMITATION_CODE_FAMILY)
LIMITATION_REGISTRY: Mapping[str, str] = LIMITATION_CODE_FAMILY

LimitationCode = Literal[
    "source_cross_check_partial",
    "counterparty_not_reported_may_hide_match",
    "ambiguous_event_origin",
    "intraday_order_unavailable",
    "source_scope_prevents_complete_lineage",
    "source_scope_raw_absent",
    "holding_lineage_root_missing",
    "personal_data_omitted",
]

FORBIDDEN_LIMITATION_CODES = frozenset({
    # User-resolvable question ambiguity.
    "ambiguity_requires_user",
    "needs_clarification",
    "clarification_required",
    # Technical failures.
    "resolver_not_ready",
    "resolver_unavailable",
    "transport_error",
    "provider_schema_error",
    "compiler_error",
    "internal_binding_error",
    # Terminal/policy/capability outcomes.
    "policy_blocked",
    "policy_refusal",
    "source_absent",
    "out_of_scope",
    "unsupported",
    "unsupported_request",
    "not_found",
})


class CompiledAnswerContractError(ValueError):
    """Raised when a v1 contract cannot be safely compiled or verified."""


class _StrictFrozenModel(BaseModel):
    """Base for every v1 model, including nested models.

    ``ContractModel`` from the legacy module intentionally is not used here: it
    forbids extras but permits Pydantic's normal coercion.  The v1 boundary must
    be strict at every nesting level.
    """

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True,
        revalidate_instances="always",
    )


def _unique(values: list[str], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label}에는 중복 값이 있을 수 없습니다")


def _canonical_json(value: Any) -> str:
    """Canonical JSON used for every v1 digest.

    Pydantic's JSON-mode dump is used before encoding so this function also works
    for a model containing future JSON-native scalar types.  UTF-8 is intentional
    because question surfaces may contain Korean text.
    """

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class ActivationPredicate(_StrictFrozenModel):
    """A boolean answer-root gate for a projected field.

    The answer-root and plan-value authorities are embedded in the execution
    plan, so this contract stores only the typed answer-root reference and the
    expected boolean.  The envelope closes the reference through those
    authorities before hand-off.
    """

    predicate_answer_root_ref: Identifier
    expected_boolean: StrictBool


class ProjectionField(_StrictFrozenModel):
    """One user-requested output field and its exclusive binding."""

    field_id: Identifier
    field_key: Identifier
    required: StrictBool = True
    binding_status: Literal["executable", "qualified", "limited"]
    answer_root_refs: list[Identifier] = Field(default_factory=list)
    limitation_refs: list[Identifier] = Field(default_factory=list)
    activation_predicate: ActivationPredicate | None = Field(
        default=None, exclude_if=lambda predicate: predicate is None)

    @model_validator(mode="after")
    def validate_binding(self) -> "ProjectionField":
        if self.required is not True:
            raise ValueError("projection field는 항상 required=true여야 합니다")
        if self.field_id.startswith("whole-target-"):
            raise ValueError(
                "named field ID는 whole-target namespace를 사용할 수 없습니다")
        _unique(self.answer_root_refs, label="answer_root_refs")
        _unique(self.limitation_refs, label="limitation_refs")
        if self.binding_status == "executable":
            if not self.answer_root_refs or self.limitation_refs:
                raise ValueError(
                    "executable field는 answer root만 가져야 합니다")
        elif self.binding_status == "qualified":
            if not self.answer_root_refs or not self.limitation_refs:
                raise ValueError(
                    "qualified field는 answer root와 limitation ref가 모두 필요합니다")
        else:
            if self.answer_root_refs or not self.limitation_refs:
                raise ValueError(
                    "limited field는 limitation ref만 가져야 합니다")
        return self


class WholeTargetBinding(_StrictFrozenModel):
    """Compiler-owned binding for the requested target object itself.

    This is deliberately not a user-named ``ProjectionField``.  Its identifier
    is derived from source item order and it carries no invented display key or
    surface.  The exact source target remains authoritative through the bound
    ``SemanticIntent`` and ``source_intent_digest``.
    """

    whole_target_id: WholeTargetId
    required: StrictBool = True
    binding_status: Literal["executable", "qualified", "limited"]
    answer_root_refs: list[Identifier] = Field(default_factory=list)
    limitation_refs: list[Identifier] = Field(default_factory=list)
    activation_predicate: ActivationPredicate | None = Field(
        default=None, exclude_if=lambda predicate: predicate is None)

    @model_validator(mode="after")
    def validate_binding(self) -> "WholeTargetBinding":
        if self.required is not True:
            raise ValueError("whole-target binding은 항상 required=true여야 합니다")
        _unique(self.answer_root_refs, label="whole-target answer_root_refs")
        _unique(self.limitation_refs, label="whole-target limitation_refs")
        if self.binding_status == "executable":
            if not self.answer_root_refs or self.limitation_refs:
                raise ValueError(
                    "executable whole-target은 answer root만 가져야 합니다")
        elif self.binding_status == "qualified":
            if not self.answer_root_refs or not self.limitation_refs:
                raise ValueError(
                    "qualified whole-target은 root와 limitation이 모두 필요합니다")
        elif self.answer_root_refs or not self.limitation_refs:
            raise ValueError(
                "limited whole-target은 limitation ref만 가져야 합니다")
        return self


class AnswerProjection(_StrictFrozenModel):
    projection_mode: Literal["named_fields", "whole_target"] = "named_fields"
    shape: Literal[
        "scalar", "record", "record_list", "comparison", "timeline",
        "narrative", "verdict",
    ]
    fields: list[ProjectionField] = Field(default_factory=list)
    whole_target: WholeTargetBinding | None = Field(
        default=None, exclude_if=lambda binding: binding is None)
    presentation: Literal["auto", "prose", "table", "list"] = "auto"
    sort: list[Identifier] | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "AnswerProjection":
        ids = [field.field_id for field in self.fields]
        _unique(ids, label="projection field_id")
        if self.projection_mode == "named_fields":
            if not self.fields or self.whole_target is not None:
                raise ValueError(
                    "named_fields projection은 field만 1개 이상 가져야 합니다")
        else:
            if self.fields or self.whole_target is None:
                raise ValueError(
                    "whole_target projection은 별도 whole-target binding만 가져야 합니다")
            if self.sort is not None:
                raise ValueError("whole_target projection에는 field sort가 없습니다")
        if self.sort is not None:
            _unique(self.sort, label="projection sort")
            if any(field_id not in set(ids) for field_id in self.sort):
                raise ValueError("projection sort는 live named field만 참조해야 합니다")
        return self


def _projection_bindings(
        projection: AnswerProjection,
        ) -> list[ProjectionField | WholeTargetBinding]:
    if projection.projection_mode == "named_fields":
        return list(projection.fields)
    if projection.whole_target is None:  # guarded by AnswerProjection validation
        raise ValueError("whole_target projection binding이 없습니다")
    return [projection.whole_target]


def _projection_binding_id(
        binding: ProjectionField | WholeTargetBinding,
        ) -> str:
    return (
        binding.field_id if isinstance(binding, ProjectionField)
        else binding.whole_target_id)


def _expected_whole_target_id(position: int) -> str:
    return f"whole-target-{position}"


class CoveragePartition(_StrictFrozenModel):
    """Coverage sets for a projection's required field IDs.

    ``qualified`` fields intentionally occur in both ``executable`` and
    ``limited``.  The overlap is closed against the projection field binding
    by :class:`CompiledAnswerItem`; this model only owns the required-union
    invariant and list-level uniqueness.
    """

    required: list[Identifier] = Field(min_length=1)
    executable: list[Identifier] = Field(default_factory=list)
    limited: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_partition(self) -> "CoveragePartition":
        _unique(self.required, label="coverage.required")
        _unique(self.executable, label="coverage.executable")
        _unique(self.limited, label="coverage.limited")
        required = set(self.required)
        executable = set(self.executable)
        limited = set(self.limited)
        if required != executable | limited:
            raise ValueError(
                "coverage는 required = executable ∪ limited이어야 합니다")
        return self


class SupportRequirement(_StrictFrozenModel):
    """System support needed for a field, without becoming a user field."""

    support_id: Identifier
    kind: Identifier
    required: StrictBool = True
    detail: str | None = None
    applies_to_field_ids: list[Identifier] = Field(default_factory=list)
    applies_to_whole_target_ids: list[WholeTargetId] = Field(
        default_factory=list, exclude_if=lambda refs: not refs)
    answer_root_refs: list[Identifier] = Field(default_factory=list)
    plan_root_refs: list[PlanRootId] = Field(
        default_factory=list, exclude_if=lambda refs: not refs)

    @model_validator(mode="after")
    def validate_support(self) -> "SupportRequirement":
        _unique(self.applies_to_field_ids, label="support field refs")
        _unique(self.applies_to_whole_target_ids,
                label="support whole-target refs")
        if set(self.applies_to_field_ids) & set(
                self.applies_to_whole_target_ids):
            raise ValueError("support field/whole-target namespace가 겹칩니다")
        if not self.applies_to_field_ids and not self.applies_to_whole_target_ids:
            raise ValueError("support는 field 또는 whole-target을 참조해야 합니다")
        _unique(self.answer_root_refs, label="support root refs")
        _unique(self.plan_root_refs, label="support plan root refs")
        return self


class SourceFieldBinding(_StrictFrozenModel):
    """Ordered source field surface -> compiled field-id proof."""

    surface: Identifier
    field_id: Identifier


class LimitationBinding(_StrictFrozenModel):
    """A registered field-level typed limitation."""

    limitation_id: Identifier
    code: LimitationCode
    family: Literal[
        "source_scope", "identity_lineage", "ordering", "capability"
    ]
    applies_to_field_ids: list[Identifier] = Field(default_factory=list)
    applies_to_whole_target_ids: list[WholeTargetId] = Field(
        default_factory=list, exclude_if=lambda refs: not refs)
    detail: str = Field(min_length=1)
    required_followup: str | None = None

    @model_validator(mode="after")
    def validate_registry_binding(self) -> "LimitationBinding":
        _unique(self.applies_to_field_ids, label="limitation field refs")
        _unique(self.applies_to_whole_target_ids,
                label="limitation whole-target refs")
        if not self.applies_to_field_ids and not self.applies_to_whole_target_ids:
            raise ValueError("limitation은 field 또는 whole-target을 참조해야 합니다")
        if set(self.applies_to_field_ids) & set(
                self.applies_to_whole_target_ids):
            raise ValueError("limitation field/whole-target namespace가 겹칩니다")
        expected = LIMITATION_CODE_FAMILY.get(self.code)
        if expected is None:
            # Literal validation normally catches this first; retain a clear
            # domain error for model_construct/third-party callers.
            if self.code in FORBIDDEN_LIMITATION_CODES:
                raise ValueError(
                    f"{self.code}는 question/technical/terminal 상태이므로 "
                    "limitation code가 될 수 없습니다")
            raise ValueError(f"등록되지 않은 limitation code입니다: {self.code}")
        if self.family != expected:
            raise ValueError(
                f"limitation code/family 불일치: {self.code} -> {expected}")
        return self


class CompiledAnswerItem(_StrictFrozenModel):
    """One atomic user promise and its complete field-level coverage."""

    item_id: Identifier
    status: Literal["ready", "partial", "unavailable"]
    projection: AnswerProjection
    coverage: CoveragePartition
    support_requirements: list[SupportRequirement] = Field(default_factory=list)
    limitation_bindings: list[LimitationBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_item(self) -> "CompiledAnswerItem":
        bindings = _projection_bindings(self.projection)
        binding_ids = [_projection_binding_id(binding) for binding in bindings]
        binding_by_id = {
            _projection_binding_id(binding): binding for binding in bindings}
        executable = {
            _projection_binding_id(binding) for binding in bindings
            if binding.binding_status in {"executable", "qualified"}
        }
        limited = {
            _projection_binding_id(binding) for binding in bindings
            if binding.binding_status in {"qualified", "limited"}
        }
        qualified = {
            _projection_binding_id(binding) for binding in bindings
            if binding.binding_status == "qualified"
        }
        if set(self.coverage.required) != set(binding_ids):
            raise ValueError(
                "coverage.required는 projection의 모든 required binding이어야 합니다")
        if set(self.coverage.executable) != executable:
            raise ValueError("coverage.executable과 field binding이 다릅니다")
        if set(self.coverage.limited) != limited:
            raise ValueError("coverage.limited와 field binding이 다릅니다")
        if (set(self.coverage.executable) & set(self.coverage.limited)) != qualified:
            raise ValueError(
                "coverage executable/limited overlap은 qualified field와 같아야 합니다")

        support_ids = [row.support_id for row in self.support_requirements]
        limitation_ids = [row.limitation_id for row in self.limitation_bindings]
        _unique(support_ids, label="support_id")
        _unique(limitation_ids, label="limitation_id")
        limitations = {row.limitation_id: row for row in self.limitation_bindings}

        for support in self.support_requirements:
            support_refs = (
                list(support.applies_to_field_ids)
                + list(support.applies_to_whole_target_ids))
            if any(binding_id not in binding_by_id for binding_id in support_refs):
                raise ValueError(
                    "support requirement가 존재하지 않는 projection을 참조합니다")
            expected_mode = self.projection.projection_mode
            if (
                (expected_mode == "named_fields"
                 and support.applies_to_whole_target_ids)
                or (expected_mode == "whole_target"
                    and support.applies_to_field_ids)
            ):
                raise ValueError("support projection mode가 item과 다릅니다")

        for binding in bindings:
            binding_id = _projection_binding_id(binding)
            if binding.binding_status == "executable":
                if binding.limitation_refs:
                    raise ValueError("executable projection에 limitation ref가 있습니다")
                continue
            if not binding.limitation_refs:
                raise ValueError(
                    f"{binding.binding_status} projection에는 limitation ref가 필요합니다")
            for ref in binding.limitation_refs:
                limitation = limitations.get(ref)
                limitation_refs = [] if limitation is None else (
                    list(limitation.applies_to_field_ids)
                    + list(limitation.applies_to_whole_target_ids))
                if limitation is None or binding_id not in limitation_refs:
                    raise ValueError(
                        "projection의 limitation ref가 item binding과 다릅니다")

        for limitation in self.limitation_bindings:
            refs = (
                list(limitation.applies_to_field_ids)
                + list(limitation.applies_to_whole_target_ids))
            if any(binding_id not in binding_by_id for binding_id in refs):
                raise ValueError("limitation이 존재하지 않는 projection을 가리킵니다")
            if any(binding_id not in limited for binding_id in refs):
                raise ValueError(
                    "limitation이 limitation-covered projection을 가리키지 않습니다")
            if any(limitation.limitation_id
                   not in binding_by_id[binding_id].limitation_refs
                   for binding_id in refs):
                raise ValueError(
                    "limitation binding과 projection 역참조가 일치하지 않습니다")
            if (
                (self.projection.projection_mode == "named_fields"
                 and limitation.applies_to_whole_target_ids)
                or (self.projection.projection_mode == "whole_target"
                    and limitation.applies_to_field_ids)
            ):
                raise ValueError("limitation projection mode가 item과 다릅니다")

        has_root_covered = bool(executable)
        has_limitation_covered = bool(limited)
        expected_status = (
            "partial" if has_root_covered and has_limitation_covered
            else "ready" if has_root_covered
            else "unavailable"
        )
        if self.status != expected_status:
            raise ValueError(
                f"item status가 field coverage와 다릅니다: 기대 {expected_status}")
        if has_limitation_covered and not self.limitation_bindings:
            raise ValueError("limited field에는 typed limitation이 필요합니다")
        if not has_limitation_covered and self.limitation_bindings:
            raise ValueError("ready item에는 limitation binding이 있을 수 없습니다")
        return self


class AnswerGroup(_StrictFrozenModel):
    group_id: Identifier
    item_ids: list[Identifier] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_group(self) -> "AnswerGroup":
        _unique(self.item_ids, label="group item_ids")
        return self


class CompiledPremiseContract(_StrictFrozenModel):
    """Compiler-owned verification contract for a user premise."""

    premise_id: Identifier
    kind: Literal["numeric", "state", "comparison", "existence", "causal"]
    raw_text: str = Field(min_length=1)
    applies_to_item_ids: list[Identifier] = Field(min_length=1)
    verification_root_refs: list[Identifier] = Field(default_factory=list)
    verification_task_refs: list[Identifier] = Field(default_factory=list)
    verification_plan_root_refs: list[PlanRootId] = Field(
        default_factory=list, exclude_if=lambda refs: not refs)
    verdict_requirement: Literal["required"] = "required"

    @model_validator(mode="after")
    def validate_premise(self) -> "CompiledPremiseContract":
        _unique(self.applies_to_item_ids, label="premise item refs")
        _unique(self.verification_root_refs, label="premise root refs")
        _unique(self.verification_task_refs, label="premise task refs")
        _unique(self.verification_plan_root_refs,
                label="premise plan root refs")
        if (not self.verification_root_refs
                and not self.verification_task_refs
                and not self.verification_plan_root_refs):
            raise ValueError(
                "premise에는 verification root/plan/task ref가 필요합니다")
        namespaces = {
            "answer": set(self.verification_root_refs),
            "plan": set(self.verification_plan_root_refs),
            "task": set(self.verification_task_refs),
        }
        for left, left_values in namespaces.items():
            for right, right_values in namespaces.items():
                if left >= right:
                    continue
                if left_values & right_values:
                    raise ValueError(
                        "premise answer/plan/task ref namespace가 겹칩니다")
        return self


class SourcePremiseAuthority(_StrictFrozenModel):
    """Lossless source-side premise identity used by the v1 builder.

    It intentionally has no verification refs or verdict: those are compiler
    outputs, not provider/source facts.  ``build`` compares this authority to
    the compiled premise's ID, kind, raw text and item applicability.
    """

    premise_id: Identifier
    kind: Literal["numeric", "state", "comparison", "existence", "causal"]
    raw_text: str = Field(min_length=1)
    applies_to_item_ids: list[Identifier] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_premise(self) -> "SourcePremiseAuthority":
        _unique(self.applies_to_item_ids, label="source premise item refs")
        return self


def _strict_semantic_intent(
        value: SemanticIntent | Mapping[str, Any],
        ) -> SemanticIntent:
    """Re-enter the strict normalized-intent boundary for every build."""

    if isinstance(value, SemanticIntent):
        return SemanticIntent.model_validate(
            value.model_dump(mode="python", warnings=False), strict=True)
    if isinstance(value, Mapping):
        return SemanticIntent.model_validate(value, strict=True)
    raise TypeError("source_intent는 SemanticIntent 또는 mapping이어야 합니다")


class CompiledAnswerContract(_StrictFrozenModel):
    """Canonical v1 answer contract for a ready or partial-ready turn."""

    schema_version: Literal[COMPILED_ANSWER_CONTRACT_V1] = (
        COMPILED_ANSWER_CONTRACT_V1)
    source_intent_digest: Digest
    completion: Literal["complete", "partial"]
    items: list[CompiledAnswerItem] = Field(min_length=1)
    groups: list[AnswerGroup] = Field(default_factory=list)
    premise_contracts: list[CompiledPremiseContract] = Field(default_factory=list)
    presentation: Literal["auto", "prose", "table", "list"] = "auto"
    execution_plan_digest: Digest
    contract_digest: Digest

    # The zero value is used only by ``create`` while computing the digest.  It
    # is never accepted by ordinary validation because Digest is 64 lowercase
    # hex and the after-validator recomputes it.
    _DIGEST_PLACEHOLDER: ClassVar[str] = "0" * 64

    @model_validator(mode="after")
    def validate_contract(self) -> "CompiledAnswerContract":
        item_ids = [item.item_id for item in self.items]
        _unique(item_ids, label="contract item_id")
        group_ids = [group.group_id for group in self.groups]
        _unique(group_ids, label="contract group_id")
        premise_ids = [row.premise_id for row in self.premise_contracts]
        _unique(premise_ids, label="premise_id")

        projection_ids = [
            _projection_binding_id(binding)
            for item in self.items
            for binding in _projection_bindings(item.projection)
        ]
        support_ids = [
            support.support_id
            for item in self.items
            for support in item.support_requirements
        ]
        limitation_ids = [
            limitation.limitation_id
            for item in self.items
            for limitation in item.limitation_bindings
        ]
        _unique(projection_ids, label="global projection binding ID")
        _unique(support_ids, label="global support_id")
        _unique(limitation_ids, label="global limitation_id")
        namespaces = {
            "projection": set(projection_ids),
            "support": set(support_ids),
            "limitation": set(limitation_ids),
        }
        for left, left_values in namespaces.items():
            for right, right_values in namespaces.items():
                if left >= right:
                    continue
                if left_values & right_values:
                    raise ValueError(
                        f"projection/support/limitation ID namespace가 겹칩니다: {left}/{right}")

        item_index = {item_id: index for index, item_id in enumerate(item_ids)}
        grouped: set[str] = set()
        for group in self.groups:
            if any(item_id not in item_index for item_id in group.item_ids):
                raise ValueError("group가 존재하지 않는 item을 참조합니다")
            if grouped & set(group.item_ids):
                raise ValueError("item은 둘 이상의 group에 속할 수 없습니다")
            grouped.update(group.item_ids)
            positions = [item_index[item_id] for item_id in group.item_ids]
            if positions != sorted(positions):
                raise ValueError("group item 순서는 질문 item 순서를 보존해야 합니다")

        for premise in self.premise_contracts:
            if any(item_id not in item_index for item_id in premise.applies_to_item_ids):
                raise ValueError("premise가 존재하지 않는 item을 참조합니다")

        executable = any(
            binding.binding_status in {"executable", "qualified"}
            for item in self.items
            for binding in _projection_bindings(item.projection))
        limited = any(
            binding.binding_status in {"qualified", "limited"}
            for item in self.items
            for binding in _projection_bindings(item.projection))
        expected_completion = "partial" if limited else "complete"
        if not executable:
            raise ValueError(
                "실행 가능한 projection이 없는 contract는 envelope로 발행할 수 없습니다")
        if self.completion != expected_completion:
            raise ValueError(
                f"completion이 item coverage와 다릅니다: 기대 {expected_completion}")
        actual = self.compute_digest(self.model_dump(mode="json"))
        if self.contract_digest != actual:
            raise ValueError("CompiledAnswerContract contract_digest가 일치하지 않습니다")
        return self

    @classmethod
    def compute_digest(cls, payload: Mapping[str, Any]) -> str:
        """Compute the digest over a payload without ``contract_digest``."""

        body = dict(payload)
        body.pop("contract_digest", None)
        return canonical_sha256(body)

    @staticmethod
    def _authority_refs(
            values: Iterable[str], *, label: str,
            ) -> set[str]:
        refs = list(values)
        if any(type(value) is not str or not value for value in refs):
            raise ValueError(f"{label}에는 비어 있지 않은 문자열 ref만 허용됩니다")
        _unique(refs, label=label)
        return set(refs)

    def validate_bindings(
            self, *, live_answer_root_refs: Iterable[str],
            live_plan_root_refs: Iterable[str] = (),
            live_task_refs: Iterable[str] = (),
            ) -> "CompiledAnswerContract":
        """Validate every root/task reference against the authoritative plan.

        The contract stores only references; the resolver/compiler owns the live
        root inventory.  A caller that has no such inventory must use the
        legacy-free ``build`` API below rather than pretending fabricated refs
        are executable.
        """

        live_roots = self._authority_refs(
            live_answer_root_refs, label="live_answer_root_refs")
        live_plan_roots = self._authority_refs(
            live_plan_root_refs, label="live_plan_root_refs")
        live_tasks = self._authority_refs(live_task_refs, label="live_task_refs")
        live_namespaces = {
            "answer": live_roots,
            "plan": live_plan_roots,
            "task": live_tasks,
        }
        for left, left_values in live_namespaces.items():
            for right, right_values in live_namespaces.items():
                if left >= right:
                    continue
                if left_values & right_values:
                    raise ValueError(
                        f"answer/plan/task ref namespace가 겹칩니다: {left}/{right}")

        referenced_roots: set[str] = set()
        referenced_plan_roots: set[str] = set()
        referenced_tasks: set[str] = set()
        for item in self.items:
            for binding in _projection_bindings(item.projection):
                referenced_roots.update(binding.answer_root_refs)
                if binding.activation_predicate is not None:
                    referenced_roots.add(
                        binding.activation_predicate.predicate_answer_root_ref)
            for support in item.support_requirements:
                referenced_roots.update(support.answer_root_refs)
                referenced_plan_roots.update(support.plan_root_refs)
        for premise in self.premise_contracts:
            referenced_roots.update(premise.verification_root_refs)
            referenced_plan_roots.update(premise.verification_plan_root_refs)
            referenced_tasks.update(premise.verification_task_refs)
        if not referenced_roots <= live_roots:
            dead = sorted(referenced_roots - live_roots)
            raise ValueError(f"fabricated/dead answer root ref입니다: {dead}")
        if not referenced_plan_roots <= live_plan_roots:
            dead = sorted(referenced_plan_roots - live_plan_roots)
            raise ValueError(f"fabricated/dead plan root ref입니다: {dead}")
        if not referenced_tasks <= live_tasks:
            dead = sorted(referenced_tasks - live_tasks)
            raise ValueError(f"fabricated/dead task ref입니다: {dead}")
        return self

    def validate_source_intent(
            self, source_intent: SemanticIntent | Mapping[str, Any],
            ) -> "CompiledAnswerContract":
        """Recheck the serialized contract against its normalized source intent.

        The contract keeps compiler-local field IDs rather than duplicating user
        surfaces.  The ordered field list is therefore validated positionally
        against the intent's ordered user fields; construction additionally
        verifies the full surface→field proof in :meth:`build`.
        """

        intent = _strict_semantic_intent(source_intent)
        if self.source_intent_digest != semantic_intent_digest(intent):
            raise ValueError("contract source_intent_digest가 strict source_intent와 다릅니다")
        if self.presentation != intent.presentation:
            raise ValueError("contract presentation이 source_intent와 다릅니다")
        if [item.item_id for item in self.items] != [
                item.item_id for item in intent.answer_items]:
            raise ValueError("contract item 순서가 source_intent와 다릅니다")
        for position, (compiled, source) in enumerate(
                zip(self.items, intent.answer_items, strict=True), start=1):
            if compiled.projection.shape != source.output.shape:
                raise ValueError("contract projection shape가 source_intent와 다릅니다")
            if compiled.projection.presentation != source.output.presentation:
                raise ValueError("contract projection presentation이 source_intent와 다릅니다")
            if compiled.projection.projection_mode != source.output.projection_mode:
                raise ValueError("contract projection mode가 source_intent와 다릅니다")
            if source.output.projection_mode == "named_fields":
                if (len(compiled.projection.fields)
                        != len(source.output.field_surfaces)):
                    raise ValueError(
                        "contract projection field count가 source_intent와 다릅니다")
            else:
                whole_target = compiled.projection.whole_target
                if (
                    source.output.field_surfaces
                    or compiled.projection.fields
                    or whole_target is None
                    or whole_target.whole_target_id
                    != _expected_whole_target_id(position)
                ):
                    raise ValueError(
                        "contract whole-target binding이 source item order와 다릅니다")
        expected_groups = [
            AnswerGroup(group_id=group.group_id, item_ids=list(group.item_ids))
            for group in intent.answer_groups
        ]
        if self.groups != expected_groups:
            raise ValueError("contract group이 source_intent와 다릅니다")
        expected_premises = [
            (premise.premise_id, premise.kind, premise.raw_text,
             premise.applies_to_item_ids)
            for premise in intent.premises
        ]
        actual_premises = [
            (premise.premise_id, premise.kind, premise.raw_text,
             premise.applies_to_item_ids)
            for premise in self.premise_contracts
        ]
        if actual_premises != expected_premises:
            raise ValueError("contract premise가 source_intent와 다릅니다")
        return self

    @classmethod
    def build(
            cls, *, source_intent: SemanticIntent | Mapping[str, Any],
            source_intent_digest: str,
            execution_plan_digest: str,
            items: list[CompiledAnswerItem | Mapping[str, Any]],
            groups: list[AnswerGroup | Mapping[str, Any]],
            source_field_bindings: Mapping[
                str, list[SourceFieldBinding | Mapping[str, Any]]
            ],
            live_answer_root_refs: Iterable[str],
            live_plan_root_refs: Iterable[str] = (),
            live_task_refs: Iterable[str] = (),
            premise_contracts: list[
                CompiledPremiseContract | Mapping[str, Any]
            ] | None = None,
            presentation: Literal["auto", "prose", "table", "list"] = "auto",
            ) -> "CompiledAnswerContract":
        """Build from authoritative source order/fields and live plan refs.

        This is the v1 entry point.  ``source_intent`` is the only source
        authority: item/group order, user-requested field surfaces, and premise
        specs are derived from its strict normalized form.  A caller may supply
        only compiler-owned field IDs/root bindings and the ordered
        surface→field-ID proof.  This prevents a matching digest string from
        being used to smuggle an unrequested field into a contract.
        """

        if type(groups) is not list:
            raise TypeError("groups는 list여야 합니다")
        if not isinstance(source_field_bindings, Mapping):
            raise TypeError("source_field_bindings는 mapping이어야 합니다")
        normalized_intent = _strict_semantic_intent(source_intent)
        actual_source_digest = semantic_intent_digest(normalized_intent)
        if source_intent_digest != actual_source_digest:
            raise ValueError("source_intent_digest가 strict source_intent와 다릅니다")
        normalized_items = [
            value if isinstance(value, CompiledAnswerItem)
            else CompiledAnswerItem.model_validate(value, strict=True)
            for value in items
        ]
        normalized_groups = [
            value if isinstance(value, AnswerGroup)
            else AnswerGroup.model_validate(value, strict=True)
            for value in groups
        ]
        normalized_premises = [
            value if isinstance(value, CompiledPremiseContract)
            else CompiledPremiseContract.model_validate(value, strict=True)
            for value in (premise_contracts or [])
        ]
        normalized_source_field_bindings = {
            item_id: [
                value if isinstance(value, SourceFieldBinding)
                else SourceFieldBinding.model_validate(value, strict=True)
                for value in values
            ]
            for item_id, values in source_field_bindings.items()
        }
        actual_order = [item.item_id for item in normalized_items]
        source_item_order = [item.item_id for item in normalized_intent.answer_items]
        if actual_order != source_item_order:
            raise ValueError("contract item 순서가 authoritative source 순서와 다릅니다")
        if set(normalized_source_field_bindings) != set(actual_order):
            raise ValueError(
                "source_field_bindings item 집합이 source item과 다릅니다")
        source_items = {
            item.item_id: item for item in normalized_intent.answer_items}
        for item in normalized_items:
            actual_fields = [field.field_id for field in item.projection.fields]
            source_output = source_items[item.item_id].output
            expected_surfaces = source_output.field_surfaces
            proof = normalized_source_field_bindings[item.item_id]
            proof_surfaces = [binding.surface for binding in proof]
            proof_field_ids = [binding.field_id for binding in proof]
            _unique(proof_surfaces, label="source field surfaces")
            _unique(proof_field_ids, label="source field proof IDs")
            if proof_surfaces != expected_surfaces:
                raise ValueError(
                    f"{item.item_id} source surface proof가 다릅니다")
            if proof_field_ids != actual_fields:
                raise ValueError(
                    f"{item.item_id} source surface -> field_id proof가 다릅니다")
            if item.projection.projection_mode != source_output.projection_mode:
                raise ValueError(
                    f"{item.item_id} compiled/source projection mode가 다릅니다")
            position = source_item_order.index(item.item_id) + 1
            if source_output.projection_mode == "whole_target":
                whole_target = item.projection.whole_target
                if (
                    proof or actual_fields or whole_target is None
                    or whole_target.whole_target_id
                    != _expected_whole_target_id(position)
                ):
                    raise ValueError(
                        f"{item.item_id} whole-target authority가 다릅니다")

        source_groups = [
            AnswerGroup(group_id=group.group_id, item_ids=list(group.item_ids))
            for group in normalized_intent.answer_groups
        ]
        if normalized_groups != source_groups:
            raise ValueError("contract group item 순서가 source group과 다릅니다")

        actual_premise_order = [premise.premise_id for premise in normalized_premises]
        source_premise_order = [
            premise.premise_id for premise in normalized_intent.premises]
        if actual_premise_order != source_premise_order:
            raise ValueError(
                "contract premise 순서/ID가 authoritative source와 다릅니다")
        normalized_source_premises = [
            SourcePremiseAuthority(
                premise_id=premise.premise_id,
                kind=premise.kind,
                raw_text=premise.raw_text,
                applies_to_item_ids=list(premise.applies_to_item_ids),
            )
            for premise in normalized_intent.premises
        ]
        source_premise_specs = [
            (premise.premise_id, premise.kind, premise.raw_text,
             premise.applies_to_item_ids)
            for premise in normalized_source_premises
        ]
        compiled_premise_specs = [
            (premise.premise_id, premise.kind, premise.raw_text,
             premise.applies_to_item_ids)
            for premise in normalized_premises
        ]
        if compiled_premise_specs != source_premise_specs:
            raise ValueError(
                "contract premise raw_text/kind/applies_to가 source와 다릅니다")

        payload: dict[str, Any] = {
            "source_intent_digest": source_intent_digest,
            "execution_plan_digest": execution_plan_digest,
            "items": normalized_items,
            "groups": normalized_groups,
            "premise_contracts": normalized_premises,
            "presentation": presentation,
            "completion": (
                "partial" if any(
                    binding.binding_status in {"qualified", "limited"}
                    for item in normalized_items
                    for binding in _projection_bindings(item.projection))
                else "complete"),
        }
        contract = cls._create_unbound(**payload)
        contract.validate_source_intent(normalized_intent)
        return contract.validate_bindings(
            live_answer_root_refs=live_answer_root_refs,
            live_plan_root_refs=live_plan_root_refs,
            live_task_refs=live_task_refs,
        )

    @classmethod
    def _create_unbound(cls, **payload: Any) -> "CompiledAnswerContract":
        """Internal constructor used only after authority checks in ``build``."""

        # ``model_construct`` intentionally skips validation (including nested
        # defaults).  Normalize every nested object first so the bytes hashed
        # here are exactly the bytes emitted by the final model serializer.
        normalized = dict(payload)
        normalized["items"] = [
            value if isinstance(value, CompiledAnswerItem)
            else CompiledAnswerItem.model_validate(value, strict=True)
            for value in normalized.get("items", [])
        ]
        normalized["groups"] = [
            value if isinstance(value, AnswerGroup)
            else AnswerGroup.model_validate(value, strict=True)
            for value in normalized.get("groups", [])
        ]
        normalized["premise_contracts"] = [
            value if isinstance(value, CompiledPremiseContract)
            else CompiledPremiseContract.model_validate(value, strict=True)
            for value in normalized.get("premise_contracts", [])
        ]
        candidate = cls.model_construct(
            **normalized, contract_digest=cls._DIGEST_PLACEHOLDER)
        body = candidate.model_dump(mode="json")
        body.pop("contract_digest", None)
        body["contract_digest"] = cls.compute_digest(body)
        return cls.model_validate(body, strict=True)

    @classmethod
    def create(
            cls, *,
            source_intent: SemanticIntent | Mapping[str, Any] | None = None,
            groups: list[AnswerGroup | Mapping[str, Any]] | None = None,
            source_field_bindings: Mapping[
                str, list[SourceFieldBinding | Mapping[str, Any]]
            ] | None = None,
            live_answer_root_refs: Iterable[str] | None = None,
            live_plan_root_refs: Iterable[str] | None = None,
            live_task_refs: Iterable[str] | None = None,
            **payload: Any,
            ) -> "CompiledAnswerContract":
        """Safe public builder; production construction requires authority."""

        authority_args = (
            source_intent, groups, source_field_bindings,
            live_answer_root_refs, live_plan_root_refs, live_task_refs)
        if any(value is not None for value in authority_args):
            if any(value is None for value in authority_args[:3]):
                raise ValueError(
                    "strict source_intent/group/field binding authority는 모두 제공해야 합니다")
            supplied_version = payload.pop("schema_version", COMPILED_ANSWER_CONTRACT_V1)
            if supplied_version != COMPILED_ANSWER_CONTRACT_V1:
                raise ValueError("CompiledAnswerContract schema_version이 다릅니다")
            supplied_completion = payload.pop("completion", None)
            source_intent_digest = payload.pop("source_intent_digest")
            execution_plan_digest = payload.pop("execution_plan_digest")
            items = payload.pop("items")
            premise_contracts = payload.pop("premise_contracts", None)
            presentation = payload.pop("presentation", "auto")
            if payload:
                raise ValueError(
                    f"CompiledAnswerContract.create에 알 수 없는 필드가 있습니다: "
                    f"{sorted(payload)}")
            contract = cls.build(
                source_intent_digest=source_intent_digest,
                source_intent=source_intent,  # type: ignore[arg-type]
                groups=groups,  # type: ignore[arg-type]
                execution_plan_digest=execution_plan_digest,
                items=items,
                source_field_bindings=source_field_bindings,  # type: ignore[arg-type]
                live_answer_root_refs=(
                    live_answer_root_refs
                    if live_answer_root_refs is not None else ()),
                live_plan_root_refs=(
                    live_plan_root_refs
                    if live_plan_root_refs is not None else ()),
                live_task_refs=(
                    live_task_refs if live_task_refs is not None else ()),
                premise_contracts=premise_contracts,
                presentation=presentation,
            )
            if supplied_completion is not None and supplied_completion != contract.completion:
                raise ValueError("CompiledAnswerContract completion이 실제 coverage와 다릅니다")
            return contract
        raise ValueError(
            "CompiledAnswerContract.create는 authoritative build 입력이 필요합니다; "
            "저수준 생성은 내부 _create_unbound만 사용합니다")

    def verify_digest(self) -> bool:
        """Return whether this instance still matches its canonical digest.

        ``model_copy(update=...)`` can intentionally create an unvalidated copy
        of a frozen Pydantic model.  This method therefore recomputes from the
        current serialized payload instead of trusting construction-time state.
        """

        try:
            # Re-validation is intentional.  ``model_copy(update=...)`` can
            # bypass Pydantic's construction validators even for frozen models.
            validated = type(self).model_validate(
                self.model_dump(mode="json", warnings=False), strict=True)
        except (TypeError, ValueError):
            return False
        return validated.contract_digest == validated.compute_digest(
            validated.model_dump(mode="json"))

    def assert_digest(self) -> "CompiledAnswerContract":
        if not self.verify_digest():
            raise CompiledAnswerContractError(
                "CompiledAnswerContract contract_digest가 일치하지 않습니다")
        return self


# Compatibility-friendly names for callers that prefer the explicit v1 suffix.
CompiledAnswerItemV1 = CompiledAnswerItem
CompiledPremiseContractV1 = CompiledPremiseContract
CompiledAnswerContractV1 = CompiledAnswerContract
Coverage = CoveragePartition
TypedLimitation = LimitationBinding


def verify_compiled_answer_contract_digest(
        value: CompiledAnswerContract | Mapping[str, Any],
        ) -> str:
    """Strictly validate a contract and return its verified digest."""

    serialized = value.model_dump(mode="json", warnings=False) if isinstance(
        value, CompiledAnswerContract) else dict(value)
    # Always round-trip through strict validation, including for model inputs.
    # This is what detects an unvalidated ``model_copy(update=...)`` instance.
    contract = CompiledAnswerContract.model_validate(serialized, strict=True)
    if not contract.verify_digest():
        raise CompiledAnswerContractError("contract digest verification failed")
    return contract.contract_digest


def compile_compiled_answer_contract_v1_schema() -> dict[str, Any]:
    """Return the canonical JSON-schema source for the top-level contract."""

    return CompiledAnswerContract.model_json_schema()


def _schema_artifact_bytes() -> tuple[bytes, bytes, str]:
    schema_text = _canonical_json(compile_compiled_answer_contract_v1_schema())
    schema_bytes = schema_text.encode("utf-8")
    digest = sha256(schema_bytes).hexdigest()
    digest_bytes = (
        f"{digest}  {SCHEMA_ARTIFACT.name}\n").encode("ascii")
    return schema_bytes, digest_bytes, digest


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


def write_compiled_answer_contract_v1_schema_artifacts() -> str:
    schema_bytes, digest_bytes, digest = _schema_artifact_bytes()
    _atomic_write(SCHEMA_ARTIFACT, schema_bytes)
    _atomic_write(SCHEMA_DIGEST_ARTIFACT, digest_bytes)
    return digest


def verify_compiled_answer_contract_v1_schema_artifacts() -> str:
    expected_schema, expected_digest, digest = _schema_artifact_bytes()
    try:
        actual_schema = SCHEMA_ARTIFACT.read_bytes()
        actual_digest = SCHEMA_DIGEST_ARTIFACT.read_bytes()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "CompiledAnswerContract v1 schema artifact가 없습니다") from exc
    if actual_schema != expected_schema or actual_digest != expected_digest:
        raise RuntimeError(
            "CompiledAnswerContract v1 schema artifact가 model/compiler와 다릅니다")
    return digest


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="CompiledAnswerContract v1 schema artifact 관리")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    digest = (
        write_compiled_answer_contract_v1_schema_artifacts()
        if args.write else verify_compiled_answer_contract_v1_schema_artifacts())
    print(f"PASS: {COMPILED_ANSWER_CONTRACT_V1} schema_sha256={digest}")
    return 0


__all__ = [
    "ActivationPredicate",
    "AnswerGroup",
    "AnswerProjection",
    "COMPILED_ANSWER_CONTRACT_V1",
    "CompiledAnswerContract",
    "CompiledAnswerContractError",
    "CompiledAnswerContractV1",
    "CompiledAnswerItem",
    "CompiledAnswerItemV1",
    "CompiledPremiseContract",
    "CompiledPremiseContractV1",
    "Coverage",
    "CoveragePartition",
    "LIMITATION_CODE_FAMILY",
    "LIMITATION_REGISTRY",
    "LIMITATION_REGISTRY_V1",
    "LimitationBinding",
    "ProjectionField",
    "WholeTargetBinding",
    "WholeTargetId",
    "PlanRootId",
    "SCHEMA_ARTIFACT",
    "SCHEMA_DIGEST_ARTIFACT",
    "SourceFieldBinding",
    "SourcePremiseAuthority",
    "SupportRequirement",
    "TypedLimitation",
    "canonical_sha256",
    "compile_compiled_answer_contract_v1_schema",
    "verify_compiled_answer_contract_digest",
    "verify_compiled_answer_contract_v1_schema_artifacts",
    "write_compiled_answer_contract_v1_schema_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(_main())
