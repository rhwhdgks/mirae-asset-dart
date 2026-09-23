"""Typed Stage1ReadyEnvelope 0.3 boundary.

The envelope is the last Stage1 hand-off before Stage2.  It embeds the full
typed ``ExecutionPlan`` and the full ``CompiledAnswerContract``; neither is
represented by an opaque digest or an unvalidated dictionary.  Every digest
verification round-trips through strict validation, which also closes the
``model_copy(update=...)`` escape hatch.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import (
    BaseModel, ConfigDict, StringConstraints, field_validator,
    model_validator,
)
from typing_extensions import Annotated

from agent.compiled_answer_contract_v1 import CompiledAnswerContract
from agent.deterministic_plan_compiler_v1 import ExecutionPlan
from agent.semantic_intent_v1 import SemanticIntent, semantic_intent_digest


ENVELOPE_VERSION = "stage1-ready-envelope/0.3"
SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_ready_envelope_v03.schema.json")
SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_ready_envelope_v03.schema.sha256")

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
BuildId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
Versioned = Annotated[str, StringConstraints(
    min_length=3,
    pattern=r"^[A-Za-z0-9_.-]+/[0-9]+(?:\.[0-9]+)*$",
)]


class Stage1ReadyEnvelopeError(ValueError):
    """Raised when an envelope is not a safe v1 Stage2 hand-off."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True,
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


def _strict_embedded(model: type[BaseModel], value: Any) -> BaseModel:
    """Accept only a full model or mapping and strictly revalidate it."""

    if isinstance(value, model):
        raw = value.model_dump(mode="json", warnings=False)
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise TypeError(
            "Envelope embedded plan/contract는 typed model 또는 full mapping이어야 합니다")
    return model.model_validate(raw, strict=True)


def _plan_task_ids(plan: ExecutionPlan) -> set[str]:
    return {task.task_id for task in plan.resolved_plan.tasks}


def _projection_bindings(item: Any) -> list[tuple[str, str, Any]]:
    """Return mode, compiler binding ID, and binding for one answer item."""
    projection = item.projection
    if projection.projection_mode == "named_fields":
        return [
            ("named_fields", field.field_id, field)
            for field in projection.fields
        ]
    if projection.whole_target is None:
        raise ValueError("whole-target projection binding이 없습니다")
    return [(
        "whole_target",
        projection.whole_target.whole_target_id,
        projection.whole_target,
    )]


class Stage1ReadyEnvelope(_StrictFrozenModel):
    """Strict, typed Stage1 v1 hand-off.

    ``status`` and ``completion`` are derived from the embedded contract.  The
    public builder accepts optional caller copies only to compare them against
    the derived values; it never lets a caller choose a clarification,
    terminal, or technical status for this envelope.
    """

    schema_version: Literal[ENVELOPE_VERSION] = ENVELOPE_VERSION
    status: Literal["ready", "partial_ready"]
    completion: Literal["complete", "partial"]
    source_intent_digest: Digest
    execution_plan_digest: Digest
    contract_digest: Digest
    canonical_build_id: BuildId
    resolver_version: Versioned
    source_intent: SemanticIntent
    execution_plan: ExecutionPlan
    answer_contract: CompiledAnswerContract
    binding_digest: Digest

    @field_validator("execution_plan", mode="before")
    @classmethod
    def strict_execution_plan(cls, value: Any) -> ExecutionPlan:
        return _strict_embedded(ExecutionPlan, value)  # type: ignore[return-value]

    @field_validator("source_intent", mode="before")
    @classmethod
    def strict_source_intent(cls, value: Any) -> SemanticIntent:
        return _strict_embedded(SemanticIntent, value)  # type: ignore[return-value]

    @field_validator("answer_contract", mode="before")
    @classmethod
    def strict_answer_contract(cls, value: Any) -> CompiledAnswerContract:
        return _strict_embedded(CompiledAnswerContract, value)  # type: ignore[return-value]

    @classmethod
    def compute_binding_digest(cls, payload: Mapping[str, Any]) -> str:
        body = dict(payload)
        body.pop("binding_digest", None)
        return canonical_sha256(body)

    @property
    def contract(self) -> CompiledAnswerContract:
        """Compatibility accessor; the serialized field remains answer_contract."""

        return self.answer_contract

    @model_validator(mode="after")
    def validate_envelope(self) -> "Stage1ReadyEnvelope":
        plan = self.execution_plan
        contract = self.answer_contract
        intent = self.source_intent

        # Cross-envelope identity and digest equality.
        if self.source_intent_digest != semantic_intent_digest(intent):
            raise ValueError("source_intent_digest와 strict source_intent가 다릅니다")
        if self.source_intent_digest != plan.source_intent_digest:
            raise ValueError("source_intent_digest와 execution plan이 다릅니다")
        if self.source_intent_digest != contract.source_intent_digest:
            raise ValueError("source_intent_digest와 answer contract가 다릅니다")
        if self.execution_plan_digest != plan.execution_plan_digest:
            raise ValueError("envelope execution_plan_digest와 plan이 다릅니다")
        if self.execution_plan_digest != contract.execution_plan_digest:
            raise ValueError("execution_plan_digest와 answer contract가 다릅니다")
        if self.contract_digest != contract.contract_digest:
            raise ValueError("envelope contract_digest와 answer contract가 다릅니다")
        if self.canonical_build_id != plan.canonical_build_id:
            raise ValueError("canonical_build_id와 execution plan이 다릅니다")
        if self.resolver_version != plan.resolver_version:
            raise ValueError("resolver_version과 execution plan이 다릅니다")

        expected_completion = contract.completion
        expected_status = (
            "partial_ready" if expected_completion == "partial" else "ready")
        if self.completion != expected_completion:
            raise ValueError("envelope completion은 contract에서 파생되어야 합니다")
        if self.status != expected_status:
            raise ValueError("envelope status는 contract completion에서 파생되어야 합니다")

        executable_bindings = [
            binding
            for item in contract.items
            for _mode, _binding_id, binding in _projection_bindings(item)
            if binding.binding_status in {"executable", "qualified"}
        ]
        if not executable_bindings:
            raise ValueError(
                "executable projection이 없는 contract는 envelope가 될 수 없습니다")

        # Contract refs must be backed by the embedded plan inventory and by
        # the exact item/field binding declared by each plan root.
        root_by_id = {row.root_id: row for row in plan.answer_roots}
        plan_value_root_by_output = {
            row.plan_output_id: row for row in plan.plan_value_roots}
        support_by_id = {row.support_id: row for row in plan.support_roots}
        task_ids = _plan_task_ids(plan)
        contract.validate_bindings(
            live_answer_root_refs=list(root_by_id),
            live_plan_root_refs=[
                row.plan_root_id for row in plan.plan_value_roots],
            live_task_refs=task_ids,
        )
        contract.validate_source_intent(intent)

        contract_answer_inventory: dict[str, tuple[str, str, str]] = {}
        for item in contract.items:
            for mode, binding_id, binding in _projection_bindings(item):
                if binding.binding_status not in {"executable", "qualified"}:
                    binding_root_refs = []
                else:
                    binding_root_refs = binding.answer_root_refs
                for root_ref in binding_root_refs:
                    authority = (item.item_id, mode, binding_id)
                    previous = contract_answer_inventory.setdefault(
                        root_ref, authority)
                    if previous != authority:
                        raise ValueError(
                            "contract executable answer root가 중복 bound되어 "
                            "있거나 projection binding이 다릅니다")
        plan_answer_inventory = [
            (row.root_id, row.item_id, row.projection_mode,
             row.projection_binding_id)
            for row in plan.answer_roots
        ]
        if (
            set((root_ref, *binding)
                for root_ref, binding in contract_answer_inventory.items())
            != set(plan_answer_inventory)
            or len(contract_answer_inventory) != len(plan_answer_inventory)
        ):
            raise ValueError(
                "contract executable answer-root inventory가 plan과 정확히 "
                "일치하지 않습니다")

        for item in contract.items:
            for mode, binding_id, binding in _projection_bindings(item):
                for root_ref in binding.answer_root_refs:
                    root = root_by_id.get(root_ref)
                    if root is None:
                        raise ValueError("contract answer root가 plan inventory에 없습니다")
                    if (
                            root.item_id != item.item_id
                            or root.projection_mode != mode
                            or root.projection_binding_id != binding_id
                    ):
                        raise ValueError(
                            "contract projection과 plan answer root binding이 다릅니다")
                if binding.activation_predicate is not None:
                    predicate_ref = (
                        binding.activation_predicate.predicate_answer_root_ref)
                    if predicate_ref not in contract_answer_inventory:
                        raise ValueError(
                            "activation predicate answer root가 contract "
                            "answer-root inventory에 없습니다")
                    predicate_root = root_by_id.get(predicate_ref)
                    if predicate_root is None or predicate_root.plan_output_id is None:
                        raise ValueError(
                            "activation predicate는 plan output answer root여야 "
                            "합니다")
                    value_root = plan_value_root_by_output.get(
                        predicate_root.plan_output_id)
                    if value_root is None or value_root.value_kind != "boolean":
                        raise ValueError(
                            "activation predicate plan output은 등록된 boolean "
                            "plan value root여야 합니다")
        contract_supports = [
            support
            for item in contract.items
            for support in item.support_requirements
        ]
        support_item_by_id = {
            support.support_id: item
            for item in contract.items
            for support in item.support_requirements
        }
        contract_support_ids = [
            support.support_id for support in contract_supports]
        plan_support_ids = [
            support.support_id for support in plan.support_roots]
        if contract_support_ids != plan_support_ids:
            raise ValueError(
                "contract/plan support inventory가 정확히 일치하지 않습니다")
        for support in contract_supports:
            # Every required support in this envelope must be live-root-bound;
            # an unbound support cannot be silently treated as satisfied.
            if support.required and not support.answer_root_refs:
                raise ValueError("required support는 live answer root에 bound되어야 합니다")
            item = support_item_by_id[support.support_id]
            plan_support = support_by_id.get(support.support_id)
            if plan_support is None:
                raise ValueError("contract support가 plan inventory에 없습니다")
            root_ref = plan_support.root_id
            root = root_by_id.get(root_ref)
            expected_field_refs = (
                [plan_support.projection_binding_id]
                if plan_support.projection_mode == "named_fields" else [])
            expected_whole_target_refs = (
                [plan_support.projection_binding_id]
                if plan_support.projection_mode == "whole_target" else [])
            if (
                support.answer_root_refs != [root_ref]
                or support.kind != plan_support.kind
                or support.applies_to_field_ids != expected_field_refs
                or support.applies_to_whole_target_ids
                != expected_whole_target_refs
                or support.plan_root_refs != plan_support.plan_root_ids
                or root is None
                or root.item_id != item.item_id
                or root.projection_mode != plan_support.projection_mode
                or root.projection_binding_id
                != plan_support.projection_binding_id
            ):
                raise ValueError(
                    "contract support와 plan support root binding이 다릅니다")

        contract_premise_ids = [
            premise.premise_id for premise in contract.premise_contracts]
        plan_premise_ids = [
            premise.premise_id for premise in plan.premise_roots]
        if contract_premise_ids != plan_premise_ids:
            raise ValueError(
                "contract/plan premise ID inventory가 정확히 일치하지 않습니다")
        for premise, plan_premise in zip(
                contract.premise_contracts, plan.premise_roots):
            if (
                premise.verification_root_refs != plan_premise.root_ids
                or premise.verification_plan_root_refs
                != plan_premise.plan_root_ids
                or premise.verification_task_refs != plan_premise.task_refs
            ):
                raise ValueError(
                    "contract premise root/task binding이 plan과 다릅니다")

        actual_binding = self.compute_binding_digest(
            self.model_dump(mode="json", warnings=False))
        if self.binding_digest != actual_binding:
            raise ValueError("Stage1ReadyEnvelope binding_digest가 일치하지 않습니다")
        return self

    @classmethod
    def create(
            cls, *,
            source_intent_digest: str,
            execution_plan_digest: str,
            canonical_build_id: str,
            resolver_version: str,
            source_intent: SemanticIntent | Mapping[str, Any],
            execution_plan: ExecutionPlan | Mapping[str, Any],
            answer_contract: CompiledAnswerContract | Mapping[str, Any],
            contract_digest: str | None = None,
            binding_digest: str | None = None,
            schema_version: str = ENVELOPE_VERSION,
            status: str | None = None,
            completion: str | None = None,
            **extra: Any,
            ) -> "Stage1ReadyEnvelope":
        if extra:
            raise ValueError(f"알 수 없는 envelope builder 필드: {sorted(extra)}")
        if schema_version != ENVELOPE_VERSION:
            raise ValueError("Stage1ReadyEnvelope schema_version이 다릅니다")

        plan = _strict_embedded(ExecutionPlan, execution_plan)
        intent = _strict_embedded(SemanticIntent, source_intent)
        contract = _strict_embedded(CompiledAnswerContract, answer_contract)
        derived_completion = contract.completion
        derived_status = (
            "partial_ready" if derived_completion == "partial" else "ready")
        if status is not None and status != derived_status:
            raise ValueError("caller status가 contract에서 파생된 status와 다릅니다")
        if completion is not None and completion != derived_completion:
            raise ValueError("caller completion이 contract와 다릅니다")
        derived_contract_digest = contract.contract_digest
        if contract_digest is not None and contract_digest != derived_contract_digest:
            raise ValueError("caller contract_digest가 embedded contract와 다릅니다")

        body: dict[str, Any] = {
            "schema_version": ENVELOPE_VERSION,
            "status": derived_status,
            "completion": derived_completion,
            "source_intent_digest": source_intent_digest,
            "execution_plan_digest": execution_plan_digest,
            "contract_digest": derived_contract_digest,
            "canonical_build_id": canonical_build_id,
            "resolver_version": resolver_version,
            "source_intent": intent,
            "execution_plan": plan,
            "answer_contract": contract,
        }
        derived_binding = cls.compute_binding_digest(
            cls.model_construct(**body, binding_digest="0" * 64).model_dump(
                mode="json", warnings=False))
        if binding_digest is not None and binding_digest != derived_binding:
            raise ValueError("caller binding_digest가 canonical body와 다릅니다")
        body["binding_digest"] = derived_binding
        return cls.model_validate(body, strict=True)

    def verify_binding_digest(self) -> bool:
        try:
            serialized = self.model_dump(mode="json", warnings=False)
            validated = type(self).model_validate(serialized, strict=True)
        except (TypeError, ValueError):
            return False
        return validated.binding_digest == type(self).compute_binding_digest(
            validated.model_dump(mode="json", warnings=False))

    def assert_binding_digest(self) -> "Stage1ReadyEnvelope":
        if not self.verify_binding_digest():
            raise Stage1ReadyEnvelopeError(
                "Stage1ReadyEnvelope binding_digest 검증에 실패했습니다")
        return self


def verify_stage1_ready_envelope_digest(
        value: Stage1ReadyEnvelope | Mapping[str, Any],
        ) -> str:
    """Strictly revalidate any instance/mapping before trusting its digest."""

    if isinstance(value, Stage1ReadyEnvelope):
        serialized = value.model_dump(mode="json", warnings=False)
    elif isinstance(value, Mapping):
        serialized = dict(value)
    else:
        raise TypeError("envelope digest verifier는 instance 또는 mapping만 받습니다")
    envelope = Stage1ReadyEnvelope.model_validate(serialized, strict=True)
    if not envelope.verify_binding_digest():
        raise Stage1ReadyEnvelopeError("envelope binding digest verification failed")
    return envelope.binding_digest


def load_stage1_ready_envelope_json(
        payload: str | bytes | bytearray,
        ) -> Stage1ReadyEnvelope:
    return Stage1ReadyEnvelope.model_validate_json(payload, strict=True)


def compile_stage1_ready_envelope_v02_schema() -> dict[str, Any]:
    return Stage1ReadyEnvelope.model_json_schema(mode="validation")


def _schema_artifact_bytes() -> tuple[bytes, bytes, str]:
    schema_text = canonical_json(compile_stage1_ready_envelope_v02_schema())
    schema = schema_text.encode("utf-8")
    digest = sha256(schema).hexdigest()
    digest_file = f"{digest}  {SCHEMA_ARTIFACT.name}\n".encode("ascii")
    return schema, digest_file, digest


def write_stage1_ready_envelope_v02_schema_artifacts() -> str:
    schema, digest_file, digest = _schema_artifact_bytes()
    SCHEMA_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    for path, data in ((SCHEMA_ARTIFACT, schema),
                       (SCHEMA_DIGEST_ARTIFACT, digest_file)):
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(data)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return digest


def verify_stage1_ready_envelope_v02_schema_artifacts() -> str:
    schema, digest_file, digest = _schema_artifact_bytes()
    try:
        if SCHEMA_ARTIFACT.read_bytes() != schema:
            raise RuntimeError("Stage1ReadyEnvelope v0.3 schema artifact drift")
        if SCHEMA_DIGEST_ARTIFACT.read_bytes() != digest_file:
            raise RuntimeError("Stage1ReadyEnvelope v0.3 schema digest drift")
    except OSError as exc:
        raise RuntimeError("Stage1ReadyEnvelope v0.3 schema artifact missing") from exc
    return digest


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write", action="store_true")
    modes.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    digest = (
        write_stage1_ready_envelope_v02_schema_artifacts()
        if args.write else verify_stage1_ready_envelope_v02_schema_artifacts())
    print(f"PASS: {ENVELOPE_VERSION} schema_sha256={digest}")
    return 0


Stage1ReadyEnvelopeV02 = Stage1ReadyEnvelope
Stage1ReadyEnvelopeV03 = Stage1ReadyEnvelope
compile_stage1_ready_envelope_v03_schema = (
    compile_stage1_ready_envelope_v02_schema)
verify_stage1_ready_envelope_v03_schema_artifacts = (
    verify_stage1_ready_envelope_v02_schema_artifacts)
write_stage1_ready_envelope_v03_schema_artifacts = (
    write_stage1_ready_envelope_v02_schema_artifacts)

__all__ = [
    "ENVELOPE_VERSION",
    "SCHEMA_ARTIFACT",
    "SCHEMA_DIGEST_ARTIFACT",
    "Stage1ReadyEnvelope",
    "Stage1ReadyEnvelopeError",
    "Stage1ReadyEnvelopeV02",
    "Stage1ReadyEnvelopeV03",
    "canonical_json",
    "canonical_sha256",
    "compile_stage1_ready_envelope_v02_schema",
    "compile_stage1_ready_envelope_v03_schema",
    "load_stage1_ready_envelope_json",
    "verify_stage1_ready_envelope_digest",
    "verify_stage1_ready_envelope_v02_schema_artifacts",
    "verify_stage1_ready_envelope_v03_schema_artifacts",
    "write_stage1_ready_envelope_v02_schema_artifacts",
    "write_stage1_ready_envelope_v03_schema_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(_main())
