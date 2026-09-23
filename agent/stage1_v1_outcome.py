"""Unified four-state Stage1 v1 outcome and orchestration boundary.

Normal completion is exactly one of ``ready``, ``partial_ready``,
``needs_clarification`` or ``terminal``.  Resolver/compiler/schema/digest
failures stay exceptions and are never serialized as a successful outcome.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)

from .deterministic_plan_compiler_v1 import (
    DeterministicPlanCompilerError,
    compile_stage1_v1_generic,
)
from .semantic_intent_v1 import SemanticIntent
from .stage1_ready_envelope_v02 import Stage1ReadyEnvelope
from .stage1_v1_resolver import (
    ClarificationAuthority,
    ResolvedAuthority,
    Stage1ResolutionDecision,
    Stage1V1Resolver,
    TerminalAuthority,
)


STAGE1_OUTCOME_VERSION = "stage1-outcome/1.1"
SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_outcome_v1.schema.json")
SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_outcome_v1.schema.sha256")

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Stage1NormalStatus = Literal[
    "ready", "partial_ready", "needs_clarification", "terminal"
]


class Stage1V1CompilerTechnicalError(RuntimeError):
    """Compiler/envelope failure outside the four normal states."""


class Stage1V1OutcomeTechnicalError(RuntimeError):
    """Outcome binding failure outside the four normal states."""


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


def _strict_model(model: type[BaseModel], value: Any) -> BaseModel:
    if isinstance(value, model):
        return model.model_validate_json(
            canonical_json(value.model_dump(mode="json", warnings=False)),
            strict=True,
        )
    if isinstance(value, Mapping):
        return model.model_validate_json(canonical_json(value), strict=True)
    raise TypeError(f"{model.__name__} instance 또는 mapping이 필요합니다")


class Stage1Outcome(_StrictFrozenModel):
    """One digest-bound normal outcome; technical failure has no wire shape."""

    schema_version: Literal[STAGE1_OUTCOME_VERSION] = STAGE1_OUTCOME_VERSION
    status: Stage1NormalStatus
    resolution_decision: Stage1ResolutionDecision
    ready_envelope: Stage1ReadyEnvelope | None = None
    outcome_digest: Digest

    @field_validator("resolution_decision", mode="before")
    @classmethod
    def strict_decision(cls, value: Any) -> Stage1ResolutionDecision:
        return _strict_model(Stage1ResolutionDecision, value)  # type: ignore[return-value]

    @field_validator("ready_envelope", mode="before")
    @classmethod
    def strict_envelope(cls, value: Any) -> Stage1ReadyEnvelope | None:
        if value is None:
            return None
        return _strict_model(Stage1ReadyEnvelope, value)  # type: ignore[return-value]

    @classmethod
    def compute_digest(cls, payload: Mapping[str, Any]) -> str:
        body = dict(payload)
        body.pop("outcome_digest", None)
        return canonical_sha256(body)

    @classmethod
    def create(
            cls,
            *,
            resolution_decision: Stage1ResolutionDecision | Mapping[str, Any],
            ready_envelope: Stage1ReadyEnvelope | Mapping[str, Any] | None = None,
            status: Stage1NormalStatus | None = None,
            ) -> "Stage1Outcome":
        decision = _strict_model(
            Stage1ResolutionDecision, resolution_decision)
        envelope = (
            None if ready_envelope is None
            else _strict_model(Stage1ReadyEnvelope, ready_envelope)
        )
        authority = decision.authority  # type: ignore[attr-defined]
        if isinstance(authority, ResolvedAuthority):
            if envelope is None:
                raise ValueError("resolved decision에는 ReadyEnvelope가 필요합니다")
            derived_status: Stage1NormalStatus = envelope.status  # type: ignore[attr-defined]
        elif isinstance(authority, ClarificationAuthority):
            if envelope is not None:
                raise ValueError("clarification에는 ReadyEnvelope를 발행할 수 없습니다")
            derived_status = "needs_clarification"
        elif isinstance(authority, TerminalAuthority):
            if envelope is not None:
                raise ValueError("terminal에는 ReadyEnvelope를 발행할 수 없습니다")
            derived_status = "terminal"
        else:  # pragma: no cover - closed union defensive guard
            raise TypeError("알 수 없는 resolution authority입니다")
        if status is not None and status != derived_status:
            raise ValueError("caller status가 authority/envelope 파생값과 다릅니다")
        body: dict[str, Any] = {
            "schema_version": STAGE1_OUTCOME_VERSION,
            "status": derived_status,
            "resolution_decision": decision,
            "ready_envelope": envelope,
        }
        digest_body = cls.model_construct(
            **body, outcome_digest="0" * 64).model_dump(
                mode="json", warnings=False)
        body["outcome_digest"] = cls.compute_digest(digest_body)
        return cls.model_validate(body, strict=True)

    @model_validator(mode="after")
    def validate_outcome(self) -> "Stage1Outcome":
        decision = self.resolution_decision
        authority = decision.authority
        envelope = self.ready_envelope

        if isinstance(authority, ResolvedAuthority):
            if envelope is None or self.status not in {"ready", "partial_ready"}:
                raise ValueError("resolved outcome status/envelope 조합이 잘못되었습니다")
            if self.status != envelope.status:
                raise ValueError("outcome status가 ReadyEnvelope와 다릅니다")
            if not envelope.execution_plan.answer_roots:
                raise ValueError("zero live answer roots에는 ReadyEnvelope가 금지됩니다")
            resolution = authority.resolution
            if (
                envelope.source_intent_digest != decision.source_intent_digest
                or envelope.source_intent.model_dump(mode="json", warnings=False)
                != decision.source_intent.model_dump(mode="json", warnings=False)
                or envelope.execution_plan.resolution_digest
                != resolution.resolution_digest
                or envelope.canonical_build_id != decision.canonical_build_id
                or envelope.resolver_version != decision.resolver_version
            ):
                raise ValueError("ready outcome resolver/plan/envelope binding이 다릅니다")
        elif isinstance(authority, ClarificationAuthority):
            if self.status != "needs_clarification" or envelope is not None:
                raise ValueError("clarification outcome은 plan/envelope를 가질 수 없습니다")
        elif isinstance(authority, TerminalAuthority):
            if self.status != "terminal" or envelope is not None:
                raise ValueError("terminal outcome은 plan/envelope를 가질 수 없습니다")
        else:  # pragma: no cover - closed union defensive guard
            raise ValueError("알 수 없는 resolution authority입니다")

        if self.outcome_digest != self.compute_digest(
                self.model_dump(mode="json", warnings=False)):
            raise ValueError("Stage1 outcome digest가 일치하지 않습니다")
        return self

    @property
    def stage2_allowed(self) -> bool:
        return self.status in {"ready", "partial_ready"}


def _log_compiler_rejection(question: str, exc: BaseException) -> None:
    """컴파일러가 왜 결속하지 못했는지를 개발용 파일에만 남긴다.

    응답의 진단(`safe_exception_diagnostics`)은 예외 메시지를 일부러 뺀다.
    그래서 「해석하지 못했습니다」만 보고는 어느 topology 가 어긋났는지 알 수
    없고, RPC-001·RPC-005 를 진단할 때 서버 안에서 따로 재현해야 했다.

    컴파일러 메시지는 우리가 쓴 topology 문구이고 사용자 데이터가 아니다.
    그래도 응답으로 내보내지 않고, HCX_REJECT_LOG 와 같은 방식으로 파일에만
    적는다. 경로를 비우면 아무것도 남기지 않는다. 기록 실패는 삼킨다 —
    진단이 파이프라인을 멈추면 안 된다.
    """

    path = os.getenv("STAGE1_COMPILER_REJECT_LOG", "")
    if not path:
        return
    chain, current = [], exc
    while current is not None and len(chain) < 4:
        chain.append(f"{type(current).__name__}: {current}")
        nxt = current.__cause__ or current.__context__
        current = nxt if isinstance(nxt, BaseException) else None
    try:
        with open(path, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(
                {"question": question, "chain": chain},
                ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - 진단 기록은 실패해도 된다
        pass


class Stage1V1Orchestrator:
    """Resolve, compile and close exactly one normal Stage1 v1 outcome."""

    def __init__(self, resolver: Stage1V1Resolver) -> None:
        self.resolver = resolver

    def run(
            self,
            *,
            question_id: str,
            question: str,
            source_intent: SemanticIntent | Mapping[str, Any],
            ) -> Stage1Outcome:
        decision = self.resolver.resolve(
            question_id=question_id,
            question=question,
            source_intent=source_intent,
        )
        if not isinstance(decision.authority, ResolvedAuthority):
            try:
                return Stage1Outcome.create(resolution_decision=decision)
            except Exception as exc:
                raise Stage1V1OutcomeTechnicalError(
                    "non-ready Stage1 outcome 결속에 실패했습니다") from exc

        try:
            compiled = compile_stage1_v1_generic(
                question,
                decision.source_intent,
                decision.authority.resolution,
            )
            envelope = Stage1ReadyEnvelope.create(
                source_intent_digest=compiled.source_intent_digest,
                execution_plan_digest=(
                    compiled.execution_plan.execution_plan_digest),
                canonical_build_id=decision.canonical_build_id,
                resolver_version=decision.resolver_version,
                source_intent=compiled.intent,
                execution_plan=compiled.execution_plan,
                answer_contract=compiled.answer_contract,
            )
        except (DeterministicPlanCompilerError, TypeError, ValueError) as exc:
            _log_compiler_rejection(question, exc)
            raise Stage1V1CompilerTechnicalError(
                "Stage1 v1 compiler/envelope 결속에 실패했습니다") from exc

        try:
            return Stage1Outcome.create(
                resolution_decision=decision,
                ready_envelope=envelope,
            )
        except Exception as exc:
            raise Stage1V1OutcomeTechnicalError(
                "ready Stage1 outcome 결속에 실패했습니다") from exc


def verify_stage1_outcome_digest(
        value: Stage1Outcome | Mapping[str, Any],
        ) -> str:
    validated = _strict_model(Stage1Outcome, value)
    return validated.outcome_digest  # type: ignore[attr-defined]


def load_stage1_outcome_json(
        payload: str | bytes | bytearray,
        ) -> Stage1Outcome:
    return Stage1Outcome.model_validate_json(payload, strict=True)


def _schema_artifact_bytes() -> tuple[bytes, bytes, str]:
    schema = canonical_json(
        Stage1Outcome.model_json_schema(mode="validation")).encode("utf-8")
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


def write_stage1_outcome_schema_artifacts() -> str:
    schema, sidecar, digest = _schema_artifact_bytes()
    _atomic_write(SCHEMA_ARTIFACT, schema)
    _atomic_write(SCHEMA_DIGEST_ARTIFACT, sidecar)
    return digest


def verify_stage1_outcome_schema_artifacts() -> str:
    schema, sidecar, digest = _schema_artifact_bytes()
    try:
        if SCHEMA_ARTIFACT.read_bytes() != schema:
            raise RuntimeError("Stage1 outcome schema artifact drift")
        if SCHEMA_DIGEST_ARTIFACT.read_bytes() != sidecar:
            raise RuntimeError("Stage1 outcome schema digest drift")
    except OSError as exc:
        raise RuntimeError("Stage1 outcome schema artifact missing") from exc
    return digest


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    digest = (
        write_stage1_outcome_schema_artifacts()
        if args.write else verify_stage1_outcome_schema_artifacts())
    print(f"PASS: {STAGE1_OUTCOME_VERSION} schema_sha256={digest}")
    return 0


__all__ = [
    "SCHEMA_ARTIFACT",
    "SCHEMA_DIGEST_ARTIFACT",
    "STAGE1_OUTCOME_VERSION",
    "Stage1NormalStatus",
    "Stage1Outcome",
    "Stage1V1CompilerTechnicalError",
    "Stage1V1Orchestrator",
    "Stage1V1OutcomeTechnicalError",
    "canonical_json",
    "canonical_sha256",
    "load_stage1_outcome_json",
    "verify_stage1_outcome_digest",
    "verify_stage1_outcome_schema_artifacts",
    "write_stage1_outcome_schema_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(_main())
