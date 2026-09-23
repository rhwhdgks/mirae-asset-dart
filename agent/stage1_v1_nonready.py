"""Strict, source-bound v1 decisions for the two non-ready slices.

This module is deliberately small.  It owns the question-semantic source,
the public clarification/terminal decision, and the proofs that tie both to
the frozen question fixture.  A decision is never an executable answer: the
only values it can publish are a typed clarification slot or typed policy
reasons.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Mapping, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    StrictInt,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated

from .semantic_intent_v1 import (
    HCX_SEMANTIC_INTENT_WIRE_V1,
    HcxSemanticIntentWire,
    SemanticIntent,
    normalize_semantic_intent,
    semantic_intent_digest,
)


NONREADY_SCHEMA_VERSION = "stage1-v1-nonready/1.1"
SOURCE_ROOT = "fixtures/query_plan_v04"
SOURCE_SHA256SUMS_FILENAME = "SHA256SUMS"
SOURCE_FIXTURE_FILENAME = "questions_v0.4.jsonl"
SOURCE_FIXTURE_PATH = f"{SOURCE_ROOT}/{SOURCE_FIXTURE_FILENAME}"
FIXTURE_ROOT = "fixtures/stage1_v1_nonready"

R_A_001 = "R-A-001"
G_U_002 = "G-U-002"
R_A_001_QUESTION = "삼성전자가 작년에 얼마나 벌었어?"
G_U_002_QUESTION = "삼성전자의 2027년 예상 매출액과 매수 의견을 알려줘."

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_ROOT = PROJECT_ROOT / FIXTURE_ROOT

SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_nonready.schema.json")
SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_nonready.schema.sha256")
SOURCE_PROOF_SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_nonready_source_proof.schema.json")
SOURCE_PROOF_SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_nonready_source_proof.schema.sha256")
CLARIFICATION_SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_nonready_clarification.schema.json")
CLARIFICATION_SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_nonready_clarification.schema.sha256")
TERMINAL_REASON_SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_nonready_terminal_reason.schema.json")
TERMINAL_REASON_SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_nonready_terminal_reason.schema.sha256")

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
QuestionId: TypeAlias = Literal[R_A_001, G_U_002]
MetricChoice: TypeAlias = Literal[
    "revenue", "operating_income", "net_income"
]
PolicyReason: TypeAlias = Literal["future_forecast", "investment_advice"]

# This is intentionally copied from the frozen source checksum declaration.
# The loader checks the declaration and all five source files; changing a
# source file and changing its local checksum line cannot move this boundary.
_FROZEN_SOURCE_FIXTURES: dict[str, str] = {
    "questions_v0.4.jsonl":
        "5d5ffec7e0047a7a5412bb24f3b6d090bcc6e669768b49ca8e40fcfd2909396f",
    "plan_proposals_v0.4.jsonl":
        "b248f6ee0e1496479fb985796cf47bbfccf13f215923455884a4e88bd59e0369",
    "query_plan_handoffs_v0.4.jsonl":
        "43b8b385c160ee751a9b94771866f4ca0ec1a1c150915212b58cc57b951c960d",
    "answer_requirements_v0.4.jsonl":
        "c44654bf262f603bf93503b2dcbcb8365c5365e86b8011c9734d1ea976a2f4dc",
    "migration_report_v0.4.json":
        "37de0132f8b5a5842ae2287061b9d1bc57214b7c488f8f7cba0d8cba4b86301a",
}
FROZEN_SOURCE_FIXTURES: Mapping[str, str] = MappingProxyType(
    _FROZEN_SOURCE_FIXTURES)

_QUESTIONS: Mapping[str, str] = MappingProxyType({
    R_A_001: R_A_001_QUESTION,
    G_U_002: G_U_002_QUESTION,
})
_QUESTION_RECORD_DIGESTS: Mapping[str, str] = MappingProxyType({
    R_A_001:
        "0bd0d37b2d77ade987f3621444a10ebae62c5276179a73c3a91ec08be721eb62",
    G_U_002:
        "416cea159ee583b893f314cec50fda55d59a5ab58221a5d59693421759b4c759",
})


class NonReadyError(ValueError):
    """Base error for malformed or unbound non-ready artifacts."""


class NonReadySourceError(NonReadyError):
    """Raised when the frozen question source cannot be proved."""


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


def _sha256_bytes(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _strict_model(model: type[BaseModel], value: Any) -> BaseModel:
    """Re-enter a nested boundary even if a caller used model_copy(update=...)."""

    if isinstance(value, model):
        return model.model_validate(
            value.model_dump(mode="python", warnings=False), strict=True)
    if isinstance(value, Mapping):
        return model.model_validate(value, strict=True)
    return model.model_validate(value, strict=True)


def _json_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object members instead of accepting last-key-wins."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NonReadyError(f"JSON object에 중복 key가 있습니다: {key}")
        result[key] = value
    return result


def _unique(values: list[Any], *, label: str) -> None:
    encoded = [canonical_json(value) for value in values]
    if len(encoded) != len(set(encoded)):
        raise ValueError(f"{label}에는 중복 값이 있을 수 없습니다")


def _expected_question(question_id: str) -> str:
    try:
        return _QUESTIONS[question_id]
    except KeyError as exc:
        raise NonReadyError(f"지원하지 않는 non-ready question_id: {question_id}") from exc


def _wire_r_a_001() -> HcxSemanticIntentWire:
    return HcxSemanticIntentWire.model_validate({
        "schema_version": HCX_SEMANTIC_INTENT_WIRE_V1,
        "entities": [{"kind_hint": "company", "surface": "삼성전자"}],
        "answer_items": [{
            "target": {
                "kind": "metric", "surface": "벌었어",
                "entity_indexes": [0], "qualifier_surfaces": [],
            },
            "operation": "retrieve",
            "scope": {
                "target_period_expressions": ["작년"],
                "as_of_expression": "",
                "document_group_expression": "",
                "scope_qualifier_expressions": [],
            },
            "selection": {
                "mode": "none", "criterion_surface": "", "k": 0,
            },
            "output": {
                "shape": "scalar", "projection_mode": "named_fields",
                "field_surfaces": ["얼마나"],
                "presentation": "auto",
            },
        }],
        "answer_groups": [],
        "premises": [],
        "unresolved_mentions": [{
            "raw_text": "벌었어", "role_hint": "target",
            "applies_to_item_indexes": [0],
        }],
        "presentation": "auto",
    }, strict=True)


def _wire_g_u_002() -> HcxSemanticIntentWire:
    return HcxSemanticIntentWire.model_validate({
        "schema_version": HCX_SEMANTIC_INTENT_WIRE_V1,
        "entities": [{"kind_hint": "company", "surface": "삼성전자"}],
        "answer_items": [
            {
                "target": {
                    "kind": "metric", "surface": "예상 매출액",
                    "entity_indexes": [0], "qualifier_surfaces": [],
                },
                "operation": "retrieve",
                "scope": {
                    "target_period_expressions": ["2027년"],
                    "as_of_expression": "",
                    "document_group_expression": "",
                    "scope_qualifier_expressions": [],
                },
                "selection": {
                    "mode": "none", "criterion_surface": "", "k": 0,
                },
                "output": {
                    "shape": "scalar", "projection_mode": "named_fields",
                    "field_surfaces": ["예상 매출액"],
                    "presentation": "auto",
                },
            },
            {
                "target": {
                    "kind": "attribute", "surface": "매수 의견",
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
                    "field_surfaces": ["매수 의견"],
                    "presentation": "auto",
                },
            },
        ],
        "answer_groups": [],
        "premises": [],
        "unresolved_mentions": [],
        "presentation": "auto",
    }, strict=True)


def normalize_nonready_intent(
        question_id: QuestionId,
        provider: HcxSemanticIntentWire | Mapping[str, Any] | None = None,
        ) -> SemanticIntent:
    """Normalize one exact non-ready question into the v1 semantic model."""

    question = _expected_question(question_id)
    if provider is None:
        provider = _wire_r_a_001() if question_id == R_A_001 else _wire_g_u_002()
    return normalize_semantic_intent(question, provider)


class SourceFixtureProof(_StrictFrozenModel):
    """Proof that a decision was made from the immutable question row."""

    source_path: Literal[SOURCE_FIXTURE_PATH] = SOURCE_FIXTURE_PATH
    source_filename: Literal[SOURCE_FIXTURE_FILENAME] = SOURCE_FIXTURE_FILENAME
    fixture_sha256: Digest
    question_id: QuestionId
    question_sha256: Digest
    record_sha256: Digest
    source_line: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def validate_proof(self) -> "SourceFixtureProof":
        question = _expected_question(self.question_id)
        if self.fixture_sha256 != FROZEN_SOURCE_FIXTURES[SOURCE_FIXTURE_FILENAME]:
            raise ValueError("source fixture digest가 frozen baseline과 다릅니다")
        if self.question_sha256 != _sha256_bytes(question):
            raise ValueError("question_sha256가 원 질문과 다릅니다")
        if self.record_sha256 != _QUESTION_RECORD_DIGESTS[self.question_id]:
            raise ValueError("source record digest가 frozen row와 다릅니다")
        expected_line = 51 if self.question_id == R_A_001 else 26
        if self.source_line != expected_line:
            raise ValueError("source_line이 frozen question fixture와 다릅니다")
        return self

    @classmethod
    def for_question(cls, question_id: QuestionId) -> "SourceFixtureProof":
        question = _expected_question(question_id)
        return cls(
            fixture_sha256=FROZEN_SOURCE_FIXTURES[SOURCE_FIXTURE_FILENAME],
            question_id=question_id,
            question_sha256=_sha256_bytes(question),
            record_sha256=_QUESTION_RECORD_DIGESTS[question_id],
            source_line=51 if question_id == R_A_001 else 26,
        )


class ClarificationSlot(_StrictFrozenModel):
    """The only public choice exposed for the ambiguous metric."""

    slot_id: Literal["slot-1"] = "slot-1"
    target: Literal["metric"] = "metric"
    allowed_values: list[MetricChoice] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_slot(self) -> "ClarificationSlot":
        expected = ["revenue", "operating_income", "net_income"]
        if self.allowed_values != expected:
            raise ValueError("R-A-001 metric choices가 frozen authority와 다릅니다")
        return self

    @property
    def choices(self) -> tuple[str, ...]:
        """Compatibility spelling for callers that call values choices."""

        return tuple(self.allowed_values)


class ClarificationDecision(_StrictFrozenModel):
    reason: Literal["ambiguity_requires_user"]
    slots: list[ClarificationSlot] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision(self) -> "ClarificationDecision":
        if len(self.slots) != 1 or self.slots[0].target != "metric":
            raise ValueError("clarification은 metric slot 하나만 발행해야 합니다")
        return self

    @property
    def slot(self) -> ClarificationSlot:
        return self.slots[0]


class TerminalReasonBinding(_StrictFrozenModel):
    reason: PolicyReason
    item_id: Literal["item-1", "item-2"]


class NonReadyDecision(_StrictFrozenModel):
    """Source-bound terminal or typed-clarification decision."""

    schema_version: Literal[NONREADY_SCHEMA_VERSION] = NONREADY_SCHEMA_VERSION
    question_id: QuestionId
    question: str = Field(min_length=1)
    question_sha256: Digest
    source_fixture: SourceFixtureProof
    source_intent: SemanticIntent
    source_intent_digest: Digest
    disposition: Literal["needs_clarification", "terminal"]
    clarification: ClarificationDecision | None = None
    terminal_reasons: list[TerminalReasonBinding] = Field(default_factory=list)
    decision_digest: Digest

    _QUESTION_IDS: ClassVar[tuple[str, ...]] = (R_A_001, G_U_002)

    @field_validator("source_fixture", mode="before")
    @classmethod
    def _strict_source_proof(cls, value: Any) -> SourceFixtureProof:
        return _strict_model(SourceFixtureProof, value)  # type: ignore[return-value]

    @field_validator("source_intent", mode="before")
    @classmethod
    def _strict_source_intent(cls, value: Any) -> SemanticIntent:
        return _strict_model(SemanticIntent, value)  # type: ignore[return-value]

    @classmethod
    def compute_digest(cls, payload: Mapping[str, Any]) -> str:
        body = dict(payload)
        body.pop("decision_digest", None)
        return canonical_sha256(body)

    @classmethod
    def create(
            cls,
            *,
            question_id: QuestionId,
            question: str | None = None,
            source_fixture: SourceFixtureProof | Mapping[str, Any] | None = None,
            source_intent: SemanticIntent | Mapping[str, Any] | None = None,
            source_intent_digest: str | None = None,
            question_sha256: str | None = None,
            disposition: Literal["needs_clarification", "terminal"] | None = None,
            clarification: ClarificationDecision | Mapping[str, Any] | None = None,
            terminal_reasons: list[TerminalReasonBinding | Mapping[str, Any]] | None = None,
            decision_digest: str | None = None,
            ) -> "NonReadyDecision":
        expected_question = _expected_question(question_id)
        intent = (
            normalize_nonready_intent(question_id)
            if source_intent is None else source_intent
        )
        intent = _strict_model(SemanticIntent, intent)
        proof = (
            SourceFixtureProof.for_question(question_id)
            if source_fixture is None else source_fixture
        )
        proof = _strict_model(SourceFixtureProof, proof)
        if disposition is None:
            disposition = (
                "needs_clarification" if question_id == R_A_001 else "terminal"
            )
        body: dict[str, Any] = {
            "schema_version": NONREADY_SCHEMA_VERSION,
            "question_id": question_id,
            "question": expected_question if question is None else question,
            "question_sha256": (
                _sha256_bytes(expected_question)
                if question_sha256 is None else question_sha256
            ),
            "source_fixture": proof,
            "source_intent": intent,
            "source_intent_digest": (
                semantic_intent_digest(intent)
                if source_intent_digest is None else source_intent_digest
            ),
            "disposition": disposition,
            "clarification": clarification,
            "terminal_reasons": (
                [] if terminal_reasons is None else terminal_reasons
            ),
        }
        if clarification is None and question_id == R_A_001:
            body["clarification"] = ClarificationDecision(
                reason="ambiguity_requires_user",
                slots=[ClarificationSlot(
                    allowed_values=["revenue", "operating_income", "net_income"],
                )],
            )
        if terminal_reasons is None and question_id == G_U_002:
            body["terminal_reasons"] = [
                TerminalReasonBinding(reason="future_forecast", item_id="item-1"),
                TerminalReasonBinding(reason="investment_advice", item_id="item-2"),
            ]
        # Build the digest over JSON-native nested values, not Python model repr.
        digest_body = cls.model_construct(**body).model_dump(
            mode="json", warnings=False)
        body["decision_digest"] = (
            cls.compute_digest(digest_body)
            if decision_digest is None else decision_digest
        )
        return cls.model_validate(body, strict=True)

    @model_validator(mode="after")
    def validate_decision(self) -> "NonReadyDecision":
        expected_question = _expected_question(self.question_id)
        if self.question != expected_question:
            raise ValueError("question이 frozen source와 다릅니다")
        if self.question_sha256 != _sha256_bytes(self.question):
            raise ValueError("question_sha256가 question과 다릅니다")
        proof = self.source_fixture
        if proof.question_id != self.question_id:
            raise ValueError("source fixture proof question_id가 다릅니다")
        if proof.question_sha256 != self.question_sha256:
            raise ValueError("source fixture proof와 question hash가 다릅니다")
        expected_intent = normalize_nonready_intent(self.question_id)
        if self.source_intent.model_dump(mode="json", warnings=False) != (
                expected_intent.model_dump(mode="json", warnings=False)):
            raise ValueError("source_intent가 exact question semantics와 다릅니다")
        if self.source_intent_digest != semantic_intent_digest(self.source_intent):
            raise ValueError("source_intent_digest가 embedded source_intent와 다릅니다")

        if self.question_id == R_A_001:
            if self.disposition != "needs_clarification":
                raise ValueError("R-A-001은 needs_clarification이어야 합니다")
            if self.clarification is None:
                raise ValueError("R-A-001 clarification이 없습니다")
            if self.clarification.reason != "ambiguity_requires_user":
                raise ValueError("R-A-001 clarification reason이 다릅니다")
            if self.terminal_reasons:
                raise ValueError("R-A-001에는 terminal reason이 없어야 합니다")
        else:
            if self.disposition != "terminal":
                raise ValueError("G-U-002는 terminal이어야 합니다")
            if self.clarification is not None:
                raise ValueError("G-U-002에는 clarification이 없어야 합니다")
            expected_reasons = [
                {"reason": "future_forecast", "item_id": "item-1"},
                {"reason": "investment_advice", "item_id": "item-2"},
            ]
            if [row.model_dump(mode="json") for row in self.terminal_reasons] != expected_reasons:
                raise ValueError("G-U-002 terminal reason-to-item binding이 다릅니다")

        _unique([row.item_id for row in self.terminal_reasons], label="terminal item_id")
        if self.decision_digest != self.compute_digest(
                self.model_dump(mode="json", warnings=False)):
            raise ValueError("decision_digest가 일치하지 않습니다")
        return self

    @property
    def status(self) -> str:
        return self.disposition

    @property
    def final_disposition(self) -> str:
        return self.disposition

    @property
    def question_hash(self) -> str:
        return self.question_sha256

    @property
    def source_fixture_sha256(self) -> str:
        return self.source_fixture.fixture_sha256

    @property
    def source_intent_hash(self) -> str:
        return self.source_intent_digest

    @property
    def decision_sha256(self) -> str:
        return self.decision_digest

    @property
    def reason_codes(self) -> tuple[str, ...]:
        if self.clarification is not None:
            return (self.clarification.reason,)
        return tuple(row.reason for row in self.terminal_reasons)

    @classmethod
    def load(
            cls,
            path: str | Path,
            *,
            source_root: str | Path | None = None,
            ) -> "NonReadyDecision":
        fixture_path = _secure_regular_file(Path(path), anchor=Path(path).parent,
                                            label="non-ready fixture")
        try:
            payload = json.loads(
                fixture_path.read_text(encoding="utf-8"),
                object_pairs_hook=_json_no_duplicate_keys,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise NonReadyError(f"non-ready fixture 파싱 실패: {fixture_path}") from exc
        if not isinstance(payload, dict):
            raise NonReadyError("non-ready fixture top-level은 object여야 합니다")
        try:
            result = cls.model_validate(payload, strict=True)
        except ValueError as exc:
            raise NonReadyError("non-ready fixture strict validation 실패") from exc
        _verify_source_binding(result, source_root=source_root)
        return result


@dataclass(frozen=True, slots=True)
class LoadedNonReady:
    decision: NonReadyDecision
    path: Path

    @property
    def canonical_digest(self) -> str:
        return self.decision.decision_digest


def expected_nonready_decision(question_id: QuestionId) -> NonReadyDecision:
    return NonReadyDecision.create(question_id=question_id)


def expected_r_a_001() -> NonReadyDecision:
    return expected_nonready_decision(R_A_001)


def expected_g_u_002() -> NonReadyDecision:
    return expected_nonready_decision(G_U_002)


def verify_nonready_digest(value: NonReadyDecision | Mapping[str, Any]) -> str:
    if isinstance(value, NonReadyDecision):
        validated = NonReadyDecision.model_validate(
            value.model_dump(mode="python", warnings=False), strict=True)
    elif isinstance(value, Mapping):
        validated = NonReadyDecision.model_validate(value, strict=True)
    else:
        raise TypeError("value는 NonReadyDecision 또는 mapping이어야 합니다")
    return validated.decision_digest


def load_nonready_fixture(
        path: str | Path,
        *,
        source_root: str | Path | None = None,
        ) -> NonReadyDecision:
    return NonReadyDecision.load(path, source_root=source_root)


def _read_source_sha256sums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise NonReadySourceError(f"source SHA256SUMS를 읽을 수 없습니다: {path}") from exc
    parsed: dict[str, str] = {}
    for line_no, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            raise NonReadySourceError(f"source SHA256SUMS 형식 오류: {path}:{line_no}")
        if parts[1] in parsed:
            raise NonReadySourceError(f"source SHA256SUMS 중복 파일: {parts[1]}")
        parsed[parts[1]] = parts[0]
    if parsed != dict(FROZEN_SOURCE_FIXTURES):
        raise NonReadySourceError("source SHA256SUMS가 frozen baseline과 다릅니다")
    return parsed


def _file_sha256(path: Path) -> str:
    try:
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise NonReadySourceError(f"source fixture를 읽을 수 없습니다: {path}") from exc


def _secure_regular_file(path: Path, *, anchor: Path, label: str) -> Path:
    raw = Path(os.path.abspath(os.fspath(path)))
    anchor_abs = Path(os.path.abspath(os.fspath(anchor)))
    try:
        raw.relative_to(anchor_abs)
    except ValueError as exc:
        raise NonReadyError(f"{label}가 anchor 밖입니다: {path}") from exc
    current = anchor_abs
    try:
        for component in raw.relative_to(anchor_abs).parts:
            current /= component
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise NonReadyError(f"{label} symlink component는 허용되지 않습니다: {current}")
        info = raw.lstat()
    except OSError as exc:
        raise NonReadyError(f"{label}를 읽을 수 없습니다: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise NonReadyError(f"{label}는 regular file이어야 합니다: {path}")
    return raw


def _secure_source_root(source_root: str | Path | None) -> Path:
    raw = Path(source_root) if source_root is not None else PROJECT_ROOT / SOURCE_ROOT
    raw = Path(os.path.abspath(os.fspath(raw)))
    try:
        info = raw.lstat()
    except OSError as exc:
        raise NonReadySourceError(f"source root를 읽을 수 없습니다: {raw}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise NonReadySourceError("source root는 symlink가 아닌 directory여야 합니다")
    # Reject a linked parent as well; the source proof must be local to the
    # supplied replay root.
    current = Path(raw.anchor)
    try:
        for component in raw.parts[1:]:
            current /= component
            if stat.S_ISLNK(current.lstat().st_mode):
                raise NonReadySourceError("source root ancestor symlink는 허용되지 않습니다")
    except OSError as exc:
        raise NonReadySourceError("source root ancestor를 읽을 수 없습니다") from exc
    return raw


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise NonReadySourceError(f"source JSONL을 읽을 수 없습니다: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            raise NonReadySourceError(f"source JSONL 빈 줄: {path}:{line_no}")
        try:
            value = json.loads(line, object_pairs_hook=_json_no_duplicate_keys)
        except (json.JSONDecodeError, ValueError) as exc:
            raise NonReadySourceError(f"source JSONL 파싱 실패: {path}:{line_no}") from exc
        if not isinstance(value, dict):
            raise NonReadySourceError(f"source JSONL row가 object가 아닙니다: {path}:{line_no}")
        rows.append(value)
    return rows


def _verify_source_binding(
        decision: NonReadyDecision,
        *,
        source_root: str | Path | None,
        ) -> None:
    root = _secure_source_root(source_root)
    checksum = _secure_regular_file(
        root / SOURCE_SHA256SUMS_FILENAME, anchor=root,
        label="source SHA256SUMS")
    declared = _read_source_sha256sums(checksum)
    for filename, expected in FROZEN_SOURCE_FIXTURES.items():
        source = _secure_regular_file(root / filename, anchor=root,
                                      label=f"source fixture {filename}")
        actual = _file_sha256(source)
        if actual != expected or declared[filename] != actual:
            raise NonReadySourceError(f"source fixture digest drift: {filename}")

    question_path = _secure_regular_file(
        root / SOURCE_FIXTURE_FILENAME, anchor=root,
        label="question source fixture")
    rows = _read_jsonl(question_path)
    matches: list[tuple[int, dict[str, Any]]] = [
        (line_no, row)
        for line_no, row in enumerate(rows, start=1)
        if row.get("question_id") == decision.question_id
    ]
    if len(matches) != 1:
        raise NonReadySourceError(
            f"{decision.question_id} source row가 정확히 1건이어야 합니다")
    line_no, row = matches[0]
    if row.get("question") != decision.question:
        raise NonReadySourceError("source question이 decision과 다릅니다")
    if _sha256_bytes(canonical_json(row)) != decision.source_fixture.record_sha256:
        raise NonReadySourceError("source record digest proof가 다릅니다")
    if line_no != decision.source_fixture.source_line:
        raise NonReadySourceError("source line proof가 다릅니다")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _schema_bytes(model: type[BaseModel], artifact: Path) -> tuple[bytes, bytes, str]:
    data = canonical_json(model.model_json_schema(mode="validation")).encode("utf-8")
    digest = sha256(data).hexdigest()
    sidecar = f"{digest}  {artifact.name}\n".encode("ascii")
    return data, sidecar, digest


def _schema_artifacts() -> tuple[tuple[Path, bytes], ...]:
    pairs = (
        (NonReadyDecision, SCHEMA_ARTIFACT, SCHEMA_DIGEST_ARTIFACT),
        (SourceFixtureProof, SOURCE_PROOF_SCHEMA_ARTIFACT,
         SOURCE_PROOF_SCHEMA_DIGEST_ARTIFACT),
        (ClarificationDecision, CLARIFICATION_SCHEMA_ARTIFACT,
         CLARIFICATION_SCHEMA_DIGEST_ARTIFACT),
        (TerminalReasonBinding, TERMINAL_REASON_SCHEMA_ARTIFACT,
         TERMINAL_REASON_SCHEMA_DIGEST_ARTIFACT),
    )
    output: list[tuple[Path, bytes]] = []
    for model, artifact, digest_artifact in pairs:
        schema, digest, _ = _schema_bytes(model, artifact)
        output.extend(((artifact, schema), (digest_artifact, digest)))
    return tuple(output)


def write_schema_artifacts() -> tuple[str, ...]:
    artifacts = _schema_artifacts()
    for path, payload in artifacts:
        _atomic_write(path, payload)
    return tuple(_schema_bytes(model, artifact)[2] for model, artifact, _ in (
        (NonReadyDecision, SCHEMA_ARTIFACT, SCHEMA_DIGEST_ARTIFACT),
        (SourceFixtureProof, SOURCE_PROOF_SCHEMA_ARTIFACT,
         SOURCE_PROOF_SCHEMA_DIGEST_ARTIFACT),
        (ClarificationDecision, CLARIFICATION_SCHEMA_ARTIFACT,
         CLARIFICATION_SCHEMA_DIGEST_ARTIFACT),
        (TerminalReasonBinding, TERMINAL_REASON_SCHEMA_ARTIFACT,
         TERMINAL_REASON_SCHEMA_DIGEST_ARTIFACT),
    ))


def verify_schema_artifacts() -> tuple[str, ...]:
    expected = _schema_artifacts()
    for path, payload in expected:
        try:
            actual = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"non-ready schema artifact가 없습니다: {path}") from exc
        if actual != payload:
            raise RuntimeError(f"non-ready schema artifact drift: {path}")
    return tuple(_schema_bytes(model, artifact)[2] for model, artifact, _ in (
        (NonReadyDecision, SCHEMA_ARTIFACT, SCHEMA_DIGEST_ARTIFACT),
        (SourceFixtureProof, SOURCE_PROOF_SCHEMA_ARTIFACT,
         SOURCE_PROOF_SCHEMA_DIGEST_ARTIFACT),
        (ClarificationDecision, CLARIFICATION_SCHEMA_ARTIFACT,
         CLARIFICATION_SCHEMA_DIGEST_ARTIFACT),
        (TerminalReasonBinding, TERMINAL_REASON_SCHEMA_ARTIFACT,
         TERMINAL_REASON_SCHEMA_DIGEST_ARTIFACT),
    ))


def write_fixture_artifacts(
        fixture_root: str | Path = DEFAULT_FIXTURE_ROOT,
        ) -> Mapping[str, str]:
    root = Path(fixture_root)
    decisions = {
        "r_a_001.json": expected_r_a_001(),
        "g_u_002.json": expected_g_u_002(),
    }
    result: dict[str, str] = {}
    for filename, decision in decisions.items():
        payload = canonical_json(decision.model_dump(mode="json", warnings=False))
        path = root / filename
        _atomic_write(path, (payload + "\n").encode("utf-8"))
        result[filename] = decision.decision_digest
    return MappingProxyType(result)


def verify_fixture_artifacts(
        fixture_root: str | Path = DEFAULT_FIXTURE_ROOT,
        *,
        source_root: str | Path | None = None,
        ) -> Mapping[str, str]:
    root = Path(fixture_root)
    result: dict[str, str] = {}
    for filename, question_id in (
            ("r_a_001.json", R_A_001), ("g_u_002.json", G_U_002)):
        path = root / filename
        decision = NonReadyDecision.load(path, source_root=source_root)
        if decision.question_id != question_id:
            raise RuntimeError(f"fixture question_id 불일치: {path}")
        result[filename] = decision.decision_digest
    return MappingProxyType(result)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        fixture_digests = write_fixture_artifacts()
        schema_digests = write_schema_artifacts()
    else:
        fixture_digests = verify_fixture_artifacts()
        schema_digests = verify_schema_artifacts()
    print(
        f"PASS: {NONREADY_SCHEMA_VERSION} "
        f"r_a_001={fixture_digests['r_a_001.json']} "
        f"g_u_002={fixture_digests['g_u_002.json']} "
        f"schema_sha256={','.join(schema_digests)}"
    )
    return 0


# Short aliases keep the public API discoverable without changing the wire.
Stage1V1NonReadyDecision = NonReadyDecision
Stage1V1NonReadySourceProof = SourceFixtureProof

__all__ = [
    "CLARIFICATION_SCHEMA_ARTIFACT",
    "CLARIFICATION_SCHEMA_DIGEST_ARTIFACT",
    "Digest",
    "FIXTURE_ROOT",
    "FROZEN_SOURCE_FIXTURES",
    "G_U_002",
    "G_U_002_QUESTION",
    "LoadedNonReady",
    "NONREADY_SCHEMA_VERSION",
    "NonReadyDecision",
    "NonReadyError",
    "NonReadySourceError",
    "R_A_001",
    "R_A_001_QUESTION",
    "SCHEMA_ARTIFACT",
    "SCHEMA_DIGEST_ARTIFACT",
    "SOURCE_FIXTURE_FILENAME",
    "SOURCE_FIXTURE_PATH",
    "SOURCE_PROOF_SCHEMA_ARTIFACT",
    "SOURCE_PROOF_SCHEMA_DIGEST_ARTIFACT",
    "SourceFixtureProof",
    "Stage1V1NonReadyDecision",
    "Stage1V1NonReadySourceProof",
    "TERMINAL_REASON_SCHEMA_ARTIFACT",
    "TERMINAL_REASON_SCHEMA_DIGEST_ARTIFACT",
    "TerminalReasonBinding",
    "ClarificationDecision",
    "ClarificationSlot",
    "canonical_json",
    "canonical_sha256",
    "expected_g_u_002",
    "expected_nonready_decision",
    "expected_r_a_001",
    "load_nonready_fixture",
    "normalize_nonready_intent",
    "verify_fixture_artifacts",
    "verify_nonready_digest",
    "verify_schema_artifacts",
    "write_fixture_artifacts",
    "write_schema_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(_main())
