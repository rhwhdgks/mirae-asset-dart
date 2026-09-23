"""Source-bound Stage1 v1 Gold expectation bundle.

This artifact is deliberately separate from the correction overlay.  It
describes the expected *meaning and normal outcome* for a question and does
not contain an execution recipe.  The checked-in bundle contains the ten
mandatory rows whose exact semantics are implemented in the current v1
modules; the internal Gold coverage gate opens only while all ten retain the
approved order.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import unicodedata
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Mapping, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    model_validator,
)
from typing_extensions import Annotated

from .compiled_answer_contract_v1 import (
    LIMITATION_REGISTRY,
    LIMITATION_REGISTRY_V1,
    LimitationCode,
)
from .semantic_intent_v1 import (
    STAGE1_SEMANTIC_INTENT_V1,
    SemanticIntent,
    semantic_intent_digest,
)


MANIFEST_SCHEMA_VERSION = "stage1-v1-gold-expectation-manifest/1.1"
ROW_SCHEMA_VERSION = "stage1-v1-gold-expectation-row/1.1"
GOLD_MANIFEST_SCHEMA_VERSION = MANIFEST_SCHEMA_VERSION
GOLD_ROW_SCHEMA_VERSION = ROW_SCHEMA_VERSION

SOURCE_ROOT = "fixtures/query_plan_v04"
GOLD_EXPECTATION_ROOT = "fixtures/stage1_v1_gold_expectations"
EXPECTATION_ROOT = GOLD_EXPECTATION_ROOT
MANIFEST_FILENAME = "manifest.json"
EXPECTATIONS_FILENAME = "expectations_v1.jsonl"
SOURCE_SHA256SUMS_FILENAME = "SHA256SUMS"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / GOLD_EXPECTATION_ROOT / MANIFEST_FILENAME

MANIFEST_SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_gold_expectation_manifest.schema.json")
MANIFEST_SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_gold_expectation_manifest.schema.sha256")
ROW_SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_gold_expectation_row.schema.json")
ROW_SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_gold_expectation_row.schema.sha256")

# Compatibility aliases make the artifact names easy to discover without
# coupling callers to the older overlay module's constants.
SCHEMA_ARTIFACT = MANIFEST_SCHEMA_ARTIFACT
SCHEMA_DIGEST_ARTIFACT = MANIFEST_SCHEMA_DIGEST_ARTIFACT

SOURCE_FIXTURE_FILES = (
    "questions_v0.4.jsonl",
    "plan_proposals_v0.4.jsonl",
    "query_plan_handoffs_v0.4.jsonl",
    "answer_requirements_v0.4.jsonl",
    "migration_report_v0.4.json",
)

# This is an immutable release baseline.  A changed source file and a changed
# source checksum declaration must not be able to move the Gold bundle.
_EXPECTED_SOURCE_FIXTURE_DIGESTS = {
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
EXPECTED_SOURCE_FIXTURE_DIGESTS: Mapping[str, str] = MappingProxyType(
    _EXPECTED_SOURCE_FIXTURE_DIGESTS)

FROZEN_QUESTION_COUNT = 70
TOTAL_EXPECTED_ROWS = 10
APPROVAL_STATE_INCOMPLETE = "incomplete"
APPROVAL_STATE_COMPLETE = "complete"

G_A_001 = "G-A-001"
G_A_004 = "G-A-004"
G_A_010 = "G-A-010"
G_I_004 = "G-I-004"
G_I_006 = "G-I-006"
G_I_009 = "G-I-009"
G_O_001 = "G-O-001"
R_A_002 = "R-A-002"
R_A_001 = "R-A-001"
G_U_002 = "G-U-002"
G_A_001_QUESTION = "삼성전자의 2025년 연결기준 매출액은 얼마인가?"
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
R_A_002_QUESTION = "삼성전자 2025년 매출 얼마야?"
R_A_001_QUESTION = "삼성전자가 작년에 얼마나 벌었어?"
G_U_002_QUESTION = "삼성전자의 2027년 예상 매출액과 매수 의견을 알려줘."
APPROVED_QUESTION_IDS = (
    G_A_001, G_A_004, G_A_010, G_I_004, G_I_006, G_I_009,
    G_O_001, R_A_002, R_A_001, G_U_002,
)
MANDATORY_GOLD_QUESTION_IDS = (
    G_A_001,
    "G-A-004",
    "G-A-010",
    "G-I-004",
    "G-I-006",
    "G-I-009",
    G_O_001,
    R_A_002,
    "R-A-001",
    "G-U-002",
)

NormalOutcome: TypeAlias = Literal[
    "ready", "partial_ready", "needs_clarification", "terminal"
]
SourceFixtureName: TypeAlias = Literal[
    "questions_v0.4.jsonl",
    "plan_proposals_v0.4.jsonl",
    "query_plan_handoffs_v0.4.jsonl",
    "answer_requirements_v0.4.jsonl",
    "migration_report_v0.4.json",
]

# Limitation/decision reasons are semantic states, not infrastructure errors.
# Keeping the negative lookahead in the formal schema prevents a technical
# incident from being laundered into a normal outcome by a JSON-only caller.
# Pydantic's Rust regex engine intentionally does not support look-around.
# The lexical part is therefore expressed in the JSON Schema and the negative
# semantic vocabulary is checked by ``_assert_safe_code`` below.
_SAFE_CODE_PATTERN = r"^[A-Za-z][A-Za-z0-9_.:\-]{0,127}$"
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[
    str, StringConstraints(min_length=1, max_length=256)
]
SafeCode = Annotated[
    str, StringConstraints(min_length=1, max_length=128,
                           pattern=_SAFE_CODE_PATTERN)
]
QuestionId = Annotated[
    str, StringConstraints(min_length=1, max_length=64,
                           pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
]
QuestionText = Annotated[
    str, StringConstraints(min_length=1, max_length=4096)
]

LimitationFamily: TypeAlias = Literal[
    "source_scope", "identity_lineage", "ordering", "capability"
]
BindingStatus: TypeAlias = Literal["executable", "qualified", "limited"]
ClarificationReasonCode: TypeAlias = Literal[
    "ambiguity_requires_user",
    "missing_entity",
    "missing_period",
    "missing_scope",
    "missing_target",
]
ClarificationRole: TypeAlias = Literal[
    "metric", "scope", "entity", "period", "target", "selection", "qualifier"
]
TerminalReasonCode: TypeAlias = Literal[
    "future_forecast",
    "investment_advice",
    "policy_refusal",
    "out_of_scope",
    "unsupported_request",
]


class GoldExpectationError(ValueError):
    """Raised when the Gold bundle or its frozen source is unsafe."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True,
        revalidate_instances="always",
    )


def _canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", warnings=False)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    """Hash canonical UTF-8 JSON with deterministic map/list encoding."""

    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def question_sha256(question: str) -> str:
    """Hash the exact UTF-8 question text, without whitespace rewriting."""

    return sha256(question.encode("utf-8")).hexdigest()


def _unique(values: list[str], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label}에는 중복 값이 있을 수 없습니다")


_TECHNICAL_CODE_WORDS = (
    "technical", "error", "exception", "failure", "failed", "timeout",
    "drift", "coercion", "serialization", "schema", "provider", "route",
    "task", "execution", "canonical", "legacy", "slot",
)


def _assert_safe_code(value: str, *, label: str) -> None:
    lowered = value.casefold()
    if any(
            lowered == word or lowered.startswith(word + "_")
            or lowered.startswith(word + ":") or lowered.startswith(word + "-")
            for word in _TECHNICAL_CODE_WORDS):
        raise ValueError(f"{label}에 기술 실패/실행 상태 code를 넣을 수 없습니다")


def _assert_source_digests(value: Mapping[str, str], *, label: str) -> None:
    if set(value) != set(SOURCE_FIXTURE_FILES):
        raise ValueError(f"{label}는 frozen source 5개 파일을 정확히 포함해야 합니다")
    for filename in SOURCE_FIXTURE_FILES:
        if value[filename] != EXPECTED_SOURCE_FIXTURE_DIGESTS[filename]:
            raise ValueError(f"{label}의 {filename} digest가 frozen baseline과 다릅니다")


def _assert_safe_relative_path(value: str, *, label: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label}는 안전한 상대 경로여야 합니다")


def _assert_no_forbidden_keys(value: Any, *, path: str = "payload") -> None:
    """Reject execution-oriented names before typed parsing.

    This is intentionally recursive so an unknown nested object cannot hide
    a prohibited field behind an otherwise valid row.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise GoldExpectationError(f"{path}의 key는 문자열이어야 합니다")
            normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
            if (
                normalized.startswith(("route", "task", "execution", "legacy", "canonical"))
                or "slot" in normalized
                or normalized in {"canonicalexecutionid", "executionid"}
            ):
                raise GoldExpectationError(
                    f"Gold expectation에 금지된 실행/legacy key가 있습니다: {path}.{key}")
            _assert_no_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_keys(child, path=f"{path}[{index}]")


def _assert_safe_schema(schema: Mapping[str, Any]) -> None:
    """Check formal schema property names for forbidden execution concepts."""

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            properties = value.get("properties")
            if isinstance(properties, Mapping):
                for key in properties:
                    normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
                    if (
                        normalized.startswith(("route", "task", "execution", "legacy", "canonical"))
                        or "slot" in normalized
                        or normalized in {"canonicalexecutionid", "executionid"}
                    ):
                        raise RuntimeError(
                            f"formal schema에 금지된 property가 있습니다: {key}")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)


def _rejected_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GoldExpectationError(f"JSON object duplicate key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text, object_pairs_hook=_rejected_duplicate_key)
    except GoldExpectationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GoldExpectationError(f"JSON 파싱 실패: {path}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise GoldExpectationError(f"JSONL 읽기 실패: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            raise GoldExpectationError(f"JSONL 빈 줄은 허용되지 않습니다: {path}:{line_no}")
        try:
            value = json.loads(line, object_pairs_hook=_rejected_duplicate_key)
        except GoldExpectationError:
            raise
        except json.JSONDecodeError as exc:
            raise GoldExpectationError(f"JSONL 파싱 실패: {path}:{line_no}") from exc
        if not isinstance(value, dict):
            raise GoldExpectationError(f"JSONL 행은 object여야 합니다: {path}:{line_no}")
        rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    try:
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise GoldExpectationError(f"파일 digest 계산 실패: {path}") from exc


def _read_source_sha256sums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise GoldExpectationError(f"SHA256SUMS 읽기 실패: {path}") from exc
    parsed: dict[str, str] = {}
    for line_no, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise GoldExpectationError(f"SHA256SUMS 형식 오류: {path}:{line_no}")
        if parts[1] in parsed:
            raise GoldExpectationError(f"SHA256SUMS 중복 파일: {parts[1]}")
        parsed[parts[1]] = parts[0]
    if set(parsed) != set(SOURCE_FIXTURE_FILES):
        raise GoldExpectationError("SHA256SUMS가 frozen source 5개 파일과 다릅니다")
    return parsed


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_path(path: Path, *, anchor: Path, label: str) -> Path:
    """Validate a path without allowing a symlink component to be followed."""

    anchor_abs = _absolute_lexical(anchor)
    path_abs = _absolute_lexical(path)
    try:
        relative = path_abs.relative_to(anchor_abs)
    except ValueError as exc:
        raise GoldExpectationError(f"{label}가 anchor 밖입니다: {path}") from exc

    # Validate ancestors too: an ancestor symlink otherwise escapes before the
    # final candidate is inspected.
    current = Path(anchor_abs.anchor)
    for component in anchor_abs.parts[1:]:
        current = current / component
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise GoldExpectationError(f"{label} ancestor symlink가 있습니다: {current}")
        except OSError as exc:
            raise GoldExpectationError(f"{label} path를 읽을 수 없습니다: {current}") from exc
    current = anchor_abs
    for component in relative.parts:
        current = current / component
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise GoldExpectationError(f"{label} symlink가 있습니다: {current}")
        except FileNotFoundError:
            # The caller reports a missing final path; a missing intermediate
            # component is not a reason to resolve anything.
            continue
        except OSError as exc:
            raise GoldExpectationError(f"{label} path를 읽을 수 없습니다: {current}") from exc
    return path_abs


def _secure_directory(path: Path, *, anchor: Path, label: str) -> Path:
    path_abs = _assert_no_symlink_path(path, anchor=anchor, label=label)
    try:
        info = path_abs.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise GoldExpectationError(f"{label}는 regular directory여야 합니다")
        resolved = path_abs.resolve(strict=True)
        resolved.relative_to(_absolute_lexical(anchor).resolve(strict=True))
    except GoldExpectationError:
        raise
    except (OSError, ValueError) as exc:
        raise GoldExpectationError(f"{label}가 안전한 directory가 아닙니다: {path}") from exc
    return resolved


def _secure_file(path: Path, *, anchor: Path, label: str) -> Path:
    path_abs = _assert_no_symlink_path(path, anchor=anchor, label=label)
    try:
        info = path_abs.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise GoldExpectationError(f"{label}는 regular file이어야 합니다")
        resolved = path_abs.resolve(strict=True)
        resolved.relative_to(_absolute_lexical(anchor).resolve(strict=True))
    except GoldExpectationError:
        raise
    except (OSError, ValueError) as exc:
        raise GoldExpectationError(f"{label}가 안전한 file이 아닙니다: {path}") from exc
    return resolved


def _secure_relative_file(root: Path, relative: str, *, label: str) -> Path:
    _assert_safe_relative_path(relative, label=label)
    return _secure_file(root / relative, anchor=root, label=label)


class ClarificationExpectation(_StrictFrozenModel):
    """Typed public clarification: role and the values the user may choose."""

    reason_code: ClarificationReasonCode
    role: ClarificationRole
    allowed_values: list[Identifier] = Field(min_length=1)
    prompt: QuestionText | None = None

    @model_validator(mode="after")
    def validate_clarification(self) -> "ClarificationExpectation":
        _unique(list(self.allowed_values), label="clarification allowed_values")
        return self

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return (self.reason_code,)


class TerminalExpectation(_StrictFrozenModel):
    """Ordered policy reasons bound to the exact SemanticIntent item IDs."""

    reason_bindings: list["TerminalReasonBinding"] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_terminal(self) -> "TerminalExpectation":
        codes = [row.code for row in self.reason_bindings]
        _unique(codes, label="terminal reason codes")
        return self


class TerminalReasonBinding(_StrictFrozenModel):
    code: TerminalReasonCode
    item_ids: list[Identifier] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_binding(self) -> "TerminalReasonBinding":
        _unique(list(self.item_ids), label="terminal reason item_ids")
        return self


class LimitationBinding(_StrictFrozenModel):
    """One ordered, registry-bound limitation attached to a Gold field."""

    code: LimitationCode
    family: LimitationFamily
    registry_version: Literal[LIMITATION_REGISTRY_V1] = LIMITATION_REGISTRY_V1

    @model_validator(mode="after")
    def validate_registry_binding(self) -> "LimitationBinding":
        expected_family = LIMITATION_REGISTRY.get(self.code)
        if expected_family != self.family:
            raise ValueError("limitation code/family registry binding이 다릅니다")
        return self


# Explicit alias for callers that want to distinguish this compact Gold
# binding from the compiler's richer field-reference binding.
GoldLimitationBinding = LimitationBinding


class FieldExpectation(_StrictFrozenModel):
    """One semantic output field and its executable/qualified/limited status.

    Gold expectations deliberately do not carry runtime roots.  A
    ``qualified`` field therefore means that a root-capable result is expected
    together with one or more ordered typed limitations; the root itself is
    proved by the separately checked vertical/compiler artifacts.
    """

    item_id: Identifier
    source_field_index: StrictInt = Field(ge=0)
    surface: QuestionText
    binding_status: BindingStatus
    limitation_bindings: list[LimitationBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_binding(self) -> "FieldExpectation":
        # Preserve declared order (it is part of the semantic expectation),
        # while rejecting an exact duplicate binding that could hide a
        # malformed repeated limitation.
        bindings = [
            binding.model_dump(mode="json", warnings=False)
            for binding in self.limitation_bindings
        ]
        _unique(
            [canonical_sha256(binding) for binding in bindings],
            label="field limitation bindings",
        )
        if self.binding_status == "executable":
            if self.limitation_bindings:
                raise ValueError("executable field에는 limitation이 결속될 수 없습니다")
        elif not self.limitation_bindings:
            raise ValueError(
                f"{self.binding_status} field에는 limitation binding이 필요합니다")
        return self

    @property
    def expected_binding(self) -> BindingStatus:
        return self.binding_status


class GoldExpectationRow(_StrictFrozenModel):
    """One question meaning contract and its normal response disposition."""

    schema_version: Literal[ROW_SCHEMA_VERSION] = ROW_SCHEMA_VERSION
    question_id: QuestionId
    question: QuestionText
    question_sha256: Digest
    source_fixture_digests: dict[SourceFixtureName, Digest]
    semantic_intent: SemanticIntent
    intent_digest: Digest
    expected_outcome: NormalOutcome
    limitation_registry_version: Literal[LIMITATION_REGISTRY_V1] = (
        LIMITATION_REGISTRY_V1)
    field_expectations: list[FieldExpectation] = Field(default_factory=list)
    clarification_expectation: ClarificationExpectation | None = None
    terminal_expectation: TerminalExpectation | None = None

    # These ten rows have stronger invariants than a future row: their exact
    # text, intent digest and normal outcome are approved by the corresponding
    # v1 implementation modules.
    _APPROVED_QUESTION_TEXT: ClassVar[Mapping[str, str]] = MappingProxyType({
        G_A_001: G_A_001_QUESTION,
        G_A_004: G_A_004_QUESTION,
        G_A_010: G_A_010_QUESTION,
        G_I_004: G_I_004_QUESTION,
        G_I_006: G_I_006_QUESTION,
        G_I_009: G_I_009_QUESTION,
        R_A_002: R_A_002_QUESTION,
        R_A_001: R_A_001_QUESTION,
        G_U_002: G_U_002_QUESTION,
    })
    _APPROVED_INTENT_DIGESTS: ClassVar[Mapping[str, str]] = MappingProxyType({
        G_A_001: "39682f26022e39c69eebe0efcde40193fdd08dce5b5eacc368f485b8d25da0fd",
        G_A_004: "939359e1c3c15f29e82cca5186175062c961e29f6d56443795654e156328d522",
        G_A_010: "30b37e0ef40659689ac31a3245489c1e176f834c8335bdeccd8448a8a2001a7b",
        G_I_004: "2a53fae7c5c277a65e367fa2a9db9faa2d4be715a8d749bf728e53667997407b",
        G_I_006: "ac21ba235e4a375a7a39d1cc132da23b81eef091f7c8f73b2da2bde7e4755710",
        G_I_009: "bbf854a53de9fb155447f6acb5edae8a714c33fe0efe0c98a3eae83ed0e89e08",
        G_O_001: "36cf683a838906147dca94255115395d3245b2da72313884cc445137bb8795ca",
        R_A_002: "7dd762c97b3250033890930e0fad0135d037fe890cf7cd161b5f259ba6c58afe",
        R_A_001: "cdb99d4892efe836e13f079b39f0d3d9e4d60dc57dba248b0bff975dee87620e",
        G_U_002: "68794e201bb91d8a9158e523f67ff0e7231f0b70f64a5506d7a25ad2d832a91c",
    })
    _APPROVED_OUTCOMES: ClassVar[Mapping[str, NormalOutcome]] = MappingProxyType({
        G_A_001: "ready",
        G_A_004: "ready",
        G_A_010: "ready",
        G_I_004: "partial_ready",
        G_I_006: "ready",
        G_I_009: "ready",
        R_A_002: "ready",
        # 이슈 #94 25 — 「얼마나 벌었어?」는 더 이상 되묻지 않는다. 매출액·
        # 영업이익·당기순이익은 손익계산서를 위에서 아래로 읽은 세 단계라
        # 셋을 나란히 답한다. 승인 근거는 `agent/concept_question_patterns.tsv`
        # 의 `fanout` 열이고, 펴는 것은 `SummaryMetricFanoutRegrounder` 다.
        R_A_001: "ready",
        G_U_002: "terminal",
    })
    _APPROVED_FIELD_BINDINGS: ClassVar[
        Mapping[str, tuple[tuple[str, int, str, str, tuple[tuple[str, str, str], ...]], ...]]
    ] = MappingProxyType({
        G_A_010: (
            ("item-1", 0, "투자 대상", "executable", ()),
            ("item-1", 1, "목적", "executable", ()),
            ("item-1", 2, "금액", "executable", ()),
            ("item-1", 3, "기간", "executable", ()),
        ),
        G_I_004: (
            (
                "item-1", 0, "내용", "qualified",
                ((
                    "intraday_order_unavailable", "ordering",
                    LIMITATION_REGISTRY_V1,
                ),),
            ),
            (
                "item-2", 0, "최종 상태", "qualified",
                ((
                    "ambiguous_event_origin", "identity_lineage",
                    LIMITATION_REGISTRY_V1,
                ),),
            ),
        ),
        G_I_006: (
            ("item-1", 0, "정정 후 계약금액과 해지금액은 같으며", "executable", ()),
            ("item-2", 0, "왜 다른가", "executable", ()),
        ),
        G_I_009: (
            ("item-1", 0, "해지된 이유", "executable", ()),
            ("item-2", 0, "계약 효력발생 조건", "executable", ()),
        ),
    })

    @model_validator(mode="after")
    def validate_row(self) -> "GoldExpectationRow":
        _assert_source_digests(
            self.source_fixture_digests, label="row source_fixture_digests")

        # Re-enter the strict boundary.  This matters for model_copy(update=)
        # because that Pydantic helper intentionally skips validation.
        intent = SemanticIntent.model_validate(
            self.semantic_intent.model_dump(mode="python", warnings=False),
            strict=True,
        )
        if self.intent_digest != semantic_intent_digest(intent):
            raise ValueError("intent_digest가 embedded SemanticIntent와 다릅니다")
        if self.question_sha256 != question_sha256(self.question):
            raise ValueError("question_sha256가 exact question text와 다릅니다")
        _assert_intent_grounded(intent, self.question)

        # Re-enter the nested field boundary as well.  ``model_copy(update=)``
        # intentionally skips validation, and a forged field must not be able
        # to bypass the executable/qualified/limited partition below.
        field_expectations = tuple(
            FieldExpectation.model_validate(field, strict=True)
            for field in self.field_expectations
        )

        approved_bindings = self._APPROVED_FIELD_BINDINGS.get(self.question_id)
        if approved_bindings is not None:
            actual_bindings = tuple(
                (
                    field.item_id,
                    field.source_field_index,
                    field.surface,
                    field.binding_status,
                    tuple(
                        (binding.code, binding.family, binding.registry_version)
                        for binding in field.limitation_bindings
                    ),
                )
                for field in field_expectations
            )
            if actual_bindings != approved_bindings:
                raise ValueError(
                    f"{self.question_id} approved field limitation binding이 다릅니다")

        outcome = self.expected_outcome
        if outcome in {"ready", "partial_ready"}:
            if self.clarification_expectation is not None:
                raise ValueError("ready outcome에는 clarification expectation이 없어야 합니다")
            if self.terminal_expectation is not None:
                raise ValueError("ready outcome에는 terminal expectation이 없어야 합니다")
            expected_fields = [
                (item.item_id, field_index, surface)
                for item in intent.answer_items
                for field_index, surface in enumerate(item.output.field_surfaces)
            ]
            actual_fields = [
                (field.item_id, field.source_field_index, field.surface)
                for field in field_expectations
            ]
            if actual_fields != expected_fields:
                raise ValueError(
                    "field_expectations가 SemanticIntent의 item/ordered field와 정확히 일치해야 합니다")
            for field in field_expectations:
                if field.surface not in self.question:
                    raise ValueError("field expectation surface가 exact question에 없습니다")
            statuses = [field.binding_status for field in field_expectations]
            if outcome == "ready" and any(status != "executable" for status in statuses):
                raise ValueError("ready에는 모든 SemanticIntent field가 executable이어야 합니다")
            if outcome == "partial_ready":
                root_capable = {"executable", "qualified"}
                limitation_bearing = {"qualified", "limited"}
                if not (
                        any(status in root_capable for status in statuses)
                        and any(status in limitation_bearing for status in statuses)
                ):
                    raise ValueError(
                        "partial_ready에는 root-capable field와 limitation-bearing "
                        "field가 각각 하나 이상 필요합니다")
        elif outcome == "needs_clarification":
            if self.field_expectations:
                raise ValueError(
                    "needs_clarification에는 field binding expectation이 없어야 합니다")
            if self.clarification_expectation is None:
                raise ValueError("needs_clarification에는 clarification expectation이 필요합니다")
            if self.terminal_expectation is not None:
                raise ValueError("clarification outcome에는 terminal expectation이 없어야 합니다")
        elif outcome == "terminal":
            if self.field_expectations:
                raise ValueError("terminal에는 field binding expectation이 없어야 합니다")
            if self.terminal_expectation is None:
                raise ValueError("terminal에는 terminal expectation이 필요합니다")
            if self.clarification_expectation is not None:
                raise ValueError("terminal outcome에는 clarification expectation이 없어야 합니다")
            item_ids = {item.item_id for item in intent.answer_items}
            bound_ids = {
                item_id for binding in self.terminal_expectation.reason_bindings
                for item_id in binding.item_ids
            }
            if bound_ids != item_ids:
                raise ValueError("terminal reason binding이 SemanticIntent item 전체를 정확히 덮어야 합니다")

        approved_question = self._APPROVED_QUESTION_TEXT.get(self.question_id)
        if approved_question is not None and self.question != approved_question:
            raise ValueError(f"{self.question_id} question이 승인 원문과 다릅니다")
        approved_intent_digest = self._APPROVED_INTENT_DIGESTS.get(self.question_id)
        if approved_intent_digest is not None and self.intent_digest != approved_intent_digest:
            raise ValueError(f"{self.question_id} intent가 승인된 SemanticIntent와 다릅니다")
        approved_outcome = self._APPROVED_OUTCOMES.get(self.question_id)
        if approved_outcome is not None and self.expected_outcome != approved_outcome:
            raise ValueError(
                f"{self.question_id} approved outcome은 {approved_outcome}여야 합니다")
        return self

    # Read-only naming aliases for callers that prefer explicit terminology.
    @property
    def question_text(self) -> str:
        return self.question

    @property
    def semantic_intent_digest(self) -> str:
        return self.intent_digest

    @property
    def expected_normal_outcome(self) -> NormalOutcome:
        return self.expected_outcome

    @property
    def limitation_codes(self) -> tuple[str, ...]:
        return tuple(
            binding.code
            for field in self.field_expectations
            for binding in field.limitation_bindings
        )

    @property
    def limitation_bindings(self) -> tuple[LimitationBinding, ...]:
        """All field limitations in row field/order declaration order."""

        return tuple(
            binding
            for field in self.field_expectations
            for binding in field.limitation_bindings
        )

    @property
    def expected_bindings(self) -> tuple[str, ...]:
        return tuple(field.binding_status for field in self.field_expectations)


class GoldExpectationManifest(_StrictFrozenModel):
    """Bundle metadata and the closed approval threshold."""

    schema_version: Literal[MANIFEST_SCHEMA_VERSION] = MANIFEST_SCHEMA_VERSION
    source_root: Identifier = SOURCE_ROOT
    source_sha256sums_path: Identifier = SOURCE_SHA256SUMS_FILENAME
    source_fixture_digests: dict[SourceFixtureName, Digest]
    source_question_count: StrictInt = FROZEN_QUESTION_COUNT
    expectations_path: Identifier = EXPECTATIONS_FILENAME
    row_count: StrictInt = Field(ge=0)
    total_expected_rows: StrictInt = TOTAL_EXPECTED_ROWS
    approval_state: Literal[APPROVAL_STATE_INCOMPLETE, APPROVAL_STATE_COMPLETE]
    expectations_digest: Digest
    manifest_digest: Digest

    @model_validator(mode="after")
    def validate_manifest(self) -> "GoldExpectationManifest":
        _assert_safe_relative_path(self.source_root, label="source_root")
        _assert_safe_relative_path(
            self.source_sha256sums_path, label="source_sha256sums_path")
        _assert_safe_relative_path(self.expectations_path, label="expectations_path")
        _assert_source_digests(
            self.source_fixture_digests, label="manifest source_fixture_digests")
        if self.source_question_count != FROZEN_QUESTION_COUNT:
            raise ValueError("frozen question count는 70이어야 합니다")
        if self.total_expected_rows != TOTAL_EXPECTED_ROWS:
            raise ValueError("total_expected_rows는 10으로 고정됩니다")
        if self.row_count > self.total_expected_rows:
            raise ValueError("Gold row_count가 mandatory 10개를 초과할 수 없습니다")
        if self.approval_state == APPROVAL_STATE_COMPLETE and (
                self.row_count < self.total_expected_rows):
            raise ValueError("10행 전에는 approval_state=complete가 될 수 없습니다")
        body = self.model_dump(mode="json", warnings=False)
        body.pop("manifest_digest", None)
        if self.manifest_digest != canonical_sha256(body):
            raise ValueError("manifest_digest가 일치하지 않습니다")
        return self

    @classmethod
    def create(
            cls,
            *,
            expectations_digest: str,
            row_count: int,
            approval_state: str = APPROVAL_STATE_INCOMPLETE,
            expectations_path: str = EXPECTATIONS_FILENAME,
            source_root: str = SOURCE_ROOT,
            source_sha256sums_path: str = SOURCE_SHA256SUMS_FILENAME,
            ) -> "GoldExpectationManifest":
        body: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "source_root": source_root,
            "source_sha256sums_path": source_sha256sums_path,
            "source_fixture_digests": dict(EXPECTED_SOURCE_FIXTURE_DIGESTS),
            "source_question_count": FROZEN_QUESTION_COUNT,
            "expectations_path": expectations_path,
            "row_count": row_count,
            "total_expected_rows": TOTAL_EXPECTED_ROWS,
            "approval_state": approval_state,
            "expectations_digest": expectations_digest,
        }
        body["manifest_digest"] = canonical_sha256(body)
        return cls.model_validate(body, strict=True)


@dataclass(frozen=True, slots=True)
class LoadedGoldExpectations:
    manifest: GoldExpectationManifest
    rows: tuple[GoldExpectationRow, ...]

    @property
    def by_question_id(self) -> Mapping[str, GoldExpectationRow]:
        return MappingProxyType({row.question_id: row for row in self.rows})

    @property
    def canonical_digest(self) -> str:
        return canonical_sha256({
            "manifest": self.manifest.model_dump(mode="json", warnings=False),
            "rows": [row.model_dump(mode="json", warnings=False)
                     for row in self.rows],
        })

    @property
    def hcx_gate(self) -> bool:
        """Whether the Gold coverage is sufficient to open the HCX gate."""

        return (
            self.manifest.approval_state == APPROVAL_STATE_COMPLETE
            and self.manifest.row_count == TOTAL_EXPECTED_ROWS
            and len(self.rows) == TOTAL_EXPECTED_ROWS
            and tuple(row.question_id for row in self.rows)
            == MANDATORY_GOLD_QUESTION_IDS
        )

    @property
    def hcx_gate_open(self) -> bool:
        return self.hcx_gate


Stage1V1GoldExpectationManifest = GoldExpectationManifest
Stage1V1GoldExpectationRow = GoldExpectationRow


def _assert_intent_grounded(intent: SemanticIntent, question: str) -> None:
    """Every user-facing semantic surface must occur in the exact question."""

    question_nfc = unicodedata.normalize("NFC", question)

    def ground(value: str, label: str) -> None:
        if unicodedata.normalize("NFC", value) not in question_nfc:
            raise ValueError(f"{label}가 exact question text에 결속되지 않았습니다")

    for index, entity in enumerate(intent.entities):
        ground(entity.surface, f"entities[{index}].surface")
    for index, item in enumerate(intent.answer_items):
        ground(item.target.surface, f"answer_items[{index}].target.surface")
        for qindex, value in enumerate(item.target.qualifier_surfaces):
            ground(value, f"answer_items[{index}].target.qualifier_surfaces[{qindex}]")
        for qindex, value in enumerate(item.scope.target_period_expressions):
            ground(value, f"answer_items[{index}].scope.target_period_expressions[{qindex}]")
        for name, value in (
                ("as_of_expression", item.scope.as_of_expression),
                ("document_group_expression", item.scope.document_group_expression)):
            if value is not None:
                ground(value, f"answer_items[{index}].scope.{name}")
        for qindex, value in enumerate(item.scope.scope_qualifier_expressions):
            ground(value, f"answer_items[{index}].scope.scope_qualifier_expressions[{qindex}]")
        if item.selection is not None:
            ground(item.selection.criterion_surface,
                   f"answer_items[{index}].selection.criterion_surface")
        for qindex, value in enumerate(item.output.field_surfaces):
            ground(value, f"answer_items[{index}].output.field_surfaces[{qindex}]")
    for index, premise in enumerate(intent.premises):
        ground(premise.raw_text, f"premises[{index}].raw_text")
    for index, mention in enumerate(intent.unresolved_mentions):
        ground(mention.raw_text, f"unresolved_mentions[{index}].raw_text")


def _intent_for(question_id: str) -> SemanticIntent:
    if question_id == G_A_001:
        payload = {
            "schema_version": STAGE1_SEMANTIC_INTENT_V1,
            "entities": [{
                "entity_id": "entity-1", "kind_hint": "company", "surface": "삼성전자",
            }],
            "answer_items": [{
                "item_id": "item-1",
                "target": {
                    "kind": "metric", "surface": "매출액",
                    "entity_refs": ["entity-1"], "qualifier_surfaces": [],
                },
                "operation": "retrieve",
                "scope": {
                    "target_period_expressions": ["2025년"],
                    "as_of_expression": None,
                    "document_group_expression": None,
                    "scope_qualifier_expressions": ["연결기준"],
                },
                "selection": None,
                "output": {
                    "shape": "scalar", "projection_mode": "named_fields",
                    "field_surfaces": ["얼마"],
                    "presentation": "auto",
                },
            }],
            "answer_groups": [], "premises": [],
            "unresolved_mentions": [], "presentation": "auto",
        }
    elif question_id == G_A_004:
        payload = {
            "schema_version": STAGE1_SEMANTIC_INTENT_V1,
            "entities": [
                {
                    "entity_id": "entity-1",
                    "kind_hint": "company",
                    "surface": "삼성전자",
                },
                {
                    "entity_id": "entity-2",
                    "kind_hint": "company",
                    "surface": "SK하이닉스",
                },
            ],
            "answer_items": [{
                "item_id": "item-1",
                "target": {
                    "kind": "metric",
                    "surface": "매출액",
                    "entity_refs": ["entity-1", "entity-2"],
                    "qualifier_surfaces": [],
                },
                "operation": "compare",
                "scope": {
                    "target_period_expressions": ["2025년"],
                    "as_of_expression": None,
                    "document_group_expression": None,
                    "scope_qualifier_expressions": ["연결"],
                },
                "selection": {
                    "criterion_surface": "큰",
                    "k": None,
                    "mode": "maximum",
                },
                "output": {
                    "shape": "comparison",
                    "projection_mode": "named_fields",
                    "field_surfaces": ["큰 기업", "차이"],
                    "presentation": "auto",
                },
            }],
            "answer_groups": [],
            "premises": [],
            "unresolved_mentions": [],
            "presentation": "auto",
        }
    elif question_id == R_A_002:
        payload = {
            "schema_version": STAGE1_SEMANTIC_INTENT_V1,
            "entities": [{
                "entity_id": "entity-1", "kind_hint": "company", "surface": "삼성전자",
            }],
            "answer_items": [{
                "item_id": "item-1",
                "target": {
                    "kind": "metric", "surface": "매출",
                    "entity_refs": ["entity-1"], "qualifier_surfaces": [],
                },
                "operation": "retrieve",
                "scope": {
                    "target_period_expressions": ["2025년"],
                    "as_of_expression": None,
                    "document_group_expression": None,
                    "scope_qualifier_expressions": [],
                },
                "selection": None,
                "output": {
                    "shape": "scalar", "projection_mode": "named_fields",
                    "field_surfaces": ["얼마"],
                    "presentation": "auto",
                },
            }],
            "answer_groups": [], "premises": [],
            "unresolved_mentions": [], "presentation": "auto",
        }
    elif question_id in {R_A_001, G_U_002}:
        # These non-ready rows are deliberately sourced from the strict
        # module rather than recreated here.  A local import avoids making
        # Gold's import graph depend on non-ready implementation details.
        from .stage1_v1_nonready import (
            expected_g_u_002,
            expected_r_a_001,
        )
        return (
            expected_r_a_001() if question_id == R_A_001
            else expected_g_u_002()
        ).source_intent
    elif question_id in {G_A_010, G_I_004, G_I_006, G_I_009, G_O_001}:
        # Reuse the exact semantic intent exercised by the current compiler
        # demo slice.  Only the intent crosses into Gold; roots, resolution
        # values, evidence and Stage2 payloads remain outside this artifact.
        from .deterministic_plan_compiler_v1 import _demo_slice
        return _demo_slice(question_id).intent
    else:
        raise GoldExpectationError(f"승인된 Gold intent가 없는 question_id: {question_id}")
    return SemanticIntent.model_validate(payload, strict=True)


def _build_row(question_id: str, question: str) -> GoldExpectationRow:
    if question_id == G_U_002:
        return _build_nonready_row(question_id, question)
    if question_id in {G_A_010, G_I_004}:
        return (
            _build_g_a_010_row(question_id, question)
            if question_id == G_A_010
            else _build_g_i_004_row(question_id, question)
        )
    intent = _intent_for(question_id)
    field_expectations = [
        {
            "item_id": item.item_id,
            "source_field_index": field_index,
            "surface": surface,
            "binding_status": "executable",
            "limitation_bindings": [],
        }
        for item in intent.answer_items
        for field_index, surface in enumerate(item.output.field_surfaces)
    ]
    return GoldExpectationRow.model_validate({
        "schema_version": ROW_SCHEMA_VERSION,
        "question_id": question_id,
        "question": question,
        "question_sha256": question_sha256(question),
        "source_fixture_digests": dict(EXPECTED_SOURCE_FIXTURE_DIGESTS),
        "semantic_intent": intent.model_dump(mode="json", warnings=False),
        "intent_digest": semantic_intent_digest(intent),
        "expected_outcome": "ready",
        "limitation_registry_version": LIMITATION_REGISTRY_V1,
        "field_expectations": field_expectations,
        "clarification_expectation": None,
        "terminal_expectation": None,
    }, strict=True)


def _build_g_i_004_row(question_id: str, question: str) -> GoldExpectationRow:
    """Build the exact semantic/qualified G-I-004 Gold expectation."""

    intent = _intent_for(question_id)
    field_expectations = [
        {
            "item_id": "item-1",
            "source_field_index": 0,
            "surface": "내용",
            "binding_status": "qualified",
            "limitation_bindings": [{
                "code": "intraday_order_unavailable",
                "family": "ordering",
                "registry_version": LIMITATION_REGISTRY_V1,
            }],
        },
        {
            "item_id": "item-2",
            "source_field_index": 0,
            "surface": "최종 상태",
            "binding_status": "qualified",
            "limitation_bindings": [{
                "code": "ambiguous_event_origin",
                "family": "identity_lineage",
                "registry_version": LIMITATION_REGISTRY_V1,
            }],
        },
    ]
    return GoldExpectationRow.model_validate({
        "schema_version": ROW_SCHEMA_VERSION,
        "question_id": question_id,
        "question": question,
        "question_sha256": question_sha256(question),
        "source_fixture_digests": dict(EXPECTED_SOURCE_FIXTURE_DIGESTS),
        "semantic_intent": intent.model_dump(mode="json", warnings=False),
        "intent_digest": semantic_intent_digest(intent),
        "expected_outcome": "partial_ready",
        "limitation_registry_version": LIMITATION_REGISTRY_V1,
        "field_expectations": field_expectations,
        "clarification_expectation": None,
        "terminal_expectation": None,
    }, strict=True)


def _build_g_a_010_row(question_id: str, question: str) -> GoldExpectationRow:
    """Build the exact semantic/complete G-A-010 Gold expectation."""

    intent = _intent_for(question_id)
    field_expectations = [
        {
            "item_id": "item-1",
            "source_field_index": 0,
            "surface": "투자 대상",
            "binding_status": "executable",
            "limitation_bindings": [],
        },
        {
            "item_id": "item-1",
            "source_field_index": 1,
            "surface": "목적",
            "binding_status": "executable",
            "limitation_bindings": [],
        },
        {
            "item_id": "item-1",
            "source_field_index": 2,
            "surface": "금액",
            "binding_status": "executable",
            "limitation_bindings": [],
        },
        {
            "item_id": "item-1",
            "source_field_index": 3,
            "surface": "기간",
            "binding_status": "executable",
            "limitation_bindings": [],
        },
    ]
    return GoldExpectationRow.model_validate({
        "schema_version": ROW_SCHEMA_VERSION,
        "question_id": question_id,
        "question": question,
        "question_sha256": question_sha256(question),
        "source_fixture_digests": dict(EXPECTED_SOURCE_FIXTURE_DIGESTS),
        "semantic_intent": intent.model_dump(mode="json", warnings=False),
        "intent_digest": semantic_intent_digest(intent),
        "expected_outcome": "ready",
        "limitation_registry_version": LIMITATION_REGISTRY_V1,
        "field_expectations": field_expectations,
        "clarification_expectation": None,
        "terminal_expectation": None,
    }, strict=True)


def _build_nonready_row(question_id: str, question: str) -> GoldExpectationRow:
    """Project the source-bound non-ready decision into the Gold row shape.

    이슈 #94 25 이후로 남은 non-ready 문항은 `G-U-002` 하나다. `R-A-001` 은
    되묻던 세 후보를 나란히 답하게 되어 ready 로 옮겼다 —
    `agent/stage1_v1_nonready.py` 의 기록은 그대로 두었다. 그것은 우리가 지금
    무엇을 내는지가 아니라 **동결된 v0.4 원본이 무엇이라 적었는지**의 증명이고,
    Gold 는 그 원본 intent 를 여전히 그대로 쓴다(`_intent_for`).
    """

    from .stage1_v1_nonready import expected_g_u_002

    decision = expected_g_u_002()
    clarification = None
    if decision.clarification is not None:
        slot = decision.clarification.slot
        clarification = {
            "reason_code": decision.clarification.reason,
            "role": slot.target,
            "allowed_values": list(slot.allowed_values),
            "prompt": "어떤 지표를 원하시나요?",
        }
    terminal = None
    if decision.terminal_reasons:
        terminal = {
            "reason_bindings": [
                {"code": reason.reason, "item_ids": [reason.item_id]}
                for reason in decision.terminal_reasons
            ],
        }
    return GoldExpectationRow.model_validate({
        "schema_version": ROW_SCHEMA_VERSION,
        "question_id": question_id,
        "question": question,
        "question_sha256": question_sha256(question),
        "source_fixture_digests": dict(EXPECTED_SOURCE_FIXTURE_DIGESTS),
        "semantic_intent": decision.source_intent.model_dump(
            mode="json", warnings=False),
        "intent_digest": decision.source_intent_digest,
        "expected_outcome": decision.disposition,
        "limitation_registry_version": LIMITATION_REGISTRY_V1,
        "field_expectations": [],
        "clarification_expectation": clarification,
        "terminal_expectation": terminal,
    }, strict=True)


def expected_g_a_001_row() -> GoldExpectationRow:
    return _build_row(G_A_001, G_A_001_QUESTION)


def expected_r_a_002_row() -> GoldExpectationRow:
    return _build_row(R_A_002, R_A_002_QUESTION)


def expected_g_a_004_row() -> GoldExpectationRow:
    return _build_row(G_A_004, G_A_004_QUESTION)


def expected_g_i_004_row() -> GoldExpectationRow:
    return _build_row(G_I_004, G_I_004_QUESTION)


def expected_g_i_006_row() -> GoldExpectationRow:
    return _build_row(G_I_006, G_I_006_QUESTION)


def expected_g_i_009_row() -> GoldExpectationRow:
    return _build_row(G_I_009, G_I_009_QUESTION)


def expected_g_o_001_row() -> GoldExpectationRow:
    return _build_row(G_O_001, G_O_001_QUESTION)


def expected_g_a_010_row() -> GoldExpectationRow:
    return _build_row(G_A_010, G_A_010_QUESTION)


def expected_r_a_001_row() -> GoldExpectationRow:
    return _build_row(R_A_001, R_A_001_QUESTION)


def expected_g_u_002_row() -> GoldExpectationRow:
    return _build_row(G_U_002, G_U_002_QUESTION)


def expected_rows() -> tuple[GoldExpectationRow, ...]:
    return (
        expected_g_a_001_row(),
        expected_g_a_004_row(),
        expected_g_a_010_row(),
        expected_g_i_004_row(),
        expected_g_i_006_row(),
        expected_g_i_009_row(),
        expected_g_o_001_row(),
        expected_r_a_002_row(),
        expected_r_a_001_row(),
        expected_g_u_002_row(),
    )


def _source_root_for(
        manifest_path: Path,
        manifest: GoldExpectationManifest,
        source_root: str | Path | None,
        ) -> Path:
    if source_root is not None:
        raw = Path(source_root)
        if not raw.is_absolute():
            raw = Path.cwd() / raw
        return _secure_directory(raw, anchor=raw.parent, label="source_root override")
    if manifest.source_root == SOURCE_ROOT:
        raw = PROJECT_ROOT / manifest.source_root
        anchor = PROJECT_ROOT
    else:
        raw = manifest_path.parent / manifest.source_root
        anchor = manifest_path.parent
    return _secure_directory(raw, anchor=anchor, label="manifest source_root")


def _verify_source_files(
        source_root: Path,
        manifest: GoldExpectationManifest,
        ) -> Mapping[str, dict[str, Any]]:
    checksum_path = _secure_relative_file(
        source_root, manifest.source_sha256sums_path, label="source SHA256SUMS")
    declared = _read_source_sha256sums(checksum_path)
    if declared != dict(EXPECTED_SOURCE_FIXTURE_DIGESTS):
        raise GoldExpectationError("source SHA256SUMS가 frozen baseline과 다릅니다")
    if manifest.source_fixture_digests != declared:
        raise GoldExpectationError("manifest source digest와 SHA256SUMS가 다릅니다")
    for filename in SOURCE_FIXTURE_FILES:
        path = _secure_relative_file(
            source_root, filename, label=f"source fixture {filename}")
        if _sha256_file(path) != EXPECTED_SOURCE_FIXTURE_DIGESTS[filename]:
            raise GoldExpectationError(f"source fixture digest drift: {filename}")

    question_path = _secure_relative_file(
        source_root, "questions_v0.4.jsonl", label="frozen questions")
    question_rows = _read_jsonl(question_path)
    if len(question_rows) != FROZEN_QUESTION_COUNT:
        raise GoldExpectationError(
            f"frozen questions 행 수가 70이 아닙니다: {len(question_rows)}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in question_rows:
        question_id = row.get("question_id")
        question = row.get("question")
        if not isinstance(question_id, str) or not isinstance(question, str):
            raise GoldExpectationError("frozen question row의 id/text가 문자열이 아닙니다")
        if question_id in by_id:
            raise GoldExpectationError(f"frozen question_id 중복: {question_id}")
        by_id[question_id] = row
    return MappingProxyType(by_id)


def _load_manifest(path: Path) -> GoldExpectationManifest:
    manifest_file = _secure_file(path, anchor=path.parent, label="Gold manifest")
    value = _read_json(manifest_file)
    if not isinstance(value, dict):
        raise GoldExpectationError("Gold manifest는 object여야 합니다")
    _assert_no_forbidden_keys(value)
    try:
        return GoldExpectationManifest.model_validate(value, strict=True)
    except ValueError as exc:
        raise GoldExpectationError("Gold manifest strict validation 실패") from exc


def load_gold_expectations(
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        *,
        source_root: str | Path | None = None,
        expectations_path: str | Path | None = None,
        ) -> LoadedGoldExpectations:
    """Load and verify the source-hash-bound Gold expectation bundle."""

    manifest_file = Path(manifest_path)
    manifest = _load_manifest(manifest_file)
    source_dir = _source_root_for(manifest_file, manifest, source_root)
    question_by_id = _verify_source_files(source_dir, manifest)

    rows_file = (
        _secure_file(Path(expectations_path), anchor=Path(expectations_path).parent,
                     label="Gold expectations override")
        if expectations_path is not None
        else _secure_relative_file(
            manifest_file.parent, manifest.expectations_path,
            label="Gold expectations")
    )
    raw_rows = _read_jsonl(rows_file)
    rows: list[GoldExpectationRow] = []
    question_ids: set[str] = set()
    for raw in raw_rows:
        _assert_no_forbidden_keys(raw)
        try:
            row = GoldExpectationRow.model_validate(raw, strict=True)
        except ValueError as exc:
            raise GoldExpectationError("Gold row strict validation 실패") from exc
        if row.question_id in question_ids:
            raise GoldExpectationError(f"Gold row question_id 중복: {row.question_id}")
        if row.question_id not in MANDATORY_GOLD_QUESTION_IDS:
            raise GoldExpectationError(
                f"Gold row question_id가 mandatory 10-slice가 아닙니다: {row.question_id}")
        source_question = question_by_id.get(row.question_id)
        if source_question is None:
            raise GoldExpectationError(
                f"Gold row question_id가 frozen 70행에 없습니다: {row.question_id}")
        if source_question.get("question") != row.question:
            raise GoldExpectationError(
                f"Gold row exact question drift: {row.question_id}")
        if question_sha256(str(source_question["question"])) != row.question_sha256:
            raise GoldExpectationError(
                f"Gold row question hash drift: {row.question_id}")
        if row.source_fixture_digests != manifest.source_fixture_digests:
            raise GoldExpectationError(
                f"Gold row와 manifest source digest가 다릅니다: {row.question_id}")
        question_ids.add(row.question_id)
        rows.append(row)

    if len(rows) != manifest.row_count:
        raise GoldExpectationError(
            f"Gold row_count 불일치: manifest={manifest.row_count}, actual={len(rows)}")
    actual_ids = tuple(row.question_id for row in rows)
    if actual_ids != APPROVED_QUESTION_IDS:
        raise GoldExpectationError(
            "Gold row question_id가 승인된 implemented-row 순서와 다릅니다")
    if canonical_sha256([
            row.model_dump(mode="json", warnings=False) for row in rows
    ]) != manifest.expectations_digest:
        raise GoldExpectationError("Gold expectations_digest가 일치하지 않습니다")
    if manifest.approval_state == APPROVAL_STATE_COMPLETE:
        if len(rows) != TOTAL_EXPECTED_ROWS or actual_ids != MANDATORY_GOLD_QUESTION_IDS:
            raise GoldExpectationError(
                "complete approval은 mandatory 10개 ID를 정확한 순서로 모두 요구합니다")
    return LoadedGoldExpectations(manifest=manifest, rows=tuple(rows))


def load_bundle(
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        *,
        source_root: str | Path | None = None,
        expectations_path: str | Path | None = None,
        ) -> LoadedGoldExpectations:
    """Short alias for callers that treat the artifact as a bundle."""

    return load_gold_expectations(
        manifest_path, source_root=source_root, expectations_path=expectations_path)


GoldExpectationBundle = LoadedGoldExpectations


def load_gold_bundle(
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        *,
        source_root: str | Path | None = None,
        expectations_path: str | Path | None = None,
        ) -> LoadedGoldExpectations:
    return load_gold_expectations(
        manifest_path, source_root=source_root, expectations_path=expectations_path)


def hcx_gate(value: LoadedGoldExpectations | None = None) -> bool:
    """Return false unless the ten-row approval threshold is fully open."""

    return load_gold_expectations().hcx_gate if value is None else value.hcx_gate


def _schema_artifact_bytes(
        model: type[BaseModel], artifact: Path,
        ) -> tuple[bytes, bytes, str]:
    schema = model.model_json_schema(mode="validation")
    _assert_safe_schema(schema)
    schema_bytes = _canonical_json(schema).encode("utf-8")
    digest = sha256(schema_bytes).hexdigest()
    return schema_bytes, f"{digest}  {artifact.name}\n".encode("ascii"), digest


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_schema_artifacts() -> tuple[str, str]:
    manifest, manifest_sha, manifest_digest = _schema_artifact_bytes(
        GoldExpectationManifest, MANIFEST_SCHEMA_ARTIFACT)
    row, row_sha, row_digest = _schema_artifact_bytes(
        GoldExpectationRow, ROW_SCHEMA_ARTIFACT)
    _atomic_write(MANIFEST_SCHEMA_ARTIFACT, manifest)
    _atomic_write(MANIFEST_SCHEMA_DIGEST_ARTIFACT, manifest_sha)
    _atomic_write(ROW_SCHEMA_ARTIFACT, row)
    _atomic_write(ROW_SCHEMA_DIGEST_ARTIFACT, row_sha)
    return manifest_digest, row_digest


def verify_schema_artifacts() -> tuple[str, str]:
    expected = (
        (MANIFEST_SCHEMA_ARTIFACT, *_schema_artifact_bytes(
            GoldExpectationManifest, MANIFEST_SCHEMA_ARTIFACT)[:2]),
        (MANIFEST_SCHEMA_DIGEST_ARTIFACT, None, _schema_artifact_bytes(
            GoldExpectationManifest, MANIFEST_SCHEMA_ARTIFACT)[1]),
        (ROW_SCHEMA_ARTIFACT, *_schema_artifact_bytes(
            GoldExpectationRow, ROW_SCHEMA_ARTIFACT)[:2]),
        (ROW_SCHEMA_DIGEST_ARTIFACT, None, _schema_artifact_bytes(
            GoldExpectationRow, ROW_SCHEMA_ARTIFACT)[1]),
    )
    for path, schema, digest in expected:
        try:
            actual = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"Gold schema artifact가 없습니다: {path}") from exc
        expected_bytes = schema if schema is not None else digest
        if actual != expected_bytes:
            raise RuntimeError(f"Gold schema artifact drift: {path}")
    return (
        _schema_artifact_bytes(GoldExpectationManifest, MANIFEST_SCHEMA_ARTIFACT)[2],
        _schema_artifact_bytes(GoldExpectationRow, ROW_SCHEMA_ARTIFACT)[2],
    )


def write_gold_expectation_artifacts(
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        *, expectations_path: str | Path | None = None,
        ) -> str:
    """Write the deterministic implemented-row bundle and return its digest."""

    manifest_file = Path(manifest_path)
    rows_file = (
        Path(expectations_path)
        if expectations_path is not None
        else manifest_file.parent / EXPECTATIONS_FILENAME
    )
    rows = expected_rows()
    row_payloads = [row.model_dump(mode="json", warnings=False) for row in rows]
    expectations_digest = canonical_sha256(row_payloads)
    expectations_rel = rows_file.name
    manifest = GoldExpectationManifest.create(
        expectations_digest=expectations_digest,
        row_count=len(row_payloads),
        approval_state=(
            APPROVAL_STATE_COMPLETE
            if len(row_payloads) == TOTAL_EXPECTED_ROWS
            else APPROVAL_STATE_INCOMPLETE),
        expectations_path=expectations_rel,
    )
    _atomic_write(
        rows_file,
        ("".join(_canonical_json(row) + "\n" for row in row_payloads)).encode("utf-8"),
    )
    _atomic_write(
        manifest_file,
        (_canonical_json(manifest.model_dump(mode="json", warnings=False)) + "\n")
        .encode("utf-8"),
    )
    return load_gold_expectations(manifest_file).canonical_digest


def verify_gold_expectation_artifacts(
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        *,
        source_root: str | Path | None = None,
        expectations_path: str | Path | None = None,
        ) -> str:
    return load_gold_expectations(
        manifest_path, source_root=source_root,
        expectations_path=expectations_path).canonical_digest


def write_bundle_artifacts(
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        *, expectations_path: str | Path | None = None,
        ) -> str:
    return write_gold_expectation_artifacts(
        manifest_path, expectations_path=expectations_path)


def verify_bundle_artifacts(
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        *,
        source_root: str | Path | None = None,
        expectations_path: str | Path | None = None,
        ) -> str:
    return verify_gold_expectation_artifacts(
        manifest_path, source_root=source_root,
        expectations_path=expectations_path)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        bundle_digest = write_gold_expectation_artifacts()
        schema_digests = write_schema_artifacts()
    else:
        bundle_digest = verify_gold_expectation_artifacts()
        schema_digests = verify_schema_artifacts()
    loaded = load_gold_expectations()
    print(
        f"PASS: {ROW_SCHEMA_VERSION} rows={len(loaded.rows)}/"
        f"{loaded.manifest.total_expected_rows} approval={loaded.manifest.approval_state} "
        f"hcx_gate={str(loaded.hcx_gate).lower()} bundle_digest={bundle_digest} "
        f"manifest_schema_sha256={schema_digests[0]} "
        f"row_schema_sha256={schema_digests[1]}"
    )
    return 0


__all__ = [
    "APPROVAL_STATE_COMPLETE", "APPROVAL_STATE_INCOMPLETE",
    "APPROVED_QUESTION_IDS", "BindingStatus", "ClarificationExpectation", "Digest",
    "FieldExpectation", "GoldLimitationBinding", "LimitationBinding",
    "GoldExpectationBundle",
    "EXPECTED_SOURCE_FIXTURE_DIGESTS", "EXPECTATION_ROOT",
    "EXPECTATIONS_FILENAME", "FROZEN_QUESTION_COUNT", "GOLD_EXPECTATION_ROOT",
    "GOLD_MANIFEST_SCHEMA_VERSION", "GOLD_ROW_SCHEMA_VERSION",
    "GoldExpectationError", "GoldExpectationManifest", "GoldExpectationRow",
    "G_A_001", "G_A_001_QUESTION", "G_A_004", "G_A_004_QUESTION",
    "G_A_010", "G_A_010_QUESTION",
    "G_I_004", "G_I_004_QUESTION", "G_I_006", "G_I_006_QUESTION",
    "G_I_009", "G_I_009_QUESTION",
    "G_O_001", "G_O_001_QUESTION",
    "G_U_002", "G_U_002_QUESTION", "LIMITATION_REGISTRY",
    "LIMITATION_REGISTRY_V1", "LimitationCode", "LimitationFamily",
    "LoadedGoldExpectations",
    "MANDATORY_GOLD_QUESTION_IDS", "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_ARTIFACT", "MANIFEST_SCHEMA_DIGEST_ARTIFACT",
    "MANIFEST_SCHEMA_VERSION", "NormalOutcome", "PROJECT_ROOT", "R_A_001",
    "R_A_001_QUESTION", "R_A_002", "R_A_002_QUESTION", "ROW_SCHEMA_ARTIFACT",
    "ROW_SCHEMA_DIGEST_ARTIFACT",
    "ROW_SCHEMA_VERSION", "SCHEMA_ARTIFACT", "SCHEMA_DIGEST_ARTIFACT",
    "SOURCE_FIXTURE_FILES", "SOURCE_ROOT", "Stage1V1GoldExpectationManifest",
    "Stage1V1GoldExpectationRow", "TOTAL_EXPECTED_ROWS", "TerminalExpectation",
    "TerminalReasonBinding",
    "canonical_sha256", "expected_g_a_001_row", "expected_g_a_004_row",
    "expected_g_a_010_row", "expected_g_i_004_row", "expected_g_i_006_row",
    "expected_g_i_009_row", "expected_g_o_001_row",
    "expected_g_u_002_row", "expected_r_a_001_row", "expected_r_a_002_row",
    "expected_rows", "hcx_gate", "load_bundle", "load_gold_bundle",
    "load_gold_expectations",
    "question_sha256", "verify_bundle_artifacts",
    "verify_gold_expectation_artifacts", "verify_schema_artifacts",
    "write_bundle_artifacts", "write_gold_expectation_artifacts",
    "write_schema_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(_main())
