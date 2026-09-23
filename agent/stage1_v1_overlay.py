"""Strict, source-hash-bound Stage1 v1 expectation overlay.

The v0.4 query-plan fixture is an immutable comparison corpus.  This module
keeps the first v1 correction in a separate, small overlay: it binds every
overlay row to all five v0.4 source-file digests, checks those files again at
load time, and records only structural expectations and correction provenance.

The public loader is intentionally fail-closed.  It rejects source drift,
manifest/row drift, duplicate or unknown question IDs, JSON extras, and
Pydantic coercion.  It never rewrites the frozen fixture.
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
    StrictInt,
    StringConstraints,
    model_validator,
)
from typing_extensions import Annotated


MANIFEST_SCHEMA_VERSION = "stage1-v1-overlay-manifest/1.0"
ROW_SCHEMA_VERSION = "stage1-v1-overlay-row/1.0"
SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_overlay_manifest.schema.json")
SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_overlay_manifest.schema.sha256")
ROW_SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_overlay_row.schema.json")
ROW_SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_overlay_row.schema.sha256")
SOURCE_ROOT = "fixtures/query_plan_v04"
OVERLAY_ROOT = "fixtures/stage1_v1_overlay"
MANIFEST_FILENAME = "manifest.json"
EXPECTATIONS_FILENAME = "expectations_v1.jsonl"
SOURCE_SHA256SUMS_FILENAME = "SHA256SUMS"
R_A_002 = "R-A-002"
R_A_002_QUESTION = "삼성전자 2025년 매출 얼마야?"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / OVERLAY_ROOT / MANIFEST_FILENAME

SOURCE_FIXTURE_FILES = (
    "questions_v0.4.jsonl",
    "plan_proposals_v0.4.jsonl",
    "query_plan_handoffs_v0.4.jsonl",
    "answer_requirements_v0.4.jsonl",
    "migration_report_v0.4.json",
)

# These values are copied from the frozen fixture's existing SHA256SUMS.  The
# loader checks both this immutable baseline and the on-disk checksum file;
# changing a source file and changing its checksum declaration together still
# cannot silently move the v1 overlay.
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

DecisionCode = Literal[
    "r_a_002_primary_statement_scope_override",
    "r_a_002_stale_answer_requirement_conflict",
]
Operation = Literal["retrieve", "compare"]
TargetKind = Literal[
    "metric", "attribute", "event", "document", "topic", "entity"
]
OutputShape = Literal[
    "scalar", "record", "record_list", "comparison", "timeline",
    "narrative", "verdict"
]
Disposition = Literal["ready", "partial_ready", "needs_clarification", "terminal"]
ItemStatus = Literal["ready", "partial", "unavailable"]

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=256)]
SourceFixtureName: TypeAlias = Literal[
    "questions_v0.4.jsonl",
    "plan_proposals_v0.4.jsonl",
    "query_plan_handoffs_v0.4.jsonl",
    "answer_requirements_v0.4.jsonl",
    "migration_report_v0.4.json",
]


class OverlayError(ValueError):
    """Raised when an overlay or its source binding is unsafe to load."""


class _StrictFrozenModel(BaseModel):
    """Strict/frozen base for every persisted overlay model."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    """Hash canonical UTF-8 JSON with stable key and list ordering."""

    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _unique(values: list[str], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label}에는 중복 값이 있을 수 없습니다")


def _assert_source_digests(value: Mapping[str, str], *, label: str) -> None:
    if set(value) != set(SOURCE_FIXTURE_FILES):
        raise ValueError(
            f"{label}는 v0.4 다섯 source fixture를 정확히 포함해야 합니다")
    for filename in SOURCE_FIXTURE_FILES:
        if value[filename] != EXPECTED_SOURCE_FIXTURE_DIGESTS[filename]:
            raise ValueError(f"{label}의 {filename} digest가 frozen baseline과 다릅니다")


def _assert_safe_relative_path(value: str, *, label: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label}는 안전한 상대 경로여야 합니다")


class NormalizedIntentExpectation(_StrictFrozenModel):
    """Structural v1 intent expectation, not a copied answer."""

    item_count: StrictInt = Field(ge=0)
    operations: list[Operation] = Field(default_factory=list)
    target_kinds: list[TargetKind] = Field(default_factory=list)
    target_surfaces: list[Identifier] = Field(default_factory=list)
    output_shapes: list[OutputShape] = Field(default_factory=list)
    ordered_field_surfaces: list[Identifier] = Field(default_factory=list)
    premise_count: StrictInt = Field(ge=0)
    group_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_item_shape(self) -> "NormalizedIntentExpectation":
        for label, values in (
            ("operations", self.operations),
            ("target_kinds", self.target_kinds),
            ("target_surfaces", self.target_surfaces),
            ("output_shapes", self.output_shapes),
        ):
            if len(values) != self.item_count:
                raise ValueError(
                    f"{label} 길이는 item_count와 같아야 합니다")
        return self


class ContractExpectation(_StrictFrozenModel):
    """Only the contract shape needed to audit completeness."""

    completion: Literal["complete", "partial"]
    item_statuses: list[ItemStatus] = Field(default_factory=list)
    limited_count: StrictInt = Field(ge=0)
    limitation_codes: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_codes(self) -> "ContractExpectation":
        _unique(self.limitation_codes, label="limitation_codes")
        if self.completion == "complete":
            if self.item_statuses and any(
                    status != "ready" for status in self.item_statuses):
                raise ValueError("complete contract에는 ready item만 허용됩니다")
            if self.limited_count != 0 or self.limitation_codes:
                raise ValueError("complete contract에는 limitation이 없어야 합니다")
        return self


class ResolutionExpectation(_StrictFrozenModel):
    scope_resolution: Literal["principal_financial_statement"]
    resolved_scope: Literal["CFS", "SFS"]
    applied_default: Literal["primary_statement_scope"]


class OverlayProvenance(_StrictFrozenModel):
    """The conflict evidence that justifies the overlay-only correction."""

    handoff_source: Literal["query_plan_handoffs_v0.4"]
    handoff_status: Literal["ready"]
    handoff_scope: Literal["CFS"]
    answer_requirement_source: Literal["answer_requirements_v0.4"]
    answer_requirement_status: Literal["needs_clarification"]
    conflict: Literal["stale_answer_requirement"]
    authority: Literal["stage1_v1_primary_statement_policy"]


class OverlayRow(_StrictFrozenModel):
    schema_version: Literal[ROW_SCHEMA_VERSION] = ROW_SCHEMA_VERSION
    question_id: Identifier
    decision_codes: list[DecisionCode] = Field(min_length=1)
    source_fixture_digests: dict[SourceFixtureName, Digest]
    disposition: Disposition
    normalized_intent_expectation: NormalizedIntentExpectation
    contract_expectation: ContractExpectation
    resolution_expectation: ResolutionExpectation
    provenance: OverlayProvenance

    _R_A_002_CODES: ClassVar[tuple[str, ...]] = (
        "r_a_002_primary_statement_scope_override",
        "r_a_002_stale_answer_requirement_conflict",
    )

    @model_validator(mode="after")
    def validate_row(self) -> "OverlayRow":
        _assert_source_digests(
            self.source_fixture_digests, label="source_fixture_digests")
        _unique(self.decision_codes, label="decision_codes")
        if self.question_id != R_A_002:
            raise ValueError(f"지원되지 않는 overlay question_id입니다: {self.question_id}")
        if tuple(self.decision_codes) != self._R_A_002_CODES:
            raise ValueError("R-A-002 decision_codes가 FINAL allowlist 순서와 다릅니다")
        if self.disposition != "ready":
            raise ValueError("R-A-002 overlay disposition은 ready여야 합니다")

        intent = self.normalized_intent_expectation
        if (
            intent.item_count != 1
            or intent.operations != ["retrieve"]
            or intent.target_kinds != ["metric"]
            or intent.target_surfaces != ["매출"]
            or intent.output_shapes != ["scalar"]
            or intent.ordered_field_surfaces != ["얼마"]
            or intent.premise_count != 0
            or intent.group_count != 0
        ):
            raise ValueError("R-A-002 normalized intent expectation이 FINAL과 다릅니다")

        contract = self.contract_expectation
        if (
            contract.completion != "complete"
            or contract.item_statuses != ["ready"]
            or contract.limited_count != 0
            or contract.limitation_codes
        ):
            raise ValueError("R-A-002 contract expectation이 FINAL과 다릅니다")

        resolution = self.resolution_expectation
        if (
            resolution.scope_resolution != "principal_financial_statement"
            or resolution.resolved_scope != "CFS"
            or resolution.applied_default != "primary_statement_scope"
        ):
            raise ValueError("R-A-002 resolution expectation이 FINAL과 다릅니다")

        provenance = self.provenance
        if (
            provenance.handoff_status != "ready"
            or provenance.handoff_scope != "CFS"
            or provenance.answer_requirement_status != "needs_clarification"
            or provenance.conflict != "stale_answer_requirement"
        ):
            raise ValueError("R-A-002 provenance가 source conflict FINAL과 다릅니다")
        return self


class OverlayManifest(_StrictFrozenModel):
    schema_version: Literal[MANIFEST_SCHEMA_VERSION] = MANIFEST_SCHEMA_VERSION
    source_root: Identifier = SOURCE_ROOT
    source_sha256sums_path: Identifier = SOURCE_SHA256SUMS_FILENAME
    source_fixture_digests: dict[SourceFixtureName, Digest]
    expectations_path: Identifier = EXPECTATIONS_FILENAME
    row_count: StrictInt = Field(ge=1)
    expectations_digest: Digest
    manifest_digest: Digest

    @model_validator(mode="after")
    def validate_manifest(self) -> "OverlayManifest":
        _assert_safe_relative_path(self.source_root, label="source_root")
        _assert_safe_relative_path(
            self.source_sha256sums_path, label="source_sha256sums_path")
        _assert_safe_relative_path(self.expectations_path, label="expectations_path")
        _assert_source_digests(
            self.source_fixture_digests, label="source_fixture_digests")
        body = self.model_dump(mode="json")
        body.pop("manifest_digest", None)
        if self.manifest_digest != canonical_sha256(body):
            raise ValueError("overlay manifest_digest가 일치하지 않습니다")
        return self

    @classmethod
    def create(
            cls, *, expectations_digest: str, row_count: int = 1,
            source_root: str = SOURCE_ROOT,
            source_sha256sums_path: str = SOURCE_SHA256SUMS_FILENAME,
            expectations_path: str = EXPECTATIONS_FILENAME,
            ) -> "OverlayManifest":
        body: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "source_root": source_root,
            "source_sha256sums_path": source_sha256sums_path,
            "source_fixture_digests": dict(EXPECTED_SOURCE_FIXTURE_DIGESTS),
            "expectations_path": expectations_path,
            "row_count": row_count,
            "expectations_digest": expectations_digest,
        }
        body["manifest_digest"] = canonical_sha256(body)
        return cls.model_validate(body)


@dataclass(frozen=True, slots=True)
class LoadedOverlay:
    manifest: OverlayManifest
    rows: tuple[OverlayRow, ...]

    @property
    def by_question_id(self) -> Mapping[str, OverlayRow]:
        return MappingProxyType({row.question_id: row for row in self.rows})

    @property
    def canonical_digest(self) -> str:
        return canonical_sha256({
            "manifest": self.manifest.model_dump(mode="json"),
            "rows": [row.model_dump(mode="json") for row in self.rows],
        })


def expected_r_a_002_row() -> OverlayRow:
    """Return the deterministic FINAL correction row used by the generator."""

    return OverlayRow.model_validate({
        "schema_version": ROW_SCHEMA_VERSION,
        "question_id": R_A_002,
        "decision_codes": list(OverlayRow._R_A_002_CODES),
        "source_fixture_digests": dict(EXPECTED_SOURCE_FIXTURE_DIGESTS),
        "disposition": "ready",
        "normalized_intent_expectation": {
            "item_count": 1,
            "operations": ["retrieve"],
            "target_kinds": ["metric"],
            "target_surfaces": ["매출"],
            "output_shapes": ["scalar"],
            "ordered_field_surfaces": ["얼마"],
            "premise_count": 0,
            "group_count": 0,
        },
        "contract_expectation": {
            "completion": "complete",
            "item_statuses": ["ready"],
            "limited_count": 0,
            "limitation_codes": [],
        },
        "resolution_expectation": {
            "scope_resolution": "principal_financial_statement",
            "resolved_scope": "CFS",
            "applied_default": "primary_statement_scope",
        },
        "provenance": {
            "handoff_source": "query_plan_handoffs_v0.4",
            "handoff_status": "ready",
            "handoff_scope": "CFS",
            "answer_requirement_source": "answer_requirements_v0.4",
            "answer_requirement_status": "needs_clarification",
            "conflict": "stale_answer_requirement",
            "authority": "stage1_v1_primary_statement_policy",
        },
    })


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise OverlayError(f"source file를 읽을 수 없습니다: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            raise OverlayError(f"JSONL 빈 줄은 허용되지 않습니다: {path}:{line_no}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OverlayError(f"JSONL 파싱 실패: {path}:{line_no}") from exc
        if not isinstance(value, dict):
            raise OverlayError(f"JSONL 행은 object여야 합니다: {path}:{line_no}")
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
        raise OverlayError(f"source fixture를 읽을 수 없습니다: {path}") from exc


def _read_source_sha256sums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise OverlayError(f"SHA256SUMS를 읽을 수 없습니다: {path}") from exc
    parsed: dict[str, str] = {}
    for line_no, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            raise OverlayError(f"SHA256SUMS 형식 오류: {path}:{line_no}")
        filename = parts[1]
        if filename in parsed:
            raise OverlayError(f"SHA256SUMS 중복 파일: {filename}")
        parsed[filename] = parts[0]
    if set(parsed) != set(SOURCE_FIXTURE_FILES):
        raise OverlayError("SHA256SUMS가 frozen 다섯 source 파일과 다릅니다")
    return parsed


def _absolute_lexical(path: Path) -> Path:
    """Make an absolute path without following symlinks."""

    return Path(os.path.abspath(os.fspath(path)))


def _assert_anchor_directory(anchor: Path, *, label: str) -> Path:
    anchor_abs = _absolute_lexical(anchor)
    # Check every existing ancestor too; checking only ``anchor`` would allow
    # an anchor such as ``/tmp/link-to-external/subdir`` to escape before the
    # candidate path itself is inspected.
    current = Path(anchor_abs.anchor)
    for component in anchor_abs.parts[1:]:
        current = current / component
        try:
            ancestor_stat = current.lstat()
        except OSError as exc:
            raise OverlayError(
                f"{label} anchor component를 읽을 수 없습니다: {current}") from exc
        if stat.S_ISLNK(ancestor_stat.st_mode):
            raise OverlayError(
                f"{label} anchor ancestor symlink는 허용되지 않습니다: {current}")
    try:
        anchor_stat = anchor_abs.lstat()
    except OSError as exc:
        raise OverlayError(f"{label} anchor를 읽을 수 없습니다: {anchor}") from exc
    if stat.S_ISLNK(anchor_stat.st_mode):
        raise OverlayError(f"{label} anchor symlink는 허용되지 않습니다: {anchor}")
    if not stat.S_ISDIR(anchor_stat.st_mode):
        raise OverlayError(f"{label} anchor가 directory가 아닙니다: {anchor}")
    return anchor_abs


def _reject_symlink_components(
        path: Path, *, anchor: Path, label: str,
        ) -> tuple[Path, Path]:
    """Reject symlink components before any resolving operation."""

    anchor_abs = _assert_anchor_directory(anchor, label=label)
    path_abs = _absolute_lexical(path)
    try:
        relative = path_abs.relative_to(anchor_abs)
    except ValueError as exc:
        raise OverlayError(f"{label} path가 anchor 밖입니다: {path}") from exc
    current = anchor_abs
    for component in relative.parts:
        current = current / component
        try:
            component_stat = current.lstat()
        except OSError:
            # The final existence/type error is reported by the caller.  Do not
            # resolve a missing component here.
            continue
        if stat.S_ISLNK(component_stat.st_mode):
            raise OverlayError(f"{label} symlink component는 허용되지 않습니다: {current}")
    return path_abs, anchor_abs


def _secure_directory(path: Path, *, anchor: Path, label: str) -> Path:
    path_abs, anchor_abs = _reject_symlink_components(
        path, anchor=anchor, label=label)
    try:
        path_stat = path_abs.lstat()
    except OSError as exc:
        raise OverlayError(f"{label} directory를 읽을 수 없습니다: {path}") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        raise OverlayError(f"{label}는 regular directory여야 합니다: {path}")
    try:
        resolved = path_abs.resolve(strict=True)
        anchor_resolved = anchor_abs.resolve(strict=True)
        resolved.relative_to(anchor_resolved)
    except (OSError, ValueError) as exc:
        raise OverlayError(f"{label}가 anchor 밖으로 resolve됩니다: {path}") from exc
    return resolved


def _secure_regular_file(
        path: Path, *, anchor: Path, label: str,
        ) -> Path:
    path_abs, anchor_abs = _reject_symlink_components(
        path, anchor=anchor, label=label)
    try:
        path_stat = path_abs.lstat()
    except OSError as exc:
        raise OverlayError(f"{label} file을 읽을 수 없습니다: {path}") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise OverlayError(f"{label}는 regular file이어야 합니다: {path}")
    try:
        resolved = path_abs.resolve(strict=True)
        anchor_resolved = anchor_abs.resolve(strict=True)
        resolved.relative_to(anchor_resolved)
    except (OSError, ValueError) as exc:
        raise OverlayError(f"{label}가 anchor 밖으로 resolve됩니다: {path}") from exc
    return resolved


def _secure_relative_file(
        root: Path, relative: str, *, label: str,
        ) -> Path:
    _assert_safe_relative_path(relative, label=label)
    return _secure_regular_file(
        root / relative, anchor=root, label=label)


def _source_root_for(
        manifest_path: Path, manifest: OverlayManifest,
        source_root: str | Path | None,
        ) -> Path:
    if source_root is not None:
        # An absolute temporary copy is the supported replay seam.  Its own
        # lexical parent is the anchor; symlink root/parents are still rejected.
        raw = Path(source_root)
        if not raw.is_absolute():
            raw = Path.cwd() / raw
        return _secure_directory(
            raw, anchor=raw.parent, label="source_root override")
    # The checked-in manifest stores a repository-relative source root.  Resolve
    # it from the project root, not from the overlay directory.
    if manifest.source_root == SOURCE_ROOT:
        raw = PROJECT_ROOT / manifest.source_root
        anchor = PROJECT_ROOT
    else:
        raw = manifest_path.parent / manifest.source_root
        anchor = manifest_path.parent
    return _secure_directory(raw, anchor=anchor, label="manifest source_root")


def _verify_source_files(
        source_root: Path, manifest: OverlayManifest,
        ) -> None:
    checksum_path = _secure_relative_file(
        source_root, manifest.source_sha256sums_path,
        label="source SHA256SUMS")
    declared_sums = _read_source_sha256sums(checksum_path)
    if declared_sums != dict(EXPECTED_SOURCE_FIXTURE_DIGESTS):
        raise OverlayError("source SHA256SUMS가 frozen baseline과 다릅니다")
    if manifest.source_fixture_digests != declared_sums:
        raise OverlayError("manifest source_fixture_digests와 SHA256SUMS가 다릅니다")
    for filename in SOURCE_FIXTURE_FILES:
        path = _secure_relative_file(
            source_root, filename, label=f"source fixture {filename}")
        actual = _sha256_file(path)
        expected = EXPECTED_SOURCE_FIXTURE_DIGESTS[filename]
        if actual != expected or actual != manifest.source_fixture_digests[filename]:
            raise OverlayError(f"source fixture digest drift: {filename}")


def _find_question_row(
        rows: list[dict[str, Any]], *, question_id: str, source_name: str,
        ) -> dict[str, Any]:
    matches = [row for row in rows if row.get("question_id") == question_id]
    if len(matches) != 1:
        raise OverlayError(
            f"{source_name}에서 {question_id}가 정확히 1건이어야 합니다: {len(matches)}건")
    return matches[0]


def _verify_r_a_002_source_conflict(source_root: Path) -> None:
    question = _find_question_row(
        _read_jsonl(source_root / "questions_v0.4.jsonl"),
        question_id=R_A_002, source_name="questions_v0.4.jsonl")
    if question.get("question") != R_A_002_QUESTION:
        raise OverlayError("R-A-002 원 질문이 frozen source와 다릅니다")

    handoff = _find_question_row(
        _read_jsonl(source_root / "query_plan_handoffs_v0.4.jsonl"),
        question_id=R_A_002, source_name="query_plan_handoffs_v0.4.jsonl")
    handoff_body = handoff.get("handoff")
    if not isinstance(handoff_body, dict):
        raise OverlayError("R-A-002 handoff object가 없습니다")
    if handoff_body.get("status") != "ready":
        raise OverlayError("R-A-002 handoff source status가 ready가 아닙니다")
    plan = handoff_body.get("plan")
    if not isinstance(plan, dict):
        raise OverlayError("R-A-002 handoff plan이 없습니다")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise OverlayError("R-A-002 handoff task 구조가 다릅니다")
    facts = tasks[0].get("facts") if isinstance(tasks[0], dict) else None
    if not isinstance(facts, list) or len(facts) != 1:
        raise OverlayError("R-A-002 handoff fact 구조가 다릅니다")
    if facts[0].get("scope") != "CFS":
        raise OverlayError("R-A-002 handoff scope가 CFS가 아닙니다")

    requirement = _find_question_row(
        _read_jsonl(source_root / "answer_requirements_v0.4.jsonl"),
        question_id=R_A_002, source_name="answer_requirements_v0.4.jsonl")
    if (
        requirement.get("expected_action") != "clarify"
        or requirement.get("expected_handoff_status") != "needs_clarification"
        or requirement.get("expected_tool_status") != "needs_clarification"
    ):
        raise OverlayError(
            "R-A-002 answer requirement source가 stale conflict 기대와 다릅니다")


def _load_manifest(path: Path) -> OverlayManifest:
    path = _secure_regular_file(
        path, anchor=path.parent, label="overlay manifest")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OverlayError(f"overlay manifest 파싱 실패: {path}") from exc
    try:
        return OverlayManifest.model_validate(value)
    except ValueError as exc:
        raise OverlayError("overlay manifest strict validation 실패") from exc


def load_overlay(
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        *, source_root: str | Path | None = None,
        expectations_path: str | Path | None = None,
        ) -> LoadedOverlay:
    """Load and verify the source-hash-bound overlay.

    ``source_root`` is an explicit test/replay seam.  Production callers leave
    it unset so the repository's frozen ``fixtures/query_plan_v04`` is used.
    """

    manifest_file = Path(manifest_path)
    manifest = _load_manifest(manifest_file)
    source_dir = _source_root_for(manifest_file, manifest, source_root)
    _verify_source_files(source_dir, manifest)
    _verify_r_a_002_source_conflict(source_dir)

    rows_file = (
        _secure_regular_file(
            Path(expectations_path),
            anchor=Path(expectations_path).parent,
            label="expectations override")
        if expectations_path is not None
        else _secure_relative_file(
            manifest_file.parent, manifest.expectations_path,
            label="overlay expectations")
    )
    raw_rows = _read_jsonl(rows_file)
    rows: list[OverlayRow] = []
    question_ids: set[str] = set()
    for raw in raw_rows:
        try:
            row = OverlayRow.model_validate(raw)
        except ValueError as exc:
            raise OverlayError("overlay row strict validation 실패") from exc
        if row.question_id in question_ids:
            raise OverlayError(f"overlay row 중복 question_id: {row.question_id}")
        question_ids.add(row.question_id)
        if row.source_fixture_digests != manifest.source_fixture_digests:
            raise OverlayError("row와 manifest의 source digest binding이 다릅니다")
        rows.append(row)
    if len(rows) != manifest.row_count:
        raise OverlayError(
            f"overlay row_count 불일치: manifest={manifest.row_count}, actual={len(rows)}")
    if canonical_sha256([
            row.model_dump(mode="json") for row in rows
    ]) != manifest.expectations_digest:
        raise OverlayError("overlay expectations_digest가 일치하지 않습니다")
    if R_A_002 not in question_ids:
        raise OverlayError("R-A-002 overlay row가 없습니다")
    return LoadedOverlay(manifest=manifest, rows=tuple(rows))


def write_overlay_artifacts(
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        *, expectations_path: str | Path | None = None,
        ) -> str:
    """Write the deterministic R-A-002 overlay and return its bundle digest."""

    manifest_file = Path(manifest_path)
    rows_file = (
        Path(expectations_path)
        if expectations_path is not None
        else manifest_file.parent / EXPECTATIONS_FILENAME
    )
    row = expected_r_a_002_row()
    row_payload = row.model_dump(mode="json")
    expectations_digest = canonical_sha256([row_payload])
    manifest = OverlayManifest.create(
        expectations_digest=expectations_digest,
        row_count=1,
        expectations_path=rows_file.name,
    )
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    rows_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_text = _canonical_json(manifest.model_dump(mode="json")) + "\n"
    row_text = _canonical_json(row_payload) + "\n"
    manifest_tmp = manifest_file.with_name(manifest_file.name + ".tmp")
    rows_tmp = rows_file.with_name(rows_file.name + ".tmp")
    manifest_tmp.write_text(manifest_text, encoding="utf-8")
    rows_tmp.write_text(row_text, encoding="utf-8")
    os.replace(rows_tmp, rows_file)
    os.replace(manifest_tmp, manifest_file)
    loaded = load_overlay(manifest_file)
    return loaded.canonical_digest


def verify_overlay_artifacts(
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        *, source_root: str | Path | None = None,
        expectations_path: str | Path | None = None,
        ) -> str:
    """Run the full loader and return the deterministic bundle digest."""

    return load_overlay(
        manifest_path,
        source_root=source_root,
        expectations_path=expectations_path,
    ).canonical_digest


def _schema_artifact_bytes(
        model: type[BaseModel], artifact: Path,
        ) -> tuple[bytes, bytes, str]:
    schema_text = _canonical_json(model.model_json_schema(mode="validation"))
    schema_bytes = schema_text.encode("utf-8")
    digest = sha256(schema_bytes).hexdigest()
    digest_bytes = f"{digest}  {artifact.name}\n".encode("ascii")
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


def _overlay_schema_artifacts() -> tuple[tuple[Path, bytes], ...]:
    manifest, manifest_digest, _ = _schema_artifact_bytes(
        OverlayManifest, SCHEMA_ARTIFACT)
    row, row_digest, _ = _schema_artifact_bytes(OverlayRow, ROW_SCHEMA_ARTIFACT)
    return (
        (SCHEMA_ARTIFACT, manifest),
        (SCHEMA_DIGEST_ARTIFACT, manifest_digest),
        (ROW_SCHEMA_ARTIFACT, row),
        (ROW_SCHEMA_DIGEST_ARTIFACT, row_digest),
    )


def write_schema_artifacts() -> tuple[str, str]:
    """Regenerate the manifest/row schemas and their hash sidecars."""

    artifacts = _overlay_schema_artifacts()
    for path, payload in artifacts:
        _atomic_write(path, payload)
    return (
        _schema_artifact_bytes(OverlayManifest, SCHEMA_ARTIFACT)[2],
        _schema_artifact_bytes(OverlayRow, ROW_SCHEMA_ARTIFACT)[2],
    )


def verify_schema_artifacts() -> tuple[str, str]:
    """Require checked-in schema bytes and SHA256 sidecars to be exact."""

    for path, expected in _overlay_schema_artifacts():
        try:
            actual = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"overlay schema artifact가 없습니다: {path}") from exc
        if actual != expected:
            raise RuntimeError(f"overlay schema artifact drift: {path}")
    return (
        _schema_artifact_bytes(OverlayManifest, SCHEMA_ARTIFACT)[2],
        _schema_artifact_bytes(OverlayRow, ROW_SCHEMA_ARTIFACT)[2],
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write:
        digest = write_overlay_artifacts()
        schema_digests = write_schema_artifacts()
    elif args.check:
        digest = verify_overlay_artifacts()
        schema_digests = verify_schema_artifacts()
    else:
        parser.error("--write 또는 --check가 필요합니다")
    print(
        f"PASS: {MANIFEST_SCHEMA_VERSION} overlay_digest={digest} "
        f"manifest_schema_sha256={schema_digests[0]} "
        f"row_schema_sha256={schema_digests[1]}"
    )
    return 0


Stage1V1OverlayManifest = OverlayManifest
Stage1V1OverlayRow = OverlayRow

__all__ = [
    "ContractExpectation",
    "DecisionCode",
    "EXPECTED_SOURCE_FIXTURE_DIGESTS",
    "EXPECTATIONS_FILENAME",
    "LoadedOverlay",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "ROW_SCHEMA_ARTIFACT",
    "ROW_SCHEMA_DIGEST_ARTIFACT",
    "NormalizedIntentExpectation",
    "OverlayError",
    "OverlayManifest",
    "OverlayProvenance",
    "OverlayRow",
    "OVERLAY_ROOT",
    "R_A_002",
    "R_A_002_QUESTION",
    "ResolutionExpectation",
    "ROW_SCHEMA_VERSION",
    "SCHEMA_ARTIFACT",
    "SCHEMA_DIGEST_ARTIFACT",
    "SourceFixtureName",
    "SOURCE_FIXTURE_FILES",
    "SOURCE_ROOT",
    "Stage1V1OverlayManifest",
    "Stage1V1OverlayRow",
    "canonical_sha256",
    "expected_r_a_002_row",
    "load_overlay",
    "verify_overlay_artifacts",
    "verify_schema_artifacts",
    "write_overlay_artifacts",
    "write_schema_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(_main())
