"""산출물 신선도 계약 — 낡은 입력으로 빌드하지 않는다 (B-01 · B-03 · X-13).

## 문제

canonical 빌드는 상위 단계 산출물을 그냥 읽는다.

```text
exchange_events  →  out/exchange/relations.jsonl   ┐
form_events      →  out/forms/relations.jsonl      ├→  canonical build
periodic_relations → out/periodic/relations.jsonl  │
correction_items →  out/corrections/*.jsonl        ┘
```

이 파일들이 **언제 무엇으로 만들어졌는지** 아무도 확인하지 않았다. 실제로 이 세션에서
`out/corrections/*.jsonl` 이 파서 수정 이전 파일인 채로 빌드에 들어가, **구조적으로 틀린
정정 항목 796건**이 산출물에 그대로 실렸다. 리포트 숫자만 보면 정상이었다.

## 도장(stamp)

각 생산자가 산출 폴더에 `_stamp.json` 을 남긴다.

```json
{"producer": "exchange_events", "code_hash": "...", "manifest_hash": "...", "rows": 1270}
```

canonical 은 읽기 전에 `code_hash`·`manifest_hash` 가 자기 것과 같은지 본다.
다르면 **실패시킨다.** 조용히 낡은 것을 쓰는 것보다 멈추는 편이 낫다.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal

import pyarrow.parquet as pq

__all__ = [
    "STAMP_NAME", "CANONICAL_REQUIRED_ARTIFACTS", "ArtifactIssue",
    "ArtifactIntegrityError", "code_hash", "file_sha256", "write_stamp",
    "check_stamp", "validate_canonical_artifacts",
]

STAMP_NAME = "_stamp.json"
_SRC_ROOT = Path(__file__).resolve().parent

#: canonical build가 빈 코퍼스가 아닌 정상 평가 코퍼스에서 항상 발행하는 정본.
#: ``cells``처럼 선택 빌드인 것은 run에 기록됐을 때만 아래 검증 대상에 추가된다.
CANONICAL_REQUIRED_ARTIFACTS = (
    "source_files", "documents", "sections", "chunks", "facts", "fields",
    "relations", "correction_items", "event_identities", "event_observations",
)

ArtifactIssueCode = Literal[
    "missing_run", "invalid_run", "unpublished", "missing_run_field",
    "invalid_schema_version", "invalid_run_config", "missing_required_column",
    "missing_artifact_record", "invalid_artifact_name", "missing_artifact",
    "untracked_artifact", "hash_mismatch", "invalid_parquet", "row_count_mismatch",
    "missing_build_id", "mixed_build_id",
]


@dataclass(frozen=True)
class ArtifactIssue:
    """runtime artifact 검증 실패 한 건."""

    code: ArtifactIssueCode
    detail: str
    artifact: str | None = None


class ArtifactIntegrityError(RuntimeError):
    """발행본이 불완전하거나 서로 다른 build가 섞여 기동할 수 없다."""

    def __init__(self, root: str | Path, issues: list[ArtifactIssue]) -> None:
        self.root = Path(root)
        self.issues = tuple(issues)
        summary = "; ".join(
            f"{i.artifact + ': ' if i.artifact else ''}{i.code} ({i.detail})"
            for i in self.issues[:6])
        if len(self.issues) > 6:
            summary += f"; 외 {len(self.issues) - 6}건"
        super().__init__(f"canonical artifact 무결성 실패: {summary}")

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)


def code_hash(scope: str | None = None) -> str:
    """소스 해시. 주석 한 줄만 바뀌어도 값이 바뀐다.

    버전 문자열을 손으로 올리는 것에 의존하지 않는다 — 실제로 첨부 감사보고서 4,525 섹션을
    추가했을 때 어떤 버전 문자열도 안 바뀌어 **산출물이 달라졌는데 build_id 가 그대로**였다.

    `scope` 로 하위 디렉터리만 볼 수 있다. 상위 산출물의 신선도는 **그것을 만든 코드**
    (`ingest`)로만 판단해야 한다. `src` 전체로 보면 canonical 만 고쳐도 상위가 낡았다고
    나와, 의미 없는 재생성을 강요한다.
    """
    root = _SRC_ROOT / scope if scope else _SRC_ROOT
    h = hashlib.sha256()
    # **데이터 파일도 넣는다.** 계정 정규화 사전(`account_map.tsv`)처럼 코드 밖에 있지만
    # 산출물을 바꾸는 입력이 있다. 빼면 사전을 고쳐도 `build_id` 가 그대로다 —
    # 첨부 감사보고서 때 겪은 것과 같은 종류의 구멍이다.
    for path in sorted(list(root.rglob("*.py")) + list(root.rglob("*.tsv"))):
        if "__pycache__" in path.parts:
            continue
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:32]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:32]


#: 상위 산출물의 신선도를 판단하는 범위 — 그것을 만든 코드만 본다
PRODUCER_SCOPE = "ingest"


def write_stamp(out_dir: str | Path, producer: str, manifest_hash: str, **extra) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / STAMP_NAME
    path.write_text(json.dumps(
        {"producer": producer, "code_hash": code_hash(PRODUCER_SCOPE),
         "manifest_hash": manifest_hash, **extra},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def check_stamp(out_dir: str | Path, producer: str, manifest_hash: str,
                expect_code: str | None = None) -> str | None:
    """신선하면 `None`, 아니면 **이유 문자열**을 준다.

    이유를 돌려주는 이유는 호출 측이 실패 목록에 그대로 넣기 위해서다.
    참/거짓만 주면 「왜 막혔는지」를 다시 조사해야 한다.
    """
    path = Path(out_dir) / STAMP_NAME
    if not path.exists():
        return f"{producer}: 도장 없음 ({path}) — 상위 단계를 다시 실행하세요"
    try:
        stamp = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"{producer}: 도장 파손 ({exc})"
    want = expect_code or code_hash(PRODUCER_SCOPE)
    if stamp.get("code_hash") != want:
        return (f"{producer}: 코드가 바뀐 뒤 재생성되지 않음 "
                f"(도장 {str(stamp.get('code_hash'))[:12]} != 현재 {want[:12]})")
    if stamp.get("manifest_hash") != manifest_hash:
        return (f"{producer}: 다른 코퍼스로 만들어짐 "
                f"(도장 {str(stamp.get('manifest_hash'))[:12]} != 현재 {manifest_hash[:12]})")
    return None


_ARTIFACT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_SCHEMA_VERSION = re.compile(r"^[0-9]+\.[0-9]+$", flags=re.ASCII)

_SCHEMA_1_8_SAFE_CONFIG = {
    "default_llm_text_column": "chunks.text_prompt_safe",
    "default_search_text_column": "chunks.text_search",
    "default_field_value_column": "fields.value_prompt_safe",
    "default_evidence_excerpt_column": "evidence.excerpt_safe",
}

_SCHEMA_1_9_EVIDENCE_CONFIG = {
    "chunk_evidence_kind": "chunk_text",
    "event_support_version": "event-support/1.0",
    "relation_support_version": "relation-support/1.1-shared-side",
}

# schema 1.8부터 조회 계층이 실제로 의존하는 보안·근거·원문 좌표 컬럼이다. run.json의
# schema_version만 올리고 예전 Parquet를 재사용하면 Python 객체를 만들 때까지 실패가
# 늦춰지거나, 더 나쁘게는 안전 projection/정정 side 좌표 없이 값이 반환될 수 있다.
# 전체 dataclass를 중복 선언하지 않고 런타임 안전성에 필요한 물리 컬럼만 기동 시 확인한다.
_SCHEMA_1_8_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "documents": frozenset({
        "doc_id", "rcept_no", "corp_code", "corp_name", "listed_name",
        "doc_group", "doc_subtype", "event_type", "report_nm", "rcept_dt",
        "is_correction", "fact_extract_status", "fact_unsupported_scope",
        "primary_source_file_id", "alternate_source_file_ids",
        "source_cross_check_status", "source_cross_check_reason",
        "source_cross_check_source_file_ids",
    }),
    "relations": frozenset({
        "relation_id", "src_kind", "src_id", "src_rcept_no", "dst_kind",
        "dst_id", "dst_rcept_no", "relation_type", "resolution_status",
        "src_rcept_dt", "resolver_version", "candidate_ids", "match_features",
        "root_missing_reason",
    }),
    "chunks": frozenset({
        "doc_id", "text", "text_search", "text_prompt_safe", "security_flags", "pii_types",
        "security_policy_version",
    }),
    "evidence": frozenset({
        "evidence_id", "doc_id", "source_file_id", "kind", "locator",
        "table_locator", "logical_row", "logical_col", "excerpt_raw",
        "excerpt_safe", "excerpt_hash", "raw_access", "extraction_method",
        "extraction_status", "security_flags", "pii_types",
        "security_policy_version", "evidence_policy_version", "rcept_dt",
    }),
    "facts": frozenset({
        "doc_id", "source_file_id", "locator", "table_locator", "logical_row",
        "logical_col", "evidence_id", "value_text", "rcept_dt",
    }),
    "fields": frozenset({
        "doc_id", "rcept_dt", "source_file_id", "path", "locator",
        "table_locator", "evidence_id",
        "value_raw", "value_masked", "value_prompt_safe", "logical_row",
        "logical_col", "pii_type", "is_pii", "security_policy_version",
    }),
    "correction_items": frozenset({
        "doc_id", "rcept_dt", "corp_name", "path", "value_before",
        "value_after", "source_file_id", "locator", "before_locator", "after_locator",
        "logical_row", "logical_col",
        "before_logical_row", "before_logical_col", "after_logical_row",
        "after_logical_col", "before_evidence_id", "after_evidence_id",
    }),
    "event_identities": frozenset({
        "identity_fingerprint", "identity_status", "resolver_version",
        "status_at_corpus_end",
    }),
    "event_observations": frozenset({
        "is_termination", "previous_observation_rcept_no",
    }),
    "source_files": frozenset({
        "source_file_id", "doc_id", "normalized_path_key", "actual_relpath", "source_selection_role",
        "parse_usable", "coverage_n_sections", "coverage_n_chars",
        "coverage_n_tables", "coverage_n_pages", "coverage_pages_with_text",
        "coverage_n_lines", "page_text_coverage", "active_nodes_removed",
        "active_attributes_removed", "locator_kind", "locator_limitations",
        "parse_warnings", "parse_error",
    }),
}

# 1.8 호환 계약에 새 열을 섞지 않는다. 1.9를 선언한 발행본만 Chunk FK와
# Event/Relation support를 필수로 검사한다.
_SCHEMA_1_9_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "chunks": frozenset({
        "evidence_id", "block_id", "doc_id", "source_file_id", "locator",
        "rcept_dt", "part_no", "n_parts", "text", "text_prompt_safe",
    }),
    "fields": frozenset({"value_status"}),
    "event_identities": frozenset({
        "event_key", "kind", "corp_code", "corp_name", "doc_group",
        "doc_subtype", "root_rcept_no", "identity_fingerprint",
        "identity_status", "resolver_version", "first_disclosed_at",
        "last_disclosed_at", "n_observations", "n_corrections",
        "status_at_corpus_end",
    }),
    "event_observations": frozenset({
        "event_key", "seq", "doc_id", "rcept_no", "observed_at",
        "is_correction", "is_termination",
        "supporting_evidence_ids", "support_roles", "support_status",
        "support_version", "support_limitation",
    }),
    "relations": frozenset({
        "relation_id", "src_id", "src_rcept_no", "src_rcept_dt",
        "supporting_evidence_ids", "support_roles", "support_status",
        "support_version", "support_limitation",
    }),
}


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return repr(value)
    return str(value)


def _parquet_has_only_build(pf: pq.ParquetFile, build_id: str) -> tuple[bool, str]:
    """row-group 통계를 우선 쓰고, 통계가 없을 때만 build_id 열을 읽는다.

    현재 472MB 발행본은 모든 row group에 문자열 min/max가 있어 파일 본문을 다시
    펼치지 않는다. 제3자 Parquet처럼 통계가 빠진 파일도 검증을 생략하지 않고 그때만
    단일 열을 스트리밍한다.
    """
    names = pf.schema_arrow.names
    if "build_id" not in names:
        return False, "build_id 컬럼 없음"
    if pf.metadata.num_rows == 0:
        return True, "빈 artifact"
    index = names.index("build_id")
    needs_scan = False
    for row_group in range(pf.metadata.num_row_groups):
        statistics = pf.metadata.row_group(row_group).column(index).statistics
        if statistics is None or not statistics.has_min_max:
            needs_scan = True
            break
        if statistics.null_count:
            return False, f"row group {row_group}: build_id null {statistics.null_count}건"
        low, high = _as_text(statistics.min), _as_text(statistics.max)
        if low != build_id or high != build_id:
            return False, f"row group {row_group}: [{low!r}, {high!r}] != {build_id!r}"
    if not needs_scan:
        return True, "row-group statistics"
    for batch_no, batch in enumerate(pf.iter_batches(
            columns=["build_id"], batch_size=100_000)):
        values = set(batch.column(0).to_pylist())
        if values != {build_id}:
            sample = sorted(repr(value) for value in values)[:4]
            return False, f"batch {batch_no}: {sample} != {build_id!r}"
    return True, "column scan"


def validate_canonical_artifacts(
        root: str | Path,
        required: tuple[str, ...] = CANONICAL_REQUIRED_ARTIFACTS) -> dict:
    """발행된 canonical 디렉터리의 self-contained runtime 계약을 검증한다.

    schema 1.7 이하의 legacy 정본은 ``run.json``이 선언한 hash·행 수·build ID 계약으로
    계속 읽는다. 1.8부터는 안전 projection·근거·원문 좌표에 필요한 핵심 물리 컬럼도
    함께 확인한다. 새 코드가 우연히 옛/새 파일을 섞거나 버전 문자열만 올린 정본을
    읽는 것은 시작 단계에서 막는다.

    검증 비용은 파일 SHA-256 순차 읽기 + Parquet footer 확인이다. build ID는 row-group
    통계가 있으면 본문 열을 읽지 않으며, 통계가 없는 파일만 한 열을 스트리밍한다.
    """
    base = Path(root)
    run_path = base / "run.json"
    if not run_path.is_file():
        raise ArtifactIntegrityError(base, [
            ArtifactIssue("missing_run", f"파일 없음: {run_path}")])
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(base, [
            ArtifactIssue("invalid_run", f"{type(exc).__name__}: {exc}")]) from exc
    if not isinstance(run, dict):
        raise ArtifactIntegrityError(base, [
            ArtifactIssue("invalid_run", "JSON object가 아님")])

    issues: list[ArtifactIssue] = []
    if run.get("published") is not True:
        issues.append(ArtifactIssue(
            "unpublished", f"published={run.get('published')!r}"))
    build_id = run.get("build_id")
    if not isinstance(build_id, str) or not build_id:
        issues.append(ArtifactIssue("missing_run_field", "build_id 없음"))
    schema_version = run.get("schema_version")
    if schema_version is None or schema_version == "":
        issues.append(ArtifactIssue("missing_run_field", "schema_version 없음"))
        version: tuple[int, ...] = ()
    elif (not isinstance(schema_version, str)
            or _SCHEMA_VERSION.fullmatch(schema_version) is None):
        issues.append(ArtifactIssue(
            "invalid_schema_version",
            f"ASCII major.minor 형식이 아님: {schema_version!r}"))
        version = ()
    else:
        try:
            version = tuple(int(part) for part in schema_version.split("."))
        except ValueError:
            # 비정상적으로 긴 숫자는 Python의 정수 변환 한도를 넘을 수 있다. 파서
            # 예외를 밖으로 새지 않고 손상된 runtime 계약으로 함께 보고한다.
            issues.append(ArtifactIssue(
                "invalid_schema_version",
                f"major.minor 숫자를 해석할 수 없음: {schema_version!r}"))
            version = ()
    if version >= (1, 8):
        config = run.get("config")
        if not isinstance(config, dict):
            issues.append(ArtifactIssue(
                "invalid_run_config",
                f"schema {schema_version} config가 object가 아님: {type(config).__name__}"))
        else:
            for key, expected in _SCHEMA_1_8_SAFE_CONFIG.items():
                if key not in config:
                    issues.append(ArtifactIssue(
                        "invalid_run_config", f"config 필수 키 없음: {key}"))
                elif config[key] != expected:
                    actual = repr(config[key])
                    if len(actual) > 120:
                        actual = actual[:117] + "..."
                    issues.append(ArtifactIssue(
                        "invalid_run_config",
                        f"config.{key}={actual}, 필수값={expected!r}"))
    if version >= (1, 9):
        config = run.get("config")
        if not isinstance(config, dict):
            # 1.8 branch가 이미 같은 구조 결함을 기록한다.
            pass
        else:
            for key, expected in _SCHEMA_1_9_EVIDENCE_CONFIG.items():
                if key not in config:
                    issues.append(ArtifactIssue(
                        "invalid_run_config", f"config 필수 키 없음: {key}"))
                elif config[key] != expected:
                    actual = repr(config[key])
                    if len(actual) > 120:
                        actual = actual[:117] + "..."
                    issues.append(ArtifactIssue(
                        "invalid_run_config",
                        f"config.{key}={actual}, 필수값={expected!r}"))
    counts = run.get("counts")
    hashes = run.get("artifact_hashes")
    if not isinstance(counts, dict):
        issues.append(ArtifactIssue("missing_run_field", "counts object 없음"))
        counts = {}
    if not isinstance(hashes, dict):
        issues.append(ArtifactIssue("missing_run_field", "artifact_hashes object 없음"))
        hashes = {}

    # Evidence는 schema 1.5에서 first-class 정본으로 들어왔다. 옛 1.2 발행본은
    # self-contract로 계속 열 수 있지만, 1.5+가 evidence.parquet를 누락하고도 기동되는
    # 것은 허용하지 않는다. 그렇지 않으면 Fact/Field가 값은 반환하면서 검증된 인용은
    # 영원히 만들 수 없는 반쪽 발행본이 정상으로 보인다.
    effective_required = list(required)
    if version >= (1, 5) and "evidence" not in effective_required:
        effective_required.append("evidence")

    for name in effective_required:
        missing = []
        if name not in counts:
            missing.append("counts")
        if name not in hashes:
            missing.append("artifact_hashes")
        if missing:
            issues.append(ArtifactIssue(
                "missing_artifact_record", f"run.json의 {', '.join(missing)}에 없음", name))

    declared = set(counts) | set(hashes)
    disk_names = {path.stem for path in base.glob("*.parquet")}
    for name in sorted(disk_names - declared):
        issues.append(ArtifactIssue(
            "untracked_artifact", "run.json에 기록되지 않은 Parquet", name))

    # 잘못된 run metadata가 있어도 가능한 문제를 한 번에 보고하되, 경로 탈출 이름은
    # 절대 파일 경로로 만들지 않는다.
    for name in sorted(declared):
        if not isinstance(name, str) or _ARTIFACT_NAME.fullmatch(name) is None:
            issues.append(ArtifactIssue(
                "invalid_artifact_name", f"허용되지 않는 이름: {name!r}"))
            continue
        path = base / f"{name}.parquet"
        if not path.is_file():
            issues.append(ArtifactIssue("missing_artifact", f"파일 없음: {path}", name))
            continue

        expected_hash = hashes.get(name)
        if not isinstance(expected_hash, str) or not expected_hash:
            if name in hashes:
                issues.append(ArtifactIssue(
                    "missing_artifact_record", "유효한 hash 문자열이 아님", name))
        else:
            actual_hash = file_sha256(path)
            if actual_hash != expected_hash:
                issues.append(ArtifactIssue(
                    "hash_mismatch",
                    f"run {expected_hash} != file {actual_hash}", name))

        try:
            pf = pq.ParquetFile(path)
        except Exception as exc:  # ArrowInvalid 등 버전별 예외형을 밖으로 새지 않는다
            issues.append(ArtifactIssue(
                "invalid_parquet", f"{type(exc).__name__}: {exc}", name))
            continue
        if version >= (1, 8):
            required_columns = _SCHEMA_1_8_REQUIRED_COLUMNS.get(name, frozenset())
            missing_columns = sorted(required_columns - set(pf.schema_arrow.names))
            if missing_columns:
                issues.append(ArtifactIssue(
                    "missing_required_column",
                    f"schema {schema_version} 필수 컬럼 없음: {', '.join(missing_columns)}",
                    name))
        if version >= (1, 9):
            required_columns = _SCHEMA_1_9_REQUIRED_COLUMNS.get(name, frozenset())
            missing_columns = sorted(required_columns - set(pf.schema_arrow.names))
            if missing_columns:
                issues.append(ArtifactIssue(
                    "missing_required_column",
                    f"schema {schema_version} 필수 컬럼 없음: {', '.join(missing_columns)}",
                    name))
        expected_count = counts.get(name)
        if (not isinstance(expected_count, int) or isinstance(expected_count, bool)
                or expected_count < 0):
            if name in counts:
                issues.append(ArtifactIssue(
                    "missing_artifact_record", f"유효한 row count가 아님: {expected_count!r}",
                    name))
        elif pf.metadata.num_rows != expected_count:
            issues.append(ArtifactIssue(
                "row_count_mismatch",
                f"run {expected_count:,} != parquet {pf.metadata.num_rows:,}", name))
        if isinstance(build_id, str) and build_id:
            same_build, detail = _parquet_has_only_build(pf, build_id)
            if not same_build:
                code = "missing_build_id" if "컬럼 없음" in detail else "mixed_build_id"
                issues.append(ArtifactIssue(code, detail, name))

    if issues:
        raise ArtifactIntegrityError(base, issues)
    return run
