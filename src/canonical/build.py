"""canonical 통합 빌드 — 전 문서군을 하나의 스키마로 적재한다.

지금까지 문서군마다 산출물 스키마가 갈려 있었다(정기 16필드 · 거래소 21 · 주요사항 17,
공통은 6개뿐이고 `rcept_dt` vs `disclosed_at` 처럼 이름도 달랐다). 이 빌더가 그것을
`src/canonical/schema.py` 계약 하나로 통일한다.

산출물 (`out/canonical/`):

```text
run.json          빌드 1건 — build_id · 버전 · 산출물 해시
source_files      4,907  본문 + 첨부 감사보고서 410 포함, sha256·format·encoding
documents         4,204  manifest + event_type
sections          정기공시 목차 단위
fields            정형 3종 라벨경로 → 값 (ACODE 포함)
relations         정정·관련 edge (상태 5종)
```

`Cell` 은 span 전파(P0-3)가 끝나 스키마상 적재 가능하지만 29,635,003 행이라
별도 실행으로 분리한다(`--with-cells`).

**모든 산출물이 같은 `build_id` 를 갖는다.** 서버는 불일치 시 기동을 실패시켜야 한다.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, replace
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Mapping, get_type_hints

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shutil

from lxml import html as lxml_html

from ..artifact import (
    ArtifactIntegrityError, check_stamp, code_hash as _pipeline_code_hash,
    file_sha256, validate_canonical_artifacts,
)

from ..ingest.corpus_paths import CorpusIndex, iter_manifest, nfc
from ..ingest.dart_form import extract_document as extract_form
from ..ingest.dart_xml import (ParseFailure, UnsupportedFormat, parse_file,
                               parse_root, parse_tree)
from ..ingest.exchange_html import cell_text as exchange_cell_text
from ..ingest.exchange_html import parse_file as parse_exchange
from ..ingest.financials import (
    ACCOUNT_MAP_PATH,
    extract_financials_detailed,
    parse_number,
)
from ..ingest.values import classify as classify_value_status
from ..ingest.pdf_html import (
    CrossCheckStatus,
    ParsedViewer,
    SourceParseResult,
    parse_pdf_html,
)
from ..ingest.relation_version import RELATION_RESOLVER_VERSION
from ..ingest.table_grid import build_grid
from .schema import (
    LOCATABLE_COLUMNS, SCHEMA_VERSION, Cell, Chunk, CorrectionItem, Document, Evidence, Field,
    ID_HEX, EventIdentity, EventObservation, Fact, Relation, Run, Section, SourceFile,
    as_row, block_id, child_locator, source_file_id,
)
from .chunker import CHUNKER_VERSION as CHUNK_VER
from .chunker import split_section
from .events import (
    EVENT_SUPPORT_ROLES, EVENT_SUPPORT_VERSION, RELATION_SUPPORT_ROLES,
    RELATION_SUPPORT_VERSION,
    add_identity_field, build_identities, event_support_decision,
    event_support_role, filter_relation_correction_support_pairs,
    is_declared_termination, normalise_relation_support_pairs,
    relation_correction_support_owner_keys, relation_correction_support_sides,
    relation_support_limitation, validate_event_history,
)
from .lineage import derive_supersedes
from .evidence import (
    EVIDENCE_POLICY_VERSION, make_evidence, table_locator_of,
)
from .security import (SECURITY_POLICY_VERSION, holding_party_types_by_row,
                       project_chunk_text, project_field_value,
                       resolve_party_type, subject_names_by_row)
from .quality import (
    policy_path_label,
    QUALITY_POLICY_PATH, QUALITY_POLICY_VERSION, evaluate_quality_decisions,
)

__all__ = ["build"]

#: 이 단계에서 실패하면 **산출물을 교체하지 않는다.** 나머지(첨부 파싱 실패, Fact 추출 실패)는
#: 부분 손실이라 이전 빌드를 덮어쓰지 않을 만큼 치명적이지는 않다.
_BLOCKING_STAGES = frozenset({
    "parse", "artifact", "stale_input", "decode", "quality_gate",
})

PARSER_VERSION = "dart_xml/1.2+table_grid"
SANITIZER_VERSION = "sanitize/1.0-whitelist34"
CHUNKER_VERSION = CHUNK_VER
RESOLVER_VERSION = RELATION_RESOLVER_VERSION

_TAG = re.compile(r"^\[[^\]]+\]")
_PAREN = re.compile(r"\(([^)]+)\)\s*$")
_OMIT = re.compile(r"해당\s*사항\s*(이)?\s*없|기재\s*생략|해당없음")
_BATCH = 2000


#: 정기공시는 괄호 안이 기준기간이다 — `사업보고서 (2023.12)`.
#: 유형으로 저장하면 1,054건 전부 `"2023.12"` 같은 값이 되어 라우팅이 깨진다.
_PERIOD_LIKE = re.compile(r"^\s*\d{4}[.\-/]\d{1,2}\s*$")

def _event_type(report_nm: str, doc_group: str) -> str | None:
    """`주요사항보고서(유상증자결정)` → `유상증자결정`.

    정기공시·거래소공시는 `doc_subtype` 이 이미 유형을 담으므로 `None` 을 돌려준다.
    괄호 안이 기간 표기면 유형이 아니므로 버린다.
    """
    if doc_group not in ("major", "holding"):
        return None
    base = _TAG.sub("", report_nm).strip()
    m = _PAREN.search(base)
    if not m:
        return None
    value = m.group(1).strip()
    return None if _PERIOD_LIKE.match(value) else value


def _correction_side_locators(item: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """schema 1.8 정정 값 좌표 계약. legacy 대표 locator fallback은 금지한다."""
    before_locator = item.get("before_locator")
    after_locator = item.get("after_locator")
    missing = [
        side for side, value, side_locator in (
            ("before", item.get("value_before"), before_locator),
            ("after", item.get("value_after"), after_locator),
        ) if value is not None and not side_locator
    ]
    if missing:
        raise ValueError(
            "CorrectionItem side locator 없음: " + ",".join(missing)
            + " (out/corrections를 schema 1.8 extractor로 재생성 필요)"
        )
    return before_locator, after_locator


def _semantic_pair(item: Mapping[str, Any], row_key: str, col_key: str, *,
                   required: bool, label: str) -> tuple[int | None, int | None]:
    """JSON semantic 좌표 한 쌍을 typed/fail-closed로 읽는다."""
    row, col = item.get(row_key), item.get(col_key)
    if row is None and col is None and not required:
        return None, None
    if (type(row) is not int or type(col) is not int or row < 0 or col < 0):
        raise ValueError(
            f"CorrectionItem {label} semantic 좌표 없음/오류: {row!r},{col!r} "
            "(out/corrections를 schema 1.8 extractor로 재생성 필요)"
        )
    return row, col


def _correction_semantic_coordinates(
        item: Mapping[str, Any],
        before_locator: str | None,
        after_locator: str | None,
        ) -> tuple[
            tuple[int | None, int | None],
            tuple[int | None, int | None],
            tuple[int | None, int | None]]:
    """대표·before·after의 parser semantic 좌표 계약을 검증한다.

    대표 합성 ROW locator에만 null 좌표를 허용한다. side 셀은 값이 null이어도
    locator가 존재하면 원문 위치가 있으므로 semantic 좌표도 반드시 보존한다.
    """
    locator = str(item.get("locator") or "")
    row_pair = _semantic_pair(
        item, "logical_row", "logical_col",
        required="/ROW[" not in locator, label="row")
    if "/ROW[" in locator and row_pair != (None, None):
        raise ValueError("CorrectionItem 합성 ROW 대표에는 semantic 셀 좌표를 둘 수 없음")

    pairs = []
    for side, side_locator in (("before", before_locator), ("after", after_locator)):
        pair = _semantic_pair(
            item, f"{side}_logical_row", f"{side}_logical_col",
            required=side_locator is not None, label=side)
        if side_locator is None and pair != (None, None):
            raise ValueError(f"CorrectionItem {side} locator 없이 semantic 좌표만 존재")
        pairs.append(pair)
    return row_pair, pairs[0], pairs[1]


#: artifact·상위 도장용 128-bit 해시는 `artifact.file_sha256` 하나만 쓴다. 아래 64자리
#: digest는 source quality 승인 guard에만 쓰며 SourceFile/도장 계약과 섞지 않는다.
_sha256 = file_sha256


def _full_sha256(path: Path) -> str:
    """원문 승인 guard용 축약하지 않은 SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _detect(path: Path) -> tuple[str, str, str]:
    """(format, encoding, decode_status). 확장자가 아니라 내용으로 판별한다."""
    raw = path.read_bytes()
    if raw.startswith(b"%PDF"):
        return "pdf", "binary", "ok"
    try:
        text = raw.decode("utf-8")
        status = "ok"
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        status = "replaced"
    head = text[:4096].lstrip().lower()
    if head.startswith("<!doctype html") or "<html" in head[:512]:
        fmt = "html"          # 거래소공시 — meta 는 euc-kr 이지만 실제 UTF-8
    elif "<document" in head or head.startswith("<?xml"):
        fmt = "dart_xml"
    else:
        fmt = "unknown"
    return fmt, "utf-8", status


def _arrow_type(annotation: Any) -> pa.DataType:
    text = str(annotation)
    if "list[str]" in text:
        return pa.list_(pa.string())
    if "dict[str, int]" in text:
        return pa.map_(pa.string(), pa.int64())
    if "dict" in text:
        return pa.map_(pa.string(), pa.string())
    if "bool" in text:
        return pa.bool_()
    if "float" in text:
        return pa.float64()
    if "int" in text:
        return pa.int64()
    return pa.string()


def _arrow_schema(cls: type) -> pa.Schema:
    """dataclass 에서 Arrow 스키마를 도출한다.

    추론에 맡기면 배치 전체가 `None` 인 컬럼(`aunit` 등)이 `null` 타입이 되어
    다음 배치와 스키마가 어긋난다. **컬럼 타입은 계약의 일부**이므로 명시한다.
    """
    order = list(LOCATABLE_COLUMNS)
    hints = get_type_hints(cls)
    names = [n for n in order if n in hints] + [n for n in hints if n not in order]
    return pa.schema([(n, _arrow_type(hints[n])) for n in names])


class _Writer:
    """행을 모았다가 배치로 Parquet 에 쓴다. 854MB 를 메모리에 올리지 않기 위해서다.

    ``sort_key`` 를 주면 닫을 때 한 번 **다시 써서** 그 열로 정렬한다. 조회가
    ``evidence_id`` 로 필터하는 산출물에 필요하다 — 해시는 자연 순서가 없어서
    문서 순서로 기록하면 모든 row group 의 min/max 가 ``000…``~``fff…`` 전 범위를
    덮고, **필터가 row group 을 하나도 잘라내지 못한다.** 그때 ``lookup()`` 한 번이
    약 1.25GB 를 훑어 2.2초가 됐다(정렬 후 0.099초).

    정렬도 첫 16진수별 run으로 분할해 run 하나만 메모리에 올린다. 따라서
    가장 큰 artifact 전체가 아니라 대략 1/16과 병합 버퍼가 메모리 상한이다.
    """

    def __init__(self, path: Path, schema: pa.Schema, *,
                 sort_key: str | None = None) -> None:
        self.path = path
        self.schema = schema
        self.sort_key = sort_key
        if sort_key is not None and sort_key not in schema.names:
            raise ValueError(f"{path.name}: 정렬 키 {sort_key} 가 스키마에 없습니다")
        self._rows: list[dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None
        self.count = 0

    def add(self, row: dict[str, Any]) -> None:
        self._rows.append(row)
        self.count += 1
        if len(self._rows) >= _BATCH:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows, schema=self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self._writer.write_table(table)
        self._rows.clear()

    def close(self) -> str | None:
        self._flush()
        if self._writer is None:
            # 0행도 정본 artifact다. 없으면 runtime 필수 스키마를 검증할 수 없다.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self._writer.close()
        if self.sort_key is not None:
            self._sort_in_place()
        # **해시는 정렬 뒤에 낸다.** 먼저 내면 run.json 이 교체 전 파일을 가리켜
        # 정본이 hash_mismatch 로 열리지 않는다.
        return _sha256(self.path)

    def _sort_in_place(self) -> None:
        """기록이 끝난 파일을 bounded external sort로 정렬한다.

        ``pq.read_table(path).sort_by(...)``는 압축 630MB인 chunks에서도
        Arrow 복사본을 함께 유지해 RSS 8.5GB를 사용했고 OOM으로
        죽었다. Evidence ID는 16진수이므로 첫 문자 16개로 버킷하면
        버킷 순서 자체가 전역 정렬 순서다. 원본을 배치별로 버킷에
        나누고, 버킷 하나만 메모리에서 정렬·재독한 뒤 순차 병합한다.
        """

        source = pq.ParquetFile(self.path)
        rows, groups = source.metadata.num_rows, source.metadata.num_row_groups
        if rows == 0:
            return
        # 원본 row group 크기를 유지한다. 크기를 바꾸면 pruning 효과와 메모리
        # 사용량이 함께 달라져 이 변경의 효과를 귀속할 수 없다.
        group_size = max(1, rows // groups) if groups else _BATCH
        target = self.path.with_suffix(".parquet.sorting")
        run_dir = self.path.with_suffix(".parquet.sorting.runs")
        target.unlink(missing_ok=True)
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True)
        alphabet = "0123456789abcdef"
        bucket_writers: dict[str, pq.ParquetWriter] = {}
        bucket_input_rows = {prefix: 0 for prefix in alphabet}
        sorted_paths: list[Path] = []
        target_writer: pq.ParquetWriter | None = None
        published = False
        try:
            # 1) 원본을 스트리밍하며 첫 hex 문자로 분할한다.
            for batch in source.iter_batches(
                    batch_size=max(group_size, 20_000)):
                table = pa.Table.from_batches([batch], schema=self.schema)
                keys = table[self.sort_key].to_pylist()
                malformed = sum(
                    not isinstance(value, str)
                    or re.fullmatch(r"[0-9a-f]{32}", value, flags=re.ASCII) is None
                    for value in keys
                )
                if malformed:
                    raise ValueError(
                        f"{self.path.name}: {self.sort_key} 형식 오류 "
                        f"{malformed:,}건 (소문자 32자리 hex 필요)")
                prefixes = pc.utf8_slice_codeunits(
                    table[self.sort_key], start=0, stop=1)
                present = set(prefixes.to_pylist())
                invalid = present - set(alphabet)
                if invalid:
                    raise ValueError(
                        f"{self.path.name}: {self.sort_key} non-hex prefix "
                        f"{sorted(invalid, key=str)}")
                for prefix in sorted(present):
                    part = table.filter(pc.equal(prefixes, prefix))
                    if part.num_rows == 0:
                        continue
                    bucket_input_rows[prefix] += part.num_rows
                    writer = bucket_writers.get(prefix)
                    if writer is None:
                        path = run_dir / f"{prefix}.unsorted.parquet"
                        writer = pq.ParquetWriter(
                            path, self.schema, compression="zstd")
                        bucket_writers[prefix] = writer
                    writer.write_table(part, row_group_size=group_size)
            for writer in bucket_writers.values():
                writer.close()
            bucket_writers.clear()

            # 2) 한 버킷씩만 메모리에 올려 정렬하고 즉시 재독
            # 검증한다. 최종 병합은 검증된 run만 읽는다.
            for prefix in alphabet:
                unsorted_path = run_dir / f"{prefix}.unsorted.parquet"
                if not unsorted_path.exists():
                    continue
                table = pq.read_table(unsorted_path).sort_by(
                    [(self.sort_key, "ascending")])
                sorted_path = run_dir / f"{prefix}.sorted.parquet"
                pq.write_table(table, sorted_path,
                               row_group_size=group_size,
                               compression="zstd")
                written = pq.read_table(sorted_path)
                if (written.num_rows != bucket_input_rows[prefix]
                        or not written.equals(table)):
                    raise ValueError(
                        f"{self.path.name}: bucket {prefix} 정렬 결과 불일치")
                sorted_paths.append(sorted_path)
                unsorted_path.unlink()
                del table, written

            # 3) prefix 순서로 이어 쓰면 전역 sort order가 된다.
            target_writer = pq.ParquetWriter(
                target, self.schema, compression="zstd")
            merged_rows = 0
            for sorted_path in sorted_paths:
                bucket = pq.ParquetFile(sorted_path)
                for batch in bucket.iter_batches(batch_size=group_size):
                    target_writer.write_batch(
                        batch, row_group_size=group_size)
                    merged_rows += batch.num_rows
            target_writer.close()
            target_writer = None

            written_file = pq.ParquetFile(target)
            if (merged_rows != rows
                    or written_file.metadata.num_rows != rows
                    or written_file.schema_arrow != self.schema):
                raise ValueError(
                    f"{self.path.name}: external sort row/schema 불일치")

            # Row count만으로는 병합 순서 결함을 잡지 못한다.
            # key 한 열만 bounded batch로 읽어 전역 단조증가를 확인한다.
            previous = None
            checked = 0
            for batch in written_file.iter_batches(
                    columns=[self.sort_key], batch_size=100_000):
                values = batch.column(0).to_pylist()
                if any(value is None for value in values):
                    raise ValueError(
                        f"{self.path.name}: {self.sort_key} null")
                if previous is not None and values and previous > values[0]:
                    raise ValueError(
                        f"{self.path.name}: bucket 경계 정렬 오류")
                if any(left > right for left, right in zip(values, values[1:])):
                    raise ValueError(
                        f"{self.path.name}: batch 내 정렬 오류")
                if values:
                    previous = values[-1]
                checked += len(values)
            if checked != rows:
                raise ValueError(
                    f"{self.path.name}: key 검증 행 수 {checked} != {rows}")

            # 런타임 point lookup의 pruning 계약도 함께 검증한다. 전역 정렬만
            # 맞고 row-group statistics가 빠지면 기능은 맞아도 조회가 다시
            # 전 파일 scan으로 퇴행한다.
            key_index = written_file.schema_arrow.names.index(self.sort_key)
            previous_max = None
            for group_index in range(written_file.metadata.num_row_groups):
                statistics = written_file.metadata.row_group(
                    group_index).column(key_index).statistics
                if (statistics is None or statistics.min is None
                        or statistics.max is None):
                    raise ValueError(
                        f"{self.path.name}: row group {group_index} "
                        f"{self.sort_key} 통계 없음")
                if (previous_max is not None
                        and previous_max > statistics.min):
                    raise ValueError(
                        f"{self.path.name}: row group {group_index} 범위 겹침")
                previous_max = statistics.max

            os.replace(target, self.path)
            published = True
        finally:
            for writer in bucket_writers.values():
                writer.close()
            if target_writer is not None:
                target_writer.close()
            if not published:
                target.unlink(missing_ok=True)
            shutil.rmtree(run_dir, ignore_errors=True)


def _validate_evidence_id_uniqueness(path: Path) -> int:
    """Evidence ID 전역 유일성을 메모리 상한이 있는 방식으로 검증한다.

    Evidence는 200만 행을 넘으므로 Python ``set[str]`` 전수 적재에 의존하지 않는다.
    staging 디렉터리의 임시 SQLite primary key에 ID만 배치 삽입해 Fact·Field·Correction
    writer 전체를 가로지르는 중복을 발행 전에 잡는다. 임시 DB는 성공·실패 모두 지운다.
    """
    parquet = pq.ParquetFile(path)
    if "evidence_id" not in parquet.schema_arrow.names:
        raise ValueError("evidence.parquet에 evidence_id 컬럼이 없습니다")

    db_path = path.with_name(".evidence-id-uniqueness.sqlite3")
    sidecars = tuple(Path(str(db_path) + suffix)
                     for suffix in ("", "-journal", "-wal", "-shm"))
    for candidate in sidecars:
        candidate.unlink(missing_ok=True)

    connection: sqlite3.Connection | None = None
    total = duplicates = 0
    try:
        connection = sqlite3.connect(db_path)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        connection.execute(
            "CREATE TABLE evidence_ids ("
            "evidence_id TEXT NOT NULL PRIMARY KEY) WITHOUT ROWID")
        connection.execute("BEGIN IMMEDIATE")
        for batch in parquet.iter_batches(
                batch_size=100_000, columns=["evidence_id"]):
            values = batch.column(0).to_pylist()
            malformed = sum(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{32}", value, flags=re.ASCII) is None
                for value in values
            )
            if malformed:
                raise ValueError(
                    f"Evidence ID 형식 오류: {malformed:,}건 (batch rows={len(values):,})")
            before = connection.total_changes
            connection.executemany(
                "INSERT OR IGNORE INTO evidence_ids(evidence_id) VALUES (?)",
                ((value,) for value in values),
            )
            inserted = connection.total_changes - before
            total += len(values)
            duplicates += len(values) - inserted
        connection.commit()
        if duplicates:
            raise ValueError(
                f"Evidence ID 전역 중복: {duplicates:,}건 (rows={total:,})")
        return total
    except sqlite3.Error as exc:
        raise RuntimeError(f"Evidence ID 전역 유일성 검사 실패: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
        for candidate in sidecars:
            candidate.unlink(missing_ok=True)


def _validate_schema19_evidence_contract(root: Path, build_id: str) -> None:
    """schema 1.9의 새 Chunk/support 참조를 발행 직전에 독립 재검증한다.

    Evidence 전체 raw를 메모리에 쌓지 않는다. Chunk는 raw/safe digest만, support는
    ID별 owner expectation만 보관하고 Evidence Parquet를 한 번 스트리밍한다.
    """
    document_columns = [
        "doc_id", "rcept_no", "rcept_dt", "corp_name", "listed_name", "doc_group",
        "doc_subtype", "event_type", "report_nm", "is_correction",
    ]
    documents = {
        str(row["doc_id"]): row for row in pq.read_table(
            root / "documents.parquet", columns=document_columns).to_pylist()
    }
    sources = {
        str(row["source_file_id"]): (str(row["build_id"]), str(row["doc_id"]))
        for row in pq.read_table(
            root / "source_files.parquet",
            columns=["build_id", "source_file_id", "doc_id"]).to_pylist()
    }
    identities: dict[str, dict] = {}
    for row in pq.read_table(
            root / "event_identities.parquet",
            columns=[
                "event_key", "kind", "doc_group", "identity_status",
                "root_rcept_no", "first_disclosed_at", "last_disclosed_at",
                "n_observations", "n_corrections", "status_at_corpus_end",
            ]).to_pylist():
        event_key = str(row.get("event_key") or "")
        if not event_key or event_key in identities:
            raise ValueError(f"EventIdentity event_key 전역 유일성 오류: {event_key!r}")
        identities[event_key] = row

    # id -> (owner_type, doc_id, role, rcept_dt). 같은 ID를 여러 relation이 재사용해도
    # tuple이 exact 같을 때만 허용한다.
    support: dict[str, tuple[str, str, str, str]] = {}

    def add_support(evidence_key: object, owner_type: str, doc_id: str,
                    role: str, rcept_dt: str) -> None:
        if not isinstance(evidence_key, str) or not evidence_key:
            raise ValueError("support evidence_id가 비어 있습니다")
        value = (owner_type, doc_id, role, rcept_dt)
        previous = support.get(evidence_key)
        if previous is not None and previous != value:
            raise ValueError(
                f"support Evidence ID owner/role 충돌: {evidence_key} "
                f"({previous!r} != {value!r})")
        support[evidence_key] = value

    event_columns = [
        "build_id", "event_key", "seq", "doc_id", "rcept_no", "observed_at",
        "is_correction", "is_termination", "previous_observation_rcept_no",
        "supporting_evidence_ids", "support_roles",
        "support_status", "support_version", "support_limitation",
    ]
    observations_by_event: dict[str, list[dict]] = defaultdict(list)
    event_seq_keys: set[tuple[str, int]] = set()
    event_doc_ids: set[str] = set()
    event_rcept_nos: set[str] = set()
    for row in pq.read_table(
            root / "event_observations.parquet", columns=event_columns).to_pylist():
        event_key = str(row.get("event_key") or "")
        doc_id = str(row.get("doc_id") or "")
        rcept_no = str(row.get("rcept_no") or "")
        seq = row.get("seq")
        identity = identities.get(event_key)
        document = documents.get(doc_id)
        event_seq_key = (event_key, seq) if type(seq) is int else None
        if (event_seq_key is None or event_seq_key in event_seq_keys
                or not doc_id or doc_id in event_doc_ids
                or not rcept_no or rcept_no in event_rcept_nos):
            raise ValueError(
                "EventObservation event/seq/doc/receipt 전역 cardinality 오류: "
                f"event={event_key!r} seq={seq!r} doc={doc_id!r} rcept={rcept_no!r}")
        event_seq_keys.add(event_seq_key)
        event_doc_ids.add(doc_id)
        event_rcept_nos.add(rcept_no)
        observations_by_event[event_key].append(row)
        expected_identity_kind = (
            "document_lineage" if document is not None
            and document.get("doc_group") == "periodic" else "business_event")
        if (row.get("build_id") != build_id or identity is None or document is None
                or document["rcept_no"] != row.get("rcept_no")
                or document["rcept_dt"] != row.get("observed_at")
                or identity.get("kind") != expected_identity_kind
                or identity.get("doc_group") != document.get("doc_group")
                or type(row.get("is_correction")) is not bool
                or row.get("is_correction") is not document["is_correction"]
                or type(row.get("is_termination")) is not bool
                or row.get("is_termination") is not is_declared_termination(document)):
            raise ValueError(f"EventObservation ownership/type 오류: {row.get('rcept_no')}")
        ids, roles = row.get("supporting_evidence_ids"), row.get("support_roles")
        if (not isinstance(ids, list) or not isinstance(roles, list)
                or len(ids) != len(roles) or ids != sorted(ids)
                or len(ids) != len(set(ids))
                or any(role not in EVENT_SUPPORT_ROLES for role in roles)
                or row.get("support_version") != EVENT_SUPPORT_VERSION
                or (identity["kind"] == "document_lineage" and bool(ids))):
            raise ValueError(f"Event support list/version 오류: {row.get('rcept_no')}")
        expected_status, expected_limitation = event_support_decision(
            document, str(identity["kind"]), set(roles))
        if (row.get("support_status") != expected_status
                or row.get("support_limitation") != expected_limitation):
            raise ValueError(f"Event support status 오류: {row.get('rcept_no')}")
        for evidence_key, role in zip(ids, roles):
            add_support(evidence_key, "field", str(row["doc_id"]), str(role),
                        str(row["observed_at"]))
    if set(observations_by_event) != set(identities):
        raise ValueError(
            "EventIdentity/EventObservation event_key 집합 불일치: "
            f"identities={len(identities)} observations={len(observations_by_event)}")
    for event_key, identity in identities.items():
        validate_event_history(
            identity, observations_by_event.get(event_key, ()), documents)

    # Relation에서 제외된 multi-owner Correction Evidence까지 limitation을 exact하게
    # 재계산해야 한다. support 목록만 보면 이미 제외된 ID가 보이지 않으므로 2,700개
    # CorrectionItem의 semantic owner를 먼저 센다. ``block_id``는 같은 대표 locator를
    # 공유하는 의미 행에서 같을 수 있어 owner key가 아니다.
    correction_columns = [
        "build_id", "doc_id", "source_file_id", "order", "rcept_dt",
        "before_evidence_id", "after_evidence_id",
    ]
    correction_rows = pq.read_table(
        root / "correction_items.parquet", columns=correction_columns).to_pylist()
    correction_owner_keys = relation_correction_support_owner_keys(correction_rows)
    ambiguous_correction_candidates = {
        evidence_key for evidence_key, owner_keys in correction_owner_keys.items()
        if len(owner_keys) > 1
    }

    relation_columns = [
        "build_id", "relation_id", "src_id", "src_rcept_no", "src_rcept_dt",
        "resolution_status", "supporting_evidence_ids", "support_roles",
        "support_status", "support_version", "support_limitation",
    ]
    relation_rows = pq.read_table(
        root / "relations.parquet", columns=relation_columns).to_pylist()
    for row in relation_rows:
        document = documents.get(str(row.get("src_id") or ""))
        ids, roles = row.get("supporting_evidence_ids"), row.get("support_roles")
        if (row.get("build_id") != build_id or document is None
                or document["rcept_no"] != row.get("src_rcept_no")
                or document["rcept_dt"] != row.get("src_rcept_dt")
                or not isinstance(ids, list) or not isinstance(roles, list)
                or len(ids) != len(roles) or ids != sorted(ids)
                or len(ids) != len(set(ids))
                or any(role not in RELATION_SUPPORT_ROLES for role in roles)
                or row.get("support_version") != RELATION_SUPPORT_VERSION):
            raise ValueError(f"Relation support 구조/owner 오류: {row.get('relation_id')}")
        for evidence_key, role in zip(ids, roles):
            owner_type = ("correction" if relation_correction_support_sides(role)
                          is not None else "field")
            add_support(evidence_key, owner_type, str(row["src_id"]), str(role),
                        str(row["src_rcept_dt"]))

    # Chunk expected recreation은 raw 자체 대신 digest만 보관한다.
    chunk_refs: dict[str, tuple] = {}
    part_groups: dict[tuple[str, str, str], list[int]] = {}
    chunk_columns = [
        "build_id", "doc_id", "source_file_id", "block_id", "parent_id", "locator",
        "part_no", "n_parts", "text", "text_prompt_safe", "security_flags",
        "pii_types", "security_policy_version", "rcept_dt", "evidence_id",
    ]
    pf = pq.ParquetFile(root / "chunks.parquet")
    for batch in pf.iter_batches(batch_size=20_000, columns=chunk_columns):
        data = batch.to_pydict()
        for values in zip(*(data[name] for name in chunk_columns)):
            row = dict(zip(chunk_columns, values))
            document = documents.get(str(row["doc_id"]))
            part_no, n_parts = row["part_no"], row["n_parts"]
            if (row["build_id"] != build_id or document is None
                    or sources.get(str(row["source_file_id"]))
                    != (build_id, str(row["doc_id"]))
                    or document["rcept_dt"] != row["rcept_dt"]
                    or type(part_no) is not int or type(n_parts) is not int
                    or not (0 <= part_no < n_parts)
                    or not isinstance(row["locator"], str)
                    or not row["locator"].endswith(f"/PART[{part_no}]")
                    or row["block_id"] != block_id(
                        str(row["doc_id"]), str(row["source_file_id"]), row["locator"])):
                raise ValueError(f"Chunk ownership/PART 오류: {row.get('block_id')}")
            expected = make_evidence(
                build_id=build_id, doc_id=str(row["doc_id"]),
                source_file_id=str(row["source_file_id"]), kind="chunk_text",
                locator=row["locator"], table_locator=None, logical_row=None,
                logical_col=None, excerpt_raw=str(row["text"]),
                extraction_method="section_chunk", rcept_dt=str(row["rcept_dt"]),
                # ``filer`` may be an individual in holding disclosures. It
                # is not a corporate-name exemption merely because it filed
                # the document; only the issuer/listed names are public-name
                # context for generic security projection.
                corporate_names=(document["corp_name"], document["listed_name"]),
            )
            if (row["evidence_id"] != expected.evidence_id
                    or row["text_prompt_safe"] != expected.excerpt_safe
                    or set(row["security_flags"] or ()) != set(expected.security_flags)
                    or set(row["pii_types"] or ()) != set(expected.pii_types)
                    or row["security_policy_version"] != expected.security_policy_version):
                raise ValueError(f"Chunk safe/Evidence FK 오류: {row.get('block_id')}")
            if expected.evidence_id in chunk_refs:
                raise ValueError(f"Chunk Evidence FK 중복: {expected.evidence_id}")
            chunk_refs[expected.evidence_id] = (
                expected.doc_id, expected.source_file_id, expected.locator,
                expected.excerpt_hash, hashlib.sha256(
                    expected.excerpt_safe.encode("utf-8")).hexdigest(), expected.rcept_dt,
            )
            key = (str(row["doc_id"]), str(row["source_file_id"]), str(row["parent_id"]))
            stats = part_groups.setdefault(key, [n_parts, 0, 0, 0])
            if stats[0] != n_parts:
                raise ValueError(f"Chunk n_parts 충돌: {row.get('parent_id')}")
            stats[1] += 1
            stats[2] += part_no
            stats[3] += part_no * part_no
    for n_parts, count, part_sum, square_sum in part_groups.values():
        if (count != n_parts or part_sum != n_parts * (n_parts - 1) // 2
                or square_sum != n_parts * (n_parts - 1) * (2 * n_parts - 1) // 6):
            raise ValueError("Chunk parent별 PART 완전성 오류")
    if set(chunk_refs) & set(support):
        raise ValueError("Evidence ID가 Chunk와 Event/Relation support에 교차 참조됨")

    targets = set(chunk_refs) | set(support) | ambiguous_correction_candidates
    evidence_rows: dict[str, dict] = {}
    evidence_columns = [
        "build_id", "evidence_id", "doc_id", "source_file_id", "kind", "locator",
        "table_locator", "logical_row", "logical_col", "excerpt_raw", "excerpt_safe",
        "excerpt_hash", "extraction_method", "security_flags", "pii_types",
        "security_policy_version", "evidence_policy_version", "rcept_dt",
    ]
    pf = pq.ParquetFile(root / "evidence.parquet")
    for batch in pf.iter_batches(batch_size=50_000, columns=evidence_columns):
        data = batch.to_pydict()
        for values in zip(*(data[name] for name in evidence_columns)):
            row = dict(zip(evidence_columns, values))
            evidence_key = row["evidence_id"]
            if evidence_key not in targets:
                continue
            if evidence_key in evidence_rows:
                raise ValueError(f"target Evidence ID 중복: {evidence_key}")
            if (row["build_id"] != build_id
                    or sources.get(str(row["source_file_id"]))
                    != (build_id, str(row["doc_id"]))
                    or documents.get(str(row["doc_id"]), {}).get("rcept_dt")
                    != row["rcept_dt"]):
                raise ValueError(f"Evidence owner/rcept 오류: {evidence_key}")
            if evidence_key in chunk_refs:
                expected = chunk_refs[evidence_key]
                actual_safe_hash = hashlib.sha256(
                    str(row["excerpt_safe"]).encode("utf-8")).hexdigest()
                if (row["kind"] != "chunk_text"
                        or row["table_locator"] is not None
                        or row["logical_row"] is not None or row["logical_col"] is not None
                        or row["extraction_method"] != "section_chunk"
                        or (row["doc_id"], row["source_file_id"], row["locator"],
                            row["excerpt_hash"], actual_safe_hash, row["rcept_dt"]) != expected
                        or row["excerpt_hash"]
                        != hashlib.sha256(str(row["excerpt_raw"]).encode("utf-8")).hexdigest()):
                    raise ValueError(f"Chunk Evidence exact recreation 오류: {evidence_key}")
            elif evidence_key in support:
                owner_type, doc_id, _role, rcept_dt = support[evidence_key]
                expected_kind = ("field_value" if owner_type == "field"
                                 else "correction_value")
                if (row["kind"] != expected_kind or row["doc_id"] != doc_id
                        or row["rcept_dt"] != rcept_dt or row["pii_types"]):
                    raise ValueError(f"support Evidence kind/doc/date/PII 오류: {evidence_key}")
            elif row["kind"] != "correction_value":
                raise ValueError(
                    f"ambiguous Correction Evidence kind 오류: {evidence_key}")
            evidence_rows[evidence_key] = row
    missing = targets - set(evidence_rows)
    if missing:
        raise ValueError(f"Chunk/support Evidence orphan: {sorted(missing)[:3]}")

    ambiguous_correction_docs: set[str] = set()
    for evidence_key in ambiguous_correction_candidates:
        evidence = evidence_rows[evidence_key]
        owner_docs = {
            doc_id for doc_id, _order in correction_owner_keys[evidence_key]
        }
        if owner_docs != {str(evidence["doc_id"])}:
            raise ValueError(
                f"ambiguous Correction Evidence owner doc 충돌: {evidence_key}")
        # PII Evidence는 producer에서도 Relation support 후보에 넣지 않으므로 owner가
        # 여러 개여도 ambiguity limitation의 대상이 아니다.
        if not evidence["pii_types"]:
            ambiguous_correction_docs.update(owner_docs)
    for row in relation_rows:
        ids = row["supporting_evidence_ids"]
        expected_limitation = relation_support_limitation(
            row.get("resolution_status"),
            has_source_evidence=bool(ids),
            has_ambiguous_correction_evidence_owner=(
                str(row["src_id"]) in ambiguous_correction_docs),
        )
        if (row.get("support_status") != "partial"
                or row.get("support_limitation") != expected_limitation):
            raise ValueError(f"Relation typed limitation exact 오류: {row.get('relation_id')}")

    field_owners: dict[str, dict] = {}
    correction_owners: dict[str, list[tuple[dict, str]]] = defaultdict(list)
    support_ids = set(support)
    field_columns = [
        "build_id", "evidence_id", "doc_id", "source_file_id", "path", "is_pii",
        "value_status", "rcept_dt",
    ]
    pf = pq.ParquetFile(root / "fields.parquet")
    for batch in pf.iter_batches(batch_size=50_000, columns=field_columns):
        data = batch.to_pydict()
        for values in zip(*(data[name] for name in field_columns)):
            row = dict(zip(field_columns, values))
            if row["evidence_id"] in support_ids:
                if row["evidence_id"] in field_owners:
                    raise ValueError(f"support Field owner 복수: {row['evidence_id']}")
                field_owners[row["evidence_id"]] = row
    for row in correction_rows:
        for side in ("before", "after"):
            evidence_key = row[f"{side}_evidence_id"]
            if evidence_key not in support_ids:
                continue
            correction_owners[evidence_key].append((row, side))
    for evidence_key, (owner_type, doc_id, role, rcept_dt) in support.items():
        evidence = evidence_rows[evidence_key]
        if owner_type == "field":
            owner = field_owners.get(evidence_key)
            actual_role = (event_support_role(
                owner.get("path"), value_status=owner.get("value_status"),
                is_pii=owner.get("is_pii")) if owner else None)
            if (owner is None or evidence_key in correction_owners
                    or owner["build_id"] != build_id or owner["doc_id"] != doc_id
                    or owner["rcept_dt"] != rcept_dt or owner["is_pii"] is not False
                    or sources.get(owner["source_file_id"]) != (build_id, doc_id)
                    or evidence["source_file_id"] != owner["source_file_id"]
                    or actual_role != role):
                raise ValueError(f"Event/Relation Field support owner 오류: {evidence_key}")
        else:
            owned = correction_owners.get(evidence_key, ())
            expected_sides = relation_correction_support_sides(role)
            actual_sides = {side for _owner, side in owned}
            same_item = bool(owned) and all(owner is owned[0][0]
                                             for owner, _side in owned)
            if (expected_sides is None or evidence_key in field_owners
                    or actual_sides != set(expected_sides)
                    or len(owned) != len(expected_sides) or not same_item):
                raise ValueError(f"Relation Correction support owner 없음/복수: {evidence_key}")
            owner = owned[0][0]
            if (owner["build_id"] != build_id or owner["doc_id"] != doc_id
                    or owner["rcept_dt"] != rcept_dt
                    or sources.get(owner["source_file_id"]) != (build_id, doc_id)
                    or evidence["source_file_id"] != owner["source_file_id"]):
                raise ValueError(f"Relation Correction support owner 오류: {evidence_key}")


def _code_hash() -> str:
    """`artifact.code_hash` 별칭. 상위 단계 도장과 **같은 값**이어야 비교가 성립한다."""
    return _pipeline_code_hash()


def _sections(record, index, build_id, failures: list[dict], trees: dict
              ) -> Iterator[tuple[dict, str, str]]:
    """본문 + 첨부의 섹션. `(행, parse_mode, role)` 을 준다.

    첨부는 감사보고서·연결감사보고서 415건이며 **본문과 같은 DART XML** 이라 같은 파서를 쓴다.
    감사의견·재무제표 주석이 거기 있어 빼면 답할 수 없는 질의가 생긴다.

    한 문서에 파일이 여럿이 되므로 두 가지를 정한다.

    - `order` 는 문서 전체에서 이어진다. `chunk_id` 가 `{doc_id}#{order}` 라 겹치면 안 된다.
    - 첨부 섹션의 `path` 앞에 문서명(`감사보고서`)을 붙인다. 본문과 첨부가
      `III. 재무에 관한 사항` 처럼 같은 경로를 가질 수 있어 그대로 두면 조회가 모호해진다.

    `locator` 는 **파일 안에서의** TITLE 순번이다. `block_id` 가 source_file_id 를 포함하므로
    파일이 달라도 충돌하지 않는다.
    """
    main = index.main_xml(record["file_path"], record["rcept_no"])
    if main is None:
        raise FileNotFoundError("본문 XML 없음")

    targets: list[tuple[Path, str]] = [(main, "main")]
    targets += [(p, "attachment")
                for p in index.attachment_xmls(record["file_path"], record["rcept_no"])]

    order = 0
    for path, role in targets:
        # (level, block_id) 스택. 상위 섹션의 ID 를 자식에게 물려준다 (L-06).
        # **파일마다 새로 시작한다** — 첨부 감사보고서의 섹션이 본문 섹션을 부모로 삼으면
        # 계층이 거짓말이 된다.
        ancestry: list[tuple[int, str]] = []
        try:
            if role == "main":
                # 본문은 트리를 **한 번만** 파싱해 재무 Fact 추출과 공유한다.
                root, mode = parse_tree(path.read_text(encoding="utf-8", errors="replace"))
                trees[record["doc_id"]] = root
                parsed = parse_root(root, mode)
            else:
                parsed = parse_file(path)
        except (ParseFailure, UnsupportedFormat) as exc:
            if role == "main":
                raise
            # 첨부 하나가 깨져도 본문과 다른 첨부는 유효하다
            failures.append({"doc_id": record["doc_id"], "stage": "attachment",
                             "reason": f"{path.name}: {type(exc).__name__}: {exc}"[:160]})
            continue
        src = source_file_id(nfc(f"{record['file_path']}/{path.name}"))
        prefix = f"{parsed.document_name} > " if role == "attachment" else ""
        for sec in parsed.sections:
            # 파서가 들고 온 **원문 element 위치**. 예전의 `SECTION[order]` 는 순번이라
            # 파서가 섹션을 하나 더 잡으면 뒤쪽 ID 가 전부 밀렸다 (L-03 · X-01).
            loc = sec.source_locator or child_locator("", "SECTION", sec.order)
            path_str = prefix + sec.path
            bid = block_id(record["doc_id"], src, loc)
            while ancestry and ancestry[-1][0] >= sec.level:
                ancestry.pop()
            parent = ancestry[-1][1] if ancestry else None
            ancestry.append((sec.level, bid))
            yield as_row(Section(
                build_id=build_id, doc_id=record["doc_id"], source_file_id=src,
                block_id=bid, parent_id=parent,
                path=path_str, order=order, locator=loc,
                level=sec.level, title=sec.path.rsplit(" > ", 1)[-1], text=sec.text,
                n_chars=sec.n_chars, n_tables=sec.n_tables,
                explicit_omission=bool(_OMIT.search(sec.text)) and sec.n_chars < 300,
                index_eligible=sec.n_chars >= 100,
                rcept_dt=record["rcept_dt"],
            )), parsed.parse_mode, role
            order += 1


def _chunk_row(record, row: dict, part, n_parts: int, build_id: str,
               order: int) -> tuple[dict, dict]:
    """Section 행 + 조각 → Chunk 행. Section 은 구조, Chunk 는 검색이다 (D2)."""
    loc = child_locator(row["locator"], "PART", part.part_no)
    projection = project_chunk_text(
        part.text, (record.get("corp_name"), record.get("listed_name"))
    )
    evidence = make_evidence(
        build_id=build_id, doc_id=record["doc_id"],
        source_file_id=row["source_file_id"], kind="chunk_text",
        # Chunk locator 자체가 이미 canonical ``.../PART[n]``다. source 좌표를
        # 보강한다며 suffix를 덧붙이지 않는다.
        locator=loc, table_locator=None, logical_row=None, logical_col=None,
        excerpt_raw=part.text, extraction_method="section_chunk",
        rcept_dt=row["rcept_dt"],
        corporate_names=(record.get("corp_name"), record.get("listed_name")),
    )
    chunk = as_row(Chunk(
        build_id=build_id, doc_id=record["doc_id"],
        source_file_id=row["source_file_id"],
        block_id=block_id(record["doc_id"], row["source_file_id"], loc),
        parent_id=row["block_id"], path=row["path"], order=order, locator=loc,
        evidence_id=evidence.evidence_id,
        part_no=part.part_no, n_parts=n_parts,
        # raw는 근거 재현용으로 보존하고, 검색/LLM은 아래 파생 projection만 사용한다.
        text=part.text,
        text_search=projection.text_search,
        text_prompt_safe=projection.text_prompt_safe,
        security_flags=list(projection.security_flags), pii_types=list(projection.pii_types),
        security_policy_version=projection.security_policy_version,
        n_chars=part.n_chars,
        header_repeated=part.header_repeated, over_budget=part.over_budget,
        content_hash=part.content_hash, chunker_version=CHUNK_VER,
        index_eligible=row["index_eligible"], rcept_dt=row["rcept_dt"],
    ))
    evidence_row = as_row(evidence)
    if (evidence_row["locator"] != chunk["locator"]
            or evidence_row["excerpt_raw"] != chunk["text"]
            or evidence_row["excerpt_safe"] != chunk["text_prompt_safe"]
            or evidence_row["table_locator"] is not None
            or evidence_row["logical_row"] is not None
            or evidence_row["logical_col"] is not None):
        raise RuntimeError(f"Chunk Evidence 재생성 계약 불일치: {chunk['block_id']}")
    return chunk, evidence_row


def _viewer_source_id(record: Mapping[str, Any], index: CorpusIndex, path: Path) -> str:
    """inventory의 NFC 키로 PDF/HTML source ID를 만든다.

    ``path.name``은 실제 디스크 표기(NFD일 수 있음)이고 ``file_path``는 manifest
    NFC다. 문자열만 조립해 열지 않고 CorpusIndex에 다시 bind해 ID 키만 얻는다.
    """
    rel = f"{record['file_path']}/{path.name}"
    return source_file_id(index.bind(rel).normalized_path_key)


def _viewer_source_fields(result: SourceParseResult | None,
                          selection_role: str | None) -> dict[str, Any]:
    """SourceParseResult를 SourceFile 1.7 물리 컬럼으로 펼친다."""
    if result is None:
        return {
            "source_selection_role": None, "parse_usable": None,
            "coverage_n_sections": None, "coverage_n_chars": None,
            "coverage_n_tables": None, "coverage_n_pages": None,
            "coverage_pages_with_text": None, "coverage_n_lines": None,
            "page_text_coverage": None, "active_nodes_removed": None,
            "active_attributes_removed": None, "locator_kind": None,
            "locator_limitations": [], "parse_warnings": [], "parse_error": None,
        }
    coverage = result.coverage
    return {
        "source_selection_role": selection_role,
        "parse_usable": result.usable,
        "coverage_n_sections": coverage.n_sections,
        "coverage_n_chars": coverage.n_chars,
        "coverage_n_tables": coverage.n_tables,
        "coverage_n_pages": coverage.n_pages,
        "coverage_pages_with_text": coverage.pages_with_text,
        "coverage_n_lines": coverage.n_lines,
        "page_text_coverage": coverage.page_text_coverage,
        "active_nodes_removed": coverage.active_nodes_removed,
        "active_attributes_removed": coverage.active_attributes_removed,
        "locator_kind": result.locator_kind,
        "locator_limitations": list(result.locator_limitations),
        "parse_warnings": list(result.warnings),
        "parse_error": result.error,
    }


def _viewer_quality_guard(parsed: ParsedViewer) -> str:
    """승인된 viewer 결과가 mode/coverage까지 그대로인지 확인하는 짧은 guard."""
    primary = parsed.primary
    if primary is None:
        primary_part = "primary=none;coverage=none"
    elif primary.source_kind == "pdf":
        primary_part = (
            f"primary={primary.parse_mode};coverage="
            f"{primary.coverage.pages_with_text}/{primary.coverage.n_pages}")
    else:
        primary_part = (
            f"primary={primary.parse_mode};coverage="
            f"{primary.coverage.n_sections}/{primary.coverage.n_tables}")
    modes = ",".join(sorted(result.parse_mode for result in parsed.source_results))
    return f"{primary_part};sources={modes}"


def _viewer_document_fields(
        parsed: ParsedViewer | None,
        source_id_by_path: Mapping[str, str]) -> dict[str, Any]:
    """ParsedViewer의 문서 단위 선택·교차검증을 Document 1.7에 보존한다."""
    if parsed is None:
        return {
            "primary_source_file_id": None, "alternate_source_file_ids": [],
            "source_cross_check_status": None, "source_cross_check_reason": None,
            "source_cross_check_source_file_ids": [],
            "source_cross_check_primary_digest": None,
            "source_cross_check_alternate_digest": None,
            "source_cross_check_conflicts_json": None,
        }

    primary_id = (source_id_by_path.get(str(parsed.primary.source_path))
                  if parsed.primary is not None else None)
    alternate_ids = [
        source_id_by_path[str(result.source_path)]
        for result in parsed.alternates
        if str(result.source_path) in source_id_by_path
    ]
    compared_ids = [
        source_id_by_path[path]
        for path in parsed.cross_check.compared_sources
        if path in source_id_by_path
    ]
    conflicts = ([asdict(conflict) for conflict in parsed.cross_check.conflicts]
                 if parsed.cross_check.conflicts else None)
    return {
        "primary_source_file_id": primary_id,
        "alternate_source_file_ids": alternate_ids,
        "source_cross_check_status": parsed.cross_check.status.value,
        "source_cross_check_reason": parsed.cross_check.reason,
        "source_cross_check_source_file_ids": compared_ids,
        "source_cross_check_primary_digest": parsed.cross_check.primary_digest,
        "source_cross_check_alternate_digest": parsed.cross_check.alternate_digest,
        "source_cross_check_conflicts_json": (
            json.dumps(conflicts, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) if conflicts is not None else None),
    }


def _viewer_sections(
        record: Mapping[str, Any], index: CorpusIndex, build_id: str,
) -> tuple[ParsedViewer, list[tuple[dict, SourceParseResult, str]]]:
    """PDF/HTML을 독립 파싱하고 **모든 usable source**의 Section을 돌려준다.

    primary는 기본 검색 대상이지만 alternate는 감사·비교용 canonical Section으로
    남긴다. alternate와 명시적 conflict는 Chunk까지 보존하되 ``index_eligible``을
    끄어 기본 검색에서 중복·충돌 주장이 승격되지 않게 한다.
    """
    files = index.files(record["file_path"])
    parsed = parse_pdf_html(files, record["report_nm"], record["corp_name"])
    ordered: list[SourceParseResult] = []
    if parsed.primary is not None:
        ordered.append(parsed.primary)
    ordered.extend(result for result in parsed.source_results if result is not parsed.primary)

    rows: list[tuple[dict, SourceParseResult, str]] = []
    order = 0
    conflict = parsed.cross_check.status == CrossCheckStatus.CONFLICT
    for result in ordered:
        if not result.usable:
            continue
        selection = "primary" if result is parsed.primary else "alternate"
        src = _viewer_source_id(record, index, result.source_path)
        ancestry: list[tuple[int, str]] = []       # source를 넘어 거짓 부모를 만들지 않는다
        source_indexable = selection == "primary" and not conflict
        for sec in result.sections:
            loc = sec.source_locator or child_locator("", "SECTION", sec.order)
            bid = block_id(record["doc_id"], src, loc)
            while ancestry and ancestry[-1][0] >= sec.level:
                ancestry.pop()
            parent = ancestry[-1][1] if ancestry else None
            ancestry.append((sec.level, bid))
            rows.append((as_row(Section(
                build_id=build_id, doc_id=record["doc_id"], source_file_id=src,
                block_id=bid, parent_id=parent, path=sec.path, order=order, locator=loc,
                level=sec.level, title=sec.path.rsplit(" > ", 1)[-1], text=sec.text,
                n_chars=sec.n_chars, n_tables=sec.n_tables,
                explicit_omission=bool(_OMIT.search(sec.text)) and sec.n_chars < 300,
                index_eligible=source_indexable and sec.n_chars >= 100,
                rcept_dt=record["rcept_dt"],
            )), result, selection))
            order += 1
    return parsed, rows


def _facts(record, index, build_id, root, issues_out: list, *,
           source_sha256: str | None = None,
           overrides_out: list | None = None) -> Iterator[tuple[dict, dict]]:
    """정기공시 재무제표 → Fact. 평가 A유형 질문이 직접 요구하는 층이다.

    Section 을 다시 파싱하지 않는다 — 마크다운은 손실 있는 표현이고 locator 가 없다.
    XML root 를 그대로 받아 `TABLE-GROUP` 에서 뽑는다.
    """
    path = index.main_xml(record["file_path"], record["rcept_no"])
    if path is None:
        return
    src = source_file_id(nfc(f"{record['file_path']}/{path.name}"))
    result = extract_financials_detailed(root, source_sha256=source_sha256)
    issues_out.extend(result.issues)
    if overrides_out is not None:
        overrides_out.extend(result.applied_overrides)
    for order, f in enumerate(result.facts):
        # Evidence는 raw 값 셀의 정확한 locator를 가리킨다. Fact block_id 계산에는 넣지 않는다.
        evidence = make_evidence(
            build_id=build_id, doc_id=record["doc_id"], source_file_id=src,
            kind="fact_value", locator=f.cell_locator, table_locator=f.table_locator,
            logical_row=f.logical_row, logical_col=f.logical_col,
            excerpt_raw=f.value_text, extraction_method="financial_table_cell",
            rcept_dt=record["rcept_dt"],
            corporate_names=(record.get("corp_name"), record.get("listed_name")),
        )
        yield as_row(Fact(
            build_id=build_id, doc_id=record["doc_id"], source_file_id=src,
            block_id=block_id(record["doc_id"], src, f.cell_locator),
            # **없는 것을 가리키는 FK 를 만들지 않는다.** 예전에는 `Table` 의 block_id 를
            # 넣었는데 `Table` 엔티티가 없어 5,775개가 전부 orphan 이었다.
            # 표는 `table_locator` 컬럼으로 지목한다. `Table` 은 소비자가 생기면 만든다.
            parent_id=None,
            path=f"{f.scope} > {f.statement} > {f.account_path}",
            order=order, locator=f.cell_locator, evidence_id=evidence.evidence_id,
            scope=f.scope, statement=f.statement, statement_title=f.statement_title,
            account_raw=f.account_raw, account_path=f.account_path,
            account_depth=f.account_depth, account_norm=f.account_norm,
            period_label=f.period_label, period_start=f.period_start,
            period_end=f.period_end, period_type=f.period_type,
            cumulative=f.cumulative, value_text=f.value_text,
            raw_value=f.raw_value, raw_unit=f.raw_unit, unit_source=f.unit_source,
            value_status=f.value_status,
            table_locator=f.table_locator,
            logical_row=f.logical_row, logical_col=f.logical_col,
            rcept_dt=record["rcept_dt"],
            acode=f.acode, account_norm_source=f.account_norm_source,
        )), as_row(evidence)


def _fields(record, index, build_id) -> Iterator[tuple[dict, dict]]:
    path = index.main_xml(record["file_path"], record["rcept_no"])
    if path is None:
        raise FileNotFoundError("본문 XML 없음")
    src = source_file_id(f"{record['file_path']}/{path.name}")
    # locator 는 **파서가 원문에서 들고 온 실제 위치**다. 예전처럼 `FIELD[order]` 를
    # 지어내면 근거를 원문으로 되짚을 수 없다 (C-01).
    if record["doc_group"] == "exchange":
        form = parse_exchange(path)
        items = [(r.label_path, r.value, None, None, None,
                  r.source_locator, r.table_locator, r.logical_row, r.logical_col,
                  r.label_locators, r.occurrence)
                 for r in form.records]
    else:
        # strict ET 가 실패하는 원본(따옴표 손상 등)도 `_grids` 와 같은 recover 폴백으로 읽는다.
        # ET.fromstring 만 쓰면 미분류 손상 1건이 stage=parse 차단으로 발행 전체를 막는다.
        root, _mode = parse_tree(path.read_text(encoding="utf-8", errors="replace"))
        fields, _cells = extract_form(root)
        items = [(f.label_path, f.value, f.acode, f.aunit, f.aunitvalue,
                  f.source_locator, f.table_locator, f.origin_row, f.origin_col,
                  f.label_locators, f.occurrence)
                 for f in fields]

    # 같은 원문 행의 성명(명칭). `직 업(사업내용)` 은 주체가 법인인지에 따라
    # 업종이 되기도 개인정보가 되기도 한다 (이슈 #139). 읽기 쪽
    # (`src/canonical/read.py`)이 **같은 규칙으로** 이 지도를 다시 만든다 —
    # 어긋나면 저장된 마스킹이 재검증을 통과하지 못한다.
    subjects = subject_names_by_row(
        (item[2], item[1], item[6], item[7]) for item in items)

    # 지분공시의 `성명(명칭)` 칸에는 법인과 자연인이 함께 온다. 값 모양으로는
    # `에코프로`가 사람으로, `Scott Samuel Braun`이 법인으로 갈린다 —
    # 공시가 스스로 적어 둔 `보고자 구분`(CRP_TP)·`구분`(SPC_TP)으로 정한다
    # (이슈 #199). 읽기 쪽도 **같은 함수로** 이 지도를 다시 만든다.
    party_types = (
        holding_party_types_by_row(
            (item[0], item[3], item[1], item[6], item[7]) for item in items)
        if record["doc_group"] == "holding" else None)

    for order, (label, value, acode, aunit, aunitvalue, loc, tloc,
                logical_row, logical_col, llocs, occ) in enumerate(items):
        party_type = resolve_party_type(
            party_types, label, value, tloc, logical_row)
        projection = project_field_value(
            label, value, (record.get("corp_name"), record.get("listed_name")),
            subject_name=subjects.get((tloc, logical_row)),
            party_type=party_type,
        )
        evidence = make_evidence(
            build_id=build_id, doc_id=record["doc_id"], source_file_id=src,
            kind="field_value", locator=loc, table_locator=tloc,
            logical_row=logical_row, logical_col=logical_col,
            excerpt_raw=value, extraction_method="structured_form_field",
            rcept_dt=record["rcept_dt"], pii_label=label,
            pii_subject_name=subjects.get((tloc, logical_row)),
            pii_party_type=party_type,
            corporate_names=(record.get("corp_name"), record.get("listed_name")),
        )
        yield as_row(Field(
            build_id=build_id, doc_id=record["doc_id"], source_file_id=src,
            block_id=block_id(record["doc_id"], src, loc), parent_id=None,
            path=label, order=order, locator=loc, evidence_id=evidence.evidence_id,
            value_raw=value, value_masked=projection.value_masked,
            value_prompt_safe=evidence.excerpt_safe,
            pii_type=projection.pii_type,
            security_policy_version=projection.security_policy_version,
            value_status=classify_value_status(value, parse_number(value)),
            acode=acode, aunit=aunit, aunitvalue=aunitvalue,
            is_pii=projection.is_pii,
            table_locator=tloc, logical_row=logical_row, logical_col=logical_col,
            label_locators=" ".join(llocs) or None,
            occurrence=occ, rcept_dt=record["rcept_dt"],
        )), as_row(evidence)


def _grids(record, path):
    """문서의 모든 표 → `TableGrid`. 포맷 차이는 여기서만 흡수한다."""
    if record["doc_group"] == "exchange":
        root = lxml_html.fromstring(path.read_text(encoding="utf-8", errors="replace"))
        for ti, table in enumerate(root.iter("table")):
            yield build_grid(table, f"TABLE[{ti}]", text_of=exchange_cell_text)
    else:
        # `parse_tree` 를 쓴다. `ET.fromstring` 만 쓰면 따옴표가 깨진 원본 79건이 통째로 실패한다.
        root, _mode = parse_tree(path.read_text(encoding="utf-8", errors="replace"))
        for ti, table in enumerate(root.iter("TABLE")):
            yield build_grid(table, f"TABLE[{ti}]")


def _cells(record, index, build_id) -> Iterator[dict]:
    """표 격자를 논리 좌표 단위로 편다.

    **span 영역의 모든 좌표가 한 행**이다. 원점만 저장하면 `(row, col)` 로 바로 조회할 수
    없고, `header_path`·`inherited_from` 이 이 엔티티의 존재 이유인데 그게 무의미해진다.

    Field 는 서식이 역할을 명시한 곳(`TE`/`TU`, `span.xforms_input`)에서만 나온다.
    `TD` 만 있는 표는 무엇이 값인지 판정할 수 없어(가설 두 개 모두 실패, §6.13)
    **역할을 지어내지 않고** 격자 그대로 여기에 남긴다.
    """
    path = index.main_xml(record["file_path"], record["rcept_no"])
    if path is None:
        raise FileNotFoundError("본문 XML 없음")
    src = source_file_id(f"{record['file_path']}/{path.name}")

    order = 0
    for grid in _grids(record, path):
        table_block = block_id(record["doc_id"], src, grid.locator)
        # 단칸 표는 DART 가 각주·단위(`(단위 : 백만원)`)를 넣는 자리다 (§6.5)
        note_table = len(grid.cells) == 1
        # 열마다 한 번만 계산한다. 셀마다 부르면 격자 크기에 헤더 행 수가 곱해진다.
        headers = [grid.header_path(c) for c in range(grid.n_cols)]
        origin_block: dict[int, str] = {}
        for p in grid.placements:
            cell = grid.cells[p.cell_index]
            loc = f"{grid.locator}::{p.logical_row},{p.logical_col}"
            bid = block_id(record["doc_id"], src, loc)
            if not p.inherited:
                origin_block[cell.index] = bid
            yield as_row(Cell(
                build_id=build_id, doc_id=record["doc_id"], source_file_id=src,
                block_id=bid, parent_id=table_block,
                path=headers[p.logical_col], order=order, locator=loc,
                origin_row=cell.origin_row, origin_col=cell.origin_col,
                logical_row=p.logical_row, logical_col=p.logical_col,
                rowspan=cell.rowspan, colspan=cell.colspan, tag=cell.tag,
                role="note" if note_table else "header" if cell.in_head else "body",
                text_raw=cell.text, acode=cell.acode, aunit=cell.aunit,
                aunitvalue=cell.aunitvalue, header_path=headers[p.logical_col],
                inherited_from=origin_block.get(cell.index) if p.inherited else None,
            ))
            order += 1


def _relation_id(src_id: str, dst_id: str | None, relation_type: str) -> str:
    """Canonical endpoint 기반 Relation ID."""
    return hashlib.sha1(
        f"{src_id}|{dst_id}|{relation_type}".encode()).hexdigest()[:ID_HEX]


def _canonical_relation(
    raw: Mapping[str, Any],
    doc_id_by_rcept: Mapping[str, str],
    rcept_dt_of: Mapping[str, str],
    build_id: str,
) -> Relation:
    """원천 접수번호 간선을 canonical Document FK 간선으로 변환한다.

    해결됐다고 선언한 endpoint가 현재 Document 집합에 없으면 행을 만들지 않고 실패한다.
    `root_missing`은 반대로 endpoint를 반드시 null로 두고 원문 힌트와 사유를 보존한다.
    """
    src_rcept_no = str(raw.get("src_rcept_no") or "")
    if not src_rcept_no or src_rcept_no not in doc_id_by_rcept:
        raise ValueError(f"Relation source Document 없음: {src_rcept_no or '<empty>'}")
    src_id = doc_id_by_rcept[src_rcept_no]
    status = str(raw.get("resolution_status") or "")
    dst_receipt_raw = raw.get("dst_rcept_no")
    dst_rcept_no = str(dst_receipt_raw) if dst_receipt_raw else None

    if status == "root_missing":
        if not raw.get("target_hint") or not raw.get("root_missing_reason"):
            raise ValueError(
                f"root_missing Relation에 hint/reason 없음: {src_rcept_no}")
        dst_rcept_no = None
        dst_id = None
    else:
        dst_id = doc_id_by_rcept.get(dst_rcept_no) if dst_rcept_no else None
        if status == "resolved" and dst_id is None:
            raise ValueError(
                f"resolved Relation target Document 없음: {src_rcept_no} -> "
                f"{dst_rcept_no or '<empty>'}")

    candidates = raw.get("candidate_ids") or ()
    candidate_ids = candidates if isinstance(candidates, str) else " ".join(candidates)
    relation_type = str(raw.get("relation_type") or "")
    return Relation(
        build_id=build_id,
        relation_id=_relation_id(src_id, dst_id, relation_type),
        src_kind="document", src_id=src_id, src_rcept_no=src_rcept_no,
        dst_kind="document" if dst_id is not None else None,
        dst_id=dst_id, dst_rcept_no=dst_rcept_no if dst_id is not None else None,
        relation_type=relation_type, resolution_status=status,
        target_hint=raw.get("target_hint"), confidence=float(raw.get("confidence") or 0.0),
        resolver_version=RESOLVER_VERSION,
        candidate_ids=candidate_ids or None,
        match_features=raw.get("match_features") or None,
        src_rcept_dt=rcept_dt_of.get(src_rcept_no),
        root_missing_reason=raw.get("root_missing_reason"),
        supporting_evidence_ids=[], support_roles=[], support_status="partial",
        support_version=RELATION_SUPPORT_VERSION,
        support_limitation="support_not_finalized",
    )


def build(corpus_root: str | Path = "data/corpus",
          out_dir: str | Path = "out/canonical",
          with_cells: bool = False, *,
          quality_policy_path: str | Path = QUALITY_POLICY_PATH) -> dict:
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    corpus_root = Path(corpus_root)
    final_dir = Path(out_dir)

    # **제자리로 덮어쓰지 않는다.** 예전에는 `out/canonical/` 에 직접 썼고, 그래서
    # 빌드 중에 읽는 쪽이 반쯤 쓰인 Parquet 을 보고 `magic bytes not found` 로 죽었다.
    # 임시 디렉터리에 다 만든 뒤 **한 번에 바꿔치기**한다. 읽는 쪽은 항상
    # 「이전 빌드 전체」 아니면 「새 빌드 전체」를 본다 (P0-8).
    lock = final_dir.with_suffix(".lock")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # **죽은 빌드의 잠금은 스스로 걷어낸다.** 예전에는 사람이 손으로 지워야 했고,
        # 그 수동 개입이 실행 중인 빌드의 staging 까지 지우는 사고를 냈다.
        # 잠금에 적힌 PID 가 살아 있는지로 판단한다.
        holder = _lock_holder(lock)
        if holder is not None and _alive(holder):
            raise RuntimeError(
                f"다른 빌드가 진행 중입니다 (PID {holder}). 기다리거나 그 빌드를 멈추세요."
            ) from None
        shutil.rmtree(final_dir.with_name(final_dir.name + ".building"), ignore_errors=True)
        lock.unlink(missing_ok=True)
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock_fd, f"{os.getpid()} {started}\n".encode())
    os.close(lock_fd)

    out_dir = final_dir.with_name(final_dir.name + ".building")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = CorpusIndex.build(corpus_root)
    records = list(iter_manifest(corpus_root))
    manifest_hash = _sha256(corpus_root / "manifest.jsonl")

    # **파일 내용으로** inventory hash 를 만든다 (B-01). 예전에는 경로 이름만 이어 붙여서,
    # 파일 바이트가 바뀌어도 hash 도 build_id 도 그대로였다. 여기서 계산한 파일별 해시를
    # 아래 SourceFile 에 재사용하므로 중복 계산은 없다.
    file_hashes: dict[Path, str] = {}
    file_full_hashes: dict[Path, str] = {}
    inv = hashlib.sha256()
    for record in records:
        for f in index.files(record["file_path"]):
            if f not in file_hashes:
                file_full_hashes[f] = _full_sha256(f)
                # SourceFile의 기존 128-bit 계약은 유지하되 승인 정책에는 아래 full hash를 쓴다.
                file_hashes[f] = file_full_hashes[f][:32]
            inv.update(nfc(f.name).encode())
            inv.update(file_hashes[f].encode())
    inventory_hash = inv.hexdigest()[:ID_HEX]
    # build_id 는 **코드 해시**로 뽑는다. 예전에는 손으로 관리하는 버전 문자열
    # (`PARSER_VERSION` 등)에 걸려 있었는데, 첨부 감사보고서 4,525 섹션을 추가했을 때
    # 그 변경이 `build.py` 안에 있어 어떤 버전 문자열도 안 건드렸고 **산출물이 달라졌는데
    # build_id 가 그대로**였다. 버전을 올리는 걸 기억하는 데 의존하면 안 된다 (C-04).
    code_hash = _code_hash()
    build_id = hashlib.sha1(
        f"{manifest_hash}|{inventory_hash}|{code_hash}|{SCHEMA_VERSION}".encode()
    ).hexdigest()[:ID_HEX]

    w_src = _Writer(out_dir / "source_files.parquet", _arrow_schema(SourceFile))
    w_doc = _Writer(out_dir / "documents.parquet", _arrow_schema(Document))
    w_sec = _Writer(out_dir / "sections.parquet", _arrow_schema(Section))
    # 아래 넷은 근거 FK 검증이 ``evidence_id`` 로 필터하는 산출물이다. 정렬해 두지
    # 않으면 row group pruning 이 0건이 되어 조회가 2.2초가 된다 (`_Writer` 주석).
    w_chk = _Writer(out_dir / "chunks.parquet", _arrow_schema(Chunk),
                    sort_key="evidence_id")
    w_evd = _Writer(out_dir / "evidence.parquet", _arrow_schema(Evidence),
                    sort_key="evidence_id")
    w_fct = _Writer(out_dir / "facts.parquet", _arrow_schema(Fact),
                    sort_key="evidence_id")
    w_fld = _Writer(out_dir / "fields.parquet", _arrow_schema(Field),
                    sort_key="evidence_id")
    w_rel = _Writer(out_dir / "relations.parquet", _arrow_schema(Relation))
    w_cit = _Writer(out_dir / "correction_items.parquet", _arrow_schema(CorrectionItem))
    w_eid = _Writer(out_dir / "event_identities.parquet", _arrow_schema(EventIdentity))
    w_eob = _Writer(out_dir / "event_observations.parquet", _arrow_schema(EventObservation))
    # Cell 은 논리 좌표 단위라 행 수가 다른 엔티티보다 한 자릿수 크다. 기본은 끈다.
    w_cel = (_Writer(out_dir / "cells.parquet", _arrow_schema(Cell)) if with_cells else None)

    stats: dict[str, int] = {}
    failures: list[dict] = []
    #: 일부 열만 원문 모순 등으로 안전하게 구조화하지 못한 경우. 발행을 막거나
    #: 프로세스를 실패시키지는 않지만 Document를 partial unsupported로 표시하고
    #: run.json에 재현 가능한 locator를 남긴다.
    warnings: list[dict] = []

    # 상위 산출물이 **지금 코드·지금 코퍼스로** 만들어졌는지 먼저 본다 (B-03 · X-13).
    # 이 세션에서 낡은 `out/corrections/*.jsonl` 이 그대로 실려 구조적으로 틀린 항목
    # 796건이 산출물에 들어간 적이 있다. 조용히 낡은 것을 쓰는 것보다 멈추는 편이 낫다.
    for folder, producer in (("out/exchange", "exchange_events"),
                             ("out/forms", "forms"),
                             ("out/periodic", "periodic_relations"),
                             ("out/corrections", "correction_items")):
        stale = check_stamp(folder, producer, manifest_hash)
        if stale:
            failures.append({"doc_id": None, "stage": "stale_input", "reason": stale})

    def bump(key: str) -> None:
        stats[key] = stats.get(key, 0) + 1

    #: source_file_id → parse_mode. 본문·첨부가 각자 다를 수 있다
    file_modes: dict[str, str] = {}
    n_att_sections = 0
    n_split = n_over_budget = 0
    #: 본문 파싱 트리를 섹션↔Fact 사이에서만 공유한다. 문서를 넘기면 즉시 버린다 —
    #: 정기공시 트리는 문서당 수백 MB 라 쌓이면 4GB 서버에서 죽는다.
    trees: dict[str, object] = {}
    #: 정규화 사전에 없는 계정의 빈도. **무엇을 채워야 하는지 빌드가 알려준다.**
    unmatched_accounts: dict[str, int] = {}
    n_fact_rows = n_fact_norm = 0
    fact_status: dict[str, str] = {}
    #: 첨부가 담고 있는 재무제표 범위 — `(첨부)재 무 제 표` = 별도, `연 결` = 연결
    attach_scope: dict[str, set[str]] = defaultdict(set)
    #: 실제로 Fact 가 나온 범위
    fact_scope: dict[str, set[str]] = defaultdict(set)
    #: 섹션을 만든 파일 역할 — PDF 인지 viewer HTML 인지 가른다
    section_role: dict[str, set[str]] = defaultdict(set)
    #: 접수번호 → 안정 사건 식별 축. 금액·비율·기간은 events allowlist에서 거른다.
    event_identity_fields: dict[str, dict[str, set[str]]] = {}
    #: 접수번호 → 실제 Field 생성 시점에 채택한 `(evidence_id, support_role)`.
    #: 값/라벨 사후 검색으로 붙이지 않는다.
    event_support_fields: dict[str, set[tuple[str, str]]] = {}
    period_override_actions: list = []

    for record in records:
        # 트리는 **문서 하나 안에서만** 산다. 예외로 pop 이 건너뛰어도 여기서 끊긴다 —
        # 정기공시 트리는 문서당 수백 MB 라 쌓이면 4GB 서버에서 죽는다.
        trees.clear()
        doc_id = record["doc_id"]
        files = index.files(record["file_path"])
        main = index.main_xml(record["file_path"], record["rcept_no"])

        parse_mode = "unsupported" if record["doc_group"] == "exchange" else "strict"
        viewer_parsed: ParsedViewer | None = None
        viewer_results: dict[str, SourceParseResult] = {}
        viewer_selection: dict[str, str] = {}
        viewer_source_id_by_path: dict[str, str] = {}
        try:
            if record["file_format"] == "pdf+html":
                # DART XML 이 아니라 PDF + 뷰어 HTML 로만 제공되는 3건.
                # source를 각각 파싱하고 primary와 alternate를 모두 canonical에 남긴다.
                viewer_parsed, rows = _viewer_sections(record, index, build_id)
                parse_mode = viewer_parsed.parse_mode
                for result in viewer_parsed.source_results:
                    src_id = _viewer_source_id(record, index, result.source_path)
                    viewer_results[src_id] = result
                    viewer_source_id_by_path[str(result.source_path)] = src_id
                    viewer_selection[src_id] = (
                        "primary" if result is viewer_parsed.primary else "alternate")
                    # shell/failed source는 Section이 없어도 실제 mode를 잊지 않는다.
                    file_modes[src_id] = result.parse_mode
                    bump(f"viewer_source:{result.parse_mode}")
                    for message in result.warnings:
                        warnings.append({
                            "doc_id": doc_id, "source_file_id": src_id,
                            "stage": "viewer_source", "code": "source_parse_warning",
                            "parse_mode": result.parse_mode, "reason": message,
                            "source_sha256s": [
                                file_full_hashes.get(result.source_path, "")],
                            "guard": f"{result.parse_mode}|{message}",
                        })
                    if result.error:
                        warnings.append({
                            "doc_id": doc_id, "source_file_id": src_id,
                            "stage": "viewer_source", "code": "source_parse_error",
                            "parse_mode": result.parse_mode, "reason": result.error,
                            "source_sha256s": [
                                file_full_hashes.get(result.source_path, "")],
                            "guard": f"{result.parse_mode}|{result.error}",
                        })

                cross = viewer_parsed.cross_check
                bump(f"viewer_cross_check:{cross.status.value}")
                if cross.status != CrossCheckStatus.MATCHED:
                    cross_fields = _viewer_document_fields(
                        viewer_parsed, viewer_source_id_by_path)
                    warnings.append({
                        "doc_id": doc_id, "stage": "viewer_cross_check",
                        "code": f"source_cross_check_{cross.status.value}",
                        "source_file_ids": cross_fields[
                            "source_cross_check_source_file_ids"],
                        "reason": cross.reason,
                        "source_sha256s": sorted(
                            file_full_hashes.get(result.source_path, "")
                            for result in viewer_parsed.source_results),
                        "matched_claims": cross.matching_claims,
                        "conflict_claims": len(cross.conflicts),
                        "guard": _viewer_quality_guard(viewer_parsed),
                    })
                if not any(result.usable for result in viewer_parsed.source_results):
                    # parser가 파일별 오류를 typed result로 돌려주므로 예외는 없다.
                    # 사용 가능한 source가 0개인 문서만 명시적 blocking failure로 올린다.
                    failures.append({
                        "doc_id": doc_id, "stage": "parse",
                        "code": "viewer_no_usable_source",
                        "reason": "PDF/HTML 중 구조화 가능한 source가 없음",
                    })

                viewer_order = 0
                for row, result, selection in rows:
                    w_sec.add(row)
                    # fact_extract_status는 기본으로 선택한 source의 구조 수준을 말한다.
                    # alternate role을 섞으면 PDF primary인 문서가 HTML로 잘못 표시된다.
                    if selection == "primary":
                        section_role[record["doc_id"]].add(result.source_kind)
                    parts = split_section(row["text"])
                    for part in parts:
                        chunk_row, chunk_evidence_row = _chunk_row(
                            record, row, part, len(parts), build_id, viewer_order)
                        w_evd.add(chunk_evidence_row)
                        w_chk.add(chunk_row)
                        viewer_order += 1
                    if len(parts) > 1:
                        n_split += 1
                    n_over_budget += sum(1 for x in parts if x.over_budget)
                bump(f"section:{parse_mode}")
            elif record["doc_group"] == "periodic" and record["file_format"] == "xml":
                rows = list(_sections(record, index, build_id, failures, trees))
                parse_mode = rows[0][1] if rows else "strict"
                order = 0
                for row, mode, role in rows:
                    w_sec.add(row)
                    file_modes[row["source_file_id"]] = mode
                    section_role[record["doc_id"]].add(role)
                    if role == "attachment":
                        n_att_sections += 1
                        # 첨부에 있는 재무제표의 **범위**를 기록한다. 문서 단위로만 보면
                        # 「연결은 되고 별도는 안 됨」을 표현할 수 없다 — 그 7문서는
                        # Fact 가 있어서 문서 상태로는 `ok` 로 보인다.
                        sc = _attach_scope(str(row.get("title") or ""))
                        if sc:
                            attach_scope[record["doc_id"]].add(sc)
                    # Section 은 구조, Chunk 는 검색이다. Section 을 지우거나 줄이지 않고
                    # 그 위에 조각을 얹는다 (D2).
                    parts = split_section(row["text"])
                    for part in parts:
                        chunk_row, chunk_evidence_row = _chunk_row(
                            record, row, part, len(parts), build_id, order)
                        w_evd.add(chunk_evidence_row)
                        w_chk.add(chunk_row)
                        order += 1
                        if part.over_budget:
                            n_over_budget += 1
                    if len(parts) > 1:
                        n_split += 1
                bump(f"section:{parse_mode}")

                # 재무제표 Fact — **섹션 파싱에서 만든 트리를 재사용한다.**
                # 다시 읽고 다시 파싱하면 문서당 0.184s 를 버린다(1,051건 = 약 3분).
                fin_root = trees.pop(record["doc_id"], None)
                if fin_root is not None:
                    try:
                        n_fact = 0
                        fin_issues: list = []
                        fin_overrides: list = []
                        for row, evidence_row in _facts(
                                record, index, build_id, fin_root, fin_issues,
                                source_sha256=(file_full_hashes.get(main)
                                               if main is not None else None),
                                overrides_out=fin_overrides):
                            w_evd.add(evidence_row)
                            w_fct.add(row)
                            fact_scope[record["doc_id"]].add(row["scope"])
                            n_fact += 1
                            n_fact_rows += 1
                            if row["account_norm"]:
                                n_fact_norm += 1
                            else:
                                key = row["account_raw"][:40]
                                unmatched_accounts[key] = unmatched_accounts.get(key, 0) + 1
                        if n_fact:
                            bump("fact:추출된 문서")
                        period_override_actions.extend(fin_overrides)
                        for _action in fin_overrides:
                            bump("fact:period_override_applied")
                        if fin_issues:
                            status = ("unsupported_partial_layout" if n_fact
                                      else "unsupported_financial_layout")
                            fact_status[record["doc_id"]] = status
                            bump(f"fact:{status}")
                            for issue in fin_issues:
                                warnings.append({
                                    "doc_id": doc_id,
                                    "stage": "facts",
                                    "code": issue.code,
                                    "locator": issue.locator,
                                    "context": issue.context,
                                    "statement_hint": issue.statement_hint,
                                    "reason": issue.detail,
                                })
                        else:
                            fact_status[record["doc_id"]] = (
                                "ok" if n_fact else "no_statements")
                    except (ParseFailure, ET.ParseError) as exc:
                        failures.append({"doc_id": doc_id, "stage": "facts",
                                         "reason": f"{type(exc).__name__}: {exc}"[:160]})
                        # **추출기가 죽은 것을 「원문에 재무제표가 없다」로 적으면 안 된다.**
                        # 상태를 안 채우면 `_fact_status` 가 `no_statements` 로 떨어져,
                        # 우리 결함이 원문 특성으로 둔갑한다 (7차 검수).
                        fact_status[record["doc_id"]] = "extractor_failed"
                    finally:
                        fin_root = None       # 트리는 문서 단위로 버린다 (메모리)
            elif record["doc_group"] in ("exchange", "major", "holding"):
                for row, evidence_row in _fields(record, index, build_id):
                    w_evd.add(evidence_row)
                    w_fld.add(row)
                    add_identity_field(
                        event_identity_fields, record,
                        (row if not evidence_row.get("pii_types")
                         else {**row, "is_pii": True}),
                        support_fields=event_support_fields)
                bump(f"field:{record['doc_group']}")

        except (ParseFailure, UnsupportedFormat, FileNotFoundError, ET.ParseError) as exc:
            parse_mode = "failed"
            failures.append({"doc_id": doc_id, "stage": "parse",
                             "reason": f"{type(exc).__name__}: {exc}"[:160]})

        roles_seen: set[str] = set()

        # Cell 은 별도 단계다. 여기서 실패해도 이미 적재된 Section·Field 는 유효하므로
        # 문서 전체를 failed 로 표시하지 않는다.
        if w_cel is not None and main is not None and record["file_format"] != "pdf":
            try:
                for row in _cells(record, index, build_id):
                    w_cel.add(row)
                bump(f"cell:{record['doc_group']}")
            except (ParseFailure, UnsupportedFormat, FileNotFoundError, ET.ParseError) as exc:
                failures.append({"doc_id": doc_id, "stage": "cells",
                                 "reason": f"{type(exc).__name__}: {exc}"[:160]})

        for f in files:
            rel = f"{record['file_path']}/{f.name}"
            # `relpath`/ID는 schema 1.2 호환 NFC 키를 유지하고, 파일을 실제로 열 수 있는
            # 표기는 inventory가 준 actual_relpath에 별도 보존한다. NFC 키로 경로를
            # 재조립하지 않는다.
            binding = index.bind(rel)
            path_key = binding.normalized_path_key
            src_id = source_file_id(path_key)
            fmt, enc, status = _detect(f)
            role = ("main" if main and f == main
                    else "pdf" if f.suffix.lower() == ".pdf"
                    else "viewer_html" if "viewer" in f.name.lower()
                    else "attachment")
            viewer_result = viewer_results.get(src_id)
            if viewer_result is not None:
                # 파일명 heuristic보다 실제 parser source kind가 권위 있다.
                role = viewer_result.source_kind
            w_src.add(as_row(SourceFile(
                build_id=build_id, source_file_id=src_id, doc_id=doc_id,
                relpath=path_key, normalized_path_key=path_key,
                actual_relpath=binding.actual_relpath,
                # inventory hash 계산 때 이미 읽은 값. 5.3GB 원문을 다시 해싱하지 않는다.
                role=role, sha256=file_hashes[f], byte_size=f.stat().st_size,
                detected_format=fmt, detected_encoding=enc, decode_status=status,
                # 첨부도 파싱하므로 파일별 실제 모드를 쓴다
                parse_mode=file_modes.get(src_id,
                                          parse_mode if role == "main" else "not_parsed"),
                **_viewer_source_fields(viewer_result, viewer_selection.get(src_id)),
            )))
            bump(f"sourcefile:{role}")
            roles_seen.add(role)
            if status == "replaced":
                # 치환 디코딩은 **글자가 깨진 채로 통과한 것**이다. 성공으로 세지 않는다 (B-05).
                failures.append({"doc_id": doc_id, "stage": "decode",
                                 "reason": f"{rel}: UTF-8 치환 디코딩"})

        w_doc.add(as_row(Document(
            build_id=build_id, doc_id=doc_id, rcept_no=record["rcept_no"],
            corp_code=record["corp_code"], corp_name=record["corp_name"],
            listed_name=record["listed_name"], stock_code=record["stock_code"],
            sector=record["sector"], industry=record["industry"],
            doc_group=record["doc_group"], doc_subtype=record["doc_subtype"],
            event_type=_event_type(record["report_nm"], record["doc_group"]),
            report_nm=record["report_nm"],
            filer=record["flr_nm"], rcept_dt=record["rcept_dt"],
            base_year=record["base_year"], base_month=record["base_month"],
            is_correction=record["is_correction"], file_format=record["file_format"],
            n_source_files=len(files),
            fact_extract_status=_fact_status(
                record, fact_status, section_role[record["doc_id"]],
                attach_scope[record["doc_id"]], fact_scope[record["doc_id"]]),
            # **어느 범위를 못 읽었는지**까지 남긴다. 이게 없으면 조회가
            # 「연결은 되고 별도만 안 됨」을 말할 수 없어 회사 전체를 미지원 취급한다.
            fact_unsupported_scope="|".join(sorted(
                attach_scope[record["doc_id"]] - fact_scope[record["doc_id"]])) or None,
            **_viewer_document_fields(viewer_parsed, viewer_source_id_by_path),
        )))

    # 기존 관계 산출물을 canonical Relation 으로 통합
    raw_rel: list[dict] = []
    for src_path in (Path("out/exchange/relations.jsonl"),
                     Path("out/forms/relations.jsonl"),
                     Path("out/periodic/relations.jsonl")):
        if not src_path.exists():
            continue
        for line in src_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw_rel.append(json.loads(line))

    # `CORRECTS`(선언된 대상)에서 `SUPERSEDES`(파생된 순서)를 뽑는다.
    # 원공시에 정정이 둘 붙으면 CORRECTS 만으로는 **어느 쪽이 최신인지 알 수 없다** (R-03).
    rcept_dt_of = {r["rcept_no"]: r["rcept_dt"] for r in records}
    doc_id_by_rcept = {r["rcept_no"]: r["doc_id"] for r in records}
    raw_rel += derive_supersedes(raw_rel, rcept_dt_of)

    # 정정 항목이 붙을 간선 — 확정되고 FK까지 검증된 CORRECTS만 대상이다.
    corrects_edge: dict[str, str] = {}
    # Relation support는 Correction Evidence 생성 뒤에야 완성된다. 1,839개 정도의
    # 작은 edge만 메모리에 두고 Fact/Field/Chunk 스트리밍 메모리 계약은 유지한다.
    canonical_relations: list[Relation] = []
    for r in raw_rel:
        try:
            relation = _canonical_relation(r, doc_id_by_rcept, rcept_dt_of, build_id)
        except ValueError as exc:
            # endpoint가 없는 resolved 간선을 기록하면 canonical FK 계약을 깨뜨린다.
            # 행을 만들지 않고 blocking artifact failure로 발행 자체를 막는다.
            failures.append({
                "doc_id": doc_id_by_rcept.get(str(r.get("src_rcept_no") or "")),
                "stage": "artifact", "reason": str(exc),
            })
            continue
        canonical_relations.append(relation)
        if (relation.relation_type == "CORRECTS"
                and relation.resolution_status == "resolved"):
            corrects_edge[relation.src_rcept_no] = relation.relation_id
        key = f"relation:{r['relation_type']}:{r['resolution_status']}"
        if r.get("root_missing_reason"):
            key += f":{r['root_missing_reason']}"
        bump(key)

    # 사건 정체성 — 문서 여럿이 같은 사건을 말한다 (S-09)
    event_documents = [
        {**record, "event_type": _event_type(record["report_nm"], record["doc_group"])}
        for record in records
    ]
    identities, observations = build_identities(
        event_documents, raw_rel, bump, identity_fields=event_identity_fields,
        support_fields=event_support_fields)
    for i in identities:
        w_eid.add(as_row(EventIdentity(build_id=build_id, **i)))
        bump(f"event:{i['status_at_corpus_end']}")
        bump(f"event_identity:{i['identity_status']}")
    for o in observations:
        w_eob.add(as_row(EventObservation(build_id=build_id, **o)))

    # 정정 항목 — `src.ingest.correction_items.build()` 산출물을 canonical 로 옮긴다
    main_src: dict[str, str] = {}
    for record in records:
        main = index.main_xml(record["file_path"], record["rcept_no"])
        if main is not None:
            main_src[record["rcept_no"]] = source_file_id(
                nfc(f"{record['file_path']}/{main.name}"))

    # 접수번호별 CorrectionItem side Evidence. Relation support는 source document
    # 소유인 이 ID만 쓴다.
    relation_correction_support: dict[str, set[tuple[str, str]]] = defaultdict(set)
    relation_correction_support_owners: dict[
        str, set[tuple[str, int]]
    ] = defaultdict(set)
    items_path = Path("out/corrections/correction_items.jsonl")
    if items_path.exists():
        by_doc: dict[str, int] = {}
        # 같은 실제 셀·값이면 같은 `(source, kind, side locator, excerpt_hash)` Evidence를 공유한다.
        # 전/후 값이 같아도 서로 다른 셀이면 위치가 다르므로 Evidence ID도 달라야 한다.
        # 이 작은 집합만 메모리에 둬 220만 Fact/Field Evidence의 스트리밍을 방해하지 않는다.
        correction_evidence_seen: dict[str, dict] = {}
        for line in items_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            seq = by_doc.get(c["doc_id"], 0)
            by_doc[c["doc_id"]] = seq + 1
            loc = c["locator"]
            # **본문 파일의 ID** 를 쓴다. 예전에는 `source_file_id(locator)` 였는데,
            # locator 는 파일 경로가 아니라 원문 위치라서 **어디에도 조인되지 않는 가짜 ID**가
            # 만들어졌다. `block_id` 도 저장된 locator 와 다른 문자열로 계산해
            # 재계산 검증이 전량 불일치였다 (L-07).
            src = main_src.get(c["rcept_no"])
            if src is None:
                failures.append({"doc_id": c["doc_id"], "stage": "correction_item",
                                 "reason": "본문 파일 ID 없음"})
                continue

            try:
                before_locator, after_locator = _correction_side_locators(c)
                (row_coordinates, before_coordinates,
                 after_coordinates) = _correction_semantic_coordinates(
                    c, before_locator, after_locator)
            except ValueError as exc:
                # schema 1.7 이하 correction JSON을 대표 행 locator로 보정하면 after가
                # before 셀을 가리키는 기존 결함을 재현한다. 추측하지 않고 재추출을 요구한다.
                failures.append({
                    "doc_id": c["doc_id"], "stage": "artifact",
                    "reason": str(exc),
                })
                continue

            def correction_evidence(value: str | None,
                                    side_locator: str | None,
                                    semantic_coordinates: tuple[
                                        int | None, int | None]) -> str | None:
                if value is None:
                    return None
                # 위 gate에서 값이 있는 side의 locator를 보장했다. cast/fallback으로
                # 대표 locator를 재사용하지 않는다 — 비정규 locator도 make_evidence가 닫는다.
                if side_locator is None:
                    raise RuntimeError("값이 있는 CorrectionItem side locator 없음")
                side_table_loc = table_locator_of(side_locator)
                side_row, side_col = semantic_coordinates
                if side_row is None or side_col is None:
                    raise RuntimeError("값이 있는 CorrectionItem semantic 좌표 없음")
                evidence = make_evidence(
                    build_id=build_id, doc_id=c["doc_id"], source_file_id=src,
                    kind="correction_value", locator=side_locator,
                    table_locator=side_table_loc,
                    logical_row=side_row, logical_col=side_col,
                    excerpt_raw=value, extraction_method="declared_correction_table",
                    rcept_dt=rcept_dt_of.get(c["rcept_no"], ""), pii_label=c["item_path"],
                    corporate_names=(c.get("corp_name"),),
                )
                evidence_row = as_row(evidence)
                previous = correction_evidence_seen.get(evidence.evidence_id)
                if previous is None:
                    correction_evidence_seen[evidence.evidence_id] = evidence_row
                    w_evd.add(evidence_row)
                elif previous != evidence_row:
                    raise RuntimeError(
                        f"Evidence ID collision: {evidence.evidence_id} ({c['doc_id']})"
                    )
                return evidence.evidence_id

            before_evidence_id = correction_evidence(
                c.get("value_before"), before_locator, before_coordinates)
            after_evidence_id = correction_evidence(
                c.get("value_after"), after_locator, after_coordinates)
            if (before_evidence_id
                    and not correction_evidence_seen[before_evidence_id]["pii_types"]):
                relation_correction_support_owners[before_evidence_id].add(
                    (c["doc_id"], seq))
                relation_correction_support[c["rcept_no"]].add(
                    (before_evidence_id, "correction_before"))
            if (after_evidence_id
                    and not correction_evidence_seen[after_evidence_id]["pii_types"]):
                relation_correction_support_owners[after_evidence_id].add(
                    (c["doc_id"], seq))
                relation_correction_support[c["rcept_no"]].add(
                    (after_evidence_id, "correction_after"))
            # **선언된 변경 항목이 어느 정정 간선의 diff 인지 잇는다.** 스키마는
            # `parent_id = Relation.relation_id` 라 계약해 놓고 빌더가 늘 `None` 이었다 —
            # Fact 의 가짜 FK 와 같은 부류다(3차 검수 ④). 원본을 못 찾은 정정
            # (`root_missing`·`ambiguous`)은 이을 대상이 없으므로 `None` 이 맞다.
            w_cit.add(as_row(CorrectionItem(
                build_id=build_id, doc_id=c["doc_id"], source_file_id=src,
                block_id=block_id(c["doc_id"], src, loc),
                parent_id=corrects_edge.get(c["rcept_no"]),
                path=c["item_path"], order=seq, locator=loc,
                rcept_no=c["rcept_no"], corp_name=c["corp_name"],
                doc_group=c["doc_group"], reason=c["reason"],
                value_before=c["value_before"], value_after=c["value_after"],
                before_locator=before_locator, after_locator=after_locator,
                logical_row=row_coordinates[0], logical_col=row_coordinates[1],
                before_logical_row=before_coordinates[0],
                before_logical_col=before_coordinates[1],
                after_logical_row=after_coordinates[0],
                after_logical_col=after_coordinates[1],
                before_evidence_id=before_evidence_id,
                after_evidence_id=after_evidence_id,
                diff_kind=c["diff_kind"], before_kind=c["before_kind"],
                after_kind=c["after_kind"],
                required_by_authority=c["required_by_authority"],
                rcept_dt=rcept_dt_of.get(c["rcept_no"], ""),
            )))
            bump(f"correction_item:{c['diff_kind']}/{c['before_kind']}→{c['after_kind']}")

    # Relation은 source 문서의 실제 Field/Correction Evidence를 연결하되, 현재 raw
    # relation producer가 exact target locator Evidence를 만들지 않으므로 resolved도
    # fully_verified라고 꾸미지 않는다. source support는 downstream 재검토 범위를 줄이고
    # typed limitation은 남은 결손을 정확히 보존한다.
    for relation in canonical_relations:
        raw_support = set(event_support_fields.get(relation.src_rcept_no, ()))
        correction_support, excluded_correction_support = (
            filter_relation_correction_support_pairs(
                relation_correction_support.get(relation.src_rcept_no, ()),
                relation_correction_support_owners,
            )
        )
        raw_support.update(correction_support)
        try:
            pairs = normalise_relation_support_pairs(raw_support)
        except ValueError as exc:
            raise RuntimeError(
                f"Relation support 정규화 실패: {relation.relation_id}: {exc}") from exc
        final_relation = replace(
            relation,
            supporting_evidence_ids=[item[0] for item in pairs],
            support_roles=[item[1] for item in pairs],
            support_status="partial",
            support_version=RELATION_SUPPORT_VERSION,
            support_limitation=relation_support_limitation(
                relation.resolution_status,
                has_source_evidence=bool(pairs),
                has_ambiguous_correction_evidence_owner=bool(
                    excluded_correction_support),
            ),
        )
        w_rel.add(as_row(final_relation))

    # 계정 정규화 커버리지 리포트 — 사전을 넓히는 다음 작업 목록이다.
    if n_fact_rows:
        top = sorted(unmatched_accounts.items(), key=lambda kv: -kv[1])[:200]
        repo_root = Path(__file__).resolve().parents[2]
        map_file = ACCOUNT_MAP_PATH.resolve().relative_to(repo_root).as_posix()
        (out_dir / "account_coverage.json").write_text(json.dumps({
            "map_file": map_file,
            "map_sha256": _sha256(ACCOUNT_MAP_PATH),
            "facts": n_fact_rows,
            "normalized": n_fact_norm,
            "coverage": round(n_fact_norm / n_fact_rows, 4),
            "unmatched_distinct": len(unmatched_accounts),
            "top_unmatched": [{"account_raw": k, "count": v} for k, v in top],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        stats["fact:정규화 커버리지(%)"] = round(100 * n_fact_norm / n_fact_rows)

    if n_att_sections:
        stats["section:attachment"] = n_att_sections
    if n_split:
        stats["chunk:분할된 섹션"] = n_split
        stats["chunk:over_budget"] = n_over_budget

    written = [("source_files", w_src), ("documents", w_doc), ("sections", w_sec),
               ("chunks", w_chk), ("evidence", w_evd),
               ("facts", w_fct), ("fields", w_fld), ("relations", w_rel),
               ("correction_items", w_cit), ("event_identities", w_eid),
               ("event_observations", w_eob)]
    if w_cel is not None:
        written.append(("cells", w_cel))
    hashes = {name: w.close() for name, w in written}
    counts = {name: w.count for name, w in written}

    # 산출물 검증 — 작성기가 센 행 수와 파일에 실제로 든 행 수가 같아야 한다.
    # `w_cel` 을 close 목록에서 빠뜨렸을 때 마지막 부분 배치 925행이 조용히 사라졌고
    # 아무것도 그것을 잡지 못했다. 리포트 숫자만 보면 정상으로 보인다.
    for name, w in written:
        if w.count == 0:
            continue
        actual = pq.ParquetFile(w.path).metadata.num_rows
        if actual != w.count:
            failures.append({"doc_id": None, "stage": "artifact",
                             "reason": f"{name}: 기록 {w.count:,} != 파일 {actual:,}"})

    run = Run(
        build_id=build_id, schema_version=SCHEMA_VERSION, started_at=started,
        corpus_inventory_hash=inventory_hash,
        manifest_hash=manifest_hash, sanitizer_version=SANITIZER_VERSION,
        security_policy_version=SECURITY_POLICY_VERSION,
        parser_version=PARSER_VERSION, chunker_version=CHUNKER_VERSION,
        code_hash=code_hash,
        config={"corpus_root": str(corpus_root),
                "default_llm_text_column": "chunks.text_prompt_safe",
                "default_search_text_column": "chunks.text_search",
                "default_field_value_column": "fields.value_prompt_safe",
                "evidence_id_policy": EVIDENCE_POLICY_VERSION,
                "default_evidence_excerpt_column": "evidence.excerpt_safe",
                "chunk_evidence_kind": "chunk_text",
                "event_support_version": EVENT_SUPPORT_VERSION,
                "relation_support_version": RELATION_SUPPORT_VERSION,
                "quality_policy_version": QUALITY_POLICY_VERSION},
        artifact_hashes={k: v for k, v in hashes.items() if v},
    )

    quality_observations = list(warnings)
    quality_observations.extend({
        "doc_id": action.doc_id,
        "stage": "facts",
        "code": "period_override_applied",
        "source_sha256s": [action.source_sha256],
        "locator": action.table_locator,
        "guard": action.raw_header,
    } for action in period_override_actions)
    try:
        quality = evaluate_quality_decisions(
            quality_observations, policy_path=Path(quality_policy_path))
    except (OSError, ValueError) as exc:
        quality = {
            "policy_version": QUALITY_POLICY_VERSION,
            "policy_path": policy_path_label(quality_policy_path),
            "observed": len(quality_observations), "applied": [],
            "unresolved": [{"reason": f"quality policy 오류: {exc}"}],
            "stale": [], "passed": False,
        }
    if not quality["passed"]:
        failures.append({
            "doc_id": None, "stage": "quality_gate",
            "code": "unresolved_source_quality",
            "reason": (f"미승인 관측 {len(quality['unresolved'])}건, "
                       f"stale 결정 {len(quality['stale'])}건"),
        })
    report = {**asdict(run), "counts": counts,
              "stats": dict(sorted(stats.items())), "warnings": warnings,
              "quality": quality, "failures": failures}
    # **차단성 실패가 있으면 교체하지 않는다.** 예전에는 실패 여부와 무관하게 rename 해서
    # 깨진 빌드가 정상 산출물을 덮었다. 종료코드만 1 이어도 파일은 이미 바뀐 뒤였다.
    return finalize(report, failures, out_dir, final_dir, lock)


def finalize(report: dict, failures: list[dict], out_dir: Path, final_dir: Path,
             lock: Path) -> dict:
    """발행할지 붙들지 결정하고, 발행이면 **원자적으로** 교체한다.

    `build()` 안에 묻혀 있어 **실패 빌드를 만들어 보는 검사를 쓸 수 없었다** —
    게이트가 「지금 빌드의 `published` 가 True 인가」만 봤다 (6차 검수).
    분리해 두면 검사가 두 갈래를 다 지나갈 수 있다.
    """
    blocking = [f for f in failures if f.get("stage") in _BLOCKING_STAGES]
    if blocking:
        lock.unlink(missing_ok=True)
        report["published"] = False
        report["blocked_by"] = blocking[:20]
        report["staging_dir"] = str(out_dir)
        (out_dir / "run.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    # Evidence writer는 Fact·Field·Correction 순서로 스트리밍되므로 각 생산자 안의
    # 지역 dedupe만으로는 종류 간/단계 간 중복을 증명할 수 없다. 최종 Parquet 전체를
    # 하나의 디스크 primary key로 확인한 뒤에만 published=True를 기록한다.
    evidence_artifact = out_dir / "evidence.parquet"
    if evidence_artifact.exists():
        try:
            _validate_evidence_id_uniqueness(evidence_artifact)
            schema_parts = tuple(
                int(part) for part in str(report.get("schema_version") or "").split(".")
            )
            if schema_parts >= (1, 9):
                _validate_schema19_evidence_contract(
                    out_dir, str(report.get("build_id") or ""))
        except Exception as exc:
            failure = {
                "doc_id": None, "stage": "artifact",
                "reason": f"prepublish Evidence/schema1.9 contract: {exc}",
            }
            failures.append(failure)
            report["failures"] = failures
            report["published"] = False
            report["blocked_by"] = [failure]
            report["staging_dir"] = str(out_dir)
            report.pop("output_dir", None)
            (out_dir / "run.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            lock.unlink(missing_ok=True)
            return report

    # `published` 는 교체 **직전에** 확정해 run.json 에 담는다. 예전에는 run.json 을 먼저
    # 쓰고 나중에 필드를 채워 **디스크에는 남지 않았다** — 게이트가 읽을 값이 없었다.
    report["published"] = True
    report["output_dir"] = str(final_dir)
    (out_dir / "run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # rename 전에 실제 runtime과 같은 validator를 staging 발행본에 통과시킨다.
    # hash/count만 맞아도 필수 컬럼이 빠진 반쪽 schema면 기존 final을 보존한다.
    try:
        validate_canonical_artifacts(out_dir)
    except ArtifactIntegrityError as exc:
        failure = {"doc_id": None, "stage": "artifact",
                   "reason": f"prepublish runtime validation: {exc}"}
        failures.append(failure)
        report["published"] = False
        report["blocked_by"] = [failure]
        report["staging_dir"] = str(out_dir)
        report.pop("output_dir", None)
        (out_dir / "run.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        lock.unlink(missing_ok=True)
        return report
    # 원자적 교체 — 이전 산출물을 옆으로 밀고 새것을 제자리에 놓은 뒤 지운다.
    previous = final_dir.with_name(final_dir.name + ".previous")
    shutil.rmtree(previous, ignore_errors=True)
    if final_dir.exists():
        final_dir.rename(previous)
    out_dir.rename(final_dir)
    shutil.rmtree(previous, ignore_errors=True)
    lock.unlink(missing_ok=True)
    return report


def _lock_holder(lock: Path) -> int | None:
    """잠금 파일에 적힌 PID. 읽을 수 없으면 `None`."""
    try:
        return int(lock.read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _alive(pid: int) -> bool:
    """그 PID 가 아직 살아 있는가. 신호 0 은 존재 확인만 한다."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # 남의 프로세스지만 살아 있다
    return True


def _attach_scope(title: str) -> str | None:
    """첨부 재무제표 제목 → 범위. 글자 사이 공백이 들어간다(`(첨부)연 결 재 무 제 표`)."""
    squeezed = "".join(title.split())
    if not squeezed.endswith("재무제표"):
        return None
    return "CFS" if "연결" in squeezed else "SFS"


def _fact_status(record: dict, seen: dict[str, str], section_roles: set[str],
                 attach_scope: set[str], fact_scope: set[str]) -> str:
    """문서별 재무 Fact 추출 결과.

    **「값 없음」과 「추출 미지원」은 다르다.** 전자는 원문에 재무제표가 없는 것이고,
    후자는 있는데 우리 추출기가 그 구조를 못 읽는 것이다. 조회가 둘을 같은 답으로
    주면 「없다」는 틀린 답이 된다 (4차 검수 조건부 수용 조건).
    """
    if record["doc_group"] != "periodic":
        return "not_applicable"              # 정기공시만 Fact 를 만든다
    got = seen.get(record["doc_id"])
    # **추출기가 죽은 것이 최우선이다.** 아래 어떤 분기로도 덮이면 안 된다.
    if got == "extractor_failed":
        return "extractor_failed"
    # 본문이 XML 이 아니면 추출기가 애초에 돌지 않는다. **섹션을 만든 파일**로 가른다 —
    # 셋 다 PDF 와 viewer HTML 을 모두 갖고 있어 파일 유무로는 구분되지 않는다.
    if not got and section_roles and "main" not in section_roles:
        if "viewer_html" in section_roles:
            return "unsupported_html"
        if "pdf" in section_roles:
            return "unsupported_pdf"
    # **첨부에 있는데 그 범위의 Fact 가 없다.** 문서 전체로는 `ok` 여도
    # 「별도는 못 읽었다」를 말해야 한다.
    if attach_scope - fact_scope:
        return "unsupported_attachment_layout"
    if got == "ok":
        return "ok"
    return got or "no_statements"


def _release_lock(out_dir: str | Path = "out/canonical") -> None:
    """빌드가 비정상 종료했을 때 잠금을 푼다."""
    Path(out_dir).with_suffix(".lock").unlink(missing_ok=True)


def _main() -> int:
    ap = argparse.ArgumentParser(description="canonical 통합 빌드")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="out/canonical")
    ap.add_argument("--with-cells", action="store_true",
                    help="표 셀을 논리 좌표 단위로 적재 (행 수가 한 자릿수 크다)")
    args = ap.parse_args()

    try:
        r = build(args.corpus, args.out, with_cells=args.with_cells)
    except BaseException:
        # 잠금이 남으면 다음 빌드가 시작조차 못 한다. 실패해도 반드시 푼다.
        _release_lock(args.out)
        shutil.rmtree(Path(args.out).with_name(Path(args.out).name + ".building"),
                      ignore_errors=True)
        raise
    print(f"build_id {r['build_id']}  schema {r['schema_version']}")
    for k, v in r["counts"].items():
        print(f"  {k:<16}{v:>10,}")
    print("\n  " + "\n  ".join(f"{k:<34}{v:>8,}" for k, v in r["stats"].items()))
    if r["failures"]:
        print(f"\n실패 {len(r['failures'])}건")
        for f in r["failures"][:5]:
            print("   ", f)
    # 실패가 있으면 0 을 주지 않는다. 성공 종료로 보이면 하위 단계가 낡은 산출물을
    # 최신인 줄 알고 쓴다 (C-04).
    return 1 if r["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(_main())
