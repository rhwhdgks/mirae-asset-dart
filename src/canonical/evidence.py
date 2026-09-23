"""First-class Evidence 생성 계약.

Evidence ID는 semantic Fact/Field/CorrectionItem ID와 독립적이다. 같은 근거 종류에서 같은
원문 파일의 같은 canonical locator와 같은 raw excerpt를 인용하면 언제 만들어도 같은 ID가
되고, 근거 종류나 excerpt가 바뀌면 Evidence ID만 바뀐다.

``evidence_id = sha256(source_file_id | kind | canonical_locator | excerpt_hash)[:32]``

``excerpt_raw``는 감사·재현용 restricted 데이터다. Agent/LLM에는 #5 보안 정책으로 만든
``excerpt_safe``만 기본 전달해야 한다.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .schema import Evidence
from .security import project_chunk_text, project_field_value

__all__ = [
    "EVIDENCE_POLICY_VERSION",
    "CanonicalLocatorError",
    "canonical_locator",
    "excerpt_hash",
    "evidence_id",
    "table_locator_of",
    "logical_position_of",
    "make_evidence",
]


EVIDENCE_POLICY_VERSION = "evidence/1.1-kind-bound"
_ID_HEX = 32
_LOCATOR_RE = re.compile(r"[A-Za-z0-9_-]+\[\d+\](?:/[A-Za-z0-9_-]+\[\d+\])*\Z")
_ROW_RE = re.compile(r"(?:^|/)(?:TR|ROW)\[(\d+)\]")
_COL_RE = re.compile(r"(?:^|/)(?:TD|TH|TE|TU|CELL)\[(\d+)\]")


class CanonicalLocatorError(ValueError):
    """Evidence가 안정적으로 재계산될 수 없는 locator를 받은 경우."""


def canonical_locator(locator: str) -> str:
    """저장된 canonical locator를 검증해 그대로 돌려준다.

    공백을 지우거나 segment를 재해석하지 않는다. 그런 보정은 잘못된 위치를 정상처럼 만드는
    조용한 실패이므로 비정규 입력은 fail-closed 한다.
    """
    value = str(locator)
    if not _LOCATOR_RE.fullmatch(value):
        raise CanonicalLocatorError(f"canonical locator가 아닙니다: {value!r}")
    return value


def excerpt_hash(excerpt_raw: str) -> str:
    """restricted raw excerpt의 전체 SHA-256."""
    return hashlib.sha256(str(excerpt_raw).encode("utf-8")).hexdigest()


def evidence_id(source_file_id: str, kind: str, locator: str, raw_hash: str) -> str:
    """source file + kind + canonical locator + excerpt hash 기반 128-bit 안정 ID."""
    loc = canonical_locator(locator)
    payload = f"{source_file_id}|{kind}|{loc}|{raw_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_ID_HEX]


def table_locator_of(locator: str) -> str | None:
    """cell locator에서 TABLE segment까지의 canonical locator."""
    loc = canonical_locator(locator)
    segments = loc.split("/")
    for index, segment in enumerate(segments):
        if segment.startswith("TABLE["):
            return "/".join(segments[:index + 1])
    return None


def logical_position_of(locator: str) -> tuple[int | None, int | None]:
    """locator에 명시된 마지막 row/column 순번. 없으면 ``None``."""
    loc = canonical_locator(locator)
    rows = _ROW_RE.findall(loc)
    cols = _COL_RE.findall(loc)
    return (int(rows[-1]) if rows else None, int(cols[-1]) if cols else None)


def make_evidence(
        *, build_id: str, doc_id: str, source_file_id: str, kind: str,
        locator: str, excerpt_raw: str, extraction_method: str, rcept_dt: str,
        table_locator: str | None = None, logical_row: int | None = None,
        logical_col: int | None = None, extraction_status: str = "ok",
        corporate_names: Iterable[str] = (), pii_label: str | None = None,
        pii_subject_name: str | None = None,
        pii_party_type: str | None = None,
        security_policy_version: str | None = None) -> Evidence:
    """raw excerpt 하나를 canonical Evidence 행으로 만든다."""
    loc = canonical_locator(locator)
    if table_locator is not None:
        table_locator = canonical_locator(table_locator)
        if not (loc == table_locator or loc.startswith(table_locator + "/")):
            raise CanonicalLocatorError(
                f"table locator가 evidence locator의 조상이 아닙니다: {table_locator!r} / {loc!r}"
            )

    raw = str(excerpt_raw)
    raw_hash = excerpt_hash(raw)
    # Field/정정 값은 라벨이 PII 유형을 말해 주므로 값 단독 regex보다 먼저 적용한다.
    # `pii_subject_name` 은 그 값이 누구의 것인지, `pii_party_type` 은 그
    # 주체가 자연인인지 법인인지다. 여기서 빠뜨리면 안전 본문
    # (`excerpt_safe`)만 가려진 채 저장돼, 읽기의 재검증이 값과 본문의
    # 어긋남으로 닫힌다 (이슈 #139·#199).
    field_projection = (project_field_value(
        pii_label, raw, corporate_names, subject_name=pii_subject_name,
        party_type=pii_party_type,
        security_policy_version=security_policy_version)
                        if pii_label is not None else None)
    safe_input = field_projection.value_masked if field_projection is not None else raw
    projection = project_chunk_text(
        safe_input, corporate_names, security_policy_version=security_policy_version)
    pii_types = set(projection.pii_types)
    flags = set(projection.security_flags)
    if field_projection is not None and field_projection.pii_type:
        pii_types.update(field_projection.pii_type.split("|"))
        flags.add("pii_masked")
    return Evidence(
        build_id=build_id,
        evidence_id=evidence_id(source_file_id, kind, loc, raw_hash),
        doc_id=doc_id,
        source_file_id=source_file_id,
        kind=kind,
        locator=loc,
        table_locator=table_locator,
        logical_row=logical_row,
        logical_col=logical_col,
        excerpt_raw=raw,
        excerpt_safe=projection.text_prompt_safe,
        excerpt_hash=raw_hash,
        raw_access="restricted",
        extraction_method=extraction_method,
        extraction_status=extraction_status,
        security_flags=sorted(flags),
        pii_types=sorted(pii_types),
        security_policy_version=projection.security_policy_version,
        evidence_policy_version=EVIDENCE_POLICY_VERSION,
        rcept_dt=rcept_dt,
    )
