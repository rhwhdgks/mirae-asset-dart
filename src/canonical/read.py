"""canonical 산출물 조회 계약 — 전처리 ↔ Agent 경계.

**이 모듈이 인계의 실체다.** parquet 9개를 넘기고 "알아서 조인하세요" 하면 전처리 담당의
일을 넘기는 것이다. 조회 규칙을 여기 한 곳에 둔다.

## 세 가지 원칙

**하나. `as_of` 는 선택이 아니라 필수다.**

평가질문 70개 중 18개가 「2025년 12월 16일까지 공개된 공시만 보면」 같은 시점 표현을 쓴다.
선택 인자로 두면 안 넘길 수 있고, 안 넘기면 **미래 문서가 샌다** — 그 오답은 조용하다.
그래서 모든 조회 메서드가 `as_of` 를 요구한다. 코퍼스 전체를 보려면 `AS_OF_ALL` 을 명시한다.

**둘. 단위를 미리 곱하지 않되, 환산은 여기서만 한다.**

`raw_value` 와 `raw_unit` 을 그대로 두는 것이 저장 원칙이다(단위 판정이 틀렸을 때 원문 값까지
오염되지 않게). 다만 환산 규칙이 호출부마다 흩어지면 **팀원과 전처리의 답이 갈린다.**
`Money.in_won()` 하나만 쓴다.

**셋. 「결과 없음」과 「미구현」을 구분한다.**

`NotReadyError` 는 산출물이 아직 없다는 뜻이고, 빈 리스트는 「찾았는데 없다」는 뜻이다.
조용히 빈 값을 주면 Agent 가 둘을 구분할 수 없다.

## 시점 관점

정정본이 여러 개인 값은 관점에 따라 답이 다르다. **어느 쪽이 맞는지는 질문이 정한다.**

| view | 뜻 |
|---|---|
| `as_filed` | 그 기간을 **처음 보고한** 문서의 값 |
| `restated` | `as_of` 까지 나온 문서 중 **가장 나중** 값 |

실측 (회사·계정경로·기간) 조합 33.6만 중 36.8%를 문서 둘 이상이 보고하고, 그 중 1.94%는
값이 다르다. 카카오 `유형자산 2023-12-31` 은 2024 사업보고서가 전기 비교치를 재작성해
`1,322,051,482,382` → `1,336,929,825,767` 로 바뀐다.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import threading
import unicodedata
from typing import Any, Iterable, Iterator, Literal, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as _ds
import pyarrow.parquet as pq

from src.artifact import (
    ArtifactIntegrityError, ArtifactIssue, file_sha256, validate_canonical_artifacts,
)
from src.ingest.corpus_paths import (
    CorpusIndex, CorpusPathNotFound, PathCollisionError, normalized_path_key,
)
from src.canonical.evidence import (
    EVIDENCE_POLICY_VERSION,
    CanonicalLocatorError,
    canonical_locator,
    evidence_id as calculate_evidence_id,
    excerpt_hash as calculate_excerpt_hash,
    make_evidence,
    table_locator_of,
)
from src.canonical.events import (
    EVENT_SUPPORT_ROLES, EVENT_SUPPORT_VERSION, RELATION_SUPPORT_ROLES,
    RELATION_SUPPORT_VERSION, event_support_decision, event_support_role,
    is_declared_termination, relation_correction_support_owner_keys,
    relation_correction_support_sides, relation_support_limitation,
    validate_event_history,
)
from src.canonical.lineage import LatestResolution, resolve_latest_known
from src.canonical.schema import (
    block_id as calculate_block_id,
    source_file_id as calculate_source_file_id,
)
from src.canonical.security import (
    PROMPT_DATA_BEGIN,
    PROMPT_DATA_END,
    SECURITY_POLICY_VERSION,
    project_chunk_text,
    project_field_value,
    subject_names_by_row,
    needs_subject_name,
    holding_party_types_by_row,
    needs_party_type,
    resolve_party_type,
)

_READ_COMPAT_SECURITY_POLICIES = frozenset({
    "pii-prompt-safe/1.1", "pii-prompt-safe/1.2", "pii-prompt-safe/1.3",
    "pii-prompt-safe/1.4"})

#: 「이 문서에는 `구분` 지도가 없다」를 캐시에 적는 표지. ``None`` 을 그대로
#: 넣으면 미조회와 구별되지 않아 지분공시가 아닌 문서마다 documents 조회가
#: 다시 일어난다.
_NO_PARTY_TYPES = object()

__all__ = [
    "AS_OF_ALL", "Money", "FactRow", "FactLookup", "FieldRow", "FieldLookup",
    "DocumentMetadataRow",
    "CorrectionItemRow", "EvidenceRow", "EventState", "EventObservationRow", "EventTimeline",
    "RelationSupportRow", "RelationSummaryRow",
    "Company", "ChunkRow", "SectionText", "SourceFileRow", "NotReadyError",
    "AmbiguousSectionError",
    "AmbiguousEventError", "EvidenceIntegrityError", "ReadSecurityError",
    "SourceIntegrityError",
    "ArtifactIssue", "ArtifactIntegrityError",
    "LatestResolution", "CanonicalReadModel",
]

#: 시점 제한 없음을 **명시적으로** 나타낸다. 기본값으로 두지 않는 이유는
#: 「안 넘겨서 전체가 보이는 것」과 「전체를 보려고 한 것」을 구분하기 위해서다.
AS_OF_ALL = "99991231"

#: 원문 단위 → 원 배수. 실측 분포는 `원` 345,061 · `백만원` 155,917 · `천원` 49,106.
_UNIT_SCALE: dict[str, int] = {
    "원": 1, "천원": 1_000, "백만원": 1_000_000, "십억원": 1_000_000_000,
    "억원": 100_000_000, "조원": 1_000_000_000_000,
}

View = Literal["as_filed", "restated"]

#: 계정명 뒤의 주석 표시. **보고서마다 번호가 다르다** — 같은 매출액이 2024 보고서에선
#: `매출액 (주29)`, 2025 보고서에선 `매출액 (주30)` 이다. 경로 비교에서 떼지 않으면
#: 같은 계정을 서로 다른 계정으로 보고 조회가 모호하다고 잘못 판정한다.
_FOOTNOTE = re.compile(r"\(\s*주\s*[\d,\s.]*\)")

#: 서식명 비교에서 표현만 다른 가운데점과 공백은 의미가 없다. 반면 Field의 ``path``
#: 비교는 문자열 포함 검색과 정확 검색을 분리해 호출자가 어느 쪽을 썼는지 알 수 있게 한다.
_CORRECTION_PREFIX = re.compile(r"^\s*\[[^]]+\]\s*")
_FORM_WRAPPER = re.compile(r"^주요사항보고서\((.*)\)$")


def _path_key(path: str) -> str:
    """계정 경로 비교용 키. 주석 번호와 공백을 뗀다."""
    return re.sub(r"\s+", "", _FOOTNOTE.sub("", path or ""))


def _text_key(text: str | None) -> str:
    """사용자 필터 비교용 키. NFC·공백·가운데점 표기차만 흡수한다."""
    value = unicodedata.normalize("NFC", str(text or "")).casefold()
    return re.sub(r"[\sㆍ·･]", "", value)


def _field_label(path: str) -> str:
    """Field path의 가장 오른쪽 라벨."""
    return re.split(r"\s*>\s*", path or "")[-1].strip()


def _report_form(report_nm: str | None) -> str:
    """정정 태그와 ``주요사항보고서(...)`` 껍질을 뗀 서식명."""
    value = _CORRECTION_PREFIX.sub("", str(report_nm or "")).strip()
    matched = _FORM_WRAPPER.fullmatch(value)
    return matched.group(1).strip() if matched else value


#: 회계 기수 표기 — 「제 50 기」·「제50기말」·「제 55 기 1분기」.
#: 기수 번호와 그 뒤의 시점 수식(말·N분기·반기)까지 한 덩어리로 잡는다.
#: 수식이 다르면 가리키는 시점도 다르므로 따로 짝지어야 한다.
PERIOD_LABEL = re.compile(
    r"제\s*[0-9]{1,3}\s*기"
    r"(?:\s*(?:말|반기|상반기|[1-4]\s*분기(?:\s*말)?|[1-4]\s*/4\s*분기))?")


def period_label_key(value: object) -> str:
    """기수 표기의 비교용 형태. 공백만 지운다 — 수식은 표면에 쓰려고 남긴다."""

    text = str(value or "").strip()
    if not text:
        return ""
    matched = PERIOD_LABEL.fullmatch(text)
    return re.sub(r"\s+", "", matched.group(0)) if matched else ""


def period_label_number(value: object) -> str:
    """기수 표기에서 **회계연도를 가리키는 번호**만 — 「제 55 기 1분기」 → 「제55기」.

    뒤의 수식(말·1분기·반기)은 그 회계연도 **안의** 구간이라 연도를 바꾸지
    않는다. 짝을 지을 때 이것을 키로 써야 「제54기1분기」밖에 없는 문서에서도
    서술 표의 「제54기」를 찾을 수 있다.

    부분 일치도 받는다 — 정본 라벨에는 「제 15 기 1분기 > 누적」처럼 꼬리가
    더 붙은 것이 있다.
    """

    matched = re.search(r"제\s*([0-9]{1,3})\s*기", str(value or ""))
    return f"제{matched.group(1)}기" if matched else ""


class NotReadyError(NotImplementedError):
    """산출물이 아직 없어 답할 수 없다. **「결과 없음」과 구분하기 위한 예외.**"""


class AmbiguousSectionError(LookupError):
    """여러 원문 파일의 Section 이 같은 ``doc_id + locator`` 에 걸렸다.

    locator 는 문서가 아니라 **원문 파일 안에서만** 유일하다. 호출자가
    ``section_id`` 또는 ``source_file_id`` 를 주지 않은 상태에서 후보가 둘 이상이면,
    첫 행을 임의로 반환하지 않고 이 예외로 닫는다.
    """


class AmbiguousEventError(LookupError):
    """접수번호 하나가 여러 Event에 들어간 손상 상태를 임의 선택하지 않는다."""


class EvidenceIntegrityError(RuntimeError):
    """Evidence ID/hash/locator/FK가 서로 맞지 않아 인용할 수 없다."""


class ReadSecurityError(RuntimeError):
    """안전 projection이 없거나 손상되어 raw로 대체할 수 없다."""


class SourceIntegrityError(RuntimeError):
    """SourceFile 메타데이터·실제 경로·바이트가 서로 맞지 않아 원문을 열 수 없다."""


#: `lookup()` 이 값을 못 고른 이유. `not_found` 와 나머지는 성격이 다르다 —
#: 전자는 데이터가 없는 것이고, 후자는 **질문이 덜 구체적인** 것이다.
LookupStatus = Literal[
    "ok", "not_found", "extract_unsupported", "ambiguous_account_path",
    "ambiguous_statement", "ambiguous_period_role", "ambiguous_unit",
    "same_document_conflict",
]


class _Unset:
    """「지정 안 함」과 「`None` 으로 지정」을 가른다.

    `cumulative=None` · `period_start=None` 은 **실재하는 값**이다. 기본값을 `None`
    으로 두면 그 값을 고를 방법이 없다 (6차 검수). 기본값은 `UNSET` 이고,
    호출부가 명시적으로 `None` 을 주면 「`None` 인 것만」으로 좁힌다.
    """

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET = _Unset()


@dataclass(frozen=True)
class FactLookup:
    """단일 조회의 결과와 **못 고른 이유**."""

    status: LookupStatus
    selected: "FactRow | None"
    candidates: tuple["FactRow", ...] = ()
    #: **다음에 지정해야 할 축.** 「이걸 주면 결정된다」가 아니다 — 주고 나서도
    #: 다른 축이 남을 수 있다 (6차 검수). 실제로 `account_path` 를 준 뒤에도
    #: `ambiguous_period_role` 이 남았다. `ok` 가 될 때까지 반복해서 좁힌다.
    next_discriminator: tuple[str, ...] = ()
    #: 이 질의 기간·범위를 담지만 **우리가 못 읽은** 문서들. `status` 가 `ok` 여도
    #: 비어 있지 않을 수 있다 — 다른 문서가 값을 공급했다는 뜻이다.
    unread_documents: tuple[str, ...] = ()

    #: 이 결과가 어느 관점(`as_filed`/`restated`)으로 계산됐는가.
    #: 근거 범위는 관점마다 다르다 — 아래 `coverage_status` 참조.
    view: str = "restated"

    @property
    def coverage_status(self) -> str:
        """**선택 결과와 근거 범위는 다른 축이다** (7차 검수).

        `status` 는 「골랐는가」, 이쪽은 「근거를 다 봤는가」를 답한다.
        `ok_with_unread` 같은 status 를 새로 만들면 두 축이 뒤엉킨다.

        답변 층은 **`status == "ok"` 이고 `coverage_status == "complete"`** 일 때만
        무조건 확정 답변을 낸다.
        """
        # 값이 있어도 first-class Evidence가 없거나 무결성 검증을 통과하지 못하면
        # 「숫자 후보」일 뿐 확정 인용 가능한 답은 아니다. 오래된 1.2 산출물도 값 조회는
        # 유지하지만 이 상태로 명확히 구분한다.
        if self.selected is not None and self.selected.evidence_status != "verified":
            return "evidence_unavailable"
        return "partial_unread" if self.unread_documents else "complete"

    @property
    def required_discriminator(self) -> tuple[str, ...]:
        """옛 이름. `next_discriminator` 를 쓴다."""
        return self.next_discriminator

    def __bool__(self) -> bool:
        """**근거를 다 본 확정 답일 때만 참이다.**

        `status` 만 보면 `if lookup:` 이 `partial_unread` 도 통과시킨다 —
        추가한 축이 곧바로 우회로가 된다 (8차 검수). 두 축을 함께 요구한다.
        """
        return self.status == "ok" and self.coverage_status == "complete"


@dataclass(frozen=True)
class Money:
    """값과 단위를 함께 들고 다닌다. 둘을 떼면 1,000배 틀린다.

    **권위 있는 값은 `text`(원문 문자열)다.** `value` 는 정렬·필터용 float 캐시다.
    코퍼스 최대 절대값 1.44e14 로 float64 정수 한계(9.0e15) 안이라 정수는 정확하지만,
    소수는 이진 근삿값이다 — `round(2.675, 2)` 이 `2.67` 이 되고
    `f"{2.675:,.0f}"` 은 `3` 이 된다. 금액 계산·반올림은 `decimal()` 에서 시작한다.
    """

    value: float | None
    unit: str | None
    text: str | None = None                  #: 원문 문자열 — 계산의 출발점

    def decimal(self) -> Decimal | None:
        """원문 문자열에서 만든 `Decimal`. float 를 거치지 않는다."""
        if self.text:
            try:
                return Decimal(self.text.replace(",", "").strip())
            except InvalidOperation:
                pass
        return None if self.value is None else Decimal(str(self.value))

    def in_won(self) -> float | None:
        """원 단위로 환산. 모르는 단위면 `None` — 추측해서 곱하지 않는다."""
        if self.value is None or self.unit is None:
            return None
        scale = _UNIT_SCALE.get(self.unit.strip())
        return None if scale is None else self.value * scale

    def in_won_decimal(self) -> Decimal | None:
        """`in_won()` 의 정확한 판. 답변에 쓰는 금액은 이쪽이다."""
        d = self.decimal()
        if d is None or self.unit is None:
            return None
        scale = _UNIT_SCALE.get(self.unit.strip())
        return None if scale is None else d * Decimal(scale)

    def __str__(self) -> str:
        """**소수부를 버리지 않는다.** `Money(2.675,"원")` 은 `3 원` 이 아니다."""
        if self.value is None:
            return "-"
        d = self.decimal()
        if d is None:
            return f"{self.value:,} {self.unit or ''}".strip()
        q = d.normalize()
        body = f"{q:,f}" if q == q.to_integral_value() else f"{q:,}"
        return f"{body} {self.unit or ''}".strip()


@dataclass(frozen=True)
class FactRow:
    """재무제표 한 칸 + 근거."""

    doc_id: str
    rcept_dt: str
    scope: str                   #: CFS | SFS
    statement: str               #: BS | IS | CI | CF
    account_raw: str
    account_path: str
    account_norm: str | None
    period_start: str | None
    period_end: str | None
    period_type: str
    cumulative: bool | None
    money: Money
    value_status: str
    locator: str

    # schema 1.5 이전 산출물에는 아래 좌표/FK 일부가 없다. 조회 값은 유지하되
    # ``evidence_status``를 unavailable로 두어 확정 인용으로 승격하지 않는다.
    source_file_id: str | None = None
    block_id: str | None = None
    parent_id: str | None = None
    path: str | None = None
    table_locator: str | None = None
    logical_row: int | None = None
    logical_col: int | None = None
    evidence_id: str | None = None
    evidence_status: str = "unavailable"   #: verified | unverified | unavailable | invalid
    acode: str | None = None               #: 값 셀의 XBRL 택소노미 요소 ID (있을 때만)
    account_norm_source: str | None = None #: account_norm 출처 — acode | label | None
    #: 사람이 읽는 표 이름 — 「2-1. 연결 재무상태표」. 이슈 #94 30 의 출처
    #: 안내가 쓴다. `locator`(`TABLE[1]/TBODY[0]/TR[25]`)는 기계 좌표라
    #: 사용자가 원문에서 찾아갈 수 없다.
    statement_title: str | None = None
    #: 공시 원문이 그 열에 붙인 기간 이름 — 「제 50 기」. 같은 행의
    #: `period_start`/`period_end` 가 그것이 실제로 몇 년인지 말해 준다.
    period_label: str | None = None

    @property
    def citation(self) -> str | None:
        """검증된 first-class Evidence만 확정 인용으로 돌려준다."""
        return self.evidence_id if self.evidence_status == "verified" else None

    @property
    def source_coordinate(self) -> str:
        """확정 인용이 아닌 원문 탐색 좌표. 첨부 간 locator 충돌을 피한다."""
        return f"{self.doc_id}#{self.source_file_id or '?'}#{self.locator}"


@dataclass(frozen=True)
class Company:
    corp_code: str
    corp_name: str


@dataclass(frozen=True)
class ChunkRow:
    """검색 인덱스에 넣을 조각 하나.

    `rcept_dt` 가 함께 온다 — **시점 필터는 인덱스 쪽에서 걸어야 한다.**
    순위를 매긴 뒤에 거르면 상위 k 가 전부 미래 문서일 때 결과가 0 이 된다.
    """

    chunk_id: str                #: block_id
    section_id: str              #: 부모 Section 의 block_id
    doc_id: str
    source_file_id: str
    rcept_dt: str
    corp_code: str
    corp_name: str
    doc_group: str
    path: str
    locator: str
    text: str                    #: 선택된 safe projection. raw가 아니다
    text_prompt_safe: str        #: LLM 전달용 projection을 명시적으로 제공
    text_projection: str         #: search | llm
    security_flags: tuple[str, ...]
    pii_types: tuple[str, ...]
    security_policy_version: str
    n_chars: int
    index_eligible: bool
    over_budget: bool
    evidence_id: str | None
    evidence_status: str         #: verified | unverified | unavailable | invalid

    @property
    def citation(self) -> str | None:
        return self.evidence_id if self.evidence_status == "verified" else None


@dataclass(frozen=True)
class SectionText:
    section_id: str              #: block_id
    doc_id: str
    source_file_id: str
    rcept_dt: str
    path: str
    locator: str
    text: str                    #: 기본 prompt_safe, 명시 옵션일 때만 restricted_raw
    text_projection: str
    security_flags: tuple[str, ...]
    pii_types: tuple[str, ...]
    security_policy_version: str
    n_chars: int
    n_tables: int


@dataclass(frozen=True)
class SourceFileRow:
    """canonical SourceFile과 문서 단위 PDF/HTML 검증 상태.

    ``normalized_path_key``는 비교·ID용일 뿐 파일 경로가 아니다. 실제 파일 접근은
    :meth:`CanonicalReadModel.resolve_source_path`가 inventory binding과 SHA를 다시
    확인한 뒤에만 허용한다.
    """

    source_file_id: str
    doc_id: str
    normalized_path_key: str
    actual_relpath: str | None
    role: str
    sha256: str
    byte_size: int
    parse_mode: str
    source_selection_role: str | None = None
    parse_usable: bool | None = None
    coverage_n_sections: int | None = None
    coverage_n_chars: int | None = None
    coverage_n_tables: int | None = None
    coverage_n_pages: int | None = None
    coverage_pages_with_text: int | None = None
    page_text_coverage: float | None = None
    locator_kind: str | None = None
    locator_limitations: tuple[str, ...] = ()
    parse_warnings: tuple[str, ...] = ()
    parse_error: str | None = None
    primary_source_file_id: str | None = None
    alternate_source_file_ids: tuple[str, ...] = ()
    source_cross_check_status: str | None = None
    source_cross_check_reason: str | None = None
    source_cross_check_source_file_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventState:
    """어떤 시점의 사건 상태. `status_at_corpus_end` 를 그냥 읽으면 오답이다."""

    event_key: str
    as_of: str
    status: str                  #: active | terminated | not_disclosed
    n_observations: int
    last_rcept_no: str | None
    last_observed_at: str | None


FieldLookupStatus = Literal["ok", "not_found", "ambiguous"]


@dataclass(frozen=True)
class FieldRow:
    """비정기공시의 라벨–값 한 쌍과 문서·원문 근거.

    ``fields.parquet``에는 회사와 서식이 없으므로 이 객체는 내부에서 Document를 조인해
    돌려준다. 호출자가 Parquet 조인을 다시 구현할 필요가 없다.
    """

    doc_id: str
    rcept_no: str
    rcept_dt: str
    corp_code: str
    corp_name: str
    doc_group: str
    event_type: str | None
    form: str
    report_nm: str
    is_correction: bool
    source_file_id: str
    block_id: str
    parent_id: str | None
    path: str
    order: int
    locator: str
    #: PII·active content·prompt-like 지시를 제거한 **scalar-safe 본문**.
    #: LLM에 전달할 때는 경계까지 포함한 :attr:`value_prompt_safe`를 쓴다.
    value_masked: str | None
    #: Evidence 생성과 동일한 회사명 context로 만든 정확한 LLM projection.
    #: 조회 후 scalar를 다시 projection하면 회사명 예외 처리가 달라질 수 있다.
    value_prompt_safe: str | None
    #: 호환 필드. 기본 조회에서는 scalar-safe 값이며, 명시 플래그에서만 실제 raw다.
    value_raw: str | None
    restricted_raw_included: bool
    pii_type: str | None
    security_policy_version: str
    evidence_id: str | None
    evidence_status: str         #: verified | unverified | unavailable | invalid
    value_status: str
    acode: str | None
    aunit: str | None
    aunitvalue: str | None
    is_pii: bool
    table_locator: str | None
    logical_row: int | None
    logical_col: int | None
    label_locators: str | None
    occurrence: int

    @property
    def label(self) -> str:
        return _field_label(self.path)

    @property
    def value(self) -> str | None:
        """Agent의 기본 scalar 값. raw·active content·prompt 지시가 아니다."""
        return self.value_masked

    @property
    def citation(self) -> str | None:
        """검증된 Evidence만 확정 citation으로 노출한다."""
        return self.evidence_id if self.evidence_status == "verified" else None

    @property
    def source_coordinate(self) -> str:
        # locator는 원문 파일 안에서만 유일하다. source_file_id를 빼면 첨부와 충돌한다.
        return f"{self.doc_id}#{self.source_file_id}#{self.locator}"


@dataclass(frozen=True)
class TableHeaderRow:
    """One source-backed table-header coordinate for a safe ontology check.

    This is deliberately metadata only: callers receive the canonical header
    path and its immutable coordinate, never a table body/value.  It lets a
    resolver prove that requested semantic slots are actually labelled by the
    source table before it commits to a structured extraction route.
    """

    doc_id: str
    rcept_no: str
    corp_code: str
    source_file_id: str
    table_locator: str
    locator: str
    header_path: str


@dataclass(frozen=True)
class FieldLookup:
    """Field 단일 조회. 후보가 둘 이상이면 ``selected``는 반드시 ``None``이다."""

    status: FieldLookupStatus
    selected: FieldRow | None
    candidates: tuple[FieldRow, ...] = ()
    next_discriminator: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        # ``status``는 후보를 하나로 골랐는지, truthiness는 근거까지 확인했는지다.
        # 둘을 합치지 않아 typed 상태는 보존하되 ``if lookup:``이 Evidence
        # invalid/unavailable 값을 확정 답으로 승격하지 못하게 한다.
        return (self.status == "ok" and self.selected is not None
                and self.selected.evidence_status == "verified")


@dataclass(frozen=True)
class CorrectionItemRow:
    """정정표 한 항목. 선언된 전·후 값이며 자동 계산 diff와 혼동하지 않는다."""

    doc_id: str
    rcept_no: str
    rcept_dt: str
    corp_code: str
    corp_name: str
    doc_group: str
    event_type: str | None
    form: str
    report_nm: str
    source_file_id: str
    block_id: str
    parent_id: str | None
    path: str
    order: int
    locator: str
    #: 기본 scalar-safe 정정사유. raw는 :attr:`reason_raw`에서만 명시 제공한다.
    reason: str | None
    reason_prompt_safe: str | None
    reason_raw: str | None
    value_before: str | None     #: prompt-safe projection
    value_after: str | None      #: prompt-safe projection
    value_before_prompt_safe: str | None
    value_after_prompt_safe: str | None
    value_before_raw: str | None
    value_after_raw: str | None
    restricted_raw_included: bool
    before_evidence_id: str | None
    after_evidence_id: str | None
    evidence_status: str         #: verified | unverified | unavailable | invalid
    before_evidence_status: str
    after_evidence_status: str
    table_locator: str | None
    logical_row: int | None
    logical_col: int | None
    #: schema 1.8의 전·후 실제 값 셀 좌표. 대표 행 ``locator``를 Evidence에
    #: 재사용하지 않는다. 구버전 artifact는 대표 좌표로 호환 조회한다.
    before_locator: str | None
    after_locator: str | None
    before_table_locator: str | None
    after_table_locator: str | None
    before_logical_row: int | None
    before_logical_col: int | None
    after_logical_row: int | None
    after_logical_col: int | None
    diff_kind: str
    before_kind: str
    after_kind: str
    required_by_authority: bool

    @property
    def citation(self) -> tuple[str, ...] | None:
        """양쪽 Evidence가 모두 검증된 경우의 citation들."""
        return self.citations if self.evidence_status == "verified" else None

    @property
    def citations(self) -> tuple[str, ...]:
        return tuple(item for item in
                     (self.before_citation, self.after_citation) if item is not None)

    @property
    def before_citation(self) -> str | None:
        return (self.before_evidence_id
                if self.before_evidence_status == "verified" else None)

    @property
    def after_citation(self) -> str | None:
        return (self.after_evidence_id
                if self.after_evidence_status == "verified" else None)

    @property
    def source_coordinate(self) -> str:
        return f"{self.doc_id}#{self.source_file_id}#{self.locator}"


@dataclass(frozen=True)
class EvidenceRow:
    """검증된 first-class Evidence.

    ``excerpt``와 ``excerpt_safe``는 항상 동일한 safe projection이다. restricted raw는
    ``get_evidence(..., include_restricted_raw=True)``를 명시한 경우에만 채워진다.
    """

    evidence_id: str
    doc_id: str
    source_file_id: str
    kind: str
    locator: str
    table_locator: str | None
    logical_row: int | None
    logical_col: int | None
    excerpt: str
    excerpt_safe: str
    excerpt_raw: str | None
    restricted_raw_included: bool
    excerpt_hash: str
    raw_access: str
    extraction_method: str
    extraction_status: str
    security_flags: tuple[str, ...]
    pii_types: tuple[str, ...]
    security_policy_version: str
    evidence_policy_version: str
    rcept_dt: str


@dataclass(frozen=True)
class EventObservationRow:
    """Event의 한 시점 관측. timeline에서는 ``as_of`` 이후 행이 제거된다."""

    doc_id: str
    event_key: str
    seq: int
    rcept_no: str
    observed_at: str
    is_correction: bool
    is_termination: bool
    previous_observation_rcept_no: str | None
    supporting_evidence_ids: tuple[str, ...]
    support_roles: tuple[str, ...]
    support_status: str
    support_version: str | None
    support_limitation: str | None
    evidence_verification_status: str

    @property
    def citations(self) -> tuple[str, ...]:
        return (self.supporting_evidence_ids
                if (self.support_status == "fully_verified"
                    and self.evidence_verification_status == "verified") else ())


@dataclass(frozen=True)
class EventTimeline:
    """기준시점까지 공개된 사건 관측과 그 시점의 파생 상태."""

    event_key: str
    kind: str
    corp_code: str
    corp_name: str
    doc_group: str
    form: str
    root_rcept_no: str
    as_of: str
    state: EventState
    observations: tuple[EventObservationRow, ...]
    support_status: str
    support_limitation: str | None
    citations: tuple[str, ...]
    observation_support_status: str
    observation_support_limitation: str | None
    identity_verification_status: str
    identity_limitation: str | None
    query_coverage_status: str
    query_coverage_limitation: str | None
    identity_fingerprint: str | None = None
    identity_status: str | None = None
    resolver_version: str | None = None


@dataclass(frozen=True)
class RelationSupportRow:
    """Relation provenance. target anchor가 없으면 citation을 확정 노출하지 않는다."""

    relation_id: str
    src_id: str
    src_rcept_no: str
    dst_id: str | None
    dst_rcept_no: str | None
    relation_type: str
    resolution_status: str
    supporting_evidence_ids: tuple[str, ...]
    support_roles: tuple[str, ...]
    support_status: str
    support_version: str | None
    support_limitation: str | None
    citations: tuple[str, ...]


@dataclass(frozen=True)
class RelationSummaryRow:
    """Safe relation coordinate used for typed lineage-scope decisions."""

    relation_id: str
    src_rcept_no: str
    dst_rcept_no: str | None
    relation_type: str
    resolution_status: str
    target_hint: str | None
    root_missing_reason: str | None


@dataclass(frozen=True)
class _DocumentInfo:
    doc_id: str
    rcept_no: str
    corp_code: str
    corp_name: str
    listed_name: str
    filer: str
    doc_group: str
    doc_subtype: str | None
    event_type: str | None
    report_nm: str
    rcept_dt: str
    base_year: int | None
    base_month: int | None
    is_correction: bool
    primary_source_file_id: str | None = None
    alternate_source_file_ids: tuple[str, ...] = ()
    source_cross_check_status: str | None = None
    source_cross_check_reason: str | None = None
    source_cross_check_source_file_ids: tuple[str, ...] = ()

    @property
    def form(self) -> str:
        return (self.doc_subtype or self.event_type or _report_form(self.report_nm))

    def matches_form(self, form: str) -> bool:
        want = _text_key(form)
        candidates = (self.form, self.doc_subtype, self.event_type,
                      _report_form(self.report_nm), self.report_nm)
        return bool(want) and want in {_text_key(v) for v in candidates if v}


@dataclass(frozen=True)
class DocumentMetadataRow:
    """Safe document selector metadata without body text or answer values."""

    doc_id: str
    rcept_no: str
    corp_code: str
    corp_name: str
    filer: str
    doc_group: str
    form: str
    report_nm: str
    rcept_dt: str
    base_year: int | None
    base_month: int | None
    is_correction: bool


def _rows(
        path: Path, columns: list[str], batch: int = 200_000,
        *, filters: object | None = None,
        ) -> Iterator[dict]:
    """Parquet 스트리밍. **`read_table` 로 통째 올리지 않는다** — 4GB 서버 전제."""
    if filters is not None:
        # Arrow applies row-group/page pruning before materialization.  Callers
        # use this branch only for a small, metadata-resolved document set, so
        # the filtered table stays bounded while avoiding a 1.6M-row Python
        # conversion for every company-specific field lookup.
        table = pq.read_table(path, columns=columns, filters=filters)
        for b in table.to_batches(max_chunksize=batch):
            d = b.to_pydict()
            for values in zip(*d.values()):
                yield dict(zip(d, values))
        return
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=batch, columns=columns):
        d = b.to_pydict()
        for values in zip(*d.values()):
            yield dict(zip(d, values))


def _rows_for_exact_string_by_row_group(
        path: Path, columns: list[str], *, selector: str, value: str,
        batch: int = 2_000,
        ) -> Iterator[dict]:
    """Stream rows for one exact string without a whole-file filtered table.

    ``pq.read_table(filters=...)`` may still scan every row group when its
    statistics cannot prune a hash-like selector.  If the selected columns
    contain several large text projections, Arrow can then hold GiBs of
    decompressed buffers for what is logically a one-document lookup.  Read
    the small selector column first and materialize heavy columns only in the
    row groups that actually contain the requested value.
    """

    parquet = pq.ParquetFile(path)
    for group_index in range(parquet.metadata.num_row_groups):
        selector_table = parquet.read_row_group(
            group_index, columns=[selector])
        selector_mask = pc.equal(
            selector_table[selector], pa.scalar(value))
        if not pc.any(selector_mask).as_py():
            continue
        table = parquet.read_row_group(group_index, columns=columns)
        mask = pc.equal(table[selector], pa.scalar(value))
        filtered = table.filter(mask)
        for record_batch in filtered.to_batches(max_chunksize=batch):
            data = record_batch.to_pydict()
            for values in zip(*data.values()):
                yield dict(zip(data, values))


_EVIDENCE_FINGERPRINT_FIELDS = (
    "build_id", "evidence_id", "doc_id", "source_file_id", "kind",
    "locator", "table_locator", "logical_row", "logical_col", "excerpt_raw",
    "excerpt_safe", "excerpt_hash", "raw_access", "extraction_method",
    "extraction_status", "security_flags", "pii_types",
    "security_policy_version", "evidence_policy_version", "rcept_dt",
)


def _evidence_fingerprint(row: Mapping[str, object]) -> bytes:
    """Evidence 비교용 compact digest.

    Chunk 18만 건의 raw/safe 문자열을 Python 객체로 계속 보유하지 않고도 저장 Evidence와
    raw에서 재생성한 계약을 exact 비교한다. 필드명·타입·순서를 함께 hash하므로 list 순서,
    null, 숫자/문자열 차이도 숨기지 않는다.
    """
    digest = hashlib.sha256()
    for name in _EVIDENCE_FINGERPRINT_FIELDS:
        payload = json.dumps(
            row.get(name), ensure_ascii=False, separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(name).to_bytes(2, "big"))
        digest.update(name.encode("ascii"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.digest()


#: process-local artifact metadata acceleration only.
#:
#: A CanonicalReadModel validates one immutable published build at startup.  A
#: newly published build must use a new process; these caches are deliberately
#: not a live-reload mechanism and cannot make a model safe across replacement.
#: The file fingerprint merely avoids reusing an entry when a test/CLI replaces
#: a file before its next lookup.
_ARTIFACT_CACHE_LOCK = threading.RLock()
_ARTIFACT_DATASET_CACHE_LIMIT = 64
_ARTIFACT_COLUMN_CACHE_LIMIT = 256


#: (absolute path, device, inode, ctime_ns, mtime_ns, size) → column names.
#
# 컬럼 집합은 한 build 안에서 불변인데, 조회마다 `ParquetFile(path)` 로 다시 열면
# footer 를 매번 읽는다.  784MB `evidence.parquet` 은 여는 데만 약 10ms 다.
# 정정 이력 한 건을 조립하는 데 파일 열기가 370회 일어나 2.6초가 데이터도 읽기 전에
# 사라졌다(EDGE-032 프로파일).
# The fingerprint is cache invalidation, not a publication protocol.
_ARTIFACT_COLUMNS: dict[tuple[str, int, int, int, int, int], frozenset[str]] = {}


#: (absolute path, device, inode, ctime_ns, mtime_ns, size) → pyarrow dataset.
#
# `pq.read_table(path, filters=…)` 은 호출마다 `ParquetDataset` 을 새로 만든다.
# 정정 이력 한 건에서 453회가 일어나 2.1초가 dataset 구성에만 쓰였다.  dataset 은
# 파일이 그대로면 재사용해도 되고, 재사용하면 같은 조회가 20.6ms → 4.8ms 가 된다.
#
# This cache is protected for incidental multi-threaded tooling.  The served
# CanonicalReadModel remains single-executor by design.
_ARTIFACT_DATASETS: dict[tuple[str, int, int, int, int, int], Any] = {}


def _artifact_key(path: Path) -> "tuple[str, int, int, int, int, int] | None":
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_ctime_ns,
            stat.st_mtime_ns, stat.st_size)


def _artifact_dataset(path: Path) -> Any:
    """Parquet dataset. 같은 파일이면 메타데이터를 다시 만들지 않는다."""

    key = _artifact_key(path)
    if key is None:
        return _ds.dataset(path, format="parquet")
    with _ARTIFACT_CACHE_LOCK:
        cached = _ARTIFACT_DATASETS.get(key)
        if cached is None:
            cached = _ds.dataset(path, format="parquet")
            if len(_ARTIFACT_DATASETS) >= _ARTIFACT_DATASET_CACHE_LIMIT:
                _ARTIFACT_DATASETS.clear()
            _ARTIFACT_DATASETS[key] = cached
        return cached


def _read_artifact_table(
        path: "str | Path", columns: "list[str] | None" = None,
        filters: "list | None" = None) -> Any:
    """`pq.read_table` 의 drop-in. dataset 을 캐시해 메타데이터를 다시 만들지 않는다.

    반환형·결과는 `pq.read_table(path, columns=…, filters=…)` 과 같다.
    """

    dataset = _artifact_dataset(Path(path))
    expression = None if filters is None else pq.filters_to_expression(filters)
    return dataset.to_table(columns=columns, filter=expression)


def _read_artifact_rows(
        path: Path, columns: list[str],
        filters: "list | None" = None) -> list[dict]:
    return _read_artifact_table(path, columns, filters).to_pylist()


def _artifact_columns(path: Path) -> frozenset[str]:
    """Parquet 컬럼 이름. 같은 파일이면 footer 를 다시 읽지 않는다."""

    try:
        stat = path.stat()
    except OSError:
        return frozenset(pq.ParquetFile(path).schema_arrow.names)
    key = (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_ctime_ns,
           stat.st_mtime_ns, stat.st_size)
    with _ARTIFACT_CACHE_LOCK:
        cached = _ARTIFACT_COLUMNS.get(key)
        if cached is None:
            cached = frozenset(pq.ParquetFile(path).schema_arrow.names)
            if len(_ARTIFACT_COLUMNS) >= _ARTIFACT_COLUMN_CACHE_LIMIT:
                _ARTIFACT_COLUMNS.clear()
            _ARTIFACT_COLUMNS[key] = cached
        return cached


def _present_columns(path: Path, wanted: list[str]) -> list[str]:
    """이전 최소 fixture도 읽기 위한 존재 컬럼 교집합."""
    present = _artifact_columns(path)
    return [name for name in wanted if name in present]


def _schema_at_least(value: str, wanted: tuple[int, ...]) -> bool:
    """발행 schema 문자열의 숫자 tuple 비교. malformed 값은 새 계약으로 보지 않는다."""
    try:
        actual = tuple(int(part) for part in value.split("."))
    except (AttributeError, ValueError):
        return False
    return actual >= wanted


def _checked_as_of(as_of: str) -> str:
    """문자열 대소 비교가 안전한 canonical 접수일만 허용한다."""
    # ``\d``는 전각 숫자 같은 Unicode decimal도 받는다. 그대로 반환해 ASCII
    # ``rcept_dt``와 사전식 비교하면 cutoff가 뒤틀려 미래 공시가 노출될 수 있다.
    if not isinstance(as_of, str) or re.fullmatch(r"[0-9]{8}", as_of) is None:
        raise ValueError("as_of는 YYYYMMDD 8자리 문자열이어야 합니다")
    try:
        date(int(as_of[:4]), int(as_of[4:6]), int(as_of[6:8]))
    except ValueError as exc:
        raise ValueError("as_of는 실제로 존재하는 YYYYMMDD 날짜여야 합니다") from exc
    return as_of


#: 한 번에 밀어 넣을 수 있는 접수번호 개수. Parquet ``in`` 술어의 doc_id 상한
#: 2048과 같은 자리에 둔다 — 그보다 많으면 pushdown 이 풀려 전량 스캔이 된다.
_MAX_RCEPT_NOS = 2048


def _checked_rcept_nos(value) -> frozenset[str] | None:
    """접수번호 묶음을 검증한다 — 하나씩 읽는 것과 같은 결과여야 한다."""

    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError("rcept_nos는 문자열이 아니라 접수번호 모음이어야 합니다")
    try:
        wanted = frozenset(value)
    except TypeError as exc:
        raise TypeError("rcept_nos는 반복 가능한 모음이어야 합니다") from exc
    if not wanted:
        raise ValueError("rcept_nos가 비어 있습니다")
    if len(wanted) > _MAX_RCEPT_NOS:
        raise ValueError(
            f"rcept_nos는 {_MAX_RCEPT_NOS}개 이하여야 합니다 (요청 {len(wanted)}개)")
    for item in wanted:
        if not isinstance(item, str) or re.fullmatch(r"[0-9]{14}", item) is None:
            raise ValueError("rcept_nos 원소는 접수번호 14자리여야 합니다")
    return wanted


#: 한 번에 밀어 넣을 수 있는 접수번호 개수. Parquet ``in`` 술어의 doc_id 상한
#: 2048과 같은 자리에 둔다 — 그보다 많으면 pushdown 이 풀려 전량 스캔이 된다.
_MAX_RCEPT_NOS = 2048


def _checked_rcept_nos(value) -> frozenset[str] | None:
    """접수번호 묶음을 검증한다 — 하나씩 읽는 것과 같은 결과여야 한다."""

    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError("rcept_nos는 문자열이 아니라 접수번호 모음이어야 합니다")
    try:
        wanted = frozenset(value)
    except TypeError as exc:
        raise TypeError("rcept_nos는 반복 가능한 모음이어야 합니다") from exc
    if not wanted:
        raise ValueError("rcept_nos가 비어 있습니다")
    if len(wanted) > _MAX_RCEPT_NOS:
        raise ValueError(
            f"rcept_nos는 {_MAX_RCEPT_NOS}개 이하여야 합니다 (요청 {len(wanted)}개)")
    for item in wanted:
        if not isinstance(item, str) or re.fullmatch(r"[0-9]{14}", item) is None:
            raise ValueError("rcept_nos 원소는 접수번호 14자리여야 합니다")
    return wanted


def _checked_bool_flag(name: str, value: bool) -> bool:
    """``"false"`` 같은 truthy 문자열이 restricted 경계를 여는 것을 막는다."""
    if type(value) is not bool:
        raise TypeError(f"{name}는 bool이어야 합니다")
    return value


class CanonicalReadModel:
    """`out/canonical/` 위의 조회 계층.

    무거운 컬럼(`text`)은 필요할 때만 읽고, 조회 축은 bounded cache로 만든다.
    Fact는 회사 문서 ID로 predicate 조회해 전량 Python 객체화를 피한다.
    """

    def __init__(
            self, root: str | Path = "out/canonical", *,
            fact_cache_max_rows: int = 50_000,
            fact_cache_max_corps: int = 8,
            ) -> None:
        if type(fact_cache_max_rows) is not int or fact_cache_max_rows < 1:
            raise ValueError("fact_cache_max_rows는 양의 정수여야 합니다")
        if type(fact_cache_max_corps) is not int or fact_cache_max_corps < 1:
            raise ValueError("fact_cache_max_corps는 양의 정수여야 합니다")
        self.root = Path(root)
        # 조회를 시작한 뒤 파일마다 제각각 실패하게 두지 않는다. published·hash·행 수와
        # 모든 Parquet build_id를 한 번에 검증해 변조/부분 교체를 기동 단계에서 닫는다.
        self.run = validate_canonical_artifacts(self.root)
        #: `직 업(사업내용)` 재검증에 쓰는 문서별 주체 지도 (이슈 #139).
        self._subject_cache: OrderedDict[
            str, dict[tuple[object, object], str]] = OrderedDict()
        #: `성명(명칭)` 재검증에 쓰는 문서별 `구분` 지도 (이슈 #199).
        self._party_type_cache: OrderedDict[str, object] = OrderedDict()
        self.build_id: str = self.run["build_id"]
        self.schema_version: str = self.run["schema_version"]
        # facts.parquet 전량을 FactRow Python 객체로 만들면 65만 행의 문자열 객체가
        # 서버 수명 동안 상주한다. 회사별 predicate 조회 결과만 weighted LRU에 둔다.
        # 회사 수와 행 수를 함께 제한해 작은 회사 다수와 큰 회사 소수 양쪽을 막는다.
        self._facts_by_corp: OrderedDict[str, tuple[FactRow, ...]] = OrderedDict()
        self._fact_cache_rows = 0
        self._fact_cache_max_rows = fact_cache_max_rows
        self._fact_cache_max_corps = fact_cache_max_corps
        self._doc_corp: dict[str, str] = {}
        self._doc_meta: dict[str, tuple[str, str, str]] = {}
        self._documents: dict[str, _DocumentInfo] = {}
        #: doc_id → {「제50기」: "2023", …}. 이슈 #94 26 — 공시 원문의 회계
        #: 기수를 실제 연도로 바꿀 때 쓴다. 문서 단위로만 뜻이 있다.
        self._period_years_by_doc: dict[str, dict[str, str]] = {}
        self._duplicate_document_ids: set[str] = set()
        self._doc_id_by_rcept: dict[str, str] = {}
        self._docs_loaded = False
        self._companies: dict[str, str] = {}
        #: corp_code → (corp_code, corp_name, listed_name, stock_code)
        self._company_surfaces: dict[str, tuple] = {}
        self._company_alias_registry: dict | None = None
        #: 자동 확정하지 않는 표기 → 후보. 확인 역질문에 쓴다.
        self._company_held_aliases: dict = {}
        self._obs: dict[str, list[dict]] = {}
        self._event_keys_by_rcept: dict[str, tuple[str, ...]] = {}
        self._event_identities: dict[str, dict] | None = None
        # Canonical artifacts are immutable for one build-bound reader.  Cache
        # only fully typed timeline results, never question/answer decisions.
        self._event_timeline_cache: OrderedDict[
            tuple[str, str, bool], EventTimeline | None] = OrderedDict()
        self._event_timeline_cache_limit = 256
        #: 재무 Fact 를 못 뽑은 문서. 「없음」과 구분해 답하려면 **어느 기간·어느 범위**가
        #: 미지원인지까지 있어야 한다 — 법인만 들면 그 회사 전체가 미지원이 된다.
        self._unsupported_docs: list[dict] = []
        self._relations: list[dict] | None = None
        self._rcept_dt: dict[str, str] = {}
        # Point lookup locality만 보존한다. Chunk/Field 전량 호출에서 safe excerpt 수십만 개를
        # 무제한 보유하면 스트리밍 API가 사실상 전체 materialization이 된다.
        self._evidence_cache: OrderedDict[str, EvidenceRow] = OrderedDict()
        self._evidence_cache_limit = 4_096
        self._evidence_source_owner_cache: dict[str, str] = {}
        # Section block_id는 해시라 Parquet row-group min/max로 곧바로 좁힐 수 없다.
        # 그래도 위치를 한 번 찾은 ID는 다음 조회에서 본문 row group 하나만 읽도록
        # bounded cache에 둔다. 본문/SectionText 자체를 cache하지 않아 대형 원문이
        # reader 수명 동안 누적되지 않는다.
        self._section_row_group_cache: OrderedDict[str, int | None] = OrderedDict()
        self._section_row_group_cache_limit = 4_096
        self._chunk_evidence_fingerprints: dict[str, bytes] | None = None
        self._chunk_evidence_source_owners: dict[str, str] | None = None
        self._corpus_indexes: dict[Path, CorpusIndex] = {}
        #: 법인별 **회계연도 말 월**. 재무상태표의 비교시점이 여기에 달려 있다.
        #: 사업보고서 기준월에서 뽑는다 — 지금 코퍼스는 70개사 전부 12월이지만
        #: 3월 결산사가 들어오면 12월 고정은 직전 연차말을 틀리게 잡는다 (9차 검수).
        self._fy_end: dict[str, int] = {}

    # ---------------------------------------------------------------- 색인
    def _load_docs(self) -> None:
        if self._docs_loaded:
            return
        path = self.root / "documents.parquet"
        if not path.exists():
            raise NotReadyError(f"Document 산출물이 없습니다: {path}")
        # ``stock_code`` 는 승인 표기 registry 의 세 번째 출처다. 빼면 종목코드로
        # 회사를 못 찾는다.
        wanted = ["doc_id", "rcept_no", "corp_code", "corp_name", "listed_name",
                  "filer",
                  "stock_code", "doc_group",
                  "doc_subtype", "event_type", "report_nm", "rcept_dt", "base_year",
                  "base_month", "is_correction", "fact_extract_status",
                  "fact_unsupported_scope", "primary_source_file_id",
                  "alternate_source_file_ids", "source_cross_check_status",
                  "source_cross_check_reason", "source_cross_check_source_file_ids"]
        modern = _schema_at_least(self.schema_version, (1, 9))
        for r in _rows(path, _present_columns(path, wanted)):
            raw_is_correction = r.get("is_correction")
            if modern and type(raw_is_correction) is not bool:
                raise EvidenceIntegrityError(
                    "schema 1.9 Document.is_correction은 non-null bool이어야 합니다: "
                    f"doc_id={r.get('doc_id')!r} value={raw_is_correction!r}")
            is_correction = (raw_is_correction if type(raw_is_correction) is bool
                             else bool(raw_is_correction))
            self._rcept_dt[r["rcept_no"]] = r["rcept_dt"]
            if (r["doc_group"] == "periodic" and r.get("base_month")
                    and "사업보고서" in str(r["report_nm"] or "")):
                self._fy_end.setdefault(r["corp_code"], int(r["base_month"]))
            self._doc_corp[r["doc_id"]] = r["corp_code"]
            self._doc_meta[r["doc_id"]] = (r["corp_code"], r["corp_name"], r["doc_group"])
            if r["doc_id"] in self._documents:
                self._duplicate_document_ids.add(r["doc_id"])
            info = _DocumentInfo(
                doc_id=r["doc_id"], rcept_no=r["rcept_no"], corp_code=r["corp_code"],
                corp_name=(r["corp_name"] if isinstance(r.get("corp_name"), str)
                           else ""),
                listed_name=(r["listed_name"] if isinstance(r.get("listed_name"), str)
                             else ""),
                filer=(r["filer"] if isinstance(r.get("filer"), str) else ""),
                doc_group=r["doc_group"],
                doc_subtype=r.get("doc_subtype"), event_type=r.get("event_type"),
                report_nm=str(r.get("report_nm") or ""), rcept_dt=r["rcept_dt"],
                base_year=(int(r["base_year"])
                           if r.get("base_year") is not None else None),
                base_month=(int(r["base_month"])
                            if r.get("base_month") is not None else None),
                is_correction=is_correction,
                primary_source_file_id=r.get("primary_source_file_id"),
                alternate_source_file_ids=tuple(r.get("alternate_source_file_ids") or ()),
                source_cross_check_status=r.get("source_cross_check_status"),
                source_cross_check_reason=r.get("source_cross_check_reason"),
                source_cross_check_source_file_ids=tuple(
                    r.get("source_cross_check_source_file_ids") or ()))
            self._documents[r["doc_id"]] = info
            self._doc_id_by_rcept[r["rcept_no"]] = r["doc_id"]
            self._companies.setdefault(r["corp_code"], r["corp_name"])
            # 상장 약칭·종목코드도 **공식 표기**다. corp_name 만 보면
            # `현대차`(거래소 약칭)로 `현대자동차`를 못 찾는다.
            self._company_surfaces.setdefault(r["corp_code"], (
                r["corp_code"], r["corp_name"],
                r.get("listed_name"), r.get("stock_code")))
            # `extractor_failed` 도 **우리가 못 읽은 것**이다 — 「원문에 없음」과 다르다.
            if str(r.get("fact_extract_status") or "").startswith(
                    ("unsupported", "extractor_failed")):
                base = (f'{int(r["base_year"]):04d}{int(r["base_month"]):02d}'
                        if r.get("base_year") and r.get("base_month") else "")
                self._unsupported_docs.append({
                    "corp_code": r["corp_code"], "rcept_dt": r["rcept_dt"],
                    "rcept_no": r["rcept_no"],
                    "is_correction": is_correction,
                    "base": base,
                    #: 첨부 배치 문제일 때만 채워진다 — 「연결은 읽었고 별도만 못 읽었다」
                    "missing_scope": tuple(
                        (r.get("fact_unsupported_scope") or "").split("|")) if
                    r.get("fact_unsupported_scope") else (),
                })
        self._docs_loaded = True

    def document_label(self, doc_id: str) -> str | None:
        """그 문서의 사람이 읽는 이름 — 「삼성전자 사업보고서 (2025.12)」.

        이슈 #94 30 — 접수번호만으로는 어느 보고서인지 알 수 없어 사용자가
        원문을 찾아갈 수 없다.

        **회사명을 앞에 붙인다.** `report_nm` 만 쓰면 삼성전자와 SK하이닉스의
        출처가 둘 다 「사업보고서 (2025.12)」로 나와, 여러 회사를 견준 답변에서
        어느 줄이 누구 것인지 알 수 없다(실측 `G-A-004`).

        모르는 doc_id 면 ``None`` 이고, 그때는 종전처럼 접수번호만 나간다.
        """

        self._load_docs()
        meta = self._documents.get(doc_id)
        if meta is None:
            return None
        parts = [str(value or "").strip() for value in
                 (meta.corp_name, meta.report_nm)]
        return " ".join(part for part in parts if part) or None

    def period_label_years(self, doc_id: str) -> "dict[str, str]":
        """그 문서에서 회계 기수가 가리키는 실제 연도 — ``{"제50기": "2023"}``.

        이슈 #94 26 — 공시 원문은 기간을 「제55기 1분기」로 적는다. 그대로
        답변에 실으면 읽는 사람이 몇 년인지 알 수 없다.

        정본이 이미 짝을 싣고 있다: 재무제표 fact 한 행에 ``period_label``
        (원문이 그 열에 붙인 이름)과 ``period_end``(그것이 실제로 언제인지)가
        함께 있다. **문서 안에서는 이 짝이 1:1이다** — 전역으로는 「제50기」가
        네 해를 가리키지만(회사마다 설립연도가 다르다) 한 문서 안에서는
        어긋나는 경우가 없다(정본 6,917쌍 전수 확인, 2026-09-03).

        **기수 번호로 짝짓는다.** 재무제표는 기간을 「제55기 1분기」처럼 잘라
        적지만 서술 표는 「제54기」처럼 회계연도 전체를 가리킨다. 라벨 전체로
        짝지으면 후자를 못 찾는다 — 실측에서 분기보고서 fact 에는 «제54기1분기»
        만 있고 «제54기» 는 없다. 기수 번호가 회계연도를 가리키고 뒤의 수식은
        그 안의 어느 구간인지를 말하므로, 번호로 연도를 정하고 수식은 표면에만
        쓰는 것이 맞다.

        연도는 그 기수의 **가장 늦은 period_end** 가 속한 해다 — 회계연도를
        그 마감 시점으로 부르는 공시 관례와 같다.

        키는 공백을 지운 번호 형태다(「제 50 기」·「제50기말」 → ``제50기``).
        기수를 못 찾는 문서면 빈 dict 이고, 그때는 원문 표기를 그대로 둔다.
        """

        cached = self._period_years_by_doc.get(doc_id)
        if cached is not None:
            return cached
        artifact = self.root / "facts.parquet"
        latest_end: dict[str, str] = {}
        if artifact.is_file():
            columns = _present_columns(
                artifact, ["doc_id", "period_label", "period_end"])
            if "period_label" in columns and "period_end" in columns:
                for row in self._fact_records_for_docs(
                        artifact, columns, [doc_id]):
                    number = period_label_number(row.get("period_label"))
                    end = str(row.get("period_end") or "")
                    if not number or len(end) < 10:
                        continue
                    if end > latest_end.get(number, ""):
                        latest_end[number] = end
        mapping = {number: end[:4] for number, end in latest_end.items()}
        self._period_years_by_doc[doc_id] = mapping
        return mapping

    def documents(
            self, *, as_of: str, corp_code: str | None = None,
            corp_name: str | None = None, doc_group: str | None = None,
            form: str | None = None, report_name_contains: str | None = None,
            is_correction: bool | None = None,
            ) -> Iterator[DocumentMetadataRow]:
        """Stream deterministic, safe document metadata for selector binding."""

        cutoff = _checked_as_of(as_of)
        self._load_docs()
        report_key = (_text_key(report_name_contains)
                      if report_name_contains is not None else None)
        rows = sorted(
            self._documents.values(),
            key=lambda row: (row.rcept_dt, row.rcept_no, row.doc_id),
        )
        for meta in rows:
            if not self._matches_document(
                    meta,
                    as_of=cutoff,
                    corp_code=corp_code,
                    corp_name=corp_name,
                    doc_group=doc_group,
                    event_type=None,
                    form=form,
                    doc_id=None,
                    rcept_no=None,
                    is_correction=is_correction):
                continue
            if (report_key is not None
                    and report_key not in _text_key(meta.report_nm)):
                continue
            yield DocumentMetadataRow(
                doc_id=meta.doc_id,
                rcept_no=meta.rcept_no,
                corp_code=meta.corp_code,
                corp_name=meta.corp_name,
                filer=meta.filer,
                doc_group=meta.doc_group,
                form=meta.form,
                report_nm=meta.report_nm,
                rcept_dt=meta.rcept_dt,
                base_year=meta.base_year,
                base_month=meta.base_month,
                is_correction=meta.is_correction,
            )

    @staticmethod
    def _fact_row(r: dict) -> FactRow:
        """Parquet projection 한 행을 기존과 동일한 typed Fact로 만든다."""

        fk = r.get("evidence_id")
        return FactRow(
            doc_id=r["doc_id"], rcept_dt=r["rcept_dt"], scope=r["scope"],
            statement=r["statement"], account_raw=r["account_raw"],
            account_path=r["account_path"], account_norm=r["account_norm"],
            period_start=r["period_start"], period_end=r["period_end"],
            period_type=r["period_type"], cumulative=r["cumulative"],
            money=Money(r["raw_value"], r["raw_unit"], r["value_text"]),
            value_status=r["value_status"], locator=r["locator"],
            source_file_id=r.get("source_file_id"), block_id=r.get("block_id"),
            parent_id=r.get("parent_id"), path=r.get("path"),
            table_locator=r.get("table_locator"), logical_row=r.get("logical_row"),
            logical_col=r.get("logical_col"), evidence_id=fk,
            acode=r.get("acode"), account_norm_source=r.get("account_norm_source"),
            statement_title=r.get("statement_title"),
            period_label=r.get("period_label"),
            evidence_status="unverified" if fk else "unavailable")

    def _cache_facts(self, corp_code: str, rows: tuple[FactRow, ...]) -> None:
        """회사 Fact를 행 수 기준 LRU에 넣는다. oversize 회사는 상주시키지 않는다."""

        if len(rows) > self._fact_cache_max_rows:
            return
        previous = self._facts_by_corp.pop(corp_code, None)
        if previous is not None:
            self._fact_cache_rows -= len(previous)
        self._facts_by_corp[corp_code] = rows
        self._facts_by_corp.move_to_end(corp_code)
        self._fact_cache_rows += len(rows)
        while (self._fact_cache_rows > self._fact_cache_max_rows
               or len(self._facts_by_corp) > self._fact_cache_max_corps):
            _, evicted = self._facts_by_corp.popitem(last=False)
            self._fact_cache_rows -= len(evicted)

    @staticmethod
    def _fact_records_for_docs(
            artifact: Path, columns: list[str], selected_doc_ids: list[str],
            ) -> Iterator[dict]:
        """선택 문서가 있는 row group만 열고 일치 행만 Python 객체로 바꾼다.

        ``pq.read_table(filters=...)``는 파일 정렬/statistics에 따라 모든 value column을
        동시에 열 수 있다. 먼저 ``doc_id`` 한 열만 row group별로 확인하면 실제 Fact
        value는 최대 row group 하나씩만 메모리에 존재한다.
        """

        parquet = pq.ParquetFile(artifact)
        wanted = pa.array(tuple(dict.fromkeys(selected_doc_ids)), type=pa.string())
        for group_index in range(parquet.metadata.num_row_groups):
            selectors = parquet.read_row_group(
                group_index, columns=["doc_id"], use_threads=False)
            mask = pc.is_in(selectors["doc_id"], value_set=wanted)
            if not pc.any(mask).as_py():
                continue
            table = parquet.read_row_group(
                group_index, columns=columns, use_threads=False)
            mask = pc.is_in(table["doc_id"], value_set=wanted)
            yield from table.filter(mask).to_pylist()

    def _load_facts(self, corp_code: str) -> tuple[FactRow, ...]:
        """한 회사 Fact만 읽는다. 전량 Python 객체화는 어떤 경우에도 하지 않는다."""

        cached = self._facts_by_corp.get(corp_code)
        if cached is not None:
            self._facts_by_corp.move_to_end(corp_code)
            return cached
        self._load_docs()
        selected_doc_ids = [
            meta.doc_id for meta in self._documents.values()
            if meta.corp_code == corp_code
        ]
        if not selected_doc_ids:
            rows: tuple[FactRow, ...] = ()
            self._cache_facts(corp_code, rows)
            return rows

        artifact = self.root / "facts.parquet"
        cols = ["doc_id", "source_file_id", "block_id", "parent_id", "path",
                "rcept_dt", "scope", "statement", "account_raw", "account_path",
                "account_norm", "period_start", "period_end", "period_type",
                "cumulative", "raw_value", "raw_unit", "locator", "table_locator",
                "logical_row", "logical_col", "evidence_id", "value_text", "value_status",
                "acode", "account_norm_source", "statement_title", "period_label"]
        columns = _present_columns(artifact, cols)
        source = self._fact_records_for_docs(
            artifact, columns, selected_doc_ids)
        rows = tuple(self._fact_row(row) for row in source)
        self._cache_facts(corp_code, rows)
        return rows

    def _load_observations(self) -> None:
        if self._obs:
            return
        path = self.root / "event_observations.parquet"
        if not path.exists():
            raise NotReadyError(f"EventObservation 산출물이 없습니다: {path}")
        obs: dict[str, list[dict]] = defaultdict(list)
        event_keys_by_rcept: dict[str, set[str]] = defaultdict(set)
        wanted = ["doc_id", "event_key", "seq", "rcept_no", "observed_at",
                  "is_correction", "is_termination", "previous_observation_rcept_no",
                  "supporting_evidence_ids", "support_roles", "support_status",
                  "support_version", "support_limitation"]
        modern = _schema_at_least(self.schema_version, (1, 9))
        event_seq_keys: set[tuple[str, int]] = set()
        doc_ids: set[str] = set()
        rcept_nos: set[str] = set()
        for r in _rows(path, _present_columns(path, wanted)):
            if modern:
                event_key = str(r.get("event_key") or "")
                doc_id = str(r.get("doc_id") or "")
                rcept_no = str(r.get("rcept_no") or "")
                seq = r.get("seq")
                seq_key = (event_key, seq) if type(seq) is int else None
                if (seq_key is None or seq_key in event_seq_keys
                        or not doc_id or doc_id in doc_ids
                        or not rcept_no or rcept_no in rcept_nos):
                    raise EvidenceIntegrityError(
                        "EventObservation event/seq/doc/receipt cardinality 오류: "
                        f"event={event_key!r} seq={seq!r} "
                        f"doc={doc_id!r} rcept={rcept_no!r}")
                event_seq_keys.add(seq_key)
                doc_ids.add(doc_id)
                rcept_nos.add(rcept_no)
            obs[r["event_key"]].append(r)
            event_keys_by_rcept[str(r["rcept_no"])].add(str(r["event_key"]))
        if modern:
            self._load_docs()
            identity_path = self.root / "event_identities.parquet"
            identity_wanted = [
                "event_key", "kind", "corp_code", "corp_name", "doc_group",
                "doc_subtype", "root_rcept_no", "identity_fingerprint",
                "identity_status", "resolver_version", "first_disclosed_at",
                "last_disclosed_at", "n_observations", "n_corrections",
                "status_at_corpus_end",
            ]
            identities: dict[str, dict] = {}
            for row in _rows(
                    identity_path, _present_columns(identity_path, identity_wanted)):
                event_key = str(row.get("event_key") or "")
                if not event_key or event_key in identities:
                    raise EvidenceIntegrityError(
                        f"EventIdentity event_key 전역 유일성 오류: {event_key!r}")
                identities[event_key] = row
            if set(identities) != set(obs):
                raise EvidenceIntegrityError(
                    "EventIdentity/EventObservation event_key 집합 불일치")
            document_contract = {
                doc_id: {
                    "rcept_no": meta.rcept_no, "rcept_dt": meta.rcept_dt,
                    "doc_group": meta.doc_group, "doc_subtype": meta.doc_subtype,
                    "event_type": meta.event_type, "report_nm": meta.report_nm,
                    "is_correction": meta.is_correction,
                }
                for doc_id, meta in self._documents.items()
            }
            for event_key, identity in identities.items():
                try:
                    validate_event_history(
                        identity, obs.get(event_key, ()), document_contract)
                except (TypeError, ValueError) as exc:
                    raise EvidenceIntegrityError(
                        f"Event history aggregate 무결성 오류: {event_key}") from exc
            self._event_identities = identities
        for v in obs.values():
            v.sort(key=(lambda x: (x["observed_at"], x["rcept_no"]))
                   if modern else (lambda x: (x["observed_at"], x["seq"])))
        self._obs = dict(obs)
        self._event_keys_by_rcept = {
            receipt: tuple(sorted(keys))
            for receipt, keys in event_keys_by_rcept.items()
        }

    def _load_event_identities(self) -> None:
        if self._event_identities is not None:
            return
        if _schema_at_least(self.schema_version, (1, 9)):
            # 1.9는 Identity aggregate를 Observation+Document와 함께 검증해야 한다.
            # 별도 dict comprehension으로 먼저 읽으면 변조 aggregate가 노출될 수 있다.
            self._load_observations()
            if self._event_identities is None:
                raise EvidenceIntegrityError("EventIdentity/Observation 검증이 완료되지 않았습니다")
            return
        path = self.root / "event_identities.parquet"
        if not path.exists():
            raise NotReadyError(f"EventIdentity 산출물이 없습니다: {path}")
        wanted = ["event_key", "kind", "corp_code", "corp_name", "doc_group",
                  "doc_subtype", "root_rcept_no", "identity_fingerprint",
                  "identity_status", "resolver_version", "first_disclosed_at",
                  "last_disclosed_at", "n_observations", "n_corrections",
                  "status_at_corpus_end"]
        identities: dict[str, dict] = {}
        for r in _rows(path, _present_columns(path, wanted)):
            event_key = str(r.get("event_key") or "")
            if (_schema_at_least(self.schema_version, (1, 9))
                    and (not event_key or event_key in identities)):
                raise EvidenceIntegrityError(
                    f"EventIdentity event_key 전역 유일성 오류: {event_key!r}")
            identities[event_key] = r
        self._event_identities = identities

    # --------------------------------------------------------- SourceFile 경로 계약
    def source_file(self, source_file_id: str) -> SourceFileRow | None:
        """SourceFile 한 건과 문서 단위 PDF/HTML 검증 상태를 typed row로 반환한다.

        경로 문자열 자체는 열지 않는다. ``normalized_path_key``를 실제 경로처럼 쓰는
        실수를 막기 위해 원문 접근은 :meth:`resolve_source_path`로 분리한다.
        """
        wanted_id = str(source_file_id or "")
        if not wanted_id:
            raise ValueError("source_file_id는 빈 문자열일 수 없습니다")
        self._load_docs()
        artifact = self.root / "source_files.parquet"
        wanted = [
            "source_file_id", "doc_id", "relpath", "normalized_path_key",
            "actual_relpath", "role", "sha256", "byte_size", "parse_mode",
            "source_selection_role", "parse_usable", "coverage_n_sections",
            "coverage_n_chars", "coverage_n_tables", "coverage_n_pages",
            "coverage_pages_with_text", "page_text_coverage", "locator_kind",
            "locator_limitations", "parse_warnings", "parse_error",
        ]
        columns = _present_columns(artifact, wanted)
        required = {"source_file_id", "doc_id", "relpath", "role", "sha256",
                    "byte_size", "parse_mode"}
        missing = sorted(required - set(columns))
        if missing:
            raise SourceIntegrityError(
                f"SourceFile 필수 컬럼이 없습니다: {', '.join(missing)}")
        table = _read_artifact_table(
            artifact, columns=columns, filters=[("source_file_id", "=", wanted_id)])
        matches = table.to_pylist()
        if not matches:
            return None
        if len(matches) != 1:
            raise SourceIntegrityError(
                f"source_file_id={wanted_id!r}가 {len(matches)}행입니다")
        row = matches[0]
        path_key = str(row.get("normalized_path_key") or row["relpath"])
        if normalized_path_key(path_key) != path_key:
            raise SourceIntegrityError(
                f"SourceFile normalized_path_key가 NFC canonical이 아닙니다: {wanted_id}")
        calculated = calculate_source_file_id(path_key)
        if calculated != wanted_id:
            raise SourceIntegrityError(
                f"SourceFile ID 재계산 불일치: 요청={wanted_id}, 계산={calculated}")
        meta = self._documents.get(row["doc_id"])
        if meta is None:
            raise SourceIntegrityError(
                f"SourceFile의 Document FK가 없습니다: {row['doc_id']}")
        return SourceFileRow(
            source_file_id=wanted_id, doc_id=row["doc_id"],
            normalized_path_key=path_key,
            actual_relpath=row.get("actual_relpath"), role=str(row["role"]),
            sha256=str(row["sha256"]), byte_size=int(row["byte_size"]),
            parse_mode=str(row["parse_mode"]),
            source_selection_role=row.get("source_selection_role"),
            parse_usable=row.get("parse_usable"),
            coverage_n_sections=row.get("coverage_n_sections"),
            coverage_n_chars=row.get("coverage_n_chars"),
            coverage_n_tables=row.get("coverage_n_tables"),
            coverage_n_pages=row.get("coverage_n_pages"),
            coverage_pages_with_text=row.get("coverage_pages_with_text"),
            page_text_coverage=row.get("page_text_coverage"),
            locator_kind=row.get("locator_kind"),
            locator_limitations=tuple(row.get("locator_limitations") or ()),
            parse_warnings=tuple(row.get("parse_warnings") or ()),
            parse_error=row.get("parse_error"),
            primary_source_file_id=meta.primary_source_file_id,
            alternate_source_file_ids=meta.alternate_source_file_ids,
            source_cross_check_status=meta.source_cross_check_status,
            source_cross_check_reason=meta.source_cross_check_reason,
            source_cross_check_source_file_ids=meta.source_cross_check_source_file_ids,
        )

    def resolve_source_path(
            self, source_file_id: str, *, corpus_root: str | Path = "data/corpus",
            allow_restricted_raw: bool = False, verify_sha256: bool = True,
            ) -> Path | None:
        """NFC key를 실제 inventory Path에 결합하고 크기·SHA를 확인한다.

        반환 경로는 restricted raw 원문을 가리키므로 호출자가
        ``allow_restricted_raw=True``를 명시해야 한다. schema 1.2도 NFC ``relpath``를
        inventory에 다시 bind하므로 NFD 폴더에서 조용히 빈 결과가 되지 않는다.
        """
        allow_restricted_raw = _checked_bool_flag(
            "allow_restricted_raw", allow_restricted_raw)
        verify_sha256 = _checked_bool_flag("verify_sha256", verify_sha256)
        if not allow_restricted_raw:
            raise ReadSecurityError(
                "원문 경로는 restricted입니다. allow_restricted_raw=True를 명시하세요")
        row = self.source_file(source_file_id)
        if row is None:
            return None
        root = Path(corpus_root).resolve()
        try:
            index = self._corpus_indexes.get(root)
            if index is None:
                index = CorpusIndex.build(root)
                self._corpus_indexes[root] = index
            binding = index.bind(row.normalized_path_key)
        except (FileNotFoundError, CorpusPathNotFound, PathCollisionError) as exc:
            raise SourceIntegrityError(str(exc)) from exc
        if (row.actual_relpath is not None
                and binding.actual_relpath != row.actual_relpath):
            raise SourceIntegrityError(
                "SourceFile actual_relpath와 현재 inventory binding이 다릅니다: "
                f"{row.actual_relpath!r} != {binding.actual_relpath!r}")
        actual = binding.path.resolve()
        try:
            actual.relative_to(root)
        except ValueError as exc:
            raise SourceIntegrityError(
                f"SourceFile 경로가 corpus_root 밖입니다: {actual}") from exc
        if not actual.is_file():
            raise SourceIntegrityError(f"SourceFile이 실제 파일이 아닙니다: {actual}")
        size = actual.stat().st_size
        if size != row.byte_size:
            raise SourceIntegrityError(
                f"SourceFile byte_size 불일치: 저장={row.byte_size}, 실제={size}")
        if verify_sha256:
            digest = file_sha256(actual)
            if digest != row.sha256:
                raise SourceIntegrityError(
                    f"SourceFile SHA-256 불일치: 저장={row.sha256}, 실제={digest}")
        return actual

    @staticmethod
    def _filtered_evidence_references(
            artifact: Path, columns: tuple[str, ...],
            filters: list[tuple[str, str, str]]) -> list[dict]:
        """Evidence FK를 가진 semantic 행만 읽고, 계약 누락은 fail-closed 한다."""
        if not artifact.exists():
            raise EvidenceIntegrityError(
                f"Evidence semantic 원천 산출물이 없습니다: {artifact}")
        present = _artifact_columns(artifact)
        missing = sorted(set(columns) - present)
        if missing:
            raise EvidenceIntegrityError(
                f"{artifact.name} Evidence 재생성 컬럼 누락: {', '.join(missing)}")
        try:
            return _read_artifact_table(
                artifact, columns=list(columns), filters=filters).to_pylist()
        except Exception as exc:
            raise EvidenceIntegrityError(
                f"{artifact.name} Evidence 참조 조회 실패") from exc

    def _evidence_document_context(
            self, doc_id: object, rcept_dt: object) -> _DocumentInfo:
        """빌드 당시 Document 회사명 문맥을 정확히 돌려준다."""
        if not isinstance(doc_id, str) or not doc_id:
            raise EvidenceIntegrityError("Evidence semantic 참조의 doc_id가 없습니다")
        if not isinstance(rcept_dt, str) or not rcept_dt:
            raise EvidenceIntegrityError(
                f"Evidence semantic 참조의 rcept_dt가 없습니다: {doc_id}")
        self._load_docs()
        if doc_id in self._duplicate_document_ids:
            raise EvidenceIntegrityError(
                f"Evidence Document FK가 복수 행입니다: {doc_id}")
        meta = self._documents.get(doc_id)
        if meta is None:
            raise EvidenceIntegrityError(
                f"Evidence semantic 참조의 Document FK가 없습니다: {doc_id}")
        if (not isinstance(meta.corp_name, str) or not meta.corp_name.strip()
                or not isinstance(meta.listed_name, str)
                or not meta.listed_name.strip()):
            raise EvidenceIntegrityError(
                f"Evidence Document 회사명 문맥이 없습니다: {doc_id}")
        if meta.rcept_dt != rcept_dt:
            raise EvidenceIntegrityError(
                "Evidence semantic 참조와 Document rcept_dt가 다릅니다: "
                f"{doc_id} ({rcept_dt} != {meta.rcept_dt})")
        return meta

    def _verify_evidence_source_owner(
            self, source_file_id: str, doc_id: str) -> None:
        """SourceFile FK가 정확히 한 행이고 semantic Document 소유인지 확인한다."""
        cached_owner = self._evidence_source_owner_cache.get(source_file_id)
        if cached_owner is not None:
            if cached_owner != doc_id:
                raise EvidenceIntegrityError(
                    "Evidence source_file_id가 다른 Document 소유입니다: "
                    f"{source_file_id} ({cached_owner} != {doc_id})")
            return
        columns = ("build_id", "source_file_id", "doc_id")
        rows = self._filtered_evidence_references(
            self.root / "source_files.parquet", columns,
            [("source_file_id", "=", source_file_id)])
        if len(rows) != 1:
            raise EvidenceIntegrityError(
                "Evidence SourceFile FK가 없거나 복수입니다: "
                f"{source_file_id} ({len(rows)}행)")
        row = rows[0]
        if row.get("build_id") != self.build_id:
            raise EvidenceIntegrityError(
                f"Evidence SourceFile build_id 불일치: {source_file_id}")
        owner = row.get("doc_id")
        if not isinstance(owner, str) or not owner:
            raise EvidenceIntegrityError(
                f"Evidence SourceFile doc_id 누락: {source_file_id}")
        if owner != doc_id:
            raise EvidenceIntegrityError(
                "Evidence source_file_id가 다른 Document 소유입니다: "
                f"{source_file_id} ({owner} != {doc_id})")
        self._evidence_source_owner_cache[source_file_id] = owner

    def _recreate_evidence(self, stored: dict):
        """모든 semantic FK에서 Evidence를 독립 재생성하고 저장 행과 대조한다."""
        wanted_id = stored["evidence_id"]
        kind = stored["kind"]
        stored_policy = stored.get("security_policy_version")
        if not isinstance(stored_policy, str) or not stored_policy:
            raise EvidenceIntegrityError(
                f"Evidence security_policy_version이 없습니다: {wanted_id}")

        def recreate(
                ref: dict, *, raw: object, locator: object,
                table_locator: object, logical_row: object, logical_col: object,
                extraction_method: str, corporate_names: tuple[str, ...],
                pii_label: str | None):
            if ref.get("build_id") != self.build_id:
                raise EvidenceIntegrityError(
                    f"Evidence semantic 참조 build_id 불일치: {wanted_id}")
            if ref.get("evidence_id") != wanted_id:
                raise EvidenceIntegrityError(
                    f"Evidence semantic FK 불일치: {wanted_id}")
            doc_id = ref.get("doc_id")
            source_file_id = ref.get("source_file_id")
            rcept_dt = ref.get("rcept_dt")
            if not isinstance(source_file_id, str) or not source_file_id:
                raise EvidenceIntegrityError(
                    f"Evidence semantic 참조 source_file_id 누락: {wanted_id}")
            if not isinstance(doc_id, str) or not doc_id:
                raise EvidenceIntegrityError(
                    f"Evidence semantic 참조 doc_id 누락: {wanted_id}")
            self._verify_evidence_source_owner(source_file_id, doc_id)
            if not isinstance(raw, str):
                raise EvidenceIntegrityError(
                    f"Evidence semantic 원문 값이 문자열이 아닙니다: {wanted_id}")
            if not isinstance(locator, str) or not locator:
                raise EvidenceIntegrityError(
                    f"Evidence semantic locator 누락: {wanted_id}")
            if not isinstance(table_locator, str) or not table_locator:
                raise EvidenceIntegrityError(
                    f"Evidence semantic table_locator 누락/오류: {wanted_id}")
            try:
                actual_table = table_locator_of(locator)
            except CanonicalLocatorError as exc:
                raise EvidenceIntegrityError(
                    f"Evidence semantic locator 오류: {wanted_id}") from exc
            if actual_table != table_locator:
                raise EvidenceIntegrityError(
                    f"Evidence semantic TABLE ancestry 불일치: {wanted_id}")
            try:
                _checked_as_of(rcept_dt)
            except (TypeError, ValueError) as exc:
                raise EvidenceIntegrityError(
                    f"Evidence semantic rcept_dt 오류: {wanted_id}") from exc
            if (type(logical_row) is not int or logical_row < 0
                    or type(logical_col) is not int or logical_col < 0):
                raise EvidenceIntegrityError(
                    f"Evidence semantic 논리 좌표 오류: {wanted_id}")
            if any(not isinstance(name, str) or not name.strip()
                   for name in corporate_names):
                raise EvidenceIntegrityError(
                    f"Evidence 회사명 문맥 누락: {wanted_id}")
            if pii_label is not None and not isinstance(pii_label, str):
                raise EvidenceIntegrityError(
                    f"Evidence PII label 오류: {wanted_id}")
            try:
                return make_evidence(
                    build_id=self.build_id, doc_id=doc_id,
                    source_file_id=source_file_id, kind=kind,
                    locator=locator, table_locator=table_locator,
                    logical_row=logical_row, logical_col=logical_col,
                    excerpt_raw=raw, extraction_method=extraction_method,
                    extraction_status="ok", rcept_dt=rcept_dt,
                    corporate_names=corporate_names, pii_label=pii_label,
                    pii_subject_name=self._pii_subject(
                        doc_id, pii_label, table_locator, logical_row),
                    pii_party_type=self._pii_party_type(
                        doc_id, pii_label, raw, table_locator, logical_row,
                        kind=kind),
                    security_policy_version=stored_policy)
            except (CanonicalLocatorError, TypeError, ValueError) as exc:
                raise EvidenceIntegrityError(
                    f"Evidence semantic 재생성 실패: {wanted_id}") from exc

        candidates = []
        # 저장 kind만 따라가면 다른 semantic 테이블에 같은 FK를 심은 변조를
        # 놓친다. 요청 ID만 최소열로 조회해 cross-kind 참조를 먼저 닫는다.
        foreign_kinds: list[str] = []
        if kind != "fact_value" and self._filtered_evidence_references(
                self.root / "facts.parquet", ("evidence_id",),
                [("evidence_id", "=", wanted_id)]):
            foreign_kinds.append("fact_value")
        if kind != "field_value" and self._filtered_evidence_references(
                self.root / "fields.parquet", ("evidence_id",),
                [("evidence_id", "=", wanted_id)]):
            foreign_kinds.append("field_value")
        if kind != "correction_value":
            for side in ("before", "after"):
                column = f"{side}_evidence_id"
                if self._filtered_evidence_references(
                        self.root / "correction_items.parquet", (column,),
                        [(column, "=", wanted_id)]):
                    foreign_kinds.append("correction_value")
                    break
        if foreign_kinds:
            raise EvidenceIntegrityError(
                "Evidence가 저장 kind와 다른 semantic 종류에도 참조됩니다: "
                f"{wanted_id} ({', '.join(foreign_kinds)})")

        if kind == "fact_value":
            columns = (
                "build_id", "evidence_id", "doc_id", "source_file_id",
                "locator", "table_locator", "logical_row", "logical_col",
                "value_text", "rcept_dt",
            )
            refs = self._filtered_evidence_references(
                self.root / "facts.parquet", columns,
                [("evidence_id", "=", wanted_id)])
            for ref in refs:
                meta = self._evidence_document_context(
                    ref.get("doc_id"), ref.get("rcept_dt"))
                candidates.append(recreate(
                    ref, raw=ref.get("value_text"), locator=ref.get("locator"),
                    table_locator=ref.get("table_locator"),
                    logical_row=ref.get("logical_row"),
                    logical_col=ref.get("logical_col"),
                    extraction_method="financial_table_cell",
                    corporate_names=(meta.corp_name, meta.listed_name),
                    pii_label=None))
        elif kind == "field_value":
            columns = (
                "build_id", "evidence_id", "doc_id", "source_file_id", "path",
                "locator", "table_locator", "logical_row", "logical_col",
                "value_raw", "rcept_dt",
            )
            refs = self._filtered_evidence_references(
                self.root / "fields.parquet", columns,
                [("evidence_id", "=", wanted_id)])
            for ref in refs:
                meta = self._evidence_document_context(
                    ref.get("doc_id"), ref.get("rcept_dt"))
                label = ref.get("path")
                # 실제 schema 1.8 Field에는 path==""도 있다. 빈 문자열은
                # 빌드의 exact pii_label 입력이고, 컬럼 부재/None/비문자열만 손상이다.
                if not isinstance(label, str):
                    raise EvidenceIntegrityError(
                        f"Field Evidence label 문맥이 없습니다: {wanted_id}")
                candidates.append(recreate(
                    ref, raw=ref.get("value_raw"), locator=ref.get("locator"),
                    table_locator=ref.get("table_locator"),
                    logical_row=ref.get("logical_row"),
                    logical_col=ref.get("logical_col"),
                    extraction_method="structured_form_field",
                    corporate_names=(meta.corp_name, meta.listed_name),
                    pii_label=label))
        elif kind == "correction_value":
            columns = (
                "build_id", "doc_id", "source_file_id", "path", "corp_name",
                "value_before", "value_after", "before_locator", "after_locator",
                "before_logical_row", "before_logical_col",
                "after_logical_row", "after_logical_col",
                "before_evidence_id", "after_evidence_id", "rcept_dt",
            )
            refs_with_side: list[tuple[dict, str]] = []
            for side in ("before", "after"):
                side_refs = self._filtered_evidence_references(
                    self.root / "correction_items.parquet", columns,
                    [(f"{side}_evidence_id", "=", wanted_id)])
                refs_with_side.extend((ref, side) for ref in side_refs)
            for ref, side in refs_with_side:
                ref = dict(ref)
                ref["evidence_id"] = ref.get(f"{side}_evidence_id")
                meta = self._evidence_document_context(
                    ref.get("doc_id"), ref.get("rcept_dt"))
                correction_corp = ref.get("corp_name")
                if (not isinstance(correction_corp, str)
                        or correction_corp != meta.corp_name):
                    raise EvidenceIntegrityError(
                        "Correction Evidence 회사 문맥과 Document가 다릅니다: "
                        f"{wanted_id}")
                label = ref.get("path")
                if not isinstance(label, str) or not label:
                    raise EvidenceIntegrityError(
                        f"Correction Evidence label 문맥이 없습니다: {wanted_id}")
                locator = ref.get(f"{side}_locator")
                try:
                    table_locator = (table_locator_of(locator)
                                     if isinstance(locator, str) else None)
                except CanonicalLocatorError as exc:
                    raise EvidenceIntegrityError(
                        f"Correction Evidence locator 오류: {wanted_id}") from exc
                if table_locator is None:
                    raise EvidenceIntegrityError(
                        f"Correction Evidence TABLE 문맥이 없습니다: {wanted_id}")
                candidates.append(recreate(
                    ref, raw=ref.get(f"value_{side}"), locator=locator,
                    table_locator=table_locator,
                    logical_row=ref.get(f"{side}_logical_row"),
                    logical_col=ref.get(f"{side}_logical_col"),
                    extraction_method="declared_correction_table",
                    corporate_names=(correction_corp,), pii_label=label))
        else:
            raise EvidenceIntegrityError(
                f"알 수 없는 Evidence kind입니다: {kind!r}")

        if not candidates:
            raise EvidenceIntegrityError(
                f"Evidence semantic FK가 없는 orphan입니다: {wanted_id}")

        comparable = (
            "build_id", "evidence_id", "doc_id", "source_file_id", "kind",
            "locator", "table_locator", "logical_row", "logical_col",
            "excerpt_raw", "excerpt_safe", "excerpt_hash", "raw_access",
            "extraction_method", "extraction_status", "security_flags",
            "pii_types", "security_policy_version", "evidence_policy_version",
            "rcept_dt",
        )
        first = candidates[0]
        for index, candidate in enumerate(candidates):
            mismatches = [
                name for name in comparable
                if getattr(candidate, name) != stored.get(name)
            ]
            if mismatches:
                raise EvidenceIntegrityError(
                    "Evidence semantic 재생성 불일치"
                    f"[{index}]: {wanted_id} ({', '.join(mismatches)})")
            if candidate != first:
                raise EvidenceIntegrityError(
                    f"Evidence 복수 semantic 참조가 상충합니다: {wanted_id}")
        return first

    # --------------------------------------------------------- first-class Evidence
    def _cached_evidence(self, evidence_id: str) -> EvidenceRow | None:
        row = self._evidence_cache.get(evidence_id)
        if row is not None:
            self._evidence_cache.move_to_end(evidence_id)
        return row

    def _cache_evidence(self, row: EvidenceRow) -> None:
        self._evidence_cache[row.evidence_id] = row
        self._evidence_cache.move_to_end(row.evidence_id)
        while len(self._evidence_cache) > self._evidence_cache_limit:
            self._evidence_cache.popitem(last=False)

    def get_evidence_many(
            self, evidence_ids: Iterable[str], *,
            include_restricted_raw: bool = False,
            expected_kind: str | None = None,
            ) -> dict[str, EvidenceRow | None]:
        """여러 Evidence를 artifact별 **한 번의 batch scan**으로 검증한다.

        ID마다 :meth:`get_evidence`를 호출하면 Evidence/Source/semantic Parquet를
        반복해서 읽는다. 이 경로는 요청 ID 집합을 먼저 고정하고 Evidence, Fact,
        Field, CorrectionItem, Chunk, SourceFile을 각각 최대 한 번 읽은 뒤 같은 exact
        recreation 계약을 적용한다. 결과 dict는 입력의 첫 등장 순서를 유지한다.
        """
        include_restricted_raw = _checked_bool_flag(
            "include_restricted_raw", include_restricted_raw)
        if expected_kind is not None and expected_kind not in {
                "fact_value", "field_value", "correction_value", "chunk_text"}:
            raise ValueError(f"알 수 없는 expected_kind입니다: {expected_kind!r}")
        modern_chunk_evidence = _schema_at_least(self.schema_version, (1, 9))
        if expected_kind == "chunk_text" and not modern_chunk_evidence:
            raise NotReadyError("chunk_text Evidence는 schema 1.9+ artifact가 필요합니다")

        ordered: list[str] = []
        seen: set[str] = set()
        for value in evidence_ids:
            if not isinstance(value, str) or not value:
                raise ValueError("evidence_id는 비어 있지 않은 문자열이어야 합니다")
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        if not ordered:
            return {}

        results: dict[str, EvidenceRow | None] = {}
        unresolved: list[str] = []
        for evidence_key in ordered:
            cached = (None if include_restricted_raw
                      else self._cached_evidence(evidence_key))
            if cached is not None:
                if expected_kind is not None and cached.kind != expected_kind:
                    raise EvidenceIntegrityError(
                        f"Evidence kind 불일치: {evidence_key} "
                        f"({cached.kind} != {expected_kind})")
                results[evidence_key] = cached
            else:
                unresolved.append(evidence_key)
        if not unresolved:
            return {key: results[key] for key in ordered}

        evidence_artifact = self.root / "evidence.parquet"
        if not evidence_artifact.exists():
            raise NotReadyError(
                f"Evidence 산출물이 없습니다: {evidence_artifact} "
                "(schema 1.5 재빌드 필요)")
        evidence_columns = [
            "build_id", "evidence_id", "doc_id", "source_file_id", "kind",
            "locator", "table_locator", "logical_row", "logical_col", "excerpt_raw",
            "excerpt_safe", "excerpt_hash", "raw_access", "extraction_method",
            "extraction_status", "security_flags", "pii_types",
            "security_policy_version", "evidence_policy_version", "rcept_dt",
        ]

        def read_rows(path: Path, columns: list[str], *,
                      id_column: str | None = None) -> list[dict]:
            if not path.exists():
                raise EvidenceIntegrityError(
                    f"Evidence semantic 원천 산출물이 없습니다: {path}")
            present = _artifact_columns(path)
            missing = sorted(set(columns) - present)
            if missing:
                raise EvidenceIntegrityError(
                    f"{path.name} Evidence 재생성 컬럼 누락: {', '.join(missing)}")
            try:
                filters = ([((id_column or "evidence_id"), "in", unresolved)]
                           if id_column is not None else None)
                return _read_artifact_table(
                    path, columns=columns, filters=filters).to_pylist()
            except Exception as exc:
                raise EvidenceIntegrityError(
                    f"{path.name} Evidence batch 조회 실패") from exc

        stored_rows = read_rows(
            evidence_artifact, evidence_columns, id_column="evidence_id")
        stored_by_id: dict[str, dict] = {}
        for stored in stored_rows:
            evidence_key = stored.get("evidence_id")
            if evidence_key in stored_by_id:
                raise EvidenceIntegrityError(
                    f"evidence_id={evidence_key!r}가 복수 행입니다")
            stored_by_id[evidence_key] = stored

        semantic: dict[str, list[tuple[str, dict, str | None]]] = defaultdict(list)
        facts = read_rows(self.root / "facts.parquet", [
            "build_id", "evidence_id", "doc_id", "source_file_id", "locator",
            "table_locator", "logical_row", "logical_col", "value_text", "rcept_dt",
        ], id_column="evidence_id")
        for ref in facts:
            semantic[str(ref["evidence_id"])].append(("fact_value", ref, None))

        fields = read_rows(self.root / "fields.parquet", [
            "build_id", "evidence_id", "doc_id", "source_file_id", "path", "locator",
            "table_locator", "logical_row", "logical_col", "value_raw", "rcept_dt",
        ], id_column="evidence_id")
        for ref in fields:
            semantic[str(ref["evidence_id"])].append(("field_value", ref, None))

        # CorrectionItem은 작고 before/after OR filter를 두 번 수행하면 같은 artifact를
        # 반복 스캔한다. 필요한 열을 한 번 읽고 요청 ID만 메모리에서 거른다.
        correction_columns = [
            "build_id", "doc_id", "source_file_id", "path", "corp_name",
            "value_before", "value_after", "before_locator", "after_locator",
            "before_logical_row", "before_logical_col", "after_logical_row",
            "after_logical_col", "before_evidence_id", "after_evidence_id", "rcept_dt",
        ]
        corrections = read_rows(
            self.root / "correction_items.parquet", correction_columns)
        unresolved_set = set(unresolved)
        for ref in corrections:
            for side in ("before", "after"):
                evidence_key = ref.get(f"{side}_evidence_id")
                if evidence_key in unresolved_set:
                    semantic[str(evidence_key)].append(
                        ("correction_value", ref, side))

        if modern_chunk_evidence:
            chunks = read_rows(self.root / "chunks.parquet", [
                "build_id", "evidence_id", "doc_id", "source_file_id", "locator",
                "part_no", "n_parts", "text", "text_prompt_safe", "security_flags",
                "pii_types", "security_policy_version", "rcept_dt",
            ], id_column="evidence_id")
            for ref in chunks:
                semantic[str(ref["evidence_id"])].append(("chunk_text", ref, None))

        source_ids = {
            str(row.get("source_file_id"))
            for row in stored_rows
            if isinstance(row.get("source_file_id"), str) and row.get("source_file_id")
        }
        for entries in semantic.values():
            for _kind, ref, _side in entries:
                value = ref.get("source_file_id")
                if isinstance(value, str) and value:
                    source_ids.add(value)
        source_artifact = self.root / "source_files.parquet"
        source_columns = ["build_id", "source_file_id", "doc_id"]
        present = _artifact_columns(source_artifact)
        missing = sorted(set(source_columns) - present)
        if missing:
            raise EvidenceIntegrityError(
                "source_files.parquet Evidence ownership 컬럼 누락: "
                + ", ".join(missing))
        source_rows = (_read_artifact_table(
            source_artifact, columns=source_columns,
            filters=[("source_file_id", "in", sorted(source_ids))]).to_pylist()
            if source_ids else [])
        source_owner: dict[str, tuple[str, str]] = {}
        for row in source_rows:
            source_key = row.get("source_file_id")
            if source_key in source_owner:
                raise EvidenceIntegrityError(
                    f"Evidence SourceFile FK가 복수 행입니다: {source_key}")
            source_owner[str(source_key)] = (
                str(row.get("build_id") or ""), str(row.get("doc_id") or ""))

        self._load_docs()

        def recreate(kind: str, ref: dict, side: str | None):
            evidence_key = str(
                ref.get(f"{side}_evidence_id") if side else ref.get("evidence_id"))
            if ref.get("build_id") != self.build_id:
                raise EvidenceIntegrityError(
                    f"Evidence semantic 참조 build_id 불일치: {evidence_key}")
            doc_id = ref.get("doc_id")
            source_file_id = ref.get("source_file_id")
            rcept_dt = ref.get("rcept_dt")
            if not isinstance(doc_id, str) or not doc_id:
                raise EvidenceIntegrityError(
                    f"Evidence semantic 참조 doc_id 누락: {evidence_key}")
            if not isinstance(source_file_id, str) or not source_file_id:
                raise EvidenceIntegrityError(
                    f"Evidence semantic 참조 source_file_id 누락: {evidence_key}")
            owner = source_owner.get(source_file_id)
            if owner != (self.build_id, doc_id):
                raise EvidenceIntegrityError(
                    "Evidence SourceFile ownership 불일치: "
                    f"{evidence_key} ({owner!r} != {(self.build_id, doc_id)!r})")
            meta = self._evidence_document_context(doc_id, rcept_dt)
            stored_policy = stored_by_id.get(evidence_key, {}).get(
                "security_policy_version")
            if not isinstance(stored_policy, str) or not stored_policy:
                raise EvidenceIntegrityError(
                    f"Evidence security_policy_version이 없습니다: {evidence_key}")

            if kind == "fact_value":
                raw = ref.get("value_text")
                locator = ref.get("locator")
                table_locator = ref.get("table_locator")
                logical_row, logical_col = ref.get("logical_row"), ref.get("logical_col")
                method = "financial_table_cell"
                names = (meta.corp_name, meta.listed_name)
                pii_label = None
            elif kind == "field_value":
                raw = ref.get("value_raw")
                locator = ref.get("locator")
                table_locator = ref.get("table_locator")
                logical_row, logical_col = ref.get("logical_row"), ref.get("logical_col")
                method = "structured_form_field"
                names = (meta.corp_name, meta.listed_name)
                pii_label = ref.get("path")
                if not isinstance(pii_label, str):
                    raise EvidenceIntegrityError(
                        f"Field Evidence label 문맥이 없습니다: {evidence_key}")
            elif kind == "correction_value":
                assert side in ("before", "after")
                raw = ref.get(f"value_{side}")
                locator = ref.get(f"{side}_locator")
                try:
                    table_locator = (table_locator_of(locator)
                                     if isinstance(locator, str) else None)
                except CanonicalLocatorError as exc:
                    raise EvidenceIntegrityError(
                        f"Correction Evidence locator 오류: {evidence_key}") from exc
                logical_row = ref.get(f"{side}_logical_row")
                logical_col = ref.get(f"{side}_logical_col")
                method = "declared_correction_table"
                correction_corp = ref.get("corp_name")
                if correction_corp != meta.corp_name:
                    raise EvidenceIntegrityError(
                        f"Correction Evidence 회사 문맥 불일치: {evidence_key}")
                names = (str(correction_corp),)
                pii_label = ref.get("path")
                if not isinstance(pii_label, str) or not pii_label:
                    raise EvidenceIntegrityError(
                        f"Correction Evidence label 문맥이 없습니다: {evidence_key}")
            elif kind == "chunk_text":
                raw = ref.get("text")
                locator = ref.get("locator")
                table_locator = logical_row = logical_col = None
                method = "section_chunk"
                names = (meta.corp_name, meta.listed_name)
                pii_label = None
                part_no, n_parts = ref.get("part_no"), ref.get("n_parts")
                if (type(part_no) is not int or type(n_parts) is not int
                        or part_no < 0 or n_parts <= 0 or part_no >= n_parts
                        or not isinstance(locator, str)
                        or not locator.endswith(f"/PART[{part_no}]")):
                    raise EvidenceIntegrityError(
                        f"Chunk PART/n_parts 계약 오류: {evidence_key}")
            else:  # pragma: no cover - caller closes vocabulary
                raise EvidenceIntegrityError(f"알 수 없는 Evidence kind: {kind}")

            if not isinstance(raw, str):
                raise EvidenceIntegrityError(
                    f"Evidence semantic raw가 문자열이 아닙니다: {evidence_key}")
            if not isinstance(locator, str) or not locator:
                raise EvidenceIntegrityError(
                    f"Evidence semantic locator 누락: {evidence_key}")
            try:
                canonical_locator(locator)
            except CanonicalLocatorError as exc:
                raise EvidenceIntegrityError(
                    f"Evidence semantic locator 오류: {evidence_key}") from exc
            if kind == "chunk_text":
                # null 좌표 허용은 chunk kind 하나뿐이다.
                pass
            else:
                if not isinstance(table_locator, str) or not table_locator:
                    raise EvidenceIntegrityError(
                        f"Evidence semantic table_locator 누락: {evidence_key}")
                if table_locator_of(locator) != table_locator:
                    raise EvidenceIntegrityError(
                        f"Evidence TABLE ancestry 불일치: {evidence_key}")
                if (type(logical_row) is not int or logical_row < 0
                        or type(logical_col) is not int or logical_col < 0):
                    raise EvidenceIntegrityError(
                        f"Evidence semantic 논리 좌표 오류: {evidence_key}")
            try:
                candidate = make_evidence(
                    build_id=self.build_id, doc_id=doc_id,
                    source_file_id=source_file_id, kind=kind, locator=locator,
                    table_locator=table_locator, logical_row=logical_row,
                    logical_col=logical_col, excerpt_raw=raw,
                    extraction_method=method, extraction_status="ok",
                    rcept_dt=str(rcept_dt), corporate_names=names,
                    pii_label=pii_label,
                    pii_subject_name=self._pii_subject(
                        doc_id, pii_label, table_locator, logical_row),
                    pii_party_type=self._pii_party_type(
                        doc_id, pii_label, raw, table_locator, logical_row,
                        kind=kind),
                    security_policy_version=stored_policy)
            except (CanonicalLocatorError, TypeError, ValueError) as exc:
                raise EvidenceIntegrityError(
                    f"Evidence semantic 재생성 실패: {evidence_key}") from exc
            if candidate.evidence_id != evidence_key:
                raise EvidenceIntegrityError(
                    f"Evidence semantic FK 불일치: {evidence_key}")
            if kind == "chunk_text":
                chunk_flags = tuple(ref.get("security_flags") or ())
                chunk_pii = tuple(ref.get("pii_types") or ())
                ref_policy = str(ref.get("security_policy_version") or "")
                if (len(chunk_flags) != len(set(chunk_flags))
                        or ref_policy not in (
                            {SECURITY_POLICY_VERSION}
                            | _READ_COMPAT_SECURITY_POLICIES)
                        or set(chunk_flags) != set(candidate.security_flags)
                        or len(chunk_pii) != len(set(chunk_pii))
                        or set(chunk_pii) != set(candidate.pii_types)
                        or ref.get("text_prompt_safe")
                        != candidate.excerpt_safe):
                    raise EvidenceIntegrityError(
                        f"Chunk Evidence safe/policy 재생성 불일치: {evidence_key}")
            return candidate

        comparable = (
            "build_id", "evidence_id", "doc_id", "source_file_id", "kind",
            "locator", "table_locator", "logical_row", "logical_col",
            "excerpt_raw", "excerpt_safe", "excerpt_hash", "raw_access",
            "extraction_method", "extraction_status", "security_flags",
            "pii_types", "security_policy_version", "evidence_policy_version",
            "rcept_dt",
        )
        for evidence_key in unresolved:
            stored = stored_by_id.get(evidence_key)
            if stored is None:
                results[evidence_key] = None
                continue
            kind = stored.get("kind")
            if expected_kind is not None and kind != expected_kind:
                raise EvidenceIntegrityError(
                    f"Evidence kind 불일치: {evidence_key} ({kind} != {expected_kind})")
            entries = semantic.get(evidence_key, ())
            if not entries:
                raise EvidenceIntegrityError(
                    f"Evidence semantic FK가 없는 orphan입니다: {evidence_key}")
            referenced_kinds = {item[0] for item in entries}
            if referenced_kinds != {kind}:
                raise EvidenceIntegrityError(
                    "Evidence cross-kind 참조입니다: "
                    f"{evidence_key} ({sorted(referenced_kinds)!r} vs {kind!r})")

            try:
                stored_locator = canonical_locator(stored.get("locator"))
            except (CanonicalLocatorError, TypeError) as exc:
                raise EvidenceIntegrityError(
                    f"Evidence locator 오류: {evidence_key}") from exc
            stored_table = stored.get("table_locator")
            if kind == "chunk_text":
                if (stored_table is not None or stored.get("logical_row") is not None
                        or stored.get("logical_col") is not None):
                    raise EvidenceIntegrityError(
                        f"chunk_text Evidence 좌표는 null이어야 합니다: {evidence_key}")
            else:
                if not isinstance(stored_table, str):
                    raise EvidenceIntegrityError(
                        f"table Evidence table_locator 누락: {evidence_key}")
                try:
                    stored_table = canonical_locator(stored_table)
                except CanonicalLocatorError as exc:
                    raise EvidenceIntegrityError(
                        f"Evidence table_locator 오류: {evidence_key}") from exc
                if not (stored_locator == stored_table
                        or stored_locator.startswith(stored_table + "/")):
                    raise EvidenceIntegrityError(
                        f"Evidence table ancestry 오류: {evidence_key}")
            raw, safe = stored.get("excerpt_raw"), stored.get("excerpt_safe")
            if not isinstance(raw, str) or not isinstance(safe, str):
                raise EvidenceIntegrityError(
                    f"Evidence raw/safe 타입 오류: {evidence_key}")
            raw_hash = calculate_excerpt_hash(raw)
            if stored.get("excerpt_hash") != raw_hash:
                raise EvidenceIntegrityError(
                    f"Evidence excerpt_hash 불일치: {evidence_key}")
            calculated_id = calculate_evidence_id(
                str(stored.get("source_file_id")), str(kind), stored_locator, raw_hash)
            if calculated_id != evidence_key or stored.get("build_id") != self.build_id:
                raise EvidenceIntegrityError(
                    f"Evidence ID/build 재계산 불일치: {evidence_key}")
            if stored.get("raw_access") != "restricted":
                raise EvidenceIntegrityError(
                    f"Evidence raw_access 오류: {evidence_key}")
            if (safe.count(PROMPT_DATA_BEGIN), safe.count(PROMPT_DATA_END)) != (1, 1):
                raise EvidenceIntegrityError(
                    f"Evidence safe boundary 오류: {evidence_key}")

            candidates = [recreate(*entry) for entry in entries]
            first = candidates[0]
            for index, candidate in enumerate(candidates):
                mismatches = [
                    name for name in comparable
                    if getattr(candidate, name) != stored.get(name)
                ]
                if mismatches:
                    raise EvidenceIntegrityError(
                        "Evidence semantic 재생성 불일치"
                        f"[{index}]: {evidence_key} ({', '.join(mismatches)})")
                if candidate != first:
                    raise EvidenceIntegrityError(
                        f"Evidence 복수 semantic 참조가 상충합니다: {evidence_key}")
            row = EvidenceRow(
                evidence_id=first.evidence_id, doc_id=first.doc_id,
                source_file_id=first.source_file_id, kind=first.kind,
                locator=first.locator, table_locator=first.table_locator,
                logical_row=first.logical_row, logical_col=first.logical_col,
                excerpt=first.excerpt_safe, excerpt_safe=first.excerpt_safe,
                excerpt_raw=(first.excerpt_raw if include_restricted_raw else None),
                restricted_raw_included=include_restricted_raw,
                excerpt_hash=first.excerpt_hash, raw_access=first.raw_access,
                extraction_method=first.extraction_method,
                extraction_status=first.extraction_status,
                security_flags=tuple(first.security_flags),
                pii_types=tuple(first.pii_types),
                security_policy_version=first.security_policy_version,
                evidence_policy_version=first.evidence_policy_version,
                rcept_dt=first.rcept_dt,
            )
            if not include_restricted_raw:
                self._cache_evidence(row)
            results[evidence_key] = row
        return {key: results.get(key) for key in ordered}

    def get_evidence(
            self, evidence_id: str, include_restricted_raw: bool = False,
            ) -> EvidenceRow | None:
        """단일 Evidence 조회도 batch resolver의 exact recreation 경로를 공유한다."""
        result = self.get_evidence_many(
            [evidence_id], include_restricted_raw=include_restricted_raw)
        return result[str(evidence_id)]

    def _evidence_fk_status(
            self, fk: str | None, *, kind: str, doc_id: str,
            source_file_id: str | None, locator: str, rcept_dt: str,
            table_locator: str | None, logical_row: int | None,
            logical_col: int | None, expected_safe: str | None = None) -> str:
        """semantic row → Evidence FK와 source coordinates를 함께 검증한다."""
        if not fk:
            return "unavailable"
        try:
            evidence = self.get_evidence(fk)
        except NotReadyError:
            return "unavailable"
        except EvidenceIntegrityError:
            return "invalid"
        if evidence is None:
            return "invalid"
        matches = (
            evidence.kind == kind
            and evidence.doc_id == doc_id
            and evidence.source_file_id == source_file_id
            and evidence.locator == locator
            and evidence.rcept_dt == rcept_dt
            and (table_locator is None or evidence.table_locator == table_locator)
            and (logical_row is None or evidence.logical_row == logical_row)
            and (logical_col is None or evidence.logical_col == logical_col)
            and (expected_safe is None or evidence.excerpt_safe == expected_safe)
        )
        return "verified" if matches else "invalid"

    def _verified_fact(self, row: FactRow) -> FactRow:
        """선택된 Fact의 Evidence를 검증한다. 실패는 값 삭제가 아니라 인용 불가 상태다."""
        if _schema_at_least(self.schema_version, (1, 8)):
            coordinate_ok = (
                isinstance(row.evidence_id, str) and bool(row.evidence_id)
                and isinstance(row.source_file_id, str) and bool(row.source_file_id)
                and isinstance(row.locator, str) and bool(row.locator)
                and isinstance(row.table_locator, str) and bool(row.table_locator)
                and type(row.logical_row) is int and row.logical_row >= 0
                and type(row.logical_col) is int and row.logical_col >= 0)
            try:
                ancestry_ok = table_locator_of(row.locator) == row.table_locator
            except CanonicalLocatorError:
                ancestry_ok = False
            if not coordinate_ok or not ancestry_ok:
                return replace(row, evidence_status="invalid")
        expected = project_chunk_text(str(row.money.text or "")).text_prompt_safe
        status = self._evidence_fk_status(
            row.evidence_id, kind="fact_value", doc_id=row.doc_id,
            source_file_id=row.source_file_id, locator=row.locator,
            rcept_dt=row.rcept_dt, table_locator=row.table_locator,
            logical_row=row.logical_row, logical_col=row.logical_col,
            expected_safe=expected)
        return replace(row, evidence_status=status)

    def verify_fact_evidence(self, row: FactRow) -> FactRow:
        """Public exact-Evidence verification for an already selected candidate.

        ``lookup`` verifies only its authoritative row to keep the common path
        cheap.  A caller that intentionally retains an older original or
        correction as supplementary comparison provenance must pass through
        the same semantic FK, source-coordinate and prompt-safe checks instead
        of treating an unverified candidate as a citation.
        """

        return self._verified_fact(row)

    def _verified_field(self, row: FieldRow) -> FieldRow:
        if (not _schema_at_least(self.schema_version, (1, 8))
                or row.table_locator is None
                or row.logical_row is None or row.logical_col is None):
            # schema <=1.7은 locator의 물리 TR/TD 순번으로 Evidence를 만들었다.
            # None 비교 생략을 통해 verified가 되는 우회를 명시적으로 닫는다.
            return replace(row, evidence_status="invalid")
        expected = row.value_prompt_safe
        status = self._evidence_fk_status(
            row.evidence_id, kind="field_value", doc_id=row.doc_id,
            source_file_id=row.source_file_id, locator=row.locator,
            rcept_dt=row.rcept_dt, table_locator=row.table_locator,
            logical_row=row.logical_row, logical_col=row.logical_col,
            expected_safe=expected)
        return replace(row, evidence_status=status)

    # ---------------------------------------------------------------- 재무
    def facts(self, corp_code: str, *, as_of: str, concept: str | None = None,
              scope: str | None = None, statement: str | None = None,
              period_end: str | None = None, cumulative: bool | None = None,
              ) -> list[FactRow]:
        """조건에 맞는 Fact 전부. **`as_of` 이후 접수 문서는 보이지 않는다.**"""
        cutoff = _checked_as_of(as_of)
        rows = self._load_facts(corp_code)
        out = []
        for f in rows:
            if f.rcept_dt > cutoff:
                continue
            if concept is not None and f.account_norm != concept:
                continue
            if scope is not None and f.scope != scope:
                continue
            if statement is not None and f.statement != statement:
                continue
            if period_end is not None and f.period_end != period_end:
                continue
            if cumulative is not None and f.cumulative != cumulative:
                continue
            out.append(f)
        return out

    def lookup(self, corp_code: str, concept: str, period_end: str, *, as_of: str,
               scope: str = "CFS", view: View = "restated",
               cumulative: "bool | None | _Unset" = UNSET,
               statement: "str | None | _Unset" = UNSET,
               account_path: "str | None | _Unset" = UNSET,
               period_start: "str | None | _Unset" = UNSET,
               unit: "str | None | _Unset" = UNSET) -> FactLookup:
        """`fact()` 의 **이유가 붙은** 판. 답변 층은 이쪽을 쓴다.

        `fact()` 는 「없음」과 「모호」를 똑같이 `None` 으로 준다. 답변 층이 둘을
        구분하지 못하면 `facts()` 로 내려가 임의로 고르게 되고, 막으려던 문제가
        한 층 아래로 옮겨갈 뿐이다. 그래서 **왜 못 골랐는지와 무엇을 더 주면 되는지**를
        함께 준다.
        """
        cutoff = _checked_as_of(as_of)
        found = self.facts(corp_code, as_of=cutoff, concept=concept, scope=scope,
                           period_end=period_end,
                           cumulative=None if cumulative is UNSET else cumulative,
                           statement=None if statement is UNSET else statement)
        # `facts()` 는 `None` 을 「안 좁힘」으로 쓴다. `lookup()` 은 둘을 가르므로
        # **`None` 으로 지정한 경우는 여기서 다시 좁힌다.**
        if cumulative is None:
            found = [f for f in found if f.cumulative is None]
        if statement is None:
            found = [f for f in found if f.statement is None]
        # **알려준 축은 실제로 받을 수 있어야 한다** (5차 검수).
        if account_path is not UNSET:
            key = _path_key(account_path) if account_path is not None else None
            found = [f for f in found
                     if (_path_key(f.account_path) if f.account_path else None) == key]
        if period_start is not UNSET:
            found = [f for f in found if f.period_start == period_start]
        if unit is not UNSET:
            found = [f for f in found if f.money.unit == unit]

        # **못 읽은 문서는 값을 찾았든 못 찾았든 알린다** (6차 검수). 2025 보고서가
        # 미지원인데 2024 보고서에 같은 기간 값이 있으면 `ok` 가 나온다 — 틀린 답은
        # 아니지만 **최신 재작성본을 못 본 채 고른 것**이라 호출부가 알아야 한다.
        # **어느 표를 묻는지 알아야 비교기간이 정해진다.** 후보의 표가 하나로 좁혀지면
        # 그것을 쓰고, 아니면 호출부가 준 축을, 둘 다 없으면 넓게 잡는다 (9차 검수).
        _st = {f.statement for f in found} if found else set()
        eff_statement = (next(iter(_st)) if len(_st) == 1
                         else (None if statement is UNSET else statement))
        unread = self._unread_docs(corp_code, period_end, cutoff, scope, view,
                                   eff_statement)
        if not found:
            # **「원문에 없다」와 「우리가 못 읽는다」는 다르다.**
            if unread:
                return FactLookup("extract_unsupported", None, (), (), unread, view)
            return FactLookup("not_found", None, (), (), (), view)
        paths = {_path_key(f.account_path) for f in found}
        if len(paths) > 1:
            return FactLookup("ambiguous_account_path", None, tuple(found),
                              ("account_path",), unread, view)
        # **축을 지정하면 풀리는 것부터 본다.** 순서가 반대면 「문서 내 충돌」처럼 보여
        # 답변 층이 무엇을 더 주면 되는지 알 수 없다.
        for dim, status in (("statement", "ambiguous_statement"),
                            ("cumulative", "ambiguous_period_role"),
                            ("period_start", "ambiguous_period_role")):
            if len({getattr(f, dim) for f in found}) > 1:
                return FactLookup(status, None, tuple(found), (dim,), unread, view)
        if len({f.money.unit for f in found}) > 1:
            return FactLookup("ambiguous_unit", None, tuple(found), ("unit",), unread, view)
        # 여기까지 왔는데 값이 갈리면 **원문이 같은 계정을 두 줄로 적은 것**이다
        per_doc: dict[str, set] = {}
        for f in found:
            per_doc.setdefault(f.doc_id, set()).add(f.money.value)
        if any(len(v) > 1 for v in per_doc.values()):
            return FactLookup("same_document_conflict", None, tuple(found), (), unread, view)
        ordered = sorted(found, key=lambda f: (f.rcept_dt, f.doc_id))
        selected_index = len(ordered) - 1 if view == "restated" else 0
        selected = self._verified_fact(ordered[selected_index])
        ordered[selected_index] = selected
        return FactLookup("ok", selected, tuple(ordered), (), unread, view)

    def fact(self, corp_code: str, concept: str, period_end: str, *, as_of: str,
             scope: str = "CFS", view: View = "restated",
             cumulative: bool | None = None,
             statement: str | None = None,
             allow_partial_coverage: bool = False) -> FactRow | None:
        """값 하나. **여러 보고서가 같은 값을 보고하면 `view` 가 고른다.**

        **근거를 다 못 본 경우도 `None` 이다.** 예전에는 이쪽으로 물으면
        `unread_documents` 경고가 통째로 사라져, 못 읽은 문서가 있어도 값이
        그냥 나왔다 (7차 검수). 그걸 알고도 받겠다면 `allow_partial_coverage=True`
        를 **명시**해야 한다 — 기본값으로 우회되지 않는다.

        `restated` 는 가장 나중 문서, `as_filed` 는 가장 이른 문서다.
        후보가 여러 계정 경로에 걸치거나 한 문서 안에서 값이 갈리면 `None` 을 준다 —
        **임의로 고르지 않는다.**

        `statement` 로 재무제표를 좁힐 수 있다. 개념 하나가 여러 표에 걸치는 일이 있다 —
        삼성전자 `owners_of_parent` 2023-09-30 은 포괄손익계산서(7,089,926)와
        손익계산서(5,501,304)에 **둘 다** 있고 서로 다른 개념이다.
        좁히지 않으면 모호하므로 `None` 이 온다. 그때 호출부가 `statement="IS"` 로 고른다.
        """
        # **판정은 `lookup()` 한 곳에서만 한다.** 예전에는 같은 규칙을 두 번 적었고,
        # `lookup` 에 넣은 근거-범위 경고가 이쪽에는 없었다 (7차 검수).
        allow_partial_coverage = _checked_bool_flag(
            "allow_partial_coverage", allow_partial_coverage)
        cutoff = _checked_as_of(as_of)
        got = self.lookup(
            corp_code, concept, period_end, as_of=cutoff, scope=scope, view=view,
            cumulative=UNSET if cumulative is None else cumulative,
            statement=UNSET if statement is None else statement)
        if got.status != "ok":
            return None
        coverage = got.coverage_status
        if coverage == "complete":
            return got.selected
        # 이 플래그는 읽지 못한 문서가 있는 ``partial_unread``만 수용한다.
        # Evidence가 없거나 무결성 검증에 실패한 ``evidence_unavailable``을
        # 같은 플래그로 열면 근거 없는 숫자가 확정 답으로 승격한다.
        if coverage == "partial_unread" and allow_partial_coverage:
            return got.selected
        return None

    # ------------------------------------------------------------ 비정기 Field
    @staticmethod
    def _matches_document(
            meta: _DocumentInfo, *, as_of: str, corp_code: str | None,
            corp_name: str | None, doc_group: str | None, event_type: str | None,
            form: str | None, doc_id: str | None, rcept_no: str | None,
            is_correction: bool | None,
            rcept_nos: frozenset[str] | None = None) -> bool:
        if meta.rcept_dt > as_of:
            return False
        if corp_code is not None and meta.corp_code != corp_code:
            return False
        if corp_name is not None and _text_key(meta.corp_name) != _text_key(corp_name):
            return False
        if doc_group is not None and meta.doc_group != doc_group:
            return False
        if (event_type is not None
                and _text_key(meta.event_type) != _text_key(event_type)):
            return False
        if form is not None and not meta.matches_form(form):
            return False
        if doc_id is not None and meta.doc_id != doc_id:
            return False
        if rcept_no is not None and meta.rcept_no != rcept_no:
            return False
        if rcept_nos is not None and meta.rcept_no not in rcept_nos:
            return False
        if rcept_nos is not None and meta.rcept_no not in rcept_nos:
            return False
        if is_correction is not None and meta.is_correction != is_correction:
            return False
        return True

    # Public receipt batching is capped at 128 by the holding resolver.  Keep
    # one full batch resident so a subsequent verified ``fields`` scan does
    # not evict the first documents while it is still walking the same batch.
    _SUBJECT_CACHE_LIMIT = 128

    def _is_holding_doc(self, doc_id: str) -> bool:
        """적재가 `구분` 지도를 만든 문서군인가.

        적재(`src/canonical/build.py`)는 `record["doc_group"] == "holding"`
        일 때만 지도를 만든다. 읽기가 다른 기준으로 물으면 저장된 마스킹과
        재검증이 어긋나므로 같은 값을 본다.
        """

        self._load_docs()
        meta = self._doc_meta.get(doc_id)
        return bool(meta) and meta[2] == "holding"

    def prime_holding_field_contexts(
            self, *, as_of: str, rcept_nos,
            ) -> int:
        """Prime holding masking contexts for a bounded receipt batch.

        ``fields()`` revalidates every stored safe projection against the raw
        source.  Holding names and occupations require two document-local
        maps for that check.  Building either map one receipt at a time causes
        a full Parquet scan per filing; this method builds both maps in one
        receipt-pushdown scan without returning raw values or weakening the
        projection check.
        """

        cutoff = _checked_as_of(as_of)
        wanted = _checked_rcept_nos(rcept_nos)
        assert wanted is not None
        if len(wanted) > self._SUBJECT_CACHE_LIMIT:
            raise ValueError(
                "holding security context batch가 cache 상한을 넘습니다")
        self._load_docs()
        doc_ids = [
            meta.doc_id for meta in self._documents.values()
            if self._matches_document(
                meta, as_of=cutoff, corp_code=None, corp_name=None,
                doc_group="holding", event_type=None, form=None, doc_id=None,
                rcept_no=None, is_correction=None, rcept_nos=wanted)
        ]
        if not doc_ids:
            return 0
        artifact = self.root / "fields.parquet"
        columns = _present_columns(
            artifact, ["doc_id", "path", "acode", "aunit", "value_raw",
                       "table_locator", "logical_row"])
        grouped: dict[str, list[dict[str, object]]] = {
            doc_id: [] for doc_id in doc_ids}
        for row in _rows(
                artifact, columns, batch=50_000,
                filters=[("doc_id", "in", doc_ids)]):
            doc_id = str(row.get("doc_id") or "")
            if doc_id in grouped:
                grouped[doc_id].append(row)
        for doc_id in doc_ids:
            rows = grouped[doc_id]
            party_types = holding_party_types_by_row(
                (row.get("path"), row.get("aunit"), row.get("value_raw"),
                 row.get("table_locator"), row.get("logical_row"))
                for row in rows)
            subjects = subject_names_by_row(
                (row.get("acode"), row.get("value_raw"),
                 row.get("table_locator"), row.get("logical_row"))
                for row in rows)
            self._party_type_cache[doc_id] = party_types
            self._party_type_cache.move_to_end(doc_id)
            self._subject_cache[doc_id] = subjects
            self._subject_cache.move_to_end(doc_id)
        while len(self._party_type_cache) > self._SUBJECT_CACHE_LIMIT:
            self._party_type_cache.popitem(last=False)
        while len(self._subject_cache) > self._SUBJECT_CACHE_LIMIT:
            self._subject_cache.popitem(last=False)
        return len(doc_ids)

    def _pii_party_type(self, doc_id, label, value_raw,
                        table_locator, logical_row, *,
                        kind: str = "field_value"):
        """근거·필드 재투영에 쓰는 `구분`. 적재가 쓴 것과 같아야 한다 (#199).

        적재(`src/canonical/build.py`)는 Field 행에만 `구분`을 준다 — 정정
        항목과 chunk 의 `make_evidence` 는 받지 않는다. 읽기가 더 주면
        재생성한 근거가 저장된 것과 어긋나 조회가 닫힌다.
        """

        if kind != "field_value":
            return None
        if not doc_id or label is None or not needs_party_type(str(label)):
            return None
        pairing = self._party_types(str(doc_id))
        if pairing is None:
            return None
        return resolve_party_type(
            pairing, str(label), str(value_raw or ""), table_locator, logical_row)

    def _party_types(self, doc_id: str):
        """문서 한 건의 `구분` 지도. 지분공시가 아니면 ``None``.

        적재가 저장된 마스킹을 만들 때 쓴 것과 **같은 규칙**
        (`holding_party_types_by_row`)을 쓴다. 어긋나면 `성명(명칭)` 행의
        재검증이 실패한다. 문서 단위로 만들어 최근 것만 들고 있는다 —
        조회는 보통 한 문서 안에서 연달아 일어난다.
        """

        cached = self._party_type_cache.get(doc_id)
        if cached is not None:
            self._party_type_cache.move_to_end(doc_id)
            return cached if cached is not _NO_PARTY_TYPES else None
        if not self._is_holding_doc(doc_id):
            found: object = _NO_PARTY_TYPES
        else:
            artifact = self.root / "fields.parquet"
            columns = _present_columns(
                artifact, ["doc_id", "path", "aunit", "value_raw",
                           "table_locator", "logical_row"])
            rows = (
                (r.get("path"), r.get("aunit"), r.get("value_raw"),
                 r.get("table_locator"), r.get("logical_row"))
                for r in _rows(artifact, columns, batch=50_000,
                               filters=[("doc_id", "==", doc_id)])
            )
            found = holding_party_types_by_row(rows)
        self._party_type_cache[doc_id] = found
        self._party_type_cache.move_to_end(doc_id)
        while len(self._party_type_cache) > self._SUBJECT_CACHE_LIMIT:
            self._party_type_cache.popitem(last=False)
        return found if found is not _NO_PARTY_TYPES else None

    def _pii_subject(self, doc_id, label, table_locator, logical_row):
        """근거 재생성에 쓰는 주체. 적재가 쓴 것과 같아야 한다 (이슈 #139)."""

        if not doc_id or label is None or not needs_subject_name(str(label)):
            return None
        return self._subject_names(str(doc_id)).get((table_locator, logical_row))

    def _subject_names(self, doc_id: str) -> dict[tuple[object, object], str]:
        """문서 한 건의 ``(table_locator, logical_row) -> 성명(명칭)``.

        적재(`src/canonical/build.py`)가 저장된 마스킹을 만들 때 쓴 것과 **같은
        규칙**(`subject_names_by_row`)을 쓴다. 어긋나면 `직 업(사업내용)` 행의
        재검증이 실패한다 (이슈 #139).

        지분공시 문서 하나가 수백~수천 행이라 매 행마다 다시 읽으면 느리다.
        문서 단위로 만들어 최근 것만 들고 있는다 — 조회는 보통 한 문서 안에서
        연달아 일어난다.
        """

        cached = self._subject_cache.get(doc_id)
        if cached is not None:
            self._subject_cache.move_to_end(doc_id)
            return cached
        artifact = self.root / "fields.parquet"
        columns = _present_columns(
            artifact, ["doc_id", "acode", "value_raw",
                       "table_locator", "logical_row"])
        rows = (
            (r.get("acode"), r.get("value_raw"),
             r.get("table_locator"), r.get("logical_row"))
            for r in _rows(artifact, columns, batch=50_000,
                           filters=[("doc_id", "==", doc_id)])
        )
        found = subject_names_by_row(rows)
        self._subject_cache[doc_id] = found
        self._subject_cache.move_to_end(doc_id)
        while len(self._subject_cache) > self._SUBJECT_CACHE_LIMIT:
            self._subject_cache.popitem(last=False)
        return found

    def fields(
            self, *, as_of: str, corp_code: str | None = None,
            corp_name: str | None = None, doc_group: str | None = None,
            event_type: str | None = None, form: str | None = None,
            doc_id: str | None = None, rcept_no: str | None = None,
            rcept_nos=None,
            label: str | None = None, path: str | None = None,
            path_contains: str | None = None, value_status: str | None = None,
            acode: str | None = None, aunit: str | None = None,
            aunitvalue: str | None = None, source_file_id: str | None = None,
            locator: str | None = None, occurrence: int | None = None,
            is_correction: bool | None = None, is_pii: bool | None = None,
            include_restricted_raw: bool = False,
            ) -> Iterator[FieldRow]:
        """조건에 맞는 Field를 스트리밍한다.

        ``label``은 마지막 path 조각에 대한 포함 검색이고, ``path``는 전체 경로 정확
        검색이다. 포함 검색은 후보 생성용일 뿐이다. 값 하나가 필요하면 반드시
        :meth:`lookup_field`를 사용해 모호성 상태를 확인한다.

        ``form``은 문서군 차이를 흡수한 서식명이다. exchange/holding에서는 주로
        ``doc_subtype``, major에서는 ``event_type``이 된다. ``event_type``은 원본
        Document 컬럼을 그대로 정확 비교한다.
        """
        cutoff = _checked_as_of(as_of)
        wanted_rcept_nos = _checked_rcept_nos(rcept_nos)
        include_restricted_raw = _checked_bool_flag(
            "include_restricted_raw", include_restricted_raw)
        self._load_docs()
        artifact = self.root / "fields.parquet"
        if not artifact.exists():
            raise NotReadyError(f"Field 산출물이 없습니다: {artifact}")

        wanted_columns = [
            "doc_id", "source_file_id", "block_id", "parent_id", "path", "order",
            "locator", "evidence_id", "value_raw", "value_masked", "value_prompt_safe", "pii_type",
            "security_policy_version", "value_status", "acode", "aunit", "aunitvalue",
            "is_pii", "table_locator", "logical_row", "logical_col",
            "label_locators", "occurrence", "rcept_dt",
        ]
        present = _artifact_columns(artifact)
        semantic_columns = {"logical_row", "logical_col"}
        modern_required = semantic_columns | {"value_prompt_safe"}
        missing_modern = sorted(modern_required - present)
        if _schema_at_least(self.schema_version, (1, 8)) and missing_modern:
            raise ArtifactIntegrityError(self.root, [ArtifactIssue(
                "invalid_parquet",
                "schema 1.8+ Field semantic/safe 컬럼 누락: "
                + ", ".join(missing_modern), "fields")])
        columns = _present_columns(artifact, wanted_columns)
        has_semantic_coordinates = semantic_columns.issubset(present)
        modern_semantics = _schema_at_least(self.schema_version, (1, 8)) and has_semantic_coordinates
        want_label = _text_key(label) if label is not None else None
        want_path = _text_key(path) if path is not None else None
        want_path_part = (_text_key(path_contains)
                          if path_contains is not None else None)
        # Document metadata is only 4K rows and is already resident.  When the
        # public selector narrows it to a bounded set, push those doc_ids into
        # Parquet instead of decoding every field row and discarding almost all
        # of them in Python.  Broad scans keep the original streaming path.
        has_document_constraint = any(value is not None for value in (
            corp_code, corp_name, doc_group, event_type, form, doc_id,
            rcept_no, is_correction, wanted_rcept_nos,
        ))
        selected_doc_ids: list[str] | None = None
        if has_document_constraint:
            selected_doc_ids = [
                meta.doc_id
                for meta in self._documents.values()
                if self._matches_document(
                    meta, as_of=cutoff, corp_code=corp_code,
                    corp_name=corp_name, doc_group=doc_group,
                    event_type=event_type, form=form, doc_id=doc_id,
                    rcept_no=rcept_no, is_correction=is_correction,
                    rcept_nos=wanted_rcept_nos,
                )
            ]
            if not selected_doc_ids:
                return
        parquet_filters = (
            [("doc_id", "in", selected_doc_ids)]
            if selected_doc_ids is not None and len(selected_doc_ids) <= 2048
            else None
        )
        for r in _rows(
                artifact, columns, batch=50_000, filters=parquet_filters):
            meta = self._documents.get(r["doc_id"])
            if meta is None or not self._matches_document(
                    meta, as_of=cutoff, corp_code=corp_code, corp_name=corp_name,
                    doc_group=doc_group, event_type=event_type, form=form,
                    doc_id=doc_id, rcept_no=rcept_no, is_correction=is_correction,
                    rcept_nos=wanted_rcept_nos):
                continue
            field_dt = str(r["rcept_dt"] or meta.rcept_dt)
            if field_dt > cutoff:              # 중복 시점 컬럼도 함께 잠근다
                continue
            field_path = str(r["path"] or "")
            if (want_label is not None
                    and want_label not in _text_key(_field_label(field_path))):
                continue
            if want_path is not None and _text_key(field_path) != want_path:
                continue
            if (want_path_part is not None
                    and want_path_part not in _text_key(field_path)):
                continue
            if value_status is not None and r["value_status"] != value_status:
                continue
            if acode is not None and r["acode"] != acode:
                continue
            if aunit is not None and r["aunit"] != aunit:
                continue
            if aunitvalue is not None and r["aunitvalue"] != aunitvalue:
                continue
            if source_file_id is not None and r["source_file_id"] != source_file_id:
                continue
            if locator is not None and r["locator"] != locator:
                continue
            if occurrence is not None and r["occurrence"] != occurrence:
                continue
            if is_pii is not None and bool(r["is_pii"]) != is_pii:
                continue
            raw = r.get("value_raw")
            if modern_semantics:
                masked = r.get("value_masked")
                if not isinstance(raw, str):
                    raise ReadSecurityError(
                        f"Field value_raw가 문자열이 아닙니다: {r['block_id']}")
                prompt_safe = r.get("value_prompt_safe")
                if (not isinstance(masked, str)
                        or not isinstance(prompt_safe, str)):
                    raise ReadSecurityError(
                        f"Field 저장 safe projection이 없습니다: {r['block_id']}")
                pii_type = r.get("pii_type")
                policy = str(r.get("security_policy_version") or "")
                if not policy:
                    raise ReadSecurityError(
                        f"Field security_policy_version이 없습니다: {r['block_id']}")
                # 주체 조회는 원문 행을 되짚으므로 필요한 라벨에서만 한다.
                subject = (
                    self._subject_names(r["doc_id"]).get(
                        (r.get("table_locator"), r.get("logical_row")))
                    if needs_subject_name(field_path) else None)
                party_type = self._pii_party_type(
                    r["doc_id"], field_path, raw,
                    r.get("table_locator"), r.get("logical_row"))
                expected_field = project_field_value(
                    field_path, raw, (meta.corp_name, meta.listed_name),
                    subject_name=subject, party_type=party_type,
                    security_policy_version=policy)
                expected_prompt = project_chunk_text(
                    expected_field.value_masked,
                    (meta.corp_name, meta.listed_name),
                    security_policy_version=policy).text_prompt_safe
                if policy not in ({SECURITY_POLICY_VERSION}
                                  | _READ_COMPAT_SECURITY_POLICIES):
                    raise ReadSecurityError(
                        f"지원하지 않는 Field security policy: {policy}")
                if (masked != expected_field.value_masked
                        or prompt_safe != expected_prompt
                        or pii_type != expected_field.pii_type
                        or bool(r.get("is_pii")) is not expected_field.is_pii):
                    raise ReadSecurityError(
                        f"Field 저장 safe projection 재검증 실패: {r['block_id']}")
            else:
                # schema <=1.7에 물리 prompt-safe 컬럼은 없다. raw를 그대로 반환하지 않고
                # 현재 정책으로 재투영하되 Evidence citation은 legacy라 invalid로 닫는다.
                projection = project_field_value(
                    field_path, str(raw or ""),
                    (meta.corp_name, meta.listed_name))
                masked = projection.value_masked
                pii_type = projection.pii_type
                policy = projection.security_policy_version
                prompt_safe = project_chunk_text(
                    masked, (meta.corp_name, meta.listed_name)).text_prompt_safe
            prefix = PROMPT_DATA_BEGIN + "\n"
            suffix = "\n" + PROMPT_DATA_END
            if not (prompt_safe.startswith(prefix) and prompt_safe.endswith(suffix)):
                raise ReadSecurityError(
                    f"Field prompt-safe 경계가 손상되었습니다: {r['block_id']}")
            scalar_safe = prompt_safe[len(prefix):-len(suffix)]
            try:
                canonical_locator(r["locator"])
            except CanonicalLocatorError as exc:
                raise ReadSecurityError(
                    f"Field locator가 canonical하지 않습니다: {r['locator']!r}") from exc
            if modern_semantics:
                logical_row, logical_col = r.get("logical_row"), r.get("logical_col")
                if (type(logical_row) is not int or type(logical_col) is not int
                        or logical_row < 0 or logical_col < 0):
                    raise ArtifactIntegrityError(self.root, [ArtifactIssue(
                        "invalid_parquet",
                        f"Field semantic 좌표 오류: block_id={r['block_id']}",
                        "fields")])
                try:
                    stored_table = canonical_locator(r.get("table_locator"))
                except CanonicalLocatorError as exc:
                    raise ArtifactIntegrityError(self.root, [ArtifactIssue(
                        "invalid_parquet", "Field table_locator 없음/오류",
                        "fields")]) from exc
                if table_locator_of(r["locator"]) != stored_table:
                    raise ArtifactIntegrityError(self.root, [ArtifactIssue(
                        "invalid_parquet", "Field table ancestry 불일치", "fields")])
            else:
                # schema <=1.7 Evidence 좌표는 locator의 물리 TR/TD 순번이었다.
                # 탐색 후보는 유지하지만 semantic 좌표/검증 인용으로 승격하지 않는다.
                logical_row = logical_col = None
            yield FieldRow(
                doc_id=meta.doc_id, rcept_no=meta.rcept_no, rcept_dt=field_dt,
                corp_code=meta.corp_code, corp_name=meta.corp_name,
                doc_group=meta.doc_group, event_type=meta.event_type, form=meta.form,
                report_nm=meta.report_nm, is_correction=meta.is_correction,
                source_file_id=r["source_file_id"], block_id=r["block_id"],
                parent_id=r["parent_id"], path=field_path, order=int(r["order"]),
                locator=r["locator"], value_masked=scalar_safe,
                value_prompt_safe=prompt_safe,
                # 이전 호출부가 ``value_raw``를 읽더라도 기본 경로에서는
                # PII+active+prompt-safe scalar만 보인다.
                value_raw=raw if include_restricted_raw else scalar_safe,
                restricted_raw_included=include_restricted_raw,
                pii_type=pii_type, security_policy_version=policy,
                evidence_id=r.get("evidence_id"),
                evidence_status=("unverified" if r.get("evidence_id")
                                 else "unavailable"),
                value_status=r["value_status"], acode=r["acode"], aunit=r["aunit"],
                aunitvalue=r["aunitvalue"], is_pii=bool(r["is_pii"]),
                table_locator=r["table_locator"], logical_row=logical_row,
                logical_col=logical_col, label_locators=r["label_locators"],
                occurrence=int(r["occurrence"]))

    def table_headers(
            self, *, as_of: str, doc_id: str,
            ) -> Iterator[TableHeaderRow]:
        """Stream canonical header coordinates for one visible document.

        A caller must still decide its own closed ontology; this method only
        proves the labels and table/column coordinates that exist in the
        source.  Missing cells or any non-visible document produce no rows.
        """

        cutoff = _checked_as_of(as_of)
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("doc_id는 비어 있을 수 없습니다")
        self._load_docs()
        meta = self._documents.get(doc_id)
        if meta is None or meta.rcept_dt > cutoff:
            return
        artifact = self.root / "cells.parquet"
        if not artifact.exists():
            raise NotReadyError(f"Cell 산출물이 없습니다: {artifact}")
        columns = _present_columns(artifact, [
            "doc_id", "source_file_id", "parent_id", "locator", "role",
            "header_path",
        ])
        required = {"doc_id", "source_file_id", "parent_id", "locator",
                    "role", "header_path"}
        if required - set(columns):
            raise ArtifactIntegrityError(self.root, [ArtifactIssue(
                "invalid_parquet", "table header 필수 Cell 컬럼 누락", "cells")])
        rows: list[TableHeaderRow] = []
        for row in _rows(artifact, columns, batch=50_000):
            if row["doc_id"] != doc_id or row["role"] != "header":
                continue
            table_locator = str(row["parent_id"] or "")
            locator = str(row["locator"] or "")
            header_path = str(row["header_path"] or "").strip()
            source_file_id = str(row["source_file_id"] or "")
            if not table_locator or not locator or not header_path or not source_file_id:
                continue
            rows.append(TableHeaderRow(
                doc_id=doc_id, rcept_no=meta.rcept_no,
                corp_code=meta.corp_code, source_file_id=source_file_id,
                table_locator=table_locator, locator=locator,
                header_path=header_path,
            ))
        yield from sorted(rows, key=lambda row: (
            row.table_locator, row.source_file_id, row.locator))

    def lookup_field(
            self, *, as_of: str, corp_code: str | None = None,
            corp_name: str | None = None, doc_group: str | None = None,
            event_type: str | None = None, form: str | None = None,
            doc_id: str | None = None, rcept_no: str | None = None,
            rcept_nos=None,
            label: str | None = None, path: str | None = None,
            path_contains: str | None = None, value_status: str | None = None,
            acode: str | None = None, aunit: str | None = None,
            aunitvalue: str | None = None, source_file_id: str | None = None,
            locator: str | None = None, occurrence: int | None = None,
            is_correction: bool | None = None, is_pii: bool | None = None,
            include_restricted_raw: bool = False,
            ) -> FieldLookup:
        """Field 하나를 고른다. 0건과 복수 후보를 구분하고 복수면 닫는다."""
        candidates = tuple(self.fields(
            as_of=as_of, corp_code=corp_code, corp_name=corp_name,
            doc_group=doc_group, event_type=event_type, form=form, doc_id=doc_id,
            rcept_no=rcept_no, label=label, path=path, path_contains=path_contains,
            value_status=value_status, acode=acode, aunit=aunit,
            aunitvalue=aunitvalue, source_file_id=source_file_id, locator=locator,
            occurrence=occurrence, is_correction=is_correction, is_pii=is_pii,
            include_restricted_raw=include_restricted_raw))
        if not candidates:
            return FieldLookup("not_found", None)
        if len(candidates) == 1:
            selected = self._verified_field(candidates[0])
            return FieldLookup("ok", selected, (selected,))
        discriminators = []
        for name in ("rcept_no", "event_type", "form", "path", "acode",
                     "value_status", "source_file_id", "occurrence", "locator"):
            if len({getattr(row, name) for row in candidates}) > 1:
                discriminators.append(name)
        return FieldLookup("ambiguous", None, candidates, tuple(discriminators))

    # --------------------------------------------------------------- 정정 항목
    def correction_items(
            self, *, as_of: str, corp_code: str | None = None,
            corp_name: str | None = None, doc_group: str | None = None,
            event_type: str | None = None, form: str | None = None,
            doc_id: str | None = None, rcept_no: str | None = None,
            rcept_nos: Iterable[str] | None = None,
            path: str | None = None, path_contains: str | None = None,
            reason_contains: str | None = None, diff_kind: str | None = None,
            before_kind: str | None = None, after_kind: str | None = None,
            required_by_authority: bool | None = None,
            include_restricted_raw: bool = False,
            verify_evidence: bool = False,
            ) -> Iterator[CorrectionItemRow]:
        """기준시점까지 공개된 선언형 정정 항목을 typed row로 스트리밍한다."""
        cutoff = _checked_as_of(as_of)
        include_restricted_raw = _checked_bool_flag(
            "include_restricted_raw", include_restricted_raw)
        verify_evidence = _checked_bool_flag("verify_evidence", verify_evidence)
        wanted_receipts: tuple[str, ...] | None = None
        if rcept_nos is not None:
            if rcept_no is not None:
                raise ValueError("rcept_no와 rcept_nos는 함께 지정할 수 없습니다")
            values = tuple(rcept_nos)
            if (not values or len(values) > 128
                    or any(not isinstance(value, str)
                           or re.fullmatch(r"[0-9]{14}", value) is None
                           for value in values)
                    or len(values) != len(set(values))):
                raise ValueError("rcept_nos는 1~128개의 고유한 14자리 접수번호여야 합니다")
            wanted_receipts = tuple(sorted(values))
        self._load_docs()
        artifact = self.root / "correction_items.parquet"
        if not artifact.exists():
            raise NotReadyError(f"CorrectionItem 산출물이 없습니다: {artifact}")
        wanted_columns = [
            "doc_id", "source_file_id", "block_id", "parent_id", "path", "order",
            "locator", "rcept_no", "corp_name", "doc_group", "reason", "value_before",
            "value_after", "before_locator", "after_locator",
            "logical_row", "logical_col",
            "before_logical_row", "before_logical_col",
            "after_logical_row", "after_logical_col",
            "before_evidence_id", "after_evidence_id", "diff_kind",
            "before_kind", "after_kind", "required_by_authority", "rcept_dt",
        ]
        present = _artifact_columns(artifact)
        semantic_columns = {
            "logical_row", "logical_col",
            "before_logical_row", "before_logical_col",
            "after_logical_row", "after_logical_col",
        }
        required_modern = semantic_columns | {"before_locator", "after_locator"}
        missing_modern = sorted(required_modern - present)
        if _schema_at_least(self.schema_version, (1, 8)) and missing_modern:
            raise ArtifactIntegrityError(self.root, [ArtifactIssue(
                "invalid_parquet", "schema 1.8+ Correction 좌표 컬럼 누락: "
                + ", ".join(missing_modern), "correction_items")])
        columns = _present_columns(artifact, wanted_columns)
        modern_semantics = _schema_at_least(self.schema_version, (1, 8))
        want_path = _text_key(path) if path is not None else None
        want_path_part = (_text_key(path_contains)
                          if path_contains is not None else None)
        want_reason = (_text_key(reason_contains)
                       if reason_contains is not None else None)
        parquet_filters = None
        if rcept_no is not None:
            parquet_filters = [("rcept_no", "=", rcept_no)]
        elif wanted_receipts is not None:
            parquet_filters = [("rcept_no", "in", list(wanted_receipts))]
        elif doc_id is not None:
            parquet_filters = [("doc_id", "=", doc_id)]

        # CorrectionItem은 한 행의 before/after마다 Evidence를 확인한다. 각 확인이
        # ``get_evidence([id])``가 되면 한 계약의 여러 정정, 특히 collection 경로에서
        # Evidence/semantic artifact를 수십 번 다시 훑게 된다. 정확 receipt set은 128건으로
        # 이미 API에서 상한이 있으므로 FK만 먼저 모아 한 batch recreation을 수행한다.
        #
        # 이 prefetch는 결과를 신뢰하는 지름길이 아니다. 아래 side_status는 cache에서 꺼낸
        # Evidence에도 doc/source/locator/semantic 좌표를 다시 대조한다. Batch 자체가 손상된
        # FK 때문에 실패하면 기존 행별 fail-closed 판정을 그대로 수행한다.
        if verify_evidence and wanted_receipts is not None:
            evidence_ids: set[str] = set()
            for prefetched in _rows(
                    artifact,
                    ["before_evidence_id", "after_evidence_id"],
                    batch=20_000, filters=parquet_filters):
                for evidence_id in (
                        prefetched.get("before_evidence_id"),
                        prefetched.get("after_evidence_id")):
                    if isinstance(evidence_id, str) and evidence_id:
                        evidence_ids.add(evidence_id)
            if evidence_ids:
                try:
                    self.get_evidence_many(
                        sorted(evidence_ids), expected_kind="correction_value")
                except (EvidenceIntegrityError, NotReadyError):
                    # A later side-specific check classifies the affected value as
                    # invalid/unavailable rather than making unrelated rows vanish.
                    pass
        for r in _rows(
                artifact, columns, batch=20_000, filters=parquet_filters):
            meta = self._documents.get(r["doc_id"])
            if meta is None or not self._matches_document(
                    meta, as_of=cutoff, corp_code=corp_code, corp_name=corp_name,
                    doc_group=doc_group, event_type=event_type, form=form,
                    doc_id=doc_id, rcept_no=rcept_no, is_correction=None):
                continue
            if (wanted_receipts is not None
                    and meta.rcept_no not in wanted_receipts):
                continue
            if str(r["rcept_dt"] or meta.rcept_dt) > cutoff:
                continue
            item_path = str(r["path"] or "")
            if want_path is not None and _text_key(item_path) != want_path:
                continue
            if want_path_part is not None and want_path_part not in _text_key(item_path):
                continue
            if want_reason is not None and want_reason not in _text_key(r["reason"]):
                continue
            if diff_kind is not None and r["diff_kind"] != diff_kind:
                continue
            if before_kind is not None and r["before_kind"] != before_kind:
                continue
            if after_kind is not None and r["after_kind"] != after_kind:
                continue
            if (required_by_authority is not None
                    and bool(r["required_by_authority"]) != required_by_authority):
                continue

            def safe_value(label: str, value: str | None) -> tuple[str | None, str | None]:
                if value is None:
                    return None, None
                field_projection = project_field_value(
                    label, str(value), (meta.corp_name, meta.listed_name))
                prompt_projection = project_chunk_text(
                    field_projection.value_masked,
                    (meta.corp_name, meta.listed_name))
                # typed scalar API는 경계 안의 안전 본문을, LLM용 명시 필드는 경계를
                # 포함한 전체 projection을 준다. 어느 쪽도 raw fallback은 아니다.
                prefix = PROMPT_DATA_BEGIN + "\n"
                suffix = "\n" + PROMPT_DATA_END
                prompt_safe = prompt_projection.text_prompt_safe
                body = (prompt_safe[len(prefix):-len(suffix)]
                        if prompt_safe.startswith(prefix) and prompt_safe.endswith(suffix)
                        else None)
                if body is None:
                    raise ReadSecurityError("CorrectionItem prompt-safe 경계가 손상되었습니다")
                return body, prompt_safe

            reason_safe, reason_prompt = safe_value("정정사유", r.get("reason"))
            before_safe, before_prompt = safe_value(item_path, r.get("value_before"))
            after_safe, after_prompt = safe_value(item_path, r.get("value_after"))
            try:
                canonical_locator(r["locator"])
                table_locator = table_locator_of(r["locator"])
            except CanonicalLocatorError as exc:
                raise ReadSecurityError(
                    f"CorrectionItem locator가 canonical하지 않습니다: {r['locator']!r}") from exc
            if table_locator is None:
                raise ReadSecurityError("CorrectionItem 대표 locator에 TABLE 조상이 없습니다")
            if modern_semantics:
                logical_row = r.get("logical_row")
                logical_col = r.get("logical_col")
                synthetic_row = "/ROW[" in str(r["locator"])
                valid_pair = (
                    (logical_row is None and logical_col is None) if synthetic_row
                    else (type(logical_row) is int and type(logical_col) is int
                          and logical_row >= 0 and logical_col >= 0))
                if not valid_pair:
                    raise ArtifactIntegrityError(self.root, [ArtifactIssue(
                        "invalid_parquet",
                        f"Correction 대표 semantic 좌표 오류: {r['block_id']}",
                        "correction_items")])
            else:
                logical_row = logical_col = None

            def side_coordinates(value: str | None, side_locator: str | None,
                                 stored_row: object, stored_col: object,
                                 ) -> tuple[str | None, str | None, int | None, int | None]:
                actual = side_locator
                if not actual:
                    if modern_semantics and (stored_row is not None or stored_col is not None):
                        raise ArtifactIntegrityError(self.root, [ArtifactIssue(
                            "invalid_parquet", "Correction side locator 없이 좌표만 존재",
                            "correction_items")])
                    return None, None, None, None
                try:
                    canonical_locator(actual)
                    side_table = table_locator_of(actual)
                except CanonicalLocatorError as exc:
                    raise ReadSecurityError(
                        f"CorrectionItem side locator가 canonical하지 않습니다: {actual!r}"
                    ) from exc
                if side_table != table_locator:
                    raise ArtifactIntegrityError(self.root, [ArtifactIssue(
                        "invalid_parquet", "Correction side/row TABLE ancestry 불일치",
                        "correction_items")])
                if modern_semantics:
                    side_row, side_col = stored_row, stored_col
                    if (type(side_row) is not int or type(side_col) is not int
                            or side_row < 0 or side_col < 0):
                        raise ArtifactIntegrityError(self.root, [ArtifactIssue(
                            "invalid_parquet", "Correction side semantic 좌표 없음/오류",
                            "correction_items")])
                else:
                    side_row = side_col = None
                return str(actual), side_table, side_row, side_col

            before_locator, before_table, before_row, before_col = side_coordinates(
                r.get("value_before"), r.get("before_locator"),
                r.get("before_logical_row"), r.get("before_logical_col"))
            after_locator, after_table, after_row, after_col = side_coordinates(
                r.get("value_after"), r.get("after_locator"),
                r.get("after_logical_row"), r.get("after_logical_col"))

            def side_status(value: str | None, fk: str | None,
                            expected_safe: str | None, side_locator: str | None,
                            side_table: str | None, side_row: int | None,
                            side_col: int | None) -> str:
                if value is None:
                    return "not_applicable" if fk is None else "invalid"
                if (not modern_semantics or side_locator is None
                        or side_table is None or side_row is None
                        or side_col is None):
                    return "invalid"
                if not verify_evidence:
                    return "unverified" if fk else "unavailable"
                return self._evidence_fk_status(
                    fk, kind="correction_value", doc_id=meta.doc_id,
                    source_file_id=r["source_file_id"], locator=side_locator,
                    rcept_dt=str(r["rcept_dt"] or meta.rcept_dt),
                    table_locator=side_table, logical_row=side_row,
                    logical_col=side_col, expected_safe=expected_safe)

            before_status = side_status(
                r.get("value_before"), r.get("before_evidence_id"), before_prompt,
                before_locator, before_table, before_row, before_col)
            after_status = side_status(
                r.get("value_after"), r.get("after_evidence_id"), after_prompt,
                after_locator, after_table, after_row, after_col)
            required_statuses = [
                status for status in (before_status, after_status)
                if status != "not_applicable"
            ]
            if required_statuses and all(status == "verified"
                                         for status in required_statuses):
                evidence_status = "verified"
            elif "invalid" in required_statuses:
                evidence_status = "invalid"
            elif "unavailable" in required_statuses or not required_statuses:
                evidence_status = "unavailable"
            else:
                evidence_status = "unverified"
            yield CorrectionItemRow(
                doc_id=meta.doc_id, rcept_no=meta.rcept_no,
                rcept_dt=str(r["rcept_dt"] or meta.rcept_dt), corp_code=meta.corp_code,
                corp_name=meta.corp_name, doc_group=meta.doc_group,
                event_type=meta.event_type, form=meta.form, report_nm=meta.report_nm,
                source_file_id=r["source_file_id"], block_id=r["block_id"],
                parent_id=r["parent_id"], path=item_path, order=int(r["order"]),
                locator=r["locator"], reason=reason_safe,
                reason_prompt_safe=reason_prompt,
                reason_raw=(r.get("reason") if include_restricted_raw else None),
                value_before=before_safe, value_after=after_safe,
                value_before_prompt_safe=before_prompt,
                value_after_prompt_safe=after_prompt,
                value_before_raw=(r.get("value_before")
                                  if include_restricted_raw else None),
                value_after_raw=(r.get("value_after")
                                 if include_restricted_raw else None),
                restricted_raw_included=include_restricted_raw,
                before_evidence_id=r.get("before_evidence_id"),
                after_evidence_id=r.get("after_evidence_id"),
                evidence_status=evidence_status,
                before_evidence_status=before_status,
                after_evidence_status=after_status,
                table_locator=table_locator, logical_row=logical_row,
                logical_col=logical_col,
                before_locator=before_locator, after_locator=after_locator,
                before_table_locator=before_table, after_table_locator=after_table,
                before_logical_row=before_row, before_logical_col=before_col,
                after_logical_row=after_row, after_logical_col=after_col,
                diff_kind=r["diff_kind"], before_kind=r["before_kind"],
                after_kind=r["after_kind"],
                required_by_authority=bool(r["required_by_authority"]))

    # ---------------------------------------------------------------- 사건
    def state_as_of(self, event_key: str, as_of: str) -> EventState:
        """그 시점의 사건 상태. 관측을 `observed_at <= as_of` 로 걸러 계산한다.

        `EventIdentity.status_at_corpus_end` 를 그냥 읽으면 과거 시점 질의가 전부
        같은 답을 낸다 — Freudenberg 계약은 2025-12-25 에 유효하고 12-26 에 해지된다.

        이 메서드는 1.8 호환용 **미검증 파생 상태**만 반환한다. 근거 지원, 사건
        식별 상태, 부재 질의 완전성을 함께 확인해야 하는 Agent/확정 응답 경로는
        :meth:`event_timeline` 을 ``verify_evidence=True`` 로 호출해야 한다.
        """
        cutoff = _checked_as_of(as_of)
        self._load_observations()
        seen = [o for o in self._obs.get(event_key, ()) if o["observed_at"] <= cutoff]
        if not seen:
            return EventState(event_key, cutoff, "not_disclosed", 0, None, None)
        status = "terminated" if any(o["is_termination"] for o in seen) else "active"
        return EventState(event_key, cutoff, status, len(seen),
                          seen[-1]["rcept_no"], seen[-1]["observed_at"])

    def event_timeline(self, *, as_of: str, event_key: str | None = None,
                       rcept_no: str | None = None,
                       verify_evidence: bool = False) -> EventTimeline | None:
        """사건의 기준시점 timeline. ``event_key``와 ``rcept_no`` 중 하나만 준다.

        identity에 저장된 코퍼스 끝 상태·전체 관측 수는 반환하지 않는다. 그것을 과거
        시점에 노출하면 미래 해지 사실이 새기 때문이다. 반환되는 ``state``와
        ``observations``는 모두 ``observed_at <= as_of``로 다시 계산된다.
        """
        cutoff = _checked_as_of(as_of)
        verify_evidence = _checked_bool_flag("verify_evidence", verify_evidence)
        if (event_key is None) == (rcept_no is None):
            raise ValueError("event_key와 rcept_no 중 정확히 하나를 지정해야 합니다")
        self._load_docs()
        self._load_observations()
        self._load_event_identities()

        if rcept_no is not None:
            keys = [
                key for key in self._event_keys_by_rcept.get(rcept_no, ())
                if any(o["rcept_no"] == rcept_no
                       and o["observed_at"] <= cutoff
                       for o in self._obs.get(key, ()))
            ]
            if not keys:
                return None
            if len(keys) > 1:
                raise AmbiguousEventError(
                    f"rcept_no={rcept_no!r}가 여러 Event에 속합니다: {sorted(keys)}")
            event_key = keys[0]

        assert event_key is not None
        cache_key = (cutoff, event_key, verify_evidence)
        if cache_key in self._event_timeline_cache:
            self._event_timeline_cache.move_to_end(cache_key)
            return self._event_timeline_cache[cache_key]
        seen = [o for o in self._obs.get(event_key, ())
                if o["observed_at"] <= cutoff]
        if not seen:
            self._event_timeline_cache[cache_key] = None
            self._event_timeline_cache.move_to_end(cache_key)
            while len(self._event_timeline_cache) > self._event_timeline_cache_limit:
                self._event_timeline_cache.popitem(last=False)
            return None
        identity = self._event_identities.get(event_key)
        if identity is None:
            raise NotReadyError(f"EventObservation의 identity가 없습니다: {event_key}")
        root_doc_id = self._doc_id_by_rcept.get(identity["root_rcept_no"])
        root_meta = self._documents.get(root_doc_id or "")
        modern_support = _schema_at_least(self.schema_version, (1, 9))
        if verify_evidence and not modern_support:
            raise NotReadyError(
                "Event support Evidence 검증은 schema 1.9+ artifact가 필요합니다")

        prepared: list[tuple[dict, tuple[str, ...], tuple[str, ...], str,
                             str | None, str | None]] = []
        all_support_ids: set[str] = set()
        for observation in seen:
            doc_id = str(observation.get("doc_id")
                         or self._doc_id_by_rcept.get(observation["rcept_no"], ""))
            if not modern_support:
                prepared.append((observation, (), (), "unavailable", None,
                                 "legacy_schema"))
                continue
            ids_raw = observation.get("supporting_evidence_ids")
            roles_raw = observation.get("support_roles")
            if (not isinstance(ids_raw, list) or not isinstance(roles_raw, list)
                    or any(not isinstance(item, str) or not item for item in ids_raw)
                    or any(role not in EVENT_SUPPORT_ROLES for role in roles_raw)
                    or len(ids_raw) != len(roles_raw)
                    or ids_raw != sorted(ids_raw)
                    or len(ids_raw) != len(set(ids_raw))):
                raise EvidenceIntegrityError(
                    f"Event support parallel list/정렬 계약 오류: {observation['rcept_no']}")
            ids, roles = tuple(ids_raw), tuple(roles_raw)
            if identity["kind"] == "document_lineage" and (ids or roles):
                raise EvidenceIntegrityError(
                    "document_lineage Event support는 빈 목록이어야 합니다: "
                    f"{observation['rcept_no']}")
            version = observation.get("support_version")
            if version != EVENT_SUPPORT_VERSION:
                raise EvidenceIntegrityError(
                    f"Event support_version 불일치: {observation['rcept_no']}")
            meta = self._documents.get(doc_id)
            if meta is None or meta.rcept_no != observation["rcept_no"]:
                raise EvidenceIntegrityError(
                    f"EventObservation Document/receipt FK 오류: {observation['rcept_no']}")
            expected_identity_kind = (
                "document_lineage" if meta.doc_group == "periodic"
                else "business_event")
            if (identity.get("kind") != expected_identity_kind
                    or identity.get("doc_group") != meta.doc_group):
                raise EvidenceIntegrityError(
                    "EventIdentity kind/doc_group과 Document 그룹 불일치: "
                    f"event={event_key!r} rcept={observation['rcept_no']!r}")
            document_context = {
                "doc_group": meta.doc_group, "doc_subtype": meta.doc_subtype,
                "event_type": meta.event_type, "report_nm": meta.report_nm,
                "is_correction": meta.is_correction,
            }
            if (type(observation.get("is_correction")) is not bool
                    or observation.get("is_correction") is not meta.is_correction
                    or type(observation.get("is_termination")) is not bool
                    or observation.get("is_termination") is not is_declared_termination(
                        document_context)):
                raise EvidenceIntegrityError(
                    f"EventObservation boolean/declared type 불일치: "
                    f"{observation['rcept_no']}")
            expected_status, expected_limitation = event_support_decision(
                document_context, str(identity["kind"]), set(roles))
            stored_status = observation.get("support_status")
            stored_limitation = observation.get("support_limitation")
            if (stored_status != expected_status
                    or stored_limitation != expected_limitation):
                raise EvidenceIntegrityError(
                    "Event support status/필수 role 불일치: "
                    f"{observation['rcept_no']}")
            prepared.append((observation, ids, roles, str(stored_status),
                             str(version), stored_limitation))
            all_support_ids.update(ids)

        verified_support: dict[str, EvidenceRow | None] = {}
        field_owner: dict[str, dict] = {}
        if verify_evidence and all_support_ids:
            verified_support = self.get_evidence_many(
                sorted(all_support_ids), expected_kind="field_value")
            field_artifact = self.root / "fields.parquet"
            field_columns = [
                "build_id", "evidence_id", "doc_id", "source_file_id", "path",
                "is_pii", "value_status", "rcept_dt",
            ]
            field_rows = _read_artifact_table(
                field_artifact, columns=field_columns,
                filters=[("evidence_id", "in", sorted(all_support_ids))]).to_pylist()
            for row in field_rows:
                evidence_key = str(row.get("evidence_id") or "")
                if evidence_key in field_owner:
                    raise EvidenceIntegrityError(
                        f"Event support Field owner 복수: {evidence_key}")
                field_owner[evidence_key] = row

        typed_rows: list[EventObservationRow] = []
        for observation, ids, roles, stored_status, version, limitation in prepared:
            doc_id = str(observation.get("doc_id")
                         or self._doc_id_by_rcept.get(observation["rcept_no"], ""))
            if verify_evidence:
                for evidence_key, role in zip(ids, roles):
                    evidence = verified_support.get(evidence_key)
                    owner = field_owner.get(evidence_key)
                    if evidence is None or owner is None:
                        raise EvidenceIntegrityError(
                            f"Event support Evidence/Field FK 없음: {evidence_key}")
                    actual_role = event_support_role(
                        owner.get("path"), value_status=owner.get("value_status"),
                        is_pii=owner.get("is_pii"))
                    if (owner.get("build_id") != self.build_id
                            or owner.get("doc_id") != doc_id
                            or evidence.doc_id != doc_id
                            or evidence.source_file_id != owner.get("source_file_id")
                            or evidence.rcept_dt != observation["observed_at"]
                            or owner.get("rcept_dt") != observation["observed_at"]
                            or owner.get("is_pii") is not False
                            or evidence.pii_types
                            or actual_role != role):
                        raise EvidenceIntegrityError(
                            f"Event support ownership/role/PII 불일치: {evidence_key}")
            typed_rows.append(EventObservationRow(
                doc_id=doc_id, event_key=event_key, seq=int(observation["seq"]),
                rcept_no=observation["rcept_no"], observed_at=observation["observed_at"],
                is_correction=bool(observation["is_correction"]),
                is_termination=bool(observation["is_termination"]),
                previous_observation_rcept_no=observation.get(
                    "previous_observation_rcept_no"),
                supporting_evidence_ids=ids, support_roles=roles,
                support_status=stored_status, support_version=version,
                support_limitation=limitation,
                evidence_verification_status=(
                    "not_applicable" if stored_status == "not_applicable"
                    else "verified" if verify_evidence and ids
                    else "unverified" if ids else "unavailable")))
        typed = tuple(typed_rows)
        state = self.state_as_of(event_key, cutoff)

        if identity["kind"] == "document_lineage":
            observation_status = "not_applicable"
            observation_limitation = None
        elif not modern_support:
            observation_status = "unavailable"
            observation_limitation = "legacy_schema"
        elif not verify_evidence:
            observation_status = "unverified"
            observation_limitation = "explicit_verification_required"
        else:
            partial = [row.support_limitation for row in typed
                       if row.support_status != "fully_verified"]
            observation_status = "fully_verified" if not partial else "partial"
            observation_limitation = (
                None if not partial else "observation_partial:" + "|".join(
                    sorted(set(item or "unspecified" for item in partial))))

        identity_status = str(identity.get("identity_status") or "unavailable")
        identity_limitation = (
            None if identity_status == "resolved"
            else f"identity_{identity_status}")
        if identity["kind"] == "document_lineage":
            support_status = "not_applicable"
            support_limitation = None
        elif (observation_status == "fully_verified" and identity_status == "resolved"
              and state.status == "terminated"):
            # 종료는 양성 전이 Field로 입증할 수 있다. 반대로 active/not_disclosed는
            # 이후/누락 공시가 없다는 query completeness proof가 없으면 확정할 수 없다.
            support_status = "fully_verified"
            support_limitation = None
        else:
            support_status = "partial" if modern_support else "unavailable"
            limitations = [item for item in
                           (observation_limitation, identity_limitation) if item]
            if state.status in {"active", "not_disclosed"}:
                limitations.append("corpus_query_completeness_not_proven")
            support_limitation = "|".join(limitations) or "support_unavailable"
        citations = (tuple(sorted(all_support_ids))
                     if support_status == "fully_verified" and verify_evidence else ())
        result = EventTimeline(
            event_key=event_key, kind=identity["kind"],
            corp_code=identity["corp_code"], corp_name=identity["corp_name"],
            doc_group=identity["doc_group"],
            form=str(identity["doc_subtype"] or (root_meta.form if root_meta else "")),
            root_rcept_no=identity["root_rcept_no"], as_of=cutoff,
            state=state, observations=typed,
            support_status=support_status, support_limitation=support_limitation,
            citations=citations,
            observation_support_status=observation_status,
            observation_support_limitation=observation_limitation,
            identity_verification_status=identity_status,
            identity_limitation=identity_limitation,
            # 코퍼스 밖/검색 누락이 없다는 query completeness proof는 이 schema가
            # 만들지 않는다. Evidence provenance와 별도 축으로 남긴다.
            query_coverage_status="not_proven",
            query_coverage_limitation="corpus_query_completeness_not_proven",
            identity_fingerprint=identity.get("identity_fingerprint"),
            identity_status=identity.get("identity_status"),
            resolver_version=identity.get("resolver_version"))
        self._event_timeline_cache[cache_key] = result
        self._event_timeline_cache.move_to_end(cache_key)
        while len(self._event_timeline_cache) > self._event_timeline_cache_limit:
            self._event_timeline_cache.popitem(last=False)
        return result

    def event_identity_candidate_roots(
            self, *, event_key: str, as_of: str,
            ) -> tuple[str, ...]:
        """Return all visible roots sharing an ambiguous event fingerprint.

        A timeline keeps one event key even when another, otherwise-identical
        event exists. Consumers that disclose that ambiguity still need every
        candidate coordinate to prove the limitation. This method exposes
        coordinates only; it never chooses or merges a candidate.
        """

        cutoff = _checked_as_of(as_of)
        if not re.fullmatch(r"[0-9a-f]{32}", str(event_key or "")):
            raise ValueError("event_key 형식이 잘못되었습니다")
        self._load_docs()
        self._load_event_identities()
        identity = self._event_identities.get(event_key)
        if identity is None:
            return ()
        root = str(identity["root_rcept_no"])
        fingerprint = identity.get("identity_fingerprint")
        if identity.get("identity_status") != "ambiguous" or not fingerprint:
            return (root,)

        candidates = []
        for sibling in self._event_identities.values():
            sibling_root = str(sibling.get("root_rcept_no") or "")
            doc_id = self._doc_id_by_rcept.get(sibling_root, "")
            meta = self._documents.get(doc_id)
            if (sibling.get("corp_code") == identity.get("corp_code")
                    and sibling.get("doc_group") == identity.get("doc_group")
                    and sibling.get("identity_fingerprint") == fingerprint
                    and sibling.get("identity_status") == "ambiguous"
                    and meta is not None and meta.rcept_dt <= cutoff):
                candidates.append(sibling_root)
        return tuple(sorted(set(candidates)))

    def event_of(self, rcept_no: str) -> str | None:
        """접수번호가 속한 사건 key."""
        self._load_observations()
        keys = list(self._event_keys_by_rcept.get(rcept_no, ()))
        if len(keys) > 1:
            raise AmbiguousEventError(
                f"rcept_no={rcept_no!r}가 여러 Event에 속합니다: {sorted(keys)}")
        return keys[0] if keys else None

    def latest_document_version(self, rcept_no: str, *, as_of: str) -> str | None:
        """그 **문서의 최신 정정본**. 정정 사슬만 따라간다.

        사건의 마지막 관측과 **다른 질문**이다 — 5차 검수 지적. 계약 체결 공시의
        최신본을 물었는데 그 계약의 **해지 공시**가 돌아오면 안 된다. 해지는 같은
        사건의 다음 관측이지 그 문서의 새 판본이 아니다.

            latest_document_version(체결)  →  [기재정정]체결      ← 이 함수
            state_as_of(사건).last_rcept_no →  해지               ← 저쪽
        """
        result = self.resolve_document_version(rcept_no, as_of=as_of)
        return result.selected if result.status == "ok" else None

    def resolve_document_version(self, rcept_no: str, *, as_of: str) -> LatestResolution:
        """최신 정정본과 **고르지 못한 이유**를 함께 반환한다.

        ``latest_document_version()``의 안전한 상세판이다. 명확한 체인은 ``ok``이고,
        후속 leaf가 여럿이면 ``ambiguous``, cycle이면 ``invalid``다. 두 경우 모두
        ``selected``는 ``None``이므로 답변 층이 접수번호 최대값으로 우회할 수 없다.
        """
        cutoff = _checked_as_of(as_of)
        self._load_relations()                # `_load_docs` 를 함께 호출한다
        # **`as_of` 이후에 접수된 문서는 그 시점에 존재하지 않는다.** 씨앗 자체가
        # 미래면 「그때의 최신본」이라는 질문이 성립하지 않는다 (8차 검수).
        seed_dt = self._rcept_dt.get(rcept_no)
        if seed_dt is None or seed_dt > cutoff:
            return LatestResolution(
                "not_found", None, (), (), "모르는 문서이거나 기준시점 이후 접수")
        edges = [e for e in self._relations
                 if e["src_rcept_no"] and (e["dst_rcept_no"] or "")
                 and self._rcept_dt.get(e["src_rcept_no"], "") <= cutoff
                 and self._rcept_dt.get(e["dst_rcept_no"], "") <= cutoff]
        got = resolve_latest_known(rcept_no, edges)
        if got.status == "not_found":
            # 문서는 존재하지만 이 시점까지 정정 계보가 없다. 자기 자신이 명확한 유효본이다.
            return LatestResolution(
                "ok", rcept_no, (rcept_no,), (rcept_no,), "no_corrections_as_of")
        return got

    def last_observation_of(self, rcept_no: str, *, as_of: str) -> str | None:
        """그 접수번호가 속한 **사건의 마지막 관측**. 해지·변경을 포함한다.

        「이 계약은 지금 어떤 상태인가」를 묻는 쪽이다.
        문서의 최신 판본은 `latest_document_version()` 이다.

        반환값은 1.8 호환용 **미검증 파생 포인터**다. 인용 가능한 근거와 typed
        limitation이 필요한 확정 경로는 :meth:`event_timeline` 을
        ``verify_evidence=True`` 로 호출한다.
        """
        # **색인을 먼저 채운다.** `event_of` 는 관측만 읽어서 `_rcept_dt` 를 채우지
        # 않는다. 새 인스턴스에서 곧바로 부르면 접수일을 몰라 미래 seed 검사가
        # 통째로 통과했다 (9차 검수 — cold start 누출).
        cutoff = _checked_as_of(as_of)
        self._load_docs()
        seed_dt = self._rcept_dt.get(rcept_no)
        if seed_dt is None or seed_dt > cutoff:
            return None                       # 모르는 문서이거나 그 시점엔 아직 없다
        key = self.event_of(rcept_no)
        if key is None:
            return None
        return self.state_as_of(key, cutoff).last_rcept_no

    def _unread_docs(self, corp_code: str, period_end: str, as_of: str,
                     scope: str, view: str = "restated",
                     statement: str | None = None) -> tuple[str, ...]:
        """그 회사의 **그 기간·그 범위**에서 우리가 못 읽은 문서들(접수번호).

        **관점마다 봐야 할 문서가 다르다** (8차 검수).
        `restated` 는 「최신 재작성본을 봤는가」이므로 **뒤에 나온** 보고서까지 본다.
        `as_filed` 는 「최초 제출본을 봤는가」이므로 **그 기간 당대**의 것만 본다 —
        뒤에 나온 재작성본을 못 읽은 것은 최초 제출값과 무관하다.

        예전에는 법인과 접수일만 봤다 — `period_end` 를 인자로 받고도 쓰지 않아,
        **한 해가 미지원이면 그 회사의 모든 연도**가 「추출 미지원」이 됐다
        (5차 검수 지적). 문서의 보고 기준월(`base_year`·`base_month`)로 기간을
        맞추고, 첨부 배치 문제는 **못 읽은 범위**(CFS/SFS)까지 맞춘다.
        """
        self._load_docs()
        # `period_end` 는 `2025-12-31` 형식이다. 구분자를 떼고 연·월만 본다 —
        # 안 떼면 `2025-1` 이 되어 어떤 문서와도 안 맞는다.
        want = period_end.replace("-", "")[:6] if period_end else ""
        # **뒤에 나온 보고서도 그 기간을 다시 적는다.** 2025 사업보고서는 2024 비교표를
        # 담으므로, 그것을 못 읽었으면 2024 질의의 **재작성본을 못 본 것**이다
        # (7차 검수 — restated 후속 누락).
        #
        # 다만 **한 기(期)까지만**이다. 한국 재무제표는 당기 + 전기 비교표를 싣는다 —
        # 2025 보고서는 2023 을 다시 적지 않는다. 「기준월이 질의 이후면 전부」로 두면
        # 그 회사의 **모든 과거 질의**가 근거 부족이 되어 조회가 사실상 멈춘다.
        out: list[str] = []
        for d in self._unsupported_docs:
            if d["corp_code"] != corp_code or d["rcept_dt"] > as_of:
                continue
            # `as_filed`는 최초 제출 관점이다. 같은 기준기간을 정정한 후속 문서를
            # 못 읽었더라도 최초 제출값의 coverage를 오염시키지 않는다. 반대로 원본
            # 자체가 미지원이면 아래에서 그대로 partial로 남는다.
            if view == "as_filed" and d.get("is_correction", False):
                continue
            if want and d["base"]:
                if not self._covers(d["base"], want, view, statement,
                                    self._fy_end.get(corp_code, 12)):
                    continue
            if d["missing_scope"] and scope not in d["missing_scope"]:
                continue                      # 그 범위는 읽었다
            out.append(d["rcept_no"])
        return tuple(sorted(out))

    def _load_relations(self) -> None:
        if self._relations is not None:
            return
        self._load_docs()
        artifact = self.root / "relations.parquet"
        present = _artifact_columns(artifact)
        receipt_columns = {"src_rcept_no", "dst_rcept_no"}
        has_receipts = receipt_columns.issubset(present)
        if receipt_columns & present and not has_receipts:
            raise ArtifactIntegrityError(self.root, [ArtifactIssue(
                "invalid_parquet", "Relation receipt 컬럼이 부분적으로만 존재합니다",
                "relations")])
        # schema 1.6부터 src_id/dst_id는 Document.doc_id다. 접수번호가 필요한 lineage
        # 알고리즘에 ID를 넘기면 조용히 `not_found → 자기 자신`이 되어 최신 정정본을 놓친다.
        # ID=접수번호였던 fallback은 발행된 schema 1.2에만 허용한다.
        if not has_receipts and self.schema_version != "1.2":
            raise ArtifactIntegrityError(self.root, [ArtifactIssue(
                "invalid_parquet",
                f"schema {self.schema_version} Relation에 receipt endpoint 컬럼이 없습니다",
                "relations")])

        wanted = ["src_kind", "src_id", "src_rcept_no", "dst_kind", "dst_id",
                  "dst_rcept_no", "relation_type", "resolution_status"]
        relations: list[dict] = []
        for r in _rows(artifact, _present_columns(artifact, wanted)):
            # **확정된 문서 간선만 따라간다.** 대상이 모호한 정정으로 사슬을 만들면
            # 틀린 「최신본」을 답하게 된다.
            if not (r["resolution_status"] == "resolved"
                    and r["src_kind"] == "document"
                    and r["dst_kind"] == "document"):
                continue
            if has_receipts:
                src_receipt = r.get("src_rcept_no")
                dst_receipt = r.get("dst_rcept_no")
                src_doc_id = r.get("src_id")
                dst_doc_id = r.get("dst_id")
                if (not src_receipt or not dst_receipt
                        or self._doc_id_by_rcept.get(src_receipt) != src_doc_id
                        or self._doc_id_by_rcept.get(dst_receipt) != dst_doc_id):
                    raise ArtifactIntegrityError(self.root, [ArtifactIssue(
                        "invalid_parquet",
                        "Relation doc_id와 receipt endpoint FK가 일치하지 않습니다",
                        "relations")])
            else:
                src_receipt = r["src_id"]
                dst_receipt = r["dst_id"]
            relations.append({
                "src_rcept_no": src_receipt,
                "dst_rcept_no": dst_receipt,
                "relation_type": r["relation_type"],
                "resolution_status": "resolved",
            })
        self._relations = relations

    def get_relation_support(
            self, relation_id: str, *, include_citations: bool = False,
            ) -> RelationSupportRow | None:
        """Relation support를 typed row로 반환한다.

        ``include_citations=True``가 아니면 Evidence를 검증하거나 citation으로 노출하지
        않는다. schema 1.9 이전에는 support 열 자체가 없으므로 빈 정상값으로 꾸미지 않고
        :class:`NotReadyError`로 닫는다.
        """
        include_citations = _checked_bool_flag("include_citations", include_citations)
        if not _schema_at_least(self.schema_version, (1, 9)):
            raise NotReadyError("Relation support API는 schema 1.9+ artifact가 필요합니다")
        wanted_id = str(relation_id or "")
        if not wanted_id:
            raise ValueError("relation_id는 빈 문자열일 수 없습니다")
        self._load_docs()
        artifact = self.root / "relations.parquet"
        columns = [
            "build_id", "relation_id", "src_id", "src_rcept_no", "src_rcept_dt",
            "dst_id", "dst_rcept_no", "relation_type", "resolution_status",
            "supporting_evidence_ids", "support_roles", "support_status",
            "support_version", "support_limitation",
        ]
        rows = _read_artifact_table(
            artifact, columns=columns,
            filters=[("relation_id", "=", wanted_id)]).to_pylist()
        if not rows:
            return None
        if len(rows) != 1:
            raise EvidenceIntegrityError(f"Relation ID가 복수 행입니다: {wanted_id}")
        row = rows[0]
        ids, roles = row.get("supporting_evidence_ids"), row.get("support_roles")
        if (row.get("build_id") != self.build_id
                or not isinstance(ids, list) or not isinstance(roles, list)
                or len(ids) != len(roles) or ids != sorted(ids)
                or len(ids) != len(set(ids))
                or any(not isinstance(item, str) or not item for item in ids)
                or any(role not in RELATION_SUPPORT_ROLES for role in roles)
                or row.get("support_version") != RELATION_SUPPORT_VERSION):
            raise EvidenceIntegrityError(f"Relation support 구조 계약 오류: {wanted_id}")
        src_doc = self._documents.get(str(row.get("src_id") or ""))
        if (src_doc is None or src_doc.rcept_no != row.get("src_rcept_no")
                or src_doc.rcept_dt != row.get("src_rcept_dt")):
            raise EvidenceIntegrityError(f"Relation source ownership 오류: {wanted_id}")
        limitation = row.get("support_limitation")
        # support 목록에서는 이미 제외된 multi-owner Evidence를 볼 수 없다. source
        # CorrectionItem의 `(doc_id, order)` owner를 다시 세고, non-PII Evidence일 때만
        # producer와 같은 ambiguity limitation을 요구한다. block_id는 동일 대표 locator를
        # 공유하는 의미 행에서 같아질 수 있어 owner key로 쓰지 않는다.
        correction_limitation_columns = [
            "build_id", "doc_id", "source_file_id", "order", "rcept_dt",
            "before_evidence_id", "after_evidence_id",
        ]
        source_corrections = _read_artifact_table(
            self.root / "correction_items.parquet",
            columns=correction_limitation_columns,
            filters=[("doc_id", "=", src_doc.doc_id)],
        ).to_pylist()
        if any(owner.get("build_id") != self.build_id
               or owner.get("doc_id") != src_doc.doc_id
               or owner.get("rcept_dt") != src_doc.rcept_dt
               for owner in source_corrections):
            raise EvidenceIntegrityError(
                f"Relation Correction limitation owner 오류: {wanted_id}")
        try:
            correction_owner_keys = relation_correction_support_owner_keys(
                source_corrections)
        except ValueError as exc:
            raise EvidenceIntegrityError(
                f"Relation Correction limitation owner key 오류: {wanted_id}") from exc
        ambiguous_candidates = {
            evidence_key for evidence_key, owner_keys in correction_owner_keys.items()
            if len(owner_keys) > 1
        }
        has_ambiguous_owner = False
        if ambiguous_candidates:
            candidate_evidence = self.get_evidence_many(ambiguous_candidates)
            for evidence_key in sorted(ambiguous_candidates):
                evidence = candidate_evidence.get(evidence_key)
                owner_sources = {
                    str(owner["source_file_id"])
                    for owner in source_corrections
                    if evidence_key in {
                        owner.get("before_evidence_id"), owner.get("after_evidence_id")
                    }
                }
                if (evidence is None or evidence.kind != "correction_value"
                        or evidence.doc_id != src_doc.doc_id
                        or evidence.rcept_dt != src_doc.rcept_dt
                        or owner_sources != {evidence.source_file_id}):
                    raise EvidenceIntegrityError(
                        f"Relation ambiguous Correction Evidence owner 오류: {evidence_key}")
                if not evidence.pii_types:
                    has_ambiguous_owner = True
        try:
            expected_limitation = relation_support_limitation(
                row.get("resolution_status"),
                has_source_evidence=bool(ids),
                has_ambiguous_correction_evidence_owner=has_ambiguous_owner,
            )
        except (TypeError, ValueError) as exc:
            raise EvidenceIntegrityError(
                f"Relation limitation 판정 입력 오류: {wanted_id}") from exc
        # 현재 schema에는 exact target-anchor Evidence role이 없다. 따라서 fully_verified
        # 행은 과대 주장이고, resolved도 typed missing anchor가 필수다. 추가/누락 token도
        # producer와 다른 의미 경계이므로 exact 문자열로 닫는다.
        if (row.get("support_status") != "partial"
                or limitation != expected_limitation):
            raise EvidenceIntegrityError(
                f"Relation typed limitation exact 오류: {wanted_id}")

        if include_citations and ids:
            evidence_rows = self.get_evidence_many(ids)
            field_ids = {
                evidence_key for evidence_key, role in zip(ids, roles)
                if relation_correction_support_sides(role) is None
            }
            field_owner: dict[str, dict] = {}
            if field_ids:
                for owner in _read_artifact_table(
                        self.root / "fields.parquet",
                        columns=["build_id", "evidence_id", "doc_id", "source_file_id",
                                 "path", "is_pii", "value_status", "rcept_dt"],
                        filters=[("evidence_id", "in", sorted(field_ids))]).to_pylist():
                    evidence_key = str(owner.get("evidence_id") or "")
                    if evidence_key in field_owner:
                        raise EvidenceIntegrityError(
                            f"Relation Field support owner 복수: {evidence_key}")
                    field_owner[evidence_key] = owner
            correction_owner: dict[str, list[tuple[dict, str]]] = defaultdict(list)
            correction_ids = set(ids) - field_ids
            if correction_ids:
                correction_columns = [
                    "build_id", "doc_id", "source_file_id", "rcept_dt",
                    "before_evidence_id", "after_evidence_id",
                ]
                for owner in _read_artifact_table(
                        self.root / "correction_items.parquet",
                        columns=correction_columns).to_pylist():
                    for side in ("before", "after"):
                        evidence_key = owner.get(f"{side}_evidence_id")
                        if evidence_key not in correction_ids:
                            continue
                        correction_owner[str(evidence_key)].append((owner, side))
            for evidence_key, role in zip(ids, roles):
                evidence = evidence_rows.get(evidence_key)
                if evidence is None or evidence.doc_id != src_doc.doc_id:
                    raise EvidenceIntegrityError(
                        f"Relation support Evidence source doc 불일치: {evidence_key}")
                if evidence.rcept_dt != src_doc.rcept_dt or evidence.pii_types:
                    raise EvidenceIntegrityError(
                        f"Relation support Evidence date/PII 오류: {evidence_key}")
                expected_sides = relation_correction_support_sides(role)
                if expected_sides is not None:
                    owned = correction_owner.get(evidence_key, ())
                    actual_sides = {side for _owner, side in owned}
                    same_item = bool(owned) and all(owner is owned[0][0]
                                                     for owner, _side in owned)
                    if (actual_sides != set(expected_sides)
                            or len(owned) != len(expected_sides) or not same_item):
                        raise EvidenceIntegrityError(
                            f"Relation Correction support role 불일치: {evidence_key}")
                    owner = owned[0][0]
                    if (owner.get("build_id") != self.build_id
                            or owner.get("doc_id") != src_doc.doc_id
                            or owner.get("source_file_id") != evidence.source_file_id
                            or owner.get("rcept_dt") != src_doc.rcept_dt):
                        raise EvidenceIntegrityError(
                            f"Relation Correction support ownership 오류: {evidence_key}")
                else:
                    owner = field_owner.get(evidence_key)
                    actual_role = (event_support_role(
                        owner.get("path"), value_status=owner.get("value_status"),
                        is_pii=owner.get("is_pii")) if owner else None)
                    if (owner is None or owner.get("build_id") != self.build_id
                            or owner.get("doc_id") != src_doc.doc_id
                            or owner.get("source_file_id") != evidence.source_file_id
                            or owner.get("rcept_dt") != src_doc.rcept_dt
                            or owner.get("is_pii") is not False
                            or actual_role != role):
                        raise EvidenceIntegrityError(
                            f"Relation Field support ownership/role 오류: {evidence_key}")

        # target anchor가 없으므로 현재 버전은 검증 요청에서도 citation을 확정하지 않는다.
        citations: tuple[str, ...] = ()
        return RelationSupportRow(
            relation_id=wanted_id, src_id=str(row["src_id"]),
            src_rcept_no=str(row["src_rcept_no"]), dst_id=row.get("dst_id"),
            dst_rcept_no=row.get("dst_rcept_no"),
            relation_type=str(row["relation_type"]),
            resolution_status=str(row["resolution_status"]),
            supporting_evidence_ids=tuple(ids), support_roles=tuple(roles),
            support_status=str(row["support_status"]),
            support_version=str(row["support_version"]),
            support_limitation=limitation, citations=citations)

    def relation_summaries(
            self, *, source_rcept_no: str, as_of: str,
            ) -> tuple[RelationSummaryRow, ...]:
        """Return non-value lineage coordinates for one visible source receipt.

        Unlike ``_load_relations`` this preserves ``root_missing`` edges: that
        absence is the typed source-scope fact consumed by partial timelines.
        """
        cutoff = _checked_as_of(as_of)
        if not re.fullmatch(r"[0-9]{14}", source_rcept_no):
            raise ValueError("source_rcept_no 형식이 잘못되었습니다")
        self._load_docs()
        columns = ["relation_id", "src_rcept_no", "dst_rcept_no",
                   "relation_type", "resolution_status", "target_hint",
                   "root_missing_reason", "src_rcept_dt"]
        rows = _read_artifact_table(
            self.root / "relations.parquet", columns=columns,
            filters=[("src_rcept_no", "=", source_rcept_no)]).to_pylist()
        result = []
        for row in rows:
            src_date = row.get("src_rcept_dt")
            if not isinstance(src_date, str) or src_date > cutoff:
                continue
            result.append(RelationSummaryRow(
                relation_id=str(row["relation_id"]),
                src_rcept_no=str(row["src_rcept_no"]),
                dst_rcept_no=row.get("dst_rcept_no"),
                relation_type=str(row["relation_type"]),
                resolution_status=str(row["resolution_status"]),
                target_hint=row.get("target_hint"),
                root_missing_reason=row.get("root_missing_reason"),
            ))
        return tuple(sorted(result, key=lambda row: row.relation_id))

    @staticmethod
    def _covers(base: str, want: str, view: str,
                statement: str | None = None, fy_end: int = 12) -> bool:
        """기준월 `base` 인 보고서가 `want` 기간의 근거인가.

        **비교기간은 표마다 다르다.** 예전에는 두 규칙을 `or` 로 묶어 어느 하나라도
        맞으면 근거로 봤다 — 그래서 IS 질의에 BS 규칙이, BS 질의에 IS 규칙이
        걸렸다 (9차 검수). 2025 1분기 손익계산서가 비교하는 것은 2024년 **1분기**이고,
        2025 1분기 재무상태표가 비교하는 것은 2024년 **연차말**이다.

            BS        현재 중간말 + **직전 회계연도말**
            IS·CI·CF  현재 기간   + **전년 동기**

        `statement` 를 모르면 둘의 합집합을 쓴다 — 놓치는 쪽보다 넓게 잡는 쪽이 낫다.
        `fy_end` 는 그 법인의 회계연도 말 월이다 (12월 고정이 아니다).
        """
        if base == want:
            return True
        if view == "as_filed":
            return False                      # 최초 제출본은 당대 문서만이 근거다
        by, bm = int(base[:4]), int(base[4:6])
        wy, wm = int(want[:4]), int(want[4:6])
        # 직전 회계연도말 — 기준월이 결산월 이후면 같은 해의 결산월이 직전이다
        prev_fy_end = (by - 1, fy_end) if bm <= fy_end else (by, fy_end)
        same_period = (by - 1, bm)
        if statement == "BS":
            return (wy, wm) == prev_fy_end
        if statement in ("IS", "CI", "CF", "SCE"):
            return (wy, wm) == same_period
        return (wy, wm) in (prev_fy_end, same_period)

    # ---------------------------------------------------------------- 회사
    def _alias_registry(self) -> dict:
        """승인 표기 registry. 코퍼스의 **공식 출처에서만** 만든다."""

        if self._company_alias_registry is None:
            from .company_alias import (
                build_alias_registry, load_approved_aliases,
                load_universe_english_names,
            )
            surfaces = list(self._company_surfaces.values())
            by_name = {row[1]: row[0] for row in surfaces}
            approved, held = load_approved_aliases(by_name)
            self._company_alias_registry = build_alias_registry(
                surfaces, approved, load_universe_english_names(by_name))
            self._company_held_aliases = held
        return self._company_alias_registry

    def companies_in_text(self, text: str) -> list[Company]:
        """텍스트 안에 **승인 표기로 등장하는** 회사들.

        질문과 모델 출력이 같은 회사를 다르게 적는 일이 흔하다 — 질문은
        ``LG엔솔``, 출력은 ``LG에너지솔루션``. 두 표현을 각각 해소해 같은
        corp_code 로 모이는지 보려면 **질문 쪽에서도** 회사를 찾아야 한다.

        registry 키를 정규화된 텍스트에서 찾는다. 한 글자 키는 아무 문장에나
        걸리므로 제외한다. **표기 하나가 여러 회사를 가리키면 그 표기는 건너뛴다** —
        질문이 회사를 정하지 못한 것이므로 여기 넣으면 모델의 추측을 근거로
        인정하게 된다.
        """

        self._load_docs()
        from .company_alias import normalize_company_key
        haystack = normalize_company_key(text)
        if not haystack:
            return []
        found: dict[str, str] = {}
        for key, aliases in self._alias_registry().items():
            if len(key) < 2 or key not in haystack:
                continue
            codes = {alias.corp_code: alias.corp_name for alias in aliases}
            # **표기 하나가 여러 회사를 가리키면 그 표기는 회사를 정하지 못한다.**
            # ``삼전`` 처럼 모호한 표기를 여기 포함하면, 모델이 둘 중 하나를 고른
            # 것을 「질문이 뒷받침한다」고 인정해 역질문 결정을 무력화한다.
            if len(codes) != 1:
                continue
            found.update(codes)
        return [Company(code, name) for code, name in found.items()]

    def held_company_candidates(self, name: str) -> list[Company]:
        """**자동 확정하지 않는** 표기의 후보. 확인 역질문에 쓴다.

        ``삼성``·``포스코``·``삼화`` 처럼 다른 실제 회사가 그 표기를 쓸 수 있다고
        판정된 것들이다. ``resolve_company`` 는 이것을 돌려주지 않는다 — 후보가
        하나일 때 호출자가 자동 확정해 버리기 때문이다. 물어볼 선택지가 필요할
        때만 여기서 꺼낸다.
        """

        self._load_docs()
        self._alias_registry()
        from .company_alias import normalize_company_key
        rows = self._company_held_aliases.get(normalize_company_key(name), ())
        seen: dict[str, str] = {}
        for alias in rows:
            seen.setdefault(alias.corp_code, alias.corp_name)
        return [Company(code, corp_name) for code, corp_name in seen.items()]

    def resolve_company(self, name: str) -> list[Company]:
        """회사명 → 후보. **부분 일치이므로 여러 개가 나올 수 있다** — 고르지 않는다.

        승인 표기(법인명·상장 약칭·종목코드)의 정규화 exact match를 **먼저** 본다.
        `현대차`는 거래소 상장 약칭이라 `현대자동차` 안에 연속 부분 문자열로
        들어 있지 않다 — 부분 일치만 쓰면 코퍼스에 있는 회사를 못 찾는다.

        퍼지 매칭은 하지 않는다. 표기가 한 corp_code로만 이어질 때만 확정하고,
        충돌하면 후보를 그대로 돌려줘 호출자가 역질문하게 한다.
        """
        self._load_docs()
        from .company_alias import lookup_alias
        # **확정 불가 판정이 최우선이다.** 부분 문자열 fallback 이 이것을 우회하면
        # ``현대중공업`` 처럼 「다른 대상 가능성」이 붙은 표기가 조용히 확정된다.
        # ``_company_held_aliases`` 는 registry 를 만들 때 채워지므로 **먼저**
        # 만들어야 한다 — 순서를 바꾸면 첫 호출에서 검사가 빈 사전을 본다.
        registry = self._alias_registry()
        from .company_alias import normalize_company_key
        if normalize_company_key(name) in self._company_held_aliases:
            return []
        aliases = lookup_alias(registry, name)
        if aliases:
            codes: dict[str, str] = {}
            for alias in aliases:
                codes.setdefault(alias.corp_code, alias.corp_name)
            return [Company(code, corp_name)
                    for code, corp_name in codes.items()]
        key = name.replace(" ", "")
        return sorted((Company(c, n) for c, n in self._companies.items()
                       if key and key in n.replace(" ", "")), key=lambda c: c.corp_name)

    # ---------------------------------------------------------------- 검색 공급
    def _load_chunk_evidence_index(self) -> tuple[dict[str, bytes], dict[str, str]]:
        """Chunk Evidence를 한 번만 훑어 compact fingerprint index로 만든다.

        일반 ``get_evidence_many``를 2,000개씩 호출하면 각 배치가 247만 Evidence와
        semantic Parquet의 predicate scan을 다시 수행한다. 이 전용 경로는 큰 raw/safe
        문자열을 보유하지 않고 32-byte fingerprint만 남기며, foreign semantic FK도
        최소 ID 열을 한 번씩 스트리밍해 동일한 cross-kind fail-closed 계약을 지킨다.
        """
        if (self._chunk_evidence_fingerprints is not None
                and self._chunk_evidence_source_owners is not None):
            return (self._chunk_evidence_fingerprints,
                    self._chunk_evidence_source_owners)
        if not _schema_at_least(self.schema_version, (1, 9)):
            raise NotReadyError("Chunk Evidence 검증은 schema 1.9+ artifact가 필요합니다")

        artifact = self.root / "evidence.parquet"
        columns = list(_EVIDENCE_FINGERPRINT_FIELDS)
        if not artifact.exists():
            raise EvidenceIntegrityError(f"Evidence 산출물이 없습니다: {artifact}")
        parquet = pq.ParquetFile(artifact)
        present = set(parquet.schema_arrow.names)
        missing = sorted(set(columns) - present)
        if missing:
            raise EvidenceIntegrityError(
                "Chunk Evidence 필수 컬럼 누락: " + ", ".join(missing))
        kind_index = parquet.schema_arrow.names.index("kind")
        fingerprints: dict[str, bytes] = {}
        for group_no in range(parquet.metadata.num_row_groups):
            statistics = parquet.metadata.row_group(group_no).column(kind_index).statistics
            if statistics is not None and statistics.has_min_max:
                minimum, maximum = statistics.min, statistics.max
                if isinstance(minimum, bytes):
                    minimum = minimum.decode("utf-8")
                if isinstance(maximum, bytes):
                    maximum = maximum.decode("utf-8")
                if not (minimum <= "chunk_text" <= maximum):
                    continue
            try:
                table = parquet.read_row_group(
                    group_no, columns=columns, use_threads=False)
            except Exception as exc:
                raise EvidenceIntegrityError(
                    f"Chunk Evidence row-group 조회 실패: {group_no}") from exc
            values = table.to_pydict()
            for index, kind in enumerate(values["kind"]):
                if kind != "chunk_text":
                    continue
                row = {name: values[name][index] for name in columns}
                evidence_id = row.get("evidence_id")
                doc_id, source_file_id = row.get("doc_id"), row.get("source_file_id")
                if (not isinstance(evidence_id, str) or not evidence_id
                        or not isinstance(doc_id, str) or not doc_id
                        or not isinstance(source_file_id, str) or not source_file_id):
                    raise EvidenceIntegrityError("Chunk Evidence ID/ownership 키가 없습니다")
                if evidence_id in fingerprints:
                    raise EvidenceIntegrityError(
                        f"Chunk Evidence ID가 복수 행입니다: {evidence_id}")
                if row.get("build_id") != self.build_id:
                    raise EvidenceIntegrityError(
                        f"Chunk Evidence build_id 불일치: {evidence_id}")
                try:
                    locator = canonical_locator(row.get("locator"))
                    _checked_as_of(row.get("rcept_dt"))
                except (CanonicalLocatorError, TypeError, ValueError) as exc:
                    raise EvidenceIntegrityError(
                        f"Chunk Evidence locator/date 오류: {evidence_id}") from exc
                if (row.get("table_locator") is not None
                        or row.get("logical_row") is not None
                        or row.get("logical_col") is not None):
                    raise EvidenceIntegrityError(
                        f"Chunk Evidence 좌표는 null이어야 합니다: {evidence_id}")
                raw, safe = row.get("excerpt_raw"), row.get("excerpt_safe")
                if not isinstance(raw, str) or not isinstance(safe, str):
                    raise EvidenceIntegrityError(
                        f"Chunk Evidence raw/safe 타입 오류: {evidence_id}")
                raw_hash = calculate_excerpt_hash(raw)
                if (row.get("excerpt_hash") != raw_hash
                        or calculate_evidence_id(
                            source_file_id, "chunk_text", locator, raw_hash)
                        != evidence_id):
                    raise EvidenceIntegrityError(
                        f"Chunk Evidence hash/ID 불일치: {evidence_id}")
                flags, pii_types = row.get("security_flags"), row.get("pii_types")
                if (not isinstance(flags, list) or not isinstance(pii_types, list)
                        or flags != sorted(set(flags))
                        or pii_types != sorted(set(pii_types))):
                    raise EvidenceIntegrityError(
                        f"Chunk Evidence 보안 메타 정렬/유일성 오류: {evidence_id}")
                if (row.get("raw_access") != "restricted"
                        or row.get("extraction_method") != "section_chunk"
                        or row.get("extraction_status") != "ok"
                        or row.get("security_policy_version") not in (
                            {SECURITY_POLICY_VERSION}
                            | _READ_COMPAT_SECURITY_POLICIES)
                        or row.get("evidence_policy_version") != EVIDENCE_POLICY_VERSION
                        or (safe.count(PROMPT_DATA_BEGIN),
                            safe.count(PROMPT_DATA_END)) != (1, 1)):
                    raise EvidenceIntegrityError(
                        f"Chunk Evidence 정책 계약 오류: {evidence_id}")
                fingerprints[evidence_id] = _evidence_fingerprint(row)

        # Chunk kind-bound ID를 Fact/Field/Correction semantic FK로 재사용하는 변조를
        # 큰 본문 없이 ID 열 한 번의 스트리밍으로 닫는다.
        if fingerprints:
            foreign_columns = (
                (self.root / "facts.parquet", ("evidence_id",)),
                (self.root / "fields.parquet", ("evidence_id",)),
                (self.root / "correction_items.parquet",
                 ("before_evidence_id", "after_evidence_id")),
            )
            for foreign_path, names in foreign_columns:
                foreign_present = _artifact_columns(foreign_path)
                foreign_missing = sorted(set(names) - foreign_present)
                if foreign_missing:
                    raise EvidenceIntegrityError(
                        f"{foreign_path.name} Evidence FK 컬럼 누락: "
                        + ", ".join(foreign_missing))
                for ref in _rows(foreign_path, list(names), batch=200_000):
                    for name in names:
                        value = ref.get(name)
                        if value in fingerprints:
                            raise EvidenceIntegrityError(
                                "chunk_text Evidence가 다른 semantic 종류에도 참조됩니다: "
                                f"{value} ({foreign_path.name})")
            for support_path in (
                    self.root / "event_observations.parquet",
                    self.root / "relations.parquet"):
                support_column = "supporting_evidence_ids"
                support_present = _artifact_columns(support_path)
                if support_column not in support_present:
                    raise EvidenceIntegrityError(
                        f"{support_path.name} support Evidence FK 컬럼 누락")
                for ref in _rows(support_path, [support_column], batch=20_000):
                    support_ids = ref.get(support_column)
                    if not isinstance(support_ids, list):
                        raise EvidenceIntegrityError(
                            f"{support_path.name} support Evidence FK 타입 오류")
                    overlap = next(
                        (value for value in support_ids if value in fingerprints), None)
                    if overlap is not None:
                        raise EvidenceIntegrityError(
                            "chunk_text Evidence가 Event/Relation support에도 참조됩니다: "
                            f"{overlap} ({support_path.name})")

        source_path = self.root / "source_files.parquet"
        source_columns = ["build_id", "source_file_id", "doc_id"]
        source_present = _artifact_columns(source_path)
        source_missing = sorted(set(source_columns) - source_present)
        if source_missing:
            raise EvidenceIntegrityError(
                "SourceFile ownership 컬럼 누락: " + ", ".join(source_missing))
        owners: dict[str, str] = {}
        for source in _rows(source_path, source_columns, batch=20_000):
            source_id, owner = source.get("source_file_id"), source.get("doc_id")
            if (source.get("build_id") != self.build_id
                    or not isinstance(source_id, str) or not source_id
                    or not isinstance(owner, str) or not owner):
                raise EvidenceIntegrityError("SourceFile Chunk Evidence ownership 오류")
            if source_id in owners:
                raise EvidenceIntegrityError(
                    f"SourceFile FK가 복수 행입니다: {source_id}")
            owners[source_id] = owner

        # Filter를 적용하기 전에 전체 Chunk의 semantic identity/PART topology를 한 번
        # 검증한다. 그렇지 않으면 index_eligible=False 행에 심은 duplicate FK나 누락 PART가
        # 기본 검색 공급 경로에서 가려져 정상 citation으로 보일 수 있다.
        self._load_docs()
        chunk_path = self.root / "chunks.parquet"
        chunk_columns = [
            "build_id", "doc_id", "source_file_id", "block_id", "parent_id",
            "locator", "part_no", "n_parts", "rcept_dt", "evidence_id",
        ]
        chunk_present = _artifact_columns(chunk_path)
        chunk_missing = sorted(set(chunk_columns) - chunk_present)
        if chunk_missing:
            raise EvidenceIntegrityError(
                "Chunk Evidence semantic 컬럼 누락: " + ", ".join(chunk_missing))
        semantic_ids: set[str] = set()
        part_groups: dict[tuple[str, str, str], list[int]] = {}
        for chunk in _rows(chunk_path, chunk_columns, batch=20_000):
            evidence_id = chunk.get("evidence_id")
            doc_id, source_id = chunk.get("doc_id"), chunk.get("source_file_id")
            locator = chunk.get("locator")
            part_no, n_parts = chunk.get("part_no"), chunk.get("n_parts")
            document = self._documents.get(str(doc_id))
            try:
                canonical_chunk_locator = canonical_locator(locator)
            except (CanonicalLocatorError, TypeError) as exc:
                raise EvidenceIntegrityError(
                    f"Chunk semantic locator 오류: {chunk.get('block_id')}") from exc
            if (chunk.get("build_id") != self.build_id or document is None
                    or str(doc_id) in self._duplicate_document_ids
                    or not isinstance(document.corp_name, str)
                    or not document.corp_name.strip()
                    or not isinstance(document.listed_name, str)
                    or not document.listed_name.strip()
                    or not isinstance(source_id, str)
                    or owners.get(source_id) != doc_id
                    or document.rcept_dt != chunk.get("rcept_dt")
                    or type(part_no) is not int or type(n_parts) is not int
                    or not (0 <= part_no < n_parts)
                    or not canonical_chunk_locator.endswith(f"/PART[{part_no}]")
                    or chunk.get("block_id") != calculate_block_id(
                        str(doc_id), source_id, canonical_chunk_locator)
                    or not isinstance(evidence_id, str) or not evidence_id):
                raise EvidenceIntegrityError(
                    f"Chunk ownership/PART/block_id 오류: {chunk.get('block_id')}")
            if evidence_id in semantic_ids:
                raise EvidenceIntegrityError(
                    f"Chunk Evidence semantic FK 중복: {evidence_id}")
            if evidence_id not in fingerprints:
                raise EvidenceIntegrityError(
                    f"Chunk Evidence FK가 저장 Evidence에 없습니다: {evidence_id}")
            semantic_ids.add(evidence_id)
            group_key = (str(doc_id), source_id, str(chunk.get("parent_id")))
            stats = part_groups.setdefault(group_key, [n_parts, 0, 0, 0])
            if stats[0] != n_parts:
                raise EvidenceIntegrityError(
                    f"Chunk parent n_parts 충돌: {chunk.get('parent_id')}")
            stats[1] += 1
            stats[2] += part_no
            stats[3] += part_no * part_no
        for n_parts, count, part_sum, square_sum in part_groups.values():
            if (count != n_parts
                    or part_sum != n_parts * (n_parts - 1) // 2
                    or square_sum
                    != n_parts * (n_parts - 1) * (2 * n_parts - 1) // 6):
                raise EvidenceIntegrityError("Chunk parent별 PART 완전성 오류")
        if semantic_ids != set(fingerprints):
            raise EvidenceIntegrityError(
                "chunk_text Evidence와 Chunk semantic FK 집합이 다릅니다")

        self._evidence_source_owner_cache.update(owners)
        self._chunk_evidence_fingerprints = fingerprints
        self._chunk_evidence_source_owners = owners
        return fingerprints, owners

    def chunks(self, *, corp_code: str | None = None, doc_group: str | None = None,
               doc_id: str | None = None,
               index_eligible_only: bool = True,
               projection: Literal["search", "llm"] = "search",
               verify_evidence: bool = False) -> Iterator[ChunkRow]:
        """검색 인덱스에 넣을 조각을 **스트리밍으로** 공급한다.

        여기서 `as_of` 를 받지 않는 것은 의도다. 인덱스는 한 번 만들고 여러 시점으로
        질의하므로, **시점 필터는 인덱스 메타데이터(`rcept_dt`)로 걸어야 한다.**
        순위를 매긴 뒤에 거르면 상위 k 가 전부 미래 문서일 때 결과가 0 이 된다.

        `index_eligible_only` 기본값이 True 다 — 100자 미만 조각(「해당사항 없음」 등)은
        검색 노이즈다. 다만 canonical 에는 남아 있어 필요하면 끌 수 있다.
        """
        if projection not in ("search", "llm"):
            raise ValueError("projection은 'search' 또는 'llm'이어야 합니다")
        verify_evidence = _checked_bool_flag("verify_evidence", verify_evidence)
        self._load_docs()
        artifact = self.root / "chunks.parquet"
        base_columns = [
            "block_id", "parent_id", "doc_id", "source_file_id", "rcept_dt",
            "path", "locator", "n_chars", "index_eligible", "over_budget",
        ]
        present = _artifact_columns(artifact)
        modern_projection = _schema_at_least(self.schema_version, (1, 8))
        modern_evidence = _schema_at_least(self.schema_version, (1, 9))
        if modern_evidence:
            base_columns.extend(("evidence_id", "part_no", "n_parts"))
        elif verify_evidence:
            raise NotReadyError("Chunk Evidence 검증은 schema 1.9+ artifact가 필요합니다")
        stored_safe = {"text_search", "text_prompt_safe", "security_flags", "pii_types",
                       "security_policy_version"}
        if stored_safe.issubset(present):
            cols = base_columns + sorted(stored_safe)
            if modern_projection:
                if "text" not in present:
                    raise ReadSecurityError("schema 1.8+ Chunk raw text가 없습니다")
                cols.append("text")
            legacy_projection = False
        elif stored_safe & present:
            missing = sorted(stored_safe - present)
            raise ReadSecurityError(
                f"Chunk safe projection 컬럼이 부분 손상되었습니다: {', '.join(missing)}")
        else:
            if modern_projection:
                raise ReadSecurityError("schema 1.8+ Chunk safe projection 컬럼이 없습니다")
            # 1.2 호환. raw를 반환하지 않고 동일한 정책을 on-the-fly 적용한다.
            cols = base_columns + ["text"]
            legacy_projection = True
        evidence_index: dict[str, bytes] = {}
        source_owners: dict[str, str] = {}
        seen_evidence_ids: set[str] = set()
        if verify_evidence:
            evidence_index, source_owners = self._load_chunk_evidence_index()

        parquet_filters = None
        if doc_id is not None:
            meta = self._documents.get(doc_id)
            if meta is None:
                return
            if (corp_code is not None and meta.corp_code != corp_code
                    or doc_group is not None and meta.doc_group != doc_group):
                return
            parquet_filters = [("doc_id", "=", doc_id)]

        # raw/search/prompt-safe 세 문자열을 함께 읽으므로 큰 batch는 Python 문자열
        # allocator의 순간 RSS를 키운다. Parquet 물리 row-group과 같은 2,000행으로
        # 제한해 4 GiB serving 환경에서도 전량 verified scan에 여유를 둔다.
        row_source = (
            _rows_for_exact_string_by_row_group(
                artifact, cols, selector="doc_id", value=doc_id, batch=2_000)
            if doc_id is not None
            else _rows(artifact, cols, batch=2_000, filters=parquet_filters)
        )
        for r in row_source:
            meta = self._doc_meta.get(r["doc_id"])
            if meta is None:
                continue
            code, cname, group = meta
            if doc_id is not None and r["doc_id"] != doc_id:
                continue
            if corp_code is not None and code != corp_code:
                continue
            if doc_group is not None and group != doc_group:
                continue
            if index_eligible_only and not r["index_eligible"]:
                continue
            expected_projection = None
            raw_chunk: str | None = None
            if legacy_projection:
                safe = project_chunk_text(str(r["text"] or ""), (cname,))
                text_search = safe.text_search
                text_prompt_safe = safe.text_prompt_safe
                flags = safe.security_flags
                pii_types = safe.pii_types
                policy = safe.security_policy_version
            else:
                text_search = r["text_search"]
                text_prompt_safe = r["text_prompt_safe"]
                if not isinstance(text_search, str) or not isinstance(text_prompt_safe, str):
                    raise ReadSecurityError(
                        f"Chunk safe projection이 없어 raw로 대체할 수 없습니다: {r['block_id']}")
                flags = tuple(r["security_flags"] or ())
                pii_types = tuple(r["pii_types"] or ())
                policy = str(r["security_policy_version"] or "")
                if not policy:
                    raise ReadSecurityError(
                        f"Chunk security_policy_version이 없습니다: {r['block_id']}")
                if modern_projection:
                    raw_chunk = r.get("text")
                    if not isinstance(raw_chunk, str):
                        raise ReadSecurityError(
                            f"Chunk raw text가 문자열이 아닙니다: {r['block_id']}")
                    doc_info = self._documents.get(r["doc_id"])
                    names = (cname, doc_info.listed_name if doc_info else "")
                    expected = project_chunk_text(
                        raw_chunk, names, security_policy_version=policy)
                    if policy not in ({SECURITY_POLICY_VERSION}
                                      | _READ_COMPAT_SECURITY_POLICIES):
                        raise ReadSecurityError(
                            f"지원하지 않는 Chunk security policy: {policy}")
                    expected_projection = expected
                    if (text_search != expected.text_search
                            or text_prompt_safe != expected.text_prompt_safe
                            or flags != expected.security_flags
                            or pii_types != expected.pii_types):
                        raise ReadSecurityError(
                            f"Chunk 저장 safe projection 재검증 실패: {r['block_id']}")
                if (text_prompt_safe.count(PROMPT_DATA_BEGIN),
                        text_prompt_safe.count(PROMPT_DATA_END)) != (1, 1):
                    raise ReadSecurityError(
                        f"Chunk prompt-safe 경계가 손상되었습니다: {r['block_id']}")
            evidence_id = r.get("evidence_id")
            evidence_status = ("unverified" if evidence_id else "unavailable")
            if verify_evidence:
                if expected_projection is None or raw_chunk is None:
                    raise EvidenceIntegrityError("Chunk Evidence 재생성 문맥이 없습니다")
                if not isinstance(evidence_id, str) or not evidence_id:
                    evidence_status = "invalid"
                else:
                    if evidence_id in seen_evidence_ids:
                        raise EvidenceIntegrityError(
                            f"Chunk Evidence semantic FK가 복수 행입니다: {evidence_id}")
                    seen_evidence_ids.add(evidence_id)
                    part_no, n_parts = r.get("part_no"), r.get("n_parts")
                    if (type(part_no) is not int or type(n_parts) is not int
                            or part_no < 0 or n_parts <= 0 or part_no >= n_parts
                            or not isinstance(r.get("locator"), str)
                            or not r["locator"].endswith(f"/PART[{part_no}]")):
                        raise EvidenceIntegrityError(
                            f"Chunk PART/n_parts 계약 오류: {evidence_id}")
                    doc_info = self._documents.get(r["doc_id"])
                    if (doc_info is None or doc_info.rcept_dt != r["rcept_dt"]
                            or source_owners.get(r["source_file_id"]) != r["doc_id"]):
                        raise EvidenceIntegrityError(
                            f"Chunk Evidence Document/Source ownership 오류: {evidence_id}")
                    raw_hash = calculate_excerpt_hash(raw_chunk)
                    calculated_id = calculate_evidence_id(
                        r["source_file_id"], "chunk_text", r["locator"], raw_hash)
                    if calculated_id != evidence_id:
                        raise EvidenceIntegrityError(
                            f"Chunk Evidence semantic FK 불일치: {evidence_id}")
                    candidate: dict[str, object] = {
                        "build_id": self.build_id,
                        "evidence_id": calculated_id,
                        "doc_id": r["doc_id"],
                        "source_file_id": r["source_file_id"],
                        "kind": "chunk_text",
                        "locator": canonical_locator(r["locator"]),
                        "table_locator": None,
                        "logical_row": None,
                        "logical_col": None,
                        "excerpt_raw": raw_chunk,
                        "excerpt_safe": expected_projection.text_prompt_safe,
                        "excerpt_hash": raw_hash,
                        "raw_access": "restricted",
                        "extraction_method": "section_chunk",
                        "extraction_status": "ok",
                        # make_evidence는 projection 메타를 set→정렬해 Evidence에 쓴다.
                        # Chunk 자체는 project_chunk_text의 policy order를 보존한다.
                        "security_flags": sorted(set(expected_projection.security_flags)),
                        "pii_types": sorted(set(expected_projection.pii_types)),
                        "security_policy_version": expected_projection.security_policy_version,
                        "evidence_policy_version": EVIDENCE_POLICY_VERSION,
                        "rcept_dt": r["rcept_dt"],
                    }
                    stored_fingerprint = evidence_index.get(evidence_id)
                    if stored_fingerprint is None:
                        evidence_status = "invalid"
                    elif stored_fingerprint != _evidence_fingerprint(candidate):
                        raise EvidenceIntegrityError(
                            f"Chunk Evidence exact 재생성 불일치: {evidence_id}")
                    else:
                        evidence_status = "verified"
            yield ChunkRow(
                chunk_id=r["block_id"], section_id=r["parent_id"],
                doc_id=r["doc_id"], source_file_id=r["source_file_id"],
                rcept_dt=r["rcept_dt"],
                corp_code=code, corp_name=cname, doc_group=group,
                path=r["path"], locator=r["locator"],
                text=text_prompt_safe if projection == "llm" else text_search,
                text_prompt_safe=text_prompt_safe, text_projection=projection,
                security_flags=tuple(flags), pii_types=tuple(pii_types),
                security_policy_version=policy,
                n_chars=r["n_chars"], index_eligible=r["index_eligible"],
                over_budget=r["over_budget"], evidence_id=evidence_id,
                evidence_status=evidence_status)

    # ---------------------------------------------------------------- 근거
    @staticmethod
    def _section_match_mask(
            table: pa.Table, *, section_id: str | None,
            doc_id: str | None, locator: str | None,
            source_file_id: str | None) -> pa.Array | pa.ChunkedArray:
        """Section point lookup의 Arrow mask를 만든다.

        이 helper는 **metadata-only row group**과 본문 row group 양쪽에서 쓴다.
        따라서 후보를 고르려고 ``text`` 열을 Python 객체로 만들지 않는다.
        """
        if section_id is not None:
            return pc.equal(table["block_id"], pa.scalar(section_id))

        mask = pc.and_(
            pc.equal(table["doc_id"], pa.scalar(doc_id)),
            pc.equal(table["locator"], pa.scalar(locator)))
        if source_file_id is not None:
            mask = pc.and_(
                mask,
                pc.equal(table["source_file_id"], pa.scalar(source_file_id)))
        return mask

    def _cached_section_row_group(self, section_id: str) -> int | None | _Unset:
        """Section ID → row group의 bounded cache. ``None``도 negative cache다."""
        if section_id not in self._section_row_group_cache:
            return UNSET
        self._section_row_group_cache.move_to_end(section_id)
        return self._section_row_group_cache[section_id]

    def _cache_section_row_group(self, section_id: str, row_group: int | None) -> None:
        self._section_row_group_cache[section_id] = row_group
        self._section_row_group_cache.move_to_end(section_id)
        while len(self._section_row_group_cache) > self._section_row_group_cache_limit:
            self._section_row_group_cache.popitem(last=False)

    def read_section(self, doc_id: str | None = None, locator: str | None = None, *,
                     section_id: str | None = None,
                     source_file_id: str | None = None,
                     include_restricted_raw: bool = False) -> SectionText | None:
        """검색 조각의 부모 Section을 기본 prompt-safe projection으로 돌려준다.

        검색 결과에서는 ``read_section(section_id=hit.section_id)`` 를 쓴다. 이 ID는
        ``doc_id + source_file_id + locator`` 로 만들어져 본문과 첨부의 locator가 같아도
        충돌하지 않는다.

        기존 ``read_section(doc_id, locator)`` 호출도 유일한 경우에는 계속 동작한다.
        다만 여러 원문 파일에서 같은 locator가 발견되면 첫 행을 임의로 고르지 않고
        :class:`AmbiguousSectionError` 를 낸다. 이때 ``source_file_id`` 를 함께 넘기거나
        ``section_id`` 로 직접 조회해야 한다.
        """
        include_restricted_raw = _checked_bool_flag(
            "include_restricted_raw", include_restricted_raw)
        if section_id is not None:
            if doc_id is not None or locator is not None or source_file_id is not None:
                raise ValueError(
                    "section_id는 doc_id/locator/source_file_id와 함께 지정할 수 없습니다")
        elif doc_id is None or locator is None:
            raise TypeError(
                "section_id 또는 doc_id와 locator를 지정해야 합니다")

        self._load_docs()
        cols = ["block_id", "doc_id", "source_file_id", "locator", "rcept_dt",
                "path", "text", "n_chars", "n_tables"]
        metadata_cols = ["block_id", "doc_id", "source_file_id", "locator"]
        artifact = self.root / "sections.parquet"
        parquet = pq.ParquetFile(artifact)

        # `sections.parquet`은 build 순서이고 block_id는 해시다. pq.read_table(...,
        # filters=...)는 row group pruning을 못 하면 모든 `text` page를 Arrow에
        # 물질화한다. 먼저 작은 selector 열만 row group 단위로 읽고, 매치가 난
        # row group에서만 text를 연다. 어느 한 row group의 크기를 넘지 않아 4GB
        # 환경에서도 point lookup이 전체 Section 본문을 누적하지 않는다.
        if section_id is not None:
            cached_group = self._cached_section_row_group(section_id)
            if cached_group is None:
                return None
            if cached_group is UNSET:
                candidate_groups: list[int] = []
                for group_index in range(parquet.metadata.num_row_groups):
                    metadata = parquet.read_row_group(
                        group_index, columns=["block_id"])
                    mask = self._section_match_mask(
                        metadata, section_id=section_id, doc_id=None,
                        locator=None, source_file_id=None)
                    if pc.any(mask).as_py():
                        candidate_groups.append(group_index)
                        # block_id is an immutable primary key. Keep legacy
                        # first-match behavior if a damaged old artifact has a
                        # duplicate, but do not read unrelated text row groups.
                        break
                located_group = candidate_groups[0] if candidate_groups else None
                self._cache_section_row_group(section_id, located_group)
            else:
                candidate_groups = [cached_group]
        else:
            # doc_id+locator legacy lookup must still inspect every metadata
            # group to detect ambiguity across source files. It never reads text
            # for a non-matching group.
            candidate_groups = []
            for group_index in range(parquet.metadata.num_row_groups):
                metadata = parquet.read_row_group(group_index, columns=metadata_cols)
                mask = self._section_match_mask(
                    metadata, section_id=None, doc_id=doc_id, locator=locator,
                    source_file_id=source_file_id)
                if pc.any(mask).as_py():
                    candidate_groups.append(group_index)

        found: SectionText | None = None
        for group_index in candidate_groups:
            # Arrow Table 상태에서는 해당 row group의 text가 Arrow buffer일 뿐이다.
            # 아래 filter 뒤의 to_pylist()는 실제 후보(보통 한 행)만 Python dict로
            # 바꾼다. 이전의 _rows(..., batch=20_000)처럼 12만 본문을 만들지 않는다.
            table = parquet.read_row_group(group_index, columns=cols)
            mask = self._section_match_mask(
                table, section_id=section_id, doc_id=doc_id, locator=locator,
                source_file_id=source_file_id)
            for r in table.filter(mask).to_pylist():
                meta = self._doc_meta.get(r["doc_id"])
                names = (meta[1],) if meta is not None else ()
                projection = project_chunk_text(str(r["text"] or ""), names)
                row = SectionText(
                    section_id=r["block_id"], doc_id=r["doc_id"],
                    source_file_id=r["source_file_id"], rcept_dt=r["rcept_dt"],
                    path=r["path"], locator=r["locator"],
                    text=(r["text"] if include_restricted_raw
                          else projection.text_prompt_safe),
                    text_projection=("restricted_raw" if include_restricted_raw
                                     else "prompt_safe"),
                    security_flags=projection.security_flags,
                    pii_types=projection.pii_types,
                    security_policy_version=projection.security_policy_version,
                    n_chars=r["n_chars"], n_tables=r["n_tables"])

                # section_id와 source_file_id를 쓴 조회는 ID 불변식상 유일하다. 모호성
                # 검사가 필요한 것은 원문 파일을 생략한 레거시 doc_id+locator 조회다.
                if section_id is not None or source_file_id is not None:
                    return row
                if found is not None:
                    raise AmbiguousSectionError(
                        f"doc_id={doc_id!r}, locator={locator!r}가 여러 원문 파일에 "
                        "존재합니다. section_id 또는 source_file_id를 지정하세요")
                found = row
        return found
