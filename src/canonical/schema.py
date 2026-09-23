"""canonical 스키마 — 전 문서군 공통 계약.

지금까지 문서군마다 산출물 스키마가 달랐다(공통 필드 6개, 이름도 `rcept_dt` vs
`disclosed_at` 으로 갈림). 이는 D1 「canonical 은 하나로 통합하고 조회만 3분할」에
어긋난다. 이 모듈이 **저장 계약을 하나로 못 박는다.**

## 설계 결정

**단일 `Block` 테이블은 쓰지 않는다.** 실측 Cell 40,004,925 행으로 나머지(Section 119,161 +
Field 1,640,405)의 20배가 넘는다. 한 테이블에 두면 Field 조회마다 Cell 을 스캔한다.
Parquet 은 컬럼 지향이라 sparse 컬럼도 낭비다.

대신 **엔티티는 나누되 앞머리 8컬럼을 예외 없이 강제한다.** 문제의 본질은 테이블 수가
아니라 공통 축의 부재였다.

## ID 와 locator 가 스키마 형태보다 중요하다

컬럼은 나중에 더할 수 있지만 ID 가 바뀌면 전 산출물을 다시 만들어야 한다.

```text
source_file_id = sha1(NFC 상대경로)[:16]
block_id       = sha1(doc_id | source_file_id | locator)[:16]
locator        = "SECTION-1[2]/SECTION-2[0]/TABLE[1]/TR[6]/TD[2]"
```

`locator` 는 같은 태그 형제 중 0-기반 순번을 이은 것이라 파서를 고쳐도 원문이 같으면
동일하다. **순번 기반 `#018` 을 대체한다** — 파서가 섹션을 하나 더 잡으면 그 뒤 ID 가
전부 밀리는 문제가 사라진다.

부수 효과로 「같은 path 가 한 문서에 여러 번」(F-01 청크의 7%) 문제도 해소된다.
path 는 같아도 locator 는 다르기 때문이다.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

__all__ = [
    "SCHEMA_VERSION", "LOCATABLE_COLUMNS", "COMMON_COLUMNS", "LOCATABLE_ENTITIES",
    "is_locatable",
    "Run", "SourceFile", "Document", "Section", "Chunk", "Field", "Cell",
    "Evidence",
    "EventIdentity", "EventObservation", "CorrectionItem", "Relation", "Fact",
    "source_file_id", "block_id", "locator_of", "child_locator", "as_row",
]

# **물리 스키마가 달라졌으므로 같은 이름을 재사용하지 않는다** (7차 검수).
#   1.2  Document.fact_unsupported_scope 추가 · fact_extract_status 에 `extractor_failed`
#        EventObservation.supersedes_rcept_no → previous_observation_rcept_no
#   1.3  SourceFile.normalized_path_key · actual_relpath 추가. 기존 relpath/ID 의미는 유지
#   1.4  Chunk/Field에 raw를 보존한 채 PII·prompt-safe 파생 projection과 정책 버전 추가
#   1.5  first-class Evidence 및 Fact/Field/CorrectionItem evidence FK 추가. semantic ID 불변
#   1.6  Relation endpoint를 canonical Document FK로 전환하고 receipt locator를 분리.
#        EventIdentity에 안정 식별 fingerprint/status/resolver를 추가
#   1.7  PDF/viewer HTML의 primary·alternate 선택, source별 parsing coverage,
#        문서 단위 typed cross-check 결과를 손실 없이 보존
#   1.8  CorrectionItem side locator/semantic 좌표 및 Field safe/semantic 좌표 추가.
#        Evidence ID는 source+kind+locator+excerpt_hash 정책 1.1로 종류까지 결합
#   1.9  Chunk 원문도 first-class Evidence로 결합하고 EventObservation/Relation에
#        검증 가능한 Field/Correction Evidence support와 typed limitation 추가.
SCHEMA_VERSION = "1.9"

#: **1.0 → 1.1 마이그레이션** (2026-08-08 동결)
#:
#: 동결 조건(검수 H 게이트) 10항목을 전부 기계로 확인하고 승격했다.
#: 재현: `tests/verify_ids.py`(ID 25항목) · `tests/verify_gate.py`(게이트 10항목) ·
#: `tests/test_*.py`(최소 반례 7종).
#:
#: | 변경 | 이유 |
#: |---|---|
#: | ID 64bit → **128bit** | Cell 4천만 행에서 64bit 충돌 확률이 무시할 수준이 아니다 (S-02) |
#: | `COMMON_COLUMNS` → `LOCATABLE_COLUMNS` | 원문 위치가 없는 엔티티에 8컬럼을 강요하지 않는다 (S-03) |
#: | `Chunk` 추가 | Section(구조)/Chunk(검색) 분리 (D2 · P0-7) |
#: | `Field.table_locator`·`label_locators` | 라벨-값 대응의 근거 (C-02) |
#: | `Run.code_hash` | 버전 문자열을 손으로 올리는 것을 신뢰하지 않는다 (C-04) |
#: | `EventIdentity`·`EventObservation` | 문서 1건 = 사건 1건이 아니다 (S-09) |
#: | `Relation.candidate_ids`·`root_missing_reason` | 판정만 남기면 재검토 불가 (R-01·R-02) |
#: | `CorrectionItem` 분류축 2분 | `diff_kind` + `before/after_kind` (S-10) |
#: | `Field.occurrence`·`table_locator`·`label_locators` | 반복 라벨·근거 보존 (S-08) |
#:
#: 산출물은 전량 재생성한다. ID 폭이 바뀌므로 부분 이관은 불가능하다.

#: **원문에 위치가 있는** 엔티티가 이 8개로 시작한다.
#: 예전 이름은 `COMMON_COLUMNS` 였는데, `Run`·`Document` 처럼 위치가 없는 엔티티에까지
#: 공통을 요구하는 것처럼 읽혔다. 공통 축이 필요한 것은 **locatable 엔티티뿐**이다.
LOCATABLE_COLUMNS = (
    "build_id", "doc_id", "source_file_id", "block_id",
    "parent_id", "path", "order", "locator",
)

#: 하위 호환 별칭
COMMON_COLUMNS = LOCATABLE_COLUMNS

#: locatable 엔티티 이름. 여기 없는 엔티티에는 8컬럼을 강요하지 않는다.
LOCATABLE_ENTITIES = frozenset({
    "Section", "Chunk", "Field", "Cell", "CorrectionItem", "Fact",
})


def is_locatable(cls: type) -> bool:
    return cls.__name__ in LOCATABLE_ENTITIES


# ------------------------------------------------------------------- ID · locator


#: ID 폭(hex 자리수). 128bit = 32자.
#: 64bit(16자)로는 Cell 4천만 행 기준 충돌 확률이 약 4e-5 로 무시할 수준이 아니다.
#: ID 가 겹치면 서로 다른 원문 위치가 같은 행으로 접혀 조용히 사라진다 (S-02).
ID_HEX = 32


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:ID_HEX]


def source_file_id(relpath: str) -> str:
    """파일 경로 기반 안정 ID.

    schema 1.3에서도 기존 산출물과의 조인 호환성을 위해 정책을 바꾸지 않는다.
    입력을 NFC로 정규화한 값의 해시이며 파일 내용 해시는 ID가 아니라 SourceFile.sha256과
    Run.corpus_inventory_hash가 담당한다.
    """
    return _sha1(unicodedata.normalize("NFC", relpath))


def block_id(doc_id: str, src_id: str, locator: str) -> str:
    return _sha1(f"{doc_id}|{src_id}|{locator}")


def locator_of(ancestors: list[tuple[str, int]]) -> str:
    """`[("SECTION-1", 2), ("TABLE", 1)]` → `"SECTION-1[2]/TABLE[1]"`."""
    return "/".join(f"{tag}[{idx}]" for tag, idx in ancestors)


def child_locator(parent: str, tag: str, index: int) -> str:
    seg = f"{tag}[{index}]"
    return f"{parent}/{seg}" if parent else seg


# ----------------------------------------------------------------------- 엔티티


@dataclass(frozen=True)
class Run:
    """빌드 1회. 모든 산출물이 이 `build_id` 를 갖고, 서버는 불일치 시 기동 실패시킨다."""

    build_id: str
    schema_version: str
    started_at: str
    corpus_inventory_hash: str
    manifest_hash: str
    sanitizer_version: str
    #: PII masking + prompt/active-content projection 정책. raw sanitizer 버전과 별개다.
    security_policy_version: str
    parser_version: str
    chunker_version: str
    #: 파이프라인 소스코드 전체의 해시. 버전 문자열을 손으로 올리는 것을 신뢰하지 않는다
    code_hash: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    artifact_hashes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceFile:
    """원문 파일 4,622개.

    본문 4,201 · **첨부 415** · PDF 3 · 뷰어 HTML 3. 첨부는 전부 `감사보고서`(208) ·
    `연결감사보고서`(207) 이며 본문과 같은 DART XML 이라 **함께 파싱한다**(§6.15).
    `parse_mode` 는 파일별 실제 값이다.
    """

    build_id: str
    source_file_id: str
    doc_id: str
    #: 하위 호환 컬럼. schema 1.2와 같이 manifest 기준 NFC 상대경로이며,
    #: schema 1.3에서는 `normalized_path_key` 와 같은 값이다. 파일을 직접 열 때 쓰지 않는다.
    relpath: str
    normalized_path_key: str     #: NFC 비교·조인·source_file_id 계산용 키
    actual_relpath: str          #: inventory에서 얻은 실제 디스크 표기(NFD일 수 있음)
    role: str                    #: main | attachment | pdf | viewer_html
    sha256: str
    byte_size: int
    detected_format: str         #: dart_xml | html | pdf | unknown
    detected_encoding: str
    decode_status: str           #: ok | replaced
    parse_mode: str              #: strict | recovered | viewer_html | pdf_text | *_shell | *_failed …
    #: PDF+viewer HTML 문서에서의 선택 역할. 일반 XML/HTML 소스는 None.
    source_selection_role: str | None      #: primary | alternate
    #: parser가 Section으로 구조화할 내용을 만들었는지. 미적용은 None.
    parse_usable: bool | None
    #: :class:`ingest.pdf_html.SourceCoverage`의 물리 보존. 일반 source는 None.
    coverage_n_sections: int | None
    coverage_n_chars: int | None
    coverage_n_tables: int | None
    coverage_n_pages: int | None
    coverage_pages_with_text: int | None
    coverage_n_lines: int | None
    page_text_coverage: float | None
    active_nodes_removed: int | None
    active_attributes_removed: int | None
    #: HTML은 sanitized DOM ordinal, PDF는 1-based pypdf text-layer page/line.
    locator_kind: str | None
    locator_limitations: list[str]
    parse_warnings: list[str]
    parse_error: str | None


@dataclass(frozen=True)
class Document:
    """manifest 4,204건 + 파싱 결과 요약. 날짜 필드 이름을 여기서 통일한다."""

    build_id: str
    doc_id: str
    rcept_no: str
    corp_code: str
    corp_name: str
    listed_name: str
    stock_code: str
    sector: str
    industry: str
    doc_group: str
    doc_subtype: str | None      #: major 는 전부 null — event_type 을 쓴다
    event_type: str | None       #: report_nm 괄호 안 정규화 (major 28종)
    report_nm: str
    filer: str                   #: 지분공시는 발행회사가 아니라 보고자
    rcept_dt: str                #: 접수일. 주최측·manifest 이름을 그대로 쓴다
    base_year: int | None
    base_month: int | None
    is_correction: bool
    file_format: str
    n_source_files: int
    #: 재무 Fact 추출 결과. **「값 없음」과 「추출 미지원」은 다르다** —
    #: 전자는 원문에 재무제표가 없는 것이고, 후자는 있는데 우리가 못 읽는 것이다.
    #: `not_applicable` 정기공시 아님 · `ok` 추출됨 · `no_statements` 원문에 없음
    #: `unsupported_pdf` PDF 평문이라 표가 없음 · `unsupported_html` viewer HTML 경로 미지원
    #: `extractor_failed` 추출기가 예외로 죽음 — **「원문에 없음」과 절대 합치지 않는다**
    fact_extract_status: str
    #: 못 읽은 재무제표 범위. `unsupported_attachment_layout` 일 때만 채워진다
    #: (`CFS` · `SFS` · `CFS|SFS`). **문서 전체가 미지원인 것과 다르다** —
    #: 「연결은 읽었고 별도만 못 읽었다」를 조회가 구분해야 5차 검수 지적이 풀린다.
    fact_unsupported_scope: str | None
    #: PDF+viewer HTML을 각각 독립 파싱한 뒤 선택한 source와 교차검증 결과.
    #: 일반 문서는 source ID는 None/[], cross-check 필드는 None/[]다.
    primary_source_file_id: str | None
    alternate_source_file_ids: list[str]
    source_cross_check_status: str | None   #: matched | partial_consistent | conflict | not_comparable | single_source
    source_cross_check_reason: str | None
    #: 교차검증에 실제로 쓴 source ID. 단순히 파일이 존재하는 것과 다르다.
    source_cross_check_source_file_ids: list[str]
    source_cross_check_primary_digest: str | None
    source_cross_check_alternate_digest: str | None
    #: conflict는 nested tuple이므로 현 Arrow 스키마 도출기에서 추정하지 않고
    #: key-sorted JSON array로 보존한다. claim-level Evidence는 locator가 생긴 뒤 별도 계약으로 추가한다.
    source_cross_check_conflicts_json: str | None


@dataclass(frozen=True)
class Section:
    """정기공시 목차 단위. 검색용 Chunk 와는 다른 객체다(D2).

    `n_chars < 100` 은 index 힌트일 뿐 생략 판정이 아니다. 명시적 생략 문구가 있을 때만
    `explicit_omission` 을 세운다 — 「해당사항 없음」과 「파싱 실패」를 구분하기 위해서다.
    """

    build_id: str
    doc_id: str
    source_file_id: str
    block_id: str
    parent_id: str | None
    path: str
    order: int
    locator: str

    level: int
    title: str
    text: str                    #: 표는 Markdown view
    n_chars: int
    n_tables: int
    explicit_omission: bool
    index_eligible: bool
    #: 이 행이 속한 문서의 접수일. **`documents` 조인 없이 시점 필터가 되도록** 비정규화한다.
    #: 평가질문 70개 중 18개가 「~까지 접수된 공시 기준」 같은 시점 표현을 쓴다.
    #: 벡터 검색에서는 후보를 고른 **뒤** 거르면 결과가 0이 될 수 있어 사전 필터가 필요하다.
    #: `rcept_dt` 는 접수번호에 붙은 불변값이라 비정규화의 위험이 낮다.
    rcept_dt: str


@dataclass(frozen=True)
class Chunk:
    """검색용 조각. `Section` 은 구조, `Chunk` 는 검색이다 (D2 · P0-7).

    섹션을 그대로 인덱싱할 수 없다. 실측 119,161 섹션 중 8,000자 초과가 12.3% 인데
    **글자로는 68.5%** 라, 분할 없이는 코퍼스 내용의 3분의 2가 임베딩되지 않는다
    (임베딩 v2 상한 8,192 토큰).

    `parent_id` 가 원본 Section 의 `block_id` 다. 조각은 언제든 버리고 다시 만들 수 있고
    **Section 은 절대 지우지 않는다.**

    `over_budget` 는 쪼갤 수 없는 단위 하나가 예산을 넘은 조각이다(표 한 행 등).
    데이터를 버리지 않으려고 그대로 뒀다는 표시이며, **모델 상한 대응은 인덱싱 단계의 몫**이다.
    """

    build_id: str
    doc_id: str
    source_file_id: str
    block_id: str
    parent_id: str | None        #: 원본 Section 의 block_id
    path: str
    order: int                   #: 문서 안에서의 청크 순번
    locator: str                 #: `"SECTION[12]/PART[3]"`
    evidence_id: str             #: Evidence(kind=chunk_text) FK

    part_no: int
    n_parts: int
    text: str                    #: 내부 근거·재현용 raw. LLM 기본 전달 금지
    text_search: str             #: PII/active content를 제거한 검색·임베딩 projection
    text_prompt_safe: str        #: prompt-like 줄도 제거하고 untrusted-data 경계로 감싼 projection
    security_flags: list[str]    #: deterministic: pii/prompt/active/control/boundary 처리 내역
    pii_types: list[str]         #: masking된 PII 종류(정책 우선순위 순, 중복 없음)
    security_policy_version: str
    n_chars: int
    header_repeated: bool        #: 표 머리글을 반복해 넣었는가 (원문에 없던 줄)
    over_budget: bool
    content_hash: str
    chunker_version: str
    index_eligible: bool         #: Section 의 판정을 이어받는다
    #: 이 행이 속한 문서의 접수일. **`documents` 조인 없이 시점 필터가 되도록** 비정규화한다.
    #: 평가질문 70개 중 18개가 「~까지 접수된 공시 기준」 같은 시점 표현을 쓴다.
    #: 벡터 검색에서는 후보를 고른 **뒤** 거르면 결과가 0이 될 수 있어 사전 필터가 필요하다.
    #: `rcept_dt` 는 접수번호에 붙은 불변값이라 비정규화의 위험이 낮다.
    rcept_dt: str



@dataclass(frozen=True)
class Field:
    """정형 3종(exchange·major·holding)의 라벨경로 → 값.

    `acode` 는 DART 의 기계판독 필드코드로 한글 라벨보다 안정적이다
    (major/holding TE 셀 100% 보유). 서식이 개정돼 문구가 바뀌어도 코드는 유지된다.
    """

    build_id: str
    doc_id: str
    source_file_id: str
    block_id: str
    parent_id: str | None
    path: str                    #: `"2. 계약내역 > 계약금액(원)"`
    order: int
    locator: str
    evidence_id: str             #: Evidence.evidence_id FK; semantic block_id 계산에는 미사용

    value_raw: str               #: 내부 근거용 원문 그대로. `-` 를 0 으로 바꾸지 않는다
    value_masked: str            #: PII masking 전용. prompt/active-safe가 아니므로 LLM 전달 금지
    #: PII + active content + prompt instruction 정책과 untrusted-data 경계를 적용한
    #: LLM 기본 전달 컬럼. 대응 Evidence.excerpt_safe와 byte-identical해야 한다.
    value_prompt_safe: str
    pii_type: str | None         #: 복수 규칙 검출 시 `|`로 이은 deterministic 유형
    security_policy_version: str
    #: `literal` · `explicit_zero` · `not_reported` · `empty` · `non_numeric` · `parse_error`.
    #: **「0」과 「값 없음」을 구분한다.** 원문은 값이 없을 때 `-` 를 쓰는데,
    #: 이를 0 으로 읽으면 공시유보 계약의 금액이 0 원이 된다.
    value_status: str
    acode: str | None
    aunit: str | None
    aunitvalue: str | None
    is_pii: bool                 #: `pii_type` 또는 PII 라벨 정책에 따른 마스킹 대상
    #: 값이 속한 표. 표 단위로 근거를 되짚을 때 쓴다
    table_locator: str | None
    #: parser가 rowspan/colspan을 펼쳐 계산한 semantic 원점 좌표. locator의 물리
    #: TR/TD 자식 순번과 다를 수 있으므로 역산하지 않는다.
    logical_row: int
    logical_col: int
    #: 이 값에 붙은 라벨 셀들의 locator (공백 구분). 라벨-값 대응의 근거다 (C-02)
    label_locators: str | None
    #: 같은 라벨 경로가 문서 안에서 몇 번째로 나왔는가 (0-기반).
    #: 예전에는 결과가 dict 라 `계약금액` 이 두 번 나오면 **뒤엣것이 앞엣것을 덮어썼다** (K-17 · S-08).
    occurrence: int
    #: 이 행이 속한 문서의 접수일. **`documents` 조인 없이 시점 필터가 되도록** 비정규화한다.
    #: 평가질문 70개 중 18개가 「~까지 접수된 공시 기준」 같은 시점 표현을 쓴다.
    #: 벡터 검색에서는 후보를 고른 **뒤** 거르면 결과가 0이 될 수 있어 사전 필터가 필요하다.
    #: `rcept_dt` 는 접수번호에 붙은 불변값이라 비정규화의 위험이 낮다.
    rcept_dt: str




@dataclass(frozen=True)
class Cell:
    """표 셀. **기본 미적재** — `--with-cells` 로 켠다 (§6.14).

    한 번 전량 적재해 40,004,925 행 / 1.08GB 임을 확인하고 뺐다. 93.8%가 정기공시이고
    실제 수치는 28.5% 뿐이며, 지금 읽는 코드가 없다. 코드와 스키마는 남겨 두고
    **정기공시 표를 프로그래밍으로 질의해야 할 때** 15분 재생성한다.

    `header_path` 와 `inherited_from` 이 이 엔티티의 존재 이유다 — 그것만 Markdown 이
    표현하지 못한다. span 영역의 **모든 논리 좌표가 한 행**이라 `(row, col)` 로 바로 찾는다.
    """

    build_id: str
    doc_id: str
    source_file_id: str
    block_id: str
    parent_id: str | None        #: table 의 block_id
    path: str
    order: int
    locator: str

    origin_row: int
    origin_col: int
    logical_row: int
    logical_col: int
    rowspan: int
    colspan: int
    tag: str                     #: TD | TH | TU | TE
    role: str                    #: header | body | note
    text_raw: str
    acode: str | None
    aunit: str | None
    aunitvalue: str | None
    header_path: str             #: `"Change > Delta"` — colspan 헤더 전파 결과
    inherited_from: str | None   #: rowspan 원본 셀의 block_id



@dataclass(frozen=True)
class Evidence:
    """원문 위치와 인용 excerpt를 묶은 first-class 근거.

    ``excerpt_raw``는 감사·재현에 필요한 restricted 컬럼이다. 검색/Agent/LLM의 기본 경로는
    #5 정책을 거친 ``excerpt_safe``여야 한다. Evidence ID는
    ``source_file_id|kind|locator|excerpt_hash``에 결합되며 semantic block ID에는 들어가지
    않으므로 보안 projection이나 excerpt 정책을 바꿔도 Fact/Field의 정체성은 유지된다.
    """

    build_id: str
    evidence_id: str
    doc_id: str
    source_file_id: str
    kind: str                    #: fact_value | field_value | correction_value | chunk_text
    locator: str                #: 원문 파일 안 canonical locator
    table_locator: str | None
    logical_row: int | None
    logical_col: int | None
    excerpt_raw: str             #: restricted — 감사·재현 전용
    excerpt_safe: str            #: PII/prompt-safe + untrusted-data boundary
    excerpt_hash: str            #: excerpt_raw 전체 SHA-256
    raw_access: str              #: 항상 restricted
    extraction_method: str       #: financial_table_cell | structured_form_field | ...
    extraction_status: str       #: ok; 의미 값 상태는 Fact/Field/CorrectionItem에 보존
    security_flags: list[str]
    pii_types: list[str]
    security_policy_version: str
    evidence_policy_version: str
    rcept_dt: str


@dataclass(frozen=True)
class Fact:
    """재무제표 한 칸 — `(범위, 표, 계정, 기간)` → 값.

    **평가 A유형 질문이 직접 요구하는 층이다.** annotation 필드
    `scope(CFS/SFS)`·`period_start/end`·`cumulative_or_discrete`·`raw_value/raw_unit`·
    `row_header_path/column_header_path` 와 1:1로 대응한다.

    Section 마크다운만으로는 부족하다. 원문 데이터 표의 열 이름은 `제 57 기` 같은
    **상대 기수**이고 그것이 2025년이라는 사실은 앞 표에 따로 있다. 회사마다 기수가 달라
    (삼성 57기, 신설사 5기) 질의마다 LLM 이 다시 맞혀야 하고, 틀리면 전년도 숫자를 답한다.
    단위도 한 섹션에 `백만원`·`원` 이 섞여 놓치면 1,000배 틀린다.

    **단위를 적용하지 않는다.** `raw_value` 는 원문 숫자 그대로이고 `raw_unit` 이 함께 온다.
    미리 곱해 두면 단위 판정이 틀렸을 때 원문 값까지 오염된다 — 정규화는 조회 시점의 일이다.

    `account_norm` 은 값 셀의 XBRL 택소노미 ID(`acode`) 를 1차로, 원문 표기 **완전일치 사전**을 2차로 채운다. 부분 문자열로 찾으면
    `매출액` 검색이 `매출채권의 감소(증가)` 를 잡는다(실측 `매출총이익` 1,155 · `매출원가` 841 ·
    `매출액` 620). 모르는 계정도 `account_raw` 로 전부 남긴다.
    """

    build_id: str
    doc_id: str
    source_file_id: str
    block_id: str
    parent_id: str | None        #: 표의 locator 기반 block_id
    path: str                    #: `"CFS > IS > 매출액"`
    order: int
    locator: str                 #: 값 셀의 원문 위치
    evidence_id: str             #: Evidence.evidence_id FK; semantic block_id 계산에는 미사용

    scope: str                   #: CFS(연결) | SFS(별도) | UNK
    statement: str               #: BS | IS | CI | SE | CF
    statement_title: str         #: 원문 제목 `"2-2. 연결 손익계산서"`
    account_raw: str             #: 원문 계정명 그대로 (`"매출액 (주30)"`)
    #: 들여쓰기로 복원한 계층. **계정명만으로는 표 안에서 유일하지 않다** —
    #: 카카오 연결재무상태표는 `자산 > 비유동자산 > 유형자산`(1조 3,220억)과
    #: `자산 > 금융업자산 > 유형자산`(148억)을 모두 담는다. 계층은 원문에 전각 공백으로 있다.
    account_path: str
    account_depth: int
    account_norm: str | None     #: 아는 것만. 주석 표시 `(주30)` 를 떼고 완전일치
    #: **열 헤더 경로.** 첫 행만 읽으면 안 된다 — 분기·반기 보고서는 2단 헤더로
    #: `제 58 기 반기 > 3개월` 과 `> 누적` 을 나눈다. 첫 행만 보면 같은 계정·같은 기간에
    #: 값이 둘이 되어 조회가 모호해진다(실측 17.6% → 1.6%).
    period_label: str
    period_start: str | None     #: 시점형 표(재무상태표)는 None
    period_end: str | None
    period_type: str             #: annual | half | quarter | instant | unknown
    cumulative: bool | None      #: 누적인가. 시점형은 None
    value_text: str              #: 원문 셀 텍스트 그대로
    raw_value: float | None      #: 숫자로 읽은 값. **단위 미적용**, `-` 는 0 이 아니라 None
    raw_unit: str | None
    #: 단위를 어디서 얻었는가 — `table`(머리말 표) | `account`(계정명에 박힘).
    #: `기본주당이익 (단위 : 원)` 처럼 **그 행만 표 단위와 다른** 경우가 실측 5,602건이다.
    #: 표 단위를 그대로 붙이면 주당이익이 100만배 틀린다.
    unit_source: str
    #: `literal` · `explicit_zero` · `not_reported` · `empty` · `non_numeric` · `parse_error`.
    #: **「0」과 「값 없음」을 구분한다.** 원문은 값이 없을 때 `-` 를 쓰는데,
    #: 이를 0 으로 읽으면 공시유보 계약의 금액이 0 원이 된다.
    value_status: str

    table_locator: str
    logical_row: int
    logical_col: int
    #: 이 행이 속한 문서의 접수일. **`documents` 조인 없이 시점 필터가 되도록** 비정규화한다.
    #: 평가질문 70개 중 18개가 「~까지 접수된 공시 기준」 같은 시점 표현을 쓴다.
    #: 벡터 검색에서는 후보를 고른 **뒤** 거르면 결과가 0이 될 수 있어 사전 필터가 필요하다.
    #: `rcept_dt` 는 접수번호에 붙은 불변값이라 비정규화의 위험이 낮다.
    rcept_dt: str
    #: 값 셀의 XBRL 택소노미 요소 ID (`ifrs-full_Revenue`). 회사·연도가 달라도 같은 요소는 같은 ID 라 표기 변형에
    #: 흔들리지 않는 정규화 축이다. 서식 필드코드·문맥 접미사는 제거. 없으면 None (schema 1.9 추가 컬럼)
    acode: str | None = None
    #: `account_norm` 의 출처 — `acode`(택소노미 ID 사전 `src/ingest/acode_map.tsv`) | `label`(표기 완전일치) | None
    account_norm_source: str | None = None


@dataclass(frozen=True)
class EventIdentity:
    """현실의 사건 하나. 정정·해지를 거쳐도 **ID 가 유지된다** (S-09 · R-04).

    지금까지 `Event` 는 문서 1건 = 사건 1건이었다. 그런데 계약 하나가
    체결 → 정정 14회 → 해지 로 이어져도 그것이 **같은 사건**이라는 사실이 어디에도 없었다.
    실측 4,204 문서 = 3,580 사건이며, 관측이 3건 이상인 사건이 108개(최대 15개)다.

    `identity_status=resolved` 인 사건의 `event_key` 는 회사·사건군·안정 식별 필드에서
    만든다. 식별값이 부족하거나 충돌하면 뿌리 접수번호 범위의 provisional/ambiguous
    키를 써 서로 다른 사건이 조용히 합쳐지지 않게 한다.
    """

    build_id: str
    event_key: str
    #: `business_event`(계약·지분 등 현실의 사건) | `document_lineage`(정기공시 원본+정정).
    #: 사업보고서와 그 정정본은 **계보로는 맞지만 「사건」이 아니다** — 실측 3,580 중 910개(25%)가
    #: 정기공시다. 빼지 않고 무엇인지 밝힌다.
    kind: str
    corp_code: str
    corp_name: str
    doc_group: str
    doc_subtype: str | None
    root_rcept_no: str           #: 현재 코퍼스에서 관측된 체인의 뿌리
    identity_fingerprint: str    #: 회사·사건군·안정 식별 필드의 정규화 fingerprint
    identity_status: Literal["resolved", "provisional", "ambiguous"]
    resolver_version: str
    first_disclosed_at: str
    last_disclosed_at: str
    n_observations: int
    n_corrections: int
    #: **코퍼스 끝 시점의 상태다.** 「지금 상태」가 아니다 —
    #: `status` 라는 이름은 과거 시점 질의에서 오답을 유도한다. Freudenberg 계약은
    #: 2025-12-25 에 유효하고 12-26 에 해지되는데, 이 필드만 읽으면 두 날 다 `terminated` 다.
    #: 시점별 상태는 `EventObservation` 을 `observed_at <= cutoff` 로 걸러 계산한다.
    status_at_corpus_end: str    #: active | terminated


@dataclass(frozen=True)
class EventObservation:
    """특정 문서가 그 사건에 대해 **그 시점에 말한 것** (S-09 · R-05).

    `seq` 와 `previous_observation_rcept_no` 로 속성 변화를 시간순으로 재생할 수 있다.
    """

    build_id: str
    event_key: str
    seq: int                     #: 사건 안에서의 시간순 번호
    doc_id: str
    rcept_no: str
    observed_at: str             #: 접수일
    is_correction: bool
    is_termination: bool
    #: **직전 관측**이다 — 문서의 이전 판본이 아니다. 예전 이름(`supersedes_rcept_no`)은
    #: 정정 사슬처럼 읽혀 「최신본」과 혼동됐다 (6차 검수). 문서 판본은
    #: `read.latest_document_version()`, 사건 관측은 이 필드다.
    previous_observation_rcept_no: str | None
    #: 실제 채택된 non-PII Field Evidence만 담는다. 두 배열은 같은 길이의
    #: `(evidence_id, role)` pair이며 evidence_id 기준 정렬·중복 제거한다.
    supporting_evidence_ids: list[str]
    support_roles: list[str]
    support_status: str          #: fully_verified | partial | not_applicable
    support_version: str
    #: partial이면 부족한 필수 role을 machine-readable 문자열로 남긴다.
    support_limitation: str | None


@dataclass(frozen=True)
class CorrectionItem:
    """정정공시가 선언한 「무엇이 어떻게 바뀌었는가」.

    `Relation` 간선은 「A 가 B 를 정정했다」까지만 담는다. 정작 사용자가 묻는
    「뭐가 바뀌었어?」에는 답할 수 없어 원문 `<CORRECTION>` 표를 별도 엔티티로 뺀다.

    **분류축을 둘로 나눈다** (S-10). 예전에는 `value_kind` 하나가 「값이 바뀌었나」와
    「값이 무엇인가」를 섞고 있었다. 그래서 `(주1) → 100` 은 `changed` 로만 남아
    **정정 전이 대체표기였다는 사실이 사라졌다.**

    | 축 | 값 |
    |---|---|
    | `diff_kind` | `same` · `changed` |
    | `before_kind` / `after_kind` | `literal` · `placeholder` · `empty` |

    `placeholder` 는 「값이 없음」이 아니라 **「표에 값을 적지 않고 각주를 가리킴」**이다
    (`(주1)`, `(표현 양식 변형)`, `별첨`). 전/후가 같다고 「변경 없음」이 아니다.

    이 엔티티가 있어야 *declared diff*(선언된 정정)와 *computed diff*(실제 값 비교)를
    대조할 수 있다.
    """

    build_id: str
    doc_id: str
    source_file_id: str
    block_id: str
    parent_id: str | None        #: 대상 `Relation.relation_id`
    path: str                    #: 정정 항목 (`"3. 계약상대"`)
    order: int
    locator: str

    rcept_no: str
    corp_name: str
    doc_group: str
    reason: str | None
    value_before: str | None
    value_after: str | None
    #: `locator`는 행 대표 위치라 전/후 값 Evidence의 좌표로 재사용할 수 없다.
    #: 각 side가 실제로 적힌 원문 셀을 별도로 보존한다. 값이 있으면 해당 locator도 필수다.
    before_locator: str | None
    after_locator: str | None
    #: 대표 locator와 각 side 셀의 semantic 격자 원점. 합성 ROW 대표만 null을
    #: 허용한다. side 값 셀이 존재하면 좌표도 함께 보존한다.
    logical_row: int | None
    logical_col: int | None
    before_logical_row: int | None
    before_logical_col: int | None
    after_logical_row: int | None
    after_logical_col: int | None
    before_evidence_id: str | None  #: value_before가 존재할 때 Evidence FK
    after_evidence_id: str | None   #: value_after가 존재할 때 Evidence FK
    diff_kind: str               #: same | changed
    before_kind: str             #: literal | placeholder | empty
    after_kind: str              #: literal | placeholder | empty              #: changed | placeholder | formatting | empty
    required_by_authority: bool  #: 당국 정정요구·명령에 따른 것인가
    #: 이 행이 속한 문서의 접수일. `documents` 조인 없이 시점 필터가 되도록 비정규화한다.
    rcept_dt: str


@dataclass(frozen=True)
class Relation:
    """정정 계보와 사건 관계. boolean 으로 평탄화하지 않는다(D11).

    미해결을 임의로 하나에 연결하지 않는다. `root_missing` 은 원공시가 코퍼스 밖이라는
    사실 자체가 정보다 — 거래소공시 정정의 절반이 여기 해당한다.
    """

    build_id: str
    relation_id: str
    src_kind: str                #: document | event
    src_id: str                  #: src_kind=document 이면 Document.doc_id
    src_rcept_no: str            #: 원천 시스템의 출발 접수번호
    dst_kind: str | None
    dst_id: str | None           #: dst_kind=document 이면 Document.doc_id
    dst_rcept_no: str | None     #: dst_id와 같은 대상의 원천 접수번호. 대상 미지정이면 null
    relation_type: str           #: CORRECTS | SUPERSEDES | RELATED | TERMINATES
    resolution_status: str       #: resolved | candidate | ambiguous | root_missing | parse_failed
    target_hint: str | None
    confidence: float
    resolver_version: str
    #: 판정 시점의 후보 접수번호(공백 구분). **판정만 남기면 재검토할 수 없다** (R-01)
    candidate_ids: str | None
    #: 매칭에 쓴 키. `"corp|group|20240711|타법인주식및출자증권양수결정"` (R-01)
    match_features: str | None
    #: 출발 문서의 접수일. **간선이 언제부터 보이는가**를 정한다 —
    #: 「2025-12-16 까지 접수된 공시 기준」 질의에서 그 이후 정정 간선이 새면 안 된다.
    src_rcept_dt: str | None
    #: `root_missing` 인 이유 (R-02). 실측 분포:
    #:
    #: | 이유 | 건수 | 뜻 |
    #: |---|---:|---|
    #: | `submitted_before_corpus` | 295 | 원공시가 2023-01-01 이전 — **데이터 한계** |
    #: | `no_submitted_date` | 38 | 제출일 파싱 실패 |
    #: | `type_mismatch` | 20 | 같은 날 문서는 있으나 유형이 다름 |
    #: | `date_absent` | 3 | 그 회사 문서는 있으나 그 날짜가 없음 |
    #: | `no_earlier_in_period` | 2 | 같은 기간에 직전본이 없음 |
    #:
    #: 82%가 수집 범위 밖이라는 뜻이므로 **결함이 아니다.** 이 구분이 없으면
    #: 358건 전부를 매칭 결함으로 오해한다.
    root_missing_reason: str | None
    #: source document의 실제 Field/Correction Evidence. target anchor가 없거나
    #: unresolved 관계이면 partial이며 fully_verified로 승격하지 않는다.
    supporting_evidence_ids: list[str]
    support_roles: list[str]
    support_status: str          #: fully_verified | partial
    support_version: str
    support_limitation: str | None


def as_row(obj: Any) -> dict[str, Any]:
    """dataclass → Parquet/JSONL 행. locatable 8컬럼이 앞에 오도록 정렬한다.

    locatable 엔티티인데 8컬럼이 하나라도 빠지면 계약 위반이므로 즉시 실패시킨다.
    """
    data = asdict(obj)
    if is_locatable(type(obj)):
        missing = [k for k in LOCATABLE_COLUMNS if k not in data]
        if missing:
            raise TypeError(f"{type(obj).__name__} 에 locatable 컬럼 누락: {missing}")
    head = {k: data.pop(k) for k in LOCATABLE_COLUMNS if k in data}
    return {**head, **data}
