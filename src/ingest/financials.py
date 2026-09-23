"""정기공시 재무제표 → Fact. 평가 A유형 질문이 직접 요구하는 층이다.

## 왜 마크다운으로는 부족한가

`G-A-001 삼성전자의 2025년 연결기준 매출액` 을 원문에서 보면 **「2025년」이라는 말이 데이터 표
안에 없다.**

```text
| 제 57 기 2025.01.01 부터 2025.12.31 까지 |   ← 머리말 표
| (단위 : 백만원) |

|  | 제 57 기 | 제 56 기 | 제 55 기 |          ← 데이터 표, 열 이름은 기수
| 매출액 (주30) | 333,605,938 | … |
```

열 이름은 `제 57 기` 이고 그것이 2025년이라는 사실은 **앞 표에 따로** 있다. 단위도 그렇고,
한 섹션에 `백만원`·`원` 이 섞이기도 한다(주당순이익). 마크다운만 주면 **기수→기간 매핑을
질의마다 LLM 이 다시 맞혀야** 하고, 회사마다 기수가 다르다(삼성 57기, 신설사 5기).
틀리면 전년도 숫자를 답하거나 단위를 놓쳐 1,000배 틀린다.

평가 annotation 필드가 요구하는 것과 이 모듈의 산출이 1:1로 대응한다 —
`scope(CFS/SFS)`, `period_start/end`, `cumulative_or_discrete`,
`row_header_path/column_header_path`, `raw_value/raw_unit`.

## 원문 구조는 예외가 없다

실측 432/432 전부 같은 모양이다.

```text
<TABLE-GROUP>
  <TITLE>2-2. 연결 손익계산서</TITLE>
  <TABLE>   머리말 — 표제 · 기간 행 · (단위 : 백만원)
  <TABLE>   데이터 — 첫 행=기수 헤더, 첫 열=계정과목
</TABLE-GROUP>
```

번호 접두가 범위를 가른다. `2-N` = 연결(CFS), `4-N` = 별도(SFS).

## 계정과목은 부분 문자열로 찾지 않는다

실측 빈도가 `매출총이익` 1,155 · `매출원가` 841 · **`매출액` 620** 순이다.
`"매출액" in 계정명` 으로 찾으면 `매출채권의 감소(증가)` 를 잡는다.
그래서 `account_norm` 은 값 셀의 XBRL 택소노미 ID(`acode_map.tsv`)를 1차로, 원문 표기 **완전일치 사전**을 2차로만 채우고,
`account_raw` 는 원문 그대로 전부 남긴다.
아는 계정만 정규화하고 모르는 것은 버리지 않는다.

## 값은 읽고, 단위는 적용하지 않는다

`333,605,938` → `333605938`, `(6,621,613)` → `-6621613`, `-` → `None` 까지가 이 모듈의 일이다.
**백만원을 원으로 바꾸지 않는다** — `raw_value` 와 `raw_unit` 을 함께 두고 정규화는 조회 시점에 한다.
미리 곱해 두면 단위 판정이 틀렸을 때 원문 값까지 오염된다. 원문 텍스트도 `value_text` 로 남긴다.
"""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Mapping

from .dart_xml import ParseFailure, _text_of
from .table_grid import build_grid
from .values import classify as classify_value_status

__all__ = ["FinancialFact", "FinancialLayoutIssue", "FinancialExtractionResult",
           "FinancialPeriodOverride", "FinancialPeriodOverrideApplied",
           "UnsupportedFinancialLayout", "STATEMENTS", "ACCOUNT_NORM", "ACODE_NORM",
           "ACCOUNT_MAP_PATH", "ACODE_MAP_PATH", "PERIOD_OVERRIDE_PATH", "load_account_map",
           "load_acode_map", "taxonomy_code",
           "load_period_overrides", "account_key", "parse_number",
           "parse_period", "extract_financials", "extract_financials_detailed"]

#: 제목에 나타나는 표 이름 → 코드. **순서가 중요하다** —
#: `포괄손익계산서` 를 `손익계산서` 보다 먼저 봐야 한다(부분 문자열 포함 관계).
STATEMENTS: tuple[tuple[str, str], ...] = (
    ("재무상태표", "BS"),
    ("포괄손익계산서", "CI"),
    ("손익계산서", "IS"),
    ("현금흐름표", "CF"),
)

#: **자본변동표는 넣지 않는다.** 이 모델은 `(계정 × 기간)` 인데 자본변동표는
#: `(사건 × 자본항목)` 이다 — 열이 `자본금`·`이익잉여금` 같은 자본 구성요소이고
#: 기간은 행(`2023.01.01 (기초자본)`)에 들어 있다.
#:
#: 억지로 넣으면 실측 311,073행이 `period_type=unknown` 에 계정명이 `2023.01.01 (기초자본)` 인
#: **틀린 의미의 데이터**가 된다. 내용은 Section 마크다운에 그대로 남으므로 검색에는 지장이 없다.
#: 별도 엔티티로 다룰 값이 생기면 그때 만든다.
EXCLUDED_STATEMENTS = ("자본변동표",)

#: 계정 정규화 사전 파일. **코드가 아니라 데이터다.**
#:
#: 코드에 박아 두면 사전을 넓히는 데 코드 수정이 필요하고, 누가 어떤 계정을 왜 넣었는지
#: diff 로 읽기 어렵다. 파일로 빼면 증분 확장이 코드 변경 없이 된다.
#: 파일 내용은 `artifact.code_hash` 에 포함되므로 **고치면 `build_id` 가 바뀐다.**
ACCOUNT_MAP_PATH = Path(__file__).with_name("account_map.tsv")

#: 원문 안에서 서로 모순되는 기간 표기를 사람이 대조해 승인한 **source-bound** 보정.
#:
#: 기업명이나 접수번호만으로 분기하지 않는다. 원문 전체 SHA-256(64 hex), 데이터 표의
#: locator, 실제 열 머리글이 모두 같을 때만 적용한다. 원문 바이트나 표 구조가 하나라도
#: 바뀌면 일반 파서로 되돌아가 typed unsupported 진단을 남긴다.
PERIOD_OVERRIDE_PATH = Path(__file__).with_name("financial_period_overrides.tsv")


def load_account_map(path: Path | None = None) -> dict[str, str]:
    """`원문표기<TAB>개념` TSV → 사전. 주석(`#`)과 빈 줄은 건너뛴다.

    같은 표기가 두 번 나오면 **조용히 덮어쓰지 않고** 올린다 — 사전이 커질수록
    중복이 생기고, 뒤엣것이 이기는 규칙은 사람이 예측하기 어렵다.
    """
    path = path or ACCOUNT_MAP_PATH
    out: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(f"{path.name}:{lineno} 형식 오류 — `표기<TAB>개념` 이어야 합니다")
        key, concept = parts[0].strip(), parts[1].strip()
        if key in out and out[key] != concept:
            raise ValueError(
                f"{path.name}:{lineno} `{key}` 가 `{out[key]}` 와 `{concept}` 로 중복 정의됐습니다")
        out[key] = concept
    return out


ACCOUNT_NORM: dict[str, str] = load_account_map()

#: XBRL 택소노미 요소 ID → 개념. 값 셀의 `ACODE` 에서 읽는 **1차** 정규화 출처 (표기 사전은 2차).
ACODE_MAP_PATH = Path(__file__).with_name("acode_map.tsv")
_TAXONOMY_CODE = re.compile(r"^(?:ifrs-full|dart)_[A-Za-z0-9]+$")


def load_acode_map(path: Path | None = None) -> dict[str, str]:
    """`ACODE<TAB>개념` TSV → 사전. 개념은 `account_map.tsv` 가 아는 어휘 안에 있어야 한다 — ACODE 로
    표기 사전에 없는 새 개념을 만들지 않는다(두 출처가 같은 어휘를 써야 조회가 한 가지 뜻으로 닫힌다)."""
    path = path or ACODE_MAP_PATH
    out: dict[str, str] = {}
    known = set(ACCOUNT_NORM.values())
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(f"{path.name}:{lineno} 형식 오류 — `ACODE<TAB>개념` 이어야 합니다")
        key, concept = parts[0].strip(), parts[1].strip()
        if _TAXONOMY_CODE.match(key) is None:
            raise ValueError(f"{path.name}:{lineno} `{key}` 는 택소노미 요소 ID 형식이 아닙니다")
        if concept not in known:
            raise ValueError(f"{path.name}:{lineno} `{concept}` 는 account_map.tsv 에 없는 개념입니다")
        if key in out and out[key] != concept:
            raise ValueError(
                f"{path.name}:{lineno} `{key}` 가 `{out[key]}` 와 `{concept}` 로 중복 정의됐습니다")
        out[key] = concept
    return out


ACODE_NORM: dict[str, str] = load_acode_map()


def taxonomy_code(acode: str | None) -> str | None:
    """DART XML 셀의 ACODE 에서 택소노미 요소 ID 만 남긴다.

    `ifrs-full_Assets|CFY2024e_…Axis_…` 처럼 축·기간 문맥이 `|` 뒤에 붙는 경우가 있고,
    `BSS_PIS_CMPY` 같은 서식 필드코드는 택소노미가 아니므로 None."""
    if not acode:
        return None
    base = acode.split("|", 1)[0].strip()
    return base if _TAXONOMY_CODE.match(base) else None


@dataclass(frozen=True)
class FinancialPeriodOverride:
    """감사 완료된 기간 열 보정 한 건.

    ``source_sha256``은 축약 digest가 아닌 64자리 전체 SHA-256이다. ``raw_header``는
    데이터 표에 실제로 적힌 열 머리글이라, 같은 파일 안의 다른 열에는 적용되지 않는다.
    """

    decision_id: str
    doc_id: str
    source_sha256: str
    table_locator: str
    raw_header: str
    period_start: str
    period_end: str
    period_type: str
    cumulative: bool
    corroborating_locator: str
    reason: str


@dataclass(frozen=True)
class FinancialPeriodOverrideApplied:
    """실제 추출에서 모든 guard를 통과해 적용된 보정의 감사 기록."""

    decision_id: str
    doc_id: str
    source_sha256: str
    table_locator: str
    raw_header: str
    corroborating_locator: str


_OVERRIDE_COLUMNS = (
    "decision_id", "doc_id", "source_sha256", "table_locator", "raw_header",
    "period_start", "period_end", "period_type", "cumulative",
    "corroborating_locator", "reason",
)


def load_period_overrides(
        path: Path | None = None) -> dict[tuple[str, str, str], FinancialPeriodOverride]:
    """versioned TSV를 ``(full hash, table locator, raw header)``로 색인한다.

    잘못된 정책 파일을 일부만 적용하는 것보다 빌드를 멈추는 편이 안전하므로 헤더, digest,
    날짜, 중복 key를 모두 엄격히 검증한다.
    """
    path = path or PERIOD_OVERRIDE_PATH
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        return {}
    header = tuple(lines[0].split("\t"))
    if header != _OVERRIDE_COLUMNS:
        raise ValueError(
            f"{path.name}: header가 계약과 다릅니다: {header!r} != {_OVERRIDE_COLUMNS!r}")
    out: dict[tuple[str, str, str], FinancialPeriodOverride] = {}
    decision_ids: set[str] = set()
    for lineno, line in enumerate(lines[1:], 2):
        values = line.split("\t")
        if len(values) != len(header):
            raise ValueError(
                f"{path.name}:{lineno} 열 수 {len(values)} != {len(header)}")
        row = dict(zip(header, values))
        digest = row["source_sha256"].lower()
        if re.fullmatch(r"[0-9a-f]{64}", digest, flags=re.ASCII) is None:
            raise ValueError(f"{path.name}:{lineno} full SHA-256 형식 오류")
        if not row["decision_id"] or row["decision_id"] in decision_ids:
            raise ValueError(f"{path.name}:{lineno} decision_id 누락/중복")
        if re.fullmatch(r"TABLE\[\d+\]", row["table_locator"]) is None:
            raise ValueError(f"{path.name}:{lineno} table_locator 형식 오류")
        if not row["raw_header"] or not row["corroborating_locator"] or not row["reason"]:
            raise ValueError(f"{path.name}:{lineno} 감사 guard/근거 누락")
        if re.fullmatch(
                r"TABLE\[\d+\]/(?:T(?:HEAD|BODY|FOOT)\[\d+\]/)?"
                r"TR\[\d+\]/(?:TD|TH)\[\d+\]",
                row["corroborating_locator"]) is None:
            raise ValueError(f"{path.name}:{lineno} corroborating_locator 형식 오류")
        for name in ("period_start", "period_end"):
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", row[name]) is None:
                raise ValueError(f"{path.name}:{lineno} {name} 형식 오류")
        if row["period_type"] not in {"annual", "half", "quarter", "instant"}:
            raise ValueError(f"{path.name}:{lineno} period_type 오류")
        if row["cumulative"] not in {"true", "false"}:
            raise ValueError(f"{path.name}:{lineno} cumulative은 true/false여야 함")
        item = FinancialPeriodOverride(
            decision_id=row["decision_id"], doc_id=row["doc_id"],
            source_sha256=digest, table_locator=row["table_locator"],
            raw_header=row["raw_header"], period_start=row["period_start"],
            period_end=row["period_end"], period_type=row["period_type"],
            cumulative=row["cumulative"] == "true",
            corroborating_locator=row["corroborating_locator"], reason=row["reason"],
        )
        key = (item.source_sha256, item.table_locator, item.raw_header)
        if key in out:
            raise ValueError(f"{path.name}:{lineno} override key 중복: {key!r}")
        out[key] = item
        decision_ids.add(item.decision_id)
    return out


_TITLE = re.compile(r"^\s*(\d+)-(\d+)\.\s*(.+)$")
_SECTION_TAGS = ("SECTION-1", "SECTION-2", "SECTION-3", "SECTION-4", "BODY")
_UNIT = re.compile(r"단\s*위\s*[:：]\s*([^)\]]{1,30})")
#: `제 57 기 2025.01.01 부터 2025.12.31 까지` / `제 57 기말 2025.12.31 현재`
_PERIOD = re.compile(
    r"제\s*(?P<no>\d+)\s*기(?P<mod>\s*말|\s*초|\s*\d+분기말|\s*\d+분기|\s*반기말|\s*반기)?\s*"
    r"(?P<d1>\d{4}[.\-]\d{1,2}[.\-]\d{1,2})\s*"
    r"(?:부터\s*(?P<d2>\d{4}[.\-]\d{1,2}[.\-]\d{1,2})\s*까지|현재)"
)
# 외부감사인이 작성한 legacy 표는 `2023.03.31` 대신 `2023년 3월 31일`을 쓴다.
# 콜론 유무, `제16기`/`제 16 기`, `1분기말`/`반기`도 문서마다 다르다.
_KO_DATE = r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일"
_PERIOD_KO = re.compile(
    rf"제\s*(?P<no>\d+)(?:\s*\([^)]*\))?\s*기"
    rf"(?P<mod>\s*\d+\s*분기\s*말|\s*\d+\s*분기|"
    rf"\s*반기\s*말|\s*반기|\s*기?\s*말|\s*기?\s*초)?"
    rf"\s*:?\s*\(?\s*(?P<d1>{_KO_DATE})\s*"
    rf"(?:부터\s*(?P<d2>{_KO_DATE})\s*까지|현재)\s*\)?"
)
_NUM = re.compile(r"^\(?\s*-?[\d,]+(?:\.\d+)?\s*\)?$")
# 금융사 manual 표의 주당이익은 셀 자체가 `3,754원`처럼 단위를 갖기도 한다.
_VALUE_UNIT = re.compile(
    r"^\s*(?P<number>\(?\s*(?:△|-)?[\d,]+(?:\.\d+)?\s*\)?)\s*"
    r"(?P<unit>백만원|천원|원)\s*$")
#: 2단 헤더 마지막 마디의 누적 여부 표기
_DISCRETE = frozenset({"3개월", "당3개월", "전3개월", "3개월간", "당분기3개월"})
_CUMULATIVE = frozenset({"누적", "당누적", "전누적", "누계", "당기누적", "전기누적"})
_SQ = lambda s: re.sub(r"\s+", "", s or "")
# manual 감사보고서 열은 `제19(당)기` / `제19기(당)기` 두 형태를 모두 쓴다.
# `(당)`·`(전)`은 비교 열 설명일 뿐 기수의 일부가 아니므로 매칭 키에서만 제거한다.
_PERIOD_ANNOTATION = re.compile(r"\((?:당|전|전전|당기|전기|전전기)\)")


def _period_key(text: str) -> str:
    key = _SQ(_PERIOD_ANNOTATION.sub("", text or ""))
    # `제19기(당)기` -> `제19기기` 같은 편집 표기를 표준 기수로 접는다.
    return re.sub(r"기기(?=(?:\d+분기|반기|말|초|$))", "기", key)
#: 기수 뒤의 수식어(`3분기`·`반기`·`말`). 열 이름이 이를 생략하는 표가 있다.
_BARE = re.compile(r"(\d+분기말?|반기말?|말|초)$")
#: 계정명 뒤의 주석 표시. `매출액 (주30)`·`판매비와관리비 (주21,22)`·`영업이익(주30,31)`
_FOOTNOTE = re.compile(r"\(\s*주\s*[\d,\s.]*\)")

#: **계정명 안에 박힌 단위.** `기본주당이익(손실) (단위 : 원)` 처럼 그 행만 표 단위와 다르다.
#: 실측 14,798건이고 그 중 5,602건은 표 단위와 실제로 다르다(백만원 4,132 · 천원 1,470).
#: 표 단위를 그대로 붙이면 주당이익이 **100만배 틀린다** — `810원` 이 `810백만원` 이 된다.
_INLINE_UNIT = re.compile(r"\(\s*단\s*위\s*[:：]\s*([^)]{1,20})\)")
# `기본주당이익(원)`도 해당 행의 단위 표시다.
_INLINE_UNIT_SHORT = re.compile(r"\(\s*(백만원|천원|원)\s*\)\s*$")


def account_key(text: str) -> str:
    """계정명을 사전 조회 키로 만든다. **주석 표시와 인라인 단위를 뗀다.**

    원문은 `매출액 (주30)` 처럼 주석 번호를, `기본주당이익 (단위 : 원)` 처럼 단위를
    계정명에 붙여 쓴다. 떼지 않으면 완전일치 사전이 맞지 않는다 —
    주석 미제거로 1,109건 중 123건만 맞았고, 인라인 단위 14,798건은 **전부** 실패했다.
    """
    return _SQ(_INLINE_UNIT_SHORT.sub("", _INLINE_UNIT.sub("", _FOOTNOTE.sub("", text or ""))))


def inline_unit(text: str) -> str | None:
    """계정명에 박힌 단위. 있으면 **표 단위보다 우선한다** — 그 행에만 적용되는 값이다."""
    m = _INLINE_UNIT.search(text or "")
    if m:
        return m.group(1).strip()
    m = _INLINE_UNIT_SHORT.search(text or "")
    return m.group(1).strip() if m else None


def value_unit(text: str) -> str | None:
    """값 셀에 직접 붙은 단위(`3,754원`). 명시적인 통화 단위만 허용한다."""
    m = _VALUE_UNIT.match(text or "")
    return m.group("unit") if m else None


#: 표기만으로는 자본잔액과 손익 귀속액을 가르지 못하는 계정.  `비지배지분` 한
#: 낱말이 재무상태표에서는 **잔액**을, 손익계산서에서는 그 해에 **귀속된 이익**
#: 을 가리킨다 — 자릿수부터 다르다.  XBRL 요소 ID 가 있으면 그것이 답하지만
#: (`acode_map.tsv`), 없는 행이 3,720건이라 부모 계정으로 가른다 (이슈 #127).
_ATTRIBUTION_LABEL_NORMS = {
    "non_controlling_interests": (
        "net_income_non_controlling", "comprehensive_income_non_controlling"),
    "owners_of_parent": (
        "net_income_owners_of_parent", "comprehensive_income_owners_of_parent"),
}
_ATTRIBUTION_NET_INCOME_PARENT = re.compile(r"순이익|순손실|순손익|계속영업")


def _attribution_norm(label_norm, statement, account_path):
    """손익표에 실린 귀속액을 자본잔액 개념으로 붙이지 않는다.

    재무상태표·자본변동표는 그대로 잔액이다. 손익계산서·포괄손익계산서에서는
    부모 계정이 `총포괄손익의 귀속` 인지 `당기순이익의 귀속` 인지로 가른다.
    둘 다 아니면 **None** 을 준다 — 조용히 틀린 개념으로 붙는 것보다 모른다고
    두는 편이 낫다 (실측 3,720건 중 78건).
    """

    split = _ATTRIBUTION_LABEL_NORMS.get(label_norm)
    if split is None or statement not in ("IS", "CI"):
        return label_norm
    parent = str(account_path).rsplit(">", 1)[0] if ">" in str(account_path) else ""
    if "포괄" in parent:
        return split[1]
    if _ATTRIBUTION_NET_INCOME_PARENT.search(parent):
        return split[0]
    return None


@dataclass(frozen=True)
class FinancialFact:
    """재무제표 한 칸. `(범위, 표, 계정, 기간)` → 값."""

    scope: str                  #: CFS(연결) | SFS(별도)
    statement: str              #: BS | IS | CI | SE | CF
    statement_title: str        #: 원문 제목 `"2-2. 연결 손익계산서"`
    account_raw: str            #: 원문 계정명 그대로 (`"매출액 (주30)"`)
    account_path: str           #: 들여쓰기로 복원한 계층 (`"자산 > 비유동자산 > 유형자산"`)
    account_depth: int          #: 들여쓰기 단계 (0 = 최상위)
    account_norm: str | None    #: 완전일치 사전으로 아는 것만
    period_label: str           #: 열 헤더 경로 (`"제 58 기 반기 > 누적"`)
    period_start: str | None    #: `"2025-01-01"` — 시점형 표는 None
    period_end: str | None      #: `"2025-12-31"`
    period_type: str            #: annual | half | quarter | instant | unknown
    cumulative: bool | None     #: 누적인가. 시점형은 None
    value_text: str             #: 원문 셀 텍스트 그대로
    raw_value: float | None     #: 숫자로 읽은 값. **단위 미적용**
    raw_unit: str | None        #: `"백만원"`. 계정명에 단위가 박혀 있으면 그쪽이 이긴다
    unit_source: str            #: table | account | value — 단위를 어디서 얻었는가
    #: literal | explicit_zero | not_reported | empty | non_numeric | parse_error.
    #: **`0` 과 「값 없음」을 구분한다** — `raw_value=None` 만으로는 갈리지 않는다.
    value_status: str
    #: 원문 위치 — 근거 추적용
    table_locator: str
    cell_locator: str
    logical_row: int
    logical_col: int
    #: 값 셀의 XBRL 택소노미 요소 ID (`ifrs-full_Revenue`). 서식 필드코드·`|` 문맥 접미사는 제거. 없으면 None
    acode: str | None = None
    #: `account_norm` 의 출처 — `acode`(택소노미 ID 사전) | `label`(원문 표기 완전일치) | None
    account_norm_source: str | None = None


@dataclass(frozen=True)
class FinancialLayoutIssue:
    """재무제표가 보이지만 안전하게 구조화하지 못한 한 지점.

    `code` 는 상위 빌드가 문자열을 다시 해석하지 않고 typed unsupported 상태로
    매핑하기 위한 안정적 식별자다. `locator`·`context` 는 사람의 재현 경로다.
    """

    code: str
    context: str
    statement_hint: str | None
    locator: str | None
    detail: str


@dataclass(frozen=True)
class FinancialExtractionResult:
    """부분 성공을 숨기지 않는 상세 추출 결과.

    기존 호출부는 :func:`extract_financials` 로 Fact 목록만 받아도 된다. 품질 gate와
    차기 canonical 빌드는 이 상세 API를 사용해 `issues`를 문서 상태에 반영할 수 있다.
    """

    facts: tuple[FinancialFact, ...]
    issues: tuple[FinancialLayoutIssue, ...]
    detected_statements: int
    applied_overrides: tuple[FinancialPeriodOverrideApplied, ...] = ()


class UnsupportedFinancialLayout(ParseFailure):
    """재무제표 근거는 있으나 지원 레이아웃으로 한 건도 읽지 못한 경우."""

    code = "unsupported_financial_layout"

    def __init__(self, issues: list[FinancialLayoutIssue] | tuple[FinancialLayoutIssue, ...]):
        self.issues = tuple(issues)
        sample = self.issues[0] if self.issues else None
        detail = (f"{sample.code} at {sample.locator or sample.context}: {sample.detail}"
                  if sample else "재무제표 구조를 안전하게 판별할 수 없습니다")
        super().__init__(f"{self.code}: {detail}")


def parse_number(text: str) -> float | None:
    """`333,605,938` → 333605938.0 · `(6,621,613)` → -6621613.0 · `-` → None.

    회계 표기에서 괄호는 음수다. `-`·공란·문자는 값이 아니므로 0 이 아니라 None 이다 —
    0 으로 바꾸면 「값 없음」과 「값이 0」이 구분되지 않는다.
    """
    t = (text or "").strip()
    if not t or t in {"-", "–", "—", "()"}:
        return None
    with_unit = _VALUE_UNIT.match(t)
    if with_unit:
        t = with_unit.group("number").strip()
    negative = t.startswith("(") and t.endswith(")")
    body = t.strip("()").strip().replace(",", "")
    if body.startswith("△"):                     # DART 음수 표기
        negative, body = True, body[1:]
    if not _NUM.match(t) and not re.fullmatch(r"-?\d+(?:\.\d+)?", body):
        return None
    try:
        value = float(body)
    except ValueError:
        return None
    return -abs(value) if negative else value


def parse_period(text: str) -> dict | None:
    """기간 행 하나 → `{label, start, end, type, cumulative}`.

    `제 N 기` 는 **회사마다 다른 상대 표기**다. 여기서 실제 날짜에 붙여 두지 않으면
    「2025년」 질의를 열 이름으로 맞출 수 없다.
    """
    match = _PERIOD.search(text or "") or _PERIOD_KO.search(text or "")
    if not match:
        return None
    mod = _SQ(match.group("mod") or "")
    if mod in ("기말", "기초"):
        mod = mod[1:]
    d1, d2 = match.group("d1"), match.group("d2")
    def norm(d: str | None) -> str | None:
        if not d:
            return None
        nums = re.findall(r"\d+", d)
        if len(nums) != 3:
            return None
        return f"{int(nums[0]):04d}-{int(nums[1]):02d}-{int(nums[2]):02d}"

    # **수식어를 라벨에 유지한다.** 데이터 표의 열 이름이 `제 29 기 3분기` 인데
    # 라벨을 `제 29 기` 로 잘라 두면 매칭이 안 된다(실측 unknown 69.4% 의 주원인).
    # `말` 은 붙여 쓰고(`제 57 기말`) 나머지 수식어는 띄어 쓴다(`제 29 기 3분기`).
    # 매칭은 공백을 무시하지만 이 값은 사람이 읽는 라벨이다.
    label = f"제 {match.group('no')} 기" + ("" if not mod else
                                            mod if mod == "말" else f" {mod}")

    if d2 is None:                                # `… 현재` — 시점형
        return {"label": label, "start": None, "end": norm(d1),
                "type": "instant", "cumulative": None}

    if "분기" in mod:
        ptype = "quarter"
    elif "반기" in mod:
        ptype = "half"
    else:
        ptype = "annual"
    # 누적 여부는 기간 길이로 판단한다. `3분기 1.1~9.30` 은 누적이다.
    months = 0
    try:
        y1, m1, _ = (int(x) for x in norm(d1).split("-"))
        y2, m2, _ = (int(x) for x in norm(d2).split("-"))
        months = (y2 - y1) * 12 + (m2 - m1) + 1
    except (ValueError, AttributeError):
        pass
    cumulative = None if not months else (months > 3 if ptype == "quarter" else True)
    return {"label": label, "start": norm(d1), "end": norm(d2),
            "type": ptype, "cumulative": cumulative}


def raw_label(node) -> str:
    """계정명을 **정리 전 원시 텍스트**로 읽는다. 들여쓰기를 보존해야 한다.

    원문은 계층을 **전각 공백**(U+3000)으로 표현한다.

    ```xml
    <P USERMARK="F-GL11">　유동자산</P>       ← 깊이 1
    <P USERMARK="F-GL11">　　유형자산</P>      ← 깊이 2
    ```

    일반 텍스트 정리는 이를 공백으로 보고 지운다. 그러면 한 표에 두 번 나오는
    `유형자산` 을 구분할 수 없다 — 카카오 연결재무상태표는
    `자산 > 비유동자산 > 유형자산`(1조 3,220억)과 `자산 > 금융업자산 > 유형자산`(148억)을
    **모두** 담는다. 뭉개면 서로 다른 계정이 하나로 접혀 「90배 재작성」처럼 보인다.
    """
    parts: list[str] = []

    def walk(el) -> None:
        if el.text:
            parts.append(el.text)
        for child in el:
            if isinstance(child.tag, str) and child.tag.upper() != "TABLE":
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(node)
    return "".join(parts).strip("\n\r\t")


def _indent_of(raw: str, *, ascii_indent: bool = False) -> int:
    """선행 공백으로 표시한 계층 깊이. adapter별 표기만 해석한다.

    DART 최신 XBRL은 주로 전각 공백(U+3000), 2023 manual 감사인 표는 반각 공백을
    사용한다. 폭을 공통 깊이로 환산하지 않고 같은 표 안의 대소만 쓰므로, manual에서는
    반각 한 칸도 명시적 계층이다.

    다만 전각과 반각이 섞인 prefix에서는 반각을 세지 않는다. 현대오토에버 별도
    재무상태표처럼 전각 뒤에 장식용 반각 한 칸이 붙는 경우가 있기 때문이다 —

    ```
    　　무형자산                       전각2        → 깊이 2
    　　 종속기업,관계기업및공동기업투자자산   전각2 + 반각1  → 깊이 3   ← 형제인데 자식이 됐다
    　　순확정급여자산                    전각2        → 깊이 2
    ```

    그러면 `account_path` 가 `비유동자산 > 무형자산 > 종속기업…` 이 되어
    자식 합이 부모와 어긋난다. **값은 맞고 부모만 틀린** 결함이라 리포트로는 안 보인다.

    두 표기법을 한 함수에서 자동 추측하지 않는다. 기존 XBRL 427개에서는 전각만
    96.2%·섞임 3.8%였고, manual adapter의 190개 표는 U+3000이 0개다. 호출자가
    ``ascii_indent=True``를 명시한 manual 표에서만 반각 prefix를 센다. 그렇지 않으면
    현대오토에버의 전각2+장식 반각1 같은 행이 거짓 자식이 된다.
    """
    prefix_match = re.match(r"^[ \u3000]*", raw)
    prefix = prefix_match.group(0) if prefix_match is not None else ""
    return prefix.count(" ") if ascii_indent else prefix.count("\u3000")


# 일부 공시인은 첫 열의 들여쓰기를 같은 단계로 평탄화하면서도, 반복되는 계정 바로
# 앞에 의미가 명확한 소계 표제를 남긴다. 예를 들어 ``기본주당이익``은
# ``계속영업과 중단영업``과 ``계속영업`` 아래에 한 번씩 나오고, 현금흐름표의
# ``단기금융상품의 순증감``은 ``투자활동 ... 현금유입``/``현금유출`` 아래에 반복된다.
# 이 표제를 무시하면 서로 다른 값이 같은 account_path로 접힌다.
#
# 여기서는 숫자 부호나 행 순서로 부모를 *추정*하지 않는다. CI의 세 가지 폐쇄된 문맥과
# CF의 정확한 활동별 유입/유출 표제만 상태로 기억한다. 그 외 반복 계정은 계속 같은
# 경로로 남아 조회 계층의 same_document_conflict가 fail-closed 하게 한다.
_ROW_NUMBER = re.compile(r"^\(?\d+\)?[.)]?")
_ROMAN_NUMBER = re.compile(r"^(?:[IVXLCDMⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+)[.)]?", re.I)
_OCI_BUCKET = re.compile(
    r"^후속적으로당기손익으로재분류(?:되는|되지않는|될수있는)항목$")
_CF_ACTIVITY_SUBTOTAL = re.compile(
    r"^(?:투자|재무)활동(?:으로인한|으로부터의)현금(?:유입|유출)(?:액)?$")


def _semantic_core(text: str) -> str:
    """표제 판별용 문자열. 원문 account_raw/account_path에는 적용하지 않는다."""
    key = account_key(text)
    key = _SQ(key).lstrip("-ㆍ·")
    key = _ROW_NUMBER.sub("", key)
    return key.lstrip("-ㆍ·")


def _ci_attribution_parent(text: str) -> bool:
    """순이익/총포괄이익의 귀속 표제만 허용한다.

    ``VII. 당기순이익``처럼 공시인이 귀속이라는 말을 생략한 표는 Roman 대항목이라는
    추가 근거가 있어야 한다. 실제 결합은 바로 뒤의 지배/비지배 leaf 두 개로 다시
    제한하므로, 일반 순이익 행이 다른 계정을 우연히 끌어들이지 않는다.
    """
    compact = _SQ(account_key(text))
    core = _ROMAN_NUMBER.sub("", compact).lstrip("-ㆍ·")
    # ``지배기업의 소유주에게 귀속되는 당기순이익``은 leaf 자체의 설명이지
    # 귀속 breakdown 표제가 아니다. 명시 표제는 ``…의 귀속``으로 끝나는 경우만 받는다.
    explicit = (core.rstrip(":：").endswith("귀속") and
                ("순이익" in core or "총포괄이익" in core or "총포괄손익" in core))
    roman = bool(_ROMAN_NUMBER.match(compact)) and (
        ("순이익" in core and "법인세" not in core and "주당" not in core)
        or "총포괄이익" in core or "총포괄손익" in core)
    return explicit or roman


def _ci_attribution_leaf(text: str) -> bool:
    core = _semantic_core(text)
    return bool(
        re.fullmatch(r"비지배(?:주주)?지분(?:순이익)?", core)
        or re.fullmatch(
            r"(?:지배기업(?:의)?(?:소유주?|주주)?|지배주주)(?:의)?지분(?:순이익)?",
            core,
        )
    )


def _ci_oci_parent(text: str) -> bool:
    return bool(_OCI_BUCKET.fullmatch(_semantic_core(text)))


def _ci_oci_repeated_leaf(text: str) -> bool:
    # 전수 충돌에서 두 OCI bucket에 실제로 반복된 항목만 결합한다. 다른 OCI 항목은
    # 기존 들여쓰기 계층을 그대로 사용하고, 평탄한 행을 포괄적으로 추정하지 않는다.
    core = re.sub(r"^\(부의\)", "", _semantic_core(text))
    return core == "지분법자본변동"


def _ci_eps_parent(text: str) -> bool:
    return _semantic_core(text) in {"계속영업과중단영업", "계속영업"}


def _ci_eps_leaf(text: str) -> bool:
    core = _semantic_core(text)
    return bool(re.fullmatch(r"(?:기본|희석)주당(?:순)?이익(?:\(손실\))?", core))


def _cf_activity_parent(text: str) -> bool:
    return bool(_CF_ACTIVITY_SUBTOTAL.fullmatch(_semantic_core(text)))


def _cf_activity_boundary(text: str) -> bool:
    """평탄한 CF에서 마지막 유출 block이 표 끝까지 번지는 것을 막는 명시 경계."""
    core = _semantic_core(text)
    if "현금및현금성자산" in core or "현금및현금등가물" in core:
        return True
    return bool(re.match(r"^(?:영업|투자|재무)활동", core) and "현금흐름" in core)


@dataclass
class _SemanticPathState:
    """한 물리 표 안에서만 유효한, 명시 표제 기반의 좁은 parent binding."""

    mode: str | None = None
    parent_path: tuple[str, ...] = ()
    parent_outer: tuple[str, ...] = ()
    remaining: int = 0

    def clear(self) -> None:
        self.mode = None
        self.parent_path = ()
        self.parent_outer = ()
        self.remaining = 0

    def start(self, mode: str, components: list[str], remaining: int) -> None:
        self.mode = mode
        self.parent_path = tuple(components)
        self.parent_outer = tuple(components[:-1])
        self.remaining = remaining

    def bind(self, components: list[str]) -> list[str] | None:
        """현재 들여쓰기와 같은 base에서만 표제를 한 단계 삽입한다."""
        if not components or not self.parent_path:
            return None
        # 정상 들여쓰기가 이미 표제를 부모로 보존했다면 그대로 둔다.
        if self.parent_path[-1] in components[:-1]:
            return components
        if tuple(components[:-1]) != self.parent_outer:
            return None
        return [*self.parent_path, components[-1]]


def _semantic_account_components(statement: str, account: str,
                                 components: list[str],
                                 state: _SemanticPathState) -> list[str]:
    """명시 표제가 증명하는 경우에만 평탄한 반복 leaf의 부모를 복원한다."""
    if statement == "CI":
        if _ci_attribution_parent(account):
            state.start("ci_attribution", components, 2)
            return components
        if _ci_oci_parent(account):
            # 두 bucket 사이 최대 간격은 실측 10행이다. 12행을 넘기면 표제가 있어도
            # 후속 문맥으로 번지지 않게 닫는다.
            state.start("ci_oci", components, 12)
            return components
        if _ci_eps_parent(account):
            state.start("ci_eps", components, 2)
            return components

        if state.mode == "ci_attribution":
            if state.remaining > 0 and _ci_attribution_leaf(account):
                bound = state.bind(components)
                state.remaining -= 1
                if state.remaining == 0:
                    state.clear()
                return bound if bound is not None else components
            state.clear()                       # 반드시 바로 뒤의 최대 두 leaf만
        elif state.mode == "ci_eps":
            if state.remaining > 0 and _ci_eps_leaf(account):
                bound = state.bind(components)
                state.remaining -= 1
                if state.remaining == 0:
                    state.clear()
                return bound if bound is not None else components
            state.clear()
        elif state.mode == "ci_oci":
            state.remaining -= 1
            if state.remaining < 0:
                state.clear()
                return components
            if _ci_oci_repeated_leaf(account):
                bound = state.bind(components)
                return bound if bound is not None else components
        return components

    if statement == "CF":
        if _cf_activity_parent(account):
            state.start("cf_activity", components, 0)
            return components
        if state.mode == "cf_activity":
            if _cf_activity_boundary(account):
                state.clear()
                return components
            bound = state.bind(components)
            if bound is None:
                state.clear()                   # 명시 활동 base를 벗어나면 즉시 종료
                return components
            return bound
        return components

    state.clear()
    return components


def _discrete_start(end: str | None) -> str | None:
    """3개월 열의 시작일 — 종료월 기준 직전 3개월의 1일.

    `2024-06-30` → `2024-04-01`. 머리말이 누적 기간만 주므로 여기서 좁힌다.
    """
    if not end:
        return None
    try:
        year, month, _ = (int(x) for x in end.split("-"))
    except ValueError:
        return None
    month -= 2
    if month <= 0:
        year, month = year - 1, month + 12
    return f"{year:04d}-{month:02d}-01"


def _header_block(table) -> tuple[dict[str, dict], str | None]:
    """머리말 표 → (기수 라벨 → 기간 정보, 단위)."""
    periods: dict[str, dict] = {}
    aliases: list[tuple[str, dict]] = []
    unit: str | None = None
    for tr in table.iter("TR"):
        text = _text_of(tr)
        if unit is None:
            m = _UNIT.search(text)
            if m:
                unit = m.group(1).strip()
        info = parse_period(text)
        if info:
            # 정확한 라벨을 먼저 전부 넣는다. `제15기 1분기`와 `제15기`가 함께 있는
            # legacy 금융사 표에서 1분기의 축약 alias가 연간 `제15기`를 선점하면
            # 연간 열을 1분기로 오매핑한다.
            exact = _period_key(info["label"])
            periods.setdefault(exact, info)
            bare = _BARE.sub("", exact)
            aliases.append((bare, info))
            if info["type"] == "instant" and not exact.endswith(("말", "초")):
                aliases.append((exact + "말", info))
                aliases.append((bare + "말", info))
    # 정확한 라벨이 없는 경우에만 수식어 생략 alias를 허용한다.
    for alias, info in aliases:
        periods.setdefault(alias, info)
    return periods, unit


def _header_nodes(nodes: list) -> tuple[dict[str, dict], str | None]:
    """TABLE·P가 섞인 manual 머리말에서 기간과 단위를 모은다."""
    periods: dict[str, dict] = {}
    aliases: list[tuple[str, dict]] = []
    unit: str | None = None
    for node in nodes:
        texts = ([_text_of(tr) for tr in node.iter("TR")]
                 if getattr(node, "tag", None) == "TABLE" else [_text_of(node)])
        for text in texts:
            if unit is None:
                m = _UNIT.search(text)
                if m:
                    unit = m.group(1).strip()
            info = parse_period(text)
            if info:
                exact = _period_key(info["label"])
                periods.setdefault(exact, info)
                bare = _BARE.sub("", exact)
                aliases.append((bare, info))
                if info["type"] == "instant" and not exact.endswith(("말", "초")):
                    aliases.append((exact + "말", info))
                    aliases.append((bare + "말", info))
    for alias, info in aliases:
        periods.setdefault(alias, info)
    return periods, unit


def _statement_of(title: str, context: str = "") -> tuple[str, str] | None:
    """제목 → `(scope, statement)`.

    번호 접두(`2-1.` = 연결 / `4-1.` = 별도)가 1차 기준이다. 다만 **번호 없이
    `재무상태표` 라고만 쓰는 문서가 있어**(실측 7건) 그때는 상위 섹션 문맥으로 가린다 —
    `III. 재무에 관한 사항 > 2. 연결재무제표` 아래면 연결이다.
    """
    m = _TITLE.match(title)
    name = _SQ(m.group(3) if m else title)
    if not name or any(w in name for w in EXCLUDED_STATEMENTS):
        return None
    kind = next((code for word, code in STATEMENTS if word in name), None)
    if kind is None:
        return None

    if m:
        top = int(m.group(1))
        if top == 2 or "연결" in name:
            return "CFS", kind
        if top == 4:
            return "SFS", kind

    ctx = _SQ(context)
    if "연결" in name or "연결재무제표" in ctx:
        return "CFS", kind
    if "재무제표" in ctx:
        return "SFS", kind
    return None      # 번호도 문맥도 없으면 범위를 지어내지 않는다


def _statement_kind(text: str) -> str | None:
    """표시 제목 → statement 코드. 글자 사이 공백이 있어도 같은 제목이다."""
    name = _SQ(text)
    if not name:
        return None
    if any(_SQ(w) in name for w in EXCLUDED_STATEMENTS):
        return "SE"
    return next((code for word, code in STATEMENTS if _SQ(word) in name), None)


def _scope_from_context(context: str) -> str | None:
    """현재 섹션 자체가 명시한 연결/별도만 사용한다. 상위의 막연한 문구는 쓰지 않는다."""
    leaf = _SQ(context.rsplit(" > ", 1)[-1])
    # `3. 연결재무제표 주석` 같은 주석 섹션은 statement 본문이 아니다.
    if "연결재무제표" in leaf and not re.match(
            r"^(?:2\.)?연결재무제표$", leaf):
        return None
    if "연결재무제표" in leaf:
        return "CFS"
    if re.match(r"^4\.?재무제표$", leaf):
        return "SFS"
    return None


def _title_excerpt(text: str, scope: str, statement: str) -> str:
    """긴 머리말에서 원문 표제 부분만 남긴다."""
    clean = " ".join((text or "").split())
    prefix = re.split(r"\s+제\s*\d", clean, maxsplit=1)[0].strip()
    if prefix and len(prefix) <= 160:
        return prefix
    names = {"BS": "재무상태표", "IS": "손익계산서", "CI": "포괄손익계산서",
             "CF": "현금흐름표"}
    return ("연결 " if scope == "CFS" else "") + names[statement]


def _match_period(segments: list[str], periods: dict[str, dict]) -> tuple[dict | None, str]:
    """열 머리글 조각에서 기간과 `3개월`/`누적` 수식어를 함께 찾는다."""
    qualifier = _period_key(segments[-1]) if len(segments) > 1 else ""
    path_qualifier = (qualifier if qualifier in _DISCRETE or qualifier in _CUMULATIVE
                      else "")
    for seg in segments:
        key = _period_key(seg)
        if key in periods:
            return periods[key], path_qualifier
        # 일부 표는 `제55기반기3개월`을 한 셀에 붙여 쓴다. 이미 확인한 기간
        # 라벨의 정확한 접두 뒤가 허용 수식어일 때만 분리한다.
        for pkey in sorted(periods, key=len, reverse=True):
            if key.startswith(pkey):
                suffix = key[len(pkey):]
                if suffix in _DISCRETE or suffix in _CUMULATIVE:
                    return periods[pkey], suffix
    for seg in segments:
        info = periods.get(_period_key(seg))
        if info:
            return info, qualifier
    return None, qualifier


def _facts_from_table(table, *, periods: dict[str, dict], unit: str | None,
                      scope: str, statement: str, title: str, locator: str,
                      allow_unknown_periods: bool,
                      unmatched_period_headers: list[str] | None = None,
                      ascii_indent: bool = False,
                      source_sha256: str | None = None,
                      period_overrides: Mapping[
                          tuple[str, str, str], FinancialPeriodOverride] | None = None,
                      applied_overrides: dict[
                          str, FinancialPeriodOverrideApplied] | None = None,
                      ) -> list[FinancialFact]:
    """기간·단위가 확정된 데이터 표 하나를 Fact로 편다."""
    grid = build_grid(table, locator)
    if grid.n_rows < 2 or grid.n_cols < 2:
        return []

    col_period: dict[int, dict] = {}
    for col in range(1, grid.n_cols):
        path = grid.header_path(col)
        if not path:
            cell = grid.cell_at(0, col)
            path = cell.text if cell else ""
        if not path:
            continue
        segments = [seg.strip() for seg in path.split(" > ") if seg.strip()]
        info, qualifier = _match_period(segments, periods)
        if info is None:
            override = ((period_overrides or {}).get((source_sha256, locator, path))
                        if source_sha256 else None)
            if override is not None:
                col_period[col] = {
                    "label": path,
                    "start": override.period_start,
                    "end": override.period_end,
                    "type": override.period_type,
                    "cumulative": override.cumulative,
                }
                if applied_overrides is not None:
                    applied_overrides.setdefault(
                        override.decision_id,
                        FinancialPeriodOverrideApplied(
                            decision_id=override.decision_id,
                            doc_id=override.doc_id,
                            source_sha256=override.source_sha256,
                            table_locator=override.table_locator,
                            raw_header=override.raw_header,
                            corroborating_locator=override.corroborating_locator,
                        ))
                continue
            if allow_unknown_periods:
                col_period[col] = {"label": path, "start": None, "end": None,
                                   "type": "unknown", "cumulative": None}
            elif (unmatched_period_headers is not None
                  and re.search(r"제\s*\d+", path)
                  and any((cell := grid.cell_at(row, col)) is not None
                          and cell.origin_row == row and cell.origin_col == col
                          and parse_number(cell.text) is not None
                          for row in range(max(1, grid.n_head_rows), grid.n_rows))):
                unmatched_period_headers.append(path)
            continue
        entry = dict(info)
        entry["label"] = path
        if qualifier in _DISCRETE:
            entry["cumulative"] = False
            entry["start"] = _discrete_start(entry["end"])
            entry["type"] = "quarter"
        elif qualifier in _CUMULATIVE:
            entry["cumulative"] = True
        col_period[col] = entry
    if not col_period:
        return []

    out: list[FinancialFact] = []
    stack: list[tuple[int, str]] = []
    semantic_state = _SemanticPathState()
    for row in range(max(1, grid.n_head_rows), grid.n_rows):
        head = grid.cell_at(row, 0)
        if head is None or not head.text:
            continue
        account = head.text
        raw = raw_label(head.node) if head.node is not None else account
        indent = _indent_of(raw, ascii_indent=ascii_indent)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, account))
        components = [name for _, name in stack]
        components = _semantic_account_components(
            statement, account, components, semantic_state)
        account_path = " > ".join(components)
        account_depth = len(components) - 1
        row_unit = inline_unit(account)
        label_norm = ACCOUNT_NORM.get(account_key(account))
        for col, info in col_period.items():
            cell = grid.cell_at(row, col)
            if (cell is None or not cell.text or cell.origin_row != row
                    or cell.origin_col != col):
                continue
            value = parse_number(cell.text)
            cell_unit = value_unit(cell.text)
            # 계정 정규화: 값 셀의 XBRL 택소노미 ID(ACODE) 가 1차, 원문 표기 완전일치 사전이 2차.
            # 요소 ID 는 회사·연도가 달라도 같은 요소를 가리켜 표기 변형(「자산」·「자산총계」·주석 표시)에 흔들리지
            # 않는다. 자본변동표(SE)는 열이 자본 구성요소라 값 셀 코드가 행 계정이 아니므로 표기 사전만 쓴다.
            code = taxonomy_code(getattr(cell, "acode", None))
            acode_norm = ACODE_NORM.get(code) if (code and statement != "SE") else None
            out.append(FinancialFact(
                scope=scope, statement=statement, statement_title=title,
                account_raw=account, account_path=account_path,
                account_depth=account_depth,
                account_norm=(acode_norm or _attribution_norm(
                    label_norm, statement, account_path)),
                acode=code,
                account_norm_source=("acode" if acode_norm else ("label" if label_norm else None)),
                period_label=info["label"], period_start=info["start"],
                period_end=info["end"], period_type=info["type"],
                cumulative=info["cumulative"],
                value_text=cell.text, raw_value=value,
                raw_unit=row_unit or cell_unit or unit,
                unit_source=("account" if row_unit else "value" if cell_unit else "table"),
                value_status=classify_value_status(cell.text, value),
                table_locator=locator, cell_locator=cell.locator,
                logical_row=row, logical_col=col,
            ))
    # 표 머리글만 잘못 붙은 경우 비숫자 행이 대량 Fact로 둔갑하지 않게 한다.
    return out if any(f.raw_value is not None for f in out) else []


def _issue(code: str, context: str, statement: str | None, locator: str | None,
           detail: str) -> FinancialLayoutIssue:
    return FinancialLayoutIssue(code, context, statement, locator, detail)


def _from_group(group, context: str, counter: dict, facts: list,
                issues: list[FinancialLayoutIssue], detected: dict[str, int], *,
                source_sha256: str | None = None,
                period_overrides: Mapping[
                    tuple[str, str, str], FinancialPeriodOverride] | None = None,
                applied_overrides: dict[
                    str, FinancialPeriodOverrideApplied] | None = None) -> None:
    kids = [c for c in group if isinstance(c.tag, str)]
    titles = [c for c in kids if c.tag == "TITLE"]
    tables = [c for c in kids if c.tag == "TABLE"]
    gi = counter["n"]
    counter["n"] += 1
    locator = f"TABLE-GROUP[{gi}]"

    # XBRL 주석 group도 ``현금흐름표`` 같은 제목을 자주 가진다. 예를 들어
    # ``{XBRL}NT_C_D851100``은 현금흐름표 자체가 아니라 그 주석의 세부 표다.
    # TITLE 문자열만 보면 statement로 오인해, 정상적으로 본표를 추출한 문서까지
    # ``unsupported_partial_layout``으로 내리는 문제가 생긴다. ACLASS가 명시한
    # note(`NT_`)는 재무제표 Fact adapter의 대상이 아니므로 진단도 만들지 않는다.
    aclass = group.attrib.get("ACLASS", "")
    if aclass.startswith("{XBRL}NT_"):
        return

    # 최신 레이아웃: group 내부 TITLE. 2023 legacy: TITLE은 상위 SECTION에 있고
    # ACLASS와 머리말 첫 줄이 statement를 명시한다.
    if titles:
        title = _text_of(titles[0])
        found = _statement_of(title, context)
        legacy = False
    else:
        head_text = _text_of(tables[0]) if tables else ""
        statement = _statement_kind(head_text)
        scope = _scope_from_context(context)
        found = ((scope, statement) if scope and statement and statement != "SE" else None)
        title = (_title_excerpt(head_text, scope, statement)
                 if found is not None else "")
        legacy = found is not None
    if found is None:
        return
    scope, statement = found
    detected["n"] += 1
    if len(tables) < 2:
        issues.append(_issue("statement_tables_missing", context, statement, locator,
                             f"데이터 표가 필요하지만 TABLE {len(tables)}개"))
        return

    # ACLASS의 `_S`는 별도 재무제표다. context와 충돌하면 어느 쪽도 추정하지 않는다.
    if legacy and aclass.startswith("{XBRL}"):
        class_scope = "SFS" if re.search(r"_S\d*$", aclass) else "CFS"
        if class_scope != scope:
            issues.append(_issue("scope_conflict", context, statement, locator,
                                 f"context={scope}, ACLASS={aclass}"))
            return

    periods, unit = _header_block(tables[0])
    if not periods:
        issues.append(_issue("period_header_unsupported", context, statement, locator,
                             "머리말에서 실제 날짜가 붙은 기수를 찾지 못함"))
        return

    data_loc = f"{locator}/TABLE[1]"
    unmatched: list[str] = []
    got = _facts_from_table(
        tables[1], periods=periods, unit=unit, scope=scope, statement=statement,
        title=title, locator=data_loc, allow_unknown_periods=not legacy,
        unmatched_period_headers=unmatched if legacy else None,
        source_sha256=source_sha256, period_overrides=period_overrides,
        applied_overrides=applied_overrides)
    if not got:
        issues.append(_issue("statement_grid_unsupported", context, statement, data_loc,
                             "기간 열과 숫자 계정 행을 안전하게 결합하지 못함"))
        return
    facts.extend(got)
    if unmatched:
        issues.append(_issue(
            "period_columns_unsupported", context, statement, data_loc,
            "기간 머리말과 열을 매칭하지 못함: " + ", ".join(dict.fromkeys(unmatched))))


def _from_manual_section(section, context: str, table_indices: dict[int, int], facts: list,
                         issues: list[FinancialLayoutIssue], detected: dict[str, int], *,
                         source_sha256: str | None = None,
                         period_overrides: Mapping[
                             tuple[str, str, str], FinancialPeriodOverride] | None = None,
                         applied_overrides: dict[
                             str, FinancialPeriodOverrideApplied] | None = None) -> None:
    """TABLE-GROUP 없는 외부감사인 작성 표를 섹션 안의 명시적 표제 순서로 읽는다."""
    scope = _scope_from_context(context)
    if scope is None:
        return
    # ACLASS XBRL statement가 있는 섹션의 P/각주를 manual 표제로 다시 읽으면
    # 같은 문서에 거짓 unsupported 진단이 생긴다. 두 adapter는 문서 구조로 분리한다.
    if any(g.attrib.get("ACLASS", "").startswith("{XBRL}")
           for g in section.iter("TABLE-GROUP")):
        return

    pending: dict | None = None

    def abandon(detail: str) -> None:
        nonlocal pending
        if pending is not None:
            issues.append(_issue("manual_statement_unsupported", context,
                                 pending["statement"], pending.get("locator"), detail))
            pending = None

    for child in section:
        if (not isinstance(child.tag, str)
                or child.tag in ("TITLE", "TABLE-GROUP") + _SECTION_TAGS):
            continue
        text = _text_of(child)
        marker = _statement_kind(text)
        marker_periods: dict[str, dict] = {}
        if marker is not None and child.tag == "TABLE":
            marker_periods, _ = _header_nodes([child])
            # `K-IFRS 제1007호(현금흐름표)` 같은 1칸 각주를 표제로 오인하지 않는다.
            if not marker_periods:
                marker = None
        elif marker is not None and len(text) > 160:
            marker = None
        if marker is not None:
            # `P: (1) 연결재무상태표` 다음의 실제 머리말 TABLE은 같은 표제의
            # 연장이다. 새 statement로 세거나 앞 P를 실패 처리하지 않는다.
            if (pending is not None and marker == pending["statement"]
                    and child.tag == "TABLE" and marker_periods):
                pending["nodes"].append(child)
                pending["locator"] = pending.get("locator") or (
                    f"TABLE[{table_indices[id(child)]}]")
                pending["title"] = _title_excerpt(text, scope, marker)
                continue
            if pending is not None:
                abandon("다음 statement 표제가 나오기 전 데이터 표를 찾지 못함")
            if marker == "SE":
                pending = None                 # 자본변동표는 의도적으로 제외
                continue
            detected["n"] += 1
            locator = (f"TABLE[{table_indices[id(child)]}]"
                       if child.tag == "TABLE" else None)
            pending = {
                "statement": marker,
                "title": _title_excerpt(text, scope, marker),
                "nodes": [child],
                "locator": locator,
            }
            continue
        if pending is None:
            continue
        if child.tag != "TABLE":
            # 단위가 P에 따로 적힌 레이아웃도 있으므로 짧은 텍스트 노드는 보존한다.
            if child.tag in ("P", "SPAN") and text:
                pending["nodes"].append(child)
            continue

        periods, unit = _header_nodes(pending["nodes"])
        loc = f"TABLE[{table_indices[id(child)]}]"
        unmatched: list[str] = []
        got = (_facts_from_table(
            child, periods=periods, unit=unit, scope=scope,
            statement=pending["statement"], title=pending["title"], locator=loc,
            allow_unknown_periods=False,
            unmatched_period_headers=unmatched, ascii_indent=True,
            source_sha256=source_sha256, period_overrides=period_overrides,
            applied_overrides=applied_overrides) if periods else [])
        if got:
            facts.extend(got)
            if unmatched:
                issues.append(_issue(
                    "period_columns_unsupported", context, pending["statement"], loc,
                    "기간 머리말과 열을 매칭하지 못함: "
                    + ", ".join(dict.fromkeys(unmatched))))
            pending = None
        else:
            # 기간표·단위표는 데이터 표 앞에 각각 따로 올 수 있다.
            pending["nodes"].append(child)
            pending["locator"] = pending.get("locator") or loc

    if pending is not None:
        abandon("섹션 끝까지 기간 열과 숫자 계정 행을 결합하지 못함")


def extract_financials_detailed(
        root, *, source_sha256: str | None = None,
        period_override_path: Path | None = None) -> FinancialExtractionResult:
    """문서 root → Fact와 typed unsupported 진단.

    최신 XBRL, 2023 ACLASS XBRL, TABLE-GROUP 없는 manual 감사보고서 레이아웃을
    같은 의미 계약으로 읽는다. 문맥·표제·실제 날짜가 모두 확인되지 않으면 추정하지 않는다.
    """
    facts: list[FinancialFact] = []
    issues: list[FinancialLayoutIssue] = []
    group_counter = {"n": 0}
    detected = {"n": 0}
    table_indices = {id(t): i for i, t in enumerate(root.iter("TABLE"))}
    if source_sha256 is not None:
        source_sha256 = source_sha256.lower()
        if re.fullmatch(r"[0-9a-f]{64}", source_sha256, flags=re.ASCII) is None:
            raise ValueError("source_sha256은 축약하지 않은 64자리 SHA-256이어야 합니다")
    period_overrides = load_period_overrides(period_override_path)
    applied_overrides: dict[str, FinancialPeriodOverrideApplied] = {}

    def walk(el, context: str) -> None:
        for child in el:
            if not isinstance(child.tag, str):
                continue
            if child.tag in _SECTION_TAGS:
                titles = [x for x in child if isinstance(x.tag, str) and x.tag == "TITLE"]
                sub = _text_of(titles[0]) if titles else ""
                child_context = f"{context} > {sub}" if sub else context
                _from_manual_section(
                    child, child_context, table_indices, facts, issues, detected,
                    source_sha256=source_sha256, period_overrides=period_overrides,
                    applied_overrides=applied_overrides)
                walk(child, child_context)
            elif child.tag == "TABLE-GROUP":
                _from_group(
                    child, context, group_counter, facts, issues, detected,
                    source_sha256=source_sha256, period_overrides=period_overrides,
                    applied_overrides=applied_overrides)
                # 중첩 TABLE-GROUP이 있으면 빠뜨리지 않는다.
                walk(child, context)
            else:
                walk(child, context)

    walk(root, "")
    # 같은 source hash에 승인 항목이 있는데 locator/raw header guard가 하나라도 맞지 않으면
    # 조용히 건너뛰지 않는다. 원문이 그대로인데 parser 구조만 바뀐 경우도 재검수 대상이다.
    expected = [item for item in period_overrides.values()
                if item.source_sha256 == source_sha256]
    for item in expected:
        if item.decision_id not in applied_overrides:
            issues.append(_issue(
                "period_override_guard_mismatch", "", None, item.table_locator,
                f"{item.decision_id}: 승인된 raw header guard를 찾지 못함: "
                f"{item.raw_header}"))
    return FinancialExtractionResult(
        tuple(facts), tuple(issues), detected["n"],
        tuple(applied_overrides[key] for key in sorted(applied_overrides)))


def extract_financials(root) -> list[FinancialFact]:
    """호환 API. 전부 미지원이면 빈 목록 대신 typed exception으로 fail closed 한다."""
    result = extract_financials_detailed(root)
    if not result.facts and result.issues:
        raise UnsupportedFinancialLayout(result.issues)
    return list(result.facts)
