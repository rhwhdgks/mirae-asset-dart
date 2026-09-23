"""지원 회사 universe의 **승인된 표기 registry**를 코퍼스에서 유도한다.

회사 표기를 손으로 적으면 평가셋에 등장한 몇 곳만 담긴 표가 되고, 그건 과적합이다.
여기서는 코퍼스가 이미 들고 있는 **공식 출처**만 쓴다 — DART 법인명, 거래소
상장 약칭, 종목코드. 사람이 지어낸 별칭은 들어가지 않는다.

자동 확정 순서(위에서부터):

1. 법인명 정규화 exact match          ``corp_name``
2. 상장 약칭 정규화 exact match        ``listed_name``
3. 종목코드 exact match                ``stock_code``
4. 영문 법인명 정규화 exact match       ``corp_eng_name`` (``universe.csv``)
5. 사람이 승인한 구어체 별칭            ``company_aliases_approved.json``

1~4는 코퍼스에서 유도하므로 검증 가능하다. 5만 사람의 판단이 들어가며 출처가
``manual_approved`` 로 구분된다.

**하나의 표기가 하나의 corp_code 로만 이어질 때만 자동 확정한다.** 충돌하면
후보를 모두 돌려주고 호출자가 역질문한다. 편집거리·자모·부분 문자열 같은
퍼지 매칭은 여기서 하지 않는다 — 닫힌 집합에서 잰 오탐률은 처음 보는 입력의
안전성을 증명하지 못하기 때문이다.

각 항목은 출처(``source``)를 들고 다닌다. 어느 단계에서 온 표기인지 구분할 수
있어야 나중에 오해소를 추적할 수 있다.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata


#: 한글 법인 형태. 같은 회사를 ``(주)삼성전자``·``삼성전자(주)``·``삼성전자``로
#: 쓰는 차이는 회사가 다르다는 뜻이 아니다. 구두점 제거 **전에** 지운다.
_KOREAN_LEGAL_FORMS = (
    "주식회사", "(주)", "㈜", "(유)", "㈜", "유한회사", "합자회사", "합명회사",
)

#: 영문 법인 형태. **구두점을 지운 뒤 꼬리에서** 떼어낸다. 실제 DART 표기가
#: ``CO.,LTD``·``CO,.LTD``·``CO., LTD.`` 처럼 제각각이라 문자열 목록으로는
#: 다 잡히지 않는다. 구두점을 먼저 없애면 변형이 한 형태로 모인다.
#:
#: ``holdings``·``group`` 같은 **사업 성격 표현은 빼지 않는다** — 그것을 지우면
#: ``POSCO홀딩스`` 가 ``posco`` 가 되어 다른 회사와 충돌할 여지가 생긴다.
_ENGLISH_LEGAL_SUFFIXES = (
    "coltd", "companylimited", "corporation", "company", "limited",
    "ltd", "inc", "corp", "plc", "llc", "co",
)

#: 자동 확정에 쓸 출처. 순서가 곧 우선순위다.
AliasSource = str
SOURCE_CORP_NAME: AliasSource = "dart_corp_name"
SOURCE_LISTED_NAME: AliasSource = "exchange_listed_name"
SOURCE_STOCK_CODE: AliasSource = "exchange_stock_code"
#: DART 공시 법인 영문명. ``data/corpus/universe.csv`` 가 출처다.
SOURCE_ENG_NAME: AliasSource = "dart_corp_eng_name"
#: 공식 출처에 없는 구어체 표기. **코퍼스로 검증할 수 없다** — 사람이 승인한다.
SOURCE_MANUAL: AliasSource = "manual_approved"

_SOURCE_ORDER = (
    SOURCE_CORP_NAME, SOURCE_LISTED_NAME, SOURCE_STOCK_CODE,
    SOURCE_ENG_NAME, SOURCE_MANUAL)

APPROVED_ALIAS_FILE = Path(__file__).resolve().parent / "company_aliases_approved.json"
UNIVERSE_FILE = (Path(__file__).resolve().parents[2]
                 / "data/corpus/universe.csv")


class ApprovedAliasError(RuntimeError):
    """승인 별칭 파일이 지원 universe와 맞지 않는다."""


def _resolve_name(corp_code_by_name: "dict[str, str]", name: str) -> str:
    """회사명 → corp_code. 없으면 **실패로 만든다.**

    이름으로 적어 두면 사람이 검토할 수 있지만, 코퍼스가 바뀌어 그 회사가
    사라지면 별칭이 조용히 죽는다. 그래서 조용히 넘기지 않고 예외를 낸다.
    """

    code = corp_code_by_name.get(name)
    if code is None:
        raise ApprovedAliasError(
            f"승인 별칭이 지원 universe 밖 회사를 가리킵니다: {name}")
    return code


def load_approved_aliases(
        corp_code_by_name: "dict[str, str]", *, path: Path | None = None,
        ) -> "tuple[list[tuple[str, str, str]], dict[str, tuple[CompanyAlias, ...]]]":
    """``(auto 목록, 확정 불가 표기 → 후보)``.

    ``auto`` 만 registry 에 넣어 자동 확정에 쓴다. ``candidates`` 와 ``blocked`` 는
    **자동 확정하지 않는다** — 다른 실제 회사가 그 표기를 쓸 수 있다는 판정이
    붙은 것들이다. 대신 후보 목록으로 돌려줘 확인 역질문에 쓴다.

    ``candidates`` 를 registry 에 넣으면 후보가 하나일 때 호출자가 자동 확정해
    버린다. 자동 확정을 막는 것이 이 층의 존재 이유이므로 섞지 않는다.
    """

    source = path or APPROVED_ALIAS_FILE
    if not source.is_file():
        return [], {}
    document = json.loads(source.read_text(encoding="utf-8"))
    auto: list[tuple[str, str, str]] = []
    for entry in document.get("auto", []):
        surface = str(entry.get("surface") or "").strip()
        name = str(entry.get("corp_name") or "").strip()
        if not surface or not name:
            raise ApprovedAliasError(f"승인 별칭 항목이 불완전합니다: {entry}")
        auto.append((surface, _resolve_name(corp_code_by_name, name), name))

    held: dict[str, list[CompanyAlias]] = {}
    for entry in document.get("candidates", []):
        surface = str(entry.get("surface") or "").strip()
        name = str(entry.get("corp_name") or "").strip()
        if not surface or not name:
            continue
        key = normalize_company_key(surface)
        held.setdefault(key, []).append(CompanyAlias(
            normalized_key=key,
            corp_code=_resolve_name(corp_code_by_name, name),
            corp_name=name, surface=surface, source=SOURCE_MANUAL,
            auto_resolve=False))
    for entry in document.get("blocked", []):
        surface = str(entry.get("surface") or "").strip()
        names = entry.get("corp_names") or []
        # **universe 안 후보가 하나면 차단하지 않는다.** ``포스코`` 는 밖에 철강
        # 사업회사가 따로 있어 워크북에서 차단됐지만, 우리는 그 회사에 답할 수
        # 없으므로 안에서는 모호하지 않다. 생성기가 그런 항목을 ``auto`` 에
        # 넣었으므로 여기서 다시 막으면 서로 어긋난다.
        if len(names) < 2:
            continue
        key = normalize_company_key(surface)
        for name in names:
            held.setdefault(key, []).append(CompanyAlias(
                normalized_key=key,
                corp_code=_resolve_name(corp_code_by_name, name),
                corp_name=name, surface=surface, source=SOURCE_MANUAL,
                auto_resolve=False))
    return auto, {key: tuple(rows) for key, rows in held.items()}


@dataclass(frozen=True, slots=True)
class CompanyAlias:
    """표기 하나와 그 출처. ``auto_resolve`` 는 충돌 검사 결과다."""

    normalized_key: str
    corp_code: str
    corp_name: str
    surface: str
    source: AliasSource
    auto_resolve: bool


def normalize_company_key(value: str) -> str:
    """표기 차이를 지운다. **회사를 고르지는 않는다.**

    NFKC → 소문자 → 법인 형태 제거 → 공백·구두점 제거 순이다. 법인 형태를
    공백 제거보다 **먼저** 지워야 ``(주) 삼성전자`` 와 ``삼성전자`` 가 같아진다.
    """

    text = unicodedata.normalize("NFKC", value).casefold()
    for form in _KOREAN_LEGAL_FORMS:
        text = text.replace(form.casefold(), " ")
    text = re.sub(r"[\s\-_.,·/()\[\]&\u2019\']+", "", text)
    # 꼬리의 영문 법인 형태를 **반복해서** 떼어낸다. ``co``+``ltd`` 처럼 두 겹인
    # 표기가 있어 한 번만 지우면 남는다. 남는 부분이 없어지면 되돌린다 —
    # 회사 이름 전체가 법인 형태인 경우까지 지우면 키가 사라진다.
    while True:
        for suffix in _ENGLISH_LEGAL_SUFFIXES:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                break
        else:
            break
    return text


def load_universe_english_names(
        corp_code_by_name: "dict[str, str]", *, path: Path | None = None,
        ) -> "list[tuple[str, str, str]]":
    """지원 universe 표의 영문 법인명. ``(surface, corp_code, corp_name)``.

    **공식 출처다** — 사람이 지어낸 별칭이 아니라 DART 등록 영문명이므로
    `manual_approved` 와 달리 검증 가능하다. 표가 없으면 조용히 건너뛴다.
    """

    source = path or UNIVERSE_FILE
    if not source.is_file():
        return []
    rows: list[tuple[str, str, str]] = []
    with source.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            english = (row.get("corp_eng_name") or "").strip()
            name = (row.get("corp_name") or "").strip()
            code = row.get("corp_code") or corp_code_by_name.get(name)
            if not english or not code or not name:
                continue
            rows.append((english, code, name))
    return rows


def build_alias_registry(
        rows: "list[tuple[str, str, str | None, str | None]]",
        approved: "list[tuple[str, str, str]] | None" = None,
        english: "list[tuple[str, str, str]] | None" = None,
        ) -> dict[str, tuple[CompanyAlias, ...]]:
    """``(corp_code, corp_name, listed_name, stock_code)`` 에서 registry를 만든다.

    같은 정규화 키에 회사가 둘 이상 걸리면 그 키의 모든 항목이
    ``auto_resolve=False`` 가 된다 — **충돌은 조용히 하나를 고르는 대신
    후보를 남긴다.**
    """

    staged: dict[str, list[CompanyAlias]] = {}
    for corp_code, corp_name, listed_name, stock_code in rows:
        for surface, source in (
            (corp_name, SOURCE_CORP_NAME),
            (listed_name, SOURCE_LISTED_NAME),
            (stock_code, SOURCE_STOCK_CODE),
        ):
            if not surface:
                continue
            key = normalize_company_key(surface)
            if not key:
                continue
            bucket = staged.setdefault(key, [])
            if any(item.corp_code == corp_code and item.source == source
                   for item in bucket):
                continue
            bucket.append(CompanyAlias(
                normalized_key=key, corp_code=corp_code, corp_name=corp_name,
                surface=surface, source=source, auto_resolve=True))

    for surface, corp_code, corp_name in (english or ()):
        key = normalize_company_key(surface)
        if not key:
            continue
        bucket = staged.setdefault(key, [])
        if any(item.corp_code == corp_code and item.source == SOURCE_ENG_NAME
               for item in bucket):
            continue
        bucket.append(CompanyAlias(
            normalized_key=key, corp_code=corp_code, corp_name=corp_name,
            surface=surface, source=SOURCE_ENG_NAME, auto_resolve=True))

    for surface, corp_code, corp_name in (approved or ()):
        key = normalize_company_key(surface)
        if not key:
            continue
        bucket = staged.setdefault(key, [])
        if any(item.corp_code == corp_code and item.source == SOURCE_MANUAL
               for item in bucket):
            continue
        bucket.append(CompanyAlias(
            normalized_key=key, corp_code=corp_code, corp_name=corp_name,
            surface=surface, source=SOURCE_MANUAL, auto_resolve=True))

    registry: dict[str, tuple[CompanyAlias, ...]] = {}
    for key, bucket in staged.items():
        unique_codes = {item.corp_code for item in bucket}
        resolvable = len(unique_codes) == 1
        registry[key] = tuple(sorted(
            (CompanyAlias(
                normalized_key=item.normalized_key, corp_code=item.corp_code,
                corp_name=item.corp_name, surface=item.surface,
                source=item.source, auto_resolve=resolvable)
             for item in bucket),
            key=lambda item: (_SOURCE_ORDER.index(item.source), item.corp_code),
        ))
    return registry


def lookup_alias(
        registry: "dict[str, tuple[CompanyAlias, ...]]", mention: str,
        ) -> tuple[CompanyAlias, ...]:
    """정규화 exact match만 한다. 없으면 빈 tuple."""

    return registry.get(normalize_company_key(mention), ())
