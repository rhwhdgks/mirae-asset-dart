"""DART XML 정제기.

제공 코퍼스의 DART XML은 raw `&`, 장식용 `<...>`, 손상된 속성 따옴표 때문에
그대로는 표준 XML 파서로 열리지 않는 문서가 많다. 원인은 셋이다.

1. 이스케이프되지 않은 `&` — `R&D`, `MD&A` 등
2. 장식용 raw `<` `>` — `<이사ㆍ감사 전체의 보수현황>`, `< TV 시장점유율 추이 >`,
   `(<60k €)`, `<PLAYSTATION>` 처럼 본문에서 괄호로 쓰인 부등호
3. 속성값 따옴표 손상 — `ENG=""Other receivables` 또는 `ENG="Other receivables""`

이 모듈은 셋 모두를 **원문 위치가 기록된 가역 패치**로 고친다. 3은 시작 태그의
속성 경계에서만 `&quot;` 로 바꾼다. 따라서 본문 텍스트의 `""` 는 건드리지 않는다.

**관대한 HTML 파서를 그냥 쓰면 안 된다.** `<이사ㆍ감사 전체의 보수현황>`을 태그로 오인해
표 머리글 텍스트를 통째로 삼킨다. 손실이 눈에 띄지 않는 것이 더 위험하다.
정제를 먼저 하고 strict 파서로 가야 텍스트와 TABLE/TR 구조가 보존된다.

## 무손실은 길이로 증명되지 않는다 (P0-6)

예전 검증은 `len(sanitize(s)) - len(s)` 가 `&`×4 + `<`×3 과 같은지만 봤다.
**늘어난 글자 수가 맞아도 엉뚱한 곳을 바꿨을 수 있다.** 길이는 필요조건일 뿐이다.

그래서 정제가 **무엇을 어디서 바꿨는지 패치 로그로 남기고**, 그 로그로 정제본을 되돌려
원문과 바이트 단위로 같은지 확인한다(`verify_roundtrip`). 코퍼스에 CDATA·주석·DOCTYPE 은
없고 `<?xml?>` 선언만 있어(실측 2,732/2,732) 단일 패스로 정확한 로그를 만들 수 있다.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

__all__ = [
    "DART_TAGS", "Patch", "sanitize", "sanitize_with_patches", "unsanitize",
    "verify_roundtrip",
]

#: 코퍼스 전수 태그 빈도 조사로 확정한 실제 DART 스키마 태그.
#: 이 목록 밖에서 `<이름>` 형태로 등장하는 48종(PLAYSTATION, SONY, XI. 등)은 전부 장식 괄호다.
DART_TAGS: frozenset[str] = frozenset(
    {
        "DOCUMENT", "DOCUMENT-NAME", "FORMULA-VERSION", "COMPANY-NAME",
        "SUMMARY", "EXTRACTION", "BODY", "LIBRARY", "COVER", "COVER-TITLE",
        "SECTION-1", "SECTION-2", "SECTION-3", "SECTION-4",
        "TITLE", "P", "SPAN", "A", "PGBRK",
        "TABLE-GROUP", "TABLE", "COLGROUP", "COL", "THEAD", "TBODY",
        "TR", "TD", "TH", "TU", "TE",
        "IMAGE", "IMG", "IMG-CAPTION", "CORRECTION",
    }
)

_BARE_AMP = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9A-Fa-f]+);)")

# 순서가 중요하다. 선언·주석·CDATA·DOCTYPE 을 먼저 통째로 잡아 보존한 뒤,
# 남은 `<` 중 화이트리스트 태그만 태그로 인정하고, 마지막으로 bare `&` 를 잡는다.
_TOKEN = re.compile(
    r"(?P<keep><\?[^>]*\?>|<!--.*?-->|<!\[CDATA\[.*?\]\]>|<!DOCTYPE[^>]*>)"
    r"|(?P<tag><(?P<slash>/?)(?P<name>[A-Za-z][-A-Za-z0-9_:.]*)(?=[\s/>]))"
    r"|(?P<lt><)"
    r"|(?P<amp>&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9A-Fa-f]+);))",
    re.S,
)

# 손상된 속성 따옴표는 **DART 시작 태그 안에서만** 찾는다. 본문의 `그는 ""예""라고 했다`를
# 전역 치환하면 원문 의미를 바꾸는 조용한 오염이다. 장식용 `<PLAYSTATION>`도 제외한다.
_START_TAG = re.compile(
    r"<(?P<name>[A-Za-z][-A-Za-z0-9_:.]*)(?=[\s/>])[^<>]*>", re.S)
_ATTR_NAME = r"[A-Za-z_:][-A-Za-z0-9_:.]*"
_ATTR_OPEN = re.compile(rf"\b{_ATTR_NAME}\s*=\s*\"")
# 닫는 quote 뒤에는 공백+다음 속성 또는 태그 끝만 올 수 있다.
_ATTR_BOUNDARY = re.compile(rf"(?:\s+{_ATTR_NAME}\s*=|\s*/?>$)", re.S)


@dataclass(frozen=True)
class Patch:
    """정제가 바꾼 한 곳. `offset` 은 **정제본** 기준이다.

    정제본 기준으로 잡는 이유는 역변환이 정제본을 입력으로 받기 때문이다.
    원문 기준이면 누적 증가분을 다시 계산해야 한다.
    """

    offset: int
    original: str        #: 원문 조각 — `"&"`, `"<..."`, 또는 손상된 `"`
    replacement: str     #: 정제 결과 — `"&amp;"`, `"&lt;..."`, 또는 `"&quot;"`


def _quote_repair_offsets(source: str) -> set[int]:
    """시작 태그 속성값 안의 invalid literal quote 위치만 돌려준다.

    속성의 닫는 따옴표는 뒤에 `다음속성=` 또는 태그 끝만 올 수 있다. 그 전에 만나는
    따옴표는 `ATTR=""X`, `ATTR="X""`, `("SOHO")` 모두 값에 섞인 문자이므로
    `&quot;` 대상이다. DART 시작 태그만 훑어 본문 텍스트는 절대 바꾸지 않는다.
    """
    offsets: set[int] = set()
    for tag in _START_TAG.finditer(source):
        if tag.group("name").upper() not in DART_TAGS:
            continue
        raw = tag.group(0)
        cursor = 0
        while attr := _ATTR_OPEN.search(raw, cursor):
            quote = raw.find('"', attr.end())
            while quote >= 0 and not _ATTR_BOUNDARY.match(raw, quote + 1):
                offsets.add(tag.start() + quote)
                quote = raw.find('"', quote + 1)
            if quote < 0:
                break  # 닫는 경계가 없는 별도 손상은 recover parser에 맡긴다
            cursor = quote + 1
    return offsets


def sanitize_with_patches(source: str) -> tuple[str, list[Patch]]:
    """정제본과 패치 로그. 로그만으로 원문을 완전히 되돌릴 수 있다."""
    # (원문 start, 원문 end, replacement). 서로 겹치지 않는 수정을 모아 원문에 한 번만
    # 적용한다. 그래야 따옴표 복구까지 포함한 최종 offset이 정확하다.
    replacements: list[tuple[int, int, str]] = [
        (offset, offset + 1, "&quot;") for offset in _quote_repair_offsets(source)
    ]
    for match in _TOKEN.finditer(source):
        token = match.group(0)
        if match.group("keep") or (match.group("name")
                                   and match.group("name").upper() in DART_TAGS):
            continue
        # 태그가 아닌 `<` 는 **첫 글자만** 바꾼다. 토큰 전체를 `&lt;` 로 갈면
        # `<PLAYSTATION` 이 `&lt;` 가 되어 이름이 사라진다.
        replacement = ("&amp;" if match.group("amp")
                       else "&lt;" + token[1:])
        replacements.append((match.start(), match.end(), replacement))

    out: list[str] = []
    patches: list[Patch] = []
    grown = 0          # 지금까지의 누적 증가분 — 정제본 기준 offset 계산용
    pos = 0
    for start, end, replacement in sorted(replacements):
        if start < pos:
            raise ValueError(f"정제 패치 중첩: {start} < {pos}")
        original = source[start:end]
        out.append(source[pos:start])
        out.append(replacement)
        patches.append(Patch(offset=start + grown, original=original,
                             replacement=replacement))
        grown += len(replacement) - len(original)
        pos = end
    out.append(source[pos:])
    return "".join(out), patches


def sanitize(source: str) -> str:
    """DART 원문 문자열을 well-formed XML로 정제한다.

    원본 파일은 수정하지 않는다. 정제본은 파생 산출물이며 원본에서 언제든 재생성된다.
    """
    return sanitize_with_patches(source)[0]


def unsanitize(sanitized: str, patches: list[Patch]) -> str:
    """정제본 + 패치 로그 → 원문. 어긋나면 `ValueError` 로 올린다.

    조용히 다른 문자열을 돌려주면 검증이 의미를 잃는다.
    """
    out: list[str] = []
    pos = 0
    for patch in patches:
        if patch.offset < pos:
            raise ValueError(f"패치 offset 역행: {patch.offset} < {pos}")
        found = sanitized[patch.offset:patch.offset + len(patch.replacement)]
        if found != patch.replacement:
            raise ValueError(
                f"offset {patch.offset} 에 {patch.replacement!r} 가 아니라 {found!r}")
        out.append(sanitized[pos:patch.offset])
        out.append(patch.original)
        pos = patch.offset + len(patch.replacement)
    out.append(sanitized[pos:])
    return "".join(out)


def verify_roundtrip(source: str) -> tuple[bool, dict]:
    """`(무손실인가, 상세)`. 해시를 비교하므로 길이만 맞는 경우를 걸러낸다."""
    sanitized, patches = sanitize_with_patches(source)
    try:
        restored = unsanitize(sanitized, patches)
        error = None
    except ValueError as exc:
        restored, error = "", str(exc)
    src_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]
    got_hash = hashlib.sha256(restored.encode("utf-8")).hexdigest()[:32]
    return src_hash == got_hash, {
        "source_sha256": src_hash, "restored_sha256": got_hash,
        "patches": len(patches),
        # `<PLAYSTATION` 처럼 **이름이 붙은 비태그 토큰**은 `<` 하나가 아니다.
        # `original == "<"` 로만 세면 합계가 맞지 않는다(실측 663건이 빠졌다).
        "amp": sum(1 for p in patches if p.original.startswith("&")),
        "lt": sum(1 for p in patches if p.original.startswith("<")),
        "quote": sum(1 for p in patches if p.replacement == "&quot;"),
        "named_lt": sum(1 for p in patches if len(p.original) > 1),
        "growth": len(sanitized) - len(source),
        "error": error,
    }
