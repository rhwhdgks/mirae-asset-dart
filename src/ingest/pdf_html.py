"""`pdf+html` 문서 3건 파서 — 텍스트가 하나도 안 나오던 구멍을 막는다.

## 무엇이 문제였나

코퍼스 4,204건 중 3건은 DART XML 이 아니라 **PDF + 뷰어 HTML** 로만 제공된다.
파이프라인이 이를 건너뛰어(`skipped:pdf_html`) **텍스트가 단 한 글자도 나오지 않았다.**
0.07% 지만 「한화에어로스페이스 2026년 1분기」를 물으면 답이 아예 없다.

| 문서 | PDF | 뷰어 HTML |
|---|---|---|
| KB금융 `[기재정정]사업보고서 (2025.12)` | 1,085쪽 | iframe 껍데기 |
| 한화오션 `[기재정정]분기보고서 (2024.03)` | 252쪽 | iframe 껍데기 |
| 한화에어로스페이스 `분기보고서 (2026.03)` | 447쪽 | **본문 (표 2,045개)** |

## 뷰어 HTML 은 두 종류다

셋 다 `*_viewer.html` 이 있지만 내용이 다르다.

- **본문형** — `<table>` 2,045개, iframe 없음. 표 구조가 살아 있어 XML 문서와 동급으로 다룰 수 있다.
- **껍데기형** — `<iframe>` 1개 + `<script>` 21개, 표 0개. 텍스트로 보이는 것이 대부분 CSS·JS 다.

그래서 **본문형이면 HTML 을 쓰고, 껍데기형이면 PDF 텍스트로 되돌린다.** 표를 살릴 수 있으면
살리는 게 낫다.

## PDF 경로는 구조가 열등하다 — 숨기지 않는다

PDF 텍스트 추출은 표를 평문으로 만든다. 행·열이 사라지므로 `n_tables=0` 이고
`parse_mode="pdf_text"` 로 표시한다. 같은 Section 인 척하면 하위 단계가 표가 있는 문서와
구분할 수 없다.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from lxml import html as lxml_html

from .dart_xml import Section, _table_to_markdown
from .table_grid import default_text

__all__ = [
    "CrossCheckStatus", "CrossCheckConflict", "CrossCheckResult",
    "SourceCoverage", "SourceParseResult", "ParsedViewer",
    "is_content_html", "cross_check_sources", "parse_pdf_html",
]

#: `I. 회사의 개요` → 1단, `1. 회사의 개요` → 2단, `가.`·`(1)` → 3단
_LEVEL = (
    (re.compile(r"^[IVXLC]+\.\s"), 1),
    (re.compile(r"^\d+\.\s"), 2),
    (re.compile(r"^[가-하]\.\s"), 3),
    (re.compile(r"^\(\d+\)\s"), 4),
)
_MIN_TITLE, _MAX_TITLE = 2, 80

# 검색/LLM Section에 실행 가능한 문서 조각이 섞이면 안 된다. 원문 파일 자체는 건드리지
# 않고, 파싱용 DOM에서만 제거한다. `object/embed/template`도 script와 같은 경계로 본다.
_ACTIVE_TAGS = frozenset({
    "script", "style", "iframe", "object", "embed", "template", "noscript",
})
_ACTIVE_URI = re.compile(r"^\s*(?:javascript|vbscript|data)\s*:", re.I)
_HIDDEN_STYLE = re.compile(
    r"(?:display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0(?:[;\s]|$))",
    re.I,
)
_HIDDEN_CLASSES = frozenset({"hidden", "d-none", "display-none", "visibility-hidden"})


class CrossCheckStatus(str, Enum):
    """두 독립 source의 비교 결과. 불확실한 유사도를 ``matched``로 승격하지 않는다."""

    MATCHED = "matched"
    PARTIAL_CONSISTENT = "partial_consistent"
    CONFLICT = "conflict"
    NOT_COMPARABLE = "not_comparable"
    SINGLE_SOURCE = "single_source"


@dataclass(frozen=True)
class CrossCheckConflict:
    """같은 section/문장 뼈대에 서로 다른 숫자가 있는 명시적 충돌."""

    claim_key: str
    html_values: tuple[str, ...]
    pdf_values: tuple[str, ...]


@dataclass(frozen=True)
class CrossCheckResult:
    status: CrossCheckStatus
    compared_sources: tuple[str, ...]
    reason: str
    conflicts: tuple[CrossCheckConflict, ...] = ()
    primary_digest: str | None = None
    alternate_digest: str | None = None
    compared_claims: int = 0
    matching_claims: int = 0
    unresolved_claims: int = 0


@dataclass(frozen=True)
class SourceCoverage:
    """한 SourceFile에서 실제로 구조화한 범위와 보안 제거량."""

    n_sections: int
    n_chars: int
    n_tables: int
    n_pages: int | None = None
    pages_with_text: int | None = None
    n_lines: int | None = None
    page_text_coverage: float | None = None
    active_nodes_removed: int = 0
    active_attributes_removed: int = 0


@dataclass(frozen=True)
class SourceParseResult:
    """HTML/PDF 하나를 다른 source와 무관하게 파싱한 결과."""

    source_path: Path
    source_kind: str                 # viewer_html | pdf
    parse_mode: str                  # viewer_html | viewer_html_shell | pdf_text | *_failed
    sections: tuple[Section, ...]
    coverage: SourceCoverage
    locator_kind: str                # dom_ordinal | page_line
    locator_limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: str | None = None

    @property
    def usable(self) -> bool:
        return bool(self.sections) and (self.coverage.n_chars > 0 or self.coverage.n_tables > 0)


class ParsedViewer:
    """기존 ``ParsedDocument`` 호환 view + source별 독립 결과.

    ``sections``/``parse_mode``는 선택된 primary의 하위 호환 view다. 원본별 결과는
    ``primary``·``alternates``·``source_results``에 그대로 남아 있어 build 단계가
    두 SourceFile을 각각 Section/Evidence로 발행할 수 있다.
    """

    def __init__(self, document_name: str, company_name: str,
                 sections: list[Section], parse_mode: str, *,
                 source_results: Iterable[SourceParseResult] = (),
                 primary: SourceParseResult | None = None,
                 cross_check: CrossCheckResult | None = None) -> None:
        self.document_name = document_name
        self.company_name = company_name
        self.sections = sections
        self.parse_mode = parse_mode
        self.source_results = tuple(source_results)
        self.primary = primary
        self.alternates = tuple(r for r in self.source_results if r is not primary)
        self.alternate = self.alternates[0] if self.alternates else None
        self.source_modes = {str(r.source_path): r.parse_mode for r in self.source_results}
        self.coverage_by_source = {str(r.source_path): r.coverage for r in self.source_results}
        self.cross_check = cross_check or CrossCheckResult(
            CrossCheckStatus.SINGLE_SOURCE, (), "교차검증할 source가 하나뿐임")
        self.cross_check_status = self.cross_check.status
        self.cross_check_reason = self.cross_check.reason
        self.cross_check_conflicts = self.cross_check.conflicts


def _hidden(el) -> bool:
    attrs = {str(k).lower(): str(v) for k, v in el.attrib.items()}
    if "hidden" in attrs or attrs.get("aria-hidden", "").lower() == "true":
        return True
    if str(el.tag).lower() == "input" and attrs.get("type", "").lower() == "hidden":
        return True
    if _HIDDEN_STYLE.search(attrs.get("style", "")):
        return True
    classes = frozenset(attrs.get("class", "").lower().split())
    return bool(classes & _HIDDEN_CLASSES)


def _sanitize_html(root) -> tuple[int, int]:
    """파싱용 DOM에서 active/hidden 노드와 실행 속성을 제거한다.

    파일의 raw bytes는 수정하지 않는다. ``drop_tree``는 제거 노드의 tail(그 뒤의 정상
    텍스트)을 보존하므로 ``<script/>뒤 문장``에서 뒤 문장까지 잃지 않는다.
    """
    removed_nodes = removed_attrs = 0
    for el in reversed(list(root.iter())):
        if not isinstance(el.tag, str):
            continue
        tag = el.tag.lower()
        if tag in _ACTIVE_TAGS or _hidden(el):
            if el.getparent() is not None:
                el.drop_tree()
                removed_nodes += 1
            else:
                # root 자체는 떼어낼 수 없으므로 내용과 속성을 비운다.
                el.clear()
                removed_nodes += 1
            continue
        for attr in list(el.attrib):
            name = attr.lower()
            value = el.attrib.get(attr, "")
            if name.startswith("on") or (
                    name in {"href", "src", "action", "formaction", "xlink:href"}
                    and _ACTIVE_URI.match(value)):
                del el.attrib[attr]
                removed_attrs += 1
    return removed_nodes, removed_attrs


def _visible_text(root) -> str:
    """active/hidden subtree를 제외한 사람이 읽는 텍스트."""
    parts: list[str] = []
    for node in root.xpath("//text()[not(ancestor::script) and not(ancestor::style) "
                           "and not(ancestor::iframe) and not(ancestor::object) "
                           "and not(ancestor::embed) and not(ancestor::template) "
                           "and not(ancestor::noscript)]"):
        parent = node.getparent()
        if parent is not None and any(_hidden(a) for a in (parent, *parent.iterancestors())):
            continue
        text = " ".join(str(node).split())
        if text:
            parts.append(text)
    return " ".join(parts)


def is_content_html(root) -> bool:
    """표 또는 번호 제목+본문이 있는 viewer HTML인가.

    iframe의 존재만으로 본문을 버리지 않는다. active subtree를 제외한 **보이는** 내용으로
    판정하므로 본문 옆에 보조 iframe이 있어도 HTML 구조를 보존할 수 있다.
    """
    visible_tables = [t for t in root.xpath("//table")
                      if not any(_hidden(a) for a in (t, *t.iterancestors()))]
    if visible_tables:
        return True
    candidates = root.xpath("//p|//a")
    has_title = any(
        (text := default_text(el))
        and _MIN_TITLE <= len(text) <= _MAX_TITLE
        and _level_of(text) is not None
        and not any(_hidden(a) for a in (el, *el.iterancestors()))
        for el in candidates
    )
    return has_title and len(_visible_text(root)) >= 12


def _level_of(text: str) -> int | None:
    for pattern, level in _LEVEL:
        if pattern.match(text):
            return level
    return None


def _sections_from_html(root, locator_prefix: str = "") -> list[Section]:
    """문서 순서대로 훑어 제목에서 섹션을 끊고 표는 Markdown 으로 남긴다.

    `<h1>`~`<h4>` 가 없어 제목을 번호 패턴으로 판정한다(`I.`·`1.`·`가.`·`(1)`).
    """
    sections: list[Section] = []
    stack: list[str] = []
    buffer: list[str] = []
    tables = 0
    order = 0
    locator = ""
    counter: dict[str, int] = {}

    def flush() -> None:
        nonlocal buffer, tables, order
        text = "\n\n".join(p for p in (b.strip() for b in buffer) if p)
        # **본문이 비어도 섹션을 만든다.** 안 만들면 그 제목이 조상 스택에 들어가지
        # 못해 뒤따르는 하위 제목이 **직전 갈래의 섹션을 부모로** 갖는다
        # (실측 713건). 제목 자체도 사라진다. 검색에는 `index_eligible` 이 막는다.
        if stack:
            sections.append(Section(
                path=" > ".join(stack), level=len(stack), order=order,
                text=text, n_tables=tables,
                source_locator=locator or f"BLOCK[{order}]"))
            order += 1
        buffer, tables = [], 0

    body = root.xpath("//body")
    for el in (body[0] if body else root).iter():
        tag = el.tag.lower() if isinstance(el.tag, str) else el.tag
        if not isinstance(tag, str):
            continue
        if (tag in _ACTIVE_TAGS
                or any(_hidden(a) or str(a.tag).lower() in _ACTIVE_TAGS
                       for a in (el, *el.iterancestors()))):
            continue
        if tag == "table":
            if (el.getparent() is not None
                    and str(el.getparent().tag).lower() in ("td", "th")):
                continue                        # 중첩 표는 상위 표 렌더링에서 함께 처리된다
            idx = counter.get("TABLE", 0)
            counter["TABLE"] = idx + 1
            markdown = _table_to_markdown(el, None)
            if markdown:
                buffer.append(markdown)
                tables += 1
        elif tag in ("p", "a"):
            if el.xpath(".//table"):
                continue
            text = default_text(el)
            if not text or len(text) > _MAX_TITLE * 8:
                if text:
                    buffer.append(text)
                continue
            level = _level_of(text) if _MIN_TITLE <= len(text) <= _MAX_TITLE else None
            if level is None:
                buffer.append(text)
                continue
            flush()
            idx = counter.get("TITLE", 0)
            counter["TITLE"] = idx + 1
            locator = f"{locator_prefix}TITLE[{idx}]"
            stack = stack[: level - 1] + [text]
    flush()
    return _dedupe(sections)


def _dedupe(sections: list[Section]) -> list[Section]:
    """같은 텍스트가 `<p>` 와 `<a>` 양쪽에서 잡히는 경우를 걷어낸다."""
    out: list[Section] = []
    seen: set[tuple[str, str]] = set()
    for s in sections:
        key = (s.path, s.text[:200])
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _sections_with_pdf_metrics(
        path: Path) -> tuple[list[Section], int, int, int, tuple[str, ...]]:
    """PDF 텍스트 → 섹션. **표 구조는 복원하지 않는다** — 평문이다.

    locator 는 **쪽·줄 번호**(`PAGE[12]/LINE[34]`)다. 순번(`PDFTEXT[n]`)으로 두면
    추출기가 섹션을 하나 더 잡는 순간 뒤쪽 ID 가 전부 밀린다(L-03 과 같은 문제).
    쪽·줄은 1-based이며 원문에 매인 값이라 섹션 판정이 바뀌어도 그대로다.

    단, pypdf의 line은 **추출된 text layer의 줄**이지 시각 좌표가 아니다. bbox를
    안정적으로 얻을 수 없는 한계를 SourceParseResult에 별도로 명시한다.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    #: (쪽 번호, 쪽 안 줄 번호, 줄 텍스트)
    lines: list[tuple[int, int, str]] = []
    pages_with_text = 0
    n_lines = 0
    warnings: list[str] = []
    for page_no, page in enumerate(reader.pages, 1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:               # 한 쪽 때문에 다른 1,084쪽을 버리지 않는다
            warnings.append(f"PAGE[{page_no}] {type(exc).__name__}: {exc}"[:200])
            continue
        if page_text.strip():
            pages_with_text += 1
        page_lines = page_text.split("\n")
        n_lines += len(page_lines)
        for line_no, raw in enumerate(page_lines, 1):
            lines.append((page_no, line_no, raw))

    sections: list[Section] = []
    stack: list[str] = []
    buffer: list[str] = []
    order = 0
    locator = ""

    def flush() -> None:
        nonlocal buffer, order
        text = "\n".join(x for x in (b.strip() for b in buffer) if x)
        # 본문이 비어도 만든다 — 위 `_sections_from_html` 과 같은 이유다.
        if stack and locator:
            sections.append(Section(
                path=" > ".join(stack), level=len(stack), order=order,
                text=text, n_tables=0, source_locator=locator))
            order += 1
        buffer = []

    for page_no, line_no, raw in lines:
        line = " ".join(raw.split())
        if not line:
            continue
        # 목차의 점선 행(`1. 회사의 개요 ......... 3`)은 제목이 아니다
        if "...." in line:
            continue
        level = _level_of(line) if _MIN_TITLE <= len(line) <= _MAX_TITLE else None
        if level is None:
            buffer.append(line)
            continue
        flush()
        locator = f"PAGE[{page_no}]/LINE[{line_no}]"
        stack = stack[: level - 1] + [line]
    flush()
    if not sections:
        visible = [(p, line, " ".join(raw.split())) for p, line, raw in lines if raw.strip()]
        if visible:
            # 번호 제목이 없는 PDF도 텍스트를 잃지 않는다. 위치는 첫 추출 줄에 고정한다.
            first_page, first_line, _ = visible[0]
            sections.append(Section(
                path="문서 본문", level=1, order=0,
                text="\n".join(text for _, _, text in visible), n_tables=0,
                source_locator=f"PAGE[{first_page}]/LINE[{first_line}]"))
    return sections, len(reader.pages), pages_with_text, n_lines, tuple(warnings)


def _coverage(sections: Iterable[Section], *, n_pages: int | None = None,
              pages_with_text: int | None = None, n_lines: int | None = None,
              active_nodes_removed: int = 0,
              active_attributes_removed: int = 0) -> SourceCoverage:
    items = tuple(sections)
    return SourceCoverage(
        n_sections=len(items),
        n_chars=sum(s.n_chars for s in items),
        n_tables=sum(s.n_tables for s in items),
        n_pages=n_pages,
        pages_with_text=pages_with_text,
        n_lines=n_lines,
        page_text_coverage=(pages_with_text / n_pages
                            if n_pages and pages_with_text is not None else None),
        active_nodes_removed=active_nodes_removed,
        active_attributes_removed=active_attributes_removed,
    )


def _parse_html_source(path: Path) -> SourceParseResult:
    try:
        root = lxml_html.fromstring(path.read_text(encoding="utf-8", errors="replace"))
        removed_nodes, removed_attrs = _sanitize_html(root)
        content = is_content_html(root)
        sections = tuple(_sections_from_html(root)) if content else ()
        mode = "viewer_html" if sections else "viewer_html_shell"
        warnings = (() if sections else
                    ("active content 제거 후 구조화할 viewer 본문이 없음",))
        return SourceParseResult(
            source_path=path, source_kind="viewer_html", parse_mode=mode,
            sections=sections,
            coverage=_coverage(
                sections, active_nodes_removed=removed_nodes,
                active_attributes_removed=removed_attrs),
            locator_kind="dom_ordinal",
            locator_limitations=(
                "TITLE/TABLE ordinal은 보안 필터를 적용한 DOM의 구조 위치이며 bbox가 아니다",
            ),
            warnings=warnings,
        )
    except Exception as exc:
        return SourceParseResult(
            source_path=path, source_kind="viewer_html", parse_mode="viewer_html_failed",
            sections=(), coverage=_coverage(()), locator_kind="dom_ordinal",
            locator_limitations=("HTML 파싱 실패로 locator를 만들지 못함",),
            error=f"{type(exc).__name__}: {exc}"[:300],
        )


def _parse_pdf_source(path: Path) -> SourceParseResult:
    try:
        sections, pages, text_pages, lines, warnings = _sections_with_pdf_metrics(path)
        sections_tuple = tuple(sections)
        mode = "pdf_text" if sections_tuple else "pdf_no_text"
        return SourceParseResult(
            source_path=path, source_kind="pdf", parse_mode=mode,
            sections=sections_tuple,
            coverage=_coverage(
                sections_tuple, n_pages=pages, pages_with_text=text_pages, n_lines=lines),
            locator_kind="page_line",
            locator_limitations=(
                "PAGE/LINE은 1-based pypdf text-layer 순서이며 시각적 행 좌표가 아니다",
                "bbox와 좌표계는 제공되지 않아 표 셀의 시각 위치를 재현할 수 없다",
                "스캔 PDF는 OCR 전까지 text coverage가 0일 수 있다",
            ),
            warnings=warnings,
        )
    except Exception as exc:
        return SourceParseResult(
            source_path=path, source_kind="pdf", parse_mode="pdf_failed",
            sections=(), coverage=_coverage(()), locator_kind="page_line",
            locator_limitations=("PDF 파싱 실패로 page/line locator를 만들지 못함",),
            error=f"{type(exc).__name__}: {exc}"[:300],
        )


_CLAIM_NUMBER = re.compile(r"(?<![0-9A-Za-z])\(?-?\d[\d,]*(?:\.\d+)?\)?")
_CROSS_PART = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?|[A-Za-z가-힣]+")
# 우연한 boilerplate 몇 줄을 `partial_consistent`로 부르지 않는 붕괴 방지선.
# production 승인은 이 일반 하한과 별도로 full source hash + exact 기대 건수를 고정한다.
_MIN_PARTIAL_CLAIMS = 20
_MIN_PARTIAL_OVERLAP = 0.05


def _normal_number(text: str) -> str:
    raw = text.strip()
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace(",", "")
    return "-" + raw.lstrip("-") if negative else raw


def _canonical(text: str) -> str:
    """서식은 접되 숫자의 부호·소수점은 보존한다.

    단순 영숫자 join은 ``(100)``과 ``100``, ``1.23``과 ``12.3``을 같게 만들어
    위험한 거짓 matched를 낸다.
    """
    parts: list[str] = []
    for match in _CROSS_PART.finditer(text or ""):
        token = match.group(0)
        if token[0].isdigit() or token[0] in "(-":
            parts.append("N" + _normal_number(token))
        else:
            parts.append("T" + token.lower())
    return "\x1d".join(parts)


def _canonical_document(result: SourceParseResult) -> str:
    # path 경계와 section 순서는 의미가 있으므로 구분자를 유지한다.
    return "\x1e".join(
        f"{_canonical(s.path)}\x1f{_canonical(s.text)}" for s in result.sections)


def _claims(result: SourceParseResult) -> dict[str, tuple[str, ...]]:
    """유일한 ``section path + 숫자 제외 문장``만 비교 가능한 주장으로 만든다."""
    found: dict[str, list[tuple[str, ...]]] = {}
    for section in result.sections:
        path = _canonical(section.path)
        for raw in section.text.splitlines():
            numbers = tuple(_normal_number(m.group(0)) for m in _CLAIM_NUMBER.finditer(raw))
            if not numbers:
                continue
            anchor = _canonical(_CLAIM_NUMBER.sub(" # ", raw))
            if len(anchor) < 4:                 # 페이지 번호·기수만 있는 줄은 주장이 아니다
                continue
            found.setdefault(f"{path}|{anchor}", []).append(numbers)
    # 같은 문장 뼈대가 한 source 안에 반복되면 어떤 행끼리 대응하는지 불명확하다.
    return {key: values[0] for key, values in found.items() if len(values) == 1}


def _best(results: Iterable[SourceParseResult]) -> SourceParseResult | None:
    items = [r for r in results if r.usable]
    return max(items, key=lambda r: (r.coverage.n_tables, r.coverage.n_chars,
                                     r.coverage.n_sections), default=None)


def cross_check_sources(results: Iterable[SourceParseResult]) -> CrossCheckResult:
    """가장 충실한 HTML/PDF 한 쌍을 보수적으로 교차검증한다.

    전체 canonical 표현이 동일할 때만 ``matched``다. 충분한 exact claim 교집합이 전부
    일치하면 그 범위만 ``partial_consistent``로 표시하고, 낮은 overlap은 계속
    ``not_comparable``로 둔다. 같은 유일 문장 뼈대의 숫자가 다르면 명시적
    ``conflict``다.
    """
    items = tuple(results)
    html = _best(r for r in items if r.source_kind == "viewer_html")
    pdf = _best(r for r in items if r.source_kind == "pdf")
    usable = tuple(r for r in items if r.usable)
    if html is None or pdf is None:
        if usable:
            return CrossCheckResult(
                CrossCheckStatus.SINGLE_SOURCE,
                tuple(str(r.source_path) for r in usable),
                "HTML/PDF 중 한쪽만 구조화 가능한 내용을 제공함")
        return CrossCheckResult(
            CrossCheckStatus.NOT_COMPARABLE, (),
            "구조화 가능한 HTML/PDF source가 없음")

    compared = (str(html.source_path), str(pdf.source_path))
    html_doc, pdf_doc = _canonical_document(html), _canonical_document(pdf)
    html_digest = hashlib.sha256(html_doc.encode()).hexdigest()
    pdf_digest = hashlib.sha256(pdf_doc.encode()).hexdigest()
    # 너무 짧은 공통 boilerplate가 우연히 같아도 matched로 부르지 않는다.
    if len(html_doc) >= 20 and html_doc == pdf_doc:
        return CrossCheckResult(
            CrossCheckStatus.MATCHED, compared,
            "section path·순서·정규화 본문이 전체 일치함",
            primary_digest=html_digest, alternate_digest=pdf_digest)

    html_claims, pdf_claims = _claims(html), _claims(pdf)
    common_keys = sorted(html_claims.keys() & pdf_claims.keys())
    conflicts: list[CrossCheckConflict] = []
    matching = 0
    unresolved = 0
    for key in common_keys:
        left, right = html_claims[key], pdf_claims[key]
        if left == right:
            matching += 1
        elif len(left) == len(right):
            conflicts.append(CrossCheckConflict(key, left, right))
        else:
            # 같은 anchor라도 숫자 개수가 다르면 어느 토큰이 대응하는지 확정할 수 없다.
            unresolved += 1
    if conflicts:
        return CrossCheckResult(
            CrossCheckStatus.CONFLICT, compared,
            f"같은 유일 claim anchor의 숫자 불일치 {len(conflicts)}건",
            tuple(conflicts), html_digest, pdf_digest,
            len(common_keys), matching, unresolved)

    # 전체 문서 동일성은 아니지만 우연한 boilerplate 몇 줄보다 큰 exact claim 교집합이
    # 전부 일치하면 그 **비교된 범위만** typed 상태로 보존한다. unmatched 범위까지
    # `matched`라고 부르지 않으며, 낮은 overlap은 계속 not_comparable이다.
    comparable_denominator = min(len(html_claims), len(pdf_claims))
    overlap = (matching / comparable_denominator if comparable_denominator else 0.0)
    if (matching >= _MIN_PARTIAL_CLAIMS
            and overlap >= _MIN_PARTIAL_OVERLAP and unresolved == 0):
        return CrossCheckResult(
            CrossCheckStatus.PARTIAL_CONSISTENT, compared,
            f"고유 claim anchor {matching}건의 숫자가 모두 일치하고 명시 충돌 0건; "
            "대응하지 않은 범위는 미검증",
            (), html_digest, pdf_digest,
            len(common_keys), matching, unresolved)
    return CrossCheckResult(
        CrossCheckStatus.NOT_COMPARABLE, compared,
        f"두 source가 모두 있으나 전체 동일성을 증명할 수 없음 "
        f"(비교 anchor {len(common_keys)}건, 일치 {matching}건, "
        f"미해결 {unresolved}건)",
        primary_digest=html_digest, alternate_digest=pdf_digest,
        compared_claims=len(common_keys), matching_claims=matching,
        unresolved_claims=unresolved)


def _primary_rank(result: SourceParseResult) -> tuple[int, int, int, int]:
    # 구조화 HTML을 우선하되 shell/실패 HTML은 PDF보다 앞세우지 않는다.
    structure = (2 if result.source_kind == "viewer_html" and result.coverage.n_tables
                 else 1 if result.source_kind == "viewer_html" else 0)
    return (int(result.usable), structure, result.coverage.n_tables, result.coverage.n_chars)


def parse_pdf_html(files: list[Path], doc_name: str, corp_name: str) -> ParsedViewer:
    """모든 viewer HTML과 PDF를 **각각 끝까지** 파싱한 뒤 primary를 고른다.

    하위 호환 ``sections``는 primary만 보여 주지만 alternate를 버리지 않는다. HTML을 먼저
    성공했다고 PDF를 건너뛰거나, PDF 성공을 HTML 실패의 대체 결과로 덮어쓰지 않는다.
    """
    supported = sorted(
        (f for f in files if f.suffix.lower() in (".html", ".htm", ".pdf")),
        key=lambda p: p.name)
    results: list[SourceParseResult] = []
    for path in supported:
        if path.suffix.lower() in (".html", ".htm"):
            results.append(_parse_html_source(path))
        else:
            results.append(_parse_pdf_source(path))

    check = cross_check_sources(results)
    primary = max(results, key=_primary_rank, default=None)
    if primary is None or not primary.usable:
        return ParsedViewer(
            doc_name, corp_name, [], "unsupported", source_results=results,
            primary=primary, cross_check=check)
    return ParsedViewer(
        doc_name, corp_name, list(primary.sections), primary.parse_mode,
        source_results=results, primary=primary, cross_check=check)
