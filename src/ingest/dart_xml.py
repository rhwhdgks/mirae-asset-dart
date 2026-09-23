"""DART 문서 XML 파서 — 목차 단위 섹션 + 마크다운 표.

산출 형태는 주최측 설명회 레퍼런스(`참고자료.pdf` p31~33)를 따른다.

- 정기공시는 목차가 정의돼 있으므로 `TITLE` 단위로 청킹한다
- `path` 는 계층을 `>` 로 이은 문자열 (`"II. 사업의 내용 > 2. 주요 제품 및 서비스"`)
- 원문 표는 마크다운으로 변환하되 **단위 표기와 표 제목을 보존**한다
- 100자 미만 섹션은 '기재 생략'으로 보고 호출 측에서 배제할 수 있도록 `n_chars` 를 함께 준다

알려진 doubled-quote 속성 손상도 가역 sanitizer로 고쳐 strict 파싱한다. lxml 복구 파서는
향후 알 수 없는 손상의 최후 수단이며, 실패를 조용히 넘기지 않고 parse mode로 드러낸다.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree as lxml_etree

from .sanitize import sanitize
from .table_grid import SourceCell, build_grid
from typing import Any

__all__ = [
    "Section",
    "ParsedDocument",
    "ParseFailure",
    "UnsupportedFormat",
    "parse_xml",
    "parse_file",
]

_SECTION_DEPTH = {"SECTION-1": 1, "SECTION-2": 2, "SECTION-3": 3, "SECTION-4": 4}
_CELL_TAGS = ("TD", "TH", "TU", "TE")
_UNIT = re.compile(r"단\s*위\s*[:：]\s*([^)\]]{1,24})")

#: 개행을 포함한 모든 공백을 하나로 접는다.
#: 개행을 남기면 Markdown 표의 행이 중간에 끊긴다 — 셀의 0.29%뿐이지만 **문서의 71%**가
#: 최소 하나를 갖고 있어 표 구조가 깨진다. DART XML의 개행은 원본 서식의 줄바꿈이며
#: ("47기\n(2020년말)", "EVA, LDPE,\nECH, PVC") 의미상 공백에 해당한다.
_WS = re.compile(r"\s+")

#: 단위만 담은 문단·셀. 이 경우에만 다음 표의 단위로 넘긴다(one-shot binding).
#: 괄호로 감싼 짧은 표현. 안쪽에 중첩 괄호가 있어도 통째로 잡되, 여는 괄호와 닫는
#: 괄호의 종류가 반드시 같아야 한다. `[제76기] (단위: 백만원)`은 한 단위 wrapper가 아니다.
_BRACKETED = re.compile(
    r"^(?:\(\s*(?P<paren>.{1,80}?)\s*\)|\[\s*(?P<square>.{1,80}?)\s*\])$", re.S)
_HAS_UNIT = re.compile(r"단\s*위\s*[:：]")
#: 문장으로 보이면 단위 표기가 아니라 각주다
_SENTENCE = re.compile(r"[.。](\s|$)|입니다|합니다|참조|기준으로")


def unit_display(text: str) -> str | None:
    r"""셀·문단 텍스트가 **통째로** 단위 표기이면 표시용 문자열을 돌려준다.

    예전에는 `단위\s*[:：]([^\)\]]{1,24})` 로 **일부만 캡처**했다. 그래서
    `(단위 : 천CGRT(조선), 천TON(해양))` 이 `천CGRT(조선` 으로 잘렸고,
    그 상태로 표까지 삭제해 원문이 사라졌다(실측 단위 단칸표의 1.65%).

    이제 짝이 맞는 wrapper만 인정하고, 괄호 종류를 포함한 원문 표기를 그대로 쓴다.
    """
    display = text.strip()
    inner = _BRACKETED.match(display)
    if not inner:
        return None
    body = inner.group("paren") or inner.group("square")
    if not _HAS_UNIT.search(body) or _SENTENCE.search(body):
        return None
    return display


class ParseFailure(RuntimeError):
    """정제 후에도 strict 파싱이 실패한 경우. 조용히 넘기지 않는다."""


class UnsupportedFormat(ParseFailure):
    """DART 문서 XML이 아닌 입력.

    거래소공시 1,469건은 확장자가 `.xml` 이지만 실제 루트가 `<html>` 이다.
    이 파서로 처리하면 `<html>`·`<div>` 가 화이트리스트 밖이라 전부 escape되어
    텍스트만 남은 껍데기가 나온다. 별도 HTML 파서로 라우팅해야 하므로
    '파싱 실패'가 아니라 '대상 아님'으로 구분해 올린다.
    """


@dataclass
class Section:
    """목차 한 마디에 대응하는 청크."""

    path: str
    level: int
    order: int
    text: str
    n_tables: int
    #: 이 섹션을 여는 `TITLE` 의 **원문 element 위치**. `"SECTION-1[2]/TITLE[0]"`.
    #: 순번 기반 `SECTION[order]` 를 대체한다 — 파서가 앞에 섹션을 하나 더 잡아도
    #: 뒤쪽 섹션 ID 가 밀리지 않는다 (L-03 · X-01).
    source_locator: str = ""

    @property
    def n_chars(self) -> int:
        return len(self.text)

    @property
    def is_omitted(self) -> bool:
        """'기재 생략' 수준의 빈 섹션. 주최측 `list_sections` 는 이를 배제한다."""
        return self.n_chars < 100


@dataclass
class ParsedDocument:
    document_name: str
    company_name: str
    sections: list[Section] = field(default_factory=list)
    #: strict = 정제 후 표준 파서 통과 / recovered = lxml 복구 파서 사용.
    #: 복구본은 조용히 섞이면 안 되므로 산출물에 그대로 기록한다.
    parse_mode: str = "strict"

    @property
    def n_tables(self) -> int:
        return sum(s.n_tables for s in self.sections)


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return _WS.sub(" ", text.replace("\xa0", " ")).strip()


def _text_of(el: ET.Element) -> str:
    """인라인 마크업을 이어붙인다.

    `" ".join(itertext())` 을 쓰면 원문에 없던 공백이 들어간다. DART는 아래첨자를
    `SPAN` 으로 표현하므로(`tCO<SPAN>2</SPAN>-eq`) 공백 join은 `tCO 2 -eq` 를 만든다.
    DART 뷰어의 렌더링과 같게 하려면 구분자 없이 이어야 한다.
    """
    return _clean("".join(el.itertext()))


def _md_escape(text: str) -> str:
    """Markdown 표 셀에 그대로 넣을 수 없는 문자를 처리한다."""
    return text.replace("\\", "\\\\").replace("|", "\\|")


def _int_attr(el: ET.Element, name: str) -> int:
    try:
        return max(1, int(el.get(name, "1")))
    except ValueError:
        return 1


#: 격자는 `table_grid` 가 만든다. 예전에는 이 모듈이 자체 격자를 갖고 있었는데
#: `table.iter("TR")` 이 **중첩 표의 행까지 가져와** 상위 표가 하위 표를 삼켰다(K-09).
#: 실측 중첩 비율은 정기 0.247% · 주요사항 0.475% 로 낮지만 정기공시 전체로는 약 3천 표다.
GridCell = SourceCell


def _table_grid(table, locator: str = "TABLE[0]") -> tuple[list[list[SourceCell | None]], int]:
    """rowspan/colspan 을 펼친 격자와 머리글 행 수.

    `table_grid.build_grid` 결과를 예전 호출부가 쓰던 행렬 형태로 바꿔 준다.
    span 영역의 모든 좌표가 원점 셀을 참조하므로 `header_paths` 와 병합 셀 반복이 가능하다.
    실측상 표의 **13.5%가 머리글 colspan**, **5.6%가 본문 rowspan** 을 갖는다.
    """
    grid = build_grid(table, locator)
    matrix = [[grid.cell_at(r, c) for c in range(grid.n_cols)] for r in range(grid.n_rows)]
    return matrix, grid.n_head_rows


def header_paths(matrix: list[list[GridCell | None]], n_head: int) -> list[str]:
    """열마다 헤더 계층을 `>` 로 이은 경로. `Change > Delta` 처럼 상위가 살아 있다."""
    if not matrix or not matrix[0]:
        return []
    out = []
    for col in range(len(matrix[0])):
        parts: list[str] = []
        for row in range(min(n_head, len(matrix))):
            cell = matrix[row][col]
            if cell and cell.text and cell.text not in parts:
                parts.append(cell.text)
        out.append(" > ".join(parts))
    return out


def _unit_only(matrix: list[list[str]]) -> str | None:
    """DART는 `(단위 : 백만원)` 을 본표 바로 앞의 1칸짜리 표로 넣는다.

    이를 표로 렌더링하면 모든 재무표 앞에 의미 없는 표가 하나씩 붙는다.
    단위 문자열만 뽑아내고 표 자체는 버린다.
    """
    seen: dict[tuple[int, int], str] = {}
    for row in matrix:
        for cell in row:
            if cell and cell.text:
                seen[(cell.origin_row, cell.origin_col)] = cell.text
    if len(seen) != 1:
        return None
    # **셀 전체가 단위 표현일 때만** 표를 버린다. 부분 일치로 보면
    # `※ 상세 현황은 (단위 : 백만원) 기준…` 같은 각주표까지 삭제된다 (T-07).
    return unit_display(next(iter(seen.values())))


def _table_to_markdown(table: ET.Element, unit: str | None) -> str:
    matrix, n_head = _table_grid(table)
    if not matrix or not matrix[0]:
        return ""

    text_at = lambda cell: cell.text if cell else ""
    if n_head > 1:
        # 2단 이상 머리글 — span 전파 덕분에 상위 헤더가 자식 열에 남는다
        header = header_paths(matrix, n_head)
        body = matrix[n_head:]
    elif n_head == 1:
        header = [text_at(c) for c in matrix[0]]
        body = matrix[1:]
    else:
        header = [text_at(c) for c in matrix[0]]
        body = matrix[1:]

    if not any(header):
        header = [f"열{i + 1}" for i in range(len(matrix[0]))]
        body = matrix

    lines = []
    if unit:
        lines.append(unit)
    lines.append("| " + " | ".join(_md_escape(h) for h in header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in body:
        values = [text_at(c) for c in row]
        if any(values):
            # 병합 셀의 값을 반복한다. `| 유럽 | 123 |` 만 남으면 상위 `수출` 이 사라져
            # LLM 도 Fact 추출기도 차원을 복원할 수 없다.
            lines.append("| " + " | ".join(_md_escape(v) for v in values) + " |")
    return "\n".join(lines)


class _Walker:
    def __init__(self) -> None:
        self.stack: list[str] = []
        #: 현재 열려 있는 섹션의 원문 위치
        self.locator = ""
        self.sections: list[Section] = []
        self.buffer: list[str] = []
        self.tables = 0
        self.order = 0
        self.pending_unit: str | None = None

    def _flush(self) -> None:
        if not self.stack:
            self.buffer.clear()
            self.tables = 0
            return
        # 블록 사이는 빈 줄로 나눈다. 한 줄로 붙이면 Markdown이 인접한 두 표를 하나로
        # 합쳐 버린다 — DART는 각주(`※ …`)를 본표 뒤 별도 1칸 표로 넣기 때문에
        # F-01 기준 표의 38.9%가 앞 표에 흡수됐다.
        text = "\n\n".join(p for p in (b.strip() for b in self.buffer) if p)
        self.sections.append(
            Section(
                path=" > ".join(self.stack),
                level=len(self.stack),
                order=self.order,
                text=text,
                n_tables=self.tables,
                source_locator=self.locator,
            )
        )
        self.order += 1
        self.buffer.clear()
        self.tables = 0
        self.pending_unit = None  # 단위는 섹션 경계를 넘지 않는다

    def _take_unit(self) -> str | None:
        """대기 중인 단위를 **한 번만** 소비한다.

        이전 구현은 최근 buffer 6개를 역스캔했다. 그 결과 이미 소비된
        `(단위: 억원)` 이 각주표·판매경로표·대금회수조건표처럼 금액이 아닌 표에까지
        붙어, Fact 추출기가 비율·텍스트를 억원으로 해석할 위험이 있었다.
        """
        unit, self.pending_unit = self.pending_unit, None
        return unit

    def walk(self, el, depth: int, path: str = "") -> None:
        # 같은 태그 형제 중 0-기반 순번을 이어 원문 위치를 만든다.
        # 파서 동작이 아니라 **원문 구조**에 매인 값이라야 ID 가 안정적이다 (L-03).
        counter: dict[str, int] = {}
        for child in el:
            tag = child.tag
            if not isinstance(tag, str):
                continue  # lxml 복구 트리의 주석·PI 노드
            idx = counter.get(tag, 0)
            counter[tag] = idx + 1
            here = f"{path}/{tag}[{idx}]" if path else f"{tag}[{idx}]"
            if tag in _SECTION_DEPTH:
                self.walk(child, _SECTION_DEPTH[tag], here)
            elif tag in ("TITLE", "COVER-TITLE"):
                self._flush()
                title = _text_of(child)
                # **빈 TITLE 도 섹션을 연다.** 실측 문서당 8개까지 있고, 이때 제목(stack)은
                # 그대로지만 내용은 갈라진다. locator 를 갱신하지 않으면 여러 섹션이 같은
                # 위치를 갖고 `block_id` 가 충돌한다(실측 섹션의 4.3%).
                self.locator = here
                if title:
                    self.stack = self.stack[: max(0, depth - 1)] + [title]
            elif tag == "TABLE":
                matrix, _ = _table_grid(child)
                unit_only = _unit_only(matrix)
                if unit_only is not None:
                    self.pending_unit = unit_only  # 본표에 붙일 단위. 표로는 내보내지 않는다
                    continue
                # 상위 표가 중첩 표의 행을 더 이상 삼키지 않으므로(K-09) 중첩 표는
                # 직접 내보내야 한다. 안 그러면 내용이 통째로 사라진다.
                for table in [child] + [t for t in child.iter("TABLE") if t is not child]:
                    markdown = _table_to_markdown(table, self._take_unit())
                    if markdown:
                        self.buffer.append(markdown)
                        self.tables += 1
            elif tag in ("P", "SPAN", "IMG-CAPTION"):
                text = _text_of(child)
                if not text:
                    continue
                unit = unit_display(text)
                if unit:
                    # 단위만 있는 문단도 다음 표에 one-shot 으로 붙인다
                    self.pending_unit = unit
                    continue
                # 단위는 **바로 다음 표**에만 붙는다. 사이에 다른 내용이 끼면 버린다.
                # 안 그러면 `(단위: 억원)` 이 문단 건너 판매경로표까지 오염시킨다 (X-09).
                self.pending_unit = None
                self.buffer.append(text)
            else:
                self.walk(child, depth, here)


def parse_xml(source: str) -> ParsedDocument:
    """정제 → strict 파싱 → 섹션 추출."""
    head = source[:4096].lstrip().lower()
    if head.startswith("<!doctype html") or "<html" in head[:512]:
        raise UnsupportedFormat("DART 문서 XML이 아니라 HTML입니다 — 거래소공시 파서로 라우팅하세요")

    return parse_root(*parse_tree(source))


def parse_root(root: Any, mode: str) -> ParsedDocument:
    """**이미 파싱된 트리**에서 섹션을 뽑는다.

    같은 문서를 섹션용·재무 Fact 용으로 두 번 파싱하던 것을 막기 위해 분리했다.
    실측 정기공시 1건당 읽기 0.006s + 파싱 0.178s 였고, 1,051건이면 **약 3분**이다.
    파싱 결과는 불변이므로 공유해도 안전하다.
    """
    if root.tag != "DOCUMENT":
        raise UnsupportedFormat(f"루트가 DOCUMENT가 아닙니다: <{root.tag}>")

    walker = _Walker()
    walker.walk(root, 1)
    walker._flush()

    return ParsedDocument(
        document_name=_clean(root.findtext("DOCUMENT-NAME")),
        company_name=_clean(root.findtext("COMPANY-NAME")),
        sections=walker.sections,
        parse_mode=mode,
    )


def parse_tree(source: str) -> tuple[Any, str]:
    """정제 → 표준 파서 → (실패 시) lxml 복구 파서. `(root, parse_mode)` 를 준다.

    known doubled-quote 손상은 sanitizer가 strict XML로 복구한다. 이 폴백은 향후 코퍼스의
    아직 분류되지 않은 손상을 위한 최후 수단이며, `recovered` mode를 산출물에 드러낸다.
    """
    cleaned = sanitize(source)
    try:
        return ET.fromstring(cleaned), "strict"
    except ET.ParseError as strict_error:
        try:
            root = lxml_etree.fromstring(
                cleaned.encode("utf-8"),
                lxml_etree.XMLParser(recover=True, huge_tree=True, encoding="utf-8"),
            )
        except Exception as exc:  # noqa: BLE001
            raise ParseFailure(f"정제·복구 모두 실패: {strict_error} / {exc}") from exc
        if root is None:
            raise ParseFailure(f"복구 파서가 트리를 만들지 못했습니다: {strict_error}")
        return root, "recovered"


def parse_file(path: str | Path) -> ParsedDocument:
    return parse_xml(Path(path).read_text(encoding="utf-8", errors="replace"))
