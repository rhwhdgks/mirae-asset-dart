"""정정 항목 추출 — 「무엇이 어떻게 바뀌었는가」.

`Relation` 간선은 「A 가 B 를 정정했다」만 담는다. 정작 사용자가 묻는
**「뭐가 바뀌었어?」** 에는 답할 수 없다. 원문 `<CORRECTION>` 블록에는 이미 항목별
정정 전/후 표가 있으므로 그것을 뽑는다.

```text
항 목            | 정정사유                    | 정 정 전        | 정 정 후
3. 계약상대       | 영업비밀 보호 요청 중 동의    | 글로벌 대형기업  | 테슬라(Tesla, Inc.)
```

## 포맷별 차이

| 문서군 | 위치 | 값 셀 |
|---|---|---|
| exchange | HTML `정정항목/정정전/정정후` 3열 | `span.xforms_input` |
| major·holding·periodic | DART XML `<CORRECTION>` 안의 표 | **`TD`** — 폼과 달리 `TE` 가 아니다 |

DART XML 은 열 구성이 문서군마다 다르다(`정정요구ㆍ명령관련 여부` 유무).
따라서 **열 위치를 고정하지 않고 헤더 이름으로 찾는다.**

이 산출물이 있어야 검수가 요구한 *declared diff(선언된 정정)* 와
*computed diff(실제 값 비교)* 대조가 가능해진다.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from lxml import html as lxml_html

from .corpus_paths import CorpusIndex, iter_manifest
from .dart_xml import _table_grid, _table_to_markdown
from .exchange_html import cell_text as exchange_cell_text
from .exchange_html import parse_file as parse_exchange
from .sanitize import sanitize
from .table_grid import build_grid
from ..artifact import file_sha256, write_stamp

__all__ = ["CorrectionItem", "extract", "build"]

_SQ = lambda s: "".join((s or "").split())

#: 원문이 값 대신 넣는 대체 표기. `(주1)` 은 「아래 각주 참조」,
#: `(표현 양식 변형)` 은 「값이 아니라 서식이 바뀜」이라는 뜻이다.
#: 전수 3,428건 중 208건(6.1%)이 여기 해당하며, 이를 「변경 없음」으로 처리하면
#: 각주가 가리키는 실제 정정 내용을 놓친다.
_PLACEHOLDER = re.compile(
    r"^\(?\s*(주\s*\d+|\*\s*\d+|표현\s*양식\s*변형|별첨|첨부|상기|하기|생략|이하\s*동일|-)\s*\)?$"
)
_HDR = {
    "item": ("항목", "정정항목"),
    "reason": ("정정사유",),
    "before": ("정정전",),
    "after": ("정정후",),
    "required": ("정정요구ㆍ명령관련여부", "정정요구·명령관련여부"),
}


@dataclass(frozen=True)
class CorrectionItem:
    doc_id: str
    rcept_no: str
    corp_name: str
    doc_group: str
    seq: int
    item_path: str               #: `"3. 계약상대"` — 무엇을 고쳤나
    reason: str | None           #: 정정사유
    value_before: str | None
    value_after: str | None
    required_by_authority: bool  #: 정정요구·명령에 따른 것인가
    #: **두 축을 분리한다** (S-10). 예전 `value_kind` 하나는 「값이 바뀌었나」와
    #: 「값이 무엇인가」를 섞어서, `(주1) → 100` 을 `changed` 로만 적고 정정 전이
    #: 대체표기였다는 사실을 잃었다.
    diff_kind: str               #: same | changed
    before_kind: str             #: literal | placeholder | empty
    after_kind: str              #: literal | placeholder | empty
    locator: str                 #: 행/block 대표 위치 — 기존 block_id 계약을 유지한다
    #: 정정 전·후 값이 실제로 적힌 **각각의 셀**. Evidence는 대표 locator가 아니라
    #: 해당 side locator를 써야 excerpt와 원문 위치가 일치한다.
    before_locator: str | None
    after_locator: str | None
    #: locator 문자열의 TR/TD 순번은 물리 자식 순번이다. THEAD/rowspan을 펼친
    #: semantic 격자 좌표는 파서가 계산한 값을 별도로 보존한다.
    logical_row: int | None
    logical_col: int | None
    before_logical_row: int | None
    before_logical_col: int | None
    after_logical_row: int | None
    after_logical_col: int | None


@dataclass(frozen=True)
class _AlignedNestedChild:
    """One provably aligned scalar row inside a parent correction cell.

    Some correction forms put a *parent* field in the correction inventory
    (for example ``4. 자금조달의 목적``), then put the actual N changed child
    values in a before/after nested table.  Treating the two nested tables as
    one markdown blob loses each child field and makes its value evidence point
    at the parent cell.  This private shape is deliberately narrower: it is
    emitted only after both sides prove a one-to-one two-cell row alignment.
    """

    path: str
    value_before: str
    value_after: str
    locator: str
    before_locator: str
    after_locator: str
    logical_row: int
    logical_col: int
    before_logical_row: int
    before_logical_col: int
    after_logical_row: int
    after_logical_col: int


def value_kind(value: str | None) -> str:
    """값 하나의 성격 — `literal` · `placeholder` · `empty`.

    `placeholder` 는 「값이 없음」이 아니라 **「표에 값을 적지 않고 다른 곳을 가리킴」**이다
    (`(주1)`, `(표현 양식 변형)`, `별첨`). 실제 내용은 각주에 따로 있다.
    """
    v = (value or "").strip()
    if not v:
        return "empty"
    return "placeholder" if _PLACEHOLDER.match(v) else "literal"


def classify_value(before: str | None, after: str | None) -> tuple[str, str, str]:
    """`(diff_kind, before_kind, after_kind)` — 서로 다른 두 축이다 (S-10).

    예전에는 하나의 `value_kind` 로 뭉쳐 있었다. 그래서
    `(주1) → 100` 은 `changed` 로만 남아 **정정 전이 대체표기였다는 사실이 사라졌고**,
    `100 → 100` 은 `formatting` 이 되어 「값은 그대로」인지 「서식만 바뀜」인지 모호했다.

    | 예 | diff | before → after |
    |---|---|---|
    | `100 → 100` | same | literal → literal |
    | `(주1) → 100` | changed | placeholder → literal |
    | `(주1) → (주1)` | same | placeholder → placeholder |
    """
    b, a = (before or "").strip(), (after or "").strip()
    return ("same" if b == a else "changed", value_kind(b), value_kind(a))


def _row_anchor(row, cols: dict[str, int], table_loc: str,
                ri: int) -> tuple[str, int | None, int | None]:
    """정정 행의 대표 원문 위치와 펼친 격자 원점 좌표.

    **그 행에서 시작하는 셀**을 골라야 한다. 항목 셀이 `rowspan` 으로 여러 행을 덮으면
    행마다 같은 locator 가 나와 `block_id` 가 충돌한다(실측 239건).

    모든 셀이 앞 행의 rowspan에서 상속됐으면 합성 ROW locator의 좌표를 추정하지 않는다.
    """
    for role in ("before", "after", "item"):
        idx = cols.get(role)
        cell = row[idx] if idx is not None and idx < len(row) else None
        if cell is not None and cell.origin_row == ri:
            return cell.locator, cell.origin_row, cell.origin_col
    return f"{table_loc}/ROW[{ri}]", None, None


def _role_cell(row, cols: dict[str, int], role: str):
    """헤더 role에 해당하는 실제 격자 셀. rowspan이면 원래 origin 셀을 돌려준다."""
    idx = cols.get(role)
    return row[idx] if idx is not None and idx < len(row) else None


def _role_locator(row, cols: dict[str, int], role: str) -> str | None:
    cell = _role_cell(row, cols, role)
    return cell.locator if cell is not None else None


def _role_position(row, cols: dict[str, int],
                   role: str) -> tuple[int | None, int | None]:
    """role 셀의 semantic 원점. locator의 물리 TD 순번을 좌표로 쓰지 않는다."""
    cell = _role_cell(row, cols, role)
    return ((cell.origin_row, cell.origin_col) if cell is not None
            else (None, None))


def _cell_value(cell) -> str | None:
    """정정 표 셀의 값. **중첩 표 안의 내용까지 가져온다.**

    격자 엔진은 중첩 표를 상위 셀 텍스트에서 제외한다(K-09) — 데이터 표에서는 그게 맞다.
    그런데 정정 표에서는 **중첩 표 자체가 정정 전/후 값**인 경우가 있다.

    ```text
    | 6. 양수예정일자 | 사유 | [표: 양수기준일 2025-03-15] | [표: 양수기준일 2025-03-31] |
    ```

    제외한 채로 두면 셀이 비어 항목이 통째로 버려지거나(실측 787건), 표 제목만 남고
    **정작 바뀐 내용이 사라진다**(`"나. 직원 등 현황"` 만 남고 현황표는 유실).

    그래서 자기 텍스트에 **중첩 표를 Markdown 으로 이어 붙인다.** 정정 전/후는 값 하나가
    아니라 「그 자리에 있던 내용 전체」이므로 표 구조를 보존해야 대조할 수 있다.
    """
    if cell is None:
        return None
    node = cell.node
    if node is None:
        return cell.text or None
    parts = [cell.text] if cell.text else []
    for nested in node.iter("TABLE"):
        markdown = _table_to_markdown(nested, None)
        if markdown:
            parts.append(markdown)
    return "\n\n".join(parts) or None


def _element_name(node) -> str:
    """Local upper-case element name for ElementTree and lxml nodes."""

    tag = getattr(node, "tag", "")
    return str(tag).rsplit("}", 1)[-1].upper() if isinstance(tag, str) else ""


def _nested_tables(cell) -> list[tuple[object, str]]:
    """Return nested tables with their physical locators below ``cell``.

    ``build_grid`` correctly stops an outer grid at a nested table, but it
    cannot infer the nested table's parent locator.  Recreate the same
    child-index path convention here so a child value's Evidence continues to
    point to its actual TD, not to the enclosing correction cell.
    """

    node = getattr(cell, "node", None)
    if node is None:
        return []
    tables: list[tuple[object, str]] = []

    def walk(parent, parent_locator: str) -> None:
        counts: dict[str, int] = {}
        for child in parent:
            name = _element_name(child)
            if not name:
                continue
            index = counts.get(name, 0)
            counts[name] = index + 1
            locator = f"{parent_locator}/{name}[{index}]"
            if name == "TABLE":
                tables.append((child, locator))
            else:
                walk(child, locator)

    walk(node, cell.locator)
    return tables


_NESTED_HEADER_WORDS = {
    "구분", "항목", "내용", "정정항목", "정정전", "정정후", "금액", "값", "일자", "비고",
}


def _simple_nested_rows(table, locator: str, *, text_of=None):
    """Read only a simple label/value child table, otherwise return ``None``.

    A correction table can contain any arbitrary embedded report table.  We
    never guess which column is the changed scalar in those tables.  The only
    safe split shape is N>=2 data rows, each with exactly two *new* scalar
    cells (label, value), no rowspan/colspan inheritance, and no further
    nested table.
    """

    grid = build_grid(table, locator, text_of=text_of)
    rows = []
    for ri in range(grid.n_head_rows, grid.n_rows):
        seen: set[str] = set()
        cells = []
        for ci in range(grid.n_cols):
            cell = grid.cell_at(ri, ci)
            if cell is not None and cell.locator not in seen:
                seen.add(cell.locator)
                cells.append(cell)
        if (len(cells) != 2
                or any(cell.origin_row != ri or cell.colspan != 1
                       or cell.rowspan != 1 for cell in cells)
                or any(_nested_tables(cell) for cell in cells)):
            return None
        label, value = ((cell.text or "").strip() for cell in cells)
        if (not label or not value or _SQ(label) in _NESTED_HEADER_WORDS
                or _SQ(value) in _NESTED_HEADER_WORDS):
            return None
        rows.append((label, value, cells[1]))
    return rows if len(rows) >= 2 else None


def _aligned_nested_child_items(parent_path: str, before_cell, after_cell,
                                *, text_of=None) -> list[_AlignedNestedChild] | None:
    """Split one parent correction row only when its child rows align exactly.

    Both sides must contain exactly one nested table and otherwise no direct
    text.  Child labels must have the same count, order, and whitespace-only
    normalized spelling.  Any ambiguous, complex, or changed table shape is
    retained as the original unsplit correction item (fail closed).
    """

    parent = (parent_path or "").strip()
    if not parent or before_cell is None or after_cell is None:
        return None
    # Direct prose plus a table is one composite correction, not a proven
    # scalar child inventory.  Do not discard that prose by splitting it.
    if (before_cell.text or "").strip() or (after_cell.text or "").strip():
        return None
    before_tables = _nested_tables(before_cell)
    after_tables = _nested_tables(after_cell)
    if len(before_tables) != 1 or len(after_tables) != 1:
        return None
    before_rows = _simple_nested_rows(*before_tables[0], text_of=text_of)
    after_rows = _simple_nested_rows(*after_tables[0], text_of=text_of)
    if (before_rows is None or after_rows is None
            or len(before_rows) != len(after_rows)
            or len({_SQ(row[0]) for row in before_rows}) != len(before_rows)
            or any(_SQ(left[0]) != _SQ(right[0])
                   for left, right in zip(before_rows, after_rows))):
        return None
    return [
        _AlignedNestedChild(
            path=f"{parent} > {before_label}",
            value_before=before_value,
            value_after=after_value,
            locator=before_value_cell.locator,
            before_locator=before_value_cell.locator,
            after_locator=after_value_cell.locator,
            logical_row=before_value_cell.origin_row,
            logical_col=before_value_cell.origin_col,
            before_logical_row=before_value_cell.origin_row,
            before_logical_col=before_value_cell.origin_col,
            after_logical_row=after_value_cell.origin_row,
            after_logical_col=after_value_cell.origin_col,
        )
        for ((before_label, before_value, before_value_cell),
             (_after_label, after_value, after_value_cell))
        in zip(before_rows, after_rows)
    ]


def _br_separated_lines(cell) -> list[tuple[str, str]] | None:
    """Return direct ``BR``-separated cell lines without guessing boundaries.

    Exchange correction forms sometimes encode a small child inventory as
    three visually aligned line lists instead of a nested table::

        item    2. 계약내역 / - 계약금액(원) / - 매출액대비(%)
        before                 / -              / -
        after                  / 72,200,000,000 / 23.13

    ``cell.text`` intentionally normalises those lines into one string, so it
    cannot prove which value belongs to which child.  Read the source DOM only
    when an actual ``BR`` element provides that boundary.  Nested tables stay
    on the separate, stricter table path above.
    """

    node = getattr(cell, "node", None)
    if node is None or _nested_tables(cell):
        return None
    if not any(_element_name(child) == "BR" for child in node.iter()):
        return None

    # Each line carries a real source anchor.  The first line belongs to the
    # cell itself; a following line is anchored by the actual BR that starts
    # it.  This makes child CorrectionItem block IDs unique without inventing
    # a pseudo ``LINE[n]`` locator that the source verifier cannot resolve.
    raw_lines: list[tuple[list[str], str]] = [([], cell.locator)]

    def walk(element, locator: str) -> None:
        if getattr(element, "text", None):
            raw_lines[-1][0].append(str(element.text))
        counts: dict[str, int] = {}
        for child in element:
            name = _element_name(child)
            if not name:
                continue
            index = counts.get(name, 0)
            counts[name] = index + 1
            child_locator = f"{locator}/{name}[{index}]"
            if name == "BR":
                raw_lines.append(([], child_locator))
            else:
                walk(child, child_locator)
            if getattr(child, "tail", None):
                raw_lines[-1][0].append(str(child.tail))

    walk(node, cell.locator)
    lines = [
        (re.sub(r"\s+", " ", "".join(parts)).strip(), locator)
        for parts, locator in raw_lines
    ]
    return [(line, locator) for line, locator in lines if line]


def _aligned_br_child_items(parent_cell, before_cell, after_cell, *, row_index: int,
                            ) -> list[_AlignedNestedChild] | None:
    """Split a source-proven aligned line-list correction, or fail closed.

    The item side must be ``parent + N bullet labels`` and both value sides
    must contain exactly N non-empty source lines.  Count/order therefore come
    from the disclosure itself.  A prose row, missing line, duplicate label,
    nested table, or unmarked child label remains one composite item.

    A line has no independent HTML element, so each scalar excerpt retains its
    containing TD locator and semantic coordinates.  The scalar excerpt and
    child path still produce distinct correction records; no synthetic source
    locator is invented.
    """

    if parent_cell is None or before_cell is None or after_cell is None:
        return None
    # An inherited rowspan cell belongs to a different physical row.  Splitting
    # it again here would reuse the same child anchors under another logical
    # correction row.  Likewise, a spanning cell does not prove one scalar
    # list per role, so keep the original composite row.
    if any(cell.origin_row != row_index or cell.rowspan != 1 or cell.colspan != 1
           for cell in (parent_cell, before_cell, after_cell)):
        return None
    item_lines = _br_separated_lines(parent_cell)
    before_lines = _br_separated_lines(before_cell)
    after_lines = _br_separated_lines(after_cell)
    if item_lines is None or before_lines is None or after_lines is None:
        return None
    if len(item_lines) < 3:
        return None

    parent = item_lines[0][0]
    if re.match(r"[-–—·ㆍ]", parent):
        return None
    labels: list[tuple[str, str]] = []
    for line, line_locator in item_lines[1:]:
        match = re.fullmatch(r"[-–—·ㆍ]\s*(.+)", line)
        if match is None or not match.group(1).strip():
            return None
        labels.append((match.group(1).strip(), line_locator))
    if (not parent or len(before_lines) != len(labels)
            or len(after_lines) != len(labels)
            or len({_SQ(label) for label, _locator in labels}) != len(labels)):
        return None

    return [
        _AlignedNestedChild(
            path=f"{parent} > {label}",
            value_before=before_value,
            value_after=after_value,
            locator=label_locator,
            before_locator=before_cell.locator,
            after_locator=after_cell.locator,
            logical_row=parent_cell.origin_row,
            logical_col=parent_cell.origin_col,
            before_logical_row=before_cell.origin_row,
            before_logical_col=before_cell.origin_col,
            after_logical_row=after_cell.origin_row,
            after_logical_col=after_cell.origin_col,
        )
        for ((label, label_locator), (before_value, _before_line_locator),
             (after_value, _after_line_locator))
        in zip(labels, before_lines, after_lines)
    ]


def _kinds(before: str | None, after: str | None) -> dict[str, str]:
    diff, bk, ak = classify_value(before, after)
    return {"diff_kind": diff, "before_kind": bk, "after_kind": ak}


def _column_map(header: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, name in enumerate(header):
        key = _SQ(name)
        for role, words in _HDR.items():
            if role in out:
                continue
            if any(w in key for w in words):
                out[role] = idx
    return out


def _from_dart_xml(root, doc: dict) -> list[CorrectionItem]:
    items: list[CorrectionItem] = []
    for node in root.iter("CORRECTION"):
        for ti, table in enumerate(node.iter("TABLE")):
            # 표 위치를 실제 경로로 준다. 셀 locator 가 여기에 이어 붙는다 (L-05).
            table_loc = f"CORRECTION[0]/TABLE[{ti}]"
            matrix, n_head = _table_grid(table, table_loc)
            if not matrix or n_head < 1:
                continue
            header = [(c.text if c else "") for c in matrix[0]]
            cols = _column_map(header)
            if "before" not in cols or "after" not in cols:
                continue
            for ri, row in enumerate(matrix[n_head:], start=n_head):
                cell = lambda role: _cell_value(_role_cell(row, cols, role))
                before, after = cell("before"), cell("after")
                if not before and not after:
                    continue
                before_cell = _role_cell(row, cols, "before")
                after_cell = _role_cell(row, cols, "after")
                item_path = cell("item") or ""
                aligned_children = _aligned_nested_child_items(
                    item_path, before_cell, after_cell)
                if aligned_children is not None:
                    for child in aligned_children:
                        items.append(CorrectionItem(
                            doc_id=doc["doc_id"], rcept_no=doc["rcept_no"],
                            corp_name=doc["corp_name"], doc_group=doc["doc_group"],
                            seq=len(items), item_path=child.path, reason=cell("reason"),
                            value_before=child.value_before, value_after=child.value_after,
                            required_by_authority=(cell("required") or "").strip() not in ("", "아니오", "-"),
                            **_kinds(child.value_before, child.value_after),
                            locator=child.locator,
                            before_locator=child.before_locator,
                            after_locator=child.after_locator,
                            logical_row=child.logical_row, logical_col=child.logical_col,
                            before_logical_row=child.before_logical_row,
                            before_logical_col=child.before_logical_col,
                            after_logical_row=child.after_logical_row,
                            after_logical_col=child.after_logical_col,
                        ))
                    continue
                locator, logical_row, logical_col = _row_anchor(
                    row, cols, table_loc, ri)
                before_row, before_col = _role_position(row, cols, "before")
                after_row, after_col = _role_position(row, cols, "after")
                items.append(CorrectionItem(
                    doc_id=doc["doc_id"], rcept_no=doc["rcept_no"],
                    corp_name=doc["corp_name"], doc_group=doc["doc_group"],
                    seq=len(items), item_path=item_path, reason=cell("reason"),
                    value_before=before, value_after=after,
                    required_by_authority=(cell("required") or "").strip() not in ("", "아니오", "-"),
                    **_kinds(before, after),
                    # 항목 셀의 **원문 위치**. 예전 `CORRECTION/TABLE[n]/TR[m]` 은
                    # 격자 행 번호라 원문 element 를 가리키지 않았다 (L-05).
                    locator=locator,
                    before_locator=_role_locator(row, cols, "before"),
                    after_locator=_role_locator(row, cols, "after"),
                    logical_row=logical_row, logical_col=logical_col,
                    before_logical_row=before_row, before_logical_col=before_col,
                    after_logical_row=after_row, after_logical_col=after_col,
                ))
        break  # CORRECTION 블록은 문서당 하나
    return items


def _from_exchange(form, doc: dict, *, source_html: str | None = None) -> list[CorrectionItem]:
    """`정정항목 | 정정전 | 정정후` 3열이 한 행씩 반복된다.

    예전에는 `form.fields` dict 를 읽어 `" [2]"` 접미로 반복을 구분했다. 그러면
    **원문 위치를 잃는다** — dict 키는 라벨 경로일 뿐이다. `form.records` 는 값마다
    실제 `source_locator` 를 들고 있으므로 그것을 쓴다 (L-05).
    """
    reason = form.get("정정사유")

    # ExchangeForm.records is intentionally a label/value projection.  It is
    # sufficient for ordinary form fields, but a correction table can have one
    # header followed by many value-only rows.  In that shape only the first
    # row inherits the header labels and the remaining before/after triples are
    # absent from ``records``.  When source HTML is available, reconstruct the
    # declared correction rows from the semantic table grid instead.  The
    # trigger is the source header roles, never an issuer, receipt, or field
    # name, so the rule generalises to every exchange correction table.
    if source_html is not None:
        root = lxml_html.fromstring(source_html)
        items: list[CorrectionItem] = []
        for ti, table in enumerate(root.iter("table")):
            # A nested table is content of its parent before/after cell, not a
            # second correction inventory.
            if table.xpath("ancestor::table"):
                continue
            table_loc = f"TABLE[{ti}]"
            grid = build_grid(table, table_loc, text_of=exchange_cell_text)
            header_row = None
            cols: dict[str, int] = {}
            for ri in range(grid.n_rows):
                header = [
                    (grid.cell_at(ri, ci).text if grid.cell_at(ri, ci) else "")
                    for ci in range(grid.n_cols)
                ]
                candidate = _column_map(header)
                if {"item", "before", "after"} <= set(candidate):
                    header_row, cols = ri, candidate
                    break
            if header_row is None:
                continue
            for ri in range(header_row + 1, grid.n_rows):
                row = [grid.cell_at(ri, ci) for ci in range(grid.n_cols)]
                cell = lambda role: _cell_value(_role_cell(row, cols, role))
                before, after = cell("before"), cell("after")
                if not before and not after:
                    continue
                item_path = cell("item") or ""
                # A repeated header marks the start of another block rather
                # than a data row.
                if _SQ(item_path) in _HDR["item"]:
                    continue
                aligned_children = _aligned_nested_child_items(
                    item_path, _role_cell(row, cols, "before"),
                    _role_cell(row, cols, "after"), text_of=exchange_cell_text)
                if aligned_children is None:
                    aligned_children = _aligned_br_child_items(
                        _role_cell(row, cols, "item"),
                        _role_cell(row, cols, "before"),
                        _role_cell(row, cols, "after"),
                        row_index=ri,
                    )
                if aligned_children is not None:
                    for child in aligned_children:
                        items.append(CorrectionItem(
                            doc_id=doc["doc_id"], rcept_no=doc["rcept_no"],
                            corp_name=doc["corp_name"], doc_group=doc["doc_group"],
                            seq=len(items), item_path=child.path,
                            reason=cell("reason") or reason,
                            value_before=child.value_before,
                            value_after=child.value_after,
                            required_by_authority=False,
                            **_kinds(child.value_before, child.value_after),
                            locator=child.locator,
                            before_locator=child.before_locator,
                            after_locator=child.after_locator,
                            logical_row=child.logical_row,
                            logical_col=child.logical_col,
                            before_logical_row=child.before_logical_row,
                            before_logical_col=child.before_logical_col,
                            after_logical_row=child.after_logical_row,
                            after_logical_col=child.after_logical_col,
                        ))
                    continue
                locator, logical_row, logical_col = _row_anchor(
                    row, cols, table_loc, ri)
                before_row, before_col = _role_position(row, cols, "before")
                after_row, after_col = _role_position(row, cols, "after")
                items.append(CorrectionItem(
                    doc_id=doc["doc_id"], rcept_no=doc["rcept_no"],
                    corp_name=doc["corp_name"], doc_group=doc["doc_group"],
                    seq=len(items), item_path=item_path,
                    reason=cell("reason") or reason,
                    value_before=before, value_after=after,
                    required_by_authority=False,
                    **_kinds(before, after),
                    locator=locator,
                    before_locator=_role_locator(row, cols, "before"),
                    after_locator=_role_locator(row, cols, "after"),
                    logical_row=logical_row, logical_col=logical_col,
                    before_logical_row=before_row,
                    before_logical_col=before_col,
                    after_logical_row=after_row,
                    after_logical_col=after_col,
                ))
        return items

    groups: dict[str, list] = {}
    for rec in form.records:
        key = _SQ(rec.label_path)
        if "정정전" not in key or "정정후" not in key:
            continue
        groups.setdefault(rec.label_path, []).append(rec)

    items: list[CorrectionItem] = []
    for recs in groups.values():
        # 3개 묶음이 반복된다: 1=항목 2=정정전 3=정정후
        for i in range(0, len(recs) - 2, 3):
            item, before, after = recs[i], recs[i + 1], recs[i + 2]
            items.append(CorrectionItem(
                doc_id=doc["doc_id"], rcept_no=doc["rcept_no"],
                corp_name=doc["corp_name"], doc_group=doc["doc_group"],
                seq=len(items), item_path=item.value, reason=reason,
                value_before=before.value, value_after=after.value,
                required_by_authority=False,
                **_kinds(before.value, after.value),
                locator=item.source_locator,
                before_locator=before.source_locator,
                after_locator=after.source_locator,
                logical_row=item.logical_row, logical_col=item.logical_col,
                before_logical_row=before.logical_row,
                before_logical_col=before.logical_col,
                after_logical_row=after.logical_row,
                after_logical_col=after.logical_col,
            ))
    return items


class NotStructurable(RuntimeError):
    """구조화된 정정 항목을 만들 수 없는 문서. **결함이 아니라 원문의 한계다.**

    `pdf+html` 문서 2건(KB금융·한화오션 기재정정)이 여기 해당한다. PDF 텍스트에는
    행·열이 없어 `정정항목 | 정정전 | 정정후` 대응을 만들 수 없다. 다만 **섹션 텍스트에는
    정정 내용이 들어 있어** LLM 은 읽을 수 있다.

    예전에는 `FileNotFoundError: 본문 XML 없음` 으로 보고돼 파일이 없는 것처럼 읽혔다.
    파일은 있고, 형식이 다르다.
    """


def extract(doc: dict, index: CorpusIndex) -> list[CorrectionItem]:
    if doc.get("file_format") == "pdf+html":
        raise NotStructurable(
            "PDF 텍스트에는 표 구조가 없어 정정 항목을 구조화할 수 없습니다 "
            "(정정 내용은 Section 텍스트에 보존됨)")
    path = index.main_xml(doc["file_path"], doc["rcept_no"])
    if path is None:
        raise FileNotFoundError("본문 XML 없음")
    if doc["doc_group"] == "exchange":
        source = path.read_text(encoding="utf-8", errors="replace")
        return _from_exchange(parse_exchange(path), doc, source_html=source)
    cleaned = sanitize(path.read_text(encoding="utf-8", errors="replace"))
    try:
        root = ET.fromstring(cleaned)
    except ET.ParseError:
        from lxml import etree as lxml_etree
        root = lxml_etree.fromstring(
            cleaned.encode("utf-8"),
            lxml_etree.XMLParser(recover=True, huge_tree=True, encoding="utf-8"),
        )
    return _from_dart_xml(root, doc)


def build(corpus_root: str | Path = "data/corpus",
          out_dir: str | Path = "out/corrections") -> dict:
    index = CorpusIndex.build(corpus_root)
    records = [r for r in iter_manifest(corpus_root) if r["is_correction"]]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_group: dict[str, int] = {}
    kinds: dict[str, int] = {}
    empty: list[str] = []
    failures: list[dict] = []
    not_structurable: list[dict] = []
    total = 0

    with (out_dir / "correction_items.jsonl").open("w", encoding="utf-8") as fh:
        for doc in records:
            try:
                items = extract(doc, index)
            except NotStructurable as exc:
                # 결함이 아니라 원문 한계다. 실패와 섞으면 게이트가 늘 붉게 보인다.
                not_structurable.append({"doc_id": doc["doc_id"], "reason": str(exc)})
                continue
            except Exception as exc:  # noqa: BLE001 — 조용히 넘기지 않는다
                failures.append({"doc_id": doc["doc_id"],
                                 "reason": f"{type(exc).__name__}: {exc}"[:160]})
                continue
            if not items:
                empty.append(doc["doc_id"])
            for item in items:
                fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
                total += 1
                key = f"{item.diff_kind}/{item.before_kind}→{item.after_kind}"
                kinds[key] = kinds.get(key, 0) + 1
            per_group[doc["doc_group"]] = per_group.get(doc["doc_group"], 0) + len(items)

    report = {"corrections": len(records), "items": total,
              "not_structurable": not_structurable,
              "items_by_kind": kinds, "items_by_group": per_group, "documents_without_items": len(empty),
              "empty_examples": empty[:10], "failures": failures}
    (out_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 이 산출물이 **무엇으로 만들어졌는지** 남긴다. canonical 이 읽기 전에 검사한다 (B-03).
    write_stamp(out_dir, "correction_items",
                file_sha256(Path(corpus_root) / "manifest.jsonl"))
    return report


def _main() -> int:
    r = build()
    print(f"정정공시 {r['corrections']:,}건 → 정정항목 {r['items']:,}개")
    for k, v in sorted(r["items_by_group"].items()):
        print(f"  {k:<12}{v:>8,}")
    print("  ── 값 유형 ──")
    for k, v in sorted(r["items_by_kind"].items(), key=lambda x: -x[1]):
        print(f"  {k:<12}{v:>8,} ({v / r['items'] * 100:5.1f}%)")
    print(f"  항목 0개 문서 {r['documents_without_items']:,} / 실패 {len(r['failures'])}")
    for e in r["empty_examples"][:3]:
        print("    항목없음 예:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
