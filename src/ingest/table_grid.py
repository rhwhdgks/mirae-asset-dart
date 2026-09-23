"""공통 표 격자 엔진 — XML·HTML 공용.

표를 다루는 코드가 세 벌이었고 기능이 제각각이었다. 지금은 셋 다 이 모듈 위에 있다.

| 모듈 | rowspan | colspan | 값 판별 | locator |
|---|---|---|---|---|
| `dart_xml._table_grid` | 원점참조 | 원점참조 | 구분 없음 | 없음 |
| `dart_form.extract_table` | carry | **미처리** | TD=라벨 고정 | 없음 |
| `exchange_html.parse_html` | carry | **미처리** | `span.xforms_input` | 없음 |

이제 격자는 여기 하나뿐이고, 각 모듈은 `is_value()` 로 **역할 판정만** 한다.

2차 검수 코드 결함 K-09·10·13·14·15·16·17·18 과 치명적 결함 C-01·C-02 가 전부
여기서 나왔다. 근본 원인은 **격자 로직과 역할 판정이 뒤엉켜** 있었다는 것이다.
격자는 포맷과 무관한데 파서마다 다시 짰다.

## 분리

```text
[공통]   표 element → SourceCell[] + CellPlacement[]      ← 이 모듈
[포맷별] 역할 판정 (header / label / value)                ← 호출 측
[공통]   같은 행에서 가장 가까운 라벨에 값을 붙임            ← 이 모듈
```

## 해결되는 것

- 원문 노드를 순회하므로 **실제 locator** 가 나온다 (C-01)
- `SourceCell`/`CellPlacement` 가 검수 권고 스키마 그대로다 (C-03)
- colspan 헤더 전파가 세 문서군 전부에 적용된다
- 라벨-값을 **행 안에서 근접 매칭**해 `자산총액 | 값 | 부채총액 | 값` 을 각각 대응시킨다 (C-02/K-14)
- 결과가 dict 가 아니라 행 목록이라 반복 라벨이 보존된다 (K-17)
- 중첩 표의 `TR` 을 상위 표가 삼키지 않는다 (K-09)
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "SourceCell", "Placement", "Pairing", "TableGrid", "build_grid",
    "pair_label_values", "default_text",
]

#: 태그 이름은 포맷마다 다르다. 대소문자를 무시하고 비교한다.
_ROW_TAGS = {"TR"}
_CELL_TAGS = {"TD", "TH", "TU", "TE"}
_TABLE_TAGS = {"TABLE"}
_HEAD_TAGS = {"THEAD"}


def _tag(el) -> str:
    t = el.tag
    return t.upper() if isinstance(t, str) else ""


def default_text(el) -> str:
    """셀 텍스트. **중첩 표 내용은 제외한다.**

    `itertext()` 는 하위 `TABLE` 안의 글자까지 긁어온다. 중첩 표를 담은 셀이
    `"안쪽B"` 처럼 하위 표 내용과 뒤섞인 값을 갖게 된다(K-09 의 텍스트 측면).
    하위 표는 별도 격자로 따로 처리하므로 여기서는 건너뛴다.
    """
    parts: list[str] = []

    def walk(node) -> None:
        if node.text:
            parts.append(node.text)
        for child in node:
            if _tag(child) not in _TABLE_TAGS:
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(el)
    return " ".join("".join(parts).split())


#: 하위 호환 별칭
_text = default_text


def _span(el, *names: str) -> int:
    for name in names:
        raw = el.get(name)
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                return 1
    return 1


@dataclass(frozen=True)
class SourceCell:
    """원문 셀 하나. **1행 = 원문 element 1개.**"""

    index: int                   #: 표 안에서의 발견 순서
    locator: str                 #: `"TABLE[1]/TR[6]/TD[2]"` — 원문 위치
    origin_row: int
    origin_col: int
    rowspan: int
    colspan: int
    tag: str
    text: str
    acode: str | None
    aunit: str | None
    aunitvalue: str | None
    in_head: bool                #: THEAD 안에 있는가
    #: 원문 element. 포맷별 역할 판정이 속성·자식을 직접 보게 열어둔다
    #: (거래소 HTML 의 `span.xforms_input` 등). 직렬화 대상이 아니므로 비교·표시에서 제외한다.
    node: object | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Pairing:
    """라벨-값 대응 한 건."""

    logical_row: int
    labels: list[SourceCell]
    value: SourceCell


@dataclass(frozen=True)
class Placement:
    """논리 좌표 한 칸. span 영역의 모든 좌표가 같은 `SourceCell` 을 가리킨다."""

    logical_row: int
    logical_col: int
    cell_index: int
    inherited: bool              #: 원점이 아니라 span 으로 채워진 자리인가


@dataclass
class TableGrid:
    locator: str
    cells: list[SourceCell] = field(default_factory=list)
    placements: list[Placement] = field(default_factory=list)
    n_head_rows: int = 0
    n_rows: int = 0
    n_cols: int = 0
    issues: list[str] = field(default_factory=list)
    #: (row, col) → cells 색인. 선형 탐색이면 셀 조회가 O(n) 이라
    #: 격자 전체를 훑는 호출부에서 O(n^2) 이 된다.
    _at: dict[tuple[int, int], int] = field(default_factory=dict, repr=False)

    def cell_at(self, row: int, col: int) -> SourceCell | None:
        idx = self._at.get((row, col))
        return None if idx is None else self.cells[idx]

    def header_path(self, col: int) -> str:
        """열의 헤더 계층. colspan 전파 덕분에 상위가 자식 열에 남는다."""
        parts: list[str] = []
        for row in range(self.n_head_rows):
            cell = self.cell_at(row, col)
            if cell and cell.text and cell.text not in parts:
                parts.append(cell.text)
        return " > ".join(parts)

    def header_locators(self, col: int) -> tuple[str, ...]:
        """`header_path` 를 만든 셀들의 원문 위치.

        라벨이 **왼쪽 칸이 아니라 열 머리글**에서 올 때도 근거를 가리킬 수 있어야 한다.
        이게 없으면 지분공시 Field 의 60% 가 「값은 있는데 라벨이 어디서 왔는지 모르는」
        상태가 된다 — 값은 맞으므로 리포트로는 보이지 않는다.
        """
        out: list[str] = []
        seen: set[str] = set()
        for row in range(self.n_head_rows):
            cell = self.cell_at(row, col)
            if cell and cell.text and cell.text not in seen:
                seen.add(cell.text)
                out.append(cell.locator)
        return tuple(out)


def _direct_rows(table, table_locator: str) -> list[tuple[object, bool, str]]:
    """중첩 표의 행을 상위 표가 삼키지 않도록 **직계 자손만** 모은다.

    `table.iter("TR")` 은 하위 표의 `TR` 까지 가져온다(K-09).
    """
    rows: list[tuple[object, bool, str]] = []

    def walk(node, in_head: bool, path: str) -> None:
        counter: dict[str, int] = {}
        for child in node:
            name = _tag(child)
            if not name:
                continue
            idx = counter.get(name, 0)
            counter[name] = idx + 1
            loc = f"{path}/{name}[{idx}]"
            if name in _TABLE_TAGS:
                continue                       # 중첩 표는 별도 격자로 처리한다
            if name in _ROW_TAGS:
                rows.append((child, in_head, loc))
            else:
                walk(child, in_head or name in _HEAD_TAGS, loc)

    walk(table, False, table_locator)
    return rows


def build_grid(table, table_locator: str = "TABLE[0]", *, text_of=None) -> TableGrid:
    """표 element → 물리 셀 + 논리 배치. 포맷(XML/HTML)에 의존하지 않는다.

    `text_of(el) -> str` 로 셀 텍스트 추출을 포맷별로 바꿀 수 있다. 기본값은
    중첩 표를 제외한 전체 텍스트. 거래소 HTML 은 `noprint` span 을 빼기 위해 넘긴다.
    """
    text_of = text_of or default_text
    grid = TableGrid(locator=table_locator)
    occupied: dict[tuple[int, int], int] = {}

    rows = _direct_rows(table, table_locator)
    # 머리글은 **선두 연속 구간만** 센다. `max(r+1)` 로 잡으면 THEAD 가 뒤에 있는
    # 표에서 앞쪽 본문 행까지 머리글로 오인한다.
    for r, (_tr, in_head, _loc) in enumerate(rows):
        if not in_head:
            break
        grid.n_head_rows = r + 1

    for r, (tr, in_head, row_loc) in enumerate(rows):
        col = 0
        counter: dict[str, int] = {}
        for child in tr:
            name = _tag(child)
            if not name:
                continue
            idx = counter.get(name, 0)
            counter[name] = idx + 1
            if name not in _CELL_TAGS:
                continue
            while (r, col) in occupied:
                col += 1
            rowspan = _span(child, "ROWSPAN", "rowspan")
            colspan = _span(child, "COLSPAN", "colspan")
            cell = SourceCell(
                index=len(grid.cells), locator=f"{row_loc}/{name}[{idx}]",
                origin_row=r, origin_col=col, rowspan=rowspan, colspan=colspan,
                tag=name, text=text_of(child), node=child,
                acode=child.get("ACODE"), aunit=child.get("AUNIT"),
                aunitvalue=child.get("AUNITVALUE"), in_head=in_head,
            )
            grid.cells.append(cell)
            for dr in range(rowspan):
                for dc in range(colspan):
                    pos = (r + dr, col + dc)
                    if pos in occupied:
                        # 조용히 덮어쓰지 않는다 (K-13)
                        grid.issues.append(f"span overlap at {pos} ({cell.locator})")
                        continue
                    occupied[pos] = cell.index
                    grid._at[pos] = cell.index
                    grid.placements.append(Placement(
                        logical_row=pos[0], logical_col=pos[1],
                        cell_index=cell.index, inherited=not (dr == 0 and dc == 0),
                    ))
            col += colspan

    grid.n_rows = max((p.logical_row for p in grid.placements), default=-1) + 1
    grid.n_cols = max((p.logical_col for p in grid.placements), default=-1) + 1
    return grid


def pair_label_values(
    grid: TableGrid,
    is_value,
    *,
    carry_labels: bool = True,
) -> list[Pairing]:
    """행 안에서 **가장 가까운 왼쪽 라벨**에 값을 붙인다.

    이전 구현은 행의 모든 라벨을 이어 하나의 경로로 만든 뒤 모든 값에 같은 경로를 줬다.
    그래서 `자산총액 | 55,245,423 | 부채총액 | 21,892,979` 가
    `자산총액 > 부채총액` 하나로 뭉개져 어느 값이 자산인지 알 수 없었다(C-02/K-14).

    `is_value(cell) -> bool` 은 포맷별 역할 판정이다. 격자는 공통, 판정만 다르다.
    `carry_labels` 는 rowspan 으로 이어지는 상위 라벨을 앞에 붙일지 여부다.

    라벨의 **span 이 소비 규칙을 정한다.**

    - `rowspan > 1` 인 라벨은 행 전체를 덮는 상위 분류다. 행 안의 모든 값에 붙고
      소비되지 않는다 (`수출 | 미주 | 유럽 | 아시아`).
    - `rowspan == 1` 인 라벨은 바로 다음 값 하나에만 붙고 소비된다
      (`자산총액 | 값 | 부채총액 | 값`).

    이 구분이 없으면 원점 행과 상속 행의 동작이 어긋난다.
    """
    out: list[Pairing] = []
    #: 이미 내보낸 값 셀. `rowspan` 으로 여러 행에 걸친 값 셀은 **하나의 값**이므로
    #: 행마다 다시 내보내면 안 된다. 그러면 같은 원문 위치를 가진 Field 가 둘이 되어
    #: `block_id` 가 충돌한다(실측 141건). 라벨은 원점 행의 것을 쓴다.
    emitted: set[int] = set()
    for row in range(grid.n_rows):
        spanning: list[SourceCell] = []     # 행 전체를 덮는 라벨 (소비되지 않음)
        pending: list[SourceCell] = []      # 다음 값 하나에만 붙는 라벨
        seen: set[int] = set()
        for col in range(grid.n_cols):
            cell = grid.cell_at(row, col)
            if cell is None or cell.index in seen:
                continue
            seen.add(cell.index)
            if is_value(cell):
                if cell.index in emitted:
                    pending = []
                    continue
                emitted.add(cell.index)
                labels = (spanning if carry_labels else []) + pending
                out.append(Pairing(logical_row=row, labels=list(labels), value=cell))
                pending = []
            elif cell.rowspan > 1 or cell.origin_row != row:
                if carry_labels:
                    spanning.append(cell)
            else:
                pending.append(cell)
    return out
