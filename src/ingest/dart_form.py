"""DART XML 서식 문서 → Field + Cell. 공통 표 격자 엔진(`table_grid`) 위에서 동작한다.

## 이전 구현의 결함 (2차 검수 C-02/K-14)

자체 rowspan carry 로직만 있고 colspan 을 다루지 않았으며, 한 행의 라벨을 전부 이어
하나의 경로로 만들었다. 그래서
`자산총액 | 55,245,423 | 부채총액 | 21,892,979` 가 `자산총액 > 부채총액` 하나로 뭉개져
**어느 값이 자산인지 알 수 없었다**. `table_grid.pair_label_values` 의 근접 매칭으로 해결.

## TD 값 문제 (K-15) — 역할을 추측하지 않는다

`TE`/`TU` 가 전혀 없고 `TD` 에 값이 들어있는 표가 실재한다. 검수는 이를 유실로 봤으나
실측 결과 **해당 표는 섹션 마크다운에 온전히 남아 있어 유실이 아니다**. 없는 것은
구조화된 Field 뿐이다.

그렇다면 `TD` 중 무엇이 값인지 판정할 수 있는가 — 코퍼스에서 두 가설을 검증했고 둘 다 실패했다:

* 열 패리티(짝수=라벨/홀수=값): `법적성격 | 주식회사` 처럼 **값이 숫자가 아닌** 표에서 무너짐
* 문서 간 불변성(불변=라벨): 적중률 64.5%. `-` 같은 값이 우연히 불변이고,
  회사명 같은 라벨이 가변이라 갈리지 않음

따라서 **역할을 추론하지 않는다**. Field 는 서식이 역할을 명시한 곳(`TE`/`TU`)에서만 만들고,
나머지 표는 `Cell` 격자로 위치·헤더와 함께 내보내 하위 계층이 판단하게 둔다.
없는 근거를 지어내느니 격자를 그대로 넘기는 편이 낫다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .table_grid import TableGrid, build_grid, pair_label_values

__all__ = ["FormField", "FormCell", "extract_table", "extract_cells", "extract_document"]

_VALUE_TAGS = {"TE", "TU"}


@dataclass(frozen=True)
class FormField:
    """서식이 값이라고 명시한 셀(`TE`/`TU`)과 그에 붙은 라벨."""

    label_path: str
    value: str
    acode: str | None
    aunit: str | None
    aunitvalue: str | None
    #: 원문 위치 — 빌더가 순번을 지어내지 않도록 파서가 직접 들고 온다 (C-01)
    source_locator: str
    table_locator: str
    origin_row: int
    origin_col: int
    #: 이 값에 붙은 라벨 셀들의 locator. 근거 추적용 (C-02)
    label_locators: tuple[str, ...]
    occurrence: int


@dataclass(frozen=True)
class FormCell:
    """역할 판정 없는 격자 셀. 위치와 헤더만 기록한다."""

    table_locator: str
    source_locator: str
    logical_row: int
    logical_col: int
    origin_row: int
    origin_col: int
    rowspan: int
    colspan: int
    tag: str
    text: str
    #: THEAD colspan 을 전파한 열 헤더 경로. 없으면 빈 문자열
    col_header_path: str
    #: 같은 행 0열의 텍스트. 라벨이라 단정하지 않고 위치 사실만 남긴다
    row_head_text: str
    in_head: bool
    #: 격자에서 파생된 위치(rowspan/colspan 상속)이면 True — 원본 셀이 아니다
    inherited: bool


def extract_table(table, table_locator: str = "TABLE[0]") -> list[FormField]:
    grid = build_grid(table, table_locator)
    return _fields(grid)


def _fields(grid: TableGrid) -> list[FormField]:
    """값 셀 → Field.

    **라벨은 두 방향에 있다.** 폼 표는 값의 **왼쪽**에, 데이터 표는 값의 **위쪽**(열 헤더)에
    있다. 왼쪽만 보면 지분공시 특별관계자 표처럼 헤더가 위에 있는 표에서 라벨이 통째로 비고,
    실측 Field 의 **56.9%가 경로 없음**이 된다. 경로 없는 값은 무엇의 값인지 알 수 없어
    검색에도 답변에도 못 쓴다.

    그래서 같은 행에 라벨이 없으면 **열 헤더 경로 + 행 머리글**로 되돌린다.
    """
    headers = [grid.header_path(c) for c in range(grid.n_cols)]
    seen: dict[str, int] = {}
    out: list[FormField] = []
    for pair in pair_label_values(grid, lambda c: c.tag in _VALUE_TAGS):
        labels, value = pair.labels, pair.value
        parts = [l.text for l in labels if l.text]
        llocs = tuple(l.locator for l in labels)
        head_txt = headers[value.origin_col] if value.origin_col < len(headers) else ""
        if not parts:
            row_head = grid.cell_at(pair.logical_row, 0)
            parts = [x for x in (
                row_head.text if row_head is not None and row_head is not value else "",
                head_txt,
            ) if x]
            # **되돌린 라벨도 근거를 남긴다.** 안 남기면 값만 있고 출처가 없다.
            llocs = tuple(x for x in (
                (row_head.locator if row_head is not None and row_head is not value
                 and row_head.text else None),
            ) if x) + (grid.header_locators(value.origin_col) if head_txt else ())
        elif head_txt and head_txt not in parts:
            # **왼쪽 라벨이 있어도 열 헤더를 붙인다.** 안 붙이면 행의 **첫 값만**
            # 열 차원을 잃는다 — 둘째 값부터는 라벨이 없어 폴백을 타므로 헤더가 붙고,
            # 첫 값은 왼쪽 라벨이 있어 폴백을 건너뛰기 때문이다. 그 결과
            # `기초잔액 | 취득 | 처분 | 기말잔액` 같은 표에서 같은 라벨 경로가
            # 서로 다른 값을 가리킨다 (실측 25,286 조합).
            parts = parts + [head_txt]
            llocs = llocs + grid.header_locators(value.origin_col)
        path = " > ".join(dict.fromkeys(parts))
        occ = seen.get(path, 0)
        seen[path] = occ + 1
        out.append(FormField(
            label_path=path, value=value.text, acode=value.acode,
            aunit=value.aunit, aunitvalue=value.aunitvalue,
            source_locator=value.locator, table_locator=grid.locator,
            origin_row=value.origin_row, origin_col=value.origin_col,
            label_locators=llocs,
            occurrence=occ,
        ))
    return out


def extract_cells(grid: TableGrid) -> list[FormCell]:
    """격자의 모든 논리 위치를 셀로 편다. `TE` 유무와 무관하게 항상 만들 수 있다."""
    out: list[FormCell] = []
    for p in grid.placements:
        cell = grid.cells[p.cell_index]
        head = grid.cell_at(p.logical_row, 0)
        out.append(FormCell(
            table_locator=grid.locator, source_locator=cell.locator,
            logical_row=p.logical_row, logical_col=p.logical_col,
            origin_row=cell.origin_row, origin_col=cell.origin_col,
            rowspan=cell.rowspan, colspan=cell.colspan, tag=cell.tag, text=cell.text,
            col_header_path=grid.header_path(p.logical_col),
            row_head_text=head.text if head is not None and p.logical_col > 0 else "",
            in_head=cell.in_head, inherited=p.inherited,
        ))
    return out


def extract_document(root, *, with_cells: bool = False
                     ) -> tuple[list[FormField], list[FormCell]]:
    """문서의 모든 표 → (Field, Cell). 중첩 표도 각각 독립 격자로 처리한다.

    Field 는 `TE`/`TU` 가 있는 표에서만 나오고, Cell 은 **모든 표**에서 나온다.

    Cell 은 `with_cells` 로 켜야 만든다. 항상 만들면 Field 만 필요한 호출부가
    major·holding 기준 240만 개를 만들어 그대로 버린다.
    """
    fields: list[FormField] = []
    cells: list[FormCell] = []
    for i, table in enumerate(root.iter("TABLE")):
        grid = build_grid(table, f"TABLE[{i}]")
        fields.extend(_fields(grid))
        if with_cells:
            cells.extend(extract_cells(grid))
    return fields, cells
