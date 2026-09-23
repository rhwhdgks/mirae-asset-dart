"""거래소공시 파서 — 확장자는 `.xml` 이지만 실제는 HTML.

코퍼스의 `raw/exchange/**/*.xml` 1,469건은 루트가 `<html>` 이며 DART 문서 XML이 아니다.
`meta charset` 은 **1,469건 전부 euc-kr 로 선언돼 있으나 실제 내용은 UTF-8** 이므로,
파서에 인코딩 자동감지를 맡기면 전량 깨진다. 반드시 UTF-8로 강제 디코딩한다.

## 구조

DART 공시뷰어가 내보내는 xforms 폼이라 라벨과 값이 CSS class로 구분된다.

```html
<td rowspan="4"><span>2. 계약내역</span></td>          ← 라벨 (class 없음)
<td><span>계약금액(원)</span></td>                      ← 라벨
<td><span class="xforms_input">22,764,764,160,000</span></td>   ← 값
```

전수 확인: span class는 `(없음)` 35,332 · `xforms_input` 30,429 · `noprint` 2,826 뿐이고
`xforms_input` 이 없는 문서는 0건이다. 400건 표본에서 **라벨·값이 한 셀에 섞이는 경우는 0%**,
빈 셀도 0% 라 셀 단위로 역할이 깨끗하게 갈린다.

## 격자는 공통 엔진이 만든다

이전 구현은 자체 rowspan carry 만 있고 **colspan 을 처리하지 않았으며**, 한 행의 라벨을
전부 이어 하나의 경로로 만든 뒤 그 행의 모든 값에 같은 경로를 줬다(C-02/K-14).
지금은 `table_grid` 가 격자를 만들고, 이 모듈은 **역할 판정만** 한다 —
`span.xforms_input` 이 있으면 값.

결과도 dict 가 아니라 **레코드 목록**이라 같은 라벨이 반복돼도 덮어쓰지 않는다(K-17).
`fields` dict 는 하위 호환용 파생값이다.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from lxml import html as lxml_html

from .table_grid import SourceCell, build_grid, pair_label_values

__all__ = [
    "ExchangeField", "ExchangeForm", "parse_html", "parse_file",
    "cell_text", "normalize_amount", "split_related",
]

_VALUE_CLASS = "xforms_input"
_SKIP_CLASS = "noprint"
_WS = re.compile(r"\s+")
_NUM = re.compile(r"^-?[\d,]+(?:\.\d+)?$")
#: `2024-10-15 단일판매ㆍ공급계약체결` 형태의 관련공시 역참조
_RELATED = re.compile(r"(\d{4}-\d{2}-\d{2})\s*([^\d]{2,40}?)(?=\s*\d{4}-\d{2}-\d{2}|\s*$)")


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return _WS.sub(" ", text.replace("\xa0", " ")).strip()


def _classes(el) -> list[str]:
    return (el.get("class") or "").split()


def cell_text(el) -> str:
    """셀 텍스트. 인쇄 전용 `noprint` span 은 뺀다."""
    spans = [s for s in el.iter("span") if _SKIP_CLASS not in _classes(s)]
    if spans:
        return _clean(" ".join(s.text_content() for s in spans))
    return _clean(el.text_content())


#: 하위 호환 별칭
_cell_text = cell_text


def _is_value(cell: SourceCell) -> bool:
    """서식이 값이라고 표시한 셀 — `span.xforms_input` 을 가진 칸."""
    node = cell.node
    if node is None:
        return False
    return any(_VALUE_CLASS in _classes(s) for s in node.iter("span"))


@dataclass(frozen=True)
class ExchangeField:
    label_path: str
    value: str
    #: 원문 위치 — 빌더가 순번을 지어내지 않도록 파서가 직접 들고 온다 (C-01)
    source_locator: str
    table_locator: str
    logical_row: int
    logical_col: int
    #: 이 값에 붙은 라벨 셀들의 locator. 근거 추적용 (C-02)
    label_locators: tuple[str, ...]
    occurrence: int
    #: 같은 행에 라벨이 없어 **앞 행의 라벨 전용 행**에서 물려받았는가
    label_from_prev_row: bool


@dataclass
class ExchangeForm:
    """거래소공시 한 건에서 뽑은 라벨→값 전체."""

    title: str
    records: list[ExchangeField] = field(default_factory=list)
    #: 값 없이 라벨만 있는 안내문·각주. 버리지 않고 남긴다.
    notes: list[str] = field(default_factory=list)

    @property
    def fields(self) -> dict[str, str]:
        """하위 호환 dict. 라벨이 반복되면 `[2]`, `[3]` … 을 붙여 덮어쓰지 않는다."""
        out: dict[str, str] = {}
        seen: dict[str, int] = {}
        for rec in self.records:
            n = seen.get(rec.label_path, 0)
            seen[rec.label_path] = n + 1
            out[rec.label_path if n == 0 else f"{rec.label_path} [{n + 1}]"] = rec.value
        return out

    def get(self, *candidates: str) -> str | None:
        """라벨 경로 일부로 값을 찾는다.

        **공백 무시 비교.** 같은 항목이라도 시장에 따라 라벨 띄어쓰기가 다르다
        (KOSPI `최근매출액(원)` vs KOSDAQ `최근 매출액(원)`).
        정확 일치 → **경로 끝 일치** → 부분 일치 순으로 찾는다.

        **부분 일치 후보가 여럿이면 고르지 않는다.** 삽입 순서의 첫 값을 집으면
        순서가 바뀔 때 조용히 다른 값이 나온다. 실제로 `최근매출액(원)` 은
        `2. 계약내역 > 최근 매출액(원)`(712억)과 `- 최근 매출액(원)`(16조) 두 곳에 걸린다 —
        계약금액 ÷ 최근매출액 = 매출액대비 산술로 확인하면 전자가 옳다.
        경로 끝 일치가 그 구분을 해 주므로 그 단계에서 결정되고,
        그래도 여럿이면 `notes` 에 남기고 `None` 을 준다.
        """
        squeezed = {k.replace(" ", ""): v for k, v in self.fields.items()}
        for candidate in candidates:
            key = candidate.replace(" ", "")
            if key in squeezed:
                return squeezed[key]
        for candidate in candidates:                       # 경로의 **마지막 마디**와 일치
            key = candidate.replace(" ", "")
            hit = [(k, v) for k, v in squeezed.items()
                   if key and k.rsplit(">", 1)[-1].lstrip("-") == key]
            if len(hit) == 1:
                return hit[0][1]
            if len(hit) > 1:
                # **번호 절 안에 있는 것이 정본이다.** `2. 계약내역 > 최근 매출액(원)` 은
                # 서식이 정한 자리고, `- 최근 매출액(원)` 은 주석 행이다.
                nested = [v for k, v in hit if ">" in k]
                if len(nested) == 1:
                    return nested[0]
                self.notes.append(f"모호한 라벨: {candidate} → 후보 {len(hit)}")
                return None
        for candidate in candidates:
            key = candidate.replace(" ", "")
            hit = [v for k, v in squeezed.items() if key and key in k]
            if len(hit) == 1:
                return hit[0]
            if len(hit) > 1:
                self.notes.append(f"모호한 라벨: {candidate} → 후보 {len(hit)}")
                return None
        return None


def _row_labels(grid, row: int) -> list[SourceCell]:
    out, seen = [], set()
    for col in range(grid.n_cols):
        cell = grid.cell_at(row, col)
        if cell is None or cell.index in seen:
            continue
        seen.add(cell.index)
        if cell.text and not _is_value(cell):
            out.append(cell)
    return out


def parse_html(source: str) -> ExchangeForm:
    """HTML 문자열 → 라벨 경로 → 원문 값."""
    root = lxml_html.fromstring(source)
    form = ExchangeForm(title=_clean(root.findtext(".//title") or ""))
    seen: dict[str, int] = {}

    for ti, table in enumerate(root.iter("table")):
        grid = build_grid(table, f"TABLE[{ti}]", text_of=_cell_text)
        by_row: dict[int, list] = defaultdict(list)
        for pair in pair_label_values(grid, _is_value):
            by_row[pair.logical_row].append(pair)

        #: 라벨만 있는 행. 긴 자유서술 항목은 라벨 행과 값 행이 분리돼 있다
        #: (`9. 기타 투자판단과 관련한 중요사항` 다음 행에 본문). 버리면 본문이 통째로 사라진다.
        pending: list[SourceCell] | None = None

        for row in range(grid.n_rows):
            pairs = by_row.get(row)
            if not pairs:
                labels = _row_labels(grid, row)
                if labels:
                    pending = labels
                continue
            for pair in pairs:
                labels, inherited = pair.labels, False
                if not any(l.text for l in labels) and pending:
                    labels, inherited = pending, True
                path = " > ".join(dict.fromkeys(l.text for l in labels if l.text))
                path = path or "(라벨 없음)"
                occ = seen.get(path, 0)
                seen[path] = occ + 1
                form.records.append(ExchangeField(
                    label_path=path, value=pair.value.text,
                    source_locator=pair.value.locator, table_locator=grid.locator,
                    logical_row=row, logical_col=pair.value.origin_col,
                    label_locators=tuple(l.locator for l in labels),
                    occurrence=occ, label_from_prev_row=inherited,
                ))
            pending = None

        if pending:
            form.notes.append(" > ".join(dict.fromkeys(l.text for l in pending if l.text)))

    return form


def parse_file(path: str | Path) -> ExchangeForm:
    """**UTF-8 강제.** meta charset(euc-kr) 을 신뢰하면 전량 깨진다."""
    return parse_html(Path(path).read_text(encoding="utf-8", errors="replace"))


# ----------------------------------------------------------------------- 정규화


def normalize_amount(raw: str | None) -> int | None:
    """`22,764,764,160,000` → int. `-`·공란·비수치는 None (0이 아니다)."""
    if not raw:
        return None
    text = raw.strip().replace(",", "")
    if text in {"-", "", "해당사항없음", "해당사항 없음"}:
        return None
    if text.startswith("△"):  # DART 음수 표기
        text = "-" + text[1:]
    if not _NUM.match(text.replace("-", "", 1) if text.startswith("-") else text):
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def split_related(raw: str | None) -> list[tuple[str, str]]:
    """관련공시 필드를 (날짜, 공시유형) 목록으로 쪼갠다.

    계약 해지 20건은 전부 이 필드로 원계약을 역참조한다
    (`"2024-10-15 단일판매ㆍ공급계약체결"`). 다만 원계약이 코퍼스에 있는 것은 11건뿐이므로,
    나머지는 `target_hint` 로 남겨야 한다 — 임의로 연결하지 않는다.
    """
    if not raw or raw.strip() in {"-", ""}:
        return []
    return [(m.group(1), _clean(m.group(2))) for m in _RELATED.finditer(raw)]
