"""Typed narrative-matrix composition input and deterministic safe rendering.

The retrieval fanout is internal execution metadata.  This module keeps that
shape through composition so a comparison is not inferred by re-parsing the
public display labels or by passing an unbounded disclosure block to HCX.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
import re
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation

from app.textkit import (
    collapse_padded_headings, josa, strip_leading_item_number)
from app.markdown_table import is_table_rule as _is_table_rule
from app.markdown_table import table_cells as _table_cells

from src.canonical.security import PROMPT_DATA_BEGIN, PROMPT_DATA_END
from .limitations import append_public_qualifications, render_safe_limitations
from .money import format_source_money_exact, normalize_source_money_unit

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_PLAIN_NUMERIC_AMOUNT = re.compile(r"[+-]?\(?\d[\d,]*(?:\.\d+)?\)?")
_NARRATIVE_LIMITS = {
    "narrative_subject_partial",
    "slot_not_confirmed", "narrative_record_budget_exhausted",
    "narrative_result_budget_exhausted", "narrative_source_roundtrip_mismatch",
    "narrative_cell_not_found", "narrative_fanout_limit",
}

_BUSINESS_EVIDENCE_MARKERS = (
    "영위", "제조", "판매", "공급", "연구개발", "설계", "서비스", "운영",
    "자회사", "계열사", "사업부문", "금융그룹", "지주회사", "경영관리",
    "보험", "제품", "기자재", "발전", "플랫폼", "파이프라인",
)

_COMPARISON_TERM_STOPWORDS = {
    "가장", "각각", "관련", "구성", "글로벌", "기업", "기업입니다", "회사", "회사는", "내용",
    "다음", "당사", "당사는", "매출", "매출액", "매출비중", "부문", "사업",
    "사업부문", "서비스", "연결실체", "영역", "완제품", "제조", "주요", "제품",
    "판매", "판매하고", "판매합니다", "생산", "생산하고", "생산합니다", "제조하고",
    "제조합니다", "합니다", "있습니다", "현황", "억원", "백만원", "조원", "천원",
    "본사", "거점", "한국과", "매출당사는", "매출실적", "매출은", "전년", "대비",
    "감소하였습니다", "증가하였습니다", "부문별로", "부문별로는", "보면", "동기",
    "구분", "설명", "품목", "사업의내용", "주요사업내역",
}

_QUALIFIED_BUSINESS_AXIS_ALIASES = {
    # Korean financial disclosures commonly label investment-banking work as
    # ``기업금융``.  This is a disclosure-domain axis alias, not an issuer or
    # question-specific rewrite.  It is used only to collect cited sentences;
    # the source surface itself is preserved in the answer.
    "투자금융": ("투자금융", "기업금융"),
    "기업금융": ("기업금융", "투자금융"),
    "자산운용": ("자산운용",),
}
_DIRECT_VALUE = (
    r"(?:\d[\d,]*(?:\.\d+)?조\s*)?\d[\d,]*(?:\.\d+)?(?:억원|"
    r"조원|"
    r"백만원|천원|원|개|%)"
)
_DIRECT_METRIC_PATTERNS = (
    re.compile(rf"(?P<label>매출(?:액|\s*총액)?)(?:은|이|가)?\s*(?P<value>{_DIRECT_VALUE})"),
    re.compile(
        rf"(?P<label>[A-Za-z][A-Za-z0-9& +.-]{{0,18}}\s*부문)"
        rf"(?:은|이|가)?\s*(?P<value>{_DIRECT_VALUE})"),
    re.compile(
        rf"(?P<label>종속기업(?:\s*등)?)(?:은|이|가)?\s*(?P<value>{_DIRECT_VALUE})"),
)

# These aliases deliberately mirror the *structural* investment-table
# ontology used by the retrieval layer.  Composition does not infer an
# investment plan from a question or company name: a compact row is emitted
# only when one cited source block itself proves all four column roles.
_INVESTMENT_SLOT_HEADERS = {
    "투자대상": ("투자대상", "대상자산", "투자명"),
    "목적": ("투자목적", "목적"),
    "금액": ("투자액", "투자금액", "총소요자금", "총투자액", "금액"),
    "기간": ("투자기간", "기간"),
}
_INVESTMENT_SPENT_AMOUNT_MARKERS = (
    "기지출", "누적지출", "실제지출", "집행금액", "기투자", "투자실적",
)
_INVESTMENT_SPENT_HEADERS = (
    "기지출금액", "기지출액", "누적지출금액", "실제지출금액",
    "집행금액", "기투자금액", "투자실적",
)


def _clean(text: str | None, limit: int = 360) -> str:
    value = (text or "").replace(PROMPT_DATA_BEGIN, "").replace(PROMPT_DATA_END, "")
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    # The ellipsis is a display boundary, not a fabricated continuation.
    return value[:limit].rstrip() + "…"


_CLAUSE_BOUNDARY = re.compile(r"(?:다\.|[.!?])\s|;\s")


def _clean_at_boundary(text: str, limit: int) -> str:
    """Bound a conclusion line without ending it in the middle of a word.

    ``_clean`` cuts at an exact character index.  That is fine for one quoted
    sentence that is merely terse, but a before/after contrast is two
    labelled clauses joined by ``; ``, and cutting between them leaves one
    side of a comparison standing alone as if it were the whole answer.
    ``EDGE-011`` ended at ``사업의 성격에 따라 철강부문, 인프라(무…`` — inside a
    parenthesis, with the 2023 side shown and the 2025 side gone.

    So a multi-clause contrast is kept whole up to twice the budget; showing
    half of a comparison is worse than showing a long one.  Anything else
    cuts on a clause boundary, and only a part with no boundary at all falls
    back to a hard cut, which still lands on a space.
    """

    value = _clean(text, limit=len(text) + 1)
    if len(value) <= limit:
        return value
    if "; " in value and len(value) <= limit * 2:
        return value
    boundaries = [match.end() for match in _CLAUSE_BOUNDARY.finditer(value)
                  if match.end() <= limit]
    if boundaries:
        kept = value[:boundaries[-1]].rstrip().rstrip(";").rstrip()
        # 공개 답변의 말줄임표는 문장이 끝났어도 값이나 조건이 잘린 듯 보인다.
        # 뒤에 근거 색인이 이어진다는 사실을 명시해 축약이 정보 부재가 아님을
        # 알린다. 이 문구는 사실 주장이 아니라 표시 범위 안내다.
        return kept + " 세부 내용은 아래 근거 공시에서 확인할 수 있습니다."
    # 완결 경계를 찾지 못한 원문은 단어 중간에서 자르지 않는다. 길이보다
    # 문장 완결성이 우선이며, wide structured matrix는 아래에서 근거 색인으로
    # 별도 축약되므로 이 예외가 다시 수천 자 표 노출로 이어지지 않는다.
    return value


def _source_body(text: str | None) -> str:
    """Remove transport fences only; never alter source data before parsing."""

    return (text or "").replace(PROMPT_DATA_BEGIN, "").replace(PROMPT_DATA_END, "").strip()


def _declared_currency_unit(value: str) -> str:
    """Return one unambiguous monetary unit from a local unit declaration."""

    parts = [part.strip() for part in value.split(",")]
    # Monetary-column callers may use a scale declared alongside percent.
    # Display callers preserve the whole declaration separately.
    money_parts = [part for part in parts if part != "%"]
    if len(money_parts) != 1:
        return ""
    unit = normalize_source_money_unit(money_parts[0])
    # Match the complete declaration: a ratio/count must not inherit its
    # neighbouring monetary scale, and 십억원 must never match only 억원.
    if not re.fullmatch(
            r"백만달러|천달러|조원|십억원|억원|백만원|천원|KRW|USD|달러|원",
            unit, flags=re.IGNORECASE):
        return ""
    return unit


def _local_table_currency_unit(lines: list[str], table_start: int) -> str:
    """Bind a unit only to the immediately adjacent investment table."""

    return _declared_currency_unit(_local_table_unit_declaration(lines, table_start))


def _local_table_unit_declaration(lines: list[str], table_start: int) -> str:
    """Preserve the complete source unit when displaying a mixed-unit table."""

    # DART viewer HTML often serializes the unit as a one-cell markdown table
    # immediately before the data table.  Walk across only blank/table rows;
    # a prose line is a hard boundary so an unrelated earlier unit cannot
    # leak into the selected investment table.
    candidates = [(table_start, lines[table_start])]
    for before in range(table_start - 1, max(-1, table_start - 8), -1):
        line = lines[before]
        if not line.strip():
            continue
        candidates.append((before, line))
        if not line.lstrip().startswith("|"):
            break
    for before, candidate in candidates:
        match = re.search(r"\(\s*단위\s*[:：]\s*([^)]+)\)", candidate)
        if match is None:
            continue
        intervening = lines[before + 1:table_start]
        if all(not row.strip() or row.lstrip().startswith("|")
               for row in intervening):
            return match.group(1).strip()
    return ""


def _local_table_period_role(lines: list[str], table_start: int) -> str:
    """Bind an adjacent ``당기``/``전기`` label to one source table."""

    for before in range(table_start - 1, max(-1, table_start - 8), -1):
        line = lines[before]
        if not line.strip():
            continue
        if not line.lstrip().startswith("|"):
            break
        cells = _table_cells(line)
        role = next((cell.strip() for cell in cells
                     if cell.strip() in {"당기", "전기"}), "")
        if role:
            intervening = lines[before + 1:table_start]
            if all(not row.strip() or row.lstrip().startswith("|")
                   for row in intervening):
                return role
    return ""


def _investment_amount_display(amount: str, unit: str) -> str:
    """Attach a cited table currency unit only to a bare numeric amount.

    Investment rows often store the unit once above the table rather than in
    every amount cell.  Keeping the cell without that unit is ambiguous, but
    appending a unit to an already-labelled value could duplicate or distort
    the source.  Therefore this projection is limited to numeric-only cells
    and a currency unit explicitly declared by the same source block.
    """

    value = amount.strip()
    if not _PLAIN_NUMERIC_AMOUNT.fullmatch(value):
        return amount
    if not unit:
        return amount
    return format_source_money_exact(value, unit) or f"{value}{unit}"


def _compact_key(value: str) -> str:
    return re.sub(r"[\s·ㆍ:()\[\]{}._/-]", "", value or "")


def _public_narrative_label(value: str | None) -> str:
    """Return a concise disclosure heading for the public answer.

    Retrieval labels may contain the complete parser section path plus an
    internal ``[확인 항목: ...]`` suffix.  Both are useful while binding a
    claim, but repeating them before every displayed row makes an answer look
    like a parser dump.  Keep the leaf heading and requested topic while
    repairing only a missing boundary that is unambiguous in DART headings.
    """

    raw = " ".join((value or "").split())
    topic_match = re.search(r"\[확인 항목:\s*([^]]+)\]", raw)
    topic = topic_match.group(1).strip() if topic_match else ""
    raw = re.sub(r"\s*\[확인 항목:\s*[^]]+\]\s*", " ", raw).strip()
    if ">" in raw:
        raw = raw.rsplit(">", 1)[-1].strip()
    # Viewer HTML occasionally joins a section heading and its following
    # sentence (``계획당분기말``).  Insert punctuation; do not rewrite either
    # source phrase or infer a new heading.
    raw = re.sub(r"계획(?=(?:당분기말|당반기말|당기말))", "계획. ", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" :-")
    # 원문 목차 번호(``2.``·``12.``·``(1)``·``가.``·``II.``)는 그 문서 안에서만
    # 뜻이 있다. 답변에 그대로 실으면 읽는 사람에게는 잡음이다.
    raw = re.sub(r"^(?:\(\d{1,2}\)\s*|(?:[0-9]{1,2}|[IVX]{1,4}|[가-하])\s*[.)]\s*)", "", raw).strip()
    # ``topic`` 은 질문 문장에서 공백을 뺀 내부 키일 수 있다
    # (``투자금융자산운용관련사업현황``). 사람이 읽는 라벨이 아니므로 이미
    # 띄어쓰기가 있는 값만 덧붙인다.
    readable_topic = topic if (topic and " " in topic) else ""
    if raw and readable_topic and _compact_key(topic) not in _compact_key(raw):
        return f"{raw} ({readable_topic})"
    return raw or readable_topic or topic or "요청 항목"


def _public_narrative_surface(text: str) -> str:
    """Remove execution/methodology wording from an already verified answer.

    This helper is display-only.  It must be called after matrix verification;
    prompt text, typed cells, citations and trace data keep their exact IDs.
    The replacements change no company, period, topic, receipt, limitation or
    source assertion.
    """

    # 공시 표의 정렬용 공백(``구 분``·``합 계``)은 원문에서는 정렬이지만
    # 답변 문장 안에서는 오타로 읽힌다.
    value = collapse_padded_headings(text)
    replacements = (
        ("물질적 핵심 사업", "실제 주요 사업"),
        ("일부 matrix 좌표를", "일부 비교 항목을"),
        ("검증된 셀을 직접 대조하면 ", ""),
        ("검증된 인용 셀에서 확인한", "공시에서 확인한"),
        ("각 인용 셀에서 직접 확인되는", "각 공시에서 확인되는"),
        ("모든 인용 셀에 공통으로", "모든 비교 대상에서 공통으로"),
        ("각 인용 셀에서만", "각 비교 대상에서만"),
        ("비교 좌표의 확인된 인용 본문", "비교한 항목"),
        ("확인된 각 좌표의 인용 본문", "비교한 각 항목의 내용"),
        ("일부 matrix 좌표", "일부 비교 항목"),
        ("모든 좌표의", "모든 비교 항목의"),
        ("해당 좌표의 관련 본문", "해당 비교 항목의 관련 내용"),
        ("인용 본문 표현", "확인 내용"),
        ("인용 본문", "확인 내용"),
        ("인용 셀", "비교 항목"),
        ("인용 범위", "확인 범위"),
        ("검증된 셀", "확인된 항목"),
        ("source-roundtrip", "공시 원문 확인"),
        ("source_roundtrip", "공시 원문 확인"),
    )
    for old, new in replacements:
        value = value.replace(old, new)
    value = re.sub(r"확인 범위는 (\u00ab[^\u00bb]+\u00bb) 확인 내용이며",
                   r"확인 범위는 \1 항목이며", value)
    value = re.sub(r"계획(?=(?:당분기말|당반기말|당기말))", "계획. ", value)
    # 공시가 쓰는 회계연도 표기(``FY'25``·``FY2025``)가 다른 답변의 ``2025년``
    # 과 섞인다. 같은 해를 두 가지로 부르지 않도록 하나로 맞춘다 (#261 A-5).
    value = re.sub(
        r"FY['’]?(\d{2}|\d{4})\b",
        lambda m: f"20{m.group(1)}년" if len(m.group(1)) == 2 else f"{m.group(1)}년",
        value)
    # 두 줄 머리글이 눌리며 같은 해가 두 번 남는다(``2025년 (2025년)``).
    # 뜻이 같은 되풀이라 하나만 둔다 (#261 A-4).
    value = re.sub(r"(20\d\d년)\s*\(\s*\1\s*\)", r"\1", value)
    return value


def _markdown_cell(value: str) -> str:
    """Escape one display-only Markdown table cell."""

    return " ".join((value or "").split()).replace("|", "\\|")


#: A Markdown table row/rule, or this module's row-count trailer.  Neither
#: ever opens a new logical bullet on its own.
_TABLE_CONTINUATION_LINE = re.compile(r"^\s*(?:\|.*\||\(외\s*[0-9]+)")


def _logical_answer_lines(text: str) -> list[str]:
    """Group raw text lines back into their logical bullets.

    A cited cell digest can itself be a multi-line Markdown table (#86 18).
    Splitting on every raw newline would tear that table away from the
    ``- [cell-id] ...`` bullet that cites it, leaving only the header behind.
    Only a table row/rule or its trailer is folded into the previous line;
    every other line (a plain model sentence, a fresh bullet, the leading
    conclusion) keeps its own independent block exactly as before.
    """

    blocks: list[str] = []
    for line in text.splitlines():
        if blocks and _TABLE_CONTINUATION_LINE.match(line):
            blocks[-1] = f"{blocks[-1]}\n{line}"
        else:
            blocks.append(line)
    return blocks


def _narrative_excerpt_is_adequate(text: str, *, topic: str) -> bool:
    """Reject a coordinate that cannot yield one complete public finding."""

    summary = _compact_cell_text(text, topic=topic, limit=600)
    if not summary:
        return False
    return summary != (
        "구조 블록이 장문 단일 문장이라 내용을 임의로 축약하지 않았습니다.")


def _single_cell_prose_lines(text: str | None) -> tuple[str, ...]:
    """Recover prose serialized as a one-column markdown table.

    Some DART viewer sections wrap an entire business overview in one table
    cell.  Ignoring every pipe-prefixed line then turns a substantive section
    into an empty/placeholder cell.  Only an explicitly proved one-column
    table is unwrapped here; multi-column data tables remain structured and
    continue through their dedicated projections.
    """

    lines = _source_body(text).splitlines()
    prose: list[str] = []
    for index in range(len(lines) - 1):
        if (not lines[index].lstrip().startswith("|")
                or not lines[index + 1].lstrip().startswith("|")):
            continue
        header = _table_cells(lines[index])
        rule = _table_cells(lines[index + 1])
        if len(header) != 1 or len(rule) != 1 or not _is_table_rule(rule):
            continue
        if header[0]:
            prose.append(header[0])
        cursor = index + 2
        while cursor < len(lines) and lines[cursor].lstrip().startswith("|"):
            cells = _table_cells(lines[cursor])
            if len(cells) != 1 or _is_table_rule(cells):
                break
            if cells[0]:
                prose.append(cells[0])
            cursor += 1
    return tuple(dict.fromkeys(prose))


def _investment_header_positions(header: list[str]) -> dict[str, int]:
    """Resolve the four plan columns without confusing plan and spent amounts.

    DART tables commonly place ``총 소요자금`` beside ``기 지출금액``.
    Substring matching the generic alias ``금액`` therefore makes the plan
    amount ambiguous even though the source labels it precisely.  Prefer the
    strong plan-amount ontology, reject execution-to-date columns, and use a
    bare ``금액`` only when it is the one exact remaining header.  Every other
    role follows exact-first, unique-contained matching and still fails closed
    on ambiguity.
    """

    compact = [_compact_key(cell) for cell in header]
    positions: dict[str, int] = {}
    for role, aliases in _INVESTMENT_SLOT_HEADERS.items():
        if role == "금액":
            eligible = [
                index for index, value in enumerate(compact)
                if not any(marker in value for marker in _INVESTMENT_SPENT_AMOUNT_MARKERS)
            ]
            strong = tuple(_compact_key(alias) for alias in aliases if alias != "금액")
            matched = [
                index for index in eligible
                if any(alias == compact[index] for alias in strong)
            ]
            if not matched:
                matched = [
                    index for index in eligible
                    if any(alias in compact[index] for alias in strong)
                ]
            if not matched:
                matched = [index for index in eligible if compact[index] == "금액"]
        else:
            normalized = tuple(_compact_key(alias) for alias in aliases)
            matched = [
                index for index, value in enumerate(compact)
                if any(alias == value for alias in normalized)
            ]
            if not matched:
                matched = [
                    index for index, value in enumerate(compact)
                    if any(alias in value for alias in normalized)
                ]
        if len(matched) != 1:
            return {}
        positions[role] = matched[0]
    if len(set(positions.values())) != len(positions):
        return {}
    return positions


def _investment_spent_position(header: list[str]) -> int | None:
    """Resolve one execution-to-date column independently of plan amount."""
    compact = [_compact_key(cell) for cell in header]
    aliases = tuple(_compact_key(value) for value in _INVESTMENT_SPENT_HEADERS)
    exact = [index for index, value in enumerate(compact) if value in aliases]
    matched = exact or [
        index for index, value in enumerate(compact)
        if any(alias in value for alias in aliases)]
    return matched[0] if len(matched) == 1 else None


def _investment_rows(
        text: str | None,
        ) -> tuple[tuple[str, str, str, str, str, str | None], ...]:
    """Extract complete investment-plan rows from one source-rounded block.

    A row is useful only when the source table has all four typed headers.  A
    look-alike table, a partial header, or an uneven row is ignored rather than
    guessing which number is the plan amount.  This is intentionally a
    composition-only projection; it changes neither the QueryPlan nor claims.
    """

    lines = _source_body(text).splitlines()
    found: list[tuple[str, str, str, str, str, str | None]] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = index
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            index += 1
        block = lines[start:index]
        if len(block) < 3:
            continue
        header = _table_cells(block[0])
        if not header or not _is_table_rule(_table_cells(block[1])):
            continue
        positions = _investment_header_positions(header)
        if len(positions) != len(_INVESTMENT_SLOT_HEADERS):
            continue
        context = _compact_key(" ".join(lines[max(0, start - 8):start]))
        amount_header = _compact_key(header[positions["금액"]])
        if ("투자현황" in context and "계획" not in context
                and "계획" not in amount_header):
            # A status table's bare 투자액 is not a promised future amount.
            # Keep it on the status renderer with its original column roles.
            continue
        unit = _local_table_currency_unit(lines, start)
        spent_position = _investment_spent_position(header)
        needed = max((*positions.values(), *(() if spent_position is None
                                             else (spent_position,))))
        for row_line in block[2:]:
            row = _table_cells(row_line)
            if len(row) <= needed or _is_table_rule(row):
                continue
            values = tuple(row[positions[role]] for role in ("투자대상", "목적", "금액", "기간"))
            if any(not value for value in values):
                continue
            spent = row[spent_position] if spent_position is not None else None
            if spent is not None and not spent.strip():
                spent = None
            typed = (*values, unit, spent)
            if typed not in found:
                found.append(typed)
    return tuple(found)


def _bounded_table_summary(prefix: str, rows: list[str], *, limit: int) -> str:
    selected: list[str] = []
    used = len(prefix)
    for row in rows:
        extra = len(row) + (2 if selected else 0)
        if selected and used + extra > limit:
            break
        selected.append(row)
        used += extra
    return prefix + ("; ".join(selected) if selected else "구조화된 행 확인")


#: A wide product/business table renders as a genuine Markdown table instead
#: of one flattened line.  「제품·서비스 표(단위: 백만원) [열 · 열 …]: 값 ·
#: 값 …」 read as a single unbroken clause once a table had more than a
#: couple of columns and rows (#47 결함 ⑨, #86 18) — a real header row plus
#: one row per line is what a reader can actually scan.  Twelve rows is a
#: display bound, not a retrieval one: every remaining row is named in an
#: 「외 N행」 trailer rather than silently dropped.
_MAX_PRODUCT_TABLE_ROWS = 12


def _bounded_table_block(prefix: str, names: list[str],
                         cells: list[list[str]], *, limit: int) -> str:
    """Render a source table as a Markdown header row plus up to 12 rows.

    ``limit`` remains a soft character budget on top of the 12-row cap, so a
    table with unusually wide cells still cannot dominate an entire answer.
    """

    heading = prefix.rstrip().rstrip(":").strip()
    header_line = "| " + " | ".join(_markdown_cell(name) for name in names) + " |"
    rule_line = "|" + "|".join(" --- " for _ in names) + "|"
    used = len(heading) + len(header_line) + len(rule_line)
    row_lines: list[str] = []
    for row in cells:
        if len(row_lines) >= _MAX_PRODUCT_TABLE_ROWS:
            break
        line = "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        if row_lines and used + len(line) > limit:
            break
        row_lines.append(line)
        used += len(line)
    if not row_lines:
        return f"{heading}: 구조화된 행 확인"
    remaining = len(cells) - len(row_lines)
    lines = [f"{heading}:", header_line, rule_line, *row_lines]
    if remaining > 0:
        lines.append(f"(외 {remaining}행)")
    return "\n".join(lines)


def _dedupe_revenue_columns(names: list[str], rows: list[list[str]]) -> list[str]:
    """Disambiguate a duplicated amount/ratio column pair as 「비중」.

    A merged 당기/전기 revenue table often repeats the same period label for
    both its amount and percentage sub-columns after header flattening (e.g.
    ``2025년 (당기)`` twice). A repeated name alone is not ratio evidence:
    require explicit percent cells, and retain geographically qualified roles.
    """

    if any(any(marker in name for marker in ("비율", "비중")) for name in names):
        return list(names)
    seen: dict[str, int] = {}
    result: list[str] = []
    qualified = _value_qualified_headers(names, rows)
    for index, name in enumerate(names):
        seen[name] = seen.get(name, 0) + 1
        result.append("비중" if seen[name] >= 2 and qualified[index].endswith("비중(%)")
                      else qualified[index])
    return result


def _investment_plan_digest(text: str | None, *, limit: int) -> str | None:
    """Project a plan-vs-actual investment table without recalculation."""

    body = _source_body(text)
    unit_match = re.search(r"\( ?단위\s*[:：]\s*([^)]+)\)", body)
    unit = unit_match.group(1).strip() if unit_match else "원문 표기"
    lines = body.splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = index
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            index += 1
        block = lines[start:index]
        if len(block) < 3:
            continue
        header = _table_cells(block[0])
        compact_header = [_compact_key(cell) for cell in header]
        if (not _is_table_rule(_table_cells(block[1]))
                or not any("투자계획" in cell for cell in compact_header)):
            continue
        rendered = []
        for row_line in block[2:]:
            row = _table_cells(row_line)
            if len(row) != len(header) or _is_table_rule(row) or not any(row):
                continue
            rendered.append(" / ".join(
                f"{_public_column_name(name)}: {value}"
                for name, value in zip(header, row)
                if name and value))
        if rendered:
            return _bounded_table_summary(
                f"투자계획 표(단위: {unit}): ", rendered, limit=limit)
    return None


def _investment_plan_table(
        text: str | None,
        ) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], str] | None:
    """Return one complete source-labelled investment-plan table.

    This is intentionally broader than :func:`_investment_rows`: some
    periodic reports disclose a plan by category (for example R&D, CAPEX and
    strategic investment) rather than by project, so it has no trustworthy
    target/purpose/period quartet.  We can still make that *same proved
    table* readable without inventing a project mapping or recomputing totals.
    """

    lines = _source_body(text).splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = index
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            index += 1
        block = lines[start:index]
        if len(block) < 3:
            continue
        header = _table_cells(block[0])
        if (not header or not _is_table_rule(_table_cells(block[1]))
                or not any("투자계획" in _compact_key(value)
                           for value in header)):
            continue
        rows = tuple(
            tuple(row)
            for row_line in block[2:]
            for row in (_table_cells(row_line),)
            if (len(row) == len(header) and not _is_table_rule(row)
                and any(row))
        )
        if rows:
            return (tuple(header), rows, _local_table_currency_unit(lines, start))
    return None


def _investment_summary(
        labels: tuple[str, ...], *, has_execution: bool, citation: str,
        ) -> str:
    """Give a reader the one finding needed before a long investment table.

    The summary only counts and quotes labels already present in the table;
    it deliberately does not add the plan amounts because adding them would
    either duplicate the table or require a cross-row total calculation.
    """

    visible_rows = tuple(
        value for value in labels
        if value and not re.search(r"^(?:합계|소계|총계)$", value.replace(" ", ""))
    )
    visible_labels = tuple(dict.fromkeys(visible_rows))
    if not visible_rows:
        opening = "공시 표에서 투자계획을 확인했습니다."
    elif len(visible_rows) == 1:
        opening = f"{visible_rows[0]} 투자계획이 확인됩니다."
    else:
        preview = "·".join(visible_labels[:3])
        preview_suffix = " 등" if len(visible_labels) > 3 else ""
        if len(visible_rows) == len(visible_labels):
            opening = (
                f"{preview}{preview_suffix} {len(visible_rows)}개 "
                "투자 항목의 계획이 확인됩니다."
            )
        else:
            # 같은 표에 이름이 같은 계획이 여러 법인·기간·목적으로
            # 공시될 수 있다. 표 행은 각각 독립된 계획 레코드이므로
            # 중복 제거한 이름 수를 전체 항목 수처럼 말하지 않는다.
            count = (
                f"계획 행 {len(visible_rows)}개"
                f"(대상명 기준 {len(visible_labels)}개)"
            )
            opening = f"{preview}{preview_suffix} {count}가 확인됩니다."
    distinction = (
        "아래 표는 계획 금액과 공시된 집행·실적을 구분했습니다."
        if has_execution else
        "아래 표는 공시된 계획 금액과 기간을 정리했습니다."
    )
    return f"핵심 요약: {opening} {distinction} (근거: {citation})"


def _investment_plan_summary(
        header: tuple[str, ...], rows: tuple[tuple[str, ...], ...], *,
        citation: str,
        ) -> str:
    """Build the summary for a category-style investment plan table."""

    compact = tuple(_compact_key(value) for value in header)
    label_index = next((
        index for index, value in enumerate(compact)
        if value in {"구분", "투자항목", "투자유형", "투자분류"}
    ), 0)
    labels = tuple(row[label_index] for row in rows if len(row) > label_index)
    has_execution = any(
        any(marker in value for marker in ("실적", "기지출", "집행", "기투자"))
        for value in compact)
    return _investment_summary(labels, has_execution=has_execution, citation=citation)


def _general_investment_narrative_digest(
        text: str | None, *, limit: int,
        ) -> str | None:
    """Project one broad plan-vs-actual investment table without inference."""

    body = _source_body(text)
    lines = body.splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = index
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            index += 1
        block = lines[start:index]
        if len(block) < 3 or not _is_table_rule(_table_cells(block[1])):
            continue
        header = _table_cells(block[0])
        compact_header = [_compact_key(cell) for cell in header]
        complete_project_roles = (
            len(_investment_header_positions(header))
            == len(_INVESTMENT_SLOT_HEADERS))
        roles = sum(any(marker in cell for cell in compact_header)
                    for marker in (
                        "기투자액", "향후투자액", "투자기대효과", "대상자산"))
        context = _compact_key("\n".join(lines[max(0, start - 12):start]))
        if (not complete_project_roles and roles < 2) or not any(
                marker in context for marker in ("투자계획", "투자현황")):
            continue
        unit = _local_table_currency_unit(lines, start) or "원문 표기"
        rendered: list[str] = []
        for row_line in block[2:]:
            row = _table_cells(row_line)
            if len(row) != len(header) or _is_table_rule(row) or not any(row):
                continue
            labels: list[str] = []
            counts: dict[str, int] = {}
            for name in header:
                base = name or "항목"
                counts[base] = counts.get(base, 0) + 1
                labels.append(
                    f"{base}{counts[base]}" if header.count(name) > 1 else base)
            rendered.append(" / ".join(
                f"{_public_column_name(name)}: {value}"
                for name, value in zip(labels, row)
                if value))
        if rendered:
            table = _bounded_table_summary(
                f"투자 현황 표(단위: {unit}): ", rendered, limit=limit)
            # A broad investment-status request explicitly allows evidence
            # outside the strict four-role plan row.  Preserve complete local
            # note sentences about acquisitions, facilities or excluded
            # investment amounts when they still fit; this is source text,
            # not an inferred additional project row.
            notes: list[str] = []
            for raw in lines[index:]:
                if raw.lstrip().startswith("|"):
                    continue
                clean = " ".join(raw.split()).strip()
                compact_note = _compact_key(clean)
                if (not clean or len(clean) > 420
                        or not any(marker in compact_note for marker in (
                            "취득", "인수", "시설", "설비", "제외", "집행",
                            "증설", "건설"))):
                    continue
                if clean not in notes:
                    notes.append(clean)
            suffix = " ".join(notes[:2])
            if suffix and len(table) + len(suffix) + 1 <= limit:
                table = f"{table} {suffix}"
            return table
    return None


def _investment_property_digest(text: str | None, *, limit: int) -> str | None:
    """Project a source-labelled investment-property balance table.

    The table is accepted only when the row itself is labelled
    ``투자부동산`` and its header exposes acquisition/carrying-value roles.
    Values are reproduced verbatim and never totalled or converted here.
    """

    lines = _source_body(text).splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = index
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            index += 1
        block = lines[start:index]
        if len(block) < 3 or not _is_table_rule(_table_cells(block[1])):
            continue
        header = _table_cells(block[0])
        compact_header = [_compact_key(cell) for cell in header]
        if (not any("장부금액" in cell for cell in compact_header)
                or not any(marker in " ".join(compact_header)
                           for marker in ("취득원가", "공정가치", "토지", "건물"))):
            continue
        rendered: list[str] = []
        for row_line in block[2:]:
            row = _table_cells(row_line)
            if (len(row) != len(header) or _is_table_rule(row)
                    or not row or "투자부동산" not in _compact_key(row[0])):
                continue
            labels = [name or "항목" for name in header]
            rendered.append(" / ".join(
                f"{_public_column_name(name)}: {value}"
                for name, value in zip(labels, row)
                if value))
        if rendered:
            unit = _local_table_currency_unit(lines, start) or "원문 표기"
            return _bounded_table_summary(
                f"투자부동산 표(단위: {unit}): ", rendered, limit=limit)
    return None


def _revenue_table_digest(text: str | None, *, limit: int) -> str | None:
    """Keep cited segment/product revenue rows as rows, never a guessed sum."""

    body = _source_body(text)
    unit_match = re.search(r"\( ?단위\s*[:：]\s*([^)]+)\)", body)
    unit = unit_match.group(1).strip() if unit_match else "원문 표기"
    lines = body.splitlines()
    index = 0
    summaries: list[tuple[int, str, list[str], list[list[str]]]] = []
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = index
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            index += 1
        block = lines[start:index]
        if len(block) < 3:
            continue
        header = _table_cells(block[0])
        compact = [_compact_key(cell) for cell in header]
        if not _is_table_rule(_table_cells(block[1])):
            continue
        identifier_header = any(
            any(marker in cell for marker in (
                "구분", "항목", "품목", "상품", "서비스", "사업", "부문", "유형"))
            for cell in compact)
        value_header = any(
            any(marker in cell for marker in (
                "매출", "수익", "손익", "금액", "비율", "합계"))
            or bool(re.search(
                r"(?:제[0-9]+(?:당|전|전전)?기|20[0-9]{2}년)", cell))
            for cell in compact)
        # Financial/insurance business tables frequently use the business
        # axes themselves as headers (WM, IB, 보험부문...) and put the revenue
        # concept in the first row.  Accept that structural form only when a
        # row identifier and an amount/total axis are present; an arbitrary
        # numeric table is not promoted to a revenue answer.
        row_labels = " ".join(
            " ".join(_table_cells(row_line)[:2])
            for row_line in block[2:6])
        revenue_rows = any(marker in _compact_key(row_labels) for marker in (
            "매출", "수익", "손익", "보험료", "이자", "수수료"))
        preceding = lines[max(0, start - 10):start]
        nearest_sales_heading = next((
            _compact_key(line) for line in reversed(preceding)
            if not line.lstrip().startswith("|")
            and "매출" in _compact_key(line)
            and "단위" not in _compact_key(line)
        ), "")
        period_columns = sum(bool(re.search(
            r"(?:제[0-9]+(?:당|전|전전)?기|20[0-9]{2}년)", cell))
            for cell in compact[1:])
        product_period_table = (
            "제품별매출" in nearest_sales_heading
            and bool(compact)
            and any(marker in compact[0] for marker in ("구분", "품목", "제품"))
            and period_columns >= 1
        )
        sales_period_table = (
            "매출실적" in nearest_sales_heading
            and bool(compact) and "구분" in compact[0]
            and period_columns >= 1)
        if (not identifier_header or not value_header
                or not (product_period_table or sales_period_table
                        or revenue_rows or any(
                    any(marker in cell for marker in (
                        "매출", "수익", "부문"))
                    for cell in compact))):
            continue
        kept_columns = [offset for offset, name in enumerate(header) if name]
        names = [_public_column_name(header[offset]) for offset in kept_columns]
        rows: list[list[str]] = []
        for row_line in block[2:]:
            row = _table_cells(row_line)
            if len(row) != len(header) or _is_table_rule(row) or not any(row):
                continue
            if not any(_NUM.search(value) for value in row):
                continue
            rows.append([row[offset] or "-" for offset in kept_columns])
        if rows:
            label = "제품별 매출 표" if product_period_table else "매출 표"
            context = _compact_key(" ".join(preceding))
            table_surface = _compact_key(" ".join((*header, row_labels)))
            priority = 0
            if any(marker in context + table_surface for marker in (
                    "영업수익", "보험수익", "보험료수익", "수입보험료",
                    "수익구조", "매출실적")):
                priority += 60
            if "부문" in table_surface:
                priority += 25
            if ("운용자산" in table_surface and "수익률" in table_surface
                    and not any(marker in table_surface for marker in (
                        "영업수익", "보험수익", "보험료수익", "수입보험료"))):
                # Asset balance/yield is not an issuer's revenue composition.
                # Do not let its generic ``운용수익`` label outrank the
                # insurance/segment revenue statement in the same section.
                continue
            summaries.append((priority, f"{label}(단위: {unit}): ", names, rows))

    # A sales section can contain both a segment table and a separately
    # labelled product-sales table.  They answer different requested axes, so
    # keep both inside one bounded digest.  Unlabelled numeric tables (sales
    # channels, regions, useful lives, etc.) remain excluded by the ontology
    # above.  Each survives as its own Markdown table (#100) — the single
    # flattened line this used to produce read as an unbroken slash-joined
    # clause once a revenue table had more than a couple of columns, exactly
    # the defect #89 already fixed for the product/service table.
    selected: list[str] = []
    remaining = limit
    for _priority, prefix, names, rows in sorted(
            summaries, key=lambda row: row[0], reverse=True):
        if remaining <= len(prefix):
            break
        material = [row for row in rows
                    if any(cell.strip() in ("합계", "소계") for cell in row)]
        ordered = [*material, *(row for row in rows if row not in material)]
        block_text = _bounded_table_block(
            prefix, _dedupe_revenue_columns(names, ordered), ordered, limit=remaining)
        selected.append(block_text)
        remaining -= len(block_text) + 2
    return "\n\n".join(selected) if selected else None


def _product_table_digest(
        text: str | None, *, limit: int,
        required_terms: tuple[str, ...] = (),
        business_only: bool = False,
        ) -> str | None:
    """Keep product/service identity, use and amount columns together.

    Product tables are authoritative because their header declares both the
    item identity (product/service/type) and at least one descriptive or
    amount axis.  This avoids the previous fallback that selected a nearby
    accounting-note table merely because it contained more short sentences.
    """

    body = _source_body(text)
    unit_match = re.search(r"\( ?단위\s*[:：]\s*([^)]+)\)", body)
    unit = unit_match.group(1).strip() if unit_match else "원문 표기"
    lines = body.splitlines()
    for start in range(max(0, len(lines) - 1)):
        if (not lines[start].lstrip().startswith("|")
                or not lines[start + 1].lstrip().startswith("|")):
            continue
        header = _table_cells(lines[start])
        rule = _table_cells(lines[start + 1])
        if not _is_table_rule(rule) or len(header) != len(rule):
            continue
        end = start + 2
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            if (end + 1 < len(lines)
                    and lines[end + 1].lstrip().startswith("|")
                    and _is_table_rule(_table_cells(lines[end + 1]))
                    and len(_table_cells(lines[end]))
                    == len(_table_cells(lines[end + 1]))):
                break
            end += 1
        block = lines[start:end]
        if len(block) < 3:
            continue
        compact = [_compact_key(cell) for cell in header]
        strong_identity_columns = [
            offset for offset, cell in enumerate(compact)
            if any(marker in cell for marker in (
                "품목", "제품", "서비스", "상품", "주요사업", "사업부문"))]
        weak_identity_columns = [
            offset for offset, cell in enumerate(compact)
            if "영업유형" in cell]
        detail_columns = [
            offset for offset, cell in enumerate(compact)
            if any(marker in cell for marker in (
                "용도", "개요", "설명", "내용", "매출", "수익", "금액",
                "비율", "구분"))
            or bool(re.search(r"(?:제[0-9]+기|20[0-9]{2}년)", cell))]
        substantive_detail_columns = [
            offset for offset, cell in enumerate(compact)
            if any(marker in cell for marker in (
                "용도", "개요", "설명", "내용", "매출", "수익", "금액",
                "비율"))
            or bool(re.search(r"(?:제[0-9]+기|20[0-9]{2}년)", cell))]
        # ``영업유형`` alone can describe a transaction channel rather than a
        # product.  It is accepted only with a substantive use/value axis;
        # explicit product/service headers remain authoritative with any
        # additional detail column.
        preceding_context = _compact_key(" ".join(lines[max(0, start - 10):start]))
        contextual_identity_columns = [
            offset for offset, cell in enumerate(compact)
            if cell == "구분" and any(marker in preceding_context for marker in (
                "주요제품", "제품에관한내용", "제품및서비스"))]
        identity_columns = (strong_identity_columns or weak_identity_columns
                            or contextual_identity_columns)
        if (not identity_columns or not detail_columns
                or (not strong_identity_columns and not substantive_detail_columns)):
            continue
        rows = []
        for row_line in block[2:]:
            row = _table_cells(row_line)
            if len(row) != len(header) or _is_table_rule(row) or not any(row):
                continue
            if all(_compact_key(row[offset]) == compact[offset]
                   for offset in identity_columns):
                # Viewer multi-row headers are serialized as ordinary rows.
                # Do not report those header labels as products.
                continue
            if not any(row[offset].strip() for offset in identity_columns):
                continue
            rows.append(row)
        if required_terms and rows:
            required = tuple(_compact_key(term).casefold() for term in required_terms)
            substantive = [
                row for row in rows
                if any(row[offset].strip() not in {"", "-", "소계", "합계", "계"}
                       for offset in identity_columns)
            ]
            seeds = [
                row for row in substantive
                if any(term in _compact_key(" ".join(row)).casefold()
                       for term in required)
            ]

            # Discover a literal bridge only when it occurs in at least two
            # distinct descriptive cells of a directly matched row.  Thus a
            # renewable row that says both ``태양광용`` and ``태양광산업`` may
            # bring in a second ``태양광 발전`` row, while a one-off material
            # such as ``폴리실리콘`` cannot pull an unrelated chemicals row.
            token_counts: dict[str, int] = {}
            generic = {
                "관련", "제품", "상품", "서비스", "사업", "부문", "소재",
                "주요", "기타", "소계", "합계", "원문", "표기",
            }

            def topical_tokens(value: str) -> set[str]:
                found = set()
                for token in re.findall(r"[가-힣A-Za-z]{2,}", value):
                    normalized = token.casefold()
                    for suffix in ("산업", "관련", "제품", "사업", "부문", "용"):
                        if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 2:
                            normalized = normalized[:-len(suffix)]
                            break
                    if len(normalized) >= 2 and normalized not in generic:
                        found.add(normalized)
                return found

            descriptive_columns = tuple(dict.fromkeys(
                (*identity_columns, *detail_columns,
                 *(offset for offset, value in enumerate(compact)
                   if any(marker in value for marker in ("사업", "부문", "회사"))))))
            for row in seeds:
                for offset in descriptive_columns:
                    for token in topical_tokens(row[offset]):
                        token_counts[token] = token_counts.get(token, 0) + 1
            bridges = {token for token, count in token_counts.items() if count >= 2}
            rows = [
                row for row in substantive
                if (row in seeds
                    or bool(bridges.intersection(topical_tokens(" ".join(row)))))
            ]

        display_columns = tuple(range(len(header)))
        if not required_terms and len(header) > 6:
            # A wide product table repeats the same amount/share axes for
            # several fiscal years.  A current filing summary needs every
            # material product row more than three years of duplicated
            # columns.  Retain all identity/description columns and only the
            # first source-declared period group; values stay verbatim.
            structural = tuple(
                offset for offset, value in enumerate(compact)
                if any(marker in value for marker in (
                    "사업", "부문", "회사", "매출유형", "품목", "제품",
                    "상품", "서비스", "용도", "내용", "구분", "생산판매개시")))
            value_offsets = tuple(
                offset for offset in range(len(header)) if offset not in structural)
            if value_offsets:
                first = header[value_offsets[0]]
                marker_match = re.search(r"(?:20[0-9]{2}년|제[0-9]+기)", first)
                if marker_match is not None:
                    marker = marker_match.group(0)
                    current_values = tuple(
                        offset for offset in value_offsets if marker in header[offset])
                else:
                    current_values = value_offsets[:2]
                display_columns = tuple(dict.fromkeys((*structural, *current_values)))
        if required_terms:
            # The question asks for a business axis, not every historical
            # value in the wide sales table.  Retaining structural identity,
            # product and use columns lets several relevant rows fit without
            # discarding the second operating axis at the character budget.
            display_columns = tuple(
                offset for offset, value in enumerate(compact)
                if any(marker in value for marker in (
                    "사업", "부문", "회사", "매출유형", "품목", "제품",
                    "상품", "서비스", "용도", "내용", "구분")))
            if len(rows) == 1:
                # A single matched row still has room for one quantitative
                # value.  Multiple matched axes prioritize identity/use so a
                # later row is not dropped merely because the table is wide.
                value_column = next((
                    offset for offset, value in enumerate(compact)
                    if offset not in display_columns
                    and any(marker in value for marker in (
                        "매출", "수익", "금액", "비율"))
                ), None)
                if value_column is not None:
                    display_columns = (*display_columns, value_column)
        omit_repeated_amounts = False
        if business_only and rows:
            # A plain business-description question needs the products, not
            # repeated numeric cells whose rowspan is no longer available in
            # Markdown. Do not infer totals; explicit revenue questions keep
            # their numeric columns unchanged.
            numeric_columns = [
                offset for offset in display_columns
                if all(re.fullmatch(r"[-+]?\(?[\d,]+(?:\.\d+)?\)?%?|-",
                                    row[offset].strip()) for row in rows)]
            groups: dict[str, list[list[str]]] = {}
            for row in rows:
                groups.setdefault(row[0], []).append(row)
            omit_repeated_amounts = len(numeric_columns) >= 2 and any(
                len(group) > 1 and len({tuple(row) for row in group}) > 1
                and all(len({row[offset] for row in group}) == 1
                        for offset in numeric_columns)
                for group in groups.values())
            if omit_repeated_amounts:
                display_columns = tuple(offset for offset in display_columns
                                        if offset not in numeric_columns)
        kept = [offset for offset in display_columns if header[offset]]
        names = [_public_column_name(header[offset]) for offset in kept]
        cells = [[row[offset] or "-" for offset in kept] for row in rows]
        cells = [row for row in cells
                 if not all(_compact_key(cell) in {"", "-", "합계", "소계", "계"}
                            for cell in row)]
        names = _value_qualified_headers(names, cells)
        if names and cells:
            has_numbers = any(_NUM.search(cell) for row in cells for cell in row)
            prefix = ("제품·서비스 표: " if omit_repeated_amounts or not has_numbers
                      else f"제품·서비스 표(단위: {unit}): ")
            return _bounded_table_block(prefix, names, cells, limit=limit)
    return None


_BUSINESS_TOPIC_STOPWORDS = {
    "관련", "주요", "사업", "사업내용", "사업부문", "영업", "부문", "제품",
    "서비스", "비교", "차이", "변화", "현황", "구조", "구성", "설명", "기준",
}


def _business_topic_terms(topic: str) -> tuple[str, ...]:
    """Return only explicit qualifiers outside the generic business axis.

    These literal terms are used to retain relevant rows from a source table;
    they are not aliases and do not expand the user's requested topic.
    """

    return tuple(dict.fromkeys(
        value for value in re.findall(r"[가-힣A-Za-z0-9]+", topic or "")
        if len(value) >= 2 and _compact_key(value).casefold() not in
        {_compact_key(stop).casefold() for stop in _BUSINESS_TOPIC_STOPWORDS}
    ))


#: 사업 «축» 을 한 문장에 나열하는 공시 문형.
#:
#: 「화공, 첨단산업, New Energy 로 나뉘어 있으며」·「토목부문, 건축부문,
#: 플랜트부문, 기타부문으로 나눌 수 있습니다」 처럼 **열거 + 구분 동사** 가 함께
#: 있는 문장이다. 회사 간 비교에서 이 문장이 가장 쓸모 있다 — 두 회사의 축을
#: 같은 층위에서 보여준다.
#:
#: 열거를 함께 요구하는 이유: 「사업본부는 서로 다른 사업과 용역을 제공하는
#: 전략적 사업단위입니다」 처럼 부문을 말하되 **이름을 대지 않는** 문장은 비교에
#: 쓸 수 없다.
_SEGMENT_AXIS_SENTENCE = re.compile(
    r"[가-힣A-Za-z0-9&()\s]{2,40}?(?:,|·)\s*(?P<tail>[가-힣A-Za-z0-9&()\s]{2,40}?)"
    r"\s*(?:으로|로)\s*(?:나뉘|나눌|구분|구성)")

#: 같은 «A, B, C 로 구성» 문형이라도 세는 대상이 사업 축이 아니라 **조직·법인·
#: 거점** 이면 비교에 쓸 수 없다.  「232개의 종속기업으로 구성된 글로벌 전자
#: 기업입니다」·「수원, 구미, 광주 … 사업장 등으로 구성되어 있으며」 가 그런
#: 문장이고, 무엇을 하는 회사인지는 한 마디도 말하지 않는다.
_NON_AXIS_ENUMERATION_HEAD = re.compile(
    r"(?:종속\s*기업|계열\s*회사|계열사|자회사|관계\s*기업|법인|사업장|지점|"
    r"공장|거점|지역\s*총괄|개사|개\s*회사)\s*(?:등)?$")


def _names_segment_axis(sentence: str) -> bool:
    """축 이름을 실제로 나열하는 문장인가.

    열거 + 구분 동사만으로는 부족하다 — 조직도나 거점을 세는 문장이 같은 문형을
    쓴다.  ``으로/로`` 바로 앞에 오는 말이 사업 축인지 조직 단위인지로 가른다.
    """

    for match in _SEGMENT_AXIS_SENTENCE.finditer(sentence):
        tail = " ".join(match.group("tail").split())
        if not _NON_AXIS_ENUMERATION_HEAD.search(tail):
            return True
    return False


_SOURCE_BUSINESS_SUBHEADING = re.compile(
    r"(?:^|(?<=[.!?]))\s*[가-하]\s*[.)]\s*"
    r"([가-힣A-Za-z0-9&/·ㆍ -]{1,24}?부문)(?=\s|$)")
_NON_AXIS_SUBHEADING = re.compile(
    r"(?:종속|자회사|조직|지원|관리|사업장|지역|매출|수익|실적)")


def _business_axis_heading_digest(body: str, *, limit: int) -> str:
    """Project repeated source-owned business subheadings into one axis list.

    Some viewer sections flatten ``가. 토목부문`` through ``라. 뉴에너지
    부문`` into one paragraph.  Sentence ranking then sees only long market
    commentary and misses the actual comparison axis.  Returning the literal
    heading labels, in source order, preserves one section/one citation and
    does not infer or rename a business category.
    """

    headings: list[str] = []
    for match in _SOURCE_BUSINESS_SUBHEADING.finditer(body or ""):
        heading = " ".join(match.group(1).split()).strip(" ,;:/")
        if (_NON_AXIS_SUBHEADING.search(heading)
                or heading in {"사업부문", "영업부문"}
                or heading in headings):
            continue
        headings.append(heading)
    if not 2 <= len(headings) <= 8:
        return ""
    digest = "; ".join(headings)
    return digest if len(digest) <= limit else ""


def _source_service_definitions(text: str, shown: str) -> tuple[str, ...]:
    """Copy concise service definitions only when the same source states them."""

    rules = (
        ("SI", r"SI\s*서비스[^.!?]{0,140}?(정보(?:화)?시스템의\s*구축\s*및\s*통합)"),
        ("SM", r"SM\s*서비스[^.!?]{0,140}?(IT\s*시스템의\s*운영\s*및\s*관리)"),
        ("EOL", r"End-of-Line\s*\(([^)]{2,65})\)"),
        ("CMI", r"CMI\s*\(([^)]{2,65})\)"),
    )
    definitions = []
    for term, pattern in rules:
        if not re.search(rf"\b{term}\b", shown):
            continue
        match = re.search(pattern, text)
        if match and re.search(r"[가-힣]", match.group(1)):
            definitions.append(f"{term}: {' '.join(match.group(1).split())}")
    return tuple(definitions)


def _business_table_axis_digest(text: str | None, *, limit: int) -> str:
    """Project one complete, source-labelled operating axis from a table.

    A section selected for a broad business topic can be the issuer's product
    table rather than its prose overview.  Selecting the first high-scoring
    sentence from that section then leaves only one product family in the
    answer even though the same verified block has the complete operating
    roster.  Keep a bounded list only when one declared identity column has
    two to eight distinct material values.  Generic geographic ``구분``
    columns and total/reconciliation rows are not business axes.
    """

    lines = _source_body(text).splitlines()
    candidates: list[tuple[int, int, int, str, tuple[str, ...]]] = []
    geography = {
        "국내", "해외", "내수", "수출", "한국", "미주", "유럽", "아시아",
    }
    for start in range(max(0, len(lines) - 1)):
        if (not lines[start].lstrip().startswith("|")
                or not lines[start + 1].lstrip().startswith("|")):
            continue
        header = _table_cells(lines[start])
        rule = _table_cells(lines[start + 1])
        if not _is_table_rule(rule) or len(header) != len(rule):
            continue
        end = start + 2
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            if (end + 1 < len(lines)
                    and lines[end + 1].lstrip().startswith("|")
                    and _is_table_rule(_table_cells(lines[end + 1]))):
                break
            end += 1
        rows = [_table_cells(line) for line in lines[start + 2:end]]
        rows = [row for row in rows if len(row) == len(header)
                and not _is_table_rule(row)]
        if not rows:
            continue
        preceding = _compact_key(" ".join(lines[max(0, start - 8):start]))
        for offset, raw_header in enumerate(header):
            column = _compact_key(raw_header)
            # Viewer note tables can have a footnote sentence containing
            # "사업부문" as their header. That is not a business identity
            # column, even when the following two notes look like rows.
            if (len(raw_header) > 48
                    or re.match(r"\s*(?:※|\(?주\s*\d+\s*\)?)", raw_header)
                    or any(marker in column for marker in ("참조", "되었음", "작성되었"))):
                continue
            if column == "사업" or any(marker in column for marker in ("사업부문", "영업부문", "사업영역")):
                score = 120
            elif column == "구분":
                score = 80
            elif any(marker in column for marker in (
                    "주요사업", "품목", "제품군", "제품명", "서비스명", "주요제품")):
                score = 60
            else:
                continue
            values: list[str] = []
            for row in rows:
                value = " ".join(row[offset].split()).strip(" -:;")
                compact = _compact_key(value)
                if (not value or compact == column or compact == "계"
                        or re.fullmatch(
                            r"\d*(?:매출액|영업손익|영업이익|당기순이익|자산|부채|자본)"
                            r"(?:\*?\d+)?", compact)
                        or any(marker in compact for marker in (
                            "합계", "총계", "소계", "순매출", "내부거래",
                            "조정", "제거"))):
                    continue
                if value not in values:
                    values.append(value)
            if not 2 <= len(values) <= 8:
                continue
            if all(_compact_key(value) in geography for value in values):
                continue
            if all(re.fullmatch(
                    r"(?:20\d{2}년|제\d+기|당기|전기|전전기)"
                    r"[\d년월일분반기말초()~./-]*", _compact_key(value))
                    for value in values):
                # A transposed period column is not a list of businesses.
                continue
            if any(marker in preceding for marker in (
                    "주요제품", "제품및서비스", "사업개요", "매출현황", "매출실적")):
                score += 30
            if any(marker in preceding for marker in (
                    "가격변동", "판매경로", "매출처", "수주상황")):
                score -= 80
            candidates.append(
                (score, -start, -offset, raw_header.strip(), tuple(values)))
    if not candidates:
        return ""
    _score, table_order, column_order, header, values = max(
        candidates, key=lambda item: item[:3])
    # 정본 ``path`` 는 표 계층을 ``상위 > 하위`` 로 잇고 단위 안내 칸까지 이름에
    # 넣는다. 그대로 실으면 ``((단위: 천원) > 사업부문)`` 처럼 괄호가 겹치고
    # 내부 좌표가 사용자에게 나간다.
    parts = [f"공시 표의 주요 사업 축({_public_column_name(header)}): "
             + "·".join(values)]
    # A business table can declare both a segment column and a material
    # product/business column.  Keeping only the highest-scored column makes a
    # literal shared product disappear from the displayed evidence and hence
    # from an otherwise auditable company comparison.  Preserve other bounded
    # identity columns from that same table, in source-column order, as long as
    # the complete digest remains within the caller's display budget.
    supplemental = sorted(
        (item for item in candidates
         if item[1] == table_order and item[2] != column_order),
        key=lambda item: -item[2])
    for _candidate_score, _table, _column, extra_header, extra_values in supplemental:
        part = f"{extra_header}: " + "·".join(extra_values)
        candidate = "; ".join((*parts, part))
        if len(candidate) <= limit:
            parts.append(part)
    for definition in _source_service_definitions(_source_body(text), "; ".join(parts)):
        if len("; ".join((*parts, definition))) <= limit:
            parts.append(definition)
    return "; ".join(parts) if len(parts[0]) <= limit else ""


#: 원문이 절 제목을 인용문 맨 앞에 ``<…>`` 로 되풀이하는 자리
#: (``<주요 제품 및 서비스> 당사가 영위하는…``, DEV-NAR-013). 답 라벨이 이미
#: 그 절을 ``(주요 제품 및 서비스)`` 로 가리키므로 같은 말이면 중복이다.
_LEADING_BRACKET_TITLE = re.compile(r"^\s*<([^<>]{1,40})>\s*")

#: 표 아래 원문 주기(``※ …``). 표가 함께 인용되지 않으면 무엇에 붙는
#: 주기인지 알 수 없다(EDGE-011).
_TRAILING_SOURCE_NOTE = re.compile(r"(?:^|\s)※\s*(?P<note>[^※]+)$")
#: 이 낱말이 주기에 있으면 값의 해석에 필요한 주기(단위·기준 변경·비교
#: 가능성)이므로 표가 없어도 남긴다 — 판단이 어려우면 남기는 쪽(보수적).
_NOTE_INTERPRETIVE_MARKERS = ("단위", "기준", "비교 가능성", "비교가능")


def _strip_duplicate_bracket_title(text: str, label: str) -> str:
    """Drop a leading ``<절 제목>`` when it only repeats the field label.

    라벨과 다른 절 제목이면 그 자체가 정보이므로 원문 그대로 둔다.
    """

    match = _LEADING_BRACKET_TITLE.match(text or "")
    if match is None:
        return text
    title_key = _compact_key(match.group(1))
    label_key = _compact_key(label or "")
    if title_key and label_key and (
            title_key in label_key or label_key in title_key):
        return text[match.end():]
    return text


def _trim_or_drop_source_note(sentence: str) -> str | None:
    """Remove an un-anchored ``※`` 주기 from one candidate excerpt sentence.

    A quote that is *only* the note has no table to point at, so it is
    dropped entirely (``None``).  A note trailing real content is trimmed
    off unless it states the unit/basis change or comparability that the
    preceding value needs to be read correctly, in which case the whole
    sentence is kept untouched (EDGE-011 — 「비교 가능성」 조치는 남긴다).
    """

    stripped = (sentence or "").strip()
    if not stripped:
        return sentence
    if stripped.startswith("※"):
        return None
    match = _TRAILING_SOURCE_NOTE.search(stripped)
    if match is None:
        return sentence
    note = match.group("note").strip()
    if any(marker in note for marker in _NOTE_INTERPRETIVE_MARKERS):
        return sentence
    return stripped[:match.start()].rstrip()


def _business_sentence_digest(
        text: str | None, *, limit: int,
        required_terms: tuple[str, ...] = (),
        suppress_business_noise: bool = False,
        topic: str = "",
        ) -> str:
    """Prefer current operating substance over a company's origin story.

    A broad business section often starts with one historical sentence and
    then names current technology, products and operating activities.  Rank
    complete source sentences by those generic signals and keep their original
    wording; never synthesize or character-slice a new assertion.
    """

    ordinary_lines = [
        " ".join(line.split()) for line in _source_body(text).splitlines()
        if line.strip() and not line.lstrip().startswith("|")]
    body = " ".join((*ordinary_lines, *_single_cell_prose_lines(text)))
    # Consolidated overviews often append independently headed affiliate
    # profiles after the issuer paragraph.  Once that explicit structural
    # boundary appears, later sentences cannot replace the selected issuer's
    # own business digest.
    body = re.split(
        r"\[(?:주요\s*)?(?:종속회사|자회사)[^]]*\]", body, maxsplit=1)[0]
    # 원문이 절 제목을 인용문 맨 앞에 <…>로 되풀이할 때, 답 라벨이 이미 같은
    # 절을 가리키면 중복이다(issue #119 DEV-NAR-013).
    body = _strip_duplicate_bracket_title(body, topic)
    business_noise = (
        "감사보고서", "재발행", "회계처리", "회계정책", "인식시점",
        "총액법", "순액법", "판매가격", "가격변동", "가격 변동",
        "재무제표의 재작성", "재무제표 재작성", "가격추이", "가격책정",
        "연결재무제표 주석", "출고가격", "공연매출", "단가가",
        "단가의 산정", "평균단가", "판매단가", "기준가", "가격산출",
        "산출방법", "판가는", "가격 산정", "단가 산정",
        # Cost, sustainability and shareholder-communication commentary sits
        # in the same section as the business description and outranks it
        # here, because ``판매관리비`` contains ``판매`` and an ESG or
        # governance sentence often carries ``핵심``.  Neither states what the
        # company does, so a business-axis digest must not select them.
        "판매관리비", "판관비", "경상경비", "탄소배출", "온실가스",
        "지속가능경영", "기후 리스크", "기후리스크", "감축방안",
        "주주가치", "주주환원", "모든 주주", "투명한 정보",
        "유동성 안정화", "재무 건전성", "미분양",
        # 주요 사업 질문에서 판매 절차·결제 조건은 무엇을 영위하는지보다
        # 유통 방식에 관한 설명이다. 제품/판매 질문 자체에서는 이 옵션을
        # 켜지 않으므로 그 답변의 정보는 손대지 않는다.
        "판매경로", "판매 경로", "판매방법", "판매 방법",
        "판매전략", "판매 전략", "결제조건", "결제 조건",
        "자기주식", "자사주", "배당정책", "배당 정책",
        "운용실적", "실적개선", "실적 개선",
    )

    def has_business_noise(sentence: str) -> bool:
        compact = _compact_key(sentence)
        return (any(marker in sentence for marker in business_noise)
                or bool(re.search(
                    r"(?:가격|단가)[가-힣A-Za-z0-9]{0,100}"
                    r"(?:결정|책정|상이|유동)", compact))
                or bool(re.search(
                    r"(?:입찰강도|타겟팅)[가-힣A-Za-z0-9]{0,100}가격",
                    compact)))
    sentences = []
    for sentence in re.split(
            r"(?<=[.!?])(?:\s+|(?=\(\d+\)|[가-힣A-Za-z]))", body):
        clean = sentence.strip()
        if suppress_business_noise:
            # Viewer headings can be glued to the following subject, e.g.
            # ``(2)콘텐츠 부문게임부문은``.  Drop only the numbered outer
            # heading and retain the source sentence ``게임부문은 ...``.
            clean = re.sub(
                r"^\(\d+\)\s*[가-힣A-Za-z0-9 ]{1,24}?\s*부문"
                r"(?=[가-힣A-Za-z0-9 ]{1,24}?부문(?:은|는))",
                "", clean)
        # 표 없이 옮겨 온 「※ …」 원문 주기는 무엇에 붙는지 알 수 없다
        # (EDGE-011). 값 해석에 필요한 주기(단위·기준 변경·비교 가능성)는
        # 그대로 두고, 그 외에는 떼거나(뒤에 붙은 경우) 후보에서 뺀다(단독인
        # 경우).
        trimmed = _trim_or_drop_source_note(clean)
        if trimmed is None:
            continue
        clean = trimmed
        # 원문 표의 각주 표시는 답에 그 각주 자체가 실리지 않아 가리킬 곳이
        # 없다(issue #119 — (주N) 참조는 본문 인용에서만 뗀다).
        clean = _FOOTNOTE_MARK.sub("", clean).strip()
        # 문장 맨 앞의 원문 항목 번호(가./나)/1)/(1))는 형제 항목 없이는
        # 뜻이 없다(issue #119). 문장 중간의 같은 모양은 손대지 않는다.
        clean = strip_leading_item_number(clean).strip()
        if (_is_business_cross_reference(clean)
                or "…" in clean or "..." in clean
                or re.search(r"['’]\s*[가-하]\.$", clean)
                or re.search(r"['’‘\"]\s*(?:[IVX]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)\.$", clean)
                or re.search(r"(?:참고사항|목차).*['’](?:의|부터|까지)", clean)):
            # A viewer's cut sentence/reference is not an operating fact.
            continue
        if re.match(r"사업\s*분야로는", clean):
            # The sentence is often a continuation inside a named segment,
            # not a second company-wide roster.  Preserve the source heading
            # when ranking lifts that sentence out of its original section.
            position = body.find(clean)
            headings = (list(_SOURCE_BUSINESS_SUBHEADING.finditer(body[:position]))
                        if position >= 0 else [])
            if headings:
                clean = f"{headings[-1].group(1).strip()}: {clean}"
        if (12 <= len(clean) <= limit
                # These sentences describe evidence preparation, accounting
                # or price mechanics rather than the operating business.
                and (not suppress_business_noise
                     or not has_business_noise(clean))):
            sentences.append(clean)
    if required_terms:
        compact_terms = tuple(
            _compact_key(term).casefold() for term in required_terms)
        sentences = [
            sentence for sentence in sentences
            if any(term in _compact_key(sentence).casefold()
                   for term in compact_terms)
        ]
    if not sentences:
        return ""

    operating = (
        "핵심", "주요", "현재", "제조", "생산", "판매", "공급", "제공", "영위",
        "연구개발", "제품개발", "상용화", "보유기술", "사업 역량", "파이프라인",
        "제품", "서비스",
    )
    high_signal = ("상용화", "주요매출", "파이프라인")
    # ``핵심`` only signals the business axis when it qualifies a business
    # noun.  ``핵심입지``/``핵심 경쟁요소`` are review prose and previously
    # outranked the sentence that actually states the operating segments.
    core_business = re.compile(
        r"핵심\s*(?:사업|제품|기술|역량|파이프라인|브랜드|서비스|소재|부품)")
    historical = ("창업", "설립", "준공", "하여 왔", "해 왔")
    scored = []
    for index, sentence in enumerate(sentences):
        compact = _compact_key(sentence)
        score = 2 * sum(marker in compact for marker in operating)
        score += 4 * sum(marker in compact for marker in high_signal)
        score += 4 * len(core_business.findall(compact))
        score += 8 * ("주력으로" in compact)
        # 축을 이름으로 나열하는 문장은 비교에서 가장 쓸모 있으므로 다른 어떤
        # 신호보다 앞선다.  제품표를 대신할 만한 문장은 사실상 이것뿐이다.
        score += 8 * _names_segment_axis(sentence)
        score += min(3, len(re.findall(r"[A-Z][A-Z0-9-]{1,}|[가-힣A-Za-z]+(?:주|액|제|란)", sentence)))
        score -= 2 * sum(marker in compact for marker in historical)
        # A continuation is a poor standalone overview when the source also
        # contains a complete main-business sentence. Preserve its wording if
        # it is ultimately needed, but prefer independently readable content.
        score -= 8 * bool(re.match(r"^(?:이외에도|그밖에도|그 밖에도|또한|이와 함께)", sentence))
        if sentence in [row[3] for row in scored]:
            continue
        scored.append((score, -index, index, sentence))
    ranked = sorted(scored, reverse=True)
    selected: list[tuple[int, str]] = []
    used = 0
    for _score, _reverse_index, index, sentence in ranked:
        extra = len(sentence) + (1 if selected else 0)
        if used + extra > limit:
            continue
        selected.append((index, sentence))
        used += extra
        if len(selected) >= 2:
            break
    return " ".join(sentence for _index, sentence in sorted(selected))


def _qualified_business_evidence_digest(
        text: str | None, *, limit: int,
        required_terms: tuple[str, ...],
        ) -> str:
    """Select only complete source lines/sentences carrying a qualified axis.

    PDF-derived sections may have no sentence punctuation across several
    unrelated headings.  Joining all lines first then selecting one "sentence"
    lets management-strategy prose leak into a request for asset management or
    investment finance.  This keeps only structural lines which literally
    contain a requested qualifier and an operating/segment marker.
    """

    requested_axes: list[tuple[str, tuple[str, ...]]] = []
    for term in required_terms:
        surface = _compact_key(term).casefold()
        aliases = _QUALIFIED_BUSINESS_AXIS_ALIASES.get(surface, (surface,))
        axis = tuple(dict.fromkeys(_compact_key(alias).casefold()
                                   for alias in aliases))
        if axis not in [markers for _label, markers in requested_axes]:
            requested_axes.append((term, axis))
    # Viewer text can wrap a Korean sentence inside a morpheme
    # (``기록하였습`` / ``니다.``).  Rebuild only until an explicit sentence
    # terminator; headings followed by a result line are separated with one
    # space, while Hangul/punctuation continuations are joined without one.
    logical_lines: list[str] = []
    buffer = ""
    for raw_line in _source_body(text).splitlines():
        line = " ".join(raw_line.split())
        if not line or raw_line.lstrip().startswith("|"):
            continue
        if buffer:
            joiner = ""
            if not (re.search(r"[가-힣]$", buffer)
                    and re.match(r"(?:[가-힣]|[,.!?])", line)):
                joiner = " "
            buffer += joiner + line
        else:
            buffer = line
        if re.search(r"[.!?]$", line):
            logical_lines.append(buffer)
            buffer = ""
        elif len(buffer) > 900:
            logical_lines.append(buffer)
            buffer = ""
    if buffer:
        logical_lines.append(buffer)

    fragments: list[str] = []
    for logical_line in logical_lines:
        for raw in re.split(
                r"(?<=[.!?])(?:\s+|(?=[가-힣A-Za-z㈜]))", logical_line):
            clean = raw.strip(" \t,;:")
            compact = _compact_key(clean).casefold()
            if (not clean or len(clean) > 520
                    or not any(marker in compact
                               for _label, markers in requested_axes
                               for marker in markers)
                    or not any(marker in compact for marker in (
                        "사업", "부문", "업무", "영위", "운용", "매매",
                        "중개", "인수", "투자", "영업", "수익"))):
                continue
            if clean not in fragments:
                fragments.append(clean)
    selected: list[str] = []
    used = 0
    missing: list[str] = []
    for label, aliases in requested_axes:
        candidates = [
            fragment for fragment in fragments
            if any(alias in _compact_key(fragment).casefold()
                   for alias in aliases)
        ]
        # Prefer an operating result or concrete activity over a bare roster
        # sentence/heading.  Selection remains source-literal and every chosen
        # fragment comes from the already round-tripped section.
        def candidate_key(fragment: str) -> tuple[int, int, bool, int]:
            compact = _compact_key(fragment)
            # Axis names themselves are not activity evidence: for example,
            # ``자산운용`` contains ``운용`` even in a bare roster and
            # must not outrank a requested 기업금융 result sentence.  Remove
            # every requested alias before scoring operating verbs, while
            # retaining the source fragment unchanged for display.
            activity = compact.casefold()
            for _axis_label, markers in requested_axes:
                for marker in markers:
                    activity = activity.replace(marker, "")
            operating = sum(marker in activity for marker in (
                "핵심사업", "주요사업", "사업을영위", "제조", "생산",
                "공급", "제공", "운영", "인수", "중개", "자문", "운용"))
            operating_result = sum(marker in compact for marker in (
                "영업이익", "영업수익"))
            return operating, operating_result, bool(_NUM.search(fragment)), len(fragment)

        candidates.sort(key=candidate_key, reverse=True)
        chosen = next((fragment for fragment in candidates
                       if fragment not in selected), None)
        if chosen is None:
            missing.append(label)
            continue
        extra = len(chosen) + (1 if selected else 0)
        if used + extra > limit:
            missing.append(label)
            continue
        selected.append(chosen)
        used += extra
    if missing and selected:
        notice = ("«" + "·".join(missing)
                  + "»에 대한 독립적인 사업 설명은 표시 범위에서 추출하지 못했습니다.")
        if used + len(notice) + (1 if selected else 0) <= limit:
            selected.append(notice)
    return " ".join(selected)


def _sentences_of(digest: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=다\.)\s*", digest) if s.strip()]


def _added_sentences(
        digest: str, earlier: "list[tuple[str, str]] | tuple",
        ) -> tuple[str, str] | None:
    """앞선 셀의 인용을 그대로 담고 문장만 덧붙었는가.

    같은 회사의 두 주제가 거의 같은 문단으로 답해지는 일이 있다.  전문을 다시
    찍으면 두 문단이 나란히 놓여 **무엇이 다른지가 오히려 묻힌다.**  덧붙은
    문장만 보여주면 그 차이가 답이 된다.  문장은 원문 그대로 두고 어느 셀에
    더해지는지 밝히므로, 인용도 출처도 유지된다.
    """

    current = _sentences_of(digest)
    if len(current) < 2:
        return None
    for cell_id, previous in earlier:
        before = _sentences_of(previous)
        if len(before) < 1 or len(before) >= len(current):
            continue
        if not set(before).issubset(set(current)):
            continue
        added = [s for s in current if s not in set(before)]
        # 덧붙은 양이 원래 인용보다 많으면 «더해진다» 고 말할 수 없다.
        if not added or len(" ".join(added)) > len(" ".join(before)):
            continue
        return cell_id, " ".join(added)
    return None


def _compact_narrative_text(
        text: str | None, *, limit: int = 420, label: str = "") -> str:
    """Use complete source sentences/lines, never a character-sliced dump."""

    body = _source_body(text)
    if not body:
        return ""
    # 원문이 절 제목을 인용문 맨 앞에 <…>로 되풀이할 때, 답 라벨이 이미 같은
    # 절을 가리키면 중복이다(issue #119 DEV-NAR-013).
    body = _strip_duplicate_bracket_title(body, label)
    # 평탄화된 부문/계열사 표는 문장 부호가 없어 아래 문장 선택기가 **전체를
    # 문장 하나로** 보고 그대로 통과시킨다.  그러면 표 뒤에 이어붙은 다른 축의
    # 표(조달금리 등)까지 답변에 실린다 — `EDGE-014` 의 KB금융 셀이 그랬다.
    # 부문 표임을 먼저 확인하고, 맞으면 그쪽이 `(단위 …)` 에서 잘라 준다.
    segments = _flattened_segment_digest(body, limit=limit)
    if segments:
        return segments
    paragraphs = [" ".join(line.split()) for line in body.splitlines()
                  if line.strip() and not line.lstrip().startswith("|")]
    for paragraph in paragraphs:
        # Keep only whole sentences.  Korean disclosures often have a long
        # heading followed by short declarative sentences; selecting the first
        # bounded sequence avoids forwarding a whole section as an answer.
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
                     if sentence.strip()]
        selected: list[str] = []
        for sentence in sentences:
            # 표 없이 옮겨 온 「※ …」 주기는 무엇에 붙는지 알 수 없다
            # (EDGE-011). 값 해석에 필요하면(단위·기준 변경·비교 가능성)
            # 그대로 두고, 그 외에는 떼거나 후보에서 뺀다.
            trimmed = _trim_or_drop_source_note(sentence)
            if trimmed is None:
                continue
            sentence = _FOOTNOTE_MARK.sub("", trimmed).strip()
            # 문장 맨 앞의 원문 항목 번호(가./나)/1)/(1))는 형제 항목 없이는
            # 뜻이 없다(issue #119). 문장 중간의 같은 모양은 손대지 않는다.
            sentence = strip_leading_item_number(sentence).strip()
            if not sentence:
                continue
            if len(sentence) > limit:
                continue
            if sum(len(part) + 1 for part in selected) + len(sentence) > limit:
                break
            selected.append(sentence)
            if len(selected) >= 2:
                break
        if selected:
            return " ".join(selected)
    # A flattened segment table carries no sentence punctuation at all, so the
    # sentence selector above can never fit it.  Its own repeating ``…부문``
    # headings are a source-owned boundary, so whole segments can be kept
    # without inventing a summary.  Everything from the next table's unit
    # marker onward belongs to a different axis and is dropped.
    segments = _flattened_segment_digest(body, limit=limit)
    if segments:
        return segments
    # If the source contains no safe sentence/line under the bounded display
    # budget, do not cut it mid-sentence and pretend it is a summary.
    return "구조 블록이 장문 단일 문장이라 내용을 임의로 축약하지 않았습니다."


def _metric_sentences(text: str | None, *, limit: int) -> str:
    """Select complete source sentences that actually carry revenue values."""

    body = " ".join(
        " ".join(line.split()) for line in _source_body(text).splitlines()
        if line.strip() and not line.lstrip().startswith("|"))
    sentences = [part.strip() for part in re.split(
        r"(?<=[.!?])(?:\s+|(?=20[0-9]{2}년|부문별|아울러))", body)
        if part.strip()]
    selected = []
    for sentence in sentences:
        leading = sentence[:220]
        if (not _NUM.search(sentence)
                or not any(marker in leading for marker in (
                    "매출", "수익", "보험료", "운용", "부문", "비중"))):
            continue
        if len(sentence) > limit:
            continue
        if selected and sum(len(item) + 1 for item in selected) + len(sentence) > limit:
            break
        selected.append(sentence)
        if len(selected) >= 4:
            break
    return " ".join(selected)


_SEGMENT_HEADING = re.compile(r"(?<![가-힣])([가-힣A-Za-z0-9]{2,12}부문)(?![가-힣])")
_TABLE_UNIT_MARKER = re.compile(r"\(\s*단위\s*[:：]")
# Segment axes this small are rendered whole rather than ranked.
_WHOLE_AXIS_ROWS = 6


#: 공시 원문의 항목 좌표(``가.``·``마.``·``3.``). 사용자에게는 뜻이 없다.
_SOURCE_ITEM_MARKER = re.compile(r"^(?:[가-하]|\d{1,2})\.\s*")


def _public_table_caption(label: str, header) -> str:
    """표 위 캡션에서 원문 항목 번호와 열 이름 나열을 뺀다.

    ``마. 설비 투자 현황 및 계획 … (투자대상, 목적, 금액, 기간)`` 에서 ``마.`` 는
    공시 원문의 항목 좌표이고, 괄호 안 목록은 **바로 아래 표 머리가 이미 보여준다.**
    둘 다 캡션에 남으면 같은 정보를 두 줄 연속으로 읽게 된다.

    괄호를 뗄 때는 그 안의 항목이 실제로 표의 열인지 확인한다 — 열 이름이 아닌
    부연이면 정보이므로 남긴다.
    """

    text = _SOURCE_ITEM_MARKER.sub("", " ".join((label or "").split()))
    match = re.search(r"\s*\(([^()]+)\)\s*$", text)
    if match is None:
        return text
    listed = [item.strip() for item in re.split(r"[,·/]", match.group(1))
              if item.strip()]
    columns = [_compact_key(str(name)) for name in header]
    if listed and all(
            any(_compact_key(item) in column or column in _compact_key(item)
                for column in columns if column)
            for item in listed):
        return text[:match.start()].rstrip()
    return text


#: 연결어미로 끝나는 포착. 명사구가 아니라 다음 절로 이어지던 문장 조각이다.
_CLAUSE_TAIL = re.compile(r"(?:이며|하며|이고|하고|이라|되며|되고|으로|하여)$")
#: 원문 표의 각주 표시(``(*주1)``·``(주2)``).
_FOOTNOTE_MARK = re.compile(r"\(\s*\*?\s*주\s*\d+\s*\)")


def _public_column_name(name: str) -> str:
    """공시 표의 열 이름을 사용자용 표면으로 줄인다.

    정본 ``path`` 는 표 안의 계층을 ``상위 > 하위`` 로 잇고, 단위 안내 칸이
    헤더로 잡히면 ``(단위: 백만원) > WM`` 같은 이름이 만들어진다. 그대로 답변에
    실으면 ``(단위: 백만원) > WM: 974,366`` 처럼 내부 좌표가 사용자에게 그대로
    나가고, 같은 단위 문구가 열마다 반복된다. 표시 전용 축약이며 값은 건드리지
    않는다 — 단위는 표 머리말이 이미 한 번 밝힌다.

    **계층 자체는 남긴다.** ``기투자액 > 2030년`` 의 상위 칸은 계획액과 기지출을
    가르는 실제 의미라, 말단만 남기면 그 구분이 사라진다. 지우는 것은 단위 안내
    칸뿐이다.
    """

    kept = [
        segment.strip() for segment in name.split(" > ")
        if segment.strip()
        and re.fullmatch(r"\(\s*단위[^)]*\)", segment.strip()) is None
    ]
    if not kept:
        return name.strip()
    # ``>`` 는 기계 문법이다. 남은 계층은 ``상위(하위)`` 로 적어 계층 의미를
    # 지키면서 사용자가 읽을 수 있는 표면으로 만든다.
    head, *rest = kept
    if not rest:
        return head
    tail = "·".join(rest)
    # 상위가 이미 괄호로 끝나면 괄호를 겹치지 않고 띄어 쓴다
    # (``제7기(당기)(2024년 1~12월)(매출액)`` 같은 중첩을 만들지 않는다).
    return f"{head} {tail}" if head.endswith(")") else f"{head}({tail})"


def _flattened_segment_digest(body: str, *, limit: int) -> str:
    """Keep whole ``…부문`` segments from a flattened segment table.

    Some filings render a segment/affiliate table as one unpunctuated line.
    The repeating segment headings are the source's own row boundary, so each
    segment can be kept verbatim.  A following table is cut at its unit
    marker because it states a different axis.  Fails closed unless the block
    really is that shape.
    """

    text = " ".join(body.split())
    if not text or "|" in text:
        return ""
    unit = _TABLE_UNIT_MARKER.search(text)
    if unit is not None:
        text = text[:unit.start()].strip()
    starts = [match.start() for match in _SEGMENT_HEADING.finditer(text)]
    if len(starts) < 3:
        return ""
    rows: list[str] = []
    used = 0
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        row = text[start:end].strip()
        if not row:
            continue
        extra = len(row) + (2 if rows else 0)
        if used + extra > limit:
            break
        rows.append(row)
        used += extra
    return "; ".join(rows)


_METRIC_YEAR_HEADER = re.compile(r"(?:(?:19|20)\d{2}\s*년?.*){2,}")
_METRIC_HEADING_LINE = re.compile(r"^\s*(?:\(?\d+\)|[가-힣]\.|[IVX]+\.)\s*\S")
_METRIC_UNIT_LINE = re.compile(r"^\s*\(\s*단위\s*[:：]?[^)]*\)\s*$")
_TABLE_RULE = re.compile(r"\s*\|(?:\s*:?-+:?\s*\|)+\s*")
_HAS_DIGIT = re.compile(r"\d")


def _metric_table_at(lines: list[str], index: int, keys: tuple[str, ...],
                     *, with_caption: bool = True) -> list[str]:
    """``index`` 행이 속한 마크다운 표에서 머리글과 표기를 담은 행(없으면 첫 행들)."""

    start = index
    while start > 0 and lines[start - 1].lstrip().startswith("|"):
        start -= 1
    end = index
    while end + 1 < len(lines) and lines[end + 1].lstrip().startswith("|"):
        end += 1
    rows = [row for row in lines[start:end + 1] if not _TABLE_RULE.fullmatch(row)]
    if not rows:
        return []
    unit = _local_table_unit_declaration(lines, start)
    unit_prefix = [f"(단위: {unit})"] if unit else []
    # 세로형 표(「| 구분 | 바이오 시밀러 |」)는 품목명이 표 바로 위 캡션 줄에 있다.
    # 짧은 캡션이면 붙여 어느 품목의 행인지 남긴다.
    caption = next((lines[back].strip() for back in range(start - 1, max(-1, start - 3), -1)
                    if lines[back].strip()), "")
    if with_caption and caption and len(caption) <= 60 and not caption.startswith("|") \
            and not _METRIC_UNIT_LINE.match(caption):
        rows = [caption] + rows
        header = rows[:2]
        body = rows[2:]
        keyed = [row for row in body if any(key in _compact_key(row).casefold() for key in keys)]
        return unit_prefix + header + (keyed[:3] if keyed else body[:6])
    header = rows[:1]
    body = rows[1:]
    keyed = [row for row in body if any(key in _compact_key(row).casefold() for key in keys)]
    # 표기가 머리글에만 있는 표(부문별 생산능력·가동률)는 부문 행을 넉넉히 보여
    # 질문의 부문(반도체 등)이 잘리지 않게 한다.
    return unit_prefix + header + (keyed[:3] if keyed else body[:6])


def _metric_digest_candidate(
        lines: list[str], compact_lines: list[str], index: int,
        keys: tuple[str, ...]) -> tuple[int, list[str]]:
    """표기를 담은 한 줄에서 값 발췌 후보와 우선순위(작을수록 좋음)."""

    line = lines[index]
    stripped = line.strip()
    if stripped.startswith("|"):
        pieces = _metric_table_at(lines, index, keys)
        return (1 if any(_HAS_DIGIT.search(row) for row in pieces[1:]) else 4, pieces)
    if len(stripped) > 160 or re.search(r"[다요]\.\s", stripped):
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[다요]\.)\s+", stripped)
                     if sentence.strip()]
        pieces = [sentence for sentence in sentences
                  if any(key in _compact_key(sentence).casefold() for key in keys)][:3]
        return (2 if any(_HAS_DIGIT.search(sentence) for sentence in pieces) else 4, pieces)
    # 짧은 라벨 줄 — 아래에 표가 오면 라벨+표, 아니면 이어지는 짧은 숫자 줄(PDF 선형화 표).
    look = index + 1
    skipped = 0
    while look < len(lines) and skipped < 4:
        candidate = lines[look].strip()
        if not candidate or _METRIC_UNIT_LINE.match(candidate):
            look += 1
            skipped += 1
            continue
        break
    if look < len(lines) and lines[look].lstrip().startswith("|"):
        pieces = [stripped] + _metric_table_at(lines, look, keys, with_caption=False)
        return (1 if any(_HAS_DIGIT.search(row) for row in pieces[1:]) else 4, pieces)
    pieces = [stripped]
    for follow in lines[index + 1:index + 7]:
        candidate = follow.strip()
        if not candidate or _METRIC_HEADING_LINE.match(candidate) or len(candidate) > 60:
            break
        pieces.append(candidate)
    header = next((
        lines[back].strip() for back in range(index - 1, max(-1, index - 9), -1)
        if _METRIC_YEAR_HEADER.search(lines[back])), None)
    if header:
        pieces.insert(0, header)
    return (3 if any(_HAS_DIGIT.search(piece) for piece in pieces[1:]) else 5, pieces)


def _dedupe_adjacent_row_cells(row: str) -> str:
    """Normalize cell surfaces without removing repeated or empty coordinates.

    Equal neighbouring values may belong to different years or amount/count
    columns. Deduplicating each row independently silently shifts its values.
    """

    cells = _table_cells(row)
    if not cells or _is_table_rule(cells):
        return row
    flattened = [" ".join(cell.replace(" > ", " ").split()) for cell in cells]
    return "| " + " | ".join(_markdown_cell(c) for c in flattened) + " |"


def _value_qualified_headers(names: list[str], rows: list[list[str]]) -> list[str]:
    """Disambiguate repeated headers using explicit percent or region cells."""

    result = list(names)
    for index, name in enumerate(names):
        if names.count(name) < 2:
            continue
        values = [row[index].strip() for row in rows
                  if len(row) == len(names) and row[index].strip() not in {"", "-"}]
        if values and all(re.fullmatch(r"[-+]?\d[\d,.]*\s*%", value)
                          for value in values):
            result[index] = f"{name} 비중(%)"
        elif (_compact_key(name) in {"품목", "제품"} and values
              and any(value in {"한국", "해외", "내수", "수출"} for value in values)
              and all(_compact_key(value) in {"한국", "해외", "내수", "수출", "합계", "소계",
                                             "내부거래조정", "내부거래조정등"} for value in values)):
            # Keep the header as short as the incorrect label it replaces:
            # growing it can evict a total row at the public table budget.
            result[index] = "구분"
    return result


def _render_metric_digest_pieces(pieces: list[str]) -> str:
    """발췌 조각을 화면에 낼 문자열로 합친다.

    표 행(``|`` 로 시작하는 조각)이 이어지면 표는 줄 단위 구조이므로 줄바꿈으로
    세우고 머리글 바로 뒤에 구분선을 넣어 마크다운 표로 선다(#112) — 다른
    경로(`_bounded_table_block`)가 이미 하는 것과 같다. 표가 아닌 조각(캡션·
    문장)이 연이으면 종전대로 " / " 로 이어 한 줄로 낸다 — 표가 전혀 없으면
    (PDF 선형화 숫자 줄·서술 문장) 결과는 종전과 같은 한 줄 그대로다.
    """

    lines: list[str] = []
    table_rows: list[str] = []
    prose: list[str] = []

    def flush_table() -> None:
        if not table_rows:
            return
        header = _dedupe_adjacent_row_cells(table_rows[0])
        header_cells = _table_cells(header)
        body = [_table_cells(row) for row in table_rows[1:]]
        header_cells = _value_qualified_headers(header_cells, body)
        lines.append("| " + " | ".join(_markdown_cell(c) for c in header_cells) + " |")
        lines.append("|" + "|".join(" --- " for _ in header_cells) + "|")
        lines.extend(_dedupe_adjacent_row_cells(row) for row in table_rows[1:])
        table_rows.clear()

    def flush_prose() -> None:
        if not prose:
            return
        lines.append(" / ".join(prose))
        prose.clear()

    for piece in pieces:
        stripped = piece.strip()
        if not stripped:
            continue
        if stripped.lstrip().startswith("|"):
            flush_prose()
            table_rows.append(stripped)
            continue
        flush_table()
        # 표 머리글의 중첩 구분자(「2025년 > 금액」)는 내부 좌표 표기다. 사용자
        # 표면에는 띄어쓰기로 평문화한다(품질 게이트 「표 좌표 노출」).
        prose.append(" ".join(stripped.replace(" > ", " ").split()))
    flush_table()
    flush_prose()
    return "\n".join(lines)


def _truncate_digest_lines(text: str, limit: int) -> str:
    """줄 단위 조각을 문자 예산에 맞춰 자른다 — 표 행 중간에서 자르지 않는다."""

    if len(text) <= limit:
        return text
    lines = text.split("\n")
    kept: list[str] = []
    used = 0
    for line in lines:
        extra = len(line) + (1 if kept else 0)
        if kept and used + extra > limit:
            break
        kept.append(line)
        used += extra
    return "\n".join(kept) if kept else text[:limit]


def _disclosed_metric_digest(
        text: str | None, topic: str, *, limit: int = 420) -> str:
    """공시 서술지표(순이자마진·고정이하여신비율·수주잔고·가동률 …)의 값 줄을 발췌한다.

    표기를 담은 줄마다 후보를 만들고 **값이 실린 후보**를 고른다 — 표 행(머리글+행) >
    숫자가 있는 문장 > 라벨+표 > 라벨+이어지는 숫자 줄(PDF 선형화 표) > 라벨만. 같은
    우선순위면 문서 순서다. 「생산능력과 가동률」처럼 지표가 둘이면 지표마다 고른다.
    표기가 본문에 없으면 빈 문자열 — 호출자가 종전 규칙으로 돌아간다. 원문 줄을
    그대로 이어 붙일 뿐 값을 계산하거나 다시 쓰지 않는다.
    """

    from agent.disclosed_metric_topics import disclosed_metrics_for_query, probe_keys

    entries = disclosed_metrics_for_query(topic)
    if not entries or not text:
        return ""
    lines = [line.rstrip() for line in _source_body(text).splitlines()]
    compact_lines = [_compact_key(line).casefold() for line in lines]
    digests: list[str] = []
    for entry in entries:
        # 정본 표기 > 별칭 순. 부문별 매출은 질문의 부문명(서치플랫폼·아시아)이 키다.
        keys = probe_keys(entry, topic)
        term_key = keys[0] if keys else ""
        hits = [index for index, compact in enumerate(compact_lines)
                if any(key in compact for key in keys)][:12]
        if not hits:
            continue
        best: tuple[tuple[int, int, int], list[str]] | None = None
        for index in hits:
            priority, pieces = _metric_digest_candidate(lines, compact_lines, index, keys)
            if not pieces:
                continue
            # 같은 우선순위면 정본 표기 줄(고정이하여신비율)이 별칭 줄(고정이하여신)보다 먼저다.
            rank = (priority, 0 if term_key and term_key in compact_lines[index] else 1, index)
            if best is None or rank < best[0]:
                best = (rank, pieces)
        if best is None:
            continue
        digest = _render_metric_digest_pieces(best[1])
        if digest and digest not in digests:
            digests.append(digest)
    if not digests:
        return ""
    # 표(줄바꿈 구조)가 하나라도 섞였으면 " / " 로 이을 수 없다 — 표는 줄
    # 단위라 한 줄에 이으면 다시 서지 못한다(#112). 지표별로 빈 줄로 나눈다.
    if any("\n" in digest for digest in digests):
        joined = "\n\n".join(digests)
        return _truncate_digest_lines(joined, limit)
    joined = " / ".join(digests)
    if len(joined) > limit:
        joined = joined[:limit].rsplit(" / ", 1)[0] or joined[:limit]
    return joined


def _is_business_cross_reference(text: str) -> bool:
    """A filing navigation instruction is not a standalone business fact."""
    return bool(re.search(
        r"(?:참고사항|주석|사업의\s*내용).{0,240}참고하시기\s*바랍니다", text))


def _subsidiary_roster_digest(text: str | None, topic: str) -> str:
    """Keep an explicitly disclosed ownership roster on its requested axis."""
    if "자회사" not in _compact_key(topic):
        return ""
    prose = " ".join(
        line.strip() for line in _source_body(text).splitlines()
        if line.strip() and not line.lstrip().startswith("|"))
    roster = re.search(
        r"(?:당사|회사)는[^.!?]{20,700}?지배회사입니다\.", prose)
    if not roster:
        return ""
    sentence = roster.group(0)
    # Require an actual list, not a general statement about holding-company
    # status. Preserve the complete source sentence, including its date.
    return sentence if "," in sentence and "현재" in sentence else ""


def _compact_cell_text(
        text: str | None, *, topic: str = "", limit: int = 420,
        temporal_business_change: bool = False,
        ) -> str:
    """Compact a matrix cell while preserving a complete investment row."""

    roster = _subsidiary_roster_digest(text, topic)
    if roster:
        return roster
    business_only = (any(word in _compact_key(topic) for word in ("사업", "제품", "서비스"))
                     and not any(word in _compact_key(topic) for word in
                                 ("매출", "금액", "수익", "비중", "비율")))
    rows = _investment_rows(text)
    if rows:
        return "; ".join(
            f"대상 {target}, 목적 {purpose}, "
            f"금액 {_investment_amount_display(amount, unit)}, 기간 {period}"
            + (f", 기지출 {_investment_amount_display(spent, unit)}"
               if spent else "")
            for target, purpose, amount, period, unit, spent in rows)
    plan = _investment_plan_digest(text, limit=limit)
    if plan:
        return plan
    metric_digest = _disclosed_metric_digest(text, topic, limit=limit)
    if metric_digest:
        return metric_digest
    if any(marker in _compact_key(topic) for marker in ("매출", "수익구조", "수익원")):
        revenue = _revenue_table_digest(text, limit=limit)
        if revenue:
            return revenue
        metrics = _metric_sentences(text, limit=limit)
        if metrics:
            return metrics
    if any(marker in _compact_key(topic) for marker in ("제품", "서비스")):
        products = _product_table_digest(text, limit=limit, business_only=business_only)
        if products:
            return products
        product_narrative = _business_sentence_digest(
            text, limit=limit, topic=topic)
        if product_narrative:
            return product_narrative
    topic_terms = _business_topic_terms(topic)
    if topic_terms and any(marker in _compact_key(topic) for marker in (
            "사업", "제품", "서비스")):
        # Qualified business requests (energy source, technology, product
        # family, etc.) may be proved by a wide segment/product table.  Keep
        # only rows carrying a literal requested qualifier so the heading does
        # not replace the actual business axis in the rendered cell.
        products = _product_table_digest(
            text, limit=limit, required_terms=topic_terms, business_only=business_only)
        if products:
            return products
        qualified = _qualified_business_evidence_digest(
            text, limit=limit, required_terms=topic_terms)
        if qualified:
            return qualified
        topic_narrative = _business_sentence_digest(
            text, limit=limit, required_terms=topic_terms, topic=topic)
        if topic_narrative:
            return topic_narrative
        # Retrieval has already proved the qualified business-axis authority.
        # A source paragraph may use the filing's own nearby term (for example
        # ``기업금융`` where the question says ``투자금융``).  Preserve that
        # source-literal operating sentence rather than inventing an alias or
        # returning a title-only placeholder.
        authority_narrative = _business_sentence_digest(
            text, limit=limit, topic=topic)
        if authority_narrative:
            return authority_narrative
        # Never let a bounded display fallback replace an explicitly qualified
        # topic with an unrelated company-history or sibling-business sentence.
        return "요청 주제와 일치하는 완전한 원문 문장이나 표 행을 표시 범위에서 추출하지 못했습니다."
    base = _compact_narrative_text(text, limit=limit, label=topic)
    topic_key = _compact_key(topic)
    business_topic = (topic_key == "사업" or any(
        marker in topic_key for marker in (
            "주요사업", "사업내용", "사업부문", "사업개요")))
    non_descriptive = (
        not base
        or bool(re.fullmatch(r"\(\s*단위\s*[:：]\s*[^)]+\)", base))
        or base == "구조 블록이 장문 단일 문장이라 내용을 임의로 축약하지 않았습니다."
    )
    if (business_topic and non_descriptive
            and not any(marker in base for marker in _BUSINESS_EVIDENCE_MARKERS)):
        # Some business sections contain only a unit line outside a fully
        # structured product/segment table.  For a broad business topic keep
        # the table's complete identity axis before considering the full
        # value-row projection; the latter belongs to the separately typed
        # product/service cell when both topics use this same source block.
        table_axes = _business_table_axis_digest(text, limit=limit)
        if table_axes:
            return table_axes
        products = _product_table_digest(text, limit=limit, business_only=business_only)
        if products:
            return products
    if business_topic:
        business = _business_sentence_digest(
            text, limit=limit,
            suppress_business_noise=_is_main_business_topic(topic),
            topic=topic)
        axis_headings = _business_axis_heading_digest(
            _source_body(text), limit=limit)
        if axis_headings and not _names_segment_axis(business):
            return axis_headings
        table_axes = _business_table_axis_digest(text, limit=limit)
        if table_axes and not _names_segment_axis(business):
            return table_axes
        products = _product_table_digest(text, limit=limit, business_only=business_only)
        # 축을 나열하는 문장에는 «영위·제조·판매» 같은 동사가 없다 — 무엇을
        # 하는지가 아니라 **어떤 축으로 나뉘는지** 를 말하기 때문이다.  동사만
        # 보면 이 문장이 탈락해 제품표가 대신 나가는데, 표는 3개년 행을 통째로
        # 쏟아내 비교에서 읽기 어렵다.  축 문장 자체가 사업 축이므로 실질로 본다.
        business_substantive = bool(business) and (
            _names_segment_axis(business)
            or any(marker in _compact_key(business) for marker in (
                "영위", "제조", "생산", "판매", "공급", "제공", "운영",
                "사업을전개", "사업부문으로구분", "개발")))
        # For a point-in-time major-business summary, a verified product or
        # segment table preserves all material operating rows and is less
        # vulnerable to adjacent pricing/accounting prose.  Temporal change
        # questions still prefer complete operational-change sentences.  But a
        # sentence that already states what the company makes, sells, or
        # operates is the business axis itself: replacing it with a table is
        # how a two-company comparison ends up with one side showing rows and
        # the other a heading, so substantive prose keeps precedence.
        if products and not business_substantive:
            return products
        if business:
            return business
        # 판매 경로·조건만 담긴 절을 회사의 '주요 사업'이라고 바꿔 말하지
        # 않는다. 제품/판매 주제에는 이 표시 경계가 적용되지 않는다.
        if (_is_main_business_topic(topic)
                and _compact_key(base).startswith((
                    "판매경로및판매방법", "판매경로", "판매방법"))):
            return (
                "해당 공시 범위에서는 주요 사업 자체보다 "
                "판매 경로와 방법만 확인됩니다.")
    if any(marker in _compact_key(topic) for marker in ("제품", "서비스")):
        metrics = _metric_sentences(text, limit=max(0, limit - len(base) - 1))
        if metrics and metrics not in base:
            return f"{base} {metrics}"
    return base


def _matrix_excerpt(text: str | None, *, limit: int = 1600) -> str:
    """Keep a source table structurally parseable inside the private matrix."""

    body = _source_body(text)
    if any(line.lstrip().startswith("|") for line in body.splitlines()):
        # A table header/row boundary is semantic evidence.  Do not flatten it
        # into spaces before the deterministic plan-row projection runs.
        return body
    return _clean(body, limit=limit)


_SOURCE_HEADING_ENDINGS = (
    "현황", "내용", "개요", "추이", "변동", "실적", "사항",
)


def _same_source_excerpt_core(left: str, right: str) -> bool:
    """Match only exact source text plus an optional structural heading.

    Search can return one source sentence both alone and with its immediately
    preceding section heading.  This is not a fuzzy similarity rule: after
    whitespace normalization, the longer value must end with the complete
    shorter value and the removed prefix must look like a disclosure heading.
    Numbers, negation, company, period, or body wording can therefore never be
    erased by this helper (#192).
    """

    first = re.sub(r"\s+", " ", _source_body(left)).strip()
    second = re.sub(r"\s+", " ", _source_body(right)).strip()
    if not first or not second:
        return False
    if first == second:
        return True
    longer, shorter = ((first, second) if len(first) > len(second)
                       else (second, first))
    if not longer.endswith(shorter):
        return False
    prefix = longer[:-len(shorter)].strip(" -:：;·")
    if not 2 <= len(prefix) <= 80 or re.search(r"[.!?]", prefix):
        return False
    compact = _compact_key(prefix)
    return any(compact.endswith(_compact_key(ending))
               for ending in _SOURCE_HEADING_ENDINGS)


def _narrative_claim_source_key(claim) -> tuple[tuple[str, str], ...]:
    """Return independently round-tripped section identities for one claim."""

    identities = {
        (citation.doc_id, citation.section_id)
        for citation in getattr(claim, "citations", ())
        if (getattr(citation, "doc_id", None)
            and getattr(citation, "section_id", None)
            and getattr(citation, "verification", None) == "source_roundtrip")
    }
    return tuple(sorted(identities))


def _dedupe_same_source_excerpts(linked) -> list[str]:
    """Keep one copy of an exact body repeated inside the same source section."""

    selected: list[tuple[tuple[tuple[str, str], ...], str]] = []
    for claim in linked:
        excerpt = _matrix_excerpt(claim.text or (
            claim.citations[0].excerpt_prompt_safe if claim.citations else None))
        if not excerpt:
            continue
        source_key = _narrative_claim_source_key(claim)
        duplicate_index = next((
            index for index, (previous_key, previous) in enumerate(selected)
            if source_key and source_key == previous_key
            and _same_source_excerpt_core(previous, excerpt)
        ), None)
        if duplicate_index is None:
            selected.append((source_key, excerpt))
            continue
        # 둘이 같은 본문이면 절 제목이 덜 붙은 짧은 쪽이 일반인에게 더 읽기
        # 쉽다. 근거 identity는 같고 본문 core는 정확히 같으므로 정보 손실이 없다.
        previous_key, previous = selected[duplicate_index]
        if len(excerpt) < len(previous):
            selected[duplicate_index] = (previous_key, excerpt)
    return [excerpt for _source_key, excerpt in selected]


def _period_label(period: object | None) -> str:
    if period is None:
        return "기간 미지정"
    start, end = getattr(period, "start", None), getattr(period, "end", None)
    if start and end:
        return f"{start:%Y-%m-%d}~{end:%Y-%m-%d}"
    return str(period)


def _public_coordinate_period_label(coordinate) -> str:
    """Use a report name instead of exposing an internal missing-period tag."""

    period = getattr(coordinate, "period", None)
    if period is not None:
        start = getattr(period, "start", None)
        end = getattr(period, "end", None)
        # 공개 좌표에서 완전한 역년은 연도만으로 뜻이 충분하다. 내부 좌표와
        # dedupe signature는 계속 ISO 범위를 사용하므로 표시 축약이 결속이나
        # 기간 동일성 판정에 영향을 주지 않는다(#192).
        if (start is not None and end is not None
                and start.year == end.year
                and (start.month, start.day) == (1, 1)
                and (end.month, end.day) == (12, 31)):
            return f"{start.year}년"
        return _period_label(period)
    selector = getattr(coordinate, "document_selector", None)
    form = getattr(selector, "form", None)
    display = {
        "annual": "사업보고서",
        "사업보고서": "사업보고서",
        "half": "반기보고서",
        "반기보고서": "반기보고서",
        "quarter": "분기보고서",
        "분기보고서": "분기보고서",
    }.get(form)
    if display:
        return display
    if getattr(selector, "doc_group", None) == "periodic":
        return "정기보고서"
    return "해당 공시"


def _receipt_numbers(claims) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        ct.rcept_no for claim in claims for ct in claim.citations if ct.rcept_no))


def _coordinate_signature(coordinate) -> tuple[object, ...]:
    """Stable identity for a typed retrieval coordinate across plan tasks."""
    selector = getattr(coordinate, "document_selector", None)
    return (
        getattr(coordinate, "corp_code", ""), getattr(coordinate, "corp_name", ""),
        _period_label(getattr(coordinate, "period", None)), getattr(coordinate, "topic", ""),
        getattr(selector, "rcept_no", None), getattr(selector, "doc_group", None),
        getattr(selector, "form", None),
    )


def _topic_axis(topic: str) -> str:
    """Normalize only disclosure-axis aliases used by the typed fanout.

    This is deliberately not semantic similarity.  It collapses spelling and
    punctuation variants of one DART section axis while keeping compound or
    unknown topics distinct.
    """

    compact = _compact_key(topic).casefold()
    if compact in {
            "주요제품및서비스", "주요제품서비스", "제품및서비스", "제품서비스",
            "주요제품및용역", "제품및용역", "제품", "서비스", "품목",
            }:
        return "products_services"
    if compact in {"매출구성", "수익구조", "수익원", "매출및수익구조"}:
        return "revenue_mix"
    return compact


def _topic_display(topic: str) -> str:
    return "주요 제품 및 서비스" if _topic_axis(topic) == "products_services" else topic


def _is_main_business_topic(topic: str) -> bool:
    compact = _compact_key(topic)
    return (compact == "사업" or any(marker in compact for marker in (
        "주요사업", "사업내용", "사업부문", "핵심사업")))


#: 회사별·기간별 matrix 비교에서 직접 변경 근거가 없는 주요 사업 축에 쓰는
#: 고정 문장.  여러 회사가 나란히 이 문장을 받으면(#114) 결론에 회사 수만큼
#: 같은 문장이 되풀이되므로, 이 상수와 정확히 같을 때만 회사 이름을 모아
#: 한 번으로 합친다.
_MAIN_BUSINESS_CHANGE_UNCONFIRMED = (
    "두 기간의 인용 범위에는 매각·인수·출시·사업 확대·"
    "축소를 직접 밝힌 변경 근거가 없어, 물질적 핵심 사업 "
    "변화 유무는 확정하지 않습니다")


def _build_display_alias_registry() -> dict:
    """Corpus-derived company alias registry, for **display only** (#114).

    A matrix cell is always keyed by the canonical DART ``corp_name`` so
    retrieval and receipts stay unambiguous.  But a reader who wrote
    ``LS일렉트릭`` in the question does not recognize ``엘에스일렉트릭`` as the
    same company when it appears unexplained in the answer.  This never
    changes which company was resolved — it only lets already-verified
    output prefer the name the question itself used.

    Built from the same official corpus sources as
    ``src/canonical/company_alias.py`` (DART corp name, listed name, stock
    code, approved manual aliases) so it stays one registry in spirit; this
    composer only lacks a live ``CanonicalReadModel`` instance to call.  Any
    read failure yields an empty registry, which falls back to the canonical
    name unchanged.
    """

    from src.canonical.company_alias import (
        build_alias_registry, load_approved_aliases, load_universe_english_names)

    root = Path(__file__).resolve().parents[2]
    universe = root / "data" / "corpus" / "universe.csv"
    rows: list[tuple[str, str, str | None, str | None]] = []
    corp_code_by_name: dict[str, str] = {}
    if universe.is_file():
        with universe.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                corp_code = (row.get("corp_code") or "").strip()
                corp_name = (row.get("corp_name") or "").strip()
                listed_name = (row.get("listed_name") or "").strip() or None
                stock_code = (row.get("stock_code") or "").strip() or None
                if not corp_code or not corp_name:
                    continue
                rows.append((corp_code, corp_name, listed_name, stock_code))
                corp_code_by_name[corp_name] = corp_code
    approved, _held = load_approved_aliases(corp_code_by_name)
    english = load_universe_english_names(corp_code_by_name)
    return build_alias_registry(rows, approved, english)


@lru_cache(maxsize=1)
def _display_alias_registry() -> dict:
    try:
        return _build_display_alias_registry()
    except Exception:                                     # noqa: BLE001
        return {}


def _question_display_surface(canonical_name: str, question: str | None) -> str:
    """Return the company surface literally in ``question`` in place of
    ``canonical_name`` when both are verified to resolve to the same
    corp_code (#114).  Falls back to ``canonical_name`` whenever the mapping
    is not unique or the registry cannot be built — this only ever swaps
    display text for a name already proven equivalent, never a guess.
    """

    if not question or not canonical_name or canonical_name in question:
        return canonical_name
    registry = _display_alias_registry()
    if not registry:
        return canonical_name
    from src.canonical.company_alias import lookup_alias
    canonical_codes = {alias.corp_code
                        for alias in lookup_alias(registry, canonical_name)}
    if len(canonical_codes) != 1:
        return canonical_name
    target_code = next(iter(canonical_codes))
    tokens = re.findall(r"[A-Za-z0-9&.+가-힣]+", question)
    particles = frozenset("은는이가을를의에도와과만")
    best: str | None = None
    for start in range(len(tokens)):
        for end in range(start + 1, min(len(tokens), start + 4) + 1):
            prefix = tokens[start:end - 1]
            tail = tokens[end - 1]
            forms = [tail]
            while forms[-1] and forms[-1][-1] in particles:
                forms.append(forms[-1][:-1])
            for final in forms:
                if len(final) < 2:
                    continue
                surface = " ".join((*prefix, final))
                codes = {alias.corp_code
                         for alias in lookup_alias(registry, surface)}
                if codes == {target_code} and (
                        best is None or len(surface) > len(best)):
                    best = surface
    return best or canonical_name


def _apply_question_company_display(
        text: str, cells: "tuple[MatrixCellEvidence, ...]",
        question: str | None) -> str:
    """Swap verified canonical company names for the question's own surface
    form throughout an already-composed matrix answer (#114)."""

    if not question:
        return text
    canonical_names = list(dict.fromkeys(
        cell.label.partition(" · ")[0] for cell in cells))
    mapping = {
        name: display for name in canonical_names
        if (display := _question_display_surface(name, question)) != name}
    if not mapping:
        return text
    pattern = re.compile("|".join(
        re.escape(name) for name in sorted(mapping, key=len, reverse=True)))
    return pattern.sub(lambda match: mapping[match.group(0)], text)


def _narrative_cell_display_limit(
        topic: str, *, temporal: bool, compact_comparison: bool,
        ) -> int:
    """Bound public evidence by the requested comparison axis.

    The private cell keeps the complete, round-tripped source block.  The
    public answer needs only enough of that block to audit the conclusion.
    Revenue tables are especially wide, so a smaller point-in-time budget
    prevents a company comparison from degenerating into a table dump.
    """

    if temporal:
        return 540
    # The tighter point-in-time budgets address wide *cross-company* table
    # comparisons.  A same-company multi-topic matrix is a bounded narrative
    # summary, not that topology; preserve its established 1,200-character
    # display allowance.  Single-coordinate NarrativeDigest never enters
    # this helper and is unchanged as well.
    if not compact_comparison:
        return 1200
    axis = _topic_axis(topic)
    if axis == "revenue_mix" or any(
            marker in _compact_key(topic) for marker in ("수익원", "수익구조")):
        return 520
    if axis == "products_services":
        return 720
    return 560


def _comparison_terms(text: str, *, limit: int = 64) -> tuple[str, ...]:
    """Return source-literal business terms suitable for a bounded contrast.

    This is intentionally not a semantic synonym mapper.  A term may be
    called common only when the same literal surface occurs in every compared
    source cell; company/date-specific dictionaries would make that boundary
    impossible to audit.  Korean case endings are removed only when doing so
    leaves a non-trivial surface that still occurs in the source.
    """

    source = _source_body(text)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9&+.-]{1,}|[가-힣]{2,}", source)
    found: list[str] = []
    for raw in tokens:
        value = raw
        if re.fullmatch(r"[가-힣]+", value):
            for suffix in ("으로는", "에서는", "으로", "에서", "에게", "부터", "까지",
                           "이며", "하고", "에는", "을", "를"):
                if value.endswith(suffix) and len(value) - len(suffix) >= 2:
                    value = value[:-len(suffix)]
                    break
        if (len(value) < 2 or len(value) > 24
                or value.casefold() in {term.casefold() for term in _COMPARISON_TERM_STOPWORDS}
                or value.isdigit()):
            continue
        if value.casefold() not in {term.casefold() for term in found}:
            found.append(value)
        if len(found) >= limit:
            break
    return tuple(found)


def _business_concepts(
        text: str, *, limit: int = 24, strict_material: bool = False,
        ) -> tuple[str, ...]:
    """Extract source-literal product/business concepts, not arbitrary words.

    The previous comparison tokenised every Korean word, so headings and case
    endings such as ``등의`` and ``다음과`` became alleged business
    differences.  This extractor accepts only values in typed product tables
    or noun phrases directly bound to an operating verb.  It deliberately
    does not map synonyms (for example 모바일 -> 스마트폰).
    """

    body = _source_body(text)
    found: list[str] = []

    def append(value: str) -> None:
        if (value and len(value) >= 2 and len(value) <= 48
                and not re.fullmatch(r"[\d.,%() -]+", value)
                and value.casefold() not in {item.casefold() for item in found}):
            found.append(value)

    def add(value: str, *, from_prose: bool = False) -> None:
        value = " ".join(value.replace("ㆍ", "·").split()).strip(" -:;,.[]")
        # PDF table extraction sometimes inserts spaces between every Korean
        # syllable in a short category (``토 목``, ``플 랜 트``).  Collapse
        # only that narrow pattern; ordinary multi-word labels keep spacing.
        if re.fullmatch(r"[가-힣](?:\s+[가-힣]){1,7}", value):
            value = value.replace(" ", "")
        # ``str.strip('()')`` removed only the closing parenthesis from a
        # balanced source term such as ``Beauty(화장품)``, yielding a broken
        # comparison phrase.  Preserve balanced source punctuation and remove
        # it only when the captured fragment itself is unbalanced.
        if value.count("(") != value.count(")"):
            value = value.replace("(", " ").replace(")", " ")
            value = " ".join(value.split())
        value = re.sub(r"^(?:가|나|다)\.\s*", "", value)
        value = re.sub(r"^\(\d+\)\s*", "", value)
        value = re.sub(
            r"^(?:주요\s*)?(?:제품|서비스)(?:\s*등)?(?:의)?\s*(?:현황|내용|매출)?\s*",
            "", value)
        value = re.sub(
            r"^(?:당사|연결실체|연결회사|회사[A-Za-z가-힣0-9]*)(?:는|가)?\s*", "", value)
        value = re.sub(r"\s*(?:등(?:의)?|관련)$", "", value)
        if strict_material and from_prose:
            # A short-object regex may capture the viewer's glued section
            # subject together with the actual product object.  Retain only
            # the object after the final ``...부문은/는`` boundary.
            value = re.sub(r"^.*부문(?:은|는|이|가)\s*", "", value)
            # ``TV를 비롯하여 모니터, ...`` is a source enumeration, not one
            # compound product name. Preserve both literal members before the
            # comma-based enumeration loop moves on to later products.
            if "비롯하여" in value:
                for enumerated in re.split(
                        r"\s*를\s+비롯하여\s+", value):
                    append(" ".join(enumerated.split()).strip(" -:;,.[]"))
                return
        # Do not strip bare ``이``/``가``: they are also lexical endings in
        # product names such as ``디스플레이``.  Object particles are enough
        # for the activity-bound phrases accepted below.
        value = re.sub(r"(?:을|를|은|는)$", "", value)
        if (strict_material and from_prose
                and (len(value.split()) > 6
                     or any(marker in value for marker in (
                         "별로", "따라", "위해", "통해", "있으며"))
                     or re.search(
                         r"(?:은|는|이|가)\s+[A-Za-z0-9]{1,8}$", value))):
            return
        if (not value or len(value) < 2 or len(value) > 48
                or re.search(r"(?:^|\s)[가-힣](?:\s|$)", value)
                or value.casefold() in {
                    "제품", "서비스", "사업", "사업 영역", "영업활동", "현황",
                    "다음과", "당사", "연결실체", "연결회사", "사업부문",
                    "품목", "설명", "제품 설명", "주요사업 내역", "구분",
                    "기타", "소계", "합계", "총계", "매출총계", "내부거래제거",
                    "조정 및 제거", "조정및제거",
                    "있으며",
                }
                or re.search(r"(?:합니다|있습니다|하였습니다)$", value)):
            return
        if from_prose and re.search(r"부문(?:에서|은|는|이|가)\s", value):
            # A clipped clause such as '부품 사업은 DS 부문에서 DRAM' is
            # neither a product noun nor an independently readable concept.
            return
        append(value)
        # Comparison axes are frequently decorated by disclosure labels such
        # as ``사업부문`` or an explanatory parenthesis.  Add only literal
        # substrings already present in the source so two issuers can align on
        # e.g. 협동로봇, 화장품, 원자력 or 신재생 without a synonym dictionary.
        for inner in re.findall(r"\(([^()]{2,32})\)", value):
            append(" ".join(inner.split()))
        def axis_atom(surface: str) -> str:
            prior = ""
            while prior != surface:
                prior = surface
                surface = re.sub(
                    r"\s*(?:사업부문|영업부문|사업|부문|발전|에너지)$",
                    "", surface)
            return " ".join(surface.split()).strip()

        atom = axis_atom(value)
        if atom != value:
            append(atom)
        # Coordinated source phrases keep both axes literal (``원자력발전과
        # 신재생에너지 사업``).  Split only an explicit Korean conjunction
        # followed by whitespace, then remove the same disclosure suffixes.
        for part in re.split(r"(?:과|와)\s+", value):
            part_atom = axis_atom(part)
            if part_atom != value:
                append(part_atom)

    # Typed tables: use only declared identity/business/use columns.
    lines = body.splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = index
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            index += 1
        block = lines[start:index]
        if len(block) < 3 or not _is_table_rule(_table_cells(block[1])):
            continue
        header = _table_cells(block[0])
        compact_header = [_compact_key(name) for name in header]
        # Prefer category/identity columns.  Descriptive prose columns such as
        # ``제품 개요`` can contain dozens of incidental nouns and previously
        # exhausted the concept budget before the second/third product row.
        positions = [
            offset for offset, name in enumerate(compact_header)
            if ((any(marker in name for marker in (
                    "사업부문", "영업부문", "주요사업", "주요제품", "제품군", "품목",
                    "서비스명", "제품명", "상품명"))
                 or name == "사업")
                and not any(marker in name for marker in (
                    "개요", "설명", "내용", "매출", "비율", "금액")))
        ]
        if not positions:
            positions = [
                offset for offset, name in enumerate(compact_header)
                if (any(marker in name for marker in (
                        "제품", "서비스", "상품", "용도", "사업", "부문"))
                    and not any(marker in name for marker in (
                        "개요", "설명", "내용", "매출", "비율", "금액")))
            ]
        material_rows: list[list[str]] = []
        for row_line in block[2:]:
            row = _table_cells(row_line)
            if len(row) != len(header) or _is_table_rule(row):
                continue
            if positions and all(
                    _compact_key(row[offset]).casefold()
                    == compact_header[offset].casefold()
                    for offset in positions):
                # Multi-level viewer headers are serialized as data rows.
                continue
            material_rows.append(row)
        # Preserve declared axis priority: collect every segment/business value
        # before moving to the product column.  Row-first traversal interleaves
        # products ahead of later segments and produces an unstable shared-axis
        # ordering even though the table itself declares the column hierarchy.
        for offset in positions:
            for row in material_rows:
                for value in re.split(r"[,/]", row[offset]):
                    add(value)

    prose = " ".join(line.strip() for line in lines
                     if line.strip() and not line.lstrip().startswith("|"))
    # Enumerations immediately feeding a disclosed operating activity.
    for match in re.finditer(
            r"(?P<objects>[^.!?]{2,180}?)(?:을|를)?\s*"
            r"(?:생산|제조|판매|공급|제공|영위|주력|운영|전개|건설|시공|"
            r"수주|개발|진행|추진|수행|확보)"
            r"(?:ㆍ|,|하고|하며|합니다|중|으로|하고\s*있)", prose):
        objects = match.group("objects")
        # Do not pull in an arbitrary preceding sentence heading.
        objects = re.split(r"(?:있습니다|입니다|현황)", objects)[-1]
        objects = re.sub(r"\s*등의?\s+사업\s+영역에서\s+", ", ", objects)
        for value in re.split(r"[,/]", objects):
            add(value, from_prose=True)
    # Short grammatical objects immediately bound to an operating verb are a
    # safer fallback for prose-heavy disclosures than dumping two source
    # paragraphs.  Keep the captured source noun phrase literal; no synonym or
    # industry dictionary is applied.
    for match in re.finditer(
            r"(?P<object>[가-힣A-Za-z0-9·()+/& -]{2,48}?)(?:을|를)\s*"
            r"(?:제작|생산|제조|판매|공급|제공|운영|건설|시공|수행|추진|전개)",
            prose):
        value = re.split(r"[.!?;,]", match.group("object"))[-1]
        words = value.split()
        if len(words) > 4:
            value = " ".join(words[-4:])
        add(value, from_prose=True)
    # Parentheses directly naming the contents of a source-labelled segment.
    for value in re.findall(r"[가-힣A-Za-z0-9· ]+사업부문\(([^()]{2,80})\)", prose):
        for part in re.split(r"[,/]", value):
            add(part, from_prose=True)
    # Some prose-first issuers state their axis as ``핵심 사업은 X 사업으로``
    # without an object particle before an operating verb.  Preserve that
    # explicitly labelled noun phrase so a company comparison does not omit
    # the issuer merely because its section has no product table.
    for match in re.finditer(
            r"(?:핵심|주요)\s*사업(?:은|으로)\s*"
            r"(?P<object>[가-힣A-Za-z0-9·()+/& -]{2,64}?)"
            r"(?=\s*(?:으로|이며|이고|입니다|,))", prose):
        add(match.group("object"), from_prose=True)

    return tuple(found[:limit])


@dataclass(frozen=True)
class _ComparableMetric:
    key: str
    label: str
    raw: str
    value: Decimal
    unit: str


def _decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None


def _comparable_value(raw: str) -> tuple[Decimal, str] | None:
    """Normalize only explicit, exact units into a comparison basis."""

    value = " ".join(raw.split())
    if value.endswith("%"):
        number = _decimal(value[:-1])
        return (number, "%") if number is not None else None
    if value.endswith("개"):
        number = _decimal(value[:-1])
        return (number, "개") if number is not None else None
    mixed = re.fullmatch(
        r"(?:(?P<jo>\d[\d,]*(?:\.\d+)?)조\s*)?"
        r"(?P<eok>\d[\d,]*(?:\.\d+)?)억원", value)
    if mixed:
        jo = _decimal(mixed.group("jo") or "0")
        eok = _decimal(mixed.group("eok"))
        if jo is not None and eok is not None:
            return jo * Decimal(10000) + eok, "억원"
    simple = re.fullmatch(
        r"(?P<number>\d[\d,]*(?:\.\d+)?)(?P<unit>조원|백만원|천원|원)", value)
    if not simple:
        return None
    number = _decimal(simple.group("number"))
    unit = simple.group("unit")
    if number is None:
        return None
    if unit == "조원":
        return number * Decimal(10000), "억원"
    return number, unit


def _metric_key(label: str, *, share: bool = False) -> str:
    key = _compact_key(label).casefold()
    if key in {"매출", "매출액", "매출총액"}:
        key = "총매출"
    elif "종속기업" in key:
        key = "종속기업수"
    return key + ("비중" if share else "매출" if key.endswith("부문") else "")


def _metric(label: str, raw: str, *, share: bool = False) -> _ComparableMetric | None:
    comparable = _comparable_value(raw)
    if comparable is None:
        return None
    value, unit = comparable
    display_label = "매출" if _metric_key(label) == "총매출" else " ".join(label.split())
    if share:
        display_label += " 구성비"
    return _ComparableMetric(
        key=_metric_key(label, share=share), label=display_label,
        raw=" ".join(raw.split()), value=value, unit=unit)


def _sentence_fragment(text: str, start: int, end: int) -> str:
    """Return a sentence window without treating decimal dots as stops."""

    boundaries = tuple(re.finditer(r"(?<!\d)\.(?!\d)", text))
    left = max((match.end() for match in boundaries if match.end() <= start),
               default=0)
    right = min((match.start() for match in boundaries if match.start() >= end),
                default=len(text))
    return text[left:right]


def _is_relative_change_metric(text: str, match: re.Match[str], value: str) -> bool:
    """Reject YoY rates and delta amounts as cross-report level values."""

    local = _sentence_fragment(text, match.start(), match.end())
    compact = _compact_key(local)
    relative_base = any(
        marker in compact for marker in ("전년", "전기", "전년도"))
    relative_base = relative_base or re.search(
        r"20\d{2}년대비", compact) is not None
    if "대비" not in compact or not relative_base:
        return False
    if value.endswith("%"):
        return True
    trailing = text[match.end():match.end() + 48]
    return re.match(
        r"\s*(?:\([^)]*%\))?\s*(?:증가|감소|상승|하락)", trailing) is not None


def _direct_metrics(text: str) -> tuple[_ComparableMetric, ...]:
    """Extract source-labelled metrics that can be compared deterministically."""

    body = " ".join(_source_body(text).split())
    found: list[_ComparableMetric] = []
    seen: set[str] = set()

    def add(item: _ComparableMetric | None) -> None:
        if item is not None and item.key not in seen:
            seen.add(item.key)
            found.append(item)

    # Total revenue and counts are unambiguous labels in prose.
    for pattern in _DIRECT_METRIC_PATTERNS:
        for match in pattern.finditer(body):
            value = match.group("value")
            if _is_relative_change_metric(body, match, value):
                continue
            add(_metric(match.group("label"), match.group("value")))

    # Segment amounts and their immediately attached shares.  Requiring the
    # grammatical particle and adjacent value keeps unrelated numbers out.
    segment_pattern = re.compile(
        rf"(?P<label>(?:[A-Za-z][A-Za-z0-9& .+-]{{0,18}}|[가-힣]{{2,12}})\s*부문|"
        rf"SDC|Harman)(?:은|는|이|가)\s*(?P<value>{_DIRECT_VALUE})"
        rf"(?:\s*\((?P<share>\d[\d,]*(?:\.\d+)?%)\))?"
    )
    for match in segment_pattern.finditer(body):
        label = " ".join(match.group("label").split())
        value = match.group("value")
        # ``전년 대비 7.5%`` in a 2025 report and ``전년 대비 6.8%`` in a
        # 2023 report have different comparison bases.  They are not two
        # levels of one metric and must never be subtracted across reports.
        relative_yoy = _is_relative_change_metric(body, match, value)
        if not relative_yoy:
            add(_metric(label, value))
        if match.group("share"):
            add(_metric(label, match.group("share"), share=True))

    # Product tables retain a declared item label, value column and local
    # unit.  Rows from two periods become comparable only on that exact key.
    lines = _source_body(text).splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = index
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            index += 1
        block = lines[start:index]
        if len(block) < 3 or not _is_table_rule(_table_cells(block[1])):
            continue
        header = _table_cells(block[0])
        compact = [_compact_key(value) for value in header]
        identity = next((i for i, value in enumerate(compact)
                         if any(marker in value for marker in ("품목", "제품", "서비스"))), None)
        amount = next((i for i, value in enumerate(compact) if "매출액" in value), None)
        share = next((i for i, value in enumerate(compact) if "비율" in value), None)
        if identity is None or amount is None:
            continue
        unit = _local_table_currency_unit(lines, start)
        for row_line in block[2:]:
            row = _table_cells(row_line)
            if len(row) != len(header) or _is_table_rule(row):
                continue
            label = " ".join(row[identity].split())
            if not label or _compact_key(label) == "단순합계":
                continue
            raw_amount = row[amount]
            if unit and _PLAIN_NUMERIC_AMOUNT.fullmatch(raw_amount):
                raw_amount += unit
            add(_metric(label + " 매출액", raw_amount))
            if share is not None and row[share]:
                add(_metric(label, row[share], share=True))
    return tuple(found)


def _subject_particle(word: str) -> str:
    """Pick 이/가 by the final syllable's coda, so the sentence reads naturally."""

    tail = word.strip()[-1:] if word.strip() else ""
    if not ("가" <= tail <= "힣"):
        return "이"
    return "이" if (ord(tail) - 0xAC00) % 28 else "가"


def _compact_with_starts(text: str) -> tuple[str, tuple[bool, ...]]:
    """Compact text for matching while remembering where source words began.

    Returns the same string ``_compact_key`` produces together with one flag
    per character: True when that character started a word in the source, i.e.
    it was the first character or the source had a separator before it.
    """

    compact = _compact_key(text).casefold()
    starts: list[bool] = []
    cursor = 0
    boundary = True
    for char in text:
        folded = _compact_key(char).casefold()
        if not folded:
            boundary = True
            continue
        for offset, _ in enumerate(folded):
            if cursor < len(compact):
                starts.append(boundary and offset == 0)
                cursor += 1
        boundary = False
    starts.extend([False] * (len(compact) - len(starts)))
    return compact, tuple(starts[:len(compact)])


# Disclosure bookkeeping labels that are never a distinguishing business.
# ``복합`` is a word-start prefix of ``복합화력발전소`` exactly as ``신재생``
# is of ``신재생에너지``, so no boundary rule separates them — it is excluded
# here because it names no business on its own, not because of where it sits.
_AXIS_NOISE = {
    "기타", "소계", "합계", "총계", "내부거래제거", "조정및제거", "부품등기타",
    "수출", "내수", "제품", "상품", "용역", "부문", "구분", "복합",
}


# Korean connective/adnominal shapes that mark a captured span as a clause
# fragment rather than a product or segment name.
_CLAUSE_FRAGMENT = re.compile(
    r"(?:^|\s)(?:서로|다양한|복잡한|새로운|주요한|다음과|이러한|해당)(?:\s|$)"
    r"|\S+(?:하고|하며|되고|되며|이루어|있으며|으로써|함으로)"
    # 「사업별로 보면」·「살펴보면」처럼 글을 이끄는 말은 사업·제품 이름이
    # 아니다. 원문 문장에서 잘려 나온 조각이므로 용어 후보에서 뺀다 (#258 ①).
    r"|\S*(?:보면|살펴보면|따르면|기준으로|중심으로|바탕으로)(?:\s|$)"
    r"|(?:^|\s)용역(?:\s|$)")


def _coordinate_heading(label: str, topic: str) -> str:
    """근거 줄의 머리글을 좌표 표기가 아니라 우리말로 쓴다.

    ``삼성생명 · 정기보고서 (사업부문)`` 은 회사·기간·축을 가운뎃점과 괄호로
    이어 붙인 내부 좌표 표기다. 정보는 그대로 두고 사람이 읽는 구로 바꾼다
    (이슈 #260 ②).
    """

    company, _, period = (label or "").partition(" · ")
    axis = _topic_display(topic) or ""
    head = " ".join(part for part in (company.strip(), period.strip()) if part)
    if head and axis:
        return f"{head}의 {axis}"
    return head or axis


def _replace_filer_first_person(text: str, label: str) -> str:
    """원문의 1인칭을 회사 이름으로 바꾼다.

    ``당사는``·``본사는`` 은 공시를 낸 회사가 스스로를 가리키는 말이다.
    답변에 그대로 실으면 말하는 주체가 우리인지 그 회사인지 섞인다 (#260 ③).
    """

    company = (label or "").partition(" · ")[0].strip()
    if not company or not text:
        return text
    return re.sub(r"(?<![가-힣])(?:당사|본사|폐사)(?=[가-힣\s,.·]|$)", company, text)


def _term_contrast(
        cells: tuple["MatrixCellEvidence", ...], *, include_unique: bool = True,
        temporal_business_change: bool = False,
        ) -> str:
    """Describe literal common/exclusive terms without inventing equivalence."""

    if len(cells) < 2:
        return ""
    # A comparison conclusion must be auditable against the very cell text
    # rendered immediately above it.  Source terms omitted by the bounded cell
    # digest cannot reappear only in the conclusion as an alleged commonality.
    # Audit against the same visible budget used by deterministic rendering.
    # Temporal matrices use a tighter budget to avoid raw historical dumps;
    # point-in-time comparisons keep the broader product-row allowance.
    visible = [
        _compact_cell_text(
            cell.excerpt, topic=cell.topic,
            limit=_narrative_cell_display_limit(
                cell.topic, temporal=temporal_business_change,
                compact_comparison=len({
                    item.label.partition(" · ")[0] for item in cells}) > 1),
            temporal_business_change=temporal_business_change)
        for cell in cells]
    terms = [
        tuple(term for term in _business_concepts(
            cell.excerpt, strict_material=temporal_business_change)
              if _compact_key(term).casefold() in _compact_key(rendered).casefold())
        for cell, rendered in zip(cells, visible, strict=True)
    ]
    # Commonality is decided against what each cell *visibly* says, not
    # against what the extractor happened to recover from it.  The two are not
    # the same: a product table yields its category column directly, while the
    # equivalent prose sentence can exceed the extractor's phrase-length
    # budget and yield nothing.  Intersecting the extracted sets then reports
    # a term as exclusive to one issuer while the other issuer's own quoted
    # cell states it verbatim — a conclusion that contradicts the evidence
    # printed directly above it.  Auditing against the rendered text keeps
    # every claim checkable and makes the comparison symmetric regardless of
    # which cell digest each side received.
    def _visible_in(term: str, offset: int) -> bool:
        """Is the term positively stated, not negated or buried in a word?

        Korean writes a noun and its suffix without a space, so plain
        containment also matches inside an unrelated compound: ``복합`` inside
        ``복합화력발전소``, ``산제`` inside ``생산제품``.  Comparing compacted
        text is still necessary because PDF extraction splits short categories
        into single syllables (``토 목``), so the separators that would mark a
        word start are gone by then.  Keep both properties by recording, per
        compacted character, whether the source had a separator in front of it,
        and accept only a match that begins at one of those positions.  A
        compound that genuinely starts with the term (``토목부문``,
        ``신재생에너지``) still matches; one that merely contains it does not.
        """

        needle = _compact_key(term).casefold()
        if not needle:
            return False
        # A limitation can carry the requested word while explicitly saying
        # it was *not* found.  Such a sentence is evidence of absence, not a
        # positive common business axis.  Audit each displayed sentence/row
        # separately so a positive occurrence elsewhere is still eligible.
        negative = (
            "확인하지못", "확인되지않", "찾지못", "파악하지못",
            "근거가없", "근거없", "미확인",
        )
        fragments = re.split(r"(?<=[.!?])\s*|[;\n]+", visible[offset])
        for fragment in fragments:
            haystack, starts = _compact_with_starts(fragment)
            if any(marker in haystack for marker in negative):
                continue
            index = haystack.find(needle)
            while index != -1:
                if starts[index]:
                    return True
                index = haystack.find(needle, index + 1)
        return False

    ordered: list[str] = []
    seen: set[str] = set()
    for row in terms:
        for term in row:
            key = _compact_key(term).casefold()
            if key not in seen:
                seen.add(key)
                ordered.append(term)
    # Korean has no orthographic word boundary, so a rendered-text containment
    # test matches inside longer words: ``복합`` hits ``복합화력발전소`` and
    # ``산제`` hits ``생산제품``, inventing a commonality neither filing
    # states.  Require the shared term to be long enough that an accidental
    # containment is implausible, and reject the same disclosure-label noise
    # and clause fragments the exclusive list already rejects.
    shared = [
        term for term in ordered
        if (_compact_key(term).casefold() not in _AXIS_NOISE
            and not _CLAUSE_FRAGMENT.search(term)
            and all(_visible_in(term, offset) for offset in range(len(cells))))
    ][:8]
    # 「각 비교 항목에 문자 그대로 함께 나타나는」은 우리가 어떻게 비교했는지를
    # 설명하는 말이라 읽는 사람에게는 군더더기다. 원문 표현을 맞대었다는 단서는
    # 축마다 되풀이하지 말고 결론 머리에 한 번만 둔다(아래 조립부).
    pieces: list[str] = []
    if shared:
        pieces.append("함께 나오는 표현은 " + "·".join(shared) + "입니다")

    if not include_unique:
        return ". ".join(pieces)

    exclusive_rows: list[str] = []
    silent_rows: list[str] = []
    company_labels = [cell.label.split(" · ", 1)[0] for cell in cells]
    use_company_only = len(set(company_labels)) == len(cells)
    for index, (cell, row) in enumerate(zip(cells, terms, strict=True)):
        unique = [
            term for term in row
            if (not any(_visible_in(term, offset)
                        for offset in range(len(cells)) if offset != index)
                # A comparison item is a product/business noun surface, not
                # a clause copied from a long prose sentence.  Compact source
                # names remain eligible; only multi-word clause-like captures
                # are suppressed.
                and len(term.split()) <= 6
                and _compact_key(term).casefold() not in _AXIS_NOISE
                # A product/segment name is a noun phrase.  Prose capture can
                # also return a clause fragment ("서로 다른 사업과 용역",
                # "이루어지고 복잡한 공정의 플랜트") whose head is a verb or a
                # bare modifier; naming those as an issuer's distinguishing
                # business is wrong even though every word is source-literal.
                # Reject a capture carrying a verbal connective or leading
                # with a pure modifier, and keep the noun phrase itself.
                and not _CLAUSE_FRAGMENT.search(term)
                and not any(marker in _compact_key(term).casefold()
                            for marker in (
                                "차별화된쇼핑경험", "성장해온", "다양한판매자",
                                "이용자의", "판매가격", "가격변동")))
        ]
        # 한 회사의 사업 이름은 명사구다. ``투자은행(Investment Banking)
        # 미래에셋증권은 기업공개(IPO)`` 처럼 주격 조사가 붙은 낱말을 품고 뒤가
        # 이어지는 포착은 문장 경계를 넘어 잘린 조각이라, 원문 그대로여도
        # 사업 이름으로 내보내면 문장이 성립하지 않는다.
        unique = [
            term for term in unique
            if re.search(r"\S+(?:은|는)\s+\S", term) is None
            # 연결어미로 끝나는 포착도 문장 조각이다. ``기침가래약 '이탄징'이며``
            # 는 다음 절로 이어지던 문장을 자른 것이라 사업 이름이 아니다.
            and _CLAUSE_TAIL.search(term) is None
        ]
        # 각주 표시는 원문 표에서 주석을 가리키는 기호다. 사업 이름의 일부가
        # 아니고, 답변에는 그 주석이 실리지 않아 가리킬 곳도 없다.
        unique = [_FOOTNOTE_MARK.sub("", term).strip() for term in unique]
        unique = [term for term in unique if term]
        # 남은 것 중 다른 항목에 이미 포함된 표현은 빼다. ``Investment
        # Banking`` 과 ``IPO`` 가 더 긴 항목 안에 있으면 나열이 같은 말을
        # 두세 번 반복하는 것으로 읽힌다.
        ordered = sorted(unique, key=len, reverse=True)
        kept_terms: list[str] = []
        for term in ordered:
            key = _compact_key(term).casefold()
            if any(key and key in _compact_key(other).casefold()
                   for other in kept_terms):
                continue
            kept_terms.append(term)
        unique = [term for term in unique if term in kept_terms][:4]
        label = company_labels[index] if use_company_only else cell.label
        if unique:
            exclusive_rows.append(f"{label}: {'·'.join(unique)}")
        else:
            # 고유 표현이 없다고 그 회사를 결론에서 지우면, 비교인데 한쪽만
            # 남거나(EDGE-038) 두 회사가 다 사라진다(DEV-NAR-012). 인용은
            # 있었으므로 이름을 세우고 「고유 표현 없음」으로 밝힌다 (#259).
            silent_rows.append(label)
    if exclusive_rows:
        # A literal expression difference does not establish an exclusive
        # business. Attribute the words to each source, never infer absence.
        suffix = (
            " (원문 표현 차이이며, 실제 제품·사업 변화로는 단정하지 않습니다)"
            if temporal_business_change else ""
        )
        listed = [
            (row.split(":", 1)[0].strip(), row.split(":", 1)[1].strip())
            for row in exclusive_rows]
        # 「A 공시에 나온 표현: x·y」는 라벨 나열이라 데이터로 읽힌다. 그렇다고
        # 「A는 x·y를 밝혔습니다」처럼 회사를 행위 주체로 쓰면 실제로 하지
        # 않은 「발표·진술」 행위를 한 것처럼 인물화된다. 축 이름을 주어에
        # 넣어 사실만 진술하는 「A의 축은 x·y입니다」로 쓴다 (#258 ② · #260).
        # 이 호출의 모든 셀은 한 비교 축(예: 「주요 제품 및 서비스」)을 놓고
        # 나뉜 것이라 축 이름이 공통이다. label 이 「회사 · 기간」 원형일 수
        # 있어(용례: 같은 회사의 두 기간 비교) 가운뎃점을 다시 흘리지 않도록
        # 좌표 인용에 쓰는 것과 같은 헬퍼로 사람이 읽는 구를 만든다.
        topic_axis = cells[0].topic
        clauses = []
        for label, terms in listed:
            heading = _coordinate_heading(label, topic_axis)
            clauses.append(f"{heading}{josa(heading, '은', '는')} {terms}입니다")
        head = (", ".join(clauses) + suffix)
        pieces.append(head)
    if silent_rows:
        listed_silent = "·".join(dict.fromkeys(silent_rows))
        pieces.append(
            f"{listed_silent}{josa(listed_silent, '은', '는')} 인용 범위에서 "
            "그 회사만의 표현이 확인되지 않습니다")
    return ". ".join(pieces)


def _revenue_source_rows(
        cell: "MatrixCellEvidence", *, limit: int = 2,
        ) -> tuple[str, ...]:
    """Return a bounded, source-labelled synopsis of a revenue-axis table.

    Financial-company disclosures often put revenue concepts on rows and
    business sources on columns (for example, 영업수익 × WM/IB/S&T or
    순이자손익 × 보험/증권).  Reproducing the whole matrix obscures the
    comparison.  This projection selects the largest positive non-total
    source for at most two explicitly labelled revenue rows.  It does not add
    unlike rows, compare unlike units, or infer a semantic equivalence.
    """

    lines = _source_body(cell.excerpt).splitlines()
    found: list[str] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = index
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            index += 1
        block = lines[start:index]
        if len(block) < 3 or not _is_table_rule(_table_cells(block[1])):
            continue
        header = _table_cells(block[0])
        unit = _local_table_currency_unit(lines, start)
        if unit not in {"억원", "백만원", "천원", "원", "조원"}:
            continue
        for row_line in block[2:]:
            row = _table_cells(row_line)
            if len(row) != len(header) or _is_table_rule(row):
                continue
            row_label = next((
                " ".join(value.split()) for value in row[:2]
                if any(marker in _compact_key(value) for marker in (
                    "매출", "수익", "손익", "보험료", "이자", "수수료"))
            ), "")
            if not row_label:
                continue
            candidates: list[tuple[Decimal, str, str]] = []
            for offset, (axis, raw) in enumerate(zip(header, row, strict=True)):
                if offset < 1 or not axis.strip() or not raw.strip():
                    continue
                axis_key = _compact_key(axis)
                if (axis_key in {"구분", "항목", "금액", "비율"}
                        or any(marker in axis_key for marker in (
                            "합계", "총계", "조정", "연결조정"))):
                    continue
                value = _table_number(raw, unit)
                if value is None or value <= 0:
                    continue
                candidates.append((value, " ".join(axis.split()), " ".join(raw.split())))
            if not candidates:
                continue
            _value, axis, raw = max(candidates, key=lambda item: item[0])
            summary = f"{row_label}: {_public_column_name(axis)} {raw}{unit}"
            if summary not in found:
                found.append(summary)
            if len(found) >= limit:
                return tuple(found)
    return tuple(found)


def _company_revenue_source_contrast(
        cells: tuple["MatrixCellEvidence", ...],
        ) -> str:
    """Contrast each company's own labelled revenue axes without a raw dump."""

    rows: list[str] = []
    for cell in cells:
        sources = _revenue_source_rows(cell)
        if not sources:
            continue
        company = cell.label.partition(" · ")[0]
        rows.append(f"{company}: {', '.join(sources)}")
    if len(rows) != len(cells) or len(rows) < 2:
        return ""
    return "회사별 공시 수익 축을 요약하면 " + "; ".join(rows)


def _format_decimal(value: Decimal) -> str:
    if value == value.to_integral():
        rendered = format(value, "f")
    else:
        rendered = format(value, "f").rstrip("0").rstrip(".")
    whole, dot, fraction = rendered.partition(".")
    return f"{int(whole):,}" + (f".{fraction}" if dot else "")


def _format_delta(value: Decimal, unit: str) -> str:
    absolute = abs(value)
    if unit == "억원" and absolute >= 10000 and absolute == absolute.to_integral():
        integer = int(absolute)
        jo, eok = divmod(integer, 10000)
        return f"{jo:,}조" + (f" {eok:,}억원" if eok else "원")
    suffix = "%p" if unit == "%" else unit
    return f"{_format_decimal(absolute)}{suffix}"


#: 공시 표가 음수를 적는 삼각 기호. 사용자 표면에서는 마이너스로 읽힌다.
_SOURCE_NEGATIVE = re.compile(r"^\s*[△▲]\s*")


def _structured_value_display(value: str, unit: str) -> str:
    """Render both sides of one table comparison in the same public scale."""

    # 공시 표는 음수를 ``△`` 로 적는다. 그대로 두면 독자가 부호로 읽지 못한다.
    negative = _SOURCE_NEGATIVE.match(value or "") is not None
    value = _SOURCE_NEGATIVE.sub("", value or "")
    if unit in {"원", "천원", "백만원", "억원", "조원"}:
        # 같은 문장의 증감액은 `_format_delta` 가 조 단위로 올린다. 전후 값만
        # ``2,589,355억원`` 으로 남으면 한 문장에 두 표기가 섞여 견줄 수 없다.
        rendered = None
        if unit == "억원":
            try:
                amount = Decimal((value or "").replace(",", ""))
            except (InvalidOperation, ValueError):
                amount = None
            if amount is not None:
                rendered = _format_delta(amount, unit)
        if rendered is None:
            rendered = format_source_money_exact(value, unit) or f"{value}{unit}"
        return f"-{rendered}" if negative else rendered
    return f"{value}{unit}"


def _structured_delta_display(value: Decimal, unit: str) -> str:
    """Render an absolute table delta without dropping its declared unit."""

    raw = _format_decimal(abs(value))
    if unit in {"원", "천원", "백만원", "억원", "조원"}:
        # 같은 문장의 전후 값은 `_structured_value_display` 가 조 단위로 올린다.
        # 괄호 안 증감액만 ``746,704억원`` 으로 남으면 한 문장에 두 표기가 섞여
        # 얼마나 늘었는지 견줄 수 없다.
        if unit == "억원":
            return _format_delta(abs(value), unit)
        return format_source_money_exact(raw, unit) or f"{raw}{unit}"
    return _format_delta(value, unit)


def _has_reclassified_comparatives(
        cells: tuple["MatrixCellEvidence", ...],
        ) -> bool:
    """A restated comparative cannot be joined to an older report's basis."""

    if len(cells) < 2:
        return False
    return (cells[-1].restated_comparatives
            or _source_has_reclassified_comparatives(cells[-1].excerpt))


def _source_has_reclassified_comparatives(excerpt: str) -> bool:
    source = _compact_key(_source_body(excerpt))
    return bool(re.search(
        r"(?:비교가능|동일기준|수치|금액|실적|재무정보|재무제표)"
        r"[^.!?。]{0,100}(?:재분류|재작성)", source))


def _propagate_cross_topic_restatement_basis(
        cells: tuple["MatrixCellEvidence", ...],
        ) -> tuple["MatrixCellEvidence", ...]:
    """Bind explicit comparative notes to sibling monetary topic cells.

    A business overview can carry a financial restatement note while its
    product table does not.  Require the same company, requested period and
    receipt; never borrow another company's note or append unrelated source
    tables to the excerpt being compared.
    """
    bases = {
        (cell.label, receipt)
        for cell in cells
        if (len(cell.receipts) == 1
            and _source_has_reclassified_comparatives(cell.excerpt))
        for receipt in cell.receipts if receipt
    }
    return tuple(
        replace(cell, restated_comparatives=True)
        if (_topic_axis(cell.topic) in {"products_services", "revenue_mix"}
            and len(cell.receipts) == 1
            and (cell.label, cell.receipts[0]) in bases)
        else cell
        for cell in cells)


def _metric_deltas(
        cells: tuple["MatrixCellEvidence", ...],
        ) -> tuple[tuple[_ComparableMetric, _ComparableMetric, Decimal], ...]:
    if len(cells) < 2 or _has_reclassified_comparatives(cells):
        return ()
    rows = [_direct_metrics(cell.excerpt) for cell in cells]
    maps = [{metric.key: metric for metric in row} for row in rows]
    shared = set(maps[0]).intersection(*(set(row) for row in maps[1:]))
    changes = []
    for metric in rows[0]:
        if metric.key not in shared:
            continue
        latest = maps[-1][metric.key]
        if any(row[metric.key].unit != metric.unit for row in maps):
            continue
        if metric.value == latest.value:
            continue
        changes.append((metric, latest, latest.value - metric.value))
    return tuple(changes)


def _metric_contrast(
        cells: tuple["MatrixCellEvidence", ...], *, topic: str = "",
        ) -> str:
    """Compare source values and calculate only exact, same-unit deltas."""

    # A main-business cell can contain incidental subsidiary counts and
    # nearby performance totals.  Without a revenue/product axis those values
    # have no stable business label, so keep the comparison narrative-only.
    topic_axis = _topic_axis(topic)
    if (topic_axis not in {"products_services", "revenue_mix"}
            and not any(marker in _compact_key(topic) for marker in ("매출", "수익"))):
        return ""
    changes: list[str] = []
    for before, after, delta in _metric_deltas(cells):
        # Product/service and revenue-mix excerpts can contain an incidental
        # sentence about the number of subsidiaries.  That count is valid
        # source data, but it is not a metric on either requested axis and
        # must not become the headline of a product or sales comparison.
        if before.key == "종속기업수":
            continue
        direction = "증가" if delta > 0 else "감소"
        if before.unit == "%":
            direction = "상승" if delta > 0 else "하락"
        changes.append(
            f"{before.label} 변화: {cells[0].label} {before.raw} → "
            f"{cells[-1].label} {after.raw} ({_format_delta(delta, before.unit)} {direction})")
        if len(changes) >= 8:
            break
    return ". ".join(changes)


@dataclass(frozen=True)
class _StructuredPeriodDelta:
    table_label: str
    period_label: str
    rows: tuple[tuple[str, str, str, Decimal, str], ...]
    largest_label: str | None


@dataclass(frozen=True)
class _StructuredReportSnapshot:
    """One requested report-year column from a source-labelled table."""

    table_label: str
    signature: tuple[str, ...]
    period_header: str
    report_year: int
    unit: str
    rows: tuple[tuple[str, str, Decimal], ...]
    unique_identity: bool


def _table_number(raw: str, unit: str) -> Decimal | None:
    value = " ".join(raw.split()).replace(",", "")
    negative = value.startswith("△") or (value.startswith("(") and value.endswith(")"))
    value = value.strip("△() ")
    if not re.fullmatch(r"\d+(?:\.\d+)?", value):
        return None
    number = _decimal(value)
    return -number if negative and number is not None else number


def _segment_axis_label(
        row: list[str], identity_positions: list[int],
        row_lines: list[str], header: list[str], *, topic: str = "",
        ) -> str:
    """Label one delta row by the table's primary identity axis.

    A revenue-mix comparison is about the segment axis (부문/구분), so the
    product-description column only makes each row a one-off composite that
    cannot pair across reports once the wording drifts.  Product/service
    comparisons keep the composite because the product identity is the axis
    being compared.  Falls back to the composite whenever the primary column
    does not separate the source rows on its own.
    """

    parts = tuple(dict.fromkeys(
        " ".join(row[offset].split()) for offset in identity_positions
        if row[offset].strip()))
    if len(parts) <= 1 or _topic_axis(topic) != "revenue_mix":
        return " / ".join(parts)
    primary = identity_positions[0]
    seen: list[str] = []
    for line in row_lines:
        cells = _table_cells(line)
        if len(cells) != len(header) or _is_table_rule(cells):
            continue
        value = " ".join(cells[primary].split())
        if value:
            seen.append(value)
    if len(seen) == len(set(seen)):
        return " ".join(row[primary].split()) or " / ".join(parts)
    return " / ".join(parts)


def _table_axis_heading(lines: list[str], start: int, *, topic: str) -> str:
    """Use a nearby source heading only when it is actually heading-like.

    Viewer extraction can glue a full latest-report YoY paragraph directly
    above the table.  Reusing that paragraph as ``table_label`` prepends an
    unrelated 2024→2025 narrative to an otherwise exact 2023→2025 delta.
    Keep concise source headings; fall back to the typed requested axis for a
    long or sentence-like candidate.
    """

    fallback = _topic_display(topic) or "비교 항목"
    candidate = next((
        " ".join(line.split())
        for line in reversed(lines[max(0, start - 8):start])
        if line.strip() and not line.lstrip().startswith("|")
        and "단위" not in line
    ), "")
    if not candidate:
        return fallback
    compact = _compact_key(candidate)
    sentence_like = (
        len(candidate) > 80
        or bool(re.search(r"(?:하였습니다|했습니다|됩니다|입니다|기록했습니다)[.!?]?$", candidate))
        or ("대비" in compact and any(
            marker in compact for marker in ("증가", "감소", "상승", "하락")))
    )
    return fallback if sentence_like else candidate


def _structured_period_deltas(
        cell: "MatrixCellEvidence", *, topic: str,
        ) -> tuple[_StructuredPeriodDelta, ...]:
    """Compare adjacent period columns inside one source-labelled table.

    The two values share the same row identity, table unit and source block.
    No comparison is made across unrelated tables or separately stated YoY
    percentages.
    """

    topic_axis = _topic_axis(topic)
    if (topic_axis not in {"products_services", "revenue_mix"}
            and not any(marker in _compact_key(topic) for marker in ("매출", "수익"))):
        return ()
    lines = _source_body(cell.excerpt).splitlines()
    results: list[_StructuredPeriodDelta] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = index
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            index += 1
        block = lines[start:index]
        if len(block) < 3 or not _is_table_rule(_table_cells(block[1])):
            continue
        header = _table_cells(block[0])
        compact = [_compact_key(value) for value in header]
        periods = [
            offset for offset, value in enumerate(compact)
            if re.fullmatch(r"(?:제\d+기|20\d{2}년)", value)
        ]
        if len(periods) < 2:
            continue
        latest_pos, prior_pos = periods[0], periods[1]
        identity_positions = [
            offset for offset, value in enumerate(compact[:latest_pos])
            if any(marker in value for marker in (
                "부문", "품목", "제품", "서비스", "구분", "항목", "유형"))
        ]
        if not identity_positions:
            continue
        # Preserve all source-declared row coordinates.  Using only the first
        # column collapses product/channel rows into dozens of indistinguish-
        # able ``엔터테인먼트`` changes and creates a raw-dump conclusion.
        unit = _local_table_currency_unit(lines, start)
        if unit not in {"억원", "백만원", "천원", "원", "조원"}:
            continue
        rows: list[tuple[str, str, str, Decimal, str]] = []
        argmax_candidates: list[tuple[Decimal, str]] = []
        for row_line in block[2:]:
            row = _table_cells(row_line)
            if (len(row) != len(header) or _is_table_rule(row)
                    or latest_pos >= len(row) or prior_pos >= len(row)):
                continue
            # The source's primary axis (부문/구분) is what the question asks
            # to compare.  Joining every identity column makes each row a
            # one-off composite, which reads as a different axis and pushes
            # sibling segments out of the rendered subset.  Composite keys are
            # kept only where the primary column alone repeats.
            label = _segment_axis_label(
                row, identity_positions, block[2:], header, topic=topic)
            latest = _table_number(row[latest_pos], unit)
            prior = _table_number(row[prior_pos], unit)
            if not label or latest is None or prior is None or latest == prior:
                continue
            delta = latest - prior
            rows.append((label, row[prior_pos], row[latest_pos], delta, unit))
            label_key = _compact_key(label)
            if (delta > 0 and not any(marker in label_key for marker in (
                    "합계", "총계", "기타", "내부거래", "조정", "제거"))):
                argmax_candidates.append((delta, label))
        if not rows:
            continue
        latest_header, prior_header = header[latest_pos], header[prior_pos]
        report_year = re.search(r"20\d{2}", cell.label.partition(" · ")[2])
        if (report_year and latest_header.startswith("제") and prior_header.startswith("제")):
            latest_year = int(report_year.group(0))
            period_label = (
                f"{latest_year - 1}→{latest_year} "
                f"({_public_column_name(prior_header)}"
                f"→{_public_column_name(latest_header)})")
        else:
            period_label = (f"{_public_column_name(prior_header)}"
                            f"→{_public_column_name(latest_header)}")
        nearest_heading = _table_axis_heading(lines, start, topic=topic)
        largest = max(argmax_candidates)[1] if argmax_candidates else None
        results.append(_StructuredPeriodDelta(
            table_label=nearest_heading, period_label=period_label,
            rows=tuple(rows), largest_label=largest))
    return tuple(results)


def _cell_report_year(cell: "MatrixCellEvidence") -> int | None:
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", cell.label.partition(" · ")[2])
    return int(match.group(1)) if match else None


def _structured_row_key(label: str) -> str:
    """Normalize only structural aggregate aliases inside a composite row."""

    parts = []
    for value in label.split(" / "):
        compact = _compact_key(value).casefold()
        parts.append("합계" if compact in {"계", "합계", "총계"} else compact)
    return "/".join(parts)


def _structured_report_snapshots(
        cell: "MatrixCellEvidence", *, topic: str,
        target_year: int | None = None, require_explicit_year: bool = False,
        ) -> tuple[_StructuredReportSnapshot, ...]:
    """Read the requested report year's values, never an internal YoY pair.

    Annual-report tables usually include the current and one or two prior
    fiscal columns.  A matrix comparison must bind the *current* column of the
    before report to the current column of the after report; comparing the two
    leading columns of only the after report silently changes a 2023→2025
    request into 2024→2025.  Explicit year headers take precedence.  With
    fiscal-number-only headers, the leading period is the report-year value.
    """

    topic_axis = _topic_axis(topic)
    if (topic_axis not in {"products_services", "revenue_mix"}
            and not any(marker in _compact_key(topic) for marker in ("매출", "수익"))):
        return ()
    report_year = target_year if target_year is not None else _cell_report_year(cell)
    if report_year is None:
        return ()
    lines = _source_body(cell.excerpt).splitlines()
    found: list[_StructuredReportSnapshot] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = index
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            index += 1
        block = lines[start:index]
        if len(block) < 3 or not _is_table_rule(_table_cells(block[1])):
            continue
        header = _table_cells(block[0])
        compact = [_compact_key(value) for value in header]
        periods = [
            offset for offset, value in enumerate(compact)
            if re.match(r"(?:제\d+기|20\d{2}년)", value)
        ]
        if not periods:
            continue
        explicit = [
            offset for offset in periods
            if re.search(rf"(?<!\d){report_year}(?!\d)", header[offset])
        ]
        if (require_explicit_year and cell.restated_comparatives
                and len(explicit) > 1):
            # Restated product tables commonly repeat each year for amount
            # and share.  Accept only one explicitly named monetary column;
            # never infer an amount from the position of ambiguous columns.
            monetary = [offset for offset in explicit
                        if (not any(marker in compact[offset]
                                    for marker in ("비중", "비율", "%"))
                            and any(marker in compact[offset]
                                    for marker in ("금액", "매출액", "수익액")))]
            excluded = [offset for offset in explicit if offset not in monetary]
            if (len(monetary) == 1 and all(
                    any(marker in compact[offset]
                        for marker in ("비중", "비율", "%"))
                    for offset in excluded)):
                explicit = monetary
        if require_explicit_year and len(explicit) != 1:
            continue
        value_pos = explicit[0] if len(explicit) == 1 else periods[0]
        identity_positions = [
            offset for offset, value in enumerate(compact[:periods[0]])
            if any(marker in value for marker in (
                "부문", "품목", "제품", "서비스", "구분", "항목", "유형"))
        ]
        if not identity_positions:
            continue
        unit = _local_table_currency_unit(lines, start)
        if unit not in {"억원", "백만원", "천원", "원", "조원"}:
            continue
        rows: list[tuple[str, str, Decimal]] = []
        identity_keys: list[str] = []
        for row_line in block[2:]:
            row = _table_cells(row_line)
            if (len(row) != len(header) or _is_table_rule(row)
                    or value_pos >= len(row)):
                continue
            # Key the row on the source's primary axis where that column
            # already separates the rows.  A composite key folds in the
            # product-description column, which annual reports reword (for
            # example ``컴퓨터`` becoming ``PC``); those segments then fail to
            # pair across reports and silently drop out of the comparison.
            label = _segment_axis_label(
                row, identity_positions, block[2:], header, topic=topic)
            value = _table_number(row[value_pos], unit)
            if label and value is not None:
                rows.append((label, row[value_pos], value))
                identity_keys.append(_structured_row_key(label))
        if not rows:
            continue
        nearest_heading = _table_axis_heading(lines, start, topic=topic)
        found.append(_StructuredReportSnapshot(
            table_label=nearest_heading,
            signature=tuple(compact[offset] for offset in identity_positions),
            period_header=header[value_pos], report_year=report_year, unit=unit,
            rows=tuple(rows),
            unique_identity=len(identity_keys) == len(set(identity_keys))))
    return tuple(found)


def _structured_cross_cell_deltas(
        cells: tuple["MatrixCellEvidence", ...], *, topic: str,
        ) -> tuple[_StructuredPeriodDelta, ...]:
    """Match identical composite row coordinates across requested reports."""

    if len(cells) < 2:
        return ()
    if _has_reclassified_comparatives(cells):
        # A latest report can reclassify historical segment values even when
        # the row labels have not changed.  Use both explicitly named years
        # from that report or decline the arithmetic; never revive old-basis
        # values through the direct-metric fallback.
        return _structured_explicit_year_fallback(cells, topic=topic)
    before_cell, after_cell = cells[0], cells[-1]
    before = _structured_report_snapshots(before_cell, topic=topic)
    after = _structured_report_snapshots(after_cell, topic=topic)
    paired: list[tuple[_StructuredReportSnapshot, _StructuredReportSnapshot]] = []
    used_before: set[int] = set()
    used_after: set[int] = set()

    # Exact header-role matches are the primary contract.  Repeated table
    # shapes (commonly 연결/별도) remain ordered within the source block.
    for before_index, prior in enumerate(before):
        latest_index = next((
            index for index, latest in enumerate(after)
            if index not in used_after and latest.signature == prior.signature
        ), None)
        if latest_index is None:
            continue
        paired.append((prior, after[latest_index]))
        used_before.add(before_index)
        used_after.add(latest_index)

    # Some annual templates rename only an identity header (e.g. 품목 ->
    # 매출유형) while retaining the same ordered composite row coordinates.
    # Join that drift only when both tables have unique row identities, the
    # identity-column count is unchanged, and exactly one unmatched candidate
    # has the same complete row-key set.  Any duplicate or ambiguity remains
    # fail-closed.
    for before_index, prior in enumerate(before):
        if before_index in used_before or not prior.unique_identity:
            continue
        prior_keys = {
            _structured_row_key(label) for label, _raw, _value in prior.rows}
        candidates = [
            (after_index, latest) for after_index, latest in enumerate(after)
            if (after_index not in used_after and latest.unique_identity
                and len(latest.signature) == len(prior.signature)
                and len(prior_keys & {
                    _structured_row_key(label)
                    for label, _raw, _value in latest.rows}) >= 2
                and len(prior_keys & {
                    _structured_row_key(label)
                    for label, _raw, _value in latest.rows})
                >= min(len(prior_keys), len(latest.rows)) * 3 / 4)
        ]
        if len(candidates) != 1:
            continue
        latest_index, latest = candidates[0]
        paired.append((prior, latest))
        used_before.add(before_index)
        used_after.add(latest_index)

    results: list[_StructuredPeriodDelta] = []
    for prior_table, latest_table in paired:
        if (prior_table.unit != latest_table.unit
                or not prior_table.unique_identity
                or not latest_table.unique_identity):
            continue
        prior_rows = {
            _structured_row_key(label): (label, raw, value)
            for label, raw, value in prior_table.rows}
        latest_rows = {
            _structured_row_key(label): (label, raw, value)
            for label, raw, value in latest_table.rows}
        changes: list[tuple[str, str, str, Decimal, str]] = []
        for key in prior_rows.keys() & latest_rows.keys():
            prior_label, prior_raw, prior_value = prior_rows[key]
            latest_label, latest_raw, latest_value = latest_rows[key]
            if prior_value == latest_value:
                continue
            label = latest_label or prior_label
            changes.append((label, prior_raw, latest_raw,
                            latest_value - prior_value, latest_table.unit))
        if not changes:
            continue
        results.append(_StructuredPeriodDelta(
            table_label=latest_table.table_label,
            period_label=(f"{prior_table.report_year}→{latest_table.report_year} "
                          f"({_public_column_name(prior_table.period_header)}"
                           f"→{_public_column_name(latest_table.period_header)})"),
            rows=tuple(changes), largest_label=None))
    return tuple(results)


def _structured_explicit_year_fallback(
        cells: tuple["MatrixCellEvidence", ...], *, topic: str,
        ) -> tuple[_StructuredPeriodDelta, ...]:
    """Compare requested years inside one verified table only when explicit.

    This fallback never maps fiscal-number headers to years.  It applies only
    when the latest requested cell's source table independently labels both
    requested calendar years, with the same unique composite row identity and
    unit.  A title-only older cell can therefore be supplemented without
    inventing a cross-document join.
    """

    if len(cells) < 2:
        return ()
    before_year = _cell_report_year(cells[0])
    after_year = _cell_report_year(cells[-1])
    if before_year is None or after_year is None or before_year == after_year:
        return ()
    source_cell = cells[-1]
    before = _structured_report_snapshots(
        source_cell, topic=topic, target_year=before_year,
        require_explicit_year=True)
    after = _structured_report_snapshots(
        source_cell, topic=topic, target_year=after_year,
        require_explicit_year=True)
    results: list[_StructuredPeriodDelta] = []
    used_after: set[int] = set()
    for prior in before:
        latest_index = next((
            index for index, latest in enumerate(after)
            if index not in used_after and latest.signature == prior.signature
        ), None)
        if latest_index is None:
            continue
        latest = after[latest_index]
        used_after.add(latest_index)
        if (prior.unit != latest.unit or not prior.unique_identity
                or not latest.unique_identity):
            continue
        prior_rows = {
            _structured_row_key(label): (label, raw, value)
            for label, raw, value in prior.rows}
        latest_rows = {
            _structured_row_key(label): (label, raw, value)
            for label, raw, value in latest.rows}
        if prior_rows.keys() != latest_rows.keys():
            continue
        changes = []
        for key in prior_rows:
            prior_label, prior_raw, prior_value = prior_rows[key]
            latest_label, latest_raw, latest_value = latest_rows[key]
            if prior_value == latest_value:
                continue
            changes.append((latest_label or prior_label, prior_raw, latest_raw,
                            latest_value - prior_value, latest.unit))
        if changes:
            results.append(_StructuredPeriodDelta(
                table_label=latest.table_label,
                period_label=(f"{before_year}→{after_year} "
                              f"({_public_column_name(prior.period_header)}"
                               f"→{_public_column_name(latest.period_header)})"),
                rows=tuple(changes), largest_label=None))
    return tuple(results)


def _structured_period_contrast(
        evidence: "MatrixCellEvidence | tuple[MatrixCellEvidence, ...]", *, topic: str,
        ) -> str:
    parts: list[str] = []
    if isinstance(evidence, tuple):
        tables = _structured_cross_cell_deltas(evidence, topic=topic)
        if not tables:
            tables = _structured_explicit_year_fallback(evidence, topic=topic)
        # Never fall back to the adjacent columns of only the latest report.
        # In a requested 2023→2025 comparison that silently answers 2024→2025
        # and can make a within-report YoY movement look like the requested
        # endpoint delta.  Without two aligned report snapshots (or explicit
        # endpoint-year columns in one verified table), fail closed.
    else:
        tables = _structured_period_deltas(evidence, topic=topic)

    def table_score(table: _StructuredPeriodDelta) -> int:
        label = _compact_key(table.table_label)
        axis = _topic_axis(topic)
        if axis == "products_services":
            score = 0
            if any(marker in label for marker in (
                    "주요제품", "제품및서비스", "제품군별", "사업부문별주요현황")):
                score += 120
            elif any(marker in label for marker in ("제품", "서비스", "품목")):
                score += 70
            if any(marker in label for marker in (
                    "프로젝트", "주석", "매출실적", "매출및수주")):
                score -= 140
            return score
        if axis == "revenue_mix":
            score = 0
            if any(marker in label for marker in (
                    "매출실적", "매출구성", "수익구조", "영업수익")):
                score += 120
            elif any(marker in label for marker in ("매출", "수익")):
                score += 60
            if any(marker in label for marker in ("프로젝트", "주석")):
                score -= 140
            return score
        return 0

    ordered = sorted(enumerate(tables),
                     key=lambda item: (-table_score(item[1]), item[0]))
    selected = [table for _index, table in ordered[:1]]
    if _topic_axis(topic) == "revenue_mix" and selected:
        # 연결/별도는 서로 다른 reporting scopes, not duplicate tables.
        # Keep at most that explicit pair; every other topic gets one table.
        first_scope = _compact_key(selected[0].table_label)
        if any(scope in first_scope for scope in ("연결", "별도")):
            counterpart = next((
                table for _index, table in ordered[1:]
                if (any(scope in _compact_key(table.table_label)
                        for scope in ("연결", "별도"))
                    and (("연결" in first_scope)
                         != ("연결" in _compact_key(table.table_label))))
            ), None)
            if counterpart is not None:
                selected.append(counterpart)

    for table in selected:
        # One aggregate plus the two largest material row movements answers a
        # period-comparison question without reproducing the entire source
        # table.  Selection is data-driven and bounded for every issuer.
        unique_rows: dict[str, tuple[str, str, str, Decimal, str]] = {}
        for row in table.rows:
            prior = unique_rows.get(row[0])
            if prior is None or abs(row[3]) > abs(prior[3]):
                unique_rows[row[0]] = row
        rows = list(unique_rows.values())
        aggregate = sorted(
            (row for row in rows if any(
                marker in _compact_key(row[0])
                for marker in ("합계", "총계"))),
            key=lambda row: (
                sum(marker in _compact_key(row[0])
                    for marker in ("합계", "총계")),
                abs(row[3])),
            reverse=True,
        )[:1]
        ranked = sorted(
            (row for row in rows if row not in aggregate
             and not any(marker in _compact_key(row[0])
                         for marker in (
                             "합계", "총계", "소계", "내부거래", "조정", "제거"))),
            key=lambda row: abs(row[3]), reverse=True,
        )
        # A plain segment axis is normally four to six rows (DX/DS/SDC/
        # Harman) and showing only the two largest reads as if the others had
        # not changed.  Render such an axis whole.  Composite rows (``부문 /
        # 매출유형 / 품목``) are a product breakdown, not an axis, so they keep
        # the bounded top-two and the answer cannot become a row dump.
        simple_axis = all(" / " not in row[0] for row in ranked)
        material = (ranked if simple_axis and len(ranked) <= _WHOLE_AXIS_ROWS
                    else ranked[:2])
        displayed = [*aggregate, *material]
        split_labels = [row[0].split(" / ") for row in displayed]
        common_prefix: list[str] = []
        if len(split_labels) >= 2:
            for values in zip(*split_labels, strict=False):
                if len(set(values)) != 1:
                    break
                common_prefix.append(values[0])
        row_parts = []
        for label, prior, latest, delta, unit in displayed:
            direction = "증가" if delta > 0 else "감소"
            parts_label = label.split(" / ")
            if common_prefix and parts_label[:len(common_prefix)] == common_prefix:
                label = " / ".join(parts_label[len(common_prefix):]) or "합계"
            row_parts.append(
                f"{label} {_structured_value_display(prior, unit)}"
                f"→{_structured_value_display(latest, unit)} "
                f"({_structured_delta_display(delta, unit)} {direction})")
        heading = table.table_label
        if common_prefix:
            heading += " [" + " / ".join(common_prefix) + "]"
        basis = (" (최신 보고서에서 다시 분류한 수치 기준)"
                 if isinstance(evidence, tuple)
                 and _has_reclassified_comparatives(evidence) else "")
        summary = f"{heading} {table.period_label}{basis}: " + ", ".join(row_parts)
        shown_positive = [row for row in material if row[3] > 0]
        if shown_positive:
            largest = max(shown_positive, key=lambda row: row[3])[0]
            if common_prefix:
                label_parts = largest.split(" / ")
                if label_parts[:len(common_prefix)] == common_prefix:
                    largest = " / ".join(label_parts[len(common_prefix):]) or "합계"
            summary += f"; 표시한 세부 행 중 증가폭이 가장 큰 항목은 {largest}입니다"
        parts.append(summary)
    return ". ".join(parts)


def _is_non_endpoint_comparison_clause(
        value: str, *, before_year: int | None, after_year: int | None,
        ) -> bool:
    """Reject a comparison whose base is not the requested first endpoint.

    A 2025 filing may state ``전년 대비`` or ``2024년 대비``.  That is a
    valid 2024→2025 observation, but it must not be appended to an exact
    2023→2025 answer as though it used the same endpoints.  A genuine dated
    event such as ``2024년 신규 사업 진출`` has no comparison base and is
    intentionally unaffected.
    """

    if before_year is None or after_year is None:
        return True
    compact = _compact_key(value)
    span = after_year - before_year
    relative_base = any(marker in compact for marker in (
        "전년대비", "전년동기대비", "전기대비", "직전연도대비"))
    if relative_base and span > 1:
        return True
    explicit_bases: list[int] = []
    explicit_base = re.compile(
        r"(?P<year>(?<!\d)20\d{2}년|(?<![\d제])['’‘]?\d{2}년)"
        r"\s*(?:동기\s*)?대비")
    for match in explicit_base.finditer(value):
        local_years = _years_in_text(match.group("year"))
        if local_years:
            explicit_bases.append(local_years[-1])
    return any(year != before_year for year in explicit_bases)


def _explicit_change_clauses(text: str, *, limit: int = 2) -> tuple[str, ...]:
    """Return clauses where the disclosure itself explicitly states change."""

    prose = " ".join(line.strip() for line in _source_body(text).splitlines()
                     if line.strip() and not line.lstrip().startswith("|"))
    found = []
    seen_material: set[str] = set()

    def add_if_new(clean: str) -> None:
        """Ignore a duplicate that differs only by a section heading."""

        first_year = re.search(
            r"(?<!\d)(?:20\d{2}|['’‘]?\d{2})년", clean)
        material = clean[first_year.start():] if first_year else clean
        key = re.sub(r"\s+", "", material)
        if key in seen_material:
            return
        seen_material.add(key)
        found.append(clean)
    focused_patterns = (
        re.compile(
            r"(?:['’‘]?\d{2}년|20\d{2}년)[^.!?]{0,260}?"
            r"(?:변경|증가|감소|상승|하락|약보합|재변경)[^.!?]{0,30}(?:[.!?]|$)"),
        re.compile(
            r"(?:조직개편|기존)[^\n]{0,260}?"
            r"(?:재변경하였습니다|변경하였습니다)"),
    )
    for pattern in focused_patterns:
        for match in pattern.finditer(prose):
            clean = " ".join(match.group(0).split()).strip()
            if clean:
                add_if_new(clean)
            if len(found) >= limit:
                return tuple(found)
    for clause in re.split(r"(?<=[.!?])\s+|(?<=다\.)", prose):
        clean = " ".join(clause.split()).strip()
        if (not clean or len(clean) > 360
                or not any(marker in clean for marker in (
                    "변경", "증가", "감소", "상승", "하락", "약보합", "재변경"))):
            continue
        add_if_new(clean)
        if len(found) >= limit:
            break
    return tuple(found)


def _material_business_change_clauses(
        text: str, *, limit: int = 2,
        ) -> tuple[str, ...]:
    """Keep source-stated changes to operating scope, not accounting noise.

    Price trends, revenue-recognition methods and reissued audit bases can
    differ between reports without representing a change in the company's
    core business.  A business-change conclusion therefore requires both an
    operating object and a scope-changing action in the same complete source
    sentence.  The function preserves the literal sentence and does not infer
    that an unmentioned product was discontinued.
    """

    prose = " ".join(line.strip() for line in _source_body(text).splitlines()
                     if line.strip() and not line.lstrip().startswith("|"))
    excluded = (
        "회계처리", "인식시점", "총액법", "순액법", "판매가격",
        "가격변동", "감사보고서", "재발행", "영업실적",
        "중단영업손익", "계속사업기준",
        # 판매망 확대 같은 유통 전략은 제품·사업 범위의 확대가 아니다.
        # 이 절을 주요 사업 변화로 올리면 판매방법 전문이 결론이 된다.
        "판매경로", "판매 경로", "판매방법", "판매 방법",
        "판매전략", "판매 전략", "결제조건", "결제 조건",
    )
    business = (
        "사업", "부문", "제품", "서비스", "플랫폼", "솔루션", "시장",
        "생산시설", "판매망",
    )
    change = (
        "매각", "인수", "신규", "출시", "종료", "중단", "철수", "진출",
        "확대", "축소", "강화", "전환", "개편", "통합", "분할",
    )
    found: list[str] = []
    for sentence in re.split(
            r"(?<=[.!?])(?:\s+|(?=[가-힣A-Za-z㈜]))", prose):
        clean = " ".join(sentence.split()).strip()
        clean = re.sub(
            r"^(?:\(\s*단위\s*[:：][^)]*\)\s*)?"
            r"(?:참고\s*\d*\s*[:：]\s*)?", "", clean)
        compact = _compact_key(clean)
        if (not clean or len(clean) > 520
                or any(marker in compact for marker in excluded)
                or not (_years_in_text(clean)
                        or re.search(r"제\s*\d+\s*기", clean)
                        or any(marker in compact for marker in (
                            "당기", "전기", "전년", "해당연도", "금년", "올해")))
                or not any(marker in compact for marker in business)
                or not any(marker in compact for marker in change)):
            continue
        if clean not in found:
            found.append(clean)
        if len(found) >= limit:
            break
    return tuple(found)


def _business_structure_clause(text: str) -> str:
    """Return one explicit operating-structure declaration, never a heading.

    Annual reports may restate their business axes without using a change
    verb.  A complete sentence that explicitly says how the business is
    divided or operated is still useful for a before/after comparison.  Bare
    section titles and product-table labels are not accepted, and the caller
    presents differing clauses as disclosure wording rather than inferring a
    disposal, launch or other material event.
    """

    prose = " ".join(line.strip() for line in _source_body(text).splitlines()
                     if line.strip() and not line.lstrip().startswith("|"))
    structural = (
        "사업을 구분", "사업부문으로 구분", "영역으로 사업을 운영",
        "영역에서 사업을", "주요 사업으로", "사업을 영위",
    )
    for sentence in re.split(
            r"(?<=[.!?])(?:\s+|(?=[가-힣A-Za-z㈜]))", prose):
        clean = " ".join(sentence.split()).strip()
        compact = _compact_key(clean)
        if (24 <= len(clean) <= 560
                and (_names_segment_axis(clean)
                     or any(_compact_key(marker) in compact
                            for marker in structural))
                and any(marker in compact for marker in (
                    "영위", "운영", "구분", "구성", "전개", "제조", "공급", "제공"))):
            return clean
    return ""


def _years_in_text(value: str) -> tuple[int, ...]:
    """Extract modern filing years, including Korean two-digit shorthand.

    ``'25년``/``’25년`` are common disclosure surfaces for 2025.  A bare
    duration such as ``25년간`` or ``25년 동안`` is not a calendar year and
    must not make an old/undated event pass the requested-window filter.
    """

    found = [
        int(year) for year in re.findall(r"(?<!\d)(20\d{2})(?!\d)", value)]
    shorthand = re.compile(
        r"(?<![\d제])['’‘]?(?P<year>\d{2})년"
        r"(?!\s*(?:간|동안|째|차|이상|미만))")
    for match in shorthand.finditer(value):
        year = 2000 + int(match.group("year"))
        if year not in found:
            found.append(year)
    return tuple(found)


def _inside_requested_window(
        value: str, *, before_year: int | None, after_year: int | None,
        include_before: bool,
        ) -> bool:
    """Admit a dated clause only when every stated year is in the window."""

    if before_year is None or after_year is None:
        return False
    years = _years_in_text(value)
    if not years:
        return False
    lower = before_year if include_before else before_year + 1
    return all(lower <= year <= after_year for year in years)


def _inside_requested_window_or_period_relative(
        value: str, *, before_year: int | None, after_year: int | None,
        source_year: int | None, include_before: bool,
        ) -> bool:
    """Bind ``당기`` change clauses to their typed report-period cell.

    A latest annual report often says ``당기 중 ... 매각`` without repeating
    the calendar year.  The cell already supplies that year, so rejecting the
    clause as undated drops a directly stated endpoint change.  Explicit
    years still use the stricter window check and can never be overridden by
    a relative-period word.
    """

    if _years_in_text(value):
        return _inside_requested_window(
            value, before_year=before_year, after_year=after_year,
            include_before=include_before)
    if (before_year is None or after_year is None or source_year is None
            or not any(marker in _compact_key(value)
                       for marker in ("당기중", "당기", "금년", "올해", "해당연도"))):
        return False
    lower = before_year if include_before else before_year + 1
    return lower <= source_year <= after_year


def _business_structure_contrast(
        cells: tuple["MatrixCellEvidence", ...], *,
        before_year: int | None = None, after_year: int | None = None,
        ) -> str:
    if len(cells) < 2:
        return ""
    before = _business_structure_clause(cells[0].excerpt)
    after = _business_structure_clause(cells[-1].excerpt)
    if (not before or not after
            or _compact_key(before).casefold() == _compact_key(after).casefold()):
        return ""
    # A present-day structure sentence can embed an old reorganisation date.
    # Keep endpoint statements and undated structure, but do not expose an
    # event outside the requested report window as the compared change.
    for value in (before, after):
        if (_years_in_text(value)
                and not _inside_requested_window(
                    value, before_year=before_year, after_year=after_year,
                    include_before=True)):
            return ""
    # 두 절 모두 완결된 문장이라 그대로 이으면 「… 있습니다.; POSCO홀딩스」가
    # 된다.  마침표와 세미콜론이 겹치는 자리에서는 세미콜론만 남긴다.
    return ("공시된 사업 구조·영위 범위는 "
            f"{cells[0].label}: {before.rstrip('.')}; "
            f"{cells[-1].label}: {after}")


@dataclass(frozen=True)
class MatrixCellEvidence:
    cell_id: str
    label: str
    topic: str
    excerpt: str
    receipts: tuple[str, ...]
    restated_comparatives: bool = False


@dataclass(frozen=True)
class NarrativeMatrix:
    cells: tuple[MatrixCellEvidence, ...]
    limitations: tuple[str, ...]
    comparison_kind: str
    incomplete: bool = False

    @property
    def has_identical_excerpts(self) -> bool:
        companies = {
            cell.label.partition(" · ")[0] for cell in self.cells}
        return (not self.incomplete and len(self.cells) >= 2
                and len(companies) == 1
                and len({re.sub(r"\s+", " ", cell.excerpt).strip() for cell in self.cells}) == 1)

    @property
    def has_structured_table(self) -> bool:
        """A table is safer to reproduce than to ask HCX to remap its columns."""

        return any(re.search(r"\|\s*:?-{3,}", cell.excerpt) for cell in self.cells)

    @classmethod
    def from_payload(cls, payload) -> "NarrativeMatrix | None":
        """Build only from a typed multi-cell fanout sidecar.

        One-cell narratives intentionally remain on the existing long-form
        guard.  A malformed/incomplete sidecar is not guessed at: its normal
        template route remains safer than a synthetic comparison.
        """
        claims_by_id = {claim.output_id: claim for claim in payload.claims}
        cells: list[MatrixCellEvidence] = []
        corp_names, periods = set(), set()
        found_narrative = False
        incomplete = False
        bindings_by_coordinate: dict[tuple[object, ...], tuple[str, ...]] = {}
        coordinate_by_claim: dict[str, tuple[object, ...]] = {}
        planned_coordinates: set[tuple[object, ...]] = set()
        for result in getattr(payload, "narrative_sidecars", ()):
            plan = getattr(result, "plan", None)
            plan_cells = tuple(getattr(plan, "cells", ()) or ())
            if not plan_cells:
                continue
            found_narrative = True
            incomplete = incomplete or bool(getattr(result, "missing_cells", ()) or
                                            getattr(plan, "omitted_cells", 0))
            binding_rows = tuple(getattr(result, "cell_claim_output_ids", ()) or ())
            if len({cell_id for cell_id, _ in binding_rows}) != len(binding_rows):
                return None
            ids_by_cell = dict(binding_rows)
            completed = set(getattr(result, "completed_cells", ()) or ())
            plan_by_id = {coordinate.cell_id: coordinate for coordinate in plan_cells}
            if len(plan_by_id) != len(plan_cells) or not completed <= set(plan_by_id):
                return None
            for coordinate in plan_cells:
                planned_coordinates.add(_coordinate_signature(coordinate))
                if coordinate.cell_id not in completed:
                    continue
                output_ids = tuple(sorted(ids_by_cell.get(coordinate.cell_id, ())))
                if not output_ids:
                    return None
                signature = _coordinate_signature(coordinate)
                # A claim must never be attributed to two company/period/topic
                # coordinates.  This is fail-closed instead of guessing which
                # cell a repeated output belongs to.
                for output_id in output_ids:
                    previous = coordinate_by_claim.setdefault(output_id, signature)
                    if previous != signature:
                        return None
                previous_binding = bindings_by_coordinate.get(signature)
                if previous_binding is not None:
                    if previous_binding != output_ids:
                        return None
                    continue  # exact duplicate coordinate; retain one cell
                linked = [claims_by_id[output_id] for output_id in output_ids
                          if output_id in claims_by_id]
                receipts = _receipt_numbers(linked)
                # A per-cell sentence without its own receipt is not allowed.
                if len(linked) != len(output_ids) or not receipts:
                    return None
                expected_receipt = getattr(
                    getattr(coordinate, "document_selector", None), "rcept_no", None)
                if expected_receipt and set(receipts) != {expected_receipt}:
                    # Claim IDs are private execution metadata, but a model or
                    # merge bug can still bind a sibling cell's claim to this
                    # coordinate.  An exact document selector makes that
                    # mismatch decidable, so fail closed instead of composing
                    # a cross-company/cross-receipt comparison.
                    return None
                # The claim body is the independently round-tripped, bounded
                # prompt-safe structure.  Citation excerpts are deliberately
                # short display snippets and can stop before a table, which
                # previously erased product rows and reconciliation rows from
                # an otherwise verified narrative matrix.
                excerpts = _dedupe_same_source_excerpts(linked)
                excerpt = " / ".join(part for part in excerpts if part)
                if not excerpt:
                    return None
                if not _narrative_excerpt_is_adequate(
                        excerpt, topic=coordinate.topic):
                    incomplete = True
                coordinate_id = f"{getattr(plan, 'task_id', 'task')}/{coordinate.cell_id}"
                public_period = _public_coordinate_period_label(coordinate)
                label = f"{coordinate.corp_name} · {public_period}"
                cells.append(MatrixCellEvidence(
                    cell_id=coordinate_id, label=label, topic=coordinate.topic,
                    excerpt=excerpt, receipts=receipts))
                bindings_by_coordinate[signature] = output_ids
                corp_names.add(coordinate.corp_name)
                periods.add(public_period)
        # A single narrative task/cell remains on the legacy long-form path.
        # Multiple independent task sidecars are a matrix as well.
        if not found_narrative or len(planned_coordinates) < 2 or not cells:
            return None
        # Collapse only enumerated aliases of one typed disclosure axis after
        # each claim has independently passed source round-trip verification.
        deduped: dict[tuple[str, str, str], MatrixCellEvidence] = {}
        for cell in cells:
            company, _, period = cell.label.partition(" · ")
            key = (company, period, _topic_axis(cell.topic))
            previous = deduped.get(key)
            if previous is None:
                deduped[key] = MatrixCellEvidence(
                    cell.cell_id, cell.label, _topic_display(cell.topic),
                    cell.excerpt, cell.receipts)
                continue
            old = re.sub(r"\s+", " ", previous.excerpt).strip()
            new = re.sub(r"\s+", " ", cell.excerpt).strip()
            excerpt = (previous.excerpt if new in old else cell.excerpt
                       if old in new else f"{previous.excerpt} / {cell.excerpt}")
            deduped[key] = MatrixCellEvidence(
                previous.cell_id, previous.label, previous.topic, excerpt,
                tuple(dict.fromkeys((*previous.receipts, *cell.receipts))))
        cells = list(deduped.values())
        if len(corp_names) > 1:
            kind = "회사별 비교"
        elif len(periods) > 1:
            kind = "기간별 변화 비교"
        else:
            kind = "항목별 요약"
        public_limitations = [
            limitation for limitation in payload.limitations
            if limitation.code.split(":", 1)[0] != "source_cross_check_partial"
        ]
        notes = list(render_safe_limitations(public_limitations))
        for limitation in public_limitations:
            if limitation.code in _NARRATIVE_LIMITS and limitation.detail not in notes:
                notes.append(limitation.detail)
        return cls(cells=tuple(cells), limitations=tuple(notes), comparison_kind=kind,
                   incomplete=incomplete)

    @property
    def receipts(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(receipt for cell in self.cells for receipt in cell.receipts))

    @property
    def comparison_labels(self) -> tuple[str, ...]:
        if self.comparison_kind == "회사별 비교":
            values = (cell.label.split(" · ", 1)[0] for cell in self.cells)
        else:
            return ()
        return tuple(dict.fromkeys(values))

    def prompt(self, question: str | None = None, *, require_delta: bool = False) -> str:
        lines = []
        if question:
            lines.append(f"[질문] {_clean(question, 500)}")
        lines += [
            "[typed narrative matrix]",
            "아래 각 셀은 source_roundtrip으로 확인한 인용 원문 발췌와 접수번호입니다.",
        ]
        for cell in self.cells:
            lines.append(f"- [{cell.cell_id}] {cell.label} | 주제: {cell.topic} | "
                         f"본문: {cell.excerpt} | 근거: {', '.join(cell.receipts)}")
        if self.limitations:
            lines.append("[한계]")
            lines.extend(f"- {note}" for note in self.limitations)
        exact_ids = ", ".join(f"[{cell.cell_id}]" for cell in self.cells)
        compared_labels = ", ".join(self.comparison_labels)
        lines += [
            "[작성 형식]",
            "각 셀을 '- [cell-id] ... (접수번호 ...)' 한 문장으로 짧게 요약하고, 그 줄에 해당 셀의 모든 접수번호를 붙이십시오.",
            f"cell-id는 줄이거나 다시 번호를 매기지 말고 다음 값을 정확히 한 번씩 쓰십시오: {exact_ids}",
            "'주요 사업' 셀은 실적 수치만 고르지 말고 영위 사업·사업부문·제품/서비스를, '수익 구조' 셀은 수익 항목·구성·비중을, '자회사 구성' 셀은 자회사·계열사와 업종을 우선하십시오.",
            f"마지막에 '{self.comparison_kind} 결론:' 한 문장으로 직접 확인되는 공통점·차이·변화만 말하십시오.",
            *((
                f"결론에는 비교 대상 {compared_labels}을 모두 직접 언급하고, 결론 문장에도 비교에 사용한 접수번호를 붙이십시오.",
            ) if compared_labels else (
                "결론 문장에도 비교에 사용한 접수번호를 붙이십시오.",
            )),
            "직접 확인되지 않으면 '제시된 인용 범위만으로는 차이의 내용을 확정하지 않습니다'라고 쓰십시오.",
            "본문·접수번호에 없는 숫자, 사실, 계산은 절대 추가하지 마십시오. 한계는 그대로 답변 끝에 쓰십시오.",
        ]
        if require_delta:
            lines.append(
                "이번에는 완전한 비교 셀의 본문이 서로 다르므로 위의 '확정하지 않습니다' 문장을 쓰지 마십시오. "
                "두 셀의 인용 본문에서 실제로 달라진 사업·제품·계획·수치 중 하나를 짧게 대조해 결론에 쓰고 양쪽 접수번호를 붙이십시오."
            )
        return "\n".join(lines)

    def _public_temporal_business_overview(self) -> str | None:
        """Render one-company, multi-axis annual comparisons for readers.

        The private matrix still carries every company/period/topic cell and
        the verifier checks the complete model-facing answer. A public answer
        to a broad business comparison, however, should not repeat six source
        blocks or promote a spelling difference above the numeric change.
        This projection uses only the typed cells and existing deterministic
        comparators, and cites every report period once.
        """

        companies = {
            cell.label.partition(" · ")[0] for cell in self.cells}
        periods = {
            cell.label.partition(" · ")[2] for cell in self.cells}
        groups: dict[str, list[MatrixCellEvidence]] = {}
        for cell in self.cells:
            groups.setdefault(_topic_axis(cell.topic), []).append(cell)
        business_cells = [
            cell for cell in self.cells
            if _is_main_business_topic(cell.topic)]
        product_cells = groups.get("products_services", [])
        revenue_cells = groups.get("revenue_mix", [])
        if (len(companies) != 1 or len(periods) < 2
                or len(business_cells) < 2
                or len(product_cells) < 2
                or len(revenue_cells) < 2):
            return None

        def ordered(cells):
            return tuple(sorted(
                cells, key=lambda cell: cell.label.partition(" · ")[2]))

        revenue = _structured_period_contrast(
            ordered(revenue_cells), topic="매출구성")
        if not revenue:
            revenue = _metric_contrast(
                ordered(revenue_cells), topic="매출구성")
        products = _term_contrast(
            ordered(product_cells), include_unique=True,
            temporal_business_change=True)
        latest_business = ordered(business_cells)[-1]
        business_changes = _material_business_change_clauses(
            latest_business.excerpt, limit=2)
        business = (
            " ".join(business_changes)
            if business_changes else
            "두 보고서의 확인 범위에는 매각·인수·출시·사업 확대·축소를 "
            "직접 밝힌 내용이 없어, 실제 핵심 사업이 바뀌었다고 단정하지 않습니다."
        )
        if not revenue or not products:
            return None

        lines = [
            "핵심 변화:",
            f"- 매출 구성: {revenue.rstrip('.')}.",
            f"- 주요 제품 및 서비스: {products.rstrip('.')}.",
            f"- 사업부문: {business.rstrip('.')}.",
            "근거 공시:",
        ]
        by_period: dict[str, list[str]] = {}
        for cell in self.cells:
            period = cell.label.partition(" · ")[2]
            by_period.setdefault(period, []).extend(cell.receipts)
        for period in sorted(by_period):
            receipts = "; ".join(
                "접수번호 " + receipt
                for receipt in dict.fromkeys(by_period[period]))
            lines.append(f"- {period}: {receipts}")
        if self.limitations:
            rendered = append_public_qualifications(
                "\n".join(lines), self.limitations)
            lines = rendered.splitlines()
        return _public_narrative_surface("\n".join(lines))

    def public_answer(self, text: str) -> str:
        """Project a verified matrix answer onto its public display contract.

        Cell IDs remain mandatory in the HCX prompt and verifier.  They are
        execution provenance, however, so they are removed only after those
        checks.  The model-facing evidence lines are also deliberately not
        replayed verbatim: they can contain a whole disclosure table even
        after the comparison itself has passed verification.  Re-rendering
        one bounded, typed digest per coordinate keeps the public answer
        auditable without turning a comparison into a raw-source dump.

        This is a display projection only.  ``text`` is verified before this
        method is called; each replacement row is regenerated from the same
        source-roundtripped cell and carries that cell's receipt(s).
        """

        temporal_overview = self._public_temporal_business_overview()
        if temporal_overview is not None:
            return temporal_overview

        topic_axes = {_topic_axis(cell.topic) for cell in self.cells}
        has_business_axis = any(
            _is_main_business_topic(cell.topic) for cell in self.cells)
        if (self.comparison_kind == "항목별 요약"
                and self.has_identical_excerpts
                and has_business_axis
                and "products_services" in topic_axes):
            # A broad one-company business question often maps both requested
            # axes to the same verified overview paragraph. Printing the
            # paragraph twice plus a meta conclusion is technically complete
            # but poor for a general reader. Render the shared operating
            # sentence once and name the covered axes in the heading.
            source = next(
                cell for cell in self.cells
                if _is_main_business_topic(cell.topic))
            digest = _compact_cell_text(
                source.excerpt, topic=source.topic, limit=320)
            # The ranked business digest can contain a second operating-like
            # sentence about offices or subsidiaries. A broad "what does the
            # company do" answer is complete after the first source sentence;
            # keep that boundary instead of turning the summary into an
            # organisation inventory.
            first_sentence = re.split(r"(?<=[.!?])\s+", digest, maxsplit=1)[0]
            if first_sentence:
                digest = first_sentence
            topics = "·".join(dict.fromkeys(
                _topic_display(cell.topic) for cell in self.cells))
            receipts = "; ".join(
                "접수번호 " + receipt for receipt in self.receipts)
            answer = f"사업 내용 요약({topics}): {digest} (근거: {receipts})"
            if self.limitations:
                answer = append_public_qualifications(
                    answer, self.limitations)
            return _public_narrative_surface(answer)

        display_by_id: dict[str, str] = {}
        cell_by_reference: dict[str, str] = {}
        short_ids: dict[str, list[tuple[str, str]]] = {}
        for cell in self.cells:
            display = _coordinate_heading(cell.label, cell.topic)
            display_by_id[cell.cell_id] = display
            display_by_id[f"matrix/{cell.cell_id}"] = display
            cell_by_reference[cell.cell_id] = cell.cell_id
            cell_by_reference[f"matrix/{cell.cell_id}"] = cell.cell_id
            short_ids.setdefault(cell.cell_id.rsplit("/", 1)[-1], []).append(
                (cell.cell_id, display))
        for short_id, bindings in short_ids.items():
            if len(bindings) == 1:
                cell_id, display = bindings[0]
                display_by_id.setdefault(short_id, display)
                display_by_id.setdefault(f"matrix/{short_id}", display)
                cell_by_reference.setdefault(short_id, cell_id)
                cell_by_reference.setdefault(f"matrix/{short_id}", cell_id)

        public_lines: list[str] = []
        evidence_by_cell: dict[str, str] = {}
        unbound_evidence: list[str] = []
        reference = re.compile(r"\[([^]]+)\]")
        for source_line in _logical_answer_lines(text):
            leading_reference = re.match(r"^\s*-\s*\[([^]]+)\]", source_line)
            line = re.sub(r"^(\s*-\s*)\[[^]]+\]\s*", r"\1", source_line,
                          count=1)

            def replace_reference(match: re.Match[str]) -> str:
                display = display_by_id.get(match.group(1))
                return display if display is not None else match.group(0)

            line = reference.sub(replace_reference, line)
            for cell in self.cells:
                # The deterministic no-overlap conclusion used to expose a
                # bare ``label(cell-id)`` pointer.  Replace that execution
                # reference with the same typed public coordinate.
                line = line.replace(
                    f"{cell.label}({cell.cell_id}) 인용 참조",
                    f"{_coordinate_heading(cell.label, cell.topic)} 확인 내용",
                )
            public_lines.append(line)
            if line.lstrip().startswith("-"):
                bound_cell = (cell_by_reference.get(leading_reference.group(1))
                              if leading_reference else None)
                if bound_cell is not None:
                    evidence_by_cell.setdefault(bound_cell, line)
                else:
                    unbound_evidence.append(line)
        # Public comparison answers lead with the finding.  The typed matrix
        # and verifier still use the model-facing conclusion-last contract;
        # this reorder happens only after validation and retains every cited
        # evidence line.  Drop the redundant ``... 결과입니다`` preface.
        marker = f"{self.comparison_kind} 결론:"
        conclusion = next((line for line in public_lines
                           if line.startswith(marker)), "")
        if conclusion:
            # The conclusion was checked against all cells before the public
            # projection.  Keep its source-grounded outcome, but lay out one
            # complete comparison statement per bullet where possible.  A
            # source table can otherwise put several thousand characters on
            # a single conclusion line, which is not a useful answer even
            # though it is technically cited.
            public_lines = [*self._public_conclusion_lines(conclusion)]
            evidence = self._public_evidence_lines(
                evidence_by_cell, unbound_evidence)
            if evidence:
                public_lines.extend(("근거별 확인:", *evidence))
            if self.limitations:
                rendered = append_public_qualifications(
                    "\n".join(public_lines), self.limitations)
                public_lines = rendered.splitlines()
        return _public_narrative_surface("\n".join(public_lines))

    def _public_evidence_lines(
            self, evidence_by_cell: dict[str, str], unbound_lines: list[str],
            ) -> list[str]:
        """Render one concise cited evidence row for every matrix coordinate.

        All matrix cells remain represented, including cells that happen to
        share a receipt.  The internal response may cite the entire paragraph
        or table in a single row; use the existing typed digest instead so
        product/revenue/investment structures retain their source labels
        without exposing neighbouring columns and repeated prior years.
        """

        temporal = len({cell.label.partition(" · ")[2] for cell in self.cells}) > 1
        company_count = len({cell.label.partition(" · ")[0] for cell in self.cells})
        cell_count = len(self.cells)
        has_structured_table = any(
            re.search(r"(?m)^\s*\|[^\n]*\|\s*$", cell.excerpt)
            for cell in self.cells)
        if has_structured_table and company_count > 1 and cell_count >= 8:
            # 넓은 회사×기간×주제 비교의 결론은 이미 모든 typed cell을 검증해
            # 계산했다. 그 뒤에 같은 8~16개 표를 다시 전사하면 일반 사용자는
            # 결론보다 원문 표를 먼저 해독해야 한다. 공개 근거는 좌표와 주제,
            # 접수번호를 하나도 버리지 않는 색인으로 접고, 내부 cell/excerpt와
            # verifier 계약은 그대로 둔다.
            grouped: dict[tuple[str, tuple[str, ...]], list[str]] = {}
            summaries: dict[tuple[str, tuple[str, ...]], list[str]] = {}
            for cell in self.cells:
                key = (cell.label, cell.receipts)
                grouped.setdefault(key, [])
                summaries.setdefault(key, [])
                topic = _topic_display(cell.topic)
                if topic not in grouped[key]:
                    grouped[key].append(topic)
                roster = _subsidiary_roster_digest(cell.excerpt, cell.topic)
                if roster and roster not in summaries[key]:
                    summaries[key].append(f"{topic}: {roster}")
                if (_is_main_business_topic(cell.topic)
                        or _topic_axis(cell.topic) == "products_services"):
                    digest = _business_table_axis_digest(cell.excerpt, limit=240)
                    if not digest:
                        digest = _business_sentence_digest(
                            cell.excerpt, limit=240, suppress_business_noise=True)
                    if digest and digest not in summaries[key]:
                        summaries[key].append(digest)
            return [
                f"- {label}: "
                + (" / ".join(summaries[(label, receipts)])
                   if summaries[(label, receipts)] else f"{'·'.join(topics)} 확인") + " "
                f"(근거: {'; '.join('접수번호 ' + receipt for receipt in receipts)})"
                for (label, receipts), topics in grouped.items()
            ]
        receipt_counts = {
            cell.receipts: sum(other.receipts == cell.receipts for other in self.cells)
            for cell in self.cells
        }
        consumed_unbound: set[int] = set()
        evidence: list[str] = []
        displayed_excerpt_keys: dict[
            tuple[tuple[str, ...], str], str] = {}
        for cell in self.cells:
            # One source can support both business axes and subsidiary names.
            # Project a proven roster before source-only deduplication and
            # model-candidate reuse, neither of which proves topic identity.
            roster = _subsidiary_roster_digest(cell.excerpt, cell.topic)
            if roster:
                receipts = "; ".join("접수번호 " + r for r in cell.receipts)
                evidence.append(
                    f"- {_coordinate_heading(cell.label, cell.topic)}: {roster} "
                    f"(근거: {receipts})")
                continue
            excerpt_key = (
                cell.receipts,
                re.sub(r"\s+", " ", cell.excerpt).strip(),
            )
            previous_display = displayed_excerpt_keys.get(excerpt_key)
            if previous_display is not None:
                # 서로 다른 요청 축이 같은 공시 본문 좌표로 닫히는 경우
                # 본문을 축마다 다시 전사하지 않는다. 축과 접수번호는
                # 남기고, 어느 앞선 확인 내용과 같은지도 일반인이 바로
                # 이해할 수 있도록 공개 좌표로 지칭한다.
                receipts = "; ".join(
                    "접수번호 " + receipt for receipt in cell.receipts)
                evidence.append(
                    f"- {_coordinate_heading(cell.label, cell.topic)}: "
                    f"{previous_display}과 확인 내용이 동일합니다. "
                    f"(근거: {receipts})")
                continue
            displayed_excerpt_keys[excerpt_key] = (
                _coordinate_heading(cell.label, cell.topic))
            # A concise model-authored sentence is useful wording and has
            # already passed the source/receipt verifier.  Keep it when it
            # is genuinely concise; re-render only table-sized source dumps.
            # This also preserves harmless Korean phrasing repairs made by
            # remove_business_performance_tail before public projection.
            candidate = evidence_by_cell.get(cell.cell_id, "")
            if (any(term in _compact_key(cell.topic)
                    for term in ("사업", "제품", "서비스"))
                    and _is_business_cross_reference(candidate)):
                candidate = ""
            # HCX can shorten task/cell IDs to a duplicate short ID.  Receipt
            # binding is safe only when that receipt tuple belongs to exactly
            # one matrix coordinate, and an unbound line is consumed at most
            # once.  Shared-receipt topic cells always regenerate from their
            # own typed evidence instead of borrowing a sibling line.
            if not candidate and receipt_counts[cell.receipts] == 1:
                candidate_index = next((
                    index for index, line in enumerate(unbound_lines)
                    if index not in consumed_unbound
                    and all(receipt in line for receipt in cell.receipts)
                ), None)
                if candidate_index is not None:
                    consumed_unbound.add(candidate_index)
                    candidate = unbound_lines[candidate_index]
            candidate_limit = (1_300 if company_count == 1 and not temporal
                               else 560 if cell_count <= 4
                               else 200 if cell_count >= 8 else 360)
            if candidate and len(candidate) <= candidate_limit and "|" not in candidate:
                evidence.append(_replace_filer_first_person(candidate, cell.label))
                continue
            # Matrix evidence is intentionally tighter than a one-coordinate
            # NarrativeDigest.  A long multi-company comparison remains
            # scannable while each requested coordinate and citation is kept.
            base_limit = _narrative_cell_display_limit(
                cell.topic, temporal=temporal,
                compact_comparison=company_count > 1)
            limit = (base_limit if company_count == 1 and not temporal
                     else min(base_limit, 360 if cell_count <= 4
                              else 180 if cell_count >= 8 else 260))
            digest = _compact_cell_text(
                cell.excerpt, topic=cell.topic, limit=limit,
                temporal_business_change=temporal)
            receipts = "; ".join("접수번호 " + receipt for receipt in cell.receipts)
            evidence.append(
                f"- {_coordinate_heading(cell.label, cell.topic)}: "
                f"{_replace_filer_first_person(digest, cell.label)} "
                f"(근거: {receipts})")
        return evidence

    def _public_conclusion_lines(self, conclusion: str) -> list[str]:
        """Turn a verified, potentially wide conclusion into readable bullets.

        The verifier consumes the original conclusion with every receipt.  In
        the public answer receipts are attached to each evidence coordinate,
        so repeating a long all-receipt suffix adds noise without improving
        traceability.  Each retained bullet is bounded at a sentence boundary
        where possible; full details remain directly below in cited evidence.
        """

        marker = f"{self.comparison_kind} 결론:"
        body = conclusion[len(marker):].strip() if conclusion.startswith(marker) else conclusion
        body = re.sub(r"\s*\(근거:\s*(?:접수번호\s*\d+[;,\s]*)+\)\s*$", "", body)
        # Statements emitted by grounded_conclusion end in a period and the
        # following typed comparison starts with a known company/coordinate.
        # An arbitrary short prefix before ``:`` is not enough: source list
        # items such as ``가. 매출 구성:`` were previously mistaken for a new
        # conclusion and left the preceding bullet as the meaningless ``가.``.
        split_prefixes = sorted({
            prefix.strip()
            for cell in self.cells
            for prefix in (
                getattr(cell, "label", ""),
                getattr(cell, "label", "").partition(" · ")[0],
                (f"{getattr(cell, 'label', '').partition(' · ')[0]} "
                 f"{_topic_display(getattr(cell, 'topic', ''))}"),
            )
            if prefix.strip()
        }, key=len, reverse=True)
        boundary = (
            r"(?<=\.)\s+(?=(?:"
            + "|".join(re.escape(prefix) for prefix in split_prefixes)
            + r")\s*(?:[·—–-][^:\n]{1,80})?:)"
            if split_prefixes else r"(?<=\.)\s+(?=[^\n:]{1,80}:)"
        )
        parts = re.split(boundary, body)
        parts = [part.strip() for part in parts if part.strip()]
        # OCR/table headings sometimes yield a lone Korean list marker such as
        # ``주요 제품 및 서비스: 가.``.  It contains no finding and the cited
        # coordinate evidence remains below, so do not promote the marker to a
        # public conclusion bullet.
        parts = [
            part for part in parts
            if re.search(r"(?:[:：]|[—–-])\s*[가-하]\.?$", part) is None
        ]
        # 같은 좌표에서 절 제목만 앞에 붙은 동일 변화문이 별도 source section
        # 으로 한 번 더 들어올 수 있다. 연도부터의 실질 문장이 완전히 같을 때만
        # 중복으로 보며, 회사·기간 좌표가 다르면 유지한다.
        deduped: list[str] = []
        seen_change_keys: set[tuple[str, str]] = set()
        for part in parts:
            coordinate = next((
                prefix for prefix in split_prefixes
                if part.startswith(prefix)
            ), "")
            content = part.split(":", 1)[-1].strip()
            year = re.search(r"(?<!\d)(?:19|20)\d{2}년", content)
            material = content[year.start():] if year else content
            change_key = (coordinate, re.sub(r"\s+", "", material))
            if year and change_key in seen_change_keys:
                continue
            if year:
                seen_change_keys.add(change_key)
            deduped.append(part)
        parts = deduped
        if not parts:
            return [marker]
        # A comparison can contain a legitimate boundary (for example, the
        # source does not establish a business reorganisation) *and* a direct
        # finding (for example, a segment-sales increase).  Lead with the
        # direct finding so a reader sees the answer before its qualification.
        # This is presentation-only: every sentence is already verified and
        # stays verbatim, with original order retained within each priority.
        material_markers = (
            "증가", "감소", "상승", "하락", "확대", "축소", "신규", "중단",
            "변경", "→", "차이", "다릅니다", "공통",
        )

        def conclusion_priority(part: str) -> int:
            if "확정하지 않습니다" in part or "확인할 수 없습니다" in part:
                return 3
            # Quantified changes answer a temporal comparison more directly
            # than a source wording difference such as ``컴퓨터`` vs ``PC``.
            if re.search(r"\d[\d,.]*.*(?:→|증가|감소|상승|하락)", part):
                return 0
            if "원문 표현 차이" in part:
                return 2
            if any(value in part for value in material_markers):
                return 1
            return 2

        priorities = [conclusion_priority(part) for part in parts]
        # Reorder whenever a qualification currently precedes a more direct
        # finding — not only for one priority pair.  ``확정하지 않습니다``
        # (priority 3) ahead of a quantified change (priority 0) is the same
        # defect as the 2-before-0 case: the reader meets the caveat first and
        # has to read past it to find the answer.
        reordered = sorted(priorities) != priorities
        if reordered:
            parts = [part for _, part in sorted(
                enumerate(parts), key=lambda row: (conclusion_priority(row[1]), row[0]))]
        # Short comparisons normally stay as one compact paragraph.  Reflow
        # only when it would otherwise become a long conclusion line, or when
        # material findings must move ahead of a stated qualification.
        if (len(self.cells) <= 4 and len(conclusion) <= 720
                and not reordered):
            return [conclusion]
        if len(self.cells) >= 8:
            # A wide company×period×topic matrix can yield dozens of correct
            # sub-conclusions.  Publishing every one duplicates the complete
            # coordinate evidence immediately below.  Keep at most two
            # verified findings per company (normally business + product), in
            # matrix order; every cell remains cited in the evidence section.
            companies = list(dict.fromkeys(
                cell.label.partition(" · ")[0] for cell in self.cells))
            selected: list[str] = []
            for company in companies:
                own = [part for part in parts if part.lstrip("- ").startswith(company)]
                if not own:
                    own = [part for part in parts if company in part]
                # Wide business comparisons normally ask for two distinct
                # axes: what the company does and how revenue is composed.
                # Taking the first two sentences can consume both slots with
                # business prose and silently hide the requested revenue
                # conclusion.  Prefer one verified sentence per axis, then
                # fill any remaining display slot in source order.
                business = next((part for part in own if any(
                    marker in part for marker in (
                        "주요 사업", "사업 변화", "사업부문"))
                    and not any(marker in part for marker in (
                        "수익 구조", "수익구조", "매출 구성", "매출구성"))), None)
                revenue = next((part for part in own if any(
                    marker in part for marker in (
                        "수익 구조", "수익구조", "매출 구성", "매출구성"))), None)
                preferred = [part for part in (business, revenue) if part]
                preferred.extend(part for part in own if part not in preferred)
                selected.extend(preferred[:2])
            if selected:
                parts = list(dict.fromkeys(selected))
        rendered = [_clean_at_boundary(part, 300 if len(self.cells) >= 8 else 480)
                    for part in parts]
        if len(rendered) == 1:
            return [f"{marker} {rendered[0]}"]
        return [f"{marker} {rendered[0]}", *(f"- {part}" for part in rendered[1:])]

    def deterministic(
            self, *, public: bool = True, question: str | None = None) -> str:
        lines = [f"{self.comparison_kind} 결과입니다."]
        normalized = []
        temporal = len({cell.label.partition(" · ")[2]
                        for cell in self.cells}) > 1
        # Preserve every typed coordinate while keeping wide historical
        # tables from becoming an answer-sized raw dump.  Digest helpers emit
        # complete rows/sentences only, so this budget never slices a claim.
        # 한 회사의 셀 여럿이 글자 그대로 같은 본문을 인용할 때가 있다.  두
        # 기간 사이에 공시가 바뀌지 않았거나(기간 축), 공시가 요청된 두 주제를
        # 따로 기술하지 않거나(주제 축) 둘 중 하나인데, **어느 쪽이든 그
        # 동일함 자체가 답**이다.  같은 문단을 두 번 찍으면 답변만 길어지고
        # 결론은 오히려 흐려진다.
        #
        # 주제가 다르면 별개 답이라고 보아 처음에는 기간 축만 묶었는데,
        # 실측에서 `G-O-001`(사업부문 ↔ 주요 제품)·`EDGE-014`(주요 사업 ↔
        # 자회사 구성)·`EDGE-013`(주요 사업 ↔ 수익구조)이 모두 주제 축이었다.
        # 참조 줄이 주제와 접수번호를 그대로 담으므로 잃는 정보가 없다.
        #
        # 셀 자체는 그대로 두어 결론 계산과 근거는 손대지 않고, 표시만 앞 셀을
        # 가리킨다.  회사가 다르면 묶지 않는다 — 서로 다른 공시가 같은 문단을
        # 담는 것은 별개 사실이다.
        rendered_before: dict[tuple[str, str, str], str] = {}
        rendered_sentences: dict[str, list[tuple[str, str]]] = {}
        for cell in self.cells:
            digest = _compact_cell_text(
                cell.excerpt, topic=cell.topic,
                limit=_narrative_cell_display_limit(
                    cell.topic, temporal=temporal,
                    compact_comparison=len({
                        item.label.partition(" · ")[0]
                        for item in self.cells}) > 1),
                temporal_business_change=temporal)
            company = cell.label.partition(" · ")[0]
            # Display digests are lossy.  Two sources can select the same
            # bounded sentence while differing elsewhere, so only an exact
            # normalized source excerpt *and the same topic projection* may
            # support an ``identical`` display claim.  One source block can
            # legitimately project to a business-axis list for one cell and a
            # product/value table for another.
            source_identity = re.sub(r"\s+", " ", cell.excerpt).strip()
            digest_identity = re.sub(r"\s+", " ", digest).strip()
            key = (company, source_identity, digest_identity)
            seen = rendered_before.get(key)
            receipts = "; ".join("접수번호 " + r for r in cell.receipts)
            if seen is not None:
                lines.append(
                    f"- [{cell.cell_id}] {cell.label} ({cell.topic}): "
                    f"[{seen}]과 인용 본문이 동일합니다. (근거: {receipts})")
                rendered_before[key] = seen
                rendered_sentences.setdefault(company, []).append(
                    (cell.cell_id, digest))
                normalized.append((source_identity, digest_identity))
                continue
            # 같은 회사의 앞선 셀이 이 본문을 **거의 그대로** 담고 있을 때가
            # 있다.  한 문장만 덧붙은 경우가 그렇다(`EDGE-014` 신한지주:
            # 자회사 구성 셀 = 주요 사업 셀 + 상장 사실 한 줄).  전문을 다시
            # 찍으면 두 문단이 거의 같은 채로 나란히 놓여 무엇이 다른지 오히려
            # 안 보인다.  문장은 그대로 두고 덧붙은 부분만 보여준다.
            extra = _added_sentences(digest, rendered_sentences.get(company, ()))
            if extra is not None:
                base_id, added = extra
                lines.append(
                    f"- [{cell.cell_id}] {cell.label} ({cell.topic}): "
                    f"[{base_id}] 인용에 다음이 더해집니다 — {added} "
                    f"(근거: {receipts})")
                rendered_sentences.setdefault(company, []).append(
                    (cell.cell_id, digest))
                normalized.append((source_identity, digest_identity))
                continue
            rendered_before[key] = cell.cell_id
            rendered_sentences.setdefault(company, []).append(
                (cell.cell_id, digest))
            lines.append(
                f"- [{cell.cell_id}] {cell.label} ({cell.topic}): "
                f"{digest} (근거: {receipts})")
            normalized.append((source_identity, digest_identity))
        if self.incomplete:
            conclusion = "일부 matrix 좌표를 확인하지 못해 모든 좌표의 차이·변화는 확정하지 않습니다."
        elif (len(set(normalized)) == 1
              and len({cell.label.partition(" · ")[0]
                       for cell in self.cells}) == 1):
            # A single company in a single period has no change axis, so
            # reporting "no change" answers a question that was never asked.
            # Say instead that the requested items share one source passage.
            if self.comparison_kind == "항목별 요약":
                topics = "·".join(dict.fromkeys(
                    _topic_display(cell.topic) for cell in self.cells
                    if _topic_display(cell.topic)))
                conclusion = (
                    f"공시에서 확인한 항목은 {topics}입니다."
                    if topics else "요청한 항목을 공시에서 확인했습니다."
                )
            else:
                # A null finding must carry its own scope.  Without it the
                # sentence reads as "this filing did not change", when what
                # was actually checked is one quoted axis of it.
                conclusion = (
                    "비교 좌표의 확인된 인용 본문이 동일하므로, "
                    "이 인용 범위에서 확인되는 변화는 없습니다. "
                    + self._checked_axis_note())
        else:
            conclusion = "검증된 셀을 직접 대조하면 " + self.grounded_conclusion()
        lines.append(f"{self.comparison_kind} 결론: {conclusion} "
                     f"(근거: {'; '.join('접수번호 ' + r for r in self.receipts)})")
        if self.limitations:
            rendered = append_public_qualifications(
                "\n".join(lines), self.limitations)
            lines = rendered.splitlines()
        answer = "\n".join(lines)
        rendered = self.public_answer(answer) if public else answer
        # `public_answer` regenerates most lines from `self.cells` (canonical
        # corp_name), so the question-surface swap happens on the finished
        # text rather than on `answer` — a pre-swap would be overwritten by
        # that regeneration (#114).
        return _apply_question_company_display(rendered, self.cells, question)

    def _checked_axis_note(self) -> str:
        """Name the axes actually quoted so a null finding is not read wide."""

        topics = list(dict.fromkeys(
            _topic_display(cell.topic) for cell in self.cells
            if _topic_display(cell.topic)))
        if not topics:
            return "확인 범위는 인용한 본문에 한정됩니다."
        return (f"확인 범위는 «{'·'.join(topics)}» 인용 본문이며 "
                "같은 보고서의 다른 공시 축은 확인하지 않았습니다.")

    def grounded_conclusion(self) -> str:
        """Synthesize only source-literal commonalities and direct changes.

        The old fallback concatenated shortened source cells.  It was safe but
        did not actually answer a comparison request.  This implementation
        groups typed coordinates, reuses literal terms and labelled values,
        and performs no synonym expansion or causal inference.  Arithmetic is
        limited to exact subtraction between the same typed metric and unit.
        """

        # 좌표 하나가 부족하다고 확인된 좌표까지 버리지 않는다. 아래 합성은
        # 인용된 셀만 쓰므로 그대로 돌리고, 확인하지 못한 좌표는 이름을 들어
        # 끝에 덧붙인다 — 무엇을 못 봤는지 밝히는 것이 「모두 확정하지
        # 않습니다」보다 읽는 사람에게 쓸모가 있다 (이슈 #259 · #261 C-1).
        unconfirmed = tuple(
            f"{cell.label} {_topic_display(cell.topic)}".strip()
            for cell in self.cells
            if not _narrative_excerpt_is_adequate(cell.excerpt, topic=cell.topic))

        companies = tuple(dict.fromkeys(
            cell.label.partition(" · ")[0] for cell in self.cells))
        periods = tuple(dict.fromkeys(
            cell.label.partition(" · ")[2] for cell in self.cells))
        if len(companies) == 1 and len(periods) == 1:
            # 「확인한 항목은 A·B입니다」는 무엇을 찾았는지만 말하고 질문에
            # 답하지 않는다. 축 이름 대신 그 축에서 공시가 실제로 쓴 표현을
            # 싣는다 — 뜻을 지어내지 않고 원문 낱말만 옮긴다 (#260 ①).
            parts: list[str] = []
            for topic in dict.fromkeys(_topic_display(c.topic) for c in self.cells):
                terms: list[str] = []
                for cell in self.cells:
                    if _topic_display(cell.topic) != topic:
                        continue
                    for term in _business_concepts(cell.excerpt, limit=4):
                        if term not in terms:
                            terms.append(term)
                if terms:
                    joined = "·".join(terms[:4])
                    parts.append(f"{topic}{josa(topic, '은', '는')} {joined}")
            if parts:
                return self._with_unconfirmed_note(
                    f"{companies[0]} 공시에서 확인한 " + ", ".join(parts) + "입니다.",
                    unconfirmed)
            topics = tuple(dict.fromkeys(_topic_display(c.topic) for c in self.cells))
            return self._with_unconfirmed_note(
                "공시에서 확인한 항목은 " + "·".join(topics) + "입니다.", unconfirmed)

        statements: list[str] = []

        # If company and period axes coexist, compare each company's exact
        # topic coordinates over time first.  Cross-company token overlap
        # cannot answer a period-change request.
        if len(periods) > 1:
            period_groups: dict[tuple[str, str], list[MatrixCellEvidence]] = {}
            for cell in _propagate_cross_topic_restatement_basis(self.cells):
                company, _, _period = cell.label.partition(" · ")
                period_groups.setdefault((company, _topic_axis(cell.topic)), []).append(cell)
            # 회사 축은 아래에서 한 번에 모아 정형 문장을 합치므로, 루프
            # 안에서는 바로 statements 에 붙이지 않고 먼저 모은다 (#114).
            period_statements: list[tuple[str, str, str]] = []
            for (company, topic), cells in period_groups.items():
                if len(cells) < 2:
                    continue
                typed_cells = tuple(sorted(
                    cells, key=lambda cell: cell.label.partition(" · ")[2]))
                display_topic = _topic_display(typed_cells[0].topic)
                main_business_change = _is_main_business_topic(display_topic)
                # An identical finding is stronger than a display-level
                # similarity.  Require the complete retained source excerpts
                # to match, not merely their bounded public digests.
                normalized = {
                    re.sub(r"\s+", " ", cell.excerpt).strip()
                    for cell in typed_cells}
                if len(normalized) == 1:
                    period_statements.append((
                        company, display_topic,
                        "두 기간의 인용 본문이 동일하여 확인되는 변화가 없습니다"
                        f"(확인 범위는 «{display_topic}» 인용 본문이며 같은 "
                        "보고서의 다른 공시 축은 확인하지 않았습니다)"))
                    continue
                structured = _structured_period_contrast(
                    typed_cells, topic=display_topic)
                # The structured comparator binds exact composite row
                # identities and already emits bounded deltas.  Do not append
                # the broader direct-metric pass over the same table.
                metric = ("" if structured else
                          _metric_contrast(typed_cells, topic=display_topic))
                clauses = []
                before_year = _cell_report_year(typed_cells[0])
                after_year = _cell_report_year(typed_cells[-1])
                latest_cell = typed_cells[-1]
                latest_direct = (
                    _material_business_change_clauses(
                        latest_cell.excerpt, limit=2)
                    if main_business_change else
                    _explicit_change_clauses(latest_cell.excerpt, limit=2))
                # A direct clause supplements the endpoint comparison only
                # when it names an event inside that requested window.  YoY
                # clauses have a different base and are never endpoint deltas.
                latest_direct = tuple(
                    value for value in latest_direct
                    if (not _is_non_endpoint_comparison_clause(
                            value, before_year=before_year,
                            after_year=after_year)
                        and _inside_requested_window_or_period_relative(
                            value, before_year=before_year,
                            after_year=after_year,
                            source_year=_cell_report_year(latest_cell),
                            include_before=False)))
                direct_by_cell = [(latest_cell, latest_direct)]
                repeated_business_clause: set[str] = set()
                if main_business_change:
                    occurrences: dict[str, int] = {}
                    for _cell, values in direct_by_cell:
                        for value in values:
                            key = _compact_key(value).casefold()
                            occurrences[key] = occurrences.get(key, 0) + 1
                    repeated_business_clause = {
                        key for key, count in occurrences.items() if count > 1}
                for cell, direct in direct_by_cell:
                    clauses.extend(
                        f"{cell.label}: {value}" for value in direct
                        if (not _is_non_endpoint_comparison_clause(
                                value, before_year=before_year,
                                after_year=after_year)
                            and _compact_key(value).casefold()
                            not in repeated_business_clause))
                # Mere wording differences across annual reports do not prove
                # a change in operating scope.  For business/product topics,
                # use only labelled metric/table deltas or a complete source
                # sentence that states an operating action.  If neither is
                # present, say so explicitly instead of promoting prose style
                # changes into a business-change conclusion.
                structure = (_business_structure_contrast(
                    typed_cells, before_year=before_year,
                    after_year=after_year)
                             if main_business_change else "")
                # A cross-report table comparison already names the exact
                # composite rows and values.  Appending token-level product
                # differences after it only reintroduces parser fragments
                # (including fused labels and orphan numeric tokens).  That
                # only holds once such a comparison exists: suppressing the
                # term contrast on the axis alone leaves a prose-only product
                # comparison with no stated commonality at all.
                terms = ("" if (main_business_change or structured
                                 or _topic_axis(display_topic) == "revenue_mix") else
                         _term_contrast(
                             typed_cells, include_unique=True,
                             # Every branch here is a period comparison.  Use
                             # the stricter material extractor for product
                             # prose as well as for the main-business axis, so
                             # connective fragments such as ``사업별로 보면``
                             # cannot become alleged products.
                             temporal_business_change=True))
                detail = ". ".join(value for value in (
                    metric, structured, "; ".join(clauses), structure, terms) if value)
                if (_has_reclassified_comparatives(typed_cells)
                        and _topic_axis(display_topic) in {"products_services", "revenue_mix"}
                        and not structured):
                    detail = (
                        "최신 공시의 비교 수치가 재작성되어 과거 공시와 기준이 다르며, "
                        "최신 공시에서 요청한 두 기간의 동일 기준 수치를 확인하지 못해 "
                        "수치 변화는 계산하지 않습니다")
                elif (_topic_axis(display_topic) == "revenue_mix"
                        and not metric and not structured):
                    detail = (
                        "요청한 두 기간의 종점에 정렬된 동일 수익 지표가 없어 "
                        "수치 변화는 계산하지 않습니다")
                if main_business_change and not detail:
                    detail = _MAIN_BUSINESS_CHANGE_UNCONFIRMED
                if not detail:
                    summaries = [
                        f"{cell.label}: {_compact_cell_text(cell.excerpt, topic=cell.topic)}"
                        for cell in typed_cells]
                    detail = ("두 기간의 인용 본문 표현은 서로 다르며, "
                              + "; ".join(summaries))
                if detail:
                    period_statements.append(
                        (company, display_topic, detail.rstrip('.')))
                # Fanout is already bounded by the retrieval contract.  Do
                # not silently drop later company/topic coordinates merely
                # because more than six independently verified comparisons
                # were requested.
            # 회사마다 되풀이되는 정형 문장(대표적으로 「변경 근거가 없어 …
            # 확정하지 않습니다」)은 회사 이름만 모아 한 번으로 말한다 —
            # 회사 수만큼 같은 문장이 되풀이되면 결론이 요약 구실을 못
            # 한다(#114).  글자 그대로 같은 문장일 때만 합친다: 다른 회사가
            # 우연히 비슷한 말을 했다고 뜻까지 같다고 단정하지 않는다.
            grouped_companies: dict[str, list[str]] = {}
            for entry_company, _entry_topic, entry_detail in period_statements:
                if entry_detail == _MAIN_BUSINESS_CHANGE_UNCONFIRMED:
                    grouped_companies.setdefault(
                        entry_detail, []).append(entry_company)
            merged_emitted: set[str] = set()
            for company, topic, detail in period_statements:
                if detail == _MAIN_BUSINESS_CHANGE_UNCONFIRMED:
                    if detail in merged_emitted:
                        continue
                    merged_emitted.add(detail)
                    names = list(dict.fromkeys(grouped_companies[detail]))
                    if len(names) > 1:
                        joined = "·".join(names)
                        statements.append(
                            f"{joined}{josa(joined, '은', '는')} {detail}.")
                    else:
                        statements.append(f"{names[0]} {topic} — {detail}.")
                    continue
                statements.append(f"{company} {topic} — {detail}.")

        # Company comparisons use one common period at a time.  When the
        # question also has a time axis, compare the latest common period in
        # addition to each company's own period change.
        if len(companies) > 1:
            company_groups: dict[tuple[str, str], list[MatrixCellEvidence]] = {}
            for cell in self.cells:
                _company, _, period = cell.label.partition(" · ")
                company_groups.setdefault((period, cell.topic), []).append(cell)
            eligible_groups = [
                (key, cells) for key, cells in company_groups.items()
                if len(cells) >= 2]
            if len(periods) > 1 and eligible_groups:
                latest_period = max(key[0] for key, _cells in eligible_groups)
                eligible_groups = [
                    (key, cells) for key, cells in eligible_groups
                    if key[0] == latest_period]
            for (period, topic), cells in eligible_groups:
                if len(cells) < 2:
                    continue
                # Requested-period matrices are already compared in exact
                # company/topic coordinates above.  A second cross-company
                # token pass over the latest report adds no value-level
                # comparison and is prone to exposing fused PDF labels or
                # orphan header fragments.  Keep cross-company term contrast
                # for point-in-time matrices only.
                if len(periods) > 1:
                    continue
                typed_company_cells = tuple(cells)
                revenue = ("" if _topic_axis(topic) != "revenue_mix"
                           and not any(marker in _compact_key(topic)
                                       for marker in ("수익원", "수익구조"))
                           else _company_revenue_source_contrast(
                               typed_company_cells))
                detail = revenue or _term_contrast(
                    typed_company_cells, include_unique=True,
                    temporal_business_change=(
                        len(periods) > 1 and _is_main_business_topic(topic)))
                if not detail:
                    # 각 셀은 바로 위에 이미 출력돼 있다.  여기서 본문을 다시
                    # 인용하면 같은 문단이 한 답변에 두 번 나오고(`EDGE-013`),
                    # 결론이 인용문에 묻힌다.  좌표만 가리킨다.
                    # 좌표까지 나열하면 「… 삼성화재해상보험 · 2026-01-01~
                    # 2026-03-31 (수익구조 차이) 확인 내용.」 처럼 기간과 축이
                    # 뒤섞여 문장이 끊긴 채 끝난다(`EDGE-013`).  누구를 맞대었
                    # 는지는 회사 이름이면 되고, 내용은 바로 아래에 있다.
                    compared = list(dict.fromkeys(
                        cell.label.split(" · ", 1)[0] for cell in cells))
                    detail = ("·".join(compared)
                              + "의 확인 내용이 서로 다릅니다."
                              " 아래 근거별 확인을 보세요")
                if detail:
                    period_prefix = f"{period} " if len(periods) > 1 else ""
                    statements.append(
                        # 이 문장 앞에는 「회사별 비교 결론: 」이 붙는다.
                        # 축 이름 뒤에도 콜론을 찍으면 한 문장에 콜론이 둘이다.
                        f"{period_prefix}{topic} — {detail.rstrip('.')}.")
                if len(statements) >= 3:
                    break
        if statements:
            # 무엇을 근거로 맞대었는지는 축마다 되풀이할 말이 아니다. 공시에
            # 적힌 표현을 그대로 비교한 것이지 뜻을 견준 것이 아니라는 단서를
            # 결론 머리에 한 번만 둔다.
            return self._with_unconfirmed_note(
                "공시 원문 표현 기준. " + " ".join(statements), unconfirmed)

        # No literal overlap or matching labelled value means there is no
        # auditable synthesis.  Expose short source terms rather than copying
        # entire paragraphs or claiming semantic equivalence.
        fallback = []
        for cell in self.cells:
            terms = _business_concepts(cell.excerpt, limit=4)
            if terms:
                fallback.append(f"{cell.label}: {'·'.join(terms)}")
        if len(fallback) >= 2:
            return self._with_unconfirmed_note(
                "각 인용 셀에서 직접 확인되는 핵심 표현은 " + "; ".join(fallback) + "입니다.",
                unconfirmed)
        return self._with_unconfirmed_note(
            "제시된 인용 범위만으로는 차이의 내용을 확정하지 않습니다.", unconfirmed)

    @staticmethod
    def _with_unconfirmed_note(conclusion: str, unconfirmed: tuple[str, ...]) -> str:
        """확인하지 못한 좌표를 이름으로 밝힌다. 없으면 결론을 그대로 둔다."""

        if not unconfirmed:
            return conclusion
        listed = "·".join(dict.fromkeys(unconfirmed))
        return f"{conclusion} 다만 {listed} 는 인용 범위에서 확인하지 못해 확정하지 않습니다."

    @property
    def derived_number_tokens(self) -> set[str]:
        """Numbers permitted only because exact cited pairs derive them."""

        periods = {cell.label.partition(" · ")[2] for cell in self.cells}
        if len(periods) < 2:
            return set()
        groups: dict[tuple[str, str], list[MatrixCellEvidence]] = {}
        for cell in self.cells:
            company, _, _period = cell.label.partition(" · ")
            groups.setdefault((company, _topic_axis(cell.topic)), []).append(cell)
        allowed: set[str] = set()
        for (_company, _topic), cells in groups.items():
            typed_cells = tuple(sorted(
                cells, key=lambda cell: cell.label.partition(" · ")[2]))
            display_topic = _topic_display(typed_cells[0].topic)
            if _metric_contrast(typed_cells, topic=display_topic):
                for before, _after, delta in _metric_deltas(typed_cells):
                    rendered = _format_delta(delta, before.unit)
                    allowed.update(
                        token.replace(",", "") for token in _NUM.findall(rendered)
                        if len(token.replace(",", "")) >= 3)
            rendered_tables = _structured_cross_cell_deltas(
                typed_cells, topic=display_topic)
            if not rendered_tables:
                rendered_tables = _structured_explicit_year_fallback(
                    typed_cells, topic=display_topic)
            for table in rendered_tables:
                # The structured renderer may translate adjacent fiscal
                # columns (for example 제56기→제57기) to the report-year pair
                # carried by the typed cell.  Those year labels are derived
                # by the same closed rule as the values and must not be
                # rejected as an ungrounded number during output validation.
                allowed.update(re.findall(r"(?<!\d)20\d{2}(?!\d)",
                                          table.period_label))
                for _label, _prior, _latest, delta, unit in table.rows:
                    rendered = _format_delta(delta, unit)
                    allowed.update(
                        token.replace(",", "") for token in _NUM.findall(rendered)
                        if len(token.replace(",", "")) >= 3)
        return allowed

    def replace_with_grounded_conclusion(self, text: str) -> str:
        """Replace only a weak conclusion with a source-only matrix synopsis.

        Cell sentences have already passed receipt, number and topic checks
        before this is called.  A second HCX request merely to restate those
        cells can exceed the server budget on an eight-cell comparison.  This
        renderer reuses those verified sentences in their typed company/
        period/topic coordinates and therefore adds neither facts nor an LLM
        call.  If a sentence cannot be bound back to its cell, the bounded
        source excerpt remains the fail-closed fallback for that coordinate.
        """
        marker = f"{self.comparison_kind} 결론:"
        body = text.rsplit(marker, 1)[0].rstrip()
        conclusion = (
            f"{marker} 검증된 셀을 직접 대조하면 {self.grounded_conclusion()} "
            f"(근거: {'; '.join('접수번호 ' + r for r in self.receipts)})"
        )
        return f"{body}\n{conclusion}" if body else conclusion

    def remove_business_performance_tail(self, text: str) -> str:
        """Drop an unrequested performance tail from a main-business cell.

        A business-axis answer may state the operating activity and then copy
        an adjacent comparative-period revenue/profit clause.  That clause is
        unnecessary and can make the period look wrong.  Only a trailing
        clause is removed, only after a business marker has already grounded
        the retained text, and citations remain untouched.
        """
        lines = text.splitlines()
        performance = ("매출액", "영업이익", "당기순이익", "순이익")
        temporal_performance = re.compile(
            r"(?:20[0-9]{2}년|당기|전기)[^.]{0,140}"
            r"(?:전년\s*대비|[0-9][0-9,.]*\s*%|증가|감소)")
        for cell in self.cells:
            topic = re.sub(r"\s+", "", cell.topic)
            if "주요사업" not in topic or "수익" in topic or "매출" in topic:
                continue
            index = next((i for i, line in enumerate(lines)
                          if f"[{cell.cell_id}]" in line
                          and all(receipt in line for receipt in cell.receipts)), None)
            if index is None:
                continue
            line = lines[index]
            candidates = [line.find(marker) for marker in performance
                          if marker in line]
            temporal = temporal_performance.search(line)
            if temporal is not None:
                candidates.append(temporal.start())
            at = min(candidates, default=-1)
            if at < 0:
                continue
            citation_at = line.rfind("(접수번호")
            if citation_at <= at:
                continue
            cut = max(line.rfind(delimiter, 0, at) for delimiter in (".", ",", ";"))
            if cut < 0:
                continue
            retained = line[:cut].rstrip(" ,.;")
            if not any(marker in retained for marker in _BUSINESS_EVIDENCE_MARKERS):
                continue
            lines[index] = f"{retained}{line[citation_at:]}"
        return "\n".join(lines)

    def restore_equivalent_cell_coordinates(self, text: str) -> str:
        """Restore coordinates HCX collapsed only when their evidence is identical.

        HCX occasionally emits one sentence for two requested topics when the
        two cells point to the same company, period, receipts and exact source
        excerpt.  Replacing the emitted coordinate with each original typed
        coordinate is then a lossless structural expansion.  Cells with even
        one differing evidence attribute are deliberately left untouched.
        """
        marker = f"{self.comparison_kind} 결론:"
        body, separator, conclusion = text.partition(marker)
        lines = body.rstrip().splitlines()
        restored: list[str] = []
        for cell in self.cells:
            exact = next((
                line for line in lines
                if f"[{cell.cell_id}]" in line
                and all(receipt in line for receipt in cell.receipts)
            ), "")
            if exact:
                restored.append(exact)
                continue
            company, _, period = cell.label.partition(" · ")
            equivalents = [
                other for other in self.cells
                if other.receipts == cell.receipts
                and other.excerpt == cell.excerpt
                and other.label.partition(" · ")[0] == company
                and other.label.partition(" · ")[2] == period
            ]
            # One unique coordinate with a mismatched model ID is ambiguous;
            # only an actually duplicated evidence group is safe to expand.
            if len(equivalents) < 2:
                return text
            source_line = next((
                line for line in lines
                if line.lstrip().startswith("- [")
                and all(receipt in line for receipt in cell.receipts)
                and company in line
            ), "")
            if not source_line:
                return text
            restored.append(re.sub(
                r"^(\s*-\s*)\[[^]]+\]", rf"\1[{cell.cell_id}]",
                source_line, count=1))
        suffix = f"{marker}{conclusion}" if separator else ""
        return "\n".join((*restored, suffix) if suffix else restored)


def _scalar_column_key(value: str) -> str:
    key = _compact_key(value.replace(" > ", " ")).casefold()
    return re.sub(r"^(20\d{2}년)(?:\1)+$", r"\1", key)


def _unique_narrative_money_cell(
        claims, question: str | None,
        ) -> tuple[str, str, str, str] | None:
    """Project one explicitly named table row and column for a scalar question.

    Narrative retrieval can return neighbouring sections even when the user
    asks for one amount. A deterministic projection is safe only when both the
    row label and column label occur literally in the question, the table
    declares a currency unit, and all matching source sections agree on one
    value. Ambiguity falls back to the existing full digest.
    """

    if not question or not re.search(r"얼마|금액", question):
        return None
    if re.search(r"비교|각각|모두|전체|목록|현황|정리", question):
        return None
    compact_question = _compact_key(question).casefold()
    question_years = set(re.findall(r"(?<!\d)20\d{2}(?!\d)", question))
    current_report_period_requested = bool(re.search(
        r"(?<!\d)20\d{2}년\s*사업보고서(?:를|의)?\s*기준", question))
    generic_axes = {
        "구분", "항목", "부문", "품목", "제품", "서비스", "금액", "합계", "총계",
    }
    candidates: dict[
        tuple[str, str, str, str, tuple[str, ...]],
        tuple[str, str, str, str],
    ] = {}
    candidate_specificity: dict[tuple, int] = {}
    for claim in claims:
        receipts = _receipt_numbers((claim,))
        if not receipts:
            continue
        lines = _source_body(claim.text).splitlines()
        index = 0
        while index < len(lines):
            if not lines[index].lstrip().startswith("|"):
                index += 1
                continue
            start = index
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                index += 1
            block = lines[start:index]
            if len(block) < 3:
                continue
            header = _table_cells(block[0])
            if not _is_table_rule(_table_cells(block[1])):
                continue
            unit = _local_table_currency_unit(lines, start)
            if unit not in {"원", "천원", "백만원", "억원", "조원"}:
                continue
            role = _local_table_period_role(lines, start)
            claim_report_years = set(re.findall(
                r"(?<!\d)(20\d{2})\s*보고서", claim.label or ""))
            # One annual-report block can contain both 당기 and 전기 tables.
            # When the requested year is the report year's typed coordinate,
            # the 전기 value is a different period and must not make the
            # otherwise unique scalar appear ambiguous.
            if (role == "전기" and "전기" not in compact_question
                    and (current_report_period_requested
                         or (question_years
                             and question_years & claim_report_years))):
                continue
            column_positions = [
                offset for offset, value in enumerate(header)
                if (len(_scalar_column_key(value)) >= 2
                    and _scalar_column_key(value) not in generic_axes
                    and (_scalar_column_key(value) in compact_question
                         or (_scalar_column_key(value).endswith("금액")
                             and len(_scalar_column_key(value)) > 4
                             and _scalar_column_key(value)[:-2] in compact_question
                             and "금액" in compact_question)))
                or (len(question_years) == 1
                    and re.fullmatch(
                        r"(?:연결|별도)?(?:" + "|".join(question_years)
                        + r")(?:년(?:도)?)?(?:금액|매출액)?",
                        _scalar_column_key(value)) is not None
                    and not ("별도" in compact_question and "연결" in value)
                    and not ("연결" in compact_question and "별도" in value))
            ]
            for row_line in block[2:]:
                row = _table_cells(row_line)
                if len(row) != len(header) or _is_table_rule(row):
                    continue
                for value_pos in column_positions:
                    raw = row[value_pos]
                    if _table_number(raw, unit) is None:
                        continue
                    matched_labels = tuple(dict.fromkeys(
                        " ".join(value.split()).removeprefix("- ")
                        for value in row[:value_pos]
                        if ((_compact_key(value) not in generic_axes
                             or (_compact_key(value) in {"합계", "총계"}
                                 and _compact_key(value) in compact_question
                                 and not re.search(r"20\d{2}", header[value_pos])))
                            and len(_compact_key(value)) >= 2
                            and _compact_key(value).casefold() in compact_question)
                    ))
                    row_label = " ".join(matched_labels)
                    if not row_label:
                        continue
                    column_label = " ".join(header[value_pos].replace(" > ", " ").split())
                    if re.fullmatch(r"20\d{2}년(?:도)?", _scalar_column_key(column_label)):
                        column_label = _scalar_column_key(column_label)
                    rendered = format_source_money_exact(raw, unit)
                    if not rendered:
                        continue
                    display_row = (
                        f"{row_label} 부문"
                        if f"{_compact_key(row_label)}부문" in compact_question
                        else row_label
                    )
                    citation = "; ".join(
                        "접수번호 " + receipt for receipt in receipts)
                    key = (display_row, column_label, raw, unit, receipts)
                    candidates[key] = (
                        display_row, column_label, rendered, citation)
                    candidate_specificity[key] = len(matched_labels)
    # A question commonly names both its period and its metric. If any
    # matching table exposes the metric as a literal non-period column
    # (``신규수주``), do not let neighbouring tables whose only match is the
    # generic year column (``2025년``) compete with that stronger coordinate.
    non_period_candidates = {
        key: value for key, value in candidates.items()
        if re.search(r"(?:19|20)\d{2}", value[1]) is None
    }
    if non_period_candidates:
        candidates = non_period_candidates
    if candidates:
        specificity = max(candidate_specificity[key] for key in candidates)
        candidates = {key: value for key, value in candidates.items()
                      if candidate_specificity[key] == specificity}
    unique_values = {
        (row, column, rendered)
        for row, column, rendered, _citation in candidates.values()
    }
    if len(unique_values) != 1 or not candidates:
        return None
    return next(iter(candidates.values()))


def _bank_card_nim_rows(claims, question: str | None) -> tuple[str, ...]:
    query = _compact_key(question or "").casefold()
    years = set(re.findall(r"20\d{2}", query))
    if (len(years) != 1 or "은행만" not in query or "카드" not in query
            or not ("nim" in query or "순이자마진" in query)):
        return ()
    year = next(iter(years))
    bank, combined = set(), set()
    for claim in claims:
        text = _source_body(claim.text)
        receipts = _receipt_numbers((claim,))
        for match in re.finditer(r"은행\s*NIM(?:\([^)]*\))?[^。\n]*?(\d+(?:\.\d+)?%)를\s*기록", text):
            for receipt in receipts:
                bank.add((match.group(1), receipt))
        lines = text.splitlines()
        for index, line in enumerate(lines[:-2]):
            header = _table_cells(line)
            columns = [i for i, cell in enumerate(header) if year in cell
                       and not re.search(r"증감|비중", cell)]
            if len(columns) != 1 or not _is_table_rule(_table_cells(lines[index + 1])):
                continue
            for raw in lines[index + 2:]:
                if not raw.lstrip().startswith("|"):
                    break
                row = _table_cells(raw)
                if len(row) != len(header):
                    continue
                labels = [cell for cell in row[:columns[0]] if re.search(r"NIM\([^)]*은행\s*\+[^)]*카드\)", cell)]
                if len(labels) != 1 or not re.fullmatch(r"\d+(?:\.\d+)?%", row[columns[0]].strip()):
                    continue
                for receipt in receipts:
                    combined.add((labels[0], row[columns[0]].strip(), receipt))
    if len(bank) != 1 or len(combined) != 1:
        return ()
    bank_value, bank_receipt = next(iter(bank))
    label, combined_value, receipt = next(iter(combined))
    if bank_receipt != receipt:
        return ()
    return (f"{year}년 순이자마진(NIM)은 계산 대상에 따라 다릅니다. (근거: 접수번호 {receipt})",
            f"- 은행만: {bank_value}", f"- {label}: {combined_value}",
            "은행만 계산한 값과 은행·카드를 함께 계산한 값이므로 같은 기준의 숫자로 비교하면 안 됩니다.")


def _monthly_capacity_year_rows(claims, question: str | None) -> tuple[str, ...]:
    """Preserve a source's monthly basis and requested year columns."""
    query = _compact_key(question or "")
    if "생산능력" not in query or "월기준" not in query:
        return ()
    years = tuple(dict.fromkeys(re.findall(r"(20\d{2})\s*년\s*말", question or "")))
    if not years:
        return ()
    candidates = set()
    for claim in claims:
        lines = _source_body(claim.text).splitlines()
        for index, line in enumerate(lines[:-2]):
            header = _table_cells(line)
            context = "\n".join(lines[max(0, index - 4):index])
            if not re.search(r"단위\s*[:：]\s*톤\s*/\s*월", context):
                continue
            if not _is_table_rule(_table_cells(lines[index + 1])):
                continue
            columns = [[j for j, cell in enumerate(header) if re.fullmatch(
                year + r"\s*년\s*말", cell.strip())] for year in years]
            if any(len(column) != 1 for column in columns):
                continue
            rows = []
            for raw in lines[index + 2:]:
                if not raw.lstrip().startswith("|"):
                    break
                rows.append(_table_cells(raw))
            if len(rows) != 1 or len(rows[0]) != len(header):
                continue
            values = tuple(rows[0][column[0]].strip() for column in columns)
            if not all(re.fullmatch(r"약?\s*[\d,]+(?:\.\d+)?톤", value) for value in values):
                continue
            for receipt in _receipt_numbers((claim,)):
                candidates.add((rows[0][0], values, receipt))
    if len(candidates) != 1:
        return ()
    label, values, receipt = next(iter(candidates))
    return (f"{label} 생산능력은 월 기준입니다. (근거: 접수번호 {receipt})",
            *(f"- {year}년 말: 월 {value}" for year, value in zip(years, values)),
            "공시에 표시된 생산능력이며 실제 생산량과는 다릅니다.")


def _nonmoney_scalar_rows(claims, question: str | None) -> tuple[str, ...]:
    """Select a source-labelled ratio/capacity cell with an explicit year."""

    query = _compact_key(question or "").casefold()
    if "얼마" not in query or any(word in query for word in ("비교", "각각", "모두")):
        return ()
    years = set(re.findall(r"20\d{2}", question or ""))
    if len(years) != 1:
        return ()
    year = next(iter(years))
    if "고정이하여신비율" in query and "은행" not in query:
        values: set[str] = set()
        receipts: list[str] = []
        for claim in claims:
            text = " ".join(_source_body(claim.text).split())
            match = re.search(
                year + r"년\s*말\s*그룹\s*연결기준\s*고정이하여신비율"
                r"(?:\([^)]*\))?([^。]*?시현하였습니다\.)", text)
            if match is None:
                continue
            ratios = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?\s*%)(?!p)", match.group(1))
            if len(ratios) == 1:
                values.add(ratios[0].replace(" ", ""))
                receipts.extend(_receipt_numbers((claim,)))
        if len(values) == 1 and len(set(receipts)) == 1:
            return (f"{year}년 말 그룹 연결 기준 고정이하여신비율은 {next(iter(values))}입니다. "
                    f"(근거: 접수번호 {receipts[0]})",
                    "빌려준 돈 중 회수에 문제가 생길 가능성이 높은 여신이 차지하는 비율입니다.")
    kind = ("bis" if ("bis기준자기자본비율" in query
                      and "그룹" in query and "연결" in query
                      and not any(word in query for word in ("은행", "자회사", "별도"))) else
            "nim" if "순이자마진" in query or "nim" in query else
            "kics" if "kics" in query else
            "capacity" if "생산능력" in query else "")
    if not kind:
        return ()
    candidates: set[tuple[str, str, str]] = set()
    receipts: list[str] = []
    for claim in claims:
        lines = _source_body(claim.text).splitlines()
        for start in range(len(lines) - 2):
            if not lines[start].lstrip().startswith("|"):
                continue
            header = _table_cells(lines[start])
            if not _is_table_rule(_table_cells(lines[start + 1])):
                continue
            end = start + 2
            while end < len(lines) and lines[end].lstrip().startswith("|"):
                end += 1
            rows = [_table_cells(line) for line in lines[start + 2:end]]
            rows = [row for row in rows if len(row) == len(header)]
            context = "\n".join(lines[max(0, start - 6):start])
            following = []
            for line in lines[end:end + 5]:
                if line.lstrip().startswith("|"):
                    break
                following.append(line)
            context_key = _compact_key(context + " ".join(following)).casefold()
            if kind == "bis" and ("그룹" not in context_key
                                  or "연결" not in context_key):
                continue
            if kind == "kics" and "kics" not in context_key:
                continue
            if kind == "capacity" and ("생산능력" not in context_key or len(rows) != 1):
                continue
            columns = [i for i, value in enumerate(header)
                       if set(re.findall(r"20\d{2}", value)) == {year}
                       and not any(word in value for word in ("증감", "비중"))]
            if len(columns) != 1:
                continue
            position = columns[0]
            for row in rows:
                labels = row[:position]
                label = next((value for value in labels if (
                    (kind == "nim" and re.match(r"NIM(?:\(|$)", value.strip(), re.I))
                    or (kind == "bis" and "bis기준자기자본비율" in _compact_key(value).casefold())
                    or (kind == "kics" and "지급여력비율" in _compact_key(value))
                    or kind == "capacity")), "")
                if not label:
                    continue
                if (kind == "nim" and "은행" in query
                        and "카드" not in query and "그룹" not in query
                        and ("+" in label or "그룹" in context_key)):
                    continue
                raw = row[position].strip()
                if kind in {"nim", "kics", "bis"}:
                    if not re.fullmatch(r"\d+(?:\.\d+)?%?", raw):
                        continue
                    if not raw.endswith("%") and "%" not in context:
                        continue
                    value = raw if raw.endswith("%") else raw + "%"
                    detail = ("BIS 기준 자기자본비율은 위험을 고려한 자산 대비 자기자본의 비율입니다."
                              if kind == "bis" else
                              "NIM은 이자로 번 이익이 운용자산에서 차지하는 비율입니다."
                              if kind == "nim" else
                              "K-ICS는 보험사가 위험에 대비해 필요한 자본을 얼마나 갖췄는지 나타내는 비율입니다.")
                else:
                    unit = re.search(r"단위\s*[:：]\s*([^)]*)", context)
                    if unit is None or not re.fullmatch(r"약?\s*[\d,]+(?:\.\d+)?톤", raw):
                        continue
                    if unit.group(1).strip() not in {"톤/월", "톤/년", "톤"}:
                        continue
                    period = {"톤/월": "월 ", "톤/년": "연 ", "톤": ""}[unit.group(1).strip()]
                    value = period + raw
                    detail = "공시에 표시된 생산능력이며 실제 생산량과는 다릅니다."
                if kind == "bis":
                    label = "그룹 연결 BIS 기준 자기자본비율"
                candidates.add((label.strip(), value, detail))
                receipts.extend(_receipt_numbers((claim,)))
    if len(candidates) != 1 or len(set(receipts)) != 1:
        return ()
    label, value, detail = next(iter(candidates))
    return (f"{year}년 {label}은 {value}입니다. (근거: 접수번호 {receipts[0]})", detail)


def _shareholder_count_rows(claims, question: str | None) -> tuple[str, ...]:
    """Read two explicitly requested person counts, not stock counts."""

    query = _compact_key(question or "")
    if (not all(term in query for term in ("소액주주수", "전체주주수"))
            or any(term in query for term in ("비율", "지분", "주식수", "비교"))):
        return ()
    years = set(re.findall(r"20\d{2}", question or ""))
    if len(years) > 1:
        return ()
    found = set()
    for claim in claims:
        lines = _source_body(claim.text).splitlines()
        for start in range(len(lines) - 2):
            header = [_compact_key(cell) for cell in _table_cells(lines[start])]
            positions = [[i for i, cell in enumerate(header) if cell.endswith(role)]
                         for role in ("소액주주수", "전체주주수")]
            if any(len(indices) != 1 for indices in positions):
                continue
            if not _is_table_rule(_table_cells(lines[start + 1])):
                continue
            dated = re.search(r"(20\d{2})년\s*(\d{1,2})월\s*(\d{1,2})일",
                              "\n".join(lines[max(0, start - 7):start]))
            if dated is None or (years and dated.group(1) not in years):
                continue
            for line in lines[start + 2:]:
                if not line.lstrip().startswith("|"):
                    break
                row = _table_cells(line)
                if len(row) != len(header):
                    return ()
                values = tuple(row[indices[0]].strip() for indices in positions)
                if not all(re.fullmatch(r"[\d,]+", value) for value in values):
                    return ()
                receipts = _receipt_numbers((claim,))
                if len(receipts) != 1:
                    return ()
                date = f"{dated.group(1)}-{int(dated.group(2)):02d}-{int(dated.group(3)):02d}"
                found.add((date, *values, receipts[0]))
    if len(found) != 1:
        return ()
    date, small, total, receipt = next(iter(found))
    return (f"{date} 기준 소액주주 수는 {small}명, 전체 주주 수는 {total}명입니다. "
            f"(근거: 접수번호 {receipt})",)


def _shareholder_ownership_rows(claims, question: str | None) -> tuple[str, ...]:
    """Project the issuer's ownership table, never its shareholder's accounts."""

    query = _compact_key(question or "")
    if ("최대주주" not in query or "주식소유" not in query
            or any(word in query for word in ("기초", "변동", "비교"))):
        return ()
    years = set(re.findall(r"20\d{2}", question or ""))
    if len(years) > 1:
        return ()
    found: dict[tuple[str, ...], tuple[str, ...]] = {}
    receipts: list[str] = []
    dates: set[str] = set()
    for claim in claims:
        lines = _source_body(claim.text).splitlines()
        for start in range(len(lines) - 2):
            if not lines[start].lstrip().startswith("|"):
                continue
            header = _table_cells(lines[start])
            compact = [_compact_key(cell) for cell in header]
            identities = ("성명", "관계", "주식의종류")
            if not all(compact.count(role) == 1 for role in identities):
                continue
            if not _is_table_rule(_table_cells(lines[start + 1])):
                continue
            amount = [i for i, cell in enumerate(compact)
                      if "기말" in cell and cell.endswith("주식수")]
            ratio = [i for i, cell in enumerate(compact)
                     if "기말" in cell and cell.endswith("지분율")]
            if len(amount) != 1 or len(ratio) != 1:
                continue
            context = "\n".join(lines[max(0, start - 7):start])
            dated = re.search(r"(20\d{2})년\s*(\d{1,2})월\s*(\d{1,2})일", context)
            if dated is None or (years and dated.group(1) not in years):
                continue
            positions = [compact.index(role) for role in identities] + amount + ratio
            table_rows = []
            for line in lines[start + 2:]:
                if not line.lstrip().startswith("|"):
                    break
                row = _table_cells(line)
                if len(row) != len(header) or _is_table_rule(row):
                    return ()  # do not silently omit an unparseable owner
                selected = tuple(row[position].strip() for position in positions)
                if (not all(selected[:3])
                        or not re.fullmatch(r"[\d,]+", selected[3])
                        or not re.fullmatch(r"\d+(?:\.\d+)?%?", selected[4])):
                    return ()
                table_rows.append(selected)
            if not table_rows:
                continue
            for selected in table_rows:
                identity = selected[:3]
                if identity in found and found[identity] != selected:
                    return ()
                found[identity] = selected
            dates.add(f"{dated.group(1)}-{int(dated.group(2)):02d}-{int(dated.group(3)):02d}")
            receipts.extend(_receipt_numbers((claim,)))
    if not found or len(set(receipts)) != 1 or len(dates) != 1:
        return ()
    return (
        f"{next(iter(dates))} 기준 최대주주 및 특수관계인의 기말 주식소유 현황입니다. "
        f"(근거: 접수번호 {receipts[0]})",
        "| 성명 | 관계 | 주식 종류 | 보유 주식 수(주) | 지분율(%) |",
        "|---|---|---|---:|---:|",
        *("| " + " | ".join(_markdown_cell(cell) for cell in row) + " |"
          for row in found.values()),
    )


def _affiliate_list_rows(claims, question: str | None) -> tuple[str, ...]:
    query = _compact_key(question or "")
    if "계열회사" not in query or "상장" not in query:
        return ()
    unlisted_only = "비상장" in query and bool(re.search(r"비상장.*(?:회사명|이름|기업명)|비상장.*만", query))
    if "비상장" in query and not unlisted_only:
        return ()
    wanted_status = "비상장" if unlisted_only else "상장"
    names: list[str] = []
    counts: dict[str, set[str]] = {}
    receipts: list[str] = []
    domestic = any(re.search(r"국내\s*계열회사", _source_body(claim.text))
                   for claim in claims)
    for claim in claims:
        lines = _source_body(claim.text).splitlines()
        for start, line in enumerate(lines[:-2]):
            header = [_compact_key(value) for value in _table_cells(line)]
            if not all(header.count(role) == 1 for role in ("상장여부", "회사수", "기업명")):
                continue
            if not _is_table_rule(_table_cells(lines[start + 1])):
                continue
            positions = [header.index(role) for role in ("상장여부", "회사수", "기업명")]
            for raw in lines[start + 2:]:
                if not raw.lstrip().startswith("|"):
                    break
                row = _table_cells(raw)
                if len(row) != len(header):
                    continue
                status, count, name = (row[pos].strip() for pos in positions)
                if status not in {"상장", "비상장"} or not count.isdecimal():
                    continue
                counts.setdefault(status, set()).add(count)
                if status == wanted_status and name not in names:
                    names.append(name)
            receipts.extend(_receipt_numbers((claim,)))
    if (not names or any(len(values) != 1 for values in counts.values())
            or len(set(receipts)) != 1):
        return ()
    listed = int(next(iter(counts.get("상장", {"0"}))))
    selected_count = int(next(iter(counts.get(wanted_status, {"0"}))))
    if selected_count != len(names) or ("국내" in query and not domestic):
        return ()
    unlisted = next(iter(counts.get("비상장", ())), None)
    count_line = f"상장 계열회사는 {listed}개사입니다."
    if unlisted_only:
        count_line = f"비상장 계열회사는 {selected_count}개사입니다."
    elif unlisted is not None:
        count_line = (f"계열회사는 총 {listed + int(unlisted)}개사로, "
                      f"상장 {listed}개사와 비상장 {unlisted}개사입니다.")
    if domestic:
        count_line = "국내 " + count_line
    return (count_line + f" (근거: 접수번호 {receipts[0]})",
            wanted_status + " 계열회사: " + ", ".join(names))


def _scoped_metric_prose_rows(claims, question: str | None) -> tuple[str, ...]:
    """Use an explicit scalar sentence instead of unrelated/ragged tables.

    Only two bounded metric shapes are supported. No numbers are calculated,
    and extra requested axes keep the existing complete rendering route.
    """
    query = _compact_key(question or "")
    if "얼마" not in query or any(word in query for word in (
            "비교", "추이", "각각", "생산능력", "생산실적", "품목별", "부문별")):
        return ()
    if "DX부문가동률" in query:
        pattern = re.compile(
            r"당사\s*DX\s*부문의\s*(20\d{2})년(?:\(제\d+기\))?\s*가동률은[^\n.]*?"
            r"\d+(?:\.\d+)?%[^\n]*?입니다\.")
    elif "수주잔고" in query:
        if any(word in query for word in ("연결", "합계", "부문", "상선", "방산", "기체", "위성")):
            return ()
        pattern = re.compile(
            r"보고서\s*작성일\s*기준\s*당사의\s*별도재무제표\s*기준\s*"
            r"수주\s*잔고는\s*[\d,]+조\s*[\d,]+억원입니다\.")
    else:
        return ()
    matches: set[tuple[str, tuple[str, ...]]] = set()
    for claim in claims:
        for match in pattern.finditer(_source_body(claim.text)):
            sentence = match.group(0)
            if "DX부문가동률" in query:
                requested_years = set(re.findall(r"20\d{2}", query))
                if requested_years and requested_years != {match.group(1)}:
                    continue
            receipts = _receipt_numbers((claim,))
            if receipts:
                matches.add((sentence, tuple(receipts)))
    if len(matches) != 1:
        return ()
    sentence, receipts = next(iter(matches))
    citation = "; ".join("접수번호 " + value for value in receipts)
    return (sentence + f" (근거: {citation})",)


def _license_contract_rows(claims, question: str | None) -> tuple[str, ...]:
    """Keep a license-contract table's stages out of merged monetary cells."""
    query = _compact_key(question or "")
    if not ("라이선스" in query or "기술이전" in query) or "계약현황" not in query:
        return ()
    if any(word in query for word in ("연구개발", "파이프라인", "비교", "추이")):
        return ()
    required = ["품목", "계약상대방", "대상지역", "계약체결일", "계약종료일",
                "총계약금액", "수취금액", "진행단계"]
    found: list[tuple[str, ...]] = []
    receipts: list[str] = []
    for claim in claims:
        lines = _source_body(claim.text).splitlines()
        for index, line in enumerate(lines[:-2]):
            if [_compact_key(cell) for cell in _table_cells(line)] != required:
                continue
            if not _is_table_rule(_table_cells(lines[index + 1])):
                continue
            for line in lines[index + 2:]:
                if not line.lstrip().startswith("|"):
                    break
                row = _table_cells(line)
                # Missing money cells must not shift a clinical stage into a
                # monetary column. Only an explicit stage in the last cell
                # makes the seven-cell variant safe to project.
                if len(row) not in (7, 8) or (len(row) == 7 and not re.search(
                        r"임상|허가|출시|CRL|NDA|MAA|BLA|\bP[123]\b", row[-1])):
                    return ()
                money = (f"총계약금액: {row[5]}; 수취금액: {row[6]}"
                         if len(row) == 8 else row[5])
                rendered = (*row[:5], money, row[-1])
                if rendered not in found:
                    found.append(rendered)
            receipts.extend(_receipt_numbers((claim,)))
    if not found or len(set(receipts)) != 1:
        return ()
    header = (*required[:5], "금액 관련 원문", "진행단계")
    return (f"공시에 기재된 기술이전 계약 현황입니다. (근거: 접수번호 {receipts[0]})",
            "금액 항목이 구분되지 않은 행은 원문 표현을 그대로 표시했습니다.",
            "| " + " | ".join(header) + " |",
            "|" + "|".join("---" for _ in header) + "|",
            *("| " + " | ".join(_markdown_cell(value) for value in row) + " |"
              for row in found))


def _rnd_classification_rows(claims, question: str | None) -> tuple[str, ...]:
    query = _compact_key(question or "")
    if not all(word in query for word in ("연구개발비", "성격별", "회계처리", "합계")):
        return ()
    years = set(re.findall(r"20\d{2}", query))
    if len(years) != 1:
        return ()
    year = next(iter(years))
    candidates = set()
    for claim in claims:
        lines = _source_body(claim.text).splitlines()
        for start, line in enumerate(lines[:-2]):
            header = _table_cells(line)
            columns = [i for i, cell in enumerate(header) if year in cell]
            if len(columns) != 1 or not _is_table_rule(_table_cells(lines[start + 1])):
                continue
            unit = _local_table_currency_unit(lines, start)
            if unit not in {"원", "천원", "백만원", "억원", "조원"}:
                continue
            totals: dict[str, set[str]] = {}
            for raw in lines[start + 2:]:
                if not raw.lstrip().startswith("|"):
                    break
                row = _table_cells(raw)
                if len(row) != len(header) or len(row) < 3:
                    continue
                if not re.fullmatch(r"연구개발비(?:용)?합계", _compact_key(row[1])):
                    continue
                role = ("비용 성격별" if "성격별" in _compact_key(row[0]) else
                        "회계처리별" if "회계처리" in _compact_key(row[0]) else "")
                value = row[columns[0]].strip()
                if role and re.fullmatch(r"[\d,]+(?:\.\d+)?", value):
                    totals.setdefault(role, set()).add(value)
            if set(totals) != {"비용 성격별", "회계처리별"} or any(len(v) != 1 for v in totals.values()):
                continue
            for receipt in _receipt_numbers((claim,)):
                candidates.add((next(iter(totals["비용 성격별"])),
                                next(iter(totals["회계처리별"])), unit, receipt))
    if len(candidates) != 1:
        return ()
    cost, accounting, unit, receipt = next(iter(candidates))
    equal = Decimal(cost.replace(",", "")) == Decimal(accounting.replace(",", ""))
    return (f"{year}년 연구개발비 합계입니다. (근거: 접수번호 {receipt})",
            f"- 비용 성격별: {cost}{unit}", f"- 회계처리별: {accounting}{unit}",
            "같은 연구개발비를 무엇에 썼는지와 회계상 어디에 기록했는지로 나눠 보여준 것입니다. 두 합계를 더하면 같은 비용을 두 번 세게 됩니다."
            if equal else "두 분류의 합계가 다르게 표시되어 있어 차이의 원인을 추가로 확인해야 합니다. 두 합계를 그대로 더하지 마세요.")


def _acquisition_commitment_rows(claims, question: str | None) -> tuple[str, ...]:
    """Keep an explicitly empty acquisition amount distinct from zero.

    Only a uniquely matched source column and an explicit amount row provide
    absence evidence. Missing search results never establish an empty cell.
    """
    query = _compact_key(question or "").casefold()
    if "인수약정" not in query or "금액" not in query:
        return ()
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9]{1,}\b", question or "")
    results: set[tuple[str, str, str]] = set()
    for claim in claims:
        if "연결" in query and "연결" not in claim.label:
            continue
        lines = _source_body(claim.text).splitlines()
        for start, line in enumerate(lines[:-2]):
            header = _table_cells(line)
            if not _is_table_rule(_table_cells(lines[start + 1])):
                continue
            columns = [i for i, label in enumerate(header)
                       if "인수약정" in _compact_key(label)
                       and (not tokens or all(token.casefold() in label.casefold()
                                               for token in tokens))]
            if len(columns) != 1:
                continue
            for raw in lines[start + 2:]:
                if not raw.lstrip().startswith("|"):
                    break
                row = _table_cells(raw)
                if len(row) != len(header) or _compact_key(row[0]) != "약정금액":
                    continue
                for receipt in _receipt_numbers((claim,)):
                    results.add((header[columns[0]], row[columns[0]].strip(), receipt))
    if len(results) != 1:
        return ()
    label, value, receipt = next(iter(results))
    if value:
        return ()  # the existing numeric renderer owns units and conversion
    return (f"해당 약정 표에서 ‘{label}’의 약정 금액 칸은 비어 있습니다. "
            f"(근거: 접수번호 {receipt})",
            "이 표에는 금액이 적혀 있지 않다는 뜻이며, 인수금액이 0원이거나 비공개라는 뜻은 아닙니다.")


def _contingency_rows(claims, question: str | None) -> tuple[str, ...]:
    query = _compact_key(question or "")
    if "우발부채" not in query or "약정" not in query:
        return ()
    selected = [claim for claim in claims
                if "우발부채" in _compact_key(claim.label)
                and "약정" in _compact_key(claim.label)]
    if "연결" in query:
        selected = [claim for claim in selected if "연결" in claim.label]
    elif "별도" in query or "개별" in query:
        selected = [claim for claim in selected if "연결" not in claim.label]
    rows: list[str] = []
    receipts: list[str] = []
    for claim in selected:
        for line in _source_body(claim.text).splitlines():
            cells = _table_cells(line)
            if len(cells) < 2:
                continue
            role = _compact_key(cells[0])
            if not (role in {"약정에대한설명", "의무의성격에대한기술,우발부채"}
                    or role.startswith("유출될경제적효익의금액과시기에대한불확실성")):
                continue
            for cell in cells[1:]:
                text = " ".join(cell.split())
                if not text or text in rows:
                    continue
                rows.append(text)
        receipts.extend(_receipt_numbers((claim,)))
    if not rows or len(set(receipts)) != 1:
        return ()
    return (f"공시에 나온 우발부채와 약정 내용입니다. (근거: 접수번호 {receipts[0]})",
            *("- " + row for row in rows))


def _clinical_phase_rows(claims, question: str | None) -> tuple[str, ...]:
    query = _compact_key(question or "")
    if not all(term in query for term in ("임상3상", "진행중", "바이오시밀러")):
        return ()
    wants_preparing_new_drugs = "바이오신약" in query and "3상준비중" in query
    names: list[str] = []
    preparing_names: list[str] = []
    receipts: list[str] = []
    for claim in claims:
        lines = _source_body(claim.text).splitlines()
        for start, line in enumerate(lines[:-2]):
            header = [_compact_key(v) for v in _table_cells(line)]
            is_similar = header == ["구분", "바이오시밀러"]
            is_new_drug = wants_preparing_new_drugs and header == ["구분", "바이오신약"]
            if not (is_similar or is_new_drug):
                continue
            caption = next((v.strip() for v in reversed(lines[max(0, start - 3):start])
                            if v.strip()), "")
            name = re.sub(r"^\s*\d+[.)]\s*", "", caption)
            if not name or len(name) > 80 or name.startswith("|"):
                continue
            for raw in lines[start + 2:]:
                if not raw.lstrip().startswith("|"):
                    break
                row = _table_cells(raw)
                if len(row) != 2:
                    continue
                target_names = None
                if (is_similar and _compact_key(row[0]) == "진행경과"
                        and re.fullmatch(r"임상\s*3상\s*진행\s*중", row[1].strip())):
                    target_names = names
                elif (is_new_drug and _compact_key(row[0]) == "향후계획"
                        and re.fullmatch(r"임상\s*3상\s*준비\s*중", row[1].strip())):
                    target_names = preparing_names
                if target_names is not None:
                    if name not in target_names:
                        target_names.append(name)
                    receipts.extend(_receipt_numbers((claim,)))
    if not names or len(set(receipts)) != 1:
        return ()
    rows = [f"공시에서 임상 3상 진행 중으로 표시한 바이오시밀러는 {len(names)}개입니다. "
            f"(근거: 접수번호 {receipts[0]})",
            *("- " + name + ": 임상 3상 진행 중" for name in names)]
    if wants_preparing_new_drugs:
        if preparing_names:
            rows.extend((
                f"공시에서 임상 3상 준비 중으로 표시한 바이오신약은 {len(preparing_names)}개입니다. "
                f"(근거: 접수번호 {receipts[0]})",
                *("- " + name + ": 임상 3상 준비 중" for name in preparing_names),
                "'준비 중'은 3상 시험을 진행 중이라는 뜻과 다릅니다.",
            ))
        else:
            rows.append("다만, 요청하신 임상 3상 준비 중인 바이오신약은 이번에 확인한 근거에서 "
                        "품목과 준비 상태를 함께 확인하지 못했습니다. 해당 품목이 없다는 뜻은 아닙니다.")
    return tuple(rows)


def _registered_director_rows(claims, question: str | None) -> tuple[str, ...]:
    """Project an explicitly registered-director table, not its date header."""

    query = _compact_key(question or "")
    if ("등기임원" not in query or any(term in query for term in (
            "미등기", "후보", "해임", "보수", "겸직"))):
        return ()
    found: dict[tuple[str, ...], tuple[str, ...]] = {}
    receipts: list[str] = []
    for claim in claims:
        lines = _source_body(claim.text).splitlines()
        for start in range(len(lines) - 2):
            if not lines[start].lstrip().startswith("|"):
                continue
            header = _table_cells(lines[start])
            compact = [_compact_key(value) for value in header]
            required = ("성명", "직위", "등기임원여부")
            if not all(compact.count(role) == 1 for role in required):
                continue
            if not _is_table_rule(_table_cells(lines[start + 1])):
                continue
            roles = (*required, "담당업무") if "담당업무" in compact else required
            positions = [compact.index(role) for role in roles]
            for line in lines[start + 2:]:
                if not line.lstrip().startswith("|"):
                    break
                row = _table_cells(line)
                if len(row) != len(header) or _is_table_rule(row):
                    continue
                values = tuple(row[position].strip() for position in positions)
                if not all(values[:3]) or values[2] not in {"사내이사", "사외이사", "기타비상무이사"}:
                    continue
                if "사외이사만" in query and values[2] != "사외이사":
                    continue
                found[values] = roles
            receipts.extend(_receipt_numbers((claim,)))
    if (not found or len(set(found.values())) != 1
            or len(set(receipts)) != 1):
        return ()
    roles = next(iter(found.values()))
    citation = "; ".join("접수번호 " + value for value in dict.fromkeys(receipts))
    return (
        f"공시 기준 {'사외이사는' if '사외이사만' in query else '등기임원은'} {len(found)}명입니다. (근거: {citation})",
        "| " + " | ".join(roles) + " |",
        "|" + "|".join("---" for _ in roles) + "|",
        *("| " + " | ".join(_markdown_cell(value) for value in row) + " |"
          for row in found),
    )


@dataclass(frozen=True)
class NarrativeDigest:
    """A bounded deterministic renderer for one long-form narrative payload.

    The public AnswerClaim remains the full source-roundtripped structural
    block.  This sidecar exists solely at composition time so a long paragraph
    or a plan table is not displayed as a raw source dump.  It is deliberately
    unavailable unless execution recorded a narrative sidecar and every claim
    is source-roundtripped narrative text.
    """

    lines: tuple[str, ...]
    limitations: tuple[str, ...]

    @classmethod
    def from_payload(
            cls, payload, *, question: str | None = None,
            ) -> "NarrativeDigest | None":
        # Matrix output keeps its richer typed coordinate renderer.  This
        # class is intentionally only the single-coordinate/long-form route.
        if NarrativeMatrix.from_payload(payload) is not None:
            return None
        sidecars = tuple(getattr(payload, "narrative_sidecars", ()) or ())
        if not sidecars or not getattr(payload, "claims", ()):
            return None
        claims = tuple(payload.claims)
        if any((not claim.text or claim.value_text or claim.state or claim.operator
                or not claim.citations
                or any(ct.verification != "source_roundtrip" for ct in claim.citations))
               for claim in claims):
            return None

        public_limitations = [
            limitation for limitation in payload.limitations
            if limitation.code.split(":", 1)[0] != "source_cross_check_partial"
        ]
        notes = list(render_safe_limitations(public_limitations))
        for limitation in public_limitations:
            if limitation.code in _NARRATIVE_LIMITS and limitation.detail not in notes:
                notes.append(limitation.detail)

        for project in (_rnd_classification_rows, _bank_card_nim_rows, _monthly_capacity_year_rows, _acquisition_commitment_rows, _scoped_metric_prose_rows, _license_contract_rows,
                        _shareholder_count_rows, _shareholder_ownership_rows, _affiliate_list_rows, _contingency_rows, _clinical_phase_rows,
                        _nonmoney_scalar_rows):
            projected = project(claims, question)
            if projected:
                return cls(lines=projected, limitations=tuple(notes))

        director_rows = _registered_director_rows(claims, question)
        if director_rows:
            return cls(lines=director_rows, limitations=tuple(notes))

        scalar = _unique_narrative_money_cell(claims, question)
        if scalar is not None:
            row_label, column_label, rendered, citation = scalar
            amount_label = column_label if column_label.endswith("금액") else f"{column_label} 금액"
            return cls(
                lines=(
                    f"{row_label} {amount_label}은 {rendered}입니다. "
                    f"(근거: {citation})",
                ),
                limitations=tuple(notes),
            )

        topic_by_output_id: dict[str, str] = {}
        for result in sidecars:
            plan = getattr(result, "plan", None)
            cells = {
                getattr(cell, "cell_id", ""): getattr(cell, "topic", "")
                for cell in (getattr(plan, "cells", ()) or ())
            }
            for cell_id, output_ids in (
                    getattr(result, "cell_claim_output_ids", ()) or ()):
                topic = cells.get(cell_id, "")
                if not topic:
                    continue
                for output_id in output_ids:
                    topic_by_output_id.setdefault(output_id, topic)

        rows: list[str] = []
        narratives: list[str] = []
        for claim in claims:
            receipts = _receipt_numbers((claim,))
            if not receipts:
                return None
            citation = "; ".join("접수번호 " + receipt for receipt in receipts)
            public_label = _public_narrative_label(claim.label)
            # A broad ``투자 현황`` coordinate must not be silently narrowed
            # merely because its cited subsection happens to contain a
            # structurally complete project row.  Preserve the broader table
            # plus directly related acquisition/facility notes first.
            if re.search(r"\[확인 항목:\s*투자현황\s*\]", claim.label):
                general_investment = _general_investment_narrative_digest(
                    claim.text, limit=1100)
                if general_investment:
                    rows.append(
                        f"- {public_label}: {general_investment} "
                        f"(근거: {citation})")
                    continue
            investment = _investment_rows(claim.text)
            if not investment and "투자현황" in _compact_key(claim.text or ""):
                status = _general_investment_narrative_digest(claim.text, limit=1100)
                if status and "계획" not in _compact_key(claim.text or ""):
                    rows.append(f"- {public_label}: {status} (근거: {citation})")
                    if question and "계획" in question:
                        notes.append(
                            "확인한 표는 투자 현황입니다. 이 표의 투자액을 "
                            "앞으로 쓸 계획 금액으로 볼 수는 없습니다.")
                    continue
            if investment:
                # A fixed-column table is substantially easier to scan than a
                # nested comma-separated list, especially when period and
                # spent-to-date values are long.  It changes presentation
                # only: every row remains bound to the same verified receipt.
                rows.append(_investment_summary(
                    tuple(row[0] for row in investment),
                    has_execution=any(row[5] is not None for row in investment),
                    citation=citation))
                rows.extend((
                    _public_table_caption(public_label, (
                        "투자 대상", "목적", "계획 금액", "기간", "집행액")),
                    "| 투자 대상 | 목적 | 계획 금액 | 기간 | 집행액 | 근거 |",
                    "|---|---|---:|---|---:|---|",
                ))
                for target, purpose, amount, period, unit, spent in investment:
                    rows.append("| " + " | ".join((
                        _markdown_cell(target),
                        _markdown_cell(purpose),
                        _markdown_cell(_investment_amount_display(amount, unit)),
                        _markdown_cell(period),
                        _markdown_cell(
                            _investment_amount_display(spent, unit)
                            if spent else "확인되지 않음"),
                        _markdown_cell(citation),
                    )) + " |")
                if any(spent is not None and spent.strip() == "-"
                       for _target, _purpose, _amount, _period, _unit, spent
                       in investment):
                    rows.append(
                        "집행액의 '-'는 원문 표기입니다. 이 표시만으로 "
                        "0원 또는 미집행으로 판단할 수 없습니다.")
                continue
            plan_table = _investment_plan_table(claim.text)
            if plan_table:
                header, plan_rows, unit = plan_table
                rows.extend((
                    _investment_plan_summary(header, plan_rows, citation=citation),
                    _public_table_caption(public_label, header)
                    + (f" (단위: {unit})" if unit else ""),
                    "| " + " | ".join(
                        _markdown_cell(value) for value in (*header, "근거")) + " |",
                    "|" + "|".join("---:" if re.search(
                        r"(?:금액|투자계획|실적|기지출|집행|기투자)",
                        _compact_key(value)) else "---"
                        for value in (*header, "근거")) + "|",
                    *("| " + " | ".join(
                        _markdown_cell(value) for value in (*row, citation)) + " |"
                      for row in plan_rows),
                ))
                continue
            plan = _investment_plan_digest(claim.text, limit=1100)
            if plan:
                rows.append(f"- {public_label}: {plan} (근거: {citation})")
                continue
            if "투자현황" in _compact_key(claim.label):
                general_investment = _general_investment_narrative_digest(
                    claim.text, limit=1100)
                if general_investment:
                    rows.append(
                        f"- {public_label}: {general_investment} "
                        f"(근거: {citation})")
                    continue
            label_key = _compact_key(claim.label)
            if "투자부동산" in label_key:
                property_summary = _investment_property_digest(
                    claim.text, limit=1100)
                if property_summary:
                    rows.append(
                        f"- {public_label}: {property_summary} "
                        f"(근거: {citation})")
                    continue
            topic_match = re.search(r"\[확인 항목:\s*([^]]+)\]", claim.label)
            topic = topic_by_output_id.get(claim.output_id, "")
            if not topic and topic_match:
                topic = topic_match.group(1).split(",", 1)[0].strip()
            if not topic:
                if any(marker in label_key for marker in (
                        "주요제품", "제품및서비스")):
                    topic = "주요 제품 및 서비스"
                elif any(marker in label_key for marker in (
                        "수익구조", "매출구성", "영업의현황")):
                    topic = "수익구조"
                elif any(marker in label_key for marker in (
                        "사업의개요", "사업부문", "주요사업")):
                    topic = "주요 사업"
            # A single verified coordinate has a larger display budget than a
            # matrix cell.  This preserves all major source rows (rather than
            # only the first two) while remaining below the existing typed
            # digest boundary used for investment tables.
            summary = (_compact_cell_text(claim.text, topic=topic, limit=1100)
                       if topic else _compact_narrative_text(
                           claim.text, label=public_label))
            if summary:
                narratives.append(
                    f"- {public_label}: {_public_narrative_surface(summary)} "
                    f"(근거: {citation})")
        lines = tuple(dict.fromkeys(rows + narratives))
        if not lines:
            return None
        return cls(lines=lines, limitations=tuple(notes))

    def deterministic(self) -> str:
        # 콜론으로 연다 — 마침표로 끝내면(구 「…입니다.」) `_hoist_run_citations`
        # 가 뒤이은 불릿의 공통 근거를 이 머리글에 얹으며 끝에 `:`를 더 붙이고,
        # `_collect_uniform_citations` 가 그 근거만 걷어 가면서 겹친 `:`는
        # 지우지 못해 「…입니다.:」로 남았다(issue #117). 머리글이 이미 `:`로
        # 끝나면 앞의 `.rstrip(":")` 가 그 하나를 먼저 떼어 내므로 되돌아와도
        # 다시 하나만 남는다.
        title = "공시에서 확인한 내용:"
        if any("| 투자 대상 | 목적 | 계획 금액 | 기간 |" in line
               or "투자계획 표(" in line
               or (line.startswith("|") and "투자계획" in line)
               for line in self.lines):
            title = "공시에서 확인한 투자계획:"
        elif any("투자 현황 표(" in line for line in self.lines):
            title = "공시에서 확인한 투자 현황:"
        lines = [title, *self.lines]
        answer = "\n".join(lines)
        if self.limitations:
            answer = append_public_qualifications(answer, self.limitations)
        return _public_narrative_surface(answer)


def verify_matrix_output(text: str, matrix: NarrativeMatrix) -> tuple[bool, str, dict]:
    """A concise HCX answer must remain tied to every matrix receipt and token."""
    if not text:
        return False, "빈 응답", {}
    missing_receipts = [receipt for receipt in matrix.receipts if receipt not in text]
    output_lines = (text or "").splitlines()
    # HCX often shortens ``task-1/cell-1`` to ``cell-1``.  Receipt numbers are
    # the authoritative binding, so accept that harmless display shortening
    # while assigning every cell to a distinct cited output line.  If two
    # cells share the same receipts, retain the strict full coordinate marker
    # because the receipt alone cannot disambiguate them.
    receipt_counts = {
        cell.receipts: sum(other.receipts == cell.receipts for other in matrix.cells)
        for cell in matrix.cells
    }
    available_lines = set(range(len(output_lines)))
    missing_cell_citations = []
    misaligned_cells = []
    for cell in matrix.cells:
        candidates = [index for index in available_lines
                      if all(receipt in output_lines[index] for receipt in cell.receipts)
                      and (receipt_counts[cell.receipts] == 1 or cell.cell_id in output_lines[index])]
        if not candidates:
            missing_cell_citations.append(cell.cell_id)
            continue
        selected_line = output_lines[candidates[0]]
        available_lines.remove(candidates[0])
        topic = re.sub(r"\s+", "", cell.topic)
        if "주요사업" in topic:
            company = cell.label.split(" · ", 1)[0]
            answer_body = selected_line.replace(company, "")
            source_markers = {
                marker for marker in _BUSINESS_EVIDENCE_MARKERS
                if marker in cell.excerpt
            }
            if source_markers and not any(
                    marker in answer_body for marker in source_markers):
                misaligned_cells.append(cell.cell_id)
    source = matrix.prompt()
    source_numbers = {token.replace(",", "") for token in _NUM.findall(source)
                      if len(token.replace(",", "")) >= 3}
    answer_numbers = {token.replace(",", "") for token in _NUM.findall(text)
                      if len(token.replace(",", "")) >= 3}
    unknown = sorted(answer_numbers - source_numbers - matrix.derived_number_tokens)
    missing_limits = [note for note in matrix.limitations if note not in text]
    if missing_receipts:
        return False, "근거 접수번호 누락", {"missing_receipts": missing_receipts}
    if missing_cell_citations:
        return False, "셀별 근거 접수번호 누락", {"missing_cell_citations": missing_cell_citations}
    if misaligned_cells:
        return False, "셀 주제 핵심 근거 누락", {"misaligned_cells": misaligned_cells}
    if unknown:
        return False, "미출처 숫자", {"unknown": unknown}
    if missing_limits:
        return False, "한계 고지 누락", {"missing_limitations": missing_limits}
    if not matrix.incomplete:
        conclusion = text.rsplit(f"{matrix.comparison_kind} 결론:", 1)[-1]
        non_conclusions = (
            "제시된 인용 범위만으로는 차이의 내용을 확정하지 않습니다",
            "차이의 내용을 확정하지 않습니다", "변화 내용을 확정하지 않습니다",
            "비교할 수 없습니다", "판단하기 어렵습니다", "확정할 수 없습니다",
            "확정하지 않습니다",
        )
        if any(phrase in conclusion for phrase in non_conclusions):
            return False, "비결론 비교 응답", {}
        if "확인된 각 좌표의 인용 본문은 서로 다릅니다" in conclusion:
            return False, "내용 없는 비교 결론", {}
        compared_labels = matrix.comparison_labels
        missing_labels = [label for label in compared_labels
                          if label not in conclusion]
        missing_conclusion_receipts = [receipt for receipt in matrix.receipts
                                       if receipt not in conclusion]
        if missing_labels:
            return False, "비교 결론 대상 누락", {"missing_labels": missing_labels}
        if missing_conclusion_receipts:
            return False, "비교 결론 근거 누락", {
                "missing_conclusion_receipts": missing_conclusion_receipts}
    return True, "", {}
