"""NarrativeTool — Tier 2 검색 경로 (합의안 §5, 설계서 v3 §6 Tier 2).

SearchIndex.search(as_of 선필터) → top-k SearchHit → rm.read_section(section_id) 왕복 성공한
Section만 SourceReference로 승격 → AnswerClaim(text=prompt_safe 본문 발췌).
검색 점수만 높은 Chunk는 근거가 아니다. 요약·비교 문장은 composer(W5)가 만든다 — 여기서는
근거 있는 본문 후보를 slot별로 정리해 넘긴다.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from decimal import Decimal
import re
from types import SimpleNamespace
from typing import Mapping

from agent.disclosed_metric_topics import (
    disclosed_metric_for_query, disclosed_metrics_for_query, key_present,
    probe_keys, search_terms_for,
)
from app.orchestrator.payload import AnswerClaim, ClaimCitation, Limitation, TraceEvent
from app.markdown_table import is_table_rule as _is_table_rule
from app.markdown_table import table_cells as _table_cells
from src.canonical.chunker import CHUNK_MAX_CHARS, split_section
from src.canonical.read import (
    PERIOD_LABEL, period_label_key, period_label_number,
)
from src.canonical.security import PROMPT_DATA_BEGIN, PROMPT_DATA_END

TOP_K = 10
MAX_SEARCH_TOP_K = 100
# 검색 index가 이미 이 구조 예산으로 표 행/문단을 자르지 않고 발행된다. 답변 층에서
# 더 작은 임의 예산을 덧씌우면 정상 canonical chunk가 다시 partial로 강등된다.
NARRATIVE_BLOCK_CHARS = CHUNK_MAX_CHARS
MAX_BLOCKS_PER_PERIOD = 8
# A matrix must not turn one bounded narrative lookup into an unbounded number
# of index/section reads.  These are execution limits, not QueryPlan fields.
MAX_NARRATIVE_FANOUT_CELLS = 16
MAX_NARRATIVE_RESULT_BLOCKS = 16

_INVESTMENT_SLOT_HEADERS = {
    # DART's equipment-investment form has two equally canonical header
    # families.  A plan table commonly calls the target ``투자명`` and the
    # planned amount ``총 소요자금``; another calls them ``대상자산`` and
    # ``투자액``.  They carry the same four requested roles, while
    # ``기 지출금액`` remains deliberately excluded (it is execution-to-date,
    # not the plan amount).
    "투자대상": ("투자대상", "대상자산", "투자명"),
    "목적": ("투자목적", "목적"),
    "금액": ("투자액", "투자금액", "총소요자금", "총투자액", "금액"),
    "기간": ("투자기간", "기간"),
}
_INVESTMENT_SLOT_ROLES = frozenset(_INVESTMENT_SLOT_HEADERS)
_INVESTMENT_SPENT_AMOUNT_MARKERS = (
    "기지출", "누적지출", "실제지출", "집행금액", "기투자", "투자실적",
)
_INVESTMENT_OPTIONAL_SLOT_HEADERS = {
    "기지출금액": (
        "기지출금액", "기지출액", "누적지출금액", "실제지출금액",
        "집행금액", "기투자금액", "투자실적",
    ),
}

# ``투자 현황을 전반적으로 정리`` is not a request for one project row.
# A periodic filing can place the status narrative and its plan-vs-actual
# table in the same parent section, so the general route needs one explicit
# subsection boundary before it is allowed to retain that parent section.
# References to investment inside prose (for example, an investor-protection
# note saying "투자현황은 ... 참조") are deliberately not headings.
_GENERAL_INVESTMENT_SUBHEADING = re.compile(
    r"(?m)(?:^|[.!?。]\s*)"
    r"(?:\s*(?:\([0-9가-하]+\)|[가-하]|\d+)\s*[.)]?\s*)?"
    r"투자\s*(?:현황|계획)(?:\s*및\s*계획)?\b"
)
# One canonical table row.  A lead-in carrying rows belongs to a different
# disclosure axis (raw materials, production), so it is not general context.
_TABLE_ROW = re.compile(r"(?m)^\s*\|")


def _rcept(doc_id: str) -> str | None:
    m = re.search(r"(\d{14})", doc_id or "")
    return m.group(1) if m else None


def _dedupe_limitations(limitations: list[Limitation]) -> list[Limitation]:
    """Keep repeated cell diagnostics from amplifying the final answer."""
    seen = set()
    out = []
    for limitation in limitations:
        key = (limitation.code, limitation.detail,
               tuple(limitation.affected_doc_ids or ()))
        if key not in seen:
            seen.add(key)
            out.append(limitation)
    return out


@dataclass(frozen=True)
class NarrativeDocumentCoordinate:
    """One selected document (or period window) for matrix expansion."""

    period: object | None
    document_selector: object | None


@dataclass(frozen=True)
class NarrativeCell:
    """One independently verifiable narrative retrieval coordinate.

    This is deliberately an execution-side type.  QueryPlan v0.4 continues to
    carry its single public narrative task; the backend expands that task only
    after its companies, periods and retrieval topic are already grounded.
    """

    cell_id: str
    corp_code: str
    corp_name: str
    period: object | None
    topic: str
    document_selector: object | None = None


@dataclass(frozen=True)
class NarrativeFanoutPlan:
    """Bounded internal sidecar for a public narrative task."""

    task_id: str
    cells: tuple[NarrativeCell, ...]
    omitted_cells: int = 0

    @classmethod
    def from_task(
            cls, task, *, corp_name: str, topics: tuple[str, ...] | None = None,
            documents: tuple[NarrativeDocumentCoordinate, ...] | None = None,
            ):
        # ``topics`` is an internal sidecar input.  In its absence the public
        # retrieval_query is exactly one topic, preserving the v0.4 wire.
        if topics is None:
            attached = getattr(task, "_narrative_topics", None)
            topics = tuple(attached) if isinstance(attached, (tuple, list)) else ()
            if not topics:
                topics = _compound_business_topics(task.retrieval_query)
        normalized_topics = tuple(dict.fromkeys(
            topic.strip() for topic in (topics or (task.retrieval_query,))
            if isinstance(topic, str) and topic.strip()))
        if not normalized_topics:
            normalized_topics = (task.retrieval_query,)

        corp_codes = tuple(task.corp_codes) or ("",)
        supplied_names = tuple(getattr(task, "corp_names", ()) or ())
        corp_names = tuple(
            supplied_names[index] if index < len(supplied_names) and supplied_names[index].strip()
            else (corp_name if len(corp_codes) == 1 else code)
            for index, code in enumerate(corp_codes))
        document_coordinates = documents or tuple(
            NarrativeDocumentCoordinate(period=period,
                                        document_selector=task.document_selector)
            for period in (tuple(task.periods) or (None,)))
        all_cells = tuple(
            NarrativeCell(cell_id="", corp_code=code, corp_name=name,
                          period=document.period, topic=topic,
                          document_selector=document.document_selector)
            for code, name in zip(corp_codes, corp_names, strict=True)
            for document in document_coordinates
            for topic in normalized_topics
        )
        all_cells = tuple(
            NarrativeCell(cell_id=f"cell-{index}", corp_code=cell.corp_code,
                          corp_name=cell.corp_name, period=cell.period, topic=cell.topic,
                          document_selector=cell.document_selector)
            for index, cell in enumerate(all_cells, start=1))
        return cls(task_id=task.task_id,
                   cells=all_cells[:MAX_NARRATIVE_FANOUT_CELLS],
                   omitted_cells=max(0, len(all_cells) - MAX_NARRATIVE_FANOUT_CELLS))


@dataclass(frozen=True)
class NarrativeFanoutResult:
    """Internal completion sidecar; never serialized into the public plan."""

    plan: NarrativeFanoutPlan
    status: str
    completed_cells: tuple[str, ...]
    missing_cells: tuple[str, ...]
    # Claim IDs are execution metadata, not a public AnswerClaim field.  They
    # let the composer bind each short synthesis sentence to its exact matrix
    # coordinate without reverse-parsing display labels.
    cell_claim_output_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class InvestmentRowSelector:
    """Request-local row operation for one proved investment table."""

    task_id: str
    issuer_name: str
    investment_name: str | None = None
    operation: str = "exact"
    criterion_year: int | None = None

    def __post_init__(self) -> None:
        allowed = {
            "exact", "start_year", "end_year_on_or_after",
            "argmax_amount", "missing_period",
        }
        if (not self.task_id or not self.issuer_name.strip()
                or self.operation not in allowed):
            raise ValueError("investment row selector가 잘못되었습니다")
        if self.operation == "exact":
            if not isinstance(self.investment_name, str) \
                    or not self.investment_name.strip() \
                    or self.criterion_year is not None:
                raise ValueError("exact investment row selector가 잘못되었습니다")
        elif self.investment_name is not None:
            raise ValueError("operator investment row selector에 행 이름이 있습니다")
        if self.operation in {"start_year", "end_year_on_or_after"}:
            if (not isinstance(self.criterion_year, int)
                    or not 1900 <= self.criterion_year <= 2200):
                raise ValueError("investment year selector가 잘못되었습니다")
        elif self.operation != "exact" and self.criterion_year is not None:
            raise ValueError("investment selector에 불필요한 연도가 있습니다")


@dataclass(frozen=True)
class InvestmentEmptySelection:
    """A proved complete table contains no row matching one closed filter."""

    source_table: str
    operation: str


@dataclass(frozen=True)
class InvestmentAggregationRequest:
    """A literal, task-bound request to sum the selected investment rows."""

    task_id: str
    issuer_name: str
    aggregate_surface: str
    expected_row_count: int | None = None
    revenue_year: int | None = None
    revenue_scope: str | None = None
    revenue_surface: str | None = None


@dataclass(frozen=True)
class InvestmentTableAggregate:
    """Exact deterministic sum of one structurally proved amount column."""

    row_count: int
    value: Decimal
    unit: str
    value_won: Decimal


_INVESTMENT_UNIT_SCALE = {
    "원": Decimal(1),
    "천원": Decimal(1_000),
    "백만원": Decimal(1_000_000),
    "억원": Decimal(100_000_000),
}
_INVESTMENT_TOTAL_MARKERS = frozenset({"소계", "합계", "총계"})


def _adjacent_investment_table_unit(
        lines: list[str], table_start: int,
        ) -> str | None:
    """Read a unit only from the structural block adjacent to the table."""

    before = table_start - 1
    blank_gap = 0
    while before >= 0 and not lines[before].strip() and blank_gap < 2:
        blank_gap += 1
        before -= 1
    if before < 0 or blank_gap > 1:
        return None
    carrier = lines[before]
    if carrier.lstrip().startswith("|"):
        cells = _table_cells(carrier)
        if not _is_table_rule(cells):
            return None
        before -= 1
        if before < 0 or not lines[before].lstrip().startswith("|"):
            return None
        carrier = lines[before]
    unit_label = re.search(r"단위\s*[:：]?", carrier)
    if unit_label is None:
        return None
    matches = set(re.findall(
        r"백만원|천원|억원|원", carrier[unit_label.end():]))
    return next(iter(matches)) if len(matches) == 1 else None


def _investment_total_row_disposition(
        cells: list[str], amount_pos: int,
        ) -> str:
    """Return ``skip``, ``data``, or ``reject`` for one possible total row."""

    non_amount = [
        _compact_heading(cell) for index, cell in enumerate(cells)
        if index != amount_pos
    ]
    total_like = [
        value for value in non_amount
        if any(marker in value for marker in _INVESTMENT_TOTAL_MARKERS)
    ]
    if not total_like:
        return "data"
    substantive = [
        value for value in non_amount
        if value not in {"", "-"} and value not in _INVESTMENT_TOTAL_MARKERS
    ]
    if (len(total_like) == 1
            and total_like[0] in _INVESTMENT_TOTAL_MARKERS
            and not substantive):
        return "skip"
    return "reject"


def _aggregate_investment_table(
        text_prompt_safe: str, request: InvestmentAggregationRequest,
        ) -> InvestmentTableAggregate | None:
    """Sum every selected data row, or return ``None`` on any uncertainty.

    The function accepts exactly one four-role investment table, one unique
    amount column, one explicit/common monetary unit, and numeric values in
    every non-total data row.  A source total is never added as another row.
    An asserted row count is an integrity condition, not a hint.
    """

    body = _strip_prompt_boundaries(text_prompt_safe)
    lines = body.splitlines()
    tables: list[tuple[int, int, list[str], dict[str, int]]] = []
    for index in range(len(lines) - 1):
        if (not lines[index].lstrip().startswith("|")
                or not lines[index + 1].lstrip().startswith("|")
                or not _is_table_rule(_table_cells(lines[index + 1]))):
            continue
        header = _table_cells(lines[index])
        positions = _investment_header_positions(header)
        if not positions:
            continue
        end = index + 2
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            end += 1
        tables.append((index, end, header, positions))
    if len(tables) != 1:
        return None

    start, end, header, positions = tables[0]
    table_unit = _adjacent_investment_table_unit(lines, start)
    amount_pos = positions["금액"]
    values_won: list[Decimal] = []
    display_values: list[tuple[Decimal, str]] = []
    for line in lines[start + 2:end]:
        cells = _table_cells(line)
        if len(cells) != len(header):
            return None
        disposition = _investment_total_row_disposition(cells, amount_pos)
        if disposition == "skip":
            continue
        if disposition == "reject":
            return None
        raw = cells[amount_pos].strip()
        numeric = re.fullmatch(
            r"(?P<value>[0-9][0-9,]*(?:\.[0-9]+)?)\s*"
            r"(?P<unit>백만원|천원|억원|원)?",
            raw,
        )
        if numeric is None:
            return None
        unit = numeric.group("unit") or table_unit
        if unit not in _INVESTMENT_UNIT_SCALE:
            return None
        value = Decimal(numeric.group("value").replace(",", ""))
        values_won.append(value * _INVESTMENT_UNIT_SCALE[unit])
        display_values.append((value, unit))
    if (not values_won
            or (request.expected_row_count is not None
                and len(values_won) != request.expected_row_count)):
        return None

    common_units = {unit for _value, unit in display_values}
    if len(common_units) == 1:
        display_unit = next(iter(common_units))
        display_value = sum(
            (value for value, _unit in display_values), Decimal(0))
    else:
        display_unit = "원"
        display_value = sum(values_won, Decimal(0))
    return InvestmentTableAggregate(
        row_count=len(values_won), value=display_value,
        unit=display_unit, value_won=sum(values_won, Decimal(0)),
    )


def _format_investment_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    integer, dot, fraction = rendered.partition(".")
    grouped = f"{int(integer):,}"
    fraction = fraction.rstrip("0")
    return f"{grouped}.{fraction}" if dot and fraction else grouped


def _disclosed_metric_match_keys(retrieval_query: str) -> tuple[str, ...]:
    """`retrieval_query` 를 답 근거와 대조할 매칭 키(정본 표기 + 괄호 동의어 하나).

    기본은 질문 원문 그대로의 표기 하나뿐이다. 다만 「순이자마진(NIM)」처럼
    질문이 정본 표기와 괄호 동의어를 함께 썼다가 `DisclosedMetricTopicRegrounder`
    가 정본 표기(「순이자마진」)로 좁힌 경우, 사전(`disclosed_metric_topics.tsv`)
    에 그 괄호 동의어가 「term(동의어)」 꼴 별칭으로 있으면 그 동의어(「NIM」)도
    함께 키로 쓴다 — 그룹 NIM 표처럼 본문 표에 「NIM(신한은행+신한카드)」만 있고
    「순이자마진」이 문자 그대로 없는 표도 근거로 인정한다.

    「임상」처럼 표기 자체가 넓은 사전 term 의 다른 별칭(「임상 1상」 등)까지는
    넓히지 않는다 — 이슈 #113 이 좁힌 그대로 유지한다. 그래서 넓히는 대상은
    질문이 가리킨 term 과 정확히 같은 표기를 쓸 때, 그 term 소유의 괄호 동의어
    하나뿐이다.
    """

    key = _compact_heading(retrieval_query)
    if not key:
        return ()
    entries = disclosed_metrics_for_query(retrieval_query)
    if len(entries) != 1 or _compact_heading(entries[0].term) != key:
        return (key,)
    entry = entries[0]
    paren_alias = next(
        (alias for alias in entry.aliases
         if alias.startswith(f"{entry.term}(") and alias.endswith(")")),
        None)
    if paren_alias is None:
        return (key,)
    synonym_key = _compact_heading(paren_alias[len(entry.term) + 1:-1])
    return (key, synonym_key) if synonym_key and synonym_key != key else (key,)


def _relevance_filtered_disclosed_metric_hits(
        hits: tuple[object, ...], retrieval_query: str,
        ) -> tuple[object, ...]:
    """이슈 #113 — 절 제목 하나로 끌어온 hit 을 질문 어휘로 다시 좁힌다.

    `_disclosed_metric_document_hits` 는 절 제목(``section_hint``, 「II. 사업의
    내용」)과 사전 표기의 넓은 동의어 하나하나로 hit 을 모은다. 「임상」처럼 표기
    자체가 흔한 낱말(「임상개발본부」 조직명·인원표 항목명에도 섞여 있다)이면,
    질문과 무관한 조직도·인원표·라이선스 "인"(들여온) 계약까지 같은 절 안이라는
    이유로 함께 실린다.

    `retrieval_query` 는 사전이 인정한, **질문 원문에 그대로 있는** 표기다(예:
    「임상 3상」) — 사전의 넓은 term(「임상」)보다 좁다. 그 표기(와 그 표기의
    괄호 동의어 — `_disclosed_metric_match_keys`)가 문자 그대로 없는 chunk 는
    답의 근거로 쓰지 않는다. 걸러서 0건이 되면 안전망으로 원래 hits 를 그대로
    돌려준다 — 이 필터가 답 전체를 지우는 일은 없어야 한다(예: 질문 표기가
    별칭뿐이고 본문은 다른 동의어로만 그 개념을 담은 경우).
    """

    keys = _disclosed_metric_match_keys(retrieval_query)
    if not keys:
        return hits
    filtered = tuple(
        h for h in hits
        if any(key_present(_compact_heading(getattr(h, "text_prompt_safe", "") or ""), key)
               for key in keys))
    return filtered or hits


class NarrativeTool:
    def __init__(self, rm, index):
        self.rm = rm
        self.index = index
        # Exact annual-report comparisons commonly ask two or more typed
        # topics from the same receipt.  Re-scanning that document's canonical
        # chunks for every topic dominated the 16-cell path.  The reader is
        # immutable for one serving process, so retain only a small bounded
        # LRU of document-local safe ChunkRows; every selected row is still
        # independently round-tripped through ``read_section`` below.
        self._exact_document_chunks: OrderedDict[
            tuple[str, str | None, str | None], tuple[object, ...]
        ] = OrderedDict()
        self._exact_document_chunks_limit = 8

    def _disclosed_metric_document_hits(
            self, entries, sub_queries, *, topic: str, rcept_no: str, as_of: str,
            corp_codes: tuple[str, ...], doc_groups: tuple[str, ...],
            date_from: str | None) -> tuple[object, ...]:
        """공시 서술지표를 **한 접수본 안에서** 찾는다.

        1차: 사전의 ``section_hint``(「II. 사업의 내용」) 경로 아래에서 동의어를
        하나씩 검색. 2차: 비면 그 문서 전체. 어느 쪽도 구조 권위(business_axis 등)를
        걸지 않는다 — 순이자마진처럼 「IV. 이사의 경영진단」에만 실리는 지표가 있다.
        결과는 rank 순 상위 ``TOP_K`` 로 자른다.
        """

        # 정본 표기(또는 부문별 매출의 부문명)가 있는 chunk 가 0순위, 별칭만 있으면 1순위.
        term_keys = tuple(dict.fromkeys(
            probe_keys(entry, topic)[0] for entry in entries if probe_keys(entry, topic)))
        alias_keys = tuple(key for key in (
            _compact_heading(surface) for entry in entries for surface in entry.surfaces) if key)
        # 「이사회 운영과 주주총회 투표제도」처럼 이음말로 묶인 지표들이 서로
        # 다른 절에 있을 수 있다(#122 CG-046 실측: 「1. 이사회에 관한 사항」·
        # 「3. 주주총회 등에 관한 사항」). entries[0] 하나의 section_hint 만
        # 쓰면 그 절에서 먼저 hit 이 나는 순간 나머지 절은 문서 전체로 넓히는
        # 안전망(아래)조차 타지 않는다 — 서로 다른 section_hint 를 모두 모은다.
        section_hints = tuple(dict.fromkeys(entry.section_hint for entry in entries))

        def _bearing_rank(row) -> int:
            # 0: 정본 표기 자체가 본문에 있다  1: 별칭만  2: 검색 토큰만(제외 대상)
            compact = _compact_heading(getattr(row, "text_prompt_safe", "") or "")
            if any(key and key_present(compact, key) for key in term_keys):
                return 0
            if any(key_present(compact, key) for key in alias_keys):
                return 1
            return 2

        def _collect(path_prefix: str | None) -> tuple[object, ...]:
            merged: dict[str, object] = {}
            for sq in sub_queries:
                for h in self.index.search(
                        sq, as_of=as_of, corp_codes=corp_codes,
                        doc_groups=doc_groups, date_from=date_from,
                        path_prefix=path_prefix, top_k=MAX_SEARCH_TOP_K):
                    if _rcept(h.doc_id) != rcept_no:
                        continue
                    if h.chunk_id not in merged or h.rank < merged[h.chunk_id].rank:
                        merged[h.chunk_id] = h
            rows = tuple(merged.values())
            # FTS 는 동의어 토큰(NPL·BIS·연구개발활동…)이 다른 문맥에 쓰인 chunk 도
            # 돌려준다. 사전 표기 자체가 본문에 있는 chunk 를 **자르기 전에** 먼저
            # 남기고 정본 표기 > 별칭 > rank 순으로 세운다. 하나도 없으면 검색 결과를
            # 그대로 두어 not_found 대신 근거 검토 기회를 남긴다.
            bearing = tuple(row for row in rows if _bearing_rank(row) < 2)
            ordered = sorted(bearing or rows,
                             key=lambda h: (_bearing_rank(h), h.rank, h.rcept_dt))
            return tuple(ordered)[:TOP_K]

        def _collect_sections(path_prefixes: tuple[str, ...]) -> tuple[object, ...]:
            # 절이 둘 이상이면 한 절(대개 더 자주 쓰이는 낱말이 있는 쪽,
            # 「이사회 운영」)의 정본 표기 hit 이 흔해 전역 상위 TOP_K 를 혼자
            # 채울 수 있다(#122 CG-046 실측 — 「주주총회 투표제도」쪽 hit 이
            # 밀려남). 절마다 최소 예산을 먼저 배정해 각 절이 대표를 남기게
            # 한다.
            per_section_budget = (
                TOP_K if len(path_prefixes) <= 1
                else max(1, TOP_K // len(path_prefixes)))
            merged: dict[str, object] = {}
            for path_prefix in path_prefixes:
                for h in _collect(path_prefix)[:per_section_budget]:
                    if h.chunk_id not in merged or h.rank < merged[h.chunk_id].rank:
                        merged[h.chunk_id] = h
            ordered = sorted(merged.values(),
                             key=lambda h: (_bearing_rank(h), h.rank, h.rcept_dt))
            return tuple(ordered)[:TOP_K]

        hits = _collect_sections(section_hints)
        if not hits or all(_bearing_rank(h) > 0 for h in hits):
            # 「II. 사업의 내용」에 표기 자체가 없고 별칭만 스친 경우(한화오션 신규수주는
            # 재무 주석 표에만 있다) 문서 전체에서 다시 찾아 정본 표기 chunk 를 앞세운다.
            whole = _collect(None)
            merged = {getattr(h, "chunk_id", id(h)): h for h in (*whole, *hits)}
            hits = tuple(sorted(merged.values(),
                                key=lambda h: (_bearing_rank(h), h.rank, h.rcept_dt)))[:TOP_K]
        return hits

    @staticmethod
    def _is_latest_whole_business_snapshot(task, fanout: NarrativeFanoutPlan) -> bool:
        """Whether a broad one-company business overview has no time coordinate.

        ``사업 내용`` names a whole disclosure section.  In that narrow shape,
        mixing a historical business-segment paragraph with a newer products
        paragraph produces an answer that has no single "current" date.  Bind
        both typed cells to one latest annual report instead.  Any user-given
        period/document, comparison, or more specific request keeps its
        original selection path.
        """

        cells = fanout.cells
        return (
            getattr(task, "operation", None) == "search"
            and len(tuple(getattr(task, "corp_codes", ()) or ())) == 1
            and not tuple(getattr(task, "periods", ()) or ())
            and getattr(task, "document_selector", None) is None
            and not tuple(getattr(task, "requested_slots", ()) or ())
            and _compact_heading(getattr(task, "retrieval_query", ""))
            in {"사업내용", "사업의내용"}
            and len(cells) == 2
            and tuple(cell.topic for cell in cells)
            == _compound_business_topics(getattr(task, "retrieval_query", ""))
            and all(cell.document_selector is None for cell in cells)
        )

    def _latest_annual_document_selector(self, *, corp_code: str, as_of: str):
        """Return the latest available annual filing identity for one issuer."""

        from agent.contracts import DocumentSelector

        documents = tuple(self.rm.documents(
            as_of=as_of, corp_code=corp_code, doc_group="periodic", form="annual"))
        if not documents:
            return None
        latest = max(documents, key=lambda row: (
            row.base_year if row.base_year is not None else -1,
            row.rcept_dt, row.rcept_no, row.doc_id))
        return DocumentSelector(
            doc_id=latest.doc_id, rcept_no=latest.rcept_no,
            doc_group="periodic", form="annual")

    def _bind_latest_whole_business_snapshot(
            self, task, fanout: NarrativeFanoutPlan, *, trace: list[TraceEvent],
            ) -> NarrativeFanoutPlan:
        if not self._is_latest_whole_business_snapshot(task, fanout):
            return fanout
        selector = self._latest_annual_document_selector(
            corp_code=fanout.cells[0].corp_code, as_of=task.as_of)
        if selector is None:
            # Do not turn an otherwise searchable request into a failure when
            # a partial corpus has no annual filing for this issuer.
            return fanout
        bound = replace(
            fanout,
            cells=tuple(replace(cell, document_selector=selector)
                        for cell in fanout.cells),
        )
        trace.append(TraceEvent(
            seq=len(trace) + 1, stage="tool",
            summary=("whole business overview bound to latest annual filing "
                     f"{selector.rcept_no}"),
            detail={"receipt_no": selector.rcept_no,
                    "topics": [cell.topic for cell in bound.cells]},
        ))
        return bound

    def _cached_exact_document_chunks(self, task) -> tuple[object, ...] | None:
        selector = getattr(task, "document_selector", None)
        receipt = getattr(selector, "rcept_no", None)
        chunks = getattr(self.rm, "chunks", None)
        if not receipt or not callable(chunks):
            return None
        doc_id = getattr(selector, "doc_id", None) or f"periodic_{receipt}"
        corp_codes = tuple(getattr(task, "corp_codes", ()) or ())
        corp_code = corp_codes[0] if len(corp_codes) == 1 else None
        doc_group = getattr(selector, "doc_group", None)
        key = (doc_id, corp_code, doc_group)
        cached = self._exact_document_chunks.get(key)
        if cached is not None:
            self._exact_document_chunks.move_to_end(key)
            return cached
        try:
            rows = tuple(chunks(
                corp_code=corp_code, doc_group=doc_group,
                doc_id=doc_id, projection="llm"))
        except (TypeError, ValueError):
            return None
        self._exact_document_chunks[key] = rows
        self._exact_document_chunks.move_to_end(key)
        while len(self._exact_document_chunks) > self._exact_document_chunks_limit:
            self._exact_document_chunks.popitem(last=False)
        return rows

    def run_task(
            self, task, *, trace: list[TraceEvent], corp_name: str,
            sidecar: NarrativeFanoutPlan | None = None,
            row_selector: InvestmentRowSelector | None = None,
            aggregation_request: InvestmentAggregationRequest | None = None,
            prefer_latest: bool = False,
            ):
        """Execute a public narrative task as independently sourced cells.

        The returned fourth value is an internal completion sidecar.  Existing
        callers may continue to ignore it; no public QueryPlan field changes.
        """
        fanout = sidecar or NarrativeFanoutPlan.from_task(task, corp_name=corp_name)
        fanout = self._bind_latest_whole_business_snapshot(task, fanout, trace=trace)
        if fanout.task_id != task.task_id:
            raise ValueError("narrative fanout sidecar task_id가 일치하지 않습니다")
        if row_selector is not None and row_selector.task_id != task.task_id:
            raise ValueError("investment row selector task_id가 일치하지 않습니다")
        if (aggregation_request is not None
                and aggregation_request.task_id != task.task_id):
            raise ValueError("investment aggregation task_id가 일치하지 않습니다")

        matrix = len(fanout.cells) > 1 or fanout.omitted_cells > 0
        claims, lims, used = [], [], []
        missing_cells: list[str] = []
        completed_cells: list[str] = []
        cell_claim_output_ids: list[tuple[str, tuple[str, ...]]] = []
        if fanout.omitted_cells:
            lims.append(Limitation(
                code="narrative_fanout_limit",
                detail=(f"narrative cell {fanout.omitted_cells}개는 실행 상한 "
                        f"{MAX_NARRATIVE_FANOUT_CELLS}을 넘어 조회하지 않았습니다."),
            ))
        if not fanout.cells:
            lims.append(Limitation(code="not_found", detail="실행할 narrative cell이 없습니다."))
            return claims, lims, used, NarrativeFanoutResult(
                plan=fanout, status="terminal", completed_cells=(), missing_cells=())

        # Keep the legacy one-cell cap exactly.  A matrix receives one bounded
        # total result budget so a broad question cannot multiply memory by the
        # number of companies/periods/topics.
        per_cell_blocks = (MAX_BLOCKS_PER_PERIOD if not matrix else max(
            1, MAX_NARRATIVE_RESULT_BLOCKS // len(fanout.cells)))
        for index, cell in enumerate(fanout.cells, start=1):
            cell_task = SimpleNamespace(
                task_id=(task.task_id if not matrix else f"{task.task_id}.{cell.cell_id}"),
                corp_codes=[cell.corp_code] if cell.corp_code else [],
                # Retrieval is scoped by corp_code, but a consolidated report
                # can still contain long sibling-company subsections.  Keep
                # the selected issuer as execution metadata so structural
                # ranking can prefer an issuer-bound overview over one
                # affiliate that happens to repeat the requested product word.
                expected_corp_name=cell.corp_name,
                document_selector=cell.document_selector,
                periods=[cell.period] if cell.period is not None else [],
                as_of=task.as_of,
                retrieval_query=cell.topic,
                operation=task.operation,
                # In matrix mode slots must be checked across all cells, not
                # repeated as a false per-cell omission.
                requested_slots=[] if matrix else task.requested_slots,
            )
            cell_claims, cell_lims, cell_used, _ = self._run_one_task(
                cell_task, trace=trace, corp_name=cell.corp_name,
                max_blocks=per_cell_blocks, prefer_latest=prefer_latest,
                row_selector=(row_selector if not matrix else None),
                aggregation_request=(
                    aggregation_request if not matrix else None))
            if cell_claims:
                completed_cells.append(cell.cell_id)
                cell_claim_output_ids.append((
                    cell.cell_id, tuple(claim.output_id for claim in cell_claims)))
                claims.extend(cell_claims)
                used.extend(cell_used)
                lims.extend(cell_lims)
                if matrix and len(cell_claims) >= per_cell_blocks:
                    lims.append(Limitation(
                        code="narrative_result_budget_exhausted",
                        detail=(f"{cell.corp_name}: {cell.cell_id}의 구조 블록은 "
                                f"matrix 결과 예산 {per_cell_blocks}개까지만 포함했습니다."),
                        affected_doc_ids=sorted(set(cell_used)),
                    ))
                continue

            missing_cells.append(cell.cell_id)
            if matrix:
                # Per-cell absence is explicit, but is not a corpus-wide
                # absence while another independently cited cell succeeded.
                lims.extend(lim for lim in cell_lims if lim.code != "not_found")
                period_label = (
                    f" [{cell.period.start}~{cell.period.end}]"
                    if cell.period is not None else "")
                lims.append(Limitation(
                    code="narrative_cell_not_found",
                    detail=(f"{cell.corp_name}: '{cell.topic}' 관련 본문을 "
                            f"{cell.cell_id} 좌표에서 찾지 못했습니다{period_label}"),
                ))
            else:
                lims.extend(cell_lims)

        topic_keys = {_compact_heading(cell.topic) for cell in fanout.cells}
        requested_keys = {
            _compact_heading(slot) for slot in task.requested_slots}
        topic_bound_fields = bool(requested_keys) and requested_keys == topic_keys
        if (matrix and task.requested_slots and claims
                and not topic_bound_fields):
            body = " ".join(claim.text or "" for claim in claims)
            missing_slots = [slot for slot in task.requested_slots if not _slot_present(slot, body)]
            if missing_slots:
                lims.append(Limitation(
                    code="slot_not_confirmed",
                    # `matrix` 는 실행 좌표 이름이지 읽는 사람이 풀어야 할
                    # 말이 아니다. 이 detail 은 답변 문장으로 그대로 나간다.
                    detail=("확인한 공시 본문에는 다음 항목이 나오지 않습니다: "
                            f"{', '.join(missing_slots)}. "
                            "추측하지 않고 확인되지 않음으로 남겼습니다"),
                    affected_doc_ids=sorted(set(used)),
                ))
        if matrix and not claims:
            # This is intentionally in addition to cell-level diagnostics so
            # Orchestrator returns not_found rather than exposing a guessed
            # matrix answer.
            lims.append(Limitation(
                code="not_found",
                detail="모든 narrative matrix cell에서 검증 가능한 본문을 찾지 못했습니다.",
            ))
        lims = _dedupe_limitations(lims)
        partial = (missing_cells or fanout.omitted_cells or any(
            limitation.code in {
                "narrative_record_budget_exhausted",
                "narrative_result_budget_exhausted",
                "slot_not_confirmed",
            }
            for limitation in lims))
        return claims, lims, used, NarrativeFanoutResult(
            plan=fanout,
            status=("partial_ready" if partial else "ready")
            if claims else "terminal",
            completed_cells=tuple(completed_cells), missing_cells=tuple(missing_cells),
            cell_claim_output_ids=tuple(cell_claim_output_ids),
        )

    def _run_one_task(
            self, task, *, trace: list[TraceEvent], corp_name: str,
            max_blocks: int = MAX_BLOCKS_PER_PERIOD,
            prefer_latest: bool = False,
            row_selector: InvestmentRowSelector | None = None,
            aggregation_request: InvestmentAggregationRequest | None = None,
            ):
        claims, lims, used = [], [], []
        proved_empty_selection = False
        corp_codes = tuple(task.corp_codes)
        ds = task.document_selector
        doc_groups = (ds.doc_group,) if ds and ds.doc_group else ()
        date_from = None
        authority_kind = _narrative_authority_kind(task)
        if aggregation_request is not None and authority_kind != "slot_table":
            raise ValueError("investment aggregation은 slot table에만 적용됩니다")
        if authority_kind in {"business_axis", "investment_plan"}:
            # Stage1 splits business questions into typed topic cells before
            # composition.  Keep one whole, highest-authority structural
            # block per cell: emitting the legacy eight blocks here lets a
            # broad overview drown out the requested axis in the matrix.
            # This is an execution-only budget; no public QueryPlan field or
            # long-narrative guard is changed.
            max_blocks = min(max_blocks, 1)
        # periods(compare): 각 기간별로 검색 (기간 종료일을 as_of 상한처럼 쓰지 않는다 — 문서 접수일 필터는 as_of)
        period_windows = []
        if task.periods:
            for pr in task.periods:
                st = pr.start.strftime("%Y%m%d") if pr.start else None
                # 사업보고서는 회계연도 종료 후 접수되므로 접수일 창을 [기간말, 기간말+~15개월]로
                en = pr.end.strftime("%Y%m%d")
                period_windows.append((pr, st, en))
        queries = [(None, task.as_of, None)] if not period_windows else \
                  [(pr, min(task.as_of, _plus_months(en, 15)), en) for pr, st, en in period_windows]

        # compare/summarize처럼 여러 축을 한 질의에 담은 경우, 축(2어절 이상 명사구)별로 나눠 검색해 recall 확보.
        # 검색어를 새로 만들지 않는다 — handoff의 retrieval_query를 결정적으로 분할할 뿐.
        sub_queries = [task.retrieval_query]
        if task.operation in ("compare", "summarize"):
            words = task.retrieval_query.split()
            if len(words) >= 4:
                sub_queries += [" ".join(words[i:i+2]) for i in range(0, len(words) - 1, 2)]
        if (authority_kind == "business_axis"
                and _business_axis_kind(task.retrieval_query) == "revenue"):
            sub_queries += ["매출 및 수주 상황", "영업의 현황", "주요 제품 매출"]
        # 공시 서술지표는 동의어(NIM/순이자마진, 수주잔고/수주잔량 …)를 **한 개씩**
        # 따로 검색한다 — FTS 는 토큰을 AND 로 묶어 한 문자열에 넣으면 recall 이 죽는다.
        disclosed_metrics = disclosed_metrics_for_query(task.retrieval_query)
        for entry in disclosed_metrics:
            sub_queries += list(search_terms_for(entry, task.retrieval_query))
        sub_queries = list(dict.fromkeys(sub_queries))
        for pr, as_of, date_from in queries:
            merged: dict[str, object] = {}
            for sq in sub_queries:
                for h in self.index.search(sq, as_of=as_of, corp_codes=corp_codes,
                                           doc_groups=doc_groups, date_from=date_from, top_k=TOP_K):
                    # 한 section 안에서도 서로 다른 표 행/문단 chunk가 필요한 경우가 있다.
                    # section_id로 합치면 첫 chunk 외의 요청 항목이 조용히 사라진다.
                    if h.chunk_id not in merged or h.rank < merged[h.chunk_id].rank:
                        merged[h.chunk_id] = h
            hits = tuple(sorted(merged.values(), key=lambda h: (h.rank, h.rcept_dt)))
            # Broad narrative retrieval still has typed structural authority.
            # If lexical top-10 contains only look-alike uses of a term, widen
            # discovery and then enforce the source structure; never accept a
            # lower-authority hit merely because it ranked highly in FTS.
            if authority_kind is not None and not (ds and ds.rcept_no):
                hits = tuple(h for h in hits if _authority_hit_matches(
                    task, h, authority_kind))
                if (not hits
                        or (prefer_latest
                            and authority_kind == "investment_plan")):
                    wide = self.index.search(
                        task.retrieval_query, as_of=as_of,
                        corp_codes=corp_codes, doc_groups=doc_groups,
                        date_from=date_from, top_k=MAX_SEARCH_TOP_K)
                    hits = tuple(h for h in wide if _authority_hit_matches(
                        task, h, authority_kind))
                hits = _prioritize_authority_hits(
                    task, hits, authority_kind,
                    prefer_latest=prefer_latest)
            # 특정 문서 지정 시 그 문서로 한정 (검색 유니버스 밖 문서는 노출하지 않음)
            if ds and ds.rcept_no and disclosed_metrics:
                # 공시 서술지표: 그 접수본 안에서 「II. 사업의 내용」 경로를 먼저
                # 찾고, 없을 때만 문서 전체로 넓힌다. 구조 권위는 걸지 않는다.
                hits = self._disclosed_metric_document_hits(
                    disclosed_metrics, sub_queries, topic=task.retrieval_query,
                    rcept_no=ds.rcept_no,
                    as_of=as_of, corp_codes=corp_codes, doc_groups=doc_groups,
                    date_from=date_from)
                # 이슈 #113 — 절 제목(「II. 사업의 내용」 아래 「6. 주요계약 및
                # 연구개발활동」)만으로 hit 을 추렸으므로 조직도·인원표·라이선스
                # "인"(들여온) 계약처럼 절 안의 무관한 하위 블록까지 같이 실린다.
                # 질문이 실제로 가리킨 표기(``retrieval_query``, 사전이 인정한
                # 질문 어휘 그대로)가 chunk 본문에 없으면 그 chunk 는 답의 근거로
                # 쓰지 않는다.
                hits = _relevance_filtered_disclosed_metric_hits(
                    hits, task.retrieval_query)
            elif ds and ds.rcept_no:
                hits = tuple(h for h in hits if _rcept(h.doc_id) == ds.rcept_no)
                if authority_kind is not None:
                    hits = tuple(h for h in hits if _authority_hit_matches(
                        task, h, authority_kind))
                if not hits or authority_kind is not None:
                    # Exact-document authority is stronger than global FTS
                    # rank.  Widen first, then enforce the proved heading or
                    # slot-table boundary inside that one document.
                    wide = self.index.search(
                        task.retrieval_query, as_of=as_of,
                        corp_codes=corp_codes, doc_groups=doc_groups,
                        date_from=date_from, top_k=MAX_SEARCH_TOP_K)
                    hits = tuple(
                        h for h in wide
                        if _rcept(h.doc_id) == ds.rcept_no
                        and (authority_kind is None or _authority_hit_matches(
                            task, h, authority_kind)))[:TOP_K]
                if authority_kind is not None:
                    hits = _prioritize_authority_hits(
                        task, hits, authority_kind,
                        prefer_latest=prefer_latest)
                    structural_hits = _exact_document_authority_hits(
                        self.rm, task, authority_kind,
                        rows=self._cached_exact_document_chunks(task))
                    if structural_hits:
                        hits = structural_hits
            if ds and ds.form:
                hits = tuple(h for h in hits if _form_of(self.rm, h.doc_id, ds.form))
            if pr is not None and ds and ds.form and "사업보고서" in ds.form:
                # A fiscal-period comparison must bind to that fiscal year's
                # annual report, not merely any annual report filed within the
                # 15-month discovery window.
                self.rm._load_docs()
                hits = tuple(
                    h for h in hits
                    if (self.rm._documents.get(h.doc_id) is not None
                        and self.rm._documents[h.doc_id].base_year == pr.end.year
                        and self.rm._documents[h.doc_id].base_month == pr.end.month))
            hits = _same_document_affiliate_detail_hits(self.rm, task, tuple(hits))
            label_p = f" [{pr.start}~{pr.end}]" if pr else ""
            trace.append(TraceEvent(seq=len(trace)+1, stage="tool",
                                    summary=f"search_chunks {task.retrieval_query!r} as_of={as_of} corp={corp_codes} → {len(hits)} hits{label_p}",
                                    detail={"top_docs": sorted({_rcept(h.doc_id) for h in hits if _rcept(h.doc_id)})[:8]}))
            if not hits:
                lims.append(Limitation(code="not_found", detail=f"{corp_name}: '{task.retrieval_query}' 관련 본문을 지원 범위에서 찾지 못함{label_p}"))
                continue
            # Section 왕복 — 성공한 section에 속하는 검색 chunk만 근거로 사용한다.
            # 전체 section을 claim으로 넘기지 않는다. 검색된 구조 chunk만 행/문단 경계로
            # 재분할하므로 최종 답변이 임의 문자 위치에서 잘리지 않는다.
            verified_sections: dict[str, object | None] = {}
            seen_chunks = set()
            period_claim_count = 0
            omitted_over_budget = 0
            roundtrip_mismatches = 0
            for h in hits:
                if h.chunk_id in seen_chunks:
                    continue
                seen_chunks.add(h.chunk_id)
                if h.section_id not in verified_sections:
                    try:
                        verified_sections[h.section_id] = self.rm.read_section(section_id=h.section_id)
                    except Exception as e:
                        verified_sections[h.section_id] = None
                        trace.append(TraceEvent(seq=len(trace)+1, stage="evidence", summary=f"read_section 실패 {h.section_id[:10]}: {type(e).__name__}", detail={}))
                sec = verified_sections[h.section_id]
                if sec is None:
                    continue
                trace.append(TraceEvent(seq=len(trace)+1, stage="evidence",
                                        summary=f"read_section {h.section_id[:10]}… ok ({sec.n_chars}자, {sec.path[:40]})",
                                        detail={"doc_id": sec.doc_id, "rank": h.rank}))
                # SearchHit is discovery metadata only.  The final claim must
                # be cut from the independently read prompt-safe section after
                # proving that the hit chunk belongs to that exact section.
                # Otherwise a stale/corrupted index row could be displayed as
                # a source_roundtrip claim merely because another section read
                # happened to succeed.
                roundtripped_chunk = _roundtripped_chunk_text(h, sec)
                if roundtripped_chunk is None:
                    roundtrip_mismatches += 1
                    trace.append(TraceEvent(
                        seq=len(trace)+1, stage="evidence",
                        summary=f"read_section 결속 불일치 {h.section_id[:10]}… — hit 제외",
                        detail={"doc_id": h.doc_id, "chunk_id": h.chunk_id},
                    ))
                    continue
                authority_text = _authority_roundtripped_text(
                    task, h, sec, roundtripped_chunk, authority_kind,
                    row_selector=row_selector, corp_name=corp_name,
                    aggregation_request=aggregation_request)
                if authority_text is None:
                    trace.append(TraceEvent(
                        seq=len(trace)+1, stage="evidence",
                        summary=(f"authority 경계 불일치 {h.section_id[:10]}… "
                                 "— hit 제외"),
                        detail={"doc_id": h.doc_id, "chunk_id": h.chunk_id,
                                "authority_kind": authority_kind},
                    ))
                    continue
                rc = _rcept(sec.doc_id)
                if isinstance(authority_text, InvestmentEmptySelection):
                    proved_empty_selection = True
                    period_claim_count += 1
                    claims.append(AnswerClaim(
                        output_id=(
                            f"{task.task_id}.block{len(claims) + 1}"
                            f"{('@' + pr.end.strftime('%Y')) if pr else ''}"),
                        label=f"{corp_name} {sec.path} [기간 미기재 항목 확인]",
                        text=_investment_empty_selection_text(
                            authority_text.source_table),
                        citations=[ClaimCitation(
                            doc_id=sec.doc_id, rcept_no=rc,
                            section_id=sec.section_id,
                            locator=h.locator or sec.locator,
                            excerpt_prompt_safe=_safe_excerpt(
                                authority_text.source_table,
                                prefer_terms=("기간",)),
                            verification="source_roundtrip",
                        )],
                    ))
                    if rc:
                        used.append(rc)
                    break
                if aggregation_request is not None:
                    aggregate = _aggregate_investment_table(
                        authority_text, aggregation_request)
                    if aggregate is None:
                        lims.append(Limitation(
                            code="investment_aggregation_unavailable",
                            detail=(
                                f"{corp_name}: 투자계획 표의 행 수·계획금액·"
                                "단위를 모두 확정할 수 없어 합계를 계산하지 "
                                "않았습니다."),
                            affected_doc_ids=[rc] if rc else [],
                        ))
                        continue
                    period_claim_count += 1
                    output_id = f"{task.task_id}.investment-sum"
                    claims.append(AnswerClaim(
                        output_id=output_id,
                        label=(
                            f"{corp_name} 투자계획 {aggregate.row_count}건 "
                            "계획금액 합계"),
                        value_text=_format_investment_decimal(aggregate.value),
                        raw_unit=aggregate.unit,
                        canonical_value=format(aggregate.value_won, "f"),
                        canonical_unit="원",
                        operator="sum",
                        citations=[ClaimCitation(
                            doc_id=sec.doc_id, rcept_no=rc,
                            section_id=sec.section_id,
                            locator=h.locator or sec.locator,
                            # Every selected row contributed to this sum.
                            # Keep the verified prompt-safe table and its unit;
                            # a short excerpt cannot substantiate the total.
                            excerpt_prompt_safe=authority_text,
                            verification="source_roundtrip",
                        )],
                    ))
                    if rc:
                        used.append(rc)
                    trace.append(TraceEvent(
                        seq=len(trace) + 1, stage="derivation",
                        summary=(
                            f"투자계획 선택 행 {aggregate.row_count}건 계획금액 "
                            "합계 계산"),
                        detail={"output_id": output_id,
                                "unit": aggregate.unit},
                    ))
                    break
                if authority_kind == "general_investment":
                    # The source boundary has already been verified below.
                    # Split the parent section at its sole investment heading
                    # so a broad status summary retains both surrounding
                    # narrative and the disclosed investment plan/status.
                    blocks = _general_investment_blocks(authority_text) or []
                elif authority_kind == "business_axis":
                    blocks = _ranked_business_blocks(authority_text, task)
                elif disclosed_metrics:
                    # 이슈 #113 — 공시 서술지표는 절 제목만으로 hit 을 고른다.
                    # 8,000자 예산 안에 절 전체(원가표·개별 품목표·조직도·
                    # 인원표)가 한 조각으로 담길 수 있으므로, 질문 표기가
                    # 실제로 있는 더 잘게 나뉜 조각만 남긴다.
                    blocks = _disclosed_metric_relevant_blocks(
                        authority_text, task.retrieval_query)
                else:
                    blocks = _narrative_blocks(authority_text)
                described_subject = (
                    _explicit_business_subject(authority_text, corp_name)
                    if authority_kind == "business_axis"
                    and _business_axis_kind(task.retrieval_query) == "topic" else None)
                if described_subject:
                    lims.append(Limitation(
                        code="narrative_subject_partial",
                        detail=(f"아래는 {corp_name} 공시에 실린 {described_subject}의 사업부문 내용입니다. "
                                f"{corp_name} 그룹 전체의 사업 현황이나 실적으로 일반화할 수 없습니다."),
                        affected_doc_ids=[rc] if rc else []))
                for block, over_budget in blocks:
                    if over_budget:
                        omitted_over_budget += 1
                        continue
                    if not block:
                        continue
                    period_claim_count += 1
                    authority_slots = _authority_bound_requested_slots(
                        task, authority_kind)
                    confirmed_slots = [
                        s for s in task.requested_slots
                        if s in authority_slots or _slot_present(s, block)]
                    slot_label = f" [확인 항목: {', '.join(confirmed_slots)}]" if confirmed_slots else ""
                    # 이슈 #94 26 — 공시 원문의 회계 기수를 실제 연도로 바꾼다.
                    # **여기서 한 번 바꾼다.** 이 블록이 표·문장·요약으로 갈려
                    # 나가므로, 하류마다 고치는 대신 원문이 들어오는 자리에서
                    # 한 번 바꾸면 전부가 같은 말을 하게 된다. 짝은 이 문서의
                    # 것이어야 한다 — 「제50기」는 회사마다 다른 해다.
                    public_block = years_for_period_labels(
                        block, _period_years(self.rm, sec.doc_id))
                    claims.append(AnswerClaim(
                        output_id=f"{task.task_id}.block{len(claims)+1}{('@'+pr.end.strftime('%Y')) if pr else ''}",
                        label=f"{corp_name} {sec.path}{slot_label}" + (f" ({pr.end.year} 보고서)" if pr else "")
                              + (f" [실제 설명 대상: {described_subject}]" if described_subject else ""),
                        text=public_block,
                        citations=[ClaimCitation(doc_id=sec.doc_id, rcept_no=rc,
                                                 section_id=sec.section_id,
                                                 locator=h.locator or sec.locator,
                                                 excerpt_prompt_safe=public_block if described_subject else _safe_excerpt(
                                                     public_block,
                                                     prefer_terms=_citation_excerpt_terms(task)),
                                                 verification="source_roundtrip")]))
                    if rc:
                        used.append(rc)
                    if period_claim_count >= max_blocks:
                        break
                if period_claim_count >= max_blocks:
                    break
            if omitted_over_budget:
                lims.append(Limitation(
                    code="narrative_record_budget_exhausted",
                    detail=(f"{corp_name}: 행 또는 문단 {omitted_over_budget}개가 "
                            f"{NARRATIVE_BLOCK_CHARS:,}자 구조 블록 예산을 초과해 중간 절단 없이 제외됨{label_p}"),
                    affected_doc_ids=sorted({_rcept(h.doc_id) for h in hits if _rcept(h.doc_id)}),
                ))
            if roundtrip_mismatches:
                lims.append(Limitation(
                    code="narrative_source_roundtrip_mismatch",
                    detail=(f"{corp_name}: 검색 chunk {roundtrip_mismatches}개가 읽어온 section과 "
                            f"동일성 검증에 실패해 답변에서 제외됨{label_p}"),
                    affected_doc_ids=sorted({_rcept(h.doc_id) for h in hits if _rcept(h.doc_id)}),
                ))
            if authority_kind is not None and period_claim_count == 0:
                lims.append(Limitation(
                    code="not_found",
                    detail=(f"{corp_name}: exact document에서 증명된 narrative "
                            f"authority 경계를 찾지 못함{label_p}"),
                    affected_doc_ids=sorted({
                        _rcept(h.doc_id) for h in hits if _rcept(h.doc_id)}),
                ))
        # requested_slots: 본문에서 확인되지 않는 slot을 typed로 남긴다 (추측 금지)
        if task.requested_slots and claims and not proved_empty_selection:
            body = " ".join(c.text or "" for c in claims)
            authority_slots = set(_authority_bound_requested_slots(
                task, authority_kind))
            handled_slots = set()
            if aggregation_request is not None:
                handled_slots.update(
                    slot for slot in task.requested_slots
                    if _investment_slot_roles((slot,)))
            missing = [
                s for s in task.requested_slots
                if (s not in authority_slots and s not in handled_slots
                    and not _slot_present(s, body))]
            if missing:
                lims.append(Limitation(code="slot_not_confirmed",
                                       detail=f"본문에서 확인되지 않은 항목: {', '.join(missing)} — 추측하지 않고 '공시에서 확인되지 않음'으로 표기",
                                       affected_doc_ids=sorted(set(used))))
        # 이중 소스(PDF/HTML) 문서면 교차검증 상태를 정본 그대로 고지 (partial_consistent 등)
        for rc in sorted(set(used)):
            xs, reason = self._cross_check(rc)
            if xs and xs not in ("single_source", "consistent"):
                lims.append(Limitation(code="source_cross_check_partial",
                                       detail=f"{rc}: PDF/HTML 이중 소스 교차검증 {xs} — {reason or '대응하지 않은 범위는 미검증'}",
                                       affected_doc_ids=[rc]))
        return claims, lims, used, None

    def _cross_check(self, rcept_no: str) -> tuple[str | None, str | None]:
        self.rm._load_docs()
        doc_id = self.rm._doc_id_by_rcept.get(rcept_no)
        meta = self.rm._documents.get(doc_id) if doc_id else None
        if meta is None:
            return None, None
        return (getattr(meta, "source_cross_check_status", None),
                getattr(meta, "source_cross_check_reason", None))


_SLOT_HINTS = {
    "투자대상": ("투자", "취득", "증설", "설비", "지분"), "목적": ("목적", "위해", "위한"),
    "금액": ("원", "억", "백만"), "기간": ("년", "월", "까지", "예정"), "계획실적구분": ("계획", "예정", "실적", "완료"),
    "기지출금액": ("기지출", "누적지출", "실제지출", "집행", "기투자"),
    "투자현황": ("투자현황", "투자계획", "기투자액", "향후투자액", "투자기대효과"),
    "주요사업": ("사업", "제품", "서비스"),
}


def _slot_present(slot: str, body: str) -> bool:
    compact_slot = _compact_heading(slot)
    hints = tuple(dict.fromkeys((
        slot,
        *_SLOT_HINTS.get(slot, ()),
        *_SLOT_HINTS.get(compact_slot, ()),
    )))
    # DART headers freely vary whitespace (``기 지출금액`` vs
    # ``기지출금액``).  Only compact the same bounded vocabulary.
    compact_body = _compact_heading(body)
    if any(hint in body or _compact_heading(hint) in compact_body
           for hint in hints if hint):
        return True
    # 공시 서술지표 사전이 아는 요청 필드는 그 사전 표기(별칭 포함)로도 본다.
    # 「계열회사 수」요청이 본문 표 헤더 「계열회사의 수」와 조사 하나만
    # 다르거나(#122 CG-045), 「주주총회 투표제도」요청이 본문에는 「전자
    # 투표제」로만 실리는 경우(#122 CG-046) — 사전이 이미 같은 개념으로
    # 인정한 표기인데 글자가 그대로 다르다는 이유로 slot_not_confirmed 를
    # 내면 방금 보여준 근거를 스스로 부정하는 문장이 된다.
    entry = disclosed_metric_for_query(slot)
    if entry is not None:
        return any(
            key_present(compact_body, _compact_heading(surface))
            for surface in entry.surfaces if surface)
    return False


def _compact_heading(value: str) -> str:
    value = re.sub(r"\s+", "", value or "")
    return re.sub(r"[^0-9A-Za-z가-힣]", "", value).casefold()


def _path_has_heading(path: str, heading: str) -> bool:
    expected = _compact_heading(heading)
    if not expected:
        return False
    for part in (path or "").split(">"):
        clean = re.sub(
            r"^\s*(?:[0-9]+|[IVXLCDM]+|[A-Za-z가-힣])\s*[.)]\s*",
            "", part, flags=re.IGNORECASE)
        if _compact_heading(clean) == expected:
            return True
    return False


def _path_contains_heading(path: str, heading: str) -> bool:
    """Whether a canonical path proves a heading with appended body text.

    Viewer-derived section titles occasionally retain the first sentence after
    a heading (for example, ``설비 투자 현황 및 계획당분기말 ...``).  The
    ordinary section-heading route keeps exact component matching; a typed
    four-slot table may instead accept that path as its authority only after
    the table itself proves every required header role.
    """

    expected = _compact_heading(heading)
    return bool(expected) and any(
        expected in _compact_heading(part) for part in (path or "").split(">"))


def _investment_slot_roles(slots: object) -> "set[str]":
    roles: set[str] = set()
    for slot in slots or ():
        key = _compact_heading(str(slot))
        for role, aliases in {
                **_INVESTMENT_SLOT_HEADERS,
                **_INVESTMENT_OPTIONAL_SLOT_HEADERS,
                }.items():
            if any(_compact_heading(alias) == key for alias in aliases):
                roles.add(role)
                break
    return roles


def _investment_header_positions(header: list[str]) -> dict[str, int]:
    """Prove one unique target/purpose/plan-amount/period header mapping."""

    compact = [_compact_heading(cell) for cell in header]
    positions: dict[str, int] = {}
    for role, aliases in _INVESTMENT_SLOT_HEADERS.items():
        if role == "금액":
            eligible = [
                index for index, value in enumerate(compact)
                if not any(marker in value for marker in _INVESTMENT_SPENT_AMOUNT_MARKERS)
            ]
            strong = tuple(_compact_heading(alias) for alias in aliases if alias != "금액")
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
            normalized = tuple(_compact_heading(alias) for alias in aliases)
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


def _general_investment_subheading_span(
        text_prompt_safe: str,
        ) -> tuple[int, int] | None:
    """Return one explicit broad-investment subsection boundary.

    This is intentionally weaker than the four-role project-table contract:
    it proves a *section* about investment status, not a particular asset,
    amount, purpose, or period.  A missing or repeated heading is not an
    invitation to choose the lexical top hit, so both cases fail closed.
    """

    body = _strip_prompt_boundaries(text_prompt_safe)
    matches = list(_GENERAL_INVESTMENT_SUBHEADING.finditer(body))
    if len(matches) != 1:
        return None
    match = matches[0]
    return match.start(), match.end()


def _is_general_investment_task(task) -> bool:
    """Whether an exact filing asks for broad investment status, not slots."""

    selector = getattr(task, "document_selector", None)
    if selector is None or not getattr(selector, "rcept_no", None):
        return False
    slots = tuple(
        _compact_heading(slot)
        for slot in (getattr(task, "requested_slots", ()) or ())
        if isinstance(slot, str) and slot.strip())
    # Stage1 keeps the user's broad topic as the one requested narrative
    # surface.  Additional slots are a structured request and must stay on
    # the stricter table path instead of borrowing this general route.
    return len(slots) == 1 and slots[0] == "투자현황"


def _general_investment_blocks(
        text_prompt_safe: str,
        ) -> list[tuple[str, bool]] | None:
    """Keep the headed general-investment subsection and any prose lead-in.

    A canonical parent section can contain raw-material and production tables
    before ``투자 현황``.  Those rows are a different disclosure axis, and
    retaining them made a broad investment answer mix unrelated figures.  A
    prose lead-in is different: it describes the same facilities the requested
    topic covers, and dropping it silently reduces a broad filing summary to
    one project table.  So the prefix is kept only while it carries no table
    row.  The heading itself is the source-owned boundary; every part remains
    prompt-safe and is still split only at structural blocks.
    """

    span = _general_investment_subheading_span(text_prompt_safe)
    if span is None:
        return None
    body = _strip_prompt_boundaries(text_prompt_safe)
    start, _end = span
    prefix = body[:start].strip()
    source_parts = [body[start:].strip()]
    if prefix and not _TABLE_ROW.search(prefix):
        source_parts.insert(0, prefix)
    blocks: list[tuple[str, bool]] = []
    for source_part in source_parts:
        blocks.extend(
            (_wrap_prompt_safe(part.text), part.over_budget)
            for part in split_section(source_part, NARRATIVE_BLOCK_CHARS)
            if part.text)
    return blocks or None


def _authority_bound_requested_slots(task, authority_kind: str | None) -> tuple[str, ...]:
    """Slots proved by the selected source authority rather than literal text.

    Planner slots such as ``주요제품과서비스`` name the same disclosure axis
    as the retrieval topic ``주요 제품 및 서비스``.  Requiring those planner
    spellings to occur verbatim inside the disclosure creates a false
    limitation after the exact business-axis authority has already been
    proved.  Unknown/compound topic slots stay literal and fail closed.
    """

    slots = tuple(
        slot for slot in (getattr(task, "requested_slots", ()) or ())
        if isinstance(slot, str) and slot.strip())
    if not slots:
        return ()
    query = getattr(task, "retrieval_query", "")
    query_key = _compact_heading(query)
    if authority_kind == "business_axis":
        query_kind = _business_axis_kind(query)
        if query_kind in {"business", "products", "revenue", "subsidiaries"}:
            return tuple(
                slot for slot in slots
                if _business_axis_kind(slot) == query_kind)
        # Qualified topics do not inherit semantic aliases.  Only the same
        # compact source topic is authority-bound.
        return tuple(slot for slot in slots
                     if _compact_heading(slot) == query_key)
    if authority_kind == "general_investment":
        return tuple(slot for slot in slots
                     if _compact_heading(slot) == "투자현황")
    if authority_kind == "section_heading":
        return tuple(slot for slot in slots
                     if _compact_heading(slot) == query_key)
    return ()


def _narrative_authority_kind(task) -> str | None:
    """Select execution-only authority enforcement from typed task fields."""

    selector = getattr(task, "document_selector", None)
    query = getattr(task, "retrieval_query", "")
    if disclosed_metrics_for_query(query):
        # 공시 서술지표(순이자마진·수주잔고·가동률 …)는 「II. 사업의 내용」 밖
        # (경영진단·재무건전성 참고사항)에도 실린다. 사업 축 구조 권위를 걸면
        # 그 hit 이 전부 걸러져 not_found 로 닫히므로 어휘 검색 + 경로 우선만 쓴다.
        return None
    if selector is not None and getattr(selector, "rcept_no", None):
        if _compact_heading(query) == "사업의내용":
            return "section_heading"
        requested_roles = _investment_slot_roles(
            getattr(task, "requested_slots", ()))
        if set(_INVESTMENT_SLOT_ROLES).issubset(requested_roles):
            return "slot_table"
        if _is_general_investment_task(task):
            # The canonical source heading can be ``투자계획`` while the
            # user's one requested axis is the broader ``투자 현황``.  The
            # single-slot shape, not the heading spelling, distinguishes this
            # from the strict four-role investment-plan contract.
            return "general_investment"
        if (not _is_investment_plan_query(query)
                and any(_compact_heading(slot) == "투자현황"
                        for slot in getattr(task, "requested_slots", ()) or ())):
            # A general investment-status summary is tied to an explicit
            # source subsection, not to four project-table roles.  Keep it
            # distinct from ``section_heading`` because the latter only
            # verifies a path title and would admit an unrelated prose hit.
            if _is_general_investment_task(task):
                return "general_investment"
            return "section_heading"
        if "투자부동산" in _compact_heading(query):
            return "section_heading"
    if _is_investment_plan_query(query):
        return "investment_plan"
    exact_document = bool(selector is not None and getattr(selector, "rcept_no", None))
    if _is_business_narrative_query(query, exact_document=exact_document):
        return "business_axis"
    return None


def _is_investment_plan_query(retrieval_query: str) -> bool:
    """「설비 투자 현황 및 계획」 표를 구조적으로 요구하는 질의만.

    「투자 현황」처럼 계획 표를 특정하지 않는 표현은 일반 서술 검색(원문 왕복 검증)으로 보낸다 —
    표가 없는 문서에서 구조 필터만으로 not_found 가 나지 않게."""
    compact = _compact_heading(retrieval_query)
    return "투자" in compact and any(
        marker in compact for marker in (
            "계획", "capex", "설비투자", "투자대상", "투자목적",
            "투자금액", "총소요자금", "기지출"))


def _compound_business_topics(retrieval_query: str) -> tuple[str, ...]:
    """Split only clearly compound disclosure axes into typed sidecar cells.

    The vocabulary is a DART section ontology, not a question template.  A
    single-axis request remains one cell; two or more independently provable
    axes become bounded matrix coordinates without changing QueryPlan v0.4.
    """

    compact = _compact_heading(retrieval_query)
    # A whole-section request names a disclosure container rather than one
    # leaf axis.  Resolve it to the two minimum independently provable
    # business coordinates so a generic overview does not collapse to a
    # subsidiary-count boilerplate sentence.  Exact matching is deliberate:
    # a mixed request such as ``사업 내용과 감사의견`` must remain on its
    # original path instead of being over-routed into this matrix.
    if compact in {"사업내용", "사업의내용"}:
        return ("사업부문", "주요 제품 및 서비스")
    topics = []
    if "주요사업" in compact:
        topics.append("주요 사업")
    if "사업부문" in compact or "영업부문" in compact:
        topics.append("사업부문")
    if "제품" in compact or "서비스" in compact:
        topics.append("주요 제품 및 서비스")
    if "매출" in compact or "수익구조" in compact or "수익원" in compact:
        topics.append("매출구성")
    return tuple(topics) if len(topics) >= 2 else ()


# These are disclosure-section concepts, not issuer or question-specific
# rules.  A matrix task has an exact annual/periodic receipt already; this
# vocabulary only confines its narrative retrieval to the relevant child of
# ``II. 사업의 내용`` rather than letting lexical FTS surface audit, director or
# accounting-note prose that happens to mention the same business word.
_BUSINESS_AXIS_ALIASES = {
    "주요사업": (
        "사업의개요", "주요사업", "사업부문", "영업부문", "사업내용", "사업목적",
        "주요제품", "제품및서비스", "영업의현황", "매출및수주상황"),
    "사업부문": ("사업부문", "영업부문", "사업내용"),
    "주요제품서비스": ("주요제품", "제품및서비스", "제품", "서비스", "매출및수주상황"),
    "수익구조": ("수익구조", "수익원", "영업수익", "보험료수익", "이자수익", "수수료수익", "매출", "영업의현황"),
    "매출구성": ("매출구성", "매출", "매출실적", "매출및수주상황", "주요제품매출"),
    "수익원차이": ("수익구조", "수익원", "영업수익", "보험료수익", "이자수익", "수수료수익", "매출", "영업의현황"),
    "핵심자회사구성": ("주요종속회사", "주요자회사", "종속회사", "자회사", "연결대상"),
    "신재생연료전지관련주요사업": (
        "신재생", "태양광", "연료전지", "에너지", "사업의개요", "사업부문"),
}


# ``II. 사업의 내용`` can itself contain a group-wide overview followed by
# many subsidiary, ESG, asset and outlook paragraphs.  These are disclosure
# structure concepts, rather than issuer names or question IDs.  They make a
# request for a business axis prove that axis instead of accepting every
# Section-II paragraph which happens to contain the word ``사업``.
_BUSINESS_CHILD_HEADINGS = (
    ("사업의개요", 48), ("사업부문", 44), ("영업부문", 42),
    ("주요제품", 40), ("제품및서비스", 40), ("영업의현황", 34),
    ("매출및수주상황", 32),
)
_BUSINESS_NON_AXIS_HEADINGS = (
    "재무건전성", "기타참고사항", "주요계약", "연구개발활동",
)
_GROUP_SCOPE_MARKERS = (
    "지배회사", "연결기업", "연결기준", "그룹", "당사는", "당사", "본회사는",
    "회사는", "사업부문", "영업부문", "주요제품", "제품및서비스",
)
_NARROW_SCOPE_MARKERS = (
    "주요종속회사", "종속회사", "종속기업", "자회사", "esg", "환경", "사회",
    "지배구조", "전망", "향후", "현금성자산", "금융자산", "투자자산", "대출채권",
    "투자부동산", "현예금", "유가증권", "운용내역", "수익률", "질권",
    "projectfinancing",
)
_REVENUE_MARKERS = (
    "수익구조", "수익원", "영업수익", "매출액", "매출", "수입보험료", "보험료수익",
    "보험수익", "보험료", "이자수익", "수수료수익", "수수료", "운용이익", "임대수익",
)
_SUBSIDIARY_STRUCTURE_MARKERS = (
    "주요종속회사", "주요자회사", "종속회사에관한사항", "종속기업현황",
    "연결대상종속", "연결대상자회사", "자회사현황", "종속기업명", "자회사명",
    "주요계열사",
)
_TOPIC_ACTIVITY_MARKERS = (
    "제조", "판매", "공급", "서비스", "설계", "개발", "운영", "건설", "정비",
    "용역", "생산", "설치", "유지보수", "기자재", "발전", "사업부문", "영업부문",
    "주요제품", "제품및서비스",
)


def _business_axis_terms(retrieval_query: str) -> tuple[str, ...]:
    compact = _compact_heading(retrieval_query)
    terms: list[str] = []
    for axis, aliases in _BUSINESS_AXIS_ALIASES.items():
        if axis in compact or compact in axis:
            terms.extend(aliases)
    # Compound requests such as ``주요 사업 및 수익구조 차이`` legitimately
    # select more than one business-heading family.  The ordered unique list
    # remains a bounded retrieval-side hint, never a public QueryPlan change.
    return tuple(dict.fromkeys(terms))


def _business_axis_kind(retrieval_query: str) -> str:
    """Classify a generic business request by the evidence it must prove."""
    compact = _compact_heading(retrieval_query)
    if "자회사" in compact or "종속회사" in compact or "종속기업" in compact:
        return "subsidiaries"
    if "수익" in compact or "수익원" in compact or "매출" in compact:
        return "revenue"
    if "제품" in compact or "서비스" in compact:
        return "products"
    # A non-boilerplate topic together with an otherwise business-shaped
    # request (e.g. a technology, energy source or product family) requires a
    # locally stated business activity, not merely a shareholder reference.
    if _business_axis_anchors(retrieval_query):
        return "topic"
    return "business"


def _business_child_score(path: str) -> int:
    compact_path = _compact_heading(path)
    if any(marker in compact_path for marker in _BUSINESS_NON_AXIS_HEADINGS):
        return -48
    return max((score for heading, score in _BUSINESS_CHILD_HEADINGS
                if heading in compact_path), default=0)


def _business_text(row) -> str:
    return _compact_heading(_strip_prompt_boundaries(
        getattr(row, "text_prompt_safe", None) or getattr(row, "text", "")))


def _count_markers(text: str, markers: tuple[str, ...]) -> int:
    return sum(marker in text for marker in markers)


def _has_substantive_product_map(body: str) -> bool:
    """Whether a product/service child contains more than a title/generic row."""

    source = _strip_prompt_boundaries(body)
    substantive: set[str] = set()
    for line in source.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = _table_cells(line)
        if _is_table_rule(cells):
            continue
        identity = _compact_heading(" ".join(cells[:3]))
        if (not identity or any(marker in identity for marker in (
                "사업부문품목", "사업부문주요", "합계", "소계", "내부거래",
                "기타"))):
            continue
        substantive.add(identity)
    if len(substantive) >= 2:
        return True
    prose = _compact_heading(" ".join(
        line for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("|")))
    return (len(prose) >= 320
            and _count_markers(prose, (
                "서비스", "제품", "검색", "커머스", "콘텐츠", "플랫폼",
                "제조", "생산", "판매", "공급")) >= 3)


def _topic_has_local_activity(body: str, anchors: tuple[str, ...]) -> bool:
    """Require a requested topic and a business action in one local window.

    A report can mention a topic only in a pledge, shareholding or a note to a
    contract.  Looking around the actual topic occurrence prevents those
    references from becoming a claimed operating business while retaining a
    full cited source block for the eventual answer.
    """
    for anchor in anchors:
        start = 0
        while True:
            at = body.find(anchor, start)
            if at < 0:
                break
            window = body[max(0, at - 96):at + len(anchor) + 128]
            if any(marker in window for marker in _TOPIC_ACTIVITY_MARKERS):
                return True
            start = at + len(anchor)
    return False


_BUSINESS_QUERY_STOPWORDS = {
    "주요", "관련", "차이", "변화", "비교", "사업", "제품", "서비스", "수익", "수익구조",
    "수익원", "구조", "구성", "보고서", "사업보고서", "기준", "설명", "정리", "및", "과",
    "의", "에", "에서", "대한", "내용", "핵심", "자회사", "종속회사", "사업부문",
    "영업부문", "사업내용", "사업목적", "매출구성",
    # ``개요`` 는 공시 표제어이지 주제어가 아니다. 빠져 있으면 「사업의 개요」가
    # ``개요`` 를 주제 앵커로 삼아 `topic` 으로 분류되고, 정작 그 표제의 본문
    # (`II. 사업의 내용 > 1. 사업의 개요`)이 사업 축 가점을 못 받아 0점이 된다.
    # 같은 뜻의 「사업의 내용」은 ``내용`` 이 목록에 있어 통과한다.
    "개요", "현황", "요약", "사업개요", "사업의개요", "주요사업",
}


def _business_axis_anchors(retrieval_query: str) -> tuple[str, ...]:
    """Specific topic words that must occur in a Section-II source block."""
    words = re.findall(r"[가-힣A-Za-z0-9]+", retrieval_query or "")
    normalized = []
    for word in words:
        value = word
        for suffix in ("에서", "으로", "에게", "부터", "까지", "의", "을", "를",
                       "이", "가", "은", "는", "과", "와", "로"):
            if value.endswith(suffix) and len(value) - len(suffix) >= 2:
                value = value[:-len(suffix)]
                break
        normalized.append(value)
    return tuple(dict.fromkeys(
        word for word in normalized
        if len(word) >= 2 and _compact_heading(word) not in _BUSINESS_QUERY_STOPWORDS))


def _is_business_narrative_query(retrieval_query: str, *, exact_document: bool = False) -> bool:
    """사업 축(``II. 사업의 내용``) authority 를 걸 질의인가.

    명시적 사업 축 용어(사업·제품·서비스·수익·자회사, 별칭 표)가 있으면 항상 그렇다. 임의 주제어
    (원전·태양광 같은 anchor)만 있는 질의는 **문서가 하나로 고정된 요청**에서만 사업 축으로 본다 —
    문서가 고정되지 않은 일반 서술 검색에 이 규칙을 쓰면 배당·감사의견·소송처럼 II장 밖 주제가
    전부 걸러져 not_found 로 닫힌다."""
    compact = _compact_heading(retrieval_query)
    if _business_axis_terms(retrieval_query) or any(
            word in compact for word in ("사업", "제품", "서비스", "수익", "자회사")):
        return True
    return exact_document and bool(_business_axis_anchors(retrieval_query))


def _business_axis_score(task, row) -> int:
    """Rank a Section-II chunk only when it proves the requested business axis.

    The path says which disclosure child a chunk belongs to; its own leading
    text says whether it describes the consolidated/group business or a
    narrow subsidiary, asset, ESG or outlook aside.  Both are needed because
    a canonical section can contain all of them.
    """
    query = getattr(task, "retrieval_query", "")
    terms = _business_axis_terms(query)
    anchors = _business_axis_anchors(query)
    if not _path_has_heading(getattr(row, "path", ""), "사업의 내용"):
        return 0
    path = _compact_heading(getattr(row, "path", ""))
    body = _business_text(row)
    child_score = _business_child_score(getattr(row, "path", ""))
    kind = _business_axis_kind(query)

    # The corp_code filter proves which filing owns a chunk; it does not prove
    # that a paragraph inside a consolidated filing describes the selected
    # issuer rather than one sibling company.  Prefer an explicit issuer
    # surface in the leading block.  This is only a ranking signal: disclosures
    # written solely as ``당사`` remain eligible and no company dictionary is
    # embedded here.
    expected_corp = _compact_heading(getattr(task, "expected_corp_name", ""))
    issuer_bound = bool(expected_corp and expected_corp in body[:900])

    # 금융지주 공시는 전사 구성을 ``영업의 종류`` 또는 본문 표의
    # ``사업부문 / 사업의 내용 / 계열사``로 설명하는 경우가 많다. 반대로
    # ``부문별 손익``은 사업의 존재를 보조할 뿐, 주요 사업 자체를 설명하는
    # authority는 아니다. 이는 회사명이 아니라 공시 구조 역할에 따른 가중치다.
    business_map = all(marker in body for marker in ("사업부문", "사업의내용", "계열사"))
    if kind in {"business", "subsidiaries"} and "사업의개요" in path:
        child_score += 36
    if (kind == "business"
            and any(marker in path for marker in (
                "주요제품및서비스", "주요제품서비스", "제품및서비스"))
            and _has_substantive_product_map(
                getattr(row, "text_prompt_safe", None)
                or getattr(row, "text", ""))):
        # For a comparison of "main business", a source-labelled product /
        # service map is more discriminative than a repeated corporate-intro
        # paragraph.  This is a filing-structure preference only: it applies
        # to every exact document and embeds no issuer, date or case value.
        # An explicit ``주요 사업`` coordinate should prefer an independently
        # available business overview; a bare broad ``사업`` query retains the
        # established product-map fallback.  Product-specific requests still
        # receive their dedicated stronger ranking below.
        overview_query = any(marker in _compact_heading(query)
                             for marker in ("주요사업", "사업개요", "사업의개요"))
        child_score += (20 if overview_query else 125)
        price_mechanics = _count_markers(body, (
            "가격변동", "판매가격", "판매단가", "평균판매", "기준가",
            "산출기준", "산출방법", "판가", "롤마진",
        ))
        if price_mechanics >= 2:
            # A product child can append a much larger price-calculation
            # subsection after a valid product table.  It remains the right
            # authority for a *product* request, but a broad main-business
            # cell should prefer the independently eligible business overview
            # rather than forwarding that price-heavy section wholesale.
            child_score -= 120
    elif kind in {"business", "subsidiaries"} and (
            "영업의종류" in path or business_map):
        child_score += 34
    elif kind == "business" and ("부문별손익" in path or "영업실적" in path):
        child_score -= 28

    # An exact receipt is not sufficient authority for a business answer when
    # it only supplies a finance/reference/R&D child.  A topic may use another
    # child only when the source itself ties the topic to an operating action.
    if kind != "topic" and child_score <= 0:
        return 0

    score = child_score
    if issuer_bound and kind != "topic":
        score += 72
    for term in terms:
        if term in path:
            score += 4
        elif term in body:
            score += 1
    anchor_hits = 0
    for anchor in anchors:
        if anchor in path:
            anchor_hits += 1
            score += 12
        elif anchor in body:
            anchor_hits += 1
            score += 3
    # Unknown, issuer-neutral industry topics (for example a technology or
    # energy term) are strict anchors.  They must not silently degrade to a
    # generic business overview or an unrelated disclosure section.
    if anchors and not anchor_hits:
        return 0
    if kind == "subsidiaries":
        # A historical sentence saying that one company became a subsidiary is
        # not a composition.  Require the disclosure's list/table/``현황``
        # structure before it can support a key-subsidiaries cell.
        listed_company_structure = (
            business_map
            or ("상장회사" in body and "비상장회사" in body)
        )
        if (not _count_markers(path + body, _SUBSIDIARY_STRUCTURE_MARKERS)
                and not listed_company_structure):
            return 0
        return (score + 48 + 4 * _count_markers(
            path + body, _SUBSIDIARY_STRUCTURE_MARKERS)
                + (32 if listed_company_structure else 0))

    if kind == "revenue":
        # ``현금성자산`` or a single affiliate's asset table may contain an
        # interest/rent number, but cannot by itself prove the group's revenue
        # structure.  Require a revenue-bearing business statement first.
        # 하나의 긴 canonical chunk 뒤쪽에 붙은 자회사 수익 문장이 앞쪽의
        # 준비금·자산 표를 수익구조로 승격시키지 않도록 leading block에서만
        # positive evidence를 센다.
        head = body[:1800]
        revenue_hits = _count_markers(head, _REVENUE_MARKERS)
        if not revenue_hits:
            return 0
        score += 26 + 3 * revenue_hits
        # A sales/order or operating-results child is normally a more direct
        # revenue-structure source than a general company introduction.
        if "매출및수주상황" in path:
            score += 72
        elif "영업의현황" in path:
            score += 20
        if _count_markers(head, _GROUP_SCOPE_MARKERS):
            score += 32
        if "사업의개요" in path and "수입보험료" in head:
            # 생명보험 공시는 전사 상품별 수입보험료 구성을 개요에서 직접
            # 제시한다. 뒤쪽 자금조달·현지법인 표보다 질문에 가까운 근거다.
            score += 16
        if _count_markers(head, _NARROW_SCOPE_MARKERS):
            score -= 24
        return score if score > 0 else 0

    if kind == "products":
        # 제품·서비스 요청은 일반 사업개요보다 제품/서비스 표·절을 우선한다.
        # 회사나 품목을 열거하지 않고 공시 구조 역할만 사용하므로 처음 보는
        # 발행인에도 같은 규칙이 적용된다.
        product_markers = (
            "주요제품", "제품및서비스", "제품", "상품", "서비스", "품목", "브랜드")
        head = body[:1800]
        if not _count_markers(path + head, product_markers):
            return 0
        if "주요제품" in path or "제품및서비스" in path:
            score += 72
        elif "매출및수주상황" in path:
            score += 48
        elif "사업의개요" in path:
            score += 8
        score += 4 * _count_markers(head, product_markers)
        if _count_markers(head, _GROUP_SCOPE_MARKERS):
            score += 12
        if _count_markers(head, _NARROW_SCOPE_MARKERS):
            score -= 12
        return score if score > 0 else 0

    if kind == "topic":
        # A company/section may mention a requested industry in a financing,
        # outlook or contract note.  It is usable only with a nearby concrete
        # operating action (manufacture, supply, design, service, etc.).
        if not _topic_has_local_activity(body, anchors):
            return 0
        score += 30 + 5 * anchor_hits
        if child_score > 0:
            score += 12
        # A requested operating topic is better supported by a disclosed
        # business-segment/operating-overview child than by an adjacent group
        # funding table that merely carries a compact business map.
        if any(marker in path for marker in (
                "사업부문별영업실적", "영업의개황", "종류별영업실적")):
            score += 24
        if "자금조달운용현황" in path:
            score -= 18
        head = body[:1800]
        if _count_markers(head, _GROUP_SCOPE_MARKERS):
            score += 12
        if _count_markers(head, _NARROW_SCOPE_MARKERS):
            score -= 12
        return score if score > 0 else 0

    # Main-business requests favour the group/segment/product overview over a
    # paragraph that happens to be about one subsidiary or a non-operating
    # programme.  The full selected paragraph/table remains untouched.
    head = body[:1800]
    if _count_markers(head, _GROUP_SCOPE_MARKERS):
        score += 28
    if _count_markers(head, _NARROW_SCOPE_MARKERS):
        score -= 26
    return score if score > 0 else 0


def _slot_table_text(
        text_prompt_safe: str, *, retrieval_query: str,
        authority_path: str | None = None,
        ) -> str | None:
    """Return one whole heading+table block with all investment headers.

    The selected markdown table is never sliced by character position.  If
    more than one table satisfies the same four-header ontology, selection is
    ambiguous and fails closed.
    """

    body = _strip_prompt_boundaries(text_prompt_safe)
    lines = body.splitlines()
    # A normal markdown subsection puts the heading immediately before its
    # table.  Some canonical chunks begin at the table, however, with that
    # heading retained only in the proved section path.  The latter is safe
    # only for the exact-document route and only with this complete header
    # ontology; do not treat an arbitrary table as an investment plan.
    path_proves_heading = _path_contains_heading(authority_path or "", retrieval_query)
    candidates: list[tuple[int | None, int, int]] = []
    index = 0
    while index + 1 < len(lines):
        # Viewer HTML can emit a tiny unit table immediately adjacent to the
        # main table without a blank line.  Detect each header+rule pair rather
        # than treating every consecutive pipe-prefixed line as one table.
        if (not lines[index].lstrip().startswith("|")
                or not lines[index + 1].lstrip().startswith("|")
                or not _is_table_rule(_table_cells(lines[index + 1]))
                or not _investment_header_positions(_table_cells(lines[index]))):
            index += 1
            continue
        start = index
        end = index + 2
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            if (end + 1 < len(lines)
                    and lines[end + 1].lstrip().startswith("|")
                    and _is_table_rule(_table_cells(lines[end + 1]))
                    and _investment_header_positions(_table_cells(lines[end]))):
                break
            end += 1
        heading_index = None
        for before in range(start - 1, max(-1, start - 12), -1):
            if (_compact_heading(retrieval_query)
                    in _compact_heading(lines[before])):
                heading_index = before
                break
        if heading_index is not None:
            candidates.append((heading_index, start, end))
        elif path_proves_heading:
            candidates.append((None, start, end))
        index = end
    if len(candidates) != 1:
        return None
    heading_index, start, end = candidates[0]
    selected_start = heading_index if heading_index is not None else start
    if heading_index is None and path_proves_heading:
        # Viewer HTML frequently stores the unit in a small borderless table
        # immediately before the main investment table, while the heading is
        # available only through the canonical path.  Preserve that unit only
        # when the bounded intervening rows are blank/table rows; this cannot
        # borrow a unit from an unrelated preceding paragraph or table.
        for before in range(start - 1, max(-1, start - 8), -1):
            if "단위" not in lines[before]:
                continue
            between = lines[before:start]
            if all(not row.strip() or row.lstrip().startswith("|")
                   for row in between):
                selected_start = before
            break
    selected = "\n".join(lines[selected_start:end]).strip()
    if heading_index is None:
        # Keep the authority label in the answer body even when the source
        # chunk starts at the table boundary rather than silently inventing a
        # heading from the user question.
        selected = f"{retrieval_query}\n{selected}"
    return _wrap_prompt_safe(selected) if selected else None


def _investment_empty_selection_text(source_table: str) -> str:
    """Describe only the heading proved by the selected source table."""

    heading = ""
    for line in _strip_prompt_boundaries(source_table).splitlines():
        if line.lstrip().startswith("|"):
            break
        compact = _compact_heading(line)
        if re.search(r"투자(?:현황|계획)", compact):
            heading = compact
    has_status = "투자현황" in heading
    has_plan = "투자계획" in heading or "투자현황및계획" in heading
    title = ("투자 현황 및 계획" if has_status and has_plan
             else "투자현황" if has_status
             else "투자계획" if has_plan else "투자 관련")
    result = (f"확인한 {title} 표에서 기간이 비어 있거나 미정·미기재 등으로 "
              "표시된 항목은 확인되지 않습니다.")
    if has_status and not has_plan:
        result += (" 이는 투자현황 표에 대한 확인 결과이며, 별도 투자계획 "
                   "표의 기간 미기재 여부까지 확인한 것은 아닙니다.")
    return result


def _select_investment_row(
        text_prompt_safe: str, selector: InvestmentRowSelector,
        ) -> str | InvestmentEmptySelection | None:
    """Return one exact issuer+investment row with its source headers/unit.

    The selector never fuzzily ranks rows.  It accepts one normalized exact
    investment-name match and, when the table provides an issuer column, one
    exact issuer match for every operation.  Zero or multiple rows fail
    closed.
    """

    body = _strip_prompt_boundaries(text_prompt_safe)
    lines = body.splitlines()
    tables: list[tuple[int, int, list[str], dict[str, int], int | None]] = []
    for index in range(len(lines) - 1):
        if (not lines[index].lstrip().startswith("|")
                or not lines[index + 1].lstrip().startswith("|")
                or not _is_table_rule(_table_cells(lines[index + 1]))):
            continue
        header = _table_cells(lines[index])
        positions = _investment_header_positions(header)
        if not positions:
            continue
        compact = [_compact_heading(cell) for cell in header]
        spent_hits = [
            pos for pos, value in enumerate(compact)
            if any(_compact_heading(alias) == value
                   for alias in _INVESTMENT_OPTIONAL_SLOT_HEADERS[
                       "기지출금액"])
        ]
        if selector.operation == "exact" and len(spent_hits) != 1:
            continue
        company_hits = [
            pos for pos, value in enumerate(compact)
            if value in {"해당회사", "회사", "회사명", "법인명"}
        ]
        if len(company_hits) > 1:
            continue
        end = index + 2
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            end += 1
        tables.append((
            index, end, header, positions,
            company_hits[0] if company_hits else None,
        ))
    if len(tables) != 1:
        return None
    start, end, header, positions, company_pos = tables[0]
    target_pos = positions["투자대상"]
    expected_target = _compact_heading(selector.investment_name or "")
    expected_issuer = _compact_heading(selector.issuer_name)
    candidate_rows: list[tuple[str, list[str]]] = []
    malformed_rows = False
    for line in lines[start + 2:end]:
        cells = _table_cells(line)
        if len(cells) != len(header):
            malformed_rows = True
            continue
        if (company_pos is not None
                and _compact_heading(cells[company_pos]) != expected_issuer):
            continue
        candidate_rows.append((line, cells))

    if selector.operation == "exact":
        matches = [
            line for line, cells in candidate_rows
            if _compact_heading(cells[target_pos]) == expected_target
        ]
    elif selector.operation in {"start_year", "end_year_on_or_after"}:
        period_pos = positions["기간"]
        matches = []
        for line, cells in candidate_rows:
            years = [int(value) for value in re.findall(
                r"(?<![0-9])(?:19|20)[0-9]{2}", cells[period_pos])]
            if not years:
                continue
            year = years[0] if selector.operation == "start_year" else years[-1]
            if (selector.operation == "start_year"
                    and year == selector.criterion_year):
                matches.append(line)
            elif (selector.operation == "end_year_on_or_after"
                    and year >= selector.criterion_year):
                matches.append(line)
    elif selector.operation == "missing_period":
        period_pos = positions["기간"]
        missing = {"", "-", "미정", "미기재", "해당없음"}
        matches = [
            line for line, cells in candidate_rows
            if _compact_heading(cells[period_pos]) in missing
        ]
    else:  # argmax_amount
        amount_pos = positions["금액"]
        table_unit = _adjacent_investment_table_unit(lines, start)
        ranked: list[tuple[Decimal, str]] = []
        for line, cells in candidate_rows:
            # A total is a row classification, not a substring ban over the
            # whole row: an ordinary investment purpose can legitimately say
            # "합계" or "소계".  The structurally identified investment-target
            # cell must itself be one of the closed total markers.
            if (_compact_heading(cells[target_pos])
                    in _INVESTMENT_TOTAL_MARKERS):
                continue
            numeric = re.fullmatch(
                r"\s*(?P<value>[0-9][0-9,]*(?:\.[0-9]+)?)\s*"
                r"(?P<unit>백만원|천원|억원|원)?\s*",
                cells[amount_pos],
            )
            if numeric is None:
                return None
            unit = numeric.group("unit") or table_unit
            if unit not in _INVESTMENT_UNIT_SCALE:
                return None
            value_won = (
                Decimal(numeric.group("value").replace(",", ""))
                * _INVESTMENT_UNIT_SCALE[unit]
            )
            ranked.append((value_won, line))
        maximum = max((value for value, _line in ranked), default=None)
        matches = [line for value, line in ranked if value == maximum]
    if not matches and selector.operation == "missing_period":
        # No issuer rows (or an unreadable row) cannot prove that every
        # relevant row discloses its period.
        if not candidate_rows or malformed_rows:
            return None
        return InvestmentEmptySelection(
            source_table=text_prompt_safe, operation=selector.operation)
    if not matches or (selector.operation in {"exact", "argmax_amount"}
                       and len(matches) != 1):
        return None
    selected = "\n".join([
        *lines[:start], lines[start], lines[start + 1], *matches,
    ]).strip()
    return _wrap_prompt_safe(selected) if selected else None


def _investment_table_rows_for_issuer(
        text_prompt_safe: str, issuer_names: tuple[str, ...],
        ) -> str:
    """Narrow one proved slot-table listing to the one named issuer's rows.

    A periodic filing can disclose a whole reporting group's equipment
    investment under one ``설비 투자 현황 및 계획`` heading, with each row's
    ``해당회사`` column naming the actual investing entity (issuer, an
    affiliate, ...).  Listing that table for one named company must show only
    that company's own rows: otherwise the listing silently mixes in
    unrelated affiliates' investment amounts, and its maximum stops matching
    the single-row ``argmax_amount`` selection (#169), which is already
    issuer-scoped via ``_select_investment_row``.  Returns the input
    unchanged whenever the table carries no unique company column, the
    caller does not name exactly one company, or that company matches no row
    (fail open — never turn a valid listing into an empty one on account of
    this narrowing alone).
    """

    if len(issuer_names) != 1 or not issuer_names[0].strip():
        return text_prompt_safe
    expected_issuer = _compact_heading(issuer_names[0])
    if not expected_issuer:
        return text_prompt_safe
    body = _strip_prompt_boundaries(text_prompt_safe)
    lines = body.splitlines()
    for index in range(len(lines) - 1):
        if (not lines[index].lstrip().startswith("|")
                or not lines[index + 1].lstrip().startswith("|")
                or not _is_table_rule(_table_cells(lines[index + 1]))):
            continue
        header = _table_cells(lines[index])
        if not _investment_header_positions(header):
            continue
        compact = [_compact_heading(cell) for cell in header]
        company_hits = [
            pos for pos, value in enumerate(compact)
            if value in {"해당회사", "회사", "회사명", "법인명"}
        ]
        if len(company_hits) != 1:
            return text_prompt_safe
        company_pos = company_hits[0]
        end = index + 2
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            end += 1
        kept = []
        for line in lines[index + 2:end]:
            cells = _table_cells(line)
            if (len(cells) == len(header)
                    and _compact_heading(cells[company_pos])
                    != expected_issuer):
                continue
            kept.append(line)
        if not kept:
            return text_prompt_safe
        selected = "\n".join([
            *lines[:index], lines[index], lines[index + 1], *kept,
            *lines[end:],
        ]).strip()
        return _wrap_prompt_safe(selected) if selected else text_prompt_safe
    return text_prompt_safe


def _investment_plan_text(text_prompt_safe: str) -> str | None:
    """Return one structurally labelled investment-plan table.

    Periodic disclosures use both project-style four-role tables and annual
    plan-vs-actual tables.  The latter is still authoritative when its header
    explicitly says ``투자계획`` and the surrounding source labels it as a
    major-investment status/plan.  Generic tables that merely contain the word
    ``계획`` are rejected.
    """

    body = _strip_prompt_boundaries(text_prompt_safe)
    lines = body.splitlines()
    candidates: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = index
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            index += 1
        end = index
        header = _compact_heading("\n".join(lines[start:min(end, start + 3)]))
        if "투자계획" not in header and "향후투자계획" not in header:
            continue
        context_start = max(0, start - 12)
        context = _compact_heading("\n".join(lines[context_start:start]))
        if not ("주요투자현황" in context
                or "투자현황및향후계획" in context
                or "투자현황및계획" in context):
            continue
        selected_start = start
        for before in range(start - 1, context_start - 1, -1):
            compact = _compact_heading(lines[before])
            if "주요투자현황" in compact:
                selected_start = before
                break
            if "단위" in compact:
                selected_start = before
        candidates.append((selected_start, end))
    if len(candidates) != 1:
        return None
    start, end = candidates[0]
    selected = "\n".join(lines[start:end]).strip()
    return _wrap_prompt_safe(selected) if selected else None


def _authority_hit_matches(task, hit, authority_kind: str) -> bool:
    if authority_kind == "section_heading":
        return _path_has_heading(
            getattr(hit, "path", ""), task.retrieval_query)
    if authority_kind == "general_investment":
        return _general_investment_subheading_span(
            getattr(hit, "text_prompt_safe", "")) is not None
    if authority_kind == "slot_table":
        return _slot_table_text(
            getattr(hit, "text_prompt_safe", ""),
            retrieval_query=task.retrieval_query,
            authority_path=getattr(hit, "path", "")) is not None
    if authority_kind == "business_axis":
        return _business_axis_score(task, hit) > 0
    if authority_kind == "investment_plan":
        return _investment_plan_text(
            getattr(hit, "text_prompt_safe", "")) is not None
    return True


def _prioritize_authority_hits(
        task, hits: tuple, authority_kind: str, *, prefer_latest: bool = False,
        ) -> tuple:
    """Prefer the canonical overview child for a whole business section."""

    if authority_kind == "business_axis":
        return tuple(sorted(
            hits,
            key=lambda hit: (-_business_axis_score(task, hit), hit.rank, hit.rcept_dt, hit.chunk_id),
        ))
    if authority_kind == "investment_plan":
        if prefer_latest:
            return tuple(sorted(
                hits, key=lambda hit: (hit.rcept_dt, -hit.rank, hit.chunk_id),
                reverse=True))
        return tuple(sorted(hits, key=lambda hit: (hit.rank, hit.rcept_dt, hit.chunk_id)))
    if (authority_kind != "section_heading"
            or _compact_heading(task.retrieval_query) != "사업의내용"):
        return hits
    preferred = tuple(
        hit for hit in hits
        if _path_has_heading(getattr(hit, "path", ""), "사업의 개요"))
    return preferred


def _exact_document_authority_hits(
        rm, task, authority_kind: str, *, rows: tuple[object, ...] | None = None,
        ) -> tuple:
    """Read bounded canonical chunks for one proved document authority.

    FTS remains discovery for ordinary narrative tasks.  Once Stage1 supplies
    an exact receipt plus a section/table authority, global lexical rank must
    not choose a sibling section.  The returned chunks are still independently
    round-tripped through ``read_section`` before becoming claims.
    """

    chunks = getattr(rm, "chunks", None)
    selector = getattr(task, "document_selector", None)
    receipt = getattr(selector, "rcept_no", None)
    if not callable(chunks) or not receipt:
        return ()
    doc_id = getattr(selector, "doc_id", None) or f"periodic_{receipt}"
    corp_codes = tuple(getattr(task, "corp_codes", ()) or ())
    corp_code = corp_codes[0] if len(corp_codes) == 1 else None
    if rows is None:
        try:
            rows = tuple(chunks(
                corp_code=corp_code, doc_group=getattr(selector, "doc_group", None),
                doc_id=doc_id, projection="llm"))
        except (TypeError, ValueError):
            return ()
    hits = tuple(
        SimpleNamespace(
            rank=index, chunk_id=row.chunk_id, section_id=row.section_id,
            doc_id=row.doc_id, source_file_id=row.source_file_id,
            corp_code=getattr(row, "corp_code", corp_code),
            corp_name=getattr(row, "corp_name", ""),
            doc_group=getattr(row, "doc_group", None),
            rcept_dt=row.rcept_dt, path=row.path, locator=row.locator,
            text_prompt_safe=row.text_prompt_safe,
            evidence_id=getattr(row, "evidence_id", None),
        )
        for index, row in enumerate(rows, start=1)
        if row.rcept_dt <= task.as_of
        and _authority_hit_matches(task, row, authority_kind)
    )
    hits = _prioritize_authority_hits(task, hits, authority_kind)
    if authority_kind == "slot_table" and len(hits) != 1:
        return ()
    return hits[:TOP_K]


def _same_document_affiliate_detail_hits(rm, task, hits: tuple) -> tuple:
    """Recover missing, explicitly labelled detail within one proved report."""

    query = _compact_heading(task.retrieval_query)
    affiliates = "계열회사" in query
    group_npl = "고정이하여신비율" in query and "은행" not in query
    if not affiliates and not group_npl:
        return hits
    documents = {hit.doc_id for hit in hits}
    chunks = getattr(rm, "chunks", None)
    if len(documents) != 1 or not callable(chunks):
        return hits
    try:
        rows = tuple(chunks(doc_id=next(iter(documents)), projection="llm"))
    except (TypeError, ValueError):
        return hits
    details = tuple(
        SimpleNamespace(
            rank=0, chunk_id=row.chunk_id, section_id=row.section_id,
            doc_id=row.doc_id, source_file_id=row.source_file_id,
            rcept_dt=row.rcept_dt, path=row.path, locator=row.locator,
            text_prompt_safe=row.text_prompt_safe)
        for row in rows
        if ((affiliates and "계열회사현황상세" in _compact_heading(row.path)
             and any({"상장여부", "회사수", "기업명"}.issubset(
                 {_compact_heading(cell) for cell in _table_cells(line)})
                     for line in row.text_prompt_safe.splitlines()))
            or (group_npl and "그룹연결기준고정이하여신비율"
                in _compact_heading(row.text_prompt_safe)))
        and row.rcept_dt <= task.as_of)
    return (*details[:2], *hits)[:TOP_K] if details else hits


def _authority_roundtripped_text(
        task, hit, section, roundtripped_chunk: str,
        authority_kind: str | None,
        *, row_selector: InvestmentRowSelector | None = None,
        corp_name: str = "",
        aggregation_request: InvestmentAggregationRequest | None = None,
        ) -> str | None:
    if authority_kind is None:
        query = _compact_heading(task.retrieval_query)
        section_path = _compact_heading(getattr(section, "path", ""))
        if "라이선스아웃" in query:
            # A search chunk may stop after five contracts although the same
            # verified section's explicitly labelled table has ten. Recover
            # only that complete table, not the surrounding R&D pipeline.
            source_lines = _strip_prompt_boundaries(
                getattr(section, "text", "")).splitlines()
            candidates = []
            expected = ["품목", "계약상대방", "대상지역", "계약체결일",
                        "계약종료일", "총계약금액", "수취금액", "진행단계"]
            for index, line in enumerate(source_lines[:-2]):
                if [_compact_heading(cell) for cell in _table_cells(line)] != expected:
                    continue
                if not _is_table_rule(_table_cells(source_lines[index + 1])):
                    continue
                caption = "\n".join(source_lines[max(0, index - 3):index])
                if "라이선스아웃" not in _compact_heading(caption):
                    continue
                end = index + 2
                while end < len(source_lines) and source_lines[end].lstrip().startswith("|"):
                    end += 1
                candidates.append(caption + "\n" + "\n".join(source_lines[index:end]))
            if len(candidates) == 1 and len(candidates[0]) <= NARRATIVE_BLOCK_CHARS:
                return _wrap_prompt_safe(candidates[0])
        if (any(term in query for term in ("우발부채", "약정"))
                and "우발부채" in section_path and "약정" in section_path
                and len(getattr(section, "text", "")) <= NARRATIVE_BLOCK_CHARS):
            # The compact note is a single proved source authority. Its
            # empty XBRL caption tables must not hide the following values.
            return getattr(section, "text", roundtripped_chunk)
        if disclosed_metrics_for_query(task.retrieval_query):
            # A structural search chunk can end at a standalone date/unit
            # table. Recover only the immediately adjacent data table from
            # the independently verified section, never a distant table.
            body = _strip_prompt_boundaries(roundtripped_chunk)
            pieces = [piece for piece in body.split("\n\n") if piece.strip()]
            source = _strip_prompt_boundaries(getattr(section, "text", ""))
            start = source.find(body)
            if pieces and _is_table_metadata_piece(pieces[-1]) and start >= 0:
                following = source[start + len(body):].lstrip("\n")
                table_lines: list[str] = []
                for line in following.splitlines():
                    if not line.lstrip().startswith("|"):
                        break
                    table_lines.append(line)
                if (len(table_lines) >= 3
                        and _is_table_rule(_table_cells(table_lines[1]))):
                    expanded = body + "\n\n" + "\n".join(table_lines)
                    if len(expanded) <= NARRATIVE_BLOCK_CHARS:
                        return _wrap_prompt_safe(expanded)
        return roundtripped_chunk
    if authority_kind == "section_heading":
        if not _path_has_heading(
                getattr(section, "path", ""), task.retrieval_query):
            return None
        return roundtripped_chunk
    if authority_kind == "general_investment":
        # Recheck after read_section: a lexical hit is not authority.  The
        # same one-heading rule also prevents two nearby investment tables
        # from being merged into an invented "overall" answer.
        return (roundtripped_chunk
                if _general_investment_subheading_span(roundtripped_chunk)
                is not None else None)
    if authority_kind == "slot_table":
        # Re-extract from the independently round-tripped *section*.  DART
        # often puts ``(단위: 백만원)`` in a tiny borderless table immediately
        # before the investment table; the indexed main-table chunk alone
        # therefore loses a material unit.  The section is still canonical
        # prompt-safe evidence and ``_slot_table_text`` returns only the one
        # heading+unit+table block that proves all four requested roles.
        table = _slot_table_text(
            getattr(section, "text", roundtripped_chunk),
            retrieval_query=task.retrieval_query,
            authority_path=getattr(section, "path", ""))
        if table is None:
            return table
        if row_selector is not None:
            return _select_investment_row(table, row_selector)
        if aggregation_request is not None:
            # A literal, task-bound sum request (for example, an asserted
            # "25건의 총 소요자금 합계") is already row-count-checked against
            # the whole disclosed table and may deliberately span every
            # issuer in it; narrowing here would silently drop asserted rows.
            return table
        # #169 — a listing (no row selector, no aggregation) must stay scoped
        # to the one named company the same way a single-row selection
        # already is; otherwise its shown maximum can exceed the
        # issuer-scoped argmax answer for the identical document/section.
        # ``task.corp_names`` is the typed field but is not always populated
        # by every compiler path; ``corp_name`` is the caller's
        # already-resolved single-issuer fallback (the same name used in
        # this task's own claim labels).
        issuer_names = tuple(getattr(task, "corp_names", ()) or ())
        if not issuer_names and corp_name:
            issuer_names = (corp_name,)
        return _investment_table_rows_for_issuer(table, issuer_names)
    if authority_kind == "business_axis":
        if _business_axis_score(task, hit) <= 0 or _business_axis_score(task, section) <= 0:
            return None
        return _issuer_owned_business_text(task, roundtripped_chunk)
    if authority_kind == "investment_plan":
        return _investment_plan_text(getattr(section, "text", roundtripped_chunk))
    return None


_AFFILIATE_BLOCK_HEADING = re.compile(
    r"(?m)^\s*\[(?:주요\s*)?(?:종속회사|자회사)[^]]*\]"
)


def _explicit_business_subject(text: str, issuer: str) -> str | None:
    """Identify an explicit subject, never infer subsidiary ownership."""
    body = _strip_prompt_boundaries(text)
    match = re.match(r"\s*([A-Za-z가-힣0-9&·.]{2,40})의\s*경영진", body)
    if match is None:
        return None
    subject = match.group(1)
    if subject in {"회사", "당사", "기업", "그룹"} or _compact_heading(subject) == _compact_heading(issuer):
        return None
    return subject


def _issuer_owned_business_text(task, text_prompt_safe: str) -> str:
    """Cut an issuer-owned business/revenue block before affiliate profiles.

    A consolidated overview can place the selected issuer and several named
    affiliates inside one canonical chunk.  The receipt/corp_code proves the
    filing owner, not the owner of every paragraph.  When the prefix itself
    explicitly names the selected issuer, an explicit affiliate heading is a
    safe structural boundary.  Subsidiary-composition and qualified-topic
    queries retain their full block because those axes legitimately describe
    subsidiaries.
    """

    if _business_axis_kind(getattr(task, "retrieval_query", "")) not in {
            "business", "revenue", "products"}:
        return text_prompt_safe
    expected = _compact_heading(getattr(task, "expected_corp_name", ""))
    if not expected:
        return text_prompt_safe
    body = _strip_prompt_boundaries(text_prompt_safe)
    # A holding-company overview can introduce subsidiaries as ordinary
    # paragraphs without explicit [자회사] headings. Its opening sentences
    # separately identify the holding company and its management role. Keep
    # that proved issuer statement; ranking later subsidiary WM/IB prose as
    # the issuer's own main business would change the subject of the answer.
    opening_body = re.sub(r"^\s*[가-하]\.[ \t]+", "", body)
    opening = re.split(r"(?<=[.!?])\s+", opening_body, maxsplit=3)[:2]
    prefix = " ".join(opening).strip()
    compact_prefix = _compact_heading(prefix)
    if (len(opening) == 2 and expected in compact_prefix
            and any(word in compact_prefix for word in ("지주사", "지주회사"))
            and "경영관리" in compact_prefix
            and any(word in compact_prefix for word in ("종속기업", "자회사"))
            and len(prefix) <= 1200):
        return _wrap_prompt_safe(prefix)
    if _business_axis_kind(getattr(task, "retrieval_query", "")) == "products":
        return text_prompt_safe
    boundary = _AFFILIATE_BLOCK_HEADING.search(body)
    if boundary is None:
        return text_prompt_safe
    prefix = body[:boundary.start()].rstrip()
    if (expected not in _compact_heading(prefix[:1600])
            or not any(marker in _compact_heading(prefix)
                       for marker in _TOPIC_ACTIVITY_MARKERS + _REVENUE_MARKERS)):
        return text_prompt_safe
    return _wrap_prompt_safe(prefix)


def _strip_prompt_boundaries(text: str) -> str:
    """prompt-safe 본문에서 전송 경계선만 제거한다.

    masking된 내용과 ``[REMOVED:...]`` 표시는 그대로 유지한다. 경계선은 사용자 답변의
    데이터가 아니라 LLM 전송 프로토콜이므로 구조 분할 전에만 걷어낸다.
    """
    boundary = {PROMPT_DATA_BEGIN, PROMPT_DATA_END}
    return "\n".join(line for line in (text or "").splitlines()
                     if line.strip() not in boundary).strip()


def _wrap_prompt_safe(text: str) -> str:
    return f"{PROMPT_DATA_BEGIN}\n{text}\n{PROMPT_DATA_END}"


def _narrative_blocks(text_prompt_safe: str) -> list[tuple[str, bool]]:
    """검색 chunk를 온전한 행/문단 블록으로 제한해 prompt-safe 상태로 반환한다."""
    body = _strip_prompt_boundaries(text_prompt_safe)
    if not body:
        return []
    return [(_wrap_prompt_safe(part.text), part.over_budget)
            for part in split_section(body, NARRATIVE_BLOCK_CHARS) if part.text]


#: 이슈 #113 — 공시 서술지표(임상·기술이전 등)는 절 제목(``section_hint``)만으로
#: hit 을 고른다. ``NARRATIVE_BLOCK_CHARS``(8,000자) 예산 안에는 절 하나가
#: 통째로(원가표·개별 품목표·조직도·인원표가 뒤섞여) 한 조각으로 담길 수 있다.
#: 관련성은 원문이 빈 줄로 나눠놓은 자연스러운 단위(표 하나·문단 하나) 마다
#: 판정해야, 질문과 무관한 표·문단이 "같은 8,000자 조각 어딘가에 표기가
#: 있다"는 이유로 함께 실리지 않는다. 글자수 예산으로 다시 나누면(예:
#: 1,200자) 원문에 없던 지점에서 표가 잘려 표 여러 개가 섞이므로 쓰지 않는다
#: — 원문이 이미 그어 놓은 빈 줄 경계만 쓴다.
#: 「1) CT-P44」처럼 표 바로 앞에 오는 짧은 번호·이름표는 그 자체로는 표기를
#: 담지 않지만 다음 조각(표)의 캡션이다. 캡션 판정은 길이로만 가른다.
_DISCLOSED_METRIC_CAPTION_MAX_CHARS = 200


_DISCLOSED_METRIC_SENTENCE_END = re.compile(r"(?:다|요|음|함|임|됨)\.\s*$")


def _is_heading_only_piece(piece: str) -> bool:
    """빈 줄로 나뉜 조각이 제목만이고 값은 바로 다음 조각에 있는가.

    「6-1. 회사의 배당정책에 관한 사항」처럼 절 표제가 그 아래 본문 문단과
    빈 줄로만 나뉜 조각이 있다(#122 CG-043 실측). 짧고, 줄바꿈이 없고,
    한국어 문장 종결(「…다.」·「…요.」 등)로 끝나지 않으면 제목만 있는 조각으로
    본다 — 완결된 짧은 문장(값이 이미 그 조각 안에 있는 경우)은 문장 종결이
    있어 걸리지 않는다.
    """

    stripped = piece.strip()
    if not stripped or "\n" in stripped:
        return False
    if len(stripped) > _DISCLOSED_METRIC_CAPTION_MAX_CHARS:
        return False
    return _DISCLOSED_METRIC_SENTENCE_END.search(stripped) is None


def _is_disclosed_metric_table_piece(piece: str) -> bool:
    """빈 줄로 나뉜 한 조각이 표(행렬)를 담고 있는지.

    표 조각 대부분은 첫 줄부터 `|` 로 시작하지만, 「(단위 : 톤/월)」같은 단위
    안내 줄이 표 머리글 바로 앞 같은 조각 안에 붙어 있는 경우가 있다(에코프로
    비엠 생산능력 표). 그런 조각도 표로 본다 — 앞머리 몇 줄 안에 `|` 로 시작하는
    행이 있으면 된다.
    """

    lines = [line.strip() for line in piece.splitlines() if line.strip()]
    return any(line.startswith("|") for line in lines[:3])


def _is_table_metadata_piece(piece: str) -> bool:
    """A date/unit table introduces the following table; it is not its data."""

    cells = [cell.strip() for line in piece.splitlines()
             if line.strip().startswith("|")
             for cell in _table_cells(line)
             if cell.strip() and not re.fullmatch(r":?-+:?", cell.strip())]
    if not cells or not any("단위" in cell or "기준일" in cell for cell in cells):
        return False
    return all(
        "단위" in cell or "기준일" in cell
        or cell in {"(", ")", "당기", "전기", "(누적)", "누적"}
        or re.fullmatch(r"[()\s\d년월일./:-]+", cell) is not None
        for cell in cells)


def _disclosed_metric_relevant_blocks(
        text_prompt_safe: str, retrieval_query: str,
        ) -> list[tuple[str, bool]]:
    """질문이 가리킨 표기가 실제로 있는, 원문 그대로의 조각만 남긴다.

    `retrieval_query` 는 사전(`agent/disclosed_metric_topics.tsv`)이 인정한,
    질문 원문에 그대로 있는 표기다(예: 「임상 3상」) — 사전의 넓은 term(「임상」)
    보다 좁다. 그 표기(와 `_disclosed_metric_match_keys` 가 더한 괄호 동의어)가
    문자 그대로 없는 조각(원문의 빈 줄 경계 기준)은 답의 근거로 쓰지 않는다.

    0건이면 안전망으로 `_narrative_blocks` 의 굵은 단위(절 전체 조각)를 그대로
    돌려준다 — 이 필터가 답 전체를 지우는 일은 없어야 한다(질문 표기가 사전
    별칭뿐이고 본문은 다른 동의어로만 그 개념을 담은 경우 등).

    **표 하나만 더 되살리는 안전망은 일부러 두지 않는다.** 검색 chunk 경계는
    서로 겹칠 수 있어(#113), 표기가 어느 hit 의 서두 문단에 스쳐 지나가듯
    있을 뿐인데 그 hit 의 chunk 경계 안에 우연히 무관한 표(조직도·인원표)가
    함께 들어있는 경우가 실측으로 확인됐다(셀트리온 P9-011: 「임상 3상」이
    회사 개요 문단에 있고, 같은 chunk 안의 표는 연구개발 조직도). 매칭된
    조각 안에 표가 없으면 그 조각(대개 서두 문단 하나)만 조용히 남기고,
    hit 안의 다른 표까지 되찾지 않는다 — 표기가 실제로 딸고 있는 표만
    남긴다는 이 함수의 원래 계약을 지킨다.
    """
    coarse = _narrative_blocks(text_prompt_safe)
    keys = _disclosed_metric_match_keys(retrieval_query)
    if not keys:
        return coarse
    body = _strip_prompt_boundaries(text_prompt_safe)
    if not body:
        return coarse
    pieces = [piece for piece in body.split("\n\n") if piece.strip()]
    if len(pieces) < 2:
        return coarse
    groups: list[list[str]] = []
    pending: list[str] = []
    attach_next = False
    require_table = True
    for piece_index, piece in enumerate(pieces):
        is_table = _is_disclosed_metric_table_piece(piece)
        if (groups and piece_index > 0 and groups[-1][-1] == pieces[piece_index - 1]
                and _is_disclosed_metric_table_piece(pieces[piece_index - 1])
                and re.match(r"\s*주\s*\d*\)", piece)
                and any(key_present(_compact_heading(piece), key) for key in keys)):
            # A source-adjacent metric definition (e.g. K-ICS rather than
            # overseas RBC) belongs to the preceding table's coordinates.
            groups[-1].append(piece)
            attach_next = False
            continue
        if attach_next and (is_table or not require_table):
            # 「<라이선스-아웃 계약 총괄표>」처럼 표기가 있는 소개 문장 바로
            # 다음 조각이 그 표 자체인 경우 — 표 칸 안에는 소개 문장의 낱말이
            # 되풀이되지 않는다. 표기가 있는 조각 바로 뒤에 이어지는 표
            # 조각은 그 표기의 표이므로 함께 남긴다. 매칭된 조각이 제목만
            # 있으면(``require_table=False``, #122 CG-043 실측) 바로 다음
            # 조각은 표가 아니어도 함께 남긴다 — 제목 조각 자체는 값이 없다.
            groups[-1].append(piece)
            # Viewer date/unit headers are often tiny standalone tables.
            # Consuming the one-table allowance here drops the real rows
            # immediately after it (CG-044 / P9-014).
            attach_next = is_table and _is_table_metadata_piece(piece)
            require_table = True
            continue
        attach_next = False
        if any(key_present(_compact_heading(piece), key) for key in keys):
            groups.append([*pending, piece])
            pending = []
            attach_next = True
            require_table = is_table or not _is_heading_only_piece(piece)
        elif len(piece) <= _DISCLOSED_METRIC_CAPTION_MAX_CHARS:
            # 「1) CT-P44」처럼 표 앞에 오는 짧은 번호·이름표는 표기를 담지
            # 않지만 다음 조각(표)의 캡션이다.
            pending.append(piece)
        else:
            pending = []
    matched = ["\n\n".join(group) for group in groups]
    if not matched:
        return coarse
    blocks: list[tuple[str, bool]] = []
    for piece in matched:
        blocks.extend(
            (_wrap_prompt_safe(part.text), part.over_budget)
            for part in split_section(piece, NARRATIVE_BLOCK_CHARS)
            if part.text)
    return blocks or coarse


def _business_block_score(task, block: str) -> int:
    """Rank complete source blocks by the requested disclosure axis.

    Structural chunking may put a title-only paragraph before the actual table
    or operating paragraph.  This score only reorders already verified blocks;
    it neither rewrites their contents nor adds issuer/question vocabulary.
    """

    body = _strip_prompt_boundaries(block)
    compact = _compact_heading(body)
    kind = _business_axis_kind(getattr(task, "retrieval_query", ""))
    anchors = _business_axis_anchors(getattr(task, "retrieval_query", ""))
    table_rows = sum(
        1 for line in body.splitlines()
        if line.lstrip().startswith("|") and not _is_table_rule(_table_cells(line)))
    numeric = min(8, len(re.findall(r"(?<![A-Za-z가-힣])\d[\d,]*(?:\.\d+)?", body)))
    score = min(20, len(body) // 80) + min(24, table_rows * 3) + numeric
    if len(body) < 80 and table_rows < 2:
        score -= 80

    if kind == "revenue":
        hits = _count_markers(compact, _REVENUE_MARKERS)
        score += 18 * hits
        if table_rows >= 2 and hits:
            score += 45
        if "중단영업" in compact and hits <= 2:
            score -= 35
    elif kind == "products":
        product_markers = (
            "주요제품", "제품및서비스", "제품", "상품", "서비스", "품목", "용도")
        hits = _count_markers(compact, product_markers)
        score += 14 * hits
        if table_rows >= 2 and hits:
            score += 45
        score += 4 * _count_markers(compact, _TOPIC_ACTIVITY_MARKERS)
    elif kind == "topic":
        anchor_hits = sum(_compact_heading(anchor) in compact for anchor in anchors)
        score += 36 * anchor_hits
        if _topic_has_local_activity(compact, tuple(
                _compact_heading(anchor) for anchor in anchors)):
            score += 55
        score += 3 * _count_markers(compact, _TOPIC_ACTIVITY_MARKERS)
    elif kind == "subsidiaries":
        score += 20 * _count_markers(compact, _SUBSIDIARY_STRUCTURE_MARKERS)
    else:
        score += 5 * _count_markers(compact, _TOPIC_ACTIVITY_MARKERS)
        score += 6 * _count_markers(compact, _GROUP_SCOPE_MARKERS)
    return score


def _ranked_business_blocks(
        text_prompt_safe: str, task,
        ) -> list[tuple[str, bool]]:
    """Return structurally complete blocks with the strongest axis first."""

    blocks = _narrative_blocks(text_prompt_safe)
    return sorted(
        blocks,
        key=lambda row: _business_block_score(task, row[0]),
        reverse=True,
    )


def _roundtripped_chunk_text(hit, section) -> str | None:
    """Return the hit text only when it is contained in the read section.

    The index is allowed to decide *where to look*, never to provide the text
    eventually attributed to ``source_roundtrip``.  Both sources are already
    prompt-safe projections, so an exact body match is also a cheap integrity
    check that does not expose raw text or attempt fuzzy reconstruction.
    """

    if (getattr(section, "section_id", None) != hit.section_id
            or getattr(section, "doc_id", None) != hit.doc_id
            or getattr(section, "text_projection", "prompt_safe") != "prompt_safe"):
        return None
    hit_text = getattr(hit, "text_prompt_safe", "")
    section_text = getattr(section, "text", "")
    if (hit_text.count(PROMPT_DATA_BEGIN), hit_text.count(PROMPT_DATA_END)) != (1, 1):
        return None
    if (section_text.count(PROMPT_DATA_BEGIN), section_text.count(PROMPT_DATA_END)) != (1, 1):
        return None
    hit_body = _strip_prompt_boundaries(hit_text)
    section_body = _strip_prompt_boundaries(section_text)
    if not hit_body or not section_body:
        return None
    # Canonical chunks may repeat a table header at the start of a later part.
    # Such a safe chunk is not a literal substring of its section, but it is a
    # deterministic structural projection of it.  Rebuild the section's
    # canonical chunks and use that rebuilt text, never the index payload.
    for part in split_section(section_body):
        if part.text == hit_body:
            return _wrap_prompt_safe(part.text)
    start = section_body.find(hit_body)
    if start < 0:
        return None
    return _wrap_prompt_safe(section_body[start:start + len(hit_body)])


#: 기수 표기 바로 앞의 연도. 원문이 이미 「2026년 1분기(제16기)」처럼 둘을
#: 나란히 적은 자리다 — 그때는 바꿀 것이 없고 괄호만 걷어내면 된다.
_YEAR_BEFORE_PERIOD_LABEL = re.compile(
    r"(?P<year>(?:19|20)[0-9]{2}\s*년[^|(]{0,12}?)\s*\(\s*"
    r"(?P<label>제\s*[0-9]{1,3}\s*기[^)]{0,10})\s*\)")

#: 연도 다음에 (괄호 없이) 곧장 이어지는 기수 — 「2025년 제21기 (당기)」.
#: 연도가 이미 앞에 있으므로 기수는 같은 말을 두 번 하는 것이다. 갭(공백)까지
#: 함께 지워야 「2025년  (당기)」처럼 빈 공백이 남지 않는다.
_YEAR_THEN_BARE_LABEL = re.compile(
    r"(?P<year>(?:19|20)[0-9]{2}\s*년)(?P<gap>\s+)"
    r"(?P<label>제\s*[0-9]{1,3}\s*기"
    r"(?:\s*(?:말|반기|상반기|[1-4]\s*분기(?:\s*말)?|[1-4]\s*/4\s*분기))?)")

#: 기수 다음에 곧장 이어지는(공백 없이 붙어 쓴 것도 포함) 연도 —
#: 「제53기2026년 1분기」. 갭은 그대로 두고 기수만 지운다.
_BARE_LABEL_THEN_YEAR = re.compile(
    r"(?P<label>제\s*[0-9]{1,3}\s*기"
    r"(?:\s*(?:말|반기|상반기|[1-4]\s*분기(?:\s*말)?|[1-4]\s*/4\s*분기))?)"
    r"(?P<gap>\s*)(?P<year>(?:19|20)[0-9]{2}\s*년)")

#: 기수 뒤에 붙는 시점 수식. 「제55기 1분기」와 「제55기」는 다른 시점이다.
_PERIOD_LABEL_SUFFIX = re.compile(
    r"제\s*[0-9]{1,3}\s*기\s*(?P<suffix>.*)$")


def _period_years(rm, doc_id: str) -> "Mapping[str, str]":
    """그 문서의 기수↔연도 짝. 정본이 못 주면 빈 dict.

    스키마 1.5 이전 산출물에는 `period_label` 열 자체가 없다. 그때는 기수를
    그대로 두는 것이 맞다 — 없는 근거로 연도를 붙이지 않는다.
    """

    lookup = getattr(rm, "period_label_years", None)
    if not callable(lookup):
        return {}
    try:
        return lookup(doc_id) or {}
    except Exception:
        return {}


def _year_of_period_number(number: str, years: "Mapping[str, str]") -> "str | None":
    """기수 번호가 가리키는 해. 표에 없으면 **아는 기수에서 셈한다**.

    기수는 정의상 회계연도마다 1씩 는다. 그래서 「제54기 = 2022년」을 알면
    「제53기 = 2021년」이다 — 짐작이 아니라 셈이다.

    셈이 필요한 이유는 실측에 있다. 2023년 1분기 보고서의 fact 는 제54·55기만
    싣는데, 그 보고서의 매출 표는 제53기까지 세 해를 나란히 보여 준다.

    **멀면 셈하지 않는다.** 정본에 근거가 있는 기수에서 10기 넘게 떨어지면
    같은 회사의 같은 연속열이라고 볼 근거가 약하다.
    """

    known = years.get(number)
    if known:
        return known
    matched = re.fullmatch(r"제([0-9]{1,3})기", number)
    if matched is None:
        return None
    target = int(matched.group(1))
    best: tuple[int, str] | None = None
    for anchor, year in years.items():
        anchor_match = re.fullmatch(r"제([0-9]{1,3})기", anchor)
        if anchor_match is None or not year.isdigit():
            continue
        distance = abs(int(anchor_match.group(1)) - target)
        if distance > 10:
            continue
        if best is None or distance < best[0]:
            best = (distance, str(int(year) - (int(anchor_match.group(1)) - target)))
    return best[1] if best else None


def _period_label_surface(label_key: str, year: str) -> str:
    """기수 하나를 사람이 읽는 기간 표면으로. 「제55기 1분기」 → 「2023년 1분기」."""

    matched = _PERIOD_LABEL_SUFFIX.match(label_key)
    suffix = (matched.group("suffix") if matched else "").strip()
    if not suffix:
        return f"{year}년"
    if suffix == "말":
        return f"{year}년 말"
    # 「1분기」·「1분기말」·「반기」 — 원문 수식을 그대로 잇는다.
    return f"{year}년 {suffix}"


def years_for_period_labels(text: str, years: "Mapping[str, str]") -> str:
    """공시 원문의 회계 기수를 실제 연도로 바꾼다 — 이슈 #94 26.

    ```
    | 부 문 | 품 목 | 제55기 1분기 | 제54기 | 제53기 |
    | 부 문 | 품 목 | 2023년 1분기 | 2022년 | 2021년 |
    ```

    원문이 이미 둘을 나란히 적었으면(「2026년 1분기(제16기)」) 괄호만
    걷어낸다 — 같은 말을 두 번 하는 것이라 연도만 남기면 된다. 괄호 없이
    나란히 적은 경우(「2025년 제21기 (당기)」·「제53기2026년 1분기」)도
    마찬가지다 — 라벨에 이미 4자리 연도가 있으면 기수 토큰을 **삭제만** 하고
    연도를 겹쳐 쓰지 않는다.

    ``years`` 는 **그 문서의** 짝이며 기수 **번호**로 찾는다
    (`CanonicalReadModel.period_label_years`). 문서를 건너 쓰면 안 된다 —
    「제50기」는 회사마다 다른 해다. 번호가 회계연도를 정하고, 뒤의 수식
    (말·1분기)은 그 안의 어느 구간인지를 표면에만 더한다.

    **짝을 모르는 기수는 그대로 둔다.** 연도를 지어내면 원문에 없는 사실을
    말하는 것이 된다. 그때는 읽기 불편할 뿐 틀리지는 않는다.
    """

    if not text or not years:
        return text

    def _paired(match: "re.Match[str]") -> str:
        # 연도가 이미 앞에 있다. 괄호 안 기수가 그 연도와 짝이면 지운다.
        # 연도가 이미 앞에 있으므로 괄호 안 기수는 같은 말을 두 번 하는 것이다.
        # 짝을 알든 모르든 걷어내도 잃을 것이 없다.
        if period_label_number(match.group("label")):
            return match.group("year").rstrip()
        return match.group(0)

    body = _YEAR_BEFORE_PERIOD_LABEL.sub(_paired, text)

    # 괄호 없이 연도와 나란히 적은 기수도 같은 말을 두 번 하는 것이다 — 삭제만
    # 하고 연도는 겹쳐 쓰지 않는다. 갭(공백)은 연도가 앞에 있는 쪽 것만 함께
    # 지운다 — 「2025년 제21기 (당기)」의 뒤 괄호 앞 공백은 원문 몫이라 둔다.
    body = _YEAR_THEN_BARE_LABEL.sub(lambda m: m.group("year"), body)
    body = _BARE_LABEL_THEN_YEAR.sub(
        lambda m: m.group("gap") + m.group("year"), body)

    def _bare(match: "re.Match[str]") -> str:
        number = period_label_number(match.group(0))
        year = _year_of_period_number(number, years) if number else None
        if not year:
            return match.group(0)
        return _period_label_surface(period_label_key(match.group(0)), year)

    return PERIOD_LABEL.sub(_bare, body)


def _citation_excerpt_terms(task) -> tuple[str, ...]:
    """Return axis words used only to choose a cited structural part."""
    query = getattr(task, "retrieval_query", "")
    disclosed_metrics = disclosed_metrics_for_query(query)
    if disclosed_metrics:
        # 공시 서술지표: 값이 실린 표 행·문장이 있는 part 를 인용한다.
        return tuple(surface for entry in disclosed_metrics for surface in entry.surfaces)
    kind = _business_axis_kind(query)
    if kind == "revenue":
        return _REVENUE_MARKERS
    if kind == "subsidiaries":
        return _SUBSIDIARY_STRUCTURE_MARKERS + (
            "사업부문", "계열사", "상장회사", "비상장회사")
    if kind == "topic":
        return _business_axis_anchors(query)
    return ()


def _safe_excerpt(
        prompt_safe_block: str, budget: int = 400,
        *, prefer_terms: tuple[str, ...] = (),
        ) -> str:
    """경계가 닫힌 citation 발췌. 표 행/문단은 중간에서 자르지 않는다."""
    body = _strip_prompt_boundaries(prompt_safe_block)
    if not body:
        return _wrap_prompt_safe("")
    parts = split_section(body, budget)
    # 표는 첫 part가 헤더뿐일 수 있으므로 기존처럼 가장 큰 완전 행을 쓴다.
    # 일반 서술은 반대로 가장 큰 part를 고르면 뒤쪽 자회사 문단이 전사 개요를
    # 대체할 수 있다. 이때는 문서 순서를 보존해 첫 실질 문단을 인용한다.
    if body.lstrip().startswith("|"):
        part = max(parts, key=lambda row: len(row.text))
    else:
        preferred = next((
            row for row in parts
            if any(_compact_heading(term) in _compact_heading(row.text)
                    for term in prefer_terms)
        ), None)
        part = preferred or next(
            (row for row in parts if len(row.text.strip()) >= 80), parts[0])
    return _wrap_prompt_safe(part.text)


def _form_of(rm, doc_id: str, form: str) -> bool:
    rm._load_docs()
    meta = rm._documents.get(doc_id)
    if meta is None:
        return True
    return form in ((getattr(meta, "report_nm", "") or "") + (getattr(meta, "form", "") or ""))


def _plus_months(yyyymmdd: str, months: int) -> str:
    y, m, d = int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:])
    m2 = m + months
    y += (m2 - 1) // 12
    m2 = (m2 - 1) % 12 + 1
    return f"{y:04d}{m2:02d}{min(d, 28):02d}"
