"""「최근 N년」「최근 N개년」「최근 N개 분기」다중 기간을 되묻지 않고 **전부 답한다**.

이슈 #58 2단계. 1단계(`agent/relative_financial_period.py`)는 「직전 분기」
「전전 분기」「전년」처럼 표면형 하나를 다른 표면형 하나로 재정렬(rewrite)했다
— 실제 연도·분기는 그대로 정본 백엔드가 정한다.  「최근 3년」은 다르다. 답이
**여러 개**(2023·2024·2025년 매출액)이고, 그 실제 연도는 질문에 리터럴로
없다. `validate_semantic_intent_grounding`(agent/stage1_v1_resolver.py)은
``target_period_expressions`` 원소가 질문의 리터럴 부분문자열이기를 요구하므로
실제 연도를 목록으로 펼칠 수 없다.

`agent.summary_metric_fanout` 이 「얼마나 벌었어」(표면 하나가 후보 개념
여럿을 가리킴) 를 여는 것과 같은 두 자리 구조를 쓴다 — 축만 개념에서
기간으로 바뀐다.

``RecentPeriodsFanoutRegrounder``
    item 하나를 **N 개**로 복제한다. 표면은 전부 「최근 3년」 그대로다 —
    실제 연도를 표면에 적으면 질문에 없는 낱말이라 결속 검사가 거절한다.

``agent.stage1_v1_financial_backend``
    같은 표면 item N 개를 보고 **형제 순서로** 실제 연도·분기를 나눠
    맡긴다. 직접 공시된 분기는 ``financial``로, FY와 9M 누계로만 구할 수
    있는 4분기는 ``financial_comparison(discrete_from_cumulative)``으로
    내린다. 순서는 항상 오름차순(과거→최근)이다.

**전부 풀릴 때만 편다.** `SummaryMetricFanoutRegrounder` 와 같은 가드다 —
N 개 기간 중 하나라도 정본에 좌표가 없으면 통째로 물러난다. 회사·기간
축이 비어 되묻는 오늘의 동작이 조용히 「답 없음」으로 바뀌는 것이 유일하게
나빠지는 길이라서다.

**컴파일러 쪽 주의.** 이 fanout item 의 모양(회사 하나·재무 스칼라 하나·
같은 표면 반복)은 `SummaryMetricFanoutRegrounder`가 만드는 개념 fanout과
구조적으로 구분이 안 된다 — 둘 다 사람이 볼 때는 "같은 표면을 N번 반복한
scalar retrieve 항목들"이다. `agent.deterministic_plan_compiler_v1`의
handler 선택(`_selected_handler`)은 오직 **구조 서명**(qid·리터럴이 빠진
enum/모양/개수)과 resolution.kind 목록만으로 라우팅하므로, 두 fanout이
정말로 같은 서명을 등록하면 `len(matches) > 1`로 컴파일이 거절된다(직접
확인함). 그래서 이 fanout은 최상위 ``intent.presentation`` 을
``"table"``로 못박아 개념 fanout(``"auto"``)과 서명을 가른다 — 스키마
(``agent/schemas/*.schema.json``)는 건드리지 않는다. ``Presentation`` 은
이미 ``Literal["auto", "prose", "table", "list"]`` 이고 「추이」를 표로
보여 달라는 요청과도 뜻이 맞는다.

「최근 N년 평균」처럼 집계를 요구하는 질문은 범위 밖이다 — 이 regrounder는
``operation == "retrieve"`` 인 닫힌 scalar 모양만 펴고, 매치 리터럴 바로
뒤에 「평균」「합계」「누계」가 붙으면 애초에 매치하지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from .planning import resolve_metric_concept
from .semantic_intent_v1 import AnswerItem, SemanticIntent
from .summary_metric_fanout import _sibling_shape

__all__ = [
    "FanoutPeriodAssignment",
    "RecentPeriodsFanout",
    "RecentPeriodsFanoutRegrounder",
    "fanout_period_assignments_by_sibling",
    "fanout_periods_by_sibling",
    "recent_periods_fanout",
]

#: 「최근 3년」「최근 3년간」「최근 3개년」의 핵심부(「간」 앞까지). N 은
#: 2~5(#58 2단계 지원 범위).
_RECENT_YEARS_CORE = re.compile(r"최근\s*([2-5])\s*개?\s*년")
#: 「최근 4개 분기」「최근 4개분기」의 핵심부.
_RECENT_QUARTERS_CORE = re.compile(r"최근\s*([2-5])\s*개\s*분기")
#: 핵심부 바로 뒤에 붙는 「간」(「최근 3년**간**」「최근 4개 분기**간**」).
_TRAILING_GAN = re.compile(r"간")
#: 그 뒤에 「평균/합계/누계」가 붙으면(집계 요구) 매치하지 않는다 — 그런
#: 질문은 이 fanout의 범위 밖이라 `unsupported_operator` 로 남아야 한다.
#: (핵심부만 보고 뒤를 정규식 하나로 묶으면 옵션 「간」을 건너뛰는
#: 되추적으로 이 lookahead를 피해 갈 수 있어— 「최근 3년간 합계」가
#: 「간」을 안 먹은 「최근 3년」으로 매치해 버린다 — 핵심부·「간」·집계어를
#: 각각 순서대로 확인한다.)
_FORBIDDEN_AGGREGATE = re.compile(r"\s*(?:평균|합계|누계)")


def _match_recent_period_span(
        question: str, core: "re.Pattern[str]",
        ) -> "tuple[str, int] | None":
    """``core`` 로 찾은 핵심부에 「간」을 이어 붙이고 집계어를 거른다."""

    match = core.search(question)
    if match is None:
        return None
    end = match.end()
    gan = _TRAILING_GAN.match(question, end)
    if gan is not None:
        end = gan.end()
    if _FORBIDDEN_AGGREGATE.match(question, end):
        return None
    return question[match.start():end], int(match.group(1))


@dataclass(frozen=True, slots=True)
class RecentPeriodsFanout:
    """펼 수 있는 상대 기간 리터럴 하나와 그 축·개수."""

    literal: str
    unit: str  # "year" | "quarter"
    count: int
    #: 질문에 연도가 각각 적힌 경우에는 최신 기간을 추정하지 않고 그 리터럴을
    #: 형제 순서대로 쓴다. 빈 tuple은 기존 「최근 N년」 경로다.
    explicit_literals: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FanoutPeriodAssignment:
    """One sibling's exact period and how Stage1 must obtain its value."""

    literal: str
    mode: str  # "direct" | "derived_discrete"


_EXPLICIT_YEAR = re.compile(r"(?<![0-9])(20[0-9]{2})\s*(?:년|회계연도)")
_EXPLICIT_YEAR_FANOUT_CUE = re.compile(r"각각|추이|연도별")


def _explicit_year_literals(question: str) -> "tuple[str, ...] | None":
    """질문에 직접 적힌 2~5개 사업연도를 순서대로 반환한다.

    단순히 날짜가 여러 개라는 이유로 독립 조회를 만들지 않는다. 「각각」,
    「추이」, 「연도별」처럼 모든 연도 값을 요구한 경우에만 열어, 기준일이나
    비교 대상 날짜가 함께 있는 질문을 보수적으로 제외한다.
    """

    if _EXPLICIT_YEAR_FANOUT_CUE.search(question) is None:
        return None
    years = [match.group(1) for match in _EXPLICIT_YEAR.finditer(question)]
    if not 2 <= len(years) <= 5 or len(set(years)) != len(years):
        return None
    return tuple(f"{year}년" for year in years)


#: 실호출(HCX-007)로 직접 확인함 — 같은 「최근 N…」류 질문도 판마다 output
#: shape가 갈린다(「최근 3개년 …」→ ``scalar``, 「… 추이」→ ``timeline``).
#: `agent.summary_metric_fanout._fanout_shaped_item` 은 ``scalar`` 만 받지만
#: 이 fanout은 둘 다 받는다 — 값을 지어내지 않듯 shape도 지어내지 않고
#: HCX가 고른 대로 N개 형제에 그대로 반복한다(컴파일러 registry도 그
#: 둘을 각각 등록해 받는다).
_FANOUT_SHAPES = frozenset({"scalar", "timeline"})


def _fanout_shaped_item(item: Any) -> bool:
    """이 item 이 **값 하나를 묻는 닫힌 재무 모양**인가."""

    return (
        item.target.kind == "metric"
        and item.operation == "retrieve"
        and item.selection is None
        and item.output.shape in _FANOUT_SHAPES
        and item.output.projection_mode == "named_fields"
        and len(item.output.field_surfaces) == 1
    )


def recent_periods_fanout(question: str, item: Any) -> "RecentPeriodsFanout | None":
    """이 item 을 「최근 N년/분기」로 펼 수 있는가. 아니면 ``None``.

    item 이 이미 다른 기간을 갖고 있으면(리터럴과 다르면) 손대지 않는다 —
    `RelativeFinancialPeriodRegrounder` 와 같은 fail-closed 규칙이다.
    """

    if not isinstance(question, str) or not question.strip():
        return None
    if not _fanout_shaped_item(item):
        return None
    found = _match_recent_period_span(question, _RECENT_QUARTERS_CORE)
    unit = "quarter"
    if found is None:
        found = _match_recent_period_span(question, _RECENT_YEARS_CORE)
        unit = "year"
    if found is None:
        explicit = _explicit_year_literals(question)
        if explicit is None:
            return None
        return RecentPeriodsFanout(
            literal=explicit[0], unit="year", count=len(explicit),
            explicit_literals=explicit)
    literal, count = found
    periods = list(item.scope.target_period_expressions or ())
    if periods and periods != [literal] and not all(
            _BARE_YEAR_OR_RANGE.fullmatch(value.strip()) for value in periods):
        # HCX-007 실호출로 직접 확인함(SG-012) — 질문에 괄호로 명시된
        # 연도범위(「(2023~2025년)」)가 있으면 그 리터럴 그대로를
        # ``target_period_expressions`` 에 이미 담아 보낸다(「최근 3년간」은
        # ``qualifier_surfaces`` 로 간다).  그 값은 이 fanout이 스스로
        # 계산할 결과와 갈리는 **다른 기간이 아니라 같은 요청의 다른
        # 표현**이므로 편다 — bare 연도/범위 리터럴이 아닌 다른 무언가가
        # 이미 있을 때만(예: 「전년」 같은 별개 상대 표현) 손을 뗀다.
        return None
    return RecentPeriodsFanout(literal=literal, unit=unit, count=count)


#: 「2023~2025년」「2023-2025」「2025년」처럼 bare 연도/연도범위 리터럴만.
#: 위 ``target_period_expressions`` 완화 가드와 아래 qualifier 제거에서
#: "이 fanout이 스스로 다시 계산할 값과 같은 축"인지 가리는 데 쓴다.
_BARE_YEAR_OR_RANGE = re.compile(
    r"^(?:19|20)\d{2}(?:\s*[~\-]\s*(?:19|20)\d{2})?\s*년?$")
#: qualifier_surfaces 원소가 「최근」/개수/bare 연도범위 **그 자체뿐**이면
#: 기간 표현이 잘못 얹힌 것으로 보고 제거한다(SK하이닉스 실호출 —
#: qualifier_surfaces가 ``["최근", "4개 분기"]`` 로 쪼개져 온다).
_BARE_PERIOD_QUALIFIER = re.compile(
    r"^(?:최근|[2-5]\s*개?\s*년(?:간)?|[2-5]\s*개\s*분기)$")


def _strip_period_qualifiers(
        qualifiers: "list[str]", literal: str) -> "list[str]":
    """target.qualifier_surfaces 에서 기간 관련 조각만 뺀다.

    실제 개념 수식어(예: 「연결」)는 그대로 둔다 — 편 형제 item의
    구조 서명 qualifier_count가 정본이 등록한 값(레지스트리의
    ``scope_qualifier_count`` 축과는 다른, target 쪽 축)과 어긋나지 않도록
    기간 흔적만 걷어낸다.
    """

    literal_key = literal.strip()
    out = []
    for value in qualifiers:
        stripped = value.strip()
        if stripped == literal_key:
            continue
        if (_BARE_PERIOD_QUALIFIER.fullmatch(stripped) is not None
                or _BARE_YEAR_OR_RANGE.fullmatch(stripped) is not None):
            continue
        out.append(value)
    return out


def _strip_recent_period_prefix(surface: str) -> str:
    """target.surface 맨 앞의 「최근 N…」조각을 뗀다.

    HCX-007 실호출로 직접 확인함(삼성전자 「최근 3개년 연결 영업이익」) —
    개념 사전이 모르는 표면을 만들며 「최근 3개년」을 표면 자체에 접어
    넣을 수 있다. `_resolved_concept` 가 이것을 뗀 나머지로 재시도한다.
    """

    text = surface.strip()
    for core in (_RECENT_QUARTERS_CORE, _RECENT_YEARS_CORE):
        match = core.match(text)
        if match is None:
            continue
        end = match.end()
        gan = _TRAILING_GAN.match(text, end)
        if gan is not None:
            end = gan.end()
        return text[end:].strip()
    return re.sub(r"^최근\s*", "", text)


def _resolved_concept(item: Any) -> "Any | None":
    from .stage1_v1_financial_backend import _bare_concept_surface

    surfaces = [item.target.surface, *item.output.field_surfaces]
    for surface in surfaces:
        candidates = [surface, _bare_concept_surface(surface)]
        stripped = _strip_recent_period_prefix(surface)
        if stripped and stripped != surface:
            candidates.extend((stripped, _bare_concept_surface(stripped)))
        # HCX may place the period in target.surface and the metric in the
        # named output field (``최근 3개 분기`` / ``연결 매출액 추이``).
        # ``추이`` is a presentation cue, not part of the accounting concept;
        # removing this literal suffix leaves a question-grounded substring.
        candidates.extend(
            re.sub(r"\s*추이\s*$", "", candidate).strip()
            for candidate in tuple(candidates))
        for candidate in candidates:
            if not candidate:
                continue
            concept = resolve_metric_concept(candidate)
            if concept is None:
                concept = resolve_metric_concept(
                    _bare_concept_surface(candidate))
            if concept is not None:
                return concept
    return None


def _actual_annual_years(
        companies: Any, *, corp_code: str, concept: Any, corpus_cutoff: str,
        ) -> "set[int]":
    """정본에 실제 annual 사실(12-31 종료)이 있는 연도 전부."""

    import re

    if not callable(getattr(companies, "facts", None)):
        return set()
    return {
        int(row.period_end[:4])
        for row in companies.facts(
            corp_code, as_of=corpus_cutoff, concept=concept.value)
        if getattr(row, "period_type", None) == "annual"
        and isinstance(getattr(row, "period_end", None), str)
        and re.fullmatch(r"[0-9]{4}-12-31", row.period_end)
        and row.period_end.replace("-", "") <= corpus_cutoff
    }


def _answerable_quarter_ends(
        companies: Any, *, corp_code: str, concept: Any, corpus_cutoff: str,
        ) -> "dict[tuple[int, int], str]":
    """Return directly reported quarters plus Q4s provable as FY minus 9M.

    A recent-quarter series asks for discrete quarters.  Q1 is equal to its
    cumulative column; Q2/Q3 must have a direct non-cumulative row.  Q4 has no
    separate DART filing, so it is answerable only when annual and 9M
    cumulative endpoints exist on the same scope/statement axis.  The exact
    requested scope is checked again when the regrounder binds coordinates.
    """

    import re

    from .stage1_v1_financial_backend import _QUARTER_ENDPOINTS

    if not callable(getattr(companies, "facts", None)):
        return {}
    rows = list(companies.facts(
        corp_code, as_of=corpus_cutoff, concept=concept.value))
    out: dict[tuple[int, int], str] = {}
    for row in rows:
        period_end = getattr(row, "period_end", None)
        if (getattr(row, "period_type", None) == "quarter"
                and isinstance(period_end, str)
                and period_end[5:] in _QUARTER_ENDPOINTS
                and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", period_end)
                and period_end.replace("-", "") <= corpus_cutoff):
            quarter = _QUARTER_ENDPOINTS[period_end[5:]]
            if quarter == 1 or getattr(row, "cumulative", None) is False:
                out[(int(period_end[:4]), quarter)] = "direct"

    annual_axes = {
        (int(row.period_end[:4]), getattr(row, "scope", None),
         getattr(row, "statement", None))
        for row in rows
        if getattr(row, "period_type", None) == "annual"
        and getattr(row, "cumulative", None) is True
        and isinstance(getattr(row, "period_start", None), str)
        and row.period_start == f"{row.period_end[:4]}-01-01"
        and isinstance(getattr(row, "period_end", None), str)
        and re.fullmatch(r"[0-9]{4}-12-31", row.period_end)
        and row.period_end.replace("-", "") <= corpus_cutoff
    }
    nine_month_axes = {
        (int(row.period_end[:4]), getattr(row, "scope", None),
         getattr(row, "statement", None))
        for row in rows
        if getattr(row, "period_type", None) == "quarter"
        and getattr(row, "cumulative", None) is True
        and isinstance(getattr(row, "period_start", None), str)
        and row.period_start == f"{row.period_end[:4]}-01-01"
        and isinstance(getattr(row, "period_end", None), str)
        and re.fullmatch(r"[0-9]{4}-09-30", row.period_end)
        and row.period_end.replace("-", "") <= corpus_cutoff
    }
    for year, _scope, _statement in annual_axes & nine_month_axes:
        out.setdefault((year, 4), "derived_discrete")
    return out


def _recent_period_assignments(
        fanout: RecentPeriodsFanout, *, companies: Any, corp_code: str,
        concept: Any, corpus_cutoff: str,
        ) -> "tuple[FanoutPeriodAssignment, ...] | None":
    """Return ``fanout.count`` exact periods in ascending order.

    Annual periods and directly reported quarters use ``direct``.  A Q4 is
    marked ``derived_discrete`` only when the canonical FY and 9M endpoints
    prove the subtraction.  One missing period closes the whole fanout.
    """

    from .stage1_v1_financial_backend import (
        _latest_actual_annual_year, _offset_quarter)

    if fanout.explicit_literals:
        actual = _actual_annual_years(
            companies, corp_code=corp_code, concept=concept,
            corpus_cutoff=corpus_cutoff)
        if not all(int(value[:4]) in actual for value in fanout.explicit_literals):
            return None
        return tuple(FanoutPeriodAssignment(value, "direct")
                     for value in fanout.explicit_literals)
    if fanout.unit == "year":
        latest_year = _latest_actual_annual_year(
            companies, corp_code=corp_code, concept=concept,
            corpus_cutoff=corpus_cutoff)
        if latest_year is None:
            return None
        years = [latest_year - offset for offset in range(fanout.count)]
        years.reverse()
        actual = _actual_annual_years(
            companies, corp_code=corp_code, concept=concept,
            corpus_cutoff=corpus_cutoff)
        if not all(year in actual for year in years):
            return None
        return tuple(FanoutPeriodAssignment(f"{year}년", "direct")
                     for year in years)
    answerable = _answerable_quarter_ends(
        companies, corp_code=corp_code, concept=concept,
        corpus_cutoff=corpus_cutoff)
    if not answerable:
        return None
    latest = max(answerable)
    quarters = [
        _offset_quarter(*latest, back=offset) for offset in range(fanout.count)]
    quarters.reverse()
    if not all(quarter in answerable for quarter in quarters):
        return None
    return tuple(
        FanoutPeriodAssignment(
            f"{year}년{quarter}분기", answerable[(year, quarter)])
        for year, quarter in quarters)


def _recent_period_literals(
        fanout: RecentPeriodsFanout, *, companies: Any, corp_code: str,
        concept: Any, corpus_cutoff: str,
        ) -> "tuple[str, ...] | None":
    """Compatibility projection used by tests and audit callers."""

    assignments = _recent_period_assignments(
        fanout, companies=companies, corp_code=corp_code, concept=concept,
        corpus_cutoff=corpus_cutoff)
    if assignments is None:
        return None
    return tuple(row.literal for row in assignments)


def fanout_period_assignments_by_sibling(
        question: str, intent: SemanticIntent, *, companies: Any,
        corpus_cutoff: str,
        ) -> "tuple[FanoutPeriodAssignment | None, ...]":
    """Return each sibling's exact period and direct/derived execution mode.

    `RecentPeriodsFanoutRegrounder` 가 편 intent 를 재무 백엔드가 다시 알아보는
    자리다(`agent.summary_metric_fanout.fanout_concepts_by_sibling` 과 같은
    역할, 축만 개념→기간). 표면·기간 리터럴이 전부 같으므로 순서 말고는
    구분할 것이 없다.

    형제가 **완전히 같은 모양**일 때만 답한다.
    """

    items = list(intent.answer_items)
    empty = (None,) * len(items)
    if (len(items) < 2 or intent.answer_groups or intent.premises
            or intent.unresolved_mentions):
        return empty
    fanout = recent_periods_fanout(question, items[0])
    if fanout is None or fanout.count != len(items):
        return empty
    first = _sibling_shape(items[0])
    if any(_sibling_shape(row) != first for row in items[1:]):
        return empty
    by_id = {row.entity_id: row for row in intent.entities}
    companies_rows = []
    for ref in items[0].target.entity_refs:
        entity = by_id.get(ref)
        if entity is None or entity.kind_hint != "company":
            continue
        rows = list(companies.resolve_company(entity.surface) or ())
        if len(rows) == 1:
            companies_rows.append(rows[0])
    if len(companies_rows) != 1:
        return empty
    concept = _resolved_concept(items[0])
    if concept is None:
        return empty
    assignments = _recent_period_assignments(
        fanout, companies=companies, corp_code=companies_rows[0].corp_code,
        concept=concept, corpus_cutoff=corpus_cutoff)
    if assignments is None or len(assignments) != len(items):
        return empty
    return assignments


def fanout_periods_by_sibling(
        question: str, intent: SemanticIntent, *, companies: Any,
        corpus_cutoff: str,
        ) -> "tuple[str | None, ...]":
    """Compatibility view returning only exact period literals."""

    return tuple(
        row.literal if row is not None else None
        for row in fanout_period_assignments_by_sibling(
            question, intent, companies=companies,
            corpus_cutoff=corpus_cutoff))


class RecentPeriodsFanoutRegrounder:
    """되묻던「최근 N년」「최근 N개 분기」item 하나를 N개로 편다.

    Fail-closed: item 이 하나가 아니거나, 회사가 둘 이상이거나, 개념을 못
    풀거나, N 개 기간 중 **하나라도** 좌표를 세우지 못하면 intent 를 그대로
    돌려준다 — 그러면 종전대로 재무 백엔드가 단일 기간 표현을 못 풀어
    되묻거나 거절한다.
    """

    def __init__(
            self, canonical: Any, *, reference_date: date, corpus_cutoff: str,
            scope_authority: Any = None,
            ) -> None:
        if not callable(getattr(canonical, "resolve_company", None)):
            raise TypeError(
                "recent periods fanout regrounder에는 company resolver가 "
                "필요합니다")
        if not isinstance(reference_date, date):
            raise TypeError(
                "recent periods fanout regrounder에는 reference_date가 "
                "필요합니다")
        if not isinstance(corpus_cutoff, str) or not corpus_cutoff:
            raise TypeError(
                "recent periods fanout regrounder에는 corpus_cutoff가 "
                "필요합니다")
        self.canonical = canonical
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff
        self.scope_authority = (
            canonical if scope_authority is None else scope_authority)

    @staticmethod
    def _closed_single_item(intent: SemanticIntent) -> "Any | None":
        if (len(intent.answer_items) != 1 or intent.answer_groups
                or intent.premises or intent.unresolved_mentions):
            return None
        item = intent.answer_items[0]
        if not _fanout_shaped_item(item):
            return None
        return item

    @staticmethod
    def _quarter_listing_item(question: str, intent: SemanticIntent) -> "Any | None":
        """Repair the closed '최근 N개 분기 ... 분기별로' wire (#202).

        The count belongs to the time window, not a ranking. Real top-k and
        aggregate demands remain untouched, and the caller still requires
        every quarter to close before returning any rewritten intent.
        """
        if (len(intent.answer_items) != 1 or intent.answer_groups
                or intent.premises or intent.unresolved_mentions):
            return None
        item = intent.answer_items[0]
        found = _match_recent_period_span(question, _RECENT_QUARTERS_CORE)
        if found is None or len(_RECENT_QUARTERS_CORE.findall(question)) != 1:
            return None
        literal, count = found
        if (item.target.kind != "metric" or item.operation != "retrieve"
                or item.selection is None or item.selection.mode != "top_k"
                or item.selection.criterion_surface != "분기별"
                or item.selection.k != count or "분기별" not in question
                or item.output.shape != "record_list"
                or item.output.projection_mode != "named_fields"
                or len(item.output.field_surfaces) != 1
                or item.scope.target_period_expressions not in ([], [literal])
                or re.search(r"최대|최소|가장|높|낮|큰|작|상위|하위|평균|합계|누계", question)):
            return None
        return item.model_copy(update={
            "selection": None,
            "output": item.output.model_copy(update={"shape": "scalar"}),
        })

    def _single_company_row(self, item: Any, intent: SemanticIntent) -> "Any | None":
        by_id = {entity.entity_id: entity for entity in intent.entities}
        companies = []
        for ref in item.target.entity_refs:
            entity = by_id.get(ref)
            if entity is None or entity.kind_hint != "company":
                continue
            rows = list(self.canonical.resolve_company(entity.surface) or ())
            if len(rows) == 1:
                companies.append(rows[0])
        corp_codes = {row.corp_code for row in companies}
        if len(companies) == 1 and len(corp_codes) == 1:
            return companies[0]
        return None

    def _every_period_closes(
            self, question: str, intent: SemanticIntent, item: Any,
            concept: Any,
            assignments: "tuple[FanoutPeriodAssignment, ...]") -> bool:
        """N 개 기간이 **전부** 같은 자리에 좌표를 세우는가.

        `SummaryMetricFanoutRegrounder._every_concept_closes` 와 같은 가드다
        — 컴파일러의 `_validate_recent_periods_fanout_resolution` 이 뒤에서
        같은 것을 다시 보므로, 여기서 먼저 걸러야 그 검증이 컴파일 오류로
        터지지 않는다.
        """

        from .stage1_v1_financial_backend import (
            _financial_coordinates_have_source_facts,
            _single_quarter_cumulative_coordinates,
            financial_coordinates,
        )

        for assignment in assignments:
            if assignment.mode == "derived_discrete":
                rows = _single_quarter_cumulative_coordinates(
                    item, intent=intent, companies=self.canonical,
                    reference_date=self.reference_date,
                    corpus_cutoff=self.corpus_cutoff,
                    scope_authority=self.scope_authority, question=question,
                    period_override=assignment.literal)
                expected = 2
            else:
                rows = financial_coordinates(
                    item, intent=intent, companies=self.canonical,
                    reference_date=self.reference_date,
                    corpus_cutoff=self.corpus_cutoff,
                    scope_authority=self.scope_authority, question=question,
                    concept_override=concept,
                    period_override=assignment.literal)
                expected = 1
            if (len(rows) != expected
                    or not _financial_coordinates_have_source_facts(
                        rows, companies=self.canonical,
                        as_of=self.corpus_cutoff)):
                return False
        return True

    def __call__(
            self, question: str, intent: SemanticIntent) -> SemanticIntent:
        from .stage1_v1_financial_backend import _bare_concept_surface

        if not isinstance(question, str) or not question.strip():
            return intent
        item = self._closed_single_item(intent)
        if item is None:
            item = self._quarter_listing_item(question, intent)
        if item is None:
            return intent
        binding_intent = intent.model_copy(update={"answer_items": [item]})
        fanout = recent_periods_fanout(question, item)
        if fanout is None:
            return intent
        company = self._single_company_row(item, intent)
        if company is None:
            return intent
        concept = _resolved_concept(item)
        if concept is None:
            return intent
        assignments = _recent_period_assignments(
            fanout, companies=self.canonical, corp_code=company.corp_code,
            concept=concept, corpus_cutoff=self.corpus_cutoff)
        if assignments is None or len(assignments) != fanout.count:
            return intent
        # HCX-007 실호출로 직접 확인함 — 「최근 N…」조각이 target.surface
        # 자체에 접혀 들어가거나(「최근 3개년 연결 영업이익」),
        # qualifier_surfaces로 쪼개져 온다(["최근", "4개 분기"]). 그 조각을
        # 떼야 output.field_surfaces(증거 대조에 쓰는 계정 표면)와 구조
        # 서명의 target qualifier_count가 깨끗하게 남는다 — 원 표면의
        # **부분문자열**만 남기므로 grounding은 그대로 유지된다. 좌표를
        # 확인하는 item도 실제로 펼 이 정리된 모양이어야 앞뒤가 맞는다.
        cleaned_fields = [
            re.sub(
                r"\s*추이\s*$", "",
                _strip_recent_period_prefix(surface) or surface).strip()
            for surface in item.output.field_surfaces
        ]
        cleaned_surface = _strip_recent_period_prefix(item.target.surface)
        if not cleaned_surface or resolve_metric_concept(
                _bare_concept_surface(cleaned_surface)) is None:
            # Some HCX wires put only the period in target.surface and keep
            # the metric in the named output field.  Reusing that literal
            # question substring restores the target without inventing text.
            cleaned_surface = next(
                (surface for surface in cleaned_fields
                 if resolve_metric_concept(
                     _bare_concept_surface(surface)) is not None),
                item.target.surface)
        cleaned_qualifiers = _strip_period_qualifiers(
            list(item.target.qualifier_surfaces), fanout.literal)
        cleaned_item = item.model_copy(update={
            "target": item.target.model_copy(update={
                "surface": cleaned_surface,
                "qualifier_surfaces": cleaned_qualifiers,
            }),
            "output": item.output.model_copy(update={
                "field_surfaces": cleaned_fields,
            }),
        })
        if not self._every_period_closes(
                question, binding_intent, cleaned_item, concept, assignments):
            return intent
        items = [
            AnswerItem(
                item_id=f"item-{index}",
                target=cleaned_item.target.model_copy(deep=True),
                operation=cleaned_item.operation,
                scope=cleaned_item.scope.model_copy(
                    deep=True,
                update={"target_period_expressions": [
                    assignments[index - 1].literal if fanout.explicit_literals
                    else fanout.literal]}),
                selection=None,
                output=cleaned_item.output.model_copy(deep=True),
            )
            for index in range(1, fanout.count + 1)
        ]
        return SemanticIntent(
            schema_version=intent.schema_version,
            entities=intent.entities,
            answer_items=items, answer_groups=[], premises=[],
            unresolved_mentions=[], presentation="table",
        )
