"""SemanticIntent 의 재무 항목을 **정본 좌표로 확정하는** v1 resolver 백엔드.

`Stage1ResolutionBackend` 의 첫 실제 구현이다. 지금까지 v1 네이티브 경로에는
`fixtures/stage1_v1_vertical_slices/` 에서 미리 구운 권위를 돌려주는 정적 백엔드밖에
없어서 8문항만 돌았고, 나머지는 v1 의미를 v0.5 wire 로 되돌리는 **호환 브리지**를
거쳤다. 그 브리지가 `counterparty` 같은 축을 버리는 것이 관측됐다.

## 무엇을 빌리고 무엇을 빌리지 않는가

resolver 계약이 명시적으로 허용한다 — 「A backend may reuse proven canonical lookup
code from the legacy resolver, but it must return one of the typed v1 authorities」.

    빌린다      회사 해석 · 개념 매핑 · 회계기간 계산 · 주재무제표 판정
                이미 검증된 조회다. 두 번 구현하면 드리프트가 난다.

    빌리지 않는다  **무엇을 물었는가**. route·task 모양·필드 선택은 전부
                `SemanticIntent` 가 정한다. 그래서 이것은 「v0.5 에 모자를 씌운 것」이
                아니다 — 결정 권한이 v1 에 있다.

## 범위

재무 항목만 확정한다. 그 밖의 의미(사건·문서·서술)는 `None` 을 돌려주어
호출자가 다른 백엔드나 종전 경로로 보내게 한다. **모르는 것을 지어내지 않는다.**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import re
from typing import Any, Callable

from .concept_alias import normalize_surface_key, resolve_from_question
from .contracts import FinancialConcept
from .semantic_intent_v1_boundary import _LEADING_FINANCIAL_SCOPE
from .planning import (
    _target_date_range,
    _financial_period,
    concept_display_name,
    CONCEPT_QUESTION_PATTERNS,
    concept_axes,
    primary_scope_for_period,
    resolve_metric_concept,
)
from .semantic_intent_v1 import SemanticIntent
from .stage1_v1_resolver import (
    ClarificationAuthority,
    ClarificationOption,
    build_resolution_premise_proofs,
)
from .recent_periods_fanout import fanout_period_assignments_by_sibling
from .summary_metric_fanout import fanout_concepts_by_sibling


#: 「연결」·「별도」를 고르는 명시 표현. 없으면 그 회사·기간의 **주재무제표**를 쓴다.
_CFS_CUES = ("연결",)
_SFS_CUES = ("별도", "개별")


@dataclass(frozen=True, slots=True)
class FinancialCoordinate:
    """한 answer item 이 가리키는 정본 재무 좌표."""

    corp_code: str
    corp_name: str
    concept: FinancialConcept
    period_start: "date | None"
    period_end: date
    period_type: str
    cumulative: "bool | None"
    scope: str
    statement: str
    #: `scope` 를 질문이 밝혔는가. 아니면 그 회사·기간의 주재무제표를 썼다는 뜻이고,
    #: 그 사실은 `applied_defaults` 에 남아야 한다.
    scope_was_explicit: bool = True

    def as_resolution(self, *, as_of: str, view: str) -> "dict[str, Any]":
        return {
            "kind": "financial",
            "corp_code": self.corp_code,
            "corp_name": self.corp_name,
            "concept": self.concept,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "period_type": self.period_type,
            "scope": self.scope,
            "statement": self.statement,
            "view": view,
            "as_of": as_of,
            "cumulative": self.cumulative,
        }


#: 개념 앞에 붙어 개념 자체를 바꾸지 않는 수식어. **표를 고르는 말**과
#: **기간을 고르는 말**이다. 정본 계정 사전은 순수한 계정명만 담으므로
#: 이것들이 붙어 있으면 개념을 못 찾는다(「연결 매출액」조차 못 찾았다).
#:
#: 떼어낸 값은 버리지 않는다 — scope 는 별도로 읽고, 기간은 이미 scope 에 있다.
_CONCEPT_PREFIXES = (
    # 「직전 분기」「전전 분기」「전년」— `agent.relative_financial_period`의
    # 상대 기간 리터럴과 짝이다. HCX가 그 표현을 target surface 안에 그대로
    # 접어 넣을 수 있다(「직전 분기 연결 영업이익」— 이 사전이 모르는 접두어가
    # 붙는다, SG-013). 그 regrounder가 같은 리터럴을 이미
    # ``target_period_expressions``로 옮기므로, 여기서도 개념 표면에서
    # 걷어내야 「연결 영업이익」이 남아 개념을 찾는다.
    "전전 분기", "전전분기", "직전 분기", "직전분기", "전년",
    # 「최근 분기」「가장 최근 분기」「최신 분기」— #171 M14,
    # `agent.relative_financial_period`의 같은 리터럴과 짝이다.
    "가장 최근 분기", "가장최근분기", "최근 분기", "최근분기",
    "최신 분기", "최신분기",
    "단일 분기", "단일분기", "누적", "연간", "분기", "반기",
    "손익계산서상", "현금흐름표상",
    "연결현금흐름표의", "별도현금흐름표의", "연결재무제표의", "별도재무제표의",
    "연결손익계산서의", "별도손익계산서의", "연결재무상태표의", "별도재무상태표의",
    "연결현금흐름표", "별도현금흐름표", "연결재무제표", "별도재무제표",
    "연결손익계산서", "별도손익계산서", "연결재무상태표", "별도재무상태표",
)


class QuestionGroundedCompoundFinancialRegrounder:
    """Restore only closed literal amount/rate, dated, and statement-pair requests."""

    def __init__(self, company_surface_regrounder):
        self.company_surface_regrounder = company_surface_regrounder

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        from .stage1_v1_closed_grounding_fallback import closed_compound_financial_intent

        if intent.premises or intent.answer_groups or intent.unresolved_mentions:
            return intent
        return closed_compound_financial_intent(
            question, company_surface_regrounder=self.company_surface_regrounder) or intent


def _bare_concept_surface(surface: str) -> str:
    """개념 표면에서 **표·기간 수식어를 반복해서** 떼어낸다.

    `_LEADING_FINANCIAL_SCOPE` 는 `연결`·`별도` 한 겹만 뗀다. 실제 질문은
    「단일 분기 연결 매출액」처럼 여러 겹으로 온다. 더 떼어낼 것이 없을 때까지
    돌리되, **개념이 통째로 사라지면 되돌린다** — 수식어만 남기고 개념을 지우면
    아무 계정에나 붙을 수 있다.
    """

    value = (surface or "").strip()
    for _ in range(4):                       # 겹쳐 봐야 서너 겹이다
        before = value
        match = _LEADING_FINANCIAL_SCOPE.fullmatch(value)
        if match is not None:
            value = match.group("rest").strip()
        for prefix in _CONCEPT_PREFIXES:
            if value.startswith(prefix) and value[len(prefix):].strip():
                value = value[len(prefix):].strip()
                break
        if value == before:
            break
    parenthesized = re.fullmatch(r"(?:capex|캐펙스)\s*\((.+)\)", value, re.IGNORECASE)
    if parenthesized is not None:
        value = parenthesized.group(1).strip()
    return value or (surface or "").strip()


def _company_surface(intent: SemanticIntent, item: Any) -> "str | None":
    """이 항목이 가리키는 회사 표면. entity 참조를 따라간다."""

    by_id = {row.entity_id: row for row in intent.entities}
    for entity_id in item.target.entity_refs:
        entity = by_id.get(entity_id)
        if entity is not None and entity.kind_hint == "company":
            return entity.surface
    return None


def _scope_from(expressions: "tuple[str, ...] | list[str]") -> "str | None":
    """명시된 연결·별도만 읽는다. 없으면 ``None`` — 주재무제표로 넘긴다."""

    joined = normalize_surface_key(" ".join(expressions or ()))
    if any(normalize_surface_key(cue) in joined for cue in _SFS_CUES):
        return "SFS"
    if any(normalize_surface_key(cue) in joined for cue in _CFS_CUES):
        return "CFS"
    return None


def resolve_financial_as_of(
        item: Any, *, reference_date: date, corpus_cutoff: str,
        ) -> "tuple[str | None, str | None]":
    """항목의 `as_of_expression` 을 기준시점(YYYYMMDD)으로 내린다.

    없으면 corpus cutoff. 「2026년 3월 기준」처럼 기간 표현이면 그 기간의 **마지막 날**, 일 단위면
    그날. cutoff 이후는 cutoff 로 자른다(그 뒤 문서는 정본에 없다). 해석하지 못하면 `(None, 사유)` —
    호출자는 cutoff 로 조용히 대체하지 말고 확정 실패로 닫아야 한다. 종전에는 표현을 읽지 않고 항상
    cutoff 를 써서 「2026년 3월 기준 2025년 매출」이 경고 없이 최신 시점 값으로 컴파일됐다.
    """
    expression = getattr(getattr(item, "scope", None), "as_of_expression", None)
    text = str(expression).strip() if expression is not None else ""
    if not text or text == "unspecified":
        return corpus_cutoff, None
    text = re.sub(r"\s*(?:기준|까지|현재|시점|기준으로)\s*$", "", text)
    start, end, error = _target_date_range(text, reference_date=reference_date)
    if error is not None or start is None or end is None:
        return None, error or "as_of 표현을 날짜로 내릴 수 없습니다"
    return min(end, corpus_cutoff), None


def financial_coordinates(
        item: Any, *, intent: SemanticIntent, companies: Any,
        reference_date: date, corpus_cutoff: str, scope_authority: Any = None,
        question: str = "",
        concept_override: "FinancialConcept | None" = None,
        period_override: "str | None" = None,
        ) -> "tuple[FinancialCoordinate, ...]":
    """이 항목이 가리키는 재무 좌표 **전부**. 확정 못 하면 빈 튜플.

    한 항목이 좌표를 둘 이상 갖는 경우가 있다.

    - **회사 둘**(「삼전과 하닉 중 큰 곳」) → 같은 기간·개념을 회사마다.
      동결 Gold 는 이것을 `financial_comparison` 으로 담는다(operand 의 corp 는
      서로 달라야 하고 기간·개념은 같아야 한다).
    - **기간 둘**(「2026년 1분기는 2025년 1분기보다」) → 같은 회사·개념을 기간마다.
      Gold 는 이것을 비교가 아니라 **fact 둘 + 파생**으로 담는다
      (`difference`·`percent_change`·`discrete_from_cumulative`).

    두 경우 모두 **좌표 계산 자체는 같다.** 축만 다르므로 곱집합으로 돈다.

    ``concept_override`` 는 표면으로 개념을 풀지 않고 호출자가 정한 것을 쓴다
    (`_one_coordinate` 와 같은 뜻). 「벌었어」처럼 표면 하나가 후보 여럿을
    가리켜 형제 item 순서로 후보를 나눠 맡을 때 쓴다(#94 25).

    ``period_override`` 는 ``concept_override`` 의 기간판이다 — 표현을
    ``item.scope.target_period_expressions`` 에서 읽지 않고 호출자가 정한
    리터럴 하나를 쓴다. 「최근 3년」처럼 표면 하나가 실제 기간 여럿을
    가리켜 형제 item 순서로 실제 연도·분기를 나눠 맡을 때 쓴다(#58 2단계,
    `agent.recent_periods_fanout`).
    """

    companies_surfaces = _company_surfaces(intent, item)
    periods = (
        [period_override] if period_override is not None
        else list(item.scope.target_period_expressions or ()))
    if not companies_surfaces or not periods:
        return ()
    scope_expressions = [
        *item.scope.scope_qualifier_expressions,
        *item.target.qualifier_surfaces,
        item.target.surface,
    ]
    joined_scope = normalize_surface_key(" ".join(scope_expressions))
    explicit_scopes = []
    if any(normalize_surface_key(cue) in joined_scope for cue in _CFS_CUES):
        explicit_scopes.append("CFS")
    if any(normalize_surface_key(cue) in joined_scope for cue in _SFS_CUES):
        explicit_scopes.append("SFS")
    scope_variants: list[str | None] = explicit_scopes or [None]

    rows: list[FinancialCoordinate] = []
    for surface in companies_surfaces:
        for expression in periods:
            for scope_override in scope_variants:
                row = _one_coordinate(
                    item, company_surface=surface, period_expression=expression,
                    companies=companies, reference_date=reference_date,
                    corpus_cutoff=corpus_cutoff, scope_authority=scope_authority,
                    question=question, scope_override=scope_override,
                    concept_override=concept_override)
                if row is None:
                    return ()      # 하나라도 못 세우면 이 항목은 확정 실패다
                rows.append(row)
    return tuple(rows)


def _financial_coordinates_have_source_facts(
        rows: "tuple[FinancialCoordinate, ...]", *, companies: Any,
        as_of: str,
        ) -> bool:
    """Whether every typed coordinate has an exact canonical source fact.

    Normal scalar resolution deliberately separates coordinate construction
    from Stage2 value lookup.  Recent-period fanout is stricter: it expands a
    single request into several answer items and promises all of them, so the
    expansion must prove every direct fact or every cumulative derivation
    operand before changing the intent shape.
    """

    if not rows or not callable(getattr(companies, "facts", None)):
        return False
    for row in rows:
        candidates = list(companies.facts(
            row.corp_code, as_of=as_of, concept=row.concept.value))
        period_start = (
            row.period_start.isoformat() if row.period_start is not None
            else None)
        period_end = row.period_end.isoformat()
        matching_axes = [fact for fact in candidates if (
                getattr(fact, "period_start", None) == period_start
                and getattr(fact, "period_end", None) == period_end
                and getattr(fact, "period_type", None) == row.period_type
                and getattr(fact, "scope", None) == row.scope
                and (row.cumulative is None
                     or getattr(fact, "cumulative", None) == row.cumulative))]
        # #202: Stage2 already retries IS -> CI (and CI -> IS) when a
        # company publishes only one income statement. Prove that same
        # fallback here without weakening period/scope/cumulative binding.
        statements = {row.statement}
        if row.statement in {"IS", "CI"}:
            statements = {"IS", "CI"}
        if not any(getattr(fact, "statement", None) in statements
                   for fact in matching_axes):
            return False
    return True


def _company_surfaces(intent: SemanticIntent, item: Any) -> "list[str]":
    """이 항목이 가리키는 회사 표면 **전부**. 질문 순서를 지킨다."""

    by_id = {row.entity_id: row for row in intent.entities}
    out: list[str] = []
    for entity_id in item.target.entity_refs:
        entity = by_id.get(entity_id)
        if entity is not None and entity.kind_hint == "company":
            out.append(entity.surface)
    return out


#: 「직전 분기」「전전 분기」— 이슈 #58, 2단계에서 정의를 루브릭 쪽으로
#: 통일했다.  기준일(코퍼스 컷오프) 시점에 「직전 분기」는 최신 실제 분기
#: 에서 한 분기 더 앞이 아니라 **정본에 실제로 있는 가장 최근 분기 그
#: 자체**다(「마지막으로 끝나고 보고된 분기」) — 「전전 분기」가 그 한 분기
#: 더 앞이며, 옛 「직전 분기」 정의를 그대로 잇는다. 다중 기간(「최근 N년」
#: 「최근 N개 분기」)은 이 단일표현 재작성에 넣지 않는다 — 전용
#: `RecentPeriodsFanoutRegrounder`가 형제를 만들고, 아래 재무 backend와
#: compiler가 직접 분기 및 FY-9M 파생 Q4를 순서대로 결속한다. 단일 상대
#: 기간(「직전 분기」「전전 분기」)은 기존 단일-``financial`` 경로를 탄다.
#: 「최근 분기」「가장 최근 분기」「최신 분기」도 「직전 분기」와 같은 뜻
#: (offset 0) — #171 M14, 루브릭 통일.
_PREVIOUS_QUARTER = re.compile(
    r"직전분기|바로전분기|직전분기말|(?:가장)?최근분기|(?:가장)?최근분기말|"
    r"최신분기|최신분기말")
_PREVIOUS_PREVIOUS_QUARTER = re.compile(r"전전분기|전전분기말")


def _relative_quarter_offset(normalized_period: str) -> "int | None":
    """정본의 최신 실제 분기에서 몇 분기를 거슬러 올라가는 표현인지.

    「전전 분기」를 먼저 본다 — 두 정규식이 겹치는 표현은 없지만(문자열
    자체가 다르다), 어느 쪽이 매치했는지로 오프셋이 갈리므로 순서를
    명시해 둔다. 상대 분기 표현이 아니면 ``None``.
    """

    if _PREVIOUS_PREVIOUS_QUARTER.fullmatch(normalized_period) is not None:
        return 1
    if _PREVIOUS_QUARTER.fullmatch(normalized_period) is not None:
        return 0
    return None


#: ``period_end`` 마지막 5자 → 분기 번호. 표준 달력 분기 종료일만 받는다.
_QUARTER_ENDPOINTS = {"03-31": 1, "06-30": 2, "09-30": 3, "12-31": 4}


def _latest_actual_quarter(
        companies: Any, *, corp_code: str, concept: FinancialConcept,
        corpus_cutoff: str,
        ) -> "tuple[int, int] | None":
    """정본에 실제로 있는 가장 최근 (연도, 분기). 없으면 ``None``.

    ``period_type`` 만 본다 — Q1은 단일분기와 누적이 같은 값이라 구분할
    필요가 없고, Q3 단일분기·9개월누적처럼 ``cumulative`` 만 다른 행이
    섞여 있어도 **어느 분기인지**를 고르는 데는 영향이 없다(둘 다
    ``period_end`` 가 같은 분기 말이다).
    """

    if not callable(getattr(companies, "facts", None)):
        return None
    candidates = [
        row for row in companies.facts(
            corp_code, as_of=corpus_cutoff, concept=concept.value)
        if getattr(row, "period_type", None) == "quarter"
        and isinstance(getattr(row, "period_end", None), str)
        and row.period_end[5:] in _QUARTER_ENDPOINTS
        and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", row.period_end)
        and row.period_end.replace("-", "") <= corpus_cutoff
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda row: row.period_end)
    return int(best.period_end[:4]), _QUARTER_ENDPOINTS[best.period_end[5:]]


def _offset_quarter(year: int, quarter: int, *, back: int) -> "tuple[int, int]":
    """``back`` 분기만큼 거슬러 올라간 (연도, 분기)."""

    zero_based = (year * 4 + (quarter - 1)) - back
    return zero_based // 4, zero_based % 4 + 1


def _latest_actual_annual_year(
        companies: Any, *, corp_code: str, concept: FinancialConcept,
        corpus_cutoff: str,
        ) -> "int | None":
    """정본에 실제로 있는 가장 최근 사업연도(annual, 12-31 종료). 없으면 ``None``.

    `_one_coordinate` 의 「최신 연간 실적」분기(문자열 트리거 ``최신``·
    ``연간``·``실제``)와 같은 후보 조건이다. 이슈 #58 2단계의 「최근 N년」
    fanout(`agent.recent_periods_fanout`)이 실제 연도를 지어내지 않고
    이 함수로 시작점을 구한다.
    """

    if not callable(getattr(companies, "facts", None)):
        return None
    candidates = [
        row for row in companies.facts(
            corp_code, as_of=corpus_cutoff, concept=concept.value)
        if getattr(row, "period_type", None) == "annual"
        and isinstance(getattr(row, "period_end", None), str)
        and re.fullmatch(r"[0-9]{4}-12-31", row.period_end)
        and row.period_end.replace("-", "") <= corpus_cutoff
    ]
    if not candidates:
        return None
    return max(int(row.period_end[:4]) for row in candidates)


def _relative_previous_quarter_expression(
        item: Any, *, intent: SemanticIntent, companies: Any,
        corpus_cutoff: str,
        ) -> "str | None":
    """「직전/전전 분기」가 Q2~Q4로 떨어지면(=파생이 필요한 분기) 그 리터럴 표현.

    Q2·Q3·Q4 단일분기는 누적 두 endpoint의 차(`_single_quarter_cumulative_
    coordinates`)로만 구할 수 있다 — 정본에 discrete Q4 사실 자체가 없다
    (연간 - 9개월누적으로만 구해진다).  Q1로 떨어지면(파생이 필요 없다)
    ``None`` — 호출자가 일반 단일좌표 경로로 그대로 처리한다.
    """

    offset = next(
        (o for value in item.scope.target_period_expressions
         if (o := _relative_quarter_offset(normalize_surface_key(value)))
         is not None),
        None,
    )
    if offset is None:
        return None
    company_surfaces = _company_surfaces(intent, item)
    if len(company_surfaces) != 1:
        return None
    resolved = companies.resolve_company(company_surfaces[0])
    if len(resolved) != 1:
        return None
    concept = resolve_metric_concept(item.target.surface)
    if concept is None:
        concept = resolve_metric_concept(_bare_concept_surface(item.target.surface))
    if concept is None:
        return None
    latest = _latest_actual_quarter(
        companies, corp_code=resolved[0].corp_code, concept=concept,
        corpus_cutoff=corpus_cutoff)
    if latest is None:
        return None
    year, quarter = _offset_quarter(*latest, back=offset)
    if quarter == 1:
        return None
    return f"{year}년{quarter}분기"


def _one_coordinate(
        item: Any, *, company_surface: str, period_expression: str,
        companies: Any, reference_date: date, corpus_cutoff: str,
        scope_authority: Any = None,
        question: str = "",
        scope_override: str | None = None,
        concept_override: "FinancialConcept | None" = None,
        ) -> "FinancialCoordinate | None":
    """재무 항목이면 좌표를, 아니면 ``None``.

    **확정하지 못하면 ``None`` 이다.** 회사를 못 찾거나 개념을 모르거나 기간이
    안 서면 지어내지 않고 물러난다 — 그 판단은 호출자가 한다.

    ``concept_override`` 가 있으면 ``item.target.surface`` 로 개념을 다시
    풀지 않는다 — 이슈 #38 의 `_ratio_request` 가 분자·분모 개념을 이미
    확정해 넘길 때 쓴다(「영업이익률」은 그 자체로 단일 개념이 아니다).
    """

    if item.target.kind != "metric":
        return None
    resolved = companies.resolve_company(company_surface)
    if len(resolved) != 1:
        return None
    company = resolved[0]

    if concept_override is not None:
        concept = concept_override
    else:
        concept = resolve_metric_concept(item.target.surface)
        if concept is None:
            # 「단일 분기 연결 매출액」처럼 수식어가 붙은 표면을 벗겨 다시 본다.
            concept = resolve_metric_concept(_bare_concept_surface(item.target.surface))
        if concept is None and re.fullmatch(
                r"이자\s*지급액", _bare_concept_surface(item.target.surface)):
            # A cash-payment label is not the IS accrual expense concept.
            concept = FinancialConcept("interest_paid")
        if concept is None and question:
            colloquial = resolve_from_question(
                question, CONCEPT_QUESTION_PATTERNS)
            if colloquial.status == "resolved":
                concept = colloquial.concept
        if concept is None:
            return None

    normalized_period = normalize_surface_key(period_expression)
    if ("최신" in normalized_period and "연간" in normalized_period
            and "실제" in normalized_period
            and callable(getattr(companies, "facts", None))):
        candidates = [
            row for row in companies.facts(
                company.corp_code, as_of=corpus_cutoff,
                concept=concept.value)
            if getattr(row, "period_type", None) == "annual"
            and isinstance(getattr(row, "period_end", None), str)
            and re.fullmatch(r"[0-9]{4}-12-31", row.period_end)
            and row.period_end.replace("-", "") <= corpus_cutoff
        ]
        if not candidates:
            return None
        latest_year = max(int(row.period_end[:4]) for row in candidates)
        effective_period_expression = f"{latest_year}년"
    elif (offset := _relative_quarter_offset(normalized_period)) is not None:
        latest = _latest_actual_quarter(
            companies, corp_code=company.corp_code, concept=concept,
            corpus_cutoff=corpus_cutoff)
        if latest is None:
            return None
        target_year, target_quarter = _offset_quarter(*latest, back=offset)
        effective_period_expression = f"{target_year}년{target_quarter}분기"
    else:
        effective_period_expression = period_expression
    if concept_axes(concept).period_semantics == "instant":
        # Balance questions commonly spell a quarter endpoint as ``1분기 말``.
        # The period parser's quarter token already denotes that exact closing
        # date; remove only this redundant suffix for instant concepts.
        effective_period_expression = re.sub(
            r"(?<=분기)\s*말$", "", effective_period_expression.strip())
    period, error = _financial_period(
        concept, year=None, expression=effective_period_expression,
        reference_date=reference_date)
    if error is not None or period is None:
        return None

    # HCX may keep an explicit ``연결``/``별도`` on the metric surface
    # (``단일분기 연결 매출액``) rather than move it into Scope.  The source
    # semantic authority still contains it, so include every metric-side
    # qualifier before applying the primary-statement default.
    scope = scope_override or _scope_from([
        *item.scope.scope_qualifier_expressions,
        *item.target.qualifier_surfaces,
        item.target.surface,
    ])
    scope_was_explicit = scope is not None
    if scope is None and scope_authority is not None:
        scope = primary_scope_for_period(
            scope_authority, corp_code=company.corp_code,
            period_end=str(period["period_end"]), as_of=corpus_cutoff)
    if scope is None:
        return None

    # `_financial_period` 가 정본 축에서 `statement` 까지 함께 내려 준다.
    statement = period.get("statement")
    if not statement:
        return None

    return FinancialCoordinate(
        corp_code=company.corp_code,
        corp_name=company.corp_name,
        concept=concept,
        period_start=period.get("period_start"),
        period_end=period["period_end"],
        period_type=str(period["period_type"]),
        cumulative=period.get("cumulative"),
        scope=scope,
        statement=statement,
        scope_was_explicit=scope_was_explicit,
    )


class FinancialResolutionBackend:
    """`Stage1ResolutionBackend` — 재무 항목을 정본 좌표로 확정한다.

    **재무만 답한다.** 한 항목이라도 확정하지 못하면 `None` 을 돌려주어 호출자가
    다른 백엔드나 종전 경로로 보내게 한다. 모르는 것을 지어내면 거절해야 할 질문에
    답이 생긴다.
    """

    def __init__(
            self, companies: Any, *, canonical_build_id: str,
            resolver_version: str, reference_date: date, corpus_cutoff: str,
            scope_authority: Any = None, view: str = "restated",
            company_alias_provenance: Callable[
                [str, SemanticIntent], tuple[dict[str, Any], ...]
            ] | None = None,
            ) -> None:
        self.companies = companies
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff
        self.scope_authority = scope_authority if scope_authority is not None else companies
        self.view = view
        self.company_alias_provenance = company_alias_provenance

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> "dict[str, Any] | ClarificationAuthority | None":
        from .deterministic_plan_compiler_v1 import AuthoritativeResolution
        from .semantic_intent_v1 import semantic_intent_digest

        if (
                len(source_intent.answer_items) == 1
                and not source_intent.answer_groups
                and not source_intent.premises
                and not source_intent.unresolved_mentions
        ):
            source_item = source_intent.answer_items[0]
            colloquial = resolve_from_question(
                question, CONCEPT_QUESTION_PATTERNS)
            if (
                    source_item.target.kind == "metric"
                    and source_item.output.shape == "scalar"
                    and source_item.output.projection_mode == "named_fields"
                    and len(source_item.output.field_surfaces) == 1
                    and colloquial.status == "ambiguous"
                    and isinstance(colloquial.surface, str)
                    and normalize_surface_key(colloquial.surface)
                    == normalize_surface_key(source_item.target.surface)
            ):
                return ClarificationAuthority.model_validate({
                    "kind": "clarification",
                    "slots": [{
                        "slot_id": "slot-1",
                        "role_hint": "value_kind",
                        "reason_code": "metric_surface_multiple_candidates",
                        "response_kind": "select_one",
                        "prompt": "어느 재무지표를 뜻하나요?",
                        "applies_to_item_ids": [source_item.item_id],
                        "mention_ids": [],
                        # 사용자가 보고 고르는 문자열이다. 정규 개념 식별자를
                        # 그대로 내보내면 한국어 질문을 한 사용자가 «cf_operating»
                        # 중 하나를 고르게 된다. 재개(resume) 계약이 쓰는 것은
                        # ``value`` 이므로 표시만 바꾼다.
                        "options": [ClarificationOption(
                            value=candidate.value,
                            label=concept_display_name(candidate.value),
                            proof_refs=[
                                "policy:financial-concept-option:"
                                f"{candidate.value}"
                            ],
                        ).model_dump(mode="json")
                        for candidate in colloquial.candidates],
                    }],
                }, strict=True)

        # 「벌었어」·「빚」처럼 되묻던 요약 구어를 편 intent 인가. 폈다면
        # 형제 순서로 후보 개념을 나눠 맡는다 — 표면이 전부 같으므로 순서
        # 말고는 어느 item 이 어느 개념인지 구분할 것이 없다 (#94 25).
        fanout_concepts = fanout_concepts_by_sibling(question, source_intent)
        # 「최근 N년」「최근 N개 분기」를 편 intent 인가(#58 2단계). 폈다면
        # 형제 순서로 실제 연도·분기를 나눠 맡는다 — 위 개념 fanout과 같은
        # 자리지만 축이 다르고(개념 vs 기간) 두 축은 서로 겹치지 않는다.
        fanout_periods = fanout_period_assignments_by_sibling(
            question, source_intent, companies=self.companies,
            corpus_cutoff=self.corpus_cutoff)

        items: list[dict[str, Any]] = []
        for index, item in enumerate(source_intent.answer_items, start=1):
            fanout_concept = fanout_concepts[index - 1]
            fanout_period = fanout_periods[index - 1]
            operand_views = None
            ratio_presentation = None
            # 이슈 #124 — 회사 비교 argmax 의 극값 방향. 기본은 "maximum"
            # (기존 동작과 동일)이고, wire selection 이 최솟값을 명시적으로
            # 요구할 때만 "minimum"으로 바뀐다.
            direction = "maximum"
            ratio_request = _ratio_request(question, item)
            if ratio_request is not None:
                numerator_concept, denominator_concept, presentation = ratio_request
                ratio_rows = _ratio_coordinates(
                    item, intent=source_intent, companies=self.companies,
                    reference_date=self.reference_date,
                    corpus_cutoff=self.corpus_cutoff,
                    scope_authority=self.scope_authority,
                    numerator=numerator_concept, denominator=denominator_concept)
                if ratio_rows is not None:
                    rows = ratio_rows
                    ratio_presentation = presentation
            if ratio_presentation is None and fanout_concept is not None:
                # 편 item 은 표면으로 개념을 풀 수 없다(「벌었어」는 후보가
                # 셋이라 `resolve_metric_concept` 가 `None` 이다). 단일분기
                # 누적 경로도 표면을 다시 푸므로 건너뛰고 개념을 직접 준다.
                rows = financial_coordinates(
                    item, intent=source_intent, companies=self.companies,
                    reference_date=self.reference_date,
                    corpus_cutoff=self.corpus_cutoff,
                    scope_authority=self.scope_authority,
                    question=question, concept_override=fanout_concept)
                if not rows:
                    return None
            elif ratio_presentation is None and fanout_period is not None:
                # 편 item 의 기간 리터럴(「최근 3년」)은 그대로는 정본 기간이
                # 아니다. 형제 순서가 정한 실제 연도/분기를 직접 준다. Q4는
                # DART에 독립 분기행이 없으므로, fanout authority가 FY와 9M
                # 양쪽을 확인한 경우에만 기존 단독분기 파생을 재사용한다.
                if fanout_period.mode == "derived_discrete":
                    rows = _single_quarter_cumulative_coordinates(
                        item, intent=source_intent, companies=self.companies,
                        reference_date=self.reference_date,
                        corpus_cutoff=self.corpus_cutoff,
                        scope_authority=self.scope_authority,
                        question=question,
                        period_override=fanout_period.literal)
                else:
                    rows = financial_coordinates(
                        item, intent=source_intent, companies=self.companies,
                        reference_date=self.reference_date,
                        corpus_cutoff=self.corpus_cutoff,
                        scope_authority=self.scope_authority,
                        question=question,
                        period_override=fanout_period.literal)
                if (not rows or not _financial_coordinates_have_source_facts(
                        rows, companies=self.companies,
                        as_of=self.corpus_cutoff)):
                    return None
            elif ratio_presentation is None:
                # 이슈 #38 — 비율·배수 사전이 확정 못 하거나(사전 밖 개념 쌍),
                # 확정해도 기간·scope 가 하나로 안 닫히면 종전 경로 그대로다.
                from agent.stage1_v1_quarter_comparison import quarter_comparison_coordinates
                rows = quarter_comparison_coordinates(
                    question, item, intent=source_intent, companies=self.companies,
                    reference_date=self.reference_date, corpus_cutoff=self.corpus_cutoff,
                    scope_authority=self.scope_authority)
                if not rows:
                    rows = _single_quarter_cumulative_coordinates(
                        item, intent=source_intent, companies=self.companies,
                        reference_date=self.reference_date,
                        corpus_cutoff=self.corpus_cutoff,
                        scope_authority=self.scope_authority, question=question)
                if not rows:
                    rows = financial_coordinates(
                        item, intent=source_intent, companies=self.companies,
                        reference_date=self.reference_date,
                        corpus_cutoff=self.corpus_cutoff,
                        scope_authority=self.scope_authority,
                        question=question)
                if not rows:
                    return None
                if len({row.corp_code for row in rows}) > 1:
                    # 이슈 #124 — company comparison execution 은 이제
                    # ``argmax`` 연산자를 그대로 쓰고 방향(``direction``)만
                    # 뒤집어 최솟값도 낸다.  wire selection 이 「가장 큰」·
                    # 「가장 작은」 밖의 다른 극값(top_k 등)을 요구하면 여전히
                    # 닫는다.
                    if (item.selection is not None
                            and item.selection.mode not in
                            ("maximum", "minimum")):
                        return None
                    if (item.selection is not None
                            and item.selection.mode == "minimum"):
                        direction = "minimum"
                    # Surface aliases can survive an upstream semantic wire.  A
                    # ranking universe is canonical-corp based, so retain the
                    # first question-ordered coordinate for each corp_code.
                    canonical_rows: dict[str, FinancialCoordinate] = {}
                    for row in rows:
                        canonical_rows.setdefault(row.corp_code, row)
                    rows = tuple(canonical_rows.values())
                operand_views = _question_grounded_view_pair(item, question=question)
                if operand_views is not None:
                    # A view comparison repeats one canonical coordinate under two
                    # independently bound reader views.  It is not a second period
                    # and must not be synthesized unless both literal view roles
                    # survive on the typed source item.
                    if len(rows) != 1 or item.operation != "compare":
                        return None
                    rows = (rows[0], rows[0])
                if (
                        len(rows) == 1
                        and item.operation == "compare"
                        and any(
                            value.strip() in {"전년비", "증가", "감소"}
                            or re.fullmatch(
                                r"전년(?:도)?\s*(?:대비|보다)", value.strip())
                            for value in item.target.qualifier_surfaces)
                        and rows[0].period_type == "annual"
                ):
                    previous = _one_coordinate(
                        item, company_surface=rows[0].corp_name,
                        period_expression=f"{rows[0].period_end.year - 1}년",
                        companies=self.companies,
                        reference_date=self.reference_date,
                        corpus_cutoff=self.corpus_cutoff,
                        scope_authority=self.scope_authority,
                        question=question, scope_override=rows[0].scope,
                    )
                    if previous is None:
                        return None
                    rows = (rows[0], previous)
            item_id = f"item-{index}"
            as_of, as_of_error = resolve_financial_as_of(
                item, reference_date=self.reference_date,
                corpus_cutoff=self.corpus_cutoff)
            if as_of is None:
                return None          # 기준시점을 못 읽으면 확정하지 않는다 (cutoff 로 대체 금지)
            operand_account_paths = None
            if operand_views is not None:
                operand_account_paths = _question_grounded_view_account_paths(
                    rows[0], item=item, question=question,
                    companies=self.companies, as_of=as_of)
                if operand_account_paths is None:
                    return None
            if ratio_presentation is not None:
                payload = _ratio_payload_for(
                    rows[0], rows[1], as_of=as_of, view=self.view,
                    presentation=ratio_presentation)
            else:
                payload = _payload_for(
                    rows, as_of=as_of, view=self.view,
                    question=question, operation=item.operation,
                    direction=direction,
                    operand_as_of=(
                        _period_bound_operand_as_of(
                            rows, companies=self.companies,
                            corpus_cutoff=self.corpus_cutoff)
                        or _instant_fiscal_year_end_operand_as_of(
                            rows, companies=self.companies,
                            corpus_cutoff=self.corpus_cutoff,
                            operand_views=operand_views)
                    ),
                    field_surfaces=tuple(item.output.field_surfaces),
                    parallel_change_operator=_parallel_annual_change_operator(
                        source_intent, index - 1),
                    operand_views=operand_views,
                    output_field_count=len(item.output.field_surfaces),
                    verification_premise=(
                        (source_intent.premises[0].premise_id,
                         source_intent.premises[0].raw_text)
                        if len(source_intent.premises) == 1
                        and source_intent.premises[0].kind == "comparison"
                        and source_intent.premises[0].applies_to_item_ids == [item.item_id]
                        else None
                    ),
                )
            if payload is None:
                return None
            alias_provenance = (
                self.company_alias_provenance(question, source_intent)
                if self.company_alias_provenance is not None else ()
            )
            applied_defaults = _applied_defaults(
                rows, alias_provenance=alias_provenance)
            if operand_account_paths is not None:
                applied_defaults.append({
                    "policy": "question_grounded_view_account_paths",
                    "basis": "unique account ancestor bound independently per filing view",
                    "value": json.dumps({
                        "as_filed": operand_account_paths[0],
                        "restated": operand_account_paths[1],
                    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    "evidence_refs": ["question:financial-account-ancestor"],
                })
            items.append({
                "item_id": item_id,
                "target_surface": item.target.surface,
                "projection_mode": item.output.projection_mode,
                "resolution": payload,
                "applied_defaults": applied_defaults,
                "field_proofs": [
                    {
                        "proof_ref": f"source-field:{item_id}:{position}",
                        "source_field_index": position,
                        "surface": surface,
                    }
                    for position, surface in enumerate(item.output.field_surfaces)
                ],
            })
        if not items:
            return None
        resolution = AuthoritativeResolution.create(
            question_id=question_id,
            source_intent_digest=semantic_intent_digest(source_intent),
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            items=items,
            premise_proofs=build_resolution_premise_proofs(
                source_intent, items),
        )
        return {"kind": "resolved", "resolution": resolution.model_dump(mode="json")}


def _applied_defaults(
        rows: "tuple[FinancialCoordinate, ...]", *,
        alias_provenance: "tuple[dict[str, Any], ...]" = (),
        ) -> "list[dict[str, Any]]":
    """질문이 안 밝혀 **기본값을 쓴 사실**을 남긴다.

    「삼성전자 2025년 매출 얼마야?」는 연결·별도를 밝히지 않는다. 그때 그 회사·
    기간의 **주재무제표**를 쓰는데, 쓴 사실을 적지 않으면 Stage2 와 사용자는 그것이
    질문에서 온 값인지 우리가 고른 값인지 알 수 없다.

    근거는 **정본 조회**를 가리킨다. 동결 Gold 는 여기서 자기 overlay 를 인용하지만
    이 백엔드는 Gold 를 읽지 않으므로 그럴 수 없고, 그래서도 안 된다.
    """

    defaults: list[dict[str, Any]] = []
    explicit = [row for row in rows if not row.scope_was_explicit]
    if explicit:
        value = explicit[0].scope
        if all(row.scope == value for row in explicit):
            defaults.append({
                "policy": "primary_statement_scope",
                "basis": "canonical primary statement for the company and period",
                "value": value,
                "evidence_refs": sorted({
                    f"canonical:primary-statement:{row.corp_code}:{row.period_end}"
                    for row in explicit
                }),
            })
    row_codes = {row.corp_code for row in rows}
    for alias in alias_provenance:
        corp_code = alias.get("corp_code")
        surfaces = alias.get("surfaces")
        if (corp_code not in row_codes or not isinstance(surfaces, list)
                or len(surfaces) < 2
                or any(not isinstance(value, str) or not value for value in surfaces)):
            continue
        defaults.append({
            "policy": "canonical_company_alias_merge",
            "basis": "question surfaces resolve uniquely to one canonical corp_code",
            "value": json.dumps({
                "corp_code": corp_code,
                "corp_name": alias.get("corp_name"),
                "surfaces": surfaces,
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "evidence_refs": [
                f"canonical:company:{corp_code}",
                *[f"question:company-alias:{index}"
                  for index in range(1, len(surfaces) + 1)],
            ],
        })
    return defaults


_PERCENT_CHANGE_FIELD = re.compile(
    r"증가\s*율|감소\s*율|"
    r"%|퍼센트|몇\s*퍼(?:\b|센트)|증감률|변동률|변화율|비율")
_AMOUNT_CHANGE_FIELD = re.compile(
    r"증감액|변동액|변화액|차이|금액|얼마나(?!\s*(?:변|바뀌))")
_EXPLICIT_AMOUNT_DIRECTION = re.compile(
    r"얼마나\s*(?:증가|감소|늘|줄)|증가(?:하거나|했|한).*?감소|"
    r"감소(?:하거나|했|한).*?증가|증감(?:액|했|한)|차이")
_AS_FILED_VIEW = re.compile(r"최초\s*제출\s*값")
_RESTATED_VIEW = re.compile(r"최신\s*재작성\s*값")


def _time_relation_operators(
        *, question: str, field_surfaces: tuple[str, ...],
        ) -> "list[str] | None":
    """Bind explicit change fields to compiler operators in source order."""

    if len(field_surfaces) == 2:
        roles: list[str | None] = []
        for surface in field_surfaces:
            percent = _PERCENT_CHANGE_FIELD.search(surface) is not None
            amount = _AMOUNT_CHANGE_FIELD.search(surface) is not None
            roles.append(
                "percent_change" if percent and not amount else
                "difference" if amount and not percent else None)
        if roles.count("percent_change") == 1 and roles.count(None) == 1:
            roles[roles.index(None)] = "difference"
        if roles.count("difference") == 1 and roles.count(None) == 1:
            roles[roles.index(None)] = "percent_change"
        if set(roles) == {"difference", "percent_change"}:
            return [str(role) for role in roles]
        return None
    if len(field_surfaces) != 1:
        return None
    # 이슈 #124(합계) — 「삼성전자의 2024년과 2025년 연결 매출액 합계는?」.
    # 같은 회사·같은 concept의 서로 다른 두 period를 더하라는 요청이다 —
    # `_CROSS_COMPANY_SUM_PATTERN`과 같은 낱말, 다만 여기는 회사가 아니라
    # period 축이 갈리는 same-company 분기라 field_surfaces로 판단한다.
    if _CROSS_COMPANY_SUM_PATTERN.search(field_surfaces[0]) is not None:
        return ["sum"]
    if (_PERCENT_CHANGE_FIELD.search(field_surfaces[0]) is not None
            or re.search(
                r"몇\s*퍼센트|몇\s*퍼|퍼센트\s*(?:증가|감소)", question)):
        return ["percent_change"]
    # An amount-named field or an explicit increase/decrease direction asks
    # only for the amount.  A broad ``얼마나 변했어`` does not choose between
    # amount and rate, so it keeps the established combined-change contract.
    if _AMOUNT_CHANGE_FIELD.search(field_surfaces[0]) is not None:
        return ["difference"]
    if (_EXPLICIT_AMOUNT_DIRECTION.search(field_surfaces[0]) is not None
            or _EXPLICIT_AMOUNT_DIRECTION.search(question) is not None):
        return ["difference"]
    # Preserve the existing combined-change contract for one broad demand.
    return ["difference", "percent_change"]


def _question_grounded_view_pair(
        item: Any, *, question: str,
        ) -> "tuple[str, str] | None":
    """Bind the two explicit filing views retained by the source intent."""

    as_filed = _AS_FILED_VIEW.search(question)
    restated = _RESTATED_VIEW.search(question)
    if as_filed is None or restated is None or re.search(r"비교", question) is None:
        return None
    roles = list(item.target.qualifier_surfaces)
    expected = sorted(
        [as_filed.group(0), restated.group(0)], key=question.index)
    if roles != expected:
        return None
    views = {
        "as_filed" if _AS_FILED_VIEW.fullmatch(role) else
        "restated" if _RESTATED_VIEW.fullmatch(role) else None
        for role in roles
    }
    if views != {"as_filed", "restated"}:
        return None
    # Resolution order is semantic, not prose order: later-restated minus
    # earliest-as-filed is compiled from this stable pair.
    return "as_filed", "restated"


def _question_grounded_view_account_paths(
        row: FinancialCoordinate, *, item: Any, question: str,
        companies: Any, as_of: str,
        ) -> "tuple[str, str] | None":
    """Bind each filing view to the exact question-selected account path.

    Account-path spelling can change between the original and a later
    restatement (for example a numbered top-level heading becoming ``자산``).
    Select each view's own literal path, but only after a unique ancestor in
    the user's ``A 중 B`` expression filters all sibling accounts.
    """

    if not callable(getattr(companies, "facts", None)):
        return None
    target = re.escape(item.target.surface.strip())
    ancestors = list(dict.fromkeys(
        match.group("ancestor").strip()
        for match in re.finditer(
            rf"(?P<ancestor>[A-Za-z0-9가-힣&+]+)\s*중(?:에서)?\s*{target}",
            question)
    ))
    if len(ancestors) != 1:
        return None
    ancestor_key = normalize_surface_key(ancestors[0])
    candidates = [
        fact for fact in companies.facts(
            row.corp_code, as_of=as_of, concept=row.concept.value,
            scope=row.scope, statement=row.statement,
            period_end=row.period_end.isoformat(), cumulative=row.cumulative)
        if isinstance(getattr(fact, "account_path", None), str)
        and ancestor_key in normalize_surface_key(fact.account_path)
        and getattr(fact, "period_start", None) == (
            row.period_start.isoformat() if row.period_start is not None else None)
    ]
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda fact: (fact.rcept_dt, fact.doc_id))
    selected_dates = (ordered[0].rcept_dt, ordered[-1].rcept_dt)
    paths = []
    for selected_date in selected_dates:
        selected_paths = {
            fact.account_path for fact in ordered if fact.rcept_dt == selected_date}
        if len(selected_paths) != 1:
            return None
        paths.append(next(iter(selected_paths)))
    return paths[0], paths[1]


def _period_bound_operand_as_of(
        rows: "tuple[FinancialCoordinate, ...]", *, companies: Any,
        corpus_cutoff: str,
        ) -> "tuple[str, ...] | None":
    """Bind each same-company period operand to its own periodic filing.

    A later annual or interim report repeats earlier periods as comparison
    columns.  A corpus-end ``restated`` lookup would therefore cite both
    operands from the later document, even when the question asks for two
    separately filed periods.  For a same-company period relation, cap each
    operand at the latest filing whose *base period* equals that operand.
    Corrections to that exact report remain visible, while a following
    period's comparison column cannot replace its source document.
    """

    if (len(rows) != 2 or rows[0].corp_code != rows[1].corp_code
            or rows[0].scope != rows[1].scope
            or rows[0].period_end == rows[1].period_end
            or not callable(getattr(companies, "documents", None))):
        return None
    cutoffs: list[str] = []
    for row in rows:
        candidates = [
            document.rcept_dt
            for document in companies.documents(
                as_of=corpus_cutoff, corp_code=row.corp_code,
                doc_group="periodic")
            if getattr(document, "base_year", None) == row.period_end.year
            and getattr(document, "base_month", None) == row.period_end.month
        ]
        if not candidates:
            return None
        cutoffs.append(max(candidates))
    if not callable(getattr(companies, "facts", None)):
        return tuple(cutoffs)
    for row, cutoff in zip(rows, cutoffs, strict=True):
        # A company that listed recently files its first annual report with
        # the prior year only as a comparison column.  Pinning that operand to
        # its own filing then proves nothing, and the request closes as if the
        # figure did not exist.  Fall back to the ordinary corpus-end restated
        # lookup for *both* operands so the answer cites the report that does
        # carry the value instead of denying it.
        available = any(
            getattr(fact, "period_end", None) == row.period_end.isoformat()
            and getattr(fact, "scope", None) == row.scope
            for fact in companies.facts(
                row.corp_code, as_of=cutoff, concept=row.concept.value)
        )
        if not available:
            return None
    return tuple(cutoffs)


def _instant_fiscal_year_end_operand_as_of(
        rows: "tuple[FinancialCoordinate, ...]", *, companies: Any,
        corpus_cutoff: str, operand_views: "tuple[str, ...] | None" = None,
        ) -> "tuple[str | None, ...] | None":
    """Per-coordinate fiscal-year-end pin for a point-in-time (``instant``) row.

    이슈 #149 — 「KB금융의 2025년 연결 기준 당기법인세부채는 얼마인가?」는
    ``period_end=2025-12-31`` 자체는 이미 맞게 좁혔지만, 그 좌표의 ``as_of``
    는 여전히 ``corpus_cutoff`` 였다. 2026 1분기 분기보고서도 「전기말」
    비교열로 2025-12-31 시점 사실을 담고 있어, 그 문서가 FY2025 사업보고서
    보다 나중에 제출됐다는 이유만으로 「최신(``restated``)」인용이 그
    분기보고서를 골랐다 — 사용자는 「2025년」을 물었지 「2026년 1분기」를
    묻지 않았고, 그 비교열이 정정으로 갱신되지 않았다고 단정할 근거도 없다.

    `_period_bound_operand_as_of` 가 **서로 다른 기간의 짝**에 하는 것과
    같은 방식으로, 시점 개념의 **회계연도말(12월 31일)** 좌표는 그 회계연도
    자체의 사업보고서 계열(과 그 정정)로 문서 검색을 좁힌다. 분기말 좌표
    (``period_end.month != 12``)나 duration 개념은 손대지 않는다 — 분기
    표현은 종전대로 두라는 요구사항이고, 기간이 다른 duration 짝은 이미
    `_period_bound_operand_as_of` 의 몫이다.

    좌표별로 독립적으로 판단한다 — 연결·별도 같은 시점을 비교하는 질문
    (DEV-FDR-006 류)에서도 각 좌표가 자기 scope 안에서 자기 연도 사업
    보고서로 좁혀야 한다. 그 연도의 사업보고서 계열 자체가 없거나(상장
    초기), 그 계열의 어느 문서에도 이 좌표의 사실이 없으면 그 좌표는
    손대지 않고 ``None`` 을 돌려준다.

    **문서 날짜가 아니라 그 문서가 실제로 낸 사실로 최댓값을 고른다.**
    같은 회계연도의 정정이 코퍼스 컷오프 바로 앞에서 접수됐지만 사실
    추출이 아직 안 된 경우(예: KB금융 FY2025 20260619000667,
    ``newer_annual_fact_extract_unsupported``)가 실측에 있다. `documents()`
    날짜만으로 최댓값을 고르면 그 미추출 정정의 접수일이 사실상 컷오프와
    같아져 pin이 그대로 무력화되고, 그 사이에 낀 무관한 후속 분기보고서가
    다시 걸린다. `facts()` 가 실제로 돌려주는 행의 `rcept_dt`(그 사실을
    낸 문서 자체)만 후보로 삼으면 미추출 정정은 애초에 후보에 없고, 사실을
    낸 마지막 정정(또는 원본)이 그대로 최댓값이 된다.

    이슈 #183 — 「카카오의 2023년 말 연결 비유동자산 중 유형자산을 최초
    제출값과 최신 재작성값으로 나눠서 비교해줘」(EG-007)는 두 operand가
    같은 좌표를 가리키면서 ``operand_views`` 로 「as_filed」·「restated」
    축만 다르다(`_question_grounded_view_pair`). 그 회계연도(2023) 사업
    보고서 계열로 좁히는 이 pin을 그대로 적용하면 restated 쪽도 2023
    사업보고서 계열 안(원본 또는 그 정정)에 갇혀, 실제 최신 재작성값이
    실린 **후속 연도** 사업보고서(예: 2026 사업보고서의 2023-12-31 재작성
    비교열)를 보지 못한다 — 「최신 재작성값」은 그 정의상 코퍼스 컷오프까지
    열어 둬야 한다.

    두 operand의 as_of를 다르게 내지는 않는다 — ``FinancialComparisonResolution``
    의 view-comparison 불변식(``agent.deterministic_plan_compiler_v1``)이
    as_filed·restated 두 operand의 as_of가 **동일**해야 한다고 강제한다.
    그래서 restated 축이 섞인 pair는 두 operand 모두 pin을 걸지 않고
    코퍼스 컷오프(종전 동작)로 둔다 — 「최초 제출값」(as_filed) 쪽은 그
    자체로 항상 **가장 이른** 사실을 고르는 view 이므로(그 안내는 view
    선택 로직의 몫이지 여기 as_of pin의 몫이 아니다) pin 없이도 여전히
    원본 문서를 그대로 가리킨다. 이 축이 없는 보통 질문(#149 의 CG-031)은
    ``operand_views`` 가 ``None`` 이라 이 예외를 타지 않고 종전대로 두
    operand 모두 그 회계연도 사업보고서 계열로 고정된다.
    """

    if (operand_views is not None
            and len(operand_views) == len(rows)
            and "restated" in operand_views):
        return None
    if not (callable(getattr(companies, "documents", None))
            and callable(getattr(companies, "facts", None))):
        return None
    overrides: list[str | None] = []
    for row in rows:
        if (row.period_end.month != 12
                or concept_axes(row.concept).period_semantics != "instant"):
            overrides.append(None)
            continue
        own_doc_ids = {
            document.doc_id
            for document in companies.documents(
                as_of=corpus_cutoff, corp_code=row.corp_code,
                doc_group="periodic")
            if getattr(document, "base_year", None) == row.period_end.year
            and getattr(document, "base_month", None) == 12
        }
        if not own_doc_ids:
            overrides.append(None)
            continue
        own_fact_rcept_dts = [
            fact.rcept_dt
            for fact in companies.facts(
                row.corp_code, as_of=corpus_cutoff, concept=row.concept.value)
            if getattr(fact, "doc_id", None) in own_doc_ids
            and getattr(fact, "period_end", None) == row.period_end.isoformat()
            and getattr(fact, "scope", None) == row.scope
        ]
        overrides.append(max(own_fact_rcept_dts) if own_fact_rcept_dts else None)
    return tuple(overrides) if any(value is not None for value in overrides) else None


def _parallel_annual_change_operator(
        intent: SemanticIntent, item_index: int,
        ) -> "str | None":
    """Bind split amount/rate demands only when their financial axes agree.

    Providers sometimes preserve ``how much`` and ``what percent`` as two
    scalar answer items.  The first can remain a ``retrieve`` item even though
    both items carry the same two annual endpoints after bounded repair.  This
    helper recognizes only that closed two-item topology and assigns one
    operator to each source field.  Different companies, metrics, period or
    scope axes are deliberately not combined.
    """

    if len(intent.answer_items) != 2 or item_index not in {0, 1}:
        return None
    first, second = intent.answer_items
    if any(
            item.target.kind != "metric"
            or len(item.target.entity_refs) != 1
            or item.output.shape != "scalar"
            or item.output.projection_mode != "named_fields"
            or len(item.output.field_surfaces) != 1
            or len(item.scope.target_period_expressions) != 2
            or item.selection is not None
            for item in (first, second)
    ):
        return None
    if (
            first.target.surface != second.target.surface
            or first.target.entity_refs != second.target.entity_refs
            or first.scope != second.scope
    ):
        return None
    roles: list[str | None] = []
    for item in (first, second):
        surface = item.output.field_surfaces[0]
        percent = _PERCENT_CHANGE_FIELD.search(surface) is not None
        amount = _AMOUNT_CHANGE_FIELD.search(surface) is not None
        roles.append(
            "percent_change" if percent and not amount else
            "difference" if amount and not percent else None)
    if set(roles) != {"difference", "percent_change"}:
        return None
    return roles[item_index]


#: 이슈 #38 — 이름 있는 비율의 결정적 사전. 분자 concept · 분모 concept ·
#: presentation("percent" ×100 `%` · "multiple" 그대로 `배`). 새 이름을
#: 추가하려면 여기와 `agent/concept_alias.NAMED_RATIO_SURFACES` 를 함께
#: 갱신해야 한다 — 그쪽은 콜로퀴얼 BLOCK 을 건너뛸지만 정하고, 분자·분모
#: 개념은 여기서 정한다.
NAMED_RATIO_CONCEPTS: "dict[str, tuple[FinancialConcept, FinancialConcept, str]]" = {
    normalize_surface_key("영업이익률"): (
        FinancialConcept.OPERATING_INCOME, FinancialConcept.REVENUE, "percent"),
    normalize_surface_key("순이익률"): (
        FinancialConcept.NET_INCOME, FinancialConcept.REVENUE, "percent"),
    normalize_surface_key("매출원가율"): (
        FinancialConcept.COST_OF_SALES, FinancialConcept.REVENUE, "percent"),
    normalize_surface_key("부채비율"): (
        FinancialConcept.TOTAL_LIABILITIES, FinancialConcept.TOTAL_EQUITY, "percent"),
    normalize_surface_key("유동비율"): (
        FinancialConcept.CURRENT_ASSETS, FinancialConcept.CURRENT_LIABILITIES, "percent"),
    normalize_surface_key("자기자본비율"): (
        FinancialConcept.TOTAL_EQUITY, FinancialConcept.TOTAL_ASSETS, "percent"),
    normalize_surface_key("ROE"): (
        FinancialConcept.NET_INCOME, FinancialConcept.TOTAL_EQUITY, "percent"),
    normalize_surface_key("ROA"): (
        FinancialConcept.NET_INCOME, FinancialConcept.TOTAL_ASSETS, "percent"),
}

#: 「A를 B로 나눈」·「A가 B의 몇 배」·「A 대비 B 비율」— 이름 사전 밖 개념 쌍을
#: **정본 계정 사전으로만** 닫는다. 세 문형은 기본 presentation="multiple" 이다
#: — 률·%·퍼센트 단서가 있으면 위 이름 있는 비율이나
#: `_time_relation_operators` 가 이미 답하는 경우가 많지만, 「A를 B로 나눈
#: 값은 몇 퍼센트인가?」처럼 이름 없는 나눗셈에 직접 퍼센트를 요청하는
#: 문형(#170)도 있다 — 이때는 아래에서 질문 원문의 퍼센트 단서를 보고
#: presentation 을 percent 로 뒤집는다.
#:
#: **원문 질문에** 매칭한다(공백을 지우는 `normalize_surface_key` 이전) —
#: 한국어는 어절 사이 공백이 유일한 경계다. `(?:^|\s)` 로 각 operand 의
#: 시작을 어절 경계에 고정하고 비탐욕(`+?`)으로 넓혀, 「가온의 2031년 연결
#: 영업이익을」처럼 앞의 회사·기간·scope 수식어가 A 에 함께 딸려 오지 않게
#: 막는다 — 앵커가 없으면 처음 나오는 조사까지 전부 삼킨다.
_DIVIDE_BY_PATTERN = re.compile(
    r"(?:^|\s)(?P<a>[가-힣a-zA-Z0-9]+?)(?:을|를)\s*"
    r"(?P<b>[가-힣a-zA-Z0-9]+?)(?:으로|로)\s*(?:나눈|나누)")
_MULTIPLE_OF_PATTERN = re.compile(
    r"(?:^|\s)(?P<a>[가-힣a-zA-Z0-9]+?)(?:이|가)\s*"
    r"(?P<b>[가-힣a-zA-Z0-9]+?)의\s*몇\s*배")
_PERCENT_OF_PATTERN = re.compile(
    r"(?:^|\s)(?P<a>[가-힣a-zA-Z0-9]+?)(?:은|는|이|가)\s*"
    r"(?P<b>[가-힣a-zA-Z0-9]+?)의\s*몇\s*(?:퍼센트|%)")
_VERSUS_RATIO_PATTERN = re.compile(
    r"(?:^|\s)(?P<a>[가-힣a-zA-Z0-9]+?)\s*대비\s*"
    r"(?P<b>[가-힣a-zA-Z0-9]+?)\s*(?:의\s*)?비율")
#: 이슈 #59 1단계 — 「삼성전자의 2025년 연결 매출액은 SK하이닉스의 몇
#: 배인가?」. 위 `_MULTIPLE_OF_PATTERN` 은 **같은 회사**의 서로 다른 두
#: 개념(「A가 B의 몇 배」)만 찾는다. 여기서는 회사 자체가 다른 배수 요청을
#: 넓게 잡는다 — 개념은 하나(질문의 target concept)이고 companies가 둘이면
#: 충분하다. 좁히는 일은 이 정규식이 아니라 호출자(`_payload_for`)가
#: `len(rows) == 2 and 서로 다른 corp_code` 조건으로 한다.
_CROSS_COMPANY_MULTIPLE_PATTERN = re.compile(r"몇\s*배")

#: 이슈 #124(합계) — 「삼성전자와 SK하이닉스의 2025년 연결 매출액 합계는
#: 얼마인가?」. 위 `_CROSS_COMPANY_MULTIPLE_PATTERN`과 같은 자리 — 회사가
#: 둘이고 개념·기간·scope가 같으면(호출자 `_payload_for`의
#: `len(rows) == 2` + coordinate 동등성 검사가 이미 강제) 이 문구가 그
#: 둘을 더하라는 뜻이다. 「평균」은 일부러 잡지 않는다 — v0.4는 합계만
#: 승인했다.
_CROSS_COMPANY_SUM_PATTERN = re.compile(r"합계|합산")

#: 이슈 #170 — 이름 없는 나눗셈/배수/비율 문형이라도 질문에 명시적 퍼센트
#: 단서(「몇 퍼센트」「%」「비율(퍼센트)」)가 있으면 배수 대신 퍼센트로 낸다.
#: 「몇 배」자체가 이미 배수를 명시하므로 그 문형과 겹칠 일은 실질적으로
#: 없다 — 문장에 두 단서가 동시에 있으면 퍼센트 단서를 우선한다.
_RATIO_PERCENT_CUE_PATTERN = re.compile(r"퍼센트|%")


def _explicit_ratio_concepts(
        question: str) -> "tuple[FinancialConcept, FinancialConcept, str] | None":
    """질문 문장에서 명시적 나눗셈의 두 정본 개념을 직접 읽는다.

    기존 concept alias 해석(`resolve_metric_concept`)을 그대로 재사용한다 —
    표기를 새로 적지 않는다.
    """

    for pattern in (
            _DIVIDE_BY_PATTERN, _MULTIPLE_OF_PATTERN, _PERCENT_OF_PATTERN,
            _VERSUS_RATIO_PATTERN):
        match = pattern.search(question)
        if match is None:
            continue
        numerator = resolve_metric_concept(match.group("a"))
        denominator = resolve_metric_concept(match.group("b"))
        if numerator is None or denominator is None or numerator is denominator:
            continue
        presentation = ("percent" if _RATIO_PERCENT_CUE_PATTERN.search(question)
                         else "multiple")
        return numerator, denominator, presentation
    return None


def _ratio_request(
        question: str, item: Any,
        ) -> "tuple[FinancialConcept, FinancialConcept, str] | None":
    """비율·배수 질문을 분자·분모 두 정본 개념 + presentation 으로 되돌린다.

    사전 밖 조합(회전율·주당배당금 등)은 그대로 ``None`` — 호출자는 종전처럼
    개념을 못 찾은 것으로 다룬다. **여기서 개념 쌍이 나와도** 실제
    ``financial_comparison`` 은 기간·scope·회사가 하나로 닫힐 때만 만들어진다
    (이 함수 밖, `_ratio_coordinates`/`FinancialResolutionBackend.resolve()` 의 몫).
    """

    if item.target.kind != "metric":
        return None
    question_key = normalize_surface_key(question)
    for name, ratio in NAMED_RATIO_CONCEPTS.items():
        if name in question_key:
            return ratio
    return _explicit_ratio_concepts(question)


def _ratio_coordinates(
        item: Any, *, intent: SemanticIntent, companies: Any,
        reference_date: date, corpus_cutoff: str, scope_authority: Any = None,
        numerator: "FinancialConcept", denominator: "FinancialConcept",
        ) -> "tuple[FinancialCoordinate, FinancialCoordinate] | None":
    """비율 질문의 분자·분모 좌표. 같은 회사·같은 기간·같은 scope 하나로만 닫는다."""

    companies_surfaces = _company_surfaces(intent, item)
    periods = list(item.scope.target_period_expressions or ())
    if len(companies_surfaces) != 1 or len(periods) != 1:
        return None
    company_surface = companies_surfaces[0]
    period_expression = periods[0]
    scope_expressions = [
        *item.scope.scope_qualifier_expressions,
        *item.target.qualifier_surfaces,
        item.target.surface,
    ]
    joined_scope = normalize_surface_key(" ".join(scope_expressions))
    explicit_scopes = []
    if any(normalize_surface_key(cue) in joined_scope for cue in _CFS_CUES):
        explicit_scopes.append("CFS")
    if any(normalize_surface_key(cue) in joined_scope for cue in _SFS_CUES):
        explicit_scopes.append("SFS")
    if len(explicit_scopes) > 1:
        return None    # 연결·별도를 동시에 요구하면 비율 하나로 닫히지 않는다
    scope_override = explicit_scopes[0] if explicit_scopes else None

    numerator_row = _one_coordinate(
        item, company_surface=company_surface, period_expression=period_expression,
        companies=companies, reference_date=reference_date,
        corpus_cutoff=corpus_cutoff, scope_authority=scope_authority,
        scope_override=scope_override, concept_override=numerator)
    denominator_row = _one_coordinate(
        item, company_surface=company_surface, period_expression=period_expression,
        companies=companies, reference_date=reference_date,
        corpus_cutoff=corpus_cutoff, scope_authority=scope_authority,
        scope_override=scope_override, concept_override=denominator)
    if numerator_row is None or denominator_row is None:
        return None
    if (numerator_row.scope != denominator_row.scope
            or numerator_row.period_end != denominator_row.period_end):
        # 기본값(주재무제표)이 분자·분모에서 갈리거나, 같은 기간 표현이 두
        # 개념에서 서로 다른 날짜로 풀리면 비율 하나로 닫히지 않는다.
        return None
    return numerator_row, denominator_row


def _ratio_payload_for(
        numerator_row: "FinancialCoordinate", denominator_row: "FinancialCoordinate",
        *, as_of: str, view: str, presentation: str,
        ) -> "dict[str, Any]":
    """비율 두 operand 를 `financial_comparison`/`concept_ratio` payload 로."""

    return {
        "kind": "financial_comparison",
        "requested_operators": ["concept_ratio"],
        "presentation": presentation,
        "verification_claim": None,
        "verification_premise_id": None,
        "operands": [
            {
                "kind": "financial_operand",
                "operand_id": f"operand-{position}",
                "proof_ref": f"source-operand:operand-{position}",
                **{k: v for k, v in row.as_resolution(as_of=as_of, view=view).items()
                   if k != "kind"},
            }
            for position, row in enumerate(
                (numerator_row, denominator_row), start=1)
        ],
    }


def _payload_for(
        rows: "tuple[FinancialCoordinate, ...]", *, as_of: str, view: str,
        question: str, operation: str, output_field_count: int,
        field_surfaces: tuple[str, ...] = (),
        operand_as_of: "tuple[str, ...] | None" = None,
        operand_views: "tuple[str, ...] | None" = None,
        parallel_change_operator: "str | None" = None,
        verification_premise: "tuple[str, str] | None" = None,
        direction: str = "maximum",
        ) -> "dict[str, Any] | None":
    """좌표 묶음을 계약이 정한 payload 로 담는다.

    - 좌표 하나 → `financial`
    - 좌표 둘 → `financial_comparison`
      - 회사가 다르면 같은 좌표의 회사 비교다. 승자와 비방향 차이를 모두
        묻는 비교는 ``argmax``와 ``absolute_difference``로 낮춘다. operand
        순서는 질문의 회사 순서일 뿐 차이의 부호를 뜻하지 않는다.
      - 회사가 같으면 서로 다른 기간의 typed financial derivation 이다.
        둘 다 정본 좌표만 담고, 실제 derivation operator는 compiler가
        `SemanticIntent.operation`과 좌표 관계로 선택한다.
    - 회사 3~8개 → 모든 원값 operand와 N-ary ``argmax``를 보존한다.
    - 그 밖 → `None`.
    """

    if len(rows) == 1:
        pinned = operand_as_of[0] if operand_as_of is not None else None
        return rows[0].as_resolution(
            as_of=(pinned if pinned is not None else as_of), view=view)
    if len(rows) not in {2, 3, 4, 5, 6, 7, 8}:
        return None
    if operand_views is not None and len(operand_views) != len(rows):
        return None
    effective_views = operand_views or tuple(view for _row in rows)
    first, second = rows[0], rows[1]
    if len(rows) > 2 and first.corp_code == second.corp_code:
        return None
    if first.corp_code != second.corp_code:
        if len({row.corp_code for row in rows}) != len(rows):
            return None
        for row in rows[1:]:
            if any(getattr(first, field) != getattr(row, field) for field in (
                    "concept", "period_start", "period_end", "period_type",
                    "scope", "statement", "cumulative")):
                return None
    else:
        # Same-company relations are meaningful only for the same accounting
        # axis at distinct endpoints.  Accepting two identical facts would
        # invent a calculation with no semantic difference.
        if first.concept != second.concept or first.statement != second.statement:
            return None
        same_period = all(
            getattr(first, field) == getattr(second, field)
            for field in ("period_start", "period_end", "period_type", "cumulative"))
        same_scope = first.scope == second.scope
        same_view = effective_views[0] == effective_views[1]
        if sum((not same_period, not same_scope, not same_view)) != 1:
            # Exactly one axis must differ: period, statement scope, or view.
            return None
    same_company = len({row.corp_code for row in rows}) == 1
    if same_company and first.scope != second.scope:
        requested_operators = ["absolute_difference"]
    elif same_company and effective_views[0] != effective_views[1]:
        if operation != "compare":
            return None
        requested_operators = ["difference"]
    elif same_company:
        if operation == "retrieve":
            # Annual endpoints are independent full-year facts, never
            # cumulative interim columns to subtract.  A provider may still
            # label a change demand as ``retrieve``; bind its explicit field
            # role instead of emitting ``discrete_from_cumulative``.
            if all(
                    row.period_type == "annual"
                    or (row.period_type == "instant"
                        and row.period_end.month == 12)
                    for row in rows):
                requested_operators = (
                    [parallel_change_operator]
                    if parallel_change_operator is not None else
                    _time_relation_operators(
                        question=question, field_surfaces=field_surfaces)
                )
                if requested_operators is None:
                    return None
            else:
                requested_operators = ["discrete_from_cumulative"]
        else:
            requested_operators = _time_relation_operators(
                question=question, field_surfaces=field_surfaces)
            if requested_operators is None:
                return None
    else:
        verification_only = bool(re.search(
            r"(?:컸지|작았지|많았지|적었지|맞지)\s*[?？]?$", question.strip()))
        # 이슈 #59 1단계 — 「삼성전자의 2025년 연결 매출액은 SK하이닉스의
        # 몇 배인가?」. 같은 concept·같은 기간의 두 회사 operand(위
        # 코디네이트 동등성 검사가 이미 강제했다)를 배수 질문 문구가 물으면
        # argmax 대신 concept_ratio 로 낮춘다 — 순위/승자가 아니라 나눗셈
        # 값 하나를 요청하는 것이라 답 모양 자체가 다르다.
        cross_company_ratio = (
            len(rows) == 2 and output_field_count == 1
            and _CROSS_COMPANY_MULTIPLE_PATTERN.search(question) is not None
        )
        # 이슈 #124(합계) — 「삼성전자와 SK하이닉스의 2025년 연결 매출액
        # 합계는 얼마인가?」. cross_company_ratio와 같은 자리 — 개념 하나·
        # 회사 둘·요청 필드 하나에 「합계」 문구가 있으면 argmax(승자) 대신
        # sum으로 낮춘다. 답 모양이 승자·차이가 아니라 덧셈 값 하나다.
        cross_company_sum = (
            len(rows) == 2 and output_field_count == 1
            and _CROSS_COMPANY_SUM_PATTERN.search(question) is not None
        )
        # A two-company comparison's operand order is source order, not a
        # direction chosen by the user.  Its requested gap is consequently a
        # distance.  One requested field is the winner alone (including a
        # premise verification); two fields are winner plus that distance.
        # Keep this tied to the typed output arity rather than Korean company
        # names or a question ID.
        if cross_company_ratio:
            requested_operators = ["concept_ratio"]
            verification_only = False
        elif cross_company_sum:
            requested_operators = ["sum"]
            verification_only = False
        elif len(rows) > 2 and operation == "compare" \
                and output_field_count in {1, 2}:
            # Stage2 applies the full competition-ranking presentation policy
            # over these existing N-ary argmax operands.  No pairwise gap is
            # synthesized unless the public plan explicitly requests one.
            requested_operators = ["argmax"]
            verification_only = False
        elif len(rows) > 2:
            return None
        elif verification_only or output_field_count == 1:
            requested_operators = ["argmax"]
        elif output_field_count == 2:
            requested_operators = ["argmax", "absolute_difference"]
        else:
            return None
    return {
        "kind": "financial_comparison",
        "requested_operators": requested_operators,
        "presentation": (
            "multiple" if requested_operators == ["concept_ratio"] else None
        ),
        # 이슈 #124 — "argmax" 가 없으면 방향은 뜻이 없으므로 항상 "maximum"
        # 으로 고정한다(authority validator와 같은 규칙).
        "direction": direction if "argmax" in requested_operators else "maximum",
        "verification_claim": (
            verification_premise[1]
            if not same_company and verification_only
            and verification_premise is not None else None
        ),
        "verification_premise_id": (
            verification_premise[0]
            if not same_company and verification_only
            and verification_premise is not None else None
        ),
        "operands": [
            {
                "kind": "financial_operand",
                "operand_id": f"operand-{position}",
                "proof_ref": f"source-operand:operand-{position}",
                **{k: v for k, v in row.as_resolution(
                    as_of=(
                        operand_as_of[position - 1]
                        if operand_as_of is not None
                        and operand_as_of[position - 1] is not None
                        else as_of),
                    view=effective_views[position - 1]).items() if k != "kind"},
            }
            for position, row in enumerate(rows, start=1)
        ],
    }


def _single_quarter_cumulative_coordinates(
        item: Any, *, intent: SemanticIntent, companies: Any,
        reference_date: date, corpus_cutoff: str, scope_authority: Any = None,
        question: str = "", period_override: str | None = None,
        ) -> "tuple[FinancialCoordinate, ...]":
    """Recover the two cumulative endpoints for an explicit single quarter.

    An interim income/cash-flow figure is commonly reported cumulatively.  A
    request whose *semantic target* explicitly says ``단일분기`` therefore
    needs the current cumulative endpoint minus its preceding cumulative
    endpoint.  This is a typed calendar rule, not a question-ID or answer
    overlay:

    ``Q2 -> H1 - Q1``, ``Q3 -> 9M - H1``, ``Q4 -> FY - 9M``.

    The normalizer sometimes also emits a redundant bare year beside the
    explicit quarter.  Only the explicit non-cumulative quarter is used; the
    bare year is not a second requested fact in this derivation shape.
    """

    if item.operation != "retrieve":
        return ()
    if len(_company_surfaces(intent, item)) != 1:
        return ()

    candidates: list[FinancialCoordinate] = []
    target_key = normalize_surface_key(item.target.surface)
    expressions = list(item.scope.target_period_expressions)
    if period_override is not None:
        # This override is supplied only by RecentPeriodsFanout after it has
        # proved that the relative-period request denotes a discrete series
        # and that both cumulative source endpoints exist.  Do not weaken the
        # ordinary requirement for a literal 단일/단독 cue.
        if _EXPLICIT_QUARTER.fullmatch(period_override.strip()) is None:
            return ()
        expressions = [period_override]
    elif "단일분기" not in target_key and "단독" not in target_key:
        # The v1 semantic boundary can preserve an abbreviated period as a
        # target qualifier (or only the user's explicit ``H1 cumulative - Q1``
        # expression), without placing it on the generic scope axis.  Recover
        # it only from that question-grounded arithmetic/anti-cumulative
        # topology.  A supplied scope period remains authoritative; this
        # branch never rewrites it.
        recovered = _question_grounded_single_quarter_expression(question)
        if recovered is not None:
            if expressions:
                normalized_recovered = normalize_surface_key(recovered)
                for value in expressions:
                    if (re.fullmatch(
                            r"(?:20)?[0-9]{2}\s*년(?:도)?", value.strip())
                            is None
                            and normalize_surface_key(value)
                            != normalized_recovered):
                        return ()
            expressions = [recovered]
        else:
            # 「직전 분기」— 이 표현은 이미 `_one_coordinate` 가 받는 리터럴
            # (질문의 부분문자열)이지만, 실제로 어느 분기인지는 정본의 최신
            # 실제 분기에서 계산해야 안다.  그 분기가 Q2~Q4로 떨어지면
            # (=단일 좌표로 직접 조회할 수 없는 파생 분기) 여기서 계산한
            # 리터럴 분기 표현으로 완전히 갈아 끼운다 — 위 일치 검사는
            # 「질문에 이미 있던 리터럴과 일치해야 한다」는 것이라
            # 원래부터 다른 이 표현에는 적용되지 않는다.
            recovered = _relative_previous_quarter_expression(
                item, intent=intent, companies=companies,
                corpus_cutoff=corpus_cutoff)
            if recovered is None:
                return ()
            expressions = [recovered]
    else:
        recovered = _question_grounded_single_quarter_expression(question)
        if recovered is not None:
            if not expressions or all(
                    re.fullmatch(r"(?:20)?[0-9]{2}\s*년(?:도)?", value.strip())
                    is not None
                    for value in expressions):
                expressions = [recovered]

    for expression in expressions:
        concept_override = None
        explicit_quarter = _EXPLICIT_QUARTER.fullmatch(expression.strip())
        standalone_target = re.fullmatch(
            r"단독\s+(.+)", _bare_concept_surface(item.target.surface))
        if (explicit_quarter is not None
                and explicit_quarter.group("quarter") in {"2", "3", "4"}
                and standalone_target is not None):
            # A literal single-quarter target may retain '단독' outside its
            # period axis. Resolve only its known account here, never in the
            # global account dictionary or the annual/cumulative path.
            concept_override = resolve_metric_concept(
                _bare_concept_surface(standalone_target.group(1)))
        row = _one_coordinate(
            item, company_surface=_company_surfaces(intent, item)[0],
            period_expression=expression, companies=companies,
            reference_date=reference_date, corpus_cutoff=corpus_cutoff,
            scope_authority=scope_authority, concept_override=concept_override)
        if row is not None and row.period_start is not None and not row.cumulative:
            candidates.append(row)
    if len(candidates) != 1:
        return ()

    discrete = candidates[0]
    # Q1 is already its own cumulative duration and needs no subtraction.
    endpoint = discrete.period_end
    predecessor_endpoints = {
        6: (date(endpoint.year, 3, 31), "quarter"),
        9: (date(endpoint.year, 6, 30), "half"),
        12: (date(endpoint.year, 9, 30), "quarter"),
    }
    previous = predecessor_endpoints.get(endpoint.month)
    if previous is None:
        return ()
    previous_end, previous_type = previous
    current_type = {6: "half", 9: "quarter", 12: "annual"}[endpoint.month]
    current = FinancialCoordinate(
        corp_code=discrete.corp_code, corp_name=discrete.corp_name,
        concept=discrete.concept, period_start=date(endpoint.year, 1, 1),
        period_end=endpoint, period_type=current_type, cumulative=True,
        scope=discrete.scope, statement=discrete.statement,
        scope_was_explicit=discrete.scope_was_explicit,
    )
    prior = FinancialCoordinate(
        corp_code=discrete.corp_code, corp_name=discrete.corp_name,
        concept=discrete.concept, period_start=date(endpoint.year, 1, 1),
        period_end=previous_end, period_type=previous_type, cumulative=True,
        scope=discrete.scope, statement=discrete.statement,
        scope_was_explicit=discrete.scope_was_explicit,
    )
    return current, prior


_EXPLICIT_QUARTER = re.compile(
    r"(?<![0-9])(?P<year>(?:20)?[0-9]{2})\s*년?\s*"
    r"(?P<quarter>[1-4])\s*(?:분기|q)(?![0-9A-Za-z])",
    flags=re.IGNORECASE,
)
_QUESTION_YEAR = re.compile(r"(?<![0-9])(?P<year>20[0-9]{2})\s*년")
_QUESTION_QUARTER = re.compile(
    r"(?<![0-9])(?P<quarter>[1-4])\s*(?:분기|q)(?![0-9A-Za-z])",
    flags=re.IGNORECASE,
)
_DISCRETE_QUARTER = re.compile(
    r"(?:(?P<quarter_before>[2-4])\s*(?:분기|q)\s*(?:단독|단일)"
    r"|(?:단독|단일)\s*(?P<quarter_after>[2-4])\s*(?:분기|q))",
    flags=re.IGNORECASE,
)
_HALF_MINUS_Q1 = re.compile(
    r"(?<![0-9])(?P<year>(?:20)?[0-9]{2})\s*년?\s*"
    r"(?:상반기|반기).*?누적.*?1\s*(?:분기|q).*?"
    r"(?:빼|차감|제외|minus|-)",
    flags=re.IGNORECASE,
)


def _question_grounded_single_quarter_expression(question: str) -> "str | None":
    """Return one canonical discrete-quarter expression from explicit meaning.

    The semantic boundary intentionally does not invent accounting axes.  It
    may, however, retain a colloquial period in a target qualifier instead of
    the scope-period field.  This recognises only two executable statements:
    an explicit Q2-Q4 with an anti-cumulative/single-quarter cue, or
    the unambiguous ``H1 cumulative minus Q1`` arithmetic.  It is independent
    of company, concept, question ID, and fixture data.
    """

    compact = normalize_surface_key(question)
    if not compact:
        return None
    arithmetic = _HALF_MINUS_Q1.search(question)
    if arithmetic is not None:
        return f"{_four_digit_year(arithmetic.group('year'))}년 2분기"

    # A bare quarter may still mean the cumulative interim column.  Require
    # an explicit user cue before changing it into a discrete derivation.
    if not any(cue in compact for cue in (
            "단일분기", "단일", "단독", "분기만",
            "누적말고", "누적아닌", "누적제외")):
        return None
    for match in _EXPLICIT_QUARTER.finditer(question):
        if match.group("quarter") in {"2", "3", "4"}:
            return (
                f"{_four_digit_year(match.group('year'))}년 "
                f"{match.group('quarter')}분기"
            )
    years = list(dict.fromkeys(
        match.group("year") for match in _QUESTION_YEAR.finditer(question)))
    discrete_quarters = list(dict.fromkeys(
        match.group("quarter_before") or match.group("quarter_after")
        for match in _DISCRETE_QUARTER.finditer(question)))
    if len(years) == 1 and len(discrete_quarters) == 1:
        return f"{years[0]}년 {discrete_quarters[0]}분기"
    quarters = list(dict.fromkeys(
        match.group("quarter") for match in _QUESTION_QUARTER.finditer(question)))
    if len(years) == 1 and len(quarters) == 1 and quarters[0] in {"2", "3", "4"}:
        return f"{years[0]}년 {quarters[0]}분기"
    return None


def _four_digit_year(surface: str) -> int:
    year = int(surface)
    return year if len(surface) == 4 else 2000 + year


__all__ = [
    "FinancialCoordinate", "FinancialResolutionBackend", "financial_coordinates",
]
