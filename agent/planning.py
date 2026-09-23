"""고정 Draft fixture를 검증된 QueryPlan으로 바꾸는 결정론적 adapter."""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date
import re
from pathlib import Path
import unicodedata
from typing import Protocol

from pydantic import Field

from .contracts import (
    ClarificationOption,
    ClarificationRequest,
    reason_token,
    ContractModel,
    DateRange,
    DocumentSelector,
    EventSelector,
    FactSpec,
    FieldOutputSpec,
    FinancialConcept,
    PlanValidation,
    ResolvedDisclosureTask,
    ResolvedDocumentTask,
    ResolvedCorrectionTask,
    ResolvedEventTask,
    ResolvedFinancialTask,
    ResolvedNarrativeTask,
    ResolvedQueryPlan,
    validate_as_of,
)
from .drafts import (
    DraftCorrectionTask,
    DraftDisclosureTask,
    DraftDocumentTask,
    DraftEventTask,
    DraftFinancialTask,
    DraftNarrativeTask,
    DraftQueryPlan,
)
from .concept_alias import (
    load_concept_question_patterns,
    normalize_surface_key,
)
from .correction_preflight import CorrectionSeedPreflight
from .date_surface import expand_two_digit_year, question_date_surfaces
from .event_preflight import EventKeyPreflight
from .periodic_document_preflight import PeriodicDocumentPreflight


_DOCUMENT_GROUPS = frozenset({"periodic", "exchange", "major", "holding"})
_FUNDING_DECISION_ALIASES = {
    "유상증자": "유상증자",
    "CB": "전환사채",
    "전환사채": "전환사채",
    "BW": "신주인수권부사채",
    "신주인수권부사채": "신주인수권부사채",
    "EB": "교환사채",
    "교환사채": "교환사채",
}


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", value).strip())


def funding_decision_keywords(surface: str) -> tuple[str, ...]:
    """Lower an explicit funding-type enumeration to canonical keywords.

    The conversion is all-or-nothing: every token must be a known funding
    instrument and at least two distinct types must be present.  Ordinary
    event types therefore remain untouched.
    """

    if not isinstance(surface, str):
        return ()
    parts = [
        _compact(part)
        for part in re.split(r"\s*[·ㆍ/,]\s*", surface)
        if part.strip()
    ]
    if len(parts) < 2:
        return ()
    result: list[str] = []
    for part in parts:
        canonical = _FUNDING_DECISION_ALIASES.get(part.upper())
        if canonical is None:
            return ()
        if canonical not in result:
            result.append(canonical)
    return tuple(result) if len(result) >= 2 else ()


def _as_yyyymmdd(year: int, month: int, day: int) -> str:
    try:
        return date(year, month, day).strftime("%Y%m%d")
    except ValueError as exc:
        raise ValueError("유효하지 않은 대상기간 날짜입니다") from exc


def _target_date_range(
        expression: str, *, reference_date: date,
        ) -> tuple[str | None, str | None, str | None]:
    """질문의 조회 대상기간을 event/disclosure metadata 범위로 내린다."""

    compact = expand_two_digit_year(_compact(expression))
    relative_years = {
        "올해": reference_date.year,
        "금년": reference_date.year,
        "작년": reference_date.year - 1,
        "지난해": reference_date.year - 1,
        "재작년": reference_date.year - 2,
    }
    if compact in relative_years:
        year = relative_years[compact]
        return f"{year:04d}0101", f"{year:04d}1231", None

    # 「2025년 1-2분기」의 「1-2」는 날짜(1월 2일)가 아니라 분기 범위다 — 뒤에 「분기」가 오면 날짜로 읽지 않는다.
    full_dates = re.findall(
        r"(?<![0-9])(20[0-9]{2})[-./년]([01]?[0-9])[-./월]([0-3]?[0-9])일?(?![0-9]|분기)",
        compact,
    )
    if full_dates:
        if len(full_dates) > 2:
            return None, None, "대상기간 날짜는 시작·종료 두 개까지만 지원합니다"
        try:
            values = [_as_yyyymmdd(*(int(part) for part in row))
                      for row in full_dates]
        except ValueError as exc:
            return None, None, str(exc)
        start, end = (values[0], values[-1])
        if start > end:
            return None, None, "대상기간 시작일은 종료일 이후일 수 없습니다"
        return start, end, None

    years = [int(value) for value in dict.fromkeys(re.findall(
        r"(?<![0-9])(20[0-9]{2})(?:년)?(?![0-9])", compact))]
    # A colloquial event-list selector may say only ``12월``.  Anchor it to
    # the latest such month not after the resolver reference date; the surface
    # remains question-grounded while the execution range is year-bounded.
    month_only = re.fullmatch(r"([01]?[0-9])월", compact)
    if not years and month_only is not None:
        month = int(month_only.group(1))
        if not 1 <= month <= 12:
            return None, None, "대상기간 월은 1~12여야 합니다"
        year = reference_date.year if month <= reference_date.month else reference_date.year - 1
        return (
            f"{year:04d}{month:02d}01",
            _as_yyyymmdd(year, month, monthrange(year, month)[1]),
            None,
        )
    if len(years) > 2:
        return None, None, "대상기간 연도는 시작·종료 두 개까지만 지원합니다"
    if len(years) == 2:
        if years[0] > years[1]:
            return None, None, "대상기간 시작연도는 종료연도 이후일 수 없습니다"
        return f"{years[0]:04d}0101", f"{years[1]:04d}1231", None
    if not years:
        return None, None, "대상기간에는 절대 연도가 필요합니다"

    year = years[0]
    # 「3~4분기」·「1-2분기」 같은 연속 분기 범위 = 시작 분기 첫날 ~ 끝 분기 마지막 날
    quarter_span = re.search(r"([1-4])\s*[~\-–—]\s*([1-4])분기", compact)
    if quarter_span is not None:
        first, last = int(quarter_span.group(1)), int(quarter_span.group(2))
        if first > last:
            return None, None, "대상기간 분기 범위의 시작이 끝보다 늦습니다"
        end_month = last * 3
        return (
            f"{year:04d}{(first - 1) * 3 + 1:02d}01",
            _as_yyyymmdd(year, end_month, monthrange(year, end_month)[1]),
            None,
        )
    quarters = tuple(dict.fromkeys(re.findall(r"([1-4])분기", compact)))
    if len(quarters) > 1:
        return None, None, "단일 selector에 복수 분기를 넣을 수 없습니다"
    if quarters:
        quarter = int(quarters[0])
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        return (
            f"{year:04d}{start_month:02d}01",
            _as_yyyymmdd(year, end_month, monthrange(year, end_month)[1]),
            None,
        )
    if "하반기" in compact:
        return f"{year:04d}0701", f"{year:04d}1231", None
    if "반기" in compact:          # 「상반기」·「반기(보고서)」 모두 1~6월
        return f"{year:04d}0101", f"{year:04d}0630", None
    # 「N월」뿐 아니라 **`2025-12`·`2025.12` 같은 숫자 표기**도 월이다. 예전에는
    # `월` 글자가 있어야만 잡아, `2025-12` 가 **연도 전체**로 넓어졌다. 「12월에
    # 깨진 계약」을 물었는데 한 해치 사건을 다 후보로 올려 역질문이 됐다.
    month_match = re.search(r"(?<![0-9])([01]?[0-9])월(?![0-9])", compact)
    if month_match is None:
        month_match = re.search(
            rf"(?<![0-9]){year:04d}[-./]([01]?[0-9])(?![0-9])", compact)
    if month_match is not None:
        month = int(month_match.group(1))
        if not 1 <= month <= 12:
            return None, None, "대상기간 월은 1~12여야 합니다"
        return (
            f"{year:04d}{month:02d}01",
            _as_yyyymmdd(year, month, monthrange(year, month)[1]),
            None,
        )
    return f"{year:04d}0101", f"{year:04d}1231", None


def _narrative_period_range(
        expression: str, *, reference_date: date,
        ) -> tuple[DateRange | None, str | None]:
    """One grounded periodic selector를 public narrative period 범위로 내린다."""

    start, end, error = _target_date_range(
        expression, reference_date=reference_date)
    if error is not None or start is None or end is None:
        return None, error or "narrative 대상기간을 범위로 바꿀 수 없습니다"
    return DateRange(
        start=date(int(start[:4]), int(start[4:6]), int(start[6:])),
        end=date(int(end[:4]), int(end[4:6]), int(end[6:])),
    ), None


_NARRATIVE_PERIODIC_FORM = {
    "annual": "사업보고서",
    "half": "반기보고서",
    "quarter": "분기보고서",
}


def _slot_value_kind(slot: str) -> str:
    compact = _compact(slot).casefold()
    if any(cue in compact for cue in ("비율", "비중", "수익률", "퍼센트", "%")):
        return "percent"
    if any(cue in compact for cue in ("일자", "날짜", "결정일", "해지일", "체결일")):
        return "date"
    if any(cue in compact for cue in ("건수", "수량", "개수")):
        return "count"
    if any(cue in compact for cue in (
            "금액", "대금", "가액", "원금", "매출", "이익", "자산")):
        return "money"
    return "text"


#: **일자 + 「까지/기준」은 대상기간이 아니라 정보 cutoff 다.**
#:
#: 「2025년 12월 16일**까지** 공개된 공시만 보면」은 조회 대상의 기간이 아니라
#: 「그날까지 공개된 것만 본다」는 정보 공개 경계다. 모델은 이것을 자주
#: `target_period_expressions` 에 넣고 `as_of` 를 비운다 — 사람이 읽으면 둘 다
#: 「그 날짜」라서 헷갈릴 자리다.
#:
#: 동결 Gold 에서 명시된 일자에 이 접미사가 붙은 문항은 **전부** 그 날짜를 as_of
#: (또는 event timepoint)로 쓴다. 반대로 「2024년에 공시한」처럼 **연·월 단위**는
#: 조회 대상 기간이므로 건드리면 안 된다. 그래서 **일 단위 날짜에만** 적용한다.
_CUTOFF_DAY = re.compile(
    r"^\s*(20[0-9]{2})\s*[-./년]\s*([01]?[0-9])\s*[-./월]\s*([0-3]?[0-9])\s*일?\s*"
    r"(?:까지|기준)\s*$")


def _cutoff_from_target_period(
        as_of_expression: "str | None",
        target_period_expressions: "list[str] | tuple[str, ...] | None",
        ) -> "tuple[str, list[str]] | None":
    """대상기간에 섞여 든 정보 cutoff 를 꺼낸다. 없으면 ``None``.

    **as_of 를 이미 적었으면 손대지 않는다.** 둘 다 있으면 무엇이 경계인지
    질문이 정한 것이므로 우리가 바꾸지 않는다.
    """

    if as_of_expression is not None and as_of_expression.strip() not in {
            "", "unspecified"}:
        return None
    rows = list(target_period_expressions or ())
    if len(rows) != 1:
        return None
    match = _CUTOFF_DAY.fullmatch(rows[0])
    if match is None:
        return None
    # **절대 날짜로 내려서 넘긴다.** `_as_of` 는 이미 정규화된 YYYYMMDD 만 받고,
    # `_event_timepoints` 도 그 모양을 그대로 받는다. 표면 문자열을 넘기면
    # 계획 전체가 `unsupported_request` 로 죽는다.
    try:
        value = _as_yyyymmdd(*(int(part) for part in match.groups()))
    except (TypeError, ValueError):
        return None
    return value, []


def _event_timepoints(
        expression: str | None, *, corpus_cutoff: str,
        ) -> tuple[list[str] | None, bool, str | None]:
    """정보 cutoff 표현을 하나 이상의 절대 Event timepoint로 내린다."""

    if expression is None or _compact(expression) in {
            "", "현재", "오늘", "지금", "최신", "기준일", "코퍼스기준일",
            "제공코퍼스기준", "제공된코퍼스기준"}:
        return [corpus_cutoff], True, None
    compact = _compact(expression)
    values: list[str] = []
    if re.fullmatch(
            r"20[0-9]{6}(?:[/,~·]|과|와|및)20[0-9]{6}(?:(?:[/,~·]|과|와|및)20[0-9]{6})*",
            compact):
        values = re.findall(r"20[0-9]{6}", compact)
    elif re.fullmatch(r"20[0-9]{6}", compact):
        values = [compact]
    else:
        full_dates = re.findall(
            r"(?<![0-9])(20[0-9]{2})[-./년]([01]?[0-9])[-./월]([0-3]?[0-9])일?(?![0-9])",
            compact,
        )
        remainder = re.sub(
            r"(?<![0-9])20[0-9]{2}[-./년][01]?[0-9][-./월][0-3]?[0-9]일?(?![0-9])",
            "", compact)
        if (full_dates
                and re.fullmatch(r"(?:(?:[/,~·]|과|와|및))*", remainder)
                is not None):
            try:
                values = [_as_yyyymmdd(*(int(part) for part in row))
                          for row in full_dates]
            except ValueError as exc:
                return None, False, str(exc)
        else:
            # HCX may preserve a question-grounded shorthand such as
            # ``25년 12월 25일`` and the coordinated omission ``26일``.  The
            # shared date scanner restores only uniquely coordinated omissions;
            # keep the accepted alphabet narrow so arbitrary prose containing a
            # date cannot silently become a timepoint selector.
            if re.fullmatch(
                    r"[0-9·./,~\-\s년월일과와및이랑랑또는]+",
                    expression.strip()) is None:
                return None, False, "Event timepoint는 절대 날짜여야 합니다"
            restored = question_date_surfaces(expression)
            if not restored or any(
                    month is None or day is None
                    for _, month, day in restored):
                return None, False, "Event timepoint는 절대 날짜여야 합니다"
            try:
                values = sorted(
                    _as_yyyymmdd(year, month, day)
                    for year, month, day in restored
                    if month is not None and day is not None
                )
            except ValueError as exc:
                return None, False, str(exc)
    if len(values) != len(set(values)):
        return None, False, "Event timepoint는 중복될 수 없습니다"
    try:
        for value in values:
            validate_as_of(value, field_name="event_timepoint")
    except ValueError as exc:
        return None, False, str(exc)
    if any(value > corpus_cutoff for value in values):
        return None, False, "Event timepoint는 corpus_cutoff 이후일 수 없습니다"
    return values, False, None


def _document_period_options(expression: str) -> tuple[str, ...]:
    """보고서 종류가 빠진 표현에만 일반적인 periodic 후보를 제시한다."""

    normalized = unicodedata.normalize("NFC", expression).strip()
    year_match = re.search(r"(?<![0-9])(20[0-9]{2})(?![0-9])", normalized)
    prefix = f"{year_match.group(1)}년 " if year_match else ""
    return tuple(
        f"{prefix}{kind}"
        for kind in ("사업보고서", "반기보고서", "1분기보고서", "3분기보고서")
    )


def _document_report_fragment(
        expression: str,
        ) -> tuple[str | None, tuple[str, ...], str | None]:
    """periodic 표면형을 canonical ``report_nm`` substring으로 내린다."""

    normalized = unicodedata.normalize("NFC", expression).strip()
    compact = re.sub(r"\s+", "", normalized)
    if compact in {"최근", "현재", "최신"}:
        return None, (), None
    years = tuple(dict.fromkeys(re.findall(
        r"(?<![0-9])(20[0-9]{2})(?![0-9])", compact)))
    quarters = tuple(dict.fromkeys(re.findall(r"([1-4])분기", compact)))
    report_kinds = sum((
        "사업보고서" in compact,
        "반기" in compact,
        bool(quarters),
    ))
    if len(years) > 1 or len(quarters) > 1 or report_kinds > 1:
        return None, (), "단일 document selector에 복수 보고서 기간을 넣을 수 없습니다"
    year = years[0] if years else None
    quarter = quarters[0] if quarters else None
    if quarter is not None and quarter not in {"1", "3"}:
        return None, (), "정기공시 문서 selector는 1분기·반기·3분기·사업보고서를 사용합니다"
    if quarter is not None:
        month = "03" if quarter == "1" else "09"
        return (
            f"분기보고서 ({year}.{month})" if year else f".{month})",
            (), None,
        )
    if "반기" in compact:
        return (
            f"반기보고서 ({year}.06)" if year else "반기보고서 (",
            (), None,
        )
    if "사업보고서" in compact:
        return (
            f"사업보고서 ({year}.12)" if year else "사업보고서 (",
            (), None,
        )
    if "분기" in compact:
        return None, _document_period_options(normalized)[2:], None
    if year is not None or "보고서" in compact:
        return None, _document_period_options(normalized), None
    return None, (), f"지원하지 않는 document target period입니다: {expression}"


class CompanyLike(Protocol):
    corp_code: str
    corp_name: str


class CompanyResolver(Protocol):
    def resolve_company(self, name: str) -> list[CompanyLike]: ...


#: 표기 정규화는 **한 곳에서만** 정의한다 — 두 곳에 있으면 승인 구어 층과 정본
#: 사전이 같은 표기를 다른 키로 본다.
_key = normalize_surface_key


#: 회사 표기 해소는 **canonical registry 한 곳에서만** 한다
#: (``src/canonical/company_alias.py`` + ``company_aliases_approved.json``).
#: 여기에 표를 두면 두 곳이 다른 답을 낸다. registry 는 ``삼전`` 을 삼성전자로
#: 확정하고(2026-08-17 결정) 그룹 약칭 8개는 후보로 두고 역질문한다.
#: 표를 비우면 registry 판정이 그대로 쓰인다.
#:
#: 이름은 남긴다 — ``planner_policy`` 가 import 한다. resolver 가 registry 를
#: 갖지 않는 fake 인 경우(계약 테스트)에도 동작이 바뀌지 않도록 빈 사전이다.
COMPANY_ALIASES: dict[str, str] = {}

#: **정본 개념 ID 는 언제나 자기 자신의 표기다.** 예전에는 이 항등 사상을 개념마다
#: 손으로 적었는데(``_key("revenue"): REVENUE``), 5종에만 적혀 있어 새 개념은
#: ``concept_mention="eps"`` 를 거절했다. 개념을 늘릴 때 같이 늘어나야 하는 줄이므로
#: 표가 아니라 규칙으로 둔다 — v0.3→v0.4 migration 이 개념 ID 를 표기로 넘긴다.
_CONCEPT_ID_ALIASES = {_key(concept.value): concept for concept in FinancialConcept}

#: 개념 ID 로 유추할 수 없는 **사람의 말**만 손으로 적는다. 정본 계정 사전
#: (``account_map.tsv``)에 없는 구어·영문 표기가 여기 온다.
METRIC_ALIASES = {
    **_CONCEPT_ID_ALIASES,
    _key("매출"): FinancialConcept.REVENUE,
    _key("매출액"): FinancialConcept.REVENUE,
    # 공시 표 라벨 「수익(매출액)」은 매출액의 동의 표기다. 괄호를 포함한
    # 질문 표면도 같은 정본 concept로 닫되, 일반 「수익」은 넓은 뜻이라
    # 별칭으로 추가하지 않는다.
    _key("수익(매출액)"): FinancialConcept.REVENUE,
    _key("sales"): FinancialConcept.REVENUE,
    _key("영업이익"): FinancialConcept.OPERATING_INCOME,
    _key("operating profit"): FinancialConcept.OPERATING_INCOME,
    _key("순이익"): FinancialConcept.NET_INCOME,
    _key("당기순이익"): FinancialConcept.NET_INCOME,
    _key("net income"): FinancialConcept.NET_INCOME,
    _key("총자산"): FinancialConcept.TOTAL_ASSETS,
    _key("자산총계"): FinancialConcept.TOTAL_ASSETS,
    _key("total assets"): FinancialConcept.TOTAL_ASSETS,
    # `capex` 단독은 **넣지 않는다.** 유형·무형 취득을 모두 뜻하는데 정본 별칭이
    # 유형으로 자동확정하고 있었다 (외부 검토 §2). 구어층 `capex_bare` 가 역질문한다.
    _key("유형자산취득"): FinancialConcept.CAPEX_PPE,
}


@dataclass(frozen=True, slots=True)
class ConceptAxes:
    """개념 하나의 회계 성질. **코드가 아니라 이 표가 늘어난다.**

    ``allowed_statements`` 가 필요한 이유: 70곳 중 49곳이 손익계산서(IS)를 따로
    내지 않고 포괄손익계산서(CI)만 제출한다. 손익 개념을 ``IS`` 로 못박으면 그
    49곳을 못 찾고, ``None``(무제약)으로 두면 **현금흐름표의 조정 항목까지**
    잡힌다. ``interest_income`` 은 CI 941건 · CF 692건으로 성질이 다른 값이
    한 이름에 걸쳐 있다.

    **값을 고르는 축이 아니다.** 같은 기간을 다른 값으로 보고하는 일의 96%는
    정정·재작성이고 그것은 ``view=restated`` 가 처리한다. 좁히지 않으면
    ``lookup`` 이 ``ambiguous_*`` 로 되돌려 역질문이 되므로, 이 축의 값어치는
    **불필요한 역질문을 줄이는 것**이다.

    ``cumulative_matters`` 가 필요한 이유: 같은 ``period_type='quarter'`` 안에
    3개월 단독과 9개월 누적이 **함께** 들어 있다. ``statement`` 를 좁혀도 이
    충돌은 남는다 — 삼성전자 2023-09-30 CFS 에 5,844,171(단독)과
    9,142,342(누적)가 모두 있다.

    ``aggregation`` 이 필요한 이유: EPS 는 주당 값이라 누적 차감·기간 합산이
    성립하지 않는다. 금액 계정과 같은 연산을 붙이면 틀린 수가 나온다.
    """

    period_semantics: str                      # instant | duration
    statement_family: str                      # BS | PL | CF
    allowed_statements: tuple[str, ...]
    measure_kind: str = "monetary"             # monetary | per_share
    aggregation: str = "additive_duration"     # instant_stock | additive_duration
                                               # | non_additive_duration
    cumulative_matters: bool = True

    def plan_statement(self) -> str | None:
        """QueryPlan 의 ``statement`` — 그 개념의 **정본 표**.

        예전에는 허용 표가 여럿이면 `None` 을 넣었다. 그래서 PL 개념 20종의
        `statement` 가 비었고, Stage2 는 어느 표를 뜻하는지 모른 채 plan 을 받았다.

        `statement` 는 **물리적 저장 위치가 아니라 계열**이다. 검증도 그렇게 한다 —
        `query_plan` 은 `allowed_statements` 안에 드는지만 보고, 사실 조회를 이
        값으로 거르지 않는다. 동결 정답지도 회사와 무관하게 `revenue` 32건을 모두
        `IS` 로 적는다(코퍼스는 43곳이 CI 에 담는데도).

        그래서 **허용 표의 첫 값**이 정본이다. 순서가 뜻을 갖는다.
        """

        return self.allowed_statements[0] if self.allowed_statements else None


#: 손익 계열의 **정본은 IS** 다. 회사에 따라 CI 에 실리기도 하지만 그것은 저장
#: 위치이고, plan 의 `statement` 는 계열을 뜻한다. 순서가 정본을 정한다.
_PL_STATEMENTS = ("IS", "CI")
#: 포괄손익 전용. `other_comprehensive_income`·`total_comprehensive_income` 은
#: 코퍼스 전수에서 **CI 에만** 나타난다 — IS 를 허용하면 없는 표를 가리킨다.
_CI_ONLY = ("CI",)

#: 개념 → 회계 성질. 지금은 등록된 5종만 담아 **동작이 바뀌지 않는지 확인한다.**
#: 개념 추가는 이 표에 줄을 넣는 일이고 코드는 바뀌지 않는다.
_CONCEPT_AXES: dict[FinancialConcept, ConceptAxes] = {
    # ── 손익: PL family. 70곳 중 49곳이 CI 만 내므로 IS 로 못박지 않는다.
    #    누적이 문제되므로 cumulative_matters=True (기본값).
    FinancialConcept.REVENUE: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.OPERATING_INCOME: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.NET_INCOME: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.COST_OF_SALES: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.GROSS_PROFIT: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.OPERATING_EXPENSES: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.OTHER_COMPREHENSIVE_INCOME: ConceptAxes(
        "duration", "PL", _CI_ONLY),
    FinancialConcept.PRETAX_INCOME: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.SGANDA: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.TOTAL_COMPREHENSIVE_INCOME: ConceptAxes(
        "duration", "PL", _CI_ONLY),

    # ── 주당 값: 누적 차감·기간 합산이 성립하지 않는다. 반기 EPS 에서
    #    1분기 EPS 를 빼면 2분기 EPS 가 되지 않는다.
    FinancialConcept.EPS: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS, measure_kind="per_share",
        aggregation="non_additive_duration"),
    FinancialConcept.DILUTED_EPS: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS, measure_kind="per_share",
        aggregation="non_additive_duration"),

    # ── 재무상태표 잔액: 시점 값이므로 누적 개념이 없다. 차이·증감률은
    #    되지만 기간 합산은 성립하지 않는다.
    FinancialConcept.TOTAL_ASSETS: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.AOCI: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.CASH_AND_EQUIVALENTS: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.CURRENT_ASSETS: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.CURRENT_LIABILITIES: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.INTANGIBLE_ASSETS: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.INVESTMENT_PROPERTY: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.NON_CURRENT_ASSETS: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.NON_CURRENT_LIABILITIES: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.PPE: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.RETAINED_EARNINGS: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.RIGHT_OF_USE_ASSETS: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.SHARE_CAPITAL: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.SHARE_PREMIUM: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.TOTAL_EQUITY: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.TOTAL_LIABILITIES: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.TOTAL_LIABILITIES_AND_EQUITY: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.TRADE_AND_OTHER_RECEIVABLES: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),


    # ── 손익이지만 원천이 CF 에도 나타나는 8종. CF 쪽은 간접법 조정항목이다.
    #    statement 를 PL 로 좁히지 않으면 성질이 다른 값이 함께 잡힌다 —
    #    interest_income 은 CI 941 · CF 692 로 CF 비중이 특히 크다.
    FinancialConcept.EQUITY_METHOD_INCOME: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.FINANCE_COSTS: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.FINANCE_INCOME: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.INCOME_TAX: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.INTEREST_EXPENSE: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.INTEREST_INCOME: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.OTHER_EXPENSES: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.OTHER_INCOME: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),

    # ── 잔액이지만 원천이 CF 에도 나타나는 7종. CF 쪽은 재고·매출채권·
    #    충당부채의 증감이고 잔액과 다른 값이다.
    FinancialConcept.CURRENT_TAX_ASSETS: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.CURRENT_TAX_LIABILITIES: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.DEFERRED_TAX_ASSETS: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.DEFERRED_TAX_LIABILITIES: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.INVENTORIES: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.PROVISIONS: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.TRADE_RECEIVABLES: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    # ── 현금흐름: 기간 값이고 CF 에만 있다.
    FinancialConcept.CAPEX_PPE: ConceptAxes(
        "duration", "CF", ("CF",)),
    FinancialConcept.CAPEX_INTANGIBLE: ConceptAxes(
        "duration", "CF", ("CF",)),
    FinancialConcept.CF_FINANCING: ConceptAxes(
        "duration", "CF", ("CF",)),
    FinancialConcept.CF_INVESTING: ConceptAxes(
        "duration", "CF", ("CF",)),
    FinancialConcept.CF_OPERATING: ConceptAxes(
        "duration", "CF", ("CF",)),
    FinancialConcept.DISPOSAL_INTANGIBLE: ConceptAxes(
        "duration", "CF", ("CF",)),
    FinancialConcept.DISPOSAL_PPE: ConceptAxes(
        "duration", "CF", ("CF",)),
    FinancialConcept.DIVIDENDS_PAID: ConceptAxes(
        "duration", "CF", ("CF",)),
    FinancialConcept.INTEREST_PAID: ConceptAxes(
        "duration", "CF", ("CF",)),
    FinancialConcept.INTEREST_RECEIVED: ConceptAxes(
        "duration", "CF", ("CF",)),
    FinancialConcept.NET_CHANGE_IN_CASH: ConceptAxes(
        "duration", "CF", ("CF",)),
    FinancialConcept.CASH_BEGINNING: ConceptAxes(
        "duration", "CF", ("CF",)),
    FinancialConcept.CASH_ENDING: ConceptAxes(
        "duration", "CF", ("CF",)),
    # ── 연결 귀속 구획 (이슈 #127) ─────────────────────────────────────────
    # 잔액은 재무상태표의 instant, 귀속액은 손익표의 duration 이다. 축이
    # 다르므로 개념도 달라야 한다.
    FinancialConcept.NON_CONTROLLING_INTERESTS: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.OWNERS_OF_PARENT: ConceptAxes(
        "instant", "BS", ("BS",), aggregation="instant_stock",
        cumulative_matters=False),
    FinancialConcept.NET_INCOME_NON_CONTROLLING: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    FinancialConcept.NET_INCOME_OWNERS_OF_PARENT: ConceptAxes(
        "duration", "PL", _PL_STATEMENTS),
    # 총포괄손익 귀속은 포괄손익계산서에만 실린다 — `other_comprehensive_income`
    # 과 같은 이유로 IS 를 허용하면 없는 표를 가리킨다.
    FinancialConcept.COMPREHENSIVE_INCOME_NON_CONTROLLING: ConceptAxes(
        "duration", "PL", _CI_ONLY),
    FinancialConcept.COMPREHENSIVE_INCOME_OWNERS_OF_PARENT: ConceptAxes(
        "duration", "PL", _CI_ONLY),
}

#: 표에 없는 개념은 **기간·표 무제약**으로 둔다. 조용히 틀리는 것보다 넓게 찾는다.
_DEFAULT_AXES = ConceptAxes("duration", "PL", _PL_STATEMENTS)


def concept_axes(concept: FinancialConcept) -> ConceptAxes:
    return _CONCEPT_AXES.get(concept, _DEFAULT_AXES)


def _canonical_account_aliases() -> dict[str, FinancialConcept]:
    """정본 계정 사전에서 **agent 가 아는 개념의 표기만** 끌어온다.

    손으로 별칭 줄을 늘리면 특정 질문에 맞춘 표가 된다. 여기서는 전처리가 코퍼스
    전체에서 만든 ``src/ingest/account_map.tsv`` 를 그대로 읽으므로, 사전이 커지면
    agent 도 같이 넓어지고 **코드는 바뀌지 않는다.**

    사전을 못 읽어도 조회는 계속돼야 하므로 실패는 조용히 무시한다 — 위의 고정
    별칭만으로도 기본 개념은 해석된다.
    """

    known = {concept.value: concept for concept in FinancialConcept}
    source = Path(__file__).resolve().parent.parent / "src/ingest/account_map.tsv"
    table: dict[str, FinancialConcept] = {}
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError:
        return table
    for line in lines:
        if line.startswith("#") or "\t" not in line:
            continue
        surface, _, concept_name = line.partition("\t")
        concept = known.get(concept_name.strip())
        if concept is not None and surface.strip():
            table.setdefault(_key(surface), concept)
    return table


#: 역질문 선택지에 **사람이 읽는 이름**으로 개념을 보여주기 위한 표기.
#:
#: 선택지 라벨이 정규 개념 식별자(``cf_operating``)로 나가면 한국어 공시 질문을
#: 한 사용자가 고를 수 없다. 그렇다고 여기서 표기를 지어내면 승인 층이 조용히
#: 틀리는 그 위험이 그대로 재현된다 — 사용자는 라벨을 보고 고르므로 라벨이 틀리면
#: 다른 계정을 **본인이** 선택하게 된다.
#:
#: 그래서 값은 **정본 계정 사전에 실제로 있는 원문 표기여야 한다.** 아래
#: ``_validate_concept_display`` 가 적재 시점에 확인하고, 사전에 없는 표기를 쓰면
#: 즉시 실패한다. 개념 하나에 표기가 여럿일 때(``결손금``/``이익잉여금``) 어느 쪽을
#: 보여줄지가 이 표의 유일한 판단이며, 공시에서 널리 쓰는 표준형을 고른다.
_CONCEPT_DISPLAY_SOURCE = {
    "revenue": "매출액",
    "operating_income": "영업이익",
    "net_income": "당기순이익",
    "gross_profit": "매출총이익",
    "total_assets": "자산총계",
    "total_liabilities": "부채총계",
    "total_equity": "자본총계",
    "current_assets": "유동자산",
    "current_liabilities": "유동부채",
    "non_current_liabilities": "비유동부채",
    "share_capital": "자본금",
    "share_premium": "자본잉여금",
    "retained_earnings": "이익잉여금",
    "cash_and_equivalents": "현금및현금성자산",
    # 「기초」·「기말」은 잔액의 시점을 가르는 말이지 꾸밈말이 아니다. 이름이 없어
    # 기초를 물은 사람이 기말 값을 받았다 (`CG-033`, 이슈 #126).
    "cash_beginning": "기초현금및현금성자산",
    "cash_ending": "기말현금및현금성자산",
    # 연결 귀속 구획. 표시 표기는 정본 계정 사전의 실제 표기여야 한다
    # (`_validate_concept_display` 가 적재 시 확인한다).
    "non_controlling_interests": "비지배지분",
    "owners_of_parent": "지배기업소유주지분",
    "net_income_non_controlling": "비지배지분순이익",
    "net_income_owners_of_parent": "지배기업소유주지분순이익",
    "comprehensive_income_non_controlling": "비지배지분총포괄손익",
    "comprehensive_income_owners_of_parent": "지배기업소유주지분총포괄손익",
    "net_change_in_cash": "현금및현금성자산의순증감",
    "trade_receivables": "매출채권",
    "trade_and_other_receivables": "매출채권및기타채권",
    "investment_property": "투자부동산",
    "income_tax": "법인세비용",
    "current_tax_assets": "당기법인세자산",
    "current_tax_liabilities": "당기법인세부채",
    "interest_income": "이자수익",
    "interest_expense": "이자비용",
    "interest_received": "이자의수취",
    "interest_paid": "이자의지급",
    "other_comprehensive_income": "기타포괄손익",
    "total_comprehensive_income": "총포괄손익",
    "cf_operating": "영업활동현금흐름",
    "cf_investing": "투자활동현금흐름",
    "capex_ppe": "유형자산의취득",
    "capex_intangible": "무형자산의취득",
    "disposal_ppe": "유형자산의처분",
    "disposal_intangible": "무형자산의처분",
    "cf_financing": "재무활동현금흐름",
    "aoci": "기타포괄손익누계액",
    "cost_of_sales": "매출원가",
    "sganda": "판매비와관리비",
    "operating_expenses": "영업비용",
    "other_income": "기타수익",
    "other_expenses": "기타비용",
    "finance_income": "금융수익",
    "finance_costs": "금융비용",
    "equity_method_income": "지분법손익",
    "pretax_income": "법인세비용차감전순이익",
    "eps": "기본주당순이익",
    "diluted_eps": "희석주당순이익",
    "deferred_tax_assets": "이연법인세자산",
    "deferred_tax_liabilities": "이연법인세부채",
    "dividends_paid": "배당금의지급",
    "inventories": "재고자산",
    "ppe": "유형자산",
    "intangible_assets": "무형자산",
    "right_of_use_assets": "사용권자산",
    "non_current_assets": "비유동자산",
    "provisions": "충당부채",
    "total_liabilities_and_equity": "부채와자본총계",
}


def _validate_concept_display() -> dict[str, str]:
    """표시 표기가 정본 계정 사전에 실제로 있는지 확인하고 굳힌다.

    사전을 못 읽는 환경에서는 검증을 건너뛴다 — 위 ``_canonical_account_aliases``
    와 같은 이유로, 사전 부재가 조회 자체를 막아서는 안 된다.
    """

    source = Path(__file__).resolve().parent.parent / "src/ingest/account_map.tsv"
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError:
        return dict(_CONCEPT_DISPLAY_SOURCE)
    spellings: dict[str, set[str]] = {}
    for line in lines:
        if line.startswith("#") or "\t" not in line:
            continue
        surface, _, concept_name = line.partition("\t")
        if surface.strip():
            spellings.setdefault(concept_name.strip(), set()).add(surface.strip())
    known = {concept.value for concept in FinancialConcept}
    for concept, display in _CONCEPT_DISPLAY_SOURCE.items():
        if concept not in known:
            raise ValueError(f"개념 표시 표기가 등록되지 않은 개념을 가리킵니다: {concept}")
        available = spellings.get(concept)
        if available and display not in available:
            raise ValueError(
                f"개념 표시 표기 {display!r} 가 {concept} 의 정본 계정 표기에 "
                f"없습니다: {sorted(available)}")
    return dict(_CONCEPT_DISPLAY_SOURCE)


CONCEPT_DISPLAY_KO = _validate_concept_display()


def concept_display_name(concept: str) -> str:
    """역질문 선택지에 보여줄 이름. 표기가 없으면 개념 이름 그대로 돌려준다."""

    return CONCEPT_DISPLAY_KO.get(concept, concept)


#: 고정 별칭이 우선한다. 정본 사전은 **넓히기만** 하고 덮어쓰지 않는다.
METRIC_ALIASES = {**_canonical_account_aliases(), **METRIC_ALIASES}

#: 계정 이름 꼬리의 **수량 표현**. 계정을 바꾸지 않고 「얼마인지」만 덧붙이는 말이다.
#: 정본 사전은 코퍼스 계정명(``유형자산취득``)으로 만들어지는데 사용자는
#: ``유형자산 취득액``·``유형자산 취득 현금유출액`` 처럼 쓴다. 특정 계정·회사와
#: 무관한 표기 규칙이다.
_ACCOUNT_QUANTITY_SUFFIXES = ("현금유출액", "현금유입액", "지출액", "금액", "액")


def _strip_account_quantity_suffix(key: str) -> str:
    """꼬리의 수량 표현을 **반복해서** 떼어낸다. 남는 부분이 없어지면 멈춘다."""

    while True:
        for suffix in _ACCOUNT_QUANTITY_SUFFIXES:
            if key.endswith(suffix) and len(key) > len(suffix):
                key = key[: -len(suffix)]
                break
        else:
            return key


#: 승인 구어 질문 패턴. **정본 사전이 아는 표기는 넘긴다** — 검증 가능한 매핑이
#: 사람 판단보다 앞선다. 파일이 없으면 빈 튜플이고 동작은 이 층이 없던 때와 같다.
#:
#: 이 층은 **질문 본문**을 본다(`resolve_from_question`). 모델이 쓴 표기는
#: grounding 이 이미 검사했고, 버려졌다면 사람이 쓴 말은 질문에만 남는다.
CONCEPT_QUESTION_PATTERNS = load_concept_question_patterns(
    frozenset(METRIC_ALIASES))


def resolve_metric_concept(surface: str) -> "FinancialConcept | None":
    """계정 표기 → 개념. 정본 사전 → 접미사 제거 → 승인 구어 순으로 본다.

    **더하기만 한다.** exact 로 찾히던 것은 그대로다. 어느 단계에서도 못 찾으면
    여전히 ``None`` 이므로, 사전에 없는 계정이 조용히 아무 개념으로 붙지는 않는다.

    승인 구어는 **마지막**이다. 정본 사전과 손으로 적은 사람 말이 먼저 답하고,
    그것이 답하지 못할 때만 승인 층을 본다 — 승인 층이 검증 가능한 매핑을 덮으면
    코퍼스에서 온 사실이 사람 판단에 밀린다.

    후보가 둘 이상인 구어(`벌었어`)는 여기서 ``None`` 이다. 자동 확정하지 않는
    것이 그 표기의 승인 내용이므로, 호출자는 `held_metric_candidates` 로 후보를
    꺼내 **묻는다.**
    """

    # Clarification option values use the public canonical enum rather than a
    # Korean display label. Accepting that enum here lets the same resolver
    # handle both question text and a typed user selection without a
    # question-specific alias table.
    try:
        return FinancialConcept(surface.strip())
    except (ValueError, AttributeError):
        pass
    key = _key(surface)
    concept = METRIC_ALIASES.get(key)
    if concept is not None:
        return concept
    stripped = _strip_account_quantity_suffix(key)
    if stripped != key:
        concept = METRIC_ALIASES.get(stripped)
        if concept is not None:
            return concept
    for pattern in CONCEPT_QUESTION_PATTERNS:
        # **표기만으로 확정되는 것(AUTO)만** 여기서 답한다. CONTEXT 는 질문 단서를
        # 봐야 하므로 `resolve_from_question` 의 몫이다 — 단서를 못 보는 자리에서
        # CONTEXT 를 확정하면 조건을 무시하고 고르는 셈이 된다.
        if pattern.key == key and pattern.mode == "AUTO":
            return pattern.candidates[0]
    return None


def held_metric_candidates(surface: str) -> tuple[FinancialConcept, ...]:
    """**자동 확정하지 않는** 구어의 후보. 확인 역질문에 쓴다.

    표기 하나를 그대로 받는 경로다(역질문 답변·fixture). 질문 문맥이 필요한
    판정은 `concept_alias.resolve_from_question` 이 한다 — 단서를 봐야 하는 것은
    질문이 있어야 판정할 수 있다.
    """

    key = _key(surface)
    for pattern in CONCEPT_QUESTION_PATTERNS:
        if pattern.key != key:
            continue
        if pattern.mode in ("CLARIFY", "ROUTE_CLARIFY"):
            return pattern.candidates
        # A CONTEXT pattern with this surface may be inapplicable while a
        # later bare CLARIFY/ROUTE_CLARIFY pattern deliberately preserves the
        # user's choice.  ``투자`` is the concrete shape: ``순액`` selects a
        # cash-flow concept, but without that cue it must ask rather than be
        # treated as an unknown target.  AUTO is still decisive because its
        # surface is independently sufficient.
        if pattern.mode == "AUTO":
            return ()
    return ()


SCOPE_ALIASES = {
    _key("연결"): "CFS",
    _key("연결재무제표"): "CFS",
    _key("CFS"): "CFS",
    _key("별도"): "SFS",
    _key("별도재무제표"): "SFS",
    _key("SFS"): "SFS",
}


class FinancialDraftFixture(ContractModel):
    """첫 주 fixture adapter의 비공개 입력형."""

    company_text: str = Field(min_length=1)
    metric_text: str = Field(min_length=1)
    year: int = Field(ge=1900, le=9999)
    scope: str | None = None
    as_of: str | None = None
    view: str = "restated"


def _concept_period(concept: FinancialConcept, year: int) -> dict:
    period, error = _financial_period(
        concept, year=year, expression=None,
        reference_date=date(year, 12, 31),
    )
    if error is not None or period is None:  # pragma: no cover - fixed annual input
        raise ValueError(error or "재무 기간을 확정하지 못했습니다")
    return period


def _expand_two_digit_year(compact: str) -> str:
    """``25년`` 처럼 두 자리로 쓴 연도를 네 자리로 편다.

    구어체 질문에서 흔한 표기이고 **특정 연도·회사와 무관한 표기 규칙**이다.
    2000년대만 다룬다 — 코퍼스가 그 범위이고, 1900년대로 해석할 여지를 두면
    ``99년`` 같은 입력에서 조용히 엉뚱한 기간이 된다.

    이미 네 자리인 것은 건드리지 않는다. ``2025년`` 의 ``25년`` 부분이 다시
    치환되지 않도록 앞에 숫자가 없을 때만 편다.
    """

    # **문자열 맨 앞의 연도 한 번만** 편다. 뒤따르는 월·일까지 잡으면
    # ``25.12.17`` 이 ``2025.2012.17`` 로 깨진다.
    return re.sub(
        r"^([0-9]{2})(?=년|[-./][01]?[0-9][-./])",
        lambda match: f"20{match.group(1)}", compact, count=1)


def _financial_period(
        concept: FinancialConcept, *, year: int | None,
        expression: str | None, reference_date: date,
        ) -> tuple[dict[str, object] | None, str | None]:
    """표준 달력 회계기간을 canonical Fact 좌표로 내린다.

    날짜 endpoint는 DART 표의 누적 열을 뜻하고, 자연어 ``N분기``는 해당
    단일분기를 뜻한다. 따라서 같은 6월 30일이라도 ``2025-06-30``은 상반기
    누적이고 ``2025년 2분기``는 4~6월 단일분기다.
    """

    if year is not None and expression is not None:
        return None, "재무 기간은 year와 period_expression 중 하나로만 지정해야 합니다"
    if year is None and expression is None:
        return None, None

    end: date
    period_start: date
    period_type: str
    cumulative: bool

    if year is not None:
        end = date(year, 12, 31)
        period_start = date(year, 1, 1)
        period_type = "annual"
        cumulative = True
    else:
        assert expression is not None
        compact = re.sub(
            r"\s+", "", unicodedata.normalize("NFC", expression).strip())
        compact = _expand_two_digit_year(compact)

        # 「2025년 1월부터 12월까지」는 달력 연도 전체를 명시한 표현이다.
        # 12개월보다 좁은 범위는 재무 기간을 임의로 만들지 않는다.
        full_year_range = re.fullmatch(
            r"(20[0-9]{2})년1월부터(?:12월(?:말|31일?)?|12월까지)", compact)
        if full_year_range is not None:
            compact = f"{full_year_range.group(1)}년"

        date_match = re.fullmatch(
            r"(20[0-9]{2})[-./]([01]?[0-9])[-./]([0-3]?[0-9])", compact)
        if date_match is None:
            date_match = re.fullmatch(
                r"(20[0-9]{2})년([01]?[0-9])월([0-3]?[0-9])일?", compact)
        if date_match is not None:
            try:
                end = date(*(int(value) for value in date_match.groups()))
            except ValueError:
                return None, "재무 기간에는 실제로 존재하는 날짜가 필요합니다"
            endpoint = {
                (3, 31): (date(end.year, 1, 1), "quarter", True),
                (6, 30): (date(end.year, 1, 1), "half", True),
                (9, 30): (date(end.year, 1, 1), "quarter", True),
                (12, 31): (date(end.year, 1, 1), "annual", True),
            }.get((end.month, end.day))
            if endpoint is None:
                return None, (
                    "지원하는 표준 재무 기간 종료일은 03-31, 06-30, "
                    "09-30, 12-31입니다")
            period_start, period_type, cumulative = endpoint
        else:
            match = re.fullmatch(
                # 「FY2025」— 회계연도 접두 표기(#171 M02). 「2025년도」
                # 「2025회계연도」는 이미 접미 표기로 받고 있었다.
                r"(?:(?:FY)?(?P<year>20[0-9]{2})(?:년도|년|회계연도)?"
                r"(?:말|연말|기말)?|"
                r"(?P<relative>올해|금년|작년|지난해|전년|재작년))"
                # 반기·9개월에도 `누적` 을 받는다. 상반기는 **그 자체가 1~6월
                # 누적**이라 「상반기 누적」은 중복 표현이고 의미가 바뀌지 않는데,
                # 사람은 실제로 그렇게 쓴다(「2025 상반기 누적에서 1분기 빼면」).
                # 분기 쪽은 이미 `누적` 을 받고 있었으므로 표기 규칙을 맞추는 것이다.
                r"(?P<period>(?:제?[1-4]분기|Q[1-4]|[1-4]Q)"
                r"(?:누적|단일분기|단일|단독)?"
                r"|(?:상반기|반기|H1)(?:누적)?|9개월(?:누적)?)?",
                compact, flags=re.IGNORECASE)
            if match is None:
                return None, f"지원하지 않는 재무 기간 표현입니다: {expression}"
            if match.group("year") is not None:
                resolved_year = int(match.group("year"))
            else:
                resolved_year = {
                    "올해": reference_date.year,
                    "금년": reference_date.year,
                    "작년": reference_date.year - 1,
                    "지난해": reference_date.year - 1,
                    "전년": reference_date.year - 1,
                    "재작년": reference_date.year - 2,
                }[match.group("relative")]

            period = match.group("period")
            if period is None:
                period_start = date(resolved_year, 1, 1)
                end = date(resolved_year, 12, 31)
                period_type = "annual"
                cumulative = True
            elif period.casefold().removesuffix("누적") in {
                    "상반기", "반기", "h1"}:
                period_start = date(resolved_year, 1, 1)
                end = date(resolved_year, 6, 30)
                period_type = "half"
                cumulative = True
            elif period.startswith("9개월"):
                period_start = date(resolved_year, 1, 1)
                end = date(resolved_year, 9, 30)
                period_type = "quarter"
                cumulative = True
            else:
                quarter_match = re.search(r"[1-4]", period)
                assert quarter_match is not None
                quarter = int(quarter_match.group())
                end_month = quarter * 3
                end = date(
                    resolved_year, end_month,
                    monthrange(resolved_year, end_month)[1])
                explicitly_cumulative = period.endswith("누적")
                if quarter == 1 or explicitly_cumulative:
                    period_start = date(resolved_year, 1, 1)
                    cumulative = True
                    period_type = (
                        "half" if quarter == 2 else
                        "annual" if quarter == 4 else "quarter")
                else:
                    period_start = date(resolved_year, end_month - 2, 1)
                    period_type = "quarter"
                    cumulative = False

    axes = concept_axes(concept)
    statement = axes.plan_statement()
    if axes.period_semantics == "instant":
        # 시점 개념은 기간 시작·누적 개념이 없다. 잔액을 그 날짜로 읽는다.
        return {
            "period_start": None,
            "period_end": end,
            "period_type": "instant",
            "cumulative": None,
            "statement": statement,
        }, None
    return {
        "period_start": period_start,
        "period_end": end,
        "period_type": period_type,
        "cumulative": cumulative,
        "statement": statement,
    }, None


def _held_company_candidates(resolver: object, query: str) -> list:
    """자동 확정하지 않는 표기의 후보. 없으면 빈 목록.

    ``삼전`` 처럼 universe 안에 후보가 둘 이상인 표기는 ``resolve_company`` 가
    아무것도 돌려주지 않는다 — 하나만 돌려주면 호출자가 자동 확정해 버리기
    때문이다. 그러면 「회사를 찾지 못했습니다」로 거절되는데 **우리는 후보를
    알고 있다.** 거절 대신 물어야 한다.

    ``FixturePlanner`` 와 ``DraftPlanResolver`` 가 함께 쓰므로 모듈 함수로 둔다.
    resolver 가 이 메서드를 갖지 않으면(테스트용 fake) 빈 목록이다.
    """

    finder = getattr(resolver, "held_company_candidates", None)
    if not callable(finder):
        return []
    try:
        return list(finder(query) or ())
    except Exception:
        return []


def _available_scopes(resolver: object, *, corp_code: str, concept: object,
                      as_of: str) -> tuple[str, ...]:
    """그 회사·개념에 **실제로 존재하는** scope. 조회할 수 없으면 빈 tuple.

    연결에만 있는 계정이 있다 — ``non_controlling_interests``(비지배지분),
    ``owners_of_parent``, ``equity_method_income`` 은 별도재무제표에 개념 자체가
    없다. 그런 계정까지 「연결이냐 별도냐」를 되묻는 것은 **답이 하나뿐인 질문을
    사용자에게 넘기는 것**이다.

    반대로 둘 다 있으면 조용히 하나를 고르지 않는다. 실측으로 (회사·계정·기간)
    조합의 86.7%가 둘 다 있으므로, 기본값을 두면 대부분의 질문에서 사용자가
    묻지 않은 기준을 임의로 적용하게 된다.
    """

    scopes, _ = _scope_availability(
        resolver, corp_code=corp_code, concept=concept, as_of=as_of)
    return scopes


def _scope_availability(resolver: object, *, corp_code: str, concept: object,
                        as_of: str) -> tuple[tuple[str, ...], bool]:
    """``(가용 scope, 조회할 수 있었는가)``.

    **「조회할 수 없음」과 「조회했는데 없음」은 다르다.** 앞의 것은 우리가 모르는
    것이고 뒤의 것은 그 회사에 그 계정이 없다는 사실이다. 둘을 같은 빈 tuple 로
    돌려주면 호출자가 구분할 수 없다 — 계약 테스트의 fake resolver 처럼 `facts()`
    자체가 없는 경우까지 「값이 없다」로 거절하게 된다.
    """

    query = getattr(resolver, "facts", None)
    if not callable(query):
        return (), False
    try:
        rows = query(corp_code, as_of=as_of, concept=getattr(concept, "value", concept))
    except Exception:
        return (), False
    return tuple(sorted({
        scope for row in rows or ()
        if isinstance(scope := getattr(row, "scope", None), str)
    })), True


def statement_for_company(
        resolver: object, *, corp_code: str, concept: object, as_of: str,
        allowed: tuple[str, ...]) -> str | None:
    """**그 회사가 그 계정을 담은 표.** 정할 수 없으면 ``None``.

    개념만으로는 정할 수 없다. 실측하면 `revenue` 를 IS 에 담는 회사가 20곳,
    CI 에 담는 회사가 43곳, 둘 다인 곳이 1곳이다. 그래서 `plan_statement()` 는
    PL 개념 20종에 대해 `None` 을 돌려주고, 그 결과 plan 의 `statement` 가 비어
    Stage2 가 어느 표를 읽을지 모른 채 받는다.

    `scope` 를 이미 코퍼스에서 확정하듯(`primary_scope_for_period`) 여기서도
    **그 회사의 사실**을 본다. 하나로 좁혀질 때만 채우고, 둘 이상이면 비운다 —
    임의로 고르면 조용히 다른 표를 읽는다.

    `allowed` 밖의 값은 무시한다. 개념 축이 허용하지 않는 표를 코퍼스가 갖고
    있어도 계약을 어기지 않는다 (`net_income` 이 CF 에도 나타난다).
    """

    query = getattr(resolver, "facts", None)
    if not callable(query):
        return None
    try:
        rows = query(corp_code, as_of=as_of,
                     concept=getattr(concept, "value", concept))
    except Exception:                                      # noqa: BLE001
        return None
    found = {
        statement for row in rows or ()
        if isinstance(statement := getattr(row, "statement", None), str)
        and statement in allowed
    }
    return found.pop() if len(found) == 1 else None


#: 슬롯 이름 앞에 붙는 공시 서식의 번호·글머리. 「3. 정정사유」·「- 투자대상」.
_SLOT_PREFIX = re.compile(r"^\s*(?:[0-9]+\s*[.)]|[-·•*])\s*")
#: 뒤에 붙는 단위 괄호. 「해지금액(원)」·「계약금액 (원)」.
_SLOT_UNIT = re.compile(r"\s*\((?:원|주|%|백만원|천원|USD|달러)\)\s*$")


def normalize_slot_name(value: str) -> str:
    """요청 슬롯 이름을 **정본 표기로** 다듬는다.

    공시 서식의 필드명은 번호·글머리·단위 괄호를 달고 다니고, 모델은 질문의 띄어쓰기를
    그대로 옮긴다. 그래서 같은 것을 「투자 대상」·「- 투자대상」·「해지금액(원)」처럼
    제각각 적는다. Stage2 가 이 이름을 키로 쓰면 그 차이가 그대로 불일치가 된다.

    **글자를 바꾸지 않는다.** 번호·글머리·단위 괄호를 떼고 공백을 없앨 뿐이다 —
    없는 이름을 만들거나 다른 이름으로 옮기지 않는다. 팀이 쓰는 파생 슬롯
    (「해지연결상태」·「최신유효본여부」)은 코퍼스에도 없어 여기서 유도할 수 없다.
    """

    if not isinstance(value, str):
        return value
    text = _SLOT_UNIT.sub("", _SLOT_PREFIX.sub("", value.strip()))
    return re.sub(r"\s+", "", text) or value.strip()


def normalize_slot_names(values: "list[str] | tuple[str, ...] | None") -> list:
    """중복을 없애며 정규화한다. 순서는 보존한다 — 답 항목의 순서다."""

    out: list[str] = []
    for value in values or ():
        name = normalize_slot_name(value)
        if name and name not in out:
            out.append(name)
    return out


def primary_scope_for_period(
        resolver: object, *, corp_code: str, period_end: str, as_of: str,
        ) -> str | None:
    """**그 회사가 그 보고기간에 쓴 주재무제표 기준.** 정할 수 없으면 ``None``.

    K-IFRS 에서 연결재무제표를 작성하는 회사는 그것이 주재무제표이고 별도는 부속
    으로 함께 제출된다. 그래서 **그 기간 보고서에 연결이 실려 있으면 연결**이
    주 보고 기준이고, 없으면 별도다.

    **회사 단위 영구 기본값으로 두면 안 된다** (독립 검수 3-4). 실측에서 같은
    회사가 기간에 따라 달랐다.

    ```
    두산퓨얼셀   2023 문서 4개 별도만 · 2024 이후 연결 있음
    시프트업     2024~25 문서 5개 별도만 · 다른 시점 연결 있음
    ```

    그래서 키는 `(corp_code, period_end)` 다. 근거는 **그 기간에 제출된 fact 의
    scope 구성**이며 정책 표가 아니다 — 정본에 주재무제표 표시 컬럼이 없어
    문서에서 파생한다. 문서 1,051개 중 1,042개가 연결·별도를 함께 싣고 9개가
    별도만 싣는 것으로 이 파생이 검증된다.

    개념을 보지 않는다. 개념별로 물으면 「그 개념이 연결에만 있다」와 「이 회사가
    연결을 작성한다」가 섞인다 — 앞의 것은 계정 성질이고 뒤의 것이 보고 기준이다.
    """

    query = getattr(resolver, "facts", None)
    if not callable(query):
        return None
    try:
        rows = query(corp_code, as_of=as_of, period_end=period_end)
    except Exception:
        return None
    scopes = {
        scope for row in rows or ()
        if isinstance(scope := getattr(row, "scope", None), str)
    }
    if not scopes:
        return None
    if "CFS" in scopes:
        return "CFS"
    if "SFS" in scopes:
        return "SFS"
    return None


def primary_scope(available: tuple[str, ...]) -> str | None:
    """가용 후보만 보고 정하는 대체 규칙. **기간 정본을 쓸 수 없을 때만** 쓴다.

    `primary_scope_for_period` 가 우선이다. 이쪽은 개념 단위 가용성만 보므로
    「그 개념이 연결에만 있다」와 「이 회사가 연결을 작성한다」를 구분하지 못한다.
    """

    if "CFS" in available:
        return "CFS"
    if "SFS" in available:
        return "SFS"
    return None


class FixturePlanner:
    """실제 LLM 전 단계에서 계약과 역질문을 검증하는 adapter."""

    def __init__(self, company_resolver: CompanyResolver, *, corpus_cutoff: str,
                 reference_date: date) -> None:
        self.company_resolver = company_resolver
        self.corpus_cutoff = validate_as_of(corpus_cutoff, field_name="corpus_cutoff")
        self.reference_date = reference_date

    def financial(self, draft: FinancialDraftFixture) -> PlanValidation:
        company_query = COMPANY_ALIASES.get(_key(draft.company_text), draft.company_text)
        companies = self.company_resolver.resolve_company(company_query)
        if not companies:
            held = _held_company_candidates(self.company_resolver, company_query)
            if held:
                path = "draft.company_text"
                return PlanValidation(
                    status="needs_clarification",
                    clarification=ClarificationRequest(
                        request_id="clarify-company",
                        type="SELECT_ONE",
                        reason_code="company_surface_multiple_candidates",
                        plan_revision=0,
                        field_paths=[path],
                        question="어느 회사를 뜻하나요?",
                        options={path: [
                            ClarificationOption(
                                value=row.corp_name, label=row.corp_name)
                            for row in held
                        ]},
                        patch_paths=[path],
                    ),
                )
            return PlanValidation(
                status="out_of_scope",
                reasons=[f"지원 코퍼스에서 회사를 찾지 못했습니다: {draft.company_text}"],
            )
        if len(companies) > 1:
            path = "draft.company_text"
            return PlanValidation(
                status="needs_clarification",
                clarification=ClarificationRequest(
                    request_id="clarify-company",
                    type="SELECT_ONE",
                    reason_code="company_surface_multiple_candidates",
                    plan_revision=0,
                    field_paths=[path],
                    question="어느 회사를 뜻하나요?",
                    options={path: [
                        ClarificationOption(value=row.corp_name, label=row.corp_name)
                        for row in companies
                    ]},
                    patch_paths=[path],
                ),
            )

        concept = resolve_metric_concept(draft.metric_text)
        if concept is None:
            return PlanValidation(
                status="unsupported_request",
                reasons=[f"지원하지 않는 재무 concept입니다: {draft.metric_text}"],
            )
        if draft.scope is None:
            # 질문에 연결·별도가 없다. 그 계정에 **실제로 존재하는** scope 를 보고
            # 하나뿐이면 확정한다. 둘이면 **그 회사의 주재무제표 기준**으로 확정한다
            # (2026-08-18 결정 · `primary_scope` 주석에 근거와 남는 위험).
            #
            # 「질문에 없으면 연결」이 아니다. 연결을 작성하는 회사는 연결이
            # 주재무제표이고, 연결을 만들지 않는 회사에서는 자동으로 별도가 된다.
            # 어느 기준에도 값이 없으면(금융지주의 revenue) 확정하지 않고 묻는다 —
            # 기준을 골라도 값이 나오지 않기 때문이다.
            available, queried = _scope_availability(
                self.company_resolver, corp_code=companies[0].corp_code,
                concept=concept, as_of=self.corpus_cutoff)
            # **사용자가 채울 수 없는 공백은 역질문이 아니다** (검수 P1-01).
            # 그 계정이 연결·별도 어느 쪽에도 없으면 사용자가 무엇을 골라도 ready 로
            # 갈 수 없다. 되물으면 무효 역질문·반복 질문이 되고 UX 지표까지 오염된다.
            # 금융지주 6곳에 `revenue` 를 묻는 경우가 실제로 여기 걸린다.
            if queried and not available:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=[f"이 회사에는 해당 계정 값이 없습니다: "
                             f"{getattr(concept, 'value', concept)}"],
                )
            chosen = (available[0] if len(available) == 1
                      else primary_scope(available))
            if chosen is not None:
                draft = draft.model_copy(update={"scope": chosen})
            else:
                path = "draft.scope"
                return PlanValidation(
                    status="needs_clarification",
                    clarification=ClarificationRequest(
                        request_id="clarify-scope",
                        type="SELECT_ONE",
                        reason_code="scope_absent_multiple_available",
                        plan_revision=0,
                        field_paths=[path],
                        question="연결재무제표와 별도재무제표 중 어느 기준으로 볼까요?",
                        options={path: [
                            ClarificationOption(value=scope, label=label)
                            for scope, label in (("CFS", "연결"), ("SFS", "별도"))
                            if not available or scope in available
                        ]},
                        patch_paths=[path],
                    ),
                )
        scope = SCOPE_ALIASES.get(_key(draft.scope))
        if scope is None:
            return PlanValidation(
                status="unsupported_request",
                reasons=[f"scope는 연결(CFS) 또는 별도(SFS)여야 합니다: {draft.scope}"],
            )
        if draft.view not in {"as_filed", "restated"}:
            return PlanValidation(
                status="unsupported_request",
                reasons=[f"지원하지 않는 view입니다: {draft.view}"],
            )
        as_of = draft.as_of or self.corpus_cutoff
        try:
            validate_as_of(as_of)
        except ValueError as exc:
            return PlanValidation(status="unsupported_request", reasons=[str(exc)])
        if as_of > self.corpus_cutoff:
            return PlanValidation(
                status="unsupported_request",
                reasons=["as_of는 corpus_cutoff 이후일 수 없습니다"],
            )

        company = companies[0]
        period = _concept_period(concept, draft.year)
        fact = FactSpec(
            output_id=f"{company.corp_code}_{concept.value}_{draft.year}_{scope.lower()}",
            corp_code=company.corp_code,
            corp_name=company.corp_name,
            concept=concept,
            scope=scope,
            **period,
        )
        # v0.4 계약의 기재 순서는 view → as_of 이고, as_of 는 **어느 시점을
        # 썼는지**까지 적는다. 값을 빼면 읽는 쪽이 어느 코퍼스 끝을 말하는지 알 수
        # 없다.
        defaults: list[str] = []
        if draft.view == "restated":
            defaults.append("view=restated")
        if draft.as_of is None:
            defaults.append(f"as_of=corpus_cutoff({self.corpus_cutoff})")
        plan = ResolvedQueryPlan(
            revision=0,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            tasks=[ResolvedFinancialTask(
                task_id="financial-1",
                as_of=as_of,
                view=draft.view,
                facts=[fact],
            )],
            applied_defaults=defaults,
        )
        return PlanValidation(status="ready", plan=plan)


class DraftPlanResolver:
    """비공개 Draft를 실행 가능한 plan 또는 typed 역질문으로 바꾼다."""

    def __init__(self, company_resolver: CompanyResolver, *, corpus_cutoff: str,
                 reference_date: date,
                 event_preflight: EventKeyPreflight | None = None,
                 correction_preflight: CorrectionSeedPreflight | None = None,
                 periodic_document_preflight: (
                     PeriodicDocumentPreflight | None) = None,
                 ) -> None:
        self.company_resolver = company_resolver
        self.corpus_cutoff = validate_as_of(
            corpus_cutoff, field_name="corpus_cutoff")
        self.reference_date = reference_date
        self.event_preflight = event_preflight
        self.correction_preflight = correction_preflight
        self.periodic_document_preflight = periodic_document_preflight

    @staticmethod
    def _clarification(
            *, revision: int, path: str, question: str,
            options: list[ClarificationOption] | None = None,
            reason_code: str | None = None,
            ) -> PlanValidation:
        """단일 field 역질문. `type`·`reason_code` 는 **여기서 파생한다.**

        호출부가 20곳이라 손으로 붙이면 서로 어긋난다. 후보 유무가 `type` 을
        정하고, field 이름과 후보 유무가 기본 `reason_code` 를 정한다 —
        평가기가 13건의 역질문을 「어느 field 를 어떤 모양으로 물었나」로 묶을 수
        있으면 충분하다. 더 구체적인 사유가 있는 곳은 `reason_code` 로 덮는다.
        """

        field_name = path.rsplit(".", 1)[-1]
        candidates = {path: options or []}
        shape = ClarificationRequest.shape_for(candidates, [path])
        if reason_code is None:
            suffix = ("unresolved" if shape == "PROVIDE_VALUE"
                      else "multiple_candidates")
            reason_code = f"{reason_token(field_name)}_{suffix}"
        return PlanValidation(
            status="needs_clarification",
            clarification=ClarificationRequest(
                request_id=f"clarify-r{revision}-{field_name}",
                plan_revision=revision,
                type=shape, reason_code=reason_code,
                field_paths=[path], question=question,
                options=candidates, patch_paths=[path],
            ),
        )

    def _company(
            self, value: str | None, *, path: str, revision: int,
            ) -> tuple[CompanyLike | None, PlanValidation | None]:
        if value is None or not value.strip():
            return None, self._clarification(
                revision=revision, path=path,
                question="어느 회사를 조회할까요?",
            )
        company_query = COMPANY_ALIASES.get(_key(value), value.strip())
        companies = self.company_resolver.resolve_company(company_query)
        if not companies:
            held = _held_company_candidates(self.company_resolver, company_query)
            if held:
                return None, self._clarification(
                    revision=revision, path=path,
                    question="어느 회사를 뜻하나요?",
                    options=[ClarificationOption(
                        value=row.corp_name, label=row.corp_name)
                        for row in held],
                )
            return None, PlanValidation(
                status="out_of_scope",
                reasons=[f"지원 코퍼스에서 회사를 찾지 못했습니다: {value}"],
            )
        if len(companies) > 1:
            return None, self._clarification(
                revision=revision, path=path,
                question="어느 회사를 뜻하나요?",
                options=[ClarificationOption(
                    value=row.corp_name, label=row.corp_name)
                    for row in companies],
            )
        return companies[0], None

    def _event_name_fields(
            self, corp_code: str, contract_name: "str | None",
            ) -> "tuple[str | None, list[str]]":
        """계약 표기를 `contract_name` 과 `keywords` 중 **맞는 칸**에 넣는다.

        두 칸은 무게가 다르다. `contract_name` 은 preflight 의 하드 필터라 코퍼스가
        모르는 값을 넣으면 사건이 하나도 안 걸려 질의가 죽는다. `keywords` 는
        Stage1 에서 아무것도 거르지 않는 힌트다.

        **검색에 쓰는 값은 바꾸지 않는다.** 여기서 정하는 것은 계획에 무엇으로
        적어 내보낼지뿐이다. 「배터리」로 후보를 좁히는 일은 그대로 하고, 계획에는
        그것이 특정 계약 이름이 아니라 범주어라는 사실을 적는다.
        """

        value = (contract_name or "").strip()
        if not value:
            return None, []
        keywords = [word for word in value.split() if word]
        # The canonical selector index stores the category ``배터리 공급``
        # without the generic contract suffix, while public event selectors
        # retain the question surface ``배터리 공급계약``.  Restore that
        # lexical suffix only for this bounded category form so keyword
        # execution does not silently change from ``공급계약`` to ``공급``.
        if keywords and keywords[-1] == "공급":
            keywords[-1] = "공급계약"
        decide = getattr(self.event_preflight, "names_a_single_contract", None)
        if decide is None:
            return None, keywords
        try:
            identifies = decide(corp_code=corp_code, surface=value)
        except Exception:                                  # noqa: BLE001
            # **확인에 실패하면 하드 필터를 걸지 않는다** (9차 검수 P2-RESOLVER-001).
            # 예전에는 여기서 contract_name 을 그대로 돌려줘서, 분류기 장애가
            # 곧바로 가장 공격적인 필터로 바뀌었다 — 이 함수가 세운 원칙과
            # 정반대였다. 확인 못 한 말은 거르지 않는 칸으로 보낸다.
            return None, keywords
        if identifies:
            return contract_name, []
        # 범주어는 질문에 적힌 낱말 그대로 나눠 담는다 — 새 말을 만들지 않는다.
        return None, keywords

    def _as_of(self, value: str | None) -> tuple[str | None, PlanValidation | None]:
        as_of = value or self.corpus_cutoff
        try:
            validate_as_of(as_of)
        except ValueError as exc:
            return None, PlanValidation(
                status="unsupported_request", reasons=[str(exc)])
        if as_of > self.corpus_cutoff:
            return None, PlanValidation(
                status="unsupported_request",
                reasons=["as_of는 corpus_cutoff 이후일 수 없습니다"],
            )
        return as_of, None

    def _financial(
            self, draft: DraftFinancialTask, *, task_index: int,
            revision: int) -> PlanValidation:
        prefix = f"draft.tasks[{task_index}]"
        if draft.year is not None and draft.period_expression is not None:
            return PlanValidation(
                status="unsupported_request",
                reasons=[
                    "재무 기간은 year와 period_expression 중 하나로만 지정해야 합니다"],
            )

        # 서로 독립적인 두 누락축은 한 번에 묻는다. 이미 주어진 지표와 기간이
        # 유효할 때만 묶어, 잘못된 provider 값을 사용자 답으로 덮지 않는다.
        metric = (
            resolve_metric_concept(draft.metric_text)
            if draft.metric_text is not None and draft.metric_text.strip()
            else None
        )
        period_supplied = (
            draft.year is not None or draft.period_expression is not None)
        if ((not draft.company_text or not draft.company_text.strip())
                and draft.scope is None and metric is not None
                and period_supplied):
            _, period_error = _financial_period(
                metric, year=draft.year, expression=draft.period_expression,
                reference_date=self.reference_date)
            if period_error is not None:
                return PlanValidation(
                    status="unsupported_request", reasons=[period_error])
            company_path = f"{prefix}.company_text"
            scope_path = f"{prefix}.scope"
            paths = [company_path, scope_path]
            return PlanValidation(
                status="needs_clarification",
                clarification=ClarificationRequest(
                    request_id=f"clarify-r{revision}-company-scope",
                    # 회사는 후보가 없고 scope 는 있다. 묶는 제약은
                    # 「회사를 받아야 한다」쪽이므로 PROVIDE_VALUE 다.
                    type="PROVIDE_VALUE",
                    reason_code="company_absent_with_scope_absent",
                    plan_revision=revision,
                    field_paths=paths,
                    question="어느 회사의 수치를 연결·별도 중 어느 기준으로 볼까요?",
                    options={
                        company_path: [],
                        scope_path: [
                            ClarificationOption(value="CFS", label="연결"),
                            ClarificationOption(value="SFS", label="별도"),
                        ],
                    },
                    patch_paths=paths,
                ),
            )

        company, issue = self._company(
            draft.company_text, path=f"{prefix}.company_text",
            revision=revision)
        if issue is not None:
            return issue
        assert company is not None

        if draft.metric_text is None or not draft.metric_text.strip():
            path = f"{prefix}.metric_text"
            # **후보를 손으로 적지 않는다.** 예전에는 개념 5종을 선택지로 냈는데,
            # 개념이 56종으로 늘어난 뒤에는 그 목록이 **답 공간을 5개로 묶는**
            # 결함이 됐다 — 「자본총계가 얼마야」류 질문에서 이 역질문이 뜨면
            # 정답이 목록에 없다. 자유 입력으로 열어 두면 정본 계정 사전의
            # 표기 118개가 전부 답이 될 수 있고, 사전 밖 계정은 그다음 회차에서
            # fail-closed 로 거절된다.
            #
            # 질문 안의 예시는 **답 공간이 아니라 예시**다. 세 재무제표를 하나씩
            # 덮어 「무엇이든 계정 이름을 적으면 된다」를 알리는 용도이며, 늘어난
            # 개념을 여기 적어 넣을 필요가 없다.
            return self._clarification(
                revision=revision, path=path,
                question=(
                    "어떤 재무지표를 조회할까요? 계정 이름을 알려주세요 "
                    "(예: 매출, 자본총계, 영업활동현금흐름)."),
            )
        concept = resolve_metric_concept(draft.metric_text)
        if concept is None:
            # 승인 구어가 후보를 여럿 가리키면(`벌었어` → 영업이익·순이익·매출)
            # **거절하지 않고 그 후보로 묻는다.** 회사 쪽 held_company_candidates
            # 와 같은 처리다 — 자동 확정하지 않는 것이 그 표기의 승인 내용이고,
            # 우리는 후보를 알고 있으므로 거절 대신 물어야 한다.
            candidates = held_metric_candidates(draft.metric_text)
            if candidates:
                path = f"{prefix}.metric_text"
                return self._clarification(
                    revision=revision, path=path,
                    question="어떤 계정을 말씀하시는 건가요?",
                    options=[
                        ClarificationOption(
                            value=candidate.value, label=candidate.value)
                        for candidate in candidates
                    ],
                    reason_code="metric_text_multiple_candidates",
                )
            return PlanValidation(
                status="unsupported_request",
                reasons=[f"지원하지 않는 재무 concept입니다: {draft.metric_text}"],
            )
        if draft.year is None and draft.period_expression is None:
            path = f"{prefix}.year"
            return self._clarification(
                revision=revision, path=path,
                question="어느 회계연도를 조회할까요?",
                options=[ClarificationOption(
                    value=year, label=f"{year}년")
                    for year in (self.reference_date.year - 1,
                                 self.reference_date.year - 2)],
            )
        if draft.scope is None:
            # **가용 scope 를 보고 정한다.** 예전에는 이 경로가 그것을 보지도 않고
            # 바로 물었다 — `_available_scopes` 는 fixture planner 쪽에만 있었고,
            # 실제 실행 경로인 여기에는 없었다. 그래서 「연결」이라고 쓰지 않은 모든
            # 재무 질문이 역질문이 됐다.
            #
            # 후보가 하나면 그것, 둘이면 **그 회사의 주재무제표 기준**으로 확정한다
            # (2026-08-18 결정 · `primary_scope` 주석에 근거와 남는 위험).
            # 어느 기준에도 값이 없으면 확정하지 않고 묻는다 — 기준을 골라도 값이
            # 나오지 않기 때문이다.
            available, queried = _scope_availability(
                self.company_resolver, corp_code=company.corp_code,
                concept=concept, as_of=self.corpus_cutoff)
            # **사용자가 채울 수 없는 공백은 역질문이 아니다** (검수 P1-01).
            # 그 계정이 연결·별도 어느 쪽에도 없으면 사용자가 무엇을 골라도 ready 로
            # 갈 수 없다. 되물으면 무효 역질문·반복 질문이 되고 UX 지표까지 오염된다.
            # 금융지주 6곳에 `revenue` 를 묻는 경우가 실제로 여기 걸린다.
            if queried and not available:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=[f"이 회사에는 해당 계정 값이 없습니다: "
                             f"{getattr(concept, 'value', concept)}"],
                )
            if len(available) == 1:
                chosen = available[0]
            else:
                # **그 회사가 그 보고기간에 쓴 주재무제표 기준**으로 정한다
                # (독립 검수 3-4). 회사 단위 영구 기본값은 틀린다 — 같은 회사가
                # 기간에 따라 연결 작성 여부가 달라진다.
                period, _ = _financial_period(
                    concept, year=draft.year,
                    expression=draft.period_expression,
                    reference_date=self.reference_date)
                chosen = None
                if period is not None:
                    chosen = primary_scope_for_period(
                        self.company_resolver, corp_code=company.corp_code,
                        period_end=period["period_end"].isoformat(),
                        as_of=self.corpus_cutoff)
                # 기간 정본으로 못 정하면 개념 가용성으로 물러난다.
                if chosen is None:
                    chosen = primary_scope(available)
            if chosen is not None:
                draft = draft.model_copy(update={"scope": chosen})
            else:
                path = f"{prefix}.scope"
                return self._clarification(
                    revision=revision, path=path,
                    question="연결재무제표와 별도재무제표 중 어느 기준으로 볼까요?",
                    options=[
                        ClarificationOption(value="CFS", label="연결"),
                        ClarificationOption(value="SFS", label="별도"),
                    ],
                    reason_code="scope_absent_no_value_in_either",
                )
        scope = SCOPE_ALIASES.get(_key(draft.scope))
        if scope is None:
            return PlanValidation(
                status="unsupported_request",
                reasons=[f"scope는 연결(CFS) 또는 별도(SFS)여야 합니다: {draft.scope}"],
            )
        if draft.view not in {"as_filed", "restated"}:
            return PlanValidation(
                status="unsupported_request",
                reasons=[f"지원하지 않는 view입니다: {draft.view}"],
            )
        as_of, issue = self._as_of(draft.as_of)
        if issue is not None:
            return issue
        assert as_of is not None

        period, period_error = _financial_period(
            concept, year=draft.year, expression=draft.period_expression,
            reference_date=self.reference_date)
        if period_error is not None or period is None:
            return PlanValidation(
                status="unsupported_request",
                reasons=[period_error or "재무 기간을 확정하지 못했습니다"],
            )
        period_end = period["period_end"]
        assert isinstance(period_end, date)
        if period_end.strftime("%Y%m%d") > as_of:
            return PlanValidation(
                status="unsupported_request",
                reasons=["재무 기간은 as_of 이후일 수 없습니다"],
            )
        mode = (
            "instant" if period["cumulative"] is None
            else "cum" if period["cumulative"] else "discrete")
        fact = FactSpec(
            output_id=(
                f"{company.corp_code}_{concept.value}_"
                f"{period_end:%Y%m%d}_{mode}_{scope.lower()}"),
            corp_code=company.corp_code, corp_name=company.corp_name,
            concept=concept, scope=scope, **period,
        )
        # v0.4 계약의 기재 순서는 view → as_of 이고, as_of 는 **어느 시점을
        # 썼는지**까지 적는다. 값을 빼면 읽는 쪽이 어느 코퍼스 끝을 말하는지 알 수
        # 없다.
        defaults: list[str] = []
        if draft.view == "restated":
            defaults.append("view=restated")
        if draft.as_of is None:
            defaults.append(f"as_of=corpus_cutoff({self.corpus_cutoff})")
        return PlanValidation(status="ready", plan=ResolvedQueryPlan(
            revision=revision, reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            tasks=[ResolvedFinancialTask(
                task_id=f"financial-{task_index + 1}", as_of=as_of,
                view=draft.view, facts=[fact])],
            applied_defaults=defaults,
        ))

    def _narrative(
            self, draft: DraftNarrativeTask, *, task_index: int,
            revision: int) -> PlanValidation:
        prefix = f"draft.tasks[{task_index}]"
        company, issue = self._company(
            draft.company_text, path=f"{prefix}.company_text",
            revision=revision)
        if issue is not None:
            return issue
        assert company is not None
        if draft.retrieval_query is None or not draft.retrieval_query.strip():
            return self._clarification(
                revision=revision, path=f"{prefix}.retrieval_query",
                question="공시 본문에서 어떤 내용을 찾을까요?",
            )
        # 「…12월 16일까지 공개된 공시만 보면」의 날짜는 조회 대상 기간이 아니라
        # 정보 cutoff 다. 모델이 대상기간 쪽에 넣었으면 여기서 역할을 되돌린다.
        moved = _cutoff_from_target_period(
            draft.as_of, draft.target_period_expressions)
        if moved is not None:
            draft = draft.model_copy(update={
                "as_of": moved[0], "target_period_expressions": moved[1]})
        as_of, issue = self._as_of(draft.as_of)
        if issue is not None:
            return issue
        assert as_of is not None

        if (draft.doc_group is not None
                and draft.doc_group not in _DOCUMENT_GROUPS):
            return PlanValidation(
                status="unsupported_request",
                reasons=[f"지원하지 않는 document group입니다: {draft.doc_group}"],
            )
        if len(draft.target_period_expressions) > 2:
            return PlanValidation(
                status="unsupported_request",
                reasons=["narrative compare는 target period 두 개까지만 지원합니다"],
            )
        if (len(draft.target_period_expressions) == 2
                and draft.operation != "compare"):
            return PlanValidation(
                status="unsupported_request",
                reasons=["복수 target period는 narrative compare에서만 지원합니다"],
            )

        selected_receipt = draft.selected_document_receipt
        periods: list[DateRange] = []
        if draft.target_period_expressions:
            if draft.doc_group not in {None, "periodic"}:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["정기보고서 기간과 document group이 충돌합니다"],
                )
            if self.periodic_document_preflight is None:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["periodic_document_preflight_unavailable"],
                )
            if len(draft.target_period_expressions) == 2:
                # Receipt is a one-document selector.  For a comparison, only
                # verify each grounded report and preserve the public form plus
                # the two requested period ranges; never choose one receipt.
                if selected_receipt is not None:
                    return PlanValidation(
                        status="unsupported_request",
                        reasons=["narrative_compare_receipt_cannot_select_two_periods"],
                    )
                candidates = []
                for expression in draft.target_period_expressions:
                    resolution = (
                        self.periodic_document_preflight.resolve_periodic_document(
                            corp_code=company.corp_code, as_of=as_of,
                            target_expression=expression,
                        ))
                    if resolution.status == "not_found":
                        return PlanValidation(
                            status="out_of_scope",
                            reasons=["periodic_document_not_found_in_corpus"],
                        )
                    if resolution.status == "unsupported":
                        return PlanValidation(
                            status="unsupported_request",
                            reasons=["periodic_document_selector_unsupported"],
                        )
                    if resolution.status == "ambiguous":
                        return PlanValidation(
                            status="unsupported_request",
                            reasons=["narrative_compare_periodic_document_ambiguous"],
                        )
                    period, period_error = _narrative_period_range(
                        expression, reference_date=self.reference_date)
                    if period_error is not None or period is None:
                        return PlanValidation(
                            status="unsupported_request",
                            reasons=[period_error or "narrative_period_unsupported"],
                        )
                    candidates.append(resolution.candidate)
                    periods.append(period)
                if any(candidate is None for candidate in candidates):
                    return PlanValidation(
                        status="unsupported_request",
                        reasons=["periodic_document_resolution_invalid"],
                    )
                forms = {
                    candidate.form for candidate in candidates
                    if candidate is not None
                }
                if len(forms) != 1:
                    return PlanValidation(
                        status="unsupported_request",
                        reasons=["narrative_compare_periodic_form_mismatch"],
                    )
                form = _NARRATIVE_PERIODIC_FORM.get(next(iter(forms)))
                if form is None:
                    return PlanValidation(
                        status="unsupported_request",
                        reasons=["narrative_compare_periodic_form_unsupported"],
                    )
                draft = draft.model_copy(update={"doc_group": "periodic"})
            else:
                resolution = (
                    self.periodic_document_preflight.resolve_periodic_document(
                        corp_code=company.corp_code,
                        as_of=as_of,
                        target_expression=draft.target_period_expressions[0],
                        selected_receipt=selected_receipt,
                    ))
                if resolution.status == "not_found":
                    return PlanValidation(
                        status="out_of_scope",
                        reasons=["periodic_document_not_found_in_corpus"],
                    )
                if resolution.status == "unsupported":
                    return PlanValidation(
                        status="unsupported_request",
                        reasons=["periodic_document_selector_unsupported"],
                    )
                if resolution.status == "ambiguous":
                    path = f"{prefix}.selected_document_receipt"
                    return self._clarification(
                        revision=revision,
                        path=path,
                        question="어느 정기보고서를 기준으로 내용을 찾을까요?",
                        options=[ClarificationOption(
                            value=row.rcept_no, label=row.label)
                            for row in resolution.candidates],
                        reason_code="selected_document_receipt_multiple_candidates",
                    )
                candidate = resolution.candidates[0]
                selected_receipt = candidate.rcept_no
                draft = draft.model_copy(update={"doc_group": "periodic"})
        elif selected_receipt is not None:
            return PlanValidation(
                status="unsupported_request",
                reasons=["narrative_receipt_requires_target_period"],
            )

        selector = (
            DocumentSelector(doc_group="periodic", form=form)
            if len(draft.target_period_expressions) == 2 else
            DocumentSelector(
                rcept_no=selected_receipt,
                doc_group=draft.doc_group,
            ) if draft.doc_group or selected_receipt else None
        )
        defaults = ([f"as_of=corpus_cutoff({self.corpus_cutoff})"]
                    if draft.as_of is None else [])
        requested_slots = (
            ["search_hits"]
            if draft.requested_slots is None and draft.operation == "search"
            else list(draft.requested_slots or [])
        )
        return PlanValidation(status="ready", plan=ResolvedQueryPlan(
            revision=revision, reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            tasks=[ResolvedNarrativeTask(
                task_id=f"narrative-{task_index + 1}",
                operation=draft.operation,
                corp_codes=[company.corp_code], corp_names=[company.corp_name],
                as_of=as_of,
                retrieval_query=draft.retrieval_query.strip(),
                document_selector=selector,
                periods=periods,
                requested_slots=normalize_slot_names(requested_slots),
            )],
            applied_defaults=defaults,
        ))

    def _document(
            self, draft: DraftDocumentTask, *, task_index: int,
            revision: int) -> PlanValidation:
        prefix = f"draft.tasks[{task_index}]"
        company, issue = self._company(
            draft.company_text, path=f"{prefix}.company_text",
            revision=revision)
        if issue is not None:
            return issue
        assert company is not None

        as_of, issue = self._as_of(draft.as_of)
        if issue is not None:
            return issue
        assert as_of is not None

        if (draft.doc_group is not None
                and draft.doc_group not in _DOCUMENT_GROUPS):
            return PlanValidation(
                status="unsupported_request",
                reasons=[f"지원하지 않는 document group입니다: {draft.doc_group}"],
            )
        if len(draft.target_period_expressions) > 1:
            return PlanValidation(
                status="unsupported_request",
                reasons=["단일 document task의 복수 target period는 지원하지 않습니다"],
            )

        selected = draft.selected_document_receipt
        seed = (draft.seed_receipt_text.strip()
                if draft.seed_receipt_text else None)
        if seed is not None and not re.fullmatch(r"[0-9]{14}", seed):
            return PlanValidation(
                status="unsupported_request",
                reasons=["문서 접수번호는 14자리 숫자여야 합니다"],
            )
        if selected is not None and seed is not None and selected != seed:
            return PlanValidation(
                status="unsupported_request",
                reasons=["사용자 선택 접수번호와 proposal 접수번호가 충돌합니다"],
            )
        receipt = selected or seed

        report_fragment: str | None = None
        if draft.target_period_expressions:
            expression = draft.target_period_expressions[0]
            report_fragment, choices, error = _document_report_fragment(expression)
            if error is not None:
                return PlanValidation(
                    status="unsupported_request", reasons=[error])
            if choices:
                path = f"{prefix}.target_period_expressions"
                return self._clarification(
                    revision=revision, path=path,
                    question="어느 종류의 정기보고서를 찾을까요?",
                    options=[ClarificationOption(value=value, label=value)
                             for value in choices],
                )

        doc_group = draft.doc_group
        if report_fragment is not None:
            if doc_group is not None and doc_group != "periodic":
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["정기보고서 기간과 document group이 충돌합니다"],
                )
            doc_group = "periodic"

        history_form: str | None = None
        history_rcept_from: str | None = None
        history_rcept_to: str | None = None
        if draft.operation == "version_history" and receipt is None:
            expression = (
                draft.target_period_expressions[0]
                if draft.target_period_expressions else ""
            )
            annual = re.fullmatch(
                r"\s*(20[0-9]{2})년?\s*(?:사업보고서|연간보고서)\s*",
                unicodedata.normalize("NFC", expression),
            )
            if annual is None:
                return self._clarification(
                    revision=revision,
                    path=f"{prefix}.selected_document_receipt",
                    question=(
                        "어느 공시 문서의 정정 이력을 볼까요? "
                        "접수번호를 알려주세요."),
                )
            # Annual reports for fiscal year Y are filed from Y+1.  Search the
            # visible filing window rather than choosing one receipt, because
            # version history needs the original and all correction versions.
            history_rcept_from = f"{int(annual.group(1)) + 1:04d}0101"
            history_rcept_to = as_of
            if history_rcept_from > history_rcept_to:
                return PlanValidation(
                    status="out_of_scope",
                    reasons=["periodic_version_history_not_yet_in_corpus"],
                )
            history_form = "사업보고서"
            doc_group = "periodic"
            report_fragment = None

        event_type = draft.event_type_text
        counterparty = draft.counterparty_text
        contract_name = draft.contract_name_text
        selector = DocumentSelector(
            rcept_no=receipt,
            doc_group=doc_group,
            event_type=event_type,
            form=history_form,
            report_name_contains=report_fragment,
            rcept_from=history_rcept_from,
            rcept_to=history_rcept_to,
        )
        event_selector = None
        if any((event_type, counterparty, contract_name)):
            named, name_keywords = self._event_name_fields(
                company.corp_code, contract_name)
            event_selector = EventSelector(
                event_type=event_type,
                counterparty=counterparty,
                contract_name=named,
                keywords=name_keywords,
            )
        if (not selector.has_identity_condition()
                and (event_selector is None
                     or not event_selector.has_identity_condition())):
            path = f"{prefix}.target_period_expressions"
            return self._clarification(
                revision=revision, path=path,
                question="어떤 보고서를 찾을까요?",
                options=[ClarificationOption(value=value, label=value)
                         for value in _document_period_options("")],
            )

        defaults = ([f"as_of=corpus_cutoff({self.corpus_cutoff})"]
                    if draft.as_of is None else [])
        return PlanValidation(status="ready", plan=ResolvedQueryPlan(
            revision=revision, reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            tasks=[ResolvedDocumentTask(
                task_id=f"document-{task_index + 1}",
                operation=draft.operation,
                corp_code=company.corp_code,
                corp_name=company.corp_name,
                as_of=as_of,
                selector=selector,
                event_selector=event_selector,
            )],
            applied_defaults=defaults,
        ))

    def _disclosure(
            self, draft: DraftDisclosureTask, *, task_index: int,
            revision: int) -> PlanValidation:
        prefix = f"draft.tasks[{task_index}]"
        company, issue = self._company(
            draft.company_text, path=f"{prefix}.company_text",
            revision=revision)
        if issue is not None:
            return issue
        assert company is not None

        # 「…12월 16일까지 공개된 공시만 보면」의 날짜는 조회 대상 기간이 아니라
        # 정보 cutoff 다. 모델이 대상기간 쪽에 넣었으면 여기서 역할을 되돌린다.
        moved = _cutoff_from_target_period(
            draft.as_of, draft.target_period_expressions)
        if moved is not None:
            draft = draft.model_copy(update={
                "as_of": moved[0], "target_period_expressions": moved[1]})
        as_of, issue = self._as_of(draft.as_of)
        if issue is not None:
            return issue
        assert as_of is not None

        if (draft.doc_group is not None
                and draft.doc_group not in _DOCUMENT_GROUPS):
            return PlanValidation(
                status="unsupported_request",
                reasons=[f"지원하지 않는 document group입니다: {draft.doc_group}"],
            )
        if len(draft.target_period_expressions) > 1:
            return PlanValidation(
                status="unsupported_request",
                reasons=["단일 disclosure task의 복수 target period는 지원하지 않습니다"],
            )
        if not draft.requested_slots:
            return self._clarification(
                revision=revision,
                path=f"{prefix}.requested_slots",
                question="공시에서 어떤 항목을 조회할까요?",
            )
        if len(draft.result_labels) > len(draft.requested_slots):
            return PlanValidation(
                status="unsupported_request",
                reasons=["disclosure result label이 requested slot보다 많습니다"],
            )

        seed = draft.seed_receipt_text.strip() if draft.seed_receipt_text else None
        if seed is not None and re.fullmatch(r"[0-9]{14}", seed) is None:
            return PlanValidation(
                status="unsupported_request",
                reasons=["문서 접수번호는 14자리 숫자여야 합니다"],
            )

        event_from: str | None = None
        event_to: str | None = None
        if draft.target_period_expressions:
            event_from, event_to, error = _target_date_range(
                draft.target_period_expressions[0],
                reference_date=self.reference_date,
            )
            if error is not None:
                return PlanValidation(
                    status="unsupported_request", reasons=[error])

        event_type = draft.event_type_text
        funding_keywords = funding_decision_keywords(event_type or "")
        if funding_keywords:
            event_type = "주요사항보고"
        counterparty = draft.counterparty_text
        contract_name = draft.contract_name_text
        document_selector = None
        if seed is not None or draft.doc_group is not None:
            document_selector = DocumentSelector(
                rcept_no=seed,
                doc_group=draft.doc_group,
                event_type=event_type,
            )
        event_selector = None
        if any((event_type, counterparty, contract_name,
                event_from, event_to)):
            named, name_keywords = self._event_name_fields(
                company.corp_code, contract_name)
            event_selector = EventSelector(
                seed_rcept_no=(seed if draft.operation == "list" else None),
                event_type=event_type,
                counterparty=counterparty,
                contract_name=named,
                keywords=list(dict.fromkeys(
                    [*funding_keywords, *name_keywords])),
                event_from=event_from,
                event_to=event_to,
            )
        if (document_selector is None and event_selector is None):
            return self._clarification(
                revision=revision,
                path=f"{prefix}.event_type_text",
                question="어떤 종류의 공시를 조회할까요?",
            )

        # **슬롯 이름은 한 번만 정규화하고 두 곳이 같은 값을 쓴다.**
        # 예전에는 field_output 이 정규화 **전** 이름을, task 의 requested_slots 가
        # 정규화 **후** 이름을 써서, 번호·불릿·단위 괄호가 붙은 슬롯이 오면
        # `Event field_output slot은 requested_slots에 있어야 합니다` 로 계획 전체가
        # 죽었다. 역질문이 앞을 막고 있어 드러나지 않던 버그다.
        requested_slots = normalize_slot_names(draft.requested_slots)
        field_outputs = [
            FieldOutputSpec(
                output_id=label,
                slot=slot,
                value_kind=_slot_value_kind(slot),
            )
            for label, slot in zip(draft.result_labels, requested_slots)
        ] if draft.result_labels else []
        defaults = ([f"as_of=corpus_cutoff({self.corpus_cutoff})"]
                    if draft.as_of is None else [])
        return PlanValidation(status="ready", plan=ResolvedQueryPlan(
            revision=revision, reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            tasks=[ResolvedDisclosureTask(
                task_id=f"disclosure-{task_index + 1}",
                operation=draft.operation,
                corp_code=company.corp_code,
                corp_name=company.corp_name,
                as_of=as_of,
                document_selector=document_selector,
                event_selector=event_selector,
                requested_slots=requested_slots,
                field_outputs=field_outputs,
            )],
            applied_defaults=defaults,
        ))

    def _event(
            self, draft: DraftEventTask, *, task_index: int,
            revision: int) -> PlanValidation:
        prefix = f"draft.tasks[{task_index}]"
        company, issue = self._company(
            draft.company_text, path=f"{prefix}.company_text",
            revision=revision)
        if issue is not None:
            return issue
        assert company is not None

        # disclosure 와 같은 이유 — 「…12월 17일 기준」은 관측 시점이다.
        moved = _cutoff_from_target_period(
            draft.as_of_expression, draft.target_period_expressions)
        if moved is not None:
            draft = draft.model_copy(update={
                "as_of_expression": moved[0],
                "target_period_expressions": moved[1]})
        timepoints, defaulted, error = _event_timepoints(
            draft.as_of_expression, corpus_cutoff=self.corpus_cutoff)
        if error is not None:
            return PlanValidation(
                status="unsupported_request", reasons=[error])
        assert timepoints is not None

        if len(draft.target_period_expressions) > 1:
            # 「12월 25일과 26일 **각각** 그날까지 유효했나」는 관측 시점이 둘인
            # 질문이고, QueryPlan v0.4 는 이미 `timepoints` 배열로 지원한다.
            # 그런데 모델은 그 날짜들을 `target_period_expressions` 에 넣고
            # `as_of_expression` 을 비운다 — 사람이 읽으면 같은 뜻이다.
            #
            # **as_of 가 기본값일 때만** 그것들을 시점으로 받는다. as_of 를
            # 따로 적었는데 대상기간도 여럿이면 무엇이 관측 시점인지 정해지지
            # 않으므로 예전처럼 거절한다.
            if not defaulted:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["단일 event task의 복수 target period는 지원하지 않습니다"],
                )
            parsed, _, parse_error = _event_timepoints(
                "과".join(draft.target_period_expressions),
                corpus_cutoff=self.corpus_cutoff)
            if parse_error is not None or not parsed:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["단일 event task의 복수 target period는 지원하지 않습니다"],
                )
            timepoints = parsed
            multi_timepoints = True
        else:
            multi_timepoints = False
        event_from: str | None = None
        event_to: str | None = None
        if draft.target_period_expressions and not multi_timepoints:
            event_from, event_to, error = _target_date_range(
                draft.target_period_expressions[0],
                reference_date=self.reference_date,
            )
            if error is not None:
                return PlanValidation(
                    status="unsupported_request", reasons=[error])
            if event_to is not None and event_to > self.corpus_cutoff:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["Event 대상기간은 corpus_cutoff 이후일 수 없습니다"],
                )

        seed = draft.seed_receipt_text.strip() if draft.seed_receipt_text else None
        if seed is not None and re.fullmatch(r"[0-9]{14}", seed) is None:
            return PlanValidation(
                status="unsupported_request",
                reasons=["Event 접수번호는 14자리 숫자여야 합니다"],
            )
        if len(draft.result_labels) > len(draft.requested_slots):
            return PlanValidation(
                status="unsupported_request",
                reasons=["event result label이 requested slot보다 많습니다"],
            )

        event_type = (draft.event_type_text.strip()
                      if draft.event_type_text and draft.event_type_text.strip()
                      else None)
        counterparty = (draft.counterparty_text.strip()
                        if (draft.counterparty_text
                            and draft.counterparty_text.strip()) else None)
        contract_name = (draft.contract_name_text.strip()
                         if (draft.contract_name_text
                             and draft.contract_name_text.strip()) else None)
        if not any((seed, event_type, counterparty, contract_name,
                    event_from, event_to)):
            return self._clarification(
                revision=revision,
                path=f"{prefix}.event_type_text",
                question="어떤 사건을 조회할까요?",
            )

        event_key: str | None = None
        resolved_seed = seed
        if draft.operation in {"status", "timeline"}:
            if self.event_preflight is None:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["event_key_preflight_not_configured"],
                )
            resolution = self.event_preflight.resolve_event_key(
                corp_code=company.corp_code,
                as_of=max(timepoints),
                seed_rcept_no=seed,
                event_type=event_type,
                counterparty=counterparty,
                contract_name=contract_name,
                event_from=event_from,
                event_to=event_to,
            )
            if resolution.status == "not_found":
                return PlanValidation(
                    status="out_of_scope",
                    reasons=["event_not_found_in_corpus"],
                )
            if resolution.status == "too_many":
                return self._clarification(
                    revision=revision,
                    path=f"{prefix}.seed_receipt_text",
                    question="후보가 많습니다. 조회할 사건의 접수번호를 알려주세요.",
                )
            if resolution.status == "ambiguous":
                # **어느 후보를 골라도 답이 같으면 묻지 않는다.**
                #
                # 후보가 여럿이라는 사실만으로 되묻는 것은 지나치게 보수적이다.
                # 요청한 슬롯의 값이 후보들 사이에서 엇갈리지 않으면 사용자에게
                # 물어도 답이 달라지지 않는다. 값이 **없는** 후보는 침묵이지
                # 반증이 아니므로 값을 가진 후보만 비교한다.
                #
                # 「같은 계약이다」라고 주장하지 않는다 — 정본은 그 둘이 같은지
                # 확정하지 못했다고 했고 그 판정은 그대로 둔다. 여기서 쓰는 것은
                # 요청한 값이 서로 다르지 않다는 사실뿐이다.
                #
                # 출처가 모호하다는 사실은 **답에 실을 한계**이지 계획의 문제가
                # 아니다. Stage2 가 `event_key` 로 조회하면서 정본의
                # `identity_fingerprint` 로 스스로 발견한다.
                agreeing = getattr(
                    self.event_preflight, "agreeing_candidate", None)
                candidate = None
                if agreeing is not None:
                    try:
                        candidate = agreeing(
                            candidates=resolution.candidates,
                            as_of=max(timepoints),
                            slots=tuple(draft.requested_slots or ()))
                    except Exception:                      # noqa: BLE001
                        candidate = None
                if candidate is None:
                    return self._clarification(
                        revision=revision,
                        path=f"{prefix}.seed_receipt_text",
                        question="어느 사건을 뜻하나요?",
                        options=[ClarificationOption(
                            value=row.seed_rcept_no, label=row.label)
                            for row in resolution.candidates],
                    )
                event_key = candidate.event_key
                resolved_seed = candidate.seed_rcept_no
            if event_key is None:
                candidate = resolution.candidates[0]
                event_key = candidate.event_key
                resolved_seed = candidate.seed_rcept_no

        named, name_keywords = self._event_name_fields(
            company.corp_code, contract_name)
        selector = EventSelector(
            event_key=event_key,
            seed_rcept_no=resolved_seed,
            event_type=event_type,
            counterparty=counterparty,
            contract_name=named,
            keywords=name_keywords,
            event_from=event_from,
            event_to=event_to,
        )
        # 슬롯 이름은 **한 번만** 정규화하고 두 곳이 같은 값을 쓴다 (위 disclosure
        # 와 같은 이유). 정규화는 공백·번호·불릿·단위 괄호를 지우므로, 한쪽만
        # 정규화하면 `field_output slot은 requested_slots에 있어야 합니다` 로
        # 계획 전체가 죽는다.
        requested_slots = normalize_slot_names(draft.requested_slots)
        field_outputs = [
            FieldOutputSpec(
                output_id=label,
                slot=slot,
                value_kind=_slot_value_kind(slot),
            )
            for label, slot in zip(draft.result_labels, requested_slots)
        ] if draft.result_labels else []
        defaults = ["timepoints=corpus_cutoff"] if defaulted else []
        return PlanValidation(status="ready", plan=ResolvedQueryPlan(
            revision=revision,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            tasks=[ResolvedEventTask(
                task_id=f"event-{task_index + 1}",
                operation=draft.operation,
                corp_code=company.corp_code,
                corp_name=company.corp_name,
                selector=selector,
                timepoints=timepoints,
                requested_slots=requested_slots,
                field_outputs=field_outputs,
            )],
            applied_defaults=defaults,
        ))

    def _correction(
            self, draft: DraftCorrectionTask, *, task_index: int,
            revision: int) -> PlanValidation:
        prefix = f"draft.tasks[{task_index}]"
        company, issue = self._company(
            draft.company_text, path=f"{prefix}.company_text",
            revision=revision)
        if issue is not None:
            return issue
        assert company is not None

        as_of, issue = self._as_of(draft.as_of)
        if issue is not None:
            return issue
        assert as_of is not None
        if (draft.doc_group is not None
                and draft.doc_group not in _DOCUMENT_GROUPS):
            return PlanValidation(
                status="unsupported_request",
                reasons=[f"지원하지 않는 document group입니다: {draft.doc_group}"],
            )
        if len(draft.target_period_expressions) > 1:
            return PlanValidation(
                status="unsupported_request",
                reasons=["단일 correction task의 복수 target period는 지원하지 않습니다"],
            )
        if len(draft.result_labels) > len(draft.requested_slots):
            return PlanValidation(
                status="unsupported_request",
                reasons=["correction result label이 requested slot보다 많습니다"],
            )

        event_from: str | None = None
        event_to: str | None = None
        if draft.target_period_expressions:
            event_from, event_to, error = _target_date_range(
                draft.target_period_expressions[0],
                reference_date=self.reference_date,
            )
            if error is not None:
                return PlanValidation(
                    status="unsupported_request", reasons=[error])
            if event_to is not None and event_to > self.corpus_cutoff:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["Correction 대상기간은 corpus_cutoff 이후일 수 없습니다"],
                )

        seed = draft.seed_receipt_text.strip() if draft.seed_receipt_text else None
        if seed is not None and re.fullmatch(r"[0-9]{14}", seed) is None:
            return PlanValidation(
                status="unsupported_request",
                reasons=["Correction 접수번호는 14자리 숫자여야 합니다"],
            )
        event_type = (draft.event_type_text.strip()
                      if draft.event_type_text and draft.event_type_text.strip()
                      else None)
        counterparty = (draft.counterparty_text.strip()
                        if (draft.counterparty_text
                            and draft.counterparty_text.strip()) else None)
        contract_name = (draft.contract_name_text.strip()
                         if (draft.contract_name_text
                             and draft.contract_name_text.strip()) else None)

        document_selector: DocumentSelector | None = None
        event_selector: EventSelector | None = None
        unique_descendant_selected = False
        if seed is not None:
            if self.correction_preflight is None:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["correction_seed_preflight_not_configured"],
                )
            resolution = self.correction_preflight.resolve_seed(
                corp_code=company.corp_code,
                as_of=as_of,
                seed_rcept_no=seed,
                operation=draft.operation,
            )
            if resolution.status == "not_found":
                return PlanValidation(
                    status="out_of_scope",
                    reasons=["correction_document_not_found_in_corpus"],
                )
            if resolution.status == "invalid":
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["correction_lineage_invalid"],
                )
            if resolution.status == "ambiguous":
                return self._clarification(
                    revision=revision,
                    path=f"{prefix}.seed_receipt_text",
                    question="어느 정정공시를 비교할까요?",
                    options=[ClarificationOption(
                        value=row.rcept_no, label=row.label)
                        for row in resolution.candidates],
                )
            assert resolution.selected_receipt is not None
            unique_descendant_selected = (
                draft.operation == "diff"
                and resolution.selected_receipt != seed)
            document_selector = DocumentSelector(
                rcept_no=resolution.selected_receipt,
                doc_group=draft.doc_group,
                event_type=event_type,
            )
        else:
            if draft.operation == "diff":
                return self._clarification(
                    revision=revision,
                    path=f"{prefix}.seed_receipt_text",
                    question="비교할 정정공시의 접수번호를 알려주세요.",
                )
            if not any((event_type, counterparty, contract_name,
                        event_from, event_to)):
                return self._clarification(
                    revision=revision,
                    path=f"{prefix}.seed_receipt_text",
                    question="정정 이력을 볼 문서의 접수번호를 알려주세요.",
                )
            if self.event_preflight is None:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["event_key_preflight_not_configured"],
                )
            event_resolution = self.event_preflight.resolve_event_key(
                corp_code=company.corp_code,
                as_of=as_of,
                event_type=event_type,
                counterparty=counterparty,
                contract_name=contract_name,
                event_from=event_from,
                event_to=event_to,
            )
            if event_resolution.status == "not_found":
                return PlanValidation(
                    status="out_of_scope",
                    reasons=["correction_event_not_found_in_corpus"],
                )
            if event_resolution.status == "too_many":
                return self._clarification(
                    revision=revision,
                    path=f"{prefix}.seed_receipt_text",
                    question="후보가 많습니다. 정정 이력을 볼 접수번호를 알려주세요.",
                )
            if event_resolution.status == "ambiguous":
                return self._clarification(
                    revision=revision,
                    path=f"{prefix}.seed_receipt_text",
                    question="어느 사건의 정정 이력을 볼까요?",
                    options=[ClarificationOption(
                        value=row.seed_rcept_no, label=row.label)
                        for row in event_resolution.candidates],
                )
            candidate = event_resolution.candidates[0]
            named, name_keywords = self._event_name_fields(
                company.corp_code, contract_name)
            event_selector = EventSelector(
                event_key=candidate.event_key,
                seed_rcept_no=candidate.seed_rcept_no,
                event_type=event_type,
                counterparty=counterparty,
                contract_name=named,
                keywords=name_keywords,
                event_from=event_from,
                event_to=event_to,
            )
            if draft.doc_group is not None:
                document_selector = DocumentSelector(
                    doc_group=draft.doc_group)

        # 슬롯 이름은 **한 번만** 정규화하고 두 곳이 같은 값을 쓴다 (위 disclosure
        # 와 같은 이유). 정규화는 공백·번호·불릿·단위 괄호를 지우므로, 한쪽만
        # 정규화하면 `field_output slot은 requested_slots에 있어야 합니다` 로
        # 계획 전체가 죽는다.
        requested_slots = normalize_slot_names(draft.requested_slots)
        field_outputs = [
            FieldOutputSpec(
                output_id=label,
                slot=slot,
                value_kind=_slot_value_kind(slot),
            )
            for label, slot in zip(draft.result_labels, requested_slots)
        ] if draft.result_labels else []
        defaults = ([f"as_of=corpus_cutoff({self.corpus_cutoff})"]
                    if draft.as_of is None else [])
        if unique_descendant_selected:
            defaults.append("correction_receipt=unique_descendant")
        return PlanValidation(status="ready", plan=ResolvedQueryPlan(
            revision=revision,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            tasks=[ResolvedCorrectionTask(
                task_id=f"correction-{task_index + 1}",
                operation=draft.operation,
                corp_code=company.corp_code,
                corp_name=company.corp_name,
                as_of=as_of,
                document_selector=document_selector,
                event_selector=event_selector,
                requested_slots=requested_slots,
                field_outputs=field_outputs,
            )],
            applied_defaults=defaults,
        ))

    def resolve(self, draft: DraftQueryPlan, *, revision: int = 0) -> PlanValidation:
        if type(revision) is not int or revision < 0:
            raise ValueError("revision은 음이 아닌 정수여야 합니다")
        task = draft.tasks[0]
        if isinstance(task, DraftFinancialTask):
            return self._financial(task, task_index=0, revision=revision)
        if isinstance(task, DraftNarrativeTask):
            return self._narrative(task, task_index=0, revision=revision)
        if isinstance(task, DraftDocumentTask):
            return self._document(task, task_index=0, revision=revision)
        if isinstance(task, DraftDisclosureTask):
            return self._disclosure(task, task_index=0, revision=revision)
        if isinstance(task, DraftEventTask):
            return self._event(task, task_index=0, revision=revision)
        if isinstance(task, DraftCorrectionTask):
            return self._correction(task, task_index=0, revision=revision)
        return PlanValidation(
            status="unsupported_request", reasons=["지원하지 않는 Draft task입니다"])
