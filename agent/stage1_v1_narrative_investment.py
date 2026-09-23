"""Question-grounded recovery for periodic narrative and investment requests.

The provider is allowed to choose a semantic shape, but it is not allowed to
turn a report name into a topic, lose an explicit period, or promote a broad
investment narrative into the stricter project-table contract.  This module
repairs only a closed grammar whose issuer, period, topic and requested fields
are literal spans of the question.  Canonical document and section authority
remain the responsibility of the ordinary Stage1 resolution backends.
"""
from __future__ import annotations

import re
from typing import Any

from agent.semantic_intent_v1 import SemanticIntent
from agent.stage1_v1_resolver import TerminalAuthority


_POSSESSIVE_ISSUER = re.compile(r"^\s*(?P<issuer>[^?？。]{1,100}?)의\s+")
_ANNUAL_REPORT = re.compile(r"(?<![0-9])20[0-9]{2}년\s*사업보고서")
_ANNUAL_YEAR = re.compile(r"(?<![0-9])20[0-9]{2}년")
_QUARTER = re.compile(
    r"(?<![0-9])(?:20[0-9]{2}|[0-9]{2})년\s*[13]\s*(?:분기|Q)(?:보고서)?",
    flags=re.IGNORECASE,
)
_STRICT_INVESTMENT = re.compile(r"(?:설비\s*)?투자\s*계획|설비투자계획")
_GENERAL_INVESTMENT_OVERRIDE = re.compile(
    r"투자\s*계획\s*표가\s*아니어도|투자계획\s*표가\s*아니어도")
_GENERAL_INVESTMENT = re.compile(r"(?:최근\s*)?투자\s*현황")
_INVESTMENT_AGGREGATE = re.compile(
    r"전부\s*합산|모두\s*합산|합산해|합계|총액")
_INVESTMENT_ROW_COUNT = re.compile(r"(?<![0-9])(?P<count>[1-9][0-9]{0,3})\s*건")
_RATIO_INTENT = re.compile(r"몇\s*퍼센트|비율|%")
_REVENUE_RATIO_OPERAND = re.compile(
    r"(?<![0-9])20[0-9]{2}년\s*(?:연결|별도|개별)\s*매출\s*액")
_INCOMPARABLE_INVESTMENT_CASHFLOW = re.compile(
    r"투자\s*계획.{0,120}(?:연결|별도|개별)?\s*현금\s*흐름표"
    r".{0,80}유형\s*자산\s*취득(?:액|\s*현금\s*유출액)?"
    r".{0,80}(?:차이|차감|빼)")

_TOPIC_PATTERNS = (
    re.compile(r"(?:주요\s*)?제품(?:\s*[·ㆍ]\s*서비스|\s*(?:및|과|와)\s*서비스)?"),
    re.compile(r"(?:주요\s*)?사업(?!\s*(?:보고서|자\s*등록번호))(?:\s*부문|\s*내용)?"),
    re.compile(r"매출\s*구성"),
    re.compile(r"수익\s*구조"),
    re.compile(r"주요\s*자회사\s*구성"),
    re.compile(r"투자\s*부동산(?:\s*현황)?"),
)
_INVESTMENT_FIELDS = (
    re.compile(r"투자\s*대상|대상"),
    re.compile(r"투자\s*목적|목적"),
    re.compile(r"계획\s*금액|총\s*소요자금|금액"),
    re.compile(r"투자\s*기간|기간"),
)


def _literal_issuer(
        question: str, company_surface_regrounder: Any,
        ) -> str | None:
    match = _POSSESSIVE_ISSUER.search(question)
    if match is None or not callable(company_surface_regrounder):
        return None
    proposed = match.group("issuer").strip()
    grounded = company_surface_regrounder(proposed, question)
    if (not isinstance(grounded, str) or not grounded.strip()
            or grounded not in question):
        return None
    return grounded


def _literal_topics(question: str) -> tuple[str, ...]:
    matches: list[tuple[int, int, str]] = []
    for pattern in _TOPIC_PATTERNS:
        matches.extend(
            (match.start(), match.end(), match.group(0).strip())
            for match in pattern.finditer(question))
    matches.sort(key=lambda row: (row[0], -(row[1] - row[0])))
    selected: list[tuple[int, int, str]] = []
    for start, end, surface in matches:
        if any(start < prior_end and end > prior_start
               for prior_start, prior_end, _ in selected):
            continue
        selected.append((start, end, surface))
    selected.sort(key=lambda row: row[0])
    return tuple(dict.fromkeys(surface for _, _, surface in selected))


def _qualified_business_topic(question: str) -> str | None:
    """Keep a literal qualified business phrase as one retrieval axis."""

    match = re.search(
        r"(?P<topic>[A-Za-z가-힣0-9]+(?:\s*[·ㆍ/]\s*[A-Za-z가-힣0-9]+)+"
        r"\s*관련\s*사업\s*현황)",
        question,
    )
    return match.group("topic").strip() if match is not None else None


def explicit_periodic_narrative_fallback_intent(
        question: str, *, company_surface_regrounder: Any,
        ) -> SemanticIntent | None:
    """Build one closed periodic narrative/investment intent from literals.

    This deliberately accepts only one possessive issuer.  Multi-company
    comparisons remain owned by the narrative-matrix parser, where every
    company and cell is independently canonicalized.
    """

    if not isinstance(question, str) or not question.strip():
        return None
    from agent.stage1_v1_narrative_grounding_fallback import (
        explicit_rnd_classification_intent, explicit_monthly_capacity_intent,
        explicit_observed_amount_history_intent)
    history = explicit_observed_amount_history_intent(
        question, company_surface_regrounder=company_surface_regrounder)
    if history is not None:
        return history
    capacity = explicit_monthly_capacity_intent(
        question, company_surface_regrounder=company_surface_regrounder)
    if capacity is not None:
        return capacity
    classified = explicit_rnd_classification_intent(
        question, company_surface_regrounder=company_surface_regrounder)
    if classified is not None:
        return classified
    issuer = _literal_issuer(question, company_surface_regrounder)
    if issuer is None:
        return None

    strict_investment = (
        _STRICT_INVESTMENT.search(question) is not None
        and _GENERAL_INVESTMENT_OVERRIDE.search(question) is None
    )
    general_investment = (
        not strict_investment
        and _GENERAL_INVESTMENT.search(question) is not None
    )
    periodic = (
        _ANNUAL_REPORT.search(question)
        or _QUARTER.search(question)
        or _ANNUAL_YEAR.search(question)
    )
    latest = re.search(r"(?:최신|최근)\s*투자", question)
    if periodic is None and latest is None:
        return None

    if strict_investment:
        target_match = _STRICT_INVESTMENT.search(question)
        assert target_match is not None
        target = target_match.group(0).strip()
        fields: list[str] = []
        for pattern in _INVESTMENT_FIELDS:
            matches = [row for row in pattern.finditer(question)
                       if row.start() >= target_match.end()]
            if matches:
                fields.append(matches[0].group(0).strip())
        if not fields:
            # The strict four-role contract is proved later from source table
            # headers.  Retain one literal surface here instead of inventing
            # role names absent from the question.
            fields = [target]
        output_shape = "record_list"
    elif general_investment:
        target_match = _GENERAL_INVESTMENT.search(question)
        assert target_match is not None
        target = re.sub(r"^최근\s*", "", target_match.group(0)).strip()
        if target not in question:
            return None
        fields = [target]
        output_shape = "narrative"
    else:
        qualified = _qualified_business_topic(question)
        topics = (qualified,) if qualified is not None else _literal_topics(question)
        if not topics:
            return None
        target = topics[0]
        fields = list(topics)
        if re.fullmatch(r"투자\s*부동산\s*현황", target):
            heading = re.sub(r"\s*현황$", "", target).strip()
            if heading in question:
                fields = [heading]
        # Several topics from one exact filing are a grounded summary, not a
        # company/period comparison.  Keeping this as one ordinary periodic
        # narrative lets the existing execution-only topic fanout separate
        # the literal axes without weakening the public matrix schema.
        output_shape = "narrative"

    document_expression = periodic.group(0).strip() if periodic is not None else None
    periods = [document_expression] if document_expression is not None else []
    document_group = None

    return SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [{
            "entity_id": "entity-1", "kind_hint": "company",
            "surface": issuer,
        }],
        "answer_items": [{
            "item_id": "item-1",
            "target": {
                "kind": "topic", "surface": target,
                "entity_refs": ["entity-1"], "qualifier_surfaces": [],
            },
            "operation": "retrieve",
            "scope": {
                "target_period_expressions": periods,
                "as_of_expression": None,
                "document_group_expression": document_group,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": output_shape,
                "projection_mode": "named_fields",
                "field_surfaces": fields,
                "presentation": "auto",
            },
        }],
        "answer_groups": [], "premises": [],
        "unresolved_mentions": [], "presentation": "auto",
    }, strict=True)


class QuestionGroundedNarrativeInvestmentRegrounder:
    """Replace a provider-flattened single-issuer request with closed shape."""

    def __init__(self, company_surface_regrounder: Any) -> None:
        if not callable(company_surface_regrounder):
            raise TypeError("narrative/investment company regrounder가 필요합니다")
        self.company_surface_regrounder = company_surface_regrounder

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        from agent.disclosed_metric_topics import match_disclosed_metric

        # A grounded notes metric is a document field, not an invitation to
        # replace its qualified business name with a business overview.
        if any(
                item.target.kind == "topic"
                and item.scope.document_group_expression is not None
                and (matched := match_disclosed_metric(item.target.surface)) is not None
                and matched[0].term == "우발부채"
                for item in intent.answer_items):
            return intent
        recovered = explicit_periodic_narrative_fallback_intent(
            question,
            company_surface_regrounder=self.company_surface_regrounder,
        )
        return recovered if recovered is not None else intent


def investment_aggregation_request_from_question(
        question: str,
        ) -> dict[str, object] | None:
    """Parse a closed row-sum request from literal question spans."""

    if (not isinstance(question, str)
            or _STRICT_INVESTMENT.search(question) is None):
        return None
    aggregate = _INVESTMENT_AGGREGATE.search(question)
    if aggregate is None:
        return None
    row_count = _INVESTMENT_ROW_COUNT.search(question)
    denominator = _REVENUE_RATIO_OPERAND.search(question)
    if _RATIO_INTENT.search(question) is not None and denominator is None:
        return None
    payload: dict[str, object] = {
        "operation": "sum_amount",
        "amount_role": "금액",
        "aggregate_surface": aggregate.group(0).strip(),
        "expected_row_count": (
            int(row_count.group("count")) if row_count is not None else None),
    }
    if denominator is not None:
        surface = denominator.group(0).strip()
        match = re.fullmatch(
            r"(?P<year>20[0-9]{2})년\s*(?P<scope>연결|별도|개별)\s*"
            r"매출\s*액", surface)
        assert match is not None
        payload["revenue_operand"] = {
            "year": int(match.group("year")),
            "scope": "CFS" if match.group("scope") == "연결" else "SFS",
            "concept": "revenue",
            "surface": surface,
        }
    return payload


class UnsupportedInvestmentOperatorBackend:
    """Fail closed when a requested ratio has no supported exact operand.

    A sum is executable only after the ordinary periodic resolver proves one
    exact four-role investment table.  A ratio additionally requires an
    explicit annual revenue year and 연결/별도 surface.  Other denominator
    shapes remain unsupported instead of silently degrading to a sum-only
    answer.  Malformed, partial, mixed-unit, or row-count-mismatched tables
    still fail closed in ``NarrativeTool``.
    """

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> TerminalAuthority | None:
        del question_id
        if (not isinstance(question, str)
                or _STRICT_INVESTMENT.search(question) is None
                or _INVESTMENT_AGGREGATE.search(question) is None
                or _RATIO_INTENT.search(question) is None
                or _REVENUE_RATIO_OPERAND.search(question) is not None):
            return None
        return TerminalAuthority(reasons=[{
            "code": "unsupported_semantic_target", "scope": "question",
            "item_ids": [item.item_id for item in source_intent.answer_items],
            "diagnostic_code": "unsupported_operator",
        }])


class IncomparableInvestmentCashflowBackend:
    """Refuse subtraction across planned rows and actual cash-flow total.

    A project-level investment-plan table and a statement-level PPE cash-flow
    line have different semantics and aggregation scopes.  Their individual
    evidence can be shown, but coercing them into one arithmetic operand is
    never safe merely because both are monetary amounts.
    """

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> TerminalAuthority | None:
        del question_id
        compact = re.sub(r"\s+", " ", question)
        if _INCOMPARABLE_INVESTMENT_CASHFLOW.search(compact) is None:
            return None
        return TerminalAuthority(reasons=[{
            "code": "unsupported_semantic_target", "scope": "question",
            "item_ids": [item.item_id for item in source_intent.answer_items],
            "diagnostic_code": "incomparable_aggregation_scope",
        }])


__all__ = [
    "IncomparableInvestmentCashflowBackend",
    "QuestionGroundedNarrativeInvestmentRegrounder",
    "UnsupportedInvestmentOperatorBackend",
    "explicit_periodic_narrative_fallback_intent",
    "investment_aggregation_request_from_question",
]
