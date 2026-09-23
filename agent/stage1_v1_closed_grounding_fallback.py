"""Closed, question-grounded recovery after provider grounding rejection.

This module does not parse answer values or canonical identifiers.  It only
recognises a few complete request topologies whose literal axes are all in the
question, then returns a minimal intent for the ordinary canonical runtime to
reground.  It is used only after a strict provider payload was accepted by the
wire schema but rejected by the semantic grounding boundary.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from agent.semantic_intent_v1 import SemanticIntent
from agent.stage1_v1_exact_event_regrounder import (
    exact_dated_event_named_fields,
)


CompanySurfaceRegrounder = Callable[[str, str], str | None]

_FULL_DATE = re.compile(
    r"(?<![0-9])20[0-9]{2}\s*년\s*"
    r"(?:1[0-2]|0?[1-9])\s*월\s*"
    r"(?:3[01]|[12][0-9]|0?[1-9])\s*일")
_YEAR = re.compile(r"(?<![0-9])20[0-9]{2}\s*년")
_MONTH_EVENT_BEFORE_AFTER = re.compile(
    r"(?P<period>20[0-9]{2}\s*년\s*(?:1[0-2]|0?[1-9])\s*월)\s*"
    r"(?P<family>[^?。\n]{2,80}?계약\s*해지)\s*"
    r"전후\s*계약\s*금액")
_COMPANY_PREFIX = re.compile(
    r"(?:^|[\s,])(?P<surface>[A-Za-z0-9&.+가-힣][A-Za-z0-9&.+가-힣 ()（）]{1,80}?)"
    r"(?:이|가|의|은|는)(?=\s|20[0-9]{2})")
_CORRECTION_HISTORY = re.compile(
    r"정정.{0,80}(?:이력|흐름|변경|바뀌|전후|접수번호별)|"
    r"(?:이력|흐름|변경|바뀌|전후).{0,80}정정")
_REPORT_VERSION = re.compile(
    r"버전|최신본|마지막\s*버전|변경\s*이력|"
    r"원본과.{0,40}정정본")
_INVESTMENT_ADVICE = re.compile(
    r"어느\s*(?:주식|종목)(?:을|를)?\s*사야|"
    r"(?:주식|종목).{0,16}추천|추천.{0,16}(?:주식|종목)|"
    r"매수\s*(?:의견|추천)|투자\s*(?:조언|추천)")
_FINANCIAL_TOKEN = re.compile(r"[A-Za-z0-9가-힣&+]+")
_PARTICLE_ENDINGS = frozenset("은는이가을를의에도와과만")
_OPEN_FINANCIAL_SEMANTICS = re.compile(
    r"만약|가정|예측|예상|전망|목표\s*주가|투자\s*(?:의견|조언)")
_MAXIMUM_WINNER = re.compile(
    r"더\s*(?:큰|크|많(?:은)?|높(?:은)?)|"
    r"(?:큰|높은|많은)\s*(?:기업|회사|곳)|"
    r"(?:누가\s*)?(?:컸지|많았지|높았지)")
_MINIMUM_WINNER = re.compile(
    r"더\s*(?:작은|작|적(?:은)?|낮(?:은)?)|"
    r"(?:작은|낮은|적은)\s*(?:기업|회사|곳)|"
    r"(?:누가\s*)?(?:작았지|적었지|낮았지)")


def _company_surface(
        question: str, regrounder: CompanySurfaceRegrounder | None,
        ) -> str | None:
    if not callable(regrounder):
        return None
    grounded = []
    for match in _COMPANY_PREFIX.finditer(question):
        surface = regrounder(match.group("surface"), question)
        if surface:
            grounded.append(surface)
    unique = list(dict.fromkeys(grounded))
    return unique[0] if len(unique) == 1 else None


def _company_surfaces(
        question: str, regrounder: CompanySurfaceRegrounder | None,
        ) -> list[str]:
    """Return literal, uniquely regrounded company spans in question order.

    The fallback has no canonical registry of its own. It probes bounded
    contiguous question tokens through the injected canonical regrounder and
    accepts only literal surfaces returned by that authority. This lets a
    closed two-company comparison recover after provider grounding failure
    without embedding company aliases in this module.
    """

    if not callable(regrounder):
        return []
    tokens = [match.group(0) for match in _FINANCIAL_TOKEN.finditer(question)]
    grounded: list[str] = []
    for start in range(len(tokens)):
        for end in range(start + 1, min(len(tokens), start + 4) + 1):
            prefix = tokens[start:end - 1]
            forms = [tokens[end - 1]]
            while forms[-1] and forms[-1][-1] in _PARTICLE_ENDINGS:
                forms.append(forms[-1][:-1])
            for tail in forms:
                if len(tail) < 2:
                    continue
                surface = " ".join((*prefix, tail))
                try:
                    value = regrounder(surface, question)
                except Exception:  # noqa: BLE001 - canonical miss is not authority
                    value = None
                if isinstance(value, str) and value and value in question:
                    grounded.append(value)
    return sorted(set(grounded), key=question.find)


def _closed_shape(question: str) -> bool:
    exact_dates = _FULL_DATE.findall(question)
    event_status = bool(
        exact_dates
        and re.search(r"계약", question)
        and re.search(r"공시|제출", question)
        and re.search(r"상태|유효|해지|종료", question)
        and re.search(r"각각|코퍼스\s*기준일|현재|최신|지금", question))
    document_version = bool(
        exact_dates
        and re.search(r"사업보고서|분기보고서|반기보고서", question)
        and _REPORT_VERSION.search(question))
    correction_history = bool(
        exact_dates
        and re.search(r"계약", question)
        and re.search(r"공시|제출", question)
        and _CORRECTION_HISTORY.search(question))
    explicit_financial_alternative = bool(
        len(_YEAR.findall(question)) == 1
        and re.search(r"판매\s*량|몇\s*대\s*팔", question)
        and re.search(r"없으면|안\s*되면|대신", question)
        and re.search(r"매출\s*액", question))
    closed_financial = _closed_financial_request(question)
    closed_financial_change = _closed_financial_change_axes(question) is not None
    return (event_status or document_version or correction_history
            or exact_dated_event_named_fields(question) is not None
            or _MONTH_EVENT_BEFORE_AFTER.search(question) is not None
            or explicit_financial_alternative
            or closed_financial
            or closed_financial_change
            or _INVESTMENT_ADVICE.search(question) is not None)


def _financial_metric_surface(question: str) -> str | None:
    """Return one longest literal financial concept, or fail on competitors.

    The dictionary/suffix scan below only reaches ``AUTO`` concepts —
    ``resolve_metric_concept`` deliberately leaves ``CONTEXT`` colloquials
    (e.g. ``설비투자`` → ``capex_ppe`` only when a cue like ``규모`` is also in
    the question) to `agent.concept_alias.resolve_from_question`, which reads
    the whole question rather than one candidate span.  When the span scan
    finds no dictionary concept at all, fall back to that question-level
    resolver once.  A genuine multi-concept collision from the dictionary
    scan is left exactly as before — competitors still fail closed.
    """

    from agent.planning import CONCEPT_QUESTION_PATTERNS, resolve_metric_concept

    tokens = [match.group(0) for match in _FINANCIAL_TOKEN.finditer(question)]
    matches: list[tuple[int, str, object]] = []
    for start in range(len(tokens)):
        for end in range(start + 1, min(len(tokens), start + 5) + 1):
            prefix = tokens[start:end - 1]
            forms = [tokens[end - 1]]
            while forms[-1] and forms[-1][-1] in _PARTICLE_ENDINGS:
                forms.append(forms[-1][:-1])
            for tail in forms:
                if len(tail) < 2:
                    continue
                surface = " ".join((*prefix, tail))
                concept = resolve_metric_concept(surface)
                if concept is not None:
                    matches.append((len(re.sub(r"\s+", "", surface)), surface, concept))
    concepts = {concept for _length, _surface, concept in matches}
    if len(concepts) == 1:
        longest_length = max(length for length, _surface, _concept in matches)
        longest = [row for row in matches if row[0] == longest_length]
        return longest[0][1]
    if concepts:
        return None
    from agent.concept_alias import resolve_from_question

    colloquial = resolve_from_question(question, CONCEPT_QUESTION_PATTERNS)
    if colloquial.status == "resolved" and colloquial.surface:
        return colloquial.surface
    return None


def _closed_financial_request(question: str) -> bool:
    """Whether the question carries one literal annual metric coordinate.

    This only gates a generic source intent after the provider has exhausted
    its one approved repair.  The canonical financial regrounder remains
    responsible for the exact company, concept, scope, and fact availability.
    """

    return (
        len(dict.fromkeys(_YEAR.findall(question))) == 1
        and _financial_metric_surface(question) is not None
    )


def _closed_financial_change_axes(
        question: str,
        ) -> tuple[str, list[str], str, str] | None:
    """Prove one explicit two-year amount-and-rate comparison from literals.

    This is the exact semantic topology already accepted by the financial
    compiler.  It is reconstructed only after the provider wire has passed the
    strict schema and then failed grounding.  No value, company identifier, or
    fixture label is consulted here.
    """

    years = list(dict.fromkeys(match.group(0) for match in _YEAR.finditer(question)))
    metric = _financial_metric_surface(question)
    amount = re.search(r"얼마나|증감\s*액|변동\s*액|변화\s*액|(?:절대\s*)?차이", question)
    rate = re.search(
        r"몇\s*퍼센트(?:\s*(?:증가|감소))?\s*"
        r"(?:변했는가|변했나|변했어|변했는지|변화했는가|변화했나)|"
        r"증감\s*률|변동\s*률|변화\s*율",
        question,
    )
    scopes = int("연결" in question) + int(
        "별도" in question or "개별" in question)
    if (len(years) != 2 or metric is None or amount is None or rate is None
            or scopes > 1 or _OPEN_FINANCIAL_SEMANTICS.search(question)):
        return None
    return metric, years, amount.group(0), rate.group(0)


def _closed_financial_winner_gap_axes(
        question: str,
        ) -> tuple[str, str, str, str, str | None] | None:
    """Prove one annual two-company directed winner, with an optional gap.

    A explicit gap request ("차이") is no longer required to close this
    topology: "A와 B 중 ... 더 큰 기업은 어디인가" is itself a closed
    two-company request once metric, year, and exactly one directed winner
    cue are literal and unique in the question — the compiler already
    accepts a single ``argmax``-only comparison field.  When a gap cue is
    also present it is still returned, so the caller can append the second,
    explicitly requested difference field exactly as before.
    """

    years = list(dict.fromkeys(match.group(0) for match in _YEAR.finditer(question)))
    metric = _financial_metric_surface(question)
    maximum = list(_MAXIMUM_WINNER.finditer(question))
    minimum = list(_MINIMUM_WINNER.finditer(question))
    gap = re.search(r"(?:절대\s*)?차이", question)
    scopes = int("연결" in question) + int(
        "별도" in question or "개별" in question)
    if (len(years) != 1 or metric is None or bool(maximum) == bool(minimum)
            or scopes > 1 or _OPEN_FINANCIAL_SEMANTICS.search(question)):
        return None
    winner, mode = (maximum[0], "maximum") if maximum else (
        minimum[0], "minimum")
    return (
        metric, years[0], winner.group(0), mode,
        gap.group(0) if gap is not None else None,
    )


def closed_compound_financial_intent(
        question: str, *, company_surface_regrounder: CompanySurfaceRegrounder | None,
        ) -> SemanticIntent | None:
    """Three literal financial requests; keep display, dates, and concepts separate."""
    if _OPEN_FINANCIAL_SEMANTICS.search(question):
        return None
    dates = list(_FULL_DATE.finditer(question))
    dated = len(dates) == 1 and question[dates[0].end():].startswith("까지")
    paired_interest = (re.search(r"손익계산서상\s*이자비용", question)
                       and re.search(r"현금흐름표상\s*이자\s*지급액", question))
    rate = re.search(r"전년\s*대비\s*증가율", question)
    rounded_change = bool(rate and re.search(r"반올림", question)
                          and re.search(r"조원\s*단위", question))
    if not (dated or paired_interest or rounded_change):
        return None
    company = _company_surface(question, company_surface_regrounder)
    if company is None:
        companies = _company_surfaces(question, company_surface_regrounder)
        company = companies[0] if len(companies) == 1 else None
    if company is None:
        return None
    period_question = _FULL_DATE.sub("", question) if dated else question
    years = list(dict.fromkeys(m.group(0) for m in _YEAR.finditer(period_question)))
    scope = ["연결"] if "연결" in question else ["별도"] if "별도" in question else []
    if "연결" in question and ("별도" in question or "개별" in question):
        return None
    items = []

    def append(metric, periods, *, operation="retrieve", field=None, as_of=None):
        items.append({
            "item_id": f"item-{len(items) + 1}", "operation": operation,
            "target": {"kind": "metric", "surface": metric,
                       "entity_refs": ["entity-1"], "qualifier_surfaces": []},
            "scope": {"target_period_expressions": periods, "as_of_expression": as_of,
                      "document_group_expression": None, "scope_qualifier_expressions": scope},
            "selection": None,
            "output": {"shape": "scalar", "projection_mode": "named_fields",
                       "field_surfaces": [field or metric], "presentation": "auto"},
        })

    if paired_interest and len(years) == 1 and not dates:
        remainder = re.sub(r"이자비용|이자\s*지급액", "", question)
        if _financial_metric_surface(remainder) is not None:
            return None
        for match in (re.search(r"이자비용", question), re.search(r"이자\s*지급액", question)):
            append(match.group(0), years)
    else:
        metric = _financial_metric_surface(period_question)
        if metric is None or re.search(r"분기|반기|개월", period_question):
            return None
        if rounded_change and len(years) == 2 and not dates:
            # The existing comparison contract retains both original amount
            # operands as cited support, without rounding its arithmetic.
            append(metric, sorted(years, reverse=True), operation="compare", field=rate.group(0))
        elif dated and len(years) == 1:
            if re.search(r"비교|증가율|증감률|차이|전년", period_question):
                return None
            append(metric, years, as_of=dates[0].group(0))
        else:
            return None
    return SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [{"entity_id": "entity-1", "kind_hint": "company", "surface": company}],
        "answer_items": items, "answer_groups": [], "premises": [],
        "unresolved_mentions": [], "presentation": "auto",
    }, strict=True)


def explicit_closed_grounding_fallback_intent(
        question: str, *,
        company_surface_regrounder: CompanySurfaceRegrounder | None,
        ) -> SemanticIntent | None:
    """Return a minimal literal intent for three closed recovery families."""

    if not isinstance(question, str):
        return None
    compound = closed_compound_financial_intent(
        question, company_surface_regrounder=company_surface_regrounder)
    if compound is not None:
        return compound
    if not _closed_shape(question):
        return None
    # Policy authority consumes the original question.  A minimal literal
    # source intent is enough to reach it even when an advice question names
    # multiple issuers, where selecting one company would itself be unsafe.
    advice = _INVESTMENT_ADVICE.search(question)
    if advice is not None:
        stock = re.search(r"주식|종목|투자", question)
        if stock is None:
            return None
        return SemanticIntent.model_validate({
            "schema_version": "stage1-semantic-intent/1.1",
            "entities": [],
            "answer_items": [{
                "item_id": "item-1", "operation": "retrieve",
                "target": {
                    "kind": "attribute", "surface": stock.group(0),
                    "entity_refs": [], "qualifier_surfaces": [],
                },
                "scope": {
                    "target_period_expressions": [],
                    "as_of_expression": None,
                    "document_group_expression": None,
                    "scope_qualifier_expressions": [],
                },
                "selection": None,
                "output": {
                    "shape": "verdict", "projection_mode": "named_fields",
                    "field_surfaces": [advice.group(0)],
                    "presentation": "auto",
                },
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }, strict=True)

    winner_gap = _closed_financial_winner_gap_axes(question)
    companies = _company_surfaces(question, company_surface_regrounder)
    if winner_gap is not None and len(companies) == 2:
        metric, year, winner, mode, gap = winner_gap
        scope = (
            ["연결"] if "연결" in question else
            ["별도" if "별도" in question else "개별"]
            if ("별도" in question or "개별" in question) else []
        )
        return SemanticIntent.model_validate({
            "schema_version": "stage1-semantic-intent/1.1",
            "entities": [
                {
                    "entity_id": f"entity-{index}",
                    "kind_hint": "company", "surface": surface,
                }
                for index, surface in enumerate(companies, start=1)
            ],
            "answer_items": [{
                "item_id": "item-1", "operation": "compare",
                "target": {
                    "kind": "metric", "surface": metric,
                    "entity_refs": ["entity-1", "entity-2"],
                    "qualifier_surfaces": [],
                },
                "scope": {
                    "target_period_expressions": [year],
                    "as_of_expression": None,
                    "document_group_expression": None,
                    "scope_qualifier_expressions": scope,
                },
                "selection": {
                    "mode": mode, "criterion_surface": winner,
                    "k": None,
                },
                "output": {
                    "shape": "comparison", "projection_mode": "named_fields",
                    "field_surfaces": (
                        [winner, gap] if gap is not None else [winner]),
                    "presentation": "auto",
                },
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }, strict=True)

    company = _company_surface(question, company_surface_regrounder)
    if company is None:
        return None
    before_after = _MONTH_EVENT_BEFORE_AFTER.search(question)
    if before_after is not None:
        period = before_after.group("period").strip()
        family = before_after.group("family").strip()
        # The two output roles are a lossless semantic expansion of the
        # literal compound ``해지 전후 계약금액``.  No date, receipt or
        # value is chosen here; the ordinary selected-event backend must prove
        # that issuer + month + form family identify one canonical filing.
        return SemanticIntent.model_validate({
            "schema_version": "stage1-semantic-intent/1.1",
            "entities": [{
                "entity_id": "entity-1", "kind_hint": "company",
                "surface": company,
            }],
            "answer_items": [{
                "item_id": "item-1", "operation": "retrieve",
                "target": {
                    "kind": "event", "surface": family,
                    "entity_refs": ["entity-1"],
                    "qualifier_surfaces": [period],
                },
                "scope": {
                    "target_period_expressions": [],
                    "as_of_expression": None,
                    "document_group_expression": None,
                    "scope_qualifier_expressions": [],
                },
                "selection": None,
                "output": {
                    "shape": "record", "projection_mode": "named_fields",
                    "field_surfaces": [
                        "해지 전 계약금액", "해지 후 계약금액"],
                    "presentation": "auto",
                },
            }],
            "answer_groups": [], "premises": [],
            "unresolved_mentions": [], "presentation": "auto",
        }, strict=True)
    exact_event = exact_dated_event_named_fields(question)
    if exact_event is not None:
        filing_date, family, fields = exact_event
        # This path is reached only after a schema-valid provider result failed
        # semantic grounding.  Rebuild the closed public coordinate, then let
        # SelectedEventResolutionBackend prove that issuer + date + family
        # identify exactly one canonical filing.  No receipt or answer value
        # is chosen here.
        return SemanticIntent.model_validate({
            "schema_version": "stage1-semantic-intent/1.1",
            "entities": [{
                "entity_id": "entity-1", "kind_hint": "company",
                "surface": company,
            }],
            "answer_items": [{
                "item_id": "item-1", "operation": "retrieve",
                "target": {
                    "kind": "event", "surface": family,
                    "entity_refs": ["entity-1"],
                    "qualifier_surfaces": [filing_date],
                },
                "scope": {
                    "target_period_expressions": [],
                    "as_of_expression": None,
                    "document_group_expression": None,
                    "scope_qualifier_expressions": [],
                },
                "selection": None,
                "output": {
                    "shape": "record", "projection_mode": "named_fields",
                    "field_surfaces": list(fields), "presentation": "auto",
                },
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }, strict=True)
    change_axes = _closed_financial_change_axes(question)
    if change_axes is not None:
        metric, years, amount, rate = change_axes
        scope = (
            ["연결"] if "연결" in question else
            ["별도" if "별도" in question else "개별"]
            if ("별도" in question or "개별" in question) else []
        )

        def item(
                item_id: str, operation: str, field: str,
                ) -> dict[str, object]:
            return {
                "item_id": item_id, "operation": operation,
                "target": {
                    "kind": "metric", "surface": metric,
                    "entity_refs": ["entity-1"], "qualifier_surfaces": [],
                },
                "scope": {
                    "target_period_expressions": years,
                    "as_of_expression": None,
                    "document_group_expression": None,
                    "scope_qualifier_expressions": scope,
                },
                "selection": None,
                "output": {
                    "shape": "scalar", "projection_mode": "named_fields",
                    "field_surfaces": [field], "presentation": "auto",
                },
            }

        return SemanticIntent.model_validate({
            "schema_version": "stage1-semantic-intent/1.1",
            "entities": [{
                "entity_id": "entity-1", "kind_hint": "company",
                "surface": company,
            }],
            "answer_items": [
                item("item-1", "retrieve", amount),
                item("item-2", "compare", rate),
            ],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }, strict=True)
    if _closed_financial_request(question):
        # Keep the source intentionally coordinate-poor.  The runtime's
        # QuestionGroundedFinancialRegrounder restores only the literal,
        # unique company/year/concept axes above before normal resolution.
        return SemanticIntent.model_validate({
            "schema_version": "stage1-semantic-intent/1.1",
            "entities": [{
                "entity_id": "entity-1", "kind_hint": "company",
                "surface": company,
            }],
            "answer_items": [{
                "item_id": "item-1", "operation": "retrieve",
                "target": {
                    "kind": "metric", "surface": question,
                    "entity_refs": ["entity-1"], "qualifier_surfaces": [],
                },
                "scope": {
                    "target_period_expressions": [],
                    "as_of_expression": None,
                    "document_group_expression": None,
                    "scope_qualifier_expressions": [],
                },
                "selection": None,
                "output": {
                    "shape": "scalar", "projection_mode": "named_fields",
                    "field_surfaces": [question], "presentation": "auto",
                },
            }],
            "answer_groups": [], "premises": [], "unresolved_mentions": [],
            "presentation": "auto",
        }, strict=True)
    return SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [{
            "entity_id": "entity-1", "kind_hint": "company",
            "surface": company,
        }],
        "answer_items": [{
            "item_id": "item-1", "operation": "retrieve",
            "target": {
                "kind": "document", "surface": question,
                "entity_refs": ["entity-1"], "qualifier_surfaces": [],
            },
            "scope": {
                "target_period_expressions": [],
                "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": "narrative", "projection_mode": "whole_target",
                "field_surfaces": [], "presentation": "auto",
            },
        }],
        "answer_groups": [], "premises": [], "unresolved_mentions": [],
        "presentation": "auto",
    }, strict=True)


__all__ = ["explicit_closed_grounding_fallback_intent"]
