"""Literal same-year Q4 versus Q3 comparison, with compiler-owned Q4 math."""
import re

from agent.semantic_intent_v1 import SemanticIntent


def _request(question):
    if not isinstance(question, str):
        return None
    match = re.match(r"^\s*(?P<company>[^?\n]{1,80}?)의\s+(?P<period>(?P<year>20\d{2})년\s*4분기)\s*단독\s*(?P<scope>연결|별도)\s*(?P<metric>[가-힣]+)", question)
    if (match is None or not all(word in question for word in (
            "연간 누적액", "9개월 누적액", "빼", "3분기 단독", "몇 퍼센트", "두 분기 금액"))
            or re.search(r"가정|예측|전망|추정|주가|매수|매도", question)):
        return None
    if not re.fullmatch(
            r"\s*연간\s*누적액에서\s*9개월\s*누적액을\s*빼서\s*구하고,?\s*"
            r"3분기\s*단독보다\s*몇\s*퍼센트\s*변했는지\s*비교에\s*쓴\s*"
            r"두\s*분기\s*금액과\s*근거를\s*함께\s*알려줘[.!?]*\s*",
            question[match.end():]):
        return None
    from agent.planning import resolve_metric_concept
    from agent.stage1_v1_financial_backend import concept_axes
    metric = match.group("metric")
    # Korean object particles belong to the request, not the account name.
    metric = re.sub(r"[을를]$", "", metric)
    concept = resolve_metric_concept(metric)
    if concept is None or concept_axes(concept).aggregation != "additive_duration":
        return None
    return match, metric


class QuestionGroundedQuarterComparisonRegrounder:
    def __call__(self, question, intent):
        request = _request(question)
        if request is None or intent.premises or intent.unresolved_mentions:
            return intent
        match, metric = request
        companies = [e for e in intent.entities if e.kind_hint == "company"
                     and e.surface == match.group("company")]
        if len(companies) != 1:
            return intent
        company = companies[0]
        return SemanticIntent.model_validate({
            "schema_version": intent.schema_version,
            "entities": [company.model_dump(mode="python")],
            "answer_items": [{"item_id": "item-1", "operation": "compare",
                "target": {"kind": "metric", "surface": metric,
                           "entity_refs": [company.entity_id], "qualifier_surfaces": []},
                "scope": {"target_period_expressions": [match.group("period"), "3분기 단독"],
                          "as_of_expression": None, "document_group_expression": None,
                          "scope_qualifier_expressions": [match.group("scope")]},
                "selection": None,
                "output": {"shape": "comparison", "projection_mode": "named_fields",
                           "field_surfaces": ["몇 퍼센트"], "presentation": "auto"}}],
            "answer_groups": [], "premises": [], "unresolved_mentions": [], "presentation": "auto",
        }, strict=True)


def quarter_comparison_coordinates(question, item, **kwargs):
    request = _request(question)
    if request is None or item.operation != "compare":
        return ()
    match, metric = request
    if (item.target.surface != metric or item.output.field_surfaces != ["몇 퍼센트"]
            or item.scope.target_period_expressions != [match.group("period"), "3분기 단독"]):
        return ()
    from agent.stage1_v1_financial_backend import financial_coordinates
    rows = []
    for quarter in (4, 3):
        coordinates = financial_coordinates(
            item, question=question, period_override=f"{match.group('year')}년 {quarter}분기", **kwargs)
        if len(coordinates) != 1:
            return ()
        rows.extend(coordinates)
    return tuple(rows)
