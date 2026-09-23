"""Closed recovery of two presentations of one disclosed R&D cost table."""
import re

from agent.semantic_intent_v1 import SemanticIntent


def explicit_monthly_capacity_intent(question, *, company_surface_regrounder):
    """Recover literal report + year columns + monthly capacity, no values."""
    if not isinstance(question, str) or not callable(company_surface_regrounder):
        return None
    match = re.fullmatch(
        r"\s*(?P<company>[^?\n]{1,80}?)의\s+"
        r"(?P<report>20\d{2}년\s*사업보고서)에서\s+"
        r"(?P<years>20\d{2}년\s*말(?:\s*[·ㆍ,]\s*20\d{2}년\s*말){1,5})\s+"
        r"(?P<metric>양극재\s*생산능력)을\s*월\s*기준으로\s*"
        r"알려주되,?\s*같은\s*값이어도\s*연도별로\s*써\s*줘[.!?]*\s*", question)
    if match is None:
        return None
    company = company_surface_regrounder(match.group("company"), question)
    if not isinstance(company, str) or not company or company not in question:
        return None
    return SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [{"entity_id": "entity-1", "kind_hint": "company", "surface": company}],
        "answer_items": [{"item_id": "item-1", "operation": "retrieve",
            "target": {"kind": "topic", "surface": "생산능력", "entity_refs": ["entity-1"],
                       "qualifier_surfaces": []},
            "scope": {"target_period_expressions": [], "as_of_expression": None,
                      "document_group_expression": match.group("report"), "scope_qualifier_expressions": []},
            "selection": None,
            "output": {"shape": "narrative", "projection_mode": "named_fields",
                       "field_surfaces": ["생산능력"], "presentation": "auto"}}],
        "answer_groups": [], "premises": [], "unresolved_mentions": [], "presentation": "auto",
    }, strict=True)


def explicit_observed_amount_history_intent(question, *, company_surface_regrounder):
    """Restore only a numbered contract's first/latest observable comparison."""
    if not isinstance(question, str) or not callable(company_surface_regrounder):
        return None
    match = re.fullmatch(
        r"\s*(?P<company>[^?\n]{1,80}?)의\s+접수번호\s*(?P<receipt>20\d{12})\s+"
        r"계약\s*정정공시와\s*연결된\s*자료에서\s*처음\s*확인되는\s*계약금액과\s*"
        r"20\d{2}년\s*\d{1,2}월\s*\d{1,2}일까지의\s*최신\s*확인값을\s*비교하고,?\s*"
        r"그\s*첫값을\s*20\d{2}년\s*원공시의\s*최초\s*금액이라고\s*불러도\s*되는지\s*설명해\s*줘[.!?]*\s*",
        question)
    if match is None:
        return None
    company = company_surface_regrounder(match.group("company"), question)
    if not isinstance(company, str) or not company or company not in question:
        return None
    return SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [{"entity_id": "entity-1", "kind_hint": "company", "surface": company}],
        "answer_items": [{"item_id": "item-1", "operation": "retrieve",
            "target": {"kind": "document", "surface": "계약", "entity_refs": ["entity-1"],
                       "qualifier_surfaces": []},
            "scope": {"target_period_expressions": [match.group("receipt")], "as_of_expression": None,
                      "document_group_expression": None, "scope_qualifier_expressions": []},
            "selection": None,
            "output": {"shape": "narrative", "projection_mode": "whole_target",
                       "field_surfaces": [], "presentation": "auto"}}],
        "answer_groups": [], "premises": [], "unresolved_mentions": [], "presentation": "auto",
    }, strict=True)


def explicit_rnd_classification_intent(question, *, company_surface_regrounder):
    if not isinstance(question, str) or not callable(company_surface_regrounder):
        return None
    pattern = re.compile(
        r"^\s*(?P<company>[^?\n]{1,80}?)의\s+"
        r"(?P<report>20\d{2}년\s*사업보고서)에서\s+"
        r"비용\s*(?:의\s*)?성격별\s*(?P<metric>연구개발비(?:용)?)\s*합계와\s*"
        r"회계처리별\s*연구개발비(?:용)?\s*합계를\s*각각\s*알려주고\s*"
        r"두\s*합계를\s*더해도\s*되는지\s*(?:쉽게\s*)?설명해\s*줘[.!?]*\s*$")
    match = pattern.fullmatch(question)
    if match is None:
        return None
    company = company_surface_regrounder(match.group("company"), question)
    if not isinstance(company, str) or not company or company not in question:
        return None
    return SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [{"entity_id": "entity-1", "kind_hint": "company", "surface": company}],
        "answer_items": [{
            "item_id": "item-1", "operation": "retrieve",
            "target": {"kind": "topic", "surface": match.group("metric"),
                       "entity_refs": ["entity-1"], "qualifier_surfaces": []},
            "scope": {"target_period_expressions": [], "as_of_expression": None,
                      "document_group_expression": match.group("report"),
                      "scope_qualifier_expressions": []},
            "selection": None,
            "output": {"shape": "narrative", "projection_mode": "named_fields",
                       "field_surfaces": [match.group("metric")], "presentation": "auto"},
        }], "answer_groups": [], "premises": [], "unresolved_mentions": [], "presentation": "auto",
    }, strict=True)
