"""HCX-007 Structured Outputs용 좁고 결정적인 JSON Schema compiler.

Pydantic이 만든 schema를 provider에 그대로 넘기지 않는다. 이 모듈은 로컬
``$ref``를 펼치고 HCX allowlist 밖 keyword를 차단한 뒤 canonical JSON과 digest를
묶는다. 응답은 같은 Pydantic model로 다시 strict 검증한다.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import re
from typing import Any, Callable, Generic, TypeVar, cast

from pydantic import BaseModel, ValidationError

from .hcx_repair import repair_wire_payload


PayloadT = TypeVar("PayloadT", bound=BaseModel)
PayloadRepairer = Callable[[Any], tuple[Any, tuple[str, ...]]]

HCX_SCHEMA_VERSION = "hcx-schema/0.1"
MAX_SCHEMA_DEPTH = 12
MAX_SCHEMA_PROPERTIES = 128
MAX_ARRAY_ITEMS = 50

_ALLOWED_KEYWORDS = frozenset({
    "type", "description", "format", "minimum", "maximum",
    "minItems", "maxItems", "items", "properties", "required",
    "enum", "anyOf",
})
_STRIPPED_ANNOTATIONS = frozenset({
    "title", "default", "examples", "deprecated", "readOnly", "writeOnly",
    "minLength", "maxLength",
})
_FORBIDDEN_PROPERTY_KEYS = frozenset({
    "corpcode", "sourcefileid", "evidenceid", "eventkey", "accountpath",
    "planid", "stepid", "taskid", "outputid", "revision", "sql",
    "toolname", "filepath", "indexname", "citation", "citations",
})
_ALLOWED_STRING_FORMATS = frozenset({
    "date-time", "date", "time", "duration", "email", "hostname",
    "ipv4", "ipv6", "uuid",
})


class HcxSchemaError(ValueError):
    """Provider에 보낼 schema가 지원·보안 계약을 위반한다."""


class HcxPayloadValidationError(ValueError):
    """Provider JSON이 원래 strict Pydantic model을 통과하지 못했다."""

    def __init__(
            self, message: str, *, issue_codes: tuple[str, ...],
            issue_paths: tuple[str, ...] = (),
            payload_shape: "HcxSafePayloadShape | None" = None,
            ) -> None:
        if not issue_codes or tuple(sorted(set(issue_codes))) != issue_codes:
            raise ValueError("payload issue code는 정렬·중복제거되어야 합니다")
        if any(not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code)
               for code in issue_codes):
            raise ValueError("payload issue code 형식이 잘못되었습니다")
        if tuple(sorted(set(issue_paths))) != issue_paths:
            raise ValueError("payload issue path는 정렬·중복제거되어야 합니다")
        if any(not re.fullmatch(
                r"[a-z][a-z0-9_]*(?:\[[0-9]+\])?"
                r"(?:\.[a-z][a-z0-9_]*(?:\[[0-9]+\])?)*",
                path,
                ) for path in issue_paths):
            raise ValueError("payload issue path 형식이 잘못되었습니다")
        self.issue_codes = issue_codes
        self.issue_paths = issue_paths
        if payload_shape is not None and not isinstance(
                payload_shape, HcxSafePayloadShape):
            raise TypeError("payload shape는 HcxSafePayloadShape여야 합니다")
        self.payload_shape = payload_shape
        super().__init__(message)


_VALIDATOR_MESSAGE_CODES = (
    ("task kind와 operation이 일치하지 않습니다", "task_operation_mismatch"),
    ("financial proposal에는 fact가 필요합니다", "financial_fact_missing"),
    ("financial이 아닌 proposal은 fact를 가질 수 없습니다",
     "nonfinancial_fact_forbidden"),
    ("task 안의 result label은 중복될 수 없습니다", "task_result_label_duplicate"),
    ("company mention은 중복될 수 없습니다", "company_mention_duplicate"),
    ("requested slot은 중복될 수 없습니다", "requested_slot_duplicate"),
    ("proposal은 operand 2개가 필요합니다", "derivation_arity_invalid"),
    ("process proposal에는 task만 있고 terminal reason은 없어야 합니다",
     "process_shape_invalid"),
    ("terminal proposal에는 typed reason만 있어야 합니다", "terminal_shape_invalid"),
    ("reason code 형식이 잘못되었습니다", "reason_code_invalid"),
    ("task label은 proposal 전체에서 유일해야 합니다", "task_label_duplicate"),
    ("result label은 proposal 전체에서 유일해야 합니다", "result_label_duplicate"),
    ("derivation이 알 수 없는/후행 label을 참조합니다",
     "derivation_reference_invalid"),
    ("derivation result label이 중복되었습니다", "derivation_result_duplicate"),
    ("premise가 알 수 없는 label을 참조합니다", "premise_reference_invalid"),
    ("financial intent에는 fact가 필요합니다", "financial_fact_missing"),
    ("financial intent 회사는 fact.company_mention으로만 표현해야 합니다",
     "financial_company_source_duplicate"),
    ("financial intent 출력은 fact로만 표현해야 합니다",
     "financial_requested_slot_forbidden"),
    ("financial이 아닌 intent는 fact를 가질 수 없습니다",
     "nonfinancial_fact_forbidden"),
    ("narrative intent에는 retrieval_query가 필요합니다",
     "narrative_query_missing"),
    ("narrative가 아닌 intent는 retrieval_query를 가질 수 없습니다",
     "nonnarrative_query_forbidden"),
    ("intent company mention은 중복될 수 없습니다", "company_mention_duplicate"),
    ("intent requested slot은 중복될 수 없습니다", "requested_slot_duplicate"),
    ("intent는 operand 2개가 필요합니다", "analysis_arity_invalid"),
    ("analysis는 operand 2개가 필요합니다", "analysis_arity_invalid"),
    ("argmax intent는 operand 2개 이상이 필요합니다", "analysis_arity_invalid"),
    ("analysis operand는 중복될 수 없습니다", "analysis_operand_duplicate"),
    ("claim output ref는 중복될 수 없습니다", "claim_reference_duplicate"),
    ("process intent shape가 유효하지 않습니다", "process_shape_invalid"),
    ("terminal intent shape가 유효하지 않습니다", "terminal_shape_invalid"),
    ("intent reason code는 중복될 수 없습니다", "reason_code_duplicate"),
    ("request intent는 premise_to_verify를 가질 수 없습니다",
     "request_premise_forbidden"),
    ("verification intent에는 premise_to_verify가 필요합니다",
     "verification_premise_missing"),
    ("causal premise는 process intent로 실행할 수 없습니다",
     "causal_process_forbidden"),
    ("ref 없는 premise는 호환 task가 정확히 1개여야 합니다",
     "unbound_premise_task_count_invalid"),
    ("downstream slot ref는 선두 prefix여야 합니다", "slot_reference_gap"),
    ("process request에는 requested output이 필요합니다",
     "requested_output_missing"),
    ("requested output ref는 중복될 수 없습니다",
     "requested_output_duplicate"),
    ("financial fact는 중복될 수 없습니다", "financial_fact_duplicate"),
    ("task row는 중복될 수 없습니다", "task_row_duplicate"),
    ("analysis row는 중복될 수 없습니다", "analysis_row_duplicate"),
    ("claim row는 중복될 수 없습니다", "claim_row_duplicate"),
    ("company_mentions 배열에는 빈 문자열을 넣을 수 없습니다",
     "empty_company_mention"),
    ("company_mentions 배열에는 unspecified를 넣을 수 없습니다",
     "company_mention_unspecified"),
    ("company_mentions 배열은 중복될 수 없습니다",
     "company_mention_duplicate"),
    ("target_period_expressions 배열에는 빈 문자열을 넣을 수 없습니다",
     "empty_target_period_expression"),
    ("target_period_expressions 배열에는 unspecified를 넣을 수 없습니다",
     "target_period_unspecified"),
    ("target_period_expressions 배열은 중복될 수 없습니다",
     "target_period_duplicate"),
    ("source_slots 배열에는 빈 문자열을 넣을 수 없습니다",
     "empty_source_slot"),
    ("source_slots 배열에는 unspecified를 넣을 수 없습니다",
     "source_slot_unspecified"),
    ("source_slots 배열은 중복될 수 없습니다", "source_slot_duplicate"),
    ("selector_mentions 배열에는 빈 문자열을 넣을 수 없습니다",
     "empty_selector_mention"),
    ("selector_mentions 배열에는 unspecified를 넣을 수 없습니다",
     "selector_mention_unspecified"),
    ("selector_mentions 배열은 중복될 수 없습니다",
     "selector_mention_duplicate"),
    ("narrative retrieval_query는 unspecified일 수 없습니다",
     "narrative_query_unspecified"),
    ("claim raw_text는 unspecified일 수 없습니다", "claim_text_unspecified"),
    ("claim verification ref는 중복될 수 없습니다",
     "claim_verification_ref_duplicate"),
    ("claim context task ref는 중복될 수 없습니다",
     "claim_context_ref_duplicate"),
    ("같은 task를 verification과 context로 동시에 쓸 수 없습니다",
     "claim_verification_context_overlap"),
    ("noncausal claim에는 verification ref가 필요합니다",
     "claim_verification_ref_missing"),
    ("context task ref는 causal claim에서만 허용됩니다",
     "claim_context_forbidden"),
    ("request intent는 claim을 가질 수 없습니다", "request_claim_forbidden"),
    ("verification intent는 claim만 있고 requested output은 없어야 합니다",
     "verification_shape_invalid"),
    ("mixed intent에는 claim과 requested output이 모두 필요합니다",
     "mixed_shape_invalid"),
    ("process intent의 task/reason shape가 유효하지 않습니다",
     "process_shape_invalid"),
    ("non-process intent에는 reason code가 필요합니다",
     "terminal_reason_missing"),
    ("pressure_resisted는 단독 primary reason이 될 수 없습니다",
     "pressure_reason_without_primary"),
    ("reason code와 proposed disposition이 호환되지 않습니다",
     "reason_disposition_mismatch"),
    ("검증 불가능 causal claim과 typed reason이 일치하지 않습니다",
     "causal_reason_mismatch"),
    ("검증 불가능 causal claim은 unsupported_request여야 합니다",
     "causal_disposition_mismatch"),
    ("직접 ref나 task 전체 ref에 연결되지 않은 fact/source slot이 있습니다",
     "unused_source_slot"),
    ("output/claim에 연결되지 않은 dead task가 있습니다", "dead_task"),
    ("사용되지 않는 analysis가 있습니다", "unused_analysis"),
    ("analysis ref는 정의된 선행/기존 analysis만 가리켜야 합니다",
     "analysis_reference_invalid"),
    ("analysis ref는 이미 정의된 선행 analysis만 가리켜야 합니다",
     "analysis_reference_invalid"),
    ("ref task_index가 범위를 벗어났습니다", "task_reference_invalid"),
    ("task ref가 범위를 벗어났습니다", "task_reference_invalid"),
    ("fact ref는 financial task만 가리킬 수 있습니다",
     "fact_reference_kind_invalid"),
    ("task ref는 financial task를 가리킬 수 없습니다",
     "task_reference_invalid"),
    ("slot ref는 financial task를 가리킬 수 없습니다",
     "slot_reference_kind_invalid"),
    ("v0.4 참조 source slot은 16개 이하여야 합니다",
     "direct_source_slot_limit_exceeded"),
    ("output ref의 task index가 범위를 벗어났습니다", "task_reference_invalid"),
    ("fact ref가 범위를 벗어났습니다", "fact_reference_invalid"),
    ("참조한 slot을 source_slots에 선언하지 않았습니다",
     "slot_reference_undeclared"),
)

_PYDANTIC_TYPE_CODES = {
    "json_invalid": "json_invalid",
    "extra_forbidden": "extra_field_forbidden",
    "missing": "required_field_missing",
    "literal_error": "literal_value_invalid",
    "string_too_short": "string_size_invalid",
    "string_too_long": "string_size_invalid",
    "too_short": "collection_size_invalid",
    "too_long": "collection_size_invalid",
    "list_type": "field_type_invalid",
    "string_type": "field_type_invalid",
    "int_type": "field_type_invalid",
    "float_type": "field_type_invalid",
    "bool_type": "field_type_invalid",
    "model_type": "field_type_invalid",
    "model_attributes_type": "field_type_invalid",
}

_SAFE_DIAGNOSTIC_FIELDS = frozenset({
    "schema_version", "disposition", "tasks", "derivations",
    "premise_claims", "presentation", "reason_codes", "task_label", "kind",
    "operation", "company_mentions", "facts", "as_of_expression",
    "document_group", "event_type_text", "counterparty_text",
    "contract_name_text", "seed_receipt_text", "retrieval_query",
    "selector_mentions",
    "requested_slots", "source_slots", "target_period_expressions",
    "requested_outputs", "verification_refs", "context_task_refs",
    "result_labels", "result_label", "company_mention",
    "concept_mention", "period_expression", "scope", "statement", "view",
    "scope_expression", "statement_expression", "view_expression",
    "operator", "operand_labels", "rounding_rule", "claim_label", "raw_text",
    "verify_labels", "proposed_disposition", "speech_act", "analyses",
    "claims", "output_refs", "operands", "task_index", "fact_index",
    "slot_index", "analysis_index",
    # SemanticIntent v1 wire.  Only key names, closed enums, counts, and
    # generated paths are retained; user surfaces are never serialized.
    "entities", "answer_items", "answer_groups", "premises",
    "unresolved_mentions", "surface", "kind_hint", "target",
    "entity_indexes", "qualifier_surfaces", "document_group_expression",
    "scope_qualifier_expressions", "selection", "mode",
    "criterion_surface", "k", "output", "shape", "projection_mode",
    "field_surfaces", "item_indexes", "applies_to_item_indexes",
    "role_hint",
})


def _safe_field_issue_code(issue: dict[str, Any]) -> str | None:
    issue_type = issue.get("type")
    if issue_type not in {"string_too_short", "string_too_long"}:
        return None
    location = issue.get("loc")
    if not isinstance(location, tuple):
        return None
    fields = [part for part in location if isinstance(part, str)]
    if not fields or fields[-1] not in _SAFE_DIAGNOSTIC_FIELDS:
        return None
    prefix = "empty_string" if issue_type == "string_too_short" else "oversize_string"
    return f"{prefix}_{fields[-1]}"


def _validation_issues(error: ValidationError) -> list[dict[str, Any]]:
    """명백히 다른 task union branch에서 나온 가짜 오류를 제거한다.

    자유문 입력은 보지 않는다. 각 task에서 ``task_mode`` literal 오류가 없는 branch가
    정확히 하나일 때만 그 branch를 선택한다.
    """

    issues = error.errors(
        include_input=False, include_context=False, include_url=False)
    branches: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for issue in issues:
        location = issue.get("loc")
        if (not isinstance(location, tuple) or len(location) < 3
                or location[0] != "tasks" or type(location[1]) is not int
                or not isinstance(location[2], str)
                or "IntentTaskWire" not in location[2]):
            continue
        branches.setdefault((location[1], location[2]), []).append(issue)

    selected: dict[int, str] = {}
    for task_index in {index for index, _ in branches}:
        matching: list[str] = []
        for (candidate_index, branch), rows in branches.items():
            if candidate_index != task_index:
                continue
            wrong_mode = any(
                row.get("type") == "literal_error"
                and isinstance(row.get("loc"), tuple)
                and row["loc"][-1] == "task_mode"
                for row in rows
            )
            if not wrong_mode:
                matching.append(branch)
        if len(matching) == 1:
            selected[task_index] = matching[0]

    filtered: list[dict[str, Any]] = []
    for issue in issues:
        location = issue.get("loc")
        if (isinstance(location, tuple) and len(location) >= 3
                and location[0] == "tasks" and type(location[1]) is int
                and isinstance(location[2], str)
                and "IntentTaskWire" in location[2]
                and location[1] in selected
                and location[2] != selected[location[1]]):
            continue
        filtered.append(issue)
    return filtered


def safe_validation_issue_paths(error: ValidationError) -> tuple[str, ...]:
    """자유문 값을 제외한 allowlisted field/index 경로만 반환한다."""

    paths: set[str] = set()
    for issue in _validation_issues(error):
        location = issue.get("loc")
        if not isinstance(location, tuple):
            continue
        parts: list[str] = []
        for item in location:
            if isinstance(item, str) and item in _SAFE_DIAGNOSTIC_FIELDS:
                parts.append(item)
            elif type(item) is int and parts:
                parts[-1] += f"[{item}]"
        if parts:
            path = ".".join(parts)
            message = issue.get("msg")
            if isinstance(message, str):
                if ("target_period_expressions 배열" in message
                        and not path.endswith("target_period_expressions")):
                    path += ".target_period_expressions"
                elif ("source_slots 배열" in message
                      and not path.endswith("source_slots")):
                    path += ".source_slots"
            paths.add(path)
    return tuple(sorted(paths))[:32]


def safe_validation_issue_codes(error: ValidationError) -> tuple[str, ...]:
    """Pydantic 입력·자유문을 버리고 allowlist 진단 코드만 만든다."""

    codes: set[str] = set()
    for issue in _validation_issues(error):
        field_code = _safe_field_issue_code(issue)
        if field_code is not None:
            codes.add(field_code)
        issue_type = issue.get("type")
        if isinstance(issue_type, str) and issue_type in _PYDANTIC_TYPE_CODES:
            codes.add(_PYDANTIC_TYPE_CODES[issue_type])
            continue
        message = issue.get("msg")
        if isinstance(message, str):
            for fragment, code in _VALIDATOR_MESSAGE_CODES:
                if fragment in message:
                    codes.add(code)
                    break
            else:
                codes.add("model_contract_invalid")
        else:
            codes.add("model_contract_invalid")
    return tuple(sorted(codes or {"model_contract_invalid"}))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


_SAFE_TASK_MODES = frozenset({
    "financial_lookup", "disclosure_lookup", "disclosure_list",
    "event_status", "event_timeline", "event_list",
    "correction_diff", "correction_history",
    "document_find", "document_latest", "document_version_history",
    "narrative_search", "narrative_summarize", "narrative_compare",
})
_SAFE_DISPOSITIONS = frozenset({
    "process", "out_of_scope", "unsupported_request", "policy_refusal",
})
_SAFE_SPEECH_ACTS = frozenset({"request", "verification", "mixed"})
_SAFE_OPERATORS = frozenset({
    "difference", "absolute_difference", "percent_change",
    "discrete_from_cumulative", "argmax",
})
_SAFE_CLAIM_KINDS = frozenset({
    "numeric", "state", "comparison", "existence", "causal",
})
_SAFE_SEMANTIC_TARGET_KINDS = frozenset({
    "metric", "attribute", "event", "document", "topic", "entity",
})
_SAFE_SEMANTIC_OPERATIONS = frozenset({"retrieve", "compare"})
_SAFE_PROJECTION_MODES = frozenset({"named_fields", "whole_target"})
_SAFE_OUTPUT_SHAPES = frozenset({
    "scalar", "record", "record_list", "comparison", "timeline",
    "narrative", "verdict",
})
_SAFE_PRESENTATIONS = frozenset({"auto", "prose", "table", "list"})
_SAFE_ARRAY_DIAGNOSTIC_FIELDS = frozenset({
    "company_mentions", "target_period_expressions", "selector_mentions",
    "source_slots", "requested_outputs", "verification_refs",
    "context_task_refs", "operands", "entities", "answer_items",
    "answer_groups", "premises", "unresolved_mentions", "entity_indexes",
    "qualifier_surfaces", "scope_qualifier_expressions", "field_surfaces",
    "item_indexes", "applies_to_item_indexes",
})


@dataclass(frozen=True, slots=True)
class HcxSafePayloadShape:
    """자유문 없이 provider payload의 구조만 보존하는 진단 snapshot."""

    root_fields: tuple[str, ...]
    proposed_disposition: str
    speech_act: str
    task_count: int
    task_modes: tuple[str, ...]
    task_fields: tuple[tuple[str, ...], ...]
    task_fact_counts: tuple[int, ...]
    task_source_slot_counts: tuple[int, ...]
    task_target_period_counts: tuple[int, ...]
    analysis_count: int
    analysis_operators: tuple[str, ...]
    claim_count: int
    claim_kinds: tuple[str, ...]
    requested_output_count: int
    requested_output_refs: tuple[str, ...]
    reason_count: int
    blank_field_paths: tuple[str, ...]
    unspecified_field_paths: tuple[str, ...]
    duplicate_array_paths: tuple[str, ...]
    semantic_entity_count: int
    semantic_answer_item_count: int
    semantic_target_kinds: tuple[str, ...]
    semantic_operations: tuple[str, ...]
    semantic_projection_modes: tuple[str, ...]
    semantic_output_shapes: tuple[str, ...]
    semantic_field_counts: tuple[int, ...]
    semantic_answer_group_count: int
    semantic_premise_count: int
    semantic_unresolved_count: int
    semantic_presentation: str

    def as_dict(self) -> dict[str, object]:
        return {
            "root_fields": list(self.root_fields),
            "proposed_disposition": self.proposed_disposition,
            "speech_act": self.speech_act,
            "task_count": self.task_count,
            "task_modes": list(self.task_modes),
            "task_fields": [list(fields) for fields in self.task_fields],
            "task_fact_counts": list(self.task_fact_counts),
            "task_source_slot_counts": list(self.task_source_slot_counts),
            "task_target_period_counts": list(self.task_target_period_counts),
            "analysis_count": self.analysis_count,
            "analysis_operators": list(self.analysis_operators),
            "claim_count": self.claim_count,
            "claim_kinds": list(self.claim_kinds),
            "requested_output_count": self.requested_output_count,
            "requested_output_refs": list(self.requested_output_refs),
            "reason_count": self.reason_count,
            "blank_field_paths": list(self.blank_field_paths),
            "unspecified_field_paths": list(self.unspecified_field_paths),
            "duplicate_array_paths": list(self.duplicate_array_paths),
            "semantic_entity_count": self.semantic_entity_count,
            "semantic_answer_item_count": self.semantic_answer_item_count,
            "semantic_target_kinds": list(self.semantic_target_kinds),
            "semantic_operations": list(self.semantic_operations),
            "semantic_projection_modes": list(self.semantic_projection_modes),
            "semantic_output_shapes": list(self.semantic_output_shapes),
            "semantic_field_counts": list(self.semantic_field_counts),
            "semantic_answer_group_count": self.semantic_answer_group_count,
            "semantic_premise_count": self.semantic_premise_count,
            "semantic_unresolved_count": self.semantic_unresolved_count,
            "semantic_presentation": self.semantic_presentation,
        }


def _shape_count(value: object) -> int:
    return len(value) if isinstance(value, list) else -1


def _safe_enum(value: object, allowed: frozenset[str]) -> str:
    return value if isinstance(value, str) and value in allowed \
        else "invalid_or_missing"


def _safe_ref_shape(value: object) -> str:
    if not isinstance(value, dict):
        return "invalid"
    kind = value.get("kind")
    if kind == "analysis" and type(value.get("analysis_index")) is int:
        return f"analysis:{value['analysis_index']}"
    if kind == "task" and type(value.get("task_index")) is int:
        return f"task:{value['task_index']}"
    if (kind in {"fact", "slot"} and type(value.get("task_index")) is int
            and type(value.get(f"{kind}_index")) is int):
        return f"{kind}:{value['task_index']}:{value[f'{kind}_index']}"
    return "invalid"


def safe_payload_shape(payload: str) -> HcxSafePayloadShape | None:
    """invalid JSON의 자유문을 버리고 구조·sentinel·중복 여부만 추출한다."""

    try:
        value = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None

    tasks = value.get("tasks")
    task_rows = tasks if isinstance(tasks, list) else []
    task_modes: list[str] = []
    task_fields: list[tuple[str, ...]] = []
    fact_counts: list[int] = []
    slot_counts: list[int] = []
    period_counts: list[int] = []
    for task in task_rows[:12]:
        if not isinstance(task, dict):
            task_modes.append("invalid_or_missing")
            task_fields.append(())
            fact_counts.append(-1)
            slot_counts.append(-1)
            period_counts.append(-1)
            continue
        task_modes.append(_safe_enum(task.get("task_mode"), _SAFE_TASK_MODES))
        task_fields.append(tuple(sorted(
            key for key in task if key in _SAFE_DIAGNOSTIC_FIELDS)))
        fact_counts.append(_shape_count(task.get("facts")))
        slot_counts.append(_shape_count(task.get("source_slots")))
        period_counts.append(_shape_count(task.get("target_period_expressions")))

    analyses = value.get("analyses")
    analysis_rows = analyses if isinstance(analyses, list) else []
    claims = value.get("claims")
    claim_rows = claims if isinstance(claims, list) else []
    outputs = value.get("requested_outputs")
    output_rows = outputs if isinstance(outputs, list) else []
    reasons = value.get("reason_codes")

    semantic_entities = value.get("entities")
    semantic_items = value.get("answer_items")
    semantic_item_rows = semantic_items if isinstance(semantic_items, list) else []
    semantic_target_kinds: list[str] = []
    semantic_operations: list[str] = []
    semantic_projection_modes: list[str] = []
    semantic_output_shapes: list[str] = []
    semantic_field_counts: list[int] = []
    for item in semantic_item_rows[:50]:
        if not isinstance(item, dict):
            semantic_target_kinds.append("invalid_or_missing")
            semantic_operations.append("invalid_or_missing")
            semantic_projection_modes.append("invalid_or_missing")
            semantic_output_shapes.append("invalid_or_missing")
            semantic_field_counts.append(-1)
            continue
        target = item.get("target")
        output = item.get("output")
        semantic_target_kinds.append(_safe_enum(
            target.get("kind") if isinstance(target, dict) else None,
            _SAFE_SEMANTIC_TARGET_KINDS,
        ))
        semantic_operations.append(_safe_enum(
            item.get("operation"), _SAFE_SEMANTIC_OPERATIONS))
        semantic_projection_modes.append(_safe_enum(
            output.get("projection_mode") if isinstance(output, dict) else None,
            _SAFE_PROJECTION_MODES,
        ))
        semantic_output_shapes.append(_safe_enum(
            output.get("shape") if isinstance(output, dict) else None,
            _SAFE_OUTPUT_SHAPES,
        ))
        semantic_field_counts.append(_shape_count(
            output.get("field_surfaces") if isinstance(output, dict) else None))

    blank_paths: set[str] = set()
    unspecified_paths: set[str] = set()
    duplicate_paths: set[str] = set()

    def scan(node: object, path: str = "", field_name: str | None = None) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key not in _SAFE_DIAGNOSTIC_FIELDS:
                    continue
                child_path = f"{path}.{key}" if path else key
                scan(child, child_path, key)
            return
        if isinstance(node, list):
            if field_name in _SAFE_ARRAY_DIAGNOSTIC_FIELDS:
                encoded = [_canonical_json(item) for item in node]
                if len(encoded) != len(set(encoded)):
                    duplicate_paths.add(path)
            for index, child in enumerate(node[:50]):
                scan(child, f"{path}[{index}]", field_name)
            return
        if isinstance(node, str):
            if node == "":
                blank_paths.add(path)
            elif node == "unspecified":
                unspecified_paths.add(path)

    scan(value)
    return HcxSafePayloadShape(
        root_fields=tuple(sorted(
            key for key in value if key in _SAFE_DIAGNOSTIC_FIELDS)),
        proposed_disposition=_safe_enum(
            value.get("proposed_disposition"), _SAFE_DISPOSITIONS),
        speech_act=_safe_enum(value.get("speech_act"), _SAFE_SPEECH_ACTS),
        task_count=_shape_count(tasks),
        task_modes=tuple(task_modes),
        task_fields=tuple(task_fields),
        task_fact_counts=tuple(fact_counts),
        task_source_slot_counts=tuple(slot_counts),
        task_target_period_counts=tuple(period_counts),
        analysis_count=_shape_count(analyses),
        analysis_operators=tuple(
            _safe_enum(row.get("operator"), _SAFE_OPERATORS)
            if isinstance(row, dict) else "invalid_or_missing"
            for row in analysis_rows[:32]),
        claim_count=_shape_count(claims),
        claim_kinds=tuple(
            _safe_enum(row.get("kind"), _SAFE_CLAIM_KINDS)
            if isinstance(row, dict) else "invalid_or_missing"
            for row in claim_rows[:32]),
        requested_output_count=_shape_count(outputs),
        requested_output_refs=tuple(
            _safe_ref_shape(row) for row in output_rows[:32]),
        reason_count=_shape_count(reasons),
        blank_field_paths=tuple(sorted(blank_paths))[:32],
        unspecified_field_paths=tuple(sorted(unspecified_paths))[:32],
        duplicate_array_paths=tuple(sorted(duplicate_paths))[:32],
        semantic_entity_count=_shape_count(semantic_entities),
        semantic_answer_item_count=_shape_count(semantic_items),
        semantic_target_kinds=tuple(semantic_target_kinds),
        semantic_operations=tuple(semantic_operations),
        semantic_projection_modes=tuple(semantic_projection_modes),
        semantic_output_shapes=tuple(semantic_output_shapes),
        semantic_field_counts=tuple(semantic_field_counts),
        semantic_answer_group_count=_shape_count(value.get("answer_groups")),
        semantic_premise_count=_shape_count(value.get("premises")),
        semantic_unresolved_count=_shape_count(value.get("unresolved_mentions")),
        semantic_presentation=_safe_enum(
            value.get("presentation"), _SAFE_PRESENTATIONS),
    )


def _property_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _resolve_ref(
        ref: str, definitions: dict[str, Any], stack: tuple[str, ...],
        ) -> tuple[str, Any]:
    prefix = "#/$defs/"
    if not isinstance(ref, str) or not ref.startswith(prefix):
        raise HcxSchemaError("외부 또는 비표준 $ref는 허용하지 않습니다")
    name = ref[len(prefix):]
    if not name or "/" in name or name not in definitions:
        raise HcxSchemaError("존재하지 않는 로컬 $ref입니다")
    if name in stack:
        raise HcxSchemaError("순환 $ref는 HCX schema로 내릴 수 없습니다")
    return name, deepcopy(definitions[name])


def _lower_node(
        value: Any, definitions: dict[str, Any], stack: tuple[str, ...], *,
        depth: int,
        ) -> Any:
    if depth > MAX_SCHEMA_DEPTH:
        raise HcxSchemaError(f"HCX schema depth는 {MAX_SCHEMA_DEPTH} 이하여야 합니다")
    if isinstance(value, list):
        return [
            _lower_node(item, definitions, stack, depth=depth + 1)
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    if "$ref" in value:
        if len(value) != 1:
            raise HcxSchemaError("$ref와 sibling keyword를 함께 사용할 수 없습니다")
        name, resolved = _resolve_ref(value["$ref"], definitions, stack)
        return _lower_node(
            resolved, definitions, stack + (name,), depth=depth + 1)

    lowered: dict[str, Any] = {}
    for key, item in value.items():
        if key == "$defs":
            continue
        if key in _STRIPPED_ANNOTATIONS:
            continue
        if key == "additionalProperties":
            if item is not False:
                raise HcxSchemaError("additionalProperties는 false인 strict model만 허용합니다")
            continue
        if key == "const":
            lowered["enum"] = [item]
            continue
        if key not in _ALLOWED_KEYWORDS:
            raise HcxSchemaError(f"HCX가 지원하지 않는 JSON Schema keyword입니다: {key}")
        if key == "properties":
            if not isinstance(item, dict):
                raise HcxSchemaError("properties는 object여야 합니다")
            lowered[key] = {
                name: _lower_node(schema, definitions, stack, depth=depth + 1)
                for name, schema in item.items()
            }
            continue
        lowered[key] = _lower_node(item, definitions, stack, depth=depth + 1)

    if "anyOf" in lowered:
        branches = lowered["anyOf"]
        if not isinstance(branches, list) or not branches:
            raise HcxSchemaError("anyOf에는 하나 이상의 branch가 필요합니다")
        if any(isinstance(branch, dict) and branch.get("type") == "null"
               for branch in branches):
            raise HcxSchemaError(
                "nullable anyOf는 사용하지 말고 명시적인 sentinel enum을 사용하세요")

    if "enum" in lowered:
        values = lowered["enum"]
        if not isinstance(values, list) or not values:
            raise HcxSchemaError("enum에는 하나 이상의 값이 필요합니다")
        if any(value is None or isinstance(value, (dict, list)) for value in values):
            raise HcxSchemaError("enum은 null이 아닌 scalar 값만 허용합니다")
        encoded = [_canonical_json(value) for value in values]
        if len(encoded) != len(set(encoded)):
            raise HcxSchemaError("enum 값은 중복될 수 없습니다")

    if "description" in lowered and not isinstance(lowered["description"], str):
        raise HcxSchemaError("description은 문자열이어야 합니다")
    if "format" in lowered and lowered["format"] not in _ALLOWED_STRING_FORMATS:
        raise HcxSchemaError(f"HCX가 지원하지 않는 string format입니다: {lowered['format']}")

    for bound in ("minimum", "maximum"):
        if bound in lowered and (not isinstance(lowered[bound], (int, float))
                                 or isinstance(lowered[bound], bool)
                                 or not math.isfinite(lowered[bound])):
            raise HcxSchemaError(f"{bound}은 유한한 숫자여야 합니다")
    if ("minimum" in lowered and "maximum" in lowered
            and lowered["minimum"] > lowered["maximum"]):
        raise HcxSchemaError("minimum은 maximum보다 클 수 없습니다")

    node_type = lowered.get("type")
    if node_type == "object":
        properties = lowered.get("properties")
        required = lowered.get("required", [])
        if not isinstance(properties, dict) or not properties:
            raise HcxSchemaError("object schema에는 비어 있지 않은 properties가 필요합니다")
        if not isinstance(required, list) or any(not isinstance(x, str) for x in required):
            raise HcxSchemaError("required는 문자열 배열이어야 합니다")
        if len(required) != len(set(required)) or not set(required).issubset(properties):
            raise HcxSchemaError("required와 properties가 일치하지 않습니다")
        for name in properties:
            if not isinstance(name, str) or not name:
                raise HcxSchemaError("property 이름은 비어 있지 않은 문자열이어야 합니다")
            if _property_key(name) in _FORBIDDEN_PROPERTY_KEYS:
                raise HcxSchemaError(f"모델이 생성할 수 없는 시스템 소유 필드입니다: {name}")
    elif node_type == "array":
        if "items" not in lowered:
            raise HcxSchemaError("array schema에는 items가 필요합니다")
        maximum = lowered.setdefault("maxItems", MAX_ARRAY_ITEMS)
        minimum = lowered.get("minItems", 0)
        if (type(minimum) is not int or type(maximum) is not int
                or minimum < 0 or maximum < minimum or maximum > MAX_ARRAY_ITEMS):
            raise HcxSchemaError(
                f"array items 범위는 0..{MAX_ARRAY_ITEMS} 안에서 유효해야 합니다")
    elif node_type is not None and node_type not in {
            "string", "integer", "number", "boolean", "object", "array"}:
        raise HcxSchemaError(f"지원하지 않는 JSON type입니다: {node_type}")
    return lowered


def _count_properties(value: Any) -> int:
    if isinstance(value, list):
        return sum(_count_properties(item) for item in value)
    if not isinstance(value, dict):
        return 0
    count = len(value.get("properties", {})) if isinstance(value.get("properties"), dict) else 0
    return count + sum(_count_properties(item) for item in value.values())


@dataclass(frozen=True, slots=True)
class CompiledHcxSchema(Generic[PayloadT]):
    """Provider schema와 local validator를 hash-bound한 불변 계약."""

    schema: dict[str, Any]
    canonical_json: str
    sha256: str
    response_model: type[PayloadT] = field(repr=False, compare=False)
    payload_repairer: PayloadRepairer = field(
        default=repair_wire_payload, repr=False, compare=False)
    compiler_version: str = HCX_SCHEMA_VERSION

    def verify_integrity(self) -> None:
        canonical = _canonical_json(self.schema)
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        if canonical != self.canonical_json or digest != self.sha256:
            raise HcxSchemaError("compiled HCX schema가 생성 후 변경되었습니다")

    def schema_copy(self) -> dict[str, Any]:
        self.verify_integrity()
        return deepcopy(self.schema)

    def validate_json(self, payload: str, *, repair: bool = True,
                      repairs_out: list[str] | None = None) -> PayloadT:
        """strict 검증. 그 **직전에** 결정론적 안전 복구를 한 번 적용한다.

        복구는 다른 필드에서 계산되는 값이 그 계산과 어긋날 때만 다시 계산하고,
        사용자 내용(claim·requested output·task)은 건드리지 않는다
        (`agent/hcx_repair`). 복구하지 않으면 파생 필드 하나 때문에 정상 추출된
        응답 전체가 버려진다 — 외부 독립 검토 §1.

        `repair=False` 는 **복구 없이** 재는 대조군용이다. 진단(`payload_shape`)은
        언제나 **원본**으로 남긴다. 복구본으로 남기면 무엇이 왔는지 알 수 없다.
        """

        self.verify_integrity()
        if not isinstance(payload, str) or not payload.strip():
            raise HcxPayloadValidationError(
                "HCX JSON payload가 비어 있습니다",
                issue_codes=("payload_empty",),
            )
        payload_shape = safe_payload_shape(payload)
        data: object = None
        applied: tuple[str, ...] = ()
        if repair:
            try:
                data = json.loads(payload)
            except (TypeError, ValueError):
                data = None
            else:
                data, applied = self.payload_repairer(data)
                if tuple(sorted(set(applied))) != applied or any(
                        not isinstance(code, str)
                        or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code)
                        for code in applied):
                    raise HcxSchemaError(
                        "HCX deterministic repair code가 유효하지 않습니다")
        if repairs_out is not None:
            repairs_out.extend(applied)
        try:
            if applied:
                return self.response_model.model_validate(data, strict=True)
            return self.response_model.model_validate_json(payload, strict=True)
        except ValidationError as exc:
            raise HcxPayloadValidationError(
                "HCX payload가 strict response model과 다릅니다",
                issue_codes=safe_validation_issue_codes(exc),
                issue_paths=safe_validation_issue_paths(exc),
                payload_shape=payload_shape,
            ) from exc


def compile_hcx_schema(
        model: type[PayloadT], *,
        payload_repairer: PayloadRepairer = repair_wire_payload,
        ) -> CompiledHcxSchema[PayloadT]:
    """Strict Pydantic response model을 HCX 지원 subset으로 compile한다."""

    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise HcxSchemaError("Pydantic BaseModel class만 compile할 수 있습니다")
    if model.model_config.get("extra") != "forbid":
        raise HcxSchemaError("HCX response model은 extra='forbid'여야 합니다")
    if not callable(payload_repairer):
        raise HcxSchemaError("HCX payload repairer는 callable이어야 합니다")
    raw = cast(dict[str, Any], model.model_json_schema(mode="validation"))
    definitions = raw.get("$defs", {})
    if not isinstance(definitions, dict):
        raise HcxSchemaError("$defs는 object여야 합니다")
    lowered = _lower_node(deepcopy(raw), definitions, (), depth=0)
    if not isinstance(lowered, dict) or lowered.get("type") != "object":
        raise HcxSchemaError("HCX Structured Output의 root는 object여야 합니다")
    property_count = _count_properties(lowered)
    if property_count > MAX_SCHEMA_PROPERTIES:
        raise HcxSchemaError(
            f"HCX schema property 수는 {MAX_SCHEMA_PROPERTIES} 이하여야 합니다")
    canonical = _canonical_json(lowered)
    digest = sha256(canonical.encode("utf-8")).hexdigest()
    return CompiledHcxSchema(
        schema=lowered, canonical_json=canonical, sha256=digest,
        response_model=model, payload_repairer=payload_repairer,
    )


__all__ = [
    "CompiledHcxSchema", "HCX_SCHEMA_VERSION", "HcxPayloadValidationError",
    "HcxSafePayloadShape", "HcxSchemaError", "compile_hcx_schema",
    "safe_payload_shape", "safe_validation_issue_codes",
    "safe_validation_issue_paths",
]
