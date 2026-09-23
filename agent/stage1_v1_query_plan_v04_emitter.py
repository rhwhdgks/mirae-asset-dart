"""Stage1 v1 outcome -> final QueryPlanHandoff 0.4 emitter.

SemanticIntent v1 and its answer contract are internal validation layers.  The
public Stage1 result for the current competition/team integration remains the
existing ``QueryPlanHandoff 0.4`` contract.  This emitter projects exactly that
query-plan boundary; it does not claim that v0.4 can carry every v1 answer
composition detail.

The older ``stage1_v1_query_plan_v04_downcast`` module remains the stricter
answer-binding audit adapter.  In contrast, this module intentionally accepts
every normal Stage1 outcome and emits the v0.4 query-plan/status view needed by
the evaluator and Stage2.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Annotated, Any, Literal, Mapping, TypeVar
import unicodedata
from uuid import NAMESPACE_URL, uuid5

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)

from .query_plan import (
    HandoffClarification,
    HandoffClarificationSlot,
    QueryPlanHandoff,
)
from .contracts import ResolvedQueryPlan
from .stage1_v1_outcome import Stage1NormalStatus, Stage1Outcome
from .stage1_v1_resolver import (
    ClarificationAuthority,
    ResolvedAuthority,
    TerminalAuthority,
)


EMITTER_VERSION = "stage1-v1-query-plan-v04-emitter/1.0"
EMISSION_SCOPE = "query-plan-handoff-v04/1.0"
SCHEMA_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_query_plan_v04_emitter.schema.json")
SCHEMA_DIGEST_ARTIFACT = Path(__file__).with_name("schemas") / (
    "stage1_v1_query_plan_v04_emitter.schema.sha256")
_HANDOFF_NAMESPACE = uuid5(
    NAMESPACE_URL, "mirae-dart/stage1-v1-query-plan-v04-emitter/1.0")
_CLARIFICATION_NAMESPACE = uuid5(
    NAMESPACE_URL, "mirae-dart/stage1-v1-query-plan-v04-clarification/1.0")

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class Stage1V1QueryPlanV04EmitterError(ValueError):
    """A normal v1 outcome cannot be represented as a v0.4 query-plan view."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )


def canonical_json(value: Any) -> str:
    def json_value(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return json_value(item.model_dump(mode="json", warnings=False))
        if isinstance(item, Mapping):
            return {key: json_value(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [json_value(child) for child in item]
        return item

    value = json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _strict_model(model: type[_ModelT], value: Any) -> _ModelT:
    if isinstance(value, BaseModel):
        raw = value.model_dump(mode="json", warnings=False)
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise TypeError(f"{model.__name__} instance 또는 full mapping이 필요합니다")
    return model.model_validate_json(canonical_json(raw), strict=True)


def _handoff_id(outcome: Stage1Outcome) -> str:
    return str(uuid5(
        _HANDOFF_NAMESPACE,
        f"{EMITTER_VERSION}:{outcome.outcome_digest}",
    ))


def _clarification_id(outcome: Stage1Outcome) -> str:
    return str(uuid5(
        _CLARIFICATION_NAMESPACE,
        f"{EMITTER_VERSION}:{outcome.outcome_digest}",
    ))


def _clarification_target(
        outcome: Stage1Outcome,
        *,
        role_hint: str,
        applies_to_item_ids: list[str],
        ) -> str:
    """Map a semantic clarification role to the public v0.4 target.

    ``target`` is made more specific only when every referenced answer item has
    the same typed target kind.  This is a structural derivation from the
    source intent, not a question-ID or Gold lookup.
    """

    if role_hint == "target":
        by_id = {
            item.item_id: item.target.kind
            for item in outcome.resolution_decision.source_intent.answer_items
        }
        kinds = {
            by_id[item_id]
            for item_id in applies_to_item_ids
            if item_id in by_id
        }
        if len(kinds) == 1:
            return next(iter(kinds))
        return "target"
    return {
        "entity": "company",
        "event": "event",
        "time": "period",
        "timepoint": "timepoint",
        # v1 calls the semantic axis a qualifier; the frozen public v0.4
        # contract names the same financial/document axis ``scope``.
        "qualifier": "scope",
        "selection": "selection",
        "value_kind": "value_kind",
    }.get(role_hint, role_hint)


def _clarification_handoff(
        outcome: Stage1Outcome,
        authority: ClarificationAuthority,
        ) -> QueryPlanHandoff:
    slots = tuple(
        HandoffClarificationSlot(
            slot_id=slot.slot_id,
            target=_clarification_target(
                outcome,
                role_hint=slot.role_hint,
                applies_to_item_ids=list(slot.applies_to_item_ids),
            ),
            allowed_values=tuple(option.value for option in slot.options),
        )
        for slot in authority.slots
    )
    clarification = HandoffClarification(
        clarification_id=_clarification_id(outcome),
        plan_revision=0,
        # 항목을 엮지 않는다. 표현은 역질문 층이 정한다.
        question=tuple(dict.fromkeys(
            slot.prompt.strip() for slot in authority.slots
            if slot.prompt.strip())),
        slots=slots,
    )
    return QueryPlanHandoff(
        handoff_id=_handoff_id(outcome),
        status="needs_clarification",
        clarification=clarification,
    )


_POLICY_CODES = frozenset({
    "policy_refusal", "future_forecast", "investment_advice",
    "prompt_injection_direct", "prompt_injection_role_impersonation",
    "external_tool_request", "external_url_request", "personal_data_request",
    "abusive_input", "off_topic_request",
})
_OUT_OF_SCOPE_CODES = frozenset({"out_of_scope", "corp_not_in_universe"})
_UNSUPPORTED_CODES = frozenset({
    "unsupported_request",
    "corpus_coverage_unavailable",
    "unsupported_temporal_scope",
    "unsupported_semantic_target",
    "causal_inference_beyond_scope",
})
_STATUS_ONLY_CODES = frozenset({
    "policy_refusal", "out_of_scope", "unsupported_request",
})


def _terminal_status(codes: tuple[str, ...]) -> str:
    classes: set[str] = set()
    if any(code in _POLICY_CODES for code in codes):
        classes.add("policy_refusal")
    if any(code in _OUT_OF_SCOPE_CODES for code in codes):
        classes.add("out_of_scope")
    if any(code in _UNSUPPORTED_CODES for code in codes):
        classes.add("unsupported_request")
    # An external-tool refusal can independently carry a temporal boundary.
    # Both reasons are public and useful, but they still describe one terminal
    # refusal rather than two incompatible handoff statuses.
    if (classes == {"policy_refusal", "unsupported_request"}
            and any(code in {
                "external_tool_request", "external_url_request"}
                for code in codes)
            and all(code not in _UNSUPPORTED_CODES
                    or code == "unsupported_temporal_scope"
                    for code in codes)):
        return "policy_refusal"
    if len(classes) != 1:
        raise Stage1V1QueryPlanV04EmitterError(
            "terminal reason이 하나의 v0.4 status로 닫히지 않습니다")
    return next(iter(classes))


def _terminal_handoff(
        outcome: Stage1Outcome,
        authority: TerminalAuthority,
        ) -> QueryPlanHandoff:
    codes = tuple(dict.fromkeys(row.code for row in authority.reasons))
    status = _terminal_status(codes)
    detailed = tuple(code for code in codes if code not in _STATUS_ONLY_CODES)
    reasons = detailed or codes
    return QueryPlanHandoff(
        handoff_id=_handoff_id(outcome),
        status=status,  # type: ignore[arg-type]
        reasons=reasons,
    )


def _document_selector(
        *,
        rcept_no: str | None = None,
        doc_group: str | None = None,
        form: str | None = None,
        ) -> dict[str, Any]:
    return {
        "doc_id": None,
        "rcept_no": rcept_no,
        "doc_group": doc_group,
        "event_type": None,
        "form": form,
        "report_name_contains": None,
        "rcept_from": None,
        "rcept_to": None,
        "is_correction": None,
    }


def _compat_query(surface: str) -> str:
    """Project a semantic topic to the stable v0.4 narrative query surface."""

    value = unicodedata.normalize("NFC", surface)
    value = re.sub(r"[·ㆍ,;/]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s+및\s+", " ", value)
    if value.startswith("주요 ") and len(value) > len("주요 "):
        value = value[len("주요 "):]
    value = re.sub(r"(?<=[0-9A-Za-z가-힣])계획$", " 계획", value)
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        raise Stage1V1QueryPlanV04EmitterError(
            "v0.4 narrative retrieval query가 비었습니다")
    return value


def _compat_slot(*, target_surface: str, field_surface: str) -> str:
    """Normalize a user field, never a source path, into a v0.4 slot label."""

    slot = re.sub(
        r"[^0-9A-Za-z가-힣]", "", unicodedata.normalize("NFC", field_surface))
    target = re.sub(
        r"[^0-9A-Za-z가-힣]", "", unicodedata.normalize("NFC", target_surface))
    if target and slot.startswith(target) and len(slot) > len(target):
        slot = slot[len(target):]
    slot = re.sub(r"(?:된|한)?이유$", "사유", slot)
    if not slot:
        raise Stage1V1QueryPlanV04EmitterError(
            "v0.4 requested slot 정규화 결과가 비었습니다")
    return slot


def _base_plan_payload(outcome: Stage1Outcome) -> dict[str, Any]:
    envelope = outcome.ready_envelope
    if envelope is None:
        raise Stage1V1QueryPlanV04EmitterError(
            "resolved outcome에 execution plan이 없습니다")
    return envelope.execution_plan.resolved_plan.model_dump(
        mode="json", warnings=False)


def _strict_plan(payload: Mapping[str, Any]) -> ResolvedQueryPlan:
    return _strict_model(ResolvedQueryPlan, payload)


def _financial_public_compatibility_plan(
        plan: ResolvedQueryPlan,
        ) -> ResolvedQueryPlan:
    """Project financial execution identifiers/defaults to stable v0.4 form.

    The v1 execution plan keeps compiler-local output identifiers.  The v0.4
    handoff instead reserves ``derived-N`` for public calculation outputs and
    makes the financial view/cutoff policy visible as explicit defaults.  This
    works on a copied public payload only; it never changes the source v1
    execution plan or resolution provenance.
    """

    payload = plan.model_dump(mode="json", warnings=False)
    financial = [task for task in payload["tasks"] if task["kind"] == "financial"]
    if not financial:
        return plan
    cutoff = payload["corpus_cutoff"]
    all_at_cutoff = all(task["as_of"] == cutoff for task in financial)

    derivations = payload["derivations"]
    source_output_ids = {
        fact["output_id"]
        for task in payload["tasks"]
        for fact in task.get("facts", [])
    }
    source_output_ids.update(
        output_id
        for task in payload["tasks"]
        for output_id in [task.get("output_id")]
        if output_id is not None
    )
    source_output_ids.update(
        output["output_id"]
        for task in payload["tasks"]
        for output in task.get("field_outputs", [])
    )
    public_ids = [f"derived-{index}" for index in range(1, len(derivations) + 1)]
    if set(public_ids) & source_output_ids:
        raise Stage1V1QueryPlanV04EmitterError(
            "financial public derived ID가 source output ID와 충돌합니다")
    old_ids = [row["output_id"] for row in derivations]
    if len(old_ids) != len(set(old_ids)):
        raise Stage1V1QueryPlanV04EmitterError(
            "financial derivation output ID가 유일하지 않습니다")
    remap = dict(zip(old_ids, public_ids, strict=True))
    for row, public_id in zip(derivations, public_ids, strict=True):
        row["output_id"] = public_id
        for operand in row["operands"]:
            operand["output_id"] = remap.get(
                operand["output_id"], operand["output_id"])
    for claim in payload["premise_claims"]:
        for output in claim["verify_with"]:
            output["output_id"] = remap.get(
                output["output_id"], output["output_id"])

    policy_defaults: list[str] = []
    for task in financial:
        default = f"view={task['view']}"
        if default not in policy_defaults:
            policy_defaults.append(default)
    # A period-bound comparison can intentionally use a different filing
    # cutoff for each operand.  Those cutoffs are explicit task coordinates,
    # not an implicit corpus-wide default.  Keep the historical public default
    # only when it truthfully applies to every financial task.
    if all_at_cutoff:
        policy_defaults.append(f"as_of=corpus_cutoff({cutoff})")
    # Keep caller-provided, evidence-backed defaults but put the stable public
    # financial policy first and remove duplicates structurally.
    payload["applied_defaults"] = [
        *policy_defaults,
        *(value for value in payload["applied_defaults"]
          if value not in policy_defaults),
    ]
    return _strict_plan(payload)


_KOREAN_NUMERIC_PREMISE = re.compile(
    r"^\s*(?P<value>-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?)\s*"
    r"(?P<unit>조원|억원|만원|원)\s*$")


def _public_numeric_premise(raw_text: str) -> tuple[str, str] | None:
    """Return a v0.4 numeric coordinate only for one literal Korean amount.

    This deliberately recognizes no prose, arithmetic, commas, ranges, or
    implicit scale.  ``raw_text`` remains the source contract; the returned
    scalar is merely its public value/unit projection.
    """

    match = _KOREAN_NUMERIC_PREMISE.fullmatch(raw_text)
    if match is None:
        return None
    return match.group("value"), match.group("unit")


def _typed_public_event_selector(
        outcome: Stage1Outcome, payload: dict[str, Any],
        ) -> None:
    """Expose only proof-bound public event selector facets.

    The canonical event key stays an internal identity.  A root receipt is
    copied only from a typed selector proof. Counterparty/product/contract-name
    facets are copied only from typed authority carrying exact canonical field
    proofs. No event-key lookup or question/fixture-specific recovery is
    performed here.
    """

    authority = outcome.resolution_decision.authority
    if not isinstance(authority, ResolvedAuthority):
        return
    typed_items = authority.resolution.items
    for item in typed_items:
        typed = item.resolution
        typed_kind = typed.kind
        if typed_kind not in {
                "selected_event", "event_amount_change",
                "event_lifecycle_composite", "correction_lineage",
                "termination_reported_status"}:
            continue
        root_receipt = getattr(typed, "root_receipt", None)
        selector_proof = getattr(typed, "selector_proof", None)
        if selector_proof is None:
            selector_proof = getattr(typed, "root_selector_proof", None)
        root_is_proven = bool(
            root_receipt is not None and selector_proof is not None
            and selector_proof.source_receipt == root_receipt
        )
        public_selector = getattr(typed, "public_selector", None)
        # In a mixed same-day composite, the document tasks are part of the
        # answer and the event key already fixes the status identity.  Adding
        # redundant public facets would narrow only one branch of that joined
        # plan and change the frozen v0.4 surface without adding authority.
        task_kinds = {task.get("kind") for task in payload["tasks"]}
        if {"document", "event"}.issubset(task_kinds):
            public_selector = None
        # Existing correction/event-amount paths need a proven root.  The
        # reported-termination authority has no unambiguous root by design and
        # may still publish field-proven facets from the status receipt.
        if not root_is_proven and public_selector is None:
            continue
        typed_corp_code = getattr(typed, "corp_code", None)
        if typed_corp_code is None:
            typed_corp_code = getattr(typed, "issuer_corp_code", None)
        for task in payload["tasks"]:
            if task.get("corp_code") != typed_corp_code:
                continue
            if (typed_kind == "correction_lineage"
                    and (task.get("kind") != "correction"
                         or task.get("operation") != typed.operation)):
                continue
            if typed_kind != "correction_lineage" and task.get("kind") != "event":
                continue
            selector = (
                task.get("selector")
                if task.get("kind") == "event"
                else task.get("event_selector")
            )
            if not isinstance(selector, dict):
                continue
            # Do not discover an event from its key.  This only supplements a
            # task whose public selector is already backed by this authority.
            if selector.get("event_key") not in {
                    None, getattr(typed, "event_key", None)}:
                continue
            # A correction ``history`` task intentionally searches the proven
            # counterparty/product lineage.  Publishing one root receipt there
            # would narrow the legacy v0.4 execution semantics from history to
            # one seeded event.  Diff and selected-event tasks retain the root.
            if root_is_proven:
                omit_seed = bool(
                    public_selector is not None
                    and (
                        (public_selector.counterparty
                         and public_selector.product_keywords)
                        or (typed_kind == "selected_event"
                            and typed.operation == "timeline"
                            and public_selector.contract_name)
                    )
                )
                if (typed_kind == "correction_lineage"
                        and typed.operation == "history") or omit_seed:
                    selector["seed_rcept_no"] = None
                else:
                    selector["seed_rcept_no"] = root_receipt
            counterparty = (
                public_selector.counterparty
                if public_selector is not None
                else getattr(typed, "counterparty", None)
            )
            if counterparty is not None:
                selector["counterparty"] = counterparty
            product_keywords = (
                public_selector.product_keywords
                if public_selector is not None
                else getattr(typed, "product_keywords", None)
            )
            if product_keywords:
                selector["keywords"] = list(product_keywords)
            contract_name = (
                public_selector.contract_name
                if public_selector is not None else None
            )
            if contract_name is not None:
                selector["contract_name"] = contract_name


def _public_premise_projection(payload: dict[str, Any]) -> None:
    """Use v0.4 claim IDs and omit v1-only task-verification links."""

    for index, claim in enumerate(payload["premise_claims"], start=1):
        # Derivation comparisons already use the compiler's stable premise
        # namespace and can carry both scalar and task proof edges.  Preserve
        # that complete public contract; scalar/state claims use the legacy
        # claim namespace.
        if claim["kind"] != "comparison":
            claim["claim_id"] = f"claim-{index}"
        # 과제 결속은 **둘 이상을 걸칠 때만** 싣는다.  전제가 과제 하나에
        # 걸리는 경우 그 결속은 계획에서 이미 명백해서 되풀이일 뿐이고,
        # 여러 과제를 걸치는 비교 전제만 어느 과제들을 함께 봐야 하는지를
        # 계획에서 읽어낼 수 없다.
        if len(claim.get("verify_tasks") or ()) < 2:
            claim.pop("verify_tasks", None)
        if claim["kind"] != "numeric" or not claim["verify_with"]:
            continue
        parsed = _public_numeric_premise(claim["raw_text"])
        if parsed is not None:
            claim["value"], claim["unit"] = parsed


def _public_v04_compatibility_projection(
        outcome: Stage1Outcome, plan: ResolvedQueryPlan,
        ) -> ResolvedQueryPlan:
    """Apply final generic public-only compatibility conventions.

    This operates on a serialized copy and leaves the v1 execution plan,
    answer roots, premise proofs, and typed resolution untouched.
    """

    payload = plan.model_dump(mode="json", warnings=False)
    _typed_public_event_selector(outcome, payload)
    _public_premise_projection(payload)
    return _strict_plan(payload)


def _periodic_document_narrative_plan(
        outcome: Stage1Outcome,
        authority: ResolvedAuthority,
        ) -> ResolvedQueryPlan:
    if len(authority.resolution.items) != 1:
        raise Stage1V1QueryPlanV04EmitterError(
            "periodic document narrative resolution item은 하나여야 합니다")
    resolved_item = authority.resolution.items[0]
    typed = resolved_item.resolution
    intent_items = outcome.resolution_decision.source_intent.answer_items
    if len(intent_items) != 1 or intent_items[0].item_id != resolved_item.item_id:
        raise Stage1V1QueryPlanV04EmitterError(
            "periodic document narrative intent/resolution binding이 다릅니다")
    intent_item = intent_items[0]
    # 답 모양은 `record_list` 로 읽히기도 `record` 로 읽히기도 한다.  둘 다 이름
    # 붙은 필드 여럿을 한 문서에서 읽는 같은 요청이고, 내려가는 v0.4 task 도 같다.
    # 표현 차이로 거절하면 같은 질문이 판마다 다르게 끝난다.
    if (
        intent_item.operation != "retrieve"
        or intent_item.output.shape not in {
            "record_list", "record", "narrative"
        }
        or intent_item.output.projection_mode != "named_fields"
    ):
        raise Stage1V1QueryPlanV04EmitterError(
            "periodic document narrative semantic shape가 다릅니다")
    slots = (
        list(typed.canonical_requested_slots)
        if typed.canonical_requested_slots else [
            _compat_slot(
                target_surface=intent_item.target.surface,
                field_surface=surface,
            )
            for surface in intent_item.output.field_surfaces
        ])
    if not slots or len(slots) != len(set(slots)):
        raise Stage1V1QueryPlanV04EmitterError(
            "periodic document narrative slot inventory가 다릅니다")
    plan = _base_plan_payload(outcome)
    if plan["derivations"] or plan["premise_claims"]:
        raise Stage1V1QueryPlanV04EmitterError(
            "periodic document narrative compatibility는 계산/premise를 축약하지 않습니다")
    plan["tasks"] = [{
        "task_id": "task-1",
        "kind": "narrative",
        "operation": "search",
        "corp_codes": [typed.corp_code],
        "corp_names": [typed.corp_name],
        "as_of": plan["corpus_cutoff"],
        "retrieval_query": (
            typed.source_retrieval_query
            or _compat_query(intent_item.target.surface)),
        "document_selector": _document_selector(
            rcept_no=typed.receipt_no,
            doc_group="periodic",
        ),
        "periods": [],
        "requested_slots": slots,
    }]
    return _strict_plan(plan)


def _document_attribute_plan(
        outcome: Stage1Outcome,
        authority: ResolvedAuthority,
        ) -> ResolvedQueryPlan:
    resolved_items = authority.resolution.items
    if len(resolved_items) != 2 or any(
            row.resolution.kind != "document_attribute_evidence"
            for row in resolved_items):
        raise Stage1V1QueryPlanV04EmitterError(
            "document attribute compatibility에는 두 source item이 필요합니다")
    intent_by_id = {
        row.item_id: row
        for row in outcome.resolution_decision.source_intent.answer_items
    }
    if set(intent_by_id) != {row.item_id for row in resolved_items}:
        raise Stage1V1QueryPlanV04EmitterError(
            "document attribute intent/resolution item inventory가 다릅니다")

    role_order = {"earlier": 0, "later": 1}
    ordered = sorted(
        resolved_items,
        key=lambda row: role_order[row.resolution.evidence.document_role],
    )
    if [row.resolution.evidence.document_role for row in ordered] != [
            "earlier", "later"]:
        raise Stage1V1QueryPlanV04EmitterError(
            "document attribute source chronology가 닫히지 않았습니다")

    plan = _base_plan_payload(outcome)
    if plan["derivations"] or plan["premise_claims"]:
        raise Stage1V1QueryPlanV04EmitterError(
            "document attribute compatibility는 계산/premise를 축약하지 않습니다")
    tasks: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        typed = row.resolution
        intent_item = intent_by_id[row.item_id]
        fields = intent_item.output.field_surfaces
        if (
            intent_item.operation != "retrieve"
            or intent_item.output.projection_mode != "named_fields"
            or len(fields) != 1
        ):
            raise Stage1V1QueryPlanV04EmitterError(
                "document attribute semantic field는 item당 하나여야 합니다")
        slot = _compat_slot(
            target_surface=intent_item.target.surface,
            field_surface=fields[0],
        )
        evidence = typed.evidence
        tasks.append({
            "task_id": f"task-{index}",
            "kind": "disclosure",
            "operation": "lookup",
            "corp_code": evidence.issuer_corp_code,
            "corp_name": evidence.issuer_corp_name,
            "as_of": plan["corpus_cutoff"],
            "document_selector": _document_selector(
                rcept_no=evidence.receipt_no),
            "event_selector": None,
            "requested_slots": [slot],
            "output_id": None,
            "field_outputs": [],
        })
    plan["tasks"] = tasks
    return _strict_plan(plan)


def _periodic_narrative_comparison_plan(
        outcome: Stage1Outcome,
        authority: ResolvedAuthority,
        ) -> ResolvedQueryPlan:
    if len(authority.resolution.items) != 1:
        raise Stage1V1QueryPlanV04EmitterError(
            "periodic narrative comparison resolution item은 하나여야 합니다")
    resolved_item = authority.resolution.items[0]
    typed = resolved_item.resolution
    intent_items = outcome.resolution_decision.source_intent.answer_items
    if len(intent_items) != 1 or intent_items[0].item_id != resolved_item.item_id:
        raise Stage1V1QueryPlanV04EmitterError(
            "periodic comparison intent/resolution binding이 다릅니다")
    intent_item = intent_items[0]
    if intent_item.operation != "compare" or len(typed.documents) != 2:
        raise Stage1V1QueryPlanV04EmitterError(
            "periodic narrative comparison semantic shape가 다릅니다")
    documents = typed.documents
    plan = _base_plan_payload(outcome)
    if plan["derivations"] or plan["premise_claims"]:
        raise Stage1V1QueryPlanV04EmitterError(
            "periodic narrative comparison compatibility는 계산/premise를 축약하지 않습니다")
    form_by_range = []
    for document in documents:
        coordinate = (
            document.period_start.month, document.period_start.day,
            document.period_end.month, document.period_end.day,
        )
        form_by_range.append({
            (1, 1, 12, 31): "사업보고서",
            (1, 1, 6, 30): "반기보고서",
            (1, 1, 3, 31): "분기보고서",
            (7, 1, 9, 30): "분기보고서",
        }.get(coordinate))
    # A single public selector must not claim annual form when one source is a
    # quarter.  Ordered exact ranges remain the authority for mixed forms.
    public_form = (
        form_by_range[0]
        if form_by_range[0] is not None and len(set(form_by_range)) == 1
        else None
    )
    plan["tasks"] = [{
        "task_id": "task-1",
        "kind": "narrative",
        "operation": "compare",
        "corp_codes": [documents[0].issuer_corp_code],
        "corp_names": [documents[0].issuer_corp_name],
        "as_of": plan["corpus_cutoff"],
        "retrieval_query": _compat_query(intent_item.target.surface),
        "document_selector": _document_selector(
            doc_group="periodic",
            form=public_form,
        ),
        "periods": [
            {
                "start": document.period_start.isoformat(),
                "end": document.period_end.isoformat(),
            }
            for document in documents
        ],
        "requested_slots": [],
    }]
    return _strict_plan(plan)


def _document_version_history_plan(
        outcome: Stage1Outcome, authority: ResolvedAuthority,
        ) -> ResolvedQueryPlan:
    """Project lineage verification to the proved annual filing root.

    v1 retains ``base_year``, root/latest receipts, and every lineage member.
    QueryPlan 0.4 has no slots for that authority, so this projection keeps
    only the root receipt and its existence claim.  Consumers
    needing lineage identity must retain the source outcome beside the handoff.
    """

    item = authority.resolution.items[0]
    typed = item.resolution
    plan = _base_plan_payload(outcome)
    if len(plan["tasks"]) != 1 or len(plan["premise_claims"]) != 1:
        raise Stage1V1QueryPlanV04EmitterError("document version plan inventory가 다릅니다")
    plan["tasks"] = [{
        "task_id": "task-1", "kind": "document", "operation": "version_history",
        "corp_code": typed.corp_code, "corp_name": typed.corp_name,
        "as_of": plan["corpus_cutoff"], "event_selector": None,
        "selector": {
            **_document_selector(doc_group="periodic", form="사업보고서"),
            "rcept_no": typed.root_receipt,
        },
    }]
    # 전제 인용은 **질문 원문 그대로** 둔다.
    #
    # 예전에는 여기서 `"정정된 적이 없지"` 를 `"정정된 적이 없다"` 로 바꿔
    # 내보냈다. 옛 Gold 가 평서형을 적어 두어 맞춘 것인데, 그러면 사용자가
    # 실제로 무엇을 물었는지 잃는다. Gold 정정 A 로 그쪽이 원문이 된 뒤로는
    # 맞출 이유도 없다. 평서형 변환이 필요하면 원문과 `kind` 로 뒤에서 한다.
    return _strict_plan(plan)


def _lifecycle_attribute_kind(surface: str) -> "str | None":
    compact = re.sub(r"\s+", "", surface)
    if "해지" in compact and any(token in compact for token in ("사유", "이유", "원인")):
        return "termination_reason"
    if "해지" in compact and any(token in compact for token in ("금액", "대금", "규모")):
        return "termination_amount"
    if "효력" in compact and "조건" in compact:
        return "effectiveness_condition"
    if "계약" in compact and any(token in compact for token in ("금액", "대금", "규모")):
        return "contract_amount"
    return None


def _lifecycle_composite_plan(
        outcome: Stage1Outcome, authority: ResolvedAuthority,
        ) -> ResolvedQueryPlan:
    """Project proof-rich lifecycle authority to stable v0.4 task shapes."""

    source_items = outcome.resolution_decision.source_intent.answer_items
    if len(source_items) != len(authority.resolution.items):
        raise Stage1V1QueryPlanV04EmitterError(
            "lifecycle intent/resolution inventory가 다릅니다")
    plan = _base_plan_payload(outcome)
    if plan["derivations"] or plan["premise_claims"]:
        raise Stage1V1QueryPlanV04EmitterError(
            "lifecycle compatibility는 계산/premise를 축약하지 않습니다")

    attribute_slots = {
        "contract_amount": "계약금액",
        "termination_amount": "해지금액",
        "termination_reason": "해지사유",
        "effectiveness_condition": "효력발생조건",
    }
    field_kinds = [
        [_lifecycle_attribute_kind(surface)
         for surface in item.output.field_surfaces]
        for item in source_items
    ]
    if all(kinds and all(kind is not None for kind in kinds)
           for kinds in field_kinds):
        task_rows: list[tuple[str, str, str, dict[str, Any]]] = []
        for source, resolved_item, kinds in zip(
                source_items, authority.resolution.items, field_kinds,
                strict=True):
            typed = resolved_item.resolution
            by_kind = {row.kind: row for row in typed.attributes}
            for kind in kinds:
                attribute = by_kind.get(kind)
                if attribute is None:
                    raise Stage1V1QueryPlanV04EmitterError(
                        "lifecycle attribute proof가 없습니다")
                task_rows.append((
                    attribute.source_receipt, source.item_id, kind, {
                    "task_id": "",  # assigned after proof-bound chronology sort
                    "kind": "disclosure", "operation": "lookup",
                    "corp_code": typed.corp_code,
                    "corp_name": typed.corp_name,
                    "as_of": typed.as_of,
                    "document_selector": _document_selector(
                        rcept_no=attribute.source_receipt),
                    "event_selector": None,
                    "requested_slots": [attribute_slots[kind]],
                    "output_id": None, "field_outputs": [],
                }))
        # A source receipt is a typed lifecycle proof coordinate.  Sort by it
        # before assigning public task IDs so v0.4 has stable chronology even
        # when semantic fields were asked in a different order.
        task_rows.sort(key=lambda row: row[:3])
        tasks = []
        for index, (_receipt, _item_id, _kind, task) in enumerate(
                task_rows, start=1):
            task["task_id"] = f"task-{index}"
            tasks.append(task)
        plan["tasks"] = tasks
        return _strict_plan(plan)

    if len(source_items) == 1:
        source = source_items[0]
        typed = authority.resolution.items[0].resolution
        correction_requested = any(
            re.search(r"정정|전후|흐름|변경", surface)
            for surface in source.output.field_surfaces)
        if (correction_requested and typed.correction_receipts
                and len(typed.status_timepoints) == 2):
            plan["tasks"] = [
                {
                    "task_id": "task-1", "kind": "event",
                    "operation": "status", "corp_code": typed.corp_code,
                    "corp_name": typed.corp_name,
                    "selector": {
                        "event_key": typed.event_key,
                        "seed_rcept_no": typed.root_receipt,
                        "event_type": None, "counterparty": None,
                        "contract_name": None, "keywords": [],
                        "event_from": None, "event_to": None,
                    },
                    "timepoints": list(typed.status_timepoints),
                    "requested_slots": [], "output_id": None,
                    "field_outputs": [],
                },
                {
                    "task_id": "task-2", "kind": "correction",
                    "operation": "history", "corp_code": typed.corp_code,
                    "corp_name": typed.corp_name, "as_of": typed.as_of,
                    "document_selector": None,
                    "event_selector": {
                        "event_key": None,
                        "seed_rcept_no": typed.root_receipt,
                        "event_type": None, "counterparty": None,
                        "contract_name": None, "keywords": [],
                        "event_from": None, "event_to": None,
                    },
                    "requested_slots": [], "output_id": None,
                    "field_outputs": [],
                },
            ]
            return _strict_plan(plan)
    envelope = outcome.ready_envelope
    if envelope is None:
        raise Stage1V1QueryPlanV04EmitterError(
            "lifecycle outcome에 execution plan이 없습니다")
    return _strict_plan(envelope.execution_plan.resolved_plan)


def _team_compatible_resolved_plan(outcome: Stage1Outcome) -> ResolvedQueryPlan:
    """Project typed v1 resolution families to the stable team v0.4 shapes."""

    authority = outcome.resolution_decision.authority
    if not isinstance(authority, ResolvedAuthority):
        raise Stage1V1QueryPlanV04EmitterError(
            "ready outcome에는 resolved authority가 필요합니다")
    kinds = tuple(row.resolution.kind for row in authority.resolution.items)
    if kinds == ("periodic_document_narrative",):
        plan = _periodic_document_narrative_plan(outcome, authority)
    elif kinds and set(kinds) == {"document_attribute_evidence"}:
        plan = _document_attribute_plan(outcome, authority)
    elif kinds == ("periodic_narrative_comparison",):
        plan = _periodic_narrative_comparison_plan(outcome, authority)
    elif kinds == ("document_version_history",):
        plan = _document_version_history_plan(outcome, authority)
    elif kinds and set(kinds) == {"event_lifecycle_composite"}:
        plan = _lifecycle_composite_plan(outcome, authority)
    else:
        envelope = outcome.ready_envelope
        if envelope is None:
            raise Stage1V1QueryPlanV04EmitterError(
                "resolved outcome에 execution plan이 없습니다")
        plan = _strict_plan(envelope.execution_plan.resolved_plan)
    plan = _financial_public_compatibility_plan(plan)
    return _public_v04_compatibility_projection(outcome, plan)


def _derive_handoff(outcome: Stage1Outcome) -> QueryPlanHandoff:
    if outcome.status in {"ready", "partial_ready"}:
        envelope = outcome.ready_envelope
        if envelope is None:
            raise Stage1V1QueryPlanV04EmitterError(
                "resolved outcome에 execution plan이 없습니다")
        return QueryPlanHandoff(
            handoff_id=_handoff_id(outcome),
            status="ready",
            plan=_team_compatible_resolved_plan(outcome),
        )

    authority = outcome.resolution_decision.authority
    if outcome.status == "needs_clarification":
        if not isinstance(authority, ClarificationAuthority):
            raise Stage1V1QueryPlanV04EmitterError(
                "clarification status와 authority가 다릅니다")
        return _clarification_handoff(outcome, authority)
    if outcome.status == "terminal":
        if not isinstance(authority, TerminalAuthority):
            raise Stage1V1QueryPlanV04EmitterError(
                "terminal status와 authority가 다릅니다")
        return _terminal_handoff(outcome, authority)
    raise Stage1V1QueryPlanV04EmitterError(
        f"알 수 없는 Stage1 outcome status입니다: {outcome.status}")


class Stage1V1QueryPlanV04Emission(_StrictFrozenModel):
    """Digest-bound final v0.4 query-plan view of one normal v1 outcome."""

    schema_version: Literal[EMITTER_VERSION] = EMITTER_VERSION
    emission_scope: Literal[EMISSION_SCOPE] = EMISSION_SCOPE
    source_status: Stage1NormalStatus
    source_outcome_digest: Digest
    source_outcome: Stage1Outcome
    handoff: QueryPlanHandoff
    emission_digest: Digest

    @field_validator("source_outcome", mode="before")
    @classmethod
    def strict_source_outcome(cls, value: Any) -> Stage1Outcome:
        return _strict_model(Stage1Outcome, value)

    @field_validator("handoff", mode="before")
    @classmethod
    def strict_handoff(cls, value: Any) -> QueryPlanHandoff:
        return _strict_model(QueryPlanHandoff, value)

    @classmethod
    def compute_digest(cls, payload: Mapping[str, Any]) -> str:
        body = dict(payload)
        body.pop("emission_digest", None)
        return canonical_sha256(body)

    @classmethod
    def create(
            cls,
            *,
            source_outcome: Stage1Outcome | Mapping[str, Any],
            ) -> "Stage1V1QueryPlanV04Emission":
        outcome = _strict_model(Stage1Outcome, source_outcome)
        body: dict[str, Any] = {
            "schema_version": EMITTER_VERSION,
            "emission_scope": EMISSION_SCOPE,
            "source_status": outcome.status,
            "source_outcome_digest": outcome.outcome_digest,
            "source_outcome": outcome,
            "handoff": _derive_handoff(outcome),
        }
        body["emission_digest"] = cls.compute_digest(body)
        return cls.model_validate_json(canonical_json(body), strict=True)

    @model_validator(mode="after")
    def validate_emission(self) -> "Stage1V1QueryPlanV04Emission":
        outcome = _strict_model(Stage1Outcome, self.source_outcome)
        expected = _derive_handoff(outcome)
        if self.source_status != outcome.status:
            raise ValueError("emission source_status가 outcome과 다릅니다")
        if self.source_outcome_digest != outcome.outcome_digest:
            raise ValueError("emission source digest가 outcome과 다릅니다")
        if canonical_json(self.handoff) != canonical_json(expected):
            raise ValueError("emitted QueryPlanHandoff가 source outcome과 다릅니다")
        if self.emission_digest != self.compute_digest(
                self.model_dump(mode="json", warnings=False)):
            raise ValueError("query-plan emission digest가 다릅니다")
        return self

    def handoff_json(self) -> str:
        verified = load_stage1_v1_query_plan_v04_emission_json(
            canonical_json(self))
        return canonical_json(_strict_model(QueryPlanHandoff, verified.handoff))


def emit_stage1_v1_query_plan_v04(
        source_outcome: Stage1Outcome | Mapping[str, Any],
        ) -> Stage1V1QueryPlanV04Emission:
    return Stage1V1QueryPlanV04Emission.create(source_outcome=source_outcome)


def load_stage1_v1_query_plan_v04_emission_json(
        payload: str | bytes | bytearray,
        ) -> Stage1V1QueryPlanV04Emission:
    return Stage1V1QueryPlanV04Emission.model_validate_json(
        payload, strict=True)


def verify_stage1_v1_query_plan_v04_emission_digest(
        value: Stage1V1QueryPlanV04Emission | Mapping[str, Any],
        ) -> str:
    verified = _strict_model(Stage1V1QueryPlanV04Emission, value)
    return verified.emission_digest


def _schema_artifact_bytes() -> tuple[bytes, bytes, str]:
    schema = canonical_json(
        Stage1V1QueryPlanV04Emission.model_json_schema(
            mode="validation")).encode("utf-8")
    digest = sha256(schema).hexdigest()
    sidecar = f"{digest}  {SCHEMA_ARTIFACT.name}\n".encode("ascii")
    return schema, sidecar, digest


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_stage1_v1_query_plan_v04_emitter_schema_artifacts() -> str:
    schema, sidecar, digest = _schema_artifact_bytes()
    _atomic_write(SCHEMA_ARTIFACT, schema)
    _atomic_write(SCHEMA_DIGEST_ARTIFACT, sidecar)
    return digest


def verify_stage1_v1_query_plan_v04_emitter_schema_artifacts() -> str:
    schema, sidecar, digest = _schema_artifact_bytes()
    try:
        if SCHEMA_ARTIFACT.read_bytes() != schema:
            raise RuntimeError("Stage1 v1 -> QueryPlanHandoff 0.4 emitter schema drift")
        if SCHEMA_DIGEST_ARTIFACT.read_bytes() != sidecar:
            raise RuntimeError(
                "Stage1 v1 -> QueryPlanHandoff 0.4 emitter schema digest drift")
    except OSError as exc:
        raise RuntimeError(
            "Stage1 v1 -> QueryPlanHandoff 0.4 emitter schema artifact missing") from exc
    return digest


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write", action="store_true")
    modes.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    digest = (
        write_stage1_v1_query_plan_v04_emitter_schema_artifacts()
        if args.write
        else verify_stage1_v1_query_plan_v04_emitter_schema_artifacts()
    )
    print(f"PASS: {EMITTER_VERSION} schema_sha256={digest}")
    return 0


__all__ = [
    "EMISSION_SCOPE",
    "EMITTER_VERSION",
    "SCHEMA_ARTIFACT",
    "SCHEMA_DIGEST_ARTIFACT",
    "Stage1V1QueryPlanV04Emission",
    "Stage1V1QueryPlanV04EmitterError",
    "canonical_json",
    "canonical_sha256",
    "emit_stage1_v1_query_plan_v04",
    "load_stage1_v1_query_plan_v04_emission_json",
    "verify_stage1_v1_query_plan_v04_emission_digest",
    "verify_stage1_v1_query_plan_v04_emitter_schema_artifacts",
    "write_stage1_v1_query_plan_v04_emitter_schema_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(_main())
