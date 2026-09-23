"""Orchestrator — QueryPlanHandoff → AnswerPayload 상태 머신.

handoff status 로 먼저 갈라지고(needs_clarification → clarify, terminal → refuse), ready 면 ToolBackend 가
계획을 실행한 뒤 **확정 3조건**(모든 claim 에 검증된 citation · blocking limitation 없음)으로 answer /
partial_answer / not_found / refuse 를 정한다. 문장 생성기는 이 payload 밖의 값을 만들 수 없다.
NullToolBackend 는 모든 조회를 닫아 backend 없이도 파이프라인과 채점기가 end-to-end 로 돈다.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Mapping
from typing import Protocol

# QueryPlanHandoff v0.4 계약 (agent.query_plan) — adapter 가 re-export
from .adapter import QueryPlanHandoff  # noqa: F401  (re-export for typing)
from .payload import (
    AnswerPayload,
    Clarification,
    Limitation,
    TraceEvent,
)


class ToolBackend(Protocol):
    """task kind 별 실행기. `execute_plan` 은 PlanExecution(claims · limitations · premise_verdicts · used_documents · clarification)을 돌려준다."""

    def execute_plan(self, plan, *, trace: list[TraceEvent]) -> PlanExecution: ...


class PlanExecution:
    def __init__(self, *, claims=None, limitations=None, premise_verdicts=None,
                 used_documents=None, clarification=None, failed=False):
        self.claims = claims or []
        self.limitations = limitations or []
        self.premise_verdicts = premise_verdicts or []
        self.used_documents = used_documents or []
        self.clarification = clarification
        self.failed = failed
        self.notes: list[str] = []          # composer가 답변에 명시할 해석·정의 고지
        # Execution-only typed sidecars (for example narrative matrix
        # completion).  They never alter the QueryPlan handoff wire.
        self.narrative_sidecars: list[object] = []


class NullToolBackend:
    """구현 전 자리표시자 — 모든 ready plan을 unsupported로 닫는다 (거짓 성공 금지)."""

    def execute_plan(self, plan, *, trace: list[TraceEvent]) -> PlanExecution:
        kinds = sorted({t.kind for t in plan.tasks})
        trace.append(TraceEvent(seq=len(trace) + 1, stage="tool",
                                summary=f"tool backend 미구현 — {kinds} 실행 불가",
                                detail={"kinds": kinds}))
        return PlanExecution(limitations=[Limitation(
            code="tool_not_implemented",
            detail=f"task kinds {kinds}에 대한 Tool이 아직 구현되지 않았습니다.")])


_REFUSAL_STATUS = {"out_of_scope", "unsupported_request", "policy_refusal"}

# QueryPlan v0.4 remains frozen; only answer-compatible, typed limitations may
# cross its internal digest-bound sidecar.
_RUNTIME_LIMITATION_CODES = frozenset({
    "explicit_disclosure_cutoff",
    "non_additive_eps",
    "fx_conversion_not_supported",
    "source_cross_check_partial",
    "counterparty_not_reported_may_hide_match",
    "ambiguous_event_origin",
    "intraday_order_unavailable",
    "source_scope_prevents_complete_lineage",
    "source_scope_raw_absent",
    "future_forecast",
    "corp_not_in_universe",
    "unsupported_operator",
    "unsupported_semantic_target",
    "incomparable_aggregation_scope",
    "personal_data_omitted",
    "holding_lineage_root_missing",
    "unsupported_semantic_target_substituted",
    # #165 — HCX-007 provider 전송 실패(429/5xx/timeout) 분류. 오늘은
    # `server/runtime.py`가 handoff 자체가 없는 실패(`Stage1Resolution.handoff
    # is None`)에서 이 코드를 직접 `error_code`/문구 선택에만 쓰고 이
    # runtime-annotation 인벤토리(ready handoff의 sidecar limitation)로는
    # 흘러들어오지 않는다 — 다만 `_SAFE_MESSAGES`에 문구가 이미 있으므로
    # (app/composer/limitations.py) 화이트리스트에 register 해 둔다.
    "upstream_rate_limited",
    "upstream_unavailable",
})
_RUNTIME_ANNOTATION_KEYS = frozenset({
    "schema_version", "handoff_sha256", "limitations",
    "narrative_row_selectors", "narrative_aggregations", "policy_fallback",
})


def _handoff_sha256(handoff: object) -> str:
    dump = getattr(handoff, "model_dump", None)
    if not callable(dump):
        raise TypeError("runtime annotation에는 typed handoff가 필요합니다")
    payload = json.dumps(
        dump(mode="json", warnings=False), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_runtime_annotations(
        handoff: object, annotations: Mapping | None,
        ) -> None:
    if annotations is None:
        return
    if (not isinstance(annotations, Mapping)
            or set(annotations) - _RUNTIME_ANNOTATION_KEYS
            or annotations.get("schema_version")
            != "stage1-runtime-annotations/1.0"
            or annotations.get("handoff_sha256") != _handoff_sha256(handoff)):
        raise ValueError("runtime annotation handoff/field 결속이 잘못되었습니다")


def _runtime_limitations(
        handoff: object, annotations: Mapping | None,
        ) -> list[Limitation]:
    if annotations is None:
        return []
    _validate_runtime_annotations(handoff, annotations)
    rows = annotations.get("limitations")
    if not isinstance(rows, list) or len(rows) > 32:
        raise ValueError("runtime limitation inventory가 잘못되었습니다")
    out: list[Limitation] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("runtime limitation row가 잘못되었습니다")
        if set(row) != {
                "code", "detail", "applies_to_field_ids",
                "applies_to_whole_target_ids"}:
            raise ValueError("runtime limitation row field가 잘못되었습니다")
        code, detail = row.get("code"), row.get("detail")
        if (code not in _RUNTIME_LIMITATION_CODES
                or not isinstance(detail, str)
                or not detail.strip() or len(detail) > 1_000):
            raise ValueError("runtime limitation code/detail이 허용되지 않습니다")
        key = (code, detail.strip())
        if key not in seen:
            seen.add(key)
            out.append(Limitation(code=code, detail=detail.strip()))
    return out


def _effective_terminal_reasons(
        reasons: list[str], runtime_limitations: list[Limitation],
        ) -> list[str]:
    """Prefer a sealed, specific operator diagnosis over a generic target one."""

    runtime_codes = {row.code for row in runtime_limitations}
    if (("unsupported_operator" in runtime_codes
            or "incomparable_aggregation_scope" in runtime_codes)
            and "unsupported_semantic_target" in reasons):
        return [reason for reason in reasons
                if reason != "unsupported_semantic_target"] or [
                    ("incomparable_aggregation_scope"
                     if "incomparable_aggregation_scope" in runtime_codes
                     else "unsupported_operator")]
    return reasons


#: A sealed fallback that actually produced a cited claim answered the
#: request; the generic ``unsupported_semantic_target`` wording ("...대신
#: 제시하지 않았습니다") would then contradict the answer above it.  Both the
#: typed-reason limitation and any sidecar detail limitation must carry this
#: substitution-specific code instead, and only once success is confirmed by
#: a real cited claim -- never merely because a sealed fallback handoff was
#: built (that alone does not guarantee tool execution finds a citable fact).
_SUBSTITUTED_SEMANTIC_TARGET_CODE = "unsupported_semantic_target_substituted"


def _substitute_semantic_target_success(codes: list[str]) -> list[str]:
    return [
        _SUBSTITUTED_SEMANTIC_TARGET_CODE
        if code == "unsupported_semantic_target" else code
        for code in codes
    ]


def _substitute_semantic_target_limitations(
        limitations: list[Limitation],
        ) -> list[Limitation]:
    return [
        Limitation(code=_SUBSTITUTED_SEMANTIC_TARGET_CODE, detail=row.detail)
        if row.code == "unsupported_semantic_target" else row
        for row in limitations
    ]


def _runtime_narrative_row_selectors(
        handoff: object, annotations: Mapping | None,
        ) -> dict[str, object]:
    """Validate digest-bound, request-local investment row selectors."""

    if annotations is None:
        return {}
    _validate_runtime_annotations(handoff, annotations)
    rows = annotations.get("narrative_row_selectors", [])
    if not isinstance(rows, list) or len(rows) > 32:
        raise ValueError("runtime narrative row selector inventory가 잘못되었습니다")
    plan = getattr(handoff, "plan", None)
    task_by_id = {
        str(getattr(task, "task_id", "")): task
        for task in tuple(getattr(plan, "tasks", ()) or ())
    }
    core_slots = ("투자대상", "목적", "금액", "기간")
    exact_slots = (*core_slots, "기지출금액")
    from app.tools.narrative import InvestmentRowSelector

    selectors: dict[str, object] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("runtime narrative row selector row가 잘못되었습니다")
        keys = set(row)
        exact = keys == {"task_id", "issuer_name", "investment_name"}
        operator = row.get("operation") if not exact else "exact"
        expected_operator_keys = (
            {"task_id", "issuer_name", "operation", "criterion_year"}
            if operator in {"start_year", "end_year_on_or_after"}
            else {"task_id", "issuer_name", "operation"}
        )
        if (not exact and (operator not in {
                "start_year", "end_year_on_or_after",
                "argmax_amount", "missing_period",
                } or keys != expected_operator_keys)):
            raise ValueError("runtime narrative row selector row가 잘못되었습니다")
        task_id = row.get("task_id")
        issuer = row.get("issuer_name")
        investment = row.get("investment_name") if exact else None
        criterion_year = row.get("criterion_year")
        if (not isinstance(task_id, str) or not task_id
                or not isinstance(issuer, str) or not issuer.strip()
                or len(issuer) > 256
                or task_id in selectors):
            raise ValueError("runtime narrative row selector 값이 잘못되었습니다")
        if (exact and (not isinstance(investment, str)
                       or not investment.strip() or len(investment) > 512)):
            raise ValueError("runtime exact narrative selector 값이 잘못되었습니다")
        if (operator in {"start_year", "end_year_on_or_after"}
                and (not isinstance(criterion_year, int)
                     or not 1900 <= criterion_year <= 2200)):
            raise ValueError("runtime narrative year selector가 잘못되었습니다")
        task = task_by_id.get(task_id)
        selector = getattr(task, "document_selector", None)
        requested_slots = tuple(getattr(task, "requested_slots", ()) or ())
        if (task is None or getattr(task, "kind", None) != "narrative"
                or tuple(getattr(task, "corp_names", ()) or ()) != (issuer,)
                or (requested_slots != exact_slots if exact
                    else requested_slots not in {core_slots, exact_slots})
                or not getattr(selector, "rcept_no", None)):
            raise ValueError("runtime narrative row selector task 결속이 잘못되었습니다")
        selectors[task_id] = InvestmentRowSelector(
            task_id=task_id, issuer_name=issuer.strip(),
            investment_name=(investment.strip() if exact else None),
            operation=str(operator),
            criterion_year=(criterion_year if isinstance(criterion_year, int)
                            else None))
    return selectors


def _runtime_narrative_aggregations(
        handoff: object, annotations: Mapping | None,
        ) -> dict[str, object]:
    """Validate digest-bound investment row-sum and revenue-ratio requests."""

    if annotations is None:
        return {}
    _validate_runtime_annotations(handoff, annotations)
    rows = annotations.get("narrative_aggregations", [])
    if not isinstance(rows, list) or len(rows) > 32:
        raise ValueError("runtime narrative aggregation inventory가 잘못되었습니다")
    plan = getattr(handoff, "plan", None)
    task_by_id = {
        str(getattr(task, "task_id", "")): task
        for task in tuple(getattr(plan, "tasks", ()) or ())
    }
    from app.tools.narrative import InvestmentAggregationRequest

    out: dict[str, object] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("runtime narrative aggregation row가 잘못되었습니다")
        base_keys = {
            "task_id", "issuer_name", "operation", "amount_role",
            "aggregate_surface", "expected_row_count",
        }
        if frozenset(row) not in {frozenset(base_keys),
                                  frozenset((*base_keys, "revenue_operand"))}:
            raise ValueError("runtime narrative aggregation field가 잘못되었습니다")
        task_id = row.get("task_id")
        issuer = row.get("issuer_name")
        surface = row.get("aggregate_surface")
        expected = row.get("expected_row_count")
        if (not isinstance(task_id, str) or not task_id
                or task_id in out
                or not isinstance(issuer, str) or not issuer.strip()
                or len(issuer) > 256
                or row.get("operation") != "sum_amount"
                or row.get("amount_role") != "금액"
                or not isinstance(surface, str) or not surface.strip()
                or len(surface) > 80
                or (expected is not None
                    and (not isinstance(expected, int)
                         or isinstance(expected, bool)
                         or not 1 <= expected <= 1000))):
            raise ValueError("runtime narrative aggregation 값이 잘못되었습니다")
        task = task_by_id.get(task_id)
        selector = getattr(task, "document_selector", None)
        if (task is None or getattr(task, "kind", None) != "narrative"
                or tuple(getattr(task, "corp_names", ()) or ()) != (issuer,)
                or tuple(getattr(task, "requested_slots", ()) or ())
                != ("투자대상", "목적", "금액", "기간")
                or not getattr(selector, "rcept_no", None)):
            raise ValueError("runtime narrative aggregation task 결속이 잘못되었습니다")
        revenue = row.get("revenue_operand")
        year = scope = revenue_surface = None
        if revenue is not None:
            if (not isinstance(revenue, Mapping)
                    or set(revenue) != {"year", "scope", "concept", "surface"}
                    or not isinstance(revenue.get("year"), int)
                    or isinstance(revenue.get("year"), bool)
                    or not 1900 <= revenue["year"] <= 2200
                    or revenue.get("scope") not in {"CFS", "SFS"}
                    or revenue.get("concept") != "revenue"
                    or not isinstance(revenue.get("surface"), str)):
                raise ValueError("runtime investment revenue operand가 잘못되었습니다")
            scope_surfaces = (
                ("연결",) if revenue["scope"] == "CFS" else ("별도", "개별"))
            expected_surfaces = {
                _compact_runtime_surface(
                    f"{revenue['year']}년 {scope_surface} 매출액")
                for scope_surface in scope_surfaces
            }
            if (_compact_runtime_surface(revenue["surface"])
                    not in expected_surfaces):
                raise ValueError("runtime investment revenue surface 결속이 잘못되었습니다")
            year, scope, revenue_surface = (
                revenue["year"], revenue["scope"], revenue["surface"])
        out[task_id] = InvestmentAggregationRequest(
            task_id=task_id, issuer_name=issuer.strip(),
            aggregate_surface=surface.strip(),
            expected_row_count=expected,
            revenue_year=year, revenue_scope=scope,
            revenue_surface=revenue_surface,
        )
    return out


def _compact_runtime_surface(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(value or "")).casefold()


def _runtime_policy_fallbacks(
        handoff: object, annotations: Mapping | None,
        ) -> list[tuple[object, Mapping | None]]:
    """Read sealed actual-fact plans attached to a policy terminal.

    The public handoff is intentionally still terminal.  This optional sidecar
    may carry either the original single ready handoff or a bounded bundle of
    two independently executable ready handoffs.  The latter is used when an
    operation is unsafe across semantic scopes but both named operands remain
    independently answerable.  Every nested runtime annotation is still bound
    to its own ready handoff; malformed rows fail closed with a typed error.
    """
    if annotations is None:
        return []
    _validate_runtime_annotations(handoff, annotations)
    row = annotations.get("policy_fallback")
    if row is None:
        return []
    if getattr(handoff, "status", None) not in _REFUSAL_STATUS:
        raise ValueError("policy fallback은 terminal handoff에만 허용됩니다")
    if not isinstance(row, Mapping):
        raise ValueError("policy fallback sidecar 형식이 잘못되었습니다")
    intent_digest = row.get("source_intent_digest")
    question_digest = row.get("source_question_sha256")
    if (not isinstance(intent_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", intent_digest)
            or not isinstance(question_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", question_digest)):
        raise ValueError("policy fallback provenance가 잘못되었습니다")

    if set(row) == {
            "source_intent_digest", "source_question_sha256", "handoff"}:
        raw_candidates = [{"handoff": row.get("handoff") }]
    elif set(row) == {
            "source_intent_digest", "source_question_sha256", "fallbacks"}:
        raw_candidates = row.get("fallbacks")
        if (not isinstance(raw_candidates, list)
                or len(raw_candidates) != 2):
            raise ValueError("policy fallback bundle은 정확히 두 피연산자여야 합니다")
    else:
        raise ValueError("policy fallback sidecar 형식이 잘못되었습니다")

    from .adapter import load_handoff_json
    fallbacks: list[tuple[object, Mapping | None]] = []
    seen_handoffs: set[str] = set()
    for raw in raw_candidates:
        if (not isinstance(raw, Mapping)
                or set(raw) - {"handoff", "runtime_annotations"}
                or "handoff" not in raw
                or not isinstance(raw.get("handoff"), dict)):
            raise ValueError("policy fallback operand 형식이 잘못되었습니다")
        fallback = load_handoff_json(raw["handoff"])
        if fallback.status != "ready" or fallback.plan is None:
            raise ValueError("policy fallback은 ready handoff여야 합니다")
        fallback_digest = _handoff_sha256(fallback)
        if fallback_digest in seen_handoffs:
            raise ValueError("policy fallback operand가 중복되었습니다")
        seen_handoffs.add(fallback_digest)
        nested = raw.get("runtime_annotations")
        if nested is not None:
            _validate_runtime_annotations(fallback, nested)
            if nested.get("policy_fallback") is not None:
                raise ValueError("policy fallback은 중첩할 수 없습니다")
        fallbacks.append((fallback, nested))
    return fallbacks


def _runtime_policy_fallback(handoff: object, annotations: Mapping | None):
    """Backward-compatible single-fallback reader used by older callers."""

    rows = _runtime_policy_fallbacks(handoff, annotations)
    return rows[0][0] if len(rows) == 1 else None


class Orchestrator:
    def __init__(self, backend: ToolBackend | None = None):
        self.backend = backend or NullToolBackend()

    def run(
            self, handoff, *, question_id: str | None = None,
            runtime_annotations: Mapping | None = None,
            ) -> AnswerPayload:
        trace: list[TraceEvent] = []
        qid = question_id or "unknown"
        hid = str(handoff.handoff_id)
        trace.append(TraceEvent(seq=1, stage="handoff",
                                summary=f"handoff 수신 status={handoff.status}",
                                detail={"status": handoff.status, "contract": handoff.contract_version}))

        # ── 1) non-ready: 문장화만 필요 ────────────────────────────────────
        if handoff.status == "needs_clarification":
            c = handoff.clarification
            # v0.4 `question`은 항목 리스트다(경계 보존). 표시는 이 층이 정한다: 1개면 그대로, 여럿이면 번호 목록.
            items = [q.strip() for q in c.question if q.strip()]
            question = items[0] if len(items) == 1 else "\n".join(f"{i}. {q}" for i, q in enumerate(items, 1))
            clar = Clarification(
                clarification_id=c.clarification_id, question=question,
                targets=[s.target for s in c.slots],
                options={s.slot_id: [str(v) for v in s.allowed_values] for s in c.slots},
            )
            trace.append(TraceEvent(seq=2, stage="compose", summary="1단계 역질문 전달",
                                    detail={"targets": clar.targets}))
            return AnswerPayload(question_id=qid, handoff_id=hid, final_status="clarify",
                                 clarification=clar, trace=trace)

        if handoff.status in _REFUSAL_STATUS:
            reasons = list(handoff.reasons)
            fallbacks = _runtime_policy_fallbacks(
                handoff, runtime_annotations)
            if fallbacks:
                # Execute each sealed same-question operand independently.
                # If any bundled operand lacks a cited claim, do not expose a
                # one-sided answer that could invite the forbidden operation.
                fact_payloads = [
                    self.run(fallback, question_id=qid,
                             runtime_annotations=nested)
                    for fallback, nested in fallbacks
                ]
                cited_by_payload = [
                    [claim for claim in payload.claims if claim.citations]
                    for payload in fact_payloads
                ]
                cited_claims = [
                    claim for claims in cited_by_payload for claim in claims]
                if len(fallbacks) > 1 and any(
                        not claims for claims in cited_by_payload):
                    cited_claims = []
                if cited_claims:
                    # The fallback answered the request: a cited claim is
                    # already in hand, so the generic "대신 제시하지 않았습니다"
                    # wording for unsupported_semantic_target would contradict
                    # it.  Substitute the success-specific runtime code
                    # everywhere that code would otherwise surface.
                    effective_reasons = _substitute_semantic_target_success(
                        list(reasons or [handoff.status]))
                    limitations = [
                        row for payload in fact_payloads
                        for row in payload.limitations]
                    seen = {(row.code, row.detail) for row in limitations}
                    fallback_runtime_limitations = (
                        _substitute_semantic_target_limitations(
                            _runtime_limitations(handoff, runtime_annotations)))
                    for row in fallback_runtime_limitations:
                        if (row.code, row.detail) not in seen:
                            limitations.append(row)
                            seen.add((row.code, row.detail))
                    for code in effective_reasons:
                        if _is_code(code) and (code, f"typed reason: {code}") not in seen:
                            limitations.append(Limitation(
                                code=code, detail=f"typed reason: {code}"))
                    trace.append(TraceEvent(
                        seq=2, stage="route",
                        summary="policy terminal의 결속된 사실 fallback 실행",
                        detail={"claims": len(cited_claims),
                                "reasons": effective_reasons,
                                "operands": len(fallbacks)}))
                    for payload in fact_payloads:
                        trace.extend(payload.trace)
                    return AnswerPayload(
                        question_id=qid, handoff_id=hid,
                        plan_revision=max(
                            payload.plan_revision for payload in fact_payloads),
                        final_status="partial_answer", claims=cited_claims,
                        limitations=limitations,
                        premise_verdicts=[
                            row for payload in fact_payloads
                            for row in payload.premise_verdicts],
                        applied_defaults=[
                            row for payload in fact_payloads
                            for row in payload.applied_defaults],
                        used_documents=sorted({
                            row for payload in fact_payloads
                            for row in payload.used_documents}),
                        reasons=effective_reasons, trace=trace)
            terminal_limitations = _runtime_limitations(
                handoff, runtime_annotations)
            reasons = _effective_terminal_reasons(
                reasons, terminal_limitations)
            terminal_seen = {
                (row.code, row.detail) for row in terminal_limitations}
            terminal_limitations.extend(
                Limitation(code=reason, detail=f"typed reason: {reason}")
                for reason in (reasons or [handoff.status])
                if (_is_code(reason)
                    and (reason, f"typed reason: {reason}")
                    not in terminal_seen)
            )
            trace.append(TraceEvent(seq=2, stage="compose",
                                    summary=f"{handoff.status} 한계 고지", detail={"reasons": reasons}))
            return AnswerPayload(question_id=qid, handoff_id=hid, final_status="refuse",
                                 reasons=reasons or [handoff.status],
                                 limitations=terminal_limitations,
                                 trace=trace)

        # ── 2) ready: 실행 ─────────────────────────────────────────────────
        plan = handoff.plan
        trace.append(TraceEvent(seq=2, stage="route",
                                # 어떤 종류의 조회로 내려갔는지가 라우팅 문제의
                                # 첫 단서인데 detail 에만 있어 답변만 보고는
                                # 알 수 없었다. K-047 이 본문 조회가 아니라
                                # 문서 찾기로 가는 것도 이 줄에서 보여야 한다.
                                summary=(f"tasks={len(plan.tasks)}"
                                         f"{'(' + ', '.join(t.kind for t in plan.tasks) + ')' if plan.tasks else ''}"
                                         f" derivations={len(plan.derivations)}"
                                         f" premises={len(plan.premise_claims)}"),
                                detail={"kinds": [t.kind for t in plan.tasks],
                                        "revision": plan.revision}))
        row_selectors = _runtime_narrative_row_selectors(
            handoff, runtime_annotations)
        aggregations = _runtime_narrative_aggregations(
            handoff, runtime_annotations)
        execute_kwargs = {"trace": trace}
        if row_selectors:
            parameters = inspect.signature(
                self.backend.execute_plan).parameters
            if "narrative_row_selectors" not in parameters:
                raise ValueError(
                    "tool backend가 runtime narrative row selector를 지원하지 않습니다")
            execute_kwargs["narrative_row_selectors"] = row_selectors
        if aggregations:
            parameters = inspect.signature(
                self.backend.execute_plan).parameters
            if "narrative_aggregations" not in parameters:
                raise ValueError(
                    "tool backend가 runtime narrative aggregation을 지원하지 않습니다")
            execute_kwargs["narrative_aggregations"] = aggregations
        ex = self.backend.execute_plan(plan, **execute_kwargs)
        sidecar_limitations = _runtime_limitations(
            handoff, runtime_annotations)
        # 이슈 #38 — `server/stage1.py::_extra_runtime_limitation_codes`는
        # concept_ratio 도입 이전에 질문 어휘(「부채비율」+「계산이 안 되면」
        # 등)만으로 unsupported_operator를 미리 붙였다. 실행이 실제로 인용
        # 있는 concept_ratio claim을 만들어냈다면(계산 성공) 「계산하지
        # 않았습니다」라는 모순 문구를 붙이지 않는다. 계산이 실패(_Div·
        # _Unit 등으로 claim 없이 limitation만 남는 경우)했을 때는 그
        # sidecar를 그대로 남긴다.
        if any(claim.operator == "concept_ratio" and claim.citations
               for claim in ex.claims):
            sidecar_limitations = [
                row for row in sidecar_limitations
                if row.code != "unsupported_operator"]
        existing = {(row.code, row.detail) for row in ex.limitations}
        ex.limitations.extend(
            row for row in sidecar_limitations
            if (row.code, row.detail) not in existing)

        if ex.failed:
            return AnswerPayload(question_id=qid, handoff_id=hid, plan_revision=plan.revision,
                                 final_status="failure",
                                 limitations=ex.limitations or [Limitation(code="failure", detail="internal failure")],
                                 trace=trace)
        if ex.clarification is not None:
            return AnswerPayload(question_id=qid, handoff_id=hid, plan_revision=plan.revision,
                                 final_status="clarify", clarification=ex.clarification, trace=trace)

        # ── 3) verify: 확정 3조건 — claim은 전부 citation 보유해야 answer ──
        cited = [c for c in ex.claims if c.citations]
        uncited = [c for c in ex.claims if not c.citations]
        trace.append(TraceEvent(seq=len(trace) + 1, stage="verify",
                                summary=f"cited={len(cited)} uncited={len(uncited)} limitations={len(ex.limitations)}",
                                detail={}))
        if not cited:
            status = "not_found" if any(
                l.code == "not_found" or l.code.startswith("not_found_")
                for l in ex.limitations) else "refuse"
            if any(l.code == "tool_not_implemented" for l in ex.limitations):
                status = "refuse"
            lims = ex.limitations or [Limitation(code="no_verified_claim", detail="검증된 근거가 있는 값을 확정하지 못함")]
            return AnswerPayload(question_id=qid, handoff_id=hid, plan_revision=plan.revision,
                                 final_status=status, limitations=lims,
                                 reasons=[l.code for l in lims],
                                 premise_verdicts=ex.premise_verdicts,
                                 applied_defaults=list(plan.applied_defaults), trace=trace)
        # 정보성 한계(원본 identity 모호 등)는 답변 완전성을 훼손하지 않는다 — 병기만 한다.
        # source_cross_check_partial: PDF/HTML 이중 소스의 교차검증 범위 고지 — 요청 슬롯이
        # 파싱 가능한 source에서 전부 확인되면 답변은 완전하다(claim은 source_roundtrip 검증됨).
        # 실제로 누락된 요청 슬롯만 blocking limitation이 된다(slot_not_confirmed).
        INFORMATIONAL = {
            "ambiguous_event_identity", "ambiguous_event_origin",
            "partial_unread", "source_cross_check_partial",
        }
        blocking = [l for l in ex.limitations if l.code not in INFORMATIONAL]
        # 다중 task에서 보조 task의 not_found(예: 정정 항목 없음)는 주 task 답이 확정됐으면 정보성
        if cited and len(plan.tasks) > 1:
            blocking = [l for l in blocking if l.code != "not_found"]
        final = "partial_answer" if (uncited or blocking) else "answer"
        payload = AnswerPayload(question_id=qid, handoff_id=hid, plan_revision=plan.revision,
                                final_status=final, claims=cited, limitations=ex.limitations,
                                premise_verdicts=ex.premise_verdicts,
                                used_documents=sorted(set(ex.used_documents)),
                                applied_defaults=list(plan.applied_defaults) + list(getattr(ex, "notes", [])),
                                trace=trace)
        payload.set_narrative_sidecars(getattr(ex, "narrative_sidecars", ()))
        return payload


def _is_code(s: str) -> bool:
    import re
    return bool(re.fullmatch(r"[a-z][a-z0-9_]{0,63}", s))
