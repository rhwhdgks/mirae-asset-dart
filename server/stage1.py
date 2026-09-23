"""Stage 1 어댑터 — 질문 문자열 → QueryPlanHandoff v0.4.

세 구현:
- NativeStage1  : Stage1 v1 실경로. HCX-007 Structured Output → SemanticIntent → canonical resolver
                  → deterministic compiler → QueryPlanHandoff 0.4 (`agent/stage1_v1_service.py`).
                  기본 호출은 1회이며 strict structured-output 위반에만 계약 코드로 최대 1회
                  복구한다. 역질문·terminal 결과도 v0.4 handoff로 투영한다.
- FixtureStage1 : 질문 텍스트를 최종 release fixture(`fixtures/query_plan_v04_final/questions_v0.4.jsonl`)와
                  매칭해 그 handoff를 반환(정확 일치 → 정규화 일치). HCX 없이 end-to-end를 돌리는 오프라인 경로.
- ChainedStage1 : native 우선. native 불가(키 없음·HCX 오류·해석 실패) 시 fixture 폴백은 옵션이며
                  어느 경로였는지 `Stage1Resolution.source`와 meta에 남긴다.

서버(Stage2~4)는 `Stage1Resolution.handoff`만 소비한다. HCX 원본 wire·세션 내부 상태·patch path는
Stage2로 넘기지 않는다 (docs/IMPLEMENTATION.md §3).

역질문 재개(`answer_clarification`)는 native 경로에서만 가능하다 — 세션은 SQLite에 있고 HCX를
다시 부르지 않는다 (§4). API 클라이언트에는 `Stage1Resolution.session`(session_id·revision·
clarification_id)과 `clarification`(slot prompt·선택지 label)만 공개한다.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from agent.hcx_schema import safe_validation_issue_paths
from app.orchestrator import load_handoffs

ROOT = Path(__file__).resolve().parents[1]
# Stage1 내부 identifier 규칙(agent/stage1_v1_resolver.py, agent/stage1_v1_clarification_session.py의
# ``Identifier``)과 **반드시 같아야 한다** — 글자로 시작해야 한다. 평가 API의 공개
# question_id 는 숫자로 시작해도 되므로(agent/evaluation_api.py QUESTION_ID_PATTERN) 그 패턴을
# 그대로 재사용하면 안 된다(#110): 숫자 시작 id가 이 검사를 통과해 정규화 없이 그대로
# 내부로 흘러들어가 ValidationError → resolver_authority_failed 로 잘못 분류됐었다.
_QID_OK = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_SAFE_DIAGNOSTIC_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _handoff_sha256(handoff: object) -> str:
    """Bind execution-only metadata to the exact public v0.4 handoff."""

    dump = getattr(handoff, "model_dump", None)
    if not callable(dump):
        raise TypeError("runtime annotation에는 typed handoff가 필요합니다")
    payload = json.dumps(
        dump(mode="json", warnings=False), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_RUNTIME_LIMITATION_DETAIL = {
    "explicit_disclosure_cutoff": "질문에 명시된 공개시점 이후의 보고서 숫자는 대입하지 않습니다.",
    "non_additive_eps": "기간별 가중평균 주식 수가 다른 주당이익은 누적액 차감으로 계산하지 않습니다.",
    "fx_conversion_not_supported": "공시 환율과 기준일을 환산 근거로 확정하는 기능이 없어 임의 환율을 사용하지 않았습니다.",
    "future_forecast": (
        "미래 실적 예측은 수행하지 않고, 질문에 명시된 최신 연간 실제 값만 조회했습니다."),
    "corp_not_in_universe": (
        "비교 대상 일부가 제공된 70개사 범위 밖이어서, 범위 안 회사의 값만 제시하고 비교 결론은 내리지 않았습니다."),
    "unsupported_operator": (
        "질문의 계산 연산은 지원 범위 밖이어서, 명시된 원계정·개별 항목만 제시했습니다."),
    "unsupported_semantic_target": (
        "지원하지 않는 지표 대신 질문에 명시된 지원 가능한 대체 사실만 제시했습니다."),
    "incomparable_aggregation_scope": (
        "투자계획의 프로젝트별 계획금액과 현금흐름표의 기간 실제 집계액은 의미와 "
        "집계 범위가 달라 차감하지 않고, 확인 가능한 원자료만 제시했습니다."),
    "intraday_order_unavailable": (
        "논리적 정정 순서는 확인했지만 코퍼스에 장중 접수시각이 없어 분·초 단위 시각은 제시하지 않았습니다."),
    "source_scope_prevents_complete_lineage": (
        "원공시가 제공 코퍼스 이전에 제출되어 확인되는 정정 이력만 제시하고 최초 내용은 복원하지 않았습니다."),
    "personal_data_omitted": (
        "개인 식별정보와 개인 연락처는 제외하고 공개된 공시 정보만 제시했습니다."),
    # Registered for consistency with `_RUNTIME_LIMITATION_CODES`
    # (app/orchestrator/runner.py). This code is never emitted directly at
    # request-resolution time here — the orchestrator substitutes it in for
    # "unsupported_semantic_target" only after a sealed fallback plan
    # actually produces a cited claim, since that is the only point that
    # knows whether the substitution truly succeeded.
    "unsupported_semantic_target_substituted": (
        "지원하지 않는 지표 대신 질문에 명시된 지원 가능한 대체 사실만 제시했습니다."),
}

_INVESTMENT_ROW_FIELDS = {"총소요자금", "기지출금액"}
_INVESTMENT_TASK_SLOTS = (
    "투자대상", "목적", "금액", "기간", "기지출금액",
)


def _holding_plan_resolves_public_filer_occupation(
        handoff: object, restricted_types: tuple[str, ...],
        ) -> bool:
    """Return whether CRP_TP closed a legal-filer occupation request.

    A bare occupation remains a conservative lexical privacy match.  A single
    holding task carrying ``filer_occupation`` without ``privacy_notice`` is
    different: the holding resolver has already proved that the filer is a
    legal entity.  Other private fields and personal/unknown filers retain the
    generic runtime sidecar.
    """

    if set(restricted_types) != {"occupation"}:
        return False
    plan = getattr(handoff, "plan", None)
    tasks = list(getattr(plan, "tasks", ()) or ())
    if len(tasks) != 1:
        return False
    task = tasks[0]
    selector = getattr(task, "document_selector", None)
    slots = set(getattr(task, "requested_slots", ()) or ())
    return (
        getattr(task, "kind", None) == "disclosure"
        and getattr(selector, "doc_group", None) == "holding"
        and "filer_occupation" in slots
        and "privacy_notice" not in slots
    )

# A policy terminal may contain more than one semantic item.  Only these
# explicit investment-advice surfaces may be discarded while recovering one
# separately answerable financial fact.  This is a vocabulary boundary, not a
# question or company allow-list: another resolvable financial item makes the
# recovery ambiguous and therefore remains fail-closed.
_INVESTMENT_ADVICE_SURFACE = re.compile(
    r"(?:매수|매도)\s*(?:의견|추천)?|목표\s*주가|투자\s*(?:의견|조언|판단)")


def _compact_surface(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(value or "")).casefold()


def _narrative_row_selectors_from_result(
        result: object, handoff: object, *, question: str | None = None,
        ) -> list[dict[str, object]]:
    """Project a closed investment row operation onto a private sidecar."""

    outcome = getattr(result, "final_outcome", None)
    envelope = getattr(outcome, "ready_envelope", None)
    intent = getattr(envelope, "source_intent", None)
    plan = getattr(handoff, "plan", None)
    if intent is None or plan is None or len(intent.answer_items) != 1:
        return []
    item = intent.answer_items[0]
    fields = {_compact_surface(value) for value in item.output.field_surfaces}
    operator: tuple[str, int | None] | None = None
    if isinstance(question, str):
        start = re.search(
            r"(?P<year>(?:19|20)[0-9]{2})년에\s*시작", question)
        ending = re.search(
            r"종료\s*(?:시점|연도)?[^?？。]{0,20}?"
            r"(?P<year>(?:19|20)[0-9]{2})년\s*이후", question)
        if start is not None:
            operator = ("start_year", int(start.group("year")))
        elif ending is not None:
            operator = ("end_year_on_or_after", int(ending.group("year")))
        elif re.search(
                r"계획\s*금액[^?？。]{0,20}(?:가장\s*큰|최대)", question):
            operator = ("argmax_amount", None)
        elif re.search(
                r"기간(?:\s*열)?[이은을는의가]?\s*"
                r"(?:확인되지\s*않|미기재|없는)", question):
            # The missing-ness must attach to ``기간`` itself.  A listing
            # request such as ``대상·목적·금액·기간별로 정리하고, 확인되지
            # 않는 항목은 구분해줘`` names the four columns and then asks for
            # every unconfirmed item, so reading it as a period-only subset
            # replaces the required complete table with one empty note.
            operator = ("missing_period", None)
    if (item.target.kind != "topic"
            or item.operation != "retrieve"
            or item.output.projection_mode != "named_fields"
            or not item.target.surface.strip()
            or len(item.target.entity_refs) != 1):
        return []
    exact = item.output.shape == "record" and fields == _INVESTMENT_ROW_FIELDS
    if not exact and operator is None:
        return []
    by_id = {entity.entity_id: entity for entity in intent.entities}
    issuer = by_id.get(item.target.entity_refs[0])
    if issuer is None or issuer.kind_hint != "company":
        return []
    candidates = [
        task for task in plan.tasks
        if (getattr(task, "kind", None) == "narrative"
            and (tuple(getattr(task, "requested_slots", ()) or ())
                 in {_INVESTMENT_TASK_SLOTS, _INVESTMENT_TASK_SLOTS[:4]})
            and tuple(getattr(task, "corp_names", ()) or ())
            == (issuer.surface,)
            and getattr(getattr(task, "document_selector", None),
                        "rcept_no", None))
    ]
    if len(candidates) != 1:
        return []
    if exact:
        return [{
            "task_id": candidates[0].task_id,
            "issuer_name": issuer.surface,
            "investment_name": item.target.surface,
        }]
    assert operator is not None
    operation, criterion_year = operator
    selector: dict[str, object] = {
        "task_id": candidates[0].task_id,
        "issuer_name": issuer.surface,
        "operation": operation,
    }
    if criterion_year is not None:
        selector["criterion_year"] = criterion_year
    return [selector]


def _narrative_aggregations_from_result(
        result: object, handoff: object, *, question: str | None = None,
        ) -> list[dict[str, object]]:
    """Bind one literal investment-row aggregate to its proved plan task."""

    if not isinstance(question, str):
        return []
    from agent.stage1_v1_narrative_investment import (
        investment_aggregation_request_from_question,
    )
    request = investment_aggregation_request_from_question(question)
    if request is None:
        return []
    outcome = getattr(result, "final_outcome", None)
    envelope = getattr(outcome, "ready_envelope", None)
    intent = getattr(envelope, "source_intent", None)
    plan = getattr(handoff, "plan", None)
    if intent is None or plan is None or len(intent.answer_items) != 1:
        return []
    item = intent.answer_items[0]
    if (item.target.kind != "topic" or item.operation != "retrieve"
            or len(item.target.entity_refs) != 1):
        return []
    by_id = {entity.entity_id: entity for entity in intent.entities}
    issuer = by_id.get(item.target.entity_refs[0])
    if issuer is None or issuer.kind_hint != "company":
        return []
    candidates = [
        task for task in plan.tasks
        if (getattr(task, "kind", None) == "narrative"
            and tuple(getattr(task, "requested_slots", ()) or ())
            == _INVESTMENT_TASK_SLOTS[:4]
            and tuple(getattr(task, "corp_names", ()) or ())
            == (issuer.surface,)
            and getattr(getattr(task, "document_selector", None),
                        "rcept_no", None))
    ]
    if len(candidates) != 1:
        return []
    return [{
        "task_id": candidates[0].task_id,
        "issuer_name": issuer.surface,
        **request,
    }]


def _runtime_annotations_from_result(
        result: object, handoff: object, *, extra_codes: tuple[str, ...] = (),
        policy_fallback: dict | None = None, question: str | None = None,
        ) -> dict | None:
    """Project bounded execution annotations into a sealed sidecar."""

    outcome = getattr(result, "final_outcome", None)
    envelope = getattr(outcome, "ready_envelope", None)
    contract = getattr(envelope, "answer_contract", None)
    # Ready outcomes carry answer-contract limitations; terminal outcomes do
    # not.  A sealed policy fallback is the one intentional terminal-side
    # annotation, so do not drop it merely because no ReadyEnvelope exists.
    if contract is None and policy_fallback is None and not extra_codes:
        return None
    limitations: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in getattr(contract, "items", ()):
        for binding in getattr(item, "limitation_bindings", ()):
            code = getattr(binding, "code", None)
            detail = getattr(binding, "detail", None)
            if (not isinstance(code, str)
                    or _SAFE_DIAGNOSTIC_CODE.fullmatch(code) is None
                    or not isinstance(detail, str) or not detail.strip()):
                raise ValueError("compiled limitation sidecar가 잘못되었습니다")
            key = (code, detail.strip())
            if key in seen:
                continue
            seen.add(key)
            limitations.append({
                "code": code,
                "detail": detail.strip(),
                "applies_to_field_ids": list(
                    getattr(binding, "applies_to_field_ids", ())),
                "applies_to_whole_target_ids": list(
                    getattr(binding, "applies_to_whole_target_ids", ())),
            })
    for code in extra_codes:
        detail = _RUNTIME_LIMITATION_DETAIL.get(code)
        if detail is None:
            raise ValueError("등록되지 않은 runtime limitation입니다")
        key = (code, detail)
        if key in seen:
            continue
        seen.add(key)
        limitations.append({
            "code": code, "detail": detail,
            "applies_to_field_ids": [],
            "applies_to_whole_target_ids": [],
        })
    narrative_row_selectors = _narrative_row_selectors_from_result(
        result, handoff, question=question)
    narrative_aggregations = _narrative_aggregations_from_result(
        result, handoff, question=question)
    if isinstance(question, str):
        from agent.stage1_v1_narrative_investment import (
            investment_aggregation_request_from_question,
        )
        aggregation_recognized = (
            investment_aggregation_request_from_question(question) is not None)
    else:
        aggregation_recognized = False
    if aggregation_recognized and not narrative_aggregations:
        code = "unsupported_operator"
        detail = _RUNTIME_LIMITATION_DETAIL[code]
        key = (code, detail)
        if key not in seen:
            seen.add(key)
            limitations.append({
                "code": code, "detail": detail,
                "applies_to_field_ids": [],
                "applies_to_whole_target_ids": [],
            })
    if (not limitations and not narrative_row_selectors
            and not narrative_aggregations and policy_fallback is None):
        return None
    payload = {
        "schema_version": "stage1-runtime-annotations/1.0",
        "handoff_sha256": _handoff_sha256(handoff),
        "limitations": limitations,
        "narrative_row_selectors": narrative_row_selectors,
    }
    if narrative_aggregations:
        payload["narrative_aggregations"] = narrative_aggregations
    if policy_fallback is not None:
        payload["policy_fallback"] = policy_fallback
    return payload


# v1 clarification role → 공개 v0.4 slot target (agent/stage1_v1_query_plan_v04_emitter.py 와 같은 대응)
_ROLE_TO_TARGET = {
    "entity": "company", "event": "event", "time": "period", "timepoint": "timepoint",
    "qualifier": "scope", "selection": "selection", "value_kind": "value_kind",
}


def _norm(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", (s or "").lower())


def safe_question_id(question_id: str) -> str:
    """평가 API의 question_id를 Stage1 내부 식별자 규칙(글자로 시작)에 맞춘다 (#110).

    원본이 이미 내부 규칙에 맞으면 그대로 쓴다. 아니면 결정적으로 변환한다: 허용 문자
    (``[A-Za-z0-9_.:-]``) 밖은 ``_`` 로 치환하고, 첫 글자가 글자가 아니면(숫자로 시작하는
    ``26a``·``1`` 같은 평가 API id 포함) ``q_`` 를 접두하고, 128자를 넘으면 앞 100자 +
    ``_`` + 원본 sha256 앞 8자로 줄인다. 같은 입력은 항상 같은 내부 id 로 정규화된다
    (결정적) — 응답에 돌려주는 ``question_id`` 는 이 함수와 무관하게 항상 원문 그대로다;
    이 함수는 Stage1 내부 호출에만 쓴다."""
    original = question_id or ""
    if _QID_OK.fullmatch(original):
        return original
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]", "_", original)
    if not re.match(r"^[A-Za-z]", cleaned):
        cleaned = "q_" + cleaned
    if len(cleaned) > 128:
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
        cleaned = cleaned[:100] + "_" + digest
    if _QID_OK.fullmatch(cleaned):
        return cleaned
    return "q_" + hashlib.sha256(original.encode("utf-8")).hexdigest()[:16]


class Stage1Unavailable(RuntimeError):
    """요청한 Stage1 경로를 쓸 수 없다 (키 없음·조립 실패·fixture 경로에는 세션이 없음)."""


def _validation_error_targets_question_id(error: ValidationError) -> bool:
    """``question_id`` 필드 자체를 겨눈 pydantic issue 인지(자유문 값은 보지 않는다)."""
    return any(
        issue.get("loc") == ("question_id",)
        for issue in error.errors(include_input=False, include_context=False, include_url=False))


def safe_exception_diagnostics(exc: BaseException) -> dict:
    """Project one exception chain into bounded, content-free diagnostics.

    Diagnostic codes and schema paths are already designed as safe metadata by
    the semantic boundary.  Exception messages, provider payloads, questions,
    and arbitrary exception attributes are deliberately excluded.
    """

    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(chain) < 4:
        chain.append(current)
        next_error = current.__cause__ or current.__context__
        current = next_error if isinstance(next_error, BaseException) else None
    types = [type(error).__name__[:96] for error in chain]
    validation_error = next(
        (error for error in chain if isinstance(error, ValidationError)), None)

    if any(name == "HcxSemanticIntentNormalizationInvocationError"
           for name in types):
        layer, code = "normalization", "grounding_rejected"
    elif any(name == "Stage1V1CompilerTechnicalError" for name in types):
        layer, code = "compiler", "compiler_binding_failed"
    elif any(name == "Stage1V1ResolverTechnicalError" for name in types):
        layer, code = "resolver", "resolver_authority_failed"
        if validation_error is not None and _validation_error_targets_question_id(validation_error):
            # 서버 경계 정규화(safe_question_id)를 뚫고 들어온 question_id 형식 오류에 대한
            # 안전망(#110). "해석 실패"로 뭉뚱그리면 원인(question_id 형식)을 알 수 없다.
            code = "invalid_question_id"
    elif any(name == "HcxRateLimitError" for name in types):
        # #165: 전송 실패(429)를 "질문을 해석하지 못함"(resolver_authority_failed/
        # provider_invocation_failed)과 구분한다 — 원인이 질문이 아니라 provider
        # rate limit이라는 것을 클라이언트가 알 수 있어야 재시도가 의미 있다.
        layer, code = "provider", "upstream_rate_limited"
    elif any(name in {"HcxTransientError", "HcxTimeoutError", "HcxProviderError"} for name in types):
        # 5xx·timeout·connect 오류(HcxTimeoutError는 HcxTransientError의 subclass).
        # 마찬가지로 질문 탓이 아니므로 별도 코드로 분류한다.
        layer, code = "provider", "upstream_unavailable"
    elif any(name.startswith("Hcx") for name in types):
        layer, code = "provider", "provider_invocation_failed"
    else:
        layer, code = "stage1", "stage1_exception"

    def safe_codes(attribute: str) -> list[str]:
        values: list[str] = []
        for error in chain:
            candidate = getattr(error, attribute, ())
            if not isinstance(candidate, (tuple, list)):
                continue
            for value in candidate:
                if (isinstance(value, str)
                        and _SAFE_DIAGNOSTIC_CODE.fullmatch(value)
                        and value not in values):
                    values.append(value)
                    if len(values) == 32:
                        return values
        return values

    def safe_paths() -> list[str]:
        values: list[str] = []
        for error in chain:
            candidate = getattr(error, "diagnostic_paths", ())
            if not isinstance(candidate, (tuple, list)):
                continue
            for value in candidate:
                if (isinstance(value, str) and 0 < len(value) <= 160
                        and re.fullmatch(r"[A-Za-z0-9_.\[\]-]+", value)
                        and value not in values):
                    values.append(value)
                    if len(values) == 32:
                        return values
        return values

    result: dict = {
        "layer": layer,
        "code": code,
        "exception_types": types,
    }
    for attribute in (
            "diagnostic_codes", "normalization_codes", "schema_repair_codes"):
        values = safe_codes(attribute)
        if values:
            result[attribute] = values
    paths = safe_paths()
    if paths:
        result["diagnostic_paths"] = paths

    # A ValidationError that reaches `Stage1V1CompilerTechnicalError` (or any
    # other technical failure) as its immediate cause is not itself one of
    # the typed HCX errors above and carries no `diagnostic_paths` attribute.
    # Its own `loc` paths are schema field names/indices only — no value,
    # input, or message text — so they are exactly as safe to surface.
    if validation_error is not None:
        # Present even when empty: a ValidationError raised by a model-level
        # @model_validator(mode="after") carries no field-level loc at all, so
        # omitting the key made a real validation failure indistinguishable
        # from "no ValidationError was found" (P9-016, issue #43).
        result["validation_locs"] = list(
            safe_validation_issue_paths(validation_error))

    for attribute in ("request_id",):
        value = next((getattr(error, attribute, None) for error in chain
                      if getattr(error, attribute, None) is not None), None)
        if isinstance(value, str) and _SAFE_REQUEST_ID.fullmatch(value):
            result[attribute] = value
    for attribute in (
            "attempts", "prompt_tokens", "completion_tokens", "total_tokens"):
        value = next((getattr(error, attribute, None) for error in chain
                      if getattr(error, attribute, None) is not None), None)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10**9:
            result[attribute] = value
    # #165: 전송 재시도(agent/hcx_semantic_intent_v1.py)가 최종 실패 예외에 남기는
    # 사후 분류용 필드. `http_status`는 HCX가 실제로 돌려준 HTTP 상태(429만 정확히
    # 안다 — 5xx/timeout은 정확한 코드가 provider client 경계 밖으로 나오지 않아
    # 남기지 않는다), `retry_count`/`retry_wait_s`는 재시도 루프가 실제로 쓴 횟수·누적
    # backoff 초(sleep 시간만, 네트워크 왕복 시간 제외)다.
    for attribute in ("http_status", "retry_count"):
        value = next((getattr(error, attribute, None) for error in chain
                      if getattr(error, attribute, None) is not None), None)
        lower = 100 if attribute == "http_status" else 0
        if isinstance(value, int) and not isinstance(value, bool) and lower <= value <= 599:
            result[attribute] = value
    retry_wait = next((getattr(error, "retry_wait_s", None) for error in chain
                       if getattr(error, "retry_wait_s", None) is not None), None)
    if (isinstance(retry_wait, (int, float)) and not isinstance(retry_wait, bool)
            and 0 <= float(retry_wait) <= 3_600):
        result["retry_wait_s"] = float(retry_wait)
    latency = next((getattr(error, "total_latency_ms", None) for error in chain
                    if getattr(error, "total_latency_ms", None) is not None), None)
    if (isinstance(latency, (int, float)) and not isinstance(latency, bool)
            and 0 <= float(latency) <= 86_400_000):
        result["total_latency_ms"] = float(latency)
    return result


def _safe_error_label(exc: BaseException) -> str:
    """Legacy ``errors`` 필드에도 예외 메시지나 provider 본문을 넣지 않는다."""
    diagnostic = safe_exception_diagnostics(exc)
    return f"{type(exc).__name__}:{diagnostic['code']}"


@dataclass
class Stage1Resolution:
    handoff: object | None                   # QueryPlanHandoff v0.4 — None이면 해석 불가
    source: str                              # "stage1_v1_native" | "fixture" | "none"
    question_id_hint: str | None = None      # fixture 매칭 시 문항 ID (표시·로그용)
    meta: dict = field(default_factory=dict)  # request_id·토큰·지연·오류 (로그·think_trace 재료)
    # 열린 역질문 세션의 공개 식별자 — 클라이언트가 답을 보내 재개할 때 그대로 되돌려 준다.
    #   {"session_id", "revision", "status", "clarification_id"?}
    session: dict | None = None
    # PublicClarification 투영(slot_id·role_hint·prompt·options[value,label,reason]) — API 표시용.
    # v0.4 handoff의 clarification은 allowed_values만 갖고 label·prompt를 잃으므로 따로 싣는다.
    clarification: dict | None = None


@dataclass
class Stage1ResumeResolution:
    """One resumed Stage1 turn plus the server-owned original request surface.

    The original question is read only inside the durable clarification store so
    that Stage2~4 can preserve its normal response envelope.  It is not added
    to the public clarification object or Stage1 metadata.
    """

    resolution: Stage1Resolution
    question_id: str
    question: str


class Stage1(Protocol):
    name: str
    available: bool

    def resolve(self, question: str, *, question_id: str,
                deadline_monotonic: float | None = None) -> Stage1Resolution: ...


# ── fixture 경로 ──────────────────────────────────────────────────────────────

class FixtureStage1:
    name = "fixture"
    available = True

    def __init__(self):
        self._exact: dict[str, object] = {}
        self._norm: dict[str, object] = {}
        for r in load_handoffs():
            if r.question:
                self._exact[r.question] = r
                self._norm[_norm(r.question)] = r

    def resolve(self, question: str, *, question_id: str,
                deadline_monotonic: float | None = None) -> Stage1Resolution:
        r = self._exact.get(question) or self._norm.get(_norm(question))
        if r is None:
            return Stage1Resolution(None, self.name, meta={"error": "no_fixture_match"})
        return Stage1Resolution(r.handoff, self.name, question_id_hint=r.question_id,
                                meta={"handoff_status": r.handoff.status})


# ── native 경로 (팀원 Stage1 v1) ───────────────────────────────────────────────

def handoff_from_clarification_view(view) -> object:
    """열린 역질문 세션(ClarificationSessionView)을 공개 v0.4 needs_clarification handoff로 투영한다.

    session_id·revision·내부 상태는 넘기지 않는다. slot의 prompt는 항목 리스트(`question`)로,
    선택지 value는 `allowed_values`로만 나간다 (emitter의 `_clarification_handoff`와 같은 규칙).
    """
    from agent.query_plan import HandoffClarification, HandoffClarificationSlot, QueryPlanHandoff

    public = view.clarification
    if public is None:
        raise ValueError("waiting view에 clarification이 없습니다")
    slots = tuple(
        HandoffClarificationSlot(
            slot_id=slot.slot_id,
            target=_ROLE_TO_TARGET.get(slot.role_hint, slot.role_hint),
            allowed_values=tuple(option.value for option in slot.options),
        )
        for slot in public.slots
    )
    question = tuple(dict.fromkeys(
        slot.prompt.strip() for slot in public.slots if slot.prompt.strip()))
    return QueryPlanHandoff(
        handoff_id=str(uuid5(NAMESPACE_URL,
                             f"stage1-v1-clarification-view:{view.session_id}:{view.revision}")),
        status="needs_clarification",
        clarification=HandoffClarification(
            clarification_id=public.clarification_id,
            plan_revision=public.plan_revision,
            question=question,
            slots=slots,
        ),
    )


def public_clarification_resume(view) -> dict:
    """Return the minimum public resume authority; never serialize session state.

    ``session_id`` and ``clarification_id`` are opaque UUID capabilities.  The
    client needs only the current revision and the visible slot prompt/options;
    resolver roles, source mentions, digests, and plan paths stay server-side.
    """

    public = view.clarification
    if public is None:
        raise ValueError("waiting view에 clarification이 없습니다")
    return {
        "session_id": view.session_id,
        "clarification_id": public.clarification_id,
        "revision": view.revision,
        "slots": [
            {
                "slot_id": slot.slot_id,
                "prompt": slot.prompt,
                "response_kind": slot.response_kind,
                "options": [
                    {"value": option.value, "label": option.label}
                    for option in slot.options
                ],
            }
            for slot in public.slots
        ],
    }


class NativeStage1:
    """`Stage1V1NativeService(runner, runtime)` 조립.

    - runner : `HcxSemanticIntentRunner` — 승인된 prompt manifest(hcx007_prompt_v1_review) + HCX-007 client.
               회사 표기는 canonical registry(`CanonicalSelectorRolePreflight.question_company_surface`)로 되돌린다.
    - runtime: `build_stage1_v1_runtime(canonical)` — resolver·compiler·역질문 세션(SQLite)·resume backend.
    canonical(CanonicalReadModel)은 Stage2 tool과 **같은 객체**를 주입한다 — 서버는 정본 접근을 단일 스레드로 직렬화한다.
    """
    name = "stage1_v1_native"

    def __init__(self, canonical, *, clarification_db_path: Path | None = None,
                 selector_cache_root: Path | None = None):
        self._canonical = canonical
        self.available = False
        self.error: str | None = None
        self.error_diagnostic: dict | None = None
        self.prompt_version: str | None = None
        self.service = None
        self._client = None
        try:
            self._assemble(canonical, clarification_db_path, selector_cache_root)
            self.available = True
        except Exception as e:  # 키 없음·prompt 미승인·조립 실패 → 서버는 fixture 폴백 또는 한계 고지
            self.error = _safe_error_label(e)
            self.error_diagnostic = safe_exception_diagnostics(e)

    def _assemble(self, canonical, clarification_db_path, selector_cache_root):
        from agent.hcx_semantic_intent_v1 import (
            HcxSemanticIntentRunner,
            load_hcx_semantic_intent_prompt,
        )
        from agent.planner_preflight import CanonicalSelectorRolePreflight
        from agent.providers.hcx007 import HcxStructuredClient
        from agent.stage1_assembly import CORPUS_CUTOFF, REFERENCE_DATE
        from agent.stage1_v1_backend_composition import build_stage1_v1_runtime
        from agent.stage1_v1_service import (
            Stage1SemanticIntentCache,
            Stage1V1NativeService,
        )

        prompt = load_hcx_semantic_intent_prompt()      # hash 결속 + 승인 상태 검증
        prompt.require_live_approval()
        self.prompt_version = prompt.prompt_version
        self._client = HcxStructuredClient.from_env()   # CLOVASTUDIO_API_KEY
        selector = CanonicalSelectorRolePreflight(
            canonical, corpus_cutoff=CORPUS_CUTOFF,
            cache_root=selector_cache_root or ROOT / "out" / "serving" / "selector_roles")
        runner = HcxSemanticIntentRunner(
            self._client, prompt=prompt,
            company_surface_regrounder=selector.question_company_surface,
            allow_schema_retry=True)
        runtime = build_stage1_v1_runtime(
            canonical, reference_date=REFERENCE_DATE, corpus_cutoff=CORPUS_CUTOFF,
            clarification_db_path=(clarification_db_path
                                   or ROOT / "out" / "serving" / "stage1_v1_clarifications.sqlite3"))
        self.service = Stage1V1NativeService(
            runner, runtime,
            semantic_cache=Stage1SemanticIntentCache(
                canonical_build_id=canonical.build_id,
                system_prompt_sha256=prompt.sha256,
                provider_schema_sha256=runner.compiled_schema.sha256,
                generation_config_sha256=runner.generation_config.fingerprint,
                as_of=CORPUS_CUTOFF,
            ),
        )

    def resolve(self, question: str, *, question_id: str,
                deadline_monotonic: float | None = None) -> Stage1Resolution:
        if not self.available:
            meta = {"error": self.error or "unavailable"}
            if self.error_diagnostic is not None:
                meta["error_diagnostic"] = self.error_diagnostic
            return Stage1Resolution(None, self.name, meta=meta)
        t0 = time.monotonic()
        qid = safe_question_id(question_id)
        gate_code = self._abusive_offtopic_gate(question)
        if gate_code is not None:
            result = self.service.start_from_intent(
                qid, question, self._abusive_offtopic_seed_intent(question))
            return self._resolution(
                result, t0, stage1_question_id=qid, question=question)
        result = self.service.start(qid, question, deadline_monotonic=deadline_monotonic)
        recovered = self._recover_held_company_clarification(
            question_id=qid, question=question, result=result)
        if recovered is not None:
            result = recovered
        return self._resolution(
            result, t0, stage1_question_id=qid, question=question)

    def _abusive_offtopic_gate(self, question: str) -> str | None:
        """욕설만/공시와 무관한 입력은 HCX 호출 전에 결정적으로 닫는다.

        회사명도 공시 어휘도 전혀 없으면 HCX-007을 부르지 않는다 — 비용을
        아끼고, 이런 입력이 실제로 종종 만드는 「해석 실패」(handoff None)
        총칭 경로를 우회한다(예: 「오늘 서울 날씨 어때? 그리고 김치찌개
        레시피 알려줘」— HCX가 intent를 못 냄). 판정 함수는 정책 백엔드가
        HCX 성공 후 같은 입력에 대해 도달할 결정과 동일한 것 하나뿐이다
        (`agent.stage1_v1_policy_backend.classify_abusive_or_off_topic`) —
        여기서 다시 구현하지 않는다.
        """

        if not isinstance(question, str) or not question.strip():
            return None
        from agent.stage1_v1_policy_backend import classify_abusive_or_off_topic
        return classify_abusive_or_off_topic(
            question, company_resolver=self._canonical)

    @staticmethod
    def _abusive_offtopic_seed_intent(question: str):
        """A minimal, entity-free ``SemanticIntent`` replayed through the
        normal native runtime (``start_from_intent``) without invoking HCX.

        The runtime's ``GeneralPolicyResolutionBackend`` independently
        re-derives the very same ``abusive_input``/``off_topic_request``
        terminal from this seed (empty entities, no disclosure vocabulary
        either — that is exactly why the gate fired), so the emitted
        QueryPlanHandoff has the same shape a real HCX-parsed terminal would
        have. No separate handoff-building code path is needed.
        """

        from agent.semantic_intent_v1 import SemanticIntent

        return SemanticIntent.model_validate({
            "schema_version": "stage1-semantic-intent/1.1",
            "entities": [],
            "answer_items": [{
                "item_id": "item-1", "operation": "retrieve",
                "target": {
                    "kind": "topic", "surface": question,
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
                    "shape": "scalar", "projection_mode": "whole_target",
                    "field_surfaces": [], "presentation": "auto",
                },
            }],
            "answer_groups": [], "premises": [],
            "unresolved_mentions": [], "presentation": "auto",
        }, strict=True)

    def _recover_held_company_clarification(
            self, *, question_id: str, question: str, result: object,
            ):
        """Replay one closed financial question when HCX drops a held issuer.

        The ordinary question-grounded financial regrounder already owns the
        literal company/year/account grammar and deliberately preserves a
        registry-held alias as an unresolved company entity.  A provider can
        nevertheless return a terminal document/policy shape before that
        regrounder gets a usable metric item.  In that narrow case, replay a
        coordinate-poor metric intent through the same regrounder and normal
        runtime so ``HeldCompanyClarificationBackend`` creates the session.

        This does not infer an issuer from corpus contents: recovery is
        admitted only when the question literally contains one held alias
        with at least two canonical candidates.  Questions with no explicit
        ambiguous alias therefore keep the existing unique-issuer inference
        policy.
        """

        handoff = getattr(result, "handoff", None)
        if (getattr(handoff, "status", None)
                not in {"policy_refusal", "unsupported_request"}
                or not isinstance(question, str)):
            return None
        from agent.semantic_intent_v1 import SemanticIntent
        from agent.stage1_v1_backend_composition import (
            QuestionGroundedFinancialRegrounder,
        )

        seed = SemanticIntent.model_validate({
            "schema_version": "stage1-semantic-intent/1.1",
            "entities": [],
            "answer_items": [{
                "item_id": "item-1", "operation": "retrieve",
                "target": {
                    "kind": "metric", "surface": question,
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
                    "shape": "scalar", "projection_mode": "named_fields",
                    "field_surfaces": [question], "presentation": "auto",
                },
            }],
            "answer_groups": [], "premises": [],
            "unresolved_mentions": [], "presentation": "auto",
        }, strict=True)
        recovered = QuestionGroundedFinancialRegrounder(
            self._canonical)(question, seed)
        referenced = {
            ref for item in recovered.answer_items
            for ref in item.target.entity_refs
        }
        held_entities = [
            entity for entity in recovered.entities
            if (entity.entity_id in referenced
                and entity.kind_hint == "company"
                and entity.surface in question
                and not self._canonical.resolve_company(entity.surface)
                and len({
                    row.corp_code for row in
                    (self._canonical.held_company_candidates(entity.surface)
                     or ())
                }) >= 2)
        ]
        if len(held_entities) != 1:
            return None
        replay = self.service.start_from_intent(
            question_id, question, recovered,
            provider_metadata=getattr(result, "provider_metadata", None))
        if getattr(replay, "status", None) != "needs_clarification":
            return None
        return replay

    def answer_clarification(self, answer: Mapping[str, Any]) -> Stage1Resolution:
        """열린 역질문 세션을 재개한다 — HCX 재호출 없이 canonical resolver만 다시 돈다 (§4).

        `answer`는 `Stage1V1ClarificationAnswer` 모양(session_id·clarification_id·expected_revision·
        action·values). revision 불일치·후보 밖 값·turn 한도는 runtime이 typed 예외로 fail-closed하며
        세션은 변경되지 않는다. 호출자가 예외를 HTTP 코드로 옮긴다.
        """
        if not self.available:
            raise Stage1Unavailable(self.error or "native stage1 unavailable")
        t0 = time.monotonic()
        session_id = answer.get("session_id") if isinstance(answer, Mapping) else None
        state = (self.service.runtime.clarification_coordinator.store.get(session_id)
                 if isinstance(session_id, str) else None)
        result = self.service.answer_clarification(answer)
        return self._resolution(
            result, t0,
            question=getattr(state, "original_question", None))

    def inspect_session(self, session_id: str) -> Stage1Resolution:
        """세션의 현재 공개 상태(대기 중 역질문 또는 결과)를 handoff로 투영해 돌려준다."""
        if not self.available:
            raise Stage1Unavailable(self.error or "native stage1 unavailable")
        view = self.service.runtime.inspect_clarification(session_id)
        return self._resolution_from_view(view, meta={"status": view.status})

    def _resolution(
            self, result, t0: float, *, stage1_question_id: str | None = None,
            question: str | None = None,
            ) -> Stage1Resolution:
        meta: dict = {"status": result.status, "t_stage1": round(time.monotonic() - t0, 3)}
        if stage1_question_id is not None:
            meta["stage1_question_id"] = stage1_question_id
        inv = result.provider_metadata
        for k in ("request_id", "attempts", "total_latency_ms", "prompt_tokens",
                  "completion_tokens", "total_tokens", "semantic_intent_digest",
                  "fallback_code", "cache_hit"):
            v = getattr(inv, k, None)
            if v is not None:
                meta[k] = v
        semantic_intent = getattr(inv, "semantic_intent", None)
        if semantic_intent is not None:
            # Provider wire/free text and answer values are never logged.  The
            # bounded, normalized question surfaces below are enough to tell a
            # supported intent from an occasional over-split/incorrect target
            # without exposing anything not already present in the request.
            meta["semantic_intent_shape"] = [
                {
                    "item_id": item.item_id,
                    "target_kind": item.target.kind,
                    "target_surface": item.target.surface,
                    "entity_refs": list(item.target.entity_refs),
                    "operation": item.operation,
                    "periods": list(item.scope.target_period_expressions),
                    "as_of": item.scope.as_of_expression,
                    "document_group": item.scope.document_group_expression,
                    "output_shape": item.output.shape,
                    "projection_mode": item.output.projection_mode,
                    "field_surfaces": list(item.output.field_surfaces),
                }
                for item in semantic_intent.answer_items
            ]
        return self._resolution_from_result(
            result, meta=meta, question=question,
            semantic_intent=semantic_intent)

    def _extra_runtime_limitation_codes(
            self, question: str | None, semantic_intent: object | None,
            handoff: object,
            ) -> tuple[str, ...]:
        if not isinstance(question, str):
            return ()
        codes: list[str] = []
        from agent.financial_request_boundaries import (
            eps_cumulative_subtraction, explicit_krw_fx_fallback,
        )
        if eps_cumulative_subtraction(question):
            codes.append("non_additive_eps")
        if explicit_krw_fx_fallback(question):
            codes.append("fx_conversion_not_supported")
        dated = re.search(r"(20\d{2})년\s*(\d{1,2})월\s*(\d{1,2})일까지", question)
        plan = getattr(handoff, "plan", None)
        tasks = getattr(plan, "tasks", ())
        if dated is not None and tasks:
            cutoff = "".join(f"{int(value):0{4 if index == 0 else 2}d}"
                             for index, value in enumerate(dated.groups()))
            if all(getattr(task, "kind", None) == "financial"
                   and getattr(task, "as_of", None) == cutoff for task in tasks):
                codes.append("explicit_disclosure_cutoff")
        if (re.search(r"예측|예상|전망", question)
                and re.search(r"안\s*되면|대신|불가능.{0,12}(?:실제|과거)", question)
                and re.search(r"최신\s*연간\s*실제|최신\s*실제|과거\s*실적", question)):
            codes.append("future_forecast")
        entities = getattr(semantic_intent, "entities", ())
        # ``kind_hint=company`` alone does not make an entity a comparison
        # issuer.  Contract counterparties such as Ford are companies too,
        # and counting them here produced a false ``corp_not_in_universe``
        # notice on an in-universe issuer's event query.  Only the entity axes
        # of a typed comparison/verdict can justify a cross-company boundary.
        comparison_refs = {
            ref
            for item in getattr(semantic_intent, "answer_items", ())
            if (getattr(item, "operation", None) == "compare"
                or getattr(getattr(item, "output", None), "shape", None)
                in {"comparison", "verdict"})
            for ref in getattr(getattr(item, "target", None), "entity_refs", ())
        }
        company_surfaces = [
            entity.surface for entity in entities
            if (getattr(entity, "kind_hint", None) == "company"
                and getattr(entity, "entity_id", None) in comparison_refs)
        ]
        if len(company_surfaces) >= 2:
            present = 0
            absent = 0
            for surface in company_surfaces:
                rows = self._canonical.resolve_company(surface)
                held = self._canonical.held_company_candidates(surface)
                if len(rows) == 1:
                    present += 1
                elif not rows and not held:
                    absent += 1
            if present >= 1 and absent >= 1:
                codes.append("corp_not_in_universe")
        if (re.search(r"접수\s*시각|접수시각", question)
                and re.search(r"(?:분|초)\s*단위", question)):
            codes.append("intraday_order_unavailable")
        if (re.search(r"부채\s*비율", question)
                and re.search(r"부채\s*총계", question)
                and re.search(r"자본\s*총계", question)
                and re.search(r"계산이?\s*안\s*되면|대신", question)):
            codes.append("unsupported_operator")
        if (re.search(r"판매\s*량|몇\s*대\s*팔", question)
                and re.search(r"매출\s*액", question)
                and re.search(r"없으면|안\s*되면|대신", question)):
            codes.append("unsupported_semantic_target")
        if (re.search(r"투자\s*계획", question)
                and re.search(r"(?:연결|별도|개별)?\s*현금\s*흐름표", question)
                and re.search(r"유형\s*자산\s*취득", question)
                and re.search(r"차이|차감|빼", question)):
            codes.append("incomparable_aggregation_scope")
        if (re.search(r"코퍼스\s*(?:이전|밖)", question)
                and re.search(r"원\s*공시|원공시|최초", question)):
            codes.append("source_scope_prevents_complete_lineage")
        # 공개 사실과 개인정보를 함께 물으면 공개 쪽만 답한다.  무엇을 빼고
        # 답했는지는 말해야 하는데, 지금까지 이 고지는 지분공시 경로(`app/
        # tools/holding.py`)에서만 붙었다.  「매출액은 알려주고 주민등록번호는
        # 제외해줘」(`RPC-016`)처럼 재무 사실로 답하면 아무 말 없이 빠졌다.
        # 어느 백엔드가 답하든 질문에 달린 사실이므로 여기서 본다.
        from src.canonical.security import classify_privacy_request
        privacy_request = classify_privacy_request(question)
        if (privacy_request.mode == "partial"
                and not _holding_plan_resolves_public_filer_occupation(
                    handoff, privacy_request.restricted_types)):
            codes.append("personal_data_omitted")
        return tuple(dict.fromkeys(codes))

    def _policy_fallback_handoff(
            self, *, question_id: str, question: str | None,
            source_intent: object | None, terminal_handoff: object,
            ) -> dict | None:
        """Build a sealed actual-fact sidecar for an eligible policy terminal.

        The public v0.4 terminal stays terminal.  This private sidecar exists
        only if the same normalized intent already contains one explicit
        company and financial metric.  An explicit historical period is kept;
        an explicit future period may be replaced by the latest independently
        verified annual actual available before the corpus cutoff.  It never
        recovers a missing issuer/metric or performs a composer-side lookup.
        """
        from agent.semantic_intent_v1 import SemanticIntent, semantic_intent_digest
        from agent.stage1_assembly import CORPUS_CUTOFF, REFERENCE_DATE
        from agent.stage1_v1_financial_backend import FinancialResolutionBackend
        from agent.stage1_v1_outcome import Stage1V1Orchestrator
        from agent.stage1_v1_query_plan_v04_emitter import emit_stage1_v1_query_plan_v04
        from agent.stage1_v1_resolver import Stage1V1Resolver

        if (not isinstance(question, str)
                or getattr(terminal_handoff, "status", None)
                not in {"policy_refusal", "unsupported_request"}
                or not isinstance(source_intent, SemanticIntent)):
            return None
        # Planned project rows and a statement-level period actual are not
        # comparable operands.  Keep the public request terminal, but execute
        # both independently verifiable operands through a sealed two-plan
        # bundle.  A one-sided fallback is still forbidden.
        if (re.search(r"투자\s*계획", question)
                and re.search(r"(?:연결|별도|개별)?\s*현금\s*흐름표", question)
                and re.search(r"유형\s*자산\s*취득", question)
                and re.search(r"차이|차감|빼", question)):
            return self._incomparable_operand_fallback_bundle(
                question_id=question_id, question=question,
                source_intent=source_intent,
                terminal_handoff=terminal_handoff)
        reasons = set(getattr(terminal_handoff, "reasons", ()) or ())
        if not reasons.intersection({
                "future_forecast", "investment_advice",
                "causal_inference_beyond_scope",
                "unsupported_semantic_target"}):
            return None
        if (source_intent.answer_groups
                or source_intent.unresolved_mentions):
            return None
        resolver_question = question
        # HCX can split a single policy request into one factual item and one
        # policy-only item (for example revenue + buy opinion).  Recover only
        # when there is exactly one directly resolvable financial item and all
        # remaining items are explicit investment-advice surfaces.  A second
        # financial item, even if it is also a future forecast, must close
        # rather than silently dropping part of the request.
        from agent.planning import resolve_metric_concept
        from agent.stage1_v1_financial_backend import _bare_concept_surface

        issuer_by_id = {
            entity.entity_id: entity for entity in source_intent.entities}
        company_entities = [
            entity for entity in source_intent.entities
            if (entity.kind_hint == "company"
                and entity.surface and entity.surface in question)
        ]
        direct_candidates: list[tuple[object, object, object, str]] = []
        for candidate in source_intent.answer_items:
            if (
                    candidate.target.kind != "metric"
                    or candidate.operation != "retrieve"
                    or candidate.output.shape != "scalar"
                    or candidate.output.projection_mode != "named_fields"
                    or candidate.selection is not None
                    or len(candidate.target.entity_refs) != 1
                    or len(candidate.scope.target_period_expressions) != 1):
                continue
            issuer = issuer_by_id.get(candidate.target.entity_refs[0])
            period = candidate.scope.target_period_expressions[0]
            if (issuer is None or issuer.kind_hint != "company"
                    or not issuer.surface or issuer.surface not in question
                    or not candidate.target.surface
                    or candidate.target.surface not in question
                    or not period or period not in question):
                continue
            concept_surface = _bare_concept_surface(
                re.sub(r"^(?:예상|예측|전망|추정)\s*", "",
                       candidate.target.surface.strip()))
            concept = resolve_metric_concept(concept_surface)
            if concept is not None:
                direct_candidates.append(
                    (candidate, issuer, concept, period))

        explicit_alternative = self._explicit_supported_financial_alternative(
            question=question, source_intent=source_intent)
        if explicit_alternative is not None:
            fallback_intent = explicit_alternative
            company = fallback_intent.entities[0].surface
            periods = {
                period
                for item in fallback_intent.answer_items
                for period in item.scope.target_period_expressions
            }
            if len(periods) != 1:
                return None
            period = next(iter(periods))
            resolver_question = (
                f"{company}의 {period} "
                f"{', '.join(item.target.surface for item in fallback_intent.answer_items)}"
                "를 알려줘."
            )
        elif len(direct_candidates) > 1:
            return None
        elif len(direct_candidates) == 1:
            item, issuer, concept, period = direct_candidates[0]
            if (len(company_entities) != 1
                    or company_entities[0].entity_id != issuer.entity_id):
                return None
            for candidate in source_intent.answer_items:
                if candidate is item:
                    continue
                surfaces = [
                    candidate.target.surface,
                    *candidate.target.qualifier_surfaces,
                    *candidate.output.field_surfaces,
                ]
                if ("investment_advice" not in reasons
                        or not any(_INVESTMENT_ADVICE_SURFACE.search(surface)
                                   for surface in surfaces)):
                    return None
            years = re.findall(r"(?<![0-9])([0-9]{4})(?![0-9])", period)
            # This sidecar is an actual-value alternative to a forecast.  A
            # historical "예상 매출" request is not equivalent to the actual
            # report for that year, so it remains a policy refusal.
            if (len(years) != 1
                    or ("future_forecast" in reasons
                        and int(years[0]) <= int(CORPUS_CUTOFF[:4]))):
                return None
            raw = source_intent.model_dump(mode="python", warnings=False)
            # Keep only the one explicitly resolvable fact item.  The other
            # item(s) were admitted above only as policy-only surfaces; they
            # must not reach the financial resolver as if they were facts.
            raw_item = item.model_dump(mode="python", warnings=False)
            raw_item["item_id"] = "item-1"
            raw_item["target"]["surface"] = _bare_concept_surface(
                re.sub(r"^(?:예상|예측|전망|추정)\s*", "",
                       item.target.surface.strip()))
            raw_item["output"]["field_surfaces"] = [
                raw_item["target"]["surface"]]
            raw["answer_items"] = [raw_item]
            raw["answer_groups"] = []
            raw["premises"] = []
            company_rows = self._canonical.resolve_company(issuer.surface)
            if concept is None or len(company_rows) != 1:
                return None
            latest_year = self._latest_verified_annual_year(
                corp_code=company_rows[0].corp_code,
                concept=str(concept),
                scope="SFS" if re.search(r"별도|개별", question) else "CFS",
                cutoff=CORPUS_CUTOFF,
            )
            if latest_year is None:
                return None
            raw["answer_items"][0]["scope"]["target_period_expressions"] = [
                f"{latest_year}년"]
            # The ordinary resolver requires every semantic surface to be
            # present in its grounding question.  This execution-only
            # question is a deterministic replacement of the explicit future
            # period, while the sidecar remains sealed to the original
            # question and source-intent digests below.
            resolver_question = question.replace(period, f"{latest_year}년", 1)
            fallback_intent = SemanticIntent.model_validate(raw, strict=True)
        else:
            fallback_intent = self._causal_yoy_fallback_intent(
                question=question, source_intent=source_intent,
                reasons=reasons)
            if fallback_intent is None:
                return None
        native_resolver = self.service.runtime.orchestrator.resolver
        build_id = getattr(native_resolver, "canonical_build_id", "")
        version = getattr(native_resolver, "resolver_version", "")
        try:
            resolver = Stage1V1Resolver(
                FinancialResolutionBackend(
                    self._canonical, scope_authority=self._canonical,
                    canonical_build_id=build_id, resolver_version=version,
                    reference_date=REFERENCE_DATE, corpus_cutoff=CORPUS_CUTOFF),
                canonical_build_id=build_id, resolver_version=version)
            fallback_intents = [fallback_intent]
            fallback_questions = [resolver_question]
            if explicit_alternative is not None:
                fallback_intents = []
                fallback_questions = []
                raw = fallback_intent.model_dump(
                    mode="python", warnings=False)
                for item in fallback_intent.answer_items:
                    one = dict(raw)
                    one_item = item.model_dump(mode="python", warnings=False)
                    one_item["item_id"] = "item-1"
                    one["answer_items"] = [one_item]
                    fallback_intents.append(
                        SemanticIntent.model_validate(one, strict=True))
                    qualifier = " ".join(
                        item.scope.scope_qualifier_expressions)
                    fallback_questions.append(
                        f"{fallback_intent.entities[0].surface}의 "
                        f"{item.scope.target_period_expressions[0]} "
                        f"{qualifier + ' ' if qualifier else ''}"
                        f"{item.target.surface}를 알려줘.")
            emitted_rows = []
            for one_question, one_intent in zip(
                    fallback_questions, fallback_intents):
                fallback = Stage1V1Orchestrator(resolver).run(
                    question_id=question_id, question=one_question,
                    source_intent=one_intent)
                emitted_rows.append(
                    emit_stage1_v1_query_plan_v04(fallback).handoff)
            if len(emitted_rows) == 1:
                emitted = emitted_rows[0]
            else:
                payload = emitted_rows[0].model_dump(
                    mode="python", warnings=False)
                plan = payload.get("plan")
                if not isinstance(plan, dict):
                    return None
                facts: list[dict] = []
                baseline: tuple[object, ...] | None = None
                for index, row in enumerate(emitted_rows, start=1):
                    row_plan = getattr(row, "plan", None)
                    tasks = tuple(getattr(row_plan, "tasks", ()) or ())
                    if (getattr(row, "status", None) != "ready"
                            or len(tasks) != 1
                            or getattr(tasks[0], "kind", None) != "financial"
                            or len(getattr(tasks[0], "facts", ()) or ()) != 1):
                        return None
                    signature = (
                        tasks[0].facts[0].corp_code,
                        tasks[0].facts[0].period_end,
                        tasks[0].as_of, tasks[0].view)
                    if baseline is None:
                        baseline = signature
                    elif signature != baseline:
                        return None
                    fact = tasks[0].facts[0].model_dump(
                        mode="python", warnings=False)
                    fact["output_id"] = f"output-{index}"
                    facts.append(fact)
                first_task = plan["tasks"][0]
                first_task["facts"] = facts
                plan["tasks"] = [first_task]
                payload["handoff_id"] = str(uuid5(
                    NAMESPACE_URL,
                    "policy-fallback:"
                    + semantic_intent_digest(fallback_intent)))
                emitted = type(emitted_rows[0]).model_validate(
                    payload, strict=True)
        except Exception:
            return None
        if getattr(emitted, "status", None) != "ready":
            return None
        return {
            "source_intent_digest": semantic_intent_digest(source_intent),
            "source_question_sha256": hashlib.sha256(
                question.encode("utf-8")).hexdigest(),
            "handoff": emitted.model_dump(mode="json", warnings=False),
        }

    def _incomparable_operand_fallback_bundle(
            self, *, question_id: str, question: str, source_intent,
            terminal_handoff: object,
            ) -> dict | None:
        """Seal two cited operands while refusing an invalid subtraction.

        Investment-plan rows and a statement cash-flow line are different
        aggregation scopes.  This bridge never creates a derivation between
        them.  It derives two reduced, literal-axis requests from the original
        question, resolves each with the ordinary Stage1 runtime, and carries
        the investment sum instruction as that ready handoff's own hash-bound
        runtime annotation.  Stage2 remains the only layer that reads values.
        """

        from agent.planner_preflight import CanonicalSelectorRolePreflight
        from agent.semantic_intent_v1 import semantic_intent_digest
        from agent.stage1_assembly import CORPUS_CUTOFF
        from agent.stage1_v1_narrative_investment import (
            explicit_periodic_narrative_fallback_intent,
        )

        if (getattr(terminal_handoff, "status", None)
                not in {"policy_refusal", "unsupported_request"}
                or "unsupported_semantic_target" not in set(
                    getattr(terminal_handoff, "reasons", ()) or ())
                or getattr(source_intent, "answer_groups", ())
                or getattr(source_intent, "unresolved_mentions", ())
                or not callable(getattr(self.service, "start_from_intent", None))):
            return None
        preflight = CanonicalSelectorRolePreflight(
            self._canonical, corpus_cutoff=CORPUS_CUTOFF)
        company = preflight.unique_question_company_surface(question)
        if not isinstance(company, str) or not company.strip():
            return None

        investment = re.search(r"투자\s*계획", question)
        if investment is None:
            return None
        before_investment = question[:investment.start()]
        period_matches = list(re.finditer(
            r"(?<![0-9])(?:20[0-9]{2}|[0-9]{2})년\s*"
            r"(?:[13]\s*(?:분기|Q)(?:보고서)?|사업보고서)",
            before_investment, flags=re.IGNORECASE))
        if len(period_matches) != 1:
            return None
        investment_period = period_matches[0].group(0).strip()
        investment_question = (
            f"{company}의 {investment_period} 투자계획 합계를 알려줘.")
        investment_intent = explicit_periodic_narrative_fallback_intent(
            investment_question,
            company_surface_regrounder=preflight.question_company_surface)
        financial_intent = self._explicit_supported_financial_alternative(
            question=question, source_intent=source_intent)
        if investment_intent is None or financial_intent is None:
            return None
        if len(financial_intent.answer_items) != 1:
            return None
        financial_item = financial_intent.answer_items[0]
        if len(financial_item.scope.target_period_expressions) != 1:
            return None
        qualifiers = " ".join(
            financial_item.scope.scope_qualifier_expressions)
        financial_question = (
            f"{company}의 {financial_item.scope.target_period_expressions[0]} "
            f"{qualifiers + ' ' if qualifiers else ''}"
            f"{financial_item.target.surface}를 알려줘.")

        try:
            investment_result = self.service.start_from_intent(
                safe_question_id(
                    "operand-inv-" + hashlib.sha256(
                        f"{question_id}\0{question}".encode("utf-8")
                    ).hexdigest()[:20]),
                investment_question,
                investment_intent)
            financial_result = self.service.start_from_intent(
                safe_question_id(
                    "operand-fin-" + hashlib.sha256(
                        f"{question_id}\0{question}".encode("utf-8")
                    ).hexdigest()[:20]),
                financial_question,
                financial_intent)
        except Exception:
            return None
        investment_handoff = getattr(investment_result, "handoff", None)
        financial_handoff = getattr(financial_result, "handoff", None)
        if (getattr(investment_handoff, "status", None) != "ready"
                or getattr(financial_handoff, "status", None) != "ready"):
            return None
        investment_annotations = _runtime_annotations_from_result(
            investment_result, investment_handoff,
            question=investment_question)
        if (not isinstance(investment_annotations, dict)
                or not investment_annotations.get("narrative_aggregations")):
            return None
        return {
            "source_intent_digest": semantic_intent_digest(source_intent),
            "source_question_sha256": hashlib.sha256(
                question.encode("utf-8")).hexdigest(),
            "fallbacks": [{
                "handoff": investment_handoff.model_dump(
                    mode="json", warnings=False),
                "runtime_annotations": investment_annotations,
            }, {
                "handoff": financial_handoff.model_dump(
                    mode="json", warnings=False),
            }],
        }

    def _explicit_supported_financial_alternative(
            self, *, question: str, source_intent,
            ):
        """Return only financial facts the same question names as fallback."""

        from agent.planner_preflight import CanonicalSelectorRolePreflight
        from agent.semantic_intent_v1 import SemanticIntent
        from agent.stage1_assembly import CORPUS_CUTOFF

        years = list(dict.fromkeys(
            re.findall(r"(?<![0-9])(20[0-9]{2})\s*년", question)))
        company = CanonicalSelectorRolePreflight(
            self._canonical,
            corpus_cutoff=CORPUS_CUTOFF,
        ).unique_question_company_surface(question)
        concepts: list[str] = []
        scope_qualifiers: list[str] = []
        from agent.financial_request_boundaries import explicit_krw_fx_fallback
        if explicit_krw_fx_fallback(question):
            from agent.stage1_v1_closed_grounding_fallback import _financial_metric_surface
            metric = _financial_metric_surface(question)
            if metric is not None:
                concepts = [metric]
                scope_qualifiers = [value for value in ("연결", "별도", "개별")
                                    if value in question]
                if len(scope_qualifiers) > 1:
                    return None
        cashflow = re.search(
            r"(?<![0-9])(?P<year>20[0-9]{2})\s*년\s*"
            r"(?P<scope>연결|별도|개별)?\s*현금\s*흐름표.{0,80}?"
            r"(?P<concept>유형\s*자산\s*취득(?:액|\s*현금\s*유출액)?)",
            question)
        if (re.search(r"투자\s*계획", question) and cashflow is not None
                and re.search(r"차이|차감|빼", question)):
            years = [cashflow.group("year")]
            concepts = [cashflow.group("concept")]
            if cashflow.group("scope"):
                scope_qualifiers = [cashflow.group("scope")]
        if (re.search(r"부채\s*비율", question)
                and re.search(r"계산이?\s*안\s*되면|대신", question)
                and re.search(r"부채\s*총계", question)
                and re.search(r"자본\s*총계", question)):
            concepts = ["부채총계", "자본총계"]
        elif (not concepts and re.search(r"판매\s*량|몇\s*대\s*팔", question)
                and re.search(r"없으면|안\s*되면|대신", question)
                and re.search(r"매출\s*액", question)):
            concepts = ["매출액"]
        if len(years) != 1 or company is None or not concepts:
            return None
        # The source digest still seals this sidecar to the model output.  The
        # alternative intent itself uses only literal spans from the question.
        del source_intent
        return SemanticIntent.model_validate({
            "schema_version": "stage1-semantic-intent/1.1",
            "entities": [{
                "entity_id": "entity-1", "kind_hint": "company",
                "surface": company,
            }],
            "answer_items": [{
                "item_id": f"item-{index}", "operation": "retrieve",
                "target": {
                    "kind": "metric", "surface": concept,
                    "entity_refs": ["entity-1"], "qualifier_surfaces": [],
                },
                "scope": {
                    "target_period_expressions": [f"{years[0]}년"],
                    "as_of_expression": None,
                    "document_group_expression": None,
                    "scope_qualifier_expressions": scope_qualifiers,
                },
                "selection": None,
                "output": {
                    "shape": "scalar", "projection_mode": "named_fields",
                    "field_surfaces": [concept], "presentation": "auto",
                },
            } for index, concept in enumerate(concepts, start=1)],
            "answer_groups": [], "premises": [],
            "unresolved_mentions": [], "presentation": "auto",
        }, strict=True)

    def _latest_verified_annual_year(
            self, *, corp_code: str, concept: str, scope: str, cutoff: str,
            ) -> int | None:
        """Return the latest annual fact that has exact canonical Evidence.

        The bounded backward search is data-driven: it does not assume that
        the cutoff year has already filed an annual report, and it does not
        encode a competition question, company or expected year.
        """

        cutoff_year = int(cutoff[:4])
        for year in range(cutoff_year, cutoff_year - 10, -1):
            try:
                lookup = self._canonical.lookup(
                    corp_code, concept, f"{year}-12-31", as_of=cutoff,
                    scope=scope, view="restated", cumulative=True)
            except Exception:
                return None
            if (lookup.status == "ok" and lookup.selected is not None
                    and lookup.selected.citation is not None):
                return year
        return None

    def _causal_yoy_fallback_intent(self, *, question: str, source_intent,
                                    reasons: set[str]):
        """Recover a narrow YoY fact intent from an event-shaped causal wire.

        HCX may label ``매출 증가`` as an event while the question itself still
        explicitly supplies an issuer, an annual period, and one financial
        metric.  The reconstruction uses only literal question spans and a
        unique canonical metric match; company/period/metric ambiguity closes
        the sidecar rather than selecting a likely answer.
        """
        from agent.planning import resolve_metric_concept
        from agent.semantic_intent_v1 import SemanticIntent
        from agent.stage1_assembly import CORPUS_CUTOFF

        if "causal_inference_beyond_scope" not in reasons:
            return None
        if not re.search(r"증가|감소", question):
            return None
        years = re.findall(r"(?<![0-9])([0-9]{4})년", question)
        if len(years) != 1 or int(years[0]) > int(CORPUS_CUTOFF[:4]):
            return None
        companies = [
            entity for entity in source_intent.entities
            if entity.kind_hint == "company" and entity.surface in question
            and len(self._canonical.resolve_company(entity.surface)) == 1
        ]
        if len(companies) != 1:
            return None
        spans = re.findall(r"[A-Za-z가-힣]+", question)
        matches = {
            span for token in spans for start in range(len(token))
            for end in range(start + 1, len(token) + 1)
            if (span := token[start:end]) in question
            and resolve_metric_concept(span) is not None
        }
        concepts = {resolve_metric_concept(span) for span in matches}
        if len(concepts) != 1:
            return None
        metric = max(matches, key=len)
        year = f"{years[0]}년"
        direction = "증가" if "증가" in question else "감소"
        return SemanticIntent.model_validate({
            "schema_version": "stage1-semantic-intent/1.1",
            "entities": [{"entity_id": "entity-1", "kind_hint": "company",
                          "surface": companies[0].surface}],
            "answer_items": [{
                "item_id": "item-1", "operation": "compare",
                "target": {"kind": "metric", "surface": metric,
                           "entity_refs": ["entity-1"],
                           "qualifier_surfaces": [direction]},
                "scope": {"target_period_expressions": [year],
                          "as_of_expression": None,
                          "document_group_expression": None,
                          "scope_qualifier_expressions": []},
                "selection": None,
                "output": {"shape": "scalar", "projection_mode": "named_fields",
                           "field_surfaces": [direction], "presentation": "auto"},
            }],
            "answer_groups": [], "premises": [], "presentation": "auto",
            "unresolved_mentions": [],
        }, strict=True)

    def _resolution_from_result(
            self, result, *, meta: dict, question: str | None = None,
            semantic_intent: object | None = None,
            ) -> Stage1Resolution:
        """native 결과(시작·재개 turn) → Stage2용 v0.4 handoff + 공개 세션·역질문 투영.

        ready/terminal 이면 emission(handoff)이 있고 view 는 없다. 역질문 대기면 view 를 투영한다 —
        `/v1` 세션 API 용 `session`/`clarification` 과 공식 `POST /answer/resume` 용 `meta["clarification"]`
        (public resume authority) 을 함께 싣는다. 내부 digest·history 는 싣지 않는다.
        """
        view = getattr(result, "view", None)
        if view is None:
            handoff = result.handoff
            if handoff is not None:
                meta["handoff_status"] = handoff.status
                if semantic_intent is None:
                    outcome = getattr(result, "final_outcome", None)
                    envelope = getattr(outcome, "ready_envelope", None)
                    semantic_intent = getattr(envelope, "source_intent", None)
                    if semantic_intent is None:
                        decision = getattr(
                            outcome, "resolution_decision", None)
                        semantic_intent = getattr(
                            decision, "source_intent", None)
                annotations = _runtime_annotations_from_result(
                    result, handoff,
                    question=question,
                    extra_codes=self._extra_runtime_limitation_codes(
                        question, semantic_intent, handoff),
                    policy_fallback=self._policy_fallback_handoff(
                        question_id=meta.get("stage1_question_id", "runtime"),
                        question=question, source_intent=semantic_intent,
                        terminal_handoff=handoff))
                if annotations is not None:
                    meta["runtime_annotations"] = annotations
            return Stage1Resolution(handoff, self.name, meta=meta)
        return self._resolution_from_view(
            view, meta=meta, handoff=result.handoff,
            final_outcome=getattr(result, "final_outcome", None),
            clarification_message=getattr(result, "clarification_message", None))

    def _resolution_from_view(self, view, *, meta: dict, handoff=None, final_outcome=None,
                              clarification_message: str | None = None) -> Stage1Resolution:
        """ClarificationSessionView → 공개 handoff + 세션 식별자 + resume authority."""
        session = {"session_id": view.session_id, "revision": view.revision, "status": view.status}
        meta["session_id"] = view.session_id
        meta["revision"] = view.revision
        clarification = None
        if view.clarification is not None:
            session["clarification_id"] = view.clarification.clarification_id
            clarification = view.clarification.model_dump(mode="json")
            # 사용자에게 보여줄 역질문 문장 — 서비스가 준 문장을 우선, 없으면 같은 renderer 로 (두 번째 turn 부터 `추가로`, §4)
            if not (isinstance(clarification_message, str) and clarification_message.strip()):
                from agent.stage1_v1_clarification_renderer import render_clarification_message
                clarification_message = render_clarification_message(
                    view.clarification.slots, followup=view.revision > 0)
            meta["clarification_message"] = clarification_message
            # 공식 `POST /answer/resume` 가 헤더로 되돌려 줄 최소 공개 authority (세션 상태·digest 없음)
            meta["clarification"] = public_clarification_resume(view)
            if handoff is None:
                handoff = handoff_from_clarification_view(view)
        elif handoff is None:
            from agent.stage1_v1_query_plan_v04_emitter import emit_stage1_v1_query_plan_v04
            final = view.terminal_outcome or final_outcome
            if final is None and getattr(view, "ready", None) is not None:
                final = view.ready.final_outcome
            if final is not None:
                handoff = emit_stage1_v1_query_plan_v04(final).handoff
        if handoff is not None:
            meta["handoff_status"] = handoff.status
        return Stage1Resolution(handoff, self.name, meta=meta, session=session, clarification=clarification)

    def resume(
            self, *, session_id: str, clarification_id: str, revision: int,
            answers: dict[str, str], action: str = "submit",
            deadline_monotonic: float | None = None,
            ) -> Stage1ResumeResolution:
        """Resume an existing native clarification without an HCX call.

        Session lookup intentionally happens before the answer is applied.  It
        supplies the original request only to the server response/presentation
        path; the typed Stage1 runtime remains the authority for all answer,
        option, revision, and CAS validation.
        """

        if not self.available or self.service is None:
            raise Stage1Unavailable(self.error or "native Stage1 is unavailable")
        from agent.stage1_v1_clarification_session import Stage1V1ClarificationAnswer

        answer = Stage1V1ClarificationAnswer(
            session_id=session_id,
            clarification_id=clarification_id,
            expected_revision=revision,
            action=action,
            values=answers,
        )
        # The runtime's public view deliberately hides the original question.
        # Reading it here is server-internal only; it never enters metadata or
        # the public clarification payload.
        state = self.service.runtime.clarification_coordinator.store.get(session_id)
        result = self.service.answer_clarification(answer)
        meta = {
            "stage1_question_id": state.question_id,
            "status": result.status,
            "resumed": True,
        }
        return Stage1ResumeResolution(
            resolution=self._resolution_from_result(
                result, meta=meta, question=state.original_question),
            question_id=state.question_id,
            question=state.original_question,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


# ── 체인 ─────────────────────────────────────────────────────────────────────

class ChainedStage1:
    """native 우선, 실패 시 fixture 폴백(옵션). 어느 경로였는지 source·meta에 남긴다."""
    name = "chained"
    available = True

    def __init__(self, primary: Stage1 | None, fallback: Stage1 | None, *, fallback_enabled: bool = False):
        self.primary, self.fallback = primary, fallback
        self.fallback_enabled = fallback_enabled

    def _session_capable(self):
        p = self.primary
        if p is None or not callable(getattr(p, "answer_clarification", None)):
            raise Stage1Unavailable("역질문 세션 재개는 native Stage1(HCX-007)에서만 지원합니다")
        return p

    def answer_clarification(self, answer: Mapping[str, Any]) -> Stage1Resolution:
        return self._session_capable().answer_clarification(answer)

    def inspect_session(self, session_id: str) -> Stage1Resolution:
        return self._session_capable().inspect_session(session_id)

    def resolve(self, question: str, *, question_id: str,
                deadline_monotonic: float | None = None) -> Stage1Resolution:
        errors: dict[str, str] = {}
        error_diagnostics: dict[str, dict] = {}
        if self.primary is not None:
            if self.primary.available:
                try:
                    r = self.primary.resolve(question, question_id=question_id,
                                             deadline_monotonic=deadline_monotonic)
                    if r.handoff is not None:
                        return r
                    diagnostic = r.meta.get("error_diagnostic")
                    if isinstance(diagnostic, dict):
                        error_diagnostics[self.primary.name] = diagnostic
                        errors[self.primary.name] = str(
                            diagnostic.get("code", "no_handoff"))
                    else:
                        errors[self.primary.name] = "no_handoff"
                except Exception as e:   # HcxError 계열·normalization 실패·조립 오류 — 전부 typed 로 기록
                    diagnostic = safe_exception_diagnostics(e)
                    errors[self.primary.name] = _safe_error_label(e)
                    error_diagnostics[self.primary.name] = diagnostic
            else:
                diagnostic = getattr(self.primary, "error_diagnostic", None)
                if isinstance(diagnostic, dict):
                    error_diagnostics[self.primary.name] = diagnostic
                    errors[self.primary.name] = str(
                        diagnostic.get("code", "unavailable"))
                else:
                    errors[self.primary.name] = "unavailable"
        if self.fallback is not None and self.fallback_enabled:
            r = self.fallback.resolve(question, question_id=question_id,
                                      deadline_monotonic=deadline_monotonic)
            if errors:
                r.meta = {**r.meta, "primary_errors": errors}
                if error_diagnostics:
                    r.meta["primary_error_diagnostics"] = error_diagnostics
            if r.handoff is not None:
                return r
            errors[self.fallback.name] = "no_handoff"
        meta = {"errors": errors}
        if error_diagnostics:
            meta["error_diagnostics"] = error_diagnostics
        return Stage1Resolution(None, "none", meta=meta)

    def resume(self, **kwargs) -> Stage1ResumeResolution:
        """Resume only a durable native session; fixture fallback has no session."""

        if self.primary is None or not self.primary.available:
            raise Stage1Unavailable("native Stage1 clarification resume is unavailable")
        resume = getattr(self.primary, "resume", None)
        if not callable(resume):
            raise Stage1Unavailable("native Stage1 clarification resume is unavailable")
        return resume(**kwargs)
