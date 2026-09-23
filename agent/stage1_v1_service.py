"""Native Stage1 v1 service facade.

This module is deliberately a small composition boundary.  HCX owns semantic
intent extraction, the v1 clarification runtime owns resolution and session
state, and the v1 emitter owns the Stage2-facing QueryPlanHandoff.  No legacy
planner or compatibility bridge is used here.
"""

from __future__ import annotations

import json
import os
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Mapping, Protocol

from pydantic import ValidationError
from .latency_diagnostics import latency_span

from .hcx_schema import safe_validation_issue_paths
from .hcx_semantic_intent_v1 import (
    DeterministicSemanticIntentInvocation,
    HcxSemanticIntentNormalizationInvocationError,
    SemanticIntentInvocation,
)
from .semantic_intent_v1 import SemanticIntent, semantic_intent_digest
from .stage1_v1_clarification_renderer import render_clarification_message
from .stage1_v1_clarification_session import (
    ClarificationSessionView,
    Stage1V1ClarificationAnswer,
)
from .stage1_v1_outcome import Stage1Outcome
from .stage1_v1_query_plan_v04_emitter import (
    Stage1V1QueryPlanV04Emission,
    emit_stage1_v1_query_plan_v04,
)
from .stage1_v1_resolver import ClarificationAuthority


class NativeStage1ServiceError(RuntimeError):
    """The native facade received an outcome outside the v1 contract."""


@dataclass(frozen=True, slots=True)
class CachedSemanticIntentInvocation:
    """Cache-hit provenance, intentionally without a prior provider request id."""

    semantic_intent: SemanticIntent = field(repr=False)
    semantic_intent_digest: str
    canonical_build_id: str
    system_prompt_sha256: str
    provider_schema_sha256: str
    generation_config_sha256: str
    as_of: str
    cache_hit: bool = True


class Stage1SemanticIntentCache:
    """Process-local, bounded cache of intents proven to yield ``ready``.

    Values contain no provider wire, request id, latency, error, or
    clarification state.  The native runtime always runs again for a hit, so
    the current request's question id and any runtime-bound handoff are fresh.
    """

    def __init__(
            self, *, canonical_build_id: str, system_prompt_sha256: str,
            provider_schema_sha256: str, generation_config_sha256: str,
            as_of: str, max_entries: int = 64) -> None:
        values = (canonical_build_id, system_prompt_sha256,
                  provider_schema_sha256, generation_config_sha256, as_of)
        if (any(not isinstance(value, str) or not value for value in values)
                or type(max_entries) is not int or max_entries < 1):
            raise ValueError("Stage1 semantic cache 계약값이 잘못되었습니다")
        self._identity = values
        self._max_entries = max_entries
        self._rows: OrderedDict[tuple[str, str, str, str, str, str], SemanticIntent] = OrderedDict()
        self._lock = RLock()

    def _key(self, question: str) -> tuple[str, str, str, str, str, str]:
        if not isinstance(question, str) or not question:
            raise ValueError("Stage1 semantic cache question이 비어 있습니다")
        # The normalizer itself operates on NFC question text.  Preserve all
        # whitespace while collapsing only Unicode-equivalent spellings.
        return (unicodedata.normalize("NFC", question), *self._identity)

    def get(self, question: str) -> CachedSemanticIntentInvocation | None:
        key = self._key(question)
        with self._lock:
            intent = self._rows.get(key)
            if intent is None:
                return None
            self._rows.move_to_end(key)
        build_id, prompt, schema, config, as_of = self._identity
        return CachedSemanticIntentInvocation(
            semantic_intent=intent,
            semantic_intent_digest=semantic_intent_digest(intent),
            canonical_build_id=build_id,
            system_prompt_sha256=prompt,
            provider_schema_sha256=schema,
            generation_config_sha256=config,
            as_of=as_of,
        )

    def put_ready(self, question: str, intent: SemanticIntent) -> None:
        if not isinstance(intent, SemanticIntent):
            raise TypeError("Stage1 semantic cache에는 strict SemanticIntent가 필요합니다")
        key = self._key(question)
        with self._lock:
            self._rows[key] = intent
            self._rows.move_to_end(key)
            while len(self._rows) > self._max_entries:
                self._rows.popitem(last=False)


_WIRE_FAILURE_LOG_ENV = "STAGE1_WIRE_FAILURE_LOG"
_DEFAULT_WIRE_FAILURE_LOG = "out/logs/stage1_wire_failures.jsonl"

#: 이슈 #64 후속 — 깔끔한 terminal(unsupported_request 등) 종료는 예외가
#: 아니라 정상 반환값이라 위 wire-failure 로거를 절대 못 본다. 같은 질문을
#: 같은 코드로 다시 불러도 HCX가 항목 모양(특히 「두 값의 차이」·「분기
#: 성장률」 판단 항목의 entity_refs·target.surface 절단)을 매 호출 다르게
#: 내 어떤 호출은 풀리고 어떤 호출은 안 풀렸다 — regrounder를 재실호출 없이
#: 사후에 재현·검증하려면 그 wire와 typed terminal reason이 남아야 한다.
_UNSUPPORTED_LOG_ENV = "STAGE1_UNSUPPORTED_LOG"
_DEFAULT_UNSUPPORTED_LOG = "out/logs/stage1_unsupported.jsonl"

#: 이슈 #74 후속(2026-09-03, SG-006 실호출) — grounding 기각
#: (``HcxSemanticIntentNormalizationInvocationError``)은 ``runner.invoke``
#: 안에서 raise 되어 위 두 로거 모두에 안 걸린다.  기본으로 켜져 있다.
_GROUNDING_REJECTED_LOG_ENV = "STAGE1_GROUNDING_REJECTED_LOG"
_DEFAULT_GROUNDING_REJECTED_LOG = "out/logs/stage1_grounding_rejected.jsonl"


_ERROR_MESSAGE_CAP = 500


def _chained_validation_error(
        exc: BaseException, *, max_depth: int = 4,
        ) -> ValidationError | None:
    """Return the first ``pydantic.ValidationError`` in ``exc``'s cause chain."""

    current: BaseException | None = exc
    depth = 0
    while current is not None and depth < max_depth:
        if isinstance(current, ValidationError):
            return current
        depth += 1
        next_error = current.__cause__ or current.__context__
        current = next_error if isinstance(next_error, BaseException) else None
    return None


def _chain_root_cause(
        exc: BaseException, *, max_depth: int = 4,
        ) -> BaseException:
    """Return the deepest exception in ``exc``'s cause chain (bounded).

    The outermost wrapper (``Stage1V1CompilerTechnicalError`` etc.) always
    carries the same fixed, hardcoded sentence regardless of the actual
    failure — useless for triage.  The root cause (a raw ``ValueError`` /
    ``TypeError`` / ``ValidationError``) is what actually explains it.
    """

    current = exc
    depth = 0
    while depth < max_depth:
        next_error = current.__cause__ or current.__context__
        if not isinstance(next_error, BaseException):
            return current
        current = next_error
        depth += 1
    return current


def _bounded_error_text(exc: BaseException) -> str:
    """A short, capped diagnostic string for one exception.

    ``str(pydantic.ValidationError)`` embeds every field's raw
    ``input_value`` by default — the exact resolved fact/company/amount
    payload, not just the question surfaces already logged separately.
    This module's other typed technical errors
    (``Stage1V1CompilerTechnicalError`` et al.) instead always carry a
    single fixed, hardcoded Korean sentence with no interpolated content, so
    ``str(exc)`` is safe for those and for ordinary ``ValueError``/
    ``TypeError`` raised by this codebase's own deterministic resolvers
    (whose messages are hand-written Korean sentences, never a stack dump or
    a provider payload).  A ``ValidationError`` is therefore rendered from
    its ``msg``-only, input-free error list instead of ``str(exc)``.
    """

    if isinstance(exc, ValidationError):
        try:
            messages = [
                str(issue.get("msg", "")) for issue in exc.errors(
                    include_input=False, include_context=False,
                    include_url=False)
            ]
            text = "; ".join(message for message in messages if message)
        except Exception:
            text = ""
        if not text:
            text = type(exc).__name__
    else:
        text = str(exc)
    return text[:_ERROR_MESSAGE_CAP]


def _audit_stage1_wire_failure(
        *, question_id: str, question: str, invocation: object,
        error: BaseException,
        ) -> None:
    """Preserve one downstream technical failure for post-mortem/few-shot mining.

    Triggered on *every* exception ``NativeClarificationRuntime.start``
    raises (compiler/resolver/outcome binding failures — the only exceptions
    that contract is documented to raise; a normal "no evidence yet"/
    "needs clarification" result is always a return value, never an
    exception) — not only a chained ``ValidationError`` (issue #43 §2-B①
    follow-up: a provider that drops a period onto ``target.qualifier_surfaces``
    instead of ``scope.target_period_expressions`` failed with
    ``resolver_authority_failed``/``ValueError``, which the earlier
    ValidationError-only trigger silently missed).

    Records the question, the raw model wire when one exists (a
    schema-fallback invocation has none), the root cause's type and a
    bounded, input-free message, and — only when a ``ValidationError`` is
    anywhere in the chain — its schema-path ``loc``s (never a value, input,
    or the wrapper's own free text).  Logging failure never affects the
    pipeline.
    """

    path = os.environ.get(_WIRE_FAILURE_LOG_ENV) or _DEFAULT_WIRE_FAILURE_LOG
    if not path:
        return
    try:
        wire = getattr(invocation, "provider_wire", None)
        root = _chain_root_cause(error)
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question_id": question_id,
            "question": question,
            "wire": (
                wire.model_dump(mode="json", warnings=False)
                if wire is not None and callable(getattr(wire, "model_dump", None))
                else None
            ),
            "error_type": type(root).__name__,
            "error_message": _bounded_error_text(root),
        }
        validation_error = _chained_validation_error(error)
        if validation_error is not None:
            record["loc"] = list(safe_validation_issue_paths(validation_error))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _audit_stage1_terminal_unsupported(
        *, question_id: str, question: str, invocation: object,
        outcome: object,
        ) -> None:
    """Preserve one clean terminal (unsupported/refused) close for triage.

    ``_audit_stage1_wire_failure`` above only fires on a raised exception.
    A resolver that cleanly decides ``unsupported_request``/``out_of_scope``/
    ``policy_refusal`` never raises — that decision is a normal
    :class:`Stage1Outcome` return value — so it was invisible to the wire
    audit trail entirely.  That gap matters because a provider can emit an
    inconsistent wire shape for the *same* question across calls (issue #64
    follow-up: a derived-judgment item's ``entity_refs``/truncated
    ``target.surface`` varied call to call, and only some calls' shapes
    matched a registered regrounder), so a live failure could not be
    reproduced afterward without the exact wire that failed.

    Runs only for :class:`Stage1Outcome` with ``status == "terminal"`` — a
    :class:`ClarificationSessionView` (needs_clarification) and any ready
    outcome are not logged here.  No PII: only the already-public question
    text (also logged by the sibling wire-failure log), the resolved
    semantic intent's typed structural fields, and typed terminal reason
    codes/scopes/item ids — never a resolved company/amount/citation value.
    Logging failure never affects the pipeline.
    """

    if not isinstance(outcome, Stage1Outcome) or outcome.status != "terminal":
        return
    path = os.environ.get(_UNSUPPORTED_LOG_ENV) or _DEFAULT_UNSUPPORTED_LOG
    if not path:
        return
    try:
        authority = outcome.resolution_decision.authority
        reasons = [
            {
                "code": reason.code,
                "scope": reason.scope,
                "item_ids": list(reason.item_ids),
                "diagnostic_code": reason.diagnostic_code,
            }
            for reason in getattr(authority, "reasons", ())
        ]
        wire = getattr(invocation, "provider_wire", None)
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question_id": question_id,
            "question": question,
            "wire": (
                wire.model_dump(mode="json", warnings=False)
                if wire is not None and callable(getattr(wire, "model_dump", None))
                else None
            ),
            "source_intent": outcome.resolution_decision.source_intent.model_dump(
                mode="json", warnings=False),
            "reasons": reasons,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _audit_stage1_grounding_rejected(
        *, question_id: str, question: str,
        error: HcxSemanticIntentNormalizationInvocationError,
        ) -> None:
    """Preserve one grounding-rejected HCX call for post-mortem/reproduction.

    ``HcxSemanticIntentRunner.invoke`` raises
    ``HcxSemanticIntentNormalizationInvocationError`` when the provider wire
    parses but a value/target surface fails strict question-grounding — this
    happens *inside* ``self.runner.invoke(question)``, before ``start`` ever
    binds an ``invocation`` object.  Both sibling loggers above only see a
    bound invocation (``_audit_stage1_wire_failure``: any exception from
    ``runtime.start``; ``_audit_stage1_terminal_unsupported``: a clean
    terminal ``Stage1Outcome``), so this failure class was invisible to
    both — SG-006 실호출(이슈 #74 후속, 2026-09-03)의 ``grounding_rejected``
    거절이 두 로그 어디에도 안 남은 것이 그래서다.

    No PII: the exception is deliberately content-free (see its own
    docstring) — it carries no raw provider text, only the already-public
    question, its typed diagnostic codes/paths, and
    ``HcxSafePayloadShape``(field names/kinds/counts only, never a literal
    surface).  ``resampled`` is whether this one HCX call already used its
    one allowed in-call schema-repair retry (``attempts > 1``) — it is not
    a caller-level re-ask of the same question, which this call has no
    visibility into.  Logging failure never affects the pipeline.
    """

    path = os.environ.get(
        _GROUNDING_REJECTED_LOG_ENV) or _DEFAULT_GROUNDING_REJECTED_LOG
    if not path:
        return
    try:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question_id": question_id,
            "question": question,
            # Not the raw provider wire — the exception carries none by
            # design.  This is its content-free structural diagnostic shape
            # (field names/kinds/counts only).
            "wire": error.diagnostic_shape.as_dict(),
            "diagnostic_codes": list(error.diagnostic_codes),
            "diagnostic_paths": list(error.diagnostic_paths),
            "normalization_codes": list(error.normalization_codes),
            "schema_repair_codes": list(error.schema_repair_codes),
            "request_id": error.request_id,
            "attempts": error.attempts,
            "resampled": error.attempts > 1,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


class NativeSemanticRunner(Protocol):
    def invoke(
            self, question: str, *, deadline_monotonic: float | None = None,
            ) -> (
                SemanticIntentInvocation
                | DeterministicSemanticIntentInvocation
            ): ...


class NativeClarificationRuntime(Protocol):
    def start(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent | Mapping[str, Any],
            ) -> Stage1Outcome | ClarificationSessionView: ...

    def answer_clarification(
            self,
            answer: Stage1V1ClarificationAnswer | Mapping[str, Any],
            ) -> ClarificationSessionView: ...


@dataclass(frozen=True, slots=True)
class Stage1V1ServiceResult:
    """One native service turn and its optional Stage2 emission.

    ``outcome`` remains the runtime's public object.  For a clarification
    ready view, ``final_outcome`` exposes the bound normal outcome and
    ``emission`` exposes the Stage2 handoff.  ``provider_metadata`` is the raw
    ``SemanticIntentInvocation`` or the explicit schema-fallback metadata from
    ``start``; replay starts may pass the sealed metadata back without
    invoking HCX again.
    """

    outcome: Stage1Outcome | ClarificationSessionView
    emission: Stage1V1QueryPlanV04Emission | None = None
    final_outcome: Stage1Outcome | None = None
    provider_metadata: object | None = None

    @property
    def status(self) -> str:
        return self.outcome.status

    @property
    def view(self) -> ClarificationSessionView | None:
        return (
            self.outcome
            if isinstance(self.outcome, ClarificationSessionView)
            else None
        )

    @property
    def handoff(self) -> object | None:
        """Return the emitted Stage2 handoff without bypassing the emission."""

        return None if self.emission is None else self.emission.handoff

    @property
    def clarification_message(self) -> str | None:
        """Return presentation text while preserving typed slot authority."""

        if isinstance(self.outcome, Stage1Outcome):
            authority = self.outcome.resolution_decision.authority
            if not isinstance(authority, ClarificationAuthority):
                return None
            slots = authority.slots
        else:
            public = self.outcome.clarification
            if public is None:
                return None
            slots = public.slots
        return render_clarification_message(
            slots,
            followup=(
                isinstance(self.outcome, ClarificationSessionView)
                and self.outcome.revision > 0
            ),
        )


class Stage1V1NativeService:
    """Compose HCX semantic extraction, native runtime, and v0.4 emission."""

    def __init__(
            self,
            runner: NativeSemanticRunner,
            runtime: NativeClarificationRuntime,
            *,
            semantic_cache: Stage1SemanticIntentCache | None = None,
            ) -> None:
        if not callable(getattr(runner, "invoke", None)):
            raise TypeError("native semantic runner에는 invoke가 필요합니다")
        if not callable(getattr(runtime, "start", None)):
            raise TypeError("native clarification runtime에는 start가 필요합니다")
        if not callable(getattr(runtime, "answer_clarification", None)):
            raise TypeError(
                "native clarification runtime에는 answer_clarification이 필요합니다")
        self.runner = runner
        self.runtime = runtime
        if semantic_cache is not None and not isinstance(
                semantic_cache, Stage1SemanticIntentCache):
            raise TypeError("semantic_cache는 Stage1SemanticIntentCache여야 합니다")
        self.semantic_cache = semantic_cache

    def start(
            self, question_id: str, question: str, *,
            deadline_monotonic: float | None = None,
            ) -> Stage1V1ServiceResult:
        """Resolve one intent (provider or ready-only cache), then run it anew."""

        invocation = (
            self.semantic_cache.get(question)
            if self.semantic_cache is not None else None
        )
        cache_miss = invocation is None
        if invocation is None:
            try:
                if deadline_monotonic is None:
                    invocation = self.runner.invoke(question)
                else:
                    invocation = self.runner.invoke(
                        question, deadline_monotonic=deadline_monotonic)
            except HcxSemanticIntentNormalizationInvocationError as exc:
                _audit_stage1_grounding_rejected(
                    question_id=question_id, question=question, error=exc)
                raise
        source_intent = getattr(invocation, "semantic_intent", None)
        if not isinstance(source_intent, SemanticIntent):
            raise NativeStage1ServiceError(
                "HCX invocation에 strict SemanticIntent가 없습니다")
        try:
            with latency_span("stage1_plan"):
                outcome = self.runtime.start(
                    question_id=question_id,
                    question=question,
                    source_intent=source_intent,
                )
        except Exception as exc:
            _audit_stage1_wire_failure(
                question_id=question_id, question=question,
                invocation=invocation, error=exc)
            raise
        _audit_stage1_terminal_unsupported(
            question_id=question_id, question=question,
            invocation=invocation, outcome=outcome)
        result = self._result(outcome, provider_metadata=invocation)
        # Only a fully emitted normal ready result earns a cache entry.  A
        # terminal, partial, clarification, transport, normalization, runtime,
        # or emitter failure cannot be replayed as an apparent success.
        if (cache_miss and self.semantic_cache is not None
                and result.status == "ready"
                # A deterministic schema-repair fallback is a usable current
                # answer, but not a successful provider semantic result.
                and getattr(invocation, "fallback_code", None) is None):
            self.semantic_cache.put_ready(question, source_intent)
        return result

    def start_from_intent(
            self,
            question_id: str,
            question: str,
            source_intent: SemanticIntent | Mapping[str, Any],
            *,
            provider_metadata: object | None = None,
            ) -> Stage1V1ServiceResult:
        """Replay a saved semantic intent without invoking HCX."""

        outcome = self.runtime.start(
            question_id=question_id,
            question=question,
            source_intent=source_intent,
        )
        _audit_stage1_terminal_unsupported(
            question_id=question_id, question=question,
            invocation=provider_metadata, outcome=outcome)
        return self._result(outcome, provider_metadata=provider_metadata)

    def answer_clarification(
            self,
            answer: Stage1V1ClarificationAnswer | Mapping[str, Any],
            ) -> Stage1V1ServiceResult:
        """Resume native clarification and emit only a bound ready view."""

        view = self.runtime.answer_clarification(answer)
        if not isinstance(view, ClarificationSessionView):
            raise NativeStage1ServiceError(
                "clarification runtime이 public ClarificationSessionView를 반환하지 않았습니다")
        return self._result(view, emit_ready_view=True)

    @staticmethod
    def _result(
            outcome: Stage1Outcome | ClarificationSessionView,
            *,
            provider_metadata: object | None = None,
            emit_ready_view: bool = False,
            ) -> Stage1V1ServiceResult:
        if isinstance(outcome, Stage1Outcome):
            emission = emit_stage1_v1_query_plan_v04(outcome)
            return Stage1V1ServiceResult(
                outcome=outcome,
                emission=emission,
                final_outcome=outcome,
                provider_metadata=provider_metadata,
            )
        if isinstance(outcome, ClarificationSessionView):
            final = outcome.ready.final_outcome if outcome.ready is not None else None
            emission = (
                emit_stage1_v1_query_plan_v04(final)
                if emit_ready_view and final is not None else None
            )
            return Stage1V1ServiceResult(
                outcome=outcome,
                emission=emission,
                final_outcome=final,
                provider_metadata=provider_metadata,
            )
        raise NativeStage1ServiceError(
            "native runtime이 Stage1Outcome 또는 ClarificationSessionView를 반환하지 않았습니다")


# Short public alias for callers that do not need to spell out "Native".
Stage1V1Service = Stage1V1NativeService


__all__ = [
    "CachedSemanticIntentInvocation",
    "NativeClarificationRuntime",
    "NativeSemanticRunner",
    "NativeStage1ServiceError",
    "Stage1SemanticIntentCache",
    "Stage1V1NativeService",
    "Stage1V1Service",
    "Stage1V1ServiceResult",
]
