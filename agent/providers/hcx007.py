"""HCX-007 Chat Completions v3 Structured Outputs transport.

이 모듈은 QueryPlan 의미를 모른다. typed message, compiled schema, 전역 호출 예산,
선제 QPM/TPM 예약과 provider 응답 계약만 책임진다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import math
import os
import random
import re
from threading import RLock
import time
from typing import Callable, Generic, Literal, TypeVar
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.latency_diagnostics import record_latency_event
from agent.hcx_schema import (
    CompiledHcxSchema,
    HcxPayloadValidationError,
    HcxSafePayloadShape,
    HcxSchemaError,
)


PayloadT = TypeVar("PayloadT", bound=BaseModel)

HCX_API_URL = "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-007"
HCX_MODEL = "HCX-007"
HCX_SUCCESS_CODE = "20000"
MAX_ATTEMPTS = 2


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class HcxMessage(_FrozenModel):
    role: Literal["system", "user"]
    content: str = Field(min_length=1, max_length=100_000, repr=False)

    @field_validator("content")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("HCX message content는 비어 있을 수 없습니다")
        return value


class HcxGenerationConfig(_FrozenModel):
    model: Literal["HCX-007"] = HCX_MODEL
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0, le=128)
    max_completion_tokens: int = Field(default=512, ge=1, le=32_768)
    repetition_penalty: float = Field(default=1.1, gt=0.0, le=2.0)
    seed: int = Field(default=1, ge=0, le=4_294_967_295)
    stop: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    include_ai_filters: bool = True

    @field_validator("stop")
    @classmethod
    def valid_stop(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not isinstance(item, str) or not item or len(item) > 100 for item in value):
            raise ValueError("stop은 1~100자 문자열이어야 합니다")
        return value

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class HcxRequest(Generic[PayloadT]):
    messages: tuple[HcxMessage, ...] = field(repr=False)
    compiled_schema: CompiledHcxSchema[PayloadT] = field(repr=False)
    estimated_input_tokens: int
    config: HcxGenerationConfig = field(default_factory=HcxGenerationConfig)
    request_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if len(self.messages) != 2:
            raise HcxRequestError(
                "planner HCX 요청은 system 1개와 user 1개만 허용합니다",
                request_id=self.request_id,
            )
        if self.messages[0].role != "system" or self.messages[1].role != "user":
            raise HcxRequestError(
                "HCX message 순서는 system, user여야 합니다",
                request_id=self.request_id,
            )
        try:
            parsed = UUID(self.request_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise HcxRequestError(
                "request_id는 UUID여야 합니다", request_id="invalid") from exc
        if str(parsed) != self.request_id.lower():
            raise HcxRequestError(
                "request_id는 canonical UUID 문자열이어야 합니다",
                request_id=self.request_id,
            )
        if type(self.estimated_input_tokens) is not int or self.estimated_input_tokens <= 0:
            raise HcxRequestError(
                "estimated_input_tokens는 양의 정수여야 합니다",
                request_id=self.request_id,
            )
        self.compiled_schema.verify_integrity()

    def with_repair_hint(
            self, issue_codes: tuple[str, ...],
            ) -> "HcxRequest[PayloadT]":
        """직전 응답이 어긴 **계약 코드만** system 에 덧붙인 요청을 만든다.

        user message(질문)는 건드리지 않는다. 질문을 바꾸면 모델이 교정문에서
        표기를 따와 grounding 에서 지워지고, 무엇을 물었는지도 흐려진다.

        코드는 `[a-z_]` 만 통과시킨다. provider 가 돌려준 값을 그대로 프롬프트에
        넣지 않기 위한 경계다.
        """

        codes = tuple(dict.fromkeys(
            c for c in issue_codes
            if isinstance(c, str) and _SAFE_CODE.fullmatch(c)))[:6]
        if not codes:
            return self
        lines = ["", "[재시도] 직전 응답이 계약 검증에서 거절됐다. 같은 질문에",
                 "다시 답하되 아래를 지킨다.", ""]
        lines += [f"- {c}: {REPAIR_HINTS[c]}" if c in REPAIR_HINTS else f"- {c}"
                  for c in codes]
        system = self.messages[0]
        patched = HcxMessage(
            role=system.role, content=system.content + "\n" + "\n".join(lines))
        return replace(self, messages=(patched, self.messages[1]))

    @property
    def prompt_hash(self) -> str:
        return _canonical_hash([
            message.model_dump(mode="json") for message in self.messages
        ])


@dataclass(frozen=True, slots=True)
class HcxCallResult(Generic[PayloadT]):
    payload: PayloadT = field(repr=False)
    request_id: str
    attempts: int
    provider_status_code: str
    provider_status_message: str
    finish_reason: Literal["stop"]
    seed: int | None
    total_latency_ms: float
    limiter_wait_ms: float
    network_latency_ms: float
    validation_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    rate_limit_remaining_requests: int | None
    rate_limit_remaining_tokens: int | None
    rate_limit_reset_requests: str | None
    rate_limit_reset_tokens: str | None
    model: str
    schema_hash: str
    prompt_hash: str
    config_hash: str
    #: strict 검증 **직전에** 적용한 결정론적 복구 이름들 (`agent/hcx_repair`).
    repairs: tuple[str, ...] = ()
    #: 계약 코드를 되먹여 한 번 더 물었는가.
    repair_retried: bool = False
    #: 두 번째 요청의 prompt hash를 재구성할 수 있는 등록된 계약 코드.
    repair_issue_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.repairs != tuple(sorted(set(self.repairs))) or any(
                not isinstance(code, str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code)
                for code in self.repairs):
            raise ValueError("HCX result repair code가 유효하지 않습니다")
        if (self.repair_issue_codes
                != tuple(dict.fromkeys(self.repair_issue_codes))
                or any(not isinstance(code, str) or _SAFE_CODE.fullmatch(code) is None
                       for code in self.repair_issue_codes)):
            raise ValueError("HCX result repair issue code가 유효하지 않습니다")
        if bool(self.repair_issue_codes) != self.repair_retried:
            raise ValueError("HCX result repair issue code와 재시도 상태가 다릅니다")


class HcxError(RuntimeError):
    """본문·prompt·API key를 포함하지 않는 provider typed error."""

    code = "hcx_error"
    retryable = False

    def __init__(
            self, message: str, *, request_id: str, attempts: int = 0,
            provider_code: str | None = None,
            diagnostic_codes: tuple[str, ...] = (),
            diagnostic_paths: tuple[str, ...] = (),
            diagnostic_shape: HcxSafePayloadShape | None = None,
            usage: tuple[int, int, int] | None = None,
            latency_ms: float = 0.0,
            repairs: tuple[str, ...] = (),
            http_status: int | None = None,
            ) -> None:
        if tuple(sorted(set(diagnostic_codes))) != diagnostic_codes:
            raise ValueError("HCX diagnostic code는 정렬·중복제거되어야 합니다")
        if any(not isinstance(code, str)
               or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code)
               for code in diagnostic_codes):
            raise ValueError("HCX diagnostic code 형식이 잘못되었습니다")
        if tuple(sorted(set(diagnostic_paths))) != diagnostic_paths:
            raise ValueError("HCX diagnostic path는 정렬·중복제거되어야 합니다")
        if any(not isinstance(path, str) or not re.fullmatch(
                r"[a-z][a-z0-9_]*(?:\[[0-9]+\])?"
                r"(?:\.[a-z][a-z0-9_]*(?:\[[0-9]+\])?)*",
                path,
                ) for path in diagnostic_paths):
            raise ValueError("HCX diagnostic path 형식이 잘못되었습니다")
        if diagnostic_shape is not None and not isinstance(
                diagnostic_shape, HcxSafePayloadShape):
            raise TypeError("HCX diagnostic shape 형식이 잘못되었습니다")
        if repairs != tuple(sorted(set(repairs))) or any(
                not isinstance(code, str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code)
                for code in repairs):
            raise ValueError("HCX error repair code가 유효하지 않습니다")
        if (not isinstance(latency_ms, (int, float))
                or isinstance(latency_ms, bool) or not math.isfinite(latency_ms)
                or latency_ms < 0):
            raise ValueError("HCX error latency는 유한한 음이 아닌 숫자여야 합니다")
        if usage is None:
            prompt_tokens = completion_tokens = total_tokens = 0
            usage_known = False
        else:
            if (not isinstance(usage, tuple) or len(usage) != 3
                    or any(type(value) is not int or value < 0 for value in usage)
                    or usage[0] + usage[1] != usage[2]):
                raise ValueError("HCX error usage 형식이 잘못되었습니다")
            prompt_tokens, completion_tokens, total_tokens = usage
            usage_known = True
        self.request_id = request_id
        self.attempts = attempts
        self.provider_code = provider_code
        self.http_status = http_status
        self.diagnostic_codes = diagnostic_codes
        self.diagnostic_paths = diagnostic_paths
        self.diagnostic_shape = diagnostic_shape
        self.repairs = repairs
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.usage_known = usage_known
        self.latency_ms = float(latency_ms)
        suffix = f" (request_id={request_id}, attempts={attempts})"
        super().__init__(f"{self.code}: {message}{suffix}")


#: 재시도에 덧붙이는 **계약 코드별 한 줄 설명**. 우리 enum 에서 나온 문장이고
#: 모델이 쓴 자유문은 절대 포함하지 않는다 — 환각을 되돌려주면 두 번째 시도가
#: 첫 번째의 오류를 굳힌다.
REPAIR_HINTS: dict[str, str] = {
    "model_contract_invalid":
        "각 객체는 response schema의 required 필드를 모두 채우고 허용된 enum과 타입만 쓴다. "
        "특히 answer_items의 target은 kind, surface, entity_refs를 빠뜨리지 않는다.",
    "slot_reference_undeclared":
        "requested_outputs 가 가리킨 slot 이 그 task 의 source_slots 에 없다. "
        "source_slots 를 먼저 채우고 그 index 로만 참조한다.",
    "claim_verification_ref_missing":
        "causal 이 아닌 claim 에는 verification_refs 가 필요하다. "
        "검증할 task 가 없으면 그 claim 을 두지 않는다.",
    "dead_task":
        "어느 requested output 이나 claim 에도 연결되지 않은 task 가 있다.",
    "unused_source_slot":
        "직접 ref 나 task 전체 ref 로 쓰이지 않은 source slot 이 있다.",
    "requested_output_missing":
        "process 제안에는 requested_outputs 가 하나 이상 필요하다.",
    "reason_disposition_mismatch":
        "reason_codes 가 제안한 disposition 과 호환되지 않는다.",
    "task_row_duplicate": "같은 task 행이 중복됐다.",
    "analysis_reference_invalid":
        "analysis ref 는 이미 정의된 선행 analysis 만 가리켜야 한다.",
}

_SAFE_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")


class HcxRequestError(HcxError):
    code = "invalid_request"


class HcxSchemaContractError(HcxError):
    code = "invalid_schema"


class HcxAuthenticationError(HcxError):
    code = "authentication_failed"


class HcxRateLimitError(HcxError):
    code = "rate_limited"
    retryable = True


class HcxTransientError(HcxError):
    code = "transient_provider_error"
    retryable = True


class HcxTimeoutError(HcxTransientError):
    code = "timeout"


class HcxProviderError(HcxError):
    code = "provider_error"


class HcxFinishReasonError(HcxError):
    code = "incomplete_generation"


class HcxResponseValidationError(HcxError):
    code = "response_validation_failed"


class HcxAttemptLimitError(HcxError):
    code = "attempt_limit_exhausted"


class AttemptBudget:
    """transport retry와 semantic repair가 공유하는 호출 횟수 예산."""

    def __init__(
            self, max_attempts: int = MAX_ATTEMPTS, *,
            allow_transport_retry: bool = True,
            ) -> None:
        if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_ATTEMPTS:
            raise ValueError(f"max_attempts는 1..{MAX_ATTEMPTS}여야 합니다")
        if type(allow_transport_retry) is not bool:
            raise TypeError("allow_transport_retry는 bool이어야 합니다")
        self.max_attempts = max_attempts
        self.allow_transport_retry = allow_transport_retry
        self._used = 0
        self._lock = RLock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.max_attempts - self._used

    def claim(self, *, request_id: str) -> int:
        with self._lock:
            if self._used >= self.max_attempts:
                raise HcxAttemptLimitError(
                    "HCX 전역 호출 예산을 모두 사용했습니다",
                    request_id=request_id, attempts=self._used,
                )
            self._used += 1
            return self._used


@dataclass(slots=True, eq=False)
class _WindowEntry:
    timestamp: float
    requests: int
    tokens: int


@dataclass(frozen=True, slots=True)
class RateReservation:
    entry: _WindowEntry = field(repr=False, compare=False)
    waited_ms: float


class _RateLimitDeadlineExceeded(TimeoutError):
    """Internal signal translated to the public content-free HCX timeout."""


class LocalRateLimiter:
    """한 프로세스 안에서 요청 전에 QPM·TPM을 함께 예약한다."""

    def __init__(
            self, *, requests_per_window: int = 60,
            tokens_per_window: int = 60_000, window_seconds: float = 60.0,
            clock: Callable[[], float] = time.monotonic,
            sleeper: Callable[[float], None] = time.sleep,
            ) -> None:
        if (type(requests_per_window) is not int or requests_per_window <= 0
                or type(tokens_per_window) is not int or tokens_per_window <= 0
                or window_seconds <= 0):
            raise ValueError("rate limit 값은 양수여야 합니다")
        self.requests_per_window = requests_per_window
        self.tokens_per_window = tokens_per_window
        self.window_seconds = float(window_seconds)
        self._clock = clock
        self._sleeper = sleeper
        self._entries: deque[_WindowEntry] = deque()
        self._lock = RLock()

    def _purge(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._entries and self._entries[0].timestamp <= cutoff:
            self._entries.popleft()

    def acquire(
            self, *, requests: int, reserved_tokens: int,
            deadline_monotonic: float | None = None,
            ) -> RateReservation:
        if (type(requests) is not int or type(reserved_tokens) is not int
                or requests <= 0 or reserved_tokens < 0):
            raise ValueError("rate reservation은 양의 request와 음이 아닌 token이어야 합니다")
        if requests > self.requests_per_window or reserved_tokens > self.tokens_per_window:
            raise ValueError("단일 요청이 설정된 QPM/TPM 한도를 초과합니다")
        if (deadline_monotonic is not None
                and (not isinstance(deadline_monotonic, (int, float))
                     or isinstance(deadline_monotonic, bool)
                     or not math.isfinite(deadline_monotonic))):
            raise ValueError("deadline_monotonic은 유한한 숫자여야 합니다")
        started = self._clock()
        while True:
            with self._lock:
                now = self._clock()
                if deadline_monotonic is not None and now >= deadline_monotonic:
                    raise _RateLimitDeadlineExceeded
                self._purge(now)
                used_requests = sum(entry.requests for entry in self._entries)
                used_tokens = sum(entry.tokens for entry in self._entries)
                if (used_requests + requests <= self.requests_per_window
                        and used_tokens + reserved_tokens <= self.tokens_per_window):
                    entry = _WindowEntry(now, requests, reserved_tokens)
                    self._entries.append(entry)
                    return RateReservation(
                        entry=entry, waited_ms=max(0.0, (now - started) * 1000.0))
                wait = max(
                    0.001,
                    self._entries[0].timestamp + self.window_seconds - now,
                )
                if deadline_monotonic is not None:
                    wait = min(wait, deadline_monotonic - now)
            self._sleeper(wait)

    def release(self, reservation: RateReservation) -> None:
        """Drop a reservation when no provider request was made."""

        with self._lock:
            try:
                self._entries.remove(reservation.entry)
            except ValueError:
                pass

    def reconcile(self, reservation: RateReservation, *, actual_tokens: int) -> None:
        if type(actual_tokens) is not int or actual_tokens < 0:
            raise ValueError("actual_tokens는 음이 아닌 정수여야 합니다")
        with self._lock:
            if reservation.entry in self._entries:
                reservation.entry.tokens = actual_tokens


def _header_int(headers: httpx.Headers, name: str) -> int | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _safe_provider_status(payload: object) -> tuple[str | None, str | None]:
    if not isinstance(payload, dict):
        return None, None
    status = payload.get("status")
    if not isinstance(status, dict):
        return None, None
    code = status.get("code")
    message = status.get("message")
    return (
        str(code) if isinstance(code, (str, int)) else None,
        str(message)[:200] if isinstance(message, str) else None,
    )


class HcxStructuredClient:
    """연결을 재사용하는 동기 HCX-007 Structured Outputs client."""

    def __init__(
            self, *, api_key: str, http_client: httpx.Client | None = None,
            limiter: LocalRateLimiter | None = None,
            timeout: httpx.Timeout | float = httpx.Timeout(30.0, connect=5.0),
            api_url: str = HCX_API_URL,
            sleeper: Callable[[float], None] = time.sleep,
            clock: Callable[[], float] = time.monotonic,
            random_source: random.Random | None = None,
            retry_backoff_base: float = 0.25,
            ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("CLOVASTUDIO_API_KEY가 비어 있습니다")
        if not isinstance(api_url, str) or not api_url.startswith("https://"):
            raise ValueError("HCX API URL은 https여야 합니다")
        if retry_backoff_base < 0:
            raise ValueError("retry_backoff_base는 음수일 수 없습니다")
        self._api_key = api_key.strip()
        self._api_url = api_url
        self._limiter = limiter or LocalRateLimiter(
            clock=clock, sleeper=sleeper)
        self._sleeper = sleeper
        self._clock = clock
        self._random = random_source or random.Random()
        self._retry_backoff_base = retry_backoff_base
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            follow_redirects=False,
        )
        self._base_timeout = self._http.timeout
        self._closed = False

    @classmethod
    def from_env(cls, **kwargs: object) -> "HcxStructuredClient":
        key = os.environ.get("CLOVASTUDIO_API_KEY", "")
        return cls(api_key=key, **kwargs)

    def __enter__(self) -> "HcxStructuredClient":
        if self._closed:
            raise RuntimeError("닫힌 HCX client는 다시 사용할 수 없습니다")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            if self._owns_client:
                self._http.close()
            self._closed = True

    def _body(self, request: HcxRequest[PayloadT]) -> dict[str, object]:
        config = request.config
        return {
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "topP": config.top_p,
            "topK": config.top_k,
            "maxCompletionTokens": config.max_completion_tokens,
            "temperature": config.temperature,
            "repetitionPenalty": config.repetition_penalty,
            "seed": config.seed,
            "stop": list(config.stop),
            "includeAiFilters": config.include_ai_filters,
            # HCX-007의 기본 inference를 명시적으로 끈다. Structured Outputs와
            # low/medium/high thinking은 함께 쓸 수 없고 공식 예제도 none을 보낸다.
            "thinking": {"effort": "none"},
            "responseFormat": {
                "type": "json",
                "schema": request.compiled_schema.schema_copy(),
            },
        }

    def _retry_delay(self, response: httpx.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    # A provider hint must not consume the request budget.
                    return min(2.0, max(0.0, float(retry_after)))
                except ValueError:
                    pass
        jitter = self._random.random() * self._retry_backoff_base
        return min(2.0, self._retry_backoff_base * max(1, attempt) + jitter)

    def _raise_or_retry(
            self, error: HcxError, *, budget: AttemptBudget,
            response: httpx.Response | None,
            deadline_monotonic: float | None,
            ) -> None:
        if (deadline_monotonic is not None
                and deadline_monotonic - self._clock() <= 0):
            raise HcxTimeoutError(
                "HCX request deadline을 초과했습니다",
                request_id=error.request_id, attempts=budget.used,
            ) from error
        if (not error.retryable or budget.remaining <= 0
                or not budget.allow_transport_retry):
            raise error
        delay = self._retry_delay(response, budget.used)
        if deadline_monotonic is None:
            self._sleeper(delay)
            return
        remaining = deadline_monotonic - self._clock()
        if remaining <= 0:
            raise HcxTimeoutError(
                "HCX request deadline을 초과했습니다",
                request_id=error.request_id, attempts=budget.used,
            ) from error
        if delay >= remaining:
            raise HcxTimeoutError(
                "HCX retry 전에 request deadline을 초과했습니다",
                request_id=error.request_id, attempts=budget.used,
            ) from error
        self._sleeper(delay)

    def _bounded_http_timeout(self, remaining: float) -> httpx.Timeout:
        def bounded(value: float | None) -> float:
            return remaining if value is None else min(value, remaining)

        return httpx.Timeout(
            connect=bounded(self._base_timeout.connect),
            read=bounded(self._base_timeout.read),
            write=bounded(self._base_timeout.write),
            pool=bounded(self._base_timeout.pool),
        )

    def _remaining_or_timeout(
            self, *, deadline_monotonic: float | None,
            request_id: str, attempts: int,
            ) -> float | None:
        if deadline_monotonic is None:
            return None
        if (not isinstance(deadline_monotonic, (int, float))
                or isinstance(deadline_monotonic, bool)
                or not math.isfinite(deadline_monotonic)):
            raise ValueError("deadline_monotonic은 유한한 숫자여야 합니다")
        remaining = float(deadline_monotonic) - self._clock()
        if remaining <= 0:
            raise HcxTimeoutError(
                "HCX request deadline을 초과했습니다",
                request_id=request_id, attempts=attempts,
            )
        return remaining

    def generate_json(
            self, request: HcxRequest[PayloadT], *,
            budget: AttemptBudget | None = None,
            deadline_monotonic: float | None = None,
            ) -> HcxCallResult[PayloadT]:
        if self._closed:
            raise RuntimeError("닫힌 HCX client입니다")
        try:
            request.compiled_schema.verify_integrity()
        except HcxSchemaError as exc:
            raise HcxSchemaContractError(
                "compiled schema integrity 검증에 실패했습니다",
                request_id=request.request_id,
            ) from exc
        attempts = budget or AttemptBudget()
        repair_retried = False
        repair_issue_codes: tuple[str, ...] = ()
        cumulative_prompt_tokens = 0
        cumulative_completion_tokens = 0
        cumulative_total_tokens = 0
        total_started = time.perf_counter()
        reserved_tokens = (
            request.estimated_input_tokens + request.config.max_completion_tokens)

        while True:
            self._remaining_or_timeout(
                deadline_monotonic=deadline_monotonic,
                request_id=request.request_id, attempts=attempts.used,
            )
            if attempts.remaining <= 0:
                attempts.claim(request_id=request.request_id)
            limiter_started = time.perf_counter()
            try:
                reservation = self._limiter.acquire(
                    requests=1, reserved_tokens=reserved_tokens,
                    deadline_monotonic=deadline_monotonic,
                )
            except _RateLimitDeadlineExceeded as exc:
                raise HcxTimeoutError(
                    "HCX rate limiter 대기 중 request deadline을 초과했습니다",
                    request_id=request.request_id, attempts=attempts.used,
                ) from exc
            finally:
                record_latency_event(
                    "provider_limiter",
                    elapsed_s=time.perf_counter() - limiter_started,
                    request_id=request.request_id,
                )
            try:
                request_timeout = self._remaining_or_timeout(
                    deadline_monotonic=deadline_monotonic,
                    request_id=request.request_id, attempts=attempts.used,
                )
            except HcxTimeoutError:
                self._limiter.release(reservation)
                raise
            try:
                attempt = attempts.claim(request_id=request.request_id)
            except HcxAttemptLimitError:
                self._limiter.release(reservation)
                raise
            network_started = time.perf_counter()
            response: httpx.Response | None = None
            try:
                kwargs: dict[str, object] = {
                    "headers": {
                        "Authorization": f"Bearer {self._api_key}",
                        "X-NCP-CLOVASTUDIO-REQUEST-ID": request.request_id,
                        "Content-Type": "application/json",
                    },
                    "json": self._body(request),
                }
                if request_timeout is not None:
                    kwargs["timeout"] = self._bounded_http_timeout(request_timeout)
                http_started = time.perf_counter()
                http_outcome: int | str = "unknown"
                try:
                    response = self._http.post(self._api_url, **kwargs)
                    http_outcome = response.status_code
                except Exception as exc:
                    http_outcome = type(exc).__name__
                    raise
                finally:
                    record_latency_event(
                        "provider_http",
                        elapsed_s=time.perf_counter() - http_started,
                        attempt=attempt,
                        outcome=http_outcome,
                        request_id=request.request_id,
                    )
            except httpx.TimeoutException:
                error = HcxTimeoutError(
                    "HCX 요청 시간이 초과되었습니다",
                    request_id=request.request_id, attempts=attempt,
                )
                self._raise_or_retry(
                    error, budget=attempts, response=None,
                    deadline_monotonic=deadline_monotonic)
                continue
            except httpx.TransportError:
                error = HcxTransientError(
                    "HCX transport 오류가 발생했습니다",
                    request_id=request.request_id, attempts=attempt,
                )
                self._raise_or_retry(
                    error, budget=attempts, response=None,
                    deadline_monotonic=deadline_monotonic)
                continue
            self._remaining_or_timeout(
                deadline_monotonic=deadline_monotonic,
                request_id=request.request_id, attempts=attempt,
            )
            network_ms = (time.perf_counter() - network_started) * 1000.0

            try:
                response_payload: object = response.json()
            except ValueError:
                response_payload = None

            provider_code, provider_message = _safe_provider_status(response_payload)
            if response.status_code in {401, 403}:
                raise HcxAuthenticationError(
                    "HCX 인증 또는 권한 검증에 실패했습니다",
                    request_id=request.request_id, attempts=attempt,
                    provider_code=provider_code,
                )
            if response.status_code == 429:
                error = HcxRateLimitError(
                    "HCX rate limit에 도달했습니다",
                    request_id=request.request_id, attempts=attempt,
                    provider_code=provider_code,
                    http_status=response.status_code,
                )
                self._raise_or_retry(
                    error, budget=attempts, response=response,
                    deadline_monotonic=deadline_monotonic)
                continue
            if response.status_code >= 500:
                error = HcxTransientError(
                    "HCX server 오류가 발생했습니다",
                    request_id=request.request_id, attempts=attempt,
                    provider_code=provider_code,
                    http_status=response.status_code,
                )
                self._raise_or_retry(
                    error, budget=attempts, response=response,
                    deadline_monotonic=deadline_monotonic)
                continue
            if not 200 <= response.status_code < 300:
                raise HcxProviderError(
                    f"HCX HTTP status가 성공이 아닙니다: {response.status_code}",
                    request_id=request.request_id, attempts=attempt,
                    provider_code=provider_code,
                )
            if not isinstance(response_payload, dict):
                raise HcxResponseValidationError(
                    "HCX 응답 body가 JSON object가 아닙니다",
                    request_id=request.request_id, attempts=attempt,
                )
            if provider_code != HCX_SUCCESS_CODE:
                raise HcxProviderError(
                    "HCX provider status가 성공이 아닙니다",
                    request_id=request.request_id, attempts=attempt,
                    provider_code=provider_code,
                )
            result = response_payload.get("result")
            if not isinstance(result, dict):
                raise HcxResponseValidationError(
                    "HCX result object가 없습니다",
                    request_id=request.request_id, attempts=attempt,
                    provider_code=provider_code,
                )
            finish_reason = result.get("finishReason")
            if finish_reason != "stop":
                # Provider 본문은 로그에 남기지 않되, 길이 초과와
                # 기타 중단을 운영에서 구분할 수 있게 제한된
                # finish reason만 safe diagnostic code로 보존한다.
                reason_surface = re.sub(
                    r"[^a-z0-9_]+", "_", str(finish_reason).lower()).strip("_")
                if not reason_surface:
                    reason_surface = "unknown"
                reason_code = f"finish_reason_{reason_surface}"[:64]
                raise HcxFinishReasonError(
                    "HCX generation이 stop으로 완료되지 않았습니다",
                    request_id=request.request_id, attempts=attempt,
                    provider_code=provider_code,
                    diagnostic_codes=(reason_code,),
                )
            message = result.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise HcxResponseValidationError(
                    "HCX result message role이 assistant가 아닙니다",
                    request_id=request.request_id, attempts=attempt,
                    provider_code=provider_code,
                )
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str):
                raise HcxResponseValidationError(
                    "HCX assistant content가 문자열이 아닙니다",
                    request_id=request.request_id, attempts=attempt,
                    provider_code=provider_code,
                )
            usage = result.get("usage")
            if not isinstance(usage, dict):
                raise HcxResponseValidationError(
                    "HCX token usage가 없습니다",
                    request_id=request.request_id, attempts=attempt,
                    provider_code=provider_code,
                )
            token_values = (
                usage.get("promptTokens"), usage.get("completionTokens"),
                usage.get("totalTokens"),
            )
            if any(type(value) is not int or value < 0 for value in token_values):
                raise HcxResponseValidationError(
                    "HCX token usage 형식이 잘못되었습니다",
                    request_id=request.request_id, attempts=attempt,
                    provider_code=provider_code,
                )
            prompt_tokens, completion_tokens, total_tokens = token_values
            assert isinstance(prompt_tokens, int)
            assert isinstance(completion_tokens, int)
            assert isinstance(total_tokens, int)
            if prompt_tokens + completion_tokens != total_tokens:
                raise HcxResponseValidationError(
                    "HCX totalTokens가 prompt+completion과 다릅니다",
                    request_id=request.request_id, attempts=attempt,
                    provider_code=provider_code,
                )
            cumulative_prompt_tokens += prompt_tokens
            cumulative_completion_tokens += completion_tokens
            cumulative_total_tokens += total_tokens
            validation_started = time.perf_counter()
            repairs: list[str] = []
            try:
                payload = request.compiled_schema.validate_json(
                    content, repairs_out=repairs)
            except HcxPayloadValidationError as exc:
                validation_ms = (
                    time.perf_counter() - validation_started) * 1000.0
                self._limiter.reconcile(
                    reservation, actual_tokens=total_tokens)
                # **1회 repair retry.** 결정론적 복구로 못 고치는 위반은 계약 코드만
                # 되먹여 한 번 더 묻는다. 모델이 쓴 자유문은 넣지 않는다.
                # 예산(`AttemptBudget`)을 transport retry 와 공유하므로 무한히
                # 늘지 않는다. 실패한 호출에만 붙으므로 비용은 실패 수만큼이다.
                if not repair_retried and attempts.remaining > 0:
                    hinted = request.with_repair_hint(exc.issue_codes)
                    if hinted is not request:
                        repair_retried = True
                        repair_issue_codes = tuple(dict.fromkeys(
                            code for code in exc.issue_codes
                            if isinstance(code, str) and _SAFE_CODE.fullmatch(code)
                        ))[:6]
                        request = hinted
                        continue
                raise HcxResponseValidationError(
                    "HCX JSON이 strict response model을 통과하지 못했습니다",
                    request_id=request.request_id, attempts=attempt,
                    provider_code=provider_code,
                    diagnostic_codes=exc.issue_codes,
                    diagnostic_paths=exc.issue_paths,
                    diagnostic_shape=exc.payload_shape,
                    usage=(cumulative_prompt_tokens,
                           cumulative_completion_tokens,
                           cumulative_total_tokens),
                    latency_ms=(time.perf_counter() - total_started) * 1000.0,
                    repairs=tuple(sorted(set(repairs))),
                ) from exc
            validation_ms = (time.perf_counter() - validation_started) * 1000.0
            self._limiter.reconcile(reservation, actual_tokens=total_tokens)
            self._remaining_or_timeout(
                deadline_monotonic=deadline_monotonic,
                request_id=request.request_id, attempts=attempt,
            )
            return HcxCallResult(
                repairs=tuple(sorted(set(repairs))),
                repair_retried=repair_retried,
                repair_issue_codes=repair_issue_codes,
                payload=payload,
                request_id=request.request_id,
                attempts=attempts.used,
                provider_status_code=provider_code,
                provider_status_message=provider_message or "",
                finish_reason="stop",
                seed=result.get("seed") if type(result.get("seed")) is int else None,
                total_latency_ms=(time.perf_counter() - total_started) * 1000.0,
                limiter_wait_ms=reservation.waited_ms,
                network_latency_ms=network_ms,
                validation_latency_ms=validation_ms,
                prompt_tokens=cumulative_prompt_tokens,
                completion_tokens=cumulative_completion_tokens,
                total_tokens=cumulative_total_tokens,
                rate_limit_remaining_requests=_header_int(
                    response.headers, "x-ratelimit-remaining-requests"),
                rate_limit_remaining_tokens=_header_int(
                    response.headers, "x-ratelimit-remaining-tokens"),
                rate_limit_reset_requests=response.headers.get(
                    "x-ratelimit-reset-requests"),
                rate_limit_reset_tokens=response.headers.get(
                    "x-ratelimit-reset-tokens"),
                model=request.config.model,
                schema_hash=request.compiled_schema.sha256,
                prompt_hash=request.prompt_hash,
                config_hash=request.config.fingerprint,
            )


__all__ = [
    "AttemptBudget", "HCX_API_URL", "HCX_MODEL", "HCX_SUCCESS_CODE",
    "HcxAttemptLimitError", "HcxAuthenticationError", "HcxCallResult",
    "HcxError", "HcxFinishReasonError", "HcxGenerationConfig", "HcxMessage",
    "HcxProviderError", "HcxRateLimitError", "HcxRequest", "HcxRequestError",
    "HcxResponseValidationError", "HcxSchemaContractError",
    "HcxStructuredClient", "HcxTimeoutError", "HcxTransientError",
    "LocalRateLimiter", "MAX_ATTEMPTS", "RateReservation",
]
