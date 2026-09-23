"""Versioned HCX-007 prompt and invocation boundary for SemanticIntent v1.

The provider receives only the exact question and the HCX-safe semantic wire
schema.  Expected handoffs, question IDs, resolver defaults, source
coordinates, and execution hints never enter either message.  The checked-in
prompt becomes live-eligible only after the canonical manifest records an
explicit user approval; callers still need an explicit live evaluation command.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping, Protocol

from agent.latency_diagnostics import record_latency_event
from agent.providers.hcx007 import (
    AttemptBudget,
    HcxCallResult,
    HcxError,
    HcxGenerationConfig,
    HcxMessage,
    HcxRateLimitError,
    HcxRequest,
    HcxResponseValidationError,
    HcxTransientError,
)
from agent.hcx_schema import HcxSafePayloadShape, safe_payload_shape
from agent.semantic_intent_v1 import (
    HCX_SEMANTIC_INTENT_WIRE_V1,
    HcxSemanticIntentWire,
    SemanticIntent,
    SemanticIntentNormalizationError,
    canonical_sha256,
    compile_hcx_semantic_intent_v1_schema,
    semantic_intent_digest,
    verify_semantic_intent_v1_schema_artifacts,
)
from agent.semantic_intent_v1_boundary import (
    SEMANTIC_INTENT_BOUNDARY_V1,
    SEMANTIC_INTENT_NORMALIZATION_CODES,
    SEMANTIC_INTENT_PRE_SCHEMA_REPAIR_CODES,
    CompanySurfaceRegrounder,
    SemanticIntentBoundaryError,
    SemanticIntentBoundaryEvidence,
    normalize_semantic_intent_bounded,
)


HCX_SEMANTIC_INTENT_PROMPT_VERSION = "hcx007-semantic-intent/1.1.4"
HCX_SEMANTIC_INTENT_PROMPT_MANIFEST_VERSION = (
    "hcx-semantic-intent-prompt-manifest/1.1"
)
PROMPT_AUDIT_VERSION = "hcx-semantic-intent-prompt-audit/1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_MANIFEST = (
    PROJECT_ROOT / "hcx007_prompt_v1_review/manifest_v114.json"
)
DEFAULT_FROZEN_QUESTIONS = (
    PROJECT_ROOT / "fixtures/query_plan_v04_final/questions_v0.4.jsonl"
)
FINAL_RELEASE_MANIFEST = (
    PROJECT_ROOT / "fixtures/query_plan_v04_final/release_manifest.json"
)
MAX_PROMPT_MANIFEST_BYTES = 8 * 1024
MAX_PROMPT_DOCUMENT_BYTES = 64 * 1024
MAX_QUESTION_CHARS = 10_000
# The final70 live run peaked at 386 completion tokens. Keep substantial
# blind-set headroom without reserving the generic 2,048-token ceiling for
# every short SemanticIntent response.
# Multi-entity comparison intents can carry one answer item with several
# entity references and a long Korean qualifier surface.  HCX occasionally
# reaches 640 before closing the strict JSON even at temperature 0.  Raising
# the ceiling does not add a second call or change the schema/prompt; billing
# remains based on actual generated tokens.
HCX_SEMANTIC_INTENT_MAX_COMPLETION_TOKENS = 1_024

# ── #165: 전송 실패(429/5xx/timeout) 재시도 ─────────────────────────────────
#
# `agent.providers.hcx007.HcxStructuredClient`도 재시도를 지원하지만
# (`AttemptBudget.allow_transport_retry`), 그 예산은 strict-schema repair와
# 같은 카운터를 공유한다(`MAX_ATTEMPTS = 2`).  여기서 그 카운터를 올려 전송
# 재시도를 얻으면, `self.allow_schema_retry=False`인 호출도 의도치 않게
# schema repair 예산을 얻어 `invoke()`의 `valid_attempt_shape` 불변식을 깬다
# (repair가 실제로 일어나면 `self.allow_schema_retry`가 False이므로
# `HcxSemanticIntentInvocationError`가 잘못 발생한다).  그래서 전송 재시도는
# **여기**(`invoke()`)에서, 매 시도마다 새 `AttemptBudget`(기존과 동일하게
# `allow_transport_retry=False`)으로 감싸 별도로 구현한다 — schema repair
# 예산·불변식은 그대로 두고 호출 자체만 되풀이한다.
#
# `Retry-After` 헤더는 `HcxStructuredClient._raise_or_retry`만 보고(응답
# 객체가 거기서 끝난다) `HcxRateLimitError`에 담겨 나오지 않는다(provider
# client 자체가 이 파일이 건드릴 수 있는 범위 밖이다) — 그래서 이 재시도는
# 헤더 값을 실제로 읽지 못하고 고정 지수 backoff만 쓴다. `retry_after_s`
# 애트리뷰트가 언젠가 노출되면 그대로 존중하도록 아래에서 우선 사용한다.
#
# 총 backoff **합**(개별 호출의 네트워크 왕복 시간은 제외, sleep 시간만) 상한.
# 서버 EXEC 예산(45초, `app/runtime_deadline`)·Stage1 몫(기본 30초,
# `server.runtime.Budgets.stage1_s`)의 절반에도 못 미치게 낮게 잡아, 재시도가
# 전부 실패해도 `server.runtime.ServerRuntime._STAGE1_RESAMPLE_MIN_S`(같은
# 12.0초)가 요구하는 재표본 여유를 재시도가 다 써버리지 않게 한다.
HCX_TRANSPORT_RETRY_BACKOFF_BUDGET_S = 12.0
#: 호출 데드라인까지 남은 시간이 이보다 적으면 재시도를 하지 않고 그대로
#: 실패를 올린다. `server.runtime.ServerRuntime._STAGE1_RESAMPLE_MIN_S`와
#: 값을 맞췄다 — agent/ 는 server/ 를 몰라야 하므로 import 로 묶지 않고
#: 상수 값만 맞춘다(둘 중 하나가 바뀌면 함께 검토해야 한다).
HCX_TRANSPORT_RETRY_MIN_REMAINING_S = 12.0
#: 지수 backoff 기준값(초) — 시도 n(1-base)마다 `_TRANSPORT_RETRY_BASE_S * 2**(n-1)`.
_TRANSPORT_RETRY_BASE_S = 1.0
#: 안전판 — backoff 합 상한(12초)이 이미 재시도 횟수를 사실상 제한하지만,
#: 무한 루프를 코드로도 배제해 둔다(1+2+4+8=15 > 12 이므로 실제로는 4회를
#: 넘기지 못한다).
_TRANSPORT_RETRY_MAX_ATTEMPTS = 8

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_QUESTION_ID = re.compile(r"\b(?:G|R)-[A-Z]-[0-9]{3}\b")
_RECEIPT_NUMBER = re.compile(r"(?<![0-9])[0-9]{14}(?![0-9])")
_HEX_DIGEST = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
_PROMPT_MANIFEST_KEYS = frozenset({
    "approval_state",
    "prompt_path",
    "prompt_sha256",
    "prompt_version",
    "provider_schema_sha256",
    "schema_version",
    "wire_version",
})
_FORBIDDEN_EXECUTION_HINTS = (
    "task_mode",
    "corp_code",
    "rcept_no",
    "receipt",
    "event_key",
    "source_file",
    "answer_root",
    "plan_root",
    "argmax",
    "difference",
    "absolute_difference",
    "percent_change",
    "discrete_from_cumulative",
    "rounding_rule",
)
_FORBIDDEN_EXAMPLE_MARKERS = ("예를 들어", "예:", "e.g.", "for example")


class HcxSemanticIntentPromptError(ValueError):
    """The versioned prompt or its review manifest is malformed."""


class HcxSemanticIntentPromptApprovalError(HcxSemanticIntentPromptError):
    """Live invocation was attempted before explicit prompt approval."""


class HcxSemanticIntentInvocationError(RuntimeError):
    """Provider result metadata does not bind to the exact v1 request."""


class HcxSemanticIntentNormalizationInvocationError(
        SemanticIntentNormalizationError):
    """Content-free normalization failure bound to one successful HCX call."""

    def __init__(
            self, *, diagnostic_codes: tuple[str, ...],
            diagnostic_paths: tuple[str, ...],
            diagnostic_shape: HcxSafePayloadShape,
            request_id: str, attempts: int, total_latency_ms: float,
            prompt_tokens: int, completion_tokens: int, total_tokens: int,
            provider_schema_sha256: str, system_prompt_sha256: str,
            request_prompt_sha256: str, generation_config_sha256: str,
            schema_repair_codes: tuple[str, ...] = (),
            normalization_codes: tuple[str, ...] = (),
            ) -> None:
        super().__init__(
            "HCX SemanticIntent wire가 질문 grounding을 통과하지 못했습니다",
            diagnostic_codes=diagnostic_codes,
            diagnostic_paths=diagnostic_paths,
        )
        if not isinstance(diagnostic_shape, HcxSafePayloadShape):
            raise TypeError("normalization diagnostic shape 형식이 잘못되었습니다")
        if schema_repair_codes != tuple(sorted(set(schema_repair_codes))) \
                or any(code not in SEMANTIC_INTENT_PRE_SCHEMA_REPAIR_CODES
                       for code in schema_repair_codes):
            raise ValueError("normalization schema repair code가 유효하지 않습니다")
        if normalization_codes != tuple(sorted(set(normalization_codes))) \
                or any(code not in SEMANTIC_INTENT_NORMALIZATION_CODES
                       for code in normalization_codes):
            raise ValueError("normalization boundary code가 유효하지 않습니다")
        self.diagnostic_shape = diagnostic_shape
        self.boundary_version = SEMANTIC_INTENT_BOUNDARY_V1
        self.schema_repair_codes = schema_repair_codes
        self.normalization_codes = normalization_codes
        self.request_id = request_id
        self.attempts = attempts
        self.total_latency_ms = float(total_latency_ms)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.provider_schema_sha256 = provider_schema_sha256
        self.system_prompt_sha256 = system_prompt_sha256
        self.request_prompt_sha256 = request_prompt_sha256
        self.generation_config_sha256 = generation_config_sha256


class SemanticIntentClient(Protocol):
    def generate_json(
            self, request: HcxRequest[HcxSemanticIntentWire], *,
            budget: AttemptBudget | None = None,
            deadline_monotonic: float | None = None,
            ) -> HcxCallResult[HcxSemanticIntentWire]: ...


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HcxSemanticIntentPromptError(
                f"prompt manifest JSON key가 중복됩니다: {key}")
        result[key] = value
    return result


def _read_regular_file(path: Path, *, maximum: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise HcxSemanticIntentPromptError(
            f"{label}는 symlink가 아닌 일반 파일이어야 합니다")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise HcxSemanticIntentPromptError(f"{label}를 읽을 수 없습니다") from exc
    if (not payload or len(payload) > maximum
            or payload.startswith(b"\xef\xbb\xbf") or b"\r" in payload):
        raise HcxSemanticIntentPromptError(f"{label} byte 계약이 잘못되었습니다")
    return payload


@dataclass(frozen=True, slots=True)
class HcxSemanticIntentPrompt:
    text: str = field(repr=False)
    sha256: str
    approval_state: str
    prompt_path: Path = field(repr=False, compare=False)
    manifest_path: Path = field(repr=False, compare=False)
    prompt_version: str = HCX_SEMANTIC_INTENT_PROMPT_VERSION
    wire_version: str = HCX_SEMANTIC_INTENT_WIRE_V1
    provider_schema_sha256: str = ""

    def __post_init__(self) -> None:
        if self.prompt_version != HCX_SEMANTIC_INTENT_PROMPT_VERSION:
            raise HcxSemanticIntentPromptError("prompt version이 정확하지 않습니다")
        if self.wire_version != HCX_SEMANTIC_INTENT_WIRE_V1:
            raise HcxSemanticIntentPromptError("wire version이 정확하지 않습니다")
        if self.approval_state not in {"pending_user_review", "approved"}:
            raise HcxSemanticIntentPromptError("prompt approval_state가 닫혀 있지 않습니다")
        if (not self.text.strip() or "\r" in self.text
                or not _DIGEST.fullmatch(self.sha256)
                or sha256(self.text.encode("utf-8")).hexdigest() != self.sha256
                or not _DIGEST.fullmatch(self.provider_schema_sha256)):
            raise HcxSemanticIntentPromptError("prompt hash 계약이 잘못되었습니다")

    @property
    def approved(self) -> bool:
        return self.approval_state == "approved"

    def require_live_approval(self) -> None:
        if not self.approved:
            raise HcxSemanticIntentPromptApprovalError(
                "SemanticIntent v1 prompt가 아직 사용자 검수 대기 상태입니다")


def audit_prompt_text(
        text: str, *,
        frozen_questions: Iterable[str] = (),
        forbidden_entity_surfaces: Iterable[str] = (),
        ) -> None:
    """Reject answer-label leakage, examples, and execution vocabulary.

    The semantic schema field names and closed enum values are allowed.  The
    audit targets information that would turn the prompt into a lookup table or
    leak downstream planning authority into the provider.
    """

    if not isinstance(text, str) or not text.strip() or "\r" in text:
        raise HcxSemanticIntentPromptError("prompt text가 비어 있거나 byte 계약을 어겼습니다")
    lowered = text.casefold()
    for token in _FORBIDDEN_EXECUTION_HINTS:
        if token.casefold() in lowered:
            raise HcxSemanticIntentPromptError(
                f"prompt에 execution hint가 포함되었습니다: {token}")
    for marker in _FORBIDDEN_EXAMPLE_MARKERS:
        if marker.casefold() in lowered:
            raise HcxSemanticIntentPromptError(
                f"prompt에 question-specific 예시 marker가 있습니다: {marker}")
    if _QUESTION_ID.search(text):
        raise HcxSemanticIntentPromptError("prompt에 question ID가 포함되었습니다")
    if _RECEIPT_NUMBER.search(text):
        raise HcxSemanticIntentPromptError("prompt에 접수번호 모양 값이 포함되었습니다")
    if _HEX_DIGEST.search(lowered):
        raise HcxSemanticIntentPromptError("prompt에 artifact digest가 포함되었습니다")
    for question in frozen_questions:
        if not isinstance(question, str) or not question:
            raise HcxSemanticIntentPromptError("frozen question audit 입력이 잘못되었습니다")
        if question in text:
            raise HcxSemanticIntentPromptError(
                "prompt에 frozen question 원문이 포함되었습니다")
    for surface in forbidden_entity_surfaces:
        if not isinstance(surface, str) or not surface:
            raise HcxSemanticIntentPromptError("entity leak audit 입력이 잘못되었습니다")
        if surface in text:
            raise HcxSemanticIntentPromptError(
                "prompt에 금지된 entity surface가 포함되었습니다")


def load_hcx_semantic_intent_prompt(
        manifest_path: str | Path = DEFAULT_PROMPT_MANIFEST,
        ) -> HcxSemanticIntentPrompt:
    """Load one hash-bound prompt block from a canonical review manifest."""

    manifest_file = Path(manifest_path)
    raw_manifest = _read_regular_file(
        manifest_file, maximum=MAX_PROMPT_MANIFEST_BYTES,
        label="SemanticIntent prompt manifest")
    if not raw_manifest.endswith(b"\n"):
        raise HcxSemanticIntentPromptError("prompt manifest는 LF로 끝나야 합니다")
    try:
        manifest_text = raw_manifest.decode("utf-8", errors="strict")
        raw = json.loads(manifest_text, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HcxSemanticIntentPromptError("prompt manifest JSON이 유효하지 않습니다") from exc
    if not isinstance(raw, dict) or frozenset(raw) != _PROMPT_MANIFEST_KEYS:
        raise HcxSemanticIntentPromptError("prompt manifest key 집합이 정확하지 않습니다")
    if manifest_text != _canonical_json(raw) + "\n":
        raise HcxSemanticIntentPromptError("prompt manifest가 canonical JSON이 아닙니다")
    if any(not isinstance(raw[key], str) for key in _PROMPT_MANIFEST_KEYS):
        raise HcxSemanticIntentPromptError("prompt manifest 값은 모두 문자열이어야 합니다")
    if raw["schema_version"] != HCX_SEMANTIC_INTENT_PROMPT_MANIFEST_VERSION:
        raise HcxSemanticIntentPromptError("prompt manifest schema_version이 다릅니다")
    if raw["prompt_version"] != HCX_SEMANTIC_INTENT_PROMPT_VERSION:
        raise HcxSemanticIntentPromptError("prompt manifest prompt_version이 다릅니다")
    if raw["wire_version"] != HCX_SEMANTIC_INTENT_WIRE_V1:
        raise HcxSemanticIntentPromptError("prompt manifest wire_version이 다릅니다")
    if (not _DIGEST.fullmatch(raw["prompt_sha256"])
            or not _DIGEST.fullmatch(raw["provider_schema_sha256"])):
        raise HcxSemanticIntentPromptError("prompt manifest digest 형식이 잘못되었습니다")
    if raw["approval_state"] not in {"pending_user_review", "approved"}:
        raise HcxSemanticIntentPromptError("prompt manifest approval_state가 잘못되었습니다")

    prompt_name = raw["prompt_path"]
    if (not prompt_name or Path(prompt_name).name != prompt_name
            or "/" in prompt_name or "\\" in prompt_name):
        raise HcxSemanticIntentPromptError("prompt_path는 같은 디렉터리의 파일명이어야 합니다")
    prompt_file = manifest_file.parent / prompt_name
    prompt_bytes = _read_regular_file(
        prompt_file, maximum=MAX_PROMPT_DOCUMENT_BYTES,
        label="SemanticIntent prompt document")
    try:
        document = prompt_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise HcxSemanticIntentPromptError("prompt document가 UTF-8이 아닙니다") from exc
    if document.count(f"`{HCX_SEMANTIC_INTENT_PROMPT_VERSION}`") != 1:
        raise HcxSemanticIntentPromptError("prompt document version marker가 정확하지 않습니다")
    blocks = re.findall(r"```text\n(.*?)\n```", document, flags=re.DOTALL)
    if len(blocks) != 1 or not blocks[0].strip():
        raise HcxSemanticIntentPromptError("prompt 실행 text block은 정확히 하나여야 합니다")
    text = blocks[0]
    audit_prompt_text(text)
    compiled = compile_hcx_semantic_intent_v1_schema()
    compiled.verify_integrity()
    if compiled.sha256 != raw["provider_schema_sha256"]:
        raise HcxSemanticIntentPromptError("prompt가 다른 provider schema에 결속돼 있습니다")
    return HcxSemanticIntentPrompt(
        text=text,
        sha256=raw["prompt_sha256"],
        approval_state=raw["approval_state"],
        prompt_path=prompt_file.resolve(),
        manifest_path=manifest_file.resolve(),
        prompt_version=raw["prompt_version"],
        wire_version=raw["wire_version"],
        provider_schema_sha256=raw["provider_schema_sha256"],
    )


@dataclass(frozen=True, slots=True)
class SemanticIntentInvocation:
    provider_wire: HcxSemanticIntentWire = field(repr=False)
    semantic_intent: SemanticIntent = field(repr=False)
    semantic_intent_digest: str
    boundary_evidence: SemanticIntentBoundaryEvidence
    request_id: str
    attempts: int
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    provider_schema_sha256: str
    system_prompt_sha256: str
    request_prompt_sha256: str
    generation_config_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(
                self.boundary_evidence, SemanticIntentBoundaryEvidence):
            raise TypeError("SemanticIntent invocation boundary evidence가 없습니다")
        wire = HcxSemanticIntentWire.model_validate(
            self.provider_wire.model_dump(mode="python", warnings=False),
            strict=True,
        )
        if canonical_sha256(wire) != self.boundary_evidence.source_wire_digest:
            raise ValueError("invocation provider wire와 boundary evidence가 다릅니다")
        digest = semantic_intent_digest(self.semantic_intent)
        if (digest != self.semantic_intent_digest
                or digest != self.boundary_evidence.semantic_intent_digest):
            raise ValueError("invocation SemanticIntent digest가 일치하지 않습니다")


@dataclass(frozen=True, slots=True)
class DeterministicSemanticIntentInvocation:
    """Question-grounded intent recovered after one failed schema repair.

    This is intentionally a different provenance type from
    :class:`SemanticIntentInvocation`: it has no valid provider wire and must
    never pretend to have boundary evidence for one.  The provider attempt and
    token metadata remain visible for operations and billing.
    """

    semantic_intent: SemanticIntent = field(repr=False)
    semantic_intent_digest: str
    fallback_code: str
    request_id: str
    attempts: int
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    provider_schema_sha256: str
    system_prompt_sha256: str
    request_prompt_sha256: str
    generation_config_sha256: str

    def __post_init__(self) -> None:
        schema_codes = {
            "explicit_narrative_matrix_schema_fallback",
            "explicit_periodic_narrative_schema_fallback",
            "explicit_closed_request_schema_fallback",
        }
        grounding_codes = {
            "explicit_narrative_matrix_grounding_fallback",
            "explicit_periodic_narrative_grounding_fallback",
            "explicit_closed_request_grounding_fallback",
        }
        if self.fallback_code not in schema_codes | grounding_codes:
            raise ValueError("deterministic semantic fallback code가 잘못되었습니다")
        if (semantic_intent_digest(self.semantic_intent)
                != self.semantic_intent_digest):
            raise ValueError("deterministic semantic fallback digest가 일치하지 않습니다")
        if self.fallback_code in schema_codes and self.attempts != 2:
            raise ValueError("deterministic semantic fallback은 schema repair 뒤에만 허용됩니다")
        if self.fallback_code in grounding_codes and self.attempts not in {1, 2}:
            raise ValueError("deterministic grounding fallback attempt가 잘못되었습니다")


class HcxSemanticIntentRunner:
    """One-question HCX semantic interpreter with no resolver authority."""

    def __init__(
            self, client: SemanticIntentClient, *,
            prompt: HcxSemanticIntentPrompt | None = None,
            generation_config: HcxGenerationConfig | None = None,
            company_surface_regrounder: CompanySurfaceRegrounder | None = None,
            allow_schema_retry: bool = False,
            ) -> None:
        self.client = client
        self.prompt = prompt or load_hcx_semantic_intent_prompt()
        self.generation_config = generation_config or HcxGenerationConfig(
            temperature=0.0,
            top_p=0.8,
            top_k=0,
            max_completion_tokens=HCX_SEMANTIC_INTENT_MAX_COMPLETION_TOKENS,
            repetition_penalty=1.1,
            seed=7,
        )
        #: 재표본 1회에만 쓰는 덮어쓰기 — ``{"seed": int, "retry_hint": str}``.
        #: temperature 0·seed 고정 재호출은 같은 표본을 돌려주므로(이슈 #19 관측),
        #: grounding 이 기각된 뒤의 재표본은 seed 를 바꾸고 기각 사유를 힌트로 준다.
        #: 서버 실행 슬롯이 1개라 요청 간에 겹치지 않으며 호출자가 try/finally 로 되돌린다.
        self.resample_overrides: dict[str, Any] | None = None
        if company_surface_regrounder is not None \
                and not callable(company_surface_regrounder):
            raise TypeError("company surface regrounder는 callable이어야 합니다")
        if type(allow_schema_retry) is not bool:
            raise TypeError("allow_schema_retry는 bool이어야 합니다")
        self.company_surface_regrounder = company_surface_regrounder
        self.allow_schema_retry = allow_schema_retry
        self.compiled_schema = compile_hcx_semantic_intent_v1_schema()
        self.compiled_schema.verify_integrity()
        if self.compiled_schema.sha256 != self.prompt.provider_schema_sha256:
            raise HcxSemanticIntentPromptError(
                "runner schema가 reviewed prompt manifest와 다릅니다")

    def build_request(self, question: str) -> HcxRequest[HcxSemanticIntentWire]:
        if (not isinstance(question, str) or not question.strip()
                or len(question) > MAX_QUESTION_CHARS):
            raise ValueError("SemanticIntent question이 비어 있거나 너무 깁니다")
        payload: dict[str, Any] = {"question": question}
        config = self.generation_config
        overrides = self.resample_overrides or {}
        hint = overrides.get("retry_hint")
        if isinstance(hint, str) and hint.strip():
            # 질문 자체는 바꾸지 않는다 — grounding 은 question 에 대해 검사된다.
            payload["retry_hint"] = hint.strip()[:400]
        seed = overrides.get("seed")
        if isinstance(seed, int) and seed != config.seed:
            config = config.model_copy(update={"seed": seed})
        user = _canonical_json(payload)
        messages = (
            HcxMessage(role="system", content=self.prompt.text),
            HcxMessage(role="user", content=user),
        )
        estimated = sum(
            len(message.content.encode("utf-8")) for message in messages)
        estimated += len(self.compiled_schema.canonical_json.encode("utf-8"))
        return HcxRequest(
            messages=messages,
            compiled_schema=self.compiled_schema,
            estimated_input_tokens=max(1, estimated),
            config=config,
        )

    def _call_with_transport_retry(
            self, request: HcxRequest[HcxSemanticIntentWire], *,
            deadline_monotonic: float | None,
            ) -> HcxCallResult[HcxSemanticIntentWire]:
        """Retry only 429/5xx/timeout failures; schema repair is untouched.

        Each attempt gets a **fresh** ``AttemptBudget`` identical to the
        non-retrying call this replaces (same ``max_attempts``,
        ``allow_transport_retry=False``), so a transport failure never
        borrows from — or grants — the strict-schema repair budget.  See the
        module-level comment above ``HCX_TRANSPORT_RETRY_BACKOFF_BUDGET_S``
        for why that budget must stay separate.
        """

        backoff_used = 0.0
        attempt = 0

        def _give_up(exc: HcxError, *, attempt: int, backoff_used: float) -> None:
            # content-free 사후 분류용 attribute만 남긴다 — provider 본문·헤더
            # 원문은 절대 붙이지 않는다(HcxError 계약, `agent/providers/hcx007.py`).
            exc.retry_count = attempt - 1
            exc.retry_wait_s = round(backoff_used, 3)
            if isinstance(exc, HcxRateLimitError):
                exc.http_status = 429

        while True:
            budget = AttemptBudget(
                max_attempts=2 if self.allow_schema_retry else 1,
                allow_transport_retry=False,
            )
            try:
                transport_started = time.perf_counter()
                transport_outcome = "success"
                try:
                    if deadline_monotonic is None:
                        return self.client.generate_json(request, budget=budget)
                    return self.client.generate_json(
                        request, budget=budget,
                        deadline_monotonic=deadline_monotonic)
                except Exception as exc:
                    transport_outcome = type(exc).__name__
                    raise
                finally:
                    record_latency_event(
                        "transport_attempt",
                        elapsed_s=time.perf_counter() - transport_started,
                        attempt=attempt + 1,
                        outcome=transport_outcome,
                        request_id=request.request_id,
                    )
            except (HcxRateLimitError, HcxTransientError) as exc:
                attempt += 1
                remaining = (
                    None if deadline_monotonic is None
                    else deadline_monotonic - time.monotonic())
                if (attempt > _TRANSPORT_RETRY_MAX_ATTEMPTS
                        # 재표본(#92)이 이 호출 실패 뒤에도 시도될 수 있도록 최소
                        # 여유를 남긴다 — 부족하면 더 재시도하지 않는다.
                        or (remaining is not None
                            and remaining < HCX_TRANSPORT_RETRY_MIN_REMAINING_S)):
                    _give_up(exc, attempt=attempt, backoff_used=backoff_used)
                    raise
                # `retry_after_s`가 언젠가 노출되면 그 값을(상한 안에서) 우선
                # 존중한다 — provider client가 지금은 이를 노출하지 않으므로
                # 실제로는 늘 계산된 지수 backoff를 쓴다.
                retry_after = getattr(exc, "retry_after_s", None)
                if isinstance(retry_after, (int, float)) and retry_after >= 0:
                    delay = float(retry_after)
                else:
                    delay = _TRANSPORT_RETRY_BASE_S * (2 ** (attempt - 1))
                delay = min(delay, HCX_TRANSPORT_RETRY_BACKOFF_BUDGET_S - backoff_used)
                if remaining is not None:
                    delay = min(delay, remaining - HCX_TRANSPORT_RETRY_MIN_REMAINING_S)
                if delay <= 0:
                    _give_up(exc, attempt=attempt, backoff_used=backoff_used)
                    raise
                backoff_used += delay
                backoff_started = time.perf_counter()
                try:
                    time.sleep(delay)
                finally:
                    record_latency_event(
                        "transport_backoff",
                        elapsed_s=time.perf_counter() - backoff_started,
                        attempt=attempt,
                        request_id=request.request_id,
                    )
                continue

    def invoke(
            self, question: str, *, deadline_monotonic: float | None = None,
            ) -> SemanticIntentInvocation | DeterministicSemanticIntentInvocation:
        """Make one call, plus at most one strict-schema repair call.

        429/5xx/timeout failures get a bounded exponential-backoff retry
        (see ``_call_with_transport_retry``) — the total sleep time is capped
        at ``HCX_TRANSPORT_RETRY_BACKOFF_BUDGET_S`` and stops early once the
        remaining budget drops below ``HCX_TRANSPORT_RETRY_MIN_REMAINING_S``.
        The strict-schema repair second attempt is unaffected: it is only
        available inside ``generate_json`` after a completed provider
        response fails the strict structured-output model, and it receives
        registered contract codes, never model free text.
        """

        self.prompt.require_live_approval()
        request = self.build_request(question)
        try:
            result = self._call_with_transport_retry(
                request, deadline_monotonic=deadline_monotonic)
        except HcxResponseValidationError as exc:
            # Only the exact failure left after the one approved strict-schema
            # repair may enter the closed question parser.  Auth, transport,
            # timeout, incomplete generation, other schema errors and the
            # default one-attempt runner retain their existing fail-closed
            # behavior.
            fallback = None
            if (self.allow_schema_retry and exc.attempts == 2
                    and exc.diagnostic_codes == ("model_contract_invalid",)):
                from agent.stage1_v1_narrative_matrix import (
                    explicit_narrative_matrix_fallback_intent,
                )
                fallback = explicit_narrative_matrix_fallback_intent(
                    question,
                    company_surface_regrounder=self.company_surface_regrounder,
                )
                fallback_code = "explicit_narrative_matrix_schema_fallback"
                if fallback is None:
                    from agent.stage1_v1_narrative_investment import (
                        explicit_periodic_narrative_fallback_intent,
                    )
                    fallback = explicit_periodic_narrative_fallback_intent(
                        question,
                        company_surface_regrounder=self.company_surface_regrounder,
                    )
                    fallback_code = "explicit_periodic_narrative_schema_fallback"
                if fallback is None:
                    from agent.stage1_v1_closed_grounding_fallback import (
                        explicit_closed_grounding_fallback_intent,
                    )
                    fallback = explicit_closed_grounding_fallback_intent(
                        question,
                        company_surface_regrounder=self.company_surface_regrounder,
                    )
                    fallback_code = "explicit_closed_request_schema_fallback"
            if fallback is None:
                raise
            return DeterministicSemanticIntentInvocation(
                semantic_intent=fallback,
                semantic_intent_digest=semantic_intent_digest(fallback),
                fallback_code=fallback_code,
                request_id=exc.request_id,
                attempts=exc.attempts,
                total_latency_ms=exc.latency_ms,
                prompt_tokens=exc.prompt_tokens,
                completion_tokens=exc.completion_tokens,
                total_tokens=exc.total_tokens,
                provider_schema_sha256=self.compiled_schema.sha256,
                system_prompt_sha256=self.prompt.sha256,
                request_prompt_sha256=request.with_repair_hint(
                    exc.diagnostic_codes).prompt_hash,
                generation_config_sha256=self.generation_config.fingerprint,
            )
        valid_attempt_shape = (
            result.attempts == 1 and not result.repair_retried
        ) or (
            self.allow_schema_retry
            and result.attempts == 2 and result.repair_retried
        )
        bound_request = (
            request.with_repair_hint(result.repair_issue_codes)
            if result.repair_retried else request
        )
        if (result.request_id != request.request_id
                or not valid_attempt_shape
                or result.schema_hash != self.compiled_schema.sha256
                or result.prompt_hash != bound_request.prompt_hash
                or result.config_hash != self.generation_config.fingerprint):
            raise HcxSemanticIntentInvocationError(
                "HCX result가 exact SemanticIntent request에 결속되지 않았습니다")
        schema_repair_codes = tuple(result.repairs)
        if schema_repair_codes != tuple(sorted(set(schema_repair_codes))) \
                or any(code not in SEMANTIC_INTENT_PRE_SCHEMA_REPAIR_CODES
                       for code in schema_repair_codes):
            raise HcxSemanticIntentInvocationError(
                "HCX result에 승인되지 않은 semantic schema repair가 있습니다")
        try:
            bounded = normalize_semantic_intent_bounded(
                question, result.payload,
                company_surface_regrounder=self.company_surface_regrounder,
            )
        except SemanticIntentNormalizationError as exc:
            from agent.stage1_v1_narrative_matrix import (
                explicit_narrative_matrix_fallback_intent,
            )
            fallback = explicit_narrative_matrix_fallback_intent(
                question,
                company_surface_regrounder=self.company_surface_regrounder,
            )
            fallback_code = "explicit_narrative_matrix_grounding_fallback"
            if fallback is None:
                from agent.stage1_v1_narrative_investment import (
                    explicit_periodic_narrative_fallback_intent,
                )
                fallback = explicit_periodic_narrative_fallback_intent(
                    question,
                    company_surface_regrounder=self.company_surface_regrounder,
                )
                fallback_code = "explicit_periodic_narrative_grounding_fallback"
            if fallback is None:
                from agent.stage1_v1_closed_grounding_fallback import (
                    explicit_closed_grounding_fallback_intent,
                )
                fallback = explicit_closed_grounding_fallback_intent(
                    question,
                    company_surface_regrounder=self.company_surface_regrounder,
                )
                fallback_code = "explicit_closed_request_grounding_fallback"
            if fallback is not None:
                return DeterministicSemanticIntentInvocation(
                    semantic_intent=fallback,
                    semantic_intent_digest=semantic_intent_digest(fallback),
                    fallback_code=fallback_code,
                    request_id=result.request_id,
                    attempts=result.attempts,
                    total_latency_ms=result.total_latency_ms,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                    provider_schema_sha256=result.schema_hash,
                    system_prompt_sha256=self.prompt.sha256,
                    request_prompt_sha256=result.prompt_hash,
                    generation_config_sha256=result.config_hash,
                )
            shape = safe_payload_shape(_canonical_json(
                result.payload.model_dump(mode="json", warnings=False)))
            if shape is None:  # A strict Pydantic payload must always be an object.
                raise HcxSemanticIntentInvocationError(
                    "strict provider wire의 safe diagnostic shape 생성에 실패했습니다",
                ) from exc
            raise HcxSemanticIntentNormalizationInvocationError(
                diagnostic_codes=exc.diagnostic_codes,
                diagnostic_paths=exc.diagnostic_paths,
                diagnostic_shape=shape,
                request_id=result.request_id,
                attempts=result.attempts,
                total_latency_ms=result.total_latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                provider_schema_sha256=result.schema_hash,
                system_prompt_sha256=self.prompt.sha256,
                request_prompt_sha256=result.prompt_hash,
                generation_config_sha256=result.config_hash,
                schema_repair_codes=schema_repair_codes,
                normalization_codes=(
                    exc.normalization_codes
                    if isinstance(exc, SemanticIntentBoundaryError) else ()),
            ) from exc
        evidence = SemanticIntentBoundaryEvidence.create(
            bounded, schema_repair_codes=schema_repair_codes)
        return SemanticIntentInvocation(
            provider_wire=result.payload,
            semantic_intent=bounded.semantic_intent,
            semantic_intent_digest=bounded.semantic_intent_digest,
            boundary_evidence=evidence,
            request_id=result.request_id,
            attempts=result.attempts,
            total_latency_ms=result.total_latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            provider_schema_sha256=result.schema_hash,
            system_prompt_sha256=self.prompt.sha256,
            request_prompt_sha256=result.prompt_hash,
            generation_config_sha256=result.config_hash,
        )

    __call__ = invoke


def _frozen_prompt_audit(prompt: HcxSemanticIntentPrompt) -> int:
    """Audit against the final70 question release without loading an answer label.

    This function runs on the live path, before a provider is constructed.  It
    deliberately reads only the question half of the pinned final release:
    expected handoffs are acceptance data and must stay unopened until the
    candidate rows have been sealed by the final70 evaluator.
    """

    question_bytes = _read_regular_file(
        DEFAULT_FROZEN_QUESTIONS,
        maximum=4 * 1024 * 1024,
        label="frozen 70 questions")
    manifest_bytes = _read_regular_file(
        FINAL_RELEASE_MANIFEST,
        maximum=MAX_PROMPT_MANIFEST_BYTES,
        label="final70 release manifest",
    )
    try:
        manifest = json.loads(
            manifest_bytes.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
        )
        expected_digest = manifest["release_fixture_digests"][
            DEFAULT_FROZEN_QUESTIONS.name]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise HcxSemanticIntentPromptError(
            "final70 release manifest에서 question digest를 읽지 못했습니다") from exc
    if not isinstance(expected_digest, str) or not _DIGEST.fullmatch(expected_digest):
        raise HcxSemanticIntentPromptError("final70 question digest 형식이 잘못되었습니다")
    if sha256(question_bytes).hexdigest() != expected_digest:
        raise HcxSemanticIntentPromptError("frozen 70 question digest가 바뀌었습니다")
    try:
        rows = [json.loads(line) for line in question_bytes.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HcxSemanticIntentPromptError("frozen 70 question JSONL이 유효하지 않습니다") from exc
    questions = [row.get("question") for row in rows if isinstance(row, Mapping)]
    if len(questions) != 70 or any(not isinstance(value, str) for value in questions):
        raise HcxSemanticIntentPromptError("frozen question universe가 70행이 아닙니다")
    audit_prompt_text(
        prompt.text,
        frozen_questions=questions,
    )
    return len(questions)


def verify_hcx_semantic_intent_prompt(
        manifest_path: str | Path = DEFAULT_PROMPT_MANIFEST,
        *, require_approved: bool = False,
        ) -> HcxSemanticIntentPrompt:
    prompt = load_hcx_semantic_intent_prompt(manifest_path)
    verify_semantic_intent_v1_schema_artifacts()
    _frozen_prompt_audit(prompt)
    if require_approved:
        prompt.require_live_approval()
    return prompt


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="HCX-007 SemanticIntent v1 reviewed prompt verification")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PROMPT_MANIFEST)
    parser.add_argument("--check-live", action="store_true")
    args = parser.parse_args(argv)
    prompt = verify_hcx_semantic_intent_prompt(
        args.manifest, require_approved=args.check_live)
    print(
        f"PASS: {prompt.prompt_version} approval={prompt.approval_state} "
        f"prompt_sha256={prompt.sha256} "
        f"provider_schema_sha256={prompt.provider_schema_sha256} "
        f"audit={PROMPT_AUDIT_VERSION}"
    )
    return 0


__all__ = [
    "DEFAULT_FROZEN_QUESTIONS",
    "DEFAULT_PROMPT_MANIFEST",
    "DeterministicSemanticIntentInvocation",
    "HCX_SEMANTIC_INTENT_MAX_COMPLETION_TOKENS",
    "HCX_SEMANTIC_INTENT_PROMPT_MANIFEST_VERSION",
    "HCX_SEMANTIC_INTENT_PROMPT_VERSION",
    "HCX_TRANSPORT_RETRY_BACKOFF_BUDGET_S",
    "HCX_TRANSPORT_RETRY_MIN_REMAINING_S",
    "HcxSemanticIntentInvocationError",
    "HcxSemanticIntentNormalizationInvocationError",
    "HcxSemanticIntentPrompt",
    "HcxSemanticIntentPromptApprovalError",
    "HcxSemanticIntentPromptError",
    "HcxSemanticIntentRunner",
    "PROMPT_AUDIT_VERSION",
    "SemanticIntentInvocation",
    "audit_prompt_text",
    "load_hcx_semantic_intent_prompt",
    "verify_hcx_semantic_intent_prompt",
]


if __name__ == "__main__":
    raise SystemExit(_main())
