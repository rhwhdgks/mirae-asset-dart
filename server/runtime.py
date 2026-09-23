"""서버 런타임 — 정본·Stage1·Stage2~4 컴포넌트의 수명과 **단일 실행 스레드**를 관리한다.

HTTP 계층(`server/app.py`, `server/api_v1.py`)은 이 모듈의 `ServerRuntime`만 호출한다. 여기에는
HTTP 지식이 없고, 반대로 라우터에는 정본·파이프라인 지식이 없다.

원칙
- pyarrow(mimalloc)는 스레드 간 교차 사용 시 segfault → 정본을 만지는 모든 작업(boot 포함)은
  `ThreadPoolExecutor(max_workers=1)` 하나에서만 실행한다. 공식 `/answer`와 `/v1/*`가 같은 스레드를
  공유하므로 어느 엔드포인트도 정본을 다른 스레드에서 건드리지 못한다.
- 예산(설계서 v3 §7): EXEC(45s) 실행 컷 → 보유 근거 없이 한계 고지(무응답 금지), HARD(55s) 큐 대기 포함
  상한. Stage1(HCX-007 포함)에는 EXEC-15s 를 준다.
- 서버는 read.py safe 경로만 사용한다 (`include_restricted_raw` 코드 경로 없음).
- 실행 슬롯(admission, `SERVER_MAX_INFLIGHT` 기본 2)은 **작업이 실제로 끝날 때까지** 점유한다. Python
  스레드는 안전하게 중단할 수 없으므로 timeout 응답 뒤에도 슬롯을 먼저 돌려주지 않아 후속 요청이 작업을
  쌓지 못하게 fail-closed 한다. 슬롯이 여러 개여도 정본을 만지는 실행은 여전히 `max_workers=1` 짜리
  스레드 하나가 직렬 처리한다 — 슬롯은 "몇 개 요청을 대기시켜 줄지"만 정하고 실제 동시 실행 수는 늘리지
  않으므로 정본 read model·HCX client 메모리·스레드 안전성에 영향이 없다. 큐 대기 상한은
  `SERVER_QUEUE_WAIT_S`(기본 30초, `Budgets.queue_wait_s`)이고, 실행 timeout 은 EXEC 예산이되 큐에서
  기다린 만큼 HARD 예산 안으로 줄어든다(`Budgets.remaining_exec_s`). 완료된 작업 뒤엔 Arrow 임시 메모리를
  반환한다.
"""
from __future__ import annotations

import app.env  # noqa: F401  (.env 자동 로드 + 키 이름 호환)
import gc
import json
import os
import sqlite3
import statistics
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from app.composer.limitations import safe_limitation_message
from agent.latency_diagnostics import capture_latency, latency_span, record_latency_event
from app.runtime_deadline import (
    RequestDeadlineExceeded,
    current_deadline_monotonic,
    request_deadline,
)
from server.stage1 import ChainedStage1, Stage1Resolution, Stage1Unavailable

ROOT = Path(__file__).resolve().parents[1]
SERVICE_NAME = "DART Disclosure Agent"
SERVICE_VERSION = "0.9"


def _redact_repo_root(message: str) -> str:
    """저장소 루트 절대경로를 상대경로로 줄인다.

    `app/tools/canonical_env.py`(`CanonicalNotBuilt`, 이 파일 밖 — 수정하지
    않는다)가 정본 부재를 알릴 때 워크트리의 디스크 절대경로를 그대로 문구에
    담는다(예: ``<repo 절대경로>/out/canonical/run.json 없음 — make build-data …``).
    이 문구는 boot 실패 시 `self.error`에 그대로 저장되고, `/readyz`와
    `/answer`(준비 전 503의 think_trace)가 그걸 그대로 내보낸다 — 스택트레이스는
    아니지만 배포 경로 구조를 드러낸다(#166, 2026-09-05 API 계약 검증 결함#4).
    `src/`도 건드리지 않고, 이 서버 경계에서 감싼다.
    """
    if not message:
        return message
    root = str(ROOT)
    return message.replace(root + os.sep, "").replace(root, "")

Stage1Mode = Literal["auto", "native", "fixture"]
ComposeMode = Literal["auto", "template"]


# ── 예외 ─────────────────────────────────────────────────────────────────────

class NotReady(RuntimeError):
    """정본·인덱스 로딩 중이거나 boot 실패."""


class QueueTimeout(RuntimeError):
    """HARD 예산 안에 실행 슬롯을 얻지 못함."""


class ExecTimeout(RuntimeError):
    """EXEC 예산 안에 실행이 끝나지 않음. 실행은 백그라운드에서 마무리되어 로그에 남는다."""


class SearchIndexUnavailable(RuntimeError):
    """Stage2 FTS5 검색 인덱스가 로드되지 않음 (`make search-index`)."""


# ── 값 객체 ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Budgets:
    exec_s: float = 45.0        # 실행 컷
    hard_s: float = 55.0        # 큐 대기 포함 상한
    stage1_s: float = 30.0      # Stage1(HCX-007 포함) 몫
    tool_s: float = 30.0        # /v1/tools 단건 조회
    #: 실행 슬롯(admission) 대기 상한 — `SERVER_QUEUE_WAIT_S`. exec/hard 예산과는 독립적으로 조정한다
    #: (평가자 병렬 GET 이 큐에서 곧바로 503 되지 않게, 이슈 #55). 슬롯을 오래 기다린 요청은
    #: `remaining_exec_s` 가 남은 실행 시간을 HARD 예산 안으로 줄여 queue+exec 가 HARD 를 넘지 않는다.
    queue_wait_s: float = 30.0

    def remaining_exec_s(self, started: float) -> float:
        """큐 대기까지 포함해 HARD 를 넘기지 않는 실행 타임아웃."""
        return max(0.0, min(self.exec_s, self.hard_s - (time.time() - started)))

    @classmethod
    def from_env(cls) -> "Budgets":
        exec_s = float(os.getenv("ANSWER_EXEC_BUDGET_S", "45"))
        return cls(
            exec_s=exec_s,
            hard_s=float(os.getenv("ANSWER_HARD_BUDGET_S", "55")),
            stage1_s=float(os.getenv("ANSWER_STAGE1_BUDGET_S", str(max(5.0, exec_s - 15.0)))),
            tool_s=float(os.getenv("ANSWER_TOOL_BUDGET_S", "30")),
            queue_wait_s=float(os.getenv("SERVER_QUEUE_WAIT_S", "30")),
        )


@dataclass
class AnswerResult:
    """question → 답변 한 번의 결과. `response`는 항상 공식 5필드다 (한계 고지 포함)."""
    response: dict
    status: str                              # ok | unresolved | timeout | error | not_ready | queue_timeout
    http_status: int = 200
    meta: dict = field(default_factory=dict)
    payload: Any | None = None               # AnswerPayload (ok 일 때)
    stage1: Stage1Resolution | None = None


def stage1_resample_overrides(stage1_meta: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """두 번째 표본이 첫 표본과 **달라지도록** 덮어쓰기를 만든다.

    temperature 0 에 seed 가 같으면 재호출은 같은 표본을 돌려준다.  그래서
    seed 를 바꾸지 않는 재표본은 실패를 그대로 되풀이한다 — `R-P-003` 이
    3회 연속 똑같이 거절된 것이 그것이다.  같은 파일이 이미 「`R-P-008` 을
    8회 호출하면 실패 2회는 모두 `unsupported_request`」라고 적어 두었는데,
    정작 그 갈래가 seed 를 바꾸지 않아 재표본이 무동작이었다.

    표본에 좌우되는 실패에만 seed 를 바꾼다.  전송·스키마 실패는 표본 문제가
    아니므로 종전처럼 같은 조건으로 한 번 더 본다 — 거기서 seed 를 바꾸면
    고칠 것도 없이 답만 흔들린다.
    """

    meta = stage1_meta or {}
    diagnostics = meta.get("error_diagnostics") or {}
    native = diagnostics.get("stage1_v1_native") if isinstance(diagnostics, Mapping) else None
    code = native.get("code") if isinstance(native, Mapping) else None
    if code == "grounding_rejected":
        paths = [str(path) for path in (native.get("diagnostic_paths") or ())][:4]
        hint = ("이전 표본의 표면이 질문 원문의 연속 부분 문자열이 아니어서 기각되었습니다"
                + (f" ({', '.join(paths)})" if paths else "")
                + ". target.surface·field_surfaces·qualifier_surfaces 는 질문에 그대로 있는 "
                  "글자 그대로의 구간이어야 합니다. 두 지표를 「과/와」로 이은 질문은 "
                  "각 항목의 surface 를 그 지표 낱말만으로 적으십시오.")
        return {"seed": 11, "retry_hint": hint}

    errors = meta.get("errors") or {}
    native_error = errors.get("stage1_v1_native") if isinstance(errors, Mapping) else None
    compiled_nothing = (isinstance(native_error, str)
                        and "compiler_binding_failed" in native_error)
    if compiled_nothing or meta.get("status") == "unsupported_request":
        return {"seed": 11, "retry_hint": (
            "이전 표본은 실행 계획으로 컴파일되지 않았습니다. 한 answer_item 에는 "
            "좌표를 하나만 담고, target.surface 와 field_surfaces 는 질문에 그대로 "
            "있는 구간으로 적으십시오. 제출일·기준일은 기간이 아니라 scope 의 "
            "as_of_expression 입니다.")}
    return None


#: 예전 이름.  grounding 기각 말고도 쓰이므로 이름이 좁아졌다.
grounding_retry_overrides = stage1_resample_overrides


def limit_response(question_id: str, question: str, msg: str, trace: str) -> dict:
    """근거 없이 한계만 고지하는 5필드 (무응답 금지)."""
    return {"question_id": question_id, "question": question, "retrieved_context": "",
            "think_trace": trace, "answer": msg}


MSG_MISSING_PARAMS = ("question_id 와 question 이 모두 필요합니다. 두 값을 채워 다시 요청해 주세요. "
                      "제공된 DART 공시 자료 범위 안에서 확인 가능한 내용만 답변드립니다.")
MSG_UNRESOLVED = ("질문을 실행 가능한 조회 계획으로 해석하지 못했습니다. 회사명·기간·지표(또는 계약·공시 유형)를 포함해 "
                  "다시 질문해 주세요. 제공된 DART 공시 자료 범위 안에서 확인 가능한 내용만 답변드립니다.")
#: question_id 형식 오류 안전망(#110) — server/stage1.py의 safe_question_id 정규화를 뚫고
#: ValidationError 가 question_id 필드에서 나면, "질문을 해석하지 못했다"는 질문 탓 문구
#: 대신 원인을 가리키는 이 문구를 쓴다. 원문은 app/composer/limitations.py 하나뿐이다.
MSG_INVALID_QUESTION_ID = (
    safe_limitation_message("invalid_question_id")
    or "question_id 형식이 올바르지 않습니다. 글자로 시작하고 영숫자와 `_`·`.`·`:`·`-`만 "
       "쓸 수 있습니다. question_id 를 고쳐 다시 요청해 주세요.")
#: #165 — HCX-007 provider 전송 실패(429). "질문을 해석하지 못했다"는 질문 탓
#: 문구(MSG_UNRESOLVED) 대신 원인(일시적 지연)을 가리키는 이 문구를 쓴다.
#: 원문은 app/composer/limitations.py 하나뿐이다(#110 MSG_INVALID_QUESTION_ID 선례).
MSG_UPSTREAM_RATE_LIMITED = (
    safe_limitation_message("upstream_rate_limited")
    or "일시적인 처리 지연으로 이번 요청에 답변을 만들지 못했습니다. 잠시 후 같은 질문을 "
       "다시 요청해 주세요. 제공된 DART 공시 자료 범위 안에서 확인 가능한 내용만 답변드립니다.")
#: #165 — HCX-007 provider 5xx·timeout·connect 오류. 문구는 429 와 같다(둘 다
#: "지금은 안 됐지만 다시 해볼 만하다"는 안내이지 질문을 고치라는 안내가 아니다).
MSG_UPSTREAM_UNAVAILABLE = (
    safe_limitation_message("upstream_unavailable")
    or "일시적인 처리 지연으로 이번 요청에 답변을 만들지 못했습니다. 잠시 후 같은 질문을 "
       "다시 요청해 주세요. 제공된 DART 공시 자료 범위 안에서 확인 가능한 내용만 답변드립니다.")
MSG_NOT_READY = "서비스 준비 중입니다(정본·인덱스 로딩). 잠시 후 다시 요청해 주세요."
MSG_QUEUE = "동시 처리 한도로 지연되었습니다. 잠시 후 다시 요청해 주세요."
MSG_TIMEOUT = ("제한 시간 안에 조회를 완료하지 못했습니다. 제공된 공시 자료 범위에서 확인 가능한 내용만 답변드리며, "
               "이번 요청은 확정 답변을 드리지 못했습니다.")
MSG_ERROR = "내부 처리 오류로 확정 답변을 드리지 못했습니다. 제공된 공시 자료 범위 안에서만 답변드립니다."


def stage1_trace_line(res: Stage1Resolution, elapsed: float) -> str:
    m = res.meta
    bits = [f"stage1={res.source}"]
    if res.handoff is not None:
        bits.append(f"status={res.handoff.status}")
    if m.get("request_id"):
        bits.append(f"HCX-007 request_id={m['request_id']} tokens={m.get('total_tokens')}")
    if res.question_id_hint:
        bits.append(f"fixture={res.question_id_hint}")
    if m.get("primary_errors"):
        bits.append(f"native_fallback={m['primary_errors']}")
    # 계획을 아예 만들지 못했을 때 왜인지가 여기 담긴다. 이 줄이 없으면
    # 응답만 보고는 「해석하지 못했습니다」의 원인을 알 수 없다.
    if m.get("errors"):
        bits.append(f"errors={m['errors']}")
    # 어떤 대상을 어떤 모양으로 읽었는지가 라우팅 문제의 첫 단서다. meta 에는
    # 이미 있었지만 응답에는 나오지 않아, K-047 이 본문 조회가 아니라 문서
    # 찾기로 가는 것을 서버 밖에서는 알 수 없었다. 질문에 이미 있는 표면만
    # 싣는다 — semantic_intent_shape 가 그렇게 만들어져 있다.
    shape = m.get("semantic_intent_shape")
    if isinstance(shape, list) and shape:
        bits.append("intent=" + " ".join(
            f"{row.get('target_kind')}/{row.get('output_shape')}"
            f"/{row.get('projection_mode')}"
            for row in shape[:3] if isinstance(row, dict)))
        # 모양만으로는 왜 결속에 실패했는지 알 수 없다. 어떤 기간·필드 표면을
        # 잡았는지가 그 다음 단서다 — 「삼전 + 25년」이 깨질 때 둘 중 무엇이
        # 안 잡혔는지 이 줄이 없으면 서버 밖에서 알 수 없었다. 질문에 이미
        # 있는 표면만 싣는다.
        surfaces = [
            f"{'/'.join(row.get('periods') or []) or '-'}"
            f":{'/'.join(row.get('field_surfaces') or []) or '-'}"
            for row in shape[:3] if isinstance(row, dict)]
        if any(s != "-:-" for s in surfaces):
            bits.append("기간:필드=" + " ".join(surfaces))
        # 기간·필드까지 같은데 결속이 갈리는 판이 있다. `RPC-010` 은 필드가
        # 「몇 %p 변했는지」로 같은데도 서버만 이번 값을 답했고, operation 과
        # target 표면이 이 줄에 없어 어느 입력이 다른지 밖에서는 볼 수 없었다.
        # 둘 다 결속 검사를 이미 통과한 값이다 — operation 은 닫힌 열거형이고
        # target 표면은 질문에 있는 표면만 허용된다.
        bound = [
            f"{row.get('operation') or '-'}:{row.get('target_surface') or '-'}"
            f"{'@' + row['as_of'] if row.get('as_of') else ''}"
            f"{'/' + row['document_group'] if row.get('document_group') else ''}"
            for row in shape[:3] if isinstance(row, dict)]
        if any(value != "-:-" for value in bound):
            bits.append("연산:대상=" + " ".join(bound))
    session = getattr(res, "session", None)
    if session:
        bits.append(f"session={session['session_id']} rev={session['revision']}")
    return f"[1 해석] {' '.join(bits)} ({elapsed:.2f}s)"


class _UnavailableStage1:
    """요청한 Stage1 경로가 조립되지 않았을 때의 자리표시자.

    체인은 예외 메시지가 아니라 content-free 진단 코드만 `errors` 에 남기므로(`safe_exception_diagnostics`
    규약), 정적 사유를 `error_diagnostic` 코드로도 싣는다 → `errors[name] == code`.
    """
    available = True

    def __init__(self, name: str, reason: str, code: str = "stage1_not_assembled"):
        self.name, self.reason, self.code = name, reason, code
        self.error_diagnostic = {"layer": "stage1", "code": code}

    def resolve(self, question, *, question_id, deadline_monotonic=None) -> Stage1Resolution:
        return Stage1Resolution(None, self.name, meta={"error": self.reason,
                                                       "error_diagnostic": self.error_diagnostic})


# ── 런타임 ────────────────────────────────────────────────────────────────────

def _stage1_yielded_no_answer(res) -> bool:
    """Stage1 이 답을 내지 못하고 끝났는가.

    변동은 두 갈래로 나타난다. 계획 자체가 서지 않거나(`handoff is None`),
    계획은 섰는데 `unsupported_request` 로 닫힌다. `R-P-008` 을 8회 호출하면
    6회는 정상 답변이고 실패 2회는 **모두 후자**였다 — 첫 갈래만 막으면 변동의
    대부분이 남는다.

    `unsupported_request` 는 정당한 거절이기도 하다. 다만 두 번째 표본도 같은
    검증(질문 문자 범위 결속)과 같은 정책을 거치므로, 정책이 결정적이면 결과가
    같다. 뒤집힌다면 그 경계가 애초에 표본에 의존했다는 뜻이고, 그것은 재표본이
    만든 문제가 아니라 드러낸 문제다. 실호출로 확인한다.
    """

    if getattr(res, "handoff", None) is None:
        return True
    return getattr(res.handoff, "status", None) == "unsupported_request"


def _stage1_error_codes(res) -> set[str]:
    """`ChainedStage1.resolve` 가 실패마다 남기는 `error_diagnostics` 코드들 (#110 안전망 판정용)."""
    meta = getattr(res, "meta", None)
    diagnostics = meta.get("error_diagnostics") if isinstance(meta, dict) else None
    if not isinstance(diagnostics, dict):
        return set()
    return {d.get("code") for d in diagnostics.values()
            if isinstance(d, dict) and isinstance(d.get("code"), str)}


def _stage1_unresolved_message(res) -> tuple[str, str | None]:
    """Keep provider outages distinct from a question interpretation failure."""

    codes = _stage1_error_codes(res)
    if "invalid_question_id" in codes:
        return MSG_INVALID_QUESTION_ID, "invalid_question_id"
    if "upstream_rate_limited" in codes:
        return MSG_UPSTREAM_RATE_LIMITED, "upstream_rate_limited"
    if "upstream_unavailable" in codes:
        return MSG_UPSTREAM_UNAVAILABLE, "upstream_unavailable"
    # Compatibility with a pre-#165 diagnostic emitted by a still-running
    # native worker during a rolling deployment.
    if "upstream_provider_unavailable" in codes:
        return MSG_UPSTREAM_UNAVAILABLE, "upstream_unavailable"
    return MSG_UNRESOLVED, None


class ServerRuntime:
    """컴포넌트 수명 + 단일 실행 스레드 + 예산. `from_env()` 로 만들고 `start()` 로 비동기 boot."""

    def __init__(self, *, budgets: Budgets | None = None, log_dir: Path | None = None,
                 use_hcx: bool = True, warm_fixtures: bool = False, fixture_fallback: bool = False,
                 assemble_native: bool = True, clarification_db_path: Path | None = None,
                 max_inflight: int = 1, capture_handoffs: bool = False):
        self.budgets = budgets or Budgets()
        self.log_dir = Path(log_dir or ROOT / "out" / "requests")
        self.use_hcx = use_hcx
        self.warm_fixtures = warm_fixtures
        self.fixture_fallback = fixture_fallback
        self.assemble_native = assemble_native
        self.clarification_db_path = Path(clarification_db_path or
                                          ROOT / "out" / "serving" / "stage1_v1_clarifications.sqlite3")
        self.started_at = datetime.now(timezone.utc)
        self.ready = False
        self.error: str | None = None
        # 컴포넌트 (install 에서 채움)
        self.backend = None
        self.pipeline = None             # HCX-005 문장화 (키 없으면 내부적으로 템플릿)
        self.pipeline_template = None    # 결정적 템플릿만
        self.stage1: ChainedStage1 | None = None
        self.stage1_native = None
        self.stage1_fixture = None
        self.canonical: dict = {}
        self.build: dict = {}
        self.stage1_info: dict = {}
        self.warm: dict = {}
        self.fixtures: list = []
        self.requirements: dict = {}
        self.capture_handoffs = capture_handoffs          # opt-in: 검증된 공개 handoff 만 replay 저장
        self.handoff_replay_store = None
        # 실행 스레드: boot 도 요청도 이 스레드에서만. 슬롯은 작업의 실제 수명 동안 점유(§원칙).
        self.max_inflight = max_inflight
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="canonical-exec")
        self._sema = threading.BoundedSemaphore(max_inflight)
        self._log_lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "ServerRuntime":
        env = os.getenv
        return cls(
            budgets=Budgets.from_env(),
            log_dir=Path(env("ANSWER_LOG_DIR", ROOT / "out" / "requests")),
            use_hcx=env("ANSWER_USE_HCX_COMPOSER", "1") == "1",
            # 전체 fixture 사전 실행은 메모리·부팅 시간을 키우므로 opt-in (4GB headroom)
            warm_fixtures=os.getenv("ANSWER_WARM_FIXTURES", "0") == "1",
            capture_handoffs=os.getenv("ANSWER_CAPTURE_HANDOFFS", "0") == "1",
            # 제출 서버는 native-only가 기본이다. 1은 오프라인 fixture 진단 서버에서만 명시한다.
            fixture_fallback=env("STAGE1_FIXTURE_FALLBACK", "0") == "1",
            assemble_native=env("STAGE1_NATIVE", "1") == "1",
            clarification_db_path=(Path(env("STAGE1_CLARIFICATION_DB_PATH"))
                                   if env("STAGE1_CLARIFICATION_DB_PATH") else None),
            # 실행 슬롯(admission) 수 — 정본 실행 자체는 여전히 워커 1개가 직렬 처리한다(§원칙).
            # 평가자 병렬 GET 이 곧바로 503 되지 않도록 기본을 2로 올린다(이슈 #55).
            max_inflight=_bounded_positive_env("SERVER_MAX_INFLIGHT", 2, upper=16),
        )

    # ── 수명 ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """boot 를 실행 스레드에 제출하고 즉시 돌아온다. 준비 전 요청은 `NotReady`/503."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.handoff_replay_store = self._make_handoff_replay_store()
        self._pool.submit(self._boot_guarded)

    def _boot_guarded(self) -> None:
        try:
            self.boot()
        except Exception as e:  # noqa
            self.error = _redact_repo_root(f"{type(e).__name__}: {e}")

    def boot(self) -> None:
        """정본 → ToolBackend → Stage2~4 파이프라인 → Stage1(native 우선, fixture 폴백) → 워밍업."""
        from app.tools import CanonicalToolBackend
        from app.tools.canonical_env import canonical_status
        from app.pipeline import AnswerPipeline
        from server.stage1 import FixtureStage1, NativeStage1

        canonical = canonical_status()
        backend = CanonicalToolBackend()
        pipeline = AnswerPipeline(backend, use_hcx=self.use_hcx)
        pipeline_template = AnswerPipeline(backend, use_hcx=False)

        t_s1 = time.time()
        native = None
        if self.assemble_native:
            native = NativeStage1(backend.rm, clarification_db_path=self.clarification_db_path)
        if not self.fixture_fallback and not (native and native.available):
            raise Stage1Unavailable(
                "native Stage1 is required when fixture fallback is disabled")
        fixture = FixtureStage1()
        stage1_info = {"native": bool(native and native.available),
                       "native_error": getattr(native, "error", None) if native else "STAGE1_NATIVE=0",
                       "prompt_version": getattr(native, "prompt_version", None) if native else None,
                       "fixture_fallback": self.fixture_fallback,
                       "assemble_s": round(time.time() - t_s1, 1)}
        self.install(backend=backend, pipeline=pipeline, pipeline_template=pipeline_template,
                     stage1_native=native, stage1_fixture=fixture, canonical=canonical,
                     stage1_info=stage1_info)
        self._warm_up()
        self.build = self._build_info()
        self.ready = True

    def install(self, *, backend, pipeline, pipeline_template=None, stage1_native=None, stage1_fixture=None,
                canonical: dict | None = None, stage1_info: dict | None = None,
                fixtures: list | None = None, requirements: dict | None = None) -> None:
        """컴포넌트 주입. boot 가 쓰고, 테스트는 정본·HCX 없이 가짜 컴포넌트로 직접 부른다."""
        from app.orchestrator import load_answer_requirements, load_handoffs
        self.backend = backend
        self.pipeline = pipeline
        self.pipeline_template = pipeline_template or pipeline
        self.stage1_native = stage1_native
        self.stage1_fixture = stage1_fixture
        self.stage1 = ChainedStage1(stage1_native, stage1_fixture, fallback_enabled=self.fixture_fallback)
        self.canonical = canonical or {}
        self.stage1_info = stage1_info or {}
        self.fixtures = fixtures if fixtures is not None else load_handoffs()
        self.requirements = requirements if requirements is not None else load_answer_requirements()
        if not self.build:
            self.build = self._build_info()

    def mark_ready(self) -> None:
        self.ready, self.error = True, None

    def _warm_up(self) -> None:
        """검색 인덱스 대표 point query 만 수행한다. Field 전량 `load_all` 과 재무 lookup 선로드는 첫 회사의
        Fact cache 를 기동 기준선에 상주시켜 4GB headroom 을 잠식하므로 하지 않는다 — 실제 요청까지 미룬다.
        release fixture 사전 실행은 `warm_fixtures`(ANSWER_WARM_FIXTURES=1) opt-in."""
        try:
            t0 = time.time()
            backend = self.backend
            if backend.search_index is not None:
                backend.search_index.search("투자 계획", as_of="20260619", top_k=1)
            if self.warm_fixtures:
                n = 0
                for rec in self.fixtures:
                    if rec.handoff.status == "ready":
                        try:
                            self.pipeline.orch.run(rec.handoff, question_id=rec.question_id); n += 1
                        except Exception:
                            pass
                self.warm["questions"] = n
            self.warm["seconds"] = round(time.time() - t0, 1)
        except Exception as e:  # noqa
            self.warm["error"] = f"{type(e).__name__}: {e}"

    def _build_info(self) -> dict:
        si = getattr(self.backend, "search_index", None)
        composer = getattr(self.pipeline, "composer", None)
        return {"canonical_build_id": self.canonical.get("build_id"), "schema": self.canonical.get("schema_version"),
                "warmup_s": self.warm.get("seconds"),
                "search_index": getattr(si, "index_build_id", None),
                "hcx_composer": bool(composer.enabled) if composer is not None else False,
                "stage1": self.stage1_info}

    def close(self) -> None:
        native = self.stage1_native
        if native is not None and callable(getattr(native, "close", None)):
            try:
                native.close()
            except Exception:
                pass
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ── 실행 스레드 ────────────────────────────────────────────────────────

    def submit(self, fn: Callable, *args, timeout: float | None = None, started: float | None = None,
               release_memory: bool = True, **kwargs):
        """`fn` 을 실행 스레드에서 돌리고 결과를 기다린다.

        큐 대기는 `budgets.queue_wait_s`(`SERVER_QUEUE_WAIT_S`, 기본 30초). 실행 타임아웃은 `timeout`(기본 EXEC) 이되, `started`
        (요청 수신 시각)가 주어지면 큐 대기를 뺀 잔여 예산으로 줄여 queue+exec ≤ HARD 를 지킨다.
        슬롯은 `_execute_holding_slot` 이 작업의 실제 수명 동안 쥔다. Arrow 임시 메모리 반환은 **실행 스레드
        안에서** `fn` 이 끝난 직후에 한다(timeout 뒤 늦게 끝나는 작업도 그때) — 정본을 만지는 스레드 밖에서
        allocator 를 건드리지 않는다.
        """
        if not self.ready:
            raise NotReady(self.error or "loading")
        if not self._sema.acquire(timeout=self.budgets.queue_wait_s):
            raise QueueTimeout("queue timeout")
        if started is not None:
            exec_timeout = self.budgets.remaining_exec_s(started)
        else:
            exec_timeout = timeout if timeout is not None else self.budgets.exec_s
        deadline_monotonic = time.monotonic() + exec_timeout
        try:
            # `_run_then_release` receives the deadline before the original
            # positional arguments.  Keep this ordering explicit: answer()
            # submits `(question_id, question)` and placing the deadline
            # after `*args` would silently install the question ID as the
            # ContextVar deadline in the worker.
            return self._execute_holding_slot(
                _run_then_release, fn, release_memory, deadline_monotonic, *args,
                timeout=exec_timeout, **kwargs)
        except FutTimeout as e:
            raise ExecTimeout(f"exec budget {exec_timeout}s exceeded") from e
        except RequestDeadlineExceeded as e:
            # A tool loop noticed the same absolute budget before the outer
            # Future wait elapsed.  It has returned from the one canonical
            # worker, so `_execute_holding_slot` has already released the
            # slot; expose the ordinary timeout result rather than an error.
            raise ExecTimeout(f"exec budget {exec_timeout}s exceeded") from e

    def _execute_holding_slot(self, fn: Callable, *args, timeout: float, **kwargs):
        """호출자가 이미 슬롯 하나를 쥔 상태에서 `fn` 을 실행 스레드에 제출하고 기다린다.

        정상·예외 종료면 여기서 정확히 한 번 슬롯을 돌려준다. timeout 이면 HTTP 는 돌아가지만 워커는
        계속 돌 수 있으므로, 완료 콜백이 슬롯을 돌려준다 — timeout 된 워커가 후속 요청과 동시에 정본을
        만지거나 작업을 쌓는 일을 막는다.
        """
        pool, slot = self._pool, self._sema       # 재로드·테스트가 속성을 바꿔도 같은 객체에 반환
        try:
            future: Future = pool.submit(fn, *args, **kwargs)
        except Exception:
            slot.release()
            raise
        try:
            result = future.result(timeout=timeout)
        except FutTimeout:
            # cancel() 전에 등록: 아직 안 도는 Future 는 cancel 성공 시 콜백을 동기 호출, 도는 Future 는
            # 실제 완료 시 호출 — 어느 쪽이든 정확히 한 번 반환된다.
            future.add_done_callback(lambda _, owned_slot=slot: owned_slot.release())
            future.cancel()
            raise
        except BaseException:
            slot.release()
            raise
        slot.release()
        return result

    # ── handoff replay 저장(opt-in) ────────────────────────────────────────

    def _make_handoff_replay_store(self):
        """검증된 공개 handoff 만 보존하는 replay 저장소. 요청·provider 데이터는 읽지 않는다."""
        if not self.capture_handoffs:
            return None
        from server.handoff_replay import HandoffReplayStore
        out_root = (ROOT / "out").resolve()
        directory = Path(os.getenv("ANSWER_HANDOFF_CAPTURE_DIR", "out/requests/handoff_replay"))
        if not directory.is_absolute():
            directory = ROOT / directory
        directory = directory.resolve()
        try:
            directory.relative_to(out_root)
        except ValueError as exc:
            raise ValueError("ANSWER_HANDOFF_CAPTURE_DIR must be under ROOT/out") from exc
        return HandoffReplayStore(
            directory,
            max_record_bytes=_bounded_positive_env("ANSWER_HANDOFF_CAPTURE_MAX_RECORD_BYTES", 256 * 1024,
                                                   upper=1024 * 1024),
            max_file_bytes=_bounded_positive_env("ANSWER_HANDOFF_CAPTURE_MAX_FILE_BYTES", 8 * 1024 * 1024,
                                                 upper=64 * 1024 * 1024),
            max_files=_bounded_positive_env("ANSWER_HANDOFF_CAPTURE_MAX_FILES", 4, upper=32),
        )

    def _capture_handoff(self, handoff) -> dict[str, str]:
        """Stage2 시작 전 검증된 공개 handoff 만 저장한다. 질문 원문·provider wire·키·세션 ID 는 받지 않는다.
        opt-in 상태에서 기록·무결성 실패는 실행을 중단한다 — 빠진 replay 증거가 완전한 실행으로 오인되지 않게."""
        store = self.handoff_replay_store
        if store is None:
            return {}
        captured = store.capture(handoff, canonical_build_id=self.build.get("canonical_build_id"))
        return {"handoff_capture_id": captured.capture_id, "handoff_sha256": captured.handoff_sha256}

    # ── question → 답변 (공식 /answer 와 /v1/answer 공용) ───────────────────

    def answer(self, question_id: str, question: str, *, stage1_mode: Stage1Mode = "auto",
               compose: ComposeMode = "auto") -> AnswerResult:
        """예산·폴백 정책까지 적용된 한 번의 답변. 어떤 경우에도 5필드를 돌려준다."""
        if not self.ready:
            return AnswerResult(limit_response(question_id, question, MSG_NOT_READY,
                                               f"[server] not ready: {self.error or 'loading'}"),
                                "not_ready", 503, meta={"error": self.error or "loading"})
        started = time.time()
        try:
            return self.submit(self._answer_sync, question_id, question,
                               stage1_mode=stage1_mode, compose=compose, started=started)
        except QueueTimeout:
            return AnswerResult(limit_response(question_id, question, MSG_QUEUE, "[server] queue timeout"),
                                "queue_timeout", 503)
        except ExecTimeout:
            return AnswerResult(limit_response(question_id, question, MSG_TIMEOUT,
                                               f"[server] exec budget {self.budgets.exec_s}s exceeded"),
                                "timeout", 200, meta={"final_status": "timeout"})
        except Exception as e:  # noqa
            return AnswerResult(limit_response(question_id, question, MSG_ERROR, f"[server] error {type(e).__name__}"),
                                "error", 200, meta={"final_status": "error",
                                                    "error": f"{type(e).__name__}: {str(e)[:200]}"})

    def _stage1_for(self, mode: Stage1Mode):
        if mode == "auto":
            return self.stage1
        if mode == "native":
            primary = self.stage1_native or _UnavailableStage1("stage1_v1_native", "native stage1 not assembled",
                                                                code="native_not_assembled")
            return ChainedStage1(primary, None, fallback_enabled=False)
        if mode == "fixture":
            self.require_fixture_access()
            fb = self.stage1_fixture or _UnavailableStage1("fixture", "fixture stage1 not loaded", code="fixture_not_loaded")
            return ChainedStage1(None, fb, fallback_enabled=True)
        raise ValueError(f"unknown stage1 mode: {mode}")

    def require_fixture_access(self) -> None:
        """제출 기본값에서는 fixture 재생/직접 실행 경로도 닫는다.

        자동 폴백만 끄고 ``/v1``의 명시적 fixture 모드를 남기면 제출 서버에서
        동결 질문을 우회 실행할 수 있다. 오프라인 진단 서버는 환경변수를 1로
        명시해 두 경로를 함께 opt-in 한다.
        """
        if not self.fixture_fallback:
            raise Stage1Unavailable(
                "fixture access is disabled; set STAGE1_FIXTURE_FALLBACK=1 only for offline diagnostics")

    def _pipeline_for(self, compose: ComposeMode):
        return self.pipeline if compose == "auto" else self.pipeline_template

    #: 재표본 한 번을 시도하기 위해 남아 있어야 하는 최소 여유(초).
    #: 이보다 적게 남으면 두 번째 호출이 예산 안에 끝날 보장이 없다.
    _STAGE1_RESAMPLE_MIN_S = 12.0

    def _resample_stage1(self, question: str, *, question_id: str,
                         stage1_mode: Stage1Mode,
                         deadline_monotonic: float | None,
                         stage1_meta: Mapping[str, Any] | None = None):
        """계획을 못 세운 Stage1 을 **한 번만** 다시 표본한다.

        HCX-007 의 structured output 은 호출마다 표본이 달라진다. 같은 질문이
        한 표본에서는 실행계획으로 풀리고 다음 표본에서는 풀리지 않는다
        (이슈 #19). 지금까지의 전체 실행은 매번 정확히 한 문항을 이 변동으로
        잃었고, `R-P-008` 을 3회 호출하면 2회는 정상 답변이 나왔다. 즉
        **무계획이라는 결과가 곧 답할 수 없는 질문이라는 뜻은 아니다.**

        첫 표본이 계획을 세우면 이 경로에 오지 않으므로 정상 응답의 추가
        호출은 0이다. 모호성·전제 위반·지원 밖 의미는 handoff 를 만든 뒤
        fail-closed 로 걸러지므로 여기로 내려오지 않는다.
        """

        now = time.monotonic()
        available = (self.budgets.stage1_s if deadline_monotonic is None
                     else deadline_monotonic - now)
        if available < self._STAGE1_RESAMPLE_MIN_S:
            return None
        stage1_deadline = now + min(self.budgets.stage1_s, available)
        if deadline_monotonic is not None:
            stage1_deadline = min(stage1_deadline, deadline_monotonic)
        overrides = stage1_resample_overrides(stage1_meta)
        runner = getattr(getattr(getattr(self, "stage1_native", None), "service", None),
                         "runner", None)
        applied = overrides is not None and runner is not None \
            and hasattr(runner, "resample_overrides")
        if applied:
            runner.resample_overrides = overrides
        try:
            with latency_span("stage1_resolve", attempt=2, seed_overridden=applied):
                retry = self._stage1_for(stage1_mode).resolve(
                    question, question_id=question_id,
                    deadline_monotonic=stage1_deadline)
            record_latency_event("stage1_resolve", attempt=2,
                                 outcome=getattr(getattr(retry, "handoff", None), "status", "no_handoff"))
        finally:
            if applied:
                runner.resample_overrides = None
        if applied and retry is not None and isinstance(getattr(retry, "meta", None), dict):
            retry.meta["resample_overrides"] = {"seed": overrides["seed"], "retry_hint": True}
        return None if _stage1_yielded_no_answer(retry) else retry

    def _answer_sync(self, question_id: str, question: str, *, stage1_mode: Stage1Mode, compose: ComposeMode) -> AnswerResult:
        if os.getenv("ANSWER_CAPTURE_LATENCY", "0") != "1":
            return ServerRuntime._answer_sync_impl(
                self, question_id, question, stage1_mode=stage1_mode, compose=compose)
        with capture_latency() as events:
            result = ServerRuntime._answer_sync_impl(
                self, question_id, question, stage1_mode=stage1_mode, compose=compose)
            result.meta["latency_profile"] = {
                "schema": "request-latency/1", "events": events,
                "provider_http_attempts": sum(e["stage"] == "provider_http" for e in events),
            }
            return result

    def _answer_sync_impl(self, question_id: str, question: str, *, stage1_mode: Stage1Mode, compose: ComposeMode) -> AnswerResult:
        from app.disclosure_rules import explain_disclosure_rule

        rule_answer = explain_disclosure_rule(question)
        if rule_answer is not None and stage1_mode != "fixture":
            return AnswerResult(
                limit_response(question_id, question, rule_answer,
                               "[server] disclosure display rule; no lookup or model call"),
                "ok", 200, meta={"final_status": "answer", "stage1": "display_rules",
                                 "t_stage1": 0.0, "t_pipeline": 0.0})
        t0 = time.time()
        deadline = current_deadline_monotonic()
        stage1_deadline = time.monotonic() + self.budgets.stage1_s
        if deadline is not None:
            stage1_deadline = min(stage1_deadline, deadline)
        with latency_span("stage1_resolve", attempt=1):
            res = self._stage1_for(stage1_mode).resolve(
                question, question_id=question_id, deadline_monotonic=stage1_deadline)
        record_latency_event("stage1_resolve", attempt=1,
                             outcome=getattr(getattr(res, "handoff", None), "status", "no_handoff"))
        resampled = False
        if _stage1_yielded_no_answer(res):
            retry = self._resample_stage1(
                question, question_id=question_id, stage1_mode=stage1_mode,
                deadline_monotonic=deadline, stage1_meta=getattr(res, "meta", None))
            if retry is not None:
                res, resampled = retry, True
        t1 = time.time()
        if res.handoff is None:
            trace = stage1_trace_line(res, t1 - t0) + " — 질문을 실행 계획으로 변환하지 못함"
            # #110/#165: question_id 계약 위반과 provider 전송 실패를
            # 일반 해석 실패와 분리해 사용자가 질문을 고쳐야 하는지,
            # 잠시 후 다시 요청해야 하는지를 알 수 있게 한다.
            msg, error_code = _stage1_unresolved_message(res)
            meta = {"stage1": res.source, "stage1_meta": res.meta, "final_status": "unresolved",
                    "stage1_resampled": resampled, "t_stage1": round(t1 - t0, 3)}
            if error_code is not None:
                meta["error_code"] = error_code
            return AnswerResult(limit_response(question_id, question, msg, trace), "unresolved", 200,
                                meta=meta, stage1=res)
        return self._run_handoff(res, question_id=question_id, question=question, compose=compose,
                                 t_stage1=t1 - t0, deadline_monotonic=deadline,
                                 extra_meta={"stage1_resampled": True} if resampled else None)

    def _run_handoff(self, res: Stage1Resolution, *, question_id: str, question: str, compose: ComposeMode,
                     t_stage1: float, extra_meta: dict | None = None,
                     deadline_monotonic: float | None = None) -> AnswerResult:
        """Stage1 결과(시작·재개 turn 공통) → Stage2~4 실행 → 5필드 + meta. handoff 는 있어야 한다."""
        if deadline_monotonic is None:
            deadline_monotonic = current_deadline_monotonic()
        capture_meta = self._capture_handoff(res.handoff)
        resp, payload, meta = self._execute_sync(res.handoff, question_id=question_id, question=question,
                                                 compose=compose,
                                                 runtime_annotations=res.meta.get(
                                                     "runtime_annotations"),
                                                 deadline_monotonic=deadline_monotonic)
        self._apply_native_clarification(resp, res)
        resp["think_trace"] = stage1_trace_line(res, t_stage1) + "\n" + resp["think_trace"]
        meta = {"stage1": res.source, "matched": getattr(res, "question_id_hint", None), "stage1_meta": res.meta,
                "handoff_status": res.handoff.status, "t_stage1": round(t_stage1, 3),
                **(extra_meta or {}), **meta, **capture_meta}
        if getattr(res, "session", None):
            meta["session"] = res.session
        return AnswerResult(resp, "ok", 200, meta=meta, payload=payload, stage1=res)

    @staticmethod
    def _apply_native_clarification(resp: dict, res: Stage1Resolution) -> None:
        """native 역질문은 v0.4 handoff 가 option `value` 만 싣고 `label`·prompt 를 잃어 composer 만으로는
        계약 후보가 접수번호로만 보인다. Stage1 renderer 가 만든 문장을 answer 로 쓴다(5필드·handoff·
        resume 헤더는 그대로). fixture 경로엔 메시지가 없어 composer 출력을 유지한다."""
        msg = res.meta.get("clarification_message")
        if (res.handoff is not None and res.handoff.status == "needs_clarification"
                and isinstance(msg, str) and msg.strip()):
            resp["answer"] = msg.strip()
            resp["think_trace"] += "\n[8 답변생성] native typed clarification renderer"

    # ── Stage1 단독 ────────────────────────────────────────────────────────

    def interpret(self, question_id: str, question: str, *, stage1_mode: Stage1Mode = "auto") -> Stage1Resolution:
        """question → QueryPlanHandoff v0.4 (Stage2 실행 없음). 역질문이면 세션 식별자를 함께 돌려준다."""
        return self.submit(self._interpret_sync, question_id, question, stage1_mode=stage1_mode,
                           timeout=self.budgets.stage1_s + 5)

    def _interpret_sync(self, question_id: str, question: str, *, stage1_mode: Stage1Mode) -> Stage1Resolution:
        deadline = current_deadline_monotonic()
        stage1_deadline = time.monotonic() + self.budgets.stage1_s
        if deadline is not None:
            stage1_deadline = min(stage1_deadline, deadline)
        return self._stage1_for(stage1_mode).resolve(
            question, question_id=question_id, deadline_monotonic=stage1_deadline)

    def clarify(self, answer: Mapping[str, Any], *, execute: bool = False, question_id: str | None = None,
                question: str = "", compose: ComposeMode = "auto") -> tuple[Stage1Resolution, AnswerResult | None]:
        """열린 역질문 세션 재개 (HCX 재호출 없음) → 필요하면 같은 스레드 작업 안에서 Stage2~4 까지.

        typed 예외(세션 없음·revision 불일치·후보 밖 값·turn 한도)는 그대로 올린다 — 라우터가 HTTP 코드로 옮긴다.
        재개 결과가 아직 역질문이면 Stage2 는 clarify payload(역질문 문장)를, ready 면 확정 답변을 만든다.
        """
        return self.submit(self._clarify_sync, answer, execute=execute, question_id=question_id,
                           question=question, compose=compose, timeout=self.budgets.exec_s)

    def _clarify_sync(self, answer: Mapping[str, Any], *, execute: bool, question_id: str | None,
                      question: str, compose: ComposeMode) -> tuple[Stage1Resolution, AnswerResult | None]:
        t0 = time.time()
        res = self._require_stage1().answer_clarification(answer)
        t1 = time.time()
        if not execute or res.handoff is None:
            return res, None
        qid = question_id or f"session-{str(answer.get('session_id', ''))[:8]}"
        return res, self._run_handoff(res, question_id=qid, question=question, compose=compose, t_stage1=t1 - t0)

    # ── 공식 POST /answer/resume ───────────────────────────────────────────

    def resume(self, *, session_id: str, clarification_id: str, revision: int, answers: Mapping[str, str],
               action: str = "submit", compose: ComposeMode = "auto") -> AnswerResult:
        """열린 **native** 세션에 typed 답을 적용하고(HCX 재호출 없음) Stage2~4 까지 — fixture 폴백은 없다.

        typed 세션 예외(세션 없음·revision 불일치·후보 밖 값·turn 한도)와 NotReady/QueueTimeout/ExecTimeout
        은 그대로 올린다. 라우터가 HTTP 코드로 옮긴다.
        """
        started = time.time()
        return self.submit(self._resume_sync, session_id=session_id, clarification_id=clarification_id,
                           revision=revision, answers=dict(answers), action=action, compose=compose,
                           started=started)

    def _resume_sync(self, *, session_id: str, clarification_id: str, revision: int, answers: dict,
                     action: str, compose: ComposeMode) -> AnswerResult:
        t0 = time.time()
        deadline = current_deadline_monotonic()
        stage1_deadline = time.monotonic() + self.budgets.stage1_s
        if deadline is not None:
            stage1_deadline = min(stage1_deadline, deadline)
        resumed = self._require_stage1().resume(
            session_id=session_id, clarification_id=clarification_id, revision=revision, answers=answers,
            action=action, deadline_monotonic=stage1_deadline)
        res = resumed.resolution
        t1 = time.time()
        qid, question = resumed.question_id, resumed.question
        if res.handoff is None:
            trace = stage1_trace_line(res, t1 - t0) + " — 재개 결과를 실행 계획으로 변환하지 못함"
            return AnswerResult(limit_response(qid, question, MSG_UNRESOLVED, trace), "unresolved", 200,
                                meta={"stage1": res.source, "stage1_meta": res.meta, "final_status": "unresolved",
                                      "resumed": True, "t_stage1": round(t1 - t0, 3)}, stage1=res)
        return self._run_handoff(res, question_id=qid, question=question, compose=compose, t_stage1=t1 - t0,
                                 extra_meta={"resumed": True})

    def inspect_session(self, session_id: str) -> Stage1Resolution:
        return self.submit(self._require_stage1().inspect_session, session_id, timeout=self.budgets.tool_s)

    def _require_stage1(self) -> ChainedStage1:
        if self.stage1 is None:
            raise Stage1Unavailable("stage1 not installed")
        return self.stage1

    # ── Stage2~4 단독 ──────────────────────────────────────────────────────

    def execute(self, handoff, *, question_id: str, question: str = "", compose: ComposeMode = "auto") -> tuple[dict, Any, dict]:
        """QueryPlanHandoff → (5필드, AnswerPayload, meta). fixture handoff 재생·Stage1 우회 디버깅용."""
        return self.submit(self._execute_sync, handoff, question_id=question_id, question=question,
                           compose=compose, runtime_annotations=None,
                           timeout=self.budgets.exec_s)

    def _execute_sync(
            self, handoff, *, question_id: str, question: str,
            compose: ComposeMode, runtime_annotations: dict | None = None,
            deadline_monotonic: float | None = None,
            ) -> tuple[dict, Any, dict]:
        t0 = time.time()
        # `/v1/stage2/execute` enters here without explicitly passing the
        # argument.  It still runs under `submit()`'s request ContextVar, so
        # recover that absolute deadline before delegating to HCX composer or
        # cooperative backends.
        if deadline_monotonic is None:
            deadline_monotonic = current_deadline_monotonic()
        run_kwargs = {"question_id": question_id, "question": question}
        if runtime_annotations is not None:
            run_kwargs["runtime_annotations"] = runtime_annotations
        resp, payload = self._pipeline_for(compose).run(
            handoff, **run_kwargs, deadline_monotonic=deadline_monotonic)
        meta = {"final_status": payload.final_status, "claims": len(payload.claims),
                "limitations": [l.code for l in payload.limitations],
                "used_documents": list(getattr(payload, "used_documents", []) or []),
                "compose": compose, "t_pipeline": round(time.time() - t0, 3)}
        return resp, payload, meta

    def run_fixture(self, question_id: str, *, compose: ComposeMode = "template") -> tuple[dict, Any, dict, Any]:
        """release fixture 한 문항을 Stage2 로 실행하고 AnswerRequirement 로 채점한다 → (5필드, payload, meta, ScoreCard)."""
        self.require_fixture_access()
        return self.submit(self._run_fixture_sync, question_id, compose=compose, timeout=self.budgets.exec_s)

    def _run_fixture_sync(self, question_id: str, *, compose: ComposeMode):
        from app.evals.scorer import score_answer_text, score_payload
        rec = self.fixture(question_id)
        req = self.requirements.get(question_id)
        resp, payload, meta = self._execute_sync(rec.handoff, question_id=question_id,
                                                 question=rec.question or "", compose=compose)
        card = None
        if req is not None:
            card = score_payload(payload, req)
            card.checks.extend(score_answer_text(resp["answer"], req, payload))
        return resp, payload, meta, card

    def fixture(self, question_id: str):
        self.require_fixture_access()
        for rec in self.fixtures:
            if rec.question_id == question_id:
                return rec
        raise KeyError(question_id)

    # ── 정본 도구 (read.py safe 경로) ──────────────────────────────────────

    def tool(self, fn: Callable, *args, **kwargs):
        """read model / 검색 인덱스 단건 호출을 실행 스레드에서. ValueError 는 그대로 (422 로 매핑)."""
        return self.submit(fn, *args, timeout=self.budgets.tool_s, release_memory=False, **kwargs)

    @property
    def rm(self):
        if self.backend is None:
            raise NotReady(self.error or "canonical backend not installed")
        return self.backend.rm

    @property
    def search_index(self):
        si = getattr(self.backend, "search_index", None)
        if si is None:
            raise SearchIndexUnavailable("검색 인덱스가 로드되지 않았습니다 — make search-index")
        return si

    # ── 진행 현황 ──────────────────────────────────────────────────────────

    def status(self) -> dict:
        """정본·인덱스·Stage1·composer·채점표·요청 로그·역질문 세션을 한 화면에. 정본 스레드를 쓰지 않는다."""
        now = datetime.now(timezone.utc)
        composer = getattr(self.pipeline, "composer", None)
        si = getattr(self.backend, "search_index", None)
        return {
            "service": {"name": SERVICE_NAME, "version": SERVICE_VERSION, "ready": self.ready, "error": self.error,
                        "started_at": self.started_at.isoformat(),
                        "uptime_s": round((now - self.started_at).total_seconds(), 1),
                        "budgets": {"exec_s": self.budgets.exec_s, "hard_s": self.budgets.hard_s,
                                    "stage1_s": self.budgets.stage1_s, "tool_s": self.budgets.tool_s,
                                    "queue_wait_s": self.budgets.queue_wait_s},
                        "max_inflight": self.max_inflight},
            "canonical": {"build_id": self.canonical.get("build_id"),
                          "schema_version": self.canonical.get("schema_version"),
                          "counts": self.canonical.get("counts", {})},
            "search_index": {"index_build_id": getattr(si, "index_build_id", None), "loaded": si is not None},
            "stage1": {**self.stage1_info, "fixture_questions": len(self.fixtures),
                       "modes": (["auto", "native", "fixture"] if self.fixture_fallback
                                 else ["auto", "native"])},
            "composer": {"hcx_enabled": bool(composer.enabled) if composer is not None else False,
                         "model": getattr(getattr(composer, "client", None), "cfg", None).model
                         if composer is not None and getattr(composer, "client", None) is not None else None},
            "warmup": dict(self.warm),
            "evaluation": {"harness": _read_score(ROOT / "out" / "score.json"),
                           "harness_compose": _read_score(ROOT / "out" / "score_compose.json")},
            "requests": _request_stats(self.log_dir, now.strftime("%Y%m%d")),
            "clarification_sessions": _session_stats(self.clarification_db_path),
        }

    # ── 요청 로그 ──────────────────────────────────────────────────────────

    def log(self, rec: dict) -> None:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **rec}, ensure_ascii=False, default=str)
        with self._log_lock:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_dir / f"answers_{day}.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def recent_requests(self, *, limit: int = 50, date: str | None = None) -> list[dict]:
        day = date or datetime.now(timezone.utc).strftime("%Y%m%d")
        path = self.log_dir / f"answers_{day}.jsonl"
        if not path.exists():
            return []
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return rows[-limit:][::-1]


# ── 메모리·환경 보조 ────────────────────────────────────────────────────────────

def _run_then_release(fn: Callable, release_memory: bool, deadline_monotonic: float, *args, **kwargs):
    """실행 스레드 안에서 `fn` 을 돌리고, 끝나면(예외·timeout 뒤 늦은 완료 포함) 같은 스레드에서 Arrow 임시
    메모리를 반환한다."""
    with request_deadline(deadline_monotonic):
        try:
            return fn(*args, **kwargs)
        finally:
            if release_memory:
                release_transient_query_memory()


def release_transient_query_memory() -> None:
    """완료된 요청의 Arrow 임시 메모리를 다음 요청 전에 반환한다.

    narrative 섹션 읽기는 요청 중 수 GiB 를 잡았다 풀 수 있는데, Arrow allocator 가 그 페이지를 쥐고 있으면
    4GB 단일 워커가 영구히 가득 찬 것처럼 보인다. dataset/footer 와 bounded typed cache 는 그대로 두고,
    참조가 끊긴 allocator 페이지만 돌려준다.
    """
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except (ImportError, AttributeError):
        pass    # Arrow 가 없거나 release_unused 가 없는 최소 테스트 환경


def _bounded_positive_env(name: str, default: int, *, upper: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not 1 <= value <= upper:
        raise ValueError(f"{name} must be between 1 and {upper}")
    return value


# ── 진행 현황 보조 (파일만 읽는다) ──────────────────────────────────────────────

def _read_score(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {"path": str(path.relative_to(ROOT)), "backend": d.get("backend"),
            "total": d.get("total"), "passed": d.get("passed"), "score_pct": d.get("score_pct"),
            "by_group": d.get("by_group", {}), "failures": len(d.get("failures", [])),
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()}


def _request_stats(log_dir: Path, day: str) -> dict:
    path = log_dir / f"answers_{day}.jsonl"
    out: dict = {"date": day, "total": 0, "by_status": {}, "p50_s": None, "p95_s": None}
    if not path.exists():
        return out
    elapsed: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        out["total"] += 1
        st = str(r.get("status", "?"))
        out["by_status"][st] = out["by_status"].get(st, 0) + 1
        if isinstance(r.get("elapsed"), (int, float)):
            elapsed.append(float(r["elapsed"]))
    if elapsed:
        ts = sorted(elapsed)
        out["p50_s"] = round(statistics.median(ts), 3)
        out["p95_s"] = round(ts[min(len(ts) - 1, int(len(ts) * 0.95))], 3)
    return out


def _session_stats(db_path: Path) -> dict:
    out: dict = {"db_path": str(db_path), "count": None}
    if not db_path.exists():
        return out
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
        try:
            out["count"] = con.execute("SELECT COUNT(*) FROM clarification_sessions").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error as e:
        out["error"] = str(e)[:100]
    return out


__all__ = ["ServerRuntime", "Budgets", "AnswerResult", "NotReady", "QueueTimeout", "ExecTimeout",
           "SearchIndexUnavailable", "Stage1Unavailable", "limit_response", "stage1_trace_line",
           "release_transient_query_memory", "SERVICE_NAME", "SERVICE_VERSION"]
