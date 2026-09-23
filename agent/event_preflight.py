"""Event status/timeline용 제한적 canonical event-key preflight."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from threading import RLock
from typing import Iterable, Literal, Protocol
import unicodedata
import tempfile

from src.canonical.events import event_support_role


EventResolutionStatus = Literal["resolved", "ambiguous", "not_found", "too_many"]
_MAX_COMPANY_ROLE_ROWS = 100_000
_MAX_OPTIONS = 20
_MAX_CACHE_BYTES = 64 * 1024 * 1024
EVENT_ROLE_INDEX_VERSION = "event-role-index/0.1"
# 0.2 — 코스닥 단일판매ㆍ공급계약 서식의 계약명 라벨을 어휘에 넣었다.
# 어휘가 바뀌면 캐시된 역할 색인이 옛 분류를 그대로 쓰므로 여기서 올린다.
EVENT_ROLE_POLICY_VERSION = "event-role-policy/0.2"


class EventRoleIndexError(RuntimeError):
    """persistent event role index가 현재 canonical 계약과 다르다."""


def _key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^0-9a-z가-힣]+", "", normalized)


@dataclass(frozen=True, slots=True)
class EventKeyCandidate:
    """사건 후보 하나와 **정본이 내려 둔 동일성 판정**.

    `identity_fingerprint` 는 전처리가 「이 공시들은 같은 계약으로 보인다」고
    적어 둔 값이다. 후보 전원이 같은 fingerprint 를 가지면 모호한 것은 계약이
    아니라 **어느 원본 공시에서 유래했는가**뿐이다.

    `observation_count` 는 **반드시 as_of 로 다시 계산된 관측 수**여야 한다.
    정본 identity 표의 `n_observations` 는 코퍼스 끝 기준이라 과거 시점 질문에
    미래의 해지 사실을 흘린다(`src/canonical/read.py::event_timeline` 참고).
    """

    event_key: str
    seed_rcept_no: str
    label: str
    identity_fingerprint: str | None = None
    identity_status: str | None = None
    observation_count: int = 0
    # An exact filing question can select a correction/termination observation
    # while the event identity remains anchored at an earlier root receipt.
    # Keep the two coordinates separate.  This field is resolver-internal; the
    # public v0.4 plan carries it through the existing ``seed_rcept_no`` slot.
    selected_observation_rcept_no: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{32}", self.event_key):
            raise ValueError("event_key 형식이 잘못되었습니다")
        if not re.fullmatch(r"[0-9]{14}", self.seed_rcept_no):
            raise ValueError("event seed 접수번호 형식이 잘못되었습니다")
        if not self.label.strip():
            raise ValueError("event candidate label은 비어 있을 수 없습니다")
        if (self.identity_fingerprint is not None
                and not re.fullmatch(r"[0-9a-f]{32}", self.identity_fingerprint)):
            raise ValueError("identity_fingerprint 형식이 잘못되었습니다")
        if type(self.observation_count) is not int or self.observation_count < 0:
            raise ValueError("observation_count는 0 이상의 정수여야 합니다")
        if (self.selected_observation_rcept_no is not None
                and not re.fullmatch(
                    r"[0-9]{14}", self.selected_observation_rcept_no)):
            raise ValueError("selected event observation 접수번호 형식이 잘못되었습니다")


@dataclass(frozen=True, slots=True)
class EventKeyResolution:
    status: EventResolutionStatus
    candidates: tuple[EventKeyCandidate, ...] = ()

    def __post_init__(self) -> None:
        keys = [(row.event_key, row.seed_rcept_no) for row in self.candidates]
        if keys != sorted(set(keys)):
            raise ValueError("event 후보는 key/receipt 기준 정렬·고유해야 합니다")
        event_keys = [row.event_key for row in self.candidates]
        if len(event_keys) != len(set(event_keys)):
            raise ValueError("event 후보는 사건 key당 하나여야 합니다")
        if self.status == "resolved" and len(self.candidates) != 1:
            raise ValueError("resolved event preflight에는 후보 하나가 필요합니다")
        if self.status == "ambiguous" and len(self.candidates) < 2:
            raise ValueError("ambiguous event preflight에는 후보 둘 이상이 필요합니다")
        if self.status in {"not_found", "too_many"} and self.candidates:
            raise ValueError("후보 없는 상태에 event candidate를 넣을 수 없습니다")


class EventKeyPreflight(Protocol):
    def names_a_single_contract(
            self, *, corp_code: str, surface: str,
            ) -> bool: ...

    def receipt_names_contract(
            self, *, corp_code: str, rcept_no: str, surface: str,
            ) -> bool: ...

    def agreeing_candidate(
            self, *, candidates: "tuple[EventKeyCandidate, ...]",
            as_of: str, slots: "tuple[str, ...]",
            ) -> "EventKeyCandidate | None": ...

    def resolve_event_key(
            self, *, corp_code: str, as_of: str,
            seed_rcept_no: str | None = None,
            event_type: str | None = None,
            counterparty: str | None = None,
            contract_name: str | None = None,
            event_from: str | None = None,
            event_to: str | None = None,
            ) -> EventKeyResolution: ...


class EventFieldLike(Protocol):
    corp_code: str
    rcept_no: str
    rcept_dt: str
    event_type: str | None
    path: str
    value_status: str
    is_pii: bool

    @property
    def value(self) -> str | None: ...


class EventTimelineLike(Protocol):
    event_key: str
    corp_code: str
    root_rcept_no: str


class EventCanonicalLike(Protocol):
    def fields(self, **kwargs: object) -> Iterable[EventFieldLike]: ...

    def event_timeline(self, **kwargs: object) -> EventTimelineLike | None: ...

    # CanonicalReadModel이 제공하는 build-bound metadata. 테스트용 reader는
    # 이 metadata가 없으면 persistent cache를 사용하지 않고 기존 lazy scan으로
    # 안전하게 돌아간다.


@dataclass(frozen=True, slots=True)
class _ReceiptRoles:
    rcept_no: str
    rcept_dt: str
    event_types: tuple[str, ...]
    counterparties: tuple[str, ...]
    contract_names: tuple[str, ...]


_ALIAS_PATH = Path(__file__).resolve().parent / "counterparty_aliases.tsv"
_ALIAS_CACHE: "dict[str, tuple[str, ...]] | None" = None


def _counterparty_aliases() -> "dict[str, tuple[str, ...]]":
    """한글 음차 → 코퍼스의 라틴 표기들. 파일이 없으면 빈 사전."""

    global _ALIAS_CACHE
    if _ALIAS_CACHE is not None:
        return _ALIAS_CACHE
    table: dict[str, set[str]] = {}
    try:
        text = _ALIAS_PATH.read_text(encoding="utf-8")
    except OSError:
        _ALIAS_CACHE = {}
        return _ALIAS_CACHE
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        latin, aliases = parts[0].strip(), parts[1]
        if not latin:
            continue
        for alias in aliases.split(","):
            key = _key(alias)
            if key:
                table.setdefault(key, set()).add(latin)
    _ALIAS_CACHE = {key: tuple(sorted(values)) for key, values in table.items()}
    return _ALIAS_CACHE


def _contains_any(query: str, values: "tuple[str, ...] | list[str]") -> bool:
    """질문 쪽 표기가 코퍼스 값과 같거나 그 **일부**인가.

    방향은 한쪽만 본다 — 질문 표기가 코퍼스 값 **안에** 있을 때만 참이다.
    반대 방향(코퍼스 값이 질문 안에)은 짧은 코퍼스 값이 긴 문장에 우연히 걸려
    엉뚱한 사건을 물어온다.
    """

    needle = _key(query)
    if not needle:
        return False
    keyed = tuple(_key(candidate) for candidate in values)
    if any(needle == value or needle in value for value in keyed):
        return True
    # **한글 음차로 부른 외국 상대를 잇는다.**
    #
    # 코퍼스의 계약상대 615종 중 133종이 라틴 표기인데, 질문은 한글로 쓴다
    # (「포드」 대 「Ford Motor Company」). 부분문자열만 보면 영영 안 걸려
    # `event_not_found_in_corpus` 가 된다.
    #
    # 사전의 **라틴 쪽은 정본 코퍼스에서 뽑았고** 한글 쪽은 일반 지식이다.
    # 동결 Gold 질문을 보고 만들지 않았으므로, 질문에 나오는 표기라도 사전에
    # 없으면 여기서 걸리지 않는다.
    for latin in _counterparty_aliases().get(needle, ()):  # noqa: SIM110
        target = _key(latin)
        if target and any(target == value or target in value for value in keyed):
            return True
    return False


class CanonicalEventKeyPreflight:
    """회사별 safe role 행만 lazy cache하고 후보 event key를 확정한다.

    ``cache_root``가 있고 canonical run metadata가 완전하면 cache ID는 build,
    schema, fields artifact hash, corpus cutoff, role policy에 결속된다. 같은
    ID의 cache가 없을 때만 ``fields()``를 한 번 전량 스캔해 재생성한다. 현재
    ID 경로에 파일이 있는데 envelope/hash/count/shape가 어긋나면 조용히 rebuild
    하지 않고 ``EventRoleIndexError``로 fail-closed한다. 따라서 stale/변조 cache가
    후보를 누락하거나 추가하는 경로가 없다.

    cache에는 Gold/question/expected fixture를 전혀 읽지 않고, corp_code별로
    receipt 날짜와 event_type/counterparty/contract_name 역할 값만 저장한다.
    """

    def __init__(
            self, canonical: EventCanonicalLike, *, corpus_cutoff: str,
            cache_root: Path | None = None,
            ) -> None:
        if any(not callable(getattr(canonical, name, None))
               for name in ("fields", "event_timeline")):
            raise TypeError("event preflight에는 fields/event_timeline reader가 필요합니다")
        if re.fullmatch(r"[0-9]{8}", corpus_cutoff) is None:
            raise ValueError("event preflight corpus cutoff 형식이 잘못되었습니다")
        self._canonical = canonical
        self._cutoff = corpus_cutoff
        if cache_root is not None:
            raw_cache_root = Path(cache_root)
            if raw_cache_root.is_symlink():
                raise EventRoleIndexError("event role cache root가 symlink입니다")
            self._cache_root = raw_cache_root.resolve()
        else:
            self._cache_root = None
        self._cache_contract = self._make_cache_contract()
        self.cache_status = (
            "disabled" if self._cache_contract is None else "not_loaded")
        self._rows: dict[str, tuple[_ReceiptRoles, ...]] = {}
        self._loaded = False
        self._lock = RLock()

    @staticmethod
    def _json_bytes(value: object) -> bytes:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")

    def _make_cache_contract(self) -> dict[str, str] | None:
        if self._cache_root is None:
            return None
        build_id = getattr(self._canonical, "build_id", None)
        schema_version = getattr(self._canonical, "schema_version", None)
        run = getattr(self._canonical, "run", None)
        artifact_hashes = run.get("artifact_hashes") if isinstance(run, dict) else None
        fields_hash = (
            artifact_hashes.get("fields")
            if isinstance(artifact_hashes, dict) else None)
        values = (build_id, schema_version, fields_hash)
        if any(not isinstance(value, str) or not value for value in values):
            return None
        identity = {
            "canonical_build_id": build_id,
            "canonical_schema_version": schema_version,
            "corpus_cutoff": self._cutoff,
            "fields_artifact_hash": fields_hash,
            "role_policy_version": EVENT_ROLE_POLICY_VERSION,
            "schema_version": EVENT_ROLE_INDEX_VERSION,
        }
        identity["index_build_id"] = sha256(
            self._json_bytes(identity)).hexdigest()[:32]
        return identity

    @property
    def cache_path(self) -> Path | None:
        if self._cache_contract is None or self._cache_root is None:
            return None
        return (self._cache_root / self._cache_contract["index_build_id"]
                / "event_roles.json")

    @staticmethod
    def _count_values(values: dict[str, tuple[_ReceiptRoles, ...]]) -> dict[str, int]:
        rows = tuple(row for company in values.values() for row in company)
        return {
            "corp_codes": len(values),
            "receipts": len(rows),
            "event_types": sum(len(row.event_types) for row in rows),
            "counterparties": sum(len(row.counterparties) for row in rows),
            "contract_names": sum(len(row.contract_names) for row in rows),
        }

    @staticmethod
    def _validate_role_values(value: object, role: str) -> tuple[str, ...]:
        if (not isinstance(value, list)
                or any(not isinstance(row, str) or not row for row in value)):
            raise EventRoleIndexError(f"event role {role} values shape가 잘못되었습니다")
        rows = tuple(value)
        if rows != tuple(sorted(set(rows))):
            raise EventRoleIndexError(f"event role {role} values가 정렬·고유하지 않습니다")
        return rows

    @classmethod
    def _validate_values(cls, value: object) -> dict[str, tuple[_ReceiptRoles, ...]]:
        if not isinstance(value, dict):
            raise EventRoleIndexError("event role values envelope이 잘못되었습니다")
        result: dict[str, tuple[_ReceiptRoles, ...]] = {}
        for corp_code, rows in value.items():
            if not isinstance(corp_code, str) or not corp_code:
                raise EventRoleIndexError("event role corp_code shape가 잘못되었습니다")
            if not isinstance(rows, list) or len(rows) > _MAX_COMPANY_ROLE_ROWS:
                raise EventRoleIndexError("event role receipt 행 상한을 넘었습니다")
            parsed: list[_ReceiptRoles] = []
            seen: set[str] = set()
            for row in rows:
                if not isinstance(row, dict) or set(row) != {
                        "rcept_no", "rcept_dt", "event_types",
                        "counterparties", "contract_names"}:
                    raise EventRoleIndexError("event role receipt row shape가 잘못되었습니다")
                rcept_no = row["rcept_no"]
                rcept_dt = row["rcept_dt"]
                if (not isinstance(rcept_no, str)
                        or re.fullmatch(r"[0-9]{14}", rcept_no) is None
                        or not isinstance(rcept_dt, str)
                        or re.fullmatch(r"[0-9]{8}", rcept_dt) is None):
                    raise EventRoleIndexError("event role receipt/date 형식이 잘못되었습니다")
                if rcept_no in seen:
                    raise EventRoleIndexError("event role receipt가 중복되었습니다")
                seen.add(rcept_no)
                parsed.append(_ReceiptRoles(
                    rcept_no=rcept_no, rcept_dt=rcept_dt,
                    event_types=cls._validate_role_values(row["event_types"], "event_type"),
                    counterparties=cls._validate_role_values(
                        row["counterparties"], "counterparty"),
                    contract_names=cls._validate_role_values(
                        row["contract_names"], "contract_name"),
                ))
            if tuple(row.rcept_no for row in parsed) != tuple(
                    sorted(row.rcept_no for row in parsed)):
                raise EventRoleIndexError("event role receipt가 정렬되지 않았습니다")
            result[corp_code] = tuple(parsed)
        return result

    def _validate_cache_parent(self, path: Path) -> None:
        """cache root와 build directory를 regular directory 경계로 잠근다."""

        if self._cache_root is None:
            return
        for candidate in (self._cache_root, path.parent, path):
            if candidate.is_symlink():
                raise EventRoleIndexError("event role cache 경로가 symlink입니다")

    def _load_cache(self) -> dict[str, tuple[_ReceiptRoles, ...]] | None:
        path = self.cache_path
        if path is None:
            return None
        self._validate_cache_parent(path)
        if not path.exists():
            return None
        if not path.is_file():
            raise EventRoleIndexError("event role cache 경로가 regular file이 아닙니다")
        try:
            if path.stat().st_size > _MAX_CACHE_BYTES:
                raise EventRoleIndexError("event role cache가 크기 상한을 넘었습니다")
            payload = json.loads(path.read_text(encoding="utf-8"))
        except EventRoleIndexError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EventRoleIndexError("event role cache를 읽을 수 없습니다") from exc
        if not isinstance(payload, dict) or self._cache_contract is None:
            raise EventRoleIndexError("event role cache envelope이 잘못되었습니다")
        for name, expected in self._cache_contract.items():
            if payload.get(name) != expected:
                raise EventRoleIndexError(
                    f"event role cache {name}이 current canonical과 다릅니다")
        values = self._validate_values(payload.get("values"))
        if any(row.rcept_dt > self._cutoff
               for rows in values.values() for row in rows):
            raise EventRoleIndexError(
                "event role cache에 corpus cutoff 이후 receipt가 있습니다")
        serialized_values = {
            corp_code: [
                {
                    "rcept_no": row.rcept_no, "rcept_dt": row.rcept_dt,
                    "event_types": list(row.event_types),
                    "counterparties": list(row.counterparties),
                    "contract_names": list(row.contract_names),
                }
                for row in rows
            ]
            for corp_code, rows in values.items()
        }
        expected_hash = sha256(self._json_bytes(serialized_values)).hexdigest()
        if payload.get("values_sha256") != expected_hash:
            raise EventRoleIndexError("event role cache values hash가 다릅니다")
        if payload.get("counts") != self._count_values(values):
            raise EventRoleIndexError("event role cache count가 다릅니다")
        return values

    def _write_cache(self, values: dict[str, tuple[_ReceiptRoles, ...]]) -> bool:
        path = self.cache_path
        if path is None or self._cache_contract is None:
            return False
        serialized_values = {
            corp_code: [
                {
                    "rcept_no": row.rcept_no, "rcept_dt": row.rcept_dt,
                    "event_types": list(row.event_types),
                    "counterparties": list(row.counterparties),
                    "contract_names": list(row.contract_names),
                }
                for row in rows
            ]
            for corp_code, rows in sorted(values.items())
        }
        payload = {
            **self._cache_contract,
            "counts": self._count_values(values),
            "values": serialized_values,
            "values_sha256": sha256(
                self._json_bytes(serialized_values)).hexdigest(),
        }
        body = self._json_bytes(payload) + b"\n"
        if len(body) > _MAX_CACHE_BYTES:
            raise EventRoleIndexError("event role cache payload가 크기 상한을 넘었습니다")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._validate_cache_parent(path)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o644)
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
        return True

    @staticmethod
    def _serialize_rows(
            grouped: dict[str, dict[str, dict[str, object]]],
            ) -> dict[str, tuple[_ReceiptRoles, ...]]:
        return {
            company_code: tuple(
                _ReceiptRoles(
                    rcept_no=rcept_no,
                    rcept_dt=str(row["rcept_dt"]),
                    event_types=tuple(sorted(row["event_types"])),  # type: ignore[arg-type]
                    counterparties=tuple(sorted(row["counterparties"])),  # type: ignore[arg-type]
                    contract_names=tuple(sorted(row["contract_names"])),  # type: ignore[arg-type]
                )
                for rcept_no, row in sorted(company_rows.items())
            )
            for company_code, company_rows in sorted(grouped.items())
        }

    def _build_index(self) -> dict[str, tuple[_ReceiptRoles, ...]]:
        grouped: dict[str, dict[str, dict[str, object]]] = {}
        n_receipts = 0
        for field in self._canonical.fields(
                as_of=self._cutoff,
                doc_group="exchange",
                is_pii=False,
                include_restricted_raw=False):
            role = event_support_role(
                field.path,
                value_status=field.value_status,
                is_pii=field.is_pii,
            )
            if role not in {"event_name", "counterparty"}:
                continue
            if (not isinstance(field.corp_code, str) or not field.corp_code
                    or not isinstance(field.rcept_no, str)
                    or re.fullmatch(r"[0-9]{14}", field.rcept_no) is None
                    or not isinstance(field.rcept_dt, str)
                    or re.fullmatch(r"[0-9]{8}", field.rcept_dt) is None):
                raise EventRoleIndexError("event role canonical receipt metadata가 잘못되었습니다")
            value = field.value
            if not isinstance(value, str) or not _key(value):
                continue
            company_rows = grouped.setdefault(field.corp_code, {})
            if field.rcept_no not in company_rows:
                n_receipts += 1
                if n_receipts > _MAX_COMPANY_ROLE_ROWS:
                    raise EventRoleIndexError("event preflight role 문서 상한을 넘었습니다")
            row = company_rows.setdefault(field.rcept_no, {
                "rcept_dt": field.rcept_dt,
                "event_types": set(),
                "counterparties": set(),
                "contract_names": set(),
            })
            if row["rcept_dt"] != field.rcept_dt:
                raise EventRoleIndexError("동일 접수번호의 event preflight 날짜가 충돌합니다")
            if field.event_type:
                if not isinstance(field.event_type, str):
                    raise EventRoleIndexError("event role event_type 형식이 잘못되었습니다")
                row["event_types"].add(field.event_type)  # type: ignore[union-attr]
            target = ("counterparties" if role == "counterparty"
                      else "contract_names")
            row[target].add(value)  # type: ignore[union-attr]
        return self._serialize_rows(grouped)

    def _company_rows(self, corp_code: str) -> tuple[_ReceiptRoles, ...]:
        with self._lock:
            if self._loaded:
                return self._rows.get(corp_code, ())
            cached = self._load_cache()
            if cached is not None:
                self._rows = cached
                self.cache_status = "cache_hit"
            else:
                self._rows = self._build_index()
                try:
                    persisted = self._write_cache(self._rows)
                except OSError:
                    persisted = False
                self.cache_status = (
                    "cold_persisted" if persisted else "memory_only")
            self._loaded = True
            return self._rows.get(corp_code, ())

    def warm(self) -> str:
        """서비스 시작 시 role index를 준비하고 cache 상태를 반환한다."""

        self._company_rows("")
        return self.cache_status

    @staticmethod
    def _matches(
            row: _ReceiptRoles, *, as_of: str,
            event_type: str | None, counterparty: str | None,
            contract_name: str | None, event_from: str | None,
            event_to: str | None,
            ) -> bool:
        if row.rcept_dt > as_of:
            return False
        if event_from is not None and row.rcept_dt < event_from:
            return False
        if event_to is not None and row.rcept_dt > event_to:
            return False
        if (event_type is not None
                and _key(event_type) not in {_key(v) for v in row.event_types}):
            return False
        # 상대방·계약명은 **사람이 말하는 축약형**으로 들어온다. 코퍼스는 법인
        # 전체 이름을 담는다 — 「Freudenberg」 대 「Freudenberg Battery Power
        # Systems, LLC」, 「배터리」 대 「전기차 배터리 공급계약」. 정확 일치만
        # 보면 코퍼스에 있는 사건을 못 찾아 `event_not_found_in_corpus` 가 된다.
        #
        # 그래서 **포함**도 허용한다. 넓게 걸려 후보가 여럿이 되면 이 preflight 는
        # 후보 목록을 돌려주고 resolver 가 역질문한다 — 조용히 하나를 고르지
        # 않으므로 fail-closed 다.
        if counterparty is not None and not _contains_any(
                counterparty, row.counterparties):
            return False
        if contract_name is not None and not _contains_any(
                contract_name, row.contract_names):
            return False
        return True

    def names_a_single_contract(self, *, corp_code: str, surface: str) -> bool:
        """이 표기가 **그 회사의 특정 계약 하나를 집어내는가.**

        `contract_name` 은 preflight 의 **하드 필터**다(`_matches`). 코퍼스가
        확인해 주지 못하는 말을 여기에 넣으면 걸리는 사건이 없어져
        `event_not_found_in_corpus` 로 질의 전체가 죽는다. 반면 `keywords` 는
        Stage1 에서 아무것도 거르지 않는 힌트라 틀려도 질의를 죽이지 않는다.

        그래서 **코퍼스가 특정 계약을 가리킨다고 확인해 줄 때만** 참이다.

        - 사건 유형 어휘(`유상증자결정` 등)의 일부면 거짓 — 계약 이름이 아니다.
        - 계약명과 같으면 참.
        - 계약명 **하나**에만 조각으로 들어가면 참 — 그 계약을 지목한다.
        - 여럿에 걸치면 거짓 — 「배터리」처럼 범주를 좁히는 말이다.
        - 코퍼스가 모르는 말이면 거짓 — 확인할 수 없으면 걸지 않는다.

        비교는 **`_matches` 와 같은 `_key` 정규화**로 한다(9차 검수 P2-RESOLVER-001).
        raw substring 으로 보면 NFKC·대소문자·공백 차이에서 분류가 흔들려,
        실제 필터가 쓰는 기준과 다른 답을 낸다.
        """

        value = _key(surface or "")
        if not value:
            return False
        # These are event-family/category nouns, not public contract names.
        # A company may happen to have only one title containing one of them,
        # but that corpus accident must not turn a broad user request into a
        # hard single-event selector.
        if value in {
                _key("계약"), _key("공급계약"), _key("판매계약"),
                _key("판매·공급계약"), _key("단일판매·공급계약"),
        }:
            return False
        rows = self._company_rows(corp_code)
        contract_names: set[str] = set()
        event_types: set[str] = set()
        for row in rows:
            contract_names.update(_key(name) for name in row.contract_names)
            event_types.update(_key(name) for name in row.event_types)
        contract_names.discard("")
        event_types.discard("")
        if any(value in name for name in event_types):
            return False
        if value in contract_names:
            return True
        return sum(1 for name in contract_names
                   if name != value and value in name) == 1

    def receipt_names_contract(
            self, *, corp_code: str, rcept_no: str, surface: str,
            ) -> bool:
        """Does this exact filing carry the requested contract-name role?

        Global uniqueness cannot link a named contract to an otherwise
        unrelated same-day correction.  This receipt-local check reuses the
        same normalized one-way containment as the ordinary event selector
        and fails closed if the role index has no unique row for the filing.
        """

        rows = [
            row for row in self._company_rows(corp_code)
            if row.rcept_no == rcept_no
        ]
        return (
            len(rows) == 1
            and _contains_any(surface, rows[0].contract_names)
        )

    def agreeing_candidate(
            self, *, candidates: "tuple[EventKeyCandidate, ...]",
            as_of: str, slots: "tuple[str, ...]",
            ) -> "EventKeyCandidate | None":
        """해지 관측이 붙은 event가 유일할 때만 그 event를 고른다.

        메서드 이름은 기존 protocol 호환 때문에 유지한다. 예전 구현은 후보별
        요청 field 값을 읽어 서로 같은지 비교했는데, 그 작업은 QueryPlan 생성이
        아니라 Stage2 답변 근거 해석에 가깝고 ``fields.parquet`` 전체 스캔을
        반복했다.

        여기서는 이미 만들어진 event timeline만 본다. 같은 상대·계약명 후보 중
        기준시점까지 **해지 전이가 관측된 event가 정확히 하나**면 그 event를
        선택한다. 이것은 어느 원계약에서 유래했는지를 확정하지 않으며 canonical의
        ``identity_status=ambiguous``도 그대로 보존한다. 해지 후보가 없거나 둘
        이상이면 ``None``으로 물러난다.
        """

        del slots
        if len(candidates) < 2:
            return None
        terminated: list[EventKeyCandidate] = []
        for row in candidates:
            timeline = self._canonical.event_timeline(
                as_of=as_of, event_key=row.event_key, verify_evidence=False)
            if timeline is None:
                continue
            observations = getattr(timeline, "observations", ()) or ()
            if any(getattr(observation, "is_termination", False)
                   for observation in observations):
                terminated.append(row)
        return terminated[0] if len(terminated) == 1 else None

    def resolve_event_key(
            self, *, corp_code: str, as_of: str,
            seed_rcept_no: str | None = None,
            event_type: str | None = None,
            counterparty: str | None = None,
            contract_name: str | None = None,
            event_from: str | None = None,
            event_to: str | None = None,
            ) -> EventKeyResolution:
        if seed_rcept_no is not None:
            timeline = self._canonical.event_timeline(
                as_of=as_of, rcept_no=seed_rcept_no, verify_evidence=False)
            if timeline is None or timeline.corp_code != corp_code:
                return EventKeyResolution("not_found")
            candidate = EventKeyCandidate(
                event_key=timeline.event_key,
                seed_rcept_no=timeline.root_rcept_no,
                label=f"{timeline.root_rcept_no} 기준 사건",
                identity_fingerprint=getattr(
                    timeline, "identity_fingerprint", None),
                identity_status=getattr(timeline, "identity_status", None),
                observation_count=len(getattr(timeline, "observations", ()) or ()),
            )
            return EventKeyResolution("resolved", (candidate,))

        matched = [
            row for row in self._company_rows(corp_code)
            if self._matches(
                row, as_of=as_of, event_type=event_type,
                counterparty=counterparty, contract_name=contract_name,
                event_from=event_from, event_to=event_to)
        ]
        by_key: dict[str, EventKeyCandidate] = {}
        for row in matched:
            timeline = self._canonical.event_timeline(
                as_of=as_of, rcept_no=row.rcept_no, verify_evidence=False)
            if timeline is None or timeline.corp_code != corp_code:
                continue
            details = tuple(dict.fromkeys(
                (*row.contract_names, *row.counterparties)))
            detail = " / ".join(details[:2])
            label = timeline.root_rcept_no + (f" · {detail}" if detail else "")
            candidate = EventKeyCandidate(
                event_key=timeline.event_key,
                seed_rcept_no=timeline.root_rcept_no,
                label=label,
                identity_fingerprint=getattr(
                    timeline, "identity_fingerprint", None),
                identity_status=getattr(timeline, "identity_status", None),
                # **as_of 로 재계산된 관측만 센다.** identity 표의 전체 관측 수를
                # 쓰면 과거 시점 질문에 미래 해지가 샌다.
                observation_count=len(getattr(timeline, "observations", ()) or ()),
            )
            previous = by_key.get(candidate.event_key)
            if previous is None or candidate.seed_rcept_no < previous.seed_rcept_no:
                by_key[candidate.event_key] = candidate
            if len(by_key) > _MAX_OPTIONS:
                return EventKeyResolution("too_many")
        candidates = tuple(sorted(
            by_key.values(), key=lambda row: (row.event_key, row.seed_rcept_no)))
        if not candidates:
            return EventKeyResolution("not_found")
        if len(candidates) == 1:
            return EventKeyResolution("resolved", candidates)
        # **fingerprint 가 같다고 같은 계약이 아니다.** 정본 builder 는 연결되지
        # 않은 두 lineage 가 같은 fingerprint 를 가질 때 **자동 병합하지 않고**
        # 둘 다 `ambiguous` 로 내려 서로 다른 event_key 를 유지한다 —
        # 「실제 코퍼스에는 같은 회사·계약명·상대·계약일인 별도 계약도 있다」
        # (`src/canonical/events.py`). 즉 `ambiguous` 는 「같은 계약」이 아니라
        # 「같은지 확정하지 못했다」는 뜻이다.
        #
        # 그 판정을 fingerprint 동등성만으로 뒤집지 않는다. 명시적인 canonical
        # alias/group 권위가 생기기 전에는 되묻는다 (9차 검수 P1-RESOLVER-001).
        return EventKeyResolution("ambiguous", candidates)


__all__ = [
    "CanonicalEventKeyPreflight", "EventKeyCandidate", "EventKeyPreflight",
    "EventKeyResolution", "EventResolutionStatus", "EventRoleIndexError",
]
