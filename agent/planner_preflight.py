"""Stage 1의 선택적 selector 역할 preflight.

질문의 중립 mention을 최종 답이나 금액으로 해석하지 않는다. 공개 QueryPlan에
필요한 ``counterparty``/``contract_name`` 역할만 정본의 safe Field label/value로
확인하며, 유일하지 않으면 고르지 않는다. 인덱스는 실제 ambiguous mention이 처음
들어왔을 때 한 번만 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from threading import Lock
from typing import Callable, Iterable, Literal, Protocol
import unicodedata
import tempfile

from .contracts import validate_as_of


SelectorRole = Literal["counterparty", "contract_name"]
SelectorResolutionStatus = Literal["resolved", "ambiguous", "not_found"]
SELECTOR_ROLE_INDEX_VERSION = "selector-role-index/0.1"
SELECTOR_ROLE_POLICY_VERSION = "selector-role-policy/0.2"
_MAX_ROLE_VALUES = 200_000
_MAX_CACHE_BYTES = 32 * 1024 * 1024


class SelectorRoleIndexError(RuntimeError):
    """build-bound selector role index가 없거나 계약과 다르다."""


class SelectorFieldLike(Protocol):
    path: str
    value: str | None
    value_status: str
    is_pii: bool

    @property
    def label(self) -> str: ...


class SelectorFieldReader(Protocol):
    def fields(self, **kwargs: object) -> Iterable[SelectorFieldLike]: ...


@dataclass(frozen=True, slots=True)
class SelectorRoleResolution:
    status: SelectorResolutionStatus
    matched_roles: tuple[SelectorRole, ...]
    role: SelectorRole | None = None
    resolved_text: str | None = None

    def __post_init__(self) -> None:
        if self.matched_roles != tuple(sorted(set(self.matched_roles))):
            raise ValueError("selector matched_roles는 정렬·중복제거되어야 합니다")
        if self.status == "resolved":
            if (self.role is None or self.resolved_text is None
                    or self.matched_roles != (self.role,)):
                raise ValueError("resolved selector role 계약이 잘못되었습니다")
        elif self.role is not None or self.resolved_text is not None:
            raise ValueError("미확정 selector role은 확정값을 가질 수 없습니다")
        if self.status == "ambiguous" and len(self.matched_roles) < 2:
            raise ValueError("ambiguous selector에는 역할 후보가 2개 이상 필요합니다")
        if self.status == "not_found" and self.matched_roles:
            raise ValueError("not_found selector에는 역할 후보가 없어야 합니다")


@dataclass(frozen=True, slots=True)
class CompanyMentionResolution:
    status: SelectorResolutionStatus
    candidate_names: tuple[str, ...]
    resolved_text: str | None = None

    def __post_init__(self) -> None:
        if self.candidate_names != tuple(sorted(set(self.candidate_names))):
            raise ValueError("company 후보는 정렬·중복제거되어야 합니다")
        if self.status == "resolved":
            if (self.resolved_text is None
                    or self.candidate_names != (self.resolved_text,)):
                raise ValueError("resolved company mention 계약이 잘못되었습니다")
        elif self.resolved_text is not None:
            raise ValueError("미확정 company mention은 확정값을 가질 수 없습니다")
        if self.status == "ambiguous" and len(self.candidate_names) < 2:
            raise ValueError("ambiguous company mention에는 후보가 2개 이상 필요합니다")
        if self.status == "not_found" and self.candidate_names:
            raise ValueError("not_found company mention에는 후보가 없어야 합니다")


@dataclass(frozen=True, slots=True, order=True)
class IssuerCandidate:
    """Safe canonical field가 가리키는 공시 issuer.

    법인명만으로 후보 수를 세면 같은 표기가 서로 다른 corp_code를 가리는 경우가
    생긴다. issuer 추론의 유일성은 정본 corp_code 기준으로 닫는다.
    """

    corp_code: str
    corp_name: str

    def __post_init__(self) -> None:
        if not self.corp_code or not self.corp_name:
            raise ValueError("issuer candidate에는 corp_code와 corp_name이 필요합니다")


@dataclass(frozen=True, slots=True)
class SelectorIssuerResolution:
    """역할이 이미 확정된 selector가 가리키는 issuer 후보.

    이 값은 event identity나 receipt를 고르지 않는다. 오직 ``company_mentions``를
    안전하게 보충할 수 있는지의 unique-or-fail-closed 판정이다.
    """

    status: SelectorResolutionStatus
    candidates: tuple[IssuerCandidate, ...]
    resolved_text: str | None = None

    def __post_init__(self) -> None:
        if self.candidates != tuple(sorted(set(self.candidates))):
            raise ValueError("issuer 후보는 정렬·중복제거되어야 합니다")
        if self.status == "resolved":
            if len(self.candidates) != 1 or self.resolved_text != self.candidates[0].corp_name:
                raise ValueError("resolved issuer 계약이 잘못되었습니다")
        elif self.resolved_text is not None:
            raise ValueError("미확정 issuer는 확정값을 가질 수 없습니다")
        if self.status == "ambiguous" and len(self.candidates) < 2:
            raise ValueError("ambiguous issuer에는 후보가 2개 이상 필요합니다")
        if self.status == "not_found" and self.candidates:
            raise ValueError("not_found issuer에는 후보가 없어야 합니다")


class SelectorRolePreflight(Protocol):
    def resolve_selector_role(
            self, mention: str, *, task_mode: str,
            ) -> SelectorRoleResolution: ...

    def resolve_company_mention(
            self, mention: str,
            ) -> CompanyMentionResolution: ...

    def resolve_question_issuer(
            self, question: str,
            ) -> SelectorIssuerResolution: ...

    def resolve_selector_issuer(
            self, mention: str, *, role: SelectorRole,
            ) -> SelectorIssuerResolution: ...


def selector_lookup_text(mention: str) -> str:
    """일반 명사 ``계약``은 버리고 식별에 필요한 원문 부분만 남긴다."""

    value = unicodedata.normalize("NFKC", mention).strip()
    value = re.sub(r"\s*계약\s*$", "", value, flags=re.IGNORECASE).strip()
    return value


def _key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^0-9a-z가-힣]+", "", normalized)


#: 회사 표기는 **한 곳에서만** 정한다 — ``src/canonical/company_alias.py`` 의
#: registry 다. 여기에 별칭표를 따로 두면 두 곳이 서로 다른 답을 낼 수 있고,
#: 실제로 그랬다. ``삼전`` 을 이 파일은 삼성전자로 확정했지만 registry 는
#: 삼성전자·삼성전기 두 후보로 두고 역질문한다. 이관 후 표는 없앴다.


_COUNTERPARTY_LABEL_CUES = tuple(map(_key, (
    "계약상대", "계약상대방", "거래상대", "상대방", "해지계약상대",
)))
_CONTRACT_NAME_LABEL_CUES = tuple(map(_key, (
    "계약명", "체결계약명", "해지계약명",
)))
_UNUSABLE_VALUE_STATUSES = {"empty", "not_reported", "parse_error"}


def _field_role(field: SelectorFieldLike) -> SelectorRole | None:
    label = _key(field.label)
    if any(cue in label for cue in _COUNTERPARTY_LABEL_CUES):
        return "counterparty"
    if any(cue in label for cue in _CONTRACT_NAME_LABEL_CUES):
        return "contract_name"
    return None


def _parenthetical_company_target(
        mention: str,
        resolved: "Callable[[str], tuple[str, ...]]",
        ) -> "tuple[str, ...] | None":
    """병기 표기 ``A(B)`` 가 가리키는 회사 하나. 모이지 않으면 ``None``.

    어간과 괄호 안을 각각 해소한다. 평범한 ``A(B)`` 는 둘 다 한 회사로
    해소되고 **그 회사가 같아야** 받는다. 명시적인 옛 이름 표지(``A(구 B)``)가
    있을 때만, 다른 쪽이 0건인 경우 한쪽의 유일 해소를 쓴다. 모호한 쪽이 있거나
    서로 다른 회사면 고르지 않는다.
    """

    match = _PARENTHETICAL_MENTION.fullmatch((mention or "").strip())
    if match is None:
        return None
    stem = resolved(selector_lookup_text(match.group("stem").strip()))
    raw_inner = match.group("inner").strip()
    former = _FORMER_NAME_PREFIX.fullmatch(raw_inner)
    inner_text = former.group("name") if former is not None else raw_inner
    inner = resolved(selector_lookup_text(inner_text.strip()))
    if len(stem) == 1 and len(inner) == 1:
        return stem if stem == inner else None
    if former is None or len(stem) > 1 or len(inner) > 1:
        return None
    if len(stem) == 1 and not inner:
        return stem
    if len(inner) == 1 and not stem:
        return inner
    return None



_PARENTHETICAL_MENTION = re.compile(
    r"^(?P<stem>[^()]+?)\s*[（(]\s*(?P<inner>[^()]+?)\s*[）)]$")
_FORMER_NAME_PREFIX = re.compile(
    r"^(?:구|前|전)(?:\s+|[.,]\s*)(?P<name>\S.*)$")

class CanonicalSelectorRolePreflight:
    """CanonicalReadModel 공개 ``fields()``로 만드는 lazy compact role index."""

    def __init__(
            self, canonical: SelectorFieldReader, *, corpus_cutoff: str,
            cache_root: Path | None = None,
            ) -> None:
        if not callable(getattr(canonical, "fields", None)):
            raise TypeError("selector preflight에는 fields() reader가 필요합니다")
        self.canonical = canonical
        self.corpus_cutoff = validate_as_of(
            corpus_cutoff, field_name="corpus_cutoff")
        self.cache_root = cache_root.resolve() if cache_root is not None else None
        self._cache_contract = self._make_cache_contract()
        self.cache_status = "disabled" if self._cache_contract is None else "not_loaded"
        self._values: dict[SelectorRole, tuple[str, ...]] | None = None
        self._issuer_values: dict[
            SelectorRole, dict[str, tuple[IssuerCandidate, ...]]
        ] | None = None
        self._lock = Lock()
        self._issuer_lock = Lock()

    @staticmethod
    def _json_bytes(value: object) -> bytes:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")

    def _make_cache_contract(self) -> dict[str, str] | None:
        if self.cache_root is None:
            return None
        build_id = getattr(self.canonical, "build_id", None)
        schema_version = getattr(self.canonical, "schema_version", None)
        run = getattr(self.canonical, "run", None)
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
            "corpus_cutoff": self.corpus_cutoff,
            "fields_artifact_hash": fields_hash,
            "role_policy_version": SELECTOR_ROLE_POLICY_VERSION,
            "schema_version": SELECTOR_ROLE_INDEX_VERSION,
        }
        identity["index_build_id"] = sha256(
            self._json_bytes(identity)).hexdigest()[:32]
        return identity

    @property
    def cache_path(self) -> Path | None:
        if self._cache_contract is None or self.cache_root is None:
            return None
        return (self.cache_root / self._cache_contract["index_build_id"]
                / "selector_roles.json")

    @staticmethod
    def _validate_values(value: object) -> dict[SelectorRole, tuple[str, ...]]:
        if not isinstance(value, dict) or set(value) != {
                "counterparty", "contract_name"}:
            raise SelectorRoleIndexError("selector role values shape가 잘못되었습니다")
        result: dict[SelectorRole, tuple[str, ...]] = {}
        for role in ("counterparty", "contract_name"):
            rows = value.get(role)
            if (not isinstance(rows, list) or len(rows) > _MAX_ROLE_VALUES
                    or any(not isinstance(row, str) or not row for row in rows)):
                raise SelectorRoleIndexError(
                    f"selector role {role} values가 유효하지 않습니다")
            normalized = tuple(rows)
            if normalized != tuple(sorted(set(normalized))):
                raise SelectorRoleIndexError(
                    f"selector role {role} values가 정렬·고유하지 않습니다")
            result[role] = normalized
        return result

    def _load_cache(self) -> dict[SelectorRole, tuple[str, ...]] | None:
        path = self.cache_path
        if path is None or not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise SelectorRoleIndexError("selector role cache 경로가 regular file이 아닙니다")
        if path.stat().st_size > _MAX_CACHE_BYTES:
            raise SelectorRoleIndexError("selector role cache가 크기 상한을 넘었습니다")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SelectorRoleIndexError("selector role cache를 읽을 수 없습니다") from exc
        if not isinstance(payload, dict) or self._cache_contract is None:
            raise SelectorRoleIndexError("selector role cache envelope이 잘못되었습니다")
        for name, expected in self._cache_contract.items():
            if payload.get(name) != expected:
                raise SelectorRoleIndexError(
                    f"selector role cache {name}이 current canonical과 다릅니다")
        values = self._validate_values(payload.get("values"))
        serialized_values = {
            role: list(rows) for role, rows in values.items()
        }
        expected_hash = sha256(self._json_bytes(serialized_values)).hexdigest()
        if payload.get("values_sha256") != expected_hash:
            raise SelectorRoleIndexError("selector role cache values hash가 다릅니다")
        counts = payload.get("counts")
        if counts != {role: len(rows) for role, rows in values.items()}:
            raise SelectorRoleIndexError("selector role cache count가 다릅니다")
        return values

    def _write_cache(self, values: dict[SelectorRole, tuple[str, ...]]) -> bool:
        path = self.cache_path
        if path is None or self._cache_contract is None:
            return False
        serialized_values = {
            role: list(rows) for role, rows in values.items()
        }
        payload = {
            **self._cache_contract,
            "counts": {role: len(rows) for role, rows in values.items()},
            "values": serialized_values,
            "values_sha256": sha256(
                self._json_bytes(serialized_values)).hexdigest(),
        }
        body = self._json_bytes(payload) + b"\n"
        if len(body) > _MAX_CACHE_BYTES:
            raise SelectorRoleIndexError("selector role cache payload가 크기 상한을 넘었습니다")
        path.parent.mkdir(parents=True, exist_ok=True)
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

    def _build_index(self) -> dict[SelectorRole, tuple[str, ...]]:
        values: dict[SelectorRole, set[str]] = {
            "counterparty": set(), "contract_name": set(),
        }
        for field in self.canonical.fields(
                as_of=self.corpus_cutoff,
                is_pii=False,
                include_restricted_raw=False,
                ):
            if field.is_pii or field.value_status in _UNUSABLE_VALUE_STATUSES:
                continue
            role = _field_role(field)
            value = field.value
            if role is None or not isinstance(value, str):
                continue
            normalized = _key(value)
            if normalized:
                values[role].add(normalized)
                if len(values[role]) > _MAX_ROLE_VALUES:
                    raise SelectorRoleIndexError(
                        f"selector role {role} unique value 상한을 넘었습니다")
        return {
            role: tuple(sorted(role_values))
            for role, role_values in values.items()
        }

    def _index(self) -> dict[SelectorRole, tuple[str, ...]]:
        if self._values is not None:
            return self._values
        with self._lock:
            if self._values is None:
                cached = self._load_cache()
                if cached is not None:
                    self._values = cached
                    self.cache_status = "cache_hit"
                else:
                    self._values = self._build_index()
                    try:
                        persisted = self._write_cache(self._values)
                    except OSError:
                        persisted = False
                    self.cache_status = (
                        "cold_persisted" if persisted else "memory_only")
        return self._values

    def warm(self) -> str:
        """서비스 시작 시 role index를 한 번 준비하고 공개 상태만 반환한다."""

        self._index()
        return self.cache_status

    def resolve_selector_role(
            self, mention: str, *, task_mode: str,
            ) -> SelectorRoleResolution:
        if task_mode == "financial_lookup" or task_mode.startswith("narrative_"):
            raise ValueError("selector role preflight 대상 task가 아닙니다")
        resolved_text = selector_lookup_text(mention)
        needle = _key(resolved_text)
        if len(needle) < 2:
            return SelectorRoleResolution(
                status="not_found", matched_roles=())
        index = self._index()
        roles = tuple(sorted(
            role for role, values in index.items()
            if any(needle in value for value in values)
        ))
        if len(roles) == 1:
            return SelectorRoleResolution(
                status="resolved", matched_roles=roles,
                role=roles[0], resolved_text=resolved_text)
        if roles:
            return SelectorRoleResolution(
                status="ambiguous", matched_roles=roles)

        # Provider surfaces often combine the issuer, a counterparty alias,
        # and a generic suffix into one event string (``LG엔솔 포드건``).
        # The whole string cannot be a corpus selector, but discarding it
        # makes a known counterparty look like an absent event.  Recover only
        # a literal sub-surface that the canonical counterparty alias table
        # maps back to a safe counterparty value.  This does not split company
        # names, select an event, or introduce an ungrounded synonym.
        from .event_preflight import _counterparty_aliases

        alias_hits: list[tuple[str, tuple[SelectorRole, ...]]] = []
        for alias, canonical_values in _counterparty_aliases().items():
            if len(alias) < 2 or alias not in needle:
                continue
            canonical_keys = {_key(value) for value in canonical_values}
            matched = tuple(sorted(
                role for role, values in index.items()
                if role == "counterparty" and any(
                    candidate == value or candidate in value
                    for candidate in canonical_keys for value in values)
            ))
            if matched:
                alias_hits.append((alias, matched))

        # Latin text is normally tokenized by whitespace, but Korean company
        # prefixes can attach directly to it.  Test each literal Latin token
        # separately after the full surface has failed; the canonical index
        # remains the authority for its role.
        for token in re.findall(r"[A-Za-z][A-Za-z0-9.&'-]*", resolved_text):
            token_key = _key(token)
            if len(token_key) < 2:
                continue
            matched = tuple(sorted(
                role for role, values in index.items()
                if any(token_key in value for value in values)
            ))
            if matched:
                alias_hits.append((token, matched))

        if alias_hits:
            # More than one matching alias may be a spelling expansion of the
            # same counterparty (``포드``/``포드자동차``).  Prefer the longest
            # literal surface, but never collapse distinct role outcomes.
            role_sets = {matched for _, matched in alias_hits}
            if len(role_sets) == 1:
                matched = next(iter(role_sets))
                if len(matched) == 1:
                    alias = max(
                        (surface for surface, _ in alias_hits), key=len)
                    return SelectorRoleResolution(
                        status="resolved", matched_roles=matched,
                        role=matched[0], resolved_text=alias)
                return SelectorRoleResolution(
                    status="ambiguous", matched_roles=matched)
        return SelectorRoleResolution(status="not_found", matched_roles=())

    def _issuer_index(
            self,
            ) -> dict[SelectorRole, dict[str, tuple[IssuerCandidate, ...]]]:
        """Safe typed fields만으로 ``role value -> issuer``를 lazy 구축한다.

        role index cache는 기존 selector-role 계약을 보존한다. issuer mapping은
        corp metadata가 포함된 ``FieldRow``가 필요하므로 별도로 (그리고 필요할
        때만) 읽는다.
        """

        if self._issuer_values is not None:
            return self._issuer_values
        with self._issuer_lock:
            if self._issuer_values is None:
                values: dict[SelectorRole, dict[str, set[IssuerCandidate]]] = {
                    "counterparty": {}, "contract_name": {},
                }
                documents = getattr(self.canonical, "documents", None)
                if callable(documents):
                    # Issuer inference is used for exchange event/correction
                    # selectors.  Periodic filings contain arbitrary narrative
                    # table labels such as `계약상대` and are not authoritative
                    # event identities; including them also decodes most of the
                    # 1.6M-row field store.  Keep the inference authority on the
                    # typed exchange document family and let other requests
                    # retain their explicit issuer/clarification path.
                    field_stream = self.canonical.fields(
                        as_of=self.corpus_cutoff,
                        doc_group="exchange",
                        is_pii=False,
                        include_restricted_raw=False,
                    )
                else:
                    field_stream = self.canonical.fields(
                        as_of=self.corpus_cutoff,
                        is_pii=False,
                        include_restricted_raw=False,
                    )
                for field in field_stream:
                    if field.is_pii or field.value_status in _UNUSABLE_VALUE_STATUSES:
                        continue
                    role = _field_role(field)
                    value = field.value
                    corp_code = getattr(field, "corp_code", None)
                    corp_name = getattr(field, "corp_name", None)
                    if (
                            role is None or not isinstance(value, str)
                            or not isinstance(corp_code, str) or not corp_code
                            or not isinstance(corp_name, str) or not corp_name
                    ):
                        continue
                    normalized = _key(value)
                    if not normalized:
                        continue
                    by_value = values[role]
                    candidates = by_value.setdefault(normalized, set())
                    candidates.add(IssuerCandidate(corp_code, corp_name))
                    if len(by_value) > _MAX_ROLE_VALUES:
                        raise SelectorRoleIndexError(
                            f"selector issuer {role} unique value 상한을 넘었습니다")
                self._issuer_values = {
                    role: {
                        value: tuple(sorted(candidates))
                        for value, candidates in by_value.items()
                    }
                    for role, by_value in values.items()
                }
        return self._issuer_values

    @staticmethod
    def _issuer_resolution(
            candidates: Iterable[IssuerCandidate],
            ) -> SelectorIssuerResolution:
        unique = tuple(sorted(set(candidates)))
        if len(unique) == 1:
            return SelectorIssuerResolution(
                status="resolved", candidates=unique,
                resolved_text=unique[0].corp_name)
        if unique:
            return SelectorIssuerResolution(status="ambiguous", candidates=unique)
        return SelectorIssuerResolution(status="not_found", candidates=())

    def resolve_selector_issuer(
            self, mention: str, *, role: SelectorRole,
            ) -> SelectorIssuerResolution:
        """확정된 counterparty/contract selector가 유일하게 가리키는 issuer.

        일반 text 검색이 아니라 같은 safe canonical field label의 typed value만
        본다. selector role은 이 메서드에 오기 전에 이미 확정돼야 한다.
        """

        if role not in {"counterparty", "contract_name"}:
            raise ValueError("selector issuer role이 지원되지 않습니다")
        needle = _key(selector_lookup_text(mention))
        if len(needle) < 2:
            return SelectorIssuerResolution(status="not_found", candidates=())
        index = self._issuer_index()
        return self._issuer_resolution(
            candidate
            for value, candidates in index[role].items()
            if needle in value
            for candidate in candidates
        )

    def resolve_question_issuer(self, question: str) -> SelectorIssuerResolution:
        """질문에 명시된 supported issuer가 하나일 때만 canonicalize 한다."""

        if not isinstance(question, str):
            raise TypeError("question issuer에는 문자열이 필요합니다")
        finder = getattr(self.canonical, "companies_in_text", None)
        if not callable(finder):
            return SelectorIssuerResolution(status="not_found", candidates=())
        try:
            rows = finder(question)
        except Exception as exc:
            raise SelectorRoleIndexError("question issuer 해소가 실패했습니다") from exc
        if not isinstance(rows, list):
            raise SelectorRoleIndexError("question issuer resolver 반환 계약이 잘못되었습니다")
        candidates: list[IssuerCandidate] = []
        for row in rows:
            corp_code = getattr(row, "corp_code", None)
            corp_name = getattr(row, "corp_name", None)
            if not isinstance(corp_code, str) or not isinstance(corp_name, str):
                raise SelectorRoleIndexError("question issuer 후보 계약이 잘못되었습니다")
            candidates.append(IssuerCandidate(corp_code, corp_name))
        return self._issuer_resolution(candidates)

    def corp_codes_for_mention(self, mention: str) -> tuple[str, ...]:
        """표기 하나가 가리키는 corp_code 들. 유일하지 않으면 여러 개다."""

        resolver = getattr(self.canonical, "resolve_company", None)
        if not callable(resolver):
            return ()
        try:
            rows = resolver(selector_lookup_text(mention))
        except Exception as exc:
            raise SelectorRoleIndexError(
                "company mention 해소가 실패했습니다") from exc
        return tuple(sorted({
            code for row in rows
            if isinstance(code := getattr(row, "corp_code", None), str)
        }))

    def companies_named_in_question(self, question: str) -> tuple[str, ...]:
        """질문이 **모호하지 않게** 지명한 회사들의 corp_code.

        모델이 질문의 축약 대신 정식 명칭을 쓰는 일이 흔하다 — 질문 ``LG엔솔``,
        출력 ``LG에너지솔루션``. 두 표현이 같은 회사인지 판단하려면 질문 쪽에서도
        회사를 해소해야 한다. 모호한 표기는 canonical 쪽에서 이미 제외된다.
        """

        finder = getattr(self.canonical, "companies_in_text", None)
        if not callable(finder):
            return ()
        try:
            rows = finder(question)
        except Exception as exc:
            raise SelectorRoleIndexError(
                "question company scan이 실패했습니다") from exc
        return tuple(sorted({
            code for row in rows
            if isinstance(code := getattr(row, "corp_code", None), str)
        }))

    def question_company_surface(
            self, mention: str, question: str,
            ) -> str | None:
        """모델이 쓴 회사 표기를 **질문에 실제로 있는 표기**로 되돌린다.

        grounding 은 질문에 없는 글자를 지운다(`_ground_surface`). 그런데 모델은
        정규화를 한다 — 질문이 「삼전」인데 `삼성전자` 를, 「LG엔솔」인데
        `LG에너지솔루션` 을 쓴다. **정규화는 옳은 동작인데** 글자가 다르다는
        이유로 지워지고, 그다음 사용자에게 「회사를 알려달라」고 되묻는다.
        질문에 이미 답이 있는데 되묻는 것이므로 틀린 역질문이다.

        여기서 **불변식을 완화하지 않는다.** 모델 표기를 승인하는 것이 아니라
        같은 회사를 가리키는 **질문 자신의 표기로 바꾼다.** 결과는 여전히 질문에
        literal 로 존재하므로 grounding 이 그대로 통과한다.

        판정은 정본 registry(`src/canonical/company_alias.py`) 한 곳에서만 한다.
        여기에 별칭표를 두면 두 곳이 다른 답을 낸다.

        되돌리는 조건 셋을 모두 만족해야 한다.

        1. 모델 표기가 **회사 하나로** 해소된다 — 후보가 여럿이면 고르지 않는다.
        2. 질문의 어떤 조각이 **같은 그 회사 하나로** 해소된다.
        3. 그 조각은 질문의 인접 토큰(또는 마지막 토큰의 접두어)이다 —
           「삼전이랑」에서 「삼전」을, ``SM Entertainment의``에서
           ``SM Entertainment``를 찾되, 질문에 없는 문자열을 만들지 않는다.

        하나라도 못 채우면 `None` 이고 기존 grounding 이 그대로 간다.
        """

        resolver = getattr(self.canonical, "resolve_company", None)
        if not callable(resolver) or not isinstance(question, str):
            return None

        def resolved(value: str) -> tuple[str, ...]:
            try:
                rows = resolver(value)
            except Exception:                            # noqa: BLE001
                return ()
            return tuple(
                name for row in (rows or ())
                if isinstance(name := getattr(row, "corp_code", None), str))

        target = resolved(selector_lookup_text(mention))
        if len(target) != 1:
            # 「케이티(KT)」·「LIG디펜스앤에어로스페이스(구 LIG넥스원)」처럼 병기한
            # 표기는 registry 에 통째로는 없다. 평범한 병기는 양쪽이 같은 회사로
            # 모일 때만 받고, 명시적인 옛 이름 병기만 한쪽 0건을 허용한다. 둘이
            # 다르거나 한쪽이 모호하면 고르지 않는다.
            target = _parenthetical_company_target(mention, resolved)
            if target is None:
                return None
        # The approved registry includes exact aliases containing spaces
        # (for example an English brand).  Test contiguous literal spans
        # before their trailing Korean case particle, not just each word.
        # The bounded window keeps this a spelling recovery rather than a
        # general text search; the registry remains the identity authority.
        tokens = re.findall(r"[A-Za-z0-9&.+가-힣]+", question)
        best: str | None = None
        particles = frozenset("은는이가을를의에도와과만")
        for start in range(len(tokens)):
            for end in range(start + 1, min(len(tokens), start + 4) + 1):
                prefix = tokens[start:end - 1]
                tail = tokens[end - 1]
                forms = [tail]
                while forms[-1] and forms[-1][-1] in particles:
                    forms.append(forms[-1][:-1])
                for final in forms:
                    if len(final) < 2:
                        continue
                    surface = " ".join((*prefix, final))
                    if resolved(surface) == target and (
                            best is None or len(surface) > len(best)):
                        best = surface
        return best

    def unique_question_company_surface(self, question: str) -> str | None:
        """Return the one canonical company surface explicitly in ``question``.

        This is intentionally separate from ``resolve_question_issuer``: the
        latter returns issuer identity, while this helper returns the literal
        question span needed to preserve SemanticIntent grounding.  It only
        succeeds when the complete question scan yields one corp code.
        """

        if not isinstance(question, str):
            return None
        finder = getattr(self.canonical, "companies_in_text", None)
        if not callable(finder):
            return None
        try:
            rows = finder(question)
        except Exception:  # noqa: BLE001
            return None
        candidates = {
            (code, name)
            for row in rows or ()
            if isinstance(code := getattr(row, "corp_code", None), str)
            and isinstance(name := getattr(row, "corp_name", None), str)
        }
        if len({code for code, _ in candidates}) != 1:
            return None
        _, corp_name = next(iter(candidates))
        return self.question_company_surface(corp_name, question)

    def resolve_company_mention(
            self, mention: str,
            ) -> CompanyMentionResolution:
        """selector로 잘못 온 표현이 exact 회사명/alias일 때만 역할을 복구한다."""

        resolver = getattr(self.canonical, "resolve_company", None)
        if not callable(resolver):
            return CompanyMentionResolution(status="not_found", candidate_names=())
        surface = selector_lookup_text(mention)
        # 표기 해소는 canonical registry 한 곳에서만 한다. 여기서 별칭을 다시
        # 들고 있으면 두 곳이 다른 답을 낸다.
        query = surface
        try:
            rows = resolver(query)
        except Exception as exc:
            raise SelectorRoleIndexError("company mention preflight가 실패했습니다") from exc
        if not isinstance(rows, list) or any(
                not isinstance(getattr(row, "corp_name", None), str)
                for row in rows):
            raise SelectorRoleIndexError("company resolver 반환 계약이 잘못되었습니다")
        # registry 가 이미 정규화 exact match 로 후보를 좁혔다. 여기서 corp_name
        # 문자열을 다시 대조하면 `삼전`처럼 **표기와 법인명이 다른** 승인 별칭이
        # 통째로 버려진다.
        exact = tuple(sorted({row.corp_name for row in rows}))
        if len(exact) == 1:
            return CompanyMentionResolution(
                status="resolved", candidate_names=exact,
                resolved_text=exact[0])
        if len(exact) > 1:
            return CompanyMentionResolution(
                status="ambiguous", candidate_names=exact)
        return CompanyMentionResolution(status="not_found", candidate_names=())


__all__ = [
    "CanonicalSelectorRolePreflight", "CompanyMentionResolution",
    "IssuerCandidate",
    "SelectorRole", "SelectorRolePreflight",
    "SelectorIssuerResolution", "SelectorRoleIndexError", "SelectorRoleResolution",
    "SELECTOR_ROLE_INDEX_VERSION", "SELECTOR_ROLE_POLICY_VERSION",
    "selector_lookup_text",
]
