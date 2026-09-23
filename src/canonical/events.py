"""사건 정체성과 관측 분리 — EventIdentity / EventObservation (S-09 · R-04 · R-05).

## 왜 나누는가

지금 `Event` 는 **문서 하나 = 사건 하나**다. 그런데 현실의 사건 하나는 여러 문서로 나뉜다.

```text
2024-10-15  단일판매ㆍ공급계약체결      계약금액 1,000억
2024-11-02  [정정] 계약체결            계약금액 1,200억
2025-03-11  단일판매ㆍ공급계약해지      해지
```

문서 단위로만 두면 「이 계약 지금 어떻게 됐어?」에 답할 수 없다. 셋이 **같은 사건**이라는
사실이 어디에도 없기 때문이다. 그래서 두 층으로 나눈다.

- `EventIdentity` — 현실의 사건 하나. 정정·해지를 거쳐도 **ID 가 유지된다**.
- `EventObservation` — 특정 문서가 그 사건에 대해 **그 시점에 말한 것**.

## 무엇으로 묶는가 — 그리고 무엇으로 묶지 않는가

| 관계 | 병합 | 이유 |
|---|---|---|
| `CORRECTS` · `SUPERSEDES` | **한다** | 정정은 정의상 같은 사건이다 |
| `RELATED` (해지 → 원계약) | **한다** | 해지 20건은 이 경로로만 원계약을 가리킨다 |
| `RELATED` (그 밖) | **안 한다** | 「관련 공시」는 보통 **다른** 계약을 참조한다 |

`RELATED` 를 무조건 병합하면 서로 다른 계약이 한 사건으로 뭉친다. 실측 `RELATED` 639건 중
해지 발 간선은 소수이므로, **해지 문서에서 나가는 간선만** 병합한다.

정체성 ID 는 회사·사건군과 계약상대·대상·사건일 같은 **안정 식별 필드**에서 만든다.
금액·비율·계약기간처럼 정정 가능한 값은 식별자에 넣지 않는다. 식별값이 부족하거나
서로 충돌하면 접수번호 범위의 provisional/ambiguous 키를 써 과병합을 막는다.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from typing import Any, Iterable, Mapping, MutableMapping

__all__ = [
    "EVENT_RESOLVER_VERSION", "EVENT_SUPPORT_VERSION", "EVENT_SUPPORT_ROLES",
    "RELATION_SUPPORT_VERSION", "RELATION_SUPPORT_ROLES",
    "UnionFind", "event_support_role", "event_support_decision",
    "is_declared_termination", "validate_event_history",
    "filter_relation_correction_support_pairs",
    "normalise_relation_support_pairs", "relation_correction_support_owner_keys",
    "relation_correction_support_sides", "relation_support_limitation",
    "add_identity_field", "build_identities",
]

_MERGE_TYPES = frozenset({"CORRECTS", "SUPERSEDES"})
EVENT_RESOLVER_VERSION = "event_identity/2.1-declared-termination"
EVENT_SUPPORT_VERSION = "event-support/1.0"

# EventObservation support는 자유 문자열 역할을 받지 않는다. 이 vocabulary 밖의 값은
# 빌드와 verifier 양쪽에서 거부한다. ``event_identity``와 상태 전이 근거는 별개라서
# 금액·해지 표지·정정 사유는 event_key 재료가 아니어도 support로는 보존한다.
EVENT_SUPPORT_ROLES = frozenset({
    "event_name", "counterparty", "target", "amount", "effective_date",
    "termination_marker", "termination_date", "termination_reason",
    "correction_reason", "correction_target",
})
RELATION_SUPPORT_VERSION = "relation-support/1.1-shared-side"
RELATION_SUPPORT_ROLES = EVENT_SUPPORT_ROLES | frozenset({
    "correction_before", "correction_after", "correction_before_after",
})
_RELATION_CORRECTION_ROLE_SIDES = {
    "correction_before": frozenset({"before"}),
    "correction_after": frozenset({"after"}),
    "correction_before_after": frozenset({"before", "after"}),
}


def relation_correction_support_sides(role: str) -> frozenset[str] | None:
    """Relation correction support role이 증명해야 하는 side 집합."""
    return _RELATION_CORRECTION_ROLE_SIDES.get(role)


def filter_relation_correction_support_pairs(
        pairs: Iterable[tuple[str, str]],
        owner_keys_by_evidence: Mapping[str, set[tuple[str, int]]],
        ) -> tuple[list[tuple[str, str]], set[str]]:
    """의미 CorrectionItem 소유자가 하나인 Relation support만 남긴다.

    한 실제 정정 셀은 여러 ``item_path``에서 재사용될 수 있고, 이 경우 동일한
    Evidence ID가 여러 CorrectionItem 행에 나타난다. Evidence 자체의 공유는
    유효하지만 Relation의 ``correction_before/after`` 역할은 어느 의미 항목을
    가리키는지 유일하지 않다. 따라서 다중 소유 ID를 citation support에서
    제외하고 Relation의 typed ``partial`` 상태를 유지한다.

    같은 *한* CorrectionItem의 before/after가 같은 Evidence를 공유하는 정상
    구조는 owner key가 하나이므로 보존되고, 뒤의 canonicalizer가
    ``correction_before_after``로 접는다.
    """
    kept: list[tuple[str, str]] = []
    excluded: set[str] = set()
    for evidence_id, role in pairs:
        if relation_correction_support_sides(role) is None:
            kept.append((evidence_id, role))
            continue
        if len(owner_keys_by_evidence.get(evidence_id, set())) == 1:
            kept.append((evidence_id, role))
        else:
            excluded.add(evidence_id)
    return kept, excluded


def relation_correction_support_owner_keys(
        rows: Iterable[Mapping[str, Any]],
        ) -> dict[str, set[tuple[str, int]]]:
    """Correction Evidence의 semantic owner를 ``(doc_id, order)``로 센다.

    ``block_id``는 대표 locator 기반이라 같은 실제 셀을 여러 의미 항목이 재사용하는
    바로 그 구조에서 같아질 수 있다. 따라서 owner cardinality key로 쓰면 2/4개
    semantic row가 하나로 접힌다. 한 행의 before/after가 같은 ID인 경우에는 owner key
    하나만 남겨 정상 shared-side 구조와 다중 의미 owner를 구분한다.
    """
    owners: dict[str, set[tuple[str, int]]] = defaultdict(set)
    seen_owner_keys: set[tuple[str, int]] = set()
    for row in rows:
        doc_id, order = row.get("doc_id"), row.get("order")
        if (not isinstance(doc_id, str) or not doc_id
                or type(order) is not int or order < 0):
            raise ValueError(
                f"CorrectionItem semantic owner key 오류: {doc_id!r}, {order!r}")
        owner_key = (doc_id, order)
        if owner_key in seen_owner_keys:
            raise ValueError(f"CorrectionItem semantic owner key 중복: {owner_key!r}")
        seen_owner_keys.add(owner_key)
        for side in ("before", "after"):
            evidence_id = row.get(f"{side}_evidence_id")
            if evidence_id is None:
                continue
            if not isinstance(evidence_id, str) or not evidence_id:
                raise ValueError(
                    f"CorrectionItem {side} evidence_id 오류: {evidence_id!r}")
            owners[evidence_id].add(owner_key)
    return dict(owners)


def relation_support_limitation(
        resolution_status: str, *, has_source_evidence: bool,
        has_ambiguous_correction_evidence_owner: bool) -> str:
    """Relation support의 typed limitation을 deterministic하게 만든다."""
    if not isinstance(resolution_status, str) or not resolution_status:
        raise ValueError(f"Relation resolution_status 오류: {resolution_status!r}")
    if (type(has_source_evidence) is not bool
            or type(has_ambiguous_correction_evidence_owner) is not bool):
        raise TypeError("Relation limitation 판정값은 bool이어야 합니다")
    limitations = [
        ("missing_exact_target_anchor" if resolution_status == "resolved"
         else f"relation_unresolved:{resolution_status}")
    ]
    if has_ambiguous_correction_evidence_owner:
        limitations.append("ambiguous_correction_evidence_owner")
    if not has_source_evidence:
        limitations.append("no_source_evidence")
    return "|".join(limitations)


def normalise_relation_support_pairs(
        pairs: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Relation support를 ID당 단일 role로 deterministic canonicalize한다.

    CorrectionItem의 before/after가 같은 raw 셀을 재사용하면 kind-bound Evidence ID도
    같다. 이 정상 구조만 combined role로 접고, Event support를 포함한 다른 복수-role
    충돌은 계속 fail-closed 한다.
    """
    roles_by_id: dict[str, set[str]] = defaultdict(set)
    for evidence_id, role in pairs:
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("Relation support evidence_id가 비어 있습니다")
        if role not in RELATION_SUPPORT_ROLES:
            raise ValueError(f"Relation support role 오류: {role!r}")
        roles_by_id[evidence_id].add(role)
    result: list[tuple[str, str]] = []
    for evidence_id, roles in roles_by_id.items():
        if roles == {"correction_before", "correction_after"}:
            role = "correction_before_after"
        elif len(roles) == 1:
            role = next(iter(roles))
        else:
            raise ValueError(
                f"Relation support Evidence ID role 충돌: {evidence_id} "
                f"({sorted(roles)!r})")
        result.append((evidence_id, role))
    return sorted(result)

# 식별자에 허용하는 축은 의도적으로 좁다. 아래 축 외 값(금액·비율·수량·기간·상태)은
# 아무리 검색에 유용해도 event_key 재료가 될 수 없다.
_IDENTITY_KEYS = ("subject", "counterparty", "target", "event_date")
_PLACEHOLDER = re.compile(
    r"^(?:[-–—]|\(?주\s*\d+\)?|해당\s*없음?|기재\s*생략|공시\s*유보|미정)$",
    re.IGNORECASE,
)
_CORRECTION_PREFIX = re.compile(r"^\s*\[[^]]+\]\s*")
_MAJOR_FORM = re.compile(r"^주요사항보고서\(([^()]*)\)$")
# 유형 끝에 명시된 해지만 상태 전이로 본다. ``해지 관련 검토``나
# 투자판단 제목의 설명 문구를 넓은 부분 검색으로 잡지 않는다.
_TERMINATION_TYPE = re.compile(r"해지(?:결정)?(?:자율공시)?$")


def _compact_label(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    # NFKC는 U+318D ``ㆍ``를 U+119E ``ᆞ``로 바꿀 수 있으므로 둘 다 제거한다.
    return re.sub(r"[\s\[\](){}<>._,:;·ㆍᆞ/\\-]+", "", value).casefold()


def _normalise_identity_value(key: str, value: str) -> str | None:
    """식별 필드 값의 보수적 정규화.

    회사명 구두점을 공격적으로 지우면 서로 다른 법인을 합칠 수 있으므로 Unicode와
    공백만 정리한다. 날짜만 표기 차이(`2024-10-14`, `2024.10.14`)를 제거한다.
    """
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if not text or _PLACEHOLDER.fullmatch(text):
        return None
    if key == "event_date":
        digits = re.sub(r"\D", "", text)
        if len(digits) != 8:
            return None
        try:
            year, month, day = int(digits[:4]), int(digits[4:6]), int(digits[6:])
        except ValueError:
            return None
        if not (1900 <= year <= 2200 and 1 <= month <= 12 and 1 <= day <= 31):
            return None
        return digits
    return text.casefold()


def _identity_key(path: str) -> str | None:
    """Field.path를 안정 식별 축으로 매핑한다.

    라벨은 DART 양식별로 번호·공백·괄호가 달라 compact 형태로 비교한다. 금액·비율·
    수량·기간 라벨은 애초 allowlist에 없으므로 수집되지 않는다.
    """
    label = _compact_label(path)
    if not label or "정정" in label:
        return None
    if any(token in label for token in (
            "계약상대", "거래상대방", "양도인", "양수인", "매수인", "매도인")):
        return "counterparty"
    if any(token in label for token in (
            "대상회사", "대상법인", "대상자산", "투자대상", "발행회사")):
        return "target"
    if any(token in label for token in (
            "체결계약명", "계약명", "계약내용", "양수목적", "양도목적",
            "취득목적", "처분목적", "투자목적")):
        return "subject"
    if any(token in label for token in (
            "계약수주일자", "계약일자", "이사회결의일결정일", "이사회결의일",
            "결정일자", "양수일자", "양도일자", "취득일자", "처분일자")):
        return "event_date"
    return None


def _support_leaf(path: str) -> str:
    """구조 경로의 마지막 라벨을 exact allowlist 비교 키로 만든다.

    부분문자열/fuzzy 매칭은 하지 않는다. DART parser가 붙이는 ``"3. "`` 같은
    ordinal prefix와 괄호·가운데점 표기만 구조적으로 정규화한다.
    """
    leaf = re.split(r"\s*>\s*", unicodedata.normalize("NFKC", path or ""))[-1]
    leaf = re.sub(r"^\s*[-–—]?\s*\d+(?:[-.)])?\s*", "", leaf)
    leaf = re.sub(r"^\s*[-–—]\s*", "", leaf)
    return _compact_label(leaf)


_EVENT_SUPPORT_LABELS: dict[str, str] = {
    # 사건 명칭
    "체결계약명": "event_name",
    "계약명": "event_name",
    "해지계약명": "event_name",
    "계약내용": "event_name",
    "신탁계약명": "event_name",
    # 코스닥 단일판매ㆍ공급계약 서식은 계약명을 `1. 판매ㆍ공급계약 내용` 으로
    # 적는다.  코스피의 `- 체결계약명` 과 같은 칸인데 어휘에 없어, 이 서식을
    # 쓰는 회사는 계약명이 통째로 사라졌다 — 사건을 이름으로 못 가른다.
    "판매공급계약내용": "event_name",
    # 상대·대상
    "계약상대": "counterparty",
    "계약상대방": "counterparty",
    "거래상대방": "counterparty",
    "해지기관": "counterparty",
    "양도인": "counterparty",
    "양수인": "counterparty",
    "매수인": "counterparty",
    "매도인": "counterparty",
    "대상회사": "target",
    "대상법인": "target",
    "대상자산": "target",
    "투자대상": "target",
    "발행회사": "target",
    # 금액 — 합산하거나 대표값으로 추정하지 않고 각각의 원문 Field만 연결한다.
    "계약금액원": "amount",
    "해지금액원": "amount",
    "투자금액원": "amount",
    "양수금액원": "amount",
    "양도금액원": "amount",
    "취득예정금액원": "amount",
    "처분예정금액원": "amount",
    # 유효일
    "계약수주일자": "effective_date",
    "계약일자": "effective_date",
    "결정일자": "effective_date",
    "이사회결의일결정일": "effective_date",
    "이사회결의일": "effective_date",
    "양수일자": "effective_date",
    "양도일자": "effective_date",
    "취득일자": "effective_date",
    "처분일자": "effective_date",
    # 상태 전이. title-derived ``is_termination``만으로는 이 역할을 채울 수 없다.
    "판매공급계약해지구분": "termination_marker",
    "해지일자": "termination_date",
    "해지예정일자": "termination_date",
    "해지주요사유": "termination_reason",
    "해지목적": "termination_reason",
    # 정정
    "정정사유": "correction_reason",
    "정정대상공시서류": "correction_target",
    "정정대상공시서류의최초제출일": "correction_target",
}


def event_support_role(path: str, *, value_status: object,
                       is_pii: object) -> str | None:
    """Field 한 행을 Event support role로 보수적으로 분류한다.

    exact leaf allowlist, usable value status, 명시적 ``is_pii=False`` 세 조건을 모두
    만족해야 한다. 호출자가 path/value로 사후 유사 매칭하는 우회로를 두지 않는다.
    """
    if type(is_pii) is not bool or is_pii:
        return None
    if value_status not in {"literal", "explicit_zero", "non_numeric"}:
        return None
    if not isinstance(path, str):
        return None
    role = _EVENT_SUPPORT_LABELS.get(_support_leaf(path))
    return role if role in EVENT_SUPPORT_ROLES else None


def add_identity_field(
    identity_fields: MutableMapping[str, MutableMapping[str, set[str]]],
    document: Mapping[str, Any],
    field: Mapping[str, Any],
    *,
    support_fields: MutableMapping[str, set[tuple[str, str]]] | None = None,
) -> bool:
    """빌드 중 Field 한 행에서 허용된 사건 식별값만 수집한다.

    반환값은 실제로 수집했는지 여부다. 원문 값은 Field/Evidence에 남고 여기에는
    정규화 값만 둔다. 금액 등 비허용 필드는 `False`로 건너뛴다.
    """
    if document.get("doc_group") == "periodic":
        return False
    rcept_no = str(document["rcept_no"])
    role = event_support_role(
        field.get("path"), value_status=field.get("value_status"),
        is_pii=field.get("is_pii"))
    evidence_id = field.get("evidence_id")
    if (support_fields is not None and role is not None
            and isinstance(evidence_id, str) and evidence_id):
        # Pair는 semantic Field를 쓰는 바로 이 시점에만 수집한다. 이후 값/라벨
        # 사후 검색으로 Evidence를 붙이지 않는다.
        support_fields.setdefault(rcept_no, set()).add((evidence_id, role))
    if field.get("is_pii") is not False:
        return False
    key = _identity_key(str(field.get("path") or ""))
    if key is None:
        return False
    if field.get("value_status") in {"not_reported", "empty", "parse_error"}:
        return False
    value = _normalise_identity_value(key, str(field.get("value_raw") or ""))
    if value is None:
        return False
    per_doc = identity_fields.setdefault(rcept_no, {})
    per_doc.setdefault(key, set()).add(value)
    return True


class UnionFind:
    """경로 압축 union-find. 정정 체인은 트리가 아니라 **임의 그래프**라 이게 필요하다."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:            # 경로 압축
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for x in list(self._parent):
            out[self.find(x)].append(x)
        return out


def _declared_event_type(doc: Mapping[str, Any]) -> str:
    """메타데이터가 명시한 사건 유형만 돌려준다.

    canonical build는 major/holding의 괄호 안 유형을 ``event_type``에,
    exchange의 유형을 ``doc_subtype``에 놓는다. 둘 다 없는 구버전 입력에서만
    정식 ``주요사항보고서(...)`` wrapper를 벗긴다. Field/본문은 절대 보지 않는다.
    """
    for key in ("event_type", "doc_subtype"):
        value = str(doc.get(key) or "").strip()
        if value:
            return value
    report = _CORRECTION_PREFIX.sub("", str(doc.get("report_nm") or "")).strip()
    matched = _MAJOR_FORM.fullmatch(report)
    # 자유 서술형 report_nm은 종료 판정 근거가 아니다. 예: 투자판단
    # 제목이 타사 계약 "해지"를 언급하는 경우. 구버전 fallback은 정식
    # ``주요사항보고서(유형)`` 완전 매치에서만 허용한다.
    return matched.group(1).strip() if matched else ""


def _is_termination(doc: Mapping[str, Any]) -> bool:
    declared = _compact_label(_declared_event_type(doc))
    return bool(declared and _TERMINATION_TYPE.search(declared))


def is_declared_termination(document: Mapping[str, Any]) -> bool:
    """공개된 strict predicate. title 자유문구가 아니라 declared type만 판정한다."""
    return _is_termination(document)


def event_support_decision(
        document: Mapping[str, Any], kind: str, roles: set[str] | frozenset[str],
        ) -> tuple[str, str | None]:
    """사건 유형별 필수 support role에서 typed status/limitation을 계산한다."""
    expected_kind = ("document_lineage"
                     if document.get("doc_group") == "periodic"
                     else "business_event")
    if kind != expected_kind:
        raise ValueError(
            "EventIdentity.kind가 Document.doc_group과 맞지 않습니다: "
            f"doc_group={document.get('doc_group')!r}, kind={kind!r}, "
            f"expected={expected_kind!r}")
    if kind == "document_lineage":
        return "not_applicable", None
    missing: list[str] = []
    if _is_termination(document):
        required = {"event_name", "termination_date", "termination_reason"}
        missing.extend(sorted(required - roles))
    elif "계약" in _compact_label(_family(dict(document))):
        required = {"event_name", "counterparty", "effective_date"}
        missing.extend(sorted(required - roles))
    else:
        if "effective_date" not in roles:
            missing.append("effective_date")
        if not ({"counterparty", "target"} & roles):
            missing.append("counterparty_or_target")
        if not ({"event_name", "target"} & roles):
            missing.append("event_name_or_target")
    if document.get("is_correction") and "correction_reason" not in roles:
        missing.append("correction_reason")
    missing = sorted(set(missing))
    if roles and not missing:
        return "fully_verified", None
    return (
        "partial",
        ("missing_required_roles:" + ",".join(missing)
         if missing else "no_allowlisted_non_pii_field_evidence"),
    )


def validate_event_history(
        identity: Mapping[str, Any], observations: Iterable[Mapping[str, Any]],
        documents: Mapping[str, Mapping[str, Any]]) -> None:
    """EventIdentity aggregate와 관측 순서를 Documents에서 exact 재계산한다.

    저장된 ``seq``나 aggregate를 정렬 근거로 다시 쓰지 않는다. 사건의 canonical
    순서는 ``(observed_at, rcept_no)``이고, previous pointer와 Identity 요약은 그
    순서 및 declared Document metadata에서만 도출한다.
    """
    event_key = identity.get("event_key")
    if not isinstance(event_key, str) or not event_key:
        raise ValueError("EventIdentity.event_key가 비어 있습니다")
    rows = list(observations)
    if not rows:
        raise ValueError(f"EventIdentity에 관측이 없습니다: {event_key}")
    for row in rows:
        if row.get("event_key") != event_key:
            raise ValueError(f"EventObservation event_key 불일치: {event_key}")
        if (not isinstance(row.get("observed_at"), str)
                or not row.get("observed_at")
                or not isinstance(row.get("rcept_no"), str)
                or not row.get("rcept_no")):
            raise ValueError(f"EventObservation 정렬 키가 비어 있습니다: {event_key}")
    ordered = sorted(rows, key=lambda row: (row["observed_at"], row["rcept_no"]))
    correction_count = 0
    terminated = False
    doc_group: str | None = None
    for expected_seq, row in enumerate(ordered):
        doc_id = row.get("doc_id")
        document = documents.get(str(doc_id or ""))
        if document is None:
            raise ValueError(f"EventObservation Document FK 없음: {event_key}/{doc_id}")
        if (document.get("rcept_no") != row.get("rcept_no")
                or document.get("rcept_dt") != row.get("observed_at")
                or type(document.get("is_correction")) is not bool
                or type(row.get("is_correction")) is not bool
                or row.get("is_correction") is not document.get("is_correction")
                or type(row.get("is_termination")) is not bool
                or row.get("is_termination") is not is_declared_termination(document)):
            raise ValueError(f"EventObservation Document metadata 불일치: {event_key}")
        current_group = document.get("doc_group")
        if not isinstance(current_group, str) or not current_group:
            raise ValueError(f"EventObservation Document group 누락: {event_key}")
        if doc_group is None:
            doc_group = current_group
        elif current_group != doc_group:
            raise ValueError(f"한 Event에 doc_group이 섞였습니다: {event_key}")
        expected_kind = ("document_lineage" if current_group == "periodic"
                         else "business_event")
        if (identity.get("kind") != expected_kind
                or identity.get("doc_group") != current_group):
            raise ValueError(f"EventIdentity kind/doc_group 불일치: {event_key}")
        expected_previous = (ordered[expected_seq - 1]["rcept_no"]
                             if expected_seq else None)
        if (type(row.get("seq")) is not int or row.get("seq") != expected_seq
                or row.get("previous_observation_rcept_no") != expected_previous):
            raise ValueError(
                f"EventObservation seq/previous 불일치: {event_key} "
                f"seq={row.get('seq')!r} expected={expected_seq}")
        correction_count += int(document["is_correction"])
        terminated |= bool(row["is_termination"])

    aggregates = {
        "root_rcept_no": ordered[0]["rcept_no"],
        "first_disclosed_at": ordered[0]["observed_at"],
        "last_disclosed_at": ordered[-1]["observed_at"],
        "n_observations": len(ordered),
        "n_corrections": correction_count,
        "status_at_corpus_end": "terminated" if terminated else "active",
    }
    for name, expected in aggregates.items():
        actual = identity.get(name)
        if isinstance(expected, int) and type(actual) is not int:
            raise ValueError(f"EventIdentity {name} 타입 불일치: {event_key}")
        if actual != expected:
            raise ValueError(
                f"EventIdentity {name} aggregate 불일치: {event_key} "
                f"({actual!r} != {expected!r})")


def _merge_harmless_ambiguous(uf: "UnionFind", relations: list[dict],
                              by_rcept: dict[str, dict]) -> int:
    """**모호해도 결과가 같은 정정은 합친다.**

    정정공시가 「2024-10-02 제출본」이라고만 밝히고 그날 후보가 둘이면
    어느 쪽인지 고를 수 없다. 하지만 **후보 둘이 이미 한 사건 안에 있다면**
    (원본과 그 원본의 정정본처럼) 어느 쪽에 붙여도 같은 사건으로 합쳐진다.

    그래서 **고르지 않고 합치기만 한다.** `dst_rcept_no` 는 여전히 비워 둔다 —
    「어느 문서를 정정했는지」는 모르는 채로 남기고, 「어느 사건인지」만 확정한다.
    후보가 서로 다른 사건에 걸쳐 있으면 손대지 않는다. 근거 없이 고르면
    엉뚱한 사건이 합쳐지고, 그 결과는 보고서상 정상으로 보인다.
    """
    merged = 0
    for r in relations:
        if (r.get("relation_type") not in _MERGE_TYPES
                or r.get("resolution_status") != "ambiguous"):
            continue
        src = r.get("src_rcept_no")
        if src not in by_rcept:
            continue
        cands = [c for c in (r.get("candidate_ids") or ()) if c in by_rcept]
        if len(cands) < 2 or len({uf.find(c) for c in cands}) != 1:
            continue
        uf.union(cands[0], src)
        merged += 1
    return merged


def _family(doc: dict) -> str:
    """공시 유형의 **계약 계열**. `단일판매공급계약체결` 과 `…해지` 는 같은 계열이다."""
    sub = str(doc.get("event_type") or doc.get("doc_subtype") or "")
    if not sub:
        # major/holding manifest의 doc_subtype가 비어 있는 구버전 입력도 최소한 보고서
        # 유형별로 분리한다. 괄호 안 세부 유형이 있으면 그것을 우선한다.
        report_nm = re.sub(r"^\[[^]]+\]\s*", "", str(doc.get("report_nm") or ""))
        match = re.search(r"\(([^()]*)\)\s*$", report_nm)
        sub = (match.group(1) if match else report_nm).strip()
    # major 신탁계약은 ``...체결결정`` / ``...해지결정``이다.
    # ``결정``만 떼면 두 family가 여전히 다르므로 짝이 되는 exact
    # suffix를 긴 순서로 먼저 제거한다.
    for suffix in ("체결결정", "해지결정", "체결", "해지", "변경"):
        if sub.endswith(suffix):
            return sub[: -len(suffix)]
    return sub


def _values_for(
    identity_fields: Mapping[str, Mapping[str, Any]], rcept_no: str, key: str,
) -> set[str]:
    raw = identity_fields.get(rcept_no, {}).get(key)
    if raw is None:
        return set()
    values = raw if isinstance(raw, (set, frozenset, list, tuple)) else (raw,)
    out: set[str] = set()
    for value in values:
        normalised = _normalise_identity_value(key, str(value))
        if normalised is not None:
            out.add(normalised)
    return out


def _identity_material(
    docs: list[dict], identity_fields: Mapping[str, Mapping[str, Any]], kind: str,
) -> tuple[str, str, dict[str, str], dict[str, list[str]]]:
    """`(status, fingerprint, unique_values, conflicts)`를 계산한다.

    원공시(비정정·비해지)가 하나라도 있으면 그 관측들만 식별자에 쓴다. 정정본의 값으로
    기존 식별자가 바뀌는 일을 막기 위해서다. 원공시가 코퍼스 밖인 경우에만 나머지
    관측을 fallback으로 쓴다.
    """
    root = docs[0]
    family_name = _compact_label(_family(root)) or "unknown"
    family = f"{str(root.get('doc_group') or 'unknown')}:{family_name}"
    primary = [d for d in docs if not d.get("is_correction") and not _is_termination(d)]
    selected = primary or docs

    unique: dict[str, str] = {}
    conflicts: dict[str, list[str]] = {}
    for key in _IDENTITY_KEYS:
        values: set[str] = set()
        for doc in selected:
            values.update(_values_for(identity_fields, str(doc["rcept_no"]), key))
        if len(values) == 1:
            unique[key] = next(iter(values))
        elif len(values) > 1:
            conflicts[key] = sorted(values)

    # resolved는 보수적으로 판정한다. 계약은 제목·상대·사건일 세 축이 모두 있어야 하며,
    # 그 밖의 비정기 사건도 사건일 + 대상/상대 + 사건 설명이 필요하다.
    contract = "계약" in family
    if kind == "document_lineage" or not root.get("corp_code"):
        status = "provisional"
    elif conflicts:
        status = "ambiguous"
    elif contract:
        status = ("resolved" if {"subject", "counterparty", "event_date"} <= unique.keys()
                  else "provisional")
    else:
        party = "counterparty" in unique or "target" in unique
        description = "subject" in unique or "target" in unique
        status = ("resolved" if party and description and "event_date" in unique
                  else "provisional")

    material: dict[str, Any] = {
        "corp_code": str(root.get("corp_code") or ""),
        "family": family,
        "identity": unique,
    }
    if conflicts:
        material["conflicts"] = conflicts
    canonical = json.dumps(material, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return status, fingerprint, unique, conflicts


def _event_key(status: str, fingerprint: str, root_rcept_no: str) -> str:
    # 해결된 키만 접수번호와 독립적이다. provisional/ambiguous는 과병합 방지를 위해
    # 현재 lineage 범위에 묶는다. 향후 식별자가 보강되면 resolved 키로 승격될 수 있다.
    material = (f"EVENT|RESOLVED|{fingerprint}" if status == "resolved"
                else f"EVENT|{status.upper()}|{fingerprint}|{root_rcept_no}")
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:32]


def build_identities(
    documents: list[dict],
    relations: list[dict],
    bump=None,
    identity_fields: Mapping[str, Mapping[str, Any]] | None = None,
    support_fields: Mapping[str, Any] | None = None,
) -> tuple[list[dict], list[dict]]:
    """`(identities, observations)`.

    `documents` 는 manifest 레코드, `relations` 는 relation jsonl 행이다.
    `identity_fields` 는 접수번호별 안정 식별 축(`subject`, `counterparty`, `target`,
    `event_date`)이다. 빌더는 :func:`add_identity_field` 로 이 allowlist만 수집한다.
    """
    by_rcept = {d["rcept_no"]: d for d in documents}
    uf = UnionFind()
    for d in documents:
        uf.find(d["rcept_no"])                    # 고립 문서도 자기 사건을 갖는다

    # 해지가 가리키는 것 중 **같은 계약 유형**만 후보다.
    # Freudenberg 해지는 「단일판매공급계약체결」과 「투자판단관련주요경영사항
    # (구매 공급에 대한 합의서 체결)」을 함께 참조한다 — 사업상 연관이지만 다른 공시다.
    # 유형을 안 보면 둘을 한 사건으로 합치거나(과병합), 유형 구분 없이 막으면
    # 해지가 정작 그 계약에도 못 붙는다.
    term_targets: dict[str, set[str]] = defaultdict(set)
    for r in relations:
        src = r.get("src_rcept_no")
        dst = r.get("dst_rcept_no")
        if (r.get("relation_type") != "RELATED"
                or r.get("resolution_status") != "resolved" or not dst
                or src not in by_rcept or dst not in by_rcept
                or not _is_termination(by_rcept[src])):
            continue
        if _family(by_rcept[src]) == _family(by_rcept[dst]):
            term_targets[src].add(dst)

    for r in relations:
        src, dst = r.get("src_rcept_no"), r.get("dst_rcept_no")
        if not dst or src not in by_rcept or dst not in by_rcept:
            continue
        if r.get("resolution_status") != "resolved":
            continue                              # 미해결 간선으로 사건을 합치지 않는다
        rtype = r.get("relation_type")
        if rtype in _MERGE_TYPES:
            uf.union(dst, src)                    # 대상(더 이른 것)을 뿌리로

    # **해지의 RELATED 는 정정을 다 반영한 뒤에 본다.** 해지 문서는 원 체결과
    # 그 정정본들을 **함께** 참조하는 것이 정상이다 — 그때는 이미 한 덩어리다.
    # 서로 **다른 덩어리**를 가리킬 때만 합치지 않는다. 「참조가 둘 이상이면 금지」로
    # 막으면 Freudenberg 처럼 정상 사건이 해지만 떨어져 나간다.
    for src, targets in term_targets.items():
        live = [t for t in targets if t in by_rcept]
        if not live or len({uf.find(t) for t in live}) != 1:
            continue                              # 서로 다른 계약 — 합치지 않는다
        uf.union(live[0], src)

    # 확정 간선을 모두 반영한 **뒤에** 봐야 「후보가 한 사건인지」를 알 수 있다
    harmless = _merge_harmless_ambiguous(uf, relations, by_rcept)
    if bump is not None:
        bump(f"event:ambiguous_merged={harmless}")

    identity_fields = identity_fields or {}
    support_fields = support_fields or {}
    groups: list[dict[str, Any]] = []
    for members in uf.groups().values():
        docs = sorted((by_rcept[m] for m in members if m in by_rcept),
                      key=lambda d: (d["rcept_dt"], d["rcept_no"]))
        if not docs:
            continue
        root = docs[0]
        terminated = any(_is_termination(d) for d in docs)
        # **문서 계보와 현실 사건은 다르다.** 사업보고서 + 정정본은 계보로는 맞지만
        # 「사건」이 아니다. 빼지 않고 무엇인지 밝힌다 — `status_at_corpus_end` 개명과 같은 처리.
        kind = "document_lineage" if root["doc_group"] == "periodic" else "business_event"
        status, fingerprint, _unique, _conflicts = _identity_material(
            docs, identity_fields, kind)
        groups.append({
            "docs": docs,
            "identity_status": status,
            "identity_fingerprint": fingerprint,
            "kind": kind,
            "terminated": terminated,
        })

    # 서로 연결되지 않은 두 lineage가 같은 resolved fingerprint를 가질 때 자동 병합하지
    # 않는다. 실제 코퍼스에는 같은 회사·계약명·상대·계약일인 별도 계약도 있다. 둘 다
    # ambiguous로 내려 서로 다른 event_key를 유지하고, 추후 식별 축 보강 대상으로 남긴다.
    resolved_by_fingerprint: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        if group["identity_status"] == "resolved":
            resolved_by_fingerprint[group["identity_fingerprint"]].append(idx)
    for indexes in resolved_by_fingerprint.values():
        if len(indexes) > 1:
            for idx in indexes:
                groups[idx]["identity_status"] = "ambiguous"

    identities: list[dict] = []
    observations: list[dict] = []
    for group in groups:
        docs = group["docs"]
        root = docs[0]
        status = group["identity_status"]
        fingerprint = group["identity_fingerprint"]
        event_key = _event_key(status, fingerprint, root["rcept_no"])
        identities.append({
            "event_key": event_key, "kind": group["kind"],
            "corp_code": root["corp_code"], "corp_name": root["corp_name"],
            "doc_group": root["doc_group"], "doc_subtype": root.get("doc_subtype"),
            "root_rcept_no": root["rcept_no"],
            "identity_fingerprint": fingerprint,
            "identity_status": status,
            "resolver_version": EVENT_RESOLVER_VERSION,
            "first_disclosed_at": root["rcept_dt"],
            "last_disclosed_at": docs[-1]["rcept_dt"],
            "n_observations": len(docs),
            "n_corrections": sum(1 for d in docs if d.get("is_correction")),
            # **코퍼스 끝 시점의 상태다.** 시점별 상태는 관측을 걸러 계산한다.
            "status_at_corpus_end": "terminated" if group["terminated"] else "active",
        })
        for seq, d in enumerate(docs):
            if group["kind"] == "document_lineage":
                support_pairs: list[tuple[str, str]] = []
                support_status = "not_applicable"
                support_limitation = None
            else:
                raw_pairs = support_fields.get(str(d["rcept_no"]), ())
                support_by_id: dict[str, str] = {}
                for evidence_id, role in raw_pairs:
                    if (not isinstance(evidence_id, str) or not evidence_id
                            or role not in EVENT_SUPPORT_ROLES):
                        continue
                    previous_role = support_by_id.get(evidence_id)
                    if previous_role is not None and previous_role != role:
                        raise ValueError(
                            "Event support Evidence ID에 role이 둘입니다: "
                            f"{evidence_id} ({previous_role}, {role})")
                    support_by_id[evidence_id] = str(role)
                support_pairs = sorted(support_by_id.items())
                roles = {role for _, role in support_pairs}
                support_status, support_limitation = event_support_decision(
                    d, group["kind"], roles)
            observations.append({
                "event_key": event_key, "seq": seq,
                "doc_id": d["doc_id"], "rcept_no": d["rcept_no"],
                "observed_at": d["rcept_dt"],
                "is_correction": d.get("is_correction", False),
                "is_termination": _is_termination(d),
                #: 직전 관측. 시간순 재생의 연결고리다 (R-05)
                "previous_observation_rcept_no": docs[seq - 1]["rcept_no"] if seq else None,
                "supporting_evidence_ids": [item[0] for item in support_pairs],
                "support_roles": [item[1] for item in support_pairs],
                "support_status": support_status,
                "support_version": EVENT_SUPPORT_VERSION,
                "support_limitation": support_limitation,
            })
    return identities, observations
