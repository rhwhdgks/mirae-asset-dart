"""대량보유상황보고서 전용 조회 도구.

발행회사와 보고자를 별도 축으로 선택하고, 반복되는 한글 라벨 대신 DART
ACODE를 우선한다. Stage1이 보고자와 정정 유효본을 해소해 공개 QueryPlan의
정확한 ``doc_id + rcept_no``로 내려보내므로, 실행 시 질문 원문이나 별도 선택
sidecar에 의존하지 않는다. 회사명·질문·접수번호별 예외표는 두지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import re

from agent.stage1_v1_holding_backend import (
    _SLOT_ALIASES as _STAGE1_SLOT_ALIASES,
)
from app.orchestrator.payload import (
    AnswerClaim, ClaimCitation, Clarification, Limitation, TraceEvent,
)
from app.textkit import (
    mask_confirmed_person_name, mask_holding_subject_name,
    party_is_private_person,
)
from src.canonical.security import party_is_person


@dataclass(frozen=True)
class HoldingPartyRow:
    """한 원문 행 안에서만 결합한 보고자/특별관계자 보유내역."""

    table_locator: str
    logical_row: int
    name: object
    relation: object | None
    count: object | None
    ratio: object | None
    nationality: object | None = None
    occupation: object | None = None
    #: 같은 원문 행의 ``구분``(``SPC_TP``) 원문. 이 자리에 법인과 자연인이
    #: 함께 오므로 값 모양이 아니라 공시의 자기 신고로 가른다 (이슈 #199).
    party_type: str | None = None


@dataclass(frozen=True)
class HoldingPartyValue:
    value_text: str
    source_rows: tuple[object, ...]


_SLOT_ACODES: dict[str, tuple[str, str, str | None]] = {
    "previous_count": ("SUM_BMT_CNT", "직전 보고서 보유주식등 수", "주"),
    "previous_ratio": ("SUM_BMT_RT", "직전 보고서 보유비율", "%"),
    "current_count": ("SUM_TMT_CNT", "이번 보고서 보유주식등 수", "주"),
    "current_ratio": ("SUM_TMT_RT", "이번 보고서 보유비율", "%"),
    "delta_count": ("MDF_STK_CNT", "보유주식등 수 증감", "주"),
    # 공시의 증감 비율은 두 비율의 차이이므로 사용자 표현은 %p다.
    "delta_ratio": ("MDF_STK_RT", "보유비율 증감", "%p"),
    "report_reason": ("SUM_CHN_RWN", "보고 사유", None),
    "holding_purpose": ("HLD_OBJ_DTL", "보유 목적", None),
    "change_method": ("CHN_HOW", "변동 방법", None),
    "change_reason": ("CHN_RSN", "변동 사유", None),
}

#: 도구에만 있는 표현들.  Stage1 표(`_STAGE1_SLOT_ALIASES`)와 합쳐서 쓴다 —
#: 두 표를 따로 관리하면 Stage1 은 slot 을 결속하는데 여기서는 못 찾아 기본
#: slot 으로 되돌아가고, 그러면 **거절이 아니라 묻지 않은 값을 답한다.**
#: `RPC-010` 이 「몇 %p 변했는지」에 이번 보유비율을 답한 것이 그것이다.
_TOOL_SLOT_ALIASES: dict[str, tuple[str, ...]] = {
    "previous_count": ("직전보유주식수", "직전보유량", "직전주식수", "직전보고서주식수"),
    "previous_ratio": ("직전보유비율", "직전지분율", "직전보고서비율"),
    "current_count": ("이번보유주식수", "이번보유량", "보유주식수", "보유량", "이번보고서주식수"),
    # 별칭 표가 `agent/stage1_v1_holding_backend.py` 에도 있다.  둘이 어긋나면
    # Stage1 은 slot 을 결속하고 여기서는 못 찾아 기본 slot 으로 되돌아간다 —
    # `RPC-010` 이 그래서 「몇 %p 변했는지」에 이번 보유비율만 답했다.  한쪽만
    # 고치지 말 것.  `_key` 가 `%` 를 지우므로 별칭은 원문대로 적는다.
    "current_ratio": (
        "이번보유비율", "이번지분율", "보유비율", "지분율", "이번보고서비율",
        "총 보유비율", "총 지분율", "합계 보유비율",
    ),
    "delta_count": ("증감주식수", "보유주식수증감", "지분증감량", "감소량", "증가량"),
    "delta_ratio": (
        "증감비율", "지분율증감", "보유비율증감", "증감률",
        "%p 변화", "%p 변동", "%p 증감", "퍼센트포인트 변화",
        "몇 %p 변했는지", "몇 %p 변했나", "몇 %p 변했어",
        "몇 %p 변했나요", "몇 %p 변동",
    ),
    "report_reason": ("보고사유", "보고이유"),
    "holding_purpose": ("보유목적", "보유목적상세"),
    "change_method": ("변동방법", "변경방법"),
    "change_reason": ("변동사유", "변경사유"),
    "issuer": ("발행회사", "대상회사"),
    "filer": ("보고자", "대량보유자"),
    "report_type": ("보고유형", "보고서유형", "일반약식", "서식"),
    "receipt": ("접수번호", "공시번호"),
    "receipt_date": ("접수일", "공시일"),
    "base_date": ("보고서작성기준일", "작성기준일", "보고기준일", "기준일"),
    "parties": ("특별관계자", "특별관계자별보유내역", "관계자별보유내역"),
    "public_entities": ("공개법인명", "법인명", "공개기관명", "기관명"),
    "nationality": (
        "국적", "특별관계자국적", "특별관계자별국적"),
    "occupation": (
        "직업", "사업내용", "직업사업내용", "특별관계자직업",
        "특별관계자별직업", "업종"),
    "filer_nationality": ("보고자국적", "보고자의국적"),
    "filer_occupation": (
        "보고자직업", "보고자의직업", "보고자직업사업내용",
        "보고자의직업사업내용"),
    # 사용자 요청에서 제한된 개인정보 field를 제외했다는 실행 제어 slot.
    # 값 조회나 answer root를 만들지 않고 안전 고지만 생성한다.
    "privacy_notice": ("개인정보제외고지",),
}

_SLOT_ALIASES: dict[str, tuple[str, ...]] = {
    slot: tuple(dict.fromkeys(_STAGE1_SLOT_ALIASES.get(slot, ()) + aliases))
    for slot, aliases in _TOOL_SLOT_ALIASES.items()
}

_SUMMARY_SURFACES = {
    "보유주식등의수및보유비율": (
        "previous_count", "previous_ratio", "current_count", "current_ratio",
        "delta_count", "delta_ratio"),
    "직전이번보유량": (
        "previous_count", "previous_ratio", "current_count", "current_ratio"),
    "주식수와지분율": ("current_count", "current_ratio"),
}


def _key(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(value or "")).casefold()


def _organization_key(value: object) -> str:
    """Collapse legal-form spelling variants without merging distinct firms."""

    compact = _key(value)
    for token in ("주식회사", "유한회사", "사단법인", "재단법인"):
        compact = compact.replace(token, "")
    return compact


def _is_private_person_row(
        name_row: object, party_type: str | None = None,
        corporate_names=(), *, name: str | None = None) -> bool:
    """이 보유 행의 주체가 자연인인가 — 판정은 한 곳에서만 한다.

    ``구분``(``SPC_TP``/``CRP_TP``) → 정본 판 1.5 의 ``pii_type`` → 옛 정본의
    값 휴리스틱 순이다. 자세한 근거는
    :func:`agent.holding_subject_mask.party_is_private_person`.
    """

    return party_is_private_person(
        name_row, party_type, corporate_names, name=name)


def _resolved_party_name(row: object, party_type: str | None = None) -> str:
    """Match key for a party's SPC_NM row, recovering a legacy-masked org name.

    A row frozen under a legacy security policy (``pii-prompt-safe/1.1``) can
    mask an institutional 특별관계자/보고자 name as ``[REDACTED:PERSON_NAME]``
    because that frozen policy's corporate-suffix boundary predates suffixes
    such as ``기금`` (RPC-017 — 국민연금기금). Stage1 already resolved and
    proved this exact party under the *current* boundary before requesting
    it by name here, so a masked row must still be matchable — an actually
    masked natural person's row stays masked (``.value`` is unchanged; only
    this match key looks at ``value_raw``, and only once
    ``is_organization_name`` independently reproves it).
    """

    masked = str(getattr(row, "value", None) or "").strip()
    if not masked.startswith("[REDACTED"):
        return masked
    raw = str(getattr(row, "value_raw", None) or "").strip()
    if not raw:
        return masked
    return masked if _is_private_person_row(row, party_type, name=raw) else raw


def _decode_requested_slot(value: object) -> tuple[str, str | None]:
    text = str(value or "")
    prefix = "holding-party:"
    if not text.startswith(prefix):
        return text, None
    try:
        payload = json.loads(text[len(prefix):])
    except (TypeError, ValueError):
        return text, None
    if (not isinstance(payload, dict) or set(payload) != {"slot", "party_name"}
            or not isinstance(payload["slot"], str)
            or not isinstance(payload["party_name"], str)
            or not payload["party_name"].strip()):
        return text, None
    return payload["slot"], payload["party_name"].strip()


def _canonical_slots(slots) -> tuple[str, ...]:
    out: list[str] = []
    for surface in slots or ():
        decoded, _party = _decode_requested_slot(surface)
        key = _key(decoded)
        expanded = _SUMMARY_SURFACES.get(key)
        if expanded:
            for slot in expanded:
                if slot not in out:
                    out.append(slot)
            continue
        # 부분일치를 정확일치와 같은 자리에서 보면 「보유비율증감」이 먼저
        # 만나는 `보유비율`(current_ratio)에 붙어 증감 대신 이번 값을 답한다.
        # 정확히 맞는 이름을 다 본 뒤에야 부분일치로 내려간다.
        exact = next((name for name, aliases in _SLOT_ALIASES.items()
                      if key == _key(name)
                      or any(key == _key(value) for value in aliases)), None)
        if exact is None:
            exact = next((name for name, aliases in _SLOT_ALIASES.items()
                          if any(len(_key(value)) >= 4 and _key(value) in key
                                 for value in aliases)), None)
        if exact is not None and exact not in out:
            out.append(exact)
    return tuple(out)


def _citation(row, *, excerpt_prompt_safe: str | None = None) -> ClaimCitation:
    return ClaimCitation(
        doc_id=row.doc_id, rcept_no=row.rcept_no,
        evidence_id=row.evidence_id, locator=row.locator,
        excerpt_prompt_safe=(excerpt_prompt_safe if excerpt_prompt_safe is not None
                             else row.value_prompt_safe or row.value or ""),
    )


def _known_private_party_names(party_rows: tuple[HoldingPartyRow, ...]) -> tuple[str, ...]:
    """Return only source-row names that canonical security already masked.

    Change-reason prose is not a name-labelled field, so a generic Korean-name
    pattern would both overreach and risk removing ordinary words. A holding
    table row, however, supplies a document-scoped identity and its canonical
    projection tells us whether that identity is private. Reuse precisely
    those raw names when they reappear verbatim in a free-text reason.
    """

    names: list[str] = []
    for party in party_rows:
        shown = str(getattr(party.name, "value", "") or "").strip()
        raw = str(getattr(party.name, "value_raw", "") or "").strip()
        if (raw and raw != shown
                and _is_private_person_row(
                    party.name, party.party_type, name=raw)
                and raw not in names):
            names.append(raw)
    return tuple(names)


def _mask_known_private_party_mentions(
        value: object, party_rows: tuple[HoldingPartyRow, ...],
        ) -> tuple[str, bool]:
    """Mask exact, document-scoped private-party names in holding prose."""

    text = str(value or "")
    masked = text
    for name in _known_private_party_names(party_rows):
        masked = masked.replace(name, "[REDACTED:PERSON_NAME]")
    return masked, masked != text


def _metadata_citation(meta, text: str) -> ClaimCitation:
    return ClaimCitation(
        doc_id=meta.doc_id, rcept_no=meta.rcept_no,
        locator="document", excerpt_prompt_safe=text,
        verification="source_roundtrip",
    )


def _report_type(meta) -> str:
    name = _key(meta.report_nm)
    return "약식" if "약식" in name else "일반" if "일반" in name else "확인 불가"


class HoldingTool:
    """Holding 문서를 metadata-first로 고른 뒤 ACODE와 행 좌표로 조회한다."""

    def __init__(self, rm, field_index):
        self.rm = rm
        self.fidx = field_index

    @staticmethod
    def _document_selector(task):
        return getattr(task, "document_selector", None)

    def _candidate_documents(self, task):
        document_selector = self._document_selector(task)
        rows = list(self.rm.documents(
            as_of=task.as_of, corp_code=task.corp_code, doc_group="holding",
            report_name_contains=(getattr(document_selector, "report_name_contains", None)
                                  if document_selector else None),
            is_correction=(getattr(document_selector, "is_correction", None)
                           if document_selector else None),
        ))
        if document_selector is not None:
            if document_selector.rcept_no:
                rows = [row for row in rows
                        if row.rcept_no == document_selector.rcept_no]
            if document_selector.doc_id:
                rows = [row for row in rows if row.doc_id == document_selector.doc_id]
            if document_selector.rcept_from:
                rows = [row for row in rows
                        if row.rcept_dt >= document_selector.rcept_from]
            if document_selector.rcept_to:
                rows = [row for row in rows
                        if row.rcept_dt <= document_selector.rcept_to]
        if document_selector is not None and document_selector.form:
            form_key = _key(document_selector.form)
            if "일반" in form_key or "약식" in form_key:
                marker = "약식" if "약식" in form_key else "일반"
                rows = [row for row in rows if marker in row.report_nm]
            else:
                rows = [row for row in rows if _key(row.form) == form_key]
        return rows

    def _effective_documents(self, rows, *, as_of: str):
        """정정 사슬별 최신 유효본. ambiguous/root-missing은 임의 선택하지 않는다."""

        by_receipt = {row.rcept_no: row for row in rows}
        selected: dict[str, object] = {}
        unresolved: list[Limitation] = []
        limitation_keys: set[tuple[str, str]] = set()
        for row in rows:
            summaries = ()
            relation_summaries = getattr(self.rm, "relation_summaries", None)
            if callable(relation_summaries) and row.is_correction:
                summaries = relation_summaries(
                    source_rcept_no=row.rcept_no, as_of=as_of)
            ambiguous_relation = any(
                summary.resolution_status in {"ambiguous", "invalid"}
                for summary in summaries)
            root_missing = any(
                summary.resolution_status == "root_missing"
                or summary.root_missing_reason
                for summary in summaries)
            if ambiguous_relation:
                key = ("holding_lineage_ambiguous", row.rcept_no)
                if key not in limitation_keys:
                    limitation_keys.add(key)
                    unresolved.append(Limitation(
                        code=key[0],
                        detail=(f"{row.rcept_no}의 정정 대상이 하나로 확정되지 않아 "
                                "해당 문서 값을 최신 유효값으로 사용하지 않았습니다."),
                        affected_doc_ids=[row.rcept_no]))
                continue
            if root_missing:
                key = ("holding_lineage_root_missing", row.rcept_no)
                if key not in limitation_keys:
                    limitation_keys.add(key)
                    unresolved.append(Limitation(
                        code=key[0],
                        detail=(f"{row.rcept_no}의 원보고서가 제공 코퍼스에 없어 "
                                "확인되는 정정 문서 내용만 사용했습니다."),
                        affected_doc_ids=[row.rcept_no]))
            result = self.rm.resolve_document_version(row.rcept_no, as_of=as_of)
            if result.status == "ok" and result.selected:
                meta = by_receipt.get(result.selected)
                if meta is None:
                    # 최신본이 selector의 correction flag 밖에 있더라도 같은
                    # 발행회사의 holding 문서에서 exact receipt로만 되찾는다.
                    meta = next((candidate for candidate in self.rm.documents(
                        as_of=as_of, corp_code=row.corp_code,
                        doc_group="holding")
                        if candidate.rcept_no == result.selected), None)
                if meta is not None:
                    selected[result.selected] = meta
            elif result.status in {"ambiguous", "invalid"}:
                key = ("holding_lineage_ambiguous", row.rcept_no)
                if key in limitation_keys:
                    continue
                limitation_keys.add(key)
                unresolved.append(Limitation(
                    code="holding_lineage_ambiguous",
                    detail=(f"{row.rcept_no}의 정정 계보를 기준시점 현재 하나의 "
                            "유효본으로 확정할 수 없습니다."),
                    affected_doc_ids=[row.rcept_no]))
            elif row.is_correction:
                key = ("holding_lineage_root_missing", row.rcept_no)
                if key in limitation_keys:
                    continue
                limitation_keys.add(key)
                unresolved.append(Limitation(
                    code="holding_lineage_root_missing",
                    detail=(f"{row.rcept_no}의 원보고서가 제공 코퍼스에 없어 "
                            "확인되는 정정 문서만 사용할 수 있습니다."),
                    affected_doc_ids=[row.rcept_no]))
                selected[row.rcept_no] = row
        return sorted(selected.values(), key=lambda row: (
            row.rcept_dt, row.rcept_no)), unresolved

    @staticmethod
    def _clarification(task, docs) -> Clarification:
        filers = sorted({row.filer for row in docs if row.filer})
        if len(filers) > 1:
            return Clarification(
                clarification_id=f"{task.task_id}.holding.filer",
                question=(f"{task.corp_name}의 대량보유보고서는 보고자가 여러 명입니다. "
                          "어느 보고자의 공시를 확인할까요?"),
                targets=["filer"], options={"filer": filers[:20]},
            )
        receipts = [f"{row.rcept_dt} · {row.rcept_no}" for row in docs[-20:]]
        return Clarification(
            clarification_id=f"{task.task_id}.holding.receipt",
            question="확인할 대량보유보고서의 접수일 또는 접수번호를 선택해 주세요.",
            targets=["receipt"], options={"receipt": receipts},
        )

    def _verified_acode(self, task, receipt: str, acode: str):
        rows = [row for row in self.fidx.rows(
            task.corp_code, as_of=task.as_of, rcept_no=receipt,
            doc_group="holding") if row.acode == acode]
        if not rows:
            return None, "not_found"
        # 요약 ACODE는 원칙상 한 칸이다. 중복이 있어도 값과 path가 같을 때만
        # 동일 사실의 복제본으로 보고 첫 좌표를 사용한다.
        identities = {(_key(row.path), row.value) for row in rows}
        if len(identities) != 1:
            return None, "ambiguous"
        row = self.fidx.verified(sorted(
            rows, key=lambda value: (value.order, value.occurrence))[0])
        return row, ("verified" if row.evidence_status == "verified" else "unverified")

    def _base_date(self, task, receipt: str):
        rows = [row for row in self.fidx.rows(
            task.corp_code, as_of=task.as_of, rcept_no=receipt,
            doc_group="holding")
            if row.aunit in {"THS_RPT_DT", "IFR_BASE"}
            or _key(row.label) in {"보고서작성기준일", "보고기준일"}]
        # The previous-report comparison row has the same visible label.
        # Prefer the typed current-report date; never mix it with BFR_RPT_DT.
        current = [row for row in rows if row.aunit in {"THS_RPT_DT", "IFR_BASE"}]
        rows = current or [row for row in rows
                           if "직전보고서" not in _key(row.path)
                           and row.aunit != "BFR_RPT_DT"]
        values = {(row.aunitvalue or row.value or "").strip() for row in rows
                  if (row.aunitvalue or row.value or "").strip() not in {"", "-"}}
        if len(values) != 1:
            return None, "ambiguous" if values else "not_found"
        row = self.fidx.verified(sorted(rows, key=lambda value: value.order)[0])
        return row, ("verified" if row.evidence_status == "verified" else "unverified")

    def _verified_filer_attribute(
            self, task, receipt: str, acode: str,
            ) -> tuple[object | None, str]:
        """Read one reporter-header attribute, never a related-party row.

        ``IFR_NT``/``IFR_JOB`` belong to the singular filer metadata block.
        The similarly worded ``SPC_NT``/``JOB`` cells are repeated related-party
        rows and must not be substituted when the header value is absent or
        redacted.
        """

        rows = [row for row in self.fidx.rows(
            task.corp_code, as_of=task.as_of, rcept_no=receipt,
            doc_group="holding") if row.acode == acode]
        usable = [row for row in rows
                  if str(row.value or "").strip() not in {"", "-"}]
        identities = {(_key(row.path), str(row.value or "").strip())
                      for row in usable}
        if not usable:
            return None, "not_found"
        if len(identities) != 1:
            return None, "ambiguous"
        row = self.fidx.verified(sorted(
            usable, key=lambda value: (value.order, value.occurrence))[0])
        if row.evidence_status != "verified":
            return None, "unverified"
        if str(row.value or "").strip().startswith("[REDACTED:"):
            return row, "redacted"
        return row, "verified"

    def _party_rows(self, task, receipt: str) -> tuple[HoldingPartyRow, ...]:
        rows = list(self.fidx.rows(
            task.corp_code, as_of=task.as_of, rcept_no=receipt,
            doc_group="holding"))
        grouped: dict[tuple[str, int], dict[str, object]] = {}
        for row in rows:
            if row.logical_row is None or not row.table_locator:
                continue
            key = (row.table_locator, row.logical_row)
            bucket = grouped.setdefault(key, {})
            if row.acode == "SPC_NM":
                bucket["name"] = row
            elif row.aunit == "SPC_TP":
                # 판별표는 `성 명(명칭)` 과 `구분` 을 같은 행에 둔다. 값
                # 모양이 아니라 이 신고가 자연인/법인을 정한다 (이슈 #199).
                bucket.setdefault("party_type", row)
            elif row.acode in {"SPC_RLT", "RELATION", "SPC_REL"} or "관계" in row.label:
                bucket.setdefault("relation", row)
            elif row.acode == "SPC_NT":
                bucket.setdefault("nationality", row)
            elif row.acode in ("JOB", "IFR_JOB"):
                bucket.setdefault("occupation", row)
            elif row.acode == "STK_CNT" and "합계" in _key(row.path):
                bucket["count"] = row
            elif row.acode == "STK_RT" and "합계" in _key(row.path):
                bucket["ratio"] = row
        output: list[HoldingPartyRow] = []
        for (table, logical_row), bucket in sorted(grouped.items()):
            name = bucket.get("name")
            if name is None or not (name.value or "").strip() or name.value == "-":
                continue
            party_type_row = bucket.get("party_type")
            output.append(HoldingPartyRow(
                table, logical_row, name, bucket.get("relation"),
                bucket.get("count"), bucket.get("ratio"),
                bucket.get("nationality"), bucket.get("occupation"),
                (str(getattr(party_type_row, "value", None) or "").strip() or None
                 if party_type_row is not None else None)))
        return tuple(output)

    def _party_value(self, task, receipt: str, *, party_name: str, slot: str):
        """Read one named party only from a same-row holding/change total."""

        if slot not in {
                "previous_count", "previous_ratio", "current_count",
                "current_ratio", "delta_count", "delta_ratio"}:
            return None, "unsupported"

        def select(*, change: bool):
            candidates = []
            for party in self._party_rows(task, receipt):
                if _key(_resolved_party_name(
                        party.name, party.party_type)) != _key(party_name):
                    continue
                value_row = party.count if slot.endswith("count") else party.ratio
                if value_row is None:
                    continue
                is_change = "증감주식등의내역" in _key(value_row.path)
                if is_change != change or value_row.value in {None, "", "-"}:
                    continue
                candidates.append(value_row)
            identities = {(row.table_locator, row.logical_row, row.value)
                          for row in candidates}
            if not candidates:
                return None, "not_found"
            if len(identities) != 1:
                return None, "ambiguous"
            verified = self.fidx.verified(candidates[0])
            if verified.evidence_status != "verified":
                return None, "unverified"
            return verified, "verified"

        if slot.startswith("current_") or slot.startswith("delta_"):
            row, status = select(change=slot.startswith("delta_"))
            return ((HoldingPartyValue(row.value, (row,)), status)
                    if row is not None else (None, status))

        current, current_status = select(change=False)
        delta, delta_status = select(change=True)
        if current is None or delta is None:
            return None, (current_status if current is None else delta_status)
        try:
            current_value = Decimal(str(current.value).replace(",", ""))
            delta_value = Decimal(str(delta.value).replace(",", ""))
        except InvalidOperation:
            return None, "non_numeric"
        previous = current_value - delta_value
        if slot.endswith("count"):
            text = f"{previous:,.0f}"
        else:
            text = format(previous.normalize(), "f")
        return HoldingPartyValue(text, (current, delta)), "verified"

    def _filer_party_type(self, meta) -> str | None:
        """이 보고서의 ``보고자 구분``(``aunit=CRP_TP``) 원문. 문서당 한 칸이다.

        DART가 문서마다 「개인(국내)」·「국내법인」·「연기금등 전문투자자」
        처럼 명시해 둔다. 접미사 없는 맨 그룹명(``영풍``·``두산``)은 값의
        글자 수만으로는 사람 이름과 구별할 수 없는데 이 필드는 정확히 갈라
        준다. 선례는 ``agent/stage1_v1_holding_backend.py::_filer_corp_type``.
        """

        cache = self.__dict__.setdefault("_filer_party_type_cache", {})
        if len(cache) >= 512:
            cache.clear()
        key = str(getattr(meta, "rcept_no", "") or "")
        if key in cache:
            return cache[key]
        try:
            rows = list(self.fidx.rows(
                meta.corp_code, as_of=meta.rcept_dt, rcept_no=meta.rcept_no,
                doc_group="holding"))
        except Exception:
            return None
        found = None
        for row in rows:
            if str(getattr(row, "aunit", "") or "") != "CRP_TP":
                continue
            text = str(getattr(row, "value", None) or "").strip()
            if text and text != "-":
                found = text
                break
        if key:
            cache[key] = found
        return found

    def _masked_filer(self, meta) -> tuple[str | None, bool]:
        """보고자 성명을 답변 표면에 실을 형태로 부분 가림한다.

        ``meta.filer`` 는 documents.parquet 문서 레벨 보고자로, 개인 이름일
        때 그대로 노출되면 이슈 #51 이 관측한 것과 같은 노출이 된다. 공시가
        스스로 적어 둔 ``보고자 구분``이 법인·기관이라고 하면 값 모양
        휴리스틱을 건너뛰고(``영풍`` 오탐 방지), 개인이라고 하면 라틴 표기
        이름까지 반드시 가린다 (이슈 #199). 구분을 못 찾은 문서에서만 종전
        값 휴리스틱으로 내려간다.
        """

        filer_text = meta.filer
        if not filer_text:
            return filer_text, False
        decided = party_is_person(self._filer_party_type(meta))
        if decided is False:
            return filer_text, False
        if decided is True:
            masked = mask_confirmed_person_name(filer_text)
            return masked, masked != filer_text
        masked = mask_holding_subject_name(
            filer_text, corporate_names=(meta.corp_name,))
        return masked, masked != filer_text

    def _metadata_claims(
            self, task, meta, slots: tuple[str, ...], output_by_slot,
            filer_text):
        claims: list[AnswerClaim] = []
        values = {
            "issuer": ("발행회사", meta.corp_name),
            "filer": ("보고자", filer_text),
            "report_type": ("보고서 유형", _report_type(meta)),
            "receipt": ("접수번호", meta.rcept_no),
            "receipt_date": ("접수일", meta.rcept_dt),
        }
        for slot in slots:
            if slot not in values:
                continue
            label, text = values[slot]
            claims.append(AnswerClaim(
                output_id=(output_by_slot.get(slot)
                           or f"{task.task_id}.{meta.rcept_no}.{slot}"),
                label=f"{meta.corp_name} {label}", text=text,
                citations=[_metadata_citation(meta, f"{label}: {text}")]))
        return claims

    def run_task(self, task, *, trace, corp_name: str):
        claims: list[AnswerClaim] = []
        limitations: list[Limitation] = []
        used: list[str] = []
        docs = self._candidate_documents(task)
        docs, lineage_limitations = self._effective_documents(
            docs, as_of=task.as_of)
        limitations.extend(lineage_limitations)
        if not docs:
            limitations.append(Limitation(
                code="not_found_holding_report",
                detail="조건에 맞는 대량보유상황보고서를 기준시점까지 찾지 못했습니다."))
            return claims, limitations, used, None

        # list는 정정 계보별 유효본을 모두 반환한다. 서로 다른 원보고서를 같은
        # 보고자라는 이유로 하나로 접으면 독립된 보고 이력이 사라진다.
        if task.operation == "list":
            docs = sorted(docs, key=lambda row: (
                _key(row.filer), _report_type(row), row.rcept_dt, row.rcept_no))
        elif len({row.filer for row in docs}) > 1:
            return claims, limitations, used, self._clarification(task, docs)
        elif len(docs) > 1:
            # Stage1 exact selector가 아닌 넓은 lookup은 최신을 임의 선택하지 않는다.
            explicit_receipt = bool(getattr(
                self._document_selector(task), "rcept_no", None))
            if explicit_receipt:
                docs = [max(docs, key=lambda row: (row.rcept_dt, row.rcept_no))]
            else:
                return claims, limitations, used, self._clarification(task, docs)

        slots = _canonical_slots(task.requested_slots)
        if not slots:
            slots = ("issuer", "filer", "receipt_date", "current_count", "current_ratio")
        output_by_slot: dict[str, str] = {}
        for output in tuple(getattr(task, "field_outputs", ()) or ()):
            decoded, party_name = _decode_requested_slot(output.slot)
            normalized = _canonical_slots((decoded,))
            if len(normalized) == 1:
                key = (f"{normalized[0]}@{_key(party_name)}"
                       if party_name else normalized[0])
                output_by_slot[key] = output.output_id

        party_requests: list[tuple[str, str]] = []
        aggregate_slots: set[str] = set()
        for raw_slot in task.requested_slots:
            decoded, party_name = _decode_requested_slot(raw_slot)
            normalized = _canonical_slots((decoded,))
            if party_name and len(normalized) == 1:
                party_requests.append((normalized[0], party_name))
            elif not party_name:
                aggregate_slots.update(normalized)

        for meta in docs:
            used.append(meta.rcept_no)
            embedded_personal_data_omitted = False
            filer_text, filer_masked = self._masked_filer(meta)
            claims.extend(self._metadata_claims(
                task, meta, slots, output_by_slot, filer_text))
            if (filer_masked
                    and not any(
                        row.code == "holding_subject_name_partially_masked"
                        for row in limitations)):
                limitations.append(Limitation(
                    code="holding_subject_name_partially_masked",
                    detail="보고자 성명은 개인정보 보호를 위해 일부만 표시했습니다.",
                    affected_doc_ids=[meta.rcept_no]))
            if "base_date" in slots:
                row, status = self._base_date(task, meta.rcept_no)
                if row is not None and status == "verified":
                    value = row.aunitvalue or row.value
                    claims.append(AnswerClaim(
                        output_id=(output_by_slot.get("base_date")
                                   or f"{task.task_id}.{meta.rcept_no}.base_date"),
                        label=f"{meta.corp_name} 보고서 작성기준일",
                        text=value, citations=[_citation(row)]))
                else:
                    limitations.append(Limitation(
                        code="holding_slot_unavailable",
                        detail=f"{meta.rcept_no}의 보고서 작성기준일을 확정하지 못했습니다.",
                        affected_doc_ids=[meta.rcept_no]))
            filer_attributes = {
                "filer_nationality": ("IFR_NT", "국적"),
                "filer_occupation": ("IFR_JOB", "직업(사업내용)"),
            }
            for slot, (acode, label) in filer_attributes.items():
                if slot not in slots:
                    continue
                # 개인 보고자의 직업·국적을 새 경로로 되살리지 않는다. 기존
                # 보고자 이름 마스킹 판정과 같은 문서 구분을 그대로 따른다.
                if filer_masked or party_is_person(
                        self._filer_party_type(meta)) is True:
                    if not any(row.code == "personal_data_omitted"
                               for row in limitations):
                        limitations.append(Limitation(
                            code="personal_data_omitted",
                            detail=("개인인 보고자의 직업·국적은 개인정보 보호를 "
                                    "위해 제시하지 않았습니다."),
                            affected_doc_ids=[meta.rcept_no]))
                    continue
                row, status = self._verified_filer_attribute(
                    task, meta.rcept_no, acode)
                if row is not None and status == "verified":
                    claims.append(AnswerClaim(
                        output_id=(output_by_slot.get(slot)
                                   or f"{task.task_id}.{meta.rcept_no}.{slot}"),
                        label=f"{filer_text or meta.corp_name} {label}",
                        text=str(row.value).strip(), citations=[_citation(row)]))
                    continue
                detail = (
                    f"{meta.rcept_no}의 보고자 {label}은 제공 자료에서 가려져 있어 "
                    "확인하지 못했습니다."
                    if status == "redacted" else
                    f"{meta.rcept_no}에서 보고자 {label}을 하나로 확정하지 "
                    f"못했습니다({status}).")
                limitations.append(Limitation(
                    code=(
                        "holding_filer_occupation_redacted"
                        if status == "redacted" and slot == "filer_occupation"
                        else "holding_filer_nationality_redacted"
                        if status == "redacted" and slot == "filer_nationality"
                        else "holding_slot_unavailable"),
                    detail=detail,
                    affected_doc_ids=[meta.rcept_no]))
            for slot in slots:
                if slot not in aggregate_slots and any(requested_slot == slot for requested_slot, _party in party_requests):
                    continue
                spec = _SLOT_ACODES.get(slot)
                if spec is None:
                    continue
                acode, label, unit = spec
                row, status = self._verified_acode(task, meta.rcept_no, acode)
                if row is None or status != "verified" or row.value is None:
                    limitations.append(Limitation(
                        code="holding_slot_unavailable",
                        detail=f"{meta.rcept_no}에서 {label}을 확정하지 못했습니다({status}).",
                        affected_doc_ids=[meta.rcept_no]))
                    continue
                numeric = bool(re.fullmatch(r"[()\-+0-9,.]+", row.value.strip()))
                text = None if numeric else row.value
                citation = _citation(row)
                if slot == "change_reason" and not numeric:
                    # `CHN_RSN` is free prose and can repeat a private holder
                    # name. Bind masking only to names from this filing's
                    # already-masked party rows; do not guess Korean names.
                    text, embedded_personal_data_omitted = (
                        _mask_known_private_party_mentions(
                            row.value, self._party_rows(task, meta.rcept_no)))
                    citation = _citation(row, excerpt_prompt_safe=text)
                claims.append(AnswerClaim(
                    output_id=(output_by_slot.get(slot)
                               or f"{task.task_id}.{meta.rcept_no}.{slot}"),
                    label=(f"보고자 및 특별관계자 합계 {label}"
                           if party_requests and slot in {
                               "previous_count", "previous_ratio", "current_count", "current_ratio"}
                           else f"{filer_text or meta.corp_name} {label}"),
                    value_text=row.value if numeric else None,
                    raw_unit=unit if numeric else None,
                    canonical_unit=unit if numeric else None,
                    text=text,
                    citations=[citation]))
            if (embedded_personal_data_omitted
                    and not any(row.code == "personal_data_omitted"
                                for row in limitations)):
                limitations.append(Limitation(
                    code="personal_data_omitted",
                    detail=("변동 사유에 포함된 개인 특별관계자의 성명은 제외하고 "
                            "공개된 변동 사유만 제시했습니다.")))
            for slot, party_name in party_requests:
                party_value, status = self._party_value(
                    task, meta.rcept_no, party_name=party_name, slot=slot)
                if party_value is None or status != "verified":
                    limitations.append(Limitation(
                        code="holding_slot_unavailable",
                        detail=(f"{meta.rcept_no}에서 {party_name}의 {slot} 값을 "
                                f"하나로 확정하지 못했습니다({status})."),
                        affected_doc_ids=[meta.rcept_no]))
                    continue
                unit = "%p" if slot == "delta_ratio" else "%" if slot.endswith(
                    "ratio") else "주"
                claims.append(AnswerClaim(
                    output_id=(output_by_slot.get(f"{slot}@{_key(party_name)}")
                               or f"{task.task_id}.{meta.rcept_no}.{slot}.{_key(party_name)}"),
                    label=f"{party_name} " + {
                        "previous_count": "직전 보고서 보유주식등 수",
                        "previous_ratio": "직전 보고서 보유비율",
                        "current_count": "보유주식등 수",
                        "current_ratio": "보유비율",
                        "delta_count": "주식등 수 증감",
                        "delta_ratio": "보유비율 증감",
                    }[slot],
                    value_text=party_value.value_text,
                    raw_unit=unit, canonical_unit=unit,
                    citations=[_citation(row) for row in party_value.source_rows]))
            if "parties" in slots:
                party_rows = tuple(
                    party for party in self._party_rows(task, meta.rcept_no)
                    if party.count is not None and party.ratio is not None
                    and "증감주식등의내역" not in _key(party.count.path)
                )
                if not party_rows:
                    limitations.append(Limitation(
                        code="holding_party_rows_unavailable",
                        detail=f"{meta.rcept_no}에서 특별관계자별 행을 안전하게 결합하지 못했습니다.",
                        affected_doc_ids=[meta.rcept_no]))
                party_lines: list[str] = []
                party_citations: list[ClaimCitation] = []
                private_names_omitted = False
                for party in party_rows:
                    verified = [self.fidx.verified(row) for row in (
                        party.name, party.relation, party.count, party.ratio) if row is not None]
                    if any(row.evidence_status != "verified" for row in verified):
                        continue
                    name = str(party.name.value or "").strip()
                    if _is_private_person_row(
                            party.name, party.party_type,
                            (meta.corp_name, meta.filer)):
                        # Repeated holding rows can contain private persons.
                        # Do not emit either the name or its citation excerpt;
                        # keeping only the numeric row would falsely attribute
                        # an anonymous value, so omit that row as one unit.
                        private_names_omitted = True
                        continue
                    pieces = [name]
                    if party.relation and party.relation.value not in {None, "", "-"}:
                        pieces.append(f"관계 {party.relation.value}")
                    if party.count and party.count.value not in {None, "", "-"}:
                        pieces.append(f"{party.count.value}주")
                    if party.ratio and party.ratio.value not in {None, "", "-"}:
                        pieces.append(f"{party.ratio.value}%")
                    party_lines.append(
                        f"{len(party_lines) + 1}. " + ", ".join(pieces))
                    party_citations.extend(_citation(row) for row in verified)
                # ``parties``는 QueryPlan의 필드 하나다. 행마다 임의 output_id를
                # 만들면 digest-bound answer root(value-N)를 만족하지 못하므로,
                # 검증된 행을 한 개의 목록 claim으로 묶어 그 root에 정확히 결속한다.
                if party_lines:
                    claims.append(AnswerClaim(
                        output_id=(output_by_slot.get("parties")
                                   or f"{task.task_id}.{meta.rcept_no}.parties"),
                        label="특별관계자별 보유내역",
                        text="\n".join(party_lines), citations=party_citations))
                if (private_names_omitted
                        and not any(row.code == "personal_data_omitted"
                                    for row in limitations)):
                    limitations.append(Limitation(
                        code="personal_data_omitted",
                        detail=("개인 특별관계자의 성명과 해당 행은 제외하고 "
                                "공개된 법인·기관 보유내역만 제시했습니다.")))
            if "public_entities" in slots:
                organizations: list[str] = []
                organization_citations: list[ClaimCitation] = []
                organization_positions: dict[str, int] = {}
                for party in self._party_rows(task, meta.rcept_no):
                    verified_name = self.fidx.verified(party.name)
                    name = str(verified_name.value or "").strip()
                    if (verified_name.evidence_status != "verified" or not name
                            or _is_private_person_row(
                                verified_name, party.party_type)):
                        continue
                    organization_key = _organization_key(name)
                    if not organization_key:
                        continue
                    position = organization_positions.get(organization_key)
                    if position is None:
                        organization_positions[organization_key] = len(organizations)
                        organizations.append(name)
                        organization_citations.append(_citation(verified_name))
                    elif len(name) < len(organizations[position]):
                        # The same source can spell one organization both as
                        # ``삼성물산`` and ``삼성물산주식회사``.  Show the shorter
                        # public label once and keep its matching citation.
                        organizations[position] = name
                        organization_citations[position] = _citation(verified_name)
                if organizations:
                    claims.append(AnswerClaim(
                        output_id=(output_by_slot.get("public_entities")
                                   or f"{task.task_id}.{meta.rcept_no}.public_entities"),
                        label="공개된 특별관계자 법인·기관명",
                        text="\n".join(
                            f"{index}. {name}"
                            for index, name in enumerate(organizations, start=1)),
                        citations=organization_citations))
                else:
                    limitations.append(Limitation(
                        code="holding_party_rows_unavailable",
                        detail=(f"{meta.rcept_no}에서 공개 법인·기관명을 "
                                "확정하지 못했습니다."),
                        affected_doc_ids=[meta.rcept_no]))

            if "nationality" in slots:
                # 국적(`SPC_NT`)은 보고자·특별관계자 표의 행별 값이다. 한 칸으로
                # 확정되지 않으므로 `_verified_acode` 를 쓰지 않고 행을 그대로
                # 늘어놓는다.  개인 행은 `public_entities` 와 같은 기준으로
                # 빼서, 국적으로 자연인을 좁혀낼 수 없게 한다.
                nationality_lines: list[str] = []
                nationality_citations: list[ClaimCitation] = []
                nationality_private_omitted = False
                for party in self._party_rows(task, meta.rcept_no):
                    if party.nationality is None:
                        continue
                    verified_name = self.fidx.verified(party.name)
                    name = str(verified_name.value or "").strip()
                    if verified_name.evidence_status != "verified" or not name:
                        continue
                    if _is_private_person_row(verified_name, party.party_type):
                        nationality_private_omitted = True
                        continue
                    verified_nationality = self.fidx.verified(party.nationality)
                    value = str(verified_nationality.value or "").strip()
                    if (verified_nationality.evidence_status != "verified"
                            or not value or value == "-"):
                        continue
                    nationality_lines.append(
                        f"{len(nationality_lines) + 1}. {name}, {value}")
                    nationality_citations.append(_citation(verified_nationality))
                if nationality_lines:
                    claims.append(AnswerClaim(
                        output_id=(output_by_slot.get("nationality")
                                   or f"{task.task_id}.{meta.rcept_no}.nationality"),
                        label="보고자·특별관계자별 국적",
                        text="\n".join(nationality_lines),
                        citations=nationality_citations))
                    if (nationality_private_omitted
                            and not any(row.code == "personal_data_omitted"
                                        for row in limitations)):
                        limitations.append(Limitation(
                            code="personal_data_omitted",
                            detail=("개인인 보고자·특별관계자의 행은 제외하고 "
                                    "공개된 법인·기관의 국적만 제시했습니다.")))
                else:
                    limitations.append(Limitation(
                        code="holding_party_rows_unavailable",
                        detail=(f"{meta.rcept_no}에서 국적을 확정하지 못했습니다."),
                        affected_doc_ids=[meta.rcept_no]))

            if "occupation" in slots:
                # 지분공시의 `직 업(사업내용)` 은 법인에게는 업종이고 자연인에게는
                # 직위다. 정본이 주체를 보고 이미 갈라 두었으므로(이슈 #139)
                # 여기서는 **가려진 값을 그대로 빼기만** 한다 — 다시 판정하면
                # 두 층이 어긋난다.
                occupation_lines: list[str] = []
                occupation_citations: list[ClaimCitation] = []
                occupation_private_omitted = False
                for party in self._party_rows(task, meta.rcept_no):
                    if party.occupation is None:
                        continue
                    verified_name = self.fidx.verified(party.name)
                    name = str(verified_name.value or "").strip()
                    if verified_name.evidence_status != "verified" or not name:
                        continue
                    if _is_private_person_row(verified_name, party.party_type):
                        occupation_private_omitted = True
                        continue
                    verified_job = self.fidx.verified(party.occupation)
                    value = str(verified_job.value or "").strip()
                    if (verified_job.evidence_status != "verified"
                            or not value or value == "-"):
                        continue
                    if value.startswith("[REDACTED"):
                        # 정본이 자연인의 것으로 판정해 가린 값이다.
                        occupation_private_omitted = True
                        continue
                    occupation_lines.append(
                        f"{len(occupation_lines) + 1}. {name}, {value}")
                    occupation_citations.append(_citation(verified_job))
                if occupation_lines:
                    claims.append(AnswerClaim(
                        output_id=(output_by_slot.get("occupation")
                                   or f"{task.task_id}.{meta.rcept_no}.occupation"),
                        label="보고자·특별관계자별 직업(사업내용)",
                        text="\n".join(occupation_lines),
                        citations=occupation_citations))
                    if (occupation_private_omitted
                            and not any(row.code == "personal_data_omitted"
                                        for row in limitations)):
                        limitations.append(Limitation(
                            code="personal_data_omitted",
                            detail=("개인인 보고자·특별관계자의 직업은 제외하고 "
                                    "공개된 법인·기관의 사업내용만 제시했습니다.")))
                else:
                    limitations.append(Limitation(
                        code="holding_party_rows_unavailable",
                        detail=(f"{meta.rcept_no}에서 직업(사업내용)을 "
                                "확정하지 못했습니다."),
                        affected_doc_ids=[meta.rcept_no]))

        if ("privacy_notice" in slots
                and not any(row.code == "personal_data_omitted"
                            for row in limitations)):
            limitations.append(Limitation(
                code="personal_data_omitted",
                detail="개인 식별정보와 개인 연락처는 제외하고 공개된 지분 정보만 제시했습니다."))
        trace.append(TraceEvent(
            seq=len(trace) + 1, stage="tool",
            summary=(f"holding documents={len(docs)} claims={len(claims)} "
                     f"limitations={len(limitations)}"),
            detail={"task_id": task.task_id,
                    "receipts": [row.rcept_no for row in docs]}))
        return claims, limitations, used, None


def is_holding_task(task) -> bool:
    if getattr(task, "kind", None) != "disclosure":
        return False
    selector = getattr(task, "document_selector", None)
    if getattr(selector, "doc_group", None) == "holding":
        return True
    return False
