"""Exact Stage1 authority for large-shareholding disclosures.

The public QueryPlan has no filer axis.  This backend therefore resolves the
issuer, filer and correction-effective receipt before compilation, then emits
an internal ``HoldingDisclosureResolution``.  The compiler lowers that typed
authority to the existing exact ``ResolvedDisclosureTask`` contract.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
import re
import unicodedata
from typing import Any

from agent.deterministic_plan_compiler_v1 import AuthoritativeResolution
from agent.holding_subject_mask import (
    MASK_CHAR,
    mask_confirmed_person_name,
    mask_holding_subject_name,
    party_is_private_person,
    unmasked_personal_name_candidates,
)
from agent.planning import _target_date_range
from agent.semantic_intent_v1 import SemanticIntent, semantic_intent_digest
from agent.stage1_v1_resolver import (
    ClarificationAuthority,
    ClarificationOption,
    ClarificationSlot,
    TerminalAuthority,
)
from src.canonical.security import (
    classify_privacy_request,
    holding_party_types_by_row,
    party_is_person,
    resolve_party_type,
)


_SLOT_ACODES: dict[str, str | None] = {
    "previous_count": "SUM_BMT_CNT",
    "previous_ratio": "SUM_BMT_RT",
    "current_count": "SUM_TMT_CNT",
    "current_ratio": "SUM_TMT_RT",
    "delta_count": "MDF_STK_CNT",
    "delta_ratio": "MDF_STK_RT",
    "report_reason": "SUM_CHN_RWN",
    "holding_purpose": "HLD_OBJ_DTL",
    "change_method": "CHN_HOW",
    "change_reason": "CHN_RSN",
    "issuer": None,
    "filer": None,
    "report_type": None,
    "receipt": None,
    "receipt_date": None,
    "base_date": None,
    "parties": None,
    "public_entities": None,
    # 보고자 자신의 국적·직업은 특별관계자 반복 행(`SPC_*`/`JOB`)과 다른
    # 문서 머리말 좌표(`IFR_NT`/`IFR_JOB`)다. 답변 주체를 잃지 않도록 별도
    # 슬롯으로 전달하고, 값은 실행 도구가 선택 문서에서 다시 검증한다.
    "filer_nationality": None,
    "filer_occupation": None,
    # 국적(`SPC_NT`)은 보고자·특별관계자 표의 **행별** 값이라 한 칸으로 확정되지
    # 않는다. `parties`·`public_entities` 와 같은 목록 슬롯이므로 ACODE 를 두지
    # 않는다 — 넣으면 「holding metadata/party slot에는 ACODE를 넣지 않습니다」로
    # 계약이 닫힌다. 행 결합은 도구(`app/tools/holding.py`)의 `_party_rows` 가 한다.
    "nationality": None,
    # 직업(사업내용)도 행별 값이라 목록 슬롯이다. 정본은 주체가 법인일 때만
    # 값을 남긴다 — 자연인의 직위는 `[REDACTED:OCCUPATION]` 이다 (이슈 #139).
    "occupation": None,
}

_SLOT_ALIASES: dict[str, tuple[str, ...]] = {
    "previous_count": ("직전보유주식수", "직전보유량", "직전주식수"),
    "previous_ratio": ("직전보유비율", "직전지분율"),
    "current_count": (
        "이번보유주식수", "보유주식수", "보유량", "주식을몇주보유",
        "몇주보유",
    ),
    # `_key` 는 `%` 를 지우므로 별칭은 원문 그대로 적고 정규화에 맡긴다.
    "current_ratio": (
        "이번보유비율", "이번지분율", "보유비율", "지분율",
        "총 보유비율", "총 지분율", "합계 보유비율",
    ),
    "delta_count": (
        "증감주식수", "변동주식수", "보유주식수증감", "지분증감량",
        "증가량", "감소량", "몇주변동", "얼마나변했어",
        "얼마나변했나요", "얼마나변동",
    ),
    # 공시는 증감 비율을 `MDF_STK_RT` 로 이미 적어 둔다 — 계산할 것이 없다.
    # 그런데 「몇 %p 변했는지」로 물으면 별칭이 없어 결속에 실패했고, HCX 가
    # 스스로 0.01 을 말하려다 근거 없는 숫자로 막혔다(`RPC-010`).  묻는 말을
    # 알아듣지 못한 것이지 자료가 없는 것이 아니었다.
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
    "parties": (
        "특별관계자", "특별관계자별보유내역", "관계자별보유내역",
        "특별관계자별보유비율", "특별관계자보유비율들",
    ),
    "nationality": (
        "국적", "특별관계자국적", "특별관계자별국적", "국적들",
    ),
    "occupation": (
        "직업", "사업내용", "직업사업내용", "특별관계자직업",
        "특별관계자별직업", "업종",
    ),
    "filer_nationality": ("보고자국적", "보고자의국적"),
    "filer_occupation": (
        "보고자직업", "보고자의직업", "보고자직업사업내용",
        "보고자의직업사업내용",
    ),
    "public_entities": (
        "공개법인명", "공개된법인명", "법인명", "공개회사명",
        "공개된회사명", "공개기관명", "공개된기관명", "기관명",
        "공개된법인기관명", "공개법인기관명", "법인기관명",
    ),
}

_SUMMARY_SLOTS: dict[str, tuple[str, ...]] = {
    "보유주식등의수및보유비율": (
        "previous_count", "previous_ratio", "current_count", "current_ratio",
        "delta_count", "delta_ratio"),
    "직전이번보유량": (
        "previous_count", "previous_ratio", "current_count", "current_ratio"),
    "주식수와지분율": ("current_count", "current_ratio"),
}

_PARTY_VALUE_SLOTS = frozenset({
    "previous_count", "previous_ratio", "current_count", "current_ratio",
    "delta_count", "delta_ratio",
})

_QUESTION_DATE = re.compile(
    r"20[0-9]{2}년(?:\s*(?:1[0-2]|0?[1-9])월"
    r"(?:\s*(?:3[01]|[12][0-9]|0?[1-9])일)?)?"
)

#: 이슈 #62 — 「20.75%에서 20.74%로 바뀐」·「상대 증감률」·「%p」처럼 두 보유비율
#: 시점(직전·이번)을 함께 요구하는 신호. 「직전」·「이번」 낱말이 「보유비율」
#: 바로 앞에 붙어 있지 않은 실제 질문(예: RPC-011)에서도 작동해야 하므로,
#: 기존 단일 `보유\s*비율` 캡처와 달리 문장 전체에서 상대적 변화 어휘를 본다.
_RATIO_CHANGE_CUE = re.compile(
    r"%\s*p|퍼센트\s*포인트|상대\s*증감률|증감\s*률|"
    r"증감\s*비율|지분율\s*증감|보유비율\s*증감|"
    r"(?:비율|%)[^,.?]{0,25}(?:에서|→)[^,.?]{0,25}(?:로|으로)"
    r"[^,.?]{0,15}(?:바뀌|바뀐|변경|변동|변했|변화)"
)

#: 위 신호가 있을 때, 「20.75%에서」·「20.74%로」처럼 값 자체에 붙은 조사로
#: 직전·이번 두 시점을 가리키는 리터럴 구간을 그대로 뽑는다. 컴파일러의
#: 결속(grounding) 검사는 output.field_surfaces 각 항목이 질문 원문의
#: 부분 문자열이어야 하므로, 「직전」·「이번」 같은 낱말이 질문에 없어도
#: (RPC-011처럼) 실제 값+조사 구간은 항상 원문 그대로다.
_RATIO_FROM_VALUE_SPAN = re.compile(r"\d+(?:\.\d+)?\s*%\s*에서")
_RATIO_TO_VALUE_SPAN = re.compile(r"\d+(?:\.\d+)?\s*%\s*(?:로|으로)")


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _append_match(values: list[str], question: str, pattern: str) -> bool:
    match = re.search(pattern, question)
    if match is None:
        return False
    _append_once(values, match.group(0))
    return True


def _question_holding_fields(question: str) -> list[str]:
    """Return only reviewed holding slots explicitly demanded by the question.

    The returned values are literal question surfaces.  Every mapping is
    activated by a reviewed Korean demand pattern and works independently of
    a company, date, receipt or fixture ID.
    """

    fields: list[str] = []
    if _append_match(
            fields, question,
            r"특별\s*관계자(?:별)?\s*보유\s*"
            r"(?:내역|비율(?:들)?|주식\s*수(?:와\s*비율)?)"):
        pass
    else:
        if re.search(
                r"보고자(?:는|가|를|의\s*성명)?\s*(?:누구|어디|알려|확인)|"
                r"(?:누가|어느\s*(?:회사|기관))\s*보고|"
                r"보고자\s*(?:와|과|및|,)",
                question):
            # Use the shortest literal field surface.  The surrounding clause
            # proves answer demand but is not part of the slot vocabulary.
            _append_once(fields, "보고자")
        delta_count = bool(re.search(
            r"변동\s*주식\s*수|증감\s*주식\s*수|"
            r"주식\s*수[^,.?]{0,24}(?:변했|변동|증감)|"
            r"몇\s*주\s*변동",
            question,
        ))
        if delta_count:
            _append_match(
                fields, question,
                r"변동\s*주식\s*수|증감\s*주식\s*수|"
                r"몇\s*주\s*변동|얼마나\s*(?:변했어|변했나요|변동)")
        else:
            _append_match(
                fields, question,
                r"보유\s*주식\s*수|보유주식수|"
                r"주식을\s*몇\s*주\s*보유|몇\s*주\s*보유")
        ratio_from = _RATIO_FROM_VALUE_SPAN.search(question)
        ratio_to = _RATIO_TO_VALUE_SPAN.search(question)
        # 주식 수에는 「증감이냐 값이냐」 갈래가 있는데 비율에는 없었다.  그래서
        # 「몇 %p 변했는지」로 물어도 늘 평범한 `보유비율` 로 되돌아갔고, 공시가
        # `MDF_STK_RT` 로 이미 적어 둔 증감 대신 이번 값을 답했다(`RPC-010`).
        # 갈림은 `%p`·퍼센트포인트·명시적 증감어일 때만이다 — 「보유비율과 변동
        # 사유」처럼 다른 항목이 뒤따르는 문장을 증감 요구로 읽으면 안 된다.
        delta_ratio = bool(re.search(
            r"%\s*p|퍼센트\s*포인트|보유\s*비율\s*증감|지분율\s*증감",
            question))
        if _RATIO_CHANGE_CUE.search(question) and ratio_from and ratio_to:
            # 이슈 #62 — 직전·이번 두 시점을 함께 요구하는 신호가 있으면, 그
            # 두 낱말이 「보유비율」에 바로 붙어 있지 않아도(RPC-011처럼
            # 「직전」·「이번」이 아예 없어도) 값+조사 구간을 그대로 두 슬롯
            # (직전·이번)으로 요청한다. 홑겹 「보유비율」 캡처만으로는 이번
            # 값 하나만 나와(#40 계열의 원인) %p·상대 증감률을 둘 다 구할 수
            # 없었다.
            _append_once(fields, ratio_from.group(0))
            _append_once(fields, ratio_to.group(0))
        elif delta_ratio:
            _append_match(
                fields, question,
                r"몇\s*%\s*p\s*변했는지|몇\s*%\s*p\s*변했나요|"
                r"몇\s*%\s*p\s*변했어|몇\s*%\s*p\s*변동|"
                r"%\s*p\s*변화|%\s*p\s*변동|%\s*p\s*증감|"
                r"퍼센트\s*포인트\s*변화|"
                r"보유\s*비율\s*증감|지분율\s*증감")
        else:
            _append_match(
                fields, question,
                r"보유\s*비율|보유비율|지분\s*율|지분율")
        _append_match(fields, question, r"변동\s*방법|변경\s*방법")
        _append_match(fields, question, r"변동\s*사유|변경\s*사유")
        # 단수 「보고자의」 속성은 IFR_* 머리말이고, 「특별관계자별」 속성은
        # 반복 SPC_* 행이다. 수식어를 포함한 리터럴 표면을 먼저 보존해야
        # 실행 단계가 둘을 한 목록으로 합치지 않는다. 「보고자의 직업과
        # 국적」처럼 수식어가 첫 병렬항에만 붙으면 `_canonical_slots`가 같은
        # 병렬 필드 목록 안의 맨몸 속성도 보고자 슬롯으로 이어 준다.
        if not _append_match(
                fields, question,
                r"보고자(?:의)?\s*직업(?:\s*\(?사업\s*내용\)?)?"):
            if not _append_match(
                    fields, question,
                    r"특별\s*관계자(?:별|의)?\s*"
                    r"직업(?:\s*\(?사업\s*내용\)?)?"):
                _append_match(
                    fields, question, r"직업(?:\s*\(?사업\s*내용\)?)?")
        if not _append_match(fields, question, r"보고자(?:의)?\s*국적"):
            if not _append_match(
                    fields, question,
                    r"특별\s*관계자(?:별|의)?\s*국적"):
                _append_match(fields, question, r"국적")
        _append_match(
            fields, question,
            r"(?:보고서\s*)?작성\s*기준일|보고\s*기준일")
        # A literal receipt supplied by the user is a selector, not an
        # additional requested output (e.g. its report base date).
        receipt_output_question = re.sub(
            r"(?:접수\s*번호|공시\s*번호)\s*20[0-9]{12}(?:인|의)?",
            "", question)
        _append_match(fields, receipt_output_question, r"접수\s*번호|공시\s*번호")
        _append_match(
            fields, question,
            r"공개(?:된)?\s*(?:법인\s*[·ㆍ/]\s*기관|법인|회사|기관)"
            r"(?:명|명칭)")

    # Restricted fields stay in the same intent so the public subset can be
    # answered while the compiler binds an explicit omission limitation.
    privacy = (
        (r"개인\s*식별\s*번호|주민등록번호|외국인등록번호|여권번호", "개인 식별번호"),
        (r"사업자등록번호|법인등록번호|(?<!접수)등록번호", "등록번호"),
        (r"개인\s*(?:성명|이름)", "개인 성명"),
        (r"생년월일|출생(?:일|연월|년도)?", "생년월일"),
        (r"(?:집|자택|개인)\s*주소|거주지", "집 주소"),
        (r"전화번호|휴대전화|연락처", "전화번호"),
    )
    for pattern, _field in privacy:
        _append_match(fields, question, pattern)
    # Preserve repeated numeric fields with their literal owners.  A report
    # aggregate and a named party can request the same metric independently.
    owned_counts = list(re.finditer(
        r"(?:보고자(?:의)?\s*총|[0-9A-Za-z가-힣㈜&().-]{2,40}의)\s*보유\s*주식\s*수",
        question))
    if len(owned_counts) > 1:
        fields = [f for f in fields if _key(f) != "보유주식수"]
        fields.extend(m.group(0) for m in owned_counts)
    # Parallel attributes inherit the owner only within their own clause.
    # Keep source substrings, not synthetic field names, for grounding.
    filer_clause = re.search(
        r"보고자(?:\s*자체)?(?:의)?\s*명칭[·ㆍ,\s]*(?:국적[·ㆍ,\s]*)?직업",
        question)
    related_clause = re.search(r"특별\s*관계자별\s*명칭[·ㆍ,\s]*국적", question)
    if filer_clause and related_clause:
        fields = [f for f in fields if _key(f) not in {"국적", "직업", "보고자"}]
        fields.append("보고자")
        clause = filer_clause.group(0)
        if "국적" in clause:
            fields.append(clause[:clause.index("국적") + 2])
        fields.extend((clause, related_clause.group(0)))
    return fields


def _question_holding_scope(question: str) -> tuple[list[str], str | None]:
    matches = list(_QUESTION_DATE.finditer(question))
    if len(matches) != 1:
        return [], None
    surface = matches[0].group(0)
    tail = question[matches[0].end():matches[0].end() + 4]
    if "까지" in tail:
        return [], surface
    return [surface], None


def _question_holding_target(question: str) -> str | None:
    match = re.search(
        r"대량\s*보유(?:상황)?\s*(?:보고서|공시)|"
        r"보유\s*주식\s*수|보유비율|지분율|"
        r"특별\s*관계자(?:별)?\s*보유\s*내역|"
        r"변동\s*주식\s*수|몇\s*주\s*변동|"
        r"주식을\s*몇\s*주\s*보유|"
        r"변동\s*(?:방법|사유)|"
        r"보유\s*주식\s*수[^,.?]{0,24}얼마나\s*변",
        question,
    )
    return match.group(0) if match else None


class QuestionGroundedHoldingIntentRegrounder:
    """Restore one exact holding request from provider topology variation.

    HCX can express the same request as metric, attribute, entity, document or
    narrative items and can bind the filer rather than the issuer.  Repair is
    allowed only when the question itself names exactly one canonical issuer
    backed by holding documents and contains reviewed holding-field wording.
    No company, receipt, value or hidden fixture coordinate is inferred.
    """

    _HOLDING_CUE = re.compile(
        r"대량\s*보유|보유\s*(?:주식|비율|목적)|지분\s*율|"
        r"특별\s*관계자|주식을\s*몇\s*주\s*보유|"
        r"몇\s*주\s*변동|변동\s*(?:주식\s*수|방법|사유)",
        re.I,
    )

    def __init__(self, canonical: Any, *, corpus_cutoff: str,
                 company_preflight: Any) -> None:
        if not callable(getattr(canonical, "companies_in_text", None)):
            raise TypeError("holding regrounder에는 question company scan이 필요합니다")
        if not callable(getattr(canonical, "resolve_company", None)):
            raise TypeError("holding regrounder에는 company resolver가 필요합니다")
        if not callable(getattr(canonical, "documents", None)):
            raise TypeError("holding regrounder에는 documents()가 필요합니다")
        if not callable(getattr(
                company_preflight, "question_company_surface", None)):
            raise TypeError("holding regrounder company preflight 계약이 잘못되었습니다")
        self.canonical = canonical
        self.corpus_cutoff = corpus_cutoff
        self.company_preflight = company_preflight

    def _issuer(self, question: str) -> tuple[Any, str] | None:
        candidates: dict[str, tuple[Any, str]] = {}
        for company in self.canonical.companies_in_text(question) or ():
            if any(True for _ in self.canonical.documents(
                    as_of=self.corpus_cutoff,
                    corp_code=company.corp_code,
                    doc_group="holding")):
                surface = self.company_preflight.question_company_surface(
                    company.corp_name, question)
                if (not isinstance(surface, str) or not surface.strip()) \
                        and company.corp_name in question:
                    surface = company.corp_name
                if isinstance(surface, str) and surface.strip():
                    candidates[company.corp_code] = (company, surface.strip())
        if not candidates:
            return None

        # When both issuer and filer/party are listed companies, the report
        # owner is the company whose literal mention introduces the holding
        # report.  A later ``X의 보유주식수`` is the party, not a second issuer.
        report_owners = [
            row for row in candidates.values()
            if re.search(
                re.escape(row[1])
                + r"\s*(?:의|에서)?\s*[^,.?]{0,35}"
                  r"(?:대량\s*보유(?:상황)?\s*(?:보고서|공시)|"
                  r"정정\s*공시)",
                question,
            )
        ]
        if len(report_owners) == 1:
            return report_owners[0]

        # In compact direct-holding wording (``A의 B 보유비율`` or
        # ``A가 B 주식을``), B is the issuer.  Require an adjacent asset/metric
        # cue so a possessive party mention is never promoted to issuer.
        asset_owners = [
            row for row in candidates.values()
            if re.search(
                re.escape(row[1])
                + r"\s*(?:주식|지분|보유\s*(?:주식|비율))",
                question,
            )
        ]
        if len(asset_owners) == 1:
            return asset_owners[0]
        if len(candidates) == 1:
            return next(iter(candidates.values()))
        return None

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str) or not self._HOLDING_CUE.search(question)
                or not intent.answer_items or intent.answer_groups
                or intent.premises or intent.unresolved_mentions):
            return intent
        fields = _question_holding_fields(question)
        target_surface = _question_holding_target(question)
        if (not fields or target_surface is None
                or _canonical_slots(fields) is None):
            return intent
        issuer = self._issuer(question)
        if issuer is None:
            return intent
        company, issuer_surface = issuer

        # Preserve literal non-issuer entities as possible filer/party
        # surfaces. Their role is proved later from the selected filing; they
        # are never used to choose an issuer or a value here.
        entities = [{
            "entity_id": "entity-1",
            "kind_hint": "company",
            "surface": issuer_surface,
        }]
        seen = {_party_key(issuer_surface)}
        for entity in intent.entities:
            surface = entity.surface.strip()
            if (not surface or _key(surface) not in _key(question)
                    or _party_key(surface) in seen):
                continue
            resolved = list(self.canonical.resolve_company(surface) or ())
            if any(row.corp_code == company.corp_code for row in resolved):
                continue
            seen.add(_party_key(surface))
            entities.append({
                "entity_id": f"entity-{len(entities) + 1}",
                "kind_hint": entity.kind_hint,
                "surface": surface,
            })

        periods, as_of = _question_holding_scope(question)
        shape = (
            "record_list" if _canonical_slots(fields)[0][2] == "parties"
            else "record" if len(fields) > 1 else "scalar"
        )
        merged = {
            "item_id": "item-1",
            "target": {
                "kind": "topic", "surface": target_surface,
                "entity_refs": ["entity-1"], "qualifier_surfaces": [],
            },
            "operation": "retrieve",
            "scope": {
                "target_period_expressions": periods,
                "as_of_expression": as_of,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": shape, "projection_mode": "named_fields",
                "field_surfaces": fields,
                "presentation": intent.presentation,
            },
        }
        payload = intent.model_dump(mode="python", warnings=False)
        payload.update({
            "entities": entities,
            "answer_items": [merged],
            "answer_groups": [],
            "premises": [],
            "unresolved_mentions": [],
        })
        return SemanticIntent.model_validate(payload, strict=True)


def _key(value: object) -> str:
    return re.sub(
        r"[^0-9A-Za-z가-힣]", "",
        unicodedata.normalize("NFKC", str(value or "")),
    ).casefold()


def _party_key(value: object) -> str:
    compact = _key(value)
    for token in ("주식회사", "유한회사", "사단법인", "재단법인"):
        compact = compact.replace(token, "")
    return compact


_LOOKUP_CACHE_LIMIT = 512


def _bounded_cache(owner: Any, attribute: str, *,
                   limit: int = _LOOKUP_CACHE_LIMIT) -> dict:
    """인스턴스에 매단 접수번호별 메모. 상한을 넘으면 통째로 비운다.

    정본 한 빌드 안에서 답이 바뀌지 않는 조회(문서의 ``보고자 구분``, 판별표의
    이름별 ``구분``)만 넣는다. 한 요청 안에서 같은 접수번호를 여러 번 읽는
    것(후보 보고자 라벨·검증, HCX 재표본)을 막는 것이 목적이라 크기는 작다.
    """

    cache = owner.__dict__.get(attribute)
    if cache is None:
        cache = {}
        owner.__dict__[attribute] = cache
    elif len(cache) >= limit:
        cache.clear()
    return cache


def _party_type_pairing(rows: list[Any]):
    """FieldRow 목록에서 문서의 ``구분`` 지도를 만든다.

    정본 적재(`src/canonical/build.py`)·읽기(`src/canonical/read.py`)와
    **같은 함수**를 쓴다. Stage1 이 따로 판정하면 정본이 공개로 둔 법인을
    역질문 선택지에서 가리는 어긋남이 생긴다 (이슈 #199).
    """

    return holding_party_types_by_row(
        (str(getattr(row, "path", "") or ""),
         getattr(row, "aunit", None),
         (getattr(row, "value_raw", None)
          if getattr(row, "value_raw", None) else getattr(row, "value", None)),
         getattr(row, "table_locator", None),
         getattr(row, "logical_row", None))
        for row in rows)


def _row_party_type(pairing, row: Any) -> str | None:
    """한 성명 행의 ``구분``. 같은 행이 먼저, 없으면 문서 안의 이름 지도."""

    raw = str(getattr(row, "value_raw", None)
              or getattr(row, "value", None) or "").strip()
    return resolve_party_type(
        pairing, str(getattr(row, "path", "") or ""), raw,
        getattr(row, "table_locator", None), getattr(row, "logical_row", None))


def _canonical_slots(
        surfaces: list[str],
        ) -> list[tuple[int, str, str, str, tuple[str, ...]]] | None:
    """Bind every source field without discarding restricted user intent.

    A mixed field such as ``보유비율과 연락처`` is one qualified field: its
    public value is executable and the private portion is explicitly limited.
    A private-only field remains limited and creates no Stage2 value root.
    Two public holding slots in one source field are not expanded because that
    would invent extra source fields; HCX must split those explicitly.
    """

    bindings: list[tuple[int, str, str, str, tuple[str, ...]]] = []
    seen: set[str | tuple[str, str]] = set()
    # 한국어 병렬구조에서는 소유격이 첫 항에만 붙을 수 있다:
    # 「보고자의 직업과 국적」. 두 field surface는 질문의 리터럴 부분문자열이어야
    # 하므로 둘째 항을 인위적으로 「보고자의 국적」으로 바꿀 수 없다. 같은 필드
    # 목록에 명시적인 보고자 속성이 있을 때에만 맨몸 직업/국적을 승격한다.
    filer_attribute_context = any(
        _key(surface) in {
            *(_key(value) for value in _SLOT_ALIASES["filer_nationality"]),
            *(_key(value) for value in _SLOT_ALIASES["filer_occupation"]),
        }
        for surface in surfaces)
    for source_index, surface in enumerate(surfaces):
        compact = _key(surface)
        owned_slot = (
            "current_count" if re.fullmatch(
                r"(?:보고자(?:의)?총|[0-9a-z가-힣]+의)보유주식수", compact)
            else "filer_nationality" if compact.startswith("보고자")
            and "명칭" in compact and compact.endswith("국적")
            else "filer_occupation" if compact.startswith("보고자")
            and "명칭" in compact and compact.endswith("직업")
            else "nationality" if compact.startswith("특별관계자별명칭")
            and compact.endswith("국적") else None)
        expanded = _SUMMARY_SLOTS.get(compact)
        if expanded is not None:
            # One source field cannot expand to several public answer fields
            # without changing the frozen field inventory.  Ask the model to
            # emit explicit fields instead of inventing compiler-only fields.
            return None
        # 이슈 #62 — 「20.75%에서」/「20.74%로」처럼 값 자체에 조사가 붙은
        # 구간은 고정 별칭 사전에 없다(값이 질문마다 다르다). 조사만으로
        # 직전/이번 시점을 가리키므로 별도 슬롯으로 직접 매핑한다.
        stripped = surface.strip()
        transition_slot = (
            "previous_ratio" if _RATIO_FROM_VALUE_SPAN.fullmatch(stripped)
            else "current_ratio" if _RATIO_TO_VALUE_SPAN.fullmatch(stripped)
            else None
        )
        if transition_slot is not None:
            if transition_slot in seen:
                return None
            seen.add(transition_slot)
            bindings.append((source_index, surface, transition_slot,
                             "executable", ()))
            continue
        promoted_filer_slot = (
            "filer_nationality"
            if filer_attribute_context and compact == _key("국적")
            else "filer_occupation"
            if filer_attribute_context and compact in {
                _key("직업"), _key("직업(사업내용)")}
            else None)
        preferred_slot = owned_slot or promoted_filer_slot
        exact = ([preferred_slot] if preferred_slot else [
            name for name, aliases in _SLOT_ALIASES.items()
            if compact == _key(name)
            or any(compact == _key(alias) for alias in aliases)])
        privacy = classify_privacy_request(surface)
        if not exact and privacy.mode == "partial":
            contained = [name for name, aliases in _SLOT_ALIASES.items()
                         if any(len(_key(alias)) >= 3 and _key(alias) in compact
                                for alias in aliases)]
            exact = list(dict.fromkeys(contained))
        if not exact and privacy.restricted_types:
            bindings.append((
                source_index, surface, "restricted_personal_data", "limited",
                privacy.restricted_types))
            continue
        if len(exact) != 1:
            return None
        slot = exact[0]
        unique_slot = (slot, compact) if owned_slot == "current_count" else slot
        if unique_slot in seen:
            return None
        seen.add(unique_slot)
        status = "qualified" if privacy.restricted_types else "executable"
        bindings.append((source_index, surface, slot, status,
                         privacy.restricted_types))
    return bindings


def _value_kind(slot: str) -> str:
    if slot in {"previous_ratio", "current_ratio", "delta_ratio"}:
        return "percent"
    if slot in {"previous_count", "current_count", "delta_count"}:
        return "count"
    if slot in {"receipt_date", "base_date"}:
        return "date"
    return "text"


def _public_filer_attribute_slots(
        slots: list[tuple[int, str, str, str, tuple[str, ...]]],
        filer_type: str | None,
        ) -> list[tuple[int, str, str, str, tuple[str, ...]]]:
    """Remove privacy qualification only for a canonically proved legal filer."""

    if filer_type is None or "개인" in filer_type:
        return slots
    return [
        (index, surface, slot,
         "executable" if slot in {
             "filer_nationality", "filer_occupation"} else status,
         () if slot in {
             "filer_nationality", "filer_occupation"} else restricted)
        for index, surface, slot, status, restricted in slots
    ]


class HoldingDisclosureResolutionBackend:
    """Resolve one issuer/reporter to one correction-effective holding filing."""

    def __init__(
            self, canonical: Any, *, canonical_build_id: str,
            resolver_version: str, reference_date: date, corpus_cutoff: str,
            ) -> None:
        required = ("resolve_company", "documents", "resolve_document_version")
        if any(not callable(getattr(canonical, name, None)) for name in required):
            raise TypeError("holding backend canonical read 계약이 잘못되었습니다")
        self.canonical = canonical
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff

    @staticmethod
    def _applies(question: str, item: Any) -> bool:
        surfaces = " ".join([
            question, item.target.surface,
            item.scope.document_group_expression or "",
            *item.target.qualifier_surfaces,
            *item.output.field_surfaces,
        ])
        return re.search(
            r"대량\s*보유|보유\s*(?:주식|비율|목적)|지분\s*율|"
            r"특별\s*관계자|주식을\s*몇\s*주\s*보유|"
            r"몇\s*주\s*변동|변동\s*(?:주식\s*수|방법|사유)|holding",
            surfaces, re.I) is not None

    def _date_scope(self, item: Any) -> tuple[str, str | None, str | None] | None:
        """Resolve filing-date scope separately from the answer data period."""

        as_of_surface = item.scope.as_of_expression
        periods = list(item.scope.target_period_expressions)
        if as_of_surface is not None and periods:
            return None
        if as_of_surface is not None:
            start, end, error = _target_date_range(
                as_of_surface, reference_date=self.reference_date)
            if error is not None or start is None or end is None:
                return None
            cutoff = end
            return (cutoff, None, None) if cutoff <= self.corpus_cutoff else None
        if periods:
            if len(periods) != 1:
                return None
            start, end, error = _target_date_range(
                periods[0], reference_date=self.reference_date)
            if error is not None or start is None or end is None:
                return None
            lower, upper = start, end
            upper = min(upper, self.corpus_cutoff)
            return (upper, lower, upper) if lower <= upper else None
        return self.corpus_cutoff, None, None

    def _issuer_candidates(self, source_intent: SemanticIntent, item: Any, as_of: str):
        by_id = {row.entity_id: row for row in source_intent.entities}
        referenced = [by_id[ref] for ref in item.target.entity_refs if ref in by_id]
        candidates: dict[str, Any] = {}

        def collect(entities: list[Any]) -> dict[str, Any]:
            found: dict[str, Any] = {}
            for entity in entities:
                if entity.kind_hint != "company":
                    continue
                for company in list(self.canonical.resolve_company(
                        entity.surface) or ()):
                    if any(True for _ in self.canonical.documents(
                            as_of=as_of, corp_code=company.corp_code,
                            doc_group="holding")):
                        found[company.corp_code] = company
            return found

        referenced_candidates = collect(referenced)
        if len(referenced_candidates) == 1:
            return list(referenced_candidates.values())
        if len(referenced_candidates) > 1:
            # Two target-linked listed companies are not an issuer/filer proof.
            # Let clarification or another backend close their semantic roles.
            return []
        for entity in source_intent.entities:
            if entity in referenced:
                continue
            if entity.kind_hint != "company":
                continue
            resolved = list(self.canonical.resolve_company(entity.surface) or ())
            for company in resolved:
                docs = list(self.canonical.documents(
                    as_of=as_of, corp_code=company.corp_code,
                    doc_group="holding"))
                if docs:
                    candidates[company.corp_code] = company
        return list(candidates.values())

    @staticmethod
    def _report_type(question: str) -> str | None:
        compact = _key(question)
        general = "일반" in compact
        short = "약식" in compact
        if general == short:
            return None
        return "general" if general else "short"

    @staticmethod
    def _filer_mentions(question: str, docs: list[Any]) -> list[str]:
        question_key = _party_key(question)
        filers = sorted({str(row.filer).strip() for row in docs if str(row.filer).strip()})
        return [filer for filer in filers
                if len(_party_key(filer)) >= 2 and _party_key(filer) in question_key]

    def _effective(
            self, docs: list[Any], as_of: str,
            ) -> tuple[list[Any], list[Any], set[str]]:
        by_receipt = {row.rcept_no: row for row in docs}
        selected: dict[str, Any] = {}
        invalid: dict[str, Any] = {}
        root_missing: set[str] = set()
        for row in docs:
            relation_summaries = getattr(self.canonical, "relation_summaries", None)
            if callable(relation_summaries) and row.is_correction:
                summaries = relation_summaries(
                    source_rcept_no=row.rcept_no, as_of=as_of)
                if any(summary.resolution_status in {"ambiguous", "invalid"}
                       for summary in summaries):
                    invalid[row.rcept_no] = row
                    continue
                if any(summary.resolution_status == "root_missing"
                       for summary in summaries):
                    root_missing.add(row.rcept_no)
            lineage = self.canonical.resolve_document_version(
                row.rcept_no, as_of=as_of)
            if getattr(lineage, "status", None) != "ok" or not lineage.selected:
                if getattr(lineage, "status", None) in {"ambiguous", "invalid"}:
                    invalid[row.rcept_no] = row
                continue
            meta = by_receipt.get(lineage.selected)
            if meta is None:
                meta = next((candidate for candidate in self.canonical.documents(
                    as_of=as_of, corp_code=row.corp_code, doc_group="holding")
                    if candidate.rcept_no == lineage.selected), None)
            if meta is not None:
                selected[meta.rcept_no] = meta
        return (
            sorted(selected.values(), key=lambda row: (
                row.rcept_dt, row.rcept_no)),
            sorted(invalid.values(), key=lambda row: (
                row.rcept_dt, row.rcept_no)),
            root_missing,
        )

    def _party_inventory(self, selected: Any) -> list[tuple[str, str]]:
        """Return source-proven holder rows and their Evidence IDs.

        접수번호별 메모. 보고자를 지정하지 않은 질문은 발행사의 지분공시 문서
        전부(고려아연 50·에코프로비엠 79건)에 대해 이 목록을 만들어 「직접
        보유 주체가 언급됐는가」를 살피고, 서버의 HCX 재표본이 같은 루프를
        되풀이한다 — 문서당 정본 읽기가 평가 서버(2vCPU)에서 1초를 넘겨
        HM-019·HM-020 이 45초 예산을 넘겼다(py-spy 실측). 목록은 정본 빌드
        안에서 불변이라 인스턴스 수명 동안 들고 있는다.
        """

        cache = _bounded_cache(self, "_party_inventory_cache", limit=4096)
        cache_key = str(getattr(selected, "rcept_no", "") or "")
        if cache_key in cache:
            return list(cache[cache_key])
        inventory = self._party_inventory_uncached(selected)
        if cache_key:
            cache[cache_key] = tuple(inventory)
        return inventory

    def _party_inventory_uncached(
            self, selected: Any) -> list[tuple[str, str]]:
        fields = getattr(self.canonical, "fields", None)
        if not callable(fields):
            return []
        try:
            filing_rows = list(fields(
                as_of=selected.rcept_dt, corp_code=selected.corp_code,
                rcept_no=selected.rcept_no, doc_group="holding",
                include_restricted_raw=True))
        except Exception:
            return []
        return self._party_inventory_from_rows(selected, filing_rows)

    def _prime_party_inventories(self, docs: list[Any]) -> None:
        """Populate per-receipt inventories with one bounded canonical scan.

        A question without a filer can inspect dozens of holding filings.  The
        old loop decoded ``fields.parquet`` once per receipt.  Exact receipt
        membership is already known here, so read all missing receipts at once
        and keep the existing per-receipt cache contract.  Any adapter that
        does not support ``rcept_nos`` simply falls back to the old path.
        """

        if len(docs) < 2:
            return
        fields = getattr(self.canonical, "fields", None)
        if not callable(fields):
            return
        cache = _bounded_cache(self, "_party_inventory_cache", limit=4096)
        missing = [
            row for row in docs
            if str(getattr(row, "rcept_no", "") or "") not in cache
        ]
        receipts = [
            str(getattr(row, "rcept_no", "") or "") for row in missing
            if str(getattr(row, "rcept_no", "") or "")
        ]
        if len(receipts) < 2 or len(receipts) > 128:
            return
        corp_codes = {
            str(getattr(row, "corp_code", "") or "") for row in missing}
        if len(corp_codes) != 1 or "" in corp_codes:
            return
        as_of = max(str(row.rcept_dt) for row in missing)
        prime_contexts = getattr(
            self.canonical, "prime_holding_field_contexts", None)
        if callable(prime_contexts):
            try:
                prime_contexts(as_of=as_of, rcept_nos=receipts)
            except Exception:
                # Compatibility and corruption handling remain fail-closed in
                # the ordinary ``fields`` call below.  Priming is only a
                # performance hint and never changes the returned projection.
                pass
        try:
            filing_rows = list(fields(
                as_of=as_of,
                corp_code=corp_codes.pop(), rcept_nos=receipts,
                doc_group="holding", include_restricted_raw=True))
        except Exception:
            return
        grouped: dict[str, list[Any]] = {}
        for field in filing_rows:
            receipt = str(getattr(field, "rcept_no", "") or "")
            if receipt in receipts:
                grouped.setdefault(receipt, []).append(field)
        by_receipt = {str(row.rcept_no): row for row in missing}
        # Do not cache an empty result for a receipt omitted by an older
        # adapter.  Its ordinary single-receipt path remains the fallback.
        for receipt, rows in grouped.items():
            cache[receipt] = tuple(self._party_inventory_from_rows(
                by_receipt[receipt], rows))

    @staticmethod
    def _party_inventory_from_rows(
            selected: Any, filing_rows: list[Any],
            ) -> list[tuple[str, str]]:
        rows = [row for row in filing_rows
                if str(getattr(row, "acode", "")) == "SPC_NM"]
        # 공시가 스스로 적어 둔 `구분`(CRP_TP/SPC_TP). 정본 적재가 쓴 것과
        # **같은 함수**로 만들어야 두 층의 판정이 갈라지지 않는다 (이슈 #199).
        pairing = _party_type_pairing(filing_rows)
        numeric_rows = [
            row for row in filing_rows
            if str(getattr(row, "acode", "")) in {"STK_CNT", "STK_RT"}
            and str(getattr(row, "value", "") or "").strip()
            not in {"", "-"}
            and "증감주식등의내역" not in _key(
                getattr(row, "path", ""))
        ]
        # The filer may also have its own direct-holding row.  Keep that row:
        # ``X's holding ratio`` asks for X's STK_RT, while ``total holding
        # ratio`` asks for SUM_TMT_RT.  Excluding the filer here silently
        # changed a direct-party question into a report aggregate.
        excluded = {_party_key(selected.corp_name), ""}
        numeric_coordinates: dict[tuple[str, int], set[str]] = {}
        for row in numeric_rows:
            coordinate = (
                str(getattr(row, "table_locator", "")),
                int(getattr(row, "logical_row", -1)),
            )
            numeric_coordinates.setdefault(coordinate, set()).add(
                str(getattr(row, "acode", "")))
        inventory: dict[str, tuple[int, str, str]] = {}
        for row in rows:
            value = str(getattr(row, "value", None) or "").strip()
            key = _party_key(value)
            if key.startswith("redacted"):
                # A legacy security-policy row (``pii-prompt-safe/1.1``) can
                # mask an institutional 특별관계자/보고자 name as a person
                # because that frozen policy's corporate-suffix boundary
                # predates suffixes such as ``기금``/``펀드`` (RPC-017 —
                # 국민연금기금 was masked to ``[REDACTED:PERSON_NAME]`` this
                # way). ``out/canonical`` is not rebuilt, so recover the
                # public organization name from ``value_raw`` only when the
                # *current* corporate-suffix boundary independently proves
                # it is an organization; an actually-masked natural person
                # stays hidden (the fallback below still drops them).
                raw_value = str(getattr(row, "value_raw", None) or "").strip()
                if raw_value and not party_is_private_person(
                        row, _row_party_type(pairing, row), name=raw_value):
                    value = raw_value
                    key = _party_key(raw_value)
                else:
                    continue
            if key in excluded or len(key) < 2 or not key:
                continue
            evidence_id = str(getattr(row, "evidence_id", ""))
            if evidence_id:
                coordinate = (
                    str(getattr(row, "table_locator", "")),
                    int(getattr(row, "logical_row", -1)),
                )
                score = len(numeric_coordinates.get(coordinate, set()))
                candidate = (score, value, evidence_id)
                previous = inventory.get(key)
                if previous is None or candidate > previous:
                    inventory[key] = candidate
        return [
            (inventory[key][1], inventory[key][2])
            for key in sorted(inventory)
        ]

    def _party_types_by_key(self, selected: Any) -> dict[str, str]:
        """이 보고서의 이름별 ``구분``. 역질문 선택지 라벨에만 쓴다.

        ``_filer_corp_type`` 이 보고자 하나만 분류하는 것과 달리, 판별표는
        특별관계자마다 ``구분`` 열을 둔다. 표시 이름(가려진 값)과 원문 이름
        양쪽을 키로 넣어, ``_party_inventory`` 가 어느 쪽을 실었든 찾을 수
        있게 한다.
        """

        fields = getattr(self.canonical, "fields", None)
        if not callable(fields):
            return {}
        cache = _bounded_cache(self, "_party_types_by_key_cache")
        cache_key = str(getattr(selected, "rcept_no", "") or "")
        if cache_key in cache:
            return dict(cache[cache_key])
        try:
            rows = list(fields(
                as_of=selected.rcept_dt, corp_code=selected.corp_code,
                rcept_no=selected.rcept_no, doc_group="holding",
                include_restricted_raw=True))
        except Exception:
            return {}
        pairing = _party_type_pairing(rows)
        found: dict[str, str] = {}
        for row in rows:
            if str(getattr(row, "acode", "")) != "SPC_NM":
                continue
            party_type = _row_party_type(pairing, row)
            if not party_type:
                continue
            for surface in (getattr(row, "value_raw", None),
                            getattr(row, "value", None)):
                key = _party_key(str(surface or "").strip())
                if key:
                    found.setdefault(key, party_type)
        if cache_key:
            cache[cache_key] = dict(found)
        return found

    @staticmethod
    def _mentioned_parties(
            question: str, inventory: list[tuple[str, str]],
            ) -> list[tuple[str, str]]:
        """Return distinct source-proven mentions without substring double-counting."""

        question_key = _party_key(question)
        matched = {
            _party_key(value): (value, evidence_id)
            for value, evidence_id in inventory
            if _party_key(value) in question_key
        }
        if not matched:
            return []
        occurrences = {
            key: [(match.start(), match.end()) for match in re.finditer(
                re.escape(key), question_key)]
            for key in matched
        }
        selected: list[tuple[str, str]] = []
        for key, value in matched.items():
            # ``삼성생명``이 ``삼성생명보험`` 안에서만 잡힌 경우는 하나의
            # 중첩 명칭이다. 반면 두 이름이 문장의 서로 다른 위치에 있으면
            # 길이와 무관하게 둘 다 유지해 clarification으로 보낸다.
            has_independent_span = any(
                not any(
                    other_start <= start and end <= other_end
                    and (other_start, other_end) != (start, end)
                    for other_key, other_spans in occurrences.items()
                    if other_key != key
                    for other_start, other_end in other_spans
                )
                for start, end in occurrences[key]
            )
            if has_independent_span:
                selected.append(value)
        return selected

    @staticmethod
    def _direct_party_mentions(
            question: str, inventory: list[tuple[str, str]],
            issuer_name: str,
            ) -> list[tuple[str, str]]:
        """Return only canonically proved parties used as direct subjects.

        Merely naming the filer in ``X가 Y에 대해 제출한 보고서`` selects a
        document; it does not change report-level SUM_* fields into X's row.
        Direct row binding is limited to three ordinary-language shapes that
        explicitly make X the holder of Y's shares.
        """

        compact = _party_key(question)
        issuer_key = _party_key(issuer_name)
        direct: list[tuple[str, str]] = []
        for name, evidence_id in inventory:
            party_key = _party_key(name)
            patterns = (
                rf"{re.escape(party_key)}의(?:{re.escape(issuer_key)})?"
                r"(?:공개)?(?:보유|지분|주식|변동)",
                rf"{re.escape(party_key)}(?:가|이){re.escape(issuer_key)}"
                r"(?:주식|지분)[^?]{0,20}(?:보유|가지)",
                rf"{re.escape(issuer_key)}[^?]{{0,30}}보고서[^?]{{0,15}}에서"
                rf"{re.escape(party_key)}의",
            )
            if any(re.search(pattern, compact) for pattern in patterns):
                direct.append((name, evidence_id))
        return direct

    @staticmethod
    def _party_subject_surfaces(
            question: str, source_intent: SemanticIntent, item: Any,
            ) -> list[str]:
        """Find explicit party subjects without treating generic holding text as one."""

        surfaces = [
            entity.surface for entity in source_intent.entities
            if entity.entity_id not in set(item.target.entity_refs)
            and entity.kind_hint in {"company", "counterparty"}
        ]
        # HCX usually extracts the party as an entity.  This narrow fallback
        # covers a missed ``X의 보유/지분`` surface and is used only to prevent
        # an aggregate fallback, never to prove a party value.
        surfaces.extend(match.group(1) for match in re.finditer(
            r"(?:^|[\s,])([0-9A-Za-z가-힣㈜&().-]{2,40})\s*의\s*"
            r"(?=[^,.?]{0,40}(?:보유|지분|주식|변동))",
            question,
        ))
        generic = {
            "보고자", "최근보고자", "현황", "대량보유현황", "보고서",
            "대량보유보고서", "대량보유상황보고서",
            "정정본", "정정공시", "공시",
        }
        return list(dict.fromkeys(
            surface.strip() for surface in surfaces
            if surface.strip() and _party_key(surface) not in generic
        ))

    @staticmethod
    def _clarify_party(
            item: Any, inventory: list[tuple[str, str]],
            party_types: dict[str, str] | None = None,
            ) -> ClarificationAuthority | TerminalAuthority:
        # 특별관계자 후보에도 문서별 신호가 있다 — 판별표의 행별 ``구분``
        # (``SPC_TP``)이다(이슈 #199). 예전에는 이 축에 신호가 없다고 보고
        # 값 문자열만으로 가렸는데, 그러면 ``에코프로``는 사람으로 가려지고
        # ``Scott Samuel Braun``은 법인으로 노출된다. 구분이 있으면 그것으로
        # 정하고 없을 때만 값 휴리스틱으로 내려간다 — 선택 매칭에 쓰는
        # ``value``는 어느 쪽이든 원문을 그대로 유지한다.
        types = party_types or {}
        labels: list[str] = []
        institutions: list[str] = []
        for name, _evidence_id in inventory[:20]:
            decided = party_is_person(types.get(_party_key(name)))
            if decided is False:
                labels.append(name)
                institutions.append(name)
            elif decided is True:
                # `_party_inventory`가 싣는 값은 정본이 이미 가린 표시용
                # 문자열일 수 있다(`이○용`). 다시 가리면 통째 치환으로
                # 떨어져 서로 다른 사람이 한 문자열로 뭉개진다.
                labels.append(name if MASK_CHAR in name
                              else mask_confirmed_person_name(name))
            else:
                labels.append(mask_holding_subject_name(name))
        # 서로 다른 두 자연인이 가림 후 같은 라벨이 되면(라틴 표기 두 사람이
        # 모두 통째 치환되는 경우) 선택지 계약이 요구하는 라벨 유일성이
        # 깨진다. `_clarify_filer`가 접수일 힌트를 붙이는 것과 같은 이유로
        # 후보 번호를 붙인다 — 구분은 궁극적으로 선택 번호가 한다.
        duplicated = {label for label in labels if labels.count(label) > 1}
        options = [ClarificationOption(
            value=name,
            label=(f"{label} (후보 {index})" if label in duplicated else label),
            proof_refs=[f"canonical:field:{evidence_id}"],
        ) for index, ((name, evidence_id), label) in enumerate(
            zip(inventory[:20], labels), start=1)]
        # 방어선: 완성된 선택지 각각이 실제로 ``value``를 가린 라벨을 달고
        # 있는지 재검증한다(HM-019 후속). 위 반복문과 별개로, option 객체
        # 자체를 다시 검사해 향후 리팩터링이 zip/index를 잘못 짝짓거나
        # 마스킹 호출을 빠뜨리는 회귀를 잡는다. ``구분``이 법인이라고 확정한
        # 이름만 예외로 넘겨, 값 휴리스틱의 오탐(``영풍``)을 지운다.
        unmasked = unmasked_personal_name_candidates(
            [option.label for option in options],
            corporate_names=institutions)
        if unmasked:
            raise ValueError(
                "holding 특별관계자 역질문 선택지에 가려지지 않은 자연인 "
                f"성명이 남아 있습니다: {unmasked}")
        if len(options) < 2:
            return TerminalAuthority(reasons=[{
                "code": "unsupported_semantic_target",
                "scope": "items", "item_ids": [item.item_id],
                "diagnostic_code": "holding_party_not_found",
            }])
        return ClarificationAuthority(slots=[ClarificationSlot(
            slot_id="slot-1", role_hint="entity",
            reason_code="holding_party_multiple_or_unknown",
            response_kind="select_one",
            prompt="어느 특별관계자의 보유내역을 확인할까요?",
            applies_to_item_ids=[item.item_id], mention_ids=[],
            options=options,
        )])

    def _filer_corp_type(self, doc: Any) -> str | None:
        """이 보고서의 ``보고자 구분``(``aunit=CRP_TP``) 원문 값.

        DART가 문서마다 이 필드에 「개인(국내)」·「국내법인」·「외국법인」·
        「연기금등 전문투자자」·「금융기관」·「법령상 조합」으로 명시적으로
        분류해 둔다. 접미사 없는 맨 그룹명(``영풍``·``두산``·``한화``·
        ``효성``)은 값의 글자 수만으로는 사람 이름과 구별할 수 없는데, 이
        필드는 값을 보지 않고도 정확히 갈라 준다. 필드가 없으면 값-기반
        휴리스틱(``mask_holding_subject_name``의 자체 판정)으로 폴백한다.
        """

        fields = getattr(self.canonical, "fields", None)
        if not callable(fields):
            return None
        # 접수번호별 메모. `_clarify_filer` 가 후보 보고자마다 두 번(라벨·검증)
        # 부르고 서버의 HCX 재표본이 같은 문서를 또 읽는다 — 정본 `fields()`
        # 한 번이 평가 서버(2vCPU)에서 1초를 넘겨 HM-019 가 45초 예산을
        # 넘겼다. 값은 정본 빌드 안에서 불변이라 인스턴스 수명 동안 들고 있다.
        cache = _bounded_cache(self, "_filer_corp_type_cache")
        key = str(getattr(doc, "rcept_no", "") or "")
        if key in cache:
            return cache[key]
        try:
            rows = list(fields(
                as_of=doc.rcept_dt, rcept_no=doc.rcept_no,
                doc_group="holding", aunit="CRP_TP"))
        except Exception:
            return None
        found = str(rows[0].value).strip() if rows else None
        if key:
            cache[key] = found
        return found

    def _masked_filer_label(self, filer: str, doc: Any) -> str:
        decided = party_is_person(self._filer_corp_type(doc))
        if decided is False:
            # DART 자체 분류가 법인·기관이라고 확정했다 — 값 모양 휴리스틱을
            # 건너뛴다(``영풍``처럼 접미사 없는 그룹명이 오탐 마스킹되는 것을
            # 막는다).
            return filer
        if decided is True:
            # 자연인으로 확정했다면 값 모양과 무관하게 가린다. 라틴 표기
            # 이름(``Scott Samuel Braun``)은 값 휴리스틱이 법인으로 읽어
            # 그대로 노출했다 (이슈 #199).
            return mask_confirmed_person_name(filer)
        return mask_holding_subject_name(filer)

    def _clarify_filer(self, item: Any, docs: list[Any]) -> ClarificationAuthority:
        filers = sorted({str(row.filer).strip() for row in docs if str(row.filer).strip()})[:20]
        representative: dict[str, Any] = {}
        for row in docs:
            filer = str(row.filer).strip()
            if filer and filer not in representative:
                representative[filer] = row
        labels = {
            filer: self._masked_filer_label(filer, representative[filer])
            for filer in filers
        }
        # 서로 다른 두 자연인이 가림 후 같은 문자열이 되면(예: 최○범/최○범)
        # 라벨만으로는 구분할 수 없다. 이때는 번호(선택 순서)로만 구분되게
        # 두고, 라벨 뒤에 접수일 힌트를 붙여 사람이 훑어볼 때도 실마리를
        # 남긴다. value(내부 매칭 키)는 항상 원문 filer라 선택 자체는
        # 라벨 충돌과 무관하게 정확하다.
        label_counts = Counter(labels.values())
        options = []
        for filer in filers:
            label = labels[filer]
            if label_counts[label] > 1:
                label = f"{label} ({representative[filer].rcept_dt} 접수)"
            options.append(ClarificationOption(
                value=filer, label=label,
                proof_refs=[f"canonical:holding:filer:{_party_key(filer)}"],
            ))
        # 방어선: CRP_TP로 독립적으로 확인한 기관 목록만 예외로 두고, 완성된
        # 선택지 라벨(접수일 힌트는 떼고)에 아직 가려지지 않은 자연인 성명이
        # 있으면 반려한다(HM-019 후속). ``labels``가 아니라 CRP_TP를 다시
        # 조회해 독립 신호로 검증하므로, ``_masked_filer_label`` 자체가
        # 잘못돼도(가리기를 빠뜨려도) 잡아낸다.
        confirmed_institutions = [
            filer for filer in filers
            if (corp_type := self._filer_corp_type(representative[filer])) is not None
            and "개인" not in corp_type
        ]
        core_labels = [
            re.sub(r" \(\d{8} 접수\)$", "", option.label) for option in options]
        unmasked = unmasked_personal_name_candidates(
            core_labels, corporate_names=confirmed_institutions)
        if unmasked:
            raise ValueError(
                "holding 보고자 역질문 선택지에 가려지지 않은 자연인 "
                f"성명이 남아 있습니다: {unmasked}")
        return ClarificationAuthority(slots=[ClarificationSlot(
            slot_id="slot-1", role_hint="entity",
            reason_code="holding_filer_multiple_candidates",
            response_kind="select_one",
            prompt="어느 대량보유 보고자의 공시를 확인할까요?",
            applies_to_item_ids=[item.item_id], mention_ids=[],
            options=options,
        )])

    def _receipt_base_dates(
            self, docs: list[Any],
            ) -> dict[str, tuple[str, str]]:
        """Return verified current-report base dates for receipt options."""

        fields = getattr(self.canonical, "fields", None)
        receipts = [str(row.rcept_no) for row in docs[:20]]
        if not callable(fields) or not receipts:
            return {}
        try:
            rows = fields(
                as_of=max(str(row.rcept_dt) for row in docs[:20]),
                corp_code=str(docs[0].corp_code), rcept_nos=receipts,
                doc_group="holding", include_restricted_raw=False)
            found: dict[str, dict[str, str]] = {}
            for row in rows:
                receipt = str(getattr(row, "rcept_no", "") or "")
                path = str(getattr(row, "path", "") or "")
                if receipt not in receipts or not (
                        path.startswith("이번보고서 > 보고서작성기준일")
                        or "귀중 > 보고서작성기준일" in path):
                    continue
                value = str(getattr(row, "value", "") or "")
                match = re.search(
                    r"(20[0-9]{2})년\s*([0-9]{1,2})월\s*([0-9]{1,2})일",
                    value)
                if match is None:
                    continue
                display = f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
                coordinate = str(
                    getattr(row, "source_coordinate", "")
                    or getattr(row, "evidence_id", "") or "")
                found.setdefault(receipt, {})[display] = coordinate
        except Exception:
            return {}
        return {
            receipt: (next(iter(values)), values[next(iter(values))])
            for receipt, values in found.items() if len(values) == 1
        }

    def _clarify_receipt(
            self, item: Any, docs: list[Any],
            ) -> ClarificationAuthority:
        base_dates = self._receipt_base_dates(docs)
        options: list[ClarificationOption] = []
        for row in docs[:20]:
            date_proof = base_dates.get(str(row.rcept_no))
            label_parts = [str(row.rcept_dt)]
            proofs = [f"canonical:document:{row.rcept_no}"]
            if date_proof is not None:
                base_date, coordinate = date_proof
                label_parts.append(f"작성기준일 {base_date}")
                if coordinate:
                    proofs.append(f"canonical:field:{coordinate}")
            label_parts.append(str(row.rcept_no))
            options.append(ClarificationOption(
                value=row.rcept_no, label=" · ".join(label_parts),
                proof_refs=proofs))
        return ClarificationAuthority(slots=[ClarificationSlot(
            slot_id="slot-1", role_hint="timepoint",
            reason_code="holding_receipt_multiple_candidates",
            response_kind="select_one",
            prompt="같은 날 제출된 공시가 여러 건입니다. 확인할 접수번호를 골라 주세요.",
            applies_to_item_ids=[item.item_id], mention_ids=[],
            options=options,
        )])

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> dict[str, Any] | ClarificationAuthority | TerminalAuthority | None:
        return self.resolve_with_selection(
            question_id=question_id,
            question=question,
            source_intent=source_intent,
        )

    def resolve_with_selection(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            selected_filer: str | None = None,
            selected_receipt: str | None = None,
            selected_party: str | None = None,
            ) -> dict[str, Any] | ClarificationAuthority | TerminalAuthority | None:
        """Resolve with one digest-bound clarification selection.

        ``selected_filer`` and ``selected_receipt`` are accepted only by the
        clarification resume backend after the session layer validated them
        against the offered options.  They are still checked against the
        current canonical snapshot here; a stale or foreign value fails
        closed instead of becoming planner text.
        """
        if (len(source_intent.answer_items) != 1 or source_intent.premises
                or source_intent.answer_groups or source_intent.unresolved_mentions):
            return None
        item = source_intent.answer_items[0]
        if not self._applies(question, item):
            return None
        slots = _canonical_slots(list(item.output.field_surfaces))
        if (slots is None or not slots
                or item.output.projection_mode != "named_fields"):
            return None
        date_scope = self._date_scope(item)
        if date_scope is None:
            return None
        as_of, rcept_from, rcept_to = date_scope
        issuers = self._issuer_candidates(source_intent, item, as_of)
        if len(issuers) != 1:
            return None
        issuer = issuers[0]
        docs = list(self.canonical.documents(
            as_of=as_of, corp_code=issuer.corp_code, doc_group="holding"))
        if rcept_from is not None:
            docs = [row for row in docs if rcept_from <= row.rcept_dt <= rcept_to]
            if not docs:
                # The issuer and holding-document family are both proven —
                # the corpus simply has no filing inside the requested
                # window.  This is a coverage gap, not an unrecognized
                # question shape, so it must not fall through to the generic
                # ``unsupported_semantic_target`` refusal (DEV-RPC-008):
                # that wording reads as "the question could not be
                # understood" instead of "no document exists here", and lets
                # a wording model reach for another period's document.
                return TerminalAuthority(reasons=[{
                    "code": "corpus_coverage_unavailable",
                    "scope": "question",
                    "item_ids": [item.item_id],
                }])
        if "정정" in question:
            docs = [row for row in docs if row.is_correction]
        explicit_receipts = list(dict.fromkeys(re.findall(
            r"(?<![0-9])20[0-9]{12}(?![0-9])", question)))
        if len(explicit_receipts) > 1:
            return self._clarify_receipt(item, [
                row for row in docs if row.rcept_no in explicit_receipts])
        if len(explicit_receipts) == 1:
            docs = [
                row for row in docs if row.rcept_no == explicit_receipts[0]
            ]
        report_type = self._report_type(question)
        if report_type is not None:
            marker = "일반" if report_type == "general" else "약식"
            docs = [row for row in docs if marker in str(row.report_nm)]
        if selected_receipt is not None:
            selected_receipt = selected_receipt.strip()
            docs = [row for row in docs if row.rcept_no == selected_receipt]
            if len(docs) != 1:
                return None
        available_filers = {
            str(row.filer).strip() for row in docs if str(row.filer).strip()}
        if selected_filer is not None:
            selected_filer = selected_filer.strip()
            if selected_filer not in available_filers:
                return None
            filers = [selected_filer]
        else:
            filers = self._filer_mentions(question, docs)
        if len(filers) > 1:
            return self._clarify_filer(item, [
                row for row in docs if row.filer in filers])
        if len(filers) == 1:
            docs = [row for row in docs
                    if _party_key(row.filer) == _party_key(filers[0])]
        else:
            # A direct-holding subject can identify the filing even when it is
            # not the filer.  Do not let a company merely mentioned elsewhere
            # in the sentence choose a document.
            self._prime_party_inventories(docs)
            party_matches: list[tuple[Any, tuple[tuple[str, str], ...]]] = []
            all_named_party_keys: set[str] = set()
            for row in docs:
                inventory = self._party_inventory(row)
                all_named_party_keys.update(
                    _party_key(name)
                    for name, _evidence_id in self._mentioned_parties(
                        question, inventory))
                matched = tuple(self._direct_party_mentions(
                    question, inventory, issuer.corp_name))
                if matched:
                    party_matches.append((row, matched))
            distinct_party_keys = {
                _party_key(name)
                for _row, matches in party_matches
                for name, _evidence_id in matches
            }
            if (len(all_named_party_keys) <= 1
                    and len(distinct_party_keys) == 1):
                docs = [row for row, _matches in party_matches]
        recent_selection = bool(re.search(
            r"최신|가장\s*최근|최근\s*(?:공시|보고서|보고자)|현재\s*기준",
            question))
        multiple_filers = (
            len({str(row.filer).strip() for row in docs if str(row.filer).strip()}) > 1)
        if not filers and multiple_filers and not recent_selection:
            return self._clarify_filer(item, docs)
        exact_receipt_selection = (
            selected_receipt is not None or len(explicit_receipts) == 1)
        if exact_receipt_selection:
            # A receipt explicitly chosen by the user/session is an as-filed
            # document coordinate.  Folding it forward would answer a
            # different receipt while claiming to honour the selection.
            effective, invalid_lineages, root_missing = (
                sorted(docs, key=lambda row: (row.rcept_dt, row.rcept_no)),
                [], set(),
            )
        else:
            effective, invalid_lineages, root_missing = self._effective(
                docs, as_of)
        if not effective:
            return None
        latest_day = max(row.rcept_dt for row in effective)
        # An unresolved correction can affect the requested latest result only
        # when it is at least as recent as the latest valid candidate.  An old
        # unrelated ambiguous lineage must not block newer valid disclosures.
        if any(row.rcept_dt >= latest_day for row in invalid_lineages):
            return None
        latest = [row for row in effective if row.rcept_dt == latest_day]
        if len(latest) != 1:
            return self._clarify_receipt(item, latest)
        selected = latest[0]
        filer_name = str(selected.filer).strip()
        if not filer_name:
            return None
        # ``classify_privacy_request('직업')``은 주체를 모르는 초기 표면
        # 단계에서는 보수적으로 개인정보 가능성을 표시한다. 하지만 선택된
        # 정본의 CRP_TP가 보고자를 법인으로 확정한 뒤에도 그 상태를 compiler로
        # 보내면, 공개 법인 직업의 원문 가림을 ``개인 식별정보 생략``으로
        # 잘못 설명한다(#232/#233). 법인임이 독립적으로 확인된 보고자 속성만
        # 공개 실행 필드로 승격한다. 자연인·불명 주체는 기존 제한을 유지한다.
        filer_type = self._filer_corp_type(selected)
        slots = _public_filer_attribute_slots(slots, filer_type)
        selected_type = "short" if "약식" in str(selected.report_nm) else "general"
        party_inventory = self._party_inventory(selected)
        if selected_party is not None:
            selected_party_key = _party_key(selected_party)
            mentioned_parties = [
                row for row in party_inventory
                if _party_key(row[0]) == selected_party_key
            ]
            if len(mentioned_parties) != 1:
                return None
        else:
            all_mentions = self._mentioned_parties(question, party_inventory)
            direct_mentions = self._direct_party_mentions(
                question, party_inventory, issuer.corp_name)
            # A separately requested registration number does not make that
            # entity an additional owner of the public holding metric.
            numeric_mentions = [
                row for row in all_mentions
                if row in direct_mentions or not re.search(
                    re.escape(_party_key(row[0])) + r"의(?:사업자|법인)?등록번호",
                    _party_key(question))]
            mentioned_parties = (
                numeric_mentions if len(numeric_mentions) > 1 else direct_mentions)
        field_parties = {
            index: matches[0]
            for index, surface, slot, _status, _restricted in slots
            if slot in _PARTY_VALUE_SLOTS
            and not re.search(r"총|전체|합계", surface)
            and len(matches := self._direct_party_mentions(
                surface, party_inventory, issuer.corp_name)) == 1
        }
        if re.search(r"총\s*|전체\s*|합계\s*", question):
            # Aggregate wording deliberately selects SUM_* even if the filer
            # also has a direct SPC_NM row.
            mentioned_parties = []
        party_capable = any(slot in _PARTY_VALUE_SLOTS
                            for _index, _surface, slot, status, _restricted in slots
                            if status != "limited")
        known_subjects = {
            _party_key(issuer.corp_name), _party_key(filer_name),
            *(_party_key(entity.surface) for entity in source_intent.entities
              if entity.entity_id in set(item.target.entity_refs)),
        }
        unknown_party_subject = any(
            _party_key(surface) not in known_subjects
            and all(_party_key(surface) != _party_key(name)
                    for name, _evidence_id in party_inventory)
            for surface in self._party_subject_surfaces(
                question, source_intent, item)
        )
        if party_capable and (len(mentioned_parties) != 1) and (
                len(mentioned_parties) > 1 or unknown_party_subject):
            return self._clarify_party(
                item, party_inventory, self._party_types_by_key(selected))
        party = mentioned_parties[0] if len(mentioned_parties) == 1 else None
        item_id = item.item_id
        executable = [row for row in slots if row[3] != "limited"]
        if not executable:
            return None
        field_proofs = [{
            "proof_ref": f"source-field:{item_id}:{index}",
            "source_field_index": index,
            "surface": surface,
        } for index, surface, _slot, _status, _restricted in slots]
        slot_bindings = [{
            "source_field_index": index,
            "surface": surface,
            "slot": slot,
            "acode": _SLOT_ACODES.get(slot),
            "value_kind": _value_kind(slot),
            "binding_status": status,
            "restricted_types": list(restricted),
            "party_name": (
                field_parties.get(index, party)[0]
                if field_parties.get(index, party) is not None and slot in _PARTY_VALUE_SLOTS
                else None),
            "party_proof": ({
                "source_receipt": selected.rcept_no,
                "proof_ref": f"canonical:field:{field_parties.get(index, party)[1]}",
            } if field_parties.get(index, party) is not None and slot in _PARTY_VALUE_SLOTS else None),
            "proof_ref": f"source-field:{item_id}:{index}",
        } for index, surface, slot, status, restricted in slots]
        # 초기 질문 분류는 ``직업``을 주체 미확정 개인정보로 보수 판정한다.
        # 선택된 CRP_TP가 법인 보고자임을 확정해 위에서 filer attribute를
        # executable로 승격했다면 더는 generic 개인정보 고지를 계획에 넣지
        # 않는다. 실제 개인·혼합 요청은 limited/qualified binding이 남으므로
        # 이전과 같이 privacy_notice가 유지된다(CG-049, #232/#233).
        privacy_notice_required = any(
            row[3] != "executable" for row in slots)
        resolution = AuthoritativeResolution.create(
            question_id=question_id,
            source_intent_digest=semantic_intent_digest(source_intent),
            canonical_build_id=self.canonical_build_id,
            resolver_version=self.resolver_version,
            reference_date=self.reference_date,
            corpus_cutoff=self.corpus_cutoff,
            items=[{
                "item_id": item_id,
                "target_surface": item.target.surface,
                "projection_mode": "named_fields",
                "resolution": {
                    "kind": "holding_disclosure",
                    "issuer_corp_code": issuer.corp_code,
                    "issuer_corp_name": issuer.corp_name,
                    "filer_name": filer_name,
                    "document_id": selected.doc_id,
                    "receipt_no": selected.rcept_no,
                    "report_type": selected_type,
                    "as_of": as_of,
                    "document_proof": {
                        "source_receipt": selected.rcept_no,
                        "proof_ref": f"canonical:document:{selected.rcept_no}",
                    },
                    "slot_bindings": slot_bindings,
                    "privacy_notice_required": privacy_notice_required,
                    "lineage_status": (
                        "root_missing"
                        if selected.rcept_no in root_missing else "complete"),
                    "selection_basis": (
                        "selected_receipt" if exact_receipt_selection
                        else "date_scoped" if rcept_from is not None
                        else "explicit_latest" if recent_selection
                        else "reporter_latest_default"),
                },
                "field_proofs": field_proofs,
                "applied_defaults": [],
            }],
            premise_proofs=[],
        )
        return {"kind": "resolved", "resolution": resolution.model_dump(mode="json")}


__all__ = [
    "HoldingDisclosureResolutionBackend",
    "QuestionGroundedHoldingIntentRegrounder",
]
