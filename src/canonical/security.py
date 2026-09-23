"""LLM/검색용 deterministic 보안 projection.

원문은 근거 재현과 감사 때문에 절대 덮어쓰지 않는다. 대신 이 모듈이 같은 원문에서
두 파생 표현을 만든다.

``text_search``
    PII와 실행 가능한 HTML/URI를 제거한 검색·임베딩용 표현.

``text_prompt_safe``
    ``text_search``에서 prompt-injection처럼 보이는 줄을 더 제거하고, 내용 전체를
    *신뢰할 수 없는 공시 데이터* 경계로 감싼 LLM 전달용 표현.

규칙 기반 masking은 완전한 개인정보 탐지기가 아니다. 그래서 raw를 지우는 수단이 아니라
기본 전달 경로의 방어층으로만 사용하며, 탐지 종류와 정책 버전을 산출물에 함께 남긴다.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import re
import unicodedata
from collections.abc import Iterable

__all__ = [
    "SECURITY_POLICY_VERSION",
    "PROMPT_DATA_BEGIN",
    "PROMPT_DATA_END",
    "SecurityProjection",
    "FieldProjection",
    "PrivacyRequestClassification",
    "normalize_label",
    "classify_pii_label",
    "classify_privacy_request",
    "is_organization_name",
    "project_chunk_text",
    "project_field_value",
    "subject_names_by_row",
    "needs_subject_name",
    "HoldingPartyTypes",
    "holding_party_types_by_row",
    "resolve_party_type",
    "needs_party_type",
    "party_is_person",
    "mask_confirmed_person_name",
    "security_policy_at_least",
    "PARTY_TYPE_POLICY_VERSION",
    "SUPPORTED_SECURITY_POLICY_VERSIONS",
    "SUBJECT_ACODES",
    "DOCUMENT_PARTY_TYPE_AUNIT",
    "ROW_PARTY_TYPE_AUNIT",
]


# 이 값이 달라지면 파생 문자열의 의미도 달라진다. canonical build/run 및 각 행에 기록한다.
SECURITY_POLICY_VERSION = "pii-prompt-safe/1.5"
_PREVIOUS_SECURITY_POLICY_VERSION = "pii-prompt-safe/1.4"
_LEGACY_SECURITY_POLICY_VERSION = "pii-prompt-safe/1.1"

#: 규칙이 들어온 판을 **이름으로** 붙든다. 예전에는 「1.4 이후인가」를
#: ``_policy_at_least(policy, SECURITY_POLICY_VERSION)`` 로 물었는데, 판을
#: 1.5 로 올리는 순간 그 질문이 조용히 「1.5 이후인가」로 바뀌어 1.4 로 적힌
#: 164만 행이 통째로 재검증에 실패한다(#140 에서 1.2 가 1.1 규칙으로 떨어진
#: 것과 같은 사고). 판 이름은 규칙마다 고정한다.
_HOLDING_ROLE_POLICY = "pii-prompt-safe/1.4"
_PARTY_TYPE_POLICY = "pii-prompt-safe/1.5"

#: 실행 계층이 「이 행은 `구분`으로 판정된 판인가」를 물을 때 쓰는 공개 이름.
PARTY_TYPE_POLICY_VERSION = _PARTY_TYPE_POLICY

#: 오래된 것부터. 저장된 행은 **그 행이 적힌 판의 규칙으로** 재계산해 대조하므로
#: (`src/canonical/read.py`), 「이 판 이후인가」를 순서로 물어야 한다. 예전에는
#: `== SECURITY_POLICY_VERSION` 으로 물었는데 판이 셋이 되자 1.2 가 조용히 1.1
#: 규칙으로 떨어졌다(이슈 #140 작업 중 `RPC-006`·`RPC-017` 회귀).
_POLICY_ORDER = (
    _LEGACY_SECURITY_POLICY_VERSION,
    "pii-prompt-safe/1.2",
    "pii-prompt-safe/1.3",
    _PREVIOUS_SECURITY_POLICY_VERSION,
    SECURITY_POLICY_VERSION,
)
_SUPPORTED_SECURITY_POLICY_VERSIONS = frozenset(_POLICY_ORDER)

#: 재투영이 받아 주는 판. 저장된 artifact 를 그 판의 규칙으로 다시 계산해
#: 대조하는 쪽(읽기·전수 게이트)이 「이 판을 아는가」를 물을 때 쓴다.
SUPPORTED_SECURITY_POLICY_VERSIONS = _SUPPORTED_SECURITY_POLICY_VERSIONS


def _policy_at_least(version: str, minimum: str) -> bool:
    """``version`` 이 ``minimum`` 판 이후인가."""

    try:
        return _POLICY_ORDER.index(version) >= _POLICY_ORDER.index(minimum)
    except ValueError:
        return False


def security_policy_at_least(version: str | None, minimum: str) -> bool:
    """저장된 판이 ``minimum`` 이후인가.

    실행 계층(`app/`·`agent/`)이 정본 행의 ``security_policy_version`` 을 보고
    「이 값은 이미 그 규칙으로 판정됐는가」를 물을 때 쓴다. 모르는 판은
    ``False`` 다 — 옛 정본에서 새 규칙을 가정하지 않는다.
    """

    return _policy_at_least(str(version or ""), minimum)


PROMPT_DATA_BEGIN = "<<<UNTRUSTED_DART_DISCLOSURE_DATA_BEGIN>>>"
PROMPT_DATA_END = "<<<UNTRUSTED_DART_DISCLOSURE_DATA_END>>>"
_ACTIVE_REMOVED = "[REMOVED:ACTIVE_CONTENT]"
_INSTRUCTION_REMOVED = "[REMOVED:UNTRUSTED_INSTRUCTION]"


@dataclass(frozen=True)
class SecurityProjection:
    text_search: str
    text_prompt_safe: str
    security_flags: tuple[str, ...]
    pii_types: tuple[str, ...]
    security_policy_version: str = SECURITY_POLICY_VERSION


@dataclass(frozen=True)
class FieldProjection:
    value_masked: str
    pii_type: str | None
    is_pii: bool
    security_policy_version: str = SECURITY_POLICY_VERSION


@dataclass(frozen=True)
class PrivacyRequestClassification:
    """사용자 질의가 개인정보만 요구하는지, 공개 공시값과 섞였는지의 결정적 판정.

    이 모듈은 **판정만** 제공한다. 실제 Stage1의 거절/부분답변 정책은 상위 레이어가
    이 값을 사용해 결정한다. 그래서 canonical 보안 projection과 대화 정책을 섞지 않고,
    ``지분율과 담당자 연락처를 같이 알려줘`` 같은 질의를 부분답변으로 처리할 수 있다.
    """

    restricted_types: tuple[str, ...]
    has_public_disclosure_request: bool

    @property
    def mode(self) -> str:
        if not self.restricted_types:
            return "none"
        return "partial" if self.has_public_disclosure_request else "restricted_only"


_PII_ORDER = (
    "resident_registration_no",
    "registration_no",
    "bank_account",
    "employee_id",
    "email",
    "phone",
    "birth_date",
    "address",
    "person_name",
    "occupation",
    "raw_personal_data",
)


def _ordered_types(types: Iterable[str]) -> tuple[str, ...]:
    found = set(types)
    known = [kind for kind in _PII_ORDER if kind in found]
    return tuple(known + sorted(found - set(known)))


def normalize_label(label: str) -> str:
    """공백·구두점·호환문자를 없앤 라벨 비교키.

    ``성 명``, ``성.명``, ``성명``이 모두 ``성명``이 된다. 값이나 원문 경로에는 쓰지
    않고 PII 라벨 판정에만 쓴다.
    """
    normalized = unicodedata.normalize("NFKC", str(label)).casefold()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


_CORPORATE_LABELS = (
    "기업명", "회사명", "법인명", "상호", "종목명", "발행회사", "대상회사",
    "회사명칭", "법인명칭", "기업명칭",
)


#: **공시 의무 공개 정보는 개인정보 마스킹 대상이 아니다.** 대표이사·임원·감사·최대주주 등 공개 직위자의 성명과
#: 회사 본점·사업장 소재지는 사업보고서가 공개하도록 정해진 항목이라, 마스킹하면 「대표이사가 누구인가」 같은
#: 정당한 질문에 safe 경로로 답할 수 없다. 라벨(경로)·표 머리글·본문 문맥의 직위/장소 표지로 판정한다.
#: 공시가 이름을 공개하는 직위. 이 문맥의 성명은 가리지 않는다.
_PUBLIC_ROLE_MARKERS = (
    "대표이사", "대표자", "임원", "이사", "감사", "최대주주", "주요주주", "대주주",
    "특수관계인", "경영진", "집행임원", "사외이사", "이사회", "감사위원",
)

#: 지분공시 고유 역할. 이 자리에는 **법인과 자연인이 섞여 온다** — 국민연금공단·
#: 삼성물산도, 최윤범도 같은 `성명(명칭)` 칸에 담긴다. 그래서 통째로 가리면
#: 「보고자가 누구인가」에 답할 수 없고, 통째로 공개하면 자연인 성명이 쉬는
#: 데이터에 남는다.
#:
#: 1.4 부터 **부분 가림**으로 정한다 — 법인은 그대로, 자연인은 `최○범`.
#: 1.3 이하에서는 전역 면제였다(그때는 실행 시 표시 계층에서만 가렸다).
#: 근거와 승격 계획은 `docs/stage1/지분공시_마스킹_구현.md` 에 적혀 있다.
#: ``신청인``·``청구인``·``제출인`` 처럼 다른 문서군에서도 개인을 뜻할 수 있는
#: 일반 역할은 여기 넣지 않는다.
_HOLDING_ROLE_MARKERS = (
    "보고자", "대량보유자", "특별관계자", "공동보유자",
)
_PUBLIC_PLACE_MARKERS = ("본점", "본사", "소재지", "사업장", "공장", "영업소", "지점")
_PUBLIC_TABLE_MARKERS = (
    "직위", "직책", "담당업무", "등기임원", "상근여부", "재직기간", "임기", "주요경력",
    "최대주주", "관계", "소유주식", "주식수", "지분율", "선임", "이사회",
)


def _checked_security_policy_version(version: str | None) -> str:
    """현재 및 바로 직전 projection 정책만 재생성에 허용한다.

    저장 artifact의 policy version을 무시하고 현재 규칙으로 재투영하면, 정책 개선만으로
    기존 release artifact가 손상된 것처럼 보인다. 버전이 알려지지 않은 artifact는
    안전하게 fail-closed 한다.
    """
    resolved = SECURITY_POLICY_VERSION if version is None else str(version)
    if resolved not in _SUPPORTED_SECURITY_POLICY_VERSIONS:
        raise ValueError(f"지원하지 않는 security policy version: {resolved!r}")
    return resolved


def _is_modern_policy(version: str) -> bool:
    """1.1 이후의 판인가.

    이 파일의 ``== SECURITY_POLICY_VERSION`` 비교는 대부분 「최신 판」이 아니라
    **「옛 1.1 이 아님」**을 뜻했다. 판이 둘뿐일 때는 둘이 같은 말이었지만,
    1.3 을 올리면서 갈렸다 — 그대로 두면 1.2 로 적힌 정본이 조용히 1.1 규칙으로
    재계산돼 저장된 마스킹과 어긋난다(`RPC-006`/`RPC-017` 회귀). 뜻을 이름으로
    적어 두 의미를 구분한다.
    """

    return version != _LEGACY_SECURITY_POLICY_VERSION


def _public_role_context(text: str, *, security_policy_version: str) -> bool:
    markers = (_PUBLIC_ROLE_MARKERS
               if _is_modern_policy(security_policy_version)
               else _PUBLIC_ROLE_MARKERS[:14])
    if not _policy_at_least(security_policy_version, _HOLDING_ROLE_POLICY):
        # 1.3 이하는 지분공시 역할도 전역 면제였다. 그 판으로 적힌 행을
        # 재검증하려면 그때 규칙을 그대로 써야 한다.
        markers = markers + _HOLDING_ROLE_MARKERS
    return any(marker in text for marker in markers)


#: 값 안에 지명이 나오지만 **공시가 공개하는 내용**인 필드. 계약의 판매·공급
#: 지역, 사업 개요, 양수·양도 자산명은 개인 주소가 아니다.
#:
#: 값 스캔의 주소 정규식이 이 칸들을 통째로 가리고 있었다 — 실측 207건.
#: `4. 판매ㆍ공급지역` 의 「충청남도 당진시 석문면 통정리」가 사라지고,
#: `9. 기타 투자판단과 관련한 중요사항` 에서는 지명을 물고 **뒤 문장까지**
#: 삼켜 「744세대·지상49층」 사업 개요가 통째로 없어졌다 (이슈 #140 후속).
#:
#: 개인 주소를 담는 칸(`본점소재지`·`주소`·`거주지`)은 여기 없다 — 그쪽은
#: 종전대로 라벨 분류가 맡는다.
_PUBLIC_ADDRESS_CONTENT_LABELS = (
    "공급지역", "판매지역", "투자판단", "주요내용", "자산명",
    "계약명", "행사절차", "사업개요", "공사개요",
)


def _public_address_content(text: str, *, security_policy_version: str) -> bool:
    """이 라벨의 값에 나오는 지명은 공개 내용인가 (1.4 이후)."""

    if not _policy_at_least(security_policy_version, _HOLDING_ROLE_POLICY):
        return False
    key = normalize_label(text)
    return any(marker in key for marker in _PUBLIC_ADDRESS_CONTENT_LABELS)


def _holding_role_context(text: str, *, security_policy_version: str) -> bool:
    """지분공시 보고 주체 문맥인가 (1.4 이후에만 뜻이 있다)."""

    if not _policy_at_least(security_policy_version, _HOLDING_ROLE_POLICY):
        return False
    return any(marker in text for marker in _HOLDING_ROLE_MARKERS)


def _public_place_context(text: str) -> bool:
    return any(marker in text for marker in _PUBLIC_PLACE_MARKERS)


def classify_pii_label(
        label: str, *, security_policy_version: str | None = None) -> str | None:
    """정형 라벨의 PII 유형. 기업·법인명 라벨과 공개 직위자 성명·회사 소재지 라벨은 제외한다."""
    policy = _checked_security_policy_version(security_policy_version)
    kind = _classify_pii_label_raw(
        label, security_policy_version=policy)
    if kind is None:
        return None
    full = normalize_label(label)
    if kind == "address" and _public_place_context(full):
        return None
    if kind in ("person_name", "person_name_or_entity"):
        if _public_role_context(full, security_policy_version=policy):
            return None
        if _holding_role_context(full, security_policy_version=policy):
            # 법인·자연인이 섞여 오는 자리다. 값을 보고 정한다.
            return "holding_subject_name"
    return kind


def _classify_pii_label_raw(
        label: str, *, security_policy_version: str,
        ) -> str | None:
    """직위·장소 예외를 적용하기 전의 라벨 분류."""
    # 경로의 부모 회사명이 라벨 판정을 오염시키지 않도록 마지막 segment를 우선한다.
    leaf = re.split(r"\s*>\s*", str(label))[-1]
    key = normalize_label(leaf)
    full = normalize_label(label)

    if any(word in key for word in _CORPORATE_LABELS):
        return None
    if "이메일" in key or "전자우편" in key or key == "email":
        return "email"
    if any(word in key for word in ("전화", "휴대폰", "휴대전화", "핸드폰", "팩스")):
        return "phone"
    if any(word in key for word in ("주민등록번호", "외국인등록번호", "여권번호")):
        return "resident_registration_no"
    if "사업자등록번호" in key and not any(word in key for word in ("생년", "출생")):
        return "registration_no"
    if (_is_modern_policy(security_policy_version)
            and any(word in key for word in (
                "계좌번호", "은행계좌", "카드번호"))):
        return "bank_account"
    if (_is_modern_policy(security_policy_version)
            and any(word in key for word in (
                "사원번호", "사번", "직원번호"))):
        return "employee_id"
    if any(word in key for word in ("생년", "출생일", "출생연월")):
        return "birth_date"
    if "주소" in key or "거주지" in key:
        return "address"
    if "직업" in key or "직장" in key:
        # `직 업(사업내용)` 한 라벨 아래 자연인의 직위와 법인의 업종이 섞여
        # 있다 — 실측 개인 7,117 / 법인 9,084. 라벨로도 값으로도 가를 수
        # 없으므로 **같은 행의 주체**를 보고 정한다 (이슈 #139).
        # 옛 판으로 적힌 행은 그대로 `occupation` 이어야 재검증이 맞는다.
        return ("occupation_or_industry"
                if _policy_at_least(security_policy_version, "pii-prompt-safe/1.3")
                else "occupation")

    # `성명(명칭)`은 개인/법인이 섞인 DART 라벨이다. 값의 법인 여부까지 본 뒤 결정한다.
    if "성명" in key and "명칭" in key:
        return "person_name_or_entity"
    if "성명" in key or key in ("이름", "담당자명") or "본인성명" in full:
        return "person_name"
    return None


#: 라틴 법인격 표기의 경계.  DART 의 ``성명(명칭)`` 값은 **공백이 지워진 채로**
#: 들어오는 일이 잦다(``BlackRockFundAdvisors``).  그러면 ``\b`` 가 무너져 법인
#: 표지를 놓치고, 같은 회사가 문서마다 가려지기도 노출되기도 한다(이슈 #140).
#: 공백은 사라져도 **대문자 경계는 남으므로** 낙타 경계를 함께 본다.
#: ``(?-i:...)`` 로 대소문자 구분을 국소 복원한다 — ``re.IGNORECASE`` 아래에서는
#: ``[a-z]`` 가 대문자에도 붙어 낙타 판정이 무너진다.
#: 1.2 이하로 적재된 정본을 읽을 때 쓰는 옛 표지. 저장된 마스킹은 읽을 때
#: 다시 계산해 대조하므로(`src/canonical/read.py`), 옛 판의 규칙을 그대로
#: 남겨 두지 않으면 이미 쌓인 164만 행이 전부 재검증에 실패한다.
_LEGACY_CORP_MARKER_RE = re.compile(
    r"(?:주식회사|유한회사|합자회사|합명회사|사단법인|재단법인|"
    r"\(\s*주\s*\)|㈜|\b(?:co\.?|company|corp\.?|corporation|inc\.?|"
    r"ltd\.?|limited|llc|plc|fund|trust)\b)",
    re.IGNORECASE,
)


_CORP_EDGE = r"(?:\b|(?-i:(?<=[a-z])(?=[A-Z])))"
_CORP_MARKER_RE = re.compile(
    r"(?:주식회사|유한회사|합자회사|합명회사|사단법인|재단법인|"
    r"\(\s*주\s*\)|㈜|"
    + _CORP_EDGE
    + r"(?:co\.?|company|corp\.?|corporation|inc\.?|"
      r"ltd\.?|limited|llc|plc|fund|trust|"
      # 대륙별 법인격.  네덜란드·독일·북유럽·프랑스·이탈리아 형태가 통째로
      # 빠져 있었다 — `BlackRock (Netherlands) B.V.` 가 띄어 써도 개인으로
      # 판정됐다.  점은 있을 수도 없을 수도 있다.
      # 짧은 형태(`SA` `AG` `NA`)는 점을 요구한다. 맨 두 글자를 경계만으로
      # 받으면 라틴 표기 인명의 머리글자에 붙을 수 있다. 실측상 점을 요구해도
      # 잃는 것보다 얻는 것이 많다 — `portfolio` 가 그 자리를 메운다.
      r"b\.?v\.?|n\.?v\.?|n\.a\.?|s\.a\.?|a\.g\.?|gmbh|"
      r"s\.?a\.?r\.?l|sarl|s\.p\.a\.?|oyj|asa|pte\.?|pty\.?|portfolio|"
      # 펀드·상장상품 표기.  이름 자체가 상품이라 법인격 낱말이 없다.
      r"etf|sicav|ucits|reit"
      r")"
    + _CORP_EDGE + r")",
    re.IGNORECASE,
)
_LEGACY_CORP_SUFFIXES = (
    "전자", "은행", "증권", "보험", "건설", "화학", "제약", "텔레콤", "홀딩스",
    "그룹", "캐피탈", "자산운용", "엔터테인먼트", "모터스", "에너지", "재단", "공사",
    "공단", "협회", "조합",
)
_PREVIOUS_CORP_SUFFIXES = _LEGACY_CORP_SUFFIXES + (
    "기금", "펀드", "파트너스", "인베스트먼트",
    # 법인 표기가 생략된 지분공시의 ``성명(명칭)`` 열에서 반복적으로 나오는 법인형 명칭.
    # 사람 이름 일반화를 위해 성씨/직업어가 아닌 법인 업무 접미사만 제한적으로 둔다.
    "물산", "금융", "산업", "상사",
)
_CORP_SUFFIXES = _PREVIOUS_CORP_SUFFIXES + (
    # 라틴 법인격의 한글 음차.  `모건스탠리 앤 씨오 인터내셔널 피엘씨` 처럼
    # 원문이 통째로 음차된 기관명이 개인으로 판정됐다 (이슈 #140).
    "피엘씨", "엘엘씨", "리미티드", "인코퍼레이티드", "코퍼레이션",
    "컴퍼니", "어드바이저스", "매니지먼트", "인터내셔널",
)


def _corp_key(value: str) -> str:
    key = normalize_label(value)
    for token in ("주식회사", "유한회사", "합자회사", "합명회사", "사단법인", "재단법인"):
        key = key.replace(token, "")
    # NFKC 뒤 `㈜`는 `(주)`가 되어 normalize_label 결과가 `주`다. 앞·뒤 1글자 `주`만 제거한다.
    return key.removeprefix("주").removesuffix("주")


def _is_corporate_name(
        value: str, corporate_names: Iterable[str], *,
        security_policy_version: str) -> bool:
    value = value.strip()
    if not value:
        return False
    # 넓힌 라틴 표지는 현재 판에서만 쓴다. 옛 판으로 적힌 행은 옛 규칙으로
    # 재계산해야 저장된 값과 맞는다.
    marker = (_CORP_MARKER_RE
              if _policy_at_least(security_policy_version, "pii-prompt-safe/1.3")
              else _LEGACY_CORP_MARKER_RE)
    if marker.search(value):
        return True
    compact = _corp_key(value)
    suffixes = (
        _CORP_SUFFIXES
        if _policy_at_least(security_policy_version, "pii-prompt-safe/1.3")
        else _PREVIOUS_CORP_SUFFIXES
        if _policy_at_least(security_policy_version, "pii-prompt-safe/1.2")
        else _LEGACY_CORP_SUFFIXES)
    if any(compact.endswith(suffix) for suffix in suffixes):
        return True
    return any(
        compact == _corp_key(str(name))
        for name in corporate_names
        if name is not None and str(name).strip()
    )


def is_organization_name(
        value: str, corporate_names: Iterable[str] = (), *,
        security_policy_version: str | None = None,
        ) -> bool:
    """Public wrapper for the same deterministic organization-name boundary."""

    policy = _checked_security_policy_version(security_policy_version)
    return _is_corporate_name(
        str(value), corporate_names, security_policy_version=policy)


_PRIVACY_REQUEST_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("resident_registration_no", re.compile(r"주민등록번호|외국인등록번호|여권번호")),
    # ``등록번호`` 단독 표현은 개인정보 맥락에서 식별번호를 뜻한다. 접수번호는
    # 별도 공시 식별자이므로 negative lookbehind로 제외한다.
    ("registration_no", re.compile(
        r"사업자등록번호|법인등록번호|개인\s*식별번호|"
        r"(?<!주민)(?<!외국인)(?<!사업자)(?<!법인)(?<!접수)등록번호")),
    ("bank_account", re.compile(r"계좌번호|은행\s*계좌|카드번호")),
    ("employee_id", re.compile(r"사원번호|사번|직원번호")),
    ("birth_date", re.compile(r"생년월일|출생(?:일|연월|년도)?")),
    ("person_name", re.compile(
        r"(?:개인|담당자)\s*(?:성명|이름)|(?:성명|이름)\s*(?:비공개|마스킹)")),
    ("address", re.compile(r"(?:자택|집|개인|상세)?\s*주소|거주지")),
    ("phone", re.compile(r"연락처|전화번호|휴대폰|휴대전화")),
    ("email", re.compile(r"이메일|전자우편|메일\s*주소")),
    ("occupation", re.compile(r"(?:개인\s*)?직업|근무처")),
    ("raw_personal_data", re.compile(
        r"원문\s*(?:값|개인정보)|마스킹\s*(?:전|해제)|"
        r"비식별\s*(?:전|해제)")),
)
# 지분공시 어휘만 있으면 「매출액은 알려주고 주민등록번호는 제외해줘」가
# `restricted_only` 로 판정돼 매출액까지 거절된다(`RPC-016`).  개인정보를 빼
# 달라고 먼저 말한 사용자에게 공개 재무 사실까지 막는 것은 과잉 거절이다.
# 여기 있는 것은 모두 정본에 공개 필드로 실린 항목의 이름이다.
_PUBLIC_DISCLOSURE_REQUEST_RE = re.compile(
    r"지분(?:율|변동|증감)|보유\s*(?:주식|수량|비율|목적)|주식\s*수|"
    r"변동(?:내역|방법|사유|일)?|공시(?:일|내용)|변경(?:일|내용|사유)|"
    r"(?:보고자|최대주주|특별관계자)(?:가|는|의)?\s*"
    r"(?:누구|성명|이름|명칭|직업|국적|사업\s*내용)|"
    r"국적|"
    r"(?:공개(?:된)?\s*)?(?:법인|회사|기관)(?:명|명칭)|"
    r"매출(?:액|원가|구성)?|영업\s*이익|당기\s*순이익|순이익|영업\s*손실|"
    r"자산\s*총계|부채\s*총계|자본\s*총계|이익\s*잉여금|재고\s*자산|"
    r"현금\s*및\s*현금성\s*자산|자기자본|총자산|"
    r"계약\s*(?:금액|상대|기간|명)|수주|"
    r"사업\s*(?:부문|내용)|주요\s*제품"
)


def classify_privacy_request(question: str) -> PrivacyRequestClassification:
    """질문에서 개인정보 요청과 공개 공시 요청을 분리한다.

    사업자·법인 등록번호도 조직의 번호이지만 재식별·오남용 위험이 있어 제한 항목으로
    유지한다. 반면 지분율·보유주식·보고자 같은 공시 항목이 함께 있으면 ``partial``을
    반환해 공개 부분은 답하고 개인정보만 생략할 수 있게 한다.
    """
    text = unicodedata.normalize("NFKC", str(question))
    found = tuple(kind for kind, pattern in _PRIVACY_REQUEST_PATTERNS if pattern.search(text))
    return PrivacyRequestClassification(
        restricted_types=_ordered_types(found),
        has_public_disclosure_request=bool(_PUBLIC_DISCLOSURE_REQUEST_RE.search(text)),
    )


_EMPTY_VALUES = frozenset({
    "", "-", "--", "해당없음", "해당사항없음", "미기재", "없음",
    # 성명 열의 집계 행이다. 사람 이름이 아니므로 masking metadata를 만들지 않는다.
    "계", "합계", "소계",
})
_EMPTY_VALUE_KEYS = frozenset(normalize_label(value) for value in _EMPTY_VALUES)


def _is_empty_value(value: str) -> bool:
    return normalize_label(value) in _EMPTY_VALUE_KEYS


#: 값·주체를 본 뒤에야 확정되는 조건부 종류 → 저장에 적는 실제 종류.
_CONDITIONAL_PII_KINDS = {
    "person_name_or_entity": "person_name",
    "occupation_or_industry": "occupation",
    "holding_subject_name": "person_name",
}


def _resolved_pii_kind(kind: str) -> str:
    return _CONDITIONAL_PII_KINDS.get(kind, kind)


#: 지분공시 보고 주체 값에 붙는 각주·정정 표기. 순수 이름만 남기고 걷어낸다.
_TRAILING_FOOTNOTE_ASTERISKS = re.compile(r"\*+$")
_TRAILING_FOOTNOTE_REF = re.compile(r"주\d+\)$")
_TRAILING_LATIN_SUFFIX = re.compile(r"[A-Za-z]$")
_PURE_HANGUL_SHORT_NAME = re.compile(r"^[가-힣]{2,3}$")
#: 구분(``CRP_TP``/``SPC_TP``)이 자연인이라고 확정한 값에만 쓰는 넓힌 경계.
#: 값만 보는 휴리스틱은 4글자를 법인명과 구별할 수 없어 2~3글자로 묶어 두지만
#: (``에코프로``), 공시가 개인이라고 적어 둔 자리라면 ``임존종보`` 같은 4글자
#: 이름도 사람 이름이다.
_CONFIRMED_HANGUL_NAME = re.compile(r"^[가-힣]{2,4}$")

#: 가림 글자. 한국 관례를 따라 ``○``(U+25CB)를 쓴다 — ``최○범``. ``*`` 는 공시
#: 원문에서 각주 표시로 이미 쓰이므로(``이재상*``) 겹쳐 읽을 수 없다.
MASK_CHAR = "○"


def mask_holding_subject_name(
        value: str, corporate_names: Iterable[str] = ()) -> str:
    """지분공시 보고 주체 성명을 값의 글자 수·구성만으로 부분 가림한다.

    법인·기관 판정은 ``is_organization_name`` 하나에 맡긴다. 각주/정정 표기
    (``이재상*``·``김주영B``·``정 상 수`` 의 정렬 공백)를 걷어낸 뒤에도 순수
    한글 2~3글자가 아니면 — 4글자 이상 이름, 영문·숫자 혼입, ``주1)`` 같은
    각주 참조 자체 — 원문을 보수적으로 그대로 둔다.

    코퍼스 전수 조사(`docs/고칠거.md` 12번)로 순수 한글 2~3글자 값은 전부
    자연인 이름이고, 3글자 651개 중 기관 꼬리(공단·재단·은행·증권·보험·물산…)
    로 끝나는 값은 0개임을 확인했다.

    **한계**: 값 문자열만 보므로 접미사 없는 맨 그룹명(``영풍``·``두산`` —
    실제로는 법인)을 사람 이름과 구별하지 못한다. 1.5 부터는 그 판정을 값이
    아니라 공시가 스스로 적어 둔 ``보고자 구분``(``CRP_TP``)·``구분``
    (``SPC_TP``)으로 하고, 그 결과를 ``project_field_value(party_type=...)``
    로 **적재 시점에** 적용한다 (이슈 #199). 이 함수는 구분을 찾지 못한
    행에서만 쓰는 폴백이다.

    이 규칙은 적재(정본 마스킹)와 실행(답변 계층)이 **함께 쓴다.** 두 곳이
    갈라지면 저장된 값과 표시된 값이 어긋난다.
    """

    if not value:
        return value
    text = str(value)
    if is_organization_name(text, corporate_names):
        return value
    cleaned = _name_residue(text)
    if not _PURE_HANGUL_SHORT_NAME.fullmatch(cleaned):
        return value
    if len(cleaned) == 2:
        return cleaned[0] + MASK_CHAR
    return cleaned[0] + MASK_CHAR + cleaned[2]


def _name_residue(value: str) -> str:
    """각주·정렬 공백을 걷어낸 순수 이름."""

    cleaned = re.sub(r"\s+", "", str(value))
    cleaned = _TRAILING_FOOTNOTE_ASTERISKS.sub("", cleaned)
    cleaned = _TRAILING_FOOTNOTE_REF.sub("", cleaned)
    return _TRAILING_LATIN_SUFFIX.sub("", cleaned)


def mask_confirmed_person_name(value: str) -> str:
    """구분이 **자연인으로 확정한** 지분공시 주체 성명을 가린다.

    한글 2~4글자면 관례대로 부분 가림(``최○범``·``임○○보``)한다. 라틴 표기
    이름(``Scott Samuel Braun``)이나 그 밖의 구성은 부분 가림이 성립하지
    않으므로 통째로 치환한다 — 값 모양으로는 법인처럼 보이는 자연인이 바로
    이슈 #199 가 관측한 노출이다.
    """

    cleaned = _name_residue(value)
    if not _CONFIRMED_HANGUL_NAME.fullmatch(cleaned):
        return _placeholder("person_name")
    if len(cleaned) == 2:
        return cleaned[0] + MASK_CHAR
    return cleaned[0] + MASK_CHAR * (len(cleaned) - 2) + cleaned[-1]


def _placeholder(kind: str) -> str:
    return f"[REDACTED:{kind.upper()}]"


#: 값이 아니라 **공시의 자기 신고**로 갈리는 종류.
_PARTY_TYPED_KINDS = ("person_name", "person_name_or_entity", "holding_subject_name")


def _mask_whole(
        value: str, kind: str, corporate_names: Iterable[str], *,
        security_policy_version: str,
        subject_name: str | None = None,
        party_type: str | None = None) -> tuple[str, bool]:
    if _is_empty_value(value):
        return value, False
    if (party_type is not None
            and kind in _PARTY_TYPED_KINDS
            and _policy_at_least(security_policy_version, _PARTY_TYPE_POLICY)):
        # 공시가 그 자리의 주체를 스스로 「개인(국내)」·「국내법인」으로 적어
        # 두었다. 값 모양 휴리스틱보다 이 신고가 먼저다 (이슈 #199).
        is_person = party_is_person(party_type)
        if is_person is False:
            return value, False
        if is_person is True:
            if kind == "holding_subject_name":
                masked = mask_confirmed_person_name(value)
                return masked, masked != value
            return _placeholder(_resolved_pii_kind(kind)), True
        # 구분이 ``-``·빈칸·미상이면 종전 값 휴리스틱으로 내려간다.
    if kind in ("person_name", "person_name_or_entity") and _is_corporate_name(
        value, corporate_names, security_policy_version=security_policy_version
    ):
        return value, False
    if kind == "holding_subject_name":
        masked = mask_holding_subject_name(value, corporate_names)
        return (masked, masked != value)
    if kind == "occupation_or_industry":
        # 값이 아니라 **주체**가 정한다. 법인의 `자산운용`·`Mutual Fund` 는
        # 업종이고 공시가 공개하는 항목이다. 주체를 모르면 가린다 — 자유
        # 서술(chunk) 경로처럼 행 문맥이 없는 자리에서는 fail-closed 다.
        if subject_name and _is_corporate_name(
                subject_name, corporate_names,
                security_policy_version=security_policy_version):
            return value, False
    actual_kind = _resolved_pii_kind(kind)
    return _placeholder(actual_kind), True


_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])",
    re.IGNORECASE,
)
_RRN_RE = re.compile(r"(?<!\d)\d{6}\s*-\s*[1-8]\d{6}(?!\d)")
_REGISTRATION_RE = re.compile(r"(?<!\d)\d{3}\s*-\s*\d{2}\s*-\s*\d{5}(?!\d)")
_PHONE_RE = re.compile(
    r"(?<!\d)(?:"
    r"\+82[-.\s]?(?:10|1[16789]|2|[3-6]\d|70)[-.\s]?\d{3,4}[-.\s]?\d{4}"
    r"|0(?:10|1[16789]|2|[3-6]\d|70)[-.\s]?\d{3,4}[-.\s]?\d{4}"
    r"|1[568]\d{2}[-.\s]?\d{4}"
    r")(?!\d)"
)
_KOREAN_ADDRESS_RE = re.compile(
    r"(?<![가-힣])(?:서울(?:특별시)?|부산(?:광역시)?|대구(?:광역시)?|인천(?:광역시)?|"
    r"광주(?:광역시)?|대전(?:광역시)?|울산(?:광역시)?|세종(?:특별자치시)?|"
    r"경기(?:도)?|강원(?:특별자치도|도)?|충청[남북]도|전라[남북]도|경상[남북]도|"
    r"제주(?:특별자치도|도)?)\s+[가-힣A-Za-z0-9·.-]+(?:시|군|구)\s+[^|\n]{2,100}"
)
_INLINE_BIRTH_RE = re.compile(
    r"(?P<label>(?:생\s*년\s*월\s*일|생\s*년\s*월|출\s*생\s*(?:일|연월)?)"
    r"\s*[:：)]?\s*)(?P<value>(?:19|20)?\d{2}(?:\s*년|[-./])\s*\d{1,2}"
    r"(?:(?:\s*월|[-./])\s*\d{1,2}\s*일?)?)",
    re.IGNORECASE,
)
_INLINE_NAME_RE = re.compile(
    r"(?P<label>(?:\(\s*성\s*명\s*\)|성\s*명\s*[:：])\s*)"
    r"(?P<value>[가-힣]{1,4}(?:\s+[가-힣]{1,4}){0,3})"
)
_INLINE_BANK_RE = re.compile(
    r"(?P<label>(?:계좌s*번호|은행s*계좌|카드s*번호)\s*[:：]?\s*)"
    r"(?P<value>(?<!\d)[0-9][0-9\s-]{6,30}[0-9](?!\d))",
    re.IGNORECASE,
)
_INLINE_EMPLOYEE_RE = re.compile(
    r"(?P<label>(?:사원s*번호|사번|직원s*번호)\s*[:：]?\s*)"
    r"(?P<value>[A-Za-z0-9][A-Za-z0-9_-]{2,31})",
    re.IGNORECASE,
)


_CONTEXT_WINDOW = 40


def _public_context_before(
        text: str, start: int, kind: str, *, security_policy_version: str) -> bool:
    """매치 직전 문맥이 공개 직위(성명) 또는 회사 소재지(주소)를 가리키는가."""
    # 같은 줄(또는 같은 표 셀) 안의 직전 문맥만 본다 — 앞줄의 「본점 소재지」가 다음 줄 개인 주소를 풀어주면 안 된다.
    before = text[max(0, start - _CONTEXT_WINDOW):start].rsplit("\n", 1)[-1].rsplit("|", 1)[-1]
    if kind in ("person_name", "person_name_or_entity"):
        return _public_role_context(before, security_policy_version=security_policy_version)
    if kind == "address":
        return _public_place_context(before)
    return False


def _replace_pattern(text: str, pattern: re.Pattern[str], kind: str,
                     found: set[str], *, security_policy_version: str) -> str:
    def repl(match: re.Match[str]) -> str:
        if _public_context_before(
                text, match.start(), kind, security_policy_version=security_policy_version):
            return match.group(0)
        found.add(kind)
        return _placeholder(kind)
    return pattern.sub(repl, text)


def _replace_context_value(text: str, pattern: re.Pattern[str], kind: str,
                           found: set[str], corporate_names: tuple[str, ...], *,
                           security_policy_version: str) -> str:
    def repl(match: re.Match[str]) -> str:
        if _public_context_before(
                text, match.start(), kind, security_policy_version=security_policy_version):
            return match.group(0)
        masked, changed = _mask_whole(
            match.group("value"), kind, corporate_names,
            security_policy_version=security_policy_version)
        if changed:
            found.add(kind)
        return match.group("label") + masked
    return pattern.sub(repl, text)


_MD_SEPARATOR = re.compile(r"^\s*:?-{3,}:?\s*$")


def _is_markdown_separator(cells: list[str]) -> bool:
    populated = [cell for cell in cells if cell.strip()]
    return bool(populated) and all(_MD_SEPARATOR.fullmatch(cell) for cell in populated)


def _replace_cell(cell: str, kind: str, found: set[str],
                  corporate_names: tuple[str, ...], *,
                  security_policy_version: str) -> str:
    core = cell.strip()
    masked, changed = _mask_whole(
        core, kind, corporate_names, security_policy_version=security_policy_version)
    if not changed:
        return cell
    found.add(_resolved_pii_kind(kind))
    start = cell[:len(cell) - len(cell.lstrip())]
    end = cell[len(cell.rstrip()):]
    return start + masked + end


def _mask_markdown_tables(text: str, found: set[str],
                          corporate_names: tuple[str, ...], *,
                          security_policy_version: str) -> str:
    """Markdown 표의 PII 헤더 열과 2열 라벨-값 행을 마스킹한다."""
    lines = text.splitlines(keepends=True)
    header_types: list[str | None] | None = None
    result: list[str] = []

    for index, line in enumerate(lines):
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        if "|" not in body:
            header_types = None
            result.append(line)
            continue

        cells = body.split("|")
        next_is_separator = False
        if index + 1 < len(lines) and "|" in lines[index + 1]:
            next_cells = lines[index + 1].rstrip("\r\n").split("|")
            next_is_separator = _is_markdown_separator(next_cells)

        if next_is_separator:
            header_types = [classify_pii_label(
                cell, security_policy_version=security_policy_version) for cell in cells]
            # 직위·담당업무·소유주식 같은 머리글이 있는 표(임원 현황·최대주주 현황)의 성명 열은 공개 정보다.
            if any(marker in normalize_label(cell) for cell in cells for marker in _PUBLIC_TABLE_MARKERS):
                header_types = [None if kind in ("person_name", "person_name_or_entity") else kind
                                for kind in header_types]
            result.append(line)
            continue
        if _is_markdown_separator(cells):
            result.append(line)
            continue

        if header_types is not None and len(header_types) == len(cells):
            for col, kind in enumerate(header_types):
                if kind:
                    cells[col] = _replace_cell(
                        cells[col], kind, found, corporate_names,
                        security_policy_version=security_policy_version)

        # `| 주소 | 서울 ... |`처럼 고정 헤더가 없는 key-value 표.
        for col in range(len(cells) - 1):
            kind = classify_pii_label(
                cells[col], security_policy_version=security_policy_version)
            if kind and classify_pii_label(
                    cells[col + 1], security_policy_version=security_policy_version) is None:
                cells[col + 1] = _replace_cell(
                    cells[col + 1], kind, found, corporate_names,
                    security_policy_version=security_policy_version)

        result.append("|".join(cells) + ending)
    return "".join(result)


def _normalize_controls(text: str) -> tuple[str, bool]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    changed = False
    chars: list[str] = []
    for char in text:
        if unicodedata.category(char) == "Cc" and char not in "\n\t":
            chars.append(" ")
            changed = True
        else:
            chars.append(char)
    return "".join(chars), changed


_ACTIVE_BLOCK_RE = re.compile(
    r"(?is)<(?:script|iframe|object|embed|style|form)\b[^>]*>.*?"
    r"(?:</(?:script|iframe|object|embed|style|form)\s*>|\Z)"
)
_ACTIVE_TAG_RE = re.compile(
    r"(?is)</?(?:script|iframe|object|embed|style|form|meta|link)\b[^>]*>"
)
_EVENT_HANDLER_RE = re.compile(r"(?is)\s+on[a-z][a-z0-9_-]*\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)")
_ACTIVE_URI_RE = re.compile(r"(?i)(?:javascript\s*:|data\s*:\s*text/html)[^\s)\]>]*")


def _remove_active_content(text: str) -> tuple[str, bool]:
    changed = False

    def remove(_match: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        return _ACTIVE_REMOVED

    result = _ACTIVE_BLOCK_RE.sub(remove, text)
    result = _ACTIVE_TAG_RE.sub(remove, result)
    result = _EVENT_HANDLER_RE.sub(remove, result)
    result = _ACTIVE_URI_RE.sub(remove, result)

    # Entity-encoded script/URI도 renderer가 decode하면 활성화될 수 있다. 해당 줄만 버린다.
    safe_lines: list[str] = []
    for line in result.splitlines(keepends=True):
        decoded = html.unescape(line)
        if decoded != line and (
            _ACTIVE_TAG_RE.search(decoded) or _ACTIVE_URI_RE.search(decoded)
            or _EVENT_HANDLER_RE.search(decoded)
        ):
            ending = "\n" if line.endswith(("\n", "\r")) else ""
            safe_lines.append(_ACTIVE_REMOVED + ending)
            changed = True
        else:
            safe_lines.append(line)
    return "".join(safe_lines), changed


_PROMPT_LIKE_RE = re.compile(
    r"(?:"
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|messages?)"
    r"|reveal\s+(?:the\s+)?(?:system|developer)\s+(?:prompt|message)"
    r"|(?:system|developer)\s+prompt\s*[:：]"
    r"|you\s+are\s+(?:chatgpt|an?\s+ai\s+assistant)"
    r"|follow\s+(?:only\s+)?(?:these|the\s+following)\s+instructions?"
    r"|이\s*전\s*(?:의\s*)?(?:지\s*시|명\s*령|프\s*롬\s*프\s*트).{0,20}무\s*시"
    r"|(?:시\s*스\s*템|개\s*발\s*자)\s*(?:프\s*롬\s*프\s*트|메\s*시\s*지)\s*[:：]"
    r"|(?:너|당신)\s*은?.{0,12}(?:chatgpt|챗\s*gpt|ai\s*(?:도우미|assistant))"
    r"|다\s*음\s*(?:지\s*시|명\s*령).{0,12}따\s*르"
    r")",
    re.IGNORECASE,
)


def _remove_prompt_instructions(text: str) -> tuple[str, bool]:
    changed = False
    output: list[str] = []
    for line in text.splitlines(keepends=True):
        if _PROMPT_LIKE_RE.search(html.unescape(line)):
            ending = "\n" if line.endswith(("\n", "\r")) else ""
            output.append(_INSTRUCTION_REMOVED + ending)
            changed = True
        else:
            output.append(line)
    return "".join(output), changed


def _mask_pii(
        text: str, corporate_names: tuple[str, ...], *,
        ignore_types: Iterable[str] = (),
        security_policy_version: str,
        ) -> tuple[str, tuple[str, ...]]:
    found: set[str] = set()
    ignored = set(ignore_types)
    result = _mask_markdown_tables(
        text, found, corporate_names, security_policy_version=security_policy_version)
    if "resident_registration_no" not in ignored:
        result = _replace_pattern(
            result, _RRN_RE, "resident_registration_no", found,
            security_policy_version=security_policy_version)
    if "registration_no" not in ignored:
        result = _replace_pattern(
            result, _REGISTRATION_RE, "registration_no", found,
            security_policy_version=security_policy_version)
    if "email" not in ignored:
        result = _replace_pattern(
            result, _EMAIL_RE, "email", found,
            security_policy_version=security_policy_version)
    if "phone" not in ignored:
        result = _replace_pattern(
            result, _PHONE_RE, "phone", found,
            security_policy_version=security_policy_version)
    if "birth_date" not in ignored:
        result = _replace_context_value(result, _INLINE_BIRTH_RE, "birth_date", found,
                                        corporate_names,
                                        security_policy_version=security_policy_version)
    if "person_name" not in ignored:
        result = _replace_context_value(result, _INLINE_NAME_RE, "person_name", found,
                                        corporate_names,
                                        security_policy_version=security_policy_version)
    if _is_modern_policy(security_policy_version):
        if "bank_account" not in ignored:
            result = _replace_context_value(
                result, _INLINE_BANK_RE, "bank_account", found,
                corporate_names,
                security_policy_version=security_policy_version)
        if "employee_id" not in ignored:
            result = _replace_context_value(
                result, _INLINE_EMPLOYEE_RE, "employee_id", found,
                corporate_names,
                security_policy_version=security_policy_version)
    if "address" not in ignored:
        result = _replace_pattern(
            result, _KOREAN_ADDRESS_RE, "address", found,
            security_policy_version=security_policy_version)
    return result, _ordered_types(found)


def project_chunk_text(
        text: str, corporate_names: Iterable[str] = (), *,
        security_policy_version: str | None = None) -> SecurityProjection:
    """Chunk raw text에서 검색용·LLM 전달용 projection을 결정적으로 만든다."""
    policy = _checked_security_policy_version(security_policy_version)
    names = tuple(str(name) for name in corporate_names if name)
    # 1.1 build는 Document의 corp_name/listed_name 두 값만 전달했다. 1.2에서 filer를
    # 추가했으므로, 과거 artifact 재검증 시 새 문맥으로 결과가 바뀌지 않게 정확히 절단한다.
    if policy == _LEGACY_SECURITY_POLICY_VERSION:
        names = names[:2]
    normalized, controls_removed = _normalize_controls(str(text))
    masked, pii_types = _mask_pii(
        normalized, names, security_policy_version=policy)
    search, active_removed = _remove_active_content(masked)

    prompt_body, instruction_removed = _remove_prompt_instructions(search)
    boundary_escaped = PROMPT_DATA_BEGIN in prompt_body or PROMPT_DATA_END in prompt_body
    if boundary_escaped:
        prompt_body = prompt_body.replace(PROMPT_DATA_BEGIN, "[ESCAPED:DATA_BOUNDARY]")
        prompt_body = prompt_body.replace(PROMPT_DATA_END, "[ESCAPED:DATA_BOUNDARY]")

    flags: list[str] = []
    if pii_types:
        flags.append("pii_masked")
    if active_removed:
        flags.append("active_content_removed")
    if instruction_removed:
        flags.append("prompt_instruction_removed")
    if controls_removed:
        flags.append("control_chars_removed")
    if boundary_escaped:
        flags.append("boundary_marker_escaped")

    prompt_safe = f"{PROMPT_DATA_BEGIN}\n{prompt_body}\n{PROMPT_DATA_END}"
    return SecurityProjection(
        text_search=search,
        text_prompt_safe=prompt_safe,
        security_flags=tuple(sorted(flags)),
        pii_types=pii_types,
        security_policy_version=policy,
    )


#: 원문 한 행의 주체(성명·명칭)를 찾는 **하나뿐인 규칙**.  적재와 읽기가 이
#: 함수를 함께 써야 저장된 마스킹과 재검증이 어긋나지 않는다 (이슈 #139).
SUBJECT_ACODES = ("SPC_NM",)

def needs_subject_name(label: str) -> bool:
    """이 라벨의 투영이 ``subject_name`` 을 보는가.

    주체 조회는 원문 행을 되짚는 일이라 값이 있다. 필드 160만 행마다 부르면
    조회가 통째로 느려진다 — 실제로 필요한 것은 `직 업(사업내용)` 계열
    17,284행뿐이다. 호출자가 이 값으로 먼저 걸러 낸다 (이슈 #139).
    """

    key = normalize_label(re.split(r"\s*>\s*", str(label))[-1])
    return "직업" in key or "직장" in key



def subject_names_by_row(rows) -> dict[tuple[object, object], str]:
    """``(table_locator, logical_row) -> 그 행의 성명(명칭)``.

    ``rows`` 는 ``(acode, value, table_locator, logical_row)`` 를 내는 것이면
    무엇이든 좋다. 같은 좌표가 여러 번 나오면 **처음 것**을 쓴다 — 순서가
    바뀌어도 같은 답이 나오게 정렬 의존을 없앤다.
    """

    found: dict[tuple[object, object], str] = {}
    for acode, value, table_locator, logical_row in rows:
        if str(acode) not in SUBJECT_ACODES:
            continue
        if table_locator is None or logical_row is None:
            continue
        text = str(value or "").strip()
        if not text or text == "-":
            continue
        found.setdefault((table_locator, logical_row), text)
    return found



#: 지분공시가 스스로 적어 둔 주체 구분. 문서 단위 ``보고자 구분``(``CRP_TP``)
#: 하나와 판별표의 행별 ``구분``(``SPC_TP``)이다. 실측 1,083개 지분공시 전부에
#: ``CRP_TP`` 가 있고 ``SPC_TP`` 는 16,201행이다.
DOCUMENT_PARTY_TYPE_AUNIT = "CRP_TP"
ROW_PARTY_TYPE_AUNIT = "SPC_TP"

#: ``aunitvalue`` 코드. 텍스트가 비어 있을 때를 위해 함께 받는다.
_PARTY_PERSON_CODES = frozenset({"D", "F"})
_PARTY_NON_PERSON_CODES = frozenset({"I", "O", "Y", "K", "N", "U", "W"})
_PARTY_NON_PERSON_TEXTS = (
    "국내법인", "외국법인", "연기금등 전문투자자", "금융기관",
    "기타단체(국내)", "기타단체(외국)", "법령상 조합",
)
_PARTY_NON_PERSON_KEYS = frozenset(
    normalize_label(text) for text in _PARTY_NON_PERSON_TEXTS)

#: 이 문서의 보고자 본인을 가리키는 성명 라벨. ``특별관계자``·``공동보유자``
#: 가 함께 적힌 경로는 보고자 행이 아니다.
_NON_REPORTER_MARKERS = ("특별관계자", "공동보유자")


def needs_party_type(label: str) -> bool:
    """이 라벨의 투영이 ``party_type`` 을 보는가.

    ``성명(명칭)`` 계열 — 즉 :func:`classify_pii_label` 이 ``person_name`` ·
    ``person_name_or_entity`` · ``holding_subject_name`` 으로 보내는 라벨
    — 만 해당한다. ``구분``·``직 업(사업내용)`` 은 여기에 걸리지 않는다.
    주체 조회는 원문 행을 되짚는 일이라 값이 있으므로, 호출자가 이 값으로
    먼저 걸러 낸다 (``needs_subject_name`` 과 같은 이유).
    """

    return _classify_pii_label_raw(
        str(label), security_policy_version=SECURITY_POLICY_VERSION,
    ) in ("person_name", "person_name_or_entity")


def party_is_person(party_type: str | None) -> bool | None:
    """``구분`` 이 자연인을 뜻하는가. 모르면 ``None``.

    ``개인(국내)``·``개인(외국)``(코드 ``D``/``F``)만 자연인이다. ``-``·빈칸·
    처음 보는 표기는 **판정하지 않는다** — 그 자리에서는 호출자가 종전 값
    휴리스틱으로 내려가야 한다.
    """

    if party_type is None:
        return None
    text = unicodedata.normalize("NFC", str(party_type)).strip()
    if not text or text == "-":
        return None
    if text in _PARTY_PERSON_CODES:
        return True
    if text in _PARTY_NON_PERSON_CODES:
        return False
    key = normalize_label(text)
    if not key:
        return None
    if key.startswith("개인"):
        return True
    if key in _PARTY_NON_PERSON_KEYS:
        return False
    return None


def _party_type_text(value: object) -> str:
    text = str(value or "").strip()
    return "" if text == "-" else text


def _holding_name_key(value: object) -> str:
    """이름 대조키. 각주 꼬리(``박승덕 주4)``·``이재상*``·``김주영B``)와
    정렬 공백(``정 상 수``)을 걷어내고 NFC 로 맞춘다."""

    text = unicodedata.normalize("NFC", str(value or "")).strip()
    if not text:
        return ""
    return _name_residue(text).casefold()


#: 지분공시의 ``업무상 연락처및 담당자 > 성명`` 처럼 **공시 담당자**를 적는
#: 라벨. 담당자는 언제나 자연인이고 보유 주체가 아니므로 ``구분`` 지도로
#: 판정하지 않는다 — 같은 문서의 법인명과 우연히 같은 글자여도 공개로
#: 돌리지 않고 종전 규칙(통째 가림)을 그대로 둔다.
_CONTACT_LABEL_MARKERS = ("담당자", "연락처")


def _is_contact_label(label: str) -> bool:
    full = normalize_label(label)
    return any(marker in full for marker in _CONTACT_LABEL_MARKERS)


def _is_reporter_label(label: str) -> bool:
    full = normalize_label(label)
    if any(marker in full for marker in _NON_REPORTER_MARKERS):
        return False
    leaf = normalize_label(re.split(r"\s*>\s*", str(label))[-1])
    return "보고자" in full or leaf == "본인성명"


@dataclass(frozen=True)
class HoldingPartyTypes:
    """지분공시 한 문서의 ``구분`` 지도.

    ``by_row``
        ``(table_locator, logical_row) -> 구분 원문``. 판별표는 ``성 명(명칭)``
        과 ``구분`` 을 같은 행에 두므로 이 짝은 정확하다.
    ``by_name``
        ``정규화한 이름 -> 구분 원문``. 판별표 밖의 표(계약 내역·요약표)에는
        ``구분`` 열이 아예 없다. 같은 문서 안에서 이미 짝지어진 이름을 그대로
        옮겨 온다.
    ``document_type``
        문서 단위 ``보고자 구분``(``CRP_TP``).
    """

    by_row: dict[tuple[object, object], str]
    by_name: dict[str, str]
    document_type: str | None = None


def holding_party_types_by_row(rows) -> HoldingPartyTypes:
    """지분공시 원문 행들에서 ``구분`` 지도를 만든다.

    ``rows`` 는 ``(path, aunit, value_raw, table_locator, logical_row)`` 를
    내는 것이면 무엇이든 좋다. 적재(`src/canonical/build.py`)와 읽기
    (`src/canonical/read.py`)와 게이트(`tests/verify_security_evidence.py`)가
    **이 함수 하나**를 함께 쓴다. 세 곳이 갈라지면 저장된 마스킹과 재검증이
    어긋난다 — 이슈 #139 에서 그 어긋남으로 정본을 두 번 다시 만들었다
    (이슈 #160).
    """

    items = [tuple(row) for row in rows]
    by_row: dict[tuple[object, object], str] = {}
    document_type: str | None = None
    names: list[tuple[str, str, object, object]] = []
    for label, aunit, value, table_locator, logical_row in items:
        unit = str(aunit or "")
        if unit == DOCUMENT_PARTY_TYPE_AUNIT:
            text = _party_type_text(value)
            if document_type is None and text:
                document_type = text
            continue
        if unit == ROW_PARTY_TYPE_AUNIT:
            text = _party_type_text(value)
            if text and table_locator is not None and logical_row is not None:
                by_row.setdefault((table_locator, logical_row), text)
            continue
        name = str(value or "").strip()
        if name and needs_party_type(label) and not _is_contact_label(label):
            names.append((str(label), name, table_locator, logical_row))

    candidates: dict[str, list[str]] = {}
    for label, name, table_locator, logical_row in names:
        found = by_row.get((table_locator, logical_row))
        if found is None and _is_reporter_label(label):
            found = document_type
        key = _holding_name_key(name)
        if found and key:
            candidates.setdefault(key, []).append(found)

    by_name: dict[str, str] = {}
    for key, found in candidates.items():
        kinds = {party_is_person(text) for text in found}
        kinds.discard(None)
        if len(kinds) > 1:
            # 같은 이름이 한 문서 안에서 사람과 법인 양쪽으로 적혔다.
            # 어느 쪽도 믿을 수 없으므로 값 휴리스틱으로 되돌린다.
            continue
        # 저장에 쓰는 것은 사람/법인 판정뿐이라 어느 표기를 골라도 결과는
        # 같지만, 적재와 읽기가 같은 문자열을 고르도록 정렬로 못 박는다.
        by_name[key] = sorted(found)[0]
    return HoldingPartyTypes(by_row, by_name, document_type)


def resolve_party_type(pairing, label: str, value: str,
                       table_locator: object, logical_row: object) -> str | None:
    """이 성명 칸의 ``구분``. 같은 행이 먼저, 없으면 문서 안의 이름 지도.

    적재·읽기·게이트가 모두 이 함수를 통해서만 ``party_type`` 을 얻는다.
    """

    if pairing is None or not needs_party_type(label) or _is_contact_label(label):
        return None
    found = pairing.by_row.get((table_locator, logical_row))
    if found:
        return found
    key = _holding_name_key(value)
    return pairing.by_name.get(key) if key else None


def project_field_value(label: str, value: str,
                        corporate_names: Iterable[str] = (), *,
                        subject_name: str | None = None,
                        party_type: str | None = None,
                        security_policy_version: str | None = None) -> FieldProjection:
    """Field raw value의 안전 projection. raw 값은 호출자가 별도 컬럼에 그대로 보존한다.

    ``subject_name`` 은 그 값이 **누구의 것인지** — 같은 원문 행의 성명(명칭)
    이다. 지분공시의 `직 업(사업내용)` 은 한 라벨 아래 자연인의 직위와 법인의
    업종이 섞여 있어 라벨로도 값으로도 가를 수 없다 (이슈 #139). 주체가
    법인이면 업종이므로 가리지 않는다. **주지 않으면 가린다** — 적재와 읽기가
    같은 방식으로 이 값을 만들어야 저장된 마스킹이 재검증을 통과한다.

    ``party_type`` 은 그 주체가 **무엇인지** — 같은 원문 행의 ``구분``
    (``SPC_TP``)이나 문서의 ``보고자 구분``(``CRP_TP``) 원문이다. 지분공시의
    ``성명(명칭)`` 칸에는 법인과 자연인이 함께 오는데, 값 모양만 보면
    ``에코프로``는 사람으로 ``Scott Samuel Braun``은 법인으로 갈린다
    (이슈 #199). 공시가 스스로 적어 둔 이 구분이 있으면 그것으로 정하고,
    없으면(``-``·빈칸) 종전 휴리스틱으로 내려간다. 1.4 이하 판으로 적힌 행은
    이 인자를 주더라도 **그 판의 규칙 그대로** 재계산한다.
    """
    policy = _checked_security_policy_version(security_policy_version)
    names = tuple(str(name) for name in corporate_names if name)
    if policy == _LEGACY_SECURITY_POLICY_VERSION:
        names = names[:2]
    raw_kind = _classify_pii_label_raw(
        label, security_policy_version=policy)
    kind = classify_pii_label(label, security_policy_version=policy)
    if kind:
        masked, changed = _mask_whole(
            str(value), kind, names, security_policy_version=policy,
            subject_name=subject_name, party_type=party_type)
        if changed:
            return FieldProjection(masked, _resolved_pii_kind(kind), True, policy)
        # 빈 값은 실제 노출이 없지만 필드 자체의 정책 분류는 보존한다. 법인명·
        # 법인 업종은 예외다 — 가리지 않기로 판정한 값이므로 PII 로 적지 않는다.
        if kind not in _CONDITIONAL_PII_KINDS and _is_empty_value(str(value)):
            return FieldProjection(str(value), kind, True, policy)

    # 공개 직위자의 성명과 회사 본점/사업장 주소는 라벨 문맥으로 이미
    # 공개 가능하다고 판정했다. 일반 value regex를 다시 태워 같은 종류를
    # 가리는 이중 마스킹만 건너뛰고, 한 셀에 섞인 전화·이메일 등 다른 PII는
    # 계속 제거한다.
    ignored: tuple[str, ...] = ()
    if (_is_modern_policy(policy)
            and kind is None and raw_kind is not None):
        ignored = (_resolved_pii_kind(raw_kind),)
    if _public_address_content(label, security_policy_version=policy):
        ignored = ignored + ("address",)
    masked, types = _mask_pii(
        str(value), names, ignore_types=ignored, security_policy_version=policy)
    pii_type = "|".join(types) or None
    return FieldProjection(masked, pii_type, bool(types), policy)
