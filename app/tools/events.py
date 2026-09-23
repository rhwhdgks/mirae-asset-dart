"""EventTool / DisclosureTool / CorrectionTool / DocumentTool — 비정기공시 Tier 1.

공통 원칙
- 판정은 read.py가 내린다 (event_timeline / lookup_field / correction_items / resolve_document_version).
- 확정된 사건은 selector.event_key로 직접 조회한다. seed·상대방·계약명은 event_key가 없는
  collection 탐색에만 쓴다.
- 확정 근거는 Field Evidence(verified)로만 인용한다. 관측 supporting Evidence는 partial일 수 있어
  claim citation으로 승격하지 않는다.
- identity_status=ambiguous이면 원본 특정 불가를 병기한다. 해지 관측의 계약기간이
  후보 하나와만 일치해도 이는 후보 구분 정보일 뿐 canonical 계보를 확정하지 않는다.
- as_of 이후 문서는 read.py가 이미 잘라내므로 여기서 다시 필터하지 않는다.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from app.orchestrator.payload import (
    AnswerClaim, ClaimCitation, Clarification, Limitation, TraceEvent,
)
from app.runtime_deadline import ensure_request_time_remaining
from app.textkit import josa
from app.tools._units import to_won

# 서식 공시에서 답변에 흔히 필요한 slot → Field 라벨(path 끝 조각) 매핑.
# 정확 path가 아니라 label 포함검색이므로 lookup_field가 ambiguous면 후보를 좁힌다.
SLOT_LABELS = {
    # 서식 공시 slot → Field 라벨(path 끝 조각) 포함검색 후보 (실측 라벨 기준, 우선순위 순)
    "계약금액": ["계약금액(원)", "계약금액"], "정정후_계약금액": ["계약금액(원)", "계약금액"],
    "계약상대": ["계약상대"], "계약상대방": ["계약상대"], "상대방": ["계약상대"],
    "해지금액": ["해지금액(원)", "해지금액"],
    # "해지사유"는 단일판매·공급계약해지 서식의 "해지 주요사유"를 먼저 찾는다.
    # 자기주식취득신탁계약해지결정(CG-001, 이슈 #151)은 같은 뜻을 "3. 해지목적"
    # 으로 공시한다 — 두 라벨이 모두 없을 때만 이 폴백까지 내려온다.
    "해지사유": ["해지 주요사유", "해지사유", "해지목적"],
    "해지일자": ["해지일자"], "해지계약명": ["해지계약명"],
    # A termination form can expose the amount as two child rows under one
    # parent label.  Keep the public semantic roles independent of the exact
    # numbered form path; ``field_value`` still requires a unique verified
    # leaf in the canonically selected receipt.
    "해지 전 계약금액": ["해지 전"], "해지전계약금액": ["해지 전"],
    "해지 후 계약금액": ["해지 후"], "해지후계약금액": ["해지 후"],
    "해지예정일": ["해지예정일자", "해지예정일"],
    "해지 목적": ["해지목적", "해지 목적"],
    "해지목적": ["해지목적", "해지 목적"],
    "계약명": ["해지계약명", "계약명", "판매ㆍ공급계약 내용", "판매·공급계약 내용"],
    "계약기간": ["계약기간"], "시작일": ["시작일"], "종료일": ["종료일"],
    "계약(수주)일자": ["계약(수주)일자", "계약일자", "계약일"],
    "효력발생조건": ["계약의 효력발생 조건", "효력발생조건", "기타 투자판단"],
    "매출액대비": ["매출액대비(%)", "매출액대비", "매출액 대비"],
    "최근매출액": ["최근매출액(원)", "최근매출액"],
    "정정사유": ["정정사유"], "공시유보여부": ["유보사유"], "유보사유": ["유보사유"], "유보기한": ["유보기한"],
    "최초제출일": ["최초제출일", "정정관련 공시서류제출일"],
    "결정금액": ["금액", "발행금액", "총액"], "결정일": ["결정일", "이사회결의일"], "조달유형": ["구분", "종류"],
    "해지연결상태": [],
    "해지종료공시관측여부": [],
    "원계약계보관측": [],
    # 주요사항보고서(해외증권시장주권등상장폐지결정) — issue #121.  질문
    # 표면("상장폐지 사유"/"해당 시장")은 서식 라벨("5. 폐지사유"/
    # "2. 상장거래소(소재국가)")과 글자 수가 달라 기존 포함검색이 걸리지
    # 않았다.  정본에 이 서식이 있는데도 값 조회가 not_found로 막혀
    # evidence_unavailable로 빠지고, composer의 상장폐지 총칭 거절 문구와
    # 겹쳐 "공시가 없다"는 거짓 단정을 만들었다.
    "상장폐지사유": ["폐지사유"],
    "해당시장": ["상장거래소"],

    # 이슈 #151 — coverage_gap v0.2 주요사항보고서 축(CG-001~012·019~025).
    # 서식별 필드 키·원문 라벨은 `out/canonical`에서 각 접수번호(grounding
    # v0.2)의 실측 Field 행을 직접 읽어 확인했다(값 해석은 바꾸지 않는다).
    #
    # 주식교환ㆍ이전결정(CG-002·019, 20260326000880) — "교환 사유"는
    # "6. 교환ㆍ이전 목적"으로 공시된다.
    "교환사유": ["교환ㆍ이전 목적"],
    # 같은 서식에서 질문은 대상법인의 전체 표제까지 포함하거나 가운데점 표기를
    # 달리할 수 있다. 슬롯 정규화는 공백만 제거하므로 이 두 공개 필드를 질문
    # 표면 그대로도 찾을 수 있게 한다. 아래 full-path 결속이 동명이의
    # "보통주식" 행을 가리므로 leaf 추측으로 넓어지지는 않는다.
    "교환·이전대상법인의발행주식총수": ["발행주식총수(주)"],
    "교환ㆍ이전대상법인의발행주식총수": ["발행주식총수(주)"],
    "교환·이전비율": ["교환ㆍ이전 비율"],
    "교환ㆍ이전비율": ["교환ㆍ이전 비율"],
    # 회사분할합병결정(CG-003, 20241210000298) — 두산에너빌리티는 분할합병과
    # 함께 모회사 자기주식을 감자하므로, 서식은 분할비율을 "라. 감자에 관한
    # 사항 > 감자비율(%)"에 싣는다(값 11.57542는 분할합병비율 산출근거의
    # "인적분할비율 1:0.1157542"와 일치 — 별도 "분할비율" 필드는 없다).
    "분할비율": ["감자비율(%)"],
    # 소송등의제기(CG-005, 20260108000598) — "소송 명칭"은 "1. 사건의 명칭"
    # 으로 공시된다. "관할 법원"·"원고"는 "4. 관할법원"·"2. 원고ㆍ신청인"에
    # 이미 포함검색이 걸려 별도 등록이 필요 없다.
    "소송명칭": ["사건의 명칭"],
    # 무상증자결정(CG-006, 20251029000121) — "1주당 액면가"는
    # "2. 1주당 액면가액 (원)"으로 공시된다(액면가액≠액면가 표기 차이).
    "액면가": ["1주당 액면가액 (원)"], "1주당액면가": ["1주당 액면가액 (원)"],
    # 교환사채권발행결정(CG-007, 20240422000499) — "교환가액"은 인접한
    # "교환가액 결정방법"·"교환가액 조정에 관한 사항"과 구분해 "(원/주)" 단위가
    # 붙은 실측 라벨로만 결속한다. "발행 지역"은 "2-1. (해외발행) > 발행지역"
    # 에 이미 포함검색이 걸린다.
    "교환가액": ["교환가액 (원/주)"],
    # 유형자산양수결정(CG-008, 20251219000396) — "자산총액 대비 비율"은
    # "2. 양수내역 > 자산총액대비(%)"로 공시된다. 거래상대방의 본점 소재지는
    # 정본에 보존하되 값만 [REDACTED:ADDRESS]로 치환한다. 필드 자체는 조회해
    # 공개 답변 렌더러가 원문 토큰 대신 "비공개"라고 안전하게 표시하게 한다.
    "자산총액대비비율": ["자산총액대비(%)"], "자산총액대비": ["자산총액대비(%)"],
    "거래상대방의본점소재지": ["본점소재지(주소)"],
    # 영업양수결정(CG-009, 20250328000083) — "양수 금액"은 "3. 양수가액(원)"
    # (가액≠금액 표기 차이), "양수 상대방"은 "7. 거래상대방 > 회사명(성명)"
    # 으로 공시된다.
    "양수금액": ["양수가액(원)"], "양수상대방": ["회사명(성명)"],
    # 영업정지(CG-010, 20251216000493) — 실호출 wire(field_surfaces)는 질문
    # 표면 그대로 "대상 사업"·"직전 사업연도 매출액"을 낸다. 라벨은
    # "1. 영업정지 분야"·"2. 영업정지 내역 > 최근매출총액 (원)"이다. (이
    # 문항은 wire가 event/metric 두 answer_item으로 쪼개지고 사건 항목에
    # 날짜 축도 없어 라우팅 자체가 별도로 막혀 있다 — #151 보고 참조.)
    "대상사업": ["영업정지 분야"],
    "정지대상사업": ["영업정지 분야"],
    "직전사업연도매출액": ["최근매출총액 (원)"],
    # 타법인주식및출자증권양도결정(CG-012, 20250221001944)·양수결정(CG-020,
    # 20250611000066) — "양도/양수 후 지분율"은 "지분비율(%)"으로 공시된다
    # (지분율≠지분비율 표기 차이). "양도 주식 수"는
    # "2. 양도내역 > 양도주식수(주)"에, "양수 목적"은 "4. 양수목적"에 이미
    # 포함검색이 걸린다.
    "양도후지분율": ["지분비율(%)"], "양수후지분율": ["지분비율(%)"],
    # 자본으로인정되는채무증권발행결정(CG-021, 20250611000057) — 이 서식은
    # 발행 상대방마다 별도 금액 행을 두는 다자간 구조다(THE CAPITAL EL
    # PTE. LTD.·OV Principal Investments LLC 등 8곳, 금액이 서로 다름).
    # "발행 대상자명"·"발행권면(전자등록) 총액(원)" 라벨을 그대로 등록하되
    # 상대방을 임의로 하나 고르지 않는다 — 값이 여럿이면 lookup은 정직하게
    # ambiguous를 낸다(그중 하나를 확정값으로 지어내지 않는다).
    "발행상대방": ["발행 대상자명"],
    "발행금액": ["발행권면(전자등록) 총액(원)"],
    "권면총액": ["발행권면(전자등록) 총액(원)"],
    "발행권면총액": ["발행권면(전자등록) 총액(원)"],
    # 자기전환사채매도결정(CG-023, 20260310002968) — "취득 사유"는
    # "6. 매도 대상 사채의 만기전 취득내역 > 취득 경위"로, "취득 금액"은 같은
    # 절의 "매도 대상 사채의 권면(전자등록)금액(원)"으로 공시된다.
    "취득사유": ["취득 경위"], "취득금액": ["매도 대상 사채의 권면(전자등록)금액(원)"],
    # 해외증권시장주권등상장결정(CG-024, 20241031000536) — 상장폐지결정의
    # "폐지사유"(#121)와 짝을 이루는 상장(신규) 결정 서식. "상장 사유"는
    # "6. 해외상장목적"으로 공시된다("해당 시장"과 짝인 "5. 상장거래소
    # (소재국가)"는 CG-011의 "해당시장" 별칭과 같은 leaf라 재사용된다).
    "상장사유": ["해외상장목적"],
}

# Some exchange forms repeat the same leaf label in unrelated table groups.
# A semantic slot for one of those fields must bind to the full DART form
# path, not whichever duplicate leaf happens to rank first.  These are form
# coordinates only: there is no issuer, receipt, question ID, or expected
# value in the mapping, so the rule transfers to any filing using the same
# disclosed structure.
SLOT_CANONICAL_PATHS = {
    "처분예정보통주식수": (
        "1. 처분예정주식(주) > 보통주식",
    ),
    "사채의권면(전자등록)총액": (
        "2. 사채의 권면(전자등록)총액 (원)",
    ),
    "분할신설회사의주요사업": (
        "7. 분할설립회사 > 주요사업",
        "7. 분할신설회사 > 주요사업",
    ),
    "재상장신청여부": (
        "7. 분할설립회사 > 재상장신청 여부",
        "7. 분할신설회사 > 재상장신청 여부",
    ),
    "해지주요사유": (
        "5. 해지 주요사유",
    ),
    "발행회차": (
        "1. 사채의 종류 > 회차",
    ),
    "운영자금규모": (
        "3. 자금조달의 목적 > 운영자금 (원)",
    ),

    # 이슈 #151 — coverage_gap v0.2 주요사항보고서 축. leaf 라벨이 같은 서식
    # 안에서 중복되는 자리라 SLOT_LABELS 포함검색만으로는 ambiguous가 난다.
    # (SLOT_LABELS 블록의 주석대로, 값·접수번호·문항 ID는 여기 없다 — 서식
    # 좌표만 있어 같은 구조의 다른 접수건에도 그대로 적용된다.)
    #
    # 자기주식취득신탁계약해지결정(CG-001) — "보통주식" leaf가 절(7절 해지
    # 전 보유현황·10절 해지 후 보유예상기간 등) 여러 곳에 반복된다. "해지
    # 후 보유 자기주식 수"는 "9. 해지예정주식(주)" 절의 보통주식 행이다.
    "해지후보유자기주식수": (
        "9. 해지예정주식(주) > 보통주식",
    ),
    # 주식교환ㆍ이전결정(CG-002·019) — "보통주식" leaf가 발행주식총수·자본
    # 등 여러 표에 나온다. 완전자회사 전환(포괄적 주식교환)이라 교환 대상
    # 주식 수는 대상법인의 발행주식총수와 같다.
    "교환대상주식수": (
        "2. 교환ㆍ이전 대상법인 > 마. 발행주식총수(주) > 보통주식",
    ),
    "교환·이전대상법인의발행주식총수": (
        "2. 교환ㆍ이전 대상법인 > 마. 발행주식총수(주) > 보통주식",
    ),
    "교환ㆍ이전대상법인의발행주식총수": (
        "2. 교환ㆍ이전 대상법인 > 마. 발행주식총수(주) > 보통주식",
    ),
    "교환·이전비율": (
        "3. 교환ㆍ이전 비율",
    ),
    "교환ㆍ이전비율": (
        "3. 교환ㆍ이전 비율",
    ),
    # 감자결정(CG-004) — "기타주식 (주)"(공백 포함)가 "1. 감자주식의 종류와
    # 수" 한 곳에만 있지만, 공백 없는 "기타주식(주)"가 "4. 감자전후
    # 발행주식수"에 다른 값으로 나와 leaf만으로는 오탐 위험이 있다.
    "감자대상주식수": (
        "1. 감자주식의 종류와 수 > 기타주식 (주)",
    ),
    # 회사분할합병결정(CG-003) — "분할 매출액"이라는 질문 표면과 정확히
    # 일치하는 leaf 라벨이 없다. 존속회사 절의 "존속사업부문 최근
    # 사업연도매출액(원)"이 그 값이다.
    "분할매출액": (
        "나. 분할 후 존속회사 > 존속사업부문 최근 사업연도매출액(원)",
    ),
    # 무상증자결정(CG-006) — "보통주식 (주)" leaf가 신주의 종류와 수·증자전
    # 발행주식총수·1주당 신주배정 주식수 세 절에 각각 다른 값으로 나온다.
    "신주배정비율": (
        "5. 1주당 신주배정 주식수 > 보통주식 (주)",
    ),
    # 해외증권시장주권등상장결정(CG-024) — "기타주식" leaf가 상장예정주식
    # 종류ㆍ수 절과 발행주식 총수 절에 각각 다른 값으로 나온다(상장폐지결정
    # 의 같은 구조는 이슈 #121이 이미 "1. 상장폐지주식 종류ㆍ수(주)"를 쓴다).
    "대상주식수": (
        "1. 상장예정주식 종류ㆍ수(주) > 기타주식",
    ),
    # 제3자의전환사채매수선택권행사(CG-025) — "금액(원)" leaf가 "7. 행사대가
    # (사채 매수금액)"·"10. 행사자 지정대가" 두 절에 나온다("행사자"는 여러
    # 행사인이 있어 그 자체가 정직하게 ambiguous다 — 별도 경로를 두지 않는다).
    "행사금액": (
        "7. 행사대가 (사채 매수금액) > 금액(원)",
    ),
}

# These slots can repeat once per explicitly named recipient/investor inside
# one disclosure.  Other event fields (for example a contract's global
# ``계약금액``) must not be forced under the counterparty path merely because
# the selected event also has a public counterparty facet.
ROW_SCOPED_SLOTS = frozenset({"권면총액", "발행권면총액", "발행금액"})

from app.textkit import STATE_KO as _STATE_KO, complete_source_excerpt


def _normalized_claim_surface(value: str | None) -> str:
    """Normalize formatting only; never erase a semantic token."""

    return " ".join((value or "").split())


def _claim_timepoint(output_id: str) -> str | None:
    """Return the explicit YYYYMMDD output coordinate, if one exists."""

    matched = re.search(r"@(\d{8})(?:$|[^0-9])", output_id)
    return matched.group(1) if matched else None


def _date_coordinate(value: str | None) -> tuple[str, str] | None:
    """Return a comparable YYYYMMDD date and its public ISO surface."""

    matched = re.search(
        r"(?<!\d)(?P<year>20\d{2})\s*(?:년|[-./])\s*"
        r"(?P<month>\d{1,2})\s*(?:월|[-./])\s*"
        r"(?P<day>\d{1,2})\s*(?:일)?(?!\d)",
        value or "",
    )
    if matched is None:
        return None
    year = int(matched.group("year"))
    month = int(matched.group("month"))
    day = int(matched.group("day"))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year:04d}{month:02d}{day:02d}", f"{year:04d}-{month:02d}-{day:02d}"


def _event_public_noun(timeline) -> str:
    """Name the disclosed event without calling every event a contract."""

    form = re.sub(r"\s+", "", str(getattr(timeline, "form", "") or ""))
    if "신규시설투자" in form:
        return "신규시설투자"
    if "타법인" in form and ("취득" in form or "처분" in form):
        return "타법인 투자"
    if "유상증자" in form or "사채" in form:
        return "자금조달 결정"
    return "계약"


def _active_observation_projection(
        *, timepoint: str, timeline, slot_claims: list[AnswerClaim],
        ) -> tuple[str, str, list[ClaimCitation]]:
    """Project canonical ``active`` as a bounded public observation.

    Canonical active means no termination observation was bound by the
    cutoff.  It does not prove that work is currently under way or that a
    contract remains legally valid.  Verified start/end dates can locate the
    cutoff in the disclosed period, but they still do not prove real-world
    completion.  Keep that distinction in the typed public claim.
    """

    dates: dict[str, tuple[str, str, AnswerClaim]] = {}
    for claim in slot_claims:
        if claim.label not in {"시작일", "종료일"}:
            continue
        coordinate = _date_coordinate(claim.text or claim.value_text)
        if coordinate is not None:
            dates[claim.label] = (*coordinate, claim)

    noun = _event_public_noun(timeline)
    start = dates.get("시작일")
    end = dates.get("종료일")
    if noun == "신규시설투자":
        if start is not None and timepoint < start[0]:
            return (
                "facility_before_start",
                f"공시상 투자 시작일({start[1]}) 이전입니다.",
                list(start[2].citations))
        # A passed planned end date alone does not prove completion.  Only
        # expose usage approval when the selected snapshot has its exact,
        # cited correction reason as well as the disclosed end date.
        approval = next((claim for claim in slot_claims
                         if claim.label.startswith("정정사유")
                         and claim.citations
                         and re.search(r"사용\s*승인에\s*따른\s*투자\s*종료일",
                                       claim.text or "")), None)
        if end is not None and timepoint >= end[0] and approval is not None:
            return (
                "facility_usage_approved",
                "건축물 사용승인에 따라 공시상 투자 종료일이 "
                f"{end[1]}로 정정되었습니다. 계약 해지를 뜻하지는 않습니다.",
                list(end[2].citations) + list(approval.citations))
    evidence_claim: AnswerClaim | None = None
    if start is not None and timepoint < start[0]:
        evidence_claim = start[2]
        text = (
            f"공시상 {noun} 시작일은 {start[1]}로 기준일보다 뒤입니다. "
            "기준일까지 해지·종료 공시는 확인되지 않았지만 아직 진행 중인 "
            "상태로 볼 수는 없습니다.")
    elif end is not None and timepoint >= end[0]:
        evidence_claim = end[2]
        text = (
            f"공시상 {noun} 종료일은 {end[1]}"
            f"{josa(end[1], '으로', '로')} 기준일까지 이미 지났습니다. "
            "이후 해지·종료 공시는 확인되지 않았지만 실제 진행 또는 완료 "
            "여부는 공시만으로 확정할 수 없습니다.")
    elif start is not None or end is not None:
        evidence_claim = end[2] if end is not None else start[2]
        bounds = []
        if start is not None:
            bounds.append(f"시작일 {start[1]}")
        if end is not None:
            bounds.append(f"종료일 {end[1]}")
        text = (
            f"공시상 {noun} 기간({', '.join(bounds)}) 안에 해당합니다. "
            "기준일까지 해지·종료 공시는 확인되지 않았지만 실제 진행 여부나 "
            "법적 유효성을 확정한 것은 아닙니다.")
    else:
        text = (
            f"선택된 {noun}의 공시 이력에서 기준일까지 해지·종료 공시는 "
            "확인되지 않았습니다. 이는 공시 확인 결과이며 실제 진행 여부나 "
            "법적 유효성을 확정한 것은 아닙니다.")

    citations = list(evidence_claim.citations) if evidence_claim is not None else []
    return "no_termination_observed", text, citations


def _same_timepoint_fact_key(claim: AnswerClaim) -> tuple | None:
    """Build a conservative key for duplicate public fact blocks.

    Only claims carrying an explicit timepoint are candidates. Source
    coordinates are part of the key, so equal-looking facts from two filings
    are not merged. Typed ``state`` and ``operator`` are included for the same
    reason: a correction delta must never disappear behind a state summary.
    """

    timepoint = _claim_timepoint(claim.output_id)
    if timepoint is None:
        return None
    sources = tuple(sorted((
        citation.doc_id,
        citation.rcept_no or "",
        citation.evidence_id or "",
        citation.section_id or "",
        citation.locator or "",
    ) for citation in claim.citations))
    return (
        timepoint,
        _normalized_claim_surface(claim.label),
        _normalized_claim_surface(claim.value_text),
        _normalized_claim_surface(claim.text),
        claim.raw_unit or "",
        claim.canonical_value or "",
        claim.canonical_unit or "",
        claim.state or "",
        claim.operator or "",
        sources,
    )


def _dedupe_same_timepoint_facts(
        claims: list[AnswerClaim]) -> list[AnswerClaim]:
    """Drop only byte-equivalent facts at the same timepoint and source."""

    seen: set[tuple] = set()
    result: list[AnswerClaim] = []
    for claim in claims:
        key = _same_timepoint_fact_key(claim)
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        result.append(claim)
    return result


def _cross_timepoint_history_key(claim: AnswerClaim) -> tuple | None:
    """Collapse only history/context facts repeated by a later snapshot.

    A multi-timepoint status lookup replays the correction history visible at
    each cutoff.  The same correction row can therefore be emitted twice even
    though its public label, value and exact evidence source are identical.
    These are filing-level facts, not separate state conclusions.  Requested
    state/termination claims and ordinary point-in-time fields are excluded so
    every temporal answer coordinate remains independently checkable.
    """

    if ".corr." not in claim.output_id and ".context." not in claim.output_id:
        return None
    sources = tuple(sorted((
        citation.doc_id,
        citation.rcept_no or "",
        citation.evidence_id or "",
        citation.section_id or "",
        citation.locator or "",
    ) for citation in claim.citations))
    return (
        _normalized_claim_surface(claim.label),
        _normalized_claim_surface(claim.value_text),
        _normalized_claim_surface(claim.text),
        claim.raw_unit or "",
        claim.canonical_value or "",
        claim.canonical_unit or "",
        claim.state or "",
        claim.operator or "",
        sources,
    )


def _dedupe_public_event_facts(
        claims: list[AnswerClaim]) -> list[AnswerClaim]:
    """Apply exact same-timepoint and safe cross-snapshot deduplication."""

    result = _dedupe_same_timepoint_facts(claims)
    seen_history: set[tuple] = set()
    deduped: list[AnswerClaim] = []
    for claim in result:
        key = _cross_timepoint_history_key(claim)
        if key is not None and key in seen_history:
            continue
        if key is not None:
            seen_history.add(key)
        deduped.append(claim)
    return deduped


def _is_verified_unchanged_placeholder(item) -> bool:
    """Whether typed correction evidence proves a meaningless ``- -> -``.

    A visual pair of dashes alone is insufficient. Both coordinates must be
    verified and canonical ``diff_kind`` must explicitly say ``same``. This
    keeps real placeholder-to-literal changes and unverified rows fail-closed.
    """

    return (
        getattr(item, "diff_kind", None) == "same"
        and getattr(item, "before_evidence_status", None) == "verified"
        and getattr(item, "after_evidence_status", None) == "verified"
        and (getattr(item, "value_before", None) or "").strip() == "-"
        and (getattr(item, "value_after", None) or "").strip() == "-"
    )


_CORRECTION_SIDE_REFERENCE = re.compile(
    r"(?:주\s*\d+\s*\)?\s*)?정정\s*(?P<side>전|후)"
)


def _is_reference_only_correction_pair(before: str, after: str) -> bool:
    """Whether before/after cells merely point to separately numbered notes."""

    left = _CORRECTION_SIDE_REFERENCE.fullmatch(before.strip())
    right = _CORRECTION_SIDE_REFERENCE.fullmatch(after.strip())
    return bool(left and right
                and left.group("side") == "전"
                and right.group("side") == "후")


#: 이슈 #112 — 정정 전/후 값이 통째로 표(예: 「6. 신주 발행가액」의 보통주식·
#: 기타주식 두 행)일 때 그 원문 마크다운 줄.
_TABLE_ROW_LINE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_RULE_CELL = re.compile(r":?-{2,}:?")


def _correction_table_rows(text: str) -> "list[list[str]] | None":
    """텍스트 전체가 마크다운 표(구분선 제외)면 데이터 행의 셀 목록, 아니면 ``None``.

    한 줄이라도 ``|`` 로 둘러싸인 표 행이 아니면 표가 아니다 — 산문 한 줄만
    표로 오인하지 않는다.
    """

    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return None
    rows: list[list[str]] = []
    for line in lines:
        if not _TABLE_ROW_LINE.match(line):
            return None
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(_TABLE_RULE_CELL.fullmatch(cell.replace(" ", "")) for cell in cells):
            continue  # 구분선(|---|---|)
        rows.append(cells)
    return rows or None


def _correction_table_cell_changes(before: str, after: str) -> "list[str] | None":
    """정정 전/후가 둘 다 순수 표면, 바뀐 셀만 ``식별: 전 → 후`` 줄로 돌려준다.

    표는 줄 단위 구조라 두 표를 한 줄로 이을 수 없다(#112) — 그 대신 같은
    자리(행·열)의 셀을 짝짓고, 값이 다른 자리만 「그 행의 나머지 셀(식별) :
    전 값 → 후 값」한 줄로 낸다. 행 수·열 수가 다르거나 한 행 안에서
    바뀐 자리가 둘 이상이면 짝을 확신할 수 없으므로 ``None`` — 호출자가
    두 표를 각각 세운다(안전망).
    """

    before_rows = _correction_table_rows(before)
    after_rows = _correction_table_rows(after)
    if before_rows is None or after_rows is None:
        return None
    if len(before_rows) != len(after_rows):
        return None
    changes: list[str] = []
    for b_row, a_row in zip(before_rows, after_rows):
        if len(b_row) != len(a_row) or len(b_row) < 2:
            return None
        diffs = [i for i, (b, a) in enumerate(zip(b_row, a_row)) if b != a]
        if not diffs:
            continue
        if len(diffs) != 1:
            transitions = {
                (b_row[index], a_row[index]) for index in diffs
            }
            # Some DART tables repeat the same value across several display
            # columns without exposing a usable column heading in the
            # correction fragment.  If every changed cell carries the exact
            # same transition, one public sentence preserves the value change
            # without dumping a nested Markdown table into another table.
            if len(transitions) != 1:
                return None
        index = diffs[0]
        identity = " ".join(
            cell for j, cell in enumerate(b_row)
            if j != index and cell and cell != "-")
        if not identity:
            return None
        changes.append(f"{identity}: {b_row[index]} → {a_row[index]}")
    return changes or None


def _correction_table_change_text(before: str, after: str) -> "str | None":
    """정정 전/후 표 값의 화면용 본문 — 바뀐 셀 문장, 못 짝지으면 표 두 개.

    바뀐 셀만 문장으로 짝지을 수 있으면 그 문장(줄마다 하나)을 돌려준다.
    짝짓지 못하면(행/열 수가 다르거나 한 행에서 여러 칸이 바뀜) 정정 전·
    후 표를 각각 온전히 세워 돌려준다 — 어느 쪽도 줄 중간에서 자르거나
    한 줄로 이어붙이지 않는다.
    """

    changes = _correction_table_cell_changes(before, after)
    if changes is not None:
        return "\n".join(changes)
    if _correction_table_rows(before) is None or _correction_table_rows(after) is None:
        return None
    return f"정정 전:\n{before}\n\n정정 후:\n{after}"


def _public_field_text(row) -> str:
    """Render a verified scalar without turning every dash into withholding."""

    value = row.value or ""
    if value.strip() == "-" and row.value_status == "not_reported":
        return "공시에서 공개되지 않음"
    return value


def _requests_intraday_order(task) -> bool:
    """Whether typed requested slots ask for within-day order/time."""

    compact = [re.sub(r"\s+", "", str(value))
               for value in tuple(getattr(task, "requested_slots", ()) or ())]
    return any(any(token in value for token in (
        "분단위", "접수시각", "접수시간", "시각", "시간", "순서",
    )) for value in compact)


def _intraday_order_limitation(task, timelines) -> Limitation | None:
    """Expose missing intraday timestamps without receipt-order guessing."""

    if not _requests_intraday_order(task):
        return None
    affected: set[str] = set()
    seen_events: set[tuple[str, str]] = set()
    for _origin, timepoint, timeline in timelines:
        identity = (str(getattr(timeline, "event_key", "")), str(timepoint))
        if identity in seen_events:
            continue
        seen_events.add(identity)
        by_day: dict[str, list[object]] = {}
        for observation in tuple(getattr(timeline, "observations", ()) or ()):
            by_day.setdefault(
                str(getattr(observation, "observed_at", "")), []).append(
                    observation)
        for rows in by_day.values():
            if len(rows) < 2:
                continue
            affected.update(
                str(getattr(row, "rcept_no", "")) for row in rows)
    affected.discard("")
    if not affected:
        return None
    return Limitation(
        code="intraday_order_unavailable",
        detail=("같은 날 공시의 논리적 계보는 관계 근거로 확인하지만 "
                "분 단위 접수 시각은 정본에 없어 접수번호 순서로 추정하지 않음"),
        affected_doc_ids=sorted(affected),
    )


def _path_segments(surface: str | None) -> tuple[str, ...]:
    """Normalize a human/canonical field path without erasing Korean words.

    Exchange correction items use quoted ``'A'의 'B'`` while QueryPlan slots
    use ``A > B``.  Treat the connector only as a segment boundary; deleting
    every ``의`` character would collide with legitimate labels such as
    ``자금조달의 목적``.
    """

    text = (surface or "").strip()
    quoted = re.findall(r"['‘’\"]([^'‘’\"]+)['‘’\"]", text)
    raw = quoted if len(quoted) >= 2 else re.split(r"\s*>\s*", text)
    return tuple(
        value for value in (
            re.sub(r"[^0-9a-z가-힣%]+", "", part.casefold())
            for part in raw)
        if value)


def _surface_unit(surface: str | None) -> str | None:
    """Infer only an explicitly labelled scalar unit from a field surface."""

    matches = re.findall(r"\((원|천원|백만원|십억원|억원|조원|%|주)\)", surface or "")
    return matches[-1] if matches else None


def _decimal_scalar(value: str | None) -> Decimal | None:
    if value is None:
        return None
    normalized = re.sub(r"\s*(?:원|KRW)\s*$", "", value.strip(),
                        flags=re.IGNORECASE).replace(",", "")
    if re.fullmatch(r"[-+]?[0-9]+(?:\.[0-9]+)?", normalized) is None:
        return None
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def _correction_scalar_projection(
        item, *, before_verified: bool = True,
        after_verified: bool = True) -> dict[str, str | None]:
    """Project a verified correction leaf into typed scalar claim fields.

    Correction rows keep their exact source strings for audit.  When the path
    itself declares a unit and the after-value is numeric, also attach the
    typed value/unit (and exact won value for money).  This lets the public
    renderer show ``72,200,000,000원`` as ``722억원`` without changing the
    canonical correction item or asking the language model to do arithmetic.
    """

    after = getattr(item, "value_after", None) if after_verified else None
    value_text = after if after and _looks_numeric(after) else None
    raw_unit = _surface_unit(getattr(item, "path", None))
    state = None
    # Only a literal dash proves that this scalar was not disclosed in the
    # correction table.  Other placeholders such as ``(주1)`` point elsewhere
    # and must not be reworded as "비공개" merely because a later value is
    # numeric.
    # ``-`` is also used for non-monetary fields such as a newly disclosed
    # contract date.  The public renderer's ``disclosed_from_withheld`` state
    # is an amount-specific presentation contract, so only attach it when the
    # path proves a supported money/ratio scalar and the after-value is
    # actually numeric.  Otherwise keep the verified raw transition intact.
    if (before_verified and after_verified
            and (getattr(item, "value_before", None) or "").strip() == "-"
            and getattr(item, "after_kind", None) == "literal"
            and value_text is not None
            and raw_unit in {"원", "천원", "백만원", "십억원", "억원", "조원", "%"}):
        state = "disclosed_from_withheld"

    canonical_value = None
    canonical_unit = None
    decimal = _decimal_scalar(value_text)
    if decimal is not None and raw_unit not in {None, "%", "주"}:
        won = to_won(decimal, raw_unit)
        if won is not None and won == won.to_integral_value():
            canonical_value = format(won, "f")
            canonical_unit = "원"
    return {
        "value_text": value_text,
        "raw_unit": raw_unit,
        "canonical_value": canonical_value,
        "canonical_unit": canonical_unit,
        "state": state,
    }


def _cite_field(f) -> ClaimCitation:
    return ClaimCitation(doc_id=f.doc_id, rcept_no=f.rcept_no, evidence_id=f.evidence_id,
                         locator=f.locator, excerpt_prompt_safe=f.value_prompt_safe or f.value or "")


@dataclass
class EventCandidate:
    rcept_no: str
    rcept_dt: str
    form: str
    counterparty: str | None
    contract_name: str | None = None
    event_key: str | None = None


@dataclass(frozen=True)
class EventStateSnapshot:
    """Execution-only, citation-bound projection of one event timepoint.

    The public QueryPlan stays at v0.4.  This sidecar closes a runtime gap:
    ``status`` alone is not a complete answer when the state-defining
    disclosure also proves withheld terms, termination details, or an
    observable correction history.  The profile is selected only from typed
    state/operation metadata and never from a question ID or issuer name.
    """

    status: str
    state_rcept_no: str | None
    slots: tuple[str, ...]
    correction_receipts: tuple[str, ...]
    observation_receipts: tuple[str, ...]
    include_correction_history: bool

    @classmethod
    def from_timeline(cls, timeline, *, operation: str,
                      requested_slots) -> "EventStateSnapshot":
        observations = tuple(timeline.observations or ())
        state_receipt = timeline.state.last_rcept_no
        if timeline.state.status == "terminated":
            terminations = tuple(
                row for row in observations
                if getattr(row, "is_termination", False))
            if terminations:
                state_receipt = terminations[-1].rcept_no

        observed_receipts = tuple(dict.fromkeys(
            row.rcept_no for row in observations))
        root_outside_observations = bool(
            timeline.root_rcept_no
            and timeline.root_rcept_no not in observed_receipts)
        first_is_correction = bool(
            observations and getattr(observations[0], "is_correction", False))
        include_history = (
            operation == "timeline"
            or first_is_correction
            or root_outside_observations)
        correction_receipts = tuple(dict.fromkeys(
            row.rcept_no for row in observations
            if getattr(row, "is_correction", False)))
        if not include_history:
            state_observation = next((
                row for row in observations
                if row.rcept_no == state_receipt), None)
            correction_receipts = (
                (state_receipt,)
                if state_observation is not None
                and getattr(state_observation, "is_correction", False)
                and state_receipt is not None
                else ())

        slots = list(requested_slots or ())
        if include_history:
            slots.extend((
                "계약금액", "최근매출액", "매출액대비", "계약상대",
                "계약(수주)일자", "시작일", "종료일"))
        if timeline.state.status == "active":
            slots.extend((
                "계약상대", "계약금액", "유보사유", "유보기한",
                "효력발생조건"))
        elif timeline.state.status == "terminated":
            slots.extend((
                "계약상대", "해지금액", "매출액대비", "해지사유",
                "해지일자", "해지계약명", "효력발생조건"))
        return cls(
            status=timeline.state.status,
            state_rcept_no=state_receipt,
            slots=tuple(dict.fromkeys(slots)),
            correction_receipts=correction_receipts,
            observation_receipts=observed_receipts,
            include_correction_history=include_history,
        )


class _SelectedEventProvenanceError(RuntimeError):
    """A selected event key did not round-trip to the requested issuer."""


class BaseTool:
    def __init__(self, rm, fidx=None):
        self.rm = rm
        from app.tools.field_index import FieldIndex
        self.fidx = fidx or FieldIndex(rm)

    #: 질문이 지분공시를 가리키는 표면. 이때만 대량보유상황보고서를 후보에 둔다.
    _HOLDING_SURFACES = ("지분", "대량보유", "보유상황", "주식등의", "의결권")

    # ── 사건/문서 후보 찾기 ─────────────────────────────────────────────
    def find_candidates(self, corp_code: str, sel, *, as_of: str,
                        forms: tuple[str, ...] | None = None) -> list[EventCandidate]:
        """selector로 비정기공시 문서 후보를 찾는다. seed_rcept_no가 있으면 그것 하나."""
        if getattr(sel, "seed_rcept_no", None):
            rows = list(self.fidx.rows(corp_code, as_of=as_of, rcept_no=sel.seed_rcept_no))
            if not rows:
                return []
            cp = next((r.value for r in rows if "계약상대" in r.label), None)
            return [EventCandidate(rows[0].rcept_no, rows[0].rcept_dt, rows[0].form, cp)]

        cp = getattr(sel, "counterparty", None)
        cn = getattr(sel, "contract_name", None)
        kws = list(getattr(sel, "keywords", []) or [])
        et = getattr(sel, "event_type", None)
        efrom = getattr(sel, "event_from", None)
        eto = getattr(sel, "event_to", None)

        DOC_GROUP_ALIAS = {"주요사항보고": "major", "주요사항보고서": "major", "거래소공시": "exchange",
                           "수시공시": "exchange", "지분공시": "holding", "정기공시": "periodic"}
        dg = DOC_GROUP_ALIAS.get(et or "")
        # When the supplied event type is an exact canonical form, preserve
        # that typed constraint through FieldIndex.  It lets the index scan the
        # matching form once rather than opening fields.parquet once per
        # candidate receipt.  Substring/event-type selectors retain the older
        # mixed-form path below.
        exact_form = None
        documents = getattr(self.rm, "documents", None)
        if et and not dg and callable(documents):
            try:
                if any(documents(as_of=as_of, corp_code=corp_code, form=et)):
                    exact_form = et
            except Exception:
                # Candidate lookup remains fail-closed at its caller.  Do not
                # silently reinterpret a selector if metadata is unavailable.
                raise
        out: list[EventCandidate] = []
        # 회사 전체 Field를 dict로 만들지 않는다. Document metadata로 receipt를 먼저
        # 정한 뒤 한 문서씩 열어, FieldIndex의 weighted receipt cache 상한을 지킨다.
        # 종류·날짜 조건은 Document metadata 만으로 판정된다. Field 를 읽고
        # 나서 거르면 버릴 문서의 행까지 다 읽는다 — 셀트리온은 문서 200건의
        # Field 295,107행을 읽고 그중 290,332행(대량보유상황보고서)을 버렸다.
        # 아래 루프의 판정과 같은 조건을, 같은 값이 실려 있는 metadata 로 먼저
        # 본다. event_type 은 metadata 에 없으므로 그때만 통과시킨다.
        # 지분공시는 사건 공시가 아니다. 코퍼스 Field 의 94%(셀트리온은 자기
        # 문서의 98%)가 대량보유상황보고서인데, 그 본문에 「계약」 같은 낱말이
        # 들어 있다는 이유로 후보에 오르고 그 행을 전부 읽는다. 전수 395건에서
        # 지분공시를 인용한 답변은 0건이다 — 훑기만 하고 쓰이지 않는다.
        #
        # 질문이 지분을 가리키면 그때는 포함한다. 판단은 선택자가 이미 들고
        # 있는 값으로만 한다 — 문서군을 지정했거나(dg) 키워드가 지분을 말할 때.
        holding_asked = dg == "holding" or any(
            any(mark in keyword for mark in self._HOLDING_SURFACES)
            for keyword in kws)

        def _keep(meta) -> bool:
            if forms and meta.form not in forms:
                return False
            if dg and meta.doc_group != dg:
                return False
            if not holding_asked and meta.doc_group == "holding":
                return False
            if efrom and meta.rcept_dt < efrom:
                return False
            if eto and meta.rcept_dt > eto:
                return False
            return True

        for rc, rows in self.fidx.iter_docs(
                corp_code, as_of=as_of, form=exact_form, keep=_keep):
            ensure_request_time_remaining()
            r0 = rows[0]
            if forms and r0.form not in forms: continue
            if dg:
                if r0.doc_group != dg: continue
            elif et and et not in (r0.form or "") and et != (r0.event_type or ""): continue
            if efrom and r0.rcept_dt < efrom: continue
            if eto and r0.rcept_dt > eto: continue
            party = next((r.value for r in rows if "계약상대" in r.label), None)
            hay = (" ".join((r.value or "") for r in rows) + " " + (r0.form or "") + " " + (r0.report_nm or "")).lower()
            name = next((r.value for r in rows if any(k in r.label for k in ("계약명", "계약 내용", "계약내용", "구분"))), None)
            if cp and (not party or cp.lower() not in party.lower()): continue
            if cn and cn.lower() not in hay: continue
            if kws:
                need_all = bool(cp or cn)
                ok = all(k.lower() in hay for k in kws) if need_all else any(k.lower() in hay for k in kws)
                if not ok: continue
            out.append(EventCandidate(rc, r0.rcept_dt, r0.form, party, name))
        return sorted(out, key=lambda c: (c.rcept_dt, c.rcept_no))

    def find_unknown_counterparty_candidates(
            self, corp_code: str, sel, *, as_of: str,
            ) -> list[EventCandidate]:
        """Return candidates whose non-identity predicates match but party is unknown.

        A counterparty filter is three-valued: a withheld/missing party is not
        evidence that the document belongs to the requested party, but it is
        also not evidence that it does not.  This side path is used only by a
        disclosure-list result to expose that completeness boundary; it never
        promotes an unknown document to a positive event match.
        """

        counterparty = getattr(sel, "counterparty", None)
        if not isinstance(counterparty, str) or not counterparty.strip():
            return []
        model_copy = getattr(sel, "model_copy", None)
        if not callable(model_copy):
            return []
        broad = model_copy(update={"counterparty": None})
        keywords = [str(value).casefold() for value in (
            getattr(sel, "keywords", ()) or ()) if str(value).strip()]
        unknown_statuses = {"empty", "not_reported", "parse_error"}
        out: list[EventCandidate] = []
        for candidate in self.find_candidates(corp_code, broad, as_of=as_of):
            ensure_request_time_remaining()
            rows = list(self.fidx.rows(
                corp_code, as_of=as_of, rcept_no=candidate.rcept_no))
            if not rows:
                continue
            party_rows = [row for row in rows if "계약상대" in row.label]
            party_unknown = not party_rows or all(
                getattr(row, "value_status", None) in unknown_statuses
                or not (row.value or "").strip()
                or (row.value or "").strip() == "-"
                for row in party_rows)
            if not party_unknown:
                continue
            haystack = " ".join((row.value or "") for row in rows).casefold()
            if keywords and not all(keyword in haystack for keyword in keywords):
                continue
            out.append(EventCandidate(
                candidate.rcept_no, candidate.rcept_dt, candidate.form,
                None, candidate.contract_name, candidate.event_key))
        return sorted(out, key=lambda value: (value.rcept_dt, value.rcept_no))

    # ── slot 값 조회 (Field) ────────────────────────────────────────────
    def field_value(
            self, corp_code: str, rcept_no: str, slot: str, *, as_of: str,
            trace, row_scope: str | None = None):
        """slot 하나를 Field에서 찾는다. 반환 (FieldRow|None, status). 확정 시 Evidence 재검증 포함."""
        # Multi-party forms repeat the same leaf once per named recipient.
        # When Stage1 preserved an exact, canonical-proofed public
        # counterparty, bind the field to that full path segment before any
        # global leaf lookup. Missing or duplicate scoped rows fail closed;
        # they must never fall back to a different party's value.
        if row_scope:
            scope_segments = _path_segments(row_scope)
            labels = SLOT_LABELS.get(slot, [slot])
            label_keys = {
                _path_segments(label)[-1]
                for label in labels if _path_segments(label)
            }
            scoped = []
            for row in self.fidx.rows(
                    corp_code, as_of=as_of, rcept_no=rcept_no):
                path = _path_segments(row.path)
                label = _path_segments(row.label)
                if (not scope_segments or not label
                        or scope_segments[0] not in path[:-1]):
                    continue
                if any(key in label[-1] or label[-1] in key
                       for key in label_keys):
                    scoped.append(row)
            values = {row.value for row in scoped}
            if len(scoped) == 1 or (scoped and len(values) == 1):
                row = self.fidx.verified(min(
                    scoped,
                    key=lambda value: (value.occurrence, value.order)))
                trace.append(TraceEvent(
                    seq=len(trace)+1, stage="tool",
                    summary=(f"lookup_field {rcept_no} row '{row_scope}' "
                             f"'{slot}' → ok/{row.evidence_status}"),
                    detail={"slot": slot, "row_scope": row_scope}))
                return row, (
                    "verified" if row.evidence_status == "verified"
                    else "unverified")
            trace.append(TraceEvent(
                seq=len(trace)+1, stage="tool",
                summary=(f"lookup_field {rcept_no} row '{row_scope}' "
                         f"'{slot}' → "
                         f"{'ambiguous' if scoped else 'not_found'}"),
                detail={"slot": slot, "row_scope": row_scope}))
            return None, "ambiguous" if scoped else "not_found"
        # Full semantic paths are stronger than their leaf label.  Resolve
        # those first so e.g. two different ``보통주식(원)`` fields do not become
        # an artificial ambiguity.
        registered_paths = SLOT_CANONICAL_PATHS.get(slot)
        path_surfaces = registered_paths or (slot,)
        wanted_paths = tuple(_path_segments(surface)
                             for surface in path_surfaces)
        for wanted_path in wanted_paths:
            if len(wanted_path) <= 1 and registered_paths is None:
                continue
            cands = [row for row in self.fidx.rows(
                corp_code, as_of=as_of, rcept_no=rcept_no)
                if _path_segments(row.path) == wanted_path]
            if cands:
                values = {row.value for row in cands}
                if len(values) == 1:
                    row = self.fidx.verified(sorted(
                        cands, key=lambda value: (value.occurrence, value.order))[0])
                    trace.append(TraceEvent(
                        seq=len(trace)+1, stage="tool",
                        summary=f"lookup_field {rcept_no} full-path '{slot}' → ok/{row.evidence_status}",
                        detail={"slot": slot}))
                    return row, ("verified" if row.evidence_status == "verified" else "unverified")
                trace.append(TraceEvent(
                    seq=len(trace)+1, stage="tool",
                    summary=f"lookup_field {rcept_no} full-path '{slot}' → ambiguous ({len(cands)} 후보)",
                    detail={"slot": slot}))
                return None, "ambiguous"
        labels = SLOT_LABELS.get(slot, [slot])
        path_leaves = [path[-1] for path in wanted_paths if len(path) > 1]
        if path_leaves:
            labels = [*path_leaves, *labels]
        for lb in labels:
            st, row, cands = self.fidx.lookup(corp_code, rcept_no, lb, as_of=as_of)
            if st == "ok" and row is not None:
                trace.append(TraceEvent(seq=len(trace)+1, stage="tool",
                                        summary=f"lookup_field {rcept_no} '{lb}' → ok/{row.evidence_status}",
                                        detail={"slot": slot}))
                return row, ("verified" if row.evidence_status == "verified" else "unverified")
            if st == "ambiguous":
                trace.append(TraceEvent(seq=len(trace)+1, stage="tool",
                                        summary=f"lookup_field {rcept_no} '{lb}' → ambiguous ({len(cands)} 후보)",
                                        detail={"slot": slot}))
                return None, "ambiguous"
        return None, "not_found"


    def _is_correction(self, corp_code: str, rcept_no: str, as_of: str) -> bool:
        # 정정 여부는 Document metadata 권위값이다. Field cache 내부 구조를 의미
        # 판정에 사용하면 receipt eviction에 따라 답이 달라진다.
        loader = getattr(self.rm, "_load_docs", None)
        if callable(loader):
            loader()
            doc_id = getattr(self.rm, "_doc_id_by_rcept", {}).get(rcept_no)
            meta = getattr(self.rm, "_documents", {}).get(doc_id)
            if meta is not None:
                return bool(meta.is_correction)
        rows = list(self.fidx.rows(
            corp_code, as_of=as_of, rcept_no=rcept_no))
        return bool(rows) and rows[0].is_correction

    def _doc_cites(self, corp_code, rcept_no, as_of):
        """문서를 인용할 대표 Field(계약상대 → 계약명 → 정정사유 → 첫 verified)."""
        for lb in ("계약상대", "계약명", "정정사유"):
            st, row, cands = self.fidx.lookup(corp_code, rcept_no, lb, as_of=as_of)
            if row is None and cands:
                row = self.fidx.verified(sorted(cands, key=lambda c: c.occurrence)[0])
            if row is not None and row.evidence_status == "verified":
                return [_cite_field(row)]
        for f in self.fidx.rows(corp_code, as_of=as_of, rcept_no=rcept_no):
            v = self.fidx.verified(f)
            if v.evidence_status == "verified":
                return [_cite_field(v)]
        # Field가 없는 문서(정기공시 등): 문서 존재 자체를 근거로 (documents 메타, source_roundtrip)
        self.rm._load_docs()
        doc_id = self.rm._doc_id_by_rcept.get(rcept_no)
        if doc_id:
            return [ClaimCitation(doc_id=doc_id, rcept_no=rcept_no, locator="document",
                                  excerpt_prompt_safe=f"{doc_id}", verification="source_roundtrip")]
        return []

    def _exact_field_cites(
            self, corp_code: str, rcept_no: str, label: str,
            expected: str, *, as_of: str) -> list[ClaimCitation]:
        """Cite a verified field only when its value closes exactly.

        Document-level representative citations are valid for proving that a
        filing exists, but not for binding a public reason/value to its source
        cell.  This helper deliberately does not pick the first ambiguous
        candidate: a scalar claim is emitted only when the field lookup and
        value both resolve exactly.
        """

        status, row, _candidates = self.fidx.lookup(
            corp_code, rcept_no, label, as_of=as_of)
        if (status != "ok" or row is None
                or row.evidence_status != "verified"):
            return []
        actual = " ".join(str(row.value or "").split())
        wanted = " ".join(str(expected or "").split())
        return [_cite_field(row)] if actual == wanted else []

    def _named_contract_event_key(self, corp_code, selector, *, as_of: str):
        """Resolve a named contract to one canonical event before expansion.

        A contract name is an explicit identity constraint, so candidate Field
        values are streamed once and then round-tripped through the canonical
        Event identity.  This avoids reopening a company's Field artifact for
        every document merely to discover that all hits belong to one lineage.
        ``None`` means the fast Field route found no candidate and callers may
        retain the ordinary selector path; every non-``None`` failure is typed
        and must not fall back to an issuer-wide collection.
        """
        contract_name = getattr(selector, "contract_name", None)
        fields = getattr(self.rm, "fields", None)
        if not isinstance(contract_name, str) or not contract_name.strip() or not callable(fields):
            return None
        wanted = contract_name.casefold()
        event_type = getattr(selector, "event_type", None)
        event_from = getattr(selector, "event_from", None)
        event_to = getattr(selector, "event_to", None)
        receipts: set[str] = set()
        try:
            for row in fields(as_of=as_of, corp_code=corp_code):
                receipt = getattr(row, "rcept_no", None)
                receipt_date = getattr(row, "rcept_dt", "")
                if not isinstance(receipt, str) or not isinstance(receipt_date, str):
                    continue
                if event_from and receipt_date < event_from:
                    continue
                if event_to and receipt_date > event_to:
                    continue
                form = getattr(row, "form", "") or ""
                row_event_type = getattr(row, "event_type", "") or ""
                if event_type and event_type not in form and event_type != row_event_type:
                    continue
                value = getattr(row, "value", None)
                if isinstance(value, str) and wanted in value.casefold():
                    receipts.add(receipt)
        except Exception:
            return "named_contract_candidate_lookup_error"
        if not receipts:
            return None
        event_keys: set[str] = set()
        try:
            for receipt in sorted(receipts):
                timeline = self.rm.event_timeline(
                    as_of=as_of, rcept_no=receipt, verify_evidence=False)
                if timeline is None or getattr(timeline, "corp_code", None) != corp_code:
                    return "named_contract_event_owner_mismatch"
                event_key = getattr(timeline, "event_key", None)
                if not isinstance(event_key, str) or not event_key:
                    return "named_contract_event_key_missing"
                event_keys.add(event_key)
        except Exception:
            return "named_contract_event_lookup_error"
        if len(event_keys) != 1:
            return "named_contract_maps_to_multiple_events"
        return next(iter(event_keys))


class EventTool(BaseTool):
    _ORIGIN_DISCRIMINATOR_GROUPS = (("시작일", "종료일"),)

    @staticmethod
    def _contract_identity_sentence(value: str) -> str | None:
        """Return the first labelled contract-identification sentence.

        The structured exchange forms commonly place the counterparty and
        supply quantity in the first ``가.`` clause of ``기타 투자판단``.
        It is useful only as an *already cited* discriminator alongside the
        exact period tuple; this helper never searches another document or
        manufactures a quantity.  If that bounded clause is not present, the
        caller simply omits it.
        """

        text = " ".join((value or "").split())
        if not text:
            return None
        match = re.search(r"(?:^|\s)가\.\s*(.+?)(?=\s+나\.|$)", text)
        return match.group(1).strip() if match else None

    def _append_listed_origin_discriminators(
            self, task, *, timelines, candidate_receipts,
            claims, trace) -> None:
        """Expose source fields that distinguish listed sibling contracts.

        A collection list can contain every original receipt in an ambiguous
        identity group.  A termination observation may have an exact,
        Evidence-verified period match to one root, but that match is only a
        candidate discriminator and never a canonical lineage edge.  Project
        the same source discriminators for *all listed roots*: period is always
        exact, and a first labelled context clause is included only when the
        source has it.

        This is deliberately gated on ``candidate_roots <= candidate_receipts``.
        A query for termination notices is not widened into a list of original
        contracts, and no lexical company/receipt/date special case is used.
        """

        emitted_root_groups: set[tuple[str, ...]] = set()
        as_of = task.timepoints[-1]
        for timeline in timelines:
            event_key = timeline.event_key
            # Period/quantity discriminators are relevant only when a later
            # termination exists but cannot be assigned to one of the sibling
            # originals.  Emitting them for an ordinary ambiguous list would
            # look like an attempted identity resolution even though there is
            # no follow-up observation to resolve.
            if (timeline.identity_status != "ambiguous"
                    or not any(
                        getattr(observation, "is_termination", False)
                        for observation in timeline.observations)):
                continue
            roots = tuple(sorted(set(self.rm.event_identity_candidate_roots(
                event_key=event_key, as_of=as_of))))
            if (len(roots) < 2
                    or not set(roots).issubset(candidate_receipts)
                    or roots in emitted_root_groups):
                continue
            emitted_root_groups.add(roots)
            for root in roots:
                rows = []
                for slot in self._ORIGIN_DISCRIMINATOR_GROUPS[0]:
                    row, status = self.field_value(
                        task.corp_code, root, slot, as_of=as_of,
                        trace=trace)
                    if (row is None or status != "verified" or not row.value
                            or row.value.strip() == "-"):
                        rows = []
                        break
                    rows.append(row)
                if rows:
                    claims.append(AnswerClaim(
                        output_id=f"{task.task_id}.{root}.contract_period",
                        label=f"{root} 계약기간",
                        text=f"{rows[0].value} ~ {rows[1].value}",
                        citations=[_cite_field(row) for row in rows]))

                context, status = self.field_value(
                    task.corp_code, root, "효력발생조건", as_of=as_of,
                    trace=trace)
                if (context is None or status != "verified"
                        or not context.value):
                    continue
                sentence = self._contract_identity_sentence(context.value)
                if sentence:
                    claims.append(AnswerClaim(
                        output_id=(f"{task.task_id}.{root}."
                                   "contract_identifier"),
                        label=f"{root} 계약 식별사항",
                        text=sentence, citations=[_cite_field(context)]))

    def _selected_timeline(self, task, event_key: str, as_of: str):
        """Read one already-resolved event without falling back to text candidates.

        ``event_key`` is Stage1's selected-event coordinate.  Treating it as a
        ranking hint would allow unrelated Field hits to replace that choice.
        The reader additionally validates the event artifacts when evidence is
        requested; this method binds the resulting timeline to the task issuer.
        """
        tl = self.rm.event_timeline(as_of=as_of, event_key=event_key,
                                    verify_evidence=True)
        if tl is None:
            return None
        if tl.event_key != event_key or tl.corp_code != task.corp_code:
            raise _SelectedEventProvenanceError(
                f"selected event ownership mismatch: key={event_key} "
                f"returned=({tl.event_key}, {tl.corp_code}) expected={task.corp_code}")
        return tl

    def run_task(self, task, *, trace, corp_name: str):
        ensure_request_time_remaining()
        claims, lims, used, clar = [], [], [], None
        selected_key = getattr(task.selector, "event_key", None)
        if not selected_key:
            binding = self._named_contract_event_key(
                task.corp_code, task.selector, as_of=max(task.timepoints))
            if isinstance(binding, str) and len(binding) == 32 and all(
                    char in "0123456789abcdef" for char in binding):
                selected_key = binding
            elif isinstance(binding, str):
                code = ("ambiguous_event_origin"
                        if binding == "named_contract_maps_to_multiple_events"
                        else "event_lookup_error")
                lims.append(Limitation(code=code, detail=binding))
                return claims, lims, used, clar
        if selected_key:
            try:
                selected = self._selected_timeline(task, selected_key, max(task.timepoints))
            except _SelectedEventProvenanceError as e:
                lims.append(Limitation(code="event_identity_mismatch", detail=str(e)))
                return claims, lims, used, clar
            except Exception as e:
                lims.append(Limitation(code="event_lookup_error", detail=str(e)[:150]))
                return claims, lims, used, clar
            if selected is None:
                lims.append(Limitation(code="not_found", detail="선택된 사건은 기준시점까지 관측되지 않음"))
                return claims, lims, used, clar
            first = selected.observations[0] if selected.observations else None
            if first is None:
                lims.append(Limitation(code="event_identity_mismatch",
                                       detail="선택된 사건에 관측 문서가 없어 provenance를 확인할 수 없음"))
                return claims, lims, used, clar
            # A typed named-field request can select one exact observation
            # (original, correction, or termination) within the canonical
            # event.  Read only that filing.  Falling through to the ordinary
            # status snapshot would prefer the latest/state-defining document
            # and could silently answer a different receipt than the user
            # named.  The membership check keeps the seed from becoming an
            # unchecked document lookup.
            seed_receipt = getattr(task.selector, "seed_rcept_no", None)
            if seed_receipt and tuple(getattr(task, "field_outputs", ()) or ()):
                observations = tuple(selected.observations or ())
                selected_observation = next((
                    row for row in observations
                    if row.rcept_no == seed_receipt), None)
                if selected_observation is None:
                    lims.append(Limitation(
                        code="event_identity_mismatch",
                        detail=(f"선택 공시 {seed_receipt}가 canonical event "
                                f"{selected_key}의 관측에 포함되지 않음"),
                        affected_doc_ids=[seed_receipt]))
                    return claims, lims, used, clar
                if selected_observation.observed_at > max(task.timepoints):
                    lims.append(Limitation(
                        code="not_found",
                        detail=(f"선택 공시 {seed_receipt}는 기준시점 이후 관측됨"),
                        affected_doc_ids=[seed_receipt]))
                    return claims, lims, used, clar
                output_by_slot = {
                    output.slot: output.output_id
                    for output in task.field_outputs}
                for slot in task.requested_slots:
                    row, status = self.field_value(
                        task.corp_code, seed_receipt, slot,
                        as_of=max(task.timepoints), trace=trace,
                        row_scope=(
                            getattr(task.selector, "counterparty", None)
                            if slot in ROW_SCOPED_SLOTS else None))
                    if row is None or status != "verified" or row.value is None:
                        lims.append(Limitation(
                            code=("ambiguous_field" if status == "ambiguous"
                                  else "evidence_unavailable"),
                            detail=f"{seed_receipt} {slot}: {status}",
                            affected_doc_ids=[seed_receipt]))
                        continue
                    if ("[REDACTED:" in _public_field_text(row)
                            and not any(lim.code == "requested_field_redacted"
                                        for lim in lims)):
                        # Keep the separately verified public fields in the
                        # answer, but state that this requested field was
                        # privacy-projected rather than silently treating it
                        # as an ordinary missing value.
                        lims.append(Limitation(
                            code="requested_field_redacted",
                            detail=f"{seed_receipt} {slot}: privacy-projected",
                            affected_doc_ids=[seed_receipt]))
                    claims.append(AnswerClaim(
                        output_id=(output_by_slot.get(slot)
                                   or f"{task.task_id}.{seed_receipt}.{slot}"),
                        label=f"{seed_receipt} {slot}",
                        value_text=(row.value
                                    if _looks_numeric(row.value) else None),
                        raw_unit=_surface_unit(row.path),
                        state=("explicit_zero"
                               if getattr(row, "value_status", None)
                               == "explicit_zero" else None),
                        text=_public_field_text(row),
                        citations=[_cite_field(row)]))
                used.append(seed_receipt)
                if not claims and not lims:
                    lims.append(Limitation(
                        code="not_found",
                        detail=f"선택 공시 {seed_receipt}에서 요청 필드를 찾지 못함",
                        affected_doc_ids=[seed_receipt]))
                return claims, lims, used, clar
            cands = [EventCandidate(selected.root_rcept_no or first.rcept_no,
                                    first.observed_at, selected.doc_group, None,
                                    event_key=selected_key)]
        else:
            cands = self.find_candidates(task.corp_code, task.selector, as_of=max(task.timepoints))
        ensure_request_time_remaining()
        trace.append(TraceEvent(seq=len(trace)+1, stage="tool",
                                # 선택자가 무엇으로 좁혔는지가 후보 수를 설명한다.
                                # 셀트리온이 문서 200건을 다 훑던 것도 여기에
                                # 종류 조건이 없고 키워드뿐인 것이 단서였다.
                                summary=(
                                    f"event candidates for {corp_name} "
                                    f"{task.selector.counterparty or task.selector.contract_name or task.selector.seed_rcept_no}"
                                    f" [type={getattr(task.selector, 'event_type', None)}"
                                    f" kws={list(getattr(task.selector, 'keywords', ()) or ())[:4]}]"
                                    f": {len(cands)}"),
                                detail={"rcept_nos": [c.rcept_no for c in cands],
                                        "event_key": selected_key}))
        if not cands:
            lims.append(Limitation(code="not_found", detail=f"{corp_name}의 해당 사건 공시를 지원 범위에서 찾지 못함"))
            return claims, lims, used, clar

        # 원본(최초 체결) 문서들: 해지·정정 아닌 것
        origins = [c for c in cands if "해지" not in c.form
                   and not self._is_correction(task.corp_code, c.rcept_no,
                                               max(task.timepoints))]
        if not origins:
            origins = cands[:1]

        if task.operation == "list":
            if getattr(task, "argmax_slot", None):
                return self._list_argmax(task, cands, corp_name, trace)
            return self._list(task, cands, corp_name, trace)

        # status/timeline: 후보 원본이 여럿이면 각각 timeline을 열어 identity로 판단
        timelines = []
        for o in origins:
            for tp in task.timepoints:
                ensure_request_time_remaining()
                try:
                    tl = (self._selected_timeline(task, o.event_key, tp)
                          if o.event_key else
                          self.rm.event_timeline(as_of=tp, rcept_no=o.rcept_no,
                                                 verify_evidence=True))
                except _SelectedEventProvenanceError as e:
                    lims.append(Limitation(code="event_identity_mismatch", detail=str(e)))
                    return claims, lims, used, clar
                except Exception as e:
                    lims.append(Limitation(code="event_lookup_error", detail=str(e)[:150])); tl = None
                if tl is not None:
                    timelines.append((o, tp, tl))
        if not timelines:
            lims.append(Limitation(code="not_found", detail="기준시점까지 관측된 사건이 없음"))
            return claims, lims, used, clar

        provisional = next((
            timeline for _, _, timeline in timelines
            if getattr(timeline, "identity_status", None) == "provisional"
        ), None)
        # A provisional correction sequence does not by itself make an exact
        # root-bound status observation provisional.  When the selected seed
        # is the canonical root and the first observation is the original
        # filing, the status task can report the canonical states.  A timeline
        # or separate correction/history task still carries the lineage
        # limitation because it promises the sequence itself.
        seed_receipt = getattr(task.selector, "seed_rcept_no", None)
        exact_root_status = (
            task.operation == "status"
            and bool(selected_key and seed_receipt)
            and all(
                timeline.root_rcept_no == seed_receipt
                and bool(timeline.observations)
                and timeline.observations[0].rcept_no == seed_receipt
                and not timeline.observations[0].is_correction
                for _, _, timeline in timelines))
        if provisional is not None and not exact_root_status:
            lims.append(Limitation(
                code="correction_identity_provisional",
                detail=(f"event_key={provisional.event_key}: "
                        "canonical identity_status=provisional"),
                affected_doc_ids=[provisional.root_rcept_no]))

        # 원공시가 코퍼스 밖(root가 정정본·해지본뿐): 최초 체결 내용은 확인 불가 → typed limitation
        for o, tp, tl in timelines[:1]:
            first = tl.observations[0] if tl.observations else None
            root_is_corr = first is not None and first.is_correction
            # A bounded timeline can begin at the first observation actually
            # present in the corpus.  In that case a pre-corpus original does
            # not prevent answering the requested interval, so do not turn a
            # complete range into a partial answer.  A one-point/full-history
            # request still receives the limitation, as do ranges that begin
            # before the first observable disclosure.
            observed_range_complete = (
                task.operation == "timeline"
                and len(task.timepoints) >= 2
                and first is not None
                and min(task.timepoints) >= first.observed_at
            )
            if (not observed_range_complete and (
                    root_is_corr or (
                        tl.root_rcept_no
                        and tl.root_rcept_no not in {
                            ob.rcept_no for ob in tl.observations}))):
                lims.append(Limitation(code="source_scope_prevents_complete_lineage",
                                       detail="제공된 공시 자료보다 앞선 원공시가 범위 밖에 있어 원공시의 최초 기재 내용과 그 이전 변경 이력은 확인할 수 없음. 제공된 정정본의 값을 최초 내용으로 단정하지 않음",
                                       affected_doc_ids=[tl.root_rcept_no] if tl.root_rcept_no else []))
                break
        # identity 모호성: 서로 다른 event_key인데 fingerprint가 같거나 identity_status=ambiguous
        keys = {tl.event_key for _, _, tl in timelines}
        ambiguous = any(
            tl.identity_status == "ambiguous"
            for _, _, tl in timelines) or (
                len(origins) > 1
                and len({tl.identity_fingerprint for _, _, tl in timelines}) < len(keys))
        ambiguous_roots: set[str] = set()
        if ambiguous:
            ambiguous_roots.update(o.rcept_no for o in origins)
            # A selected event_key can point at one representative timeline
            # even though canonical identity evidence records several possible
            # original receipts.  The limitation must cite the complete
            # candidate set; otherwise the displayed first receipt looks like
            # a silently chosen origin.  This is the same fail-closed contract
            # used by list queries below.
            for _origin, _tp, timeline in timelines:
                ambiguous_roots.update(
                    self.rm.event_identity_candidate_roots(
                        event_key=timeline.event_key,
                        as_of=max(task.timepoints)))
            used.extend(sorted(ambiguous_roots))
            lims.append(Limitation(
                code="ambiguous_event_origin",
                detail=(f"원공시 후보가 {len(ambiguous_roots)}건"
                        f"({', '.join(sorted(ambiguous_roots))})으로 사건 "
                        "identity가 확정되지 않음 — 해지·정정이 어느 "
                        "원계약에 대한 것인지 공시만으로 특정 불가"),
                affected_doc_ids=sorted(ambiguous_roots)))

        # A seed receipt names one original disclosure, while an event_key may
        # represent the broader event candidate selected by Stage1.  If a
        # termination is observed somewhere in an ambiguous sibling group,
        # neither original may inherit ``active`` or ``terminated`` from that
        # event-level observation.  Counterparty/event-level queries (no seed)
        # and direct termination-receipt queries remain answerable.
        root_bound_ambiguous = bool(
            ambiguous and seed_receipt in ambiguous_roots)

        def group_termination_observations(tp: str):
            if not root_bound_ambiguous:
                return []
            found = {}
            for root in sorted(ambiguous_roots):
                sibling = self.rm.event_timeline(
                    as_of=tp, rcept_no=root, verify_evidence=True)
                if sibling is None or sibling.corp_code != task.corp_code:
                    continue
                for observation in sibling.observations:
                    if getattr(observation, "is_termination", False):
                        found.setdefault(observation.rcept_no, observation)
            return list(found.values())

        # 시점별 상태 claim — 후보 원본이 여럿이면 '어느 원본이든 관측된 사건 상태'를 시점별로 요약
        by_tp: dict[str, list] = {}
        for o, tp, tl in timelines:
            by_tp.setdefault(tp, []).append((o, tl))
        # The typed three-point timeline is ``history start / history end /
        # later status cutoff``.  Emit the accumulated changes once at the
        # history end and the state once at the cutoff; repeating all changes
        # at every point adds no evidence and can multiply the answer size.
        three_point_range = (
            task.operation == "timeline" and len(task.timepoints) == 3)
        history_detail_tp = (
            task.timepoints[-2] if three_point_range
            else task.timepoints[-1])
        state_timepoints = (
            {task.timepoints[-1]} if three_point_range
            else set(task.timepoints))
        for tp in task.timepoints:
            ensure_request_time_remaining()
            group = by_tp.get(tp, [])
            if not group:
                continue
            emit_state = tp in state_timepoints
            emit_history = (
                task.operation == "timeline" and tp == history_detail_tp)
            if not emit_state and not emit_history:
                continue
            # 해지 관측이 있는 timeline 우선 (해지 공시는 사건 전체에 대한 사실)
            term = [(o, tl) for o, tl in group if tl.state.status == "terminated"]
            o, tl = (term[0] if term else group[0])
            state = tl.state.status
            ambiguous_terminations = group_termination_observations(tp)
            if root_bound_ambiguous:
                who = (task.selector.counterparty
                       or task.selector.contract_name or "")
                label = (
                    f"{corp_name}–{who} 계약" if who
                    else f"{corp_name} 계약")
                if emit_state:
                    observation_text = (
                        "해지 공시는 확인되지만 "
                        if ambiguous_terminations else
                        "기준일까지 해지 공시는 관측되지 않았지만 ")
                    claims.append(AnswerClaim(
                        output_id=f"{task.task_id}.state@{tp}",
                        label=(f"{label} 식별 공시 {seed_receipt}의 "
                               f"{tp[:4]}-{tp[4:6]}-{tp[6:]} 기준 상태"),
                        state=None,
                        text=("개별 상태 미확정 — " + observation_text
                              + "동일 조건의 원공시 후보가 복수이므로 "
                                "공시 관측만으로 이 계약의 법적 상태를 "
                                "확정하지 않음"),
                        citations=self._doc_cites(
                            task.corp_code, seed_receipt, tp)))
                    if not ambiguous_terminations:
                        claims.append(AnswerClaim(
                            output_id=(f"{task.task_id}."
                                       f"termination_observation@{tp}"),
                            label=f"{tp[:4]}-{tp[4:6]}-{tp[6:]} 기준 관측",
                            state="no_termination_observed",
                            text=("제공된 공시 자료에서는 기준일까지 "
                                  "해지·종료 공시가 확인되지 않았습니다. "
                                  "이는 공시 관측 결과이며 개별 원계약의 "
                                  "법적 유효성을 뜻하지 않습니다."),
                            citations=self._doc_cites(
                                task.corp_code, seed_receipt, tp)))
                for observation in ambiguous_terminations:
                    claims.append(AnswerClaim(
                        output_id=(f"{task.task_id}.termination_observation."
                                   f"{observation.rcept_no}@{tp}"),
                        label=f"해지 공시 {observation.rcept_no}",
                        state="termination_observed",
                        text=("해지 공시는 확인되지만 연결되는 원계약은 "
                              "공시만으로 특정할 수 없음"),
                        citations=self._doc_cites(
                            task.corp_code, observation.rcept_no, tp)))
                # Explicitly requested termination fields remain useful as
                # event-level observations.  Their labels and citations name
                # the termination receipt so they cannot be mistaken for
                # fields of the seeded original contract.
                for slot in task.requested_slots:
                    for observation in ambiguous_terminations:
                        row, status = self.field_value(
                            task.corp_code, observation.rcept_no, slot,
                            as_of=tp, trace=trace)
                        if (row is None or status != "verified"
                                or row.value is None):
                            continue
                        claims.append(AnswerClaim(
                            output_id=(f"{task.task_id}.termination_observation."
                                       f"{observation.rcept_no}.{slot}@{tp}"),
                            label=f"해지 공시 {observation.rcept_no} {slot}",
                            value_text=(row.value
                                        if _looks_numeric(row.value) else None),
                            raw_unit=_surface_unit(row.path),
                            text=_public_field_text(row),
                            citations=[_cite_field(row)]))
                        break
                if emit_history:
                    # Ambiguity blocks the semantic state assignment, not the
                    # existence of the original/correction filings themselves.
                    # Preserve those verified observations.  The termination
                    # row is intentionally excluded here because it was
                    # emitted above as an event-level observation and must not
                    # appear inside the seeded original's lineage.
                    for observation in tl.observations:
                        if getattr(observation, "is_termination", False):
                            continue
                        kind = ("정정" if observation.is_correction
                                else "체결/원본")
                        observed_at = observation.observed_at
                        claims.append(AnswerClaim(
                            output_id=(f"{task.task_id}.obs{observation.seq}"
                                       f"@{tp}"),
                            label=f"확인된 공시 이력 {observation.seq + 1}",
                            text=(f"{observed_at[:4]}-{observed_at[4:6]}-"
                                  f"{observed_at[6:]} {kind} 공시 "
                                  f"(접수번호 {observation.rcept_no})"),
                            citations=self._doc_cites(
                                task.corp_code, observation.rcept_no, tp)))
                        if observation.is_correction:
                            claims.extend(
                                self._correction_observation_claims(
                                    task.corp_code, tl.event_key,
                                    observation, tp, limit=8))
                        used.append(observation.rcept_no)
                used.append(seed_receipt)
                used.extend(
                    observation.rcept_no
                    for observation in ambiguous_terminations)
                continue
            if "해지종료공시관측여부" in task.requested_slots:
                # The user asked about an observable disclosure, not whether
                # the contract is legally valid.  Canonical event ``active``
                # only means no termination observation has been bound by the
                # cutoff; never render it as "유효(진행 중)" here.
                observations = tuple(tl.observations or ())
                termination_rows = [
                    row for row in observations
                    if bool(getattr(row, "is_termination", False))
                ]
                citations: list[ClaimCitation] = []
                citation_receipts: set[str] = set()
                evidence_rows = termination_rows or list(observations)
                for observation in evidence_rows:
                    for citation in self._doc_cites(
                            task.corp_code, observation.rcept_no, tp):
                        key = citation.rcept_no or citation.doc_id
                        if key not in citation_receipts:
                            citation_receipts.add(key)
                            citations.append(citation)
                who = (task.selector.counterparty
                       or task.selector.contract_name or "")
                label = (
                    f"{corp_name}–{who} 계약의 " if who
                    else f"{corp_name} 계약의 ")
                label += (
                    f"{tp[:4]}-{tp[4:6]}-{tp[6:]} 기준 "
                    "해지·종료 공시 관측 여부")
                if termination_rows:
                    receipts = ", ".join(
                        row.rcept_no for row in termination_rows)
                    text = (
                        "선택된 사건 계보에서 기준일까지 "
                        f"해지·종료 공시가 확인됨({receipts}).")
                    public_state = "termination_observed"
                else:
                    text = (
                        "선택된 사건의 제공된 공시 자료에서 기준일까지 "
                        "해지·종료 공시는 확인되지 않았습니다. 이는 공시 "
                        "확인 결과이며 계약의 법적 유효성을 판단한 것은 "
                        "아닙니다.")
                    public_state = "no_termination_observed"
                claims.append(AnswerClaim(
                    output_id=f"{task.task_id}.termination_observed@{tp}",
                    label=label, state=public_state, text=text,
                    citations=citations))
                used.extend(row.rcept_no for row in observations)
                continue
            lineage_observation_only = "원계약계보관측" in task.requested_slots
            if lineage_observation_only:
                # A missing canonical root is not an invitation to infer an
                # active/valid contract state from the first in-corpus
                # correction.  Keep the answer to the bounded observations;
                # the typed missing-root limitation supplied by Stage1 states
                # why the original contract itself cannot be established.
                for observation in tl.observations:
                    kind = (
                        "해지" if observation.is_termination else
                        ("정정" if observation.is_correction else "체결/원본"))
                    observed_at = observation.observed_at
                    display_date = (
                        f"{observed_at[:4]}-{observed_at[4:6]}-"
                        f"{observed_at[6:]}")
                    claims.append(AnswerClaim(
                        output_id=(f"{task.task_id}.obs{observation.seq}"
                                   f"@{tp}"),
                        label=f"확인된 공시 이력 {observation.seq + 1}",
                        text=(f"{display_date} {kind} 공시 "
                              f"(접수번호 {observation.rcept_no})"),
                        citations=self._doc_cites(
                            task.corp_code, observation.rcept_no, tp)))
                # The missing root prevents us from calling the first visible
                # correction the *initial* contract, but it does not erase the
                # verified content of that correction or the later termination
                # filing.  Reuse the ordinary timeline snapshot to expose
                # those in-corpus facts under the existing typed limitation.
                # This is still one selected event; no issuer-wide search or
                # inferred pre-corpus value is introduced.
                snapshot = EventStateSnapshot.from_timeline(
                    tl, operation="timeline", requested_slots=[])
                _state_receipt, _state_cites, bounded_facts = self._state_evidence(
                    task.corp_code, tl, tp, trace, snapshot)
                claims.extend(bounded_facts)
                if tl.observations:
                    latest_observation = tl.observations[-1]
                    latest_kind = (
                        "해지" if latest_observation.is_termination else
                        ("정정" if latest_observation.is_correction
                         else "체결/원본"))
                    latest_day = latest_observation.observed_at
                    claims.append(AnswerClaim(
                        output_id=f"{task.task_id}.latest_observation@{tp}",
                        label="기준시점까지 확인된 마지막 공시",
                        state=latest_kind,
                        text=(
                            f"{latest_day[:4]}-{latest_day[4:6]}-"
                            f"{latest_day[6:]} {latest_kind} 공시 "
                            f"(접수번호 {latest_observation.rcept_no})"),
                        citations=self._doc_cites(
                            task.corp_code, latest_observation.rcept_no, tp),
                    ))
                    termination_observed = any(
                        row.is_termination for row in tl.observations)
                    if ambiguous and not termination_observed:
                        # ``ambiguous_event_origin`` means the observations
                        # cannot be assigned to one legal contract lineage.
                        # Preserve the bounded negative observation, but never
                        # expose canonical ``active`` as a structured state.
                        claims.append(AnswerClaim(
                            output_id=f"{task.task_id}.semantic_status@{tp}",
                            label="기준시점 해지·종료 공시 관측 여부",
                            state="no_termination_observed",
                            text=(
                                "제공된 공시 자료에서 기준일까지 해지·종료 "
                                "공시가 확인되지 않았습니다. 이는 공시 관측 "
                                "결과이며 개별 원계약의 법적 유효성을 뜻하지 "
                                "않습니다."),
                            citations=self._doc_cites(
                                task.corp_code,
                                latest_observation.rcept_no, tp),
                        ))
                    elif ambiguous and termination_observed:
                        termination = next(
                            row for row in reversed(tl.observations)
                            if row.is_termination)
                        claims.append(AnswerClaim(
                            output_id=f"{task.task_id}.semantic_status@{tp}",
                            label="기준시점 해지·종료 공시 관측 여부",
                            state="termination_observed",
                            text=(
                                "해지·종료 공시는 확인되지만 원공시 후보가 "
                                "복수이므로 개별 원계약의 법적 상태는 "
                                "확정하지 않습니다."),
                            citations=self._doc_cites(
                                task.corp_code, termination.rcept_no, tp),
                        ))
                    elif tl.state.status == "active" and not termination_observed:
                        claims.append(AnswerClaim(
                            output_id=f"{task.task_id}.semantic_status@{tp}",
                            label="기준시점 공시 상태",
                            state="no_termination_observed",
                            text=(
                                "제공된 공시 자료에서 기준일까지 해지·종료 "
                                "공시가 확인되지 않았습니다. 이는 공시 확인 "
                                "결과이며 계약의 법적 유효성을 판단한 것은 "
                                "아닙니다."),
                            citations=self._doc_cites(
                                task.corp_code,
                                latest_observation.rcept_no, tp),
                        ))
                    elif tl.state.status == "terminated" and termination_observed:
                        termination = next(
                            row for row in reversed(tl.observations)
                            if row.is_termination)
                        claims.append(AnswerClaim(
                            output_id=f"{task.task_id}.semantic_status@{tp}",
                            label="기준시점 공시 상태",
                            state="terminated",
                            text=(
                                "제공된 공시 자료에서 해지·종료 공시가 "
                                "확인되어 공시상 종료된 상태입니다."),
                            citations=self._doc_cites(
                                task.corp_code, termination.rcept_no, tp),
                        ))
                used.extend(row.rcept_no for row in tl.observations)
                continue
            # 근거: 상태를 결정한 마지막 관측 문서의 Field (해지면 해지사유/해지금액, 아니면 계약상대)
            snapshot = EventStateSnapshot.from_timeline(
                # A multi-timepoint status request is a comparison of the
                # observable lifecycle at each date.  Its intervening
                # corrections are therefore answer evidence, just as they are
                # for an explicit timeline request.  Single-timepoint status
                # remains compact.
                tl, operation=("timeline" if len(task.timepoints) >= 2
                               else task.operation),
                requested_slots=task.requested_slots)
            state_rcept_no, cites, slot_claims = self._state_evidence(
                task.corp_code, tl, tp, trace, snapshot)
            who = task.selector.counterparty or task.selector.contract_name or ""
            noun = _event_public_noun(tl)
            subject = (f"{corp_name}–{who} 계약" if who and noun == "계약"
                       else f"{corp_name} {noun}")
            label = f"{subject}의 {tp[:4]}-{tp[4:6]}-{tp[6:]} 기준 공시상 상태"
            if emit_state:
                public_state = state
                # 근거는 claim 의 citations 로 붙어 답변 표면에
                # ``(접수번호 …)`` 로 렌더된다. 문장 안에 한 번 더 적으면 같은
                # 답변에서 인용 형식이 두 가지가 되고 접수번호가 두 번 나온다.
                public_text = _STATE_KO.get(state, state)
                public_cites = list(cites)
                if state == "active":
                    public_state, public_text, period_cites = (
                        _active_observation_projection(
                            timepoint=tp, timeline=tl,
                            slot_claims=slot_claims))
                    seen_cites = {
                        (citation.doc_id, citation.rcept_no,
                         citation.evidence_id, citation.locator)
                        for citation in public_cites}
                    for citation in period_cites:
                        key = (citation.doc_id, citation.rcept_no,
                               citation.evidence_id, citation.locator)
                        if key not in seen_cites:
                            seen_cites.add(key)
                            public_cites.append(citation)
                claims.append(AnswerClaim(
                    output_id=f"{task.task_id}.state@{tp}", label=label,
                    state=public_state,
                    # A terminated event is still bound to its positive
                    # termination observation.  Canonical active is projected
                    # above to a bounded observation instead of legal status.
                    text=public_text, citations=public_cites))
            # State completeness is independent of whether the caller also
            # asked for the observation history.  Previously single-point
            # ``event_state`` verified amount/ratio/date/reason here but then
            # discarded those claims, leaving the wording model only a status
            # sentence and allowing it to contradict a public amount.
            if emit_history:
                claims.extend(slot_claims)
            elif emit_state:
                # The final cutoff can contain a correction filed after the
                # middle history boundary.  Keep the complete verified
                # snapshot here; the exact evidence-aware deduper below folds
                # rows already emitted at the middle point without dropping a
                # genuinely later correction/context fact.
                claims.extend(slot_claims)
            used.extend(x.rcept_no for x in tl.observations)
            if tl.query_coverage_status == "not_proven" and not ambiguous:
                pass  # 완전 집합 증명은 identity 계약 밖 — limitation으로 올리지 않음(gold 미요구)
            if emit_history:
                for ob in tl.observations:
                    kind = "해지" if ob.is_termination else ("정정" if ob.is_correction else "체결/원본")
                    dt = f"{ob.observed_at[:4]}-{ob.observed_at[4:6]}-{ob.observed_at[6:]}"
                    claims.append(AnswerClaim(output_id=f"{task.task_id}.obs{ob.seq}@{tp}",
                                              label=f"이력 {ob.seq+1}", text=f"{dt} {kind} 공시 (접수번호 {ob.rcept_no})",
                                              citations=self._doc_cites(task.corp_code, ob.rcept_no, tp)))
        # 상태·일반 필드는 시점별로 보존한다. 다만 뒤 시점 snapshot이 이미
        # 제시한 동일 정정행/관련 중요사항을 같은 원근거로 재방출한 경우만
        # 한 번으로 접어 장문 중복을 줄인다.
        return _dedupe_public_event_facts(claims), lims, used, clar

    def _correction_observation_claims(
            self, corp_code: str, event_key: str, observation, as_of: str,
            *, limit: int,
            ) -> list[AnswerClaim]:
        """Expose verified before/after changes for one observed correction."""

        correction_tool = CorrectionTool(self.rm, self.fidx)
        claims: list[AnswerClaim] = []
        items = list(correction_tool._items_verified(
            corp_code, observation.rcept_no, as_of))
        # Prefer an actual literal change over a form-level placeholder row,
        # while preserving canonical item order within the same class.
        items.sort(key=lambda item: (
            (item.value_before or "").strip() in {"", "-"}
            and (item.value_after or "").strip() in {"", "-"},
            item.order,
        ))
        for item in items:
            if _is_verified_unchanged_placeholder(item):
                continue
            citations = []
            if item.before_evidence_status == "verified" and item.before_evidence_id:
                citations.append(ClaimCitation(
                    doc_id=item.doc_id, rcept_no=item.rcept_no,
                    evidence_id=item.before_evidence_id,
                    locator=item.before_locator or item.locator,
                    excerpt_prompt_safe=item.value_before_prompt_safe or ""))
            if item.after_evidence_status == "verified" and item.after_evidence_id:
                citations.append(ClaimCitation(
                    doc_id=item.doc_id, rcept_no=item.rcept_no,
                    evidence_id=item.after_evidence_id,
                    locator=item.after_locator or item.locator,
                    excerpt_prompt_safe=item.value_after_prompt_safe or ""))
            if item.reason:
                reason_cites = self._exact_field_cites(
                    corp_code, observation.rcept_no, "정정사유",
                    item.reason, as_of=as_of)
                known = {citation.evidence_id for citation in citations}
                citations.extend(
                    citation for citation in reason_cites
                    if citation.evidence_id not in known)
            if not citations:
                continue
            label = item.path.split(">")[-1].strip() if item.path else "정정항목"
            # This is the same scalar projection used by CorrectionTool's
            # ordinary history response.  Event snapshots must not turn a
            # verified correction amount into display-only prose: downstream
            # arithmetic and the public renderer rely on value_text/raw_unit.
            claims.append(AnswerClaim(
                output_id=(f"{event_key[:8]}.lineage.corr."
                           f"{observation.rcept_no}.{item.order}@{as_of}"),
                label=f"정정({observation.rcept_no}) {label}",
                **_correction_scalar_projection(
                    item,
                    before_verified=(
                        item.before_evidence_status == "verified"
                        and bool(item.before_evidence_id)),
                    after_verified=(
                        item.after_evidence_status == "verified"
                        and bool(item.after_evidence_id))),
                text=(f"{item.value_before or '(없음)'} → "
                      f"{item.value_after or '(없음)'}"
                      + (f" [사유: {item.reason}]" if item.reason else "")),
                operator="correction_diff",
                citations=citations,
            ))
            if len(claims) >= limit:
                break
        return claims

    def _state_evidence(
            self, corp_code, tl, tp, trace,
            snapshot: EventStateSnapshot):
        # 상태가 해지면 **해지를 성립시킨 관측**을 근거로 쓴다.
        #
        # `state.last_rcept_no` 는 그 시점까지의 *마지막* 관측이다.  같은 날 해지
        # 공시 뒤에 다른 공시(예: 유보 해제 체결 정정)가 접수되면, 상태를 정한
        # 문서와 인용하는 문서가 갈린다.  실제로 같은 날 해지 공시와 체결 정정공시가
        # 접수된 계약에서 「해지됨」의 근거로 체결 정정공시가 인용된 적이 있다.
        # 상태 판정은 이미 해지 관측을 우선해 고르고 있으므로,
        # 근거도 같은 관측을 따라가야 둘이 어긋나지 않는다.
        last = snapshot.state_rcept_no
        cites = self._doc_cites(corp_code, last, tp) if last else []
        slot_claims = []
        # A status answer needs the correction payload only when the correction
        # is the state-defining last observation.  A timeline answer is
        # different: every correction observation is part of the requested
        # lifecycle even when a later termination defines the final state.
        # Read each proved correction receipt independently so an intermediate
        # change cannot disappear behind that termination.
        correction_tool = CorrectionTool(self.rm, self.fidx)
        for correction_receipt in snapshot.correction_receipts:
            all_items = correction_tool._items_verified(
                corp_code, correction_receipt, tp)
            items = [item for item in all_items
                     if not _is_verified_unchanged_placeholder(item)]
            reasons = sorted({it.reason for it in all_items if it.reason})
            receipt_claims = []
            for it in items[:8]:
                side_cites = []
                if it.after_evidence_status == "verified" and it.after_evidence_id:
                    side_cites.append(ClaimCitation(doc_id=it.doc_id, rcept_no=it.rcept_no, evidence_id=it.after_evidence_id,
                                                    locator=it.after_locator or it.locator, excerpt_prompt_safe=it.value_after_prompt_safe or ""))
                elif it.before_evidence_status == "verified" and it.before_evidence_id:
                    side_cites.append(ClaimCitation(doc_id=it.doc_id, rcept_no=it.rcept_no, evidence_id=it.before_evidence_id,
                                                    locator=it.before_locator or it.locator, excerpt_prompt_safe=it.value_before_prompt_safe or ""))
                if it.reason:
                    reason_cites = self._exact_field_cites(
                        corp_code, correction_receipt, "정정사유",
                        it.reason, as_of=tp)
                    known = {citation.evidence_id for citation in side_cites}
                    side_cites.extend(
                        citation for citation in reason_cites
                        if citation.evidence_id not in known)
                if not side_cites:
                    continue
                lab = it.path.split(">")[-1].strip() if it.path else "정정항목"
                receipt_claims.append(AnswerClaim(
                    output_id=(f"{tl.event_key[:8]}.corr."
                               f"{correction_receipt}.{it.order}@{tp}"),
                    label=f"정정({correction_receipt}) {lab}",
                    **_correction_scalar_projection(
                        it,
                        before_verified=(
                            it.before_evidence_status == "verified"
                            and bool(it.before_evidence_id)),
                        after_verified=(
                            it.after_evidence_status == "verified"
                            and bool(it.after_evidence_id))),
                    text=f"{it.value_before or '(없음)'} → {it.value_after or '(없음)'}"
                         + (f" [사유: {it.reason}]" if it.reason else ""),
                    operator="correction_diff",
                    citations=side_cites))
            slot_claims.extend(receipt_claims)
            if reasons and receipt_claims:
                for reason_index, reason in enumerate(reasons):
                    reason_cites = self._exact_field_cites(
                        corp_code, correction_receipt, "정정사유", reason,
                        as_of=tp)
                    if not reason_cites:
                        continue
                    reason_id = (f"{tl.event_key[:8]}.corr.reason."
                                 f"{correction_receipt}.{reason_index}@{tp}")
                    if any(claim.output_id == reason_id
                           for claim in slot_claims):
                        continue
                    slot_claims.insert(0, AnswerClaim(
                        output_id=reason_id,
                        label=f"정정사유({correction_receipt})",
                        text=reason, citations=reason_cites))
        # 요청 slot + 해지면 해지금액·해지사유 기본 포함
        slots = list(snapshot.slots)
        # slot 값은 상태를 결정한 문서에서 우선, 없으면 관측 문서를 역순으로
        docs = [last] + [o.rcept_no for o in reversed(tl.observations) if o.rcept_no != last]
        for s in slots:
            for rc in docs:
                row, st = self.field_value(corp_code, rc, s, as_of=tp, trace=trace)
                if row is not None and st == "verified" and row.value is not None:
                    val = row.value
                    txt = _public_field_text(row)
                    citations = [_cite_field(row)]
                    # A withheld amount is one semantic fact with its reason
                    # and deadline, not three optional prose fragments.  Bind
                    # the companions to the same snapshot so a wording model
                    # cannot keep '-' while dropping why/until.
                    if s == "계약금액" and val.strip() == "-":
                        companions = []
                        for companion in ("유보사유", "유보기한"):
                            extra, extra_status = self.field_value(
                                corp_code, rc, companion,
                                as_of=tp, trace=trace)
                            if (extra is None or extra_status != "verified"
                                    or not extra.value
                                    or extra.value.strip() == "-"):
                                continue
                            companions.append(f"{companion} {extra.value}")
                            citations.append(_cite_field(extra))
                        if companions:
                            txt += " — " + ", ".join(companions)
                    # A field the source explicitly wrote as ``0`` (schema 1.8
                    # ``value_status=explicit_zero``) is a disclosed fact, not
                    # an absence.  Tag it so a wording model can say "명시된
                    # 0원" instead of ever downgrading it to blank/undisclosed
                    # (issue #63 — RPC-006).
                    state = ("explicit_zero"
                             if getattr(row, "value_status", None) == "explicit_zero"
                             else None)
                    slot_claims.append(AnswerClaim(output_id=f"{tl.event_key[:8]}.{s}@{tp}", label=s,
                                                   value_text=val if _looks_numeric(val) else None,
                                                   raw_unit=_surface_unit(row.path),
                                                   state=state,
                                                   text=txt, citations=citations))
                    break
        if snapshot.include_correction_history:
            # 정정공시는 원공시의 유의사항 문구를 그대로 다시 싣는 경우가 많다.
            # 정정 건마다 한 줄씩 내면 **같은 문단이 답변에 다섯 번** 나온다
            # (DEV-EVT-009).  값이 아니라 표시의 중복이므로, 같은 문구는 한 번만
            # 내고 접수번호와 근거는 모두 남긴다 — 어느 정정에 붙어 있었는지가
            # 사라지면 안 된다.
            grouped: dict[str, list[tuple[str, object]]] = {}
            for rc in [observation.rcept_no for observation in tl.observations]:
                row, status = self.field_value(
                    corp_code, rc, "효력발생조건", as_of=tp, trace=trace)
                if (row is None or status != "verified" or not row.value
                        or row.value.strip() == "-"):
                    continue
                grouped.setdefault(
                    " ".join(row.value.split()), []).append((rc, row))
            emitted = {
                " ".join((claim.text or "").split())
                for claim in slot_claims if claim.label == "효력발생조건"
            }
            for _key, rows in grouped.items():
                first_rc, first_row = rows[0]
                if " ".join(first_row.value.split()) in emitted:
                    # 같은 문구를 이미 효력발생조건으로 냈다.  다시 내면 본문에
                    # 두 번 나온다.
                    continue
                receipts = ", ".join(rc for rc, _row in rows)
                excerpt = complete_source_excerpt(
                    first_row.value, max_chars=900)
                if not excerpt:
                    continue
                slot_claims.append(AnswerClaim(
                    output_id=f"{tl.event_key[:8]}.context.{first_rc}@{tp}",
                    label=f"관련 중요사항({receipts})",
                    text=excerpt,
                    citations=[_cite_field(row) for _rc, row in rows]))
        return last, cites, slot_claims

    def _list(self, task, cands, corp_name, trace):
        claims, lims, used = [], [], []
        requested_slots = list(task.requested_slots)
        if ({"해지금액", "해지사유"} <= set(requested_slots)
                or {"해지금액", "해지 주요사유"} <= set(requested_slots)):
            # A compact comparison of multiple termination events should keep
            # the two same-document context fields that make the amount and
            # reason interpretable.  This is a closed typed shape, so ordinary
            # list queries do not fetch extra fields.
            requested_slots.extend(("매출액대비", "해지일자"))
        requested_slots = list(dict.fromkeys(requested_slots))
        # list: 후보 각각을 timeline으로 열어 상태·연결을 나열
        seen_keys = {}
        emitted_ambiguous_terminations: set[str] = set()
        for c in cands:
            tp = task.timepoints[-1]
            try:
                tl = (self._selected_timeline(task, c.event_key, tp)
                      if c.event_key else
                      self.rm.event_timeline(as_of=tp, rcept_no=c.rcept_no,
                                             verify_evidence=True))
            except _SelectedEventProvenanceError as e:
                lims.append(Limitation(code="event_identity_mismatch", detail=str(e)))
                return claims, lims, used, None
            except Exception:
                tl = None
            if tl is None:
                continue
            k = tl.event_key
            if k in seen_keys:
                continue
            seen_keys[k] = tl
            state = tl.state.status
            amb = tl.identity_status == "ambiguous"
            termination_receipts = [
                observation.rcept_no for observation in tl.observations
                if getattr(observation, "is_termination", False)]
            candidate_is_termination = c.rcept_no in termination_receipts
            public_state = state
            txt = f"{c.counterparty or ''} {c.contract_name or ''} — {_STATE_KO.get(state, state)}"
            if state == "active" and not amb:
                public_state = "no_termination_observed"
                txt = (
                    f"{c.counterparty or ''} {c.contract_name or ''} — "
                    "기준일까지 해지·종료 공시 미확인"
                    "(실제 진행 여부·법적 유효성 미확정)")
            elif amb:
                # A termination disclosure can prove that one event ended
                # while canonical evidence still cannot choose which of two
                # original receipts owns it.  Those originals are candidates,
                # not individually active/terminated contracts.  Keep the
                # collection-level termination count below, but expose no
                # per-origin state until an identity facet resolves the group.
                if candidate_is_termination:
                    public_state = "termination_observed"
                    txt = (f"{c.counterparty or ''} {c.contract_name or ''} — "
                           "해지 공시 확인(연결되는 원계약 미확정)")
                else:
                    public_state = None
                    txt = (f"{c.counterparty or ''} {c.contract_name or ''} — "
                           "이 원공시에 직접 결속된 해지는 확인되지 않음"
                           "(개별 상태 미확정)")
            citations = self._doc_cites(task.corp_code, c.rcept_no, tp)
            if state == "terminated":
                # A terminated list row names the original contract but its
                # state is proved by the termination observation.  Bind both
                # coordinates so the public citation cannot make an original
                # receipt appear to prove its own later termination.
                for receipt in termination_receipts:
                    for citation in self._doc_cites(
                            task.corp_code, receipt, tp):
                        if not any(existing.rcept_no == citation.rcept_no
                                   for existing in citations):
                            citations.insert(0, citation)
            claims.append(AnswerClaim(output_id=f"{task.task_id}.{c.rcept_no}", label=f"사건 {c.rcept_no}",
                                      state=public_state, text=txt, citations=citations))
            if state == "terminated" and amb and not candidate_is_termination:
                for receipt in termination_receipts:
                    if receipt in emitted_ambiguous_terminations:
                        continue
                    emitted_ambiguous_terminations.add(receipt)
                    claims.append(AnswerClaim(
                        output_id=(f"{task.task_id}.termination_observation."
                                   f"{receipt}"),
                        label=f"해지 공시 {receipt}",
                        state="termination_observed",
                        text=(f"{c.counterparty or ''} {c.contract_name or ''} — "
                              "해지 공시는 확인되지만 연결되는 원계약은 "
                              "공시만으로 특정할 수 없음"),
                        citations=self._doc_cites(
                            task.corp_code, receipt, tp)))
            used.extend(o.rcept_no for o in tl.observations)
            if state == "terminated" and (not amb or candidate_is_termination):
                etc, st_ = self.field_value(task.corp_code, c.rcept_no, "효력발생조건", as_of=tp, trace=trace)
                if etc is not None and st_ == "verified" and etc.value:
                    excerpt = complete_source_excerpt(
                        etc.value, max_chars=300)
                    if excerpt:
                        claims.append(AnswerClaim(
                            output_id=f"{task.task_id}.{c.rcept_no}.기타사항",
                            label=f"{c.rcept_no} 기타사항", text=excerpt,
                            citations=[_cite_field(etc)]))
            for s in requested_slots:
                if s == "최종상태":
                    # The collection's state claim above already carries the
                    # canonical uncertainty and termination evidence.
                    continue
                if s == "접수번호":
                    signed, signed_status = self.field_value(
                        task.corp_code, c.rcept_no, "계약(수주)일자", as_of=tp, trace=trace)
                    signed_text = (f"계약일 {signed.value}, "
                                   if signed is not None and signed_status == "verified" else "")
                    claims.append(AnswerClaim(
                        output_id=f"{task.task_id}.{c.rcept_no}.receipt",
                        label=f"{c.rcept_no} 원공시",
                        text=f"{signed_text}공시일 {c.rcept_no[:4]}-{c.rcept_no[4:6]}-{c.rcept_no[6:8]}, 접수번호 {c.rcept_no}",
                        citations=([_cite_field(signed)] if signed_text else citations)))
                    continue
                if s == "금액공개여부":
                    # An ambiguous termination cannot lend its disclosed
                    # amount to either original contract. State the original
                    # field and its withholding note, with the scope explicit.
                    amount, amount_status = self.field_value(
                        task.corp_code, c.rcept_no, "계약금액", as_of=tp, trace=trace)
                    note, note_status = self.field_value(
                        task.corp_code, c.rcept_no, "효력발생조건", as_of=tp, trace=trace)
                    source_rows = []
                    pieces = []
                    if amount is not None and amount_status == "verified":
                        pieces.append(f"원공시 계약금액: {amount.value}")
                        if str(amount.value).strip() in {"", "-"}:
                            pieces.append("금액 미기재이며 0원을 뜻하지 않음")
                        source_rows.append(amount)
                    if note is not None and note_status == "verified":
                        identity_sentence = self._contract_identity_sentence(note.value)
                        if identity_sentence:
                            pieces.append(identity_sentence)
                            source_rows.append(note)
                    if pieces:
                        claims.append(AnswerClaim(
                            output_id=f"{task.task_id}.{c.rcept_no}.amount_availability",
                            label=f"{c.rcept_no} 금액 공개 여부",
                            text="; ".join(pieces), citations=[_cite_field(row) for row in source_rows]))
                    else:
                        lims.append(Limitation(code="evidence_unavailable",
                            detail=f"{c.rcept_no} 원공시 금액 공개 여부를 이번 조회에서 확인하지 못함",
                            affected_doc_ids=[c.rcept_no]))
                    continue
                if amb and not candidate_is_termination:
                    slot_receipts = [c.rcept_no] + [
                        observation.rcept_no
                        for observation in reversed(tl.observations)
                        if not getattr(observation, "is_termination", False)
                    ]
                elif state == "terminated":
                    # Same-day contract corrections may follow the actual
                    # termination receipt.  Termination amount/ratio/date and
                    # reason must come from the event-defining disclosure,
                    # not from that later contract snapshot.
                    slot_receipts = termination_receipts + [
                        observation.rcept_no
                        for observation in reversed(tl.observations)
                        if observation.rcept_no not in termination_receipts
                    ]
                else:
                    slot_receipts = [tl.state.last_rcept_no] + [
                        observation.rcept_no
                        for observation in reversed(tl.observations)
                    ]
                for rc in dict.fromkeys(slot_receipts):
                    row, st = self.field_value(task.corp_code, rc, s, as_of=tp, trace=trace)
                    if row is not None and st == "verified" and row.value:
                        claims.append(AnswerClaim(output_id=f"{task.task_id}.{c.rcept_no}.{s}", label=f"{c.rcept_no} {s}",
                                                  value_text=row.value if _looks_numeric(row.value) else None,
                                                  raw_unit=_surface_unit(row.path),
                                                  text=row.value, citations=[_cite_field(row)]))
                        break

        candidate_receipts = {candidate.rcept_no for candidate in cands}
        self._append_listed_origin_discriminators(
            task, timelines=tuple(seen_keys.values()),
            candidate_receipts=candidate_receipts,
            claims=claims, trace=trace)
        if seen_keys:
            n_all = len(seen_keys)
            n_term = sum(1 for tl in seen_keys.values() if tl.state.status == "terminated")
            n_amb = sum(1 for tl in seen_keys.values() if tl.state.status == "terminated" and tl.identity_status == "ambiguous")
            # A membership/existence question ("...해지된 계약이 존재하는가")
            # is answered by this one leading claim, not by reading every
            # per-event row below it.  Name the unambiguous terminated
            # contracts here so the answer states existence, count, and
            # receipts up front; the individual rows for every other
            # (unterminated or identity-ambiguous) candidate remain below for
            # a reader who wants the full scan, none are removed.
            terminated_pairs = sorted(
                (tl.root_rcept_no, receipt)
                for tl in seen_keys.values()
                if tl.state.status == "terminated" and tl.identity_status != "ambiguous"
                for receipt in sorted({
                    observation.rcept_no for observation in tl.observations
                    if getattr(observation, "is_termination", False)})[:1])
            existence_summary = (
                " — " + ", ".join(
                    f"{root} 체결 → {term} 해지"
                    for root, term in terminated_pairs)
                if terminated_pairs else "")
            claims.insert(0, AnswerClaim(output_id=f"{task.task_id}.count", label="사건 건수",
                                         value_text=str(n_all), text=(
                                             f"조건에 맞는 사건 {n_all}건, 그중 해지 {n_term}건"
                                             + (f" (원본 identity 모호 {n_amb}건 포함 — 확정 연결 아님)" if n_amb else "")
                                             + existence_summary),
                                         citations=claims[0].citations if claims else []))
            claims.insert(1, AnswerClaim(output_id=f"{task.task_id}.terminated_count", label="해지 공시 관측 건수",
                                         value_text=str(n_term), text=(f"해지 공시가 관측된 사건 {n_term}건"
                                              + (f" (원계약 특정 불가 {n_amb}건 포함)" if n_amb else "")),
                                         citations=claims[1].citations if len(claims) > 1 else []))
            # When Stage1 resolves an issuer omitted from the question through
            # one unique corpus-backed role, state that issuer explicitly.
            # This prevents a general reader from mistaking the named
            # counterparties for the company that filed the disclosures.
            issuer_citations = next((
                list(claim.citations) for claim in claims
                if claim.citations), [])
            claims.insert(2, AnswerClaim(
                output_id=f"{task.task_id}.issuer",
                label="공시 주체", text=corp_name,
                citations=issuer_citations))
        ambiguous_roots: set[str] = set()
        for timeline in seen_keys.values():
            if timeline.identity_status != "ambiguous":
                continue
            roots = set(self.rm.event_identity_candidate_roots(
                event_key=timeline.event_key, as_of=task.timepoints[-1]))
            ambiguous_roots.update(roots)
        notice_roots = sorted(ambiguous_roots)
        if notice_roots:
            # 후보 전체가 limitation 자체의 근거다. 추가 식별 축이
            # 생기기 전에는 단일 원공시로 승격하지 않는다.  A locally
            # matched period can establish a listed state, but is not a
            # mutation of the canonical identity graph.
            used.extend(notice_roots)
            lims.append(Limitation(
                code="ambiguous_event_origin",
                detail=(
                    "해지 공시의 존재와 내용은 확인되지만 동일 조건의 "
                    "원공시 후보가 복수라 canonical 원계약 계보는 미확정임; "
                    "계약기간 일치는 후보 구분 정보일 뿐 후보 하나를 "
                    "해지 원계약으로 확정하지 않음"),
                affected_doc_ids=notice_roots))
        if not claims:
            lims.append(Limitation(code="not_found", detail=f"{corp_name}: 조건에 맞는 사건 없음"))
        return claims, lims, used, None

    def _list_argmax(self, task, cands, corp_name, trace):
        """Reduce a canonical event list to its ``argmax_slot`` extreme row.

        이슈 #59 2단계 — resolver 는 이미 후보 전체(예: 한 해에 공시된 단일
        판매·공급계약 n건)를 canonical 하게 증명해 뒀다(``cands``). 값 자체는
        여기서, Stage2 가 검증된 Field evidence 로만 읽는다 — compiler 는
        어느 slot 을 줄일지만 지시했다(``task.argmax_slot``). 비공개(``-``)
        값은 0원으로 취급하지 않고 그냥 제외한다; 확인된 값이 하나도 없으면
        fail-closed 로 닫는다(``not_found``에 가까운 전용 코드).

        이슈 #124 — ``task.argmax_direction``(기본 "maximum")이 "minimum"
        이면 같은 축소를 반대 방향(최솟값)으로 한다.
        """
        argmax_slot = task.argmax_slot
        direction = getattr(task, "argmax_direction", "maximum")
        minimize = direction == "minimum"
        extreme_word = "최솟값" if minimize else "최댓값"
        other_slots = [slot for slot in task.requested_slots if slot != argmax_slot]
        tp = task.timepoints[-1]
        rows: list[tuple[Any, AnswerClaim | None, Decimal | None]] = []
        seen_keys: set[str] = set()
        for c in cands:
            ensure_request_time_remaining()
            try:
                tl = (self._selected_timeline(task, c.event_key, tp)
                      if c.event_key else
                      self.rm.event_timeline(as_of=tp, rcept_no=c.rcept_no,
                                             verify_evidence=True))
            except _SelectedEventProvenanceError as e:
                return [], [Limitation(
                    code="event_identity_mismatch", detail=str(e))], [], None
            except Exception:
                tl = None
            if tl is None:
                continue
            if tl.event_key in seen_keys:
                continue
            seen_keys.add(tl.event_key)
            row, status = self.field_value(
                task.corp_code, c.rcept_no, argmax_slot, as_of=tp, trace=trace)
            claim = None
            if row is not None and status == "verified" and row.value:
                claim = AnswerClaim(
                    output_id=f"{task.task_id}.{c.rcept_no}.{argmax_slot}",
                    label=f"{c.rcept_no} {argmax_slot}",
                    value_text=row.value if _looks_numeric(row.value) else None,
                    raw_unit=_surface_unit(row.path), text=row.value,
                    citations=[_cite_field(row)])
            rows.append((c, claim, _decimal_won(claim)))
        total = len(rows)
        if total == 0:
            return [], [Limitation(
                code="not_found",
                detail=f"{corp_name}: 조건에 맞는 사건 없음")], [], None
        confirmed_rows = [row for row in rows if row[2] is not None]
        confirmed = len(confirmed_rows)
        used = [c.rcept_no for c, _claim, _won in rows]
        if not confirmed_rows:
            return [], [Limitation(
                code="argmax_all_withheld",
                detail=(f"{corp_name}: 조건에 맞는 사건 {total}건 모두 "
                        f"{argmax_slot}이 비공개(-)로 표시되어 {extreme_word}을 "
                        "확정할 수 없음"))], used, None
        extreme_won = (min if minimize else max)(
            won for _c, _claim, won in confirmed_rows)
        winners = [(c, claim) for c, claim, won in confirmed_rows if won == extreme_won]
        lims: list[Limitation] = []
        if len(winners) > 1:
            lims.append(Limitation(
                code="argmax_tie",
                detail=(f"{argmax_slot} {extreme_word} 동률 {len(winners)}건: "
                        + ", ".join(c.rcept_no for c, _cl in winners)),
                affected_doc_ids=[c.rcept_no for c, _cl in winners]))
        winner, winner_claim = winners[0]
        claims = [winner_claim]
        for slot in other_slots:
            row, status = self.field_value(
                task.corp_code, winner.rcept_no, slot, as_of=tp, trace=trace)
            if row is not None and status == "verified" and row.value:
                claims.append(AnswerClaim(
                    output_id=f"{task.task_id}.{winner.rcept_no}.{slot}",
                    label=f"{winner.rcept_no} {slot}",
                    value_text=row.value if _looks_numeric(row.value) else None,
                    raw_unit=_surface_unit(row.path), text=row.value,
                    citations=[_cite_field(row)]))
        caveat = (
            f"; 비공개(-)로 표시된 건이 있어 전체 {extreme_word}은 확정할 수 없음 — "
            f"공개된 값 중 {extreme_word}임"
            if confirmed < total else "")
        summary = AnswerClaim(
            output_id=f"{task.task_id}.argmax.{argmax_slot}",
            label=f"{argmax_slot} {extreme_word}", state="argmax_summary",
            value_text=winner_claim.value_text, raw_unit=winner_claim.raw_unit,
            text=(f"{corp_name} 조건에 맞는 사건 {total}건 중 {argmax_slot} "
                  f"확인된 {confirmed}건 기준 {extreme_word}은 {winner_claim.text}"
                  f"({winner.rcept_no} 공시){caveat}"),
            citations=list(winner_claim.citations))
        claims.insert(0, summary)
        return claims, lims, used, None


class DisclosureTool(BaseTool):
    @staticmethod
    def _matches_financing_types(rows, keywords) -> bool:
        """Match requested issuance forms, not incidental funding wording."""
        forms = {
            "유상증자": ("유상증자결정", "유무상증자결정"),
            "전환사채": ("전환사채권발행결정",),
            "신주인수권부사채": ("신주인수권부사채권발행결정",),
            "교환사채": ("교환사채권발행결정",),
        }
        requested = [word for word in keywords if word in forms]
        if not requested:
            return True
        return any(
            form in re.sub(r"\s+", "", row.form or row.report_nm or "")
            for row in rows[:1] for word in requested for form in forms[word])

    def run_task(self, task, *, trace, corp_name: str):
        claims, lims, used, clar = [], [], [], None
        docs: list[str] = []
        unknown_counterparty_candidates: list[EventCandidate] = []
        if task.document_selector and task.document_selector.rcept_no:
            docs = [task.document_selector.rcept_no]
        elif task.event_selector:
            cands = self.find_candidates(task.corp_code, task.event_selector, as_of=task.as_of)
            docs = [c.rcept_no for c in cands]
            if task.operation == "list":
                unknown_counterparty_candidates = (
                    self.find_unknown_counterparty_candidates(
                        task.corp_code, task.event_selector, as_of=task.as_of))
        elif task.document_selector:
            ds = task.document_selector
            for rc, rows in self.fidx.iter_docs(
                    task.corp_code, as_of=task.as_of,
                    doc_group=ds.doc_group, form=ds.form,
                    is_correction=ds.is_correction):
                f = rows[0]
                if ds.rcept_from and f.rcept_dt < ds.rcept_from: continue
                if ds.rcept_to and f.rcept_dt > ds.rcept_to: continue
                if ds.report_name_contains and ds.report_name_contains not in (f.report_nm or ""): continue
                docs.append(rc)
        if (task.operation == "list" and task.event_selector
                and ("조달유형" in task.requested_slots
                     or any("자금조달" in slot for slot in task.requested_slots))):
            docs = [receipt for receipt in docs if self._matches_financing_types(
                list(self.fidx.rows(task.corp_code, as_of=task.as_of,
                                    rcept_no=receipt)),
                getattr(task.event_selector, "keywords", ()) or ())]
        trace.append(TraceEvent(seq=len(trace)+1, stage="tool", summary=f"disclosure {task.operation} docs={len(docs)}",
                                detail={"rcept_nos": docs[:20]}))
        if not docs:
            lims.append(Limitation(code="not_found", detail=f"{corp_name}: 조건에 맞는 공시 없음"))
            return claims, lims, used, clar
        if task.operation == "lookup" and len(docs) > 1:
            clar = Clarification(clarification_id=f"clarify-{task.task_id}-doc",
                                 question="조건에 맞는 공시가 여러 건입니다. 어느 접수번호 기준으로 확인할까요?",
                                 targets=["rcept_no"], options={"rcept_no": docs[:10]})
            return claims, lims, used, clar
        if task.operation == "list":
            forms_seen = {}
            for rc in docs:
                rows = list(self.fidx.rows(task.corp_code, as_of=task.as_of, rcept_no=rc))
                if not rows: continue
                r0 = rows[0]
                res = self.rm.resolve_document_version(rc, as_of=task.as_of)
                latest = res.selected if res.status == "ok" else None
                forms_seen.setdefault(r0.form, []).append(rc)
                claims.append(AnswerClaim(output_id=f"{task.task_id}.{rc}.doc", label=f"{r0.form} {rc}",
                                          state=r0.form, text=f"{r0.form} {rc} ({r0.rcept_dt}){' 정정' if r0.is_correction else ' 원본'}"
                                               + (f", 최신 유효본 {latest}" if latest else ""),
                                          citations=[_cite_field(self.fidx.verified(rows[0]))] if self.fidx.verified(rows[0]).evidence_status == "verified" else self._doc_cites(task.corp_code, rc, task.as_of)))
                used.append(rc)
            if forms_seen:
                claims.insert(0, AnswerClaim(output_id=f"{task.task_id}.forms", label="확인된 공시 유형",
                                             text="확인됨: " + ", ".join(f"{k} {len(v)}건" for k, v in forms_seen.items()),
                                             citations=claims[0].citations if claims else []))
                if ("조달유형" in task.requested_slots
                        or any("자금조달" in slot
                               for slot in task.requested_slots)):
                    projected, projected_used = self._financing_summary(
                        task, docs, trace)
                    claims.extend(projected)
                    used.extend(projected_used)
            if unknown_counterparty_candidates:
                receipts = []
                for candidate in unknown_counterparty_candidates:
                    citations = self._doc_cites(
                        task.corp_code, candidate.rcept_no, task.as_of)
                    if not citations:
                        continue
                    receipts.append(candidate.rcept_no)
                    name = candidate.contract_name or "계약명 미확인"
                    claims.append(AnswerClaim(
                        output_id=(f"{task.task_id}.{candidate.rcept_no}."
                                   "counterparty_unknown"),
                        label=f"계약상대 비공개 후보 {candidate.rcept_no}",
                        text=(f"{candidate.form} {candidate.rcept_no} "
                              f"({candidate.rcept_dt}), {name} — 계약상대가 "
                              "공개되지 않아 요청 상대방과의 일치 여부 판정 불가"),
                        citations=citations))
                    used.append(candidate.rcept_no)
                if receipts:
                    lims.append(Limitation(
                        code="counterparty_not_reported_may_hide_match",
                        detail=("계약상대 predicate가 미확정인 관련 후보: "
                                + ", ".join(receipts)),
                        affected_doc_ids=receipts))
        requested_slots = list(task.requested_slots)
        if any(s in {"해지사유", "해지 주요사유"}
               for s in requested_slots):
            # The terse form row is not always the complete disclosed reason.
            # The same termination disclosure can define what the termination
            # date means and state the bounded factual background in its
            # ``기타 투자판단`` paragraph.  Add those same-document fields as
            # typed supporting context; do not synthesize an explanation from
            # the short reason string and do not scan unrelated documents.
            requested_slots.extend(("해지일자", "효력발생조건"))
            # A multi-event comparison that already asks for termination
            # amount and reason is materially clearer with the disclosed
            # revenue ratio.  Fetch it only for this closed termination shape;
            # a reason-only lookup does not acquire unrelated numeric fields.
            if any(s == "해지금액" for s in requested_slots):
                requested_slots.append("매출액대비")
        requested_slots = list(dict.fromkeys(requested_slots))
        for rc in docs:
            for s in requested_slots:
                row, st = self.field_value(task.corp_code, rc, s, as_of=task.as_of, trace=trace)
                if row is None:
                    if st == "ambiguous":
                        lims.append(Limitation(code="ambiguous_field", detail=f"{rc} {s}: 후보 복수", affected_doc_ids=[rc]))
                    continue
                if st != "verified":
                    lims.append(Limitation(code="evidence_unavailable", detail=f"{rc} {s}: Evidence 미검증", affected_doc_ids=[rc])); continue
                val = row.value or ""
                oid = None
                for fo in getattr(task, "field_outputs", []) or []:
                    if fo.slot == s: oid = fo.output_id
                if (val.strip() == "-"
                        and row.value_status == "not_reported"
                        and s in ("계약금액", "정정후_계약금액")):
                    # '-'가 실제 not_reported로 분류된 경우에만 비공개로 설명한다.
                    # 근거가 확인된 유보사유·기한은 같은 claim에 결속한다.
                    reason, reason_status = self.field_value(
                        task.corp_code, rc, "유보사유",
                        as_of=task.as_of, trace=trace)
                    until, until_status = self.field_value(
                        task.corp_code, rc, "유보기한",
                        as_of=task.as_of, trace=trace)
                    # 이 값이 들어가는 표의 열 이름이 이미 ``계약금액 공개
                    # 여부`` 다. 셀에서 같은 말을 반복하면 한 칸이 길어지고
                    # 정작 답인 ``확인할 수 없음`` 이 뒤로 밀린다.
                    txt = "확인할 수 없음 — 공시에서 공개되지 않았습니다"
                    citations = [_cite_field(row)]
                    if (reason is not None and reason_status == "verified"
                            and reason.value and reason.value.strip() != "-"):
                        txt += f" — 유보사유: {reason.value}"
                        citations.append(_cite_field(reason))
                    if (until is not None and until_status == "verified"
                            and until.value and until.value.strip() != "-"):
                        txt += f", 유보기한: {until.value}"
                        citations.append(_cite_field(until))
                    claims.append(AnswerClaim(output_id=oid or f"{task.task_id}.{rc}.{s}", label=f"{rc} {s}",
                                              state="withheld", text=txt, citations=citations))
                    etc, _ = self.field_value(task.corp_code, rc, "효력발생조건", as_of=task.as_of, trace=trace)
                    if etc is not None and etc.value:
                        excerpt = complete_source_excerpt(
                            etc.value, max_chars=400)
                        if excerpt:
                            claims.append(AnswerClaim(
                                output_id=f"{task.task_id}.{rc}.기타사항",
                                label=f"{rc} 기타 투자판단 사항",
                                text=excerpt, citations=[_cite_field(etc)]))
                    used.append(rc); continue
                claims.append(AnswerClaim(output_id=oid or f"{task.task_id}.{rc}.{s}", label=f"{rc} {s}",
                                          value_text=val if _looks_numeric(val) else None,
                                          raw_unit=_surface_unit(row.path),
                                          state=("explicit_zero"
                                                 if row.value_status == "explicit_zero"
                                                 else None),
                                          text=_public_field_text(row),
                                          citations=[_cite_field(row)]))
                used.append(rc)
        if not claims:
            lims.append(Limitation(code="not_found", detail="요청 slot 값을 확정하지 못함"))
        return claims, lims, used, clar

    def _financing_summary(self, task, docs: list[str], trace):
        """Collapse correction versions into logical financing decisions.

        The projection is driven by the selected document lineage and explicit
        form paths.  It never counts four corrections as four issuances and it
        scopes zero results to the requested issuer/date/document family.
        """

        claims: list[AnswerClaim] = []
        used: list[str] = []
        groups: dict[tuple[str, ...], tuple[str, list[str]]] = {}
        for receipt in docs:
            resolution = self.rm.resolve_document_version(
                receipt, as_of=task.as_of)
            if resolution.status != "ok" or not resolution.selected:
                continue
            members = list(getattr(resolution, "members", ()) or (receipt,))
            key = tuple(members)
            groups.setdefault(key, (resolution.selected, members))

        observed_surfaces: list[str] = []
        if groups:
            first_receipt = next(iter(groups.values()))[0]
            claims.append(AnswerClaim(
                output_id=f"{task.task_id}.financing.logical_count",
                label="자금조달 의사결정 건수",
                value_text=str(len(groups)),
                text=(f"문서 {len(docs)}건은 정정 계보를 접으면 "
                      f"자금조달 의사결정 {len(groups)}건"),
                citations=self._doc_cites(
                    task.corp_code, first_receipt, task.as_of)))
        for selected, members in groups.values():
            selected_rows = list(self.fidx.rows(
                task.corp_code, as_of=task.as_of, rcept_no=selected))
            if not selected_rows:
                continue
            form = selected_rows[0].form or selected_rows[0].report_nm or "자금조달"
            observed_surfaces.append(
                " ".join((row.form or "") + " " + (row.report_nm or "")
                         for row in selected_rows[:1]))
            lineage_cites = []
            for member in members:
                lineage_cites.extend(self._doc_cites(
                    task.corp_code, member, task.as_of))
            claims.append(AnswerClaim(
                output_id=f"{task.task_id}.financing.{selected}.lineage",
                label=f"{form} 정정 계보",
                state="confirmed",
                text=" → ".join(members) + f"; 최신 유효본 {selected}",
                citations=lineage_cites))
            used.extend(members)

            # A request for both issuance terms and funding purpose asks for
            # the *change process*, not just the root/latest snapshots.  Read
            # the declared correction rows in lineage order and expose only
            # the two exact top-level table families relevant to those roles.
            # This is intentionally gated by public requested-slot semantics;
            # ordinary financing summaries retain their compact output.
            compact_slots = [
                re.sub(r"\s+", "", str(slot))
                for slot in tuple(getattr(task, "requested_slots", ()) or ())
            ]
            wants_issuance_and_funding = any(
                "발행" in slot and "자금조달" in slot
                for slot in compact_slots)
            if wants_issuance_and_funding and len(members) > 1:
                correction_reader = CorrectionTool(self.rm, self.fidx)
                relevant_path = re.compile(
                    r"^\s*[0-9]+\.\s*(?:신주\s*발행가액|자금조달의\s*목적)\s*$")
                for member in members[1:]:
                    try:
                        correction_items = correction_reader._items_verified(
                            task.corp_code, member, task.as_of)
                    except Exception:
                        continue
                    for correction_item in correction_items:
                        if (not relevant_path.fullmatch(
                                str(correction_item.path or ""))
                                or correction_item.before_evidence_status
                                != "verified"
                                or correction_item.after_evidence_status
                                != "verified"
                                or not correction_item.before_evidence_id
                                or not correction_item.after_evidence_id
                                or correction_item.value_before is None
                                or correction_item.value_after is None):
                            continue
                        leaf = correction_item.path.split(">")[-1].strip()
                        citations = [
                            ClaimCitation(
                                doc_id=correction_item.doc_id,
                                rcept_no=correction_item.rcept_no,
                                evidence_id=correction_item.before_evidence_id,
                                locator=(correction_item.before_locator
                                         or correction_item.locator),
                                excerpt_prompt_safe=(
                                    correction_item.value_before_prompt_safe)),
                            ClaimCitation(
                                doc_id=correction_item.doc_id,
                                rcept_no=correction_item.rcept_no,
                                evidence_id=correction_item.after_evidence_id,
                                locator=(correction_item.after_locator
                                         or correction_item.locator),
                                excerpt_prompt_safe=(
                                    correction_item.value_after_prompt_safe)),
                        ]
                        # 이슈 #112 — 정정 전/후 값이 둘 다 표(「6. 신주
                        # 발행가액」의 보통주식·기타주식 두 행 등)면 표는
                        # 줄 단위 구조라 " → " 로 한 줄에 이을 수 없다.
                        # 바뀐 셀만 짝지어 문장으로 내고, 못 짝지으면 표
                        # 두 개를 각각 세운다. 사유는 표 행에 붙이지 않고
                        # 별도 줄이다.
                        table_change = _correction_table_change_text(
                            correction_item.value_before,
                            correction_item.value_after)
                        reason = (f"[사유: {correction_item.reason}]"
                                  if correction_item.reason else "")
                        if table_change is not None:
                            change_text = (f"{table_change}\n{reason}"
                                          if reason else table_change)
                        else:
                            change_text = (
                                f"{correction_item.value_before} → "
                                f"{correction_item.value_after}"
                                + (f" {reason}" if reason else ""))
                        claims.append(AnswerClaim(
                            output_id=(f"{task.task_id}.financing.{member}."
                                       f"correction-{correction_item.order}"),
                            label=f"정정 단계 {member} · {leaf}",
                            **_correction_scalar_projection(correction_item),
                            text=change_text,
                            operator="correction_diff",
                            citations=citations,
                        ))
                        used.append(member)

            def exact(path: str):
                row, status = self.field_value(
                    task.corp_code, selected, path,
                    as_of=task.as_of, trace=trace)
                return row if row is not None and status == "verified" else None

            scalar_paths = (
                ("신주 수", "1. 신주의 종류와 수 > 보통주식 (주)"),
                ("1주당 액면가", "2. 1주당 액면가액 (원)"),
                ("신주 발행가", "6. 신주 발행가액 > 확정발행가 > 보통주식 (원)"),
            )
            for label, path in scalar_paths:
                row = exact(path)
                if row is None or not row.value or row.value.strip() == "-":
                    continue
                claims.append(AnswerClaim(
                    output_id=f"{task.task_id}.financing.{selected}.{label}",
                    label=f"최신 유효본 {label}",
                    value_text=row.value if _looks_numeric(row.value) else None,
                    raw_unit=_surface_unit(row.path), text=row.value,
                    citations=[_cite_field(row)]))

            purpose_rows = []
            for candidate in selected_rows:
                if not _path_segments(candidate.path)[:1] == (
                        _path_segments("4. 자금조달의 목적")[0],):
                    continue
                verified = self.fidx.verified(candidate)
                if (verified.evidence_status != "verified"
                        or not verified.value or verified.value.strip() == "-"):
                    continue
                amount = _decimal_scalar(verified.value)
                if amount is not None:
                    purpose_rows.append((verified, amount))
            total = sum((amount for _, amount in purpose_rows), Decimal(0))
            total_cites = []
            for row, amount in purpose_rows:
                total_cites.append(_cite_field(row))
                claims.append(AnswerClaim(
                    output_id=(f"{task.task_id}.financing.{selected}.purpose."
                               f"{row.order}"),
                    label=f"최신 유효본 {row.label}",
                    value_text=row.value, raw_unit="원",
                    text=f"{row.label} {amount:,.0f}원",
                    citations=[_cite_field(row)]))
            if purpose_rows:
                claims.append(AnswerClaim(
                    output_id=f"{task.task_id}.financing.{selected}.total",
                    label="최신 유효본 총 조달금액",
                    value_text=f"{total:,.0f}", raw_unit="원",
                    canonical_value=str(total), canonical_unit="원",
                    text=f"{total:,.0f}원", citations=total_cites))

            if len(members) > 1:
                original = members[0]
                original_rows = list(self.fidx.rows(
                    task.corp_code, as_of=task.as_of, rcept_no=original))
                original_purposes = []
                for candidate in original_rows:
                    if not _path_segments(candidate.path)[:1] == (
                            _path_segments("4. 자금조달의 목적")[0],):
                        continue
                    verified = self.fidx.verified(candidate)
                    amount = _decimal_scalar(verified.value)
                    if verified.evidence_status == "verified" and amount is not None:
                        original_purposes.append((verified, amount))
                original_total = sum(
                    (amount for _, amount in original_purposes), Decimal(0))
                if purpose_rows and original_purposes:
                    original_cites = []
                    for row, amount in original_purposes:
                        citation = _cite_field(row)
                        original_cites.append(citation)
                        claims.append(AnswerClaim(
                            output_id=(f"{task.task_id}.financing.{original}."
                                       f"original-purpose.{row.order}"),
                            label=f"원본 {row.label}",
                            value_text=row.value, raw_unit="원",
                            text=f"{row.label} {amount:,.0f}원",
                            citations=[citation]))
                    claims.append(AnswerClaim(
                        output_id=(f"{task.task_id}.financing.{original}."
                                   "original-total"),
                        label="원본 총 조달금액",
                        value_text=f"{original_total:,.0f}", raw_unit="원",
                        canonical_value=str(original_total),
                        canonical_unit="원",
                        text=f"{original_total:,.0f}원",
                        citations=original_cites))
                    delta = total - original_total
                    change_cites = [
                        _cite_field(row)
                        for row, _ in original_purposes + purpose_rows]
                    claims.append(AnswerClaim(
                        output_id=f"{task.task_id}.financing.{selected}.change",
                        label="원본 대비 총 조달금액 변화",
                        value_text=f"{abs(delta):,.0f}", raw_unit="원",
                        canonical_value=str(abs(delta)), canonical_unit="원",
                        state="decreased" if delta < 0 else "increased",
                        text=(f"원본 {original_total:,.0f}원 → 최신 "
                              f"{total:,.0f}원; {abs(delta):,.0f}원 "
                              f"{'감소' if delta < 0 else '증가'}"),
                        citations=change_cites))

        scope = (f"{task.corp_name}·{getattr(task.event_selector, 'event_from', '')}~"
                 f"{getattr(task.event_selector, 'event_to', '')}·주요사항보고")
        surfaces = " ".join(observed_surfaces)
        aliases = {
            "유상증자": "유상증자", "전환사채": "CB(전환사채)",
            "신주인수권부사채": "BW(신주인수권부사채)",
            "교환사채": "EB(교환사채)",
        }
        for keyword in getattr(task.event_selector, "keywords", ()) or ():
            label = aliases.get(keyword, keyword)
            confirmed = keyword in surfaces
            cites = (claims[0].citations if confirmed and claims else [])
            # A scoped zero-result is still a deterministic retrieval result;
            # cite the observed family only for positive claims.  Negative
            # claims carry the explicit scope in text and are not promoted as
            # source-proved document facts.
            if confirmed:
                claims.append(AnswerClaim(
                    output_id=f"{task.task_id}.financing.type.{keyword}",
                    label=f"{label} 확인 여부", state="confirmed",
                    text=f"{label} 확인됨", citations=cites))
            else:
                # Give the zero result a citation to the bounded query's
                # observed document family so Orchestrator retains it; the
                # wording itself remains explicitly scope-limited.
                fallback_cites = claims[0].citations if claims else []
                claims.append(AnswerClaim(
                    output_id=f"{task.task_id}.financing.type.{keyword}",
                    label=f"{label} 확인 여부", state="not_found_in_scope",
                    text=f"{scope} 범위에서 확인되지 않음",
                    citations=fallback_cites))
        if claims:
            claims.append(AnswerClaim(
                output_id=f"{task.task_id}.financing.execution_boundary",
                label="결정과 실제 조달의 구분", state="not_inferred",
                text=("위 건수와 금액은 발행을 결정한 공시 기준입니다. "
                      "결정 공시만으로 실제 납입 완료나 돈을 조달한 사실을 "
                      "확정할 수 없습니다."),
                citations=claims[0].citations))
        keywords = tuple(getattr(task.event_selector, "keywords", ()) or ())
        start = getattr(task.event_selector, "event_from", None)
        end = getattr(task.event_selector, "event_to", None)
        if len(set(keywords) & set(aliases)) == 4 and start and end:
            # An excluded type is still a member of the bounded issuer/year
            # document universe.  List its identity, never its amount as
            # financing.  Do not infer execution or economic purpose from it.
            for receipt, rows in self.fidx.iter_docs(
                    task.corp_code, as_of=task.as_of, doc_group="major"):
                if not rows or not start <= rows[0].rcept_dt <= end:
                    continue
                if self._matches_financing_types(rows, keywords):
                    continue
                citations = self._doc_cites(task.corp_code, receipt, task.as_of)
                if not citations:
                    continue
                form = rows[0].form or rows[0].report_nm
                claims.append(AnswerClaim(
                    output_id=f"{task.task_id}.financing.excluded.{receipt}",
                    label="네 유형 밖의 공시(조달 건수에서 제외)",
                    state="excluded_from_requested_types",
                    text=f"{form} · 접수번호 {receipt} · 공시일 {rows[0].rcept_dt}",
                    citations=citations))
                used.append(receipt)
        return claims, used


class CorrectionTool(BaseTool):
    _CACHE_MAX_RECEIPTS = 128
    _CACHE_MAX_ITEMS = 4096

    def __init__(self, rm, fidx=None):
        super().__init__(rm, fidx)
        self._verified_cache: OrderedDict[tuple[str, str], list] = OrderedDict()
        self._verified_cache_items = 0

    def _items_verified(self, corp_code, rcept_no, as_of):
        key = (corp_code, rcept_no)
        if key not in self._verified_cache:
            # Exact receipt predicate pushdown first; never preload an issuer's
            # complete correction/evidence inventory.
            pending = list(self.rm.correction_items(
                as_of="20991231", corp_code=corp_code, rcept_no=rcept_no))
            evidence_ids = list(dict.fromkeys(
                evidence_id for item in pending
                for evidence_id in (
                    item.before_evidence_id, item.after_evidence_id)
                if evidence_id))
            if evidence_ids:
                self.rm.get_evidence_many(evidence_ids)
            verified = list(self.rm.correction_items(
                as_of="20991231", corp_code=corp_code,
                rcept_no=rcept_no, verify_evidence=True))
            if len(verified) <= self._CACHE_MAX_ITEMS:
                self._verified_cache[key] = verified
                self._verified_cache.move_to_end(key)
                self._verified_cache_items += len(verified)
                while (len(self._verified_cache) > self._CACHE_MAX_RECEIPTS
                       or self._verified_cache_items > self._CACHE_MAX_ITEMS):
                    _, evicted = self._verified_cache.popitem(last=False)
                    self._verified_cache_items -= len(evicted)
            else:
                return [it for it in verified if it.rcept_dt <= as_of]
        else:
            self._verified_cache.move_to_end(key)
        return [it for it in self._verified_cache[key] if it.rcept_dt <= as_of]

    def _latest_semantic_status_claim(self, task, sequence):
        """Project the canonical end-state of one resolved correction lineage.

        A history answer that stops at the last document leaves users to infer
        whether the event remains ongoing.  Emit a semantic status only when
        the same resolved event identity used for the correction sequence has
        one canonical timeline and a citable state-defining observation.
        Provisional/ambiguous identities remain limited rather than upgraded.
        """

        if (task.operation != "history"
                or getattr(sequence, "identity_status", None) != "resolved"):
            return None
        try:
            timeline = self.rm.event_timeline(
                as_of=task.as_of, event_key=sequence.event_key,
                verify_evidence=True)
        except Exception:
            return None
        if (timeline is None
                or getattr(timeline, "event_key", None) != sequence.event_key
                or getattr(timeline, "corp_code", None) != task.corp_code
                or getattr(timeline, "identity_status", None) != "resolved"):
            return None
        observations = tuple(getattr(timeline, "observations", ()) or ())
        state = getattr(getattr(timeline, "state", None), "status", None)
        if not observations or state not in {"active", "terminated"}:
            return None
        if state == "terminated":
            evidence = next((row for row in reversed(observations)
                             if bool(getattr(row, "is_termination", False))), None)
            if evidence is None:
                return None
            text = (
                "제공된 공시 자료에서 해지·종료 공시가 확인되어 "
                "공시상 종료된 상태입니다.")
            public_state = state
            period_citations: list[ClaimCitation] = []
        else:
            if any(bool(getattr(row, "is_termination", False))
                   for row in observations):
                return None
            evidence = observations[-1]
            period_claims: list[AnswerClaim] = []
            for slot in ("시작일", "종료일"):
                for observation in reversed(observations):
                    row, status = self.field_value(
                        task.corp_code, observation.rcept_no, slot,
                        as_of=task.as_of, trace=[])
                    if row is None or status != "verified" or not row.value:
                        continue
                    period_claims.append(AnswerClaim(
                        output_id=f"{task.task_id}.{slot}@{task.as_of}",
                        label=slot, text=_public_field_text(row),
                        citations=[_cite_field(row)]))
                    break
            public_state, text, period_citations = (
                _active_observation_projection(
                    timepoint=task.as_of, timeline=timeline,
                    slot_claims=period_claims))
        receipt = getattr(evidence, "rcept_no", None)
        citations = (
            self._doc_cites(task.corp_code, receipt, task.as_of)
            if isinstance(receipt, str) else [])
        seen_citations = {
            (citation.doc_id, citation.rcept_no,
             citation.evidence_id, citation.locator)
            for citation in citations}
        for citation in period_citations:
            key = (citation.doc_id, citation.rcept_no,
                   citation.evidence_id, citation.locator)
            if key not in seen_citations:
                seen_citations.add(key)
                citations.append(citation)
        if not citations:
            return None
        return AnswerClaim(
            output_id=f"{task.task_id}.semantic_status@{task.as_of}",
            label="기준시점 공시 상태", state=public_state, text=text,
            citations=citations)

    def _named_event_sequence(self, task, roles):
        """Bind an explicitly named contract to one canonical event key.

        A contract-name selector is not a request to enumerate every correction
        document of the issuer.  Candidate text is used only to locate receipts;
        the result is accepted only when those receipts round-trip to exactly one
        issuer-owned typed event.  Multiple keys are result-impact ambiguity, not
        an excuse to merge their correction histories.
        """
        from app.tools.correction_sequence import (
            CorrectionSequenceResult, build_event_correction_sequence,
        )

        selector = task.event_selector
        if not (getattr(selector, "seed_rcept_no", None)
                or getattr(selector, "contract_name", None)):
            return None
        binding = self._named_contract_event_key(
            task.corp_code, selector, as_of=roles.as_of)
        if isinstance(binding, str) and len(binding) == 32 and all(
                char in "0123456789abcdef" for char in binding):
            return build_event_correction_sequence(
                self.rm, corp_code=task.corp_code, event_key=binding,
                roles=roles)
        if isinstance(binding, str):
            return CorrectionSequenceResult(
                "ambiguous" if binding == "named_contract_maps_to_multiple_events"
                else "unavailable", reason=binding)
        try:
            candidates = self.find_candidates(
                task.corp_code, selector, as_of=roles.as_of)
        except Exception:
            return CorrectionSequenceResult(
                "unavailable", reason="named_contract_candidate_lookup_error")
        if not candidates:
            return CorrectionSequenceResult(
                "not_found", reason="named_contract_candidate_not_found")

        event_keys: set[str] = set()
        for candidate in candidates:
            try:
                timeline = self.rm.event_timeline(
                    as_of=roles.as_of, rcept_no=candidate.rcept_no,
                    verify_evidence=False)
            except Exception:
                return CorrectionSequenceResult(
                    "unavailable", reason="named_contract_event_lookup_error")
            if timeline is None or getattr(timeline, "corp_code", None) != task.corp_code:
                return CorrectionSequenceResult(
                    "unavailable", reason="named_contract_event_owner_mismatch")
            event_key = getattr(timeline, "event_key", None)
            if not isinstance(event_key, str) or not event_key:
                return CorrectionSequenceResult(
                    "unavailable", reason="named_contract_event_key_missing")
            event_keys.add(event_key)
        if len(event_keys) != 1:
            return CorrectionSequenceResult(
                "ambiguous", reason="named_contract_maps_to_multiple_events")
        return build_event_correction_sequence(
            self.rm, corp_code=task.corp_code, event_key=next(iter(event_keys)),
            roles=roles)

    @staticmethod
    def _decimal_value(value: str | None) -> Decimal | None:
        if value is None:
            return None
        normalized = re.sub(r"\s*(?:원|KRW)\s*$", "", value.strip(),
                            flags=re.IGNORECASE).replace(",", "")
        if re.fullmatch(r"[-+]?[0-9]+(?:\.[0-9]+)?", normalized) is None:
            return None
        try:
            return Decimal(normalized)
        except InvalidOperation:
            return None

    @classmethod
    def _difference_text(cls, before: str | None, after: str | None) -> str:
        left, right = cls._decimal_value(before), cls._decimal_value(after)
        if left is None or right is None:
            return ""
        value = right - left
        # ``int('-0')`` is zero, so formatting a sub-unit decrease such as
        # -0.62 through the integer part alone would silently lose its sign.
        sign = "-" if value < 0 else ""
        rendered = f"{abs(value):f}"
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        integer, dot, fraction = rendered.partition(".")
        rendered = sign + f"{int(integer):,}" + (f".{fraction}" if dot else "")
        return f" (차이 {rendered})"

    def _exact_bound_sequence_root(
            self, task, sequence_result, sequences: tuple) -> str | None:
        """Return the proven root for one explicitly bound correction chain.

        A seed receipt is the strongest coordinate.  Ranged Stage1 plans use
        an exact canonical event key plus the root filing day instead; accept
        that equivalent shape only after the timeline and document-version
        lineage both round-trip.  This suppresses a misleading provisional
        identity warning for the selected chain without upgrading issuer-wide
        or date-only searches.
        """

        if (sequence_result is None
                or sequence_result.status != "resolved"
                or len(sequences) != 1):
            return None
        sequence = sequences[0]
        selector = getattr(task, "event_selector", None)
        if selector is None:
            return None
        seed_receipt = getattr(selector, "seed_rcept_no", None)
        root_receipt = (
            seed_receipt
            if isinstance(seed_receipt, str)
            and seed_receipt
            and sequence.root_receipt == seed_receipt
            else None)
        if root_receipt is None:
            event_key = getattr(selector, "event_key", None)
            event_from = getattr(selector, "event_from", None)
            event_to = getattr(selector, "event_to", None)
            if (not isinstance(event_key, str)
                    or event_key != sequence.event_key
                    or not isinstance(event_from, str)
                    or event_from != event_to
                    or event_from != sequence.root_observed_at):
                return None
            try:
                timeline = self.rm.event_timeline(
                    as_of=task.as_of, event_key=event_key,
                    verify_evidence=True)
            except Exception:
                return None
            observations = tuple(
                getattr(timeline, "observations", ()) or ())
            if (timeline is None
                    or getattr(timeline, "corp_code", None) != task.corp_code
                    or getattr(timeline, "root_rcept_no", None)
                    != sequence.root_receipt
                    or not observations
                    or observations[0].rcept_no != sequence.root_receipt
                    or observations[0].observed_at != event_from
                    or observations[0].is_correction):
                return None
            root_receipt = sequence.root_receipt
        try:
            resolution = self.rm.resolve_document_version(
                root_receipt, as_of=task.as_of)
            members = tuple(getattr(resolution, "members", ()) or ())
            step_receipts = {step.rcept_no for step in sequence.steps}
            if (resolution.status != "ok"
                    or not members
                    or root_receipt not in members
                    or not step_receipts.issubset(set(members))):
                return None
        except Exception:
            return None
        return root_receipt

    @staticmethod
    def _filter_requested_items(items: list, slots: list[str]) -> list:
        """Limit a collection to explicitly named corrected fields.

        ``before``/``after``/``difference`` are value roles, not source paths.
        A generic changed-item request keeps all paths; otherwise every named
        semantic field must match the canonical correction path.
        """

        if not slots:
            return items
        role_words = {
            "before", "after", "difference", "diff", "차이", "증감",
            "정정전", "정정후", "변경전", "변경후", "변경항목", "정정사유",
            "계약별구분",
        }
        normalized_slots: list[tuple[str, ...]] = []
        for slot in slots:
            segments = list(_path_segments(slot))
            cleaned_segments: list[str] = []
            for segment in segments:
                for role in ("정정전", "정정후", "변경전", "변경후"):
                    segment = segment.replace(role, "")
                if segment:
                    cleaned_segments.append(segment)
            value = "".join(cleaned_segments)
            if value and value not in role_words:
                normalized_slots.append(tuple(cleaned_segments))
        if not normalized_slots:
            return items
        selected = []
        for item in items:
            item_segments = _path_segments(item.path)
            def matches(slot_segments: tuple[str, ...]) -> bool:
                if item_segments == slot_segments:
                    return True
                if len(slot_segments) != 1 or not item_segments:
                    return False
                slot = slot_segments[0]
                leaf = item_segments[-1]
                if re.match(r"^[0-9]", slot):
                    # A numbered canonical coordinate names that exact field,
                    # not every longer sibling that happens to begin with it
                    # (``6. 신주 발행가액`` must not also select
                    # ``6. 신주발행가액 확정예정일``).  Unit suffixes are
                    # display metadata and remain admissible.
                    return re.fullmatch(
                        re.escape(slot) + r"(?:원|천원|백만원|십억원|억원|조원|주|건|%)?",
                        leaf,
                    ) is not None
                return slot in leaf

            if any(matches(slot_segments)
                   for slot_segments in normalized_slots):
                selected.append(item)
        return selected

    @staticmethod
    def _is_narrative_item(item) -> bool:
        tail = (_path_segments(getattr(item, "path", None)) or ("",))[-1]
        return any(token in tail for token in (
            "기타투자판단", "기타사항", "참고사항", "비고"))

    @staticmethod
    def _narrative_delta(before: str | None, after: str | None) -> str | None:
        """Summarise enumerated clause deletion/addition without document rules."""

        marker = re.compile(r"(?:^|\s)([가-힣])\.\s*")

        def clauses(value: str | None) -> dict[str, str]:
            text = value or ""
            found = list(marker.finditer(text))
            return {
                match.group(1): text[match.end():
                                     (found[index + 1].start()
                                      if index + 1 < len(found) else len(text))].strip()
                for index, match in enumerate(found)
            }

        old, new = clauses(before), clauses(after)
        if not old and not new:
            return None
        deleted = [key for key in old
                   if key not in new or "삭제" in new.get(key, "")]
        added = [key for key in new if key not in old]
        parts = []
        if deleted:
            parts.append("삭제 " + "·".join(deleted))
        if added:
            details = []
            for key in added:
                body = new[key]
                # The full cited before/after value is already emitted by the
                # caller.  This is only a compact structural summary, so omit
                # the body when no complete source unit fits rather than
                # publishing a misleading raw character fragment.
                excerpt = complete_source_excerpt(body, max_chars=220)
                details.append(f"{key}({excerpt})" if excerpt else key)
            parts.append("신설 " + ", ".join(details))
        return "; ".join(parts) or None

    def run_task(self, task, *, trace, corp_name: str):
        claims, lims, used = [], [], []
        sequence_result = None
        rc = task.document_selector.rcept_no if task.document_selector else None
        docs = [rc] if rc else []
        item_dates: dict[str, str] = {}
        # A correction receipt identifies one step, while the user-facing
        # grouping axis is the underlying contract/event.  Preserve the typed
        # sequence root as a public, deterministic contract identifier so an
        # issuer-wide history cannot collapse unrelated contracts into one
        # date-only list.  This is intentionally a receipt coordinate rather
        # than a guessed lexical contract name.
        item_contract_receipts: dict[str, str] = {}
        items = []
        if not docs and task.event_selector:
            from app.tools.correction_sequence import (
                CorrectionDateRoles, build_event_correction_sequence,
                collect_correction_sequences,
            )
            try:
                roles = CorrectionDateRoles.from_task(task)
            except ValueError as exc:
                lims.append(Limitation(
                    code="correction_date_role_invalid", detail=str(exc)))
                return claims, lims, used, None
            selected_key = getattr(task.event_selector, "event_key", None)
            direct_selector_docs = False
            if selected_key:
                selected_timeline = self.rm.event_timeline(
                    as_of=roles.as_of, event_key=selected_key,
                    verify_evidence=True)
                if (selected_timeline is not None
                        and selected_timeline.corp_code == task.corp_code
                        and selected_timeline.observations
                        and selected_timeline.observations[0].is_correction):
                    lims.append(Limitation(
                        code="source_scope_prevents_complete_lineage",
                        detail=("관측 계보의 첫 자료가 정정공시입니다. 정정 전후 값은 "
                                "제공 자료에서 처음 확인되는 값일 뿐 원공시의 최초 "
                                "금액으로 단정할 수 없습니다."),
                        affected_doc_ids=[selected_timeline.root_rcept_no]))
                sequence_result = build_event_correction_sequence(
                    self.rm, corp_code=task.corp_code,
                    event_key=selected_key, roles=roles)
            else:
                sequence_result = self._named_event_sequence(task, roles)
                named_selector = any((
                    getattr(task.event_selector, "seed_rcept_no", None),
                    getattr(task.event_selector, "contract_name", None),
                    getattr(task.event_selector, "counterparty", None),
                ))
                # Counterparty/name predicates constrain the requested
                # correction itself even when the canonical graph cannot bind
                # it to one original event.  Read only matching correction
                # receipts; never widen to every correction of the issuer.
                if (named_selector and (sequence_result is None
                                        or sequence_result.status == "ambiguous")):
                    candidates = self.find_candidates(
                        task.corp_code, task.event_selector, as_of=roles.as_of)
                    correction_candidates = [
                        candidate for candidate in candidates
                        if self._is_correction(
                            task.corp_code, candidate.rcept_no, roles.as_of)
                        and (roles.correction_from is None
                             or candidate.rcept_dt >= roles.correction_from)
                        and candidate.rcept_dt <= roles.correction_to]
                    if correction_candidates:
                        direct_selector_docs = True
                        for candidate in correction_candidates:
                            docs.append(candidate.rcept_no)
                            item_dates[candidate.rcept_no] = candidate.rcept_dt
                            items.extend(self._items_verified(
                                task.corp_code, candidate.rcept_no, roles.as_of))
                        lims.append(Limitation(
                            code="ambiguous_event_origin",
                            detail=("상대방·계약 조건에 맞는 정정공시는 확인했으나 "
                                    "복수 원공시 중 어느 계약에 귀속되는지는 "
                                    "공시만으로 특정할 수 없음"),
                            affected_doc_ids=[
                                candidate.rcept_no
                                for candidate in correction_candidates]))
                if sequence_result is None and named_selector and not direct_selector_docs:
                    lims.append(Limitation(
                        code="not_found",
                        detail="명시한 상대방·계약 조건에 맞는 정정공시가 없음"))
                    return claims, lims, used, None
                if sequence_result is None and not direct_selector_docs:
                    doc_group = (getattr(task.document_selector, "doc_group", None)
                                 if task.document_selector else None)
                    sequence_result = collect_correction_sequences(
                        self.rm, corp_code=task.corp_code, roles=roles,
                        doc_group=doc_group,
                        event_type=getattr(task.event_selector, "event_type", None))
            if direct_selector_docs:
                sequence_result = None
            if sequence_result is not None and sequence_result.status not in {"resolved", "partial"}:
                code = ("ambiguous_correction_sequence"
                        if sequence_result.status == "ambiguous"
                        else "correction_sequence_unavailable"
                        if sequence_result.status == "unavailable"
                        else "not_found")
                lims.append(Limitation(
                    code=code,
                    detail=sequence_result.reason or sequence_result.status))
                # A selected canonical event can still prove which correction
                # filings were observed even when the pre-corpus root or the
                # legal contract lineage is ambiguous.  Do not erase those
                # positive observations.  Read each correction receipt on its
                # own, label the result as an observed filing history, and
                # keep the ambiguity limitation.  No row is called the legal
                # latest valid contract.
                observation_timeline = None
                if selected_key and sequence_result.status == "ambiguous":
                    try:
                        observation_timeline = self.rm.event_timeline(
                            as_of=roles.as_of, event_key=selected_key,
                            verify_evidence=True)
                    except Exception:
                        observation_timeline = None
                if (observation_timeline is None
                        or observation_timeline.corp_code != task.corp_code):
                    return claims, lims, used, None
                observations = tuple(observation_timeline.observations or ())
                correction_observations = tuple(
                    observation for observation in observations
                    if observation.is_correction)
                if not correction_observations:
                    return claims, lims, used, None
                for index, observation in enumerate(
                        correction_observations, start=1):
                    docs.append(observation.rcept_no)
                    item_dates[observation.rcept_no] = observation.observed_at
                    items.extend(self._items_verified(
                        task.corp_code, observation.rcept_no, roles.as_of))
                    day = observation.observed_at
                    claims.append(AnswerClaim(
                        output_id=(f"{task.task_id}.observed-correction-"
                                   f"{index}"),
                        label=f"확인된 공시 이력 {index}",
                        text=f"{day[:4]}-{day[4:6]}-{day[6:]} 정정 공시",
                        citations=self._doc_cites(
                            task.corp_code, observation.rcept_no,
                            roles.as_of),
                    ))
                latest_observation = correction_observations[-1]
                latest_day = latest_observation.observed_at
                claims.append(AnswerClaim(
                    output_id=f"{task.task_id}.latest-observed-correction",
                    label="기준시점까지 확인된 마지막 공시",
                    state="corrected",
                    text=(f"{latest_day[:4]}-{latest_day[4:6]}-"
                          f"{latest_day[6:]} 정정 공시"),
                    citations=self._doc_cites(
                        task.corp_code, latest_observation.rcept_no,
                        roles.as_of),
                ))
                sequence_result = None
            if sequence_result is not None and sequence_result.status == "partial":
                partial_code = (
                    "evidence_unavailable"
                    if (sequence_result.reason or "").startswith(
                        "omitted_unverified_steps=")
                    else "ambiguous_correction_sequence")
                lims.append(Limitation(
                    code=partial_code,
                    detail=sequence_result.reason or "일부 사건 identity 미확정"))
            sequences = (() if sequence_result is None
                         else sequence_result.sequences)
            exact_bound_sequence = self._exact_bound_sequence_root(
                task, sequence_result, sequences) is not None
            for sequence in sequences:
                # A provisional issuer-wide identity remains a limitation.
                # It does not, however, prevent answering a correction
                # history that is explicitly anchored to the verified root
                # receipt and whose sequence builder closed every step.  This
                # exception uses typed coordinates only; name/date matching is
                # never promoted to an exact lineage.
                if (sequence.identity_status != "resolved"
                        and not exact_bound_sequence):
                    lims.append(Limitation(
                        code="correction_identity_provisional",
                        detail=(f"event_key={sequence.event_key}: "
                                f"identity_status={sequence.identity_status}"),
                        affected_doc_ids=[sequence.root_receipt]))
                for step in sequence.steps:
                    docs.append(step.rcept_no)
                    item_dates[step.rcept_no] = step.observed_at
                    item_contract_receipts[step.rcept_no] = sequence.root_receipt
                    items.extend(change.row for change in step.changes)
        else:
            for d in docs:
                items.extend(self._items_verified(task.corp_code, d, task.as_of))
        # ``history`` ordinarily promises the complete observable change set.
        # The one typed exception is an explicit projection role (grouping or
        # before/after/difference) alongside a real source field.  That shape
        # asks to compare only the named coordinates, so unrelated corrected
        # fields would be misleading.  A lone value/grouping role still keeps
        # the full observable history.  These capability markers distinguish
        # the shapes without a question/issuer/receipt rule.  ``diff`` remains
        # field-projected.
        requested_slots = list(
            getattr(task, "requested_slots", ()) or ())
        normalized_slots = [
            "".join(_path_segments(slot)) for slot in requested_slots]
        history_projection_roles = {
            "before", "after", "difference", "diff", "차이", "증감",
            "정정전", "정정후", "변경전", "변경후", "변경항목",
            "계약별구분",
        }
        explicit_history_projection = (
            task.operation == "history"
            and any(slot in history_projection_roles
                    for slot in normalized_slots)
            and any(slot and slot not in history_projection_roles
                    for slot in normalized_slots))
        if task.operation != "history" or explicit_history_projection:
            items = self._filter_requested_items(items, requested_slots)
        unchanged_placeholders = [
            item for item in items
            if _is_verified_unchanged_placeholder(item)]
        items = [item for item in items
                 if not _is_verified_unchanged_placeholder(item)]
        # 전체 항목 수(선언된 정정 항목)는 Evidence 검증과 무관하게 센다 — verified 목록과 동일 항목 집합
        all_items = list(items)
        trace.append(TraceEvent(seq=len(trace)+1, stage="tool", summary=f"correction_items {task.operation} → {len(items)}건",
                                detail={"docs": docs[:10],
                                        "omitted_unchanged_placeholders":
                                            len(unchanged_placeholders)}))
        if not items:
            if unchanged_placeholders:
                citations = []
                seen_evidence: set[str] = set()
                for item in unchanged_placeholders:
                    for evidence_id, locator, excerpt in (
                            (item.before_evidence_id,
                             item.before_locator or item.locator,
                             item.value_before_prompt_safe or ""),
                            (item.after_evidence_id,
                             item.after_locator or item.locator,
                             item.value_after_prompt_safe or "")):
                        if not evidence_id or evidence_id in seen_evidence:
                            continue
                        seen_evidence.add(evidence_id)
                        citations.append(ClaimCitation(
                            doc_id=item.doc_id, rcept_no=item.rcept_no,
                            evidence_id=evidence_id, locator=locator,
                            excerpt_prompt_safe=excerpt))
                    used.append(item.rcept_no)
                claims.append(AnswerClaim(
                    output_id=f"{task.task_id}.no_value_change",
                    label="정정 결과",
                    value_text="0",
                    text=("정정 전후가 모두 미기재로 확인된 항목만 있어 "
                          "공개된 값의 변경은 확인되지 않았습니다."),
                    citations=citations))
                return claims, lims, list(dict.fromkeys(used)), None
            lims.append(Limitation(code="not_found", detail=f"{corp_name}: 정정 항목 없음"))
            return claims, lims, used, None
        first_cites = None
        for it in items:
            cites = []
            before_verified = (
                it.before_evidence_status == "verified"
                and it.before_evidence_id)
            after_verified = (
                it.after_evidence_status == "verified"
                and it.after_evidence_id)
            if it.before_evidence_status == "verified" and it.before_evidence_id:
                cites.append(ClaimCitation(doc_id=it.doc_id, rcept_no=it.rcept_no, evidence_id=it.before_evidence_id,
                                           locator=it.before_locator or it.locator, excerpt_prompt_safe=it.value_before_prompt_safe or ""))
            if it.after_evidence_status == "verified" and it.after_evidence_id:
                cites.append(ClaimCitation(doc_id=it.doc_id, rcept_no=it.rcept_no, evidence_id=it.after_evidence_id,
                                           locator=it.after_locator or it.locator, excerpt_prompt_safe=it.value_after_prompt_safe or ""))
            if it.reason:
                reason_cites = self._exact_field_cites(
                    task.corp_code, it.rcept_no, "정정사유", it.reason,
                    as_of=task.as_of)
                known = {citation.evidence_id for citation in cites}
                cites.extend(
                    citation for citation in reason_cites
                    if citation.evidence_id not in known)
            if not cites:
                lims.append(Limitation(code="evidence_unavailable", detail=f"{it.rcept_no} {it.path}: 전후 Evidence 미검증", affected_doc_ids=[it.rcept_no]))
                continue
            label = it.path.split(">")[-1].strip() if it.path else "정정항목"
            oid = None
            for fo in getattr(task, "field_outputs", []) or []:
                slot_segments = _path_segments(
                    fo.slot.replace("정정후_", "").replace("_", ""))
                item_segments = _path_segments(it.path)
                if (slot_segments == item_segments
                        or (len(slot_segments) == 1 and item_segments
                            and slot_segments[0] in item_segments[-1])
                        or fo.slot in (label, it.path)):
                    oid = fo.output_id
            before = it.value_before if before_verified else None
            after = it.value_after if after_verified else None
            if first_cites is None:
                first_cites = cites
            date_prefix = (f"{item_dates[it.rcept_no][:4]}-"
                           f"{item_dates[it.rcept_no][4:6]}-"
                           f"{item_dates[it.rcept_no][6:]} "
                           if it.rcept_no in item_dates else "")
            contract_prefix = (
                f"계약 식별 접수번호 {item_contract_receipts[it.rcept_no]} · "
                if it.rcept_no in item_contract_receipts else "")
            narrative_delta = (self._narrative_delta(before, after)
                               if self._is_narrative_item(it) else None)
            table_changes = (
                _correction_table_cell_changes(before, after)
                if before is not None and after is not None else None)
            transition = (
                "; ".join(table_changes)
                if table_changes is not None
                else (f"{before if before is not None else '(정정 전 근거 미확인)'}"
                      f" → {after if after is not None else '(정정 후 근거 미확인)'}"
                      f"{self._difference_text(before, after)}"))
            claims.append(AnswerClaim(output_id=(oid if len(items) == 1 else None)
                                      or f"{task.task_id}.{it.rcept_no}.{it.order}",
                                      label=f"{contract_prefix}{date_prefix}정정 {label}",
                                      **_correction_scalar_projection(
                                          it,
                                          before_verified=before_verified,
                                          after_verified=after_verified),
                                      text=(f"[{it.reason or '사유 미기재'}] "
                                            f"{transition}"
                                            + (f" [구조 변경: {narrative_delta}]"
                                               if narrative_delta else "")),
                                      operator="correction_diff",
                                      citations=cites))
            used.append(it.rcept_no)
        # Evidence 미검증 항목의 before/after 텍스트는 다른 항목의 인용으로
        # 대체하지 않는다. 좌표별 근거가 없으면 limitation만 남긴다.
        cited_paths = {(it.rcept_no, it.path) for it in items if it.before_evidence_status == "verified" or it.after_evidence_status == "verified"}
        for it in all_items:
            if (it.rcept_no, it.path) in cited_paths: continue
            lims.append(Limitation(
                code="evidence_unavailable",
                detail=f"{it.rcept_no} {it.path}: before/after 값 미확정",
                affected_doc_ids=[it.rcept_no]))
        if claims:
            src = all_items or items
            # Long narrative clauses are reported as a separate structural
            # change, not counted as scalar "changed fields".  This preserves
            # the distinction between four edited form fields and one edited
            # explanatory paragraph without relying on a specific document.
            scalar_src = [it for it in src if not self._is_narrative_item(it)]
            n_changed = sum(1 for it in scalar_src
                            if it.diff_kind and it.diff_kind.startswith("changed"))
            claims.insert(0, AnswerClaim(output_id=f"{task.task_id}.count", label="정정 항목 수",
                                         value_text=str(n_changed or len(scalar_src)),
                                         text=f"값이 변경된 필드 {n_changed or len(scalar_src)}건"
                                              + (f", 서술형 항목 {len(src) - len(scalar_src)}건 별도"
                                                 if len(src) != len(scalar_src) else ""),
                                         citations=first_cites or []))
            reason_rows = []
            seen_reason_rows = set()
            for item in src:
                if not item.reason:
                    continue
                key = (item.rcept_no, item.reason)
                if key in seen_reason_rows:
                    continue
                seen_reason_rows.add(key)
                reason_rows.append(key)
            for index, (receipt, reason) in enumerate(reason_rows, start=1):
                reason_cites = self._exact_field_cites(
                    task.corp_code, receipt, "정정사유", reason,
                    as_of=task.as_of)
                if not reason_cites:
                    continue
                label = (
                    "정정사유" if len(reason_rows) == 1
                    else f"정정사유({receipt})"
                )
                claims.insert(index, AnswerClaim(
                    output_id=f"{task.task_id}.reason.{receipt}",
                    label=label, text=reason,
                    citations=reason_cites))

            # Bind a correction receipt to its source-document chain.  The
            # before value is quoted from the correction table, but users also
            # need the original receipt coordinate; never invent that link
            # from dates or receipt-number ordering.
            seen_document_lineages: set[tuple[str, ...]] = set()
            for correction_receipt in list(dict.fromkeys(docs)):
                resolution = self.rm.resolve_document_version(
                    correction_receipt, as_of=task.as_of)
                members = list(getattr(resolution, "members", ()) or ())
                if resolution.status != "ok" or len(members) < 2:
                    continue
                lineage = tuple(members)
                if lineage in seen_document_lineages:
                    continue
                seen_document_lineages.add(lineage)
                lineage_cites = []
                for member in members:
                    lineage_cites.extend(self._doc_cites(
                        task.corp_code, member, task.as_of))
                claims.insert(0, AnswerClaim(
                    output_id=f"{task.task_id}.{members[0]}.lineage",
                    label="문서 정정 계보",
                    text=" → ".join(members), citations=lineage_cites))
                used.extend(members)
            sequences = tuple(
                getattr(sequence_result, "sequences", ()) or ())
            if len(sequences) == 1:
                semantic_status = self._latest_semantic_status_claim(
                    task, sequences[0])
                if semantic_status is not None:
                    claims.append(semantic_status)
                    used.extend(
                        citation.rcept_no for citation in semantic_status.citations
                        if citation.rcept_no)
        return claims, lims, used, None


class DocumentTool(BaseTool):
    def _version_change_claims(
            self, *, task, chain: tuple[str, ...], output_prefix: str,
            trace: list[TraceEvent], limitations: list[Limitation],
            used: list[str]) -> list[AnswerClaim]:
        """Return only evidence-closed changes declared by correction members.

        Document version history proves which filing replaced which, but that
        relation alone says nothing about the edited fields.  Read correction
        items only from the issuer-bound receipts in the resolved chain and
        require verified before *and* after evidence before exposing a value.
        If any declared row cannot close, retain the proved lineage and state
        the missing change-detail boundary instead of guessing from a title or
        nearby filing.
        """

        reader = CorrectionTool(self.rm, self.fidx)
        claims: list[AnswerClaim] = []
        exact_change_count = 0
        unavailable_receipts: list[str] = []
        declared_count = 0
        for member in chain[1:]:
            try:
                items = reader._items_verified(
                    task.corp_code, member, task.as_of)
            except Exception:
                unavailable_receipts.append(member)
                continue
            declared_count += len(items)
            for item in items:
                if _is_verified_unchanged_placeholder(item):
                    continue
                before_verified = (
                    item.before_evidence_status == "verified"
                    and item.before_evidence_id)
                after_verified = (
                    item.after_evidence_status == "verified"
                    and item.after_evidence_id)
                if not (before_verified and after_verified):
                    if member not in unavailable_receipts:
                        unavailable_receipts.append(member)
                    continue
                before = item.value_before
                after = item.value_after
                if before is None or after is None:
                    if member not in unavailable_receipts:
                        unavailable_receipts.append(member)
                    continue
                label = (item.path.split(">")[-1].strip()
                         if item.path else "정정항목")
                citations = [
                    ClaimCitation(
                        doc_id=item.doc_id, rcept_no=item.rcept_no,
                        evidence_id=item.before_evidence_id,
                        locator=item.before_locator or item.locator,
                        excerpt_prompt_safe=item.value_before_prompt_safe),
                    ClaimCitation(
                        doc_id=item.doc_id, rcept_no=item.rcept_no,
                        evidence_id=item.after_evidence_id,
                        locator=item.after_locator or item.locator,
                        excerpt_prompt_safe=item.value_after_prompt_safe),
                ]
                if _is_reference_only_correction_pair(before, after):
                    # The table proves the edited section and stated reason,
                    # but its two cells only point to numbered notes.  Preserve
                    # that useful scope without promoting the markers to the
                    # actual before/after content.
                    claims.append(AnswerClaim(
                        output_id=(f"{output_prefix}.change-scope-{member}-"
                                   f"{item.order}"),
                        label=f"확인된 정정 대상 · {label}",
                        text=(f"[{item.reason or '정정사유 미기재'}] "
                              f"정정표 참조 표기 {before} → {after}"),
                        citations=citations,
                    ))
                    if member not in unavailable_receipts:
                        unavailable_receipts.append(member)
                    used.append(member)
                    continue
                claims.append(AnswerClaim(
                    output_id=(f"{output_prefix}.change-{member}-"
                               f"{item.order}"),
                    label=f"확인된 변경 항목 · {label}",
                    **_correction_scalar_projection(item),
                    text=(f"[{item.reason or '정정사유 미기재'}] "
                          f"{before} → {after}"),
                    operator="correction_diff",
                    citations=citations,
                ))
                exact_change_count += 1
                used.append(member)

        if not claims or unavailable_receipts:
            affected = list(dict.fromkeys(
                unavailable_receipts or list(chain[1:])))
            limitations.append(Limitation(
                code="correction_change_details_unavailable",
                detail=("정정 계보는 확인했으나 정정 전후가 모두 검증된 "
                        "구체 변경 항목을 완전하게 확인하지 못함"),
                affected_doc_ids=affected,
            ))
        trace.append(TraceEvent(
            seq=len(trace) + 1, stage="tool",
            summary=("document version changes "
                     f"declared={declared_count} exact={exact_change_count}"),
            detail={"receipts": list(chain[1:]),
                    "unavailable": unavailable_receipts},
        ))
        return claims

    def _append_provisional_correction_lineage(
            self, *, corp_code: str, rcept_no: str, as_of: str,
            limitations: list[Limitation]) -> None:
        """Expose an unresolved correction edge without guessing its root.

        A direct document ``find`` is often the only task emitted for a
        question about a correction filing.  In that shape no CorrectionTool
        sequence is run, so an otherwise visible ``CORRECTS`` edge with a
        missing/ambiguous root used to disappear from the public answer.
        Preserve that boundary here from canonical relation metadata.  The
        relation status, not a filing name/date, controls the result.
        """

        try:
            relations = self.rm.relation_summaries(
                source_rcept_no=rcept_no, as_of=as_of)
        except Exception:
            return
        for relation in relations:
            if (relation.relation_type != "CORRECTS"
                    or relation.resolution_status == "resolved"):
                continue
            affected = [rcept_no]
            destination = getattr(relation, "dst_rcept_no", None)
            if destination:
                affected.append(destination)
            limitation = Limitation(
                code="correction_identity_provisional",
                detail=("정정 관계의 원공시 연결이 "
                        f"{relation.resolution_status} 상태여서 원계약·"
                        "동일일 접수 선후를 추정하지 않음"),
                affected_doc_ids=affected)
            if not any(
                    existing.code == limitation.code
                    and existing.affected_doc_ids == limitation.affected_doc_ids
                    for existing in limitations):
                limitations.append(limitation)
            if (relation.resolution_status == "root_missing"
                    and getattr(relation, "root_missing_reason", None)
                    == "submitted_before_corpus"):
                scope_limitation = Limitation(
                    code="source_scope_prevents_complete_lineage",
                    detail=("정정 공시의 원공시가 제공된 공시 자료의 시작 전에 "
                            "제출되어 전체 원계약 계보를 복원할 수 없음"),
                    affected_doc_ids=affected)
                if not any(
                        existing.code == scope_limitation.code
                        and existing.affected_doc_ids
                        == scope_limitation.affected_doc_ids
                        for existing in limitations):
                    limitations.append(scope_limitation)

    def run_task(
            self, task, *, trace, corp_name: str,
            include_change_details: bool = False):
        claims, lims, used = [], [], []
        seeds: list[str] = []
        if task.selector and task.selector.rcept_no:
            seeds = [task.selector.rcept_no]
        elif getattr(task, "event_selector", None) and not task.event_selector.is_empty():
            seeds = [c.rcept_no for c in self.find_candidates(task.corp_code, task.event_selector, as_of=task.as_of)]
        elif task.selector:
            ds = task.selector
            for rc, rows in self.fidx.iter_docs(
                    task.corp_code, as_of=task.as_of,
                    doc_group=ds.doc_group, form=ds.form,
                    is_correction=ds.is_correction):
                f = rows[0]
                if ds.rcept_from and f.rcept_dt < ds.rcept_from: continue
                if ds.rcept_to and f.rcept_dt > ds.rcept_to: continue
                seeds.append(rc)
            # 정기공시(사업보고서 등)는 fields가 없을 수 있음 → documents에서 직접
            if not seeds:
                seeds = self._periodic_docs(task.corp_code, ds, task.as_of)
        trace.append(TraceEvent(seq=len(trace)+1, stage="tool", summary=f"document {task.operation} seeds={len(seeds)}",
                                detail={"seeds": seeds[:10]}))
        if not seeds:
            lims.append(Limitation(code="not_found", detail=f"{corp_name}: 조건에 맞는 문서 없음"))
            return claims, lims, used, None
        latest_map = {}
        for s in seeds:
            res = self.rm.resolve_document_version(s, as_of=task.as_of)
            latest_map[s] = res
            if res.status == "ambiguous":
                lims.append(Limitation(code="ambiguous_document_version", detail=f"{s}: 최신본 후보 복수 {res.candidates}", affected_doc_ids=[s]))
            elif res.status == "invalid":
                lims.append(Limitation(code="document_lineage_invalid", detail=f"{s}: 정정 계보 cycle", affected_doc_ids=[s]))
        if task.operation == "latest":
            # 사건 기준: seed들 중 as_of까지 접수된 가장 늦은 유효본
            resolved = [(s, r.selected) for s, r in latest_map.items() if r.status == "ok" and r.selected]
            if not resolved:
                lims.append(Limitation(code="not_found", detail="최신본 확정 불가")); return claims, lims, used, None
            # 각 seed의 최신본 접수일
            best = None
            for s, sel in resolved:
                dt = self._rcept_dt(sel)
                if best is None or dt > best[1]: best = (sel, dt, s)
            sel, dt, seed = best
            claims.append(AnswerClaim(output_id=f"{task.task_id}.latest", label=f"기준시점 {task.as_of} 최신 공시",
                                      text=f"{sel} (접수 {dt}, seed {seed})", state=sel,
                                      citations=self._doc_cites(task.corp_code, sel, task.as_of)))
            used.append(sel)
        elif task.operation == "version_history":
            seen_chains: set[tuple[str, ...]] = set()
            for s, r in latest_map.items():
                chain = tuple(getattr(r, "members", None) or (s,))
                if chain in seen_chains:
                    continue
                seen_chains.add(chain)
                had_corr = len(chain) > 1
                chain_cites = []
                for member in chain:
                    chain_cites.extend(self._doc_cites(
                        task.corp_code, member, task.as_of))
                claims.append(AnswerClaim(output_id=f"{task.task_id}.{s}", label="문서 정정 계보",
                                          state="corrected" if had_corr else "no_correction",
                                          text=f"{'정정 있음: ' + ' → '.join(chain) if had_corr else '기준시점까지 정정 없음'}",
                                          citations=chain_cites or self._doc_cites(
                                              task.corp_code, s, task.as_of)))
                # Emit effective-version boundaries as source-derived facts.
                # A wording model must not have to infer temporal direction
                # from the lineage arrow, which can reverse an as-of answer.
                # Same-day members deliberately get no clock ordering here;
                # canonical SUPERSEDES below is the only authority for them.
                member_dates = [self._rcept_dt(member) for member in chain]
                for member_index, member in enumerate(chain):
                    start = member_dates[member_index]
                    next_day = (
                        member_dates[member_index + 1]
                        if member_index + 1 < len(member_dates) else None)
                    if next_day is not None and start < next_day:
                        interval_text = (
                            f"{member} 버전은 {start}부터 다음 정정 접수일 "
                            f"{next_day} 전까지 유효")
                    elif next_day is None:
                        interval_text = (
                            f"{member} 버전은 {start}부터 기준시점 "
                            f"{task.as_of}까지 논리적 최신본")
                    else:
                        continue
                    claims.append(AnswerClaim(
                        output_id=(
                            f"{task.task_id}.{s}.effective-{member_index + 1}"),
                        label="버전 유효 구간",
                        state=member,
                        text=interval_text,
                        citations=self._doc_cites(
                            task.corp_code, member, task.as_of),
                    ))
                selected = getattr(r, "selected", None)
                if selected:
                    claims.append(AnswerClaim(
                        output_id=f"{task.task_id}.{s}.logical-latest",
                        label="논리적 최신본",
                        state=selected,
                        text=f"논리적 최신본은 {selected}",
                        citations=self._doc_cites(
                            task.corp_code, selected, task.as_of),
                    ))
                # Receipt-number order is not a clock.  State a logical order
                # only when canonical lineage explicitly proves SUPERSEDES,
                # and cite both endpoints of that relation.
                chain_members = set(chain)
                supersedes: list[tuple[str, str]] = []
                for source in chain:
                    for relation in self.rm.relation_summaries(
                            source_rcept_no=source, as_of=task.as_of):
                        if (relation.relation_type == "SUPERSEDES"
                                and relation.resolution_status == "resolved"
                                and relation.dst_rcept_no in chain_members):
                            edge = (relation.dst_rcept_no, source)
                            if edge not in supersedes:
                                supersedes.append(edge)
                for index, (previous, later) in enumerate(supersedes, start=1):
                    claims.append(AnswerClaim(
                        output_id=f"{task.task_id}.{s}.supersedes-{index}",
                        label="논리적 정정 순서",
                        text=(f"{previous} → {later} "
                              f"({later} 공시가 {previous} 공시를 대체)"),
                        citations=(
                            self._doc_cites(task.corp_code, previous, task.as_of)
                            + self._doc_cites(task.corp_code, later, task.as_of)),
                    ))
                    if self._rcept_dt(previous) == self._rcept_dt(later):
                        affected = [previous, later]
                        if not any(
                                limitation.code == "intraday_order_unavailable"
                                and limitation.affected_doc_ids == affected
                                for limitation in lims):
                            lims.append(Limitation(
                                code="intraday_order_unavailable",
                                detail=(
                                    "canonical SUPERSEDES는 같은 날 문서의 "
                                    "논리적 순서를 증명하지만 제공된 자료에 장중 "
                                    "접수시각이 없어 실제 제출 시각은 확인 불가"),
                                affected_doc_ids=affected,
                            ))
                if had_corr and include_change_details:
                    claims.extend(self._version_change_claims(
                        task=task, chain=chain,
                        output_prefix=f"{task.task_id}.{s}", trace=trace,
                        limitations=lims, used=used))
                used.extend(chain)
        else:  # find
            for s in seeds:
                rows = list(self.fidx.rows(
                    task.corp_code, as_of=task.as_of, rcept_no=s))
                if rows:
                    row = rows[0]
                    if "해지" in (row.form or ""):
                        document_kind = "해지공시"
                    elif row.is_correction:
                        document_kind = "정정공시"
                    else:
                        document_kind = row.form or "공시"
                    # `document_kind` is only a coarse category (form name or
                    # 정정/해지 marker).  A whole-form document collection
                    # (issue #44) needs the actual filing title to tell same-
                    # form documents apart — append it when it adds
                    # information the category alone does not already carry.
                    title = " ".join((row.report_nm or "").split())
                    if title and title != document_kind:
                        detail = (title if document_kind in title
                                  else f"{document_kind} {title}")
                    else:
                        detail = document_kind
                    text = (f"{detail} (접수번호 {s}, "
                            f"접수일 {row.rcept_dt})")
                else:
                    document_kind = "공시"
                    text = f"공시 (접수번호 {s})"
                claims.append(AnswerClaim(
                    output_id=f"{task.task_id}.{s}", label=f"문서 {s}",
                    state=document_kind, text=text,
                    citations=self._doc_cites(
                        task.corp_code, s, task.as_of)))
                used.append(s)
                if rows and row.is_correction:
                    self._append_provisional_correction_lineage(
                        corp_code=task.corp_code, rcept_no=s,
                        as_of=task.as_of, limitations=lims)
                    resolution = self.rm.resolve_document_version(
                        s, as_of=task.as_of)
                    members = list(getattr(resolution, "members", ()) or ())
                    if resolution.status == "ok" and len(members) > 1:
                        lineage_cites = []
                        for member in members:
                            lineage_cites.extend(self._doc_cites(
                                task.corp_code, member, task.as_of))
                        claims.append(AnswerClaim(
                            output_id=f"{task.task_id}.{s}.lineage",
                            label=f"{s} 정정 계보",
                            text=" → ".join(members),
                            citations=lineage_cites))
                        used.extend(members)
                    # A direct correction-document lookup asks for the filing
                    # content, not mere existence.  Return a small stable
                    # projection of its reason, core amount and explanatory
                    # context when those fields are present.
                    for slot in ("정정사유", "계약금액", "효력발생조건"):
                        value, status = self.field_value(
                            task.corp_code, s, slot,
                            as_of=task.as_of, trace=trace)
                        if (value is None or status != "verified"
                                or not value.value or value.value.strip() == "-"):
                            continue
                        excerpt = complete_source_excerpt(
                            value.value, max_chars=700)
                        if _looks_numeric(value.value) or excerpt:
                            claims.append(AnswerClaim(
                                output_id=f"{task.task_id}.{s}.{slot}",
                                label=f"{s} {slot}",
                                value_text=(value.value
                                            if _looks_numeric(value.value) else None),
                                raw_unit=_surface_unit(value.path),
                                text=(value.value if _looks_numeric(value.value)
                                      else excerpt),
                                citations=[_cite_field(value)]))
                        else:
                            lims.append(Limitation(
                                code="source_excerpt_boundary_unavailable",
                                detail=(f"{s} {slot}: 값은 검증됐지만 공개용 "
                                        "길이 안에 완결된 문장·항목 경계가 "
                                        "없어 중간 절단 없이 표시할 수 없음"),
                                affected_doc_ids=[s]))
        return claims, lims, used, None

    def _rcept_dt(self, rcept_no):
        self.rm._load_docs()
        return self.rm._rcept_dt.get(rcept_no, "")

    def _periodic_docs(self, corp_code, ds, as_of):
        self.rm._load_docs()
        out = []
        for doc_id, meta in self.rm._documents.items():
            if getattr(meta, "corp_code", None) != corp_code: continue
            rc = getattr(meta, "rcept_no", None) or doc_id.split("_")[-1]
            dt = getattr(meta, "rcept_dt", None) or self.rm._rcept_dt.get(rc, "")
            if dt > as_of: continue
            if ds.doc_group and getattr(meta, "doc_group", None) != ds.doc_group: continue
            if ds.form and ds.form not in ((getattr(meta, "report_nm", "") or "") + (getattr(meta, "form", "") or "")): continue
            if ds.rcept_from and dt < ds.rcept_from: continue
            if ds.rcept_to and dt > ds.rcept_to: continue
            out.append(rc)
        return sorted(out)


def _looks_numeric(v: str) -> bool:
    return bool(re.fullmatch(r"[\(\-]?\d[\d,]*(?:\.\d+)?\)?(\s*(원|백만원|천원|%))?", (v or "").strip()))


def _decimal_won(claim: AnswerClaim) -> Decimal | None:
    """Verified-evidence money value in 원, or ``None`` when unconfirmed.

    이슈 #59 2단계 — 거래소 사건 argmax 리덕션에 쓴다. 값이 없거나(``-``
    비공개) 숫자로 못 읽으면 ``None`` — 0원으로 취급하지 않는다.
    """
    if claim is None or claim.value_text is None:
        return None
    try:
        value = Decimal(claim.value_text.replace(",", ""))
    except InvalidOperation:
        return None
    unit = claim.raw_unit
    if unit and unit not in ("원",):
        try:
            return to_won(value, unit)
        except Exception:  # noqa: BLE001 - unknown unit stays unconfirmed
            return None
    return value
