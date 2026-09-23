"""Conservative deterministic boundary for HCX SemanticIntent v1.

The provider remains the owner of semantic candidates.  This module repairs
only values that are derivable from the candidate structure and exact question
surfaces: closed selection sentinels, explicit period/financial-scope prefixes,
scalar interrogative demand spans, and explicitly requested presentation.

It deliberately does not create entities, answer items, groups, premises, or
document/content routes.  Reviewed exceptions are limited to preserving an
already-declared existence premise as a numeric premise when its applied item
exposes exactly one numeric field surface copied verbatim from the question,
and recovering the explicit terminal ``정정된 적이 없지?`` confirmation for
one annual business-report document item.  All other semantic operations
require judgment and stay fail-closed until a separate reviewed contract
authorizes them.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import re
from typing import Any, Callable, Mapping, TypeAlias
import unicodedata

from pydantic import ValidationError

from .display_units_v1 import (
    is_display_rounding_only_surface, is_display_unit_only_surface,
)
from .hcx_schema import safe_validation_issue_codes, safe_validation_issue_paths
from .semantic_intent_v1 import (
    HCX_SEMANTIC_INTENT_WIRE_V1,
    HcxSemanticIntentWire,
    SemanticIntent,
    SemanticIntentNormalizationError,
    canonical_sha256,
    normalize_semantic_intent,
    semantic_intent_digest,
)


SEMANTIC_INTENT_BOUNDARY_V1 = "stage1-semantic-intent-boundary/1.0"

PARTICLE_DROPPED_SURFACE_REGROUNDED = "particle_dropped_surface_regrounded"
COORDINATED_HEAD_SURFACE_REGROUNDED = "coordinated_head_surface_regrounded"
UNFOUNDED_SELECTION_CRITERION_DROPPED = "unfounded_selection_criterion_dropped"
UNFOUNDED_SELECTION_DROPPED = "unfounded_selection_dropped"
SELECTION_K_RECOMPUTED = "selection_k_recomputed"
PERIOD_FROM_TARGET_QUALIFIER = "period_from_target_qualifier"
PERIOD_PREFIX_SPLIT = "period_prefix_split"
SCOPE_FROM_TARGET_QUALIFIER = "scope_from_target_qualifier"
SCOPE_PREFIX_SPLIT = "scope_prefix_split"
TARGET_SCOPE_PREFIX_REMOVED = "target_scope_prefix_removed"
FIELD_SCOPE_PREFIX_REMOVED = "field_scope_prefix_removed"
SCALAR_DEMAND_SURFACE_SELECTED = "scalar_demand_surface_selected"
ANSWER_DEMAND_SURFACE_REGROUNDED = "answer_demand_surface_regrounded"
ANSWER_STATE_SURFACES_REGROUNDED = "answer_state_surfaces_regrounded"
COMPARISON_TARGET_SURFACE_REGROUNDED = (
    "comparison_target_surface_regrounded")
COMPARISON_DEMAND_SPAN_REGROUNDED = "comparison_demand_span_regrounded"
COMPARISON_DEMAND_FIELDS_REGROUNDED = "comparison_demand_fields_regrounded"
PERCENT_CHANGE_DEMAND_SURFACE_REGROUNDED = (
    "percent_change_demand_surface_regrounded")
PERCENT_CHANGE_BASE_METRIC_TARGET_REGROUNDED = (
    "percent_change_base_metric_target_regrounded")
DATE_SURFACE_REGROUNDED = "date_surface_regrounded"
ENTITY_SURFACE_REGROUNDED = "entity_surface_regrounded"
IMPOSSIBLE_ENTITY_REFERENCES_REMOVED = (
    "impossible_entity_references_removed")
TARGET_CONCEPT_SURFACE_REGROUNDED = "target_concept_surface_regrounded"
SELECTION_CRITERION_NARROWED = "selection_criterion_narrowed"
PRESENTATION_RECOMPUTED = "presentation_recomputed"
PRESENTATION_AMBIGUOUS = "presentation_ambiguous"
NUMERIC_PREMISE_SURFACE_NARROWED = "numeric_premise_surface_narrowed"
DOCUMENT_NEGATIVE_CONFIRMATION_PREMISE_RECOVERED = (
    "document_negative_confirmation_premise_recovered")
UNGROUNDED_DOCUMENT_FIELDS_DROPPED = (
    "ungrounded_document_fields_dropped")
SCOPE_COMPARISON_ITEMS_SPLIT = "scope_comparison_items_split"
TARGET_RESTATED_FIELD_DROPPED = "target_restated_field_dropped"
TARGET_RESTATING_ENTITY_DROPPED = "target_restating_entity_dropped"
ASSERTED_QUANTITY_PREMISE_RECOVERED = "asserted_quantity_premise_recovered"
TARGET_ENTITY_PREFIX_REMOVED = "target_entity_prefix_removed"
SELECTION_RESTATING_TARGET_DROPPED = "selection_restating_target_dropped"
COMPARISON_DEMAND_SPLIT_FROM_TARGET = "comparison_demand_split_from_target"
SCOPE_STATEMENT_PREFIX_SPLIT = "scope_statement_prefix_split"
PARENTHETICAL_FORMER_NAME_MERGED = "parenthetical_former_name_merged"
UNGROUNDED_FIELD_NARROWED_TO_TARGET = "ungrounded_field_narrowed_to_target"
PERIOD_FIELDS_MOVED_TO_SCOPE = "period_fields_moved_to_scope"
COORDINATE_SEPARATOR_REGROUNDED = "coordinate_separator_regrounded"
PERCENT_CHANGE_AXIS_INHERITED = "percent_change_axis_inherited"
UNGROUNDED_DOCUMENT_GROUP_DROPPED = "ungrounded_document_group_dropped"
PERCENT_CHANGE_ITEMS_COLLAPSED = "percent_change_items_collapsed"
SINGLE_QUARTER_OUTPUT_COLLAPSED = "single_quarter_output_collapsed"
ONE_BASED_ENTITY_INDEXES_REBASED = "one_based_entity_indexes_rebased"
QUOTED_CONTRACT_SURFACE_REGROUNDED = "quoted_contract_surface_regrounded"
CORRECTION_COLLECTION_TARGET_REGROUNDED = (
    "correction_collection_target_regrounded")
HOLDING_TARGET_SURFACE_REGROUNDED = "holding_target_surface_regrounded"
DISPLAY_UNIT_QUALIFIER_DROPPED = "display_unit_qualifier_dropped"
DISPLAY_UNIT_FIELD_SURFACES_DROPPED = "display_unit_field_surfaces_dropped"
QUOTE_MARKS_DROPPED_SURFACE_REGROUNDED = "quote_marks_dropped_surface_regrounded"
TRAILING_TERMINATION_VERB_DROPPED = "trailing_termination_verb_dropped"
CONTRACT_FAMILY_TARGET_REGROUNDED = "contract_family_target_regrounded"

SEMANTIC_INTENT_PRE_SCHEMA_REPAIR_CODES = frozenset({
    SELECTION_K_RECOMPUTED,
})
SEMANTIC_INTENT_NORMALIZATION_CODES = frozenset({
    PERIOD_FROM_TARGET_QUALIFIER,
    PERIOD_PREFIX_SPLIT,
    SCOPE_FROM_TARGET_QUALIFIER,
    SCOPE_PREFIX_SPLIT,
    TARGET_SCOPE_PREFIX_REMOVED,
    FIELD_SCOPE_PREFIX_REMOVED,
    SCALAR_DEMAND_SURFACE_SELECTED,
    ANSWER_DEMAND_SURFACE_REGROUNDED,
    ANSWER_STATE_SURFACES_REGROUNDED,
    PARTICLE_DROPPED_SURFACE_REGROUNDED,
    COORDINATED_HEAD_SURFACE_REGROUNDED,
    UNFOUNDED_SELECTION_CRITERION_DROPPED,
    UNFOUNDED_SELECTION_DROPPED,
    COMPARISON_TARGET_SURFACE_REGROUNDED,
    COMPARISON_DEMAND_SPAN_REGROUNDED,
    COMPARISON_DEMAND_FIELDS_REGROUNDED,
    PERCENT_CHANGE_DEMAND_SURFACE_REGROUNDED,
    PERCENT_CHANGE_BASE_METRIC_TARGET_REGROUNDED,
    DATE_SURFACE_REGROUNDED,
    ENTITY_SURFACE_REGROUNDED,
    IMPOSSIBLE_ENTITY_REFERENCES_REMOVED,
    TARGET_CONCEPT_SURFACE_REGROUNDED,
    SELECTION_CRITERION_NARROWED,
    PRESENTATION_RECOMPUTED,
    NUMERIC_PREMISE_SURFACE_NARROWED,
    DOCUMENT_NEGATIVE_CONFIRMATION_PREMISE_RECOVERED,
    UNGROUNDED_DOCUMENT_FIELDS_DROPPED,
    SCOPE_COMPARISON_ITEMS_SPLIT,
    TARGET_RESTATED_FIELD_DROPPED,
    TARGET_RESTATING_ENTITY_DROPPED,
    ASSERTED_QUANTITY_PREMISE_RECOVERED,
    TARGET_ENTITY_PREFIX_REMOVED,
    SELECTION_RESTATING_TARGET_DROPPED,
    COMPARISON_DEMAND_SPLIT_FROM_TARGET,
    SCOPE_STATEMENT_PREFIX_SPLIT,
    PARENTHETICAL_FORMER_NAME_MERGED,
    UNGROUNDED_FIELD_NARROWED_TO_TARGET,
    PERIOD_FIELDS_MOVED_TO_SCOPE,
    COORDINATE_SEPARATOR_REGROUNDED,
    PERCENT_CHANGE_AXIS_INHERITED,
    UNGROUNDED_DOCUMENT_GROUP_DROPPED,
    PERCENT_CHANGE_ITEMS_COLLAPSED,
    SINGLE_QUARTER_OUTPUT_COLLAPSED,
    ONE_BASED_ENTITY_INDEXES_REBASED,
    QUOTED_CONTRACT_SURFACE_REGROUNDED,
    CORRECTION_COLLECTION_TARGET_REGROUNDED,
    HOLDING_TARGET_SURFACE_REGROUNDED,
    DISPLAY_UNIT_QUALIFIER_DROPPED,
    DISPLAY_UNIT_FIELD_SURFACES_DROPPED,
    QUOTE_MARKS_DROPPED_SURFACE_REGROUNDED,
    TRAILING_TERMINATION_VERB_DROPPED,
    CONTRACT_FAMILY_TARGET_REGROUNDED,
})

_NON_K_SELECTIONS = frozenset({
    "none", "latest", "earliest", "maximum", "minimum",
})
_PERIOD = re.compile(
    # An explicit event month is a source temporal coordinate.  Moving it
    # into scope preserves it for canonical event filtering instead of
    # silently falling back to the corpus cutoff.
    # 상대 연도 어휘는 백엔드(`stage1_v1_backend_composition`)와 맞춘다.
    # 여기서만 빠지면 그 말이 target qualifier 에 남아 기간으로 못 간다.
    # 긴 것부터 적어야 「재작년」이 「작년」으로 잘리지 않는다.
    r"(?:[12][0-9]{3}년(?:\s*(?:1[0-2]|0?[1-9])월|\s*[1-4]분기)?"
    r"|재작년|지난해|작년|올해|금년)")
_PERIOD_FULL = re.compile(rf"^{_PERIOD.pattern}$")
#: 「25년 연결 매출」처럼 회계연도를 두 자리로 쓰기도 한다.  날짜 파서는 이미
#: 이것을 2025 로 읽는데(`_target_date_range('25년')`) 경계에서만 못 알아봐서
#: 기간이 **필드 자리**에 앉고 그 뒤 전부가 무너졌다(`RPC-013`).
#:
#: 넓히는 자리는 여기 하나뿐이다.  `_PERIOD` 를 통째로 넓혔더니 「… 23년
#: 연료전지 계약, 25년 1월 23일이랑 4월 2일 상태가 어떻게 달라?」의 `23년` 이
#: 계약을 가리키는 수식어인데도 세 번째 기간이 되어 답이 무너졌다.  필드가
#: **전부** 기간일 때만 보는 이 검사에서는 그런 수식어가 애초에 필드로 오지
#: 않으므로 안전하다.
_FIELD_PERIOD_FULL = re.compile(
    rf"^(?:{_PERIOD.pattern}|(?<![0-9])2[0-9]년(?:\s*[1-4]분기)?)$")
_LEADING_PERIOD = re.compile(rf"^(?P<scope>{_PERIOD.pattern})\s+(?P<rest>\S.*)$")
# 비교 요구를 나타내는 낱말.  지표 이름의 일부가 아니다.
_COMPARISON_DEMANDS = ("차이", "차액", "증감", "변화", "변동")
_FINANCIAL_SCOPE_FULL = re.compile(
    r"^(?:연결\s*기준|별도\s*기준|연결|별도|개별|CFS|SFS)$",
    re.IGNORECASE,
)
_LEADING_FINANCIAL_SCOPE = re.compile(
    r"^(?P<scope>연결\s*기준|별도\s*기준|연결|별도|개별|CFS|SFS)"
    r"\s+(?P<rest>\S.*)$",
    re.IGNORECASE,
)
# 재무제표 이름은 범위와 **붙여 쓴다.**  「연결현금흐름표상 유형자산 취득」의
# `연결` 뒤에는 공백이 없어 위 패턴이 잡지 못한다.  공백을 요구하면 같은 뜻의
# 질문이 조사 하나로 갈린다 — 실제로 「연결현금흐름표**의**」는 통과하고
# 「연결현금흐름표**상**」은 지표 이름을 통째로 들고 가 실패했다.
#
# 표제부를 **재무제표 이름으로 한정**해서 가른다.  `연결` 로 시작하는 임의의
# 낱말을 쪼개지 않으므로 넓어지지 않는다.
_FINANCIAL_STATEMENT_NOUNS = (
    "포괄손익계산서", "재무상태표", "손익계산서", "현금흐름표",
    "자본변동표", "대차대조표",
)
_LEADING_SCOPED_STATEMENT = re.compile(
    r"^(?P<scope>연결|별도|개별)"
    r"(?P<statement>" + "|".join(_FINANCIAL_STATEMENT_NOUNS) + r")"
    r"(?:상|의|에서|\s)\s*(?P<rest>\S.*)$"
)
_SCALAR_DEMAND = re.compile(r"얼마나|얼마|어디|누가|무엇|뭐")
_SINGLE_QUARTER_CUE = re.compile(r"(?:단일\s*분기|단독)")
_QUARTER_SURFACE = re.compile(r"(?<![0-9])[1-4]\s*(?:분기|q)(?![0-9A-Za-z])", re.I)
_CUMULATIVE_ARITHMETIC_CUE = re.compile(
    r"누적.*(?:차이|빼|차감)|(?:차이|빼|차감).*누적")
_FULL_KOREAN_DATE = re.compile(
    r"^(?P<year>20[0-9]{2})년"
    r"(?:\s*(?P<month>1[0-2]|0?[1-9])월)?"
    r"(?:\s*(?P<day>3[01]|[12][0-9]|0?[1-9])일)?$")
_COMPARISON_TARGET = re.compile(
    r"더\s+(?:(?:많이|적게|크게|작게|높게|낮게)\s+)?"
    r"[가-힣]+?(?=(?:\s+(?:차이|격차|얼마|어느|누가|무엇|뭐|왜|어떻게)"
    r"|[,.?!]|$))")
# 「전년 대비 몇 퍼센트」를 묻는 말은 한 가지가 아니다.  「증가 또는 감소했는가」만
# 받으면 같은 요구가 「변했는가」·「증감했나」로 오면 안전망이 꺼진다.  방향을 묻는
# **변화 동사**가 붙을 때만 받으므로 「매출액 대비 영업이익률 몇 퍼센트」 같은
# 일반 비율은 여전히 걸리지 않는다.
_PERCENT_CHANGE_DEMAND = re.compile(
    r"몇\s*(?:퍼센트|퍼|%)\s*"
    r"(?:증가\s*또는\s*감소|감소\s*또는\s*증가"
    r"|증감|변동|변화|변했|늘었|줄었|증가|감소)"
    r"(?:했는가|했나|했어|하였는가|한\s*건가|인가|나요|는가|나|해)?")
_PERCENT_CHANGE_DERIVED_METRIC = re.compile(
    r"^(?P<base>.+?)(?:의)?(?:(?:전년|전기)(?:대비)?|대비)?"
    r"(?:증감|증가|감소|변동|변화)(?:률|율)?$")
_PURE_PERCENT_CHANGE_PLACEHOLDER = re.compile(
    r"^(?:%|퍼센트)?(?:증감|증가|감소|변동|변화)(?:률|율)?$")
_PERCENT_CHANGE_RATE_CUE = re.compile(
    r"(?:증감|증가|감소|변동|변화)(?:률|율)")
# 답 **필드** 자리는 비율 단위만 오기도 한다 — 「몇 퍼센트 변했나」의 답 이름을
# 모델이 그냥 「퍼센트」로 적는다.  대상 자리(`_PURE_PERCENT_CHANGE_PLACEHOLDER`)는
# 여전히 변화 명사를 요구하므로, 이 완화는 **필드 한 칸에만** 적용된다.
_RATE_ANSWER_FIELD = re.compile(
    rf"{_PURE_PERCENT_CHANGE_PLACEHOLDER.pattern[:-1]}|^(?:%|퍼센트|퍼)$")
_STATE_SURFACE = re.compile(
    r"(?:살아있(?:고|는지|어|나)?|"
    r"끝(?:난(?:\s+거야)?|났고|났는지|났어)?|"
    r"유효(?:했고|하고|한|했는지|인지|였고|였는지)?|"
    r"해지(?:됐고|되었고|된|됐는지|인지)?|"
    r"종료(?:됐고|되었고|된|됐는지|인지)?|"
    r"취소(?:됐고|되었고|된|됐는지|인지)?|"
    r"진행(?:됐고|중이고|중인지|인지)?|"
    r"완료(?:됐고|된|됐는지|인지)?)")
_PRESENTATION_PATTERNS = (
    ("table", re.compile(r"(?:표|테이블)\s*(?:로|으로|형식|형태)")),
    ("list", re.compile(r"목록\s*(?:으로|형식|형태)")),
    ("prose", re.compile(r"(?:서술|문장)\s*(?:로|으로|형식|형태)")),
)
_SELECTION_CUES = {
    "latest": re.compile(r"최신|가장\s*최근|최근"),
    "earliest": re.compile(r"최초|가장\s*이른"),
    "maximum": re.compile(r"가장\s*큰|가장\s*높은|최대|큰|높은|많은"),
    "minimum": re.compile(r"가장\s*작은|가장\s*낮은|최소|작은|낮은|적은"),
}

_HOLDING_QUESTION_TARGET = re.compile(
    r"대량\s*보유(?:상황)?\s*(?:보고서|공시)|"
    r"보유\s*주식\s*수|보유비율|지분율|"
    r"특별\s*관계자(?:별)?\s*보유\s*내역|"
    r"변동\s*주식\s*수|몇\s*주\s*변동|"
    r"주식을\s*몇\s*주\s*보유|"
    r"변동\s*(?:방법|사유)|"
    r"보유\s*주식\s*수[^,.?]{0,24}얼마나\s*변"
)


def _holding_question_target_surface(question: str) -> str | None:
    match = _HOLDING_QUESTION_TARGET.search(question)
    return match.group(0) if match is not None else None


def _holding_question_field_surfaces(question: str) -> list[str]:
    """Return reviewed literal holding demands in question order.

    This is only a grounding repair.  Canonical slot binding remains in the
    dedicated holding backend, so recognizing a phrase here cannot authorize
    a value or a document coordinate.
    """

    patterns = [
        r"특별\s*관계자(?:별)?\s*보유\s*"
        r"(?:내역|비율(?:들)?|주식\s*수(?:와\s*비율)?)",
        r"보고자(?=\s*(?:는|가|를|와|과|및|,|의\s*성명|누구|알려|확인))",
        r"변동\s*주식\s*수|증감\s*주식\s*수|몇\s*주\s*변동|"
        r"얼마나\s*(?:변했어|변했나요|변동)",
        r"보유\s*주식\s*수|보유주식수|주식을\s*몇\s*주\s*보유|"
        r"몇\s*주\s*보유",
        r"보유\s*비율|보유비율|지분\s*율|지분율",
        r"변동\s*방법|변경\s*방법",
        r"변동\s*사유|변경\s*사유",
        r"보고서\s*작성\s*기준일|작성\s*기준일|보고\s*기준일",
        r"접수\s*번호|공시\s*번호",
        r"공개(?:된)?\s*(?:법인\s*[·ㆍ/]\s*기관|법인|회사|기관)"
        r"(?:명|명칭)",
        r"개인\s*식별\s*번호|주민등록번호|외국인등록번호|여권번호",
        r"사업자등록번호|법인등록번호|(?<!접수)등록번호",
        r"개인\s*(?:성명|이름)",
        r"생년월일|출생(?:일|연월|년도)?",
        r"(?:집|자택|개인)\s*주소|거주지",
        r"전화번호|휴대전화|연락처",
    ]
    matches: list[tuple[int, str]] = []
    for pattern in patterns:
        match = re.search(pattern, question)
        if match is not None:
            matches.append((match.start(), match.group(0)))
    return list(dict.fromkeys(
        surface for _start, surface in sorted(matches, key=lambda row: row[0])
    ))

# This accepts only standalone numeric answer fields, optionally with a money
# or percentage unit.  It deliberately excludes dates (``2025년``), ranges,
# and prose such as ``약 400조원``: those require a semantic decision rather
# than a surface-preserving boundary repair.
_NUMERIC_PREMISE_FIELD_SURFACE = re.compile(
    r"^[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
    r"(?:\s*(?:%|퍼센트|bp|bps|원|천원|만원|억원|백만원|천만원|"
    r"십억원|조원|조|억))?$",
    re.IGNORECASE,
)
_ANNUAL_BUSINESS_REPORT = re.compile(r"^20[0-9]{2}년\s*사업보고서$")
_DOCUMENT_NEGATIVE_CONFIRMATION = re.compile(
    r"(?P<surface>정정된\s*적(?:이)?\s*없지)\s*\?\s*$")

CompanySurfaceRegrounder: TypeAlias = Callable[[str, str], str | None]


class SemanticIntentBoundaryError(SemanticIntentNormalizationError):
    """The candidate cannot be repaired without semantic invention."""

    def __init__(
            self, message: str, *, diagnostic_codes: tuple[str, ...],
            diagnostic_paths: tuple[str, ...] = (),
            normalization_codes: tuple[str, ...] = (),
            ) -> None:
        super().__init__(
            message,
            diagnostic_codes=diagnostic_codes,
            diagnostic_paths=diagnostic_paths,
        )
        normalized = tuple(sorted(set(normalization_codes)))
        if normalized != normalization_codes or any(
                code not in SEMANTIC_INTENT_NORMALIZATION_CODES
                for code in normalized):
            raise ValueError("boundary normalization code가 유효하지 않습니다")
        self.normalization_codes = normalized


@dataclass(frozen=True, slots=True)
class SemanticIntentBoundaryEvidence:
    """Hash-bound proof for one pre/post-schema boundary application."""

    boundary_version: str
    question_sha256: str
    source_wire_digest: str
    repaired_wire_digest: str
    semantic_intent_digest: str
    schema_repair_codes: tuple[str, ...]
    normalization_codes: tuple[str, ...]
    evidence_digest: str

    def __post_init__(self) -> None:
        if self.boundary_version != SEMANTIC_INTENT_BOUNDARY_V1:
            raise ValueError("semantic boundary evidence version이 다릅니다")
        for value in (
                self.question_sha256, self.source_wire_digest,
                self.repaired_wire_digest, self.semantic_intent_digest,
                self.evidence_digest):
            if not isinstance(value, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", value):
                raise ValueError("semantic boundary evidence digest 형식이 잘못되었습니다")
        if self.schema_repair_codes != tuple(sorted(set(
                self.schema_repair_codes))) or any(
                code not in SEMANTIC_INTENT_PRE_SCHEMA_REPAIR_CODES
                for code in self.schema_repair_codes):
            raise ValueError("semantic boundary schema repair code가 유효하지 않습니다")
        if self.normalization_codes != tuple(sorted(set(
                self.normalization_codes))) or any(
                code not in SEMANTIC_INTENT_NORMALIZATION_CODES
                for code in self.normalization_codes):
            raise ValueError("semantic boundary normalization code가 유효하지 않습니다")
        if self.evidence_digest != canonical_sha256(self._body()):
            raise ValueError("semantic boundary evidence digest가 일치하지 않습니다")

    def _body(self) -> dict[str, object]:
        return {
            "boundary_version": self.boundary_version,
            "normalization_codes": list(self.normalization_codes),
            "question_sha256": self.question_sha256,
            "repaired_wire_digest": self.repaired_wire_digest,
            "schema_repair_codes": list(self.schema_repair_codes),
            "semantic_intent_digest": self.semantic_intent_digest,
            "source_wire_digest": self.source_wire_digest,
        }

    def as_dict(self) -> dict[str, object]:
        return {**self._body(), "evidence_digest": self.evidence_digest}

    @classmethod
    def create(
            cls, bounded: "BoundedSemanticIntent", *,
            schema_repair_codes: tuple[str, ...] = (),
            ) -> "SemanticIntentBoundaryEvidence":
        if not isinstance(bounded, BoundedSemanticIntent):
            raise TypeError("bounded SemanticIntent evidence source가 잘못되었습니다")
        body = {
            "boundary_version": SEMANTIC_INTENT_BOUNDARY_V1,
            "normalization_codes": list(bounded.normalization_codes),
            "question_sha256": bounded.question_sha256,
            "repaired_wire_digest": canonical_sha256(bounded.repaired_wire),
            "schema_repair_codes": list(schema_repair_codes),
            "semantic_intent_digest": bounded.semantic_intent_digest,
            "source_wire_digest": canonical_sha256(bounded.source_wire),
        }
        return cls(
            boundary_version=SEMANTIC_INTENT_BOUNDARY_V1,
            question_sha256=bounded.question_sha256,
            source_wire_digest=str(body["source_wire_digest"]),
            repaired_wire_digest=str(body["repaired_wire_digest"]),
            semantic_intent_digest=bounded.semantic_intent_digest,
            schema_repair_codes=schema_repair_codes,
            normalization_codes=bounded.normalization_codes,
            evidence_digest=canonical_sha256(body),
        )


@dataclass(frozen=True, slots=True)
class BoundedSemanticIntent:
    """Offline proof tying one accepted wire to one deterministic repair."""

    question: str = field(repr=False)
    source_wire: HcxSemanticIntentWire = field(repr=False)
    repaired_wire: HcxSemanticIntentWire = field(repr=False)
    semantic_intent: SemanticIntent = field(repr=False)
    normalization_codes: tuple[str, ...]
    question_sha256: str
    semantic_intent_digest: str
    company_surface_regrounder: CompanySurfaceRegrounder | None = field(
        default=None, repr=False, compare=False)
    boundary_version: str = SEMANTIC_INTENT_BOUNDARY_V1

    def __post_init__(self) -> None:
        if self.boundary_version != SEMANTIC_INTENT_BOUNDARY_V1:
            raise ValueError("semantic boundary version이 정확하지 않습니다")
        exact_question = _question(self.question)
        question = _nfc(exact_question)
        if self.question_sha256 != sha256(
                exact_question.encode("utf-8")).hexdigest():
            raise ValueError("bounded question digest가 일치하지 않습니다")
        if self.normalization_codes != tuple(sorted(set(
                self.normalization_codes))):
            raise ValueError("boundary code는 정렬·중복제거되어야 합니다")
        if any(not isinstance(code, str) or not re.fullmatch(
                r"[a-z][a-z0-9_]{0,63}", code)
               for code in self.normalization_codes):
            raise ValueError("boundary code 형식이 잘못되었습니다")
        if any(code not in SEMANTIC_INTENT_NORMALIZATION_CODES
               for code in self.normalization_codes):
            raise ValueError("승인되지 않은 boundary code입니다")

        # ``model_copy(update=...)`` skips Pydantic validation.  Re-enter all
        # three strict nested contracts and reconstruct the proof instead of
        # trusting a caller-created dataclass instance.
        source = _strict_wire(self.source_wire)
        repaired = _strict_wire(self.repaired_wire)
        expected_repaired, expected_codes = _repair_roles(
            question, source,
            company_surface_regrounder=self.company_surface_regrounder,
        )
        if repaired.model_dump(mode="json", warnings=False) != \
                expected_repaired.model_dump(mode="json", warnings=False):
            raise ValueError("bounded repaired wire가 deterministic 결과와 다릅니다")
        if self.normalization_codes != expected_codes:
            raise ValueError("bounded normalization code가 실제 repair와 다릅니다")
        expected_intent = normalize_semantic_intent(question, repaired)
        actual_intent = SemanticIntent.model_validate(
            self.semantic_intent.model_dump(mode="python", warnings=False),
            strict=True,
        )
        if actual_intent.model_dump(mode="json", warnings=False) != \
                expected_intent.model_dump(mode="json", warnings=False):
            raise ValueError("bounded SemanticIntent가 repaired wire와 다릅니다")
        if self.semantic_intent_digest != semantic_intent_digest(actual_intent):
            raise ValueError("bounded semantic intent digest가 일치하지 않습니다")


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _question(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticIntentBoundaryError(
            "원 질문은 비어 있을 수 없습니다",
            diagnostic_codes=("question_invalid",),
        )
    return value


def _strict_wire(value: HcxSemanticIntentWire | Mapping[str, Any]) \
        -> HcxSemanticIntentWire:
    payload = (
        value.model_dump(mode="python", warnings=False)
        if isinstance(value, HcxSemanticIntentWire) else dict(value)
    )
    try:
        return HcxSemanticIntentWire.model_validate(payload, strict=True)
    except ValidationError as exc:
        raise SemanticIntentBoundaryError(
            "SemanticIntent source wire가 strict하지 않습니다",
            diagnostic_codes=safe_validation_issue_codes(exc),
            diagnostic_paths=safe_validation_issue_paths(exc),
        ) from exc


#: 한국어 조사. 모델이 질문의 구간을 옮겨 적으면서 **이것만** 흘리는 일이 잦다.
#: 「주가 상승의 원인」 → 「주가 상승 원인」 처럼.
#: **질문이 스스로 선택을 요구하는 단서.** 최상급·순위·시점 선택 표현이다.
#: 이것이 있으면 모델이 기준 표면을 바꿔 적었더라도 선택 의도 자체는 사용자의
#: 것이므로 모드를 지운다.
_SELECTION_CUE = re.compile(
    "|".join((
        "우세", "가장", "최대", "최소", "최고", "최저", "제일",
        "더 ?큰", "더 ?많", "더 ?높", "더 ?작", "더 ?적", "더 ?낮",
        "큰 ?곳", "큰 ?기업", "누가 ?더", "어느 ?쪽",
        "최근", "최신", "마지막", "직전", "첫", "처음",
        "상위", "하위", "top", "랭킹", "순위",
    )))


_PARTICLES = frozenset("의를을이가은는에서와과로으도만")
_SKIPPABLE = _PARTICLES | {" ", "\t", "\u3000"}


_COORDINATE_SEPARATOR = re.compile(r"\s*(?:·|ㆍ|・|및|와|과|,)\s*")


def reground_coordinate_separator(surface: str, question: str) -> "str | None":
    """가운뎃점과 「및」은 같은 이음말이다. 질문 자신의 표기 구간으로 되돌린다.

    질문이 「주요 사업과 제품·서비스를 비교해줘」인데 모델은 「제품 및 서비스」로
    적는다.  뜻은 같고 이음말 표기만 다른데, 글자가 다르다는 이유로 grounding 이
    응답 **전체**를 버린다.

    `reground_dropped_particles` 와 같은 규율을 지킨다 — 질문에 이미 있는 연속
    구간으로 되돌릴 뿐 새 뜻을 만들지 않는다.

    - 이음말 자리만 바꿔 본다. 이음말 **사이의 낱말은 그대로**여야 한다.
    - 그렇게 찾은 구간이 **정확히 하나**여야 한다. 둘 이상이면 되돌리지 않는다.
    """

    normalized = _nfc(surface or "").strip()
    if not normalized or normalized in question:
        return None
    parts = [part for part in _COORDINATE_SEPARATOR.split(normalized)]
    if len(parts) < 2 or not all(part.strip() for part in parts):
        return None
    pattern = _COORDINATE_SEPARATOR.pattern.join(
        re.escape(part.strip()) for part in parts)
    found = list(dict.fromkeys(
        match.group(0) for match in re.finditer(pattern, question)))
    return found[0] if len(found) == 1 else None


_COORDINATION_GAP = re.compile(
    r"^\s*(?P<sibling>[^\s,·ㆍ及]+?)\s*(?:과|와|및|,|·|ㆍ)\s*$")


def reground_coordinated_head(surface: str, question: str) -> "tuple[str, str] | None":
    """「반도체 부문 생산능력과 가동률」을 두 항목으로 나눈 모델이 두 번째 항목에
    수식어를 다시 붙여 「반도체 부문 가동률」로 적으면, 그 표면은 질문에 없다.

    되돌리는 규칙은 하나다 — 표면이 「수식어 + 머리말」이고 질문에 **수식어, 형제
    한 낱말, 이음말(과·와·및·,·가운뎃점), 머리말**이 그 순서로 정확히 한 번 있을 때,
    머리말을 표면으로 쓰고 수식어를 한정어로 돌려준다. 형제 자리에 낱말이 둘 이상
    끼거나 그런 구간이 둘이면 되돌리지 않는다 — 다른 지표를 하나로 묶어 오답을
    만드는 것이 조사 하나 흘린 것보다 훨씬 나쁘다.

    돌려주는 것은 ``(머리말, 수식어)`` 이고 둘 다 질문의 연속 구간이다.
    """

    normalized = _nfc(surface or "").strip()
    grounded = _nfc(question or "")
    if not normalized or normalized in grounded:
        return None
    tokens = normalized.split()
    if len(tokens) < 2:
        return None
    head = tokens[-1]
    modifier = " ".join(tokens[:-1])
    if head not in grounded or modifier not in grounded:
        return None
    spans: list[tuple[str, str]] = []
    for match in re.finditer(re.escape(modifier), grounded):
        rest = grounded[match.end():]
        head_at = rest.find(head)
        if head_at < 0:
            continue
        gap = rest[:head_at]
        if _COORDINATION_GAP.match(gap) is None:
            continue
        spans.append((head, modifier))
    if len(spans) != 1:
        return None
    return spans[0]


def reground_dropped_particles(surface: str, question: str) -> "str | None":
    """근거 없는 표면이 **질문의 실제 구간에서 조사·공백만 흘린 것**이면 그 구간.

    없으면 ``None`` — 그때는 지금처럼 거부한다.

    라이브 3회 반복에서 관측된 근거 실패는 셋으로 갈렸다.

    ```
    '주가 상승 원인'   질문 「주가 상승의 원인」      조사 「의」만 누락   ← 여기서 고친다
    '달라진 것'        질문 「무엇이 달라졌는가」      동사를 명사로 재구성  ← 고치지 않는다
    '해지 사유'        질문에 선택 기준이 없음         기준을 지어냄        ← 고치지 않는다
    ```

    **재구성과 발명은 되돌리지 않는다.** 이 함수는 질문에 이미 있는 연속 구간으로
    되돌릴 뿐 새 뜻을 만들지 않는다. 그래서 세 조건을 모두 건다.

    - 표면의 **첫 글자에서 정확히 시작**한다. 앞 조사를 함께 먹으면 같은 표면에
      대해 구간이 여럿 생겨 무엇으로 되돌릴지 정해지지 않는다.
    - 건너뛰는 글자는 **조사와 공백뿐**이다. 다른 글자를 건너뛰면 뜻이 바뀐다.
    - 그렇게 찾은 구간이 **정확히 하나**여야 한다. 둘 이상이면 되돌리지 않는다.
    """

    normalized = unicodedata.normalize("NFC", surface)
    grounded = unicodedata.normalize("NFC", question)
    if not normalized.strip() or normalized in grounded:
        return None
    needle = [character for character in normalized if not character.isspace()]
    if not needle:
        return None
    spans: set[str] = set()
    for start in range(len(grounded)):
        if grounded[start] != needle[0]:
            continue
        index, matched = start, 0
        while index < len(grounded) and matched < len(needle):
            if grounded[index] == needle[matched]:
                matched += 1
                index += 1
            elif grounded[index] in _SKIPPABLE:
                index += 1
            else:
                break
        if matched == len(needle):
            spans.add(grounded[start:index])
    spans = {span for span in spans if span.strip()}
    if len(spans) != 1:
        return None
    return next(iter(spans))


def reground_quoted_contract_surface(
        surface: str, question: str) -> "str | None":
    """Recover a quoted name when HCX moves ``계약`` inside the quote."""

    normalized = _nfc(surface or "").strip("'\"‘’“” ")
    if not normalized or normalized in question:
        return None
    surface_key = re.sub(r"\s+", "", normalized)
    found: list[str] = []
    for match in re.finditer(
            r"(?P<quote>['\"‘“])(?P<value>[^'\"’”]{2,300})['\"’”]\s*계약",
            question):
        value = match.group("value").strip()
        if re.sub(r"\s+", "", value + "계약") == surface_key:
            found.append(value)
    found = list(dict.fromkeys(found))
    return found[0] if len(found) == 1 else None


_QUOTE_MARK = frozenset("「」『』'\"'‘’“”")


def reground_dropped_quote_marks(surface: str, question: str) -> "str | None":
    """SG-008 실호출(이슈 #74·#75 후속, 2026-09-03): HCX가 「신규시설투자」
    공시처럼 낫표로 서식명을 인용한 질문의 target을 낫표만 뗀 채
    ``신규시설투자 공시``로 낸다.  질문에는 낫표가 그대로 있어 이 표면이 문자
    그대로는 없다.

    `reground_dropped_particles`와 규율은 같다 — 질문에 이미 있는 연속 구간
    으로만 되돌리고, 건너뛰는 글자는 조사·공백에 낫표/따옴표류 인용부호를 더한
    것뿐이다.  구간이 둘 이상이면 되돌리지 않는다.
    """

    normalized = _nfc(surface or "").strip()
    grounded = _nfc(question or "")
    if not normalized or normalized in grounded:
        return None
    needle = [character for character in normalized if not character.isspace()]
    if not needle:
        return None
    skippable = _SKIPPABLE | _QUOTE_MARK
    spans: set[str] = set()
    for start in range(len(grounded)):
        if grounded[start] != needle[0]:
            continue
        index, matched = start, 0
        while index < len(grounded) and matched < len(needle):
            if grounded[index] == needle[matched]:
                matched += 1
                index += 1
            elif grounded[index] in skippable:
                index += 1
            else:
                break
        if matched == len(needle):
            # Extend the span backward through any immediately preceding
            # skippable mark (예: 낫표 여는 괄호) so an opening quote right
            # before the matched text is not silently orphaned — the needle
            # itself never contains it (it starts at the first non-space
            # character of ``surface``), only the forward walk could consume
            # a closing one.
            span_start = start
            while span_start > 0 and grounded[span_start - 1] in _QUOTE_MARK:
                span_start -= 1
            spans.add(grounded[span_start:index])
    spans = {span for span in spans if span.strip()}
    if len(spans) != 1:
        return None
    return next(iter(spans))


_TRAILING_TERMINATION_VERB = re.compile(
    r"\s*(?:해지|종료|취소|철회|파기)\s*$")


def reground_trailing_termination_verb(surface: str, question: str) -> "str | None":
    """SG-006 실호출(이슈 #74 후속, 2026-09-03): "한미반도체가 2025년에 체결한
    단일판매·공급계약 중 이후 해지된 계약이 존재하는가?" 처럼 계약명을 지정하지
    않는 개방형 해지-존재 질의에서, HCX가 target을 ``단일판매·공급계약 해지``
    로 압축한다.  질문에는 "계약"과 "해지" 사이에 "중 이후 …된"이 끼어 있어
    이 압축형은 문자 그대로 없다.

    존재 판정 자체는 target.surface가 아니라 질문 전체에서 다시 읽으므로
    (``_root_contract_termination_membership``·``_SET_EXISTENCE``), 트레일링
    해지류 동사 하나만 걷어 낸 나머지가 질문의 연속 구간이면 그것으로
    충분하다 — 새 뜻을 만들지 않고 압축으로 붙은 동사만 뗀다.
    """

    normalized = _nfc(surface or "").strip()
    if not normalized or normalized in question:
        return None
    trimmed = _TRAILING_TERMINATION_VERB.sub("", normalized).strip()
    if not trimmed or trimmed == normalized or trimmed not in question:
        return None
    return trimmed


_CONTRACT_FAMILY_SPAN = re.compile(r"단일\s*판매\s*[·ㆍ・]?\s*공급\s*계약")


def reground_contract_family_target(surface: str, question: str) -> "str | None":
    """Last-resort fallback for a still-ungrounded 단일판매·공급계약 target.

    SG-006 후속 실호출(이슈 #74 후속, 2026-09-03): ``reground_trailing_termination_verb``
    는 압축형이 정확히 해지류 동사로 끝날 때만 되돌린다.  같은 wire가 다른
    호출에서 "…단일판매·공급계약 중 해지된 계약"처럼 "계약"으로 끝나거나
    어순이 또 다르게 압축되면 그 함수도, 다른 어떤 조사·낫표 복구도 닿지
    못하고 ``surface_not_grounded``로 기각된다 — 압축의 모양이 매번 달라
    트레일링 동사 하나만 떼는 규칙으로는 다 못 잡는다.

    이 함수는 압축의 정확한 모양을 더는 따지지 않는다.  ungrounded 표면이
    이미 "단일판매·공급계약"(가운뎃점 표기 무관) 패밀리 명사를 담고 있고,
    질문에 그 명사의 리터럴 구간이 **정확히 하나** 있으면, 압축으로 붙은
    나머지 낱말(체결·중·이후·해지·된·계약·여부 …)을 전부 버리고 그 구간
    하나로 되돌린다.  존재 판정 자체는 target.surface가 아니라 질문
    전체에서 다시 읽으므로(``_root_contract_termination_membership``·
    ``_SET_EXISTENCE``), 압축이 뭘 버렸든 이 하나의 닫힌 DART 이벤트
    패밀리 명사만 있으면 충분하다 — 새 뜻을 만들지 않고, 질문에 없는
    다른 패밀리로 넓히지도 않는다.
    """

    normalized = _nfc(surface or "").strip()
    if not normalized or normalized in question:
        return None
    if _CONTRACT_FAMILY_SPAN.search(normalized) is None:
        return None
    spans = list(dict.fromkeys(
        match.group(0) for match in _CONTRACT_FAMILY_SPAN.finditer(question)))
    return spans[0] if len(spans) == 1 else None


_CORRECTION_COLLECTION_FAMILY = re.compile(
    r"단일\s*판매\s*[·ㆍ・]?\s*공급\s*계약\s*정정\s*공시")


def _reground_correction_collection_target(
        payload: dict[str, Any], item: dict[str, Any], question: str,
        ) -> str | None:
    """Restore the literal correction-disclosure family for a closed query.

    HCX occasionally paraphrases only the event target while keeping the
    issuer-wide collection and requested fields intact.  Replacement is
    allowed only when the question independently states the issuer, corpus
    cutoff, contract grouping, before/after amount and difference axes.  The
    returned surface is the unique literal family span from the question.
    """

    target = item.get("target")
    if not isinstance(target, dict) or target.get("kind") not in {
            "event", "document"}:
        return None
    source = _nfc(str(target.get("surface", ""))).strip()
    compact = re.sub(r"[\s·ㆍ・]+", "", source)
    if (source in question or "정정" not in compact
            or not any(token in compact for token in ("계약", "공급", "판매"))):
        return None
    referenced = set(target.get("entity_indexes", []))
    company_indexes = {
        index for index, entity in enumerate(payload.get("entities", []))
        if isinstance(entity, dict) and entity.get("kind_hint") == "company"
        and isinstance(entity.get("surface"), str)
        and entity["surface"] in question
    }
    matches = list(dict.fromkeys(
        match.group(0) for match in _CORRECTION_COLLECTION_FAMILY.finditer(
            question)))
    if (len(referenced & company_indexes) != 1 or len(matches) != 1
            or re.search(r"코퍼스\s*기준일(?:까지)?", question) is None
            or re.search(r"계약\s*별(?:로)?\s*구분", question) is None
            or re.search(r"정정\s*전후\s*계약\s*금액", question) is None
            or re.search(r"차이|증감", question) is None):
        return None
    return matches[0]


def _rebase_unambiguous_one_based_entity_indexes(
        payload: dict[str, Any]) -> bool:
    """Convert only a wire that is globally and provably one-based."""

    size = len(payload.get("entities", []))
    refs = [
        index
        for item in payload.get("answer_items", [])
        for index in item.get("target", {}).get("entity_indexes", [])
    ]
    if (size <= 0 or not refs or size not in refs
            or any(type(index) is not int or index < 1 or index > size
                   for index in refs)):
        return False
    for item in payload.get("answer_items", []):
        target = item.get("target", {})
        target["entity_indexes"] = [
            index - 1 for index in target.get("entity_indexes", [])]
    return True


def repair_semantic_intent_wire_payload(
        value: Any,
        ) -> tuple[Any, tuple[str, ...]]:
    """Repair only selection ``k`` values fully derived from ``mode``.

    This pre-schema helper never creates semantic rows or user surfaces.  It is
    intentionally not connected to the live provider client yet.
    """

    if not isinstance(value, dict) \
            or value.get("schema_version") != HCX_SEMANTIC_INTENT_WIRE_V1:
        return value, ()
    items = value.get("answer_items")
    if not isinstance(items, list):
        return value, ()
    repaired = deepcopy(value)
    changed = False
    for item in repaired.get("answer_items", []):
        if not isinstance(item, dict):
            continue
        selection = item.get("selection")
        if not isinstance(selection, dict):
            continue
        mode = selection.get("mode")
        criterion = selection.get("criterion_surface")
        current_k = selection.get("k")
        if (
            mode in _NON_K_SELECTIONS
            and type(current_k) is int
            and current_k != 0
            and (
                mode != "none"
                or criterion in {None, ""}
            )
        ):
            selection["k"] = 0
            changed = True
    return (
        repaired,
        (SELECTION_K_RECOMPUTED,) if changed else (),
    )


def _ordered_unique(values: list[str], question: str) -> list[str]:
    rows = list(dict.fromkeys(values))
    return sorted(rows, key=lambda row: (
        question.find(row) if row in question else len(question), row))


def _strip_edge(value: str, surface: str, question: str) -> str:
    """Remove one exact prefix/suffix only when the remainder is grounded."""

    candidate = value
    if candidate.startswith(surface):
        remainder = candidate[len(surface):].strip()
    elif candidate.endswith(surface):
        remainder = candidate[:-len(surface)].strip()
    else:
        return value
    return remainder if remainder and remainder in question else value


def _explicit_presentation(question: str) -> str:
    matches = [name for name, pattern in _PRESENTATION_PATTERNS
               if pattern.search(question)]
    unique = tuple(dict.fromkeys(matches))
    if len(unique) > 1:
        raise SemanticIntentBoundaryError(
            "서로 다른 명시 presentation을 자동 배분할 수 없습니다",
            diagnostic_codes=(PRESENTATION_AMBIGUOUS,),
        )
    return unique[0] if unique else "auto"


def _unique_scalar_demand(question: str) -> str | None:
    spans = _ordered_scalar_demands(question)
    return spans[0] if len(spans) == 1 else None


def _explicit_single_quarter_arithmetic(question: str) -> bool:
    """Whether the user explicitly asks for one discrete-quarter result."""

    return (
        _SINGLE_QUARTER_CUE.search(question) is not None
        and _QUARTER_SURFACE.search(question) is not None
        and _CUMULATIVE_ARITHMETIC_CUE.search(question) is not None
    )


def _single_quarter_metric_surface(mention: str, question: str) -> str | None:
    """Return one literal metric alias shared by target and question."""

    from .planning import METRIC_ALIASES

    compact_mention = re.sub(r"[^0-9A-Za-z가-힣]", "", mention).casefold()
    compact_question = re.sub(
        r"[^0-9A-Za-z가-힣]", "", question).casefold()
    candidates: list[tuple[int, str, object]] = []
    for alias, concept in METRIC_ALIASES.items():
        if (not alias or alias not in compact_mention
                or alias not in compact_question):
            continue
        if alias in question.casefold():
            candidates.append((len(alias), alias, concept))
    concepts = {row[2] for row in candidates}
    if len(concepts) != 1 or not candidates:
        return None
    longest = max(row[0] for row in candidates)
    surfaces = sorted({row[1] for row in candidates if row[0] == longest})
    return surfaces[0] if len(surfaces) == 1 else None


def _ordered_scalar_demands(question: str) -> list[str]:
    return [match.group(0) for match in _SCALAR_DEMAND.finditer(question)]


def _comparison_demand_span(question: str) -> str | None:
    matches = list(_SCALAR_DEMAND.finditer(question))
    if len(matches) < 2:
        return None
    span = question[matches[0].start():matches[-1].end()]
    return span if span and span in question else None


def _unique_percent_change_demand(question: str) -> str | None:
    """Return one literal increase/decrease rate demand, if unambiguous.

    Provider wires regularly abbreviate this user-facing request as
    ``증감률``.  That label is useful semantically but cannot cross the exact
    grounding boundary.  The correction is safe only for the explicit
    two-sided percentage grammar below; it never infers a rate demand from a
    generic comparison or rewrites multiple possible spans.
    """

    matches = list(_PERCENT_CHANGE_DEMAND.finditer(question))
    surfaces = list(dict.fromkeys(match.group(0) for match in matches))
    return surfaces[0] if len(surfaces) == 1 else None


def _question_date_surface(value: str, question: str) -> str | None:
    """Map one expanded Korean date back to one exact abbreviated span."""

    match = _FULL_KOREAN_DATE.fullmatch(value)
    if match is None:
        if re.fullmatch(r"20[0-9]{2}", value):
            surface = f"{value}년"
            return surface if question.count(surface) == 1 else None
        return None
    year = match.group("year")
    month = match.group("month")
    day = match.group("day")
    short_year = year[2:]
    patterns: list[str] = []
    if month is not None and day is not None:
        patterns.extend((
            rf"(?<![0-9]){short_year}년\s*0?{int(month)}월\s*0?{int(day)}일",
            rf"(?<![0-9])0?{int(month)}월\s*0?{int(day)}일",
            rf"(?<![0-9])0?{int(day)}일",
        ))
    elif month is not None:
        patterns.extend((
            rf"(?<![0-9]){short_year}년\s*0?{int(month)}월",
            rf"(?<![0-9])0?{int(month)}월",
        ))
    else:
        patterns.append(rf"(?<![0-9]){short_year}년")
    for pattern in patterns:
        matches = [row.group(0) for row in re.finditer(pattern, question)]
        unique = list(dict.fromkeys(matches))
        if len(unique) == 1:
            return unique[0]
    return None


def _unique_comparison_target(question: str) -> str | None:
    matches = [row.group(0) for row in _COMPARISON_TARGET.finditer(question)]
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


def _ordered_state_surfaces(question: str) -> list[str]:
    return list(dict.fromkeys(
        row.group(0) for row in _STATE_SURFACE.finditer(question)))


def _question_metric_surface(mention: str, question: str) -> str | None:
    """Return one approved colloquial metric surface from the question.

    The provider often canonicalizes ``팔았고`` to ``매출액``.  That meaning is
    useful, but normalized SemanticIntent surfaces must still be literal spans
    from the question.  Reuse the reviewed concept registry and only regress
    to the question surface when both sides resolve to the same single concept.
    """

    from .concept_alias import resolve_from_question
    from .planning import CONCEPT_QUESTION_PATTERNS, resolve_metric_concept

    canonical = resolve_metric_concept(mention)
    if canonical is None:
        return None
    outcome = resolve_from_question(question, CONCEPT_QUESTION_PATTERNS)
    if (outcome.status != "resolved" or outcome.concept is not canonical
            or not isinstance(outcome.surface, str)):
        return None
    surface = _nfc(outcome.surface)
    return surface if surface and surface in question else None


def _period_qualified_question_metric_surface(question: str) -> str | None:
    """Return one literal metric surface after a moved period loses its target.

    A provider can put a canonicalized metric phrase in ``target.surface`` and
    the literal year in ``target.qualifier_surfaces``.  Once that year moves
    into Scope, the original target may still be outside the question and
    otherwise makes the whole invocation fail grounding before the native
    financial regrounder can run.  Recover only a *resolved* approved
    colloquial surface, and reject it when the question also carries a
    different direct account-map concept.  This is intentionally unavailable
    to ordinary ungrounded targets: a literal period must already have moved.
    """

    from .concept_alias import normalize_surface_key, resolve_from_question
    from .planning import CONCEPT_QUESTION_PATTERNS, METRIC_ALIASES

    outcome = resolve_from_question(question, CONCEPT_QUESTION_PATTERNS)
    if (outcome.status != "resolved" or outcome.concept is None
            or not isinstance(outcome.surface, str)):
        return None
    surface = _nfc(outcome.surface)
    if not surface or surface not in question:
        return None
    question_key = normalize_surface_key(question)
    direct_concepts = {
        concept for alias, concept in METRIC_ALIASES.items()
        if alias and alias in question_key
    }
    if direct_concepts and direct_concepts != {outcome.concept}:
        return None
    return surface


def _collapse_percent_change_items(payload: dict[str, Any]) -> bool:
    """전년 대비 변화 요구를 **항목 하나로** 합친다.

    「2025년 매출이 2024년 대비 얼마나, 몇 퍼센트 변했나」는 의미상 한 요구다 —
    한 지표의 두 기간 사이 변화. 모델은 이것을 조회+비교 두 항목으로도,
    조회+조회+비교 세 항목으로도 적는다.

    동결 Gold 가 쓰는 모양은 **compare 한 항목**이다.

    ```text
    op=compare kind=metric surface='매출액' shape=comparison proj=whole_target
    fields=[] period=['2025년','2024년'] scope=['연결']
    ```

    쪼개진 채로 두면 compiler 에 그 구조를 받는 handler 가 없어 **질문 전체가 닫힌다.**
    합치는 것은 파생 지표 복구가 「같은 지표·같은 개체축」임을 이미 증명한 뒤이므로
    새 뜻을 만들지 않는다.

    비교 항목이 정확히 하나이고 나머지가 전부 조회이며, 모두 같은 지표 표면·같은
    개체축일 때만 합친다. 하나라도 어긋나면 그대로 둔다.
    """

    items = payload.get("answer_items") or []
    if len(items) < 2:
        return False
    compares = [row for row in items
                if isinstance(row, dict) and row.get("operation") == "compare"]
    retrieves = [row for row in items
                 if isinstance(row, dict) and row.get("operation") == "retrieve"]
    if len(compares) != 1 or not retrieves \
            or len(compares) + len(retrieves) != len(items):
        return False
    compare = compares[0]
    axis = (compare.get("target") or {}).get("entity_indexes")
    surfaces = {_nfc(str((row.get("target") or {}).get("surface") or ""))
                for row in items}
    if len(surfaces) != 1 or any(
            (row.get("target") or {}).get("entity_indexes") != axis
            for row in items):
        return False
    periods: list[str] = []
    scopes: list[str] = []
    for row in items:
        scope = row.get("scope") or {}
        for value in scope.get("target_period_expressions") or ():
            if value not in periods:
                periods.append(value)
        for value in scope.get("scope_qualifier_expressions") or ():
            if value not in scopes:
                scopes.append(value)
    if len(periods) < 2:
        return False
    compare["scope"] = {
        "target_period_expressions": periods,
        "as_of_expression": (compare.get("scope") or {}).get("as_of_expression") or "",
        "document_group_expression": "",
        "scope_qualifier_expressions": scopes,
    }
    compare["output"] = {
        "shape": "comparison",
        "projection_mode": "whole_target",
        "field_surfaces": [],
        "presentation": (compare.get("output") or {}).get("presentation") or "auto",
    }
    compare["selection"] = {"mode": "none", "criterion_surface": "", "k": 0}
    payload["answer_items"] = [compare]
    return True


def _inherit_percent_change_axis(payload: dict[str, Any]) -> bool:
    """전년 대비 변화 항목이 **기준 항목의 기간·범위를 물려받는다.**

    파생 지표 복구가 비교 항목의 지표를 기준 지표로 되돌리고 나면, 두 항목은
    같은 개체축의 **같은 계열**이다.  그런데 모델이 비교 항목의 기간·범위를 비워
    보내면 그 자리가 빈 채로 남아 백엔드가 물러난다 — 지표는 맞는데 「어느 기간의
    무슨 기준」인지가 없기 때문이다.

    비교 항목이 **정확히 하나**이고, 조회 항목들이 **같은 지표 표면**을 가리키며,
    비교 항목의 그 자리가 **비어 있을 때만** 채운다.  모델이 적어 보낸 값은
    덮어쓰지 않는다.
    """

    items = payload.get("answer_items") or []
    compares = [item for item in items
                if isinstance(item, dict) and item.get("operation") == "compare"]
    retrieves = [item for item in items
                 if isinstance(item, dict) and item.get("operation") == "retrieve"]
    if len(compares) != 1 or not retrieves \
            or len(compares) + len(retrieves) != len(items):
        return False
    compare = compares[0]
    axis = (compare.get("target") or {}).get("entity_indexes")
    surfaces = {_nfc(str((item.get("target") or {}).get("surface") or ""))
                for item in retrieves}
    compare_surface = _nfc(str((compare.get("target") or {}).get("surface") or ""))
    if len(surfaces) != 1 or compare_surface not in surfaces or any(
            (item.get("target") or {}).get("entity_indexes") != axis
            for item in retrieves):
        return False
    donor_periods: list[str] = []
    donor_scopes: list[str] = []
    for item in retrieves:
        scope = item.get("scope") or {}
        for value in scope.get("target_period_expressions") or ():
            if value not in donor_periods:
                donor_periods.append(value)
        for value in scope.get("scope_qualifier_expressions") or ():
            if value not in donor_scopes:
                donor_scopes.append(value)
    scope = compare.setdefault("scope", {})
    changed = False
    if not (scope.get("target_period_expressions") or ()) and donor_periods:
        scope["target_period_expressions"] = list(donor_periods)
        changed = True
    if not (scope.get("scope_qualifier_expressions") or ()) and donor_scopes:
        scope["scope_qualifier_expressions"] = list(donor_scopes)
        changed = True
    return changed


def _move_period_only_fields_to_scope(
        payload: dict[str, Any], question: str, codes: set[str]) -> bool:
    """**기간은 답 필드가 아니다.** 필드가 전부 기간이면 scope 로 옮긴다.

    「2025년 매출이 2024년 대비 얼마나 변했나」를 모델이
    `field_surfaces=['2025년','2024년']` 로 읽는 판이 있다.  기간이 필드에 앉으면
    항목이 「무엇을 답할지」를 잃고, 파생 지표 복구도 필드가 하나가 아니라는
    이유로 물러난다.

    **필드가 전부 기간일 때만** 옮긴다.  하나라도 실제 필드가 섞여 있으면 모델이
    기간을 필드로 착각한 것이 아니므로 건드리지 않는다.  비운 자리는 지표 자신으로
    채운다 — 이미 있는 whole-target 관례와 같다.
    """

    changed = False
    for item in payload.get("answer_items", []):
        if not isinstance(item, dict):
            continue
        output = item.get("output") or {}
        fields = output.get("field_surfaces")
        target_surface = _nfc(str((item.get("target") or {}).get("surface") or ""))
        if (
                not isinstance(fields, list)
                or not fields
                or not target_surface
                or not all(_FIELD_PERIOD_FULL.fullmatch(_nfc(str(value)).strip())
                           for value in fields)
        ):
            continue
        scope = item.setdefault("scope", {})
        periods = list(scope.get("target_period_expressions") or [])
        for value in fields:
            moved = _nfc(str(value)).strip()
            if moved not in periods:
                periods.append(moved)
        scope["target_period_expressions"] = _ordered_unique(periods, question)
        output["field_surfaces"] = [target_surface]
        changed = True
    return changed


def _drop_ungrounded_document_groups(
        payload: dict[str, Any], question: str, codes: set[str]) -> bool:
    """Drop only provider-added document-family surfaces absent from input.

    A financial question may say ``연결기준`` while the provider invents the
    more specific string ``연결재무제표`` as a document group.  Rejecting the
    whole wire would also discard its valid companies, metric, period and
    comparison demand.  Document selection remains canonical preflight's
    responsibility, so removing this optional hint is fail-closed.  This must
    not depend on whether the output fields happen to be periods.
    """

    changed = False
    for item in payload.get("answer_items", []):
        if not isinstance(item, dict):
            continue
        scope = item.get("scope")
        if not isinstance(scope, dict):
            continue
        document_group = _nfc(str(scope.get("document_group_expression") or ""))
        if document_group and document_group not in question:
            # WireScope uses the empty string as its null sentinel. ``None`` is
            # valid only after wire -> SemanticIntent normalization.
            scope["document_group_expression"] = ""
            codes.add(UNGROUNDED_DOCUMENT_GROUP_DROPPED)
            changed = True
    return changed


def _recover_percent_change_base_metric_target(
        payload: dict[str, Any], question: str,
        ) -> tuple[str, str | None] | None:
    """Recover one derived compare target from its literal scalar base.

    A two-item financial request can ask first for a value and then for that
    same value's year-over-year rate.  The latter target occasionally reaches
    the wire as ``매출 증가율`` although the question only literally names
    ``매출액``.  Rebind only this closed topology: two scalar metric items,
    one retrieve and one compare, the same entity axis, exactly one approved
    base concept in the question, and an explicit increase/decrease-rate
    demand.  A generic ratio such as ``매출액 대비 영업이익률`` is not a
    percent-change target and remains ungrounded.
    """

    items = payload.get("answer_items")
    if (
            not isinstance(items, list)
            or len(items) < 2
            or payload.get("answer_groups")
            or payload.get("premises")
            or payload.get("unresolved_mentions")
            or _unique_percent_change_demand(question) is None
    ):
        return None
    if any(
            not isinstance(item, dict)
            or item.get("target", {}).get("kind") != "metric"
            or item.get("output", {}).get("shape") != "scalar"
            or item.get("output", {}).get("projection_mode") != "named_fields"
            or not isinstance(item.get("output", {}).get("field_surfaces"), list)
            or len(item["output"]["field_surfaces"]) != 1
            or item.get("selection", {}).get("mode") != "none"
            for item in items
    ):
        return None
    # 같은 요구를 모델이 두 항목으로도, 세 항목으로도 적는다.
    #
    #   2항목  retrieve(기준·비교 기간을 함께) + compare(증감률)
    #   3항목  retrieve(2025년) + retrieve(2024년) + compare(증감률)
    #
    # 둘 다 「한 지표의 전년 대비 변화」라는 같은 위상이다.  항목 수로 문을 잠그면
    # 같은 질문이 판마다 다르게 닫힌다.  대신 **비교 항목이 정확히 하나**이고
    # 나머지가 전부 조회이며, 조회들이 **같은 개체축과 같은 기준 지표**를 가리킬
    # 때만 받는다.  다른 지표가 섞이면 전년 대비 변화가 아니므로 물러난다.
    compares = [item for item in items if item.get("operation") == "compare"]
    retrieves = [item for item in items if item.get("operation") == "retrieve"]
    if len(compares) != 1 or not retrieves \
            or len(compares) + len(retrieves) != len(items):
        return None
    compare = compares[0]
    retrieve = retrieves[0]
    axis = compare["target"].get("entity_indexes")
    base_surfaces = {
        _nfc(str(item["target"].get("surface") or "")) for item in retrieves}
    if len(base_surfaces) != 1 or any(
            item["target"].get("entity_indexes") != axis
            for item in retrieves):
        return None

    base_target = _nfc(str(retrieve["target"].get("surface") or ""))
    compare_surface = _nfc(str(compare["target"].get("surface") or ""))
    if not base_target or base_target not in question or \
            not compare_surface or compare_surface in question:
        return None

    from .concept_alias import normalize_surface_key
    from .planning import METRIC_ALIASES, resolve_metric_concept

    base_candidates = [base_target]
    scoped = _LEADING_FINANCIAL_SCOPE.fullmatch(base_target)
    if scoped is not None:
        base_candidates.append(scoped.group("rest"))
    base = next((
        (surface, concept)
        for surface in base_candidates
        if surface in question
        and (concept := resolve_metric_concept(surface)) is not None
    ), None)
    if base is None:
        return None
    base_surface, base_concept = base
    question_key = normalize_surface_key(question)
    question_concepts = {
        concept for surface, concept in METRIC_ALIASES.items()
        if surface and surface in question_key
    }
    if question_concepts != {base_concept}:
        return None

    # A direct approved alias is safe.  Otherwise accept only the rate-change
    # suffix grammar after resolving the remaining base mention to the exact
    # same concept.  This excludes arbitrary derived ratios and event/document
    # wording without adding a new routing vocabulary.
    if resolve_metric_concept(compare_surface) is base_concept:
        return base_surface, None
    match = _PERCENT_CHANGE_DERIVED_METRIC.fullmatch(
        normalize_surface_key(compare_surface))
    if match is not None:
        # 파생 지표명이 범위를 달고 온다 — 「연결 매출액 변화율」.  기준 지표에
        # 하던 것과 똑같이 접두어를 떼고도 물어본다.  떼지 않으면 같은 요구가
        # 「매출액 변화율」일 때만 통과한다.
        derived_base = match.group("base")
        derived_candidates = [derived_base]
        for pattern in (_LEADING_FINANCIAL_SCOPE, _LEADING_SCOPED_STATEMENT):
            scoped_derived = pattern.fullmatch(_nfc(compare_surface))
            if scoped_derived is not None:
                inner = _PERCENT_CHANGE_DERIVED_METRIC.fullmatch(
                    normalize_surface_key(scoped_derived.group("rest")))
                if inner is not None:
                    derived_candidates.append(inner.group("base"))
        if any(resolve_metric_concept(candidate) is base_concept
               for candidate in derived_candidates):
            return base_surface, None

    # A literal rate-demand field may arrive as both the compare target and
    # output field (for example ``% 변화율``).  It carries no competing base
    # concept, so in this already-closed topology it can inherit the unique
    # literal retrieve base.  Do not accept arbitrary unknown target text.
    placeholder = normalize_surface_key(compare_surface)
    fields = compare["output"]["field_surfaces"]
    if (
            _PURE_PERCENT_CHANGE_PLACEHOLDER.fullmatch(placeholder) is None
            or len(fields) != 1
            or _RATE_ANSWER_FIELD.fullmatch(
                normalize_surface_key(str(fields[0]))) is None
    ):
        return None
    return base_surface, _unique_percent_change_demand(question)


def _narrow_selection_criterion(
        mode: str, criterion: str, question: str,
        ) -> str:
    pattern = _SELECTION_CUES.get(mode)
    if pattern is None:
        return criterion
    matches = [match.group(0) for match in pattern.finditer(question)]
    candidates = list(dict.fromkeys(matches))
    if len(candidates) != 1:
        return criterion
    if criterion and candidates[0] not in criterion:
        return criterion
    return candidates[0]


def _unique_numeric_premise_field_surface(
        payload: dict[str, Any], premise: dict[str, Any], question: str,
        ) -> str | None:
    """Return one exact numeric field attached to this premise, or ``None``.

    ``existence`` is often a provider-safe umbrella for a sentence that
    contains a number.  Narrowing it is safe only when the same premise points
    to exactly one answer item field whose complete surface is a number,
    money, or percentage expression and that surface is copied verbatim into
    both the question and the premise.  More than one matching field (even
    with the same text on separate items) is intentionally ambiguous.
    """

    raw_text = premise.get("raw_text")
    item_indexes = premise.get("applies_to_item_indexes")
    items = payload.get("answer_items")
    if (not isinstance(raw_text, str) or not isinstance(item_indexes, list)
            or not isinstance(items, list)):
        return None
    raw_text = _nfc(raw_text)
    candidates: list[tuple[int, str]] = []
    for item_index in item_indexes:
        if type(item_index) is not int or not 0 <= item_index < len(items):
            return None
        item = items[item_index]
        if not isinstance(item, dict):
            return None
        output = item.get("output")
        if not isinstance(output, dict):
            return None
        field_surfaces = output.get("field_surfaces")
        if not isinstance(field_surfaces, list):
            return None
        for field_surface in field_surfaces:
            if not isinstance(field_surface, str):
                return None
            surface = _nfc(field_surface)
            if (
                _NUMERIC_PREMISE_FIELD_SURFACE.fullmatch(surface)
                and surface in question
                and surface in raw_text
            ):
                candidates.append((item_index, surface))
    return candidates[0][1] if len(candidates) == 1 else None


def _recover_asserted_quantity_premise(
        payload: dict[str, Any], question: str, codes: set[str],
        ) -> None:
    """출력 필드 자리에 놓인 **수량**을 전제로 되살린다.

    필드 표현은 「무엇을 뽑을지」의 이름이다.  `400조원` 같은 값이 그 자리에
    오면 이름이 아니라 **사용자가 주장한 값**이다.  전제가 비면 그 주장이
    통째로 사라져, 「맞아?」라는 물음에 답하지 않고 조회만 하는 계획이 된다.
    후보를 못 내는 것보다 나쁘다 — 답은 나오는데 물은 것에 답하지 않는다.

    지어내는 것이 없다.  되살린 원문은 모델이 준 필드 표현 그대로이고
    질문에 그대로 있는지 확인한다.  종류는 `existence` 로 두어 바로 뒤의
    좁히기가 `numeric` 으로 확정하게 한다.
    """

    premises = payload.get("premises")
    items = payload.get("answer_items")
    if premises != [] or not isinstance(items, list) or len(items) != 1:
        return
    item = items[0]
    if not isinstance(item, dict):
        return
    output = item.get("output")
    if not isinstance(output, dict) or output.get("shape") != "scalar":
        return
    surfaces = output.get("field_surfaces")
    if not isinstance(surfaces, list) or len(surfaces) != 1:
        return
    surface = _nfc(surfaces[0])
    if not _NUMERIC_PREMISE_FIELD_SURFACE.fullmatch(surface):
        return
    if surface not in question:
        return
    premises.append({
        "kind": "existence",
        "raw_text": surface,
        "applies_to_item_indexes": [0],
    })
    codes.add(ASSERTED_QUANTITY_PREMISE_RECOVERED)


def _narrow_exact_numeric_existence_premises(
        payload: dict[str, Any], question: str, codes: set[str],
        ) -> None:
    """Preserve one exact numeric claim without creating a semantic row."""

    premises = payload.get("premises")
    if not isinstance(premises, list):
        return
    for premise in premises:
        if not isinstance(premise, dict) or premise.get("kind") != "existence":
            continue
        surface = _unique_numeric_premise_field_surface(
            payload, premise, question)
        if surface is None:
            continue
        premise["kind"] = "numeric"
        premise["raw_text"] = surface
        codes.add(NUMERIC_PREMISE_SURFACE_NARROWED)


def _recover_document_negative_confirmation_premise(
        payload: dict[str, Any], question: str, codes: set[str],
        ) -> None:
    """Recover one literal correction-history premise from a closed grammar.

    This does not generalize Korean negative questions.  It is limited to one
    ``retrieve`` item for an explicit ``YYYY년 사업보고서`` and its ``정정``
    scalar field, with no provider premise already present.  The recovered
    text remains the exact matched question surface so the normalizer can
    prove grounding without an alias, fixture, or question identifier.
    """

    match = _DOCUMENT_NEGATIVE_CONFIRMATION.search(question)
    if match is None:
        return
    premises = payload.get("premises")
    items = payload.get("answer_items")
    if premises != [] or not isinstance(items, list) or len(items) != 1:
        return
    item = items[0]
    if not isinstance(item, dict) or item.get("operation") != "retrieve":
        return
    target = item.get("target")
    output = item.get("output")
    if not isinstance(target, dict) or not isinstance(output, dict):
        return
    surface = target.get("surface")
    if not isinstance(surface, str) or not _ANNUAL_BUSINESS_REPORT.fullmatch(
            _nfc(surface)):
        return
    if (
            output.get("shape") != "scalar"
            or output.get("projection_mode") != "named_fields"
            or output.get("field_surfaces") != ["정정"]
    ):
        return
    premises.append({
        "kind": "existence",
        "raw_text": match.group("surface"),
        "applies_to_item_indexes": [0],
    })
    codes.add(DOCUMENT_NEGATIVE_CONFIRMATION_PREMISE_RECOVERED)


def _names_latest_periodic(surface: str) -> bool:
    """대상 표현이 정기보고서 **최신 지시**로 확정되는가."""

    from agent.periodic_document_preflight import (
        PeriodicDocumentPreflightError, parse_periodic_expression)

    try:
        selector = parse_periodic_expression(surface)
    except PeriodicDocumentPreflightError:
        return False
    return bool(selector.latest)


# A former-name marker is a token, not the first syllable of a company name.
# Require whitespace or punctuation so current names such as 「전방」 and
# 「구영테크」 are never interpreted as 「전 방」 / 「구 영테크」.
_FORMER_NAME_MARKER = re.compile(
    r"^(?:구|前|전)(?:\s+|[.,]\s*)(?P<name>\S.*)$")


def _merge_parenthetical_former_name_entities(
        payload: dict[str, Any], question: str) -> bool:
    """「A(구 B)」의 B 는 A 의 옛 이름이지 다른 회사가 아니다.

    사명이 바뀐 회사를 물을 때 한국어는 새 이름 뒤 괄호에 옛 이름을 병기한다.
    모델이 이것을 회사 **둘**로 읽으면 회사 하나를 요구하는 조각이 전부 막히고,
    질문에 정본과 똑같은 표기가 이미 있는데도 답을 못 만든다.

    합치는 조건은 셋을 **모두** 만족할 때뿐이다.

      - 개체 표면이 옛 이름 표지(`구`·`前`·`전`)로 시작한다
      - 그 표면이 질문의 괄호 안에 있다
      - 괄호 바로 앞이 남길 회사 개체의 표면이다

    하나라도 어긋나면 남긴다.  괄호 안에 **실제로 다른 회사**를 적은 질문을
    합치지 않는다.
    """

    entities = payload["entities"]
    if len(entities) < 2:
        return False
    compact_question = _nfc(question)
    dropped: dict[int, int] = {}
    for index, row in enumerate(entities):
        if row["kind_hint"] != "company":
            continue
        surface = _nfc(row["surface"])
        if _FORMER_NAME_MARKER.fullmatch(surface) is None:
            continue
        for other, anchor_row in enumerate(entities):
            if other == index or anchor_row["kind_hint"] != "company":
                continue
            anchor_surface = _nfc(anchor_row["surface"])
            if not anchor_surface:
                continue
            paired = f"{anchor_surface}({surface})"
            if paired in compact_question:
                dropped[index] = other
                break
    if not dropped:
        return False
    kept = [index for index in range(len(entities)) if index not in dropped]
    remap = {old: new for new, old in enumerate(kept)}
    payload["entities"] = [entities[index] for index in kept]
    for item in payload["answer_items"]:
        target = item["target"]
        target["entity_indexes"] = _ordered_unique_indexes([
            remap[dropped[index]] if index in dropped else remap[index]
            for index in target["entity_indexes"]
            if index in remap or index in dropped
        ])
    return True


def _drop_target_restating_entities(
        payload: dict[str, Any], question: str) -> bool:
    """대상을 되풀이하는 개체를 걷어낸다.

    개체 목록은 **회사 목록**이다.  문서·사건은 `target.surface` 가 이미
    담고 있는데, 모델이 그것을 개체로 한 번 더 적고 항목이 회사 대신 그쪽을
    가리키는 일이 있다.  그러면 「회사 하나」를 요구하는 조각들이 전부
    막힌다.

    지우는 조건은 **같은 것을 같은 종류로 두 번 적었을 때**뿐이다.  개체의
    `kind_hint` 가 그 대상의 `kind` 와 같아야 한다.  종류가 다르면(대상은
    문서인데 개체는 사건) 개체가 대상에 없는 것을 말하고 있으므로 남긴다.

    회사 개체가 남아 있을 때만 지운다.  사건·문서 하나만 오는 의도는
    그 자체가 정당한 모양이라 건드리지 않는다.
    """

    entities = payload["entities"]
    if len(entities) < 2:
        return False
    companies = [index for index, row in enumerate(entities)
                 if row["kind_hint"] == "company"]
    if len(companies) != 1:
        return False
    restated: set[str] = set()
    for item in payload["answer_items"]:
        target = item["target"]
        for row in entities:
            if (row["kind_hint"] == target["kind"]
                    and _nfc(row["surface"]) == _nfc(target["surface"])):
                restated.add(_nfc(row["surface"]))
    dropped = {
        index for index, row in enumerate(entities)
        if index not in companies
        and _nfc(row["surface"]) in restated
        and _nfc(row["surface"]) in question
    }
    if not dropped:
        return False
    kept = [index for index in range(len(entities)) if index not in dropped]
    remap = {old: new for new, old in enumerate(kept)}
    company = remap[companies[0]]
    payload["entities"] = [entities[index] for index in kept]
    for item in payload["answer_items"]:
        target = item["target"]
        target["entity_indexes"] = _ordered_unique_indexes(
            [remap.get(index, company) for index in target["entity_indexes"]])
    return True


def _ordered_unique_indexes(values: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _split_scope_comparison_items(
        items: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """한 항목의 필드들이 **서로 다른 scope** 를 달고 있으면 scope 별로 나눈다.

    「연결 매출액과 별도 매출액의 차이」처럼 scope 끼리 견주는 질문은 항목
    하나로 와도 scope 가 필드마다 다르다.  그대로 두면 뒤의 끌어올리기가
    scope 하나만 항목 수준으로 올리고 나머지는 필드에 남겨, scope 는 연결인데
    필드는 별도인 모순된 의도가 된다.

    나눈 뒤에는 각 항목이 scope 하나만 달게 되어 기존 규칙이 그대로 통한다.
    """

    if len(items) != 1:
        return None
    item = items[0]
    if item["target"]["kind"] != "metric":
        return None
    surfaces = [_nfc(row) for row in item["output"]["field_surfaces"]]
    if len(surfaces) < 2:
        return None
    scopes: list[str] = []
    for surface in surfaces:
        match = _LEADING_FINANCIAL_SCOPE.fullmatch(surface)
        if match is None:
            return None
        scopes.append(match.group("scope"))
    if len(set(scopes)) < 2:
        return None
    split: list[dict[str, Any]] = []
    for surface in surfaces:
        row = deepcopy(item)
        row["operation"] = "retrieve"
        row["output"]["shape"] = "scalar"
        row["output"]["field_surfaces"] = [surface]
        row["target"]["surface"] = surface
        split.append(row)
    return split


def _target_restating_field(
        target: dict[str, Any], field_surfaces: list[str], question: str,
        ) -> tuple[str, list[str]] | None:
    """대상을 되풀이하는 필드를 걷어내고, 대상을 그 앞부분으로 줄인다.

    `target.surface` 가 절 전체("배터리 공급계약 중 … 해지 공시")로 오고
    그 앞부분이 필드로도 오는 일이 있다.  이때 앞부분이 대상이고 나머지가
    뽑을 것이다.  고르는 두 문자열 모두 모델이 준 값이며, 질문에 그대로
    있는지 확인한 뒤에만 바꾼다.
    """

    surface = _nfc(target["surface"])
    grounded = [_nfc(row) for row in field_surfaces]
    for index, candidate in enumerate(grounded):
        if not candidate or candidate == surface or not surface.startswith(candidate):
            continue
        if candidate not in question:
            continue
        remaining = [row for position, row in enumerate(grounded)
                     if position != index]
        if not remaining:
            continue
        return candidate, remaining
    return None


def _repair_roles(
        question: str, wire: HcxSemanticIntentWire, *,
        company_surface_regrounder: CompanySurfaceRegrounder | None = None,
        ) -> tuple[HcxSemanticIntentWire, tuple[str, ...]]:
    payload = wire.model_dump(mode="python", warnings=False)
    codes: set[str] = set()

    if company_surface_regrounder is not None \
            and not callable(company_surface_regrounder):
        raise TypeError("company surface regrounder는 callable이어야 합니다")
    if _rebase_unambiguous_one_based_entity_indexes(payload):
        codes.add(ONE_BASED_ENTITY_INDEXES_REBASED)
    for entity in payload["entities"]:
        surface = _nfc(entity["surface"])
        if entity["kind_hint"] != "company":
            replacement = reground_dropped_particles(surface, question)
            if replacement is None:
                replacement = reground_coordinate_separator(surface, question)
            if replacement is None:
                replacement = reground_quoted_contract_surface(surface, question)
            if replacement is not None:
                entity["surface"] = replacement
                codes.add(ENTITY_SURFACE_REGROUNDED)
            continue
        if company_surface_regrounder is None:
            continue
        # 질문에 있는 표기는 원래 그대로 둔다.  예외는 **병기 표기** 하나다 —
        # 「케이티(KT)」는 질문에 literal 로 있지만 registry 에 통째로는 없어
        # 회사 해소가 0건이 되고, 그다음 조각들이 전부 막힌다.  괄호가 있을
        # 때만 되돌리기를 물어보므로 다른 표기의 비용·동작은 그대로다.
        if surface in question and "(" not in surface and "（" not in surface:
            continue
        replacement = company_surface_regrounder(surface, question)
        if not isinstance(replacement, str):
            continue
        replacement = _nfc(replacement)
        if replacement and replacement in question:
            entity["surface"] = replacement
            codes.add(ENTITY_SURFACE_REGROUNDED)

    if _merge_parenthetical_former_name_entities(payload, question):
        codes.add(PARENTHETICAL_FORMER_NAME_MERGED)

    if _drop_target_restating_entities(payload, question):
        codes.add(TARGET_RESTATING_ENTITY_DROPPED)

    presentation = _explicit_presentation(question)
    if payload["presentation"] != presentation:
        payload["presentation"] = presentation
        codes.add(PRESENTATION_RECOMPUTED)

    split_items = _split_scope_comparison_items(payload["answer_items"])
    if split_items is not None:
        payload["answer_items"] = split_items
        codes.add(SCOPE_COMPARISON_ITEMS_SPLIT)

    _drop_ungrounded_document_groups(payload, question, codes)

    # 기간이 답 필드에 앉아 있으면 아래 파생 지표 복구가 (필드가 하나가 아니라는
    # 이유로) 물러난다.  옮기는 것이 먼저다.
    if _move_period_only_fields_to_scope(payload, question, codes):
        codes.add(PERIOD_FIELDS_MOVED_TO_SCOPE)

    percent_change_repair = _recover_percent_change_base_metric_target(
        payload, question)
    if percent_change_repair is not None:
        percent_change_target, percent_change_field = percent_change_repair
        for item in payload["answer_items"]:
            if item["operation"] == "compare":
                item["target"]["surface"] = percent_change_target
                codes.add(PERCENT_CHANGE_BASE_METRIC_TARGET_REGROUNDED)
                if percent_change_field is not None:
                    item["output"]["field_surfaces"] = [percent_change_field]
                    codes.add(PERCENT_CHANGE_DEMAND_SURFACE_REGROUNDED)
                break

    for item in payload["answer_items"]:
        target = item["target"]
        scope = item["scope"]
        output = item["output"]

        correction_target = _reground_correction_collection_target(
            payload, item, question)
        if correction_target is not None:
            target["surface"] = correction_target
            codes.add(CORRECTION_COLLECTION_TARGET_REGROUNDED)

        # 대상 표현이 필드 표현으로 **시작하면** 그 필드는 대상을 한 번 더
        # 적은 것이다.  대상을 그 앞부분으로 줄이고 필드에서 뺀다.  두 문자열
        # 모두 모델이 준 것이라 지어내는 것이 없다.
        restated = _target_restating_field(target, output["field_surfaces"],
                                           question)
        if restated is not None:
            target["surface"], output["field_surfaces"] = restated
            codes.add(TARGET_RESTATED_FIELD_DROPPED)

        original_target = _nfc(target["surface"])
        target_surface = original_target
        moved: list[str] = []
        period_from_target_qualifier = False

        if not payload["entities"] and target["entity_indexes"]:
            target["entity_indexes"] = []
            codes.add(IMPOSSIBLE_ENTITY_REFERENCES_REMOVED)

        if target["kind"] == "metric" and target_surface not in question:
            replacement = _question_metric_surface(target_surface, question)
            if replacement is not None:
                target_surface = replacement
                codes.add(TARGET_CONCEPT_SURFACE_REGROUNDED)
            elif item["operation"] == "compare":
                replacement = _unique_comparison_target(question)
                if replacement is not None:
                    target_surface = replacement
                    codes.add(COMPARISON_TARGET_SURFACE_REGROUNDED)
        elif (target["kind"] in {"document", "event"}
                and target_surface not in question
                and "사업보고서" in target_surface.replace(" ", "")
                and "정정" in target_surface.replace(" ", "")
                and re.search(r"기재\s*정정|정정\s*공시", question)):
            reports = list(dict.fromkeys(
                match.group(0) for match in re.finditer(
                    r"20[0-9]{2}년\s*사업보고서", question)))
            if len(reports) == 1:
                target_surface = reports[0]
                codes.add(TARGET_CONCEPT_SURFACE_REGROUNDED)
                count = re.search(r"기재\s*정정\s*공시\s*두\s*건", question)
                if (count is not None and output["field_surfaces"]
                        and any(row not in question
                                for row in output["field_surfaces"])):
                    output["field_surfaces"] = [count.group(0)]
                    codes.add(ANSWER_DEMAND_SURFACE_REGROUNDED)

        # HCX sometimes canonicalizes a holding metric (for example
        # ``보유주식수``) while the question says ``주식을 몇 주 보유``.
        # The exact holding wording in the question is a safe target anchor;
        # the dedicated source-intent regrounder later binds reviewed slots.
        if target_surface not in question:
            holding_surface = _holding_question_target_surface(question)
            if holding_surface is not None:
                target_surface = holding_surface
                codes.add(HOLDING_TARGET_SURFACE_REGROUNDED)

        periods: list[str] = []
        for row in scope["target_period_expressions"]:
            period = _nfc(row)
            if re.fullmatch(r"20[0-9]{2}", period) \
                    and question.count(f"{period}년") == 1:
                period = f"{period}년"
                codes.add(DATE_SURFACE_REGROUNDED)
            elif period not in question:
                replacement = _question_date_surface(period, question)
                if replacement is not None:
                    period = replacement
                    codes.add(DATE_SURFACE_REGROUNDED)
            periods.append(period)
        as_of = _nfc(scope["as_of_expression"])
        if as_of and as_of not in question:
            replacement = _question_date_surface(as_of, question)
            if replacement is not None:
                scope["as_of_expression"] = replacement
                codes.add(DATE_SURFACE_REGROUNDED)
        scope_qualifiers = [
            _nfc(row) for row in scope["scope_qualifier_expressions"]]
        remaining_target_qualifiers: list[str] = []
        for qualifier in target["qualifier_surfaces"]:
            grounded = _nfc(qualifier)
            if re.fullmatch(r"20[0-9]{2}", grounded) \
                    and question.count(f"{grounded}년") == 1:
                grounded = f"{grounded}년"
                codes.add(DATE_SURFACE_REGROUNDED)
            elif grounded not in question:
                replacement = _question_date_surface(grounded, question)
                if replacement is not None:
                    grounded = replacement
                    codes.add(DATE_SURFACE_REGROUNDED)
            if _PERIOD_FULL.fullmatch(grounded):
                periods.append(grounded)
                moved.append(grounded)
                period_from_target_qualifier = True
                codes.add(PERIOD_FROM_TARGET_QUALIFIER)
            elif (
                target["kind"] == "metric"
                and _FINANCIAL_SCOPE_FULL.fullmatch(grounded)
            ):
                scope_qualifiers.append(grounded)
                moved.append(grounded)
                codes.add(SCOPE_FROM_TARGET_QUALIFIER)
            elif is_display_unit_only_surface(grounded):
                # A display-unit instruction ("정확한 원 단위", "조원 단위", …)
                # is formatting guidance, not a target concept qualifier.  It
                # has no dedicated wire slot, so a provider can leave it here
                # instead; grounding must not treat it as a metric qualifier.
                # The composer recovers the same instruction from the
                # original question text at render time (see
                # ``agent.display_units_v1``), so dropping it here loses
                # nothing — it never crossed the schema boundary to begin
                # with.
                moved.append(grounded)
                codes.add(DISPLAY_UNIT_QUALIFIER_DROPPED)
            else:
                remaining_target_qualifiers.append(grounded)

        # A leading date inside a document title is part of the document
        # coordinate; it is not silently reclassified as a target period.
        if target["kind"] != "document":
            period_match = _LEADING_PERIOD.fullmatch(target_surface)
            if period_match is not None:
                periods.append(period_match.group("scope"))
                moved.append(period_match.group("scope"))
                target_surface = period_match.group("rest")
                codes.add(PERIOD_PREFIX_SPLIT)

        # 대상 표현 앞에 붙은 **회사 이름**은 지표 이름이 아니다.
        # 「○○ 별도 매출」은 `○○` 라는 지표가 아니라 그 회사의 「별도 매출」
        # 이다.  떼어내야 뒤의 scope 분리가 `별도` 를 집어낸다.  회사는
        # `entities` 에 이미 있으므로 잃는 정보가 없다.
        #
        # **지표일 때만** 뗀다.  문서·사건에서는 이름에 붙은 상대방이 그 대상의
        # 정체다 — 「○○ 계약」에서 상대를 떼면 그냥 「계약」이 되어 어느
        # 계약인지 알 수 없다.
        if target["kind"] == "metric":
            for entity in payload["entities"]:
                if entity["kind_hint"] != "company":
                    continue
                stripped = _strip_edge(target_surface, _nfc(entity["surface"]),
                                       question)
                if stripped != target_surface:
                    target_surface = stripped
                    codes.add(TARGET_ENTITY_PREFIX_REMOVED)

        # 「매출 차이」의 `차이` 는 지표 이름이 아니라 **무엇을 물었는가**다.
        # 지표는 `매출` 이고 `차이` 는 비교 요구다.  붙여 두면 그런 계정을
        # 찾다가 물러난다.  떼어낸 앞부분이 질문에 그대로 있을 때만 바꾼다.
        if target["kind"] == "metric" and item["operation"] == "compare":
            for demand in _COMPARISON_DEMANDS:
                stripped = _strip_edge(target_surface, demand, question)
                if stripped != target_surface:
                    target_surface = stripped
                    codes.add(COMPARISON_DEMAND_SPLIT_FROM_TARGET)
                    break

        if target["kind"] == "metric":
            scope_match = _LEADING_FINANCIAL_SCOPE.fullmatch(target_surface)
            if scope_match is not None:
                scope_qualifiers.append(scope_match.group("scope"))
                moved.append(scope_match.group("scope"))
                target_surface = scope_match.group("rest")
                codes.add(SCOPE_PREFIX_SPLIT)
            else:
                # 범위가 재무제표 이름에 **붙어 있는** 판.  범위만 거두고 제표
                # 이름은 지표에서 뗀다.  둘 다 질문 안의 연속 부분 문자열이라
                # 표면 불변식을 지킨다.
                statement_match = _LEADING_SCOPED_STATEMENT.fullmatch(
                    target_surface)
                if statement_match is not None:
                    scope_surface = statement_match.group("scope")
                    if scope_surface in question:
                        scope_qualifiers.append(scope_surface)
                        moved.append(scope_surface)
                        target_surface = statement_match.group("rest")
                        codes.add(SCOPE_STATEMENT_PREFIX_SPLIT)

        if target["kind"] != "document":
            for surface in moved:
                stripped = _strip_edge(target_surface, surface, question)
                if stripped != target_surface:
                    target_surface = stripped
                    codes.add(TARGET_SCOPE_PREFIX_REMOVED)

        if (
                target["kind"] == "metric"
                and period_from_target_qualifier
                and target_surface not in question
        ):
            replacement = _period_qualified_question_metric_surface(question)
            if replacement is not None:
                target_surface = replacement
                codes.add(TARGET_CONCEPT_SURFACE_REGROUNDED)

        if target_surface != original_target:
            target["surface"] = target_surface
        target["qualifier_surfaces"] = remaining_target_qualifiers
        scope["target_period_expressions"] = _ordered_unique(periods, question)
        scope["scope_qualifier_expressions"] = _ordered_unique(
            scope_qualifiers, question)

        field_surfaces: list[str] = []
        for field_surface in output["field_surfaces"]:
            repaired_field = _nfc(field_surface)
            for surface in moved:
                stripped = _strip_edge(repaired_field, surface, question)
                if stripped != repaired_field:
                    repaired_field = stripped
                    codes.add(FIELD_SCOPE_PREFIX_REMOVED)
            field_surfaces.append(repaired_field)

        if (
            _holding_question_target_surface(question) is not None
            and output["projection_mode"] == "named_fields"
            and field_surfaces
            and any(surface not in question for surface in field_surfaces)
        ):
            reviewed_holding_fields = _holding_question_field_surfaces(question)
            if reviewed_holding_fields:
                field_surfaces = reviewed_holding_fields
                codes.add(ANSWER_DEMAND_SURFACE_REGROUNDED)

        if (
            target["kind"] == "metric"
            and item["operation"] in {"retrieve", "compare"}
            and output["shape"] in {"scalar", "comparison"}
            and output["projection_mode"] == "named_fields"
            and len(field_surfaces) >= 1
            and (len(field_surfaces) > 1
                 or any(surface not in question for surface in field_surfaces))
            and _explicit_single_quarter_arithmetic(question)
        ):
            metric_surface = (
                _question_metric_surface(target_surface, question)
                or _single_quarter_metric_surface(target_surface, question)
                or (target_surface if target_surface in question else None)
            )
            if metric_surface is not None:
                target_surface = metric_surface
                target["surface"] = metric_surface
                item["operation"] = "retrieve"
                output["shape"] = "scalar"
                field_surfaces = [metric_surface]
                codes.add(TARGET_CONCEPT_SURFACE_REGROUNDED)
                codes.add(SINGLE_QUARTER_OUTPUT_COLLAPSED)

        if (
            item["operation"] == "compare"
            and output["shape"] == "comparison"
            and output["projection_mode"] == "named_fields"
            and len(field_surfaces) >= 2
            and any(surface not in question for surface in field_surfaces)
        ):
            demands = _ordered_scalar_demands(question)
            if len(demands) == len(field_surfaces):
                field_surfaces = [
                    surface if surface in question else demands[index]
                    for index, surface in enumerate(field_surfaces)
                ]
                codes.add(COMPARISON_DEMAND_FIELDS_REGROUNDED)

        if (
            target["kind"] == "metric"
            and item["operation"] == "retrieve"
            and output["shape"] == "scalar"
            and output["projection_mode"] == "named_fields"
            and item["selection"]["mode"] in {"maximum", "minimum"}
            and len(target.get("entity_indexes", [])) >= 3
            and len(field_surfaces) >= 2
            and any(surface not in question for surface in field_surfaces)
        ):
            # HCX can encode a three-company superlative as scalar/retrieve
            # and paraphrase the answer roles as ``기업``/``금액``.  The
            # topology is repaired later by
            # NaryFinancialSuperlativeShapeRegrounder, but grounding must
            # first preserve the literal user demands (``어디``/``얼마``).
            # Rebind only a one-to-one ordered demand list; no entity,
            # selection direction, metric, period, or answer value is added.
            demands = _ordered_scalar_demands(question)
            if len(demands) == len(field_surfaces):
                field_surfaces = [
                    surface if surface in question else demands[index]
                    for index, surface in enumerate(field_surfaces)
                ]
                codes.add(COMPARISON_DEMAND_FIELDS_REGROUNDED)

        if (
            target["kind"] == "metric"
            and item["operation"] in {"retrieve", "compare"}
            and output["projection_mode"] == "named_fields"
            and len(field_surfaces) == 2
            and any(surface not in question for surface in field_surfaces)
        ):
            # HCX can preserve the amount field but paraphrase the rate field
            # (for example ``증감률``) even though the question literally asks
            # ``얼마`` and ``몇 퍼센트 변했어``.  Rebind only this closed pair
            # of unique, ordered answer demands.  The downstream financial
            # regrounder still has to prove the annual comparison topology.
            amount_demand = _unique_scalar_demand(question)
            rate_demand = _unique_percent_change_demand(question)
            if amount_demand is not None and rate_demand is not None:
                field_surfaces = sorted(
                    [amount_demand, rate_demand], key=question.index)
                codes.add(COMPARISON_DEMAND_FIELDS_REGROUNDED)

        if (
            item["operation"] == "compare"
            and output["shape"] == "comparison"
            and output["projection_mode"] == "named_fields"
            and len(field_surfaces) == 1
            and field_surfaces[0] not in question
        ):
            demand_span = _comparison_demand_span(question)
            if demand_span is not None:
                field_surfaces = [demand_span]
                codes.add(COMPARISON_DEMAND_SPAN_REGROUNDED)

        if (
            target["kind"] == "metric"
            and item["operation"] == "compare"
            and output["shape"] == "comparison"
            and output["projection_mode"] == "named_fields"
            and len(field_surfaces) == 1
            and field_surfaces[0] not in question
        ):
            percent_change_demand = _unique_percent_change_demand(question)
            if percent_change_demand is not None:
                field_surfaces = [percent_change_demand]
                codes.add(PERCENT_CHANGE_DEMAND_SURFACE_REGROUNDED)

        if (
            target["kind"] == "metric"
            and output["projection_mode"] == "named_fields"
            and len(field_surfaces) == 1
        ):
            demand = _unique_scalar_demand(question)
            current = field_surfaces[0]
            if (
                demand is not None
                and current != demand
                and (
                    (
                        output["shape"] == "scalar"
                        and (
                            current == target_surface
                            or target_surface in current
                            or demand.startswith(current)
                        )
                    )
                    or current not in question
                )
            ):
                field_surfaces = [demand]
                codes.add(
                    SCALAR_DEMAND_SURFACE_SELECTED
                    if output["shape"] == "scalar"
                    else ANSWER_DEMAND_SURFACE_REGROUNDED
                )
        # 요구어(「얼마」·「어디」…) 없이 지표만 묻는 질문이 있다.  「카카오 2025년
        # 매출 알려줘」에는 위 복구가 쓸 표면이 없어 모델이 지어낸 필드 표면이
        # 그대로 남고, grounding 이 질문 전체를 닫는다.  이때 답 필드는 지표
        # 자신이다 — 질문이 그것을 물었기 때문이다.
        #
        # **이미 근거를 잃은 필드에만** 적용한다.  지금 통과하는 요청은 이 가지에
        # 들어오지 않으므로 동작이 바뀌지 않는다.
        if (
            target["kind"] == "metric"
            and output["projection_mode"] == "named_fields"
            and len(field_surfaces) == 1
            and field_surfaces[0] not in question
            and _unique_scalar_demand(question) is None
            and _PERCENT_CHANGE_RATE_CUE.search(question) is None
            and target_surface
            and target_surface in question
        ):
            field_surfaces = [target_surface]
            codes.add(UNGROUNDED_FIELD_NARROWED_TO_TARGET)

        if (
            item["operation"] == "retrieve"
            and target["kind"] == "document"
            and output["shape"] == "narrative"
            and output["projection_mode"] == "named_fields"
            and field_surfaces
            and all(surface not in question for surface in field_surfaces)
        ):
            # A document narrative can answer its whole target without a
            # provider-invented field label.  Dropping every ungrounded label
            # is strictly less semantic invention than keeping one; the
            # question-grounded document target and operation remain intact.
            output["projection_mode"] = "whole_target"
            field_surfaces = []
            codes.add(UNGROUNDED_DOCUMENT_FIELDS_DROPPED)

        if (
            target["kind"] == "metric"
            and output["shape"] in {"record", "scalar"}
            and output["projection_mode"] == "named_fields"
            and field_surfaces
            and all(is_display_unit_only_surface(surface)
                    or is_display_rounding_only_surface(surface)
                    for surface in field_surfaces)
            and target_surface
        ):
            # A provider can answer a plain scalar metric question by naming
            # the requested display units (and/or a separate rounding-digit
            # instruction — #170/M35-b, "조원 단위로 얼마인가? 소수점 둘째
            # 자리까지 반올림해줘.") as the field labels ("정확한 원 단위",
            # "조원 단위", "소수점 둘째 자리" — see RPC-004/RPC-005 and M35-b
            # in ``out/logs/stage1_wire_failures.jsonl``).  That is a display
            # instruction wearing a multi-field record shape, or a bare
            # "scalar" shape whose only field(s) are formatting cues rather
            # than an answer field; no downstream handler recognizes either
            # shape when every field is a unit/rounding cue.  Collapse it
            # back to the one scalar metric the question actually asks for.
            # The composer recovers the same display instruction from the
            # question text independently (see ``agent.display_units_v1``),
            # so no information is lost here.
            output["shape"] = "scalar"
            field_surfaces = [target_surface]
            codes.add(DISPLAY_UNIT_FIELD_SURFACES_DROPPED)

        output["field_surfaces"] = field_surfaces

        if (
            target["kind"] == "event"
            and output["projection_mode"] == "named_fields"
            and len(field_surfaces) >= 2
            and any(surface not in question for surface in field_surfaces)
        ):
            state_surfaces = _ordered_state_surfaces(question)
            if len(state_surfaces) == len(field_surfaces):
                output["field_surfaces"] = [
                    surface if surface in question else state_surfaces[index]
                    for index, surface in enumerate(field_surfaces)
                ]
                codes.add(ANSWER_STATE_SURFACES_REGROUNDED)

        expected_item_presentation = presentation
        if output["presentation"] != expected_item_presentation:
            output["presentation"] = expected_item_presentation
            codes.add(PRESENTATION_RECOMPUTED)

        selection = item["selection"]
        if selection["mode"] != "none":
            criterion = selection["criterion_surface"]
            narrowed = _narrow_selection_criterion(
                selection["mode"], criterion, question)
            if narrowed != criterion:
                selection["criterion_surface"] = narrowed
                codes.add(SELECTION_CRITERION_NARROWED)

    # 파생 지표 복구가 지표를 되돌린 뒤, 비교 항목의 기간·범위가 빈 채로 남는 판이
    # 있다.  항목별 보정이 모두 끝난 지금이 물려줄 값이 확정된 시점이다.
    if percent_change_repair is not None and _inherit_percent_change_axis(payload):
        codes.add(PERCENT_CHANGE_AXIS_INHERITED)

    # 항목 병합(`_collapse_percent_change_items`)은 **꺼져 있다.**
    #
    # compiler crash 를 typed refuse 로 바꾸는 대신, 이미 답이 나오던 질문 둘을
    # 막았다. 실측 답변 생성이 35/43 → 33/43 으로 줄어 순손실이었다.  사용자에게는
    # crash 든 refuse 든 답이 없는 것은 같으므로 교환이 성립하지 않는다.
    #
    # 함수는 남겨 둔다 — compiler 가 받는 구조를 넓힌 뒤 다시 켤 후보다.
    # 실측 근거는 43문항 실호출 기록(2026-08-25)에 있다.

    # **마지막으로 조사 누락만 되돌린다.** 앞의 보정들이 다루지 못한 표면 중,
    # 질문에 실제 구간이 있고 조사·공백만 흘린 것이 남아 있다. 라이브 3회 반복에서
    # `주가 상승의 원인` → `주가 상승 원인` 이 두 번 관측됐다.
    #
    # 재구성(`달라졌는가` → `달라진 것`)과 발명(없는 선택 기준)은 되돌리지 않는다.
    # `reground_dropped_particles` 가 그 둘에 대해 `None` 을 낸다.
    for item in payload.get("answer_items", []):
        if not isinstance(item, dict):
            continue
        target = item.get("target")
        if isinstance(target, dict):
            fixed = reground_dropped_particles(target.get("surface") or "", question)
            if fixed is not None:
                target["surface"] = fixed
                codes.add(PARTICLE_DROPPED_SURFACE_REGROUNDED)
            else:
                fixed = reground_coordinate_separator(
                    target.get("surface") or "", question)
                if fixed is not None:
                    target["surface"] = fixed
                    codes.add(COORDINATE_SEPARATOR_REGROUNDED)
                else:
                    fixed = reground_quoted_contract_surface(
                        target.get("surface") or "", question)
                    if fixed is not None:
                        target["surface"] = fixed
                        codes.add(QUOTED_CONTRACT_SURFACE_REGROUNDED)
                    else:
                        split = reground_coordinated_head(
                            target.get("surface") or "", question)
                        if split is not None:
                            head, modifier = split
                            target["surface"] = head
                            qualifiers = target.get("qualifier_surfaces")
                            if isinstance(qualifiers, list) and modifier not in qualifiers:
                                qualifiers.append(modifier)
                            codes.add(COORDINATED_HEAD_SURFACE_REGROUNDED)
                        else:
                            fixed = reground_dropped_quote_marks(
                                target.get("surface") or "", question)
                            if fixed is not None:
                                target["surface"] = fixed
                                codes.add(QUOTE_MARKS_DROPPED_SURFACE_REGROUNDED)
                            else:
                                fixed = reground_trailing_termination_verb(
                                    target.get("surface") or "", question)
                                if fixed is not None:
                                    target["surface"] = fixed
                                    codes.add(TRAILING_TERMINATION_VERB_DROPPED)
                                else:
                                    fixed = reground_contract_family_target(
                                        target.get("surface") or "", question)
                                    if fixed is not None:
                                        target["surface"] = fixed
                                        codes.add(
                                            CONTRACT_FAMILY_TARGET_REGROUNDED)
        output = item.get("output")
        if isinstance(output, dict) and isinstance(
                output.get("field_surfaces"), list):
            surfaces = []
            for surface in output["field_surfaces"]:
                fixed = reground_dropped_particles(surface or "", question)
                if fixed is not None:
                    surfaces.append(fixed)
                    codes.add(PARTICLE_DROPPED_SURFACE_REGROUNDED)
                    continue
                fixed = reground_coordinate_separator(surface or "", question)
                if fixed is None:
                    split = reground_coordinated_head(surface or "", question)
                    if split is None:
                        surfaces.append(surface)
                        continue
                    surfaces.append(split[0])
                    codes.add(COORDINATED_HEAD_SURFACE_REGROUNDED)
                    continue
                surfaces.append(fixed)
                codes.add(COORDINATE_SEPARATOR_REGROUNDED)
            output["field_surfaces"] = surfaces

    # **근거 없는 선택 기준은 지운다. 만들지 않는다.**
    #
    # 질문에 없는 기준을 모델이 지어내면(「두 배터리 계약」을 「해지 사유」 기준으로)
    # 지금은 응답 **전체**가 버려진다. entity·target·필드·전제까지 같이 잃는다.
    # 지우면 선택만 잃고 나머지는 남는다 — 이 교환은 언제나 이득이다.
    #
    # 모드에 따라 다르게 지운다. 관측된 selection 8건 중 7건이 `latest` 인데,
    # 이것은 기준 없이도 자족한다(「가장 최근」). 반면 `maximum` 계열은 무엇의
    # 최대인지가 없으면 선택 자체가 정의되지 않는다.
    #
    # 「해지한 **두** 계약을 …별로 비교」 같은 질문에는 선택 구성 자체가 없다.
    # 수량은 필터 결과일 뿐 기준이 아니다 — 지우는 쪽이 맞다.
    for item in payload.get("answer_items", []):
        if not isinstance(item, dict):
            continue
        selection = item.get("selection")
        if not isinstance(selection, dict):
            continue
        mode = selection.get("mode")
        criterion = selection.get("criterion_surface")
        if mode in {None, "none"} or not isinstance(criterion, str):
            continue

        # 대상이 이미 「가장 최근 …보고서」로 문서를 못박았다면, 같은 말을
        # 되풀이하는 선택은 **고를 것이 없는 선택**이다.  그대로 두면 컴파일러가
        # 「여럿 중 하나 고르기」로 읽어 계획 자체를 거절한다.
        #
        # 대상 표현이 정기보고서 **최신 지시**로 확정될 때만, 그리고 기준이 그
        # 표현 안에 들어 있을 때만 거둔다.  둘 중 하나라도 아니면 그대로 둔다.
        if mode in {"latest", "earliest"} and criterion:
            surface = _nfc((item.get("target") or {}).get("surface") or "")
            if criterion in surface and _names_latest_periodic(surface):
                selection["mode"] = "none"
                selection["criterion_surface"] = ""
                selection["k"] = 0
                codes.add(SELECTION_RESTATING_TARGET_DROPPED)
                continue
        if not criterion:
            # A non-self-describing selection with no criterion and no
            # question cue is not a selection request.  Keep self-describing
            # latest/earliest modes only when the earlier narrowing step found
            # their exact cue in the question.
            if not _SELECTION_CUE.search(question):
                selection["mode"] = "none"
                selection["criterion_surface"] = ""
                selection["k"] = 0
                codes.add(UNFOUNDED_SELECTION_DROPPED)
            continue
        if criterion in question:
            continue
        # **질문이 선택을 요구했으면 모드를 지운다.** 근거 없는 것은 표면뿐이다.
        #
        # 예전에는 기준 표면이 근거를 잃으면 선택 **전체**를 거뒀다. 그런데
        # 「…우세한 곳은?」처럼 사용자가 승자를 명시적으로 물었는데 모델이 기준을
        # 바꿔 적기만 한 경우에도 승자 요청이 통째로 사라졌다. 동결 Gold 는 실제
        # 최상급이 있을 때 `maximum` 을 **보존**한다.
        #
        # 그래서 질문 자체에 선택 단서가 있는지로 가른다. 있으면 모드를 지키고
        # 근거 없는 표면만 비운다. 없으면 모델이 지어낸 것이므로 거둔다.
        if _SELECTION_CUE.search(question):
            selection["criterion_surface"] = ""
            codes.add(UNFOUNDED_SELECTION_CRITERION_DROPPED)
        elif mode in {"latest", "earliest"}:
            selection["criterion_surface"] = ""
            codes.add(UNFOUNDED_SELECTION_CRITERION_DROPPED)
        else:
            selection["mode"] = "none"
            selection["criterion_surface"] = ""
            selection["k"] = 0
            codes.add(UNFOUNDED_SELECTION_DROPPED)

    # An existing premise can be narrowed only to an exact numeric output
    # field of one of its own applied items.  This runs after field repairs so
    # the proof binds the final bounded surface, never a provider-only alias.
    _recover_asserted_quantity_premise(payload, question, codes)
    _narrow_exact_numeric_existence_premises(payload, question, codes)
    _recover_document_negative_confirmation_premise(payload, question, codes)

    try:
        repaired = HcxSemanticIntentWire.model_validate(payload, strict=True)
    except ValidationError as exc:
        raise SemanticIntentBoundaryError(
            "deterministic semantic repair가 strict wire를 만들지 못했습니다",
            diagnostic_codes=safe_validation_issue_codes(exc),
            diagnostic_paths=safe_validation_issue_paths(exc),
            normalization_codes=tuple(sorted(codes)),
        ) from exc
    return repaired, tuple(sorted(codes))


def normalize_semantic_intent_bounded(
        question: str,
        provider: HcxSemanticIntentWire | Mapping[str, Any],
        *,
        company_surface_regrounder: CompanySurfaceRegrounder | None = None,
        ) -> BoundedSemanticIntent:
    """Apply the reviewed conservative repair and normalize exact surfaces."""

    exact_question = _question(question)
    source = _strict_wire(provider)
    question_nfc = _nfc(exact_question)
    repaired, codes = _repair_roles(
        question_nfc, source,
        company_surface_regrounder=company_surface_regrounder,
    )
    try:
        intent = normalize_semantic_intent(question_nfc, repaired)
    except SemanticIntentNormalizationError as exc:
        raise SemanticIntentBoundaryError(
            str(exc),
            diagnostic_codes=exc.diagnostic_codes,
            diagnostic_paths=exc.diagnostic_paths,
            normalization_codes=codes,
        ) from exc
    return BoundedSemanticIntent(
        question=exact_question,
        source_wire=source,
        repaired_wire=repaired,
        semantic_intent=intent,
        normalization_codes=codes,
        question_sha256=sha256(exact_question.encode("utf-8")).hexdigest(),
        semantic_intent_digest=semantic_intent_digest(intent),
        company_surface_regrounder=company_surface_regrounder,
    )


__all__ = [
    "BoundedSemanticIntent",
    "CompanySurfaceRegrounder",
    "SEMANTIC_INTENT_BOUNDARY_V1",
    "SEMANTIC_INTENT_NORMALIZATION_CODES",
    "SEMANTIC_INTENT_PRE_SCHEMA_REPAIR_CODES",
    "SemanticIntentBoundaryEvidence",
    "SemanticIntentBoundaryError",
    "normalize_semantic_intent_bounded",
    "repair_semantic_intent_wire_payload",
]
