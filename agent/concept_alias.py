"""계정 개념의 **승인 구어 질문 패턴** 층.

정본 계정 사전(`src/ingest/account_map.tsv`)은 공시 계정명으로 만들어졌다. 사람은
「얼마나 팔았어」로 묻고 `팔았어` 는 사전에 없다. 그러면 개념이 확정되지 않아
역질문이 되는데, 한국어 사용자가 망설임 없이 답하는 질문을 되묻는 것은 제품 품질의
손실이다.

**단순한 `surface → concept` 사전으로는 안 된다.** 「팔았어」는 제품 판매(매출),
자산 처분, 사업 매각을 모두 뜻할 수 있다. 표기 하나가 유일 후보를 가리킨다는
이유만으로 자동 확정하면 조용히 틀린 계정을 조회한다 (외부 검수 지적).

그래서 표기마다 **단서 조건**을 함께 둔다.

```
얼마나 팔았어 · 얼마 팔았어      금액 단서 있음        → revenue
설비 팔았어 · 공장 팔았어        대상 명사 있음(우선)   → disposal_ppe
몇 개 팔았어                   수량 단서 있음        → 미지원 (수량 개념이 없다)
팔았어?                       단서 없음            → 역질문
```

「얼마」와 「얼마에」와 「몇 개」는 서로 다른 질문이다. 이 규칙은 **단서의 부재가
아니라 존재**에 기댄다 — 부재에 기대면 「자산어가 없으면 매출」이 되어 조용히
기울고, 오늘 그 형태(질문·출력 mention 수렴 grounding)를 이미 한 번 철회했다.

**금지 단서가 필요 단서를 이긴다.** 「설비 얼마나 팔았어」는 금액 단서가 있어도
처분이다.

이 층의 위험은 회사 level 5 와 같다. 코퍼스 유래 층은 틀리면 거부되지만 **승인 층이
틀리면 조용히 다른 계정을 조회한다.** 그래서 사람 검토를 받고, 개념을 **이름으로**
적어 등록되지 않은 개념을 가리키면 적재가 실패한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal
import unicodedata

from .contracts import FinancialConcept


#: 승인 질문 패턴 파일. 검토된 표에서 생성한다
#: (`scripts/build_concept_alias_registry.py`).
APPROVED_CONCEPT_ALIAS_FILE = Path(__file__).resolve().parent / (
    "concept_question_patterns.tsv")

#: 패턴의 해소 방식. 외부 검수의 판정 이름을 그대로 쓴다.
ResolutionMode = Literal[
    "AUTO",           # 표기만으로 확정. 명시적 계정 표현
    "CONTEXT",        # 필요 단서가 있고 금지 단서가 없을 때만 확정
    "CLARIFY",        # 지원 후보가 둘 이상 — 역질문
    "ROUTE_CLARIFY",  # 재무계정과 공시·사건 경로가 함께 가능 — route 부터 확인
    "BLOCK",          # 의미는 알지만 지원 밖 — 미지원 안내
]

_MODES: frozenset[str] = frozenset(
    ("AUTO", "CONTEXT", "CLARIFY", "ROUTE_CLARIFY", "BLOCK"))

#: 결과 상태. **모호(물어야 함)와 미지원(물어도 못 답함)을 섞지 않는다.**
OutcomeStatus = Literal[
    "resolved", "ambiguous", "unsupported", "route_conflict", "unknown",
]

_NONE_TOKENS = frozenset(("", "-", "—", "none", "unspecified"))
#: 9열(선택)의 참 값. 열이 없으면 꺼진 것으로 읽는다.
_FANOUT_TOKENS = frozenset(("fanout", "yes", "y", "true"))


class ApprovedConceptAliasError(RuntimeError):
    """승인 파일이 계약을 어겼다. **조용히 건너뛰지 않는다.**"""


def normalize_surface_key(value: str) -> str:
    """표기를 비교 가능한 형태로 좁힌다.

    `agent/planning.py` 가 이 함수를 `_key` 로 가져다 쓴다 — 정의가 두 곳에 있으면
    승인 층과 정본 사전이 **다른 키로** 같은 표기를 보게 된다.
    """

    normalized = unicodedata.normalize("NFC", value).casefold()
    return re.sub(r"[\s·ㆍ㈜()_-]+", "", normalized)


#: 「파생지표」 question_kind. 이 종류의 BLOCK 은 표기 길이 경쟁 **밖에서** 먼저
#: 판정한다. 「순이익이 자기자본의 몇 퍼센트」에서 `자기자본`(4자)이 `의몇퍼센트`와
#: 길이로 겨루면 자본총계를 조용히 답한다. 표기가 겹치는 문제가 아니라 **질문
#: 종류가 다른** 것이므로 경쟁시키지 않는다.
DERIVED_METRIC_KIND = "derived_metric"

#: 이슈 #38 — **이름 있는 비율**의 결정적 사전(표기만). 분자·분모 개념·표시
#: 방식(percent/multiple)은 `agent/stage1_v1_financial_backend._ratio_request`
#: 의 몫이다 — 이 모듈은 `planning.py` **아래**에 있어(`planning`이 이 모듈을
#: 가져다 쓴다) 정본 계정 사전(`resolve_metric_concept`)을 가져올 수 없다.
#: 여기서는 순수하게 「표기 경쟁을 건너뛸지」만 결정한다 — 실제로 두 정본
#: 개념·기간·scope가 하나로 닫히는지는 그 함수가 따로 확인하고, 못 닫으면
#: `FinancialResolutionBackend.resolve()` 가 여전히 `None`(확정 실패)이다.
NAMED_RATIO_SURFACES: frozenset[str] = frozenset(
    normalize_surface_key(name) for name in (
        "영업이익률", "순이익률", "매출원가율", "부채비율", "유동비율",
        "자기자본비율", "ROE", "ROA",
    ))

def _ratio_dictionary_closes(question_key: str) -> bool:
    """질문이 이름 있는 비율(`NAMED_RATIO_SURFACES`)을 담고 있는가.

    참이면 `derived_metric` BLOCK 패턴(비율·이익률 등)이 이 질문을 막지 않는다
    — 사전 밖 조합(회전율·주당배당금 등)은 그대로 막힌다.

    **「A를 B로 나눈」·「A가 B의 몇 배」 같은 명시적 나눗셈은 일부러 여기서
    안 푼다.** 이 모듈은 정본 계정 사전(`resolve_metric_concept`, `planning.py`)
    을 가져올 수 없어(순환 임포트) 두 표현이 실제 정본 개념을 가리키는지
    모양만으로는 확인할 수 없다 — 「배당 총액을 발행주식 수로 나눈」처럼
    사전 밖 나눗셈까지 모양이 같다는 이유로 풀면 안 된다(challenge set 회귀로
    확인됨). 그 문형은 `agent/stage1_v1_financial_backend._ratio_request` 가
    이 콜로퀴얼 층보다 **먼저** 가로채 정본 개념으로 직접 닫으므로, 이 층이
    「unsupported」로 남아 있어도 실제 답변 경로는 막히지 않는다.
    """

    return any(name in question_key for name in NAMED_RATIO_SURFACES)


@dataclass(frozen=True, slots=True)
class ConceptQuestionPattern:
    """구어 표기 하나와 그 해소 조건."""

    pattern_id: str
    surface: str
    key: str
    mode: ResolutionMode
    candidates: tuple[FinancialConcept, ...]
    required_cues: tuple[str, ...]
    forbidden_cues: tuple[str, ...]
    question_kind: str
    note: str
    #: 이 표기를 **안에 포함하는 더 긴 정본 계정 표기**들. 질문에 그것이 있으면
    #: 이 구어 패턴은 적용하지 않는다 — 「총포괄손익」 질문에서 「포괄손익」 구어가
    #: 잡히면 정본 표기가 있는데도 역질문이 된다.
    #:
    #: 정본 사전은 **모델의 표기**만 검사하는데 이 층은 **질문**을 훑는다. 그
    #: 비대칭이 이 필드의 존재 이유다.
    covering_surfaces: tuple[str, ...] = ()
    #: 후보 집합 자체가 「대표 요약」이라 **묻는 대신 전부 답해도 되는가**.
    #: `CLARIFY` 에서만 켤 수 있다(#94 25).
    #:
    #: 「벌었어」의 후보 셋(매출액·영업이익·당기순이익)은 손익계산서를 위에서
    #: 아래로 읽은 세 단계다 — 「얼마나 벌었는가」에 셋을 나란히 주는 것이 하나를
    #: 고르라고 되묻는 것보다 낫다. 「빚」의 셋(총부채·유동·비유동)도 총계와 그
    #: 분해라 같다.
    #:
    #: **모든 CLARIFY 가 그렇지는 않다.** 「현금」의 후보는 잔액 하나와 현금흐름표
    #: 세 줄이 섞여 있어, 잔액을 물은 사람에게 흐름 셋을 얹으면 묻지 않은 값을
    #: 답하는 것이 된다. 「투자」도 같다. 그래서 opt-in 이다.
    fanout: bool = False

    def cue_verdict(self, question_key: str) -> tuple[bool, str | None]:
        """이 패턴이 질문에 적용되는가. ``(적용 가능, 막은 금지 단서)``.

        **금지 단서가 필요 단서를 이긴다.** 「설비 얼마나 팔았어」는 금액 단서가
        있어도 처분이므로, 매출 패턴은 여기서 막혀야 한다.
        """

        for surface in self.covering_surfaces:
            if surface in question_key:
                return False, surface
        for cue in self.forbidden_cues:
            if cue in question_key:
                return False, cue
        if self.required_cues and not any(
                cue in question_key for cue in self.required_cues):
            return False, None
        return True, None


@dataclass(frozen=True, slots=True)
class ColloquialOutcome:
    """구어 해소 결과. **역질문과 미지원을 구분해서 담는다.**"""

    status: OutcomeStatus
    concept: FinancialConcept | None = None
    candidates: tuple[FinancialConcept, ...] = ()
    surface: str | None = None
    pattern_id: str | None = None
    question_kind: str | None = None
    note: str | None = None
    blocked_by: str | None = None
    #: `ambiguous` 일 때만 뜻이 있다 — 후보를 묻지 않고 전부 답해도 되는가.
    #: 근거는 `ConceptQuestionPattern.fanout`.
    fanout: bool = False


def _split_cues(value: str) -> tuple[str, ...]:
    if value.strip().casefold() in _NONE_TOKENS:
        return ()
    return tuple(dict.fromkeys(
        normalize_surface_key(part) for part in value.split("|")
        if part.strip()))


def load_concept_question_patterns(
        known_surfaces: frozenset[str] | None = None, *,
        path: Path | None = None,
        ) -> tuple[ConceptQuestionPattern, ...]:
    """승인 질문 패턴을 **긴 표기 우선**으로 돌려준다.

    긴 표기가 먼저 와야 `설비팔았어` 가 `팔았어` 보다 먼저 잡힌다. 짧은 표기가
    이기면 대상이 있는 질문이 매출로 해소된다.

    `known_surfaces` 를 주면 **정본 사전이 이미 아는 표기는 건너뛴다** — 승인 층이
    검증 가능한 매핑을 덮지 못하게 한다.

    파일이 없으면 빈 결과다. 구어층이 아직 승인되지 않은 상태에서도 조회는
    계속돼야 하고, 그때 동작은 이 층이 없던 때와 **완전히 같다.**
    """

    source = path or APPROVED_CONCEPT_ALIAS_FILE
    if not source.is_file():
        return ()
    by_name = {concept.value: concept for concept in FinancialConcept}
    patterns: list[ConceptQuestionPattern] = []
    seen_ids: set[str] = set()
    seen_keys: set[tuple] = set()

    for number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) < 8:
            raise ApprovedConceptAliasError(
                f"{source.name}:{number} 열이 8개여야 합니다 "
                f"(pattern_id · surface · mode · candidates · required_cues · "
                f"forbidden_cues · question_kind · note), 받은 것 {len(parts)}개")
        pattern_id, surface, mode, raw_candidates = parts[:4]
        required, forbidden, question_kind, note = parts[4:8]
        # 9열은 **선택**이다. 이 표는 172행인데 fanout 을 켜는 행은 둘뿐이라,
        # 나머지 170행에 «-» 를 채워 넣는 대신 열을 생략하게 둔다.
        raw_fanout = parts[8] if len(parts) > 8 else "-"
        if not pattern_id or not surface:
            raise ApprovedConceptAliasError(
                f"{source.name}:{number} pattern_id·surface 가 비어 있습니다")
        if pattern_id in seen_ids:
            raise ApprovedConceptAliasError(
                f"{source.name}:{number} pattern_id 중복: {pattern_id}")
        if mode not in _MODES:
            raise ApprovedConceptAliasError(
                f"{source.name}:{number} 알 수 없는 mode: {mode!r} "
                f"(허용 {sorted(_MODES)})")
        names = [] if raw_candidates.casefold() in _NONE_TOKENS else [
            name for name in (
                part.strip() for part in raw_candidates.split("|")) if name]
        unknown = [name for name in names if name not in by_name]
        if unknown:
            # 개념을 이름으로 적게 한 이유가 이것이다 — 등록되지 않은 개념을
            # 가리키면 조용히 죽는 대신 적재가 실패한다.
            raise ApprovedConceptAliasError(
                f"{source.name}:{number} 등록되지 않은 개념: {unknown}")
        candidates = tuple(dict.fromkeys(by_name[name] for name in names))
        if mode == "BLOCK":
            if candidates:
                raise ApprovedConceptAliasError(
                    f"{source.name}:{number} BLOCK 은 후보를 가질 수 없습니다")
        elif not candidates and mode != "ROUTE_CLARIFY":
            raise ApprovedConceptAliasError(
                f"{source.name}:{number} {mode} 는 후보가 필요합니다")
        # ROUTE_CLARIFY 는 **후보 0개도 정상**이다. 「계약금액」처럼 애초에 재무
        # 개념이 아닌 표기가 재무 task 로 들어온 경우다 — 재무 후보가 있을 수 없다.
        # 이 층은 재무 task 에서만 돌므로(`_colloquial_concept_patch`), 여기 걸렸다는
        # 것 자체가 task 종류를 잘못 골랐다는 신호다. 미지원으로 거절하면 「지원하지
        # 않는 재무 concept」이 되는데, 사용자는 지원되는 공시 값을 물은 것이다.
        if mode in ("AUTO", "CONTEXT") and len(candidates) != 1:
            raise ApprovedConceptAliasError(
                f"{source.name}:{number} {mode} 는 후보가 정확히 1개여야 합니다 "
                f"(받은 것 {len(candidates)}개). 둘 이상이면 CLARIFY 입니다")
        if mode == "CLARIFY" and len(candidates) < 2:
            # 후보가 하나인데 CLARIFY 면 사용자가 고를 것이 없다.
            raise ApprovedConceptAliasError(
                f"{source.name}:{number} {mode} 는 후보가 2개 이상이어야 합니다")
        if raw_fanout.casefold() in _FANOUT_TOKENS:
            fanout = True
        elif raw_fanout.casefold() in _NONE_TOKENS:
            fanout = False
        else:
            raise ApprovedConceptAliasError(
                f"{source.name}:{number} 알 수 없는 fanout 값: {raw_fanout!r} "
                f"(허용 {sorted(_FANOUT_TOKENS)} 또는 빈 값)")
        if fanout and mode != "CLARIFY":
            # AUTO·CONTEXT 는 이미 하나로 닫히고, BLOCK 은 후보가 없으며,
            # ROUTE_CLARIFY 는 재무 task 인지부터 확인해야 한다 — 어느 쪽도
            # 「후보를 전부 답한다」가 뜻이 통하지 않는다.
            raise ApprovedConceptAliasError(
                f"{source.name}:{number} fanout 은 CLARIFY 에서만 켤 수 "
                f"있습니다 (mode={mode})")
        # ROUTE_CLARIFY 는 **후보 1개도 정상**이다. 사용자에게 보이는 선택지는
        # 「재무제표 숫자」와 「공시·사건」 둘이고 후보 목록은 노출되지 않는다
        # (`_route_clarification`). 2개를 강요하면 관계없는 개념을 채워 넣게 되고,
        # 사용자가 「재무제표 숫자」를 고른 다음 회차에서 그 쓰레기 후보가 나온다.
        # 실제로 「배당」에 `net_income` 이 그렇게 들어가 있었다 (외부 검토 §2).
        required_cues = _split_cues(required)
        forbidden_cues = _split_cues(forbidden)
        # BLOCK 도 단서를 가질 수 있다 — 「세금」은 동사에 따라 미지원(현금 납부)과
        # 역질문(비용·자산·부채)으로 갈린다.
        if mode == "CONTEXT" and not required_cues:
            raise ApprovedConceptAliasError(
                f"{source.name}:{number} CONTEXT 는 required_cues 가 필요합니다 "
                f"— 없으면 AUTO 와 같습니다")
        overlap = sorted(set(required_cues) & set(forbidden_cues))
        if overlap:
            raise ApprovedConceptAliasError(
                f"{source.name}:{number} 같은 단서가 필요·금지 양쪽에 있습니다: "
                f"{overlap}")
        key = normalize_surface_key(surface)
        # **표기 하나에 패턴이 여럿일 수 있다.** 「설비」는 동사에 따라 잔액·취득·
        # 처분으로 갈리므로(검수서 §2-4) 표기로 유일성을 걸면 그 구분을 표현할 수
        # 없다. 유일해야 하는 것은 `pattern_id` 다.
        if (key, tuple(sorted(_split_cues(required)))) in seen_keys:
            raise ApprovedConceptAliasError(
                f"{source.name}:{number} 표기+필요단서 조합이 중복입니다: {surface}")
        seen_keys.add((key, tuple(sorted(_split_cues(required)))))
        seen_ids.add(pattern_id)
        if known_surfaces is not None and key in known_surfaces:
            continue
        covering = tuple(sorted(
            (known for known in (known_surfaces or ())
             if len(known) > len(key) and key in known),
            key=lambda value: (-len(value), value)))
        patterns.append(ConceptQuestionPattern(
            pattern_id=pattern_id, surface=surface, key=key, mode=mode,
            candidates=candidates, required_cues=required_cues,
            forbidden_cues=forbidden_cues,
            question_kind=question_kind, note=note,
            covering_surfaces=covering, fanout=fanout))

    # **구체적인 것부터 본다.**
    #   ① 긴 표기 우선 — `설비팔았어` 가 `팔았어` 보다 먼저
    #   ② 같은 표기면 필요 단서가 많은 것 우선 — `설비+팔았` 이 `설비` 단독보다 먼저
    #   ③ 그다음 pattern_id — 같은 구체성이면 결정론적으로
    # 순서를 뒤집으면 단독 CLARIFY 가 단서 있는 확정을 가로채 역질문이 된다.
    patterns.sort(key=lambda row: (
        -len(row.key), -len(row.required_cues), row.pattern_id))
    return tuple(patterns)


def refine_canonical(
        mention: str, question: str,
        patterns: tuple["ConceptQuestionPattern", ...],
        canonical: "FinancialConcept",
        ) -> "FinancialConcept | None":
    """정본 사전이 답한 개념을 **질문의 단서가 뒤집는 경우**만 바로잡는다.

    모델 표기는 grounding 때문에 **질문의 조각**이다(`_ground_surface`). 그래서
    조각만 보면 맞는데 문장 전체로는 틀린 답이 나온다 — 실측 사례:

    ```
    질문  기초와 기말을 비교했을 때 현금성 자산의 순변동액은 얼마야?
    표기  「현금성 자산」 → 정본 사전 → cash_and_equivalents   (잔액)
    질문  「순변동액」                                        (순증감)
    ```

    정본 사전은 표기만 보므로 이 어긋남을 볼 수 없고, 구어층은 정본이 답하면
    아예 돌지 않아 잡을 기회가 없었다.

    **좁게 연다.** 뒤집는 조건 셋을 모두 만족해야 한다.

    1. 패턴의 표기가 **모델이 고른 그 표기와 같다** — 문장 아무 데나 걸린 다른
       구어가 정본을 덮지 못한다.
    2. 패턴에 **필요 단서가 있고** 질문이 그 단서를 갖고 있다 — 단서 없는
       기본형은 정본을 이기지 못한다.
    3. 개념 하나로 확정되는 패턴이다 — 역질문으로 떠넘기지 않는다.

    조건을 못 채우면 `None` 이고 정본 사전의 답이 그대로 간다.
    """

    key = normalize_surface_key(mention)
    if not key:
        return None
    question_key = normalize_surface_key(question)
    for pattern in patterns:
        if pattern.key != key or not pattern.required_cues:
            continue
        if pattern.mode != "CONTEXT" or len(pattern.candidates) != 1:
            continue
        applies, _ = pattern.cue_verdict(question_key)
        if not applies:
            continue
        if pattern.candidates[0] is canonical:
            return None
        return pattern.candidates[0]
    return None


def _derived_metric_verdict(
        question_key: str, patterns: tuple["ConceptQuestionPattern", ...],
        ) -> "ColloquialOutcome | None":
    """계산으로만 나오는 지표를 **표기 경쟁 이전에** 거절한다.

    v0.4 파생 연산자는 `difference`·`absolute_difference`·`percent_change`·
    `discrete_from_cumulative`·`argmax` 다 — **나눗셈이 없다.** 그래서 두 개념을
    나누는 지표(ROE·부채비율·회전율·주당배당금)는 조회할 방법 자체가 없다.

    반대로 **같은 개념의 두 시점 비교**는 `percent_change` 로 지원한다. 그래서
    「퍼센트」를 통으로 막으면 안 된다 — 동결 문항 G-A-005(「2024년보다 몇 퍼센트
    증가했는가」)가 `ready` 로 답지에 있다. 나눗셈 단서만 잡는다.

    이슈 #38(v0.5) 이후로는 나눗셈 자체가 없는 것이 아니라 **결정적 사전 밖**
    나눗셈만 여기서 막는다 — 영업이익률·부채비율 등은 `_ratio_dictionary_closes`
    가 참이면 지나가고, 그 뒤 `_ratio_request` 가 실제로 두 정본 개념·기간·
    scope 를 하나로 닫을 때만 계산된다. 회전율·주당배당금 등 사전 밖은 여전히
    여기서 막힌다.
    """

    for pattern in patterns:
        if pattern.question_kind != DERIVED_METRIC_KIND:
            continue
        if pattern.key not in question_key:
            continue
        applies, _ = pattern.cue_verdict(question_key)
        if not applies:
            continue
        if _ratio_dictionary_closes(question_key):
            continue
        return ColloquialOutcome(
            status="unsupported", surface=pattern.surface,
            pattern_id=pattern.pattern_id,
            question_kind=pattern.question_kind, note=pattern.note)
    return None


def resolve_from_question(
        question: str, patterns: tuple[ConceptQuestionPattern, ...],
        ) -> ColloquialOutcome:
    """질문 본문에서 승인 구어를 찾아 개념을 정한다.

    **모델이 쓴 표기가 아니라 질문을 본다.** grounding 이 모델 표기를 버리면
    개념이 `unspecified` 가 되고 원래 사람이 쓴 말은 질문에만 남는다 — 그 말을
    승인 사전으로 해소하는 것이 이 층의 일이다. 질문에 있는 말이므로 grounding
    위반이 아니다.

    긴 표기부터 본다. 첫 번째로 **적용 가능한** 패턴이 답이다 — 금지 단서에 막힌
    패턴은 건너뛰고 다음 후보를 본다(「설비 팔았어」에서 매출 패턴이 막히면
    처분 패턴이 잡힌다).
    """

    if not patterns:
        return ColloquialOutcome(status="unknown")
    question_key = normalize_surface_key(question)
    derived = _derived_metric_verdict(question_key, patterns)
    if derived is not None:
        return derived
    blocked: tuple[str, str] | None = None
    for pattern in patterns:
        if pattern.key not in question_key:
            continue
        applies, blocker = pattern.cue_verdict(question_key)
        if not applies:
            if blocker is not None and blocked is None:
                blocked = (pattern.surface, blocker)
            continue
        if pattern.mode == "BLOCK":
            if (pattern.question_kind == DERIVED_METRIC_KIND
                    and _ratio_dictionary_closes(question_key)):
                continue
            return ColloquialOutcome(
                status="unsupported", surface=pattern.surface,
                pattern_id=pattern.pattern_id,
                question_kind=pattern.question_kind, note=pattern.note)
        if pattern.mode == "ROUTE_CLARIFY":
            return ColloquialOutcome(
                status="route_conflict", candidates=pattern.candidates,
                surface=pattern.surface, pattern_id=pattern.pattern_id,
                question_kind=pattern.question_kind, note=pattern.note)
        if pattern.mode == "CLARIFY":
            return ColloquialOutcome(
                status="ambiguous", candidates=pattern.candidates,
                surface=pattern.surface, pattern_id=pattern.pattern_id,
                question_kind=pattern.question_kind, note=pattern.note,
                fanout=pattern.fanout)
        return ColloquialOutcome(
            status="resolved", concept=pattern.candidates[0],
            candidates=pattern.candidates, surface=pattern.surface,
            pattern_id=pattern.pattern_id,
            question_kind=pattern.question_kind, note=pattern.note)
    if blocked is not None:
        # 표기는 찾았는데 금지 단서에 막혔고 대체 패턴도 없다. **매출로 기울지
        # 않고** 모르는 것으로 둔다 — 조용히 고르는 것을 막는 것이 이 층의 목적이다.
        return ColloquialOutcome(
            status="unknown", surface=blocked[0], blocked_by=blocked[1])
    return ColloquialOutcome(status="unknown")
