"""공시 서술지표 사전과 그 regrounder.

재무제표 계정이 아니라 정기보고서 본문(주로 「II. 사업의 내용」)에 값이 실리는
지표 — 순이자마진·고정이하여신비율·수주잔고·생산능력·가동률·임상·연구개발비 —
는 재무 개념 사전이 모른다. 그래서 HCX-007 이 `target.kind="metric"` 으로 읽은
질문이 `FinancialResolutionBackend` 에서 확정되지 못하고 `capability_unmatched`
로 끝났다(2026-09-02 실서버 탐침 7/7 실패).

이 모듈은 그런 표면을 **정기공시 narrative 경로(topic)** 로 돌린다. 값을 계산하거나
추정하지 않는다 — 원문 표·문장을 근거로 인용하는 기존 narrative 경로에 좌표만 넘긴다.

세 자리에서 쓰인다.
1. `DisclosedMetricTopicRegrounder` — SemanticIntent 의 metric 표면을 topic 으로 재기입.
2. `PeriodicNarrativeResolutionBackend` — 선택된 정기보고서 본문에 그 표기가 실제로
   있을 때만 `retrieval_query` 로 채택(source-visible 원칙).
3. `app/tools/narrative.py` — 동의어 sub-query 와 「II. 사업의 내용」 우선 검색.

사전은 `agent/disclosed_metric_topics.tsv` 다. 표기는 질문 원문에 그대로 있어야
grounding 을 지키므로 띄어쓰기 변형을 alias 로 따로 적는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.semantic_intent_v1 import SemanticIntent

DISCLOSED_METRIC_TOPICS_FILE = Path(__file__).with_name("disclosed_metric_topics.tsv")

_ALLOWED_TARGET_KINDS = frozenset({"metric", "attribute", "event", "document", "entity"})
#: 「임상 3상 진행 현황」처럼 모델이 event 로 읽는 표기는 **정기보고서를 명시한
#: 질문**에서만 narrative 로 보낸다. 「임상 관련 주요경영사항 공시」는 거래소공시
#: 사건이라 그쪽 경로가 맞다.
_PERIODIC_REPORT_WORD = re.compile(r"사업보고서|분기\s*보고서|반기\s*보고서|분기|반기")
#: 이슈 #150 CG-051 — 「투자판단 관련 주요경영사항」/「투자판단관련주요경영사항」
#: 리터럴 서식명 + 공시 날짜가 함께 있으면 그 거래소 공시 하나를 가리키는
#: 사건 경로(#44/#49 형제)가 우선해야 한다. 이 regrounder가 먼저 잡아 서술
#: 주제(topic)로 돌리면 사건 경로가 절대 못 본다 — PR #148(#122)이 넓힌
#: 표면 매칭이 이 리터럴 서식을 삼킬 수 있으므로 여기서 닫는다.
_INVESTMENT_JUDGMENT_FORM_WORD = re.compile(
    r"투자\s*판단\s*(?:관련)?\s*주요\s*경영\s*사항")
_EXACT_DAY_WORD = re.compile(
    r"(?:19|20)[0-9]{2}\s*년\s*[0-9]{1,2}\s*월\s*[0-9]{1,2}\s*일")


def _is_literal_investment_judgment_document_request(item: Any) -> bool:
    """이 항목이 날짜가 특정된 「투자판단관련주요경영사항」 리터럴 서식 요청인가."""

    surfaces = (
        item.target.surface, *item.target.qualifier_surfaces,
        *item.output.field_surfaces)
    if not any(_INVESTMENT_JUDGMENT_FORM_WORD.search(surface)
               for surface in surfaces):
        return False
    return any(_EXACT_DAY_WORD.search(surface) for surface in surfaces)


class DisclosedMetricTopicError(RuntimeError):
    """사전 파일이 계약을 어겼다. 조용히 건너뛰지 않는다."""


def _compact(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", value or "").casefold()


@dataclass(frozen=True, slots=True)
class DisclosedMetricTopic:
    term: str
    aliases: tuple[str, ...]
    topic_query: str
    section_hint: str
    sector_hint: str

    @property
    def surfaces(self) -> tuple[str, ...]:
        """질문·표면에 그대로 등장할 수 있는 표기 전부(정본 term 우선)."""
        return (self.term, *self.aliases)

    @property
    def search_terms(self) -> tuple[str, ...]:
        """FTS 는 토큰을 AND 로 묶으므로 동의어는 **한 개씩** 따로 검색한다."""
        return tuple(dict.fromkeys(
            [*self.term.split(), *self.topic_query.split()]))


@lru_cache(maxsize=4)
def load_disclosed_metric_topics(
        path: Path | None = None) -> tuple[DisclosedMetricTopic, ...]:
    source = path or DISCLOSED_METRIC_TOPICS_FILE
    if not source.is_file():
        return ()
    entries: list[DisclosedMetricTopic] = []
    seen_terms: set[str] = set()
    seen_surfaces: dict[str, str] = {}
    for number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) != 5:
            raise DisclosedMetricTopicError(
                f"{source.name}:{number} 열이 5개여야 합니다 "
                f"(term · aliases · topic_query · section_hint · sector_hint), "
                f"받은 것 {len(parts)}개")
        term, raw_aliases, topic_query, section_hint, sector_hint = parts
        if not term or not topic_query or not section_hint:
            raise DisclosedMetricTopicError(
                f"{source.name}:{number} term·topic_query·section_hint 는 비울 수 없습니다")
        if term in seen_terms:
            raise DisclosedMetricTopicError(f"{source.name}:{number} term 중복: {term}")
        aliases = tuple(dict.fromkeys(
            alias.strip() for alias in raw_aliases.split("|") if alias.strip()))
        for surface in (term, *aliases):
            owner = seen_surfaces.get(surface)
            if owner is not None and owner != term:
                raise DisclosedMetricTopicError(
                    f"{source.name}:{number} 표기 {surface!r} 가 {owner!r} 와 겹칩니다")
            seen_surfaces[surface] = term
        seen_terms.add(term)
        entries.append(DisclosedMetricTopic(
            term=term, aliases=aliases, topic_query=topic_query,
            section_hint=section_hint, sector_hint=sector_hint))
    return tuple(entries)


def key_present(compact_text: str, key: str) -> bool:
    """``key``(공백 제거 표기)가 ``compact_text`` 안에 **그 개념으로** 있는가.

    부분 문자열 포함만 보면 「미등기임원」이 「등기임원」에도 걸린다 — 부정 접두
    「미」가 붙어 반대 개념(등기하지 않은 임원)이 된 표를 「등기임원」 질문의
    근거로 잘못 세운다(#122 CG-044 실측). ``key`` 앞 글자가 「미」이면 그 자리
    match 는 세지 않는다 — ``key`` 자체가 이미 「미」로 시작하면(그런 사전
    표기가 생기면) 이 규칙을 적용하지 않는다.
    """

    if not key or not compact_text or key.startswith("미"):
        return bool(key) and key in compact_text
    start = compact_text.find(key)
    while start != -1:
        if start == 0 or compact_text[start - 1] != "미":
            return True
        start = compact_text.find(key, start + 1)
    return False


def match_disclosed_metric(
        text: str, *, topics: tuple[DisclosedMetricTopic, ...] | None = None,
        ) -> tuple[DisclosedMetricTopic, str] | None:
    """``text`` 안에 **그대로** 등장하는 가장 긴 표기와 그 사전 항목.

    grounding 이 요구하는 것은 원문 부분 문자열이므로 정규화 비교를 하지 않는다.
    """

    if not isinstance(text, str) or not text:
        return None
    best: tuple[DisclosedMetricTopic, str] | None = None
    for entry in (topics if topics is not None else load_disclosed_metric_topics()):
        for surface in entry.surfaces:
            if surface and surface in text and (
                    best is None or len(surface) > len(best[1])):
                best = (entry, surface)
    return best


_COORDINATION_SPLIT = re.compile(r"\s*(?:과|와|및|,|·|ㆍ)\s*")


def disclosed_metrics_for_query(
        retrieval_query: str, *,
        topics: tuple[DisclosedMetricTopic, ...] | None = None,
        ) -> tuple[DisclosedMetricTopic, ...]:
    """retrieval_query 가 사전 표기(들)를 그대로 가리키면 그 항목들.

    「생산능력과 가동률」처럼 이음말로 묶인 질의는 낱말마다 본다. **모든** 부분이
    사전 표기여야 한다 — 하나라도 아니면 빈 튜플이다(사전 밖 주제를 지표로 오인하지
    않는다). 같은 항목은 한 번만 돌려준다.
    """

    if not isinstance(retrieval_query, str) or not retrieval_query.strip():
        return ()
    entries = topics if topics is not None else load_disclosed_metric_topics()
    by_key = {
        _compact(surface): entry for entry in entries for surface in entry.surfaces
        if _compact(surface)}
    whole = by_key.get(_compact(retrieval_query))
    if whole is not None:
        return (whole,)
    # 「서치플랫폼 부문 매출액」처럼 부문명이 앞에 붙은 부문별 매출 질의.
    segment = next((entry for entry in entries if entry.term == SEGMENT_REVENUE_TERM), None)
    if segment is not None and any(
            _compact(retrieval_query).endswith(_compact(surface))
            for surface in segment.surfaces if _compact(surface)):
        return (segment,)
    parts = [part for part in _COORDINATION_SPLIT.split(retrieval_query) if part.strip()]
    if len(parts) < 2:
        return ()
    found: list[DisclosedMetricTopic] = []
    for part in parts:
        entry = by_key.get(_compact(part))
        if entry is None:
            return ()
        if entry not in found:
            found.append(entry)
    return tuple(found)


#: 부문별 매출 표기 안의 일반 낱말 — 남는 낱말(서치플랫폼·아시아)이 부문명이다.
_SEGMENT_GENERIC = frozenset({
    "부문", "부문별", "사업부문", "사업부문별", "지역", "지역별", "국가별", "제품별",
    "매출", "매출액", "게임", "기준", "별", "사업보고서", "분기보고서", "반기보고서",
    "연결", "별도", "연결기준", "별도기준"})
#: 모델 표면 앞에 붙는 「2025년 사업보고서 기준 」 류 접두 — 부문명이 아니다.
_REPORT_PREFIX = re.compile(
    r"^(?:(?:19|20)\d{2}년\s*)?(?:[1-4]\s*분기(?:\s*보고서)?|반기(?:\s*보고서)?|사업보고서)?"
    r"\s*(?:기준|상)?\s*")
SEGMENT_REVENUE_TERM = "부문별 매출"


def probe_keys(entry: DisclosedMetricTopic, topic: str) -> tuple[str, ...]:
    """본문 대조·발췌에 쓰는 키(정규화). 정본 표기가 먼저다.

    「부문별 매출」은 표기 자체가 아니라 **질문의 부문명**(서치플랫폼·아시아)이 값 행을
    가리키므로, 주제에서 일반 낱말을 뺀 나머지를 키로 쓴다. 남는 낱말이 없으면 표기로
    돌아간다.
    """

    if entry.term == SEGMENT_REVENUE_TERM:
        leftovers = [
            _compact(token) for token in re.split(r"[\s·ㆍ/()]+", topic or "")
            if token and token not in _SEGMENT_GENERIC and _compact(token)
            and not re.fullmatch(r"(?:19|20)\d{2}년?(?:도|말)?|제?\d+기", token)]
        if leftovers:
            return tuple(dict.fromkeys(leftovers))
    keys = [_compact(entry.term)] + sorted(
        {key for key in (_compact(surface) for surface in entry.aliases)
         if key and key != _compact(entry.term)}, key=len, reverse=True)
    return tuple(key for key in keys if key)


def search_terms_for(entry: DisclosedMetricTopic, topic: str) -> tuple[str, ...]:
    """FTS sub-query 목록. 부문별 매출은 부문명을 함께 검색한다."""

    terms = list(entry.search_terms)
    if entry.term == SEGMENT_REVENUE_TERM:
        terms = [token for token in re.split(r"[\s·ㆍ/()]+", topic or "")
                 if token and token not in _SEGMENT_GENERIC] + terms
    return tuple(dict.fromkeys(term for term in terms if term))


def disclosed_metric_for_query(
        retrieval_query: str, *,
        topics: tuple[DisclosedMetricTopic, ...] | None = None,
        ) -> DisclosedMetricTopic | None:
    """retrieval_query 가 가리키는 첫 사전 항목(없으면 ``None``)."""

    found = disclosed_metrics_for_query(retrieval_query, topics=topics)
    return found[0] if found else None


def coordinated_surface_span(question: str, surfaces: list[str]) -> str | None:
    """질문에서 ``surfaces`` 를 순서대로 덮는 최소 연속 구간 — 사이에 이음말만 있을 때.

    「반도체 부문 생산능력과 가동률」→「생산능력과 가동률」. 사이에 다른 낱말이
    있으면 ``None`` — 두 지표 사이의 뜻을 지어내지 않는다.
    """

    if len(surfaces) < 2 or len(set(surfaces)) != len(surfaces):
        return None
    spans: list[tuple[int, int]] = []
    cursor = 0
    for surface in surfaces:
        start = question.find(surface, cursor)
        if start < 0:
            return None
        spans.append((start, start + len(surface)))
        cursor = start + len(surface)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        gap = question[end:start]
        if not re.fullmatch(r"\s*(?:과|와|및|,|·|ㆍ)\s*", gap):
            return None
    value = question[spans[0][0]:spans[-1][1]].strip()
    return value or None


class DisclosedMetricTopicRegrounder:
    """재무 사전이 모르는 공시 서술지표 metric 을 정기공시 topic 으로 재기입한다.

    닫힌 조건에서만 움직인다 — 항목 하나, 회사 하나, 기간 표현 하나, 재무 정본·구어
    사전 모두 미해결, 표기가 사전과 그대로 일치. 하나라도 어긋나면 intent 를 그대로
    돌려주어 종전 경로(재무 → capability_unmatched)가 그대로 진행된다.
    """

    _PERIOD = re.compile(
        r"(?<![0-9])(?:19|20)[0-9]{2}년"
        r"(?:\s*(?:[1-4]\s*분기(?:\s*보고서)?|반기(?:\s*보고서)?|사업보고서))?")

    def __init__(self, company_preflight: Any, *,
                 topics: tuple[DisclosedMetricTopic, ...] | None = None) -> None:
        if not callable(getattr(
                company_preflight, "unique_question_company_surface", None)):
            raise TypeError("서술지표 regrounder company preflight 계약이 잘못되었습니다")
        self.company_preflight = company_preflight
        self.topics = topics

    @staticmethod
    def _financial_dictionary_knows(question: str, surfaces: tuple[str, ...],
                                    matched: str) -> bool:
        """재무 정본·구어 사전이 이 질문을 이미 답하거나 거절하는가."""

        from agent.concept_alias import resolve_from_question
        from agent.planning import CONCEPT_QUESTION_PATTERNS, resolve_metric_concept

        if any(resolve_metric_concept(surface) is not None for surface in surfaces):
            return True
        colloquial = resolve_from_question(question, CONCEPT_QUESTION_PATTERNS)
        if colloquial.status == "unsupported":
            # 진짜 파생지표(영업이익률 등)라 여기서도 살리지 않는다.
            return True
        if colloquial.status in {"resolved", "ambiguous", "route_conflict"}:
            # 「BIS기준 자기자본비율」 안의 「자기자본」(자본총계로 확정), 「보통주
            # 자본비율」 안의 「자본」처럼 더 긴 사전 표기가 구어 표기를 품고 있으면
            # 사전이 이긴다 — 그렇지 않으면 자기자본비율 질문에 자본총계를 답한다
            # (2026-09-02 P9-003 실측).
            inner = _compact(colloquial.surface or "")
            if inner and inner in _compact(matched) and len(_compact(matched)) > len(inner):
                return False
            if colloquial.status == "resolved":
                return True
            # 「기술이전 계약 현황」의 「계약」은 공시 사건 route 로 갈리지만, 질문이
            # 정기보고서를 명시하면 그 보고서의 연구개발활동 절이 출처다.
            if (colloquial.status == "route_conflict"
                    and _PERIODIC_REPORT_WORD.search(question) is not None):
                return False
            return True
        return False

    def _item_surface(self, question: str, item: Any) -> str | None:
        """이 항목이 가리키는 사전 표기(질문 원문 그대로). 아니면 ``None``.

        document-kind 항목의 정기보고서 확인은 `_item_surfaces`(복수형)가 이미
        마쳤다 — target 표면 자체가 아니라 entity 참조로만 정기보고서를 실어
        오는 판(#122 CG-046 실측)도 그쪽에서 받으므로 여기서 다시 걸지 않는다.
        """

        if (item.target.kind not in _ALLOWED_TARGET_KINDS
                or item.selection is not None
                or item.operation != "retrieve"):
            return None
        # 항목 자체의 표면(모델이 쓴 말)에서 먼저 찾고, 없으면 질문에서 찾되 그
        # 표기가 항목 표면 안에 있어야 한다 — 다른 지표를 묻는 항목에 사전 단어가
        # 곁들여진 경우를 걸러낸다.
        model_surfaces = (item.target.surface, *item.output.field_surfaces,
                          *item.target.qualifier_surfaces)
        matched = match_disclosed_metric(item.target.surface, topics=self.topics)
        if matched is None:
            # 모델이 대상 표면을 「사업보고서」처럼 문서 전체로 뭉뚱그리고, 실제
            # 사전 표기는 답 필드 하나에만 그대로 적을 수 있다(#122 CG-042 실측:
            # 「최대주주 및 특수관계인의 주식소유 현황」이 필드 셋 「최대주주」·
            # 「특수관계인」·「주식소유 현황」으로 쪼개짐). 질문 전체에서 가장 긴
            # 표기를 찾으면 그 표기가 어느 모델 표면에도 없어 아래 판정에서
            # 버려지므로, 필드마다 먼저 본다 — 찾은 표기는 그 필드 자신 안에
            # 있으므로 아래 판정을 항상 통과한다.
            field_matches = [
                found for found in (
                    match_disclosed_metric(field, topics=self.topics)
                    for field in item.output.field_surfaces)
                if found is not None]
            if field_matches:
                matched = max(field_matches, key=lambda found: len(found[1]))
        if matched is None:
            matched = match_disclosed_metric(question, topics=self.topics)
        if matched is None:
            return None
        entry, surface = matched
        if item.target.kind == "entity" and entry.term not in {"등기임원", "계열회사"}:
            return None
        if surface not in question:
            return None
        if not any(surface in value for value in model_surfaces):
            return None
        if self._financial_dictionary_knows(
                question, (item.target.surface, surface), surface):
            return None
        # 「순이자마진(NIM)」처럼 괄호 병기 표기가 잡히면 본문 대조는 정본 표기
        # (순이자마진)로 한다 — 병기 형태 그대로는 본문에 없어 「확인되지 않은 항목」
        # 고지가 붙는다. 정본 표기도 질문에 그대로 있을 때만 바꾼다(grounding).
        if "(" in surface and entry.term in question and entry.term in surface:
            return entry.term
        # 부문별 매출은 부문명(서치플랫폼)까지 포함한 모델 표면 전체가 주제다.
        if (entry.term == SEGMENT_REVENUE_TERM
                and item.target.surface in question
                and item.target.surface.endswith(surface)):
            trimmed = _REPORT_PREFIX.sub("", item.target.surface).strip()
            return trimmed if trimmed and trimmed in question and trimmed.endswith(surface) \
                else item.target.surface
        return surface

    @staticmethod
    def _periodic_reference_present(item: Any, intent: SemanticIntent) -> bool:
        """이 document-kind 항목이 정기보고서를 가리키는가.

        보통은 ``target.surface`` 자체에 「사업보고서」가 있다. 그런데 모델이
        정기보고서를 별도 document entity 로 적고 target 은 그 entity 를
        ``entity_refs`` 로만 가리킬 수 있다(#122 CG-046 실측: target.surface
        =「이사회 운영과 주주총회 투표제도」, entity-2 surface=「2025년
        사업보고서」). 그 entity 표기도 함께 본다.
        """

        if _PERIODIC_REPORT_WORD.search(item.target.surface) is not None:
            return True
        entities_by_id = {entity.entity_id: entity for entity in intent.entities}
        return any(
            (entity := entities_by_id.get(ref)) is not None
            and _PERIODIC_REPORT_WORD.search(entity.surface) is not None
            for ref in item.target.entity_refs)

    def _coordinated_field_surfaces(
            self, question: str, item: Any) -> list[str] | None:
        """항목 하나의 답 필드 여럿이 각자 다른 사전 지표를 가리킬 때 그 표기들.

        HCX 는 「이사회 운영과 주주총회 투표제도」를 항목 하나(target=전체 구
        절)에 필드 둘(「이사회 운영」·「주주총회 투표제도」)로 낼 수 있다(#122
        CG-046 실측). ``target.surface`` 하나로만 찾으면 그중 더 긴 표기
        하나만 남아 나머지 절이 조용히 빠진다. 필드마다 찾고, **전부** 사전에
        걸리고 서로 다르며 질문에서 이음말로만 이어진 연속 구간일 때만
        받는다 — 하나라도 아니면 ``None``으로 물러나 기존 단일 표면 경로에
        맡긴다.
        """

        fields = list(item.output.field_surfaces)
        if len(fields) < 2:
            return None
        found: list[str] = []
        for field in fields:
            matched = match_disclosed_metric(field, topics=self.topics)
            if matched is None:
                return None
            entry, surface = matched
            if surface not in question or surface in found:
                return None
            if self._financial_dictionary_knows(question, (field, surface), surface):
                return None
            found.append(surface)
        if coordinated_surface_span(question, found) is None:
            return None
        return found

    def _coordinated_target_and_field(
            self, question: str, item: Any) -> list[str] | None:
        """target 표면 자체가 지표 하나, 필드가 지표 하나뿐인 이음말 항목.

        HCX 는 같은 「이사회 운영과 주주총회 투표제도」를 판마다 다르게
        읽는다 — 필드 둘로(`_coordinated_field_surfaces`) 낼 때도, target=
        「이사회 운영」·필드=[「주주총회 투표제도」]처럼 대상과 필드 하나씩을
        나눠 낼 때도 있다(#122 CG-046 실측: 두 판이 실제로 관측됨). 필드가
        정확히 하나일 때만 target 표면도 후보로 같이 본다 — 필드가 둘 이상인
        경우는 위 메서드가 이미 받으므로 여기서 다시 보지 않는다.
        """

        fields = list(item.output.field_surfaces)
        if len(fields) != 1 or item.target.surface == fields[0]:
            return None
        found: list[str] = []
        for candidate in (item.target.surface, fields[0]):
            matched = match_disclosed_metric(candidate, topics=self.topics)
            if matched is None:
                return None
            entry, surface = matched
            if surface not in question or surface in found:
                return None
            if self._financial_dictionary_knows(question, (candidate, surface), surface):
                return None
            found.append(surface)
        if coordinated_surface_span(question, found) is None:
            return None
        return found

    def _item_surfaces(
            self, question: str, item: Any, intent: SemanticIntent,
            ) -> list[str] | None:
        """이 항목이 가리키는 사전 표기 목록(보통 하나, 다중 필드면 여럿).

        ``None`` 이면 이 항목은 서술지표 재기입 대상이 아니다 — 호출자는
        intent 전체를 손대지 않고 돌려준다."""

        if (item.target.kind not in _ALLOWED_TARGET_KINDS
                or item.selection is not None
                or item.operation != "retrieve"):
            return None
        if _is_literal_investment_judgment_document_request(item):
            return None
        if (item.target.kind == "document"
                and not self._periodic_reference_present(item, intent)):
            # 「사업보고서」를 target 이나 entity 어디로도 확인할 수 없으면
            # 받지 않는다.
            return None
        multi = (self._coordinated_field_surfaces(question, item)
                 or self._coordinated_target_and_field(question, item))
        if multi is not None:
            return multi
        single = self._item_surface(question, item)
        return [single] if single is not None else None

    def _unique_company_surface(
            self, question: str, surfaces: list[str]) -> str | None:
        """질문에 유일하게 등장하는 회사 표기.

        사전 표기는 회사명이 아니다. 그런데 「CSM」 안의 「SM」이 에스엠 별칭으로
        잡혀 회사가 둘로 보이는 식의 오탐이 있으므로, 유일 판정이 실패하면 사전
        표기를 공백으로 가린 질문으로 한 번 더 판정한다. 돌려주는 표기는 원 질문에
        그대로 있는 회사명이라 grounding 은 그대로 지켜진다.
        """

        found = self.company_preflight.unique_question_company_surface(question)
        if found is not None:
            return found
        masked = question
        for entry, _ in (match_disclosed_metric(surface, topics=self.topics)
                         for surface in surfaces):
            for alias in entry.surfaces:
                masked = masked.replace(alias, " " * len(alias))
        if masked != question:
            found = self.company_preflight.unique_question_company_surface(masked)
            if found is not None and found in question:
                return found
        return None

    @staticmethod
    def _entity_company_surface(question: str, intent: SemanticIntent) -> str | None:
        """정본 preflight 가 회사를 하나로 못 좁힐 때(「에코프로비엠」 안의 「에코프로」)
        모델이 읽은 회사 entity 가 하나뿐이고 질문에 그대로 있으면 그것을 쓴다.
        표면은 이미 grounding 을 통과한 질문 구간이다."""

        companies = [entity.surface for entity in intent.entities
                     if entity.kind_hint == "company" and entity.surface in question]
        return companies[0] if len(companies) == 1 else None

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str)
                or not intent.answer_items
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        from agent.stage1_v1_narrative_grounding_fallback import explicit_rnd_classification_intent
        classified = explicit_rnd_classification_intent(
            question, company_surface_regrounder=lambda surface, raw: (
                surface if self.company_preflight.unique_question_company_surface(raw) == surface
                else None))
        if classified is not None:
            return classified
        # 「생산능력과 가동률」처럼 항목이 둘 이상이면 **전부** 사전 지표여야 한다.
        # 재무 계정과 섞이면 어느 경로도 통째로 맡을 수 없으므로 손대지 않는다.
        surfaces: list[str] = []
        for item in intent.answer_items:
            item_surfaces = self._item_surfaces(question, item, intent)
            if item_surfaces is None:
                return intent
            surfaces.extend(item_surfaces)
        # The same disclosed metric can be requested for two scopes (bank
        # versus bank + card). Retrieve its one source table once; the full
        # question, not a deduplicated answer, controls row projection.
        surfaces = list(dict.fromkeys(surfaces))
        periods = list(dict.fromkeys(
            match.group(0).strip() for match in self._PERIOD.finditer(question)))
        # 「2025년 사업보고서 기준 2025년 말 …」처럼 같은 연도가 되풀이되면 보고서를
        # 명시한 쪽(가장 긴 표현)으로 합친다. 연도가 둘 이상이면 비교 질문이라 물러난다.
        years = {re.match(r"(?:19|20)\d{2}", period).group(0) for period in periods}
        explicit_reports = [period for period in periods
                            if _PERIODIC_REPORT_WORD.search(period)]
        if len(explicit_reports) == 1:
            # Years of columns inside one explicitly named report are not
            # requests to fetch three different annual reports.
            periods = explicit_reports
        if len(years) == 1 and len(periods) > 1:
            periods = [max(periods, key=len)]
        if len(periods) != 1:
            return intent
        if (any(item.target.kind == "event" for item in intent.answer_items)
                and _PERIODIC_REPORT_WORD.search(periods[0]) is None):
            return intent
        company_surface = (self._unique_company_surface(question, surfaces)
                           or self._entity_company_surface(question, intent))
        if company_surface is None:
            return intent
        # 지표가 둘 이상이면 **한 항목**으로 낸다 — 결정적 컴파일러는 같은 문서를 향한
        # narrative 항목 둘을 바인딩하지 못한다(실측 compiler_binding_failed). 질문에서
        # 이음말로만 이어진 연속 구간(「생산능력과 가동률」)이어야 하고, 각 지표는
        # 답 필드로 남겨 backend 가 다중 topic 질의를 만들게 한다.
        if len(surfaces) == 1:
            target_surface = surfaces[0]
        else:
            target_surface = coordinated_surface_span(question, surfaces)
            if target_surface is None:
                return intent
        return SemanticIntent.model_validate({
            "schema_version": intent.schema_version,
            "entities": [{
                "entity_id": "entity-1", "kind_hint": "company",
                "surface": company_surface,
            }],
            "answer_items": [{
                "item_id": "item-1", "operation": "retrieve",
                "target": {
                    "kind": "topic", "surface": target_surface,
                    "entity_refs": ["entity-1"], "qualifier_surfaces": [],
                },
                "scope": {
                    "target_period_expressions": [],
                    "as_of_expression": None,
                    "document_group_expression": periods[0],
                    "scope_qualifier_expressions": [],
                },
                "selection": None,
                "output": {
                    "shape": "narrative",
                    "projection_mode": "named_fields",
                    "field_surfaces": list(surfaces), "presentation": "auto",
                },
            }],
            "answer_groups": [], "premises": [],
            "unresolved_mentions": [], "presentation": "auto",
        }, strict=True)


__all__ = [
    "DISCLOSED_METRIC_TOPICS_FILE",
    "DisclosedMetricTopic",
    "DisclosedMetricTopicError",
    "DisclosedMetricTopicRegrounder",
    "SEGMENT_REVENUE_TERM",
    "coordinated_surface_span",
    "disclosed_metric_for_query",
    "disclosed_metrics_for_query",
    "key_present",
    "probe_keys",
    "search_terms_for",
    "load_disclosed_metric_topics",
    "match_disclosed_metric",
]
