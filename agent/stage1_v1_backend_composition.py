"""v1 결의 백엔드를 순서대로 태우는 합성기와 사건 역질문 백엔드.

`Stage1V1Resolver` 는 백엔드를 하나만 받는다.  실제 질문은 재무·사건·문서로
갈리므로, 어느 하나가 전부를 덮을 수 없다.  여기서는 **확정한 첫 백엔드의
권위를 쓰고, 아무도 확정하지 못하면 물러난다.**

물러남(`None`)은 실패가 아니라 계약이다.  확정하지 못한 자리에서 무언가를
돌려주면 거절해야 할 질문에 답이 생긴다.  호출자는 물러난 문항을 후보 없음으로
남기거나 다른 경로로 보낸다.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import date
from pathlib import Path
import re
from threading import Lock
from typing import Any, Callable, Mapping, Protocol

from agent.event_clarification_options_v1 import (
    EventMetadataCanonical,
    build_event_clarification_decision,
    build_event_clarification_options,
)
from agent.date_surface import parse_date_surface, question_date_surfaces
from agent.concept_alias import normalize_surface_key, resolve_from_question
from agent.planning import (
    CONCEPT_QUESTION_PATTERNS,
    _target_date_range,
    resolve_metric_concept,
)
from agent.semantic_intent_v1 import SemanticIntent, semantic_intent_digest
from agent.stage1_v1_funding_semantics import (
    funding_categories,
    funding_comparison_demand_surfaces,
    recover_funding_comparison_fields,
)
from agent.stage1_v1_narrative_investment import (
    IncomparableInvestmentCashflowBackend,
    QuestionGroundedNarrativeInvestmentRegrounder,
    UnsupportedInvestmentOperatorBackend,
)
from agent.stage1_v1_resolver import (
    BackendResolutionResult,
    ClarificationAuthority,
    ClarificationOption,
    ClarificationSlot,
    ResolutionAuthority,
    normalize_backend_resolution_result,
)


class Stage1ResolutionBackendLike(Protocol):
    def resolve(
            self,
            *,
            question_id: str,
            question: str,
            source_intent: SemanticIntent,
            ) -> "ResolutionAuthority | Mapping[str, Any] | BackendResolutionResult | None": ...


class CanonicalQuestionCompanyRegrounder:
    """Recover one omitted issuer from the exact original question.

    HCX sometimes returns a useful financial/narrative item but leaves its
    company list empty.  This adapter only fills that absence when the
    canonical registry finds exactly one issuer in the question.  It never
    expands a model surface, chooses among candidates, or reads a question
    identifier/fixture.  Other semantic shapes remain untouched.
    """

    def __init__(
            self, canonical: Any, *, corpus_cutoff: str,
            preflight: Any | None = None,
            ) -> None:
        from .planner_preflight import CanonicalSelectorRolePreflight

        self._preflight = (
            preflight if preflight is not None
            else CanonicalSelectorRolePreflight(
                canonical, corpus_cutoff=corpus_cutoff)
        )
        if not callable(getattr(
                self._preflight, "unique_question_company_surface", None)):
            raise TypeError("company regrounder preflight 계약이 잘못되었습니다")

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str)
                or len(intent.answer_items) != 1
                or intent.premises
                or intent.answer_groups):
            return intent
        item = intent.answer_items[0]
        recoverable_mention = None
        if intent.unresolved_mentions:
            if len(intent.unresolved_mentions) != 1:
                return intent
            mention = intent.unresolved_mentions[0]
            if (mention.role_hint != "entity"
                    or mention.applies_to_item_ids != [item.item_id]
                    or mention.raw_text not in question):
                return intent
            recoverable_mention = mention
        if item.output.shape not in {"scalar", "narrative"}:
            return intent
        if (item.output.shape == "scalar" and item.target.kind != "metric"):
            return intent
        if (item.output.shape == "narrative"
                and item.target.kind not in {"topic", "event", "document"}):
            return intent
        by_id = {entity.entity_id: entity for entity in intent.entities}
        if any(
                by_id.get(ref) is not None
                and by_id[ref].kind_hint == "company"
                for ref in item.target.entity_refs):
            return intent
        surface = self._preflight.unique_question_company_surface(question)
        if surface is None:
            return intent
        if (recoverable_mention is not None
                and recoverable_mention.raw_text != surface):
            return intent

        payload = intent.model_dump(mode="python", warnings=False)
        entities = payload["entities"]
        entity_id = f"entity-{len(entities) + 1}"
        entities.append({
            "entity_id": entity_id,
            "kind_hint": "company",
            "surface": surface,
        })
        target = payload["answer_items"][0]["target"]
        target["entity_refs"] = [*target["entity_refs"], entity_id]
        if recoverable_mention is not None:
            payload["unresolved_mentions"] = []
        # The provider may flatten the issuer into a topic surface while also
        # omitting the company entity (for example ``삼성전자 사업 내용``).
        # Once the exact leading question span has been uniquely regrounded as
        # the issuer, remove only that same literal prefix from the topic.  The
        # remainder must be non-empty and already occur in the question; no
        # new retrieval wording is synthesized here.
        if item.output.shape == "narrative" and target["kind"] == "topic":
            topic = str(target["surface"])
            match = re.fullmatch(rf"{re.escape(surface)}\s+(.+)", topic)
            if match is not None:
                remainder = match.group(1).strip()
                if remainder and remainder in question:
                    target["surface"] = remainder
        return SemanticIntent.model_validate(payload, strict=True)


class DerivedComparisonJudgmentItemRegrounder:
    """Drop derived-difference/judgment items that name no entity or period.

    이슈 #64 — 「삼성전자의 2025년 연결 연간 매출액과 4분기 단독 매출액을
    각각 알려주고, 두 값의 차이를 분기 성장률이라고 불러도 되는지
    판단해줘」 decomposes into four literal metric items: the two concrete
    retrievals (each with their own entity + period) plus a bare 「두 값의
    차이」 item and a 「분기 성장률」 verdict item. The last two refer back to
    the first two and carry no entity_refs/period of their own, so
    financial_backend cannot resolve them independently — one unresolved
    sibling item used to close the *whole* request as
    ``unsupported_request`` even though the two retrieve items alone are
    answerable (batch #1's 「하나라도 못 풀면 None」 rule).

    Dropping them loses no fact.  The runtime already answers the judgment
    itself: DerivationExecutor's period-length guard
    (``_period_over_period_order``, issue #64) refuses to compute a
    difference/percent_change across mismatched period lengths, and
    ``CanonicalToolBackend._supplement_period_length_notice`` attaches a
    public 「기간 길이가 달라 …로 표현하지 않았습니다」 notice whenever an
    annual value and the discrete-quarter value derived from it both reach
    the answer. So keeping only the two concrete items is sufficient — the
    judgment sentence comes from the kept items' own typed coordinates, not
    from a separately resolved item.

    HCX does not emit one stable shape for this judgment pair across calls.
    Observed variants (live diagnostics, issue #64 follow-up) both drop
    ``entity_refs`` inconsistently — sometimes empty, sometimes carrying the
    same company as the kept items — and one variant truncates the surface
    itself (``target.surface == "두 값의"``, the literal field stays intact
    at ``"두 값의 차이"``).  The reliable structural signal is not
    entity_refs; it is that the item asks for **no period of its own** and
    is shaped as a comparison/verdict (``operation == "compare"`` or
    ``output.shape in {"comparison", "verdict"}``) naming difference/growth/
    judgment vocabulary somewhere in its surface or field surfaces.
    """

    _JUDGMENT_SURFACE = re.compile(
        r"차이|증감률|성장률|변화율|증감\s*률|변동률|두\s*값|판단")
    _QUARTER_STANDALONE_SURFACE = re.compile(r"분기\s*(?:단독|만의|만을|만)")
    _ANNUAL_SURFACE = re.compile(r"연간|연\s*환산")

    @classmethod
    def _names_judgment_vocabulary(cls, item: Any) -> bool:
        return (
            cls._JUDGMENT_SURFACE.search(item.target.surface) is not None
            or any(cls._JUDGMENT_SURFACE.search(surface) is not None
                   for surface in item.output.field_surfaces)
        )

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (
                not isinstance(question, str)
                or len(intent.answer_items) < 3
                or intent.answer_groups
                or intent.premises
                or intent.unresolved_mentions
        ):
            return intent

        items = list(intent.answer_items)
        kept, dropped = [], []
        for item in items:
            # entity_refs·operation·output.shape 유무는 신호로 쓰지 않는다 —
            # 위 docstring대로 호출마다 달라 신뢰할 수 없다(같은 질문의
            # 「두 값의 차이」항목이 어떤 호출은 operation=compare·shape=
            # comparison, 어떤 호출은 operation=retrieve·shape=scalar로도
            # 온다 — 실호출 diag 로그 3가지 변형 확인). 유일하게 안정적인
            # 신호는 「이 항목 자신의 기간이 없다」+「표면 어딘가(target.
            # surface 또는 field_surfaces)에 차이·증감·성장률·판단 어휘가
            # 있다」이다.
            is_derived_judgment = (
                item.target.kind == "metric"
                and not item.scope.target_period_expressions
                and item.scope.as_of_expression is None
                and item.scope.document_group_expression is None
                and self._names_judgment_vocabulary(item)
            )
            (dropped if is_derived_judgment else kept).append(item)
        if not dropped or len(kept) < 2:
            return intent
        # Every surviving item must keep a real entity and period — otherwise
        # this would silently narrow a genuinely broader, still-unresolved
        # request rather than only remove the self-referential judgment items.
        if any(
                not item.target.entity_refs
                or not item.scope.target_period_expressions
                for item in kept
        ):
            return intent

        # 이슈 #64 — 연간 값과 그 값으로 만든 4분기 단독 값을 나란히 요청한
        # 경우, 4분기 단독 항목 하나만 남겨도 정보 손실이 없다. 그 항목의
        # 해석 경로는 연간 누적값을 스스로 조회해(FY-9M) discrete_from_
        # cumulative의 재료로 쓰고, 그 재료(연간값)도 독립 claim으로 함께
        # 나온다(app/tools/backend.py 실행 결과). 두 항목을 나란히 둔 병렬
        # 재무 조회는 compiler가 지원하는 구조 신호(연결/별도 scope 쌍)와
        # 달라 언제나 unsupported_request로 닫혔으므로, 이 축소가 그
        # 우회로다 — 새 claim을 지어내지 않고 이미 있는 실행 경로 하나로
        # 합칠 뿐이다.
        if len(kept) == 2:
            quarter_index = next((
                index for index, item in enumerate(kept)
                if self._QUARTER_STANDALONE_SURFACE.search(item.target.surface)
            ), None)
            if quarter_index is not None:
                annual_index = 1 - quarter_index
                quarter_item, annual_item = kept[quarter_index], kept[annual_index]
                quarter_scopes = quarter_item.scope.scope_qualifier_expressions
                annual_scopes = annual_item.scope.scope_qualifier_expressions
                # 이슈 #64 후속(실호출 diag) — 「2025년 연결 연간 매출액과
                # 4분기 단독 매출액」처럼 「연결」이 앞쪽 항목의 target.surface
                # 접두사로만 붙어 있으면(뒤 항목은 되풀이하지 않는 한국어
                # 생략), boundary가 그 범위를 연간 항목에만 옮기고 분기 항목
                # scope_qualifier_expressions는 빈 채로 남는다. 완전 일치만
                # 요구하면 실제로 같은 범위인데도 절대 합쳐지지 않는다 — 둘 중
                # 하나가 비어 있으면(둘 다 있는데 다른 경우만 거절) 호환으로
                # 본다.
                scopes_compatible = (
                    quarter_scopes == annual_scopes
                    or not quarter_scopes or not annual_scopes
                )
                if (
                        self._ANNUAL_SURFACE.search(annual_item.target.surface)
                        is not None
                        and self._QUARTER_STANDALONE_SURFACE.search(
                            annual_item.target.surface) is None
                        and quarter_item.target.entity_refs
                            == annual_item.target.entity_refs
                        and quarter_item.scope.target_period_expressions
                            == annual_item.scope.target_period_expressions
                        and scopes_compatible
                ):
                    if not quarter_scopes and annual_scopes:
                        # 두 값이 같은 범위(예: 연결)라는 정보를 새로 만들지
                        # 않고, 연간 항목에만 남았던 그 범위를 그대로
                        # 옮긴다 — 나중에 생략된 쪽이 다른 범위라고
                        # 추측하지 않는다.
                        quarter_item = quarter_item.model_copy(update={
                            "scope": quarter_item.scope.model_copy(update={
                                "scope_qualifier_expressions": list(annual_scopes),
                            }),
                        })
                    kept = [quarter_item]

        payload = intent.model_dump(mode="python", warnings=False)
        rows = []
        for index, item in enumerate(kept):
            row = item.model_dump(mode="python", warnings=False)
            row["item_id"] = f"item-{index + 1}"
            rows.append(row)
        referenced = {
            ref for row in rows for ref in row["target"]["entity_refs"]}
        payload.update({
            "entities": [
                entity for entity in payload["entities"]
                if entity["entity_id"] in referenced
            ],
            "answer_items": rows,
        })
        return SemanticIntent.model_validate(payload, strict=True)


class CrossCompanyMultipleEntityRegrounder:
    """Restore a dropped second company from closed cross-company arithmetic.

    이슈 #59 1단계 실호출 — 「삼성전자의 2025년 연결 매출액은 SK하이닉스의
    몇 배인가?」의 실제 wire는 ``entities`` 에 두 회사를 다 담고도
    ``answer_items[0].target.entity_refs`` 에는 질문에 먼저 나온 회사
    하나만 남기고, ``output.shape`` 는 ``comparison`` 이다(감사 로그
    ``stage1_wire_failures.jsonl``, 2026-09-04). ``concept_ratio`` 컴파일러
    경로(#59 1)는 두 회사를 참조하는 scalar 배수 item을 기대하므로, 남은
    엔터티 하나·질문에 리터럴로 있는 다른 회사 하나·명시적 selection
    없음·배수 신호(「몇 배」/「배수」)가 모두 맞을 때만 entity_refs를
    질문 등장 순서로 채우고 shape를 scalar로 되돌린다. 표면·값 자체는
    건드리지 않는다 — 회사를 못 찾거나 신호가 없으면 그대로 둔다.
    """

    _ARITHMETIC_CUE = re.compile(r"몇\s*배|배수|합계|합산")

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (len(intent.answer_items) != 1 or intent.answer_groups
                or intent.premises or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        if (item.target.kind != "metric"
                or item.operation not in {"retrieve", "compare"}
                or item.selection is not None
                or item.output.projection_mode != "named_fields"
                or len(item.output.field_surfaces) != 1):
            return intent
        cue_text = " ".join((*item.output.field_surfaces, question))
        if self._ARITHMETIC_CUE.search(cue_text) is None:
            return intent
        companies = [entity for entity in intent.entities
                     if entity.kind_hint == "company"]
        if len(companies) != 2:
            return intent
        refs = list(item.target.entity_refs)
        company_ids = {entity.entity_id for entity in companies}
        if len(refs) != 1 or refs[0] not in company_ids:
            return intent
        present = next(entity for entity in companies if entity.entity_id == refs[0])
        missing = next(entity for entity in companies if entity.entity_id != refs[0])
        if missing.surface not in question or present.surface not in question:
            return intent
        present_pos = question.find(present.surface)
        missing_pos = question.find(missing.surface)
        new_refs = (
            [present.entity_id, missing.entity_id]
            if present_pos < missing_pos
            else [missing.entity_id, present.entity_id]
        )
        payload = intent.model_dump(mode="python", warnings=False)
        for row in payload["answer_items"]:
            if row["item_id"] == item.item_id:
                row["target"]["entity_refs"] = new_refs
                if row["output"]["shape"] == "comparison":
                    row["output"]["shape"] = "scalar"
        return SemanticIntent.model_validate(payload, strict=True)


class NamedRatioOperandItemRegrounder:
    """Absorb a named ratio's own numerator/denominator restated as items.

    ``«부채비율을 계산하고, 계산이 안 되면 부채총계와 자본총계만 알려줘」``
    is one request, but a provider can decompose it into three literal
    metric items: the named ratio itself, plus its two operand concepts
    restated verbatim from the question's own fallback clause. Those two
    operand items never carry their own period — the fallback clause never
    repeats the year — so before ``concept_ratio``(이슈 #38) existed they
    could not resolve independently, and an unresolved sibling item closed
    the whole request as ``unsupported_request`` even though the ratio item
    alone was answerable.

    ``concept_ratio`` execution already emits both operand values (with
    citations) as its own ``.numerator``/``.denominator`` claims — the
    "계산 근거" line — so a literal duplicate item adds no new fact and is
    safe to drop. A genuinely unrelated additional item (different concept,
    different company, or one that specifies its own period) is left alone
    and still reaches its normal ``financial`` task.
    """

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (
                not isinstance(question, str)
                or len(intent.answer_items) < 2
                or intent.answer_groups
                or intent.premises
                or intent.unresolved_mentions
        ):
            return intent
        from agent.planning import resolve_metric_concept
        from agent.stage1_v1_financial_backend import NAMED_RATIO_CONCEPTS

        items = list(intent.answer_items)
        ratio_indexes = [
            index for index, item in enumerate(items)
            if item.target.kind == "metric"
            and item.output.shape == "scalar"
            and item.scope.target_period_expressions
            and normalize_surface_key(item.target.surface) in NAMED_RATIO_CONCEPTS
        ]
        # Exactly one ratio item avoids guessing which of two named ratios in
        # the same question owns which operand restatement.
        if len(ratio_indexes) != 1:
            return intent
        ratio_item = items[ratio_indexes[0]]
        numerator, denominator, _presentation = NAMED_RATIO_CONCEPTS[
            normalize_surface_key(ratio_item.target.surface)]
        operand_concepts = {numerator, denominator}

        kept = [ratio_item]
        absorbed = False
        for index, item in enumerate(items):
            if index == ratio_indexes[0]:
                continue
            is_own_operand = (
                item.target.kind == "metric"
                and item.output.shape == "scalar"
                and item.target.entity_refs == ratio_item.target.entity_refs
                and not item.scope.target_period_expressions
                and item.scope.as_of_expression is None
                and item.scope.document_group_expression is None
                and item.scope.scope_qualifier_expressions
                    == ratio_item.scope.scope_qualifier_expressions
                and resolve_metric_concept(item.target.surface)
                    in operand_concepts
            )
            if is_own_operand:
                absorbed = True
                continue
            kept.append(item)
        if not absorbed:
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        payload["answer_items"] = [
            item.model_dump(mode="python", warnings=False) for item in kept
        ]
        return SemanticIntent.model_validate(payload, strict=True)


class QuestionGroundedFinancialRegrounder:
    """Restore one closed scalar financial request from its literal question.

    This is intentionally narrower than a question parser.  It only repairs a
    single already-metric retrieval whose response shape cannot carry another
    demand.  The question must contain exactly one annual year, one financial
    concept (or an approved concept clarification surface), and one issuer
    identity.  A held company alias is retained as an entity so
    ``HeldCompanyClarificationBackend`` can ask the user; it is never chosen
    here.

    Keeping this between the generic company recovery and the financial
    backend lets the existing canonical resolver remain the sole authority for
    scope defaults, account statements, and fact availability.
    """

    _YEAR = re.compile(r"(?<![0-9])20[0-9]{2}\s*년(?:도)?")
    # ``2025년 12월 16일까지`` is an as-of date, not the annual period this
    # repair is scoped to.  Reusing it as ``2025년`` silently widens a dated
    # disclosure-visibility question into a full-year fact lookup.
    _DATED_YEAR = re.compile(
        r"(?<![0-9])20[0-9]{2}\s*년(?:도)?\s*(?:1[0-2]|0?[1-9])\s*월")
    # A quarter/half/9-month token `agent.planning._financial_period` already
    # parses (P9-016/P9-017b).  The final generic branch below used to
    # blindly recompute a bare annual year from the literal question even
    # when the source item already carried one of these — silently widening
    # an explicit interim request into a full fiscal year that may not even
    # be closed yet in the corpus.
    _INTERIM_PERIOD_TOKEN = re.compile(
        r"(?<![0-9])(?:19|20)[0-9]{2}\s*(?:년도|년|회계연도)?\s*"
        r"(?:제?[1-4]\s*(?:분기|Q)(?:\s*(?:누적|단일분기|단일|단독))?"
        r"|(?:상반기|반기|H1)\s*(?:누적)?"
        r"|9\s*개월\s*(?:누적)?)",
        flags=re.IGNORECASE)
    _INSTANT_ENDPOINT = re.compile(
        r"(?<![0-9])20[0-9]{2}\s*년(?:도)?\s*"
        r"(?:[1-4]\s*분기\s*)?말")
    _AMOUNT_DIRECTION = re.compile(
        r"얼마나\s*(?:증가|감소|늘|줄)|증가(?:하거나|했|한).*?감소|"
        r"감소(?:하거나|했|한).*?증가|증감(?:액|했|한)|차이")
    _PERCENT_DEMAND = re.compile(r"%|퍼센트|몇\s*퍼|증감률|변동률|변화율")
    # Broader than ``_AMOUNT_DIRECTION`` above (which the 보다/대비 endpoint
    # branch owns and which only recognises 증감액/차이 phrasing): a 비교
    # wording two-endpoint question may ask for the amount as 증감액, 변동액,
    # or 변화액, and this branch must not miss the 변동액/변화액 spellings.
    _CHANGE_AMOUNT = re.compile(
        r"증감\s*액|변동\s*액|변화\s*액|(?:절대\s*)?차이")
    _AS_FILED = re.compile(r"최초\s*제출\s*값")
    _RESTATED = re.compile(r"최신\s*재작성\s*값")
    _COMPARISON = re.compile(r"비교")
    _CUMULATIVE_ARITHMETIC = re.compile(
        r"누적.*(?:차이|차감|빼|제외)|(?:차이|차감|빼|제외).*누적")
    _STATEMENT_CONTAINER = re.compile(
        r"(?:연결|별도|개별)?\s*(?:포괄)?손익계산서|"
        r"(?:연결|별도|개별)?\s*재무상태표|"
        r"(?:연결|별도|개별)?\s*현금흐름표")
    _TOKEN = re.compile(r"[A-Za-z0-9가-힣&+]+")
    _PARTICLE_ENDINGS = frozenset("은는이가을를의에도와과만")

    def __init__(self, canonical: Any) -> None:
        if not callable(getattr(canonical, "resolve_company", None)):
            raise TypeError("financial regrounder에는 company resolver가 필요합니다")
        if not callable(getattr(canonical, "held_company_candidates", None)):
            raise TypeError("financial regrounder에는 held alias registry가 필요합니다")
        self.canonical = canonical

    @staticmethod
    def _forms(token: str) -> tuple[str, ...]:
        """Return literal token forms after a Korean case particle only."""

        values = [token]
        while (
                values[-1]
                and values[-1][-1]
                in QuestionGroundedFinancialRegrounder._PARTICLE_ENDINGS):
            values.append(values[-1][:-1])
        return tuple(value for value in values if len(value) >= 2)

    def _company_surface(self, question: str) -> str | None:
        """Return one literal issuer surface or fail closed on competitors.

        Exact aliases can span several lexical tokens (``LG U+``, ``LG CNS``,
        ``SM Entertainment``).  Scan those first: a short held group alias
        embedded in the same text must not override an approved exact alias.
        Only when no exact company span exists may a held alias request
        clarification, preserving genuine questions such as ``LG ...``.
        """

        tokens = [match.group(0) for match in self._TOKEN.finditer(question)]
        direct: dict[str, tuple[int, str]] = {}
        for start in range(len(tokens)):
            for end in range(start + 1, min(len(tokens), start + 4) + 1):
                prefix = tokens[start:end - 1]
                for tail in self._forms(tokens[end - 1]):
                    surface = " ".join((*prefix, tail))
                    rows = list(self.canonical.resolve_company(surface) or ())
                    if len(rows) == 1:
                        candidate = (
                            len(normalize_surface_key(surface)), surface)
                        previous = direct.get(rows[0].corp_code)
                        if previous is None or candidate[0] > previous[0]:
                            direct[rows[0].corp_code] = candidate
        if direct:
            if len(direct) != 1:
                return None
            return max(direct.values())[1]

        held: dict[str, str] = {}
        for token in tokens:
            for surface in self._forms(token):
                rows = list(self.canonical.held_company_candidates(surface) or ())
                if len({row.corp_code for row in rows}) >= 2:
                    held.setdefault(normalize_surface_key(surface), surface)
        return next(iter(held.values())) if len(held) == 1 else None

    @classmethod
    def _concept_surface(cls, question: str) -> str | None:
        """Return a single approved/account-map metric span in ``question``.

        The approved colloquial layer normally gets first claim because it can
        preserve a required clarification (``이익났어``, ``투자``).  A phrase
        embedded in a named statement container is not the requested metric,
        however: ``포괄손익계산서상 희석주당순이익`` asks for the latter.  In
        that closed case the longest literal account-map match takes priority.
        """

        def matches_in(text: str) -> list[tuple[int, str, Any]]:
            tokens = [match.group(0) for match in cls._TOKEN.finditer(text)]
            matches: list[tuple[int, str, Any]] = []
            for start in range(len(tokens)):
                for end in range(start + 1, min(len(tokens), start + 5) + 1):
                    prefix = tokens[start:end - 1]
                    for tail in cls._forms(tokens[end - 1]):
                        surface = " ".join((*prefix, tail))
                        concept = resolve_metric_concept(surface)
                        if concept is not None:
                            matches.append((
                                len(normalize_surface_key(surface)),
                                surface, concept))
            return matches

        # ``비유동자산 중 유형자산``처럼 큰 분류와 그 안의 요청 계정이
        # 함께 적힌 경우에는 ``중`` 뒤의 literal account가 실제 target이다.
        # suffix 안에서도 후보가 하나로 닫힐 때만 이 구조를 사용한다.
        drilldown_parts = re.split(r"\s+중(?:에서)?\s+", question)
        matches = matches_in(drilldown_parts[-1]) if len(drilldown_parts) > 1 else []
        if not matches:
            matches = matches_in(question)
        colloquial = resolve_from_question(question, CONCEPT_QUESTION_PATTERNS)
        colloquial_in_statement = False
        if isinstance(colloquial.surface, str) and colloquial.surface in question:
            colloquial_in_statement = any(
                colloquial.surface in match.group(0)
                for match in cls._STATEMENT_CONTAINER.finditer(question))
        if (not colloquial_in_statement
                and colloquial.status in {
                    "resolved", "ambiguous", "route_conflict"}
                and isinstance(colloquial.surface, str)
                and colloquial.surface in question):
            return colloquial.surface
        # A broad account can be a literal prefix of the requested account
        # (``유형자산`` inside ``유형자산 취득 현금유출액``).  The longest
        # literal match wins only when it is concept-unique; equal-length
        # competing concepts still fail closed.
        max_length = max((length for length, _surface, _concept in matches), default=0)
        longest = [row for row in matches if row[0] == max_length]
        concepts = {concept for _length, _surface, concept in longest}
        if len(concepts) != 1:
            return None
        _length, surface, _concept = longest[0]
        return surface

    def _compatible_single_metric_item(
            self, question: str, intent: SemanticIntent, *, company: str,
            concept_surface: str,
            ) -> bool:
        """Whether one provider item can be safely narrowed to the closed axis."""

        if (len(intent.answer_items) != 1 or intent.answer_groups
                or intent.premises or intent.unresolved_mentions):
            return False
        item = intent.answer_items[0]
        if (item.target.kind != "metric" or item.operation not in {
                "retrieve", "compare"} or item.selection is not None
                or item.output.projection_mode not in {
                    "named_fields", "whole_target"}
                or item.output.shape not in {
                    "scalar", "comparison", "record"}):
            return False
        by_id = {entity.entity_id: entity for entity in intent.entities}
        selected = list(self.canonical.resolve_company(company) or ())
        held_alias = not selected and self._held_alias(company)
        if len(selected) != 1 and not held_alias:
            return False
        for ref in item.target.entity_refs:
            entity = by_id.get(ref)
            if entity is None or entity.kind_hint != "company":
                return False
            if held_alias:
                # No issuer is chosen yet.  Admit the reference only when it
                # is the same unresolved alias, so clarification still asks.
                if (normalize_surface_key(entity.surface)
                        != normalize_surface_key(company)):
                    return False
                continue
            rows = list(self.canonical.resolve_company(entity.surface) or ())
            if len(rows) != 1 or rows[0].corp_code != selected[0].corp_code:
                return False
        source_concept = resolve_metric_concept(item.target.surface)
        requested_concept = resolve_metric_concept(concept_surface)
        if (source_concept is not None and requested_concept is not None
                and source_concept != requested_concept):
            return False
        literal_surfaces = [
            *item.target.qualifier_surfaces,
            *item.scope.target_period_expressions,
            *item.scope.scope_qualifier_expressions,
            *item.output.field_surfaces,
        ]
        return all(surface in question for surface in literal_surfaces)

    def _compatible_discrete_pair(
            self, question: str, intent: SemanticIntent, *, company: str,
            concept_surface: str,
            ) -> bool:
        """Whether HCX split one explicit cumulative subtraction into 2 items.

        The pair is collapsible only when both items name the same issuer,
        metric, year and scope, and their sole distinct qualifiers are the two
        cumulative endpoints written in the question. This is a provider-shape
        repair; the arithmetic itself remains the typed financial backend's
        ``discrete_from_cumulative`` authority.
        """

        if (len(intent.answer_items) != 2 or intent.answer_groups
                or intent.premises or intent.unresolved_mentions):
            return False
        selected = list(self.canonical.resolve_company(company) or ())
        requested_concept = resolve_metric_concept(concept_surface)
        if len(selected) != 1 or requested_concept is None:
            return False
        by_id = {entity.entity_id: entity for entity in intent.entities}
        qualifiers: list[str] = []
        axes: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        target_keys: set[str] = set()
        for item in intent.answer_items:
            if (item.target.kind != "metric"
                    or item.operation not in {"retrieve", "compare"}
                    or item.selection is not None
                    or item.output.projection_mode != "named_fields"
                    or item.output.shape not in {"scalar", "comparison"}
                    or len(item.target.qualifier_surfaces) != 1
                    or len(item.output.field_surfaces) != 1):
                return False
            for ref in item.target.entity_refs:
                entity = by_id.get(ref)
                if entity is None or entity.kind_hint != "company":
                    return False
                rows = list(self.canonical.resolve_company(entity.surface) or ())
                if len(rows) != 1 or rows[0].corp_code != selected[0].corp_code:
                    return False
            source_concept = resolve_metric_concept(item.target.surface)
            if source_concept is not None and source_concept != requested_concept:
                return False
            if (source_concept is None
                    and normalize_surface_key(concept_surface)
                    not in normalize_surface_key(item.target.surface)):
                return False
            qualifier = item.target.qualifier_surfaces[0]
            literal_surfaces = [
                qualifier, *item.scope.target_period_expressions,
                *item.scope.scope_qualifier_expressions,
                *item.output.field_surfaces,
            ]
            if not all(surface in question for surface in literal_surfaces):
                return False
            qualifiers.append(qualifier)
            axes.append((
                tuple(item.scope.target_period_expressions),
                tuple(item.scope.scope_qualifier_expressions)))
            target_keys.add(normalize_surface_key(item.target.surface))
        if len(set(qualifiers)) != 2 or len(set(axes)) != 1 or len(target_keys) != 1:
            return False
        compact = [normalize_surface_key(value) for value in qualifiers]
        return (
            sum(bool(re.search(r"9개월.*누적", value)) for value in compact) == 1
            and sum(bool(re.search(r"(?:상반기|6개월).*누적", value))
                    for value in compact) == 1)

    def _compatible_endpoint_pair(
            self, question: str, intent: SemanticIntent, *, company: str,
            concept_surface: str,
            ) -> tuple[str, str] | None:
        """Whether HCX split one explicit two-instant-endpoint comparison.

        Same closed structural shape as `_compatible_discrete_pair` (one
        issuer/concept/scope, two items, one literal qualifier each) — issue
        #76: "SK하이닉스의 2024년 말과 2025년 말 연결 이익잉여금을 비교해
        증감액과 증감률을 계산해줘" comes back from HCX as two independent
        retrieve items (one qualifier surface each: "2024년 말"/"2025년 말"),
        never as one item carrying both.  This is distinct from
        `_compatible_discrete_pair`: the two qualifiers must themselves be
        two literal ``_INSTANT_ENDPOINT`` ("N년 말") spans, not the 9개월/
        상반기 cumulative endpoints that method already owns — a genuine
        cumulative pair never matches here, and vice versa.  Returns the two
        endpoints in question order (not answer-item order — an earlier
        endpoint may be the second item), or ``None``.
        """

        if (len(intent.answer_items) != 2 or intent.answer_groups
                or intent.premises or intent.unresolved_mentions):
            return None
        selected = list(self.canonical.resolve_company(company) or ())
        requested_concept = resolve_metric_concept(concept_surface)
        if len(selected) != 1 or requested_concept is None:
            return None
        endpoints = [
            match.group(0).strip()
            for match in self._INSTANT_ENDPOINT.finditer(question)]
        if len(set(endpoints)) != 2:
            return None
        by_id = {entity.entity_id: entity for entity in intent.entities}
        qualifiers: list[str] = []
        axes: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        target_keys: set[str] = set()
        for item in intent.answer_items:
            if (item.target.kind != "metric"
                    or item.operation not in {"retrieve", "compare"}
                    or item.selection is not None
                    or item.output.projection_mode != "named_fields"
                    or item.output.shape not in {"scalar", "comparison"}
                    or len(item.target.qualifier_surfaces) != 1
                    or len(item.output.field_surfaces) != 1):
                return None
            for ref in item.target.entity_refs:
                entity = by_id.get(ref)
                if entity is None or entity.kind_hint != "company":
                    return None
                rows = list(self.canonical.resolve_company(entity.surface) or ())
                if len(rows) != 1 or rows[0].corp_code != selected[0].corp_code:
                    return None
            source_concept = resolve_metric_concept(item.target.surface)
            if source_concept is not None and source_concept != requested_concept:
                return None
            if (source_concept is None
                    and normalize_surface_key(concept_surface)
                    not in normalize_surface_key(item.target.surface)):
                return None
            qualifier = item.target.qualifier_surfaces[0].strip()
            if qualifier not in endpoints:
                return None
            literal_surfaces = [
                qualifier, *item.scope.target_period_expressions,
                *item.scope.scope_qualifier_expressions,
                *item.output.field_surfaces,
            ]
            if not all(surface in question for surface in literal_surfaces):
                return None
            qualifiers.append(qualifier)
            axes.append((
                tuple(item.scope.target_period_expressions),
                tuple(item.scope.scope_qualifier_expressions)))
            target_keys.add(normalize_surface_key(item.target.surface))
        if len(set(qualifiers)) != 2 or len(set(axes)) != 1 or len(target_keys) != 1:
            return None
        return endpoints[0], endpoints[1]

    def _held_alias(self, surface: str) -> bool:
        """Whether ``surface`` is a registry-held group alias, not one issuer.

        ``resolve_company`` returns **no** row for a group alias such as
        ``현대`` by design; ``held_company_candidates`` is the companion
        authority and lists the issuers it could mean.  Such an alias must
        still reach the plan as a literal entity, because
        ``HeldCompanyClarificationBackend`` is what asks which issuer is
        meant.  Treating "does not resolve to exactly one company" as a
        reason to give up turns that question into a refusal.
        """

        if self.canonical.resolve_company(surface):
            return False
        rows = self.canonical.held_company_candidates(surface) or ()
        return len({row.corp_code for row in rows}) >= 2

    def _company_id(
            self, payload: dict[str, Any], *, company_surface: str,
            ) -> str | None:
        selected = list(self.canonical.resolve_company(company_surface) or ())
        held_alias = not selected and self._held_alias(company_surface)
        if len(selected) != 1 and not held_alias:
            return None
        matches = []
        for row in payload["entities"]:
            if row.get("kind_hint") != "company":
                continue
            if held_alias:
                # The alias names no issuer yet, so bind it literally.
                if (normalize_surface_key(row["surface"])
                        == normalize_surface_key(company_surface)):
                    matches.append(row["entity_id"])
                continue
            resolved = list(self.canonical.resolve_company(row["surface"]) or ())
            if len(resolved) == 1 and resolved[0].corp_code == selected[0].corp_code:
                matches.append(row["entity_id"])
        if len(matches) > 1:
            return None
        if matches:
            return matches[0]
        entity_id = f"entity-{len(payload['entities']) + 1}"
        payload["entities"].append({
            "entity_id": entity_id, "kind_hint": "company",
            "surface": company_surface,
        })
        return entity_id

    @staticmethod
    def _scope_surfaces(question: str) -> list[str]:
        scopes = []
        if "연결" in question:
            scopes.append("연결")
        if "별도" in question or "개별" in question:
            scopes.append("별도")
        return scopes

    def _replace_closed_item(
            self, intent: SemanticIntent, *, company_surface: str,
            concept_surface: str, operation: str, periods: list[str],
            qualifiers: list[str], output_shape: str,
            field_surfaces: list[str], question: str,
            ) -> SemanticIntent:
        payload = intent.model_dump(mode="python", warnings=False)
        company_id = self._company_id(
            payload, company_surface=company_surface)
        if company_id is None:
            return intent
        row = payload["answer_items"][0]
        row.update({"item_id": "item-1", "operation": operation,
                    "selection": None})
        row["target"].update({
            "surface": concept_surface, "entity_refs": [company_id],
            "qualifier_surfaces": qualifiers,
        })
        row["scope"].update({
            "target_period_expressions": periods,
            "scope_qualifier_expressions": self._scope_surfaces(question),
        })
        row["output"].update({
            "shape": output_shape, "projection_mode": "named_fields",
            "field_surfaces": field_surfaces,
        })
        payload["answer_items"] = [row]
        return SemanticIntent.model_validate(payload, strict=True)

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if not isinstance(question, str):
            return intent
        concept_surface = self._concept_surface(question)
        company = self._company_surface(question)
        if concept_surface is None or company is None:
            return intent
        discrete_pair = self._compatible_discrete_pair(
            question, intent, company=company,
            concept_surface=concept_surface)
        single_item = self._compatible_single_metric_item(
            question, intent, company=company,
            concept_surface=concept_surface)
        endpoint_pair = self._compatible_endpoint_pair(
            question, intent, company=company, concept_surface=concept_surface)
        if not (single_item or discrete_pair or endpoint_pair is not None):
            return intent
        if len(self._scope_surfaces(question)) > 1:
            return intent

        from agent.stage1_v1_financial_backend import (
            _question_grounded_single_quarter_expression,
        )

        quarter = _question_grounded_single_quarter_expression(question)
        if (quarter is not None
                and self._CUMULATIVE_ARITHMETIC.search(question) is not None
                and self._PERCENT_DEMAND.search(question) is None):
            literal_years = list(dict.fromkeys(
                match.group(0).strip() for match in self._YEAR.finditer(question)))
            periods = (
                [quarter] if quarter in question else
                literal_years if len(literal_years) == 1 else [])
            if not periods:
                return intent
            return self._replace_closed_item(
                intent, company_surface=company,
                concept_surface=concept_surface, operation="retrieve",
                periods=periods, qualifiers=[], output_shape="scalar",
                field_surfaces=[concept_surface], question=question)

        # RPC-001/015 — "N년 말과 M년 말 ... 비교해 증감액과 증감률을
        # 계산해줘" comes back as the same two-item shape `discrete_pair`
        # above matches structurally, but its two qualifiers are literal
        # annual endpoints, not the 9개월/상반기 cumulative pair that method
        # owns. Check it before the blanket two-item bailout below so this
        # closed comparison is not silently swallowed as an unhandled
        # cumulative split. (``endpoint_pair`` is already computed above so
        # it can also gate the top-level single_item/discrete_pair bailout.)
        if endpoint_pair is not None and self._COMPARISON.search(question):
            amount_change = self._CHANGE_AMOUNT.search(question)
            rate_change = self._PERCENT_DEMAND.search(question)
            change_fields = sorted(
                (match for match in (amount_change, rate_change)
                 if match is not None),
                key=lambda match: match.start())
            if change_fields:
                return self._replace_closed_item(
                    intent, company_surface=company,
                    concept_surface=concept_surface, operation="compare",
                    periods=list(endpoint_pair), qualifiers=[],
                    output_shape="comparison",
                    field_surfaces=[
                        match.group(0) for match in change_fields],
                    question=question)

        # A two-item provider split is admitted only by the exact cumulative
        # subtraction branch above. It must not fall through into unrelated
        # instant/view rewrites.
        if discrete_pair:
            return intent

        endpoint_matches = list(self._INSTANT_ENDPOINT.finditer(question))
        relations = list(re.finditer(r"보다|대비", question))
        endpoints: list[str] = []
        if len(endpoint_matches) == 2 and len(relations) == 1:
            bases = [match for match in endpoint_matches
                     if match.end() <= relations[0].start()]
            if bases:
                base = max(bases, key=lambda match: match.end())
                compared = next(
                    match for match in endpoint_matches if match is not base)
                endpoints = [compared.group(0).strip(), base.group(0).strip()]
        amount_direction = self._AMOUNT_DIRECTION.search(question)
        if (len(endpoints) == 2 and amount_direction is not None
                and self._PERCENT_DEMAND.search(question) is None
                and len(set(endpoints)) == 2):
            return self._replace_closed_item(
                intent, company_surface=company,
                concept_surface=concept_surface, operation="compare",
                periods=endpoints, qualifiers=[], output_shape="comparison",
                field_surfaces=[amount_direction.group(0)], question=question)

        # "N년 말과 M년 말을 비교해 증감액과 증감률을 계산해줘" wording never
        # uses 보다/대비, and it may ask for the amount, the rate, or both —
        # the 보다/대비 branch above hard-excludes any percent demand because
        # it only ever bound one field.  This closed 비교 topology is the same
        # two-instant-endpoint relation with 1-2 explicit change fields, and
        # the answer's earlier-period-as-base ordering is a downstream
        # concern (`_period_over_period_order`), so the two literal endpoints
        # are passed through in question order regardless of which one reads
        # first — a reversed-order restatement of the same question must
        # still close here.
        if len(endpoint_matches) == 2 and self._COMPARISON.search(question):
            amount_change = self._CHANGE_AMOUNT.search(question)
            rate_change = self._PERCENT_DEMAND.search(question)
            change_fields = sorted(
                (match for match in (amount_change, rate_change)
                 if match is not None),
                key=lambda match: match.start())
            distinct_periods = {
                match.group(0).strip() for match in endpoint_matches}
            if change_fields and len(distinct_periods) == 2:
                return self._replace_closed_item(
                    intent, company_surface=company,
                    concept_surface=concept_surface, operation="compare",
                    periods=[
                        match.group(0).strip() for match in endpoint_matches],
                    qualifiers=[], output_shape="comparison",
                    field_surfaces=[
                        match.group(0) for match in change_fields],
                    question=question)

        as_filed = self._AS_FILED.search(question)
        restated = self._RESTATED.search(question)
        comparison = self._COMPARISON.search(question)
        year_ends = list(dict.fromkeys(
            match.group(0).strip() for match in re.finditer(
                r"(?<![0-9])20[0-9]{2}\s*년(?:도)?\s*말", question)))
        if (as_filed is not None and restated is not None
                and comparison is not None and len(year_ends) == 1
                and re.search(r"전년(?:도)?|전기", question) is None
                and self._PERCENT_DEMAND.search(question) is None):
            view_roles = sorted(
                [as_filed.group(0), restated.group(0)], key=question.index)
            return self._replace_closed_item(
                intent, company_surface=company,
                concept_surface=concept_surface, operation="compare",
                periods=year_ends, qualifiers=view_roles,
                output_shape="comparison",
                field_surfaces=[
                    as_filed.group(0), restated.group(0), comparison.group(0)],
                question=question)

        item = intent.answer_items[0]
        if (
                item.target.kind != "metric"
                or item.operation != "retrieve"
                or item.selection is not None
                or item.output.shape != "scalar"
                or item.output.projection_mode != "named_fields"
                or len(item.output.field_surfaces) != 1
        ):
            return intent
        years = list(dict.fromkeys(
            match.group(0).strip() for match in self._YEAR.finditer(question)))
        if len(years) != 1 or self._DATED_YEAR.search(question) is not None:
            return intent
        scopes = self._scope_surfaces(question)
        if len(scopes) > 1:
            return intent

        # A scalar repair cannot choose one endpoint of a multi-period item.
        if len(item.scope.target_period_expressions) > 1:
            return intent

        # Prefer an interim period the item already carries, literally, over
        # the bare annual year just recomputed above (P9-016/P9-017b).  The
        # semantic boundary already promotes a target-qualifier period into
        # scope for the plain ``N년 [1-4]분기`` shape
        # (``agent.semantic_intent_v1_boundary``, ``PERIOD_FROM_TARGET_QUALIFIER``);
        # this only has to *not undo* that, and additionally covers the
        # 반기/상반기/9개월 wording the boundary promotion does not.
        periods = years
        existing_scope = [
            value for value in item.scope.target_period_expressions
            if value in question
            and self._INTERIM_PERIOD_TOKEN.fullmatch(value.strip())
        ]
        if len(item.scope.target_period_expressions) == 1 and existing_scope:
            periods = existing_scope
        else:
            existing_qualifier = [
                value for value in item.target.qualifier_surfaces
                if value in question
                and self._INTERIM_PERIOD_TOKEN.fullmatch(value.strip())
            ]
            if (not item.scope.target_period_expressions
                    and len(item.target.qualifier_surfaces) == 1
                    and existing_qualifier):
                periods = existing_qualifier

        # Issue #221 — the provider may retain only the year and move/drop
        # "상반기 누적" (CG-058). Recover the one literal interim coordinate,
        # including its cumulative/discrete suffix, rather than widening it
        # to annual. A second period, even one sharing the first year's text,
        # belongs to the comparison/fanout paths, not this scalar repair.
        interim_matches = list(self._INTERIM_PERIOD_TOKEN.finditer(question))
        if interim_matches:
            if len(interim_matches) != 1:
                return intent
            interim = interim_matches[0]
            remainder = question[:interim.start()] + question[interim.end():]
            if re.search(
                    r"[1-4]\s*(?:분기|Q)|Q\s*[1-4]|상반기|하반기|반기|"
                    r"H[12]|9\s*개월|연간|전년|전기|작년|지난해",
                    remainder, flags=re.IGNORECASE):
                return intent
            periods = [interim.group(0).strip()]

        company_surface = company
        payload = intent.model_dump(mode="python", warnings=False)
        entities = payload["entities"]
        company_id = next((
            row["entity_id"] for row in entities
            if row.get("kind_hint") == "company"
            and row.get("surface") == company_surface
        ), None)
        if company_id is None:
            company_id = f"entity-{len(entities) + 1}"
            entities.append({
                "entity_id": company_id,
                "kind_hint": "company",
                "surface": company_surface,
            })
        row = payload["answer_items"][0]
        row["target"].update({
            "surface": concept_surface,
            "entity_refs": [company_id],
            "qualifier_surfaces": [],
        })
        row["scope"].update({
            "target_period_expressions": periods,
            "scope_qualifier_expressions": scopes,
        })
        row["output"].update({
            "shape": "scalar",
            "projection_mode": "named_fields",
            "field_surfaces": [concept_surface],
        })
        return SemanticIntent.model_validate(payload, strict=True)


class CanonicalCompanyAliasMergeRegrounder:
    """Merge question-grounded company aliases by canonical ``corp_code``.

    A provider can emit both the current name and a parenthesized alias as two
    company entities, or preserve them as one compound surface.  They are
    source aliases, not separate financial operands.  This adapter collapses
    only uniquely resolved, referenced company entities that share one
    canonical corporation.  A compound surface is normalized only when its
    outer and parenthetical aliases each resolve uniquely to that same
    corporation.  The source surfaces are retained in a small internal
    provenance sidecar consumed by the financial backend; no private field is
    added to ``SemanticIntent``.  Its cache identity uses the question and
    normalized company sequence so later regrounders may add selection or
    output structure without disconnecting the provenance.
    """

    _PARENTHETICAL_ALIAS = re.compile(
        r"^\s*(?P<outer>[^()（）]+?)\s*[\(（]\s*"
        r"(?P<inner>[^()（）]+?)\s*[\)）]\s*$")

    def __init__(self, canonical: Any, *, max_entries: int = 256) -> None:
        if not callable(getattr(canonical, "resolve_company", None)):
            raise TypeError("company alias regrounder canonical 계약이 잘못되었습니다")
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("company alias provenance cache 크기가 잘못되었습니다")
        self._canonical = canonical
        self._max_entries = max_entries
        self._provenance: OrderedDict[
            tuple[str, str], tuple[dict[str, Any], ...]
        ] = OrderedDict()
        self._lock = Lock()

    def _company_identity(self, intent: SemanticIntent) -> str:
        companies: list[str] = []
        for entity in intent.entities:
            if entity.kind_hint != "company":
                continue
            resolved = self._resolve_surface(entity.surface)
            if resolved is None:
                companies.append(f"unresolved:{entity.surface}")
            else:
                companies.append(resolved[1].corp_code)
        return "\x1f".join(companies)

    def _resolve_surface(
            self, surface: str,
            ) -> tuple[str, Any, tuple[str, ...]] | None:
        matches = self._canonical.resolve_company(surface)
        if len(matches) == 1:
            return surface, matches[0], (surface,)

        compound = self._PARENTHETICAL_ALIAS.fullmatch(surface)
        if compound is None:
            return None
        outer = compound.group("outer").strip()
        inner = compound.group("inner").strip()
        if not outer or not inner or outer == inner:
            return None
        outer_matches = self._canonical.resolve_company(outer)
        inner_matches = self._canonical.resolve_company(inner)
        if len(outer_matches) != 1 or len(inner_matches) != 1:
            return None
        outer_company = outer_matches[0]
        inner_company = inner_matches[0]
        if outer_company.corp_code != inner_company.corp_code:
            return None
        surfaces = tuple(dict.fromkeys((surface, outer, inner)))
        return outer, outer_company, surfaces

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if not isinstance(question, str):
            return intent
        referenced = {
            ref for item in intent.answer_items if item.target.kind == "metric"
            for ref in item.target.entity_refs
        }
        nonfinancial_refs = {
            ref for item in intent.answer_items if item.target.kind != "metric"
            for ref in item.target.entity_refs
        }
        groups: dict[str, list[tuple[Any, Any, str, tuple[str, ...]]]] = {}
        normalized_surface_by_id: dict[str, str] = {}
        for entity in intent.entities:
            if (entity.entity_id not in referenced
                    or entity.entity_id in nonfinancial_refs
                    or entity.kind_hint != "company"
                    or entity.surface not in question):
                continue
            resolved = self._resolve_surface(entity.surface)
            if resolved is None:
                continue
            normalized_surface, company, source_surfaces = resolved
            if normalized_surface != entity.surface:
                normalized_surface_by_id[entity.entity_id] = normalized_surface
            groups.setdefault(company.corp_code, []).append(
                (entity, company, normalized_surface, source_surfaces))
        duplicate_groups = [rows for rows in groups.values() if len(rows) > 1]
        if not duplicate_groups and not normalized_surface_by_id:
            return intent

        representative_by_id: dict[str, str] = {}
        provenance_surfaces: dict[str, list[str]] = {}
        provenance_companies: dict[str, Any] = {}
        for corp_code, rows in groups.items():
            for _entity, company, _normalized, source_surfaces in rows:
                if len(source_surfaces) < 2:
                    continue
                provenance_companies[corp_code] = company
                aliases = provenance_surfaces.setdefault(corp_code, [])
                for surface in source_surfaces:
                    if surface not in aliases:
                        aliases.append(surface)
        for rows in duplicate_groups:
            representative = rows[0][0]
            surfaces = list(dict.fromkeys(
                surface
                for row in rows
                for surface in row[3]
            ))
            if len(surfaces) < 2:
                continue
            for entity, _company, _normalized, _source_surfaces in rows:
                representative_by_id[entity.entity_id] = representative.entity_id
            company = rows[0][1]
            provenance_companies[company.corp_code] = company
            aliases = provenance_surfaces.setdefault(company.corp_code, [])
            for surface in surfaces:
                if surface not in aliases:
                    aliases.append(surface)
        if not representative_by_id and not normalized_surface_by_id:
            return intent

        payload = intent.model_dump(mode="python", warnings=False)
        kept = []
        old_to_new: dict[str, str] = {}
        for entity in payload["entities"]:
            old_id = entity["entity_id"]
            representative_id = representative_by_id.get(old_id, old_id)
            if old_id != representative_id:
                continue
            new_id = f"entity-{len(kept) + 1}"
            old_to_new[old_id] = new_id
            entity["entity_id"] = new_id
            if old_id in normalized_surface_by_id:
                entity["surface"] = normalized_surface_by_id[old_id]
            kept.append(entity)
        for old_id, representative_id in representative_by_id.items():
            old_to_new[old_id] = old_to_new[representative_id]
        payload["entities"] = kept
        for item in payload["answer_items"]:
            refs = []
            for old_id in item["target"]["entity_refs"]:
                new_id = old_to_new[old_id]
                if new_id not in refs:
                    refs.append(new_id)
            item["target"]["entity_refs"] = refs
        merged = SemanticIntent.model_validate(payload, strict=True)

        provenance = [
            {
                "corp_code": corp_code,
                "corp_name": provenance_companies[corp_code].corp_name,
                "surfaces": surfaces,
            }
            for corp_code, surfaces in provenance_surfaces.items()
            if len(surfaces) >= 2
        ]

        key = (question, self._company_identity(merged))
        with self._lock:
            self._provenance[key] = tuple(provenance)
            self._provenance.move_to_end(key)
            while len(self._provenance) > self._max_entries:
                self._provenance.popitem(last=False)
        return merged

    def provenance_for(
            self, question: str, intent: SemanticIntent,
            ) -> tuple[dict[str, Any], ...]:
        key = (question, self._company_identity(intent))
        with self._lock:
            value = self._provenance.get(key, ())
            if value:
                self._provenance.move_to_end(key)
            return tuple(dict(row) for row in value)


class ClosedQuestionSemanticRegrounder:
    """Repair a few closed, fully question-grounded composite structures.

    HCX can preserve every requested surface while flattening a coordinated
    request into one record, or leaving two answer items ungrouped.  That is a
    structural loss rather than a corpus-resolution decision.  This adapter
    repairs only grammars whose items, fields, premise and grouping can all be
    copied literally from the question.  It never reads a fixture, question
    identifier, receipt number or company allowlist.
    """

    _COMPARISON_REASON = re.compile(
        r"(?P<verdict>정정\s*후\s*계약금액과\s*해지금액은\s*같으며)"
        r"\s*,?\s*다르다면\s*(?P<reason>왜\s*다른가)")
    _TERMINATION_REASON = re.compile(r"(?P<surface>해지된\s*이유)")
    _EFFECTIVE_CONDITION = re.compile(
        r"(?P<surface>계약\s*효력\s*발생\s*조건)")
    _STATUS_AND_TERMINATION_AMOUNT = re.compile(
        r"(?:살아\s*있|유효|상태).{0,80}(?:해지\s*금액|해지금액)")
    _DATED_STATUS_ASSERTION = re.compile(
        r"(?P<raw>(?:살아\s*있(?:음|나|어)?|유효(?:함|하지)?|"
        r"끝난\s*거\s*맞아?))")
    _CORRECTION_LIFECYCLE = re.compile(
        r"(?P<surface>정정\s*전후\s*(?:흐름|이력))")
    _NAMED_CONTRACT_LIFECYCLE = re.compile(
        r"(?P<root>최초\s*(?:공시|체결(?:\s*내용)?))"
        r"(?P<bridge>.{0,40}?)"
        r"(?P<change>"
        r"(?:(?:어떤|무슨)\s*내용이\s*)?(?:전체\s*)?"
        r"(?:변경\s*(?:됐는지|되었는지|되었는가|된\s*내용|사항|내용|이력)"
        r"|바뀌(?:었는지|었는가)|바뀐\s*내용"
        r"|달라(?:졌는지|졌는가|진\s*내용)))")
    _MULTI_CONTRACT_CUE = re.compile(
        r"계약\s*들|계약별|각각|모든\s*계약|여러\s*계약")
    _TERMINATION_CUE = re.compile(r"해지|종료|취소|철회|깨진|파기")

    @staticmethod
    def _group(payload: dict[str, Any]) -> None:
        items = payload["answer_items"]
        if len(items) < 2 or payload.get("answer_groups"):
            return
        payload["answer_groups"] = [{
            "group_id": "group-1",
            "item_ids": [item["item_id"] for item in items],
        }]

    def _latest_document_and_status(
            self, payload: dict[str, Any], question: str) -> bool:
        items = payload.get("answer_items")
        entities = payload.get("entities")
        if (
                not isinstance(items, list) or len(items) != 2
                or not isinstance(entities, list) or len(entities) != 1
                or "최신 공시 내용" not in question
                or "최종 상태" not in question
                or "제공 코퍼스 기준" not in question
        ):
            return False
        first, second = items
        if (
                first.get("operation") != "retrieve"
                or second.get("operation") != "retrieve"
                or first.get("selection", {}).get("mode") != "latest"
        ):
            return False
        entity_id = entities[0]["entity_id"]
        entities[0]["kind_hint"] = "counterparty"
        first["target"].update({
            "kind": "document", "surface": "공시",
            "entity_refs": [entity_id], "qualifier_surfaces": [],
        })
        # Regrounding is allowed both at the capability gate and again inside
        # the resolver.  Preserve an already grounded cutoff on the second
        # pass instead of replacing it with ``None`` after periods were moved.
        as_of = first["scope"].get("as_of_expression") or next((
            value for value in first["scope"]["target_period_expressions"]
            if value in question
        ), None)
        first["scope"].update({
            "as_of_expression": as_of,
            "target_period_expressions": [],
        })
        first["output"].update({
            "shape": "narrative", "projection_mode": "named_fields",
            "field_surfaces": ["내용"],
        })
        second["target"].update({
            "kind": "event", "surface": "계약",
            "entity_refs": [entity_id], "qualifier_surfaces": [],
        })
        second["scope"].update({
            "as_of_expression": None,
            "target_period_expressions": [],
            "scope_qualifier_expressions": ["제공 코퍼스 기준"],
        })
        second["selection"] = None
        second["output"].update({
            "shape": "scalar", "projection_mode": "named_fields",
            "field_surfaces": ["최종 상태"],
        })
        self._group(payload)
        return True

    def _comparison_with_reason(
            self, payload: dict[str, Any], question: str) -> bool:
        match = self._COMPARISON_REASON.search(question)
        items = payload.get("answer_items")
        entities = payload.get("entities")
        # The claim being checked ("…은 같으며") is exactly what this rewrite
        # turns into an explicit verdict item, so one comparison premise bound
        # to this same item is redundant rather than a second demand.  Any
        # other premise, or one bound elsewhere, still closes the rewrite.
        premises = payload.get("premises") or []
        redundant_premise = (
            len(premises) == 1
            and isinstance(premises[0], dict)
            and premises[0].get("kind") == "comparison"
            and list(premises[0].get("applies_to_item_ids") or [])
            == [items[0].get("item_id")] if isinstance(items, list) and items
            else False
        )
        if (
                match is None or not isinstance(items, list) or len(items) != 1
                or not isinstance(entities, list) or len(entities) not in {1, 2}
                or (premises and not redundant_premise)
        ):
            return False
        source = items[0]
        raw_fields = list(source.get("output", {}).get("field_surfaces", []))
        fields = {re.sub(r"\s+", "", value) for value in raw_fields}
        # A provider may qualify the same role in the question's own words
        # (``계약의 정정 후 계약금액``).  Accept that only when the whole
        # surface is a literal question span ending in the role name, so the
        # role is recovered without inventing an alias table.
        for label in ("계약금액", "해지금액"):
            if label in fields:
                continue
            qualified = [
                value for value in raw_fields
                if isinstance(value, str) and value in question
                and re.sub(r"\s+", "", value).endswith(label)
            ]
            if len(qualified) == 1:
                fields.add(label)
        if not {"계약금액", "해지금액"}.issubset(fields):
            return False
        event_surface = source["target"]["surface"]
        if event_surface not in question or "계약" not in event_surface:
            return False
        event_entity = next((
            entity for entity in entities
            if entity.get("surface") in event_surface
            and entity.get("surface") != "계약"
        ), entities[0])
        event_entity.update({
            "entity_id": "entity-1", "kind_hint": "event",
            "surface": event_surface,
        })
        payload["entities"] = [event_entity]
        verdict = match.group("verdict")
        reason = match.group("reason")
        common_target = {
            "kind": "event", "surface": event_surface,
            "entity_refs": [event_entity["entity_id"]],
            "qualifier_surfaces": [],
        }
        common_scope = {
            "target_period_expressions": [], "as_of_expression": None,
            "document_group_expression": None,
            "scope_qualifier_expressions": [],
        }
        common_selection = None
        presentation = source["output"].get("presentation", "auto")
        payload["answer_items"] = [
            {
                "item_id": "item-1", "target": dict(common_target),
                "operation": "retrieve", "scope": dict(common_scope),
                "selection": common_selection,
                "output": {
                    "shape": "verdict", "projection_mode": "named_fields",
                    "field_surfaces": [verdict], "presentation": presentation,
                },
            },
            {
                "item_id": "item-2", "target": dict(common_target),
                "operation": "retrieve", "scope": dict(common_scope),
                "selection": common_selection,
                "output": {
                    "shape": "narrative", "projection_mode": "named_fields",
                    "field_surfaces": [reason], "presentation": presentation,
                },
            },
        ]
        payload["answer_groups"] = [{
            "group_id": "group-1", "item_ids": ["item-1", "item-2"],
        }]
        payload["premises"] = [{
            "premise_id": "premise-1", "kind": "comparison",
            "raw_text": verdict,
            "applies_to_item_ids": ["item-1", "item-2"],
        }]
        return True

    def _event_attribute_pair(
            self, payload: dict[str, Any], question: str) -> bool:
        reason = self._TERMINATION_REASON.search(question)
        condition = self._EFFECTIVE_CONDITION.search(question)
        items = payload.get("answer_items")
        if reason is None or condition is None or not isinstance(
                items, list) or len(items) != 2:
            return False
        if any(item.get("operation") != "retrieve" for item in items):
            return False
        refs = items[0]["target"].get("entity_refs", [])
        if not refs or any(item["target"].get("entity_refs", []) != refs
                           for item in items):
            return False
        surfaces = [reason.group("surface"), condition.group("surface")]
        for item, surface in zip(items, surfaces):
            item["target"].update({
                "kind": "event", "surface": "계약",
                "qualifier_surfaces": [],
            })
            item["selection"] = None
            item["output"].update({
                "shape": "scalar", "projection_mode": "named_fields",
                "field_surfaces": [surface],
            })
        self._group(payload)
        return True

    def _status_and_termination_amount(
            self, payload: dict[str, Any], question: str) -> bool:
        """Join a split status/termination-amount request into one event read.

        HCX occasionally emits the status question and its ``해지 금액도``
        follow-up as an event item plus a metric item.  They are not two
        independent retrievals: the amount is a field of the same status
        observation.  Join only when the exact entity axis is shared and the
        event item carries one explicit day.  No company, receipt, event key,
        or corpus value is introduced here; the reported-status backend still
        has to prove a unique termination report and its origin limitation.
        """

        items = payload.get("answer_items")
        if (
                self._STATUS_AND_TERMINATION_AMOUNT.search(question) is None
                or not isinstance(items, list) or len(items) != 2
                or payload.get("premises")
        ):
            return False
        event_item = next((
            item for item in items if item.get("target", {}).get("kind") == "event"
        ), None)
        amount_item = next((
            item for item in items if item.get("target", {}).get("kind") == "metric"
        ), None)
        if event_item is None or amount_item is None:
            return False
        event_target = event_item.get("target", {})
        amount_target = amount_item.get("target", {})
        date_axes = [
            *event_target.get("qualifier_surfaces", []),
            *event_item.get("scope", {}).get("target_period_expressions", []),
        ]
        if (
                event_item.get("operation") != "retrieve"
                or amount_item.get("operation") != "retrieve"
                or event_target.get("entity_refs") != amount_target.get("entity_refs")
                or len(date_axes) != 1
        ):
            return False
        day = date_axes[0]
        if day not in question:
            return False
        event_fields = event_item.get("output", {}).get("field_surfaces", [])
        amount_fields = amount_item.get("output", {}).get("field_surfaces", [])
        if (
                not any("상태" in field or "살아" in field or "유효" in field
                        for field in event_fields)
                or not any("해지" in field and "금액" in field
                           for field in amount_fields)
        ):
            return False
        event_item["scope"].update({
            "as_of_expression": day,
            "target_period_expressions": [],
            "document_group_expression": None,
            "scope_qualifier_expressions": [],
        })
        event_item["target"]["qualifier_surfaces"] = [day]
        event_item["output"].update({
            "shape": "record", "projection_mode": "named_fields",
            "field_surfaces": [*event_fields, *amount_fields],
        })
        payload["answer_items"] = [event_item]
        payload["answer_groups"] = []
        # A dated "is it still alive?" clause is a claim to verify, not just
        # an output label.  Preserve it as an explicit state premise so the
        # status task has to establish the predicate before it can answer the
        # associated termination amount.
        assertion = self._DATED_STATUS_ASSERTION.search(question)
        if assertion is not None:
            payload["premises"] = [{
                "premise_id": "premise-1", "kind": "state",
                "raw_text": assertion.group("raw"),
                "applies_to_item_ids": [event_item["item_id"]],
            }]
        return True

    def _correction_lifecycle_whole_target(
            self, payload: dict[str, Any], question: str) -> bool:
        """Bind a whole-event narrative to its explicit lifecycle output.

        A provider may keep the event phrase but emit the requested
        ``정정 전후 흐름`` as an unlabelled ``whole_target``.  The lifecycle
        compiler needs that literal output axis to choose correction-history
        lowering.  Copy only the exact question span, and only for a closed
        single-event correction/termination request; event identity remains a
        canonical resolver decision.
        """

        match = self._CORRECTION_LIFECYCLE.search(question)
        items = payload.get("answer_items")
        if (
                match is None
                or self._TERMINATION_CUE.search(question) is None
                or not isinstance(items, list) or len(items) != 1
                or payload.get("answer_groups") or payload.get("premises")
                or payload.get("unresolved_mentions")
        ):
            return False
        item = items[0]
        target = item.get("target", {})
        output = item.get("output", {})
        target_surface = target.get("surface")
        if (
                item.get("operation") != "retrieve"
                or target.get("kind") != "event"
                or not isinstance(target_surface, str)
                or "계약" not in target_surface
                or target_surface not in question
                or item.get("selection") is not None
                or output.get("shape") != "narrative"
                or output.get("projection_mode") != "whole_target"
                or output.get("field_surfaces") != []
        ):
            return False
        output.update({
            "projection_mode": "named_fields",
            "field_surfaces": [match.group("surface")],
        })
        return True

    def _named_contract_lifecycle_whole_target(
            self, payload: dict[str, Any], question: str) -> bool:
        """Restore two literal axes lost by document whole-target fallback.

        ``최초 공시 후 무엇이 변경됐나`` asks for one named contract's
        complete event lifecycle.  It is not an issuer document collection,
        and it is not correction-only: a later termination is also a lifecycle
        change.  The selected-event backend already owns this two-axis
        narrative, including typed partial provenance when the original filing
        predates the corpus.  Restore only the two exact question spans needed
        by that existing path; canonical event identity remains unresolved
        here.
        """

        match = self._NAMED_CONTRACT_LIFECYCLE.search(question)
        items = payload.get("answer_items")
        entities = payload.get("entities")
        if (
                match is None
                or re.search(r"후|이후|부터|과|와", match.group("bridge")) is None
                or self._MULTI_CONTRACT_CUE.search(question) is not None
                or not isinstance(items, list) or len(items) != 1
                or not isinstance(entities, list)
                or payload.get("answer_groups") or payload.get("premises")
                or payload.get("unresolved_mentions")
        ):
            return False
        item = items[0]
        target = item.get("target", {})
        scope = item.get("scope", {})
        output = item.get("output", {})
        company_ids = {
            entity.get("entity_id") for entity in entities
            if entity.get("kind_hint") == "company"
        }
        target_surface = target.get("surface")
        literal_target = (
            target_surface.strip("'\"‘’“” ")
            if isinstance(target_surface, str) else "")
        if (
                len(company_ids) != 1
                or target.get("entity_refs") != list(company_ids)
                or item.get("operation") != "retrieve"
                or target.get("kind") not in {"document", "event"}
                or "계약" not in literal_target or literal_target not in question
                or target.get("qualifier_surfaces")
                or item.get("selection") is not None
                or scope.get("target_period_expressions")
                or scope.get("as_of_expression") is not None
                or scope.get("document_group_expression") is not None
                or scope.get("scope_qualifier_expressions")
                or output.get("shape") != "narrative"
                or output.get("projection_mode") != "whole_target"
                or output.get("field_surfaces") != []
        ):
            return False
        output.update({
            "projection_mode": "named_fields",
            "field_surfaces": [match.group("root"), match.group("change")],
        })
        return True

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        payload = intent.model_dump(mode="python", warnings=False)
        changed = (
            self._latest_document_and_status(payload, question)
            or self._comparison_with_reason(payload, question)
            or self._event_attribute_pair(payload, question)
            or self._status_and_termination_amount(payload, question)
            or self._named_contract_lifecycle_whole_target(payload, question)
            or self._correction_lifecycle_whole_target(payload, question)
        )
        return (
            SemanticIntent.model_validate(payload, strict=True)
            if changed else intent
        )


class QuestionGroundedEventStatusRegrounder:
    """Recover closed event/status axes from literals in the question.

    Provider output often collapses ``issuer + counterparty + event`` into one
    event surface, or stores observation dates as answer fields.  This repair
    only re-arranges literals already present in the question.  Issuer identity
    must be unique in the canonical company registry; an external party must be
    a literal Latin token or a literal alias from the shared counterparty
    dictionary.  Event identity itself remains the document backend's job.

    A confirmation such as ``...끝난 거 맞아`` is not merely a status field.
    It is retained as a typed state premise so a later compiler cannot silently
    turn premise verification into an ordinary lookup.
    """

    _STATUS_CUE = re.compile(r"상태|유효|살아\s*있|끝난|해지")
    _MONEY_CUE = re.compile(r"얼마짜리|해지\s*금액|해지금액")
    _COMPARISON_FIELD = re.compile(
        r"살아\s*있[^?？,.]{0,30}끝난|상태(?:가)?\s*(?:어떻게\s*)?달라|상태\s*비교")
    _DATED_STATE_PREMISE = re.compile(
        r"(?P<raw>(?<![0-9])[0-9]{2,4}(?:년|[-./])[^?？]{0,80}?"
        r"(?:끝난\s*거\s*맞아|끝난\s*거야))")
    _CURRENT_STATE_PREMISE = re.compile(r"(?P<raw>아직\s*유효하지)")
    _PARTIAL_DATE = re.compile(
        r"^(?:[01]?[0-9]월\s*[0-3]?[0-9]일|[0-3]?[0-9]일|[0-3]?[0-9])$")

    def __init__(self, canonical: Any, *, company_preflight: Any) -> None:
        self.canonical = canonical
        self.company_preflight = company_preflight
        if not callable(getattr(
                company_preflight, "unique_question_company_surface", None)):
            raise TypeError("event status company preflight 계약이 잘못되었습니다")

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))

    def _counterparty_literal(
            self, question: str, intent: SemanticIntent,
            issuer_surface: str | None,
            ) -> str | None:
        """Return one literal external-party surface, or decline.

        The canonical event preflight remains the authority that proves the
        value against documents.  Here we only separate a literal token from a
        collapsed question surface.  The alias vocabulary is shared with that
        preflight, so no second project/company table is introduced.
        """

        from agent.event_preflight import _counterparty_aliases, _key

        surfaces = [
            intent.answer_items[0].target.surface,
            *(entity.surface for entity in intent.entities),
        ]
        candidates: list[tuple[str, tuple[str, ...]]] = []
        aliases = _counterparty_aliases()
        for original in surfaces:
            if original == issuer_surface:
                continue
            residual = original
            if issuer_surface:
                residual = residual.replace(issuer_surface, " ")
            normalized = _key(residual)
            alias_hits = [
                (alias, canonical_values)
                for alias, canonical_values in aliases.items()
                if len(alias) >= 2 and alias in normalized
            ]
            if alias_hits:
                value_sets = {values for _, values in alias_hits}
                if len(value_sets) == 1:
                    alias, values = max(alias_hits, key=lambda row: len(row[0]))
                    if alias in question:
                        candidates.append((alias, values))
            latin = re.findall(r"[A-Za-z][A-Za-z0-9.&'-]*", residual)
            if len(latin) == 1 and latin[0] in question:
                candidates.append((latin[0], (latin[0].casefold(),)))
        by_identity: dict[tuple[str, ...], str] = {}
        for surface, identity in candidates:
            current = by_identity.get(identity)
            if current is None or len(surface) > len(current):
                by_identity[identity] = surface
        return next(iter(by_identity.values())) if len(by_identity) == 1 else None

    @classmethod
    def _ordered_date_surfaces(
            cls, question: str, item: Any,
            ) -> list[str]:
        pools = [
            list(item.scope.target_period_expressions),
            list(item.target.qualifier_surfaces),
            list(item.output.field_surfaces),
        ]
        candidates = cls._dedupe([
            value for pool in pools for value in pool
            if value in question and (
                (
                    (parts := parse_date_surface(value)) is not None
                    and parts[1] is not None and parts[2] is not None
                )
                or cls._PARTIAL_DATE.fullmatch(value.strip()) is not None)
        ])
        candidates.sort(key=question.index)
        return candidates

    @staticmethod
    def _question_days(question: str) -> list[str]:
        values = [
            f"{year:04d}{month:02d}{day:02d}"
            for year, month, day in question_date_surfaces(question)
            if month is not None and day is not None
        ]
        return sorted(set(values))

    @classmethod
    def _premise_surface(cls, question: str) -> str | None:
        for pattern in (cls._DATED_STATE_PREMISE, cls._CURRENT_STATE_PREMISE):
            match = pattern.search(question)
            if match is not None:
                return match.group("raw")
        return None

    @classmethod
    def _status_field(cls, question: str, current: list[str]) -> str | None:
        usable = [
            surface for surface in current
            if surface in question and cls._STATUS_CUE.search(surface)
            and parse_date_surface(surface) is None
        ]
        if len(usable) == 1:
            return usable[0]
        match = cls._COMPARISON_FIELD.search(question)
        if match is not None:
            return match.group(0)
        for pattern in (r"끝난\s*거", r"유효", r"상태", r"살아\s*있"):
            match = re.search(pattern, question)
            if match is not None:
                return match.group(0)
        return None

    @classmethod
    def _money_field(cls, question: str, current: list[str]) -> str | None:
        usable = [
            surface for surface in current
            if surface in question and cls._MONEY_CUE.search(surface)
        ]
        if len(usable) == 1:
            return usable[0]
        match = cls._MONEY_CUE.search(question)
        return match.group(0) if match is not None else None

    @staticmethod
    def _entity(entity_id: str, kind: str, surface: str) -> dict[str, str]:
        return {"entity_id": entity_id, "kind_hint": kind, "surface": surface}

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (
                not isinstance(question, str)
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.unresolved_mentions
        ):
            return intent
        source = intent.answer_items[0]
        if (
                source.target.kind not in {"event", "document"}
                or source.selection is not None
                or not self._STATUS_CUE.search(question)
        ):
            return intent

        issuer = self.company_preflight.unique_question_company_surface(question)
        counterparty = self._counterparty_literal(question, intent, issuer)
        date_surfaces = self._ordered_date_surfaces(question, source)
        question_days = self._question_days(question)
        premise = self._premise_surface(question)
        fields = list(source.output.field_surfaces)
        payload = intent.model_dump(mode="python", warnings=False)
        item = payload["answer_items"][0]
        changed = False

        # Exact dated status + monetary field is the closed reported-termination
        # topology.  Both roles and the repeated observation date remain literal.
        status_field = self._status_field(question, fields)
        money_field = self._money_field(question, fields)
        if (
                issuer is not None and counterparty is not None
                and len(question_days) == 1 and money_field is not None
                and status_field is not None
        ):
            exact = next((
                value for value in date_surfaces
                if parse_date_surface(value) is not None
            ), source.scope.as_of_expression)
            if exact is None or exact not in question:
                return intent
            payload["entities"] = [
                self._entity("entity-1", "company", issuer),
                # The downstream reported-termination contract calls the
                # external legal person a company; event role is still proved
                # by canonical counterparty fields, never by this hint alone.
                self._entity("entity-2", "company", counterparty),
            ]
            item["target"].update({
                "kind": "event", "entity_refs": ["entity-1", "entity-2"],
                "qualifier_surfaces": [exact],
            })
            item["operation"] = "retrieve"
            item["scope"].update({
                "target_period_expressions": [], "as_of_expression": exact,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            })
            item["output"].update({
                "shape": "record", "projection_mode": "named_fields",
                "field_surfaces": [status_field, money_field],
            })
            changed = True

        # Two exact observation days.  With an external party the dates belong
        # on the event target; with only an issuer an existing non-day qualifier
        # remains the event seed and the observation days remain in scope.
        elif len(question_days) == 2 and len(date_surfaces) >= 2:
            if issuer is not None and counterparty is not None:
                payload["entities"] = [
                    self._entity("entity-1", "company", issuer),
                    self._entity("entity-2", "event", counterparty),
                ]
                item["target"].update({
                    "kind": "event", "entity_refs": ["entity-2"],
                    "qualifier_surfaces": date_surfaces[:2],
                })
                item["scope"]["target_period_expressions"] = []
                item["output"]["shape"] = "record"
            elif issuer is None and counterparty is not None:
                payload["entities"] = [
                    self._entity("entity-1", "event", counterparty),
                ]
                item["target"].update({
                    "kind": "event", "entity_refs": ["entity-1"],
                    "qualifier_surfaces": date_surfaces[:2],
                })
                item["scope"]["target_period_expressions"] = []
                item["output"]["shape"] = "record"
            elif issuer is not None:
                payload["entities"] = [
                    self._entity("entity-1", "company", issuer),
                ]
                item["target"]["entity_refs"] = ["entity-1"]
                item["scope"]["target_period_expressions"] = date_surfaces[:2]
                item["target"]["qualifier_surfaces"] = [
                    value for value in item["target"]["qualifier_surfaces"]
                    if parse_date_surface(value) is not None
                    and parse_date_surface(value)[1] is None
                ]
            else:
                return intent
            field = self._status_field(question, fields)
            if field is None:
                return intent
            item["operation"] = "retrieve"
            item["scope"].update({
                "as_of_expression": None, "document_group_expression": None,
                "scope_qualifier_expressions": [],
            })
            item["output"].update({
                "projection_mode": "named_fields", "field_surfaces": [field],
            })
            changed = True

        # A status predicate with no explicit observation date is evaluated at
        # corpus cutoff by the backend.  Keep only the literal issuer authority;
        # the external party remains available in the target surface for the
        # canonical selector proof.
        elif not question_days and issuer is not None and counterparty is not None:
            payload["entities"] = [
                self._entity("entity-1", "company", issuer),
                self._entity("entity-2", "company", counterparty),
            ]
            item["target"].update({
                "kind": "event", "entity_refs": ["entity-1", "entity-2"],
                "qualifier_surfaces": [],
            })
            item["operation"] = "retrieve"
            item["scope"].update({
                "target_period_expressions": [], "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            })
            field = self._status_field(question, fields)
            if field is None:
                return intent
            item["output"].update({
                "shape": "scalar", "projection_mode": "named_fields",
                "field_surfaces": [field],
            })
            changed = True

        if premise is not None and not payload["premises"]:
            payload["premises"] = [{
                "premise_id": "premise-1", "kind": "state",
                "raw_text": premise, "applies_to_item_ids": ["item-1"],
            }]
            changed = True
        return (
            SemanticIntent.model_validate(payload, strict=True)
            if changed else intent
        )


class MisplacedPeriodQualifierRegrounder:
    """Move a question-grounded date accidentally placed in target qualifiers.

    Dates are temporal coordinates in v1.  The provider occasionally preserves
    a colloquial two-digit year exactly but stores it under target qualifiers;
    this repair changes only the axis, never the surface or its meaning.
    """

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        payload = intent.model_dump(mode="python", warnings=False)
        changed = False
        for item in payload["answer_items"]:
            scope = item["scope"]
            target = item["target"]
            if scope["target_period_expressions"]:
                continue
            date_qualifiers = [
                value for value in target["qualifier_surfaces"]
                if value in question and parse_date_surface(value) is not None
            ]
            if len(date_qualifiers) != 1:
                continue
            value = date_qualifiers[0]
            as_of = scope["as_of_expression"]
            # The same exact day on both axes is a deliberate event-status
            # observation coordinate.  Moving it into target-period scope makes
            # it an occurrence filter and breaks reported-termination binding.
            if (as_of is not None
                    and parse_date_surface(as_of) == parse_date_surface(value)):
                continue
            # In a two-observation event status the complete first day and an
            # abbreviated second day intentionally share target qualifiers.
            # The question-level scanner has already proved the pair; moving
            # only the complete member would split one temporal axis in two.
            full_days = [
                parts for parts in question_date_surfaces(question)
                if parts[1] is not None and parts[2] is not None
            ]
            if len(set(full_days)) == 2 and len(
                    target["qualifier_surfaces"]) >= 2:
                continue
            scope["target_period_expressions"] = [value]
            target["qualifier_surfaces"] = [
                row for row in target["qualifier_surfaces"] if row != value
            ]
            changed = True
        return (
            SemanticIntent.model_validate(payload, strict=True)
            if changed else intent
        )


class FundingComparisonDemandRegrounder:
    """Restore only question-literal demands hidden by funding categories."""

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if len(intent.answer_items) != 1:
            return intent
        fields = recover_funding_comparison_fields(
            question, intent.answer_items[0])
        if fields is None:
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        payload["answer_items"][0]["output"]["field_surfaces"] = list(fields)
        return SemanticIntent.model_validate(payload, strict=True)


class FundingComparisonSplitItemRegrounder:
    """Join one narrowly proven split funding-comparison topology.

    The provider can put literal funding categories on a comparison item and
    its literal answer demands (for example type grouping, conditions, and
    correction history) on a second document item.  This is one collection,
    not two independent requests, only when both halves bind the same issuer
    and explicit multi-year scope.  The adapter deliberately cannot merge a
    general pair of document/event items.
    """

    _FULL_YEAR = re.compile(r"20[0-9]{2}\s*년")
    _TYPE_GROUPING = re.compile(r"유형\s*별")
    _ORGANIZE_OR_COMPARE = re.compile(r"(?:정리|비교)")
    _EXPLICIT_SELECTION = re.compile(
        r"(?:최신|가장\s*최근|최근|최초|첫|처음|"
        r"가장\s*이른|가장\s*(?:큰|작은|높은|낮은)|최대|최소)")

    @staticmethod
    def _item_surfaces(item: Any) -> tuple[str, ...]:
        return (
            item.target.surface,
            *item.target.qualifier_surfaces,
            *item.scope.target_period_expressions,
            *((item.scope.as_of_expression,)
              if item.scope.as_of_expression is not None else ()),
            *((item.scope.document_group_expression,)
              if item.scope.document_group_expression is not None else ()),
            *item.scope.scope_qualifier_expressions,
            *item.output.field_surfaces,
            *((item.selection.criterion_surface,)
              if item.selection is not None else ()),
        )

    @classmethod
    def _redundant_latest_selection(
            cls, question: str, selection: Any,
            ) -> bool:
        """Allow only a non-question-grounded ``latest`` hallucination.

        ``발행 결정`` itself may be a literal target phrase while its
        ``latest`` mode is not.  Removing it is safe only for the explicit
        multi-year type organization/comparison handled below; a literal
        latest/first/largest request stays an independent selector.
        """

        return (
            selection.mode == "latest"
            and selection.k is None
            and selection.criterion_surface in question
            and cls._EXPLICIT_SELECTION.search(question) is None
        )

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (
                not isinstance(question, str)
                or len(intent.answer_items) != 2
                or intent.answer_groups
                or intent.premises
                or intent.unresolved_mentions
        ):
            return intent
        items = list(intent.answer_items)
        if any(
                item.target.kind not in {"document", "event", "topic"}
                or item.operation not in {"retrieve", "compare"}
                or item.output.projection_mode != "named_fields"
                or item.output.shape not in {
                    "record", "record_list", "comparison", "narrative"}
                or not item.output.field_surfaces
                or any(surface not in question
                       for surface in self._item_surfaces(item))
                for item in items
        ):
            return intent

        # Exactly one comparison half and one retrieval half avoids absorbing
        # two independently answerable comparisons into an event collection.
        compare_indexes = [
            index for index, item in enumerate(items)
            if item.operation == "compare"
        ]
        if len(compare_indexes) != 1:
            return intent

        by_id = {entity.entity_id: entity for entity in intent.entities}
        if any(entity.surface not in question for entity in intent.entities):
            return intent
        company_refs = []
        for item in items:
            if len(item.target.entity_refs) != 1:
                return intent
            entity = by_id.get(item.target.entity_refs[0])
            if entity is None or entity.kind_hint != "company":
                return intent
            company_refs.append(entity.entity_id)
        if len(set(company_refs)) != 1:
            return intent

        scope_signatures = [(
            tuple(item.scope.target_period_expressions),
            item.scope.as_of_expression,
            item.scope.document_group_expression,
            tuple(item.scope.scope_qualifier_expressions),
        ) for item in items]
        if scope_signatures[0] != scope_signatures[1]:
            return intent
        periods = scope_signatures[0][0]
        if (
                len(periods) < 2
                or len(set(periods)) != len(periods)
                or any(self._FULL_YEAR.fullmatch(period.strip()) is None
                       for period in periods)
        ):
            return intent

        # Categories must occur as literal provider/question surfaces, not be
        # inferred from a generic "funding" label.  Requiring them in the
        # items also prevents a non-funding two-item topology from borrowing
        # category words that happen to appear elsewhere in the question.
        item_category_surfaces = [
            surface for item in items for surface in self._item_surfaces(item)
        ]
        if len(funding_categories(*item_category_surfaces)) < 2:
            return intent
        if (self._TYPE_GROUPING.search(question) is None
                or self._ORGANIZE_OR_COMPARE.search(question) is None):
            return intent
        demands = funding_comparison_demand_surfaces(question)
        if not demands:
            return intent

        selections = [
            item.selection for item in items if item.selection is not None
        ]
        if selections and (
                len(selections) != 1
                or not self._redundant_latest_selection(question, selections[0])
        ):
            return intent

        payload = intent.model_dump(mode="python", warnings=False)
        merged = payload["answer_items"][compare_indexes[0]]
        merged["selection"] = None
        # Categories remain membership selectors in the literal target/question;
        # outputs are only the explicit answer demands, in question order.
        merged["output"]["field_surfaces"] = list(demands)
        payload["answer_items"] = [merged]
        return SemanticIntent.model_validate(payload, strict=True)


class CompatibleEventCollectionItemRegrounder:
    """Merge a lifecycle-status side item into the same event collection."""

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str)
                or len(intent.answer_items) < 2
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        items = list(intent.answer_items)
        first = items[0]
        signature = (
            first.target.kind, first.target.surface,
            tuple(first.target.entity_refs), tuple(first.target.qualifier_surfaces),
            first.operation,
        )
        if (first.target.kind not in {"event", "document", "topic"}
                or first.operation not in {"retrieve", "compare"}
                or any((
                    item.target.kind, item.target.surface,
                    tuple(item.target.entity_refs),
                    tuple(item.target.qualifier_surfaces), item.operation,
                ) != signature
                       or item.selection is not None
                       or item.output.projection_mode != "named_fields"
                       or item.output.shape not in {
                           "record", "record_list", "comparison"}
                       for item in items)):
            return intent
        # This adapter is intentionally narrower than a generic item merger:
        # later items may only add member-level revision/follow-up status.
        if not all(
                any(token in surface for token in ("정정", "후속", "상태", "여부"))
                for item in items[1:] for surface in item.output.field_surfaces):
            return intent

        sentinel = object()

        def _compatible(values: list[Any]) -> Any:
            nonempty = [value for value in values if value not in (None, [], ())]
            unique = {repr(value): value for value in nonempty}
            if not unique:
                return None
            return next(iter(unique.values())) if len(unique) == 1 else sentinel

        periods = _compatible([
            list(item.scope.target_period_expressions) for item in items])
        as_of = _compatible([item.scope.as_of_expression for item in items])
        document_group = _compatible([
            item.scope.document_group_expression for item in items])
        scope_qualifiers = _compatible([
            list(item.scope.scope_qualifier_expressions) for item in items])
        if any(value is sentinel for value in (
                periods, as_of, document_group, scope_qualifiers)):
            return intent
        fields = [
            surface for item in items for surface in item.output.field_surfaces]
        if (not fields or len(fields) != len(set(fields))
                or not all(surface in question for surface in fields)
                or first.target.surface not in question):
            return intent

        payload = intent.model_dump(mode="python", warnings=False)
        merged = payload["answer_items"][0]
        merged["scope"].update({
            "target_period_expressions": periods or [],
            "as_of_expression": as_of,
            "document_group_expression": document_group,
            "scope_qualifier_expressions": scope_qualifiers or [],
        })
        merged["output"]["field_surfaces"] = fields
        payload["answer_items"] = [merged]
        return SemanticIntent.model_validate(payload, strict=True)


class SplitNarrativeMatrixItemRegrounder:
    """Merge equivalent per-company report items into one typed matrix.

    HCX may emit one otherwise identical item per company for an explicit
    multi-company comparison.  The matrix authority expects one item whose
    company axis contains all requested issuers.  Merge only a closed latest-
    periodic shape: every item has the same document, scope, operation and
    literal topic fields, while each contributes exactly one distinct company.
    """

    _LATEST_PERIODIC = re.compile(
        r"(?:가장\s*)?최근\s*(?:정기보고서|사업보고서|반기보고서|분기보고서)")
    _TOPIC_CONTAINER = re.compile(r"주요\s*사업")

    @staticmethod
    def _topic_span(question: str, fields: list[str]) -> str | None:
        positions = [(question.find(surface), question.find(surface) + len(surface))
                     for surface in fields]
        if any(start < 0 for start, _end in positions):
            return None
        start = min(value[0] for value in positions)
        field_end = max(value[1] for value in positions)
        container = SplitNarrativeMatrixItemRegrounder._TOPIC_CONTAINER.search(
            question, field_end)
        if container is None:
            return None
        topic = question[start:container.end()].strip()
        if not topic or len(topic) > 80 or re.search(r"[,;!?？。]", topic):
            return None
        remainder = topic
        for surface in fields:
            remainder = remainder.replace(surface, "", 1)
        remainder = re.sub(r"주요\s*사업", "", remainder, count=1)
        if re.sub(r"(?:\s|[·,/・•ㆍ]|그리고|및|관련)", "", remainder):
            return None
        return topic

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str)
                or not 2 <= len(intent.answer_items) <= 4
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        items = list(intent.answer_items)
        first = items[0]
        latest = self._LATEST_PERIODIC.search(question)
        if latest is None:
            return intent

        # Some HCX samples attach the same question-explicit ``latest``
        # selector to every per-company row, while others encode it only in
        # the target phrase.  These are equivalent for this closed matrix.
        # Accept only an all-none form or one identical, literal latest form;
        # mixed/different selectors retain their original fail-closed shape.
        selections = [
            (item.selection.model_dump(mode="python", warnings=False)
             if item.selection is not None else None)
            for item in items
        ]
        identical_latest_selection = (
            all(selection is not None for selection in selections)
            and all(selection == selections[0] for selection in selections[1:])
            and selections[0]["mode"] == "latest"
            and selections[0]["criterion_surface"] in latest.group(0)
        )
        if not (all(selection is None for selection in selections)
                or identical_latest_selection):
            return intent
        if any(
                item.target.kind != "document"
                or item.operation != first.operation
                or item.operation not in {"retrieve", "compare"}
                or item.target.surface != first.target.surface
                or item.target.qualifier_surfaces
                != first.target.qualifier_surfaces
                or item.scope != first.scope
                or item.output.shape != first.output.shape
                or item.output.shape not in {"comparison", "narrative"}
                or item.output.projection_mode != "named_fields"
                or item.output.field_surfaces != first.output.field_surfaces
                or not item.output.field_surfaces
                or len(item.target.entity_refs) != 1
                for item in items):
            return intent

        if (first.target.surface not in latest.group(0)
                or first.target.surface not in question):
            return intent
        fields = list(first.output.field_surfaces)
        if len(fields) != len(set(fields)) or any(
                surface not in question for surface in fields):
            return intent
        topic = self._topic_span(question, fields)
        if topic is None:
            return intent

        by_id = {entity.entity_id: entity for entity in intent.entities}
        company_entities = []
        for item in items:
            entity = by_id.get(item.target.entity_refs[0])
            if (entity is None or entity.kind_hint != "company"
                    or entity.surface not in question):
                return intent
            company_entities.append(entity)
        if len({entity.entity_id for entity in company_entities}) != len(items):
            return intent
        company_entities.sort(key=lambda entity: question.find(entity.surface))

        payload = intent.model_dump(mode="python", warnings=False)
        merged = payload["answer_items"][0]
        merged["item_id"] = "item-1"
        merged["target"]["surface"] = latest.group(0)
        merged["target"]["entity_refs"] = [
            entity.entity_id for entity in company_entities]
        # ``가장 최근 정기보고서`` is now the canonical target surface.  A
        # second selector would double-encode the same request coordinate.
        merged["selection"] = None
        merged["output"]["field_surfaces"] = [topic]
        payload["answer_items"] = [merged]
        return SemanticIntent.model_validate(payload, strict=True)


class NaryFinancialSuperlativeShapeRegrounder:
    """이슈 #136 — 3개사 이상 「가장 큰/작은」 scalar wire 를 comparison 으로.

    CG-053(``fixtures/coverage_gap_v01/questions_v0.2.jsonl``) 실호출:
    「삼성전자·SK하이닉스·기아 중 2025년 연결 매출액이 가장 작은 기업은
    어디이고 얼마인가?」를 HCX 가 ``operation="retrieve"``,
    ``output.shape="scalar"`` 로 내(기대 ``compare``/``comparison``) N-ary
    재무 순위 컴파일러(``_validate_financial_comparison_intent``,
    ``agent/deterministic_plan_compiler_v1.py``)에 닿기 전 거절된다.
    「가장 큰」 판·「비교해」·「규모 순위」 대체 문구도 같다 — 방향과 무관한
    기존 wire 격차(#59 3 의 5~8개사 실호출은 통과했으니 회사 조합·문구에
    따라 갈린다).

    2개사 비교(#59 1, G-A-004: 「…큰 기업은 어디이며 차이는 얼마인가?」)는
    이미 ``operation="compare"``·``output.shape="comparison"`` 로 나와
    ``ClosedFinancialComparisonRegrounder``/자연 wire 가 그대로 받는다 —
    회사가 셋 이상일 때만 같은 두 축(operation·shape)이 남아 있다.
    다만 그 2개사 wire 가 **항상** 그대로 컴파일되는 것은 아니다: 같은
    질문의 어떤 표본은 두 축을 옳게 내면서 ``selection`` 만 ``none`` 으로
    비워 보내, 컴파일러의 2개사 비교 검증(``_validate_financial_comparison_
    intent``)이 「executable maximum/minimum selection이 필요합니다」로
    거절한다(R-P-003, 실호출로 직접 확인함). 그 축은 아래 형제
    ``TwoCompanySuperlativeSelectionRegrounder`` 가 되돌린다 — 이쪽은
    operation·shape 두 축만, 그쪽은 selection 한 축만 본다.

    옮기는 것은 두 축뿐이다 — 회사·기간·selection·답 필드 표면 자체는
    새로 고르지 않는다. 다만 컴파일러의 N-ary 검증(``_nary_ranking_field_
    indexes``)은 필드가 둘이면 그중 하나가 리터럴 "순위" 표면이어야
    유일하게 결속된다 — 질문에 없는 "순위" 말을 지어내는 대신, 이미 질문에
    있는 필드 표면 하나(첫 리터럴 표면)만 남긴다. 필드가 하나든 둘이든
    ``argmax``/``argmin`` 연산자는 항상 전체 순위를 계산하므로(PR #106,
    #135) 회사·값 모두 답에 남는다 — 필드 수를 하나로 줄여도 답의 내용은
    줄지 않는다.

    실호출 표본 변동(2026-09-04, 같은 질문 재현 중 관측) — 어떤 표본은
    ``operation`` 을 이미 "compare"로 내면서도 ``output.shape`` 만
    "scalar"에 남긴다(``selection``/``criterion_surface`` 는 그대로).
    ``operation`` 을 "retrieve"로 고정하지 않고 두 값 다 받아, 이 축이
    이미 옳게 나온 wire 도 shape 만 마저 고친다.
    """

    _MAX_COMPANIES = 8

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str) or not question.strip()
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        target = item.target
        selection = item.selection
        output = item.output
        companies = [entity for entity in intent.entities
                     if entity.kind_hint == "company"]
        if (
                item.operation not in ("retrieve", "compare")
                or target.kind != "metric"
                or output.shape != "scalar"
                or output.projection_mode != "named_fields"
                or not output.field_surfaces
                or selection is None
                or selection.mode not in ("maximum", "minimum")
                or not selection.criterion_surface
                or selection.criterion_surface not in question
                or len(companies) < 3
                or len(companies) > self._MAX_COMPANIES
                or [entity.entity_id for entity in companies]
                != list(target.entity_refs)
                or len(item.scope.target_period_expressions) != 1
        ):
            return intent
        field = next(
            (surface for surface in output.field_surfaces if surface in question),
            None)
        if field is None:
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        item_payload = payload["answer_items"][0]
        item_payload["operation"] = "compare"
        item_payload["output"]["shape"] = "comparison"
        item_payload["output"]["field_surfaces"] = [field]
        return SemanticIntent.model_validate(payload, strict=True)


#: 2개사 승자 비교의 최상급 단서. 질문에 **리터럴로** 있는 표면만 쓴다
#: (경계층이 근거 없는 criterion 표면을 거절한다). 긴 표기가 먼저 와야
#: 「더 많이」가 「더 많」보다 먼저 잡힌다 — `agent/concept_alias.py` 의
#: 표기 경쟁과 같은 규칙이다.
_TWO_COMPANY_MAXIMUM_CUES: tuple[str, ...] = (
    "더 많이", "더 높은", "더 많은", "더 크게", "더 큰", "더 많", "더 큼")
_TWO_COMPANY_MINIMUM_CUES: tuple[str, ...] = (
    "더 적게", "더 낮은", "더 적은", "더 작게", "더 작은", "더 적", "더 작")


def _two_company_superlative_cue(question: str) -> tuple[str, str] | None:
    """질문에서 최상급 단서 하나를 고른다. 없거나 양쪽이면 ``None``."""

    maximum = next(
        (cue for cue in _TWO_COMPANY_MAXIMUM_CUES if cue in question), None)
    minimum = next(
        (cue for cue in _TWO_COMPANY_MINIMUM_CUES if cue in question), None)
    if (maximum is None) == (minimum is None):
        # 둘 다 없으면 복구할 근거가 없고, 둘 다 있으면 어느 쪽 승자를
        # 묻는지 모른다 — 임의로 집으면 반대편 회사를 답한다.
        return None
    return ("maximum", maximum) if maximum is not None else ("minimum", minimum)


class TwoCompanySuperlativeSelectionRegrounder:
    """2개사 「누가 더 많이 …고 차이 얼마」의 빈 selection 을 되돌린다.

    R-P-003 「작년에 삼전이랑 하닉 중 누가 더 많이 팔았고 차이 얼마임?」.
    실호출 표본은 entity 둘·기간·``operation="compare"``·
    ``output.shape="comparison"``·답 필드 둘(승자·차이)까지 모두 옳게 내면서
    ``selection.mode`` 만 ``"none"`` 으로 비운다. 그러면 resolver 는 정상
    동작해(``financial_comparison``, 두 회사 연결 매출액, ``argmax`` +
    ``absolute_difference``) 권위를 만들지만, 컴파일러의 2개사 비교 검증
    (``_validate_financial_comparison_intent``,
    ``agent/deterministic_plan_compiler_v1.py``)이 「company comparison에는
    executable maximum/minimum selection이 필요합니다」로 거절해
    ``DeterministicPlanCompilerError`` 가 된다 — 같은 질문이 표본에 따라
    답하거나 못 답하는 자리다(실호출로 직접 확인함).

    **선택을 지어내지 않는다.** 경계층은 근거 없는 선택을 만들지 않는 것을
    계약으로 삼는다(``agent/semantic_intent_v1_boundary``: 「근거 없는 선택
    기준은 지운다. 만들지 않는다.」) — 이 regrounder도 같은 계약 아래
    있다. 질문 자체에 최상급 단서가 **리터럴로** 있을 때만, 그 리터럴을
    그대로 ``criterion_surface`` 로 쓴다. 단서가 없거나 최대·최소 단서가
    함께 있으면 손을 뗀다.

    회사가 셋 이상인 wire 는 형제
    ``NaryFinancialSuperlativeShapeRegrounder`` 의 몫이다(그쪽은
    operation·shape 두 축을 옮기고 selection 은 이미 있는 것을 요구한다) —
    이쪽은 정확히 2개사, 그리고 selection 한 축만 본다. 기간·연산·shape·
    entity·답 필드는 그대로 둔다.
    """

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str) or not question.strip()
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        companies = [entity for entity in intent.entities
                     if entity.kind_hint == "company"]
        if (
                item.target.kind != "metric"
                or item.operation != "compare"
                or item.output.shape != "comparison"
                or item.output.projection_mode != "named_fields"
                or len(item.output.field_surfaces) not in (1, 2)
                or len(companies) != 2
                or [entity.entity_id for entity in companies]
                != list(item.target.entity_refs)
                or len(item.scope.target_period_expressions) > 1
        ):
            return intent
        selection = item.selection
        if selection is not None and (
                selection.mode not in (None, "none")
                and selection.criterion_surface):
            return intent  # 이미 실행 가능한 선택이 있다.
        cue = _two_company_superlative_cue(question)
        if cue is None:
            return intent
        mode, criterion = cue

        payload = intent.model_dump(mode="python", warnings=False)
        # `k` 는 **없어야** 한다 — `SemanticIntent` 는 maximum/minimum 에
        # k 가 붙으면 거절한다(k 는 top_k 계열의 축이다).
        payload["answer_items"][0]["selection"] = {
            "mode": mode, "criterion_surface": criterion, "k": None}
        return SemanticIntent.model_validate(payload, strict=True)


def _selection_absent(selection: Any) -> bool:
    if selection is None:
        return True
    return (isinstance(selection, dict)
            and str(selection.get("mode") or "none") == "none"
            and not str(selection.get("criterion_surface") or "").strip()
            and not selection.get("k"))


def _metric_span_pattern(*literals: str) -> "re.Pattern[str]":
    surfaces = {pattern.key for pattern in CONCEPT_QUESTION_PATTERNS if pattern.key}
    ordered = sorted(surfaces, key=len, reverse=True)
    return re.compile("|".join([*literals, *(re.escape(s) for s in ordered)]))


class ClosedFinancialComparisonRegrounder:
    """Recover only question-explicit coordinates lost from metric compares."""

    _PERIOD = re.compile(
        r"(?<![0-9])(?:20)?[0-9]{2}\s*년도?|작년|지난해|재작년")
    #: 「2026년 3월 10일 낸 사업보고서」의 2026 은 제출한 날이지 답할 기간이
    #: 아니다.  아래 as_of 가드가 같은 말을 하지만 모델이 그 날짜를
    #: `as_of_expression` 에 담았을 때만 걸린다.  담지 않으면 연도만 주워
    #: 두 번째 기간이 되고, 그 기간의 기말이 as_of 보다 뒤라 계획이 거부된다
    #: (`Fact 기간은 Task as_of 이후일 수 없습니다`, `RPC-013`).
    _FILING_DATE_PHRASE = re.compile(
        r"(?<![0-9])(?:20)?[0-9]{2}\s*년\s*(?:1[0-2]|0?[1-9])\s*월\s*"
        r"(?:3[01]|[12][0-9]|0?[1-9])\s*일"
        r"(?=[^.?!]{0,12}(?:낸|내놓은|제출|공시|접수))")
    _MAXIMUM = re.compile(
        r"더\s*(?:큰|크|큼|많(?:은|음)?|높(?:은|음)?)|"
        r"(?:큰|높은|많은)\s*(?:기업|회사|곳)|"
        r"(?:누가\s*)?(?:컸지|많았지|높았지)",
        flags=re.IGNORECASE)
    _MINIMUM = re.compile(
        r"더\s*(?:작은|작|작음|적(?:은|음)?|낮(?:은|음)?)|"
        r"(?:작은|낮은|적은)\s*(?:기업|회사|곳)|"
        r"(?:누가\s*)?(?:작았지|적었지|낮았지)",
        flags=re.IGNORECASE)
    _RANKING = re.compile(r"순위")
    _COMPARISON = re.compile(r"(?<![A-Za-z])vs\.?(?![A-Za-z])|비교", re.IGNORECASE)
    _VERIFY_ONLY = re.compile(
        r"(?:컸지|작았지|많았지|적었지|맞지)\s*[?？]?$",
        flags=re.IGNORECASE)
    _METRIC = _metric_span_pattern(
        r"유형자산(?:의)?\s*취득(?:액|\s*현금유출액)?|설비\s*투자|"
        r"매출액?|영업\s*이익|당기\s*순이익|순이익|"
        r"자산\s*총계|총\s*자산|재고\s*자산|이익\s*잉여금|"
        r"매출\s*원가")
    # A comparison answer's field naturally may be labelled with the bare role
    # noun ("기업"/"회사"/"곳") rather than any company/metric/gap surface —
    # HCX's own field label for "who is bigger" is often just "기업"/"회사".
    # `_closed_role_surface` requires at least one anchor to be consumed from
    # the field before stripping neutral wrapper text, so a field that is
    # *itself* one whole neutral entity-role word never had anything to
    # consume and always failed closed. These words are already members of
    # `_closed_role_surface`'s own neutral grammar; listing them as anchors
    # here lets a bare instance of one be consumed (self-matched) so the
    # all-neutral residue check that follows can still accept it.
    _ENTITY_ROLE_WORDS = ("기업", "회사", "곳")

    def __init__(
            self,
            question_company_surface: Callable[[str, str], str | None] | None = None,
            ) -> None:
        """Create a question-grounded comparison repairer.

        The structured model often normalizes an alias (``LG엔솔`` ->
        ``LG에너지솔루션``).  At runtime the canonical company preflight can
        prove that the normalized surface is the same literal company mention
        in the question.  Isolated callers may omit the authority; then only
        literal entity surfaces are accepted, preserving fail-closed behavior.
        """

        if (question_company_surface is not None
                and not callable(question_company_surface)):
            raise TypeError("question company surface authority가 잘못되었습니다")
        self._question_company_surface = question_company_surface

    def _question_company_ids(
            self, payload: dict[str, Any], question: str,
            ) -> list[str]:
        """Company entity ids proven by a literal or canonical question alias."""

        company_ids: list[str] = []
        for row in payload.get("entities", []):
            if (not isinstance(row, dict)
                    or row.get("kind_hint") != "company"
                    or not isinstance(row.get("entity_id"), str)
                    or not isinstance(row.get("surface"), str)):
                continue
            surface = row["surface"]
            if surface not in question:
                authority = self._question_company_surface
                if authority is None:
                    continue
                try:
                    grounded = authority(surface, question)
                except Exception:  # noqa: BLE001 - authority failure stays closed
                    continue
                if not isinstance(grounded, str) or grounded not in question:
                    continue
            if row["entity_id"] not in company_ids:
                company_ids.append(row["entity_id"])
        return company_ids

    @classmethod
    def _span_concept(cls, surface: str, question: str) -> Any | None:
        """Resolve one ``_METRIC`` span to a concept, dictionary first.

        ``resolve_metric_concept`` only answers dictionary/suffix (``AUTO``)
        spellings.  A ``CONTEXT`` colloquial such as ``설비투자`` (→
        ``capex_ppe``, only with a cue like ``규모``/``얼마`` elsewhere in the
        question) needs the question-level resolver instead
        (`agent.concept_alias.resolve_from_question`); it is accepted here
        only when its registered surface is this exact matched span, so an
        unrelated colloquial elsewhere in the question cannot leak in.
        """

        concept = resolve_metric_concept(surface)
        if concept is not None:
            return concept
        colloquial = resolve_from_question(question, CONCEPT_QUESTION_PATTERNS)
        if colloquial.status == "resolved" and colloquial.surface == surface:
            return colloquial.concept
        return None

    @classmethod
    def _question_metric_match(cls, question: str) -> re.Match[str] | None:
        """Return one financial concept surface, never one of several metrics."""

        matches = list(cls._METRIC.finditer(question))
        concepts = {
            cls._span_concept(match.group(0), question) for match in matches}
        concepts.discard(None)
        if len(concepts) != 1:
            return None
        concept = next(iter(concepts))
        candidates = [
            match for match in matches
            if cls._span_concept(match.group(0), question) == concept]
        return max(candidates, key=lambda match: len(match.group(0))) \
            if candidates else None

    @classmethod
    def _closed_role_surface(
            cls, surface: Any, *, question: str, anchors: list[str],
            ) -> bool:
        """Accept bounded role labels only when their residue is neutral.

        Mere substring containment is unsafe: ``매출액 순위`` contains
        ``매출액`` but requests another operation.  A provider may still attach
        a company or statement wrapper to the same metric, so remove only
        question-proven role anchors and a small grammar of non-demand
        wrappers.  Anything else remains fail-closed.
        """

        if not isinstance(surface, str) or surface not in question:
            return False
        compact = re.sub(r"\s+", "", surface)
        neutral = re.compile(
            r"(?:연결|별도|개별|현금흐름표|재무제표|손익계산서|"
            r"재무상태표|기준|상|값|금액|결과|누가|기업|회사|곳|얼마나?|"
            r"은|는|이|가|의|과|와|을|를|에서|중)+")
        keys = sorted({
            re.sub(r"\s+", "", anchor)
            for anchor in anchors
            if isinstance(anchor, str) and anchor
        }, key=len, reverse=True)
        residue = compact
        consumed = False
        for key in keys:
            if key in residue:
                residue = residue.replace(key, "")
                consumed = True
        return consumed and not neutral.sub("", residue)

    @classmethod
    def _question_metric_anchors(
            cls, question: str, metric: re.Match[str],
            ) -> list[str]:
        """Return only aliases explicitly equated with the resolved metric.

        Bare ``capex`` is intentionally ambiguous in the project contract.
        ``capex(유형자산 취득액)`` is different: the question itself supplies
        the disambiguating expansion, so both the parenthesized surface and
        its short label can safely be treated as the same output role.
        """

        concept = cls._span_concept(metric.group(0), question)
        anchors = [
            match.group(0) for match in cls._METRIC.finditer(question)
            if cls._span_concept(match.group(0), question) == concept
        ]
        for match in re.finditer(
                r"(?P<alias>capex|캐펙스)\s*\(\s*(?P<detail>[^()]+?)\s*\)",
                question, flags=re.IGNORECASE):
            detail = cls._METRIC.search(match.group("detail"))
            if (detail is not None
                    and cls._span_concept(detail.group(0), question) == concept):
                anchors.extend([match.group(0), match.group("alias")])
        return list(dict.fromkeys(anchors))

    @classmethod
    def _explicit_winner_direction(
            cls, question: str,
            ) -> tuple[re.Match[str], str] | None:
        """Return only a direction that the question explicitly states."""

        maximum = list(cls._MAXIMUM.finditer(question))
        minimum = list(cls._MINIMUM.finditer(question))
        if maximum and minimum:
            return None
        if maximum:
            return maximum[0], "maximum"
        if minimum:
            return minimum[0], "minimum"
        return None

    @classmethod
    def _winner_direction(
            cls, question: str,
            ) -> tuple[re.Match[str], str] | None:
        """Return the closed two-company product comparison direction.

        Explicit maximum/minimum language always wins.  The frozen v0.4
        contract defines a bare two-company financial ``비교``/``vs`` as both
        values plus the larger side and their absolute gap.  Callers handling
        wider company sets must use :meth:`_explicit_winner_direction`
        instead, so this product default cannot invent an N-ary ranking.
        """

        explicit = cls._explicit_winner_direction(question)
        if explicit is not None:
            return explicit
        comparison = cls._COMPARISON.search(question)
        return (comparison, "maximum") if comparison is not None else None

    @classmethod
    def _comparison_markers(cls, question: str) -> list[str]:
        """Preserve every literal comparison marker as a redundant role."""

        return list(dict.fromkeys(
            match.group(0) for match in cls._COMPARISON.finditer(question)))

    def _rebuild_closed_company_winner_gap(
            self, payload: dict[str, Any], question: str,
            ) -> bool:
        """Canonicalize one explicit two-company winner-and-gap request."""

        items = payload.get("answer_items")
        entities = payload.get("entities")
        if (not isinstance(items, list) or not 1 <= len(items) <= 3
                or not isinstance(entities, list)
                or payload.get("answer_groups") or payload.get("premises")
                or payload.get("unresolved_mentions")):
            return False
        company_ids = self._question_company_ids(payload, question)
        periods = list(dict.fromkeys(
            match.group(0).strip() for match in self._PERIOD.finditer(question)))
        metric = self._question_metric_match(question)
        winner_direction = self._winner_direction(question)
        gap = re.search(r"(?:절대\s*)?차이", question)
        if (len(company_ids) != 2 or len(periods) > 1 or metric is None
                or winner_direction is None or gap is None):
            return False
        winner, mode = winner_direction

        allowed_refs = set(company_ids)
        anchors = [
            *(row["surface"] for row in entities
              if isinstance(row, dict) and row.get("entity_id") in allowed_refs),
            *periods, *self._question_metric_anchors(question, metric),
            winner.group(0), gap.group(0), "차이", *self._ENTITY_ROLE_WORDS,
        ]
        for row in items:
            target = row.get("target", {})
            output = row.get("output", {})
            refs = target.get("entity_refs", [])
            if (target.get("kind") != "metric"
                    or not isinstance(refs, list)
                    or any(ref not in allowed_refs for ref in refs)
                    or not _selection_absent(row.get("selection"))
                    or output.get("projection_mode") not in {
                        "named_fields", "whole_target"}):
                return False
            for surface in output.get("field_surfaces", []):
                if isinstance(surface, str) and surface not in question:
                    continue
                if not self._closed_role_surface(
                        surface, question=question, anchors=anchors):
                    return False

        row = items[0]
        row["item_id"] = "item-1"
        row["operation"] = "compare"
        row["target"].update({
            "surface": metric.group(0),
            "entity_refs": company_ids,
            "qualifier_surfaces": [],
        })
        row["scope"].update({
            "target_period_expressions": periods,
            "scope_qualifier_expressions": (
                ["연결"] if "연결" in question else
                ["별도"] if any(token in question for token in ("별도", "개별"))
                else []),
        })
        row["selection"] = {
            "mode": mode, "criterion_surface": winner.group(0), "k": None,
        }
        row["output"].update({
            "shape": "comparison", "projection_mode": "named_fields",
            "field_surfaces": [winner.group(0), gap.group(0)],
        })
        payload["answer_items"] = [row]
        return True

    @staticmethod
    def _split_single_annual_change_item(
            payload: dict[str, Any], question: str,
            ) -> bool:
        """Expand one combined amount/rate demand into the compiler contract.

        Structured output sometimes represents ``얼마나, 몇 퍼센트
        변했는가`` as one comparison item, while other samples emit two or
        three items.  Those are alternative surfaces for the same closed
        request only when the question itself proves one issuer, one financial
        metric, two annual endpoints, and both requested operators.  Existing
        non-empty axes must be subsets of those literal question coordinates;
        anything wider remains fail-closed.
        """

        items = payload.get("answer_items")
        entities = payload.get("entities")
        if (not isinstance(items, list) or len(items) != 1
                or not isinstance(entities, list)
                or payload.get("answer_groups") or payload.get("premises")
                or payload.get("unresolved_mentions")):
            return False
        company_ids = [
            row.get("entity_id") for row in entities
            if isinstance(row, dict)
            and row.get("kind_hint") == "company"
            and isinstance(row.get("surface"), str)
            and row["surface"] in question
        ]
        if len(company_ids) != 1:
            return False
        row = items[0]
        target = row.get("target", {})
        scope = row.get("scope", {})
        output = row.get("output", {})
        if (target.get("kind") != "metric"
                or target.get("entity_refs") != company_ids
                or row.get("operation") not in {"retrieve", "compare"}
                or row.get("selection") is not None
                or output.get("shape") not in {"scalar", "comparison"}
                or output.get("projection_mode") not in {
                    "named_fields", "whole_target"}
                or target.get("surface") not in question):
            return False

        periods = list(dict.fromkeys(
            match.group(0).strip()
            for match in ClosedFinancialComparisonRegrounder._PERIOD.finditer(
                question)))
        if len(periods) != 2:
            return False
        current_periods = scope.get("target_period_expressions", [])
        if (not isinstance(current_periods, list)
                or not set(current_periods).issubset(set(periods))):
            return False
        qualifiers = scope.get("scope_qualifier_expressions", [])
        if (not isinstance(qualifiers, list)
                or any(not isinstance(value, str) or value not in question
                       for value in qualifiers)):
            return False

        amount = re.search(r"얼마나", question)
        percent = re.search(
            r"몇\s*퍼센트(?:\s*(?:증가|감소))?\s*"
            r"(?:변했는가|변했나|변했어|변했는지|변화했는가|변화했나)",
            question)
        if amount is None or percent is None:
            return False
        allowed_fields = {
            amount.group(0), percent.group(0), target.get("surface"), *periods,
        }
        fields = output.get("field_surfaces", [])
        if (not isinstance(fields, list)
                or any(not isinstance(value, str) or value not in question
                       or not any(
                           re.sub(r"\s+", "", value)
                           in re.sub(r"\s+", "", allowed)
                           or re.sub(r"\s+", "", allowed)
                           in re.sub(r"\s+", "", value)
                           for allowed in allowed_fields if isinstance(allowed, str))
                       for value in fields)):
            return False

        amount_row = deepcopy(row)
        rate_row = deepcopy(row)
        for out_row in (amount_row, rate_row):
            out_row["scope"]["target_period_expressions"] = list(periods)
            if "연결" in question and not qualifiers:
                out_row["scope"]["scope_qualifier_expressions"] = ["연결"]
            out_row["selection"] = None
            out_row["output"].update({
                "shape": "scalar", "projection_mode": "named_fields",
            })
        amount_row.update({"item_id": "item-1", "operation": "retrieve"})
        amount_row["output"]["field_surfaces"] = [amount.group(0)]
        rate_row.update({"item_id": "item-2", "operation": "compare"})
        rate_row["output"]["field_surfaces"] = [percent.group(0)]
        payload["answer_items"] = [amount_row, rate_row]
        return True

    @staticmethod
    def _canonicalize_parallel_annual_change(
            payload: dict[str, Any], question: str,
            ) -> bool:
        """Canonicalize one closed amount-and-rate annual comparison.

        A provider can express the same explicit request as retrieve/retrieve,
        compare/compare, or in percent-first order.  Those stochastic labels
        must not change the executable contract.  Repair is deliberately
        limited to two scalar fields over one identical company/metric/scope;
        both field surfaces and both annual endpoints must be literal question
        spans.  Mismatched axes remain fail-closed.
        """

        items = payload.get("answer_items")
        if (not isinstance(items, list) or len(items) != 2
                or payload.get("answer_groups") or payload.get("premises")
                or payload.get("unresolved_mentions")):
            return False
        if any(
                row.get("target", {}).get("kind") != "metric"
                or len(row.get("target", {}).get("entity_refs", [])) != 1
                or row.get("selection") is not None
                or row.get("output", {}).get("shape") != "scalar"
                or row.get("output", {}).get("projection_mode") != "named_fields"
                or len(row.get("output", {}).get("field_surfaces", [])) != 1
                for row in items):
            return False

        target_signatures = [(
            row["target"].get("surface"),
            tuple(row["target"].get("entity_refs", [])),
            tuple(row["target"].get("qualifier_surfaces", [])),
        ) for row in items]
        scope_signatures = [(
            row["scope"].get("as_of_expression"),
            row["scope"].get("document_group_expression"),
            tuple(row["scope"].get("scope_qualifier_expressions", [])),
        ) for row in items]
        if len(set(target_signatures)) != 1 or len(set(scope_signatures)) != 1:
            return False

        periods = list(dict.fromkeys(
            match.group(0).strip()
            for match in ClosedFinancialComparisonRegrounder._PERIOD.finditer(
                question)))
        if len(periods) != 2:
            return False
        question_periods = set(periods)
        current_periods = [
            row["scope"].get("target_period_expressions", []) for row in items]
        if any(
                not values
                or not set(values).issubset(question_periods)
                for values in current_periods):
            return False

        amount_row = None
        percent_row = None
        for row in items:
            surface = row["output"]["field_surfaces"][0]
            if surface not in question:
                return False
            is_amount = re.search(
                r"증감\s*액|변동\s*액|변화\s*액|차이|얼마나", surface)
            is_percent = re.search(
                r"%|퍼센트|몇\s*퍼(?:\b|센트)|증감\s*률|변동\s*률|변화\s*율", surface)
            if bool(is_amount) == bool(is_percent):
                return False
            if is_amount:
                if amount_row is not None:
                    return False
                amount_row = row
            else:
                if percent_row is not None:
                    return False
                percent_row = row
        if amount_row is None or percent_row is None:
            return False

        amount_row["operation"] = "retrieve"
        percent_row["operation"] = "compare"
        for row in (amount_row, percent_row):
            row["scope"]["target_period_expressions"] = list(periods)
        amount_row["item_id"] = "item-1"
        percent_row["item_id"] = "item-2"
        payload["answer_items"] = [amount_row, percent_row]
        return True

    @staticmethod
    def _collapse_redundant_period_retrieval(
            payload: dict[str, Any], question: str,
            ) -> bool:
        """Close one split two-period amount/rate comparison.

        HCX may emit both operand lookups plus a comparison as three answer
        items.  The operand rows are not independent user promises when all
        rows share one metric, entity axis and scope, the comparison already
        names the same two periods, and the question literally requests an
        amount change and a rate change.  In that closed case retain two
        compare demands; any mismatched coordinate declines unchanged.
        """

        items = payload.get("answer_items")
        if (not isinstance(items, list) or len(items) != 3
                or payload.get("answer_groups") or payload.get("premises")
                or payload.get("unresolved_mentions")):
            return False
        compares = [row for row in items if row.get("operation") == "compare"]
        retrieves = [row for row in items if row.get("operation") == "retrieve"]
        if len(compares) != 1 or len(retrieves) != 2:
            return False
        compare = compares[0]
        signatures = [(
            row.get("target", {}).get("kind"),
            row.get("target", {}).get("surface"),
            tuple(row.get("target", {}).get("entity_refs", [])),
            tuple(row.get("target", {}).get("qualifier_surfaces", [])),
            row.get("scope", {}).get("as_of_expression"),
            row.get("scope", {}).get("document_group_expression"),
            tuple(row.get("scope", {}).get("scope_qualifier_expressions", [])),
        ) for row in items]
        if len(set(signatures)) != 1 or signatures[0][0] != "metric":
            return False
        compare_periods = compare.get("scope", {}).get(
            "target_period_expressions", [])
        retrieve_periods = [
            row.get("scope", {}).get("target_period_expressions", [])
            for row in retrieves
        ]
        if (len(compare_periods) != 2
                or any(len(periods) != 1 for periods in retrieve_periods)
                or set(compare_periods) != {
                    periods[0] for periods in retrieve_periods
                }):
            return False
        amount_surface = next((
            surface
            for row in retrieves
            for surface in row.get("output", {}).get("field_surfaces", [])
            if surface in question
            and re.search(r"얼마나|증감\s*액|차이", surface)
        ), None)
        percent_surface = next((
            surface
            for surface in compare.get("output", {}).get("field_surfaces", [])
            if surface in question
            and re.search(r"퍼센트|퍼|증감\s*률|변동\s*률", surface)
        ), None)
        if amount_surface is None or percent_surface is None:
            return False

        amount = retrieves[0]
        amount["scope"]["target_period_expressions"] = list(compare_periods)
        amount["output"].update({
            "shape": "scalar",
            "projection_mode": "named_fields",
            "field_surfaces": [amount_surface],
        })
        compare["output"].update({
            "shape": "scalar",
            "projection_mode": "named_fields",
            "field_surfaces": [percent_surface],
        })
        amount["item_id"] = "item-1"
        compare["item_id"] = "item-2"
        payload["answer_items"] = [amount, compare]
        return True

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        payload = intent.model_dump(mode="python", warnings=False)
        if self._rebuild_closed_company_winner_gap(payload, question):
            return SemanticIntent.model_validate(payload, strict=True)
        if self._collapse_redundant_period_retrieval(payload, question):
            return SemanticIntent.model_validate(payload, strict=True)
        if self._canonicalize_parallel_annual_change(payload, question):
            return SemanticIntent.model_validate(payload, strict=True)
        if self._split_single_annual_change_item(payload, question):
            return SemanticIntent.model_validate(payload, strict=True)
        if len(intent.answer_items) != 1:
            return intent
        item = intent.answer_items[0]
        if item.target.kind != "metric":
            return intent
        # One metric, one fiscal period, and one exact as-of day is already a
        # closed document-coordinate read. The filing date's calendar year is
        # not a second comparison period (for example a 2025 filing carrying
        # 2024 annual statements).
        if (item.operation == "retrieve"
                and len(item.scope.target_period_expressions) == 1
                and item.scope.as_of_expression is not None):
            start, end, error = _target_date_range(
                item.scope.as_of_expression, reference_date=date(2026, 1, 1))
            if error is None and start is not None and start == end:
                return intent
        row = payload["answer_items"][0]
        changed = False

        # 이미 기간이 있으면 제출일에서 연도를 더 주워 오지 않는다.  기간이
        # 아예 없을 때는 그 연도가 유일한 좌표일 수 있으므로 건드리지 않는다.
        current = row["scope"]["target_period_expressions"]
        filing_spans = ([match.span()
                         for match in self._FILING_DATE_PHRASE.finditer(question)]
                        if current else [])
        periods = list(dict.fromkeys(
            match.group(0).strip()
            for match in self._PERIOD.finditer(question)
            if not any(start <= match.start() < end
                       for start, end in filing_spans)))
        for period in periods:
            if any(period in expression for expression in current):
                continue
            qualifiers = row["target"]["qualifier_surfaces"]
            exact_qualifiers = [
                value for value in qualifiers
                if re.sub(r"\s+", "", value) == re.sub(r"\s+", "", period)
            ]
            if exact_qualifiers:
                row["target"]["qualifier_surfaces"] = [
                    value for value in qualifiers if value not in exact_qualifiers
                ]
            elif any(period in value for value in qualifiers):
                # A year embedded in ``25년 2Q`` belongs to the closed
                # single-quarter expression; adding an annual period would
                # create a second, unintended coordinate.
                continue
            if period not in current:
                current.append(period)
                changed = True

        if ("연결" in question
                and not row["scope"]["scope_qualifier_expressions"]):
            row["scope"]["scope_qualifier_expressions"] = ["연결"]
            changed = True
        elif (any(token in question for token in ("별도", "개별"))
                and not row["scope"]["scope_qualifier_expressions"]):
            token = "별도" if "별도" in question else "개별"
            row["scope"]["scope_qualifier_expressions"] = [token]
            changed = True

        company_ids = self._question_company_ids(payload, question)
        # A two-company financial comparison is one closed v0.4 product
        # contract even when HCX labels it as scalar retrieval or emits only
        # one answer field.  An ordinary comparison requests the selected side
        # and absolute gap; a sentence-final premise check (``A가 B보다
        # 컸지?``) requests only the selected side used to verify that premise.
        # Both forms still expose the two source operand values downstream.
        # This is intentionally limited to one question-grounded metric and
        # one annual period; an unknown output demand remains untouched and
        # therefore fails closed.
        winner_direction = self._winner_direction(question)
        metric = self._question_metric_match(question)
        gap = re.search(r"(?:절대\s*)?차이", question)
        if (len(company_ids) == 2 and len(current) == 1
                and winner_direction is not None
                and metric is not None
                and not payload["premises"]
                and not payload["unresolved_mentions"]):
            winner, mode = winner_direction
            metric_surface = metric.group(0)
            verification_only = self._VERIFY_ONLY.search(question) is not None
            desired_fields = (
                [winner.group(0)]
                if verification_only else
                [winner.group(0), (
                    gap.group(0) if gap is not None else metric_surface)]
            )
            # Overwriting the field list would silently drop any further
            # demand the provider recorded (for example ``순위``).  Only
            # canonicalize when every existing field is already a company,
            # metric, comparison, winner, period, scope, or gap span from this
            # same question.  An extra demand keeps the received shape.
            company_surfaces: list[str] = []
            for entity in intent.entities:
                if entity.entity_id not in company_ids:
                    continue
                if entity.surface in question:
                    company_surfaces.append(entity.surface)
                    continue
                authority = self._question_company_surface
                if authority is not None:
                    try:
                        grounded = authority(entity.surface, question)
                    except Exception:  # noqa: BLE001
                        grounded = None
                    if isinstance(grounded, str) and grounded in question:
                        company_surfaces.append(grounded)

            anchors = [
                *desired_fields, *company_surfaces,
                *self._question_metric_anchors(question, metric),
                winner.group(0), *current,
                *(row["scope"].get("scope_qualifier_expressions") or []),
                *self._comparison_markers(question), *self._ENTITY_ROLE_WORDS,
            ]
            extra_demand = any(
                not self._closed_role_surface(
                    value, question=question, anchors=anchors)
                for value in row["output"]["field_surfaces"]
            )
            if extra_demand:
                return SemanticIntent.model_validate(payload, strict=True) \
                    if changed else intent
            if row["operation"] != "compare":
                row["operation"] = "compare"
                changed = True
            if row["target"]["surface"] != metric_surface:
                row["target"]["surface"] = metric_surface
                changed = True
            if row["target"]["entity_refs"] != company_ids:
                row["target"]["entity_refs"] = company_ids
                changed = True
            if row["selection"] != {
                    "mode": mode,
                    "criterion_surface": winner.group(0), "k": None}:
                row["selection"] = {
                    "mode": mode,
                    "criterion_surface": winner.group(0), "k": None,
                }
                changed = True
            if (row["output"]["shape"] != "comparison"
                    or row["output"]["projection_mode"] != "named_fields"
                    or row["output"]["field_surfaces"] != desired_fields):
                row["output"].update({
                    "shape": "comparison",
                    "projection_mode": "named_fields",
                    "field_surfaces": desired_fields,
                })
                changed = True
        # A provider may flatten "A기간 매출과 B기간 매출의 차이" into a
        # retrieve of a synthetic metric called "매출 차이".  The question
        # itself already provides two periods, a known financial metric and
        # the difference operator, so restore the comparison topology without
        # consulting an answer value or fixture.
        difference = re.search(r"(?P<surface>(?:매출액?|영업\s*이익|당기\s*순이익|순이익|총\s*자산)\s*차이)", question)
        metric = self._METRIC.search(question)
        if (len(company_ids) == 1 and len(current) == 2
                and difference is not None and metric is not None):
            metric_surface = metric.group(0)
            if row["operation"] != "compare":
                row["operation"] = "compare"
                changed = True
            if row["target"]["surface"] != metric_surface:
                row["target"]["surface"] = metric_surface
                changed = True
            if row["output"]["shape"] != "comparison":
                row["output"]["shape"] = "comparison"
                changed = True
            if row["output"]["field_surfaces"] != [difference.group("surface")]:
                row["output"]["field_surfaces"] = [difference.group("surface")]
                changed = True

        if row["operation"] == "compare":
            if len(company_ids) in {2, 3, 4, 5, 6, 7, 8} \
                    and row["target"]["entity_refs"] != company_ids:
                row["target"]["entity_refs"] = company_ids
                changed = True
            if len(company_ids) in {2, 3, 4, 5, 6, 7, 8} and row["selection"] is None:
                # The v0.4 bare-comparison default is strictly a two-company
                # contract and is already handled above.  A wider comparison
                # needs an explicit direction or literal ranking demand.
                winner_direction = self._explicit_winner_direction(question)
                if (winner_direction is None
                        and (ranking := self._RANKING.search(question)) is not None):
                    # A named financial ranking is conventionally descending;
                    # unlike a bare ``비교``/``vs``, it explicitly requests an
                    # ordering result.  Keep this route limited to the already
                    # proved 2~8-company financial comparison branch.
                    winner_direction = (ranking, "maximum")
                if winner_direction is not None:
                    match, mode = winner_direction
                    row["selection"] = {
                        "mode": mode, "criterion_surface": match.group(0),
                        "k": None,
                    }
                    changed = True
            verification = self._VERIFY_ONLY.search(question)
            if (len(company_ids) == 2 and verification is not None
                    and not payload["premises"]):
                raw_text = question.strip().rstrip("?？").strip()
                if raw_text and raw_text in question:
                    payload["premises"] = [{
                        "premise_id": "premise-1",
                        "kind": "comparison",
                        "raw_text": raw_text,
                        "applies_to_item_ids": [row["item_id"]],
                    }]
                    changed = True
            # Close the explicit two-company ``argmax + difference`` topology
            # even when HCX copied company names into output fields.  The
            # compiler needs the two requested operations, not entity labels.
            # This is deliberately bounded: every discarded field must be a
            # literal company/metric/criterion/difference span from the same
            # question.  An unknown third demand therefore remains fail-closed
            # instead of being silently erased.
            difference = re.search(r"차이(?:\s*(?:는|가)?\s*얼마)?", question)
            selection = row["selection"]
            if (len(company_ids) == 2 and selection is not None
                    # 이슈 #124 — 「가장 작은」도 같은 두 회사 argmax+difference
                    # 위상이다. 방향만 반대다.
                    and selection.get("mode") in ("maximum", "minimum")
                    and self._VERIFY_ONLY.search(question) is None
                    and difference is not None):
                company_surfaces: list[str] = []
                by_id = {entity.entity_id: entity for entity in intent.entities}
                for entity_id in company_ids:
                    entity = by_id.get(entity_id)
                    if entity is None:
                        continue
                    if entity.surface in question:
                        company_surfaces.append(entity.surface)
                        continue
                    authority = self._question_company_surface
                    if authority is not None:
                        try:
                            grounded = authority(entity.surface, question)
                        except Exception:  # noqa: BLE001
                            grounded = None
                        if isinstance(grounded, str) and grounded in question:
                            company_surfaces.append(grounded)
                criterion = selection["criterion_surface"]
                metric_surface = row["target"]["surface"]
                metric = self._question_metric_match(question)
                anchors = [
                    *company_surfaces, criterion, difference.group(0), "차이",
                    *self._comparison_markers(question), *self._ENTITY_ROLE_WORDS,
                    *(self._question_metric_anchors(question, metric)
                      if metric is not None else [metric_surface]),
                ]

                fields = row["output"]["field_surfaces"]
                normalized = [criterion, difference.group(0)]
                if (all(self._closed_role_surface(
                            surface, question=question, anchors=anchors)
                        for surface in fields)
                        and criterion != difference.group(0)
                        and fields != normalized):
                    row["output"]["field_surfaces"] = normalized
                    changed = True
            if len(company_ids) == 1 and len(current) == 2:
                relation = re.search(
                    r"몇\s*퍼센트\s*증가\s*또는\s*감소|얼마나\s*늘었냐|전년비\s*몇퍼",
                    question,
                )
                if relation is not None and row["output"]["field_surfaces"] != [
                        relation.group(0)]:
                    row["output"]["field_surfaces"] = [relation.group(0)]
                    changed = True

        return (
            SemanticIntent.model_validate(payload, strict=True)
            if changed else intent
        )


class QuestionGroundedLatestEventRegrounder:
    """Restore an explicit latest selector omitted by the provider."""

    _LATEST = re.compile(r"가장\s*최근|최근")
    _MANY = re.compile(r"표(?:로)?|목록|리스트|모두|전부|각각|여러\s*건")

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (len(intent.answer_items) != 1 or intent.answer_groups
                or intent.premises or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        match = self._LATEST.search(question)
        if (match is None or self._MANY.search(question) is not None
                or item.target.kind != "event"
                or item.operation != "retrieve"
                or item.selection is not None
                or item.output.projection_mode != "named_fields"
                or not item.output.field_surfaces):
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        payload["answer_items"][0]["selection"] = {
            "mode": "latest", "criterion_surface": match.group(0), "k": None,
        }
        return SemanticIntent.model_validate(payload, strict=True)


class NamedEventFieldsRegrounder:
    """Normalize one selected event's ordered named fields to a record.

    The adapter changes only the output container.  Every field surface must
    already be a literal question span and the downstream canonical preflight
    must still prove one event.  It never chooses a receipt or splits fields
    across events.
    """

    def __init__(self, canonical: Any | None = None) -> None:
        self.canonical = canonical

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str)
                or len(intent.answer_items) != 1
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]

        # Structured output may preserve ``계약상대와 계약명`` as one literal
        # field instead of the two literal answer demands.  Split only this
        # closed, question-grounded contract grammar; no synonym or answer is
        # invented here.  The canonical event preflight below must still prove
        # one event before the compiler can emit a plan.
        contract_pair = re.search(
            r"(?P<counterparty>계약\s*상대(?:방)?|거래\s*상대|상대방)\s*"
            r"(?:와|과|및|·)\s*"
            r"(?P<contract_name>(?:체결|해지)?\s*계약\s*명)",
            question,
        )
        field_surfaces = list(item.output.field_surfaces)
        if (contract_pair is not None and len(field_surfaces) == 1
                and field_surfaces[0] == contract_pair.group(0)):
            payload = intent.model_dump(mode="python", warnings=False)
            payload["answer_items"][0]["output"]["field_surfaces"] = [
                contract_pair.group("counterparty"),
                contract_pair.group("contract_name"),
            ]
            intent = SemanticIntent.model_validate(payload, strict=True)
            item = intent.answer_items[0]
        companies = [entity for entity in intent.entities
                     if entity.kind_hint == "company"]
        if self.canonical is None:
            issuer_ids = ([companies[0].entity_id]
                          if len(companies) == 1 else [])
        else:
            issuer_ids = []
            issuer_codes: set[str] = set()
            for entity in companies:
                rows = self.canonical.resolve_company(entity.surface)
                if len(rows) == 1:
                    issuer_ids.append(entity.entity_id)
                    issuer_codes.add(rows[0].corp_code)
            if len(issuer_codes) != 1:
                issuer_ids = []
        entity_ids = {entity.entity_id for entity in intent.entities}
        if (
                len(issuer_ids) != 1
                or not item.target.entity_refs
                or not set(item.target.entity_refs).issubset(entity_ids)
                or (issuer_ids[0] not in item.target.entity_refs
                    and not (len(intent.entities) == 2
                             and len(item.target.entity_refs) == 1))
                or item.target.kind not in {"event", "document", "entity"}
                or item.operation != "retrieve"
                or item.target.qualifier_surfaces
                or item.scope.target_period_expressions
                or item.scope.as_of_expression is not None
                or item.scope.document_group_expression is not None
                or item.scope.scope_qualifier_expressions
                or getattr(item.selection, "mode", None) not in {None, "latest"}
                or item.output.projection_mode != "named_fields"
                or len(item.output.field_surfaces) < 2
                or any(surface not in question
                       for surface in item.output.field_surfaces)
                or item.output.shape not in {"scalar", "record", "narrative"}
        ):
            return intent

        # A named contract object can arrive as a generic ``entity``.  Do not
        # route arbitrary entity records to event lookup: accept only the
        # closed contract grammar whose literal field pair asks for both a
        # counterparty and a contract name.  Canonical event preflight still
        # has to prove one event, so this introduces no receipt or answer.
        compact_fields = {
            re.sub(r"\s+", "", surface)
            for surface in item.output.field_surfaces
        }
        counterparty_fields = {
            "계약상대", "계약상대방", "거래상대", "상대방",
        }
        contract_name_fields = {
            "계약명", "체결계약명", "해지계약명",
        }
        named_contract_record = (
            contract_pair is not None
            and item.target.surface in question
            and bool(compact_fields & counterparty_fields)
            and bool(compact_fields & contract_name_fields)
            and compact_fields.issubset(
                counterparty_fields | contract_name_fields)
        )
        if item.target.kind == "entity" and not named_contract_record:
            return intent
        ordered_surfaces = sorted(
            enumerate(item.output.field_surfaces),
            key=lambda row: (question.index(row[1]), row[0]),
        )
        question_order = [surface for _, surface in ordered_surfaces]
        if (not named_contract_record and item.output.shape == "record"
                and item.output.field_surfaces == question_order):
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        if named_contract_record:
            payload["answer_items"][0]["target"]["kind"] = "event"
            # The selected-event compiler binds one target axis.  Keep the
            # canonical issuer on that axis; the explicit counterparty entity
            # remains in the intent and is independently consumed by event
            # preflight as a selector facet.
            payload["answer_items"][0]["target"]["entity_refs"] = [
                issuer_ids[0]]
        payload["answer_items"][0]["output"]["shape"] = "record"
        payload["answer_items"][0]["output"]["field_surfaces"] = question_order
        return SemanticIntent.model_validate(payload, strict=True)


class ClosedFinancialRelationRegrounder:
    """Recover explicit scope-pair and year-over-year relation topologies.

    The repair copies only spans present in the question.  The resolver, not
    this adapter, derives the previous accounting coordinate for ``전년비``.
    """

    _METRIC = re.compile(
        r"유형자산\s*취득(?:액|\s*현금유출액)?|영업이익|당기순이익|순이익|"
        r"자산총계|총자산|재고\s*자산|이익\s*잉여금|매출\s*원가|"
        r"매출액|매출")
    _YEAR = re.compile(r"(?<![0-9])(?:20)?[0-9]{2}\s*년(?:도)?")
    _YOY = re.compile(r"전년비\s*몇\s*퍼(?:센트)?")

    @staticmethod
    def _question_company_ids(
            payload: dict[str, Any], question: str,
            ) -> list[str]:
        return list(dict.fromkeys(
            row.get("entity_id") for row in payload.get("entities", [])
            if isinstance(row, dict) and row.get("kind_hint") == "company"
            and isinstance(row.get("surface"), str)
            and row["surface"] in question
        ))

    @classmethod
    def _scope_gap_field_is_closed(
            cls, surface: Any, *, question: str, metric_concept: Any,
            ) -> bool:
        """Whether one provider field is only a scope-gap answer label."""

        if not isinstance(surface, str) or not surface.strip():
            return False
        metric_matches = list(cls._METRIC.finditer(surface))
        if metric_matches and any(
                resolve_metric_concept(match.group(0)) != metric_concept
                for match in metric_matches):
            return False
        remainder = surface
        for match in reversed(metric_matches):
            remainder = remainder[:match.start()] + remainder[match.end():]
        compact = re.sub(r"[^0-9A-Za-z가-힣]", "", remainder)
        compact = re.sub(
            r"(?:연결|별도|개별|절대|차이|격차|값|금액|결과|계산|"
            r"얼마|알려줘?|구해줘?|[과와의을를])", "", compact)
        if not compact:
            return True
        # A literal demand copied from the closed question is also harmless.
        return surface in question and bool(re.fullmatch(
            r"(?:절대\s*)?(?:차이|격차)|얼마|계산(?:해줘)?", surface.strip()))

    def _closed_scope_gap(
            self, payload: dict[str, Any], question: str,
            ) -> bool:
        """Recover one question-closed CFS/SFS absolute-gap topology.

        HCX can express the same request as one, two, or three metric items and
        vary retrieve/compare plus scalar/comparison/record output shapes.  A
        unique company, metric concept, annual period, both statement scopes,
        and one non-directional gap demand fully determine the executable
        topology, so those representational differences carry no authority.
        """

        items = payload.get("answer_items", [])
        if (not 1 <= len(items) <= 3 or payload.get("answer_groups")
                or payload.get("premises") or payload.get("unresolved_mentions")
                or "연결" not in question
                or not any(token in question for token in ("별도", "개별"))):
            return False
        companies = self._question_company_ids(payload, question)
        years = list(dict.fromkeys(
            match.group(0).strip() for match in self._YEAR.finditer(question)))
        metric_matches = list(self._METRIC.finditer(question))
        metric_concepts = {
            resolve_metric_concept(match.group(0)) for match in metric_matches}
        metric_concepts.discard(None)
        difference = re.search(r"(?:절대\s*)?(?:차이|격차)", question)
        if (len(companies) != 1 or len(years) != 1
                or len(metric_concepts) != 1 or not metric_matches
                or difference is None):
            return False
        metric_concept = next(iter(metric_concepts))
        company_id = companies[0]

        for row in items:
            target = row.get("target", {})
            scope = row.get("scope", {})
            output = row.get("output", {})
            target_metric = self._METRIC.search(str(target.get("surface", "")))
            refs = target.get("entity_refs", [])
            periods = scope.get("target_period_expressions", [])
            qualifiers = [
                *target.get("qualifier_surfaces", []),
                *scope.get("scope_qualifier_expressions", []),
            ]
            if (target.get("kind") != "metric"
                    or target_metric is None
                    or resolve_metric_concept(target_metric.group(0))
                    != metric_concept
                    or row.get("operation") not in {"retrieve", "compare"}
                    or not isinstance(refs, list)
                    or not refs or any(ref != company_id for ref in refs)
                    or row.get("selection") is not None
                    or output.get("shape") not in {
                        "scalar", "comparison", "record"}
                    or output.get("projection_mode") not in {
                        "named_fields", "whole_target"}
                    or any(period not in question or period != years[0]
                           for period in periods)
                    or any(not isinstance(value, str) or value not in question
                           for value in qualifiers)
                    or any(not self._scope_gap_field_is_closed(
                        surface, question=question,
                        metric_concept=metric_concept)
                        for surface in output.get("field_surfaces", []))):
                return False

        row = items[0]
        row["item_id"] = "item-1"
        row["operation"] = "compare"
        row["target"].update({
            "surface": metric_matches[0].group(0),
            "entity_refs": [company_id], "qualifier_surfaces": [],
        })
        row["scope"].update({
            "target_period_expressions": years,
            "scope_qualifier_expressions": ["연결", "별도"],
        })
        row["selection"] = None
        row["output"].update({
            "shape": "comparison", "projection_mode": "named_fields",
            "field_surfaces": [difference.group(0)],
        })
        payload["answer_items"] = [row]
        return True

    def _question_closed_scope_gap(
            self, payload: dict[str, Any], question: str,
            ) -> bool:
        """Normalize provider-shape variants of a fully literal scope gap.

        Unlike :meth:`_closed_scope_gap`, this fallback does not treat HCX's
        retrieve/compare or scalar/record choice as authority.  It runs only
        when the original question itself uniquely closes issuer, year,
        metric, both statement scopes and the non-directional operation.
        """

        items = payload.get("answer_items", [])
        if (not 1 <= len(items) <= 3 or payload.get("answer_groups")
                or payload.get("premises") or payload.get("unresolved_mentions")
                or "연결" not in question
                or not any(token in question for token in ("별도", "개별"))):
            return False
        companies = self._question_company_ids(payload, question)
        years = list(dict.fromkeys(
            match.group(0).strip() for match in self._YEAR.finditer(question)))
        metric_matches = list(self._METRIC.finditer(question))
        metric_concepts = {
            resolve_metric_concept(match.group(0)) for match in metric_matches}
        metric_concepts.discard(None)
        difference = re.search(r"(?:절대\s*)?(?:차이|격차)", question)
        if (len(companies) != 1 or len(years) != 1
                or len(metric_concepts) != 1 or not metric_matches
                or difference is None):
            return False
        # The question closing a scope gap does not license overwriting an
        # item the provider bound to a *different* accounting axis, nor one
        # carrying a row selection.  Either is a second demand rather than a
        # shape variant of this one, so the request stays as received instead
        # of silently losing it.
        concept = next(iter(metric_concepts))
        for item in items:
            if item.get("selection") is not None:
                return False
            for surface in (item.get("output", {}).get("field_surfaces") or []):
                if not isinstance(surface, str):
                    continue
                if any(resolve_metric_concept(match.group(0)) != concept
                       for match in self._METRIC.finditer(surface)):
                    return False

        row = items[0]
        target = row.setdefault("target", {})
        scope = row.setdefault("scope", {})
        output = row.setdefault("output", {})
        row.update({
            "item_id": "item-1", "operation": "compare", "selection": None,
        })
        target.update({
            "kind": "metric", "surface": metric_matches[0].group(0),
            "entity_refs": companies, "qualifier_surfaces": [],
        })
        scope.update({
            "target_period_expressions": years,
            "as_of_expression": None,
            "document_group_expression": None,
            "scope_qualifier_expressions": ["연결", "별도"],
        })
        output.update({
            "shape": "comparison", "projection_mode": "named_fields",
            "field_surfaces": [difference.group(0)], "presentation": "auto",
        })
        payload["answer_items"] = [row]
        return True

    def _flattened_scope_pair(
            self, payload: dict[str, Any], question: str,
            ) -> bool:
        """Recover a one-item CFS/SFS absolute-gap topology."""

        items = payload.get("answer_items", [])
        if (len(items) != 1 or payload.get("answer_groups")
                or payload.get("premises") or payload.get("unresolved_mentions")
                or "연결" not in question
                or not any(token in question for token in ("별도", "개별"))):
            return False
        row = items[0]
        target = row.get("target", {})
        output = row.get("output", {})
        companies = self._question_company_ids(payload, question)
        years = list(dict.fromkeys(
            match.group(0).strip() for match in self._YEAR.finditer(question)))
        metric = self._METRIC.search(question)
        difference = re.search(r"(?:절대\s*)?차이", question)
        if (len(companies) != 1 or len(years) != 1 or metric is None
                or difference is None or target.get("kind") != "metric"
                or row.get("selection") is not None
                or output.get("projection_mode") not in {
                    "named_fields", "whole_target"}):
            return False
        refs = target.get("entity_refs", [])
        if not isinstance(refs, list) or any(ref != companies[0] for ref in refs):
            return False
        anchors = [metric.group(0), difference.group(0), "차이", "얼마"]
        for surface in output.get("field_surfaces", []):
            if not isinstance(surface, str) or surface not in question:
                return False
            compact = re.sub(r"\s+", "", surface)
            if not any(
                    compact in re.sub(r"\s+", "", anchor)
                    or re.sub(r"\s+", "", anchor) in compact
                    for anchor in anchors):
                return False
        row["operation"] = "compare"
        row["target"].update({
            "surface": metric.group(0), "entity_refs": companies,
            "qualifier_surfaces": [],
        })
        row["scope"].update({
            "target_period_expressions": years,
            "scope_qualifier_expressions": ["연결", "별도"],
        })
        row["output"].update({
            "shape": "comparison", "projection_mode": "named_fields",
            "field_surfaces": [difference.group(0)],
        })
        return True

    def _annual_amount_and_rate(
            self, payload: dict[str, Any], question: str,
            ) -> bool:
        """Recover later-minus-earlier amount and prior-base rate demands."""

        items = payload.get("answer_items", [])
        if (not 1 <= len(items) <= 3 or payload.get("answer_groups")
                or payload.get("premises") or payload.get("unresolved_mentions")):
            return False
        companies = self._question_company_ids(payload, question)
        metric = self._METRIC.search(question)
        years = list(dict.fromkeys(
            match.group(0).strip() for match in self._YEAR.finditer(question)))
        relative = re.search(r"전년(?:도)?\s*(?:대비|보다)|전년비", question)
        explicit_relation = re.search(
            r"(?:20)?[0-9]{2}\s*년(?:도)?(?:\s*말)?\s*(?:대비|보다)",
            question)
        amount_surfaces = [
            match.group(0) for match in re.finditer(r"얼마나|얼마|금액", question)]
        percent_surfaces = [
            match.group(0) for match in re.finditer(r"몇\s*퍼센트|비율", question)]
        # ``2024년 대비 얼마나 변했는가`` is the same closed prior-year change
        # demand as the amount-and-rate phrasing; requiring both surfaces left
        # the amount-only wording to depend on the provider emitting the
        # comparison period itself, which it does not always do.
        if (len(companies) != 1 or metric is None
                or not (amount_surfaces or percent_surfaces)
                or len(years) not in {1, 2}
                or re.search(r"[1-4]\s*분기\s*말", question) is not None
                or (relative is None and explicit_relation is None)):
            return False

        # ``A 대비 B`` means B minus A regardless of the surface order in the
        # sentence.  The compiler assigns direction from operand order, so put
        # the compared endpoint first and the base endpoint second.  This is
        # the same convention already used by ``B는 A 대비`` wording.
        if len(years) == 2 and explicit_relation is not None:
            base_match = re.search(
                r"(?P<base>(?:20)?[0-9]{2}\s*년(?:도)?)"
                r"(?:\s*말)?\s*(?:대비|보다)", question)
            if base_match is not None:
                base_key = re.sub(r"\s+", "", base_match.group("base"))
                base = next((
                    value for value in years
                    if re.sub(r"\s+", "", value) == base_key
                ), None)
                compared = [value for value in years if value != base]
                if base is not None and len(compared) == 1:
                    years = [compared[0], base]

        anchors = [
            metric.group(0), *amount_surfaces, *percent_surfaces, *years,
            *(value.group(0) for value in (relative, explicit_relation)
              if value is not None),
        ]

        for row in items:
            target = row.get("target", {})
            output = row.get("output", {})
            refs = target.get("entity_refs", [])
            if (target.get("kind") != "metric"
                    or not isinstance(refs, list)
                    or any(ref != companies[0] for ref in refs)
                    or row.get("selection") is not None
                    or output.get("projection_mode") not in {
                        "named_fields", "whole_target"}
                    ):
                return False
            for surface in output.get("field_surfaces", []):
                if not isinstance(surface, str) or surface not in question:
                    return False
                compact = re.sub(r"\s+", "", surface)
                if not any(
                        compact in re.sub(r"\s+", "", anchor)
                        or re.sub(r"\s+", "", anchor) in compact
                        for anchor in anchors):
                    return False

        row = items[0]
        row["item_id"] = "item-1"
        row["operation"] = "compare"
        row["target"].update({
            "surface": metric.group(0), "entity_refs": companies,
            "qualifier_surfaces": (
                [relative.group(0)] if len(years) == 1 and relative is not None
                else []),
        })
        row["scope"].update({
            "target_period_expressions": years,
            "scope_qualifier_expressions": (
                ["연결"] if "연결" in question else
                ["별도"] if any(token in question for token in ("별도", "개별"))
                else []),
        })
        row["selection"] = None
        row["output"].update({
            "shape": "comparison", "projection_mode": "named_fields",
            "field_surfaces": [
                surface for surface in
                (amount_surfaces[-1] if amount_surfaces else None,
                 percent_surfaces[-1] if percent_surfaces else None)
                if surface is not None],
        })
        payload["answer_items"] = [row]
        return True

    def _scope_pair_values(
            self, payload: dict[str, Any], question: str,
            ) -> bool:
        """Recover two independent CFS/SFS values, without inventing a gap.

        A question such as ``연결 기준과 별도 기준으로 나눠서`` asks for two
        scalar facts, not an absolute difference.  Only a single company,
        annual period and canonical metric with both literal scope roles may
        enter this repair.  Any arithmetic or unrelated provider field keeps
        the request fail-closed.
        """

        items = payload.get("answer_items", [])
        if (not 1 <= len(items) <= 2 or payload.get("answer_groups")
                or payload.get("premises") or payload.get("unresolved_mentions")
                or "연결" not in question
                or not any(token in question for token in ("별도", "개별"))
                or re.search(
                    r"차이|격차|증감|변화|변했|비교|더\s*(?:큰|작은|많|적)",
                    question)
                or re.search(r"나눠서|각각|각\s*기준", question) is None):
            return False
        companies = self._question_company_ids(payload, question)
        years = list(dict.fromkeys(
            match.group(0).strip() for match in self._YEAR.finditer(question)))
        metric_matches = list(self._METRIC.finditer(question))
        metric_concepts = {
            resolve_metric_concept(match.group(0)) for match in metric_matches}
        metric_concepts.discard(None)
        if (len(companies) != 1 or len(years) != 1
                or len(metric_concepts) != 1 or not metric_matches):
            return False
        concept = next(iter(metric_concepts))
        company_id = companies[0]
        allowed_scopes = {"연결", "별도", "개별"}
        for row in items:
            target = row.get("target", {})
            scope = row.get("scope", {})
            output = row.get("output", {})
            target_match = self._METRIC.search(str(target.get("surface", "")))
            refs = target.get("entity_refs", [])
            if (target.get("kind") != "metric"
                    or target_match is None
                    or resolve_metric_concept(target_match.group(0)) != concept
                    or row.get("operation") not in {"retrieve", "compare"}
                    or not isinstance(refs, list) or not refs
                    or any(ref != company_id for ref in refs)
                    or row.get("selection") is not None
                    or scope.get("as_of_expression") is not None
                    or scope.get("document_group_expression") is not None
                    or any(value not in question
                           for value in scope.get(
                               "target_period_expressions", []))
                    or any(value not in allowed_scopes or value not in question
                           for value in scope.get(
                               "scope_qualifier_expressions", []))
                    or output.get("projection_mode") not in {
                        "named_fields", "whole_target"}
                    or any(not isinstance(value, str) or value not in question
                           for value in output.get("field_surfaces", []))):
                return False

        metric_surface = max(
            metric_matches,
            key=lambda match: len(normalize_surface_key(match.group(0))),
        ).group(0)
        cfs_surface = (re.search(r"연결\s*기준", question) or
                       re.search(r"연결", question))
        sfs_surface = (re.search(r"(?:별도|개별)\s*기준", question) or
                       re.search(r"별도|개별", question))
        if cfs_surface is None or sfs_surface is None:
            return False
        template = deepcopy(items[0])
        rebuilt = []
        for index, (scope_value, field_surface) in enumerate((
                ("연결", cfs_surface.group(0)),
                ("별도", sfs_surface.group(0))), start=1):
            row = deepcopy(template)
            row.update({
                "item_id": f"item-{index}", "operation": "retrieve",
                "selection": None,
            })
            row["target"].update({
                "kind": "metric", "surface": metric_surface,
                "entity_refs": [company_id], "qualifier_surfaces": [],
            })
            row["scope"].update({
                "target_period_expressions": years,
                "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [scope_value],
            })
            row["output"].update({
                "shape": "scalar", "projection_mode": "named_fields",
                "field_surfaces": [field_surface],
            })
            rebuilt.append(row)
        payload["answer_items"] = rebuilt
        return True

    def _scope_pair(self, payload: dict[str, Any], question: str) -> bool:
        items = payload.get("answer_items", [])
        if len(items) != 2 or payload.get("answer_groups"):
            return False
        first, second = items
        if not all(
                item.get("operation") == "retrieve"
                and item.get("target", {}).get("kind") == "metric"
                and item.get("output", {}).get("shape") == "scalar"
                for item in items):
            return False
        if (
                first["target"]["surface"] != second["target"]["surface"]
                or first["target"]["entity_refs"] != second["target"]["entity_refs"]
                or first["scope"]["target_period_expressions"]
                != second["scope"]["target_period_expressions"]
        ):
            return False
        qualifiers = [
            item["scope"]["scope_qualifier_expressions"] for item in items]
        compact = [{value.replace(" ", "") for value in values}
                   for values in qualifiers]
        if not ({"연결"} in compact and ({"별도"} in compact or {"개별"} in compact)):
            return False
        difference = re.search(r"차이", question)
        if difference is None:
            return False
        periods = first["scope"]["target_period_expressions"]
        if not periods:
            periods = list(dict.fromkeys(
                match.group(0).strip()
                for match in self._YEAR.finditer(question)))
            if len(periods) != 1:
                return False
        merged = first
        merged["operation"] = "compare"
        merged["scope"]["target_period_expressions"] = periods
        merged["scope"]["scope_qualifier_expressions"] = ["연결", "별도"]
        merged["output"].update({
            "shape": "comparison",
            "field_surfaces": [difference.group(0)],
        })
        payload["answer_items"] = [merged]
        return True

    def _flattened_yoy(self, payload: dict[str, Any], question: str) -> bool:
        items = payload.get("answer_items", [])
        if len(items) != 1 or payload.get("answer_groups"):
            return False
        item = items[0]
        if (
                item.get("target", {}).get("kind") != "metric"
                or item.get("operation") != "retrieve"
                or item.get("output", {}).get("shape") != "scalar"
                or item.get("scope", {}).get("target_period_expressions")
        ):
            return False
        metric = self._METRIC.search(question)
        year = self._YEAR.search(question)
        yoy = self._YOY.search(question)
        if metric is None or year is None or yoy is None:
            return False
        company_refs = item["target"].get("entity_refs", [])
        if len(company_refs) != 1:
            return False
        item["target"].update({
            "surface": metric.group(0),
            "qualifier_surfaces": ["전년비"],
        })
        item["operation"] = "compare"
        item["scope"].update({
            "target_period_expressions": [year.group(0).strip()],
            "scope_qualifier_expressions": (
                ["연결"] if "연결" in question else
                ["별도"] if "별도" in question else []),
        })
        item["output"].update({
            "shape": "comparison",
            "field_surfaces": [yoy.group(0)],
        })
        return True

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        payload = intent.model_dump(mode="python", warnings=False)
        changed = (
            self._closed_scope_gap(payload, question)
            or self._question_closed_scope_gap(payload, question)
            or self._scope_pair_values(payload, question)
            or self._annual_amount_and_rate(payload, question)
            or self._flattened_yoy(payload, question)
        )
        return SemanticIntent.model_validate(payload, strict=True) if changed else intent


class ClosedLiteralQuestionIntentRegrounder:
    """Prefer a fully closed literal question over a bad-but-valid HCX shape.

    Structured output proves the JSON contract, not that the model selected the
    right semantic topology. Four common disclosure questions already contain
    every coordinate needed to build that topology: one exact-dated event with
    two named fields, a two-company winner plus gap, a two-year amount change,
    and two independent CFS/SFS values. Rebuild only those complete grammars
    from canonical company identities and literal question spans.

    This is deliberately not an answer fixture: it never reads a question id,
    receipt, expected value, or company-specific rule. Any missing/ambiguous
    company, extra period, unknown metric, open-ended wording, or different
    operation leaves the provider intent untouched and therefore fail-closed.
    """

    _OPEN = re.compile(r"만약|가정|예측|예상|전망|추천|투자\s*(?:의견|조언)")
    _EXACT_CORRECTION = re.compile(
        r"(?:접수\s*번호\s*)?(?P<receipt>20[0-9]{12})\s*"
        r"(?P<family>[^?\n]{2,80}?)\s*(?:기재\s*)?정정\s*공시"
        r"(?:에서|의)?[^?\n]{0,80}(?:변경된\s*내용|바뀐\s*내용|무엇이\s*바뀌)")
    _GENERIC_EXACT_CORRECTION = re.compile(
        r"(?P<coordinate>(?:접수\s*번호\s*)?20[0-9]{12}|"
        r"[0-9]{4}\s*년\s*[0-9]{1,2}\s*월\s*[0-9]{1,2}\s*일)\s*"
        r"(?P<surface>(?:기재\s*)?정정\s*공시)"
        r"(?:에서|의)?[^?\n]{0,100}(?:변경|바뀌|달라)",
    )
    _QUOTED_CONTRACT_HISTORY = re.compile(
        r"['‘’\"](?P<target>[^'‘’\"]{2,300})['‘’\"]\s*계약의\s*"
        r"(?P<root>최초\s*(?:공시|체결(?:\s*내용)?))\s*(?:과|와)\s*"
        r"(?P<history>(?:전체\s*)?(?:변경|정정)\s*(?:이력|내역|흐름))")

    def __init__(self, canonical: Any, company_preflight: Any) -> None:
        if not callable(getattr(
                company_preflight, "question_company_surface", None)):
            raise TypeError("closed literal regrounder company preflight가 잘못되었습니다")
        self.canonical = canonical
        self.company_preflight = company_preflight

    def _companies(self, question: str) -> list[tuple[str, str]]:
        """Return unique canonical companies with literal question spans."""

        scan = getattr(self.canonical, "companies_in_text", None)
        if not callable(scan):
            return []
        rows = scan(question)
        found: dict[str, str] = {}
        for row in rows or ():
            code = getattr(row, "corp_code", None)
            name = getattr(row, "corp_name", None)
            if not isinstance(code, str) or not isinstance(name, str):
                continue
            surface = self.company_preflight.question_company_surface(
                name, question)
            if surface is not None:
                found[code] = surface
        return sorted(found.items(), key=lambda row: question.find(row[1]))

    @staticmethod
    def _scope(question: str) -> list[str]:
        if "연결" in question:
            return ["연결"]
        if "별도" in question:
            return ["별도"]
        if "개별" in question:
            return ["개별"]
        return []

    @staticmethod
    def _intent(
            companies: list[tuple[str, str]],
            answer_items: list[dict[str, Any]],
            ) -> SemanticIntent:
        entities = [
            {
                "entity_id": f"entity-{index}",
                "kind_hint": "company",
                "surface": surface,
            }
            for index, (_code, surface) in enumerate(companies, start=1)
        ]
        return SemanticIntent.model_validate({
            "schema_version": "stage1-semantic-intent/1.1",
            "entities": entities,
            "answer_items": answer_items,
            "answer_groups": [],
            "premises": [],
            "unresolved_mentions": [],
            "presentation": "auto",
        }, strict=True)

    @staticmethod
    def _item(
            *, item_id: str, operation: str, kind: str, surface: str,
            entity_refs: list[str], periods: list[str], scopes: list[str],
            shape: str, fields: list[str],
            selection: dict[str, Any] | None = None,
            qualifiers: list[str] | None = None,
            ) -> dict[str, Any]:
        return {
            "item_id": item_id,
            "operation": operation,
            "target": {
                "kind": kind,
                "surface": surface,
                "entity_refs": entity_refs,
                "qualifier_surfaces": qualifiers or [],
            },
            "scope": {
                "target_period_expressions": periods,
                "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": scopes,
            },
            "selection": selection,
            "output": {
                "shape": shape,
                "projection_mode": "named_fields",
                "field_surfaces": fields,
                "presentation": "auto",
            },
        }

    def _exact_event(
            self, question: str, companies: list[tuple[str, str]],
            ) -> SemanticIntent | None:
        from agent.stage1_v1_exact_event_regrounder import (
            exact_dated_event_named_fields,
        )

        axes = exact_dated_event_named_fields(question)
        if axes is None or len(companies) != 1:
            return None
        filing_date, family, fields = axes
        return self._intent(companies, [self._item(
            item_id="item-1", operation="retrieve", kind="event",
            surface=family, entity_refs=["entity-1"], periods=[], scopes=[],
            qualifiers=[filing_date], shape="record", fields=list(fields),
        )])

    def _exact_correction(
            self, question: str, companies: list[tuple[str, str]],
            ) -> SemanticIntent | None:
        """Rebuild one receipt-selected correction-diff request.

        The receipt, disclosure family and broad change demand are all public
        question literals.  Values and lineage identity remain the correction
        backend's responsibility.  Rebuilding this closed topology prevents a
        valid-but-wrong provider sample from treating changed share counts as
        an unsupported arithmetic target.
        """

        match = self._EXACT_CORRECTION.search(question)
        generic_match = (
            self._GENERIC_EXACT_CORRECTION.search(question)
            if match is None else None
        )
        if ((match is None and generic_match is None)
                or len(companies) != 1 or self._OPEN.search(question)):
            return None
        family = (
            match.group("family").strip()
            if match is not None
            else generic_match.group("surface").strip()
        )
        if not family or re.search(r"최초|이후|부터|보고서", family):
            return None
        coordinate = (
            match.group("receipt")
            if match is not None
            else generic_match.group("coordinate")
        )
        return self._intent(companies, [{
            "item_id": "item-1", "operation": "retrieve",
            "target": {
                "kind": "document", "surface": family,
                "entity_refs": ["entity-1"], "qualifier_surfaces": [],
            },
            "scope": {
                "target_period_expressions": [coordinate],
                "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": "narrative", "projection_mode": "whole_target",
                "field_surfaces": [], "presentation": "auto",
            },
        }])

    def _named_contract_history(
            self, question: str, companies: list[tuple[str, str]],
            ) -> SemanticIntent | None:
        """Preserve a quoted contract's root-content and whole-history roles.

        This is a selected lifecycle, not two form-field names.  A timeline
        shape lets canonical event resolution enumerate observations and, when
        the root predates the corpus, attach the existing typed limitation.
        """

        match = self._QUOTED_CONTRACT_HISTORY.search(question)
        if match is None or len(companies) != 1 or self._OPEN.search(question):
            return None
        return self._intent(companies, [self._item(
            item_id="item-1", operation="retrieve", kind="document",
            surface=match.group("target").strip(), entity_refs=["entity-1"],
            periods=[], scopes=[], shape="timeline",
            fields=[match.group("root"), match.group("history")],
        )])

    def _winner_gap(
            self, question: str, companies: list[tuple[str, str]],
            ) -> SemanticIntent | None:
        periods = list(dict.fromkeys(
            match.group(0).strip()
            for match in ClosedFinancialComparisonRegrounder._PERIOD.finditer(
                question)))
        metrics = list(ClosedFinancialComparisonRegrounder._METRIC.finditer(
            question))
        concepts = {resolve_metric_concept(match.group(0)) for match in metrics}
        concepts.discard(None)
        winner_direction = ClosedFinancialComparisonRegrounder._winner_direction(
            question)
        gap = re.search(r"(?:절대\s*)?차이", question)
        if (len(companies) != 2 or len(periods) != 1 or len(concepts) != 1
                or not metrics or winner_direction is None or gap is None
                or self._OPEN.search(question)):
            return None
        winner, mode = winner_direction
        metric = max(
            metrics,
            key=lambda match: len(normalize_surface_key(match.group(0))),
        ).group(0)
        return self._intent(companies, [self._item(
            item_id="item-1", operation="compare", kind="metric",
            surface=metric, entity_refs=["entity-1", "entity-2"],
            periods=periods, scopes=self._scope(question),
            selection={
                "mode": mode, "criterion_surface": winner.group(0),
                "k": None,
            },
            shape="comparison", fields=[winner.group(0), gap.group(0)],
        )])

    def _amount_change(
            self, question: str, companies: list[tuple[str, str]],
            ) -> SemanticIntent | None:
        years = list(dict.fromkeys(
            match.group(0).strip()
            for match in ClosedFinancialRelationRegrounder._YEAR.finditer(
                question)))
        metrics = list(ClosedFinancialRelationRegrounder._METRIC.finditer(
            question))
        concepts = {resolve_metric_concept(match.group(0)) for match in metrics}
        concepts.discard(None)
        relation = re.search(
            r"(?P<base>(?:20)?[0-9]{2}\s*년(?:도)?)"
            r"(?:\s*말)?\s*(?:대비|보다)", question)
        amount = re.search(r"얼마나|얼마|금액|증감\s*액|변동\s*액|변화\s*액", question)
        percent = re.search(r"몇\s*퍼센트|비율|증감\s*률|변동\s*률|변화\s*율", question)
        if (len(companies) != 1 or len(years) != 2 or len(concepts) != 1
                or not metrics or relation is None or amount is None
                or percent is not None or self._OPEN.search(question)):
            return None
        base_key = normalize_surface_key(relation.group("base"))
        base = next((year for year in years
                     if normalize_surface_key(year) == base_key), None)
        compared = [year for year in years if year != base]
        if base is None or len(compared) != 1:
            return None
        metric = max(
            metrics,
            key=lambda match: len(normalize_surface_key(match.group(0))),
        ).group(0)
        return self._intent(companies, [self._item(
            item_id="item-1", operation="compare", kind="metric",
            surface=metric, entity_refs=["entity-1"],
            periods=[compared[0], base], scopes=self._scope(question),
            shape="comparison", fields=[amount.group(0)],
        )])

    def _scope_values(
            self, question: str, companies: list[tuple[str, str]],
            ) -> SemanticIntent | None:
        years = list(dict.fromkeys(
            match.group(0).strip()
            for match in ClosedFinancialRelationRegrounder._YEAR.finditer(
                question)))
        metrics = list(ClosedFinancialRelationRegrounder._METRIC.finditer(
            question))
        concepts = {resolve_metric_concept(match.group(0)) for match in metrics}
        concepts.discard(None)
        cfs = re.search(r"연결\s*기준|연결", question)
        sfs = re.search(r"(?:별도|개별)\s*기준|별도|개별", question)
        if (len(companies) != 1 or len(years) != 1 or len(concepts) != 1
                or not metrics or cfs is None or sfs is None
                or re.search(r"나눠서|각각|각\s*기준", question) is None
                or re.search(
                    r"차이|격차|증감|변화|비교|더\s*(?:큰|작은|많|적)",
                    question)
                or self._OPEN.search(question)):
            return None
        metric = max(
            metrics,
            key=lambda match: len(normalize_surface_key(match.group(0))),
        ).group(0)
        items = [
            self._item(
                item_id="item-1", operation="retrieve", kind="metric",
                surface=metric, entity_refs=["entity-1"], periods=years,
                scopes=["연결"], shape="scalar", fields=[cfs.group(0)]),
            self._item(
                item_id="item-2", operation="retrieve", kind="metric",
                surface=metric, entity_refs=["entity-1"], periods=years,
                scopes=["별도"], shape="scalar", fields=[sfs.group(0)]),
        ]
        return self._intent(companies, items)

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if not isinstance(question, str) or not question.strip():
            return intent
        companies = self._companies(question)
        rebuilt = (
            self._exact_correction(question, companies)
            or self._named_contract_history(question, companies)
            or self._exact_event(question, companies)
            or self._winner_gap(question, companies)
            or self._amount_change(question, companies)
            or self._scope_values(question, companies)
        )
        return rebuilt if rebuilt is not None else intent


class CompositeSourceIntentRegrounder:
    """Apply independent question-grounded regrounders in stable order."""

    def __init__(self, *regrounders: Any) -> None:
        self.regrounders = regrounders

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        current = intent
        for regrounder in self.regrounders:
            current = regrounder(question, current)
            if not isinstance(current, SemanticIntent):
                raise TypeError("source intent regrounder 반환값이 잘못되었습니다")
        return current


class InvestmentPlanIntentRegrounder:
    """Recover the closed periodic investment-table request topology.

    HCX may describe a quoted investment row as a non-company entity or emit
    an aggregate operation which the frozen v0.4 plan cannot represent.  The
    existing periodic resolver already proves the exact report and investment
    table headers.  This adapter only restores that executable subset when the
    question itself uniquely supplies one issuer, one quarterly report, and
    the investment-table vocabulary.  It never performs the requested sum.
    """

    # PeriodicDocumentPreflight의 공용 문법과 같은 표면을 받는다.  사용자는
    # 「2026년 1분기」뿐 아니라 「26년 1Q」도 흔히 쓰므로, 여기서만 더
    # 좁게 받으면 HCX 표본에 따라 동일 질의가 unsupported로 퇴행한다.
    _PERIOD = re.compile(
        r"(?:20[0-9]{2}|[0-9]{2})년\s*[13]\s*(?:분기|Q)(?:보고서)?",
        flags=re.IGNORECASE,
    )
    _QUOTED = re.compile(r"['\"](?P<value>[^'\"]{2,200})['\"]")
    _INVESTMENT = re.compile(
        r"투자\s*계획|설비\s*투자|투자.{0,24}(?:소요자금|지출금액|계획금액)")
    _WHOLE_TABLE = re.compile(
        r"투자.{0,16}(?:뭐|무엇|어떤).{0,16}(?:한다|하는지|했|계획)"
        r"|(?:뭐|무엇|어떤).{0,16}투자")
    _AGGREGATE = re.compile(r"전부\s*합산|모두\s*합산|총액")

    def __init__(self, company_preflight: Any) -> None:
        if not callable(getattr(
                company_preflight, "unique_question_company_surface", None)):
            raise TypeError("투자계획 regrounder company preflight 계약이 잘못되었습니다")
        self.company_preflight = company_preflight

    @staticmethod
    def _first_surface(question: str, values: tuple[str, ...]) -> str | None:
        found = [value for value in values if value in question]
        found.sort(key=question.index)
        return found[0] if found else None

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str)
                or (self._INVESTMENT.search(question) is None
                    and self._WHOLE_TABLE.search(question) is None)
                or re.search(r"현금흐름표|재무제표", question)):
            return intent
        periods = list(dict.fromkeys(
            match.group(0) for match in self._PERIOD.finditer(question)))
        company_surface = self.company_preflight.unique_question_company_surface(
            question)
        if len(periods) != 1 or company_surface is None:
            return intent

        quoted = list(self._QUOTED.finditer(question))
        aggregate = self._AGGREGATE.search(question) is not None
        if len(quoted) == 1 and not aggregate:
            target_surface = quoted[0].group("value").strip()
            fields = [surface for surface in (
                self._first_surface(question, ("총 소요자금", "총소요자금")),
                self._first_surface(question, ("기 지출금액", "기지출금액", "기 지출액")),
            ) if surface is not None]
            if len(fields) != 2 or target_surface not in question:
                return intent
            output_shape = "record"
        elif aggregate:
            target_surface = self._first_surface(
                question, ("투자계획", "투자 계획", "설비투자"))
            field = self._first_surface(
                question, ("계획금액", "계획 금액", "총 소요자금", "총소요자금"))
            if target_surface is None or field is None:
                return intent
            fields = [field]
            output_shape = "record_list"
        elif self._WHOLE_TABLE.search(question) is not None:
            # 표 전체를 묻는 구어형은 특정 행을 추정하지 않는다. 공시 표가
            # 증명하는 네 역할만 요청하고 모든 행을 그대로 보존한다. target
            # 표면은 grounding gate를 통과하도록 질문에 실제 존재하는 말을
            # 보존하며, 이를 곧 특정 행 이름으로 승격하지 않는다.
            target_surface = self._first_surface(
                question, ("투자 계획", "투자계획", "설비투자", "투자"))
            if target_surface is None:  # regex와 같은 근거를 쓰는 닫힌 guard
                return intent
            # 공개 intent의 모든 표면은 원 질문에 그대로 결속되어야 한다.
            # 네 canonical 역할은 이후 section-slot proof가 증명해 sidecar로
            # 전달하므로, 여기서는 질문에 실제 있는 「투자」만 보존한다.
            fields = [target_surface]
            output_shape = "record_list"
        else:
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
                    "shape": output_shape,
                    "projection_mode": "named_fields",
                    "field_surfaces": fields, "presentation": "auto",
                },
            }],
            "answer_groups": [], "premises": [],
            "unresolved_mentions": [], "presentation": "auto",
        }, strict=True)


class SameDayPeriodicVersionIntentRegrounder:
    """Recover one periodic-report version-history request.

    Logical lineage and intraday receipt time are separate facts.  The former
    is executable through the existing document-version authority; the latter
    is attached later as a typed limitation.  This adapter is entered only for
    an explicit annual report, one full correction date, a plural correction
    count, and an ordering/time request.
    """

    _REPORT = re.compile(
        r"20[0-9]{2}년\s*(?:1\s*분기보고서|반기보고서|"
        r"3\s*분기보고서|사업보고서)")
    _PLAIN_ANNUAL = re.compile(r"사업보고서")
    _REQUEST = re.compile(r"기재\s*정정|정정\s*공시")
    _PLURAL = re.compile(r"두\s*건|복수")
    _ORDER = re.compile(r"먼저|선후|순서|접수\s*시각|접수시각")
    _VERSION = re.compile(r"변경\s*이력|버전|원본과.{0,30}정정본")

    def __init__(self, company_preflight: Any) -> None:
        self.company_preflight = company_preflight

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if not isinstance(question, str):
            return intent
        reports = list(dict.fromkeys(
            match.group(0) for match in self._REPORT.finditer(question)))
        full_days = {
            parts for parts in question_date_surfaces(question)
            if parts[1] is not None and parts[2] is not None
        }
        company_surface = self.company_preflight.unique_question_company_surface(
            question)
        same_day_clock = bool(
            len(reports) == 1 and len(full_days) == 1
            and self._REQUEST.search(question) is not None
            and self._PLURAL.search(question) is not None
            and self._ORDER.search(question) is not None)
        version = self._VERSION.search(question)
        plain_annual = (
            reports == [] and len(full_days) == 1
            and self._PLAIN_ANNUAL.search(question) is not None
            and version is not None)
        if (company_surface is None
                or not (same_day_clock or version is not None)
                or (len(reports) != 1 and not plain_annual)):
            return intent
        field = version.group(0) if version is not None else "정정"
        premise = re.search(
            r"기재\s*정정\s*공시\s*두\s*건", question)
        premise_surface = premise.group(0) if premise is not None else field
        report_surface = (
            reports[0] if reports else self._PLAIN_ANNUAL.search(question).group(0))
        return SemanticIntent.model_validate({
            "schema_version": intent.schema_version,
            "entities": [{
                "entity_id": "entity-1", "kind_hint": "company",
                "surface": company_surface,
            }],
            "answer_items": [{
                "item_id": "item-1", "operation": "retrieve",
                "target": {
                    "kind": "document", "surface": report_surface,
                    "entity_refs": ["entity-1"], "qualifier_surfaces": [],
                },
                "scope": {
                    "target_period_expressions": [],
                    "as_of_expression": None,
                    "document_group_expression": None,
                    "scope_qualifier_expressions": [],
                },
                "selection": None,
                "output": {
                    "shape": "scalar", "projection_mode": "named_fields",
                    "field_surfaces": [field], "presentation": "auto",
                },
            }],
            "answer_groups": [],
            "premises": [{
                "premise_id": "premise-1", "kind": "existence",
                "raw_text": premise_surface,
                "applies_to_item_ids": ["item-1"],
            }],
            "unresolved_mentions": [], "presentation": "auto",
        }, strict=True)


class MixedSupportedFinancialRegrounder:
    """Keep the executable financial subset of an explicit mixed request.

    This handles two generic contracts: a forecast with an explicit actual
    fallback, and a comparison where exactly one issuer is in the provided
    universe.  It never fabricates a comparison result from one operand.
    """

    _FUTURE = re.compile(r"예측|예상|전망")
    _FALLBACK = re.compile(r"안\s*되면|대신|불가능.{0,12}(?:실제|과거)")
    _LATEST_ACTUAL = re.compile(r"최신\s*연간\s*실제|최신\s*실제|과거\s*실적")

    def __init__(self, canonical: Any, *, corpus_cutoff: str) -> None:
        self.canonical = canonical
        self.corpus_cutoff = corpus_cutoff

    def _known_company_refs(
            self, intent: SemanticIntent,
            ) -> tuple[list[str], list[str]]:
        known, absent = [], []
        for entity in intent.entities:
            if entity.kind_hint != "company":
                continue
            rows = self.canonical.resolve_company(entity.surface)
            held = self.canonical.held_company_candidates(entity.surface)
            if len(rows) == 1:
                known.append(entity.entity_id)
            elif not rows and not held:
                absent.append(entity.entity_id)
        return known, absent

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if not isinstance(question, str) or not intent.answer_items:
            return intent
        payload = intent.model_dump(mode="python", warnings=False)

        if (self._FUTURE.search(question) and self._FALLBACK.search(question)
                and self._LATEST_ACTUAL.search(question)):
            metric_items = [item for item in intent.answer_items
                            if item.target.kind == "metric"]
            if metric_items:
                base = next((item for item in metric_items
                             if not any(re.search(r"(?:19|20)[0-9]{2}", value)
                                        for value in item.scope.target_period_expressions)),
                            metric_items[-1])
                row = base.model_dump(mode="python", warnings=False)
                row["item_id"] = "item-1"
                row["operation"] = "retrieve"
                row["scope"]["target_period_expressions"] = [
                    self._LATEST_ACTUAL.search(question).group(0)]
                row["selection"] = None
                row["output"].update({
                    "shape": "scalar", "projection_mode": "named_fields",
                    "field_surfaces": [base.target.surface],
                })
                payload.update({
                    "answer_items": [row], "answer_groups": [],
                    "premises": [], "unresolved_mentions": [],
                })
                return SemanticIntent.model_validate(payload, strict=True)

        known, absent = self._known_company_refs(intent)
        # Only a *comparison* licenses dropping the out-of-universe operand.
        # A company named as a contract counterparty (``LG에너지솔루션의 Ford
        # 배터리 공급계약``) is an identity that narrows the request, not a
        # second operand; discarding it silently widens the answer to every
        # contract of the issuer.
        compares = (
            any(item.operation == "compare" for item in intent.answer_items)
            or re.search(r"중\s*(?:어느|누가|어디)|보다|더\s*(?:큰|많|높)|비교|차이",
                         question) is not None)
        if len(known) == 1 and absent and compares and all(
                item.target.kind == "metric" for item in intent.answer_items):
            rows = []
            for item in payload["answer_items"]:
                refs = item["target"]["entity_refs"]
                if known[0] not in refs:
                    continue
                item["target"]["entity_refs"] = [known[0]]
                item["operation"] = "retrieve"
                item["selection"] = None
                item["output"].update({
                    "shape": "scalar", "projection_mode": "named_fields",
                    "field_surfaces": [item["target"]["surface"]],
                })
                rows.append(item)
            if rows:
                rows[0]["item_id"] = "item-1"
                referenced = set(rows[0]["target"]["entity_refs"])
                payload.update({
                    "entities": [
                        entity for entity in payload["entities"]
                        if entity["entity_id"] in referenced
                    ],
                    "answer_items": [rows[0]], "answer_groups": [],
                    "premises": [], "unresolved_mentions": [],
                })
                return SemanticIntent.model_validate(payload, strict=True)
        return intent


class RootMissingCorrectionDocumentRegrounder:
    """Preserve the visible correction when the requested root predates corpus.

    The user asks to restore a missing original, but the safe executable
    subset is the correction filing that is actually present.  This adapter
    creates no root/event identity and copies the correction-family surface
    from the question.
    """

    _REQUEST = re.compile(
        r"(?P<surface>[0-9A-Za-z가-힣()·ㆍ\s]{1,40}?정정\s*공시)"
        r"(?:의|에서|을|를|\s)")

    def __init__(self, canonical: Any) -> None:
        self.canonical = canonical

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (not isinstance(question, str)
                or not re.search(r"코퍼스\s*(?:이전|밖)", question)
                or not re.search(r"원\s*공시|원공시|최초", question)):
            return intent
        companies = [
            entity for entity in intent.entities
            if entity.kind_hint == "company"
            and entity.surface in question
            and len(self.canonical.resolve_company(entity.surface)) == 1
        ]
        match = self._REQUEST.search(question)
        if len(companies) != 1 or match is None:
            return intent
        surface = match.group("surface").strip()
        # Drop a leading date/verb fragment while retaining the literal
        # disclosure-family phrase itself.
        family = re.search(r"[0-9A-Za-z가-힣()·ㆍ]+\s*정정\s*공시$", surface)
        if family is not None:
            surface = family.group(0).strip()
        company = companies[0].model_copy(update={"entity_id": "entity-1"})
        from agent.semantic_intent_v1 import AnswerItem, OutputRequest, Scope, Target
        item = AnswerItem(
            item_id="item-1",
            target=Target(kind="document", surface=surface,
                          entity_refs=["entity-1"], qualifier_surfaces=[]),
            operation="retrieve", scope=Scope(), selection=None,
            output=OutputRequest(
                shape="narrative", projection_mode="whole_target",
                field_surfaces=[], presentation="auto"),
        )
        return SemanticIntent(
            schema_version=intent.schema_version, entities=[company],
            answer_items=[item], answer_groups=[], premises=[],
            unresolved_mentions=[], presentation="auto")


class RootMissingCorrectionDocumentBackend:
    """Select only the visible correction receipt from a root-missing proof."""

    def __init__(self, lineage_backend: Any, canonical: Any, **shared: Any) -> None:
        self.lineage_backend = lineage_backend
        self.canonical = canonical
        self.shared = shared

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> Mapping[str, Any] | None:
        from agent.stage1_v1_correction_backend import (
            RootMissingCorrectionIntermediate,
        )
        intermediate = self.lineage_backend.build_payload(
            question=question, source_intent=source_intent,
            cutoff=self.shared["corpus_cutoff"])
        if not isinstance(intermediate, RootMissingCorrectionIntermediate):
            return None
        if (len(source_intent.answer_items) != 1
                or source_intent.answer_items[0].output.projection_mode
                != "whole_target"):
            return None
        docs = [
            row for row in self.canonical.documents(
                as_of=self.shared["corpus_cutoff"],
                corp_code=intermediate.issuer_corp_code)
            if str(getattr(row, "rcept_no", ""))
            == intermediate.correction_receipt
        ]
        if len(docs) != 1:
            return None
        doc = docs[0]
        from agent.deterministic_plan_compiler_v1 import AuthoritativeResolution
        item = source_intent.answer_items[0]
        authority = AuthoritativeResolution.create(
            question_id=question_id,
            source_intent_digest=semantic_intent_digest(source_intent),
            canonical_build_id=self.shared["canonical_build_id"],
            resolver_version=self.shared["resolver_version"],
            reference_date=self.shared["reference_date"],
            corpus_cutoff=self.shared["corpus_cutoff"],
            items=[{
                "item_id": item.item_id,
                "target_surface": item.target.surface,
                "projection_mode": "whole_target",
                "resolution": {
                    "kind": "document_collection",
                    "corp_code": intermediate.issuer_corp_code,
                    "corp_name": intermediate.issuer_corp_name,
                    "as_of": self.shared["corpus_cutoff"],
                    "selector_proof_ref": intermediate.relation_proof_ref,
                    "selected_document_id": str(getattr(doc, "doc_id", "")),
                    "selected_receipt_no": intermediate.correction_receipt,
                    "selected_document_proof": {
                        "source_receipt": intermediate.correction_receipt,
                        "proof_ref": intermediate.relation_proof_ref,
                    },
                    "retrieval_query": None, "recent_selection": False,
                },
                "field_proofs": [], "applied_defaults": [],
            }], premise_proofs=[])
        return {"kind": "resolved",
                "resolution": authority.model_dump(mode="json")}


class HeldCompanyClarificationBackend:
    """Turn a registry-held group alias into a canonical entity choice.

    ``resolve_company`` intentionally returns no row for aliases such as a
    group name.  The companion ``held_company_candidates`` list is the
    authority for asking, never for automatically selecting one issuer.
    """

    def __init__(self, canonical: Any) -> None:
        finder = getattr(canonical, "held_company_candidates", None)
        if not callable(finder):
            raise TypeError("held company clarification에는 후보 registry가 필요합니다")
        self.canonical = canonical

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> ClarificationAuthority | None:
        del question_id
        referenced = {
            ref for item in source_intent.answer_items
            for ref in item.target.entity_refs
        }
        held: list[tuple[Any, list[Any], list[str]]] = []
        for entity in source_intent.entities:
            if (entity.entity_id not in referenced
                    or entity.kind_hint != "company"
                    or entity.surface not in question
                    or self.canonical.resolve_company(entity.surface)):
                continue
            candidates = list(self.canonical.held_company_candidates(
                entity.surface))
            unique = {row.corp_code: row for row in candidates}
            if len(unique) < 2:
                continue
            item_ids = [
                item.item_id for item in source_intent.answer_items
                if entity.entity_id in item.target.entity_refs
            ]
            held.append((entity, list(unique.values()), item_ids))
        if not held:
            return None
        slots = []
        for index, (entity, candidates, item_ids) in enumerate(held, start=1):
            slots.append(ClarificationSlot(
                slot_id=f"slot-{index}", role_hint="entity",
                reason_code="company_surface_multiple_candidates",
                response_kind="select_one",
                prompt=f"'{entity.surface}' 중 어느 회사를 뜻하나요?",
                applies_to_item_ids=item_ids, mention_ids=[],
                options=[ClarificationOption(
                    value=row.corp_code, label=row.corp_name,
                    proof_refs=[f"canonical:company:{row.corp_code}"],
                ) for row in candidates],
            ))
        return ClarificationAuthority(slots=slots)


class EventClarificationResolutionBackend:
    """후보가 여럿인 사건 조회를 v1 역질문 권위로 닫는다.

    후보가 하나로 좁혀지거나 아예 없으면 물러난다.  **여럿일 때만** 역질문이
    성립하기 때문이다.  선택지는 canonical 근거에서 만들고 질문 원문을 읽지
    않는다.
    """

    def __init__(
            self,
            canonical: EventMetadataCanonical,
            *,
            as_of: str,
            reference_date: date,
            corpus_cutoff: str | None = None,
            event_preflight: Any | None = None,
            ) -> None:
        self.canonical = canonical
        self.as_of = as_of
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff
        self.event_preflight = event_preflight
        if (event_preflight is not None and not callable(
                getattr(event_preflight, "resolve_event_key", None))):
            raise TypeError("event clarification preflight 계약이 잘못되었습니다")

    def _event_window(
            self, source_intent: SemanticIntent,
            ) -> tuple[str | None, str | None]:
        """의미가 든 대상기간을 사건 조회 범위로 내린다.

        기간을 버리면 후보가 넓어지고, 넓어진 후보 때문에 **답할 수 있는
        질문에도 되묻게 된다.**  기간이 하나일 때만 쓴다 — 둘 이상이면 어느
        쪽이 사건 범위인지 의미만으로 정해지지 않는다.
        """

        # A date bound to the target is an event-identity coordinate.  Status
        # observation points such as ``지금`` live in scope and must not
        # replace that identity date when both are present.
        qualifier_dates = [
            expression
            for item in source_intent.answer_items
            if item.target.kind in {"event", "document"}
            for expression in item.target.qualifier_surfaces
            if parse_date_surface(expression) is not None
        ]
        if len(qualifier_dates) == 1:
            expressions = qualifier_dates
        else:
            expressions = [
                expression
                for item in source_intent.answer_items
                if item.target.kind in {"event", "document"}
                for expression in item.scope.target_period_expressions
            ]
        if not expressions:
            expressions = [
                expression
                for item in source_intent.answer_items
                if item.target.kind in {"event", "document"}
                for expression in item.target.qualifier_surfaces
                if parse_date_surface(expression) is not None
            ]
        if len(expressions) != 1:
            return None, None
        start, end, error = _target_date_range(
            expressions[0], reference_date=self.reference_date)
        if error is not None:
            return None, None
        return start, end

    def _counterparty_surface(
            self, source_intent: SemanticIntent) -> str | None:
        surfaces = [
            entity.surface for entity in source_intent.entities
            if entity.kind_hint == "counterparty" and entity.surface.strip()
        ]
        return surfaces[0] if len(surfaces) == 1 else None

    @staticmethod
    def _unspecified_contract_completion_reason(
            source_intent: SemanticIntent,
            ) -> "tuple[Any, Any, str] | None":
        """Recognize one unselected-contract false-premise question.

        ``정상 완료된 이유`` is not evidence that a contract completed.
        With no event selector, period or counterparty, the event identity must
        be supplied before that premise can be checked.
        """

        if (len(source_intent.answer_items) != 1
                or source_intent.answer_groups
                or source_intent.unresolved_mentions):
            return None
        item = source_intent.answer_items[0]
        if (item.target.kind not in {"event", "document"}
                or item.operation != "retrieve"
                or item.selection is not None
                or item.target.qualifier_surfaces
                or item.scope.target_period_expressions
                or item.scope.as_of_expression is not None
                or item.scope.document_group_expression is not None
                or item.scope.scope_qualifier_expressions
                or any(entity.kind_hint == "counterparty"
                       for entity in source_intent.entities)):
            return None
        surface = " ".join((
            item.target.surface, *item.output.field_surfaces,
            *(premise.raw_text for premise in source_intent.premises),
        ))
        compact = re.sub(r"\s+", "", surface)
        if ("계약" not in compact or "정상완료" not in compact
                or not any(token in compact for token in ("이유", "사유", "원인"))):
            return None
        by_id = {entity.entity_id: entity for entity in source_intent.entities}
        companies = [
            by_id[ref] for ref in item.target.entity_refs
            if ref in by_id and by_id[ref].kind_hint == "company"
        ]
        if len(companies) != 1:
            return None
        return item, companies[0], "계약"

    def _completion_premise_clarification(
            self, source_intent: SemanticIntent,
            ) -> "ClarificationAuthority | None":
        matched = self._unspecified_contract_completion_reason(source_intent)
        if matched is None:
            return None
        item, company, contract_surface = matched
        lookup = build_event_clarification_options(
            self.canonical,
            company_surface=company.surface,
            as_of=self.as_of,
            contract_surface=contract_surface,
            preflight=self.event_preflight,
            corpus_cutoff=self.corpus_cutoff,
        )
        if lookup.status not in {"ambiguous", "too_many"}:
            return None
        return ClarificationAuthority(slots=[ClarificationSlot(
            slot_id="slot-1",
            role_hint="event",
            reason_code="event_target_multiple_candidates",
            response_kind="provide_value",
            # 계약명은 후보를 못 가른다.  한 회사의 공급계약 여러 건이 같은
            # 이름을 쓰는 일이 흔해, 계약명을 답해도 후보가 그대로 남는다.
            # 되묻고도 못 좁히는 역질문은 역질문이 아니다.  계약상대는
            # 공시 서식의 `3. 계약상대` 로 건마다 실려 실제로 가른다.
            prompt="어느 계약을 뜻하나요? 접수번호 또는 계약상대를 알려주세요.",
            applies_to_item_ids=[item.item_id],
            mention_ids=[], options=[],
        )])

    def resolve(
            self,
            *,
            question_id: str,
            question: str,
            source_intent: SemanticIntent,
            ) -> "ResolutionAuthority | None":
        del question_id                     # 문항 식별자로 분기하지 않는다
        false_premise = self._completion_premise_clarification(source_intent)
        if false_premise is not None:
            return false_premise
        event_from, event_to = self._event_window(source_intent)
        correction_observed_on = None
        if re.search(r"정정\s*공시|기재\s*정정", question):
            complete_dates = []
            for parts in question_date_surfaces(question):
                if parts[1] is not None and parts[2] is not None:
                    complete_dates.append(
                        f"{parts[0]:04d}{parts[1]:02d}{parts[2]:02d}")
            if len(set(complete_dates)) == 1:
                correction_observed_on = complete_dates[0]
        decision = build_event_clarification_decision(
            self.canonical,
            source_intent=source_intent,
            as_of=self.as_of,
            preflight=self.event_preflight,
            corpus_cutoff=self.corpus_cutoff,
            counterparty_surface=self._counterparty_surface(source_intent),
            event_from=event_from,
            event_to=event_to,
            correction_observed_on=correction_observed_on,
        )
        authority = decision.authority
        if authority is None:
            return None
        # **후보가 여럿인 것과 물음이 모호한 것은 다르다.**
        #
        # 「2024년 계약 중 해지가 확인되는 것은 무엇인가」는 후보가 열둘이어도
        # 모호하지 않다.  그 열둘이 답을 고르는 대상이기 때문이다.  의미가
        # 미해결 언급을 하나도 표시하지 않았다면 되물을 자리가 없고, 슬롯의
        # `mention_ids` 도 비어 답을 되돌려 묶을 곳이 없다.
        corpus_discovered_same_day = (
            (correction_observed_on is not None
             or (event_from is not None and event_from == event_to
                 and re.search(r"그\s*계약|어느\s*계약|해지됐", question)))
            and re.search(r"계약", question) is not None
            and all(slot.options for slot in authority.slots)
        )
        if (not any(slot.mention_ids for slot in authority.slots)
                and not corpus_discovered_same_day):
            return None
        return authority


class CompositeResolutionBackend:
    """등록 순서대로 물어보고 **처음 확정한 권위**를 돌려준다."""

    def __init__(
            self, *backends: Stage1ResolutionBackendLike,
            source_intent_regrounder: Any | None = None,
            ) -> None:
        if not backends:
            raise ValueError("백엔드가 최소 하나는 필요합니다")
        self.backends = backends
        self.source_intent_regrounder = source_intent_regrounder

    def reground_source_intent(
            self, question: str, source_intent: SemanticIntent,
            ) -> SemanticIntent:
        hook = self.source_intent_regrounder
        if hook is None:
            return source_intent
        if not callable(hook):
            raise TypeError("source intent regrounder는 callable이어야 합니다")
        value = hook(question, source_intent)
        if not isinstance(value, SemanticIntent):
            raise TypeError("source intent regrounder 반환값이 잘못되었습니다")
        return value

    def resolve(
            self,
            *,
            question_id: str,
            question: str,
            source_intent: SemanticIntent,
            ) -> "BackendResolutionResult":
        coverage_result: BackendResolutionResult | None = None
        for backend in self.backends:
            result = normalize_backend_resolution_result(backend.resolve(
                question_id=question_id,
                question=question,
                source_intent=source_intent,
            ), none_status="not_applicable")
            if result.status == "resolved":
                return result
            if result.status == "coverage_unavailable" and coverage_result is None:
                coverage_result = result
        if coverage_result is not None:
            return coverage_result
        return BackendResolutionResult.capability_unmatched(
            diagnostic_code="resolver_capability_unmatched")


__all__ = [
    "CanonicalQuestionCompanyRegrounder",
    "CompositeResolutionBackend",
    "DerivedComparisonJudgmentItemRegrounder",
    "EventClarificationResolutionBackend",
    "Stage1ResolutionBackendLike",
]


#: Stage1 v1 의 **유일한** 백엔드 구성 지점.
#:
#: 예전에는 채점 스크립트마다 같은 구성을 복사해 두어, 백엔드를 하나 고치면 두
#: 군데를 고쳐야 했고 한쪽을 빠뜨리면 두 채점이 다른 답을 냈다.  구성은 여기
#: 한 곳에만 둔다.
DEFAULT_RESOLVER_VERSION = "stage1-resolver/1.0"
DEFAULT_BUILD_ID = "a" * 32
DEFAULT_EVENT_CACHE_ROOT = (
    Path(__file__).resolve().parents[1] / "out/serving/event_roles")


def build_stage1_v1_backend(
        canonical: Any,
        *,
        reference_date: date,
        corpus_cutoff: str,
        canonical_build_id: str = DEFAULT_BUILD_ID,
        resolver_version: str = DEFAULT_RESOLVER_VERSION,
        event_preflight: Any | None = None,
        event_cache_root: str | Path | None = DEFAULT_EVENT_CACHE_ROOT,
        ) -> CompositeResolutionBackend:
    """정본 preflight 를 태운 v1 백엔드 전체를 **확정 강도 순서**로 세운다.

    순서가 계약이다.  하나로 좁혀지면 그것을 쓰고, 좁혀지지 않지만 사실이
    확인되면 한계를 기록하며 답하고, 그것도 아니면 되묻는다.  되묻기가 앞에
    오면 답할 수 있는 질문에도 되묻게 된다.
    """

    from agent.stage1_v1_document_backends import (
        BusinessContentNarrativeRegrounder,
        BusinessNarrativeClauseFieldRegrounder,
        DocumentAttributeEvidenceBackend,
        DocumentCollectionResolutionBackend,
        DocumentVersionHistoryResolutionBackend,
        DocumentFactComparisonBackend,
        InvestmentJudgmentTitleDateRegrounder,
        PeriodicNarrativeTopicRegrounder,
        PeriodicNarrativeWholeTopicRegrounder,
        PeriodicNarrativeComparisonBackend,
        PeriodicNarrativeResolutionBackend,
        SameDayDocumentCandidatesBackend,
        TerminationReportedStatusBackend,
        SelectedEventResolutionBackend,
    )
    from agent.stage1_v1_narrative_matrix import (
        AnnualBusinessMatrixIntentRegrounder,
        NarrativeMatrixCapacityClarificationBackend,
        NarrativeMatrixResolutionBackend,
    )
    from agent.stage1_v1_context_clarification_backend import (
        MissingContextClarificationBackend,
    )
    from agent.stage1_v1_financial_backend import (
        FinancialResolutionBackend, QuestionGroundedCompoundFinancialRegrounder,
    )
    from agent.stage1_v1_quarter_comparison import QuestionGroundedQuarterComparisonRegrounder
    from agent.quantity_contract_collection import QuantityContractCollectionRegrounder
    from agent.stage1_v1_event_collection_backend import (
        EventCollectionResolutionBackend,
    )
    from agent.stage1_v1_event_collection import (
        ContractAmountPairClarificationRegrounder,
        ContractAmountAvailabilityFieldRegrounder,
        EventArgmaxAmountFieldTargetRegrounder,
        EventRecordDateQualifierRegrounder,
        EventSingleFieldNarrativeShapeRegrounder,
        EventTwoFieldItemMergeRegrounder,
        TerminationExistenceIntentRegrounder,
    )
    from agent.stage1_v1_literal_form_regrounder import (
        LiteralDisclosureFormFieldRegrounder,
    )
    from agent.stage1_v1_lifecycle_composite_backend import (
        LifecycleCompositeResolutionBackend,
    )
    from agent.stage1_v1_correction_backend import (
        CorrectionLineageResolutionBackend,
    )
    from agent.stage1_v1_correction_lineage import (
        CanonicalCorrectionLineageSelector,
    )
    from agent.stage1_v1_correction_regrounder import (
        CorrectionLineageIntentRegrounder,
    )
    from agent.stage1_v1_event_history_regrounder import (
        EventHistoryIntentRegrounder,
    )
    from agent.exact_periodic_filing import (
        ExactPeriodicFinancialRegrounder,
        ReportFormFinancialPeriodRegrounder,
    )
    from agent.relative_financial_period import (
        RelativeFinancialPeriodRegrounder,
    )
    from agent.recent_periods_fanout import RecentPeriodsFanoutRegrounder
    from agent.summary_metric_fanout import SummaryMetricFanoutRegrounder
    from agent.stage1_v1_exact_event_regrounder import (
        ExactDatedEventFieldsRegrounder,
    )
    from agent.stage1_v1_policy_backend import (
        CausalPolicyResolutionBackend,
        GeneralPolicyResolutionBackend,
    )
    from agent.stage1_v1_clarification_candidate_filter import (
        ConceptClarificationExistenceFilterBackend,
    )
    from agent.stage1_v1_holding_backend import (
        HoldingDisclosureResolutionBackend,
        QuestionGroundedHoldingIntentRegrounder,
    )
    if event_preflight is None:
        from agent.event_preflight import CanonicalEventKeyPreflight
        event_preflight = CanonicalEventKeyPreflight(
            canonical, corpus_cutoff=corpus_cutoff,
            cache_root=(None if event_cache_root is None
                        else Path(event_cache_root)))
    if not callable(getattr(event_preflight, "resolve_event_key", None)):
        raise TypeError("shared event preflight 계약이 잘못되었습니다")

    shared = {
        "canonical_build_id": canonical_build_id,
        "resolver_version": resolver_version,
        "reference_date": reference_date,
        "corpus_cutoff": corpus_cutoff,
    }
    from agent.disclosed_metric_topics import DisclosedMetricTopicRegrounder
    from agent.planner_preflight import CanonicalSelectorRolePreflight
    company_preflight = CanonicalSelectorRolePreflight(
        canonical, corpus_cutoff=corpus_cutoff)
    company_alias_regrounder = CanonicalCompanyAliasMergeRegrounder(canonical)
    financial_backend = FinancialResolutionBackend(
        canonical, scope_authority=canonical,
        company_alias_provenance=company_alias_regrounder.provenance_for,
        **shared)
    correction_selector = CanonicalCorrectionLineageSelector(
        canonical, corpus_cutoff=corpus_cutoff,
        event_preflight=event_preflight)
    correction_backend = CorrectionLineageResolutionBackend(
        correction_selector, **shared)
    return CompositeResolutionBackend(
        HeldCompanyClarificationBackend(canonical),
        GeneralPolicyResolutionBackend(canonical, corpus_cutoff=corpus_cutoff),
        CausalPolicyResolutionBackend(),
        # Holding has a second company-like axis: the filer.  Close it to one
        # correction-effective receipt before generic document/event paths can
        # collapse the filer into the issuer or scan the whole holding family.
        HoldingDisclosureResolutionBackend(canonical, **shared),
        UnsupportedInvestmentOperatorBackend(),
        IncomparableInvestmentCashflowBackend(),
        correction_backend,
        RootMissingCorrectionDocumentBackend(
            correction_backend, canonical, **shared),
        # 개념 CLARIFY(예: 「이자」 4택)의 후보를 정본 존재로 거른다
        # (issue #172 M31) — financial_backend 자체는 건드리지 않는다.
        ConceptClarificationExistenceFilterBackend(
            financial_backend, canonical, corpus_cutoff=corpus_cutoff),
        DocumentVersionHistoryResolutionBackend(canonical, **shared),
        NarrativeMatrixCapacityClarificationBackend(),
        NarrativeMatrixResolutionBackend(canonical, **shared),
        PeriodicNarrativeComparisonBackend(canonical, **shared),
        PeriodicNarrativeResolutionBackend(canonical, **shared),
        EventCollectionResolutionBackend(canonical, **shared),
        LifecycleCompositeResolutionBackend(canonical, **shared),
        # Reported termination must run before ordinary selected-event status.
        # Both can prove the same event key, but only the former retains the
        # ambiguous-origin limitation required when two originals share it.
        TerminationReportedStatusBackend(canonical, **shared),
        SelectedEventResolutionBackend(
            canonical, event_preflight=event_preflight, **shared),
        DocumentFactComparisonBackend(canonical, **shared),
        SameDayDocumentCandidatesBackend(canonical, **shared),
        DocumentAttributeEvidenceBackend(
            canonical, event_preflight=event_preflight, **shared),
        DocumentCollectionResolutionBackend(canonical, **shared),
        MissingContextClarificationBackend(
            canonical, event_preflight=event_preflight, **shared),
        EventClarificationResolutionBackend(
            canonical, as_of=corpus_cutoff, reference_date=reference_date,
            corpus_cutoff=corpus_cutoff,
            event_preflight=event_preflight),
        source_intent_regrounder=CompositeSourceIntentRegrounder(
            # A holding question may be emitted as separate scalar metric
            # items bound to the filer/party. Restore only the unique issuer
            # that is literal in the question and backed by holding docs
            # before the generic universe policy can reject the party.
            QuestionGroundedHoldingIntentRegrounder(
                canonical, corpus_cutoff=corpus_cutoff,
                company_preflight=company_preflight),
            # A schema-valid model sample can still choose the wrong semantic
            # topology. Prefer only four fully literal, canonical-company
            # grammars; open or ambiguous requests keep provider authority.
            QuestionGroundedCompoundFinancialRegrounder(
                company_preflight.question_company_surface),
            QuestionGroundedQuarterComparisonRegrounder(),
            QuantityContractCollectionRegrounder(
                company_preflight.question_company_surface),
            ClosedLiteralQuestionIntentRegrounder(
                canonical, company_preflight),
            CorrectionLineageIntentRegrounder(),
            # 이슈 #74 — "체결한 계약 중 이후 해지된 계약이 존재하는가" 류
            # 계약명 미지정 개방형 존재 질의를 event/verdict/whole_target
            # 모양에서 event collection 이 이미 처리하는 named_fields 모양
            # 으로 복구한다.  다른 어떤 regrounder도 이 verdict shape를
            # 건드리지 않으므로 순서는 이르든 늦든 상관없지만, 닫힌 literal
            # 복구 규칙들 옆에 둔다.
            TerminationExistenceIntentRegrounder(),
            # 이슈 #75 — 낫표로 서식명을 리터럴 지정한 질문(「신규시설투자」
            # 「주요사항보고서(전환사채권발행결정)」)을 재무 regrounder들과
            # financial_backend 보다 먼저 사건 슬롯 모양으로 돌린다. 늦게
            # 두면 QuestionGroundedFinancialRegrounder·
            # DisclosedMetricTopicRegrounder 가 먼저 target.kind=metric
            # 표면을 재무 개념으로 바꿔 버려 여기서 알아볼 수 없다.
            LiteralDisclosureFormFieldRegrounder(),
            RootMissingCorrectionDocumentRegrounder(canonical),
            CanonicalQuestionCompanyRegrounder(
                canonical, corpus_cutoff=corpus_cutoff,
                preflight=company_preflight),
            # 재무 사전이 모르는 공시 서술지표(순이자마진·수주잔고·생산능력·가동률
            # ·임상·연구개발비 …)는 정기공시 narrative topic 으로 보낸다. 재무
            # 정본·구어 사전이 답하거나 거절하는 표면은 건드리지 않는다. 재무
            # regrounder 보다 앞에 두는 이유: 그들이 「순이자마진」 안의 「이자」
            # 같은 계정 별칭을 먼저 잡아 표면을 바꿔 버리면 여기서 알아볼 수 없다.
            DisclosedMetricTopicRegrounder(company_preflight),
            # 이슈 #38 — 「부채비율을 계산하고, 안 되면 부채총계와 자본총계만」
            # 처럼 이름 있는 비율과 그 분자·분모를 별도 항목으로 되풀이하는
            # 3항목 intent를 비율 항목 하나로 흡수한다. financial_backend가
            # 각 항목을 개별로 못 풀면(분자·분모 항목은 자기 기간이 없다)
            # 전체를 unsupported_request로 닫았던 경로를 없앤다.
            NamedRatioOperandItemRegrounder(),
            # 이슈 #59 1단계 — 회사 간 「몇 배」 질문에서 HCX가 두 번째
            # 회사의 entity_ref를 빠뜨리는 wire 모양을 복구한다. 재무
            # regrounder들이 target.surface·entity_refs를 먼저 건드리기
            # 전에 둔다.
            CrossCompanyMultipleEntityRegrounder(),
            # 이슈 #64 — 「연간 매출액과 4분기 단독 매출액을 각각 알려주고,
            # 두 값의 차이를 분기 성장률이라고 불러도 되는지 판단해줘」처럼
            # 자기 entity·period가 없는 파생/판단 항목(「두 값의 차이」·「분기
            # 성장률」)을 흡수한다. 판단 자체는 runtime의 기간 길이 가드가
            # 답한다 — 위 named-ratio 흡수와 같은 자리(재무 regrounder들보다
            # 먼저)에 둔다.
            DerivedComparisonJudgmentItemRegrounder(),
            # 이슈 #58 2단계 — 「최근 N년」「최근 N개 분기」다중 기간 전개.
            # 아래 `QuestionGroundedFinancialRegrounder`(그리고
            # `ExactPeriodicFinancialRegrounder`)는 질문 아무 데나 있는
            # bare 「20XX년」리터럴을 통째로 붙잡아 단일 기간으로 확정한다
            # (SG-012 「최근 3년간(2023~2025년)」의 괄호 속 마지막 연도
            # 「2025년」을 그렇게 가져가 이 fanout이 이미 다른 기간이 있다고
            # 보고 손을 떼는 것을 직접 확인함). 그래서 item 하나를 N개로
            # 늘리는 이 fanout은 그 두 regrounder보다 **먼저** 와야 한다 —
            # `SummaryMetricFanoutRegrounder`(개념 fanout)와 달리 이 fanout은
            # 실제 연도·분기를 정본에서 스스로 계산하므로(기간 축 regrounder
            # 들의 복구를 기다릴 필요가 없다) 회사 축 복구
            # (`CanonicalQuestionCompanyRegrounder`, 이미 위에서 끝났다)만
            # 있으면 된다. item 을 N개로 늘리므로, 뒤에 오는 단일-item
            # 전제 regrounder들은 이 코드베이스의 일반 규약대로 조용히
            # 손을 뗀다(`len(answer_items) != 1` → 그대로 반환).
            RecentPeriodsFanoutRegrounder(
                canonical, reference_date=reference_date,
                corpus_cutoff=corpus_cutoff, scope_authority=canonical),
            # 정확한 제출일+보고서 형식이 함께 명시된 재무 질문은 최신본 기본값이
            # 아니라 그 as-filed 문서 좌표를 쓴다. 메타데이터가 한 건으로 닫힐
            # 때만 intent를 고치므로 일반 연도 질의에는 영향이 없다.
            ExactPeriodicFinancialRegrounder(
                canonical, as_of=corpus_cutoff),
            # One closed scalar financial request may lose coordinates at the
            # semantic wire.  Restore only literal, unique coordinates before
            # the financial resolver; event and narrative shapes are excluded.
            QuestionGroundedFinancialRegrounder(canonical),
            # 이슈 #171 M04 — 위 regrounder의 개념 표면 재정렬(`_concept_surface`)
            # 은 토큰을 공백으로 다시 이어 붙인다. 「수익(매출액)」처럼 이미
            # 정본 사전에 있는 괄호 병기 별칭이 그 과정에서 「수익 매출액」
            # (공백, 질문에 없는 문자열)으로 깨진다(실호출로 직접 확인함,
            # `Stage1V1ResolverTechnicalError:resolver_authority_failed`).
            # 바로 다음 자리에서 grounding이 깨진 target.surface를 원 질문의
            # 괄호 리터럴로 되돌린다.
            ParenthesizedConceptAliasSurfaceRegrounder(),
            # A bare「N년 X보고서」mention (no calendar day, so the exact
            # filing regrounder above cannot bind it) still needs its report
            # form translated into the period grammar `_financial_period`
            # actually knows, or the item silently falls back to the full
            # annual year (P9-017).  This must run *after*
            # ``QuestionGroundedFinancialRegrounder``: that regrounder
            # deterministically restores ``target_period_expressions`` from
            # every bare-year literal in the question for any closed single
            # scalar metric item — including one this regrounder already
            # translated — so a translation placed before it would be
            # silently reverted to the bare year for the same closed
            # questions this fix targets.
            ReportFormFinancialPeriodRegrounder(canonical),
            # 이슈 #58 — 「직전 분기」같은 단일 상대 기간 표현.  위 두
            # regrounder는 명시적 20XX년 리터럴이 있어야 움직이므로
            # (``_YEAR``/``_REPORT_FORM_MENTION``) 상대 기간뿐인 질문에는
            # 손대지 않는다 — 순서는 서로 영향이 없다.
            RelativeFinancialPeriodRegrounder(canonical),
            # 이슈 #171 M02 — 「FY2025」「2025 회계연도」. 위 상대 기간
            # regrounder와 같은 자리(순서 무관, 서로 다른 표현을 본다).
            AbsoluteFiscalYearLiteralRegrounder(canonical),
            # 이슈 #171 M09 — 시점 잔액의 「N년 말과 M년 말 … 차이는
            # 얼마인가」. 위 두 regrounder와 같은 자리(순서 무관).
            InstantEndpointDifferenceRegrounder(canonical),
            # 이슈 #171 M11 — 「A의 … X와 B의 … X의 합계는 얼마인가」가
            # 두 독립 item으로 오는 wire 변형. 두 item을 CG-056과 같은
            # 한-item 두-entity 모양으로 합친 다음에야 아래 단일-item
            # regrounder들(전부 이 두-item intent에는 조용히 손을 뗀다)과
            # #145의 sum 판정이 이어받을 수 있으므로 이 앞자리에 둔다.
            CrossCompanySumItemMergeRegrounder(canonical),
            # 이슈 #171 M24 — 「2025년 1월부터 12월까지」(1~12월 전체 범위)
            # 는 그 해 연간과 같은 뜻이다. 위 regrounder들과 같은 자리
            # (순서 무관).
            FullYearMonthRangeRegrounder(canonical),
            company_alias_regrounder,
            SameDayPeriodicVersionIntentRegrounder(company_preflight),
            QuestionGroundedNarrativeInvestmentRegrounder(
                company_preflight.question_company_surface),
            InvestmentPlanIntentRegrounder(company_preflight),
            AnnualBusinessMatrixIntentRegrounder(),
            SplitNarrativeMatrixItemRegrounder(),
            PeriodicNarrativeWholeTopicRegrounder(),
            BusinessNarrativeClauseFieldRegrounder(),
            InvestmentJudgmentTitleDateRegrounder(),
            PeriodicNarrativeTopicRegrounder(),
            BusinessContentNarrativeRegrounder(canonical),
            QuestionGroundedEventStatusRegrounder(
                canonical, company_preflight=company_preflight),
            MisplacedPeriodQualifierRegrounder(),
            # A split ``최종 계약금액 + 해지금액`` request is one
            # contract-record question.  Restore that topology while keeping
            # the generic contract target unresolved, so two canonical roots
            # produce a useful SELECT_ONE clarification rather than an
            # arbitrary amount or a generic refusal.
            ContractAmountPairClarificationRegrounder(
                company_preflight=company_preflight),
            # 이슈 #156 — CG-008·009·010·019·022: 한 공시의 필드 두 개를
            # 묻는 닫힌 질문을 HCX가 answer_items 두 개(kind=attribute/
            # metric/entity 제각각)로 쪼갠다. 사건 백엔드는 단일 event
            # 항목 + record shape만 받으므로 이 모양은 그대로
            # unsupported_semantic_target으로 닫힌다. 이 regrounder는
            # len(answer_items)==2에만 반응하므로(단일-item 규약, 이
            # 코드베이스 일반 관례) 바로 위·아래의 단일-item 전용 규칙과는
            # 겹치지 않는다 — 두 항목을 event 항목 하나로 합치고 날짜를
            # scope.target_period_expressions에 두므로, 바로 다음
            # EventRecordDateQualifierRegrounder가 그 축을
            # event_field_read가 요구하는 자리(target.qualifier_surfaces)로
            # 옮겨 마무리한다. 반드시 그 규칙 앞에 둔다.
            EventTwoFieldItemMergeRegrounder(company_preflight=company_preflight),
            # 이슈 #151 — MisplacedPeriodQualifierRegrounder가 단일사건
            # record/scalar 질문(CG-005·CG-025)의 사건 발생일을 scope로
            # 옮겨 event_field_read 조건을 깨는 것을 되돌린다. 바로 위
            # EventTwoFieldItemMergeRegrounder는 2-item 질문에만 반응해
            # 이 단일-item 질문들은 그대로 지나치므로, 기능적으로는 여전히
            # MisplacedPeriodQualifierRegrounder 바로 다음이다 — 그 규칙이
            # 옮긴 자리를 되돌리는 것이므로 이 상대적 순서(모든 단일-item
            # 축 이동 규칙보다 먼저)는 유지한다.
            EventRecordDateQualifierRegrounder(),
            FundingComparisonSplitItemRegrounder(),
            CompatibleEventCollectionItemRegrounder(),
            FundingComparisonDemandRegrounder(),
            ClosedFinancialRelationRegrounder(),
            # 이슈 #136 — 3개사 이상 「가장 큰/작은」이 scalar/retrieve 로
            # 나와 N-ary 재무 순위 컴파일러 전에 거절되는 wire 를
            # compare/comparison 으로 되돌린다. 2개사 비교는
            # len(companies) < 3 가드로 그대로 둔다.
            NaryFinancialSuperlativeShapeRegrounder(),
            # R-P-003 — 위 형제의 2개사 가정을 채운다. operation·shape 는
            # 이미 옳고 selection 만 비어 오는 표본이 컴파일러의 2개사 비교
            # 검증에서 거절되던 자리다. 위 형제와 순서는 무관하지만(그쪽은
            # 3개사 이상, 이쪽은 정확히 2개사) 같은 「컴파일러가 요구하는
            # 축을 되돌린다」 규칙이므로 나란히 둔다. 아래
            # ClosedFinancialComparisonRegrounder 는 기간이 비어 있으면
            # 질문에서 되살리는데, 그 판정은 selection 을 보지 않으므로
            # 이 앞뒤 순서도 결과를 바꾸지 않는다.
            TwoCompanySuperlativeSelectionRegrounder(),
            ClosedFinancialComparisonRegrounder(
                company_preflight.question_company_surface),
            QuestionGroundedLatestEventRegrounder(),
            # RPC-007 실호출 — HCX가 「N년 M월 D일까지 공개된 A 의 B 계약
            # 공시에서 계약금액이 `-`로 표시된 것은 0원인지 비공개인지
            # 구분해줘」를 document/narrative/named_fields 한 필드로 낼 때,
            # as-of·상대방을 event collection availability 축으로 되돌린다.
            ContractAmountAvailabilityFieldRegrounder(canonical),
            # 이슈 SG-011 — 「계약금액이 가장 큰 건」의 금액 필드를
            # target.kind=metric 으로 읽고 최상급·답 필드 표면을 흩어 버린
            # wire 를, PR #106 이 이미 인식하는 event argmax 모양(SG-010,
            # target.kind=event)으로 되돌린다. 재무 regrounder들이 이미
            # 앞서 지나며 「계약금액」을 재무 개념으로 건드리지 않는 것을
            # 확인했으므로(사전 밖 표면) 자리는 다른 event 표면 regrounder
            # 들 옆이면 된다.
            EventArgmaxAmountFieldTargetRegrounder(),
            # 이슈 #132 — 이미 target.kind=event 로 사건은 올바르게 읽었지만
            # 필드가 하나뿐이면 output.shape=narrative 로 내고 날짜를
            # scope.target_period_expressions 에 두는 wire 를, 두 필드
            # 질문(#121 CG-011)이 이미 통과하는 shape=record·날짜=
            # target.qualifier_surfaces 모양으로 되돌린다.
            EventSingleFieldNarrativeShapeRegrounder(),
            ExactDatedEventFieldsRegrounder(),
            NamedEventFieldsRegrounder(canonical),
            ClosedQuestionSemanticRegrounder(),
            # Keep event identity dates on the target axis after the generic
            # period normalizers have run; lifecycle status dates remain on
            # the observation axis inside this regrounder.
            EventHistoryIntentRegrounder(company_preflight),
            # Keep this last: once an unsupported operand is removed, no
            # downstream comparison repairer may recreate a winner or gap.
            MixedSupportedFinancialRegrounder(
                canonical, corpus_cutoff=corpus_cutoff),
            # 이슈 #171 M16 — 「삼성전자의 2023년, 2024년, 2025년 연결
            # 매출액을 각각 알려줘」처럼 형제 item마다 **이미** 서로 다른
            # 리터럴 기간을 낸 재무 fanout. 값을 계산하지 않고 구조만 보고
            # presentation을 못박으므로 canonical 조회가 필요 없다 — 그래도
            # 위 모든 회사·기간 축 복구 regrounder가 이미 끝난 **최종적으로
            # 닫힌 모양**만 보고 판정해야 하므로(바로 아래
            # `SummaryMetricFanoutRegrounder`와 같은 이유) 그 바로 앞에 둔다.
            # 그 regrounder처럼 이후로도 손댈 것이 없다 — 항목 수를
            # 바꾸지 않으므로 뒤(개념 fanout)는 `len(answer_items) == 1`
            # 가정이 깨져 조용히 손을 뗀다.
            ExplicitPeriodsFanoutRegrounder(),
            # 이슈 #94 25 — 「얼마나 벌었어」·「빚이 얼마야」의 후보를 되묻지 않고
            # 전부 답한다. **정말로 마지막이다.** item 하나를 셋으로 늘리므로,
            # 뒤에 오는 regrounder 가 단일 item 을 가정하면 조용히 손을 뗀다
            # (`len(answer_items) != 1` → 그대로 반환). 앞의 기간·회사 복구가
            # 모두 끝난 **확정된 모양**을 보고 펴야 후보 좌표 확인도 실제 값과
            # 같은 자리에서 이뤄진다.
            SummaryMetricFanoutRegrounder(
                canonical, reference_date=reference_date,
                corpus_cutoff=corpus_cutoff, scope_authority=canonical),
        ),
    )


__all__ += ["DEFAULT_BUILD_ID", "DEFAULT_RESOLVER_VERSION",
            "DEFAULT_EVENT_CACHE_ROOT",
            "ClosedQuestionSemanticRegrounder",
            "QuestionGroundedEventStatusRegrounder",
            "ClosedFinancialComparisonRegrounder",
            "ClosedFinancialRelationRegrounder",
            "ClosedLiteralQuestionIntentRegrounder",
            "CanonicalCompanyAliasMergeRegrounder",
            "QuestionGroundedLatestEventRegrounder",
            "NamedEventFieldsRegrounder",
            "MisplacedPeriodQualifierRegrounder",
            "FundingComparisonSplitItemRegrounder",
            "CompatibleEventCollectionItemRegrounder",
            "FundingComparisonDemandRegrounder",
            "HeldCompanyClarificationBackend",
            "QuestionGroundedFinancialRegrounder",
            "QuestionGroundedNarrativeInvestmentRegrounder",
            "UnsupportedInvestmentOperatorBackend",
            "InvestmentPlanIntentRegrounder",
            "MixedSupportedFinancialRegrounder",
            "RootMissingCorrectionDocumentBackend",
            "RootMissingCorrectionDocumentRegrounder",
            "SameDayPeriodicVersionIntentRegrounder",
            "CompositeSourceIntentRegrounder", "build_stage1_v1_backend"]


def build_stage1_v1_runtime(
        canonical: Any,
        *,
        reference_date: date,
        corpus_cutoff: str,
        canonical_build_id: str | None = None,
        resolver_version: str = DEFAULT_RESOLVER_VERSION,
        max_turns: int | None = None,
        clarification_store: Any | None = None,
        clarification_db_path: str | Path = (
            "out/serving/stage1_v1_clarifications.sqlite3"),
        event_cache_root: str | Path | None = DEFAULT_EVENT_CACHE_ROOT,
        ) -> Any:
    """역질문까지 이어진 v1 런타임. **묻기와 재개가 한 객체에서 닫힌다.**

    `open`(되묻기)만 붙이고 `resume`(답 받고 재개)을 안 붙이면 사용자가 고른
    접수번호를 받을 곳이 없다.  둘은 한 쌍이므로 여기서 함께 세운다.
    """

    from agent.stage1_v1_clarification_session import (
        MAX_CLARIFICATION_TURNS,
        SQLiteStage1V1ClarificationStore,
        Stage1V1ClarificationCoordinator,
        Stage1V1ClarificationRuntime,
    )
    from agent.stage1_v1_context_resume_backend import (
        CanonicalContractAmountChangeResumeBackend,
        CanonicalFinancialContextResumeBackend,
        CanonicalGenericEventContextResumeBackend,
        CanonicalHoldingContextResumeBackend,
        CanonicalNarrativeMatrixResumeBackend,
        CompositeClarificationResumeBackend,
    )
    from agent.stage1_v1_event_clarification_backend import (
        CanonicalSelectedEventResumeBackend,
    )
    from agent.stage1_v1_outcome import Stage1V1Orchestrator
    from agent.stage1_v1_resolver import Stage1V1Resolver

    actual_build_id = getattr(canonical, "build_id", None)
    if (not isinstance(actual_build_id, str)
            or len(actual_build_id) != 32
            or any(character not in "0123456789abcdef"
                   for character in actual_build_id)):
        raise ValueError("v1 runtime에는 실제 canonical build_id가 필요합니다")
    if canonical_build_id is None:
        canonical_build_id = actual_build_id
    elif canonical_build_id != actual_build_id:
        raise ValueError("주입한 canonical_build_id가 실제 snapshot과 다릅니다")

    from agent.event_preflight import CanonicalEventKeyPreflight
    event_preflight = CanonicalEventKeyPreflight(
        canonical, corpus_cutoff=corpus_cutoff,
        cache_root=(None if event_cache_root is None
                    else Path(event_cache_root)))
    # Shift the role-index load to process startup.  All event backends below
    # share this exact object, so no request rebuilds or reloads the index.
    event_preflight.warm()

    backend = build_stage1_v1_backend(
        canonical, reference_date=reference_date, corpus_cutoff=corpus_cutoff,
        canonical_build_id=canonical_build_id,
        resolver_version=resolver_version,
        event_preflight=event_preflight,
        event_cache_root=event_cache_root)
    orchestrator = Stage1V1Orchestrator(Stage1V1Resolver(
        backend, canonical_build_id=canonical_build_id,
        resolver_version=resolver_version))
    event_resume = CanonicalSelectedEventResumeBackend(
        canonical, canonical_build_id=canonical_build_id,
        resolver_version=resolver_version,
        reference_date=reference_date, corpus_cutoff=corpus_cutoff,
        event_preflight=event_preflight)
    financial_resume = CanonicalFinancialContextResumeBackend(
        canonical, canonical_build_id=canonical_build_id,
        resolver_version=resolver_version,
        reference_date=reference_date, corpus_cutoff=corpus_cutoff)
    holding_resume = CanonicalHoldingContextResumeBackend(
        canonical, canonical_build_id=canonical_build_id,
        resolver_version=resolver_version,
        reference_date=reference_date, corpus_cutoff=corpus_cutoff)
    contract_change_resume = CanonicalContractAmountChangeResumeBackend(
        canonical, canonical_build_id=canonical_build_id,
        resolver_version=resolver_version,
        reference_date=reference_date, corpus_cutoff=corpus_cutoff,
        event_preflight=event_preflight)
    generic_event_resume = CanonicalGenericEventContextResumeBackend(
        canonical, canonical_build_id=canonical_build_id,
        resolver_version=resolver_version,
        reference_date=reference_date, corpus_cutoff=corpus_cutoff,
        event_preflight=event_preflight)
    narrative_matrix_resume = CanonicalNarrativeMatrixResumeBackend(
        canonical, canonical_build_id=canonical_build_id,
        resolver_version=resolver_version,
        reference_date=reference_date, corpus_cutoff=corpus_cutoff)
    store = (
        clarification_store
        if clarification_store is not None
        else SQLiteStage1V1ClarificationStore(clarification_db_path)
    )
    coordinator = Stage1V1ClarificationCoordinator(
        CompositeClarificationResumeBackend(
            event_resume, financial_resume, contract_change_resume,
            generic_event_resume, narrative_matrix_resume,
            holding_backend=holding_resume),
        store,
        max_turns=MAX_CLARIFICATION_TURNS if max_turns is None else max_turns,
        current_canonical_build_id=canonical_build_id,
    )
    return Stage1V1ClarificationRuntime(
        orchestrator=orchestrator, clarification_coordinator=coordinator)


#: 「FY2025」「2025 회계연도」— 이슈 #171 M02. HCX가 이 절대연도 표기를
#: target surface 안에 통째로 접어 넣는다(실호출로 직접 확인함,
#: 「FY2025 연결 매출액」·「2025 회계연도 연결 매출액」— scope는 비어
#: 있다). `agent.planning._financial_period`는 공백을 지운 뒤
#: 「20XX회계연도」는 이미 알지만(접미 표기) 「FY20XX」(접두 표기)는
#: 이 배치에서 그 함수 자체에 새로 가르친다 — 여기서는 리터럴을 그
#: target_period_expressions로 옮기는 표면 재정렬만 한다.
_ABSOLUTE_FISCAL_YEAR_LITERAL = re.compile(
    r"FY\s*20[0-9]{2}|20[0-9]{2}\s*회계\s*연도", re.IGNORECASE)


class AbsoluteFiscalYearLiteralRegrounder:
    """「FY2025」「2025 회계연도」를 재무 기간 좌표가 받는 리터럴로 되돌린다.

    `agent.relative_financial_period.RelativeFinancialPeriodRegrounder`와
    같은 자리(#58)의 절대연도판이다. 다만 그 표현은 연도마다 달라
    `agent.stage1_v1_financial_backend._CONCEPT_PREFIXES`(고정 문자열 목록)에
    미리 넣어 둘 수 없다 — 그래서 여기서는 매치된 리터럴을 target surface에서
    직접 떼어내 개념이 남긴 나머지로 다시 개념을 찾는다(`_bare_concept_surface`
    는 그 나머지에 남은 「연결」류 접두어를 마저 벗긴다).

    Fail-closed: 회사가 둘 이상, 이미 닫힌 단일 scalar 모양이 아니거나
    다른 기간이 이미 있거나, 리터럴을 뗀 나머지로도 개념을 못 찾으면
    손대지 않는다.
    """

    def __init__(self, canonical: Any) -> None:
        if not callable(getattr(canonical, "resolve_company", None)):
            raise TypeError(
                "absolute fiscal year regrounder에는 company resolver가 필요합니다")
        self.canonical = canonical

    @staticmethod
    def _closed_single_item(intent: SemanticIntent) -> Any | None:
        if (len(intent.answer_items) != 1 or intent.answer_groups
                or intent.premises or intent.unresolved_mentions):
            return None
        item = intent.answer_items[0]
        if (item.target.kind != "metric" or item.operation != "retrieve"
                or item.selection is not None
                or item.output.shape != "scalar"
                or item.output.projection_mode != "named_fields"
                or len(item.output.field_surfaces) != 1):
            return None
        return item

    def _single_company(self, item: Any, intent: SemanticIntent) -> bool:
        by_id = {entity.entity_id: entity for entity in intent.entities}
        companies = []
        for ref in item.target.entity_refs:
            entity = by_id.get(ref)
            if entity is None or entity.kind_hint != "company":
                continue
            rows = list(self.canonical.resolve_company(entity.surface) or ())
            if len(rows) == 1:
                companies.append(rows[0])
        corp_codes = {row.corp_code for row in companies}
        return len(companies) == 1 and len(corp_codes) == 1

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if not isinstance(question, str) or not question.strip():
            return intent
        match = _ABSOLUTE_FISCAL_YEAR_LITERAL.search(question)
        if match is None:
            return intent
        item = self._closed_single_item(intent)
        if item is None:
            return intent
        if not self._single_company(item, intent):
            return intent
        literal = match.group(0)
        if list(item.scope.target_period_expressions) == [literal]:
            return intent
        if item.scope.target_period_expressions:
            return intent  # 이미 다른 기간이 있으면 되묻지 않고 손 뗀다.

        from agent.stage1_v1_financial_backend import _bare_concept_surface

        surface = item.target.surface
        remainder = surface.replace(literal, "", 1).strip() if literal in surface else ""
        new_surface = remainder or surface
        concept = resolve_metric_concept(new_surface)
        if concept is None:
            concept = resolve_metric_concept(_bare_concept_surface(new_surface))
        if concept is None:
            return intent

        payload = intent.model_dump(mode="python", warnings=False)
        row = payload["answer_items"][0]
        row["scope"]["target_period_expressions"] = [literal]
        if remainder and remainder != surface:
            row["target"]["surface"] = remainder
        row["target"]["qualifier_surfaces"] = [
            value for value in row["target"]["qualifier_surfaces"]
            if value != literal]
        return SemanticIntent.model_validate(payload, strict=True)


#: 「수익(매출액)」류 괄호 병기 별칭. 정본 계정사전은 이미 이 리터럴을
#: 안다(공백/괄호를 지운 정규화 키로, `agent.planning.METRIC_ALIASES`) —
#: 개념을 새로 가르칠 필요가 없다. 문제는 그 리터럴이 grounding에서
#: 깨지는 자리뿐이다(`ParenthesizedConceptAliasSurfaceRegrounder` 참고).
_PARENTHESIZED_ALIAS = re.compile(
    r"[가-힣A-Za-z0-9]+\([가-힣A-Za-z0-9]+\)")


class ParenthesizedConceptAliasSurfaceRegrounder:
    """괄호 병기 별칭이 재정렬되며 깨진 target.surface를 원 리터럴로 되돌린다.

    이슈 #171 M04. ``QuestionGroundedFinancialRegrounder._concept_surface``
    는 질문을 단어 단위로 토큰화한 뒤(``_TOKEN`` 은 괄호를 토큰에 넣지
    않는다) 공백으로 다시 이어 붙여 개념 표면을 고른다. 「수익(매출액)」
    처럼 그 자체로 이미 정본 사전에 있는 괄호 병기 표기가 이 과정을 거치면
    「수익 매출액」(공백, 질문에 없는 문자열)이 되어, grounding이 이미
    통과했던 target.surface를 오히려 질문에 없는 문자열로 덮어써
    버린다(실호출로 직접 확인함).

    이 regrounder는 그 재정렬 바로 다음 자리에서, target.surface(그리고 같은
    재정렬이 함께 덮어쓰는 output.field_surfaces)가 질문에 없으면 원 질문의
    괄호 리터럴(있고, 같은 개념으로 풀리면)로 되돌린다. 다른 원인으로
    surface가 안 맞는 경우(괄호 리터럴이 없거나 다른 개념을 가리키면)는
    손대지 않는다 — grounding 실패는 여전히 하류가 그대로 거절한다.
    """

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if not isinstance(question, str) or not question.strip():
            return intent
        if (len(intent.answer_items) != 1 or intent.answer_groups
                or intent.premises or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        if item.target.kind != "metric" or item.operation != "retrieve":
            return intent
        surface = item.target.surface
        broken_fields = [
            field for field in item.output.field_surfaces
            if field not in question]
        if surface in question and not broken_fields:
            return intent  # 이미 정상적으로 결속됐다.
        current_concept = resolve_metric_concept(surface)
        if current_concept is None:
            # 재정렬 전에도 개념을 못 찾았다면 이 표기가 원래 무엇을
            # 가리켰는지 알 수 없다 — 질문에 있는 아무 괄호 별칭이나 집어
            # 오면 다른 개념을 조용히 대신 답할 위험이 있다.
            return intent
        for match in _PARENTHESIZED_ALIAS.finditer(question):
            literal = match.group(0)
            concept = resolve_metric_concept(literal)
            if concept is None or concept != current_concept:
                continue
            payload = intent.model_dump(mode="python", warnings=False)
            row = payload["answer_items"][0]
            if surface not in question:
                row["target"]["surface"] = literal
            row["output"]["field_surfaces"] = [
                literal if field not in question else field
                for field in row["output"]["field_surfaces"]]
            return SemanticIntent.model_validate(payload, strict=True)
        return intent


#: 시점 잔액의 두 연도말 endpoint. `QuestionGroundedFinancialRegrounder`
#: 의 같은 이름 정규식과 짝이다(리터럴이 다르면 함께 바꿔야 한다).
_INSTANT_ENDPOINT_LITERAL = re.compile(
    r"(?<![0-9])20[0-9]{2}\s*년(?:도)?\s*(?:[1-4]\s*분기\s*)?말")
#: 「차이」「차액」— #171 M09. `agent.stage1_v1_financial_backend.
#: _AMOUNT_CHANGE_FIELD`/`_EXPLICIT_AMOUNT_DIRECTION`은 이미 이 낱말을
#: difference 로 받지만, 그 판정은 두 endpoint가 **이미 하나의 compare
#: 항목**으로 닫힌 다음에야 닿는다 — 이 regrounder가 그 앞 자리(두
#: endpoint가 아직 한 retrieve/scalar 항목의 qualifier_surfaces 둘에
#: 나뉘어 있는 모양)를 채운다.
_DIFFERENCE_DEMAND = re.compile(r"차이|차액")


class InstantEndpointDifferenceRegrounder:
    """시점 잔액의 「N년 말과 M년 말 …의 차이는 얼마인가」를 compare로 되돌린다.

    이슈 #171 M09. `QuestionGroundedFinancialRegrounder`는 같은 두-endpoint
    모양을 「비교」또는 「보다/대비」가 있을 때만 compare로 되돌린다(실호출로
    직접 확인함) — 「차이는 얼마인가」만 있으면 그 세 갈래 모두 손대지
    않아, 단일 retrieve/scalar 항목(두 endpoint가 qualifier_surfaces 둘에
    나뉜 모양)이 그대로 unsupported_semantic_target으로 닫힌다. 같은 사실을
    묻는 「얼마나 늘거나 줄었는가」(M09-d)는 이미 compare/comparison 모양으로
    와 정상 동작하므로, 이 regrounder는 그 결과와 같은 모양으로 되돌릴
    뿐이다.

    Fail-closed: 회사가 둘 이상, 이미 닫힌 단일 scalar 항목이 아니거나,
    다른 기간이 이미 있거나, target qualifier_surfaces가 정확히 그 두
    endpoint가 아니거나(다른 수식어가 섞였으면 손 떼야 한다), 개념을 못
    찾으면 손대지 않는다.
    """

    def __init__(self, canonical: Any) -> None:
        if not callable(getattr(canonical, "resolve_company", None)):
            raise TypeError(
                "instant endpoint difference regrounder에는 company resolver가 "
                "필요합니다")
        self.canonical = canonical

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if not isinstance(question, str) or not question.strip():
            return intent
        if (len(intent.answer_items) != 1 or intent.answer_groups
                or intent.premises or intent.unresolved_mentions):
            return intent
        item = intent.answer_items[0]
        if (item.target.kind != "metric" or item.operation != "retrieve"
                or item.selection is not None
                or item.output.shape != "scalar"
                or item.output.projection_mode != "named_fields"
                or item.scope.target_period_expressions):
            return intent
        demand = _DIFFERENCE_DEMAND.search(question)
        if demand is None:
            return intent
        endpoints = list(dict.fromkeys(
            match.group(0).strip()
            for match in _INSTANT_ENDPOINT_LITERAL.finditer(question)))
        if len(endpoints) != 2:
            return intent
        qualifiers = sorted(
            value.strip() for value in item.target.qualifier_surfaces)
        if qualifiers != sorted(endpoints):
            return intent
        by_id = {entity.entity_id: entity for entity in intent.entities}
        companies = []
        for ref in item.target.entity_refs:
            entity = by_id.get(ref)
            if entity is None or entity.kind_hint != "company":
                continue
            rows = list(self.canonical.resolve_company(entity.surface) or ())
            if len(rows) == 1:
                companies.append(rows[0])
        corp_codes = {row.corp_code for row in companies}
        if len(companies) != 1 or len(corp_codes) != 1:
            return intent

        from agent.stage1_v1_financial_backend import _bare_concept_surface

        if (resolve_metric_concept(item.target.surface) is None
                and resolve_metric_concept(
                    _bare_concept_surface(item.target.surface)) is None):
            return intent

        payload = intent.model_dump(mode="python", warnings=False)
        row = payload["answer_items"][0]
        row["operation"] = "compare"
        row["target"]["qualifier_surfaces"] = []
        row["scope"]["target_period_expressions"] = list(endpoints)
        row["output"]["shape"] = "comparison"
        row["output"]["field_surfaces"] = [demand.group(0)]
        return SemanticIntent.model_validate(payload, strict=True)


#: 「합계」「합산」— #145(CG-056)와 같은 낱말. 그 배치는 한 항목이
#: 이미 두 entity_ref를 가진 wire 모양만 받는다.
_CROSS_COMPANY_SUM_CUE = re.compile(r"합계|합산")


class CrossCompanySumItemMergeRegrounder:
    """두 회사의 독립된 scalar 항목 둘을 CG-056과 같은 한-항목 두-entity 모양으로 합친다.

    이슈 #171 M11. 「삼성전자의 2025년 연결 매출액과 SK하이닉스의 2025년
    연결 매출액의 합계는 얼마인가?」의 실제 wire(HCX-007 실호출로 직접
    확인함)는 회사마다 독립된 retrieve/scalar 항목 둘(entity_ref 하나씩)
    이다 — #145가 고친 CG-056 모양(한 항목·entity_ref 둘)과 다르다.
    결정론적 plan compiler의 handler registry는 이 두-독립-item 구조를
    모르므로 `compiler_binding_failed`로 깨진다(감사 로그
    ``stage1_wire_failures.jsonl``와 같은 계열).

    이 regrounder는 두 항목이 **같은 개념·같은 기간·같은 scope**를 가리키고
    회사만 다를 때, 질문에 「합계/합산」이 있으면 첫 항목의 모양을 그대로
    쓰되 두 회사의 entity_ref를 (질문 등장 순서로) 모두 담은 한 항목으로
    합친다 — #145의 sum 판정(``_CROSS_COMPANY_SUM_PATTERN``, 질문 문구만
    본다)이 그 뒤를 그대로 잇는다.

    Fail-closed: 항목이 정확히 둘이 아니거나, 각각 닫힌 단일 scalar
    retrieve가 아니거나, 회사가 둘 다 유일하게 잡히지 않거나, 개념·기간·
    scope가 다르거나, 질문에 「합계/합산」이 없으면 손대지 않는다. 실제
    덧셈 자체는 여전히 하류(``_payload_for`` 의 ``cross_company_sum`` 판정)
    가 한다 — 여기서는 값을 만들지 않는다.
    """

    def __init__(self, canonical: Any) -> None:
        if not callable(getattr(canonical, "resolve_company", None)):
            raise TypeError(
                "cross-company sum merge regrounder에는 company resolver가 "
                "필요합니다")
        self.canonical = canonical

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if not isinstance(question, str) or not question.strip():
            return intent
        if (len(intent.answer_items) != 2 or intent.answer_groups
                or intent.premises or intent.unresolved_mentions):
            return intent
        if _CROSS_COMPANY_SUM_CUE.search(question) is None:
            return intent
        items = list(intent.answer_items)
        by_id = {entity.entity_id: entity for entity in intent.entities}
        resolved: list[tuple[Any, Any]] = []
        for item in items:
            if (item.target.kind != "metric" or item.operation != "retrieve"
                    or item.selection is not None
                    or item.output.shape != "scalar"
                    or item.output.projection_mode != "named_fields"
                    or len(item.output.field_surfaces) != 1
                    or len(item.target.entity_refs) != 1):
                return intent
            entity = by_id.get(item.target.entity_refs[0])
            if entity is None or entity.kind_hint != "company":
                return intent
            rows = list(self.canonical.resolve_company(entity.surface) or ())
            if len(rows) != 1:
                return intent
            resolved.append((entity, rows[0]))
        corp_codes = {row.corp_code for _entity, row in resolved}
        if len(corp_codes) != 2:
            return intent

        from agent.stage1_v1_financial_backend import _bare_concept_surface

        def _concept(surface: str) -> Any | None:
            return (resolve_metric_concept(surface)
                    or resolve_metric_concept(_bare_concept_surface(surface)))

        concepts = {_concept(item.target.surface) for item in items}
        if len(concepts) != 1 or None in concepts:
            return intent
        periods = {tuple(item.scope.target_period_expressions) for item in items}
        if len(periods) != 1 or not next(iter(periods)):
            return intent
        scopes = {
            tuple(sorted(item.scope.scope_qualifier_expressions))
            for item in items}
        if len(scopes) != 1:
            return intent

        (entity0, _row0), (entity1, _row1) = resolved
        pos0 = question.find(entity0.surface)
        pos1 = question.find(entity1.surface)
        if pos0 < 0 or pos1 < 0:
            return intent
        ordered_ids = (
            [entity0.entity_id, entity1.entity_id] if pos0 <= pos1
            else [entity1.entity_id, entity0.entity_id])

        payload = intent.model_dump(mode="python", warnings=False)
        merged = payload["answer_items"][0]
        merged["item_id"] = "item-1"
        merged["target"]["entity_refs"] = ordered_ids
        payload["answer_items"] = [merged]
        return SemanticIntent.model_validate(payload, strict=True)


#: 「2025년 1월부터 12월까지」「2025년 1월~12월」— 1~12월 **전체** 범위일
#: 때만 받는다. 부분 범위(「1월부터 6월까지」등)는 grounding이 안전하게
#: 표현할 수 있는 리터럴이 질문에 없어(실제 반기 등을 가리키는 표현
#: 자체가 없다) 이 regrounder의 범위 밖으로 남겨 종전대로 거절/한계로
#: 둔다 — #171 M24.
_FULL_YEAR_MONTH_RANGE = re.compile(
    r"(20[0-9]{2})년\s*1\s*월\s*(?:부터|~|-|∼)\s*12\s*월(?:\s*까지)?")


class FullYearMonthRangeRegrounder:
    """「2025년 1월부터 12월까지」류 1~12월 전체 범위 표현을 연간으로 되돌린다.

    이슈 #171 M24. HCX는 이 범위를 target_period_expressions에 조각(「2025년
    1월」·「12월」)으로 쪼개 넣는다(실호출로 직접 확인함) — 그중 「2025년
    1월」은 `agent.planning._financial_period`가 아는 분기/반기/9개월
    어휘가 아니라 파싱에 실패해 unsupported_semantic_target으로 닫힌다.
    1~12월 전체 범위는 그냥 그 해 연간과 같은 뜻이므로, 이미 grounding을
    통과한 「NNNN년」부분 문자열로 표현을 되돌린다.

    Fail-closed: 회사가 둘 이상, 이미 닫힌 단일 scalar 항목이 아니거나,
    현재 기간 표현 중 이 범위 리터럴의 부분문자열이 아닌 것이 섞여 있으면
    (다른 기간이 별도로 있다는 뜻이므로) 손대지 않는다. 개념을 못 찾아도
    손대지 않는다.
    """

    def __init__(self, canonical: Any) -> None:
        if not callable(getattr(canonical, "resolve_company", None)):
            raise TypeError(
                "full year month range regrounder에는 company resolver가 "
                "필요합니다")
        self.canonical = canonical

    @staticmethod
    def _closed_single_item(intent: SemanticIntent) -> Any | None:
        if (len(intent.answer_items) != 1 or intent.answer_groups
                or intent.premises or intent.unresolved_mentions):
            return None
        item = intent.answer_items[0]
        if (item.target.kind != "metric" or item.operation != "retrieve"
                or item.selection is not None
                or item.output.shape != "scalar"
                or item.output.projection_mode != "named_fields"
                or len(item.output.field_surfaces) != 1):
            return None
        return item

    def _single_company(self, item: Any, intent: SemanticIntent) -> bool:
        by_id = {entity.entity_id: entity for entity in intent.entities}
        companies = []
        for ref in item.target.entity_refs:
            entity = by_id.get(ref)
            if entity is None or entity.kind_hint != "company":
                continue
            rows = list(self.canonical.resolve_company(entity.surface) or ())
            if len(rows) == 1:
                companies.append(rows[0])
        corp_codes = {row.corp_code for row in companies}
        return len(companies) == 1 and len(corp_codes) == 1

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if not isinstance(question, str) or not question.strip():
            return intent
        match = _FULL_YEAR_MONTH_RANGE.search(question)
        if match is None:
            return intent
        item = self._closed_single_item(intent)
        if item is None:
            return intent
        if not self._single_company(item, intent):
            return intent
        literal = f"{match.group(1)}년"
        current = list(item.scope.target_period_expressions)
        if current == [literal]:
            return intent
        span = match.group(0)
        if current and not all(value in span for value in current):
            return intent  # 이 범위 밖의 다른 기간이 이미 있다 — 손 떼야 한다.

        from agent.stage1_v1_financial_backend import _bare_concept_surface

        if (resolve_metric_concept(item.target.surface) is None
                and resolve_metric_concept(
                    _bare_concept_surface(item.target.surface)) is None):
            return intent
        payload = intent.model_dump(mode="python", warnings=False)
        payload["answer_items"][0]["scope"]["target_period_expressions"] = [literal]
        return SemanticIntent.model_validate(payload, strict=True)


class ExplicitPeriodsFanoutRegrounder:
    """되묻지 않고 형제마다 **이미 서로 다른** 리터럴 기간을 낸 재무
    fanout을 받아들인다.

    이슈 #171 M16 — 「삼성전자의 2023년, 2024년, 2025년 연결 매출액을
    각각 알려줘」처럼 HCX가 같은 회사·같은 표면을 놓고 형제 item마다 이미
    확정된(질문에 리터럴로 있는) 기간을 낼 때가 있다. 연간·분기가 섞이기도
    한다(예: 2024년/2025년/2026년 1분기).
    `agent.summary_metric_fanout.SummaryMetricFanoutRegrounder`(개념
    fanout, #94 25)가 여는 유일한 3-item 재무 handler는 형제 전원 동일
    기간을 요구해 이 모양을 거절하고,
    `agent.recent_periods_fanout.RecentPeriodsFanoutRegrounder`(#58
    2단계)는 정반대로 형제 전원이 같은 **상대** 기간 표현(「최근 N년」)
    이어야 편다 — 이 fanout은 리터럴이 형제마다 이미 다르므로 둘 다와
    다르다.

    각 항목이 이미 자기 좌표를 세울 리터럴을 갖고 있으므로 이 regrounder는
    아무 값도 계산하지 않는다(canonical 조회가 필요 없다) — 형제 모양이
    **이미 닫혀 있는지**만 구조로 확인하고 ``intent.presentation`` 을
    ``"list"`` 로 못박는다.
    `agent.deterministic_plan_compiler_v1`의 handler 선택은 구조 서명
    (top-level presentation 포함)만으로 라우팅하므로, 그러지 않으면 이
    fanout이 위 두 규격과 같은 서명을 등록해 컴파일이 ambiguous로
    거절한다(#58 2단계와 같은 이유, 직접 확인함).
    """

    @staticmethod
    def _closed_items(intent: SemanticIntent) -> "tuple[Any, ...] | None":
        from agent.recent_periods_fanout import _fanout_shaped_item

        items = tuple(intent.answer_items)
        if (
                len(items) < 2 or len(items) > 5
                or intent.answer_groups or intent.premises
                or intent.unresolved_mentions
                or intent.presentation != "auto"
        ):
            return None
        if not all(_fanout_shaped_item(item) for item in items):
            return None
        return items

    def __call__(self, question: str, intent: SemanticIntent) -> SemanticIntent:
        items = self._closed_items(intent)
        if items is None:
            return intent
        companies = [row for row in intent.entities if row.kind_hint == "company"]
        if len(companies) != 1:
            return intent
        company = companies[0]
        first = items[0]
        if (
                first.target.entity_refs != [company.entity_id]
                or len(first.scope.target_period_expressions) != 1
        ):
            return intent
        period_keys = {tuple(first.scope.target_period_expressions)}
        for item in items[1:]:
            if (
                    item.target.surface != first.target.surface
                    or item.target.entity_refs != first.target.entity_refs
                    or item.target.qualifier_surfaces
                    != first.target.qualifier_surfaces
                    or item.operation != first.operation
                    or len(item.scope.target_period_expressions) != 1
                    or item.scope.as_of_expression != first.scope.as_of_expression
                    or item.scope.document_group_expression
                    != first.scope.document_group_expression
                    or item.scope.scope_qualifier_expressions
                    != first.scope.scope_qualifier_expressions
                    or item.output.shape != first.output.shape
                    or item.output.field_surfaces != first.output.field_surfaces
                    or item.output.presentation != first.output.presentation
            ):
                return intent
            period_key = tuple(item.scope.target_period_expressions)
            if period_key in period_keys:
                return intent
            period_keys.add(period_key)
        payload = intent.model_dump(mode="python", warnings=False)
        payload["presentation"] = "list"
        return SemanticIntent.model_validate(payload, strict=True)


__all__ += ["build_stage1_v1_runtime"]
