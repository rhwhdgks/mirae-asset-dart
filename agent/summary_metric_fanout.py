"""후보 집합 자체가 「대표 요약」인 구어를 되묻지 않고 **전부 답한다**.

이슈 #94 25. 「삼성전자가 작년에 얼마나 벌었어?」는 매출액·영업이익·당기순이익
중 무엇이냐고 되물었다. 그 셋은 손익계산서를 위에서 아래로 읽은 세 단계라
**나란히 주는 것이 고르라고 묻는 것보다 낫다.** 「빚」(총부채·유동·비유동)도
총계와 그 분해라 같다.

어느 표기를 펴는지는 코드가 아니라 승인 표가 정한다
(`agent/concept_question_patterns.tsv` 9열 · `ConceptQuestionPattern.fanout`).
「현금」·「투자」는 후보에 잔액과 흐름이 섞여 있어 켜지 않았다 — 잔액을 물은
사람에게 흐름 셋을 얹으면 묻지 않은 값을 답하는 것이 된다.

**펴는 자리는 둘로 나뉜다.**

``SummaryMetricFanoutRegrounder``
    item 하나를 **후보 수만큼** 복제한다. 표면은 넷 다 그대로 「벌었어」다 —
    개념 이름(「매출액」)을 표면에 적으면 질문에 없는 낱말이라 결속 검사가
    거절한다(`stage1_v1_resolver` 의 「원 질문에 결속되지 않았습니다」).

``agent.stage1_v1_financial_backend``
    같은 표면 item N개를 보고 **형제 순서로** 후보를 나눠 맡긴다. 후보를
    전부 답하므로 순서가 바뀌어도 값이 틀리지 않고 줄 순서만 바뀐다.

**전부 풀릴 때만 편다.** 재무 백엔드는 item 하나라도 좌표를 못 세우면 통째로
물러난다(`financial_coordinates` 의 ``if row is None: return ()``). 셋 중
하나가 없는 회사·기간에서 그냥 펴면 되묻기 대신 답이 **아예 사라진다** —
오늘보다 나빠지는 유일한 길이다. 그래서 regrounder 가 펴기 전에 후보를 모두
좌표로 세워 보고, 하나라도 실패하면 손대지 않는다(되묻기가 그대로 남는다).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from agent.concept_alias import normalize_surface_key
from agent.semantic_intent_v1 import AnswerItem, SemanticIntent

__all__ = [
    "SummaryMetricFanout",
    "SummaryMetricFanoutRegrounder",
    "fanout_concepts_by_sibling",
    "summary_metric_fanout",
]


@dataclass(frozen=True, slots=True)
class SummaryMetricFanout:
    """펼 수 있는 구어 하나와 그 후보 개념들."""

    surface: str
    pattern_id: str
    concepts: tuple[Any, ...]


def _fanout_shaped_item(item: Any) -> bool:
    """이 item 이 **값 하나를 묻는 닫힌 재무 모양**인가.

    복제해도 뜻이 유지되는 모양만 편다. 선택자(`selection`)나 비교 연산이
    붙은 item 은 후보마다 되풀이해도 같은 뜻이 되지 않으므로 제외한다.
    """

    return (
        item.target.kind == "metric"
        and item.operation == "retrieve"
        and item.selection is None
        and item.output.shape == "scalar"
        and item.output.projection_mode == "named_fields"
        and len(item.output.field_surfaces) == 1
    )


def summary_metric_fanout(
        question: str, item: Any) -> "SummaryMetricFanout | None":
    """이 item 을 후보만큼 펴도 되는가. 아니면 ``None``.

    **승인 표가 켠 표기만** 편다. 표기가 item 의 것과 같아야 한다 — 질문
    어딘가에 「벌었어」가 있다는 것만으로는 부족하고, 모델이 그 말을 이
    item 의 대상으로 읽었어야 한다.
    """

    from agent.concept_alias import resolve_from_question
    from agent.planning import CONCEPT_QUESTION_PATTERNS

    if not isinstance(question, str) or not question.strip():
        return None
    if not _fanout_shaped_item(item):
        return None
    outcome = resolve_from_question(question, CONCEPT_QUESTION_PATTERNS)
    if (outcome.status != "ambiguous" or not outcome.fanout
            or not isinstance(outcome.surface, str)
            or len(outcome.candidates) < 2):
        return None
    if (normalize_surface_key(outcome.surface)
            != normalize_surface_key(item.target.surface)):
        return None
    return SummaryMetricFanout(
        surface=outcome.surface,
        pattern_id=str(outcome.pattern_id),
        concepts=tuple(outcome.candidates),
    )


class SummaryMetricFanoutRegrounder:
    """되묻던 요약 구어 item 하나를 후보 수만큼 편다.

    Fail-closed: item 이 하나가 아니거나, 승인 표가 켜지 않은 표기거나,
    후보 중 **하나라도** 좌표를 세우지 못하면 intent 를 그대로 돌려준다 —
    그러면 종전대로 `FinancialResolutionBackend` 가 되묻는다.
    """

    def __init__(
            self, canonical: Any, *, reference_date: date, corpus_cutoff: str,
            scope_authority: Any = None,
            ) -> None:
        if not callable(getattr(canonical, "resolve_company", None)):
            raise TypeError(
                "summary metric fanout regrounder에는 company resolver가 "
                "필요합니다")
        if not isinstance(reference_date, date):
            raise TypeError(
                "summary metric fanout regrounder에는 reference_date가 "
                "필요합니다")
        if not isinstance(corpus_cutoff, str) or not corpus_cutoff:
            raise TypeError(
                "summary metric fanout regrounder에는 corpus_cutoff가 "
                "필요합니다")
        self.canonical = canonical
        self.reference_date = reference_date
        self.corpus_cutoff = corpus_cutoff
        self.scope_authority = (
            canonical if scope_authority is None else scope_authority)

    #: 후보 좌표가 **개념 말고는 전부 같아야 하는** 축. 컴파일러의
    #: `_validate_summary_metric_fanout_resolution` 이 뒤에서 같은 것을
    #: 검사하므로, 여기서 먼저 걸러야 그 검사가 컴파일 오류로 터지지 않는다
    #: — 오류는 되묻기조차 못 하고 질문을 깨뜨린다.
    _SHARED_AXES = (
        "corp_code", "corp_name", "period_start", "period_end",
        "period_type", "cumulative", "scope", "statement",
    )

    def _every_concept_closes(
            self, question: str, intent: SemanticIntent, item: Any,
            concepts: tuple[Any, ...],
            ) -> bool:
        """후보가 **전부** 같은 자리에 좌표를 세우는가.

        하나라도 못 세우거나 개념 밖의 축이 갈리면 펴지 않는다 — 그대로
        두면 종전처럼 되묻는다.
        """

        from agent.stage1_v1_financial_backend import financial_coordinates

        built = []
        for concept in concepts:
            rows = financial_coordinates(
                item, intent=intent, companies=self.canonical,
                reference_date=self.reference_date,
                corpus_cutoff=self.corpus_cutoff,
                scope_authority=self.scope_authority, question=question,
                concept_override=concept)
            # 좌표는 회사·개념·기간이 모두 서야 나온다. 후보마다 회사·기간이
            # 같으므로 여기서 갈리는 것은 개념뿐이지만, 그것을 짐작하지 않고
            # 실제로 세워 본다.
            if len(rows) != 1:
                return False
            built.append(rows[0])
        first = built[0]
        if any(getattr(row, axis) != getattr(first, axis)
               for row in built[1:] for axis in self._SHARED_AXES):
            return False
        return len({row.concept for row in built}) == len(built)

    @staticmethod
    def _supersedes_unresolved(intent: SemanticIntent, item: Any) -> bool:
        """미해소 언급이 **바로 이 표기 하나뿐**인가.

        모델은 「벌었어」를 target 으로 적으면서 동시에 「벌었어를 해소하지
        못했다」고 표시할 수 있다(R-A-001 의 기록된 wire 가 그렇다). 그 표시는
        모델이 개념을 못 골랐다는 뜻이고, 승인 층은 **고를 수 있다** — 후보를
        알고 있으므로 셋 다 답한다.

        그래서 그 언급 하나는 여기서 걷어낸다. 그것 말고 다른 미해소가 있으면
        (다른 item, 다른 표기, 다른 역할) 손대지 않는다 — 우리가 답하지 못하는
        것이 남아 있다는 뜻이라 조용히 지우면 묻지 않은 답을 내게 된다.
        """

        mentions = intent.unresolved_mentions
        if not mentions:
            return True
        if len(mentions) != 1:
            return False
        mention = mentions[0]
        return (
            mention.role_hint == "target"
            and list(mention.applies_to_item_ids) == [item.item_id]
            and (normalize_surface_key(mention.raw_text)
                 == normalize_surface_key(item.target.surface))
        )

    def __call__(
            self, question: str, intent: SemanticIntent) -> SemanticIntent:
        if (len(intent.answer_items) != 1 or intent.answer_groups
                or intent.premises):
            return intent
        item = intent.answer_items[0]
        if not self._supersedes_unresolved(intent, item):
            return intent
        fanout = summary_metric_fanout(question, item)
        if fanout is None:
            return intent
        if not self._every_concept_closes(
                question, intent, item, fanout.concepts):
            return intent
        items = [
            AnswerItem(
                item_id=f"item-{index}",
                target=item.target.model_copy(deep=True),
                operation=item.operation,
                scope=item.scope.model_copy(deep=True),
                selection=None,
                output=item.output.model_copy(deep=True),
            )
            for index in range(1, len(fanout.concepts) + 1)
        ]
        return SemanticIntent(
            schema_version=intent.schema_version,
            entities=intent.entities,
            answer_items=items, answer_groups=[], premises=[],
            unresolved_mentions=[], presentation=intent.presentation,
        )


def _sibling_shape(item: Any) -> Any:
    """item_id 를 뺀 모양. 형제인지 볼 때 쓴다."""

    payload = item.model_dump(mode="json")
    payload.pop("item_id", None)
    return payload


def fanout_concepts_by_sibling(
        question: str, intent: SemanticIntent) -> "tuple[Any | None, ...]":
    """형제 순서로 나눠 맡을 후보 개념. 펼 모양이 아니면 전부 ``None``.

    `SummaryMetricFanoutRegrounder` 가 편 intent 를 재무 백엔드가 다시 알아보는
    자리다. 표면이 넷 다 같으므로 **순서 말고는 구분할 것이 없다** — 후보를
    전부 답하기 때문에 순서가 바뀌어도 값이 틀리지 않고 줄 순서만 바뀐다.

    형제가 **완전히 같은 모양**일 때만 답한다. 하나라도 기간·회사·출력이
    다르면 그것은 사용자가 서로 다른 것을 물은 것이지 편 결과가 아니다.
    """

    items = list(intent.answer_items)
    empty = (None,) * len(items)
    if (len(items) < 2 or intent.answer_groups or intent.premises
            or intent.unresolved_mentions):
        return empty
    fanout = summary_metric_fanout(question, items[0])
    if fanout is None or len(fanout.concepts) != len(items):
        return empty
    first = _sibling_shape(items[0])
    if any(_sibling_shape(row) != first for row in items[1:]):
        return empty
    return fanout.concepts
