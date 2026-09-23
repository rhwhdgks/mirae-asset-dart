"""상대 기간(「직전 분기」「전전 분기」「전년」) 재무 기간 regrounder.

이슈 #58. `agent/exact_periodic_filing.py`의 ``ReportFormFinancialPeriodRegrounder``
와 같은 계열이다 — 표면형만 재정렬(rewrite)하고, 실제 연도·분기·계정·scope
확정은 여전히 `agent.stage1_v1_financial_backend`(``_one_coordinate``)가 한다.
이 파일의 리터럴 표현과 그 쪽의 정규식은 짝을 이루므로 한쪽만 바꾸면 반드시
함께 바꿔야 한다.

**「직전 분기」의 정의(루브릭 통일, #58 2단계).** 기준일(코퍼스 컷오프)
시점에 「직전 분기」는 최신 실제 분기에서 한 분기 더 앞이 아니라, **정본에
실제로 있는 가장 최근 분기 그 자체**다 — 「마지막으로 끝나고 보고된 분기」를
가리키는 일상어라서다(예: 컷오프 2026-06-19, 최신 실제 분기 2026 1Q →
「직전 분기」= 2026 1Q). 「전전 분기」는 그 한 분기 더 앞(2025 4Q)이며, 옛
「직전 분기」 정의를 그대로 잇는다.  실제 오프셋 계산은
`agent.stage1_v1_financial_backend._relative_quarter_offset`가 한다.

**「최근 N년」「최근 N개 분기」를 여기 넣지 않은 이유.**
``validate_semantic_intent_grounding``(agent/stage1_v1_resolver.py)은
``target_period_expressions`` 원소 하나하나가 질문의 리터럴 부분문자열이기를
요구한다.  「최근 3년」의 실제 연도(2023·2024·2025년)는 질문에 리터럴로 없으므로
그 자체를 목록으로 펼칠 수 없고, 우회하려면 answer_item 하나를 N개(각각 같은
리터럴 「최근 3년」을 재사용하고 item 순번만으로 실제 연도를 가리키는)로
늘려야 한다 — grounding은 표현이 질문에 있는지만 보고 몇 번 재사용됐는지는
따지지 않으므로 그 자체는 가능하다.  하지만 그렇게 늘어난 "같은 회사·다른
연도의 독립된 ``financial`` 사실 N개" 라는 항목 구조를, 결정론적 plan
compiler(`agent.deterministic_plan_compiler_v1`)의 닫힌 handler
registry(``STAGE1_V1_HANDLER_REGISTRY``)가 아직 모른다 — 등록된
``("financial", "financial")`` 쌍은 CFS/SFS scope 짝(같은 기간, 다른
scope)만 받고, 기간이 다른 독립 사실 나열을 받는 handler는 없다.  그 항목을
만들어 넘기면 컴파일이 `DeterministicPlanCompilerError`로 막혀 질문 전체가
기술 오류로 깨진다(직접 확인함).  이를 받으려면 새 structural-signature
registry 항목(N=2·3)과 ``validate_intent``/``validate_resolution``/``lower``
세 함수를 그 compiler에 추가해야 하는데, 이 정본 배치의 안전 범위(스키마
불변, 컴파일러 핵심부 최소 변경)를 벗어나 별도 이슈로 미룬다.

단일 상대 기간(「직전 분기」「전전 분기」「전년」)은 이미 등록된 단일-
``financial`` 경로(``("financial",)``)를 그대로 타므로 안전하다.  이 셋은
다중 기간이 아니라 표면형 하나를 다른 표면형 하나로 바꾸는 rewrite다.

**「전년」을 왜 여기서 다루는가.** ``agent.semantic_intent_v1_boundary`` 의
``_PERIOD`` 는 「작년」「지난해」「올해」「금년」「재작년」을 target
qualifier에서 scope로 승격하지만 「전년」은 그 목록에 없다 — 그 경계
정규화기를 건드리는 대신, 이 regrounder가 원 질문을 직접 훑어(다른
question-grounded regrounder들과 같은 방식) 리터럴 「전년」을 그대로
``target_period_expressions`` 에 넣는다. 실제 연도(reference_date 기준
전년)는 `agent.planning._financial_period` 가 안다(이 배치에서 relative 목록에
추가).  「전년대비/전년비」(이미 지원되는 compare 축의 qualifier)는 잡지
않는다 — 그건 기간이 아니라 비교 요구다.
"""

from __future__ import annotations

import re
from typing import Any

from .planning import resolve_metric_concept
from .semantic_intent_v1 import SemanticIntent
from .stage1_v1_financial_backend import _bare_concept_surface


#: 「직전 분기」— 단일 상대 분기(확장 없음). 정본의 최신 실제 분기 그
#: 자체로 정한다(`agent.stage1_v1_financial_backend._relative_quarter_offset`
#: — offset 0). 「전전 분기」와 겹치지 않도록 그 표현을 먼저 시도한다.
#: 「최근 분기」「가장 최근 분기」「최신 분기」도 일상어로는 같은 뜻이다
#: (#171 M14) — 「직전 분기」와 정의를 통일한다(offset 0).
PREVIOUS_QUARTER = re.compile(
    r"직전\s*분기(?:\s*말)?|바로\s*전\s*분기|"
    r"(?:가장\s*)?최근\s*분기(?:\s*말)?|최신\s*분기(?:\s*말)?")
#: 「전전 분기」— 정본의 최신 실제 분기에서 한 분기 더 앞(offset 1). 옛
#: 「직전 분기」 정의를 그대로 잇는다.
PREVIOUS_PREVIOUS_QUARTER = re.compile(r"전전\s*분기(?:\s*말)?")
#: 「전년」— 단일 상대 연도(확장 없음). 「전년대비/전년비」는 비교 축의
#: qualifier이지 기간이 아니므로 제외한다.
PREVIOUS_YEAR = re.compile(r"전년(?!\s*(?:대비|보다|비))")


class RelativeFinancialPeriodRegrounder:
    """「직전 분기」「전전 분기」「전년」을 재무 기간 좌표가 받는 리터럴 표현으로 되돌린다.

    Fail-closed: 회사가 둘 이상, 개념이 안 잡힘, item이 이미 닫힌 단일
    scalar 모양이 아니거나 다른 기간이 이미 있으면 손대지 않는다.
    """

    def __init__(self, canonical: Any) -> None:
        if not callable(getattr(canonical, "resolve_company", None)):
            raise TypeError(
                "relative financial period regrounder에는 company resolver가 필요합니다")
        self.canonical = canonical

    #: HCX가 「최근 분기」「최신 분기」의 「최근/최신」을 selection 축(latest)
    #: 으로도 겹쳐 뽑을 수 있다(SG-013류와 같은 wire 실호출로 확인함,
    #: 삼성전자 「최근 분기 연결 매출액」— ``selection={mode:latest,
    #: criterion_surface:"최근"}``). 그 selection은 이 표현이 이미 기간
    #: 축으로 옮기는 것과 같은 뜻이므로(다른 대상을 고르는 게 아니라 이미
    #: 하나뿐인 회사·개념의 "가장 최근" 시점을 가리킬 뿐이다) no-op으로
    #: 지우고 지나간다. 그 밖의 selection(선택 기준이 다르거나 k가 있는
    #: 등)은 여전히 fail-closed로 손대지 않는다.
    _LATEST_SELECTION_CUES = frozenset({"최근", "최신", "가장 최근"})

    @classmethod
    def _closed_single_item(cls, intent: SemanticIntent) -> Any | None:
        if (len(intent.answer_items) != 1 or intent.answer_groups
                or intent.premises or intent.unresolved_mentions):
            return None
        item = intent.answer_items[0]
        if (item.target.kind != "metric" or item.operation != "retrieve"
                or item.output.shape != "scalar"
                or item.output.projection_mode != "named_fields"
                or len(item.output.field_surfaces) != 1):
            return None
        if item.selection is not None and not (
                item.selection.mode == "latest"
                and item.selection.criterion_surface
                in cls._LATEST_SELECTION_CUES):
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
        # 「전전 분기」를 먼저 본다 — 「직전 분기」 정규식이 「전전 분기」
        # 안의 「전 분기」 조각과 겹치지 않지만(문자열 자체가 다르다), 순서를
        # 명시해 두 표현이 앞으로 겹칠 여지를 남기지 않는다.
        match = PREVIOUS_PREVIOUS_QUARTER.search(question)
        if match is None:
            match = PREVIOUS_QUARTER.search(question)
        if match is None:
            match = PREVIOUS_YEAR.search(question)
        if match is None:
            return intent
        item = self._closed_single_item(intent)
        if item is None:
            return intent
        if not self._single_company(item, intent):
            return intent
        # HCX가 상대 기간 표현을 target surface 안에 그대로 접어 넣을 수
        # 있다(「직전 분기 연결 영업이익」— 카카오 SG-013). 이 개념 사전은
        # 「직전 분기」를 모르니 그 표면 그대로는 못 찾는다.
        # ``_bare_concept_surface``(agent.stage1_v1_financial_backend)가
        # 정확히 이 접두어들을 벗기는 helper이고, 아래 `_one_coordinate`도
        # 개념을 못 찾으면 같은 helper로 재시도하므로 여기서 같은 fallback을
        # 써야 gate가 하류와 일치한다.
        if (resolve_metric_concept(item.target.surface) is None
                and resolve_metric_concept(
                    _bare_concept_surface(item.target.surface)) is None):
            return intent
        literal = match.group(0)
        if list(item.scope.target_period_expressions) == [literal]:
            return intent
        if item.scope.target_period_expressions:
            return intent  # 이미 다른 기간이 있으면 되묻지 않고 손 뗀다.
        payload = intent.model_dump(mode="python", warnings=False)
        payload["answer_items"][0]["scope"]["target_period_expressions"] = [literal]
        if item.selection is not None:
            # 「최근/최신」 selection 축은 이 기간 표현과 같은 뜻이었을
            # 뿐이다 — 위 ``_closed_single_item`` 이 이미 그 no-op 모양만
            # 통과시켰으므로 여기서 지운다.
            payload["answer_items"][0]["selection"] = None
        return SemanticIntent.model_validate(payload, strict=True)
