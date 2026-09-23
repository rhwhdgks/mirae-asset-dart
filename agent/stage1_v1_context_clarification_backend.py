"""Typed first-turn clarification for questions with missing context.

This module only emits user-answerable slots. Accepted values are kept in the
digest-bound clarification sidecar and resolved by the matching resume backend;
they never become an untracked JSON patch to the original ``SemanticIntent``.
"""

from __future__ import annotations

from datetime import date
import re
from typing import Any

from .event_clarification_options_v1 import build_event_clarification_options
from .planning import held_metric_candidates, resolve_metric_concept
from .semantic_intent_v1 import SemanticIntent
from .stage1_v1_resolver import (
    ClarificationAuthority,
    ClarificationOption,
    ResolutionAuthority,
)


def _empty_coordinates(item: Any) -> bool:
    return (
        item.scope.as_of_expression is None
        and item.scope.document_group_expression is None
        and not item.scope.target_period_expressions
        and not item.scope.scope_qualifier_expressions
        and item.selection is None
        and not item.target.qualifier_surfaces
    )


def _empty_scope(item: Any) -> bool:
    return _empty_coordinates(item) and not item.target.entity_refs


#: 개념 표시 표기는 ``agent.planning`` 이 정본 계정 사전으로 검증한 하나의 표를
#: 쓴다. 여기 네 개만 두었을 때는 나머지 개념이 식별자로 노출됐다.
def _metric_label(concept: str) -> str:
    from agent.planning import concept_display_name
    return concept_display_name(concept)


def _authority(*, item_id: str, slots: list[dict[str, Any]]) -> ClarificationAuthority:
    return ClarificationAuthority.model_validate({
        "kind": "clarification",
        "slots": [
            {
                "slot_id": f"slot-{index}",
                "role_hint": row["role_hint"],
                "reason_code": row.get("reason_code", "missing_context"),
                "response_kind": row.get("response_kind", "provide_value"),
                "prompt": row["prompt"],
                "applies_to_item_ids": [item_id],
                # There is no source-grounded mention to erase: these are
                # deliberately first-turn context axes, not JSON-patch paths.
                "mention_ids": [],
                "options": row.get("options", []),
            }
            for index, row in enumerate(slots, start=1)
        ],
    }, strict=True)


class MissingContextClarificationBackend:
    """Fail closed on missing event/comparison coordinates.

    The two shapes are narrow enough that an ordinary metric or event query
    cannot be silently converted to a clarification.  No question ID, answer
    fixture, or user literal selects a branch.
    """

    def __init__(
            self, canonical: Any | None = None, *,
            corpus_cutoff: str | None = None,
            event_preflight: Any | None = None,
            canonical_build_id: str | None = None,
            resolver_version: str | None = None,
            reference_date: date | None = None,
            ) -> None:
        self.canonical = canonical
        self.corpus_cutoff = corpus_cutoff
        self.event_preflight = event_preflight
        self.canonical_build_id = canonical_build_id
        self.resolver_version = resolver_version
        self.reference_date = reference_date

    @staticmethod
    def _entity_surface(intent: SemanticIntent, item: Any) -> str | None:
        by_id = {entity.entity_id: entity for entity in intent.entities}
        surfaces = [
            by_id[ref].surface for ref in item.target.entity_refs
            if ref in by_id and by_id[ref].kind_hint == "company"
        ]
        return surfaces[0] if len(surfaces) == 1 else None

    def _contract_event_lookup(
            self, source_intent: SemanticIntent, item: Any,
            ) -> Any | None:
        company = self._entity_surface(source_intent, item)
        if (company is None or self.canonical is None
                or self.corpus_cutoff is None):
            return None
        return build_event_clarification_options(
            self.canonical, company_surface=company,
            as_of=self.corpus_cutoff, target_surface="계약",
            preflight=self.event_preflight,
            corpus_cutoff=self.corpus_cutoff,
        )

    def _latest_event_authority(
            self, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> ResolutionAuthority | None:
        """「가장 최근 건」은 되묻지 않고 **가장 새 것을 골라 답한다.**

        후보가 여럿이어도 「최근」이 그중 하나를 정한다.  여기서 되묻는 것은
        사용자가 이미 준 기준을 무시하는 것이다.  Gold 도 같은 정책을 적어
        두었다 — 「'최근'은 역질문 아님 — 기본값+명시」.

        고르는 값은 canonical 접수번호이고, 접수번호는 앞자리가 날짜라 최대값이
        가장 새 건이다.  지어내는 값이 없다.
        """

        if (self.canonical is None or self.corpus_cutoff is None
                or self.canonical_build_id is None
                or self.resolver_version is None
                or self.reference_date is None
                or len(source_intent.answer_items) != 1
                or source_intent.answer_groups
                or source_intent.premises
                or source_intent.unresolved_mentions):
            return None
        item = source_intent.answer_items[0]
        selection = item.selection
        if (selection is None or getattr(selection, "mode", None) != "latest"
                or item.operation != "retrieve"
                or item.target.kind != "event"
                or item.output.projection_mode != "named_fields"
                or not item.output.field_surfaces
                or item.scope.as_of_expression is not None
                or item.scope.scope_qualifier_expressions):
            return None
        lookup = self._contract_event_lookup(source_intent, item)
        receipts = (
            [option.value for option in lookup.options
             if isinstance(option.value, str) and option.value.isdigit()]
            if lookup is not None and lookup.status in {"ambiguous", "resolved"}
            else []
        )
        if not receipts:
            # Candidate-option preflight deliberately caps very broad issuer
            # searches.  "latest" still gives a deterministic selector, so
            # enumerate the same canonical event family through the collection
            # selector and keep only its newest root.  The temporary list shape
            # is an internal candidate-universe probe; the emitted authority is
            # still bound to the untouched source intent below.
            from .stage1_v1_event_collection import select_event_collection
            probe_item = item.model_copy(update={
                "selection": None,
                "output": item.output.model_copy(update={"shape": "record_list"}),
            })
            probe_intent = source_intent.model_copy(update={
                "answer_items": [probe_item],
            })
            selected = select_event_collection(
                canonical=self.canonical, intent=probe_intent,
                question=question, reference_date=self.reference_date,
                corpus_cutoff=self.corpus_cutoff,
            )
            if selected is not None:
                receipts = [event.root_receipt for event in selected.events]
        if not receipts:
            return None
        from .stage1_v1_event_clarification_backend import (
            _selected_event_authority,
        )
        # 의도는 **원본 그대로** 넘긴다.  권위는 리졸버가 받은 의도에 결속되므로
        # 여기서 손대면 요약이 어긋난다.  「최근」이 해소됐다는 사실은 고른
        # 접수번호가 말한다.
        try:
            return _selected_event_authority(
                self.canonical,
                question_id=question_id,
                source_intent=source_intent,
                receipt=max(receipts),
                canonical_build_id=self.canonical_build_id,
                resolver_version=self.resolver_version,
                reference_date=self.reference_date,
                corpus_cutoff=self.corpus_cutoff,
            )
        except Exception:                                  # noqa: BLE001
            # 증명하지 못하면 물러난다.  다음 백엔드가 판단한다.
            return None

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> ResolutionAuthority | None:
        latest = self._latest_event_authority(
            question_id, question, source_intent)
        if latest is not None:
            return latest
        if (len(source_intent.answer_items) != 1
                or source_intent.answer_groups
                or source_intent.premises
                or source_intent.unresolved_mentions):
            return None
        item = source_intent.answer_items[0]
        if (
                item.item_id != "item-1"
                or item.operation not in {"retrieve", "compare"}
                or item.output.projection_mode not in {
                    "named_fields", "whole_target"}
                or item.output.shape not in {"scalar", "narrative", "comparison"}
                # 「○○ 상태 어때?」처럼 뽑을 칸을 못 정한 판은 `whole_target`
                # 으로 필드가 비어서 온다.  그것도 좌표가 없다는 신호이므로
                # 입구에서 막지 않고 아래 분기가 판단하게 한다.
                or len(item.output.field_surfaces) != (
                    0 if item.output.projection_mode == "whole_target" else 1)
                or item.output.presentation != "auto"
                or source_intent.presentation != "auto"
                ):
            return None
        normalized = "".join(item.target.surface.split()).casefold()
        is_contract_amount_change = (
            item.target.kind == "metric"
            and _empty_coordinates(item)
            and "계약" in normalized
            and "금액" in normalized
            and any(token in normalized for token in ("변동", "차이", "증감"))
        )
        if is_contract_amount_change:
            return _authority(item_id=item.item_id, slots=[
                {
                    "role_hint": "entity",
                    "prompt": "어느 회사인지 알려주세요.",
                },
                {
                    "role_hint": "event",
                    "prompt": "어느 계약인지 접수번호 또는 계약명을 알려주세요.",
                },
                {
                    "role_hint": "timepoint",
                    "prompt": (
                        "변동을 비교할 두 시점을 알려주세요. "
                        "(예: 2024-12-31, 2025-12-31)"),
                },
                {
                    "role_hint": "value_kind",
                    "prompt": "계약금액과 해지금액 중 무엇을 비교할까요?",
                },
            ])
        # 대상이 **개체 자체**이고 좌표가 하나도 없으면 무엇을 물었는지 정해지지
        # 않는다.  「회사 상태」는 재무일 수도, 공시일 수도, 계약일 수도 있다.
        #
        # 이 백엔드는 사슬 끝이다 — 여기 닿았다는 것은 앞의 모든 백엔드가
        # 증명하지 못했다는 뜻이다.  그러므로 표면 낱말을 세지 않고 구조만 본다.
        # 선택지는 **우리가 실제로 답할 수 있는 것**으로만 둔다.
        if item.target.kind == "entity" and _empty_coordinates(item):
            return _authority(item_id=item.item_id, slots=[{
                "role_hint": "target",
                "prompt": "무엇을 알려드릴까요?",
                "response_kind": "select_one",
                "options": [ClarificationOption(
                    value=value, label=label,
                    proof_refs=[f"policy:entity-aspect-option:{value}"],
                ).model_dump(mode="json") for value, label in (
                    ("financial", "재무 실적 (매출·영업이익·자산)"),
                    ("periodic_document", "최근 정기보고서 사업 내용"),
                    ("contract_event", "계약·해지 공시"),
                )],
            }])

        if item.target.kind == "metric":
            if (
                    item.operation == "compare"
                    and item.output.shape == "comparison"
                    and len(item.target.entity_refs) == 2
                    and not item.scope.target_period_expressions
                    and any(token in normalized for token in ("더커", "누가"))
            ):
                return _authority(item_id=item.item_id, slots=[
                    {
                        "role_hint": "target",
                        "prompt": "어느 재무지표를 비교할까요?",
                        "response_kind": "select_one",
                        "options": [ClarificationOption(
                            value=value, label=_metric_label(value),
                            proof_refs=[f"policy:financial-concept-option:{value}"],
                        ).model_dump(mode="json") for value in (
                            "revenue", "total_assets", "operating_income")],
                    },
                    {
                        "role_hint": "time",
                        "prompt": "어느 기간을 기준으로 비교할까요?",
                    },
                ])
            if item.output.shape != "scalar":
                return None
            is_contract_amount = (
                _empty_coordinates(item)
                and "계약" in normalized
                and "금액" in normalized
            )
            if is_contract_amount:
                company_missing = not item.target.entity_refs
                if company_missing:
                    return _authority(item_id=item.item_id, slots=[
                        {
                            "role_hint": "entity",
                            "prompt": "어느 회사인지 알려주세요.",
                        },
                        {
                            "role_hint": "event",
                            "prompt": "어느 계약인지 알려주세요.",
                        },
                        {
                            "role_hint": "value_kind",
                            "prompt": "계약금액과 해지금액 중 무엇을 확인할까요?",
                        },
                        {
                            "role_hint": "timepoint",
                            "prompt": "어느 시점 기준으로 확인할까요?",
                        },
                    ])
                lookup = self._contract_event_lookup(source_intent, item)
                if lookup is None:
                    return None
                if lookup.status == "resolved" and len(lookup.options) == 1:
                    if (
                            self.canonical_build_id is None
                            or self.resolver_version is None
                            or self.reference_date is None
                            or self.corpus_cutoff is None
                    ):
                        return None
                    from .stage1_v1_event_clarification_backend import (
                        _selected_event_authority,
                    )
                    return _selected_event_authority(
                        self.canonical,
                        question_id=question_id,
                        source_intent=source_intent,
                        receipt=lookup.options[0].value,
                        canonical_build_id=self.canonical_build_id,
                        resolver_version=self.resolver_version,
                        reference_date=self.reference_date,
                        corpus_cutoff=self.corpus_cutoff,
                    )
                if lookup.status == "too_many":
                    # The canonical preflight intentionally does not expose
                    # an unbounded candidate universe.  The missing axis is
                    # still event identity, so ask for a value that can close
                    # it instead of falling through to a generic terminal.
                    return _authority(item_id=item.item_id, slots=[{
                        "role_hint": "event",
                        "reason_code": "event_target_multiple_candidates",
                        "prompt": (
                            "후보 계약이 많습니다. 해당 공시의 접수번호(14자리)를 "
                            "알려주세요. 접수번호가 없으면 정확한 계약명을 "
                            "알려주세요."),
                    }])
                if lookup.status != "ambiguous" or len(lookup.options) < 2:
                    return None
                return _authority(item_id=item.item_id, slots=[{
                    "role_hint": "event",
                    "prompt": "어느 계약을 말씀하시나요?",
                    "response_kind": "select_one",
                    "options": [row.model_dump(mode="json")
                                for row in lookup.options],
                }])

            metric = resolve_metric_concept(item.target.surface)
            candidates = held_metric_candidates(item.target.surface)
            if candidates:
                slots: list[dict[str, Any]] = [{
                    "role_hint": "target",
                    "prompt": "어느 재무지표를 뜻하나요?",
                    "response_kind": "select_one",
                    "options": [ClarificationOption(
                        value=value.value,
                        label=_metric_label(value.value),
                        proof_refs=[
                            f"policy:financial-concept-option:{value.value}"
                        ],
                    ).model_dump(mode="json") for value in candidates],
                }]
                if not item.scope.target_period_expressions:
                    slots.append({
                        "role_hint": "time",
                        "prompt": "어느 기간을 기준으로 확인할까요?",
                    })
                return _authority(item_id=item.item_id, slots=slots)
            if metric is not None:
                company_missing = not item.target.entity_refs
                period_missing = not item.scope.target_period_expressions
                slots = []
                if company_missing:
                    slots.append({
                        "role_hint": "entity",
                        "prompt": "어느 회사인지 알려주세요.",
                    })
                if period_missing:
                    slots.append({
                        "role_hint": "time",
                        "prompt": "어느 기간을 기준으로 확인할까요?",
                    })
                # 회사가 정해져야 primary statement를 판단할 수 있다. 이
                # turn에서는 사용자가 scope까지 원자적으로 확정하게 한다.
                if company_missing:
                    slots.append({
                        "role_hint": "qualifier",
                        "prompt": "연결과 별도 중 어느 기준인가요?",
                        "response_kind": "select_one",
                        "options": [
                            ClarificationOption(
                                value="CFS", label="연결",
                                proof_refs=["policy:financial-scope:CFS"],
                            ).model_dump(mode="json"),
                            ClarificationOption(
                                value="SFS", label="별도",
                                proof_refs=["policy:financial-scope:SFS"],
                            ).model_dump(mode="json"),
                        ],
                    })
                if slots:
                    return _authority(item_id=item.item_id, slots=slots)
            return None
        if not _empty_coordinates(item):
            return None
        if item.target.kind in {"event", "document"}:
            # An event surface may itself carry identity (counterparty,
            # project, failure condition).  Ask only when stripping generic
            # status/reason wording leaves no identifying term.
            identity_terms = re.sub(
                r"계약|공시|정정|변경|바뀐|해지|정상|완료|종료|끝|상태|유효|"
                r"이유|왜|그런|거|임|여부|알려|줘|이|가|은|는|을|를|어|었|났|했|까|요",
                "", normalized,
            )
            if item.target.entity_refs and identity_terms:
                return None
            slots = []
            if not item.target.entity_refs:
                slots.append({
                    "role_hint": "entity", "prompt": "어느 회사인지 알려주세요.",
                })
            prompt = "어느 계약인지 접수번호 또는 계약명을 알려주세요."
            if "정정" in question:
                prompt = (
                    "정정된 계약 공시가 여러 건일 수 있습니다. "
                    "어느 계약인지 계약상대 또는 접수번호를 알려주세요.")
            elif re.search(r"해지|종료|취소|철회|파기", question):
                prompt = (
                    "해지된 계약 공시가 여러 건입니다. "
                    "어느 계약인지 계약상대 또는 접수번호를 알려주세요.")
            slots.append({
                "role_hint": "event",
                "prompt": prompt,
            })
            return _authority(item_id=item.item_id, slots=slots)
        return None


__all__ = ["MissingContextClarificationBackend"]
