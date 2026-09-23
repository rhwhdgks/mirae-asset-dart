"""Strict PlanProposal과 Stage 1→4 QueryPlan handoff v0.4.

``PlanProposal``은 HCX wire를 결정론적으로 정규화한 비권위 의미 후보다. 회사 코드,
``event_key``, ``task_id``, ``output_id``와 날짜 cutoff는 이 모델에 존재하지 않는다.
Python resolver가 canonical preflight를 거쳐 만든 ``ResolvedQueryPlan``만 실행할 수
있고, 외부에는 JSON path가 없는 ``QueryPlanHandoff``만 직렬화한다.
"""

from __future__ import annotations

from datetime import date
import json
import re
from typing import Any, Literal, Mapping, Protocol
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError, model_validator

from .contracts import (
    AsOf,
    ClarificationRequest,
    ContractModel,
    Derivation,
    HexId,
    OutputRef,
    PlanValidation,
    PremiseClaim,
    PresentationSpec,
    ResolvedFinancialTask,
    ResolvedQueryPlan,
    TaskVerificationRef,
)
from .drafts import (
    DraftCorrectionTask,
    DraftDisclosureTask,
    DraftDocumentTask,
    DraftEventTask,
    DraftFinancialTask,
    DraftNarrativeTask,
    DraftQueryPlan,
)
from .planning import DraftPlanResolver, concept_axes
from .session import (
    PendingClarification, QuestionOrigin,
    ProposalAdapter,
    QuestionSessionState,
    UnsafeClarificationError,
)


QUERY_PLAN_HANDOFF_VERSION = "0.4"
PLAN_PROPOSAL_VERSION = "plan-proposal/0.4"
UNSPECIFIED = "unspecified"

ProposalDisposition = Literal[
    "process", "out_of_scope", "unsupported_request", "policy_refusal",
]
ProposalTaskKind = Literal[
    "financial", "disclosure", "event", "correction", "document", "narrative",
]
ProposalOperation = Literal[
    "lookup", "list", "status", "timeline", "diff", "history", "find",
    "latest", "version_history", "search", "summarize", "compare",
]


class ProposedFact(ContractModel):
    """HCX가 뽑은 atomic 재무 요청. 모든 값은 아직 비권위 문자열이다."""

    result_label: str = Field(min_length=1, max_length=80)
    company_mention: str = Field(min_length=1, max_length=200)
    concept_mention: str = Field(min_length=1, max_length=200)
    period_expression: str = Field(min_length=1, max_length=100)
    scope: Literal["unspecified", "CFS", "SFS"]
    statement: Literal["unspecified", "BS", "IS", "CI", "CF"]
    view: Literal["unspecified", "as_filed", "restated"]


class PlanProposalTask(ContractModel):
    """한 semantic task 후보. unused 필드는 빈 배열/``unspecified``로 고정한다."""

    task_label: str = Field(min_length=1, max_length=80)
    kind: ProposalTaskKind
    operation: ProposalOperation
    company_mentions: list[str] = Field(max_length=8)
    facts: list[ProposedFact] = Field(max_length=16)
    as_of_expression: str = Field(min_length=1, max_length=100)
    document_group: Literal[
        "unspecified", "periodic", "exchange", "major", "holding",
    ]
    event_type_text: str = Field(max_length=200)
    counterparty_text: str = Field(max_length=300)
    contract_name_text: str = Field(max_length=500)
    seed_receipt_text: str = Field(max_length=100)
    retrieval_query: str = Field(max_length=1000)
    requested_slots: list[str] = Field(max_length=32)
    result_labels: list[str] = Field(max_length=16)

    @model_validator(mode="after")
    def validate_task_shape(self) -> "PlanProposalTask":
        operations = {
            "financial": {"lookup"},
            "disclosure": {"lookup", "list"},
            "event": {"status", "timeline", "list"},
            "correction": {"diff", "history"},
            "document": {"find", "latest", "version_history"},
            "narrative": {"search", "summarize", "compare"},
        }
        if self.operation not in operations[self.kind]:
            raise ValueError("task kind와 operation이 일치하지 않습니다")
        if self.kind == "financial":
            if not self.facts:
                raise ValueError("financial proposal에는 fact가 필요합니다")
        elif self.facts:
            raise ValueError("financial이 아닌 proposal은 fact를 가질 수 없습니다")
        labels = [row.result_label for row in self.facts] + self.result_labels
        if len(labels) != len(set(labels)):
            raise ValueError("task 안의 result label은 중복될 수 없습니다")
        if len(self.company_mentions) != len(set(self.company_mentions)):
            raise ValueError("company mention은 중복될 수 없습니다")
        if len(self.requested_slots) != len(set(self.requested_slots)):
            raise ValueError("requested slot은 중복될 수 없습니다")
        return self


class ProposedDerivation(ContractModel):
    """output ID가 아니라 proposal 내부 semantic label만 참조한다."""

    result_label: str = Field(min_length=1, max_length=80)
    operator: Literal[
        "difference", "absolute_difference", "percent_change",
        "discrete_from_cumulative", "argmax", "concept_ratio",
    ]
    operand_labels: list[str] = Field(min_length=1, max_length=32)
    rounding_rule: Literal[
        "unspecified", "round_half_up_0", "round_half_up_1",
        "round_half_up_2", "truncate_0",
    ]

    @model_validator(mode="after")
    def validate_arity(self) -> "ProposedDerivation":
        if self.operator != "argmax" and len(self.operand_labels) != 2:
            raise ValueError(f"{self.operator} proposal은 operand 2개가 필요합니다")
        if self.operator == "argmax" and len(self.operand_labels) < 2:
            raise ValueError("argmax proposal은 operand 2개 이상이 필요합니다")
        return self


class ProposedPremise(ContractModel):
    claim_label: str = Field(min_length=1, max_length=80)
    kind: Literal["numeric", "state", "comparison", "existence", "causal"]
    raw_text: str = Field(min_length=1, max_length=1000)
    verify_labels: list[str] = Field(max_length=32)


class PlanProposal(ContractModel):
    """Provider wire보다 엄격하고 ResolvedQueryPlan보다 비권위인 의미 후보."""

    schema_version: Literal["plan-proposal/0.4"]
    disposition: ProposalDisposition
    tasks: list[PlanProposalTask] = Field(max_length=12)
    derivations: list[ProposedDerivation] = Field(max_length=32)
    premise_claims: list[ProposedPremise] = Field(max_length=32)
    presentation: Literal["unspecified", "prose", "table", "list"]
    reason_codes: list[str] = Field(max_length=16)

    @model_validator(mode="after")
    def validate_proposal(self) -> "PlanProposal":
        if self.disposition == "process":
            if not self.tasks or self.reason_codes:
                raise ValueError("process proposal에는 task만 있고 terminal reason은 없어야 합니다")
        elif self.tasks or self.derivations or self.premise_claims or not self.reason_codes:
            raise ValueError("terminal proposal에는 typed reason만 있어야 합니다")
        if any(not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code)
               for code in self.reason_codes):
            raise ValueError("reason code 형식이 잘못되었습니다")
        task_labels = [task.task_label for task in self.tasks]
        if len(task_labels) != len(set(task_labels)):
            raise ValueError("task label은 proposal 전체에서 유일해야 합니다")

        known: set[str] = set()
        for task in self.tasks:
            for label in [row.result_label for row in task.facts] + task.result_labels:
                if label in known:
                    raise ValueError("result label은 proposal 전체에서 유일해야 합니다")
                known.add(label)
        for derivation in self.derivations:
            missing = [label for label in derivation.operand_labels if label not in known]
            if missing:
                raise ValueError(f"derivation이 알 수 없는/후행 label을 참조합니다: {missing}")
            if derivation.result_label in known:
                raise ValueError("derivation result label이 중복되었습니다")
            known.add(derivation.result_label)
        for premise in self.premise_claims:
            missing = [label for label in premise.verify_labels if label not in known]
            if missing:
                raise ValueError(f"premise가 알 수 없는 label을 참조합니다: {missing}")
        return self


class PlanProposalBackend(Protocol):
    def create_proposal(
            self, question: str, *, reference_date: date,
            corpus_cutoff: str) -> PlanProposal | dict[str, Any] | str: ...


class PlanProposalOutputError(ValueError):
    """HCX 결과가 strict PlanProposal 계약을 통과하지 못했다."""


class PlanProposalParser:
    """provider object/JSON을 strict PlanProposal로 바꾸는 단일 입구."""

    def __init__(self, backend: PlanProposalBackend) -> None:
        self.backend = backend

    def create_proposal(
            self, question: str, *, reference_date: date,
            corpus_cutoff: str) -> PlanProposal:
        if not isinstance(question, str) or not question.strip():
            raise PlanProposalOutputError("질문은 비어 있지 않은 문자열이어야 합니다")
        try:
            payload = self.backend.create_proposal(
                question.strip(), reference_date=reference_date,
                corpus_cutoff=corpus_cutoff)
            if isinstance(payload, PlanProposal):
                return payload
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                raise TypeError("PlanProposal provider는 JSON object를 반환해야 합니다")
            return PlanProposal.model_validate(payload, strict=True)
        except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as exc:
            if isinstance(exc, PlanProposalOutputError):
                raise
            raise PlanProposalOutputError("HCX PlanProposal schema 오류") from exc


class ProposalResolver(Protocol):
    def resolve(
            self, proposal: PlanProposal, *, revision: int = 0,
            ) -> PlanValidation: ...


_TASK_PATCH_TARGETS = {
    "company_mentions": "company",
    "as_of_expression": "as_of",
    "document_group": "document_group",
    "event_type_text": "event_type",
    "counterparty_text": "counterparty",
    "contract_name_text": "contract_name",
    "seed_receipt_text": "seed_receipt",
    "retrieval_query": "retrieval_query",
    "requested_slots": "requested_slots",
}
_FACT_PATCH_TARGETS = {
    "company_mention": "company",
    "concept_mention": "metric",
    "period_expression": "period",
    "scope": "scope",
    "statement": "statement",
    "view": "view",
}
_TASK_PATH = re.compile(r"^proposal\.tasks\[([0-9]+)]\.([a-z_]+)$")
_FACT_PATH = re.compile(
    r"^proposal\.tasks\[([0-9]+)]\.facts\[([0-9]+)]\.([a-z_]+)$")


class PlanProposalAdapter(ProposalAdapter[PlanProposal]):
    """semantic slot 답을 서버 내부 allowlist path에만 원자 적용한다."""

    def __init__(self, resolver: ProposalResolver) -> None:
        self.resolver = resolver

    def resolve(
            self, proposal: PlanProposal, *, revision: int = 0,
            origin: QuestionOrigin | None = None,
            ) -> PlanValidation:
        del origin
        return self.resolver.resolve(proposal, revision=revision)

    def resolve_with_context(
            self, proposal: PlanProposal, *, revision: int = 0,
            origin: QuestionOrigin | None = None,
            task_target_period_expressions: tuple[tuple[str, ...], ...] = (),
            premise_task_indices: tuple[tuple[int, ...], ...] = (),
            selected_company_mention: str | None = None,
            selected_document_receipt: str | None = None,
            ) -> PlanValidation:
        """Stage1 compiler의 비권위 sidecar를 resolver 경계까지 보존한다."""

        del origin
        contextual = getattr(self.resolver, "resolve_with_context", None)
        if callable(contextual):
            return contextual(
                proposal,
                revision=revision,
                task_target_period_expressions=(
                    task_target_period_expressions),
                premise_task_indices=premise_task_indices,
                selected_company_mention=selected_company_mention,
                selected_document_receipt=selected_document_receipt,
            )
        if (task_target_period_expressions or premise_task_indices
                or selected_company_mention is not None
                or selected_document_receipt is not None):
            return PlanValidation(
                status="unsupported_request",
                reasons=["proposal_resolution_context_not_supported"],
            )
        return self.resolver.resolve(proposal, revision=revision)

    @staticmethod
    def _parse_path(
            proposal: PlanProposal, path: str,
            ) -> tuple[int, int | None, str, str] | None:
        match = _FACT_PATH.fullmatch(path)
        if match is not None:
            task_index, fact_index = (int(match.group(1)), int(match.group(2)))
            field_name = match.group(3)
            if (field_name not in _FACT_PATCH_TARGETS
                    or not 0 <= task_index < len(proposal.tasks)
                    or not 0 <= fact_index < len(proposal.tasks[task_index].facts)):
                return None
            return task_index, fact_index, field_name, _FACT_PATCH_TARGETS[field_name]
        match = _TASK_PATH.fullmatch(path)
        if match is None:
            return None
        task_index = int(match.group(1))
        field_name = match.group(2)
        if (field_name not in _TASK_PATCH_TARGETS
                or not 0 <= task_index < len(proposal.tasks)):
            return None
        return task_index, None, field_name, _TASK_PATCH_TARGETS[field_name]

    def target_for_path(self, proposal: PlanProposal, path: str) -> str | None:
        parsed = self._parse_path(proposal, path)
        return parsed[3] if parsed is not None else None

    def apply(
            self, proposal: PlanProposal, values_by_path: Mapping[str, Any],
            ) -> PlanProposal:
        if not values_by_path:
            raise UnsafeClarificationError("적용할 clarification 값이 없습니다")
        payload = proposal.model_dump(mode="python")
        for path, value in values_by_path.items():
            parsed = self._parse_path(proposal, path)
            if parsed is None:
                raise UnsafeClarificationError("PlanProposal allowlist 밖 path입니다")
            task_index, fact_index, field_name, _ = parsed
            if fact_index is None:
                if field_name == "company_mentions" and not isinstance(value, list):
                    value = [value]
                if field_name == "requested_slots" and not isinstance(value, list):
                    value = [value]
                payload["tasks"][task_index][field_name] = value
            else:
                if field_name == "period_expression" and type(value) is int:
                    value = str(value)
                payload["tasks"][task_index]["facts"][fact_index][field_name] = value
        try:
            return PlanProposal.model_validate(payload, strict=True)
        except ValueError as exc:
            raise UnsafeClarificationError(
                "clarification batch가 PlanProposal schema를 통과하지 못했습니다") from exc


def _translate_validation(
        validation: PlanValidation, mapping: Mapping[str, str],
        ) -> PlanValidation:
    request = validation.clarification
    if validation.status != "needs_clarification" or request is None:
        return validation
    try:
        paths = [mapping[path] for path in request.patch_paths]
        field_paths = [mapping[path] for path in request.field_paths]
        options = {mapping[path]: values for path, values in request.options.items()}
    except KeyError as exc:
        raise ValueError("current-slice resolver가 알 수 없는 Draft path를 반환했습니다") from exc
    return PlanValidation(
        status="needs_clarification",
        clarification=ClarificationRequest(
            request_id=request.request_id,
            plan_revision=request.plan_revision,
            # path 만 옮긴다 — 왜 물었는지는 바뀌지 않는다.
            type=request.type, reason_code=request.reason_code,
            field_paths=field_paths,
            question=request.question,
            options=options,
            patch_paths=paths,
        ),
    )


def _rebase_validation(
        validation: PlanValidation, *, task_index: int,
        fact_index: int | None = None,
        ) -> PlanValidation:
    """단일-task delegate의 내부 path를 원래 proposal 좌표로 되돌린다."""

    request = validation.clarification
    if validation.status != "needs_clarification" or request is None:
        return validation

    def rebase(path: str) -> str:
        fact_prefix = "proposal.tasks[0].facts[0]"
        task_prefix = "proposal.tasks[0]"
        if fact_index is not None and path.startswith(fact_prefix):
            return (
                f"proposal.tasks[{task_index}].facts[{fact_index}]"
                f"{path[len(fact_prefix):]}"
            )
        if path.startswith(task_prefix):
            return f"proposal.tasks[{task_index}]{path[len(task_prefix):]}"
        # document route의 서버 sidecar path는 proposal path가 아니므로 유지한다.
        return path

    return PlanValidation(
        status="needs_clarification",
        clarification=ClarificationRequest(
            request_id=request.request_id,
            plan_revision=request.plan_revision,
            type=request.type, reason_code=request.reason_code,
            field_paths=[rebase(path) for path in request.field_paths],
            question=request.question,
            options={rebase(path): values for path, values in request.options.items()},
            patch_paths=[rebase(path) for path in request.patch_paths],
        ),
    )


class CurrentSliceProposalResolver:
    """여섯 task를 원자적으로 조립하는 PlanProposal full resolver.

    각 task는 기존 결정론적 Draft resolver로 독립 검증한다. 하나라도 역질문 또는
    terminal이면 부분 plan을 내보내지 않고 즉시 중단한다. 전부 ready일 때만 task,
    계산 DAG, 사용자 전제와 presentation을 하나의 ``ResolvedQueryPlan``으로 묶는다.
    """

    def __init__(self, delegate: DraftPlanResolver) -> None:
        self.delegate = delegate

    @staticmethod
    def _terminal(proposal: PlanProposal) -> PlanValidation | None:
        if proposal.disposition == "process":
            return None
        # PlanProposal.disposition is model-originated and therefore only a
        # proposal.  TerminalDecisionGate must bind it to the original question
        # and a Python authority before a terminal result can leave Stage 1.
        return PlanValidation(
            status="unsupported_request",
            reasons=["terminal_decision_gate_required"],
        )

    def resolve(
            self, proposal: PlanProposal, *, revision: int = 0,
            ) -> PlanValidation:
        return self.resolve_with_context(proposal, revision=revision)

    def resolve_with_context(
            self, proposal: PlanProposal, *, revision: int = 0,
            task_target_period_expressions: tuple[tuple[str, ...], ...] = (),
            premise_task_indices: tuple[tuple[int, ...], ...] = (),
            selected_company_mention: str | None = None,
            selected_document_receipt: str | None = None,
            ) -> PlanValidation:
        terminal = self._terminal(proposal)
        if terminal is not None:
            return terminal
        if (task_target_period_expressions
                and len(task_target_period_expressions) != len(proposal.tasks)):
            return PlanValidation(
                status="unsupported_request",
                reasons=["document_target_period_sidecar_shape_mismatch"],
            )
        if (premise_task_indices
                and len(premise_task_indices) != len(proposal.premise_claims)):
            return PlanValidation(
                status="unsupported_request",
                reasons=["premise_task_sidecar_shape_mismatch"],
            )

        company_context_index: int | None = None
        if selected_company_mention is not None:
            candidates = [
                index for index, task in enumerate(proposal.tasks)
                if task.kind in {
                    "correction", "disclosure", "document", "event", "narrative",
                }
            ]
            if len(candidates) != 1:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["company_context_ambiguous_across_tasks"],
                )
            company_context_index = candidates[0]

        document_context_index: int | None = None
        if selected_document_receipt is not None:
            candidates = [
                index for index, task in enumerate(proposal.tasks)
                if task.kind in {"document", "narrative"}
            ]
            if len(candidates) != 1:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["document_receipt_context_ambiguous_across_tasks"],
                )
            document_context_index = candidates[0]

        resolved_tasks: list[Any] = []
        applied_defaults: list[str] = []
        label_to_output: dict[str, str] = {}
        task_ids_by_proposal_index: dict[int, tuple[str, ...]] = {}
        reference_date: date | None = None
        corpus_cutoff: str | None = None

        def collect_plan(validation: PlanValidation) -> ResolvedQueryPlan | None:
            nonlocal reference_date, corpus_cutoff
            if validation.status != "ready" or validation.plan is None:
                return None
            plan = validation.plan
            if reference_date is None:
                reference_date = plan.reference_date
                corpus_cutoff = plan.corpus_cutoff
            elif (plan.reference_date != reference_date
                  or plan.corpus_cutoff != corpus_cutoff):
                raise ValueError("task resolver의 기준일/cutoff가 서로 다릅니다")
            for default in plan.applied_defaults:
                if default not in applied_defaults:
                    applied_defaults.append(default)
            return plan

        for task_index, task in enumerate(proposal.tasks):
            expressions = (
                (task_target_period_expressions[task_index],)
                if task_target_period_expressions else ()
            )
            company_context = (
                selected_company_mention
                if company_context_index == task_index else None
            )
            document_context = (
                selected_document_receipt
                if document_context_index == task_index else None
            )

            if task.kind == "financial":
                grouped: dict[tuple[str, str], list[Any]] = {}
                group_order: list[tuple[str, str]] = []
                for fact_index, proposed_fact in enumerate(task.facts):
                    one_task = task.model_copy(update={"facts": [proposed_fact]})
                    one_proposal = proposal.model_copy(update={
                        "tasks": [one_task],
                        "derivations": [],
                        "premise_claims": [],
                        "presentation": UNSPECIFIED,
                    })
                    validation = self._resolve_single_with_context(
                        one_proposal, revision=revision)
                    validation = _rebase_validation(
                        validation, task_index=task_index,
                        fact_index=fact_index)
                    if validation.status != "ready":
                        return validation
                    plan = collect_plan(validation)
                    assert plan is not None
                    resolved = plan.tasks[0]
                    if not isinstance(resolved, ResolvedFinancialTask):
                        return PlanValidation(
                            status="unsupported_request",
                            reasons=["financial_resolver_returned_wrong_task"],
                        )
                    fact = resolved.facts[0]
                    if proposed_fact.statement != UNSPECIFIED:
                        # 예전에는 개념 5종의 허용 표를 여기 손으로 적었고, 표에
                        # 없는 개념이 오면 ``KeyError`` 로 죽었다. 개념을 늘릴 때
                        # 같이 늘어나야 하는 값이므로 ``ConceptAxes`` 한 곳에서 온다.
                        allowed = set(concept_axes(fact.concept).allowed_statements)
                        if proposed_fact.statement not in allowed:
                            return PlanValidation(
                                status="unsupported_request",
                                reasons=["financial_statement_concept_mismatch"],
                            )
                    fact = fact.model_copy(update={
                        "output_id": proposed_fact.result_label,
                        "statement": (
                            fact.statement
                            if proposed_fact.statement == UNSPECIFIED
                            else proposed_fact.statement),
                    })
                    label_to_output[proposed_fact.result_label] = fact.output_id
                    group_key = (resolved.as_of, resolved.view)
                    if group_key not in grouped:
                        grouped[group_key] = []
                        group_order.append(group_key)
                    grouped[group_key].append(fact)

                for group_index, group_key in enumerate(group_order):
                    as_of, view = group_key
                    suffix = "" if len(group_order) == 1 else f"-{group_index + 1}"
                    resolved_tasks.append(ResolvedFinancialTask(
                        task_id=f"financial-{task_index + 1}{suffix}",
                        as_of=as_of,
                        view=view,  # type: ignore[arg-type]
                        facts=grouped[group_key],
                    ))
                task_ids_by_proposal_index[task_index] = tuple(
                    task.task_id for task in resolved_tasks
                    if (task.task_id == f"financial-{task_index + 1}"
                        or task.task_id.startswith(
                            f"financial-{task_index + 1}-")))
                continue

            one_proposal = proposal.model_copy(update={
                "tasks": [task],
                "derivations": [],
                "premise_claims": [],
                "presentation": UNSPECIFIED,
            })
            validation = self._resolve_single_with_context(
                one_proposal,
                revision=revision,
                task_target_period_expressions=expressions,
                selected_company_mention=company_context,
                selected_document_receipt=document_context,
            )
            validation = _rebase_validation(
                validation, task_index=task_index)
            if validation.status != "ready":
                return validation
            plan = collect_plan(validation)
            assert plan is not None
            resolved = plan.tasks[0]
            resolved = resolved.model_copy(update={
                "task_id": f"{task.kind}-{task_index + 1}"})

            field_outputs = list(getattr(resolved, "field_outputs", []))
            task_output = getattr(resolved, "output_id", None)
            available_outputs = ([task_output] if task_output else []) + [
                row.output_id for row in field_outputs]
            if task.result_labels and len(task.result_labels) != len(available_outputs):
                # Document/Narrative는 task 자체가 sink이며 scalar output 계약이 없다.
                if task.kind not in {"document", "narrative"}:
                    return PlanValidation(
                        status="unsupported_request",
                        reasons=["result_label_binding_mismatch"],
                    )
            else:
                for label, output_id in zip(
                        task.result_labels, available_outputs, strict=True):
                    label_to_output[label] = output_id
            resolved_tasks.append(resolved)
            task_ids_by_proposal_index[task_index] = (resolved.task_id,)

        # 개념의 **연산 정책**을 집행하려면 output 이 어느 개념인지 알아야 한다.
        # wire 는 계정을 자유 텍스트로만 말하므로 이 층이 개념을 아는 첫 지점이다.
        aggregation_by_output: dict[str, str] = {}
        for resolved in resolved_tasks:
            for fact in getattr(resolved, "facts", []) or []:
                concept = getattr(fact, "concept", None)
                output_id = getattr(fact, "output_id", None)
                if concept is None or not output_id:
                    continue
                aggregation_by_output[output_id] = concept_axes(concept).aggregation

        derivations: list[Derivation] = []
        for derivation in proposal.derivations:
            missing = [
                label for label in derivation.operand_labels
                if label not in label_to_output
            ]
            if missing:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["derivation_operand_has_no_scalar_output"],
                )
            # 누적 차감은 **더할 수 있는 기간 값**에서만 성립한다. 주당 값(EPS)은
            # 반기에서 1분기를 빼도 2분기 EPS 가 되지 않고, 시점 잔액(자산총계)은
            # 누적이라는 개념 자체가 없다. 금액 계정과 같은 연산을 붙이면 틀린 수가
            # 조용히 나온다 — 개념별 예외가 아니라 aggregation 축의 규칙이다.
            if derivation.operator == "discrete_from_cumulative":
                offending = [
                    label for label in derivation.operand_labels
                    if aggregation_by_output.get(
                        label_to_output[label], "additive_duration")
                    != "additive_duration"
                ]
                if offending:
                    return PlanValidation(
                        status="unsupported_request",
                        reasons=["derivation_operator_not_additive"],
                    )
            resolved = Derivation(
                output_id=derivation.result_label,
                operator=derivation.operator,
                operands=[OutputRef(output_id=label_to_output[label])
                          for label in derivation.operand_labels],
                rounding_rule=(
                    None if derivation.rounding_rule == UNSPECIFIED
                    else derivation.rounding_rule),
            )
            derivations.append(resolved)
            label_to_output[derivation.result_label] = resolved.output_id

        premises: list[PremiseClaim] = []
        for premise_index, premise in enumerate(proposal.premise_claims):
            missing = [
                label for label in premise.verify_labels
                if label not in label_to_output
            ]
            if missing:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["premise_reference_has_no_scalar_output"],
                )
            task_indices = (
                premise_task_indices[premise_index]
                if premise_task_indices else ())
            missing_task_indices = [
                index for index in task_indices
                if index not in task_ids_by_proposal_index
            ]
            if missing_task_indices:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["premise_reference_has_no_resolved_task"],
                )
            task_refs = [
                TaskVerificationRef(task_id=task_id)
                for index in task_indices
                for task_id in task_ids_by_proposal_index[index]
            ]
            scalar_refs = [
                OutputRef(output_id=label_to_output[label])
                for label in premise.verify_labels
            ]
            if not scalar_refs and not task_refs:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["premise_reference_missing"],
                )
            premises.append(PremiseClaim(
                claim_id=f"premise-{premise_index + 1}",
                kind=premise.kind,
                raw_text=premise.raw_text,
                verify_with=scalar_refs,
                verify_tasks=task_refs,
            ))

        if reference_date is None or corpus_cutoff is None:
            return PlanValidation(
                status="unsupported_request", reasons=["resolved_task_missing"])
        presentation = (
            None if proposal.presentation == UNSPECIFIED
            else PresentationSpec(format=proposal.presentation)
        )
        try:
            plan = ResolvedQueryPlan(
                revision=revision,
                reference_date=reference_date,
                corpus_cutoff=corpus_cutoff,
                tasks=resolved_tasks,
                derivations=derivations,
                premise_claims=premises,
                applied_defaults=applied_defaults,
                presentation=presentation,
            )
        except (TypeError, ValueError, ValidationError):
            return PlanValidation(
                status="unsupported_request",
                reasons=["resolved_plan_assembly_failed"],
            )
        return PlanValidation(status="ready", plan=plan)

    def _resolve_single_with_context(
            self, proposal: PlanProposal, *, revision: int = 0,
            task_target_period_expressions: tuple[tuple[str, ...], ...] = (),
            selected_company_mention: str | None = None,
            selected_document_receipt: str | None = None,
            ) -> PlanValidation:
        terminal = self._terminal(proposal)
        if terminal is not None:
            return terminal
        if (task_target_period_expressions
                and len(task_target_period_expressions) != len(proposal.tasks)):
            return PlanValidation(
                status="unsupported_request",
                reasons=["document_target_period_sidecar_shape_mismatch"],
            )
        if len(proposal.tasks) != 1:
            return PlanValidation(
                status="unsupported_request",
                reasons=["multi_task_runtime_not_implemented"],
            )
        task = proposal.tasks[0]
        if (selected_company_mention is not None
                and task.kind not in {
                    "correction", "disclosure", "document", "event", "narrative",
                }):
            return PlanValidation(
                status="unsupported_request",
                reasons=["company_context_on_unsupported_task"],
            )
        if (selected_document_receipt is not None
                and task.kind not in {"document", "narrative"}):
            return PlanValidation(
                status="unsupported_request",
                reasons=["document_receipt_context_on_non_document_task"],
            )
        if task.kind == "financial":
            if len(task.facts) != 1 or proposal.derivations:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["multi_fact_or_derivation_runtime_not_implemented"],
                )
            fact = task.facts[0]
            company = fact.company_mention
            if company == UNSPECIFIED:
                company = task.company_mentions[0] if task.company_mentions else None
            draft = DraftQueryPlan(tasks=[DraftFinancialTask(
                company_text=company,
                metric_text=(None if fact.concept_mention == UNSPECIFIED
                             else fact.concept_mention),
                period_expression=(
                    None if fact.period_expression == UNSPECIFIED
                    else fact.period_expression),
                scope=None if fact.scope == UNSPECIFIED else fact.scope,
                as_of=(None if task.as_of_expression in {
                    UNSPECIFIED, "현재", "오늘", "지금", "최신",
                }
                       else task.as_of_expression),
                view="restated" if fact.view == UNSPECIFIED else fact.view,
            )])
            validation = self.delegate.resolve(draft, revision=revision)
            mapping = {
                "draft.tasks[0].company_text": "proposal.tasks[0].facts[0].company_mention",
                "draft.tasks[0].metric_text": "proposal.tasks[0].facts[0].concept_mention",
                "draft.tasks[0].year": "proposal.tasks[0].facts[0].period_expression",
                "draft.tasks[0].period_expression": (
                    "proposal.tasks[0].facts[0].period_expression"),
                "draft.tasks[0].scope": "proposal.tasks[0].facts[0].scope",
                "draft.tasks[0].as_of": "proposal.tasks[0].as_of_expression",
                "draft.tasks[0].view": "proposal.tasks[0].facts[0].view",
            }
            return _translate_validation(validation, mapping)
        if task.kind == "disclosure":
            company_mentions = (
                [selected_company_mention]
                if selected_company_mention is not None
                else task.company_mentions
            )
            if len(company_mentions) > 1:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["disclosure_task_requires_single_company"],
                )
            expressions = (
                task_target_period_expressions[0]
                if task_target_period_expressions else ()
            )
            draft = DraftQueryPlan(tasks=[DraftDisclosureTask(
                operation=task.operation,
                company_text=(company_mentions[0]
                              if company_mentions else None),
                as_of=(None if task.as_of_expression in {
                    "", UNSPECIFIED, "현재", "오늘", "지금",
                } else task.as_of_expression),
                doc_group=(None if task.document_group == UNSPECIFIED
                           else task.document_group),
                event_type_text=(None if task.event_type_text in {"", UNSPECIFIED}
                                 else task.event_type_text),
                counterparty_text=(
                    None if task.counterparty_text in {"", UNSPECIFIED}
                    else task.counterparty_text),
                contract_name_text=(
                    None if task.contract_name_text in {"", UNSPECIFIED}
                    else task.contract_name_text),
                seed_receipt_text=(
                    None if task.seed_receipt_text in {"", UNSPECIFIED}
                    else task.seed_receipt_text),
                target_period_expressions=list(expressions),
                requested_slots=list(task.requested_slots),
                result_labels=list(task.result_labels),
            )])
            validation = self.delegate.resolve(draft, revision=revision)
            mapping = {
                "draft.tasks[0].company_text": (
                    "proposal.tasks[0].company_mentions"),
                "draft.tasks[0].as_of": (
                    "proposal.tasks[0].as_of_expression"),
                "draft.tasks[0].doc_group": (
                    "proposal.tasks[0].document_group"),
                "draft.tasks[0].event_type_text": (
                    "proposal.tasks[0].event_type_text"),
                "draft.tasks[0].counterparty_text": (
                    "proposal.tasks[0].counterparty_text"),
                "draft.tasks[0].contract_name_text": (
                    "proposal.tasks[0].contract_name_text"),
                "draft.tasks[0].seed_receipt_text": (
                    "proposal.tasks[0].seed_receipt_text"),
                "draft.tasks[0].requested_slots": (
                    "proposal.tasks[0].requested_slots"),
            }
            return _translate_validation(validation, mapping)
        if task.kind == "event":
            company_mentions = (
                [selected_company_mention]
                if selected_company_mention is not None
                else task.company_mentions
            )
            if len(company_mentions) > 1:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["event_task_requires_single_company"],
                )
            expressions = (
                task_target_period_expressions[0]
                if task_target_period_expressions else ()
            )
            draft = DraftQueryPlan(tasks=[DraftEventTask(
                operation=task.operation,
                company_text=(company_mentions[0]
                              if company_mentions else None),
                as_of_expression=(
                    None if task.as_of_expression in {
                        "", UNSPECIFIED, "현재", "오늘", "지금", "최신",
                    } else task.as_of_expression),
                event_type_text=(
                    None if task.event_type_text in {"", UNSPECIFIED}
                    else task.event_type_text),
                counterparty_text=(
                    None if task.counterparty_text in {"", UNSPECIFIED}
                    else task.counterparty_text),
                contract_name_text=(
                    None if task.contract_name_text in {"", UNSPECIFIED}
                    else task.contract_name_text),
                seed_receipt_text=(
                    None if task.seed_receipt_text in {"", UNSPECIFIED}
                    else task.seed_receipt_text),
                target_period_expressions=list(expressions),
                requested_slots=list(task.requested_slots),
                result_labels=list(task.result_labels),
            )])
            validation = self.delegate.resolve(draft, revision=revision)
            mapping = {
                "draft.tasks[0].company_text": (
                    "proposal.tasks[0].company_mentions"),
                "draft.tasks[0].as_of_expression": (
                    "proposal.tasks[0].as_of_expression"),
                "draft.tasks[0].event_type_text": (
                    "proposal.tasks[0].event_type_text"),
                "draft.tasks[0].counterparty_text": (
                    "proposal.tasks[0].counterparty_text"),
                "draft.tasks[0].contract_name_text": (
                    "proposal.tasks[0].contract_name_text"),
                "draft.tasks[0].seed_receipt_text": (
                    "proposal.tasks[0].seed_receipt_text"),
                "draft.tasks[0].requested_slots": (
                    "proposal.tasks[0].requested_slots"),
            }
            return _translate_validation(validation, mapping)
        if task.kind == "correction":
            company_mentions = (
                [selected_company_mention]
                if selected_company_mention is not None
                else task.company_mentions
            )
            if len(company_mentions) > 1:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["correction_task_requires_single_company"],
                )
            expressions = (
                task_target_period_expressions[0]
                if task_target_period_expressions else ()
            )
            draft = DraftQueryPlan(tasks=[DraftCorrectionTask(
                operation=task.operation,
                company_text=(company_mentions[0]
                              if company_mentions else None),
                as_of=(None if task.as_of_expression in {
                    "", UNSPECIFIED, "현재", "오늘", "지금", "최신",
                } else task.as_of_expression),
                doc_group=(None if task.document_group == UNSPECIFIED
                           else task.document_group),
                event_type_text=(
                    None if task.event_type_text in {"", UNSPECIFIED}
                    else task.event_type_text),
                counterparty_text=(
                    None if task.counterparty_text in {"", UNSPECIFIED}
                    else task.counterparty_text),
                contract_name_text=(
                    None if task.contract_name_text in {"", UNSPECIFIED}
                    else task.contract_name_text),
                seed_receipt_text=(
                    None if task.seed_receipt_text in {"", UNSPECIFIED}
                    else task.seed_receipt_text),
                target_period_expressions=list(expressions),
                requested_slots=list(task.requested_slots),
                result_labels=list(task.result_labels),
            )])
            validation = self.delegate.resolve(draft, revision=revision)
            mapping = {
                "draft.tasks[0].company_text": (
                    "proposal.tasks[0].company_mentions"),
                "draft.tasks[0].as_of": (
                    "proposal.tasks[0].as_of_expression"),
                "draft.tasks[0].doc_group": (
                    "proposal.tasks[0].document_group"),
                "draft.tasks[0].event_type_text": (
                    "proposal.tasks[0].event_type_text"),
                "draft.tasks[0].counterparty_text": (
                    "proposal.tasks[0].counterparty_text"),
                "draft.tasks[0].contract_name_text": (
                    "proposal.tasks[0].contract_name_text"),
                "draft.tasks[0].seed_receipt_text": (
                    "proposal.tasks[0].seed_receipt_text"),
                "draft.tasks[0].requested_slots": (
                    "proposal.tasks[0].requested_slots"),
            }
            return _translate_validation(validation, mapping)
        if task.kind == "document":
            company_mentions = (
                [selected_company_mention]
                if selected_company_mention is not None
                else task.company_mentions
            )
            if len(company_mentions) > 1:
                return PlanValidation(
                    status="unsupported_request",
                    reasons=["document_task_requires_single_company"],
                )
            expressions = (
                task_target_period_expressions[0]
                if task_target_period_expressions else ()
            )
            draft = DraftQueryPlan(tasks=[DraftDocumentTask(
                operation=task.operation,
                company_text=(company_mentions[0] if company_mentions else None),
                as_of=(None if task.as_of_expression in {
                    "", UNSPECIFIED, "현재", "오늘",
                } else task.as_of_expression),
                doc_group=(None if task.document_group == UNSPECIFIED
                           else task.document_group),
                event_type_text=(None if task.event_type_text in {"", UNSPECIFIED}
                                 else task.event_type_text),
                counterparty_text=(
                    None if task.counterparty_text in {"", UNSPECIFIED}
                    else task.counterparty_text),
                contract_name_text=(
                    None if task.contract_name_text in {"", UNSPECIFIED}
                    else task.contract_name_text),
                seed_receipt_text=(
                    None if task.seed_receipt_text in {"", UNSPECIFIED}
                    else task.seed_receipt_text),
                target_period_expressions=list(expressions),
                selected_document_receipt=selected_document_receipt,
            )])
            validation = self.delegate.resolve(draft, revision=revision)
            mapping = {
                "draft.tasks[0].company_text": (
                    "proposal.tasks[0].company_mentions"),
                "draft.tasks[0].as_of": (
                    "proposal.tasks[0].as_of_expression"),
                "draft.tasks[0].doc_group": (
                    "proposal.tasks[0].document_group"),
                "draft.tasks[0].target_period_expressions": (
                    "stage1.document_target_period"),
                "draft.tasks[0].selected_document_receipt": (
                    "stage1.selected_document_receipt"),
            }
            return _translate_validation(validation, mapping)
        if task.kind == "narrative":
            company = (
                selected_company_mention
                if selected_company_mention is not None
                else (task.company_mentions[0] if task.company_mentions else None)
            )
            draft = DraftQueryPlan(tasks=[DraftNarrativeTask(
                operation=task.operation,
                company_text=company,
                retrieval_query=(None if task.retrieval_query == UNSPECIFIED
                                 else task.retrieval_query),
                as_of=(None if task.as_of_expression in {UNSPECIFIED, "현재", "오늘"}
                       else task.as_of_expression),
                doc_group=(None if task.document_group == UNSPECIFIED
                           else task.document_group),
                target_period_expressions=list(
                    task_target_period_expressions[0]
                    if task_target_period_expressions else ()),
                selected_document_receipt=selected_document_receipt,
                requested_slots=list(task.requested_slots),
            )])
            validation = self.delegate.resolve(draft, revision=revision)
            mapping = {
                "draft.tasks[0].company_text": "proposal.tasks[0].company_mentions",
                "draft.tasks[0].retrieval_query": "proposal.tasks[0].retrieval_query",
                "draft.tasks[0].as_of": "proposal.tasks[0].as_of_expression",
                "draft.tasks[0].doc_group": "proposal.tasks[0].document_group",
                "draft.tasks[0].target_period_expressions": (
                    "stage1.document_target_period"),
                "draft.tasks[0].selected_document_receipt": (
                    "stage1.selected_document_receipt"),
            }
            return _translate_validation(validation, mapping)
        return PlanValidation(
            status="unsupported_request",
            reasons=[f"{task.kind}_runtime_not_implemented"],
        )


JsonScalar = str | int | float | bool


class HandoffClarificationSlot(ContractModel):
    slot_id: str = Field(pattern=r"^slot-[1-9][0-9]*$")
    target: str = Field(min_length=1)
    allowed_values: tuple[JsonScalar, ...] = Field(default_factory=tuple)


class HandoffClarification(ContractModel):
    """되묻기 하나. `question` 은 **항목 리스트**다.

    항목이 여럿일 때 한 문자열로 엮으면 경계가 사라져 되돌릴 수 없다. 역질문
    층이 「1) 2) 3)」으로 보이든 한 문장으로 잇든, 그 표현은 층이 정한다.
    다턴 역질문도 항목 단위로 돌기 때문에 색인으로 꺼낼 수 있어야 한다.
    """

    clarification_id: str = Field(min_length=1)
    plan_revision: int = Field(ge=0)
    question: tuple[str, ...] = Field(min_length=1)
    slots: tuple[HandoffClarificationSlot, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def non_blank_questions(self) -> "HandoffClarification":
        if any(not item.strip() for item in self.question):
            raise ValueError("clarification 항목은 공백일 수 없습니다")
        return self

    @model_validator(mode="after")
    def unique_slots(self) -> "HandoffClarification":
        ids = [slot.slot_id for slot in self.slots]
        if len(ids) != len(set(ids)):
            raise ValueError("handoff clarification slot이 중복되었습니다")
        return self


class QueryPlanHandoff(ContractModel):
    """Stage1의 ready/clarification/terminal 상태를 표현하는 v0.4 handoff.

    ready 상태를 Stage2로 넘길 때는 이 객체를 단독 전달하지 않고
    ``Stage1ReadyEnvelope``에서 output/snapshot과 결속한다.
    """

    handoff_id: str = Field(min_length=1)
    contract_version: Literal["0.4"] = QUERY_PLAN_HANDOFF_VERSION
    status: Literal[
        "ready", "needs_clarification", "out_of_scope",
        "unsupported_request", "policy_refusal",
    ]
    plan: ResolvedQueryPlan | None = None
    clarification: HandoffClarification | None = None
    reasons: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_handoff(self) -> "QueryPlanHandoff":
        try:
            UUID(self.handoff_id)
        except (ValueError, AttributeError) as exc:
            raise ValueError("handoff_id는 UUID여야 합니다") from exc
        if self.status == "ready":
            if self.plan is None or self.clarification is not None or self.reasons:
                raise ValueError("ready handoff에는 resolved plan만 있어야 합니다")
        elif self.status == "needs_clarification":
            if self.plan is not None or self.clarification is None or self.reasons:
                raise ValueError("clarification handoff는 실행 plan/reason을 노출할 수 없습니다")
        elif self.plan is not None or self.clarification is not None or not self.reasons:
            raise ValueError("terminal handoff에는 typed reason만 있어야 합니다")
        if any(not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code)
               for code in self.reasons):
            raise ValueError("handoff reason code 형식이 잘못되었습니다")
        return self


def _public_clarification(value: PendingClarification) -> HandoffClarification:
    slots: list[HandoffClarificationSlot] = []
    for slot in value.slots:
        allowed: list[JsonScalar] = []
        for item in slot.allowed_values:
            if not isinstance(item, (str, int, float, bool)) or item is None:
                raise ValueError("public clarification option은 JSON scalar여야 합니다")
            allowed.append(item)
        slots.append(HandoffClarificationSlot(
            slot_id=slot.slot_id,
            target=slot.target,
            allowed_values=tuple(allowed),
        ))
    # 세션은 역질문을 문자열 하나로 들고 있다. 계약은 항목 리스트이므로
    # 한 항목짜리로 싣는다 — 여기서 항목을 쪼갤 근거가 없다.
    question = value.question
    return HandoffClarification(
        clarification_id=value.clarification_id,
        plan_revision=value.created_revision,
        question=(question,) if isinstance(question, str) else tuple(question),
        slots=tuple(slots),
    )


def handoff_from_session(
        state: QuestionSessionState[BaseModel], *, handoff_id: str,
        ) -> QueryPlanHandoff:
    """세션의 내부 patch path를 버리고 공개 v0.4 handoff를 만든다."""

    if state.status == "ready":
        assert state.resolved_plan is not None
        return QueryPlanHandoff(
            handoff_id=handoff_id, status="ready", plan=state.resolved_plan)
    if state.status == "waiting_for_clarification":
        assert state.pending_clarification is not None
        return QueryPlanHandoff(
            handoff_id=handoff_id,
            status="needs_clarification",
            clarification=_public_clarification(state.pending_clarification),
        )
    assert state.terminal_status is not None
    reason_codes = tuple(dict.fromkeys(
        reason if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", reason)
        else state.terminal_status
        for reason in state.terminal_reasons
    ))
    return QueryPlanHandoff(
        handoff_id=handoff_id,
        status=state.terminal_status,  # type: ignore[arg-type]
        reasons=reason_codes,
    )


class ExpectedOutput(ContractModel):
    output_label: str = Field(min_length=1)
    value_text: str = Field(min_length=1)
    raw_unit: str = "not_applicable"


class ExpectedExecutionResult(ContractModel):
    """Gold의 Stage 2~4 기대치를 자유 dict 대신 검증하는 최소 계약."""

    status: Literal[
        "success", "partial", "needs_clarification", "not_found",
        "out_of_scope", "unsupported", "policy_refusal",
    ]
    outputs: tuple[ExpectedOutput, ...] = Field(default_factory=tuple)
    required_doc_ids: tuple[str, ...] = Field(default_factory=tuple)
    required_evidence_ids: tuple[HexId, ...] = Field(default_factory=tuple)
    limitation_codes: tuple[str, ...] = Field(default_factory=tuple)
    clarification_targets: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_expected_result(self) -> "ExpectedExecutionResult":
        if self.status == "success":
            if not self.outputs or not self.required_evidence_ids:
                raise ValueError("Gold success에는 output과 Evidence가 필요합니다")
            if self.limitation_codes or self.clarification_targets:
                raise ValueError("Gold success에는 limitation/clarification이 없어야 합니다")
        elif self.status == "partial":
            if not self.outputs or not self.required_evidence_ids or not self.limitation_codes:
                raise ValueError("Gold partial에는 output/Evidence/limitation이 필요합니다")
        elif self.status == "needs_clarification":
            if (self.outputs or self.required_evidence_ids or self.limitation_codes
                    or not self.clarification_targets):
                raise ValueError("Gold clarification 계약이 잘못되었습니다")
        elif self.outputs or self.required_evidence_ids or self.clarification_targets:
            raise ValueError("Gold non-answer는 확정 output/Evidence를 가질 수 없습니다")
        if len(self.required_doc_ids) != len(set(self.required_doc_ids)):
            raise ValueError("Gold required_doc_ids가 중복되었습니다")
        if len(self.required_evidence_ids) != len(set(self.required_evidence_ids)):
            raise ValueError("Gold required_evidence_ids가 중복되었습니다")
        if (len(self.limitation_codes) != len(set(self.limitation_codes))
                or any(not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code)
                       for code in self.limitation_codes)):
            raise ValueError("Gold limitation code가 잘못되었거나 중복되었습니다")
        if len(self.clarification_targets) != len(set(self.clarification_targets)):
            raise ValueError("Gold clarification target이 중복되었습니다")
        return self


class QuestionFixtureV04(ContractModel):
    schema_version: Literal["query-question/0.4"] = "query-question/0.4"
    question_id: str = Field(pattern=r"^[A-Z]-[A-Z]-[0-9]{3}$")
    semantic_id: str = Field(pattern=r"^[A-Z]-[A-Z]-[0-9]{3}$")
    group: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_date: date
    corpus_cutoff: AsOf


class PlanProposalFixtureV04(ContractModel):
    schema_version: Literal["plan-proposal-fixture/0.4"] = "plan-proposal-fixture/0.4"
    question_id: str = Field(pattern=r"^[A-Z]-[A-Z]-[0-9]{3}$")
    proposal: PlanProposal
    migration_notes: tuple[str, ...] = Field(default_factory=tuple)


class QueryPlanHandoffFixtureV04(ContractModel):
    schema_version: Literal["query-handoff-fixture/0.4"] = "query-handoff-fixture/0.4"
    question_id: str = Field(pattern=r"^[A-Z]-[A-Z]-[0-9]{3}$")
    handoff: QueryPlanHandoff
    migration_notes: tuple[str, ...] = Field(default_factory=tuple)


class AnswerClaimRequirement(ContractModel):
    claim: str = Field(min_length=1)
    value_text: str = "not_applicable"
    raw_value: str = "not_applicable"
    raw_unit: str = "not_applicable"
    canonical_value: str = "not_applicable"
    canonical_unit: str = "not_applicable"


class AnswerRequirement(ContractModel):
    """팀원의 Stage 2~4 구현을 구속하되 최종 문장 생성을 대신하지 않는 계약."""

    schema_version: Literal["answer-requirement/0.4"] = "answer-requirement/0.4"
    question_id: str = Field(pattern=r"^[A-Z]-[A-Z]-[0-9]{3}$")
    source: Literal["core_gold_v1", "variant", "robustness"]
    expected_handoff_status: Literal[
        "ready", "needs_clarification", "out_of_scope",
        "unsupported_request", "policy_refusal",
    ]
    expected_tool_status: str = Field(min_length=1)
    expected_action: str = Field(min_length=1)
    required_documents: tuple[str, ...] = Field(default_factory=tuple)
    required_slots: tuple[str, ...] = Field(default_factory=tuple)
    required_claims: tuple[AnswerClaimRequirement, ...] = Field(default_factory=tuple)
    calculation_requirement: str = "not_applicable"
    behavior_requirements: tuple[str, ...] = Field(default_factory=tuple)
    premise_verdict: str = "not_applicable"
    forbidden_claims: tuple[str, ...] = Field(default_factory=tuple)
    accepted_assumptions: tuple[str, ...] = Field(default_factory=tuple)
    required_limitation_codes: tuple[str, ...] = Field(default_factory=tuple)
    team_reference_answer: str = "not_provided"
    team_reference_answer_ref: str = "not_applicable"
    team_reference_answer_authoritative: Literal[False] = False
    review_status: Literal["candidate_unreviewed", "review_required"]
    review_reasons: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_answer_requirement(self) -> "AnswerRequirement":
        for values, label in (
                (self.required_documents, "required_documents"),
                (self.required_slots, "required_slots"),
                (self.behavior_requirements, "behavior_requirements"),
                (self.forbidden_claims, "forbidden_claims"),
                (self.required_limitation_codes, "required_limitation_codes"),
                (self.review_reasons, "review_reasons")):
            if len(values) != len(set(values)):
                raise ValueError(f"AnswerRequirement {label}가 중복되었습니다")
        if any(not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code)
               for code in self.required_limitation_codes + self.review_reasons):
            raise ValueError("AnswerRequirement typed code 형식이 잘못되었습니다")
        if self.review_status == "review_required" and "independent_review_pending" not in self.review_reasons:
            raise ValueError("review_required에는 independent review pending이 필요합니다")
        return self


__all__ = [
    "AnswerClaimRequirement", "AnswerRequirement", "CurrentSliceProposalResolver",
    "ExpectedExecutionResult", "ExpectedOutput",
    "HandoffClarification", "HandoffClarificationSlot", "PLAN_PROPOSAL_VERSION",
    "PlanProposal", "PlanProposalAdapter", "PlanProposalBackend",
    "PlanProposalOutputError", "PlanProposalParser", "PlanProposalTask",
    "PlanProposalFixtureV04", "ProposedDerivation", "ProposedFact",
    "ProposedPremise", "QuestionFixtureV04", "QueryPlanHandoffFixtureV04",
    "QUERY_PLAN_HANDOFF_VERSION", "QueryPlanHandoff", "handoff_from_session",
]
