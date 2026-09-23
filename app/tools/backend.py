"""CanonicalToolBackend — Orchestrator.ToolBackend 구현.

task kind(financial · event · disclosure · correction · document · narrative)별 typed tool 을 부르고,
검증된 스칼라(citation 있는 값)만으로 파생 계산(DerivationExecutor)과 전제 검증을 한다. 검색 인덱스가
없어 narrative tool 을 만들지 못했으면 그 task 는 `tool_not_implemented` 로 닫는다(거짓 성공 금지).
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from app.orchestrator.payload import AnswerClaim, Limitation, PremiseVerdict, TraceEvent
from app.orchestrator.runner import PlanExecution
from app.tools.canonical_env import read_model
from app.tools.derivation import DerivationExecutor, VerifiedScalar
from app.tools.financial import FinancialTool
from app.tools.events import EventTool, DisclosureTool, CorrectionTool, DocumentTool
from app.tools.holding import HoldingTool, is_holding_task
from app.textkit import complete_source_excerpt
from app.tools.narrative import (
    InvestmentAggregationRequest, InvestmentRowSelector,
    NarrativeFanoutPlan, NarrativeTool,
)
from app.tools._units import parse_money, to_won
from agent.contracts import (
    Derivation, OutputRef, RECENT_Q4_FANOUT_APPLIED_DEFAULT,
)
from agent.contracts import FactSpec, ResolvedFinancialTask

_SUBJECT_PARTICLES = ("가", "이", "은", "는", "의")


def _requests_document_change_details(plan, task) -> bool:
    """Bind version diffs only to an explicit original/correction comparison.

    ``version_history`` is also used for version validity and correction-
    existence questions.  Those shapes must not inherit change-detail
    limitations.  The frozen document task has no projection slots, so use
    only its existing rooted selector plus a plan premise that names both
    document roles; this is semantic topology, not a question/company rule.
    """

    if (task.kind != "document" or task.operation != "version_history"
            or not getattr(task.selector, "rcept_no", None)):
        return False
    for premise in getattr(plan, "premise_claims", ()) or ():
        surface = "".join((getattr(premise, "raw_text", None) or "").split())
        if "원본" in surface and ("정정본" in surface or "정정공시" in surface):
            return True
    return False


def _comparison_subject(raw_text: str, target, by_id: dict) -> str | None:
    """argmax 에 참여한 회사명 가운데 전제 문장에 **가장 먼저** 등장하는 이름. 없으면 None."""
    names: list[str] = []
    for output_id in target.derived_from:
        claim = by_id.get(output_id)
        state = getattr(claim, "state", None) if claim is not None else None
        if state and state not in names:
            names.append(state)
    for entry in (getattr(target, "ranking", None) or []):
        label = getattr(entry, "label", None)
        if label and label not in names:
            names.append(label)
    # 주체는 주격·관형 조사(가/이/은/는/의)가 바로 뒤에 붙은 이름이다 — 「A가 B보다」, 「B보다 A가」 모두
    # A. 「B보다」·「A와」처럼 비교 대상·병렬로 쓰인 자리는 주체가 아니다. 조사가 붙은 이름이 없고 후보가
    # 하나뿐이면 그 이름, 아니면 검증 불가(별칭으로 적힌 주체를 다른 회사로 오인하지 않기 위해).
    def marked_position(name: str) -> int | None:
        position = raw_text.find(name)
        while position >= 0:
            tail = raw_text[position + len(name):position + len(name) + 2]
            if not (tail.startswith("보다") or tail[:1] in ("과", "와")) and tail[:1] in _SUBJECT_PARTICLES:
                return position
            position = raw_text.find(name, position + 1)
        return None

    marked = [(pos, -len(name), name) for name in names if (pos := marked_position(name)) is not None]
    if marked:
        return min(marked)[2]
    present = [name for name in names
               if name in raw_text and not raw_text[raw_text.find(name) + len(name):].startswith("보다")]
    return present[0] if len(present) == 1 else None


def _financial_view_labels(plan) -> dict[str, str]:
    """Label only facts participating in an explicit view comparison.

    QueryPlan v0.4 already represents as-filed/restated as separate financial
    tasks.  The execution layer previously discarded that distinction when it
    built public claim labels, leaving two otherwise identical operands.  This
    derives display labels solely from the existing typed tasks and therefore
    changes no public contract or financial authority.
    """

    grouped: dict[tuple[object, ...], list[tuple[str, str]]] = {}
    for task in plan.tasks:
        if getattr(task, "kind", None) != "financial":
            continue
        for fact in (getattr(task, "facts", None) or ()):
            concept = getattr(fact, "concept", None)
            concept = getattr(concept, "value", concept)
            key = (
                fact.corp_code, str(concept), fact.period_start,
                fact.period_end, fact.period_type, fact.cumulative,
                fact.scope, fact.statement,
            )
            grouped.setdefault(key, []).append((fact.output_id, task.view))
    labels: dict[str, str] = {}
    for rows in grouped.values():
        if {view for _output_id, view in rows} != {"as_filed", "restated"}:
            continue
        for output_id, view in rows:
            labels[output_id] = (
                "최초 제출값" if view == "as_filed" else "최신 재작성값")
    return labels


class CanonicalToolBackend:
    def __init__(self, rm=None, search_index=None,
                 narrative_sidecars: dict[str, NarrativeFanoutPlan] | None = None):
        self.rm = rm or read_model()
        self.search_index = search_index
        if self.search_index is None:
            try:
                from app.tools.canonical_env import search_index as _si
                self.search_index = _si()
            except Exception:
                self.search_index = None   # 인덱스 미빌드 → narrative는 tool_not_implemented로 닫힘
        self.fin = FinancialTool(self.rm)
        self.der = DerivationExecutor()
        from app.tools.field_index import FieldIndex
        self.fidx = FieldIndex(self.rm)
        self.evt = EventTool(self.rm, self.fidx)
        self.disc = DisclosureTool(self.rm, self.fidx)
        self.corr = CorrectionTool(self.rm, self.fidx)
        self.doc = DocumentTool(self.rm, self.fidx)
        self.holding = HoldingTool(self.rm, self.fidx)
        self.narr = NarrativeTool(self.rm, self.search_index) if self.search_index else None
        # This execution-only mapping is how a typed planner can provide
        # multiple grounded topics without extending the public v0.4 task.
        self.narrative_sidecars = dict(narrative_sidecars or {})

    def execute_plan(
            self, plan, *, trace: list[TraceEvent],
            narrative_row_selectors: dict[str, InvestmentRowSelector] | None = None,
            narrative_aggregations: dict[str, InvestmentAggregationRequest] | None = None,
            ) -> PlanExecution:
        claims, lims, used = [], [], []
        narrative_results = []
        verified: dict[str, VerifiedScalar] = {}
        clarification = None
        row_selectors = dict(narrative_row_selectors or {})
        aggregations = dict(narrative_aggregations or {})
        if any(not isinstance(value, InvestmentRowSelector)
               for value in row_selectors.values()):
            raise TypeError("investment row selector 타입이 잘못되었습니다")
        if any(not isinstance(value, InvestmentAggregationRequest)
               for value in aggregations.values()):
            raise TypeError("investment aggregation 타입이 잘못되었습니다")
        # 정의 고지 문구는 요청 단위다. backend 는 서버에서 싱글턴이므로, 역질문 조기 반환·예외로
        # 끝난 이전 요청의 문구가 다음 답변에 붙지 않도록 실행 시작 시 비운다.
        self.fin.definitions_used = []
        financial_view_labels = _financial_view_labels(plan)

        for task in plan.tasks:
            if task.kind == "financial":
                run_kwargs = {"trace": trace}
                if financial_view_labels:
                    run_kwargs["view_labels"] = financial_view_labels
                for o in self.fin.run_task(task, **run_kwargs):
                    if o.clarification is not None and clarification is None:
                        clarification = o.clarification
                    if o.claim is not None:
                        claims.append(o.claim)
                        used.append(o.rcept_no)
                        if o.decimal_value is not None:
                            verified[o.output_id] = VerifiedScalar(o.output_id, o.decimal_value, o.unit, o.won, o.claim)
                    if o.limitation is not None:
                        lims.append(o.limitation)
            elif task.kind == "narrative" and self.narr is not None:
                c, l, u, sidecar = self.narr.run_task(
                    task, trace=trace, corp_name=", ".join(task.corp_names),
                    sidecar=self.narrative_sidecars.get(task.task_id),
                    row_selector=row_selectors.get(task.task_id),
                    aggregation_request=aggregations.get(task.task_id),
                    prefer_latest=any(
                        isinstance(default, str) and "최근" in default
                        for default in plan.applied_defaults))
                claims.extend(c); lims.extend(l); used.extend(u)
                if sidecar is not None:
                    narrative_results.append(sidecar)
            elif task.kind == "disclosure" and is_holding_task(task):
                c, l, u, cl = self.holding.run_task(
                    task, trace=trace, corp_name=task.corp_name)
                claims.extend(c); lims.extend(l); used.extend(u)
                if cl is not None and clarification is None:
                    clarification = cl
                for cm in c:
                    if cm.value_text and cm.output_id and not cm.derived_from:
                        dec, unit, won = parse_money(cm.value_text)
                        if dec is not None:
                            if unit is None and cm.raw_unit:
                                unit = cm.raw_unit
                                won = (to_won(dec, unit)
                                       if unit not in {"%", "%p", "주"} else None)
                            verified[cm.output_id] = VerifiedScalar(
                                cm.output_id, dec, unit, won, cm)
            elif task.kind in ("event", "disclosure", "correction", "document"):
                tool = {"event": self.evt, "disclosure": self.disc,
                        "correction": self.corr, "document": self.doc}[task.kind]
                tool_kwargs = {
                    "trace": trace, "corp_name": task.corp_name}
                if task.kind == "document":
                    tool_kwargs["include_change_details"] = (
                        _requests_document_change_details(plan, task))
                c, l, u, cl = tool.run_task(task, **tool_kwargs)
                claims.extend(c); lims.extend(l); used.extend(u)
                if cl is not None and clarification is None:
                    clarification = cl
                # 수치 claim(field_outputs·정정 후 값·slot 값)은 계산·전제 검증에 참여
                for cm in c:
                    if cm.value_text and cm.output_id and not cm.derived_from:
                        dec, unit, won = parse_money(cm.value_text)
                        if dec is not None:
                            # Structured form values often store the unit in
                            # the field path rather than the value cell.  Tool
                            # claims preserve that explicit path unit; use it
                            # only when the value surface itself omitted one.
                            if unit is None and cm.raw_unit:
                                unit = cm.raw_unit
                                won = (to_won(dec, unit)
                                       if unit != "%" else None)
                            verified[cm.output_id] = VerifiedScalar(cm.output_id, dec, unit, won, cm)
                            if cm.canonical_value is None and won is not None:
                                cm.canonical_value = str(won); cm.canonical_unit = "원"
            else:
                trace.append(TraceEvent(seq=len(trace)+1, stage="tool",
                                        summary=f"{task.kind} task 를 실행할 tool 없음", detail={"task_id": task.task_id}))
                lims.append(Limitation(code="tool_not_implemented",
                                       detail=f"{task.kind} task 를 실행할 tool 이 없습니다(검색 인덱스 미로드 또는 미지원 kind)."))

        # HCX may preserve the two verified scalar lookups while omitting the
        # comparison/reason roots that connect them.  Recover only the closed,
        # typed topology ``one contract amount + one termination amount``
        # across exact-document disclosure/correction tasks.  This is an
        # execution-side robustness rule, not a question/issuer dispatch.
        pair_claims, pair_lims = self._supplement_contract_termination_pair(
            plan, claims, verified, trace)
        claims.extend(pair_claims)
        lims.extend(pair_lims)

        ratio_claims, ratio_lims, ratio_used = self._supplement_investment_ratio(
            plan, claims, verified, trace, aggregations=aggregations)
        claims.extend(ratio_claims)
        lims.extend(ratio_lims)
        used.extend(ratio_used)

        # 이슈 #62 — 대량보유보고서 한 건 안의 직전/이번 총 보유비율 두 슬롯
        # 비교(%p·상대 증감률). 컴파일러가 holding field에 대한 derivation
        # 모양을 아직 못 내므로(#40), 이미 검증된 두 슬롯 claim에서 실행 시
        # 직접 만든다 — 위 계약/해지 쌍 보강과 같은 자리, 같은 패턴이다.
        holding_ratio_claims, holding_ratio_lims = (
            self._supplement_holding_ratio_change(plan, claims, verified, trace))
        claims.extend(holding_ratio_claims)
        lims.extend(holding_ratio_lims)

        # Several exact-document tasks on the same receipt date can prove the
        # candidate set, but a DART receipt number carries no intraday clock.
        # Preserve that ordering boundary whenever the plan also asks for the
        # related event state; never choose a global "latest" by receipt order.
        document_receipts = [
            getattr(getattr(task, "selector", None), "rcept_no", None)
            for task in plan.tasks
            if getattr(task, "kind", None) == "document"
            and getattr(task, "operation", None) == "find"]
        document_receipts = [
            receipt for receipt in document_receipts
            if isinstance(receipt, str) and len(receipt) >= 8]
        if (any(getattr(task, "kind", None) == "event" for task in plan.tasks)
                and len(document_receipts) >= 2):
            by_day: dict[str, list[str]] = {}
            for receipt in document_receipts:
                by_day.setdefault(receipt[:8], []).append(receipt)
            for day, receipts in sorted(by_day.items()):
                unique = sorted(set(receipts))
                if len(unique) < 2:
                    continue
                lims.append(Limitation(
                    code="intraday_order_unavailable",
                    detail=(f"{day} 동일일 공시 {', '.join(unique)}: "
                            "접수 시각이 없어 선후 확정 불가"),
                    affected_doc_ids=unique))

        if clarification is not None:
            return PlanExecution(clarification=clarification)

        # 계산
        if plan.derivations:
            # 보조값(증감액·증감률) 판별용 — operand가 같은 회사·개념의 기간 비교인지 plan facts로 확인
            fact_meta = {}
            for task in plan.tasks:
                for f in (getattr(task, "facts", None) or []):
                    pe = getattr(f, "period_end", None)
                    ps = getattr(f, "period_start", None)
                    fact_meta[f.output_id] = {
                        "corp_code": getattr(f, "corp_code", None),
                        "corp_name": getattr(f, "corp_name", None),
                        "concept": getattr(f, "concept", None),
                        "period_start": ps.isoformat() if ps is not None else None,
                        "period_end": pe.isoformat() if pe is not None else None,
                        "period_type": getattr(f, "period_type", None),
                        "cumulative": getattr(f, "cumulative", None),
                        "scope": getattr(f, "scope", None),
                    }
            dclaims, dlims = self.der.execute(plan.derivations, verified, trace=trace, fact_meta=fact_meta)
            claims.extend(dclaims); lims.extend(dlims)
            # 이슈 #64 — 연간 값이 4분기 단독(discrete_from_cumulative FY−9M)의
            # 재료로도 함께 답에 쓰이면, 기간 길이가 달라 그 차이를 분기
            # 성장률로 부를 수 없다는 고지를 붙인다. 두 값 자체는 그대로 둔다.
            lims.extend(self._supplement_period_length_notice(
                plan, claims, fact_meta))
            # difference 결과에 동일/상이 판정을 typed로 병기 (질문이 "같은가"를 묻는 경우 composer 재료)
            # — plan이 명시한 difference에만. 보조값(.delta)에는 붙이지 않는다
            plan_der_ids = {d.output_id for d in plan.derivations}
            for dc in dclaims:
                if dc.operator in ("difference", "absolute_difference") and dc.value_text is not None and dc.output_id in plan_der_ids:
                    try:
                        z = Decimal(dc.value_text.replace(",", "")) == 0
                    except Exception:
                        continue
                    claims.append(AnswerClaim(output_id=f"{dc.output_id}.same", label="동일 여부",
                                              state="same" if z else "different", text="같다" if z else "다르다",
                                              derived_from=list(dc.derived_from), operator="compare",
                                              citations=dc.citations))
        # RPC-002 — 연결·별도 짝(financial/parallel-scope-retrieval)은 파생을
        # 만들지 않으므로 위 ``if plan.derivations:`` 안이 아니라 여기서
        # 무조건 확인한다. 두 값이 함께 답에 실렸을 때만 고지가 붙는다.
        lims.extend(self._supplement_scope_mismatch_notice(plan, claims))
        # 전제 검증 (numeric · comparison · state · existence)
        verdicts = self._verify_premises(plan, claims, verified, trace)

        ex = PlanExecution(claims=claims, limitations=lims, premise_verdicts=verdicts,
                           used_documents=[u for u in used if u])
        ex.notes = list(self.fin.definitions_used)   # composer 필수 고지 문구
        ex.narrative_sidecars = narrative_results
        self.fin.definitions_used = []
        return ex

    def _supplement_investment_ratio(
            self, plan, claims, verified, trace, *, aggregations):
        """Divide one proved row sum by one exact annual revenue operand."""

        additions, limitations, used = [], [], []
        for task in plan.tasks:
            if getattr(task, "kind", None) != "narrative":
                continue
            request = aggregations.get(task.task_id)
            if request is None or request.revenue_surface is None:
                continue
            sums = [
                claim for claim in claims
                if claim.output_id == f"{task.task_id}.investment-sum"
                and claim.operator == "sum"
                and claim.canonical_unit == "원"
                and claim.canonical_value is not None
            ]
            if len(sums) != 1:
                limitations.append(Limitation(
                    code="investment_ratio_missing_sum",
                    detail="매출액 대비 비율의 검증된 투자계획 합계가 없어 계산하지 않았습니다."))
                continue
            corp_codes = tuple(getattr(task, "corp_codes", ()) or ())
            corp_names = tuple(getattr(task, "corp_names", ()) or ())
            if (len(corp_codes) != 1 or len(corp_names) != 1
                    or request.revenue_year is None
                    or request.revenue_scope not in {"CFS", "SFS"}):
                limitations.append(Limitation(
                    code="investment_ratio_operand_unresolved",
                    detail="매출액 대비 비율의 회사·연도·연결/별도 축을 하나로 확정하지 못했습니다."))
                continue
            year = request.revenue_year
            output_id = f"{task.task_id}.investment-ratio-revenue"
            financial_task = ResolvedFinancialTask(
                task_id=f"{task.task_id}.investment-ratio-financial",
                as_of=task.as_of, view="restated",
                facts=[FactSpec(
                    output_id=output_id, corp_code=corp_codes[0],
                    corp_name=corp_names[0], concept="revenue",
                    period_start=date(year, 1, 1),
                    period_end=date(year, 12, 31), period_type="annual",
                    cumulative=True, scope=request.revenue_scope,
                    statement=None, account_path=None, unit=None,
                )],
            )
            outcomes = self.fin.run_task(financial_task, trace=trace)
            outcome = outcomes[0] if len(outcomes) == 1 else None
            if outcome is None or outcome.claim is None \
                    or outcome.won is None or not outcome.claim.citations:
                if outcome is not None and outcome.limitation is not None:
                    limitations.append(outcome.limitation)
                else:
                    limitations.append(Limitation(
                        code="investment_ratio_operand_unresolved",
                        detail="매출액 대비 비율의 검증된 매출액 피연산자를 확정하지 못했습니다."))
                continue
            revenue = outcome.claim
            additions.append(revenue)
            if outcome.rcept_no:
                used.append(outcome.rcept_no)
            total_won = Decimal(sums[0].canonical_value)
            if outcome.won == 0:
                limitations.append(Limitation(
                    code="investment_ratio_zero_denominator",
                    detail="매출액 피연산자가 0이어서 비율을 계산하지 않았습니다."))
                continue
            ratio = (total_won / outcome.won * Decimal(100)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP)
            citations = list(sums[0].citations)
            seen = {(c.doc_id, c.locator, c.evidence_id) for c in citations}
            citations.extend(
                citation for citation in revenue.citations
                if (citation.doc_id, citation.locator, citation.evidence_id)
                not in seen)
            additions.append(AnswerClaim(
                output_id=f"{task.task_id}.investment-revenue-ratio",
                label=(f"{sums[0].label}의 {request.revenue_surface} 대비 비율"),
                value_text=f"{ratio:f}", raw_unit="%",
                canonical_value=f"{ratio:f}", canonical_unit="%",
                derived_from=[sums[0].output_id, revenue.output_id],
                operator="ratio", citations=citations,
            ))
            trace.append(TraceEvent(
                seq=len(trace) + 1, stage="derivation",
                summary="투자계획 합계 / 연간 매출액 비율 계산",
                detail={"output_id": additions[-1].output_id},
            ))
        return additions, limitations, used

    def _supplement_holding_ratio_change(self, plan, claims, verified, trace):
        """One holding filing's own 직전/이번 총 보유비율 slots, compared.

        Issue #62 (RPC-011) — 「20.75%에서 20.74%로」는 정정 체인 두 문서가
        아니라 **같은 대량보유보고서** 안의 두 슬롯(SUM_BMT_RT/SUM_TMT_RT)
        이다. The holding compiler has no derivation shape for these fields
        yet (#40 계열), so — exactly like ``_supplement_contract_termination_
        pair`` above — build the comparison here from claims the tool already
        proved, once per receipt, only when the pairing is unambiguous.
        """

        additions: list[AnswerClaim] = []
        limitations: list[Limitation] = []
        scalar_claims = [
            claim for claim in claims
            if claim.output_id in verified and not claim.derived_from
            and claim.raw_unit == "%"
        ]

        def bucket(claim) -> str | None:
            label = claim.label or ""
            if label.endswith("직전 보고서 보유비율"):
                return "previous"
            if label.endswith("이번 보고서 보유비율") or label.endswith("보유비율"):
                return "current"
            return None

        def receipt_of(claim) -> str | None:
            return next((c.rcept_no for c in claim.citations if c.rcept_no), None)

        grouped: dict[str, dict[str, list]] = {}
        for claim in scalar_claims:
            key = bucket(claim)
            receipt = receipt_of(claim)
            if key is None or not receipt:
                continue
            grouped.setdefault(receipt, {}).setdefault(key, []).append(claim)

        for receipt, buckets in grouped.items():
            previous_rows = buckets.get("previous") or []
            current_rows = buckets.get("current") or []
            # More than one candidate per slot (e.g. the report total plus a
            # named special-related-party row) means the pairing is not
            # provably unique — do not guess which two cells to compare.
            if len(previous_rows) != 1 or len(current_rows) != 1:
                continue
            previous, current = previous_rows[0], current_rows[0]
            pair_ids = {previous.output_id, current.output_id}
            if any(
                    derivation.operator in {"difference", "percent_change"}
                    and {ref.output_id for ref in derivation.operands} == pair_ids
                    for derivation in plan.derivations):
                continue
            runtime_derivations = [
                Derivation(
                    output_id=f"{current.output_id}.holding-ratio-diff",
                    operator="difference",
                    operands=[OutputRef(output_id=current.output_id),
                              OutputRef(output_id=previous.output_id)]),
                Derivation(
                    output_id=f"{current.output_id}.holding-ratio-pct",
                    operator="percent_change",
                    operands=[OutputRef(output_id=current.output_id),
                              OutputRef(output_id=previous.output_id)]),
            ]
            derived, derivation_limits = self.der.execute(
                runtime_derivations, verified, trace=trace)
            additions.extend(derived)
            limitations.extend(derivation_limits)
            trace.append(TraceEvent(
                seq=len(trace) + 1, stage="derivation",
                summary="holding 총 보유비율 직전/이번 슬롯 보강(%p·상대 증감률)",
                detail={"receipt": receipt, "previous": previous.output_id,
                        "current": current.output_id}))
        return additions, limitations

    def _supplement_scope_mismatch_notice(self, plan, claims):
        """Notice when a CFS and an SFS value for the same fact both answer.

        RPC-002 — 연결과 별도는 같은 회사·계정·기간이라도 종속회사 포함
        여부가 다른 서로 다른 집계 범위다(이슈 #64/RPC-003 이 기간 길이에
        대해 하는 일과 같은 뜻이고, 축만 시간이 아니라 범위다). 두 값이
        함께 답에 실리면 그 차이를 「전년 대비 증감률」 같은 시간 증감으로
        부르지 않도록 무조건 고지한다 — 사용자가 판단을 요청했는지와
        무관하다(#64 선례와 동일한 설계).

        `plan.tasks` 의 FactSpec 을 직접 훑는다. CFS/SFS 짝
        (`financial/parallel-scope-retrieval`)은 파생을 만들지 않는 순수
        조회 전용이라 `plan.derivations` 로는 이 짝을 볼 수 없다.
        """

        base_ids = {claim.output_id for claim in claims if not claim.derived_from}
        by_axis: dict[tuple, dict[str, str]] = {}
        for task in plan.tasks:
            for fact in (getattr(task, "facts", None) or []):
                output_id = getattr(fact, "output_id", None)
                scope = getattr(fact, "scope", None)
                if (output_id is None or output_id not in base_ids
                        or scope not in ("CFS", "SFS")):
                    continue
                axis = (
                    getattr(fact, "corp_code", None), getattr(fact, "concept", None),
                    getattr(fact, "period_start", None),
                    getattr(fact, "period_end", None),
                    getattr(fact, "period_type", None),
                    getattr(fact, "cumulative", None),
                )
                if axis[0] is None or axis[1] is None:
                    continue
                by_axis.setdefault(axis, {})[scope] = output_id
        limitations: list[Limitation] = []
        for scoped in by_axis.values():
            if set(scoped) != {"CFS", "SFS"}:
                continue
            pair = ", ".join(scoped[key] for key in ("CFS", "SFS"))
            limitations.append(Limitation(
                code="scope_mismatch",
                detail=(f"{pair}: 연결 값과 별도 값은 집계 범위가 달라 "
                        "시간 증감으로 계산하지 않았습니다.")))
        return limitations

    def _supplement_period_length_notice(self, plan, claims, fact_meta):
        """Notice when an annual value also seeds a discrete quarter value.

        Issue #64 (RPC-003) — a plan that answers both the full-year figure
        and the quarter it fed into ``discrete_from_cumulative`` (FY − 9M)
        must not let a reader mistake their gap for a comparable time-series
        change; DerivationExecutor already refuses to compute that gap
        itself (see ``_period_over_period_order`` in derivation.py). This
        adds the public notice unconditionally whenever both values are in
        the answer, regardless of whether a difference was ever attempted —
        neither value nor any other claim is touched.
        """

        # In a recent-quarter fanout the annual value is an internal operand,
        # not a separately requested answer.  Its compiler-owned public plan
        # note makes that distinction explicit; the FY-vs-quarter warning
        # would otherwise turn a complete timeline into a partial answer.
        if RECENT_Q4_FANOUT_APPLIED_DEFAULT in (
                getattr(plan, "applied_defaults", ()) or ()):
            return []

        base_ids = {claim.output_id for claim in claims if not claim.derived_from}
        limitations: list[Limitation] = []
        seen: set[str] = set()
        for derivation in plan.derivations:
            if (derivation.operator != "discrete_from_cumulative"
                    or len(derivation.operands) != 2):
                continue
            annual_id = derivation.operands[0].output_id
            if annual_id in seen or annual_id not in base_ids:
                continue
            annual_meta = fact_meta.get(annual_id) or {}
            if annual_meta.get("period_type") != "annual":
                continue
            time_relations = [row for row in plan.derivations
                              if row.operator in {"percent_change", "difference", "absolute_difference"}]
            relation_inputs = {ref.output_id for row in time_relations for ref in row.operands}
            # The annual fact is only an internal FY−9M operand when the
            # requested time change consumes the resulting quarter instead.
            # Retain the warning if any time relation directly uses FY.
            if (derivation.output_id in relation_inputs
                    and annual_id not in relation_inputs):
                continue
            seen.add(annual_id)
            limitations.append(Limitation(
                code="period_length_mismatch",
                detail=(f"{annual_id}: 연간 값과 그 값으로 만든 분기 단독 값은 "
                        "기간 길이가 달라 시간 증감으로 계산하지 않았습니다.")))
        return limitations

    def _supplement_contract_termination_pair(
            self, plan, claims, verified, trace):
        scalar_claims = [
            claim for claim in claims
            if claim.output_id in verified and not claim.derived_from]

        def compact(value: str | None) -> str:
            import re
            return re.sub(r"[^0-9A-Za-z가-힣]", "", value or "").casefold()

        contract_amounts = [
            claim for claim in scalar_claims
            if "계약금액" in compact(claim.label)
            and "해지금액" not in compact(claim.label)]
        termination_amounts = [
            claim for claim in scalar_claims
            if "해지금액" in compact(claim.label)]
        if len(contract_amounts) != 1 or len(termination_amounts) != 1:
            return [], []
        contract, termination = contract_amounts[0], termination_amounts[0]
        pair_ids = {contract.output_id, termination.output_id}
        if any(
                derivation.operator in {
                    "equal", "difference", "absolute_difference"}
                and {operand.output_id for operand in derivation.operands}
                == pair_ids
                for derivation in plan.derivations):
            return [], []

        def receipt_of(claim):
            return next((citation.rcept_no for citation in claim.citations
                         if citation.rcept_no), None)

        contract_receipt = receipt_of(contract)
        termination_receipt = receipt_of(termination)
        if (not contract_receipt or not termination_receipt
                or contract_receipt == termination_receipt):
            return [], []
        tasks_by_receipt = {}
        for task in plan.tasks:
            if task.kind not in {"disclosure", "correction"}:
                continue
            selector = getattr(task, "document_selector", None)
            receipt = getattr(selector, "rcept_no", None)
            if receipt:
                tasks_by_receipt[receipt] = task
        contract_task = tasks_by_receipt.get(contract_receipt)
        termination_task = tasks_by_receipt.get(termination_receipt)
        if (contract_task is None or termination_task is None
                or contract_task.corp_code != termination_task.corp_code):
            return [], []

        runtime_derivations = [
            Derivation(
                output_id=f"{termination.output_id}.runtime-equal",
                operator="equal",
                operands=[OutputRef(output_id=contract.output_id),
                          OutputRef(output_id=termination.output_id)]),
            Derivation(
                output_id=f"{termination.output_id}.runtime-gap",
                operator="absolute_difference",
                operands=[OutputRef(output_id=contract.output_id),
                          OutputRef(output_id=termination.output_id)]),
        ]
        derived, derivation_limits = self.der.execute(
            runtime_derivations, verified, trace=trace)

        explanation_claims = []
        explanation_limits = []
        explanation, status = self.disc.field_value(
            termination_task.corp_code, termination_receipt,
            "8. 기타 투자판단과 관련한 중요사항",
            as_of=termination_task.as_of, trace=trace)
        explanation_excerpt = complete_source_excerpt(
            explanation.value if explanation is not None else None,
            max_chars=1200)
        if (explanation is not None and status == "verified"
                and explanation.value and explanation.value.strip() != "-"
                and explanation_excerpt):
            from app.tools.events import _cite_field
            explanation_claims.append(AnswerClaim(
                output_id=f"{termination.output_id}.runtime-basis",
                label="해지금액 산정 근거",
                text=explanation_excerpt,
                operator="source_explanation",
                citations=[_cite_field(explanation)]))
        elif (explanation is not None and status == "verified"
              and explanation.value and explanation.value.strip() != "-"):
            explanation_limits.append(Limitation(
                code="comparison_explanation_excerpt_unavailable",
                detail=("계약금액과 해지금액의 차이는 계산했으며 같은 "
                        "해지공시에 산정 근거도 있으나, 공개 길이 안에 "
                        "완결된 문장·항목 경계가 없어 중간 절단 없이 "
                        "표시하지 않음"),
                affected_doc_ids=[termination_receipt]))
        else:
            explanation_limits.append(Limitation(
                code="comparison_explanation_not_found",
                detail=("계약금액과 해지금액의 차이는 계산했으나 "
                        "같은 해지공시에서 산정 근거를 확인하지 못해 "
                        "원인은 단정하지 않음"),
                affected_doc_ids=[termination_receipt]))
        trace.append(TraceEvent(
            seq=len(trace)+1, stage="derivation",
            summary="typed contract/termination amount pair 보강",
            detail={
                "contract_output": contract.output_id,
                "termination_output": termination.output_id,
                "explanation": bool(explanation_claims),
            }))
        return derived + explanation_claims, derivation_limits + explanation_limits

    # ────────────────────────────────────────────────────────────────────
    def _verify_premises(self, plan, claims, verified, trace) -> list[PremiseVerdict]:
        out: list[PremiseVerdict] = []
        by_id = {c.output_id: c for c in claims}
        for pc in plan.premise_claims:
            refs = [r.output_id for r in pc.verify_with]
            if pc.kind == "numeric" and pc.value is not None:
                target = next((by_id[r] for r in refs if r in by_id), None)
                if target is None:
                    target = next((c for c in claims if c.canonical_value and not c.derived_from), None)
                if target is None or target.canonical_value is None:
                    out.append(PremiseVerdict(claim_id=pc.claim_id, verdict="unverifiable",
                                              detail="비교할 검증값 없음", compared_output_ids=refs))
                    continue
                try:
                    claimed = to_won(Decimal(pc.value.replace(",", "")), pc.unit)
                except Exception:
                    claimed = None
                actual = Decimal(target.canonical_value)
                if claimed is None:
                    out.append(PremiseVerdict(claim_id=pc.claim_id, verdict="unverifiable",
                                              detail=f"단위 {pc.unit} 환산 불가", compared_output_ids=refs))
                    continue
                # 1% 이내면 참으로 인정 (사용자 어림 표현 허용)
                same = actual != 0 and abs(claimed - actual) / abs(actual) <= Decimal("0.01")
                out.append(PremiseVerdict(claim_id=pc.claim_id, verdict="true" if same else "false",
                                          detail=f"주장 {claimed:,.0f}원 vs 실제 {actual:,.0f}원",
                                          compared_output_ids=[target.output_id]))
                trace.append(TraceEvent(seq=len(trace)+1, stage="premise",
                                        summary=f"premise {pc.claim_id} → {'true' if same else 'false'}",
                                        detail={"claimed": str(claimed), "actual": str(actual)}))
            elif pc.kind == "comparison":
                # An explicit equality derivation is the authoritative result
                # for a "same/different" premise.  Argmax remains the route for
                # winner claims; do not reinterpret one as the other.
                equal_target = next((
                    by_id[ref] for ref in refs
                    if ref in by_id and by_id[ref].operator == "equal"), None)
                if equal_target is None:
                    equal_target = next((
                        claim for claim in claims
                        if claim.operator == "equal"
                        and set(claim.derived_from) == set(refs)), None)
                raw = pc.raw_text
                says_equal = any(token in raw for token in (
                    "같", "동일", "일치")) and not any(
                        token in raw for token in ("다르", "차이", "불일치"))
                says_different = any(token in raw for token in (
                    "다르", "차이", "불일치"))
                if equal_target is not None and (says_equal or says_different):
                    actual_equal = equal_target.state == "same"
                    same = (actual_equal if says_equal else not actual_equal)
                    out.append(PremiseVerdict(
                        claim_id=pc.claim_id,
                        verdict="true" if same else "false",
                        detail=(f"주장 '{raw}' vs 검증 결과 "
                                f"{equal_target.text}"),
                        compared_output_ids=[equal_target.output_id]))
                    continue
                target = next((by_id[r] for r in refs if r in by_id), None)
                if target is None or target.operator != "argmax":
                    target = next((c for c in claims if c.operator == "argmax"), None)
                if target is None:
                    out.append(PremiseVerdict(claim_id=pc.claim_id, verdict="unverifiable",
                                              detail="비교 결과 없음", compared_output_ids=refs))
                    continue
                # 주장 주체 = 비교에 참여한 회사명 가운데 전제 문장에서 가장 먼저 나오는 이름
                # (「A가 B보다 크다」의 A). 조사 「가/이」로 문장을 자르면 이마트·LG이노텍·SK하이닉스처럼
                # 이름에 그 글자가 든 회사가 깨진다. 어떤 회사명도 문장에 없으면 검증 불가로 둔다.
                winner = target.state or ""
                subject = _comparison_subject(pc.raw_text, target, by_id)
                if not subject:
                    out.append(PremiseVerdict(claim_id=pc.claim_id, verdict="unverifiable",
                                              detail="전제 문장에서 비교 주체(회사명)를 찾지 못함",
                                              compared_output_ids=[target.output_id]))
                    continue
                same = bool(winner) and subject == winner
                out.append(PremiseVerdict(claim_id=pc.claim_id, verdict="true" if same else "false",
                                          detail=f"주장 주체 '{subject}' vs 실제 최대 '{winner}'",
                                          compared_output_ids=[target.output_id]))
            elif pc.kind == "state":
                # Direction words on a correction amount are not event-state
                # claims.  When the earlier value was withheld, publication of
                # the later value proves neither an increase nor a decrease.
                # Mark the asserted direction false instead of comparing it to
                # active/terminated event state.
                direction = pc.raw_text
                says_up = any(token in direction for token in (
                    "늘어", "증가", "커지", "상승"))
                says_down = any(token in direction for token in (
                    "줄어", "감소", "작아", "하락"))
                if says_up or says_down:
                    disclosure_claim = next((
                        claim for claim in claims
                        if claim.state == "disclosed_from_withheld"), None)
                    if disclosure_claim is not None:
                        out.append(PremiseVerdict(
                            claim_id=pc.claim_id, verdict="false",
                            detail=("정정 전 값이 공시유보·미기재 상태여서 "
                                    "정정 후 공개를 증감으로 판정할 수 없음"),
                            compared_output_ids=[disclosure_claim.output_id]))
                        continue
                # 사건 상태 claim(state=active/terminated)과 전제 문구를 대조
                st_claims = [c for c in claims if c.state in ("active", "terminated", "not_disclosed")]
                if not st_claims:
                    out.append(PremiseVerdict(claim_id=pc.claim_id, verdict="unverifiable",
                                              detail="사건 상태 확인 불가", compared_output_ids=refs)); continue
                actual = st_claims[-1].state   # 가장 늦은 시점의 상태
                t = pc.raw_text
                says_alive = any(k in t for k in ("유효", "살아", "진행", "정상 완료", "완료"))
                says_dead = any(k in t for k in ("해지", "끝", "종료", "깨진", "취소"))
                if says_alive and says_dead:
                    # A two-timepoint premise is a conjunction, not a claim
                    # about only the latest state.  Compare the ordered state
                    # sequence when the sentence contains one explicit alive
                    # cue and one explicit terminated cue.  This keeps
                    # ``A일 유효, B일 해지`` true when both source-backed
                    # observations agree, while any missing/mismatched point
                    # remains false or unverifiable instead of being collapsed
                    # to the final observation.
                    alive_positions = [
                        t.find(token) for token in
                        ("유효", "살아", "진행", "정상 완료", "완료")
                        if token in t]
                    dead_positions = [
                        t.find(token) for token in
                        ("해지", "끝", "종료", "깨진", "취소")
                        if token in t]
                    if len(st_claims) < 2:
                        out.append(PremiseVerdict(
                            claim_id=pc.claim_id, verdict="unverifiable",
                            detail="복수 시점 전제를 검증할 상태 관측이 부족함",
                            compared_output_ids=[c.output_id for c in st_claims]))
                        continue
                    expected = (
                        ["active", "terminated"]
                        if min(alive_positions) < min(dead_positions)
                        else ["terminated", "active"])
                    actual_states = [claim.state for claim in st_claims[:2]]
                    same = actual_states == expected
                    out.append(PremiseVerdict(
                        claim_id=pc.claim_id,
                        verdict="true" if same else "false",
                        detail=(f"주장 '{t}' vs 실제 상태 흐름 "
                                f"{' → '.join(actual_states)}"),
                        compared_output_ids=[
                            claim.output_id for claim in st_claims[:2]]))
                    continue
                if says_dead:
                    same = actual == "terminated"
                elif says_alive:
                    same = actual == "active"
                else:
                    out.append(PremiseVerdict(claim_id=pc.claim_id, verdict="unverifiable",
                                              detail="전제 문구의 상태 방향 불명", compared_output_ids=refs)); continue
                out.append(PremiseVerdict(claim_id=pc.claim_id, verdict="true" if same else "false",
                                          detail=f"주장 '{t}' vs 실제 상태 {actual}",
                                          compared_output_ids=[st_claims[-1].output_id]))
            elif pc.kind == "existence":
                doc_claims = [c for c in claims if c.state in ("corrected", "no_correction")]
                if not doc_claims:
                    out.append(PremiseVerdict(claim_id=pc.claim_id, verdict="unverifiable",
                                              detail="문서 정정 이력 확인 불가", compared_output_ids=refs)); continue
                any_corr = any(c.state == "corrected" for c in doc_claims)
                says_none = "없" in pc.raw_text
                same = (not any_corr) if says_none else any_corr
                out.append(PremiseVerdict(claim_id=pc.claim_id, verdict="true" if same else "false",
                                          detail=f"주장 '{pc.raw_text}' vs 정정 {'있음' if any_corr else '없음'}",
                                          compared_output_ids=[c.output_id for c in doc_claims]))
            else:
                out.append(PremiseVerdict(claim_id=pc.claim_id, verdict="unverifiable",
                                          detail=f"{pc.kind} 전제는 자동 검증 대상 아님", compared_output_ids=refs))
        return out
