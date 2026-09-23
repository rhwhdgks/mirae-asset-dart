"""FinancialTool — ResolvedFinancialTask → FactSpec별 AnswerClaim.

원칙: rm.lookup()을 호출만 한다. 판정(모호성·coverage·Evidence)은 read.py가 내리고
여기서는 그 결과를 AnswerClaim / Limitation / Clarification으로 옮기기만 한다.
확정 = FactLookup이 참(ok+complete) AND selected.citation != None (verified Evidence).
"""
from __future__ import annotations

import re

from decimal import Decimal

from agent.planning import concept_display_name
from app.orchestrator.payload import (
    AnswerClaim, ClaimCitation, Clarification, Limitation, TraceEvent,
)
from .financial_limitations import financial_source_limitation

_AMBIGUOUS = {"ambiguous_account_path", "ambiguous_statement",
              "ambiguous_period_role", "ambiguous_unit"}


def _equivalent_account_path(candidates) -> str | None:
    """Return one deterministic path when path choice cannot change the answer.

    A statement can expose the same total once as a parent row and once as its
    only child (for example ``영업수익`` and ``영업수익 > 매출액``).  Asking the
    user to choose between those paths is not useful when every answer-bearing
    axis and the normalized value are identical.  This is intentionally
    narrower than generic ambiguity collapse: different statement/period
    roles or different values still require clarification.
    """

    rows = tuple(candidates or ())
    paths = {str(getattr(row, "account_path", "") or "") for row in rows}
    if len(rows) < 2 or len(paths) < 2 or "" in paths:
        return None
    # Equal numbers in separate filings are not an equivalence proof.  A
    # repeated parent/child label is collapsible only within one exact source
    # table; restated comparisons are resolved by the explicit peer-table
    # authority below instead.
    coordinates = {_table_coordinate(row) for row in rows}
    if len(coordinates) != 1 or None in coordinates:
        return None
    semantic_axes = {
        (
            getattr(row, "scope", None), getattr(row, "statement", None),
            getattr(row, "period_start", None), getattr(row, "period_end", None),
            getattr(row, "period_type", None), getattr(row, "cumulative", None),
        )
        for row in rows
    }
    if len(semantic_axes) != 1:
        return None
    values = set()
    for row in rows:
        money = getattr(row, "money", None)
        if money is None:
            return None
        normalized = money.in_won_decimal()
        if normalized is not None:
            values.add(("won", normalized))
            continue
        raw = money.decimal()
        unit = (getattr(money, "unit", None) or "").strip()
        if raw is None or not unit:
            return None
        values.add((unit, raw))
    if len(values) != 1:
        return None
    # Prefer the total/root label over a repeated leaf, then stay deterministic.
    return min(paths, key=lambda path: (path.count(">"), len(path), path))


def _table_coordinate(row) -> tuple[str, str, str, str, str] | None:
    """Return a source-table identity, never a value-based equivalence key."""

    values = (
        getattr(row, "doc_id", None), getattr(row, "source_file_id", None),
        getattr(row, "table_locator", None), getattr(row, "scope", None),
        getattr(row, "statement", None),
    )
    return tuple(str(value) for value in values) if all(values) else None


def _comparison_table_account_path(candidates, anchors) -> str | None:
    """Choose a path only when an adjacent comparison fact proves its table.

    A restated annual filing often contains both the current and preceding
    annual columns.  The preceding value can coexist with an older filing
    whose account label changed (``매출`` -> ``매출액``).  The common exact
    document/source-file/table coordinate is authority for that comparison;
    equal numeric values are deliberately not inspected here.
    """

    anchor_coordinates = {
        coordinate for row in (anchors or ())
        if (coordinate := _table_coordinate(row)) is not None
    }
    if not anchor_coordinates:
        return None
    matching = [
        row for row in (candidates or ())
        if _table_coordinate(row) in anchor_coordinates
    ]
    paths = {str(getattr(row, "account_path", "") or "") for row in matching}
    return next(iter(paths)) if len(paths) == 1 and "" not in paths else None


def _same_comparison_axis(left, right) -> bool:
    """A peer may anchor only the same issuer/concept statement family."""

    def concept(spec):
        value = getattr(spec, "concept", None)
        return str(value.value if hasattr(value, "value") else value)

    return (
        getattr(left, "corp_code", None) == getattr(right, "corp_code", None)
        and concept(left) == concept(right)
        and getattr(left, "scope", None) == getattr(right, "scope", None)
        and getattr(left, "statement", None) == getattr(right, "statement", None)
        and getattr(left, "period_type", None) == getattr(right, "period_type", None)
        and getattr(left, "cumulative", None) == getattr(right, "cumulative", None)
        and getattr(left, "unit", None) == getattr(right, "unit", None)
        and getattr(left, "period_end", None) != getattr(right, "period_end", None)
    )


def _same_fact_semantics_and_value(left, right) -> bool:
    """Whether two rows independently prove the same requested fact.

    This deliberately requires every answer-bearing axis and the normalized
    value to match.  It is used only to retain additional original/correction
    citations for a multi-fact comparison; a different value is never hidden
    behind the currently selected restated row.
    """

    axes = (
        "scope", "statement", "account_path", "period_start", "period_end",
        "period_type", "cumulative",
    )
    if any(getattr(left, axis, None) != getattr(right, axis, None) for axis in axes):
        return False
    left_won = left.money.in_won_decimal()
    right_won = right.money.in_won_decimal()
    if left_won is not None or right_won is not None:
        return left_won is not None and left_won == right_won
    return (
        left.money.decimal() is not None
        and left.money.decimal() == right.money.decimal()
        and (left.money.unit or "").strip() == (right.money.unit or "").strip()
    )


class FactOutcome:
    """FactSpec 하나의 조회 결과 — claim(확정) 또는 limitation/clarification."""

    def __init__(self, output_id: str):
        self.output_id = output_id
        self.claim: AnswerClaim | None = None
        self.limitation: Limitation | None = None
        self.clarification: Clarification | None = None
        self.decimal_value: Decimal | None = None
        self.unit: str | None = None
        self.won: Decimal | None = None
        self.rcept_no: str | None = None
        self.corp_name: str | None = None


def _rcept_from_doc(doc_id: str) -> str | None:
    m = re.search(r"(\d{14})", doc_id or "")
    return m.group(1) if m else None


def _readable_source(rm, fact) -> "dict[str, str]":
    """사람이 원문에서 찾아갈 수 있는 자리. 없으면 빈 dict.

    이슈 #94 30 — 접수번호는 어느 **문서**인지만 말한다. 그 값이 보고서
    어디에 있는지는 정본이 이미 알고 있다.

    ```
    documents.corp_name +
      documents.report_nm   삼성전자 사업보고서 (2025.12)
    facts.statement_title   2-1. 연결 재무상태표
    facts.account_path      부채 > Ⅰ.유동부채
    ```

    `locator`(`TABLE[1]/TBODY[0]/TR[25]`)는 전처리 좌표라 사용자에게 줄 수
    없다 — 그것은 지금처럼 typed payload 와 trace 에만 남는다.

    **없으면 넣지 않는다.** 옛 정본 산출물에는 열 자체가 없고, 그때는 종전대로
    접수번호만 나가야 한다.
    """

    out: dict[str, str] = {}
    try:
        report = rm.document_label(fact.doc_id)
    except Exception:
        report = None
    if report:
        out["report_name"] = report
    parts = [str(value).strip() for value in
             (getattr(fact, "statement_title", None),
              getattr(fact, "account_path", None))
             if value and str(value).strip()]
    if parts:
        out["source_path"] = " > ".join(parts)
    return out


CONCEPT_KO = {"revenue": "매출액", "operating_income": "영업이익", "net_income": "당기순이익",
              "total_assets": "자산총계", "capex_ppe": "유형자산 취득 현금유출액", "gross_profit": "매출총이익",
              "net_income_parent": "지배주주순이익", "total_liabilities": "부채총계", "total_equity": "자본총계",
              "equity_parent": "지배주주지분", "operating_cash_flow": "영업활동현금흐름", "capital_stock": "자본금",
              "current_assets": "유동자산", "current_liabilities": "유동부채"}
SCOPE_KO = {"CFS": "연결", "SFS": "별도"}


def _period_ko(spec) -> str:
    y = spec.period_end.year
    if spec.period_type == "annual":
        return f"{y}년"
    if spec.period_type == "half":
        return f"{y}년 상반기 누적" if spec.cumulative else f"{y}년 2분기"
    if spec.period_type == "quarter":
        m = spec.period_end.month
        q = {3: "1분기", 6: "2분기", 9: "3분기", 12: "4분기"}.get(m, f"{m}월 분기")
        return f"{y}년 {q}" + (" 누적" if spec.cumulative and m > 3 else "")
    return f"{spec.period_end.isoformat()} 기준"


from app.composer.definitions import capex_ppe_definition


_LOSS_CONCEPT_KO = {
    "operating_income": "영업손실",
    "net_income": "당기순손실",
    "gross_profit": "매출총손실",
    "net_income_parent": "지배주주순손실",
}


def _concept_label(concept: str, value: Decimal | None, *, account_raw: str = "") -> str:
    """Preserve the source revenue label and name loss facts as losses."""

    # #233: canonical revenue is a lookup concept, not permission to rename
    # a verified source row '영업수익' to '매출액' in the public answer.
    if concept == "revenue" and re.match(r"^\s*영업수익(?:\s|\(|$)", account_raw):
        return "영업수익"
    if value is not None and value < 0:
        return _LOSS_CONCEPT_KO.get(
            concept, CONCEPT_KO.get(concept, concept_display_name(concept)))
    # ``CONCEPT_KO`` predates the complete, source-validated display registry
    # in ``agent.planning``.  Falling back to the internal identifier leaked
    # names such as ``inventories`` and ``retained_earnings`` into otherwise
    # Korean answers.  Reuse the shared registry so every supported concept is
    # public-safe while preserving the few established wording overrides above.
    return CONCEPT_KO.get(concept, concept_display_name(concept))


class FinancialTool:
    def __init__(self, rm):
        self.rm = rm
        self.definitions_used: list[str] = []

    def _comparison_table_anchors(self, task, spec):
        """Find independently selected peer facts for a bounded task-local comparison."""

        anchors = []
        for peer in task.facts:
            if peer is spec or not _same_comparison_axis(spec, peer):
                continue
            kwargs = dict(as_of=task.as_of, scope=peer.scope, view=task.view)
            if peer.cumulative is not None:
                kwargs["cumulative"] = peer.cumulative
            if peer.statement is not None:
                kwargs["statement"] = peer.statement
            if peer.account_path is not None:
                kwargs["account_path"] = peer.account_path
            if peer.period_start is not None:
                kwargs["period_start"] = peer.period_start.isoformat()
            if peer.unit is not None:
                kwargs["unit"] = peer.unit
            concept = str(peer.concept.value if hasattr(peer.concept, "value") else peer.concept)
            lookup = self.rm.lookup(
                peer.corp_code, concept, peer.period_end.isoformat(), **kwargs)
            if lookup.status == "not_found" and kwargs.get("statement") in ("IS", "CI"):
                alt = "CI" if kwargs["statement"] == "IS" else "IS"
                lookup = self.rm.lookup(
                    peer.corp_code, concept, peer.period_end.isoformat(),
                    **{**kwargs, "statement": alt})
            if lookup.status == "ok" and lookup.selected is not None:
                anchors.append(lookup.selected)
        return anchors

    def _comparison_citations(self, task, lookup, selected, selected_evidence):
        """Keep verified, value-identical filing citations for comparisons.

        ``restated`` still determines the value authority and therefore stays
        first.  Older originals/corrections are supplementary provenance only.
        Single-fact retrieval remains concise and unchanged.
        """

        selected_rcept = _rcept_from_doc(selected.doc_id) or selected.doc_id
        citations = [ClaimCitation(
            doc_id=selected.doc_id,
            rcept_no=selected_rcept,
            evidence_id=selected.evidence_id,
            locator=selected.locator,
            excerpt_prompt_safe=selected_evidence.excerpt_safe,
            **_readable_source(self.rm, selected),
        )]
        if len(task.facts) < 2:
            return citations

        seen = {(selected.doc_id, selected.evidence_id, selected.locator)}
        for candidate in sorted(
                lookup.candidates, key=lambda item: (item.rcept_dt, item.doc_id)):
            key = (candidate.doc_id, candidate.evidence_id, candidate.locator)
            if key in seen or candidate.evidence_id is None:
                continue
            if not _same_fact_semantics_and_value(selected, candidate):
                continue
            row = self.rm.verify_fact_evidence(candidate)
            if row.citation is None:
                continue
            evidence = self.rm.get_evidence(row.evidence_id)
            if evidence is None:
                continue
            seen.add(key)
            citations.append(ClaimCitation(
                doc_id=row.doc_id,
                rcept_no=_rcept_from_doc(row.doc_id) or row.doc_id,
                evidence_id=row.evidence_id,
                locator=row.locator,
                excerpt_prompt_safe=evidence.excerpt_safe,
                **_readable_source(self.rm, row),
            ))
            if len(citations) >= 4:
                break
        return citations

    def run_task(
            self, task, *, trace: list[TraceEvent],
            view_labels: dict[str, str] | None = None,
            ) -> list[FactOutcome]:
        outcomes: list[FactOutcome] = []
        display_labels = view_labels or {}
        for spec in task.facts:
            o = FactOutcome(spec.output_id)
            kwargs = dict(as_of=task.as_of, scope=spec.scope, view=task.view)
            # 축을 알려준 경우에만 좁힌다 (UNSET 유지 = read.py가 판단)
            if spec.cumulative is not None:
                kwargs["cumulative"] = spec.cumulative
            if spec.statement is not None:
                kwargs["statement"] = spec.statement
            if spec.account_path is not None:
                kwargs["account_path"] = spec.account_path
            if spec.period_start is not None:
                kwargs["period_start"] = spec.period_start.isoformat()
            if spec.unit is not None:
                kwargs["unit"] = spec.unit

            concept = str(spec.concept.value if hasattr(spec.concept, "value") else spec.concept)
            if concept == "capex_ppe":
                definition = capex_ppe_definition(spec.scope)
                if definition not in self.definitions_used:
                    self.definitions_used.append(definition)
            lk = self.rm.lookup(spec.corp_code, concept, spec.period_end.isoformat(), **kwargs)
            # 같은 손익 계정이 IS 대신 CI에 있는 경우를 먼저 완화한다. 완화
            # 결과가 account_path 모호성이면 아래의 동일 표 좌표 규칙이 이어서
            # 처리해야 하므로 이 단계가 모호성 처리보다 앞에 있어야 한다.
            if lk.status == "not_found" and kwargs.get("statement") in ("IS", "CI"):
                alt = "CI" if kwargs["statement"] == "IS" else "IS"
                lk2 = self.rm.lookup(spec.corp_code, concept, spec.period_end.isoformat(),
                                     **{**kwargs, "statement": alt})
                if lk2.status != "not_found":
                    trace.append(TraceEvent(seq=len(trace)+1, stage="tool",
                                            summary=f"statement {kwargs['statement']}→{alt} 완화 재조회 (단일 포괄손익계산서 회사)",
                                            detail={"output_id": spec.output_id}))
                    # 대체 statement에서 찾은 후속 account_path는 동일
                    # statement 축에서 다시 조회해야 한다. 선택된 축을
                    # kwargs에 반영하지 않으면 뒤에서 원래 IS 조건으로
                    # 돌아가 동일 표 좌표 확정이 불가능해진다.
                    kwargs = {**kwargs, "statement": alt}
                    lk = lk2
            if lk.status == "ambiguous_account_path":
                anchors = self._comparison_table_anchors(task, spec)
                anchored_path = _comparison_table_account_path(lk.candidates, anchors)
                if anchored_path is not None:
                    narrowed = self.rm.lookup(
                        spec.corp_code, concept, spec.period_end.isoformat(),
                        **{**kwargs, "account_path": anchored_path})
                    if (narrowed.status == "ok" and narrowed.selected is not None
                            and _table_coordinate(narrowed.selected)
                            in {_table_coordinate(row) for row in anchors}):
                        trace.append(TraceEvent(
                            seq=len(trace) + 1, stage="tool",
                            summary="비교 기준 동일 공시 표 좌표로 account_path 확정",
                            detail={"output_id": spec.output_id,
                                    "account_path": anchored_path,
                                    "candidate_count": len(lk.candidates)}))
                        lk = narrowed
            if lk.status == "ambiguous_account_path":
                equivalent_path = _equivalent_account_path(lk.candidates)
                if equivalent_path is not None:
                    narrowed = self.rm.lookup(
                        spec.corp_code, concept, spec.period_end.isoformat(),
                        **{**kwargs, "account_path": equivalent_path})
                    if narrowed.status == "ok":
                        trace.append(TraceEvent(
                            seq=len(trace) + 1, stage="tool",
                            summary="동일 원문 표 내 account_path 중복을 결과 영향 없이 축약",
                            detail={"output_id": spec.output_id,
                                    "account_path": equivalent_path,
                                    "candidate_count": len(lk.candidates)}))
                        lk = narrowed
            trace.append(TraceEvent(
                seq=len(trace) + 1, stage="tool",
                summary=f"lookup_fact {spec.corp_name} {spec.concept} {spec.period_end} {spec.scope} → {lk.status}/{lk.coverage_status}",
                detail={"output_id": spec.output_id, "status": lk.status,
                        "coverage": lk.coverage_status, "candidates": len(lk.candidates),
                        "unread": list(lk.unread_documents), "next": list(lk.next_discriminator)}))

            if lk.status in _AMBIGUOUS:
                # canonical 모호성 → 조회 중 역질문 (next_discriminator 보존)
                opts = sorted({str(getattr(c, lk.next_discriminator[0], "")) for c in lk.candidates}) \
                    if lk.next_discriminator else []
                o.clarification = Clarification(
                    clarification_id=f"clarify-{spec.output_id}-{lk.next_discriminator[0] if lk.next_discriminator else 'axis'}",
                    question=f"{spec.corp_name} {spec.concept} 조회에 {lk.next_discriminator} 지정이 필요합니다.",
                    targets=list(lk.next_discriminator),
                    options={lk.next_discriminator[0]: opts} if lk.next_discriminator and opts else {})
                outcomes.append(o)
                continue

            if lk.status == "not_found":
                o.limitation = Limitation(
                    code=("not_found_financial_account" if lk.coverage_status == "complete"
                          else "not_found"),
                    detail=f"{spec.corp_name} {spec.concept} {spec.period_end} {spec.scope}: 지원 범위에서 확인되지 않음")
                if spec.concept in {"interest_expense", "interest_paid"}:
                    o.limitation = Limitation(
                        code=f"not_found_{spec.concept}", detail=o.limitation.detail)
                source_limitation = financial_source_limitation(
                    corp_code=spec.corp_code, concept=concept,
                    period_end=spec.period_end.isoformat(), scope=spec.scope)
                if source_limitation is not None:
                    o.limitation = Limitation(
                        code=f"not_found_{source_limitation.code}",
                        detail=source_limitation.detail)
                outcomes.append(o); continue
            if lk.status == "extract_unsupported":
                source_limitation = financial_source_limitation(
                    corp_code=spec.corp_code, concept=concept,
                    period_end=spec.period_end.isoformat(), scope=spec.scope)
                o.limitation = Limitation(
                    code=(source_limitation.code if source_limitation is not None
                          else "extract_unsupported"),
                    detail=(source_limitation.detail if source_limitation is not None
                            else "관련 원문을 구조적으로 읽지 못해 존재 여부 판정 불가"),
                    affected_doc_ids=list(lk.unread_documents))
                outcomes.append(o); continue
            if lk.status == "same_document_conflict":
                o.limitation = Limitation(code="same_document_conflict",
                                          detail="같은 문서 안에 같은 계정 값이 충돌")
                outcomes.append(o); continue
            if lk.status != "ok" or lk.selected is None:
                o.limitation = Limitation(code=f"lookup_{lk.status}", detail=f"lookup status {lk.status}")
                outcomes.append(o); continue

            row = lk.selected
            if lk.coverage_status == "evidence_unavailable" or row.citation is None:
                o.limitation = Limitation(code="evidence_unavailable",
                                          detail="값 후보는 있으나 Evidence 검증 실패 — 확정 불가",
                                          affected_doc_ids=[row.doc_id])
                outcomes.append(o); continue

            # Evidence 재검증 (확정 인용 직전)
            ev = self.rm.get_evidence(row.evidence_id)
            trace.append(TraceEvent(seq=len(trace) + 1, stage="evidence",
                                    summary=f"get_evidence {row.evidence_id[:12]}… → {'ok' if ev else 'missing'}",
                                    detail={"output_id": spec.output_id, "doc_id": row.doc_id,
                                            "locator": row.locator}))
            if ev is None:
                o.limitation = Limitation(code="evidence_unavailable", detail="Evidence 재조회 실패",
                                          affected_doc_ids=[row.doc_id])
                outcomes.append(o); continue

            money = row.money
            dec = money.decimal()
            won = money.in_won_decimal()
            rcept = _rcept_from_doc(row.doc_id) or row.doc_id
            o.decimal_value, o.unit, o.won, o.rcept_no = dec, money.unit, won, rcept
            o.corp_name = spec.corp_name
            o.claim = AnswerClaim(
                output_id=spec.output_id,
                label=((f"{display_labels[spec.output_id]} "
                        if spec.output_id in display_labels else "")
                       + f"{spec.corp_name} {_period_ko(spec)} "
                       f"{SCOPE_KO.get(spec.scope, spec.scope)} "
                       f"{_concept_label(concept, dec, account_raw=getattr(row, 'account_raw', ''))}"),
                state=spec.corp_name,   # 비교(argmax) 결과의 주체 식별용
                value_text=money.text or (str(dec) if dec is not None else None),
                raw_unit=money.unit,
                canonical_value=str(won) if won is not None else None,
                canonical_unit="원" if won is not None else None,
                citations=self._comparison_citations(task, lk, row, ev),
            )
            if lk.coverage_status == "partial_unread":
                o.limitation = Limitation(code="partial_unread",
                                          detail="같은 기간을 담은 일부 문서를 읽지 못함 — 최신 재작성본 미확인 가능",
                                          affected_doc_ids=list(lk.unread_documents))
            outcomes.append(o)
        return outcomes
