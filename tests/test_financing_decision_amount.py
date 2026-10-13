"""Keep typed financing categories, decision amounts, and payment claims apart."""

from decimal import Decimal
from types import SimpleNamespace

from agent.contracts import EventSelector, ResolvedDisclosureTask
from app.composer.template import TemplateComposer
from app.orchestrator.payload import AnswerPayload, ClaimCitation
from app.tools.events import DisclosureTool, EventCandidate

_RECEIPT = "20250611000051"
_FORM = "전환사채권발행결정"


class _ReadModel:
    @staticmethod
    def resolve_document_version(rcept_no, *, as_of):
        return SimpleNamespace(
            status="ok", selected=rcept_no, members=(rcept_no,))


class _Money:
    def __init__(self, value):
        self.value = value
        self.text = f"{value:,}"
        self.unit = "원"

    def in_won_decimal(self):
        return Decimal(self.value)


class _CashflowReadModel(_ReadModel):
    def __init__(self, rows):
        self._facts = rows

    def facts(self, corp_code, **kwargs):
        return self._facts

    @staticmethod
    def verify_fact_evidence(row):
        return row

    @staticmethod
    def get_evidence(evidence_id):
        return SimpleNamespace(excerpt_safe="19,650,000,000")

    @staticmethod
    def document_label(doc_id):
        return "디앤디파마텍 사업보고서 (2025.12)"


class _FieldIndex:
    def __init__(self, rows):
        self._rows = rows

    def rows(self, corp_code, *, as_of, rcept_no):
        return self._rows if rcept_no == _RECEIPT else []

    @staticmethod
    def verified(row):
        return row

    @staticmethod
    def iter_docs(corp_code, *, as_of, doc_group):
        return iter(())


class _FundingDisclosureTool(DisclosureTool):
    def find_candidates(self, corp_code, selector, *, as_of):
        return [EventCandidate(
            rcept_no=_RECEIPT,
            rcept_dt="20250611",
            form=_FORM,
            counterparty=None,
        )]

    @staticmethod
    def find_unknown_counterparty_candidates(*args, **kwargs):
        return []

    @staticmethod
    def field_value(corp_code, rcept_no, slot, *, as_of, trace):
        # This generic leaf is deliberately ambiguous. For explicit typed
        # categories the canonical filing form, not this leaf, is authoritative.
        return None, "ambiguous" if slot == "조달유형" else "not_found"

    @staticmethod
    def _doc_cites(corp_code, rcept_no, as_of):
        return [ClaimCitation(
            doc_id=f"major_{rcept_no}",
            rcept_no=rcept_no,
            evidence_id=f"doc-{rcept_no}",
            locator="TABLE[1]",
        )]


def _row(path: str, value: str, order: int):
    return SimpleNamespace(
        doc_id=f"major_{_RECEIPT}",
        rcept_no=_RECEIPT,
        evidence_id=f"field-{order}",
        locator=f"TABLE[1]/TR[{order}]",
        value_prompt_safe=value,
        value=value,
        value_status="literal",
        evidence_status="verified",
        form=_FORM,
        report_nm=f"주요사항보고서({_FORM})",
        rcept_dt="20250611",
        is_correction=False,
        path=path,
        label=path.split(">")[-1].strip(),
        order=order,
    )


def test_typed_funding_list_shows_decision_amount_without_claiming_payment():
    rows = [
        _row("2. 사채의 권면(전자등록)총액 (원)", "34,326,120,000", 1),
        _row("3. 자금조달의 목적 > 운영자금 (원)", "19,650,000,000", 2),
        _row(
            "3. 자금조달의 목적 > 타법인 증권 취득자금 (원)",
            "14,676,120,000",
            3,
        ),
    ]
    task = ResolvedDisclosureTask(
        task_id="P9-018",
        operation="list",
        corp_code="12345678",
        corp_name="디앤디파마텍",
        as_of="20260619",
        event_selector=EventSelector(
            event_type="주요사항보고",
            event_from="20250101",
            event_to="20251231",
            keywords=[
                "유상증자", "전환사채", "신주인수권부사채", "교환사채",
            ],
        ),
        requested_slots=["조달유형", "자금조달"],
    )
    tool = _FundingDisclosureTool(_ReadModel(), _FieldIndex(rows))
    claims, limitations, _used, clarification = tool.run_task(
        task, trace=[], corp_name="디앤디파마텍")
    payload = AnswerPayload(
        question_id="P9-018",
        handoff_id="offline-test",
        final_status="answer",
        claims=claims,
        limitations=limitations,
    )

    answer = TemplateComposer().compose(
        payload,
        question=("디앤디파마텍이 2025년에 공시한 자금조달 내역을 "
                  "유형별(유상증자·전환사채·신주인수권부사채·교환사채)로 정리해줘"),
    )

    assert clarification is None
    assert not limitations
    assert "전환사채권발행결정 1건" in answer
    assert "전환사채권발행결정 발행결정액은 343억 2,612만원입니다" in answer
    assert "유상증자·BW(신주인수권부사채)·EB(교환사채)" in answer
    assert "결정 공시만으로 실제 납입 완료나 돈을 조달한 사실을 확정할 수 없습니다" in answer
    assert "정정 계보와 최신 유효본" not in answer
    assert "운영자금" not in answer
    assert "타법인 증권 취득자금" not in answer
    assert "총 조달금액" not in answer


def _cashflow_fact():
    return SimpleNamespace(
        doc_id="periodic_20260320001088",
        period_start="2025-01-01",
        period_end="2025-12-31",
        period_type="annual",
        account_path="재무활동현금흐름 > 전환사채의 발행",
        statement_title="2-4. 연결 현금흐름표",
        value_status="literal",
        evidence_status="verified",
        evidence_id="cashflow-fact",
        locator="TABLE-GROUP[15]/TABLE[1]/TBODY[0]/TR[14]/TE[1]",
        money=_Money(19_650_000_000),
    )


def _compose_with_read_model(read_model):
    rows = [
        _row("2. 사채의 권면(전자등록)총액 (원)", "34,326,120,000", 1),
        _row("3. 자금조달의 목적 > 운영자금 (원)", "19,650,000,000", 2),
        _row(
            "3. 자금조달의 목적 > 타법인 증권 취득자금 (원)",
            "14,676,120,000",
            3,
        ),
    ]
    task = ResolvedDisclosureTask(
        task_id="P9-018",
        operation="list",
        corp_code="12345678",
        corp_name="디앤디파마텍",
        as_of="20260619",
        event_selector=EventSelector(
            event_type="주요사항보고",
            event_from="20250101",
            event_to="20251231",
            keywords=["유상증자", "전환사채", "신주인수권부사채", "교환사채"],
        ),
        requested_slots=["조달유형", "자금조달"],
    )
    tool = _FundingDisclosureTool(read_model, _FieldIndex(rows))
    claims, limitations, _used, clarification = tool.run_task(
        task, trace=[], corp_name="디앤디파마텍")
    assert clarification is None
    assert not limitations
    payload = AnswerPayload(
        question_id="P9-018",
        handoff_id="offline-test",
        final_status="answer",
        claims=claims,
        limitations=limitations,
    )
    return TemplateComposer().compose(
        payload,
        question=("디앤디파마텍이 2025년에 공시한 자금조달 내역을 "
                  "유형별(유상증자·전환사채·신주인수권부사채·교환사채)로 정리해줘"),
    )


def test_full_year_financial_statement_cashflow_is_separate_from_face_amount():
    answer = _compose_with_read_model(
        _CashflowReadModel([_cashflow_fact()]))

    assert "전환사채권발행결정 발행결정액은 343억 2,612만원입니다" in answer
    assert "후속 재무제표의 실제 현금 유입" in answer
    assert "2025년 연결 현금흐름표 전환사채 발행 현금 유입은 196억 5,000만원입니다" in answer
    assert "개별 발행결정의 납입 상태나 액면총액 전액의 납입을 직접 확인" in answer
    assert "사업보고서 (2025.12)" in answer
    assert "2-4. 연결 현금흐름표 > 재무활동현금흐름 > 전환사채의 발행" in answer


def test_cashflow_duplicate_candidates_fail_closed():
    answer = _compose_with_read_model(_CashflowReadModel([
        _cashflow_fact(), _cashflow_fact(),
    ]))

    assert "후속 재무제표의 실제 현금 유입" not in answer
    assert "결정 공시만으로 실제 납입 완료나 돈을 조달한 사실을 확정할 수 없습니다" in answer
