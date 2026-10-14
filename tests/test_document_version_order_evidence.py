"""Document version selection is edge-bound, never an intraday ID heuristic."""

from types import SimpleNamespace as NS

import pytest

from app.composer.template import (
    _correction_history_table, _document_versions_at_requested_dates,
)
from app.orchestrator.payload import AnswerClaim, ClaimCitation
from app.tools.events import DocumentTool
from src.canonical.lineage import LatestResolution

CORP = "00000001"
ROOT = "20250307900001"
FIRST = "20250310900009"
LAST = "20250310900002"  # smaller ID can be later in an explicit lineage


class _Fields:
    def iter_docs(self, *_args, **_kwargs):
        return iter(())


class _Reader:
    def __init__(self, resolutions, edges=()):
        self.resolutions = resolutions
        receipts = set(resolutions)
        for result in resolutions.values():
            receipts.update(result.members)
        self._documents = {
            "periodic_" + rc: NS(
                doc_id="periodic_" + rc, rcept_no=rc, corp_code=CORP,
                rcept_dt=rc[:8], report_nm="가상 보고서", form="사업보고서",
                is_correction=rc != ROOT)
            for rc in receipts}
        self._doc_id_by_rcept = {rc: "periodic_" + rc for rc in receipts}
        self._rcept_dt = {rc: rc[:8] for rc in receipts}
        self.edges = edges

    def _load_docs(self):
        pass

    def resolve_document_version(self, receipt, *, as_of):
        return self.resolutions[receipt]

    def relation_summaries(self, *, source_rcept_no, as_of):
        return [NS(relation_type=kind, resolution_status="resolved",
                   dst_rcept_no=older)
                for newer, older, kind in self.edges if newer == source_rcept_no]


def _task(operation, receipt=ROOT):
    return NS(task_id="versions", operation=operation, corp_code=CORP,
              as_of="20260619", selector=NS(rcept_no=receipt), event_selector=None)


def _run_latest(resolutions):
    tool = DocumentTool(_Reader(resolutions), _Fields())
    # A collection selector may contain multiple independent document seeds.
    tool._periodic_docs = lambda *_args: list(resolutions)
    task = _task("latest")
    task.selector = NS(rcept_no=None, doc_group="periodic", form="사업보고서",
                       is_correction=None, rcept_from=None, rcept_to=None)
    return tool.run_task(task, trace=[], corp_name="가상회사")


def test_independent_same_day_latest_candidates_are_not_arbitrarily_selected():
    claims, limitations, used, _ = _run_latest({
        FIRST: LatestResolution("ok", FIRST, (FIRST,), (FIRST,)),
        LAST: LatestResolution("ok", LAST, (LAST,), (LAST,)),
    })
    assert claims == [] and used == []
    assert [lim.code for lim in limitations] == ["ambiguous_document_version"]
    assert set(limitations[0].affected_doc_ids) == {FIRST, LAST}


def test_multiple_seeds_resolving_to_one_proved_leaf_remain_answerable():
    result = LatestResolution("ok", LAST, (LAST,), (ROOT, FIRST, LAST))
    claims, limitations, used, _ = _run_latest({FIRST: result, LAST: result})
    assert not limitations
    assert claims[0].state == LAST and used == [LAST]
    assert "접수일: 20250310" in claims[0].citations[0].excerpt_prompt_safe


@pytest.mark.parametrize("status", ["ambiguous", "invalid"])
def test_unresolved_history_does_not_claim_latest_or_effective_intervals(status):
    result = LatestResolution(status, None, (FIRST, LAST), (ROOT, FIRST, LAST))
    tool = DocumentTool(_Reader({ROOT: result}), _Fields())
    claims, limitations, _, _ = tool.run_task(
        _task("version_history"), trace=[], corp_name="가상회사")
    assert limitations
    assert claims and all(claim.label == "계보 미확정인 공시" for claim in claims)
    assert all(claim.state is None for claim in claims)
    assert not any("최신본" in (claim.text or "") for claim in claims)


def test_explicit_edges_order_same_day_versions_and_prove_both_interval_ends():
    result = LatestResolution("ok", LAST, (LAST,), tuple(sorted((ROOT, FIRST, LAST))))
    tool = DocumentTool(_Reader({ROOT: result}, [
        (FIRST, ROOT, "CORRECTS"), (LAST, ROOT, "CORRECTS"),
        (LAST, FIRST, "SUPERSEDES"),
    ]), _Fields())
    claims, limitations, _, _ = tool.run_task(
        _task("version_history"), trace=[], corp_name="가상회사")
    lineage = next(c for c in claims if c.label == "문서 정정 계보")
    assert lineage.text == f"정정 있음: {ROOT} → {FIRST} → {LAST}"
    first_interval = next(c for c in claims if c.label == "버전 유효 구간")
    assert {ROOT, FIRST} == {cite.rcept_no for cite in first_interval.citations}
    assert any(lim.code == "intraday_order_unavailable" for lim in limitations)
    assert not any("분" in (c.text or "") for c in claims)


def test_unique_latest_without_unique_middle_order_does_not_invent_chronology():
    final = "20250311900004"
    result = LatestResolution("ok", final, (final,), (ROOT, FIRST, LAST, final))
    tool = DocumentTool(_Reader({ROOT: result}, [
        (FIRST, ROOT, "CORRECTS"), (LAST, ROOT, "CORRECTS"),
        (final, FIRST, "SUPERSEDES"), (final, LAST, "SUPERSEDES"),
    ]), _Fields())
    claims, limitations, _, _ = tool.run_task(
        _task("version_history"), trace=[], corp_name="가상회사")
    lineage = next(c for c in claims if c.label == "문서 정정 계보")
    assert "→" not in lineage.text
    assert not any(c.label == "버전 유효 구간" for c in claims)
    assert next(c for c in claims if c.label == "논리적 최신본").state == final
    assert any(lim.code == "document_order_unavailable" for lim in limitations)


def test_contract_grouping_shows_verified_identity_without_renaming_it_original():
    def cite(receipt, text):
        return ClaimCitation(doc_id="exchange_" + receipt, rcept_no=receipt,
                             excerpt_prompt_safe=text, evidence_id=receipt + text)

    identity = AnswerClaim(
        output_id="corrections.contract-identity." + ROOT,
        label="계약 식별 공시 " + ROOT,
        text="계약명: 가상 도시 철도 공급; 계약상대: 가상 발주처",
        citations=[cite(FIRST, "가상 도시 철도 공급")])
    change = AnswerClaim(
        output_id="corrections.amount", operator="correction_diff",
        label=f"계약 식별 접수번호 {ROOT} · 2025-03-10 정정 계약금액",
        text="1,000 → 800", citations=[cite(LAST, "1,000 → 800")])
    table = _correction_history_table([identity, change])
    assert "가상 도시 철도 공급" in table and "가상 발주처" in table
    assert ROOT in table and FIRST in table and LAST in table
    assert "원공시" not in table


def _dated_payload():
    old = "20240813900001"
    new = "20240926900002"
    def citation(rc):
        return ClaimCitation(doc_id="periodic_" + rc, rcept_no=rc,
                             locator="document", verification="source_roundtrip",
                             excerpt_prompt_safe="가상 보고서 메타데이터")
    return NS(limitations=[], claims=[
        AnswerClaim(output_id="old", label="버전 유효 구간", state=old,
                    text=f"{old} 버전은 20240813부터 다음 정정 접수일 20240926 전까지 유효",
                    citations=[citation(old), citation(new)]),
        AnswerClaim(output_id="new", label="버전 유효 구간", state=new,
                    text=f"{new} 버전은 20240926부터 기준시점 20260619까지 논리적 최신본",
                    citations=[citation(new)]),
    ])


def test_parallel_query_dates_select_their_own_proved_effective_versions():
    answer = _document_versions_at_requested_dates(
        _dated_payload(),
        "가상회사의 2024년 반기보고서는 2024년 9월 25일과 26일 기준으로 어떤 버전이야?")
    assert "| 2024-09-25 | 20240813900001 |" in answer
    assert "| 2024-09-26 | 20240926900002 |" in answer
    assert answer.index("2024-09-25") < answer.index("2024-09-26")


@pytest.mark.parametrize("question", [
    "가상회사의 보고서는 2024년 8월 12일과 13일 기준으로 어떤 버전이야?",
    "가상회사의 보고서는 2026년 6월 19일과 20일 기준으로 어떤 버전이야?",
    "가상회사의 원본과 2024년 9월 26일 정정본의 변경 이력은?",
])
def test_version_projection_does_not_invent_unproved_dates_or_query_roles(question):
    assert _document_versions_at_requested_dates(_dated_payload(), question) is None


def test_version_projection_requires_unique_intervals_and_own_document_evidence():
    question = "가상회사의 보고서는 2024년 9월 25일과 26일 기준으로 어떤 버전이야?"
    payload = _dated_payload()
    payload.claims.append(payload.claims[0])
    assert _document_versions_at_requested_dates(payload, question) is None
    payload = _dated_payload()
    payload.claims[0].citations = [payload.claims[0].citations[1]]
    assert _document_versions_at_requested_dates(payload, question) is None
