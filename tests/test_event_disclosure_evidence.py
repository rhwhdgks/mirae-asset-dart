"""Event conclusions preserve filing roles and cite the actual source cells."""

from types import SimpleNamespace as NS

import pytest

from app.composer.template import (
    TemplateComposer, _contract_event_listing_table, _event_qualification_notes,
)
from app.composer.hcx import HcxComposer
from app.orchestrator.payload import AnswerClaim, AnswerPayload, ClaimCitation, Limitation
from app.tools.events import EventCandidate, EventTool

CORP = "00000001"
ORIGINAL = "20250110900001"
TERMINATION = "20250422900002"
OTHER = "20250213900003"


def _meta(receipt, *, date=None, correction=False, termination=False, corp=CORP):
    return NS(
        doc_id="exchange_" + receipt, rcept_no=receipt, corp_code=corp,
        rcept_dt=date or receipt[:8], is_correction=correction,
        form="단일판매ㆍ공급계약해지" if termination else "단일판매ㆍ공급계약체결",
        report_nm="[기재정정]공급계약" if correction else "공급계약",
    )


def _field(receipt, label, value, *, verified=True):
    return NS(
        doc_id="exchange_" + receipt, rcept_no=receipt, label=label,
        value=value, value_prompt_safe=value, value_status="reported",
        evidence_status="verified" if verified else "unverified",
        evidence_id=receipt + ":" + label, locator="TABLE[0]/" + label,
        path=label, occurrence=0, order=0,
    )


def _observation(receipt, *, termination=False, correction=False):
    return NS(rcept_no=receipt, observed_at=receipt[:8],
              is_termination=termination, is_correction=correction)


def _timeline(root, observations, *, ambiguous=False):
    return NS(
        event_key="event:" + root, root_rcept_no=root, corp_code=CORP,
        identity_status="ambiguous" if ambiguous else "resolved",
        observations=observations,
        state=NS(status=("terminated" if any(o.is_termination for o in observations)
                         else "active"), last_rcept_no=observations[-1].rcept_no),
    )


class _Fields:
    def __init__(self, rows):
        self.values = rows

    def rows(self, _corp, *, rcept_no, as_of):
        return [row for row in self.values if row.rcept_no == rcept_no
                and row.rcept_no[:8] <= as_of]

    def lookup(self, corp, receipt, label, *, as_of):
        matches = [row for row in self.rows(corp, rcept_no=receipt, as_of=as_of)
                   if label in row.label]
        return ("ok", matches[0], matches) if len(matches) == 1 else (
            "ambiguous" if matches else "not_found", None, matches)

    def verified(self, row):
        return row


class _Reader:
    def __init__(self, metadata, timelines=()):
        self._documents = {row.doc_id: row for row in metadata}
        self._doc_id_by_rcept = {row.rcept_no: row.doc_id for row in metadata}
        self.timelines = {row.root_rcept_no: row for row in timelines}

    def _load_docs(self):
        pass

    def event_timeline(self, *, as_of, rcept_no=None, event_key=None,
                       verify_evidence=True):
        return self.timelines.get(rcept_no or event_key.removeprefix("event:"))

    def event_identity_candidate_roots(self, **_kwargs):
        return (ORIGINAL, OTHER)


def _task(slots):
    return NS(task_id="list", corp_code=CORP, timepoints=["20260619"],
              requested_slots=slots)


def test_contract_table_keeps_signing_date_separate_from_filing_metadata():
    # Deliberately disagree with the receipt prefix as well as the signing
    # date: only metadata is authoritative for the filing date.
    metadata = [_meta(ORIGINAL, date="20250111"), _meta(OTHER)]
    timelines = [_timeline(rc, [_observation(rc)]) for rc in (ORIGINAL, OTHER)]
    rows = [
        _field(rc, label, value) for rc, values in (
            (ORIGINAL, {"계약상대": "가상 A", "계약명": "설비 공급",
                        "계약(수주)일자": "2025-01-03"}),
            (OTHER, {"계약상대": "가상 B", "계약명": "시설 건설"}),
        ) for label, value in values.items()
    ]
    tool = EventTool(_Reader(metadata, timelines), _Fields(rows))
    claims, limitations, _, _ = tool._list(
        _task(["계약상대", "계약명"]),
        [EventCandidate(rc, rc[:8], "공급계약", None) for rc in (ORIGINAL, OTHER)],
        "가상회사", [],
    )
    answer = _contract_event_listing_table(
        NS(claims=claims, limitations=limitations),
        "가상회사의 계약을 계약상대·계약명·체결일별로 표로 정리해줘.")

    assert "체결일(접수일)" not in answer
    assert "| 계약 체결일 | 식별 공시 접수일 |" in answer
    assert "| 2025-01-03 | 2025-01-11 |" in answer
    assert "| 확인되지 않음 | 2025-02-13 |" in answer
    filing_claim = next(c for c in claims if c.output_id.endswith(ORIGINAL + ".공시접수일"))
    assert "접수일: 20250111" in filing_claim.citations[0].excerpt_prompt_safe


def test_termination_conclusion_and_count_cite_each_actual_termination_body():
    other_termination = "20250612900004"
    timelines = [
        _timeline(ORIGINAL, [_observation(ORIGINAL),
                             _observation(TERMINATION, termination=True)]),
        _timeline(OTHER, [_observation(OTHER),
                          _observation(other_termination, termination=True)]),
    ]
    metadata = [_meta(ORIGINAL), _meta(OTHER),
                _meta(TERMINATION, termination=True),
                _meta(other_termination, termination=True)]
    rows = [_field(rc, label, value) for rc in (TERMINATION, other_termination)
            for label, value in {
                "해지계약명": "가상 설비 공급", "해지일자": "2025-04-21",
                "해지 주요사유": "상대방의 사업 철회", "계약상대": "가상 구매사",
            }.items()]
    rows += [_field(rc, "계약상대", "가상 구매사") for rc in (ORIGINAL, OTHER)]
    tool = EventTool(_Reader(metadata, timelines), _Fields(rows))
    claims, _, _, _ = tool._list(
        _task([]), [EventCandidate(rc, rc[:8], "공급계약", "가상 구매사")
                    for rc in (ORIGINAL, OTHER)], "가상회사", [])

    for root, term in ((ORIGINAL, TERMINATION), (OTHER, other_termination)):
        conclusion = next(c for c in claims if c.output_id == "list." + root)
        cited = {c.evidence_id for c in conclusion.citations}
        assert term + ":해지 주요사유" in cited
        assert term + ":해지일자" in cited
        assert term + ":해지계약명" in cited
    count = next(c for c in claims if c.output_id.endswith(".terminated_count"))
    assert count.value_text == "2"
    assert {TERMINATION, other_termination} <= {c.rcept_no for c in count.citations}
    answer = TemplateComposer().compose(AnswerPayload(
        question_id="membership", handoff_id="offline", final_status="answer",
        claims=claims), question="가상회사의 계약 중 해지된 계약이 존재하는가?")
    assert answer.count("해지일: 2025-04-21") == 2
    assert answer.count("사유: 상대방의 사업 철회") == 2
    assert TERMINATION in answer and other_termination in answer


def test_unverified_termination_detail_is_not_promoted_to_citation():
    tool = EventTool(_Reader([_meta(TERMINATION, termination=True)]), _Fields([
        _field(TERMINATION, "해지 주요사유", "검증 안 된 사유", verified=False)]))
    cites = tool._termination_cites(CORP, TERMINATION, "20260619")
    assert len(cites) == 1
    assert cites[0].locator == "document"
    assert "검증 안 된 사유" not in cites[0].excerpt_prompt_safe


@pytest.mark.parametrize("metadata,as_of", [
    (_meta(ORIGINAL, corp="00000002"), "20260619"),
    (_meta(ORIGINAL), "20250109"),
])
def test_filing_metadata_is_issuer_and_cutoff_bound(metadata, as_of):
    tool = EventTool(_Reader([metadata]), _Fields([]))
    assert tool._metadata_cites(CORP, ORIGINAL, as_of) == []


def test_ambiguous_related_corrections_are_never_labelled_original_candidates():
    limitation = Limitation(
        code="ambiguous_event_origin", detail="계보 미확정",
        affected_doc_ids=[ORIGINAL, OTHER])
    payload = AnswerPayload(
        question_id="generic", handoff_id="offline", final_status="partial_answer",
        claims=[AnswerClaim(output_id="observed", label="확인된 공시", text="정정 공시")],
        limitations=[limitation])
    for notes in (_event_qualification_notes([limitation]),
                  TemplateComposer._public_notes(payload, question=None)):
        assert any(ORIGINAL in note and OTHER in note for note in notes)
        assert all("원공시 후보" not in note for note in notes)
        assert any("연결이 미확정" in note for note in notes)


def test_missing_original_conclusion_leads_an_identification_question_once():
    payload = AnswerPayload(
        question_id="root-query", handoff_id="offline", final_status="partial_answer",
        claims=[AnswerClaim(
            output_id="change", label="2025-03-10 정정 계약금액",
            text="1,000 → 800", operator="correction_diff",
        )], limitations=[
            Limitation(code="source_scope_prevents_complete_lineage", detail="범위 제한"),
            Limitation(code="source_scope_raw_absent", detail="원문 부재"),
        ])
    answer = TemplateComposer().compose(
        payload, question="가상 공급계약의 원계약을 특정할 수 있는지 정정 이력을 설명해줘.")
    assert answer.startswith("제공된 자료보다 앞선 원공시가 없어")
    assert answer.count("이전 변경 이력은 확인할 수 없습니다") == 1
    assert "원공시가 제공된 코퍼스보다 앞서 제출되어" not in answer


def test_negative_filing_observation_cannot_lose_its_corpus_boundary_to_wording():
    class _NoWording:
        cfg = NS(enabled=True)

        def generate(self, *_args, **_kwargs):
            raise AssertionError("bounded observation must not be freely reworded")

    payload = AnswerPayload(
        question_id="generic-observation", handoff_id="offline", final_status="answer",
        claims=[AnswerClaim(
            output_id="state@20260619", label="가상 계약의 해지·종료 공시 관측 여부",
            state="no_termination_observed",
            text=("제공된 공시 자료에서는 기준일까지 해지·종료 공시가 확인되지 "
                  "않았습니다. 이는 계약의 법적 유효성을 뜻하지 않습니다."),
            citations=[ClaimCitation(
                doc_id="exchange_" + ORIGINAL, rcept_no=ORIGINAL,
                locator="document", verification="source_roundtrip",
                excerpt_prompt_safe="가상 공급계약 공시")],
        )])
    answer = HcxComposer(client=_NoWording()).compose(
        payload, question="가상 공급계약에 해지 공시가 나왔어?")
    assert "제공된 공시 자료" in answer
    assert "법적 유효성을 뜻하지 않습니다" in answer


def test_identical_negative_findings_at_different_dates_are_both_displayed():
    claims = [AnswerClaim(
        output_id="event.termination_observed@" + day,
        label=f"가상 계약의 {day} 기준 해지·종료 공시 관측 여부",
        state="no_termination_observed",
        text="제공 자료에서 해지·종료 공시가 확인되지 않았습니다.",
        citations=[ClaimCitation(
            doc_id="exchange_" + receipt, rcept_no=receipt,
            locator="document", verification="source_roundtrip",
            excerpt_prompt_safe="가상 계약 공시")],
    ) for day, receipt in (("20241231", "20240110900001"),
                           ("20260619", "20251210900002"))]
    payload = AnswerPayload(
        question_id="two-observations", handoff_id="offline",
        final_status="answer", claims=claims)
    answer = TemplateComposer().compose(payload, question="두 날짜의 해지 공시 확인 여부는?")
    assert "| 2024-12-31 |" in answer
    assert "| 2026-06-19 |" in answer
    assert "20240110900001" in answer and "20251210900002" in answer
    assert "법적 유효성" in answer
