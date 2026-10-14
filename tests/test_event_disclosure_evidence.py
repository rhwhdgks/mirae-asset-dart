"""Event conclusions preserve filing roles and cite the actual source cells."""

from types import SimpleNamespace as NS

import pytest

from app.composer.template import (
    TemplateComposer, _contract_event_listing_table, _event_qualification_notes,
)
from app.orchestrator.payload import AnswerClaim, AnswerPayload, Limitation
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
