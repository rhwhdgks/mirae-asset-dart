"""A document title/date list cites the metadata that supplies both values."""

from types import SimpleNamespace

from app.composer.template import _investment_judgment_title_date_list
from app.orchestrator.payload import ClaimCitation
from app.tools.events import DocumentTool

_FORM = "투자판단관련주요경영사항"
_RECEIPT = "20250317900376"
_TITLE = f"{_FORM} (가상 기술 계약 체결)"


def _task(form=_FORM):
    selector = SimpleNamespace(
        rcept_no=None, doc_group=None, form=form, is_correction=None,
        rcept_from="20250101", rcept_to="20251231")
    return SimpleNamespace(
        task_id="title-date", operation="find", corp_code="00000001",
        as_of="20260619", selector=selector, event_selector=None)


def _row(*, title=_TITLE):
    return SimpleNamespace(
        doc_id="exchange_" + _RECEIPT, rcept_no=_RECEIPT,
        rcept_dt="20250317", report_nm=title, form=_FORM,
        is_correction=False)


def _metadata(*, title=_TITLE):
    return SimpleNamespace(
        doc_id="exchange_" + _RECEIPT, rcept_no=_RECEIPT,
        rcept_dt="20250317", report_nm=title, form=_FORM,
        is_correction=False)


class _Fields:
    def __init__(self, row):
        self.row = row

    def iter_docs(self, *_args, **_kwargs):
        yield _RECEIPT, [self.row]

    def rows(self, *_args, **_kwargs):
        return [self.row]


class _Reader:
    def __init__(self, metadata):
        self.metadata = metadata

    def documents(self, *_args, **_kwargs):
        if self.metadata is not None:
            yield self.metadata

    def resolve_document_version(self, *_args, **_kwargs):
        return SimpleNamespace(status="ok", selected=_RECEIPT)


def test_title_and_date_are_cited_from_exact_document_metadata():
    tool = DocumentTool(_Reader(_metadata()), _Fields(_row()))
    claims, limitations, used, _ = tool.run_task(
        _task(), trace=[], corp_name="가상회사")

    assert not limitations
    assert used == [_RECEIPT]
    assert len(claims) == 1
    claim = claims[0]
    assert _TITLE in claim.text
    assert "접수일 20250317" in claim.text
    assert len(claim.citations) == 1
    citation = claim.citations[0]
    assert citation.doc_id == "exchange_" + _RECEIPT
    assert citation.verification == "source_roundtrip"
    assert citation.locator == "document"
    assert citation.excerpt_prompt_safe == (
        f"공시명: {_TITLE}\n접수일: 20250317\n접수번호: {_RECEIPT}")
    answer = _investment_judgment_title_date_list(
        SimpleNamespace(claims=claims, limitations=[]),
        "가상회사가 2025년에 공시한 투자판단 관련 주요경영사항의 "
        "제목과 공시일을 정리해줘")
    assert "가상 기술 계약 체결" in answer
    assert "2025-03-17" in answer


def test_title_date_list_does_not_cite_an_unrelated_field_on_mismatch():
    tool = DocumentTool(_Reader(_metadata(title="다른 공시명")), _Fields(_row()))
    claims, limitations, used, _ = tool.run_task(
        _task(), trace=[], corp_name="가상회사")

    assert claims == []
    assert used == []
    assert [item.code for item in limitations] == ["evidence_unavailable"]


def test_other_document_find_keeps_its_existing_field_citation():
    row = _row()
    row.form = "다른공시"
    row.report_nm = "다른공시 (가상 내용)"
    tool = DocumentTool(_Reader(None), _Fields(row))
    tool._doc_cites = lambda *_args: [ClaimCitation(
        doc_id=row.doc_id, rcept_no=row.rcept_no,
        evidence_id="field-evidence", excerpt_prompt_safe="가상 내용")]

    claims, limitations, used, _ = tool.run_task(
        _task(form="다른공시"), trace=[], corp_name="가상회사")

    assert not limitations
    assert used == [_RECEIPT]
    assert claims[0].text == (
        "다른공시 (가상 내용) (접수번호 20250317900376, 접수일 20250317)")
    assert claims[0].citations[0].evidence_id == "field-evidence"
