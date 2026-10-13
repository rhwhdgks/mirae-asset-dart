"""Offline regression for P9-019's per-filing title/date provenance."""

from pathlib import Path

import pytest

from agent.semantic_intent_v1 import SemanticIntent
from agent.stage1_assembly import CORPUS_CUTOFF, REFERENCE_DATE
from agent.stage1_v1_backend_composition import build_stage1_v1_runtime
from agent.stage1_v1_service import Stage1V1NativeService
from app.composer import TemplateComposer
from app.orchestrator import Orchestrator
from app.tools import CanonicalToolBackend
from src.canonical.read import CanonicalReadModel


pytestmark = pytest.mark.canonical

_QUESTION = (
    "알테오젠이 2025년에 공시한 투자판단 관련 주요경영사항의 "
    "제목과 공시일을 정리해줘"
)
_CORP_CODE = "00989619"
_RECEIPTS = {
    "20250317900376", "20250317900379",
    "20251001900011", "20251127900375",
}
_ROOT = Path(__file__).resolve().parents[1]


def _display_title(report_name: str) -> str:
    prefix = "투자판단관련주요경영사항"
    title = report_name.removeprefix(prefix).strip()
    if title.startswith("(") and title.endswith(")"):
        return title[1:-1].strip()
    return title


class _NoProvider:
    def invoke(self, question: str):  # pragma: no cover - must never run
        raise AssertionError(f"saved-intent regression invoked a provider: {question}")


def _intent() -> SemanticIntent:
    return SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [{
            "entity_id": "entity-1", "surface": "알테오젠",
            "kind_hint": "company",
        }],
        "answer_items": [{
            "item_id": "item-1", "operation": "retrieve",
            "target": {
                "kind": "document",
                "surface": "공시한 투자판단 관련 주요경영사항",
                "entity_refs": ["entity-1"], "qualifier_surfaces": [],
            },
            "scope": {
                "target_period_expressions": ["2025년"],
                "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": "record", "projection_mode": "named_fields",
                "field_surfaces": ["제목", "공시일"],
                "presentation": "auto",
            },
        }],
        "answer_groups": [], "premises": [], "unresolved_mentions": [],
        "presentation": "auto",
    }, strict=True)


def _celltrion_intent() -> SemanticIntent:
    return SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [{
            "entity_id": "entity-1", "surface": "셀트리온",
            "kind_hint": "company",
        }],
        "answer_items": [{
            "item_id": "item-1", "operation": "retrieve",
            "target": {
                "kind": "document",
                "surface": "공시한 투자판단관련주요경영사항 제목",
                "entity_refs": ["entity-1"], "qualifier_surfaces": [],
            },
            "scope": {
                "target_period_expressions": ["2025년"],
                "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": "record_list", "projection_mode": "named_fields",
                "field_surfaces": ["공시일"], "presentation": "auto",
            },
        }],
        "answer_groups": [], "premises": [], "unresolved_mentions": [],
        "presentation": "auto",
    }, strict=True)


@pytest.fixture(scope="module")
def canonical() -> CanonicalReadModel:
    run_manifest = _ROOT / "out/canonical/run.json"
    if not run_manifest.is_file():
        pytest.skip("local canonical corpus is not built")
    return CanonicalReadModel(_ROOT / "out/canonical")


def test_p9019_pairs_each_title_and_date_with_its_own_verified_document(
        canonical: CanonicalReadModel, tmp_path: Path):
    service = Stage1V1NativeService(
        _NoProvider(),
        build_stage1_v1_runtime(
            canonical, reference_date=REFERENCE_DATE,
            corpus_cutoff=CORPUS_CUTOFF,
            clarification_db_path=tmp_path / "p9-019.sqlite3",
        ),
    )
    result = service.start_from_intent("P9-019-offline", _QUESTION, _intent())

    assert result.handoff is not None and result.handoff.status == "ready"
    task = result.handoff.plan.tasks[0]
    assert task.kind == "document" and task.operation == "find"
    assert task.corp_code == _CORP_CODE
    assert task.selector.form == "투자판단관련주요경영사항"
    assert task.selector.rcept_from == "20250101"
    assert task.selector.rcept_to == "20251231"

    payload = Orchestrator(CanonicalToolBackend(rm=canonical)).run(
        result.handoff, question_id="P9-019-offline")
    answer = TemplateComposer().compose(payload, question=_QUESTION)
    metadata_by_receipt = {
        row.rcept_no: row for row in canonical.documents(
            as_of=CORPUS_CUTOFF, corp_code=_CORP_CODE,
            doc_group="exchange", form="투자판단관련주요경영사항")
        if row.rcept_no in _RECEIPTS
    }
    citations_by_receipt = {}
    for claim in payload.claims:
        for citation in claim.citations:
            if citation.rcept_no in _RECEIPTS:
                citations_by_receipt.setdefault(citation.rcept_no, []).append(citation)

    assert payload.final_status == "answer"
    assert set(metadata_by_receipt) == _RECEIPTS
    assert set(citations_by_receipt) == _RECEIPTS
    assert "4건" in answer
    assert "ALT-B4" in answer
    for receipt, document in metadata_by_receipt.items():
        assert _display_title(document.report_nm) in answer
        display_date = (
            f"{document.rcept_dt[:4]}-{document.rcept_dt[4:6]}-"
            f"{document.rcept_dt[6:8]}"
        )
        assert display_date in answer
        expected_excerpt = (
            f"공시명: {document.report_nm}\n"
            f"접수일: {document.rcept_dt}\n"
            f"접수번호: {receipt}"
        )
        for citation in citations_by_receipt[receipt]:
            assert citation.doc_id == document.doc_id
            if citation.locator != "document":
                pytest.fail(
                    f"{receipt}: locator={citation.locator!r}, "
                    f"excerpt={citation.excerpt_prompt_safe!r}")
            assert citation.excerpt_prompt_safe == expected_excerpt
            assert citation.verification == "source_roundtrip"


def test_p9019b_keeps_all_celltrion_title_date_citations_bound_to_metadata(
        canonical: CanonicalReadModel, tmp_path: Path):
    question = (
        "셀트리온이 2025년에 공시한 투자판단관련주요경영사항 제목과 "
        "공시일 목록을 알려줘."
    )
    service = Stage1V1NativeService(
        _NoProvider(),
        build_stage1_v1_runtime(
            canonical, reference_date=REFERENCE_DATE,
            corpus_cutoff=CORPUS_CUTOFF,
            clarification_db_path=tmp_path / "p9-019b.sqlite3",
        ),
    )
    result = service.start_from_intent(
        "P9-019b-offline", question, _celltrion_intent())

    assert result.handoff is not None and result.handoff.status == "ready"
    task = result.handoff.plan.tasks[0]
    assert task.kind == "document" and task.operation == "find"
    assert task.corp_code == "00413046"
    assert task.selector.form == "투자판단관련주요경영사항"
    assert task.selector.rcept_from == "20250101"
    assert task.selector.rcept_to == "20251231"

    payload = Orchestrator(CanonicalToolBackend(rm=canonical)).run(
        result.handoff, question_id="P9-019b-offline")
    answer = TemplateComposer().compose(payload, question=question)
    documents = {
        row.rcept_no: row for row in canonical.documents(
            as_of=CORPUS_CUTOFF, corp_code="00413046",
            doc_group="exchange", form="투자판단관련주요경영사항")
        if row.rcept_dt.startswith("2025")
    }
    # A corrected filing may also produce a separate correction-lineage claim
    # with source-table evidence. Restrict this check to the document-title
    # claims; the two claim types have different evidence roles by design.
    document_claims = {
        claim.label.removeprefix("문서 "): claim
        for claim in payload.claims
        if claim.label.startswith("문서 ")
        and claim.label.removeprefix("문서 ") in documents
    }

    assert payload.final_status == "answer"
    assert len(documents) == 26
    assert set(document_claims) == set(documents)
    assert "26건" in answer
    for receipt, document in documents.items():
        assert _display_title(document.report_nm) in answer
        display_date = (
            f"{document.rcept_dt[:4]}-{document.rcept_dt[4:6]}-"
            f"{document.rcept_dt[6:8]}"
        )
        assert display_date in answer
        claim = document_claims[receipt]
        assert document.report_nm in claim.text
        assert receipt in claim.text
        assert len(claim.citations) == 1
        citation = claim.citations[0]
        expected_excerpt = (
            f"공시명: {document.report_nm}\n"
            f"접수일: {document.rcept_dt}\n"
            f"접수번호: {receipt}"
        )
        assert citation.doc_id == document.doc_id
        assert citation.rcept_no == receipt
        assert citation.locator == "document"
        assert citation.excerpt_prompt_safe == expected_excerpt
        assert citation.verification == "source_roundtrip"
