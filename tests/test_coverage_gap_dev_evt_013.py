"""Offline end-to-end regression for DEV-EVT-013's correction timeline."""

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
    "현대건설이 2023년 7월 25일 공시한 계약의 전체 정정 내역을 "
    "최신 유효본 기준으로 정리해줘."
)
_CORP_CODE = "00164478"
_RECEIPTS = {
    "20230725800160", "20250724800687",
    "20250725800639", "20260226800506",
}
_ROOT = Path(__file__).resolve().parents[1]


class _NoProvider:
    def invoke(self, question: str):  # pragma: no cover - must never run
        raise AssertionError(f"saved-intent regression invoked a provider: {question}")


def _intent() -> SemanticIntent:
    return SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [{
            "entity_id": "entity-1", "kind_hint": "company",
            "surface": "현대건설",
        }],
        "answer_items": [{
            "item_id": "item-1", "operation": "retrieve",
            "target": {
                "kind": "document", "surface": "계약",
                "entity_refs": ["entity-1"],
                "qualifier_surfaces": ["2023년 7월 25일"],
            },
            "scope": {
                "target_period_expressions": [],
                "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": "narrative", "projection_mode": "whole_target",
                "field_surfaces": [], "presentation": "prose",
            },
        }],
        "answer_groups": [], "premises": [], "unresolved_mentions": [],
        "presentation": "prose",
    }, strict=True)


@pytest.fixture(scope="module")
def canonical() -> CanonicalReadModel:
    run_manifest = _ROOT / "out/canonical/run.json"
    if not run_manifest.is_file():
        pytest.skip("local canonical corpus is not built")
    return CanonicalReadModel(_ROOT / "out/canonical")


def test_dev_evt_013_shows_the_verified_full_correction_timeline(
        canonical: CanonicalReadModel, tmp_path: Path):
    service = Stage1V1NativeService(
        _NoProvider(),
        build_stage1_v1_runtime(
            canonical, reference_date=REFERENCE_DATE,
            corpus_cutoff=CORPUS_CUTOFF,
            clarification_db_path=tmp_path / "dev-evt-013.sqlite3",
        ),
    )
    result = service.start_from_intent(
        "DEV-EVT-013-offline", _QUESTION, _intent())

    assert result.handoff is not None and result.handoff.status == "ready"
    payload = Orchestrator(CanonicalToolBackend(rm=canonical)).run(
        result.handoff, question_id="DEV-EVT-013-offline")
    answer = TemplateComposer().compose(payload, question=_QUESTION)
    cited_receipts = {
        citation.rcept_no for claim in payload.claims
        for citation in claim.citations
    }

    assert payload.final_status == "partial_answer"
    assert _RECEIPTS <= cited_receipts
    assert "2023-11-24" in answer and "2025-07-24" in answer
    assert "2026-06-30" in answer and "2026-12-31" in answer
    assert "원문에 내용 없이 「-」로 기재됐습니다" in answer
    assert "앞선 원공시가 없어" in answer
    assert any(
        limitation.code == "source_scope_prevents_complete_lineage"
        for limitation in payload.limitations)
