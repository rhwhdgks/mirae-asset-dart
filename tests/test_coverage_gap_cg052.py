"""Offline regression for CG-052's investment-judgment correction route."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.semantic_intent_v1 import SemanticIntent
from agent.stage1_assembly import CORPUS_CUTOFF, REFERENCE_DATE
from agent.stage1_v1_backend_composition import build_stage1_v1_runtime
from agent.stage1_v1_document_backends import InvestmentJudgmentTitleDateRegrounder
from agent.stage1_v1_service import Stage1V1NativeService
from app.composer import TemplateComposer
from app.composer.render import build_answer_response
from app.orchestrator import Orchestrator
from app.tools import CanonicalToolBackend
from server.stage1 import NativeStage1
from src.canonical.read import CanonicalReadModel


pytestmark = pytest.mark.canonical

_QUESTION = (
    "LG에너지솔루션이 2026년 3월 31일 정정한 투자판단 관련 "
    "주요경영사항은 무엇이 바뀌었어?"
)
_RECEIPT = "20260331801967"
_CORP_CODE = "01515323"
_ROOT = Path(__file__).resolve().parents[1]


class _MockWireRunner:
    """Replay the captured semantic shape; this test never calls an LLM."""

    def __init__(self, intent: SemanticIntent) -> None:
        self.intent = intent

    def invoke(self, question: str, *, deadline_monotonic=None):
        del question, deadline_monotonic
        return SimpleNamespace(semantic_intent=self.intent)


def _intent() -> SemanticIntent:
    return SemanticIntent.model_validate({
        "schema_version": "stage1-semantic-intent/1.1",
        "entities": [{
            "entity_id": "entity-1", "kind_hint": "company",
            "surface": "LG에너지솔루션",
        }],
        "answer_items": [{
            "item_id": "item-1", "operation": "retrieve",
            "target": {
                "kind": "document", "surface": "투자판단 관련 주요경영사항",
                "entity_refs": ["entity-1"],
                "qualifier_surfaces": ["2026년 3월 31일 정정"],
            },
            "scope": {
                "target_period_expressions": [], "as_of_expression": None,
                "document_group_expression": None,
                "scope_qualifier_expressions": [],
            },
            "selection": None,
            "output": {
                "shape": "narrative", "projection_mode": "named_fields",
                "field_surfaces": ["무엇이 바뀌었어"],
                "presentation": "prose",
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


def test_cg052_correction_wording_is_restored_to_correction_lineage_shape():
    intent = _intent()

    result = InvestmentJudgmentTitleDateRegrounder()(_QUESTION, intent)

    item = result.answer_items[0]
    assert item.target.surface == "투자판단 관련 주요경영사항"
    assert item.target.qualifier_surfaces == []
    assert item.scope.target_period_expressions == ["2026년 3월 31일"]
    assert item.output.projection_mode == "whole_target"
    assert item.output.field_surfaces == []
    assert item.output.shape == "narrative"


def test_cg052_returns_the_selected_correction_without_live_calls(
        canonical: CanonicalReadModel, tmp_path: Path):
    native = NativeStage1.__new__(NativeStage1)
    native._canonical = canonical
    native.service = Stage1V1NativeService(
        _MockWireRunner(_intent()),
        build_stage1_v1_runtime(
            canonical, reference_date=REFERENCE_DATE,
            corpus_cutoff=CORPUS_CUTOFF,
            clarification_db_path=tmp_path / "cg052.sqlite3",
        ),
    )
    result = native.service.start("CG-052-offline", _QUESTION)
    metadata: dict = {}
    resolution = native._resolution_from_result(
        result, meta=metadata, question=_QUESTION)
    handoff = resolution.handoff
    assert handoff is not None and handoff.status == "ready", handoff

    payload = Orchestrator(CanonicalToolBackend()).run(
        handoff, question_id="CG-052-offline",
        runtime_annotations=metadata.get("runtime_annotations"))
    answer_text = TemplateComposer().compose(payload, question=_QUESTION)
    answer = build_answer_response(
        "CG-052-offline", _QUESTION, payload, answer_text)["answer"]

    cited_receipts = {
        citation.rcept_no for claim in payload.claims
        for citation in claim.citations
    }
    assert payload.final_status == "answer"
    assert "정정" in answer
    assert "합작법인 지분구조 변경" in answer
    assert "채무보증" in answer
    assert _RECEIPT in cited_receipts
