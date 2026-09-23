"""end-to-end 조립: QueryPlanHandoff → AnswerPayload → answer text → /answer 5필드.

question 원문은 Orchestrator에는 넘기지 않는다(계약). composer와 응답 필드에만 사용한다.
"""
from __future__ import annotations

import app.env  # noqa: F401  (.env 자동 로드)
from app.composer import HcxComposer, TemplateComposer, build_answer_response
from app.orchestrator import Orchestrator


class AnswerPipeline:
    def __init__(self, backend=None, *, use_hcx: bool = True):
        self.orch = Orchestrator(backend)
        self.composer = HcxComposer() if use_hcx else None
        self.template = TemplateComposer()

    def run(
            self, handoff, *, question_id: str, question: str = "",
            runtime_annotations: dict | None = None,
            deadline_monotonic: float | None = None,
            ) -> tuple[dict, object]:
        payload = self.orch.run(
            handoff, question_id=question_id,
            runtime_annotations=runtime_annotations)
        if self.composer is not None:
            text = self.composer.compose(
                payload, question=question or None, trace=payload.trace,
                deadline_monotonic=deadline_monotonic)
        else:
            from app.orchestrator.payload import TraceEvent
            payload.trace.append(TraceEvent(seq=len(payload.trace)+1, stage="compose", summary="template composer", detail={}))
            text = self.template.compose(payload, question=question or None)
        return build_answer_response(question_id, question, payload, text), payload
