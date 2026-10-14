"""Recording must not change answers, field contracts, or exception behavior."""
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from agent.latency_diagnostics import capture_latency, record_latency_event
from app.pipeline import AnswerPipeline
from server.app import create_app
from server.runtime import AnswerResult, ServerRuntime


class FakeRuntime:
    ready = True
    build = {"latency_capture": True}
    def __init__(self, status=200):
        self.logs = []
        self.http_status = status
    def answer(self, question_id, question):
        return AnswerResult({"question_id": question_id, "question": question,
                             "retrieved_context": "", "think_trace": "", "answer": "원래 답변"},
                            "ok", self.http_status, meta={})
    def log(self, row):
        self.logs.append(row)
    def close(self):
        pass


@pytest.mark.parametrize("http_status", [200, 503])
def test_request_ids_link_server_log_without_changing_five_fields(http_status):
    rt = FakeRuntime(http_status)
    attempt = str(uuid4())
    with TestClient(create_app(rt, autostart=False)) as client:
        response = client.get("/answer", params={"question_id": "Q", "question": "질문"},
                              headers={"X-Live-Attempt-ID": attempt})
    assert response.status_code == http_status
    assert set(response.json()) == {"question_id", "question", "retrieved_context", "think_trace", "answer"}
    assert response.json()["answer"] == "원래 답변"
    assert rt.logs[0]["req_id"] == response.headers["X-Request-ID"]
    assert rt.logs[0]["client_attempt_id"] == attempt


@pytest.mark.parametrize("attempt", ["secret-value", "x" * 36])
def test_arbitrary_correlation_headers_are_not_logged(attempt):
    rt = FakeRuntime()
    with TestClient(create_app(rt, autostart=False)) as client:
        response = client.get("/answer", params={"question_id": "Q", "question": "질문"},
                              headers={"X-Live-Attempt-ID": attempt})
    assert response.status_code == 200
    assert "client_attempt_id" not in rt.logs[0]


def test_missing_params_also_have_exact_request_identity():
    rt = FakeRuntime()
    with TestClient(create_app(rt, autostart=False)) as client:
        response = client.get("/answer")
    assert response.headers["X-Request-ID"] == rt.logs[0]["req_id"]
    assert rt.logs[0]["status"] == "invalid_params"


def test_readiness_exposes_capture_mode_without_credentials(monkeypatch):
    fake = SimpleNamespace(backend=None, pipeline=None, canonical={"build_id": "fake"},
                           warm={}, stage1_info={})
    monkeypatch.setenv("ANSWER_CAPTURE_LATENCY", "1")
    assert ServerRuntime._build_info(fake)["latency_capture"] is True
    monkeypatch.delenv("ANSWER_CAPTURE_LATENCY")
    assert ServerRuntime._build_info(fake)["latency_capture"] is False


@pytest.mark.parametrize("use_model", [False, True])
def test_pipeline_spans_wrap_real_boundaries_without_changing_response(monkeypatch, use_model):
    import app.pipeline as pipeline_module
    payload = SimpleNamespace(trace=[])
    calls = []
    class Orch:
        def run(self, handoff, **kwargs):
            calls.append("retrieve")
            return payload
    class Composer:
        def compose(self, given, **kwargs):
            assert given is payload
            calls.append("compose")
            return "같은 답변"
    pipeline = object.__new__(AnswerPipeline)
    pipeline.orch, pipeline.template = Orch(), Composer()
    pipeline.composer = Composer() if use_model else None
    monkeypatch.setattr(pipeline_module, "build_answer_response", lambda qid, q, p, text: {"answer": text})
    with capture_latency() as events:
        response, returned = pipeline.run(object(), question_id="Q", question="질문")
    assert response == {"answer": "같은 답변"} and returned is payload
    assert calls == ["retrieve", "compose"]
    assert [event["stage"] for event in events] == ["retrieval_validation", "compose"]
    assert all(event["elapsed_s"] >= 0 for event in events)


def test_pipeline_exception_is_recorded_and_propagated():
    class Broken:
        def run(self, *args, **kwargs):
            raise RuntimeError("same failure")
    pipeline = object.__new__(AnswerPipeline)
    pipeline.orch = Broken()
    with capture_latency() as events, pytest.raises(RuntimeError, match="same failure"):
        pipeline.run(object(), question_id="Q")
    assert events == [{"stage": "retrieval_validation", "elapsed_s": events[0]["elapsed_s"],
                       "outcome": "RuntimeError"}]


@pytest.mark.parametrize("capture", [False, True])
def test_runtime_capture_preserves_response_and_existing_meta(monkeypatch, capture):
    monkeypatch.setenv("ANSWER_CAPTURE_LATENCY", "1" if capture else "0")
    def answer(self, *args, **kwargs):
        record_latency_event("compose", elapsed_s=0.25)
        return AnswerResult({"answer": "원래 답변"}, "ok", 200,
                            meta={"t_stage1": 2.0, "stage1_meta": {"cache_hit": True}})
    monkeypatch.setattr(ServerRuntime, "_answer_sync_impl", answer)
    result = ServerRuntime._answer_sync(SimpleNamespace(), "Q", "질문", stage1_mode="auto", compose="auto")
    assert result.response == {"answer": "원래 답변"}
    assert result.meta["t_stage1"] == 2.0 and result.meta["stage1_meta"]["cache_hit"] is True
    assert ("latency_profile" in result.meta) is capture
    if capture:
        assert result.meta["latency_profile"]["events"] == [{"stage": "compose", "elapsed_s": 0.25}]
