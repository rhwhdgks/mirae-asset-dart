"""Offline recording regressions: no corpus, credentials, or provider calls."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import stat
import sys

import httpx
import pytest

from scripts.live_run_evidence import (
    LiveRunEvidence, collect_diagnostics, main as recollect, safe_base_url,
)
from scripts.run_edge43_live import invoke
from scripts import run_live_question_set as runner


def _invoke(handler, **kwargs):
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        return invoke(client, "http://test", "Q1", "질문", retry_503=kwargs.pop("retry_503", 0),
                      retry_wait_s=0, capture_diagnostics=kwargs.pop("capture_diagnostics", True),
                      **kwargs)


def _result():
    return _invoke(lambda request: httpx.Response(200, json={"answer": "답변"},
                                                  headers={"X-Request-ID": "server-id"}))


def _record(result, **extra):
    return {"endpoint": "/answer", "question_id": result["question_id"],
            "question": result["question"], "req_id": "server-id",
            "client_attempt_id": result["attempts"][0]["client_attempt_id"],
            "t_stage1": 3.0, "t_pipeline": 2.0, "stage1_meta": {"cache_hit": True},
            "latency_profile": {"events": [{"stage": "compose", "elapsed_s": 1.0}]}, **extra}


def test_all_retry_bodies_and_unique_attempt_ids_are_kept():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(503 if len(requests) == 1 else 200,
                              json={"answer": "最初失敗" if len(requests) == 1 else "성공"},
                              headers={"X-Request-ID": str(len(requests))})
    result = _invoke(handler, retry_503=1)
    assert [a["http"] for a in result["attempts"]] == [503, 200]
    assert result["attempts"][0]["response"] == {"answer": "最初失敗"}
    assert len({a["client_attempt_id"] for a in result["attempts"]}) == 2
    assert result["total_elapsed"] >= result["elapsed"]
    assert result["attempts"][0]["server_request_id"] == "1"
    for request, attempt in zip(requests, result["attempts"]):
        assert request.headers["X-Live-Attempt-ID"] == attempt["client_attempt_id"]
        assert datetime.fromisoformat(attempt["started_at"]).tzinfo == timezone.utc


def test_non_json_and_timeout_are_evidence_not_success():
    raw = _invoke(lambda request: httpx.Response(502, text="upstream error"))
    assert raw["attempts"][0]["response"] == {"raw_body": "upstream error"}
    def timeout(request):
        raise httpx.ReadTimeout("bounded timeout", request=request)
    result = _invoke(timeout)
    assert result["http"] is None and result["response"] is None
    assert "ReadTimeout" in result["attempts"][0]["error"]
    assert result["attempts"][0]["client_attempt_id"]
    assert result["attempts"][0]["finished_at"]


def test_legacy_invoke_is_opt_in_only():
    result = _invoke(lambda request: httpx.Response(200, json={}), capture_diagnostics=False)
    assert set(result) == {"question_id", "question", "http", "elapsed", "attempt_count",
                           "attempts", "response", "error"}
    assert set(result["attempts"][0]) == {"attempt", "http", "elapsed", "response"}


def test_exact_id_join_preserves_profile_and_excludes_stale_question_match():
    result = _result()
    current = _record(result)
    stale = _record(result, req_id="older", client_attempt_id="older")
    wrong_question = _record(result, question="다른 질문")
    out = collect_diagnostics(None, "", result, records=[stale, wrong_question, current])
    assert out["attempts"][0]["state"] == "matched"
    assert out["attempts"][0]["records"] == [current]
    assert out["attempts"][0]["records"][0]["stage1_meta"]["cache_hit"] is True


def test_id_conflict_and_duplicate_records_are_not_silently_selected():
    result = _result()
    conflict = _record(result, client_attempt_id="other")
    assert collect_diagnostics(None, "", result, records=[conflict])["attempts"][0]["state"] == "not_found"
    current = _record(result)
    assert collect_diagnostics(None, "", result, records=[current, current])["attempts"][0]["state"] == "ambiguous"


def test_late_server_completion_can_join_client_timeout_by_attempt_id():
    def timeout(request):
        raise httpx.ReadTimeout("timeout", request=request)
    result = _invoke(timeout)
    missing = collect_diagnostics(None, "", result, records=[])
    assert missing["attempts"][0]["state"] == "not_found"
    later = collect_diagnostics(None, "", result, records=[_record(result)])
    assert later["attempts"][0]["state"] == "matched"
    assert result["http"] is None  # later diagnostics never rewrite a failure


def test_api_log_lookup_handles_midnight_and_failure():
    result = _result()
    result["attempts"][0].update(started_at="2026-10-13T23:59:59+00:00",
                                 finished_at="2026-10-14T00:00:01+00:00")
    days = []
    def handler(request):
        days.append(request.url.params["date"])
        return httpx.Response(200, json={"items": [_record(result)] if days[-1] == "20261014" else []})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        out = collect_diagnostics(client, "http://test", result)
    assert days == ["20261013", "20261014"]
    assert out["attempts"][0]["state"] == "matched"
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503))) as client:
        out = collect_diagnostics(client, "http://test", result)
    assert out["attempts"][0]["state"] == "collection_error"
    assert len(out["collection_errors"]) == 2


def test_bundle_is_private_complete_and_never_overwrites(tmp_path):
    path = tmp_path / "responses.jsonl"
    questions = b'{"id":"Q1","question":"question"}\n'
    with LiveRunEvidence(path, question_bytes=questions, manifest={"expected_count": 1}) as evidence:
        evidence.save_response(_result())
        evidence.save_diagnostics({"attempts": [{"state": "not_found"}]})
    manifest = json.loads(evidence.paths["manifest"].read_text())
    assert manifest["questions_sha256"] == sha256(questions).hexdigest()
    assert evidence.paths["questions"].read_bytes() == questions
    assert json.loads(evidence.paths["summary"].read_text())["state"] == "completed"
    for artifact in (path, *evidence.paths.values()):
        assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        LiveRunEvidence(path, question_bytes=questions, manifest={})
    assert path.read_bytes() == before


@pytest.mark.parametrize("collision", ["manifest", "questions", "diagnostics", "summary", "symlink"])
def test_any_sidecar_collision_blocks_before_responses(tmp_path, collision):
    path = tmp_path / "responses.jsonl"
    suffix = {"manifest": ".manifest.json", "questions": ".questions.jsonl",
              "diagnostics": ".diagnostics.jsonl", "summary": ".summary.json"}
    if collision == "symlink":
        path.symlink_to(tmp_path / "nonexistent")
    else:
        Path(str(path) + suffix[collision]).write_text("keep me")
    with pytest.raises(FileExistsError):
        LiveRunEvidence(path, question_bytes=b"", manifest={})
    if collision != "symlink":
        assert not path.exists()


def test_interrupted_run_keeps_flushed_response(tmp_path):
    path = tmp_path / "responses.jsonl"
    with pytest.raises(RuntimeError):
        with LiveRunEvidence(path, question_bytes=b"", manifest={}) as evidence:
            evidence.save_response(_result())
            raise RuntimeError("diagnostic failed after response")
    assert json.loads(path.read_text())["response"] == {"answer": "답변"}
    summary = json.loads(evidence.paths["summary"].read_text())
    assert summary["state"] == "interrupted" and summary["recorded_questions"] == 1


@pytest.mark.parametrize("url", ["http://user:secret@test", "http://test?key=secret",
                                  "http://test#secret", "file:///private"])
def test_credential_or_non_http_url_is_rejected(url):
    with pytest.raises(ValueError):
        safe_base_url(url)


def test_curated_runner_records_one_question_without_external_calls(tmp_path, monkeypatch):
    questions = tmp_path / "input.jsonl"
    questions.write_text('{"id":"Q1","question":"질문"}\n', encoding="utf-8")
    output = tmp_path / "responses.jsonl"
    requests, logs = [], []
    def handler(request):
        requests.append(request.url.path)
        if request.url.path == "/readyz":
            return httpx.Response(200, json={"ready": True, "latency_capture": True,
                "canonical_build_id": "fake-build", "stage1": {"native": True, "fixture_fallback": False}})
        if request.url.path == "/answer":
            result = {"question_id": "Q1", "question": "질문", "attempts": [
                {"client_attempt_id": request.headers["X-Live-Attempt-ID"]}]}
            logs.append(_record(result))
            return httpx.Response(200, json={"answer": "답변"}, headers={"X-Request-ID": "server-id"})
        assert request.url.path == "/v1/requests"
        return httpx.Response(200, json={"items": logs})
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(httpx, "get", lambda url, **kw: real_client(
        transport=httpx.MockTransport(handler)).get(url, **kw))
    monkeypatch.setattr(sys, "argv", ["runner", "--questions", str(questions), "--expected-count", "1",
                                      "--output", str(output), "--base-url", "http://test"])
    assert runner.main() == 0
    assert requests == ["/readyz", "/answer", "/v1/requests"]
    assert json.loads(Path(str(output) + ".diagnostics.jsonl").read_text())["attempts"][0]["state"] == "matched"
    assert json.loads(Path(str(output) + ".manifest.json").read_text())["server_ready"]["canonical_build_id"] == "fake-build"


def test_recollection_is_read_only_and_preserves_original_failure(tmp_path, monkeypatch):
    result = _result()
    result.update(http=None, response=None, error="ReadTimeout")
    result["attempts"][0].pop("server_request_id")
    responses, server_log, output = [tmp_path / name for name in ("responses.jsonl", "server.jsonl", "later.jsonl")]
    responses.write_text(json.dumps(result), encoding="utf-8")
    server_log.write_text("incomplete trailing line\n" + json.dumps(_record(result)) + "\n", encoding="utf-8")
    before = responses.read_bytes(), server_log.read_bytes()
    monkeypatch.setattr(sys, "argv", ["recollect", "--responses", str(responses),
                                      "--server-log", str(server_log), "--output", str(output)])
    assert recollect() == 0
    out = json.loads(output.read_text())
    assert out["attempts"][0]["state"] == "matched" and out["malformed_log_lines"] == 1
    assert before == (responses.read_bytes(), server_log.read_bytes())
    with pytest.raises(FileExistsError):
        recollect()


@pytest.mark.parametrize("stage1,capture", [
    ({"native": False, "fixture_fallback": False}, True),
    ({"native": True, "fixture_fallback": True}, True),
    ({"native": True, "fixture_fallback": False}, False),
])
def test_incomplete_live_configuration_blocks_before_any_answer(tmp_path, monkeypatch, stage1, capture):
    questions = tmp_path / "input.jsonl"
    questions.write_text('{"id":"Q1","question":"질문"}\n', encoding="utf-8")
    output = tmp_path / "responses.jsonl"
    requests = []
    def get(url, **kwargs):
        requests.append(url)
        return httpx.Response(200, json={"stage1": stage1, "latency_capture": capture})
    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(sys, "argv", ["runner", "--questions", str(questions), "--expected-count", "1",
                                      "--output", str(output)])
    with pytest.raises(RuntimeError):
        runner.main()
    assert len(requests) == 1 and requests[0].endswith("/readyz")
    assert not output.exists()


def test_code_identity_hashes_exact_diff_bytes_without_retaining_content(monkeypatch, tmp_path):
    import src.code_identity as identity
    diff = b"fake patch with trailing newline\n"
    def git(command, **kwargs):
        if command[1] == "diff":
            return diff
        return {"rev-parse": b"head-sha\n", "branch": b"test-branch\n", "status": b" M app/pipeline.py\n"}[command[1]]
    monkeypatch.setattr(identity.subprocess, "check_output", git)
    result = identity.code_identity(tmp_path)
    assert result["tracked_diff_sha256"] == sha256(diff).hexdigest()
    assert result["git_head"] == "head-sha" and result["tracked_dirty"] is True
    assert "fake patch" not in json.dumps(result)
    def missing(*args, **kwargs):
        raise FileNotFoundError("git missing")
    monkeypatch.setattr(identity.subprocess, "check_output", missing)
    assert identity.code_identity(tmp_path)["git_head"] is None
