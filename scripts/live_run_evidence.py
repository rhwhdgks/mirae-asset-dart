"""Private, immutable live-run artifacts and exact request-log correlation.

This module never invokes /answer, loads .env, or reads provider credentials.
The optional CLI only recollects diagnostics for already recorded attempts.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

import httpx

from src.code_identity import code_identity as runner_identity


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def private_file(path: Path):
    """Exclusive creation; even a dangling symlink must not be followed."""
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                     "w", encoding="utf-8")


def write_row(stream, row: dict) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def safe_base_url(value: str) -> str:
    url = httpx.URL(value)
    if (url.scheme not in {"http", "https"} or not url.host
            or url.username or url.password or url.query or url.fragment):
        raise ValueError("base URL must be HTTP(S), without credentials/query/fragment")
    return str(url).rstrip("/")


def collect_diagnostics(client: httpx.Client | None, base_url: str, result: dict,
                        *, records: list[dict] | None = None) -> dict:
    """Join UUIDs only, never question text/time proximity alone.

    A timeout can finish on the server later. A missing record remains missing
    until an explicit recollection, not a zero-duration or successful request.
    """
    errors = []
    if records is None:
        records = []
        days = set()
        for attempt in result["attempts"]:
            for key in ("started_at", "finished_at"):
                if attempt.get(key):
                    days.add(datetime.fromisoformat(attempt[key]).astimezone(
                        timezone.utc).strftime("%Y%m%d"))
        for day in sorted(days):
            try:
                response = client.get(f"{base_url}/v1/requests",
                                      params={"date": day, "limit": 500}, timeout=10.0)
                response.raise_for_status()
                items = response.json()["items"]
                if not isinstance(items, list) or not all(isinstance(r, dict) for r in items):
                    raise ValueError("invalid request-log items")
                records.extend(items)
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                errors.append({"date": day, "error_type": type(exc).__name__})
    matched = []
    for attempt in result["attempts"]:
        request_id = attempt.get("server_request_id")
        client_id = attempt.get("client_attempt_id")
        candidates = []
        for row in records:
            if (row.get("endpoint") != "/answer"
                    or row.get("question_id") != result["question_id"]
                    or row.get("question") != result["question"]):
                continue
            if request_id and row.get("req_id") != request_id:
                continue
            if client_id and row.get("client_attempt_id") not in (None, client_id):
                continue
            if ((request_id and row.get("req_id") == request_id)
                    or (client_id and row.get("client_attempt_id") == client_id)):
                candidates.append(row)
        state = ("matched" if len(candidates) == 1 else "ambiguous" if candidates
                 else "collection_error" if errors else "not_found")
        matched.append({"attempt": attempt["attempt"], "state": state,
                        "client_attempt_id": client_id, "server_request_id": request_id,
                        "records": candidates})
    return {"question_id": result["question_id"], "collected_at": utc_now(),
            "collection_errors": errors, "attempts": matched}


class LiveRunEvidence:
    """Reserve every path before paid requests; interrupted runs lack a summary."""

    def __init__(self, output: Path, *, question_bytes: bytes, manifest: dict):
        self.output = output
        self.paths = {key: output.with_name(output.name + suffix) for key, suffix in (
            ("manifest", ".manifest.json"), ("questions", ".questions.jsonl"),
            ("diagnostics", ".diagnostics.jsonl"), ("summary", ".summary.json"))}
        for path in (output, *self.paths.values()):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"evidence path already exists: {path}")
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.run_id = str(uuid4())
        self.manifest = {**manifest, "schema": "live-run-evidence/1", "run_id": self.run_id,
                         "started_at": utc_now(), "concurrency": 1,
                         "questions_sha256": sha256(question_bytes).hexdigest(),
                         "files": {"responses": output.name,
                                   **{k: p.name for k, p in self.paths.items()}}}
        self.stack = ExitStack()
        self.rows = 0
        self.http_counts: Counter = Counter()
        self.diagnostic_counts: Counter = Counter()
        try:
            # Reserve the summary too; an empty summary means interrupted.
            self.stream = self.stack.enter_context(private_file(output))
            self.diagnostics = self.stack.enter_context(private_file(self.paths["diagnostics"]))
            self.summary = self.stack.enter_context(private_file(self.paths["summary"]))
            with private_file(self.paths["questions"]) as stream:
                stream.write(question_bytes.decode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            with private_file(self.paths["manifest"]) as stream:
                write_row(stream, self.manifest)
        except BaseException:
            self.stack.close()
            raise

    def save_response(self, result: dict) -> None:
        write_row(self.stream, {"run_id": self.run_id, **result})
        self.rows += 1
        self.http_counts[str(result["http"])] += 1

    def save_diagnostics(self, result: dict) -> None:
        write_row(self.diagnostics, {"run_id": self.run_id, **result})
        self.diagnostic_counts.update(a["state"] for a in result["attempts"])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            write_row(self.summary, {
                "run_id": self.run_id, "finished_at": utc_now(),
                "state": "interrupted" if exc_type else "completed",
                "recorded_questions": self.rows, "http_counts": dict(self.http_counts),
                "diagnostic_counts": dict(self.diagnostic_counts),
                "error_type": exc_type.__name__ if exc_type else None,
            })
        finally:
            self.stack.close()


def main() -> int:
    """Recollect from local logs, including records completed after a timeout."""
    import argparse
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [json.loads(line) for line in args.responses.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    # Retain only this run's exact IDs, not other users' log rows.
    ids = {a[key] for r in results for a in r["attempts"]
           for key in ("server_request_id", "client_attempt_id") if a.get(key)}
    records, malformed = [], 0
    for path in args.server_log:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("not an object")
                except ValueError:
                    malformed += 1
                    continue
                if row.get("req_id") in ids or row.get("client_attempt_id") in ids:
                    records.append(row)
    with private_file(args.output) as stream:
        for result in results:
            row = collect_diagnostics(None, "", result, records=records)
            write_row(stream, {"run_id": result.get("run_id"),
                               "malformed_log_lines": malformed, **row})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
