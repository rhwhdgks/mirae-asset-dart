"""Bounded, integrity-checked QueryPlanHandoff capture for Stage2~4 replay.

Only the public v0.4 handoff contract is persisted.  In particular this module
has no parameters for the raw question, provider response/wire, request id, or
environment.  Captures are disabled by the server unless explicitly enabled.
"""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from agent.query_plan import QueryPlanHandoff
from app.orchestrator.adapter import load_handoff_json


CAPTURE_SCHEMA = "query-plan-handoff-replay/0.1"
CAPTURE_FILE = "handoffs.jsonl"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_BUILD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_REQUIRED_KEYS = {
    "capture_schema", "capture_id", "captured_at",
    "handoff_sha256", "handoff",
}
_OPTIONAL_KEYS = {"canonical_build_id"}


class HandoffReplayError(RuntimeError):
    """Base class for capture configuration and integrity failures."""


class HandoffReplayLimitError(HandoffReplayError):
    """A record or segment exceeds a configured bound."""


class HandoffReplayIntegrityError(HandoffReplayError):
    """A replay record is malformed, non-canonical, or digest-mismatched."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HandoffReplayIntegrityError("capture is not canonical JSON") from exc


def _validated_handoff_payload(
        handoff: QueryPlanHandoff | dict,
        ) -> tuple[QueryPlanHandoff, dict, bytes]:
    """Round-trip through the production adapter before hashing or replay."""

    if isinstance(handoff, QueryPlanHandoff):
        candidate = handoff.model_dump(mode="json", warnings=False)
    elif isinstance(handoff, dict):
        candidate = handoff
    else:
        raise HandoffReplayIntegrityError("handoff must be an object")
    try:
        validated = load_handoff_json(candidate)
    except Exception as exc:
        raise HandoffReplayIntegrityError("handoff contract validation failed") from exc
    payload = validated.model_dump(mode="json", warnings=False)
    encoded = _canonical_json(payload)
    return validated, payload, encoded


@dataclass(frozen=True)
class CapturedHandoff:
    capture_id: str
    captured_at: str
    handoff_sha256: str
    handoff: QueryPlanHandoff
    canonical_build_id: str | None = None


class HandoffReplayStore:
    """Append-only, rotated JSONL store protected by an inter-process lock.

    ``max_files`` counts the active segment plus rotated segments.  Every
    record is written while holding ``flock`` and the file descriptor uses
    ``O_APPEND``; a partial write is completed before releasing the lock.
    """

    def __init__(
            self, directory: Path, *,
            max_record_bytes: int = 256 * 1024,
            max_file_bytes: int = 8 * 1024 * 1024,
            max_files: int = 4,
            fsync: bool = True,
            ) -> None:
        self.directory = Path(directory)
        self.max_record_bytes = int(max_record_bytes)
        self.max_file_bytes = int(max_file_bytes)
        self.max_files = int(max_files)
        self.fsync = bool(fsync)
        if self.max_record_bytes < 1024:
            raise ValueError("max_record_bytes must be at least 1024")
        if self.max_file_bytes < self.max_record_bytes:
            raise ValueError("max_file_bytes must cover one maximum record")
        if not 1 <= self.max_files <= 32:
            raise ValueError("max_files must be between 1 and 32")

    @property
    def active_path(self) -> Path:
        return self.directory / CAPTURE_FILE

    @property
    def lock_path(self) -> Path:
        return self.directory / f".{CAPTURE_FILE}.lock"

    def _prepare_directory(self) -> None:
        if self.directory.is_symlink():
            raise HandoffReplayIntegrityError("capture directory cannot be a symlink")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.directory, 0o700)

    def _rotate(self) -> None:
        if self.max_files == 1:
            try:
                self.active_path.unlink()
            except FileNotFoundError:
                pass
            return

        oldest = self.directory / f"{CAPTURE_FILE}.{self.max_files - 1}"
        try:
            oldest.unlink()
        except FileNotFoundError:
            pass
        for index in range(self.max_files - 2, 0, -1):
            source = self.directory / f"{CAPTURE_FILE}.{index}"
            target = self.directory / f"{CAPTURE_FILE}.{index + 1}"
            try:
                os.replace(source, target)
            except FileNotFoundError:
                pass
        try:
            os.replace(self.active_path, self.directory / f"{CAPTURE_FILE}.1")
        except FileNotFoundError:
            pass

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short append to handoff replay store")
            view = view[written:]

    def capture(
            self, handoff: QueryPlanHandoff | dict, *,
            canonical_build_id: str | None = None,
            ) -> CapturedHandoff:
        validated, payload, handoff_bytes = _validated_handoff_payload(handoff)
        digest = hashlib.sha256(handoff_bytes).hexdigest()
        capture_id = str(uuid.uuid4())
        captured_at = datetime.now(timezone.utc).isoformat()
        record: dict[str, object] = {
            "capture_schema": CAPTURE_SCHEMA,
            "capture_id": capture_id,
            "captured_at": captured_at,
            "handoff_sha256": digest,
            "handoff": payload,
        }
        if canonical_build_id is not None:
            if (not isinstance(canonical_build_id, str)
                    or not _BUILD_ID_RE.fullmatch(canonical_build_id)):
                raise HandoffReplayIntegrityError("canonical_build_id must be non-blank")
            record["canonical_build_id"] = canonical_build_id
        line = _canonical_json(record) + b"\n"
        if len(line) > self.max_record_bytes:
            raise HandoffReplayLimitError(
                f"capture record exceeds {self.max_record_bytes} bytes")

        self._prepare_directory()
        lock_fd = os.open(
            self.lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.fchmod(lock_fd, 0o600)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                try:
                    current_size = self.active_path.stat().st_size
                except FileNotFoundError:
                    current_size = 0
                if current_size + len(line) > self.max_file_bytes:
                    self._rotate()
                flags = (
                    os.O_CREAT | os.O_APPEND | os.O_WRONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                fd = os.open(self.active_path, flags, 0o600)
                try:
                    os.fchmod(fd, 0o600)
                    self._write_all(fd, line)
                    if self.fsync:
                        os.fsync(fd)
                finally:
                    os.close(fd)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

        return CapturedHandoff(
            capture_id=capture_id,
            captured_at=captured_at,
            handoff_sha256=digest,
            handoff=validated,
            canonical_build_id=canonical_build_id,
        )


def _capture_paths(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    paths = [path / CAPTURE_FILE]
    paths.extend(path / f"{CAPTURE_FILE}.{i}" for i in range(1, 33))
    return [candidate for candidate in paths if candidate.is_file()]


def _bounded_lines(path: Path, *, max_record_bytes: int) -> Iterator[bytes]:
    with open(path, "rb") as stream:
        while True:
            line = stream.readline(max_record_bytes + 1)
            if not line:
                return
            if len(line) > max_record_bytes or not line.endswith(b"\n"):
                raise HandoffReplayLimitError(
                    f"oversized or incomplete record in {path.name}")
            yield line[:-1]


def _validate_record(record: object) -> CapturedHandoff:
    if not isinstance(record, dict):
        raise HandoffReplayIntegrityError("capture record must be an object")
    keys = set(record)
    if not _REQUIRED_KEYS <= keys or keys - (_REQUIRED_KEYS | _OPTIONAL_KEYS):
        raise HandoffReplayIntegrityError("capture record keys do not match schema")
    if record["capture_schema"] != CAPTURE_SCHEMA:
        raise HandoffReplayIntegrityError("unsupported capture schema")
    try:
        capture_id = str(uuid.UUID(record["capture_id"]))
    except (TypeError, ValueError, AttributeError) as exc:
        raise HandoffReplayIntegrityError("invalid capture_id") from exc
    if capture_id != record["capture_id"]:
        raise HandoffReplayIntegrityError("capture_id is not canonical")
    captured_at = record["captured_at"]
    if not isinstance(captured_at, str):
        raise HandoffReplayIntegrityError("captured_at must be a string")
    try:
        parsed_at = datetime.fromisoformat(captured_at)
    except ValueError as exc:
        raise HandoffReplayIntegrityError("invalid captured_at") from exc
    if parsed_at.tzinfo is None:
        raise HandoffReplayIntegrityError("captured_at must include timezone")
    digest = record["handoff_sha256"]
    if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
        raise HandoffReplayIntegrityError("invalid handoff_sha256")
    handoff_value = record["handoff"]
    if not isinstance(handoff_value, dict):
        raise HandoffReplayIntegrityError("handoff must be an object")
    validated, normalized, handoff_bytes = _validated_handoff_payload(handoff_value)
    if _canonical_json(normalized) != _canonical_json(handoff_value):
        raise HandoffReplayIntegrityError("handoff JSON is not canonical contract output")
    actual = hashlib.sha256(handoff_bytes).hexdigest()
    if not hmac.compare_digest(actual, digest):
        raise HandoffReplayIntegrityError("handoff digest mismatch")
    build_id = record.get("canonical_build_id")
    if (build_id is not None
            and (not isinstance(build_id, str)
                 or not _BUILD_ID_RE.fullmatch(build_id))):
        raise HandoffReplayIntegrityError("invalid canonical_build_id")
    return CapturedHandoff(
        capture_id=capture_id,
        captured_at=captured_at,
        handoff_sha256=digest,
        handoff=validated,
        canonical_build_id=build_id,
    )


def load_capture(
        path: Path, *, capture_id: str | None = None,
        handoff_sha256: str | None = None,
        expected_canonical_build_id: str | None = None,
        max_record_bytes: int = 256 * 1024,
        max_file_bytes: int = 8 * 1024 * 1024,
        ) -> CapturedHandoff:
    """Select exactly one validated record from a segment or store directory."""

    if (capture_id is None) == (handoff_sha256 is None):
        raise ValueError("select exactly one of capture_id or handoff_sha256")
    if not 1024 <= max_record_bytes <= 1024 * 1024:
        raise ValueError("max_record_bytes must be between 1024 and 1048576")
    if not max_record_bytes <= max_file_bytes <= 64 * 1024 * 1024:
        raise ValueError("max_file_bytes must cover records and be at most 67108864")
    matches: list[CapturedHandoff] = []
    for capture_path in _capture_paths(Path(path)):
        if capture_path.stat().st_size > max_file_bytes:
            raise HandoffReplayLimitError(
                f"capture segment exceeds {max_file_bytes} bytes")
        for raw in _bounded_lines(capture_path, max_record_bytes=max_record_bytes):
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HandoffReplayIntegrityError(
                    f"invalid JSON in {capture_path.name}") from exc
            captured = _validate_record(value)
            if capture_id is not None and captured.capture_id == capture_id:
                matches.append(captured)
            if handoff_sha256 is not None and captured.handoff_sha256 == handoff_sha256:
                matches.append(captured)
    if not matches:
        raise HandoffReplayIntegrityError("selected capture was not found")
    if len(matches) != 1:
        raise HandoffReplayIntegrityError("selected capture is not unique")
    selected = matches[0]
    if (expected_canonical_build_id is not None
            and selected.canonical_build_id != expected_canonical_build_id):
        raise HandoffReplayIntegrityError("canonical build id mismatch")
    return selected


__all__ = [
    "CAPTURE_FILE", "CAPTURE_SCHEMA", "CapturedHandoff",
    "HandoffReplayError", "HandoffReplayIntegrityError",
    "HandoffReplayLimitError", "HandoffReplayStore", "load_capture",
]
