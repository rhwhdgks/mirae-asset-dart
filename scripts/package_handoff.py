#!/usr/bin/env python3
"""Create the small team handoff ZIP.

Raw corpus, canonical Parquet, serving SQLite, local review logs, and caches are
deliberately excluded.  Four curated final live-response logs are included for
handoff evidence.  Every included file is listed with its size and SHA-256
inside ``PACKAGE_MANIFEST.json`` and rechecked from the finished archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT.parent / "mirae-dart-agent_20260826_handoff.zip"
ARCHIVE_ROOT = "mirae-dart-agent"
ROOT_FILES = (
    ".env.example", ".gitignore", "Makefile", "README.md", "requirements.txt",
)
TREE_RULES = {
    "agent": frozenset({".py", ".tsv", ".md", ".json", ".sha256"}),
    "app": frozenset({".py", ".tsv", ".md", ".json", ".jsonl"}),
    "server": frozenset({".py"}),
    "eval": frozenset({".py", ".json", ".jsonl"}),
    "src": frozenset({".py", ".tsv", ".json", ".jsonl"}),
    "tests": frozenset({".py", ".csv", ".json", ".jsonl", ".md"}),
    "scripts": frozenset({".py", ".sh"}),
    "docs": frozenset({".md", ".json"}),
    "fixtures": frozenset({".md", ".tsv", ".csv", ".json", ".jsonl"}),
    "hcx007_prompt_v1_review": frozenset({".md", ".json"}),
}
EXACT_FILES = (
    "data/README.md",
    "out/README.md",
    "out/final_evidence/FINAL.json",
    "out/final_evidence/run.json",
    "out/final_evidence/technical_candidate_schema19.log",
    "out/requests/final70_live_final_allclosed_20260826.jsonl",
    "out/requests/edge43_live_final_allclosed_20260826.jsonl",
    "out/requests/final_targeted2_allclosed_20260826.jsonl",
    "out/requests/final_targeted_investment_unit_20260826.jsonl",
)
EXCLUDED_DIRS = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov",
})
EXCLUDED_NAMES = frozenset({
    "HCX-007 Planner 구조 검수 보고서 v1.0.docx",
    "미래에셋 과제자료.pdf",
})
EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})
ZIP_TIMESTAMP = (2026, 8, 13, 12, 0, 0)


class PackageError(RuntimeError):
    """Handoff package contract violation."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _regular_file(path: Path) -> None:
    if path.is_symlink():
        raise PackageError(f"symlink는 패키징할 수 없습니다: {path.relative_to(ROOT)}")
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except OSError as exc:
        raise PackageError(f"파일을 읽을 수 없습니다: {path}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise PackageError(f"regular file이 아닙니다: {path.relative_to(ROOT)}")


def collect_files() -> list[Path]:
    files: list[Path] = []
    for relative in (*ROOT_FILES, *EXACT_FILES):
        path = ROOT / relative
        if not path.exists():
            raise PackageError(f"필수 전달 파일이 없습니다: {relative}")
        _regular_file(path)
        files.append(path)

    for dirname, allowed_suffixes in TREE_RULES.items():
        base = ROOT / dirname
        if not base.is_dir() or base.is_symlink():
            raise PackageError(f"전달 source directory가 없습니다: {dirname}")
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(ROOT)
            if any(part in EXCLUDED_DIRS for part in relative.parts):
                continue
            if path.name in EXCLUDED_NAMES:
                if path.is_symlink() or not path.is_file():
                    raise PackageError(f"제외 대상 검수 원본이 regular file이 아닙니다: {relative}")
                continue
            if path.is_symlink():
                raise PackageError(f"source symlink 금지: {relative}")
            if path.is_dir():
                continue
            _regular_file(path)
            if path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            if (path.name != "SHA256SUMS"
                    and path.suffix.lower() not in allowed_suffixes):
                raise PackageError(f"비허용 source file 형식: {relative}")
            files.append(path)

    unique = {path.relative_to(ROOT).as_posix(): path for path in files}
    forbidden_prefixes = ("data/corpus/", "out/canonical/", "out/serving/", "review/")
    leaked = [name for name in unique if name.startswith(forbidden_prefixes)]
    if leaked:
        raise PackageError(f"대용량/검토 전용 파일이 패키지에 포함됩니다: {leaked[:3]}")
    return [unique[name] for name in sorted(unique)]


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o100644 & 0xFFFF) << 16
    return info


def build_package(output: Path) -> dict:
    files = collect_files()
    run = json.loads((ROOT / "out/final_evidence/run.json").read_text(encoding="utf-8"))
    entries = []
    payloads: list[tuple[str, bytes]] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        entries.append({"path": relative, "size": len(data), "sha256": _sha256(data)})
        payloads.append((relative, data))

    manifest = {
        "schema_version": "mirae-dart-handoff-package/2.1",
        "canonical_schema_version": run["schema_version"],
        "canonical_build_id": run["build_id"],
        "canonical_build_id_role": (
            "last_approved_release_evidence_not_local_snapshot"),
        "canonical_snapshot_included": False,
        "approved_release_evidence_included": True,
        "raw_corpus_included": False,
        "canonical_parquet_included": False,
        "serving_index_included": False,
        "serving_index_rebuild_command": "make search-index",
        "agent_vertical_slice_included": True,
        "full_agent_runtime_included": True,
        "final_live_validation_logs_included": True,
        "known_semantic_partial": "G-I-012 intentionally left unchanged",
        "difference_guide": "docs/IMPLEMENTATION.md",
        "file_count_excluding_manifest": len(entries),
        "files": entries,
    }
    manifest_data = (json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED,
                compresslevel=9, strict_timestamps=True) as archive:
            for relative, data in payloads:
                archive.writestr(_zip_info(f"{ARCHIVE_ROOT}/{relative}"), data)
            archive.writestr(
                _zip_info(f"{ARCHIVE_ROOT}/PACKAGE_MANIFEST.json"), manifest_data)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        os.chmod(output, 0o644)
        directory_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def verify_package(path: Path, manifest: dict) -> None:
    expected_root = ARCHIVE_ROOT + "/"
    with zipfile.ZipFile(path) as archive:
        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise PackageError(f"ZIP CRC 오류: {bad_crc}")
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise PackageError("ZIP member 경로가 중복됩니다")
        if any(not name.startswith(expected_root) or name.startswith("/")
               or ".." in Path(name).parts for name in names):
            raise PackageError("ZIP member 경로 계약 오류")
        if any(not (info.flag_bits & 0x800)
               for info in infos if not info.filename.isascii()):
            raise PackageError("한글 ZIP 경로의 UTF-8 flag가 없습니다")
        packaged = {
            name.removeprefix(expected_root)
            for name in names if name != expected_root + "PACKAGE_MANIFEST.json"
        }
        expected = {entry["path"] for entry in manifest["files"]}
        if packaged != expected:
            raise PackageError("ZIP member와 PACKAGE_MANIFEST가 다릅니다")
        archived_manifest = json.loads(archive.read(
            expected_root + "PACKAGE_MANIFEST.json").decode("utf-8"))
        if archived_manifest != manifest:
            raise PackageError("ZIP 내부 PACKAGE_MANIFEST 내용이 생성 manifest와 다릅니다")
        expected_entries = {entry["path"]: entry for entry in manifest["files"]}
        for relative, entry in expected_entries.items():
            payload = archive.read(expected_root + relative)
            if len(payload) != entry["size"] or _sha256(payload) != entry["sha256"]:
                raise PackageError(f"ZIP member digest 불일치: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = build_package(output)
    verify_package(output, manifest)
    print(json.dumps({
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output.read_bytes()),
        "members": manifest["file_count_excluding_manifest"] + 1,
        "canonical_build_id": manifest["canonical_build_id"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
