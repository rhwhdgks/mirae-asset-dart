#!/usr/bin/env python3
"""정본·검색 인덱스 현황을 `out/` 산출물에서 읽어 마크다운 블록으로 낸다 — 문서의 수치는 손으로 적지 않는다.

    PYTHONPATH=. .venv/bin/python scripts/emit_build_status.py            # 마크다운 블록 출력
    PYTHONPATH=. .venv/bin/python scripts/emit_build_status.py --json     # JSON
    PYTHONPATH=. .venv/bin/python scripts/emit_build_status.py --write README.md docs/IMPLEMENTATION.md
        # 파일 안의 `<!-- BEGIN generated: scripts/emit_build_status.py -->` … `<!-- END generated: build-status -->`
        # 사이를 갱신한다 (표시가 없는 파일은 건너뛴다)

읽는 것: out/canonical/run.json (build_id·schema·counts·warnings·policy), out/serving/*/manifest.json 중 현재 build 에
결속된 검색 인덱스, out/periodic/relations_report.json (정정 계보 판정 분포·파싱 실패 집계).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- BEGIN generated: scripts/emit_build_status.py -->"
END = "<!-- END generated: build-status -->"


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def collect() -> dict:
    run = _load(ROOT / "out/canonical/run.json") or {}
    build_id = run.get("build_id")
    index = None
    for manifest_path in sorted((ROOT / "out/serving").glob("*/manifest.json")):
        manifest = _load(manifest_path) or {}
        if build_id and manifest.get("canonical_build_id") == build_id:
            index = {"index_build_id": manifest.get("index_build_id") or manifest_path.parent.name,
                     "meta_rows": manifest.get("meta_rows"), "fts_rows": manifest.get("fts_rows"),
                     "chunks_hash": manifest.get("chunks_hash")}
            break
    relations = _load(ROOT / "out/periodic/relations_report.json") or {}
    return {
        "build_id": build_id,
        "schema_version": run.get("schema_version"),
        "published": run.get("published"),
        "code_hash": run.get("code_hash"),
        "security_policy_version": run.get("security_policy_version"),
        "failures": len(run.get("failures", [])),
        "warnings": len(run.get("warnings", [])),
        "counts": run.get("counts", {}),
        "search_index": index,
        "periodic_relations": {k: relations.get(k) for k in
                               ("corrections", "relations", "status", "parse_failures", "parse_failures_by_stage")},
    }


def render(status: dict) -> str:
    counts = status["counts"]
    order = ["source_files", "documents", "sections", "chunks", "evidence", "facts", "fields", "relations",
             "correction_items", "event_identities", "event_observations"]
    rows = " · ".join(f"{name} {counts[name]:,}" for name in order if name in counts)
    index = status["search_index"] or {}
    lines = [
        "| 항목 | 값 |", "|---|---|",
        f"| canonical build | `{status['build_id']}` · schema `{status['schema_version']}` · "
        f"published {status['published']} · failures {status['failures']} · warnings {status['warnings']} |",
        f"| code_hash | `{status['code_hash']}` |",
        f"| security policy | `{status['security_policy_version']}` |",
        f"| 행 수 | {rows} |",
    ]
    if index:
        lines.append(f"| 검색 인덱스 | `out/serving/{index['index_build_id']}` — meta {index.get('meta_rows') or '?'} / "
                     f"FTS {index.get('fts_rows') or '?'} 행, canonical build 에 결속 |")
    else:
        lines.append("| 검색 인덱스 | (현재 build 에 결속된 인덱스 없음 — make search-index) |")
    rel = status["periodic_relations"]
    if rel.get("corrections") is not None:
        lines.append(f"| 정기공시 정정 계보 | 정정 {rel['corrections']}건 → 관계 {rel['relations']}개, 판정 {rel['status']}, "
                     f"파싱 실패 {rel.get('parse_failures', 0)}건 |")
    return "\n".join(lines)


def splice(text: str, block: str) -> str | None:
    if BEGIN not in text or END not in text:
        return None
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{BEGIN}\n{block}\n{END}{tail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write", nargs="*", metavar="FILE", help="생성 블록 표시가 있는 파일을 갱신")
    args = parser.parse_args(argv)
    status = collect()
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    block = render(status)
    if args.write:
        for name in args.write:
            path = ROOT / name
            updated = splice(path.read_text(encoding="utf-8"), block)
            if updated is None:
                print(f"skip (표시 없음): {name}", file=sys.stderr)
                continue
            path.write_text(updated, encoding="utf-8")
            print(f"wrote: {name}")
        return 0
    print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
