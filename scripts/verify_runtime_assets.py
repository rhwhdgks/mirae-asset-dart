#!/usr/bin/env python3
"""Verify the canonical snapshot and its one matching serving search index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.serving.index import SearchIndex
from src.canonical.read import AS_OF_ALL, CanonicalReadModel


def _matching_index(canonical: CanonicalReadModel, serving: Path) -> Path:
    chunks_hash = canonical.run["artifact_hashes"]["chunks"]
    matches: list[Path] = []
    for manifest_path in sorted(serving.glob("*/manifest.json")):
        root = manifest_path.parent
        if root.name.startswith(".") or root.is_symlink():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if (
                manifest.get("canonical_build_id") == canonical.build_id
                and manifest.get("chunks_hash") == chunks_hash
        ):
            matches.append(root)
    if len(matches) != 1:
        names = [path.name for path in matches]
        raise RuntimeError(
            "현재 canonical과 일치하는 search index는 정확히 하나여야 합니다: "
            f"{names}"
        )
    return matches[0]


def verify(canonical_root: Path, serving_root: Path, *, as_of: str) -> dict:
    canonical = CanonicalReadModel(canonical_root)
    index_root = _matching_index(canonical, serving_root)
    index = SearchIndex(
        index_root,
        expected_canonical_build_id=canonical.build_id,
    )
    hits = index.search("사업 내용", as_of=as_of, top_k=1)
    if len(hits) != 1:
        raise RuntimeError("대표 검색 결과가 정확히 한 건이 아닙니다")
    evidence = canonical.get_evidence(hits[0].evidence_id)
    if evidence is None or evidence.kind != "chunk_text":
        raise RuntimeError("대표 검색 결과의 Chunk Evidence 검증에 실패했습니다")

    company = canonical.resolve_company("삼성전자")
    if len(company) != 1:
        raise RuntimeError("삼성전자 회사 해석 결과가 1건이 아닙니다")
    result = canonical.lookup(
        company[0].corp_code,
        "revenue",
        "2025-12-31",
        as_of=AS_OF_ALL,
        scope="CFS",
        statement="IS",
        cumulative=True,
    )
    if result.status != "ok" or result.selected is None:
        raise RuntimeError("대표 재무 Fact 조회에 실패했습니다")
    fact_evidence = canonical.get_evidence(result.selected.citation)
    if fact_evidence is None or fact_evidence.kind != "fact_value":
        raise RuntimeError("대표 재무 Fact Evidence 검증에 실패했습니다")

    return {
        "status": "ok",
        "canonical_build_id": canonical.build_id,
        "canonical_schema_version": canonical.schema_version,
        "search_index_build_id": index.index_build_id,
        "search_rows": index.manifest["meta_rows"],
        "search_fts_rows": index.manifest["fts_rows"],
        "representative_search_evidence_id": hits[0].evidence_id,
        "representative_fact_evidence_id": result.selected.citation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=Path("out/canonical"))
    parser.add_argument("--serving", type=Path, default=Path("out/serving"))
    parser.add_argument("--as-of", default="20260619")
    args = parser.parse_args()
    try:
        result = verify(args.canonical, args.serving, as_of=args.as_of)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"RUNTIME_ASSET_CHECK_FAIL: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
