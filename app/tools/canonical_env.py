"""canonical 정본(`out/canonical`)과 Stage2 검색 인덱스(`out/serving`)를 로드한다.

- 정본: `src/canonical/read.py`의 `CanonicalReadModel` 하나만 사용한다. Parquet를 직접 읽지 않는다.
- `run.json`이 `published=true`여야 한다 (`make build-data` 산출물, 또는 승인 snapshot 배치).
- 검색 인덱스: `scripts/build_search_index.py`(app/serving)가 발행한 것 중 현재 canonical build와
  일치하는 하나를 로드한다. 불일치는 fail-closed.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DIR = ROOT / "out" / "canonical"
SERVING_DIR = ROOT / "out" / "serving"

if str(ROOT) not in sys.path:          # `src.canonical.read` 는 저장소 루트 기준 패키지
    sys.path.insert(0, str(ROOT))


class CanonicalNotBuilt(RuntimeError):
    pass


def canonical_status() -> dict:
    run = CANONICAL_DIR / "run.json"
    if not run.exists():
        raise CanonicalNotBuilt(f"{run} 없음 — make build-data 또는 승인 snapshot 배치 필요")
    r = json.loads(run.read_text(encoding="utf-8"))
    if not r.get("published"):
        raise CanonicalNotBuilt(f"canonical published=false (failures={r.get('failures')})")
    return {"build_id": r["build_id"], "schema_version": r["schema_version"],
            "counts": r.get("counts", {}),
            "chunks_hash": (r.get("artifact_hashes") or {}).get("chunks"),
            "security_policy_version": r.get("security_policy_version"),
            "evidence_policy_version": (r.get("config") or {}).get("evidence_id_policy")}


@lru_cache(maxsize=1)
def read_model():
    canonical_status()
    from src.canonical.read import CanonicalReadModel  # noqa: E402
    return CanonicalReadModel(str(CANONICAL_DIR))


@lru_cache(maxsize=1)
def search_index():
    """`out/serving/*/manifest.json` 중 현재 canonical build 에 결속된 인덱스 하나를 연다. 불일치는 fail-closed."""
    from app.serving.index import SearchIndex

    st = canonical_status()
    rejected: list[str] = []
    for path in sorted(p for p in SERVING_DIR.glob("*") if (p / "manifest.json").exists()):
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            rejected.append(f"{path.name}: manifest {type(exc).__name__}")
            continue
        if not (isinstance(manifest, dict) and "db_sha256" in manifest and "meta_rows" in manifest):
            rejected.append(f"{path.name}: unknown manifest family")
            continue
        try:
            return SearchIndex(path, expected_canonical_build_id=st["build_id"])
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            rejected.append(f"{path.name}: {type(exc).__name__}: {str(exc)[:160]}")
    detail = f"; rejected={rejected}" if rejected else ""
    raise CanonicalNotBuilt(
        "현재 canonical과 검증된 검색 인덱스 없음 — make search-index 실행 필요" + detail)
