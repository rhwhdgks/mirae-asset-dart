#!/usr/bin/env python3
"""팀원 재생성본을 공식 기술 정본과 빠르게 대조한다.

빌드 자체가 수행한 prepublish 무결성 검사를 다시 전수 반복하지 않는다. 대신 발행
아티팩트를 한 번 검증하고, 시작 시각·출력 경로를 제외한 결정적 run 지문을 전달본의
기준 run과 비교한 뒤 대표 Fact→Evidence 조회 한 건을 실행한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from src.artifact import ArtifactIntegrityError, validate_canonical_artifacts
from src.canonical.read import AS_OF_ALL, CanonicalReadModel


NONDETERMINISTIC_RUN_FIELDS = frozenset({"started_at", "output_dir"})


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON을 읽을 수 없습니다: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON 최상위가 객체가 아닙니다: {path}")
    return value


def _stable_run(run: dict[str, Any]) -> dict[str, Any]:
    stable = {key: value for key, value in run.items()
              if key not in NONDETERMINISTIC_RUN_FIELDS}
    # 같은 정책 파일이라도 sibling RC·팀원 PC에서는 절대경로 접두사가 달라진다.
    # 정책 적용 결과와 버전은 비교하되 머신 경로 자체는 지문에 넣지 않는다.
    quality = stable.get("quality")
    if isinstance(quality, dict):
        normalized_quality = dict(quality)
        normalized_quality.pop("policy_path", None)
        stable["quality"] = normalized_quality
    return stable


def verify(root: Path, baseline_path: Path) -> None:
    canonical = root / "out" / "canonical"
    run_path = canonical / "run.json"
    if not run_path.is_file():
        raise RuntimeError(f"canonical run.json이 없습니다: {run_path}")
    if not baseline_path.is_file():
        raise RuntimeError(f"전달본 기준 run.json이 없습니다: {baseline_path}")

    # count/hash/build-id/schema/startup required-column 계약을 검증한다.
    validate_canonical_artifacts(canonical)
    actual = _read_json(run_path)
    baseline = _read_json(baseline_path)
    if not baseline.get("published") or baseline.get("failures"):
        raise RuntimeError("전달본 기준 run이 published 기술 정본이 아닙니다")

    actual_stable = _stable_run(actual)
    baseline_stable = _stable_run(baseline)
    if actual_stable != baseline_stable:
        differing = sorted(
            key for key in set(actual_stable) | set(baseline_stable)
            if actual_stable.get(key) != baseline_stable.get(key))
        raise RuntimeError(
            "공식 기술 정본과 지문이 다릅니다: " + ", ".join(differing))

    # 공개 read 표면의 대표 exact lookup과 first-class Evidence를 한 번 확인한다.
    read_model = CanonicalReadModel(canonical)
    companies = read_model.resolve_company("삼성전자")
    if len(companies) != 1:
        raise RuntimeError(f"삼성전자 회사 해석 결과가 1개가 아닙니다: {len(companies)}")
    result = read_model.lookup(
        companies[0].corp_code,
        "revenue",
        "2025-12-31",
        as_of=AS_OF_ALL,
        scope="CFS",
        statement="IS",
        cumulative=True,
    )
    fact = result.selected
    if (result.status != "ok" or result.coverage_status != "complete"
            or fact is None or fact.money.value != 333_605_938
            or fact.money.unit != "백만원" or fact.citation is None):
        raise RuntimeError(
            "대표 삼성전자 2025 연결 매출 Fact/Evidence smoke가 실패했습니다")
    evidence = read_model.get_evidence(fact.citation)
    if (evidence is None or evidence.kind != "fact_value"
            or evidence.doc_id != fact.doc_id
            or evidence.source_file_id != fact.source_file_id
            or evidence.locator != fact.locator
            or evidence.extraction_status != "ok"):
        raise RuntimeError("대표 Fact의 Evidence exact recreation이 실패했습니다")

    print("HANDOFF_CHECK_PASS")
    print(f"  schema:   {actual['schema_version']}")
    print(f"  build_id: {actual['build_id']}")
    print(f"  artifacts:{len(actual['artifact_hashes']):>3}")
    print("  smoke:    삼성전자 2025 CFS revenue + verified Evidence")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--baseline", type=Path,
        default=Path("out/final_evidence/run.json"),
        help="전달 ZIP에 포함된 공식 기술 정본 run.json",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    baseline = args.baseline
    if not baseline.is_absolute():
        baseline = root / baseline
    try:
        verify(root, baseline)
    except (RuntimeError, ArtifactIntegrityError) as exc:
        print(f"HANDOFF_CHECK_FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
