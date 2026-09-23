#!/usr/bin/env python3
"""정본 Parquet 을 ``evidence_id`` 로 정렬해 다시 쓴다. **행은 하나도 바뀌지 않는다.**

왜 필요한가. ``evidence_id`` 는 해시라 자연 순서가 없고, 빌드는 문서 순서대로
스트리밍 기록한다. 그 결과 모든 row group 의 min/max 가 ``000…``~``fff…`` 전 범위를
덮어 **``evidence_id`` 필터가 row group 을 하나도 잘라내지 못한다.** 근거 FK 검증이
조회마다 약 1.25GB 를 훑고, ``lookup()`` 한 번이 2.0~2.7초(페이지 캐시가 차갑면
5.6초)가 된다.

정렬하면 row group 통계가 서로 겹치지 않아 pruning 이 작동한다. 실측:

```
evidence   1.064s → 0.022s   48배      524MB → 748MB
chunks     1.364s → 0.016s   84배      601MB → 816MB
fields     0.192s → 0.015s   13배       90MB → 155MB
facts      0.081s → 0.007s   12배       40MB →  73MB
합계       2.701s → 0.060s   45배    1,255MB → 1,792MB (+43%)
```

용량이 커지는 것은 해시 정렬이 문서 순서의 압축 국소성을 깨기 때문이다. 조회를
45배 줄이는 값으로 받는다.

**무결성 검사는 하나도 약화되지 않는다.** 대안으로 읽기 경로에 ``doc_id`` 필터를
넣는 방법(15배, 재적재 불필요)이 있었는데, 그러면 ``doc_id`` 가 어긋난 semantic 행이
필터에서 걸러져 「SourceFile ownership 불일치」 진단이 「orphan」으로 뭉개진다.
그래서 데이터를 고치는 쪽을 골랐다.

안전 장치:

* 원본을 백업한 뒤에만 교체한다 (``--backup-dir``).
* 새 파일의 내용이 원본과 **같은지 확인한 뒤에만** 교체한다 — 같은 키로 정렬해
  전체 테이블을 비교한다. 행 수·스키마도 함께 본다.
* ``run.json`` 의 ``artifact_hashes`` 를 같은 실행에서 갱신한다. 빼먹으면
  ``ArtifactIntegrityError: hash_mismatch`` 로 정본이 열리지 않는다.
* ``--dry-run`` 은 아무것도 쓰지 않고 계획만 보여준다.

``correction_items`` 는 대상이 아니다 — ``evidence_id`` 열이 없고(``before_``/
``after_``) 0.4MB·row group 2개라 스캔이 0.001초다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent.parent

#: 조회 경로가 ``evidence_id`` 로 필터하는 산출물. 비용이 큰 순서다.
TARGETS = ("evidence", "chunks", "fields", "facts")

SORT_KEY = "evidence_id"


def file_sha256_32(path: Path) -> str:
    """``src/artifact.py`` 의 ``file_sha256`` 과 **같은 값**이어야 한다 — 앞 32자다."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:32]


def megabytes(path: Path) -> float:
    return path.stat().st_size / 1048576


def repack_one(path: Path, *, dry_run: bool) -> dict:
    """하나를 정렬해 옆에 쓰고 내용 동일성을 확인한다. 교체는 하지 않는다."""

    source = pq.ParquetFile(path)
    if SORT_KEY not in source.schema_arrow.names:
        raise SystemExit(f"{path.name}: {SORT_KEY} 열이 없습니다")
    rows = source.metadata.num_rows
    groups = source.metadata.num_row_groups
    # 원본의 row group 크기를 유지한다. 크기를 바꾸면 pruning 효과와 메모리
    # 사용량이 함께 달라져 이 변경의 효과를 귀속할 수 없다.
    group_size = max(1, rows // groups) if groups else 2000
    report = {
        "name": path.stem, "rows": rows, "row_groups": groups,
        "group_size": group_size, "mb_before": megabytes(path),
    }
    if dry_run:
        return report

    started = time.time()
    table = pq.read_table(path)
    if table.num_rows != rows:
        raise SystemExit(f"{path.name}: 읽은 행 수가 metadata 와 다릅니다")
    sorted_table = table.sort_by([(SORT_KEY, "ascending")])
    del table

    target = path.with_suffix(".parquet.repack")
    pq.write_table(sorted_table, target, row_group_size=group_size,
                   compression="zstd")
    report["seconds"] = time.time() - started
    report["mb_after"] = megabytes(target)

    # **내용이 같은지 확인한 뒤에만 교체한다.** 정렬만 했으므로 같은 키로 정렬한
    # 테이블끼리는 완전히 같아야 한다. 여기서 통과하지 못하면 원본을 건드리지 않는다.
    written = pq.read_table(target)
    if written.num_rows != rows:
        raise SystemExit(f"{path.name}: 행 수 불일치 {written.num_rows} != {rows}")
    if written.schema.names != sorted_table.schema.names:
        raise SystemExit(f"{path.name}: 열 구성이 달라졌습니다")
    if not written.equals(sorted_table):
        raise SystemExit(f"{path.name}: 다시 읽은 내용이 정렬 결과와 다릅니다")
    del sorted_table, written

    # row group 통계가 실제로 겹치지 않는지 본다 — 이게 목적이다.
    packed = pq.ParquetFile(target)
    index = packed.schema_arrow.names.index(SORT_KEY)
    previous = None
    for group in range(packed.metadata.num_row_groups):
        stats = packed.metadata.row_group(group).column(index).statistics
        if stats is None or stats.min is None:
            raise SystemExit(f"{path.name}: row group 통계가 없습니다")
        if previous is not None and stats.min < previous:
            raise SystemExit(f"{path.name}: 정렬 후에도 row group 이 겹칩니다")
        previous = stats.max
    report["pruning"] = True
    report["target"] = target
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="정본 Parquet 을 evidence_id 로 정렬해 재기록")
    parser.add_argument("--canonical", type=Path, default=ROOT / "out/canonical")
    parser.add_argument("--backup-dir", type=Path, required=True,
                        help="원본을 옮겨 둘 디렉터리. 교체 전에 반드시 채운다.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    canonical = args.canonical
    run_path = canonical / "run.json"
    if not run_path.is_file():
        raise SystemExit(f"run.json 이 없습니다: {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    hashes = run.get("artifact_hashes")
    if not isinstance(hashes, dict):
        raise SystemExit("run.json 에 artifact_hashes object 가 없습니다")

    paths = [canonical / f"{name}.parquet" for name in TARGETS]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"산출물이 없습니다: {path}")
        # 지금 기록된 해시가 맞는지 먼저 본다. 이미 어긋나 있으면 이 스크립트의
        # 갱신이 그 사실을 덮어 버린다.
        current = file_sha256_32(path)
        declared = hashes.get(path.stem)
        if declared != current:
            raise SystemExit(
                f"{path.name}: run.json 해시가 이미 어긋나 있습니다 "
                f"({declared} != {current}). 먼저 원인을 확인하십시오.")

    print(f"  대상 {len(paths)}개 · 정렬 키 {SORT_KEY}"
          f"{' · dry-run' if args.dry_run else ''}\n")
    reports = [repack_one(path, dry_run=args.dry_run) for path in paths]

    if args.dry_run:
        for report in reports:
            print(f"  {report['name']:<12}{report['rows']:>10,}행  "
                  f"row group {report['row_groups']:>5} × {report['group_size']}행  "
                  f"{report['mb_before']:>7.1f} MB")
        print("\n  아무것도 쓰지 않았습니다.")
        return 0

    # 여기까지 왔으면 새 파일이 전부 검증을 통과했다. 원본을 백업한 뒤 교체한다.
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, args.backup_dir / path.name)
    shutil.copy2(run_path, args.backup_dir / run_path.name)
    print(f"  원본 백업 → {args.backup_dir}\n")

    for report in reports:
        path = canonical / f"{report['name']}.parquet"
        os.replace(report["target"], path)
        digest = file_sha256_32(path)
        hashes[report["name"]] = digest
        print(f"  {report['name']:<12}{report['mb_before']:>7.1f} → "
              f"{report['mb_after']:>7.1f} MB "
              f"({report['mb_after'] / report['mb_before'] - 1:+.0%})  "
              f"{report['seconds']:>5.1f}s  {digest}")

    run_path.write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n  run.json artifact_hashes 갱신 ({len(reports)}개)")

    before = sum(r["mb_before"] for r in reports)
    after = sum(r["mb_after"] for r in reports)
    print(f"  용량 {before:,.0f} → {after:,.0f} MB ({after / before - 1:+.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
