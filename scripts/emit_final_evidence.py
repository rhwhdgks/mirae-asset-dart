#!/usr/bin/env python3
"""현재 정본(out/canonical/run.json)으로 승인 증빙 `out/final_evidence/`를 재발행한다.

    PYTHONPATH=. .venv/bin/python scripts/emit_final_evidence.py --gate-log out/logs/run_all_loaded.log \
        --note "cleanup 4단계: ACODE 1차 정규화·마스킹 예외·recover 폴백 반영 재빌드"

산출물(전부 저장소 추적, `make handoff-check` 의 기준선):
- run.json                        : 정본 run.json 사본 (build_id·code_hash·counts·artifact_hashes 지문)
- FINAL.json                      : 기술 후보 태그 — 지문 요약 + 실행한 게이트 결과 + 남은 승인 항목
- technical_candidate_schema19.log: 사람이 읽는 게이트 로그

정직성 규칙: 여기 적히는 PASS 는 **이 실행에서 실제로 돌린** 게이트뿐이다(`--gate-log` 로 넘긴 run_all --loaded 출력을
그대로 인용). 암호학적 승인·독립 RC-B 재빌드처럼 하지 않은 것은 PENDING 으로 남긴다.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "out/final_evidence"
KST = timezone(timedelta(hours=9))


def _gate_lines(log_path: Path | None) -> list[str]:
    if log_path is None or not log_path.exists():
        return []
    text = re.sub(r"\x1b\[[0-9;]*m", "", log_path.read_text(encoding="utf-8", errors="replace"))
    return [line.rstrip() for line in text.splitlines() if re.match(r"^\s+(PASS|FAIL)\s", line) or line.startswith("전체:")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gate-log", type=Path, default=None, help="tests.run_all --loaded 출력 파일")
    parser.add_argument("--note", default="", help="이 후보를 만든 변경 요약")
    args = parser.parse_args()

    run = json.loads((ROOT / "out/canonical/run.json").read_text(encoding="utf-8"))
    if not run.get("published") or run.get("failures"):
        raise SystemExit("published=false 또는 failures 가 있는 정본은 증빙으로 발행하지 않는다")
    now = datetime.now(KST).replace(microsecond=0)
    gates = _gate_lines(args.gate_log)
    failed = [line for line in gates if line.strip().startswith("FAIL")]

    final = {
        "tag": f"schema-{run['schema_version']}-technical-candidate",
        "tagged_at": now.isoformat(),
        "build_id": run["build_id"],
        "schema_version": run["schema_version"],
        "published": True,
        "release_ready": False,
        "release_status": "pending_signed_approvals_and_rc_exact_compare",
        "code_hash": run.get("code_hash"),
        "manifest_hash": run.get("manifest_hash"),
        "security_policy_version": run.get("security_policy_version"),
        "evidence_id_policy": (run.get("config") or {}).get("evidence_id_policy"),
        "counts": run.get("counts", {}),
        "artifact_hashes": run.get("artifact_hashes", {}),
        "warnings_accepted": len(run.get("warnings", [])),
        "quality_policy": run.get("quality"),
        "candidate_note": args.note,
        "gates_run": {
            "source": str(args.gate_log) if args.gate_log else None,
            "passed": len([g for g in gates if g.strip().startswith("PASS")]),
            "failed": [g.strip() for g in failed],
        },
        "required_before_final_release": [
            "authorized finite Gold approval",
            "authorized account owner approval",
            "authorized correction/Core limitation approvals",
            "read-only or digest-pinned runtime",
            "signed RC-A baseline",
            "independent empty RC-B exact compare",
        ],
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (EVIDENCE / "FINAL.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = run.get("counts", {})
    log = [
        f"SCHEMA {run['schema_version']} TECHNICAL CANDIDATE — {now.isoformat()}",
        "=" * 60, "",
        f"build_id: {run['build_id']}", f"schema_version: {run['schema_version']}", f"code_hash: {run.get('code_hash')}",
        f"security_policy_version: {run.get('security_policy_version')}", "published: true",
        f"failures: {len(run.get('failures', []))}", f"warnings: {len(run.get('warnings', []))}", "release_ready: false", "",
        f"note: {args.note}" if args.note else "",
        "[CANONICAL]",
        f"PASS  {len(run.get('artifact_hashes', {}))} Parquet artifacts published atomically (run.json published=true, failures=0)",
        "PASS  " + " · ".join(f"{k}={counts[k]:,}" for k in ("source_files", "documents", "sections", "chunks", "evidence", "facts", "fields", "relations", "correction_items", "event_identities", "event_observations") if k in counts),
        "",
        "[GATES RUN IN THIS ISSUE] (tests.run_all --loaded)",
        *(gates or ["INFO  gate log not supplied"]),
        "",
        "[RELEASE]",
        "PENDING  authorized SSHSIG approvals",
        "PENDING  read-only or digest-pinned runtime",
        "PENDING  signed RC-A baseline and independent RC-B exact compare",
        "",
        "SENTINEL: TECHNICAL_CANDIDATE_NOT_FINAL_RELEASE",
    ]
    (EVIDENCE / "technical_candidate_schema19.log").write_text("\n".join(line for line in log if line is not None) + "\n", encoding="utf-8")
    print(f"issued: build {run['build_id']} → {EVIDENCE.relative_to(ROOT)} (gates PASS {final['gates_run']['passed']}, FAIL {len(failed)})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
