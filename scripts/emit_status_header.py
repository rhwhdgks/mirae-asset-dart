#!/usr/bin/env python3
"""`docs/IMPLEMENTATION.md` 의 **현재 상태 블록을 코드·산출물에서 생성한다.**

현황은 final70 단일 계약을 표시한다. historical gold10/remaining60 산출물은
호환·원인분석 기록일 뿐 현재 acceptance의 입력이나 게이트가 아니다.

버전표(`scripts/emit_version_table.py`)와 같은 규칙을 쓴다.

    이 블록은 손으로 고치지 않는다. 소스와 산출물이 바뀌면 블록도 바뀐다.

```bash
PYTHONPATH=. .venv/bin/python scripts/emit_status_header.py          # 갱신
PYTHONPATH=. .venv/bin/python scripts/emit_status_header.py --check  # drift 판정
```
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TARGET = ROOT / "docs/IMPLEMENTATION.md"
BEGIN = "<!-- BEGIN generated: 현재 상태 (scripts/emit_status_header.py) -->"
END = "<!-- END generated: 현재 상태 -->"


def _final70_reports() -> "list[dict]":
    rows: list[dict] = []
    for path in sorted((ROOT / "out/evaluation").glob(
            "stage1_final70_*/report.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        dual = payload.get("dual_score")
        if (
            payload.get("evaluation_mode") == "official_candidate_only"
            and isinstance(dual, dict)
            and dual.get("selected_count") == 70
        ):
            rows.append({
                "dir": path.parent.name,
                "path": path,
                "report_sha256": sha256(path.read_bytes()).hexdigest(),
                **payload,
            })
    return rows


def render() -> str:
    from agent.hcx_semantic_intent_v1 import (
        HCX_SEMANTIC_INTENT_PROMPT_VERSION,
        verify_hcx_semantic_intent_prompt,
    )
    from agent.hcx_semantic_intent_v1_eval import (
        HCX_SEMANTIC_INTENT_EVAL_VERSION,
    )

    prompt = verify_hcx_semantic_intent_prompt(require_approved=False)
    reports = _final70_reports()

    lines = [
        f"- v1 프롬프트: `{HCX_SEMANTIC_INTENT_PROMPT_VERSION}`"
        f" — manifest `{prompt.prompt_version}` · 승인 `{prompt.approval_state}`",
        f"- v1 eval 계약: `{HCX_SEMANTIC_INTENT_EVAL_VERSION}`",
    ]
    if reports:
        report = reports[-1]
        dual = report["dual_score"]
        lines.extend((
            f"- 공식 acceptance: final70 candidate-only `{report['dir']}` · "
            f"semantic {dual['semantic_count']}/{dual['selected_count']} · "
            f"exact {dual['exact_count']}/{dual['selected_count']} · "
            f"missing {dual['missing_count']}",
            f"- 실행 횟수: provider {report['provider_calls']} · "
            f"canonical read {report['canonical_reads']} · "
            f"resolver {report['resolver_runs']} (채점 단계)",
            f"- 근거: `{report['path'].relative_to(ROOT)}` · "
            f"report SHA-256 `{report['report_sha256']}` · "
            f"release manifest SHA-256 `{report['input_hashes']['release_manifest_sha256']}`",
        ))
    else:
        lines.append("- 공식 acceptance: final70 candidate-only report 없음")
    lines.append(
        "- acceptance 입력은 `fixtures/query_plan_v04_final/`와 release manifest로 "
        "고정하며, historical 분할 실행은 현재 평가가 아니다.")
    lines.append(
        "- 이 블록은 `scripts/emit_status_header.py` 가 코드·산출물에서 생성한다."
        " **손으로 고치지 않는다.**")
    return "\n".join(lines)


def splice(text: str, block: str) -> str:
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{TARGET} 에 생성 블록 표시가 없습니다: {BEGIN}")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{BEGIN}\n{block}\n{END}{tail}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="쓰지 않고 최신 여부만 판정한다 (drift 면 exit 1)")
    args = ap.parse_args()

    current = TARGET.read_text(encoding="utf-8")
    updated = splice(current, render())
    if args.check:
        if current != updated:
            print(f"drift: {TARGET.relative_to(ROOT)} 가 코드·산출물과 다릅니다")
            return 1
        print(f"ok: {TARGET.relative_to(ROOT)}")
        return 0
    if current == updated:
        print(f"unchanged: {TARGET.relative_to(ROOT)}")
        return 0
    TARGET.write_text(updated, encoding="utf-8")
    print(f"wrote: {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
