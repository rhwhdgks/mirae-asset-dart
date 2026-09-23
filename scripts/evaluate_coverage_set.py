#!/usr/bin/env python3
"""커버리지 세트를 **v1 네이티브 경로**로 돌린다.

Gold 채점기는 정확히 70문항에 고정돼 있다(개수·digest·순서). 그 잠금은 정본
무결성 장치이므로 풀지 않고, 커버리지 세트는 이 러너로 돈다.

재는 것은 **기업이 바뀌어도 되는가**다. 오늘까지의 규칙들(음차 사전, 라벨 맞춤,
상대 판별, 절 선택)은 Gold 9개 기업 데이터로만 검증됐다.

    PYTHONPATH=. .venv/bin/python scripts/evaluate_coverage_set.py --live \\
      --questions fixtures/coverage_set_v01/questions_company_v02.jsonl \\
      --facts fixtures/coverage_set_v01/answer_facts_company_v02.jsonl \\
      --output out/evaluation/coverage_company_v02
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REFERENCE = date(2026, 6, 19)
CUTOFF = "20260619"


def _load(path: Path) -> "list[dict]":
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _failure_payload(question_id: str, error: Exception) -> dict:
    """Persist only typed, content-free normalization diagnostics.

    ``HcxSemanticIntentNormalizationInvocationError`` deliberately carries a
    safe structural shape rather than the provider wire.  Coverage runs used
    to throw that information away and leave only ``str(error)`` behind,
    which makes a grounding failure impossible to diagnose or reproduce
    offline.  Keep the ordinary short error for compatibility, but never add
    a provider payload (or any free-text provider field) to this artifact.
    """

    payload: dict = {
        "question_id": question_id,
        "status": "failure",
        "error": f"{type(error).__name__}: {error}"[:200],
    }
    from agent.hcx_semantic_intent_v1 import (
        HcxSemanticIntentNormalizationInvocationError,
    )

    if isinstance(error, HcxSemanticIntentNormalizationInvocationError):
        payload["failure"] = {
            "layer": "normalization",
            "code": "grounding_rejected",
            "diagnostic_codes": list(error.diagnostic_codes),
            "diagnostic_paths": list(error.diagnostic_paths),
            "diagnostic_shape": error.diagnostic_shape.as_dict(),
            "boundary_version": error.boundary_version,
            "schema_repair_codes": list(error.schema_repair_codes),
            "normalization_codes": list(error.normalization_codes),
        }
    return payload


def run_live(questions: "list[dict]", output: Path) -> "dict[str, dict]":
    """질문마다 한 번 호출하고 행을 남긴다. 이미 있는 행은 건너뛴다."""

    from dotenv import load_dotenv
    from agent.hcx_semantic_intent_v1 import (
        HcxSemanticIntentRunner, verify_hcx_semantic_intent_prompt)
    from agent.planner_preflight import CanonicalSelectorRolePreflight
    from agent.providers.hcx007 import HcxStructuredClient
    from src.canonical.read import CanonicalReadModel

    output.mkdir(parents=True, exist_ok=True)
    prompt = verify_hcx_semantic_intent_prompt(require_approved=True)
    load_dotenv(ROOT / ".env", override=False)
    corpus = CanonicalReadModel(ROOT / "out/canonical")
    regrounder = CanonicalSelectorRolePreflight(corpus, corpus_cutoff=CUTOFF)

    rows: dict[str, dict] = {}
    with HcxStructuredClient.from_env() as client:
        runner = HcxSemanticIntentRunner(
            client, prompt=prompt,
            company_surface_regrounder=regrounder.question_company_surface)
        for row in questions:
            qid = row["question_id"]
            path = output / f"{qid}.json"
            if path.is_file():
                rows[qid] = json.loads(path.read_text(encoding="utf-8"))
                continue
            try:
                invocation = runner.invoke(row["question"])
                payload = {
                    "question_id": qid,
                    "status": "success",
                    "semantic_intent": invocation.semantic_intent.model_dump(
                        mode="json", warnings=False),
                }
            except Exception as exc:                       # noqa: BLE001
                payload = _failure_payload(qid, exc)
            path.write_text(json.dumps(payload, ensure_ascii=False,
                                       sort_keys=True), encoding="utf-8")
            rows[qid] = payload
    return rows


def score(questions, rows, facts) -> dict:
    """네이티브 경로로 계획을 만들고 **정본 문서를 짚었는지** 본다."""

    from agent.semantic_intent_v1 import SemanticIntent
    from agent.stage1_v1_backend_composition import (
        DEFAULT_BUILD_ID, DEFAULT_RESOLVER_VERSION, build_stage1_v1_backend)
    from agent.stage1_v1_outcome import Stage1V1Orchestrator
    from agent.stage1_v1_query_plan_v04_emitter import (
        emit_stage1_v1_query_plan_v04)
    from agent.stage1_v1_resolver import Stage1V1Resolver
    from src.canonical.read import CanonicalReadModel

    corpus = CanonicalReadModel(ROOT / "out/canonical")
    backend = build_stage1_v1_backend(
        corpus, reference_date=REFERENCE, corpus_cutoff=CUTOFF)
    orchestrator = Stage1V1Orchestrator(Stage1V1Resolver(
        backend, canonical_build_id=DEFAULT_BUILD_ID,
        resolver_version=DEFAULT_RESOLVER_VERSION))

    report = {"의미": 0, "확정": 0, "계획": 0, "문서적중": 0,
              "물러남": 0, "실패": [], "적중없음": []}
    for row in questions:
        qid = row["question_id"]
        saved = rows.get(qid) or {}
        if saved.get("status") != "success":
            continue
        report["의미"] += 1
        intent = SemanticIntent.model_validate(
            saved["semantic_intent"], strict=True)
        try:
            authority = backend.resolve(
                question_id=qid, question=row["question"], source_intent=intent)
        except Exception as exc:                           # noqa: BLE001
            report["실패"].append((qid, f"backend {type(exc).__name__}"))
            continue
        if authority is None:
            report["물러남"] += 1
            continue
        report["확정"] += 1
        try:
            emitted = emit_stage1_v1_query_plan_v04(orchestrator.run(
                question_id=qid, question=row["question"], source_intent=intent))
        except Exception as exc:                           # noqa: BLE001
            cause = exc.__cause__ or exc
            report["실패"].append((qid, str(cause)[:56]))
            continue
        report["계획"] += 1
        wanted = set(facts.get(qid, {}).get("required_documents") or ())
        blob = json.dumps(emitted.model_dump(mode="json"), ensure_ascii=False)
        if wanted and any(receipt in blob for receipt in wanted):
            report["문서적중"] += 1
        elif wanted:
            report["적중없음"].append(qid)
    return report


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--facts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)

    questions = _load(args.questions)
    facts = ({row["question_id"]: row for row in _load(args.facts)}
             if args.facts else {})
    if args.live:
        rows = run_live(questions, args.output)
    else:
        rows = {}
        for path in sorted(args.output.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows[payload["question_id"]] = payload

    report = score(questions, rows, facts)
    print(f"질문 {len(questions)} · 의미 {report['의미']} · 백엔드 확정 "
          f"{report['확정']} · 계획 {report['계획']} · 정본 문서 적중 "
          f"{report['문서적중']}")
    print(f"  물러남 {report['물러남']} · 실패 {len(report['실패'])}")
    for qid, why in report["실패"][:10]:
        print(f"    {qid}  {why}")
    if report["적중없음"]:
        print(f"  계획은 났으나 정본 문서를 못 짚음: {report['적중없음'][:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
