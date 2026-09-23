#!/usr/bin/env python3
"""Offline Gold10 v1 outcome -> QueryPlanHandoff 0.4 dual evaluation.

This is a compiler/emitter compatibility check, not an HCX accuracy claim.
It builds the ten approved v1 vertical/non-ready outcomes, emits the final v0.4
handoff, and compares it with the final team-compatible v0.4 release using exact and
semantic metrics.  It performs zero provider calls.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from agent.deterministic_plan_compiler_v1 import vertical_slice_inputs
from agent.query_plan_v04_final_release import RELEASE_ID
from agent.stage1_v1_gold_expectations import MANDATORY_GOLD_QUESTION_IDS
from agent.stage1_v1_nonready import (
    G_U_002,
    G_U_002_QUESTION,
    R_A_001,
    R_A_001_QUESTION,
    expected_nonready_decision,
)
from agent.stage1_v1_outcome import Stage1Outcome, Stage1V1Orchestrator
from agent.stage1_v1_query_plan_v04_emitter import (
    EMITTER_VERSION,
    emit_stage1_v1_query_plan_v04,
)
from agent.stage1_v1_resolver import (
    ClarificationAuthority,
    ClarificationOption,
    ClarificationSlot,
    ResolvedAuthority,
    Stage1V1Resolver,
    TerminalAuthority,
    TerminalReasonBinding,
)
from scripts.score_query_plan_v04_dual import (
    DEFAULT_EXPECTED,
    evaluate_query_plan_v04_rows,
    load_query_plan_v04_rows,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    ROOT / "out/evaluation/stage1_v1_gold10_query_plan_v04_offline")
BUILD_ID = "0" * 32
RESOLVER_VERSION = "stage1-resolver/1.0"
READY_IDS = MANDATORY_GOLD_QUESTION_IDS[:8]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class StaticBackend:
    def __init__(self, authority: Any) -> None:
        self.authority = authority

    def resolve(self, **_: Any) -> Any:
        return self.authority


def _resolver(
        authority: Any,
        *,
        build_id: str = BUILD_ID,
        version: str = RESOLVER_VERSION,
        ) -> Stage1V1Resolver:
    return Stage1V1Resolver(
        StaticBackend(authority),
        canonical_build_id=build_id,
        resolver_version=version,
    )


def _ready_outcome(question_id: str) -> Stage1Outcome:
    question, intent, resolution = vertical_slice_inputs(question_id)
    return Stage1V1Orchestrator(_resolver(
        ResolvedAuthority(resolution=resolution),
        build_id=resolution.canonical_build_id,
        version=resolution.resolver_version,
    )).run(
        question_id=question_id,
        question=question,
        source_intent=intent,
    )


def _clarification_outcome() -> Stage1Outcome:
    decision = expected_nonready_decision(R_A_001)
    assert decision.clarification is not None
    slot = decision.clarification.slots[0]
    authority = ClarificationAuthority(slots=[ClarificationSlot(
        slot_id=slot.slot_id,
        role_hint="target",
        reason_code=decision.clarification.reason,
        response_kind="select_one",
        prompt="어느 재무지표를 알려드릴까요?",
        applies_to_item_ids=["item-1"],
        mention_ids=["unresolved-1"],
        options=[
            ClarificationOption(value=value, label=value)
            for value in slot.allowed_values
        ],
    )])
    return Stage1V1Orchestrator(_resolver(authority)).run(
        question_id=R_A_001,
        question=R_A_001_QUESTION,
        source_intent=decision.source_intent,
    )


def _terminal_outcome() -> Stage1Outcome:
    decision = expected_nonready_decision(G_U_002)
    authority = TerminalAuthority(reasons=[
        TerminalReasonBinding(
            code=row.reason,
            scope="items",
            item_ids=[row.item_id],
        )
        for row in decision.terminal_reasons
    ])
    return Stage1V1Orchestrator(_resolver(authority)).run(
        question_id=G_U_002,
        question=G_U_002_QUESTION,
        source_intent=decision.source_intent,
    )


def _gold10_outcomes() -> dict[str, Stage1Outcome]:
    outcomes = {
        question_id: _ready_outcome(question_id)
        for question_id in READY_IDS
    }
    outcomes[R_A_001] = _clarification_outcome()
    outcomes[G_U_002] = _terminal_outcome()
    if tuple(outcomes) != MANDATORY_GOLD_QUESTION_IDS:
        raise RuntimeError("Gold10 outcome order가 expectation과 다릅니다")
    return outcomes


def evaluate(output_dir: Path) -> dict[str, Any]:
    outcomes = _gold10_outcomes()
    emitted = {
        question_id: emit_stage1_v1_query_plan_v04(outcome)
        for question_id, outcome in outcomes.items()
    }
    candidate_rows = [
        {
            "schema_version": "stage1-v1-query-plan-v04-offline-row/1.0",
            "question_id": question_id,
            "source_status": emitted[question_id].source_status,
            "source_outcome_digest": emitted[question_id].source_outcome_digest,
            "emission_digest": emitted[question_id].emission_digest,
            "handoff": emitted[question_id].handoff.model_dump(
                mode="json", warnings=False),
        }
        for question_id in MANDATORY_GOLD_QUESTION_IDS
    ]
    candidate_payload = (
        "\n".join(_canonical_json(row) for row in candidate_rows) + "\n"
    ).encode("utf-8")
    candidate_path = output_dir / "query_plan_handoffs_v0.4.candidate.jsonl"
    _atomic_write(candidate_path, candidate_payload)

    expected = load_query_plan_v04_rows(DEFAULT_EXPECTED)
    actual = load_query_plan_v04_rows(candidate_path)
    score = evaluate_query_plan_v04_rows(
        expected,
        actual,
        question_ids=MANDATORY_GOLD_QUESTION_IDS,
    )
    report = {
        "schema_version": "stage1-v1-gold10-query-plan-v04-offline/1.0",
        "evaluation_claim": (
            "approved_v1_gold10_compiler_emitter_compatibility_not_hcx_accuracy"),
        "provider_calls": 0,
        "emitter_version": EMITTER_VERSION,
        "expected_fixture_release": RELEASE_ID,
        "runtime_candidate_rewrite_applied": False,
        "expected_handoffs_sha256": sha256(
            DEFAULT_EXPECTED.read_bytes()).hexdigest(),
        "candidate_handoffs_sha256": sha256(candidate_payload).hexdigest(),
        "dual_score": score,
    }
    report_payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(output_dir / "report.json", report_payload)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = evaluate(args.output_dir)
    print(json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
