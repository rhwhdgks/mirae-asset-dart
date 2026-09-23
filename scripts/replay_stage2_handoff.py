#!/usr/bin/env python3
"""Replay Stage2~4 from one validated QueryPlanHandoff sidecar record.

The default uses the deterministic template composer and never calls HCX.
``--use-hcx`` is an explicit, potentially billable opt-in.  Raw questions and
provider wires are neither loaded nor reconstructed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.env  # noqa: E402,F401  load the same runtime configuration
from app.pipeline import AnswerPipeline  # noqa: E402
from app.tools import CanonicalToolBackend  # noqa: E402
from app.tools.canonical_env import canonical_status  # noqa: E402
from server.handoff_replay import load_capture  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capture_path", type=Path,
        help="handoffs.jsonl file or bounded store directory")
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument("--capture-id")
    selected.add_argument("--handoff-sha256")
    parser.add_argument(
        "--use-hcx", action="store_true",
        help="use HCX-005 composer (explicit external, billable call)")
    args = parser.parse_args()

    status = canonical_status()
    captured = load_capture(
        args.capture_path,
        capture_id=args.capture_id,
        handoff_sha256=args.handoff_sha256,
        expected_canonical_build_id=status["build_id"],
    )
    if captured.handoff.status != "ready":
        raise SystemExit(
            f"capture status {captured.handoff.status!r} has no Stage2 plan to replay")

    backend = CanonicalToolBackend()
    pipeline = AnswerPipeline(backend, use_hcx=args.use_hcx)
    response, payload = pipeline.run(
        captured.handoff,
        question_id=f"replay-{captured.capture_id}",
        question="",
    )
    result = {
        "capture_id": captured.capture_id,
        "handoff_sha256": captured.handoff_sha256,
        "canonical_build_id": captured.canonical_build_id,
        "composer": "hcx-005" if args.use_hcx else "template",
        "final_status": payload.final_status,
        "response": response,
    }
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
