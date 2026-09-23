#!/usr/bin/env python3
"""Evaluate HCX-007 through the Stage1 SemanticIntent v1 shadow path.

The default is always a zero-call dry run.  Once the prompt manifest is
explicitly approved, ``all70`` runs the canonical release in a single pass;
``gold10`` and ``remaining60`` remain available as narrower slices.

The question source is the final v0.4 release, and its bytes are pinned by that
release's own manifest.  Scoring the run against Gold happens downstream — this
script only produces evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NoReturn


ROOT = Path(__file__).resolve().parent.parent


class _NoCallClient:
    def generate_json(self, *args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("dry-run client는 provider를 호출할 수 없습니다")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HCX-007 SemanticIntent v1 phased evaluation")
    parser.add_argument(
        "--phase", choices=("gold10", "remaining60", "all70"),
        default="gold10")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument(
        "--question-id", action="append", dest="question_ids",
        help="phase 안에서 지정한 문항만 입력 순서대로 평가합니다 (반복 가능)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument(
        "--rate-limit-retries", type=int, default=3,
        help="속도 제한으로 실패한 문항을 같은 요청으로 몇 번까지 다시 보낼지. "
             "transport 재전송이라 각 행은 attempts=1 을 유지합니다. 0 이면 끕니다.")
    parser.add_argument(
        "--gold10-summary", type=Path,
        help="remaining60 live 에서 **선택**. 주면 같은 prompt/schema/config/"
             "questions 로 돌린 Gold10 10/10 을 호출 전에 검증합니다. 주지 않으면 "
             "산출물에 gated=false 로 기록됩니다.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    from agent.hcx_semantic_intent_v1 import (
        HcxSemanticIntentRunner,
        verify_hcx_semantic_intent_prompt,
    )
    from agent.hcx_semantic_intent_v1_eval import (
        DEFAULT_QUESTIONS_PATH,
        build_dry_run_plan,
        SemanticIntentEvalError,
        load_frozen_semantic_intent_questions,
        require_gold10_gate,
        run_live_evaluation,
    )

    prompt_kwargs = {}
    if args.prompt_manifest is not None:
        prompt_kwargs["manifest_path"] = args.prompt_manifest
    prompt = verify_hcx_semantic_intent_prompt(
        **prompt_kwargs, require_approved=args.live)
    questions = load_frozen_semantic_intent_questions(
        args.questions or DEFAULT_QUESTIONS_PATH)

    if args.question_ids and args.max_calls is not None:
        raise SystemExit("--question-id와 --max-calls는 함께 사용할 수 없습니다")

    if not args.live:
        runner = HcxSemanticIntentRunner(_NoCallClient(), prompt=prompt)
        plan = build_dry_run_plan(
            runner, phase=args.phase, questions=questions,
            question_ids=args.question_ids)
        print(json.dumps(
            plan, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False))
        return 0

    if args.output is None:
        raise SystemExit("live evaluation에는 새 --output directory가 필요합니다")
    if args.phase != "remaining60" and args.gold10_summary is not None:
        raise SystemExit("--gold10-summary는 remaining60 phase에만 씁니다")

    # **Gold10 게이트는 선택이다** (사용자 결정 2026-08-22).
    #
    # 9차 검수 P0-EVAL-001 로 필수화했다가 되돌렸다. 그 게이트가 규칙이 먹히는지
    # 확인하는 진단까지 막았기 때문이다 — Gold10 이 2/9 인 동안은 remaining60 을
    # 한 번도 못 태우고, remaining60 에만 있는 실패를 고칠 근거를 얻을 수 없다.
    #
    # `--gold10-summary` 를 주면 예전처럼 **provider client 를 만들기 전에** 검증한다.
    # 실패하면 호출은 0회다. 주지 않으면 그대로 돌되 산출물에 `gated=false` 가 남는다.
    gold10_prerequisite = None
    if args.phase == "remaining60" and args.gold10_summary is not None:
        gate_runner = HcxSemanticIntentRunner(_NoCallClient(), prompt=prompt)
        try:
            gold10_prerequisite = require_gold10_gate(
                args.gold10_summary, runner=gate_runner, questions=questions)
        except SemanticIntentEvalError as exc:
            # 실패 이유만 한 줄로 보인다. 여기서 끝나므로 호출은 0회다.
            raise SystemExit(f"Gold10 게이트 실패 — remaining60을 돌리지 않습니다: {exc}")

    # Environment loading happens only after all no-call gates have passed.
    from dotenv import load_dotenv
    from agent.planner_preflight import CanonicalSelectorRolePreflight
    from agent.providers.hcx007 import HcxStructuredClient
    from agent.stage1_assembly import CORPUS_CUTOFF
    from src.canonical.read import CanonicalReadModel

    load_dotenv(ROOT / ".env", override=False)
    company_surface_authority = CanonicalSelectorRolePreflight(
        CanonicalReadModel(ROOT / "out/canonical"),
        corpus_cutoff=CORPUS_CUTOFF,
    )
    with HcxStructuredClient.from_env() as client:
        runner = HcxSemanticIntentRunner(
            client,
            prompt=prompt,
            company_surface_regrounder=(
                company_surface_authority.question_company_surface),
        )
        summary = run_live_evaluation(
            runner,
            phase=args.phase,
            questions=questions,
            output_dir=args.output,
            max_calls=args.max_calls,
            question_ids=args.question_ids,
            gold10_prerequisite=gold10_prerequisite,
            rate_limit_retries=args.rate_limit_retries,
        )
    print(json.dumps(
        summary, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
