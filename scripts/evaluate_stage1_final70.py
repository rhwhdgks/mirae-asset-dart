#!/usr/bin/env python3
"""정본 70문항을 한 번에 채점한다: HCX 실호출 → QueryPlanHandoff 0.4 → final 대조.

질문도 정답도 ``fixtures/query_plan_v04_final/`` 하나에서만 읽는다.  base archive
(``fixtures/query_plan_v04/``)는 비교·복구용 snapshot 이라 평가 입력이 아니다.
두 파일 모두 그 release 자신의 ``release_manifest.json`` 이 고정한 바이트인지
호출 전에 확인하고, 어긋나면 provider 를 한 번도 부르지 않고 멈춘다.

분할 실행을 acceptance에 쓰지 않는다. 한 번에 돌면 prompt·schema·generation
config·questions binding 이 구성상 같으므로, 실행을 나중에 이어 붙일 때 필요한
binding drift 검사가 애초에 성립하지 않는다.

순서가 계약이다. 후보를 전부 디스크에 봉인한 **뒤에야** final expected handoff를
연다. 생성 단계는 expected handoff를 읽지 않는다.

    PYTHONPATH=. .venv/bin/python scripts/evaluate_stage1_final70.py --live
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from hashlib import sha256
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RELEASE_ROOT = ROOT / "fixtures/query_plan_v04_final"
QUESTIONS_PATH = RELEASE_ROOT / "questions_v0.4.jsonl"
EXPECTED_PATH = RELEASE_ROOT / "query_plan_handoffs_v0.4.jsonl"
RELEASE_MANIFEST_PATH = RELEASE_ROOT / "release_manifest.json"
MAX_CANDIDATE_BYTES = 16 * 1024 * 1024


class Final70EvalError(ValueError):
    """정본 입력이나 한 번에 도는 70문항 평가가 성립하지 않는다."""


def verify_release_inputs() -> dict[str, str]:
    """질문·정답이 release manifest 가 고정한 바이트인지 확인한다."""

    if RELEASE_MANIFEST_PATH.is_symlink() or not RELEASE_MANIFEST_PATH.is_file():
        raise Final70EvalError(f"release manifest가 없습니다: {RELEASE_MANIFEST_PATH}")
    manifest = json.loads(RELEASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    pinned = manifest.get("release_fixture_digests")
    if not isinstance(pinned, dict):
        raise Final70EvalError("release manifest에 release_fixture_digests가 없습니다")

    digests: dict[str, str] = {}
    for path in (QUESTIONS_PATH, EXPECTED_PATH):
        if path.is_symlink() or not path.is_file():
            raise Final70EvalError(f"정본 파일이 없습니다: {path}")
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != pinned.get(path.name):
            raise Final70EvalError(
                f"정본 {path.name} 이 release manifest 고정값과 다릅니다")
        digests[path.name] = actual
    return digests


def verify_live_question_input() -> str:
    """Verify the question input without opening the final expected handoffs.

    The release manifest commits to both files, but live candidate generation
    needs only questions.  The expected handoffs are byte-verified immediately
    before post-seal scoring in :func:`main`.
    """

    if RELEASE_MANIFEST_PATH.is_symlink() or not RELEASE_MANIFEST_PATH.is_file():
        raise Final70EvalError(f"release manifest가 없습니다: {RELEASE_MANIFEST_PATH}")
    manifest = json.loads(RELEASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    pinned = manifest.get("release_fixture_digests")
    if not isinstance(pinned, dict):
        raise Final70EvalError("release manifest에 release_fixture_digests가 없습니다")
    if QUESTIONS_PATH.is_symlink() or not QUESTIONS_PATH.is_file():
        raise Final70EvalError(f"정본 파일이 없습니다: {QUESTIONS_PATH}")
    actual = sha256(QUESTIONS_PATH.read_bytes()).hexdigest()
    if actual != pinned.get(QUESTIONS_PATH.name):
        raise Final70EvalError(
            f"정본 {QUESTIONS_PATH.name} 이 release manifest 고정값과 다릅니다")
    return actual


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _question_ids_from_final_release() -> tuple[str, ...]:
    """Read only question identities from the already pinned final release."""

    payload = QUESTIONS_PATH.read_bytes()
    try:
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Final70EvalError("final question rows를 읽을 수 없습니다") from exc
    ids = tuple(row.get("question_id") if isinstance(row, dict) else None
                for row in rows)
    if len(ids) != 70 or any(not isinstance(value, str) or not value for value in ids):
        raise Final70EvalError("final question ID inventory가 정확히 70개가 아닙니다")
    if len(ids) != len(set(ids)):
        raise Final70EvalError("final question ID가 중복되었습니다")
    return ids  # type: ignore[return-value]


def _read_candidate_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise Final70EvalError("candidate JSONL은 symlink가 아닌 일반 파일이어야 합니다")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise Final70EvalError("candidate JSONL을 읽을 수 없습니다") from exc
    if (
            not payload or len(payload) > MAX_CANDIDATE_BYTES
            or payload.startswith(b"\xef\xbb\xbf") or b"\r" in payload
            or not payload.endswith(b"\n")):
        raise Final70EvalError("candidate JSONL byte 계약이 잘못되었습니다")
    return payload


def score_sealed_candidates(
        candidate_path: Path, output_dir: Path) -> dict[str, Any]:
    """Score an already sealed 70-row candidate file without any planner work.

    This path is intentionally limited to JSONL parsing and final-fixture dual
    scoring.  It never imports a provider, canonical read model, or resolver.
    """

    if output_dir.is_symlink() or (
            output_dir.exists() and (not output_dir.is_dir()
                                     or any(output_dir.iterdir()))):
        raise Final70EvalError("candidate-only output directory는 새 빈 경로여야 합니다")
    candidate_bytes = _read_candidate_bytes(candidate_path)
    release_digests = verify_release_inputs()

    # These scorer imports are deliberately inside the sealed-candidate path;
    # they parse QueryPlanHandoff JSON only and do not construct candidates.
    from scripts.score_query_plan_v04_dual import (
        evaluate_query_plan_v04_rows,
        load_query_plan_v04_rows,
    )

    candidates = load_query_plan_v04_rows(candidate_path)
    expected = load_query_plan_v04_rows(EXPECTED_PATH)
    question_ids = _question_ids_from_final_release()
    if len(candidates) != 70 or tuple(candidates) != question_ids:
        raise Final70EvalError(
            "candidate JSONL은 final question 순서·ID와 정확히 같은 70행이어야 합니다")
    if tuple(expected) != question_ids:
        raise Final70EvalError("final expected handoff ID inventory가 questions와 다릅니다")

    dual_score = evaluate_query_plan_v04_rows(
        expected, candidates, question_ids=question_ids)
    if output_dir.exists():
        # Exists only as an empty directory, checked above.
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir.mkdir(parents=True)
    score_body = {
        "evaluation_mode": "official_candidate_only",
        "dual_score": dual_score,
    }
    report = {
        "schema_version": "stage1-final70-candidate-only-eval/1.0",
        **score_body,
        "input_hashes": {
            "release_manifest_sha256": sha256(
                RELEASE_MANIFEST_PATH.read_bytes()).hexdigest(),
            "questions_sha256": release_digests[QUESTIONS_PATH.name],
            "expected_handoffs_sha256": release_digests[EXPECTED_PATH.name],
            "candidate_jsonl_sha256": sha256(candidate_bytes).hexdigest(),
        },
        "output_hashes": {
            "score_body_sha256": sha256(
                _canonical_json(score_body).encode("utf-8")).hexdigest(),
        },
        "provider_calls": 0,
        "canonical_reads": 0,
        "resolver_runs": 0,
    }
    report_bytes = (_canonical_json(report) + "\n").encode("utf-8")
    _atomic_write(output_dir / "report.json", report_bytes)
    receipt = {
        "schema_version": "stage1-final70-candidate-only-receipt/1.0",
        "evaluation_mode": "official_candidate_only",
        "input_hashes": report["input_hashes"],
        "output_hashes": {
            "report_json_sha256": sha256(report_bytes).hexdigest(),
        },
    }
    _atomic_write(
        output_dir / "receipt.json",
        (_canonical_json(receipt) + "\n").encode("utf-8"))
    return report


def run_live_intents(
        output_dir: Path, *, rate_limit_retries: int,
        prompt_manifest: Path | None = None) -> dict[str, Any]:
    """정본 70문항을 한 패스로 실호출해 SemanticIntent v1 행을 남긴다."""

    from agent.hcx_semantic_intent_v1 import (
        HcxSemanticIntentRunner,
        verify_hcx_semantic_intent_prompt,
    )
    from agent.hcx_semantic_intent_v1_eval import (
        load_frozen_semantic_intent_questions,
        run_live_evaluation,
    )

    prompt = verify_hcx_semantic_intent_prompt(
        require_approved=True,
        **({} if prompt_manifest is None
           else {"manifest_path": prompt_manifest}))
    questions = load_frozen_semantic_intent_questions(QUESTIONS_PATH)

    # provider 를 만드는 것은 호출 없는 검사를 전부 통과한 뒤다.
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
        return run_live_evaluation(
            runner,
            phase="all70",
            questions=questions,
            output_dir=output_dir,
            rate_limit_retries=rate_limit_retries,
        )


def verify_final70_intents(intents_dir: Path, *, questions_sha256: str) -> None:
    """Accept only one sealed all70 run as the official scoring input."""

    summary_path = intents_dir / "summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise Final70EvalError(f"all70 summary가 없습니다: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Final70EvalError("all70 summary를 읽을 수 없습니다") from exc
    if not isinstance(summary, dict) or (
            summary.get("phase") != "all70"
            or summary.get("selected_count") != 70
            or summary.get("questions_sha256") != questions_sha256):
        raise Final70EvalError(
            "공식 final70 채점 입력은 같은 release의 단일 all70 run이어야 합니다")


def score_against_final(
        intents_dir: Path, score_dir: Path, *,
        questions_sha256: str) -> dict[str, Any]:
    """봉인된 all70 의미를 QueryPlan v0.4 로 내린 뒤 final Gold와 비교한다.

    final70은 Stage1 QueryPlan 평가다.  v1 native evidence backend는 답변 근거를
    결속하려고 ``fields.parquet`` 문서 후보를 탐색하므로 이 경로에 넣지 않는다.
    질문에서 얻은 SemanticIntent를 generic v0.4 planner로 내리는 동안 필요한
    selector preflight만 허용하고, 봉인 후에만 final Gold를 연다.
    """

    verify_final70_intents(intents_dir, questions_sha256=questions_sha256)

    from scripts.evaluate_stage1_v1_live_query_plan_v04 import evaluate

    return evaluate(
        run_dirs=(intents_dir,),
        questions_path=QUESTIONS_PATH,
        expected_path=EXPECTED_PATH,
        output_dir=score_dir,
        evaluation_mode="official",
        # 브리지 없이 v1 resolver/emitter 만 쓴다. 네이티브가 70문항을 덮은 뒤로
        # 브리지는 점수를 지키는 보험이 아니라 **결과를 흐리는 경로**다.
        native_only=True,
    )


def render_path_summary(report: dict[str, Any]) -> str:
    """어느 경로가 후보를 냈고, 그중 몇이 맞았는지.

    커버리지와 정확도를 한 줄로 합치지 않는다.  합치면 「확정 못 하면 물러난다」는
    설계가 손해처럼 보이고, 지어내는 쪽이 좋아 보인다.
    """

    native = report.get("native_candidate_count", 0)
    matched = report.get("native_gold_match_count", 0)
    closed = report.get("closed_composite_candidate_count", 0)
    return (
        f"후보 경로: v1 네이티브 {native} (final 일치 {matched})"
        f" · 닫힌 복합 QueryPlan {closed}"
        f" · 호환 브리지 {report.get('bridge_candidate_count', 0)}"
        f" · 실패 {report['candidate_error_count']}")


def render_table(report: dict[str, Any]) -> str:
    """문항별 semantic/exact 를 사람이 읽는 표로."""

    dual = report["dual_score"]
    lines = [
        f"{'문항':<10}{'semantic':<10}{'exact':<8}주요 불일치",
        "-" * 72,
    ]
    for row in dual["rows"]:
        semantic = "통과" if row["semantic"] else "실패"
        exact = "일치" if row["exact"] else "불일치"
        codes = row["semantic_issue_codes"] or row["exact_issue_codes"]
        lines.append(
            f"{row['question_id']:<10}{semantic:<10}{exact:<8}"
            f"{', '.join(codes[:3])}")
    for question_id in dual["missing_question_ids"]:
        lines.append(f"{question_id:<10}{'후보없음':<10}{'-':<8}")
    lines.extend((
        "-" * 72,
        f"semantic {dual['semantic_count']}/{dual['selected_count']}"
        f" · exact {dual['exact_count']}/{dual['selected_count']}"
        f" · 후보 없음 {dual['missing_count']}",
    ))
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true",
        help="주지 않으면 정본 검증만 하고 provider를 부르지 않는다")
    parser.add_argument(
        "--intents-dir", type=Path,
        help="이미 돈 all70 실행을 재사용한다. 주면 실호출을 건너뛴다")
    parser.add_argument(
        "--candidate-jsonl", type=Path,
        help="봉인된 QueryPlanHandoff 70행을 resolver 없이 final과 재채점한다")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rate-limit-retries", type=int, default=4)
    parser.add_argument(
        "--prompt-manifest", type=Path,
        help="승인된 prompt manifest. 주지 않으면 런타임 기본 프롬프트를 쓴다")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.candidate_jsonl is not None:
        if args.live or args.intents_dir is not None or args.prompt_manifest is not None:
            raise SystemExit(
                "--candidate-jsonl은 --live/--intents-dir/--prompt-manifest와 함께 쓸 수 없습니다")
        if args.output_dir is None:
            raise SystemExit("candidate-only 재채점에는 새 --output-dir가 필요합니다")
        report = score_sealed_candidates(args.candidate_jsonl, args.output_dir)
        print(render_table(report))
        print(f"\n보고서: {args.output_dir / 'report.json'}")
        return 0

    if not args.live and args.intents_dir is None:
        digests = verify_release_inputs()
        print(json.dumps({
            "mode": "verify_only_no_provider_call",
            "questions_path": str(QUESTIONS_PATH.relative_to(ROOT)),
            "expected_path": str(EXPECTED_PATH.relative_to(ROOT)),
            "release_pinned_digests": digests,
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    questions_sha256 = (
        verify_live_question_input() if args.intents_dir is None
        else verify_release_inputs()[QUESTIONS_PATH.name])
    output_dir = args.output_dir or (
        ROOT / "out/evaluation" / f"stage1_final70_{date.today():%Y%m%d}")
    intents_dir = args.intents_dir or (output_dir / "intents")
    score_dir = output_dir / "score"

    if args.intents_dir is None:
        summary = run_live_intents(
            intents_dir, rate_limit_retries=args.rate_limit_retries,
            prompt_manifest=args.prompt_manifest)
        print(
            f"실호출 완료: 성공 {summary['success_count']}"
            f"/{summary['selected_count']}"
            f" · 기술 실패 {summary['technical_failure_count']}")

    # The expected handoffs have remained unopened through live candidate
    # generation.  Verify their pinned bytes only at this post-seal boundary.
    digests = verify_release_inputs()
    report = score_against_final(
        intents_dir, score_dir,
        questions_sha256=questions_sha256)
    print(render_table(report))
    print(render_path_summary(report))
    print(f"\n보고서: {score_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
