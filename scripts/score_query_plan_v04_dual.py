#!/usr/bin/env python3
"""Score final QueryPlanHandoff 0.4 with exact and semantic metrics.

``semantic`` is the primary acceptance metric.  It compares the Stage2
execution meaning (status, coordinates, selectors, task operations, requested
slots, derivations and premises) while allowing plan-local ID renaming and
task ordering differences that preserve the dependency graph.

``exact`` is diagnostic.  It compares the complete public handoff payload
except the transport/provenance ``handoff_id``.  Therefore task/output IDs,
ordering, defaults, clarification IDs/text and reason ordering still count.

The default Gold is the complete ``fixtures/query_plan_v04_final`` release.
The immutable migration snapshot remains available as ``ARCHIVE_EXPECTED``
for historical diagnostics.  The script is offline-only, never invokes HCX,
and never rewrites a candidate row from either fixture.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from agent.query_plan import QueryPlanHandoff


# ---------------------------------------------------------------------------
# 실행 의미 서명 — 예전에는 v0.5 평가 스크립트가 갖고 있었다.
#
# 채점기가 자기 채점 기준을 남의 스크립트에서 빌려오면, 그 스크립트를 지울 때
# 채점이 함께 죽는다(실제로 그랬다).  기준은 채점기가 소유한다.
# ---------------------------------------------------------------------------

_DROPPED = object()

def _collapse_discrete_from_cumulative(plan: dict) -> dict:
    """누적 두 값의 차감을 **단일분기 직접 조회와 같은 정규형**으로 접는다.

    DART 는 같은 분기를 두 방식으로 담는다 — 3개월 단독값이 원문에 직접 있고,
    누적 두 개를 빼도 같은 값이 나온다. 동결 정답지의 ``answer_requirements`` 도
    두 경로를 **동등 정답으로 명시**한다. 그런데 fact 행만 대조하면 한쪽은
    행 2개, 다른 쪽은 행 1개라 실행 의미가 같은데도 불일치로 잡힌다.

    ``discrete_from_cumulative`` derivation 이 있고 두 operand 가 같은 회사·개념·
    범위의 연초 시작 누적이면, 두 행을 **차구간 단일 행 하나**로 접는다.
    특정 회사·연도와 무관한 회계 항등식이다.
    """

    # 소유 task 의 ``as_of`` 를 같이 들고 다닌다. 접은 행에서 이 축이 빠지면
    # 다른 fact 행과 형태가 달라져 같은 실행 의미가 불일치로 잡힌다.
    facts = {
        fact.get("output_id"): (fact, task.get("as_of"))
        for task in plan.get("tasks", []) or []
        for fact in (task.get("facts") or [])
        if fact.get("output_id")
    }
    result: dict = {}
    for derivation in plan.get("derivations", []) or []:
        if derivation.get("operator") != "discrete_from_cumulative":
            continue
        operands = derivation.get("operands") or []
        if len(operands) != 2:
            continue
        pair = [facts.get(operand.get("output_id")) for operand in operands]
        if any(row is None or not row[0].get("cumulative") for row in pair):
            continue
        (current, current_as_of), (previous, _) = pair
        keys = ("corp_code", "concept", "scope", "period_start")
        if any(current.get(key) != previous.get(key) for key in keys):
            continue
        if not (previous.get("period_end") or "") < (current.get("period_end") or ""):
            continue
        try:
            start = (date.fromisoformat(previous["period_end"])
                     + timedelta(days=1)).isoformat()
        except (KeyError, TypeError, ValueError):
            continue
        result[current["output_id"]] = (
            "financial", current.get("corp_code"), current.get("concept"),
            start, current.get("period_end"), "quarter", False,
            current.get("scope"), current_as_of,
        )
        result[previous["output_id"]] = _DROPPED
    return result

def _fact_index(plan: dict) -> dict:
    """`output_id` → 그 fact 의 **의미**. 이름 자체는 비교하지 않는다.

    정답지는 `output-1`, 우리는 `result_1` 을 쓴다. 이름이 다르다고 틀린 계획이
    아니다 — plan 안에서만 통하는 식별자다.
    """

    index: dict = {}
    for task in plan.get("tasks") or []:
        for fact in task.get("facts") or []:
            output_id = fact.get("output_id")
            if isinstance(output_id, str):
                index[output_id] = (
                    fact.get("corp_code"), fact.get("concept"),
                    fact.get("period_start"), fact.get("period_end"),
                    fact.get("cumulative"), fact.get("scope"))
    return index

def _derivation_rows(plan: dict) -> tuple:
    """계산 구조.

    `argmax` 는 **순서를 보지 않는다** — 최대값을 고르는 연산이라 후보 순서가 답을
    바꾸지 않는다. `difference` 는 순서가 부호를 바꾸므로 그대로 본다.

    `discrete_from_cumulative` 는 뺀다. fact 축에서 이미 접히고
    (`_collapse_discrete_from_cumulative`), 코퍼스에 `cumulative=False` 단일분기
    행이 실제로 있어 단일 fact 계획이 **실행 가능하고 더 단순**하다.
    """

    index = _fact_index(plan)
    rows = []
    for derivation in plan.get("derivations") or []:
        operator = derivation.get("operator")
        if operator == "discrete_from_cumulative":
            continue
        operands = tuple(
            (index.get(operand.get("output_id"), operand.get("output_id")),
             operand.get("field"))
            for operand in derivation.get("operands") or [])
        if operator == "argmax":
            operands = tuple(sorted(operands, key=str))
        rows.append((operator, operands, derivation.get("rounding_rule")))
    return tuple(sorted(rows, key=str))

def _slot_rows(plan: dict) -> tuple:
    rows = [
        (task.get("kind"), task.get("operation"),
         tuple(sorted(task.get("requested_slots") or [])))
        for task in plan.get("tasks") or []
        if task.get("requested_slots")
    ]
    return tuple(sorted(rows, key=str))

def _claim_rows(plan: dict) -> tuple:
    rows = [
        (claim.get("kind"), tuple(sorted(claim.get("verification_refs") or [])))
        for claim in plan.get("premise_claims") or []
    ]
    return tuple(sorted(rows, key=str))

def _canonical(value: object) -> object:
    """비교 가능한 모양으로 접는다. **비어 있음은 한 가지로만 표현한다.**

    `None`·`[]`·`{}` 와 「모든 값이 None 인 selector」는 모두 `None` 이다 —
    한쪽이 selector 를 안 내고 다른 쪽이 빈 selector 를 내는 것은 실행이 같다.
    """

    if isinstance(value, dict):
        rows = tuple(sorted(
            (key, _canonical(item)) for key, item in value.items()
            if _canonical(item) is not None))
        return rows or None
    if isinstance(value, (list, tuple)):
        rows = tuple(_canonical(item) for item in value)
        return rows or None
    if value == "":
        return None
    return value

_SELECTOR_IDENTITY = ("event_key", "doc_id", "rcept_no")

def _selector_signature(selector: object) -> object:
    if not isinstance(selector, dict):
        return _canonical(selector)
    for name in _SELECTOR_IDENTITY:
        value = selector.get(name)
        if value:
            return (name, value)
    return _canonical(selector)

_EXECUTABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "financial": ("as_of", "view"),
    "narrative": ("operation", "corp_codes", "as_of", "retrieval_query",
                  "document_selector", "periods"),
    "disclosure": ("operation", "corp_code", "as_of",
                   "document_selector", "event_selector", "field_outputs"),
    "event": ("operation", "corp_code", "selector", "timepoints",
              "field_outputs"),
    "correction": ("operation", "corp_code", "as_of",
                   "document_selector", "event_selector", "field_outputs"),
    "document": ("operation", "corp_code", "as_of",
                 "selector", "event_selector"),
}

#: 실행을 바꾸지 않는 필드. `_EXECUTABLE_FIELDS` 에도 여기에도 없는 필드가 계약
#: 모델에 생기면 `assert_scorer_covers_contract` 가 **채점기를 실패시킨다** —
#: 조용히 빠지는 것이 이번 P0 의 원인이었으므로 fail-open 을 두지 않는다.
_NON_EXECUTABLE_FIELDS = frozenset({
    "kind",           # 서명의 머리로 이미 들어간다
    "task_id",        # plan 안에서만 통하는 이름
    "output_id",      # 〃
    "corp_name",      # corp_code 에서 registry 로 유도되는 표시용 이름
    "corp_names",     # 〃
    "facts",          # fact 좌표로 따로 본다
    "requested_slots",  # `_slot_rows` 로 따로 본다
})


def _task_signature(task: dict) -> tuple:
    """한 task 의 **실행 서명**. kind 를 머리로 두고 그 kind 의 필드를 담는다."""

    kind = task.get("kind")
    fields = _EXECUTABLE_FIELDS.get(kind)
    if fields is None:
        # 모르는 kind 를 통과시키지 않는다.
        return ("unknown_task_kind", kind, _canonical(task))
    return (kind, *(
        _selector_signature(task.get(name)) if name.endswith("selector")
        else _canonical(task.get(name))
        for name in fields))

def _fact_semantics(plan: dict | object) -> tuple:
    """**Stage2 가 실행하는 축 전부.** 이것 하나로만 채점한다.

    예전에는 조회 좌표(회사·개념·기간·범위·기준일)만 보는 「관대」 기준이 따로
    있었다. 그것으로 「35/70 통과」라고 말했는데 Stage2 는 plan 을 그대로 실행하므로
    빠진 축이 있으면 연동이 깨진다. 실제로 `derivations` 를 보지 않아 **계산을 통째로
    빼먹은 계획이 통과**하고 있었다. 기준을 하나로 합쳤다.

    보는 것: fact 좌표 + task kind 별 실행 필드(`_EXECUTABLE_FIELDS`) +
    `statement` · `derivations` · `requested_slots` · `premise_claims`.

    보지 않는 것은 **표현일 뿐 실행을 바꾸지 않는 것**이다 — `output_id` 이름,
    task 순서, `corp_name` 같은 표시용 이름, `concept_mention` 같은 중간 표기.
    """

    if not isinstance(plan, dict):
        plan = plan.model_dump(mode="json")
    collapsed = _collapse_discrete_from_cumulative(plan)
    rows = []
    for task in plan.get("tasks", []):
        signature = _task_signature(task)
        facts = task.get("facts") or []
        for fact in facts:
            replacement = collapsed.get(fact.get("output_id"))
            if replacement is _DROPPED:
                continue
            if replacement is not None:
                # 접힌 행은 `statement` 를 담지 않는다. 한쪽만 붙으면 길이가 달라
                # **값이 아니라 형태 때문에** 불일치가 된다.
                rows.append(tuple(replacement) + (fact.get("statement"), signature))
                continue
            rows.append((
                task.get("kind"), fact.get("corp_code"), fact.get("concept"),
                fact.get("period_start"), fact.get("period_end"),
                fact.get("period_type"), fact.get("cumulative"),
                fact.get("scope"), task.get("as_of"), fact.get("statement"),
                signature,
            ))
        if not facts:
            rows.append(signature)
    return (tuple(sorted(rows, key=lambda row: tuple(str(x) for x in row))),
            _derivation_rows(plan), _slot_rows(plan), _claim_rows(plan))


ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_EXPECTED = (
    ROOT / "fixtures/query_plan_v04/query_plan_handoffs_v0.4.jsonl")
DEFAULT_EXPECTED = (
    ROOT / "fixtures/query_plan_v04_final/query_plan_handoffs_v0.4.jsonl")
MAX_JSONL_BYTES = 16 * 1024 * 1024


class QueryPlanV04DualEvalError(ValueError):
    """Candidate/Gold rows cannot be compared without opening the contract."""


@dataclass(frozen=True, slots=True)
class QueryPlanV04DualScore:
    question_id: str
    exact: bool
    semantic: bool
    exact_issue_codes: tuple[str, ...]
    semantic_issue_codes: tuple[str, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QueryPlanV04DualEvalError(f"JSON key가 중복됩니다: {key}")
        result[key] = value
    return result


def _strict_handoff(value: Any) -> QueryPlanHandoff:
    if isinstance(value, QueryPlanHandoff):
        value = value.model_dump(mode="json", warnings=False)
    if not isinstance(value, dict):
        raise QueryPlanV04DualEvalError("handoff는 JSON object여야 합니다")
    # 역질문 항목은 리스트다. 문자열 하나로 적힌 옛 산출물도 읽을 수 있게
    # 한 항목짜리로 감싼다 — 채점 대상은 항목 내용이지 자료형 세대가 아니다.
    clarification = value.get("clarification")
    if isinstance(clarification, dict) and isinstance(
            clarification.get("question"), str):
        clarification = dict(clarification)
        clarification["question"] = [clarification["question"]]
        value = {**value, "clarification": clarification}
    try:
        return QueryPlanHandoff.model_validate_json(
            _canonical_json(value), strict=True)
    except ValueError as exc:
        raise QueryPlanV04DualEvalError(
            "QueryPlanHandoff 0.4 strict validation 실패") from exc


def _exact_payload(handoff: QueryPlanHandoff) -> dict[str, Any]:
    """계획의 **내용**만 남긴다. 실행 단위 식별자는 뺀다.

    `handoff_id` 는 처음부터 제외해 왔다. `clarification_id` 도 같은 성격이다 —
    무엇을 되묻는지가 아니라 그 되묻기를 가리키는 이름이다.

    Gold 는 이 자리에 `clarification-r-a-001` 처럼 **채점 하네스의 문항 번호**를
    적는다(8행 중 7행). 서비스 Stage1 은 문항 번호를 받지 않고 사용자 질문만
    받으므로, 어떤 올바른 구현도 그 값을 만들어낼 수 없다. Gold 자신도 R-F-004
    에서는 `clarify-contract-event` 라는 의미 기반 이름을 쓴다 — 규칙이 하나가
    아니다.

    `plan.applied_defaults` 도 뺀다. 이 자리에 적히는 값은 **전부 계획의 다른
    칸에서 되살릴 수 있다** — `view=restated` 는 `tasks[].view`, `as_of` 는
    `corpus_cutoff`, `작년→2025` 는 `facts[].period_start/end`, `scope=CFS` 는
    `facts[].scope`, `최근→…선택` 은 selector 의 접수번호다. 계획의 실행 내용은
    이 줄이 없어도 한 글자도 달라지지 않는다.

    답변 동작을 지시하는 항목("답변에 명시" 등)도 고유하지 않다. 그 지시는
    `answer_requirements.behavior_requirements` 에 원래 자리가 있고, Gold 도
    거기에 같은 말을 적어두었다. 여기 남은 것은 되풀이거나 사람이 남긴 메모
    (`정책 미확정: R-A-002 역질문 정책과의 일관성 팀 확인 필요`)이며, 뒤쪽은
    어떤 구현도 글자까지 만들어낼 수 없다.

    되묻는 **내용**(슬롯·선택지·문구)은 그대로 비교한다. 빠지는 것은 이름과
    이 되풀이뿐이다.
    """

    payload = handoff.model_dump(mode="json", warnings=False)
    payload.pop("handoff_id", None)
    clarification = payload.get("clarification")
    if isinstance(clarification, dict):
        clarification.pop("clarification_id", None)
    plan = payload.get("plan")
    if isinstance(plan, dict):
        plan.pop("applied_defaults", None)
    return payload


def _clarification_signature(handoff: QueryPlanHandoff) -> tuple[Any, ...]:
    clarification = handoff.clarification
    if clarification is None:
        return ()
    return tuple(sorted(
        (
            slot.target,
            tuple(sorted(
                _canonical_json(value) for value in slot.allowed_values)),
        )
        for slot in clarification.slots
    ))


def _semantic_issue_codes(
        expected: QueryPlanHandoff,
        actual: QueryPlanHandoff,
        ) -> tuple[str, ...]:
    if expected.status != actual.status:
        return ("status_mismatch",)
    if expected.status == "ready":
        if expected.plan is None or actual.plan is None:
            return ("ready_plan_missing",)
        if _fact_semantics(expected.plan) != _fact_semantics(actual.plan):
            return ("execution_semantics_mismatch",)
        return ()
    if expected.status == "needs_clarification":
        if _clarification_signature(expected) != _clarification_signature(actual):
            return ("clarification_semantics_mismatch",)
        return ()
    if tuple(sorted(expected.reasons)) != tuple(sorted(actual.reasons)):
        return ("terminal_reason_semantics_mismatch",)
    return ()


def score_query_plan_v04_handoff(
        question_id: str,
        expected: QueryPlanHandoff | dict[str, Any],
        actual: QueryPlanHandoff | dict[str, Any],
        ) -> QueryPlanV04DualScore:
    gold = _strict_handoff(expected)
    candidate = _strict_handoff(actual)
    semantic_issues = _semantic_issue_codes(gold, candidate)
    exact = _canonical_json(_exact_payload(gold)) == _canonical_json(
        _exact_payload(candidate))
    semantic = not semantic_issues
    if exact and not semantic:
        raise QueryPlanV04DualEvalError(
            "exact match가 semantic mismatch가 되는 scorer 불변식 오류")
    return QueryPlanV04DualScore(
        question_id=question_id,
        exact=exact,
        semantic=semantic,
        exact_issue_codes=() if exact else ("query_plan_exact_mismatch",),
        semantic_issue_codes=semantic_issues,
    )


def load_query_plan_v04_rows(path: str | Path) -> dict[str, QueryPlanHandoff]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise QueryPlanV04DualEvalError(
            f"query-plan JSONL이 일반 파일이 아닙니다: {source}")
    payload = source.read_bytes()
    if (
        not payload
        or len(payload) > MAX_JSONL_BYTES
        or payload.startswith(b"\xef\xbb\xbf")
        or b"\r" in payload
        or not payload.endswith(b"\n")
    ):
        raise QueryPlanV04DualEvalError(
            f"query-plan JSONL byte 계약이 잘못되었습니다: {source}")
    try:
        lines = payload.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise QueryPlanV04DualEvalError("query-plan JSONL은 UTF-8이어야 합니다") from exc

    rows: dict[str, QueryPlanHandoff] = {}
    for line_no, line in enumerate(lines, 1):
        if not line:
            raise QueryPlanV04DualEvalError(
                f"query-plan JSONL에 빈 행이 있습니다: {source}:{line_no}")
        try:
            row = json.loads(line, object_pairs_hook=_strict_object)
        except (json.JSONDecodeError, QueryPlanV04DualEvalError) as exc:
            raise QueryPlanV04DualEvalError(
                f"query-plan JSONL 파싱 실패: {source}:{line_no}") from exc
        if not isinstance(row, dict):
            raise QueryPlanV04DualEvalError(
                f"query-plan row는 object여야 합니다: {source}:{line_no}")
        question_id = row.get("question_id")
        handoff = row.get("handoff")
        if (
            not isinstance(question_id, str)
            or not question_id
            or question_id in rows
            or not isinstance(handoff, dict)
        ):
            raise QueryPlanV04DualEvalError(
                f"query-plan row binding이 잘못되었습니다: {source}:{line_no}")
        rows[question_id] = _strict_handoff(handoff)
    return rows


def evaluate_query_plan_v04_rows(
        expected: dict[str, QueryPlanHandoff],
        actual: dict[str, QueryPlanHandoff],
        *,
        question_ids: Iterable[str] | None = None,
        ) -> dict[str, Any]:
    selected = tuple(expected) if question_ids is None else tuple(question_ids)
    if not selected or len(selected) != len(set(selected)):
        raise QueryPlanV04DualEvalError("평가 question ID가 비었거나 중복됩니다")
    unknown = tuple(question_id for question_id in selected if question_id not in expected)
    if unknown:
        raise QueryPlanV04DualEvalError(
            f"Gold에 없는 question ID입니다: {list(unknown)}")

    scores = tuple(
        score_query_plan_v04_handoff(
            question_id,
            expected[question_id],
            actual[question_id],
        )
        for question_id in selected
        if question_id in actual
    )
    missing = tuple(question_id for question_id in selected if question_id not in actual)
    semantic_pass_exact_fail = tuple(
        score.question_id for score in scores
        if score.semantic and not score.exact
    )
    semantic_fail = tuple(
        score.question_id for score in scores if not score.semantic)
    issue_counts: dict[str, int] = {}
    for score in scores:
        for code in score.semantic_issue_codes:
            issue_counts[code] = issue_counts.get(code, 0) + 1
    return {
        "schema_version": "query-plan-v04-dual-eval/1.0",
        "evaluation_claim": (
            "semantic_primary_exact_diagnostic_handoff_id_excluded"),
        "selected_count": len(selected),
        "evaluated_count": len(scores),
        "missing_count": len(missing),
        "exact_count": sum(score.exact for score in scores),
        "semantic_count": sum(score.semantic for score in scores),
        "all_selected_semantic_passed": (
            not missing and all(score.semantic for score in scores)),
        "semantic_pass_exact_fail_question_ids": list(
            semantic_pass_exact_fail),
        "semantic_fail_question_ids": list(semantic_fail),
        "missing_question_ids": list(missing),
        "semantic_issue_counts": dict(sorted(issue_counts.items())),
        "rows": [
            {
                "question_id": score.question_id,
                "exact": score.exact,
                "semantic": score.semantic,
                "exact_issue_codes": list(score.exact_issue_codes),
                "semantic_issue_codes": list(score.semantic_issue_codes),
            }
            for score in scores
        ],
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    expected = load_query_plan_v04_rows(args.expected)
    actual = load_query_plan_v04_rows(args.actual)
    summary = evaluate_query_plan_v04_rows(
        expected,
        actual,
        question_ids=args.question_id or None,
    )
    payload = (
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if args.output is not None:
        _atomic_write(args.output, payload)
    print(payload.decode("utf-8"), end="")
    return 0 if summary["all_selected_semantic_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
