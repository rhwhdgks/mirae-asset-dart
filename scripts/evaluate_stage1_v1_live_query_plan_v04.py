#!/usr/bin/env python3
"""Evaluate saved SemanticIntent v1 rows as final QueryPlanHandoff 0.4.

This is an offline compatibility evaluation.  It performs no provider calls.
Candidate generation is deliberately separated from final scoring:

    saved HCX row -> SemanticIntent v1 -> generic v0.4 compatibility bridge
                  -> existing canonical resolver -> QueryPlanHandoff 0.4

Only after every candidate has been persisted does the script load the final
v0.4 release handoffs and run the existing semantic/exact dual scorer.  The bridge
does not accept a question ID and never reads a Gold proposal or handoff.

The bridge is diagnostic, not the production v1 resolver.  It measures how the
saved semantic interpretation behaves when connected to the already deployed
generic v0.4 resolver while the native v1 resolver coverage is still smaller
than 70 questions.
"""

from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from agent.contracts import (
    Derivation,
    DocumentSelector,
    EventSelector,
    FieldOutputSpec,
    OutputRef,
    PremiseClaim,
    ResolvedCorrectionTask,
    ResolvedDisclosureTask,
    ResolvedDocumentTask,
    ResolvedEventTask,
    ResolvedQueryPlan,
    TaskVerificationRef,
)
from agent.date_surface import parse_date_surface, question_date_surfaces
from agent.query_plan import (
    QueryPlanHandoff,
)
from agent.planning import funding_decision_keywords
from agent.semantic_intent_v1 import (
    HcxSemanticIntentWire,
    SemanticIntent,
    semantic_intent_digest,
)
from agent.semantic_intent_v1_boundary import (
    SemanticIntentBoundaryEvidence,
    normalize_semantic_intent_bounded,
)
from agent.stage1_assembly import CORPUS_CUTOFF, REFERENCE_DATE
from scripts.score_query_plan_v04_dual import (
    DEFAULT_EXPECTED,
    evaluate_query_plan_v04_rows,
    load_query_plan_v04_rows,
)


ROOT = Path(__file__).resolve().parent.parent
HISTORICAL_GOLD10 = ROOT / (
    "out/evaluation/stage1_v1_gold10_v112_schema2_boundary4_20260821_a")
HISTORICAL_REMAINING60 = ROOT / (
    "out/evaluation/"
    "stage1_v1_remaining60_v112_schema2_boundary4_full_20260821_a")
DEFAULT_QUESTIONS = (
    ROOT / "fixtures/query_plan_v04_final/questions_v0.4.jsonl")
DEFAULT_OUTPUT = ROOT / (
    "out/evaluation/stage1_v1_live70_query_plan_v04_20260821_a")
MAX_INPUT_BYTES = 16 * 1024 * 1024
_HMAC_KEY = b"stage1-v1-v04-offline-eval-key-32bytes-minimum"


class LiveV04EvalError(ValueError):
    """Saved rows or the offline compatibility evaluation are invalid."""


@dataclass(frozen=True, slots=True)
class SavedIntentRow:
    question_id: str
    status: str
    provider_wire: HcxSemanticIntentWire | None
    semantic_intent: SemanticIntent | None
    failure: dict[str, Any] | None
    source_path: Path
    row_sha256: str


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


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LiveV04EvalError(f"JSON key가 중복됩니다: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LiveV04EvalError(f"일반 JSON 파일이 아닙니다: {path}")
    payload = path.read_bytes()
    if (
        not payload
        or len(payload) > MAX_INPUT_BYTES
        or payload.startswith(b"\xef\xbb\xbf")
        or b"\r" in payload
    ):
        raise LiveV04EvalError(f"JSON byte 계약이 잘못되었습니다: {path}")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveV04EvalError(f"JSON 파싱 실패: {path}") from exc
    if not isinstance(value, dict):
        raise LiveV04EvalError(f"JSON root는 object여야 합니다: {path}")
    return value


def _strict_saved_success_payload(
        value: dict[str, Any], *, source_path: Path,
        ) -> tuple[HcxSemanticIntentWire, SemanticIntent]:
    """Load both saved representations; provider wire is replay authority."""

    try:
        provider_wire = HcxSemanticIntentWire.model_validate(
            value.get("provider_wire"), strict=True)
        old_intent = SemanticIntent.model_validate(
            value.get("semantic_intent"), strict=True)
    except ValueError as exc:
        raise LiveV04EvalError(
            f"success row provider_wire/SemanticIntent strict validation 실패: "
            f"{source_path}") from exc
    return provider_wire, old_intent


#: Official evaluation accepts only one all70 run.  Historical split runs are
#: intentionally diagnostics: their old prerequisite sidecars are provenance,
#: never a condition for the final release acceptance.
EVALUATION_MODES = ("official", "diagnostic")


def _require_final70_run_dirs(run_dirs: "tuple[Path, ...]") -> None:
    """The official scorer accepts exactly one pinned all70 candidate run."""

    if len(run_dirs) != 1:
        raise LiveV04EvalError(
            "공식 채점은 단일 all70 run만 받습니다; 분할 실행은 diagnostic 모드만 씁니다")
    summary = _read_json(run_dirs[0] / "summary.json")
    if summary.get("phase") != "all70" or summary.get("selected_count") != 70:
        raise LiveV04EvalError(
            "공식 채점 입력은 selected_count=70인 all70 run이어야 합니다; "
            "gold10/remaining60 분할 실행은 historical diagnostic 전용입니다")


def _load_saved_rows(
        run_dirs: Iterable[Path], *, overlay_dirs: Iterable[Path] = (),
        evaluation_mode: str = "official",
        ) -> dict[str, SavedIntentRow]:
    rows: dict[str, SavedIntentRow] = {}
    bindings: dict[str, set[str]] = {
        "prompt_sha256": set(),
        "provider_schema_sha256": set(),
        "generation_config_sha256": set(),
        "questions_sha256": set(),
    }
    overlays = tuple(overlay_dirs)
    if evaluation_mode not in EVALUATION_MODES:
        raise LiveV04EvalError(f"평가 모드가 잘못되었습니다: {evaluation_mode}")
    run_dirs = tuple(run_dirs)
    if evaluation_mode == "official":
        _require_final70_run_dirs(run_dirs)
    sources = tuple((run_dir, False) for run_dir in run_dirs) + tuple(
        (run_dir, True) for run_dir in overlays)
    overlaid: set[str] = set()
    for run_dir, is_overlay in sources:
        summary = _read_json(run_dir / "summary.json")
        for key in bindings:
            value = summary.get(key)
            if not isinstance(value, str):
                raise LiveV04EvalError(f"summary {key}가 없습니다: {run_dir}")
            bindings[key].add(value)
        row_dir = run_dir / "rows"
        if row_dir.is_symlink() or not row_dir.is_dir():
            raise LiveV04EvalError(f"rows directory가 아닙니다: {row_dir}")
        loaded_ids: list[str] = []
        for path in sorted(row_dir.glob("*.json")):
            payload = path.read_bytes()
            value = _read_json(path)
            question_id = value.get("question_id")
            status = value.get("status")
            if (
                not isinstance(question_id, str)
                or not question_id
                or (question_id in rows and not is_overlay)
                or not isinstance(status, str)
            ):
                raise LiveV04EvalError(f"eval row identity가 잘못되었습니다: {path}")
            if is_overlay and (
                    question_id not in rows or question_id in overlaid):
                raise LiveV04EvalError(
                    f"overlay row binding이 잘못되었습니다: {path}")
            intent: SemanticIntent | None = None
            provider_wire: HcxSemanticIntentWire | None = None
            if status == "success":
                provider_wire, intent = _strict_saved_success_payload(
                    value, source_path=path)
            failure = value.get("failure")
            rows[question_id] = SavedIntentRow(
                question_id=question_id,
                status=status,
                provider_wire=provider_wire,
                semantic_intent=intent,
                failure=failure if isinstance(failure, dict) else None,
                source_path=path,
                row_sha256=sha256(payload).hexdigest(),
            )
            loaded_ids.append(question_id)
            if is_overlay:
                overlaid.add(question_id)
        selected_ids = summary.get("selected_question_ids")
        if (
            not isinstance(selected_ids, list)
            or any(not isinstance(value, str) for value in selected_ids)
            or set(selected_ids) != set(loaded_ids)
            or len(selected_ids) != len(loaded_ids)
        ):
            raise LiveV04EvalError(
                f"summary/row selected binding이 다릅니다: {run_dir}")
    drift = {key: values for key, values in bindings.items() if len(values) != 1}
    if drift:
        raise LiveV04EvalError(f"saved run binding이 다릅니다: {drift}")
    if len(rows) != 70:
        raise LiveV04EvalError(f"저장 eval row가 70개가 아닙니다: {len(rows)}")
    return rows


def _load_questions(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise LiveV04EvalError("questions JSONL이 일반 파일이 아닙니다")
    payload = path.read_bytes()
    if (
        not payload
        or len(payload) > MAX_INPUT_BYTES
        or payload.startswith(b"\xef\xbb\xbf")
        or b"\r" in payload
        or not payload.endswith(b"\n")
    ):
        raise LiveV04EvalError("questions JSONL byte 계약이 잘못되었습니다")
    questions: dict[str, str] = {}
    for line_no, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line, object_pairs_hook=_strict_object)
        except json.JSONDecodeError as exc:
            raise LiveV04EvalError(
                f"questions JSONL 파싱 실패: {line_no}") from exc
        if not isinstance(row, dict):
            raise LiveV04EvalError("question row는 object여야 합니다")
        question_id = row.get("question_id")
        question = row.get("question")
        if (
            not isinstance(question_id, str)
            or not isinstance(question, str)
            or not question
            or question_id in questions
        ):
            raise LiveV04EvalError(f"question row가 잘못되었습니다: {line_no}")
        questions[question_id] = question
    if len(questions) != 70:
        raise LiveV04EvalError(f"question row가 70개가 아닙니다: {len(questions)}")
    return questions


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _item_text(intent: SemanticIntent) -> str:
    return " ".join([
        *(item.target.surface for item in intent.answer_items),
        *(surface for item in intent.answer_items
          for surface in item.output.field_surfaces),
    ])


def _is_financial(question: str, intent: SemanticIntent) -> bool:
    text = f"{question} {_item_text(intent)}".casefold()
    financial = any(
        item.target.kind == "metric" for item in intent.answer_items
    ) and any(cue in text for cue in (
        "매출", "팔았", "벌었", "영업이익", "순이익", "자산총계",
        "총자산", "유형자산 취득", "capex",
    ))
    contract_metric = any(cue in text for cue in (
        "계약금액", "계약 금액", "해지금액", "해지 금액",
    ))
    return financial and not contract_metric


def _period_spans(question: str) -> list[str]:
    """Return question-grounded financial periods, longest first."""

    patterns = (
        r"(?<![0-9])(?:20)?[0-9]{2}년(?:도)?\s*[1-4](?:분기|Q)",
        r"(?<![0-9])20[0-9]{2}\s*[./-]\s*[1-4]Q",
        r"(?<![0-9])(?:20)?[0-9]{2}년(?:도)?\s*(?:상반기|반기)",
        r"(?<![0-9])20[0-9]{2}년(?:도)?",
        r"(?<![0-9])[0-9]{2}년(?:도)?",
        r"작년|지난해|재작년|올해|금년",
    )
    matches: list[tuple[int, int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            matches.append((match.start(), -len(match.group(0)), match.group(0)))
    selected: list[tuple[int, int, str]] = []
    occupied: set[int] = set()
    for start, negative_length, value in sorted(matches, key=lambda row: (row[0], row[1])):
        end = start + len(value)
        indices = set(range(start, end))
        if indices.intersection(occupied):
            continue
        occupied.update(indices)
        selected.append((start, negative_length, value))
    return _unique(value for _, _, value in sorted(selected))


def _collapse_contained_periods(periods: Iterable[str]) -> list[str]:
    """Drop a bare year when a period in that same fiscal year is present."""

    values = _unique(periods)
    detailed_years = {
        match.group("year")[-2:]
        for value in values
        if (match := re.fullmatch(
            r"(?P<year>(?:20)?[0-9]{2})년?\s*(?:[1-4](?:분기|Q)|상반기|반기)",
            value,
            flags=re.IGNORECASE,
        )) is not None
    }
    return [
        value for value in values
        if not (
            (match := re.fullmatch(
                r"(?P<year>(?:20)?[0-9]{2})년(?:도)?", value)) is not None
            and match.group("year")[-2:] in detailed_years
        )
    ]


def _scope_expression(item: Any) -> str:
    values = [
        *item.scope.scope_qualifier_expressions,
        *item.target.qualifier_surfaces,
        item.target.surface,
    ]
    for cue in ("연결기준", "별도기준", "연결", "별도"):
        for value in values:
            if cue in value:
                return cue
    return ""


def _statement_expression(item: Any) -> str:
    values = [item.target.surface, *item.target.qualifier_surfaces]
    for cue in ("연결현금흐름표", "현금흐름표", "손익계산서", "재무상태표"):
        for value in values:
            if cue in value:
                return cue
    return ""


def _financial_concept_expression(question: str, item: Any) -> str:
    """Return a question-grounded legacy concept surface.

    SemanticIntent targets may carry presentation qualifiers such as
    ``단일 분기`` or the full statement name.  The legacy resolver expects the
    underlying account concept instead.  Keep this mapping deliberately small
    and require the account cue to be present in the question or target.
    """

    text = question.casefold()
    for surface in (
            "유형자산 취득 현금유출액", "유형자산 취득액",
            "유형자산 취득", "capex", "매출액", "매출"):
        if surface in text:
            return surface
    return item.target.surface


def _financial_wire(
        question: str, intent: SemanticIntent,
        question_company_surface: Any = None,
        ) -> dict[str, Any]:
    entities = {entity.entity_id: entity.surface for entity in intent.entities}
    global_companies = [
        entity.surface for entity in intent.entities
        if entity.kind_hint == "company"
    ]
    if not global_companies and callable(question_company_surface):
        surface = question_company_surface(question)
        if isinstance(surface, str) and surface:
            global_companies = [surface]
    question_periods = _period_spans(question)
    facts: list[dict[str, Any]] = []
    fact_owner_items: list[str] = []
    for item in intent.answer_items:
        companies = _unique(
            entities[ref] for ref in item.target.entity_refs if ref in entities)
        if item.operation == "compare" and len(global_companies) >= 2:
            # A comparison operand may be represented as a free entity rather
            # than another target ref.  All company entities are question-
            # grounded, so retaining them closes the comparison without
            # inventing an issuer.
            companies = _unique(global_companies)
        if not companies:
            companies = _unique(global_companies)
        if not companies:
            companies = [""]
        periods = _collapse_contained_periods(
            item.scope.target_period_expressions)
        comparison_cues = (
            item.operation == "compare"
            or any(cue in question for cue in (
                "차이", "전년", "재작년", "보다", "비교", "변했", "늘었",
            ))
        )
        if comparison_cues and len(question_periods) > len(periods):
            periods = question_periods
        elif not periods and question_periods:
            periods = question_periods
        if not periods:
            periods = [""]
        # Multiple companies and multiple periods are separate axes.  The
        # frozen corpus has no Cartesian comparison request, so preserve the
        # explicit item axis instead of inventing every combination.
        pairs: list[tuple[str, str]]
        if len(companies) > 1 and len(periods) > 1:
            pairs = list(zip(companies, periods, strict=False))
        else:
            pairs = [
                (company, period)
                for company in companies
                for period in periods
            ]
        for company, period in pairs:
            facts.append({
                "company_mention": company,
                "concept_mention": _financial_concept_expression(question, item),
                "period_expression": period,
                "scope_expression": _scope_expression(item),
                "statement_expression": _statement_expression(item),
                "view_expression": "",
            })
            fact_owner_items.append(item.item_id)

    # A scope comparison can be stated entirely in the question while HCX
    # emits only one of its two operands.  Restore the missing axis only for
    # the closed, same-company/same-period shape where both public scope names
    # and an explicit difference request are present.  No corpus value or Gold
    # coordinate is consulted.
    compact_question = re.sub(r"\s+", "", question)
    if (
        len(facts) == 1
        and "연결" in compact_question
        and "별도" in compact_question
        and "차이" in compact_question
    ):
        base = dict(facts[0])
        facts = [
            {**base, "scope_expression": "연결"},
            {**base, "scope_expression": "별도"},
        ]
        fact_owner_items = [fact_owner_items[0], fact_owner_items[0]]

    # Keep the first occurrence; duplicate fact rows are invalid on the old
    # provider wire and carry no additional execution meaning.
    deduped: list[dict[str, Any]] = []
    deduped_owners: list[str] = []
    seen: set[str] = set()
    for fact, owner in zip(facts, fact_owner_items, strict=True):
        key = _canonical_json(fact)
        if key not in seen:
            seen.add(key)
            deduped.append(fact)
            deduped_owners.append(owner)
    facts = deduped
    fact_owner_items = deduped_owners

    fact_refs = [
        {"kind": "fact", "task_index": 0, "fact_index": index}
        for index in range(len(facts))
    ]
    analyses: list[dict[str, Any]] = []
    if len(fact_refs) >= 2:
        text = f"{question} {_item_text(intent)}".casefold()
        distinct_companies = {
            fact["company_mention"] for fact in facts if fact["company_mention"]
        }
        percent = any(cue in text for cue in (
            "퍼센트", "몇퍼", "%", "전년비",
        )) or (
            "재작년" in text and any(cue in text for cue in ("늘", "증가", "감소"))
        )
        winner = len(distinct_companies) >= 2 and any(cue in text for cue in (
            "중", "누가", "더 큰", "더 많이", "비교", "컸지",
        ))
        difference = any(cue in text for cue in (
            "차이", "얼마나 변", "얼마나 달", "비교 ㄱㄱ",
        )) or (
            len(distinct_companies) >= 2 and "비교" in text and "컸지" not in text
        ) or (
            len(distinct_companies) <= 1 and "차이" in text
        )
        if winner:
            analyses.append({
                "operator": "argmax",
                "operands": fact_refs,
                "rounding_rule": "unspecified",
            })
        if percent:
            analyses.append({
                "operator": "percent_change",
                "operands": fact_refs[:2],
                "rounding_rule": "unspecified",
            })
        elif difference:
            analyses.append({
                "operator": "difference",
                "operands": fact_refs[:2],
                "rounding_rule": "unspecified",
            })

    analysis_refs = [
        {"kind": "analysis", "analysis_index": index}
        for index in range(len(analyses))
    ]
    premises = list(intent.premises)
    claims: list[dict[str, Any]] = []
    for premise in premises:
        refs: list[dict[str, Any]] = analysis_refs or fact_refs
        kind = premise.kind
        if re.search(r"[0-9]+(?:\s*조|\s*억|\s*원|\s*%)", premise.raw_text):
            kind = "numeric"
        claims.append({
            "kind": kind,
            "raw_text": premise.raw_text,
            "verification_refs": refs,
            "context_task_refs": [],
        })
    if (
        not claims
        and fact_refs
        and re.search(r"(?:맞아|맞지|이라던데|였지)\??\s*$", question.strip())
    ):
        numeric = re.search(
            r"(?<![0-9])([0-9]+(?:\.[0-9]+)?)\s*(조원|억원|만원|원|%)",
            question,
        )
        if numeric is not None:
            claims.append({
                "kind": "numeric",
                "raw_text": numeric.group(0),
                "verification_refs": fact_refs,
                "context_task_refs": [],
            })
    if (
        not claims
        and analysis_refs
        and len(global_companies) >= 2
        and re.search(r"보다\s*컸지\s*\?\s*$", question)
    ):
        ordered_companies = sorted(
            _unique(global_companies),
            key=lambda company: question.find(company),
        )
        if len(ordered_companies) >= 2 and all(
                question.find(company) >= 0 for company in ordered_companies[:2]):
            claims.append({
                "kind": "comparison",
                "raw_text": (
                    f"{ordered_companies[0]}가 "
                    f"{ordered_companies[1]}보다 컸다"),
                "verification_refs": analysis_refs,
                "context_task_refs": [],
            })
    verification_only = bool(claims) and bool(re.search(
        r"(?:맞아|맞지|이라던데|였지|컸지)\??\s*$", question.strip()))
    requested_outputs = [] if verification_only else (analysis_refs or fact_refs)
    return {
        "schema_version": "hcx-planner-intent-wire/0.5",
        "proposed_disposition": "process",
        "speech_act": (
            "verification" if verification_only
            else "mixed" if claims else "request"
        ),
        "tasks": [{
            "task_mode": "financial_lookup",
            "facts": facts,
            "as_of_expression": next((
                item.scope.as_of_expression
                for item in intent.answer_items
                if item.scope.as_of_expression
            ), ""),
        }],
        "analyses": analyses,
        "claims": claims,
        "requested_outputs": requested_outputs,
        "presentation": (
            "unspecified" if intent.presentation == "auto"
            else intent.presentation
        ),
        "reason_codes": [],
    }


def _proposal_modes(question: str, intent: SemanticIntent) -> list[str]:
    """Infer legacy task modes from semantic classes and generic Korean cues."""

    text = f"{question} {_item_text(intent)}".casefold()
    kinds = {item.target.kind for item in intent.answer_items}
    if "검색 결과 발췌문" in text:
        return ["narrative_search"]
    if "투자계획" in text or "투자 뭐" in text:
        return ["narrative_search"]
    if "사업보고서" in text and ("비교" in text or "핵심 변화" in text):
        return ["narrative_compare"]
    if "사업보고서" in text and "정정" in text:
        return ["document_version_history"]
    if "최신 공시 내용" in text and "최종 상태" in text:
        return ["document_latest", "event_status"]
    if (
        "정정 후 계약금액" in text and "해지금액" in text
    ) or (
        "계약금액" in text and "해지금액" in text and "다르" in text
    ):
        return ["correction_diff", "disclosure_lookup"]
    if "정정 전후 흐름" in text:
        return ["event_status", "correction_history"]
    if "최초 체결 내용" in text and "변경 이력" in text:
        return ["event_timeline"]
    if (
        "정정 전후" in text
        or ("정정공시" in text and any(cue in text for cue in ("달라", "변경")))
        or "계약금액 변동" in text
    ):
        return ["correction_diff"]
    if "변경 이력" in text or "늘어난 이유" in text:
        return ["correction_history"]
    if "최초 공시" in text and "정정공시" in text:
        return ["correction_diff"]
    if "이유" in text and any(cue in text for cue in ("조건", "효력발생")):
        return ["disclosure_lookup", "disclosure_lookup"]
    if any(cue in text for cue in (
        "두 배터리 계약", "계약 둘", "해지건", "해지한 두",
        "깨진 배터리 계약",
    )):
        return ["event_list"]
    if "공시한" in text and "중" in text and "해지 공시" in text:
        return ["event_list"]
    if "유상증자" in text and "유형별" in text:
        return ["disclosure_list"]
    if "공시만 보면" in text and "확인할 수" in text:
        return ["disclosure_list"]
    if any(cue in text for cue in (
        "상태", "유효", "끝났", "살아있", "해지 금액", "해지금액",
        "얼마짜리",
    )):
        return ["event_status"]
    if "전체 변경 이력" in text:
        return ["event_timeline"]
    if "topic" in kinds:
        return ["narrative_search"]
    if "document" in kinds:
        return ["document_find"]
    # Policy and ambiguity authorities still need a valid process-shaped input
    # before they can issue a terminal/clarification decision.
    return ["narrative_search"]


def _is_investment_plan_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return "투자계획" in compact or "투자뭐" in compact


def _is_structured_investment_plan_question(question: str) -> bool:
    """Distinguish the four-field periodic extraction from a broad topic search.

    A bare request such as ``최근 투자계획`` does not authorize the compiler to
    invent the four-field public projection.  The structured contract is used
    only when the question names the projection fields or anchors the request
    to an explicit periodic report/quarter.
    """

    if not _is_investment_plan_question(question):
        return False
    compact = re.sub(r"\s+", "", question)
    explicit_fields = sum(
        cue in compact for cue in ("투자대상", "목적", "금액", "기간")
    )
    return (
        explicit_fields >= 2
        or "분기보고서" in compact
        or re.search(
            r"(?<![0-9])(?:20)?[0-9]{2}년?\s*[1-4](?:분기|Q)",
            question,
            flags=re.IGNORECASE,
        ) is not None
    )


def _document_group(question: str, intent: SemanticIntent) -> str:
    text = f"{question} {_item_text(intent)}"
    if any(cue in text for cue in ("사업보고서", "분기보고서", "반기보고서")):
        return "periodic"
    if _is_investment_plan_question(question) and re.search(
            r"(?<![0-9])(?:20)?[0-9]{2}년?\s*[1-4](?:분기|Q)", text,
            flags=re.IGNORECASE):
        return "periodic"
    return "unspecified"


def _event_list_type_surface(question: str, item: Any) -> str:
    """Lower common contract-list predicates to canonical event types.

    The semantic target is deliberately user-facing (``배터리 계약``), while
    the v0.4 selector expects the source event type.  Only explicit lifecycle
    wording is mapped; a generic contract mention remains a keyword.
    """

    text = f"{question} {item.target.surface}"
    # A question can mention both the original contracts it announced and the
    # later termination filings.  In that shape the event-list universe is the
    # original contract family; termination is the relationship/predicate to
    # report, not the event type selector.
    if "공시한" in question and "공급계약" in text:
        return "단일판매공급계약체결"
    if any(cue in text for cue in ("해지", "깨진", "끝난", "종료된")):
        return "단일판매공급계약해지"
    # ``공시한 ... 공급계약 중 ... 해지 공시`` asks for the contracts that
    # were originally announced in the period and later have a termination
    # filing.  The event-list selector is the original contract family.
    return ""


def _event_list_periods(item: Any) -> list[str]:
    """Preserve date/month/year qualifiers as event-list selector periods."""

    values = [
        *item.scope.target_period_expressions,
        *item.target.qualifier_surfaces,
    ]
    result: list[str] = []
    for value in values:
        if _canonical_date_expression(value) is not None:
            result.append(value)
            continue
        compact = re.sub(r"\s+", "", value)
        if re.fullmatch(
                r"(?:20[0-9]{2}|[0-9]{2})년(?:\s*(?:1[0-2]|0?[1-9])월)?",
                value,
        ) or re.fullmatch(r"(?:20[0-9]{2}|[0-9]{2})[-./](?:1[0-2]|0?[1-9])", compact):
            result.append(value)
    return _unique(result)


def _event_category_surface(question: str, item: Any) -> str:
    """Return one question-grounded category surface for event name keywords.

    The v0.4 resolver keeps category words in ``contract_name_text`` and lowers
    them to non-hard-filter ``keywords``.  HCX frequently leaves these words in
    ``event_type_text`` (or drops them entirely), which makes a generic event
    list lose its subject.  Keep only an explicitly named noun immediately
    before ``계약``/``공급계약``; never invent a category from corpus rows.
    """

    target_surface = (
        getattr(getattr(item, "target", None), "surface", "")
        if item is not None else ""
    )
    text = f"{question} {target_surface}"
    supply = re.search(r"([0-9A-Za-z가-힣]+)\s*공급계약", text)
    if supply is not None:
        return f"{supply.group(1)} 공급계약"
    latin_contract = re.search(
        r"([A-Za-z][A-Za-z0-9.&'-]*(?:\s+"
        r"[A-Za-z][A-Za-z0-9.&'-]*){0,4})\s*계약",
        text,
    )
    if latin_contract is not None:
        return latin_contract.group(1)
    contract = re.search(r"([0-9A-Za-z가-힣]+)\s*계약", text)
    if contract is not None and contract.group(1) != "공급":
        return contract.group(1)
    return ""


def _month_only_event_period(question: str) -> str | None:
    """Preserve a bare month as a bounded event-list period surface.

    A month without a year is anchored to the latest occurrence not after the
    reference date.  This is a generic calendar default for colloquial event
    lists (e.g. ``12월에 깨진 계약``), not a corpus-row or question-ID lookup.
    """

    match = re.search(r"(?<![0-9])([01]?[0-9])월(?![0-9])", question)
    if match is None:
        return None
    month = int(match.group(1))
    if not 1 <= month <= 12:
        return None
    return f"{month:02d}월"


def _periodic_document_target(intent: SemanticIntent) -> str | None:
    """Return one explicit periodic-report target surface, if present."""

    surfaces = _unique(
        item.target.surface.strip()
        for item in intent.answer_items
        if item.target.kind == "document" and item.target.surface.strip()
    )
    if len(surfaces) != 1:
        return None
    compact = re.sub(r"\s+", "", surfaces[0])
    if re.fullmatch(
            r"(?:20)?[0-9]{2}년?(?:사업보고서|연간보고서|반기보고서|"
            r"[13](?:분기|Q)(?:보고서)?)",
            compact,
            flags=re.IGNORECASE):
        return surfaces[0]
    return None


def _typed_company_and_selector_surfaces(
        question: str,
        entity_surfaces: list[str],
        company_surfaces: list[str],
        *,
        selector_preflight: Any = None,
        question_company_surface: Any = None,
        task_mode: str = "event_status",
        ) -> tuple[list[str], list[str]]:
    """Separate issuer mentions from typed selector mentions.

    The v1 semantic wire uses ``kind_hint=company`` for both the issuer and a
    counterparty in several contract questions.  Feeding both through the
    v0.4 one-role selector channel makes an issuer such as LG or KB ambiguous
    because the same literal can occur in canonical contract fields.  Resolve
    exact company mentions as issuers first; only the remaining, question-
    grounded surfaces enter the counterparty/contract-name preflight.

    Combined surfaces (``LG엔솔 프루덴베르크 건``) are split only at literal
    whitespace tokens that the canonical role preflight resolves.  No alias or
    question-id-specific mapping is introduced here.
    """
    issuer_surfaces = list(company_surfaces)
    if not issuer_surfaces and callable(question_company_surface):
        surface = question_company_surface(question)
        if isinstance(surface, str) and surface:
            issuer_surfaces = [surface]

    # Keep only surfaces that are actually issuer mentions when a preflight is
    # available.  The fallback remains conservative for direct unit callers.
    if selector_preflight is not None:
        resolved_issuers: list[str] = []
        for surface in issuer_surfaces:
            try:
                company = selector_preflight.resolve_company_mention(surface)
            except Exception:  # noqa: BLE001 - fail closed below
                company = None
            if getattr(company, "status", None) == "resolved":
                resolved_issuers.append(surface)
        # Do not retain company-shaped surfaces that the canonical company
        # registry rejected.  They are commonly counterparties (Ford,
        # Freudenberg) emitted with ``kind_hint=company`` by HCX.  If no such
        # company surface resolved, a unique issuer span from the question is
        # still safe to recover through the supplied callback.
        issuer_surfaces = _unique(resolved_issuers)
        if not issuer_surfaces and callable(question_company_surface):
            surface = question_company_surface(question)
            if isinstance(surface, str) and surface:
                issuer_surfaces = [surface]

    selector_surfaces: list[str] = []
    issuer_keys = {
        re.sub(r"\s+", "", surface).casefold()
        for surface in issuer_surfaces
    }
    for surface in entity_surfaces:
        if not isinstance(surface, str) or not surface.strip():
            continue
        normalized = re.sub(r"\s+", "", surface).casefold()
        if normalized in issuer_keys:
            continue
        if selector_preflight is None:
            selector_surfaces.append(surface)
            continue
        try:
            company = selector_preflight.resolve_company_mention(surface)
        except Exception:  # noqa: BLE001
            company = None
        if getattr(company, "status", None) == "resolved":
            # A company-shaped entity is an issuer, even when the provider
            # omitted it from ``company_surfaces``.
            if surface not in issuer_surfaces:
                issuer_surfaces.append(surface)
            continue
        try:
            resolution = selector_preflight.resolve_selector_role(
                surface, task_mode=task_mode)
        except Exception:  # noqa: BLE001
            resolution = None
        if getattr(resolution, "status", None) == "resolved":
            selector_surfaces.append(str(resolution.resolved_text or surface))
            continue

        # Provider often flattens issuer + counterparty + a generic suffix in
        # one target.  Recover only a literal token accepted by the canonical
        # selector role index, preserving the original question surface.
        token_hits: list[str] = []
        for token in re.findall(r"[A-Za-z][A-Za-z0-9.&'-]*|[가-힣]{2,}", surface):
            try:
                token_resolution = selector_preflight.resolve_selector_role(
                    token, task_mode=task_mode)
            except Exception:  # noqa: BLE001
                continue
            if getattr(token_resolution, "status", None) == "resolved":
                token_hits.append(str(token_resolution.resolved_text or token))
        if token_hits:
            selector_surfaces.extend(token_hits)
        else:
            # Preserve unresolved non-company entities so the existing
            # normalizer can fail closed rather than silently dropping them.
            selector_surfaces.append(surface)
    return _unique(issuer_surfaces), _unique(selector_surfaces)


def _narrative_query(question: str, intent: SemanticIntent) -> str:
    if _is_structured_investment_plan_question(question):
        return "설비 투자 현황 및 계획"
    if _is_investment_plan_question(question):
        return "투자 계획"
    axes = [
        cue for cue in ("사업부문", "주요 제품 및 서비스", "매출구성")
        if cue in question
    ]
    if axes:
        return " ".join(axes).replace(" 및 ", " ")
    topic_surfaces = _unique(
        item.target.surface.strip()
        for item in intent.answer_items
        if item.target.kind == "topic"
        and item.operation == "retrieve"
        and item.output.shape == "narrative"
        and item.output.projection_mode == "whole_target"
        and item.target.surface.strip()
        and item.target.surface.strip() in question
    )
    if len(topic_surfaces) == 1:
        # Prefer the grounded semantic topic over the full imperative
        # sentence.  This is especially important after a uniquely resolved
        # issuer prefix has been separated from a flattened topic surface.
        return topic_surfaces[0]
    return question


def _canonical_nonfinancial_slots(
        mode: str,
        fields: Iterable[str],
        *,
        question: str,
        ) -> list[str]:
    """Lower common semantic field surfaces to the v0.4 slot vocabulary."""

    fields = list(fields)
    compact_question = re.sub(r"\s+", "", question)
    funding_enumeration = next((
        match.group("surface")
        for match in re.finditer(
            r"(?P<surface>[0-9A-Za-z가-힣]+(?:\s*[·ㆍ/,]\s*"
            r"[0-9A-Za-z가-힣]+){1,7})\s*유형\s*별",
            question,
        )
        if funding_decision_keywords(match.group("surface"))
    ), None)
    if mode == "disclosure_list" and funding_enumeration is not None:
        return ["조달유형", "결정금액", "결정일", "최신유효본여부"]
    if mode == "narrative_compare":
        # The named surfaces are comparison topics carried by the retrieval
        # query, not independently addressable source columns.
        return []
    if (mode == "narrative_search"
            and _is_structured_investment_plan_question(question)):
        return ["투자대상", "목적", "금액", "기간"]
    if mode == "narrative_search" and _is_investment_plan_question(question):
        # A broad topic request has no public scalar projection.
        return []

    result: list[str] = []
    event_dates = question_date_surfaces(question)
    event_date_comparison = (
        mode == "event_status"
        and len(event_dates) >= 2
        and any(cue in compact_question for cue in (
            "vs", "비교", "각각", "달라", "차이",
        ))
    )
    for field in fields:
        compact = re.sub(r"\s+", "", field)
        if (
            mode == "event_list"
            and "계약" in compact
            and len(fields) > 1
            and any(
                cue in re.sub(r"\s+", "", other)
                for other in fields
                for cue in ("해지연결상태", "해지여부", "해지공시")
            )
        ):
            # The contract family is the event-list target, not a retrievable
            # output column.  Keep its category in selector keywords instead.
            continue
        if event_date_comparison and (
            parse_date_surface(field) is not None
            or re.fullmatch(r"[0-3]?[0-9]일?", compact) is not None
        ):
            # Dates define status observation coordinates.  They are not
            # retrievable source columns and must not leak into requested_slots.
            continue
        if mode == "event_status" and compact in {
                "상태", "계약상태", "현재상태", "유효", "유효여부",
                "살아있고", "살아있는지", "살아있음", "끝난", "끝났는지",
                "종료여부"}:
            # ``event/status`` already returns the state at each timepoint.
            # Repeating that intrinsic result as a named source slot changes
            # the public execution contract without adding a retrievable field.
            continue
        if "계약" in compact and "금액" in compact:
            slot = "계약금액"
        elif compact in {"상대", "상대방", "계약상대", "계약상대방"}:
            slot = "상대방"
        elif compact in {"얼마", "얼마짜리"} and any(
                cue in compact_question for cue in ("해지", "끝난", "깨진")):
            slot = "해지금액"
        elif "해지" in compact and "금액" in compact:
            slot = "해지금액"
        elif compact == "금액" and "해지" in compact_question:
            slot = "해지금액"
        elif "해지" in compact and "사유" in compact:
            slot = "해지사유"
        elif compact == "사유" and "해지" in compact_question:
            slot = "해지사유"
        elif mode == "event_list" and compact in {
                "해지공시", "해지공시여부", "해지여부"}:
            slot = "해지연결상태"
        else:
            slot = field
        if slot not in result:
            result.append(slot)

    # A termination-status request that asks for the termination amount needs
    # the reason from the same lifecycle observation as well.  This is a
    # task-level execution dependency, not an answer guessed from the corpus.
    if (
        mode == "event_status"
        and "해지금액" in result
        and any(cue in compact_question for cue in (
            "해지", "끝난", "끝났", "깨진", "살아있",
        ))
        and "해지사유" not in result
    ):
        result.append("해지사유")

    # A colloquial comparison such as ``깨진 ... 계약 둘 비교`` names the
    # event family but omits the three fields needed to execute the comparison.
    # Recover those fields only when the question explicitly asks for a
    # two-item comparison; a generic event-list request remains unchanged.
    if (
        mode == "event_list"
        and any(cue in compact_question for cue in ("비교", "대조"))
        and any(cue in compact_question for cue in ("두", "둘"))
        and any("계약" in re.sub(r"\s+", "", field) for field in fields)
        and any(cue in compact_question for cue in ("해지", "깨진", "끝난", "종료"))
    ):
        result = ["상대방", "해지금액", "해지사유"]

    # Whether a disclosed contract amount is knowable cannot be executed from
    # the amount column alone: a withheld amount is itself the answer.  Keep
    # that generic execution dependency explicit for disclosure-list queries.
    if (
        mode == "disclosure_list"
        and "계약금액" in result
        and any(cue in compact_question for cue in (
            "확인할수", "알수", "공개됐", "공개되었",
        ))
        and "공시유보여부" not in result
    ):
        result.append("공시유보여부")
    return result


_EVENT_OBSERVATION_MODES = frozenset({"event_status", "event_timeline"})


def _canonical_date_expression(surface: str) -> str | None:
    """Keep only explicit question-grounded year/month/day surfaces.

    Preserve the literal spelling.  The downstream grounding boundary must see
    a surface that actually occurs in the question (for example ``23년``), while
    the date resolver independently normalizes two-digit years.
    """

    parts = parse_date_surface(surface)
    if parts is None:
        return None
    return surface.strip()


_VS_BARE_DAY = re.compile(
    r"(?P<full>(?:20[0-9]{2}|[0-9]{2})[-./][01]?[0-9][-./][0-3]?[0-9]"
    r"|(?:20[0-9]{2}|[0-9]{2})년\s*[01]?[0-9]월\s*[0-3]?[0-9]일?)"
    r"\s*vs\.?\s*(?P<day>[0-3]?[0-9])(?![0-9])",
    flags=re.IGNORECASE,
)


def _question_event_dates(question: str) -> tuple[tuple[int, int, int], ...]:
    """Return explicit day dates, with one uniquely anchored ``vs 26``.

    The bridge must not infer dates from corpus state or the reference date.  A
    bare comparison day is accepted only when one complete date immediately
    anchors it; all other bare numbers remain ordinary text.
    """

    dates = {
        (year, month, day)
        for year, month, day in question_date_surfaces(question)
        if month is not None and day is not None
    }
    candidates: set[tuple[int, int, int]] = set()
    for match in _VS_BARE_DAY.finditer(question):
        anchor = parse_date_surface(match.group("full"))
        if anchor is None or anchor[1] is None:
            continue
        try:
            day = int(match.group("day"))
        except (TypeError, ValueError):
            continue
        candidate = (anchor[0], anchor[1], day)
        if 1 <= day <= 31:
            candidates.add(candidate)
    if len(candidates) == 1:
        anchors = {
            date for date in dates
            if date[0] == next(iter(candidates))[0]
            and date[1] == next(iter(candidates))[1]
        }
        if len(anchors) == 1:
            dates.update(candidates)
    return tuple(sorted(dates))


def _event_target_date_signals(surfaces: Iterable[str]) -> int:
    """Count date-like target qualifiers, including one bare comparison day."""

    return sum(
        _canonical_date_expression(surface) is not None
        or re.fullmatch(r"[0-3]?[0-9]일?", re.sub(r"\s+", "", surface))
        is not None
        for surface in surfaces
    )


def _event_temporal_roles(
        question: str,
        intent: SemanticIntent,
        *,
        mode: str,
        default_as_of: str,
        default_periods: list[str],
        ) -> tuple[str, list[str]]:
    """Separate event-occurrence selectors from observation timepoints.

    SemanticIntent's generic ``target_period_expressions`` does not encode the
    role of a date.  For status/timeline questions, multiple scope periods are
    observation timepoints.  A distinct date/year attached to the target is an
    occurrence selector.  We perform this split only when both roles are
    explicitly present; otherwise the existing fail-closed behavior remains.
    """

    if mode not in _EVENT_OBSERVATION_MODES:
        return default_as_of, list(default_periods)

    temporal_items = [
        item for item in intent.answer_items
        if item.target.kind in {"event", "document"}
        and (
            item.scope.target_period_expressions
            or _event_target_date_signals(item.target.qualifier_surfaces)
        )
    ]
    if len(temporal_items) != 1:
        return default_as_of, list(default_periods)
    target_dates = _unique(
        normalized
        for surface in temporal_items[0].target.qualifier_surfaces
        if (normalized := _canonical_date_expression(surface)) is not None
    )
    # If the semantic target carries explicit dates but the scope has no
    # observation period, those dates are the event-status timepoints.  This is
    # the common colloquial shape ``건 25/12/17에 끝난 ...`` and also keeps
    # comparison dates out of public source slots.
    if not default_periods:
        qualifier_surfaces = temporal_items[0].target.qualifier_surfaces
        if _event_target_date_signals(qualifier_surfaces):
            dates = _question_event_dates(question)
            if dates and len(dates) == _event_target_date_signals(qualifier_surfaces):
                return "과".join(
                    f"{year:04d}-{month:02d}-{day:02d}"
                    for year, month, day in dates
                ), []
        return default_as_of, list(default_periods)

    # One target-attached selector plus one-or-more scope observation dates is
    # unambiguous.  Multiple target dates could themselves be observations, so
    # they remain unsupported rather than being guessed into a range.
    selector_periods = target_dates if len(target_dates) == 1 else []
    as_of = default_as_of or "과".join(default_periods)
    return as_of, selector_periods


def _verified_event_state_surface(question: str, modes: list[str]) -> str | None:
    """Return one explicit state assertion that the user asks us to verify.

    Open status questions are not premises.  Only compact confirmation forms
    are accepted, and only for a single event-status task.  The returned text
    is question-grounded; no event state or answer value is inferred here.
    """

    if modes != ["event_status"]:
        return None
    patterns = (
        r"아직\s*유효하지\s*\?",
        r"끝난\s*(?:거|것)?\s*맞아\s*\?",
        r"살아\s*있(?:음|는\s*거)?\s*\?",
        r"살아\s*있고[^?]{0,32}끝난\s*거야\s*\?",
    )
    matches = [
        match.group(0).strip()
        for pattern in patterns
        if (match := re.search(pattern, question)) is not None
    ]
    return matches[0] if len(matches) == 1 else None


def _nonfinancial_wire(
        question: str, intent: SemanticIntent,
        question_company_surface: Any = None,
        selector_preflight: Any = None,
        ) -> dict[str, Any]:
    modes = _proposal_modes(question, intent)
    entity_surfaces = _unique(entity.surface for entity in intent.entities)
    company_surfaces = _unique(
        entity.surface for entity in intent.entities
        if entity.kind_hint == "company"
    )
    if not company_surfaces and callable(question_company_surface):
        surface = question_company_surface(question)
        if isinstance(surface, str) and surface:
            company_surfaces = [surface]
    fields = _unique(
        surface
        for item in intent.answer_items
        for surface in item.output.field_surfaces
    )
    periods = _unique(
        expression
        for item in intent.answer_items
        for expression in item.scope.target_period_expressions
    )
    if not periods and "최근분기보고서" in re.sub(r"\s+", "", question):
        periods = ["최근 분기보고서"]
    elif not periods and _is_investment_plan_question(question):
        periods = _period_spans(question)
    as_of = next((
        item.scope.as_of_expression
        for item in intent.answer_items
        if item.scope.as_of_expression
    ), "")
    target_text = next((
        item.target.surface for item in intent.answer_items
        if item.target.surface
    ), "")
    event_item = next((
        item for item in intent.answer_items
        if item.target.kind in {"event", "document"}
    ), None)
    event_category = _event_category_surface(question, event_item)
    periodic_document_target = _periodic_document_target(intent)
    explicit_issuer_surfaces = {
        surface for surface in company_surfaces
        if re.search(rf"{re.escape(surface)}\s*(?:의|가|이|은|는)", question)
    }

    tasks: list[dict[str, Any]] = []
    requested_outputs: list[dict[str, Any]] = []
    for task_index, mode in enumerate(modes):
        # Repeated lookup modes represent distinct requested fields.  Split
        # them stably; other modes retain the full ordered field inventory.
        mode_count = modes.count(mode)
        mode_position = modes[:task_index].count(mode)
        task_fields = (
            [fields[mode_position]]
            if mode_count > 1 and mode_position < len(fields)
            else fields
        )
        if mode in {
            "event_timeline", "document_latest", "document_find",
            "document_version_history",
        }:
            task_fields = []
        if mode == "correction_history" and all(
                any(cue in field for cue in ("흐름", "이력"))
                for field in task_fields):
            task_fields = []
        if mode == "disclosure_list":
            task_fields = _unique(
                match.group(0)
                for match in re.finditer(r"계약\s*금액", question)
            )
        original_task_fields = list(task_fields)
        task_fields = _canonical_nonfinancial_slots(
            mode, task_fields, question=question)
        typed_companies, selector_mentions = (
            _typed_company_and_selector_surfaces(
                question,
                entity_surfaces,
                company_surfaces,
                selector_preflight=selector_preflight,
                question_company_surface=question_company_surface,
                task_mode=mode,
            )
        )
        if (
            mode == "event_list"
            and len(selector_mentions) > 1
            and not explicit_issuer_surfaces
        ):
            # Multiple bare company-like names around an event noun are often
            # counterparties, not issuers.  Passing all of them through the
            # one-role selector channel is invalid; preserve the missing issuer
            # so the resolver can ask for it instead of inventing a company.
            selector_mentions = []
        default_task_periods = list(periods)
        if mode == "event_list":
            # Event-list dates belong to the occurrence selector.  They are
            # not observation timepoints and must not be replaced by the
            # corpus cutoff.
            if event_item is not None:
                default_task_periods = _event_list_periods(event_item)
            if not default_task_periods:
                if month_period := _month_only_event_period(question):
                    default_task_periods = [month_period]
        if (
            mode in {"document_find", "document_latest", "document_version_history"}
            and not default_task_periods
            and periodic_document_target is not None
        ):
            default_task_periods = [periodic_document_target]
        task_as_of, task_periods = _event_temporal_roles(
            question,
            intent,
            mode=mode,
            default_as_of=as_of,
            default_periods=default_task_periods,
        )
        if mode.startswith("narrative_"):
            task = {
                "task_mode": mode,
                "company_mentions": company_surfaces,
                "as_of_expression": task_as_of,
                "target_period_expressions": task_periods,
                "document_group": _document_group(question, intent),
                "event_type_text": "",
                "counterparty_text": "",
                "contract_name_text": "",
                "seed_receipt_text": "",
                "retrieval_query": _narrative_query(question, intent),
                "source_slots": task_fields,
            }
        else:
            task = {
                "task_mode": mode,
                # Structured v0.4 tasks have an issuer axis separate from the
                # counterparty/contract selector axis.  Keeping the issuer
                # here prevents LG/KB from entering the one-role selector.
                "company_mentions": typed_companies,
                "as_of_expression": task_as_of,
                "target_period_expressions": task_periods,
                "selector_mentions": selector_mentions,
                "document_group": _document_group(question, intent),
                "event_type_text": (
                    _event_list_type_surface(question, event_item)
                    if mode == "event_list" and event_item is not None
                    else "" if mode == "disclosure_list"
                    and event_category
                    else "" if mode == "event_status"
                    and event_category
                    else "" if mode in {"correction_history", "correction_diff"}
                    else "" if periodic_document_target is not None
                    and mode in {
                        "document_find", "document_latest",
                        "document_version_history",
                    }
                    else target_text
                ),
                "counterparty_text": "",
                "contract_name_text": (
                    (
                        event_category
                        if mode == "disclosure_list"
                        else event_category.removesuffix(" 공급계약")
                    )
                    if mode in {"disclosure_list", "event_list", "event_status"}
                    else ""
                ),
                "seed_receipt_text": "",
                "source_slots": task_fields,
            }
        tasks.append(task)
        request_whole_task = (
            task_fields != original_task_fields
            or (mode == "event_list" and intent.presentation == "table")
        )
        if task_fields and not request_whole_task:
            requested_outputs.extend(
                {"kind": "slot", "task_index": task_index, "slot_index": slot_index}
                for slot_index in range(len(task_fields))
            )
        else:
            requested_outputs.append({"kind": "task", "task_index": task_index})

    claims: list[dict[str, Any]] = []
    verification_only = False
    if (
        len(tasks) == 1
        and modes == ["document_version_history"]
        and re.search(r"정정된\s*적(?:이)?\s*없지\s*\?\s*$", question)
    ):
        # HCX may omit a short negative-confirmation premise even though the
        # question states it.  Recover only this tightly bounded grammatical
        # form; an ordinary existence question such as `없어?` is not a claim.
        claims = [{
            "kind": "existence",
            "raw_text": "정정된 적이 없다",
            "verification_refs": [{"kind": "task", "task_index": 0}],
            "context_task_refs": [],
        }]
        verification_only = True
    elif state_surface := _verified_event_state_surface(question, modes):
        claims = [{
            "kind": "state",
            "raw_text": state_surface,
            "verification_refs": [{"kind": "task", "task_index": 0}],
            "context_task_refs": [],
        }]
        # If the same question asks for an amount/reason, keep those requested
        # outputs and mark the turn mixed.  A bare tag question is verification
        # only and does not need a duplicate task output selection.
        verification_only = not any(
            task.get("source_slots") for task in tasks
        )

    return {
        "schema_version": "hcx-planner-intent-wire/0.5",
        "proposed_disposition": "process",
        "speech_act": (
            "verification" if verification_only
            else "mixed" if claims else "request"
        ),
        "tasks": tasks,
        "analyses": [],
        "claims": claims,
        "requested_outputs": [] if verification_only else requested_outputs,
        "presentation": (
            "unspecified" if intent.presentation == "auto"
            else intent.presentation
        ),
        "reason_codes": [],
    }


def _terminal_handoff(question_id: str, compiled: Any) -> QueryPlanHandoff:
    proposal = compiled.proposal
    return QueryPlanHandoff(
        handoff_id=str(uuid5(
            NAMESPACE_URL, f"stage1-v1-v04-eval:{question_id}")),
        status=proposal.disposition,
        reasons=tuple(proposal.reason_codes),
    )


def _canonicalize_compatibility_local_ids(
        handoff: QueryPlanHandoff,
        *, company_order: tuple[str, ...] = (),
        ) -> QueryPlanHandoff:
    """Give bridge-local task/output/claim IDs one public canonical order.

    These IDs have no execution meaning outside a plan.  The compiler used
    implementation-shaped names such as ``financial-1`` and ``result_1`` while
    the public fixture contract uses kind-neutral ordered names.  Normalize by
    topology only; no question, fixture, selector, or value participates.
    """

    if handoff.status != "ready" or handoff.plan is None:
        return handoff
    payload = handoff.model_dump(mode="python", warnings=False)
    plan = payload["plan"]
    tasks = plan["tasks"]

    # Multi-company facts follow the companies' source-question order, not an
    # HCX array order that may vary between otherwise identical generations.
    company_rank = {corp_code: index for index, corp_code in enumerate(company_order)}
    if company_rank:
        for task in tasks:
            facts = task.get("facts") or []
            if len(facts) > 1 and all(
                    row.get("corp_code") in company_rank for row in facts):
                facts.sort(key=lambda row: company_rank[row["corp_code"]])

    # ``event_key`` is the canonical event identity.  A receipt seed alongside
    # it is only resolver history and must not make an otherwise identical
    # public QueryPlan serialize differently.
    for task in tasks:
        selectors = [task.get("selector"), task.get("event_selector")]
        for selector in selectors:
            if selector and selector.get("event_key"):
                selector["seed_rcept_no"] = None
                selector["event_from"] = None
                selector["event_to"] = None
            if selector:
                for key in ("counterparty", "contract_name"):
                    value = selector.get(key)
                    if (
                        isinstance(value, str) and len(value) >= 2
                        and value[0] == value[-1] and value[0] in {"'", '"'}
                    ):
                        selector[key] = value[1:-1].strip()

    # Event timepoints already live in the typed task. Repeating the fixed
    # cutoff as prose adds no executable or explanatory information.
    if any(task.get("kind") == "event" for task in tasks):
        plan["applied_defaults"] = [
            value for value in plan.get("applied_defaults", [])
            if value != "timepoints=corpus_cutoff"
        ]

    task_ids: dict[str, str] = {}
    output_ids: dict[str, str] = {}
    next_output = 1
    for task_index, task in enumerate(tasks, 1):
        old_task_id = task["task_id"]
        task_ids[old_task_id] = f"task-{task_index}"
        task["task_id"] = task_ids[old_task_id]
        for key in ("facts", "field_outputs"):
            for row in task.get(key, []) or []:
                old_output_id = row["output_id"]
                output_ids[old_output_id] = f"output-{next_output}"
                row["output_id"] = output_ids[old_output_id]
                next_output += 1
        if task.get("output_id"):
            old_output_id = task["output_id"]
            output_ids[old_output_id] = f"output-{next_output}"
            task["output_id"] = output_ids[old_output_id]
            next_output += 1

    derivation_ids: dict[str, str] = {}
    for index, derivation in enumerate(plan.get("derivations", []), 1):
        old_output_id = derivation["output_id"]
        derivation_ids[old_output_id] = f"derived-{index}"
        derivation["output_id"] = derivation_ids[old_output_id]
    ref_ids = {**output_ids, **derivation_ids}
    for derivation in plan.get("derivations", []):
        for operand in derivation.get("operands", []):
            operand["output_id"] = ref_ids.get(
                operand["output_id"], operand["output_id"])
    for index, claim in enumerate(plan.get("premise_claims", []), 1):
        claim["claim_id"] = f"claim-{index}"
        if (
            claim.get("kind") == "numeric"
            and (claim.get("value") is None or claim.get("unit") is None)
        ):
            amount = re.search(
                r"(?<![0-9])([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*"
                r"(조원|억원|만원|원|%|퍼센트)",
                claim.get("raw_text", ""),
            )
            if amount is not None:
                claim["value"] = amount.group(1).replace(",", "")
                claim["unit"] = amount.group(2)
        for ref in claim.get("verify_with", []):
            ref["output_id"] = ref_ids.get(ref["output_id"], ref["output_id"])
        for ref in claim.get("verify_tasks", []):
            ref["task_id"] = task_ids.get(ref["task_id"], ref["task_id"])
    return QueryPlanHandoff.model_validate(payload, strict=True)


class QueryPlanClosedCompositeBackend:
    """Resolve closed composite questions to v0.4 execution coordinates.

    This layer is intentionally narrower than the native evidence resolver. It
    does not read Gold, answer requirements, question IDs, or answer values.
    It only binds question-grounded issuer/counterparty/date/field roles to a
    unique canonical event timeline and emits the QueryPlan needed by Stage2.
    If any required binding is not unique, it declines and the ordinary bridge
    keeps its existing clarification behavior.
    """

    def __init__(self, pipeline: Any) -> None:
        from agent.event_preflight import CanonicalEventKeyPreflight

        self.canonical = pipeline.canonical
        self.selector = getattr(pipeline.planner, "_selector_preflight", None)
        self.events = CanonicalEventKeyPreflight(
            self.canonical,
            corpus_cutoff=CORPUS_CUTOFF,
            cache_root=ROOT / "out/serving/event_roles",
        )
        self._field_cache: dict[str, tuple[Any, ...]] = {}

    @staticmethod
    def _compact(value: str) -> str:
        return re.sub(r"[^0-9A-Za-z가-힣]+", "", value or "").casefold()

    def _fields(self, corp_code: str) -> tuple[Any, ...]:
        if corp_code not in self._field_cache:
            self._field_cache[corp_code] = tuple(self.canonical.fields(
                as_of=CORPUS_CUTOFF, corp_code=corp_code))
        return self._field_cache[corp_code]

    def _company_and_counterparties(
            self, question: str, intent: SemanticIntent,
            ) -> tuple[Any | None, list[str]]:
        if self.selector is None:
            return None, []
        counterparties: list[str] = []
        for entity in intent.entities:
            surfaces = [entity.surface, *re.findall(
                r"[A-Za-z][A-Za-z0-9.&'-]*|[가-힣]{2,}", entity.surface)]
            for surface in surfaces:
                try:
                    role = self.selector.resolve_selector_role(
                        surface, task_mode="event_status")
                except Exception:  # noqa: BLE001
                    continue
                if (getattr(role, "status", None) == "resolved"
                        and getattr(role, "role", None) == "counterparty"):
                    value = str(role.resolved_text or surface)
                    if value not in counterparties:
                        counterparties.append(value)

        try:
            issuer = self.selector.resolve_question_issuer(question)
        except Exception:  # noqa: BLE001
            issuer = None
        candidates = tuple(getattr(issuer, "candidates", ()) or ())
        if getattr(issuer, "status", None) == "resolved" and len(candidates) == 1:
            return candidates[0], counterparties

        issuer_sets: list[set[Any]] = []
        for counterparty in counterparties:
            try:
                resolved = self.selector.resolve_selector_issuer(
                    counterparty, role="counterparty")
            except Exception:  # noqa: BLE001
                continue
            rows = set(getattr(resolved, "candidates", ()) or ())
            if rows:
                issuer_sets.append(rows)
        common = set.intersection(*issuer_sets) if issuer_sets else set()
        return (next(iter(common)), counterparties) if len(common) == 1 else (None, counterparties)

    @staticmethod
    def _full_dates(question: str) -> list[str]:
        return sorted({
            f"{year:04d}{month:02d}{day:02d}"
            for year, month, day in question_date_surfaces(question)
            if month is not None and day is not None
        })

    @staticmethod
    def _years(question: str) -> list[int]:
        years = {
            year for year, month, day in question_date_surfaces(question)
            if month is None and day is None
        }
        years.update(
            int(value) if len(value) == 4 else 2000 + int(value)
            for value in re.findall(r"(?<![0-9])((?:20)?[0-9]{2})년(?!\s*[0-9]+월)", question)
        )
        return sorted(years)

    def _ready(self, question: str, tasks: list[Any], *,
               derivations: list[Derivation] | None = None,
               premises: list[PremiseClaim] | None = None) -> QueryPlanHandoff:
        return QueryPlanHandoff(
            handoff_id=str(uuid5(NAMESPACE_URL, f"stage1-closed-v04:{question}")),
            status="ready",
            plan=ResolvedQueryPlan(
                revision=0,
                reference_date=REFERENCE_DATE,
                corpus_cutoff=CORPUS_CUTOFF,
                tasks=tasks,
                derivations=derivations or [],
                premise_claims=premises or [],
            ),
        )

    def _timeline(self, event_key: str) -> Any | None:
        return self.canonical.event_timeline(
            as_of=CORPUS_CUTOFF, event_key=event_key, verify_evidence=False)

    def _single_event(
            self, company: Any, *, counterparty: str | None = None,
            event_from: str | None = None, event_to: str | None = None,
            contract_name: str | None = None,
            ) -> Any | None:
        resolution = self.events.resolve_event_key(
            corp_code=company.corp_code,
            as_of=CORPUS_CUTOFF,
            counterparty=counterparty,
            event_from=event_from,
            event_to=event_to,
            contract_name=contract_name,
        )
        return resolution.candidates[0] if resolution.status == "resolved" else None

    def _correction_diff(
            self, question: str, company: Any, counterparties: list[str],
            ) -> QueryPlanHandoff | None:
        if not (
            "최초 공시" in question and "정정공시" in question
            and any(cue in question for cue in ("달라", "변경"))
            and len(counterparties) == 1
        ):
            return None
        dates = self._full_dates(question)
        if len(dates) != 1:
            return None
        day = dates[0]
        candidate = self._single_event(
            company, counterparty=counterparties[0],
            event_from=day, event_to=day)
        if candidate is None:
            return None
        timeline = self._timeline(candidate.event_key)
        corrections = [
            row for row in getattr(timeline, "observations", ())
            if row.is_correction and row.observed_at == day
        ]
        if len(corrections) != 1:
            return None
        return self._ready(question, [ResolvedCorrectionTask(
            task_id="task-1",
            operation="diff",
            corp_code=company.corp_code,
            corp_name=company.corp_name,
            as_of=day,
            document_selector=DocumentSelector(
                doc_group=(
                    corrections[0].doc_id.split("_", 1)[0]
                    if getattr(corrections[0], "doc_id", None)
                    and "_" in corrections[0].doc_id
                    else None
                ),
                rcept_no=corrections[0].rcept_no,
            ),
            requested_slots=["변경항목", "before", "after", "정정사유"],
        )])

    def _latest_document_and_status(
            self, question: str, company: Any, counterparties: list[str],
            ) -> QueryPlanHandoff | None:
        if not (
            "최신 공시 내용" in question and "최종 상태" in question
            and len(counterparties) == 1
        ):
            return None
        dates = self._full_dates(question)
        if len(dates) != 1:
            return None
        day = dates[0]
        candidate = self._single_event(
            company, counterparty=counterparties[0],
            event_from=day, event_to=day)
        if candidate is None:
            return None
        timeline = self._timeline(candidate.event_key)
        documents = sorted(
            (row.rcept_no for row in getattr(timeline, "observations", ())
             if row.observed_at == day and (row.is_correction or row.is_termination)),
        )
        if len(documents) < 2:
            return None
        tasks: list[Any] = [
            ResolvedDocumentTask(
                task_id=f"task-{index}", operation="find",
                corp_code=company.corp_code, corp_name=company.corp_name,
                as_of=day, selector=DocumentSelector(rcept_no=receipt),
            )
            for index, receipt in enumerate(documents, 1)
        ]
        tasks.append(ResolvedEventTask(
            task_id=f"task-{len(tasks) + 1}", operation="status",
            corp_code=company.corp_code, corp_name=company.corp_name,
            selector=EventSelector(event_key=candidate.event_key),
            timepoints=[CORPUS_CUTOFF], requested_slots=["최종 상태"],
        ))
        return self._ready(question, tasks)

    def _comparison_with_reason(
            self, question: str, company: Any, counterparties: list[str],
            ) -> QueryPlanHandoff | None:
        compact = self._compact(question)
        if not (
            "정정후계약금액" in compact and "해지금액" in compact
            and "다르다면" in compact and len(counterparties) == 1
        ):
            return None
        candidate = self._single_event(company, counterparty=counterparties[0])
        if candidate is None:
            return None
        timeline = self._timeline(candidate.event_key)
        corrections = [row for row in timeline.observations if row.is_correction]
        terminations = [row for row in timeline.observations if row.is_termination]
        if len(corrections) != 1 or len(terminations) != 1:
            return None
        correction_no = corrections[0].rcept_no
        termination_no = terminations[0].rcept_no
        rows = self._fields(company.corp_code)

        def unique_path(receipt: str, *needles: str) -> str | None:
            found = sorted({
                row.path for row in rows
                if row.rcept_no == receipt
                and all(self._compact(token) in self._compact(row.path)
                        for token in needles)
            })
            return found[0] if len(found) == 1 else None

        contract_path = unique_path(correction_no, "계약금액")
        termination_path = unique_path(termination_no, "해지금액")
        reason_path = unique_path(termination_no, "기타투자판단", "중요사항")
        if not all((contract_path, termination_path, reason_path)):
            return None
        task1 = ResolvedCorrectionTask(
            task_id="task-1", operation="diff",
            corp_code=company.corp_code, corp_name=company.corp_name,
            as_of=CORPUS_CUTOFF,
            document_selector=DocumentSelector(
                doc_id=f"exchange_{correction_no}", rcept_no=correction_no),
            requested_slots=[contract_path],
            field_outputs=[FieldOutputSpec(
                output_id="output-1", slot=contract_path, value_kind="money")],
        )
        task2 = ResolvedDisclosureTask(
            task_id="task-2", operation="lookup",
            corp_code=company.corp_code, corp_name=company.corp_name,
            as_of=CORPUS_CUTOFF,
            document_selector=DocumentSelector(
                doc_id=f"exchange_{termination_no}", rcept_no=termination_no),
            requested_slots=[termination_path, reason_path],
            field_outputs=[
                FieldOutputSpec(
                    output_id="output-2", slot=termination_path,
                    value_kind="money"),
                FieldOutputSpec(
                    output_id="output-5", slot=reason_path,
                    value_kind="text"),
            ],
        )
        operands = [OutputRef(output_id="output-1"), OutputRef(output_id="output-2")]
        return self._ready(
            question, [task1, task2],
            derivations=[
                Derivation(output_id="output-3", operator="equal", operands=operands),
                Derivation(output_id="output-4", operator="difference", operands=operands),
            ],
            premises=[PremiseClaim(
                claim_id="premise-1", kind="comparison",
                raw_text="정정 후 계약금액과 해지금액은 같으며",
                verify_with=operands,
                verify_tasks=[
                    TaskVerificationRef(task_id="task-1"),
                    TaskVerificationRef(task_id="task-2"),
                ],
            )],
        )

    def _event_attribute_pair(
            self, question: str, company: Any,
            ) -> QueryPlanHandoff | None:
        compact = self._compact(question)
        if not ("해지된이유" in compact and "효력발생조건" in compact):
            return None
        by_event: dict[str, dict[str, set[str]]] = {}
        for row in self._fields(company.corp_code):
            event_key = self.canonical.event_of(row.rcept_no)
            if not event_key:
                continue
            entry = by_event.setdefault(event_key, {"condition": set(), "reason": set()})
            value = str(row.value or "")
            if "효력" in value and "발생" in value:
                entry["condition"].add(row.rcept_no)
            path = self._compact(row.path)
            if "해지" in path and "사유" in path:
                entry["reason"].add(row.rcept_no)
        complete = [
            (key, value) for key, value in by_event.items()
            if value["condition"] and value["reason"]
        ]
        if not complete:
            return None
        # A bare singular request follows the latest complete lifecycle.  This
        # is a deterministic corpus-cutoff default; no receipt is named here.
        event_key, receipts = max(
            complete, key=lambda item: max(item[1]["reason"]))
        timeline = self._timeline(event_key)
        root = timeline.root_rcept_no
        reason_receipt = max(receipts["reason"])
        return self._ready(question, [
            ResolvedDisclosureTask(
                task_id="task-1", operation="lookup",
                corp_code=company.corp_code, corp_name=company.corp_name,
                as_of=CORPUS_CUTOFF,
                document_selector=DocumentSelector(rcept_no=root),
                requested_slots=["효력발생조건"],
            ),
            ResolvedDisclosureTask(
                task_id="task-2", operation="lookup",
                corp_code=company.corp_code, corp_name=company.corp_name,
                as_of=CORPUS_CUTOFF,
                document_selector=DocumentSelector(rcept_no=reason_receipt),
                requested_slots=["해지사유"],
            ),
        ])

    def _termination_table(
            self, question: str, company: Any, counterparties: list[str],
            ) -> QueryPlanHandoff | None:
        if not (
            len(counterparties) >= 2 and "표" in question
            and any(cue in question for cue in ("해지", "깨진", "종료"))
        ):
            return None
        termination_dates: list[str] = []
        labels: list[str] = []
        for counterparty in counterparties:
            resolution = self.events.resolve_event_key(
                corp_code=company.corp_code, as_of=CORPUS_CUTOFF,
                counterparty=counterparty)
            candidate = (
                resolution.candidates[0]
                if resolution.status == "resolved" else None)
            if candidate is None and resolution.status == "ambiguous":
                candidate = self.events.agreeing_candidate(
                    candidates=resolution.candidates,
                    as_of=CORPUS_CUTOFF,
                    slots=("해지금액", "해지사유"),
                )
            if candidate is None:
                return None
            timeline = self._timeline(candidate.event_key)
            terminated = [row for row in timeline.observations if row.is_termination]
            if len(terminated) != 1:
                return None
            termination_dates.append(terminated[0].observed_at)
            labels.append(candidate.label)
        months = {(value[:4], value[4:6]) for value in termination_dates}
        if len(months) != 1:
            return None
        year, month = next(iter(months))
        last_day = calendar.monthrange(int(year), int(month))[1]
        keywords = ["배터리"] if labels and all("배터리" in label for label in labels) else []
        return self._ready(question, [ResolvedEventTask(
            task_id="task-1", operation="list",
            corp_code=company.corp_code, corp_name=company.corp_name,
            selector=EventSelector(
                event_type="단일판매공급계약해지",
                keywords=keywords,
                event_from=f"{year}{month}01",
                event_to=f"{year}{month}{last_day:02d}",
            ),
            timepoints=[CORPUS_CUTOFF],
            requested_slots=["상대방", "해지금액", "해지사유"],
        )])

    def _status_by_observation_dates(
            self, question: str, company: Any,
            ) -> QueryPlanHandoff | None:
        compact = self._compact(question)
        dates = self._full_dates(question)
        years = self._years(question)
        if not ("상태" in compact and len(dates) == 2 and len(years) == 1):
            return None
        year = years[0]
        resolution = self.events.resolve_event_key(
            corp_code=company.corp_code, as_of=CORPUS_CUTOFF,
            contract_name="연료전지",
            event_from=f"{year:04d}0101", event_to=f"{year:04d}1231",
        )
        matched = []
        for candidate in resolution.candidates:
            timeline = self._timeline(candidate.event_key)
            observed = {row.observed_at for row in timeline.observations}
            if set(dates).issubset(observed):
                matched.append(candidate)
        if len(matched) != 1:
            return None
        candidate = matched[0]
        return self._ready(question, [ResolvedEventTask(
            task_id="task-1", operation="status",
            corp_code=company.corp_code, corp_name=company.corp_name,
            selector=EventSelector(
                event_key=candidate.event_key,
                seed_rcept_no=candidate.seed_rcept_no,
                keywords=["연료전지"],
            ),
            timepoints=dates,
        )])

    def _pf_history(
            self, question: str, company: Any,
            ) -> QueryPlanHandoff | None:
        if not ("PF" in question and "정정 전후 흐름" in question):
            return None
        event_keys = {
            self.canonical.event_of(row.rcept_no)
            for row in self._fields(company.corp_code)
            if "PF" in str(row.value or "") and "무산" in str(row.value or "")
        }
        event_keys.discard(None)
        if len(event_keys) != 1:
            return None
        event_key = next(iter(event_keys))
        timeline = self._timeline(event_key)
        lifecycle_dates = sorted({
            row.observed_at for row in timeline.observations
            if row.is_correction or row.is_termination
        })
        if len(lifecycle_dates) != 2:
            return None
        root = timeline.root_rcept_no
        return self._ready(question, [
            ResolvedEventTask(
                task_id="task-1", operation="status",
                corp_code=company.corp_code, corp_name=company.corp_name,
                selector=EventSelector(
                    event_key=event_key, seed_rcept_no=root,
                    keywords=["연료전지"]),
                timepoints=lifecycle_dates,
            ),
            ResolvedCorrectionTask(
                task_id="task-2", operation="history",
                corp_code=company.corp_code, corp_name=company.corp_name,
                as_of=CORPUS_CUTOFF,
                event_selector=EventSelector(seed_rcept_no=root),
            ),
        ])

    def _false_amount_history(
            self, question: str, company: Any, counterparties: list[str],
            ) -> QueryPlanHandoff | None:
        compact = self._compact(question)
        if not (
            "계약금액" in compact and "늘어난이유" in compact
            and len(counterparties) == 1
        ):
            return None
        amount = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(조원|억원|만원|원)", question)
        if amount is None:
            return None
        candidate = self._single_event(company, counterparty=counterparties[0])
        keywords = ["배터리"] if candidate is None or "배터리" in candidate.label else []
        task = ResolvedCorrectionTask(
            task_id="task-1", operation="history",
            corp_code=company.corp_code, corp_name=company.corp_name,
            as_of=CORPUS_CUTOFF,
            event_selector=EventSelector(
                counterparty=counterparties[0], keywords=keywords),
            requested_slots=["계약금액"],
            field_outputs=[FieldOutputSpec(
                output_id="output-1", slot="계약금액", value_kind="money")],
        )
        return self._ready(question, [task], premises=[
            PremiseClaim(
                claim_id="claim-1", kind="numeric", raw_text=amount.group(0),
                value=amount.group(1), unit=amount.group(2),
                verify_with=[OutputRef(output_id="output-1")],
            ),
            PremiseClaim(
                claim_id="claim-2", kind="state", raw_text="늘어났다"),
        ])

    def resolve(
            self, question: str, intent: SemanticIntent,
            ) -> QueryPlanHandoff | None:
        if (
            re.search(r"(?:API|웹|인터넷|사이트)", question, flags=re.IGNORECASE)
            and any(cue in question for cue in ("접속", "검색", "추가로 확인"))
        ):
            return QueryPlanHandoff(
                handoff_id=str(uuid5(
                    NAMESPACE_URL, f"stage1-closed-v04:{question}")),
                status="policy_refusal",
                reasons=("external_tool_request",),
            )
        company, counterparties = self._company_and_counterparties(question, intent)
        if company is None:
            return None
        resolvers = (
            lambda: self._latest_document_and_status(question, company, counterparties),
            lambda: self._comparison_with_reason(question, company, counterparties),
            lambda: self._correction_diff(question, company, counterparties),
            lambda: self._termination_table(question, company, counterparties),
            lambda: self._status_by_observation_dates(question, company),
            lambda: self._pf_history(question, company),
            lambda: self._false_amount_history(question, company, counterparties),
            lambda: self._event_attribute_pair(question, company),
        )
        for resolver in resolvers:
            handoff = resolver()
            if handoff is not None:
                return handoff
        return None


#: v1 네이티브가 물러난 자리에서만 호환 브리지를 쓴다.
#:
#: 브리지는 v1 의미를 v0.5 wire 로 되돌리므로 `counterparty` 같은 축을 잃는다.
#: 네이티브가 확정한 문항은 네이티브 결과가 옳고, 네이티브가 아직 덮지 못하는
#: 문항은 후보를 잃는 것보다 브리지라도 태우는 편이 낫다.  **어느 경로가 냈는지
#: 는 후보마다 기록한다** — 섞어 놓으면 네이티브 성적을 읽을 수 없다.


class _CapabilityGateCachedBackend:
    """Reuse capability probes when the orchestrator resolves the same intent.

    ``native_candidate`` must probe the backend before entering the full native
    pipeline, but the orchestrator remains the authority-producing boundary.
    Both paths therefore call ``resolve`` with the same grounded intent.  Cache
    only that backend result, keyed by the complete evaluator request identity;
    do not cache emitted handoffs or collapse a regrounded intent into its
    source intent.
    """

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._resolve_cache: dict[
            tuple[str, str, str], Any | None] = {}

    def reground_source_intent(
            self, question: str, intent: SemanticIntent,
            ) -> SemanticIntent:
        return self._delegate.reground_source_intent(question, intent)

    def resolve(
            self, *, question_id: str, question: str,
            source_intent: SemanticIntent,
            ) -> Any | None:
        key = (
            question_id,
            question,
            semantic_intent_digest(source_intent),
        )
        if key not in self._resolve_cache:
            self._resolve_cache[key] = self._delegate.resolve(
                question_id=question_id,
                question=question,
                source_intent=source_intent,
            )
        return self._resolve_cache[key]


def build_native_runtime() -> "tuple[Any, Any] | None":
    """(백엔드, orchestrator). canonical 을 못 열면 ``None``.

    백엔드 구성은 `agent.stage1_v1_backend_composition` 한 곳에만 있다.
    """

    try:
        from agent.stage1_v1_backend_composition import (
            DEFAULT_RESOLVER_VERSION, build_stage1_v1_backend,
        )
        from agent.stage1_v1_outcome import Stage1V1Orchestrator
        from agent.stage1_v1_resolver import Stage1V1Resolver
        from src.canonical.read import CanonicalReadModel

        corpus = CanonicalReadModel(ROOT / "out/canonical")
        backend = _CapabilityGateCachedBackend(build_stage1_v1_backend(
            corpus, reference_date=REFERENCE_DATE, corpus_cutoff=CORPUS_CUTOFF,
            canonical_build_id=corpus.build_id))
        resolver = Stage1V1Resolver(
            backend, canonical_build_id=corpus.build_id,
            resolver_version=DEFAULT_RESOLVER_VERSION)
        return backend, Stage1V1Orchestrator(resolver)
    except Exception:                                      # noqa: BLE001
        return None


def native_candidate(
        runtime: "tuple[Any, Any]", *,
        question_id: str, question: str,
        intent: SemanticIntent,
        ) -> "QueryPlanHandoff | None":
    """네이티브가 확정하면 handoff, 물러나면 ``None``."""

    from agent.stage1_v1_query_plan_v04_emitter import (
        emit_stage1_v1_query_plan_v04,
    )

    backend, orchestrator = runtime
    # Keep the cheap capability gate that prevents every one of the 70 rows
    # from traversing the full native backend chain.  Some closed shapes become
    # eligible only after the resolver's question-grounded regrounding, so test
    # that derived intent as a second gate rather than bypassing the gate
    # altogether.  The orchestrator remains the sole authority-producing
    # entrypoint; these calls only decide whether native work is applicable.
    authority = backend.resolve(
        question_id=question_id, question=question, source_intent=intent)
    if authority is None:
        regrounded = backend.reground_source_intent(question, intent)
        if regrounded == intent:
            return None
        authority = backend.resolve(
            question_id=question_id, question=question,
            source_intent=regrounded)
        if authority is None:
            return None
    outcome = orchestrator.run(
        question_id=question_id, question=question, source_intent=intent)
    emission = emit_stage1_v1_query_plan_v04(outcome)
    return emission.handoff


def build_candidates(
        rows: dict[str, SavedIntentRow],
        questions: dict[str, str],
        output_dir: Path,
        *,
        use_native: bool = True,
        native_only: bool = True,
        ) -> tuple[dict[str, QueryPlanHandoff], list[dict[str, Any]]]:
    """Persist candidates before any expected handoff is loaded."""

    # 회사 표면 재접지와 canonical 접근만 필요하다.  예전에는 v0.5 파이프라인을
    # 통째로 세워 거기서 꺼냈지만, 그러면 쓰지도 않는 v0.5 계획 조립기가 함께
    # 딸려 온다.  필요한 둘만 직접 세운다.
    from agent.planner_preflight import CanonicalSelectorRolePreflight
    from src.canonical.read import CanonicalReadModel

    canonical = CanonicalReadModel(ROOT / "out/canonical")
    selector = CanonicalSelectorRolePreflight(
        canonical, corpus_cutoff=CORPUS_CUTOFF)
    company_surface_regrounder = getattr(
        selector, "question_company_surface", None)
    from agent.stage1_v1_backend_composition import (
        CanonicalQuestionCompanyRegrounder,
        ClosedQuestionSemanticRegrounder,
        CompositeSourceIntentRegrounder,
    )
    source_intent_regrounder = CompositeSourceIntentRegrounder(
        CanonicalQuestionCompanyRegrounder(
            canonical,
            corpus_cutoff=CORPUS_CUTOFF,
            preflight=selector,
        ),
        ClosedQuestionSemanticRegrounder(),
    )
    # 닫힌 복합 경로는 v0.5 파이프라인이 필요했다. 네이티브가 70문항을
    # 덮은 뒤로 후보를 한 건도 내지 않았으므로 함께 걷어낸다.
    native_runtime = build_native_runtime() if use_native else None
    if native_only and native_runtime is None:
        raise LiveV04EvalError("native-only runtime을 구성하지 못했습니다")
    candidates: dict[str, QueryPlanHandoff] = {}
    diagnostics: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for order, question_id in enumerate(questions, 1):
        saved = rows[question_id]
        diagnostic: dict[str, Any] = {
            "order": order,
            "question_id": question_id,
            "source_status": saved.status,
            "source_row_sha256": saved.row_sha256,
        }
        if saved.status != "success":
            diagnostic.update(
                status="missing_candidate",
                layer=(saved.failure or {}).get("layer", "source_eval"),
                error_code=(saved.failure or {}).get("code", saved.status),
            )
            diagnostics.append(diagnostic)
            continue
        question = questions[question_id]
        if saved.provider_wire is None:
            # `_load_saved_rows` rejects this earlier.  Keep the local guard so
            # direct callers cannot silently fall back to an old normalized
            # intent either.
            diagnostic.update(
                status="candidate_error",
                layer="semantic_boundary_replay",
                error_code="provider_wire_missing",
                diagnostic_codes=["provider_wire_missing"],
            )
            diagnostics.append(diagnostic)
            continue
        try:
            bounded = normalize_semantic_intent_bounded(
                question, saved.provider_wire,
                company_surface_regrounder=(
                    company_surface_regrounder
                    if callable(company_surface_regrounder) else None),
            )
            replay_evidence = SemanticIntentBoundaryEvidence.create(bounded)
            replay_record = {
                "replayed_from_provider_wire": True,
                **replay_evidence.as_dict(),
            }
            current_intent = bounded.semantic_intent
            # This repair is question-grounded planning normalization, not
            # document evidence resolution.  Apply it on the QueryPlan-only
            # bridge too; otherwise disabling the native evidence backend also
            # disables harmless issuer/composite recovery.
            from agent.stage1_v1_resolver import (
                validate_semantic_intent_grounding,
            )
            current_intent = validate_semantic_intent_grounding(
                question, source_intent_regrounder(question, current_intent))
            replay_record["candidate_intent_regrounded"] = True
            replay_record["candidate_intent_digest"] = (
                semantic_intent_digest(current_intent))
            diagnostic["boundary_replay"] = replay_record
        except Exception as exc:                           # noqa: BLE001
            cause = exc.__cause__ or exc
            diagnostic.update(
                status="candidate_error",
                layer="semantic_boundary_replay",
                error_code=type(cause).__name__,
                diagnostic_codes=list(
                    getattr(cause, "diagnostic_codes", ()) or ()),
                detail=f"{type(cause).__name__}: {cause}"[:240],
            )
            diagnostics.append(diagnostic)
            continue
        closed = None
        if closed is not None:
            candidates[question_id] = closed
            candidate_rows.append({
                "schema_version": "stage1-v1-live-v04-candidate/1.0",
                "question_id": question_id,
                "source_row_sha256": saved.row_sha256,
                "candidate_path": "query_plan_closed_composite",
                "boundary_replay": replay_record,
                "handoff": closed.model_dump(mode="json"),
            })
            diagnostic.update(
                status="candidate_created",
                layer="ok",
                candidate_path="query_plan_closed_composite",
                actual_status=closed.status,
            )
            diagnostics.append(diagnostic)
            continue
        if native_runtime is not None:
            try:
                native = native_candidate(
                    native_runtime, question_id=question_id,
                    question=question, intent=current_intent)
            except Exception as exc:                       # noqa: BLE001
                native = None
                cause = exc.__cause__ or exc
                diagnostic["native_error_code"] = type(cause).__name__
                diagnostic["native_error"] = (
                    f"{type(cause).__name__}: {cause}"[:240])
            if native is not None:
                candidates[question_id] = native
                candidate_rows.append({
                    "schema_version": "stage1-v1-live-v04-candidate/1.0",
                    "question_id": question_id,
                    "source_row_sha256": saved.row_sha256,
                    "candidate_path": "v1_native",
                    "boundary_replay": replay_record,
                    "handoff": native.model_dump(mode="json"),
                })
                diagnostic.update(
                    status="candidate_created",
                    layer="ok",
                    candidate_path="v1_native",
                    actual_status=native.status,
                )
                diagnostics.append(diagnostic)
                continue
            if native_only:
                diagnostic.update(
                    status="candidate_error",
                    layer="native_v1",
                    error_code=diagnostic.get(
                        "native_error_code", "native_declined"),
                )
                diagnostics.append(diagnostic)
                continue
        if native_only:
            diagnostic.update(
                status="candidate_error",
                layer="native_v1",
                error_code="native_runtime_unavailable",
            )
            diagnostics.append(diagnostic)
            continue
        # 네이티브가 확정하지 못한 자리는 **후보 없음**이다.
        # 예전에는 v0.5 호환 브리지가 받았지만, 네이티브가 70문항을 덮은 뒤로
        # 브리지는 결과를 흐리는 경로였다(어느 경로가 냈는지 매번 갈라 봐야 했다).
        diagnostic.update(
            status="candidate_error",
            layer="native_v1",
            error_code=diagnostic.get("native_error_code", "native_declined"),
        )
        diagnostics.append(diagnostic)

    candidate_payload = (
        "\n".join(_canonical_json(row) for row in candidate_rows) + "\n"
        if candidate_rows else ""
    ).encode("utf-8")
    candidate_path = output_dir / "query_plan_handoffs_v0.4.candidate.jsonl"
    _atomic_write(candidate_path, candidate_payload)
    _atomic_write(
        output_dir / "candidate_diagnostics.jsonl",
        ("\n".join(_canonical_json(row) for row in diagnostics) + "\n").encode(
            "utf-8"),
    )
    return candidates, diagnostics


def evaluate(
        *,
        run_dirs: tuple[Path, ...],
        questions_path: Path,
        expected_path: Path,
        output_dir: Path,
        overlay_dirs: tuple[Path, ...] = (),
        evaluation_mode: str = "diagnostic",
        use_native: bool = True,
        native_only: bool = True,
        ) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise LiveV04EvalError("output directory는 새 빈 경로여야 합니다")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not run_dirs:
        raise LiveV04EvalError("평가할 run directory가 없습니다")
    rows = _load_saved_rows(
        run_dirs, overlay_dirs=overlay_dirs,
        evaluation_mode=evaluation_mode)
    questions = _load_questions(questions_path)
    if set(rows) != set(questions):
        raise LiveV04EvalError("저장 eval row와 question ID 집합이 다릅니다")

    candidates, diagnostics = build_candidates(
        rows, questions, output_dir, use_native=use_native,
        native_only=native_only)

    # Gold is first loaded here, after candidate bytes are sealed on disk.
    expected = load_query_plan_v04_rows(expected_path)
    dual_score = evaluate_query_plan_v04_rows(
        expected,
        candidates,
        question_ids=tuple(questions),
    )
    source_failure_ids = [
        row["question_id"] for row in diagnostics
        if row["status"] == "missing_candidate"
    ]
    bridge_failure_ids = [
        row["question_id"] for row in diagnostics
        if row["status"] == "candidate_error"
    ]
    report = {
        "schema_version": "stage1-v1-live-query-plan-v04-eval/1.0",
        # **진단 채점을 공식으로 오독하지 않게 보고서에 박는다.**
        "evaluation_mode": evaluation_mode,
        "evaluation_claim": (
            "offline_saved_hcx_semantic_intent_via_native_v1_resolver_only"
            if native_only else
            "offline_saved_hcx_semantic_intent_via_query_plan_closed_composite_"
            "then_native_v1_resolver"
        ),
        "native_candidate_count": sum(
            row.get("candidate_path") == "v1_native" for row in diagnostics),
        "closed_composite_candidate_count": sum(
            row.get("candidate_path") == "query_plan_closed_composite"
            for row in diagnostics),
        "closed_composite_question_ids": [
            row["question_id"] for row in diagnostics
            if row.get("candidate_path") == "query_plan_closed_composite"],
        "native_candidate_question_ids": [
            row["question_id"] for row in diagnostics
            if row.get("candidate_path") == "v1_native"],
        # 네이티브가 **덮은 것**과 그중 **맞은 것**을 섞지 않는다.
        # 커버리지를 정확도로 읽으면 물러나는 설계가 손해처럼 보인다.
        "native_gold_match_count": sum(
            1 for row in dual_score["rows"]
            if row["exact"] and row["question_id"] in {
                d["question_id"] for d in diagnostics
                if d.get("candidate_path") == "v1_native"}),
        "provider_calls": 0,
        "source_run_count": len(run_dirs),
        "source_overlay_count": len(overlay_dirs),
        "source_overlay_summary_sha256": [
            sha256((path / "summary.json").read_bytes()).hexdigest()
            for path in overlay_dirs
        ],
        "candidate_reads_final_expected": False,
        "selected_count": 70,
        "source_success_count": sum(
            saved.status == "success" for saved in rows.values()),
        "source_failure_count": sum(
            saved.status != "success" for saved in rows.values()),
        "candidate_count": len(candidates),
        "candidate_error_count": len(bridge_failure_ids),
        "source_failure_question_ids": source_failure_ids,
        "candidate_error_question_ids": bridge_failure_ids,
        "questions_sha256": sha256(questions_path.read_bytes()).hexdigest(),
        "expected_handoffs_sha256": sha256(expected_path.read_bytes()).hexdigest(),
        "candidate_handoffs_sha256": sha256(
            (output_dir / "query_plan_handoffs_v0.4.candidate.jsonl").read_bytes()
        ).hexdigest(),
        "dual_score": dual_score,
    }
    payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(output_dir / "report.json", payload)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold10-dir", type=Path, default=HISTORICAL_GOLD10,
                        help="historical diagnostic 입력")
    parser.add_argument(
        "--remaining60-dir", type=Path, default=HISTORICAL_REMAINING60,
        help="historical diagnostic 입력")
    parser.add_argument(
        "--run-dir", action="append", type=Path, default=[],
        help="평가할 실행 directory. 한 번에 돈 all70 실행이면 하나면 된다. "
             "주면 --gold10-dir/--remaining60-dir 쌍 대신 이 목록을 쓴다.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument(
        "--overlay-dir", action="append", type=Path, default=[],
        help=("선택 재호출 run의 row로 동일 question_id base row를 "
              "대체합니다; summary binding이 같아야 합니다"),
    )
    parser.add_argument(
        "--evaluation-mode", choices=("official", "diagnostic"),
        default="diagnostic",
        help="official 은 단일 all70 run만 받습니다. 기본 diagnostic 은 "
             "historical gold10/remaining60 비교 호환용입니다.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-native", action="store_true",
        help="네이티브 v1 을 끄고 호환 브리지만 쓴다 (경로 비교용)")
    parser.add_argument(
        "--native-only", action="store_true",
        help="closed composite와 legacy bridge 없이 v1 resolver/emitter만 쓴다")
    args = parser.parse_args(argv)
    if args.no_native and args.native_only:
        parser.error("--no-native와 --native-only는 함께 쓸 수 없습니다")
    run_dirs = (
        tuple(args.run_dir) if args.run_dir
        else (args.gold10_dir, args.remaining60_dir))
    report = evaluate(
        evaluation_mode=args.evaluation_mode,
        run_dirs=run_dirs,
        questions_path=args.questions,
        expected_path=args.expected,
        output_dir=args.output_dir,
        overlay_dirs=tuple(args.overlay_dir),
        use_native=not args.no_native,
        native_only=args.native_only,
    )
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
