"""Canonical warning의 승인·수정 결정을 원문 hash에 묶는 fail-closed gate.

warning을 빈 목록으로 만드는 모듈이 아니다. 빌드가 관측한 warning/repair를 versioned TSV의
결정과 1:1로 대조하고, 새 관측이나 사라진 결정이 있으면 발행을 차단한다.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

__all__ = [
    "QUALITY_POLICY_PATH", "QUALITY_POLICY_VERSION", "QualityDecision",
    "load_quality_decisions", "evaluate_quality_decisions",
]


QUALITY_POLICY_VERSION = "source-quality/1.0"
QUALITY_POLICY_PATH = Path(__file__).with_name("quality_decisions.tsv")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def policy_path_label(path: "str | Path | None") -> str:
    """run.json 에 남길 정책 파일 표기 — **저장소 상대경로**. 절대경로를 적으면 빌드한 사람의 홈 디렉터리가
    발행 증빙에 남는다(실제로 그랬다)."""
    target = Path(path or QUALITY_POLICY_PATH).resolve()
    try:
        return target.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return target.name

_COLUMNS = (
    "decision_id", "doc_id", "stage", "code", "disposition",
    "source_sha256s", "locator", "guard", "expected_matches",
    "expected_conflicts", "rationale",
)
_DISPOSITIONS = frozenset({"fixed_verified", "accepted_exception", "accepted_limited"})


@dataclass(frozen=True)
class QualityDecision:
    decision_id: str
    doc_id: str
    stage: str
    code: str
    disposition: str
    source_sha256s: tuple[str, ...]
    locator: str | None
    guard: str | None
    expected_matches: int | None
    expected_conflicts: int | None
    rationale: str


def _hashes(raw: str, *, label: str) -> tuple[str, ...]:
    values = tuple(sorted(x.strip().lower() for x in raw.split(",") if x.strip()))
    if not values or any(re.fullmatch(r"[0-9a-f]{64}", x, flags=re.ASCII) is None
                         for x in values):
        raise ValueError(f"{label}: source_sha256s는 64자리 full SHA-256 목록이어야 합니다")
    if len(values) != len(set(values)):
        raise ValueError(f"{label}: source_sha256s 중복")
    return values


def _optional_int(raw: str, *, label: str) -> int | None:
    if raw == "":
        return None
    if re.fullmatch(r"\d+", raw, flags=re.ASCII) is None:
        raise ValueError(f"{label}: 음이 아닌 정수여야 합니다")
    return int(raw)


def load_quality_decisions(path: Path | None = None) -> tuple[QualityDecision, ...]:
    path = path or QUALITY_POLICY_PATH
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{path.name}: quality decision이 비어 있습니다")
    header = tuple(lines[0].split("\t"))
    if header != _COLUMNS:
        raise ValueError(f"{path.name}: header 계약 불일치")
    out: list[QualityDecision] = []
    ids: set[str] = set()
    signatures: set[tuple] = set()
    for lineno, line in enumerate(lines[1:], 2):
        values = line.split("\t")
        if len(values) != len(header):
            raise ValueError(f"{path.name}:{lineno} 열 수 {len(values)} != {len(header)}")
        row = dict(zip(header, values))
        decision_id = row["decision_id"]
        if not decision_id or decision_id in ids:
            raise ValueError(f"{path.name}:{lineno} decision_id 누락/중복")
        if row["disposition"] not in _DISPOSITIONS:
            raise ValueError(f"{path.name}:{lineno} disposition 오류")
        if not all(row[x] for x in ("doc_id", "stage", "code", "rationale")):
            raise ValueError(f"{path.name}:{lineno} 필수 결정 필드 누락")
        item = QualityDecision(
            decision_id=decision_id, doc_id=row["doc_id"], stage=row["stage"],
            code=row["code"], disposition=row["disposition"],
            source_sha256s=_hashes(row["source_sha256s"],
                                   label=f"{path.name}:{lineno}"),
            locator=row["locator"] or None, guard=row["guard"] or None,
            expected_matches=_optional_int(
                row["expected_matches"], label=f"{path.name}:{lineno}"),
            expected_conflicts=_optional_int(
                row["expected_conflicts"], label=f"{path.name}:{lineno}"),
            rationale=row["rationale"],
        )
        signature = (
            item.doc_id, item.stage, item.code, item.source_sha256s,
            item.locator, item.guard, item.expected_matches, item.expected_conflicts,
        )
        if signature in signatures:
            raise ValueError(f"{path.name}:{lineno} decision signature 중복")
        ids.add(decision_id)
        signatures.add(signature)
        out.append(item)
    return tuple(out)


def _observation_signature(raw: Mapping) -> tuple:
    hashes = _hashes(
        ",".join(str(x) for x in raw.get("source_sha256s") or ()),
        label=f"observation {raw.get('doc_id')}/{raw.get('code')}")
    return (
        str(raw.get("doc_id") or ""), str(raw.get("stage") or ""),
        str(raw.get("code") or ""), hashes,
        (str(raw["locator"]) if raw.get("locator") else None),
        (str(raw["guard"]) if raw.get("guard") else None),
        (int(raw["matched_claims"]) if raw.get("matched_claims") is not None else None),
        (int(raw["conflict_claims"]) if raw.get("conflict_claims") is not None else None),
    )


def evaluate_quality_decisions(
        observations: Iterable[Mapping], *, policy_path: Path | None = None) -> dict:
    """관측과 결정을 정확히 1:1 대조한다.

    새 warning뿐 아니라 더 이상 관측되지 않는(stale) 승인도 차단한다. 원문이 바뀌어 문제가
    해결됐더라도 과거 승인을 자동 재사용하지 않고 사람이 결정을 닫게 하기 위해서다.
    """
    decisions = load_quality_decisions(policy_path)
    by_signature = {
        (d.doc_id, d.stage, d.code, d.source_sha256s, d.locator, d.guard,
         d.expected_matches, d.expected_conflicts): d
        for d in decisions
    }
    applied: list[dict] = []
    unresolved: list[dict] = []
    seen: set[str] = set()
    observed_signatures: set[tuple] = set()
    for raw in observations:
        try:
            signature = _observation_signature(raw)
        except (TypeError, ValueError) as exc:
            unresolved.append({"observation": dict(raw), "reason": str(exc)})
            continue
        if signature in observed_signatures:
            unresolved.append({"observation": dict(raw), "reason": "관측 signature 중복"})
            continue
        observed_signatures.add(signature)
        decision = by_signature.get(signature)
        if decision is None:
            unresolved.append({"observation": dict(raw), "reason": "승인 결정 없음"})
            continue
        seen.add(decision.decision_id)
        applied.append(asdict(decision))
    stale = [asdict(d) for d in decisions if d.decision_id not in seen]
    return {
        "policy_version": QUALITY_POLICY_VERSION,
        "policy_path": policy_path_label(policy_path),
        "observed": len(observed_signatures),
        "applied": applied,
        "unresolved": unresolved,
        "stale": stale,
        "passed": not unresolved and not stale,
    }
