"""정정 체인 — CORRECTS 와 SUPERSEDES 를 나눈다 (R-03 · X-12).

## 두 관계가 섞여 있었다

정정공시는 **원공시를 가리킨다.** 원공시 R 에 정정 A·B 가 차례로 붙으면 둘 다
`A CORRECTS R`, `B CORRECTS R` 이다. 이것만으로는 답할 수 없는 질문이 있다.

```text
CORRECTS 만 있을 때        R ← A,  R ← B      "지금 유효한 건 뭐야?" → 모름
SUPERSEDES 를 더하면       A ← B              → B (leaf 가 유일하다)
```

- `CORRECTS` — **선언된 대상.** 원문이 「무엇을 정정하는가」로 적은 것.
- `SUPERSEDES` — **파생된 순서.** 같은 대상을 정정한 것들을 접수일순으로 이은 것.

둘을 한 관계로 두면 체인의 leaf 를 계산할 수 없고, 「최신본」 질의가 깨진다.

## 파생 규칙

같은 `dst`(원공시)를 가리키는 `CORRECTS` 를 접수일·접수번호 순으로 정렬해 이웃끼리 잇는다.
`resolved` 가 아닌 간선은 대상이 확정되지 않았으므로 체인에 넣지 않는다 —
추측으로 순서를 만들면 틀린 「최신본」을 답하게 된다.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "LatestResolution", "LineageResolutionError", "derive_supersedes",
    "resolve_latest", "resolve_latest_known", "latest_of", "latest_of_known",
    "multi_target_sources",
]


LineageResolutionStatus = Literal["ok", "not_found", "ambiguous", "invalid"]


@dataclass(frozen=True)
class LatestResolution:
    """정정 계보 최신본 판정. 갈라짐과 cycle은 선택값 없이 드러낸다."""

    status: LineageResolutionStatus
    selected: str | None
    leaves: tuple[str, ...] = ()
    members: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def candidates(self) -> tuple[str, ...]:
        return self.leaves

    def __bool__(self) -> bool:
        return self.status == "ok" and self.selected is not None


class LineageResolutionError(LookupError):
    """문자열 반환형 ``latest_of``가 안전하게 표현할 수 없는 계보 상태."""

    def __init__(self, resolution: LatestResolution) -> None:
        self.resolution = resolution
        super().__init__(
            f"latest lineage {resolution.status}: {resolution.reason or resolution.leaves}")


def multi_target_sources(edges: list[dict]) -> dict[str, list[str]]:
    """**한 정정이 원본을 둘 이상 가리키는 경우.** 있으면 계보가 갈라진다.

    정정공시는 원본이 하나다. 둘을 가리키면 해석기가 잘못 붙인 것이고,
    `latest_of` 의 무방향 탐색이 **서로 다른 계보를 한 가족으로 합친다** (7차 검수).
    지금 코퍼스에는 0건이지만 막지 않으면 조용히 번진다.
    """
    by_src: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if (e.get("relation_type") == "CORRECTS"
                and e.get("resolution_status") == "resolved" and e.get("dst_rcept_no")):
            by_src[e["src_rcept_no"]].append(e["dst_rcept_no"])
    return {k: sorted(set(v)) for k, v in by_src.items() if len(set(v)) > 1}


def derive_supersedes(edges: list[dict], rcept_dt: dict[str, str]) -> list[dict]:
    """`CORRECTS` 목록 → `SUPERSEDES` 간선 목록.

    `edges` 는 relation jsonl 행, `rcept_dt` 는 접수번호 → 접수일이다.
    """
    by_root: dict[str, list[str]] = defaultdict(list)
    corrections: set[str] = set()          # 스스로 정정본인 문서
    for e in edges:
        if (e.get("relation_type") == "CORRECTS"
                and e.get("resolution_status") == "resolved"
                and e.get("dst_rcept_no")):
            by_root[e["dst_rcept_no"]].append(e["src_rcept_no"])
            corrections.add(e["src_rcept_no"])

    out: list[dict] = []

    # **정정의 정정** — `B CORRECTS A` 이고 `A` 자신이 정정본이면 B 가 A 를 대체한다.
    # 형제(같은 원본을 정정한 여럿)만 보면 이 사슬이 통째로 빠진다 — 실측 99건.
    for dst, srcs in by_root.items():
        if dst not in corrections:
            continue                        # 원공시를 정정한 것은 아래 형제 규칙이 다룬다
        for src in dict.fromkeys(srcs):
            out.append({
                "src_rcept_no": src, "dst_rcept_no": dst,
                "relation_type": "SUPERSEDES", "resolution_status": "resolved",
                "target_hint": None, "confidence": 1.0,
                "candidate_ids": [dst], "match_features": "chain|direct",
                "root_missing_reason": None,
            })

    for root, srcs in by_root.items():
        if len(srcs) < 2:
            continue
        # 접수일 → 접수번호 순. 같은 날 복수 정정도 순서가 정해진다.
        ordered = sorted(dict.fromkeys(srcs), key=lambda r: (rcept_dt.get(r, ""), r))
        for prev, cur in zip(ordered, ordered[1:]):
            out.append({
                "src_rcept_no": cur, "dst_rcept_no": prev,
                "relation_type": "SUPERSEDES", "resolution_status": "resolved",
                "target_hint": None, "confidence": 1.0,
                "candidate_ids": [prev], "match_features": f"chain|{root}",
                "root_missing_reason": None,
            })
    return out


def resolve_latest(rcept_no: str, edges: list[dict]) -> LatestResolution:
    """정정·대체 계보를 판정하되 복수 leaf나 cycle을 임의 선택하지 않는다.

    간선은 ``src``가 더 새 문서, ``dst``가 그 문서가 대체하는 이전 문서다. 연결 성분
    전체를 잡아 어느 구성원에서 시작해도 같은 답을 내되, leaf가 정확히 하나일 때만
    ``ok``다. 두 leaf 중 접수번호가 큰 것을 고르던 옛 동작은 데이터 결손을 답으로
    위장하므로 ``ambiguous``로 닫는다.
    """
    succ: dict[str, set[str]] = defaultdict(set)
    both: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        if (e.get("relation_type") in ("CORRECTS", "SUPERSEDES")
                and e.get("dst_rcept_no") and e.get("src_rcept_no")
                and e.get("resolution_status", "resolved") == "resolved"):
            src, dst = e["src_rcept_no"], e["dst_rcept_no"]
            succ[dst].add(src)
            both[dst].add(src)
            both[src].add(dst)

    known = set(both)
    if rcept_no not in known:
        return LatestResolution("not_found", None, (), (), "계보 간선에 없는 접수번호")

    # **가족 전체를 잡는다 — 방향을 따지지 않고.** 앞으로만 따라가면 형제에서 출발했을
    # 때 자기 자신에서 멈춘다. `A→R` · `B→R` · `C→A` 에서 B 는 뒤가 없으므로
    # `latest_of(B)=B` 가 됐다 — 같은 계보인데 출발점에 따라 답이 달라졌다 (6차 검수).
    # 무방향으로 연결 성분을 잡은 뒤, 그 안에서 **뒤가 없는 것**만 후보로 둔다.
    seen = {rcept_no}
    stack = [rcept_no]
    while stack:
        cur = stack.pop()
        for nb in both.get(cur, ()):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)

    # directed cycle은 leaf 수와 별개로 계보 자체가 무효다. cycle 뒤에 별도 leaf가 하나
    # 있더라도 그것을 고르면 손상된 중간 순서를 정상 체인으로 위장한다.
    color: dict[str, int] = {}

    def visit(node: str) -> bool:
        color[node] = 1
        for nxt in succ.get(node, ()):
            if nxt not in seen:
                continue
            if color.get(nxt) == 1:
                return True
            if color.get(nxt, 0) == 0 and visit(nxt):
                return True
        color[node] = 2
        return False

    has_cycle = any(color.get(node, 0) == 0 and visit(node) for node in sorted(seen))
    members = tuple(sorted(seen))
    if has_cycle:
        return LatestResolution("invalid", None, (), members, "cycle")

    leaves = tuple(sorted(r for r in seen if not succ.get(r)))
    if len(leaves) > 1:
        return LatestResolution("ambiguous", None, leaves, members, "multiple_leaves")
    if not leaves:
        # cycle 검사가 참이어야 하지만, 손상된 입력을 안전하게 닫는 방어 분기다.
        return LatestResolution("invalid", None, (), members, "no_leaf")
    return LatestResolution("ok", leaves[0], leaves, members)


def resolve_latest_known(rcept_no: str, edges: list[dict]) -> LatestResolution:
    """명시적 이름의 안전 조회. 모르는 접수번호는 ``not_found``다."""
    return resolve_latest(rcept_no, edges)


def latest_of(rcept_no: str, edges: list[dict]) -> str:
    """기존 문자열 API. 명확한 체인은 호환하고 손상 계보는 typed 예외로 닫는다.

    계보에 없는 입력을 자기 자신으로 돌려주는 옛 호환 동작은 유지한다. 존재 여부까지
    확인해야 하는 runtime 경로는 :func:`resolve_latest_known`을 사용한다.
    """
    result = resolve_latest(rcept_no, edges)
    if result.status == "not_found":
        return rcept_no
    if result.status != "ok":
        raise LineageResolutionError(result)
    assert result.selected is not None
    return result.selected


def latest_of_known(rcept_no: str, edges: list[dict]) -> str | None:
    """`latest_of` 와 같되, **계보에 없는 접수번호면 `None`**.

    `latest_of` 는 모르는 문서를 받으면 그것을 그대로 돌려준다 — 「자기 자신이
    최신본」이라는 뜻이 되어, 오타나 미래 접수번호가 **정상 답처럼** 보인다
    (7차 검수). 호출부가 「계보에 있는가」를 알아야 하면 이쪽을 쓴다.
    """
    result = resolve_latest_known(rcept_no, edges)
    return result.selected if result.status == "ok" else None
