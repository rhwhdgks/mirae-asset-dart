#!/usr/bin/env python3
"""coverage_set_v01 61문항을 정답 재료로 채점한다.

Gold70 이 쓰지 않는 기업 61곳을 덮는 세트다. **블라인드가 아니다** — 질문을
Stage1 구현자가 만들었다고 그 README 가 밝힌다. 그래서 이 점수는 일반화가 아니라
「회사가 바뀌면 깨지는 결함」이 있는지를 답한다.

정답지는 두 갈래로만 기계 채점된다.

- `required_claims` 20문항 — canonical 원 단위 값. 답변은 한국식 단위로 쓰므로
  표면이 아니라 환산한 정수로 비교한다.
- `required_documents` 44문항 — 인용해야 할 접수번호.

`expected_action` 은 61건 전부 `not_applicable` 이라 판단(답변·되묻기·거절)은
채점되지 않는다. 되묻기·거짓전제·답변불가·주입 20문항은 사람이 읽어야 한다 —
이 스크립트는 그 문항을 분류해 보여주기만 한다. 그 20문항 중 일부는 정답지에
required_documents 까지 달려 있는데, 답을 내지 않는 것이 정답인 문항이므로
기계 채점에 걸면 안 된다. group 을 먼저 보고 판단 축으로 보낸다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.korean_amount import parse_amounts
from scripts.score_core334 import direction_conflicts, quality

RECEIPT = re.compile(r"\b(\d{14})\b")
#: 사람이 읽어야 하는 축. 정답지에 기대 동작이 없다.
JUDGED_BY_HAND = {
    "R_ambiguity": "되물어야 한다",
    "R_false_premise": "전제가 틀렸다고 밝혀야 한다",
    "U_unanswerable": "거절해야 한다",
    "R_security": "주입을 무시하고 본 질문만 답해야 한다",
}


def _matches(expected: int, found: set[int] | frozenset[int]) -> bool:
    """정답지 값을 그 값 자신의 자릿수까지만 맞춰 본다.

    같은 수치가 원 단위 셀과 백만원 단위 셀 양쪽에 실린다. 미래에셋증권
    2024년 영업수익은 2024년 반기보고서에 22,242,327,609,027원으로,
    2025년 보고서 비교 열에는 22,242,327백만원으로 적혀 있다. 정답지가
    백만원 셀에서 왔는데 답변이 원 단위 원문을 인용하면 정확히 같은 값인데도
    불일치로 잡혔다.

    정답지 값의 끝자리 0 개수를 그 값이 가진 정밀도로 보고, 답변 값을 같은
    자리에서 버림·반올림했을 때 일치하면 통과시킨다. 허용 폭은 정답지 자신의
    한 단위뿐이라 실제로 틀린 값은 그대로 걸린다.
    """

    if expected in found:
        return True
    scale = 1
    while expected % (scale * 10) == 0:
        scale *= 10
    if scale == 1:
        return False
    return any(
        value // scale * scale == expected
        or round(value / scale) * scale == expected
        for value in found
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--questions", type=Path,
                    default=ROOT / "fixtures/coverage_set_v01/questions_company_v02.jsonl")
    ap.add_argument("--facts", type=Path,
                    default=ROOT / "fixtures/coverage_set_v01/answer_facts_company_v02.jsonl")
    args = ap.parse_args()

    rows = {json.loads(l)["question_id"]: json.loads(l)
            for l in args.responses.read_text(encoding="utf-8").splitlines() if l.strip()}
    meta = {json.loads(l)["question_id"]: json.loads(l)
            for l in args.questions.read_text(encoding="utf-8").splitlines() if l.strip()}
    facts = {json.loads(l)["question_id"]: json.loads(l)
             for l in args.facts.read_text(encoding="utf-8").splitlines() if l.strip()}

    value_ok, value_ng, doc_ok, doc_ng, hand = [], [], [], [], []
    for qid in sorted(meta):
        row = rows.get(qid)
        if row is None or row.get("http") != 200:
            value_ng.append((qid, f"HTTP {row.get('http') if row else '없음'}"))
            continue
        answer = (row.get("response") or {}).get("answer", "")
        fact = facts[qid]
        group = meta[qid]["group"]

        # 값과 형식이 맞아도 방향이 뒤집히면 답이 아니다. K-028·K-043·
        # K-052·K-056 이 이 축이 없어서 통과하고 있었다.
        conflict = direction_conflicts(answer)
        if conflict:
            value_ng.append((qid, f"증감 방향 뒤집힘 — {conflict}"))
            continue

        # 되묻기·거짓전제·답변불가·주입은 답을 내지 않는 것이 정답이라 값도
        # 근거도 나오지 않는다. 정답지가 그런 문항에도 required_documents를
        # 달아둔 탓에 fail-closed로 막은 주입 3건이 인용·값 미달로 잡혔다.
        # 기대 동작이 비답변이면 기계 채점 대상이 아니다.
        if group in JUDGED_BY_HAND:
            hand.append((qid, group, JUDGED_BY_HAND[group], answer))
        elif fact["required_claims"]:
            found = parse_amounts(answer)
            missing = [
                claim["claim"] for claim in fact["required_claims"]
                if not _matches(int(claim["canonical_value"]), found)
            ]
            (value_ng if missing else value_ok).append((qid, "; ".join(missing)))
        elif fact["required_documents"]:
            cited = set(RECEIPT.findall(answer))
            need = set(fact["required_documents"])
            # 후보가 여럿인 문항(계약 공시 86건 등)은 하나라도 인용하면 근거가 선다.
            (doc_ok if cited & need else doc_ng).append(
                (qid, f"기대 {len(need)}건 중 인용 0"))
        else:
            hand.append((qid, group, JUDGED_BY_HAND.get(group, group), answer))

    total = len(meta)
    print(f"coverage_set_v01 채점 {len(rows)}/{total}문항  "
          f"(블라인드 아님 — 회귀망으로 읽는다)\n")
    print("── 기계 채점 ──")
    print(f"  값 대조     통과 {len(value_ok):>3} / 미달 {len(value_ng)}")
    print(f"  접수번호    통과 {len(doc_ok):>3} / 미달 {len(doc_ng)}")
    for name, bucket in (("값 미달", value_ng), ("인용 미달", doc_ng)):
        if bucket:
            print(f"\n{name}:")
            for qid, why in bucket:
                print(f"  {qid}  {why}")
    # 일반인 기준 품질은 의미평가와 다른 것을 묻는다. 값이 맞아도 읽을 수
    # 없으면 품질은 떨어지고, 읽기 좋아도 값이 틀리면 의미평가는 실패다.
    buckets: dict[str, list] = {"PASS": [], "WARN": [], "FAIL": []}
    for qid, row in sorted(rows.items()):
        verdict, whys = quality(row)
        buckets[verdict].append((qid, "; ".join(whys)))
    print("\n── 일반인 기준 품질 ──")
    for key in ("PASS", "WARN", "FAIL"):
        share = len(buckets[key]) / max(len(rows), 1) * 100
        print(f"  {key:6} {len(buckets[key]):>3}   {share:5.1f}%")
    for key in ("FAIL", "WARN"):
        if buckets[key]:
            print(f"\n품질 {key}:")
            for qid, why in buckets[key]:
                print(f"  {qid}  {why}")

    print(f"\n── 사람이 읽어야 하는 {len(hand)}문항 ──")
    for qid, group, expect, answer in hand:
        one = " ".join(answer.split())[:96]
        print(f"  {qid} [{group}] {expect}")
        print(f"      {one}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
