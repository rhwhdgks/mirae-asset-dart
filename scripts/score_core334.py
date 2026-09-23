#!/usr/bin/env python3
"""Core334 응답을 의미평가와 일반인 기준 품질로 나눠 채점한다.

두 축은 서로 다른 것을 묻는다. 사실이 맞아도 읽을 수 없으면 품질은 떨어지고,
읽기 좋아도 값이 틀리면 의미평가는 실패다. 한 점수로 합치면 어느 쪽이 문제인지
보이지 않는다 — Core334 응답품질 감사(2026-08-31)가 두 축을
분리한 이유다.

**의미평가는 기준선 대조다.** 334문항 전부에 대한 독립 정답지는 없다. EG2 29건은
`fixtures/edge_gap_v02` 정답지로 직접 채점하고, 나머지는 직전 실행 대비 값·접수번호
보존과 거절 전환으로 회귀를 본다. 그래서 이 점수는 「이번 변경이 의미를 깨뜨렸는가」를
답하지, 「원래 정답인가」를 답하지 않는다.
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

from scripts.korean_amount import parse_amounts, expected_value
from scripts.audit_answer_quality import (
    INTERNAL_SURFACE, UNCONVERTED_UNIT, BARE_NEGATIVE, SOURCE_FOOTER,
    repeated_sentences,
)

FIVE = {"question_id", "question", "answer", "retrieved_context", "think_trace"}
RECEIPT = re.compile(r"\b(\d{14})\b")
REFUSE = re.compile(
    r"근거와 함께 확인할 수 없어|해석하지 못했습니다|지원하지 않는 대상|"
    r"안전하게 대응시키지 못했습니다|확인하지 못했습니다\(원문 부재")
#: 표 좌표가 사용자 표면으로 새어 나온 흔적.
PATHY = re.compile(r"\(단위[^)]*\)\s*>|\s>\s")
#: 이슈 #94 30 — 답 하단 「출처」 블록. 그 안의 " > " 는 표 좌표 유출이
#: 아니라 `보고서명 > 목차 > 표명 > 행명` 형태로 **일부러** 적은 사람이
#: 읽는 경로다(`app/composer/template.py`의 `_source_footer`). `PATHY`
#: 는 그 기능이 생기기 전에 쓴 휴리스틱이라 이 블록만은 검사하지 않는다.
#: `SOURCE_FOOTER` 는 `audit_answer_quality.repeated_sentences` 가 같은
#: 경계로 문장 중복도 걸러내도록(#109) `audit_answer_quality` 에서 가져온다.
#: "계산 근거: A …2024년… 값 / B …2025년… 값" 에서 연도와 값을 짝지어 읽는다.
BASIS_YEAR = re.compile(r"(\d{4})년")


def direction_conflicts(answer: str) -> str | None:
    """계산 근거의 두 시점 값이 말하는 방향과 답변의 방향이 어긋나는가.

    K-028 은 매출이 3조 8,811억원 늘었는데 「감소」로 답했다. 값의 크기도
    형식도 맞아서 두 축 모두 통과했다 — 뒤집힌 것은 방향뿐이었다. 방향은
    값과 따로 확인해야 잡힌다.

    계산 근거 줄이 두 연도의 값을 싣고 있을 때만 본다. 음수 값(적자)이
    섞이면 크기 비교로는 방향을 알 수 없으므로 판정하지 않는다 —
    삼성SDI 2025년 당기순손실처럼 절댓값이 커도 실제로는 감소다.
    """

    basis = [line for line in answer.splitlines()
             if line.startswith("계산 근거:")]
    if not basis:
        return None
    parts = basis[0][len("계산 근거:"):].split(" / ")
    if len(parts) != 2:
        return None
    if "손실" in answer or "적자" in answer or "-" in basis[0]:
        return None
    pairs = []
    for part in parts:
        years = BASIS_YEAR.findall(part)
        values = parse_amounts(part)
        if years and values:
            pairs.append((int(years[0]), max(values)))
    if len(pairs) != 2 or pairs[0][0] == pairs[1][0]:
        return None
    (_, early), (_, late) = sorted(pairs)
    actual = "증가" if late > early else "감소" if late < early else None
    stated = ("증가" if "증가했습니다" in answer or "(증가)" in answer
              else "감소" if "감소했습니다" in answer or "(감소)" in answer
              else None)
    if actual and stated and actual != stated:
        return f"근거는 {actual}인데 답변은 {stated}"
    return None


#: 질문이 전량·이력·목록을 요구했다는 표지. 길이는 그때 정당하다.
WANTS_FULL = re.compile(r"모두|전체|전부|각각|이력|목록|나열|정리해|비교")


def answer_of(row: dict) -> str:
    return ((row.get("response") or {}).get("answer") or "")


def load(path: Path) -> dict:
    return {json.loads(l)["question_id"]: json.loads(l)
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def semantic(row: dict, base: dict | None, rubric: dict | None) -> tuple[str, str]:
    """(판정, 사유). 계약 위반과 회귀를 실패로, 사실 축소를 부분으로 본다."""
    body = row.get("response") or {}
    if row.get("http") != 200:
        return "FAIL", f"HTTP {row.get('http')}"
    if set(body) != FIVE:
        return "FAIL", "5필드 계약 위반"
    text = answer_of(row).strip()
    if not text:
        return "FAIL", "빈 답변"

    # 기준선 없이도 판정되는 절대 오류다. K-028 은 매출이 3조 8,811억원
    # 늘었는데 「감소」로 답했고, 값도 형식도 맞아 두 축 모두 통과했다.
    conflict = direction_conflicts(text)
    if conflict:
        return "FAIL", f"증감 방향 뒤집힘 — {conflict}"

    if rubric is not None:
        got = parse_amounts(text)
        missing = []
        for fact in rubric["required_facts"]:
            want = expected_value(fact)
            # 표 셀 그대로 인용된 값(「44,351,125」·「0.28」·「1.90%」)은 단위가 없어
            # 금액 파서가 읽지 못한다. 정답지의 숫자 리터럴이 답변에 그대로 있으면 같은
            # 사실로 본다 — p9 서술지표 문항(fixtures/p9_gap_v01)이 이 형태다.
            literals = re.findall(r"\d[\d,]*(?:\.\d+)?%?", fact)
            literal_ok = bool(literals) and all(number in text for number in literals)
            if want is None:
                core = [w for w in re.split(r"[\s:]+",
                        re.sub(r"[\d,().%]+", "", fact).strip()) if len(w) >= 2][:3]
                if literals and not literal_ok:
                    missing.append(fact)
                elif not literals and core and not any(w in text for w in core):
                    missing.append(fact)
            elif want not in got and not literal_ok:
                missing.append(fact)
        cited = set(RECEIPT.findall(text))
        groups = rubric.get("required_citation_groups")
        any_of = rubric.get("required_citation_any")
        cite_ok = (all(cited & set(g) for g in groups) if groups
                   else bool(cited & set(any_of)) if any_of else True)
        # edge-gap-rubric/0.1 은 이 필드를 `expected_initial_action`으로 썼다.
        expected_action = rubric.get("expected_action") or rubric.get("expected_initial_action")
        if expected_action == "not_found":
            ok = "확인" in text and ("못" in text or "없" in text)
            return ("PASS", "") if ok else ("FAIL", "없는 계정에 값을 제시")
        if missing:
            return "FAIL", f"정답지 사실 누락 {missing}"
        if not cite_ok:
            return "PARTIAL", "요구 접수번호 미인용"
        return "PASS", ""

    if base is None:
        return "PASS", ""
    prior = answer_of(base)
    if REFUSE.search(text) and not REFUSE.search(prior):
        return "FAIL", "직전에 답하던 질문이 거절로 바뀜"
    lost_cites = set(RECEIPT.findall(prior)) - set(RECEIPT.findall(text))
    lost_values = parse_amounts(prior) - parse_amounts(text)
    if lost_cites and lost_values:
        return "PARTIAL", f"근거 {len(lost_cites)}·값 {len(lost_values)}건 사라짐"
    return "PASS", ""


def quality(row: dict) -> tuple[str, list[str]]:
    """(판정, 사유들). 일반인이 그대로 읽을 수 있는지만 본다."""
    text = answer_of(row)
    question = row.get("question", "")
    fail: list[str] = []
    warn: list[str] = []

    hit = INTERNAL_SURFACE.search(text)
    if hit:
        fail.append(f"내부 식별자 노출({hit.group(0)[:20]})")
    body_before_footer = SOURCE_FOOTER.split(text, maxsplit=1)[0]
    if PATHY.search(body_before_footer):
        fail.append("표 좌표 노출")
    if UNCONVERTED_UNIT.search(text):
        warn.append("단위 미변환")
    if BARE_NEGATIVE.search(text):
        warn.append("음수 직접 표기")
    dups = repeated_sentences(text)
    if dups:
        warn.append(f"문장 중복 {dups}")
    if len(text) > 1500 and not WANTS_FULL.search(question):
        # 질문이 전량을 요구하지 않았는데 길면 읽는 부담이 실제로 있다.
        warn.append(f"요구 대비 장문 {len(text)}자")
    if fail:
        return "FAIL", fail + warn
    if warn:
        return "WARN", warn
    return "PASS", []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--rubric", type=Path,
                    default=ROOT / "fixtures/edge_gap_v02/rubric_v0.2.jsonl")
    ap.add_argument("--show", type=int, default=40)
    args = ap.parse_args()

    rows = load(args.responses)
    base = load(args.baseline) if args.baseline else {}
    rub = {json.loads(l)["case_id"]: json.loads(l)
           for l in args.rubric.read_text(encoding="utf-8").splitlines() if l.strip()}

    sem: dict[str, list] = {"PASS": [], "PARTIAL": [], "FAIL": []}
    qua: dict[str, list] = {"PASS": [], "WARN": [], "FAIL": []}
    for qid, row in sorted(rows.items()):
        verdict, why = semantic(row, base.get(qid), rub.get(qid))
        sem[verdict].append((qid, why))
        verdict_q, whys = quality(row)
        qua[verdict_q].append((qid, "; ".join(whys)))

    total = len(rows)
    print(f"채점 {total}문항  (기준선 대조: {'있음' if base else '없음'})\n")
    print("── 의미평가 ──")
    for key in ("PASS", "PARTIAL", "FAIL"):
        print(f"  {key:8} {len(sem[key]):>4}   {len(sem[key])/total*100:5.1f}%")
    print("\n── 일반인 기준 품질 ──")
    for key in ("PASS", "WARN", "FAIL"):
        print(f"  {key:8} {len(qua[key]):>4}   {len(qua[key])/total*100:5.1f}%")
    for name, bucket in (("의미 FAIL", sem["FAIL"]), ("의미 PARTIAL", sem["PARTIAL"]),
                         ("품질 FAIL", qua["FAIL"])):
        if bucket:
            print(f"\n{name} ({len(bucket)}):")
            for qid, why in bucket[:args.show]:
                print(f"  {qid:14} {why}")
    if qua["WARN"]:
        print(f"\n품질 WARN ({len(qua['WARN'])}):")
        for qid, why in qua["WARN"][:args.show]:
            print(f"  {qid:14} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
