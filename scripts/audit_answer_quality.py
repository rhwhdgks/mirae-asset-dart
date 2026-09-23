#!/usr/bin/env python3
"""Core334 답변을 일반인 기준으로 기계 점검한다.

Core334 응답품질 감사(2026-08-31)가 정한 6개 검사 중
기계가 판정할 수 있는 것만 본다. 사람 판정을 대신하지 않고 **읽어야 할 문항을
좁힌다.** 통과가 곧 품질 PASS 를 뜻하지는 않는다.

실행 중인 JSONL 에도 그대로 쓴다. 한 줄씩 flush 되므로 완료분만 채점된다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


#: 사용자 답변에 나와서는 안 되는 내부 표현.
INTERNAL_SURFACE = re.compile(
    r"TABLE(?:-GROUP)?\[|\bTBODY\b|\bTR\[|\bTE\[|"
    r"\b(?:retained_earnings|cost_of_sales|total_assets|net_income|"
    r"operating_income|cf_operating|cf_investing|cf_financing|revenue|"
    r"inventories|trade_receivables|intangible_assets|current_liabilities|"
    r"deferred_tax_assets|sganda|finance_income|diluted_eps|capex_ppe|"
    r"share_capital|interest_paid|dividends_paid|total_comprehensive_income)\b|"
    r"\bCFS\b|\bSFS\b|<<<|UNTRUSTED")

#: 띄어 쓴 단위는 원화 표면 정규화를 통과하지 못한 흔적이다.
UNCONVERTED_UNIT = re.compile(r"\d\s*백만\s+원|\d\s*억\s+원|\d\s*조\s+원")

#: 현금유출을 방향어 없이 음수 기호로만 쓴 표면.
#: 현금 유출을 부호로만 적은 표면. 「지급이 마이너스」는 일반 독자가 읽지 못한다.
#:
#: **모든 음수가 결함은 아니다.** 부문 간 내부거래 제거액(``내부매출액 -82조``),
#: 연결조정, 사업부문 「기타」는 실제로 음수이고 부호가 맞는 표기다. 그래서
#: 현금흐름 항목 근처에 있을 때만 지적한다.
BARE_NEGATIVE = re.compile(
    r"(?:현금흐름|지급|취득|상환|납부|유출)[^.\n]{0,40}[-−]\s*\d[\d,]*\s*(?:조|억|백만)"
    r"|[-−]\s*\d[\d,]*\s*(?:조|억|백만)[^.\n]{0,20}(?:현금흐름|지급|취득|상환)")

CALC_BASIS = re.compile(r"계산 근거")
#: 이미 밝힌 접수번호를 다시 쓰는 표기. 표 행마다 붙으면 25행에서 25번 나온다.
CITED_RECEIPT = re.compile(r"접수번호\s*\d{14}")
RECEIPT_DIGITS = re.compile(r"\d{14}")
FIVE_FIELDS = ("question_id", "question", "answer",
               "retrieved_context", "think_trace")


#: 인용은 사실마다 붙는다. 같은 접수번호가 여러 번 나오는 것은 중복이 아니다.
CITATION = re.compile(r"\(?\s*(?:근거[:：]?\s*)?접수번호[^)]*\)?")

#: 이슈 #94 30 — 답 하단 「출처」 블록의 경계. `score_core334.py` 의 `PATHY`
#: 검사가 이 경계 아래를 건너뛰는 것과 같은 정규식이다(`app/composer/template.py`
#: 의 `_source_footer`). 이슈 #109 — 이 블록의 두 줄은 좌표만 다르고
#: `보고서명 (연월) · 접수번호 … ·` 접두부는 같을 수 있어서(예: 「2-2. 연결
#: 손익계산서 > 매출액 (주30)」과 「4-2. 손익계산서 > 매출액 (주29)」), 인용을
#: 지우면 서로 다른 두 줄이 같은 문자열로 뭉쳐 오탐 중복이 된다.
SOURCE_FOOTER = re.compile(r"\n\n출처\n")

# Company abbreviations contain a period but do not end a sentence.  Keep the
# lookbehinds fixed-width so this remains compatible with Python's `re`.
SENTENCE_BOUNDARY = re.compile(
    r"(?<!Inc\.)(?<!Ltd\.)(?<!Corp\.)(?<!Co\.)(?<=[.!?])\s+",
    re.IGNORECASE,
)


def repeated_sentences(answer: str) -> int:
    """같은 문장이 두 번 이상 나오면 사용자에게는 중복으로 읽힌다.

    인용 표기는 먼저 지운다. 지우지 않으면 문장 분리가 ``(접수번호 …)`` 를
    독립 조각으로 잘라내고, 서로 다른 사실에 붙은 근거가 중복으로 집계된다.

    표 행과 출처 절은 세지 않는다(#109). 표 렌더(#86·#100)는 마크다운 구분선
    (``| --- | --- |``) 같은 머리글이 표마다 반복되는 것이 정상이고, 「출처」
    절(``SOURCE_FOOTER``) 아래는 서로 다른 좌표가 인용을 지운 뒤 같은
    문자열로 뭉쳐 오탐을 낸다.

    표 행은 **줄 단위로, 문장으로 쪼개기 전에** 거른다(P9-011 회귀). 표 칸
    안에 마침표가 여럿이면(「…있습니다. …있음.」류 각주가 칸 하나에 이어
    붙는 표) 줄 전체를 먼저 문장으로 쪼갠 뒤에 걸렀을 때 칸 중간·꼬리
    조각이 ``|`` 로 시작하지 않아 걸러지지 않고, 품목마다 반복되는 같은
    각주 문구가 문장 중복으로 잘못 잡힌다. 줄 앞의 공백·불릿(``- ``) 뒤에
    오는 ``|`` 도 표 행으로 본다(``line.lstrip(" -")`` 로 판정) — 목록으로
    렌더된 표 행도 마찬가지로 반복되는 머리글·구분선을 갖는다.
    """
    body = SOURCE_FOOTER.split(answer, maxsplit=1)[0]
    stripped = CITATION.sub("", body)
    parts: list[str] = []
    for line in stripped.split("\n"):
        if line.lstrip(" -").startswith("|"):
            continue
        parts.extend(SENTENCE_BOUNDARY.split(line))
    parts = [s.strip() for s in parts]
    parts = [s for s in parts if len(s) >= 20]
    seen: dict[str, int] = {}
    for sentence in parts:
        seen[sentence] = seen.get(sentence, 0) + 1
    return sum(count - 1 for count in seen.values() if count > 1)


def audit(row: dict, *, long_chars: int) -> list[str]:
    flags: list[str] = []
    body = row.get("response") or {}
    if row.get("http") != 200:
        return [f"http_{row.get('http')}"]
    missing = [f for f in FIVE_FIELDS if f not in body]
    if missing:
        flags.append(f"필드누락:{','.join(missing)}")
    if set(body) - set(FIVE_FIELDS):
        flags.append(f"필드초과:{','.join(sorted(set(body) - set(FIVE_FIELDS)))}")
    answer = (body.get("answer") or "").strip()
    if not answer:
        return flags + ["빈답변"]

    if len(answer) > long_chars:
        flags.append(f"장문:{len(answer)}자")
    hit = INTERNAL_SURFACE.search(answer)
    if hit:
        flags.append(f"내부표현:{hit.group(0)[:24]}")
    if UNCONVERTED_UNIT.search(answer):
        flags.append("단위미변환")
    if BARE_NEGATIVE.search(answer):
        flags.append("현금유출_음수표기")
    repeats = repeated_sentences(answer)
    if repeats:
        flags.append(f"문장중복:{repeats}")
    # 근거는 한 번 밝히면 된다. 같은 접수번호가 여러 번 나오면 그만큼이
    # 사용자가 읽어야 할 글자로 남는다 — 품질감사의 「접수번호가 불필요하게
    # 반복되지 않는가」 항목이고, 실측에서 334건 합계 3,306자였다.
    shown = CITED_RECEIPT.findall(answer)
    unique = len(set(RECEIPT_DIGITS.findall(answer)))
    if unique and len(shown) >= max(4, unique * 3):
        flags.append(f"접수번호반복:{len(shown)}회/고유{unique}")
    # 실제 연산이 없는데 `계산 근거` 를 붙이면 사용자는 없는 계산을 찾게 된다.
    if CALC_BASIS.search(answer):
        operands = re.findall(r"[\d][\d,]{4,}", answer)
        if len(set(operands)) < 2:
            flags.append("계산근거_피연산자부족")
    return flags


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--long-chars", type=int, default=1500)
    parser.add_argument("--show", type=int, default=60)
    args = parser.parse_args()

    rows = [json.loads(line)
            for line in args.responses.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    flagged: list[tuple[str, list[str], int]] = []
    for row in rows:
        flags = audit(row, long_chars=args.long_chars)
        if flags:
            answer = ((row.get("response") or {}).get("answer") or "")
            flagged.append((row["question_id"], flags, len(answer)))

    print(f"채점 대상 {len(rows)}건 / 지적 {len(flagged)}건 "
          f"({len(rows) - len(flagged)}건 무지적)")
    counter: dict[str, int] = {}
    for _, flags, _ in flagged:
        for flag in flags:
            counter[flag.split(":")[0]] = counter.get(flag.split(":")[0], 0) + 1
    print("\n원인별:")
    for name, count in sorted(counter.items(), key=lambda kv: -kv[1]):
        print(f"  {name:26} {count}")
    print(f"\n문항별 (상위 {args.show}):")
    for qid, flags, size in sorted(
            flagged, key=lambda item: -item[2])[:args.show]:
        print(f"  {qid:14} {size:>6}자  {'; '.join(flags)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
