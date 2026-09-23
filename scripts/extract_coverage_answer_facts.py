#!/usr/bin/env python3
"""커버리지 세트의 **사실 정답**을 코퍼스에서 기계적으로 뽑는다.

판단(answer/clarify/refuse, 슬롯, 금지 주장)은 외부가 작성했고, 이 스크립트는
`required_documents` 와 `required_claims` 만 채운다.  둘을 나눈 이유는 코퍼스를
못 보는 모델이 접수번호·금액을 **형식만 맞는 가짜**로 만들기 때문이다.

여기서 하는 일은 판단이 아니라 조회다.  질문 문면에서 회사·기간·상대를 읽고
정본에 있는 값을 그대로 옮긴다.  **찾지 못하면 비워 둔다** — 지어내면 정답지
자체가 오염된다.

    PYTHONPATH=. .venv/bin/python scripts/extract_coverage_answer_facts.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SET_ROOT = ROOT / "fixtures/coverage_set_v01"
CUTOFF = "20260619"
REFERENCE_YEAR = 2026


def _load(path: Path) -> "list[dict]":
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _companies_in(question: str, corpus) -> "list":
    """질문 문면에서 정본에 등록된 회사를 **전부** 찾는다.

    한국어는 이름에 조사가 붙는다(`셀트리온의`).  토큰을 그대로 조회하면 대부분
    실패하므로 뒤에서 한 글자씩 줄이며 **가장 긴 해석되는 형태**를 취한다.
    질문이 두 회사를 비교할 수도 있으므로 하나로 제한하지 않는다.
    """

    found = []
    for token in re.findall(r"[가-힣A-Za-z][가-힣A-Za-z0-9&\.]{1,15}", question):
        for end in range(len(token), 1, -1):
            rows = corpus.resolve_company(token[:end])
            if len(rows) == 1:
                found.append(rows[0])
                break
    unique = {row.corp_code: row for row in found}
    return list(unique.values())


def _annual_revenue(corpus, corp_code: str, year: int, scope: str):
    for row in corpus.facts(corp_code, as_of=CUTOFF):
        if (row.account_norm == "revenue" and row.period_type == "annual"
                and row.scope == scope
                and str(row.period_end).startswith(str(year))):
            return row
    return None


#: 공시 단위를 원으로 환산하는 배수. Gold 는 canonical_value 를 항상 원으로 적는다.
_UNIT_SCALE = {"원": 1, "천원": 1_000, "백만원": 1_000_000, "십억원": 1_000_000_000,
               "억원": 100_000_000, "조원": 1_000_000_000_000}


def _claim(text: str, money=None) -> dict:
    """Gold 형식의 주장 하나.

    `canonical_*` 는 **원 단위로 환산한 값**, `raw_*` 는 공시에 적힌 그대로다.
    Gold 가 그렇게 쓴다 — 「백만원 333,605,938」과 「원 333605938000000」이 한 행에
    같이 있다.  환산 배수를 모르는 단위는 환산하지 않고 원문만 남긴다.
    """

    if money is None:
        return {key: "not_applicable" for key in
                ("canonical_unit", "canonical_value", "raw_unit",
                 "raw_value", "value_text")} | {"claim": text}
    unit = money.unit or "원"
    scale = _UNIT_SCALE.get(unit)
    canonical_value = ("not_applicable" if scale is None
                       else str(int(round(money.value * scale))))
    return {
        "claim": text,
        "canonical_unit": "원" if scale is not None else "not_applicable",
        "canonical_value": canonical_value,
        "raw_unit": unit,
        "raw_value": money.text,
        "value_text": money.text,
    }


def extract(corpus, question_id: str, question: str) -> "tuple[list[str], list[dict]]":
    """(required_documents, required_claims). 확정 못 하면 빈 목록."""

    companies = _companies_in(question, corpus)
    if not companies:
        return [], []

    years = [int(y) for y in re.findall(r"(20\d{2})년", question)]
    # 「25년」처럼 두 자리로 쓰는 판도 있다
    years += [2000 + int(y) for y in re.findall(r"(?<!\d)(\d{2})년", question)
              if 20 <= int(y) <= 30]
    # 「작년·재작년·올해」는 reference_date 기준 상대 표현이다.
    # 기준일이 2026-06-19 이고 연간 실적은 직전 사업연도까지 확정되므로
    # 작년 = 2025, 재작년 = 2024 로 읽는다.
    if "재작년" in question:
        years.append(REFERENCE_YEAR - 2)
    if re.search(r"작년", question):
        years.append(REFERENCE_YEAR - 1)
    if "올해" in question:
        years.append(REFERENCE_YEAR)
    documents: list[str] = []
    claims: list[dict] = []

    # 재무 — 연결/별도가 문면에 있을 때만 그 범위를 쓴다
    if any(word in question for word in ("매출", "실적", "영업이익")) and years:
        scopes = []
        if "연결" in question:
            scopes.append("CFS")
        if "별도" in question:
            scopes.append("SFS")
        if not scopes:
            scopes = ["CFS"]
        for company in companies:
            for year in sorted(set(years)):
                for scope in scopes:
                    row = _annual_revenue(corpus, company.corp_code, year, scope)
                    if row is None:
                        continue
                    claims.append(_claim(
                        f"{company.corp_name} {year}년 "
                        f"{'연결' if scope == 'CFS' else '별도'} 매출액",
                        row.money))
                    receipt = str(getattr(row, "doc_id", "")).split("_")[-1]
                    if re.fullmatch(r"[0-9]{14}", receipt):
                        documents.append(receipt)

    # 사건 — 질문에 나온 상대와 일치하는 공시 접수번호
    if any(word in question for word in ("계약", "해지", "공시")):
        from agent.stage1_v1_document_backends import (
            normalize_form_label, _label_tail)
        grouped: dict[str, dict] = defaultdict(dict)
        for company in companies:
          for row in corpus.fields(as_of=CUTOFF, corp_code=company.corp_code):
            leaf = normalize_form_label(_label_tail(row.path))
            if leaf:
                grouped[row.rcept_no][leaf] = (row.value_masked or "").strip()

        for receipt, labels in grouped.items():
            counterparty = labels.get(normalize_form_label("계약상대"), "")
            if not counterparty:
                continue
            head = re.split(r"[ ,(]", counterparty)[0]
            if len(head) >= 3 and head in question:
                documents.append(receipt)

    # 정기보고서 서술 — 「…보고서에서」 꼴이면 그 보고서 접수번호를 붙인다
    if "보고서" in question:
        from agent.periodic_document_preflight import PeriodicDocumentPreflight
        preflight = PeriodicDocumentPreflight(corpus)
        expressions = re.findall(
            r"(20\d{2}년\s*(?:1분기|2분기|3분기|반기|사업|분기)보고서)", question)
        for company in companies:
            for expression in expressions:
                try:
                    resolved = preflight.resolve_periodic_document(
                        corp_code=company.corp_code, as_of=CUTOFF,
                        target_expression=expression)
                except Exception:                          # noqa: BLE001
                    continue
                if resolved.status == "resolved" and resolved.candidate is not None:
                    documents.append(resolved.candidate.rcept_no)

    # 「가장 최근 정기보고서」처럼 연도가 없는 요청 — 마감 이전 최신 정기보고서
    if "정기보고서" in question or "최근" in question and "보고서" in question:
        import pyarrow.compute as _pc
        import pyarrow.parquet as _pq
        table = _pq.read_table(
            ROOT / "out/canonical/documents.parquet",
            columns=["corp_code", "doc_group", "rcept_no", "rcept_dt"])
        for company in companies:
            mask = _pc.and_(
                _pc.equal(table["corp_code"], company.corp_code),
                _pc.equal(table["doc_group"], "periodic"))
            rows_ = table.filter(mask).to_pylist()
            rows_ = [r for r in rows_ if str(r["rcept_dt"]) <= CUTOFF]
            if rows_:
                documents.append(max(rows_, key=lambda r: r["rcept_dt"])["rcept_no"])

    # 「정정공시」 — 그 회사의 정정 공시 접수번호
    if "정정" in question:
        import pyarrow.compute as _pc
        import pyarrow.parquet as _pq
        table = _pq.read_table(
            ROOT / "out/canonical/documents.parquet",
            columns=["corp_code", "is_correction", "rcept_no", "rcept_dt", "report_nm"])
        for company in companies:
            for row in table.filter(
                    _pc.equal(table["corp_code"], company.corp_code)).to_pylist():
                if row["is_correction"] and "단일판매" in (row["report_nm"] or ""):
                    documents.append(row["rcept_no"])

    # 상대 이름 없이 계약 전반을 묻는 요청 — 그 회사의 계약 공시
    if any(word in question for word in
           ("단일판매", "공급계약", "계약 해지", "계약 공시", "계약을 맺", "계약 맺")):
        import pyarrow.compute as _pc
        import pyarrow.parquet as _pq
        table = _pq.read_table(
            ROOT / "out/canonical/documents.parquet",
            columns=["corp_code", "rcept_no", "report_nm"])
        want_termination = "해지" in question
        for company in companies:
            for row in table.filter(
                    _pc.equal(table["corp_code"], company.corp_code)).to_pylist():
                name = row["report_nm"] or ""
                if "단일판매" not in name:
                    continue
                if want_termination and "해지" not in name:
                    continue
                if not want_termination and "해지" in name:
                    continue
                documents.append(row["rcept_no"])

    # 「2023년 2월 공시」처럼 기간만 준 정리 요청 — 그 기간의 공시를 전부 단다
    period = re.search(r"(20\d{2})년\s*(\d{1,2})월\s*공시", question)
    if period:
        prefix = f"{period.group(1)}{int(period.group(2)):02d}"
        for company in companies:
            for row in corpus.fields(as_of=CUTOFF, corp_code=company.corp_code):
                if str(row.rcept_no).startswith(prefix):
                    documents.append(row.rcept_no)

    return sorted(dict.fromkeys(documents)), claims


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path,
                        default=SET_ROOT / "questions_v0.4.jsonl")
    parser.add_argument("--judgment", type=Path,
                        default=SET_ROOT / "judgment_key_v01.jsonl")
    parser.add_argument("--out", type=Path,
                        default=SET_ROOT / "answer_facts_v01.jsonl")
    args = parser.parse_args(argv)

    from src.canonical.read import CanonicalReadModel
    corpus = CanonicalReadModel(ROOT / "out/canonical")

    questions = {row["question_id"]: row["question"]
                 for row in _load(args.questions)}
    judgment = {row["question_id"]: row for row in _load(args.judgment)}

    out_rows = []
    filled = 0
    for question_id, question in questions.items():
        documents, claims = extract(corpus, question_id, question)
        if documents or claims:
            filled += 1
        out_rows.append({
            "question_id": question_id,
            "expected_action": judgment.get(question_id, {}).get(
                "expected_action", "not_applicable"),
            "required_documents": documents,
            "required_claims": claims,
        })
    args.out.write_text("\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True)
        for row in out_rows) + "\n", encoding="utf-8")

    print(f"{args.out} · {len(out_rows)}행 · 사실 확보 {filled}문항")
    for row in out_rows:
        if row["required_documents"] or row["required_claims"]:
            print(f"  {row['question_id']}  문서 {len(row['required_documents'])} "
                  f"· 주장 {len(row['required_claims'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
