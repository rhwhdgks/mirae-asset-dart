#!/usr/bin/env python3
"""out/requests/*.jsonl 답변의 접수번호·금액·주식수·비율을 out/canonical 정본과 오프라인 대조.

사용법:
  cd <repo> && \
  PYTHONPATH=. .venv/bin/python <verify>/verify_d/crosscheck.py \
    --repo . --out <verify>/reports/D_canonical_crosscheck.md

읽기 전용. 서버 기동·HCX 호출 없음. 본체 저장소를 수정하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from decimal import Decimal
from pathlib import Path

import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).parent))
import units  # noqa: E402

REQUEST_FILES = [
    "final_7a22fb8_target38_live_20260904.jsonl",
    "final_7a22fb8_specgap_live_20260904.jsonl",
    "final_7a22fb8_p9gap_live_20260904.jsonl",
    "final_7a22fb8_covgap_live_20260904.jsonl",
    "final_d3ac793_hm02_live_20260904.jsonl",
]
HM_FILE = "final_d3ac793_hm02_live_20260904.jsonl"

REFUSAL_MARKERS = [
    "확인할 수 없습니다", "확인되지 않습니다", "확정하지 않습니다", "확정할 수 없습니다",
    "제시하지 않았습니다", "여쭤볼게요", "다시 질문해 주세요", "찾아서 제공하지 않습니다",
    "해석하지 못했습니다", "질문에만 답합니다",
]


def load_universe(path: Path) -> dict:
    aliases = {}  # alias(text) -> corp_code
    corp_info = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            code = row["corp_code"]
            corp_info[code] = row
            for key in ("corp_name", "listed_name"):
                name = (row.get(key) or "").strip()
                if name:
                    aliases[name] = code
    # 긴 이름부터 매칭(짧은 이름의 부분 문자열 오탐 방지)
    ordered = sorted(aliases.items(), key=lambda kv: -len(kv[0]))
    return {"aliases": ordered, "corp_info": corp_info}


def companies_in_text(text: str, universe: dict) -> set[str]:
    found = set()
    for name, code in universe["aliases"]:
        if name in text:
            found.add(code)
    return found


def load_documents(repo: Path) -> dict[str, list[dict]]:
    d = ds.dataset(str(repo / "out/canonical/documents.parquet"))
    t = d.to_table(columns=["rcept_no", "corp_code", "corp_name", "doc_id",
                             "doc_group", "doc_subtype", "report_nm", "rcept_dt"])
    idx: dict[str, list[dict]] = {}
    for row in t.to_pylist():
        idx.setdefault(row["rcept_no"], []).append(row)
    return idx


def load_facts_for_docs(repo: Path, doc_ids: set[str]) -> dict[str, list[dict]]:
    if not doc_ids:
        return {}
    d = ds.dataset(str(repo / "out/canonical/facts.parquet"))
    t = d.to_table(
        columns=["doc_id", "account_norm", "account_raw", "scope", "statement",
                 "period_end", "period_start", "cumulative", "raw_value", "raw_unit",
                 "value_status"],
        filter=ds.field("doc_id").isin(list(doc_ids)),
    )
    out: dict[str, list[dict]] = {}
    for row in t.to_pylist():
        if row["raw_value"] is None or row["raw_unit"] is None:
            continue
        scale = units.UNIT_SCALE.get(row["raw_unit"])
        if scale is None:
            continue
        try:
            won = Decimal(str(row["raw_value"])) * scale
        except Exception:
            continue
        row["_won"] = won
        out.setdefault(row["doc_id"], []).append(row)
    return out


_NUM_RE = re.compile(r"^-?\(?\d[\d,]*(?:\.\d+)?\)?$")


def _parse_field_decimal(v: str) -> Decimal | None:
    if v is None:
        return None
    s = v.strip()
    if not _NUM_RE.match(s):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "")
    try:
        d = Decimal(s)
    except Exception:
        return None
    return -d if neg else d


def load_fields_for_docs(repo: Path, doc_ids: set[str]) -> dict[str, list[dict]]:
    if not doc_ids:
        return {}
    d = ds.dataset(str(repo / "out/canonical/fields.parquet"))
    t = d.to_table(
        columns=["doc_id", "acode", "value_raw", "table_locator", "logical_row",
                 "logical_col", "is_pii"],
        filter=ds.field("doc_id").isin(list(doc_ids)),
    )
    out: dict[str, list[dict]] = {}
    for row in t.to_pylist():
        dec = _parse_field_decimal(row.get("value_raw"))
        if dec is None:
            continue
        row["_dec"] = dec
        out.setdefault(row["doc_id"], []).append(row)
    return out


def load_chunk_text_for_docs(repo: Path, doc_ids: set[str]) -> dict[str, str]:
    """서술형 원문 텍스트 — periodic 문서는 chunks.parquet, holding/major/exchange 등
    이벤트성 문서는 chunks.parquet 이 비어 있고 fields.parquet 의 비구조 value_raw(주석·
    정정사유 등 긴 문자열)에 같은 내용이 들어있어 둘 다 합친다."""
    if not doc_ids:
        return {}
    out: dict[str, list[str]] = {}
    d = ds.dataset(str(repo / "out/canonical/chunks.parquet"))
    t = d.to_table(columns=["doc_id", "text"],
                    filter=ds.field("doc_id").isin(list(doc_ids)))
    for row in t.to_pylist():
        if row["text"]:
            out.setdefault(row["doc_id"], []).append(row["text"])
    d2 = ds.dataset(str(repo / "out/canonical/fields.parquet"))
    t2 = d2.to_table(columns=["doc_id", "value_raw"],
                      filter=ds.field("doc_id").isin(list(doc_ids)))
    for row in t2.to_pylist():
        v = row["value_raw"]
        if v and len(v) > 8 and not _NUM_RE.match(v.strip()):
            out.setdefault(row["doc_id"], []).append(v)
    return {k: "\n".join(v) for k, v in out.items()}


def load_answers(repo: Path) -> list[dict]:
    records = []
    for fn in REQUEST_FILES:
        p = repo / "out/requests" / fn
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                resp = d.get("response") or {}
                qid = resp.get("question_id") or d.get("question_id")
                question = resp.get("question") or d.get("question") or ""
                answer = resp.get("answer") or ""
                records.append({
                    "source_file": fn,
                    "line_no": i,
                    "question_id": qid,
                    "question": question,
                    "answer": answer,
                })
    return records


def is_refusal(answer: str) -> bool:
    return any(m in answer for m in REFUSAL_MARKERS)


SCOPE_LABEL = {"CFS": "연결", "SFS": "별도"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()

    universe = load_universe(repo / "data/corpus/universe.csv")
    documents = load_documents(repo)

    answers = load_answers(repo)

    # Pass 1: 텍스트 추출
    for rec in answers:
        text = rec["answer"]
        rec["is_refusal"] = is_refusal(text) and not units.extract_money(text) \
            and not units.RCEPT_RE.search(text)
        rec["rcept_nos"] = units.extract_rcept_nos(text)
        rec["money"] = units.extract_money(text)
        rec["other"] = units.extract_other(text)
        rec["q_companies"] = companies_in_text(rec["question"], universe)
        rec["a_companies"] = companies_in_text(text, universe)

    # 인용된 접수번호 → doc_id 귀속 판정
    all_doc_ids = set()
    for rec in answers:
        rec["cited_docs"] = []  # list of document row dict (유효한 것만)
        rec["unknown_rcepts"] = []
        for rc in rec["rcept_nos"]:
            docs = documents.get(rc)
            if not docs:
                rec["unknown_rcepts"].append(rc)
                continue
            for doc in docs:
                rec["cited_docs"].append(doc)
                all_doc_ids.add(doc["doc_id"])

    facts_by_doc = load_facts_for_docs(repo, all_doc_ids)
    fields_by_doc = load_fields_for_docs(repo, all_doc_ids)

    # Pass 2: 값 대조
    defects = []          # 심각도순 결함 후보
    table_rows = []       # 문항별 표
    weak_evidence = []    # facts/fields 에는 없으나 본문(chunks)에서 문자열로 확인된 값(참고용)
    chunk_text_cache: dict[str, str] = {}

    def get_chunk_text(doc_ids: list[str]) -> str:
        missing = [d for d in doc_ids if d not in chunk_text_cache]
        if missing:
            fetched = load_chunk_text_for_docs(repo, set(missing))
            for d in missing:
                chunk_text_cache[d] = fetched.get(d, "")
        return "\n".join(chunk_text_cache[d] for d in doc_ids)

    for rec in answers:
        qid = rec["question_id"]
        text = rec["answer"]

        # ---- 규칙1: 접수번호 존재·귀속 ----
        for rc in rec["unknown_rcepts"]:
            defects.append({
                "severity": 1, "kind": "다른 문서 인용(접수번호 없음)",
                "question_id": qid, "source_file": rec["source_file"],
                "detail": f"접수번호 {rc} 가 정본 documents.parquet 에 없음",
                "repro": (f"documents.parquet 에서 rcept_no=='{rc}' 조회 결과 0건"
                          f" (out/canonical/documents.parquet)"),
            })
        cited_corp_codes = {d["corp_code"] for d in rec["cited_docs"]}
        cited_corp_names = {d["corp_name"] for d in rec["cited_docs"]}
        ref_companies = rec["q_companies"] | rec["a_companies"]
        if rec["cited_docs"] and ref_companies:
            mismatched = cited_corp_codes - ref_companies
            if mismatched and not (cited_corp_codes & ref_companies):
                # 인용 문서 회사가 질문/답변에 등장하는 어떤 회사와도 안 겹칠 때만 결함
                names = ", ".join(sorted(cited_corp_names))
                defects.append({
                    "severity": 2, "kind": "다른 회사 문서 인용",
                    "question_id": qid, "source_file": rec["source_file"],
                    "detail": f"인용 문서 회사({names})가 질문의 회사와 불일치",
                    "repro": (f"documents.parquet 에서 rcept_no in "
                              f"{[d['rcept_no'] for d in rec['cited_docs']]} 의 corp_name 확인"),
                })

        # ---- 값 풀 구성(인용 문서들의 facts/fields 전체) ----
        doc_ids = [d["doc_id"] for d in rec["cited_docs"]]
        pool: list[Decimal] = []
        pool_fact_meta: dict[str, list[dict]] = {}  # value(str) -> [fact rows] (스코프/기간 조회용)
        for did in doc_ids:
            for f in facts_by_doc.get(did, []):
                pool.append(f["_won"])
                if f["_won"] < 0:
                    # CF표 유출 항목은 정본에 음수로 저장되지만 답은 유출액을 양수로
                    # 말한다("취득 현금유출액은 X원") — 절대값도 풀에 같이 둔다.
                    pool.append(-f["_won"])
                pool_fact_meta.setdefault(str(f["_won"]), []).append(f)
            for fl in fields_by_doc.get(did, []):
                pool.append(fl["_dec"])
                if fl["_dec"] < 0:
                    pool.append(-fl["_dec"])
        poolset = set(pool)

        n_extracted = len(rec["money"]) + len(rec["other"])
        n_ok = 0
        n_bad = 0
        row_defects = []

        # ---- 규칙2: 금액 ----
        # 1차: facts/fields 풀과 직접·반올림·(풀 내부) 파생값 대조.
        money_status = []  # [span, status, hit_text_or_None]
        for m in rec["money"]:
            if not doc_ids:
                continue
            status, _ = _match_value(m.value_won, pool, approx=m.approx,
                                      granularity=m.granularity, poolset=poolset)
            money_status.append([m, status, None])
        # 2차: 아직 못 찾은 값은 본문(chunks) 원문 문자열로 대조(구조화 fact 아님, 참고용).
        for entry in money_status:
            m, status, _ = entry
            if status == "miss":
                hit = text_fallback_match(m.value_won, get_chunk_text(doc_ids), raw_span=m.text)
                if hit:
                    entry[1] = "text"
                    entry[2] = hit
        # 3차: 그래도 못 찾은 값은 "같은 답 안에서 이미 확인된 다른 금액들"의 합/차로 검산한다
        # (서술형 표의 두 시점 값 A·B 는 개별적으로 확인되지만 그 차이 C 는 본문에 따로
        # 적히지 않는 경우가 많다 — C == A-B 인지 재계산).
        verified_vals = {m.value_won for m, s, _ in money_status if s in ("direct", "approx", "derived", "text")}
        for entry in money_status:
            m, status, _ = entry
            if status == "miss" and verified_vals:
                for a in verified_vals:
                    b = a - m.value_won
                    if b in verified_vals and b != a:
                        entry[1] = "derived"
                        break
                    s = m.value_won - a
                    if s in verified_vals and s != a:
                        entry[1] = "derived"
                        break

        for m, status, hit in money_status:
            if status == "text":
                n_ok += 1
                weak_evidence.append({
                    "question_id": qid, "source_file": rec["source_file"],
                    "detail": f"답 값 {m.text} — facts/fields 구조화 표에는 없으나 본문(chunks) "
                              f"원문에 '{hit}' 문자열로 존재(구조화 fact 아님, 참고용)",
                })
            elif status in ("direct", "approx", "derived"):
                n_ok += 1
            else:
                n_bad += 1
                row_defects.append({
                    "severity": 1 if not m.approx else 4,
                    "kind": "값 불일치" if not m.approx else "표시 반올림 후보(미해결)",
                    "question_id": qid, "source_file": rec["source_file"],
                    "detail": f"답 값 {m.text}({m.value_won}원)이 인용 문서 facts/fields/본문/파생계산으로 확인 안 됨",
                    "repro": (f"CanonicalReadModel('out/canonical').facts(corp_code, as_of=AS_OF_ALL) "
                              f"중 doc_id in {doc_ids} 필터; 접수번호 {[d['rcept_no'] for d in rec['cited_docs']]}"),
                })

        # ---- 규칙2: 비율/배수/주식수 ----
        for o in rec["other"]:
            if not doc_ids:
                continue
            if o.kind == "share":
                status, _ = _match_value(o.value, pool, approx=False, granularity=Decimal(1), poolset=poolset)
            elif o.kind in ("percent", "percentpoint"):
                status, _ = _match_ratio(o.value, pool, poolset=poolset)
            elif o.kind == "multiple":
                status, _meta = _match_multiple(o.value, pool, poolset=poolset)
            else:
                status, _meta = "skip", None
            if status in ("direct", "approx", "derived"):
                n_ok += 1
            elif status == "skip":
                pass
            else:
                hit = text_fallback_match(o.value, get_chunk_text(doc_ids), raw_span=o.text,
                                           allow_scale=(o.kind == "share"))
                if hit:
                    n_ok += 1
                    weak_evidence.append({
                        "question_id": qid, "source_file": rec["source_file"],
                        "detail": f"답 값 {o.text} — facts/fields 구조화 표에는 없으나 본문(chunks) "
                                  f"원문에 '{hit}' 문자열로 존재(구조화 fact 아님, 참고용)",
                    })
                else:
                    n_bad += 1
                    row_defects.append({
                        "severity": 1,
                        "kind": f"값 불일치({o.kind})",
                        "question_id": qid, "source_file": rec["source_file"],
                        "detail": f"답 값 {o.text}({o.value})이 인용 문서 facts/fields/본문에서 확인·산출 안 됨",
                        "repro": (f"fields.parquet 에서 doc_id in {doc_ids} 의 value_raw 직접 대조; "
                                  f"또는 facts 두 값의 비율/증감률 재계산"),
                    })

        defects.extend(row_defects)
        table_rows.append({
            "question_id": qid, "source_file": rec["source_file"],
            "is_refusal": rec["is_refusal"], "n_rcept": len(rec["rcept_nos"]),
            "n_unknown_rcept": len(rec["unknown_rcepts"]),
            "n_extracted": n_extracted, "n_ok": n_ok, "n_bad": n_bad,
            "verdict": ("대조 대상 아님" if (rec["is_refusal"] and n_extracted == 0)
                        else ("결함 후보 있음" if n_bad or rec["unknown_rcepts"] else "일치")),
        })

    parser_gaps = find_parser_gaps(answers)
    write_report(Path(args.out), answers, table_rows, defects, weak_evidence, parser_gaps)
    print(f"문항 {len(answers)}건 처리. 결함 후보 {len(defects)}건 "
          f"(본문 참고 통과 {len(weak_evidence)}건 별도). 보고서: {args.out}")


def _match_value(value: Decimal, pool: list[Decimal], approx: bool, granularity: Decimal,
                  poolset: set[Decimal] | None = None):
    poolset = poolset if poolset is not None else set(pool)
    if value in poolset:
        return "direct", value
    # 표시 반올림 허용폭 — 「약」표기든 아니든, 답이 보인 소수 자릿수만큼만 허용한다
    # (예: 「80,991.5억원」→0.1억 허용, 「약 4,791억원」→1억 허용). 정수 표기는 허용폭이
    # 원 단위까지 좁아져 사실상 정확 일치만 통과한다.
    tol = granularity
    for p in poolset:
        if abs(p - value) <= tol:
            return "approx", p
    # 파생값: 풀 내 두 값의 합/차 — O(n) (집합 조회)
    for a in poolset:
        b = a - value
        if b in poolset and b != a:
            return "derived", (a, b, "diff")
        s = value - a
        if s in poolset and s != a:
            return "derived", (a, s, "sum")
    return "miss", None


def _match_ratio(value: Decimal, pool: list[Decimal], poolset: set[Decimal] | None = None):
    poolset = poolset if poolset is not None else set(pool)
    if value in poolset:
        return "direct", value
    for a in poolset:
        for b in poolset:
            if a == b or b == 0:
                continue
            pct = (a - b) / b * 100
            if abs(pct - value) <= Decimal("0.02"):
                return "derived", (a, b, "growth_pct")
    return "miss", None


_TEXT_SCALES = [Decimal(1), Decimal(10) ** 3, Decimal(10) ** 4, Decimal(10) ** 6,
                Decimal(10) ** 8, Decimal(10) ** 9, Decimal(10) ** 12]


def _fmt_commas(n: Decimal) -> str:
    return f"{int(n):,}"


def text_fallback_match(value: Decimal, text: str, raw_span: str | None = None,
                         allow_scale: bool = True) -> str | None:
    """facts/fields 구조화 표에 없는 값(사업부문별 매출 등 서술형 표)을 본문 원문에서 찾는다.

    구조화 canonical 이 아니므로 통과해도 약한 근거("본문에서 확인됨")로만 표기한다.
    """
    if not text:
        return None
    if raw_span:
        # 원문 표기 그대로("9.8조원", "10%") 나오는지 먼저 본다 — 소수 표기(3.33조원 등)는
        # 정수 스케일로 못 옮기므로 이 tier 가 유일한 확인 경로다.
        pat = r"\s+".join(re.escape(tok) for tok in raw_span.strip().split())
        if re.search(pat, text):
            return raw_span.strip()
    if not allow_scale:
        return None
    # 음수(현금유출·감소 등)는 원문에서 부호를 '-' 대신 '△'·괄호로 적는 경우가 많다
    # (예: 「△301,146」). 절대값 자릿수도 같이 찾아본다 — 부호 자체는 이미 raw_span
    # 단계에서 원문 그대로 못 찾았다는 뜻이므로 크기만 이 tier 에서 재확인한다.
    candidates = {value}
    if value < 0:
        candidates.add(-value)
    for cand in candidates:
        for scale in _TEXT_SCALES:
            if cand % scale != 0:
                continue
            n = cand / scale
            if n == 0:
                continue
            formatted = _fmt_commas(n)
            pat = re.compile(r"(?<![\d,])" + re.escape(formatted) + r"(?![\d,])")
            if pat.search(text):
                return formatted
    return None


def _match_multiple(value: Decimal, pool: list[Decimal], poolset: set[Decimal] | None = None):
    poolset = poolset if poolset is not None else set(pool)
    for a in poolset:
        for b in poolset:
            if a == b or b == 0:
                continue
            ratio = a / b
            if abs(ratio - value) <= Decimal("0.005"):
                return "derived", (a, b, "ratio")
    return "miss", None


_USD_RE = re.compile(r"\$[\d,]+(?:\.\d+)?")


def find_parser_gaps(answers) -> list[dict]:
    """이 스크립트가 통째로 못 읽는 표기 — 결함으로 세지 않고 별도로만 기록한다."""
    gaps = []
    for rec in answers:
        usd = _USD_RE.findall(rec["answer"])
        if usd:
            gaps.append({"question_id": rec["question_id"], "source_file": rec["source_file"],
                         "detail": f"달러(USD) 표기 {sorted(set(usd))} — 이 파서는 원화(KRW) "
                                   f"조/억/만/원 표기만 환산한다. 대조 대상에서 제외."})
    return gaps


def write_report(out_path: Path, answers, table_rows, defects, weak_evidence, parser_gaps):
    lines = []
    lines.append("# D. 정본 대조 검증 보고서\n")
    lines.append(f"- 대상 답변 {len(answers)}건 (target38 38 / specgap 18 / p9gap 20 / covgap 60 / hm02 24)")
    lines.append(f"- 결함 후보 {len(defects)}건, 본문(chunks) 참고 통과(구조화 fact 아님) {len(weak_evidence)}건\n")
    lines.append("## 문항별 표\n")
    lines.append("| question_id | 파일 | 추출값수 | 일치 | 불일치 | 판정 |")
    lines.append("|---|---|---:|---:|---:|---|")
    for r in table_rows:
        lines.append(f"| {r['question_id']} | {r['source_file']} | {r['n_extracted']} | "
                      f"{r['n_ok']} | {r['n_bad']} | {r['verdict']} |")
    lines.append("\n## 결함 후보(심각도순: 값 오류 > 다른 문서 인용 > 기간·범위 불일치 > 표시 반올림)\n")
    for d in sorted(defects, key=lambda x: x["severity"]):
        lines.append(f"### [{d['kind']}] {d['question_id']} ({d['source_file']})")
        lines.append(f"- {d['detail']}")
        lines.append(f"- 재현: {d['repro']}")
        lines.append("")
    lines.append("\n## 참고: 구조화 정본(facts/fields)에는 없으나 본문(chunks) 원문 문자열로 확인된 값\n")
    lines.append("(사업부문별 매출 등 서술형 표는 facts.parquet/fields.parquet 에 구조화되지 않고 "
                  "chunks.parquet 원문 텍스트에만 있다 — 결함 아님, 대조 신뢰도가 낮음을 표시)\n")
    for w in weak_evidence:
        lines.append(f"- {w['question_id']} ({w['source_file']}): {w['detail']}")

    lines.append("\n## 파서 한계(결함 아님)\n")
    lines.append("- 마크다운 표 셀의 숫자 중 단위 접미사(원/주/%/배/조/억/만)가 붙지 않은 "
                  "순수 숫자(예: P9-001 표의 「1,167,225」 같은 수주잔고·재무지표 원자료 셀)는 "
                  "이 파서가 추출하지 않는다 — 단위가 없으면 어느 스케일로 정본과 대조할지 "
                  "정할 수 없어서다. p9gap 계열 답변의 표 셀 상당수가 여기 해당하며, 표 형태의 "
                  "서술형 원문(공시 인용)이라 결함으로 세지 않았다.")
    for g in parser_gaps:
        lines.append(f"- {g['question_id']} ({g['source_file']}): {g['detail']}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
