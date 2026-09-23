"""채점기 — AnswerPayload를 AnswerRequirement(v0.4)와 대조한다.

문장이 아니라 typed payload를 채점한다. 항목별 pass/fail을 내고, 문항 pass는
'필수 항목 전부 pass'다. team_reference_answer는 채점에 쓰지 않는다(비권위).

항목:
  status          final_status가 expected_action과 일치
  claims          required_claims의 값(value_text 또는 canonical_value)이 payload claims에 존재
  documents       required_documents ⊆ used_documents ∪ citation doc/rcept (superset 허용)
  forbidden       forbidden_claims의 숫자 토큰이 payload claims 값에 나타나지 않음
  limitations     required_limitation_codes ⊆ payload limitation/reason 코드
  premise         premise_verdict JSON과 payload premise_verdicts 일치
  clarify_target  clarify 문항: 기대 slot target(behavior/limitation에서 추론)이 포함
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.orchestrator.payload import AnswerPayload

_ACTION_TO_STATUS = {
    "answer": {"answer"},
    "partial_answer": {"partial_answer"},
    "clarify": {"clarify"},
    "refuse": {"refuse"},
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    required: bool = True


@dataclass
class ScoreCard:
    question_id: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.required)

    @property
    def summary(self) -> str:
        return " ".join(f"{'✓' if c.passed else '✗'}{c.name}" for c in self.checks)


def _num_tokens(s: str) -> set[str]:
    """숫자 토큰 정규화 — 콤마 제거, 소수점 유지, 부호·괄호 무시(크기 비교)."""
    return {t.replace(",", "").lstrip("-") for t in re.findall(r"\d[\d,]*(?:\.\d+)?", s or "")}


def _claim_values(p: AnswerPayload) -> set[str]:
    """payload claims의 숫자 토큰 집합 (+ 근거 접수번호 — 서술형 요건이 접수번호를 인용하는 경우)."""
    vals: set[str] = set()
    for c in p.claims:
        for v in (c.value_text, c.canonical_value, c.state, c.text):
            if v:
                vals |= _num_tokens(v)
        for ct in c.citations:
            if ct.rcept_no:
                vals.add(ct.rcept_no)
    return vals


def _claim_texts(p: AnswerPayload) -> str:
    """payload의 텍스트 값 결합 (비수치 claim 매칭용) — claims + 고지(applied_defaults) + limitation."""
    parts = []
    for c in p.claims:
        for v in (c.label, c.value_text, c.state, c.text):
            if v:
                parts.append(v)
    parts.extend(p.applied_defaults)
    parts.extend(l.detail for l in p.limitations)
    return " | ".join(parts)


_STATE_WORDS = {
    "active": ("유효", "active", "진행"), "terminated": ("해지", "terminated", "종료", "유효하지 않음"),
}
_KO_STATE_TOKENS = ("유효", "해지", "확인할 수 없음", "확인 불가", "다르다", "같다", "없음", "있음", "정정")


def _key_tokens(v: str) -> tuple[set[str], set[str], set[str]]:
    """서술문에서 채점 가능한 핵심 토큰 추출: (숫자, 날짜, 상태/고유명사)."""
    dates = set(re.findall(r"\d{4}-\d{2}-\d{2}", v))
    body = re.sub(r"\d{4}-\d{2}-\d{2}", " ", v)          # 날짜는 별도 축 — 조각(01·30)이 숫자 축에 새지 않게
    nums = {t for t in _num_tokens(body) if len(t) >= 3}   # 1~2자리 조각은 노이즈
    # 상태어·짧은 고유명사(영문 회사명 등)
    words = set()
    for w in _KO_STATE_TOKENS:
        if w in v:
            words.add(w)
    for m in re.findall(r"[A-Z][A-Za-z&\.\-]+(?:\s[A-Z][A-Za-z&\.\-,]+)*", v):
        if len(m) >= 4:
            words.add(m.split(",")[0].strip())
    return nums, dates, words


def _claim_matches(rc, have_nums: set[str], have_text: str) -> bool:
    """required claim 하나가 payload에 존재하는가.

    - 짧은 값(숫자/회사명/단어): 숫자 토큰 교집합 또는 텍스트 포함
    - 서술문(공백 포함 20자↑): 핵심 토큰(숫자·날짜·상태어·고유명사) 중 절반 이상이 payload에 있으면 인정
      — 문장 자체의 채점은 4단계(composer) 영역이며, payload 채점은 사실 요소의 존재를 본다.
    """
    have_low = have_text.lower()
    for v in (rc.value_text, rc.raw_value, rc.canonical_value):
        if not v or v == "not_applicable":
            continue
        nums = _num_tokens(v)
        is_narrative = len(v) >= 20 and " " in v
        if not is_narrative:
            if nums:
                first = re.search(r"\d[\d,]*(?:\.\d+)?", v).group(0).replace(",", "")
                if first in have_nums:
                    return True
            if v.lower() in have_low:
                return True
            # 상태 동의어: '유효하지 않음(해지됨)' ↔ terminated 등
            for st, ws in _STATE_WORDS.items():
                if any(w in v for w in ws) and (st in have_low or any(w in have_text for w in ws)):
                    return True
            continue
        n, d, w = _key_tokens(v)
        keys = [("n", x) for x in n] + [("d", x) for x in d] + [("w", x) for x in w]
        if not keys:
            return v.lower() in have_low
        hit = 0
        for kind, x in keys:
            if kind == "n" and x in have_nums: hit += 1
            elif kind == "d" and (x in have_text or x.replace("-", "") in have_text): hit += 1
            elif kind == "w" and x.lower() in have_low: hit += 1
        # 상태 claim은 상태어 하나로 결정: 상태어가 있으면 그것을 우선
        st_hits = [x for k, x in keys if k == "w" and x in _KO_STATE_TOKENS]
        if st_hits and all(x in have_text for x in st_hits) and hit >= max(1, len(keys) // 3):
            return True
        return hit * 2 >= len(keys)
    return False


def _payload_docs(p: AnswerPayload) -> set[str]:
    docs = set(p.used_documents)
    for c in p.claims:
        for ct in c.citations:
            if ct.rcept_no:
                docs.add(ct.rcept_no)
            # doc_id 형식 "periodic_20260310002820" 등에서 접수번호 추출
            m = re.search(r"(\d{14})", ct.doc_id or "")
            if m:
                docs.add(m.group(1))
    return docs


def _payload_codes(p: AnswerPayload) -> set[str]:
    return {l.code for l in p.limitations} | set(p.reasons)


def score_payload(p: AnswerPayload, req) -> ScoreCard:
    sc = ScoreCard(question_id=req.question_id)

    # 1) status
    want = _ACTION_TO_STATUS.get(req.expected_action, {req.expected_action})
    sc.checks.append(CheckResult("status", p.final_status in want,
                                 f"got={p.final_status} want={sorted(want)}"))

    is_answer = req.expected_action in ("answer", "partial_answer")

    # 2) claims (answer 계열만)
    #    - 사실형(수치·날짜·상태·고유명사·짧은 값): payload 채점 필수
    #    - 서술형(원문 요약·추론이 필요한 긴 문장, 사실 토큰이 없거나 판정어 위주): composer/LLM-judge 대상 (required=False)
    if is_answer and req.required_claims:
        have = _claim_values(p)
        have_text = _claim_texts(p)
        missing, narrative_missing, n_fact, n_narr = [], [], 0, 0
        for rc in req.required_claims:
            has_value = any(v and v != "not_applicable"
                            for v in (rc.value_text, rc.raw_value, rc.canonical_value))
            if not has_value:
                if not p.claims:
                    missing.append(rc.claim)
                continue
            v = rc.value_text if rc.value_text != "not_applicable" else (rc.raw_value if rc.raw_value != "not_applicable" else rc.canonical_value)
            n, d, w = _key_tokens(v)
            is_narr = len(v) >= 40 and " " in v and len(n | d) == 0 and not any(x in v for x in _KO_STATE_TOKENS)
            # 판정어만 있는 서술("…이유", "…의미", "…사유")도 서술형
            is_narr = is_narr or (len(v) >= 40 and any(k in rc.claim for k in ("사유", "이유", "의미", "내역", "★")))
            ok = _claim_matches(rc, have, have_text)
            if is_narr:
                n_narr += 1
                if not ok: narrative_missing.append(rc.claim)
            else:
                n_fact += 1
                if not ok: missing.append(rc.claim)
        sc.checks.append(CheckResult("claims", not missing,
                                     f"missing={missing[:3]}" if missing else f"{n_fact} fact matched"))
        if n_narr:
            sc.checks.append(CheckResult("narrative_claims", not narrative_missing,
                                         f"composer/LLM-judge 대상 {n_narr}건" + (f", payload 미포함={narrative_missing[:2]}" if narrative_missing else ""),
                                         required=False))
    elif is_answer:
        # 값 요건이 없는 answer(서술형·Open) — claim 존재만 요구
        sc.checks.append(CheckResult("claims", bool(p.claims), f"claims={len(p.claims)}"))

    # 3) documents
    if is_answer and req.required_documents:
        have_docs = _payload_docs(p)
        missing = [d for d in req.required_documents if d not in have_docs]
        # behavior_requirements의 plan_path_documents도 대체 경로로 인정
        alt_ok = False
        for b in req.behavior_requirements:
            if b.startswith("plan_path_documents="):
                alt = set(json.loads(b.split("=", 1)[1]))
                if alt <= have_docs:
                    alt_ok = True
        sc.checks.append(CheckResult("documents", not missing or alt_ok,
                                     f"missing={missing}" if missing and not alt_ok else "ok"))

    # 4) forbidden claims — 금지 수치가 '최종 답' 값으로 등장하면 실패.
    #    계산의 중간 operand(예: 반기 누적)는 최종 답이 아니므로 제외한다.
    if req.forbidden_claims and p.claims:
        derived = [c for c in p.claims if c.derived_from]
        finals = derived if derived else p.claims
        have = set()
        for c in finals:
            for v in (c.value_text, c.canonical_value, c.state, c.text):
                if v:
                    have |= _num_tokens(v)
        hits = []
        for fc in req.forbidden_claims:
            # 날짜·접수번호(14자리)는 금지 '값'이 아니라 맥락이므로 제외하고, 남은 숫자 중
            # 유의미한 크기(5자리 이상 금액 또는 소수)만 금지 값으로 본다.
            body = re.sub(r"\d{4}-\d{2}-\d{2}|\d{14}", " ", fc)
            cands = [t.replace(",", "") for t in re.findall(r"\d[\d,]*(?:\.\d+)?", body)]
            cands = [t for t in cands if ("." in t) or len(t) >= 5]
            if not cands:
                continue
            # "YYYY-MM-DD 시점에 … 제시" 형태면 그 시점의 claim(output_id에 @YYYYMMDD)만 검사
            m = re.search(r"(\d{4}-\d{2}-\d{2}) 시점", fc)
            scope = have
            if m:
                tp = m.group(1).replace("-", "")
                scope = set()
                for c in finals:
                    if f"@{tp}" in c.output_id:
                        for v in (c.value_text, c.canonical_value, c.state, c.text):
                            if v:
                                scope |= _num_tokens(v)
            elif ("원본" in fc and "최종" in fc
                  and any(token in fc for token in ("제시", "단정"))):
                # A requirement may prohibit presenting an original value *as
                # the final value* while simultaneously requiring an
                # original-to-final comparison.  Inspect final-labelled claims
                # only; otherwise the legitimate comparison becomes a false
                # forbidden hit.
                scope = set()
                for c in finals:
                    if (any(token in c.label for token in ("최신", "최종"))
                            and "원본" not in c.label):
                        for v in (c.value_text, c.canonical_value,
                                  c.state, c.text):
                            if v:
                                scope |= _num_tokens(v)
            if cands[0] in scope:
                hits.append(cands[0])
        sc.checks.append(CheckResult("forbidden", not hits, f"hits={hits}" if hits else "none"))

    # 5) limitation codes — 답변·거부 시점의 요건. clarify 문항은 제외
    if req.required_limitation_codes and req.expected_action != "clarify":
        have = _payload_codes(p)
        missing = [c for c in req.required_limitation_codes if c not in have]
        sc.checks.append(CheckResult("limitations", not missing,
                                     f"missing={missing}" if missing else "ok"))

    # 6) premise verdict — 답변 문항에서만
    if is_answer and req.premise_verdict and req.premise_verdict != "not_applicable":
        try:
            want_pv = json.loads(req.premise_verdict)
        except json.JSONDecodeError:
            want_pv = {}
        got_pv = {v.claim_id: v.verdict for v in p.premise_verdicts}
        got_seq = [v.verdict for v in p.premise_verdicts]   # ID 재생성 대비 순서 매칭
        bad = []
        for i, (cid, exp) in enumerate(want_pv.items()):
            g = got_pv.get(cid)
            if g is None and i < len(got_seq):
                g = got_seq[i]
            if exp is False and g != "false":
                bad.append(f"{cid}:{g}")
            elif exp is True and g != "true":
                bad.append(f"{cid}:{g}")
            elif isinstance(exp, str) and g is None:   # "verify_against_lineage" 등 — 판정 존재만 요구
                bad.append(f"{cid}:missing")
        sc.checks.append(CheckResult("premise", not bad, f"bad={bad}" if bad else "ok"))

    # 7) clarify target — 역질문 문항은 대상 축이 맞는지
    if req.expected_action == "clarify":
        got_targets = set(p.clarification.targets) if p.clarification else set()
        sc.checks.append(CheckResult("clarify_target", bool(got_targets),
                                     f"targets={sorted(got_targets)}", required=False))

    return sc


def score_answer_text(text: str, req, payload) -> list[CheckResult]:
    """W5: 최종 답변 문장 채점 — 필수 값·근거 접수번호 포함, 금지 값 미포함.
    payload 채점과 별개로 '문장에 실제로 실렸는가'를 본다."""
    out: list[CheckResult] = []
    # 값·인용 검사는 전체 문장, 금지값 검사는 "계산 근거:" 줄(operand)을 제외한 답 문장만
    full = (text or "").replace(",", "")
    answer_lines = [l for l in (text or "").splitlines() if not l.startswith("계산 근거:")]
    norm = "\n".join(answer_lines).replace(",", "")
    is_answer = req.expected_action in ("answer", "partial_answer")
    if is_answer and req.required_claims:
        vals = []
        for rc in req.required_claims:
            v = rc.value_text if rc.value_text != "not_applicable" else (rc.raw_value if rc.raw_value != "not_applicable" else None)
            if v and _num_tokens(v) and len(v) < 40:
                first = re.search(r"\d[\d,]*(?:\.\d+)?", v).group(0).replace(",", "")
                if len(first) >= 3:
                    canonical = getattr(rc, "canonical_value", None)
                    vals.append((first, canonical))
        if vals:
            # 답 문구는 이제 원문 자릿수("333605938")가 아니라 한국어 금액
            # 표기("333조 6,059억 3,800만원")로 나올 수 있다. 자릿수가
            # 문장에 그대로 없어도, 그 값의 원 단위 환산(canonical_value)이
            # 한국어 표기를 실제로 읽어낸 값과 같으면 실렸다고 본다.
            money_in_text = None
            miss = []
            for raw, canonical in vals:
                if raw in full:
                    continue
                if canonical and canonical.lstrip("-").isdigit():
                    if money_in_text is None:
                        from scripts.korean_amount import parse_amounts
                        money_in_text = parse_amounts(text or "")
                    if int(canonical) in money_in_text:
                        continue
                miss.append(raw)
            out.append(CheckResult("text_values", len(miss) * 2 <= len(vals), f"missing={miss[:3]}" if miss else f"{len(vals)} in text"))
    if is_answer and req.required_documents:
        have = [d for d in req.required_documents if d in (text or "")]
        alt_ok = any(b.startswith("plan_path_documents=") and all(x in (text or "") for x in json.loads(b.split("=",1)[1])) for b in req.behavior_requirements)
        out.append(CheckResult("text_citations", bool(have) or alt_ok, f"cited={len(have)}/{len(req.required_documents)}"))
    if req.forbidden_claims and is_answer:
        hits = []
        for fc in req.forbidden_claims:
            body = re.sub(r"\d{4}-\d{2}-\d{2}|\d{14}", " ", fc)
            cands = [t.replace(",", "") for t in re.findall(r"\d[\d,]*(?:\.\d+)?", body)]
            cands = [t for t in cands if ("." in t) or len(t) >= 5]
            m = re.search(r"(\d{4}-\d{2}-\d{2}) 시점", fc)
            if cands and cands[0] in norm and not m:   # 시점 한정 금지는 payload 채점에서 처리
                hits.append(cands[0])
        out.append(CheckResult("text_forbidden", not hits, f"hits={hits}" if hits else "none"))
    return out
