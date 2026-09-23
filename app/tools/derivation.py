"""DerivationExecutor — 검증된 output만으로 Decimal 계산.

합의안 원칙: LLM이 계산하지 않는다. Evidence 미검증 값은 계산에 안 쓴다. 계산값은 새
Evidence를 만들지 않고 피연산자의 citation을 계승한다. 단위가 다르면 unit_mismatch로 닫는다.

v0.4가 승인한 연산: difference · absolute_difference · percent_change ·
discrete_from_cumulative · argmax · concept_ratio(이슈 #38 — 지표 간 나눗셈) ·
sum(이슈 #124 — 같은 concept·같은 기간·같은 scope 인 값들의 합계)
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

from app.orchestrator.payload import (
    AnswerClaim, Limitation, RankingEntry, TraceEvent,
)
from app.textkit import fold_shared_label_prefix, josa


class VerifiedScalar:
    def __init__(self, output_id: str, value: Decimal, unit: str | None, won: Decimal | None,
                 claim: AnswerClaim):
        self.output_id, self.value, self.unit, self.won, self.claim = output_id, value, unit, won, claim


def _fmt(d: Decimal) -> str:
    q = d.normalize()
    return f"{q:,f}" if q == q.to_integral_value() else f"{q:,}"


#: 원문이 괄호(음수)로 적히는 **현금유출** 개념. 이들끼리의 「더 크다」는 유출 크기(절댓값) 비교다.
#: 손익 적자(영업이익·순이익 음수)나 잔액 음수는 부호 그대로 비교한다 — -50억이 -100억보다 크다.
_OUTFLOW_CONCEPTS = frozenset({
    "capex_ppe", "capex_intangible", "dividends_paid", "interest_paid",
})


def _outflow_magnitudes(vals, fact_meta: dict[str, dict] | None) -> bool:
    """현금유출 operand 를 유출 크기(절댓값)로 비교해도 되는가.

    재무 Fact의 개념이 현금유출이면 원문 표의 부호 표기가 회사별로 섞어도
    전부 절댓값으로 비교한다. 손익 적자·잔액 음수는 부호 그대로다.
    개념을 모르는 값(사건·정정 금액)은 종전 호환을 위해 전부 음수일 때만
    괄호 표기 유출로 본다."""
    known = [concept for concept in (
        ((fact_meta or {}).get(v.output_id) or {}).get("concept") for v in vals) if concept]
    if known:
        return len(known) == len(vals) and all(
            concept in _OUTFLOW_CONCEPTS for concept in known)
    return all(
        (v.won if v.won is not None else v.value) < 0 for v in vals)


def _period_over_period_order(vals, fact_meta: dict[str, dict] | None):
    """기간 비교의 피연산자를 시간 순서로 세운다 — 나중 기간이 앞.

    `difference`·`percent_change` 는 첫 피연산자를 당기로, 둘째를 전기로 읽는다.
    그 순서가 계획에서는 질문에 나온 순서를 따르므로, 「2024년에서 2025년
    사이에 얼마나 변했나」처럼 이른 해를 먼저 말하면 당기와 전기가 뒤바뀐다.
    매출이 늘었는데 「감소」로 답하고, 증감률의 분모로 나중 값을 써서
    「2024년 매출액 전기 대비」라는 자기모순 문구가 나왔다.

    「전기 대비」는 시간이 정하는 말이지 어순이 정하는 말이 아니다. 같은
    회사·개념·범위·기간유형을 서로 다른 기간에 읽은 두 값일 때만 나중
    기간을 앞에 세운다. `period_start` 가 둘 다 있으면 구간 길이가 같을
    때만 순서를 바꾼다 — 누적 구간이 서로 다른 `discrete_from_cumulative`
    (3분기 누적 - 상반기 누적)는 뺄 순서 자체가 뜻이므로 건드리지 않는다.
    `period_start` 가 없는 레거시 fact_meta는 `period_end` 만으로 시간순을
    판단한다(그래도 corp_code/concept/scope/period_type 동등성은 유지한다).
    """

    if len(vals) != 2:
        return vals
    left = (fact_meta or {}).get(vals[0].output_id) or {}
    right = (fact_meta or {}).get(vals[1].output_id) or {}
    if not left or not right:
        return vals
    # 이슈 #64 — 같은 회사·계정·범위인데 기간유형(연간/분기 등)이 다르면
    # 애초에 「같은 길이의 두 시점」이 아니다. period_type이 다르다는 이유로
    # 아래 4-key 가드가 조용히 재정렬만 건너뛰면, 연산 자체는 그대로 진행돼
    # 「연간 매출 - 4분기 단독 매출」처럼 뜻이 서지 않는 차이가 「분기 성장률」
    # 처럼 읽히는 결과를 낸다. 길이 차이가 기존 15일 허용치를 넘으면 재정렬이
    # 아니라 계산 자체를 거절한다. period_type까지 같은 페어(3분기 누적 -
    # 상반기 누적 같은 누적 구간 차이)는 이 분기 대신 아래 기존 경로로 간다.
    if (all(left.get(key) == right.get(key)
            for key in ("corp_code", "concept", "scope"))
            and left.get("period_type") != right.get("period_type")):
        try:
            from datetime import date
            l_end = date.fromisoformat(left["period_end"])
            r_end = date.fromisoformat(right["period_end"])
            l_start = date.fromisoformat(left["period_start"])
            r_start = date.fromisoformat(right["period_start"])
        except (KeyError, TypeError, ValueError):
            l_end = r_end = l_start = r_start = None
        if None not in (l_end, r_end, l_start, r_start):
            l_days = (l_end - l_start).days + 1
            r_days = (r_end - r_start).days + 1
            if abs((l_end - l_start) - (r_end - r_start)).days > 15:
                raise _PeriodMismatch(
                    "기간유형이 달라(예: 연간 vs 분기) 두 기간 길이가 "
                    f"크게 다름({l_days}일 vs {r_days}일) — 시간 증감으로 "
                    "계산하지 않음")
    if any(left.get(key) != right.get(key)
           for key in ("corp_code", "concept", "scope", "period_type")):
        return vals
    try:
        from datetime import date
        left_end = date.fromisoformat(left["period_end"])
        right_end = date.fromisoformat(right["period_end"])
    except (KeyError, TypeError, ValueError):
        return vals
    # ``period_start`` is present-but-``None`` for an instant/point-in-time
    # fact (e.g. ``이익잉여금`` at a balance date) — there is no interval to
    # measure, so a bare key-presence check here used to walk into
    # ``date.fromisoformat(None)``, raise, and silently keep the unordered
    # question-text operands instead of reordering by ``period_end`` below.
    # Only run the cumulative-length guard when both sides actually carry a
    # start date; two point-in-time facts skip straight to the date compare.
    if left.get("period_start") and right.get("period_start"):
        try:
            left_start = date.fromisoformat(left["period_start"])
            right_start = date.fromisoformat(right["period_start"])
        except (TypeError, ValueError):
            return vals
        # 길이가 다르면 누적 구간 차이(분기 단독 계산)라 순서가 곧 뜻이다.
        # 윤년이면 같은 연간 구간도 하루 어긋나므로 정확 일치로 보면 안 된다.
        # 분기 누적끼리는 90일 넘게 벌어지니 15일이면 둘을 가른다.
        if abs((left_end - left_start) - (right_end - right_start)).days > 15:
            return vals
    return (vals[1], vals[0]) if left_end < right_end else vals


def _discrete_period_label(a: VerifiedScalar, b: VerifiedScalar,
                           fact_meta: dict[str, dict] | None) -> str:
    """Build a standalone-quarter label from typed operand coordinates.

    Labels are presentation, not authority.  Only replace the legacy label
    when both facts prove the same issuer/concept/scope and their cumulative
    duration boundaries identify one quarter.  Otherwise retain the old safe
    label rather than guessing from a question or a localized label string.
    """

    left = (fact_meta or {}).get(a.output_id) or {}
    right = (fact_meta or {}).get(b.output_id) or {}
    if not left or not right:
        return a.claim.label
    if any(left.get(key) != right.get(key)
           for key in ("corp_code", "concept", "scope")):
        return a.claim.label
    if left.get("cumulative") is not True or right.get("cumulative") is not True:
        return a.claim.label
    try:
        from datetime import date
        left_start = date.fromisoformat(left["period_start"])
        left_end = date.fromisoformat(left["period_end"])
        right_start = date.fromisoformat(right["period_start"])
        right_end = date.fromisoformat(right["period_end"])
    except (KeyError, TypeError, ValueError):
        return a.claim.label
    if left_start != right_start or left_end <= right_end:
        return a.claim.label
    quarter_by_month = {3: 1, 6: 2, 9: 3, 12: 4}
    quarter = quarter_by_month.get(left_end.month)
    previous = quarter_by_month.get(right_end.month)
    if (quarter is None or previous != quarter - 1
            or left_end.year != right_end.year):
        return a.claim.label
    scope = {"CFS": "연결", "SFS": "별도"}.get(left.get("scope"))
    concept = str(left.get("concept") or "").strip()
    corp_name = str(left.get("corp_name") or "").strip()
    if not scope or not concept:
        return a.claim.label
    # The verified base claim already owns the public Korean account name.
    # Remove only its proved coordinate prefix and preserve the remaining
    # concept surface, so this works for every supported financial concept.
    metric = a.claim.label
    for token in (corp_name, f"{left_end.year}년", scope, "누적"):
        if token:
            metric = metric.replace(token, " ", 1)
    metric = " ".join(metric.split()) or concept
    prefix = f"{corp_name} " if corp_name else ""
    return f"{prefix}{left_end.year}년 {quarter}분기 단독 {scope} {metric}"


def _discrete_output_fact_meta(a: VerifiedScalar, b: VerifiedScalar,
                               fact_meta: dict[str, dict] | None) -> dict | None:
    """Synthetic fact coordinates for a ``discrete_from_cumulative`` result.

    The output itself is never a plan ``FactSpec`` so it has no entry of its
    own in ``fact_meta`` — a later ``difference``/``percent_change`` that
    (incorrectly) reaches for it therefore sees no coordinates and skips the
    period-length guard above entirely (issue #64). Register one only when
    the same guard `_discrete_period_label` already trusts (same issuer/
    concept/scope, both cumulative, matching start, ``a`` the longer span)
    holds, so the synthetic period is exactly the derived quarter window.
    """

    left = (fact_meta or {}).get(a.output_id) or {}
    right = (fact_meta or {}).get(b.output_id) or {}
    if not left or not right:
        return None
    if any(left.get(key) != right.get(key)
           for key in ("corp_code", "concept", "scope")):
        return None
    if left.get("cumulative") is not True or right.get("cumulative") is not True:
        return None
    try:
        from datetime import date, timedelta
        left_start = date.fromisoformat(left["period_start"])
        left_end = date.fromisoformat(left["period_end"])
        right_start = date.fromisoformat(right["period_start"])
        right_end = date.fromisoformat(right["period_end"])
    except (KeyError, TypeError, ValueError):
        return None
    if left_start != right_start or left_end <= right_end:
        return None
    return {
        "corp_code": left.get("corp_code"), "corp_name": left.get("corp_name"),
        "concept": left.get("concept"), "scope": left.get("scope"),
        "period_type": "quarter", "cumulative": False,
        "period_start": (right_end + timedelta(days=1)).isoformat(),
        "period_end": left_end.isoformat(),
    }


class DerivationExecutor:
    def execute(self, derivations, verified: dict[str, VerifiedScalar], *,
                trace: list[TraceEvent],
                fact_meta: dict[str, dict] | None = None) -> tuple[list[AnswerClaim], list[Limitation]]:
        claims: list[AnswerClaim] = []
        lims: list[Limitation] = []
        # 계산 결과도 후속 계산의 operand가 될 수 있다
        pool: dict[str, VerifiedScalar] = dict(verified)
        # 로컬 사본 — discrete_from_cumulative 결과의 합성 좌표(위 함수)를
        # 같은 실행 안의 후속 difference/percent_change가 읽을 수 있어야
        # 하지만, 호출자가 준 dict를 이 실행 밖에서까지 바꾸지는 않는다.
        fact_meta = dict(fact_meta) if fact_meta else {}

        for d in derivations:
            ops = [r.output_id for r in d.operands]
            missing = [o for o in ops if o not in pool]
            if missing:
                lims.append(Limitation(code="missing_operand",
                                       detail=f"{d.output_id}: 검증된 operand 없음 {missing}"))
                trace.append(TraceEvent(seq=len(trace)+1, stage="derivation",
                                        summary=f"{d.operator} {d.output_id} 건너뜀 — operand 미검증 {missing}", detail={}))
                continue
            vals = [pool[o] for o in ops]
            # Preserve at least one citation per operand before appending
            # secondary citations.  A correction operand can carry both its
            # before/after cells; naive flattening placed those first and the
            # public two-citation renderer then hid the other operand's
            # receipt entirely.
            primary_cites = []
            secondary_cites = []
            seen = set()
            for v in vals:
                operand_cites = []
                for c in v.claim.citations:
                    k = (c.doc_id, c.evidence_id, c.locator)
                    if k not in seen:
                        seen.add(k)
                        operand_cites.append(c)
                if operand_cites:
                    primary_cites.append(operand_cites[0])
                    secondary_cites.extend(operand_cites[1:])
            cites = primary_cites + secondary_cites

            try:
                if d.operator == "argmax":
                    # 단위가 다르면 원 환산값으로 비교, 그것도 없으면 unit_mismatch
                    # 현금유출 개념(capex 등)이 전부 음수(괄호 표기)면 유출 크기(절댓값)로 비교한다.
                    # 손익 적자는 부호 그대로 비교한다 (_outflow_magnitudes 참조).
                    all_neg = _outflow_magnitudes(vals, fact_meta)
                    if len({v.unit for v in vals}) > 1:
                        if any(v.won is None for v in vals):
                            raise _Unit(f"argmax operands 단위 불일치 {[v.unit for v in vals]}")
                        keyed = [((abs(v.won) if all_neg else v.won), v) for v in vals]
                    else:
                        keyed = [((abs(v.value) if all_neg else v.value), v) for v in vals]
                    # 이슈 #124 — argmax 연산자는 그대로 두고 극값 방향만
                    # 뒤집는다.  기본(direction="maximum")은 기존 동작과
                    # 완전히 같다.  ``getattr``은 direction 필드가 없는(예:
                    # 테스트 이중체) 구식 Derivation과의 호환을 지킨다.
                    direction = getattr(d, "direction", "maximum")
                    minimize = direction == "minimum"
                    if len(vals) in {3, 4}:
                        # v0.4 already permits N-ary argmax.  For 3~4 operands,
                        # consume that same plan as a full deterministic
                        # competition ranking (1,2,2,4), retaining source order
                        # as the display tie-break without breaking the tie.
                        indexed = list(enumerate(keyed))
                        sort_key = (
                            (lambda row: (row[1][0], row[0])) if minimize else
                            (lambda row: (-row[1][0], row[0])))
                        ordered = sorted(indexed, key=sort_key)
                        ranking: list[RankingEntry] = []
                        previous_key = None
                        previous_rank = 0
                        for position, (_source_index, (key, value)) in enumerate(
                                ordered, start=1):
                            rank = (previous_rank if previous_key is not None
                                    and key == previous_key else position)
                            ranking.append(RankingEntry(
                                rank=rank,
                                output_id=value.output_id,
                                label=(value.claim.state
                                       or value.claim.label.split(" ")[0]),
                                value_text=value.claim.value_text,
                                raw_unit=value.claim.raw_unit,
                                canonical_value=value.claim.canonical_value,
                                canonical_unit=value.claim.canonical_unit,
                            ))
                            previous_key, previous_rank = key, rank
                        winner = ranking[0]
                        claim = AnswerClaim(
                            output_id=d.output_id, label="비교 순위",
                            state=winner.label,
                            text="; ".join(
                                f"{entry.rank}위 {entry.label}"
                                for entry in ranking),
                            derived_from=ops, operator="argmax",
                            direction=direction,
                            citations=cites, ranking=ranking,
                        )
                        claims.append(claim)
                        trace.append(TraceEvent(
                            seq=len(trace)+1, stage="derivation",
                            summary=f"argmax competition ranking({direction}) → {winner.output_id}",
                            detail={
                                "operands": ops,
                                "ranking": [entry.model_dump(mode="json")
                                            for entry in ranking],
                            },
                        ))
                        continue
                    best = min(keyed, key=lambda kv: kv[0]) if minimize \
                        else max(keyed, key=lambda kv: kv[0])
                    ties = [v for k, v in keyed if k == best[0]]
                    if len(ties) > 1:
                        lims.append(Limitation(code="argmax_tie", detail=f"{d.output_id}: 동률 {[t.output_id for t in ties]}"))
                        continue
                    win = best[1]
                    claim = AnswerClaim(output_id=d.output_id, label="비교 결과",
                                        state=win.claim.state or win.claim.label.split(" ")[0],
                                        text=win.claim.label, derived_from=ops, operator="argmax",
                                        direction=direction,
                                        citations=cites)
                    claims.append(claim)
                    trace.append(TraceEvent(seq=len(trace)+1, stage="derivation",
                                            summary=f"argmax({direction}) → {win.output_id}", detail={"operands": ops}))
                    # 2항 비교면 차이도 보조 산출 — 비교 질문의 정답은 승자와 함께 차이값을 요구한다
                    if len(vals) == 2:
                        va, vb = vals
                        if va.unit == vb.unit:
                            x2, y2, u2 = va.value, vb.value, va.unit
                        elif va.won is not None and vb.won is not None:
                            x2, y2, u2 = va.won, vb.won, "원"
                        else:
                            x2 = None
                        if x2 is not None:
                            if all_neg:          # 현금유출 크기 비교와 같은 기준
                                x2, y2 = abs(x2), abs(y2)
                            for sc in self._supplements(d, derivations, va, vb, x2, y2, u2, cites, fact_meta):
                                claims.append(sc)
                                trace.append(TraceEvent(seq=len(trace)+1, stage="derivation",
                                                        summary=f"보조값 {sc.operator} → {sc.value_text}{sc.raw_unit or ''}",
                                                        detail={"output_id": sc.output_id}))
                    continue

                if d.operator == "sum":
                    # 이슈 #124 — 같은 concept·같은 scope·같은 period(회사만
                    # 다름) 또는 같은 회사의 서로 다른 period 값 2개(이상)를
                    # 더한다. 단위가 같으면 원문 Decimal 그대로, 다르면 원
                    # 환산값으로, 그것도 없으면 unit_mismatch로 닫는다 —
                    # difference/argmax와 같은 원칙이다. N-ary(argmax와
                    # 같이 2개 이상)를 받아 두되, 컴파일러는 지금 2개만
                    # 낸다.
                    units = {v.unit for v in vals}
                    if len(units) > 1:
                        if any(v.won is None for v in vals):
                            raise _Unit(f"sum operands 단위 불일치 {[v.unit for v in vals]}")
                        addends = [v.won for v in vals]
                        out_unit = "원"
                    else:
                        addends = [v.value for v in vals]
                        out_unit = next(iter(units))
                    r = Decimal("0")
                    for addend in addends:
                        r += addend
                    if len(vals) == 2:
                        a, b = vals
                        # 이슈 #118과 같은 원칙 — 공유하는 회사명·기간·기준을
                        # 한 번만 쓴다(같은 회사의 다른 기간 합계). 공유
                        # 접두가 없으면(회사가 다른 합계) 두 라벨을 그대로
                        # 잇는다 — 그래서 「삼성전자 2025년 연결 매출액과
                        # SK하이닉스 2025년 연결 매출액의 합계」처럼 나온다.
                        folded = fold_shared_label_prefix(a.claim.label, b.claim.label)
                        if folded is not None:
                            prefix, remainder_a, remainder_b = folded
                            label = (
                                f"{prefix} {remainder_a}"
                                f"{josa(remainder_a, '과', '와')} {remainder_b}의 합계")
                        else:
                            label = (
                                f"{a.claim.label}{josa(a.claim.label, '과', '와')} "
                                f"{b.claim.label}의 합계")
                    else:
                        label = (
                            "·".join(v.claim.label for v in vals) + "의 합계")
                    won_out = None
                    if out_unit == "원":
                        won_out = r
                    elif out_unit and out_unit not in {"%", "%p", "배"}:
                        from app.tools._units import to_won
                        won_out = to_won(r, out_unit)
                    claim = AnswerClaim(
                        output_id=d.output_id, label=label,
                        value_text=_fmt(r), raw_unit=out_unit,
                        canonical_value=str(won_out) if won_out is not None else None,
                        canonical_unit="원" if won_out is not None else None,
                        derived_from=ops, operator="sum", citations=cites)
                    claims.append(claim)
                    pool[d.output_id] = VerifiedScalar(
                        d.output_id, r, out_unit, won_out, claim)
                    trace.append(TraceEvent(
                        seq=len(trace)+1, stage="derivation",
                        summary=f"sum {ops} → {_fmt(r)} {out_unit}",
                        detail={"output_id": d.output_id}))
                    continue

                if len(vals) != 2:
                    raise _Arity(f"{d.operator} requires 2 operands, got {len(vals)}")
                if d.operator in ("difference", "percent_change"):
                    # 이슈 #35 — 같은 계열의 두 기간은 **시간순**으로 뺀다. 플랜이
                    # 「2024년, 2025년」 순서로 operand 를 내면 `2024 − 2025` 가 되어
                    # 실제로는 줄었는데 「(증가)」로 읽혔다. 다른 계열(회사·계정 비교)
                    # 은 operand 순서가 곧 질문의 방향이므로 그대로 둔다.
                    original_ops = ops
                    vals = _period_over_period_order(vals, fact_meta)
                    ops = [v.output_id for v in vals]
                    if ops != original_ops:
                        trace.append(TraceEvent(
                            seq=len(trace)+1, stage="derivation",
                            summary=f"{d.operator} {d.output_id} operand 를 시간순(늦은 기간 − 이른 기간)으로 정렬",
                            detail={"operands": original_ops}))
                a, b = vals
                # 단위 통일: 같은 단위면 원문 Decimal, 다르면 원 환산
                if a.unit == b.unit:
                    x, y, unit = a.value, b.value, a.unit
                elif a.won is not None and b.won is not None:
                    x, y, unit = a.won, b.won, "원"
                else:
                    raise _Unit(f"단위 불일치 {a.unit} vs {b.unit} (환산 불가)")

                if d.operator == "equal":
                    same = x == y
                    claim = AnswerClaim(
                        output_id=d.output_id, label="동일 여부",
                        state="same" if same else "different",
                        text="같다" if same else "다르다",
                        derived_from=ops, operator="equal", citations=cites)
                    claims.append(claim)
                    trace.append(TraceEvent(
                        seq=len(trace)+1, stage="derivation",
                        summary=f"equal {ops} → {claim.state}", detail={}))
                    continue
                if d.operator in ("difference", "absolute_difference"):
                    if _outflow_magnitudes((a, b), fact_meta):   # 현금유출(음수 표기)끼리는 크기 차이
                        x, y = abs(x), abs(y)
                    r = x - y
                    if d.operator == "absolute_difference":
                        r = abs(r)
                    # argmax와 같은 operand 쌍의 signed difference는 operand 순서가
                    # 바뀌면 음수가 된다. QueryPlan에는 absolute_difference가 별도
                    # 있으므로 런타임이 뜻을 바꾸지 않고 안전하게 partial로 닫는다.
                    same_pair_argmax = any(
                        dd.operator == "argmax"
                        and {ref.output_id for ref in dd.operands} == set(ops)
                        for dd in derivations)
                    if d.operator == "difference" and same_pair_argmax and r < 0:
                        lims.append(Limitation(
                            code="comparison_difference_requires_absolute",
                            detail=(f"{d.output_id}: argmax 비교의 비방향 차이는 "
                                    "absolute_difference로 계획되어야 함")))
                        trace.append(TraceEvent(
                            seq=len(trace)+1, stage="derivation",
                            summary=f"difference {d.output_id} 건너뜀 — 비교 차이 부호 불일치",
                            detail={"operands": ops, "result": str(r)}))
                        continue
                    # 이슈 #118 — 공유하는 회사명·기간·기준을 한 번만 쓴다.
                    # 회사가 다르면(공유 접두 없음) 종전대로 두 라벨을 그대로
                    # 잇는다.
                    folded = fold_shared_label_prefix(a.claim.label, b.claim.label)
                    if folded is not None:
                        prefix, remainder_a, remainder_b = folded
                        label = (
                            f"{prefix} {remainder_a}"
                            f"{josa(remainder_a, '과', '와')} {remainder_b}의 차이")
                    else:
                        label = (
                            f"{a.claim.label}{josa(a.claim.label, '과', '와')} "
                            f"{b.claim.label}의 차이")
                    # 이슈 #62 — 피연산자가 이미 %(비율)면 그 뺄셈은 퍼센트포인트다.
                    # 「20.75% - 20.74%」의 답은 「-0.01%」가 아니라 「-0.01%p」 —
                    # %로 두면 상대 증감률(아래 percent_change)과 자릿수·뜻이
                    # 겹쳐 보여 혼동을 부른다.
                    out_unit = "%p" if unit == "%" else unit
                elif d.operator == "discrete_from_cumulative":
                    # 누적 - 직전 누적 = 해당 분기 단독. 라벨은 기간 종료월로 분기 표기
                    r = x - y
                    lab = _discrete_period_label(a, b, fact_meta)
                    if lab == a.claim.label:
                        for k, v in (("상반기 누적", "2분기 단독"), ("3분기 누적", "3분기 단독"), ("4분기 누적", "4분기 단독"), (" 누적", " 단독")):
                            if k in lab:
                                lab = lab.replace(k, v); break
                    label = lab; out_unit = unit
                    # 이슈 #64 — 이 결과 자체의 좌표(기간유형=quarter, 실제
                    # 분기 구간)를 등록해, 같은 실행 안에서 이 값을 다시
                    # operand로 쓰는 difference/percent_change가 연간 값과의
                    # 기간 길이 차이를 알아볼 수 있게 한다(위 가드 참고).
                    discrete_meta = _discrete_output_fact_meta(a, b, fact_meta)
                    if discrete_meta is not None:
                        fact_meta[d.output_id] = discrete_meta
                elif d.operator == "percent_change":
                    if y == 0:
                        raise _Div("base=0")
                    # 기저가 음수(적자)면 분모를 |y|로 둔다. 부호를 그대로 두면 -50→-100 이 「+100% 증가」가
                    # 되어 증감액(.delta = -50)과 방향이 어긋난다. |y| 기준이면 증감률·증감액 부호가 일치한다.
                    r = ((x - y) / abs(y) * Decimal(100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    # 이슈 #62 — 피연산자가 이미 %(비율)면 이 값은 「비율의
                    # 비율」, 즉 상대 증감률이다. 위 difference의 %p(절대
                    # 변화)와 나란히 나올 때 헷갈리지 않도록 라벨에서부터
                    # 「상대」를 밝힌다. 금액·건수 등 다른 단위의 전기 대비
                    # 증감률은 기존 라벨을 그대로 쓴다.
                    label = (f"{a.claim.label} 전기 대비 상대 증감률" if unit == "%"
                             else f"{a.claim.label} 전기 대비 증감률")
                    out_unit = "%"
                    direction = "증가" if r > 0 else ("감소" if r < 0 else "변동 없음")
                elif d.operator == "concept_ratio":
                    # 지표 간 나눗셈 — a 가 분자, b 가 분모(compiler 가 operand
                    # 순서를 이미 그렇게 낸다). 「률·비율」은 percent(×100 %),
                    # 「몇 배」는 multiple(그대로 배). `_period_over_period_order`·
                    # `_chronological_pair` 는 difference·percent_change 전용이라
                    # 여기서는 계획이 낸 operand 순서를 그대로 분자/분모로 쓴다.
                    if y == 0:
                        raise _Div("base=0")
                    if d.presentation == "percent":
                        r = (x / y * Decimal(100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                        out_unit = "%"
                        ratio_word = "비율"
                    else:
                        r = (x / y).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                        out_unit = "배"
                        ratio_word = "배수"
                    # 이슈 #118 — difference와 같은 원칙으로 공유 접두를 접는다.
                    folded = fold_shared_label_prefix(a.claim.label, b.claim.label)
                    if folded is not None:
                        prefix, remainder_a, remainder_b = folded
                        label = f"{prefix} {remainder_a}의 {remainder_b} 대비 {ratio_word}"
                    else:
                        label = f"{a.claim.label}의 {b.claim.label} 대비 {ratio_word}"
                else:
                    lims.append(Limitation(code="unsupported_operator", detail=f"{d.operator} 미구현 (v0.4 보류)"))
                    continue

                # 현금유출은 위에서 magnitude로 정규화했다. ``difference``의
                # 부호는 여전히 ordered operands의 증감 방향이고,
                # ``absolute_difference``만 그 방향을 지운 거리다.
                won_out = None
                if out_unit == "원":
                    won_out = r
                elif out_unit and out_unit not in {"%", "%p", "배"}:
                    from app.tools._units import to_won
                    won_out = to_won(r, out_unit)
                claim = AnswerClaim(output_id=d.output_id, label=label,
                                    value_text=_fmt(r), raw_unit=out_unit,
                                    text=(direction if d.operator == "percent_change" else None),
                                    canonical_value=str(won_out) if won_out is not None else (str(r) if out_unit in {"%", "%p", "배"} else None),
                                    canonical_unit=("원" if won_out is not None else (out_unit if out_unit in {"%", "%p", "배"} else None)),
                                    derived_from=ops, operator=d.operator, citations=cites)
                claims.append(claim)
                pool[d.output_id] = VerifiedScalar(d.output_id, r, out_unit, won_out, claim)
                trace.append(TraceEvent(seq=len(trace)+1, stage="derivation",
                                        summary=f"{d.operator} {ops} → {_fmt(r)} {out_unit}",
                                        detail={"output_id": d.output_id}))
                # 보조값(증감액·증감률·분자·분모)은 여기(Python)에서만 계산한다 —
                # LLM 이 스스로 빼기·나누기를 하다 환각내는 것을 원천 차단한다.
                for sc in self._supplements(d, derivations, a, b, x, y, unit, cites, fact_meta):
                    claims.append(sc)
                    trace.append(TraceEvent(seq=len(trace)+1, stage="derivation",
                                            summary=f"보조값 {sc.operator} → {sc.value_text}{sc.raw_unit or ''}",
                                            detail={"output_id": sc.output_id}))
            except _Unit as e:
                lims.append(Limitation(code="unit_mismatch", detail=str(e)))
            except _Div as e:
                lims.append(Limitation(code="divide_by_zero", detail=str(e)))
            except _Arity as e:
                lims.append(Limitation(code="derivation_arity", detail=str(e)))
            except _PeriodMismatch as e:
                lims.append(Limitation(code="period_length_mismatch", detail=str(e)))
            except InvalidOperation as e:
                lims.append(Limitation(code="derivation_invalid", detail=str(e)))
        return claims, lims


    def _supplements(self, d, derivations, a, b, x, y, unit, cites, fact_meta):
        """승인 연산의 파생 보조값 — 답변이 함께 요구하는 증감액·증감률·분자분모 슬롯.
        - percent_change → 증감액 병기 (증감률 질문의 정답은 비율+증감액+원값)
        - 같은 회사·같은 concept의 기간 비교 difference → 증감률 병기
          회사 간 비교는 corp_code가 달라 제외된다.
        - concept_ratio → 분자·분모 두 값을 그대로 병기 (자료에 있는 값을 없다고
          말하지 않도록, 비율만이 아니라 그 값을 만든 두 원값도 함께 보인다).
        plan이 같은 operand로 해당 연산을 이미 명시했으면 중복 생성하지 않는다."""
        ops = [r.output_id for r in d.operands]

        def plan_has(op: str) -> bool:
            return any(dd.operator == op and sorted(r.output_id for r in dd.operands) == sorted(ops)
                       for dd in derivations)

        out: list[AnswerClaim] = []
        if d.operator == "argmax" and not (plan_has("difference") or plan_has("absolute_difference")):
            # argmax 보조값은 순서 있는 증감이 아니라 두 비교값 사이의 거리다.
            gap = abs(x - y)
            won = None
            if unit == "원":
                won = gap
            elif unit and unit != "%":
                from app.tools._units import to_won
                won = to_won(gap, unit)
            out.append(AnswerClaim(
                output_id=f"{d.output_id}.gap", label="두 값의 차이",
                value_text=_fmt(gap), raw_unit=unit,
                canonical_value=str(won) if won is not None else None,
                canonical_unit="원" if won is not None else None,
                derived_from=list(ops), operator="absolute_difference", citations=list(cites)))
        if d.operator == "percent_change" and not plan_has("difference"):
            delta = x - y
            # 이슈 #62 — 비율끼리의 증감액은 %p다(difference와 같은 규칙).
            delta_unit = "%p" if unit == "%" else unit
            won = None
            if delta_unit == "원":
                won = delta
            elif delta_unit and delta_unit not in {"%", "%p"}:
                from app.tools._units import to_won
                won = to_won(delta, delta_unit)
            out.append(AnswerClaim(
                output_id=f"{d.output_id}.delta", label=f"{a.claim.label} 전기 대비 증감액",
                value_text=_fmt(delta), raw_unit=delta_unit,
                canonical_value=str(won) if won is not None else None,
                canonical_unit="원" if won is not None else None,
                derived_from=list(ops), operator="difference", citations=list(cites)))
        if d.operator == "difference" and not plan_has("percent_change") and fact_meta and y > 0:
            ma, mb = fact_meta.get(ops[0]), fact_meta.get(ops[1])
            same_series = (ma and mb and ma.get("corp_code") and ma.get("corp_code") == mb.get("corp_code")
                           and ma.get("concept") == mb.get("concept")
                           and ma.get("period_end") != mb.get("period_end"))
            if same_series:
                rr = ((x - y) / y * Decimal(100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                # 이슈 #62 — 원 피연산자가 비율이면 이 보조값도 상대 증감률이다.
                rate_label = (f"{a.claim.label} 전기 대비 상대 증감률" if unit == "%"
                              else f"{a.claim.label} 전기 대비 증감률")
                out.append(AnswerClaim(
                    output_id=f"{d.output_id}.rate", label=rate_label,
                    value_text=_fmt(rr), raw_unit="%",
                    text="증가" if rr > 0 else ("감소" if rr < 0 else "변동 없음"),
                    canonical_value=str(rr), canonical_unit="%",
                    derived_from=list(ops), operator="percent_change", citations=list(cites)))
        if d.operator == "concept_ratio":
            out.append(AnswerClaim(
                output_id=f"{d.output_id}.numerator", label=a.claim.label,
                value_text=a.claim.value_text, raw_unit=a.claim.raw_unit,
                canonical_value=a.claim.canonical_value,
                canonical_unit=a.claim.canonical_unit,
                derived_from=[ops[0]], operator="concept_ratio_operand",
                citations=list(a.claim.citations)))
            out.append(AnswerClaim(
                output_id=f"{d.output_id}.denominator", label=b.claim.label,
                value_text=b.claim.value_text, raw_unit=b.claim.raw_unit,
                canonical_value=b.claim.canonical_value,
                canonical_unit=b.claim.canonical_unit,
                derived_from=[ops[1]], operator="concept_ratio_operand",
                citations=list(b.claim.citations)))
        return out


class _Unit(Exception): ...
class _Div(Exception): ...
class _Arity(Exception): ...
class _PeriodMismatch(Exception): ...
